# ═══ 多策略 RAG 文件問答（桌面版）· CustomTkinter + Groq + ChromaDB + LINE Bot 投放 ═══
# 執行：python rag_desktop.py
# LINE：在側欄貼上 Channel Access Token，問答結果（500 字內）可 Broadcast 給所有好友
# 需要：Python 3.9+（Linux 請另裝 tkinter：sudo apt install python3-tk）
from __future__ import annotations

import csv
import os
import queue
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import chromadb
import customtkinter as ctk
import numpy as np
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from groq import Groq
from linebot.v3.messaging import ApiClient, BroadcastRequest, Configuration, MessagingApi, TextMessage
from linebot.v3.messaging.exceptions import ApiException
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定 ──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")   # 可換 multilingual-e5-base、BAAI/bge-m3
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
ENV_KEY = os.getenv("GROQ_API_KEY", "").strip()
ENV_LINE = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_LIMIT, LINE_HOST = 500, "https://api.line.me"   # LINE 訊息字數上限（可自行調整）
FONT = {"win32": "Microsoft JhengHei UI", "darwin": "PingFang TC"}.get(sys.platform, "Noto Sans CJK TC")
ACCENT, ACCENT_HOVER = "#10b981", "#059669"

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p
        fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


@dataclass
class Doc:
    sid: str
    name: str
    chunks: list
    labels: list
    n_child: int = 0
    tfidf: object = field(default=None, repr=False)
    tm: object = field(default=None, repr=False)


class LLM:
    def __init__(self, key, model):
        key = (key or "").strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model or MODELS[0]

    def __call__(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client:
            return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                kw.pop("reasoning_effort", None)
                kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err:
                raise
            return None


def list_models(key):
    try:
        ids = [m.id for m in Groq(api_key=key.strip()).models.list().data if getattr(m, "active", True) is not False]
    except Exception:
        return []
    return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))


class RAG:
    """嵌入模型 + 向量庫 + 目前載入的單一文件。"""

    def __init__(self):
        self._embedder, self._lock = None, threading.Lock()
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = chromadb.EphemeralClient()  # 記憶體模式，關閉程式即清空
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40].strip("_")
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
        self.doc: Doc | None = None

    @property
    def ready(self):
        return self._embedder is not None

    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                    m.max_seq_length = min(512, m.max_seq_length or 512)
                    self._embedder = m
        return self._embedder

    # ── 文件載入 ──
    def load(self, path):
        t0, path = time.time(), Path(path)
        if path.suffix.lower() not in (".pdf", ".docx"):
            raise ValueError("僅支援 PDF / DOCX")
        segs = self._pdf(path) if path.suffix.lower() == ".pdf" else self._docx(path)
        pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
        if not pairs:
            raise ValueError("無可擷取文字（可能是掃描檔，需先 OCR）")
        doc = Doc(uuid.uuid4().hex, path.name, [p[0] for p in pairs], [p[1] for p in pairs])
        try:
            self._add(self.col, [f"{doc.sid}_{i}" for i in range(len(pairs))], doc.chunks,
                      [{"sid": doc.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{doc.sid}_{p}_{c}", t, p) for p, par in enumerate(doc.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            doc.n_child = len(kids)
            self._add(self.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": doc.sid, "idx": k[2]} for k in kids])
            doc.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            doc.tm = doc.tfidf.fit_transform(doc.chunks)
        except Exception:
            self.drop(doc)
            raise
        self.drop(self.doc)  # 新文件成功後才釋放舊文件
        self.doc = doc
        return f"✓ {doc.name}\n{len(doc.chunks)} 片段 · {time.time() - t0:.1f}s"

    def drop(self, doc):
        if doc is None:
            return
        for c in (self.col, self.ccol):
            try:
                c.delete(where={"sid": doc.sid})
            except Exception:
                pass

    def _pdf(self, path):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, path):
        doc, segs, head, buf, nt = Document(str(path)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush()
                    head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1
                rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    flush()
                    segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return self.embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=self._enc(docs[i:i + b]))

    def vec(self, doc, qs, k, child=False):
        total = doc.n_child if child else len(doc.chunks)
        if not total:
            return [[] for _ in qs]
        res = (self.ccol if child else self.col).query(query_embeddings=self._enc(qs, True), n_results=max(1, min(k, total)),
                                                       where={"sid": doc.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    @staticmethod
    def kw(doc, q, k):
        s = (doc.tm @ doc.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)):
                sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, doc, q, strategy, k, llm):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec(doc, [q], n * 2)[0], self.kw(doc, q, n * 2)], n)
        if strategy == "semantic":
            idx = self.vec(doc, [q], k)[0]
        elif strategy == "tfidf":
            idx = self.kw(doc, q, k)
        elif strategy == "parent_child":
            idx = list(dict.fromkeys(self.vec(doc, [q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec(doc, [q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec(doc, [q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12))
            idx = cands[:k]
            if llm.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {doc.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                        15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else:  # hybrid、compress
            idx = hybrid(k)
        hits = [(i, doc.chunks[i]) for i in idx]
        if strategy == "compress" and llm.client and hits:
            def comp(h):
                r = llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex:
                hits = list(ex.map(comp, hits))
        return hits

    def answer(self, q, strategy, k, model, key):
        """回傳 (回答, 參考片段, 耗時說明)。"""
        doc = self.doc
        if doc is None:
            return "請先載入 PDF 或 DOCX 文件。", "", ""
        llm = LLM(key, model)
        t0 = time.time()
        hits = self.retrieve(doc, q, strategy, int(k), llm)
        t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{doc.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX:
                break
            blocks.append(b)
            used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {doc.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits:
            return "找不到相關片段，請換個問法或策略。", src, ""
        if not llm.client:
            return "⚠ 尚未設定 Groq API Key（檢索結果請看「檢索片段」分頁）", src, ""
        ans = llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                  f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                  1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        return ans or "（模型未回傳內容）", src, f"{llm.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s"


# ── 匯出：Word / CSV ──
def export_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 開啟中文不亂碼
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def export_docx(records, path, with_src=True):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Microsoft JhengHei", Pt(11)
    st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line):
                doc.add_heading(m.group(1).replace("**", ""), level=2)
                continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
                _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number")
                _md_runs(par, m.group(1))
            else:
                _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph()
            run = h.add_run("參考片段")
            run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip())
                p.paragraph_format.left_indent = Pt(12)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    doc.save(path)


def plain(md: str) -> str:
    """Tk 文字框不渲染 Markdown：移除粗體標記，標題轉成一般行。"""
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    return re.sub(r"^\s*#{1,6}\s+", "", md, flags=re.M)


# ── LINE Bot 投放 ──
def line_text(q, ans, limit=LINE_LIMIT):
    """組成 LINE 訊息：問題 + 回答（去除 Markdown 與 [片段n] 標註），總長不超過 limit 字。"""
    body = re.sub(r"[ \t]*\[片段[^\]]*\]", "", plain(ans))
    body = re.sub(r"[ \t]+(?=\n|$)", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    q = " ".join(q.split())
    msg = f"❓ {q[:59] + '…' if len(q) > 60 else q}\n\n{body}"
    if len(msg) > limit:
        cut = msg[:limit - 1]
        end = max(cut.rfind(c) for c in "\n。！？；")  # 盡量在句尾截斷
        msg = (cut[:end + 1] if end > limit * 0.6 else cut).rstrip() + "…"
    return msg


def line_send(token, text, host=LINE_HOST):
    """Broadcast 給所有加入此官方帳號的好友（會消耗每月訊息額度）。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        MessagingApi(c).broadcast(BroadcastRequest(messages=[TextMessage(text=text)]))


def line_check(token, host=LINE_HOST):
    """驗證 Token，回傳官方帳號名稱。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        return MessagingApi(c).get_bot_info().display_name


def line_err(e):
    if isinstance(e, ApiException):
        hint = {400: "請求格式錯誤", 401: "Token 無效或已過期", 403: "權限不足", 429: "超過訊息額度或請求過於頻繁"}.get(e.status, "")
        return f"HTTP {e.status} {hint}".strip()
    return f"{type(e).__name__}: {e}"


# ═══════════════ 介面 ═══════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("green")
        self.title("📚 RAG 文件問答")
        self.geometry("1180x760")
        self.minsize(860, 560)
        self.rag = RAG()
        self.records: list[dict] = []
        self.busy = False
        self.last_qa: tuple | None = None   # 最後一則成功回答 (問題, 回答)，供 LINE 投放
        self._q: queue.Queue = queue.Queue()
        self._strategy_by_label = {v[0]: k for k, v in STRATEGIES.items()}
        self.f = lambda size=13, weight="normal": ctk.CTkFont(family=FONT, size=size, weight=weight)

        # 視窗縮放：右側主區隨視窗伸縮；視窗過窄時側欄自動收成可捲動
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(100, self._poll)
        self._init_key()
        self._init_line()
        self.footer.configure(text="⏳ 正在載入嵌入模型（首次需下載，約 470 MB）…")
        self._bg(self._warmup, self._on_warm)  # 背景預載嵌入模型

    # ── 背景工作：在執行緒跑，結果透過 queue 回到 UI 執行緒 ──
    def _bg(self, fn, done):
        def run():
            try:
                res, err = fn(), None
            except Exception as e:  # noqa: BLE001
                res, err = None, e
            self._q.put((done, res, err))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        try:
            while True:
                done, res, err = self._q.get_nowait()
                done(res, err)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ── 側欄 ──
    def _build_sidebar(self):
        side = ctk.CTkScrollableFrame(self, width=290, corner_radius=14, label_text="")
        side.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        side.grid_columnconfigure(0, weight=1)
        self.side = side
        f = self.f

        ctk.CTkLabel(side, text="📚 RAG 文件問答", font=f(20, "bold"), anchor="w").grid(row=0, column=0, sticky="ew", pady=(4, 0))
        ctk.CTkLabel(side, text="Groq × ChromaDB", font=f(12), text_color="gray", anchor="w").grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(side, text="🔑 Groq API Key", font=f(13, "bold"), anchor="w").grid(row=2, column=0, sticky="ew")
        self.key_entry = ctk.CTkEntry(side, show="•", placeholder_text="gsk_…（Enter 套用）", font=f())
        self.key_entry.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        self.key_entry.bind("<Return>", lambda _e: self.apply_key())
        self.key_status = ctk.CTkLabel(side, text="", font=f(12), anchor="w", justify="left", wraplength=250)
        self.key_status.grid(row=4, column=0, sticky="ew", pady=(2, 8))

        ctk.CTkLabel(side, text="模型", font=f(13, "bold"), anchor="w").grid(row=5, column=0, sticky="ew")
        self.model_box = ctk.CTkComboBox(side, values=MODELS, font=f())
        self.model_box.set(MODELS[0])
        self.model_box.grid(row=6, column=0, sticky="ew", pady=(2, 10))

        ctk.CTkLabel(side, text="文件", font=f(13, "bold"), anchor="w").grid(row=7, column=0, sticky="ew")
        self.upload_btn = ctk.CTkButton(side, text="📂 選擇 PDF / DOCX", font=f(), command=self.pick_file,
                                        fg_color=ACCENT, hover_color=ACCENT_HOVER, height=36)
        self.upload_btn.grid(row=8, column=0, sticky="ew", pady=(2, 4))
        self.status = ctk.CTkLabel(side, text="尚未載入文件", font=f(12), text_color="gray", anchor="w", justify="left", wraplength=250)
        self.status.grid(row=9, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(side, text="檢索策略", font=f(13, "bold"), anchor="w").grid(row=10, column=0, sticky="ew")
        self.strategy_box = ctk.CTkOptionMenu(side, values=[v[0] for v in STRATEGIES.values()], font=f(),
                                              command=self._on_strategy, fg_color=ACCENT, button_color=ACCENT_HOVER,
                                              button_hover_color=ACCENT_HOVER)
        self.strategy_box.set(STRATEGIES["hybrid"][0])
        self.strategy_box.grid(row=11, column=0, sticky="ew", pady=(2, 0))
        self.desc = ctk.CTkLabel(side, text=STRATEGIES["hybrid"][1], font=f(11), text_color="gray", anchor="w",
                                 justify="left", wraplength=250)
        self.desc.grid(row=12, column=0, sticky="ew", pady=(2, 10))

        head = ctk.CTkFrame(side, fg_color="transparent")
        head.grid(row=13, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="片段數量 Top-K", font=f(13, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        self.k_label = ctk.CTkLabel(head, text="4", font=f(13, "bold"), text_color=ACCENT)
        self.k_label.grid(row=0, column=1)
        self.k_slider = ctk.CTkSlider(side, from_=1, to=10, number_of_steps=9, button_color=ACCENT,
                                      button_hover_color=ACCENT_HOVER, progress_color=ACCENT,
                                      command=lambda v: self.k_label.configure(text=str(int(v))))
        self.k_slider.set(4)
        self.k_slider.grid(row=14, column=0, sticky="ew", pady=(4, 12))

        ctk.CTkLabel(side, text="匯出問答紀錄", font=f(13, "bold"), anchor="w").grid(row=15, column=0, sticky="ew")
        self.with_src = ctk.CTkCheckBox(side, text="Word 包含參考片段", font=f(12))
        self.with_src.select()
        self.with_src.grid(row=16, column=0, sticky="w", pady=(2, 4))
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.grid(row=17, column=0, sticky="ew")
        row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(row, text="📄 Word", font=f(12), height=32, command=self.save_docx).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ctk.CTkButton(row, text="📊 CSV", font=f(12), height=32, command=self.save_csv).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        ctk.CTkLabel(side, text="💬 LINE Bot 投放", font=f(13, "bold"), anchor="w").grid(row=18, column=0, sticky="ew", pady=(14, 0))
        self.line_entry = ctk.CTkEntry(side, show="•", placeholder_text="Channel Access Token（Enter 驗證）", font=f())
        self.line_entry.grid(row=19, column=0, sticky="ew", pady=(2, 0))
        self.line_entry.bind("<Return>", lambda _e: self.apply_line())
        self.line_status = ctk.CTkLabel(side, text="", font=f(12), anchor="w", justify="left", wraplength=250)
        self.line_status.grid(row=20, column=0, sticky="ew", pady=(2, 4))
        self.auto_line = ctk.CTkCheckBox(side, text=f"回答後自動投放（≤{LINE_LIMIT} 字）", font=f(12), command=self._on_auto_line)
        self.auto_line.grid(row=21, column=0, sticky="w", pady=(0, 4))
        self.line_btn = ctk.CTkButton(side, text="📤 投放最後一則回答", font=f(12), height=32, command=self.send_line_manual)
        self.line_btn.grid(row=22, column=0, sticky="ew", pady=(0, 4))

        ctk.CTkLabel(side, text="外觀", font=f(13, "bold"), anchor="w").grid(row=23, column=0, sticky="ew", pady=(14, 0))
        self.mode = ctk.CTkSegmentedButton(side, values=["系統", "淺色", "深色"], font=f(12),
                                           command=lambda v: ctk.set_appearance_mode({"系統": "system", "淺色": "light", "深色": "dark"}[v]))
        self.mode.set("系統")
        self.mode.grid(row=24, column=0, sticky="ew", pady=(2, 4))

    # ── 主區 ──
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=14)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)
        f = self.f

        self.tabs = ctk.CTkTabview(main, corner_radius=14)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        t_chat, t_src = self.tabs.add("💬 對話"), self.tabs.add("🔎 檢索片段")
        for t in (t_chat, t_src):
            t.grid_columnconfigure(0, weight=1)
            t.grid_rowconfigure(0, weight=1)
        self.chat = ctk.CTkTextbox(t_chat, wrap="word", font=f(14), corner_radius=10)
        self.chat.grid(row=0, column=0, sticky="nsew")
        self.chat.tag_config("user", foreground=ACCENT)
        self.chat.tag_config("meta", foreground="#9ca3af")
        self.src_box = ctk.CTkTextbox(t_src, wrap="word", font=f(13), corner_radius=10)
        self.src_box.grid(row=0, column=0, sticky="nsew")
        self._say("meta", "上傳文件後開始提問。例如：這份文件的重點是什麼？\n")
        self.src_box.configure(state="disabled")

        bar = ctk.CTkFrame(main, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(bar, placeholder_text="輸入問題，Enter 送出", font=f(14), height=40)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self.ask())
        self.send_btn = ctk.CTkButton(bar, text="送出", font=f(14, "bold"), width=90, height=40,
                                      fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.ask)
        self.send_btn.grid(row=0, column=1)
        ctk.CTkButton(bar, text="🗑", width=40, height=40, font=f(14), fg_color="gray50", hover_color="gray40",
                      command=self.clear).grid(row=0, column=2, padx=(8, 0))

        ex = ctk.CTkFrame(main, fg_color="transparent")
        ex.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for i, text in enumerate(("這份文件的重點是什麼？", "文件提到哪些規定或限制？", "有哪些風險或注意事項？")):
            ctk.CTkButton(ex, text=text, font=f(12), height=28, fg_color="transparent", border_width=1,
                          text_color=("gray20", "gray80"), hover_color=("gray85", "gray25"),
                          command=lambda t=text: self._fill(t)).grid(row=0, column=i, padx=(0, 6))

        self.footer = ctk.CTkLabel(main, text="", font=f(11), text_color="gray", anchor="w")
        self.footer.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    # ── 小工具 ──
    def _say(self, tag, text):
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _fill(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.focus()

    def _on_strategy(self, label):
        self.desc.configure(text=STRATEGIES[self._strategy_by_label[label]][1])

    def _set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for w in (self.send_btn, self.upload_btn):
            w.configure(state=state)

    def key(self):
        return self.key_entry.get().strip() or ENV_KEY

    # ── API Key ──
    def _init_key(self):
        if ENV_KEY:
            self.key_status.configure(text="已從環境變數 GROQ_API_KEY 載入，驗證中…")
            self._bg(lambda: list_models(ENV_KEY), self._on_models)
        else:
            self.key_status.configure(text="⚠ 尚未設定 Key", text_color="#f59e0b")

    def apply_key(self):
        k = self.key()
        if not k:
            self.key_status.configure(text="⚠ 尚未設定 Key", text_color="#f59e0b")
            return
        self.key_status.configure(text="驗證中…", text_color="gray")
        self._bg(lambda: list_models(k), self._on_models)

    def _on_models(self, ms, _err):
        if not ms:
            self.key_status.configure(text="✗ Key 無效或無法取得模型", text_color="#ef4444")
            return
        cur = self.model_box.get()
        self.model_box.configure(values=ms)
        self.model_box.set(cur if cur in ms else (MODELS[0] if MODELS[0] in ms else ms[0]))
        self.key_status.configure(text=f"✓ Key 有效 · {len(ms)} 個模型", text_color=ACCENT)

    # ── LINE Bot ──
    def line_token(self):
        return self.line_entry.get().strip() or ENV_LINE

    def _init_line(self):
        if ENV_LINE:
            self.line_status.configure(text="已從環境變數 LINE_CHANNEL_ACCESS_TOKEN 載入，驗證中…", text_color="gray")
            self._bg(lambda: line_check(ENV_LINE), self._on_line_checked)
        else:
            self.line_status.configure(text="⚠ 尚未設定 LINE Token", text_color="#f59e0b")

    def apply_line(self):
        tok = self.line_token()
        if not tok:
            self.line_status.configure(text="⚠ 尚未設定 LINE Token", text_color="#f59e0b")
            return
        self.line_status.configure(text="驗證中…", text_color="gray")
        self._bg(lambda: line_check(tok), self._on_line_checked)

    def _on_line_checked(self, name, err):
        if err:
            self.line_status.configure(text=f"✗ Token 驗證失敗：{line_err(err)}", text_color="#ef4444")
        else:
            self.line_status.configure(text=f"✓ Token 有效 · {name or 'LINE 官方帳號'}", text_color=ACCENT)

    def _on_auto_line(self):
        if self.auto_line.get() and not messagebox.askyesno(
                "自動投放到 LINE", "啟用後，每次回答都會以 Broadcast 群發給所有加入此官方帳號的好友，\n並消耗每月訊息額度。確定啟用？"):
            self.auto_line.deselect()

    def send_line_manual(self):
        if not self.last_qa:
            messagebox.showinfo("尚無回答", "請先提問並取得回答後再投放。")
            return
        if not self.line_token():
            messagebox.showinfo("尚未設定 Token", "請先在側欄貼上 LINE Channel Access Token。")
            return
        text = line_text(*self.last_qa)
        if messagebox.askyesno("確認投放到 LINE", f"將群發給所有好友（{len(text)} 字）：\n\n{text}"):
            self._send_line(text)

    def _send_line(self, text):
        tok = self.line_token()
        if not tok:
            self.line_status.configure(text="⚠ 尚未設定 LINE Token，未投放", text_color="#f59e0b")
            return
        self.line_btn.configure(state="disabled")
        self.line_status.configure(text="⏳ 投放中…", text_color="gray")
        self._bg(lambda: line_send(tok, text), lambda _r, err: self._on_line_sent(len(text), err))

    def _on_line_sent(self, n, err):
        self.line_btn.configure(state="normal")
        if err:
            msg = f"✗ LINE 投放失敗：{line_err(err)}"
            self.line_status.configure(text=msg, text_color="#ef4444")
        else:
            msg = f"✓ 已投放到 LINE（{n} 字）"
            self.line_status.configure(text=f"{msg} · {datetime.now():%H:%M:%S}", text_color=ACCENT)
        self._say("meta", f"{msg}\n")

    # ── 嵌入模型 ──
    def _warmup(self):
        self.rag.embedder()

    def _on_warm(self, _res, err):
        if err:
            self.footer.configure(text=f"✗ 嵌入模型載入失敗：{type(err).__name__}: {err}", text_color="#ef4444")
        else:
            self.footer.configure(text="嵌入模型已就緒", text_color="gray")

    # ── 載入文件 ──
    def pick_file(self):
        path = filedialog.askopenfilename(title="選擇文件", filetypes=[("PDF / DOCX", "*.pdf *.docx"), ("所有檔案", "*.*")])
        if not path:
            return
        self._set_busy(True)
        self.status.configure(text="⏳ 解析並建立索引中…", text_color="gray")
        self._bg(lambda: self.rag.load(path), self._on_loaded)

    def _on_loaded(self, msg, err):
        self._set_busy(False)
        if err:
            self.status.configure(text=f"✗ {err}" if isinstance(err, ValueError) else f"✗ 載入失敗：{type(err).__name__}: {err}",
                                  text_color="#ef4444")
        else:
            self.status.configure(text=msg, text_color=ACCENT)
            self.footer.configure(text="", text_color="gray")

    # ── 提問 ──
    def ask(self):
        q = self.entry.get().strip()
        if not q or self.busy:
            return
        if self.rag.doc is None:
            messagebox.showinfo("尚未載入文件", "請先選擇 PDF 或 DOCX 文件。")
            return
        strategy = self._strategy_by_label[self.strategy_box.get()]
        k, model, key = int(self.k_slider.get()), self.model_box.get().strip(), self.key()
        self.entry.delete(0, "end")
        self._say("user", f"🙋 {q}\n")
        self._think_start = self.chat.index("end-1c")
        self._say("meta", "思考中…\n")
        self._set_busy(True)

        def work():
            ans, src, timing = self.rag.answer(q, strategy, k, model, key)
            return q, strategy, model, ans, src, timing
        self._bg(work, self._on_answer)

    def _on_answer(self, res, err):
        self._set_busy(False)
        self.chat.configure(state="normal")
        self.chat.delete(self._think_start, "end-1c")  # 移除「思考中…」
        self.chat.configure(state="disabled")
        if err:
            self._say("meta", f"✗ {type(err).__name__}: {err}\n\n")
            return
        q, strategy, model, ans, src, timing = res
        self._say(None, f"🤖 {plain(ans)}\n")
        if timing:
            self._say("meta", f"{timing}\n")
        self._say(None, "\n")
        self.src_box.configure(state="normal")
        self.src_box.delete("1.0", "end")
        self.src_box.insert("1.0", src)
        self.src_box.configure(state="disabled")
        self.records.append({"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文件": self.rag.doc.name, "問題": q,
                             "回答": ans, "策略": STRATEGIES[strategy][0], "模型": model or MODELS[0],
                             "參考片段": src.split("─" * 40, 1)[-1].strip()})
        if timing:  # timing 非空 = 模型確實產生了回答（排除未載入文件／無 Key／無結果等提示）
            self.last_qa = (q, ans)
            if self.auto_line.get():
                self._send_line(line_text(q, ans))

    def clear(self):
        self.records.clear()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.src_box.configure(state="normal")
        self.src_box.delete("1.0", "end")
        self.src_box.configure(state="disabled")
        self._say("meta", "對話已清除。\n")

    # ── 匯出 ──
    def _save(self, ext, label, writer):
        if not self.records:
            messagebox.showinfo("無紀錄", "目前沒有問答紀錄可匯出。")
            return
        path = filedialog.asksaveasfilename(defaultextension=f".{ext}", filetypes=[(label, f"*.{ext}")],
                                            initialfile=f"RAG問答紀錄_{datetime.now():%Y%m%d_%H%M%S}.{ext}")
        if not path:
            return
        try:
            writer(path)
            messagebox.showinfo("匯出完成", f"已儲存：\n{path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("匯出失敗", f"{type(e).__name__}: {e}")

    def save_docx(self):
        self._save("docx", "Word 文件", lambda p: export_docx(self.records, p, bool(self.with_src.get())))

    def save_csv(self):
        self._save("csv", "CSV", lambda p: export_csv(self.records, p))


if __name__ == "__main__":
    App().mainloop()

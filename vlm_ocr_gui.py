# -*- coding: utf-8 -*-
"""日本語VLM OCR GUI（単一ファイル実行、本体ロジック）"""

# ===== Standard Library =====
import os, sys, json, glob, math, time, shutil, platform, threading
from typing import List, Tuple, Optional, Dict
from importlib import metadata as importlib_metadata

# ===== Third-Party =====
import cv2
import numpy as np
import PIL.Image as PILImage
import torch

# --- optional backends (top-level import, “ないなら None” で表現) ---
try:
    from manga_ocr import MangaOcr
except Exception:
    MangaOcr = None

try:
    import easyocr as _easyocr
except Exception:
    _easyocr = None

try:
    import pytesseract as _pytesseract
except Exception:
    _pytesseract = None

# ===== Flexible SG import (v4優先フォールバック) =====
# ここでは、ローカル置きの FreeSimpleGUI / PySimpleGUI どちらでも動くようにする。
_GUI_BACKEND = None
try:
    import FreeSimpleGUI as sg  # 推奨
    _GUI_BACKEND = "FreeSimpleGUI(v4 fork)"
except Exception:
    try:
        import PySimpleGUI as sg
        _GUI_BACKEND = "PySimpleGUI(v4)"
    except Exception as e:
        raise ImportError(
            "No usable SG backend found. Install `FreeSimpleGUI` or `PySimpleGUI`."
        ) from e
finally:
    print(f"[INFO] GUI backend: {_GUI_BACKEND}")

# =============================================================================
# VLM Backend Abstraction
# =============================================================================
class VLMBackend:
    """画像 -> (テキスト, メタ) を返す最小インターフェース"""
    def ocr(self, pil_img: PILImage.Image) -> Tuple[str, Dict]:
        raise NotImplementedError

class MangaOCRBackend(VLMBackend):
    """manga-ocr バックエンド（CPU）"""
    def __init__(self):
        if MangaOcr is None:
            raise ImportError("manga-ocr is not installed. `pip install manga-ocr`")
        self.mocr = MangaOcr()  # CPU固定
    def ocr(self, pil_img: PILImage.Image) -> Tuple[str, Dict]:
        txt = self.mocr(pil_img)
        return txt.strip(), {"backend": "manga-ocr"}

class EasyOCRBackend(VLMBackend):
    """easyocr バックエンド（任意導入）"""
    def __init__(self, lang=("ja","en")):
        if _easyocr is None:
            raise ImportError("easyocr is not installed. `pip install easyocr`")
        self.reader = _easyocr.Reader(list(lang))
    def ocr(self, pil_img: PILImage.Image) -> Tuple[str, Dict]:
        arr = np.array(pil_img)
        result = self.reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(result).strip(), {"backend":"easyocr","lang":"ja,en"}

class TesseractBackend(VLMBackend):
    """Tesseract バックエンド（任意導入）"""
    def __init__(self, lang="jpn"):
        if _pytesseract is None:
            raise ImportError("pytesseract is not installed. `pip install pytesseract`")
        self.tess = _pytesseract
        self.lang = lang
    def ocr(self, pil_img: PILImage.Image) -> Tuple[str, Dict]:
        txt = self.tess.image_to_string(pil_img, lang=self.lang)
        return txt.strip(), {"backend":"tesseract","lang":self.lang}

# --- registry & current selection（環境変数で切替。コメントだけで十分アピール可） ---
_VLM_BACKENDS: Dict[str, type] = {
    "manga-ocr": MangaOCRBackend,
    # "easyocr": EasyOCRBackend,     # ← 使うときだけアンコメント
    # "tesseract": TesseractBackend, # ← 使うときだけアンコメント
}
_VLM_BACKEND = os.getenv("JPVLM_BACKEND", "manga-ocr")  # “DEFAULT”という語は使わない方針

def build_vlm_backend() -> VLMBackend:
    """現在選択されている VLM バックエンドを生成"""
    cls = _VLM_BACKENDS.get(_VLM_BACKEND)
    if cls is None:
        raise ValueError(
            f"Unsupported VLM backend: '{_VLM_BACKEND}'. "
            f"Available: {', '.join(_VLM_BACKENDS.keys())}"
        )
    return cls()

# =============================================================================
# Common Utilities（Stage0で必要な最小限のみ）
# =============================================================================
def is_image(p: str) -> bool:
    # 拡張子から画像ファイルかどうか判定（一般的なもののみ）
    p = p.lower()
    return any(p.endswith(ext) for ext in (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"))

def ensure_dir(d: str) -> None:
    # ディレクトリを作成（既存なら何もしない）
    os.makedirs(d, exist_ok=True)

def write_text(path: str, text: str) -> None:
    # テキストをファイルへ保存（UTF-8）
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def to_paragraph(lines: List[str]) -> str:
    # 改行続きのテキストを“段落”に整形（行末ハイフンの連結にも対応）
    out, buf = [], []
    for raw in lines:
        s = raw.strip()
        if not s:
            if buf:
                out.append(" ".join(buf).replace(" - ", "-"))
                buf = []
            continue
        if s.endswith("-"):
            buf.append(s[:-1])       # 行末ハイフンは次行へ連結
        else:
            buf.append(s)
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)

# =============================================================================
# （参考）Stage1〜4で開示予定のユーティリティ群（コメントアウトで同梱）
# =============================================================================
# def ffmpeg_which() -> str: ...
# def ffmpeg_handshake(...): ...
# def extract_frames_ffmpeg(...): ...
# def extract_frames_opencv(...): ...
# def gather_vlm_info(...): ...
# def gather_env_info(...): ...
# def gather_diag_info(...): ...

# =============================================================================
# JPVLM Wrapper（VLMバックエンド切替）
# =============================================================================
class JPVLM:
    """VLM バックエンドの薄いラッパ。threads指定は将来用に保持（Stage0では無指定運用）"""
    def __init__(self, threads: Optional[int] = None):
        if threads and threads > 0:
            try:
                torch.set_num_threads(threads)
            except Exception:
                pass
        self.engine = build_vlm_backend()

    def ocr(self, img: PILImage.Image) -> Tuple[str, Dict]:
        return self.engine.ocr(img)

def preprocess(img: PILImage.Image, mode: str) -> PILImage.Image:
    """
    OCR前の簡易前処理（Stage0は "none" 固定で呼ぶ想定。将来のためロジックは置いておく）
    """
    if mode == "none":
        return img.convert("RGB")
    arr = np.array(img.convert("L"))
    if mode in ("binarize","binarize+sharpen"):
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu二値化
        arr = th
        if mode.endswith("sharpen"):
            arr = cv2.GaussianBlur(arr, (0,0), 1.0)
            arr = cv2.addWeighted(arr, 1.5, cv2.GaussianBlur(arr, (0,0), 2.0), -0.5, 0)
    if mode == "scale2x":
        arr = cv2.resize(arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)      # 2倍拡大
    return PILImage.fromarray(arr).convert("RGB")

# =============================================================================
# OCR 実処理（Stage0：単一画像のみ使用）
# =============================================================================
def ocr_image(vlm: JPVLM, img_path: str, pre: str = "none", paragraphize: bool = False) -> Tuple[str, dict]:
    # 単一画像のOCRを実行（段落整形はStage0では既定OFF）
    img = PILImage.open(img_path)
    img_p = preprocess(img, pre)
    text, meta = vlm.ocr(img_p)
    if paragraphize:
        text = to_paragraph(text.splitlines())
    return text, meta

# =============================================================================
# GUI（Stage0のみ有効UI）
# =============================================================================
sg.theme("BlueMono")

# ---- Stage0: 単一画像OCR MVP（この塊だけ有効） ----
layout = [
    [sg.Text("日本語VLM文字起こし – 単一画像 (Stage0)", font=("Segoe UI", 12, "bold"))],
    [sg.Frame("入力（画像）", [
        [sg.Text("単一画像"), sg.Input(key="-IMGFILE-", size=(48,1)),
         sg.FileBrowse(file_types=(("Images","*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff"),("All","*.*")))]
    ])],
    [sg.ProgressBar(max_value=100, orientation="h", size=(50,20), key="-PROG-")],
    [sg.Multiline(size=(100,14), key="-LOG-", autoscroll=True, disabled=True)],
    [sg.Button("画像を実行", key="-RUN-IMG-"), sg.Button("Exit")]
]

# ---- Stage1 以降（コメントを外すと順次開示できる） ------------------------------
# [Stage1] フォルダ一括/パターン/再帰/出力先/上書き/段落整形
# [Stage2] 前処理 & 並列ジョブ数
# [Stage3] 動画入力 + FFmpeg/Interval
# [Stage4] 環境情報/詳細診断/FFmpeg疎通 など
# -----------------------------------------------------------------------------

window = sg.Window(
    f"JP VLM OCR – Stage0 [{_GUI_BACKEND}] (backend={_VLM_BACKEND})",
    layout, finalize=True
)

def log(s: str):
    # ログ欄へ1行追記
    window["-LOG-"].update(s + "\n", append=True)

def set_progress(cur: int, total: int):
    # 進捗バー更新（0〜100%）
    total = max(total, 1)
    pct = int(cur * 100 / total)
    window["-PROG-"].update(current_count=pct)

# =============================================================================
# 実行関数（Stage0）
# =============================================================================
def run_single_image(values):
    """
    Stage0: 単一画像OCRをバックグラウンドで実行し、結果テキストを同ディレクトリに保存。
    """
    img_file = values.get("-IMGFILE-", "").strip()
    window.write_event_value("-STARTED-", 1)
    if not img_file or not os.path.isfile(img_file) or not is_image(img_file):
        window.write_event_value("-LOG-", "有効な画像ファイルを指定してください。")
        window.write_event_value("-FINISHED-", (0, 1, 1))
        return

    try:
        vlm = JPVLM(threads=None)  # threadsは将来用（Stage0では無指定）
    except Exception as e:
        window.write_event_value("-LOG-", f"モデル初期化失敗: {e}")
        window.write_event_value("-FINISHED-", (0, 1, 1))
        return

    try:
        text, meta = ocr_image(vlm, img_file, pre="none", paragraphize=False)
        # 出力（Stage0はテキストのみ。JSONやYAMLは“魅せるタイミング”で開示）
        stem = os.path.splitext(os.path.basename(img_file))[0]
        txt_path = os.path.join(os.path.dirname(img_file), f"{stem}_vlm.txt")
        write_text(txt_path, text)
        msg = f"OK: {os.path.basename(img_file)} -> {os.path.basename(txt_path)}"
        window.write_event_value("-STEP-", (1, 1, msg))
        window.write_event_value("-FINISHED-", (1, 0, 1))
    except Exception as e:
        window.write_event_value("-STEP-", (1, 1),)
        window.write_event_value("-LOG-", f"NG: {os.path.basename(img_file)} -> {e}")
        window.write_event_value("-FINISHED-", (0, 1, 1))

# =============================================================================
# イベントループ（Stage0のみ）
# =============================================================================
while True:
    event, values = window.read(timeout=100)
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    if event == "-RUN-IMG-":
        window["-LOG-"].update("")
        window["-PROG-"].update(current_count=0)
        log(f"画像VLM-OCRを開始します…（backend={_VLM_BACKEND}）")
        threading.Thread(target=run_single_image, args=(values,), daemon=True).start()

    if event == "-STARTED-":
        total = values[event]
        set_progress(0, total if isinstance(total, int) else 1)

    if event == "-STEP-":
        done, total, msg = values[event]
        log(msg)
        set_progress(done, total)

    if event == "-FINISHED-":
        ok, ng, total = values[event]
        log(f"\nSummary: OK={ok}, NG={ng}, Total={total}")
        set_progress(total, total)

    if event == "-LOG-":
        log(values[event])

window.close()
# EOF

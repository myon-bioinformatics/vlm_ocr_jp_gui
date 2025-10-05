# -*- coding: utf-8 -*-
"""日本語VLM OCR GUI（単一ファイル実行、本体ロジック）"""

# =============================================================================
# SHOWCASE NOTE
# - これは「Stage0（単一画像の即時OCR）」のみ有効化したデモ用実装です。
# - Stage1〜4 相当の GUI 部・実行関数・イベント分岐は “丸ごとコメントアウト” で同梱します。
# - SHOWCASE_LEVEL は概念のみ。分岐ロジックは実装しません（コメントで明示）。
# =============================================================================

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
# def has_ffmpeg() -> bool: ...
# def ffmpeg_handshake(timeout_sec: int = 10) -> dict: ...
# def extract_frames_ffmpeg(video_path: str, out_dir: str, fps: float) -> List[Tuple[str, int]]: ...
# def extract_frames_opencv(video_path: str, out_dir: str, interval_ms: int) -> List[Tuple[str, int]]: ...
# def gather_vlm_info(vlm: "JPVLM") -> str: ...
# def gather_env_info(vlm: Optional["JPVLM"] = None) -> str: ...
# def gather_diag_info() -> str: ...

# =============================================================================
# JPVLM Wrapper（バックエンド切替の薄い皮）
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
# GUI（Stage0有効／Stage1+はコメント同梱）
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

# -----------------------------------------------------------------------------
# ↓↓↓ Stage1〜4 の GUI 要素（“残すために”丸ごとコメントアウト） ↓↓↓
# -----------------------------------------------------------------------------
# # [Stage1] フォルダ一括/パターン/再帰/出力先/上書き/段落整形
# layout_stage1 = [
#     [sg.Frame("入力（画像・拡張）", [
#         [sg.Text("単一画像"), sg.Input(key="-IMGFILE-", size=(48,1)),
#          sg.FileBrowse(file_types=(("Images","*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff"),("All","*.*")))],
#         [sg.Text("画像フォルダ"), sg.Input(key="-IMGDIR-", size=(48,1)), sg.FolderBrowse()],
#         [sg.Checkbox("サブフォルダ再帰", key="-REC-", default=True),
#          sg.Text("パターン( ; 区切り)"), sg.Input("*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff", key="-PAT-", size=(30,1))]
#     ])],
#     [sg.Frame("出力設定", [
#         [sg.Text("出力先（未指定なら元の場所）"), sg.Input(key="-OUTDIR-", size=(40,1)), sg.FolderBrowse()],
#         [sg.Checkbox("既存出力を上書き", key="-OVW-")],
#         [sg.Checkbox("段落整形（文章化）", key="-MERGE-", default=True)]
#     ])],
#     [sg.Button("画像を一括実行", key="-RUN-IMGS-")],
# ]
#
# # [Stage2] 前処理 & 並列ジョブ数
# layout_stage2 = [
#     [sg.Frame("前処理/並列", [
#         [sg.Text("前処理"), sg.Combo(["none","binarize","binarize+sharpen","scale2x"],
#                                      default_value="none", key="-PRE-", size=(20,1))],
#         [sg.Text("並列ジョブ数(画像のみ)"),
#          sg.Spin([i for i in range(1, 17)], initial_value=max(1, (os.cpu_count() or 4) - 1), key="-JOBS-", size=(5,1))]
#     ])],
# ]
#
# # [Stage3] 動画入力 + FFmpeg/Interval
# layout_stage3 = [
#     [sg.Frame("入力（動画）", [
#         [sg.Text("動画ファイル"), sg.Input(key="-VIDFILE-", size=(48,1)),
#          sg.FileBrowse(file_types=(("Video","*.mp4;*.mov;*.mkv;*.avi"),("All","*.*")))],
#         [sg.Checkbox("FFmpegで抽出（推奨）", key="-USEFFMPEG-", default=True),
#          sg.Text("FPS(FFmpeg)"), sg.Input("0.5", key="-FPS-", size=(6,1)),
#          sg.Text("または Interval(ms, OpenCV)"), sg.Input("2000", key="-INTMS-", size=(8,1))]
#     ])],
#     [sg.Button("動画を実行", key="-RUN-VID-")],
# ]
#
# # [Stage4] 環境情報/詳細診断/FFmpeg疎通
# layout_stage4 = [
#     [sg.Frame("診断/環境", [
#         [sg.Text("FFmpeg:"), sg.Text("未確認", key="-FFMPEG-STATUS-", text_color="orange"),
#          sg.Button("疎通テスト", key="-FFMPEG-PING-"),
#          sg.Button("環境情報", key="-ENV-INFO-"),
#          sg.Button("詳細診断", key="-DIAG-INFO-")],
#     ])],
# ]
#
# # 全体にマージする場合は layout = layout + layout_stage1 + layout_stage2 + layout_stage3 + layout_stage4
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

# -----------------------------------------------------------------------------
# ↓↓↓ Stage1〜4 の 実行関数（“残すために”丸ごとコメントアウト） ↓↓↓
# -----------------------------------------------------------------------------
# def gather_images(file_input: Optional[str], dir_input: Optional[str], pattern: str, recursive: bool) -> List[str]:
#     out: List[str] = []
#     if file_input:
#         f = os.path.abspath(file_input)
#         if os.path.isfile(f) and is_image(f):
#             out.append(f)
#     if dir_input and os.path.isdir(dir_input):
#         root = os.path.abspath(dir_input)
#         pats = [p.strip() for p in pattern.split(";") if p.strip()]
#         if not pats:
#             pats = ["*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff"]
#         for pat in pats:
#             pat2 = ("**/" + pat) if recursive else pat
#             for p in glob.iglob(os.path.join(root, pat2), recursive=recursive):
#                 if os.path.isfile(p) and is_image(p):
#                     out.append(os.path.abspath(p))
#     return list(dict.fromkeys(out))
#
# def out_paths_for_image(img_path: str, outdir: Optional[str]) -> Tuple[str, str]:
#     stem = os.path.splitext(os.path.basename(img_path))[0]
#     if outdir:
#         ensure_dir(outdir)
#         return os.path.join(outdir, f"{stem}_vlm.txt"), os.path.join(outdir, f"{stem}_vlm.json")
#     base = os.path.dirname(img_path)
#     return os.path.join(base, f"{stem}_vlm.txt"), os.path.join(base, f"{stem}_vlm.json")
#
# def run_images(values):
#     img_file = values.get("-IMGFILE-", "").strip()
#     img_dir  = values.get("-IMGDIR-", "").strip()
#     recursive = values.get("-REC-", True)
#     pattern   = values.get("-PAT-", "").strip()
#     outdir    = (values.get("-OUTDIR-", "") or None)
#     overwrite = values.get("-OVW-", False)
#     pre       = values.get("-PRE-", "none")
#     merge     = values.get("-MERGE-", True)
#     jobs      = int(values.get("-JOBS-", 1) or 1)
#
#     targets = gather_images(img_file, img_dir, pattern, recursive)
#     window.write_event_value("-STARTED-", len(targets))
#     if not targets:
#         window.write_event_value("-LOG-", "画像が見つかりません。")
#         window.write_event_value("-FINISHED-", (0, 0, 0))
#         return
#
#     try:
#         vlm = JPVLM(threads=None)
#     except Exception as e:
#         window.write_event_value("-LOG-", f"モデル初期化失敗: {e}")
#         window.write_event_value("-FINISHED-", (0, 0, 0))
#         return
#
#     def _do_one(p: str):
#         try:
#             txt_path, json_path = out_paths_for_image(p, outdir)
#             if (not overwrite) and os.path.exists(txt_path) and os.path.exists(json_path):
#                 return True, f"SKIP (exists): {os.path.basename(p)}"
#             text, meta = ocr_image(vlm, p, pre, merge)
#             write_text(txt_path, text)
#             with open(json_path, "w", encoding="utf-8") as f:
#                 json.dump({"image": p, "text": text, "meta": meta}, f, ensure_ascii=False, indent=2)
#             return True, f"OK: {os.path.basename(p)}"
#         except Exception as e:
#             return False, f"NG: {os.path.basename(p)} -> {e}"
#
#     ok = ng = done = 0
#     if jobs <= 1:
#         for p in targets:
#             success, msg = _do_one(p)
#             done += 1; ok += int(success); ng += int(not success)
#             window.write_event_value("-STEP-", (done, len(targets), msg))
#     else:
#         from concurrent.futures import ThreadPoolExecutor, as_completed
#         with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
#             futs = {ex.submit(_do_one, p): p for p in targets}
#             for fut in as_completed(futs):
#                 success, msg = fut.result()
#                 done += 1; ok += int(success); ng += int(not success)
#                 window.write_event_value("-STEP-", (done, len(targets), msg))
#     window.write_event_value("-FINISHED-", (ok, ng, len(targets)))
#
# def extract_frames_ffmpeg(video_path: str, out_dir: str, fps: float) -> List[Tuple[str, int]]:
#     ensure_dir(out_dir)
#     pattern = os.path.join(out_dir, "frame_%06d.png")
#     cmd = ["ffmpeg","-hide_banner","-nostdin","-loglevel","error","-y","-i",video_path,"-vf",f"fps={fps}",pattern]
#     import subprocess
#     subprocess.run(cmd, check=True)
#     frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
#     out = []
#     for i, fp in enumerate(frames, start=1):
#         ms = int(1000 * (i-1) / fps)
#         out.append((fp, ms))
#     return out
#
# def extract_frames_opencv(video_path: str, out_dir: str, interval_ms: int) -> List[Tuple[str, int]]:
#     ensure_dir(out_dir)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError("Could not open video")
#     frames, t, idx = [], 0, 0
#     while True:
#         cap.set(cv2.CAP_PROP_POS_MSEC, t)
#         ok, frame = cap.read()
#         if not ok: break
#         idx += 1
#         fp = os.path.join(out_dir, f"frame_{idx:06d}.png")
#         cv2.imwrite(fp, frame)
#         frames.append((fp, t))
#         t += interval_ms
#     cap.release()
#     return frames
#
# def ocr_video(vlm: JPVLM, video_path: str, out_dir: str,
#               use_ffmpeg: bool, fps: float, interval_ms: int,
#               pre: str, paragraphize: bool,
#               min_chars: int = 2) -> Tuple[str, List[dict]]:
#     frames_dir = os.path.join(out_dir, "frames")
#     frames = extract_frames_ffmpeg(video_path, frames_dir, fps) if use_ffmpeg else \
#              extract_frames_opencv(video_path, frames_dir, interval_ms)
#     merged_blocks: List[str] = []; records: List[dict] = []
#     for fp, ms in frames:
#         text, meta = ocr_image(vlm, fp, pre, paragraphize=False)
#         lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_chars]
#         if not lines: continue
#         block = "\n".join(lines)
#         if not merged_blocks or block != merged_blocks[-1]:
#             merged_blocks.append(block)
#         records.append({"frame": os.path.basename(fp), "ms": ms, "text": "\n".join(lines), "meta": meta})
#         write_text(os.path.join(out_dir, f"{os.path.splitext(os.path.basename(fp))[0]}.txt"), "\n".join(lines))
#     merged_text = "\n\n".join(merged_blocks)
#     if paragraphize:
#         merged_text = to_paragraph(merged_text.splitlines())
#     return merged_text, records
#
# def run_video(values):
#     vid = values.get("-VIDFILE-", "").strip()
#     if not vid:
#         window.write_event_value("-LOG-", "動画ファイルを指定してください。")
#         return
#     outdir = values.get("-OUTDIR-", "").strip() or os.path.join(os.path.dirname(vid), os.path.splitext(os.path.basename(vid))[0] + "_vlm_ocr")
#     overwrite = values.get("-OVW-", False)
#     use_ffm = values.get("-USEFFMPEG-", True)
#     fps = float(values.get("-FPS-", "0.5") or "0.5")
#     int_ms = int(values.get("-INTMS-", "2000") or "2000")
#     pre = values.get("-PRE_", "none")
#     merge = values.get("-MERGE_", True)
#
#     ensure_dir(outdir)
#     merged_txt = os.path.join(outdir, "video_vlm_merged.txt")
#     jsonl_path = os.path.join(outdir, "video_vlm.jsonl")
#     if (not overwrite) and os.path.exists(merged_txt):
#         window.write_event_value("-LOG-", f"SKIP: {merged_txt} が既に存在（上書きOFF）")
#         return
#     try:
#         vlm = JPVLM(threads=None)
#     except Exception as e:
#         window.write_event_value("-LOG-", f"モデル初期化失敗: {e}")
#         return
#     window.write_event_value("-STARTED-", 100)
#     try:
#         merged_text, records = ocr_video(vlm, vid, outdir, use_ffm, fps, int_ms, pre, merge)
#         write_text(merged_txt, merged_text)
#         if os.path.exists(jsonl_path) and overwrite:
#             os.remove(jsonl_path)
#         for r in records:
#             with open(jsonl_path, "a", encoding="utf-8") as f:
#                 f.write(json.dumps({"video": vid, **r}, ensure_ascii=False) + "\n")
#         window.write_event_value("-STEP-", (100, 100, f"OK: {os.path.basename(vid)} -> {merged_txt}"))
#         window.write_event_value("-FINISHED-", (1, 0, 1))
#     except Exception as e:
#         window.write_event_value("-FINISHED-", (0, 1, 1))
#         window.write_event_value("-LOG-", f"動画処理失敗: {e}")
# -----------------------------------------------------------------------------

# =============================================================================
# イベントループ（Stage0のみ実働／Stage1+はコメントで温存）
# =============================================================================
while True:
    event, values = window.read(timeout=100)
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    # --- Stage0: 単一画像ボタン ---
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

    # -------------------------------------------------------------------------
    # ↓↓↓ Stage1〜4 の イベント分岐（“残すために”丸ごとコメントアウト） ↓↓↓
    # -------------------------------------------------------------------------
    # if event == "-RUN-IMGS-":       # [Stage1] 画像一括
    #     window["-LOG-"].update("")
    #     window["-PROG-"].update(current_count=0)
    #     log("画像一括OCRを開始します…")
    #     threading.Thread(target=run_images, args=(values,), daemon=True).start()
    #
    # if event == "-RUN-VID-":        # [Stage3] 動画処理
    #     window["-LOG-"].update("")
    #     window["-PROG-"].update(current_count=0)
    #     log("動画VLM-OCR開始…")
    #     threading.Thread(target=run_video, args=(values,), daemon=True).start()
    #
    # if event == "-FFMPEG-PING-":    # [Stage4] FFmpeg疎通テスト
    #     def _ping():
    #         try:
    #             import subprocess, shutil
    #             cmd = ["ffmpeg","-hide_banner","-loglevel","info","-version"]
    #             proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    #             out = (proc.stdout or proc.stderr or "").splitlines()
    #             res = {"ok": proc.returncode == 0,
    #                    "path": shutil.which("ffmpeg") or "",
    #                    "version": out[0] if out else ""}
    #         except Exception as e:
    #             res = {"ok": False, "path": "", "version": f"ERR:{e}"}
    #         window.write_event_value("-FFMPEG-PING-RESULT-", res)
    #     threading.Thread(target=_ping, daemon=True).start()
    #
    # if event == "-FFMPEG-PING-RESULT-":  # [Stage4] 表示側
    #     res = values[event]
    #     window["-FFMPEG-STATUS-"].update("OK" if res.get("ok") else "NG",
    #                                      text_color=("green" if res.get("ok") else "red"))
    #     log(f"FFmpeg: path={res.get('path')}, ver={res.get('version')}")
    #
    # if event == "-ENV-INFO-":        # [Stage4] 環境情報（軽版）
    #     try:
    #         import numpy as _np
    #         info = [
    #             f"OS     : {platform.system()} {platform.release()}",
    #             f"Python : {platform.python_version()}",
    #             f"GUI    : {_GUI_BACKEND}",
    #             f"NumPy  : {_np.__version__}",
    #             f"Pillow : {getattr(PILImage, '__version__', 'unknown')}",
    #             f"Torch  : {torch.__version__}",
    #         ]
    #         sg.popup_scrolled("\n".join(info), title="環境情報", size=(80, 20), modal=True, font=("Consolas", 10))
    #     except Exception as e:
    #         log(f"環境情報取得失敗: {e}")
    #
    # if event == "-DIAG-INFO-":       # [Stage4] 詳細診断（超軽版）
    #     try:
    #         intra = getattr(torch, "get_num_threads", lambda: "unknown")()
    #         inter = getattr(torch, "get_num_interop_threads", lambda: "unknown")()
    #         txt = "\n".join([f"intra-op threads: {intra}", f"interop threads : {inter}"])
    #         sg.popup_scrolled(txt, title="詳細診断", size=(80, 20), modal=True, font=("Consolas", 10))
    #     except Exception as e:
    #         log(f"詳細診断失敗: {e}")
    # -------------------------------------------------------------------------

window.close()
# EOF

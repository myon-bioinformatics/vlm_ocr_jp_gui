# -*- coding: utf-8 -*-
"""
日本語VLM OCR GUI（manga-ocr）
- 画像/フォルダ一括 + 動画の等間隔フレームOCR（FFmpegあり/なし両対応）
- PySimpleGUI v4 / FreeSimpleGUI の順で柔軟フォールバック
- 並列ジョブ数（画像処理のみ）・FFmpeg疎通テスト
- 「環境情報」ボタン（主要ライブラリやFFmpeg等のバージョン）
- 「詳細診断」ボタン（Torchスレッド数・CPU名・メモリ使用量など）
"""

# ===== Standard Library (grouped import) =====
import sys, os, glob, json, shutil, subprocess, threading, platform, textwrap, math, time, importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from importlib import metadata as importlib_metadata

# ===== Third-Party =====
import cv2
import numpy as np
import PIL.Image as PILImage  # FreeSimpleGUIのImageと衝突しないよう別名
import torch
from manga_ocr import MangaOcr

# Optional diagnostics (任意導入。未インストールならスキップ)
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None
try:
    import cpuinfo  # py-cpuinfo
except Exception:  # pragma: no cover
    cpuinfo = None

# ===== Flexible SG import (v4優先) =====
tp = os.path.join(os.path.dirname(__file__), "third_party")
if os.path.isdir(tp) and tp not in sys.path:
    sys.path.insert(0, tp)
_GUI_BACKEND = None

try:
    # 1) プロジェクト同梱の PySimpleGUI.py（v4ラッパー/単体ファイル）
    import FreeSimpleGUI as sg
    _GUI_BACKEND = "FreeSimpleGUI(v4 fork)"
except Exception:
    try:
        # 2) FreeSimpleGUI（v4系LGPLフォーク）
        import PySimpleGUI as sg
        _GUI_BACKEND = "PySimpleGUI(v4 local/wrapper)"
    except Exception as e:
        raise ImportError(
            "No usable SG backend found. "
            "Place PySimpleGUI.py (v4) locally OR `pip install FreeSimpleGUI` "
        ) from e
finally:
    print(f"[INFO] GUI backend: {_GUI_BACKEND}")

# =========================
#  共通ユーティリティ
# =========================
def suggest_jobs() -> int:
    n = os.cpu_count() or 4
    return max(1, min(16, n - 1))

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def ffmpeg_which() -> str:
    p = shutil.which("ffmpeg")
    return os.path.abspath(p) if p else ""

def ffmpeg_handshake(timeout_sec: int = 10) -> dict:
    path = ffmpeg_which()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-version"]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            timeout=timeout_sec
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        first = out.splitlines()[0] if out else (err.splitlines()[0] if err else "")
        return {
            "ok": (proc.returncode == 0),
            "returncode": proc.returncode,
            "path": path,
            "cmd": " ".join(cmd),
            "version_line": first,
            "stdout_head": "\n".join(out.splitlines()[:40]),
            "stderr_head": "\n".join(err.splitlines()[:40]),
        }
    except FileNotFoundError as e:
        return {"ok": False, "returncode": None, "path": path, "cmd": " ".join(cmd),
                "version_line": "", "stdout_head": "", "stderr_head": f"FileNotFoundError: {e}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "path": path, "cmd": " ".join(cmd),
                "version_line": "", "stdout_head": "", "stderr_head": f"Timeout: {timeout_sec}s"}

def is_image(p: str) -> bool:
    p = p.lower()
    return any(p.endswith(ext) for ext in [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"])

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, data: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def gather_images(file_input: Optional[str], dir_input: Optional[str], pattern: str, recursive: bool) -> List[str]:
    out: List[str] = []
    if file_input:
        f = os.path.abspath(file_input)
        if os.path.isfile(f) and is_image(f):
            out.append(f)
    if dir_input and os.path.isdir(dir_input):
        root = os.path.abspath(dir_input)
        pats = [p.strip() for p in pattern.split(";") if p.strip()]
        if not pats:
            pats = ["*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff"]
        for pat in pats:
            pat2 = ("**/" + pat) if recursive else pat
            for p in glob.iglob(os.path.join(root, pat2), recursive=recursive):
                if os.path.isfile(p) and is_image(p):
                    out.append(os.path.abspath(p))
    return list(dict.fromkeys(out))

def to_paragraph(lines: List[str]) -> str:
    out, buf = [], []
    for raw in lines:
        s = raw.strip()
        if not s:
            if buf:
                out.append(" ".join(buf).replace(" - ", "-"))
                buf = []
            continue
        if s.endswith("-"):
            buf.append(s[:-1])
        else:
            buf.append(s)
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)

# === バージョン/環境情報ユーティリティ ===
def _pkg_ver(pkg: str) -> str:
    try:
        return importlib_metadata.version(pkg)
    except Exception:
        return "unknown"

def _safe_getattr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None

def _bytes_h(n: int) -> str:
    if n is None:
        return "unknown"
    units = ["B","KB","MB","GB","TB","PB"]
    if n == 0:
        return "0 B"
    i = int(math.floor(math.log(n, 1024)))
    i = min(i, len(units)-1)
    val = n / (1024 ** i)
    return f"{val:.2f} {units[i]}"

def gather_vlm_info(vlm: "JPVLM") -> str:
    """manga-ocr内部の分かる範囲のモデル情報をざっくり抽出"""
    lines = []
    try:
        core = vlm.mocr
        lines.append(f"manga-ocr pkg : {_pkg_ver('manga-ocr')}")
        lines.append(f"MangaOcr class: {type(core).__name__}")
        for attr in ["detector", "recognizer", "ocr", "model"]:
            obj = _safe_getattr(core, attr)
            if obj is None:
                continue
            lines.append(f"- {attr}: {type(obj).__name__}")
            for sub in ["name", "model_name", "pretrained_model_name_or_path",
                        "config", "ckpt_path", "weight_path", "path"]:
                val = _safe_getattr(obj, sub)
                if val is None:
                    continue
                sval = str(val)
                if len(sval) > 200:
                    sval = sval[:200] + " ... (truncated)"
                lines.append(f"    {sub}: {sval}")
    except Exception as e:
        lines.append(f"(introspection failed: {e})")
    return "\n".join(lines)

def gather_env_info(vlm: Optional["JPVLM"] = None) -> str:
    """環境情報（主要ライブラリとFFmpeg等）のサマリ"""
    gpu = "CUDA:ON" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "CUDA:OFF"
    mps = "MPS:ON" if _safe_getattr(torch.backends, "mps") and torch.backends.mps.is_available() else "MPS:OFF"
    res = ffmpeg_handshake(timeout_sec=5)

    info = [
        f"OS            : {platform.system()} {platform.release()}",
        f"Python        : {platform.python_version()}",
        f"Backend (SG)  : {_GUI_BACKEND}",
        f"PySimpleGUI   : {_pkg_ver('PySimpleGUI')}",
        f"FreeSimpleGUI : {_pkg_ver('FreeSimpleGUI')}",
        f"NumPy         : {np.__version__}",
        f"OpenCV        : {cv2.__version__}",
        f"Pillow        : {getattr(PILImage, '__version__', _pkg_ver('Pillow'))}",
        f"Torch         : {torch.__version__}  [{gpu}, {mps}]",
        f"fugashi       : {_pkg_ver('fugashi')}",
        f"unidic-lite   : {_pkg_ver('unidic-lite')}",
        f"manga-ocr     : {_pkg_ver('manga-ocr')}",
        f"FFmpeg path   : {res['path'] or '(not found)'}",
        f"FFmpeg ver    : {res['version_line'] or '(unknown)'}",
    ]
    try:
        info.append(f"OpenCV build  : {cv2.getBuildInformation().splitlines()[0]}")
    except Exception:
        pass
    if vlm is not None:
        info += ["", "== VLM (manga-ocr) =="]
        info += [gather_vlm_info(vlm)]
    return "\n".join(info)

def gather_diag_info() -> str:
    """詳細診断（Torchスレッド、CPU、メモリなど）。psutil/cpuinfo は任意。"""
    lines = []
    # Torch threads
    try:
        intra = torch.get_num_threads()
    except Exception:
        intra = "unknown"
    try:
        inter = torch.get_num_interop_threads()
    except Exception:
        inter = "unknown"

    lines += [
        "== Torch threads ==",
        f"intra-op threads : {intra}",
        f"interop threads  : {inter}",
    ]

    # CPU
    cpu_name = platform.processor() or ""
    if not cpu_name and cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            cpu_name = info.get("brand_raw") or info.get("brand") or ""
        except Exception:
            pass
    if not cpu_name:
        cpu_name = "(unknown CPU)"
    logical = os.cpu_count()
    physical = None
    if psutil:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            pass

    lines += [
        "",
        "== CPU ==",
        f"CPU name        : {cpu_name}",
        f"Logical cores   : {logical}",
        f"Physical cores  : {physical if physical is not None else '(unknown)'}",
    ]

    # Memory
    if psutil:
        try:
            vm = psutil.virtual_memory()
            pm = psutil.Process(os.getpid()).memory_info()
            lines += [
                "",
                "== Memory ==",
                f"System total    : {_bytes_h(vm.total)}",
                f"System avail    : {_bytes_h(vm.available)}",
                f"System used     : {_bytes_h(vm.used)} ({vm.percent}%)",
                f"Process RSS     : {_bytes_h(pm.rss)}",
                f"Process VMS     : {_bytes_h(getattr(pm, 'vms', 0))}",
            ]
        except Exception:
            lines += ["", "== Memory ==", "(psutil error)"]
    else:
        lines += ["", "== Memory ==", "(psutil not installed)"]

    # Optional: torch parallel info (head)
    try:
        pi = torch.__config__.parallel_info()
        if pi:
            head = "\n".join([ln for ln in pi.splitlines()][:25])
            lines += ["", "== Torch parallel info (head) ==", head]
    except Exception:
        pass

    return "\n".join(lines)

# =========================
#  VLM (manga-ocr)
# =========================
class JPVLM:
    def __init__(self, threads: Optional[int] = None):
        if threads and threads > 0:
            try:
                torch.set_num_threads(threads)
            except Exception:
                pass
        self.mocr = MangaOcr()  # CPU使用

    def ocr(self, img: PILImage.Image) -> Tuple[str, Dict]:
        text = self.mocr(img)
        return text.strip(), {"engine": "manga-ocr"}

def preprocess(img: PILImage.Image, mode: str) -> PILImage.Image:
    if mode == "none":
        return img.convert("RGB")
    arr = np.array(img.convert("L"))
    if mode in ("binarize","binarize+sharpen"):
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr = th
        if mode.endswith("sharpen"):
            arr = cv2.GaussianBlur(arr, (0,0), 1.0)
            arr = cv2.addWeighted(arr, 1.5, cv2.GaussianBlur(arr, (0,0), 2.0), -0.5, 0)
    if mode == "scale2x":
        arr = cv2.resize(arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return PILImage.fromarray(arr).convert("RGB")

# =========================
#  動画フレーム抽出
# =========================
def extract_frames_ffmpeg(video_path: str, out_dir: str, fps: float) -> List[Tuple[str, int]]:
    ensure_dir(out_dir)
    pattern = os.path.join(out_dir, "frame_%06d.png")
    cmd = ["ffmpeg","-hide_banner","-nostdin","-loglevel","error","-y","-i",video_path,"-vf",f"fps={fps}",pattern]
    subprocess.run(cmd, check=True)
    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
    out = []
    for i, fp in enumerate(frames, start=1):
        ms = int(1000 * (i-1) / fps)
        out.append((fp, ms))
    return out

def extract_frames_opencv(video_path: str, out_dir: str, interval_ms: int) -> List[Tuple[str, int]]:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    frames, t, idx = [], 0, 0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        fp = os.path.join(out_dir, f"frame_{idx:06d}.png")
        cv2.imwrite(fp, frame)
        frames.append((fp, t))
        t += interval_ms
    cap.release()
    return frames

# =========================
#  OCR 実処理
# =========================
def ocr_image(vlm: JPVLM, img_path: str, pre: str, paragraphize: bool) -> Tuple[str, dict]:
    img = PILImage.open(img_path)
    img_p = preprocess(img, pre)
    text, meta = vlm.ocr(img_p)
    if paragraphize:
        text = to_paragraph(text.splitlines())
    return text, meta

def ocr_video(vlm: JPVLM, video_path: str, out_dir: str,
              use_ffmpeg: bool, fps: float, interval_ms: int,
              pre: str, paragraphize: bool,
              min_chars: int = 2) -> Tuple[str, List[dict]]:
    frames_dir = os.path.join(out_dir, "frames")
    frames = extract_frames_ffmpeg(video_path, frames_dir, fps) if (use_ffmpeg and has_ffmpeg()) \
             else extract_frames_opencv(video_path, frames_dir, interval_ms)

    merged_blocks: List[str] = []; records: List[dict] = []
    for fp, ms in frames:
        text, meta = ocr_image(vlm, fp, pre, paragraphize=False)
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_chars]
        if not lines:
            continue
        block = "\n".join(lines)
        if not merged_blocks or block != merged_blocks[-1]:
            merged_blocks.append(block)
        records.append({
            "frame": os.path.basename(fp),
            "ms": ms,
            "text": "\n".join(lines),
            "meta": meta
        })
        write_text(os.path.join(out_dir, f"{os.path.splitext(os.path.basename(fp))[0]}.txt"), "\n".join(lines))

    merged_text = "\n\n".join(merged_blocks)
    if paragraphize:
        merged_text = to_paragraph(merged_text.splitlines())
    return merged_text, records

# =========================
#  GUI
# =========================
sg.theme("BlueMono")
layout = [
    [sg.Text("日本語VLM文字起こし（CPU / manga-ocr） – 画像 & 動画", font=("Segoe UI", 12, "bold"))],
    [sg.Frame("入力（画像）", [
        [sg.Text("単一画像"), sg.Input(key="-IMGFILE-", size=(48,1)), sg.FileBrowse(file_types=(("Images","*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff"),("All","*.*")))],
        [sg.Text("画像フォルダ"), sg.Input(key="-IMGDIR-", size=(48,1)), sg.FolderBrowse()],
        [sg.Checkbox("サブフォルダ再帰", key="-REC-", default=True),
         sg.Text("パターン( ; 区切り)"), sg.Input("*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff", key="-PAT-", size=(30,1))]
    ])],
    [sg.Frame("入力（動画）", [
        [sg.Text("動画ファイル"), sg.Input(key="-VIDFILE-", size=(48,1)), sg.FileBrowse(file_types=(("Video","*.mp4;*.mov;*.mkv;*.avi"),("All","*.*")))],
        [sg.Checkbox("FFmpegで抽出（推奨）", key="-USEFFMPEG-", default=True),
         sg.Text("FPS(FFmpeg)"), sg.Input("0.5", key="-FPS-", size=(6,1)),
         sg.Text("または Interval(ms, OpenCV)"), sg.Input("2000", key="-INTMS-", size=(8,1))]
    ])],
    [sg.Frame("前処理 & 出力", [
        [sg.Text("前処理"), sg.Combo(["none","binarize","binarize+sharpen","scale2x"], default_value="none", key="-PRE-", size=(16,1)),
         sg.Checkbox("段落整形（文章化）", key="-MERGE-", default=True),
         sg.Text("並列ジョブ数(画像のみ)"), sg.Spin([i for i in range(1, 17)], initial_value=suggest_jobs(), key="-JOBS-", size=(5,1))],
        [sg.Text("FFmpeg:"), sg.Text("未確認", key="-FFMPEG-STATUS-", text_color="orange"),
         sg.Button("疎通テスト", key="-FFMPEG-PING-"),
         sg.Button("環境情報", key="-ENV-INFO-"),
         sg.Button("詳細診断", key="-DIAG-INFO-"),
         sg.Text("出力先（未指定なら元の場所）"), sg.Input(key="-OUTDIR-", size=(28,1)), sg.FolderBrowse()],
        [sg.Checkbox("既存出力を上書き", key="-OVW-")]
    ])],
    [sg.ProgressBar(max_value=100, orientation="h", size=(50,20), key="-PROG-")],
    [sg.Multiline(size=(100,14), key="-LOG-", autoscroll=True, disabled=True)],
    [sg.Button("画像を実行", key="-RUN-IMG-"), sg.Button("動画を実行", key="-RUN-VID-"), sg.Button("Exit")]
]
window = sg.Window(f"JP VLM OCR (manga-ocr) – PySimpleGUI v4互換  [{_GUI_BACKEND}]", layout, finalize=True)

# 起動時FFmpegステータス
window["-FFMPEG-STATUS-"].update("OK" if has_ffmpeg() else "NG",
                                 text_color=("green" if has_ffmpeg() else "red"))
# 起動サマリを出したい場合は下行を有効化

def log(s: str):
    window["-LOG-"].update(s + "\n", append=True)

def set_progress(cur: int, total: int):
    total = max(total, 1)
    pct = int(cur * 100 / total)
    window["-PROG-"].update(current_count=pct)

def out_paths_for_image(img_path: str, outdir: Optional[str]) -> Tuple[str, str]:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if outdir:
        ensure_dir(outdir)
        return os.path.join(outdir, f"{stem}_vlm.txt"), os.path.join(outdir, f"{stem}_vlm.json")
    base = os.path.dirname(img_path)
    return os.path.join(base, f"{stem}_vlm.txt"), os.path.join(base, f"{stem}_vlm.json")

# 直近で使ったVLM（環境情報/診断のために保持）
_last_vlm = None

def run_images(values):
    global _last_vlm
    img_file = values["-IMGFILE-"].strip()
    img_dir = values["-IMGDIR-"].strip()
    recursive = values["-REC-"]
    pattern = values["-PAT-"].strip()
    outdir = values["-OUTDIR-"].strip() or None
    overwrite = values["-OVW-"]
    pre = values["-PRE-"]
    merge = values["-MERGE-"]
    jobs = int(values["-JOBS-"] or 1)

    targets = gather_images(img_file, img_dir, pattern, recursive)
    window.write_event_value("-STARTED-", len(targets))
    if not targets:
        window.write_event_value("-LOG-", "画像が見つかりません。")
        return

    try:
        vlm = JPVLM(threads=None)  # 推論は内部でCPUスレッドを管理
        _last_vlm = vlm
    except Exception as e:
        window.write_event_value("-LOG-", f"モデル初期化失敗: {e}")
        return

    ok = ng = done = 0

    def _do_one(p: str):
        try:
            txt_path, json_path = out_paths_for_image(p, outdir)
            if (not overwrite) and os.path.exists(txt_path) and os.path.exists(json_path):
                return True, f"SKIP (exists): {os.path.basename(p)}"
            text, meta = ocr_image(vlm, p, pre, merge)
            write_text(txt_path, text)
            write_json(json_path, {"image": p, "text": text, "meta": meta})
            return True, f"OK: {os.path.basename(p)}"
        except Exception as e:
            return False, f"NG: {os.path.basename(p)} -> {e}"

    if jobs <= 1:
        for p in targets:
            success, msg = _do_one(p)
            done += 1; ok += int(success); ng += int(not success)
            window.write_event_value("-STEP-", (done, len(targets), msg))
    else:
        # 注意：CPUオンリー推論は並列しすぎると逆に遅くなる場合あり
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = {ex.submit(_do_one, p): p for p in targets}
            for fut in as_completed(futs):
                success, msg = fut.result()
                done += 1; ok += int(success); ng += int(not success)
                window.write_event_value("-STEP-", (done, len(targets), msg))

    window.write_event_value("-FINISHED-", (ok, ng, len(targets)))

def run_video(values):
    global _last_vlm
    vid = values["-VIDFILE-"].strip()
    if not vid:
        window.write_event_value("-LOG-", "動画ファイルを指定してください。")
        return
    outdir = values["-OUTDIR-"].strip() or os.path.join(os.path.dirname(vid), os.path.splitext(os.path.basename(vid))[0] + "_vlm_ocr")
    overwrite = values["-OVW-"]
    use_ffm = values["-USEFFMPEG-"]
    fps = float(values["-FPS-"] or "0.5")
    int_ms = int(values["-INTMS-"] or "2000")
    pre = values["-PRE-"]
    merge = values["-MERGE-"]

    ensure_dir(outdir)
    merged_txt = os.path.join(outdir, "video_vlm_erged.txt".replace("erged","merged"))
    jsonl_path = os.path.join(outdir, "video_vlm.jsonl")
    if (not overwrite) and os.path.exists(merged_txt):
        window.write_event_value("-LOG-", f"SKIP: {merged_txt} が既に存在（上書きOFF）")
        return

    try:
        vlm = JPVLM(threads=None)
        _last_vlm = vlm
    except Exception as e:
        window.write_event_value("-LOG-", f"モデル初期化失敗: {e}")
        return

    window.write_event_value("-STARTED-", 100)
    try:
        merged_text, records = ocr_video(vlm, vid, outdir, use_ffm, fps, int_ms, pre, merge)
        write_text(merged_txt, merged_text)
        if os.path.exists(jsonl_path) and overwrite:
            os.remove(jsonl_path)
        for r in records:
            append_jsonl(jsonl_path, {"video": vid, **r})
        window.write_event_value("-STEP-", (100, 100, f"OK: {os.path.basename(vid)} -> {merged_txt}"))
        window.write_event_value("-FINISHED-", (1, 0, 1))
    except subprocess.CalledProcessError as e:
        window.write_event_value("-FINISHED-", (0, 1, 1))
        window.write_event_value("-LOG-", f"FFmpeg実行失敗: {e}")
    except Exception as e:
        window.write_event_value("-FINISHED-", (0, 1, 1))
        window.write_event_value("-LOG-", f"動画処理失敗: {e}")

# =========================
#  イベントループ
# =========================
while True:
    event, values = window.read(timeout=100)
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    if event == "-RUN-IMG-":
        window["-LOG-"].update("")
        window["-PROG-"].update(current_count=0)
        log("画像VLM-OCRを開始します…（初回はモデルDLあり）")
        threading.Thread(target=run_images, args=(values,), daemon=True).start()

    if event == "-RUN-VID-":
        window["-LOG-"].update("")
        window["-PROG-"].update(current_count=0)
        ffm = "OK" if has_ffmpeg() else "NG"
        log(f"動画VLM-OCR開始…（FFmpeg検出: {ffm}）")
        threading.Thread(target=run_video, args=(values,), daemon=True).start()

    if event == "-FFMPEG-PING-":
        def _ping():
            res = ffmpeg_handshake(timeout_sec=10)
            window.write_event_value("-FFMPEG-PING-RESULT-", res)
        threading.Thread(target=_ping, daemon=True).start()

    if event == "-FFMPEG-PING-RESULT-":
        res = values[event]
        window["-FFMPEG-STATUS-"].update("OK" if res["ok"] else "NG",
                                         text_color=("green" if res["ok"] else "red"))
        summary = [
            f"OS: {platform.system()} {platform.release()}",
            f"Python: {platform.python_version()}",
            f"Command: {res['cmd']}",
            f"ReturnCode: {res['returncode']}",
            f"Path: {res['path'] or '(not found)'}",
            f"Version: {res['version_line']}",
            "",
            "---- stdout (head) ----",
            res["stdout_head"],
            "",
            "---- stderr (head) ----",
            res["stderr_head"],
        ]
        text = "\n".join(summary)
        window["-LOG-"].update(text + "\n\n", append=True)

    if event == "-ENV-INFO-":
        txt = gather_env_info(_last_vlm)
        sg.popup_scrolled(
            txt,
            title="環境情報",
            size=(90, 34),
            font=("Consolas", 10),  # 等幅で綺麗に
            modal=True,
        )

    if event == "-DIAG-INFO-":
        txt = gather_diag_info()
        sg.popup_scrolled(
            txt,
            title="詳細診断",
            size=(90, 34),
            font=("Consolas", 10),
            modal=True,
        )

    if event == "-STARTED-":
        total = values[event]
        set_progress(0, total if isinstance(total, int) else 100)

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
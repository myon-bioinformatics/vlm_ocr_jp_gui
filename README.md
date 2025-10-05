# vlm_ocr_jp_gui
# VLM OCR GUI (manga-ocr, CPU-only)

A lightweight desktop GUI for **Japanese OCR** that runs **fully on CPU**.

## Features
- OCR **images** (single file or batch)
- OCR **videos** by extracting frames (via **FFmpeg** FPS or **OpenCV** interval sampling)
- **Environment Info** popup (Torch, manga-ocr, OpenCV, Pillow, NumPy, FFmpeg path/version, SG backend)
- **Diagnostics** popup (Torch thread counts, CPU name/cores, memory usage via `psutil` if available)
- Prefers local **PySimpleGUI v4** if present; otherwise falls back to **FreeSimpleGUI** (LGPL).

## Requirements
- Python **3.8+** (tested on 3.10)
- OS: Windows / Linux / macOS
- Packages  
  - Required: `manga-ocr`, `torch` (CPU build), `opencv-python`, `Pillow`, `numpy`, **FreeSimpleGUI** or local `PySimpleGUI.py`  
  - Optional (Diagnostics): `psutil`, `py-cpuinfo`  
  - Optional (Video FPS extraction): **FFmpeg** in PATH

## Install
```bash
# (recommended) create a virtual env
python -m venv .VLM_venv
# Windows
.VLM_venv\Scripts\activate
# macOS/Linux
source .VLM_venv/bin/activate
```

```bash
pip install --upgrade pip
pip install manga-ocr torch --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python Pillow numpy FreeSimpleGUI
# optional
pip install psutil py-cpuinfo
```
> To use PySimpleGUI v4, place PySimpleGUI.py next to the script; it will be preferred over FreeSimpleGUI.

## FFmpeg (optional, for FPS-based video sampling)
```bash
Windows: winget install Gyan.FFmpeg

macOS: brew install ffmpeg

Linux: sudo apt-get install ffmpeg
```

## Run
```python
python vlm_ocr_jp_gui.py
```

## Usage
### Image Input

- Choose a single image or a folder (patterns like *.png;*.jpg). Enable Recursive to include subfolders.

### Video Input

- Select a video. With FFmpeg checked: extract by FPS. Without it: OpenCV extracts by interval (ms).

## Preprocessing & Output

- Preprocess: none, binarize, binarize+sharpen, scale2x

- Paragraphize merges lines into paragraphs

- Parallel jobs (images only) can speed up batch work (too many may slow CPU-only inference)

- Output directory optional (defaults next to inputs)

- Overwrite to replace existing outputs

### Buttons

- Run Images: OCR all matched images

- Run Video: Extract frames → OCR → merged transcript

- FFmpeg Test: probe ffmpeg -version

- Environment Info: popup only

- Diagnostics: popup only

### Outputs

#### Images

- {name}_vlm.txt — OCR text

- {name}_vlm.json — { image, text, meta }

#### Video

- video_vlm_merged.txt — de-duplicated merged transcript

- video_vlm.jsonl — per-frame {frame, ms, text, meta}

- frames/ — extracted PNG frames

## Tips

First run downloads the manga-ocr model (cached afterwards).

On CPU-only, binarize / binarize+sharpen often helps low-contrast scans.

For long videos, use low FPS (e.g., 0.3–0.5) or larger intervals (e.g., 2000–4000 ms).

## Troubleshooting

Image attribute error: avoid from FreeSimpleGUI import *. This project uses import FreeSimpleGUI as sg and import PIL.Image as PILImage.

> | None TypeError: use Python ≥ 3.10 or the code’s Optional[...] annotations.

FFmpeg not found: install FFmpeg and ensure it’s in PATH; use the FFmpeg Test button.

## License Notes

Your GUI code is your own. Respect third-party licenses:

manga-ocr, torch, opencv-python, Pillow, numpy

FreeSimpleGUI (LGPLv3) / bundled PySimpleGUI v4 (follow its terms).

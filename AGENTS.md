# AGENTS.md

## What this is

AR marker (ArUco) detection + overlay system. Webcam detects printed ArUco markers on paper pages, identifies page IDs, and overlays images via perspective transform.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# generate_pages.py also needs: pip install Pillow reportlab
```

## Scripts (all standalone, run with `python <file>`)

| File | Purpose |
|------|---------|
| `generate.py` | Print sheet of 50 ArUco markers (DICT_4X4_50) → `markers.png` |
| `generate_pages.py` | Generate 10 A4 PDFs with 8 markers each (DICT_4X4_100) → `pages/` |
| `detect.py` | Webcam overlay: 4 fixed markers (IDs 1,11,13,15) → warp `pic.jpg` onto paper |
| `detect_paper.py` | Webcam detection: identifies page by marker group (8 markers/page), draws page outline + center dot |

## Key conventions

- Two detection modes exist: `detect.py` (single overlay, hardcoded marker IDs) vs `detect_paper.py` (multi-page, marker ID = page_num * 8 + local_idx)
- `detect.py` uses DICT_4X4_50; `detect_paper.py` uses DICT_4X4_100 — don't mix dictionaries between generate and detect
- Webcam opens at 4K (3840x2160) with MJPG codec; press `q` to quit
- `pic.jpg` must exist in repo root for `detect.py`
- No tests, no CI, no lint config

# SpatialComputing

AR marker (ArUco) detection + overlay system. Webcam detects printed ArUco markers on paper pages, identifies page IDs, and overlays images via perspective transform.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# for PDF generation: pip install Pillow reportlab
```

## Usage

| Script | Description |
|--------|-------------|
| `generate.py` | Print sheet of 50 ArUco markers (DICT_4X4_50) → `markers.png` |
| `generate_pages.py` | Generate 10 A4 PDFs with 8 markers each (DICT_4X4_100) → `pages/` |
| `detect.py` | Webcam overlay: 4 fixed markers (IDs 1,11,13,15) → warp `pic.jpg` onto paper |
| `detect_paper.py` | Webcam detection: identifies page by marker group (8 markers/page), draws page outline + center dot |
| `calibrate.py` | Camera calibration |
| `browse_demo.py` | Browse demo |

Run any script with `python <script.py>`.

## Demo

[![Demo video](https://img.youtube.com/vi/dxgt13AxaLY/hqdefault.jpg)](https://youtube.com/shorts/dxgt13AxaLY)

## Notes

- `detect.py` uses DICT_4X4_50; `detect_paper.py` uses DICT_4X4_100 — don't mix dictionaries between generate and detect.
- Webcam opens at 4K (3840x2160) with MJPG codec; press `q` to quit.
- `pic.jpg` must exist in repo root for `detect.py`.

"""Generate 10 unique pages with 8 ArUco markers each, output as PDFs."""
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import os

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
MARKER_SIZE_CM = 2.5
PAGE_W_CM = 21.0
PAGE_H_CM = 29.7

positions_cm = [
    (0, PAGE_H_CM - MARKER_SIZE_CM),
    (PAGE_W_CM - MARKER_SIZE_CM, PAGE_H_CM - MARKER_SIZE_CM),
    (PAGE_W_CM - MARKER_SIZE_CM, 0),
    (0, 0),
    ((PAGE_W_CM - MARKER_SIZE_CM) / 2, PAGE_H_CM - MARKER_SIZE_CM),
    (PAGE_W_CM - MARKER_SIZE_CM, (PAGE_H_CM - MARKER_SIZE_CM) / 2),
    ((PAGE_W_CM - MARKER_SIZE_CM) / 2, 0),
    (0, (PAGE_H_CM - MARKER_SIZE_CM) / 2),
]

os.makedirs("pages", exist_ok=True)

for page_num in range(10):
    pdf_path = f"pages/page_{page_num:03d}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    
    marker_ids = list(range(page_num * 8, page_num * 8 + 8))
    
    for i, marker_id in enumerate(marker_ids):
        marker_img = DICT.generateImageMarker(marker_id, 400)
        marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(marker_img)
        
        temp_path = f"/tmp/marker_{marker_id}.png"
        pil_img.save(temp_path)
        
        x_cm, y_cm = positions_cm[i]
        x_pt = x_cm * cm
        y_pt = y_cm * cm
        
        c.drawImage(temp_path, x_pt, y_pt, 
                   width=MARKER_SIZE_CM * cm, 
                   height=MARKER_SIZE_CM * cm)
    
    c.save()
    print(f"Created {pdf_path} with markers {marker_ids}")

print("\nGenerated 10 pages in pages/ folder")

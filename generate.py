"""Print 50 unique ArUco markers as a single sheet (cut them out, glue to papers)."""
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
SIZE_PX = 400          # marker side length in pixels
COLS = 5
ROWS = 10             # 5 * 10 = 50
PAD = 40
LABEL_H = 50

W = COLS * SIZE_PX + (COLS + 1) * PAD
H = ROWS * (SIZE_PX + LABEL_H) + (ROWS + 1) * PAD
sheet = np.full((H, W, 3), 255, dtype=np.uint8)

for i in range(50):
    marker = DICT.generateImageMarker(i, SIZE_PX)
    marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    r, c = divmod(i, COLS)
    x = PAD + c * (SIZE_PX + PAD)
    y = PAD + r * (SIZE_PX + LABEL_H + PAD)
    sheet[y:y + SIZE_PX, x:x + SIZE_PX] = marker
    cv2.putText(sheet, f"ID {i}", (x, y + SIZE_PX + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

cv2.imwrite("markers.png", sheet)
print(f"wrote markers.png  ({W}x{H})  — print, cut, stick one marker per paper")

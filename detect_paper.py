"""Webcam → detect ArUco markers → identify papers → overlay images on pages.
Displays warped images on projector via pygame."""

import cv2
import numpy as np
import pygame
import json
from collections import defaultdict
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- ArUco setup ---
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

# --- images for each page (img1.jpg → page 0, img2.jpg → page 1, ...) ---
IMAGES = []
for i in range(1, 11):
    path = os.path.join(script_dir, f"img{i}.jpg")
    img = cv2.imread(path)
    if img is None:
        print(f"warning: couldn't load {path}")
    IMAGES.append(img)

# --- page geometry (matches generate_pages.py) ---
MARKERS_PER_PAGE = 8
PAGE_W_CM = 21.0
PAGE_H_CM = 29.7

EXPECTED_POSITIONS = {
    0: (0, PAGE_H_CM - 2.5),
    1: (PAGE_W_CM - 2.5, PAGE_H_CM - 2.5),
    2: (PAGE_W_CM - 2.5, 0),
    3: (0, 0),
    4: ((PAGE_W_CM - 2.5) / 2, PAGE_H_CM - 2.5),
    5: (PAGE_W_CM - 2.5, (PAGE_H_CM - 2.5) / 2),
    6: ((PAGE_W_CM - 2.5) / 2, 0),
    7: (0, (PAGE_H_CM - 2.5) / 2),
}

# --- projector calibration (camera → projector homography) ---
cal_path = os.path.join(script_dir, "calibration.json")
if os.path.exists(cal_path):
    with open(cal_path) as f:
        cal = json.load(f)
    H_PROJ = np.array(cal["projector_homography"], dtype=np.float64)
    print(f"loaded projector homography from {cal_path}")
else:
    H_PROJ = np.eye(3, dtype=np.float64)
    print("warning: no calibration.json — projector will mirror camera view")

PROJ_W, PROJ_H = 1920, 1080

# --- camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("no webcam")

# --- pygame (projector display) ---
pygame.init()
proj_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=1)
pygame.display.set_caption("projector")
clock = pygame.time.Clock()

smoothed_corners = {}
SMOOTHING_ALPHA = 0.3
last_seen_frame = {}
STALE_FRAMES = 10
frame_count = 0
DETECT_EVERY_N = 5

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            raise SystemExit()

    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1

    if frame_count % DETECT_EVERY_N == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = DETECTOR.detectMarkers(gray)

        if ids is not None:
            pages = defaultdict(list)
            for i, marker_id in enumerate(ids.flatten()):
                page_num = marker_id // MARKERS_PER_PAGE
                local_idx = marker_id % MARKERS_PER_PAGE
                center = corners[i][0].mean(axis=0)
                pages[page_num].append((local_idx, center))

            for page_num, markers in pages.items():
                if len(markers) >= 4:
                    corner_centers = {}
                    for local_idx, detected_center in markers:
                        if local_idx in EXPECTED_POSITIONS:
                            corner_centers[local_idx] = detected_center

                    if all(i in corner_centers for i in [0, 1, 2, 3]):
                        last_seen_frame[page_num] = frame_count
                        page_corners_cam = np.array(
                            [
                                corner_centers[0],
                                corner_centers[1],
                                corner_centers[2],
                                corner_centers[3],
                            ],
                            dtype=np.float32,
                        )

                        if page_num in smoothed_corners:
                            old = smoothed_corners[page_num]
                            smoothed_c = (
                                SMOOTHING_ALPHA * page_corners_cam
                                + (1 - SMOOTHING_ALPHA) * old
                            ).astype(np.float32)
                        else:
                            smoothed_c = page_corners_cam

                        smoothed_corners[page_num] = smoothed_c

    # --- render projector view ---
    proj_frame = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)

    for page_num, last_seen in list(last_seen_frame.items()):
        if frame_count - last_seen <= STALE_FRAMES and page_num in smoothed_corners:
            img_idx = page_num % len(IMAGES)
            img = IMAGES[img_idx]
            if img is not None:
                img_h, img_w = img.shape[:2]
                src_pts = np.array(
                    [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32
                )

                # --- projector overlay ---
                cam_corners = smoothed_corners[page_num]
                proj_corners = cv2.perspectiveTransform(
                    cam_corners.reshape(-1, 1, 2).astype(np.float32), H_PROJ
                ).reshape(4, 2)

                def shrink(pts, factor=0.85):
                    c = pts.mean(axis=0)
                    return (c + (pts - c) * factor).astype(np.float32)

                proj_corners = shrink(proj_corners)
                H_warp, _ = cv2.findHomography(src_pts, proj_corners)
                warped = cv2.warpPerspective(img, H_warp, (PROJ_W, PROJ_H))

                mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_f = mask_bin.astype(np.float32)[:, :, None] / 255.0
                proj_frame = (proj_frame * (1 - mask_f) + warped * mask_f).astype(
                    np.uint8
                )

        elif frame_count - last_seen > STALE_FRAMES:
            del last_seen_frame[page_num]
            del smoothed_corners[page_num]

    proj_rgb = cv2.cvtColor(proj_frame, cv2.COLOR_BGR2RGB)
    proj_surface = pygame.image.frombuffer(proj_rgb.tobytes(), (PROJ_W, PROJ_H), "RGB")
    proj_screen.blit(proj_surface, (0, 0))
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()

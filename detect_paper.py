"""Webcam → detect ArUco markers → identify papers → overlay images on pages."""
import cv2
import numpy as np
from collections import defaultdict

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

IMAGES = [
    cv2.imread('/home/fakhir/Code/PaperWorld/img1.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img2.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img3.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img4.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img5.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img6.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img7.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img8.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img9.jpg'),
    cv2.imread('/home/fakhir/Code/PaperWorld/img10.jpg'),
]

MARKERS_PER_PAGE = 8

MARKER_SIZE_CM = 2.5
PAGE_W_CM = 21.0
PAGE_H_CM = 29.7

EXPECTED_POSITIONS = {
    0: (0, PAGE_H_CM - MARKER_SIZE_CM),
    1: (PAGE_W_CM - MARKER_SIZE_CM, PAGE_H_CM - MARKER_SIZE_CM),
    2: (PAGE_W_CM - MARKER_SIZE_CM, 0),
    3: (0, 0),
    4: ((PAGE_W_CM - MARKER_SIZE_CM) / 2, PAGE_H_CM - MARKER_SIZE_CM),
    5: (PAGE_W_CM - MARKER_SIZE_CM, (PAGE_H_CM - MARKER_SIZE_CM) / 2),
    6: ((PAGE_W_CM - MARKER_SIZE_CM) / 2, 0),
    7: (0, (PAGE_H_CM - MARKER_SIZE_CM) / 2),
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("no webcam")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

cv2.namedWindow("paper detection  (q to quit)", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("paper detection  (q to quit)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

smoothed_corners = {}
SMOOTHING_ALPHA = 0.3
last_seen_frame = {}
STALE_FRAMES = 10
frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    frame_count += 1
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
                    page_corners_cam = np.array([
                        corner_centers[0],
                        corner_centers[1],
                        corner_centers[2],
                        corner_centers[3],
                    ], dtype=np.float32)
                    
                    if page_num in smoothed_corners:
                        old = smoothed_corners[page_num]
                        smoothed_c = (SMOOTHING_ALPHA * page_corners_cam + 
                                     (1 - SMOOTHING_ALPHA) * old).astype(np.float32)
                    else:
                        smoothed_c = page_corners_cam
                    
                    smoothed_corners[page_num] = smoothed_c
    
    for page_num, last_frame in list(last_seen_frame.items()):
        if frame_count - last_frame <= STALE_FRAMES and page_num in smoothed_corners:
            img_idx = page_num % len(IMAGES)
            if IMAGES[img_idx] is not None:
                img_h, img_w = IMAGES[img_idx].shape[:2]
                src_pts = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
                dst_pts = smoothed_corners[page_num]
                
                H, _ = cv2.findHomography(src_pts, dst_pts)
                warped = cv2.warpPerspective(IMAGES[img_idx], H, (frame.shape[1], frame.shape[0]))
                
                mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask = mask / 255.0
                
                frame = (frame * (1 - mask) + warped * mask).astype(np.uint8)
        elif frame_count - last_frame > STALE_FRAMES:
            del last_seen_frame[page_num]
            del smoothed_corners[page_num]
    
    cv2.imshow("paper detection  (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

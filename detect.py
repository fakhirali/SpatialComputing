"""Webcam → detect ArUco markers → show paper ID on each one."""
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

overlay_img = cv2.imread("pic.jpg")
if overlay_img is None:
    raise SystemExit("pic.jpg not found")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("no webcam")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
print(f"camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
      f"@ {int(cap.get(cv2.CAP_PROP_FPS))}fps  fourcc={int(cap.get(cv2.CAP_PROP_FOURCC)):08x}")

cv2.namedWindow("paper id  (q to quit)", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("paper id  (q to quit)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
last_warped = None
last_mask = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for corner, mid in zip(corners, ids.flatten()):
            x, y = corner[0][0].astype(int)
            cv2.putText(frame, f"Paper {mid}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ids_flat = ids.flatten()
        target_ids = [1, 11, 13, 15]
        if all(mid in ids_flat for mid in target_ids):
            centers = {}
            for mid in target_ids:
                idx = np.where(ids_flat == mid)[0][0]
                center = corners[idx][0].mean(axis=0).astype(int)
                centers[mid] = center
            
            dst_pts = np.array([centers[15], centers[13], centers[11], centers[1]], dtype=np.float32)
            h, w = overlay_img.shape[:2]
            src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(overlay_img, M, (frame.shape[1], frame.shape[0]))
            mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, M, (frame.shape[1], frame.shape[0]))
            last_warped = warped
            last_mask = mask
        
        if last_warped is not None:
            mask_inv = cv2.bitwise_not(last_mask)
            frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
            frame = cv2.add(frame, last_warped)
    
    cv2.imshow("paper id  (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

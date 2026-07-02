"""Webcam → two-hand image manipulation (q to quit)."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def create_overlay_image():
    img = np.zeros((200, 200, 4), dtype=np.uint8)
    img[:, :, :3] = [255, 100, 100]
    img[:, :, 3] = 200
    cv2.putText(img, "GRAB ME", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("no webcam")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
print(f"camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
      f"@ {int(cap.get(cv2.CAP_PROP_FPS))}fps")

WIN = "two-hand control  (q to quit)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.HandLandmarker.create_from_options(options)

overlay = create_overlay_image()
base_size = 200
scale = 1.0
rotation = 0.0
overlay_x, overlay_y = 400, 400
grabbing = False
prev_center_x, prev_center_y = 0, 0
prev_hand_dist = 0
prev_hand_angle = 0

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    h, w = frame.shape[:2]
    fists = []

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            wrist = points[0]
            finger_tips = [points[4], points[8], points[12], points[16], points[20]]
            avg_dist = np.mean([np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2) for tip in finger_tips])
            fist = avg_dist < 100

            if fist:
                fists.append((wrist, points))

            for i, j in HAND_CONNECTIONS:
                cv2.line(frame, points[i], points[j], (0, 255, 0), 2)
            for x, y in points:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    current_size = int(base_size * scale)
    img_left = overlay_x
    img_right = overlay_x + current_size
    img_top = overlay_y
    img_bottom = overlay_y + current_size

    def is_touching_boundary(point, margin=50):
        x, y = point
        near_left = abs(x - img_left) < margin and img_top <= y <= img_bottom
        near_right = abs(x - img_right) < margin and img_top <= y <= img_bottom
        near_top = abs(y - img_top) < margin and img_left <= x <= img_right
        near_bottom = abs(y - img_bottom) < margin and img_left <= x <= img_right
        return near_left or near_right or near_top or near_bottom

    touching_hands = []
    for wrist, points in fists:
        if is_touching_boundary(wrist):
            touching_hands.append((wrist, points))

    if len(touching_hands) == 2:
        (w1, _), (w2, _) = touching_hands
        center_x = (w1[0] + w2[0]) // 2
        center_y = (w1[1] + w2[1]) // 2
        hand_dist = np.sqrt((w1[0] - w2[0])**2 + (w1[1] - w2[1])**2)
        hand_angle = np.arctan2(w2[1] - w1[1], w2[0] - w1[0]) * 180 / np.pi

        if not grabbing:
            grabbing = True
            prev_center_x, prev_center_y = center_x, center_y
            prev_hand_dist = hand_dist
            prev_hand_angle = hand_angle
        else:
            dx = center_x - prev_center_x
            dy = center_y - prev_center_y
            overlay_x += dx
            overlay_y += dy

            dist_ratio = hand_dist / prev_hand_dist if prev_hand_dist > 0 else 1.0
            scale *= dist_ratio
            scale = max(0.3, min(3.0, scale))

            angle_delta = hand_angle - prev_hand_angle
            if angle_delta > 180:
                angle_delta -= 360
            elif angle_delta < -180:
                angle_delta += 360
            rotation += angle_delta

            prev_center_x, prev_center_y = center_x, center_y
            prev_hand_dist = hand_dist
            prev_hand_angle = hand_angle
    elif len(touching_hands) == 1:
        wrist, _ = touching_hands[0]
        if not grabbing:
            grabbing = True
            prev_center_x, prev_center_y = wrist
        else:
            dx = wrist[0] - prev_center_x
            dy = wrist[1] - prev_center_y
            overlay_x += dx
            overlay_y += dy
            prev_center_x, prev_center_y = wrist
    else:
        grabbing = False

    overlay_resized = cv2.resize(overlay, (current_size, current_size))

    center_x, center_y = overlay_x + current_size // 2, overlay_y + current_size // 2
    M = cv2.getRotationMatrix2D((center_x - overlay_x, center_y - overlay_y), rotation, 1.0)
    overlay_rotated = cv2.warpAffine(overlay_resized, M, (current_size, current_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    y1, y2 = overlay_y, overlay_y + current_size
    x1, x2 = overlay_x, overlay_x + current_size

    if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
        roi = frame[y1:y2, x1:x2]
        overlay_rgb = overlay_rotated[:, :, :3]
        alpha = overlay_rotated[:, :, 3:4] / 255.0
        blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended

    cv2.imshow(WIN, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()

"""Projector calibration: click 4 projected dots; sub-pixel centroid snap."""
import cv2
import numpy as np
import pygame
import json
import os
import sys

CALIBRATION_FILE = "calibration.json"
CAMERA_INDEX = 0
CAMERA_W = 3840
CAMERA_H = 2160

DOT_RADIUS_PX = 6
SNAP_RADIUS = 40
SNAP_BRIGHT_THRESH = 200
ZOOM_BOX = 80
ZOOM_SCALE = 6

def snap_to_bright_centroid(gray, x, y):
    h, w = gray.shape
    r = SNAP_RADIUS
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r), min(h, int(y) + r)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float(x), float(y)
    patch = gray[y0:y1, x0:x1]
    mask = patch >= SNAP_BRIGHT_THRESH
    if not mask.any():
        return float(x), float(y)
    ys, xs = np.where(mask)
    weights = patch[ys, xs].astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return float(x), float(y)
    cx = xs @ weights / total + x0
    cy = ys @ weights / total + y0
    return float(cx), float(cy)

def calibrate_projector(proj_w, proj_h, display_idx=0):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("no webcam")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

    pygame.init()
    screen = pygame.display.set_mode((proj_w, proj_h), pygame.FULLSCREEN, display=display_idx)
    pygame.display.set_caption("projector calibration")
    font = pygame.font.SysFont("monospace", 20)
    label_font = pygame.font.SysFont("monospace", 16)

    margin = 60
    proj_pts = np.array([
        [margin, margin],
        [proj_w - margin, margin],
        [proj_w - margin, proj_h - margin],
        [margin, proj_h - margin],
    ], dtype=np.float32)

    cam_pts = []
    snap_offsets = []
    current_idx = 0
    H = None
    last_gray = None
    last_click = None

    def mouse_handler(event, x, y, flags, param):
        nonlocal current_idx, last_click
        if event == cv2.EVENT_LBUTTONDOWN and current_idx < len(proj_pts):
            last_click = (x, y)
            if last_gray is not None:
                sx, sy = snap_to_bright_centroid(last_gray, x, y)
                dx, dy = sx - x, sy - y
                cam_pts.append(np.array([sx, sy], dtype=np.float32))
                snap_offsets.append((dx, dy))
                print(f"corner {current_idx}: click=({x},{y}) snapped=({sx:.2f},{sy:.2f}) Δ=({dx:+.2f},{dy:+.2f})")
            else:
                cam_pts.append(np.array([x, y], dtype=np.float32))
                snap_offsets.append((0.0, 0.0))
            current_idx += 1
            if current_idx >= len(proj_pts):
                compute_homography()

    def compute_homography():
        nonlocal H
        cp = np.array(cam_pts, dtype=np.float32)
        H, status = cv2.findHomography(cp, proj_pts)
        print(f"homography computed: {status.ravel().sum()}/{len(status)} inliers")

    cv2.namedWindow("calibration (click 4 corners, ENTER=done, R=restart, esc=abort)")
    cv2.setMouseCallback("calibration (click 4 corners, ENTER=done, R=restart, esc=abort)", mouse_handler)

    print("Click the 4 white dots in order: top-left, top-right, bottom-right, bottom-left")
    print(f"Snap: bright-centroid in {SNAP_RADIUS*2}x{SNAP_RADIUS*2}px window around click.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        screen.fill((0, 0, 0))
        for i, (px, py) in enumerate(proj_pts):
            color = (0, 255, 0) if i < current_idx else (255, 255, 255)
            pygame.draw.circle(screen, color, (int(px), int(py)), DOT_RADIUS_PX)
            label = label_font.render(str(i), True, (255, 255, 0))
            screen.blit(label, (int(px) + 10, int(py) - 8))
        if current_idx >= len(proj_pts):
            msg = font.render("calibrated! ENTER=accept, R=restart", True, (0, 255, 0))
            screen.blit(msg, (10, proj_h - 40))
        pygame.display.flip()

        display = frame.copy()
        for i, pt in enumerate(cam_pts):
            cv2.drawMarker(display, (int(round(pt[0])), int(round(pt[1]))),
                           (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
            cv2.putText(display, str(i), (int(pt[0]) + 10, int(pt[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if last_click is not None and current_idx > 0 and current_idx <= len(cam_pts):
            i = current_idx - 1
            cx, cy = cam_pts[i]
            x0, y0 = max(0, int(cx) - ZOOM_BOX // 2), max(0, int(cy) - ZOOM_BOX // 2)
            x1, y1 = x0 + ZOOM_BOX, y0 + ZOOM_BOX
            if x1 <= display.shape[1] and y1 <= display.shape[0]:
                patch = last_gray[y0:y1, x0:x1]
                zoom = cv2.resize(patch, (ZOOM_BOX * ZOOM_SCALE, ZOOM_BOX * ZOOM_SCALE),
                                  interpolation=cv2.INTER_NEAREST)
                zoom = cv2.cvtColor(zoom, cv2.COLOR_GRAY2BGR)
                lx, ly = int(cx) - x0, int(cy) - y0
                lx, ly = lx * ZOOM_SCALE, ly * ZOOM_SCALE
                cv2.line(zoom, (lx - 30, ly), (lx + 30, ly), (0, 0, 255), 1)
                cv2.line(zoom, (lx, ly - 30), (lx, ly + 30), (0, 0, 255), 1)
                cv2.rectangle(zoom, (0, 0), (zoom.shape[1] - 1, zoom.shape[0] - 1), (0, 255, 0), 2)
                zx, zy = 20, 20
                display[zy:zy + zoom.shape[0], zx:zx + zoom.shape[1]] = zoom
                dx, dy = snap_offsets[i]
                info = f"corner {i}: snap dx={dx:+.2f} dy={dy:+.2f}"
                cv2.putText(display, info, (zx, zy + zoom.shape[0] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if current_idx < len(proj_pts):
            cv2.putText(display, f"click corner {current_idx}/{len(proj_pts)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "calibrated!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("calibration (click 4 corners, ENTER=done, R=restart, esc=abort)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release()
            pygame.quit()
            cv2.destroyAllWindows()
            return None
        elif key == 13 and H is not None:
            break
        elif key == ord("r"):
            cam_pts.clear()
            snap_offsets.clear()
            current_idx = 0
            H = None
            last_click = None

    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()
    return H.tolist()

def main():
    proj_w = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
    proj_h = int(sys.argv[2]) if len(sys.argv) > 2 else 1080
    display_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        if "projector_homography" in cal:
            ans = input(f"calibration found in {CALIBRATION_FILE}. recalibrate? [y/N] ").strip().lower()
            if ans != "y":
                print("keeping existing calibration")
                return
            print("recalibrating")
        else:
            print(f"{CALIBRATION_FILE} missing projector_homography, recalibrating")

    print("=== PROJECTOR CALIBRATION ===")
    H = calibrate_projector(proj_w, proj_h, display_idx)
    if H is None:
        print("calibration aborted")
        return

    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"projector_homography": H}, f)
    print(f"saved to {CALIBRATION_FILE}")

if __name__ == "__main__":
    main()

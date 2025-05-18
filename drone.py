import argparse
import imutils
import cv2
import sys
import time
import os
from collections import deque

class CentroidTracker:
    def __init__(self, max_disappeared=40):
        self.next_object_id = 0
        self.objects = {}  # object_id: centroid
        self.disappeared = {}  # object_id: # frames disappeared
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = cv2.norm(cv2.UMat(input_centroids), cv2.UMat(object_centroids), cv2.NORM_L2)

            for i, centroid in enumerate(input_centroids):
                self.objects[object_ids[i]] = centroid
                self.disappeared[object_ids[i]] = 0

        return self.objects

def is_target(contour, approx, min_dim=25, aspect_min=0.8, aspect_max=1.2, solidity_thresh=0.9):
    if len(approx) < 4 or len(approx) > 6:
        return False

    (x, y, w, h) = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))

    if hull_area == 0:
        return False

    solidity = area / hull_area

    return (
        w > min_dim and h > min_dim and
        aspect_min <= aspect_ratio <= aspect_max and
        solidity >= solidity_thresh
    )

def draw_target(frame, approx, object_id=None):
    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    M = cv2.moments(approx)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        (x, y, w, h) = cv2.boundingRect(approx)
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)

        cv2.line(frame, (cX - pad_x, cY), (cX + pad_x, cY), (0, 255, 0), 2)
        cv2.line(frame, (cX, cY - pad_y), (cX, cY + pad_y), (0, 255, 0), 2)

        if object_id is not None:
            cv2.putText(frame, f"ID {object_id}", (cX + 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return (x, y, w, h)

    return None

def save_cropped_target(frame, x, y, w, h, count):
    target = frame[y:y+h, x:x+w]
    filename = f"detected_target_{count}.png"
    cv2.imwrite(filename, target)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="Path to input video file")
    args = vars(ap.parse_args())

    if not os.path.exists("output"):
        os.makedirs("output")
    os.chdir("output")

    cap = cv2.VideoCapture(args["video"])
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args['video']}")
        sys.exit(1)

    print("[INFO] Processing video... Press 'q' to quit.")
    fps_start = time.time()
    frame_count = 0
    save_count = 0

    tracker = CentroidTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 150)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        status = "No Targets"
        centroids = []
        boxes = []

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)

            if is_target(c, approx):
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))
                    boxes.append((approx, c))

        objects = tracker.update(centroids)

        for object_id, centroid in objects.items():
            for (approx, c) in boxes:
                (x, y, w, h) = cv2.boundingRect(approx)
                if abs(centroid[0] - (x + w // 2)) < 20 and abs(centroid[1] - (y + h // 2)) < 20:
                    draw_target(frame, approx, object_id)
                    save_cropped_target(frame, x, y, w, h, save_count)
                    save_count += 1
                    status = "Target(s) Acquired"
                    break

        # Display status and FPS
        frame_count += 1
        fps = frame_count / (time.time() - fps_start)
        cv2.putText(frame, f"{status} | FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Target Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Process completed. {save_count} targets saved.")

if __name__ == "__main__":
    main()

import argparse
import imutils
import cv2
import sys
import time
import os
from collections import deque
import numpy as np
from scipy.spatial import distance as dist

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
        # If no centroids, mark all existing objects as disappeared
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # If we have no objects being tracked, register all input centroids
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Otherwise, try to match input centroids to existing objects
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distances between each pair of object and input centroids
            D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
            
            # Find the smallest value in each row, then sort by row index
            rows = D.min(axis=1).argsort()
            
            # Find the smallest value in each column, then sort by col index
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Loop over combinations of rows and columns
            for (row, col) in zip(rows, cols):
                # If we've already used this row or column, skip
                if row in used_rows or col in used_cols:
                    continue
                
                # Otherwise, update the object with the new centroid
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Mark row and column as used
                used_rows.add(row)
                used_cols.add(col)
            
            # Find unused rows and columns
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            
            # If we have more objects than centroids, check if any disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            # Otherwise, register new centroids
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
                    
        return self.objects

def is_target(contour, approx, min_dim=15, aspect_min=0.5, aspect_max=2.0, solidity_thresh=0.65, min_area=150):
    # More relaxed criteria for target detection, optimized for drone footage
    # For drone footage targets are often small and might have fewer vertices due to distance
    if len(approx) < 3 or len(approx) > 12:  # Even more relaxed vertex count
        return False

    (x, y, w, h) = cv2.boundingRect(approx)
    
    # If the contour is too small, ignore it
    if w < min_dim or h < min_dim:
        return False
        
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    
    # Filter by area (important for drone footage)
    if area < min_area:
        return False
        
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # Prevent division by zero
    if hull_area == 0:
        return False

    solidity = float(area) / hull_area
    
    # For debugging
    # print(f"Contour: vertices={len(approx)}, aspect={aspect_ratio:.2f}, solidity={solidity:.2f}, dims={w}x{h}, area={area}")

    return (
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
    print(f"[INFO] Saved target image: {filename}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="Path to input video file")
    ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    ap.add_argument("-y", "--youtube", action="store_true", help="Video source is a YouTube URL")
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

    tracker = CentroidTracker(max_disappeared=20) 
    
    # Create debug windows if needed
    if args["debug"]:
        cv2.namedWindow("Edge Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Filter", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        
        # Convert to grayscale for standard processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use HSV color space for better detection outdoors where lighting varies
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create multiple detection methods for different target types
        
        # 1. Standard edge detection 
        edged = cv2.Canny(blurred, 30, 130)
        
        # 2. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. HSV color filtering (useful for detecting specific colored targets)
        lower_bound = np.array([0, 0, 100])  
        upper_bound = np.array([180, 40, 255])  
        hsv_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 4. Motion detection via frame differencing could be added here
        
        # Clean up binary images
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
        
        if args["debug"]:
            cv2.imshow("Edge Detection", edged)
            cv2.imshow("Thresholded", thresh)
            cv2.imshow("HSV Filter", hsv_mask)

        # Get contours from multiple processing methods
        cnts_edge = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_edge = imutils.grab_contours(cnts_edge)
        
        cnts_thresh = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_thresh = imutils.grab_contours(cnts_thresh)
        
        cnts_hsv = cv2.findContours(hsv_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_hsv = imutils.grab_contours(cnts_hsv)
        
        # Combine all contours from different methods
        all_cnts = cnts_edge + cnts_thresh + cnts_hsv

        status = "No Targets"
        centroids = []
        boxes = []

        # Target detection parameters adjusted for drone footage
        min_dim = 15  
        aspect_min = 0.4 
        aspect_max = 2.5
        solidity_thresh = 0.65 
        min_area = 150 

        for c in all_cnts:
            # Filter by minimum area for efficiency
            area = cv2.contourArea(c)
            if area < min_area:
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True) 

            if is_target(c, approx, min_dim, aspect_min, aspect_max, solidity_thresh, min_area):
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))
                    boxes.append((approx, c))
                    
                    if args["debug"]:
                        # Draw all potential targets in debug mode (blue)
                        cv2.drawContours(frame, [approx], -1, (255, 0, 0), 1)

        # Filter targets that are too close together (likely duplicates)
        filtered_centroids = []
        filtered_boxes = []
        
        for i, (centroid, box) in enumerate(zip(centroids, boxes)):
            is_duplicate = False
            for fc in filtered_centroids:
                # If centroids are very close, treat as duplicate
                if np.sqrt((centroid[0] - fc[0])**2 + (centroid[1] - fc[1])**2) < 20:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_centroids.append(centroid)
                filtered_boxes.append(box)
        
        # Update with filtered centroids
        objects = tracker.update(filtered_centroids)

        # Draw tracked targets
        for (i, ((approx, c), centroid)) in enumerate(zip(filtered_boxes, filtered_centroids)):
            status = "Target(s) Acquired"
            (x, y, w, h) = cv2.boundingRect(approx)
            
            # Find the ID for this centroid
            object_id = None
            for ID, cent in objects.items():
                if np.sqrt((cent[0] - centroid[0])**2 + (cent[1] - centroid[1])**2) < 10:
                    object_id = ID
                    break
                    
            # Draw the target with its ID
            box_info = draw_target(frame, approx, object_id)
            if box_info is not None:
                x, y, w, h = box_info
                save_cropped_target(frame, x, y, w, h, save_count)
                save_count += 1

        # Display status and FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"{status} | FPS: {fps:.2f}", (20, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Target Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):  
            snapshot_name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snapshot_name, frame)
            print(f"[INFO] Saved snapshot: {snapshot_name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Process completed. {save_count} targets saved.")

if __name__ == "__main__":
    main()

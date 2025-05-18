# TargetLock-Real-Time-Target-Detection-Tracking-in-Drone-Feeds

# TargetLock: Real-Time Target Detection & Tracking in Drone Feeds

## Project Description

**TargetLock** is a Python-based computer vision application designed to perform real-time detection and tracking of targets from drone video feeds. Leveraging OpenCV's powerful image processing capabilities, the system identifies polygon-shaped objects in each video frame, tracks their positions using a centroid-based tracking algorithm, and saves cropped images of detected targets for further analysis.

This project aims to support applications in surveillance, search and rescue, wildlife monitoring, and autonomous drone navigation by providing robust and efficient target tracking in live video streams.

## Data Types and Input

- **Input Data**: Video files or live video feeds captured from drones.
- **Frame Processing**: Each frame is resized, converted to grayscale, blurred, and processed to detect edges.
- **Contours**: Polygonal contours extracted from edges are analyzed based on shape, size, solidity, and aspect ratio to identify targets.
- **Centroids**: The geometric centers of detected targets are tracked across frames to maintain object identity.
- **Output Data**: Cropped images of detected targets saved as PNG files, along with real-time visualization of tracking results.

## Project Details

1. **Preprocessing**  
   - Input video frames are resized to a fixed width for consistent processing.  
   - Grayscale conversion and Gaussian blur reduce noise.  
   - Canny edge detection highlights object boundaries.

2. **Target Detection**  
   - Contours are extracted from the edge map.  
   - Polygonal approximation filters contours with 4 to 6 vertices to focus on relevant shapes.  
   - Solidity (contour area / convex hull area) and aspect ratio thresholds ensure targets are well-defined and approximately regular.

3. **Tracking Algorithm**  
   - A centroid-based tracker assigns IDs to detected targets and maintains their identities frame-to-frame.  
   - Disappearance handling ensures that temporarily occluded or missing targets are deregistered after a configurable timeout.

4. **Output and Visualization**  
   - Detected targets are highlighted with contours and crosshairs in the displayed video.  
   - Each detected target’s cropped image is saved for offline analysis.  
   - Frame rate and detection status are shown on-screen.

5. **Use Cases and Applications**  
   - Real-time tracking in drone surveillance for identifying objects of interest.  
   - Monitoring wildlife or assets in remote areas.  
   - Integration with autonomous navigation systems requiring target awareness.

## How to Run

```bash
pip install opencv-python imutils numpy
python targetlock.py --video path_to_video_file.mp4
Press 'q' to quit the video window at any time
```

This project is licensed under the MIT License.

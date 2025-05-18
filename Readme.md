# TargetLock: Real-Time Target Detection & Tracking in Drone Feeds

TargetLock is a powerful computer vision system designed to detect, track, and log objects of interest in aerial drone footage. Using advanced computer vision techniques, it provides real-time identification and persistent tracking of targets across video frames.

## Features

- **Multi-Method Detection**: Combines edge detection, adaptive thresholding, and HSV color filtering for robust target identification in varying environments
- **Persistent Tracking**: Assigns unique IDs to detected targets and tracks them across frames
- **Automatic Target Extraction**: Saves cropped images of detected targets for further analysis
- **Real-time Visual Feedback**: Displays detection status, bounding boxes, and tracking IDs
- **Debug Mode**: Visualize the detection process with multiple visual filters
- **Duplicate Filtering**: Smart algorithms to prevent multiple detections of the same target

## Requirements

- Python 3.6+
- OpenCV 4.2+
- NumPy
- SciPy
- imutils

Install dependencies with:
```bash
pip install opencv-python numpy scipy imutils
```

## Usage

### Basic Usage
```bash
python targetlock.py -v path/to/video.mp4
```

### Enable Debug Mode
```bash
python targetlock.py -v path/to/video.mp4 -d
```

## How It Works

TargetLock employs a multi-stage detection and tracking pipeline:

1. **Pre-processing**: Resizes frames and applies various image processing techniques
2. **Multi-method detection**: 
   - Edge detection using Canny algorithm
   - Adaptive thresholding for binary segmentation
   - HSV color space filtering for specific target colors
3. **Contour analysis**: Examines shapes, sizes, and geometric properties
4. **Target validation**: Filters potential targets based on configurable criteria:
   - Size constraints
   - Aspect ratio
   - Shape complexity (vertex count)
   - Solidity (area vs. convex hull area)
5. **Centroid tracking**: Assigns and maintains unique IDs for detected targets
6. **Visualization**: Renders detection status and tracking information on video frames
7. **Target extraction**: Saves cropped images of confirmed targets

## Configuration

The system can be tuned through several parameters in the code:

- **Target Detection Parameters**: 
  - `min_dim`: Minimum width/height of targets (default: 15px)
  - `aspect_min`/`aspect_max`: Acceptable aspect ratio range (default: 0.4-2.5)
  - `solidity_thresh`: Minimum solidity value (default: 0.65)
  - `min_area`: Minimum contour area (default: 150px²)
  
- **Tracking Parameters**:
  - `max_disappeared`: Maximum frames a target can disappear before its ID is deregistered (default: 20)

- **HSV Color Filtering**:
  - `lower_bound`/`upper_bound`: HSV color range for filtering specific targets

## Output

Detected targets are saved in the `output` directory as PNG images, named with sequential IDs.

## Tips for Best Results

- **Lighting**: Consistent lighting conditions produce more reliable detections
- **Camera Movement**: Smoother drone movement reduces false positives
- **Target Selection**: For specific target types, adjust HSV bounds accordingly:
  - For white/light targets: Keep saturation low, value high
  - For colored targets: Adjust hue range appropriately

## Troubleshooting

- **No targets detected**: Try enabling debug mode (`-d`) to visualize detection steps
- **False positives**: Adjust target validation parameters to be more restrictive
- **Target IDs changing**: Lower the `max_disappeared` value for faster ID reassignment
- **Performance issues**: Reduce frame size for faster processing

## Future Improvements

- Machine learning-based target detection
- Support for additional video sources (RTSP, webcams)
- Target classification and automatic object recognition
- Path prediction for occluded targets
- Web interface for remote monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*TargetLock is designed for educational and research purposes only.*

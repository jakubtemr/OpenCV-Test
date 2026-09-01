# OpenCV Filter Network Defect Detection

Computer vision experiment that detects defects (holes) in a filter mesh from a photo using classic OpenCV image processing — no machine learning involved.

## How it works

`vision.py` runs a single pipeline over one input image:

1. **Preprocessing** — grayscale conversion, Gaussian blur (5×5), binary threshold at 120.
2. **Morphology** — dilation (4 iterations) followed by erosion (2 iterations) with a 5×5 kernel to close noise inside the mesh.
3. **Mesh detection** — external contours approximated with `approxPolyDP`; contours with 4 vertices and an area above 100 px are treated as cells of the filter network.
4. **Defect detection** — every external contour that overlaps the mesh mask and is larger than 10 000 px is reported as a hole.
5. **Output** — bounding boxes are drawn on a copy of the input (green = mesh cell, red = defect) and saved to `data/out.jpg`.

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

The input path is hardcoded at the top of `vision.py`:

```python
image = cv2.imread('data/2.jpg')
```

Change it to another sample and run:

```bash
python vision.py
```

The annotated result is written to `data/out.jpg`.

## Sample data

`data/1.jpg` – `data/5.jpg` are test photos of filter meshes; `data/out.jpg` is the last generated result.

## Limitations

Thresholds (binary threshold 120, minimum cell area 100 px, minimum defect area 10 000 px) are tuned for the resolution and lighting of the sample images. Photos taken under different conditions will need those constants re-tuned.

## License

MIT — see [LICENSE](LICENSE).

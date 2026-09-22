# Digital Image Processing: Core Concepts

This repository demonstrates four fundamental concepts of Digital Image Processing (DIP) using Python, OpenCV, and Matplotlib.

## Features Implemented

1. **Matrix Representation**:
   - Inspects 2D (grayscale) and 3D (RGB) matrix shapes.
   - Extracts minimum and maximum intensity values across overall and per-channel matrices.
   - Displays sample intensity sub-matrices.

2. **Sampling (Spatial Resolution)**:
   - Downsamples image spatial dimensions using nearest-neighbor interpolation to demonstrate pixelation artifacts.

3. **Quantization (Bit Depth Reduction)**:
   - Reduces pixel intensity levels from standard 8-bit depth to 4-bit and 2-bit representations to show false contouring.

4. **Resolution Scaling**:
   - Rescales overall image resolution (50% and 25%) using area interpolation.

## Prerequisites

Install the required Python packages:

```bash
py -m pip install opencv-python numpy matplotlib

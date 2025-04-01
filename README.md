Below is a structured English README for the provided Python script:

---

# Image Enhancement Pipeline

This repository contains a Python script (`main.py`) that implements an advanced image enhancement pipeline. The pipeline integrates multiple image processing techniques such as dehazing, noise reduction, morphological operations, and pyramid-based multi-scale processing to improve the quality of grayscale ultrasound images.

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Features](#features)
4. [Usage](#usage)
5. [Functions](#functions)
6. [Contributing](#contributing)

---

### Overview

The `main.py` script processes input grayscale images through a series of steps designed to enhance their visual quality. It includes methods for:

- **Dehazing**: Removes atmospheric scattering effects using a dark channel prior.
- **Noise Reduction**: Applies consistency diffusion filtering to reduce noise while preserving edges.
- **Morphological Operations**: Implements closing operations to fill gaps in the image.
- **Pyramid Processing**: Decomposes the image into multiple scales, processes each layer individually, and reconstructs the final result.
- **Contrast Enhancement**: Uses adaptive histogram equalization to improve contrast.

The pipeline is modular, allowing users to customize or extend individual components.

---

### Dependencies

To run this script, ensure the following Python packages are installed:

- `numpy`: For numerical computations.
- `matplotlib`: For visualization.
- `scipy`: For signal processing and convolution operations.
- `skimage`: For advanced image processing functions (e.g., adaptive histogram equalization, morphological filters).

Install dependencies via pip:
```bash
pip install numpy matplotlib scipy scikit-image
```


---

### Features

- **Dark Channel Dehazing**: Effectively removes haze from images using a modified dark channel prior.
- **Consistency Diffusion Filtering**: Reduces noise while maintaining structural details.
- **Gaussian Pyramid Decomposition**: Breaks down the image into multiple resolutions for hierarchical processing.
- **Layer-wise Processing**: Each pyramid layer undergoes tailored enhancements (e.g., dehazing, diffusion filtering).
- **Reconstruction**: Combines processed layers using weighted fusion for a coherent final output.
- **Adaptive Histogram Equalization**: Enhances contrast globally and locally.

---

### Usage

1. Place your input image in the `../images/` directory (e.g., `10.png`).
2. Run the script:
   ```bash
   python main.py
   ```

3. The script will display intermediate results and the final enhanced image in a graphical window.

---

### Functions

#### 1. Auxiliary Functions
- **validate_image(img)**: Ensures the input image is a valid grayscale image.
- **safe_uint8_conversion(img)**: Converts images to `uint8` format safely.

#### 2. Core Algorithms
- **guided_filter(I, p, radius=10, eps=1e-3)**: Implements guided filtering for smoothing and edge-preserving.
- **morphological_close(img, kernel_size=15)**: Performs morphological closing to remove small holes.
- **dehaze_ultrasound(img, omega=0.3, t0=0.2, kernel_size=21)**: Removes haze using a dark channel prior.
- **consistency_diffusion(img, iterations=15, k=40, lambda_=0.15)**: Reduces noise with consistency diffusion filtering.
- **black_hole_filling(img, ta=15, t1=15)**: Fills low-intensity regions to prevent black holes.
- **build_gaussian_pyramid(img, levels=4, cached_kernel=None)**: Constructs a Gaussian pyramid for multi-scale processing.
- **reconstruct_pyramid(pyramid, cached_kernel=None)**: Reconstructs the image from the processed pyramid layers.

#### 3. Main Workflow
The script follows these steps:
1. Reads and validates the input image.
2. Applies dehazing, noise reduction, and morphological operations.
3. Decomposes the image into a Gaussian pyramid.
4. Processes each pyramid layer individually.
5. Reconstructs the image from the processed layers.
6. Enhances the final result with adaptive histogram equalization.
7. Displays intermediate and final results.

---

### Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Make changes or improvements.
3. Submit a pull request with detailed explanations.

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Example Output

The script generates the following outputs:
- Original image.
- Dehazed image.
- Denoised image.
- Filled image (after black hole filling).
- Reconstructed image (after pyramid processing).
- Final enhanced image.

These outputs are displayed in a single figure for easy comparison.

---

This pipeline serves as a foundation for advanced image enhancement tasks, particularly suited for medical imaging applications like ultrasound. Feel free to modify and extend it for specific use cases!

# CUDA Box Filter

A high-performance image processing application that demonstrates the significant speedup achievable by implementing box filter algorithms on GPU using CUDA compared to traditional CPU processing.

## What It Does

This project applies a **box filter** (averaging filter) to high-resolution images, comparing the performance between:
- **CPU Implementation**: OpenCV-based sequential processing
- **GPU Implementation**: Custom CUDA kernel with parallel processing

The box filter smooths images by replacing each pixel with the average of its surrounding neighborhood, effectively reducing noise and creating a blur effect.

## How It Works

### Box Filter Algorithm
1. **Kernel Size**: Uses a 5×5 pixel neighborhood (configurable)
2. **Process**: Each pixel value = average of 25 surrounding pixels
3. **Effect**: Smoothing/blurring with noise reduction

### CPU vs GPU Implementation
- **CPU**: Sequential pixel-by-pixel processing using OpenCV
- **GPU**: Massively parallel processing with CUDA threads
  - Each thread processes one pixel
  - 16×16 thread blocks for optimal memory access
  - Shared memory optimization for kernel data

### Technical Details
- **Language**: C++/CUDA
- **Libraries**: OpenCV 4.8.0, CUDA Runtime
- **Compiler**: NVCC with Visual Studio 2022
- **Processing**: Both color (RGB) and grayscale modes

## Performance Results

### Color Images (3 Channels)

| Image | Dimensions | CPU Time (ms) | GPU Time (ms) | Speedup |
|-------|------------|---------------|---------------|---------|
| 200MP_sample.jpg | 12240×16320 | 499.45 | 28.74 | **17.38×** |
| car_window.jpg | 5204×9248 | 121.83 | 7.35 | **16.58×** |
| drone_shot.jpg | 3000×4000 | 30.54 | 1.80 | **16.97×** |

### Grayscale Images (1 Channel)

| Image | Dimensions | CPU Time (ms) | GPU Time (ms) | Speedup |
|-------|------------|---------------|---------------|---------|
| 200MP_sample.jpg | 12240×16320 | 168.99 | 10.57 | **15.99×** |
| car_window.jpg | 5204×9248 | 40.63 | 3.04 | **13.36×** |
| drone_shot.jpg | 3000×4000 | 10.93 | 0.61 | **18.03×** |

### Key Insights
- **Average Speedup**: ~16× faster on GPU
- **Best Performance**: Largest images show highest speedup
- **Efficiency**: GPU excels at parallel pixel processing
- **Scalability**: Performance advantage increases with image size

## Quick Start

### Prerequisites
- CUDA Toolkit (12.3+)
- Visual Studio 2022
- Windows 10/11

### Build & Run
```bash
# Compile (downloads OpenCV automatically)
compile_cuda.bat

# Run the program
cd build
box_filter.exe
```

### Output
- **Results**: `results/` directory
- **Files**: `box_cpu_color.jpg`, `box_cuda_color.jpg`, etc.
- **Performance**: Console output with timing comparisons

## Project Structure
```
box-filter-cuda/
├── code/
│   └── box_filter.cu          # Main CUDA implementation
├── assets/                    # Test images
├── results/                   # Output images & performance data
├── compile_cuda.bat          # Build script
└── README.md                 # This file
```

## Algorithm Details

The CUDA kernel implements:
- **Boundary handling**: Clamp-to-edge for image borders
- **Memory coalescing**: Optimized memory access patterns
- **Thread organization**: 2D grid matching image dimensions
- **Kernel normalization**: 1/(kernelSize²) weighting

Perfect for demonstrating GPU computing advantages in image processing applications.

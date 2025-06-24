#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

cv::Mat applyBoxFilterCPU(const cv::Mat& image, int kernelSize) {
    std::cout << "Primjena Box filtera na CPU-u..." << std::endl;
    cv::Mat outputImage;
    cv::boxFilter(image, outputImage, -1, cv::Size(kernelSize, kernelSize));
    return outputImage;
}

std::vector<float> createBoxKernel(int kernelSize) {
    std::vector<float> kernel(kernelSize * kernelSize);
    float kernelValue = 1.0f / (kernelSize * kernelSize);

    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        kernel[i] = kernelValue;
    }
    return kernel;
}

__global__ void boxFilterKernel(const unsigned char* inputImage,
    unsigned char* outputImage,
    int rows, int cols, int channels,
    const float* kernel, int kernelSize) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = kernelSize / 2;

    if (row < rows && col < cols) {
        // Process each channel
        for (int ch = 0; ch < channels; ++ch) {
            float pixelSum = 0.0f;

            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int imgY = row + (ky - offset);
                    int imgX = col + (kx - offset);

                    // Handle border by clamping
                    imgY = max(0, min(rows - 1, imgY));
                    imgX = max(0, min(cols - 1, imgX));

                    pixelSum += static_cast<float>(
                        inputImage[(imgY * cols + imgX) * channels + ch]) *
                        kernel[ky * kernelSize + kx];
                }
            }

            unsigned char finalPixelValue = static_cast<unsigned char>(
                fminf(255.0f, fmaxf(0.0f, pixelSum)));

            outputImage[(row * cols + col) * channels + ch] = finalPixelValue;
        }
    }
}

void processImage(const cv::Mat& image, const std::string& imageName, int kernelSize) {
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();
    size_t imageSizeInBytes = rows * cols * channels;

    std::cout << "\n--- " << imageName << " CPU Box Filter ---\n";
    std::cout << "Dimenzije: " << rows << "x" << cols << " kanala: " << channels << std::endl;
    
    auto startCpu = std::chrono::high_resolution_clock::now();
    cv::Mat cpuResult = applyBoxFilterCPU(image, kernelSize);
    auto endCpu = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endCpu - startCpu).count();
    std::cout << "CPU vrijeme: " << cpuTime / 1000.0 << " ms\n";
    
    std::string cpuOutputPath = "../results/box_cpu_" + imageName + ".jpg";
    cv::imwrite(cpuOutputPath, cpuResult);

    std::cout << "\n--- " << imageName << " CUDA Box Filter ---\n";
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_kernel = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, imageSizeInBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imageSizeInBytes));

    CUDA_CHECK(cudaMemcpy(d_input, image.data, imageSizeInBytes, cudaMemcpyHostToDevice));

    std::vector<float> h_kernel = createBoxKernel(kernelSize);
    size_t kernelBytes = h_kernel.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_kernel, kernelBytes));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernelBytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    cudaEvent_t startEvent, endEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&endEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

    boxFilterKernel<<<blocks, threads>>>(d_input, d_output, rows, cols, channels, d_kernel, kernelSize);

    CUDA_CHECK(cudaEventRecord(endEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(endEvent));

    float cudaTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cudaTime, startEvent, endEvent));
    std::cout << "CUDA vrijeme: " << cudaTime << " ms\n";

    cv::Mat gpuResult(rows, cols, channels == 1 ? CV_8UC1 : CV_8UC3);
    CUDA_CHECK(cudaMemcpy(gpuResult.data, d_output, imageSizeInBytes, cudaMemcpyDeviceToHost));
    
    std::string cudaOutputPath = "../results/box_cuda_" + imageName + ".jpg";
    cv::imwrite(cudaOutputPath, gpuResult);

    std::cout << "\n--- " << imageName << " Usporedba ---\n";
    std::cout << "CPU: " << cpuTime / 1000.0 << " ms\n";
    std::cout << "GPU: " << cudaTime << " ms\n";
    std::cout << "Ubrzanje: " << (cpuTime / 1000.0) / cudaTime << "x\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(endEvent));
}

int main() {
    std::string imagePath = "../assets/drone_shot.jpg";
    std::cout << "\nUcitavanje slike: " << imagePath << std::endl;
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Greska: Slika se ne moze ucitati.\n";
        return EXIT_FAILURE;
    }

    // Kernel size - 5x5 
    int kernelSize = 5;

    // Process color version
    std::cout << "\n========== OBRADA SLIKE U BOJI ==========\n";
    processImage(image, "color", kernelSize);

    // Process grayscale version
    std::cout << "\n========== OBRADA SLIKE U SIVIM TONOVIMA ==========\n";
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    processImage(grayImage, "grayscale", kernelSize);

    return 0;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unsharpMaskKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, float strength)
{
	// Pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
        int pixelIndex = y * width + x;

        // Apply the unsharp mask filter
        float blurredPixel = 0.f;
        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                int neighborX = x + i;
                int neighborY = y + j;

                // Check for neighbor coords in image bounds
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height)
                {
                    int neighborIndex = neighborY * width + neighborX;
                    blurredPixel += inputImage[neighborIndex];
                }
            }
        }

        blurredPixel /= 9.f;
        float sharpenedPixel = inputImage[pixelIndex] + strength * (inputImage[pixelIndex] - blurredPixel);

        // Clamp to [0, 255]
        outputImage[pixelIndex] = std::max(std::min((int)sharpenedPixel, 255), 0);
	}
}

void unsharpMaskCUDA(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, float strength)
{
    const int blockSize = 16;
    dim3 gridDim((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);
    dim3 blockDim(blockSize, blockSize);

    unsigned char* devInputImage = nullptr;
    unsigned char* devOutputImage = nullptr;

    cudaMalloc((void**)&devInputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&devOutputImage, width * height * sizeof(unsigned char));

    cudaMemcpy(devInputImage, inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsharpMaskKernel << <gridDim, blockDim >> > (devInputImage, devOutputImage, width, height, strength);

    cudaMemcpy(outputImage, devOutputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(devInputImage);
    cudaFree(devOutputImage);
}

int main()
{
    std::string inputFileName = "input.png";
    std::string outputFileName1 = "output1.png";
    std::string outputFileName2 = "output2.png";

    cv::Mat inputImage = cv::imread(inputFileName, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty())
    {
        std::cout << "Failed to open input file: " << inputFileName << std::endl;
        return 1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat outputImage1(height, width, CV_8UC1);
    cv::Mat outputImage2(height, width, CV_8UC1);

    unsharpMaskCUDA(inputImage.data, outputImage1.data, width, height, 0.3);
    unsharpMaskCUDA(inputImage.data, outputImage2.data, width, height, 0.3);

    cv::imwrite(outputFileName1, outputImage1);
    cv::imwrite(outputFileName2, outputImage2);

    std::cout << "Output files saved: " << outputFileName1 << ", " << outputFileName2 << std:endl;

    return 0;
}

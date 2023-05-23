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

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

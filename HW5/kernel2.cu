#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_res, int resX, int resY, int maxIterations, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisx = blockIdx.x * blockDim.x + threadIdx.x;
    int thisy = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + thisx * stepX;
    float y = lowerY + thisy * stepY;
    
    float z_re = x, z_im = y;
    int i;
    for (i = 0; i < maxIterations; ++i)
    {
      if (z_re * z_re + z_im * z_im > 4.f)
        break;
      float new_re = z_re * z_re - z_im * z_im;
      float new_im = 2.f * z_re * z_im;
      z_re = x + new_re;
      z_im = y + new_im;
    }
    int *row = (int*)((char*)d_res + thisy * pitch);
    row[thisx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int size = resX * sizeof(int);
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int * h_img; cudaHostAlloc((void **)&h_img, size * resY, 0);
    size_t pitch;
    int * d_img; cudaMallocPitch((void **)&d_img, &pitch, size, resY);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    
    mandelKernel<<<numBlock, blockSize>>>(lowerX, lowerY, stepX, stepY, d_img, resX, resY, maxIterations, pitch);

    cudaDeviceSynchronize();

    cudaMemcpy2D(h_img, size, d_img, pitch, size, resY, cudaMemcpyDeviceToHost);

    memcpy(img, h_img, size * resY);

    cudaFree(d_img);
    cudaFree(h_img);
}

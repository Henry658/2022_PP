#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__device__ int mandel(float c_re, float c_im, int maxIteration) {
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < maxIteration; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }
  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_res, int resX, int resY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisx = blockIdx.x * blockDim.x + threadIdx.x;
    int thisy = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisx < resX && thisy < resY) {
        int idx = thisy * resX + thisx;
        float x = lowerX + thisx * stepX;
        float y = lowerY + thisy * stepY;
        d_res[idx] = mandel(x, y, maxIterations);
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int size = resX * resY * sizeof(int);
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int * h_img = (int *)malloc(size);
    int * d_img; cudaMalloc((void **)&d_img, size);

    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    
    mandelKernel<<<numBlock, blockSize>>>(lowerX, lowerY, stepX, stepY, d_img, resX, resY, maxIterations);

    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

    memcpy(img, h_img, size);

    cudaFree(d_img);
    free(h_img);
}

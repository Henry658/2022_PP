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

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_res, int resX, int resY, int maxIterations, int x_pixels, int y_pixels) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i, j, idx;
    float x, y;
    
    int thisx = (blockIdx.x * blockDim.x + threadIdx.x) * x_pixels;
    int thisy = (blockIdx.y * blockDim.y + threadIdx.y) * y_pixels;

    if (thisx < resX && thisy < resY) {
      for(j = thisy ; j < thisy + y_pixels ; j++){
        if(j >= resY) return;
        for(i = thisx ; i < thisx + x_pixels ; i++){
          if(i >= resX) continue;
          x = lowerX + i * stepX;
          y = lowerY + j * stepY;
          idx = j * resX + i;
          d_res[idx] = mandel(x, y, maxIterations);
        }
      }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int size = resX * resY * sizeof(int);
    int x_pixels = 2, y_pixels = 2;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
   
    size_t pitch;
    int * d_img; cudaMallocPitch((void **)&d_img, &pitch, resX * sizeof(int), resY);

    dim3 numBlock(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE / x_pixels, BLOCK_SIZE / y_pixels);

     int * h_img; cudaHostAlloc((void **)&h_img, size, cudaHostAllocMapped);
    
    mandelKernel<<<numBlock, blockSize>>>(lowerX, lowerY, stepX, stepY, d_img, resX, resY, maxIterations, x_pixels, y_pixels);

    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

    memcpy(img, h_img, size);

    cudaFree(d_img);
    cudaFree(h_img);
}

#include "lqt.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>    
#include <linux/cuda.h>

#define ENDIANSWAP(a) (3 - a)

__global__ void cuda_cuda_nodify(struct point* points, unsigned char* array,
                                 const size_t depth, ord_t xstart, ord_t xend, 
                                 ord_t ystart, ord_t yend, size_t numPoints) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t locationLen = depth / 4ul;
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;
  const size_t lastPoint = numPoints * fullPointLen - fullPointLen;

  const size_t pointPos = fullPointLen * i;
  unsigned char* thisArrayPoint = &array[pointPos];
  struct point* thisPoint = &points[i];

  if(pointPos > lastPoint)
    return; // skip the final block remainder

  ord_t currentXStart = xstart;
  ord_t currentXEnd   = xend;
  ord_t currentYStart = ystart;
  ord_t currentYEnd   = yend;
  for(size_t j = 0, jend = depth; j != jend; ++j) {
    const size_t bitsPerLocation = 2;
    const size_t bit1 = thisPoint->y > (currentYStart + (currentYEnd - currentYStart) / 2);
    const size_t bit2 = thisPoint->x > (currentXStart + (currentXEnd - currentXStart) / 2);
    const size_t currentPosBits = (bit1 << 1) | bit2;

    const size_t byte = j / 4;
    const size_t ebyte = byte / 4 * 4 + ENDIANSWAP(byte % 4); // the / 4 * 4 rounds down
    // @note it may be more efficient to create the node, and then loop and 
    //       use an intrinsic, e.g. __builtin_bswap32(pointAsNum[j]). Intrinsics are fast.

    thisArrayPoint[ebyte] = (thisArrayPoint[ebyte] << bitsPerLocation) | currentPosBits;
      
    const ord_t newWidth = (currentXEnd - currentXStart) / 2;
    currentXStart = floor((thisPoint->x - currentXStart) / newWidth) * newWidth + currentXStart;
    currentXEnd = currentXStart + newWidth;

    const ord_t newHeight = (currentYEnd - currentYStart) / 2;
    currentYStart = floor((thisPoint->y - currentYStart) / newHeight) * newHeight + currentYStart;
    currentYEnd = currentYStart + newHeight;
  }

  const size_t pointXPos = locationLen;
  const size_t pointYPos = pointXPos + sizeof(ord_t);
  const size_t keyPos = pointYPos + sizeof(ord_t);

  ord_t* arrayPointX = (ord_t*)&thisArrayPoint[pointXPos];
  *arrayPointX = thisPoint->x;
  thisArrayPoint[pointXPos] = thisPoint->x;
  ord_t* arrayPointY = (ord_t*)&thisArrayPoint[pointYPos];
  *arrayPointY = thisPoint->y;
  key_t* arrayPointKey = (key_t*)&thisArrayPoint[keyPos];
  *arrayPointKey = thisPoint->key;
}

unsigned char* cuda_nodify(struct point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth) {
  // depth must evenly divide 4
//  *depth = sizeof(ord_t) * 8 / 2;
  *depth = 32;
  const size_t locationLen = ceil(*depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;
  const size_t arrayLen = fullPointLen * len;

  const size_t THREADS_PER_BLOCK = 512;

  unsigned char* array = (unsigned char*)malloc(arrayLen);
  unsigned char* cuda_array;
  struct point* cuda_points;
  cudaMalloc((void**)&cuda_array, arrayLen);
  cudaMalloc((void**)&cuda_points, len * sizeof(point));
  cudaMemcpy(cuda_points, points, len * sizeof(point), cudaMemcpyHostToDevice);
  cudaMemset(cuda_array, 0, arrayLen); // debug

  cuda_cuda_nodify<<<(len + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, fullPointLen>>>(cuda_points, cuda_array, *depth, xstart, xend, ystart, yend, len);

  cudaMemcpy(array, cuda_array, arrayLen, cudaMemcpyDeviceToHost);
  cudaFree(cuda_array);
  cudaFree(cuda_points);
  return array;
}

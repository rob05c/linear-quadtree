#include "lqt.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>    
#include <linux/cuda.h>

__global__ void cuda_cuda_nodify(struct lqt_point* points, location_t* locations,
                                 const size_t depth, ord_t xstart, ord_t xend, 
                                 ord_t ystart, ord_t yend, size_t len) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= len)
    return; // skip the final block remainder

  struct lqt_point* thisPoint = &points[i];

  ord_t currentXStart = xstart;
  ord_t currentXEnd = xend;
  ord_t currentYStart = ystart;
  ord_t currentYEnd = yend;
  for(size_t j = 0, jend = depth; j != jend; ++j) {
    const location_t bit1 = thisPoint->y > (currentYStart + (currentYEnd - currentYStart) / 2);
    const location_t bit2 = thisPoint->x > (currentXStart + (currentXEnd - currentXStart) / 2);
    const location_t currentPosBits = (bit1 << 1) | bit2;
    locations[i] = (locations[i] << 2) | currentPosBits;

    const ord_t newWidth = (currentXEnd - currentXStart) / 2;
    currentXStart = floor((thisPoint->x - currentXStart) / newWidth) * newWidth + currentXStart;
    currentXEnd = currentXStart + newWidth;
    const ord_t newHeight = (currentYEnd - currentYStart) / 2;
    currentYStart = floor((thisPoint->y - currentYStart) / newHeight) * newHeight + currentYStart;
    currentYEnd = currentYStart + newHeight;
  }
}

struct linear_quadtree cuda_nodify(struct lqt_point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth) {
  *depth = LINEAR_QUADTREE_DEPTH;

  const size_t THREADS_PER_BLOCK = 512;

  location_t*       cuda_locations;
  struct lqt_point* cuda_points;
  cudaMalloc((void**)&cuda_locations, len * sizeof(location_t));
  cudaMalloc((void**)&cuda_points, len * sizeof(struct lqt_point));
  cudaMemcpy(cuda_points, points, len * sizeof(struct lqt_point), cudaMemcpyHostToDevice);
  cudaMemset(cuda_locations, 0, len * sizeof(location_t)); // debug
  fprintf(stderr, "cn calling cuda_nodify_nodify\n");
  cuda_cuda_nodify<<<(len + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(cuda_points, cuda_locations, *depth, xstart, xend, ystart, yend, len);
  fprintf(stderr, "cn cuda_nodify_nodify returned\n");
  fprintf(stderr, "cn locations malloc\n");
  location_t* locations = (location_t*) malloc(len * sizeof(location_t));
  fprintf(stderr, "cn locations memcpy\n");
  cudaMemcpy(locations, cuda_locations, len * sizeof(location_t), cudaMemcpyDeviceToHost);
  fprintf(stderr, "cn locations free\n");
  cudaFree(cuda_locations);
  fprintf(stderr, "cn points free\n");
  cudaFree(cuda_points);

  struct linear_quadtree lqt;
  lqt.points    = points;
  lqt.locations = locations;
  lqt.length    = len;
  return lqt;
}

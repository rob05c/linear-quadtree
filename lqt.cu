#define CUB_STDERR

#include "lqt.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>    
#include <linux/cuda.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

using namespace cub; // debug



/// \todo fix to not be global
CachingDeviceAllocator g_allocator(true); // CUB caching allocator for device memory

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

struct linear_quadtree lqt_create_cuda(struct lqt_point* points, size_t len, 
                                       ord_t xstart, ord_t xend, 
                                       ord_t ystart, ord_t yend,
                                       size_t* depth) {
  return lqt_sortify_cuda(lqt_nodify_cuda(points, len, xstart, xend, ystart, yend, depth));
}

struct linear_quadtree lqt_nodify_cuda(struct lqt_point* points, size_t len, 
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

struct linear_quadtree lqt_sortify_cuda(struct linear_quadtree lqt) {
  DoubleBuffer<location_t> d_keys;
  DoubleBuffer<lqt_point> d_values;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(location_t) * lqt.length));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(location_t) * lqt.length));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(lqt_point) * lqt.length));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(lqt_point) * lqt.length));

  size_t temp_storage_bytes = 0;
  void* d_temp_storage = NULL;

  CubDebugExit( cudaMemcpy(d_keys.d_buffers[0], lqt.locations, sizeof(location_t) * lqt.length, cudaMemcpyHostToDevice));
  CubDebugExit( cudaMemcpy(d_values.d_buffers[0], lqt.points, sizeof(lqt_point) * lqt.length, cudaMemcpyHostToDevice));
//  CubDebugExit( cudaMemcpy(d_keys.d_buffers[1], lqt.locations, sizeof(location_t) * lqt.length, cudaMemcpyHostToDevice));
//  CubDebugExit( cudaMemcpy(d_values.d_buffers[1], lqt.points, sizeof(lqt_point) * lqt.length, cudaMemcpyHostToDevice));

  // get size of temp storage
  CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, lqt.length));
  CubDebugExit( g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  printf("CS temp size: %u\n", temp_storage_bytes);

  CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, lqt.length));
  
  CubDebugExit( cudaMemcpy(lqt.locations, d_keys.Current(), lqt.length * sizeof(location_t), cudaMemcpyDeviceToHost));
  CubDebugExit( cudaMemcpy(lqt.points, d_values.Current(), lqt.length * sizeof(lqt_point), cudaMemcpyDeviceToHost));

  CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[0]));
  CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[1]));
  CubDebugExit( g_allocator.DeviceFree(d_values.d_buffers[0]));
  CubDebugExit( g_allocator.DeviceFree(d_values.d_buffers[1]));
  CubDebugExit( g_allocator.DeviceFree(d_temp_storage));
  return lqt;
}

void print_array_uint(unsigned int* array, const size_t len) {
  if(len == 0)
    return;
  printf("[%u", array[0]);
  for(size_t i = 1, end = len; i != end; ++i)
    printf(" %u", array[i]);
  printf("]");
}
void print_array_int(int* array, const size_t len) {
  if(len == 0)
    return;
  printf("[%d", array[0]);
  for(size_t i = 1, end = len; i != end; ++i)
    printf(" %d", array[i]);
  printf("]");
}

template <typename T> struct fmt_traits;
template <>
struct fmt_traits<int> {
  static const char* str() {return "%d";}
};
template <>
struct fmt_traits<unsigned int> {
  static const char* str() {return "%u";}
};
template <>
struct fmt_traits<location_t> {
  static const char* str() {return "%lu";}
};

template <typename T>
void print_array(T* array, const size_t len) {
  if(len == 0)
    return;
  printf("[");
  printf(fmt_traits<T>::str(), array[0]);
  for(size_t i = 1, end = len; i != end; ++i) {
    printf(" ");
    printf(fmt_traits<T>::str(), array[i]);
  }
  printf("]");
}

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

/// \returns the device totalGlobalMem
inline size_t GetDeviceMemory() {
  cudaDeviceProp properties;
  int deviceNum;
  CubDebugExit(cudaGetDevice(&deviceNum));
  CubDebugExit(cudaGetDeviceProperties(&properties, deviceNum));
  return properties.totalGlobalMem;
}

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

inline size_t find_min(location_t* keys, const size_t keys_len) {
  if(keys_len == 0)
    return 0;
  location_t min = keys[0];
  size_t min_key = 0;
  for(size_t i = 0, end = keys_len; i != end; ++i) {
    if(keys[i] < min) {
      min_key = i;
      min = keys[i];
    }
  }
  return min_key;
}

/// \param[out] keys must be at least block_len large
/// \return whether all iterators are past their length. That is, when this is false, we can stop merging.
inline bool get_keys(location_t* keys, const struct linear_quadtree* array_blocks, const size_t block_len, const size_t* iterators) {
  bool got_key = false;
  for(int i = 0, end = block_len; i != end; ++i) {
    if(iterators[i] >= array_blocks[i].length) {
      keys[i] = location_t_max; // we've iterated past this block's len; make sure this key is never the min.
      continue;
    }
    got_key = true;
    keys[i] = array_blocks[i].locations[iterators[i]];
  }
  return got_key;
}

struct linear_quadtree lqt_merge(struct linear_quadtree* array_blocks, const size_t block_len, struct lqt_point* points, const size_t len) {
  struct linear_quadtree lqt;
  lqt.points    = points;
  lqt.locations = (location_t*) malloc(sizeof(location_t) * len);
  lqt.length    = len;
  if(len == 0)
    return lqt;

  size_t lqt_iterator = 0;
  size_t* iterators = (size_t*) malloc(sizeof(size_t) * block_len);
  for(size_t i = 0, end = block_len; i != end; ++i)
    iterators[i] = 0;

  {
    location_t keys[block_len];  
    for(size_t i = 0; get_keys(keys, array_blocks, block_len, iterators); ++i) {
      const size_t min_block = find_min(keys, block_len);
      lqt.locations[lqt_iterator] = array_blocks[min_block].locations[iterators[min_block]];
      lqt.points[lqt_iterator]    = array_blocks[min_block].points[iterators[min_block]];
      ++iterators[min_block];
      ++lqt_iterator;
    }
  }
  
  free(iterators);
  return lqt;
}

__global__ void nodify_kernel(struct lqt_point* points, location_t* locations,
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
  // debug
  size_t cuda_mem_free = 0;
  size_t cuda_mem_total = 0;
  CubDebugExit(cudaMemGetInfo(&cuda_mem_free, &cuda_mem_total));

  const size_t array_size = (sizeof(struct lqt_point) + sizeof(location_t)) * len * 2; // *2 for double-buffers
  const size_t num_blocks = array_size / cuda_mem_free + 1;
  printf("num blocks: %lu\n", num_blocks); // debug
  const size_t array_block_size = array_size / num_blocks;
  printf("free: %lu\tarray: %lu\tblocks: %lu\tblock size: %lu\n", cuda_mem_free, array_size, num_blocks, array_block_size); // debug
  
  const size_t block_len = len / num_blocks + (len % num_blocks != 0 ? 1 : 0);
  struct linear_quadtree* array_blocks = (struct linear_quadtree*) malloc(num_blocks * sizeof(linear_quadtree));

  for(size_t i = 0, end = num_blocks; i != end; ++i) {
    array_blocks[i].length = block_len;
    if(block_len * i + block_len  > len)
      array_blocks[i].length -= block_len * num_blocks - len; // fix the last block overlap
    array_blocks[i].points = (struct lqt_point*) malloc(sizeof(struct lqt_point) * array_blocks[i].length);
    memcpy(array_blocks[i].points, points + block_len * i, array_blocks[i].length * sizeof(struct lqt_point));
    array_blocks[i] = lqt_sortify_cuda_mem(lqt_nodify_cuda_mem(array_blocks[i].points, array_blocks[i].length, xstart, xend, ystart, yend, depth));
  }
  
  struct linear_quadtree lqt = lqt_merge(array_blocks, num_blocks, points, len);
  for(size_t i = 0, end = num_blocks; i != end; ++i)
    lqt_delete(array_blocks[i]);
  free(array_blocks);
  return lqt;
}

/// unnecessarily allocates and frees CUDA memory twice
struct linear_quadtree lqt_create_cuda_slow(struct lqt_point* points, size_t len, 
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
  nodify_kernel<<<(len + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(cuda_points, cuda_locations, *depth, xstart, xend, ystart, yend, len);
  location_t* locations = (location_t*) malloc(len * sizeof(location_t));
  cudaMemcpy(locations, cuda_locations, len * sizeof(location_t), cudaMemcpyDeviceToHost);
  cudaFree(cuda_locations);
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


  CubDebugExit( cudaMemcpy(d_keys.d_buffers[0], lqt.locations, sizeof(location_t) * lqt.length, cudaMemcpyHostToDevice));
  CubDebugExit( cudaMemcpy(d_values.d_buffers[0], lqt.points, sizeof(lqt_point) * lqt.length, cudaMemcpyHostToDevice));

  size_t temp_storage_bytes = 0;
  void* d_temp_storage = NULL;
  CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, lqt.length));
  CubDebugExit( g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

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

// @return CUDA-allocated points and locations, along with existing host-allocated points
struct linear_quadtree_cuda lqt_nodify_cuda_mem(struct lqt_point* points, size_t len, 
                                                ord_t xstart, ord_t xend, 
                                                ord_t ystart, ord_t yend,
                                                size_t* depth) {
  const size_t THREADS_PER_BLOCK = 512;
  *depth = LINEAR_QUADTREE_DEPTH;
  location_t*       cuda_locations;
  struct lqt_point* cuda_points;

  CubDebugExit(g_allocator.DeviceAllocate((void**)&cuda_locations, sizeof(location_t) * len));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&cuda_points, sizeof(lqt_point) * len));
//  cudaMalloc((void**)&cuda_locations, len * sizeof(location_t));
//  cudaMalloc((void**)&cuda_points, len * sizeof(struct lqt_point));
  cudaMemcpy(cuda_points, points, len * sizeof(struct lqt_point), cudaMemcpyHostToDevice);
  cudaMemset(cuda_locations, 0, len * sizeof(location_t)); // debug
  nodify_kernel<<<(len + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(cuda_points, cuda_locations, *depth, xstart, xend, ystart, yend, len);

  struct linear_quadtree_cuda lqt;
  lqt.points         = points;
  lqt.cuda_locations = cuda_locations;
  lqt.cuda_points    = cuda_points;
  lqt.length         = len;
  return lqt;
}


struct linear_quadtree lqt_sortify_cuda_mem(struct linear_quadtree_cuda cuda_lqt) {
  //  printf("DEBUG lqt_sortify_cuda_mem\n"); // debug

  DoubleBuffer<location_t> d_keys;
  DoubleBuffer<lqt_point> d_values;
  d_keys.d_buffers[0]   = cuda_lqt.cuda_locations; // reuse the nodify CUDA memory for the cub buffers
  d_values.d_buffers[0] = cuda_lqt.cuda_points;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(location_t) * cuda_lqt.length));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(lqt_point) * cuda_lqt.length));

  size_t temp_storage_bytes = 0;
  void* d_temp_storage = NULL;
  CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, cuda_lqt.length));
  //  printf("temp storage: %lu\n", temp_storage_bytes);  // debug
  CubDebugExit( g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, cuda_lqt.length));

  struct linear_quadtree lqt;
  lqt.length = cuda_lqt.length;
  lqt.locations = (location_t*) malloc(lqt.length * sizeof(location_t));
  CubDebugExit( cudaMemcpy(lqt.locations, d_keys.Current(), lqt.length * sizeof(location_t), cudaMemcpyDeviceToHost));
  lqt.points = cuda_lqt.points;
  CubDebugExit( cudaMemcpy(lqt.points, d_values.Current(), lqt.length * sizeof(lqt_point), cudaMemcpyDeviceToHost));

  CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[0]));
  CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[1]));
  CubDebugExit( g_allocator.DeviceFree(d_values.d_buffers[0]));
  CubDebugExit( g_allocator.DeviceFree(d_values.d_buffers[1]));
  CubDebugExit( g_allocator.DeviceFree(d_temp_storage));
  return lqt;
}

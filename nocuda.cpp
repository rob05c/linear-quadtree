/// this file exists to remove C++11 from CUDA, to support outdated nvcc compilers
#include "lqt.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>    
#include <assert.h>
#include <math.h>

#include <vector>
#include <utility>
#include <iostream>
#include <chrono>
#include <future>

#include "mergesort.hh"
#include "samplesort.hh"
#include "tbb/tbb.h"

using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::promise;
using std::future;

namespace {
/*
/// \todo put these in a header, so they aren't duplicated
/// initialize boundary so the first udpate overrides it.
inline void init_boundary(struct rtree_rect* boundary) {
  boundary->top = ord_t_max;
  boundary->bottom = ord_t_lowest;
  boundary->left = ord_t_max;
  boundary->right = ord_t_lowest;
}
inline void update_boundary(struct rtree_rect* boundary, struct rtree_point* p) {
  /// \todo replace these with CUDA min/max which won't use conditionals
  boundary->top = fmin(p->y, boundary->top);
  boundary->bottom = fmax(p->y, boundary->bottom);
  boundary->left = fmin(p->x, boundary->left);
  boundary->right = fmax(p->x, boundary->right);
}
inline void update_boundary(struct rtree_rect* boundary, struct rtree_rect* node) {
  /// \todo replace these with CUDA min/max which won't use conditionals
  boundary->top = fmin(node->top, boundary->top);
  boundary->bottom = fmax(node->bottom, boundary->bottom);
  boundary->left = fmin(node->left, boundary->left);
  boundary->right = fmax(node->right, boundary->right);
}
*/
} // namespace


// x value ALONE is used for comparison, to create an xpack
bool operator<(const lqt_unified_node& rhs, const lqt_unified_node& lhs) {
  return rhs.location < lhs.location;
}

size_t tbb_num_default_thread() {
  return tbb::task_scheduler_init::default_num_threads();
}

void tbb_test_scheduler_init() {
  for(size_t i = 1; i < 1025; i *= 2) {
    const size_t num_threads = i;
    const auto start = std::chrono::high_resolution_clock::now();
    tbb::task_scheduler_init init(num_threads);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "tbb scheduler init test - threads: " << num_threads << "\tms: " << duration << std::endl;
  }
}

linear_quadtree_unified tbb_sortify_unified(linear_quadtree_unified lqt, const size_t threads) {
//  auto lowxpack = [](const rtree_point& rhs, const struct rtree_point& lhs) {
//    return rhs.x < rhs.y;
//  };
//  tbb::task_scheduler_init init(threads);
  tbb::parallel_sort(lqt.nodes, lqt.nodes + lqt.length);
  return lqt;
}

linear_quadtree_unified sisd_sortify_unified(linear_quadtree_unified lqt, const size_t threads) {
  std::sort(lqt.nodes, lqt.nodes + lqt.length);
  return lqt;
}

/// does not block for GPU memory. Will fail, if GPU memory is insufficient.
linear_quadtree_unified lqt_create_heterogeneous(lqt_point* points, size_t len, 
                                                       ord_t xstart, ord_t xend, 
                                                       ord_t ystart, ord_t yend,
                                                       size_t* depth, const size_t threads) {
  return tbb_sortify_unified(lqt_nodify_cuda_unified(points, len, xstart, xend, ystart, yend, depth), threads);
}

/// does not block for GPU memory. Will fail, if GPU memory is insufficient.
linear_quadtree_unified lqt_create_sisd(lqt_point* points, size_t len, 
                                                       ord_t xstart, ord_t xend, 
                                                       ord_t ystart, ord_t yend,
                                                       size_t* depth, const size_t threads) {
  return sisd_sortify_unified(lqt_nodify_cuda_unified(points, len, xstart, xend, ystart, yend, depth), threads);
}


linear_quadtree_unified merge_sortify_unified(linear_quadtree_unified lqt, const size_t threads) {
  lqt.nodes = parallel_mergesort(lqt.nodes, lqt.nodes + lqt.length, threads);
  return lqt;
}

linear_quadtree_unified lqt_create_heterogeneous_mergesort(lqt_point* points, size_t len, 
                                                           ord_t xstart, ord_t xend, 
                                                           ord_t ystart, ord_t yend,
                                                           size_t* depth, const size_t threads) {
  return merge_sortify_unified(lqt_nodify_cuda_unified(points, len, xstart, xend, ystart, yend, depth), threads);
}

linear_quadtree_unified sample_sortify_unified(linear_quadtree_unified lqt, const size_t threads) {
  lqt.nodes = parallel_samplesort(lqt.nodes, lqt.nodes + lqt.length, threads);
  return lqt;
}

linear_quadtree_unified lqt_create_heterogeneous_samplesort(lqt_point* points, size_t len, 
                                                           ord_t xstart, ord_t xend, 
                                                           ord_t ystart, ord_t yend,
                                                           size_t* depth, const size_t threads) {
  return sample_sortify_unified(lqt_nodify_cuda_unified(points, len, xstart, xend, ystart, yend, depth), threads);
}

/*
/// SISD sort via single CPU core (for benchmarks)
struct rtree cuda_create_rtree_sisd(struct rtree_point* points, const size_t len) {
  std::sort(points, points + len);
  struct rtree_leaf* leaves = cuda_create_leaves_together(points, len);
  const size_t leaves_len = DIV_CEIL(len, RTREE_NODE_SIZE);

  rtree_node* previous_level = (rtree_node*) leaves;
  size_t      previous_len = leaves_len;
  size_t      depth = 1; // leaf level is 0
  while(previous_len > RTREE_NODE_SIZE) {
    previous_level = cuda_create_level(previous_level, previous_len);
    previous_len = DIV_CEIL(previous_len, RTREE_NODE_SIZE);
    ++depth;
  }

  rtree_node* root = (rtree_node*) malloc(sizeof(rtree_node));
  init_boundary(&root->bounding_box);
  root->num = previous_len;
  root->children = previous_level;
  for(size_t i = 0, end = previous_len; i != end; ++i)
    update_boundary(&root->bounding_box, &root->children[i].bounding_box);
  ++depth;

  struct rtree tree = {depth, root};
  return tree;
}
*/

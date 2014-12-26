#include "lqt.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// generate a uniform random between min and max exclusive
static inline ord_t uniformFrand(const ord_t min, const ord_t max) {
  const double r = (double)rand() / RAND_MAX;
  return min + r * (max - min);
}

static inline void test_endian_2(const size_t len, const size_t threads) {
  printf("test_endian_2\n");

//  static_assert(sizeof(unsigned int) == 4, "sizeof(int) is not 4, fix the below code")

  unsigned char a[4];
  unsigned char* array = a;
  // array[0] = 11
  array[0] = 0x0;
  array[0] = (array[0] << 2) | 0x1;
  array[0] = (array[0] << 2) | 0x2;
  array[0] = (array[0] << 2) | 0x3;

  // array[1] = 10
  array[1] = 0x3;
  array[1] = (array[1] << 2) | 0x2;
  array[1] = (array[1] << 2) | 0x1;
  array[1] = (array[1] << 2) | 0x0;

  // array[2] = 01
  array[2] = 0x0;
  array[2] = (array[2] << 2) | 0x3;
  array[2] = (array[2] << 2) | 0x2;
  array[2] = (array[2] << 2) | 0x1;

  // array[3] = 00 00 00 00
  array[3] = 0x3;
  array[3] = (array[3] << 2) | 0x2;
  array[3] = (array[3] << 2) | 0x0;
  array[3] = (array[3] << 2) | 0x1;

  unsigned int* iarray = (unsigned int*)array;
//  unsigned int endian = (array[0] << 24) | (array[1] << 16) | (array[2] << 8) | array[3];
  printf("endian: %u\n", *iarray);
}

static inline void test_many(const size_t len, const size_t threads) {
  printf("test_many\n");
  struct lqt_point* points = malloc(len * sizeof(struct lqt_point));
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = len; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  {
    printf("creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify(points, len, 
                                                     min, max, min, max, &depth);
    printf("sorting...\n");
    lqt_sortify(lqt);
    printf("\ndone\n");
    lqt_print_nodes(lqt, false);
    lqt_delete(lqt);
  }

  {
    printf("cuda creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify_cuda(points, len, min, max, min, max, &depth);
    printf("cuda sorting...\n");
    lqt_sortify_cuda(lqt);
    printf("\ncuda done\n");
    lqt_print_nodes(lqt, false);
    lqt_delete(lqt);
  }

}

static inline void test_endian(const size_t len, const size_t threads) {
  printf("test_endian\n");
  typedef unsigned char uchar;
  typedef unsigned long sort_t;

  //  const unsigned short esa8[8] = {7, 6, 5, 4, 3, 2, 1, 0}; ///< lookup table
  //# define ENDIANSWAP8(a) (esa8[(a) % 8] + (a) / 8 * 8)

  //  const unsigned short esa4[4] = {3, 2, 1, 0}; ///< lookup table
  //# define ENDIANSWAP4(a) (esa4[(a) % 4] + (a) / 4 * 4)

  uchar chars[8];
  chars[0] = 37;
  chars[1] = 228;
  chars[2] = 99;
  chars[3] = 42;

//  sort_t* ichars = (sort_t*)chars; 
//  sort_t val = *ichars;
//  std::cout << "val " << val << std::endl;

//  uchar newchars[4];
//  for(int i = 0, end = sizeof(sort_t); i != end; ++i)
//    newchars[i] = chars[ENDIANSWAP4(i)];
//
//  ichars = (sort_t*)newchars; 
//  val = *ichars;

  sort_t val = 0;
  val = chars[3] | (chars[2] << 8) | (chars[1] << 16) | (chars[0] << 24);
  
  printf("eval %lu\n", val);
}

static inline void test_few(const size_t len, const size_t threads) {
  printf("test_few\n");
  struct lqt_point* points = malloc(len * sizeof(struct lqt_point));
  const ord_t min = 0.0;
  const ord_t max = 300.0;
  printf("creating points...\n");
  points[0].x = 299.999;
  points[0].y = 299.999;
  points[0].key = 42;
  points[1].x = 7.0;
  points[1].y = 14.0;
  points[1].key = 99;

  {
    printf("creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify(points, len, 
                                        min, max, min, max, &depth);
    printf("sorting...\n");
    lqt_sortify(lqt);
    printf("\ndone\n");
    lqt_print_nodes(lqt, true);
    lqt_delete(lqt);
  }

  {
    printf("cuda creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify_cuda(points, len, 
                                             min, max, min, max, &depth);
    printf("cuda sorting...\n");
    lqt_sortify_cuda(lqt);
    printf("\ncuda done\n");
    lqt_print_nodes(lqt, true);
  }

}

static inline void test_time(const size_t numPoints, const size_t threads) {
  printf("test_time\n");
  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  size_t depth;
  printf("cpu nodify...\n");
  const clock_t start = clock();
  struct linear_quadtree lqt = lqt_nodify(points, numPoints, 
                                      min, max, min, max, &depth);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  printf("cpu nodify time: %fs\n", elapsed_s);
  lqt_delete(lqt);
  // lqt and points not valid henceforth and hereafter.

  printf("creating cuda points...\n");
  struct lqt_point* cuda_points = malloc(sizeof(struct lqt_point) * numPoints);
  printf("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    cuda_points[i].x = uniformFrand(min, max);
    cuda_points[i].y = uniformFrand(min, max);
    cuda_points[i].key = i;
  }

  printf("gpu nodify...\n");
  const clock_t start_cuda = clock();
  struct linear_quadtree cuda_lqt = lqt_nodify_cuda(cuda_points, numPoints, 
                                                min, max, min, max, &depth);
  const clock_t end_cuda = clock();
  const double elapsed_s_cuda = (end_cuda - start_cuda) / (double)CLOCKS_PER_SEC;
  const double speedup = elapsed_s / elapsed_s_cuda;
  printf("gpu nodify time: %fs\n", elapsed_s_cuda);
  printf("gpu speedup: %f\n", speedup);
  lqt_delete(cuda_lqt);
}

static inline void test_sorts(const size_t numPoints, const size_t threads) {
  printf("test_sorts\n");

  struct lqt_point* points = malloc(numPoints * sizeof(struct lqt_point));
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  printf("creating nodes...\n");
  size_t depth;
  struct linear_quadtree qt = lqt_nodify(points, numPoints, 
                                            min, max, min, max, &depth);
  struct linear_quadtree qt_cuda;
  lqt_copy(&qt_cuda, &qt);

  printf("sorting...\n");
  lqt_sortify(qt);
  printf("sorting cuda...\n");
  lqt_sortify_cuda(qt_cuda);

  printf("nodes:\n");
  lqt_print_nodes(qt, false);
  printf("cuda nodes:\n");
  lqt_print_nodes(qt_cuda, false);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

static inline void test_sort_time(const size_t numPoints, const size_t threads) {
  printf("test_sort_time\n");
  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  printf("creating nodes...\n");
  size_t depth;
  struct linear_quadtree qt = lqt_nodify(points, numPoints, 
                                     min, max, min, max, &depth);
  struct linear_quadtree qt_cuda;
  lqt_copy(&qt_cuda, &qt);

  printf("sorting...\n");
  const clock_t start = clock();
  lqt_sortify(qt);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  printf("sort time: %fs\n", elapsed_s);

  printf("sorting cuda...\n");
  const clock_t start_cuda = clock();
  lqt_sortify_cuda(qt_cuda);
  const clock_t end_cuda = clock();
  const double elapsed_s_cuda = (end_cuda - start_cuda) / (double)CLOCKS_PER_SEC;
  const double cuda_speedup = elapsed_s / elapsed_s_cuda;
  printf("cuda sort time: %fs\n", elapsed_s_cuda);
  printf("cuda speedup: %f\n", cuda_speedup);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

static inline void test_unified_sorts(const size_t numPoints, const size_t threads) {
  printf("test_unified_sorts\n");
  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }
  struct lqt_point* points_cuda = malloc(numPoints * sizeof(struct lqt_point));
  memcpy(points_cuda, points, numPoints * sizeof(struct lqt_point));

  printf("points: %lu\n", numPoints);

  printf("creating quadtree...\n");
  size_t depth;
  struct linear_quadtree qt = lqt_create(points, numPoints, 
                                         min, max, min, max, &depth);
  printf("creating quadtree with CUDA...\n");
  struct linear_quadtree qt_cuda = lqt_create_cuda(points_cuda, numPoints, 
                                                   min, max, min, max, &depth);
  printf("nodes:\n");
  lqt_print_nodes(qt, false);
  printf("cuda nodes:\n");
  lqt_print_nodes(qt_cuda, false);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

static inline void test_unified(const size_t numPoints, const size_t threads) {
  printf("test_unified\n");
  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }
  struct lqt_point* points_cuda = malloc(numPoints * sizeof(struct lqt_point));
  memcpy(points_cuda, points, numPoints * sizeof(struct lqt_point));

  printf("points: %lu\n", numPoints);
  printf("creating quadtree...\n");
  const clock_t start = clock();
  size_t depth;
  struct linear_quadtree qt = lqt_create(points, numPoints, 
                                         min, max, min, max, &depth);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  printf("cpu time: %fs\n", elapsed_s);
  printf("ms per point: %f\n", 1000.0 * elapsed_s / numPoints);

  printf("creating quadtree with CUDA...\n");
  const clock_t start_cuda = clock();
  struct linear_quadtree qt_cuda = lqt_create_cuda(points_cuda, numPoints, 
                                                   min, max, min, max, &depth);
  const clock_t end_cuda = clock();
  const double elapsed_s_cuda = (end_cuda - start_cuda) / (double)CLOCKS_PER_SEC;
  const double cuda_speedup = elapsed_s / elapsed_s_cuda;
  printf("cuda time: %fs\n", elapsed_s_cuda);
  printf("ms per cuda point: %f\n", 1000.0 * elapsed_s_cuda / numPoints);
  printf("cuda speedup: %f\n", cuda_speedup);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

static inline void test_heterogeneous(const size_t numPoints, const size_t threads) {
  printf("test_unified\n");
  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }
  struct lqt_point* points_cuda = malloc(numPoints * sizeof(struct lqt_point));
  memcpy(points_cuda, points, numPoints * sizeof(struct lqt_point));

  printf("points: %lu\n", numPoints);
  printf("creating quadtree...\n");
  const clock_t start = clock();
  size_t depth;
  struct linear_quadtree_unified qt = lqt_create_heterogeneous(points, numPoints, min, max, min, max, &depth, threads);

  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  printf("cpu time: %fs\n", elapsed_s);
  printf("ms per point: %f\n", 1000.0 * elapsed_s / numPoints);

  lqt_delete_unified(qt);
}

void(*test_funcs[])(const size_t, const size_t threads) = {
  test_endian_2,
  test_many,
  test_endian,
  test_few,
  test_time,
  test_sorts,
  test_sort_time,
  test_unified,
  test_unified_sorts,
  test_heterogeneous,
};

static const char* default_app_name = "mergesort";

const char* tests[][2] = {
  {"test_endian_2"     , "test endianness conversions between 4-byte array"},
  {"test_many"         , "print brief reports for many points"},
  {"test_endian"       , "test endian shifting in 4-byte array"},
  {"test_few"          , "print detailed reports for a few points"},
  {"test_time"         , "benchmark the time to create nodes using CPU vs CUDA"},
  {"test_sorts"        , "test the values produced by sorting with CPU vs CUDA"},
  {"test_sort_time"    , "benchmark the time to sort using CPU vs CUDA"},
  {"test_unified"      , "benchmark the time to create and sort using CPU vs CUDA"},
  {"test_unified_sorts", "test the values produced by CPU vs CUDA with unified create+sort function"},
  {"test_heterogeneous", "benchmark the time to create using CUDA and sort using CPU"},
};

const size_t test_num = sizeof(tests) / (sizeof(const char*) * 2);

struct app_arguments {
  bool        success;
  const char* app_name;
  size_t      test_num;
  size_t      array_size;
  size_t      threads;
};

static struct app_arguments parseArgs(const int argc, const char** argv) {
  struct app_arguments args;
  args.success = false;

  if(argc < 1)
    return args;
  args.app_name = argv[0];

  if(argc < 2)
    return args;
  args.test_num = strtol(argv[1], NULL, 10);

  if(argc < 3)
    return args;
  args.array_size = strtol(argv[2], NULL, 10);

  if(argc < 4)
    return args;
  args.threads = strtol(argv[3], NULL, 10);

  args.success = true;
  return args;
}

/// \param[out] msg
/// \param[out] msg_len
static void print_usage(const char* app_name) {
  printf("usage: %s test_num array_size threads\n", strlen(app_name) == 0 ? default_app_name : app_name);
  printf(" (threads is only used for heterogeneous test(s)\n");
  printf("\n");
  printf("       num test            description\n");
  for(size_t i = 0, end = test_num; i != end; ++i) {
    printf("       %-3.1lu %-15.15s %s\n", i, tests[i][0], tests[i][1]);
  }
  printf("\n");
}

int main(const int argc, const char** argv) {
  srand(time(NULL));

  const struct app_arguments args = parseArgs(argc, argv);
  if(!args.success) {
    print_usage(args.app_name);
    return 0;
  }

  test_funcs[args.test_num](args.array_size, args.threads);
  printf("\n");
  return 0;
}

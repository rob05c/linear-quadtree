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

void test_endian_2() {
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

void test_many() {
  printf("test_many\n");
  const size_t len = 10000;
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
    struct linear_quadtree unsortedQuadtree = nodify(points, len, 
                                                     min, max, min, max, &depth);
    printf("sorting...\n");
    sortify(unsortedQuadtree);
    printf("\ndone\n");
    printNodes(unsortedQuadtree, false);
    delete_linear_quadtree(unsortedQuadtree);
  }
/*
  {
    printf("cuda creating nodes...\n");
    size_t depth;
    struct linear_quadtree unsortedQuadtree = cuda_nodify(points, len, min, max, min, max, &depth);
    printf("cuda sorting...\n");
    sortify(unsortedQuadtree);
    printf("\ncuda done\n");
    printNodes(unsortedQuadtree, false);
    delete_linear_quadtree(unsortedQuadtree);
  }
*/
}

void test_endian() {
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

void test_few() {
  printf("test_few\n");
  const size_t len = 2;
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
    struct linear_quadtree unsortedQuadtree = nodify(points, len, 
                                                     min, max, min, max, &depth);
    printf("sorting...\n");
    sortify(unsortedQuadtree);
    printf("\ndone\n");
    printNodes(unsortedQuadtree, true);
    delete_linear_quadtree(unsortedQuadtree);
  }
/*
  {
    printf("cuda creating nodes...\n");
    size_t depth;
    unsigned char* unsortedQuadtree = cuda_nodify(points, len, 
                                                  min, max, min, max, &depth);
    printf("cuda sorting...\n");
    sortify(unsortedQuadtree, sizeof(points) / sizeof(struct point), depth);
    printf("\ncuda done\n");
    printNodes(unsortedQuadtree, sizeof(points), depth, false);
  }
*/
}

void test_time() {
  printf("test_time\n");
  const size_t numPoints = 100000000;
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
  const clock_t start = clock();
  printf("cpu nodify...\n");
  struct linear_quadtree unsortedQuadtree = nodify(points, numPoints, 
                                                   min, max, min, max, &depth);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  printf("cpu nodify time: %fs\n", elapsed_s);
  delete_linear_quadtree(unsortedQuadtree);

/*
  printf("gpu nodify...\n");
  const clock_t start_cuda = clock();
  unsortedQuadtree = cuda_nodify(points, numPoints, 
                                  min, max, min, max, &depth);
  const clock_t end_cuda = clock();
  const double elapsed_s_cuda = (end_cuda - start_cuda) / (double)CLOCKS_PER_SEC;
  const double speedup = elapsed_s / elapsed_s_cuda;
  printf("gpu nodify time: %fs\n", elapsed_s_cuda);
  printf("gpu speedup: %f\n", speedup);

*/
}

void test_sorts() {
  printf("test_sorts\n");

  const size_t numPoints = 10000;
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
  struct linear_quadtree qt_bubble = nodify(points, numPoints, 
                                            min, max, min, max, &depth);
  struct linear_quadtree qt_radix;
  linear_quadtree_copy(&qt_radix, &qt_bubble);

  printf("sorting bubble...\n");
  sortify_bubble(qt_bubble);
  printf("sorting radix...\n");
  sortify_radix(qt_radix);
  printf("bubble nodes:\n");
  printNodes(qt_bubble, false);
  printf("radix nodes:\n");
  printNodes(qt_radix, false);
  delete_linear_quadtree(qt_bubble);
  delete_linear_quadtree(qt_radix);
}

void test_sort_time() {
  printf("test_sort_time\n");
  const size_t numPoints = 100000;
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
  struct linear_quadtree qt_bubble = nodify(points, numPoints, 
                                            min, max, min, max, &depth);
  struct linear_quadtree qt_radix;
  linear_quadtree_copy(&qt_radix, &qt_bubble);

  printf("sorting bubble...\n");
  const clock_t start = clock();
  sortify_bubble(qt_bubble);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  printf("bubble sort time: %fs\n", elapsed_s);

  printf("sorting radix...\n");
  const clock_t start_radix = clock();
  sortify_radix(qt_radix);
  const clock_t end_radix = clock();
  const double elapsed_s_radix = (end_radix - start_radix) / (double)CLOCKS_PER_SEC;
  const double speedup = elapsed_s / elapsed_s_radix;
  printf("radix sort time: %fs\n", elapsed_s_radix);
  printf("radix speedup: %f\n", speedup);
  delete_linear_quadtree(qt_bubble);
  delete_linear_quadtree(qt_radix);
}


int main() {
  srand(time(NULL));

  test_sort_time();
  printf("\n");
  return 0;
}

#include "lqt.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// generate a uniform random between min and max exclusive
ord_t uniformFrand(ord_t min, ord_t max) {
  const double r = (double)rand() / RAND_MAX;
  return min + r * (max - min);
}

void debug_testEndian() {
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

void test1() {
  printf("TEST1\n");
  struct point points[10000];
  const size_t min = 1000;
  const size_t max = 1100;
  printf("creating points...\n");
  for(int i = 0, end = sizeof(points) / sizeof(struct point); i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  printf("creating nodes...\n");
  size_t depth;
  unsigned char* unsortedQuadtree = nodify(points, sizeof(points) / sizeof(struct point), 
                                  min, max, min, max, &depth);
  printf("sorting...\n");
  sortify(unsortedQuadtree, sizeof(points) / sizeof(struct point), depth);
  printf("\ndone\n");
  printNodes(unsortedQuadtree, sizeof(points), depth, false);
}

void teste() {
  printf("TESTE\n");
  //  typedef unsigned long ulong;
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

//  debug_testEndian();

}

void test2() {
  printf("TEST2\n");
  struct point points[1];
  const size_t min = 0;
  const size_t max = 1000;
  printf("creating points...\n");
  points[0].x = 229;
  points[0].y = 297;
  points[0].key = 42;
  printf("creating nodes...\n");
  size_t depth;
  unsigned char* unsortedQuadtree = nodify(points, sizeof(points) / sizeof(struct point), 
                                  min, max, min, max, &depth);
  printf("sorting...\n");
  sortify(unsortedQuadtree, sizeof(points) / sizeof(struct point), depth);
  printf("\ndone\n");
  printNodes(unsortedQuadtree, sizeof(points), depth, true);
  free(unsortedQuadtree);
}

int main() {
  srand(time(NULL));

  test2();

  return 0;
}

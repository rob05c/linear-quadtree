#include "lqt.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define ENDIANSWAP(a) (3 - a)

/* 
 * Turn an array of points into an unsorted quadtree of nodes.
 * You'll probably want to call sortify() to sort the list into a
 * useful quadtree.
 *
 * @param[out] depth the depth of the quadtree. This is important for
 *             a linear quadtree, as it signifies the number of
 *             identifying bit-pairs preceding the node
 *
 * @return a new array representing the unsorted nodes of the quadtree. caller takes ownership
 */
unsigned char* nodify(struct point* points, size_t len, 
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

  unsigned char* array = malloc(sizeof(unsigned char) * arrayLen);

  for(size_t i = 0, end = len; i != end; ++i) {

    const size_t pointPos = fullPointLen * i;
    unsigned char* thisArrayPoint = &array[pointPos];
    struct point* thisPoint = &points[i];

    ord_t currentXStart = xstart;
    ord_t currentXEnd = xend;
    ord_t currentYStart = ystart;
    ord_t currentYEnd = yend;
    for(size_t j = 0, jend = *depth; j != jend; ++j) {
      const size_t bitsPerLocation = 2;
      const size_t bit1 = thisPoint->y > (currentYStart + (currentYEnd - currentYStart) / 2);
      const size_t bit2 = thisPoint->x > (currentXStart + (currentXEnd - currentXStart) / 2);
      const size_t currentPosBits = (bit1 << 1) | bit2;

      const size_t byte = j / 4;
      const size_t ebyte = byte / 4 * 4 + ENDIANSWAP(byte % 4);
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
  return array;
}

/*
 * swap the memory of the given quadtree points
 */
static inline void swapify(unsigned char* firstPoint, unsigned char* secondPoint, const size_t depth) {
  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;

  unsigned char* temp = malloc(sizeof(unsigned char) * fullPointLen);
  memcpy(temp, firstPoint, fullPointLen);
  memcpy(firstPoint, secondPoint, fullPointLen);
  memcpy(secondPoint, temp, fullPointLen);
  free(temp);
}

/*
 * Sort an unsorted linear quadtree. Unsorted linear quadtrees aren't
 * very useful.
 * 
 * Currently uses bubblesort, because I'm lazy. This implementation is
 * primarily a test to be ported to a GPU. Hence, I don't really care
 * how it's sorted. It would be trivial to change this to Mergesort.
 *
 * @param array unsorted linear quadtree
 * @param len   number of points in the quadtree
 * @param depth depth of the quadtree. 
 */
void sortify(unsigned char* array, const size_t len, const size_t depth) {
  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;

  typedef unsigned int sort_t;
  const size_t sortDepths = ceil((depth / 4) / (double)sizeof(sort_t));

  bool swapped = true;
  // bubble sort - will iterate a maximum of n times
  while(swapped) { 
    swapped = false;
    for(size_t i = 0, end = len * fullPointLen; i < end; i += fullPointLen) { //must be < not !=
      if(i + fullPointLen >= len * fullPointLen)
        break; // last point

      unsigned char* point = &array[i];
      unsigned char* nextPoint = &array[i + fullPointLen];

      const sort_t* pointAsNum = (sort_t*)point;
      const sort_t* nextPointAsNum = (sort_t*)nextPoint;
      
      for(size_t j = 0, jend = sortDepths; j < jend; ++j) { // must be < not !=
        const sort_t key = pointAsNum[j];
        const sort_t nextKey = nextPointAsNum[j];
        if(key < nextKey)
          break;
        if(key > nextKey) {
          swapify(point, nextPoint, depth);
          swapped = true;
          break;
        }
        // keys are equal - loop into next depth
      }
    }
  }
}


/*
 * print out a quadtree node
 * @param depth the quadtree depth. Necessary, because it indicates
 *              the number of position bit-pairs
 */
void printNode(unsigned char* node, const size_t depth, const bool verbose) {
  const size_t locationLen = ceil(depth / 4ul);

  if(verbose)
  {
    for(size_t i = 0, end = ceil(depth/4); i != end; ++i) {
      const unsigned char thisByte = node[i];
      printf("%d%d %d%d %d%d %d%d ", 
	     ((thisByte & 0x80) == 0 ? 0 : 1), 
	     ((thisByte & 0x40) == 0 ? 0 : 1),
	     ((thisByte & 0x20) == 0 ? 0 : 1),
	     ((thisByte & 0x10) == 0 ? 0 : 1),
	     ((thisByte & 0x8) == 0 ? 0 : 1),
	     ((thisByte & 0x4) == 0 ? 0 : 1),
	     ((thisByte & 0x2) == 0 ? 0 : 1),
	     ((thisByte & 0x1) == 0 ? 0 : 1));
    }
  }

  typedef unsigned int sort_t;
  const size_t sortDepths = ceil((depth / 4) / (double)sizeof(sort_t));
  const sort_t* pointAsNum = (sort_t*)node;

  if(verbose)
  {
    for(size_t j = 0, jend = sortDepths; j < jend; ++j) { // must be <
      const sort_t key = pointAsNum[j];
      printf("%u ", key);
    }
  }

  const size_t pointXPos = locationLen;
  const size_t pointYPos = pointXPos + sizeof(ord_t);
  const size_t keyPos = pointYPos + sizeof(ord_t);

  const ord_t* arrayPointX = (ord_t*)&node[pointXPos];
  const ord_t* arrayPointY = (ord_t*)&node[pointYPos];
  const key_t* arrayPointKey = (key_t*)&node[keyPos];

  printf("%.15f\t%.15f\t%d\n", *arrayPointX, *arrayPointY, *arrayPointKey);
}

/* 
 * print out all the nodes in a linear quadtree
 * @param array the linear quadtree
 * @param len the number of nodes in the quadtree
 * @param depth the depth of the quadtree.
 */
void printNodes(unsigned char* array, const size_t len, const size_t depth, const bool verbose) {
  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;

  printf("linear quadtree: \n");
  if(verbose) {
    for(size_t i = 0, end = ceil(depth/4); i < end; ++i) {
      printf("            ");
    }
  }

  printf("x\ty\tkey\n");
  for(size_t i = 0, end = len; i < end; i += fullPointLen) { // must be < not !=
    printNode(&array[i], depth, verbose);
  }
  printf("\n");
}

#undef ENDIANSWAP

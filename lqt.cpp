#include "lqt.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <climits> ///< @todo remove
#include <limits>

namespace {
using std::cout;
using std::endl;

const unsigned int esa[4] = {3, 2, 1, 0}; ///< endianness-swapping lookup table, to avoid conditionals
}

#define ENDIANSWAP(a) (esa[(a) % 4] + (a) / 4 * 4)

namespace linear_quadtree {

/* 
 * Turn an array of points into an unsorted quadtree of nodes.
 * You'll probably want to call sortify() to sort the list into a
 * useful quadtree.
 *
 * @param[out] depth the depth of the quadtree. This is important for
 *             a linear quadtree, as it signifies the number of
 *             identifying bit-pairs preceding the node
 *
 * @return a new array representing the unsorted nodes of the quadtree.
 */
unsigned char* nodify(point* points, size_t len, 
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

//  cout << "arraylen " << arrayLen << endl;
  

  unsigned char* array = new unsigned char[arrayLen];

  for(size_t i = 0, end = len; i != end; ++i) {

    const size_t pointPos = fullPointLen * i;
    unsigned char* thisArrayPoint = &array[pointPos];
    point* thisPoint = &points[i];

//    cout << "pointpos " << pointPos << endl;

    ord_t currentXStart = xstart;
    ord_t currentXEnd = xend;
    ord_t currentYStart = ystart;
    ord_t currentYEnd = yend;
    for(size_t j = 0, jend = *depth; j != jend; ++j) {
      const size_t currentLocationByte = j % 4;
      const size_t bitsPerLocation = 2;
      const size_t bit1 = thisPoint->y > (currentYStart + (currentYEnd - currentYStart) / 2);
      const size_t bit2 = thisPoint->x > (currentXStart + (currentXEnd - currentXStart) / 2);
      const size_t currentPosBits = (bit1 << 1) | bit2;

//      cout << "j/4 " << j/4 << endl;
//      cout << "endianswap " << ENDIANSWAP(j/4) << endl;
//      cout << "arrayval " << pointPos + ENDIANSWAP(j/4) << endl;
//      cout << "prevArrayVal " << pointPos + j/4 << endl;

      size_t ebyte = j / 4;
//      ebyte = (ebyte / 4 * 4) + ENDIANSWAP(ebyte % 4);

      thisArrayPoint[ebyte] = (thisArrayPoint[ebyte] << bitsPerLocation) | currentPosBits;
      
      const ord_t newWidth = (currentXEnd - currentXStart) / 2;
      const ord_t pointRight = thisPoint->x - currentXStart;
      const ord_t pointRightRound = floor(pointRight / newWidth) * newWidth;
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
  const size_t charsPerSortT = sizeof(sort_t);
  const size_t sortDepths = ceil((depth / 4) / (double)sizeof(sort_t));

  bool swapped = true;
  while(swapped) { // bubble sort - will iterate a maximum of n times
    swapped = false;
    for(size_t i = 0, end = len * fullPointLen; i < end; i += fullPointLen) { //must be < not !=
      if(i + fullPointLen >= len * fullPointLen)
        break; // last point

      unsigned char* point = &array[i];
      unsigned char* nextPoint = &array[i + fullPointLen];

      const sort_t* pointAsNum = (sort_t*)point;
      const sort_t* nextPointAsNum = (sort_t*)nextPoint;
      
      for(size_t j = 0, jend = sortDepths; j < jend; ++j) { // must be < not !=
//        const sort_t key = *((unsigned int*)&point[j]);
//        const sort_t nextKey = *((unsigned int*)&nextPoint[j]);
        const sort_t key = __builtin_bswap32(pointAsNum[j]);
        const sort_t nextKey = __builtin_bswap32(nextPointAsNum[j]);
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
 * swap the memory of the given quadtree points
 */
void swapify(unsigned char* firstPoint, unsigned char* secondPoint, const size_t depth) {
  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;

  unsigned char* temp = new unsigned char[fullPointLen];
  memcpy(temp, firstPoint, fullPointLen);
  memcpy(firstPoint, secondPoint, fullPointLen);
  memcpy(secondPoint, temp, fullPointLen);
  delete[] temp;
}

/*
 * print out a quadtree node
 * @param depth the quadtree depth. Necessary, because it indicates
 *              the number of position bit-pairs
 */
void printNode(unsigned char* node, const size_t depth, const bool verbose) {
  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;
//  const size_t arrayLen = fullPointLen * len;

  if(verbose)
  {
    for(size_t i = 0, end = ceil(depth/4); i != end; ++i) {
      const unsigned char& thisByte = node[i];
      cout << ((thisByte & 0x80) == 0 ? 0 : 1);
      cout << ((thisByte & 0x40) == 0 ? 0 : 1);
      cout << " ";
      cout << ((thisByte & 0x20) == 0 ? 0 : 1);
      cout << ((thisByte & 0x10) == 0 ? 0 : 1);
      cout << " ";
      cout << ((thisByte & 0x8) == 0 ? 0 : 1);
      cout << ((thisByte & 0x4) == 0 ? 0 : 1);
      cout << " ";
      cout << ((thisByte & 0x2) == 0 ? 0 : 1);
      cout << ((thisByte & 0x1) == 0 ? 0 : 1);
      cout << " ";
    }
  }

  typedef unsigned int sort_t;
  const size_t charsPerSortT = sizeof(sort_t);
  const size_t sortDepths = ceil((depth / 4) / (double)sizeof(sort_t));
  const sort_t* pointAsNum = (sort_t*)node;

//  const size_t lastTrail = (depth / 4) % sizeof(sort_t);
//  cout << endl;
//  cout << endl << "charsPerSortT " << charsPerSortT << endl;

//  cout << endl;
//  cout << "depth/4 " << depth / 4 << endl;
//  cout << "sizeof(sort_t) " << sizeof(sort_t) << endl;
//  cout << "depth/4 / sizeof(sort_t) " << depth / 4 / sizeof(sort_t) << endl;
//  cout << "sortDepths " << sortDepths << endl;

//  cout << "lastTrail " << lastTrail << endl;

  if(verbose)
  {
    for(size_t j = 0, jend = sortDepths; j < jend; ++j) { // must be <
      const sort_t key = __builtin_bswap32(pointAsNum[j]);
      cout << key << " ";
    }
  }

  // mod comes later
//  sort_t lastMod = 0;
//  for(size_t i = 0, end = lastTrail; i < end; ++i) { // must be < 
//    lastMod = lastMod << CHAR_BIT;
//    lastMod += UCHAR_MAX;
//  }
//  lastMod = lastMod ^ std::numeric_limits<sort_t>::max();
//  const sort_t lastKey = pointAsNum[sortDepths] & lastMod;
//  const sort_t lastKeyU = __builtin_bswap32(pointAsNum[sortDepths]);

//  cout << endl;
//  cout << "lastMod " << lastMod << endl;
//  cout << "lastKey " << lastKey << endl;
//  cout << "lastKeyUnmodded " << lastKeyU << endl;
//  cout << "firstChar " << (sort_t)node[0] << endl;
//  cout << "secondChar " << (sort_t)node[1] << endl;
//  cout << "sortTSize " << sizeof(sort_t) << endl;

  const size_t pointXPos = locationLen;
  const size_t pointYPos = pointXPos + sizeof(ord_t);
  const size_t keyPos = pointYPos + sizeof(ord_t);

  const ord_t* arrayPointX = (ord_t*)&node[pointXPos];
  const ord_t* arrayPointY = (ord_t*)&node[pointYPos];
  const key_t* arrayPointKey = (key_t*)&node[keyPos];

  cout << std::fixed << std::setprecision(15);
  cout << *arrayPointX << "\t" << *arrayPointY << "\t" << *arrayPointKey << endl;
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

  cout << "linear quadtree: " << endl;
  if(verbose) {
    for(size_t i = 0, end = ceil(depth/4); i < end; ++i) {
      cout << "            ";
    }
  }

  cout << "x\ty\tkey" << endl;
  for(size_t i = 0, end = len; i < end; i += fullPointLen) { // must be < not !=
    printNode(&array[i], depth, false);
  }
  cout << endl;
}

} // namespace linear_quadtree

#undef ENDIANSWAP

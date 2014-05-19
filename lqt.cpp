#include "lqt.h"
#include <iostream>
#include <cmath>

namespace {
using std::cout;
using std::endl;
}

namespace linear_quadtree {

char* nodify(point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth) {
  // depth must evenly divide 4
  *depth = 8; ///< @todo dynamically calculate depth
  const size_t locationLen = ceil(*depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;
  const size_t arrayLen = fullPointLen * len;

/*
  // debug
  cout << "numPoints: " << len << endl;
  cout << "depth: " << *depth << endl;
  cout << "locationLen: " << locationLen << endl;
  cout << "pointLen: " << pointLen << endl;
  cout << "fullPointLen: " << fullPointLen << endl;
  cout << "arrayLen: " << arrayLen << endl;
*/

  char* array = new char[arrayLen];

  for(size_t i = 0, end = len; i != end; ++i) {
//    cout << endl << "## I LOOP ## " << i << endl;

    const size_t pointPos = fullPointLen * i;
    char* thisArrayPoint = &array[pointPos];
    point* thisPoint = &points[i];

//    cout << "pointPos: " << pointPos << endl;
//    cout << "thisArrayPoint: " << thisPoint->x << ',' << thisPoint->y << '|' << thisPoint->key << endl;
    
    ord_t currentXStart = xstart;
    ord_t currentXEnd = xend;
    ord_t currentYStart = ystart;
    ord_t currentYEnd = yend;
    for(size_t j = 0, jend = *depth; j != jend; ++j) {
      /*
      cout << "%% J LOOP %% " << j << endl;
      cout << "cxs: " << currentXStart << endl;
      cout << "cxe: " << currentXEnd << endl;
      cout << "cys: " << currentYStart << endl;
      cout << "cye: " << currentYEnd << endl;
      cout << "--" << endl;
      */
      const size_t currentLocationByte = j % 4;
      const size_t bitsPerLocation = 2;
      const size_t bit1 = thisPoint->y > (currentYStart + (currentYEnd - currentYStart) / 2);
      const size_t bit2 = thisPoint->x > (currentXStart + (currentXEnd - currentXStart) / 2);
      const size_t currentPosBits = (bit1 << 1) | bit2;
      /*
      cout << "clByte: " << currentLocationByte << endl;
      cout << "bit1: " << bit1 << endl;
      cout << "bit2: " << bit2 << endl;
      cout << "currentPosBits: " << currentPosBits << endl;
      */
      thisArrayPoint[j/4] = (thisArrayPoint[j/4] << bitsPerLocation) | currentPosBits;

      const ord_t newWidth = (currentXEnd - currentXStart) / 2;
//      cout << "newWidth: " << newWidth << endl;
      const ord_t pointRight = thisPoint->x - currentXStart;
      const ord_t pointRightRound = floor(pointRight / newWidth) * newWidth;
//      cout << "pointRight: " << pointRight << endl;
//      cout << "pointRightRound: " << pointRightRound << endl;
      currentXStart = floor((thisPoint->x - currentXStart) / newWidth) * newWidth + currentXStart;
      currentXEnd = currentXStart + newWidth;

      const ord_t newHeight = (currentYEnd - currentYStart) / 2;
//      cout << "newHeight: " << newHeight << endl;
      currentYStart = floor((thisPoint->y - currentYStart) / newHeight) * newHeight + currentYStart;
      currentYEnd = currentYStart + newHeight;
    }
//    cout << "posNumber: " << *((unsigned int*)thisArrayPoint) << endl;

    const size_t pointXPos = locationLen;
    const size_t pointYPos = pointXPos + sizeof(ord_t);
    const size_t keyPos = pointYPos + sizeof(ord_t);
    /*
    cout << "pointXPos: " << pointXPos << endl;
    cout << "pointYPos: " << pointYPos << endl;
    cout << "keyPos: " << keyPos << endl;
    */
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

void printNode(char* node, const size_t depth) {
  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;
//  const size_t arrayLen = fullPointLen * len;

  for(size_t i = 0, end = ceil(depth/4); i != end; ++i) {
    const char& thisByte = node[i];
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

  const size_t pointXPos = locationLen;
  const size_t pointYPos = pointXPos + sizeof(ord_t);
  const size_t keyPos = pointYPos + sizeof(ord_t);


/*
  cout << endl << "locationLen: " << locationLen << endl;
  cout << endl << "pointLen: " << pointLen << endl;
  cout << endl << "fullPointLen: " << fullPointLen << endl;
  cout << endl << "pointXPos: " << pointXPos << endl;
  cout << endl << "pointYPos: " << pointYPos << endl;
  cout << endl << "keyPos: " << keyPos << endl;
*/

  const ord_t* arrayPointX = (ord_t*)&node[pointXPos];
  const ord_t* arrayPointY = (ord_t*)&node[pointYPos];
  const key_t* arrayPointKey = (key_t*)&node[keyPos];

  cout << *arrayPointX << "\t" << *arrayPointY << "\t" << *arrayPointKey << endl;
}

void printNodes(char* array, const size_t len, const size_t depth) {
  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;

  cout << "linear quadtree:" << endl;
  for(size_t i = 0, end = ceil(depth/4); i < end; ++i) {
    cout << "            ";
  }

  cout << "x\ty\tkey" << endl;
  for(size_t i = 0, end = len; i < end; i += fullPointLen) {
    printNode(&array[i], depth);
  }
  cout << endl;
}

} // namespace linear_quadtree

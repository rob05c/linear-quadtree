#include "lqt.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

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

void sortify_bubble(unsigned char* array, const size_t len, const size_t depth) {
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

struct rs_list_node {
  unsigned char* value;
  struct rs_list_node* next;
};

struct rs_list {
  struct rs_list_node* head;
  struct rs_list_node* tail;
};
void rs_list_insert(struct rs_list* l, const unsigned char* val, const size_t val_len) {
  struct rs_list_node* n = (struct rs_list_node*)malloc(sizeof(struct rs_list_node));
  n->value = malloc(val_len);
  memcpy(n->value, val, val_len);
  n->next = NULL;
  if(l->head == NULL) {
    l->head = n;
    l->tail = n;
    return;
  }
  l->tail->next = n;
  l->tail = n;
}
void rs_list_init(struct rs_list* l) {
  l->head = NULL;
  l->tail = NULL;
}
void rs_list_clear(struct rs_list* l) {
  for(struct rs_list_node* node = l->head; node;) {
    struct rs_list_node* toDelete = node;
    node = node->next;
    free(toDelete->value);
    free(toDelete);
  }
  l->head = NULL;
  l->tail = NULL;
}

/// @todo change this to not be global
#define BASE 10 
#define MULT_WILL_OVERFLOW(a, b, typemax) ((b) > (typemax) / (a))

/// As above, depth MUST be a multiple of 32 => the position code MUST
/// be a multiple of 64.
/// @todo fix this to work for depths > 32
void sortify_radix(unsigned char* array, const size_t len, const size_t depth) {
  typedef uint64_t sort_t;
  const sort_t sort_t_max = ~0ULL;

  const size_t locationLen = ceil(depth / 4ul);
  const size_t pointLen = sizeof(ord_t) + sizeof(ord_t) + sizeof(key_t);
  const size_t fullPointLen = locationLen + pointLen;

  struct rs_list buckets[BASE];
  for(int i = 0, end = BASE; i != end; ++i) 
    rs_list_init(&buckets[i]);

  const sort_t max = sort_t_max; ///< @todo pass max? iterate to find?

//  fprintf(stderr, "sr: max: %lu\n", max);

  int i;
  for(sort_t n = 1; max / n > 0; n *= BASE) {
//    fprintf(stderr, "sr: base %lu\n", n);
    // sort list of numbers into buckets
    for(i = 0; i < len; ++i) {
      const size_t pointPos = fullPointLen * i;
      unsigned char* thisArrayPoint = &array[pointPos];
      const sort_t location = *((sort_t*)thisArrayPoint);
      // replace array[i] in bucket_index with position code
      const size_t bucket_index = (location / n) % BASE;
      rs_list_insert(&buckets[bucket_index], thisArrayPoint, fullPointLen);
    }

    // merge buckets back into list
    for(int k = i = 0; i < BASE; rs_list_clear(&buckets[i++])) {
      for(struct rs_list_node* j = buckets[i].head; j != NULL; j = j->next) {
        memcpy(array + k * fullPointLen, j->value, fullPointLen);
        ++k;
      }
    }
    if(MULT_WILL_OVERFLOW(n, BASE, sort_t_max))
      break;
  }
  for(int i = 0, end = BASE; i != end; ++i) 
    rs_list_clear(&buckets[i]);
}

/*
 * Sort an unsorted linear quadtree. Unsorted linear quadtrees aren't
 * very useful.
 *
 * @param array unsorted linear quadtree
 * @param len   number of points in the quadtree
 * @param depth depth of the quadtree. 
 */
void sortify(unsigned char* array, const size_t len, const size_t depth) {
  sortify_bubble(array, len, depth);
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

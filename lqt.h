#ifndef lqtH
#include <stdlib.h>
#include <stdbool.h>

typedef int key_t;
typedef float ord_t; // ordinate. There's only one, so it's not a coordinate.
struct point {
  ord_t x;
  ord_t y;
  key_t key;
};

unsigned char* nodify(struct point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth);

void sortify(unsigned char* array, const size_t len, const size_t depth);
void swapify(unsigned char* firstPoint, unsigned char* secondPoint, const size_t depth);

void printNode(unsigned char* node, const size_t depth, const bool verbose);
void printNodes(unsigned char* array, const size_t len, const size_t depth, const bool verbose);

#endif

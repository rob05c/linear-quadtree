#ifndef lqtH

#include <bitset>

namespace linear_quadtree {

typedef int key_t;
typedef double ord_t; // ordinate. There's only one, so it's not a coordinate.
struct point {
  ord_t x;
  ord_t y;
  key_t key;
};

char* nodify(point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth);

void sortify(char* array, const size_t len, const size_t depth);
void swapify(char* firstPoint, char* secondPoint, const size_t depth);

void printNode(char* node, const size_t depth);
void printNodes(char* array, const size_t len, const size_t depth);

} // namespace linear_quadtree
#endif

#ifndef lqtH

#include <bitset>

namespace linear_quadtree {

typedef int key_t;
typedef float ord_t; // ordinate. There's only one, so it's not a coordinate.
struct point {
  ord_t x;
  ord_t y;
  key_t key;
};

/// @param[out] depth the depth of the quadtree. This is important for a linear 
///             quadtree, as it signifies the number of identifying bit-pairs preceding t he node
/// @return a new array representing the unsorted nodes of the quadtree.
char* nodify(point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth);

void printNode(char* node, const size_t depth);
void printNodes(char* array, const size_t len, const size_t depth);

} // namespace linear_quadtree
#endif

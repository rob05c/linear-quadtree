#ifndef lqtH
#define lqtH
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

// nvcc is C++, not C
#ifdef __cplusplus
extern "C" {
#endif

typedef int key_t;
typedef float ord_t; // ordinate. There's only one, so it's not a coordinate.
struct lqt_point {
  ord_t x;
  ord_t y;
  key_t key;
};

typedef uint64_t location_t;
extern const location_t location_t_max;



struct linear_quadtree {
  location_t*       locations;
  struct lqt_point* points;
  size_t            length;
};

#define LINEAR_QUADTREE_DEPTH (sizeof(location_t) * CHAR_BIT / 2)
void linear_quadtree_copy(struct linear_quadtree* destination, struct linear_quadtree* source);
void delete_linear_quadtree(struct linear_quadtree);

struct linear_quadtree nodify(struct lqt_point* points, size_t len, 
                              ord_t xstart, ord_t xend, 
                              ord_t ystart, ord_t yend,
                              size_t* depth);

struct linear_quadtree cuda_nodify(struct lqt_point* points, size_t len, 
                      ord_t xstart, ord_t xend, 
                      ord_t ystart, ord_t yend,
                      size_t* depth);

void cuda_sortify(struct linear_quadtree);

void sortify(struct linear_quadtree);
void sortify_radix(struct linear_quadtree);
void sortify_bubble(struct linear_quadtree);

void printNode(const location_t* location, const struct lqt_point* point, const bool verbose);
void printNodes(struct linear_quadtree lqt, const bool verbose);

void test_cub();

#ifdef __cplusplus
}
#endif
#endif

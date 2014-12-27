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

struct linear_quadtree lqt_create(struct lqt_point* points, size_t len, 
                                  ord_t xstart, ord_t xend, 
                                  ord_t ystart, ord_t yend,
                                  size_t* depth);
struct linear_quadtree lqt_nodify(struct lqt_point* points, size_t len, 
                                  ord_t xstart, ord_t xend, 
                                  ord_t ystart, ord_t yend,
                                  size_t* depth);
struct linear_quadtree lqt_sortify(struct linear_quadtree);

struct linear_quadtree lqt_create_cuda(struct lqt_point* points, size_t len, 
                                       ord_t xstart, ord_t xend, 
                                       ord_t ystart, ord_t yend,
                                       size_t* depth);
struct linear_quadtree lqt_create_cuda_slow(struct lqt_point* points, size_t len, 
                                            ord_t xstart, ord_t xend, 
                                            ord_t ystart, ord_t yend,
                                            size_t* depth);
struct linear_quadtree lqt_nodify_cuda(struct lqt_point* points, size_t len, 
                                       ord_t xstart, ord_t xend, 
                                       ord_t ystart, ord_t yend,
                                       size_t* depth);
struct linear_quadtree lqt_sortify_cuda(struct linear_quadtree);

void lqt_copy(struct linear_quadtree* destination, struct linear_quadtree* source);
void lqt_delete(struct linear_quadtree);
void lqt_print_node(const location_t* location, const struct lqt_point* point, const bool verbose);
void lqt_print_nodes(struct linear_quadtree lqt, const bool verbose);

struct linear_quadtree_cuda {
  struct lqt_point* points;
  location_t*       cuda_locations;
  struct lqt_point* cuda_points;
  size_t            length;
};
struct linear_quadtree_cuda lqt_nodify_cuda_mem(struct lqt_point* points, size_t len, 
                                                ord_t xstart, ord_t xend, 
                                                ord_t ystart, ord_t yend,
                                                size_t* depth);
struct linear_quadtree lqt_sortify_cuda_mem(struct linear_quadtree_cuda);


///
/// unified / heterogeneous
///
struct lqt_unified_node {
  location_t        location;
  struct lqt_point  point;
};

struct linear_quadtree_unified {
  struct lqt_unified_node* nodes;
  size_t                   length;
};
void lqt_delete_unified(struct linear_quadtree_unified);

struct linear_quadtree_unified lqt_nodify_cuda_unified(struct lqt_point* points, size_t len, 
                                                       ord_t xstart, ord_t xend, 
                                                       ord_t ystart, ord_t yend,
                                                       size_t* depth);
struct linear_quadtree_unified tbb_sortify_unified(struct linear_quadtree_unified lqt, const size_t threads);

struct linear_quadtree_unified lqt_create_heterogeneous(struct lqt_point* points, size_t len, 
                                                        ord_t xstart, ord_t xend, 
                                                        ord_t ystart, ord_t yend,
                                                        size_t* depth, const size_t threads);
struct linear_quadtree_unified lqt_create_sisd(struct lqt_point* points, size_t len, 
                                                        ord_t xstart, ord_t xend, 
                                                        ord_t ystart, ord_t yend,
                                                        size_t* depth, const size_t threads);

size_t tbb_num_default_thread();
void tbb_test_scheduler_init();

#ifdef __cplusplus
}
#endif
#endif

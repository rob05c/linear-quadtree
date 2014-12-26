#include "lqt.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

const location_t location_t_max = ~0ULL;

void lqt_delete(struct linear_quadtree q) {
  free(q.locations);
  free(q.points);
}

/// @param points points to construct a quadtree from. Takes ownership. MUST be dynamically allocated
/// @return linear quadtree. Caller takes ownership and must call lqt_delete()
struct linear_quadtree lqt_create(struct lqt_point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth) {
  return lqt_sortify(lqt_nodify(points, len, xstart, xend, ystart, yend, depth));
}
/* 
 * Turn an array of points into an unsorted quadtree of nodes.
 * You'll probably want to call sortify() to sort the list into a
 * useful quadtree.
 *
 * @param points points to create a quadtree from. Takes ownership. MUST be dynamically allocated
 *
 * @param[out] depth the depth of the quadtree. This is important for
 *             a linear quadtree, as it signifies the number of
 *             identifying bit-pairs preceding the node
 *
 * @return a new unsorted linear_quadtree. caller takes ownership, and must call lqt_delete()
 */
struct linear_quadtree lqt_nodify(struct lqt_point* points, size_t len, 
             ord_t xstart, ord_t xend, 
             ord_t ystart, ord_t yend,
             size_t* depth) {
  *depth = LINEAR_QUADTREE_DEPTH;

  struct linear_quadtree lqt;
  lqt.locations = malloc(sizeof(location_t) * len);
  memset(lqt.locations, 0, sizeof(location_t) * len);
  lqt.points = points;
  lqt.length = len;

  for(size_t i = 0, end = len; i != end; ++i) {
    struct lqt_point* thisPoint = &lqt.points[i];

    ord_t currentXStart = xstart;
    ord_t currentXEnd = xend;
    ord_t currentYStart = ystart;
    ord_t currentYEnd = yend;
    for(size_t j = 0, jend = *depth; j != jend; ++j) {
      const location_t bit1 = thisPoint->y > (currentYStart + (currentYEnd - currentYStart) / 2);
      const location_t bit2 = thisPoint->x > (currentXStart + (currentXEnd - currentXStart) / 2);
      const location_t currentPosBits = (bit1 << 1) | bit2;
      lqt.locations[i] = (lqt.locations[i] << 2) | currentPosBits;

      const ord_t newWidth = (currentXEnd - currentXStart) / 2;
      currentXStart = floor((thisPoint->x - currentXStart) / newWidth) * newWidth + currentXStart;
      currentXEnd = currentXStart + newWidth;
      const ord_t newHeight = (currentYEnd - currentYStart) / 2;
      currentYStart = floor((thisPoint->y - currentYStart) / newHeight) * newHeight + currentYStart;
      currentYEnd = currentYStart + newHeight;
    }
  }
  return lqt;
}

struct rs_list_node {
  location_t           location;
  struct lqt_point     point;
  struct rs_list_node* next;
};
struct rs_list {
  struct rs_list_node* head;
  struct rs_list_node* tail;
};
/// @todo determine if a location pointer is faster
void rs_list_insert(struct rs_list* l, const location_t location, const struct lqt_point* point) {
  struct rs_list_node* n = (struct rs_list_node*)malloc(sizeof(struct rs_list_node));
  n->location = location;
  n->point    = *point;
  n->next     = NULL;

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
    free(toDelete);
  }
  l->head = NULL;
  l->tail = NULL;
}

/// @todo fix this to not be global
#define BASE 10 
#define MULT_WILL_OVERFLOW(a, b, typemax) ((b) > (typemax) / (a))

// radix sort an unsorted quadtree
struct linear_quadtree lqt_sortify(struct linear_quadtree lqt) {
  struct rs_list buckets[BASE];
  for(int i = 0, end = BASE; i != end; ++i) 
    rs_list_init(&buckets[i]);

  const location_t max = location_t_max; ///< @todo pass max? iterate to find?

  int i;
  for(location_t n = 1; max / n > 0; n *= BASE) {
    // sort list of numbers into buckets
    for(i = 0; i < lqt.length; ++i) {
      const location_t location = lqt.locations[i];
      // replace array[i] in bucket_index with position code
      const size_t bucket_index = (location / n) % BASE;
      rs_list_insert(&buckets[bucket_index], location, &lqt.points[i]);
    }

    // merge buckets back into list
    for(int k = i = 0; i < BASE; rs_list_clear(&buckets[i++])) {
      for(struct rs_list_node* j = buckets[i].head; j != NULL; j = j->next) {
        lqt.locations[k] = j->location;
        lqt.points[k]    = j->point;
        ++k;
      }
    }
    if(MULT_WILL_OVERFLOW(n, BASE, location_t_max))
      break;
  }
  for(int i = 0, end = BASE; i != end; ++i) 
    rs_list_clear(&buckets[i]);
  return lqt;
}

/*
 * print out a quadtree node
 * @param depth the quadtree depth. Necessary, because it indicates
 *              the number of position bit-pairs
 */
void lqt_print_node(const location_t* location, const struct lqt_point* point, const bool verbose) {
  if(verbose)
  {
    for(int j = sizeof(location_t) * CHAR_BIT - 1, jend = 0; j >= jend; j -= 2)
      printf("%lu%lu ", (*location >> j) & 0x01, (*location >> (j - 1)) & 0x01);
    printf("%lu ", *location);
  }
  printf("%.15f\t%.15f\t%d\n", point->x, point->y, point->key);
}

/* 
 * print out all the nodes in a linear quadtree
 * @param array the linear quadtree
 * @param len the number of nodes in the quadtree
 * @param depth the depth of the quadtree.
 */
void lqt_print_nodes(struct linear_quadtree lqt, const bool verbose) {
  printf("linear quadtree: \n");
  if(verbose) {
    for(size_t i = 0, end = sizeof(location_t); i != end; ++i)
      printf("            ");
  }

  printf("x\ty\tkey\n");
  for(size_t i = 0, end = lqt.length; i != end; ++i) {
    lqt_print_node(&lqt.locations[i], &lqt.points[i], verbose);
  }
  printf("\n");
}

/// copies the tree from the source into destination.
/// caller takes ownership of destination, and must call delete_linear_quadtree()
/// does not free destination, if destination is an allocated quadtree. Call delete_linear_quadtree(destination) first.
void lqt_copy(struct linear_quadtree* destination, struct linear_quadtree* source) {
  destination->length = source->length;
  destination->locations = (location_t*) malloc(destination->length * sizeof(location_t));
  memcpy(destination->locations, source->locations, source->length * sizeof(location_t));
  destination->points = (struct lqt_point*) malloc(destination->length * sizeof(struct lqt_point));
  memcpy(destination->points, source->points, source->length * sizeof(struct lqt_point));
}

///
/// unified
///

void lqt_delete_unified(struct linear_quadtree_unified q) {
  free(q.nodes);
}

#undef ENDIANSWAP

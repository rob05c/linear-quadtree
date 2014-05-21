#include <iostream>
#include <cstdlib>
#include "lqt.h"

namespace {
using std::cout;
using std::endl;
namespace lqt = linear_quadtree;
using namespace linear_quadtree; //debug

// generate a uniform random between min and max exclusive
ord_t uniformFrand(ord_t min, ord_t max) {
  const double r = (double)rand() / RAND_MAX;
  return min + r * (max - min);
}
}

int main() {
  srand(time(NULL));

  point points[10000];
  const size_t min = 1000;
  const size_t max = 1100;
  cout << "creating points...";
  for(int i = 0, end = sizeof(points) / sizeof(point); i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  cout << "creating nodes...";
  size_t depth;
  char* unsortedQuadtree = nodify(points, sizeof(points) / sizeof(point), 
                                  min, max, min, max, &depth);

//  printNodes(unsortedQuadtree, sizeof(points), depth);

  cout << endl << "sorting...";
  sortify(unsortedQuadtree, sizeof(points) / sizeof(point), depth);
  cout << endl << "done" << endl;
  printNodes(unsortedQuadtree, sizeof(points), depth);

  return 0;
}

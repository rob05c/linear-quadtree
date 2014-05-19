#include <iostream>
#include <cstdlib>
#include "lqt.h"

namespace {
using std::cout;
using std::endl;
namespace lqt = linear_quadtree;
using namespace linear_quadtree; //debug
}

int main() {
  srand(time(NULL));
  point points[2];
  for(int i = 0, end = sizeof(points) / sizeof(point); i != end; ++i) {
    points[i].x = rand() % 100 + 1000;
    points[i].y = rand() % 100 + 1000;
    points[i].key = i;
  }

  size_t depth;
  char* unsortedQuadtree = nodify(points, sizeof(points) / sizeof(point), 
                                  1000, 1100, 1000, 1100, &depth);

  printNodes(unsortedQuadtree, sizeof(points), depth);

  return 0;
}

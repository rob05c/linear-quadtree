#ifndef testH
#define testH
#include <stdbool.h>
#include <stdlib.h>

// nvcc is C++, not C
#ifdef __cplusplus
extern "C" {
#endif

void test_matmul(const size_t matrix_size, const bool print);
  
#ifdef __cplusplus
}
#endif
#endif

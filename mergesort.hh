// Appears in Structured Parallel Programming, published by Elsevier, Inc.
#ifndef hello_hh
#define hello_hh

#include <cstdlib>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <memory>
#include "tbb/tbb.h"

namespace {
using std::vector;
using std::string;
using std::numeric_limits;
using std::cout;
using std::endl;
using std::to_string;
using std::unique_ptr;
using tbb::parallel_for;
using tbb::blocked_range;
using tbb::parallel_do;
using tbb::parallel_do_feeder;

template <typename T>
void print_array(T* array, const size_t size) {
  string outstr = "{";
  for(int i = 0, end = size; i != end; ++i)
    outstr += to_string(array[i]) + ", ";
  outstr.erase(outstr.size() - 2);
  outstr += "}";
  cout << outstr << endl;
}

template <typename T>
void serial_merge(T* first, T* first_end,
		  T* second, T* second_end, T* merged) {
  while(first != first_end && second != second_end)
    *merged++ = std::move(*second < *first ? *second++ : *first++); 
  std::move(first, first_end, merged);
  std::move(second, second_end, merged);
}

/// \param first first array to merge; must be sorted
/// \param first_len length of first
/// \param second second array to merge; must be sorted
/// \param second_len length of second
/// \param[out] merged array to put the merged arrays into. Must be at least first_len+second_len
template <typename T>
void parallel_merge(T* first, T* first_end,
		    T* second, T* second_end, T* merged) {
  const size_t SERIAL_CUTOFF = 500;
  if(first_end - first + second_end - second <= SERIAL_CUTOFF) {
    serial_merge<T>(first, first_end, second, second_end, merged);
    return;
  }

  T* first_middle;
  T* second_middle;
  if(first_end - first < second_end - second) {
    second_middle = second + (second_end - second) / 2;
    first_middle = std::upper_bound(first, first_end, *second_middle);
  } else {
    first_middle = first + (first_end - first) / 2;
    second_middle = std::upper_bound(second, second_end, *first_middle);
  }
  T* merged_middle = merged + (first_middle - first) + (second_middle - second);
  /// \todo fix all-capture ?
  tbb::parallel_invoke([=]{parallel_merge(first, first_middle, second, second_middle, merged);},
  		       [=]{parallel_merge(first_middle, first_end, second_middle, second_end, merged_middle);});
}

/// \param inplace whether this recursion is in the array or buffer. 
///                Each recurse switches between them.
template < typename T >
void mergesort_(T* array, T* array_end, T* buffer, const bool inplace) {
  const size_t SERIAL_CUTOFF = 2000;

  if(array_end - array <= SERIAL_CUTOFF) {
    std::stable_sort(array, array_end);
    if(!inplace) {
      std::move(array, array_end, buffer);
    }
    return;
  }
  T* array_middle = array + (array_end - array) / 2;
  T* buffer_middle = buffer + (array_middle - array);
  T* buffer_end = buffer + (array_end - array);
  tbb::parallel_invoke([=]{mergesort_<T>(array, array_middle, buffer, !inplace);},
                       [=]{mergesort_<T>(array_middle, array_end, buffer_middle, !inplace);});

  if(inplace) {
    parallel_merge<T>(buffer, buffer_middle, buffer_middle, buffer_end, array);
  } else {
    parallel_merge<T>(array, array_middle, array_middle, array_end, buffer);
  }
}

} // namespace

template <typename T>
T* parallel_mergesort(T* array, T* array_end, size_t threads) {
  tbb::task_scheduler_init init(threads);

  unique_ptr<T[]> buffer(new T[array_end - array]);
  mergesort_<T>(array, array_end, buffer.get(), true);
  return array;
}

#endif // hello_hh

// Appears in Structured Parallel Programming, published by Elsevier, Inc.
#ifndef quicksort_hh
#define quicksort_hh

// Size of parallel base case.
ptrdiff_t QUICKSORT_CUTOFF = 500;

// Choose median of three keys.
template <typename T>
T* median_of_three(T* x, T* y, T* z) {
  return *x<*y ? *y<*z ? y : *x<*z ? z : x 
                                : *z<*y ? y : *z<*x ? z : x;
}

// Choose a partition key as median of medians.
template <typename T>
T* choose_partition_key( T* first, T* last ) {
  size_t offset = (last-first)/8;
  return median_of_three(
    median_of_three(first, first+offset, first+offset*2),
    median_of_three(first+offset*3, first+offset*4, last-(3*offset+1)), 
    median_of_three(last-(2*offset+1), last-(offset+1), last-1 )
    );
}

// Choose a partition key and partition [first...last) with it.
// Returns pointer to where the partition key is in partitioned
// sequence.
// Returns NULL if all keys in [first...last) are equal.
template <typename T>
T* divide( T* first, T* last ) {
  // Move partition key to front.
  std::swap( *first, *choose_partition_key(first,last) );
  // Partition 
  T key = *first;
  T* middle = std::partition( first+1, last, [=](const T& x) {return x<key;} ) - 1;
  if( middle!=first ) {
    // Move partition key to between the partitions
    std::swap( *first, *middle );
  } else {
    // Check if all keys are equal
    if( last==std::find_if( first+1, last, [=](const T& x) {return key<x;} ) )
      return NULL;
  }
  return middle;
}

template <typename T>
class quicksort_task: public tbb::task {
  /*override*/tbb::task* execute();
  T *first, *last;
  bool has_local_join;
  void prepare_self_as_stealable_continuation();
public:
  quicksort_task( T* first_, T* last_ ) : first(first_), last(last_), has_local_join(false) {}
};

template <typename T>
void quicksort_task<T>::prepare_self_as_stealable_continuation() {
  if( !has_local_join ) {
    task* local_join  = new( allocate_continuation() ) tbb::empty_task();
    local_join->set_ref_count(1);
    set_parent(local_join);
    has_local_join = true;
  }
  recycle_to_reexecute();
}

template <typename T>
tbb::task* quicksort_task<T>::execute() {
  if( last-first<=QUICKSORT_CUTOFF ) {
    std::sort(first,last);
    // Return NULL continuation
    return NULL;
  } else {
    // Divide
    T* middle = divide(first,last);
    if( !middle ) return NULL; 

    // Now have two subproblems: [first..middle) and [middle+1..last)

    // Set up current task object as continuation of itself.
    prepare_self_as_stealable_continuation();

    // Now recurse on smaller subproblem.
    tbb::task* smaller;
    if( middle-first < last-(middle+1) )  {
      // Left problem (first..middle) is smaller.
      smaller = new( allocate_additional_child_of(*parent()) ) quicksort_task( first, middle );
      // Continuation will do larger subproblem
      first = middle+1;
    } else {
      // Right problem (middle..last) is smaller.
      smaller = new( allocate_additional_child_of(*parent()) ) quicksort_task( middle+1, last );
      // Continuation will do larger subproblem
      last = middle;
    }
    // Dive into smaller subproblem
    return smaller;
  }
}

template <typename T>
void parallel_quicksort( T* first, T* last ) {
  // Create root task
  tbb::task& t = *new( tbb::task::allocate_root() ) quicksort_task<T>( first, last );
  // Run it
  tbb::task::spawn_root_and_wait(t);
}

#endif

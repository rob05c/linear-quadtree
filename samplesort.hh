#ifndef samplesort_hh
#define samplesort_hh
#include <cstdlib>
#include <cassert>
#include <memory>
#include <tbb/tbb.h>
#include "quicksort.hh"

// Max number of bins.  Must not exceed 256.
const size_t M_MAX = 32; 

const size_t SAMPLE_SORT_CUT_OFF = 2000;

size_t floor_lg2( size_t n ) {
  size_t k = 0;
  for( ; n>1; n>>=1 ) 
    ++k;
  return k;
}

size_t choose_number_of_bins( size_t n ) {
  const size_t BIN_CUTOFF = 1024;
  return std::min( M_MAX, size_t(1)<<floor_lg2(n/BIN_CUTOFF));
}

typedef unsigned char bindex_type;

// Auxilary routine used by sample_sort.
template <typename T>
void repack_and_subsort( T* xs, T* xe, size_t m, const T* y, const size_t tally[M_MAX][M_MAX] ) {
  // Compute column sums of tally, thus forming the running sum of bin
  // sizes.
  size_t col_sum[M_MAX];
  std::fill_n(col_sum,m,0); 
  for( size_t i=0; i<m; ++i )
    for( size_t j=0; j<m; ++j )
      col_sum[j] += tally[i][j];
  assert( col_sum[m-1]==xe-xs );

  // Move the bins back into the original array and do the subsorts
  size_t block_size = ((xe-xs)+m-1)/m;
  tbb::parallel_for( size_t(0), m, [=,&col_sum](size_t j){ 
      T* x_bin = xs + (j==0 ? 0 : col_sum[j-1]);
      T* x = x_bin;
      for( size_t i=0; i<m; ++i ) {
        const T* src_row = y+i*block_size;
        x = std::move(src_row+(j==0?0:tally[i][j-1]), src_row+tally[i][j], x);
      }
      parallel_quicksort(x_bin,x);
    });
}

// Assumes that m is a power of 2
template <typename T>
void build_sample_tree( const T* xs, const T* xe, T tree[], size_t m ) {
  // Compute oversampling coefficient o as approximately log(xe-xs)
  assert(m<=M_MAX);
  size_t o = floor_lg2(xe-xs);
  const size_t O_MAX = 8*(sizeof(size_t));
  size_t n_sample = o*m-1;
  T tmp[O_MAX*M_MAX-1];
  size_t r = (xe-xs-1)/(n_sample-1);
  // Generate oversampling
  for( size_t i=0; i<n_sample; ++i )
    tmp[i] = xs[i*r];
  // Sort the samples
  std::sort( tmp, tmp+n_sample );
  // Select samples and put them into the tree
  size_t step = n_sample+1;
  for( size_t level=1; level<m; level*=2 ) {
    for( size_t k=0; k<level; ++k )
      tree[level-1+k] = tmp[step/2-1+k*step];
    step /= 2;
  }
}

// Set bindex[0..n) to the bin index of each key in x[0..n), using the
// given implicit binary tree with m-1 entries. 
template <typename T>
void map_keys_to_bins( const T x[], size_t n, const T tree[], size_t m, bindex_type bindex[], size_t freq[] ) {
  size_t d = floor_lg2(m);
  std::fill_n(freq,m,0);
  for( size_t i=0; i<n; ++i ) {
    size_t k = 0;
    for( size_t j=0; j<d; ++j )
      k = 2*k+2 - (x[i] < tree[k]);
    ++freq[bindex[i] = k-(m-1)];
  }
}

template <typename T>
void bin( T* xs, T* xe, size_t m, T* y, size_t tally[M_MAX][M_MAX] ) {
  T tree[M_MAX-1];
  build_sample_tree( xs, xe, tree, m );

  size_t block_size = ((xe-xs)+m-1)/m;
  bindex_type* bindex = new bindex_type[xe-xs];
  tbb::parallel_for( size_t(0), m, [=,&tree](size_t i) { 
      size_t js = i*block_size;
      size_t je = std::min( js+block_size, size_t(xe-xs) );

      // Map keys to bins
      size_t freq[M_MAX];
      map_keys_to_bins( xs+js, je-js, tree, m, bindex+js, freq );

      // Compute where each bin starts
      T* dst[M_MAX];
      size_t s = 0;
      for( size_t j=0; j<m; ++j ) {
        dst[j] = y+js+s;
        s += freq[j];
        tally[i][j] = s;
      }

      // Scatter keys into their respective bins
      for( size_t j=js; j<je; ++j )
        *dst[bindex[j]]++ = std::move(xs[j]);
    });
  delete[] bindex;
}

template <typename T>
T* parallel_samplesort( T* xs, T* xe, const size_t /*threads*/) {
//  tbb::task_scheduler_init init(threads);

  if( xe-xs<=SAMPLE_SORT_CUT_OFF ) {
    parallel_quicksort(xs,xe);
  } else {
    size_t m = choose_number_of_bins(xe-xs);
    size_t tally[M_MAX][M_MAX];
    T* y = new T[xe-xs];
    bin(xs, xe, m, y, tally);
    repack_and_subsort(xs, xe, m, y, tally);
    delete[] y;
  }
  return xs;
}

#endif

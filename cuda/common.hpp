#ifndef __FFTCUDA_COMMON_H__
#define __FFTCUDA_COMMON_H__

#include <complex>
#include <sys/time.h>
#include <cassert>
#include <cstdio>

typedef std::complex<float> cpx;
static const float pi = 3.1415926535897932384626433832795;

inline double getTime() {
  timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}
extern double tic_v;
inline void tic() {
  tic_v = getTime();
}
inline double toc() {
  return getTime() - tic_v;
}

template<typename T>
inline bool eqfloat(T a, T b, float eps1 = 1e-3, float eps2 = 1e-3) {
  if (std::abs(b) < eps1)
    return std::abs(a-b) < eps1;
  return std::abs(a-b)/std::abs(b) < eps2;
}

//redefine assert if build for torch : it doesn't work by default
#ifdef TORCH_BUILD

#ifdef assert
#undef assert
#endif

inline void assert_func(bool test, char* linename, char* file, int line) {
  if (!test) {
    fprintf(stderr, "Assertion failed : in file %s, at line %d : %s\n",
	    file, line, linename);
    exit(0);
  }
}
#define fft_assert(x) (assert_func((x), (#x), (__FILE__), (__LINE__)))

#else
#include <cassert>
#define fft_assert(x) (assert((x)))
#endif

#endif

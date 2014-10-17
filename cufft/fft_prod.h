#ifndef __FFT_PROD_H__
#define __FFT_PROD_H__

#include <cuComplex.h>

// M is the dimension common between input and output
// F is the dimension common between input and kernel, and it is summed over
// G is the dimension common between kernel and output
// <a>s<b> are strides, were <a> can be (i)nput, (k)ernel or (o)utput
//  and <b> can be m, f or g (corresponding to these dimensions)
void fourier_prod(const cuComplex* inputF,
		  const cuComplex* kernelF,
		  cuComplex* outputF,
		  const int N,
		  const int M, const int ism, const int osm,
		  const int F, const int isf, const int ksf,
		  const int G, const int ksg, const int osg);

#endif

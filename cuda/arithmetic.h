#ifndef __ARITHMETIC_H__
#define __ARITHMETIC_H__

// integer ceil
#define DIVUP(a,b) (((a)+(b)-1)/(b))
//#define DIVUP(a,b) ((a/b)+(a%b != 0))

// fast integer power
inline int ipow(int a, int n) {
  if (n == 0)
    return 1;
  if (n == 1)
    return a;
  if (n & 1) // if ((i % 2) != 0)
    return ipow(a, n-1) * a;
  int b = ipow(a, n/2);
  return b*b;
}

#endif

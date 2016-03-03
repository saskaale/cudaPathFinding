#ifndef __MATRIX_TOOOLS_H_41242432412431234
#define __MATRIX_TOOOLS_H_41242432412431234


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>




void allocMem(int& size, int ** &mtx);
void emptyMem(int& size, int ** &mtx);
void load(std::istream& s, int& size, int ** &mtx);
void dump(std::ostream& os, int& size, int ** &mtx);








const int MAXLEN = INT_MAX/3;


#endif
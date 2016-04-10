#ifndef __MATRIX_TOOOLS_H_41242432412431234
#define __MATRIX_TOOOLS_H_41242432412431234


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>


#define MATRIX_AT(i,j) mtx[(i)*size+(j)]
#define MATRIX_SIZE() ((size)*(size))

void allocMem(int size, int * &mtx);
void emptyMem(int size, int * &mtx);
void load(std::istream& s, int& size, int * &mtx);
void randomMtx(int wantedSize, int& size, int * &mtx);
void dump(std::ostream& os, int& size, int * &mtx);








const int MAXLEN = INT_MAX/3;


#endif
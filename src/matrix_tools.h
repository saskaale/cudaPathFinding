#ifndef __MATRIX_TOOOLS_H_41242432412431234
#define __MATRIX_TOOOLS_H_41242432412431234


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>


#ifndef MATRIX_BLOCKSIZE
#define MATRIX_BLOCKSIZE 8
#endif

//#define MATRIX_AT(i,j) mtx[(i)*(size+MATRIX_BLOCKSIZE)+(j)]
//#define MATRIX_SIZE() ((size+MATRIX_BLOCKSIZE)*(size+MATRIX_BLOCKSIZE))

#define MATRIX_AT(i,j) mtx[(i)*(size)+(j)]
#define MATRIX_SIZE() ((size)*(size))

void allocMem(int size, int * &mtx);
void emptyMem(int size, int * &mtx);
void load(std::istream& s, int& size, int * &mtx);
void randomMtx(int wantedSize, int& size, int * &mtx);
void dump(std::ostream& os, int& size, int * &mtx);

const int MAXLEN = INT_MAX/3;


#endif

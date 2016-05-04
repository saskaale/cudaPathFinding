#ifndef __MATRIX_TOOOLS_H_41242432412431234
#define __MATRIX_TOOOLS_H_41242432412431234


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef MATRIX_BLOCKSIZE
#define MATRIX_BLOCKSIZE 8
#endif

#define MATRIX_AT(i,j) mtx[(i)*(size+MATRIX_BLOCKSIZE)+(j)]
#define MATRIX_SIZE() ((size+MATRIX_BLOCKSIZE)*(size+MATRIX_BLOCKSIZE))

void allocMem(int size, int * &mtx);
void emptyMem(int size, int * &mtx);
void load(std::istream& s, int& size, int * &mtx);
void randomMtx(int wantedSize, int& size, int * &mtx);
void dump(std::ostream& os, int& size, int * &mtx);

static void HandleError( cudaError_t err, const char* file, int line);


#define HANDLE_ERROR(err) (HandleError((err), __FILE__, __LINE__))







const int MAXLEN = INT_MAX/3;


#endif

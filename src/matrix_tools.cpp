#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>

#include "matrix_tools.h"


using namespace std;

static void HandleError( cudaError_t err, const char* file, int line) {
    if( err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void load(istream& s, int& size, int * &mtx){ 
    cin >> size;

    allocMem(size, mtx);

    for(int i = 0; i < size; i++){
	for(int j = 0; j < size; j++){
	    int dist = 0;
	    cin >> dist;
	    MATRIX_AT(i,j) = dist;
	}
    }
}


void randomMtx(int wantedSize, int& size, int * &mtx){
    size = wantedSize;

    allocMem(size, mtx);



   time_t t;
   /* Intializes random number generator */
   srand((unsigned) time(&t));

    for(int i = 0; i < size; i++){
	for(int j = 0; j < size; j++){
	    int dist = rand()%MAXLEN;
	    MATRIX_AT(i,j) = dist;
	}
    }
}

void dump(ostream& os, int& size, int * &mtx){
  os << size<<endl;
  for(int i = 0; i < size; i++){
      for(int j = 0; j < size; j++){
	  if(MATRIX_AT(i,j) >= MAXLEN)
	    os << "-1";
	  else
	    os << MATRIX_AT(i,j);
	  
	  os << '\t';
      }
      os << endl;
  }
  
}

void emptyMem(int size, int * &mtx){
    delete[] mtx;
}

void allocMem(int size, int * &mtx){
    int s = MATRIX_SIZE();
//    mtx = new int[s];
    HANDLE_ERROR( cudaHostAlloc((void**)&mtx, s*sizeof(int), cudaHostAllocDefault) );
    for(int i = 0; i < s; i++)
	mtx[i] = MAXLEN;
}
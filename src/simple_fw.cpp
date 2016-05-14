#include <cstdio>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <climits>

#include "matrix_tools.h"


using namespace std;

void do_FW();



int size;
int * mtx;

#define HANDLE_ERROR(err) (HandleError((err), __FILE__, __LINE__))


void do_FW(){
  //prepare matrix of length
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++){
      if(MATRIX_AT(i,j) < 0){	//not an edge
	MATRIX_AT(i,j) = MAXLEN;
      }
    }
  }
  
  //length from i to i is 0
  for(int i = 0; i < size; i++){
    MATRIX_AT(i,i) = 0;
  }
  
  
  //Floyd Warshall main loop
  for(int k = 0; k < size; k++){
    for(int i = 0; i < size; i++){
      for(int j = 0; j < size; j++){
	int other = MATRIX_AT(i,k) + MATRIX_AT(k,j);
	if(MATRIX_AT(i,j) > other){
	  MATRIX_AT(i,j) = other;
	}
      }
    }
  }

}


int main(int argc, char ** argv){
    randomMtx(atoi(argv[1]), size, mtx);
    //load(cin, size, mtx);
    
    do_FW();
    
    dump(cout, size, mtx);
    
    emptyMem(size, mtx);
}

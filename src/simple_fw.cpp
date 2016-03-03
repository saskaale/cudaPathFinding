#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>

#include "matrix_tools.h"


using namespace std;

void do_FW();



int size;
int ** mtx;




void do_FW(){
  //prepare matrix of length
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++){
      if(mtx[i][j] < 0){	//not an edge
	mtx[i][j] = MAXLEN;
      }
    }
  }
  
  //length from i to i is 0
  for(int i = 0; i < size; i++){
    mtx[i][i] = 0;
  }
  
  //Floyd Warshall main loop
  for(int k = 0; k < size; k++){
    for(int i = 0; i < size; i++){
      for(int j = 0; j < size; j++){
	int other = mtx[i][k] + mtx[k][j];
	if(mtx[i][j] > other){
	  mtx[i][j] = other;
	}
      }
    }
  }
}


int main(){
    load(cin, size, mtx);
    
    do_FW();
    
    dump(cout, size, mtx);
    
    emptyMem(size, mtx);
}
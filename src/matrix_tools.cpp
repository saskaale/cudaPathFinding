#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>

#include "matrix_tools.h"


using namespace std;

void load(istream& s, int& size, int ** &mtx){ 
    cin >> size;
    
    allocMem(size, mtx);


    for(int i = 0; i < size; i++){
	for(int j = 0; j < size; j++){
	    int dist = 0;
	    cin >> dist;
	    mtx[i][j] = dist;
	}
    }
}

void dump(ostream& os, int& size, int ** &mtx){
  os << size<<endl;
  for(int i = 0; i < size; i++){
      for(int j = 0; j < size; j++){
	  if(mtx[i][j] >= MAXLEN)
	    os << "-1";
	  else
	    os << mtx[i][j];
	  
	  os << '\t';
      }
      os << endl;
  }
  
}

void emptyMem(int& size, int ** &mtx){
    for(int i = 0; i < size; i++){
      delete[] mtx[i];
    }
    delete[] mtx;
}

void allocMem(int& size, int ** &mtx){
    mtx = new int*[size];
    for(int i = 0; i < size; i++){
      mtx[i] = new int[size];
    }
}
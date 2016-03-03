#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>

#include "matrix_tools.h"


using namespace std;

void do_FW();



int size;
int ** mtx;




void do_Dijkstra(){

  //TODO DIJKSTRA ALGORITHM
  
  
}


int main(){
    load(cin, size, mtx);
    
    do_Dijkstra();
    
    dump(cout, size, mtx);
    
    emptyMem(size, mtx);
}
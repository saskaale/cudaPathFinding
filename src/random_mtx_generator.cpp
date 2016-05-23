#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <string.h>

#include "matrix_tools.h"

using namespace std;

int main(int argc, char ** argv){
    int size;
    int * mtx;
    if (argc<2) exit(1);
    else randomMtx(atoi(argv[1]), size, mtx);
    
    dump(cout, size, mtx);
    
    emptyMem(size, mtx);
}
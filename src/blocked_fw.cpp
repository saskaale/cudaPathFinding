#include <cstdio>
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <climits>

//#include <cuda_runtime_api.h>
//#include <cuda.h>

#include "matrix_tools.h"


using namespace std;

void do_BlockedFW(int& size, int*& mtx);
void do_FW(int& size, int*& mtx);



int size;
int * mtx;


void do_FW(int& size, int*& mtx){
  //prepare matrix of length
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++){
      if(MATRIX_AT(i,j) < 0){   //not an edge
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


#define HANDLE_ERROR(err) (HandleError((err), __FILE__, __LINE__))

#define SOLVE_IJK(i,j,k) \
    int other = MATRIX_AT(i,k) + MATRIX_AT(k,j); \
    if(MATRIX_AT(i,j) > other){ \
	MATRIX_AT(i,j) = other; \
    }


void do_BlockedFW(int& size, int*& mtx){
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
  
  
//  int* md;
  
  //copy matrix to cuda
///  cudaMalloc((void**)&md,MATRIX_SIZE());
//  cudaMemcpy(md, mtx, MATRIX_SIZE()*sizeof(int), cudaMemcpyHostToDevice);
  
  
  int n = size;	//size of matrix

  if(n%MATRIX_BLOCKSIZE != 0){
    n += MATRIX_BLOCKSIZE - n%MATRIX_BLOCKSIZE;	//align to size of matrix_blocksize
  }


  const int s = MATRIX_BLOCKSIZE;	//size of block

  //Floyd Warshall main loop
  for(int b = 0; b < n/s; b++){
    //independent block first
    for(int k = b*s; k < (b+1)*s; k++){
	for(int i = b*s; i < (b+1)*s; i++){
	    for(int j = b*s; j < (b+1)*s; j++){
		SOLVE_IJK(i,j,k);
	    }
	}
    }
    
    //tasks in singly dependent block depend on the independent block
    for(int ib = 0; ib <= n/s; ib++){
	for(int k = b*s; k < (b+1)*s; k++){
	    for(int i = b*s; i < (b+1)*s; i++){
		for(int j = ib*s; j < (ib+1)*s; j++){
		    if(i >= size || j >= size || k >= size){
//			cout << "1: "<<i<<","<<j<<","<<k<<endl;
			continue;
		    }
		    SOLVE_IJK(i,j,k);
		}
	    }
	}
    }
    
    //j-aligned singly dependent blocks
    for(int jb = 0; jb <= n/s; jb++){
	for(int k = b*s; k < (b+1)*s; k++){
	    for(int i = jb*s; i < (jb+1)*s; i++){
		for(int j = b*s; j < (b+1)*s; j++){
		    if(i >= size || j >= size || k >= size){
//			cout << "2: "<<i<<","<<j<<","<<k<<endl;
			continue;
		    }
		    SOLVE_IJK(i,j,k);
		}
	    }
	}
    }
    
    //most blocks are doubly dependent. Notice that k is now innermost
    for(int ib = 0; ib <= n/s; ib++){
	for(int jb = 0; jb <= n/s; jb++){
	    for(int i = jb*s; i < (jb+1)*s; i++){
		for(int j = ib*s; j < (ib+1)*s; j++){
		    for(int k = b*s; k < (b+1)*s; k++){
			if(i >= size || j >= size || k >= size){
//			    cout << "3: "<<i<<","<<j<<","<<k<<endl;
			    continue;
			}
			SOLVE_IJK(i,j,k);
		    }
		}
	    }
	}
    }
  }


//  cudaMemcpy(mtx, md, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost);
//  cudaFree(md);
}



void runTests(){
    int* mtx1;
    int mtx1_size;
    int* mtx2;
    int mtx2_size;

    const int TEST_SIZE = 7;

    //prepare mtx1
    randomMtx(TEST_SIZE,mtx1_size, mtx1);

    //copy it to mtx2
    mtx2_size = mtx1_size;
    allocMem(TEST_SIZE,mtx2);

//    cout << "copy "<<(mtx1_size+MATRIX_BLOCKSIZE)*(mtx1_size+MATRIX_BLOCKSIZE)<<endl;
//    cout << "mtx1_size "<<(mtx1_size)<<endl;
    memcpy(mtx2, mtx1, sizeof(int)*(mtx1_size+MATRIX_BLOCKSIZE)*(mtx1_size+MATRIX_BLOCKSIZE));



    cout << "do Blocked" << endl;
    //apply fw cuda on mtx1
    do_BlockedFW(mtx1_size, mtx1);


    cout << "do SIMPLE"<<endl;
    //apply fw on mtx2
    do_FW(mtx2_size, mtx2);


//    dump(cout, mtx1_size, mtx1);
//    dump(cout, mtx2_size, mtx2);


    cout << "compare"<<endl;
    //compare results

    int res = memcmp(mtx1, mtx2, sizeof(int)*(mtx1_size+MATRIX_BLOCKSIZE)*(mtx1_size+MATRIX_BLOCKSIZE));
    cout << "Tests: "<<res<<endl;
    
    emptyMem(mtx2_size,mtx2);
    emptyMem(mtx1_size,mtx1);
//    emptyMem(size, mtx);
}


int main(int argc, char* argv[]){
//    runTests();
//    return 0;
//    load(cin, size, mtx);
//    cout << "Zadej velikost matice"<<endl;

    int s = atoi(argv[1]);
//    cin >> s;
    randomMtx(s,size, mtx);

    do_BlockedFW(size, mtx);

//    dump(cout, size, mtx);

    emptyMem(size, mtx);
}
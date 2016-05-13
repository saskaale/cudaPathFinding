#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <string.h>
#include <fstream>

#include "matrix_tools.h"
#include "priority_queue.hpp"

using namespace std;

int size;
int * mtx;

__global__
void DijkstraCuda(int * mtx, int * dists_gpu, int size, int start_idx){
    int source=start_idx+threadIdx.x;
    priority_queue queue(size);
    
    //not very nice - distances are also in priority_queue
    int *dist=dists_gpu+size*source;
    
    queue.update(source, 0);
    dist[source]=0;
    
    while(! queue.empty()){
        int c=queue.pop();
        
        while(! queue.empty()){
            int c=queue.pop();
            int dist_to_c=queue.get_dist(c);
            dist[c]=dist_to_c;
            for (int i=0; i<size; i++) {
                if (MATRIX_AT(c,i)<0) continue; //not a neighbour
                int new_dist_to_i=dist_to_c+MATRIX_AT(c,i);
                if (new_dist_to_i<queue.get_dist(i)) {
                    queue.update(i, new_dist_to_i);
                }
            }
        }
    }
}

#define USE_CUDA 1
#define BURST_SIZE 1024
int main(int argc, char ** argv){
    if (argc<2) load(cin, size, mtx);
    else randomMtx(atoi(argv[1]), size, mtx);
    //int * mtx_backup=new int[MATRIX_SIZE()];
    //memcpy(mtx_backup, mtx, MATRIX_SIZE()*sizeof(int));
    
    //dump(cout, size, mtx);
#if USE_CUDA
    int * mtx_gpu;
    int * dists_gpu;
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaMalloc(&mtx_gpu, MATRIX_SIZE()*sizeof(int));
    cudaMemcpy(mtx_gpu, mtx, MATRIX_SIZE()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dists_gpu, MATRIX_SIZE()*sizeof(int));
    for (int block=0; block<size/BURST_SIZE; block++) {
        DijkstraCuda<<<1, BURST_SIZE>>>(mtx_gpu, dists_gpu, size, block*BURST_SIZE);
        cudaDeviceSynchronize();
    }
    if (size%BURST_SIZE!=0){
        DijkstraCuda<<<1, size%BURST_SIZE>>>(mtx_gpu, dists_gpu, size, size-(size%BURST_SIZE));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(mtx, dists_gpu, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(mtx_gpu);
    cudaFree(dists_gpu);
#else /*USE_CUDA*/
    do_Dijkstra();
#endif /*USE_CUDA*/
//     ofstream f1("dijkstra_cuda.txt");
//     dump(f1, size, mtx);
//     f1.close();

    /*delete [] mtx;
    mtx=mtx_backup;

    do_Dijkstra();
    ofstream f2("dijkstra.txt");
    dump(f2, size, mtx);
    f2.close();*/
    //printf("mtx size %d\n", size);
    
    dump(cout, size, mtx);
    
    emptyMem(size, mtx);
}

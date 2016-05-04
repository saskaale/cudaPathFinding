#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <string.h>
#include <fstream>

#include "matrix_tools.h"


using namespace std;

#define SWAP_HEAP_ITEM(a,b) { int _id, _pri; \
_id=a.id; _pri=a.pri; \
a.id=b.id; a.pri=b.pri; \
b.id=_id; b.pri=_pri; }

//linear search for now, I don't want to bother with binary heap now
class priority_queue{
private:
    struct heap_item{
        int id;
        int pri;
    };
    int count;
    heap_item * m_arr;
public:
    __host__ __device__ priority_queue(int cnt){
        m_arr=new heap_item[cnt];
        count=cnt;
        for (int i=0; i<count; i++) {
            m_arr[i].id=i;
            m_arr[i].pri=INT_MAX;
        }
    }
    __host__ __device__ ~priority_queue(){
        delete [] m_arr;
    }
    __host__ __device__ int pop(){
        int min=INT_MAX, min_idx;
        for (int i=0; i<count; i++) {
            if (m_arr[i].pri<min) {
                min=m_arr[i].pri;
                min_idx=i;
            }
        }
        SWAP_HEAP_ITEM(m_arr[min_idx], m_arr[count-1]);
        count--;
        return min_idx;
    }
    __host__ __device__ bool empty(){
        return (count==0);
    }
    __host__ __device__ void update(int id, int pri){
        for (int i=0; i<count; i++){
            if (m_arr[i].id==id) {
                m_arr[i].pri=pri;
                return;
            }
        }
    }
};

#define MTX_AT(i, j) mtx[i*size+j]

int size;
int * mtx;

__global__
void DijkstraCuda(int * mtx, int * dists_gpu, int size){
    int source=threadIdx.x;
    priority_queue queue(size);
    
    //not very nice - distances are also in priority_queue
    int *dist=dists_gpu+size*source;
    for (int i=0; i<size; i++) dist[i]=INT_MAX;
    
    queue.update(source, 0);
    dist[source]=0;
    
    while(! queue.empty()){
        int c=queue.pop();
        
        for (int i=0; i<size; i++) {
            if (MATRIX_AT(c,i)<0) continue; //not a neighbour
            int new_dist=dist[c]+MATRIX_AT(c,i);
            if (new_dist<dist[i]) {
                dist[i]=new_dist;
                queue.update(i, new_dist);
            }
        }
    }
}

void do_Dijkstra(){
    
    int ** dists=new int*[size];
    
    //extension of Dijkstra to find distances between all points -> run it with all points as sources
    //maybe there is some better way...
    
    for (int source=0; source<size; source++){
    
        priority_queue queue(size);
        
        //not very nice - distances are also in priority_queue
        int *dist=new int[size];
        dists[source]=dist;
        for (int i=0; i<size; i++) dist[i]=INT_MAX;
        
        queue.update(source, 0);
        dist[source]=0;
        
        while(! queue.empty()){
            int c=queue.pop();
            
            for (int i=0; i<size; i++) {
                if (MATRIX_AT(c,i)<0) continue; //not a neighbour
                int new_dist=dist[c]+MATRIX_AT(c,i);
                if (new_dist<dist[i]) {
                    dist[i]=new_dist;
                    queue.update(i, new_dist);
                }
            }
        }
    
    }
    
    //return results to mtx - I need original mtx during whole algorithm, I cannot simply write new values into it
    
    for (int i=0; i<size; i++) {
        memcpy(&(mtx[i*size]), dists[i], sizeof(int)*size);
        delete [] dists[i];
    }
    delete [] dists;
  
}

#define USE_CUDA 1
int main(){
    //load(cin, size, mtx);
    randomMtx(1000, size, mtx);
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
    DijkstraCuda<<<1, size>>>(mtx_gpu, dists_gpu, size);
    cudaDeviceSynchronize();
    cudaMemcpy(mtx, dists_gpu, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(mtx_gpu);
    cudaFree(dists_gpu);
#else /*USE_CUDA*/
    do_Dijkstra();
#endif /*USE_CUDA*/
    ofstream f1("dijkstra_cuda.txt");
    dump(f1, size, mtx);
    f1.close();

    /*delete [] mtx;
    mtx=mtx_backup;

    do_Dijkstra();
    ofstream f2("dijkstra.txt");
    dump(f2, size, mtx);
    f2.close();*/
    //printf("mtx size %d\n", size);
    
    emptyMem(size, mtx);
}

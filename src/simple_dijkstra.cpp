#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <string.h>

#include "matrix_tools.h"
#include "priority_queue.hpp"

using namespace std;



int size;
int * mtx;

void do_Dijkstra(){
    
    int ** dists=new int*[size];
    
    //extension of Dijkstra to find distances between all points -> run it with all points as sources
    //maybe there is some better way...
    
    for (int source=0; source<size; source++){
    
        priority_queue queue(size);
        
        int *dist=new int[size];
        dists[source]=dist;
        
        queue.update(source, 0);
        
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
    
    //return results to mtx - I need original mtx during whole algorithm, I cannot simply write new values into it
    for (int i=0; i<size; i++) {
        memcpy(&(mtx[i*size]), dists[i], sizeof(int)*size);
        delete [] dists[i];
    }
    delete [] dists;
  
}

int main(int argc, char ** argv){
    if (argc<2) load(cin, size, mtx);
    else randomMtx(atoi(argv[1]), size, mtx);
    
    do_Dijkstra();
    
    
    dump(cout, size, mtx);
    //printf("mtx size %d\n", size);
    
    emptyMem(size, mtx);
}

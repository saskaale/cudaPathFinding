#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <string.h>

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
    heap_item * m_arr;
    int count;
public:
    priority_queue(int cnt){
        m_arr=new heap_item[cnt];
        count=cnt;
        for (int i=0; i<count; i++) {
            m_arr[i].id=i;
            m_arr[i].pri=INT_MAX;
        }
    }
    ~priority_queue(){
        delete [] m_arr;
    }
    int pop(){
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
    bool empty(){
        return (count==0);
    }
    void update(int id, int pri){
        for (int i=0; i<count; i++){
            if (m_arr[i].id==id) {
                m_arr[i].pri=pri;
                return;
            }
        }
    }
};



int size;
int * mtx;




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


int main(){
    load(cin, size, mtx);
    
    do_Dijkstra();
    
    dump(cout, size, mtx);
    
    emptyMem(size, mtx);
}
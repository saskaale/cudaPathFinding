#ifndef PRIORITY_QUEUE_HPP_
#define PRIORITY_QUEUE_HPP_

#ifdef __NVCC__
#define HOST_AND_DEVICE __device__ __host__ 
#else /*__NVCC__*/
#define HOST_AND_DEVICE
#endif /*__NVCC__*/

//not general priority queue, specialized for this case (some operations may not cover all options)
class priority_queue{
private:
    struct heap_item{
        int id;
        int pri;
    };
    heap_item * m_arr;
    int * m_pos;
    int count;
    HOST_AND_DEVICE void swap_heap(int p1, int p2){
        heap_item tmp;
        memcpy(&tmp, &m_arr[p1], sizeof(heap_item));
        memcpy(&m_arr[p1], &m_arr[p2], sizeof(heap_item));
        memcpy(&m_arr[p2], &tmp, sizeof(heap_item));
        m_pos[m_arr[p1].id]=p1;
        m_pos[m_arr[p2].id]=p2;
//         for (int i=0; i<count; i++) {
//             if (i!=m_arr[m_pos[i]].id) {
//                 printf(" sanity check failed: %d=>%d\n", i, m_arr[m_pos[i]].id);
//                 exit(1);
//             }
//         }
    }
    HOST_AND_DEVICE inline int parent(int p){
        return (p-1)/2;
    }
    HOST_AND_DEVICE inline int left_child(int p){
        return 2*p+1;
    }
    HOST_AND_DEVICE inline int right_child(int p){
        return 2*p+2;
    }
    HOST_AND_DEVICE void heapify_up(int p){
        while (m_arr[parent(p)].pri > m_arr[p].pri){
            swap_heap(parent(p), p);
            p=parent(p);
//             if (!p) return;
        }
    }
    HOST_AND_DEVICE void heapify(int p){
        int minp;
        if (left_child(p)<count && (m_arr[left_child(p)].pri < m_arr[p].pri)) minp=left_child(p);
        else minp=p;
        if (right_child(p)<count && (m_arr[right_child(p)].pri < m_arr[minp].pri)) minp=right_child(p);
        if (minp!=p) {
            swap_heap(p, minp);
            heapify(minp);
        }
    }
public:
    HOST_AND_DEVICE priority_queue(int cnt){
        m_arr=new heap_item[cnt];
        m_pos=new int[cnt];
        count=cnt;
        for (int i=0; i<count; i++) {
            m_arr[i].id=i;
            m_arr[i].pri=INT_MAX;
            m_pos[i]=i;
        }
    }
    HOST_AND_DEVICE ~priority_queue(){
        delete [] m_arr;
        delete [] m_pos;
    }
    HOST_AND_DEVICE int pop(){
        int ret=m_arr[0].id;
        swap_heap(0, count-1);
        m_pos[ret]=count-1;
        count--;
        heapify(0);
        return ret;
    }
    HOST_AND_DEVICE bool empty(){
        return (count==0);
    }
    HOST_AND_DEVICE void update(int id, int pri){ //actually only decrease value -> heapify_up
        int pos=m_pos[id];
        m_arr[pos].pri=pri;
//         for (int i=0; i<count; i++) printf("%d(%d) ", m_arr[i].id, m_arr[i].pri);
//         printf("\n\n");
        heapify_up(pos);
    }
    HOST_AND_DEVICE inline int get_dist(int i){
        return m_arr[m_pos[i]].pri;
    }
};

#endif
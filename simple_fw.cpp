#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>



using namespace std;




void allocMem(int s);
void emptyMem();
void load(istream& s);
void dump(ostream& os);
void do_FW();








int MAXLEN = INT_MAX/3;

int size;
int ** mtx;

void load(istream& s){ 
    cin >> size;
    
    allocMem(size);


    for(int i = 0; i < size; i++){
	for(int j = 0; j < size; j++){
	    int dist = 0;
	    cin >> dist;
	    mtx[i][j] = dist;
	}
    }
}

void dump(ostream& os){
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

void emptyMem(){
    for(int i = 0; i < size; i++){
      delete[] mtx[i];
    }
    delete[] mtx;
}

void allocMem(int s){
    size = s;
    mtx = new int*[size];
    for(int i = 0; i < size; i++){
      mtx[i] = new int[size];
    }
}

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
    load(cin);
    
    do_FW();
    
    dump(cout);
    
    emptyMem();
}
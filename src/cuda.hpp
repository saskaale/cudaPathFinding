#ifndef CUDA_HPP_
#define CUDA_HPP_

#include "matrix_tools.h"

static void HandleError( cudaError_t err, const char* file, int line) {
    if( err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError((err), __FILE__, __LINE__))

#endif
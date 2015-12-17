#include <stdio.h>
#define cudaAllocDevArray(dp_,size_) do { \
    void *lptr = NULL; \
    CUDA_CALL(cudaMalloc(&lptr, size_)); \
    CUDA_CALL(cudaMemcpyToSymbol(dp_, &lptr, sizeof(void *))); \
    } while(0)

#define CUDA_CALL(x) __checkerr((x), __FILE__, __LINE__) 

__host__ static void 
__checkerr(cudaError_t e, char * file, int line)
{
    if (e != cudaSuccess){
      printf("Error %s at %s:%d\n",cudaGetErrorString(e),file,line); 
      exit(1);
    }
}

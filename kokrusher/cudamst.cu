/* This is the REAL "hello world" for CUDA!  * It takes the string "Hello ", prints it, then passes it to CUDA with an array
 * of offsets. Then the offsets are added in parallel to produce the string "World!"
 * By Ingemar Ragnemalm 2010
 */
 
/*using namespace std;*/

#include <cstdio>
#ifdef __cplusplus
extern "C" {
#endif

#include "board.h"
#include "engine.h"
#include "debug.h"
#include "move.h"

#ifdef __cplusplus
}
#endif

#include "kokrusher/cudamst.h"

//#ifdef __cplusplus
//}
//#endif

const int N = 16; 
const int blocksize = 16; 

__global__ void hello(char *a, int *b){
    a[threadIdx.x] += b[threadIdx.x];
}

void say_hello(){
    char a[N] = "Hello \0\0\0\0\0\0";
    int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    char *ad;
    int *bd;
    const int csize = N*sizeof(char);
    const int isize = N*sizeof(int);

    fprintf(stderr, a);

    cudaMalloc( (void**)&ad, csize ); 
    cudaMalloc( (void**)&bd, isize ); 
    cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
    cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( 1, 1 );
    hello<<<dimGrid, dimBlock>>>(ad, bd);
    cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
    fprintf(stderr, a);
    cudaFree( ad );
    cudaFree( bd );
}

coord_t *cuda_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive){
    say_hello();
    coord_t *coord;
    struct board b2;
    board_copy(&b2, b);
    for (int i=0; i<=19; i++){
        for (int j=0; j<=19; j++){
            coord = coord_init(i,j,board_size(&b2));
            if (board_is_valid_play(&b2, color, *coord)){
                return coord;
            }
        }
    }
    *coord = -1;
    return coord;
}

/*using namespace std;*/

#include <cuda.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "kokrusher/cuboard.h"
#include "kokrusher/uct.h"
extern "C" {
#include "board.h"
}
#include "kokrusher/cudamst.h"

#define MAX_PLAYS 400
#define N_PLAYOUTS 30

#define CUDA_CALL(x) __checkerr((x), __FILE__, __LINE__) 

#define cudaAllocDevArray(dp_,size_) do { \
    void *lptr = NULL; \
    CUDA_CALL(cudaMalloc(&lptr, size_)); \
    CUDA_CALL(cudaMemcpyToSymbol(dp_, &lptr, sizeof(void *))); \
    } while(0)

//Global board state array variables.
//Yes this code is horrible. I'm sorry.
__device__ curandState randStates[M*N];
__device__ int g_flen[M*N];
__device__ int next_alloc;
__device__ stone (*g_b)[M*N];
__device__ int (*g_tree)[4];
__device__ coord_t (*g_f)[M*N];
__device__ coord_t (*g_p)[M*N];
__device__ group_t (*g_g)[M*N];
__device__ int (*g_libs)[M*N];
__device__ unsigned char (*g_watermark)[M*N];
__device__ coord_t (*g_gi)[GROUP_KEEP_LIBS][M*N];
__device__ int g_caps[S_MAX][M*N];
__device__ char (*g_ncol)[S_MAX][M*N];

__global__ void reset_tree(enum stone color, int size, void *data){
    //call with one block of n threads where n is the number of theoretically board positions
    //ei 81 for a 9X9 board
    if (threadIdx.x == 0){
        init_tree(0, -1);
        tree_child(0) = 1;
        tree_visits(0) = 1;
        next_alloc = 1 + child_size;
    }
    cuboard_copy(data, size);
    int i = threadIdx.x + 1;
    init_tree(i, 0);
    if (cuboard_is_valid_play(color, index_2_coord(threadIdx.x, size), size)) {
        tree_visits(i) = 0;
    }
}

__global__ void harvest_data(int* data, int size) {
    *data = best_move(size);
}

// this method roughly corresponds to the Simulate method from the pseudocode
__global__ void run_sims(enum stone color, int moves, int passes,float offset, int size, void *data){
    /*cuboard_init(size);*/
    curandState myState = randStates[bid];
    enum stone cur_col = color;
    int win;
    int i;
    //where uct happens: SimTree method from pseudocode
    for (i=0; i<N_PLAYOUTS; i++) {
        cuboard_copy(data, size);
        /*int node = walk_down(&cur_col, size);*/
        float score  = cuda_play_random_game(color, myState, moves, passes, size) + offset;
	    win = ((color == S_WHITE) ^ (score < 0));
        /*backup(node, win);*/
    }
    //record move: Backup method from pseudocode
}

// this method corresponds to SimDefault in the pseudocode
__device__ int 
cuda_play_random_game(enum stone starting_color, curandState rState, int moves, int passes, int size)
{
	int gamelen = MAX_PLAYS - moves;
	enum stone color = starting_color;
	while (gamelen-- && passes < 2) {
        coord_t coord;
		cuboard_play_random(color, &coord, rState, size);
		if (IS_PASS(coord)) {
			passes++;
		} else {
			passes = 0;
		}
		color = custone_other(color);
	}
	return cuboard_fast_score(size);
}

// this going to correspond to the UctSearch method in the pseudocode from the literature
coord_t *cuda_genmove(struct board *b, struct time_info *ti, enum stone color){
	int passes = IS_PASS(b->last_move.coord) && b->moves > 0;
    void *data = NULL, *hData = NULL;
    int data_size = copy_essential_board_data(b, &hData);
    float offset = b->komi + b->handicap;
    assert(hData != NULL);

    //allocate and copy board data
    CUDA_CALL(cudaMalloc(&data, data_size));
    CUDA_CALL(cudaMemcpy(data, hData, data_size, cudaMemcpyHostToDevice));
    free(hData);

    reset_tree<<<1,(b->size-2)*(b->size-2)>>>(color,b->size, data);
    CUDA_CALL(cudaDeviceSynchronize());
    printf("Reset Tree\n");

    run_sims<<<M,N>>>(color, b->moves, passes, offset, b->size, data);
    CUDA_CALL(cudaPeekAtLastError());

    coord_t * best_move;
    coord_t *my_move = (coord_t *) malloc(sizeof(coord_t));

    CUDA_CALL(cudaMalloc(&best_move, sizeof(coord_t)));
    harvest_data<<<1,1>>>(best_move, b->size);

    CUDA_CALL(cudaDeviceSynchronize());
    printf("Collected Results\n");

    CUDA_CALL(cudaMemcpy(my_move, best_move, sizeof(coord_t), cudaMemcpyDeviceToHost));
    printf("Playing %d\n", *my_move);
    return my_move;
}

void init_kokrusher(struct board *b){
    size_t size = b->size * b->size;
    /*CUDA_CALL(cudaThreadSetLimit(cudaLimitMallocHeapSize, heap_size));*/
    cudaAllocDevArray(g_watermark, (M*N*(size/8) * sizeof(unsigned char)));
    cudaAllocDevArray(g_b, sizeof(enum stone) * M*N * size);
    cudaAllocDevArray(g_p, sizeof(coord_t) * M*N * size);
    cudaAllocDevArray(g_g, sizeof(group_t) * M*N * size);
    cudaAllocDevArray(g_libs, sizeof(int) * M*N * size);
    cudaAllocDevArray(g_gi, sizeof(coord_t) * GROUP_KEEP_LIBS * M*N * size);
    cudaAllocDevArray(g_ncol, sizeof(char) * S_MAX * M*N * size);
    cudaAllocDevArray(g_f, sizeof(coord_t) * M*N * size);
    cudaAllocDevArray(g_tree, sizeof(int) * 4 * MAX_ALLOC);

    //initialize random states
    cuda_rand_init<<<M,N>>>(time(NULL));    
    CUDA_CALL(cudaPeekAtLastError());

    printf("engine initialized\n");
    return;
}

size_t copy_essential_board_data(struct board * b, void **d){
    int size2 = board_size2(b);
    int bsize = size2 * sizeof(*b->b);
	int fsize = size2 * sizeof(*b->f);
	int psize = size2 * sizeof(*b->p);
	int gsize = size2 * sizeof(*b->g);
    int glibsize = sizeof(int) * size2;
    int gisize = sizeof(int) * size2 * GROUP_KEEP_LIBS;
    int ncolsize = sizeof(int) * size2 * S_MAX;
    int capsize = sizeof(int) * S_MAX;
    int flensize = sizeof(int);
    int total = bsize + fsize + psize + gsize + glibsize + gisize + ncolsize + capsize + flensize;
    int i,j;
    char *data = (char *) malloc(total);
    *d = (void *) data;
    memcpy(data, b->b, bsize);
    data += bsize;
    memcpy(data, b->f, fsize);
    data += fsize;
    memcpy(data, b->p, psize);
    data += psize;
    memcpy(data, b->g, gsize);
    data += gsize;
    for (i=0; i<size2; i++)
        ((int *) data)[i] = b->gi[i].libs;
    data += glibsize;
    for (i=0; i<size2; i++)
        for (j=0; j<GROUP_KEEP_LIBS; j++)
            ((int *) data)[(i*GROUP_KEEP_LIBS) + j] = b->gi[i].lib[j];
    data += gisize;
    for (i=0; i<size2; i++)
        for (j=0; j<S_MAX; j++)
            ((int *) data)[(i*S_MAX) + j] = b->n[i].colors[j];
    data += ncolsize;
    for (i=0; i<S_MAX; i++)
        ((int *) data)[i] = b->captures[i];
    data += capsize;
    *((int *) data) = b->flen;
    return total;
}

void __checkerr(cudaError_t e, char * file, int line){
    if (e != cudaSuccess){
      printf("Error %s at %s:%d\n",cudaGetErrorString(e),file,line); 
      exit(1);
    }
}

__global__ void cuda_rand_init(unsigned long seed){
    curand_init(seed, bid, 0, &randStates[bid]);
}

/*using namespace std;*/

#include <cuda.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "kokrusher/cuboard.h"
#include "kokrusher/uct.h"
#include "kokrusher/cudautil.h"
extern "C" {
#include "board.h"
}
#include "kokrusher/cudamst.h"

#define MAX_PLAYS 200
#define N_PLAYOUTS 60
/*#define BENCHMARK*/
#ifdef BENCHMARK
#include <sys/time.h>
#endif
//Global board state array variables.
//Yes this code is horrible. I'm sorry.
__device__ curandState randStates[M*N];
__device__ __constant__ int b_size;
__device__ __constant__ int g_data[board_data_size(BOARD_MAX_SIZE + 2)];
__device__ int g_flen[M*N];
__device__ stone (*g_b)[M*N];
__device__ coord_t (*g_f)[M*N];
__device__ coord_t (*g_p)[M*N];
__device__ group_t (*g_g)[M*N];
__device__ int (*g_libs)[M*N];
__device__ unsigned char (*g_watermark)[M*N];
__device__ coord_t (*g_gi)[GROUP_KEEP_LIBS][M*N];
__device__ int g_caps[S_MAX][M*N];
__device__ char (*g_ncol)[S_MAX][M*N];

// this method roughly corresponds to the Simulate method from the pseudocode
__global__ void 
run_sims(enum stone color, int moves, float komi)
{
    float score;
    int win;
    m_tree_t *node;
    //where uct happens: SimTree method from pseudocode
    for (int i=0; i<N_PLAYOUTS; i++) {
        cuboard_copy();
        node = walk_down(&color);
        cuda_playout(color, moves); //divergent section
        score = cuboard_fast_score() + komi;
	    win = (color == S_WHITE ^ score < 0);
        backup(node, win);
    }
    //record move: Backup method from pseudocode
}

// this method corresponds to SimDefault in the pseudocode
__device__ void 
cuda_playout(enum stone color, int moves)
{
    int passes = 0;
	int gamelen = MAX_PLAYS - moves;
	while (gamelen-- && passes < 2) {
        coord_t coord;
		cuboard_play_random(color, &coord);
		if (IS_PASS(coord)) {
			passes++;
		} else {
			passes = 0;
		}
		color = custone_other(color);
	}
}

// this going to correspond to the UctSearch method in the pseudocode from the literature
coord_t *
cuda_genmove(struct board *b, struct time_info *ti, enum stone color)
{
    copy_essential_board_data(b);
    float offset = b->komi + b->handicap;

#ifdef BENCHMARK
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
#endif

    reset_tree(color, b->flen);
    run_sims<<<M,N>>>(color, b->moves, offset);
    CUDA_CALL(cudaDeviceSynchronize());
    coord_t *my_move = (coord_t *) malloc(sizeof(coord_t));
    *my_move= get_best_move(b);

#ifdef BENCHMARK
    gettimeofday(&t2, 0);
    double duration = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    printf("SIMS %d, duration %lf\n", M*N*N_PLAYOUTS, duration);
#endif

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
    cudaMemcpyToSymbol(b_size, &b->size, sizeof(int));

    //initialize random states
    cuda_rand_init<<<M,N>>>(time(NULL));    
    CUDA_CALL(cudaPeekAtLastError());
    return;
}

void copy_essential_board_data(struct board * b){
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
    void *start = malloc(total);
    char *data = (char *) start;
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
    cudaMemcpyToSymbol(g_data, start, total);
}

__global__ void cuda_rand_init(unsigned long seed){
    curand_init(seed, bid, 0, &randStates[bid]);
}

/*using namespace std;*/

#include <cuda.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "kokrusher/cuboard.h"
extern "C" {
#include "board.h"
}
#include "kokrusher/cudamst.h"

#define MAX_PLAYS 400
#define FLEN(_b) (_b->my_flen > 81 ? 81 : b->my_flen)
#define N_PLAYOUTS 10

#define cudaAllocDevArray(dp_,size_) do { \
    void *lptr = NULL; \
    CUDA_CALL(cudaMalloc(&lptr, size_)); \
    CUDA_CALL(cudaMemcpyToSymbol(dp_, &lptr, sizeof(void *))); \
    } while(0)

#define cuda0AllocDevArray(dp_,size_) do { \
    void *lptr = NULL; \
    CUDA_CALL(cudaMalloc(&lptr, size_)); \
    CUDA_CALL(cudaMemset(lptr, 0, size_)); \
    CUDA_CALL(cudaMemcpyToSymbol(dp_, &lptr, sizeof(void *))); \
    } while(0)

#define symbolicMemclear(dp_,size_) do { \
    void *lptr = NULL; \
    CUDA_CALL(cudaGetSymbolAddress(&lptr, dp_)); \
    CUDA_CALL(cudaMemset(lptr, 0, size_)); \
} while(0)

#define CUDA_CALL(x) __checkerr((x), __FILE__, __LINE__) 

__device__ int g_flen[M*N];
__device__ stone (*g_b)[M*N];
__device__ coord_t (*g_f)[M*N];
__device__ coord_t (*g_p)[M*N];
__device__ group_t (*g_g)[M*N];
__device__ int (*g_libs)[M*N];
__device__ coord_t (*g_gi)[GROUP_KEEP_LIBS][M*N];
__device__ int g_caps[S_MAX][M*N];
__device__ char (*g_ncol)[S_MAX][M*N];



void __checkerr(cudaError_t e, char * file, int line){
    if (e != cudaSuccess){
      printf("Error %s at %s:%d\n",cudaGetErrorString(e),file,line); 
      exit(1);
    }
}

__device__ curandState *randStates;

__global__ void cuda_rand_init(unsigned long seed){
    int id = threadIdx.x;
    curand_init(seed, id, 0, randStates + id);
}

// this method corresponds to SimDefault in the pseudocode
__device__ int 
cuda_play_random_game(enum stone starting_color, curandState rState, int moves, int passes, int size)
{
	int gamelen = MAX_PLAYS - moves;
	enum stone color = starting_color;
	while (gamelen-- && passes < 2) {
		color = custone_other(color);
        coord_t coord;
		cuboard_play_random(color, &coord, rState, size);
		if (IS_PASS(coord)) {
			passes++;
		} else {
			passes = 0;
		}
	}

	floating_t score = cuboard_fast_score(size);
	int result = (starting_color == S_WHITE ? score * 2 : - (score * 2));
	return result;
}

// this method roughly corresponds to the Simulate method from the pseudocode
__global__ void run_sims(enum stone color, int *votes, int moves, int passes, int size){
    //make first move according to blckid and tid
    cuboard_init(size);
    curandState myState = randStates[bid];
    int win;
    int i = bid % my_flen;
    //where uct happens: SimTree method from pseudocode
    /*for (i=0; i<81; i++) {*/
        cuboard_play(color, i, size);
        win = cuda_play_random_game(color, myState, moves, passes, size);
        atomicAdd(&votes[i], win);
    /*}*/
    //record move: Backup method from pseudocode
}

/*__global__ void make_moves(struct move *moves, int nmoves){*/
    /*int i;*/
    /*for (i=0; i<nmoves; i++){*/
        /*cuboard_play(device_board, &moves[i]); */
    /*}*/
/*}*/

/*__host__ int reconcile(struct board *b, struct mst_dat * d) */
/*{*/
    /*printf("prereconciling");*/
    /*struct board *old = d->host_board;*/
    /*if (old->moves == b->moves)*/
        /*return 1;*/
    /*if (old->moves+4 < b->moves || old->moves > b->moves)*/
        /*//not enough information to recreate moves*/
        /*return 0;*/
    /*printf("reconciling");*/
    /*int i=0, nmoves = b->moves - old->moves;*/
    /*size_t msize = nmoves * sizeof(struct move);*/
    /*struct move *makeup_moves=NULL, *dev_moves=NULL;  */
    /*makeup_moves = (struct move *) malloc(msize);*/
    /*CUDA_CALL(cudaMalloc((void**)&dev_moves, msize));*/
    /*for (;nmoves > 1; nmoves--){*/
        /*if (nmoves == 4) { */
            /*makeup_moves[i].coord = b->last_move4.coord;*/
            /*makeup_moves[i].color = b->last_move4.color;*/
        /*}*/
        /*if (nmoves == 3) {*/
            /*makeup_moves[i].coord = b->last_move3.coord;*/
            /*makeup_moves[i].color = b->last_move3.color;*/
        /*}*/
        /*if (nmoves == 2) {*/
            /*makeup_moves[i].coord = b->last_move2.coord;*/
            /*makeup_moves[i].color = b->last_move2.color;*/
        /*}*/
        /*if (nmoves == 1) {*/
            /*makeup_moves[i].coord = b->last_move.coord;*/
            /*makeup_moves[i].color = b->last_move.color;*/
        /*}*/
        /*board_play(old, &makeup_moves[i]);*/
        /*i++;*/
    /*}*/
    /*CUDA_CALL(cudaMemcpy(dev_moves,makeup_moves, msize, cudaMemcpyHostToDevice));*/
    /*make_moves<<<1,1>>>(dev_moves, nmoves);*/
    /*free(makeup_moves);*/
    /*free(dev_moves);*/
    /*return 1;*/
/*}*/

// this going to correspond to the UctSearch method in the pseudocode from the literature
coord_t *cuda_genmove(void *d, struct board *b, struct time_info *ti, enum stone color){
    struct mst_dat * data = (struct mst_dat*) d;
    int *votes = NULL, *hVotes=NULL;
    int vote_size = 81 * sizeof(int);
    /*if (!reconcile(b, data)) {*/
        /*fprintf(stderr, "failed to reconcile device board with given board");*/
    /*}*/
	int passes = IS_PASS(b->last_move.coord) && b->moves > 0;
    
    //allocate vote array
    hVotes = (int *) malloc(vote_size);
    CUDA_CALL(cudaMalloc(&votes, vote_size));
    CUDA_CALL(cudaMemset(votes, 0, vote_size));

    run_sims<<<M,N>>>(color, votes, b->moves, passes, b->size);
    CUDA_CALL(cudaPeekAtLastError());

    CUDA_CALL(cudaMemcpy(hVotes, votes, vote_size, cudaMemcpyDeviceToHost));

    int i,maxi=0, max=0;
    for (i=0; i<81; i++){
        printf("vote  %d=%d\n", i, hVotes[i]);
        if (hVotes[i] > max) {
            max = hVotes[i];
            maxi = i;
        }
    }
    coord_t *my_move = (coord_t*) malloc(sizeof(coord_t));
    *my_move = b->f[maxi];
    return  my_move;
}

void * init_kokrusher(struct board *b){
    //if board is not fresh, fail
    //copy the board to a new location so we can use the miror
    if (b->moves != 0) {
        fprintf(stderr, "Cannot initialize kokrusher engine with a board that has already been played on\n");
        exit(1);
    }
    size_t size = b->size * b->size;

    /*CUDA_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,128*1024*1024 ));*/
    /*cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);*/

    symbolicMemclear(g_flen, sizeof(int) * M*N);
    symbolicMemclear(g_caps, sizeof(coord_t) * S_MAX * M*N);

    cuda0AllocDevArray(g_b, sizeof(enum stone) * M*N * size);
    cuda0AllocDevArray(g_p, sizeof(coord_t) * M*N * size);
    cuda0AllocDevArray(g_g, sizeof(group_t) * M*N * size);
    cuda0AllocDevArray(g_libs, sizeof(int) * M*N * size);
    cuda0AllocDevArray(g_gi, sizeof(coord_t) * GROUP_KEEP_LIBS * M*N * size);
    cuda0AllocDevArray(g_ncol, sizeof(char) * S_MAX * M*N * size);
    cudaAllocDevArray(g_f, sizeof(coord_t) * M*N * size);
    cudaAllocDevArray(randStates, sizeof(curandState) * M*N);

    //initialize random states
    cuda_rand_init<<<M,N>>>(time(NULL));    
    CUDA_CALL(cudaPeekAtLastError());

    printf("engine initialized, size=%d \n", size);
    return (void*) b;
}

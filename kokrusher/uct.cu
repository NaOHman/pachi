#include "kokrusher/cuboard.h"
#include "kokrusher/uct.h"
#include "kokrusher/cudautil.h"

extern "C" {
#include "board.h"
}

#define UCT_CONST 1.0
#define DIV_ZERO_CONST 2.0
#define MAX_FREE  (BOARD_MAX_SIZE * BOARD_MAX_SIZE)
#define MAX_ALLOC 25000000

__device__ float g_wheel[MAX_FREE][M*N];
__device__ int next_tree;
__device__ m_tree_t root[MAX_ALLOC];
extern __device__ int g_flen[M*N];
extern __device__ int (*g_f)[M*N];

extern __device__ curandState randStates[M*N];

__device__ float 
uct(int p_visits, int visits, int wins) 
{
    float logv = logf((float) (p_visits == 0 ? 1.0 : p_visits));
    float radicand = (visits <= 0 ? DIV_ZERO_CONST : logv /(float) visits);
    float my_q = (visits <= 0 ? 0 : (float) wins / (float) visits);
    return my_q + (UCT_CONST * sqrt(radicand));		
}

__device__ float 
roulette_scaling_function(float uct) 
{
    // 0 should probably output as 0, so maybe not an exponential function?
    return uct*uct;
}

__device__ int 
roulette_uct(m_tree_t *node) 
{
    float total = 0.0;
    float score;
    for (int i=0; i<node->c_len; i++) { // iterate over length of array: size^2
        score = uct(node->visits, node->children[i].visits, node->children[i].wins);
        score = roulette_scaling_function(score);
        total = total + score;
        g_wheel[i][bid] = score;
    }
    float pos = curand_uniform(&(randStates[bid])) * total;
    int result = node->c_len - 1;
    for (int i=0; i<node->c_len-1; i++) {
        pos = pos - g_wheel[i][bid];
        if (i < result && pos <= 0.0) {//ensures we always return something
            result = i; 
        } 
    }
    return result; 
}

__device__ void
init_node(m_tree_t *self, m_tree_t *parent, coord_t move)
{
    self->visits = 0;
    self->wins = 0;
    self->parent = parent;
    self->children = NULL;
    self->c_len = -1;
    self->move = move;
}

/* must be called when the thread's board is 
   in the same state as the node */
__device__ void
alloc_children(m_tree_t *node)
{
    assert(next_tree < MAX_ALLOC);
    int child_offset = atomicAdd(&next_tree, my_flen);
    node->children = &root[child_offset];
    node->c_len = my_flen;
    for (int i=0; i<my_flen; i++){
        init_node(&(node->children[i]), node, nth_free(i));
    }
}

__device__  m_tree_t *
walk_down(enum stone *play_color) 
{
    /* 
    This method traverses our tree data structure and finds a leaf node.
    It uses the roulette_select method above to decide which node to traverse to
    at each layer, and that method only returns legal moves.
    When the node selected at the next layer already exists, it recursively
    traverses to that node.  Otherwise, it creates a new node and returns it.
    The board is a thread-local object, passed into this method so that it can
    be updated with moves corresponding to the tree traversal. 
    */
    int next;
    m_tree_t* current = &(root[0]);
    // check if next node exists, if not, create it.  first time always true
    do { 
        if (current->c_len == -1)
            alloc_children(current);
        if (current->c_len == -1) //game is in end state, just return it the playout won't happen
            return current;
        next = roulette_uct(current);
        current = current->children + next;
        cuboard_play(*play_color, current->move);
        *play_color = custone_other(*play_color);
    } while (current->visits > 0);
    return current;
}

__device__ void 
backup(m_tree_t *node, bool is_win) 
{
    while (node != NULL) {
        atomicAdd(&(node->visits), 1); //actually fairly high chance of collision now that I think about it...
        atomicAdd(&(node->wins), is_win);
        node = node->parent;
    } 
}

__global__ void
harvest_best(int *best)
{
    float highest_uct = 0.25; //if odds are we loose 75% for each move, just pass.
    float score;
    *best = 1;
    m_tree_t node  = root[0];
    for (int i=0; i<node.c_len; i++) { // iterate over length of array: size^2
        score = uct(node.visits, node.children[i].visits, node.children[i].wins);
        if (score > highest_uct){
            *best = node.children[i].move;
            highest_uct = score;
        }
    }
}

__host__ int
get_best_move(struct board *b)
{
    int *dBest, *hBest = (int*) malloc(sizeof(int));
    CUDA_CALL(cudaMalloc(&dBest, sizeof(int)));
    harvest_best<<<1,1>>>(dBest);
    CUDA_CALL(cudaMemcpy(hBest,dBest,sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(dBest));
    for (int i=0; i<b->flen; i++){
        //sanity check
        if (*hBest == b->f[i])
            return *hBest;
    }
    return PASS;
}

__global__ void
alloc_root(enum stone color)
{
    cuboard_copy();
    m_tree_t *self = &root[threadIdx.x + 1];
    init_node(self, root, nth_free(threadIdx.x));
    cuboard_play(color, nth_free(threadIdx.x));
    alloc_children(self);
    if (threadIdx.x == 0) {
        init_node(root, NULL, PASS);
        root->children = root+1;
        root->c_len = blockDim.x;
    }
}

__host__ void
reset_tree(enum stone color, int flen)
{
    int child_len = 1 + flen;
    CUDA_CALL(cudaMemcpyToSymbol(next_tree, &child_len, sizeof(int))); //must happen before tree is reset
    alloc_root<<<1,flen>>>(color);
}

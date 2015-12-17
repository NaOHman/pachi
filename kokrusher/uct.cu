/**
 * This file contains all of the UCT related functions needed
 * to make MCTS work
 */
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

/**
 * Calculate the UCT score for the given values of parent visits, visits, and wins.
 * Since all number may be zero we have to accomodate by replacing with constants
 */
__device__ float 
uct(int p_visits, int visits, int wins) 
{
    float logv = logf((float) (p_visits == 0 ? 1.0 : p_visits));
    float radicand = (visits <= 0 ? DIV_ZERO_CONST : logv /(float) visits);
    float my_q = (visits <= 0 ? 0 : (float) wins / (float) visits);
    return my_q + (UCT_CONST * sqrt(radicand));		
}

/**
 * a function that scales the UCT scores. This is necessary because the difference
 * between good and bad scores is often quite small
 */
__device__ float 
roulette_scaling_function(float uct) 
{
    return uct*uct*uct*uct*uct*uct;
}

/**
 * Uses Roulette Selection to pick a child node of the given node. Children are
 * Weighted by their scaled UCT score. Returns the chosen child
 */
__device__ m_tree_t * 
roulette_uct(m_tree_t *node) 
{
    float total = 0.0;
    float score;
    for (int i=0; i<node->c_len; i++) { 
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
    return node->children + result; 
}

/**
 * initialize a tree node given it's parent and the move it represents
 */
__device__ void
init_node(m_tree_t *self, m_tree_t *parent, coord_t move)
{
    self->visits = 0;
    self->wins = 0;
    self->parent = parent;
    self->children = NULL;
    self->c_len = -1; //signifies this node has not been expanded
    self->move = move;
}

/** 
 * expand a node of the tree, it must be called when the thread's board is 
 * in the same state as the node. There is a very high chance of a collision
 * and therefore wasted memory but since there's no critical section support
 * in CUDA we just have to live with it.
 */
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

/**
 * Selects a leaf node for expansion and returns that node.
 * Modifies the play_color and the thread-local board to reflect the path
 */
__device__  m_tree_t *
walk_down(enum stone *play_color) 
{
    m_tree_t* current = &(root[0]);
    // check if next node exists, if not, create it.  first time always true
    do { 
        if (current->c_len == -1)
            alloc_children(current);
        if (current->c_len == 0) //game is in end state, just return it the playout won't do anything harmful
            return current;
        current = roulette_uct(current);
        cuboard_play(*play_color, current->move);
        *play_color = custone_other(*play_color);
    } while (current->visits > 0);
    return current;
}

/**
 * record the results of a simulation, is_win = whether the color corresponding to the 
 * node won NOT whether the player being simulated won
 */
__device__ void 
backup(m_tree_t *node, bool is_win) 
{
    while (node != NULL) {
        atomicAdd(&(node->visits), 1); //very high chance of collision
        atomicAdd(&(node->wins), is_win);
        is_win = !is_win;
        node = node->parent;
    } 
}

/**
 * a kernel that fetches the node with the highest UCT score
 */
__global__ void
harvest_best(int *best)
{
    float highest_rate = 0.25; //if odds are we loose 75% for each move, just pass.
    float win_rate;
    *best = 1;
    m_tree_t node  = root[0];
    for (int i=0; i<node.c_len; i++) { // iterate over length of array: size^2
        //just use the win percentage, no bonus for unexplored nodes
        win_rate =(float) node.children[i].wins/ (float) node.children[i].visits;
        if (win_rate > highest_rate){
            *best = node.children[i].move;
            highest_rate = win_rate;
        }
    }
}

/**
 * the host side wrapper for harvest_best
 */
__host__ int
get_best_move(struct board *b)
{
    int *dBest, *hBest = (int*) malloc(sizeof(int));
    CUDA_CALL(cudaMalloc(&dBest, sizeof(int)));
    harvest_best<<<1,1>>>(dBest);
    CUDA_CALL(cudaMemcpy(hBest,dBest,sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(dBest));
    for (int i=0; i<b->flen; i++){
        //sanity check TODO check for ko violations
        if (*hBest == b->f[i])
            return *hBest;
    }
    return PASS;
}

/**
 * pre allocate the first two layers to avoid collisions
 * you must call copy_essential_board_data before this
 */
__global__ void
alloc_root(enum stone color)
{
    cuboard_reset();
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

/**
 * Host side wrapper for alloc_root,
 * you must call copy_essential_board_data before this
 */
__host__ void
reset_tree(enum stone color, int flen)
{
    int child_len = 1 + flen;
    CUDA_CALL(cudaMemcpyToSymbol(next_tree, &child_len, sizeof(int))); //must happen before tree is reset
    alloc_root<<<1,flen>>>(color);
}

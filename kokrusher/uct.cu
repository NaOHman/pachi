#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kokrusher/cuboard.h"
#include "kokrusher/uct.h"

/*extern __device__ __constant__ int b_size;*/
extern __device__ curandState randStates[M*N];
extern __device__ int next_alloc;
extern __device__ int (*g_tree)[4];
#define UCT_CONST 1.0
#define DIV_ZERO_CONST 2.0


__device__ void init_tree(tree_t id, tree_t parent) {
    tree_parent(id) = parent;
    tree_wins(id) =  0;
    tree_visits(id) = -1; //negative when invalid, it's easier to assume they're invalid
    tree_child(id) = -1;
}

__device__ static void alloc_children(tree_t id, enum stone color, int size) {
    //relies on the thread's board being in the state corresponding to the given id.
    if (tree_child(id) < 1) {
        assert(next_alloc < MAX_ALLOC); 
        __threadfence();
        int first_child = atomicAdd(&next_alloc, child_size);
        for (int i=0; i<child_size; i++){
            init_tree(first_child + i, id);
            if (cuboard_is_valid_play(color, (coord_t) index_2_coord(i, size), size)) {
                tree_visits(first_child + i) = 0;
            }
        }
        tree_child(id) = first_child;
    }
    __threadfence();
}

__device__ float uct(int my_wins, int parent_visits, int my_visits) {
    float logv = logf((float) parent_visits + 1.0);
    float radicand = (my_visits <= 0 ? DIV_ZERO_CONST : logv /(float) my_visits);
    float my_q = (my_visits <= 0 ? 0 : (float) my_wins / (float) my_visits);
    return my_q + (UCT_CONST * sqrt(radicand));		
}

__device__ float roulette_scaling_function(float uct) {
    // this function needs to convert the UCT value into
    // the value we want for the roulette selection
    // possibly an exponential function or a basic polynomial function?
    // 0 should probably output as 0, so maybe not an exponential function?
    return uct*uct;
}

__device__ int roulette_select(tree_t node, int bsize) {
    int size = bsize - 2;
    float total = 0.0;
    bool children = false;
    float *uctArray = (float*) malloc(sizeof(float)*size*size);
    for (int i=0; i<size*size; i++) { // iterate over length of array: size^2
        tree_t child = tree_nth_child(node, i);
        if (tree_visits(child) > -1) {
            // use visits == -1 to indicate illegal moves
            //uct on child
            children = true;
            float temp = uct(tree_wins(child), tree_visits(node), tree_visits(child));
            temp = roulette_scaling_function(temp);
            total = total + temp;
            uctArray[i] = temp;
        } else {
            uctArray[i] = 0;
        }
    }
    assert(children);
    assert(total > 0.0);
    float pos = curand_uniform(&(randStates[bid])) * total;
    assert(pos < total);
    assert(pos > 0.0);
    for (int i=0; i<size*size; i++) {
        pos = pos - uctArray[i];
        if (pos <= 0.0) {
            free(uctArray);
            return i; // this should always return, but just in case...
        } 
    }
    free(uctArray);
    return size*size-1; // in case of emergencies, return final entry
}

__device__ tree_t walk_down(enum stone *play_color, int size) {
    /* 
    This method traverses our tree data structure and finds a leaf node.
    It uses the roulette_select method above to decide which node to traverse to
    at each layer, and that method only returns legal moves.
    When the node selected at the next layer already exists, it recursively
    traverses to that node.  Otherwise, it creates a new node and returns it.
    The board is a thread-local object, passed into this method so that it can
    be updated with moves corresponding to the tree traversal. 
    */

    // TODO: check if game is over
    int next;
    tree_t current = 0;
    // check if next node exists, if not, create it.  first time always true
    do { 
        if (tree_child(current) <= 0)
            alloc_children(current, *play_color, size);
        next = roulette_select(current, size);
        //TODO make sure there was a valid move
        cuboard_play(*play_color, index_2_coord(next, size), size);
        *play_color = custone_other(*play_color);
        current = tree_nth_child(current, next);
    } while (tree_visits(current) > 0);
    return current; //TODO double check that this is the right color
}

__device__ void backup(tree_t n, bool is_win) {
    atomicAdd(&tree_visits(n), 1); //actually fairly high chance of collision now that I think about it...
    atomicAdd(&tree_wins(n), is_win);
    while (tree_parent(n) != -1){ // we don't need q values for the parent node
        n = tree_parent(n);
        atomicAdd(&tree_visits(n), 1); //actually fairly high chance of collision now that I think about it...
        atomicAdd(&tree_wins(n), is_win);
    }
}

__device__ coord_t best_move(int size) {
    coord_t move = -1;
    float best = -0.5;
    for(int i=1; i<=child_size; i++){
        if (tree_visits(i) > 0){
            float next = (float) tree_wins(i) / (float) tree_visits(i);
            if (next > best) {
                move = index_2_coord(i-1, size);
                best = next;
            }
        }
    }
    return move;
}

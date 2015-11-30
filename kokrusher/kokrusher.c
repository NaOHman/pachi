#include <stdio.h>
#include <stdlib.h>

#include "board.h"
#include "engine.h"
#include "debug.h"
#include "move.h"
#include "kokrusher/kokrusher.h"

#ifdef CUDA
#include "kokrusher/cudamst.h"
#endif

static coord_t *
kokrusher_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive)
{ // this method is equivalent to the UctSearch pseudocode method
#ifdef CUDA
    //If cuda support is enabled generate the move with cuda
    fprintf(stderr, "CUDA KoKrusher\n");
    return cuda_genmove(e, b, ti, color, pass_all_alive);
#else
    //No cuda support just do your own thing
    fprintf(stderr, "Sequential KoKrusher\n");
    coord_t *coord;
    struct board b2;
    board_copy(&b2, b);
    for (int i=0; i<=19; i++){
        for (int j=0; j<=19; j++){
            coord = coord_init(i,j,board_size(&b2));
            if (board_is_valid_play(&b2, color, *coord))
                return coord;
        }
    }
    *coord = -1;
    return coord;
#endif
}

struct engine *
engine_kokrusher_init(char *arg, struct board *b)
{
    struct engine *e = calloc2(1, sizeof(struct engine));
    e->name = "kokrusher";
    e->comment = "Macalester's own KoKrusher";
    e->genmove = kokrusher_genmove;
    return e;
}

struct tree_node {
		struct tree_node* parent;
		struct tree_node** children;
		int* child_visits;
		float* child_q;
}

float uct(float child_q, int parent_visits, int child_visits) {
		// child_q + c * sqrt(log(parent_visits)/child_visits)
		// c is an arbitrary predefined constant
		// need to handle child_visits == 0 also
		return child_q;
}

float rouletteScalingFunction(float uct) {
		// this function needs to convert the UCT value into 
		// the value we want for the roulette selection
		// possibly an exponential function or a basic polynomial function?
		// 0 should probably output as 0, so maybe not an exponential function?
		return uct;
}

int roulette_select(int parent_visits, float* q_array, int* child_visits_array) {
		float total = 0.0;
		float* uctArray = malloc(sizeof(float)*n*n); // is that how to init that?
		for (int i=0; i<n*n; i++) { // iterate over length of array: n^2
				// need to ignore illegal moves here
				if (child_visits_array[i] > -1) { // use visits == -1 to indicate illegal moves
						float temp = uct(q_array[i], parent_visits, child_visits_array[i]);
						temp = rouletteScalingFunction(temp);
						total += temp;
						uctArray[i] = temp;
				} else {
						uctArray[i] = 0;
				}
		}
		float random = 0.0; // set up random number generation here, assuming from [0,1)
		float pos = random * total;
		for (int i=0; i<n*n; i++) {
				pos -= uctArray[i];
				if (pos < 0) return i; // this should always return, but just in case...
		}
		return n*n-1; // in case of emergencies, return final entry
}

struct move get_move(int n) {
		// this method needs to covert an integer between 0 and n^2-1 into a pachi board move
		return;
}

struct tree_node* sim_tree(int visits, struct tree_node* current, struct board* b) {
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
		// lets assume that roulette_select only returns legal moves
		int next = roulette_select(visits, current->child_q, current->child_visits);
		struct move m = get_move(next);
		board_play(b, &m);
		if (current->child_visits[next] > 0) { // check if next node exists, if not, create it
				return sim_tree(current->child_visits[next], current->children[next], b);
				// is it bad that this is recursive?
		} else {
				return new_node(current, next, b);
		}
}

struct tree_node* new_node(struct tree_node* parent, int index, struct board* b) {
		struct tree_node* n = malloc(/* ? */);
		n->parent = parent;
		parent->children[index] = n;
		// mark illegal children of n here
		return n;
}

void backup(struct tree_node* n, bool is_win) {
		while ( /* n has parent */ ) {
				struct tree_node* parent = n->parent;
				int i;
				for (i=0; i<n*n; i++) {
						if (parent->children[i] == n) {
								break;
						}
				}
				parent->child_visits[i]++;
				parent->child_q = (is_win - child_q) / parent->child_visits[i];
				n = parent;
		}
}


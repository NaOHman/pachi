#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "board.h"
#include "engine.h"
#include "debug.h"
#include "move.h"

#define UCT_CONST 1
#define DIV_ZERO_CONST 2.0

struct tree_node* init_tree(struct board* b) {
		int size = board_size(b);
		struct tree_node *n = (struct tree_node*) malloc(sizeof(struct tree_node));
		n->parent = NULL;
		n->child_visits = (int*) memset(malloc(sizeof(int)*size*size), -1, size*size);
		n->child_q = (float*) memset(malloc(sizeof(float)*size*size), 0, size*size);
		for (int i=0; i < b->flen; i++) {
				int index = get_index(b->f[i], size);
				n->child_visits[index] = 0;
		}
		return n;
}

struct move get_move(int index, int size, enum stone color) {
		coord_t *c = coord_init(index / size + 1, index % size + 1, size);
		struct move m = { *c, color };
		return m;
}

int get_index(coord_t c, int size) {
		int x = c / (size + 2) - 1;
		int y = c % (size + 2) - 1;
		index = x * size + y; // might need to swap x and y here
		return index;
}

float uct(float child_q, int parent_visits, int child_visits, bool my_turn) {
		float radicand = (child_visits <= 0 ? DIV_ZERO_CONST : log(parent_visits)/child_visits);
		return child_q + (my_turn ? 1.0 : -1.0) * UCT_CONST * sqrt(radicand);		
}

float roulette_scaling_function(float uct, bool my_turn) {
		// this function needs to convert the UCT value into
		// the value we want for the roulette selection
		// possibly an exponential function or a basic polynomial function?
		// 0 should probably output as 0, so maybe not an exponential function?
		uct = (my_turn ? 1.0 : -1.0) * uct;
		return uct*uct;
}

int roulette_select(int parent_visits, float* q_array, int* child_visits_array, bool my_turn, int size) {
		float total = 0.0;
		float *uctArray = (float*) malloc(sizeof(float)*size*size);
		for (int i=0; i<size*size; i++) { // iterate over length of array: size^2
				if (child_visits_array[i] > -1) { // use visits == -1 to indicate illegal moves
						float temp = uct(q_array[i], parent_visits, child_visits_array[i], my_turn);
						temp = roulette_scaling_sunction(temp, my_turn);
						total += temp;
						uctArray[i] = temp;
				} else {
						uctArray[i] = 0;
				}
		}
		float random = 0.0; // TODO: JEFFREY set up random number generation here, assuming from [0,1)
		float pos = random * total;
		for (int i=0; i<size*size; i++) {
				pos -= uctArray[i];
				if (pos < 0) return i; // this should always return, but just in case...
		}
		return size*size-1; // in case of emergencies, return final entry
}

struct tree_node* new_node(struct tree_node* parent, int index, struct board* b) {
		struct tree_node *n = init_tree(b);
		n->parent = parent;
		parent->children[index] = n;
		return n;
}

struct tree_node* sim_tree(int visits, struct tree_node* current, struct board* b, enum stone play_color, enum stone my_color) {
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
		int next = -1;
		while (visits > 0) { // check if next node exists, if not, create it.  first time always true
				if (next != -1) { // don't do this the first time, do it every other time
						current = current->children[next];
				}
				next = roulette_select(visits, current->child_q, current->child_visits, play_color == my_color, board_size(b));
				struct move m = get_move(next);
				board_play(b, &m);
				play_color = custone_other(play_color);
				visits = current->child_visits[next];
		}
		return new_node(current, next, b);
}

void backup(struct tree_node* n, bool is_win) {
		while (n->parent != NULL) {
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


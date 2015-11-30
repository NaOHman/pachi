#include "board.h"

struct tree_node {
		struct tree_node* parent;
		struct tree_node** children;
		int* child_visits;
		float* child_q;
};

struct tree_node* init_tree(struct board* b);

float uct(float child_q, int parent_visits, int child_visits, bool my_turn);

int roulette_select(int parent_visits, float* q_array, int* child_visits_array, bool my_turn);

struct tree_node* new_node(struct tree_node* parent, int index, struct board* b);

struct tree_node* sim_tree(int visits, struct tree_node* current, struct board* b, enum stone play_color, enum stone my_color);

void backup(struct tree_node* n, bool is_win);
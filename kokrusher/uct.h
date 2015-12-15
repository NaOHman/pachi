#ifndef PACHI_CUUCT_H
#define PACHI_CUUCT_H

#include "kokrusher/cuboard.h"

#define MAX_ALLOC 32400001
//#define child_size ((b_size - 2) * (b_size -2))
#define child_size 81

#define tree_parent(t_) (g_tree[t_][0])
#define tree_wins(t_) (g_tree[t_][1])
#define tree_visits(t_) (g_tree[t_][2])
#define tree_child(t_) (g_tree[t_][3])
#define tree_nth_child(t_,n_) (g_tree[t_][3] + n_)
#define coord_2_index(c_, sz_) ((((c_)/sz_) - 1) * (sz_ - 2) + (((c_) % sz_) -1))
#define index_2_coord(i_, sz_) ((((i_)/(sz_-2)) * (sz_)) + ((i_) % (sz_-2)) + sz_ + 1)

typedef int tree_t;

__device__ void init_tree(tree_t id, tree_t parent);

__device__ float uct(int my_wins, int parent_visits, int my_visits);

__device__ int roulette_select(tree_t node, int size);

__device__ tree_t walk_down(enum stone *play_color, int size);

__device__ void backup(tree_t n, bool is_win);

__device__ coord_t best_move(int size);

#endif

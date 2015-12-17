#ifndef PACHI_KKUCT
#define PACHI_KKUCT

typedef struct m_tree {
    struct m_tree *parent, *children;
    int visits, wins, c_len;
    coord_t move;
} m_tree_t;

__host__ void
reset_tree(enum stone color, int flen);

__host__ int
get_best_move(struct board *b);

__device__ void 
backup(m_tree_t *n, bool is_win);

__device__  m_tree_t *
walk_down(enum stone *play_color);

#endif

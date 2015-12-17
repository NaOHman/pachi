#ifndef PACHI_KKUCT
#define PACHI_KKUCT

typedef struct m_tree {
    struct m_tree *parent, *child;
    int visits, wins;
} m_tree_t;

__device__ int
roulette_uct(int p_visits, int *visits, int* wins, int len);

__host__ __device__  float
uct(int p_visits, int my_visits, int my_wins);
#endif

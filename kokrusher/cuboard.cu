/**
 * this file is a rewrite of board.c for use on CUDA
 * all functions do what their counterparts in board.c do
 * but with some functionality stripped away
 */

#include "kokrusher/cuboard.h"

extern __device__ curandState randStates[M*N];
extern __device__ int g_flen[M*N];
extern __device__ __constant__ int b_size;
extern __device__ __constant__ int g_data[board_data_size(BOARD_MAX_SIZE + 2)];
extern __device__ stone (*g_b)[M*N];
extern __device__ coord_t (*g_f)[M*N];
extern __device__ coord_t (*g_p)[M*N];
extern __device__ group_t (*g_g)[M*N];
extern __device__ int (*g_libs)[M*N];
extern __device__ unsigned char (*g_watermark)[M*N];
extern __device__ coord_t (*g_gi)[GROUP_KEEP_LIBS][M*N];
extern __device__ int g_caps[S_MAX][M*N];
extern __device__ char (*g_ncol)[S_MAX][M*N];

/**
 * reset the thread local board so that it's a copy of the global
 * board stored in g_data
 */
__device__ void 
cuboard_reset()
{
    int size2 = b_size * b_size;
    int i,j;
    void* data = (void *) g_data;
    //yes this could be one for loop but locality is a thing
    for (i=0; i<size2; i++){
        nth_stone(i) = *((enum stone *) data);
        data = (void*) (((enum stone *) data) + 1);
    }
    for (i=0; i<size2; i++){
        g_f[i][bid] = *((coord_t *) data);
        data = (void*) (((coord_t *) data) + 1);
    }
    for (i=0; i<size2; i++){
        g_p[i][bid]  = *((coord_t *) data);
        data = (void*) (((coord_t *) data) + 1);
    }
    for (i=0; i<size2; i++){
        g_g[i][bid] = *((group_t *) data);
        data = (void*) (((group_t *) data) + 1);
    }
    for (i=0; i<size2; i++){
        g_libs[i][bid] = *((int *) data);
        data = (void*) (((int *) data) + 1);
    }
    for (i=0; i<size2; i++){
        for(j=0; j<GROUP_KEEP_LIBS; j++){
            g_gi[i][j][bid] = *((int *) data);
            data = (void*) (((int *) data) + 1);
        }
    }
    for (i=0; i<size2; i++){
        for(j=0; j<S_MAX; j++){
            g_ncol[i][j][bid] = *((int *) data);
            data = (void*) (((int *) data) + 1);
        }
    }
     for (i=0; i<S_MAX; i++){
        g_caps[i][bid] = *((int *) data);
        data = (void*) (((int *) data) + 1);
    }
    my_flen = *((int *) data);
}

/**
 * add a liberty at coord to the given group
 */
__device__ static void
cuboard_group_addlib(group_t group, coord_t coord)
{
    int lc = lib_count(group);
    if (lc < GROUP_KEEP_LIBS) {
        for (int i = 0; i < GROUP_KEEP_LIBS; i++) {
            if (nth_lib(group,i) == coord)
                return;
        }
        nth_lib(group, lc) = coord;
        lib_count(group) = lc + 1;
    }
}

/**
 * check for extra liberties not already counted by the group
 * moderately expensive so don't call too often
 */
__device__ static void
cuboard_group_find_extra_libs(group_t g, coord_t avoid)
{
    /* Add extra liberty from the board to our liberty list. */
    int len = (b_size*b_size) / 8;
    for (int i=0; i<len; i++)
        g_watermark[i][bid]=0;
#define watermark_get(cx)	(g_watermark[cx >> 3][bid] & (1 << (cx & 7)))
#define watermark_set(cx)	(g_watermark[cx >> 3][bid] |= (1 << (cx & 7)))

    for (int i = 0; i < GROUP_KEEP_LIBS - 1; i++)
        watermark_set(nth_lib(g,i));
    watermark_set(avoid);

    for_each_in_group(g) {
        coord_t coord2 = c;
        for_each_neighbor(coord2, {
            if (nth_stone(c) + watermark_get(c) != S_NONE)
                continue;
            watermark_set(c);
            nth_lib(g, lib_count(g)) = c;
            lib_count(g) = lib_count(g) + 1;
            if (lib_count(g) >= GROUP_KEEP_LIBS)
                return;
        });
    } for_each_in_group_end;
#undef watermark_get
#undef watermark_set
}

/**
 * Somebody played at coord, remove it as a liberty from the group
 */
__device__ static void
cuboard_group_rmlib(group_t g, coord_t coord)
{
    for (int i = 0; i < lib_count(g) && i < GROUP_KEEP_LIBS; i++) {
        if (nth_lib(g,i) == coord){
            lib_count(g)--;
            nth_lib(g,i) = nth_lib(g,lib_count(g));
            nth_lib(g,lib_count(g)) = 0;
            if (lib_count(g) == GROUP_REFILL_LIBS)
                cuboard_group_find_extra_libs(g, coord);
            return;
        }
    }
}


/**
 * Remove a stone from the board
 * This is a low-level routine that doesn't maintain consistency
 * of all the board data structures. 
 */
__device__ static void
cuboard_remove_stone(group_t group, coord_t c)
{
    enum stone color = nth_stone(c);
    nth_stone(c) = S_NONE;
    nth_group(c) = 0;
    /* Increase liberties of surrounding groups */
    coord_t coord = c;
    for_each_neighbor(coord, {
        dec_neighbor_count(c, color);
        group_t g = nth_group(c);
        if (g && g != group){
            cuboard_group_addlib(g, coord);
        }
    });

    nth_free(my_flen) = c;
    my_flen++;
}

/**
 * capture a group, remove the stones and update global variables accordingly
 */
__device__ static int 
cuboard_group_capture(group_t group)
{
    int stones = 0;

    for_each_in_group(group) {
        captures(custone_other(nth_stone(c)))++;
        cuboard_remove_stone(group, c);
        stones++;
    } for_each_in_group_end;

    assert(lib_count(group) == 0);
    for (int i=0; i<GROUP_KEEP_LIBS; i++)
        nth_lib(group,i) = 0;
    return stones;
}

/**
 * add a stone to the group after prevstone
 */
__device__ static void 
add_to_group(group_t group, coord_t prevstone, coord_t coord)
{
    nth_group(coord) = group;
    next_group(coord) = next_group(prevstone);
    next_group(prevstone) = coord;
    for_each_neighbor(coord, {
        if (nth_stone(c) == S_NONE)
            cuboard_group_addlib(group, c);
    });
}

/**
 * combine two groups into one
 * fairly expensive operation, but unavoidable
 * The goto is not mine and I'm afraid of touching it
 * because I have no idea what it's doing
 */
__device__ static void
merge_groups(group_t to, group_t from)
{
    if (lib_count(to) < GROUP_KEEP_LIBS) {
        for (int i = 0; i < lib_count(from); i++) {
            for (int j = 0; j < lib_count(to); j++){
                if (nth_lib(to,j) == nth_lib(from,i)){
                    goto next_from_lib;
                }
            }
            nth_lib(to, lib_count(to)) = nth_lib(from,i);
            lib_count(to)++;
            if (lib_count(to) >= GROUP_KEEP_LIBS)
                break;
next_from_lib:;
        }
    }
    coord_t last_in_group;
    for_each_in_group(from) {
        last_in_group = c;
        nth_group(c) = to;
    } for_each_in_group_end;
    next_group(last_in_group) = next_group(to);
    next_group(to) = from;
    lib_count(from) = 0;
    for (int i=0;i<GROUP_KEEP_LIBS;i++)
        nth_lib(from,i) = 0;
}

/**
 * create a new singleton group containing the coord
 */
__device__ static group_t
new_group(coord_t coord)
{
    group_t g = coord;
    for_each_neighbor(coord, {
        if (nth_stone(c) == S_NONE)
            nth_lib(g,lib_count(g)) = c;
            lib_count(g)++;
    });
    nth_group(coord) = g;
    next_group(coord) = 0;
    return g;
}

/**
 * a stone was played at coord, belonging to group. Update c to reflect
 * this change
 */
__device__ static inline group_t
play_one_neighbor(coord_t coord, enum stone color, enum stone other_color,
        coord_t c, group_t group)
{
    enum stone ncolor = nth_stone(c);
    group_t ngroup = nth_group(c);

    inc_neighbor_count(c, color);

    if (!ngroup)
        return group;

    cuboard_group_rmlib(ngroup, coord);
    if (ncolor == color && ngroup != group) {
        if (!group) {
            group = ngroup;
            add_to_group(group, c, coord);
        } else {
            merge_groups(group, ngroup);
        }
    } else if (is_group_captured(ngroup) && ncolor== other_color ) {
        cuboard_group_capture(ngroup);
    }
    return group;
}

/**
 * We played on a place with at least one liberty. We will become a member of
 * some group for sure. 
 */
__device__ static group_t
cuboard_play_outside(enum stone color, coord_t coord, int f)
{
    enum stone other_color = custone_other(color);
    group_t group = 0;

    my_flen--;
    nth_free(f) = nth_free(my_flen);
    for_each_neighbor(coord, {
        group = play_one_neighbor(coord, color, other_color, c, group);
    });

    nth_stone(coord) = color;
    if (!group)
        group = new_group(coord);
    return group;
}

/**
 * We played in an eye-like shape. Either we capture at least one of the eye
 * sides in the process of playing, or return -1. For invalid move
 */
__device__ static int 
cuboard_play_in_eye(enum stone color, coord_t coord, int f)
{
    int captured_groups = 0;

    for_each_neighbor(coord, {
        group_t g = nth_group(c);
        captured_groups += (lib_count(g) == 1);
    });

    if (captured_groups == 0) {
        return -1;
    }
    nth_free(f) = nth_free(--my_flen);
    for_each_neighbor(coord, {
        inc_neighbor_count(c, color);
        group_t group = nth_group(c);
        if (!group)
            continue;
        cuboard_group_rmlib(group, coord);
        if (is_group_captured(group)) {
            cuboard_group_capture(group);
        }
    });
    nth_stone(coord) = color;
    group_t group = new_group(coord);
    return !!group;
}

/**
 * play color at coord which corresponds to index f
 * in the free queue
 */
__device__ static int __attribute__((flatten))
cuboard_play_f(enum stone color, coord_t coord, int f)
{
    if (!cuboard_is_eyelike(coord, custone_other(color))) {
		 /*NOT playing in an eye. Thus this move has to succeed. (This*/
         /*is thanks to New Zealand rules. Otherwise, multi-stone*/
		 /*suicide might fail.) */
        group_t group = cuboard_play_outside(color, coord, f);
        if (is_group_captured(group)) {
            cuboard_group_capture(group);
        }
        return 0;
    } else {
        return cuboard_play_in_eye(color, coord, f);
    }
}

/**
 * play a color at coord, return 0 if coord was pass or resign,
 * and -1 if the move is invalid. Otherwise return > 0
 */
__device__ int 
cuboard_play(enum stone color, coord_t coord)
{
    if (IS_PASS(coord) || IS_RESIGN(coord))
        return 0;
    int f;
    for (f = 0; f < my_flen; f++)
        if (nth_free(f) == coord)
            return cuboard_play_f(color, coord, f);
    return -1;
}

/**
 * try a random move, if it does not succeed, try try again
 */
__device__ static inline bool
cuboard_try_random_move(enum stone color, coord_t *coord, int f)
{
    *coord = nth_free(f);
    if (cuboard_is_one_point_eye(*coord, color) /* bad idea to play into one, usually */
        || !cuboard_is_valid_play(color, *coord))
        return false;
    return cuboard_play_f(color, *coord, f) >= 0;
}

/**
 * generate a random move for the given color, the resulting move will be
 * stored in coord
 */
__device__ void cuboard_play_random(enum stone color, coord_t *coord)
{
    if (my_flen != 0){
        int f;
        int base = curand_uniform(&randStates[bid]) * my_flen;
        /*assert(base >= 0);*/
        /*assert(base < my_flen);*/
        assert(82 > my_flen);
        for (f = base; f < my_flen; f++)
            if (cuboard_try_random_move(color, coord, f))
                return;
        for (f = 0; f < base; f++)
            if (cuboard_try_random_move(color, coord, f))
                return;
    }
    *coord = PASS;
    cuboard_play(color, *coord);
}

/**
 * generate a naive score, misses some uncommon shapes but is very fast
 * does not account for komi or handicap
 */
__device__ floating_t 
cuboard_fast_score()
{
    int scores[S_MAX];
    memset(scores, 0, sizeof(scores));

    for_each_point {
        enum stone color = nth_stone(c);
        if(color == S_NONE)
            color = cuboard_get_one_point_eye(c);
        scores[color]++;
    } for_each_point_end;

    return scores[S_WHITE] - scores[S_BLACK];
}

/**
 * return true if all the diagonals are one color
 */
__device__ bool 
cuboard_is_false_eyelike(coord_t coord, enum stone eye_color)
{
    int color_diag_libs[S_MAX] = {0, 0, 0, 0};
	 /*XXX: We attempt false eye detection but we will yield false*/
	 /*positives in case of http://senseis.xmp.net/?TwoHeadedDragon :-( */

    color_diag_libs[nth_stone((coord-b_size) -1)]++;
    color_diag_libs[nth_stone((coord-b_size) +1)]++;
    color_diag_libs[nth_stone((coord+b_size) -1)]++;
    color_diag_libs[nth_stone((coord+b_size) +1)]++;

	 /*For false eye, we need two enemy stones diagonally in the*/
     /*middle of the board, or just one enemy stone at the edge*/
	 /*or in the corner. */
    color_diag_libs[custone_other(eye_color)] += !!color_diag_libs[S_OFFBOARD];
    return color_diag_libs[custone_other(eye_color)] >= 2;
}

/**
 * returns true  if position is a single eye
 */
__device__ bool 
cuboard_is_one_point_eye(coord_t c, enum stone eye_color)
{
    return cuboard_is_eyelike(c, eye_color)
        && !cuboard_is_false_eyelike(c, eye_color);
}

/**
 * returns the color of a single eye
 */
__device__ enum stone
cuboard_get_one_point_eye(coord_t c)
{
    if (cuboard_is_one_point_eye(c, S_WHITE))
        return S_WHITE;
    else if (cuboard_is_one_point_eye(c, S_BLACK))
        return S_BLACK;
    else
        return S_NONE;
}

/**
 * true if point is surrounded by all one color
 */
__device__ inline bool
cuboard_is_eyelike(coord_t coord, enum stone eye_color)
{
    return (neighbor_count(coord, eye_color)
            + neighbor_count(coord, S_OFFBOARD)) == 4;
}

/**
 * true if move is a valid play, does not account for ko
 */
__device__ inline bool
cuboard_is_valid_play(enum stone color, coord_t coord)
{
    if (nth_stone(coord) != S_NONE)
        return false;
    if (!cuboard_is_eyelike(coord, custone_other(color)))
        return true;
    int groups_in_atari = 0;
    for_each_neighbor(coord, {
        group_t g = nth_group(c);
        groups_in_atari += (lib_count(g) == 1);
    });
    return !!groups_in_atari;
}

#include "kokrusher/cuboard.h"

extern __device__ int g_flen[M*N];
extern __device__ stone (*g_b)[M*N];
extern __device__ coord_t (*g_f)[M*N];
extern __device__ coord_t (*g_p)[M*N];
extern __device__ group_t (*g_g)[M*N];
extern __device__ int (*g_libs)[M*N];
extern __device__ coord_t (*g_gi)[GROUP_KEEP_LIBS][M*N];
extern __device__ int g_caps[S_MAX][M*N];
extern __device__ char (*g_ncol)[S_MAX][M*N];

__device__ void 
cuboard_init(int size) 
{
    int i;
    for (i=0; i<S_MAX;i++)
        captures(i) = 0;
    /* Draw the offboard margin */
    int top_row = (size*size) - size;
    for (i = 0; i < size; i++)
        nth_stone(i) = nth_stone(top_row + i) = S_OFFBOARD;
    for (i = 0; i <= top_row; i += size)
        nth_stone(i) = nth_stone((size-1) + i) = S_OFFBOARD;
    for_each_point(size) {
        coord_t coord = c;
        if (nth_stone(coord) == S_OFFBOARD)
            continue;
        for_each_neighbor(size, c, {
            inc_neighbor_count(coord, nth_stone(c));
        } );
    } for_each_point_end;

    //all non margin points are free
    my_flen = 0;
    for (i = size; i < (size - 1) * size; i++)
        if (i % size != 0 && i % size != size - 1)
            nth_free(my_flen++) = i;
}

__device__ void 
cuboard_copy(void* data, int size)
{
    int size2 = size * size;
    int i,j;
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

__device__ static void
cuboard_group_find_extra_libs(group_t g, coord_t avoid, int sz)
{
    /* Add extra liberty from the board to our liberty list. */
    int len = (sz*sz) / 8;
    unsigned char *watermark = (unsigned char *) cMalloc(sizeof(unsigned char) * len);
    for (int i=0; i<len; i++)
        watermark[i]=0;
#define watermark_get(cx)	(watermark[cx >> 3] & (1 << (cx & 7)))
#define watermark_set(cx)	(watermark[cx >> 3] |= (1 << (cx & 7)))

    for (int i = 0; i < GROUP_KEEP_LIBS - 1; i++)
        watermark_set(nth_lib(g,i));
    watermark_set(avoid);

    for_each_in_group(g) {
        coord_t coord2 = c;
        for_each_neighbor(sz, coord2, {
            if (nth_stone(c) + watermark_get(c) != S_NONE)
                continue;
            watermark_set(c);
            nth_lib(g, lib_count(g)) = c;
            lib_count(g) = lib_count(g) + 1;
            if (lib_count(g) >= GROUP_KEEP_LIBS)
                return;
        } );
    } for_each_in_group_end;
    free(watermark);
#undef watermark_get
#undef watermark_set
}

__device__ static void
cuboard_group_rmlib(group_t g, coord_t coord, int size)
{
    for (int i = 0; i < lib_count(g) && i < GROUP_KEEP_LIBS; i++) {
        if (nth_lib(g,i) == coord){
            lib_count(g)--;
            nth_lib(g,i) = nth_lib(g,lib_count(g));
            nth_lib(g,lib_count(g)) = 0;
            if (lib_count(g) == GROUP_REFILL_LIBS)
                cuboard_group_find_extra_libs(g, coord, size);
            return;
        }
    }
}


/* This is a low-level routine that doesn't maintain consistency
 * of all the board data structures. */
__device__ static void
cuboard_remove_stone(group_t group, coord_t c, int size)
{
    enum stone color = nth_stone(c);
    nth_stone(c) = S_NONE;
    nth_group(c) = 0;
    /* Increase liberties of surrounding groups */
    coord_t coord = c;
    for_each_neighbor(size, coord, {
        dec_neighbor_count(c, color);
        group_t g = nth_group(c);
        if (g && g != group){
            cuboard_group_addlib(g, coord);
        }
    });

    nth_free(my_flen) = c;
    my_flen++;
}

__device__ static int 
cuboard_group_capture(group_t group, int size)
{
    int stones = 0;

    for_each_in_group(group) {
        captures(custone_other(nth_stone(c)))++;
        cuboard_remove_stone(group, c, size);
        stones++;
    } for_each_in_group_end;

    assert(lib_count(group) == 0);
    for (int i=0; i<GROUP_KEEP_LIBS; i++)
        nth_lib(group,i) = 0;
    return stones;
}

__device__ static void 
add_to_group(group_t group, coord_t prevstone, coord_t coord, int size)
{
    nth_group(coord) = group;
    next_group(coord) = next_group(prevstone);
    next_group(prevstone) = coord;
    for_each_neighbor(size, coord, {
        if (nth_stone(c) == S_NONE)
            cuboard_group_addlib(group, c);
    });
}

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

__device__ static group_t
new_group(coord_t coord, int size)
{
    group_t g = coord;
    for_each_neighbor(size, coord, {
        if (nth_stone(c) == S_NONE)
            nth_lib(g,lib_count(g)) = c;
            lib_count(g)++;
    });
    nth_group(coord) = g;
    next_group(coord) = 0;
    return g;
}

__device__ static inline group_t
play_one_neighbor(coord_t coord, enum stone color, enum stone other_color,
        coord_t c, group_t group, int size)
{
    enum stone ncolor = nth_stone(c);
    group_t ngroup = nth_group(c);

    inc_neighbor_count(c, color);

    if (!ngroup)
        return group;

    cuboard_group_rmlib(ngroup, coord, size);
    if (ncolor == color && ngroup != group) {
        if (!group) {
            group = ngroup;
            add_to_group(group, c, coord, size);
        } else {
            merge_groups(group, ngroup);
        }
    } else if (is_group_captured(ngroup) && ncolor== other_color ) {
        cuboard_group_capture(ngroup, size);
    }
    return group;
}

/* We played on a place with at least one liberty. We will become a member of
 * some group for sure. */
__device__ static group_t
cuboard_play_outside(enum stone color, coord_t coord, int f, int size)
{
    enum stone other_color = custone_other(color);
    group_t group = 0;

    my_flen--;
    nth_free(f) = nth_free(my_flen);
    for_each_neighbor(size, coord, {
        group = play_one_neighbor(coord, color, other_color, c, group, size);
    });

    nth_stone(coord) = color;
    if (!group)
        group = new_group(coord, size);
    return group;
}

/* We played in an eye-like shape. Either we capture at least one of the eye
 * sides in the process of playing, or return -1. */
__device__ static int 
cuboard_play_in_eye(enum stone color, coord_t coord, int f, int size)
{
    int captured_groups = 0;

    for_each_neighbor(size, coord, {
        group_t g = nth_group(c);
        captured_groups += (lib_count(g) == 1);
    });

    if (captured_groups == 0) {
        return -1;
    }
    nth_free(f) = nth_free(--my_flen);
    for_each_neighbor(size, coord, {
        inc_neighbor_count(c, color);
        group_t group = nth_group(c);
        if (!group)
            continue;
        cuboard_group_rmlib(group, coord, size);
        if (is_group_captured(group)) {
            cuboard_group_capture(group,size);
        }
    });
    nth_stone(coord) = color;
    group_t group = new_group(coord, size);
    return !!group;
}


__device__ static int __attribute__((flatten))
cuboard_play_f(enum stone color, coord_t coord, int f, int size)
{
    if (!cuboard_is_eyelike(coord, custone_other(color))) {
		 /*NOT playing in an eye. Thus this move has to succeed. (This*/
         /*is thanks to New Zealand rules. Otherwise, multi-stone*/
		 /*suicide might fail.) */
        group_t group = cuboard_play_outside(color, coord, f, size);
        if (is_group_captured(group)) {
            cuboard_group_capture(group, size);
        }
        return 0;
    } else {
        return cuboard_play_in_eye(color, coord, f, size);
    }
}

__device__ int 
cuboard_play(enum stone color, coord_t coord, int size)
{
    if (IS_PASS(coord) || IS_RESIGN(coord))
        return 0;
    int f;
    for (f = 0; f < my_flen; f++)
        if (nth_free(f) == coord)
            return cuboard_play_f(color, coord, f, size);
    return -1;
}

__device__ static inline bool
cuboard_try_random_move(enum stone color, coord_t *coord, int f, int size)
{
    *coord = nth_free(f);
    if (cuboard_is_one_point_eye(*coord, color, size) /* bad idea to play into one, usually */
        || !cuboard_is_valid_play(color, *coord, size))
        return false;
    return cuboard_play_f(color, *coord, f, size) >= 0;
}

__device__ void cuboard_play_random(enum stone color, coord_t *coord, curandState rState, int size)
{
    if (my_flen != 0){
        int f;
        int base = curand_uniform(&rState) * my_flen;
        /*assert(base >= 0);*/
        /*assert(base < my_flen);*/
        assert(82 > my_flen);
        for (f = base; f < my_flen; f++)
            if (cuboard_try_random_move(color, coord, f, size))
                return;
        for (f = 0; f < base; f++)
            if (cuboard_try_random_move(color, coord, f, size))
                return;
    }
    *coord = PASS;
    cuboard_play(color, *coord, size);
}

//DOES NOT COUNT KOMI OR HANDICAP
__device__ floating_t 
cuboard_fast_score(int size)
{
    int scores[S_MAX];
    memset(scores, 0, sizeof(scores));

    for_each_point(size) {
        enum stone color = nth_stone(c);
        if(color == S_NONE)
            color = cuboard_get_one_point_eye(c,size);
        scores[color]++;
    } for_each_point_end;

    return scores[S_WHITE] - scores[S_BLACK];
}

__device__ bool 
cuboard_is_false_eyelike(coord_t coord, enum stone eye_color, int size)
{
    int color_diag_libs[S_MAX] = {0, 0, 0, 0};
	 /*XXX: We attempt false eye detection but we will yield false*/
	 /*positives in case of http://senseis.xmp.net/?TwoHeadedDragon :-( */

    color_diag_libs[nth_stone((coord-size) -1)]++;
    color_diag_libs[nth_stone((coord-size) +0)]++;
    color_diag_libs[nth_stone((coord+size) -1)]++;
    color_diag_libs[nth_stone((coord+size) +1)]++;

	 /*For false eye, we need two enemy stones diagonally in the*/
     /*middle of the board, or just one enemy stone at the edge*/
	 /*or in the corner. */
    color_diag_libs[custone_other(eye_color)] += !!color_diag_libs[S_OFFBOARD];
    return color_diag_libs[custone_other(eye_color)] >= 2;
}

__device__ bool 
cuboard_is_one_point_eye(coord_t c, enum stone eye_color, int size)
{
    return cuboard_is_eyelike(c, eye_color)
        && !cuboard_is_false_eyelike(c, eye_color,size);
}

__device__ enum stone
cuboard_get_one_point_eye(coord_t c, int size)
{
    if (cuboard_is_one_point_eye(c, S_WHITE, size))
        return S_WHITE;
    else if (cuboard_is_one_point_eye(c, S_BLACK, size))
        return S_BLACK;
    else
        return S_NONE;
}

__device__ inline bool
cuboard_is_eyelike(coord_t coord, enum stone eye_color)
{
    return (neighbor_count(coord, eye_color)
            + neighbor_count(coord, S_OFFBOARD)) == 4;
}

__device__ inline bool
cuboard_is_valid_play(enum stone color, coord_t coord, int size)
{
    if (nth_stone(coord) != S_NONE)
        return false;
    if (!cuboard_is_eyelike(coord, custone_other(color)))
        return true;
    int groups_in_atari = 0;
    for_each_neighbor(size, coord, {
        group_t g = nth_group(c);
        groups_in_atari += (lib_count(g) == 1);
    });
    return !!groups_in_atari;
}

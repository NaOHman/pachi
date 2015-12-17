#ifndef PACHI_CUBOARD_H
#define PACHI_CUBOARD_H
#ifdef  __cplusplus

#include <cuda.h>
#include <curand_kernel.h>
#include <assert.h>

#define board_data_size(sz_) (((5 + S_MAX+GROUP_KEEP_LIBS) * (sz_) * (sz_)) + S_MAX + 1) 
#define cMalloc(c) check_malloc(c)
extern "C" {
#endif

#include "util.h"
#include "stone.h"

#ifdef  __cplusplus
}
#endif

#define M 64
#define N 64
#define BOARD_MAX_SIZE 19

#define IS_PASS(c) (c == -1)
#define PASS -1
#define IS_RESIGN(c) (c == -2)
#define RESIGN -2

typedef int coord_t;
typedef coord_t group_t;

#define GROUP_KEEP_LIBS 10
#define GROUP_REFILL_LIBS 5

#define bid (threadIdx.x + (blockIdx.x * blockIdx.x))
#define nth_free(c_) (g_f[c_][bid])
#define my_flen (g_flen[bid])
#define next_group(c_) (g_p[c_][bid])
#define nth_group(i_) (g_g[i_][bid])
#define nth_lib(g_,i_) (g_gi[i_][g_][bid])
#define lib_count(g_) (g_libs[g_][bid])
#define nth_stone(i_) (g_b[i_][bid])
#define captures(c_) (g_g[c_][bid])
#define neighbor_count(i_,c_) (g_ncol[i_][c_][bid])

/* Warning! Neighbor count is not kept up-to-date for S_NONE! */
#define set_neighbor_count(coord, color, count) (neighbor_count(coord, color) = (count))
#define inc_neighbor_count(coord, color) (neighbor_count(coord, color)++)
#define dec_neighbor_count(coord, color) (neighbor_count(coord, color)--)
#define immediate_libs(coord) (4 - neighbor_count(coord, S_BLACK) - neighbor_count(coord, S_WHITE) - neighbor_count(coord, S_OFFBOARD))



#define is_group_captured(g_) (lib_count(g_) == 0)
#define for_each_point \
    do { \
        coord_t c = 0; \
        for (; c < b_size * b_size; c++)
#define for_each_point_and_pass \
    do { \
        coord_t c = pass; \
        for (; c < b_size * b_size; c++)
#define for_each_point_end \
    } while (0)

#define for_each_free_point \
    do { \
        int fmax__ = g_flen[bid]; \
        for (int f__ = 0; f__ < fmax__; f__++) { \
            coord_t c = g_f[f__][bid];
#define for_each_free_point_end \
        } \
    } while (0)

#define for_each_in_group(group_) \
    do { \
        coord_t c = (group_); \
        coord_t c2 = c; c2 = next_group(c2); \
        do {
#define for_each_in_group_end \
            c = c2; c2 = next_group(c2); \
        } while (c != 0); \
    } while (0)

/* NOT VALID inside of for_each_point() or another for_each_neighbor(), or rather
 * on S_OFFBOARD coordinates. */
#define for_each_neighbor(coord_, loop_body) \
    do { \
        coord_t coord__ = coord_; \
        coord_t c; \
        c = coord__ - b_size; do { loop_body } while (0); \
        c = coord__ - 1; do { loop_body } while (0); \
        c = coord__ + 1; do { loop_body } while (0); \
        c = coord__ + b_size; do { loop_body } while (0); \
    } while (0)

__device__ __host__ inline enum stone custone_other(enum stone s);

__device__ void cuboard_init();

__device__ void cuboard_copy();

__device__ int cuboard_play(enum stone color, coord_t coord);

__device__ void cuboard_play_random(enum stone color, coord_t *coord);

__device__ static bool cuboard_is_valid_play(enum stone color, coord_t coord);

__device__ floating_t cuboard_fast_score();

__device__ static bool cuboard_is_eyelike(coord_t coord, enum stone eye_color);

__device__ bool cuboard_is_false_eyelike(coord_t coord, enum stone eye_color);

__device__ bool cuboard_is_one_point_eye(coord_t c, enum stone eye_color);

__device__ enum stone cuboard_get_one_point_eye(coord_t c);

__device__ __host__ inline enum stone __attribute__((always_inline)) 
custone_other(enum stone s)
{
    switch (s) {
        case 0:
            return S_NONE;
        case 1:
            return S_WHITE;
        case 2:
            return S_BLACK;
        default:
            return S_OFFBOARD;
    }
}

__device__ static inline void *
check_malloc(size_t size)
{
	void *p = malloc(size);
	if (!p) {
        assert(false);
	}
	return p;
}
#endif

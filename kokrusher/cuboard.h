#ifndef PACHI_CUBOARD_H
#define PACHI_CUBOARD_H
#ifdef  __cplusplus

#include <cuda.h>
#include <curand_kernel.h>
#include <assert.h>

#define cMalloc(c) check_malloc(c)
extern "C" {
#endif

#include "util.h"
#include "stone.h"

#ifdef  __cplusplus
}
#endif

#define M 256
#define N 256
#define BOARD_MAX_SIZE 19

#define IS_PASS(c) (c == -1)
#define PASS -1
#define IS_RESIGN(c) (c == -2)
#define RESIGN -2

typedef int coord_t;
typedef int bid_t;
typedef coord_t group_t;

#define GROUP_KEEP_LIBS 10
#define GROUP_REFILL_LIBS 5

#define bid (threadIdx.x + (blockIdx.x * gridDim.x))
#define nth_free(c_) (g_f[c_][bid])
#define my_flen (g_flen[bid])
#define groupnext_at(c_) (g_p[c_][bid])
#define group_at(i_) (g_g[i_][bid])
#define nth_lib(g_,i_) (g_gi[i_][g_][bid])
#define lib_count(g_) (g_libs[g_][bid])
#define board_at(i_) (g_b[i_][bid])
#define captures(c_) (g_g[c_][bid])
#define neighbor_count_at(i_,c_) (g_ncol[i_][c_][bid])

/* Warning! Neighbor count is not kept up-to-date for S_NONE! */
#define set_neighbor_count_at(coord, color, count) (neighbor_count_at(coord, color) = (count))
#define inc_neighbor_count_at(coord, color) (neighbor_count_at(coord, color)++)
#define dec_neighbor_count_at(coord, color) (neighbor_count_at(coord, color)--)
#define immediate_liberty_count(coord) (4 - neighbor_count_at(coord, S_BLACK) - neighbor_count_at(coord, S_WHITE) - neighbor_count_at(coord, S_OFFBOARD))


//#define groupnext_atxy(b_, x, y) ((b_)->p[(x) + board_size(b_) * (y)])

#define group_base(g_) (g_)
#define group_is_onestone(b_, g_) (groupnext_at(b_, group_base(g_)) == 0)
//#define board_group_info(b_, g_) ((b_)->gi[(g_)])
#define board_group_captured(g_) (lib_count(g_) == 0)
#define board_group_other_lib(b_, g_, l_) (nth_lib(g_,[nth_lib(g_,[0]) != (l_) ? 0 : 1]))

//#define foreach_point(size_)
//#define foreach_point_and_pass(size_) 
//#define foreach_neighbor(size, coord_, loop_body) 
//#define foreach_8neighbor(coord_)
//#define foreach_in_group(group_)
//#define foreach_diag_neighbor(coord_) 
#define foreach_point(size_) \
    do { \
        coord_t c = 0; \
        for (; c < size_ * size_; c++)
#define foreach_point_and_pass(size_) \
    do { \
        coord_t c = pass; \
        for (; c < size_ * size_; c++)
#define foreach_point_end \
    } while (0)

#define foreach_free_point \
    do { \
        int fmax__ = g_flen[bid]; \
        for (int f__ = 0; f__ < fmax__; f__++) { \
            coord_t c = g_f[f__][bid];
#define foreach_free_point_end \
        } \
    } while (0)

#define foreach_in_group(group_) \
    do { \
        coord_t c = group_base(group_); \
        coord_t c2 = c; c2 = groupnext_at(c2); \
        do {
#define foreach_in_group_end \
            c = c2; c2 = groupnext_at(c2); \
        } while (c != 0); \
    } while (0)

/* NOT VALID inside of foreach_point() or another foreach_neighbor(), or rather
 * on S_OFFBOARD coordinates. */
#define foreach_neighbor(size_, coord_, loop_body) \
    do { \
        coord_t coord__ = coord_; \
        coord_t sz_ = size_; \
        coord_t c; \
        c = coord__ - sz_; do { loop_body } while (0); \
        c = coord__ - 1; do { loop_body } while (0); \
        c = coord__ + 1; do { loop_body } while (0); \
        c = coord__ + sz_; do { loop_body } while (0); \
    } while (0)

__device__ __host__ inline enum stone custone_other(enum stone s);

__device__ void cuboard_init(int size);

__device__ void cuboard_copy(bid_t b1);

__device__ int cuboard_play(enum stone color, coord_t coord, int size);

__device__ void cuboard_play_random(enum stone color, coord_t *coord, curandState rState, int size);

__device__ static bool cuboard_is_valid_play(enum stone color, coord_t coord, int size);

__device__ floating_t cuboard_fast_score(int size);

__device__ static bool cuboard_is_eyelike(coord_t coord, enum stone eye_color);

__device__ bool cuboard_is_false_eyelike(coord_t coord, enum stone eye_color, int size);

__device__ bool cuboard_is_one_point_eye(coord_t c, enum stone eye_color, int size);

__device__ enum stone cuboard_get_one_point_eye(coord_t c, int size);

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

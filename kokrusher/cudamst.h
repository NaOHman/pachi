#ifndef PACHI_KOKRUSHER_CUDA_H
#define PACHI_KOKRUSHER_CUDA_H

#ifdef __cplusplus
extern "C" int *cuda_genmove(struct board *b, struct time_info *ti, enum stone color);
extern "C" void init_kokrusher(struct board *b);
void copy_essential_board_data(struct board * b);
void __checkerr(cudaError_t e, char * file, int line);
__global__ void cuda_rand_init(unsigned long seed);
__device__ int cuda_play_random_game(enum stone starting_color, int moves, int passes, float offset);
#else
coord_t *cuda_genmove(struct board *b, struct time_info *ti, enum stone color);
void init_kokrusher(struct board *b);
#endif

#endif

#ifndef PACHI_KOKRUSHER_CUDA_H
#define PACHI_KOKRUSHER_CUDA_H

#ifdef __cplusplus
extern "C" int *cuda_genmove(void *data, struct board *b, struct time_info *ti, enum stone color);
extern "C" void * init_kokrusher(struct board *b);
#else
coord_t *cuda_genmove(void *data, struct board *b, struct time_info *ti, enum stone color);
void * init_kokrusher(struct board *b);
#endif

#endif

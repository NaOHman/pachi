#ifndef PACHI_KOKRUSHER_CUDA_H
#define PACHI_KOKRUSHER_CUDA_H

#ifdef __cplusplus
extern "C" coord_t *cuda_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive);
extern "C" void say_hello();
#else
coord_t *cuda_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive);
void say_hello();
#endif

#endif

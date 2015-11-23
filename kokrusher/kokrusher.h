#ifndef PACHI_KOKRUSHER_H
#define PACHI_KOKRUSHER_H

#include "engine.h"

#ifdef __cplusplus
extern "C" static coord_t * kokrusher_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive);
extern "C" struct engine *engine_kokrusher_init(char *arg, struct board *b);
#else
static coord_t * kokrusher_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive);
struct engine *engine_kokrusher_init(char *arg, struct board *b);
#endif

#endif

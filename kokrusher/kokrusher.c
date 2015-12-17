#include <stdio.h>
#include <stdlib.h>

#include "kokrusher/kokrusher.h"

#ifdef CUDA
#include "kokrusher/mcts.h"
#else

#endif

/**
 * Generate a move on CUDA, will fail if not compiled with CUDA support
 */
static coord_t *
kokrusher_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive)
{ // this method is equivalent to the UctSearch pseudocode method
#ifdef CUDA
    //If cuda support is enabled generate the move with cuda
    return cuda_genmove(b, ti, color);
#else
    //No cuda support, resign
    fprintf(stderr, "CUDA not supported, please rebuild with the CUDA flag\n");
    *coord = -2;
    return coord;
#endif
}

/**
 * Setup the CUDA engine, mostly allocating global arrays
 */
struct engine *
engine_kokrusher_init(char *arg, struct board *b)
{
    struct engine *e = calloc2(1, sizeof(struct engine));
    e->name = "kokrusher";
    e->comment = "Macalester's own kokrusher";
    e->genmove = kokrusher_genmove;
#ifdef CUDA
    init_kokrusher(b);
#endif
    return e;
}

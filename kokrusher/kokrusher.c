#include <stdio.h>
#include <stdlib.h>

#include "board.h"
#include "engine.h"
#include "debug.h"
#include "move.h"
#include "kokrusher/kokrusher.h"

#ifdef CUDA
#include "kokrusher/cudamst.h"
#endif

static coord_t *
kokrusher_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive)
{ // this method is equivalent to the UctSearch pseudocode method
#ifdef CUDA
    //If cuda support is enabled generate the move with cuda
    fprintf(stderr, "CUDA KoKrusher\n");
    return cuda_genmove(e, b, ti, color, pass_all_alive);
#else
    //No cuda support just do your own thing
    fprintf(stderr, "Sequential KoKrusher\n");
    coord_t *coord;
    struct board b2;
    board_copy(&b2, b);
    for (int i=0; i<=19; i++){
        for (int j=0; j<=19; j++){
            coord = coord_init(i,j,board_size(&b2));
            if (board_is_valid_play(&b2, color, *coord))
                return coord;
        }
    }
    *coord = -1;
    return coord;
#endif
}

struct engine *
engine_kokrusher_init(char *arg, struct board *b)
{
    struct engine *e = calloc2(1, sizeof(struct engine));
    e->name = "kokrusher";
    e->comment = "Macalester's own kokrusher";
    e->genmove = kokrusher_genmove;
    return e;
}

typedef struct node {

}


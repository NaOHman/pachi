#include <stdio.h>
#include <stdlib.h>

#include "board.h"
#include "engine.h"
#include "move.h"
#include "proof/proof.h"

static coord_t *
proof_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive)
{
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
}

struct engine *
engine_proof_init(char *arg, struct board *b)
{
    struct engine *e = calloc2(1, sizeof(struct engine));
    e->name = "proof";
    e->comment = "Proof that we can make an engine";
    e->genmove = proof_genmove;
    return e;
}

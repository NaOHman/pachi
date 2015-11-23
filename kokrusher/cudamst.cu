/* This is the REAL "hello world" for CUDA!  * It takes the string "Hello ", prints it, then passes it to CUDA with an array
 * of offsets. Then the offsets are added in parallel to produce the string "World!"
 * By Ingemar Ragnemalm 2010
 */
 
/*using namespace std;*/

#include <cstdio>
#ifdef __cplusplus
extern "C" {
#endif

#include "board.h"
#include "engine.h"
#include "debug.h"
#include "move.h"

#ifdef __cplusplus
}
#endif

#include "kokrusher/cudamst.h"

//#ifdef __cplusplus
//}
//#endif

#define MAX_PLAYS 400

const int N = 16; 
const int blocksize = 16; 

// this method roughly corresponds to the Simulate method from the pseudocode
__global__ void running_every_thread(struct board *b, enum stone color, int[] votes){
    struct *myBoard  = malloc(sizeof(struct board));
    board_copy(myBoard, b);
    //make first move according to blckid and tid
    //where uct happens: SimTree method from pseudocode
    int win = cuda_play_random_game(myBoard, color);
    //record move: Backup method from pseudocode
}

// this method corresponds to SimDefault in the pseudocode
__device__ int cuda_play_random_game(struct board *b, enum stone starting_color)
{
	int gamelen = MAX_PLAYS - b->moves;
	enum stone color = starting_color;
	int passes = is_pass(b->last_move.coord) && b->moves > 0;

	while (gamelen-- && passes < 2) {
		color = stone_other(color);
		coord_t coord = cuda_play_random_move(b, color);
		if (unlikely(is_pass(coord))) {
			passes++;
		} else {
			passes = 0;
		}
	}

	floating_t score = board_fast_score(b);
	int result = (starting_color == S_WHITE ? score * 2 : - (score * 2));
	if (b->ps)
		free(b->ps);
	return result;
}

__device coord_t cuda_play_random_move(struct board *b, enum stone color){
	coord_t coord;
	if (unlikely(b->flen == 0)) {
        struct move m = { pass, color };
        board_play(b, &m);
        return pass;
    }
	int base = fast_random(b->flen), f;
	for (f = base; f < b->flen; f++)
		if (cuda_try_random_move(b, color, &coord, f))
			return coord;
	for (f = 0; f < base; f++)
		if (cuda_try_random_move(b, color, &coord, f))
			return coord;
}

static inline bool
cuda_try_random_move(struct board *b, enum stone color, coord_t *coord, int f)
{
	*coord = b->f[f];
	struct move m = { *coord, color };
	if (unlikely(board_is_one_point_eye(b, *coord, color)) /* bad idea to play into one, usually */
		|| !board_is_valid_move(b, &m))
		return false;
	if (m.coord == *coord) {
		return likely(board_play_f(b, &m, f) >= 0);
	} else {
		*coord = m.coord; // permit modified the coordinate
		return likely(board_play(b, &m) >= 0);
	}
}

// this going to correspond to the UctSearch method in the pseudocode from the literature
coord_t *cuda_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive){
    //create array of possible moves
    //copy board to device
    //copy array to device
    //launch kernel
    //copy array back
    //pick winning move
}

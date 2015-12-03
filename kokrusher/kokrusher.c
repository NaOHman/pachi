#include <stdio.h>
#include <stdlib.h>

#include "kokrusher/kokrusher.h"

#ifdef CUDA
#include "kokrusher/cudamst.h"
#else
/*#include "board.h"*/
/*#include "debug.h"*/
/*#include "move.h"*/

#endif

static coord_t *
kokrusher_genmove(struct engine *e, struct board *b, struct time_info *ti, enum stone color, bool pass_all_alive)
{ // this method is equivalent to the UctSearch pseudocode method
#ifdef CUDA
    //If cuda support is enabled generate the move with cuda
    fprintf(stderr, "CUDA KoKrusher\n");
    if (e->data == NULL)
        printf("Data is NULL!\n");
    return cuda_genmove(e->data, b, ti, color);
#else
    //No cuda support just do your own thing
    /*fprintf(stderr, "Sequential KoKrusher\n");*/
    /*coord_t *coord;*/
    /*struct board b2;*/
    /*board_copy(&b2, b);*/
    /*int i,j;*/
    /*for (i=0; i<=19; i++){*/
        /*for (j=0; j<=19; j++){*/
            /*coord = coord_init(i,j,board_size(&b2));*/
            /*if (board_is_valid_play(&b2, color, *coord))*/
                /*return coord;*/
        /*}*/
    /*}*/
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
#ifdef CUDA
    e->data = init_kokrusher(b);
#endif
    return e;
}

/*struct tree_node {*/
		/*struct tree_node* parent;*/
		/*struct tree_node** children;*/
		/*int* child_visits;*/
		/*float* child_q;*/
/*};*/

/*float uct(float q, int parent_visits, int child_visits) {*/
		/*// q + c * sqrt(log(parent_visits)/child_visits)*/
		/*// c is an arbitrary predefined constant*/
		/*// need to handle child_visits == 0 also*/
		/*// need to handle possibly illegal moves -> output 0*/
		/*return q;*/
/*}*/

/*float rouletteScalingFunction(float uct) {*/
		/*// this function needs to convert the UCT value into */
		/*// the value we want for the roulette selection*/
		/*// possibly an exponential function or a basic polynomial function?*/
		/*// 0 should probably output as 0, so maybe not an exponential function?*/
		/*return uct;*/
/*}*/


/*int rouletteSelect(int parent_visits, float* q_array, int* child_visits_array, int n) {*/
		/*float total = 0.0;*/
		/*float* uctArray; // I forget how to init this*/
		/*for (int i=0; i<n*n; i++) { // iterate over length of array: n^2*/
				/*float temp = uct(q_array[i], parent_visits, child_visits_array[i]);*/
				/*temp = rouletteScalingFunction(temp);*/
				/*total += temp;*/
				/*uctArray[i] = temp;*/
		/*}*/
		/*float random = 0.0; // set up random number generation here, assuming from [0,1)*/
		/*float pos = random * total;*/
		/*for (int i=0; i<n*n; i++) {*/
				/*pos -= uctArray[i];*/
				/*if (pos < 0) return i; // this should always return...?*/
		/*}*/
		/*return n*n-1; // in case of emergencies, return final entry*/
/*}*/

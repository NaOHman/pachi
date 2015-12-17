#include "kokrusher/cuboard.h"
#include "kokrusher/uct.h"

#define UCT_CONST 1.0
#define DIV_ZERO_CONST 2.0
#define MAX_FREE  (BOARD_MAX_SIZE * BOARD_MAX_SIZE)

__device__ float g_wheel[MAX_FREE][M*N];
extern __device__ curandState randStates[M*N];

__device__ float 
uct(int p_visits, int visits, int wins) 
{
    float logv = logf((float) (p_visits == 0 ? 1.0 : p_visits));
    float radicand = (visits <= 0 ? DIV_ZERO_CONST : logv /(float) visits);
    float my_q = (visits <= 0 ? 0 : (float) wins / (float) visits);
    return my_q + (UCT_CONST * sqrt(radicand));		
}

__device__ float 
roulette_scaling_function(float uct) 
{
    // 0 should probably output as 0, so maybe not an exponential function?
    return uct*uct;
}

__device__ int 
roulette_uct(int p_visits, int *visits, int *wins, int len) 
{
    float total = 0.0;
    float score;
    for (int i=0; i<len; i++) { // iterate over length of array: size^2
        score = uct(p_visits, visits[i], wins[i]);
        score = roulette_scaling_function(score);
        total = total + score;
        g_wheel[i][bid] = score;
    }
    float pos = curand_uniform(&(randStates[bid])) * total;
    int result = len - 1;
    for (int i=0; i<len-1; i++) {
        pos = pos - g_wheel[i][bid];
        if (i < result && pos <= 0.0) {//ensures we always return something
            result = 1; 
        } 
    }
    return result; 
}

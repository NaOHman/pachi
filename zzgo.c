#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "board.h"
#include "debug.h"
#include "engine.h"
#include "montecarlo/montecarlo.h"
#include "random/random.h"
#include "gtp.h"
#include "random.h"

int debug_level = 1;
int seed;

int main(int argc, char *argv[])
{
	struct board *b = board_init();
	enum { E_RANDOM, E_MONTECARLO } engine = E_MONTECARLO;

	seed = time(NULL);

	int opt;
	while ((opt = getopt(argc, argv, "e:d:s:")) != -1) {
		switch (opt) {
			case 'e':
				if (!strcasecmp(optarg, "random")) {
					engine = E_RANDOM;
				} else if (!strcasecmp(optarg, "montecarlo")) {
					engine = E_MONTECARLO;
				} else {
					fprintf(stderr, "%s: Invalid -e argument %s\n", argv[0], optarg);
					exit(1);
				}
				break;
			case 'd':
				debug_level = atoi(optarg);
				break;
			case 's':
				seed = atoi(optarg);
				break;
			default: /* '?' */
				fprintf(stderr, "Usage: %s [-e random|montecarlo] [-d DEBUG_LEVEL] [-s RANDOM_SEED] [ENGINE_ARGS]\n",
						argv[0]);
				exit(1);
		}
	}

	fast_srandom(seed);
	if (debug_level > 0)
		fprintf(stderr, "Random seed: %d", seed);

	char *e_arg = NULL;
	if (optind < argc)
		e_arg = argv[optind];
	struct engine *e;
	switch (engine) {
		case E_RANDOM:
		default:
			e = engine_random_init(e_arg); break;
		case E_MONTECARLO:
			e = engine_montecarlo_init(e_arg); break;

	}

	char buf[256];
	while (fgets(buf, 256, stdin)) {
		if (debug_level > 1)
			fprintf(stderr, "IN: %s", buf);
		gtp_parse(b, e, buf);
		if (debug_level > 1)
			board_print(b, stderr);
	}
	return 0;
}

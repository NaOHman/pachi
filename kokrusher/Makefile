INCLUDES=-I..

all: kokrusher.a
ifdef CUDA
kokrusher.a: mcts.o cuboard.o kokrusher.o uct.o
	nvcc $(NVCC_ARCH) -lib mcts.o uct.o cuboard.o kokrusher.o -o kokrusher.a
else
kokrusher.a: kokrusher.o
endif

clean:
	rm -f *.o *.a
clean-profiled:
	rm -f *.gcda *.gcno

-include ../Makefile.lib

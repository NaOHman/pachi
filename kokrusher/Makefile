INCLUDES=-I..
OBJS=kokrusher.o

all: kokrusher.a

ifdef CUDA
kokrusher.a: kokrusher.a cudamst.cu
	gcc -I.. -D CUDA -std=c99 -o kokrusher.o -c kokrusher.c
	nvcc -I.. -c cudamst.cu cudamst.o
	nvcc --lib -o kokrusher.a kokrusher.o cudamst.o
else
kokrusher.a: $(OBJS)
endif

clean:
	rm -f *.o *.a
clean-profiled:
	rm -f *.gcda *.gcno

-include ../Makefile.lib

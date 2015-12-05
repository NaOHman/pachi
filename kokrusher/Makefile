INCLUDES=-I..

all: kokrusher.a
ifdef CUDA
kokrusher.a: cudamst.o cuboard.o kokrusher.o
	nvcc -lib cudamst.o cuboard.o kokrusher.o -o kokrusher.a
else
kokrusher.a: kokrusher.o
endif

clean:
	rm -f *.o *.a
clean-profiled:
	rm -f *.gcda *.gcno

-include ../Makefile.lib

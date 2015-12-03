INCLUDES=-I..
OBJS=kokrusher.o
MYFLAGS=-gencode arch=compute_20,code=sm_20 -lineinfo

all: kokrusher.a

ifdef CUDA
cudaboard.o: cuboard.cu
	nvcc -I.. -dc cuboard.cu $(MYFLAGS)
cudamst.o: cudamst.cu
	nvcc -I.. -dc cudamst.cu $(MYFLAGS)
kokrusher.a: cudamst.o cudaboard.o kokrusher.o
	#gcc -I.. -D CUDA -std=c99 -o kokrusher.o -c kokrusher.c
	nvcc -lib cudamst.o cuboard.o kokrusher.o -o kokrusher.a $(MYFLAGS)
else
kokrusher.a: $(OBJS)
endif

clean:
	rm -f *.o *.a
clean-profiled:
	rm -f *.gcda *.gcno

-include ../Makefile.lib

INCLUDES=-I..
OBJS=proof.o

all: proof.a
proof.a: $(OBJS)

clean:
	rm -f *.o *.a
clean-profiled:
	rm -f *.gcda *.gcno

-include ../Makefile.lib

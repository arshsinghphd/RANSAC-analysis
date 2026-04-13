# compiler
CC = gcc

# flags
CFLAGS = -Wall -std=c11
LFLAGS = -lm

# paths
SRC = ../src

# headers — recompile if any change
HEADERS = experiments.h \
          $(SRC)/model.h \
          $(SRC)/generator.h \
          $(SRC)/ransac.h

# shared source files
SHARED = experiments.c \
         $(SRC)/model.c \
         $(SRC)/generator.c \
         $(SRC)/ransac.c

all: exp1 exp2 exp3

# compile and run experiment 1 — vary epsilon
# $ make exp1
exp1: experiment1.c $(SHARED) $(HEADERS)
	$(CC) $(CFLAGS) -o exp1.out experiment1.c $(SHARED) $(LFLAGS)
	./exp1.out

# compile and run experiment 2 — vary structural bias pr
# $ make exp2
exp2: experiment2.c $(SHARED) $(HEADERS)
	$(CC) $(CFLAGS) -o exp2.out experiment2.c $(SHARED) $(LFLAGS)
	./exp2.out

# compile and run experiment 3 — vary model degree
# $ make exp3
exp3: experiment3.c $(SHARED) $(HEADERS)
	$(CC) $(CFLAGS) -o exp3.out experiment3.c $(SHARED) $(LFLAGS)
	./exp3.out

# compile and run experiments.c sanity check
# $ make check
check: experiments.c $(HEADERS) $(SRC)/model.c \
       $(SRC)/generator.c $(SRC)/ransac.c
	$(CC) $(CFLAGS) -o check.out experiments.c \
	    $(SRC)/model.c $(SRC)/generator.c $(SRC)/ransac.c $(LFLAGS)
	./check.out

# run all experiments in sequence
# $ make run
run: exp1 exp2 exp3

# remove all compiled output
# $ make clean
clean:
	rm -f *.out
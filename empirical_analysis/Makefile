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

all: exp01 exp02 exp03 exp1 exp2 exp3

# compile and run experiments.c sanity check
# $ make check
check: experiments.c $(HEADERS) $(SRC)/model.c \
       $(SRC)/generator.c $(SRC)/ransac.c
	$(CC) $(CFLAGS) -o check.out experiments.c \
	    $(SRC)/model.c $(SRC)/generator.c $(SRC)/ransac.c $(LFLAGS)
	./check.out

# compile and run experiment 01 — vary k, measure wall-clock time
# $ make exp01
exp01: experiment0_1.c $(SHARED) $(HEADERS)
	$(CC) $(CFLAGS) -o exp01.out experiment0_1.c $(SHARED) $(LFLAGS)
	./exp01.out

# compile and run experiment 02 — vary epsilon, measure wall-clock time
# $ make exp02
exp02: experiment0_2.c $(SHARED) $(HEADERS)
	$(CC) $(CFLAGS) -o exp02.out experiment0_2.c $(SHARED) $(LFLAGS)
	./exp02.out

# compile and run experiment 03 — vary threshold, measure wall-clock time
# $ make exp03
exp03: experiment0_3.c $(SHARED) $(HEADERS)
	$(CC) $(CFLAGS) -o exp03.out experiment0_3.c $(SHARED) $(LFLAGS)
	./exp03.out

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

# compile and run experiment 3 — effect of budget k
# $ make exp3
exp3: experiment3.c $(SHARED) $(HEADERS)
	$(CC) $(CFLAGS) -o exp3.out experiment3.c $(SHARED) $(LFLAGS)
	./exp3.out

# run all experiments in sequence
# $ make run
run: exp01 exp02 exp03 exp1 exp2 exp3

# remove all compiled output
# $ make clean
clean:
	rm -f *.out
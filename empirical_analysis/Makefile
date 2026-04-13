# compiler
CC = gcc

# flags
CFLAGS = -Wall -std=c11
LFLAGS = -lm

# paths
SRC     = ../src/C
HEADERS = $(SRC)/model.h $(SRC)/generator.h $(SRC)/ransac.h

# source files
EXP_FILES = experiments.c $(SRC)/model.c $(SRC)/generator.c $(SRC)/ransac.c

all: experiments

# compile experiment runner
# $ make experiments
experiments: $(EXP_FILES) $(HEADERS)
	$(CC) $(CFLAGS) -o experiments.out $(EXP_FILES) $(LFLAGS)

# compile and run — writes CSVs to results/
# $ make run
run: experiments
	./experiments.out

# remove compiled output
# $ make clean
clean:
	rm -f experiments.out
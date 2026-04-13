# compiler
CC = gcc

# flags
CFLAGS = -Wall -std=c11
LFLAGS = -lm

# paths
SRC     = ../src/
HEADERS = experiments.h $(SRC)/model.h $(SRC)/generator.h $(SRC)/ransac.h 

# source files
EXP1_FILES = experiment1.c $(SRC)/model.c $(SRC)/generator.c $(SRC)/ransac.c
EXP2_FILES = experiment1.c $(SRC)/model.c $(SRC)/generator.c $(SRC)/ransac.c
EXP3_FILES = experiment1.c $(SRC)/model.c $(SRC)/generator.c $(SRC)/ransac.c

all: experiment1 

# compile experiment runner
# $ make experiment1
experiment1: $(EXP1_FILES) $(HEADERS)
	$(CC) $(CFLAGS) -o experiments.out $(EXP_FILES) $(LFLAGS)

# compile and run — writes CSVs to results/
# $ make run
run: experiment1
	./experiments.out

# remove compiled output
# $ make clean
clean:
	rm -f *.out 
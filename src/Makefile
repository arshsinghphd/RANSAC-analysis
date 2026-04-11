# compiler
CC = gcc

# standard flags
CFLAGS = -Wall -std=c11 -lm

# debug flags — enables DEBUG blocks in code e.g. #ifdef DEBUG
DFLAGS = -Wall -std=c11 -lm -DDEBUG

# plot flags — enables plot output blocks in code e.g. #ifdef PLOT
PFLAGS = -Wall -std=c11 -lm -DPLOT

# header files — listed so make knows to recompile if they change
HEADERS = model.h generator.h ransac.h

# source files shared across targets
MODEL_FILES     = model.c
GENERATOR_FILES = model.c generator.c
RANSAC_FILES    = model.c generator.c ransac.c

all: testmodel testgenerator testransac

# test model functions
# $ make testmodel
testmodel: $(MODEL_FILES) test_model.c $(HEADERS)
	$(CC) $(CFLAGS) -o test_model.out test_model.c $(MODEL_FILES)
	./test_model.out

# test generator functions
# $ make testgenerator
testgenerator: $(GENERATOR_FILES) test_generator.c $(HEADERS)
	$(CC) $(CFLAGS) -o test_generator.out test_generator.c $(GENERATOR_FILES)
	./test_generator.out

# test ransac functions
# $ make testransac
testransac: $(RANSAC_FILES) test_ransac.c $(HEADERS)
	$(CC) $(CFLAGS) -o test_ransac.out test_ransac.c $(RANSAC_FILES)
	./test_ransac.out

# test model with debug output
# $ make debugmodel
debugmodel: $(MODEL_FILES) test_model.c $(HEADERS)
	$(CC) $(DFLAGS) -o test_model.out test_model.c $(MODEL_FILES)
	./test_model.out

# test generator with debug output
# $ make debuggenerator
debuggenerator: $(GENERATOR_FILES) test_generator.c $(HEADERS)
	$(CC) $(DFLAGS) -o test_generator.out test_generator.c $(GENERATOR_FILES)
	./test_generator.out

# test ransac with debug output
# $ make debugransac
debugransac: $(RANSAC_FILES) test_ransac.c $(HEADERS)
	$(CC) $(DFLAGS) -o test_ransac.out test_ransac.c $(RANSAC_FILES)
	./test_ransac.out

# test ransac with plot output — prints csv data for graphing
# $ make plotransac
plotransac: $(RANSAC_FILES) test_ransac.c $(HEADERS)
	$(CC) $(PFLAGS) -o test_ransac.out test_ransac.c $(RANSAC_FILES)
	./test_ransac.out

# remove all compiled output files
# $ make clean
clean:
	rm -f *.out
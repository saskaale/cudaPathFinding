DEL_FILES = rm -rf
MKDIR = mkdir -p
CC = g++

# Compiler flags:
#  -std=c99  C99 standard
#  -g        adds debugging information to the executable file
#  -Wall     turns on most, but not all, compiler warnings
#  -Wextra   turns extra compiler warnings
#  -pedantic turns even more compiler warnings ;)
CFLAGS_COMMON = -Wall -Werror -pedantic -std=c++0x
CFLAGS_DEBUG = -g $(CFLAGS_COMMON)
#CFLAGS_RELEASE = -msse -msse2 -msse3 -mfpmath=sse -O3 $(CFLAGS_COMMON)
#CFLAGS_RELEASE_VERBOSE = -ftree-vectorizer-verbose=5 -fopt-info-vec-missed $(CFLAGS_RELEASE)

CFLAGS = $(CFLAGS_DEBUG)

# Libraries:
#  -lgomp  OpenMP
#  -lm     Math
#LIBS = -lgomp -lm
#LIBS = -lboost_mpi -lboost_serialization
LIBS = 

BIN_DIR = bin
OBJECTSDIR = build
SRC_DIR = src



all: dirs simple_fw

remake: clean all

matrix_tools.o: $(SRC_DIR)/matrix_tools.cpp  $(SRC_DIR)/matrix_tools.h
		$(CC) -c -o $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/matrix_tools.cpp $(CFLAGS)

simple_fw: matrix_tools.o $(SRC_DIR)/simple_fw.cpp
		$(CC) -o $(BIN_DIR)/simple_fw $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/simple_fw.cpp $(CFLAGS) $(LIBS)

clean:
	$(DEL_FILES) $(BIN_DIR) $(OBJECTSDIR)

dirs:
	@$(MKDIR) -p $(BIN_DIR) $(OBJECTSDIR)


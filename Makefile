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
LIBS = -lboost_mpi -lboost_serialization
BIN_DIR = bin


all: dirs simple_fw

remake: clean all

simple_fw: simple_fw.cpp
		$(CC) -o $(BIN_DIR)/simple_fw ./simple_fw.cpp $(CFLAGS) $(LIBS)

clean:
	$(DEL_FILES) bin

dirs:
	@$(MKDIR) -p $(BIN_DIR)


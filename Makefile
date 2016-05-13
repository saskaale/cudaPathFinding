DEL_FILES = rm -rf
MKDIR = mkdir -p
CC = g++
NVCC = nvcc

# Compiler flags:
#  -std=c99  C99 standard
#  -g        adds debugging information to the executable file
#  -Wall     turns on most, but not all, compiler warnings
#  -Wextra   turns extra compiler warnings
#  -pedantic turns even more compiler warnings ;)
CFLAGS_COMMON = -Wall -pedantic -std=c++0x
CFLAGS_DEBUG = -g $(CFLAGS_COMMON)
CFLAGS_NVCC = 
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



all: dirs simple_fw simple_dijkstra simple_dijkstra_cuda blocked_fw blocked_fw_cuda random_mtx_generator

remake: clean all

matrix_tools.o: $(SRC_DIR)/matrix_tools.cpp  $(SRC_DIR)/matrix_tools.h
		$(CC) -c -o $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/matrix_tools.cpp $(CFLAGS)
		
random_mtx_generator: matrix_tools.o $(SRC_DIR)/random_mtx_generator.cpp
		$(CC) -o $(BIN_DIR)/random_mtx_generator $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/random_mtx_generator.cpp $(CFLAGS)

simple_fw: matrix_tools.o $(SRC_DIR)/simple_fw.cpp
		$(CC) -o $(BIN_DIR)/simple_fw $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/simple_fw.cpp $(CFLAGS) $(LIBS)

blocked_fw: matrix_tools.o $(SRC_DIR)/blocked_fw.cpp
		$(CC) -o $(BIN_DIR)/blocked_fw $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/blocked_fw.cpp $(CFLAGS) $(LIBS)

blocked_fw_cuda: matrix_tools.o $(SRC_DIR)/blocked_fw.cu
		$(NVCC) -o $(BIN_DIR)/blocked_fw_cuda $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/blocked_fw.cu $(CFLAGS_NVCC) $(LIBS)

simple_dijkstra: matrix_tools.o $(SRC_DIR)/priority_queue.hpp $(SRC_DIR)/simple_dijkstra.cpp
		$(CC) -o $(BIN_DIR)/simple_dijkstra $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/simple_dijkstra.cpp $(CFLAGS) $(LIBS)

simple_dijkstra_cuda: matrix_tools.o $(SRC_DIR)/priority_queue.hpp $(SRC_DIR)/simple_dijkstra.cu
		$(NVCC) -o $(BIN_DIR)/simple_dijkstra_cuda $(OBJECTSDIR)/matrix_tools.o ./$(SRC_DIR)/simple_dijkstra.cu $(CFLAGS_NVCC) $(LIBS)

clean:
	$(DEL_FILES) $(BIN_DIR) $(OBJECTSDIR)

dirs:
	@$(MKDIR) -p $(BIN_DIR) $(OBJECTSDIR)


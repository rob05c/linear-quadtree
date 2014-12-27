CC=gcc
FLAGS=-g -std=c99 -Wall -Wpedantic -Werror -Wfatal-errors -g
CUDA_CC=nvcc
CUDA_FLAGS=-g -I /usr/local/cuda/include -I ../../cuda/cub
LINK_CC=g++
LINK_FLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -ltbb
CC_CPP=g++
CPP_FLAGS= -std=c++11 -Wall -Wpedantic -Werror -Wfatal-errors -g

all: lqt
lqt: lqt.o main.o lqtcuda.o nocuda.o
	$(LINK_CC) main.o lqt.o lqtcuda.o nocuda.o -o lqt -lm $(LINK_FLAGS)
main.o: 
	$(CC_CPP) $(CPP_FLAGS) -c main.cpp -o main.o
lqt.o:
	$(CC) $(FLAGS) -c lqt.c -o lqt.o
lqtcuda.o:
	$(CUDA_CC) $(CUDA_FLAGS) -c lqt.cu -o lqtcuda.o
nocuda.o:
	$(CC_CPP) $(CPP_FLAGS) -c nocuda.cpp -o nocuda.o
clean:
	rm -f *o lqt

CC=clang
FLAGS=-g -std=c99 -Wall -Wpedantic -Werror -Wfatal-errors -g
CUDA_CC=nvcc
CUDA_FLAGS=-g -I /opt/cuda/include -I ../../cuda/cub
LINK_CC=clang++
LINK_FLAGS=-L/opt/cuda/lib64 -lcuda -lcudart

all: lqt
lqt: lqt.o main.o lqtcuda.o
	$(LINK_CC) $(LINK_FLAGS) main.o lqt.o lqtcuda.o -o lqt -lm
main.o: 
	$(CC) $(FLAGS) -c main.c -o main.o
lqt.o:
	$(CC) $(FLAGS) -c lqt.c -o lqt.o
lqtcuda.o:
	$(CUDA_CC) $(CUDA_FLAGS) -c lqt.cu -o lqtcuda.o
clean:
	rm -f *o lqt

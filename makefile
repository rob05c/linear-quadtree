CC=c99
FLAGS=-Wall -Wpedantic -Werror -Wfatal-errors -g
CUDA_CC=nvcc
CUDA_FLAGS= -I /opt/cuda/include -Xptxas -Werror
LINK_CC=g++
LINK_FLAGS= -L/opt/cuda/lib64 -lcuda -lcudart

all: lqt.o main.o test.o
	$(LINK_CC) $(LINK_FLAGS) main.o lqt.o test.o -o lqt -lm
main.o: 
	$(CC) $(FLAGS) -c main.c -o main.o
lqt.o:
	$(CC) $(FLAGS) -c lqt.c -o lqt.o
test.o:
	$(CUDA_CC) $(CUDA_FLAGS) -c test.cu -o test.o
clean:
	rm -f *o lqt

CC=c99
all: lqt.o main.o
	$(CC) -g main.o lqt.o -o lqt -lm
main.o: 
	$(CC) -c main.c -o main.o
lqt.o:
	$(CC) -c lqt.c -o lqt.o
clean:
	rm -f *o lqt

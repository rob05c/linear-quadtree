CC=g++
all: lqt.o main.o
	$(CC) -g main.o lqt.o -o lqt
main.o: 
	g++ -c main.cpp -o main.o
lqt.o:
	g++ -c lqt.cpp -o lqt.o
clean:
	rm -f *o lqt

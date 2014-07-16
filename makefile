CC=c99
FLAGS=-Wall -Wpedantic -Werror -Wfatal-errors -g
all: lqt.o main.o
	$(CC) $(FLAGS) main.o lqt.o -o lqt -lm
main.o: 
	$(CC) $(FLAGS) -c main.c -o main.o
lqt.o:
	$(CC) $(FLAGS) -c lqt.c -o lqt.o
clean:
	rm -f *o lqt

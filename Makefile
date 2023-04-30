
CC=gcc
FLAGS=-Wextra -Wpedantic
TARGET=main

all: ann.c
	$(CC) $(FLAGS) -o ann main.c -lm

run: all
	./ann 
clean:
	echo "cleaning up:"
	rm ./ann 

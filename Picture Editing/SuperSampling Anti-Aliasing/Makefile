all: ssaa

ssaa: ssaa.c ssaa.h main.c
	gcc -o ssaa ssaa.c main.c -lpthread -Wall -lm

.PHONY: clean
clean:
	rm -f ssaa

all: mrenderer

mrenderer: mrenderer.c mrenderer.h main.c
	gcc -o mrenderer mrenderer.c main.c -lpthread -Wall -lm

.PHONY: clean
clean:
	rm -f mrenderer

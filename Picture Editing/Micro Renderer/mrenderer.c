#include "mrenderer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

line l = {-1, 2, 0};
space s = {100, 100};
float limit = 3;

int num_threads;
int resolution;

image *imgspace;

P5Pixel *allocateP5Pixmap(unsigned int width, unsigned int height) {
    P5Pixel *map = (P5Pixel *) malloc (sizeof (P5Pixel) * height * width);
    if (map == NULL) {
        perror("Unable to allocate memory.\n");
        exit(-1);
    }
    return map;
}

void initialize(image *im) {
    im -> width = resolution;
    im -> height = resolution;
    im -> maxval = 255;
    im -> pixmap = allocateP5Pixmap(im -> width, im -> height);
}

float distanceToLine (unsigned int xpix, unsigned int ypix) {
    float xreal = (xpix + 0.5) * (s.width / imgspace -> width);
    float yreal = (imgspace -> height - ypix - 0.5) * (s.height / imgspace -> width);

    return abs (l.a * xreal + l.b * yreal + l.c) / sqrt (l.a * l.a + l.b * l.b);
}

void render(image *im) {
    if (memset (im -> pixmap, 255,im -> width * im -> height) == NULL) {
        perror("Unable to fill the memory.\n");
        exit(-1);
    }

    imgspace = im;

    pthread_t threads[num_threads];
	unsigned int thread_id[num_threads];

	for(int i = 0; i < num_threads; ++i)
		thread_id[i] = i;

	for(int i = 0; i < num_threads; ++i) {
		pthread_create(&(threads[i]), NULL, parallelRender, &(thread_id[i]));
	}

	for(int i = 0; i < num_threads; ++i) {
		pthread_join(threads[i], NULL);
	}
}

void *parallelRender (void *argument) {
    unsigned int thread_id = *(unsigned int *) argument;

    unsigned int start = thread_id * ceil( (double) imgspace -> height/ num_threads);
    unsigned int end = (imgspace -> height <= (thread_id + 1) * ceil( (double) imgspace -> height/ num_threads)) 
                     ? imgspace -> height : (thread_id + 1) * ceil( (double) imgspace -> height/ num_threads);


    for (int i = start; i < end; ++i) {
        for (int j = 0; j < imgspace -> width; ++j) {
            if (distanceToLine (j, i) <= limit)
                imgspace -> pixmap[i * imgspace -> width + j] = 0;
        }
    }
    return NULL;
}

void writeData(const char * filename, image *img) {
    FILE *filepointer;

    filepointer = fopen (filename, "wb");

    if (filepointer == NULL) {
        perror("Unable to open the output file.\n");
        exit(-1);
    }

    fprintf(filepointer, "P5\n%u %u\n%hu\n", img -> width, img -> height, img -> maxval);    
    fwrite (img -> pixmap, sizeof (P5Pixel), img -> width * img -> height, filepointer);

    fclose (filepointer);
}


#include "ssaa.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>

int num_threads;
int resize_factor;

unsigned int square;
image *tmpin;
image *tmpout;

P5Pixel *allocateP5Pixmap(unsigned int width, unsigned int height) {
    P5Pixel *map = (P5Pixel *) malloc (sizeof (P5Pixel) * height * width);
    if (map == NULL) {
        perror("Unable to allocate memory.\n");
        exit(-1);
    }
    return map;
}

P6Pixel *allocateP6Pixmap(unsigned int width, unsigned int height) {
    P6Pixel *map = (P6Pixel *) malloc (sizeof (P6Pixel) * height * width);
    if (map == NULL) {
        perror("Unable to allocate memory.\n");
        exit(-1);
    }
    return map;
}

void readInput(const char * filename, image *img) {
    FILE *filepointer;
    char buffer[3];

    filepointer = fopen (filename, "rb");

    if (filepointer == NULL) {
        perror("Unable to open the input file.\n");
        exit(-1);
    }

    if (fgets (buffer, 3, filepointer) == NULL) {
        perror("Unable to read the input format.\n");
        exit(-1);
    }

    if (buffer[0] == 'P') {
        switch (buffer[1]) {
            case '5':
                img -> type = P5;
                break;
            case '6':
                img -> type = P6;
                break;
            default:
                perror("Wrong type of the input file.\n");
                exit(-1);
        }
    } else {
        perror("Wrong type of the input file.\n");
        exit(-1);
    }

    fscanf (filepointer, "%u", &img -> width);
    fscanf (filepointer, "%u", &img -> height);
    fscanf (filepointer, "%hu", &img -> maxval);
    fseek(filepointer, 1, SEEK_CUR);
    
    switch (img -> type) {
        case P5:
            img -> p6pixmap = NULL;
            img -> p5pixmap = allocateP5Pixmap (img -> width, img -> height);
            fread (img -> p5pixmap, sizeof (P5Pixel), img -> width * img -> height, filepointer);
            break;

        case P6:
            img -> p5pixmap = NULL;
            img -> p6pixmap = allocateP6Pixmap (img -> width, img -> height);
            fread (img -> p6pixmap, sizeof (P6Pixel), img -> width * img -> height, filepointer);
            break;

        default: 
            return;
    }
    fclose (filepointer);
}

void writeData(const char * filename, image *img) {
    FILE *filepointer;

    filepointer = fopen (filename, "wb");

    if (filepointer == NULL) {
        perror("Unable to open the output file.\n");
        exit(-1);
    }

    switch (img -> type) {
        case P5:
            fprintf(filepointer, "P5\n%u %u\n%hu\n", img -> width, img -> height, img -> maxval);    
            fwrite (img -> p5pixmap, sizeof (P5Pixel), img -> width * img -> height, filepointer);
            break;
        case P6:
            fprintf(filepointer, "P6\n%u %u\n%hu\n", img -> width, img -> height, img -> maxval);
            fwrite (img -> p6pixmap, sizeof (P6Pixel), img -> width * img -> height, filepointer);
            break;
        default:
            perror("Wrong type of the output file.\n");
            exit(-1);
    }

    fclose (filepointer);
}

void resize(image *in, image * out) {
    square = resize_factor * resize_factor;
    tmpin = in;
    tmpout = out;

    out -> type = in -> type;
    out -> width = in -> width / resize_factor;
    out -> height = in -> height / resize_factor;
    out -> maxval = in -> maxval;

    switch (in -> type) {
        case P5:
            out -> p6pixmap = NULL;
            out -> p5pixmap = allocateP5Pixmap(out -> width, out -> height);
            break;
        case P6:
            out -> p5pixmap = NULL;
            out -> p6pixmap = allocateP6Pixmap(out -> width, out -> height);
            break;
        default:
            return;
    }

    pthread_t threads[num_threads];
	unsigned int thread_id[num_threads];

	for(int i = 0; i < num_threads; ++i)
		thread_id[i] = i;

	for(int i = 0; i < num_threads; ++i) {
		pthread_create(&(threads[i]), NULL, computePixels, &(thread_id[i]));
	}

	for(int i = 0; i < num_threads; ++i) {
		pthread_join(threads[i], NULL);
	}
}

void *computePixels (void *argument) {
    unsigned int thread_id = *(unsigned int *) argument;

    unsigned int start = thread_id * ceil( (double) tmpout -> height/ num_threads);
    unsigned int end = (tmpout -> height <= (thread_id + 1) * ceil( (double) tmpout -> height/ num_threads)) 
                     ? tmpout -> height : (thread_id + 1) * ceil( (double) tmpout -> height/ num_threads);

    switch (tmpout -> type) {
        case P5:
            if (resize_factor % 2 == 0) {
                unsigned int pixelsum;
                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < tmpout -> width; ++j) {
                        pixelsum = 0;
                        for (int k = i * resize_factor; k < (i + 1) * resize_factor; ++k) {
                            for (int l = j * resize_factor; l < (j + 1) * resize_factor; ++l) {
                                pixelsum += tmpin -> p5pixmap[k * tmpin -> width + l];
                            }
                        }
                        tmpout -> p5pixmap[i * tmpout -> width + j] = pixelsum / square;
                    }
                }
                return NULL;
            }

            if (resize_factor == 3) {
                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < tmpout -> width; ++j) {
                        tmpout -> p5pixmap[i * tmpout -> width + j] = 
                                     ( 1 * tmpin -> p5pixmap[i * 3 * tmpin -> width + j * 3]
                                     + 2 * tmpin -> p5pixmap[i * 3 * tmpin -> width + (j * 3 + 1)]
                                     + 1 * tmpin -> p5pixmap[i * 3 * tmpin -> width + (j * 3 + 2)]
                                     + 2 * tmpin -> p5pixmap[(i * 3 + 1) * tmpin -> width + j * 3]
                                     + 4 * tmpin -> p5pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 1)]
                                     + 2 * tmpin -> p5pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 2)]
                                     + 1 * tmpin -> p5pixmap[(i * 3 + 2) * tmpin -> width + j * 3]
                                     + 2 * tmpin -> p5pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 1)]
                                     + 1 * tmpin -> p5pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 2)]) / 16;
                    }
                }
                return NULL;
            }
            break;
        case P6:
            if (resize_factor % 2 == 0) {
                unsigned int redsum, greensum, bluesum;
                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < tmpout -> width; ++j) {
                        redsum = 0;
                        greensum = 0;
                        bluesum = 0;
                        for (int k = i * resize_factor; k < (i + 1) * resize_factor; ++k) {
                            for (int l = j * resize_factor; l < (j + 1) * resize_factor; ++l) {
                                redsum += tmpin -> p6pixmap[k * tmpin -> width + l].red;
                                greensum += tmpin -> p6pixmap[k * tmpin -> width + l].green;
                                bluesum += tmpin -> p6pixmap[k * tmpin -> width + l].blue;
                            }
                        }
                        tmpout -> p6pixmap[i * tmpout -> width + j].red = redsum / square;
                        tmpout -> p6pixmap[i * tmpout -> width + j].green = greensum / square;
                        tmpout -> p6pixmap[i * tmpout -> width + j].blue = bluesum / square;
                    }
                }
                return NULL;
            }

            if (resize_factor == 3) {
                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < tmpout -> width; ++j) {
                        tmpout -> p6pixmap[i * tmpout -> width + j].red = 
                                     ( 1 * tmpin -> p6pixmap[i * 3 * tmpin -> width + j * 3].red
                                     + 2 * tmpin -> p6pixmap[i * 3 * tmpin -> width + (j * 3 + 1)].red
                                     + 1 * tmpin -> p6pixmap[i * 3 * tmpin -> width + (j * 3 + 2)].red
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + j * 3].red
                                     + 4 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 1)].red
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 2)].red
                                     + 1 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + j * 3].red
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 1)].red
                                     + 1 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 2)].red) / 16;
                        tmpout -> p6pixmap[i * tmpout -> width + j].green = 
                                     ( 1 * tmpin -> p6pixmap[i * 3 * tmpin -> width + j * 3].green
                                     + 2 * tmpin -> p6pixmap[i * 3 * tmpin -> width + (j * 3 + 1)].green
                                     + 1 * tmpin -> p6pixmap[i * 3 * tmpin -> width + (j * 3 + 2)].green
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + j * 3].green
                                     + 4 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 1)].green
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 2)].green
                                     + 1 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + j * 3].green
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 1)].green
                                     + 1 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 2)].green) / 16;
                        tmpout -> p6pixmap[i * tmpout -> width + j].blue = 
                                     ( 1 * tmpin -> p6pixmap[i * 3 * tmpin -> width + j * 3].blue
                                     + 2 * tmpin -> p6pixmap[i * 3 * tmpin -> width + (j * 3 + 1)].blue
                                     + 1 * tmpin -> p6pixmap[i * 3 * tmpin -> width + (j * 3 + 2)].blue
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + j * 3].blue
                                     + 4 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 1)].blue
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 1) * tmpin -> width + (j * 3 + 2)].blue
                                     + 1 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + j * 3].blue
                                     + 2 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 1)].blue
                                     + 1 * tmpin -> p6pixmap[(i * 3 + 2) * tmpin -> width + (j * 3 + 2)].blue) / 16;
                    }
                }
                return NULL;
            }
            break;
        default:
            return NULL;
    }
    return NULL;
}

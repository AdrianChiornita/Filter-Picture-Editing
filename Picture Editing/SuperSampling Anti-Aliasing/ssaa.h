#ifndef SSAA_H
#define SSAA_H

typedef enum {P5, P6} pnmtype;

typedef struct {
    unsigned char red;
    unsigned char green; 
    unsigned char blue;
} P6Pixel;

typedef unsigned char P5Pixel;

typedef struct {
    pnmtype type;
    unsigned int width;
    unsigned int height;
    unsigned short maxval;
    P5Pixel *p5pixmap;
    P6Pixel *p6pixmap;
} image;

void *computePixels (void *argument);

P5Pixel * allocateP5Pixmap(unsigned int width, unsigned int height);
P6Pixel * allocateP6Pixmap(unsigned int width, unsigned int height);

void readInput(const char * filename, image *img);
void writeData(const char * filename, image *img);

void resize(image *in, image * out);

#endif

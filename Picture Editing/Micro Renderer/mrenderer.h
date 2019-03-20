#ifndef MRENDERER_H
#define MRENDERER_H

typedef unsigned char P5Pixel;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned short maxval;
    P5Pixel *pixmap;
}image;

typedef struct {
    float a, b, c;
}line;

typedef struct {
    float width, height;
}space;

P5Pixel * allocateP5Pixmap(unsigned int width, unsigned int height);

void initialize(image *im);
float distanceToLine (unsigned int x, unsigned int y);

void render(image *im);
void *parallelRender (void *argument);

void writeData(const char * filename, image *img);

#endif

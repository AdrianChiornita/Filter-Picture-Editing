#include<mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define KERNELSIZE 3
#define RGB_PIXEL_SIZE (3 * sizeof (unsigned char))
#define GRAYSCALE_PIXEL_SIZE sizeof (unsigned char)
#define OFFSET_BORDER(RANK, P) (((P) != 1) ? (((RANK) == 0 || (RANK) == (P) - 1) ? 1 : 2) : 0)

typedef enum {P5, P6} pnmtype;
typedef enum {RED, GREEN, BLUE} rgb_color;

typedef unsigned char Pixel;

typedef struct {
    pnmtype type;
    unsigned int width;
    unsigned int height;
    unsigned short maxval;
    Pixel *pixmap;
} image;

typedef struct {
    pnmtype type;
    unsigned int start;
    unsigned int end;
    unsigned int max_height;
    unsigned int max_width;
    Pixel *pixmap;
    Pixel *new_pixmap;
} thread_part;

typedef float kernel_filter[KERNELSIZE * KERNELSIZE];
typedef enum {identity, smooth, blur, sharpen, mean, emboss} filter_type;

kernel_filter IDN = {
                        0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 0.0f,
                    };

kernel_filter SMT = {
                        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                    };

kernel_filter BLR = {
                        1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
                        2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
                        1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f
                    };

kernel_filter SHR = {
                              0.0f, -2.0f/3.0f,       0.0f,
                        -2.0f/3.0f, 11.0f/3.0f, -2.0f/3.0f,
                              0.0f, -2.0f/3.0f,       0.0f
                    };

kernel_filter MEN = {
                        -1.0f, -1.0f, -1.0f,
                        -1.0f,  9.0f, -1.0f,
                        -1.0f, -1.0f, -1.0f
                    };

kernel_filter EBS = {
                        0.0f,  1.0f, 0.0f,
                        0.0f,  0.0f, 0.0f,
                        0.0f, -1.0f, 0.0f
                    };

int rank, num_procs, P, num_filters;
thread_part fragment;
image img;

void send_pixmap_parts(image *img)
{
    for (int rank_index = 1; rank_index < P; ++rank_index)
    {
        unsigned int rank_start = (fragment.max_height <= rank_index * ceil( (double) fragment.max_height/P))
                ? fragment.max_height : rank_index * ceil( (double) fragment.max_height/P);
        unsigned int rank_end = (fragment.max_height <= (rank_index + 1) * ceil( (double) fragment.max_height/P)) 
                ? fragment.max_height : (rank_index + 1) * ceil( (double) fragment.max_height/P);
 
        if (fragment.type == P5)
            MPI_Send(
                img->pixmap +  GRAYSCALE_PIXEL_SIZE * (rank_start - 1) * fragment.max_width,
                GRAYSCALE_PIXEL_SIZE * fragment.max_width * (rank_end - rank_start + OFFSET_BORDER(rank_index, P)),
                MPI_UNSIGNED_CHAR,
                rank_index,
                0,
                MPI_COMM_WORLD
            );
        if (fragment.type == P6)
            MPI_Send(
                img->pixmap + RGB_PIXEL_SIZE * (rank_start - 1) * fragment.max_width,
                RGB_PIXEL_SIZE * fragment.max_width * (rank_end - rank_start + OFFSET_BORDER(rank_index, P)),
                MPI_UNSIGNED_CHAR,
                rank_index,
                0,
                MPI_COMM_WORLD
            );
    }
}

void recv_pixmap_parts(image *img)
{
    for (int rank_index = 1; rank_index < P; ++rank_index)
    {
        unsigned int rank_start = (fragment.max_height <= rank_index * ceil( (double) fragment.max_height/P))
                ? fragment.max_height : rank_index * ceil( (double) fragment.max_height/P);
        unsigned int rank_end = (fragment.max_height <= (rank_index + 1) * ceil( (double) fragment.max_height/P)) 
                ? fragment.max_height : (rank_index + 1) * ceil( (double) fragment.max_height/P);
        
        if (fragment.type == P5)
            MPI_Recv(
                img->pixmap +  GRAYSCALE_PIXEL_SIZE * rank_start * fragment.max_width,
                GRAYSCALE_PIXEL_SIZE * fragment.max_width * (rank_end - rank_start),
                MPI_UNSIGNED_CHAR,
                rank_index,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        if (fragment.type == P6)
            MPI_Recv(
                img->pixmap + RGB_PIXEL_SIZE * rank_start * fragment.max_width,
                RGB_PIXEL_SIZE * fragment.max_width * (rank_end - rank_start),
                MPI_UNSIGNED_CHAR,
                rank_index,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
    }
}

Pixel *p5_pixmap_alloc(unsigned int width, unsigned int height) 
{
    Pixel *map = (Pixel *) malloc (GRAYSCALE_PIXEL_SIZE * height * width);
    if (map == NULL) exit(-1);
    return map;
}

Pixel *p6_pixmap_alloc(unsigned int width, unsigned int height) 
{
    Pixel *map = (Pixel *) malloc (RGB_PIXEL_SIZE * height * width);
    if (map == NULL) exit(-1);
    return map;
}

void read_pnm_image(const char * filename, image *img) 
{
    FILE *filepointer;
    char fileformat[3];

    filepointer = fopen (filename, "rb");

    if (filepointer == NULL) exit(-1);

    if (fgets (fileformat, 3, filepointer) == NULL) exit(-1);

    if (fileformat[0] == 'P') 
    {
        switch (fileformat[1]) 
        {
            case '5':
                img -> type = P5;
                break;
            case '6':
                img -> type = P6;
                break;
            default:
                exit(-1);
        }
    } 
    else exit(-1);

    fscanf (filepointer, "%u", &img -> width);
    fscanf (filepointer, "%u", &img -> height);
    fscanf (filepointer, "%hu", &img -> maxval);
    fseek(filepointer, 1, SEEK_CUR);
    
    switch (img -> type) 
    {
        case P5:
            img -> pixmap = p5_pixmap_alloc (img -> width, img -> height);
            fread (img -> pixmap, 1, img -> width * img -> height, filepointer);
            break;

        case P6:
            img -> pixmap = NULL;
            img -> pixmap = p6_pixmap_alloc (img -> width, img -> height);
            fread (img -> pixmap, 3, img -> width * img -> height, filepointer);
            break;

        default: 
            return;
    }
    fclose (filepointer);
}

void write_pnm_image(const char * filename, image *img) 
{
    FILE *filepointer;

    filepointer = fopen (filename, "wb");

    if (filepointer == NULL) exit(-1);

    switch (img -> type) 
    {
        case P5:
            fprintf(filepointer, "P5\n%u %u\n%hu\n", img -> width, img -> height, img -> maxval);    
            fwrite (img -> pixmap, 1, img -> width * img -> height, filepointer);
            break;
        case P6:
            fprintf(filepointer, "P6\n%u %u\n%hu\n", img -> width, img -> height, img -> maxval);
            fwrite (img -> pixmap, 3, img -> width * img -> height, filepointer);
            break;
        default:
            exit(-1);
    }
    fclose (filepointer);
}

void apply_filter_p5_pixel(unsigned int index_height, unsigned int index_width, filter_type filter)
{
    switch (filter)
    {
        case identity:
            fragment.new_pixmap[index_height * fragment.max_width + index_width] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width - 1)] * IDN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width +       index_width] * IDN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width + 1)] * IDN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width - 1)] * IDN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width +       index_width] * IDN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width + 1)] * IDN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width - 1)] * IDN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width +       index_width] * IDN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width + 1)] * IDN[8];
            break;
        case smooth:
            fragment.new_pixmap[index_height * fragment.max_width + index_width] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width - 1)] * SMT[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width +       index_width] * SMT[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width + 1)] * SMT[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width - 1)] * SMT[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width +       index_width] * SMT[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width + 1)] * SMT[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width - 1)] * SMT[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width +       index_width] * SMT[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width + 1)] * SMT[8];
            break;
        case blur:
            fragment.new_pixmap[index_height * fragment.max_width + index_width] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width - 1)] * BLR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width +       index_width] * BLR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width + 1)] * BLR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width - 1)] * BLR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width +       index_width] * BLR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width + 1)] * BLR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width - 1)] * BLR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width +       index_width] * BLR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width + 1)] * BLR[8];
            break;
        case sharpen:
            fragment.new_pixmap[index_height * fragment.max_width + index_width] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width - 1)] * SHR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width +       index_width] * SHR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width + 1)] * SHR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width - 1)] * SHR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width +       index_width] * SHR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width + 1)] * SHR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width - 1)] * SHR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width +       index_width] * SHR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width + 1)] * SHR[8];
            break;
        case mean:
            fragment.new_pixmap[index_height * fragment.max_width + index_width] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width - 1)] * MEN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width +       index_width] * MEN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width + 1)] * MEN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width - 1)] * MEN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width +       index_width] * MEN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width + 1)] * MEN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width - 1)] * MEN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width +       index_width] * MEN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width + 1)] * MEN[8];
            break;
        case emboss:
            fragment.new_pixmap[index_height * fragment.max_width + index_width] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width - 1)] * EBS[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width +       index_width] * EBS[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width + (index_width + 1)] * EBS[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width - 1)] * EBS[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width +       index_width] * EBS[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width + (index_width + 1)] * EBS[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width - 1)] * EBS[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width +       index_width] * EBS[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width + (index_width + 1)] * EBS[8];
            break;
        default:
            break;
    }
}

void compute_filtered_p5_image(const filter_type filter)
{
    if (P != 1)
    {
        if (rank == 0)
            fragment.end++;
        else if (rank == P - 1)
            fragment.start--;
        else
        {
            fragment.start--;
            fragment.end++;
        }
    }

    for (unsigned int i = 0; i < (fragment.end - fragment.start); ++i)
    {
        for (unsigned int j = 0; j < fragment.max_width; ++j)
        {
            if (i == 0 || i == (fragment.end - fragment.start - 1) || j == 0 || j == (fragment.max_width - 1))
                fragment.new_pixmap[i * fragment.max_width + j] = fragment.pixmap[i * fragment.max_width + j];
            else
                apply_filter_p5_pixel(i, j, filter);
        }
    }

    if (P != 1)
    {
        if (rank == 0)
            fragment.end--;
        else if (rank == P - 1)
            fragment.start++;
        else
        {
            fragment.start++;
            fragment.end--;
        }
    }
}

void apply_filter_p6_pixel(unsigned int index_height, unsigned int index_width, filter_type filter)
{
    switch (filter)
    {
        case identity:
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + RED] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * IDN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * IDN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * IDN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * IDN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * IDN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * IDN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * IDN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * IDN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * IDN[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + GREEN] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * IDN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * IDN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * IDN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * IDN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * IDN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * IDN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * IDN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * IDN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * IDN[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + BLUE] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * IDN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * IDN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * IDN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * IDN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * IDN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * IDN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * IDN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * IDN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * IDN[8];
            break;
        case smooth:
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + RED] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * SMT[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * SMT[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * SMT[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * SMT[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * SMT[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * SMT[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * SMT[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * SMT[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * SMT[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + GREEN] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * SMT[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * SMT[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * SMT[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * SMT[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * SMT[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * SMT[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * SMT[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * SMT[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * SMT[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + BLUE] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * SMT[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * SMT[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * SMT[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * SMT[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * SMT[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * SMT[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * SMT[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * SMT[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * SMT[8];
            break;
        case blur:
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + RED] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * BLR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * BLR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * BLR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * BLR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * BLR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * BLR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * BLR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * BLR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * BLR[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + GREEN] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * BLR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * BLR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * BLR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * BLR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * BLR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * BLR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * BLR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * BLR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * BLR[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + BLUE] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * BLR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * BLR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * BLR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * BLR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * BLR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * BLR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * BLR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * BLR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * BLR[8];
            break;
        case sharpen:
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + RED] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * SHR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * SHR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * SHR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * SHR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * SHR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * SHR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * SHR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * SHR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * SHR[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + GREEN] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * SHR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * SHR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * SHR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * SHR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * SHR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * SHR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * SHR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * SHR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * SHR[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + BLUE] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * SHR[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * SHR[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * SHR[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * SHR[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * SHR[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * SHR[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * SHR[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * SHR[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * SHR[8];
            break;
        case mean:
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + RED] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * MEN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * MEN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * MEN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * MEN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * MEN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * MEN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * MEN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * MEN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * MEN[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + GREEN] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * MEN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * MEN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * MEN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * MEN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * MEN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * MEN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * MEN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * MEN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * MEN[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + BLUE] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * MEN[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * MEN[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * MEN[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * MEN[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * MEN[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * MEN[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * MEN[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * MEN[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * MEN[8];
            break;
        case emboss:
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + RED] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * EBS[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * EBS[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * EBS[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * EBS[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * EBS[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * EBS[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + RED)] * EBS[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + RED] * EBS[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + RED)] * EBS[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + GREEN] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * EBS[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * EBS[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * EBS[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * EBS[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * EBS[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * EBS[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + GREEN)] * EBS[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + GREEN] * EBS[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + GREEN)] * EBS[8];
            fragment.new_pixmap[index_height * fragment.max_width * RGB_PIXEL_SIZE + index_width + BLUE] =
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * EBS[0] + 
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * EBS[1] +
                (float) fragment.pixmap[(index_height - 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * EBS[2] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * EBS[3] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * EBS[4] +
                (float) fragment.pixmap[      index_height * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * EBS[5] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width - RGB_PIXEL_SIZE + BLUE)] * EBS[6] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE +                    index_width + BLUE] * EBS[7] +
                (float) fragment.pixmap[(index_height + 1) * fragment.max_width * RGB_PIXEL_SIZE + (index_width + RGB_PIXEL_SIZE + BLUE)] * EBS[8];
            break;
        default:
            break;
    }
}

void compute_filtered_p6_image(const filter_type filter)
{
    if (P != 1)
    {
        if (rank == 0)
            fragment.end++;
        else if (rank == P - 1)
            fragment.start--;
        else
        {
            fragment.start--;
            fragment.end++;
        }
    }

    for (unsigned int i = 0; i < (fragment.end - fragment.start); ++i)
    {
        for (unsigned int j = 0; j < (RGB_PIXEL_SIZE * fragment.max_width); j += RGB_PIXEL_SIZE)
        {
            if (i == 0 || i == (fragment.end - fragment.start - 1) || j == 0 || j == (RGB_PIXEL_SIZE * (fragment.max_width - 1)))
            {
                fragment.new_pixmap[i * fragment.max_width * RGB_PIXEL_SIZE + (j + RED)] = fragment.pixmap[i * fragment.max_width * RGB_PIXEL_SIZE + (j + RED)];
                fragment.new_pixmap[i * fragment.max_width * RGB_PIXEL_SIZE + (j + GREEN)] = fragment.pixmap[i * fragment.max_width * RGB_PIXEL_SIZE + (j + GREEN)];
                fragment.new_pixmap[i * fragment.max_width * RGB_PIXEL_SIZE + (j + BLUE)] = fragment.pixmap[i * fragment.max_width * RGB_PIXEL_SIZE + (j + BLUE)];
            }
            else
               apply_filter_p6_pixel(i, j, filter);
        }
    }

    if (P != 1)
    {
        if (rank == 0)
            fragment.end--;
        else if (rank == P -1)
            fragment.start++;
        else
        {
            fragment.start++;
            fragment.end--;
        }
    }
}

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    num_filters = argc - 3;
    filter_type filters[num_filters];

    for (int index = 3; index < argc; ++index)
    {
        if      (strcmp(argv[index], "smooth") == 0)        filters[index - 3] = smooth;
        else if (strcmp(argv[index], "blur") == 0)          filters[index - 3] = blur;
        else if (strcmp(argv[index], "sharpen") == 0)       filters[index - 3] = sharpen;
        else if (strcmp(argv[index], "mean") == 0)          filters[index - 3] = mean;
        else if (strcmp(argv[index], "emboss") == 0)        filters[index - 3] = emboss;
        else if(    strcmp(argv[index], "identity") == 0)   filters[index - 3] = identity;
        else
            filters[index - 3] = identity;
    }

    if (rank == 0) 
    {
        read_pnm_image(argv[1], &img);
        fragment.max_height = img.height;
        fragment.max_width = img.width;
        fragment.type = img.type;
    }

    MPI_Bcast (&fragment.max_height, 1, MPI_UNSIGNED, 0,  MPI_COMM_WORLD);
    MPI_Bcast (&fragment.max_width, 1, MPI_UNSIGNED, 0,  MPI_COMM_WORLD);
    MPI_Bcast (&fragment.type, 1, MPI_INT, 0,  MPI_COMM_WORLD);     

    P = (fragment.max_height < num_procs) ? fragment.max_height : num_procs;

    if (rank < P)
    {
        fragment.start = (fragment.max_height <= rank * ceil( (double) fragment.max_height/P))
                ? fragment.max_height : rank * ceil( (double) fragment.max_height/P);
        fragment.end = (fragment.max_height <= (rank + 1) * ceil( (double) fragment.max_height/P)) 
                ? fragment.max_height : (rank + 1) * ceil( (double) fragment.max_height/P);
        
        if (fragment.start != fragment.end)
        {
            if (fragment.type == P5)
            {
                fragment.pixmap     = p5_pixmap_alloc(fragment.max_width, (fragment.end - fragment.start + OFFSET_BORDER(rank, P)));
                fragment.new_pixmap = p5_pixmap_alloc(fragment.max_width, (fragment.end - fragment.start + OFFSET_BORDER(rank, P)));
            }
            if (fragment.type == P6)
            {
                fragment.pixmap     = p6_pixmap_alloc(fragment.max_width, (fragment.end - fragment.start + OFFSET_BORDER(rank, P)));
                fragment.new_pixmap = p6_pixmap_alloc(fragment.max_width, (fragment.end - fragment.start + OFFSET_BORDER(rank, P)));
            }

            if (rank == 0)
            {
                if (fragment.type == P5)
                {
                    for (int filter_index = 0; filter_index < num_filters; ++filter_index)
                    {
                        send_pixmap_parts(&img);

                        memcpy(fragment.pixmap, img.pixmap, GRAYSCALE_PIXEL_SIZE * (fragment.end - fragment.start + OFFSET_BORDER(rank,P)) * fragment.max_width);
                        compute_filtered_p5_image(filters[filter_index]);
                        memcpy(img.pixmap, fragment.new_pixmap, GRAYSCALE_PIXEL_SIZE * (fragment.end - fragment.start) * fragment.max_width);

                        recv_pixmap_parts(&img);
                    }

                    write_pnm_image(argv[2], &img);
                }
                if (fragment.type == P6)
                {
                    for (int filter_index = 0; filter_index < num_filters; ++filter_index)
                    {
                        send_pixmap_parts(&img);

                        memcpy(fragment.pixmap, img.pixmap, RGB_PIXEL_SIZE * (fragment.end - fragment.start + OFFSET_BORDER(rank,P)) * fragment.max_width);
                        compute_filtered_p6_image(filters[filter_index]);
                        memcpy(img.pixmap, fragment.new_pixmap, RGB_PIXEL_SIZE * (fragment.end - fragment.start) * fragment.max_width);
                        
                        recv_pixmap_parts(&img);
                    }
                    write_pnm_image(argv[2], &img);
                }
            }
            else
            {
                if (fragment.type == P5)
                {
                    for (int filter_index = 0; filter_index < num_filters; ++filter_index)
                    {
                        MPI_Recv(
                            fragment.pixmap,
                            GRAYSCALE_PIXEL_SIZE * fragment.max_width * (fragment.end - fragment.start + OFFSET_BORDER(rank, P)),
                            MPI_UNSIGNED_CHAR,
                            0,
                            0,
                            MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE
                        );

                        compute_filtered_p5_image(filters[filter_index]);

                        MPI_Send(
                            fragment.new_pixmap + GRAYSCALE_PIXEL_SIZE * fragment.max_width,
                            GRAYSCALE_PIXEL_SIZE * fragment.max_width * (fragment.end - fragment.start),
                            MPI_UNSIGNED_CHAR,
                            0,
                            0,
                            MPI_COMM_WORLD
                        );
                    }
                }

                if (fragment.type == P6)
                {
                    for (int filter_index = 0; filter_index < num_filters; ++filter_index)
                    {
                        MPI_Recv(
                            fragment.pixmap,
                            RGB_PIXEL_SIZE * fragment.max_width * (fragment.end - fragment.start + OFFSET_BORDER(rank, P)),
                            MPI_UNSIGNED_CHAR,
                            0,
                            0,
                            MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE
                        );

                        compute_filtered_p6_image(filters[filter_index]);

                        MPI_Send(
                            fragment.new_pixmap + RGB_PIXEL_SIZE * fragment.max_width,
                            RGB_PIXEL_SIZE * fragment.max_width * (fragment.end - fragment.start),
                            MPI_UNSIGNED_CHAR,
                            0,
                            0,
                            MPI_COMM_WORLD
                        );
                    }
                }
            }
        }
    }
    MPI_Finalize();
    
    return 0;
}
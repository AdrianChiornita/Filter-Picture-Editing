#include "mrenderer.h"
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

extern int num_threads;
extern int resolution;

int main(int argc, char * argv[]) {

	if(argc < 4) {
		printf("Incorrect number of arguments\n");
		exit(-1);
	}
	resolution = atoi(argv[2]);
	num_threads = atoi(argv[3]);

	image im;
	initialize(&im);

	struct timespec start, finish;
	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);
	render(&im);
	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("%lf\n", elapsed);
	writeData(argv[1], &im);
	return 0;
}

#include "ssaa.h"
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

extern int num_threads;
extern int resize_factor;

int main(int argc, char * argv[]) {
	if(argc < 5) {
		printf("Incorrect number of arguments\n");
		exit(-1);
	}

	resize_factor = atoi(argv[3]);
	num_threads = atoi(argv[4]);

	image input;
	image output;

	readInput(argv[1], &input);

	struct timespec start, finish;
	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);
	resize(&input, &output);
	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("%lf\n", elapsed);
	writeData(argv[2], &output);
	return 0;
}

build: image_processing

image_processing: image_processing.c
	mpicc -o image_processing image_processing.c -lm -Wall

serial: image_processing
	mpirun -np 1 image_processing imagini.in

distrib: image_processing
	mpirun -np 4 image_processing imagini.in

clean:
	rm -f image_processing

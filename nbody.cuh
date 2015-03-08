#ifndef _NBODY_
#define _NBODY_

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

//void initCUDA(int N, int blockSize); 
int ComputeNextFrameNBodySimlation();
__global__ void hello(char *a, int *b);

#endif

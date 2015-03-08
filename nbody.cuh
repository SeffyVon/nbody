#ifndef _NBODY_
#define _NBODY_


#include <stdio.h>
#include <vector_types.h>
#include <cstdlib>
#include <cmath>

#define GL_GLEXT_PROTOTYPES

#include <GL/glut.h>

#include <cuda_gl_interop.h>

#define ORTHO_VERSION 1

#define WINDOW_W 1024
#define WINDOW_H 768

#define N_SIZE 2560
#define BLOCK_SIZE 1024
#define GRID_SIZE 1000

#define GRAVITY 0.000667 //9.81f
#define EPSILON2 0.01f // epsilon ^ 2
#define TIME_STEP 0.01f


struct Body {
	float3 pos; // position
	float3 a; // acceleration
	float3 v; // velocity
	float mass; // mass

	Body() {
		pos.x = -320 + ((float)rand()/(float)(RAND_MAX)) * 640;
		pos.y =-240 + ((float)rand()/(float)(RAND_MAX)) * 480;
		pos.z = 0.0f ;
		a.x = -5 + ((float)rand()/(float)(RAND_MAX))*10;
		a.y = -5 + ((float)rand()/(float)(RAND_MAX))*10;
		a.z = -5 + ((float)rand()/(float)(RAND_MAX))*10;

		v.x = -5 + ((float)rand()/(float)(RAND_MAX))*10;
		v.y = -5 + ((float)rand()/(float)(RAND_MAX))*10;
		v.z = -5 + ((float)rand()/(float)(RAND_MAX))*10;

		mass = ((float)rand()/(float)(RAND_MAX)) * 100000;
	}

	Body(float x, float y, float z, float mass){
		pos.x = x;
		pos.y = y;
		pos.z = z;
		a.x = a.y = a.z = 0.0f;
		v.x = v.y = v.z = 0.0f;
		this->mass = mass;
	}

};


extern int bodies_size;
extern Body *bodies_dev;
extern Body bodies[N_SIZE];
extern GLuint vertexArray;
extern float cx,cy,cz;
 


void init();
void deinit();


void initCUDA();
void initGL();


int runKernelNBodySimulation();

//__device__ 
//void updateAcceleration(Body &body);

__device__
void updateVelocity(Body &body);

__device__
void updatePosition(Body &body);
 
__global__ 
void nbody(Body *body);

__device__
void bodyBodyInteraction(Body& self, Body other);
#endif

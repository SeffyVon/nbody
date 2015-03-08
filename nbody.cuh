#ifndef _NBODY_
#define _NBODY_


#include <stdio.h>
#include <vector_types.h>
#include <cstdlib>

#define GL_GLEXT_PROTOTYPES

#include <GL/glut.h>

#include <cuda_gl_interop.h>

#define N_SIZE 10
#define BLOCK_SIZE 1024
#define GRID_SIZE 1


struct Body {
	float3 pos; // position
	float3 a; // acceleration
	float3 v; // velocity
	float mass; // mass

	Body() {
		pos.x = pos.y = pos.z = 1.0f;
		a.x = a.y = a.z = 0.0f;
		v.x = v.y = v.z = 0.0f;
		mass = 1.0f;
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
 



void init();
void deinit();


void initCUDA();
void initGL();


int runKernelNBodySimulation();

__device__ 
void updateAcceleration(Body &body);
__device__
void updateVelocity(Body &body);

__device__
void updatePosition(Body &body);
 
__global__ 
void nbody(Body *body);


#endif

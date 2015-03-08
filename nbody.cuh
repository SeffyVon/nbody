#ifndef _NBODY_
#define _NBODY_

#include <stdio.h>
#include <vector_types.h>
#include <cstdlib>
#include <cmath>


#define N_SIZE 10
#define BLOCK_SIZE 1024
#define GRID_SIZE 1

#define GRAVITY 9.81f
#define EPSILON2 0.01f // epsilon ^ 2
#define TIME_STEP 0.01f


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
 



void init();

void initCUDA();

void deinit();

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

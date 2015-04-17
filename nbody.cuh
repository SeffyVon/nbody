#ifndef _NBODY_
#define _NBODY_


#include <stdio.h>
#include <vector_types.h>
#include <cstdlib>
#include <cmath>

#define GL_GLEXT_PROTOTYPES
//#include <GL/glew.h>
#include <GL/glut.h>


#include <cuda_gl_interop.h>

const int ORTHO_VERSION=0;

#define WINDOW_W 1920
#define WINDOW_H 1080

#define N_SIZE 10000
#define BLOCK_SIZE 1024
#define GRID_SIZE 1000
#define SOFT_FACTOR 0.00125f

#define GRAVITACIONAL_CONSTANT 0.01f//0.000667 //9.81f
//#define EPSILON2 4.930380657631323783822134085449116758237409e-32// epsilon ^ 2
#define TIME_STEP 0.001f
//#define DAMPING 0.995f
#define PI 3.14152926f
#define DENSITY 100000

struct Body {
	float3 pos; // position
	float3 a; // acceleration
	float3 v; // velocity
	float mass; // mass
	float radius;
	float r, g, b, alpha; //color


	Body() {

		if(ORTHO_VERSION) {
			pos.x = (-WINDOW_W/2 + ((float)rand()/(float)(RAND_MAX)) * WINDOW_W) * 0.9;
			pos.y = (-WINDOW_H/2 + ((float)rand()/(float)(RAND_MAX)) * WINDOW_H) * 0.9;
			pos.z = 0.0f ;
			a.x = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			a.y = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			a.z = -5 + ((float)rand()/(float)(RAND_MAX))*10;

			v.x = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			v.y = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			v.z = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			
		}
		else{
			pos.x = (-WINDOW_W/2 + ((float)rand()/(float)(RAND_MAX)) * WINDOW_W) * 0.9;
			pos.y = (-WINDOW_H/2 + ((float)rand()/(float)(RAND_MAX)) * WINDOW_H) * 0.9;
			pos.z = (-100 + ((float)rand()/(float)(RAND_MAX)) * 100) * 0.9 ;
			a.x = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			a.y = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			a.z = -5 + ((float)rand()/(float)(RAND_MAX))*10;

			v.x = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			v.y = -5 + ((float)rand()/(float)(RAND_MAX))*10;
			v.z = -5 + ((float)rand()/(float)(RAND_MAX))*10;
		}
		
		radius = ((float)rand()/(float)(RAND_MAX))*3.0;
		mass = 4.0/3.0*PI * radius*radius*radius * DENSITY;
		r = 0.5f;
		g = 1.0f;
		b = 1.0f;
		alpha = 1.0f;

	}


	Body(float x, float y, float z, float radius){
		pos.x = x;
		pos.y = y;
		pos.z = z;
		a.x = a.y = a.z = 0.0f;
		v.x = v.y = v.z = 0.0f;
		this->radius = radius;
		this->mass = 4.0/3.0* PI * radius*radius*radius * DENSITY;
		r = ((float)rand()/(float)(RAND_MAX));
		g = ((float)rand()/(float)(RAND_MAX));
		b = ((float)rand()/(float)(RAND_MAX));
		alpha = 1.0f;
	}

};

struct Camera {
    float camX, camY, camZ;
    float forwardX, forwardY, forwardZ; 
    float upX, upY, upZ; 

    float theta, phi;

    Camera() {
        camX = 0, camY = 0, camZ = 200;
        forwardX = 0, forwardY = 0, forwardZ = -1;
        upX = 0, upY = 1, upZ = 0;

        theta = 0; phi = M_PI;
    }

};



extern Camera camera;
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
void bodyBodyCollision(Body &self, Body &other, float3 &cur_a);
 
__global__ 
void nbody(Body *body);

__device__
void bodyBodyInteraction(Body &self, Body &other, float3 &cur_a, float3 dist3, float dist_sqr);
#endif

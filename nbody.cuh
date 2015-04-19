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

const int ORTHO_VERSION=0; // 1 is 2D version, 0 is 3D version.

#define WINDOW_W 1920
#define WINDOW_H 1080

#define N_SIZE 10000
#define BLOCK_SIZE 1024
#define GRID_SIZE 1000
#define SOFT_FACTOR 0.00125f

#define GRAVITATIONAL_CONSTANT 0.01f
#define TIME_STEP 0.001f
#define PI 3.14152926f
#define DENSITY 1000000

struct Body {
	float3 pos; // position
	float3 a; // acceleration
	float3 v; // velocity
	float mass; // mass
	float radius;

	// default initialization
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
			pos.z = (-500 + ((float)rand()/(float)(RAND_MAX)) * 500) * 0.9 ;

			a.x = -50 + ((float)rand()/(float)(RAND_MAX))*50;
			a.y = -50 + ((float)rand()/(float)(RAND_MAX))*50;
			a.z = -50 + ((float)rand()/(float)(RAND_MAX))*50;

			v.x = -50 + ((float)rand()/(float)(RAND_MAX))*50;
			v.y = -50 + ((float)rand()/(float)(RAND_MAX))*50;
			v.z = -50 + ((float)rand()/(float)(RAND_MAX))*50;
		}
		
		radius = ((float)rand()/(float)(RAND_MAX))*3.0;
		mass = 4.0/3.0*PI * radius*radius*radius * DENSITY;

	}

	// initialization with parameters
	Body(float x, float y, float z, float radius){
		pos.x = x;
		pos.y = y;
		pos.z = z;

		a.x = a.y = a.z = 0.0f;
		v.x = v.y = v.z = 0.0f;

		this->radius = radius;
		this->mass = 4.0/3.0* PI * radius*radius*radius * DENSITY;
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

__device__
void bodyBodyCollision(Body &self, Body &other, float3 &cur_a);
 
__global__ 
void nbody(Body *body);

__device__
void bodyBodyInteraction(Body &self, Body &other, float3 &cur_a, float3 dist3, float dist_sqr);
#endif


#include "nbody.cuh"
#include <iostream>
#include <fstream>


Camera camera;
int bodies_size = 0;
Body *bodies_dev = NULL;

//ody bodies[N_SIZE]= {  Body(0,0,0,1000000), Body(13,2,0,1000000),  Body(8,3,0,1000000), Body(2,7,0,1000000), Body(7,2,0,1000000), Body(9,2,0,1000000),  Body(4,3,0,1000000), Body(2,4,0,1000000) };
Body bodies[N_SIZE] = {Body(0, 0, 0, 1.0f) , Body(0,100,0,1.0f)};
GLuint vertexArray;

__device__
int icbrt2(unsigned x) {
   int s;
   unsigned y, b, y2;

   y2 = 0;
   y = 0;
   for (s = 30; s >= 0; s = s - 3) {
      y2 = 4*y2;
      y = 2*y;
      b = (3*(y2 + y) + 1) << s;
      if (x >= b) {
         x = x - b;
         y2 = y2 + 2*y + 1;
         y = y + 1;
      }
   }
   return y;
}

void initCUDA()
{

	bodies_size = N_SIZE * sizeof(Body);
	cudaMalloc( (void**)&bodies_dev, bodies_size ); 
	cudaMemcpy( bodies_dev, bodies, bodies_size, cudaMemcpyHostToDevice );
	//cudaGLRegisterBufferObject(vertexArray);

}

void initGL()
{

    glEnable(GL_CULL_FACE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glEnable(GL_LIGHTING);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);

    glEnable(GL_LIGHT0);

    glEnable(GL_COLOR_MATERIAL);


	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    /*void glOrtho(GLdouble  left,  GLdouble  right,  GLdouble  bottom,  GLdouble  top,  GLdouble  nearVal,  GLdouble  farVal);*/
    if( ORTHO_VERSION )
    {
    	glOrtho(-WINDOW_W/2, WINDOW_W/2, -WINDOW_H/2, WINDOW_H/2, -100, 100);
    }
    else
    {
    	gluPerspective (45, (float)WINDOW_W/(float)WINDOW_H, 1, 2000);
    }
   
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if( !ORTHO_VERSION )
   		  gluLookAt(camera.camX,camera.camY,camera.camZ, //Camera position
              camera.camX+camera.forwardX,camera.camY+camera.forwardY,camera.camZ+camera.forwardZ, //Position of the object to look at
              camera.upX,camera.upY,camera.upZ); //Camera up direction


	glEnable(GL_DEPTH_TEST);

	glEnable(GL_FOG);
	
}

void init()
{
	initGL();
	initCUDA();
	atexit(deinit);
}

void deinit()
{
	cudaFree( bodies_dev );
}


__device__
void updatePosAndVel(Body &body, float3 cur_a)
{
	float newvx = body.v.x + (body.a.x + cur_a.x ) / 2 * TIME_STEP;
	float newvy = body.v.y + (body.a.y + cur_a.y ) / 2 * TIME_STEP;
	float newvz = body.v.z + (body.a.z + cur_a.z ) / 2 * TIME_STEP;

	//update position
	body.pos.x += newvx * TIME_STEP + body.a.x * TIME_STEP * TIME_STEP /2;
	body.pos.y += newvy * TIME_STEP + body.a.y * TIME_STEP * TIME_STEP /2;
	body.pos.z += newvz * TIME_STEP + body.a.z * TIME_STEP * TIME_STEP /2;

	//update velocity
	body.v.x = newvx;
	body.v.y = newvy;
	body.v.z = newvz; 
}

__device__
void bodyBodyInteraction(Body &self, Body &other, float3 &cur_a, float3 dist3, float dist_sqr)
{

	float dist_six = dist_sqr * dist_sqr * dist_sqr;
	float dist_cub = sqrtf(dist_six);

	cur_a.x += (other.mass * dist3.x) / dist_cub;
	cur_a.y += (other.mass * dist3.y) / dist_cub;
	cur_a.z += (other.mass * dist3.z) / dist_cub;
}

__device__
void bodyBodyCollision(Body &self, Body &other, float3 &cur_a)
{

	
	float m = self.mass+other.mass;
	float3 vertex;
	vertex.x = (self.v.x * self.mass +other.v.x * other.mass)/m;
	vertex.y = (self.v.y * self.mass +other.v.y * other.mass)/m;
	vertex.z = (self.v.z * self.mass +other.v.z * other.mass)/m;

	float3 zero_vertex;
	zero_vertex.x = 0.0f;
	zero_vertex.y = 0.0f;
	zero_vertex.z = 0.0f;

	if(self.mass>other.mass){
		self.v = vertex;
		self.mass = m;

		other.mass = 0.0f;
		other.a = zero_vertex;
		other.v = zero_vertex;

	}else{
		other.v = vertex;
		other.mass = m;

		self.mass = 0.0f;
		self.a = zero_vertex;
		self.v = zero_vertex;
	}

}


__global__ 
void nbody(Body *body) 
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(idx < N_SIZE && body[idx].mass != 0)
	{
		float mass_before = body[idx].mass;

		float3 cur_a;
		cur_a.x = 0 ;
		cur_a.y = 0 ;
		cur_a.z = 0 ;

		for(int i = 0; i < N_SIZE; i++){

			if( i != idx && body[i].mass!= 0){

				if(body[idx].mass!=0){

					float3 dist3;

					dist3.x = body[i].pos.x - body[idx].pos.x;
					dist3.y = body[i].pos.y - body[idx].pos.y;
					dist3.z = body[i].pos.z - body[idx].pos.z;

					float dist_sqr = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z + SOFT_FACTOR;

					if( sqrt(dist_sqr) > body[idx].radius + body[i].radius )
						bodyBodyInteraction(body[idx], body[i], cur_a, dist3, dist_sqr);
					else{
						bodyBodyCollision(body[idx], body[i], cur_a);	
					}

				}
			}
		}

		cur_a.x *= GRAVITATIONAL_CONSTANT;
		cur_a.y *= GRAVITATIONAL_CONSTANT;
		cur_a.z *= GRAVITATIONAL_CONSTANT;
		
		updatePosAndVel(body[idx], cur_a);

		body[idx].a.x = cur_a.x;
		body[idx].a.y = cur_a.y;
		body[idx].a.z = cur_a.z;

		if(body[idx].mass != mass_before)
			body[idx].radius = icbrt2(body[idx].mass/ (DENSITY * 4.0/3.0*PI)); 
	}
}



int runKernelNBodySimulation()
{
	// Map the buffer to CUDA

	nbody<<<GRID_SIZE, BLOCK_SIZE>>>(bodies_dev);

	cudaMemcpy( bodies, bodies_dev, bodies_size, cudaMemcpyDeviceToHost ); 


	return EXIT_SUCCESS;
}
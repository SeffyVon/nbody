
#include "nbody.cuh"
#include <iostream>
#include <fstream>



int bodies_size = 0;
Body *bodies_dev = NULL;

Body bodies[N_SIZE];// {  Body(0,0,0,1000000), Body(13,2,0,1),  Body(8,3,0,1), Body(2,7,0,1), Body(7,2,0,1), Body(9,2,0,1),  Body(4,3,0,1), Body(2,4,0,1) };
GLuint vertexArray;




void readFromFile2()
{
	int i = 0;

	FILE* f_sample = fopen("dubinski.tab","r");
	double mass;
	double x, y, z, vx, vy, vz;
	while ( i < N_SIZE && fscanf( f_sample, "%lf %lf %lf %lf %lf %lf %lf", &mass, &x, &y, &z, &vx, &vy, &vz )>0 ){ // the ith element will store in the sample[i+head_blocks_size]
	    bodies[i].mass = (float)mass;
	    bodies[i].pos.x = (float)x;
	    bodies[i].pos.y = (float)y;
	    bodies[i].pos.z = (float)z;
	    bodies[i].v.x = (float)vx;
	    bodies[i].v.y =(float) vy;
	    bodies[i].v.z = (float)vz;
	    i++;
  }
  fclose(f_sample);
}

void initCUDA()
{

	readFromFile2();
	bodies_size = N_SIZE * sizeof(Body);
	cudaMalloc( (void**)&bodies_dev, bodies_size ); 
	cudaMemcpy( bodies_dev, bodies, bodies_size, cudaMemcpyHostToDevice );

	//cudaGLRegisterBufferObject(vertexArray);

}

void initGL()
{
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    /*void glOrtho(GLdouble  left,  GLdouble  right,  GLdouble  bottom,  GLdouble  top,  GLdouble  nearVal,  GLdouble  farVal);*/
    glOrtho(-400, 400, -300, 300, -100, 100);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	//glGenBuffers(1,&vertexArray);
	//glBindBuffer(GL_ARRAY_BUFFER, vertexArray);
	//glBufferData(GL_ARRAY_BUFFER, N_SIZE*sizeof(Body), bodies, GL_DYNAMIC_COPY);
	
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

/*
__device__ 
void updateAcceleration(Body &body)
{
	body.a.x += a.x;
	body.a.y += a.y;
	body.a.z += a.z;
}
*/
__device__
void updateVelocity(Body &body, float3 cur_a)
{
	body.v.x += (body.a.x + cur_a.x ) / 2 * TIME_STEP;
	body.v.y += (body.a.y + cur_a.y ) / 2 * TIME_STEP;
}

__device__
void updatePosition(Body &body)
{
	body.pos.x += body.v.x * TIME_STEP + body.a.x * TIME_STEP * TIME_STEP /2;
	body.pos.y += body.v.y * TIME_STEP + body.a.y * TIME_STEP * TIME_STEP /2;
}

__device__
void bodyBodyInteraction(Body self, Body other, float3 &cur_a)
{
	float3 dist3;
	dist3.x = other.pos.x - self.pos.x;
	dist3.y = other.pos.y - self.pos.y;

	float dist_sqr = dist3.x * dist3.x + dist3.y * dist3.y + EPSILON2; 
	float dist_six = dist_sqr * dist_sqr * dist_sqr;
	float dist_cub = sqrtf(dist_six);

	cur_a.x += (other.mass * dist3.x) / dist_cub;
	cur_a.y += (other.mass * dist3.y) / dist_cub;

}

__global__ 
void nbody(Body *body) 
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(idx < N_SIZE)
	{
		float3 cur_a;
		cur_a.x = 0.0f;
		cur_a.y = 0.0f;

		for(int i = 0; i < N_SIZE; i++){
			if( i != idx )
				bodyBodyInteraction(body[idx], body[i], cur_a);
		}

		cur_a.x *= GRAVITY;
		cur_a.y *= GRAVITY;
		
		updatePosition(body[idx]);
		updateVelocity(body[idx], cur_a);




		body[idx].a.x = cur_a.x;
		body[idx].a.y = cur_a.y;

	}
}


int runKernelNBodySimulation()
{
	// Map the buffer to CUDA
	//cudaGLMapBufferObject(&bodies_dev, vertexArray);

	nbody<<<GRID_SIZE, BLOCK_SIZE>>>(bodies_dev);

	cudaMemcpy( bodies, bodies_dev, bodies_size, cudaMemcpyDeviceToHost ); 


	for(int i = 0; i < N_SIZE; i++)
	{
		//printf("a[%d]=(%f,%f) ", i, bodies[i].a.x, bodies[i].a.y);
		//printf("v[%d]=(%f,%f)\n", i, bodies[i].v.x, bodies[i].v.y);
		//printf("pos[%d]=(%f,%f)\n", i, bodies[i].pos.x, bodies[i].pos.y);
	}

	// Unmap the buffer
	//cudaGLUnmapbufferObject(vertexArray);
	
	/*
	for(int i = 0; i < N_SIZE; i++){
		printf("a=(%f,%f,%f)\n", bodies[i].a.x, bodies[i].a.y, bodies[i].a.z);
	}*/

	

	return EXIT_SUCCESS;
}

#include "nbody.cuh"

int bodies_size = 0;
Body *bodies_dev = NULL;
Body bodies[N_SIZE] =  {  Body(1,0,4,4), Body(13,2,0,1),  Body(8,3,4,2), Body(2,7,3,2), Body(1,2,0,4), Body(3,2,0,3),  Body(4,3,4,2), Body(2,4,3,2) };
GLuint vertexArray;

void initCUDA()
{
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
    glOrtho(0, 640, 480, 0, -100, 100);

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

__device__ 
void updateAcceleration(Body &body){
	body.a.x ++;
	body.a.y ++;
	//body.a.z ++;
}

__device__
void updateVelocity(Body &body){
	body.v.x += body.a.x;
	body.v.y += body.a.y;
	//body.v.z += body.a.z;
}

__device__
void updatePosition(Body &body){
	body.pos.x += body.v.x;
	body.pos.y += body.v.y;
	//body.pos.z += body.v.z;
}
 
__global__ 
void nbody(Body *body) 
{
	updateAcceleration(body[threadIdx.x]);
	updateVelocity(body[threadIdx.x]);
	updatePosition(body[threadIdx.x]);
}

int runKernelNBodySimulation()
{
	// Map the buffer to CUDA
	//cudaGLMapBufferObject(&bodies_dev, vertexArray);

	nbody<<<GRID_SIZE, BLOCK_SIZE>>>(bodies_dev);

	cudaMemcpy( bodies, bodies_dev, bodies_size, cudaMemcpyDeviceToHost ); 

	// Unmap the buffer
	//cudaGLUnmapbufferObject(vertexArray);
	
	/*
	for(int i = 0; i < N_SIZE; i++){
		printf("a=(%f,%f,%f)\n", bodies[i].a.x, bodies[i].a.y, bodies[i].a.z);
	}*/
	

	return EXIT_SUCCESS;
}
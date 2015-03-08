
#include "nbody.cuh"

int bodies_size = 0;
Body *bodies_dev = NULL;

Body bodies[N_SIZE] =  {  Body(1,0,0,1), Body(13,2,0,1),  Body(8,3,0,1), Body(2,7,0,1), Body(7,2,0,1), Body(9,2,0,1),  Body(4,3,0,1), Body(2,4,0,1) };
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

/*
__device__ 
<<<<<<< HEAD
void updateAcceleration(Body &body)
{
	body.a.x += a.x;
	body.a.y += a.y;
	body.a.z += a.z;
}
*/
__device__
void updateVelocity(Body &body, float3 delta_a)
{
	body.v.x += (body.a.x + delta_a.x) * TIME_STEP / 2;
	body.v.y += (body.a.y + delta_a.y) * TIME_STEP/ 2;
}

__device__
void updatePosition(Body &body)
{
	body.pos.x += body.v.x * TIME_STEP + body.a.x * TIME_STEP * TIME_STEP /2;
	body.pos.y += body.v.y * TIME_STEP + body.a.y * TIME_STEP * TIME_STEP /2;
}

__device__
void bodyBodyInteraction(Body self, Body other, float3 &delta_a)
{
	float3 dist3;
	dist3.x = other.pos.x - self.pos.x;
	dist3.y = other.pos.y - self.pos.y;

	float dist_sqr = dist3.x * dist3.x + dist3.y * dist3.y + EPSILON2; 
	float dist_six = dist_sqr * dist_sqr * dist_sqr;
	float dist_cub = sqrtf(dist_six);

	delta_a.x += (other.mass * dist3.x) / dist_cub;
	delta_a.y += (other.mass * dist3.y) / dist_cub;
=======
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
>>>>>>> e20656691d508a7146bd9e1d0a1361a89bb15087
}

__global__ 
void nbody(Body *body) 
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(idx < N_SIZE)
	{
		float3 delta_a;
		delta_a.x = 0.0f;
		delta_a.y = 0.0f;

		for(int i = 0; i < N_SIZE; i++){
			if( i != idx )
				bodyBodyInteraction(body[idx], body[i], delta_a);
		}

		delta_a.x *= GRAVITY;
		delta_a.y *= GRAVITY;
			
		updateVelocity(body[idx], delta_a);

		updatePosition(body[idx]);


		body[idx].a.x += delta_a.x;
		body[idx].a.y += delta_a.y;

	}
}


int runKernelNBodySimulation()
{
	// Map the buffer to CUDA
	//cudaGLMapBufferObject(&bodies_dev, vertexArray);

	nbody<<<GRID_SIZE, BLOCK_SIZE>>>(bodies_dev);

	cudaMemcpy( bodies, bodies_dev, bodies_size, cudaMemcpyDeviceToHost ); 

<<<<<<< HEAD
	for(int i = 0; i < N_SIZE; i++)
	{
		printf("a[%d]=(%f,%f) ", i, bodies[i].a.x, bodies[i].a.y);
		//printf("v[%d]=(%f,%f)\n", i, bodies[i].v.x, bodies[i].v.y);
		printf("pos[%d]=(%f,%f)\n", i, bodies[i].pos.x, bodies[i].pos.y);
	}
=======
	// Unmap the buffer
	//cudaGLUnmapbufferObject(vertexArray);
	
	/*
	for(int i = 0; i < N_SIZE; i++){
		printf("a=(%f,%f,%f)\n", bodies[i].a.x, bodies[i].a.y, bodies[i].a.z);
	}*/
>>>>>>> e20656691d508a7146bd9e1d0a1361a89bb15087
	

	return EXIT_SUCCESS;
}
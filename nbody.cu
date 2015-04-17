
#include "nbody.cuh"
#include <iostream>
#include <fstream>


Camera camera;
int bodies_size = 0;
Body *bodies_dev = NULL;

//ody bodies[N_SIZE]= {  Body(0,0,0,1000000), Body(13,2,0,1000000),  Body(8,3,0,1000000), Body(2,7,0,1000000), Body(7,2,0,1000000), Body(9,2,0,1000000),  Body(4,3,0,1000000), Body(2,4,0,1000000) };
Body bodies[N_SIZE] = {Body(0, 0, 0, 1.0f) , Body(0,100,0,1.0f)};
GLuint vertexArray;

GLuint v,f,f2,p;



void readFromFile2()
{
	int i = 0;

	FILE* f_sample = fopen("dubinski.tab","r");
	double mass;
	double x, y, z, vx, vy, vz;
	int j = 0;
	while ( i < N_SIZE && fscanf( f_sample, "%lf %lf %lf %lf %lf %lf %lf", &mass, &x, &y, &z, &vx, &vy, &vz )>0 ){ // the ith element will store in the sample[i+head_blocks_size]
	    
		if( j < 1024|| j > 32768 && j < 32768+512 || j > (32768 + 16384) ){

		    bodies[i].mass = (float)mass*10;
		    bodies[i].pos.x = (float)x;
		    bodies[i].pos.y = (float)y;
		    bodies[i].pos.z = (float)z;
		    bodies[i].v.x = (float)vx*5;
		    bodies[i].v.y =(float) vy*5;
		    bodies[i].v.z = (float)vz*5;
		    i++;
		}
		j++;

  }
  fclose(f_sample);
}

char* readShaderFromFile(const char* filename){


    std::ifstream t;
    int length;
    t.open(filename);   
    t.seekg(0, std::ios::end);    
    length = t.tellg();           
    t.seekg(0, std::ios::beg);    
    char* buffer = new char[length];   
    t.read(buffer, length);      
    t.close();

    return buffer;

}

void setShaders() {

    char *vs = NULL,*fs = NULL;

    v = glCreateShader(GL_VERTEX_SHADER);
    f = glCreateShader(GL_FRAGMENT_SHADER);
    //f2 = glCreateShader(GL_FRAGMENT_SHADER);

    vs = readShaderFromFile("minimal.vert");
    fs = readShaderFromFile("minimal.frag");

    const char * vv = vs;
    const char * ff = fs;

    glShaderSource(v, 1, &vv,NULL);
    //glShaderSource(f, 1, &ff,NULL);

    free(vs);free(fs);

    glCompileShader(v);
    glCompileShader(f);

    //printShaderInfoLog(v);
    //printShaderInfoLog(f);
    //printShaderInfoLog(f2);

    p = glCreateProgram();
    glAttachShader(p,v);
    glAttachShader(p,f);

    glLinkProgram(p);
    //printProgramInfoLog(p);

    glUseProgram(p);

}

void initCUDA()
{

	//readFromFile2();
	bodies_size = N_SIZE * sizeof(Body);
	cudaMalloc( (void**)&bodies_dev, bodies_size ); 
	cudaMemcpy( bodies_dev, bodies, bodies_size, cudaMemcpyHostToDevice );

	//cudaGLRegisterBufferObject(vertexArray);

}

void initGL()
{
	

	glewInit();
    if (glewIsSupported("GL_VERSION_2_0"))
        printf("Ready for OpenGL 2.0\n");
    else {
        printf("OpenGL 2.0 not supported\n");
        exit(1);
    }

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
	//glGenBuffers(1,&vertexArray);
	//glBindBuffer(GL_ARRAY_BUFFER, vertexArray);
	//glBufferData(GL_ARRAY_BUFFER, N_SIZE*sizeof(Body), bodies, GL_DYNAMIC_COPY);

	glEnable(GL_DEPTH_TEST);
	
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


	/*for merging*/
	if(self.mass>other.mass){

		self.v.x = (self.v.x * self.mass +other.v.x * other.mass)/(self.mass+other.mass);
		self.v.y = (self.v.y * self.mass +other.v.y * other.mass)/(self.mass+other.mass);
		self.v.z = (self.v.z * self.mass +other.v.z * other.mass)/(self.mass+other.mass);
		self.mass += other.mass;
		self.radius = cbrt(self.mass/ (DENSITY * 4.0/3.0*PI)) ; 

		other.mass = 0.0f; 

		other.a.x = 0.0f;
		other.a.y = 0.0f;
		other.a.z = 0.0f;

		other.v.x = 0.0f;
		other.v.y = 0.0f;
		other.v.z = 0.0f;
	}else{

		other.v.x = (self.v.x * self.mass +other.v.x * other.mass)/(self.mass+other.mass);
		other.v.y = (self.v.y * self.mass +other.v.y * other.mass)/(self.mass+other.mass);
		other.v.z = (self.v.z * self.mass +other.v.z * other.mass)/(self.mass+other.mass);

		other.mass += self.mass;
		other.radius = cbrt(other.mass/ (DENSITY * 4.0/3.0*PI)) ; 
		self.mass = 0.0f; 

		self.a.x = 0.0f;
		self.a.y = 0.0f;
		self.a.z = 0.0f;

		self.v.x = 0.0f;
		self.v.y = 0.0f;
		self.v.z = 0.0f;
	}

	//other.alpha = 0.0f;
	// other.radius = 0;

	//Update alpha value according to the mass (if mass == 0 then alpha is also set to zero, so the body becomes transparent)

	/*no merging*/
	// other.v.x = self.v.x;
	// other.v.y = self.v.y;
	// other.v.z = self.v.z;

	// cur_a.x = 0;
	// cur_a.y = 0;
	// cur_a.z = 0;
}


__global__ 
void nbody(Body *body) 
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(idx < N_SIZE && body[idx].mass != 0)
	{
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

		cur_a.x *= GRAVITACIONAL_CONSTANT;
		cur_a.y *= GRAVITACIONAL_CONSTANT;
		cur_a.z *= GRAVITACIONAL_CONSTANT;
		
		updatePosAndVel(body[idx], cur_a);

		body[idx].a.x = cur_a.x;
		body[idx].a.y = cur_a.y;
		body[idx].a.z = cur_a.z;

		if(body[idx].radius<2.0f)
			body[idx].r = body[idx].radius/2.0f;
	}

	__syncthreads();
	if( idx < N_SIZE) {
		
		if(body[idx].mass == 0.0 )
			body[idx].alpha = 0.0;
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
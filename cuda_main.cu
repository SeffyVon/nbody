// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include <GL/glut.h>
#include <vector_types.h>

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

Body bodies[N_SIZE] =  {  Body(1,0,4,4), Body(13,2,0,1),  Body(8,3,4,2), Body(2,7,3,2), Body(1,2,0,4), Body(3,2,0,3),  Body(4,3,4,2), Body(2,4,3,2) };


__device__ 
void updateAcceleration(Body &body){
	body.a.x ++;
	body.a.y ++;
	body.a.z ++;
}

__device__
void updateVelocity(Body &body){
	body.v.x += body.a.x;
	body.v.y += body.a.y;
	body.v.z += body.a.z;
}

__device__
void updatePosition(Body &body){
	body.pos.x += body.v.x;
	body.pos.y += body.v.y;
	body.pos.z += body.v.z;
}
 
__global__ 
void nbody(Body *body) 
{
	updateAcceleration(body[threadIdx.x]);
	updateVelocity(body[threadIdx.x]);
	updatePosition(body[threadIdx.x]);
}


int mainTestCUDA()
{

	int bodies_size = N_SIZE * sizeof(Body);
	Body *bodies_dev;
	cudaMalloc( (void**)&bodies_dev, bodies_size ); 
	cudaMemcpy( bodies_dev, bodies, bodies_size, cudaMemcpyHostToDevice ); 

	nbody<<<GRID_SIZE, BLOCK_SIZE>>>(bodies_dev);
	
	for(int i = 0; i < N_SIZE; i++){
		printf("a=(%f,%f,%f)\n", bodies[i].a.x, bodies[i].a.y, bodies[i].a.z);
	}
	cudaMemcpy( bodies, bodies_dev, bodies_size, cudaMemcpyDeviceToHost ); 
	cudaFree( bodies_dev );
	return EXIT_SUCCESS;
}



void draw(void) {

    // Black background
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //Draw i
    glFlush();

    mainTestCUDA();

  //  glutPostRedisplay(); //force it to render continueously

}

//Main program
int main(int argc, char **argv) {

    glutInit(&argc, argv);

    /*Setting up  The Display
    /    -RGB color model + Alpha Channel = GLUT_RGBA
    */
    glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE);

    //Configure Window Postion
    glutInitWindowPosition(50, 25);

    //Configure Window Size
    glutInitWindowSize(480,480);

    //Create Window
    glutCreateWindow("Hello OpenGL");


    //Call to the drawing function
    glutDisplayFunc(draw);

    // Loop require by OpenGL
    glutMainLoop();
    return 0;
}

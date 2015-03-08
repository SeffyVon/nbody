#include "util.cuh"
#include "nbody.cuh"


extern Body bodies[N_SIZE];

void timerFunc(int value)
{
     glutPostRedisplay();
    glutTimerFunc (5, timerFunc, 10);
    
}
 
//utility functions definitions go here
void draw() {

    // Black background
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);



    runKernelNBodySimulation();

    glColor3ub( 255, 0, 255 );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );
    glVertexPointer( 2, GL_FLOAT, sizeof(Body), &bodies[0].pos.x );
    glColorPointer( 3, GL_UNSIGNED_BYTE, sizeof(Body), &bodies[0].v.x );
    glPointSize( 5.0 );
    glDrawArrays( GL_POINTS, 0, N_SIZE );
    glDisableClientState( GL_VERTEX_ARRAY );
    glDisableClientState( GL_COLOR_ARRAY );

    glFlush();
    
    //Draw stuff
   // for(int i = 0; i < N_SIZE; i++){
	//	printf("HOHO a=(%f,%f,%f)\n", bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
	//}


  // glutPostRedisplay();

}


 

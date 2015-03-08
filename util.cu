#include "util.cuh"
#include "nbody.cuh"

 
//utility functions definitions go here
void draw() {

    // Black background
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glFlush();

    runKernelNBodySimulation();


    //glutPostRedisplay();

}


 

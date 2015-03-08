// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include <GL/glut.h>
#include "util.cuh"
#include "nbody.cuh"

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
    glutCreateWindow("NBody Simulation - Little Hamster and Sheep");


    //Call to the drawing function
    glutDisplayFunc(draw);


    init();

    // Loop require by OpenGL
    glutMainLoop();


    return 0;
}

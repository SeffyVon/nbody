#include "util.cuh"
#include "nbody.cuh"
#include <iostream>
#include <fstream>
extern Body bodies[N_SIZE];


//Main program
int main(int argc, char **argv) {

    srand(time(NULL));

    glutInit(&argc, argv);

    /*Setting up  The Display
    /    -RGB color model + Alpha Channel = GLUT_RGBA
    */
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);

    //Configure Window Postion
    glutInitWindowPosition(50, 25);

    //Configure Window Size
    glutInitWindowSize(WINDOW_W,WINDOW_H);

    //Create Window
    glutCreateWindow("NBody Simulation - Little Hamster and Sheep");

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable( GL_BLEND );


    //Call to the drawing function
    glutDisplayFunc(draw2);
    //glutTimerFunc(1000,timerFunc,10);

    glutKeyboardFunc(keyboardFunc);


    init();

    // Loop require by OpenGL
    glutMainLoop();


    return 0;
}

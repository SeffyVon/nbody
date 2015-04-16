#include "util.cuh"
#include "nbody.cuh"


extern Body bodies[N_SIZE];
float cx=-200,cy=-200,cz=-500;

void timerFunc(int value)
{
    glutPostRedisplay();
    //glutTimerFunc (1, timerFunc, 10);
    
}

void keyboardFunc(unsigned char key, int x, int y)
{
    if( ORTHO_VERSION )
        return;
    
    float vel = 10;
    if( key == 'w' )
    {
        cy+=vel;
    }
    if( key == 'a' )
    {
        cx-=vel;
    }
    if( key == 's' )
    {
        cy-=vel;
    }
    if( key == 'd' )
    {
        cx+=vel;
    }
    if( key == '+' )
    {
        cz+=vel;
    }
    if( key == '-' )
    {
        cz+=vel;
    }

    if( key == 'r')
    {
       cx= -200;
       cy = 100;
       cz = -500;
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(cx,cy,cz,0,0,0,0,1,0);
}
 
//utility functions definitions go here
void draw() {

    // Black background
    glClearColor(0.7f,0.7f,0.7f,0.7f);
    glClear(GL_COLOR_BUFFER_BIT);

    runKernelNBodySimulation();

 //  glColor3ub( 255, 0, 255 );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(Body), &bodies[0].pos.x );
    glColorPointer( 4, GL_FLOAT, sizeof(Body), &bodies[0].r );
    glPointSize( 5.0 );
    glDrawArrays( GL_POINTS, 0, N_SIZE );
    glDisableClientState( GL_VERTEX_ARRAY );
    glDisableClientState( GL_COLOR_ARRAY );

   	glutSwapBuffers();
    
    //Draw stuff
   // for(int i = 0; i < N_SIZE; i++){
	//	printf("HOHO a=(%f,%f,%f)\n", bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
	//}


   glutPostRedisplay();

}

void DrawCircle(float cx, float cy, float r, int num_segments) {
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++)   {
        float theta = 2.0f * PI * float(ii) / float(num_segments);//get the current angle 
        float x = r * cosf(theta);//calculate the x component 
        float y = r * sinf(theta);//calculate the y component 
        glVertex2f(x + cx, y + cy);//output vertex 
    }
    glEnd();
}


void draw2(){
    glClearColor(0.7f,0.7f,0.7f,0.7f);
    glClear(GL_COLOR_BUFFER_BIT);

    runKernelNBodySimulation();

    // glEnableClientState( GL_VERTEX_ARRAY );
    // glEnableClientState( GL_COLOR_ARRAY );

    // glVertexPointer( 3, GL_FLOAT, sizeof(Body), &bodies[0].pos.x );
    // glColorPointer( 4, GL_FLOAT, sizeof(Body), &bodies[0].r );

    // glPointSize( 5.0 );

    // glDrawArrays( GL_POINTS, 0, N_SIZE );

    // glDisableClientState( GL_VERTEX_ARRAY );
    // glDisableClientState( GL_COLOR_ARRAY );

    for(int i = 0; i < N_SIZE; i ++){
        if(bodies[i].alpha>0)
            DrawCircle(bodies[i].pos.x, bodies[i].pos.y, bodies[i].radius, 10);
    }

    glutSwapBuffers();


    glutPostRedisplay();

}





 

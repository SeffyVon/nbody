#include "util.cuh"
#include "nbody.cuh"

#include <string>

extern Body bodies[N_SIZE];
float prevX = WINDOW_W/2, prevY = WINDOW_H/2;
bool mouseUp = 0;

bool toggleHelp = true;

GLfloat lpos[4] = {-0.3,0.0,200,0}; //Positioned light
GLfloat light_specular[4] = {1, 0.6, 1, 0}; //specular light intensity (color)
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };//diffuse light intensity (color)
GLfloat light_ambient[] = { 0.2, 0.2, 0.2, 0.0 }; //ambient light intensity (color)
GLfloat a;
GLfloat mat_emission[] = {0.8, 0.5, 0.3, 0.0}; //object material preperty emission of light
GLfloat mat_specular[] = { 4.0, 0.5, 2.0, 0.0 }; //object material specularity
GLfloat low_shininess[] = { 50 };
GLfloat fogColor[] = {0.5f, 0.5f, 0.5f, 1};

void timerFunc(int value)
{
    glutPostRedisplay();
    //glutTimerFunc (1, timerFunc, 10);
    
}

void resizeCallback(int w, int h) {
    if( ORTHO_VERSION ) return;
    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if(h == 0)
        h = 1;

    float ratio = 1.0* w / h;

    // Reset the coordinate system before modifying
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    // Set the viewport to be the entire window
    glViewport(0, 0, w, h);

    // Set the correct perspective.
    gluPerspective(45,ratio,1,1000);
    glMatrixMode(GL_MODELVIEW);
    
}


void keyboardFunc(unsigned char key, int x, int y) {

    if (key == 27) 
        exit(0);

    float vel = 5.0;
    float rightX, rightY, rightZ;
    //printf("forwardX: %f forwardY: %f forwardZ: %f\n",camera.forwardX,camera.forwardY,camera.forwardZ);
    //printf("upX: %f upY: %f upZ: %f\n", camera.upX,camera.upY, camera.upZ);
    cross(camera.forwardX,camera.forwardY,camera.forwardZ, camera.upX,camera.upY, camera.upZ, rightX, rightY, rightZ);
    float sizeRight = sqrtf(rightX*rightX + rightY*rightY + rightZ*rightZ);
    rightX /= sizeRight; rightY /= sizeRight; rightZ /= sizeRight;
    //printf("rightX: %f rightY: %f rightZ: %f\n",rightX, rightY, rightZ);
    if( key == 'w' )
    {
        camera.camX += camera.forwardX*vel;
        camera.camY += camera.forwardY*vel;
        camera.camZ += camera.forwardZ*vel;
    }
    if( key == 's' )
    {
        camera.camX -= camera.forwardX*vel;
        camera.camY -= camera.forwardY*vel;
        camera.camZ -= camera.forwardZ*vel;
    }
    if( key == 'a' )
    {

        camera.camX -= rightX*vel;
        camera.camY -= rightY*vel;
        camera.camZ -= rightZ*vel;
    }
    if( key == 'd' )
    {
        camera.camX += rightX*vel;
        camera.camY += rightY*vel;
        camera.camZ += rightZ*vel;
    }
     printf("camX: %f camY: %f camZ: %f\n",camera.camX,camera.camY,camera.camY);


    if( key == 'h' )
    {
        toggleHelp = !toggleHelp;
    }


}
void PassiveMouseMotion( int x, int y ){
    prevX = x, prevY = y;

}
void mouseCallback(int x, int y){


        //camera.phi += (0.5 - (float(x)/WINDOW_W))*M_PI*0.015;
        //camera.theta += (0.5 - (float(y)/WINDOW_H))*M_PI*0.015;

        

        float velx = (float(x -prevX)/WINDOW_W);
        float vely = (float(y -prevY)/WINDOW_H);
        prevX = x;
        prevY = y;
        camera.phi += -velx*M_PI*0.9;
        camera.theta += -vely*M_PI*0.9;


        //printf("phi: %f theta: %f x: %d y: %d\n",camera.phi, camera.theta, x, y);

        float rightX, rightY, rightZ;
        rightX = sinf(camera.phi - M_PI/2.0f);
        rightY = 0;
        rightZ = cosf(camera.phi - M_PI/2.0f);
        float sizeRight = sqrtf(rightX*rightX + rightY*rightY + rightZ*rightZ);
        rightX /= sizeRight; rightY /= sizeRight; rightZ /= sizeRight;


        camera.forwardX = cosf(camera.theta)*sinf(camera.phi);
        camera.forwardY = sinf(camera.theta);
        camera.forwardZ = cosf(camera.theta)*cosf(camera.phi);
        float sizeForward = sqrtf(camera.forwardX*camera.forwardX + camera.forwardY*camera.forwardY + camera.forwardZ*camera.forwardZ);
        camera.forwardX /= sizeForward; camera.forwardY /= sizeForward; camera.forwardZ /= sizeForward;



        //printf("%f %f %f\n",camera.forwardX, camera.forwardY, camera.forwardZ);

       

        float newUpX, newUpY, newUpZ;

        cross(rightX,rightY,rightZ, camera.forwardX,camera.forwardY, camera.forwardZ, newUpX, newUpY, newUpZ);
        float sizeUp = sqrtf(newUpX*newUpX + newUpY*newUpY + newUpZ*newUpZ);
        camera.upX = newUpX/sizeUp; camera.upY = newUpY/sizeUp; camera.upZ = newUpZ/sizeUp;
        


}

void cross(float x1, float y1, float z1, float x2, float y2, float z2,float& rightX, float& rightY, float& rightZ){
    rightX = y1*z2 - z1*y2;
    rightY = x1*z2 - x1*z2;
    rightZ = x1*y2 - y1*x1;

}

 
//utility functions definitions go here
void draw() {

    // Black background
    glClearColor(0.7f,0.7f,0.7f,0.7f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(camera.camX,camera.camY,camera.camZ, //Camera position
              camera.camX+camera.forwardX,camera.camY+camera.forwardY,camera.camZ+camera.forwardZ, //Position of the object to look at
              camera.upX,camera.upY,camera.upZ); //Camera up direction

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

void drawText(std::string text, float x, float y){
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    //glLoadIdentity();

    glColor3f(1.0f, 0.0f, 0.0f);//needs to be called before RasterPos
    glRasterPos2f(x, y);
    //glScalef(10,10,10);
    
    void * font = GLUT_BITMAP_TIMES_ROMAN_24;

    for (std::string::iterator i = text.begin(); i != text.end(); ++i)
    {
        char c = *i;
        //this does nothing, color is fixed for Bitmaps when calling glRasterPos
        //glColor3f(1.0, 0.0, 1.0); 
        glutBitmapCharacter(font, c);
    }
    glPopMatrix();
}


void setLights(){

    //glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_ambient);
    glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, low_shininess);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, light_diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, lpos);


    //Adding fog
    glFogfv(GL_FOG_COLOR, fogColor);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 10.0f);
    glFogf(GL_FOG_END, 1000.0f);
}

void draw2(){
    glClearColor(0.3f,0.3f,0.3f,0.3f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if( !ORTHO_VERSION ) {
        gluLookAt(camera.camX,camera.camY,camera.camZ, //Camera position
              camera.camX+camera.forwardX,camera.camY+camera.forwardY,camera.camZ+camera.forwardZ, //Position of the object to look at
              camera.upX,camera.upY,camera.upZ); //Camera up direction
    }
    setLights();

    runKernelNBodySimulation();

    if( toggleHelp ){
        if( !ORTHO_VERSION )
        {
            drawText("USAGE INFO", 50,60);
            drawText("Use keys w, a, s, d to move", 50,50);
            drawText("Hold mouse+left button to look around", 50,40);
            drawText("Press h to show/hide this help info", 50,30);
        }
        else
        {
            drawText("USAGE INFO", 50,80);
            drawText("Use keys w, a, s, d to move", 50,60);
            drawText("Hold mouse+left button to look around", 50,40);
            drawText("Press h to show/hide this help info", 50,10);
        }
    }
    

    glColor3f(0.5f, 0.5f, 0.3f);
    for(int i = 0; i < N_SIZE; i ++){
        if(bodies[i].alpha>0)
        {
            if( !ORTHO_VERSION ) 
            {
                glPushMatrix();
                glTranslatef(bodies[i].pos.x, bodies[i].pos.y,bodies[i].pos.z);
                glutSolidSphere(bodies[i].radius,10,10);
                glPopMatrix();
            }
            else{

                DrawCircle(bodies[i].pos.x, bodies[i].pos.y, bodies[i].radius, 10);
            }
        }
            
            
    }

    glutSwapBuffers();


}





 

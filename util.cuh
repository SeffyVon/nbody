#ifndef _UTIL_
#define _UTIL_


//utility functions declarations go here

// defined the key board function
void keyboardFunc(unsigned char key, int x, int y);

// defined call back function triggered by mouse
void mouseCallback(int x, int y);

// defined callback function for resizing the view
void resizeCallback(int w, int h);

// for the last mouse motion
void PassiveMouseMotion( int x, int y );

void timerFunc(int value);

void draw2(void);

void cross(float x1, float y1, float z1, float x2, float y2, float z2,float& rightX, float& rightY, float& rightZ);


#endif
#ifndef _UTIL_
#define _UTIL_



void keyboardFunc(unsigned char key, int x, int y);
void mouseCallback(int x, int y);
void resizeCallback(int w, int h);
void PassiveMouseMotion( int x, int y );

void timerFunc(int value);
//utility functions declarations go here
void draw(void);

void draw2(void);

void cross(float x1, float y1, float z1, float x2, float y2, float z2,float& rightX, float& rightY, float& rightZ);


#endif
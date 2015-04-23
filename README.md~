About the program

- In the beginning, our galaxy has 10,000 static stars. Each star is influenced by the gravitational force of other 9,999 stars. The gravitational force between the stars causes the acceleration on each star. The stars gradually moved to the centre, where lies the central mass of the galaxy.

- When two stars collide, a perfectly inelastic collision will happen. The two stars emerged, and the mass and velocity of the two stars will be changed, according to the energy and momentum conservation laws in physics.

- At the end, there will be one big star in the middle, with some other small stars orbiting around.

This program is not the same as other N-body simulation program. The main difference is the collision handling. The other N-body simulation programs usually avoid the situation of collision.

==================================================================================================================

Usage

- Left click+dragging the mouse to look around.
- Press keys W, S, A, D to move the camera forwards, backwards, left and right.
- Press key H for hiding/showing the info.

==================================================================================================================

Link to the screen-cast

https://vimeo.com/125337768 (password: balulun)
Better to turn on the volume and the subtitles for better viewing experience! :)

==================================================================================================================

How to run 

- run "module load CUDA/4.0.17" in the terminal
- extract the packet
- go to the nbody directory
- run "mkdir build" in the nbody directory
- run "cd build"  
- run "cmake .." in the nbody/build directory (remember to load CUDA first)
- run "make" in the nbody/build directory
- run "./main" for running the program

==================================================================================================================

Optimization:

- Changed the BLOCKSIZE from 1024 to 256 for better performance.

- Used as little code divergence as possible. In the program there are only two unavoidable code divergence:
	- Different effects depend on whether two bodies collide or not.
	- Different behaviors of heavier body and lighter body when they collide.

- Changed the array of structs Body to normal arrays of velocity, position, mass, acceleration and radius; with each entry refers to the attribute of one body.

- The effect of loop-unrolling is not obvious, so the program did not use it.

- The program also tried to use shared memory. However, by considering the case that we needs to change the mass and velocity of the bodies when the bodies collide, we excluded this approach. Because the shared memory will not be updated when some entries in global memory are changed, the threads in a block will reference the out-dated content in the shared memory. 

===================================================================================================================

Benchmark (The program can be seen in see benchmark.cu, running "build/benchmark" to run it)

No unroll
Run 10 kernels.
Elapsed time : 3576.995605 ms
Throughput: 0.279564 GFLOPS

#pragma unroll 4
Run 10 kernels.
Elapsed time : 3576.622314 ms
Throughput: 0.279593 GFLOPS

#pragma unroll 8
Run 10 kernels.
Elapsed time : 3574.892822 ms
Throughput: 0.279729 GFLOPS

Loop unrolling only has slightly increase the efficiency.

==============================================================
BLOCK SIZE

BLOCK_SIZE = 1024
Run 10 kernels.
Elapsed time : 3574.892822 ms
Throughput: 0.279729 GFLOPS

BLOCK_SIZE = 512
Run 10 kernels.
Elapsed time : 3567.016113 ms
Throughput: 0.280346 GFLOPS

BLOCK_SIZE = 256
Run 10 kernels.
Elapsed time : 3556.120850 ms
Throughput: 0.281205 GFLOPS

Changing BLOCK SIZE to 256 has slightly increase the efficiency.

==============================================================

(A)Array of structs compared to (B)Normal Array

ITERATION = 10

(A)
Run 10 kernels.
Elapsed time : 3556.120850 ms
Throughput: 0.281205 GFLOPS

(B)
Run 10 kernels.
Elapsed time : 3837.648193 ms
Throughput: 0.260576 GFLOPS

------------------------------

(A)
Run 100 kernels.
Elapsed time : 34097.648438 ms
Throughput: 0.293275 GFLOPS

(B)
Running 100 kernels
Elapsed time : 34019.390625 ms
Throughput: 0.293950 GFLOPS

The speed of the (B) normal array also slighly increased on (A) the array of structs. 



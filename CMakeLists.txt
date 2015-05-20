# CMakeLists.txt for G4CU project                                                                                                                                                                          
project(test-cuda-thrust-gdb)

# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA REQUIRED)

#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
cuda_include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)
#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
cuda_include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

#########################################################
# FIND GLEW
#########################################################
#find_package(GLEW REQUIRED)
#include_directories(${GLEW_INCLUDE_DIRS})
#link_directories(${GLEW_LIBRARY_DIRS})
#add_definitions(${GLEW_DEFINITIONS})
#if(NOT GLEW_FOUND)
#    message(ERROR " OPENGL not found!")
#endif(NOT GLEW_FOUND)

set(GENCODE_SM10 -gencode arch=compute_10,code=sm_10 -gencode arch=compute_11,code=sm_11)
set(GENCODE_SM13 -gencode arch=compute_13,code=sm_13)
set(GENCODE_SM20 -gencode arch=compute_20,code=sm_20)
set(GENCODE_SM30 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)

set(GENCODE_FLAGS ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${GENCODE_SM30})

# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${GENCODE_FLAGS})


set(INC util.cuh
		nbody.cuh
)
set(SRC util.cu
		nbody.cu
)


cuda_add_executable(main cuda_main.cu ${SRC} ${INC} )

target_link_libraries(main ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

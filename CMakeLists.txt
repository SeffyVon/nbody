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

# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
#set(CUDA_CUT_INCLUDE_DIR ${CUDA_CUT_INCLUDE_DIR} /bham/pd/packages/EL6/x86_64/cuda-4.0.17-sdk/C/common)
#set(CUDA_CUT_TARGET_LINK ${CUDA_CUT_TARGET_LINK}:)

set(INC util.cuh
		nbody.cuh
)
set(SRC util.cu
		nbody.cu
)

set(BENINC util.cuh
		   nbody.cuh
)

cuda_add_executable(main cuda_main.cu ${SRC} ${INC} )
cuda_add_executable(benchmark benchmark.cu ${SRC} ${BENINC})

target_link_libraries(main ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
target_link_libraries(benchmark ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
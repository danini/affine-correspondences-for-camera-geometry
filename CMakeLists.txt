cmake_minimum_required(VERSION 3.8)

project(AffineGuideline LANGUAGES C CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if OPENMP should be enabled
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

SET (CMAKE_BUILD_TYPE "RELEASE")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-fpermissive")

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV)
if (OPENCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
  	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (OPENCV_FOUND)

# ==============================================================================
# Find OpenMP
# ==============================================================================
FIND_PACKAGE(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (USE_OPENMP)


# ==============================================================================
# Find Eigen
# ==============================================================================
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# ==============================================================================
# VLFeat
# ==============================================================================
add_subdirectory(lib)

# ==============================================================================
# Includes
# ==============================================================================
SET(SOURCE_DIR "src/pyrobustac")

include_directories (
	${PROJECT_SOURCE_DIR}/lib/graph-cut-ransac/src/pygcransac/include
	${PROJECT_SOURCE_DIR}/lib
)

# ==============================================================================
# Structure: Graph-Cut RANSAC Library
# ==============================================================================
# Set header files for the library
file(GLOB_RECURSE HDRS_GraphCutRANSAC
	"lib/graph-cut-ransac/src/pygcransac/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GraphCutRANSAC 
	"lib/graph-cut-ransac/src/pygcransac/include/*.cpp"
)

add_library(GraphCutRANSAC STATIC
	${HDRS_GraphCutRANSAC}
	${SRCS_GraphCutRANSAC}
)

target_link_libraries(GraphCutRANSAC
	${OpenCV_LIBS}
	Eigen3::Eigen
)

# ==============================================================================
# Structure: Affine Guideline Library
# ==============================================================================
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR}/include)

# Set header files for the library
file(GLOB_RECURSE HDRS_AffineGuideline
	"${SOURCE_DIR}/include/*.h"
) 

# Set header files for the library
file(GLOB_RECURSE SRCS_AffineGuideline
	"${SOURCE_DIR}/include/*.cpp"
) 

# ==============================================================================
# Structure: Python binding
# ==============================================================================
# Generate python module
add_subdirectory(lib/graph-cut-ransac/lib/pybind11)

pybind11_add_module(pyrobustac 
	${SOURCE_DIR}/src/bindings.cpp 
	${SOURCE_DIR}/src/robustac_python.cpp 
	${HDRS_AffineGuideline} 
	${SRCS_AffineGuideline})

target_link_libraries(pyrobustac PRIVATE  
	${OpenCV_LIBS} 
	Eigen3::Eigen 
	vlfeat
	GraphCutRANSAC)
	
add_library(${PROJECT_NAME} STATIC
	${HDRS_AffineGuideline}
	${SRCS_AffineGuideline}
)

target_link_libraries (${PROJECT_NAME} 
	${OpenCV_LIBS}
	Eigen3::Eigen
	vlfeat
	GraphCutRANSAC)
	
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)

# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(SampleProject
		examples/cpp_example.cpp)
				
	target_link_libraries(SampleProject PUBLIC
		${OpenCV_LIBS}
		Eigen3::Eigen
		vlfeat
		GraphCutRANSAC
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)
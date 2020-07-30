cmake_minimum_required(VERSION 3.8)

project(AffineGuideline LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

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
if (OPENMP_FOUND)
# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (OPENMP_FOUND)

SET (CMAKE_BUILD_TYPE "RELEASE")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# ==============================================================================
# Find Eigen
# ==============================================================================
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# ==============================================================================
# Includes
# ==============================================================================
include_directories (
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/lib/magsac/graph-cut-ransac/src/pygcransac/include
	${PROJECT_SOURCE_DIR}/lib/magsac/include
)

# ==============================================================================
# Structure: Graph-Cut RANSAC Library
# ==============================================================================
# Set header files for the library
file(GLOB_RECURSE HDRS_GraphCutRANSAC
	"lib/magsac/graph-cut-ransac/src/pygcransac/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GraphCutRANSAC 
	"lib/magsac/graph-cut-ransac/src/pygcransac/include/*.cpp"
)

add_library(GraphCutRANSAC STATIC
	${HDRS_GraphCutRANSAC}
	${SRCS_GraphCutRANSAC}
)

target_link_libraries(GraphCutRANSAC
	${OpenCV_LIBS}
	Eigen3::Eigen)

# ==============================================================================
# Structure: MAGSAC Library
# ==============================================================================
# Set header files for the library
file(GLOB_RECURSE HDRS_MAGSAC
	"lib/magsac/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_MAGSAC 
	"lib/magsac/include/*.cpp"
)

add_library(MAGSAC STATIC
	${HDRS_MAGSAC}
	${SRCS_MAGSAC}
)

target_link_libraries(MAGSAC
	${OpenCV_LIBS}
	Eigen3::Eigen)

# ==============================================================================
# Structure: Affine Guideline Library
# ==============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Set header files for the library
file(GLOB_RECURSE HDRS_AffineGuideline
	"include/*.h"
	"include/*.cpp"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_AffineGuideline 
	"src/*.cpp"
)

add_executable(${PROJECT_NAME}  
	${HDRS_AffineGuideline} 
	${SRCS_AffineGuideline}
)

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
	Eigen3::Eigen
)

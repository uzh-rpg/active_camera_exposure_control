SET(PROJECT_NAME photometric_camera)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(${PROJECT_NAME})

add_definitions(-DUSE_DOUBLE)

SET(CMAKE_BUILD_TYPE RelWithDebugInfo) # Release, RelWithDebInfo
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++11")

IF(DEFINED ENV{ARM_ARCHITECTURE})
  message("ARM")
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfpu=vfpv3" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon -mfpu=vfpv3" )
ENDIF()

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenCV REQUIRED)

cs_add_library(${PROJECT_NAME} SHARED
  src/photometric_model.cpp
  src/photometric_dataset.cpp
  include/photometric_camera/photometric_model.h
  include/photometric_camera/photometric_dataset.h
  )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

cs_add_executable(test_photometric_model tests/test_photometric_model.cpp)
target_link_libraries(test_photometric_model
    ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
)

cs_export()

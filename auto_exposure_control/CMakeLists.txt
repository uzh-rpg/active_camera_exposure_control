SET(PROJECT_NAME auto_exposure_control)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(${PROJECT_NAME})

#add_definitions(-DDEBUG_OUTPUT)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++11")

IF(DEFINED ENV{ARM_ARCHITECTURE})
  message("ARM")
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfpu=vfpv3" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon -mfpu=vfpv3" )
ENDIF()

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME} SHARED
  src/auto_exposure_percentile.cpp
  src/auto_exposure_sweep.cpp
  src/auto_exposure_shim.cpp
  src/auto_exposure_intensity.cpp
  src/auto_exp_utils.cpp
  src/auto_camera_settings.cpp
  src/auto_camera_settings_factory.cpp
  src/frame.cpp
  include/auto_exposure_control/auto_exposure_percentile.h
  include/auto_exposure_control/auto_exposure_sweep.h
  include/auto_exposure_control/auto_exposure_shim.h
  include/auto_exposure_control/auto_exposure_intensity.h
  include/auto_exposure_control/auto_exp_utils.h
  include/auto_exposure_control/fixed_size_buffer.h
  include/auto_exposure_control/auto_camera_settings.h
  include/auto_exposure_control/auto_camera_settings_factory.h
  include/auto_exposure_control/frame.h
  include/vikit/timer.h
  include/vikit/params_helper.h
)

cs_export()

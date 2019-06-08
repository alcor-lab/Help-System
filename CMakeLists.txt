
cmake_minimum_required(VERSION 3.1.3)
project(Help-System)
enable_testing()

# ============================== ProjectCMake =================================

set(PCM_DIR ${PROJECT_SOURCE_DIR}/Dependencies/ProjectTools/Libraries/ProjectCMake)

if(NOT EXISTS ${PCM_DIR})
  set(PCM_DIR ${PROJECT_SOURCE_DIR}/Dependencies/ProjectTools/Outputs/Libraries/ProjectCMake)
endif()

include("${PCM_DIR}/Main.cmake")

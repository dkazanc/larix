#   Copyright 2018 Edoardo Pasca
#cmake_minimum_required (VERSION 3.0)

project(i23seg_core)

# The version number.
set (i23seg_VERSION $ENV{i23seg_VERSION} CACHE INTERNAL "i23 segmentation module" FORCE)

# conda orchestrated build
message("i23seg_VERSION ${i23seg_VERSION}")
#include (GenerateExportHeader)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   
endif()

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message("CMAKE_C_FLAGS ${CMAKE_C_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

if(WIN32)
  set (FLAGS "/DWIN32 /EHsc /openmp")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
  set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  
  set (EXTRA_LIBRARIES)
		
  message("library lib: ${LIBRARY_LIB}")
  
elseif(UNIX)
   set (FLAGS "-O2 -funsigned-char -Wall  -Wl,--no-undefined")  
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
   set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
  
   set (EXTRA_LIBRARIES 
		"gomp"
		"m"
		)
   
endif()
message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

message("Adding i23 segmentation as a shared library")

add_library(i23seg SHARED
    	    ${CMAKE_CURRENT_SOURCE_DIR}/functions_CPU/MASK_merge_core.c
	    )
target_link_libraries(i23seg ${EXTRA_LIBRARIES} )
include_directories(i23seg PUBLIC 
                      ${LIBRARY_INC}/include 
					  ${CMAKE_CURRENT_SOURCE_DIR}
		              ${CMAKE_CURRENT_SOURCE_DIR}/functions_CPU/ )

## Install
if (UNIX)
message ("I'd install into ${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS i23seg
	LIBRARY DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)
elseif(WIN32)
message ("I'd install into ${CMAKE_INSTALL_PREFIX} lib bin")
  install(TARGETS i23seg 
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)
endif()

#cmake_minimum_required (VERSION 3.0)

#project(larix_core)

project(larix_core LANGUAGES CXX CUDA)
include(CheckLanguage)
check_language(CUDA)

# The version number.
set (larix_VERSION $ENV{larix_VERSION} CACHE INTERNAL "Larix is an image processing library" FORCE)

# conda orchestrated build
message("larix_VERSION ${larix_VERSION}")
#include (GenerateExportHeader)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message("Creating Larix as a shared library")

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message("CMAKE_C_FLAGS ${CMAKE_C_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

if(WIN32)
  set (FLAGS "/DWIN32 /EHsc /DDLSCore_EXPORTS /openmp")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  message("library lib: ${LIBRARY_LIB}")
elseif(UNIX)
   set (FLAGS "-O2 -funsigned-char -Wall  -Wl,--no-undefined")
   set(EXTRA_LIBRARIES "m")
endif()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
  set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

message("Adding Larix software as a shared library")

add_library(larix SHARED
    	    ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/MASK_merge_core.c
    	    ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/MaskIterative_core.c
    	    ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/autocropper_core.c
          ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/MedianFilt_core.c
          ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/NonlocalMarching_Inpaint_core.c
          ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/Inpaint_simple_core.c
          ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/Diffusion_Inpaint_core.c
    	    ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/utils.c
	    )
target_link_libraries(larix ${EXTRA_LIBRARIES} -static-libgcc -static-libstdc++)
include_directories(larix PUBLIC
                      ${LIBRARY_INC}/include
					  ${CMAKE_CURRENT_SOURCE_DIR}
		              ${CMAKE_CURRENT_SOURCE_DIR}/CPU_modules/)

## Install
if (UNIX)
message ("I'd install into ${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS larix
	LIBRARY DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE}
	)
elseif(WIN32)
message ("I'd install into ${CMAKE_INSTALL_PREFIX} lib bin")
  install(TARGETS larix
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE}
	)
endif()

if (BUILD_CUDA)
    #find_package(CUDA)
    #if (CUDA_FOUND)
      #set(CUDA_NVCC_FLAGS "-Xcompiler -fPIC -shared -D_FORCE_INLINES -arch=sm_35 -rdc=true")
      #message("CUDA FLAGS ${CUDA_NVCC_FLAGS}")
      ADD_LIBRARY(larixcuda SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/GPU_modules/MedianFilt_shared_GPU_core.cu)
      #CUDA_ADD_LIBRARY(larixcuda SHARED
      #  ${CMAKE_CURRENT_SOURCE_DIR}/GPU_modules/MedianFilt_shared_GPU_core.cu
      #)
      if (UNIX)
        message ("I'd install into ${CMAKE_INSTALL_PREFIX}/lib")
        install(TARGETS larixcuda
        LIBRARY DESTINATION lib
        CONFIGURATIONS ${CMAKE_BUILD_TYPE}
        )
      elseif(WIN32)
        message ("I'd install into ${CMAKE_INSTALL_PREFIX} lib bin")
        install(TARGETS larixcuda
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        CONFIGURATIONS ${CMAKE_BUILD_TYPE}
        )
      endif()
    #else()
    #  message("CUDA NOT FOUND")
    #endif()
endif()

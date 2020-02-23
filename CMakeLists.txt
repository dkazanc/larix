#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

cmake_minimum_required (VERSION 3.0)

project(COLLEDA)

set (COLLEDA_VERSION $ENV{COLLEDA_VERSION} CACHE INTERNAL "Colleda software" FORCE)
message(STATUS "COLLEDA_VERSION set to " ${COLLEDA_VERSION})

# set the Python variables for the Conda environment
option (BUILD_PYTHON_WRAPPER "Build Python Wrappers" ON)
option (CONDA_BUILD "Conda Build" OFF)

set(PYTHON_DEST_DIR "" CACHE PATH "Directory of the Python wrappers")
if (PYTHON_DEST_DIR)
 set(PYTHON_DEST "${PYTHON_DEST_DIR}")
else()
 set(PYTHON_DEST "${CMAKE_INSTALL_PREFIX}/python")
endif()
message(STATUS "Python wrappers will be installed in " ${PYTHON_DEST})

add_subdirectory(src)

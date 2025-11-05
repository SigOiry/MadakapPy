# - Config file for the Shark package
# It defines the following variables
#  SHARK_INCLUDE_DIRS - include directories for SHARK
#  SHARK_LIBRARIES    - libraries to link against
#  SHARK_LIBRARY_DIRS - path to the libraries

	# Compute the installation prefix from this SharkConfig.cmake file location.
	get_filename_component(SHARK_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
		get_filename_component(SHARK_INSTALL_PREFIX "${SHARK_INSTALL_PREFIX}" DIRECTORY)
		get_filename_component(SHARK_INSTALL_PREFIX "${SHARK_INSTALL_PREFIX}" DIRECTORY)
		get_filename_component(SHARK_INSTALL_PREFIX "${SHARK_INSTALL_PREFIX}" DIRECTORY)
 
set(SHARK_INCLUDE_DIRS "${SHARK_INSTALL_PREFIX}/include/shark/;C:/build/otb/xdk/include/boost-1_82")
set(SHARK_LIBRARY_DIRS "${SHARK_INSTALL_PREFIX}/lib")
 
# Our library dependencies (contains definitions for IMPORTED targets)
include("${SHARK_INSTALL_PREFIX}/lib/cmake/Shark/SharkTargets.cmake")
 
# The Shark version number
set(SHARK_VERSION_MAJOR "3")
set(SHARK_VERSION_MINOR "1")
set(SHARK_VERSION_PATCH "0")

# The C and C++ flags added by Shark to the cmake-configured flags.
SET(SHARK_REQUIRED_C_FLAGS "")
SET(SHARK_REQUIRED_CXX_FLAGS "")
SET(SHARK_REQUIRED_EXE_LINKER_FLAGS "")
SET(SHARK_REQUIRED_SHARED_LINKER_FLAGS "")
SET(SHARK_REQUIRED_MODULE_LINKER_FLAGS "")

# The location of the UseShark.cmake file.
SET(SHARK_USE_FILE "${SHARK_INSTALL_PREFIX}/lib/cmake/Shark/UseShark.cmake")

set(SHARK_LIBRARIES "Boost::serialization;Boost::filesystem;Boost::system;shark")

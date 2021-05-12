cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(MPI)
enable_language(C)
enable_language(CXX)

find_package(MPI)

if(NOT MPI_C_FOUND AND NOT MPI_CXX_FOUND)
  message(ERROR "MPI not found. Install MPI first or load respective modules.")
endif()

configure_file(config.py.in ${EXTERNAL_INSTALL_LOCATION}/config.py)

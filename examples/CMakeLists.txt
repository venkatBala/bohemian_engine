cmake_minimum_required(VERSION 3.10)
project(bhimecpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

find_package(LAPACK REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)
find_package(PNG REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIRS})

add_executable(bhimecpp src/main.cc)
target_link_libraries(bhimecpp Eigen3::Eigen ${LAPACK_LIBRARIES}
                      ${GSL_LIBRARIES} ${PNG_LIBRARIES})

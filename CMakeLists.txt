CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT("cppporting")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(my_utils)

link_libraries(pthread)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(PROJECT_VERSION_MAJOR 0)
SET(PROJECT_VERSION_MINOR 1)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} [OpenCV.cmake PATH])


#SET(OUTPUT_ELF "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.out")
#SET(SRC_FILES server.cpp client.cpp)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(server server.cpp)
ADD_EXECUTABLE(client client.cpp)

include_directories(${OpenCV_INCLUDE_DIR})

target_link_libraries(server "${TORCH_LIBRARIES}" my_utils)
target_link_libraries(client "${TORCH_LIBRARIES}" my_utils ${OpenCV_LIBRARIES})

target_include_directories(server PUBLIC ${CMAKE_SOURCE_DIR}/my_utils)

target_compile_features(server PRIVATE cxx_std_17)
#add_compile_options(server PUBLIC -Wall)
#set_property(TARGET server PROPERTY CXX_STANDARD 17)
target_compile_features(client PRIVATE cxx_std_17)
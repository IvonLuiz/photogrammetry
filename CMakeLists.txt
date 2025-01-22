#cmake_minimum_required(VERSION 2.8)
#project(photogrammetry)
#find_package( OpenCV REQUIRED )
#add_executable( lerImagem lerImagem.cpp )
#target_link_libraries( lerImagem ${OpenCV_LIBS} )
#

cmake_minimum_required(VERSION 2.8)

project(photogrammetry)

find_package(OpenCV REQUIRED)

add_executable(extractKeypoints extractKeypoints.cpp)

target_link_libraries(extractKeypoints ${OpenCV_LIBS})

cmake_minimum_required(VERSION 2.8)
project(photogrammetry)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Library
add_library(feature_extractor featureExtractor.cpp)

# executables 
# add_executable(extractKeypointsDemo featureExtractor.cpp)
add_executable(anmsOrb adaptiveNonMaxSuppressionORB.cpp)
add_executable(featureMatcher featureMatcher.cpp)


# LInk library and dependencies
#target_link_libraries(extractKeypointsDemo feature_extractor ${OpenCV_LIBS})
target_link_libraries(anmsOrb ${OpenCV_LIBS})
target_link_libraries(feature_extractor ${OpenCV_LIBS})
target_link_libraries(featureMatcher feature_extractor ${OpenCV_LIBS})

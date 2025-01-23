#include <iostream>
#include <opencv2/opencv.hpp>

#include "featureExtractor.h"


int main(int argc, char* argv[])
{    
    std::string boxImgPath = "box.png";
    std::string envImgPath = "box_in_scene.png";
    FeatureExtractor extractor;

    std::vector<cv::KeyPoint> keypoints_box, keypoints_env;
    cv::Mat descriptors_box, descriptors_env;

    extractor.extract(boxImgPath, &keypoints_box, &descriptors_box, "ORB", true);  // Passando ponteiro para keypoints_box
    extractor.extract(envImgPath, &keypoints_env, &descriptors_env, "ORB", true);  // Passando ponteiro para keypoints_env
}
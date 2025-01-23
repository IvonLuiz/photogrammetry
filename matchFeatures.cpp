#include <iostream>
#include <opencv2/opencv.hpp>

#include "featureExtractor.h"


int main(int argc, char* argv[])
{    
    std::string boxImgPath = "box.png";
    std::string envImgPath = "box_in_scene.png";
    cv::Mat boxImg = cv::imread(boxImgPath, cv::IMREAD_GRAYSCALE);
    cv::Mat envImg = cv::imread(envImgPath, cv::IMREAD_GRAYSCALE);

    //-- Step 1: Detect the keypoints using Feature Extractor Class, compute the descriptors
    FeatureExtractor extractor;

    std::vector<cv::KeyPoint> keypoints_box, keypoints_env;
    cv::Mat descriptors_box, descriptors_env;

    extractor.extract(boxImgPath, &keypoints_box, &descriptors_box, "ORB");
    extractor.extract(envImgPath, &keypoints_env, &descriptors_env, "ORB");

    //-- Step 2: Matching descriptor vectors with a brute force matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_box, descriptors_env, matches);

    // Show matches
    cv::Mat img_matches;
    cv::drawMatches(boxImg, keypoints_box, envImg, keypoints_env, matches, img_matches);

    cv::imshow("Matches", img_matches);

    cv::waitKey();
    return 0;

}
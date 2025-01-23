#ifndef EXTRACT_KEYPOINTS_H
#define EXTRACT_KEYPOINTS_H

#include <iostream>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

class FeatureExtractor {
public:
    FeatureExtractor();
    void extract(const std::string& imgPath,
                 std::vector<cv::KeyPoint>* keypoints = nullptr,
                 cv::Mat* descriptors = nullptr,
                 const std::string featureType = "ORB",
                 bool plot = true);

private:
    // Replace Condition Dispatcher with Command using hashmap (to substitute if else logic)
    std::unordered_map<std::string, std::function<cv::Ptr<cv::Feature2D>()>> dispatcher;
    
    void checkFeatureExtractor(const std::string& featureType);
    cv::Mat getImg(const std::string& imgPath);
    void showImage(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const std::string& featureType);
};

#endif // EXTRACT_KEYPOINTS_H

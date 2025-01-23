#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

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
    void extract(const std::string& imgPath,
                 const std::string featureType = "ORB",
                 bool plot = true);
                 
private:
    // Replace Condition Dispatcher with Command using hashmap (to substitute if else logic)
    std::unordered_map<std::string, std::function<cv::Ptr<cv::Feature2D>()>> dispatcher;
    
    void checkFeatureExtractor(const std::string& featureType);
    cv::Mat getImg(const std::string& imgPath);
    void showImage(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const std::string& featureType);
};

#endif // FEATURE_EXTRACTOR_H

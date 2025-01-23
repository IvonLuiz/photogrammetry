#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

class FeatureMatcher {
public:
    FeatureMatcher();

    std::vector<cv::DMatch> matchFeatures(const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& descriptors1,
                       const std::vector<cv::KeyPoint>& keypoints2, const cv::Mat& descriptors2,
                       const std::string& featureType);

private:
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
};

#endif // FEATURE_MATCHER_H
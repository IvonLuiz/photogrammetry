#include <iostream>
#include <unordered_map>
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

// using namespace std;
// using namespace cv;


class FeatureExtractor {
public:
    // Mapa de dispatcher: associa o nome do algoritmo à função criadora correspondente
    std::unordered_map<std::string, std::function<cv::Ptr<cv::Feature2D>()>> dispatcher; 
    
    // Constructor
    FeatureExtractor()
    {
        dispatcher["ORB"] = []() { return cv::ORB::create(); };
        dispatcher["SIFT"] = []() { return cv::SIFT::create(); };
        dispatcher["SURF"] = []() { return cv::xfeatures2d::SURF::create(); };
        //dispatcher[""]

    }

    void extract(const std::string& imgPath, const std::string& featureType)
    {
        checkFeatureExtractor(featureType);
        cv::Mat img = getImg(imgPath);

        auto detector = dispatcher[featureType]();
        std::vector<cv::KeyPoint> keypoints;
        detector->detect( img, keypoints );

        std::cout << "Detected " << keypoints.size() << " keypoints using " << featureType << "." << std::endl;
    }


    void showImage()
    {

    }

private:

    void checkFeatureExtractor(std::string featureType)
    {
        if (dispatcher.find(featureType) == dispatcher.end()) {
            throw std::invalid_argument("Feature type not supported: " + featureType);
        }
    }

    cv::Mat getImg(std::string imgPath)
    {
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

        if (img.empty()) {
            throw std::runtime_error("Could not open or find the image!");
        }

        return img;
    }
};


int main(int argc, char* argv[]){
    std::string imgPath = "image.jpg";
    
    FeatureExtractor extractor;
    extractor.extract(imgPath, "SIFT");
    extractor.extract(imgPath, "SURF");
    extractor.extract(imgPath, "ORB");

    return 0;
}
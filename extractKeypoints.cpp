#include <iostream>
#include <unordered_map>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

class FeatureExtractor {
public:
    // Replace Condition Dispatcher with Command (to substitute if else logic)
    std::unordered_map<std::string, std::function<cv::Ptr<cv::Feature2D>()>> dispatcher;

    FeatureExtractor()
    {
        dispatcher["ORB"] = []() { return cv::ORB::create(); };
        dispatcher["SIFT"] = []() { return cv::SIFT::create(); };
        dispatcher["SURF"] = []() { return cv::xfeatures2d::SURF::create(); };
        dispatcher["AKAZE"] = []() { return cv::AKAZE::create(); };
        dispatcher["BRISK"] = []() { return cv::BRISK::create(); };
        dispatcher["KAZE"] = []() { return cv::KAZE::create(); };
    }

    void extract(const std::string& imgPath,
                 std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                 const std::string& featureType = "ORB",
                 bool plot=false)
    {
        // Checks if feature extractor is valid
        checkFeatureExtractor(featureType);
        cv::Mat img = getImg(imgPath);

        auto detector = dispatcher[featureType]();

        // Extract features and descriptors and stop keep track of timer
        auto start = std::chrono::high_resolution_clock::now();

        detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        int scaledDuration = duration_ms.count();

        // Results
        std::cout << "Detected " << keypoints.size() << " keypoints and descriptors using " << featureType << ".\n";
        std::cout << "Descriptors size: " << descriptors.rows << "x" << descriptors.cols << "\n";
        std::cout << "Time taken for " << featureType << ": " << scaledDuration << " milliseconds." << std::endl;
        
        if (plot == true)
            showImage(img, keypoints, featureType);
    }

    // Implementation to not require keypoints and descriptors as arguments
    void extract(const std::string& imgPath, const std::string& featureType = "ORB", bool plot = false) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        extract(imgPath, keypoints, descriptors, featureType, plot);
    }

    void showImage(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const std::string& featureType)
    {
        cv::Mat img_keypoints;
        cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        
        cv::imshow("Keypoints - " + featureType, img_keypoints);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

private:
    void checkFeatureExtractor(const std::string& featureType)
    {
        if (dispatcher.find(featureType) == dispatcher.end())
            throw std::invalid_argument("Feature type not supported: " + featureType);
    }

    cv::Mat getImg(const std::string& imgPath)
    {
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

        if (img.empty())
            throw std::runtime_error("Could not open or find the image at path: " + imgPath);
        
        return img;
    }
};

int main(int argc, char* argv[]) {
    std::string imgPath = "image.png";
    FeatureExtractor extractor;

    extractor.extract(imgPath, "SIFT");
    extractor.extract(imgPath, "SURF");
    extractor.extract(imgPath, "ORB");
    extractor.extract(imgPath, "KAZE");
    extractor.extract(imgPath, "AKAZE");
    extractor.extract(imgPath, "BRISK");
 
    return 0;
}
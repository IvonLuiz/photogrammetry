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
        //dispatcher[""]

    }

    void extract(const std::string& imgPath, const std::string& featureType)
    {
        checkFeatureExtractor(featureType);
        cv::Mat img = getImg(imgPath);

        auto detector = dispatcher[featureType]();
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(img, keypoints);

        std::cout << "Detected " << keypoints.size() << " keypoints using " << featureType << "." << std::endl;
        showImage(img, keypoints, featureType);
    }

    void showImage(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const std::string& featureType)
    {
        cv::Mat img_keypoints;
        cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        cv::imshow("Keypoints - " + featureType, img_keypoints);
        cv::waitKey(0);  // Aguarda até que o usuário pressione uma tecla
        cv::destroyAllWindows();  // Fecha as janelas
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
    try {
        std::string imgPath = "image.jpg";  // Caminho para a imagem
        FeatureExtractor extractor;
        extractor.extract(imgPath, "SIFT");
        extractor.extract(imgPath, "ORB");
        #ifdef HAVE_OPENCV_XFEATURES2D
        extractor.extract(imgPath, "SURF");
        #endif
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

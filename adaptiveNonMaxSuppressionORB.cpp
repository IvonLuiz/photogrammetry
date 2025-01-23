#include <iostream>
#include <unordered_map>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                       const int numToKeep )
{
    if( keypoints.size() < numToKeep ) { return; }

    // Sort by response
    std::sort( keypoints.begin(), keypoints.end(),
                [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
                {
                return lhs.response > rhs.response;
                } );

    // Store the radius values for each keypoint
    std::vector<double> radii;
    radii.resize( keypoints.size() );
    std::vector<double> radiiSorted;
    radiiSorted.resize( keypoints.size() );

    // Constant to slightly adjust the response values for robustness on paper
    const float robustCoeff = 1.11;

    // For each keypoint i, calculates radius
    for( int i = 0; i < keypoints.size(); ++i )
    {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max(); // big number
        
        for( int j = 0; j < i && keypoints[j].response > response; ++j )
        {
            radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        
        radii[i]       = radius;
        radiiSorted[i] = radius;
    }

    // Descending order
    std::sort( radiiSorted.begin(), radiiSorted.end(),
                [&]( const double& lhs, const double& rhs )
                {
                    return lhs > rhs;
                } );

    const double decisionRadius = radiiSorted[numToKeep];
    
    std::vector<cv::KeyPoint> anmsPts;
    for( int i = 0; i < radii.size(); ++i )
    {
        if( radii[i] >= decisionRadius )
            anmsPts.push_back( keypoints[i] );
    }

    // Change original keypoints
    anmsPts.swap( keypoints );
}


int main(int argc, char *argv[])
{
    std::string imgPath = "image.png";
    
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

    if (img.empty())
        throw std::runtime_error("Could not open or find the image at path: " + imgPath);

    // ORB object
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints;
    orb->detect(img, keypoints);

    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    std::cout << "Detected " << keypoints.size() << " keypoints and descriptors using \n";
    cv::imshow("Keypoints with ORB standard", img_keypoints);

    adaptiveNonMaximalSuppresion(keypoints, keypoints.size()/5);

    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints with NMS ORB algorithm", img_keypoints);

    cv::waitKey(0);
    cv::destroyAllWindows();
}
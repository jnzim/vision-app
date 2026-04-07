#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vision
{

struct Detection
{
    cv::Rect box;
    float score;
};

class FaceDetector_OpenCVDNN
{
public:
    FaceDetector_OpenCVDNN();
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::Ptr<cv::FaceDetectorYN> m_detector;
    std::string m_modelPath{"models/face_detection_yunet_2023mar.onnx"};
    int m_inputWidth{320};
    int m_inputHeight{320};
    float m_scoreThreshold{0.5f};
    float m_nmsThreshold{0.3f};
    int m_topK{5000};
};

} // namespace vision
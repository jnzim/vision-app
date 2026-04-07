#include "FaceDetector_OpenCVDNN.h"

#include <opencv2/objdetect/face.hpp>
#include <stdexcept>

namespace vision
{

FaceDetector_OpenCVDNN::FaceDetector_OpenCVDNN()
{
    m_detector = cv::FaceDetectorYN::create(
        m_modelPath,
        "",
        cv::Size(m_inputWidth, m_inputHeight),
        m_scoreThreshold,
        m_nmsThreshold,
        m_topK);
}

std::vector<Detection> FaceDetector_OpenCVDNN::detect(const cv::Mat& frame)
{
    std::vector<Detection> out;

    if (frame.empty())
        return out;

    if (!m_detector)
        return out;

    m_detector->setInputSize(frame.size());

    cv::Mat faces;
    m_detector->detect(frame, faces);

    if (faces.empty())
        return out;

    for (int i = 0; i < faces.rows; ++i)
    {
        Detection det{};

        const float x = faces.at<float>(i, 0);
        const float y = faces.at<float>(i, 1);
        const float w = faces.at<float>(i, 2);
        const float h = faces.at<float>(i, 3);
        const float score = faces.at<float>(i, 14);

        det.box = cv::Rect(
            static_cast<int>(x),
            static_cast<int>(y),
            static_cast<int>(w),
            static_cast<int>(h));

        det.score = score;

        out.push_back(det);
    }

    return out;
}

} // namespace vision
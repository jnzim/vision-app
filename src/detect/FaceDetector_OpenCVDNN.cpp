// FaceDetector_OpenCVDNN.cpp
//
// YuNet face detector using OpenCV FaceDetectorYN.
// Requires OpenCV that provides <opencv2/objdetect/face.hpp> (you have 4.13).
//
// Key behavior:
// - FaceDetectorYN requires the input image size to match the configured input size.
// - We set the input size to the current frame size, and update via setInputSize()
//   if the camera resolution changes.

#include <vision/Interfaces/IFaceDetector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/face.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vision {

namespace fs = std::filesystem;

namespace {

static cv::Rect clampRect(const cv::Rect& r, int w, int h)
{
    return r & cv::Rect(0, 0, w, h);
}

static void logModelInfoOrThrow(const std::string& modelPath)
{
    std::cerr << "[FaceDetector] cwd: " << fs::current_path() << "\n";
    std::cerr << "[FaceDetector] model: " << modelPath << "\n";

    if (!fs::exists(modelPath))
        throw std::runtime_error("Model not found: " + modelPath);

    const auto sz = fs::file_size(modelPath);
    std::cerr << "[FaceDetector] model bytes: " << sz << "\n";

    if (sz == 0)
        throw std::runtime_error("Model file is empty: " + modelPath);
}

} // namespace

class FaceDetector_OpenCVDNN final : public IFaceDetector
{
public:
    explicit FaceDetector_OpenCVDNN(std::string modelPath,
                                   float scoreThresh = 0.6f,
                                   float nmsThresh   = 0.3f,
                                   int topK          = 5000)
        : m_modelPath(std::move(modelPath)),
          m_scoreThresh(scoreThresh),
          m_nmsThresh(nmsThresh),
          m_topK(topK)
    {}

    std::vector<Detection> detect(const cv::Mat& frameBgr) override
    {
        if (frameBgr.empty())
            return {};

        ensureInitialized(frameBgr.size());

        // faces: Nx15 -> [x y w h score l0x l0y ... l4x l4y]
        cv::Mat faces;
        m_yn->detect(frameBgr, faces);

        std::vector<Detection> out;
        out.reserve(static_cast<size_t>(faces.rows));

        for (int i = 0; i < faces.rows; ++i)
        {
            const float score = faces.at<float>(i, 4);
            if (score < m_scoreThresh)
                continue;

            cv::Rect box(
                static_cast<int>(faces.at<float>(i, 0)),
                static_cast<int>(faces.at<float>(i, 1)),
                static_cast<int>(faces.at<float>(i, 2)),
                static_cast<int>(faces.at<float>(i, 3)));

            box = clampRect(box, frameBgr.cols, frameBgr.rows);
            if (box.area() <= 0)
                continue;

            out.push_back(Detection{box, score});
        }

        return out;
    }

private:
    void ensureInitialized(const cv::Size& frameSize)
    {
        if (!m_yn)
        {
            logModelInfoOrThrow(m_modelPath);

            m_inputSize = frameSize;

            m_yn = cv::FaceDetectorYN::create(
                m_modelPath,
                "",             // config unused for ONNX
                m_inputSize,
                m_scoreThresh,
                m_nmsThresh,
                m_topK
            );

            if (!m_yn)
                throw std::runtime_error("FaceDetectorYN::create failed");

            std::cerr << "[FaceDetector] FaceDetectorYN initialized "
                      << m_inputSize.width << "x" << m_inputSize.height
                      << " score=" << m_scoreThresh
                      << " nms=" << m_nmsThresh << "\n";
        }
        else if (m_inputSize != frameSize)
        {
            m_inputSize = frameSize;
            m_yn->setInputSize(m_inputSize);

            std::cerr << "[FaceDetector] FaceDetectorYN updated input size to "
                      << m_inputSize.width << "x" << m_inputSize.height << "\n";
        }
    }

private:
    std::string m_modelPath;
    float m_scoreThresh = 0.6f;
    float m_nmsThresh   = 0.3f;
    int   m_topK        = 5000;

    cv::Size m_inputSize{};
    cv::Ptr<cv::FaceDetectorYN> m_yn;
};

std::unique_ptr<IFaceDetector> CreateFaceDetector_OpenCVDNN()
{
    return std::make_unique<FaceDetector_OpenCVDNN>(
        "models/face_detection_yunet_2023mar.onnx",
        /*scoreThresh*/ 0.6f,
        /*nmsThresh*/   0.3f,
        /*topK*/        5000
    );
}

} // namespace vision

// FaceEmbedder_OpenCVDNN.cpp
//
// Face embedding wrapper for your IFaceEmbedder interface.
// Uses OpenCV Zoo SFace ONNX: face_recognition_sface_2021dec.onnx
//
// Preferred path: cv::FaceRecognizerSF (if available in your OpenCV build)
// Fallback path : cv::dnn::Net (readNetFromONNX + forward)
//
// Output: L2-normalized embedding vector<float>.
//
// Model download (repo root):
//   mkdir -p models
//   curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

#include <vision/Interfaces/IFaceEmbedder.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#if __has_include(<opencv2/objdetect/face.hpp>)
  #include <opencv2/objdetect/face.hpp>  // FaceRecognizerSF
  #define HAS_OPENCV_SFACE 1
#else
  #define HAS_OPENCV_SFACE 0
#endif

namespace vision {

namespace fs = std::filesystem;

namespace {

static void logModelInfoOrThrow(const std::string& modelPath)
{
    static bool logged = false;
    if (logged) return;
    logged = true;

    std::cerr << "[FaceEmbedder] cwd: " << fs::current_path() << "\n";
    std::cerr << "[FaceEmbedder] model: " << modelPath << "\n";

    if (!fs::exists(modelPath))
        throw std::runtime_error("Embedder model not found: " + modelPath);

    const auto sz = fs::file_size(modelPath);
    std::cerr << "[FaceEmbedder] model bytes: " << sz << "\n";
    if (sz == 0)
        throw std::runtime_error("Embedder model file is empty: " + modelPath);
}

static std::vector<float> matToVector(const cv::Mat& m)
{
    if (m.empty())
        return {};

    CV_Assert(m.depth() == CV_32F);
    CV_Assert(m.total() > 0);

    std::vector<float> out;
    out.resize(m.total());

    const float* p = m.ptr<float>(0);
    std::copy(p, p + m.total(), out.begin());
    return out;
}

static void l2Normalize(std::vector<float>& v)
{
    double ss = 0.0;
    for (float x : v) ss += (double)x * (double)x;
    const double n = std::sqrt(ss);
    if (n <= 1e-12) return;
    for (float& x : v) x = (float)(x / n);
}

} // namespace

// Optional helper you can use from VisionProcessor (declare in your header if you want it globally).
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || b.empty() || a.size() != b.size())
        return -1.0f;

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    if (denom <= 1e-12) return -1.0f;
    return (float)(dot / denom);
}

class FaceEmbedder_OpenCVDNN final : public IFaceEmbedder
{
public:
    explicit FaceEmbedder_OpenCVDNN(std::string modelPath,
                                   int inputW = 112,
                                   int inputH = 112)
        : m_modelPath(std::move(modelPath)),
          m_inputW(inputW),
          m_inputH(inputH)
    {}

    std::vector<float> embed(const cv::Mat& faceBgr) override
    {
        if (faceBgr.empty())
            return {};

        ensureInitialized();

#if HAS_OPENCV_SFACE
        if (m_sface)
            return embedWithSFace(faceBgr);
#endif
        return embedWithDNN(faceBgr);
    }

private:
    void ensureInitialized()
    {
        if (m_initialized)
            return;

        logModelInfoOrThrow(m_modelPath);

#if HAS_OPENCV_SFACE
        try
        {
            // Second argument is config; unused for ONNX
            m_sface = cv::FaceRecognizerSF::create(m_modelPath, "");
            if (m_sface)
            {
                std::cerr << "[FaceEmbedder] FaceRecognizerSF init OK. input="
                          << m_inputW << "x" << m_inputH << "\n";
                m_initialized = true;
                return;
            }
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "[FaceEmbedder] FaceRecognizerSF init failed, falling back to DNN: " << e.what() << "\n";
        }
#endif

        // DNN fallback
        try
        {
            m_net = cv::dnn::readNetFromONNX(m_modelPath);
        }
        catch (const cv::Exception& e)
        {
            throw std::runtime_error(std::string("FaceEmbedder readNetFromONNX failed: ") + e.what());
        }

        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::cerr << "[FaceEmbedder] DNN init OK. input=" << m_inputW << "x" << m_inputH << "\n";
        m_initialized = true;
    }

#if HAS_OPENCV_SFACE
    std::vector<float> embedWithSFace(const cv::Mat& faceBgr)
    {
        // FaceRecognizerSF expects an aligned crop typically, but it still works well
        // with a simple resized face ROI for “same-person” gating.
        cv::Mat resized;
        cv::resize(faceBgr, resized, cv::Size(m_inputW, m_inputH), 0, 0, cv::INTER_LINEAR);

        cv::Mat feat;
        // feature() outputs a 1xD CV_32F row
        m_sface->feature(resized, feat);

        auto v = matToVector(feat);
        l2Normalize(v);
        return v;
    }
#endif

    std::vector<float> embedWithDNN(const cv::Mat& faceBgr)
    {
        cv::Mat resized;
        cv::resize(faceBgr, resized, cv::Size(m_inputW, m_inputH), 0, 0, cv::INTER_LINEAR);

        // Many face recognition models are trained on RGB normalized to [0,1]
        // If similarity looks bad, try swapRB=false or mean/std normalization.
        const double scale = 1.0 / 255.0;
        const cv::Scalar mean(0, 0, 0);
        const bool swapRB = true;
        const bool crop = false;

        cv::Mat blob = cv::dnn::blobFromImage(resized, scale, cv::Size(m_inputW, m_inputH),
                                              mean, swapRB, crop, CV_32F);

        m_net.setInput(blob);

        cv::Mat out = m_net.forward();

        // Flatten any shape to a 1-D vector<float>
        cv::Mat outFlat = out.reshape(1, 1); // 1 x (total)
        outFlat.convertTo(outFlat, CV_32F);

        auto v = matToVector(outFlat);
        l2Normalize(v);
        return v;
    }

private:
    std::string m_modelPath;
    int m_inputW = 112;
    int m_inputH = 112;

    bool m_initialized = false;

#if HAS_OPENCV_SFACE
    cv::Ptr<cv::FaceRecognizerSF> m_sface;
#endif

    cv::dnn::Net m_net;
};

std::unique_ptr<IFaceEmbedder> CreateFaceEmbedder_OpenCVDNN()
{
    const std::string model = "models/face_recognition_sface_2021dec.onnx";
    return std::make_unique<FaceEmbedder_OpenCVDNN>(model, 112, 112);
}

} // namespace vision

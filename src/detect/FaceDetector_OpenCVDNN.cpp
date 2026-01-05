#include <vision/Interfaces/IFaceDetector.h>

namespace vision {

class FaceDetector_OpenCVDNN final : public IFaceDetector {
public:
    std::vector<Detection> detect(const cv::Mat&) override {
        // Stub: no detection yet
        return {};
    }
};

} // namespace vision

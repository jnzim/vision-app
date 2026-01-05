#include <vision/Interfaces/IFaceEmbedder.h>

namespace vision {

class FaceEmbedder_OpenCVDNN final : public IFaceEmbedder {
public:
    std::vector<float> embed(const cv::Mat&) override {
        // Stub: no embedding yet
        return {};
    }
};

} // namespace vision

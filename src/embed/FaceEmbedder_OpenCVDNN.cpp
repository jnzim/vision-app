#include <vision/Interfaces/IFaceEmbedder.h>
#include <memory>

namespace vision 
{

    class FaceEmbedder_OpenCVDNN final : public IFaceEmbedder   
    {
        public:
        std::vector<float> embed(const cv::Mat&) override 
        {
            // Stub: no embedding yet
            return {};
        }
    };

    std::unique_ptr<IFaceEmbedder> CreateFaceEmbedder_OpenCVDNN()
    {
        return std::make_unique<FaceEmbedder_OpenCVDNN>();
    }

} // namespace vision

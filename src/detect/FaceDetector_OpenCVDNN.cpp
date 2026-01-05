#include <vision/Interfaces/IFaceDetector.h>
#include <memory>
namespace vision 
{

    class FaceDetector_OpenCVDNN final : public IFaceDetector 
    {
    public:
        std::vector<Detection> detect(const cv::Mat&) override 
        {
            // Stub: no detection yet
            return {};
        }
    };

    std::unique_ptr<IFaceDetector> CreateFaceDetector_OpenCVDNN()
    {
        return std::make_unique<FaceDetector_OpenCVDNN>();
    }

} // namespace vision

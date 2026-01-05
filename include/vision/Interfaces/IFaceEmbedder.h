#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace vision {

class IFaceEmbedder {
public:
    virtual ~IFaceEmbedder() = default;
    virtual std::vector<float> embed(const cv::Mat& alignedFaceBgr) = 0;
};

} // namespace vision

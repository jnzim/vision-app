#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <vision/Types/Types.h>

namespace vision {

class IFaceDetector {
public:
    virtual ~IFaceDetector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& frameBgr) = 0;
};

} // namespace vision

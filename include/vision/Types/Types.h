#pragma once
#include <opencv2/core.hpp>

namespace vision {

struct Detection {
    cv::Rect box;
    float score = 0.0f;
};

} // namespace vision

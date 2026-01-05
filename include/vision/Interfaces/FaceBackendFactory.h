#pragma once

#include <memory>
#include <vision/Interfaces/IFaceDetector.h>
#include <vision/Interfaces/IFaceEmbedder.h>

namespace vision 
{

    // OpenCV backend factories (stubs for now; real impl later)
    std::unique_ptr<IFaceDetector> CreateFaceDetector_OpenCVDNN();
    std::unique_ptr<IFaceEmbedder> CreateFaceEmbedder_OpenCVDNN();

} // namespace vision

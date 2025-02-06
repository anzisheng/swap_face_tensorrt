#pragma once
#include <vector>
#include "NvInfer.h"
using namespace std;
template<typename T>
class IEngine
{
    public:
    virtual ~IEngine() = default;
    virtual bool buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals={0.f, 0.f, 0.f},
                                  const std::array<float, 3> &divVals={1.f, 1.f, 1.f}, bool normalize = true) = 0;
    virtual bool loadNetwork(std::string trtModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f}, 
                             const std::array<float, 3> &divVals={1.f, 1.f, 1.f}, bool normalize= true) = 0;
    virtual bool runInterface(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                              std::vector<std::vector<std::vector<T>>> &featureVectors) = 0;
    virtual const std::vector<nvinfer1::Dims3> &getInputDims() = 0;
    virtual const std::vector<nvinfer1::Dims> &getOutputDims() = 0;


}
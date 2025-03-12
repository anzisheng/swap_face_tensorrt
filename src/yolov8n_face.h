#pragma once
#include "engine.h"
#include <string>
using namespace std;

// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Segmentation config options
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose estimation options
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class thresholds (default are COCO classes)
    std::vector<std::string> classNames = {
        "person"};    
};
struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};



class YoloV8_face
{
    public:
    //Build the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV8_face(const string &onnxModelPath, const YoloV8Config &config);

    //Detect the object in the image
    std::vector<Object> detectObjects(const cv::Mat &inputImageBGR);
    std::vector<Object> detectObjects(congst cv::cuda::GpuMat&inputImageBGR);

    //Draw the object bounding box and labels on the image
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale = 2);
    private:
    //preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg);
    
    //post process the output
    std::vector<Object> postprocessDetect(std::vector<float> &featureVector);

    std::unique_ptr<Engine<float>> m_trtEngine = nullptr;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    // Filter thresholds
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    // Segmentation constants
    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    // Object classes as strings
    const std::vector<std::string> CLASS_NAMES;

    // Pose estimation constant
    const int NUM_KPS;
    const float KPS_THRESHOLD;


};
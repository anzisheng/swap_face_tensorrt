# ifndef DETECT_FACE68LANDMARKS_trt
# define DETECT_FACE68LANDMARKS_trt
#include "engine.h"
#include "yolov8n_face.h" 
#include "utils.h"
//#include "utile.h"
using namespace std;
using namespace cv;
class Face68Landmarks_trt
{
public:    
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    Face68Landmarks_trt(const std::string &onnxModelPath, const YoloV8Config &config, int method = 0);
    
    vector<Point2f> detectlandmark(const cv::cuda::GpuMat &inputImageBGR, Object& object,vector<Point2f> &face_landmark_5of68);    
    vector<Point2f> detectlandmark(const cv::Mat &inputImageBGR,Object& object, vector<Point2f> &face_landmark_5of68) ;
    
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg, Object& object);
    // Postprocess the output
    std::vector<cv::Point2f> postprocess(std::vector<float> &featureVector, vector<Point2f> &face_landmark_5of68);

    cv::Mat m_srcImg;
    cv::Mat inv_affine_matrix;
private:
    std::unique_ptr<Engine<float>> m_trtEngine_landmark = nullptr;
    

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;
    std::string m_onnxname;
    
};
#endif
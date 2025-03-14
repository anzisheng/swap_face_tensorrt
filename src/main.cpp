#include "cmd_line_parser.h"
#include "logger.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include "yolov8n_face.h"
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string onnxModelPathLandmark;
    std::string inputImage = "1.jpg";

    YoloV8_face yoloV8("yoloface_8n.onnx", config); //
    
    std::cout << "hello, world" <<std::endl;

    return 0;
}

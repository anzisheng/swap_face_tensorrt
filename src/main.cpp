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
    std::string socrceImage = "1.jpg";

    YoloV8_face yoloV8("yoloface_8n.onnx", config); //
    


    cv::Mat source_img = cv::imread(socrceImage);
    std::vector<Object> object = yoloV8.detectObjects(source_img);
    std::cout << "object sizes: " << object.size() << std::endl;
    std::cout << "hello, world" <<std::endl;

    #ifdef SHOW  
     // draw the bounding box on the image
     yoloV8.drawObjectLabels(source_img, object);
    #endif
    return 0;
}

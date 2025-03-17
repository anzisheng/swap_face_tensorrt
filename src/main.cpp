#include "cmd_line_parser.h"
#include "logger.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include "yolov8n_face.h"

extern std::string g_token = "";

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string onnxModelPathLandmark;
    std::string sourceImage = "1.jpg";

    YoloV8_face yoloV8("yoloface_8n.onnx", config); //
    


    cv::Mat source_img = cv::imread(sourceImage);
    std::vector<Object> object = yoloV8.detectObjects(source_img);
    std::cout << "object sizes: " << object.size() << std::endl;
    std::cout << "hello, world" <<std::endl;

    #ifdef SHOW  
     // draw the bounding box on the image
     
     yoloV8.drawObjectLabels(source_img, object);
     g_token = sourceImage.substr(0, sourceImage.find_last_of('.')) + "_annotated.jpg";
     cv::imwrite(g_token, source_img);


    #endif
    return 0;
}

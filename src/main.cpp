#include "cmd_line_parser.h"
#include "logger.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include "yolov8n_face.h"
#include "Face68LandMarks_trt.h"
#include "facerecognizer_trt.h"

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
    Face68Landmarks_trt detect_68landmarks_net_trt("2dfan4.onnx", config);

    std::vector<cv::Point2f> face_landmark_5of68_trt;
    std::vector<cv::Point2f> face68landmarks_trt = detect_68landmarks_net_trt.detectlandmark(source_img, object[0], face_landmark_5of68_trt);

    #ifdef SHOW
    std::cout << "\nface68landmarks_trt size: " <<face68landmarks_trt.size()<<std::endl;
    std::cout << "face_landmark_5of68_trt size: " <<face_landmark_5of68_trt.size()<<std::endl;
    // for(int i =0; i < face68landmarks_trt.size(); i++)
	// {
    //     cout << face68landmarks_trt[i] << " ";
	// }    
    // for(int i =0; i < face_landmark_5of68_trt.size(); i++)
	// {
    //     cout << face_landmark_5of68_trt[i] << " ";
	// }
    #endif
    FaceEmbdding_trt face_embedding_net_trt("arcface_w600k_r50.onnx", config);
    vector<float> source_face_embedding = face_embedding_net_trt.detect(source_img, face_landmark_5of68_trt);
    std::cout << "\nsource_face_embedding size: " << source_face_embedding.size() <<endl;

    
    return 0;
}

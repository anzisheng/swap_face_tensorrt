#include "cmd_line_parser.h"
#include "logger.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include "yolov8n_face.h"
#include "Face68LandMarks_trt.h"
#include "facerecognizer_trt.h"
#include "faceswap_trt.h"
#include "faceenhancer_trt.h"
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

   /////////////////////////////////////////////////////////////////////////////////////
   std::string targetImage = "12.jpg"; 
   cv::Mat target_img = cv::imread(targetImage);
   cv::Mat target_img2 =target_img.clone();

   std::vector<Object>objects_target = yoloV8.detectObjects(target_img);

#ifdef SHOW
    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(target_img2, objects_target);
    cout << "Detected " << objects_target.size() << " objects" << std::endl;
    // Save the image to disk
    auto target_image_name = targetImage.substr(0, targetImage.find_last_of('.'));
    g_token = target_image_name;
    std::cout << "target:::::"<<g_token << std::endl;
    //const auto outputName_target = outputImage.substr(0, outputImage.find_last_of('.')) + "_annotated.jpg";
    const auto outputName_target = g_token + "_annotated.jpg";
    cv::imwrite(outputName_target, target_img2);
    std::cout << "Saved annotated image to: " << outputName_target << std::endl;
#endif
    int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> target_landmark_5(5);    
    //得到目标人脸的landmark
	detect_68landmarks_net_trt.detectlandmark(target_img, objects_target[position], target_landmark_5);

    ////////////////////////////////////////////////
    SwapFace_trt swap_face_net_trt("inswapper_128.onnx", config, 1);
    samplesCommon::BufferManager buffers(swap_face_net_trt.m_trtEngine_faceswap->m_engine);
    
    FaceEnhance_trt enhance_face_net_trt("gfpgan_1.4.onnx", config, 1);
    samplesCommon::BufferManager buffers_enhance(enhance_face_net_trt.m_trtEngine_enhance->m_engine);

    cv::Mat swapimg = swap_face_net_trt.process(target_img, source_face_embedding, target_landmark_5, buffers);
    //imwrite("target_img.jpg", target_img);
//#ifdef SHOW        
    //std::cout << "swap_face_net.process end" <<std::endl;
    //imwrite("swapimg.jpg", swapimg);
//#endif    
    
    cv::Mat resultimg = enhance_face_net_trt.process(swapimg, target_landmark_5, buffers_enhance);
    
    imwrite("resultimgend.jpg", resultimg);

    return 0;
}

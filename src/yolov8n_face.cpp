#include "yolov8n_face.h"

YoloV8_face::YoloV8_face(const std::string &onnxModelPath, const YoloV8Config &config)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold), NMS_THRESHOLD(config.nmsThreshold), TOP_K(config.topK),
      SEG_CHANNELS(config.segChannels), SEG_H(config.segH), SEG_W(config.segW), SEGMENTATION_THRESHOLD(config.segmentationThreshold),
      CLASS_NAMES(config.classNames), NUM_KPS(config.numKPS), KPS_THRESHOLD(config.kpsThreshold) {
        
        //specify options for GPU inference
        Options options;
        options.optBatchoptBatch = 1;
        options.maxBatchSize = 1;
        

        options.precision = config.precision;
        options.calibrationDataDirectoryPath = config.calibrationDirectoryPath;

        if(options.precision == Precision::INT8){
            if(options.calibrationDataDirectoryPath.empty()){
                throw std::runtime_error("Must suply calibartion data path");

            }
        }
        //create out TensorRT interference Engine
        m_trtEngine = std::make_unique<Engine<float>>(Options);

        //Build the onnx model into TensorRT engine file, cache the file to the disk,
        // and then load the TensorRT engie file into memory
        // the engine file is rebuilt any time the above options are channged
        auto succ = m_trtEngine->buildLoadNetwork(onnxModelPath, SUB_Vals, DIV_VALS, NORMALIZE);
        if(!succ)
        {
            throw std::runtime_error("error!");
        }
      
      
      }

std::vetor<std::vector<cv::cuda::GpuMat>> YoloV8_face::preprocess(const cv::cuda::GpuMat &gpuImg){
    //populate the input vectors
    const auto &inputDims = m_trtEngine->getInputDims();
    std::cout <<std::endl << "yolo input dims: "<< inputDims[0].d[1] << " ," <<inputDims[0].d[2] <<std::endl;
    //convert BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized =  rgbMat;
    
    //resize the model to the expected size while maintaining the aspect ratio with the use of padding
    if(resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2])
    {
        //only resize if not already the right size to avoid unnecessary copy
        resized = Engine<float>::resizedKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);

    }
    


}




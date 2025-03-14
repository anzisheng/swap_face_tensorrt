#include "yolov8n_face.h"

YoloV8_face::YoloV8_face(const std::string &onnxModelPath, const YoloV8Config &config)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold), NMS_THRESHOLD(config.nmsThreshold), TOP_K(config.topK),
      SEG_CHANNELS(config.segChannels), SEG_H(config.segH), SEG_W(config.segW), SEGMENTATION_THRESHOLD(config.segmentationThreshold),
      CLASS_NAMES(config.classNames), NUM_KPS(config.numKPS), KPS_THRESHOLD(config.kpsThreshold) {
        
        //specify options for GPU inference
        Options options;
        options.optBatchSize = 1;
        options.maxBatchSize = 1;
        

        options.precision = config.precision;
        options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

        if(options.precision == Precision::INT8){
            if(options.calibrationDataDirectoryPath.empty()){
                throw std::runtime_error("Must suply calibartion data path");

            }
        }
        //create out TensorRT interference Engine
        m_trtEngine = std::make_unique<Engine<float>>(options);

        //Build the onnx model into TensorRT engine file, cache the file to the disk,
        // and then load the TensorRT engie file into memory
        // the engine file is rebuilt any time the above options are channged
        auto succ = m_trtEngine->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
        if(!succ)
        {
            throw std::runtime_error("error!");
        }
      
      
      }

    //std::vector<vector<cv::Mat> yolov8n_face::preprocess(cv:cuda::GpuMat &gpuImg)
    std::vector<std::vector<cv::cuda::GpuMat>> YoloV8_face::preprocess(const cv::cuda::GpuMat &gpuImg)
    {
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
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
        //convert to format expected by our inference engine
        //The reason for the strange format is because it supports models with multiple inpus as well as batching
        //In our case thought,  the model only has a single input and we are using a batch size of 1.
        std::vector<cv::cuda::GpuMat> input{std::move(resized)};
        std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

        //These params will be used in the post-processing stage
        m_imgHeight = rgbMat.rows;
        m_imgWidth = rgbMat.cols;
        m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));
        return inputs;

    }
}
    
std::vector<Object> YoloV8_face::detectObjects(const cv::cuda::GpuMat& inputImageBGR){
    const auto input = preprocess(inputImageBGR);
    
    std::vector<std::vector<std::vector<float>>> featureVectors; 
    auto succ = m_trtEngine->runInference(input, featureVectors);
    //check if our model does only detect object detection
    std::vector<Object> ret;
    const auto &numOutputs = m_trtEngine->getOutputDims().size();
    if (numOutputs == 1)
    {
        //does only object detection, no segmetatio  and no pose
        //Since we have a batch size of 1 and only 1 output,
        //we must convert  the output from a 3d array to a 1d array
        std::vector<float> featureVector;   
        Engine<float>::transformOutput(featureVectors, featureVector);

        const auto &outputDims = m_trtEngine->getOutputDims();
        int numChannels = outputDims[outputDims.size() -1].d[1];
        if (numChannels == 56)
        {
            //pose estimation
        }
        else
        {
            //object detection
            ret = postprocessDetect(featureVector);
        }
        

    }
    else
    {
        //segmentation
    }
    return ret;
}    

std::vector<Object> YoloV8_face::detectObjects(const cv::Mat &inputImageBGR)
{
    //upload the image to Gpu memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);

    //call detectObjects with the GPU image
    return detectObjects(gpuImg);


}


std::vector<Object> YoloV8_face::postprocessDetect(std::vector<float> &featureVector)
{
    const auto &outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];

    auto numClasses = CLASS_NAMES.size();
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t(); //转置
    //Get all the YOLO proposals
    for(int i = 0; i < numAnchors; i++){
       auto rowPtr = output.row(i).ptr<float>();
       auto bboxesPtr = rowPtr;
       auto scoresPtr = rowPtr + 4;
       auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
       float score = *maxSPtr;
       if (score> PROBABILITY_THRESHOLD)
       {
        float x = *bboxesPtr++;
        float y = *bboxesPtr++;
        float w = *bboxesPtr++;
        float h = *bboxesPtr;        

        float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
        float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
        float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
        float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

   
       }
       
    }
 // Run NMS
 cv::dnn::NMSBoxes(bboxes, scores, /*labels,*/ PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);
 std::vector<Object> objects;   
   // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices) {
        if (cnt >= TOP_K) {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;

}



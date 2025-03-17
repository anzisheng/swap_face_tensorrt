#include "utils.h"

using namespace std;
using namespace cv;

Mat warp_face_by_face_landmark_5(const Mat temp_vision_frame, Mat &crop_img, const vector<Point2f> face_landmark_5, const vector<Point2f> normed_template, const Size crop_size)
{    
    Mat affine_matrix = cv::estimateAffinePartial2D(face_landmark_5, normed_template, cv::noArray(), cv::RANSAC, 100.0);    
    warpAffine(temp_vision_frame, crop_img, affine_matrix, crop_size, cv::INTER_AREA, cv::BORDER_REPLICATE);
    return affine_matrix;
}
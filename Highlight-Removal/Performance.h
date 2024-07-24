#ifndef PERFORMANCE_H
#define PERFORMANCE_H


#include<opencv2/opencv.hpp>

double CalcCOV(const cv::Mat &RGBImg, const cv::Mat &HighlightImg, double &MeanValueAve, double &StandardValAve);

double CalcMSE(const cv::Mat &RGBImg, const cv::Mat &LowRankImage, const cv::Mat &HighlightImg);

double CalcPSNR(double MSE);
double CalcSSIM(const cv::Mat &RGBImg, const cv::Mat &LowRankImg, const cv::Mat &HighlightImg);

#endif // PERFORMANCE_H

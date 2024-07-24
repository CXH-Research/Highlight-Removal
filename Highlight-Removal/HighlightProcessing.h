#ifndef HIGHLIGHTPROCESSING_H
#define HIGHLIGHTPROCESSING_H

#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include"opencv2/core/eigen.hpp"
#include<mkl.h>
#include<RPCA.h>
//#include"omp.h"
//#include<omp.h>

#define ADAPTIVEBITHRESHOLDS 1//自适应双阈值
#define EMPERICALBITHRESHOLDS 2//经验阈值
#define MSVTHRESHOLDS 3//
#define MSVG 4 //参考信息：最小值通道、饱和度通道、强度值通道、梯度信息

#define EMPERICALTHRESHOLD_S 0.3
#define EMPERICALTHRESHOLD_V 0.7
#define PERCENTOFTHRESHOLD 0.02

#define REPLACE 1
#define ADAWEIGHT 2

#define RPCA_FPCP 1
#define RPCA_LAGQN 2
#define RPCA_ADARPCA 3
#define RPCA_PCP 4
#define RPCA_NONCONVEX 5
#define RPCA_NONCONVEXBATCH 6


#define METHODGRADDILATION 1
#define METHODGRADCONTOURS 2


#define EnhanceSobel 1
#define EnhanceGrad 2
#define EnhanceNone 3

void ImgHighlightRemoval(cv::Mat RGBImg, cv::Mat &dstImg, cv::Mat &LImg, cv::Mat &SImg, cv::Mat &HighlightImg, int enhancedMethod, int DetectionMethod, \
                         int RPCA_Type, bool BATCH = false, int batchesRow = 1, int batchesCol = 1);
void ImgHighlightRemoval(std::string RGBImgPath, std::string LowRankPath, std::string SparsePath, std::string HighlightPath, std::string dstImgPath, int RPCA_Type, int DetectionMethod, int EnhanceMethod, bool BATCH = false, int batchesRow = 1, int batchesCol = 1);
void TestHighlightRemoval_ImgSequences(cv::Mat Frame, cv::Mat BorderImg, cv::Mat &LImg,cv::Mat &SImg, cv::Mat &dstImg, cv::Mat &HighlightImg, int Method, int Enhancement_Method, int RPCA_Type=RPCA_NONCONVEX);
void InitMatrixVec(std::vector<Eigen::MatrixXd> &Sequences, int size, int nr, int nc);

void SplitImg(cv::Mat &RGBImg, std::vector<Eigen::MatrixXd> &BGRSequences);
void RGB2MSV(std::vector<Eigen::MatrixXd> &BGRSequences, std::vector<Eigen::MatrixXd> &MSVSequences);


void CalculateMeanStd(Eigen::MatrixXd &matrix, double &MeanVal, double &stdVal);
void IsolateWithMSV(std::vector<Eigen::MatrixXd> &MSVSequences, double threshold_s, double threshold_v, Eigen::MatrixXd &Highlight);
void HighlightDetectionGrad(std::vector<Eigen::MatrixXd> &MSVSequences, Eigen::MatrixXd &Grad, Eigen::MatrixXd &Highlight, int Method = METHODGRADCONTOURS);
void HighlightDetection(std::vector<Eigen::MatrixXd> &MSVSequences,  Eigen::MatrixXd &Highlight, int Methods);
void IsolateWithSV(Eigen::MatrixXd &Saturation, double threshold_s, Eigen::MatrixXd &ValueChan, double threshold_v, Eigen::MatrixXd &Highlight);
void InitData(std::vector<Eigen::MatrixXd> &BGRSequences, std::vector<Eigen::MatrixXd> &LowRankSequences, \
    std::vector<Eigen::MatrixXd> &SparseSequences, Eigen::MatrixXd &Highlight);
void Blocking(std::vector<Eigen::MatrixXd> &ImgMatrix, std::vector<Eigen::MatrixXd> &Batches, int BatchRows, int BatchCols, int nr, int nc);
void Blocking(Eigen::MatrixXd &ImgMatrix, std::vector<Eigen::MatrixXd> &Batches, int BatchRows, int BatchCols, int nr, int nc);
void Merging(std::vector<Eigen::MatrixXd> &Batches, cv::Mat &RGBImg, int BatchRows, int BatchCols, int nr, int nc, double ratio);
void HighlightRefine(cv::Mat &SparseImg, cv::Mat &Highlight);
int IsHighlightPixel(uchar pixel);
void AddWeight(cv::Mat &Input1, cv::Mat &Input2, cv::Mat &Weight, cv::Mat &Output);
void HighlightReconstruct(cv::Mat &RGBImg, cv::Mat &LImg, cv::Mat Highlight, cv::Mat &dstImg, int METHODS, bool Dilation);



void TestHighlightRemoval_ImgSequences_withoutBorder(std::string RGBRootPath, std::string LowRankRootPath, std::string SparseRootPath, \
    std::string HighlightRootPath, std::string dstImgRootPath, int Method, int Enhancement_Method);
void TestHighlightRemoval_ImgSequences(std::string RGBRootPath, std::string LowRankRootPath, std::string SparseRootPath, \
    std::string HighlightRootPath, std::string dstImgRootPath, int Method, int Enhancement_Method);
#endif // HIGHLIGHTPROCESSING_H

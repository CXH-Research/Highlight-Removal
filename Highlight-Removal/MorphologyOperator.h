#ifndef MORPHOLOGYOPERATOR_H
#define MORPHOLOGYOPERATOR_H
#include<Eigen/Dense>
#include<opencv2/core/eigen.hpp>
#include<iostream>
#include<vector>
#include<opencv2\opencv.hpp>

void EigenDilate(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &dstMat, int  kernelRows, int kernelCols);

void MyFilter2D(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &dstMat, Eigen::MatrixXd &kernel);

void CalcGradient(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &Grad);

void CalcGradient(Eigen::MatrixXd &srcMat, std::vector<Eigen::MatrixXd> &Grad);

double grayThreshOSTU(int *Hist, int length);

void imbinarize(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &dstMat);

std::vector<int>  findIndex(Eigen::MatrixXd &srcMat);

std::vector<double>  findEle(Eigen::MatrixXd &srcMat);

std::vector<uchar>  findEle(cv::Mat &grayImg);

std::vector<double>  findEleD(cv::Mat &srcMat);

double graythresh(std::vector<uchar> &EleVec);

double graythresh(std::vector<double> &EleVec);

void EigenThreshold(Eigen::MatrixXd &InputOutPutMatrix, double thre, double MaxVal = 1);

void Rgb2Gray(std::vector<Eigen::MatrixXd>&BGRSequences, Eigen::MatrixXd &GrayMatrix);

void AddWeight(std::vector<Eigen::MatrixXd> &Sequences1, std::vector<Eigen::MatrixXd> &Sequences2, std::vector<Eigen::MatrixXd> &Sequences3);

void NormlizeMax(Eigen::MatrixXd &Matr);

void AddWeight(std::vector<Eigen::MatrixXd> &InputSequences, Eigen::MatrixXd &Matr, std::vector<Eigen::MatrixXd> &OutputSequences, double theta);

void MSVTuning(std::vector<Eigen::MatrixXd> &MSVSequences, Eigen::MatrixXd &Grad, double theta);

void SequencesSobel(std::vector<Eigen::MatrixXd> &Input, std::vector<Eigen::MatrixXd> &OutPut);
#endif // MORPHOLOGYOPERATOR_H

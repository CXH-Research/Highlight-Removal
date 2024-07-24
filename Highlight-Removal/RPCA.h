#ifndef RPCA_H
#define RPCA_H

#include<Matrix_Decomposition.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<MorphologyOperator.h>
#include<iostream>
#define SVDNPOWER 1
#define SVDOFFSET 5
//定义一些公共方法

typedef struct
{
    int iteration;
    int oldRank;
    Eigen::MatrixXd Vold;
    int SVDoffset;
    int SVDnPower;
}Record;
typedef struct
{
    int oldRank;
    Eigen::MatrixXd Vold;
    int iteration;
    double ratio;
}RecordForNonConv;


void Shrinkage(Eigen::MatrixXd &srcData, double tau);

int SignOfEle(double data);

int projNuclear(Eigen::MatrixXd &X, double tauL, Record &record);

int projNuclear(Eigen::MatrixXd &X, double tau, RecordForNonConv &record, int times);//针对新方法的核范数投影

double CalcFNorm(Eigen::MatrixXd &Input1, Eigen::MatrixXd &Input2);

void LagQN(Eigen::MatrixXd &AY, Eigen::MatrixXd &L, Eigen::MatrixXd &S, double lambdaS);


//用Alternating Direction方法求解PCP问题，来自于2009年的文章
void PCP(Eigen::MatrixXd &AY, Eigen::MatrixXd &L, Eigen::MatrixXd &S, double mu = -1, double lambda = -1);

void ADMM_NonConv(Eigen::MatrixXd &Y, Eigen::MatrixXd &L, Eigen::MatrixXd &S, int tau, double epsilon, double rho, std::vector<unsigned int> &Position, int times, RecordForNonConv &record);

int ell(int n1, int n2, int n3);


void CalcHighlightIndex(Eigen::MatrixXd &Highlight, std::vector<unsigned int> &PositionIndex);

void FPCP(Eigen::MatrixXd &AY, Eigen::MatrixXd &Lowrank, Eigen::MatrixXd &Sparse, int loops = 2);

double AdaRPCA(std::vector<Eigen::MatrixXd> &BGRSequences, std::vector<Eigen::MatrixXd> &LowRankSequences, std::vector<Eigen::MatrixXd> &SparseSequences, Eigen::MatrixXd &Highlight);

#endif // RPCA_H

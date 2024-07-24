#ifndef MATRIX_DECOMPOSITION_H
#define MATRIX_DECOMPOSITION_H
#pragma once
#include<Eigen/Dense>
#include<Eigen/Core>
#include<mkl.h>
#include<iostream>
#define EIGEN_USING_MKL_ALL

typedef struct
{
    double *data;
    int rows;
    int cols;
}MyMatrix;
void QR_Economy_GetR(Eigen::MatrixXd &srcData, Eigen::MatrixXd &R);

void QR_Economy_GetQ(Eigen::MatrixXd &srcData, Eigen::MatrixXd &Q);

void QR_Economy(Eigen::MatrixXd &srcData, Eigen::MatrixXd &Q, Eigen::MatrixXd &R);

void QR_Economy_GetQ(Eigen::MatrixXd &Q);

void SVD_Economy(Eigen::MatrixXd &srcData, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V);

void randomizedSVD(Eigen::MatrixXd &srcData, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V, int r, int rEst, int nPower, Eigen::MatrixXd &warmStart);
#endif // MATRIX_DECOMPOSITION_H

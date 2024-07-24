#include"Matrix_Decomposition.h"

void QR_Economy_GetR(Eigen::MatrixXd &srcData, Eigen::MatrixXd &R)
{
    Eigen::MatrixXd A = srcData;
    int m = srcData.rows();
    int n = srcData.cols();
    int tau_length = m > n ? n : m;
    double *tau = (double*)MKL_malloc(sizeof(double)* tau_length, 64);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, (double*)A.data(), m, tau);
    if (m > n)
        R = A.block(0, 0, n, n).triangularView<Eigen::Upper>();
    else
        R = A.triangularView<Eigen::Upper>();
    MKL_free(tau);
}
void QR_Economy_GetQ(Eigen::MatrixXd &srcData, Eigen::MatrixXd &Q)
{
    Eigen::MatrixXd A = srcData;
    int m = srcData.rows();
    int n = srcData.cols();

    int tau_length = m > n ? n : m;
    double *tau = (double*)MKL_malloc(sizeof(double)* tau_length, 64);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, (double*)A.data(), m, tau);
    if (m > n)
    {
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, (double*)A.data(), m, tau);
        Q = A.block(0, 0, m, n);
    }
    else
    {
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, m, (double*)A.data(), m, tau);
        Q = A.block(0, 0, m, m);
    }
    MKL_free(tau);
}
void QR_Economy_GetQ(Eigen::MatrixXd &Q)
{
    int m = Q.rows();
    int n = Q.cols();

    int tau_length = m > n ? n : m;
    double *tau = (double*)MKL_malloc(sizeof(double)* tau_length, 64);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, (double*)Q.data(), m, tau);
    if (m > n)
    {
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, (double*)Q.data(), m, tau);
        Q = Q.block(0, 0, m, n);
    }
    else
    {
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, m, (double*)Q.data(), m, tau);
        Q = Q.block(0, 0, m, m);
    }
    MKL_free(tau);
}
void QR_Economy(Eigen::MatrixXd &srcData, Eigen::MatrixXd &Q, Eigen::MatrixXd &R)
{
    Eigen::MatrixXd A = srcData;
    int m = srcData.rows();
    int n = srcData.cols();
    int tau_length = m > n ? n : m;
    double *tau = (double*)MKL_malloc(sizeof(double)* tau_length, 64);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, (double*)A.data(), m, tau);
    if (m > n)
    {
        R = A.block(0, 0, n, n).triangularView<Eigen::Upper>();
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, (double*)A.data(), m, tau);
        Q = A.block(0, 0, m, n);
    }
    else
    {
        R = A.triangularView<Eigen::Upper>();
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, m, (double*)A.data(), m, tau);
        Q = A.block(0, 0, m, m);
    }
    MKL_free(tau);
}
void SVD_Economy(Eigen::MatrixXd &srcData, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V)
{
    Eigen::MatrixXd A = srcData;
    int m = A.rows();
    int n = A.cols();
    int length_superb = m > n ? n : m;
    //double *superb = (double*)MKL_malloc(sizeof(double)*length_superb * 1, 64);
    double *superb = new double[length_superb - 1];
    if (m > n)
    {
        U = Eigen::MatrixXd(m, n);
        S = Eigen::MatrixXd(1, n);
        V = Eigen::MatrixXd(n, n);

        //LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'A', m, n, (double*)A.data(), m, (double*)S.data(),
        //	(double*)U.data(), m, (double*)V.data(), n, superb);
        LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'O', m, n, (double*)A.data(), m, (double*)S.data(),
            NULL, m, (double*)V.data(), n);
        U = A;
    }
    else if (m == n)
    {
        U = Eigen::MatrixXd(m, m);
        S = Eigen::MatrixXd(1, n);
        V = Eigen::MatrixXd(n, n);

        /*LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, (double*)A.data(), m, (double*)S.data(),
            (double*)U.data(), m, (double*)V.data(), n, superb);*/
        LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', m, n, (double*)A.data(), m, (double*)S.data(),
            (double*)U.data(), m, (double*)V.data(), n);
    }
    else
    {
        U = Eigen::MatrixXd(m, m);
        S = Eigen::MatrixXd(1, m);
        V = Eigen::MatrixXd(m, n);

        /*	LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'S', m, n, (double*)A.data(), m, (double*)S.data(),
                (double*)U.data(), m, (double*)V.data(), m, superb);*/

        LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'O', m, n, (double*)A.data(), m, (double*)S.data(),
            (double*)U.data(), m, (double*)V.data(), m);
        V = A;
    }
    free(superb);
}
void Eigen_Randn(Eigen::MatrixXd &A, int m, int n)
{
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, NULL);
    A = Eigen::MatrixXd(m, n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, m * n, (double*)A.data(), 0, 1);
    vslDeleteStream(&stream);
}
void randomizedSVD(Eigen::MatrixXd &X, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V, int r, int rEst, int nPower, Eigen::MatrixXd &warmStart)
{
    //Eigen::MatrixXd X = srcData;
    int n = X.cols();
    Eigen::MatrixXd Q;
    if (warmStart.rows() == 0 || warmStart.cols() == 0)
    {
        Eigen_Randn(Q, n, rEst);
    }
    else
    {
        Q = warmStart;
        //std::cout << Q.rows() << Q.cols() << n << std::endl;
        assert(Q.rows() == n);
        if (Q.cols() > rEst)
            std::cout << "Warning: warmStartLarge" << std::endl;
        else
        {
            Eigen::MatrixXd temp;
            Eigen_Randn(temp, n, rEst - Q.cols());
            Eigen::MatrixXd C(Q.rows(), Q.cols() + temp.cols());
            C << Q, temp;
            Q = C;
        }
    }
    //Q = X * Q;
    Eigen::MatrixXd temp_Matrix(X.rows(), Q.cols());
    temp_Matrix.setZero();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, X.rows(), Q.cols(), X.cols(), 1, (double*)X.data(),
        X.rows(), (double*)Q.data(), Q.rows(), 1, (double*)temp_Matrix.data(), X.rows());
    Q = temp_Matrix;


    for (int j = 1; j < nPower - 1; j++)
    {
        QR_Economy_GetQ(Q);
        //Q = X.transpose()*Q;
        temp_Matrix = Eigen::MatrixXd(X.cols(), Q.cols());
        temp_Matrix.setZero();
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), Q.cols(), X.rows(), 1, (double*)X.data(),
            X.rows(), (double*)Q.data(), Q.rows(), 1, (double*)temp_Matrix.data(), X.cols());
        Q = temp_Matrix;


        QR_Economy_GetQ(Q);
        //Q = X * Q;
        temp_Matrix = Eigen::MatrixXd(X.rows(), Q.cols());
        temp_Matrix.setZero();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, X.rows(), Q.cols(), X.cols(), 1, (double*)X.data(),
            X.rows(), (double*)Q.data(), Q.rows(), 1, (double*)temp_Matrix.data(), X.rows());
        Q = temp_Matrix;

    }
    QR_Economy_GetQ(Q);


    //V = X.transpose()*Q;
    temp_Matrix = Eigen::MatrixXd(X.cols(), Q.cols());
    temp_Matrix.setZero();
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), Q.cols(), X.rows(), 1, (double*)X.data(),
        X.rows(), (double*)Q.data(), Q.rows(), 1, (double*)temp_Matrix.data(), X.cols());
    V = temp_Matrix;

    Eigen::MatrixXd R, VVt, VV, s;
    QR_Economy(V, V, R);
    Eigen::MatrixXd Rt = R.transpose();
    SVD_Economy(Rt, U, s, VVt);
    VV = VVt.transpose();

    //U = Q * U;
    temp_Matrix = Eigen::MatrixXd(Q.rows(), U.cols());
    temp_Matrix.setZero();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Q.rows(), U.cols(), Q.cols(), 1, (double*)Q.data(),
        Q.rows(), (double*)U.data(), U.rows(), 1, (double*)temp_Matrix.data(), Q.rows());
    U = temp_Matrix;


    //V = V * VV;
    temp_Matrix = Eigen::MatrixXd(V.rows(), VV.cols());
    temp_Matrix.setZero();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, V.rows(), VV.cols(), V.cols(), 1, (double*)V.data(),
        V.rows(), (double*)VV.data(), VV.rows(), 1, (double*)temp_Matrix.data(), V.rows());
    V = temp_Matrix;

    Eigen::VectorXd temp(s.rows()*s.cols());
    memcpy(temp.data(), s.data(), sizeof(double)*s.cols()*s.rows());
    S = Eigen::MatrixXd(temp.asDiagonal());

    //cv::Mat U_Mat, V_Mat, S_Mat;
    //cv::eigen2cv(U, U_Mat);
    //cv::eigen2cv(V, V_Mat);
    //cv::eigen2cv(S, S_Mat);

    Eigen::MatrixXd U_sub = U.block(0, 0, U.rows(), r);
    Eigen::MatrixXd V_sub = V.block(0, 0, V.rows(), r);
    Eigen::MatrixXd S_sub = S.block(0, 0, r, r);

    U = U_sub;
    V = V_sub;
    S = S_sub;
}

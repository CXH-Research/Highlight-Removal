#include<RPCA.h>


int ell(int n1, int n2, int n3)
{
    return MIN(MIN(n1, n2), n3);
}

inline double Abs(double x)
{
    return x > 0 ? x : -x;
}

void Shrinkage(Eigen::MatrixXd &matrix, double tau)
{
    double *data = matrix.data();
    int length = matrix.rows()*matrix.cols();
    double Curr;
    for (int i = 0; i < length; i++)
    {
        Curr = data[i];
        data[i] = SignOfEle(Curr)*MAX(Abs(Curr) - Abs(tau), 0.0);
    }
}

int SignOfEle(double pixel)
{
    return (pixel >= 0) ? (pixel > 0 ? 1 : 0) : -1;
}

int projNuclear(Eigen::MatrixXd &X, double tauL, Record &record)
{
    int rEst;
    if (record.oldRank == 0)//第一次执行奇异值收缩算法
        //rEst = MIN(MIN(X.rows(), X.cols()), 10);
        rEst = 10;
    else
        rEst = record.oldRank + 2;

    record.iteration += 1;
    int nr = (int)X.rows();
    int nc = (int)X.cols();
    int minN = MIN(nr, nc);

    int rankMax;
    switch (record.iteration)//所允许的最大的秩
    {
    case 1:
        rankMax = round(minN / 4);
        break;
    case 2:
        rankMax = round(minN / 2);
        break;
    default:
        rankMax = minN;
        break;
    }
    if (tauL == 0)//即不进行奇异值收缩
    {
        X.setZero();
        return rEst;
    }

    Eigen::MatrixXd opts;
    if (record.Vold.data())//Vold.rows()==0||Vold.cols()==0
        opts = record.Vold;

    Eigen::MatrixXd U, S, V, s;
    bool ok = false;
    double lambda_temp = 0;
    if (tauL < 0)
        tauL = abs(tauL);

    while (!ok)
    {
        rEst = MIN(rEst, rankMax);

        //X=U*S*VT
        randomizedSVD(X, U, S, V, rEst, ell(rEst + SVDOFFSET, nr, nc), SVDNPOWER, opts);

        double minVal = S((int)S.rows() - 1, (int)S.cols() - 1);
        ok = (minVal < tauL) || (rEst == rankMax);
        if (ok)
            break;
        rEst = rEst * 2;
    }
    int cnt = 0;
    for (int i = 0; i < S.rows(); i++)
    {
        if (S(i, i) > tauL)
            cnt++;
        else
            break;
    }
    //Shrinkage(S, tauL);
    rEst = MIN(cnt, rankMax);
    for (int i = 0; i < rEst; i++)
        S(i, i) -= tauL;

    if (rEst == 0)
        X.setZero();
    else
    {
        Eigen::MatrixXd U_sub = U.middleCols(0, rEst);
        Eigen::MatrixXd S_sub = S.block(0, 0, rEst, rEst);
        Eigen::MatrixXd V_sub = V.middleCols(0, rEst);

        Eigen::MatrixXd temp_Matrix(U_sub.rows(), S_sub.cols());
        temp_Matrix.setZero();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, U_sub.rows(), S_sub.cols(), U_sub.cols(), 1, (double*)U_sub.data(),
            U_sub.rows(), (double*)S_sub.data(), S_sub.rows(), 1, (double*)temp_Matrix.data(), U_sub.rows());

        X.setZero();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, temp_Matrix.rows(), V_sub.rows(), temp_Matrix.cols(), 1, (double*)temp_Matrix.data(),
            temp_Matrix.rows(), (double*)V_sub.data(), V_sub.rows(), 1, (double*)X.data(), temp_Matrix.rows());
    }
    record.oldRank = rEst;
    record.Vold = V.middleCols(0, rEst);
    return rEst;
}

double CalcFNorm(Eigen::MatrixXd &Input1, Eigen::MatrixXd &Input2)
{
    int nr = Input1.rows();
    int nc = Input1.cols();

    int length = nr * nc;
    double *data1 = Input1.data();
    double *data2 = Input2.data();
    double NormVal = 0;
    for (int i = 0; i < length; i++)
    {
        NormVal += data1[i] * data1[i];
        NormVal += data2[i] * data2[i];
    }
    return sqrt(NormVal);
}


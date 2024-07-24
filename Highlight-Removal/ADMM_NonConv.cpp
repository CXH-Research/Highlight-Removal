#include<RPCA.h>
#include"RPCA.h"
#define MINVALUE_GAP 0.1
#define SVDOFFSET 5
#define SVDNPOWER 1

bool Descend(double i, double  j);
double CalculateA(int tau, double epsilon, double ratio);

double CalculateEpsilon(double scaler, RecordForNonConv &record);
void CalculateGradient(Eigen::MatrixXd &X, Eigen::MatrixXd &Gradient, int tau, double epsilon, double a, double base, bool IsSigular);
int SingularValueShrinkage(Eigen::MatrixXd &S, Eigen::MatrixXd &Gradient, double thresh);
void ThresholdS(Eigen::MatrixXd &S, Eigen::MatrixXd &Gradient, std::vector<unsigned int> &indexdata, int tau, double offset);



void ADMM_NonConv(Eigen::MatrixXd &Y, Eigen::MatrixXd &L, Eigen::MatrixXd &S, int tau, double epsilon, double rho, std::vector<unsigned int> &Position, int times, RecordForNonConv &record)
{
    int m = Y.rows();
    int n = Y.cols();

    Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(m, n);
    Eigen::MatrixXd dS = Eigen::MatrixXd::Zero(m, n);

    Eigen::MatrixXd errHist(20, 1), Gradient;
    double nrmS;
    Eigen::MatrixXd temp;
    int rEst;

    cv::Mat LImg, SImg;
    int i;
    for (i = 0; i < 20; i++)
    {
        nrmS = S.norm();//对于矩阵计算的就是F范数
        temp = Y + 1. / rho * Lambda;
        L = temp - S;

        //cv::eigen2cv(L, LImg);
        rEst = projNuclear(L, 0.5, record, times);
        //cv::eigen2cv(L, LImg);

        dS = S;

        //cv::eigen2cv(S, SImg);
        S = temp - L;
        ThresholdS(S, Gradient, Position, tau, 0.005);
        //cv::eigen2cv(S, SImg);

        dS = S - dS;
        Lambda += rho * (Y - L - S);
        rho = MIN(rho*1.2, 5);
        errHist(i, 0) = dS.norm() / (1 + nrmS);

        if (i > 10)
        {
            if (abs(errHist(i, 0) - errHist(i - 1, 0)) < 1e-3)
            {
                //std::cout << i + 1 << std::endl;
                break;
            }
        }
    }
    //if (i == 20)
    //	std::cout << i << std::endl;
}

double CalculateA(int tau, double epsilon, double ratio)
{
    double Grad = 0.002;//作为梯度值快速上升的临界值
    double r = 1 / Grad;
    double a = pow(ratio, 4);
    double b = 2 * pow(ratio, 2)*pow(epsilon, 2) - 2 * r*ratio*pow(epsilon, 2)*(1 + pow(epsilon, 2));
    double c = pow(epsilon, 4);
    double delta = pow(b, 2) - 4 * a*c;//证明了是恒正
    return sqrt((-1 * b + sqrt(delta)) / (2 * a));
}
bool Descend(double i, double  j) { return (i > j); }
void CalcHighlightIndex(Eigen::MatrixXd &Highlight, std::vector<unsigned int> &PositionIndex)
{
//    cv::Mat ShowImg;
//    cv::eigen2cv(Highlight, ShowImg);
    double *HData = Highlight.data();
    int nr = Highlight.rows();
    int nc = Highlight.cols();
    PositionIndex.clear();
    double sum = 0;
    int cnt = 0;
    for (unsigned int i = 0, length = nr * nc; i < length; i++)
    {
        if (HData[i] == 1)
        {
            PositionIndex.push_back(i);
            cnt++;
        }
        sum += HData[i];
    }
    //std::cout << cnt << std::endl;
//	int a = 1;
}
int projNuclear(Eigen::MatrixXd &X, double tau, RecordForNonConv &record, int times)
{
    int rEst = 0;
    if (record.oldRank == 0)
        rEst = 10;
    else
        rEst = record.oldRank + 2;

    record.iteration += 1;

    int nr = X.rows();
    int nc = X.cols();
    int minN = nr < nc ? nr : nc;//MIN(nr,nc)

    int rankMax;
    switch (record.iteration)
    {
    case 1:
        rankMax = round(minN / (times * 4));
        break;
    case 2:
        rankMax = round(minN / (times * 2));
        break;
    default:
        rankMax = round(minN / times);
        break;
    }
    if (tau == 0)
        X.setZero();
    else
    {
        Eigen::MatrixXd opts;
        if (record.Vold.data() != NULL)
            opts = record.Vold;
        bool ok = false;
        Eigen::MatrixXd U, S, V;
        Eigen::MatrixXd Gradient;
        if (tau < 0)
            tau = abs(tau);
        while (!ok)
        {
            rEst = rEst > rankMax ? rankMax : rEst;
            randomizedSVD(X, U, S, V, rEst, ell(rEst + SVDOFFSET, nr, nc), SVDNPOWER, opts);
            if (record.ratio != 0)
            {
                double epsilon = CalculateEpsilon(S(0, 0), record);
                CalculateGradient(S, Gradient, 2, epsilon, 1, 0, 1);
            }
            else
                Gradient = Eigen::MatrixXd::Zero(S.rows(), S.cols());
            ok = (S(S.rows() - 1, S.cols() - 1) - Gradient(S.rows() - 1, S.cols() - 1) < tau) || (rEst == rankMax);
            if (ok)
                break;
            else
                rEst = 2 * rEst;
        }
        rEst = MIN(rankMax, SingularValueShrinkage(S, Gradient, tau));
//        int tempVal=SingularValueShrinkage(S, Gradient, tau);
//        rEst=rankMax<SingularValueShrinkage(S, Gradient, tau)?rankMax:SingularValueShrinkage(S, Gradient, tau);
        /*
        1、rankMax>rEst，则取rEst,所需收缩部分刚好等于已收缩部分
        2、rankMax<rEst, 则取rankMax，实际收缩过的奇异值多于所需，取所需即可
        */
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

            X = Eigen::MatrixXd(temp_Matrix.rows(), V_sub.rows());
            X.setZero();
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, temp_Matrix.rows(), V_sub.rows(), temp_Matrix.cols(), 1, (double*)temp_Matrix.data(),
                temp_Matrix.rows(), (double*)V_sub.data(), V_sub.rows(), 1, (double*)X.data(), temp_Matrix.rows());
        }

        record.oldRank = rEst;
        record.Vold = V.middleCols(0, rEst);
    }
    return rEst;
}
double CalculateEpsilon(double scaler, RecordForNonConv &record)
{
    double epsilon;
    double ratio = record.ratio;
    if (scaler == 2)
    {
        if (record.ratio > 0.7)
        {
            std::cerr << "Bad argument\n" << std::endl;
            exit(0);
        }
        else
            epsilon = sqrt(2 * pow(ratio, 4) / (2 - 4 * pow(ratio, 2)));
    }
    else
    {
        double delta = 8 * scaler*pow(ratio, 4) - 8 * scaler*pow(ratio, 2) + 4;
        if (scaler < 2)
            epsilon = sqrt(((2 - 2 * scaler*pow(ratio, 2)) - sqrt(delta)) / (2 * scaler - 4));
        else
        {
            double offset = sqrt((scaler - 2)*scaler) / (2 * scaler);
            double ratio1 = sqrt(0.5 - offset);
            ratio = MIN(ratio, ratio1);//在该范围，只要有限的可以被选择
            record.ratio = ratio;
            delta = 8 * scaler*pow(ratio, 4) - 8 * scaler*pow(ratio, 2) + 4;
            delta = MAX(0.0, delta);
            epsilon = sqrt(((2 - 2 * scaler*pow(ratio, 2)) - sqrt(delta)) / (2 * scaler - 4));
        }
    }
    return epsilon;
}

void CalculateGradient(Eigen::MatrixXd &X, Eigen::MatrixXd &Gradient, int tau, double epsilon, double a, double base, bool IsSigular)
{
    double maxVal;
    int nr = X.rows();
    int nc = X.cols();

    double currentVal;//当前要处理的像素值
    double offset = pow(epsilon, tau);
    double Multiplier = a * tau*offset*(1 + offset);

    double scaler;
    Gradient = Eigen::MatrixXd::Zero(nr, nc);
    double tempVal;
    if (IsSigular)//如果是奇异值矩阵
    {
        maxVal = X(0, 0);
        scaler = a / maxVal;
        for (int i = 0; i < nr; i++)
        {
            currentVal = X(i, i) *scaler;//归一化处理
            tempVal = pow(currentVal, tau - 1);
            Gradient(i, i) = Multiplier * tempVal / pow((tempVal*currentVal + offset), 2);
        }
    }
    else
    {
        int length = nr * nc;
        //maxVal = X.maxCoeff();
        maxVal = X.cwiseAbs().maxCoeff();
        scaler = a / maxVal;
        double *GradData = Gradient.data();
        double *srcData = X.data();
        for (int i = 0; i < length; i++)
        {
            currentVal = srcData[i] * scaler;
            //tempVal = pow(currentVal, tau - 1);
            GradData[i] = Multiplier * pow(currentVal, tau - 1) / pow((pow(currentVal, tau) + offset), 2) + SignOfEle(currentVal)*base;
        }
    }
    //cv::Mat show;
    //cv::eigen2cv(Gradient, show);
}

int SingularValueShrinkage(Eigen::MatrixXd &S, Eigen::MatrixXd &Gradient, double thresh)
{
    int nr = S.rows();
    int rEst = 0;
    for (int i = 0; i < nr; i++)
        if (S(i, i) - Gradient(i, i) > thresh)
        {
            S(i, i) -= Gradient(i, i) + thresh;
            rEst++;
        }
        else
            break;
    return rEst;
}

void ThresholdS(Eigen::MatrixXd &S, Eigen::MatrixXd &Gradient, std::vector<unsigned int> &indexdata, int tau, double offset)
{
    //首先用indexdata来将确定的高光位置像素值取出
    int length = indexdata.size();
    double *data = new double[length];
    double *SData = S.data();
    for (int i = 0; i < length; i++)
        data[i] = SData[indexdata.at(i)];

    std::sort(data, data + length, Descend);

    int cnt = 0;
    for (int i = 0; i < length; i++)
        if (data[i] > 0)
            cnt++;
        else
            break;

    double ratio_Min_Max = MAX(data[cnt - 1], 0.15) / data[0];
    double a = CalculateA(tau, 0.001, ratio_Min_Max);
    //std::cout << a << std::endl;

    CalculateGradient(S, Gradient, tau, 0.001, a, offset, 0);
    double *GradData = Gradient.data();
    for (int i = 0, length = S.rows()*S.cols(); i < length; i++)
        SData[i] = SignOfEle(SData[i])*MAX(abs(SData[i]) - abs(GradData[i]), 0);

    delete[]data;
}

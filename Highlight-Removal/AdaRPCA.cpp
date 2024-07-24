#include<RPCA.h>


void imbinarize(Eigen::MatrixXd &Matr)
{
    double *data = Matr.data();
    int nr = Matr.rows();
    int nc = Matr.cols();
    cv::Mat Img;
    cv::eigen2cv(Matr, Img);
    Img.convertTo(Img, CV_8UC1, 255);
    double thresh = cv::threshold(Img, Img, 0, 255, cv::THRESH_OTSU)*1.0f / 255;
    for (int i = 0, length = nr * nc; i < length; i++)
        data[i] = data[i] >= thresh ? 1 : 0;
}
int NNT(Eigen::MatrixXd &Matr)
{
    double *data = Matr.data();
    int nr = Matr.rows();
    int nc = Matr.cols();
    int cnt = 0;
    for (int i = 0, length = nr * nc; i < length; i++)
        if (data[i] != 0)
            cnt++;
    return cnt;
}
double AdaRPCA(std::vector<Eigen::MatrixXd> &BGRSequences, std::vector<Eigen::MatrixXd> &LowRankSequences, std::vector<Eigen::MatrixXd> &SparseSequences, Eigen::MatrixXd &Highlight)
{
    double lambdaS = 0.02;//初始值
    int nr = BGRSequences.at(0).rows();
    int nc = BGRSequences.at(0).cols();
    double PercentGT = NNT(Highlight)*1.0f / (nr*nc);
    double PercentDT;//分解得到的
    double minus;
    Eigen::MatrixXd Gray(nr, nc);
    double tic = cv::getTickCount();
    for (int i = 0; i < 3; i++)
        LagQN(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i), lambdaS);
    Rgb2Gray(SparseSequences, Gray);
    cv::Mat Show;
    cv::eigen2cv(Gray, Show);
    imbinarize(Gray);
    cv::eigen2cv(Gray, Show);

    PercentDT = NNT(Gray)*1.0f / (nr*nc);
    minus = PercentGT - PercentDT;

    bool Addition = false, Subtraction = false;
    int step = 0;
    std::cout << "minus:  " << minus << "   lambdaS:" << lambdaS << std::endl;
    while (minus > 0.005 || minus < -0.003)
    {
        std::cout << "minus:  " << minus << "   lambdaS:" << lambdaS << std::endl;
        if (minus < -0.003)//分解结果中稀疏成分多了，要增加lambda
        {
            if (Addition != true) { Addition = true; Subtraction = false; step += 1; }
            lambdaS += 0.0002;
        }
        else
        {
            if (Subtraction != true) { Subtraction = true; Addition = false; step += 1; }
            lambdaS -= 0.0005;
        }
        if (step == 5 || lambdaS<0.005)
            break;

        tic = cv::getTickCount();
        for (int i = 0; i < 3; i++)
        {
            LowRankSequences.at(i).setZero();
            SparseSequences.at(i).setZero();
            LagQN(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i), lambdaS);
        }
        Rgb2Gray(SparseSequences, Gray);
        cv::eigen2cv(Gray, Show);

        imbinarize(Gray);
        cv::eigen2cv(Gray, Show);

        PercentDT = NNT(Gray)*1.0f / (nr*nc);
        minus = PercentGT - PercentDT;
    }
    return (cv::getTickCount() - tic) / cv::getTickFrequency();
}

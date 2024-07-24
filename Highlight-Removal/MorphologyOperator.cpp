#include<MorphologyOperator.h>

void EigenDilate(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &dstMat, int  kernelRows, int kernelCols)
{
    assert(kernelRows % 2 == 1);
    assert(kernelCols % 2 == 1);
    assert(srcMat.data());
    if (dstMat.data() == NULL)
        dstMat = Eigen::MatrixXd(srcMat.rows(), srcMat.cols());
    double *SData = srcMat.data();
    double *DData = dstMat.data();

    int nr = srcMat.rows();
    int nc = srcMat.cols();

    int OffsetX = kernelCols / 2;
    int OffsetY = kernelRows / 2;
    memcpy(DData, SData, sizeof(double)*nr*nc);


    int BaseAddr, TempIndex;
    double MaxVal;
    std::vector<double> ROIPixels(kernelRows*kernelCols);

    for (int col = OffsetX; col < nc - OffsetX; col++)
    {
        for (int row = OffsetY; row < nr - OffsetY; row++)
        {
            MaxVal = 0;
            for (int i = 0; i < kernelCols; i++)
            {
                BaseAddr = (col - OffsetX + i)*nr + row - OffsetY;
                for (int j = 0; j < kernelRows; j++)
                    if (SData[BaseAddr + j] > MaxVal)
                        MaxVal = SData[BaseAddr + j];
            }
            DData[col*nr + row] = MaxVal;
        }
    }
}

void MyFilter2D(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &dstMat, Eigen::MatrixXd &kernel)
{
    int nr = srcMat.rows();
    int nc = srcMat.cols();
    if (dstMat.data() == NULL || (dstMat.rows() != nr && (dstMat.cols() != nc)))//直接赋值两种情况：目标矩阵为空；尺寸不一致；其实可以都归结于最后一种
    {
        dstMat = Eigen::MatrixXd(nr, nc);
        memcpy(dstMat.data(), srcMat.data(), sizeof(double)*nr*nc);
    }
    else
        memcpy(dstMat.data(), srcMat.data(), sizeof(double)*nr*nc);
    int nr_kernel = kernel.rows();
    int nc_kernel = kernel.cols();

    assert(nr_kernel % 2 == 1);
    assert(nc_kernel % 2 == 1);

    double *SData = srcMat.data();
    double *DData = dstMat.data();
    double *KData = kernel.data();
    memset(DData, 0, sizeof(double)*nr*nc);
    int rowIndex_src, colIndex_src, dstIndex;
    double sum = 0;
    int offset = 2;
    for (int col = nc_kernel / 2 + offset; col < nc - nc_kernel / 2 - offset; col++)
        for (int row = nr_kernel / 2 + offset; row < nr - nr_kernel / 2 - offset; row++)
        {
            dstIndex = col * nr + row;
            sum = 0;
            for (int j = -nc_kernel / 2; j <= nc_kernel / 2; j++)
                for (int i = -nr_kernel / 2; i <= nr_kernel / 2; i++)
                {
                    sum += SData[(col + j)*nr + (row + i)] * KData[(nc_kernel / 2 + j)*nr_kernel + (nr_kernel / 2 + i)];
                }
            DData[dstIndex] = sum;
        }
}

void CalcGradient(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &Grad)
{
    Eigen::MatrixXd kernel1(3, 3), kernel2(3, 3);
    kernel1 << 1, 2, 1, 0, 0, 0, -1, -2, -1;
    kernel2 << -1, 0, 1, -2, 0, 2, -1, 0, 1;

    Eigen::MatrixXd SG1, SG2;
    MyFilter2D(srcMat, SG1, kernel1);
    MyFilter2D(srcMat, SG2, kernel2);

    int nr = srcMat.rows();
    int nc = srcMat.cols();

    Grad = Eigen::MatrixXd(nr, nc);
    double *SGData1 = SG1.data();
    double *SGData2 = SG2.data();
    double *GData = Grad.data();

    for (int i = 0, length = nr * nc; i < length; i++)
        //GData[i] = sqrt(SGData1[i] * SGData1[i] + SGData2[i] * SGData2[i]);
        GData[i] = abs(SGData1[i]) + abs(SGData2[i]);

}

void CalcGradient(Eigen::MatrixXd &srcMat, std::vector<Eigen::MatrixXd> &Grad)
{
    std::vector<Eigen::MatrixXd> SobelKernel(4);
    for (int i = 0; i < 4; i++)
        SobelKernel[i] = Eigen::MatrixXd(3, 3);

    Eigen::MatrixXd kernel1(3, 3), kernel2(3, 3);
    Eigen::MatrixXd kernel3(3, 3), kernel4(3, 3);
    SobelKernel.at(0) << 1, 2, 1, 0, 0, 0, -1, -2, -1;
    SobelKernel.at(1) << -1, 0, 1, -2, 0, 2, -1, 0, 1;
    SobelKernel.at(2) << 2, 1, 0, 1, 0, -1, 0, -1, -2;
    SobelKernel.at(3) << 0, 1, 2, -1, 0, 1, -2, -1, 0;
#pragma omp parallel for
    for (int i = 0; i < 4; i++)
    {
        MyFilter2D(srcMat, Grad.at(i), SobelKernel.at(i));
        NormlizeMax(Grad.at(i));
    }

    int nr = srcMat.rows();
    int nc = srcMat.cols();

    double *SGData1 = Grad.at(0).data();
    double *SGData2 = Grad.at(1).data();
    double *SGData3 = Grad.at(2).data();
    double *SGData4 = Grad.at(3).data();

    double *GData = Grad.at(4).data();

    for (int i = 0, length = nr * nc; i < length; i++)
        GData[i] = sqrt(SGData1[i] * SGData1[i] + SGData2[i] * SGData2[i] + SGData3[i] * SGData3[i] + SGData4[i] * SGData4[i]);
    NormlizeMax(Grad.at(4));
}

double grayThreshOSTU(int *Hist, int length)
{
    double *CumHist = new double[length];//Cumulative histogram
    double *CumGraySum = new double[length];//Cumulative Summation of pixel Value
    CumHist[0] = Hist[0];
    CumGraySum[0] = 0;//0 * n=0
    for (int i = 1; i < length; i++)
    {
        CumHist[i] = CumHist[i - 1] + Hist[i];
        CumGraySum[i] = Hist[i] * i + CumGraySum[i - 1];
    }

    double w0 = 0, w1 = 0, u0 = 0, u1 = 0;
    double varValue = 0;
    int T = 0;
    for (int i = 0; i < length; i++)
    {
        w1 = CumHist[i];
        u1 = CumGraySum[i];

        if (w1 == 0)
            continue;

        u1 /= w1;//MeanValue of background
        w1 /= CumHist[length - 1];//ratio of background

        w0 = CumHist[length - 1] - CumHist[i];
        u0 = CumGraySum[length - 1] - CumGraySum[i];

        if (w0 == 0)
            break;

        u0 /= w0;
        w0 /= CumHist[length - 1];
        double varValueCurr = w0 * w1*(u1 - u0)*(u1 - u0);
        if (varValue < varValueCurr)
        {
            varValue = varValueCurr;
            T = i;
        }
    }
    delete[]CumHist;
    delete[]CumGraySum;
    return T * 1.0f / 255;
}

void imbinarize(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &dstMat)
{
    assert(srcMat.data());
    int nr = srcMat.rows();
    int nc = srcMat.cols();
    if (dstMat.data() == NULL)
        dstMat = Eigen::MatrixXd(nr, nc);

    double *SData = srcMat.data();
    int hist[256] = { 0 };
    for (int i = 0, length = nr * nc; i < length; i++)
        if (SData[i] < 0)
            hist[0]++;
        else if (SData[i] >= 1)
            hist[255]++;
        else
            hist[static_cast<int>(SData[i] * 255)]++;
    double thre = grayThreshOSTU(hist, 256);

    double *DData = dstMat.data();
    for (int i = 0, length = nr * nc; i < length; i++)
        DData[i] = SData[i] > thre ? 1 : 0;
}

std::vector<int>  findIndex(Eigen::MatrixXd &srcMat)
{
    assert(srcMat.data());
    int nr = srcMat.rows();
    int nc = srcMat.cols();

    std::vector<int> ValidIndex;
    double *SData = srcMat.data();
    for (int i = 0, length = nr * nc; i < length; i++)
        if (SData[i] != 0)
            ValidIndex.push_back(i);
    return ValidIndex;
}

std::vector<double>  findEle(Eigen::MatrixXd &srcMat)
{
    assert(srcMat.data());
    int nr = srcMat.rows();
    int nc = srcMat.cols();

    std::vector<double> ValidEle;
    double *SData = srcMat.data();
    for (int i = 0, length = nr * nc; i < length; i++)
        if (SData[i] != 0)
            ValidEle.push_back(SData[i]);
    return ValidEle;
}
std::vector<double>  findEleD(cv::Mat &srcMat)
{
    assert(!srcMat.empty());
    int nr = srcMat.rows;
    int nc = srcMat.cols;

    std::vector<double> ValidEle;
    for (int row = 0; row < nr; row++)
        for (int col = 0; col < nc; col++)
            if (srcMat.at<double>(row, col) != 0)
                ValidEle.push_back(srcMat.at<double>(row, col));
    return ValidEle;
}

std::vector<uchar>  findEle(cv::Mat &grayImg)
{
    assert(!grayImg.empty());
    int nr = grayImg.rows;
    int nc = grayImg.cols;

    std::vector<uchar> ValidEle;
    for (int row = 0; row < nr; row++)
        for (int col = 0; col < nc; col++)
            if (grayImg.at<uchar>(row, col) != 0)
                ValidEle.push_back(grayImg.at<uchar>(row, col));
    return ValidEle;
}

double graythresh(std::vector<double> &EleVec)
{
    assert(!EleVec.empty());
    int length = EleVec.size();
    int hist[256] = { 0 };
    for (int i = 0; i < length; i++)
        if (EleVec[i] < 0)
            hist[0]++;
        else if (EleVec[i] >= 1)
            hist[255]++;
        else
            hist[static_cast<int>(EleVec[i] * 255)]++;
    return grayThreshOSTU(hist, 256);
}

double graythresh(std::vector<uchar> &EleVec)
{
    assert(!EleVec.empty());
    int length = EleVec.size();
    int hist[256] = { 0 };
    for (int i = 0; i < length; i++)
        hist[static_cast<int>(EleVec[i])]++;
    return grayThreshOSTU(hist, 256);
}

void EigenThreshold(Eigen::MatrixXd &InputOutPutMatrix, double thre, double MaxVal)
{
    assert(InputOutPutMatrix.data());
    int nr = InputOutPutMatrix.rows();
    int nc = InputOutPutMatrix.cols();
    double *data = InputOutPutMatrix.data();
    for (int i = 0, length = nr * nc; i < length; i++)
        data[i] = data[i] >= thre ? MaxVal : 0;
}

void Rgb2Gray(std::vector<Eigen::MatrixXd>&BGRSequences, Eigen::MatrixXd &GrayMatrix)
{
    int nr = GrayMatrix.rows();
    int nc = GrayMatrix.cols();
    double *BData = BGRSequences.at(0).data();
    double *GData = BGRSequences.at(1).data();
    double *RData = BGRSequences.at(2).data();
    double *GrayData = GrayMatrix.data();
    for (int i = 0, length = nr * nc; i < length; i++)
        GrayData[i] = BData[i] * 0.114 + GData[i] * 0.587 + RData[i] * 0.299;
}

void AddWeight(std::vector<Eigen::MatrixXd> &Sequences1, std::vector<Eigen::MatrixXd> &Sequences2, std::vector<Eigen::MatrixXd> &Sequences3)
{
    assert(Sequences1.size() == Sequences2.size() && Sequences1.size() == Sequences3.size());
    assert(!Sequences1.empty());

    //忽略对每一个通道数据的断言处理
    int nr = Sequences1.at(0).rows();
    int nc = Sequences1.at(0).cols();
    int nChannels = Sequences1.size();
    double **DataGroup1 = new double *[nChannels];
    double **DataGroup2 = new double*[nChannels];
    double **DataGroup3 = new double*[nChannels];

    for (int i = 0; i < nChannels; i++)
    {
        DataGroup1[i] = Sequences1.at(i).data();
        DataGroup2[i] = Sequences2.at(i).data();
        DataGroup3[i] = Sequences1.at(i).data();
    }
    int length = nr * nc;
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < nChannels; j++)
            DataGroup3[j][i] = DataGroup1[j][i] + DataGroup2[j][i];
    }
    delete[]DataGroup1;
    delete[]DataGroup2;
    delete[]DataGroup3;
}

void AddWeight(std::vector<Eigen::MatrixXd> &InputSequences, Eigen::MatrixXd &Matr, std::vector<Eigen::MatrixXd> &OutputSequences, double theta)
{
    assert(InputSequences.size() == OutputSequences.size());
    assert(!InputSequences.empty());

    //忽略对每一个通道数据的断言处理
    int nr = InputSequences.at(0).rows();
    int nc = InputSequences.at(0).cols();
    int nChannels = InputSequences.size();
    double **DataGroup1 = new double *[nChannels];
    double **DataGroup2 = new double*[nChannels];

    for (int i = 0; i < nChannels; i++)
    {
        DataGroup1[i] = InputSequences.at(i).data();
        DataGroup2[i] = OutputSequences.at(i).data();
    }
    double *MatrData = Matr.data();
    int length = nr * nc;
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < nChannels; j++)
            DataGroup2[j][i] = (theta)*DataGroup1[j][i] + MatrData[i] * (1 - theta);
    }
    delete[]DataGroup1;
    delete[]DataGroup2;
}

void NormlizeMax(Eigen::MatrixXd &Matr)
{
    double *data = Matr.data();
    int nr = Matr.rows();
    int nc = Matr.cols();

    double MaxVal = data[0];

    for (int i = 1, length = nr * nc; i < length; i++)
        if (data[i] > MaxVal)
            MaxVal = data[i];
    if (MaxVal == 0)
        return;
    for (int i = 1, length = nr * nc; i < length; i++)
        data[i] /= MaxVal;
}

void MSVTuning(std::vector<Eigen::MatrixXd> &MSVSequences, Eigen::MatrixXd &Grad, double theta)
{
    assert(Grad.data());
    assert(MSVSequences.size() == 3);
    assert(MSVSequences.at(0).data());
    assert(MSVSequences.at(1).data());
    assert(MSVSequences.at(2).data());

    double *GradData = Grad.data();
    double *VData = MSVSequences.at(2).data();
    double *SData = MSVSequences.at(1).data();
    double *MData = MSVSequences.at(0).data();
    int nr = Grad.rows();
    int nc = Grad.cols();
    double tempVal;
    for (int i = 0, length = nr * nc; i < length; i++)
    {
        //tempVal = (1 - theta)*GradData[i];
        //if (VData[i] * theta + tempVal == 0)
        //	SData[i] = 0;
        //else
        //	SData[i] = (SData[i] * VData[i])*theta / (VData[i] * theta + tempVal);
        //VData[i] = VData[i] * theta + tempVal;
        //MData[i] = MData[i] * theta + tempVal;
        VData[i] = MIN(1.0, GradData[i] * (1 - theta) + VData[i]);
        //VData[i] = MIN(1, GradData[i] * 0.2 + 0.8*VData[i]);
    }
    //std::vector<cv::Mat> BGRImgVec(3);
    //for (int i = 0; i < BGRImgVec.size(); i++)
    //	cv::eigen2cv(MSVSequences.at(i), BGRImgVec.at(i));
}

void SequencesSobel(std::vector<Eigen::MatrixXd> &Input, std::vector<Eigen::MatrixXd> &OutPut)
{
    assert(Input.size() == OutPut.size());
    Eigen::MatrixXd sobeloperator(3, 3);
    sobeloperator << 1, 2, 1, 0, 0, 0, -1, -2, -1;
#pragma omp parallel for
    for (int i = 0; i < Input.size(); i++)
        MyFilter2D(Input.at(i), OutPut.at(i), sobeloperator);


    double *IBData = Input.at(0).data();
    double *IGData = Input.at(1).data();
    double *IRData = Input.at(2).data();

    double *OBData = OutPut.at(0).data();
    double *OGData = OutPut.at(1).data();
    double *ORData = OutPut.at(2).data();

    int length = Input.at(0).rows()*Input.at(0).cols();
    double theta = 0.1;
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        OBData[i] = OBData[i] * theta + IBData[i] * (1 - theta);
        OGData[i] = OGData[i] * theta + IGData[i] * (1 - theta);
        ORData[i] = ORData[i] * theta + IRData[i] * (1 - theta);
    }
}

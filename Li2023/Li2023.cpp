#include<Eigen/Dense>
#include<Eigen/Core>
#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<mkl.h>
#include<iostream>
#include<vector>
#include<dirent.h>
#include<string>
#include<omp.h>
#include<ostream>
#include<stdexcept>

class progressbar {

    public:
      // default destructor
      ~progressbar()                             = default;

      // delete everything else
      progressbar           (progressbar const&) = delete;
      progressbar& operator=(progressbar const&) = delete;
      progressbar           (progressbar&&)      = delete;
      progressbar& operator=(progressbar&&)      = delete;

      // default constructor, must call set_niter later
      inline progressbar();
      inline progressbar(int n, bool showbar=true, std::ostream& out=std::cerr);

      // reset bar to use it again
      inline void reset();
     // set number of loop iterations
      inline void set_niter(int iter);
      // chose your style
      inline void set_done_char(const std::string& sym) {done_char = sym;}
      inline void set_todo_char(const std::string& sym) {todo_char = sym;}
      inline void set_opening_bracket_char(const std::string& sym) {opening_bracket_char = sym;}
      inline void set_closing_bracket_char(const std::string& sym) {closing_bracket_char = sym;}
      // to show only the percentage
      inline void show_bar(bool flag = true) {do_show_bar = flag;}
      // set the output stream
      inline void set_output_stream(const std::ostream& stream) {output.rdbuf(stream.rdbuf());}
      // main function
      inline void update();

    private:
      int progress;
      int n_cycles;
      int last_perc;
      bool do_show_bar;
      bool update_is_called;

      std::string done_char;
      std::string todo_char;
      std::string opening_bracket_char;
      std::string closing_bracket_char;

      std::ostream& output;
};

inline progressbar::progressbar() :
    progress(0),
    n_cycles(0),
    last_perc(0),
    do_show_bar(true),
    update_is_called(false),
    done_char("#"),
    todo_char(" "),
    opening_bracket_char("["),
    closing_bracket_char("]"),
    output(std::cerr) {}

inline progressbar::progressbar(int n, bool showbar, std::ostream& out) :
    progress(0),
    n_cycles(n),
    last_perc(0),
    do_show_bar(showbar),
    update_is_called(false),
    done_char("#"),
    todo_char(" "),
    opening_bracket_char("["),
    closing_bracket_char("]"),
    output(out) {}

inline void progressbar::reset() {
    progress = 0,
    update_is_called = false;
    last_perc = 0;
    return;
}

inline void progressbar::set_niter(int niter) {
    if (niter <= 0) throw std::invalid_argument(
        "progressbar::set_niter: number of iterations null or negative");
    n_cycles = niter;
    return;
}

inline void progressbar::update() {

    if (n_cycles == 0) throw std::runtime_error(
            "progressbar::update: number of cycles not set");

    if (!update_is_called) {
        if (do_show_bar == true) {
            output << opening_bracket_char;
            for (int _ = 0; _ < 50; _++) output << todo_char;
            output << closing_bracket_char << " 0%";
        }
        else output << "0%";
    }
    update_is_called = true;

    int perc = 0;

    // compute percentage, if did not change, do nothing and return
    perc = progress*100./(n_cycles-1);
    if (perc < last_perc) return;

    // update percentage each unit
    if (perc == last_perc + 1) {
        // erase the correct  number of characters
        if      (perc <= 10)                output << "\b\b"   << perc << '%';
        else if (perc  > 10 and perc < 100) output << "\b\b\b" << perc << '%';
        else if (perc == 100)               output << "\b\b\b" << perc << '%';
    }
    if (do_show_bar == true) {
        // update bar every ten units
        if (perc % 2 == 0) {
            // erase closing bracket
            output << std::string(closing_bracket_char.size(), '\b');
            // erase trailing percentage characters
            if      (perc  < 10)               output << "\b\b\b";
            else if (perc >= 10 && perc < 100) output << "\b\b\b\b";
            else if (perc == 100)              output << "\b\b\b\b\b";

            // erase 'todo_char'
            for (int j = 0; j < 50-(perc-1)/2; ++j) {
                output << std::string(todo_char.size(), '\b');
            }

            // add one additional 'done_char'
            if (perc == 0) output << todo_char;
            else           output << done_char;

            // refill with 'todo_char'
            for (int j = 0; j < 50-(perc-1)/2-1; ++j) output << todo_char;

            // readd trailing percentage characters
            output << closing_bracket_char << ' ' << perc << '%';
        }
    }
    last_perc = perc;
    ++progress;
    output << std::flush;

    return;
}

#define EIGEN_USING_MKL_ALL

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

#define MINVALUE_GAP 0.1
#define SVDOFFSET 5
#define SVDNPOWER 1

typedef struct
{
    double *data;
    int rows;
    int cols;
}MyMatrix;
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

void InitMatrixVec(std::vector<Eigen::MatrixXd> &Sequences, int size, int nr, int nc)
{
    Sequences.clear();
    for (int i = 0; i < size; i++)
        Sequences.push_back(Eigen::MatrixXd(nr, nc));
}

void SplitImg(cv::Mat &RGBImg, std::vector<Eigen::MatrixXd> &BGRSequences)
{
    assert(BGRSequences.size() == 3);
    assert(RGBImg.type() == CV_64FC3);
    int nr = RGBImg.rows;
    int nc = RGBImg.cols;
    int length = nr * nc;
    int colIndex, rowIndex;

    double *RData = BGRSequences.at(2).data();
    double *GData = BGRSequences.at(1).data();
    double *BData = BGRSequences.at(0).data();

    cv::Vec3d pixel;
    //#pragma omp parallel for private(colIndex, rowIndex, pixel)
    for (int i = 0; i < length; i++)
    {
        colIndex = i / nr;//In Eigen Data, colMajor
        rowIndex = i % nr;
        pixel = RGBImg.at<cv::Vec3d>(rowIndex, colIndex);
        BData[i] = pixel[0];
        GData[i] = pixel[1];
        RData[i] = pixel[2];
    }
}
void RGB2MSV(std::vector<Eigen::MatrixXd> &BGRSequences, std::vector<Eigen::MatrixXd> &MSVSequences)
{
    assert(BGRSequences.size() == 3);
    assert(MSVSequences.size() == 3);
    double *BData = BGRSequences.at(0).data();
    double *GData = BGRSequences.at(1).data();
    double *RData = BGRSequences.at(2).data();

    double *MinData = MSVSequences.at(0).data();
    double *SData = MSVSequences.at(1).data();
    double *VData = MSVSequences.at(2).data();

    double MinVal, MaxVal;
    int nr = BGRSequences.at(0).rows();
    int nc = BGRSequences.at(0).cols();
    int length = nr * nc;
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        VData[i] = MAX(MAX(RData[i], GData[i]), BData[i]);
        MinData[i] = MIN(MIN(RData[i], GData[i]), BData[i]);
        if (VData[i] == 0)
            SData[i] = 0;
        else
            SData[i] = (VData[i] - MinData[i]) / VData[i];
    }
}

void CalculateMeanStd(Eigen::MatrixXd &matrix, double &MeanVal, double &stdVal)
{
    int nr = matrix.rows();
    int nc = matrix.cols();

    MeanVal = 0;
    stdVal = 0;
    double *MData = matrix.data();
    int length = nr * nc;
    for (int i = 0; i < length; i++)
    {
        MeanVal += MData[i];
        stdVal += MData[i] * MData[i];
    }
    stdVal /= length;
    MeanVal /= length;
    stdVal = stdVal - MeanVal * MeanVal;
    stdVal = sqrt(stdVal);
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

void IsolateWithMSV(std::vector<Eigen::MatrixXd> &MSVSequences, double threshold_s, double threshold_v, Eigen::MatrixXd &Highlight)
{
    assert(MSVSequences.size() == 3);
    int nr = MSVSequences.at(0).rows();
    int nc = MSVSequences.at(0).cols();
    double *MData = MSVSequences.at(0).data();
    double *SData = MSVSequences.at(1).data();
    double *VData = MSVSequences.at(2).data();
    double *HData = Highlight.data();

    int Hist[256] = { 0 };
    int NumberOfHighlight_Stage1 = 0;
    int NumberOfHighlight_Stage2 = 0;
    for (int i = 0, length = nr * nc; i < length; i++)
    {
        if (SData[i] <= threshold_s && VData[i] >= threshold_v)
        {
            HData[i] = 1;
            Hist[static_cast<uchar>(MData[i] * 255)]++;
            NumberOfHighlight_Stage1++;
        }
        else
            HData[i] = 0;
    }
    //cv::Mat HighlightImg;
    //cv::eigen2cv(Highlight, HighlightImg);
    double thresh = grayThreshOSTU(Hist, 256);
    double *TempData = new double[nr*nc];//新建一个数组
    for (int i = 0, length = nr * nc; i < length; i++)
    {
        if (HData[i] == 1 && MData[i] >= thresh)
        {
            TempData[i] = 1;
            NumberOfHighlight_Stage2++;
        }
        else
            TempData[i] = 0;
    }
    if (((NumberOfHighlight_Stage1 - NumberOfHighlight_Stage2)*1.0f / (nr*nc)) > PERCENTOFTHRESHOLD)//如果经过最小值通道的处理之后，高光含量变化明显，说明之前检测多了
        memcpy(HData, TempData, sizeof(double)*nr*nc);
    //cv::eigen2cv(Highlight, HighlightImg);
    delete[]TempData;
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

void imbinarize2(Eigen::MatrixXd &srcMat, Eigen::MatrixXd &dstMat)
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

void HighlightDetectionGrad(std::vector<Eigen::MatrixXd> &MSVSequences, Eigen::MatrixXd &Grad, Eigen::MatrixXd &Highlight, int Method)
{
    double MeanVal_S, MeanVal_V;
    double StdVal_S, StdVal_V;
    double threshold_s, threshold_v;
    cv::Mat ShowImg, HighlightImg;
    cv::Mat HighlightDilateImg;
    cv::Mat GradImg;
    CalculateMeanStd(MSVSequences.at(1), MeanVal_S, StdVal_S);
    CalculateMeanStd(MSVSequences.at(2), MeanVal_V, StdVal_V);
    threshold_s = MAX(MeanVal_S - 0.9*StdVal_S, 0.3);
    threshold_v = MIN(MeanVal_V + 1.6*StdVal_V, 0.8);
    IsolateWithMSV(MSVSequences, threshold_s, threshold_v, Highlight);//先用MSV三个属性对高光进行检测
    cv::eigen2cv(Grad, GradImg);
    cv::eigen2cv(Highlight, HighlightImg);
    /*
    对接下来利用梯度的流程整理如下：
        1.对初步的高光检测结果进行膨胀处理，并提取该区域内的梯度信息
        2.取该区域内的非0梯度值，用OTSU求取阈值，并进行阈值分割
        3.对阈值分割的结果进行膨胀，使得边缘闭合
    */
    //1.对初步的高光检测结果进行膨胀处理，并提取该区域内的梯度信息
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(HighlightImg, HighlightDilateImg, kernel);
    cv::Mat EdgeImg;
    cv::multiply(HighlightDilateImg, GradImg, EdgeImg);
    //EigenDilate(Highlight, HighlightDilate, 7, 7);
    //用膨胀后的初步高光检测结果提取在该部分内所有像素点的梯度信息

    //		2.取该区域内的非0梯度值，用OTSU求取阈值，并进行阈值分割
    std::vector<double> EleVec = findEleD(EdgeImg);
    if (EleVec.empty())
    {
        Highlight.setZero();
        return;
    }
    cv::threshold(EdgeImg, EdgeImg, graythresh(EleVec), 1, cv::THRESH_BINARY);


    //对强边缘进行膨胀处理，目的是使得边缘尽可能闭合
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
    cv::dilate(EdgeImg, ShowImg, kernel);
    Eigen::MatrixXd ValidEdgeDilate;
    if (Method == METHODGRADCONTOURS)
    {
        //轮廓检测的结果放置于ShowIMg内，并将该矩阵重新存到Eigen矩阵中
        //cv::eigen2cv(HighlightDilate, ShowImg);
        ShowImg.convertTo(ShowImg, CV_8UC1, 255);//opencv要求数据使用的是CV_8UC1

        std::vector<std::vector<cv::Point>>contours;
        std::vector<cv::Vec4i> hierarchy;
        findContours(ShowImg, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
        int idx = 0;
        ShowImg.setTo(0);
        for (; idx < contours.size(); idx++)
        {
            drawContours(ShowImg, contours, idx, cv::Scalar(255), cv::FILLED, 8, hierarchy);
        }
        //cv::erode(ShowImg, ShowImg, kernel);
        ShowImg.convertTo(ShowImg, CV_64FC1, 1.0f / 255);
        cv::cv2eigen(ShowImg, ValidEdgeDilate);
        imbinarize2(ValidEdgeDilate, ValidEdgeDilate);//这里挺奇怪的，不处理的话，高光像素不是1
        Highlight = Highlight.array()*ValidEdgeDilate.array();
    }
    else
    {
        //此类方法的问题：膨胀之后不见得能填充的了高光边缘所组成的闭合区间
        cv::dilate(ShowImg, ShowImg, kernel);
        cv::cv2eigen(ShowImg, ValidEdgeDilate);
        Highlight = Highlight.array()*ValidEdgeDilate.array();
    }
}

void IsolateWithSV(Eigen::MatrixXd &Saturation, double threshold_s, Eigen::MatrixXd &ValueChan, double threshold_v, Eigen::MatrixXd &Highlight)
{
    int nr = Saturation.rows();
    int nc = Saturation.cols();
    double *SData = Saturation.data();
    double *VData = ValueChan.data();
    double *HData = Highlight.data();
    for (int i = 0, length = nr * nc; i < length; i++)
        HData[i] = (SData[i]<=threshold_s&&VData[i]>=threshold_v) ? 1 : 0;
}

void HighlightDetection(std::vector<Eigen::MatrixXd> &MSVSequences, Eigen::MatrixXd &Highlight, int Methods)
{
    double MeanVal_S, MeanVal_V;
    double StdVal_S, StdVal_V;
    double threshold_s, threshold_v;
    switch(Methods)
    {
    case ADAPTIVEBITHRESHOLDS:
        CalculateMeanStd(MSVSequences.at(1), MeanVal_S, StdVal_S);
        CalculateMeanStd(MSVSequences.at(2), MeanVal_V, StdVal_V);
        threshold_s = MAX(MeanVal_S - 0.9*StdVal_S, 0.3);
        threshold_v = MIN(MeanVal_V + 1.6*StdVal_V, 0.8);
        IsolateWithSV(MSVSequences.at(1), threshold_s, MSVSequences.at(2), threshold_v, Highlight);
        break;
    case MSVTHRESHOLDS:
        CalculateMeanStd(MSVSequences.at(1), MeanVal_S, StdVal_S);
        CalculateMeanStd(MSVSequences.at(2), MeanVal_V, StdVal_V);
        threshold_s = MAX(MeanVal_S - 0.9*StdVal_S, 0.3);
        threshold_v = MIN(MeanVal_V + 1.6*StdVal_V, 0.8);
        IsolateWithMSV(MSVSequences, threshold_s, threshold_v, Highlight);
        break;
    case EMPERICALBITHRESHOLDS:
        IsolateWithSV(MSVSequences.at(1), EMPERICALTHRESHOLD_S, MSVSequences.at(2), EMPERICALTHRESHOLD_V, Highlight);
        break;
    default:
        IsolateWithSV(MSVSequences.at(1), EMPERICALTHRESHOLD_S, MSVSequences.at(2), EMPERICALTHRESHOLD_V, Highlight);
        break;
    }
}


void InitData(std::vector<Eigen::MatrixXd> &BGRSequences, std::vector<Eigen::MatrixXd> &LowRankSequences, \
              std::vector<Eigen::MatrixXd> &SparseSequences, Eigen::MatrixXd &Highlight)
{
    int nr = BGRSequences[0].rows();
    int nc = BGRSequences[0].cols();
    int nChannels = BGRSequences.size();

    double **DataMatrix = new double *[nChannels];
    double **BGRData = new double*[nChannels];
    double **LowRankData = new double *[nChannels];
    double **SparseData = new double*[nChannels];
    for (int i = 0; i < nChannels; i++)
    {
        DataMatrix[i] = new double[nr];
        BGRData[i] = BGRSequences[i].data();
        LowRankData[i] = LowRankSequences[i].data();
        SparseData[i] = SparseSequences[i].data();
    }
    double *HData = Highlight.data();

#pragma omp parallel for
    for (int i = 0; i < nChannels; i++)//逐通道操作
    {
        double MedianVal;
        for (int j = 0; j < nc; j++)//对每一列进行操作
        {
            memcpy(DataMatrix[i], BGRData[i] + j * nr, sizeof(double)*nr);
            std::sort(DataMatrix[i], DataMatrix[i] + nr);
            MedianVal = (nr % 2 == 0) ? (DataMatrix[i][nr / 2] + DataMatrix[i][nr / 2 - 1]) / 2 : DataMatrix[i][nr / 2];

            for (int k = 0; k < nr; k++)
            {
                LowRankData[i][j*nr + k] = MedianVal;
                SparseData[i][j*nr + k] = (BGRData[i][j*nr + k] - MedianVal)*HData[j*nr + k] * 0.8;
            }
        }
    }

    for (int i = 0; i < nChannels; i++)
        delete[]DataMatrix[i];
    delete[]BGRData;
    delete[]LowRankData;
    delete[]SparseData;
    delete[]DataMatrix;
}

void InitData(std::vector<Eigen::MatrixXd> &BGRSequences, std::vector<Eigen::MatrixXd> &LowRankSequences, \
    std::vector<Eigen::MatrixXd> &SparseSequences)
{
    int nr = BGRSequences[0].rows();
    int nc = BGRSequences[0].cols();
    int nChannels = BGRSequences.size();

    double **DataMatrix = new double *[nChannels];
    double **BGRData = new double*[nChannels];
    double **LowRankData = new double *[nChannels];
    double **SparseData = new double*[nChannels];
    for (int i = 0; i < nChannels; i++)
    {
        DataMatrix[i] = new double[nr];
        BGRData[i] = BGRSequences[i].data();
        LowRankData[i] = LowRankSequences[i].data();
        SparseData[i] = SparseSequences[i].data();
    }

#pragma omp parallel for
    for (int i = 0; i < nChannels; i++)//逐通道操作
    {
        double MedianVal;
        for (int j = 0; j < nc; j++)//对每一列进行操作
        {
            memcpy(DataMatrix[i], BGRData[i] + j * nr, sizeof(double)*nr);
            std::sort(DataMatrix[i], DataMatrix[i] + nr);
            MedianVal = (nr % 2 == 0) ? (DataMatrix[i][nr / 2] + DataMatrix[i][nr / 2 - 1]) / 2 : DataMatrix[i][nr / 2];

            for (int k = 0; k < nr; k++)
            {
                LowRankData[i][j*nr + k] = MedianVal;
                SparseData[i][j*nr + k] = (BGRData[i][j*nr + k] - MedianVal);
            }
        }
    }

    for (int i = 0; i < nChannels; i++)
        delete[]DataMatrix[i];
    delete[]BGRData;
    delete[]LowRankData;
    delete[]SparseData;
    delete[]DataMatrix;
}

void Blocking(std::vector<Eigen::MatrixXd> &ImgMatrix, std::vector<Eigen::MatrixXd> &Batches, int BatchRows, int BatchCols, int nr, int nc)
{
    /*
    srcImg 浮点型数据，源图像
    RGBBatches 将图像分块后数据组织为Eigen对象，用vector对所有的块进行封装
    BatchesRows 纵向上图像分几个块
    BatchCols 横向上图像分几个块
    nr 每一个块的高度
    nc 每一个块的宽度
    */
    //ratio 对RGB图像中非高光元素做逐像素乘法
    int total = nr * nc;//每一个分块的像素个数
    int sz = BatchRows * BatchCols;//将图像的单个通道分多少块
    int nChannels = ImgMatrix.size();
    assert(nChannels >= 1);
    assert(nChannels*sz == Batches.size());

    int dstRow, dstCol;//小图像块中的行列号
    int srcRow, srcCol;//原图像中的行列号
    int srcBatchRow, srcBatchCol;//图像块 行的序号、列的序号
    int dstIndex;

    int srcnr = ImgMatrix[0].rows();
    int srcnc = ImgMatrix[0].cols();

    double **ImgMatData = new double *[nChannels];
    double **BatchesData = new double*[Batches.size()];


    for (int i = 0; i < nChannels; i++)//分块时同一块的多个通道紧挨着放，比如，如果是3通道的矩阵（RGB的矩阵），R1,G1,B1;R2,G2,B2;
    {
        ImgMatData[i] = ImgMatrix[i].data();
        for (int j = 0; j < sz; j++)
        {
            BatchesData[j*nChannels + i] = Batches[j*nChannels + i].data();
        }
    }

    //#pragma omp parallel for private(dstCol,dstRow,srcRow, srcCol,srcBatchRow, srcBatchCol,dstIndex, pixel)
    for (int i = 0; i < total; i++)//遍历图像块中的每一个像素
    {
        //在每一个图像块中的行列序号
        dstCol = i / nr;//Eigen中元素按列排放
        dstRow = i % nr;

        for (int j = 0; j < sz; j++)//遍历每一个块
        {
            srcBatchRow = j / BatchCols;//从行上看，第几块 （按行排放）
            srcBatchCol = j % BatchCols;

            srcRow = srcBatchRow * nr + dstRow;
            srcCol = srcBatchCol * nc + dstCol;
            dstIndex = srcCol * srcnr + srcRow;

            for (int k = 0; k < nChannels; k++)
                BatchesData[j*nChannels + k][i] = ImgMatData[k][dstIndex];
        }
    }
    delete[]ImgMatData;
    delete[]BatchesData;
}
void Blocking(Eigen::MatrixXd &ImgMatrix, std::vector<Eigen::MatrixXd> &Batches, int BatchRows, int BatchCols, int nr, int nc)
{
    /*
    srcImg 浮点型数据，源图像
    RGBBatches 将图像分块后数据组织为Eigen对象，用vector对所有的块进行封装
    BatchesRows 纵向上图像分几个块
    BatchCols 横向上图像分几个块
    nr 每一个块的高度
    nc 每一个块的宽度
    */
    //ratio 对RGB图像中非高光元素做逐像素乘法
    int total = nr * nc;//每一个分块的像素个数
    int sz = BatchRows * BatchCols;//将图像的单个通道分多少块
    assert(sz == Batches.size());

    int dstRow, dstCol;//小图像块中的行列号
    int srcRow, srcCol;//原图像中的行列号
    int srcBatchRow, srcBatchCol;//图像块 行的序号、列的序号
    int dstIndex;

    //原数据的尺寸
    int srcnr = ImgMatrix.rows();
    int srcnc = ImgMatrix.cols();

    double *ImgMatData = ImgMatrix.data();
    double **BatchesData = new double*[Batches.size()];


    for (int i = 0; i < sz; i++)
        BatchesData[i] = Batches.at(i).data();

    //#pragma omp parallel for private(dstCol,dstRow,srcRow, srcCol,srcBatchRow, srcBatchCol,dstIndex, pixel)
    for (int i = 0; i < total; i++)//遍历图像分块中的每一个像素
    {
        //在每一个图像分块中的行列序号
        dstCol = i / nr;//Eigen中元素按列排放
        dstRow = i % nr;

        for (int j = 0; j < sz; j++)//遍历每一个块
        {
            srcBatchRow = j / BatchCols;//从行上看，第几块 （按行排放）
            srcBatchCol = j % BatchCols;

            srcRow = srcBatchRow * nr + dstRow;
            srcCol = srcBatchCol * nc + dstCol;
            dstIndex = srcCol * srcnr + srcRow;
            BatchesData[j][i] = ImgMatData[dstIndex];
        }
    }
    delete[]BatchesData;
}


void Merging(std::vector<Eigen::MatrixXd> &Batches, cv::Mat &RGBImg, int BatchRows, int BatchCols, int nr, int nc, double ratio)
{
    int total = nr * nc;//每一个分块的像素个数
    int sz = Batches.size() / 3;//将图像分多少块 (除以3 ： 每一个块分3个通道）
    int srcRow, srcCol;//图像子块中的行列位置
    int dstRow, dstCol;//整合好的图像位置
    int dstIndex;
    int dstBatchRow, dstBatchCol;//图像块 行的序号、列的序号
    int dstnr = BatchRows * nr;
    int dstnc = BatchCols * nc;
    double **BatchesData = new double*[sz * 3];
    for (int i = 0; i < sz; i++)
    {
        BatchesData[3 * i] = Batches[3 * i].data();
        BatchesData[3 * i + 1] = Batches[3 * i + 1].data();
        BatchesData[3 * i + 2] = Batches[3 * i + 2].data();

    }
    cv::Vec3d pixel;

    for (int i = 0; i < total; i++)
    {
        srcCol = i / nr;
        srcRow = i % nr;
        for (int j = 0; j < sz; j++)
        {
            dstBatchRow = j / BatchCols;//从行上看，第几块 （按行排放）
            dstBatchCol = j % BatchCols;
            dstRow = dstBatchRow * nr + srcRow;
            dstCol = dstBatchCol * nc + srcCol;

            pixel[0] = BatchesData[3 * j][i] / ratio;
            pixel[1] = BatchesData[3 * j + 1][i] / ratio;
            pixel[2] = BatchesData[3 * j + 2][i] / ratio;

            RGBImg.at<cv::Vec3d>(dstRow, dstCol) = pixel;
        }
    }
}

inline int IsHighlightPixel(uchar pixel)
{
    return pixel == 255 ? 1 : 0;
    //if (pixel == 255)
    //	return 1;
    //else
    //	return 0;
}

void HighlightRefine(cv::Mat &SparseImg, cv::Mat &Highlight)
{
    std::vector<cv::Mat> RGBSeq;
    
    SparseImg.convertTo(SparseImg, CV_8UC1, 255);

    cv::split(SparseImg, RGBSeq);

    

    cv::threshold(RGBSeq[0], RGBSeq[0], 0, 255, cv::THRESH_OTSU);
    cv::threshold(RGBSeq[1], RGBSeq[1], 0, 255, cv::THRESH_OTSU);
    cv::threshold(RGBSeq[2], RGBSeq[2], 0, 255, cv::THRESH_OTSU);
    Highlight.setTo(0);
    cv::Vec3b pixel;
    double SatVal;
    for (int row = 0; row < Highlight.rows; row++)
        for (int col = 0; col < Highlight.cols; col++)
        {
            if (IsHighlightPixel(RGBSeq[0].at<uchar>(row, col)) + IsHighlightPixel(RGBSeq[1].at<uchar>(row, col)) + IsHighlightPixel(RGBSeq[2].at<uchar>(row, col)) >= 2)
                Highlight.at<uchar>(row, col) = 255;
        }
}

void AddWeight(cv::Mat &Input1, cv::Mat &Input2, cv::Mat &Weight, cv::Mat &Output)
{
    int nr = Input1.rows;
    int nc = Input1.cols;
    cv::Vec3b pixel1, pixel2;
    double weight;
    cv::Vec3b dstPixel;
    for (int row = 0; row < nr; row++)
        for (int col = 0; col < nc; col++)
        {
            pixel1 = Input1.at<cv::Vec3b>(row, col);
            pixel2 = Input2.at<cv::Vec3b>(row, col);
            weight = Weight.at<double>(row, col);
            dstPixel = pixel1 * (1 - weight) + pixel2 * weight;
            Output.at<cv::Vec3b>(row, col) = dstPixel;
        }
}

void HighlightReconstruct(cv::Mat &RGBImg, cv::Mat &LImg, cv::Mat Highlight, cv::Mat &dstImg, int METHODS, bool Dilation)
{
    if (Dilation)
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
        cv::Mat Belt;
        Highlight.convertTo(Highlight, CV_64FC1, 1.0f / 255);
        cv::dilate(Highlight, Highlight, kernel);
        cv::dilate(Highlight, Belt, kernel);
        Belt = (Belt - Highlight)*0.5;
        Highlight += Belt;
    }
    if (METHODS == ADAWEIGHT)
    {
        cv::blur(Highlight, Highlight, cv::Size(11, 11));
    }
    //cv::Mat storeImg;
    //Highlight.convertTo(Highlight, CV_8UC1, 255);
    //cv::cvtColor(Highlight, Highlight, cv::COLOR_GRAY2BGR);
    //cv::imwrite("RefinedImg.jpg", Highlight);
    AddWeight(RGBImg, LImg, Highlight, dstImg);
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

int ell(int n1, int n2, int n3)
{
    return MIN(MIN(n1, n2), n3);
}

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

int SignOfEle(double pixel)
{
    return (pixel >= 0) ? (pixel > 0 ? 1 : 0) : -1;
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

inline double Abs(double x)
{
    return x > 0 ? x : -x;
}

void Shrinkage(Eigen::MatrixXd &matrix, double tau)
{
    double *data = matrix.data();
    int length = matrix.rows() * matrix.cols();
    double Curr;
    for (int i = 0; i < length; i++)
    {
        Curr = data[i];
        data[i] = SignOfEle(Curr) * MAX(Abs(Curr) - Abs(tau), 0.0);
    }
}

void LagQN(Eigen::MatrixXd &AY, Eigen::MatrixXd &L, Eigen::MatrixXd &S, double lambda_S)
{
    double tol = 1e-4, tau = -1;
    int n1 = AY.rows();
    int n2 = AY.cols();
    //L.setZero();
    //S.setZero();
    bool SMALL = (n1*n2 < pow(50, 2));
    bool MEDIUM = (n1*n2 <= pow(200, 2) && !SMALL);
    bool LARGE = (n1*n2 <= pow(1000, 2) && !SMALL && !MEDIUM);
    bool HUGE_Matrix = (n1*n2 > std::pow(1000, 2));

    int maxIts = 1e3*(SMALL | MEDIUM) + 400 * LARGE + 200 * HUGE_Matrix;
    int printEvery = 100 * SMALL + 50 * MEDIUM + 5 * LARGE + 1 * HUGE_Matrix;

    double Lip;
    double stepsizeQN;

    int restart, trueObj;
    bool displayTime, SVDwarmStart;
    double lambda, lambdaL, lambdaS, stepsize;

    Lip = 2;
    stepsizeQN = 0.8 * 2 / Lip;
    restart = INT_MIN;
    trueObj = 0;
    displayTime = LARGE | HUGE_Matrix;
    SVDwarmStart = true;
    lambda = abs(tau);

    lambdaL = lambda;
    lambdaS = lambda * lambda_S;

    Record record;
    record.iteration = 0;
    record.oldRank = 0;
    record.Vold = Eigen::MatrixXd();
    record.SVDnPower = 1;
    record.SVDoffset = 5;

    RecordForNonConv new_record;
    new_record.iteration = record.iteration;
    new_record.oldRank = record.oldRank;
    new_record.Vold = record.Vold;
    new_record.ratio = 0.005; // ? 待定

    Eigen::MatrixXd errHist, L_old, S_old, dL, dS, Grad, S_temp;

    stepsize = 1 / Lip;
    errHist = Eigen::MatrixXd::Zero(maxIts, 2);
    Grad = Eigen::MatrixXd::Zero(n1, n2);
    dL = Eigen::MatrixXd::Zero(n1, n2);
    dS = Eigen::MatrixXd::Zero(n1, n2);
    L_old = L;
    S_old = S;

    bool BREAK = false;
    int k, rnk;
    Grad = L + S - AY;

    Eigen::MatrixXd temp;

    cv::Mat SShow, LShow;
    cv::Mat errHistShow;
    //project = @(L, S, varargin) projectMax(lambda, lambda*lambda_S, SVDopts, L, S, varargin{ : });
    double tauS = abs(lambdaS*stepsizeQN);
    double tauL = -abs(lambdaL*stepsizeQN);
    for (k = 0; k < maxIts; k++)
    {
        dL = L - L_old;
        S_old = S;


        S_temp = S - stepsizeQN * (Grad + dL);
        Shrinkage(S_temp, tauS);
//        cv::eigen2cv(S_temp, SShow);
        dS = S_temp - S_old;


        L_old = L;
        L = L - stepsizeQN * (Grad + dS);
        rnk = projNuclear(L, tauL, new_record, 1); // ? 待定
//        cv::eigen2cv(L, LShow);

        dL = L - L_old;
        S = S - stepsizeQN * (Grad + dL);
        Shrinkage(S, tauS);
//        cv::eigen2cv(S, SShow);

        Grad = L + S - AY;

        double res = Grad.norm();
        // std::cout<<res<<std::endl;
        errHist(k, 0) = res;
        errHist(k, 1) = 1.0 / 2 * (res*res);
        cv::eigen2cv(errHist, errHistShow);
        if (k > 0 && abs(errHist(k, 0) - errHist(k - 1, 0)) / res < tol)
            BREAK = true;
        if (BREAK)
            break;
    }
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
    // std::cout << "minus:  " << minus << "   lambdaS:" << lambdaS << std::endl;
    while (minus > 0.005 || minus < -0.003)
    {
        // std::cout << "minus:  " << minus << "   lambdaS:" << lambdaS << std::endl;
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

void PCP(Eigen::MatrixXd &AY, Eigen::MatrixXd &L, Eigen::MatrixXd &S, double mu, double lambda)
{
    int nr = AY.rows();
    int nc = AY.cols();

    if (mu == -1)
        mu = nr * nc*1.0f / (4 * AY.lpNorm<1>());
    if (lambda == -1)
        lambda = 1.0f / sqrt((std::max(nr, nc)));

    if (S.data() == NULL)
        S = Eigen::MatrixXd::Zero(nr, nc);

    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(nr, nc);
    double sca = 1.0f / mu;
    Record record;
    record.iteration = 0;
    record.oldRank = 0;
    record.Vold = Eigen::MatrixXd();
    record.SVDnPower = 1;
    record.SVDoffset = 5;

    RecordForNonConv new_record;
    new_record.iteration = record.iteration;
    new_record.oldRank = record.oldRank;
    new_record.Vold = record.Vold;
    new_record.ratio = 0.005; // ? 待定

    bool BREAK = false;
    int maxIts = 1000;
    double nrmLS, perc;
    Eigen::MatrixXd dL, dS;
    double tol = 1e-5;
    for (int i = 0; i < maxIts; i++)
    {
        nrmLS = CalcFNorm(L, S);

        //解L最小化的问题
        dL = L;
        L = AY - S + sca * Y;
        projNuclear(L, 1.0f/mu, new_record, 1); // ? 待定
        dL = L - dL;

        //解S最小化的问题
        dS = S;
        S = AY - L + sca * Y;
        Shrinkage(S, lambda/mu);
        dS = S - dS;

        perc = CalcFNorm(dL, dS) / (1 + nrmLS);
        if (perc < tol)
            break;
        Y = Y + mu * (AY - L - S);
    }
}

void FPCP(Eigen::MatrixXd &AY, Eigen::MatrixXd &Lowrank, Eigen::MatrixXd &Sparse, int loops)
{
    int nr = AY.rows();
    int nc = AY.cols();
    double lambda = 1.0f / sqrt(MAX(nr, nc));
    double lambdaFactor = 1;
    double rankThreshold = 0.01;
    int rank0 = 1;
    int inc_rank = 1;

    int rank = rank0;
    std::vector<int> RankVec;
    RankVec.push_back(rank);
    Eigen::MatrixXd U, S, V;
    Eigen::MatrixXd opts;
    randomizedSVD(AY, U, S, V, rank, MIN(MIN(rank + SVDOFFSET, nr), nc), SVDNPOWER, opts);

    //U*S
    Eigen::MatrixXd TempMatrix = Eigen::MatrixXd::Zero(U.rows(), S.cols());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, U.rows(), S.cols(), U.cols(), 1, (double*)U.data(),
        U.rows(), (double*)S.data(), S.rows(), 1, (double*)TempMatrix.data(), U.rows());
    //L=U*S*VT
    Lowrank.setZero();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, TempMatrix.rows(), V.rows(), TempMatrix.cols(), 1, (double*)TempMatrix.data(),
        TempMatrix.rows(), (double*)V.data(), V.rows(), 1, (double*)Lowrank.data(), TempMatrix.rows());

    Sparse = AY - Lowrank;
    Shrinkage(Sparse, lambda);
    double rho;
    for (int k = 2; k <= loops; k++)
    {
        if (inc_rank == 1)
        {
            lambda = lambda * lambdaFactor;
            rank = rank + 1;
        }
        Lowrank = AY - Sparse;
        randomizedSVD(Lowrank, U, S, V, rank, ell(rank + SVDOFFSET, nr, nc), SVDNPOWER, opts);
        double LastVal = S(rank - 1, rank - 1), Sum = 0;
        for (int i = 0; i < rank - 1; i++)
            Sum += S(i, i);
        RankVec.push_back(rank);
        rho = LastVal / Sum;
        if (rho < rankThreshold)
            inc_rank = 0;
        else
            inc_rank = 1;
        TempMatrix = Eigen::MatrixXd::Zero(U.rows(), S.rows());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, U.rows(), S.cols(), U.cols(), 1, (double*)U.data(),
            U.rows(), (double*)S.data(), S.rows(), 1, (double*)TempMatrix.data(), U.rows());
        //L=U*S*VT
        Lowrank.setZero();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, TempMatrix.rows(), V.rows(), TempMatrix.cols(), 1, (double*)TempMatrix.data(),
            TempMatrix.rows(), (double*)V.data(), V.rows(), 1, (double*)Lowrank.data(), TempMatrix.rows());

        Sparse = AY - Lowrank;
        Shrinkage(Sparse, lambda);
    }
}

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



void EigenThreshold(Eigen::MatrixXd &InputOutPutMatrix, double thre, double MaxVal)
{
    assert(InputOutPutMatrix.data());
    int nr = InputOutPutMatrix.rows();
    int nc = InputOutPutMatrix.cols();
    double *data = InputOutPutMatrix.data();
    for (int i = 0, length = nr * nc; i < length; i++)
        data[i] = data[i] >= thre ? MaxVal : 0;
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

// void PCP(Eigen::MatrixXd &AY, Eigen::MatrixXd &L, Eigen::MatrixXd &S, double mu, double lambda)
// {
//     int nr = AY.rows();
//     int nc = AY.cols();

//     if (mu == -1)
//         mu = nr * nc*1.0f / (4 * AY.lpNorm<1>());
//     if (lambda == -1)
//         lambda = 1.0f / sqrt((std::max(nr, nc)));

//     if (S.data() == NULL)
//         S = Eigen::MatrixXd::Zero(nr, nc);

//     Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(nr, nc);
//     double sca = 1.0f / mu;
//     Record record;
//     record.iteration = 0;
//     record.oldRank = 0;
//     record.Vold = Eigen::MatrixXd();
//     record.SVDnPower = 1;
//     record.SVDoffset = 5;

//     bool BREAK = false;
//     int maxIts = 1000;
//     double nrmLS, perc;
//     Eigen::MatrixXd dL, dS;
//     double tol = 1e-5;
//     for (int i = 0; i < maxIts; i++)
//     {
//         nrmLS = CalcFNorm(L, S);

//         //解L最小化的问题
//         dL = L;
//         L = AY - S + sca * Y;
//         projNuclear(L, 1.0f/mu, record);
//         dL = L - dL;

//         //解S最小化的问题
//         dS = S;
//         S = AY - L + sca * Y;
//         Shrinkage(S, lambda/mu);
//         dS = S - dS;

//         perc = CalcFNorm(dL, dS) / (1 + nrmLS);
//         if (perc < tol)
//             break;
//         Y = Y + mu * (AY - L - S);
//     }
// }

int projNuclear(Eigen::MatrixXd &X, double tauL, Record &record)
{
    int rEst;
    if (record.oldRank == 0) // 第一次执行奇异值收缩算法
        // rEst = MIN(MIN(X.rows(), X.cols()), 10);
        rEst = 10;
    else
        rEst = record.oldRank + 2;

    record.iteration += 1;
    int nr = (int)X.rows();
    int nc = (int)X.cols();
    int minN = MIN(nr, nc);

    int rankMax;
    switch (record.iteration) // 所允许的最大的秩
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
    if (tauL == 0) // 即不进行奇异值收缩
    {
        X.setZero();
        return rEst;
    }

    Eigen::MatrixXd opts;
    if (record.Vold.data()) // Vold.rows()==0||Vold.cols()==0
        opts = record.Vold;

    Eigen::MatrixXd U, S, V, s;
    bool ok = false;
    double lambda_temp = 0;
    if (tauL < 0)
        tauL = abs(tauL);

    while (!ok)
    {
        rEst = MIN(rEst, rankMax);

        // X=U*S*VT
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
    // Shrinkage(S, tauL);
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
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, U_sub.rows(), S_sub.cols(), U_sub.cols(), 1, (double *)U_sub.data(),
                    U_sub.rows(), (double *)S_sub.data(), S_sub.rows(), 1, (double *)temp_Matrix.data(), U_sub.rows());

        X.setZero();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, temp_Matrix.rows(), V_sub.rows(), temp_Matrix.cols(), 1, (double *)temp_Matrix.data(),
                    temp_Matrix.rows(), (double *)V_sub.data(), V_sub.rows(), 1, (double *)X.data(), temp_Matrix.rows());
    }
    record.oldRank = rEst;
    record.Vold = V.middleCols(0, rEst);
    return rEst;
}

cv::Mat ImgHighlightRemoval(cv::Mat RGBImg, int EnhanceMethod, int DetectionMethod, int RPCA_Type, bool BATCH, int batchesRow, int batchesCol)
{
    cv::Mat Frame = RGBImg;
    RGBImg.convertTo(RGBImg,CV_64FC3,1.0f/255);
    int nr = RGBImg.rows;
    int nc = RGBImg.cols;

    std::vector<Eigen::MatrixXd> BGRSequences, LowRankSequences, SparseSequences, MSVSequences, GradVec, BGREnhanceSequences;
    Eigen::MatrixXd Highlight(nr, nc), HighlightDilate(nr, nc);
    InitMatrixVec(BGRSequences, 3, nr, nc);
    InitMatrixVec(LowRankSequences, 3, nr, nc);
    InitMatrixVec(SparseSequences, 3, nr, nc);
    InitMatrixVec(MSVSequences, 3, nr, nc);
    InitMatrixVec(BGREnhanceSequences, 5, nr, nc);
    InitMatrixVec(GradVec, 5, nr, nc);

    SplitImg(RGBImg, BGRSequences);
    switch (EnhanceMethod)
    {
    case EnhanceGrad:
        RGB2MSV(BGRSequences, MSVSequences);
        CalcGradient(MSVSequences.at(0), GradVec);
        MSVTuning(MSVSequences, GradVec.at(4), 0.8);
        break;
    case EnhanceSobel:
        SequencesSobel(BGRSequences, BGREnhanceSequences);
        RGB2MSV(BGREnhanceSequences, MSVSequences);
        break;
    default:
        RGB2MSV(BGRSequences, MSVSequences);
        break;
    }
    if (DetectionMethod == MSVG)
    {
        if (EnhanceMethod != EnhanceGrad)
            CalcGradient(MSVSequences.at(0), GradVec);
        HighlightDetectionGrad(MSVSequences, GradVec.at(4), Highlight, 2); // ? 待定
    }
    else
        HighlightDetection(MSVSequences, Highlight, DetectionMethod);

    EigenDilate(Highlight, HighlightDilate, 3, 3);
    HighlightDilate=Highlight;

    InitData(BGRSequences, LowRankSequences, SparseSequences, HighlightDilate);//对L和S进行初始化操作
    std::vector<Eigen::MatrixXd> BGRBatches, LowRankBatches, SparseBatches, HighlightBatches;
    int nr_batch, nc_batch;
    int nChannels = BGRSequences.size();
    if (BATCH == false)
    {
        batchesRow = 1;
        batchesCol = 1;
        nr_batch = nr;
        nc_batch = nc;
    }
    else
    {
        nr_batch = nr / batchesRow;
        nc_batch = nc / batchesCol;
        InitMatrixVec(BGRBatches, nChannels*batchesCol*batchesRow, nr_batch, nc_batch);
        InitMatrixVec(LowRankBatches, nChannels*batchesCol*batchesRow, nr_batch, nc_batch);
        InitMatrixVec(SparseBatches, nChannels*batchesCol*batchesRow, nr_batch, nc_batch);
        InitMatrixVec(HighlightBatches, batchesRow*batchesCol, nr_batch, nc_batch);
    }

    cv::Mat LImgD(nr, nc, CV_64FC3);
    cv::Mat SImgD(nr, nc, CV_64FC3);//浮点型存储的数据
    cv::Mat HighlightImg(nr, nc, CV_8UC1);
    cv::Mat dstImg(nr, nc, CV_8UC3);
    switch (RPCA_Type)
    {
    case RPCA_FPCP:
    {
        for (int i = 0; i < 3; i++)
            FPCP(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences[i], 10);
        break;
    }
    case RPCA_LAGQN:
    {
        for (int i = 0; i < 3; i++)
            LagQN(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i), 0.02);
        break;
    }
    case RPCA_PCP:
    {
        for (int i = 0; i < 3; i++)
            PCP(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i), 0, 0); // ? 待定
        break;
    }
    case RPCA_NONCONVEX:
    {
        std::vector<unsigned int> Position;
        CalcHighlightIndex(HighlightDilate, Position);
//#pragma omp parallel for
        for (int i = 0; i < 3; i++)
        {
            // std::cout<<i<<std::endl;
            RecordForNonConv record;
            record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
            ADMM_NonConv(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i), 2, 0.01, 2, Position, 1, record);
        }
        break;
    }
    case RPCA_NONCONVEXBATCH:
    {
        assert(BATCH == true);
        Blocking(BGRSequences, BGRBatches, batchesRow, batchesCol, nr_batch, nc_batch);
        Blocking(LowRankSequences, LowRankBatches, batchesRow, batchesCol, nr_batch, nc_batch);
        Blocking(SparseSequences, SparseBatches, batchesRow, batchesCol, nr_batch, nc_batch);
        Blocking(HighlightDilate, HighlightBatches, batchesRow, batchesCol, nr_batch, nc_batch);

        std::vector<std::vector<unsigned int>> PositionVec(batchesRow*batchesCol);
        for (int i = 0; i < batchesRow*batchesCol; i++)
            CalcHighlightIndex(HighlightBatches[i], PositionVec[i]);
#pragma omp parallel for
        for (int i = 0; i < batchesCol*batchesRow*nChannels; i++)
        {
            // std::cout<<i<<std::endl;
            RecordForNonConv record;
            record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
            ADMM_NonConv(BGRBatches[i], LowRankBatches[i], SparseBatches[i], 2, 0.01, 2, PositionVec[i / 3], 10, record);
        }
        break;
    }
    case RPCA_ADARPCA:
    {
        AdaRPCA(BGRSequences, LowRankSequences, SparseSequences, HighlightDilate);
    }
    default:
        break;
    }
    if (BATCH)
    {
        Merging(LowRankBatches, LImgD, batchesRow, batchesCol, nr_batch, nc_batch, 1);
        Merging(SparseBatches, SImgD, batchesRow, batchesCol, nr_batch, nc_batch, 1);
    }
    else
    {
        Merging(LowRankSequences, LImgD, batchesRow, batchesCol, nr_batch, nc_batch, 1);
        Merging(SparseSequences, SImgD, batchesRow, batchesCol, nr_batch, nc_batch, 1);
    }
    // LImgD.convertTo(LImg, CV_8UC3, 255);
    // SImgD.convertTo(SImg, CV_8UC3, 255);
    HighlightRefine(SImgD, HighlightImg);
    HighlightReconstruct(Frame, LImgD, HighlightImg, dstImg, ADAWEIGHT, true);
    cv::cvtColor(HighlightImg, HighlightImg, cv::COLOR_GRAY2BGR);
    return dstImg;
    //            cv::imshow("Lowrank",LImg);
    //            cv::imshow("Sparse",SImg);
    //            cv::imshow("Highlight",HighlightImg);
    //            cv::waitKey(0);

}

int main(int argc, char** argv) {
    // Check input
	// main ../dataset/Adobe/train/input/ ./ 一定要有最后的/
	if (argc != 3) {
		std::cout << "Usage: ./main InputLocation OutputLocation" << std::endl; 
		return -1;
	}

	struct dirent *entry;
    DIR *dp;

    dp = opendir(argv[1]);
    if (dp == NULL) {
        perror("opendir: Path does not exist or could not be read.");
        return -1;
    }

    std::vector<std::string> files;
    while ((entry = readdir(dp))) {
        std::string f_name = entry->d_name;
        if (f_name.find(".jpg") != std::string::npos || f_name.find(".png") != std::string::npos) {
            files.push_back(f_name);
        }
    }
    closedir(dp); // Close the directory as it's no longer needed

    progressbar bar(files.size());

    omp_set_num_threads(16);

    #pragma omp parallel for
    for (int i = 0; i < files.size(); ++i) {
        std::string f_name = files[i];
        // std::cout << "Processing: " << f_name << std::endl;

        std::string src = std::string(argv[1]) + f_name;
        std::string dst = std::string(argv[2]) + f_name;

        // Read the image
        cv::Mat image = cv::imread(src, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << src << std::endl;
            continue;
        }

        // Convert color space if needed
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Perform image processing (make sure ImgHighlightRemoval and related functions are thread-safe)
        cv::Mat out = ImgHighlightRemoval(image, EnhanceGrad, MSVTHRESHOLDS, RPCA_NONCONVEX, false, 1, 1);

        // Write the processed image to the output directory
        cv::imwrite(dst, out);
        #pragma omp critical
        bar.update();
    }

    return 0;
}
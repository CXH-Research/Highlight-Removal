#include<HighlightProcessing.h>
//#include<omp.h>

void ImgHighlightRemoval(cv::Mat RGBImg, cv::Mat &dstImg, cv::Mat &LImg, cv::Mat &SImg, cv::Mat &HighlightImg, int EnhanceMethod, int DetectionMethod, int RPCA_Type, bool BATCH, int batchesRow, int batchesCol)
{
    cv::Mat Frame=RGBImg;
    RGBImg.convertTo(RGBImg,CV_64FC3,1.0f/255);
    int nr=RGBImg.rows;
    int nc=RGBImg.cols;

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
        HighlightDetectionGrad(MSVSequences, GradVec.at(4), Highlight);
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
    HighlightImg.create(nr, nc, CV_8UC1);
    dstImg.create(nr, nc, CV_8UC3);
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
            PCP(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i));
        break;
    }
    case RPCA_NONCONVEX:
    {
        std::vector<unsigned int> Position;
        CalcHighlightIndex(HighlightDilate, Position);
//#pragma omp parallel for
        for (int i = 0; i < 3; i++)
        {
            std::cout<<i<<std::endl;
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
            std::cout<<i<<std::endl;
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
    LImgD.convertTo(LImg, CV_8UC3, 255);
    SImgD.convertTo(SImg, CV_8UC3, 255);
    HighlightRefine(SImg, HighlightImg);
    HighlightReconstruct(Frame, LImg, HighlightImg, dstImg, ADAWEIGHT, true);
    cv::cvtColor(HighlightImg,HighlightImg,cv::COLOR_GRAY2BGR);
    //            cv::imshow("Lowrank",LImg);
    //            cv::imshow("Sparse",SImg);
    //            cv::imshow("Highlight",HighlightImg);
    //            cv::waitKey(0);

}


void TestHighlightRemoval_ImgSequences(cv::Mat Frame, cv::Mat BorderImg, cv::Mat &LImg, cv::Mat &SImg, cv::Mat &dstImg, cv::Mat &HighlightImg, int Method, int Enhancement_Method, int RPCA_Type)
{
    cv::Mat RGBROI, RGBImg;
    std::vector<Eigen::MatrixXd> BGRSequences, MSVSequences, GradVec, BGREnhance, LowRankSequences, SparseSequences;
    Eigen::MatrixXd Highlight, HighlightDilate;
    cv::Mat ShowImg, LImgD, SImgD, HighlightStore;
    int nr, nc, nChannels;
    std::vector<cv::Mat> BGRImgVec(3), LImgVec(3), SImgVec(3);
    cv::Mat TemplateImg;
    Eigen::MatrixXd TempMat;
    Frame.convertTo(RGBImg, CV_64FC3, 1.0f / 255);


    nChannels = RGBImg.channels();
    nr = RGBImg.rows;
    nc = RGBImg.cols;
    ShowImg.create(nr, nc, CV_8UC1);
    LImg = cv::Mat::zeros(nr, nc, CV_8UC3);
    SImg = cv::Mat::zeros(nr, nc, CV_8UC3);
    dstImg = cv::Mat::zeros(nr, nc, CV_8UC3);
    HighlightImg = cv::Mat(nr, nc, CV_8UC1);

    //根据Border提取感兴趣区域
    cv::cvtColor(BorderImg, BorderImg, cv::COLOR_BGR2GRAY);
    cv::threshold(BorderImg, BorderImg, 0, 255, cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(BorderImg, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
    int idx = 0;
    ShowImg.setTo(0);
    cv::Rect rect;
    for (; idx < contours.size(); idx++)
    {
        drawContours(ShowImg, contours, idx, cv::Scalar(255), cv::FILLED, 8, hierarchy);
        rect = cv::boundingRect(contours.at(idx));
        cv::Point2f P[4];
        cv::rectangle(ShowImg, rect, cv::Scalar(255));
    }
    int width = rect.width;
    int height = rect.height;
    RGBROI = RGBImg(rect);

    TemplateImg = BorderImg(rect); TemplateImg.convertTo(TemplateImg, CV_64FC1, 1.0f / 255); cv::cv2eigen(TemplateImg, TempMat);
    InitMatrixVec(BGRSequences, nChannels, height, width);
    InitMatrixVec(MSVSequences, nChannels, height, width);
    InitMatrixVec(BGREnhance, nChannels, height, width);
    InitMatrixVec(LowRankSequences, 3, height, width);
    InitMatrixVec(SparseSequences, 3, height, width);
    InitMatrixVec(GradVec, 5, height, width);
    Highlight = Eigen::MatrixXd(height, width);
    HighlightDilate = Eigen::MatrixXd(height, width);
    LImgD = cv::Mat(height, width, CV_64FC3);
    SImgD = cv::Mat(height, width, CV_64FC3);


    SplitImg(RGBROI, BGRSequences);
    switch (Enhancement_Method)
    {
    case EnhanceGrad:
        RGB2MSV(BGRSequences, MSVSequences);
        CalcGradient(MSVSequences.at(0), GradVec);
        MSVTuning(MSVSequences, GradVec.at(4), 0.8);
        break;
    case EnhanceSobel:
        SequencesSobel(BGRSequences, BGREnhance);
        RGB2MSV(BGREnhance, MSVSequences);
        break;
    default:
        RGB2MSV(BGRSequences, MSVSequences);
        break;
    }
    if (Method == MSVG)
    {
        if (Enhancement_Method != EnhanceGrad)
            CalcGradient(MSVSequences.at(0), GradVec);
        HighlightDetectionGrad(MSVSequences, GradVec.at(4), Highlight);
    }
    else
        HighlightDetection(MSVSequences, Highlight, Method);
    EigenDilate(Highlight, HighlightDilate, 3, 3);
    InitData(BGRSequences, LowRankSequences, SparseSequences, HighlightDilate);//对L和S进行初始化操作

    {
        double *OBData = BGRSequences.at(0).data();
        double *OGData = BGRSequences.at(1).data();
        double *ORData = BGRSequences.at(2).data();

        double *LBData = LowRankSequences.at(0).data();
        double *LGData = LowRankSequences.at(1).data();
        double *LRData = LowRankSequences.at(2).data();
        double *TData = TempMat.data();
        for (int i = 0; i < width*height; i++)
        {
            OBData[i] = TData[i] * OBData[i] + (1 - TData[i])*LBData[i];
            OGData[i] = TData[i] * OGData[i] + (1 - TData[i])*LGData[i];
            ORData[i] = TData[i] * ORData[i] + (1 - TData[i])*LRData[i];
        }

    }
//    for (int j = 0; j < 3; j++)
//    {
//        cv::eigen2cv(BGRSequences.at(j), BGRImgVec.at(j));
//        cv::eigen2cv(LowRankSequences.at(j), LImgVec.at(j));
//        cv::eigen2cv(SparseSequences.at(j), SImgVec.at(j));
//    }
    //该代码块调用的是我的方法
    {
//        std::vector<unsigned int> Position;
//        CalcHighlightIndex(HighlightDilate, Position);
//        for (int j = 0; j < 3; j++)
//        {
//            RecordForNonConv record;
//            record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
//            ADMM_NonConv(BGRSequences.at(j), LowRankSequences.at(j), SparseSequences.at(j), 2, 0.01, 2, Position, 1, record);
//        }
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
                PCP(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i));
            break;
        }
        case RPCA_NONCONVEX:
        case RPCA_NONCONVEXBATCH:
        {
            std::vector<unsigned int> Position;
            CalcHighlightIndex(HighlightDilate, Position);
    //#pragma omp parallel for
            for (int i = 0; i < 3; i++)
            {
                std::cout<<i<<std::endl;
                RecordForNonConv record;
                record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
                ADMM_NonConv(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i), 2, 0.01, 2, Position, 1, record);
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

    }
    for (int j = 0; j < 3; j++)
    {
        cv::eigen2cv(LowRankSequences.at(j), LImgVec.at(j));
        cv::eigen2cv(SparseSequences.at(j), SImgVec.at(j));
        cv::multiply(SImgVec.at(j), TemplateImg, SImgVec.at(j));
        cv::multiply(LImgVec.at(j), TemplateImg, LImgVec.at(j));
    }

        cv::merge(LImgVec, LImgD);
        cv::merge(SImgVec, SImgD);
        LImgD.convertTo(LImgD, CV_8UC3, 255);
        SImgD.convertTo(SImgD, CV_8UC3, 255);
        LImg.setTo(cv::Scalar(0, 0, 0));
        SImg.setTo(cv::Scalar(0, 0, 0));
        LImgD.copyTo(LImg(rect));
        SImgD.copyTo(SImg(rect));

        HighlightImg.create(nr, nc, CV_8UC1);
        HighlightRefine(SImg, HighlightImg);

        HighlightReconstruct(Frame, LImg, HighlightImg, dstImg, ADAWEIGHT, true);
        cv::cvtColor(HighlightImg, HighlightImg, cv::COLOR_GRAY2BGR);

}
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
        imbinarize(ValidEdgeDilate, ValidEdgeDilate);//这里挺奇怪的，不处理的话，高光像素不是1
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
void HighlightRefine(cv::Mat &SparseImg, cv::Mat &Highlight)
{
    std::vector<cv::Mat> RGBSeq;
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

inline int IsHighlightPixel(uchar pixel)
{
    return pixel == 255 ? 1 : 0;
    //if (pixel == 255)
    //	return 1;
    //else
    //	return 0;
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


double VideoHighlightRemoval(std::string RGBVideoPath, std::string LowRankVideoPath, std::string SparseVideoPath, std::string HighlightVideoPath, std::string dstVideoPath, int RPCA_Type, int DetectionMethod, int EnhanceMethod, bool BATCH, int batchesRow, int batchesCol)
{
    cv::VideoCapture cap(RGBVideoPath);
    int nr = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int nc = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    cv::Size sz(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter lowrank_writer = cv::VideoWriter(LowRankVideoPath, CV_FOURCC('M', 'J', 'P', 'G'), 25, sz);
    cv::VideoWriter sparse_writer = cv::VideoWriter(SparseVideoPath, CV_FOURCC('M', 'J', 'P', 'G'), 25, sz);
    cv::VideoWriter highlight_writer = cv::VideoWriter(HighlightVideoPath, CV_FOURCC('M', 'J', 'P', 'G'), 25, sz);
    cv::VideoWriter dstImage_writer = cv::VideoWriter(dstVideoPath, CV_FOURCC('M', 'J', 'P', 'G'), 25, sz);
    cv::Mat Frame, RGBImg;//Frame是读取的原图，RGBImg是该图的浮点型表示

    std::vector<Eigen::MatrixXd> BGRSequences, LowRankSequences, SparseSequences, MSVSequences, BGREnhanceSequences;
    Eigen::MatrixXd Highlight(nr, nc), HighlightDilate(nr, nc);
    InitMatrixVec(BGRSequences, 3, nr, nc);
    InitMatrixVec(LowRankSequences, 3, nr, nc);
    InitMatrixVec(SparseSequences, 3, nr, nc);
    InitMatrixVec(MSVSequences, 3, nr, nc);
    InitMatrixVec(BGREnhanceSequences, 3, nr, nc);
    std::vector<Eigen::MatrixXd> GradVec;
    InitMatrixVec(GradVec, 5, nr, nc);

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
    cv::Mat LImg, SImg;
    cv::Mat HighlightImg(nr, nc, CV_8UC1);
    cv::Mat Highlight_Store;
    cv::Mat dstImg(nr, nc, CV_8UC3);
    double time = 0;
    while (true)
    {
        cap >> Frame;
        if (Frame.empty())
            break;
        Frame.convertTo(RGBImg, CV_64FC3, 1.0f / 255);
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
            HighlightDetectionGrad(MSVSequences, GradVec.at(4), Highlight);
        }
        else
            HighlightDetection(MSVSequences, Highlight, DetectionMethod);
        EigenDilate(Highlight, HighlightDilate, 3, 3);
        InitData(BGRSequences, LowRankSequences, SparseSequences, HighlightDilate);//对L和S进行初始化操作
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
                PCP(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i));
            break;
        }
        case RPCA_NONCONVEX:
        {
            std::vector<unsigned int> Position;
            CalcHighlightIndex(HighlightDilate, Position);
            for (int i = 0; i < 3; i++)
            {
                RecordForNonConv record;
                record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
                ADMM_NonConv(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i), 2, 0.01, 2, Position, 10, record);
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
                RecordForNonConv record;
                record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
                ADMM_NonConv(BGRBatches[i], LowRankBatches[i], SparseBatches[i], 2, 0.01, 2, PositionVec[i / 3], 10, record);
            }
            break;
        }
        case RPCA_ADARPCA:
        {

            time += AdaRPCA(BGRSequences, LowRankSequences, SparseSequences, HighlightDilate);
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
        LImgD.convertTo(LImg, CV_8UC3, 255);
        SImgD.convertTo(SImg, CV_8UC3, 255);

        //HighlightRefine(SImg, HighlightImg);
        cv::eigen2cv(Highlight, HighlightImg);
        HighlightImg.convertTo(HighlightImg, CV_8UC1, 255);
        //HighlightRefineGrad(Frame, SImg, GradVec.at(4), HighlightImg);
        HighlightReconstruct(Frame, LImg, HighlightImg, dstImg, ADAWEIGHT, true);
        cv::cvtColor(HighlightImg, Highlight_Store, cv::COLOR_GRAY2BGR);
        lowrank_writer.write(LImg);
        sparse_writer.write(SImg);
        highlight_writer.write(Highlight_Store);
        dstImage_writer.write(dstImg);
    }
    lowrank_writer.release();
    sparse_writer.release();
    highlight_writer.release();
    dstImage_writer.release();
    return time;
}

void TestHighlightRemoval_ImgSequences(std::string RGBRootPath, std::string LowRankRootPath, std::string SparseRootPath, \
    std::string HighlightRootPath, std::string dstImgRootPath, int Method, int Enhancement_Method)
{
    std::vector<cv::String> file_names;
    cv::glob(RGBRootPath, file_names);
    int FilesNumber = file_names.size();

    cv::Mat RGBROI, Frame, RGBImg;
    std::vector<Eigen::MatrixXd> BGRSequences, MSVSequences, GradVec, BGREnhance, LowRankSequences, SparseSequences;
    Eigen::MatrixXd Highlight, HighlightDilate;
    cv::Mat ShowImg, LImgD, SImgD, LImg, SImg, dstImg, HighlightImg, HighlightStore;
    int nr, nc, nChannels;
    std::string borderRootPath = "G:\\ExperimentalData\\CVC\\border\\";
    cv::Mat BorderImg;
    std::vector<cv::Mat> BGRImgVec(3), LImgVec(3), SImgVec(3);
    cv::Mat TemplateImg;
    Eigen::MatrixXd TempMat;
    for (int i = 0; i < FilesNumber; i++)
    {
        Frame = cv::imread(file_names.at(i));
        Frame.convertTo(RGBImg, CV_64FC3, 1.0f / 255);

        if (i == 0)
        {
            nChannels = RGBImg.channels();
            nr = RGBImg.rows;
            nc = RGBImg.cols;
            ShowImg.create(nr, nc, CV_8UC1);
            LImg = cv::Mat::zeros(nr, nc, CV_8UC3);
            SImg = cv::Mat::zeros(nr, nc, CV_8UC3);
            dstImg = cv::Mat::zeros(nr, nc, CV_8UC3);
            HighlightImg = cv::Mat(nr, nc, CV_8UC1);
        }
        int pos = file_names.at(i).find_last_of("\\");
        std::string dstFileName = file_names.at(i).substr(pos + 1);
        BorderImg = cv::imread(borderRootPath + dstFileName);
        cv::cvtColor(BorderImg, BorderImg, cv::COLOR_BGR2GRAY);
        cv::threshold(BorderImg, BorderImg, 0, 255, cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point>>contours;
        std::vector<cv::Vec4i> hierarchy;
        findContours(BorderImg, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
        int idx = 0;
        ShowImg.setTo(0);
        cv::Rect rect;
        for (; idx < contours.size(); idx++)
        {
            drawContours(ShowImg, contours, idx, cv::Scalar(255), cv::FILLED, 8, hierarchy);
            rect = cv::boundingRect(contours.at(idx));
            cv::Point2f P[4];
            cv::rectangle(ShowImg, rect, cv::Scalar(255));
        }
        int width = rect.width;
        int height = rect.height;
        RGBROI = RGBImg(rect);
        TemplateImg = BorderImg(rect); TemplateImg.convertTo(TemplateImg, CV_64FC1, 1.0f / 255); cv::cv2eigen(TemplateImg, TempMat);
        InitMatrixVec(BGRSequences, nChannels, height, width);
        InitMatrixVec(MSVSequences, nChannels, height, width);
        InitMatrixVec(BGREnhance, nChannels, height, width);
        InitMatrixVec(LowRankSequences, 3, height, width);
        InitMatrixVec(SparseSequences, 3, height, width);
        InitMatrixVec(GradVec, 5, height, width);
        Highlight = Eigen::MatrixXd(height, width);
        HighlightDilate = Eigen::MatrixXd(height, width);
        LImgD = cv::Mat(height, width, CV_64FC3);
        SImgD = cv::Mat(height, width, CV_64FC3);
        //HighlightImg = cv::Mat(height, width, CV_8UC1);
    /*	dstImg = cv::Mat(height, width, CV_8UC3);*/


        SplitImg(RGBROI, BGRSequences);
        switch (Enhancement_Method)
        {
        case EnhanceGrad:
            RGB2MSV(BGRSequences, MSVSequences);
            CalcGradient(MSVSequences.at(0), GradVec);
            MSVTuning(MSVSequences, GradVec.at(4), 0.8);
            break;
        case EnhanceSobel:
            SequencesSobel(BGRSequences, BGREnhance);
            RGB2MSV(BGREnhance, MSVSequences);
            break;
        default:
            RGB2MSV(BGRSequences, MSVSequences);
            break;
        }
        if (Method == MSVG)
        {
            if (Enhancement_Method != EnhanceGrad)
                CalcGradient(MSVSequences.at(0), GradVec);
            HighlightDetectionGrad(MSVSequences, GradVec.at(4), Highlight);
        }
        else
            HighlightDetection(MSVSequences, Highlight, Method);
        EigenDilate(Highlight, HighlightDilate, 3, 3);
        InitData(BGRSequences, LowRankSequences, SparseSequences, HighlightDilate);//对L和S进行初始化操作

        {
            double *OBData = BGRSequences.at(0).data();
            double *OGData = BGRSequences.at(1).data();
            double *ORData = BGRSequences.at(2).data();

            double *LBData = LowRankSequences.at(0).data();
            double *LGData = LowRankSequences.at(1).data();
            double *LRData = LowRankSequences.at(2).data();
            double *TData = TempMat.data();
            for (int i = 0; i < width*height; i++)
            {
                OBData[i] = TData[i] * OBData[i] + (1 - TData[i])*LBData[i];
                OGData[i] = TData[i] * OGData[i] + (1 - TData[i])*LGData[i];
                ORData[i] = TData[i] * ORData[i] + (1 - TData[i])*LRData[i];
            }

        }
        for (int j = 0; j < 3; j++)
        {
            cv::eigen2cv(BGRSequences.at(j), BGRImgVec.at(j));
            cv::eigen2cv(LowRankSequences.at(j), LImgVec.at(j));
            cv::eigen2cv(SparseSequences.at(j), SImgVec.at(j));
        }
        //该代码块调用的是我的方法
        //{
        //	std::vector<unsigned int> Position;
        //	CalcHighlightIndex(HighlightDilate, Position);
        //	for (int j = 0; j < 3; j++)
        //	{
        //		RecordForNonConv record;
        //		record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
        //		ADMM_NonConv(BGRSequences.at(j), LowRankSequences.at(j), SparseSequences.at(j), 2, 0.01, 2, Position, 1, record);
        //	}

        //}
        //该代码块调用的是PCP方法
        {
            for (int j = 0; j < 3; j++)
                PCP(BGRSequences.at(j), LowRankSequences.at(j), SparseSequences[j]);
        }
        //调用的是LagQN的方法
        /*{
            for (int j = 0; j < 3; j++)
                LagQN(BGRSequences.at(j), LowRankSequences.at(j), SparseSequences[j], 0.02);
        }*/
        //AdaRPCA(BGRSequences, LowRankSequences, SparseSequences, HighlightDilate);
        for (int j = 0; j < 3; j++)
        {
            cv::eigen2cv(LowRankSequences.at(j), LImgVec.at(j));
            cv::eigen2cv(SparseSequences.at(j), SImgVec.at(j));
            cv::multiply(SImgVec.at(j), TemplateImg, SImgVec.at(j));
            cv::multiply(LImgVec.at(j), TemplateImg, LImgVec.at(j));
        }
        /*Merging(LowRankSequences, LImgD, 1, 1, nr, nc, 1);
        Merging(SparseSequences, SImgD, 1, 1, nr, nc, 1);*/
        cv::merge(LImgVec, LImgD);
        cv::merge(SImgVec, SImgD);
        LImgD.convertTo(LImgD, CV_8UC3, 255);
        SImgD.convertTo(SImgD, CV_8UC3, 255);
        LImg.setTo(cv::Scalar(0, 0, 0));
        SImg.setTo(cv::Scalar(0, 0, 0));
        LImgD.copyTo(LImg(rect));
        SImgD.copyTo(SImg(rect));

        HighlightImg.create(nr, nc, CV_8UC1);
        HighlightRefine(SImg, HighlightImg);

        //dstImg.copyTo(dstImgFinal(rect));
        //cv::Mat HighlightStore(nr, nc, CV_8UC3);
        HighlightReconstruct(Frame, LImg, HighlightImg, dstImg, ADAWEIGHT, true);
        cv::cvtColor(HighlightImg, HighlightImg, cv::COLOR_GRAY2BGR);

        std::string LowRankPath = LowRankRootPath + dstFileName;
        std::string SparsePath = SparseRootPath + dstFileName;
        std::string HighlightPath = HighlightRootPath + dstFileName;
        std::string dstImgPath = dstImgRootPath + dstFileName;

        if (LowRankPath != "")
            cv::imwrite(LowRankPath, LImg);
        if (SparsePath != "")
            cv::imwrite(SparsePath, SImg);
        if (HighlightPath != "")
            cv::imwrite(HighlightPath, HighlightImg);
        if (dstImgPath != "")
            cv::imwrite(dstImgPath, dstImg);
    }
}

void ImgHighlightRemoval(std::string RGBImgPath, std::string LowRankPath, std::string SparsePath, std::string HighlightPath, std::string dstImgPath, int RPCA_Type, int DetectionMethod, int EnhanceMethod, bool BATCH, int batchesRow, int batchesCol)
{
    cv::Mat Frame, RGBImg;//Frame是读取的原图，RGBImg是该图的浮点型表示
    Frame = cv::imread(RGBImgPath);
    Frame.convertTo(RGBImg, CV_64FC3, 1.0f / 255);

    int nr = Frame.rows;
    int nc = Frame.cols;
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
        HighlightDetectionGrad(MSVSequences, GradVec.at(4), Highlight);
    }
    else
        HighlightDetection(MSVSequences, Highlight, DetectionMethod);
    cv::Mat AbsHImg;
    cv::eigen2cv(Highlight, AbsHImg);
    AbsHImg.convertTo(AbsHImg, CV_8UC1, 255);
    cv::cvtColor(AbsHImg, AbsHImg, cv::COLOR_GRAY2BGR);
    cv::imwrite("AbsoluteHighlight.jpg", AbsHImg);
    EigenDilate(Highlight, HighlightDilate, 3, 3);
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
    cv::Mat LImg, SImg;
    cv::Mat HighlightImg(nr, nc, CV_8UC1);
    cv::Mat dstImg(nr, nc, CV_8UC3);
    //Merging(LowRankSequences, LImgD, batchesRow, batchesCol, nr_batch, nc_batch, 1);
    //Merging(SparseSequences, SImgD, batchesRow, batchesCol, nr_batch, nc_batch, 1);
    //LImgD.convertTo(LImg, CV_8UC3, 255);
    //SImgD.convertTo(SImg, CV_8UC3, 255);
    //imwrite("InitLImg.jpg", LImg);
    //imwrite("InitSImg.jpg", SImg);
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
            PCP(BGRSequences.at(i), LowRankSequences.at(i), SparseSequences.at(i));
        break;
    }
    case RPCA_NONCONVEX:
    {
        std::vector<unsigned int> Position;
        CalcHighlightIndex(HighlightDilate, Position);
        for (int i = 0; i < 3; i++)
        {
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
        for (int i = 0; i < BGRBatches.size(); i++)
        {
            cv::Mat Temp;
            cv::eigen2cv(BGRBatches[i], Temp);
            Temp.convertTo(Temp, CV_8UC1, 255);
            cv::cvtColor(Temp, Temp, cv::COLOR_GRAY2BGR);
            std::string filename = "./Batches/RGBImg/" + std::to_string(i) + ".jpg";
            cv::imwrite(filename,Temp);
        }
        for (int i = 0; i < HighlightBatches.size(); i++)
        {
            cv::Mat Temp;
            cv::eigen2cv(HighlightBatches[i], Temp);
            Temp.convertTo(Temp, CV_8UC1, 255);
            cv::cvtColor(Temp, Temp, cv::COLOR_GRAY2BGR);
            std::string filename = "./Batches/Highlight/" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, Temp);
        }
        std::vector<std::vector<unsigned int>> PositionVec(batchesRow*batchesCol);
        for (int i = 0; i < batchesRow*batchesCol; i++)
            CalcHighlightIndex(HighlightBatches[i], PositionVec[i]);
#pragma omp parallel for
        for (int i = 0; i < batchesCol*batchesRow*nChannels; i++)
        {
            RecordForNonConv record;
            record.iteration = 0; record.oldRank = 0; record.ratio = 0.005;  record.Vold = Eigen::MatrixXd();
            ADMM_NonConv(BGRBatches[i], LowRankBatches[i], SparseBatches[i], 2, 0.01, 2, PositionVec[i / 3], 10, record);
        }

        for (int i = 0; i < LowRankBatches.size(); i++)
        {
            cv::Mat Temp;
            cv::eigen2cv(LowRankBatches[i], Temp);
            Temp.convertTo(Temp, CV_8UC1, 255);
            cv::cvtColor(Temp, Temp, cv::COLOR_GRAY2BGR);
            std::string filename = "./Batches/LowRank/" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, Temp);
        }
        for (int i = 0; i < SparseBatches.size(); i++)
        {
            cv::Mat Temp;
            cv::eigen2cv(SparseBatches[i], Temp);
            Temp.convertTo(Temp, CV_8UC1, 255);
            cv::cvtColor(Temp, Temp, cv::COLOR_GRAY2BGR);
            std::string filename = "./Batches/Sparse/" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, Temp);
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

    LImgD.convertTo(LImg, CV_8UC3, 255);
    SImgD.convertTo(SImg, CV_8UC3, 255);
    //二选一
    //HighlightRefine(SImg, HighlightImg);
    cv::cvtColor(AbsHImg, HighlightImg, cv::COLOR_BGR2GRAY);
    //HighlightRefineGrad(Frame, SImg, GradVec.at(2), HighlightImg);
    //HighlightRefineGrad(Frame, SImg, GradVec.at(2), HighlightImg);
    HighlightReconstruct(Frame, LImg, HighlightImg, dstImg, ADAWEIGHT, true);

    cv::cvtColor(HighlightImg, HighlightImg, cv::COLOR_GRAY2BGR);
    if (LowRankPath != "")
        cv::imwrite(LowRankPath, LImg);
    if (SparsePath != "")
        cv::imwrite(SparsePath, SImg);
    if (HighlightPath != "")
        cv::imwrite(HighlightPath, HighlightImg);
    if (dstImgPath != "")
        cv::imwrite(dstImgPath, dstImg);
}

void TestHighlightRemoval_ImgSequences_withoutBorder(std::string RGBRootPath, std::string LowRankRootPath, std::string SparseRootPath, \
    std::string HighlightRootPath, std::string dstImgRootPath, int Method, int Enhancement_Method)
{
    std::vector<cv::String> file_names;
    cv::glob(RGBRootPath, file_names);
    int FilesNumber = file_names.size();

    for (int i = 0; i < FilesNumber; i++)
    {
        std::string RGBImgName = file_names.at(i);
        int pos = file_names.at(i).find_last_of("\\");
        std::string dstFileName = file_names.at(i).substr(pos + 1);
        std::string LowRankPath = LowRankRootPath + dstFileName;
        std::string dstImgPath = dstImgRootPath + dstFileName;
        std::string SparsePath = SparseRootPath + dstFileName;
        std::string HighlightPath = HighlightRootPath + dstFileName;
        ImgHighlightRemoval(RGBImgName, LowRankPath, SparsePath, HighlightPath, dstImgPath, RPCA_NONCONVEX, MSVTHRESHOLDS, EnhanceGrad);
    }
}
//

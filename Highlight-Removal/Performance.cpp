#include<Performance.h>

double CalcCOV(const cv::Mat &RGBImg, const cv::Mat &HighlightImg, double &MeanValueAve, double &StandardValAve)
{
    cv::Mat Highlight, grayImg;
    cv::cvtColor(HighlightImg, Highlight, cv::COLOR_BGR2GRAY);
    cv::threshold(Highlight, Highlight, 0, 255, cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(Highlight, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
    //std::vector<cv::Mat> BGRImgVec(3);
    //cv::split(RGBImg, BGRImgVec);
    //grayImg = BGRImgVec.at(2);

    cv::cvtColor(RGBImg, grayImg,cv::COLOR_BGR2GRAY);
    double MeanValue, Square;
    double SumCov = 0;
    int NumberOfNonZero, cnt = 0;
    MeanValueAve = 0;
    StandardValAve = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Mat Mask(HighlightImg.size(), CV_8UC1);
        Mask.setTo(0);
        double area1 = cv::contourArea(contours[i]);
        if (area1 < 10)
            continue;
        drawContours(Mask, contours, i, cv::Scalar(255), cv::FILLED, 8, hierarchy);
        NumberOfNonZero = cv::countNonZero(Mask);
        Mask &= grayImg;
        Mask.convertTo(Mask, CV_64FC1, 1.0f / 255);
        MeanValue = cv::sum(Mask)[0] * 1.0f / NumberOfNonZero;
        if (MeanValue == 0)
            continue;
        cv::multiply(Mask, Mask, Mask);
        Square = cv::sum(Mask)[0] * 1.0f / NumberOfNonZero;
        Square -= MeanValue * MeanValue;
        Square = sqrt(Square);
        SumCov += Square / MeanValue;
        cnt++;
        MeanValueAve += MeanValue;
        StandardValAve += Square;
    }
    if (cnt != 0)
        SumCov /= cnt;
    //std::cout << SumCov << std::endl;
    MeanValueAve /= cnt;
    StandardValAve /= cnt;
    return SumCov;
}

double CalcMSE(const cv::Mat &RGBImg, const cv::Mat &LowRankImage, const cv::Mat &HighlightImg)
{
    //分别在3个通道上执行计算
    int nr=RGBImg.rows;
    int nc=RGBImg.cols;
    cv::Mat mask;
    cv::cvtColor(HighlightImg,mask,cv::COLOR_BGR2GRAY);
    cv::Vec3b p1,p2;
    std::vector<double> MSE(3,0);
    for(int row=0;row<nr;row++)
        for(int col=0;col<nc;col++)
        {
            p1=RGBImg.at<cv::Vec3b>(row,col);
            p2=LowRankImage.at<cv::Vec3b>(row,col);
            if(mask.at<uchar>(row,col)==0)
            {
                MSE[0]+=(p1[0]-p2[0])*(p1[0]-p2[0]);
                MSE[1]+=(p1[1]-p2[1])*(p1[1]-p2[1]);
                MSE[0]+=(p1[2]-p2[2])*(p1[2]-p2[2]);
            }
        }
    MSE[0]/=(nr*nc);
    MSE[1]/=(nr*nc);
    MSE[2]/=(nr*nc);
    return (MSE[0]+MSE[1]+MSE[2])/3;
}

double CalcPSNR(double MSE)
{
    return 10*log10((255*255)/MSE);
}

double CalcSSIM(const cv::Mat &RGBImg, const cv::Mat &LowRankImg, const cv::Mat &HighlightImg)
{
    cv::Mat RGB,LowRank,mask;
    cv::cvtColor(HighlightImg,mask,cv::COLOR_BGR2GRAY);
    mask.convertTo(mask,CV_64FC1,1.0f/255);

    RGBImg.convertTo(RGB,CV_64FC3);
    LowRankImg.convertTo(LowRank,CV_64FC3);

    std::vector<cv::Mat> RGBSequences,LowRankSequences;
    cv::split(RGB,RGBSequences);
    cv::split(LowRank,LowRankSequences);

    std::vector<double> Mean1(3,0),Mean2(3,0);
    std::vector<double> Std1(3,0),Std2(3,0);
    std::vector<double> Cov(3,0);
    std::vector<double> dstVec(3,0);

    double k1=0.01,k2=0.03,L=255;
    double c1=(k1*L)*(k1*L);
    double c2=(k2*L)*(k2*L);
    for(int i=0;i<3;i++)
    {
        cv::Mat temp1,temp2,temp;
        cv::Scalar s1,s2;
        cv::multiply(RGBSequences[i],255-mask,temp1);

        cv::meanStdDev(temp1,s1,s2);
        Mean1[i]=s1[0];
        Std1[i]=s2[0];

        cv::multiply(LowRankSequences[i],255-mask,temp2);
        cv::meanStdDev(temp2,s1,s2);
        Mean2[i]=s1[0];
        Std2[i]=s2[0];


        cv::multiply(temp1,temp2,temp);
        cv::meanStdDev(temp,s1,s2);
        Cov[i]=s1[0]-Mean1[i]*Mean2[i];
        dstVec[i]=(2*Mean1[i]*Mean2[i]+c1)*(2*Cov[i]+c2)/((Mean1[i]*Mean1[i]+Mean2[i]*Mean2[i]+c1)*(Std1[i]*Std1[i]+Std2[i]*Std2[i]+c2));
        temp1.convertTo(temp1,CV_8UC1);
        temp2.convertTo(temp2,CV_8UC1);

        cv::waitKey(0);
    }

//    cv::imshow("RGB",RGB);
//    cv::imshow("LowRank",LowRank);
//    cv::waitKey(0);
    return (dstVec[0]+dstVec[1]+dstVec[2])/3;
}

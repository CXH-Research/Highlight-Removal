#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->displayInfor->setEnabled(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::clearWidgets()
{
    ui->srcImgLabel->clear();
    ui->dstImgLabel->clear();
    ui->displayInfor->clear();
    hasBorder=false;//默认无边界
}
void MainWindow::on_pushButton_clicked()
{
    //运行程序，需要的工作
    int DetectionMethod=ui->detectionBox->currentIndex();
    int RPCA_Type=ui->RPCATypeBox->currentIndex();
    double TC=0;
    if(DetectionMethod==0)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("please choose method for highlight detection!"));
        msgBox.exec();
    }
    else if(RPCA_Type==0)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("please choose type of RPCA algorithm!"));
        msgBox.exec();
    }
    else
    {
        //执行一串代码
        double tic=cv::getTickCount();
//        ImgHighlightRemoval(srcImage, dstImage, LImg, SImg, HighlightImg, EnhanceGrad, MSVTHRESHOLDS, RPCA_NONCONVEXBATCH, true, 2, 2);
        if(hasBorder)
        {
            TestHighlightRemoval_ImgSequences(srcImage,borderImg,LImg,SImg,dstImage,HighlightImg,MSVTHRESHOLDS,EnhanceGrad, RPCA_Type);
        }
        else
        {
            if(RPCA_Type==RPCA_NONCONVEXBATCH)
                ImgHighlightRemoval(srcImage, dstImage, LImg, SImg, HighlightImg, EnhanceGrad, MSVTHRESHOLDS, RPCA_NONCONVEXBATCH, true, 2, 2);
            else
                ImgHighlightRemoval(srcImage,dstImage,LImg,SImg,HighlightImg,EnhanceGrad,DetectionMethod,RPCA_Type);
        }

        TC=(cv::getTickCount()-tic)/cv::getTickFrequency();
        //目标图像显示
        cv::Mat dstImageShow;
        cv::cvtColor(dstImage,dstImageShow,CV_BGR2RGB);
        dstImg=QImage((const unsigned char *)(dstImageShow.data),
                      dstImageShow.cols,
                      dstImageShow.rows,
                      dstImageShow.cols*dstImageShow.channels(),
                      QImage::Format_RGB888);

        ui->dstImgLabel->clear();
        int ImgWidth=dstImg.width();
        int ImgHeight=dstImg.height();

        int LabelWidth=ui->dstImgLabel->width();
        if(ImgWidth>LabelWidth||ImgHeight>LabelWidth)
        {
            double rate=LabelWidth*1.0/MAX(ImgWidth,ImgHeight);
            int nw=static_cast<int>(rate*ImgWidth);
            int nh=static_cast<int>(rate*ImgHeight);
            dstImg=dstImg.scaled(QSize(nw,nh),Qt::KeepAspectRatio);
        }
        ui->dstImgLabel->setPixmap(QPixmap::fromImage(dstImg));

        double MSE=CalcMSE(srcImage,LImg,HighlightImg);
        double PSNR=CalcPSNR(MSE);
        double MeanVal,StdVal;
        double COV=CalcCOV(LImg,HighlightImg,MeanVal,StdVal);
        double SSIM=CalcSSIM(srcImage,LImg,HighlightImg);


        QString message1="高光去除算法执行完毕,用时:"+QString::number(TC,'d',2);
        QString str1="MSE of LowRank Image:"+QString::number(MSE,'d',2);
        QString str2="PSNR of LowRank Image:"+QString::number(PSNR,'d',2);
        QString str3="SSIM of LowRank Image:"+QString::number(SSIM,'d',2);
        QString str4="COV of LowRank Image:"+QString::number(COV,'d',2);

        ui->displayInfor->append(message1);
        ui->displayInfor->append(str1);
        ui->displayInfor->append(str2);
        ui->displayInfor->append(str3);
        ui->displayInfor->append(str4);
    }
}

void MainWindow::on_actionopen_triggered()
{
    //清空界面
    clearWidgets();
    //获取输入文件名
    QString filename = QFileDialog::getOpenFileName(this,tr("Open Image"),"",tr("Image File(*.bmp *.jpg *.jpeg *.png)"));
    QTextCodec *code = QTextCodec::codecForName("gb18030");
    std::string name = code->fromUnicode(filename).data();

    //读取文件，并判断文件的有效性
    srcImage = cv::imread(name);
    cv::Mat srcImageShow;
    if(srcImage.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Can't open the file"));
        msgBox.exec();
    }
    else
    {
        cv::cvtColor(srcImage,srcImageShow,CV_BGR2RGB);
        srcImg = QImage((const unsigned char*)(srcImageShow.data),srcImageShow.cols,srcImageShow.rows, srcImageShow.cols*srcImageShow.channels(),  QImage::Format_RGB888);
        ui->srcImgLabel->clear();
        int ImgWidth=srcImg.width();
        int ImgHeight=srcImg.height();

        int LabelWidth=ui->srcImgLabel->width();
        if(ImgWidth>LabelWidth||ImgHeight>LabelWidth)
        {
            double rate=LabelWidth*1.0/MAX(ImgWidth,ImgHeight);
            int nw=static_cast<int>(rate*ImgWidth);
            int nh=static_cast<int>(rate*ImgHeight);
            srcImg=srcImg.scaled(QSize(nw,nh),Qt::KeepAspectRatio);
        }
        ui->srcImgLabel->setPixmap(QPixmap::fromImage(srcImg));
    }
}



void MainWindow::on_actionopenborderImg_triggered()
{
    clearWidgets();
    QString filename = QFileDialog::getOpenFileName(this,tr("Open Image"),"",tr("Image File(*.bmp *.jpg *.jpeg *.png)"));
    QTextCodec *code = QTextCodec::codecForName("gb18030");
    std::string name = code->fromUnicode(filename).data();

    int i=name.find_last_of('/');
    int j=name.find_last_of('.');
    std::string pathName=name.substr(0,i+1);
    std::string fileName=name.substr(i+1,j-i-1);
    std::string extensionName=name.substr(j+1);
//    std::cout<<FileName<<std::endl;
    std::string borderFileName=pathName+fileName+ "_border." +extensionName;
    std::cout<<borderFileName<<std::endl;

    srcImage=cv::imread(name);//读取内镜图像
    borderImg=cv::imread(borderFileName);//读取边界文件
    cv::Mat srcImageShow;
    if(srcImage.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Can't open the endoscopic image file"));
        msgBox.exec();
    }
    else if(borderImg.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Can't open the border image file"));
        msgBox.exec();
    }
    else
    {
        hasBorder=true;
        cv::cvtColor(srcImage,srcImageShow,CV_BGR2RGB);
        srcImg = QImage((const unsigned char*)(srcImageShow.data),srcImageShow.cols,srcImageShow.rows, srcImageShow.cols*srcImageShow.channels(),  QImage::Format_RGB888);
        ui->srcImgLabel->clear();
        int ImgWidth=srcImg.width();
        int ImgHeight=srcImg.height();

        int LabelWidth=ui->srcImgLabel->width();
        if(ImgWidth>LabelWidth||ImgHeight>LabelWidth)
        {
            double rate=LabelWidth*1.0/MAX(ImgWidth,ImgHeight);
            int nw=static_cast<int>(rate*ImgWidth);
            int nh=static_cast<int>(rate*ImgHeight);
            srcImg=srcImg.scaled(QSize(nw,nh),Qt::KeepAspectRatio);
        }
        ui->srcImgLabel->setPixmap(QPixmap::fromImage(srcImg));
    }
}

void MainWindow::on_actionsaveRecon_triggered()
{
    QString filePath = QFileDialog::getSaveFileName(this,QString::fromLocal8Bit("保存文件"),"",tr("Images (*.png *.xpm *.jpg);;Text file(*.txt);;XML files (*.xml)"));
    QTextCodec *code = QTextCodec::codecForName("gb18030");
    std::string name = code->fromUnicode(filePath).data();
    if(dstImage.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("failed to save this image"));
        msgBox.exec();
    }
    else
        cv::imwrite(name,dstImage);
}

void MainWindow::on_actionshowSrc_triggered()
{
    if(srcImage.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("empty srcImage"));
        msgBox.exec();
    }
    else
    {
        cv::imshow("srcImage",srcImage);
        cv::waitKey(0);
    }
}

void MainWindow::on_actionshowRecon_triggered()
{
    if(dstImage.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("empty lowrank image"));
        msgBox.exec();
    }
    else
    {
        cv::imshow("destination image",dstImage);
        cv::waitKey(0);
    }
}

void MainWindow::on_actionshowLowrank_triggered()
{
    if(LImg.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("empty lowrank image"));
        msgBox.exec();
    }
    else
    {
        cv::imshow("Lowrank image",LImg);
        cv::waitKey(0);
    }
}

void MainWindow::on_actionshowSparse_triggered()
{
    if(SImg.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("empty sparse Image"));
        msgBox.exec();
    }
    else
    {
        cv::imshow("sparse Image",SImg);
        cv::waitKey(0);
    }
}

void MainWindow::on_actionshowHighlight_triggered()
{
    if(HighlightImg.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("empty srcImage"));
        msgBox.exec();
    }
    else
    {
        cv::imshow("srcImage",HighlightImg);
        cv::waitKey(0);
    }
}

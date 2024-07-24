#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QTextCodec>
#include <QString>
#include <QImage>
#include <QMessageBox>
#include <HighlightProcessing.h>
#include <Performance.h>
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void clearWidgets();

private slots:
    void on_pushButton_clicked();

    void on_actionopen_triggered();

    void on_actionopenborderImg_triggered();

    void on_actionsaveRecon_triggered();

    void on_actionshowSrc_triggered();

    void on_actionshowRecon_triggered();

    void on_actionshowLowrank_triggered();

    void on_actionshowSparse_triggered();

    void on_actionshowHighlight_triggered();

private:
    Ui::MainWindow *ui;
    cv::Mat srcImage,dstImage;
    cv::Mat LImg,SImg,HighlightImg;
    QImage srcImg,dstImg;
    cv::Mat borderImg;
    bool hasBorder;
};
#endif // MAINWINDOW_H

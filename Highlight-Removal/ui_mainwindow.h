/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.14.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionopen;
    QAction *actionopenborderImg;
    QAction *actionsaveRecon;
    QAction *actionshowSrc;
    QAction *actionshowRecon;
    QAction *actionshowLowrank;
    QAction *actionshowSparse;
    QAction *actionshowHighlight;
    QWidget *centralwidget;
    QLabel *srcImgLabel;
    QLabel *dstImgLabel;
    QComboBox *detectionBox;
    QPushButton *pushButton;
    QComboBox *RPCATypeBox;
    QTextEdit *displayInfor;
    QMenuBar *menubar;
    QMenu *menu;
    QMenu *menu_2;
    QMenu *menu_3;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 558);
        actionopen = new QAction(MainWindow);
        actionopen->setObjectName(QString::fromUtf8("actionopen"));
        actionopenborderImg = new QAction(MainWindow);
        actionopenborderImg->setObjectName(QString::fromUtf8("actionopenborderImg"));
        actionsaveRecon = new QAction(MainWindow);
        actionsaveRecon->setObjectName(QString::fromUtf8("actionsaveRecon"));
        actionshowSrc = new QAction(MainWindow);
        actionshowSrc->setObjectName(QString::fromUtf8("actionshowSrc"));
        actionshowRecon = new QAction(MainWindow);
        actionshowRecon->setObjectName(QString::fromUtf8("actionshowRecon"));
        actionshowLowrank = new QAction(MainWindow);
        actionshowLowrank->setObjectName(QString::fromUtf8("actionshowLowrank"));
        actionshowSparse = new QAction(MainWindow);
        actionshowSparse->setObjectName(QString::fromUtf8("actionshowSparse"));
        actionshowHighlight = new QAction(MainWindow);
        actionshowHighlight->setObjectName(QString::fromUtf8("actionshowHighlight"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        srcImgLabel = new QLabel(centralwidget);
        srcImgLabel->setObjectName(QString::fromUtf8("srcImgLabel"));
        srcImgLabel->setGeometry(QRect(30, 40, 350, 300));
        srcImgLabel->setFrameShape(QFrame::Box);
        srcImgLabel->setAlignment(Qt::AlignCenter);
        dstImgLabel = new QLabel(centralwidget);
        dstImgLabel->setObjectName(QString::fromUtf8("dstImgLabel"));
        dstImgLabel->setGeometry(QRect(420, 40, 350, 300));
        dstImgLabel->setFrameShape(QFrame::Box);
        dstImgLabel->setAlignment(Qt::AlignCenter);
        detectionBox = new QComboBox(centralwidget);
        detectionBox->addItem(QString());
        detectionBox->addItem(QString());
        detectionBox->addItem(QString());
        detectionBox->addItem(QString());
        detectionBox->setObjectName(QString::fromUtf8("detectionBox"));
        detectionBox->setGeometry(QRect(30, 360, 351, 41));
        QFont font;
        font.setPointSize(14);
        detectionBox->setFont(font);
        pushButton = new QPushButton(centralwidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setGeometry(QRect(700, 360, 71, 41));
        QFont font1;
        font1.setPointSize(14);
        font1.setBold(false);
        font1.setWeight(50);
        pushButton->setFont(font1);
        RPCATypeBox = new QComboBox(centralwidget);
        RPCATypeBox->addItem(QString());
        RPCATypeBox->addItem(QString());
        RPCATypeBox->addItem(QString());
        RPCATypeBox->addItem(QString());
        RPCATypeBox->addItem(QString());
        RPCATypeBox->addItem(QString());
        RPCATypeBox->addItem(QString());
        RPCATypeBox->setObjectName(QString::fromUtf8("RPCATypeBox"));
        RPCATypeBox->setGeometry(QRect(420, 360, 271, 41));
        RPCATypeBox->setFont(font);
        displayInfor = new QTextEdit(centralwidget);
        displayInfor->setObjectName(QString::fromUtf8("displayInfor"));
        displayInfor->setGeometry(QRect(30, 406, 741, 111));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 23));
        menu = new QMenu(menubar);
        menu->setObjectName(QString::fromUtf8("menu"));
        menu_2 = new QMenu(menubar);
        menu_2->setObjectName(QString::fromUtf8("menu_2"));
        menu_3 = new QMenu(menubar);
        menu_3->setObjectName(QString::fromUtf8("menu_3"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menu->menuAction());
        menubar->addAction(menu_2->menuAction());
        menubar->addAction(menu_3->menuAction());
        menu->addAction(actionopen);
        menu->addSeparator();
        menu->addAction(actionopenborderImg);
        menu->addSeparator();
        menu_2->addAction(actionsaveRecon);
        menu_3->addAction(actionshowSrc);
        menu_3->addSeparator();
        menu_3->addAction(actionshowRecon);
        menu_3->addSeparator();
        menu_3->addAction(actionshowLowrank);
        menu_3->addSeparator();
        menu_3->addAction(actionshowSparse);
        menu_3->addSeparator();
        menu_3->addAction(actionshowHighlight);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "\345\237\272\344\272\216\351\235\236\345\207\270\344\275\216\347\247\251\345\210\206\350\247\243\347\232\204\345\206\205\351\225\234\345\275\261\345\203\217\351\253\230\345\205\211\345\216\273\351\231\244", nullptr));
        actionopen->setText(QCoreApplication::translate("MainWindow", "\346\211\223\345\274\200", nullptr));
#if QT_CONFIG(shortcut)
        actionopen->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+O", nullptr));
#endif // QT_CONFIG(shortcut)
        actionopenborderImg->setText(QCoreApplication::translate("MainWindow", "\346\211\223\345\274\200\345\270\246\350\276\271\347\225\214\345\233\276\345\203\217", nullptr));
#if QT_CONFIG(shortcut)
        actionopenborderImg->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+N", nullptr));
#endif // QT_CONFIG(shortcut)
        actionsaveRecon->setText(QCoreApplication::translate("MainWindow", "\344\277\235\345\255\230\351\207\215\345\273\272\345\233\276\345\203\217", nullptr));
#if QT_CONFIG(shortcut)
        actionsaveRecon->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+S", nullptr));
#endif // QT_CONFIG(shortcut)
        actionshowSrc->setText(QCoreApplication::translate("MainWindow", "\345\261\225\347\244\272\350\276\223\345\205\245\345\233\276\345\203\217", nullptr));
        actionshowRecon->setText(QCoreApplication::translate("MainWindow", "\345\261\225\347\244\272\351\207\215\345\273\272\345\233\276\345\203\217", nullptr));
        actionshowLowrank->setText(QCoreApplication::translate("MainWindow", "\345\261\225\347\244\272\344\275\216\347\247\251\345\233\276\345\203\217", nullptr));
        actionshowSparse->setText(QCoreApplication::translate("MainWindow", "\345\261\225\347\244\272\347\250\200\347\226\217\345\233\276\345\203\217", nullptr));
        actionshowHighlight->setText(QCoreApplication::translate("MainWindow", "\345\261\225\347\244\272\351\253\230\345\205\211\345\233\276\345\203\217", nullptr));
        srcImgLabel->setText(QCoreApplication::translate("MainWindow", "source Image", nullptr));
        dstImgLabel->setText(QCoreApplication::translate("MainWindow", "destination Image", nullptr));
        detectionBox->setItemText(0, QCoreApplication::translate("MainWindow", "\351\253\230\345\205\211\346\243\200\346\265\213\346\226\271\346\263\225\351\200\211\346\213\251", nullptr));
        detectionBox->setItemText(1, QCoreApplication::translate("MainWindow", "MS\350\207\252\351\200\202\345\272\224\351\230\210\345\200\274\345\210\206\345\211\262", nullptr));
        detectionBox->setItemText(2, QCoreApplication::translate("MainWindow", "\347\273\217\351\252\214\351\230\210\345\200\274\345\210\206\345\211\262", nullptr));
        detectionBox->setItemText(3, QCoreApplication::translate("MainWindow", "MSV\350\207\252\351\200\202\345\272\224\351\230\210\345\200\274\345\210\206\345\211\262", nullptr));

        pushButton->setText(QCoreApplication::translate("MainWindow", "Run", nullptr));
        RPCATypeBox->setItemText(0, QCoreApplication::translate("MainWindow", "\344\275\216\347\247\251\347\237\251\351\230\265\345\210\206\350\247\243\347\256\227\346\263\225\351\200\211\346\213\251", nullptr));
        RPCATypeBox->setItemText(1, QCoreApplication::translate("MainWindow", "FastPCP", nullptr));
        RPCATypeBox->setItemText(2, QCoreApplication::translate("MainWindow", "LagQN", nullptr));
        RPCATypeBox->setItemText(3, QCoreApplication::translate("MainWindow", "AdaRPCA", nullptr));
        RPCATypeBox->setItemText(4, QCoreApplication::translate("MainWindow", "SPCP", nullptr));
        RPCATypeBox->setItemText(5, QCoreApplication::translate("MainWindow", "Non-RPCA", nullptr));
        RPCATypeBox->setItemText(6, QCoreApplication::translate("MainWindow", "Non-Batches RPCA", nullptr));

        menu->setTitle(QCoreApplication::translate("MainWindow", " \346\226\207\344\273\266", nullptr));
        menu_2->setTitle(QCoreApplication::translate("MainWindow", "\344\277\235\345\255\230", nullptr));
        menu_3->setTitle(QCoreApplication::translate("MainWindow", "\345\261\225\347\244\272", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H

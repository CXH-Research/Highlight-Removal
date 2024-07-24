QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp\
    ADMM_NonConv.cpp \
    AdaRPCA.cpp \
    Common.cpp \
    FPCP.cpp \
    HighlightProcessing.cpp \
    LagQN.cpp \
    Matrix_Decomposition.cpp \
    MorphologyOperator.cpp \
    PCP.cpp \
    Performance.cpp

HEADERS += \
    HighlightProcessing.h \
    Matrix_Decomposition.h \
    MorphologyOperator.h \
    Performance.h \
    RPCA.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

#openmp
#QMAKE_CXXFLAGS += -fopenmp
#QMAKE_LFLAGS += -fopenmp
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -LE:/codePackages/opencv/opencv/build/x64/vc15/lib/ -lopencv_world340
else:win32:CONFIG(debug, debug|release): LIBS += -LE:/codePackages/opencv/opencv/build/x64/vc15/lib/ -lopencv_world340d
else:unix: LIBS += -LE:/codePackages/opencv/opencv/build/x64/vc15/lib/ -lopencv_world340

INCLUDEPATH+= E:/codePackages/opencv/opencv/build/include
              E:/codePackages/opencv/opencv/build/include/opencv
              E:/codePackages/opencv/opencv/build/include/opencv2
INCLUDEPATH+= E:/codePackages/Eigen/Eigen3/Eigen3
INCLUDEPATH+= E:/codePackages/MKL/IntelSWTools/compilers_and_libraries_2019.4.245/windows/mkl/include


unix|win32: LIBS += -LE:/codePackages/MKL/IntelSWTools/compilers_and_libraries_2019.4.245/windows/compiler/lib/intel64_win/ -llibiomp5md
unix|win32: LIBS += -LE:/codePackages/MKL/IntelSWTools/compilers_and_libraries_2019.4.245/windows/mkl/lib/intel64_win/ -lmkl_core
unix|win32: LIBS += -LE:/codePackages/MKL/IntelSWTools/compilers_and_libraries_2019.4.245/windows/mkl/lib/intel64_win/ -lmkl_intel_lp64
unix|win32: LIBS += -LE:/codePackages/MKL/IntelSWTools/compilers_and_libraries_2019.4.245/windows/mkl/lib/intel64_win/ -lmkl_intel_thread

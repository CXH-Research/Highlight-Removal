# source /home/wsqlab/intel/oneapi/setvars.sh
g++ -I /usr/include/eigen3 -I/opt/intel/oneapi/mkl/latest/include -fopenmp -o tmi tmi.cpp -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

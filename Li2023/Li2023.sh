# source /home/wsqlab/intel/oneapi/setvars.sh
g++ -I /usr/include/eigen3 -I${MKLROOT}/include -fopenmp -o cmpb cmpb.cpp -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

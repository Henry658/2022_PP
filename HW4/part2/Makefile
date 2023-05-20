all: default
default: main.cc matrix.cc
	mpicxx -fPIC -O3 -mavx2 -mavx -mfma -pedantic main.cc matrix.cc -o matmul -ffast-math -funsafe-math-optimizations -floop-parallelize-all -fprefetch-loop-arrays -fno-tree-vectorize -w

	


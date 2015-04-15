torch_path = '/home/mbhenaff/torch_sc/build/distro/install/'

all:
	nvcc -arch=sm_35 -lTH -lTHC -L /usr/local/pkg/cuda/6.5/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include -I ${torch_path}/include/TH -I ${torch_path}/include/THC init.cu cufft.cpp -o spectralcuda.so


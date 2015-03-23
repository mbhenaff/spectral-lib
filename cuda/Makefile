#torch_path = "/misc/vlgscratch3/LecunGroup/bruna/torch-private/torch/installed/"
#torch_path = "/home/mbhenaff/lib/torch-private/torch/installed/"
#torch_path = "/home/mbhenaff/lib/torch-private/torch/installed/"

torch_path = '/home/mbhenaff/torch_sc/build/distro/install/'


#all:
#	nvcc -arch=sm_35 -lTH -lTHC -L /usr/local/pkg/cuda/6.5/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include/torch -I ${torch_path}/include/torch/TH -I ${torch_path}/include/torch/THC cufft.cpp -o libcufft.so
#	nvcc -arch=sm_35 -lTH -lTHC -L /usr/local/pkg/cuda/6.5/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include/torch -I ${torch_path}/include/torch/TH -I ${torch_path}/include/torch/THC init.cu -o spectralcuda.so


all:
	nvcc -arch=sm_35 -lTH -lTHC -L /usr/local/pkg/cuda/6.5/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include -I ${torch_path}/include/TH -I ${torch_path}/include/THC cufft.cpp -o libcufft.so
	nvcc -arch=sm_35 -lTH -lTHC -L /usr/local/pkg/cuda/6.5/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include -I ${torch_path}/include/TH -I ${torch_path}/include/THC init.cu -o spectralcuda.so


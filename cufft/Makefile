torch_path = "/misc/vlgscratch3/LecunGroup/bruna/torch-private/torch/installed/"

all:
	nvcc -arch=sm_35 -lTH -lTHC -L /usr/local/pkg/cuda/5.5/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include/torch -I ${torch_path}/include/torch/TH -I ${torch_path}/include/torch/THC cufft.cpp -o libcufft.so
	nvcc -arch=sm_35 -lTH -lTHC -L /usr/local/pkg/cuda/5.5/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include/torch -I ${torch_path}/include/torch/TH -I ${torch_path}/include/torch/THC complex_prod.cu -o cucomplex.so


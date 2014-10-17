torch_path = "/home/mbhenaff/lib/torch-private/torch/installed/"

all:
	nvcc -lTH -lTHC -L /usr/local/cuda/cuda/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include/torch -I ${torch_path}/include/torch/TH -I ${torch_path}/include/torch/THC cufft.cpp -o libcufft.so
	
	nvcc -lTH -lTHC -L /usr/local/cuda/cuda/lib64 -lcuda -lcufft -Xcompiler -fPIC -shared -O3 -I ${torch_path}/include/torch -I ${torch_path}/include/torch/TH -I ${torch_path}/include/torch/THC complex_prod.cu -o cucomplex.so


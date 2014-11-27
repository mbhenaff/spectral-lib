To run experiments, use the train.lua script with the desired options (see params.lua).

For example, 

torch -i train.lua -conv spectral -dataset cifar -kH 5 -kW 5 -interp spline -real mod -gpunum 1
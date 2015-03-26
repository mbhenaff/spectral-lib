To run experiments, cd to test. 
All parameters are listed in params.lua.
To modify the GFT matrix, change the path at the beginning of model.lua to load the proper thing. 


The -model parameter specfies model architecture. fc2 refers to fully connected with 2 layers, etc. gconv2 refers to 2 layers of graph convolution followed by a fully connected layer. 
The -nhidden parameter refers to number of hidden units for fully connected models and number of feature maps for graph convolutions. 
The -k parameter refers to the number of subsampled weights in the graph convolution layers.  


To train on TIMIT with a 3-layer fully connected net with 2000 hidden units:

th -i train.lua -dataset timit -model fc3 -nhidden 2000

To train on TIMIT with a net consisting of 2 layers of graph conv followed by a fully connected, with 20 feature maps and 20 subsampled weights at each layer:

th -i train.lua -dataset timit -model gconv2 -nhidden 20 -k 20

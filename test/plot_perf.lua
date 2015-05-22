dofile('params.lua')
cutorch.setDevice(3)
dataset = 'cifar'
savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/new/gc3/merck/'
epochs = 100
plots = {}

--stopwords= {'npts=100','weightDecay=0.01','weightDecay=0.002','weightDecay=0.005'}
stopwords = {}

indx = 1
for f in paths.files(savePath) do
   file=f
   if file:sub(-5,-1) == 'model' and string.match(file,dataset) then
      local stop = false 
      for i = 1,#stopwords do 
         if string.match(file,stopwords[i]) then stop = true end
      end
      if not stop then
         local x = torch.load(savePath .. file)
         local name = file
         name = string.gsub(name,'%.model','')
         name = string.gsub(name,'batchSize%-128','')
         name = string.gsub(name, 'optim=sgd','')
         --      name = string.gsub(name,'learningRate=0.01','')
         if x.teloss:nElement() >= epochs then
            plots[indx] = {name,torch.range(1,epochs),x.teloss[{{1,epochs}}],'-'}
            indx = indx + 1
         end
      end
      collectgarbage()
   end
end
print(plots)
gnuplot.pdffigure('plot.pdf')
gnuplot.plot(plots)
gnuplot.movelegend('right','bottom')
gnuplot.xlabel('epoch')
gnuplot.ylabel('test accuracy')
gnuplot.raw('set key font ",2"')
gnuplot.plotflush()

   
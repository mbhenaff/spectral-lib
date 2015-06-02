dofile('params.lua')
cutorch.setDevice(3)
dataset = 'merck5'
savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/new/merck/'
epochs = 200
plots = {}

stopwords= {'fc2'}
--stopwords = {}

indx = 1
for f in paths.files(savePath) do
   file=f
   if file:sub(-5,-1) == 'model' and string.match(file,dataset) then
      local stop = true 
      if string.match(file,'dnn4') then stop = false end
--      if (string.match(file,'fc')) then stop = false end
      for i = 1,#stopwords do 
         if string.match(file,stopwords[i]) then stop = true end
      end
      --stop = true
      print(file)
      if not stop then
         local x = torch.load(savePath .. file)
         --print(file)
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

   
dofile('params.lua')

savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/'
epochs = 20
plots = {}

indx = 1
for f in paths.files(savePath) do
   file=f
   if file:sub(-5,-1) == 'model' then
      x = torch.load(savePath .. file)
      plots[indx] = {file:sub(15,-6),torch.range(1,epochs),x.teacc,'-'}
      indx = indx + 1
   end
end
gnuplot.pdffigure('plot.pdf')
gnuplot.plot(plots)
gnuplot.movelegend('right','bottom')
gnuplot.xlabel('epoch')
gnuplot.ylabel('test accuracy')
gnuplot.raw('set key font ",2"')
gnuplot.plotflush()

   
dofile('params.lua')

model = torch.load(opt.savePath .. opt.modelFile .. '.model').model
real,imag,mod = model:get(1):printFilters()
local t = image.toDisplayTensor(real)
image.save(opt.savePath .. opt.modelFile .. '-filters-real.png',t)
local t = image.toDisplayTensor(imag)
image.save(opt.savePath .. opt.modelFile .. '-filters-imag.png',t)
local t = image.toDisplayTensor(mod)
image.save(opt.savePath .. opt.modelFile .. '-filters-modulus.png',t)



require 'torch'
require 'nn'
require 'cudnn'
require 'paths'

require 'bnn'
require 'optim'

require 'gnuplot'
require 'image'
require 'xlua'
local utils = require 'utils'
local opts = require('opts')(arg)

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local model = torch.load('models/facealignment_binary_aflw.t7')
model:evaluate()

local fileLists = utils.getFileList(opts)
local predictions = {}
local noPoints = 68
local output = torch.Tensor(1,noPoints,64,64)

for i = 1, #fileLists do	
	local img = image.load(fileLists[i].image)
	local originalSize = img:size()

	img = utils.crop(img, fileLists[i].center, fileLists[i].scale, 256)
	img = img:view(1,3,256,256)
	
	output:copy(model:forward(img))
	output:add(utils.flip(utils.shuffleLR(opts, model:forward(utils.flip(img)))))

	local preds_hm, preds_img = utils.getPreds(output, fileLists[i].center, fileLists[i].scale)
	
	utils.plot(fileLists[i].image,preds_img:view(noPoints,2),torch.Tensor{originalSize[3],originalSize[2]})
	io.read() -- Wait for user input
	
end

if opts.mode == 'demo' then gnuplot.closeall() end

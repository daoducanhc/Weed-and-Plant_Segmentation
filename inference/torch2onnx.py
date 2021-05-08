import onnx
import os
import torch
import numpy as np
import sys
sys.path.append('/home/daoducanhc/Desktop/Weed-and-Plant_Segmentation')
import setup.dataset as dataset
import setup.ResUNet as ResUNet
import setup.classifier as classifier
from torch.utils.data import SubsetRandomSampler

np.random.seed(0)
torch.manual_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

FILTER_LIST = [16,32,64,128,256]

# # print(os. getcwd())
# path_parent = os. path. dirname(os. getcwd())
# os. chdir(path_parent)
# # print(os. getcwd())

model = ResUNet.ResUNet(FILTER_LIST).to(device)
path = 'outputs/ResUNet.pt'

classifier = classifier.WeedClassifier(model, device)
if torch.cuda.is_available():
    classifier.model.load_state_dict(torch.load(path))    
else:
    classifier.model.load_state_dict(torch.load(path, map_location='cpu'))

model.eval()

batch_size = 2
x = torch.rand(batch_size, 4, 512, 512, requires_grad=True).to(device)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "outputs/model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


onnx_model = onnx.load("outputs/model.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
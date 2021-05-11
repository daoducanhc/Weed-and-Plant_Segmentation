import inference.inference as inference
import inference.engine as engine
import torch

engine = engine.load_engine('outputs/ResUNet.plan')

input = torch.rand((1, 4, 512, 512))
print(inference.do_inference(engine, input).shape)

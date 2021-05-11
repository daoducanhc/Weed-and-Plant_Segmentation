import inference.inference as inference
import inference.engine as engine
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch

def do_inference(engine, input, data_type=np.float32):
    with engine.create_execution_context() as context:
        h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=data_type)
        h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=data_type)
        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        input = np.array(input, dtype=np.float32, order='C')
        h_input = input

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        output = h_output.reshape((input.shape[0], 3, 512, 512))
        return torch.tensor(output)
        # output = torch.tensor(output)
        # output = torch.argmax(output, dim=1)
        # print(output.sum())

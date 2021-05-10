import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, shape=[2,4,512,512]):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    # convert GB to bytes: shift to left by 30 bits << 30
    # convert MB to bytes: shift to left by 20 bits << 20
    config.max_workspace_size = 1 << 30
    config.set_flag(trt.BuilderFlag.FP16)
    network.get_input(0).shape = shape
    engine = builder.build_engine(network, config)
    return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
       f.write(buf)

def load_engine(plan_path):
    with open(plan_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

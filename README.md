# Weed-and-Plant_Segmentation.

conda create -n tung python=3.8.5

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install -c anaconda cudnn

python3 -m pip install --upgrade setuptools pip

python3 -m pip install nvidia-pyindex

python3 -m pip install --upgrade nvidia-tensorrt==8.0.0.3

3080? 3090 RTX have to update cudatoolkit to 11.1

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/

https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/?fbclid=IwAR2Aympuu6zjBA6RA9dP9efTC-IUO8EXMU1doqv1zbesgMdArs6wlPA7RxI

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#build_model

https://netron.app/

https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html

# The organizers' base image: your code is re-run inside this image with no network.
# Build it once:   docker build -t eurs-base docker/
# Exact versions: these are the versions in the image the organizers run (PyTorch 2.4.0 comes with the base image), so that building this
# file reproduces it. pillow and pyyaml already come with the PyTorch image; they are listed so the versions are explicit.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN pip install --no-cache-dir numpy==2.4.6 pillow==10.4.0 opencv-python-headless==5.0.0.93 timm==1.0.30 einops==0.8.2 onnx==1.23.2 pyyaml==6.0.1 scipy==1.17.1 scikit-image==0.26.0 huggingface_hub==2.2.0 httpx2==2.13.1 httpcore2==2.13.1 truststore==0.10.4
WORKDIR /pkg

FROM python:3.12.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=4
WORKDIR /app
COPY wheelhouse /wheelhouse
COPY requirements-linux.lock wheelhouse-manifest.json /app/
COPY tools/verify_wheelhouse.py /app/tools/verify_wheelhouse.py
RUN python /app/tools/verify_wheelhouse.py --directory /wheelhouse --manifest /app/wheelhouse-manifest.json
ENV PIP_NO_INDEX=1 PIP_FIND_LINKS=/wheelhouse PIP_CONSTRAINT=/app/requirements-linux.lock SAM2_BUILD_CUDA=0
RUN pip install --no-cache-dir setuptools==75.6.0 wheel==0.45.1
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 numpy==1.26.4 Pillow==11.1.0
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY tests /app/tests
COPY tools /app/tools
RUN pip install --no-cache-dir --no-build-isolation . httpx==0.28.1 pytest==8.3.4 omegaconf==2.3.0
RUN pip install --no-cache-dir --no-build-isolation /wheelhouse/sam2-2b90b9f.zip
RUN pip freeze --all > /app/environment.lock
ENV VISION_DATA=/data VISION_MODELS=/models
EXPOSE 18761
CMD ["image2ppt-vision", "serve", "--host", "0.0.0.0", "--port", "18761"]

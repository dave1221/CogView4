# 使用包含CUDA 12.1和cuDNN 8的Ubuntu 22.04基础镜像
ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS runtime

# 设置环境变量避免交互式提示
# 日志立即输出，stdout/stderr立即刷新（不走缓冲）
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 安装系统依赖，后两个是Gradio可能需要的图形库依赖
# 使用 OpenCV（cv2.imshow()）或需要解析.webp/.svg/.gif时才会触发libGL。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git git-lfs build-essential  \
        curl tar wget libsndfile1 \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 软链 python → python3
RUN ln -s /usr/bin/python3 /usr/bin/python && git lfs install

# 安装UV
RUN curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install.sh  \
    && bash -x /tmp/install.sh
ENV PATH="/root/.local/bin:${PATH}"

# 设置工作目录
WORKDIR /app

# 把 requirements.txt 拷进去（先 COPY，它一变镜像就会重新缓存）
COPY ../inference/requirements.txt .

# --system 让 uv 直接写入系统 site-packages（而非建 venv）
RUN uv pip install --system -r requirements.txt \
    && uv pip install --system torchao==0.11.0 --index-url https://pypi.org/simple

# 复制应用程序文件
# 若目录结构为  <项目根>/inference/gradio_web_demo_custom.py
COPY ../inference/gradio_web_demo_custom.py .

# 预留模型挂载路径；容器内用 /models 目录
RUN mkdir -p /models

# 暴露Gradio服务端口（与启动参数中的--server-port一致）
EXPOSE 7134

# ENTRYPOINT 固定脚本；CMD 用来给脚本传参，可在 docker run 时覆盖
ENTRYPOINT ["python", "-u", "gradio_web_demo_custom.py"]
CMD ["-c", "/models/CogView4-6B", "--mode", "3", \
     "--server-port", "7134", "--server-name", "0.0.0.0" ]

# docker构建
# docker build -f docker/Dockerfile -t cogview4-demo:stable .
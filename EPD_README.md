1. 部署vLLM和vLLM ascend
1.1 下载vLLM和vLLM ascend代码包
wget https://github.com/hsliuustc0106/vllm/archive/refs/tags/v0.9.1-EPD.tar.gz
wget -0 v0.9.1-ascend-EPD.tar.gz https://github.com/hsliuustc0106/vllm-ascend/archive/refs/tags/v0.9.1-EPD.tar.gz

1.2 解压代码包
tar -zxvf v0.9.1-EPD.tar.gz
tar -zxvf v0.9.1-ascend-EPD.tar.gz

1.3 部署vLLM
cd vllm-0.9.1-EPD
pip install -r requirements/build.txt

SETUPTOOLS_SCM_PRETEND_VERSION=0.9.1 VLLM_TARGET_DEVICE=empty pip install -e .

1.4 部署vLLM-ascend
cd ../vllm-ascend-0.9.1-EPD
pip install -e .

2. 启动E + PD和推理服务(python API + zmq)
cd ../vllm-0.9.1-EPD/examples/offline_inference/epd
增加模型patch和修改run.sh参数
在vllm-0.9.1-EPD/examples/offline_inference/epd/chat_with_image.py文件下修改推理参数
执行bash run.sh进行推理

3. 启动E + PD和推理服务(zmq + http)
cd ../vllm-0.9.1-EPD/examples/offline_inference/epd
增加模型patch和修改run_zmq_http.sh参数
执行bash run_zmq_http.sh启动推理服务

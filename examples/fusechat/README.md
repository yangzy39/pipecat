# 基于Qwen3-ASR + FuseChat-3.0 + Qwen3-TTS实现的端到端语音交互框架

## 核心修改

pipecat/examples/fusechat/bot.py （pipeline定义，执行入口）
pipecat/src/pipecat/services/qwen （实现dashscope调用api）

## Quickstart

> Run your first Pipecat bot in under 5 minutes

This quickstart guide will help you build and deploy your first Pipecat voice AI bot. You'll create a simple conversational agent that you can talk to in real-time, then deploy it to production on Pipecat Cloud.

**Two steps**: Local Development (5 min) → Production Deployment (5 min)

### Prerequisites

**Environment**

* Python 3.10 or later
* [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager installed

### Setup

1. Clone the quickstart repository

```bash  theme={null}
git clone https://github.com/pipecat-ai/pipecat-quickstart.git
cd pipecat-quickstart
```

2. Set up virtual environment and install dependencies

```bash  theme={null}
uv sync
```

### Run your bot locally

1. Deploy FuseChat model using vllm

```bash
export BIN="<YOUR_PYTHON_EXE_DIR>"
export MODEL_CKPT="MODEL_PATH"

${BIN}/vllm serve ${MODEL_CKPT} \
    --served-model-name Fusechat-3.0 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --port 23547 
```

2. Configure your API keys

set api_key in [pipecat/examples/fusechat/bot.py](https://github.com/yangzy39/pipecat/blob/a04c4e421c2573feb267d3f45a038cccbd71f143/examples/fusechat/bot.py#L127)
set vllm model name and host in [pipecat/examples/fusechat/bot.py](https://github.com/yangzy39/pipecat/blob/a04c4e421c2573feb267d3f45a038cccbd71f143/examples/fusechat/bot.py#L159)

3. Start server
```bash  theme={null}
uv run bot.py
```

You should see output similar to this:

```
🚀 WebRTC server starting at http://localhost:7860/client
   Open this URL in your browser to connect!
```

Open [http://localhost:7860/client](http://localhost:7860/client) in your browser and click **Connect** to start talking to your bot.


🎉 **Success!** Your bot is running locally. Now let's deploy it to production so others can use it.

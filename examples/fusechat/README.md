<img width="1650" height="1181" alt="image" src="https://github.com/user-attachments/assets/cac2e77e-3912-4843-ac55-300b789e9124" /># 基于Qwen3-ASR + FuseChat-3.0 + Qwen3-TTS实现的端到端语音交互框架


## 核心修改

pipecat/examples/fusechat/bot.py （pipeline定义，执行入口）

pipecat/src/pipecat/services/qwen （实现dashscope调用api）

## Quickstart

### Prerequisites

**Environment**

* Python 3.12

### Setup

1. Install pipecat
   
```bash  theme={null}
cd pipecat
pip install -e .
```

2. Set up virtual environment and install dependencies

```bash  theme={null}
pip install -r requirements.txt
```

3. Download Ollama

3.1 See https://ollama.com/ to download ollama

3.2 Download model

```bash  theme={null}
# download gguf model
ollama pull modelscope.cn/bartowski/FuseChat-Qwen-2.5-7B-Instruct-GGUF
# rename
ollama cp modelscope.cn/bartowski/FuseChat-Qwen-2.5-7B-Instruct-GGUF FuseChat-3.0-7B
ollama rm modelscope.cn/bartowski/FuseChat-Qwen-2.5-7B-Instruct-GGUF
```

4. Download ngrok

See https://ngrok.com/

   

### Run your bot locally

1. Deploy FuseChat model using ollama

```bash  theme={null}
ollama run FuseChat-3.0-7B
```

2. Configure your Dash API keys

export DASHSCOPE_API_KEY="sk-xx"

3. Generate ngrok proxy

```bash  theme={null}
ngrok http 8000 --region=jp
```

copy the proxy (https://xxx.ngrok-free.dev) from output like

```
Forwarding                    https://xxx.ngrok-free.dev -> http://localhost:8000
```

4. Start server
```bash  theme={null}
python bot.py --port 8000 -proxy <your_ngrok_proxy>
```

You should see output similar to this:

```
🚀 WebRTC server starting at http://localhost:8000/client
   Open this URL in your browser to connect!
```

Open [http://localhost:8000/client](http://localhost:8000/client) in your browser or <your_ngrok_proxy> on another device, then click **Connect** to start talking to your bot.


🎉 **Success!** Your bot is running locally. Now let's deploy it to production so others can use it.

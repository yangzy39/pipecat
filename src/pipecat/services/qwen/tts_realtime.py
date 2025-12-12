import os
import base64
import asyncio
import threading
import dashscope
import re
from typing import AsyncGenerator, Dict, Optional, List
from loguru import logger

# 尝试导入实时语音 SDK，处理可能的依赖缺失
try:
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
        AudioFormat
    )
except ImportError:
    logger.error("Missing 'dashscope' SDK or version too low. Please run: pip install dashscope>=1.20.0")
    raise

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# os.environ['SSL_CERT_FILE'] = certifi.where()
# os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


# # 定义支持的音色列表
VALID_VOICES: Dict[str, str] = {
    # 常用推荐
    "cherry": "Cherry",       # 芊悦
    "serena": "Serena",       # 苏瑶
    "ethan": "Ethan",         # 晨煦
    
    # 特色音色
    "chelsie": "Chelsie",     # 千雪
    "momo": "Momo",           # 茉兔
    "vivian": "Vivian",       # 十三
    "moon": "Moon",           # 月白
    "maia": "Maia",           # 四月
    "kai": "Kai",             # 凯
    "nofish": "Nofish",       # 不吃鱼
    "bella": "Bella",         # 萌宝
    "jennifer": "Jennifer",   # 詹妮弗
    "ryan": "Ryan",           # 甜茶
    "katerina": "Katerina",   # 卡捷琳娜
    "aiden": "Aiden",         # 艾登
    "eldric": "Eldric Sage",  # 沧明子
    "mia": "Mia",             # 乖小妹
    "mochi": "Mochi",         # 沙小弥
    "bellona": "Bellona",     # 燕铮莺
    "vincent": "Vincent",     # 田叔
    "bunny": "Bunny",         # 萌小姬
    "neil": "Neil",           # 阿闻
    "elias": "Elias",         # 墨讲师
    "arthur": "Arthur",       # 徐大爷
    "nini": "Nini",           # 邻家妹妹
    "ebona": "Ebona",         # 诡婆婆
    "seren": "Seren",         # 小婉
    "pip": "Pip",             # 顽屁小孩
    "stella": "Stella",       # 少女阿月
    
    # 方言与外语特色
    "bodega": "Bodega",       # 西班牙大叔
    "sonrisa": "Sonrisa",     # 拉美大姐
    "alek": "Alek",           # 战斗民族
    "dolce": "Dolce",         # 意大利大叔
    "sohee": "Sohee",         # 韩国欧尼
    "ono": "Ono Anna",        # 小野杏
    "lenn": "Lenn",           # 德国青年
    "emilien": "Emilien",     # 法国大哥哥
    "andre": "Andre",         # 沉稳男生
    "radio": "Radio Gol",     # 足球诗人
    
    # 中国方言
    "jada": "Jada",           # 上海-阿珍
    "dylan": "Dylan",         # 北京-晓东
    "li": "Li",               # 南京-老李
    "marcus": "Marcus",       # 陕西-秦川
    "roy": "Roy",             # 闽南-阿杰
    "peter": "Peter",         # 天津-李彼得
    "sunny": "Sunny",         # 四川-晴儿
    "eric": "Eric",           # 四川-程川
    "rocky": "Rocky",         # 粤语-阿强
    "kiki": "Kiki",           # 粤语-阿清
}


class _PipecatDashScopeCallback(QwenTtsRealtimeCallback):
    """
    内部回调类，用于将 DashScope SDK 的回调事件桥接到 asyncio 队列中。
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self.session_id = None

    def on_open(self) -> None:
        # 连接建立，一般不需要特别处理，等待 session.created
        pass

    def on_close(self, close_status_code, close_msg) -> None:
        # 如果非正常关闭，可以在这里记录日志
        if close_status_code != 1000:
            logger.warning(f"DashScope WS closed: {close_status_code} {close_msg}")

    def on_event(self, response: dict) -> None:
        try:
            msg_type = response.get('type')
            
            if msg_type == 'session.created':
                self.session_id = response['session']['id']
                logger.debug(f"DashScope Session Created: {self.session_id}")
            
            elif msg_type == 'response.audio.delta':
                # 接收到音频数据 (Base64编码)
                b64_data = response.get('delta')
                if b64_data:
                    pcm_data = base64.b64decode(b64_data)
                    # 线程安全地放入队列
                    self.loop.call_soon_threadsafe(self.queue.put_nowait, pcm_data)
            
            elif msg_type == 'session.finished':
                logger.debug(f"DashScope Session Finished: {self.session_id}")
                # 发送结束信号 (None)
                self.loop.call_soon_threadsafe(self.queue.put_nowait, None)
                
            elif msg_type == 'error':
                 # 处理错误
                error_msg = response.get('message', 'Unknown DashScope Error')
                code = response.get('code', 'Unknown')
                logger.error(f"DashScope Error {code}: {error_msg}")
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait, Exception(f"DashScope Error {code}: {error_msg}")
                )

        except Exception as e:
            logger.error(f"Error in DashScope callback: {e}")
            self.loop.call_soon_threadsafe(self.queue.put_nowait, e)


class DashScopeTTSRealTimeService(TTSService):
    """
    DashScope (Aliyun) Realtime TTS Service integration for Pipecat.
    Uses WebSocket (QwenTtsRealtime) for low-latency streaming.
    """

    DASHSCOPE_SAMPLE_RATE = 24000
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice: str = "cherry",
        model: str = "qwen3-tts-flash-realtime-2025-11-27", # 推荐使用 realtime 模型
        sample_rate: Optional[int] = 24000,
        wss_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime", # 默认北京，新加坡需更改
        **kwargs,
    ):
        target_sample_rate = sample_rate or self.DASHSCOPE_SAMPLE_RATE
        
        if sample_rate and sample_rate != self.DASHSCOPE_SAMPLE_RATE:
            logger.warning(
                f"DashScope Realtime TTS outputs {self.DASHSCOPE_SAMPLE_RATE}Hz. "
                f"Resampling may be required if {sample_rate}Hz is strictly needed by downstream."
            )
        
        super().__init__(sample_rate=target_sample_rate, **kwargs)

        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DashScope API Key is required.")
        
        # 全局设置 API Key
        dashscope.api_key = self._api_key
        
        self.set_model_name(model)
        self.set_voice(voice)
        self._wss_url = wss_url

    def set_voice(self, voice: str):
        voice_lower = voice.lower()
        if voice_lower in VALID_VOICES:
            self._voice_id = VALID_VOICES[voice_lower]
        else:
            logger.warning(f"Voice '{voice}' not found in internal map, using raw ID.")
            self._voice_id = voice

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        # Remove markdown syntax before TTS
        # text = self._remove_markdown(text)
        text = text.replace("\n", " ")
        logger.debug(f"{self}: Generating Realtime TTS for [{text}]")

        try:
            await self.start_ttfb_metrics()

            # 1. 准备异步队列和 EventLoop
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            # 2. 初始化回调
            callback = _PipecatDashScopeCallback(loop, queue)

            # 3. 在独立线程中运行 SDK 逻辑，避免阻塞 asyncio loop
            # DashScope Python SDK 的 connect/send 可能是同步阻塞的
            def run_sdk_client():
                try:
                    client = QwenTtsRealtime(
                        model=self.model_name,
                        callback=callback,
                        url=self._wss_url
                    )
                    client.connect()

                    # 更新 Session 配置
                    client.update_session(
                        voice=self._voice_id,
                        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                        mode='server_commit' # 提交模式，只有明确 finish 才会结束 session
                    )

                    # 发送文本
                    client.append_text(text)

                    # 结束流 (触发合成)
                    client.finish()

                except Exception as e:
                    logger.error(f"DashScope SDK Thread Error: {e}")
                    loop.call_soon_threadsafe(queue.put_nowait, e)

            # 启动 SDK 线程
            threading.Thread(target=run_sdk_client, daemon=True).start()

            yield TTSStartedFrame()

            first_chunk_received = False

            # 4. 从队列中读取音频数据并 Yield
            while True:
                # 等待数据，如果 SDK 线程异常或网络卡顿，这里会异步等待
                item = await queue.get()

                # 收到 None 表示结束
                if item is None:
                    break

                # 收到 Exception 表示出错
                if isinstance(item, Exception):
                    raise item

                # 收到 Bytes 数据
                if isinstance(item, bytes):
                    if not first_chunk_received:
                        await self.stop_ttfb_metrics()
                        first_chunk_received = True

                    yield TTSAudioRawFrame(
                        audio=item,
                        sample_rate=self.sample_rate,
                        num_channels=1
                    )

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"{self} Exception: {e}")
            yield ErrorFrame(error=f"DashScope TTS execution error: {str(e)}")

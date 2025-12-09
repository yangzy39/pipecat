#
# DashScope (Aliyun) TTS Service for Pipecat
# 基于 qwen3-tts-flash 模型
#

import os
import base64
import numpy as np
import dashscope
from typing import AsyncGenerator, Dict, Literal, Optional
from loguru import logger
from http import HTTPStatus

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

# 定义支持的音色列表 (根据您提供的表格整理)
VALID_VOICES: Dict[str, str] = {
    # 常用推荐
    "cherry": "Cherry",       # 芊悦 - 阳光积极、亲切自然小姐姐
    "serena": "Serena",       # 苏瑶 - 温柔小姐姐
    "ethan": "Ethan",         # 晨煦 - 标准普通话，阳光暖男
    
    # 特色音色
    "chelsie": "Chelsie",     # 千雪 - 二次元虚拟女友
    "momo": "Momo",           # 茉兔 - 撒娇搞怪
    "vivian": "Vivian",       # 十三 - 拽拽的小暴躁
    "moon": "Moon",           # 月白 - 率性帅气
    "maia": "Maia",           # 四月 - 知性与温柔
    "kai": "Kai",             # 凯 - 耳朵SPA
    "nofish": "Nofish",       # 不吃鱼 - 不会翘舌音的设计师
    "bella": "Bella",         # 萌宝 - 喝酒不打醉拳的小萝莉
    "jennifer": "Jennifer",   # 詹妮弗 - 电影质感美语
    "ryan": "Ryan",           # 甜茶 - 节奏拉满
    "katerina": "Katerina",   # 卡捷琳娜 - 御姐音色
    "aiden": "Aiden",         # 艾登 - 美语大男孩
    "eldric": "Eldric Sage",  # 沧明子 - 沉稳睿智老者
    "mia": "Mia",             # 乖小妹 - 温顺如春水
    "mochi": "Mochi",         # 沙小弥 - 聪明伶俐小大人
    "bellona": "Bellona",     # 燕铮莺 - 声音洪亮，江湖气
    "vincent": "Vincent",     # 田叔 - 沙哑烟嗓
    "bunny": "Bunny",         # 萌小姬 - 萌属性爆棚
    "neil": "Neil",           # 阿闻 - 专业新闻主持
    "elias": "Elias",         # 墨讲师 - 严谨叙事
    "arthur": "Arthur",       # 徐大爷 - 质朴旱烟嗓
    "nini": "Nini",           # 邻家妹妹 - 软糯甜美
    "ebona": "Ebona",         # 诡婆婆 - 恐怖童年阴影
    "seren": "Seren",         # 小婉 - 助眠声线
    "pip": "Pip",             # 顽屁小孩 - 调皮捣蛋
    "stella": "Stella",       # 少女阿月 - 迷糊少女/正义战士
    
    # 方言与外语特色
    "bodega": "Bodega",       # 博德加 - 西班牙大叔
    "sonrisa": "Sonrisa",     # 索尼莎 - 拉美大姐
    "alek": "Alek",           # 阿列克 - 战斗民族
    "dolce": "Dolce",         # 多尔切 - 意大利大叔
    "sohee": "Sohee",         # 素熙 - 韩国欧尼
    "ono": "Ono Anna",        # 小野杏 - 鬼灵精怪
    "lenn": "Lenn",           # 莱恩 - 德国青年
    "emilien": "Emilien",     # 埃米尔安 - 法国大哥哥
    "andre": "Andre",         # 安德雷 - 沉稳男生
    "radio": "Radio Gol",     # 足球诗人 - 解说风
    
    # 中国方言
    "jada": "Jada",           # 上海-阿珍
    "dylan": "Dylan",         # 北京-晓东
    "li": "Li",               # 南京-老李
    "marcus": "Marcus",       # 陕西-秦川
    "roy": "Roy",             # 闽南-阿杰 (台普)
    "peter": "Peter",         # 天津-李彼得 (相声风)
    "sunny": "Sunny",         # 四川-晴儿
    "eric": "Eric",           # 四川-程川
    "rocky": "Rocky",         # 粤语-阿强
    "kiki": "Kiki",           # 粤语-阿清
}

class DashScopeTTSService(TTSService):
    """DashScope (Aliyun) TTS Service integration for Pipecat."""

    # Qwen3-TTS-Flash / CosyVoice 默认输出采样率为 24000Hz
    DASHSCOPE_SAMPLE_RATE = 24000

    def __init__(
            self,
            *,
            api_key: Optional[str] = None,
            voice: str = "cherry",
            model: str = "qwen3-tts-flash",
            sample_rate: Optional[int] = 24000,
            **kwargs,
        ):
            """
            初始化 DashScope TTS 服务。
            """
            # 1. 确定最终使用的采样率
            target_sample_rate = sample_rate or self.DASHSCOPE_SAMPLE_RATE
            
            # 2. 检查采样率警告
            if sample_rate and sample_rate != self.DASHSCOPE_SAMPLE_RATE:
                logger.warning(
                    f"DashScope TTS usually outputs {self.DASHSCOPE_SAMPLE_RATE}Hz. "
                    f"Current config {sample_rate}Hz might cause playback speed issues."
                )
            
            # 3. 调用父类初始化
            super().__init__(sample_rate=target_sample_rate, **kwargs)

            self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            if not self._api_key:
                raise ValueError("DashScope API Key is required.")

            # 配置 DashScope
            dashscope.api_key = self._api_key
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

            self.set_model_name(model)
            self.set_voice(voice)

    def set_voice(self, voice: str):
        """设置音色，支持大小写不敏感匹配"""
        voice_lower = voice.lower()
        if voice_lower in VALID_VOICES:
            self._voice_id = VALID_VOICES[voice_lower]
        else:
            # 如果找不到映射，假设用户直接传了 DashScope 的原始 ID (如 "Cherry")
            logger.warning(f"Voice '{voice}' not found in internal map, using raw ID.")
            self._voice_id = voice

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        调用 DashScope MultiModalConversation 接口进行流式 TTS 生成。
        """
        logger.debug(f"{self}: Generating TTS for text: [{text}]")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # 构建请求参数
            # 注意：DashScope Python SDK 的 call 方法目前是同步阻塞的。
            # 在高并发生产环境中，建议将其放入 thread executor 中运行，
            # 但为了保持流式响应的简单性，此处直接在 async 循环中迭代。
            response = dashscope.MultiModalConversation.call(
                model=self.model_name,
                text=text,
                voice=self._voice_id,
                stream=True,  # 开启流式
                # language_type="Chinese", # 可选，SDK会自动推断
            )

            is_first_chunk = True

            for chunk in response:
                if chunk.status_code != HTTPStatus.OK:
                    error_msg = f"DashScope Error: {chunk.code} - {chunk.message}"
                    logger.error(error_msg)
                    yield ErrorFrame(error=error_msg)
                    return

                if chunk.output and chunk.output.audio and chunk.output.audio.data:
                    # 获取 base64 音频数据
                    b64_data = chunk.output.audio.data
                    if b64_data:
                        # 解码 base64
                        wav_bytes = base64.b64decode(b64_data)
                        
                        # 处理 WAV 头 (RIFF)
                        # DashScope 流式返回的每一块可能不带头，也可能第一块带头。
                        # 我们这里简单处理：如果是第一块且包含 RIFF 头，去掉 44 字节头。
                        audio_data = wav_bytes
                        if is_first_chunk and wav_bytes.startswith(b'RIFF'):
                             # 简单的 WAV 头剥离 (通常是 44 字节)
                             audio_data = wav_bytes[44:]
                        
                        if len(audio_data) > 0:
                            if is_first_chunk:
                                await self.stop_ttfb_metrics()
                                is_first_chunk = False
                            
                            # 生成音频帧
                            # 1 channel, 16-bit PCM (DashScope默认)
                            yield TTSAudioRawFrame(
                                audio=audio_data, 
                                sample_rate=self.sample_rate, 
                                num_channels=1
                            )
                            await self.start_tts_usage_metrics(text)

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"{self} Exception: {e}")
            yield ErrorFrame(error=f"DashScope TTS execution error: {str(e)}")


async def main():
    # 1. 配置 API Key (请替换为真实 Key 或确保环境变量存在)
    api_key = "xx"
    
    # 2. 初始化服务 (测试不同的音色)
    # voice="cherry" (芊悦), "peter" (天津话), "mariana" (假设的其他音色)
    tts_service = DashScopeTTSService(
        api_key=api_key,
        voice="cherry", 
        model="qwen3-tts-flash",
        sample_rate=24000
    )

    text = "你好，我是 Pipecat。这是一个基于阿里云 DashScope 的实时语音合成测试。今天的风儿甚是喧嚣。"
    output_filename = "/mnt/workspace/yitong.yzy/projects/serve_models/audios/output.wav"

    logger.info(f"开始合成文本: {text}")
    
    # 用于收集所有音频数据
    all_audio_data = bytearray()

    # 3. 运行 TTS 并收集帧
    try:
        async for frame in tts_service.run_tts(text):
            if isinstance(frame, TTSAudioRawFrame):
                all_audio_data.extend(frame.audio)
                print(".", end="", flush=True) # 打印进度点
            elif isinstance(frame, ErrorFrame):
                logger.error(f"\n收到错误帧: {frame.error}")
                return
    except Exception as e:
        logger.error(f"\n发生异常: {e}")
        return

    print("\n合成结束。")

    # 4. 保存为 WAV 文件
    if len(all_audio_data) > 0:
        with wave.open(output_filename, "wb") as wf:
            # 配置 WAV 参数
            # DashScope 默认: 单声道(1), 2字节(16bit), 采样率(24000)
            wf.setnchannels(1)
            wf.setsampwidth(2) 
            wf.setframerate(24000)
            wf.writeframes(all_audio_data)
        
        logger.success(f"音频已保存至: {os.path.abspath(output_filename)}")
        logger.info(f"文件大小: {len(all_audio_data)} 字节")
    else:
        logger.warning("未接收到任何音频数据。")

if __name__ == "__main__":
    import asyncio
    import wave

    asyncio.run(main())

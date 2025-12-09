import os
import base64
import re
import asyncio
import functools
import numpy as np
import dashscope
from typing import AsyncGenerator, Dict, Literal, Optional, List
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


# # 定义支持的音色列表 (根据您提供的表格整理)
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
    """DashScope (Aliyun) TTS Service integration for Pipecat with full parallel chunking."""

    DASHSCOPE_SAMPLE_RATE = 24000
    # 设置安全的分段长度，略小于600以留有余地
    MAX_CHUNK_CHARS = 300 

    def __init__(
            self,
            *,
            api_key: Optional[str] = None,
            voice: str = "cherry",
            model: str = "qwen3-tts-flash",
            sample_rate: Optional[int] = 24000,
            **kwargs,
        ):
            target_sample_rate = sample_rate or self.DASHSCOPE_SAMPLE_RATE
            
            if sample_rate and sample_rate != self.DASHSCOPE_SAMPLE_RATE:
                logger.warning(
                    f"DashScope TTS usually outputs {self.DASHSCOPE_SAMPLE_RATE}Hz. "
                    f"Current config {sample_rate}Hz might cause playback speed issues."
                )
            
            super().__init__(sample_rate=target_sample_rate, **kwargs)

            self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            if not self._api_key:
                raise ValueError("DashScope API Key is required.")

            dashscope.api_key = self._api_key
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

            self.set_model_name(model)
            self.set_voice(voice)

    def set_voice(self, voice: str):
        voice_lower = voice.lower()
        if voice_lower in VALID_VOICES:
            self._voice_id = VALID_VOICES[voice_lower]
        else:
            logger.warning(f"Voice '{voice}' not found in internal map, using raw ID.")
            self._voice_id = voice

    def _split_text(self, text: str) -> List[str]:
        """
        根据标点符号智能拆分文本，确保每段不超过 MAX_CHUNK_CHARS。
        """
        if len(text) < self.MAX_CHUNK_CHARS:
            return [text]

        chunks = []
        curr_chunk = ""
        # 使用正则拆分句子，保留分隔符
        sentences = re.split(r'([。.!！?？\n]+)', text)
        
        for item in sentences:
            if len(curr_chunk) + len(item) > self.MAX_CHUNK_CHARS:
                if curr_chunk:
                    chunks.append(curr_chunk)
                curr_chunk = item
            else:
                curr_chunk += item
        
        if curr_chunk:
            chunks.append(curr_chunk)
            
        return [c for c in chunks if c.strip()]

    def _fetch_audio_sync(self, text: str) -> bytes:
        """
        同步辅助函数：获取单个片段的完整音频数据（已去除 WAV 头）。
        """
        try:
            # 即使我们要等待所有结果，开启 stream=True 依然能更快地让 API 响应首包
            # 只不过我们在本地会把它积攒起来再返回
            response = dashscope.MultiModalConversation.call(
                model=self.model_name,
                text=text,
                voice=self._voice_id,
                stream=True 
            )
            
            full_audio = bytearray()
            is_first_chunk = True

            for chunk in response:
                if chunk.status_code != HTTPStatus.OK:
                    logger.error(f"DashScope Error in chunk fetch: {chunk.code} - {chunk.message}")
                    continue

                if chunk.output and chunk.output.audio and chunk.output.audio.data:
                    wav_bytes = base64.b64decode(chunk.output.audio.data)
                    
                    data_to_add = wav_bytes
                    # 剥离 WAV 44字节头
                    if is_first_chunk and wav_bytes.startswith(b'RIFF'):
                        data_to_add = wav_bytes[44:]
                        is_first_chunk = False
                    
                    full_audio.extend(data_to_add)
            
            return bytes(full_audio)
        except Exception as e:
            logger.error(f"Error fetching audio chunk: {e}")
            return b""

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS for text length: {len(text)}")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # 1. 拆分文本
            text_chunks = self._split_text(text)
            if not text_chunks:
                yield TTSStoppedFrame()
                return

            loop = asyncio.get_running_loop()
            tasks = []

            # 2. 为【所有】分段创建并行任务
            # 注意：这里不再单独处理第一段，所有段都扔进线程池并行跑
            for chunk_text in text_chunks:
                task = loop.run_in_executor(
                    None, 
                    functools.partial(self._fetch_audio_sync, chunk_text)
                )
                tasks.append(task)

            # 3. 等待【所有】任务完成 (Parallel Wait)
            # gather 会按照传入 task 的顺序返回结果列表，保证音频顺序正确
            results = await asyncio.gather(*tasks)

            # 4. 停止计时 (此时才算 TTS 准备完毕)
            await self.stop_ttfb_metrics()

            # 5. 合并并输出音频
            # 我们可以把所有音频拼接成一个巨大的 bytes，然后切片输出，
            # 这样可以避免发送过大的单帧导致传输拥堵。
            combined_audio = b"".join(results)
            
            if len(combined_audio) > 0:
                # # 定义切片大小，例如 8192 字节 (约 170ms) 或 16384 (约 340ms)
                # chunk_size = 16384 
                # for offset in range(0, len(combined_audio), chunk_size):
                #     sub_chunk = combined_audio[offset:offset+chunk_size]
                yield TTSAudioRawFrame(
                    audio=combined_audio,
                    sample_rate=self.sample_rate,
                    num_channels=1
                )
                
                # 记录 Tokens 使用情况 (简单累加文本长度)
                await self.start_tts_usage_metrics(text)

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"{self} Exception: {e}")
            yield ErrorFrame(error=f"DashScope TTS execution error: {str(e)}")

async def main():
    # 1. 配置 API Key (请替换为真实 Key 或确保环境变量存在)
    api_key = "sk-xx"
    
    # 2. 初始化服务 (测试不同的音色)
    # voice="cherry" (芊悦), "peter" (天津话), "mariana" (假设的其他音色)
    tts_service = DashScopeTTSService(
        api_key=api_key,
        voice="cherry", 
        model="qwen3-tts-flash",
        sample_rate=24000
    )

    text = """永和九年，岁在癸丑，暮春之初，会于会稽山阴之兰亭，修禊事也。群贤毕至，少长咸集。此地有崇山峻岭，茂林修竹；又有清流激湍，映带左右，引以为流觞曲水，列坐其次。虽无丝竹管弦之盛，一觞一咏，亦足以畅叙幽情。"""
    output_filename = "xx/output.wav"

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

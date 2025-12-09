import base64
import asyncio
import os
from typing import Optional
from http import HTTPStatus

import dashscope


from loguru import logger
from pipecat.services.whisper.base_stt import BaseWhisperSTTService
from pipecat.transcriptions.language import Language

class DashScopeTranscription:
    """即兴创建一个简单的类来模拟 OpenAI Transcription 对象，
    满足 run_stt 中 response.text 的调用需求。
    """
    def __init__(self, text: str):
        self.text = text

class DashScopeSTTService(BaseWhisperSTTService):
    """DashScope (Aliyun) Speech-to-Text service implementation.
    
    Uses Qwen-ASR (e.g., qwen3-asr-flash) via DashScope SDK.
    """

    def __init__(
        self,
        *,
        model: str = "qwen3-asr-flash",
        api_key: Optional[str] = None,
        language: Optional[Language] = Language.ZH, # 默认设为中文，根据需求调整
        prompt: Optional[str] = None,
        **kwargs,
    ):
        """Initialize DashScope STT service.

        Args:
            model: Model name (e.g., "qwen3-asr-flash").
            api_key: DashScope API Key. If None, looks for DASHSCOPE_API_KEY env var.
            language: Language of the audio input.
            prompt: Context/Prompt for the model (mapped to system message if needed).
        """
        if dashscope is None:
            raise ImportError("DashScope SDK is not installed. Please run `pip install dashscope`.")

        # 调用父类初始化
        # 注意：BaseWhisperSTTService 会尝试创建一个 OpenAI 客户端，但我们在本类中不会使用它。
        # 我们传入 None 作为 api_key 给父类，避免父类初始化 AsyncOpenAI 时报错（如果父类强制校验的话），
        # 或者我们可以重写 _create_client。这里为了简单，我们让父类初始化完成基础设置。
        super().__init__(
            model=model,
            api_key=api_key or "placeholder", # 传个占位符防止父类报错，实际不使用 self._client
            language=language,
            prompt=prompt,
            **kwargs,
        )

        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DashScope API Key is required.")
        
        # 设置全局 API Key (或者在调用时传入)
        dashscope.api_key = self._api_key
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    def _create_client(self, api_key: Optional[str], base_url: Optional[str]):
        """Override to prevent creating an OpenAI client."""
        return None

    async def _transcribe(self, audio: bytes) -> DashScopeTranscription:
        """
        Transcribe audio using DashScope MultiModalConversation API.
        """
        
        # 1. 准备音频数据 (Base64 Data URI)
        # Pipecat 默认传递的 audio bytes 通常是 WAV 格式 (由 Pipeline 中的 VAD/Serializer 决定)
        audio_mime_type = "audio/wav" 
        base64_str = base64.b64encode(audio).decode("utf-8")
        data_uri = f"data:{audio_mime_type};base64,{base64_str}"

        # 2. 构造 Messages
        # 如果有 prompt，可以作为 system prompt 传入 (Qwen-ASR 支持部分定制化)
        messages = []
        if self._prompt:
             messages.append({"role": "system", "content": [{"text": self._prompt}]})
        
        messages.append({
            "role": "user", 
            "content": [{"audio": data_uri}]
        })

        # 3. 准备参数
        # 映射 Pipecat Language 到 DashScope 支持的简写 (如 zh, en)
        # 这里做一个简单的映射示例，实际使用可能需要更完整的映射表
        lang_code = self._language # BaseWhisperSTTService 已经把 enum 转成了 string code (如 'zh', 'en')
        
        asr_options = {
            "enable_itn": False # 是否开启逆文本标准化（如将“一二三”转为“123”）
        }
        
        if lang_code:
            asr_options["language"] = lang_code

        # 4. 调用 DashScope API
        # 由于 dashscope.call 是同步阻塞的，我们需要在线程中运行以避免阻塞 asyncio 事件循环
        try:
            response = await asyncio.to_thread(
                dashscope.MultiModalConversation.call,
                model=self.model_name,
                messages=messages,
                result_format="message",
                api_key=self._api_key,
                asr_options=asr_options
            )
        except Exception as e:
            logger.error(f"DashScope API call failed: {e}")
            raise e

        # 5. 解析结果
        if response.status_code == HTTPStatus.OK:
            # 提取文本
            # 结构通常是 output.choices[0].message.content[0]['text']
            try:
                content_list = response.output.choices[0].message.content
                transcribed_text = ""
                for item in content_list:
                    if "text" in item:
                        transcribed_text += item["text"]
                
                return DashScopeTranscription(text=transcribed_text)
            except (AttributeError, IndexError, KeyError) as e:
                logger.error(f"Failed to parse DashScope response: {response}")
                return DashScopeTranscription(text="")
        else:
            logger.error(f"DashScope API Error: {response.code} - {response.message}")
            raise Exception(f"DashScope Error: {response.message}")


if __name__ == "__main__":
    # 创建 DashScope STT 服务实例
    # stt_service = DashScopeSTTService(model="qwen3-asr-flash")
    stt = DashScopeSTTService(
        api_key="skxxx",
        model="qwen3-asr-flash",
        language=Language.ZH,
        prompt="" # 可选
    )

    # 创建一个音频文件路径
    audio_file_path = "voice1.wav"

    # 读取音频文件
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()

    # 调用 STT 服务进行识别
    transcription = asyncio.run(stt._transcribe(audio_data))
    print(transcription.text)  # 访问 transcription 对象的 text 属性

#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FuseChat Bot Implementation.

This module implements a chatbot using a flexible architecture that can support
multiple LLMs and TTS services. It includes:
- Real-time audio/video interaction
- Animated robot avatar
- Push-to-talk functionality
- Progressive sentence aggregation for TTS
"""

import os
import random
import re
from typing import AsyncIterator

from dotenv import load_dotenv
from loguru import logger
from PIL import Image

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMRunFrame,
    OutputImageRawFrame,
    SpriteFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIClientMessageFrame,
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.ollama.llm import OLLamaLLMService
# from pipecat.services.qwen.llm import QwenLLMService
from pipecat.services.qwen.stt import DashScopeSTTService
from pipecat.services.qwen.tts_realtime import DashScopeTTSRealTimeService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator, Aggregation, AggregationType
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
# from pipecat.transports.daily.transport import DailyParams

load_dotenv(override=True)

sprites = []
script_dir = os.path.dirname(__file__)

# Load sequential animation frames
for i in range(1, 26):
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states."""
    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


class PushToTalkGate(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._gate_opened = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, RTVIClientMessageFrame):
            self._handle_rtvi_frame(frame)
            await self.push_frame(frame, direction)

        if not self._gate_opened and isinstance(
            frame,
            (
                InputAudioRawFrame,
                UserStartedSpeakingFrame,
                InterruptionFrame,
            ),
        ):
            logger.trace(f"{frame.__class__.__name__} suppressed - Button not pressed")
        else:
            await self.push_frame(frame, direction)

    def _handle_rtvi_frame(self, frame: RTVIClientMessageFrame):
        if frame.type == "push_to_talk" and frame.data:
            data = frame.data
            if data.get("state") == "start":
                self._gate_opened = True
                logger.info("Input gate opened - user started talking")
            elif data.get("state") == "stop":
                self._gate_opened = False
                logger.info("Input gate closed - user stopped talking")


class HistoryResetter(FrameProcessor):
    def __init__(self, context: LLMContext, system_prompt: dict):
        super().__init__()
        self._context = context
        self._system_prompt = system_prompt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip().lower()
            if len(text) < 15 and any(flag in text for flag in ["clear history","清除历史","清空历史","清除对话","清空对话"]):
            # if text in ["clear history", "clear history.", "清除历史对话。", "清除历史对话", "清空历史对话。""清空历史对话。", "清空历史对话", "清除对话。", "清除对话", "清空对话。", "清空对话"]:
                logger.info("Resetting conversation history.")
                self._context.set_messages([self._system_prompt])
                # await self.push_frame(TranscriptionFrame(text="重新介绍自己。"))
                # Consume the frame and don't pass it down.
                # return

        await self.push_frame(frame, direction)


class ProgressiveSentenceAggregator(BaseTextAggregator):
    def __init__(self, min_sentences=1, max_sentences=5):
        self._text = ""
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.yield_count = 0
        self.sentence_endings = re.compile(r'(?<!\b\d)(?:[.:：；．｡!?。！？…]+)(?!\s*[)\d])\s*')
        self.limit_list = [1, 1, 2, 3, 5, 10, 20, 40, 100, 100, 100]

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._text.strip(), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        self._text += text
        sentences_found = 0
        last_pos = 0
        for match in self.sentence_endings.finditer(self._text):
            sentences_found += 1
            last_pos = match.end()
            if sentences_found >= self.limit_list[self.yield_count]:
                sentence_block = self._text[:last_pos]
                self._text = self._text[last_pos:]
                self.yield_count += 1
                yield Aggregation(text=sentence_block.strip(), type=AggregationType.SENTENCE)
                sentences_found = 0
                break

    async def flush(self):
        if self._text:
            result = self._text
            self._text = ""
            self.yield_count = 0
            return Aggregation(text=result.strip(), type=AggregationType.SENTENCE)
        self.yield_count = 0
        return None

    async def handle_interruption(self):
        self._text = ""
        self.yield_count = 0

    async def reset(self):
        self.yield_count = 0


VALID_VOICES = {
    # 常用推荐
    "cherry": "Cherry",  # 芊悦 - 阳光积极、亲切自然小姐姐
    # "serena": "Serena",  # 苏瑶 - 温柔小姐姐
    # "ethan": "Ethan",  # 晨煦 - 标准普通话，阳光暖男

    # 特色音色
    # "chelsie": "Chelsie",     # 千雪 - 二次元虚拟女友
    # "momo": "Momo",           # 茉兔 - 撒娇搞怪
    # "vivian": "Vivian",       # 十三 - 拽拽的小暴躁
    # "moon": "Moon",           # 月白 - 率性帅气
    # "maia": "Maia",           # 四月 - 知性与温柔
    # "kai": "Kai",             # 凯 - 耳朵SPA
    # "nofish": "Nofish",       # 不吃鱼 - 不会翘舌音的设计师
    # "bella": "Bella",         # 萌宝 - 喝酒不打醉拳的小萝莉
    # "jennifer": "Jennifer",   # 詹妮弗 - 电影质感美语
    # "ryan": "Ryan",  # 甜茶 - 节奏拉满
    # "katerina": "Katerina",  # 卡捷琳娜 - 御姐音色
    # "aiden": "Aiden",         # 艾登 - 美语大男孩
    # "eldric": "Eldric Sage",  # 沧明子 - 沉稳睿智老者
    # "mia": "Mia",             # 乖小妹 - 温顺如春水
    # "mochi": "Mochi",         # 沙小弥 - 聪明伶俐小大人
    # "bellona": "Bellona",     # 燕铮莺 - 声音洪亮，江湖气
    # "vincent": "Vincent",     # 田叔 - 沙哑烟嗓
    # "bunny": "Bunny",         # 萌小姬 - 萌属性爆棚
    # "neil": "Neil",  # 阿闻 - 专业新闻主持
    # "elias": "Elias",         # 墨讲师 - 严谨叙事
    # "arthur": "Arthur",       # 徐大爷 - 质朴旱烟嗓
    # "nini": "Nini",           # 邻家妹妹 - 软糯甜美
    # "ebona": "Ebona",         # 诡婆婆 - 恐怖童年阴影
    # "seren": "Seren",         # 小婉 - 助眠声线
    # "pip": "Pip",             # 顽屁小孩 - 调皮捣蛋
    # "stella": "Stella",       # 少女阿月 - 迷糊少女/正义战士

    # # 方言与外语特色
    # "bodega": "Bodega",       # 博德加 - 西班牙大叔
    # "sonrisa": "Sonrisa",     # 索尼莎 - 拉美大姐
    # "alek": "Alek",           # 阿列克 - 战斗民族
    # "dolce": "Dolce",         # 多尔切 - 意大利大叔
    # "sohee": "Sohee",         # 素熙 - 韩国欧尼
    # "ono": "Ono Anna",        # 小野杏 - 鬼灵精怪
    # "lenn": "Lenn",           # 莱恩 - 德国青年
    # "emilien": "Emilien",     # 埃米尔安 - 法国大哥哥
    # "andre": "Andre",         # 安德雷 - 沉稳男生
    # "radio": "Radio Gol",     # 足球诗人 - 解说风

    # # 中国方言
    # "jada": "Jada",           # 上海-阿珍
    # "dylan": "Dylan",         # 北京-晓东
    # "li": "Li",               # 南京-老李
    # "marcus": "Marcus",       # 陕西-秦川
    # "roy": "Roy",             # 闽南-阿杰 (台普)
    # "peter": "Peter",         # 天津-李彼得 (相声风)
    # "sunny": "Sunny",         # 四川-晴儿
    # "eric": "Eric",           # 四川-程川
    # "rocky": "Rocky",         # 粤语-阿强
    # "kiki": "Kiki",           # 粤语-阿清
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY")

    params = BaseOpenAILLMService.InputParams(
        seed=42,
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        max_completion_tokens=768,
    )

    stt = DashScopeSTTService(
        api_key=api_key,
        model="qwen3-asr-flash-2025-09-08",
        prompt=""
    )

    tts_realtime = DashScopeTTSRealTimeService(
        api_key=api_key,
        voice=random.choice(list(VALID_VOICES.keys())),
        model="qwen3-tts-flash-realtime-2025-11-27",
        sample_rate=24000,
        text_aggregator=ProgressiveSentenceAggregator(),
        text_filters=[MarkdownTextFilter()],
    )

    llm = OLLamaLLMService(model='FuseChat-3.0-7B', params=params)

    messages = [
        {
            "role": "system",
            "content": "你是FuseChat，由深圳河套学院（Shenzhen Loop Area Institute）创造的人工智能对话助手。请以自然、简短、口语化的风格进行交流，就像日常和朋友聊天一样。在你的回答中，请不要使用任何Markdown格式（比如粗体或列表）或表情符号（emoji）。",
        },
    ]
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are FuseChat, created by Shenzhen Loop Area Institute (深圳河套学院). Engage in natural, brief conversations. Avoid Markdown or emojis in your responses.",
    #     },
    # ]

    context = LLMContext(messages)
    system_prompt = messages[0]
    history_resetter = HistoryResetter(context, system_prompt)
    context_aggregator = LLMContextAggregatorPair(context)

    # push_to_talk_gate = PushToTalkGate()
    ta = TalkingAnimation()
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            # push_to_talk_gate,
            stt,
            history_resetter,
            context_aggregator.user(),
            llm,
            ParallelPipeline(
                [tts_realtime, ta, transport.output()],
                [context_aggregator.assistant()]
            )
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    await task.queue_frame(quiet_frame)

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport_params = {
        # "daily": lambda: DailyParams(
        #     audio_in_enabled=True,
        #     audio_out_enabled=True,
        #     video_in_enabled=False,  # Disable video input
        #     video_out_enabled=True,
        #     video_out_width=1024,
        #     video_out_height=576,
        #     audio_out_sample_rate=24000,
        #     vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        #     turn_analyzer=LocalSmartTurnAnalyzerV3(),
        # ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,
            video_out_enabled=True,
            video_out_width=1024,
            video_out_height=576,
            audio_out_sample_rate=24000,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()

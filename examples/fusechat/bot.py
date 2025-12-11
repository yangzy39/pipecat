#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os
import certifi  

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.qwen.tts import DashScopeTTSService
from pipecat.services.qwen.stt import DashScopeSTTService
from pipecat.services.qwen.tts_realtime import DashScopeTTSRealTimeService
from pipecat.services.qwen.llm import QwenLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
# from pipecat.services.cartesia.tts import CartesiaTTSService
# from pipecat.services.deepgram.stt import DeepgramSTTService
# from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

import random
# random.seed(42)

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

VALID_VOICES = {
    # 常用推荐
    "cherry": "Cherry",       # 芊悦 - 阳光积极、亲切自然小姐姐
    "serena": "Serena",       # 苏瑶 - 温柔小姐姐
    "ethan": "Ethan",         # 晨煦 - 标准普通话，阳光暖男
    
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
    # "ryan": "Ryan",           # 甜茶 - 节奏拉满
    # "katerina": "Katerina",   # 卡捷琳娜 - 御姐音色
    # "aiden": "Aiden",         # 艾登 - 美语大男孩
    # "eldric": "Eldric Sage",  # 沧明子 - 沉稳睿智老者
    # "mia": "Mia",             # 乖小妹 - 温顺如春水
    # "mochi": "Mochi",         # 沙小弥 - 聪明伶俐小大人
    # "bellona": "Bellona",     # 燕铮莺 - 声音洪亮，江湖气
    # "vincent": "Vincent",     # 田叔 - 沙哑烟嗓
    # "bunny": "Bunny",         # 萌小姬 - 萌属性爆棚
    # "neil": "Neil",           # 阿闻 - 专业新闻主持
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

async def run_2bots(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    api_key = "sk-"

    params=BaseOpenAILLMService.InputParams(
        # frequency_penalty=0.0,
        # presence_penalty=0.0,
        seed=42,
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        max_completion_tokens=1024,
        # service_tier="standard",
        extra={}
    )

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    stt = DashScopeSTTService(
        api_key=api_key,
        model="qwen3-asr-flash-2025-09-08",
        # language=Language.ZH,
        prompt="" # 可选
    )

    tts = DashScopeTTSService(
        api_key=api_key,
        voice="Serena",  # British Reading Lady
        model="qwen3-tts-flash-2025-11-27",
        sample_rate=24000
    )


# 
    llm = QwenLLMService(api_key=api_key,model='deepseek-v3',params=params)

    messages = [
        {
            "role": "system",
            "content": "You are FuseChat-3.0, created by Sun Yat-sen University Language Intelligence Technology (SLIT) AI Lab.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    context2 = LLMContext(messages)
    context_aggregator2 = LLMContextAggregatorPair(context2)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))


    tts2 = DashScopeTTSService(
        api_key=api_key,
        voice="Ethan",  # British Reading Lady
        model="qwen3-tts-flash-2025-11-27",
        sample_rate=24000
    )
    
    llm2 = QwenLLMService(api_key=api_key,model='qwen-max',params=params)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            ParallelPipeline( 
                # Agent 1:
                [
                    context_aggregator.user(),  # User responses
                    llm,  # LLM
                    tts,  # TTS
                    transport.output(),  # Transport bot output
                ],
                # Agent 2: 
                [
                    context_aggregator2.user(),  # User responses
                    llm2,  # LLM
                    tts2,  # TTS
                    transport.output(),  # Transport bot output
                ]
            ),
            context_aggregator.assistant(),  # Assistant spoken responses
            context_aggregator2.assistant(),  # Assistant spoken responses
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        pass
        # Kick off the conversation.
        # messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        # await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    api_key = "sk-"

    # api_key = "sk-"

    params=BaseOpenAILLMService.InputParams(
        # frequency_penalty=0.0,
        # presence_penalty=0.0,
        seed=42,
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        max_completion_tokens=768,
        # service_tier="standard",
        extra={}
    )

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    stt = DashScopeSTTService(
        api_key=api_key,
        model="qwen3-asr-flash-2025-09-08",
        # language=Language.ZH,
        prompt="" # 可选
    )

    # tts = DashScopeTTSService(
    #     api_key=api_key,
    #     voice="Serena",  # British Reading Lady
    #     model="qwen3-tts-flash-2025-11-27",
    #     sample_rate=24000
    # )

    tts = DashScopeTTSService(
        api_key=api_key,
        voice=random.choice(list(VALID_VOICES.keys())),  # British Reading Lady
        model="qwen3-tts-flash-2025-11-27",
        sample_rate=24000
    )

    tts_realtime = DashScopeTTSRealTimeService(
        api_key=api_key,
        voice=random.choice(list(VALID_VOICES.keys())),  # British Reading Lady
        model="qwen3-tts-flash-realtime-2025-11-27",
        sample_rate=24000
    )


    llm = QwenLLMService(api_key=api_key,model='qwen2.5-7b-instruct',params=params)

    # VERIFIER_HOST="22.8.148.33"
    # VERIFIER_PORT="23547"
    # API_BASE="http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"

    
    # VERIFIER_HOST="22.8.158.84"
    # VERIFIER_PORT="23547"
    # API_BASE=f"http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"

    # llm = BaseOpenAILLMService(base_url="http://localhost:11434/v1/", api_key="ollama",model='FuseChat-3.0-1B',params=params)
    # llm = BaseOpenAILLMService(api_key=API_BASE,model='FuseChat-3.0',params=params)
    # llm = BaseOpenAILLMService(base_url=API_BASE, api_key="EMPTY",model='gpt-oss-120b',params=params)

    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are FuseChat-3.0, created by Sun Yat-sen University Language Intelligence Technology (SLIT) AI Lab.",
    #     },
    # ]

    messages = [
        {
            "role": "system",
            "content": "You are FuseChat-3.0, created by Shenzhen Loop Area Institute (深圳河套学院). Your response should not contain Markdown syntax or emojis.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))


    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            # ParallelPipeline(
            #     [tts, transport.output()],
            #     [context_aggregator.assistant()]
            # )
            tts_realtime,  # TTS
            transport.output(),  
            context_aggregator.assistant()
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,  # 关键修改：禁用打断功能
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        pass
        # Kick off the conversation.
        # messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        # await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,      # Disable video input
            video_out_enabled=False, 
            audio_out_sample_rate=24000,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,      # Disable video input
            video_out_enabled=False, 
            audio_out_sample_rate=24000,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":

    from pipecat.runner.run import main

    main()
    logger.info(f"Starting bot")

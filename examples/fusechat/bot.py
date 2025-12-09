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
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.qwen.tts import DashScopeTTSService
from pipecat.services.qwen.stt import DashScopeSTTService
from pipecat.services.qwen.llm import QwenLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
# from pipecat.services.cartesia.tts import CartesiaTTSService
# from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    api_key = "sk-xxx"

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
        model="qwen3-asr-flash",
        # language=Language.ZH,
        prompt="" # 可选
    )

    tts = DashScopeTTSService(
        api_key=api_key,
        voice="Cherry",  # British Reading Lady
        model="qwen3-tts-flash",
        sample_rate=24000
    )
# 
    # llm = QwenLLMService(api_key=api_key,model='deepseek-v3',params=params)

    VERIFIER_HOST="xxx"
    VERIFIER_PORT="xxx"
    API_BASE="http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"

    llm = BaseOpenAILLMService(base_url=API_BASE, api_key="EMPTY",model='gpt-oss-120b',params=params)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant, created by Sun Yat-sen University Language Intelligence Technology (SLIT) AI Lab.",
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
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":

    from pipecat.runner.run import main

    main()
#     logger.info(f"Starting bot")

#     api_key = "sk-xxx"

#     params=BaseOpenAILLMService.InputParams(
#         # frequency_penalty=0.0,
#         # presence_penalty=0.0,
#         seed=42,
#         temperature=0.7,
#         top_p=0.95,
#         max_tokens=1024,
#         max_completion_tokens=1024,
#         # service_tier="standard",
#         extra={}
#     )

#     # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

#     stt = DashScopeSTTService(
#         api_key=api_key,
#         model="qwen3-asr-flash",
#         # language=Language.ZH,
#         prompt="" # 可选
#     )

#     tts = DashScopeTTSService(
#         api_key=api_key,
#         voice="Cherry",  # British Reading Lady
#         model="qwen3-tts-flash",
#         sample_rate=24000
#     )
# # 
#     # llm = QwenLLMService(api_key=api_key,model='deepseek-v3',params=params)

#     VERIFIER_HOST="22.7.253.17"
#     VERIFIER_PORT="23547"
#     API_BASE="http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"

#     llm = BaseOpenAILLMService(api_key=API_BASE,model='gpt-oss-120b',params=params)

#     print(stt)
#     print(tts)
#     print(llm)


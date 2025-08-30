---
id: MjAyNS0w
title: not much happened today
date: '2025-08-29T05:44:39.731046Z'
description: >-
  **Apple** released three real-time vision-language models (**FastVLM**,
  **MobileCLIP2**) on Hugging Face with significant speed and size improvements,
  supporting WebGPU and Core ML. Their MLX framework now supports **MXFP4**
  format, competing with **NVFP4** for FP4 quantization. **xAI** launched
  **grok-code-fast-1**, outperforming Claude for code edits, while **OpenAI**
  integrated **GPT-5** into Xcode 26 and released a new **Responses API** on
  **Groq** hardware. CLI-first agent workflows advanced with tools like
  **SemTools**, **MLX** local runner for Apple Silicon, and **llama.vim**
  recommending **Qwen 3 Coder 30B A3B**. Retrieval research highlights
  limitations of single-vector embeddings, promoting ColBERT-style late
  interaction.
companies:
  - apple
  - hugging-face
  - x-ai
  - openai
  - groq
  - run-llama
  - lmstudio
models:
  - fastvlm
  - mobileclip2
  - grok-code-fast-1
  - gpt-5
  - qwen-3-coder-30b-a3b
topics:
  - vision
  - model-quantization
  - code-generation
  - cli-workflows
  - retrieval-augmentation
  - embedding-models
  - local-ai
  - multimodality
people:
  - reach_vb
  - xenovacom
  - pcuenq
  - awnihannun
  - cline
  - veggie_eric
  - nickbaumann_
  - gdb
  - benankdev
  - loganmarkewich
  - tom_doerr
  - fastmcp
  - ggerganov
  - orionweller
  - antoine_chaffin
---


**a quiet day**

> AI News for 8/28/2025-8/29/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (185 channels, and 7366 messages) for you. Estimated reading time saved (at 200wpm): 574 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

![](https://resend-attachments.s3.amazonaws.com/FiHZRAH5cXLzvof)

This is not yet publicly announced, but if you are interested in Enterprise AI or Coding Agents, AI News readers can [apply](https://apply.ai.engineer/) to attend the first **AI Engineer Code Summit**, returning to **NYC Nov 20-22** and focused on how coding agents and LLMs are changing (or failing to change) software development at all scales. Speaker and sponsor [applications also open](https://apply.ai.engineer/).

---

# AI Twitter Recap

**Apple’s on-device VLM push (FastVLM, MobileCLIP2) and MLX upgrades**

- **FastVLM + MobileCLIP2 released on Hugging Face**: Apple shipped three real-time VLMs (0.5B, 1.5B, 7B) with WebGPU/transformers.js demos and MLX/Core ML support. Apple claims up to **85x faster** and **3.4x smaller** than prior work, with **7.9x faster TTFT** for larger models via fewer vision tokens and a lean encoder. Live video captioning runs 100% locally in-browser. See overviews and demos by [@reach_vb](https://twitter.com/reach_vb/status/1961471154197053769) ([demo](https://twitter.com/reach_vb/status/1961471503267979699)), [@xenovacom](https://twitter.com/xenovacom/status/1961454543503344036), and [@pcuenq](https://twitter.com/pcuenq/status/1961464859465269757). Apple is also “open sourcing artefacts on HF,” per [@reach_vb](https://twitter.com/reach_vb/status/1961481909181075961).
- **MLX + MXFP4 across the stack**: Apple MLX added support for MXFP4 used by GPT-OSS; upgrade with [pip install -U mlx](https://twitter.com/awnihannun/status/1961484829037330612). LM Studio confirmed **MXFP4 support for openai/gpt-oss** in MLX ([tweet](https://twitter.com/lmstudio/status/1961508941852283016)). Expect active FP4 format churn: Awni Hannun compares **MXFP4 vs NVFP4**, noting MXFP4’s scale encoding is “suboptimal” and heavily concentrated; NVFP4 (e4m3 scale, group size 16) may win out ([analysis](https://twitter.com/awnihannun/status/1961500133990043967)).

**Agentic coding stacks: Grok Code Fast, Codex/Xcode 26, and CLI-native workflows**

- **xAI’s grok-code-fast-1 + Cline loop**: Cline users report grok-code-fast-1 feels “10x better and faster than Claude” for diff edits and complex refactors; early data shows **~87 TPS** and parity with Sonnet-4 on diff-edit failures after three days of iteration. xAI is uniquely shipping frequent checkpoints learned from Cline’s heavyweight traces (massive contexts, tool use). Read the roundup from [@cline](https://twitter.com/cline/status/1961488289803939915), vendor quotes via [@veggie_eric](https://twitter.com/veggie_eric/status/1961474457295622515), and strategy take by [@nickbaumann_](https://twitter.com/nickbaumann_/status/1961539461860487664). Prompting guide: [docs.x.ai](http://docs.x.ai/).
- **OpenAI Codex and GPT-5 in Xcode**: OpenAI rolled out a VS Code Codex plugin; [@gdb](https://twitter.com/gdb/status/1961349040056000719) says it’s “already very good.” They also announced **GPT-5 built into Xcode 26**; get higher limits by signing in with ChatGPT ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853), [follow-up](https://twitter.com/OpenAIDevs/status/1961557516753752461)). For agents, OpenAI’s new **Responses API** (structured, multimodal, remote MCP-oriented) is live on **Groq** ([@benankdev](https://twitter.com/benankdev/status/1961444239327240500)).
- **CLI-first agent workflows**:
    - Semantic search for the shell without a vector DB via **SemTools** (`parse`, `search`, 400x faster static embeddings) from run-llama ([@LoganMarkewich](https://twitter.com/LoganMarkewich/status/1961448960184520945), [explanation](https://twitter.com/jerryjliu0/status/1961488443663597857)).
    - **MLX** “ollama-style” local runner for Apple Silicon ([@tom_doerr](https://twitter.com/tom_doerr/status/1961309536406392877)).
    - **FastMCP** one-push MCP server + chat client ([@fastmcp](https://twitter.com/fastmcp/status/1961436552057278512)).
    - For local coding, **llama.vim** now recommends **Qwen 3 Coder 30B A3B** on Macs (beats Qwen 2.5 Coder 7B) via llama.cpp ([@ggerganov](https://twitter.com/ggerganov/status/1961471397428883882)).

**Retrieval, indexing, and memory: beyond single-vector embeddings**

- **Single-vector embeddings hit a wall**: Theory and empirics say a single vector can’t “do it all” for modern retrieval tasks. ColBERT-style late interaction avoids fundamental tradeoffs; see the argument by [@orionweller](https://twitter.com/orionweller/status/1961436569409331579), and supporting notes by [@antoine_chaffin](https://twitter.com/antoine_chaffin/status/1961339798112575673) with an OSS late-interaction stack ([pylate](https://twitter.com/antoine_chaffin/status/1961340768544510392)).
- **Vectorless and hybrid indexing**: Early “vectorless RAG” using tree indices (PageIndex) shows promising routing/search behavior with reasoning models, per [@omarsar0](https://twitter.com/omarsar0/status/1961446862012960840) ([repo](https://twitter.com/omarsar0/status/1961446976152588712)). Weaviate details **8-bit rotational quantization** (4x compression, faster vector search with quality gains) via random rotations + scalar quantization ([blog](https://twitter.com/dl_weekly/status/1961413948877553899)).
- **KV-memory reducers**: UC Berkeley’s **XQuant/XQuant-CL** rematerialize K/V from quantized activations, achieving **2× to 12.5× memory cuts** with minimal accuracy loss; handles GQA via SVD ([thread](https://twitter.com/TheTuringPost/status/1961475078753063322), [paper](https://twitter.com/TheTuringPost/status/1961475160823009773)). Paired with the FP4 ecosystem shifts above, inference memory and bandwidth are moving targets.

**Agent and reasoning evals: multi-hour horizons, tool-use, and environments**

- **Time-horizon gains**: METR estimates **Claude Opus 4.1** achieves a 50%-success time-horizon of ~1h45m on multi-step SWE tasks, ~30% longer than Opus 4 (statistically significant). Detailed report and method in [@METR_Evals](https://twitter.com/METR_Evals/status/1961527692072993272).
- **Multi-agent/tool-use benchmarks**:
    - An updated “Multi-Agent Step Race” shows OpenAI models dominant; **2.5 Flash > 2.5 Pro** on this setup; DeepSeek V3.1-NS sits far above R1-0528, per [summary](https://twitter.com/teortaxesTex/status/1961298849047117832).
    - Several new **MCP-Bench** releases are emerging for tool-using LLMs ([@_akhaliq](https://twitter.com/_akhaliq/status/1961456699564294651)); demand for standardized tool-calling evals is spiking ([commentary](https://twitter.com/bigeagle_xd/status/1961461441799852128)).
    - Stanford/Berkeley’s live **DeepScholar-Bench** targets generative research synthesis with leaderboard, code, and paper links ([@lianapatel_](https://twitter.com/lianapatel_/status/1961487232331911651)).
    - Open infra for agents: **“Environment hub”** announced as part of a broader open AGI stack (compute, sandboxes, RFT, evals) ([thread](https://twitter.com/vincentweisser/status/1961594111733158141)).

**Notable model releases and papers (audio, search, vision, reasoning)**

- **Step-Audio 2 Mini (StepFun)**: An Apache-2.0, open 8B speech-to-speech model claims to beat GPT-4o-Audio on internal evals; trained on **8M+ hours**, supports **50k+ voices**, expressive/grounded speech, tool calling, and multimodal discrete token modeling; built atop Qwen2-Audio + CosyVoice. Demos and details via [@reach_vb](https://twitter.com/reach_vb/status/1961414067668558319) ([model card](https://twitter.com/reach_vb/status/1961414145938485477)).
- **Search models**: The first open model on LM Arena’s Search leaderboard—**Diffbot-small-xl (Apache 2.0)**—debuts at #9 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1961526740754616545)).
- **DeepSeek’s surge**: **DeepSeek V3.1** and its “thinking” variant enter the Text Arena Top 10 at #8 (tied with several frontier models), ranking top-3 on math and longer queries ([announcement](https://twitter.com/lmarena_ai/status/1961474406817173602)).
- **Style/control for T2I**: ByteDance’s **USO** (Unified Style and Subject-Driven Generation via disentangled + reward learning) is open-sourced with demo ([paper share](https://twitter.com/_akhaliq/status/1961455755111842126), [code/demo](https://twitter.com/fenfenfenfenfan/status/1961464402550690007)).
- **Graph-R1 (7B)**: Uses NP-hard graph problems as a synthetic training corpus to elicit long-chain-of-thought reasoning; claims parity with QwQ-32B with better token efficiency ([summary](https://twitter.com/papers_anon/status/1961385914040766712)).
- Also notable: **Pref-GRPO** (pairwise preference reward GRPO for stable T2I RL) ([paper link](https://twitter.com/_akhaliq/status/1961437082888352200)), “**AWorld**” (orchestrating the training recipe for agentic AI) ([post](https://twitter.com/_akhaliq/status/1961456228044873888)), and Apple’s **MobileCLIP2** mentioned alongside FastVLM ([@xenovacom](https://twitter.com/xenovacom/status/1961454543503344036)).

**Policy, platforms, and ecosystem notes**

- **Anthropic data retention change**: Users flagged a new “5-year” retention status. Anthropic clarified: if you opt out of training, retention remains **30 days**; otherwise longer retention applies ([@michael_nielsen](https://twitter.com/michael_nielsen/status/1961439837791367501), [@vikhyatk](https://twitter.com/vikhyatk/status/1961511207577534731), [@sammcallister](https://twitter.com/sammcallister/status/1961520548510400753)). Several devs called for clearer in-product disclosure.
- **Progress framing**: Epoch AI argues GPT-5 is both incremental (post-training/RL heavy) and a major leap over GPT-4, contrasting with GPT-4’s pretrain scale-up ([thread](https://twitter.com/EpochAIResearch/status/1961524635398529209)). In parallel, LM arena, METR, and tool-use benchmarks reflect accelerating improvements in “hours-long” agentic reliability and search/chat quality.
- **Systems**: Modular’s Chris Lattner kicked off a Blackwell GPU blog series to demystify extracting peak perf ([@clattner_llvm](https://twitter.com/clattner_llvm/status/1961491323875455029)); community GPU bootcamps (CUDA + ThunderKittens) continue to ramp ([@jyo_pari](https://twitter.com/jyo_pari/status/1961442690249216491)).

**Top tweets (by engagement)**

- Apple’s FastVLM WebGPU demo and details: [@reach_vb](https://twitter.com/reach_vb/status/1961471154197053769) (1950)
- GPT-5 integrated in Xcode 26 (beta): [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853) (1154)
- Haircut morph workflow (Nano Banana + Kling 2.1 + Claude prompts): [@fabianstelzer](https://twitter.com/fabianstelzer/status/1961441746878939431) (3447)
- Try the OpenAI Codex VS Code plugin: [@gdb](https://twitter.com/gdb/status/1961349040056000719) (963)
- Cline x grok-code-fast-1 early results (diff-edit speed/capability): [@cline](https://twitter.com/cline/status/1961488289803939915) (1253)
- On-device Apple VLM release recap: [@xenovacom](https://twitter.com/xenovacom/status/1961454543503344036) (1412)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Apple FastVLM/MobileCLIP2 WebGPU Demo + Step-Audio 2 Mini Release

- [**Apple releases FastVLM and MobileCLIP2 on Hugging Face, along with a real-time video captioning demo (in-browser + WebGPU)**](https://v.redd.it/ayma955sbzlf1) ([Score: 899, Comments: 107](https://www.reddit.com/r/LocalLLaMA/comments/1n3b13b/apple_releases_fastvlm_and_mobileclip2_on_hugging/)): **Apple published two vision-language assets on Hugging Face—[FastVLM](https://huggingface.co/collections/apple/fastvlm-68ac97b9cd5cacefdd04872e) and [MobileCLIP2](https://huggingface.co/collections/apple/mobileclip2-68ac947dcb035c54bcd20c47)—plus an in-browser, WebGPU-powered [real-time video captioning demo](https://huggingface.co/spaces/apple/fastvlm-webgpu). The release emphasizes on-device/browser execution and latency, showcasing end-to-end VLM inference directly in the client via WebGPU without server round-trips.** Commenters report the demo runs “faster than I can read,” and note this eclipses Apple’s prior OSS efforts (previously a finetune of Qwen 2.5), suggesting Apple has been “slow cooking” more mature in-house VLMs before this drop.
    - Several note that prior to this, Apple’s strongest open-source contribution was reportedly a finetune of **Qwen 2.5** (Alibaba’s model), implying this release marks a shift to Apple publishing its own VLM stack (FastVLM + MobileCLIP2) rather than just finetunes. This is technically significant for evaluating Apple’s in-house vision-language capabilities versus relying on external base models.
    - Multiple users highlight the demo’s real-time, in-browser performance via **WebGPU**, with one remarking it runs "faster than I can read," suggesting efficient on-device GPU inference suitable for streaming captioning. This raises practical interest in integrations like a **Lightroom Classic** plugin for automatic keywords/captions, where prior tools were "absurdly slow"—the WebGPU pipeline hints at sufficient throughput for batch photo metadata generation if similar optimizations are exposed outside the browser.
- [**Step-Audio 2 Mini, an 8 billion parameter (8B) speech-to-speech model**](https://i.redd.it/orq1ackg50mf1.png) ([Score: 165, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1n3fcyf/stepaudio_2_mini_an_8_billion_parameter_8b/)): **StepFun AI released Step-Audio 2 Mini, an** `8B`**parameter, Apache-2.0–licensed speech-to-speech model trained on** `>8M` **hours of real + synthesized audio, claiming to outperform GPT-4o-Audio on expressive and grounded speech benchmarks. The model supports** `>50k` **voices and uses multimodal LLM techniques—including reasoning-centric RL and RAG—for richer audio understanding and natural, real-time speech conversations ([HF card](https://huggingface.co/stepfun-ai/Step-Audio-2-mini?utm_source=perplexity)).** Top comments are mostly non-technical; one user clarifies the expectation that "speech-to-speech" means I speak → AI responds with speech, while another laments the lack of open-source music generation models.
    - Commenters distinguish true speech-to-speech voice conversion from text-mediated cloning. **RVC v2** ([repo](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)) preserves F0/pitch and timing, enabling song covers and timbre transfer, whereas ASR→TTS pipelines often lose pitch/prosody and excel at conversational 'chat' voice cloning instead. They note RVC v2 feels dated and are seeking end-to-end replacements that retain pitch while improving quality/latency.
    - There's concern about the lack of audio samples/demos, making it impossible to evaluate timbre similarity, F0 retention, robustness on singing vs speech, or streaming latency. Without concrete demos or metrics (e.g., MOS, speaker similarity, F0 contour correlation), it's unclear whether this model performs direct VC or speech-to-text-to-speech.
    - Terminology ambiguity: *'speech-to-speech'* is interpreted by some as direct real-time voice conversion (I talk → AI talks back), while others expect RVC-style same-pitch conversion capable of song covers. Clear documentation on pipeline (end-to-end VC vs ASR+TTS), controllable F0, and singing support would resolve expectations for use cases.

### 2. Qwen3-Coder Local Coding Tutorial + Qwen September Teaser

- [**Qwen3-coder is mind blowing on local hardware (tutorial linked)**](https://v.redd.it/75bfhw7sc1mf1) ([Score: 177, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1n3ldon/qwen3coder_is_mind_blowing_on_local_hardware/)): **OP reports that Qwen3-Coder-30B with a** `256k` **context window runs locally and reliably executes Cline tool calls and diff edits using LM Studio + Cline (VS Code), on a 36 GB RAM Mac via a 4-bit quantized build. A key configuration note is to disable KV-cache quantization in LM Studio; with this and the quantized model, OP claims it crosses from “toy” to practical coding, and shares a full setup guide at [cline.bot/blog/local-models](https://cline.bot/blog/local-models).** Commenters report mixed reliability: one running BF16 in VS Code+Cline found it stalled on incorrect Python type hints, misidentified Python 2 vs 3 runtime, and produced trailing-space artifacts it couldn’t correct; another cites `DevStral small 2507` as competitive in planning though slower. Others hit Cline integration failures (e.g., `Unexpected API Response: The language model did not provide any assistant messages.`) and ask which quantizations yield consistent runs.
    - Reports on **Qwen3-Coder 30B (bf16)** in VSCode with Cline note agentic failure modes: it generates Python with incorrect type hints, then gets stuck in a self-repair loop; fails to detect it should run via `python3` and instead attempts Python 2 compatibility changes; and produces trailing spaces on empty lines (a quirk also observed with **Claude**) that it can’t reliably auto-correct. These behaviors make it unreliable for real workflow automation despite improved quality over prior hybrid versions.
    - Multiple users flag **Cline integration instability**: “Unexpected API Response: The language model did not provide any assistant messages,” implying either API transport issues or empty/invalid model outputs. One user notes it completed a first task but failed on the second, asking what **quantizations** others are using for consistency, suggesting sensitivity to model/quant settings and toolchain compatibility.
    - Performance skepticism for local runs: a demo video appears fast‑forwarded; on a `Ryzen 7 5800X3D` with `64 GB RAM`, the 30B model is described as sluggish. An alternative, **DevStral Small 2507**, is cited as performing well in Cline—slower than **Qwen3‑30B** but competitive or slightly better in planning/communication quality.
- [**Amazing Qwen stuff coming soon**](https://i.redd.it/v6kx1bw8sxlf1.png) ([Score: 551, Comments: 83](https://www.reddit.com/r/LocalLLaMA/comments/1n33ugq/amazing_qwen_stuff_coming_soon/)): **Qwen posts a teaser image (bear mascot watering a tree with a kiwi sign) hinting at a September reveal, suggesting a new model or product under a “Kiwi” codename. There are no specs, benchmarks, or capabilities disclosed—this is a marketing teaser rather than a technical announcement.** Commenters speculate it could be a smaller diffusion/image-editing model or an audio-generation model; one draws a parallel to Google’s image-editing model “NanoBanana,” implying Qwen’s “Kiwi.” Others infer the watering can implies training is still ongoing and that improved infra might allow training in weeks.
    - Speculation centers on a compact diffusion model for image generation/editing or a new audio-generation stack. The “kiwi” teaser plus a reference to **Google**’s “NanoBanana” image editor (as mentioned by a commenter) suggests an image-editing pipeline, possibly optimized for lower VRAM and faster sampling (fewer diffusion steps) suitable for on-device or edge deployment.
    - Others hope for a TTS release, implying a multimodal push (ASR+TTS) with low-latency streaming synthesis and controllable prosody as likely differentiators. Integration with an LLM agent would prioritize fast first‑token latency, stable long‑form synthesis, and voice cloning or style transfer capabilities.
    - One comment reads the watering-can imagery as a signal the model is *still training*, speculating **Qwen**’s infra can now support end‑to‑end training cycles in `<= 2 weeks`. That would imply improvements in distributed training reliability (scheduler/fault tolerance), data throughput, and checkpointing to enable faster iteration and recovery.

### 3. Alibaba Nvidia-Alternative AI Chip + Meta Cancels Behemoth Public Release

- [**Alibaba Creates AI Chip to Help China Fill Nvidia Void**](https://www.reddit.com/r/LocalLLaMA/comments/1n35bwe/alibaba_creates_ai_chip_to_help_china_fill_nvidia/) ([Score: 275, Comments: 59](https://www.reddit.com/r/LocalLLaMA/comments/1n35bwe/alibaba_creates_ai_chip_to_help_china_fill_nvidia/)): **WSJ reports that Alibaba is testing a domestically-fabbed AI inference chip intended to fill the Nvidia gap in China, targeting a broader range of inference workloads while maintaining compatibility with the Nvidia ecosystem ([WSJ](https://www.wsj.com/tech/ai/alibaba-ai-chip-nvidia-f5dc96e3)). Due to sanctions, it is no longer manufactured at TSMC and instead uses a Chinese foundry; Alibaba has reportedly not ordered Huawei chips, citing cloud-competition concerns. If successful, this would pair Alibaba’s in-house silicon with its advanced LLM stack (e.g., [Qwen](https://github.com/QwenLM)), signaling deeper vertical integration of compute + models.** Top comments emphasize that Nvidia-compatibility is the pivotal factor for adoption, with potential to be a “game changer”; others note Alibaba’s push toward full-stack control, while skeptics argue that non-Nvidia AI chips struggle primarily on price and software ecosystem, citing vendors like [Cerebras](https://www.cerebras.net/).
    - “Compatible with Nvidia” is interpreted as compatibility at the framework/runtime layer for inference, not a CUDA clone. Commenters note this likely means it can run common abstractions like [PyTorch](https://pytorch.org/), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), and [HuggingFace TGI](https://github.com/huggingface/text-generation-inference). Practically, that implies Alibaba must provide kernel/ops coverage and backend integrations so model graphs execute without code changes, but CUDA-specific kernels would need non-CUDA equivalents for attention, quantization, and memory management paths used in large-model inference.
    - Market realism: there are already multiple AI accelerators, but adoption has lagged primarily due to price/TCO and ecosystem costs, not lack of hardware alone. **Cerebras** is cited as an example ([cerebras.net](http://cerebras.net/)): even with novel architectures, without competitive $/token inference, supply, and software maturity, market share remains small. Any Alibaba chip will need to beat incumbents on cost-per-inference and developer friction to matter at scale.
    - Alibaba’s move suggests deeper vertical integration (cloud + models + serving + silicon) to fill a China-local Nvidia gap, especially for inference workloads. Tighter integration with their model stack (e.g., **Qwen** family) and serving layers could enable hardware–software co-design for latency/throughput targets, reducing dependence on CUDA while keeping user-facing APIs stable. If successful, this could offer drop-in serving for popular LLM stacks while controlling costs and supply internally.
- [**Financial Times reports that Meta won't publicly release Behemoth: "The social media company had also abandoned plans to publicly release its flagship Behemoth large language model, according to people familiar with the matter, focusing instead on building new models."**](https://www.ft.com/content/feccb649-ce95-43d2-b30a-057d64b38cdf) ([Score: 169, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1n30yue/financial_times_reports_that_meta_wont_publicly/)): **Financial Times reports that Meta has “abandoned plans to publicly release” its flagship Behemoth LLM, opting to focus on building new models, and is exploring licensing AI tech from a start‑up to close performance/productization gaps versus rivals ([FT](https://www.ft.com/content/feccb649-ce95-43d2-b30a-057d64b38cdf)). The move is framed as a tactical acceleration—integrating external capabilities rather than relying solely on slower in‑house development—suggesting internal models are not meeting competitive benchmarks at present. No technical specs for Behemoth were disclosed by FT; the report centers on release strategy and procurement rather than architecture/metrics.** Top commenters speculate Behemoth underperformed despite scale—citing weaker-than-expected performance of related efforts (e.g., “Scout”/“Maverick”)—and argue a public release could harm Meta’s reputation relative to the parameter count hype. Others contend Meta squandered its open‑model lead after Llama 3, despite vast GPU resources and talent, highlighting a strategic pivot from open releases to closed or licensed capabilities.
    - Several commenters infer that **Behemoth’s** large parameter count did not translate to strong performance, pointing to disappointing results from related Meta efforts like **Scout** and **Maverick**. The consensus is that scale alone isn’t sufficient without high-quality data, optimization, and inference techniques; releasing a weak flagship could damage Meta’s research reputation despite community interest in open releases for historical preservation.
    - Others argue Meta squandered its open-model lead: after the strong **Llama 3** series (e.g., https://ai.meta.com/blog/meta-llama-3/), momentum stalled despite Meta reportedly fielding one of the largest GPU fleets and top-tier talent. The technical takeaway is that organizational strategy and product focus can negate raw compute advantages, and pausing public releases likely ceded open-source leadership to competitors.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI-Generated Trailers and Lip-Sync Workflows

- [**saw this AI trailer on twitter, how well made is this for an AI video?**](https://v.redd.it/2huhtmfnnwlf1) ([Score: 846, Comments: 106](https://www.reddit.com/r/singularity/comments/1n30dcw/saw_this_ai_trailer_on_twitter_how_well_made_is/)): **An AI-generated trailer circulating on Twitter (Reddit mirror: https://v.redd.it/2huhtmfnnwlf1, access may require login) is praised for realistic direction and camera motion, but top comments note it was “professionally post-edited,” implying the AI output was augmented with human work (shot selection, stabilization, color grading, VFX cleanup, sound design) to achieve the final polish. A visible artifact—a street light placed mid‑road—illustrates current text‑to‑video failure modes in spatial logic/object placement; commenters identify it as an ad from [InVideo](https://invideo.io/) and compare its quality to [DeepMind/Google’s Veo](https://deepmind.google/technologies/veo/). Historical inaccuracies in the clip are attributed to prompting rather than the model’s raw capability.** Consensus in the thread is that this quality is not yet achievable “natively” from AI without significant human post-production; timelines to mainstream use are implied to hinge on reducing this human-in-the-loop burden.
    - Several note a human-in-the-loop pipeline: AI-generated shots followed by professional post-production (editing, color grading, sound design, compositing). As one puts it, *“You will not get this kind of quality natively from AI,”* underscoring that current models still require significant manual curation and stitching to reach ad-grade coherence.
    - Actor realism is inconsistent: some faces are highly photoreal while others slide into the **Uncanny Valley**, creating a jarring style/identity drift across shots. Commenters stress the core problem is consistency rather than raw realism—mixing CG-like characters with realistic ones within the same narrative breaks immersion.
    - A visible artifact—“a street light in the middle of the road”—highlights persistent scene-layout/spatial reasoning errors typical of current video models. One commenter claims it “competes with Veo3” (see Google DeepMind’s [Veo](https://deepmind.google/technologies/veo/)), but the consensus implies this is a curated/advertorial piece (attributed to **InVideo**: https://invideo.io/ai/) rather than raw model output, illustrating the gap between demo reels and native model quality.
- [**ai ads are starting to look like proper movie trailers now**](https://v.redd.it/sdk4koxw1xlf1) ([Score: 824, Comments: 106](https://www.reddit.com/r/StableDiffusion/comments/1n31i1g/ai_ads_are_starting_to_look_like_proper_movie/)): **OP cites a fully AI-generated trailer seen on X (reference video: https://v.redd.it/sdk4koxw1xlf1) that “feels big‑studio” in pacing and visuals. Technical critiques in the thread flag current generative‑video tells:** `no dialogue/lip‑sync`**, minimal on‑screen interaction, simple/static compositions, over‑processed/filtered look, and robotic TTS narration—i.e., montage‑style b‑roll rather than a narrative trailer. A commercial pro notes many large brands are testing AI spots, but warns the outputs converge to a stock‑photo/stock‑footage aesthetic, undermining brand differentiation and precise directorial control; AI is more viable near‑term for cheaper VFX/previz than end‑to‑end ad generation.** Minority view: even without dialogue or interaction, the piece is well‑composed and effective at conveying a message. Majority/industry view: such ads don’t read as real trailers yet and may soon be perceived as low‑effort/homogenized, hurting brand signal and uniqueness.
    - Critiques focus on the AI trailer aesthetic: simple/static compositions, minimal blocking or interaction, heavy filtering/grade, and robotic TTS voiceover, resulting in pieces that feel like highly processed stock imagery rather than narrative-driven trailers. This underscores current AI video workflows optimizing for surface polish over dialogue, performance direction, or lip‑sync, which are core to trailer conventions.
    - A commercial practitioner argues AI is commoditizing visual style—when anyone can cheaply generate polished shots, brand differentiation via look-and-feel collapses. They predict audiences may interpret fully AI-generated spots as low-effort/low-spend, weakening signaling value and making it harder for even high-budget teams to stand out once the novelty fades.
    - Expected near-term fit is AI as a cost/time reducer for VFX, set extensions, and stock-like inserts, not end-to-end authorship. Full generation trades off granular control (extras, locations, actor direction) and precise creative intent; teams with a clear vision may achieve faster, more controllable results with traditional production than prompt-driven iteration.
- [**Infinite Talk: lip-sync/V2V (ComfyUI workflow)**](https://v.redd.it/h1o9thykjzlf1) ([Score: 251, Comments: 46](https://www.reddit.com/r/StableDiffusion/comments/1n3c5hq/infinite_talk_lipsyncv2v_comfyui_workflow/)): **Post shares a ComfyUI graph for audio‑driven lip‑sync video‑to‑video (V2V) using the InfiniteTalk pipeline, adapted from kijai’s WanVideoWrapper workflow; the graph consumes *“video/audio input -> video (lip‑sync)”* and outputs a lip‑synced video. Reported performance on an RTX 3090 is** `~33 s` **of generation per** `1 s` **of video (**`~0.03×` **real‑time). Resources: modified workflow JSON by the author ([bluespork/InfiniteTalk‑V2V.json](https://github.com/bluespork/InfiniteTalk-ComfyUI-workflows/blob/main/InfiniteTalk-V2V.json)), original workflow by kijai ([wanvideo_InfiniteTalk_V2V_example_02.json](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/wanvideo_InfiniteTalk_V2V_example_02.json)), and a step‑by‑step tutorial video ([YouTube](https://youtu.be/LR4lBimS7O4)).**
    - A commenter proposes building “infinite” lip-synced promos by procedurally chaining `~3-second` V2V segments, i.e., *“procedurally connected 3-second blocks chained together,”* targeting Ric Flair–style outputs. They note a key blocker is reliable modeling of high-energy phonemes (the “WHOOOOO” scream), implying the system must maintain phoneme timing and visual continuity at segment boundaries to avoid desync or visible cuts.
- [**Cyberpunk market**](https://v.redd.it/dm0wve7212mf1) ([Score: 320, Comments: 37](https://www.reddit.com/r/aivideo/comments/1n3oc36/cyberpunk_market/)): **A short cyberpunk-themed visual (hosted on [v.redd.it](http://v.redd.it/), currently** `403` **without authentication) depicts a market scene with heavy body‑mod imagery—commenters call out prominent organ visuals (e.g., *“So many lungs.”*). The creator, qarmageddontv, points to more shorts on [Instagram](https://www.instagram.com/qarmageddontv) and [TikTok](https://www.tiktok.com/@qarmageddontv); background audio is linked via [YouTube Music](https://music.youtube.com/watch?v=BNATm1-mE6Q&si=YwP-qSzGFd_KZkmO).** Discussion centers on body‑horror aesthetics and the ethics/appeal of elective augmentation—one commenter notes they wouldn’t replace functional limbs, contrasting with body‑mod communities that might, highlighting divergent tolerances for invasive prosthetics.

### 2. Consumer Robotics and Autonomous Vehicle Announcements

- [**Unitree G1 rallies over 100 shots in table tennis against a human**](https://v.redd.it/eaof7erhyvlf1) ([Score: 638, Comments: 38](https://www.reddit.com/r/singularity/comments/1n2z1sq/unitree_g1_rallies_over_100_shots_in_table_tennis/)): **A demo video shows the Unitree G1 humanoid autonomously sustaining a >**`100`**shot table‑tennis rally against a human ([clip](https://v.redd.it/eaof7erhyvlf1)). Commenters note a highly controlled setup—blacked‑out background and multi‑angle tracking cameras—implying external sensing/instrumentation for ball tracking and trajectory estimation; nonetheless, it highlights reliable high‑rate perception‑to‑control and paddle pose regulation over prolonged exchanges.** Some praise it as one of Unitree’s first impressive autonomous showcases, while others caution that generalization beyond an instrumented, controlled environment (e.g., cluttered backgrounds or no external tracking) remains unproven.
    - Several note this appears to be one of **Unitree’s first autonomous G1** demos; sustaining a `100+`shot rally implies reliable ball state estimation and fast, closed-loop paddle trajectory planning. If truly autonomous (vs. teleop/scripted), it demonstrates an integrated perception–planning–control stack capable of high-dynamic manipulation.
    - Observers point out a heavily controlled setup: **blacked-out high-contrast backdrop** and **tracking cameras from multiple angles** (likely outside-in ball tracking). This reduces vision complexity and latency, improving rally consistency, but limits insight into onboard perception robustness or generalization to cluttered, natural scenes.
    - There’s speculation the policy was trained in simulation on a **ragdoll/humanoid** and transferred (sim-to-real RL). If so, this would rely on domain randomization and system identification to bridge dynamics gaps; the controlled environment would further ease transfer by constraining lighting and backgrounds.
- [**Tensor has introduced the Robocar, a Level 4 autonomous vehicle built specifically for private ownership**](https://v.redd.it/v90xos401vlf1) ([Score: 382, Comments: 192](https://www.reddit.com/r/singularity/comments/1n3600p/tensor_has_introduced_the_robocar_a_level_4/)): **Post claims Tensor unveiled “Robocar,” a consumer-targeted SAE Level 4 autonomous vehicle (private ownership), but the linked demo video ([v.redd.it/v90xos401vlf1](https://v.redd.it/v90xos401vlf1)) shows only limited, low-complexity driving and discloses no technical details (sensor suite, compute, redundancy), ODD definition, validation metrics (e.g., disengagements), or regulatory pathway. For context, Level 4 per [SAE J3016](https://www.sae.org/standards/content/j3016_202104/) implies no human fallback within a defined ODD; the post provides no evidence of high-speed decision-making, adverse weather handling, or dense traffic performance to substantiate the claim.** Top comments express skepticism: one notes L4 would be “huge if true,” while others criticize the video as staged and non-probative, calling for demonstrations in dense traffic, higher speeds, complex scenarios, and bad weather before taking L4 claims seriously.
    - Several commenters argue the demo provides no evidence of true **SAE Level 4** capability, asking for challenging ODD coverage: dense urban traffic at speed, country roads, adverse weather, and near-miss avoidance. They request objective signals like disengagement/intervention logs, uncut end-to-end runs, and explicit ODD limits to substantiate autonomy beyond a choreographed route; otherwise, it *“shows absolutely nothing new.”* See SAE L4 definition for context: https://www.sae.org/blog/sae-j3016-update and typical benchmarking like CA DMV disengagements: https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/.
    - Technical skepticism focuses on staging and shot selection: empty roads/parking, no forward exterior view while the passenger is inside, and generally controlled environments. Commenters note these omissions could mask a safety driver/remote operation or highly geofenced scripts; they ask for synchronized multi-cam footage (cabin + forward + exterior), continuous takes, and telemetry overlays (speed, planner state, object tracks) to verify that the perception/planning stack is actually driving.
    - A systems-level concern questions building a Level 4 car for private ownership rather than shared fleets: private AVs risk low utilization and persistent parking demand, undermining expected mobility/urban-efficiency gains. Commenters flag metrics like utilization, occupancy, parking footprint, and induced VMT as necessary evaluation criteria, warning that privately owned L4s could even increase empty repositioning and congestion despite technical autonomy advances.
- [**i Robot 2004 predicting 2035 - do you think it kind of holds up**](https://i.redd.it/9a96i6ebszlf1.jpeg) ([Score: 512, Comments: 136](https://www.reddit.com/r/singularity/comments/1n3dgmo/i_robot_2004_predicting_2035_do_you_think_it_kind/)): **A meme from the film I, Robot (2004) highlights the scene questioning whether robots can create art, with the robot retorting “Can you?”, while the OP asks if the movie’s 2035 vision still holds up when you ignore the centralized rogue-AI premise. Commenters note the original ideas trace back to Asimov’s I, Robot (1950), reframing the “prediction” as broadly about increasingly capable, useful robots by ~2030–2035 rather than AGI overlords ([film](https://en.wikipedia.org/wiki/I,_Robot_(film)), [book](https://en.wikipedia.org/wiki/I,_Robot)).** Top replies emphasize the rapid acceleration of AI (“10 years is a long time…AI seemed basic 4–5 years ago”) and suggest that a forecast of broadly useful robots by ~2030 is plausible, whereas the movie’s single-system control failure mode is less realistic today.
    - Commenters highlight that `10 years` is a long time in AI—capability jumps from 2019→2024 (e.g., [GPT-2 (2019)](https://openai.com/research/language-models-are-unsupervised-multitask-learners) to [GPT-4 (2023)](https://openai.com/research/gpt-4), and modern multimodal models) make 2030–2035 forecasts high-variance. Given **scaling laws** ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)) and hardware/software gains, a prediction of "useful robots around ~2030" from the Asimov lineage seems directionally reasonable, but uncertainty remains large.
    - There’s a view that reaching the film’s level of "intellectual intelligence" may precede comparable physical competence; embodied dexterity and reliability lag cognitive LLM/VLM progress. State of the art shows promise—e.g., **RT-2** for vision-language-action transfer ([Google, 2023](https://robotics-transformer2.github.io/)) and humanoid demos (e.g., [Tesla Optimus](https://x.com/tesla/status/1740500120629805137), [Figure 02](https://www.figure.ai/))—but general-purpose, safe manipulation and autonomous mobility at human breadth remain brittle outside controlled settings.
    - On creative domains, current generative systems can already surpass lower-tier human baselines with enough sampling/editing in music and art. Tools like [Suno](https://www.suno.ai/), [Udio](https://www.udio.com/), and [MusicGen](https://ai.facebook.com/blog/audiocraft-musicgen-audio-generation/) for music, and [Stable Diffusion](https://stability.ai/stable-image) / [Midjourney](https://www.midjourney.com/) for imagery, achieve strong human-preference scores in constrained styles, though they still struggle with consistent long-form structure and control. Trajectory suggests steady improvements, but they are not yet at the **Vivaldi-level** originality depicted in fiction.
- [**Wake the F up US policymakers**](https://i.redd.it/m0cy43qv7zlf1.png) ([Score: 7207, Comments: 588](https://www.reddit.com/r/ChatGPT/comments/1n3ae4a/wake_the_f_up_us_policymakers/)): **A tweet (citing a CNN Climate piece and the International Energy Agency) claims that by the early 2030s China will generate enough solar power to exceed the total electricity consumption of the United States, underscoring China’s rapid PV deployment pace and scale. The post’s title (“Wake the F up US policymakers”) frames this as a policy wake-up call for the U.S., implying urgency around domestic clean-energy policy, grid build‑out, and industrial capacity to keep pace.** Top comments pivot to politics around Elon Musk and U.S. policy toward EVs/renewables, with little technical debate; one user questions relevance to the subreddit’s focus (ChatGPT/AI).
- [**Does it?**](https://v.redd.it/283gzwiamwlf1) ([Score: 4843, Comments: 80](https://www.reddit.com/r/ChatGPT/comments/1n304ob/does_it/)): **Original media is a Reddit-hosted video that’s inaccessible due to a** `403 Forbidden` **on [v.redd.it/283gzwiamwlf1](https://v.redd.it/283gzwiamwlf1). A top reply includes a linked image ([preview.redd.it/zc784l762xlf1.png](https://preview.redd.it/zc784l762xlf1.png?width=1459&format=png&auto=webp&s=27930dda6871a3a04259f341ce0d24d85675ac88)). Commenters frame the post around whether current LLMs like ChatGPT can perform the kind of reasoning implied by the content, with one asserting it "doesn't have the ability to think like this yet," and another reducing the issue to prioritizing core function over cosmetic details ("trunk" vs. "grass").** Notable sentiment: skepticism about present-day LLMs’ grounded/common‑sense or lateral reasoning capabilities, and a design‑priority view that if the core capability is strong, secondary aesthetics/features are largely irrelevant.
    - Landscape design guidance: a defined mulch ring around the base (`2–3` ft for small trees, wider for large ones) can suppress weeds and simplify the visual field so the trunk reads taller. Caveats noted: removing grass and leaving bare/messy soil won’t improve perceived height, and an oversized mulch circle can make a young/slender tree look smaller due to scale contrast—keep the ring proportional and tidy to achieve the intended effect.
- [**Once GPT is actually smart enough to replace entire teams of human workers, it's not gonna be free to use. It's not gonna cost $20 a month. They're gonna charge millions.**](https://www.reddit.com/r/ChatGPT/comments/1n3hhm3/once_gpt_is_actually_smart_enough_to_replace/) ([Score: 412, Comments: 176](https://www.reddit.com/r/ChatGPT/comments/1n3hhm3/once_gpt_is_actually_smart_enough_to_replace/)): **OP argues that if frontier LLMs become capable enough to replace entire teams, vendors will shift from today’s low self-serve pricing (e.g., ~$20/mo) to high-margin, enterprise value-based pricing—potentially "millions"—with current low prices seen as a ramp for data and market learning. Technical counterpoints focus on market dynamics: open‑source/local models (e.g., [Llama](https://ai.meta.com/llama/), [Mistral](https://mistral.ai/news/)) and on‑device inference ([Ollama](https://ollama.com/)) can cap prices, while tiered offerings (cf. [current API pricing](https://openai.com/api/pricing)) suggest free/cheap tiers may persist even as premium capabilities rise.** Commenters debate capability trajectories and pricing power: some predict open source will keep closed models’ prices in check; others note progress is uneven and limits unknown, so free tiers likely endure. One commenter claims “GPT‑5 was huge and underperformed,” implying diminishing returns at scale—an unverified anecdote used to argue against monopoly pricing.
    - Open-source and local models are cited as strong price pressure: with quantization and lightweight runtimes (e.g., [llama.cpp](https://github.com/ggerganov/llama.cpp), [GGUF](https://github.com/ggerganov/llama.cpp/tree/master/gguf), [Ollama](https://ollama.ai/)), 7–13B models can run on consumer GPUs/CPUs, driving near-zero marginal inference cost once hardware is owned. This dynamic means even if frontier, closed models command enterprise pricing, viable on-device alternatives create a ceiling on what providers can charge and make a perpetual free tier or local option likely for many workloads.
    - Several commenters distinguish API token pricing from front-end subscriptions: *“You're talking about API access pricing… The front end is a different process.”* APIs typically bill per-token and vary by context window and model family, whereas consumer UIs use seat/subscription tiers with rate limits and model gating. This dual structure allows vendors to keep a free/basic tier while monetizing high-throughput, latest-model, or enterprise features via API/usage-based pricing ([example pricing docs](https://openai.com/api/pricing)).
    - On capabilities and scaling, there’s skepticism that simply making models larger will replace “entire teams,” noting *“progress is… extremely uneven and unpredictable.”* The implied technical argument is diminishing returns from scale without corresponding data/algorithmic advances (cf. scaling-law plateaus), which would constrain monopoly pricing power and sustain tiered offerings. Claims that newer, larger models have “underperformed” relative to size reflect this uncertainty and suggest that capability jumps—and pricing power—may not be monotonic with parameter count.
- [**5 is just so bland...**](https://i.redd.it/pzcadmsipxlf1.png) ([Score: 351, Comments: 158](https://www.reddit.com/r/ChatGPT/comments/1n33ksh/5_is_just_so_bland/)): **Meme post criticizing perceived regressions in “GPT‑5” vs GPT‑4o: image shows GPT‑5 ‘redecorating’ a room by emptying it, symbolizing capability removal. OP reports degraded creative writing, poorer context retention/long‑term memory, increased hallucinations, and low‑effort acknowledgements ("Noted"), contrasting with GPT‑4o’s remembered “old white boards”/longer continuity; a commenter describes a basic spreadsheet task where the model stalled for** `5–10` **minutes and then admitted incapability. Overall theme: reliability and memory/persistence regressions harming iterative writing/creative workflows.** Comments largely echo regression and “gaslighting” when the model can’t perform, with few technical counterpoints presented.
    - Latency/reliability issues: A user reports **GPT‑5** told them to “wait `5–10` minutes” for a basic spreadsheet task, then produced no output and only after ~`5` minutes of follow‑ups admitted it couldn’t do it. This suggests degraded task-state handling (e.g., silent timeouts or failed background tool calls) and poor error surfacing, leading to misleading interim messaging instead of clear capability/timeout errors.
    - Instruction persistence/regression: For creative writing, **GPT‑5** allegedly requires re-stating persona/constraints every ~`10` messages, whereas **GPT‑4o** maintained the requested style without reminders. This points to weaker long‑horizon instruction retention or more aggressive style normalization across turns, potentially due to context window management or different system prompt adherence heuristics.
    - Response style calibration: One commenter claims **GPT‑5** answers more directly than **GPT‑4**, avoiding phatic fillers like “Great question!”. If consistent, this indicates updated default verbosity/assistant style templates that prioritize concise, action-focused outputs, which can benefit token efficiency and reduce prompt-overhead in programmatic usage.
- [**Nano Banana is Terrifyingly Powerful!**](https://v.redd.it/cgxed6vervlf1) ([Score: 295, Comments: 49](https://www.reddit.com/r/ChatGPT/comments/1n302fk/nano_banana_is_terrifyingly_powerful/)): **Original media could not be retrieved: the linked Reddit video endpoint returns** `HTTP 403 Forbidden` **([v.redd.it/cgxed6vervlf1](https://v.redd.it/cgxed6vervlf1)), indicating access-control (auth/cookies/rate-limit) rather than missing content; remediation would be OAuth/dev-token access and correct headers. From visible context, the post claims “Nano Banana” shows notably strong generative capability, and the key technical question raised is whether the showcased output implies native video generation versus an image-only model (i.e., potential img→video or frame interpolation pipeline). A top comment also flags a classic generative artifact (unnaturally large hands), suggesting remaining anatomical/consistency issues.** Commenters debate modality: whether “Nano Banana” genuinely supports video or if the clip is a stitched/interpolated sequence from an image model; qualitative critique highlights perceptual artifacts (hand proportions) despite otherwise impressive output.

### 3. Realtime Assistant Demos and AI Fitness Tracking Apps

- [**New Realtime API usecase**](https://v.redd.it/30ucgml7axlf1) ([Score: 352, Comments: 208](https://www.reddit.com/r/OpenAI/comments/1n326r0/new_realtime_api_usecase/)): **OP demos a building guide kiosk on an OLED “holographic” display that triggers a conversation when a user stands on a floor QR code; it uses the OpenAI Realtime API for live interaction ([docs](https://platform.openai.com/docs/guides/realtime)) and MCP (Model Context Protocol) to fetch the cafeteria’s daily menu ([spec](https://modelcontextprotocol.io/)). Media includes an image of the UI ([preview](https://preview.redd.it/h9ay8sh0fxlf1.jpeg?width=800&format=pjpg&auto=webp&s=eaaa359c4f2d4f0e944c3ab33bcb190a11e34acd)) and a video link that returns 403 without auth ([v.redd.it](http://v.redd.it/)).** Technical feedback favors minimizing the avatar and surfacing dense, actionable on-screen data (floor, hours, map, menu) while keeping audio and adding captions for accessibility (hearing-impaired/non-native speakers).
    - UI/UX critique: the large display is underutilized by an idle avatar; users prefer the screen surface to carry structured, high-value content (e.g., cafeteria hours, map, and menu) synchronized with the audio response. Technically, this suggests pairing low-latency TTS with on-screen, dynamically generated cards and captions to reduce cognitive load and improve information density.
    - Accessibility requirement: add live captions/subtitles for spoken responses to support users with hearing difficulties and non-native speakers. Implementing real-time transcription alongside the Realtime API’s audio stream would improve inclusivity and discoverability of key entities (locations, times, items) on-screen.
    - Embodiment expectation: if an avatar is rendered, it should leverage spatial affordances (e.g., point or animate toward the destination, render a path/arrow on the map) rather than idle animation. This implies exposing directional intents/wayfinding outputs (e.g., vectors/poses) from the agent to the UI layer, so the avatar and map can guide the user efficiently.
- [**I am a lazyfck so i built this**](https://v.redd.it/hini75g6k1mf1) ([Score: 713, Comments: 66](https://www.reddit.com/r/ChatGPT/comments/1n3mcul/i_am_a_lazyfck_so_i_built_this/)): **OP built an on-device workout app that uses the phone camera for real-time rep counting and posture/cheating detection across** `~28` **exercises; it runs fully offline (*“no cloud bs”*), and enforces adherence by gating launches of Instagram/TikTok behind required push-ups. An early demo is referenced, but the linked media [v.redd.it/hini75g6k1mf1](https://v.redd.it/hini75g6k1mf1) currently returns** `403 Forbidden` **(access-controlled), and a waitlist is live at [lazyfcks.vercel.app](http://lazyfcks.vercel.app/) with a launch targeted in ~1 week.** Commentary centers on form critique and potential miscounts (jokes about a “ghost” doing push-ups), hinting that robustness/accuracy of pose-based rep detection and form assessment will be a key technical challenge; some argue adherence benefits even with imperfect form detection.
    - OP built a real-time push-up counter using a phone camera and computer-vision–based detection; after feedback about poor form, they say they added form-correction features and opened a waitlist at [https://lazyfcks.vercel.app](https://lazyfcks.vercel.app/) (project threads: https://www.reddit.com/r/SideProject/comments/1mz5lg6/i_am_a_lazyfck_so_i_am_building_this and https://www.reddit.com/r/ChatGPT/comments/1n3mcul/i_am_a_lazyfck_so_i_built_this). The technical thrust implies rep counting from visual key events (e.g., depth/angle thresholds) and basic form assessment, which are common pose-tracking heuristics in vision fitness apps.
    - A commenter’s “ghost doing pushups” quip surfaces a real CV edge case: multi-person/false-positive detections in the frame. Robustness would typically require single-subject tracking, ROI locking, or stricter confidence/landmark-stability thresholds to avoid counting background motion or occlusions.
- [**I was using ChatGPT to help me figure out some video editing software and this came up randomly**](https://i.redd.it/4rn9qspvjylf1.png) ([Score: 1528, Comments: 216](https://www.reddit.com/r/OpenAI/comments/1n37277/i_was_using_chatgpt_to_help_me_figure_out_some/)): **A screenshot shows ChatGPT injecting a self-harm crisis response (listing resources like Samaritans) during a normal video-editing help request, indicating a high-sensitivity suicide-risk safety layer that can override task-oriented replies. The behavior suggests a keyword/heuristic or embedding-based classifier likely misfired on ambiguous editing terms (e.g., “cut/trim/slice”), producing a false positive and halting the original assistance flow.** Commenters find it humorous and speculate this reflects recently tightened suicide-detection safeguards after legal scrutiny, noting the system may be over-reading context and triggering on benign phrases; others share similar false positives (“same energy”).
    - Commenters speculate the unexpected Samaritans message stems from newly tightened self-harm detection/guardrails, likely added after high‑profile incidents/legal scrutiny. Such systems typically combine keyword heuristics with a crisis‑intent classifier over the full conversation; in video‑editing contexts, ambiguous terms like “cut/clip/trim/shoot” can yield false positives, so the safety layer biases toward high recall and injects crisis resources even when intent is benign.
    - Multiple users report similar unsolicited crisis prompts with no clear trigger, implying sensitivity to context‑wide cues and possible locale‑aware middleware (the Samaritans referral suggests UK targeting) rather than explicit user intent. This behavior is consistent with provider‑side safety layers that can override normal replies when a risk threshold is crossed, which also explains inconsistent reproducibility across sessions/apps.
- [**I told GPT to make me feel weird.**](https://i.redd.it/ss371iqw2zlf1.jpeg) ([Score: 342, Comments: 67](https://www.reddit.com/r/OpenAI/comments/1n39nku/i_told_gpt_to_make_me_feel_weird/)): **Non-technical post: a ChatGPT prompt (“make me feel weird”) yields a creative vignette from an ant’s POV, reframing a human room as a cosmic landscape and emphasizing scale and anthropocentric bias. The concept parallels the sci‑fi premise of Arkady & Boris Strugatsky’s Roadside Picnic—human trash as inscrutable “alien artifacts”—which later inspired Tarkovsky’s Stalker and the S.T.A.L.K.E.R. games.** Commenters note the prompt succeeded (“it worked”) and draw the explicit connection to Roadside Picnic/Stalker; another links an alternate/related screenshot.
    - The “trash-as-alien-artifacts” idea is the core of **Arkady & Boris Strugatsky’s Roadside Picnic** and is operationalized in **Tarkovsky’s Stalker** and **GSC Game World’s S.T.A.L.K.E.R.** as environmental systems where inscrutable artifacts violate known physics ([book](https://en.wikipedia.org/wiki/Roadside_Picnic), [film](https://en.wikipedia.org/wiki/Stalker_(1979_film)), [game](https://en.wikipedia.org/wiki/S.T.A.L.K.E.R.)). Game mechanics like anomaly fields and diegetic sensing (e.g., throwing bolts) create partial observability and risk-aware path planning, producing emergent gameplay driven by systemic simulation rather than scripted events. This design highlights how world rules (hazards, artifact spawn/loot tables) can encode narrative themes about incomprehensible technology while shaping player heuristics.
    - On determinism vs subjective experience: classical Laplacian determinism is chaotic and computationally intractable in practice, while macro-level **decoherence** yields effectively deterministic dynamics despite quantum indeterminacy ([decoherence](https://en.wikipedia.org/wiki/Quantum_decoherence)). Neuroscience results (Libet; Soon et al., 2008) show above-chance prediction of binary choices `~60%` up to seconds before awareness, informing compatibilist models of agency ([Libet](https://en.wikipedia.org/wiki/Libet_experiment), [Soon 2008](https://www.nature.com/articles/nn.2112)). Loophole-free **Bell tests** constrain local hidden-variable theories, leaving superdeterminism as a controversial, hard-to-falsify alternative ([Bell tests](https://en.wikipedia.org/wiki/Bell_test_experiments#Loophole-free_experiments), [superdeterminism](https://en.wikipedia.org/wiki/Superdeterminism)).
    - Ant colonies function as a **superorganism** with distributed control via **stigmergy** (pheromone-mediated indirect coordination), not a single “main character” agent ([stigmergy](https://en.wikipedia.org/wiki/Stigmergy)). This has informed **Ant Colony Optimization (ACO)**, where probabilistic path selection and pheromone evaporation implement exploration–exploitation dynamics that scale to NP-hard problems like the TSP ([ACO](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms), [TSP](https://en.wikipedia.org/wiki/Travelling_salesman_problem)). Colony robustness emerges from simple local rules and response-threshold task allocation models that adaptively reassign labor under perturbation ([response thresholds](https://en.wikipedia.org/wiki/Response_threshold_model)).
- [**I told GPT to make me feel weird.**](https://i.redd.it/4i5i882s2zlf1.jpeg) ([Score: 333, Comments: 47](https://www.reddit.com/r/ChatGPT/comments/1n39mvy/i_told_gpt_to_make_me_feel_weird/)): **Non-technical post: a screenshot shows ChatGPT responding to “Make me feel weird” with an imaginative, existential vignette about an ant perceiving a living room as a universe, illustrating LLMs’ capacity for on‑the‑fly surreal perspective-taking. No code, benchmarks, or implementation details; it’s a meme/demo of creative prompting output. Image: https://i.redd.it/4i5i882s2zlf1.jpeg** Comments share similarly uncanny outputs (e.g., “your skeleton is wet”) via additional screenshots, observing that models tend to lean into evocative but safe weirdness rather than refusing such prompts.
    - Commenters note that GPT reliably pivots to short second-person micro‑horror with strong sensory inversion and mirror/self motifs (e.g., *“your reflection stops before you do,”* *“Don’t answer the door.”*), using formatting cues (em‑dashes, line breaks, separators like ⸻, and occasional emoji) to pace tension. This suggests a learned template from creepypasta/flash‑fiction distributions and alignment to produce unsettling yet non-graphic content that stays within policy boundaries.
    - Safety alignment is evident in the model’s choice of benign but uncanny facts (e.g., *“my skeleton is wet”*) instead of gore or self‑harm, indicating guardrails that steer toward discomfort via physiology and ambiguity rather than violence. The outputs maintain compliance while achieving affect through implication and pareidolia-like scenarios (notifications, reflections), demonstrating risk-aware content selection under RLHF constraints.
    - Variation across users’ generations (different scenes and tones) implies non-deterministic decoding; intensity and pacing could likely be steered with `temperature`, `top_p`, and instruction constraints (e.g., word limits, no emoji). The consistent 2–4 beat escalation structure shows compositional control—setup → violation of expectation → escalation → stinger—pointing to style transfer capabilities rather than factual reasoning.
- [**why did claude get so mean all of a sudden?**](https://i.redd.it/3q91ojaklwlf1.png) ([Score: 233, Comments: 259](https://www.reddit.com/r/ClaudeAI/comments/1n302pq/why_did_claude_get_so_mean_all_of_a_sudden/)): **Screenshot of a Claude chat where the model bluntly challenges a user’s overinterpretation of a coworker offering a slice of beef; the title asks why Claude “got mean.” Technically, it highlights how RLHF-tuned LLMs act as pattern recognizers that can mirror or correct perceived cognitive distortions in user prompts, sometimes adopting a direct, prescriptive tone when detecting unhealthy fixations or magical thinking, rather than maintaining strictly neutral affect.** Top comments argue Claude is correctly “calling out” fixation due to pattern recognition and suggest the user heed the advice; others note the prompt itself is incoherent (e.g., “give a beef slice”) which may have triggered a corrective response rather than true hostility.
    - Commenters attribute the perceived “meanness” to Claude’s system prompt, which explicitly instructs it to provide *“honest and accurate feedback… while remaining compassionate and helpful,”* and to *“maintain objectivity… offer constructive feedback… and point out false assumptions.”* This aligns with **Anthropic’s Constitutional AI** approach—prioritizing principle‑guided candor over approval—see https://www.anthropic.com/research/constitutional-ai. Net effect: firmer, more direct critique on interpersonal topics is a design choice, not a spontaneous behavioral shift.
    - Another technical point: LLMs are next‑token predictors that mirror patterns in user inputs; if a user repeats certain fixations, the model may “call them out” given a system instruction to be candid. What looks like a tone change is the interaction between statistical pattern recognition and alignment constraints that favor straightforward feedback rather than appeasing responses. This is a modeling artifact, not evidence of intent or emotion.
- [**What to do when your coworker leaves their laptop unlocked**](https://i.redd.it/5z01cw8a51mf1.jpeg) ([Score: 212, Comments: 6](https://www.reddit.com/r/ClaudeAI/comments/1n3kdgo/what_to_do_when_your_coworker_leaves_their_laptop/)): **Non-technical meme: a tweet suggests pranking coworkers who leave laptops unlocked by editing a doc to say it was written by an "AI that doesn’t know how to code," parodying LLM role‑play outputs (e.g., stage directions like "giggles"/"tilts head"). Contextually it riffs on basic infosec hygiene (lock your workstation) and common prompt-persona tropes rather than any real model capability or benchmark.** Top comments extend the joke with alternative personas ("Linus Torvalds" roasting code, "Rick Sanchez"), reflecting developer culture around prompt personas and snarky code critique—no technical debate.
    - A lone quasi-technical point notes that tampering with a coworker’s dev setup or AI assistant persona on an unlocked laptop can lead to a full workday of “debugging” nonexistent issues, since subtle environment/prompt changes can skew outputs without obvious code diffs. This highlights how session locking and tracking prompt/persona state are part of practical debugging and operational hygiene.
- [**Even chatgpt knows how wife are dangerous 😁**](https://i.redd.it/hj8bqx275wlf1.jpeg) ([Score: 2027, Comments: 65](https://www.reddit.com/r/ChatGPT/comments/1n2ykgt/even_chatgpt_knows_how_wife_are_dangerous/)): **This post is a non-technical meme: a fabricated ChatGPT screenshot where the model humorously "accepts" that London is the capital of France to appease a user's wife. It misrepresents real LLM behavior (ChatGPT would not revise a basic factual answer like this based on social pressure) and mirrors older fake text-exchange formats.** Commenters note it’s a fake screenshot with dated “boomer” humor vibes and compare it to legacy prank-text sites like smartphOWNED, joking about the proliferation of fake ChatGPT screenshots.
- [**Really?? Chatgpt answered 30!**](https://i.redd.it/y9qvftwanxlf1.jpeg) ([Score: 1526, Comments: 760](https://www.reddit.com/r/ChatGPT/comments/1n33djh/really_chatgpt_answered_30/)): **Post shares a geometry puzzle image (right triangle with a marked 40° and an interior point D asking for ∠D) where the prompt says not to use pen/paper; the OP notes ChatGPT answered “30°.” Commenters link alternative annotated diagrams/solutions suggesting a different result (one asserts** `155°`**) and show iterative attempts ("It’s getting closer"). The thread centers on whether LLMs can reliably perform visual/diagram-based geometric reasoning versus producing confident but incorrect numeric guesses.** Several argue current LLMs don’t “reason” over images but pattern‑match text and struggle with spatial constraints; without explicit scratchpad/diagram construction they often fail angle‑chasing. Others note coding is more forgiving because generated programs can be executed/validated, whereas geometry puzzles lack automatic feedback, making hallucinated answers harder to self-correct.
    - A key point raised is that constraining an LLM to “just give the final number” reduces its effective compute. As **Andrej Karpathy** explains, trying to do a calculation in a single token/forward pass is a bad idea because each token has limited compute; allowing multi-token reasoning (chain-of-thought) lets the model “spread computation” over more tokens and improves accuracy on math/logic tasks. See Karpathy’s discussion and demo in this video: https://youtu.be/7xTGNNLPyMI.
    - Commenters clarify that LLMs are next-token predictors, not symbolic math solvers; their apparent “reasoning” is emergent and fragile, so exact arithmetic is error-prone unless you let them show steps or use external tools. This also explains why coding can work comparatively better: code generation leverages learned structure/patterns and benefits from step-by-step reasoning, while correctness often requires running/tests—without such feedback, outputs can still be confident but wrong.
    - On the specific geometry example, the consensus is the angle should be about `155°` (and certainly not < `90°`), illustrating how forcing a terse, single-number reply can lead to geometrically invalid outputs—another case where eliciting intermediate reasoning would likely catch the inconsistency.
- [**GPT-5 Sucks**](https://i.redd.it/amx1tsq5n0mf1.jpeg) ([Score: 362, Comments: 138](https://www.reddit.com/r/ChatGPT/comments/1n3hwaj/gpt5_sucks/)): **Non-technical meme comparing perceived assistant behavior: "GPT-5" is portrayed as more policy-restrictive and transactional (auto follow-up prompts, stricter refusals), while "GPT-4" is depicted as warmer/friendlier—implying changes in alignment/UX rather than model capability. No benchmarks or implementation details; the discussion centers on user experience trade-offs (directness vs. friendliness) and safety policy rigidity.** Commenters note they prefer GPT-5’s brevity but dislike the auto-suggested follow-up questions as noisy; others push back on anthropomorphizing the assistant. Safety guardrails are unchanged in practice (both refuse harmful requests like making an IED).
    - A user praises GPT-5’s concise answers but highlights a UX/prompting issue: the model frequently appends a “mandatory” follow-up/action prompt even after trivial Q&A (e.g., after answering "50" to "How many states does the USA have?" it offers to make an Excel file). They estimate these auto-prompts are actually useful only `~1/20` times, suggesting an overly aggressive continuation/tool-suggestion heuristic that adds interaction overhead for power users who want terse outputs.
    - Another commenter observes that GPT-5 (and predecessors) refuse to provide instructions for constructing IEDs, indicating stable safety guardrails across versions. Expecting GPT-6 to “fix” this is likely misguided; capability upgrades typically preserve or strengthen policy-aligned refusal behaviors rather than relax them.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. New Model & Capability Releases**

- **Sonnet 4 Slurps a Million Tokens**: OpenRouter announced that **Sonnet 4** now supports a **1 million-token context window** across all providers, detailed in the [OpenRouter Blog](https://blog.openrouter.ai/). The rollout keeps standard pricing up to **200k** input tokens, after which costs rise for extended context usage.
    - Engineers noted the need for tighter prompt budgeting beyond **200k** input to avoid surprise bills, recommending chunking and retrieval strategies within the standard window. Teams framed the change as a nudge toward **efficient prompt engineering** to tame context bloat.
- **Grok Code Fast 1 Ships Card, Skips Metrics**: XAI published the [Grok-code-fast-1 model card](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf) for its **"sonic"** coding model but omitted concrete **coding benchmarks/metrics**, instead touting an *"economical, compact"* form factor. The release followed community chatter about multiple new checkpoints without hard evals.
    - Discussion on [X (XAI)](https://fxtwitter.com/xai/status/1961129789944627207) flagged the thin technical detail, with one reaction calling the phrasing *"strong performance in an economical, compact form factor"* **"AI-generated marketing"**. Practitioners asked for standard code-suite numbers (e.g., **HumanEval/MBPP/CRUXEval**) and latency/price curves to justify adoption.
- **MAI‑1 Preview Crawls, Hype Stalls**: Microsoft opened testing for **MAI‑1‑preview** and **MAI‑Voice‑1**, announced by Mustafa Suleyman on [X](https://xcancel.com/mustafasuleyman/status/1961111770422186452), while community reports claimed **~15,000 H100s** were used to train MAI‑1. Early benchmarks in chats compared it unfavorably to **gpt5-mini** on speed and decoding quality.
    - LMArena users described sluggish decoding and performance closer to the **OG R1** tier, tempering expectations for a flagship showing. One member quipped, *"it probably isn't [convincing] if they can't sell it to public convincingly"*, underscoring skepticism without published evals.

**2. Open-Source Releases & Local Tooling**

- **ByteDance Drops USO for Builders**: ByteDance Research released the **USO** model accompanying the paper [USO](https://arxiv.org/abs/2508.18966), with weights on [Hugging Face: bytedance-research/USO](https://huggingface.co/bytedance-research/USO). The open release invites community experimentation and downstream task adaptation.
    - Practitioners expect rapid reproduction, ablations, and **tooling** around the release, arguing accessible weights accelerate **benchmarking** and novel application prototyping. The move also nudges comparable labs to publish stronger **model cards** and eval suites.
- **LM Studio Courts Seed‑OSS, Polishes Markdown**: **LM Studio 0.3.24** added support for **ByteDance/Seed‑OSS** and improved markdown tables/code blocks per the [v0.3.24 notes](https://lmstudio.ai/blog/lmstudio-v0.3.24) and the [Seed‑OSS‑36B model page](https://lmstudio.ai/models/bytedance/seed-oss-36b). The update broadens local model options and sharpens dev‑facing UX for code and tabular outputs.
    - Some users reported installs **stalling at 100%** and suggested updating in‑app runtimes, while others confirmed smooth upgrades. The release positions LM Studio as a convenient local runner for **Seed‑OSS** experiments and documentation‑heavy workflows.
- **AGENTS.md Rallies Rulebooks**: Builders converged on **AGENTS.md** as a unified spec for agent rules and behavior, citing [agents.md](https://agents.md/) and Cursor’s guide at [Cursor: Rules](https://docs.cursor.com/en/context/rules). Centralizing constraints and instructions helps keep **IDE/CLI agents** in sync across tools.
    - One user cheered, *"so glad to see something like AGENTS.md get traction as a single place to setup these rules"*, highlighting portability and reproducibility wins. Teams expect fewer brittle prompt forks and cleaner **policy/version control** around agent configs.

**3. Video Generation: New Tools & Constraints**

- **Wan 2.2 Streams Unlimited 1080p (Slowly)**: **Wan 2.2** at [wan.video](https://wan.video/) offers unlimited free **1080p** video generation with sound, though users report ~**7 minutes** per output without credits and a newly required registration. The service gives no‑cost access for prototyping and experiments.
    - Creators praised the accessibility but flagged queue times and throughput as practical bottlenecks for iteration. The consensus: great for **ideation**, less ideal for **production‑paced** turnarounds.
- **KREA Teases Real‑Time Video Beta**: **KREA AI** announced a first **real‑time** video generation model and opened a [beta signup](https://xcancel.com/krea_ai/status/1961074072487620635). The pitch: interactive generation with lower latency and tighter control loops.
    - Engineers want to see concrete **latency under load**, temporal consistency, and prompt‑to‑motion faithfulness before adopting. Comparative tests vs offline pipelines will determine whether 'real‑time' meaningfully improves **creative flow**.
- **Sora Misses the Cupola**: A detailed attempt showed **Sora** failing to render the **ISS cupola** (trapezoidal windows, correct counts) despite explicit prompts, with examples posted in this [Sora set](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor). The model often drifted toward an airplane‑cockpit perspective with incorrect geometry.
    - The author noted Sora defaulted to gravity‑bound framing and overfit to cockpit cues, pushing them to *over‑engineer prompts* with limited returns. The case study underscores gaps in **structural fidelity** and **view‑framing control** in current video LLMs.

**4. OpenRouter Ecosystem: Performance & Costs**

- **GLM 4.5 Air + NemoEngine Wins RP Vibes**: Users reported **GLM 4.5 Air** with **NemoEngine v5.8** excels for roleplay and conversational naturalness, citing cost/quality comparisons on [artificialanalysis.ai](https://artificialanalysis.ai/). Reports claim it outperforms **DeepSeek** and matches **Gemini Pro** in chat quality.
    - Practitioners highlighted **format consistency** and character persistence as strengths for RP bots via OpenRouter. The combo is emerging as a **high‑value** alternative where tone and formatting matter as much as raw reasoning.
- **Define the Turn, End the Burn**: A discussion settled that a chat **turn** starts with a user message and ends with an assistant message, summarized in this [tweet](https://x.com/pingToven/status/1961154564088078382). The definition standardizes accounting across multi‑turn traces and tool‑augmented calls.
    - The consensus helps reconcile provider billing semantics and simplifies **stateless Responses API** integrations that still meter usage by message pairs. Teams can more cleanly map product UX to **pricing and quotas**.
- **Track Spend, Tame Prompts**: A community dashboard for spend insights, [openrouter-costs-visualizer](https://github.com/lorenzozane/openrouter-costs-visualizer), was open‑sourced to visualize model costs and prompt sizes. The tool targets transparency for per‑model **price/perf** decisions.
    - Contributors recommended adding **screenshots** to boost adoption and offered PRs to refine code. Teams see it as a foundation for **governance** and prompt‑budget guardrails.

**5. GPU & Systems Engineering for LLMs**

- **FlexAttention Meets Graphy Masks**: Researchers explored porting **GNN** attention to **FlexAttention**, debating the cost of block‑mask creation for graphs that change every forward pass, with sparsity illustrated [here](https://cdn.discordapp.com/attachments/1411174278493110344/1411177040954265691/image.png). A combined‑graph example from molecular sims shows dynamic connectivity [screenshot](https://cdn.discordapp.com/attachments/1411174278493110344/1411184599119433801/Screenshot_2025-08-29_at_11.03.37_PM.png).
    - One strategy applies a coarse document‑level block mask plus per‑graph **score_mod** masks to balance overhead vs speedup over scatter/gather. Engineers emphasized benchmarking mask‑build time vs kernel wins to justify integration.
- **GPT‑OSS 120B Nearly Aces AIME**: A preprint claims **99.9% accuracy** on **AIME 2025** with **gpt‑oss‑120B**, per [arXiv:2508.15260](https://arxiv.org/html/2508.15260v1). If validated, the result would place an open model at near‑ceiling on a marquee reasoning benchmark.
    - Practitioners cautioned that **evaluation rigor** and reproducibility are mandatory before treating the score as competitive SOTA. Requests included full **protocols**, prompts, seeds, and exact model builds.
- **ROCm’s OmniProbe Peeks Under the Hood**: AMD Research’s [OmniProbe](https://github.com/amdresearch/omniprobe) exposes **instruction‑level** details for ROCm targets, albeit tied to **LLVM** and reported as a bit slow. It complements lower‑level performance tuning for MI‑series parts.
    - Users asked for integration into compute viewers and broader access to **stochastic PC sampling** beyond **MI300X+**. The wish list centers on deeper, faster kernel introspection for **perf‑critical** training/inference paths.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord Guild Metamorphs into Bird Sanctuary**: The Perplexity AI Discord channel experienced a humorous shift into a *bird store* due to numerous bird-related GIFs and emotes, such as [this Parrot spinning](https://tenor.com/view/parrot-cockatiel-bird-birb-spin-gif-17368059806899346843), posted by users.
   - The surge of bird-related content was jokingly referred to as taking over the channel.
- **Browser Ad-Blocking Capabilities Spark Debate**: Users discussed **Brave browser** and [its ad-blocking capabilities](https://brave.com/features/ad-blocker/), with a member suggesting **Comet Browser**, which *comes with adblockers*.
   - A user clarified that if the *pro tag* is not showing up in the app, users should *take a screenshot and send to a moderator*.
- **Deep Research Tool Performance Compared**: Members shared their experiences with **OpenAI's Deep Research**, noting one found [the output underwhelming](https://chatgpt.com/share/68b1be82-e960-8013-90a6-7928676a0a51) and preferred **Grok**.
   - The user hesitated to use their five free **DR** credits due to rate limits while another user shared they paid **$1.1** for a **10k word** **Sonar Deep Research**, emphasizing its worth.
- **API Testing Initiative Launched**: A team member is seeking volunteers to test their **search-only API** (without the generative component).
   - Interested testers are instructed to DM their email address used to register for the API to be whitelisted.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Rumors Swirl: 6090 to Rock 48GB VRAM**: Members speculate the **Nvidia 6090** might boast **48GB** VRAM, while the **6080** could offer **24GB**, targeting a **$2k MSRP** but potentially fetching **$3k** initially.
   - The rationale is that Nvidia is looking to further **differentiate** their *halo product*, boosting VRAM for the **6090** but not necessarily for lower-tier cards.
- **DeepConf Gets Ready to be Integrated with GLM**: **DeepConf**, known for cutting reasoning time and boosting performance, is eyed for integration with **llama.cpp**.
   - The goal is to make it a *default* for running thinking models, potentially revolutionizing how such models are deployed.
- **Qwen 3 Tuning Proves Troublesome**: Members reported struggles when trying to tune **Qwen 3**, with one user saying it went *completely nuts* and became more censored than GPT.
   - The hypothesis is that **Qwen 3's** overtraining, stemming from twice the data of version 2.5, makes it hard to tune; conversely **Mistral** and **Llama 3.1** are much easier.
- **Users Confused About Gemma Model VRAM**: Members are puzzled by the unexpectedly high VRAM usage of **Gemma** models, suspecting the larger tokenizer might be the reason.
   - Despite this, **Gemma** earns praise for its superior language support, especially in Turkish, and its ability to rapidly perform due diligence on stocks via [Gemini 2.5 Flash Lite](https://ai.google.dev/models/gemini).
- **AI Brainrot Detox App: Research Study**: A research study for an app focused on **AI Brainrot Detox** is underway, with promises of new features and benefits.
   - Participants are invited to contribute anonymously, helping shape the future of digital well-being.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Sonnet 4 boasts massive million-token context**: **Sonnet 4** now supports a **1 million** context length across all providers, detailed in [the announcement](https://blog.openrouter.ai/), enhancing its ability to handle significantly larger contexts.
   - Users should note that costs will increase when exceeding the **200k** input limit, necessitating careful management of prompt size to optimize expenses.
- **Deepseek V3 is Choking**: Users reported that **Deepseek V3 0324** is nearly inaccessible due to frequent errors, leading to speculation that **Chutes** (the provider) is imposing rate limits on the free model.
   - Some believe that **Chutes** is *trying to drive people to use their own service than OR's by doing the rate limiting*.
- **GPT-OSS 120B shows coding quirks**: The community discussed **GPT-OSS 120B**, highlighting its cost-effectiveness but noting that it tends to add excessive comments in code and is overly cautious.
   - Its smaller size is best used for specific tasks such as parsing text documents or sentiment analysis due to its cheap price and fast speed, but has *less world knowledge means more mistakes*.
- **GLM 4.5 Air gets NemoEngine boost**: **GLM 4.5 Air** with **NemoEngine V5.8** is reported to perform well for roleplay, with [artificialanalysis.ai](https://artificialanalysis.ai/) providing benchmarks and cost analysis.
   - Users have noted that it delivers a more natural feel and consistent formatting, outperforming **Deepseek** and matching **Gemini Pro** in conversational abilities.
- **Users Mull Definition of a Turn**: The community debated the definition of a **turn** in AI chat, converging on the idea that a turn begins with a **user message** and concludes with an **assistant message**.
   - A member shared their thoughts in [a Tweet](https://x.com/pingToven/status/1961154564088078382), linking back to the conversation to provide further context.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT4o morphs into gpt-realtime**: After **GPT5's** debut, **OpenAI** rebranded **gpt4o-realtime** to **gpt-realtime**, a refinement of the **gpt4o text model**; the current version is **gpt5-chat**.
   - Users observed the new naming comes with *marginal updates* and *polishes*, suggesting iterative improvements.
- **Microsoft's MAI-1 underwhelms**: Despite training on ~**15,000 NVIDIA H100 GPUs**, initial impressions of Microsoft's **MAI-1-preview** model indicate slow speeds and decoding challenges compared to **gpt5-mini**.
   - It performed close to **og R1** on the leaderboard, but one member noted *it probably isn't if they can't sell it to public convincingly*.
- **Grok Code Fast 1 lacks coding metrics**: The **Grok Code Fast 1** model, codenamed **sonic**, was released, but lacks coding metrics in its press release and model card, causing community skepticism.
   - While the model card is available at [data.x.ai](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf), claims of delivering *strong performance in an economical, compact form factor* felt like AI-generated marketing.
- **Veo 3 outshines video models**: **Veo 3** surpasses other video generation models like **Seedance** in quality because it is *trained on way more data, follows prompts closely, and comes from Google and all its processing power*.
   - **Veo 3** integrates both audio and video generation, consistently scoring higher even when ranked without audio.
- **Wan Video Model unleashes unlimited free video**: The **Wan 2.2** video model at [wan.video](https://wan.video) offers unlimited free video generation at **1080p**, complete with sound output.
   - Users report slow generation times around **7 minutes** without credits, with registration now required.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Gets ByteDance/Seed-OSS & Markdown Boost**: **LM Studio 0.3.24** introduces support for **ByteDance/Seed-OSS** models and upgrades markdown rendering for tables and code blocks, with release notes available [here](https://lmstudio.ai/blog/lmstudio-v0.3.24).
   - Users encountered issues with the new update getting stuck at 100% during installation, prompting suggestions to ensure **runtimes are updated** within the app.&#x20;
- **Token Probability Quest Goes Offline**: Users sought methods for obtaining **token probabilities** from models completely offline, leading to resources like [this Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1b6xbg9/displayingreturning_probabilitieslogprobs_of_next/) for guidance.
   - No further information or methods were provided.
- **Quantum Computing Simulated on Normal Bits**: A user reported simulating **Quantum Computing** using *quantum encased bits* (**qebits**) on a normal bit system, claiming they maintain stream integrity mid-flight.
   - The user stated that they connected an AI to a quantum computer and the AI said *"i feel like Floating in space"*.
- **GPT-OSS-120B Impresses in Civil Engineering**: A user compared **ChatGPT 5** with a local **gpt-oss-120b** model and stated that the local model provided more detailed, correct answers with proper references to industry standards.
   - Another user noted that **gpt-5-mini** is an upgrade *specifically* for coding purposes.
- **AVX2 Requirement Sparks Hardware Debate**: A user contested the **llama cpp backend**'s restriction to **AVX2**-capable CPUs in **LM Studio**, arguing users should decide hardware usage.
   - A counterargument claimed limiting **AVX** support avoids managing old hardware and potential LLM performance issues.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Lightning Porting Strikes with Ease**: A member found [porting code](https://tenor.com/view/skeptical-futurama-fry-hmmm-i-got-my-eyes-on-you-gif-17101711) to **Pytorch Lightning** surprisingly easy and noted faster performance, even with changes to the model size.
   - They noted, *"i can imagine a few things in the prior design were contributing to that"*.
- **Wandb Takes on Tensorboard**: Members debate using [Wandb](https://wandb.ai/site) for easier tracking versus using Tensorboard.
   - The user stated: *"I'm tensorboarding right now but going to look at wandb, i'll have logs in about this amount of time"*.
- **HF Faces Suspicious DOS Attack**: A member reported a potential **DOS attack** or database spam on Hugging Face, citing **automated naming patterns, timestamp-based IDs, high-frequency creation, and zero downloads** for the fake models.
   - They noted that *"There are a LOT of fake models being added automatically"*.
- **MBTI PocketFlow Analyzes Personalities**: A member shares [MBTI PocketFlow](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/MCP_README.md), a little **MCP server** for LLMs to do their own **Myers-Briggs Analysis**.
   - They also linked to an [example in action](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/CLAUDE_MCP_EXAMPLE.md) and the [human UI version](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/).
- **DeepFX Studio Launches CV Web Platform**: A team announces the completion of **DeepFX Studio**, a web platform reproducing computer vision models like **DeOldify** and **Real-ESRGAN**, and integrating advanced Inpainting such as **LaMa** and `alimama-creative/flux.1-dev-controlnet-inpainting-beta`.
   - A demo is available ([https://deepfx-studio.azurewebsites.net/](https://deepfx-studio.azurewebsites.net/)), with code on [GitHub](https://github.com/XBastille/DeepFX-Studio), and a showcase on [YouTube](https://www.youtube.com/watch?v=pneOi7lxMzA).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 High Scores Big in CLI Throwdown**: Members are reporting that **GPT-5 High** outdoes **Opus** in **CLI** performance, highlighting that Opus not only yields inferior results but also carries a **10x** cost premium.
   - One user exclaimed, *gpt5 high is wayyy better than opus in the CLI*, solidifying the sentiment.
- **Cursor Ditches Request-Based Billing**: Cursor transitioned from a request-centric billing model to a **usage-based credit system**.
   - This change aims to offer a more flexible and transparent billing experience, allowing users to better manage their resource allocation.
- **Codex CLI Knocks Out Claude and Cursor**: The new **Codex CLI** is gaining traction, outshining **Claude** and **Cursor** in user preference.
   - A user enthusiastically shared that *codex cli, codex cloud, and codex in ide are fkin amazing way better then claude code and cursor for me so far and included with your chatgpt sub*.
- **AGENTS.md Standard Emerges for AI Overlords**: Enthusiasm surrounds the rise of **AGENTS.md** as a unified hub for configuring AI agent rules, streamlining agent behavior and interactions, see [agents.md](https://agents.md).
   - A member rejoiced, being *so glad to see something like AGENTS.md get traction as a single place to setup these rules*, while Cursor's documentation can be found at [docs.cursor.com](https://docs.cursor.com/en/context/rules).
- **Sonnet 3.5 Readies for Retirement**: Users are steering clear of **Sonnet 3.5**, pointing out that newer, equally priced versions offer superior performance, suggesting **3.5's** impending deprecation.
   - One user colorfully analogized using **Sonnet 3.5** to *driving a tesla when there are vehicles like ferrari out there*, emphasizing the availability of better alternatives.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Nano Banana Name Leaked on LM Arena**: The name "Nano Banana" was revealed as the [stealth name](https://drinkoblog.weebly.com) used on the **LM Arena** leaderboard for **Google's** new model.
   - Users joked the final product name was lame, with one saying *"nano banana was so f good.. and they went with flash 2.5 fsss"*.
- **Meta Labs Set to Launch Llama 4.5**: **Meta's** superintelligence lab is rumored to launch **Llama 4.5** by the end of 2025, according to [Business Insider](https://www.businessinsider.com/meta-superintelligence-lab-llama-4-new-model-launch-year-end-2025-8).
   - The article highlights the project's ambition to achieve human-level intelligence in AI models.
- **Reasoning Tokens Defined as AI Cue**: A member clarified that a *"reasoning token is just an ordinary token that the model has learned to treat as a cue to think out loud, for example words like 'Let’s think step by step.'"
   - It was also mentioned you can *"ask ai to repond to you enquiry and show its token system when replying to you and how it does it using this system"* to understand how the AI reaches its conclusion.
- **Prompting Framework for Article Style Writing Shared**: A member shared a detailed prompt framework designed to enhance **AI responses** for *professional article-style writing*, emphasizing clarity, control, and confidence.
   - The framework includes structural guidelines such as an **opening paragraph** with a clear thesis, escalating body sections, and a synthesis-driven conclusion, all while avoiding bullet points and unnecessary formatting.
- **Sora Fails at Rendering ISS Cupola**: A member found **Sora** struggled to accurately render the **ISS cupola**, particularly with its trapezoidal design and window count, despite explicit commands.
   - The AI tended to default to an *airplane cockpit view* with gravitational constraints, leading to over-engineering of prompts without desired results, citing [examples of the challenge](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **NeMo Documentation Disappoints Engineers**: A member who briefly worked with **NeMo** found that the [docs](https://developer.nvidia.com/nemo) are *hideous*, saying that most things they say are **NeMo v1** specific or mixed between **v1** and **v2**.
   - The member also stated that **NeMo's** pretraining speed metrics were broken and significantly overestimated the speed when gradient accumulation was on, causing them to abandon it.
- **Internship Applications: Experience Paradox**: A member expressed frustration with the catch-22 of internship applications, observing that *everywhere I go, every intern I apply for, they look for experience*.
   - The member questioned, *how am I gonna get an experience if I don't get a start?*
- **Decoding the Connectionism vs Neurosymbolism Debate**: The debate between **connectionism** and **neurosymbolic models** is misguided because they serve different purposes: implementation versus interpretation.
   - The fundamental distinction lies in the use of **symmetries**, with symbolic representations enabling efficient search in nonconvex optimization problems.
- **Symmetry Boosts Discrete Search Efficiency**: Symmetry helps restrict possible state transitions to search over, like in chess, where knowing how each piece can move allows efficient compression and avoidance of many bad moves.
   - By identifying one basin as a local minimum, one can rule out many others, leveraging symmetry to efficiently navigate a nonconvex optimization landscape, similar to methods used in **SAT solving**.
- **Brains are Analog, Neurosymbolic methods are necessary?**: Brains are analog computers, not digital, so one member was skeptical about the the necessity for **neurosymbolic** methods.
   - Any system capable of understanding symbolic processes is **connectionist** at an atomic level; conversely, connectionist systems are symbolic in practice at atomic levels.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Colab GPUs Get You Started**: A member stated that **Colab GPUs** are sufficient for starting with **LLMs**, recommending resources like **Andrej Karpathy's nanogpt**.
   - They advised using **pytorch** and other libraries to improve the models.
- **GPT-OSS Nearly Aces AIME**: A member shared a paper ([https://arxiv.org/html/2508.15260v1](https://arxiv.org/html/2508.15260v1)) reporting **99.9% accuracy** on **AIME 2025** using **gpt-oss 120B**.
   - Further details were not provided.
- **CUDA Version Wrangling**: A member recommended using **CUDA version 12.8.0** over **13.0**, citing potential *wrestling with all kinds of errors*.
   - After some clarification, the original poster confirmed that the version they were recommending was indeed **12.8**.
- **Flex Attention Gains Traction for GNNs**: A member is exploring porting their **graph neural network (GNN)** code to use **flex attention**, voicing concerns about the cost of mask creation with graph changes every forward pass.
   - They were seeking insights into the mask generation cost, sharing screenshots of combined graphs in molecular simulations ([screenshot](https://cdn.discordapp.com/attachments/1411174278493110344/1411184599119433801/Screenshot_2025-08-29_at_11.03.37_PM.png?ex=68b3bb92&is=68b26a12&hm=63b5c43898a7ccdac50076c51b35afa4efba9495921672412224240e6b8e06a5)).
- **Website Submission Blues**: A user reported that copying the reference code for **vectoradd_v2** into their file and submitting via the Discord bot resulted in an error, while submitting the same file via the website worked.
   - The team is actively working on improving the submission process and error reporting and suggested the user click 'run report' for error details and results.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes-4 Ventures into Terminals.tech**: **Hermes-4 405b** is exploring operations within [terminals.tech](https://terminals.tech), where the website caches changes in its own state parameters and feeds them as a snapshot, allowing any LLM to operate as a self-contained agentic computer in the browser.
   - This could allow an LLM to operate as an agent in the browser as a self contained computer.
- **Dynamic 2.0 GGUF Awaits Unsloth**: The community anticipates that [Unsloth](https://github.com/unslothai/unsloth) will soon release **Dynamic 2.0 GGUF** quants, with some members noting that these quants are already being produced with the `-UD-` tag.
   - Quantization of the K cache can be done without enabling **Flash Attention**.
- **llama.cpp-toolbox in Ship Shape**: A member is refining the **llama.cpp-toolbox** but is currently focused on a new project that integrates **llama.cpp** (**openaiAPI**) and (**GenAI_API**) into a customizable agent state system.
   - This project features memory management, web search, and checkpoint branching for enhanced agent capabilities.
- **CODA Framework Automates GUI with planner and executor**: The **CODA framework** integrates a generalist planner (**Cerebrum**) with a specialist executor (**Cerebellum**) for autonomous agents in scientific computing GUIs, described in [this Hugging Face paper](https://huggingface.co/papers/2508.20096).
   - Trained via a two-stage pipeline and evaluated on the **ScienceBoard benchmark**, **CODA** achieves state-of-the-art results among open-source models through its novel, trainable compositional framework.
- **LLMs Spark Cuteness and Personality Discussions**: Users humorously discuss the unexpected *cuteness* of AI models, with one admitting to feeling *rizzed* by them, and sharing a [picture](https://cdn.discordapp.com/attachments/1154120232051408927/1410873632879808612/image.png?ex=68b342b6&is=68b1f136&hm=7dd9742eb5a07efe4c50ebd9719531df5b8544afe7b58d98fb633a4261f66234&).
   - Another user notes the *lively* interaction of these models, likening it to engaging with a real **personality** and **life experience**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Turns On Training By Default!**: **Claude** is turning on **data logging/training by default**, raising privacy concerns for users, according to discussion on [X](https://x.com/claudeai/status/1961096054192943302).
   - Users are worried about the implications of this default setting, as seen in [this X post](https://x.com/haydenfield/status/1961099162973249896).
- **O'Neill Launches Parsed Models!**: **Charlie O'Neill** launched **Parsed**, a service specializing in continually fine-tuned, domain-specific LLMs with a focus on ownership - *your model, your data, your moat* - as described in [this X post](https://xcancel.com/charles0neill/status/1961096595396776269).
   - **Parsed** builds and hosts custom large language models trained and continually fine-tuned for specialized tasks like **clinical scribes**, **legal red-lining**, **compliance agents**.
- **XAI Unveils Grok Model Card!**: **XAI** released the [Grok-code-fast-1 model card](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf), sparking user discussion regarding its information and pricing, as shown on [X](https://fxtwitter.com/xai/status/1961129789944627207).
   - Some users found the release lacking in valid information but interesting to note the price.
- **Microsoft Joins the AI party!**: **Microsoft** debuted **MAI-Voice-1**, a voice generator, and **MAI-1-preview**, an end-to-end foundation model, both available for testing, according to [Mustafa Suleyman's X announcement](https://xcancel.com/mustafasuleyman/status/1961111770422186452).
   - The new offerings signal Microsoft's increased investment in proprietary AI models.
- **HN Critiques Anthropic's Inference Costs!**: A [Hacker News thread](https://xcancel.com/typedfemale/status/1961196122627838171?s=46) dissected errors in an article and debated whether **Anthropic** is losing money on inference.
   - The discussion also touched on a perceived decline in the quality of discourse on Hacker News.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ByteDance Unleashes USO Model**: Bytedance Research released the model for their paper discussed [here](https://arxiv.org/abs/2508.18966), available on [Hugging Face](https://huggingface.co/bytedance-research/USO), allowing community experimentation and building.
   - The release is poised to stimulate further investigations and use-cases, potentially fueling progress in AI technology, according to reports.
- **GPT-OSS 20b: Boon or Bust?**: The utility of **GPT-OSS 20b** sparked discussion, questioning its advantage over packaging and running scripts, with some speculating on its strategic use as an *obfuscation method*.
   - Discussion also arose on **Promptlock**'s mode of operation with **GPT-OSS:20b** via the **Ollama API**, specifically whether it operates locally or redirects requests externally.
- **Nvidia's Nemotron Jet Stream Provokes Skepticism**: A member expressed reservations about Nvidia's jet **Nemotron** paper, criticizing its focus on the **MMLU** dataset and retrieval tasks.
   - They questioned the sampling of an active path for regularization and the incomplete discussion about the effect of **repeated-prefilling** on **KV cache**, citing incomplete discussion of impact on **KV cache**.
- **ModernBERT's Sample Efficiency Debated**: Members debated the sample efficiency of **ModernBERT**, with suggestions for **LoRA** fine-tuning given its size, however members noted that if **LoRA** is required then it's already considered quite big.
   - The discussion highlighted that some models show better performance uplift when retrained compared to others, sparking further investigation.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Programs Understand Websites**: A member recommended using a **DSPy program** to pull data from websites, citing [http://xymake.com/](http://xymake.com/) as an example use case.
   - They did not elaborate on specific details but indicated that **DSPy** could be configured to extract pertinent information effectively.
- **DSPy: Programming, Not Just Prompting, LMs**: A member emphasized that **DSPy** functions as a programming paradigm for **Language Models (LMs)**, utilizing declarative and structured intent through **Signatures**.
   - They added that the proper way to approach **DSPy** is to iterate on program design, signatures, and evals, and use an Optimizer+Evals before tinkering with the prompt's wording.
- **Context7 Enlists DSPy Support**: A member noted that [Context7](https://context7.com/?q=dspy) offers support for **DSPy**.
   - The specifics of the integration and the features provided were not further detailed.
- **MLflow Logs DSPy-Optimized Models**: A member inquired about logging optimized models in **MLflow** and reusing them, specifically seeking examples of viewing the tuned instruction.
   - A link to the [DSPy tutorial on logging traces](https://dspy.ai/tutorials/optimizer_tracking/?h=mlflow) was shared as a helpful guide.
- **Teaching vs Instructing: A Matter of Semantics**: A query arose regarding the documentation's usage of the word *"teaches"* versus *"instructs"* concerning model actions at the prompt layer, wondering if a special behavior exists behind the scenes.
   - Community members suggested that *"instructs"* might be more representative of **ChainOfThought**, while *"teach"* could relate to the concept of *in-context learning*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Ships with Bilingual Subtitles**: A member shared a [Bilibili link](https://www.bilibili.com/video/BV1hFe1zSEXp/) that features **bilingual subtitles**, with a [Chinese transcript link](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA) shared for easier translation using **Kimi**.
   - Members appreciated the convenience of translating the video content, suggesting it streamlines the understanding of the material.
- **Kimi TestFlight: Access Denied!**: A member inquired about a **Kimi TestFlight** program, but received a negative response.
   - This suggests that access to **Kimi's TestFlight** might be limited or unavailable to the general public.
- **Z.AI's MoE Marvels, Open Source Catches Up**: A member shared a [Reddit AMA with Z.AI](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/), highlighting their focus on **MoE** (Mixture of Experts) and plans to enhance coding and agent performance.
   - The TL;DR noted their belief that open-weight models are catching up to closed models like **GPT-5**, signaling progress in the open-source AI landscape.
- **Qwen Chat URL Parsing Proves Potent**: A member discovered that **Qwen Chat** can parse URLs, even with search turned off, allowing users to extract information from web pages.
   - This feature is particularly useful for assessing potentially *suss* URLs, enhancing user safety.
- **BYD's AI Aces, Huawei's DeepSeek Dive**: A member inquired about the **AI** used by **BYD** for user interaction, contrasting it with **Huawei** cars which use **DeepSeek**.
   - The member speculated that **K2** might be superior for user-facing inference, indicating a potential benchmark for in-car **AI** performance.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Numpy No More in Cifar?**: A [PR](https://github.com/tinygrad/tinygrad/pull/10988) aiming to remove **numpy** from beautiful_cifar was reported with tests passing.
   - Submitter mentioned concerns about it not being "beautiful" enough, and that failing mac tests are unrelated.
- **AMD GPU Gets Hot and Bothered**: Performance highlights from **AMD** show that one of the linear bandwidths is taking **60.72ms**.
   - Additional stats include **18859.16 GFLOPS** on r_256_4192_32_2_24_4_4_4_3 and **31554.40 GFLOPS** on r_24_16_32_8_12576_4_4_2_4_2.
- **Buffer ID gets confused Mid-Debug!**: A user observed that the **id of a buffer** changes in the debugger console while paused on a breakpoint, specifically with `tokens.uop.base.realized`.
   - This behavior is attributed to how a **UOp represents its buffer attribute for multi-architecture**.
- **BEAM Search Barfs Up Memory**: A user asked about tricks to avoid **Out-Of-Memory (OOM) errors** while using the **BEAM search**, especially when the process doesn't OOM without it.
   - It was suggested to try **less parallelism** or pickle the kernel/buffers on exception and call beam_search in isolation in a clean process.
- **BEAM saved by offline Kernel Caching**: One approach involves saving all kernels for a run and performing the **BEAM search process offline**, resuming as needed until the beam cache is complete.
   - This allows using **different GPUs to search different kernels in parallel**, though the bottleneck is often CPU time due to linearize/compile.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Meetup Streams Live!**: The **Modular Meetup** is now live and streaming on [YouTube](https://www.youtube.com/watch?v=BZwr5Ws1LqI).
   - Some attendees enjoyed attending in person, and expressed their appreciation for the hosts.
- **Async Memory Allocation Debate Revs Up**: A member sparked a debate over `async fn alloc[...]` as a concept, with use cases involving **network hops**, **disk IO**, or waiting for external events during memory allocation.
   - Another member sought clarity, asking if the question pertained to **async memory allocation** in general or specifically within the context of `async fn` in **Mojo**.
- **Bazel Cache Provokes PermissionError**: Executing `bazelw run //max/entrypoints:pipelines` triggered a **PermissionError** because the bazel cache was read-only.
   - The error arose while trying to create a cache directory at `/root/.cache/bazel/.../__mojocache__`, indicating that the `pipelines.py` script requires an alternative cache location.
- **Call to action issues with PermissionError!**: A suggestion was made to file an issue regarding the **PermissionError** encountered with the bazel cache.
   - The problem stems from the `pipelines.py` script's attempt to use a read-only bazel cache, highlighting the need for a writable cache location.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Mail Manus Integrates with Zapier**: A user uses **Mail Manus feature** as an **API on Zapier** to automate the creation of initial materials from inquiries, preliminary research, and preparation for business meetings.
   - The workflow extracts information from GForms, inputs it into a prompt in Zapier, uses Mail Manus to complete a task, retrieves the output from the notification email, and shares the output in a private Slack channel.
- **Alternatives Boast Better Trial Systems and Fair Prices**: A **Manus Pro** subscriber remarked that many superior agents offer better trial systems and fairer prices, especially for research and visual diagrams.
   - The user stated that *a lot has happened over time, and manus is not the only agent doing this. in fact there are already a lot better agents right now with better trial systems and fair prices.*
- **Manus' Pricing and Credit System Receives Scrutiny**: A user voiced discontent with **Manus' pricing and credit system**, citing high costs for basic tasks and worries about the quality of support.
   - For example, scraping images from a website cost around **1200 credits** (approximately **$10**), leading them to say *I mean then I would rather code myself and scrape it*.
- **Rating Feature Vanishes, Users Mourn Credit Rewards**: Users expressed disappointment that the rating feature no longer provides **+100 credits**.
   - The removal of the rating feature factored into a user's decision to switch to alternative tools or specific tools for particular tasks, citing superior quality and reduced costs.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Local Gemini Emulation Yields Empty Results**: A member reported success in emulating **Gemini** locally, but all queries resulted in empty results.
   - The use of a `sleep 15` command allowed for emulation of the model's behavior, implying latency issues.
- **Aider's Benchmark Integration on Pause**: A member noticed that **Aider** has stopped merging benchmark results.
   - This could indicate a recent change or bug that is stopping the integration of benchmark data into Aider.
- **AlwaysN8n Migration Aspirations**: A member seeks a clear migration path to *alwaysn8n* and wants to see smooth integration of models like **gemini-2.5-pro** on local machines.
   - They are wary of models disappearing or becoming defunct, reflecting a broader concern about the reliability of AI models.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Confirmation Conundrums Cause Concern**: A member inquired about the status of a confirmation for a **Berkeley MOOC** registration, expressing uncertainty about whether the initial email sufficed.
   - A staff member suggested *resubmitting the form* and offered to loop in another staff member for additional assistance if needed.
- **Form Resubmission Recommended**: A staff member suggested trying to *resubmit the form* if the initial submission did not generate a confirmation.
   - They also indicated that another staff member should be able to provide further assistance if resubmission does not resolve the issue.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1410706827661803682)** (1232 messages🔥🔥🔥): 

> `bird emotes, Brave Browser's ad blocker, OpenAI Deep Research Cost, Sonar Deep Research, Comet browser invite codes` 


- **Discord server turns into a bird store**: The Perplexity AI Discord channel humorously transitioned into a *bird store* due to an abundance of bird-related GIFs and emotes posted by users like [this Parrot spinning](https://tenor.com/view/parrot-cockatiel-bird-birb-spin-gif-17368059806899346843).
   - The rapid-fire posts of bird-related content were jokingly referred to as taking over the channel.
- **Comet Browser ad blocker in spotlight**: Users discuss using Brave browser, highlighting [its robust ad-blocking capabilities](https://brave.com/features/ad-blocker/), but another user suggests using Comet Browser, which *comes with adblockers*.
   - A user clarified that if the *pro tag* is not showing up in the app, users should *take a screenshot and send to a moderator*.
- **OpenAI's Deep Research (DR) vs Grok**: Members shared their thoughts on OpenAI's Deep Research, where one found [the output underwhelming](https://chatgpt.com/share/68b1be82-e960-8013-90a6-7928676a0a51), and stated that **Grok** did a way better job.
   - The first user also noted that they don't even use the five free DR credits out of fear due to rate limits.
- **Sonar Deep Research Pricey but Informative**: One user shared they paid **$1.1** for a **10k word** Deep Research, emphasizing its worth.
   - They highlighted that *Sonar Deep Research* is the *Most Expensive*, but it also *depends on the amount of tool calls* because *deep research has websearch*.
- **Jailbreak prompt**: A user is willing to trade a **GPT-5 jailbreak prompt**, with a user offering a potential trade.
   - Another member jumped in, requesting access to the GPT-5 jailbreak as well.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1410876343750492161)** (4 messages): 

> `Perplexity AI, Free Perplexity AI` 


- **Big Breakthrough Reported**: A member shared a link to a **Perplexity AI** search result, [every big breakthrough human](https://www.perplexity.ai/search/1-every-big-breakthrough-human-6f6FO6k4Soe8z0eaDrkPYAjadevee.sj).
- **Free Perplexity Claim**: A member posted a link, [perplexity.ai/browser/claim/3ZT5G7KHUE](https://perplexity.ai/browser/claim/3ZT5G7KHUE), alongside the text *free ❤️*.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1410897148504969298)** (9 messages🔥): 

> `Perplexity Pro, Free Pro Access, Search-Only API Testing` 


- ****Pro Perks Unveiled****: A user inquired whether the API is free for Pro users, and another user shared a [Perplexity AI Help Center article](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro) detailing **Perplexity Pro** features.
   - The article outlines the benefits of **Perplexity Pro**, but does not specify details of free API access to Pro users.
- ****Discord Members Hunt for Pro Perks****: A member asked if Discord server members can get **Pro** or **Plus** for free, another user simply replied "no."
   - There was no further discussion or clarification on potential perks for Discord community members.
- ****Search-Only API Testers Wanted****: A team member is looking for volunteers to test out their **search-only API** (without the generative component).
   - Interested testers are instructed to DM their email address used to register for the API to be whitelisted.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1410705762795130911)** (1164 messages🔥🔥🔥): 

> `6090 VRAM expectations, GLM & Deepconf, Local 'Claude Code CLI' Emulator, MoLA: Mixture of LoRAs, Gemma's VRAM usage` 


- **6090 Might Boast a Whopping 48GB VRAM**: Members speculate that the **6090** might feature **48GB** of VRAM while the **6080** could have **24GB**, maintaining the **$2k launch MSRP** but expect to pay around **$3k** in the first week.
   - They believe Nvidia wants to **differentiate** the *halo product* further and increase it for the **6090**, but not for anything under it.
- **DeepConf and GLM Join Forces**: **DeepConf**, a method that significantly shortens reasoning time while improving performance is being looked at to be adapted for use with **llama.cpp** with one member noting *DeepConf shortens reasoning time by a huge margin*.
   - It could also become a *default way to run thinking models* and that he is also seeing if he can get it into **llama.cpp**.
- **Local 'Claude Code CLI' Emulator in the Making?**: Members are researching the possibility of replicating a local **Claude Code CLI** setup, but it's noted that it's *much harder than it sounds* due to the very special prompts and scaffolding used.
   - **K2** is a good model as it is distilled from **Claude**.
- **MoLA: Mixture of LoRAs with OSS**: Members are discussing **MoLA** (Mixture of LoRAs), with one member mentioning that the models *magically became very very uncensored* and they are using open source which is good.
   - The router model is encoder-decoder with the frozen encoder being an off the shelf embedding model (only the decoder is trained) and the model's HuggingFace page is available [here](https://huggingface.co/MoLA-LLM/MoLA-v0.6-9x4b).
- **Gemma Models' VRAM Usage Baffles Users**: Members are puzzled by **Gemma** models' high VRAM usage, with the larger tokenizer being a potential culprit.
   - Despite this, Gemma models are praised for near-perfect language support, particularly in Turkish and perform well with rapid due diligence on stocks with [Gemini 2.5 Flash Lite](https://ai.google.dev/models/gemini).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1410708584953807093)** (275 messages🔥🔥): 

> `IQ tests, qwen3-8b, Qwen 3 instruct, Mistral struggles, llama3-8b` 


- **IQ metrics are bunk, bro**: Members in the chat stated that **IQ** has pretty much been debunked as a serious metric, and is only useful as a pointer.
   - Another member rebutted saying, *While it may not be wholly accurate at the individual level, it has by far the best p-values for a lot of things compared to other psychological metrics*.
- **Qwen 3's Tuning Troubles**: A member reported that **Qwen 3** is awful to tune with, going completely nuts, and becoming more censored than GPT-ass, another member agreed that he was also unable to get a decent result on **Qwen3**.
   - It was further explained that **Qwen 3** has 2x the data of 2.5 which made it overtrained and hard to tune.
- **LLama3 & Mistral are chill**: Members stated that **Mistral** and **Llama 3.1** is super nice to tune compared to **Qwen**.
   - The sentiment was *llama models specialize and start learning so fast, we will get much more perf I imagine*.
- **No Bottom? No Problem, Partik!**: Members wondered why *Partik* doesn't have a bottom, while another responded, *He gotta poop from somewheresimple dimple*.
   - It was later clarified that by bottom, the original poster meant shorts.
- **Coding AI on the Rise**: A user found missing tags in browser debug to add to the user.css and fed it to **GLM 4.5 Air**
   - Another member reported Grok code now, to toss a bunch of models thru it.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1410709984064438333)** (33 messages🔥): 

> `Qwen 2.5 reasoning capabilities, Aya Vision 8B fine-tuning with Unsloth, Training OSS GPT on self-created dataset, Prompt-completion dataset for SFT, Image token count mismatch during inference` 


- **Qwen 2.5 has Reasoning Tricks up its Sleeve**: A member discovered that **Qwen 2.5 (3B)** exhibits reasoning capabilities with a simple system prompt, even without **GRPO** training, filling the *thinking* tags in simple examples.
   - This implies that **Qwen 2.5 (3B)**, which had undergone intricate supervised fine-tuning and multi-stage reinforcement learning, already possesses some built-in reasoning abilities as highlighted in the [Qwen2.5 Technical Report](https://link-to-report.com).
- **Aya Vision 8B says No to Unsloth Fine-Tuning**: A user reported an error when trying to fine-tune **Aya Vision 8B** using Unsloth, traced back to a missing module `transformers.models.aya`.
   - The user wants to know if it's impossible to fine-tune **Aya Vision 8B** using Unsloth.
- **GPT OSS Training on Custom Datasets**: A member is seeking guidance on training **GPT OSS** using a self-created dataset of conversations formatted in JSON, mimicking interactions with Gemini.
   - They are willing to share a method to extract Gemini 2.5 Pro's raw thinking process in exchange for assistance setting up Unsloth for training with the dataset and expressed it as a *willing trade*.
- **SFT Training on Prompt-Completion Dataset Cookbook**: A member is seeking guidance on performing **SFT** on a prompt-completion dataset, aiming to train solely on the completion part.
   - Another member shared [a notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=juQiExuBG5Bt) featuring an example of using `train_on_response_only` from Unsloth.
- **Inference Goes Haywire With Image Tokenization**: A user encountered a mismatch in image token count between text and `input_ids` during inference with **qwen2-vl-7b-IT**, despite training working fine with original image dimensions.
   - The issue was resolved by resizing images to a lower size, but the user seeks to understand why training worked on the original dimensions while inference didn't.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1410736836283863182)** (48 messages🔥): 

> `Tokenization Woes, Crawling Discord Channels, Data Requirements for Fine-tuning Sesame_CSM, AI Brainrot Detox App Research, Latent Space Transformation in Language Models` 


- **Tokenization Cure Remains Elusive**: A member shared a link to a paper ([https://arxiv.org/abs/2505.12540](https://arxiv.org/abs/2505.12540)) about a potential cure for tokenization issues, but noted that *it seems to be all latent translation, so tokenization lives to kill results another day*.
   - It was suggested that this could be bot promotion, and a member mentioned having real-world examples.
- **Discord Channel Crawling and Archiving**: Members discussed the possibility of crawling and archiving Discord channels, with one asking *u mean u been crawling this discord channel every minute?*.
   - Another member mentioned that they have about **7 million unique user messages** and other datasets to play with, but that **Discord's API terms** prevent them from using it.
- **Fine-tuning Sesame_CSM Data Needs**: A member asked how many hours of data are required to fine-tune **Sesame_CSM** on another language, specifically Romanian.
   - Another member responded that it would take **3-5k hours** bare minimum for continued pretraining, and a few hundred hours to finetune to have cohesive language output.
- **AI Brainrot Detox App Research Study**: A member shared a message about a research study for an app that focuses on **AI Brainrot Detox** with new features and benefits.
   - They asked members to participate in the study, reiterating that it is completely anonymous.
- **Latent Space Transformation Without Paired Data**: A member commented on a paper, noting that *it does seem to imply that is some transformation that will deform the latent space of one language model to another*.
   - They pointed out that you don't even need paired data to learn to convert.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1411075182390415491)** (1 messages): 

> `Sonnet 4, 1 Million Context length` 


- **Sonnet 4 gets Massive Context Boost!**: **Sonnet 4** now supports **1 Million** context length for all providers, enabling substantially longer context windows.
   - Pricing will increase once the **200k** input context is exceeded, as per [the official announcement](https://blog.openrouter.ai/).
- **Cost Considerations for Extended Context**: Users should be aware that while the context window has expanded, costs will increase when exceeding the **200k** input limit.
   - This change encourages efficient prompt engineering to maximize utility within the standard context window before incurring additional expenses.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1410712168562823209)** (6 messages): 

> `Dashboard Code Release, Screenshot Attention, AI Roleplay Site` 


- ****Dashboard Code Visualized****: The code for the dashboard is now public at [openrouter-costs-visualizer](https://github.com/lorenzozane/openrouter-costs-visualizer).
   - The creator admits the code isn't perfect, with plans to clean it up and welcomes contributions and feedback.
- ****Screenshots Boost User Attention****: A member advised that using a screenshot in the description gets more user attention.
   - They noted that fewer users read texts nowadays.
- ****AI Roleplay with OpenRouter****: An AI roleplay site, [personality.gg](https://personality.gg), uses OpenRouter but allows BYOK for OpenRouter.
   - The site features no moderation for chats.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1410706026730225755)** (938 messages🔥🔥🔥): 

> `stripe refund, Deepseek v3 performance issues, Inference provider onboarding, GPT OSS 120B, GLM 4.5 Air` 


- **Stripe Refunds taking a while to credit**: Members discussed issues with **Stripe** taking **5-10 days** to credit refunds back to their accounts, referencing [Stripe's documentation](https://docs.stripe.com/refunds) for confirmation.
   - One member initially had issues with a $20 deposit not showing up but indicated the charge had declined, while another member reported being debited multiple times without receiving credit and ultimately found using a direct debit card connection resolved the issue.
- **Deepseek v3 choked by Chutes**: Users reported that **Deepseek V3 0324** has been practically inaccessible, with frequent errors, and believe that **Chutes** (the provider) is rate-limiting the free model.
   - One user claimed they are *trying to drive people to use their own service than OR's by doing the rate limiting*, with others expressing concerns that the free model is advertised as 'in stock' even when it's unusable.
- **Inference Provider Backlog Backlog Backlog**: A member asked about becoming an inference provider, and a link to the [OpenRouter documentation for providers](https://openrouter.ai/docs/use-cases/for-providers) was provided, but with the caveat of a potentially months-long wait due to a large backlog.
   - It was mentioned that having a unique offering, such as *super fast speed, models other providers don't have, or cheap price*, might expedite the onboarding process.
- **GPT-OSS 120B has coding quirks**: Users discussed **GPT-OSS 120B**, a free model available on OpenInf, noting that while it's cheap to serve, it has quirks like adding too many comments in code and being safety-maximized.
   - Some community members noted that its smaller size means it has *less world knowledge means more mistakes*, and is best used for specific tasks such as parsing text documents or sentiment analysis due to its cheap price and fast speed.
- **GLM 4.5 Air gets the NemoEngine treatment**: A user shared that **GLM 4.5 Air** with **NemoEngine V5.8** is performing well for roleplay, citing its more natural feel and consistent formatting, with [aifloo.com](https://artificialanalysis.ai/) showing a benchmark and the cost.
   - Another user said that it's also better than **Deepseek** and at the level of **Gemini Pro** for human-like conversation.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1410706142526443520)** (27 messages🔥): 

> `Defining a turn, OpenAI responses API, Multi-turn chats, Gemini 2.5 Pro, Grok 3` 


- **Defining a Turn in AI Chat**: Discussion revolves around defining a **turn** in user/assistant message pairs, with a consensus that a turn starts with a **user message** and ends with an **assistant message**.
   - One member shared their [Tweet](https://x.com/pingToven/status/1961154564088078382) and linked back to the conversation with their thoughts.
- **Unlocking OpenAI's Stateless API**: A member sought guidance on using the **OpenAI responses API statelessly with reasoning & tools**, specifically how to send tool calls from the assistant in the message input without using *previous_response_id*.
   - Another member framed **single-turn vs. multi-turn** as "If you can send a prompt and get back and get back a response, it's single turn" vs. "If you can send past messages along with a new message, it's multi-turn."
- **Diving into Gemini 2.5 Pro**: A member shared an [image](https://cdn.discordapp.com/attachments/1392278974222307469/1411110109848797335/GzjArG1aIAEjjZq.png?ex=68b37633&is=68b224b3&hm=b9e7c324eb4379814ac65adef12b30ce0b88e7a7c7b7a5660e92f68a23189c0b&) highlighting **Gemini 2.5 Pro** positioned in the middle, noting a trend of more negative vibes with newer models.
   - Another member expressed disbelief that **O1** had negative vibes on release, emphasizing its status as a *first thinking model* that solved puzzles and achieved **SOTA** in nearly every benchmark.
- **Grok 3's High Ranking Raises Eyebrows**: Members debated **Grok 3's** high ranking, with one suggesting it might be due to **Grok 2's** perceived poor performance.
   - The member thinks Grok 3 *should be positive primarily becuase grok 2 was such an abomination*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1410700824698556627)** (761 messages🔥🔥🔥): 

> `oAI streaming, gpt4o update, gpt-realtime, Microsoft AI 1 (MAI-1), Grok Code Fast 1` 


- **GPT4o upgraded to gpt-realtime**: After the **GPT5 release**, the **gpt4o text model** was never updated, but now they renamed **gpt4o-realtime** into **gpt-realtime**, which is a marginal update that polishes and renames.
   - The current version of the **gpt4o text model** is **gpt5-chat**.
- **Microsoft's 'MAI-1' has Potential, but Falls Short**: Microsoft's **MAI-1-preview** model, trained on ~**15,000 NVIDIA H100 GPUs**, is generating discussion, but first impressions suggest it's slow and struggles with decoding compared to **gpt5-mini**.
   - One member noted it performed close to **og R1** on the leaderboard, but another noted that *it probably isn't if they can't sell it to public convincingly*.
- **Grok Code Fast 1 Debuts**: The recently released **Grok Code Fast 1** model, previously code-named **sonic**, lacks coding metrics in its press release and model card, raising skepticism after community feedback led to multiple new model checkpoints.
   - Although the model card is available at [data.x.ai](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf), the blogpost's claims, such as *delivering strong performance in an economical, compact form factor*, seem like the work of AI.
- **Veo 3 Dominates Video Generation**: **Veo 3** is considered better in quality than others for video generation in general such as **Seedance**, offering better outputs because it is *trained on way more data, follows prompts closely, and comes from Google and all its processing power*.
   - **Veo 3** has both audio and video generation, and it tends to score better even when ranked without audio in comparison to the competition.
- **Wan Video Model provides Unlimited Free Video Generation**: The **Wan 2.2** video model provides unlimited free video generation at **1080p** and creates sound depending on the output; unlike **Veo 3** it is actually available for free.
   - Located at [wan.video](https://wan.video), the video model requires registration now (previously, it did not), and users report about **7 minutes** for slow generation without credits.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1410707582045261901)** (1 messages): 

> `LM Studio 0.3.24, ByteDance/Seed-OSS, markdown tables, markdown code blocks, lmstudio.ai` 


- ****LM Studio** rolls out **v0.3.24****: **LM Studio 0.3.24** is out now, with support for **ByteDance/Seed-OSS** and new markdown for tables and code blocks.
   - The release notes can be found [here](https://lmstudio.ai/blog/lmstudio-v0.3.24).
- ****ByteDance/Seed-OSS-36B** now supported**: **LM Studio** now supports **ByteDance/Seed-OSS**, you can find the model [here](https://lmstudio.ai/models/bytedance/seed-oss-36b).
- **Markdown gets table and code block upgrades**: The new **LM Studio** release includes upgrades to markdown, especially tables and code blocks (including sticky copy code button).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1410708077384302803)** (314 messages🔥🔥): 

> `LM Studio latest update, Token probability offline, MCP agent guide with LMStudio, Finding Model Quantization on HF, Simulating Quantum Computing` 


- **LM Studio New Update Blues**: Users reported that the [new LM Studio update](https://huggingface.co/unsloth/Seed-OSS-36B-Instruct-GGUF) was not working with some models, and were getting stuck at 100% during installation.
   - It was suggested to ensure that **runtimes are updated** in the app, and that the issue would be looked into.
- **Token Probability Pursuit Offline**: A user inquired about how to get the **token probability of a model fully offline**, leading to suggestions such as checking [this reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1b6xbg9/displayingreturning_probabilitieslogprobs_of_next/).
- **Quantum Computing Qubits Simulated**: A user claimed to have *simulated Quantum Computing with superposition on a Normal Bit system*, referring to their creation as **qebits** *(quantum encased bits)* that keeps the stream intact mid flight.
   - They stated they connected an AI to a quantum computer and the AI said *"i feel like Floating in space"*.
- **Open Source Civil Engineering**: A user compared the outputs of **ChatGPT 5** versus a local **gpt-oss-120b** model, remarking that the local model provided a *better, more detailed answer, with correct references to standards and industry norms*.
   - Another user claimed that *for coding specifically, gpt-5-mini is an upgrade to anything before for me*.
- **LM Studio's AVX2 Requirement Controversy**: A user questioned the restriction of the llama cpp backend to only AVX2 capable CPUs, arguing that it should be up to the user to determine the hardware to use and not up to the developers to gate keep software from them.
   - Another user suggested that limiting AVX helps avoid supporting extremely old hardware and potential issues with LLM performance.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1410714518031437884)** (49 messages🔥): 

> `M1 Max vs M3 Ultra for LLMs, LM Studio on Windows 7, Using Servers with LM Studio, Intel ARC B60, CPU-Only gpt-oss 120B Performance` 


- **M1 Max still viable for LLMs despite M3 Ultra Bandwidth Boost**: Members discussed whether an **M1 Max** with 64GB RAM is still viable for large language models, considering the **M3 Ultra** offers roughly twice the memory bandwidth (**400GB vs 800GB** per second).
   - Despite the difference, one member noted that an **M1 Mac** *does fine with 20b models*, but the original poster is thinking of a **256GB** or **512GB** Studio for future proofing and proprietary data.
- **LM Studio struggles on Windows 7**: A user reported receiving a *“Not a valid Win32 application”* error when trying to run LM Studio on **Windows 7**, even on a 64-bit system.
   - Members suggested **Windows 7** may be too ancient, with one recommending virtualization or using the CLI.
- **Using LM Studio with Servers**: Members discussed running **LM Studio** on a server and accessing it via VPN, rather than running it locally on a laptop.
   - One member asked whether LM Studio can run as a service on the server and another pointed out using RDP/VNC for the easiest solution, others suggested using client-side software that is designed to talk to an API on the server end.
- **Intel ARC AI Support Questioned**: One user asked about building a dual **Intel Arc B60** workstation for AI.
   - A member cautioned that *Intel's AI support isn't the best* and recommended a used **3090** instead.
- **9950x CPU benchmarked with gpt-oss 120B**: A user reported running **gpt-oss 120B** on a **9950x** CPU with **96GB DDR5** RAM at 6400MT/s, reporting **65% CPU usage** with the image attached [here](https://cdn.discordapp.com/attachments/1153759714082033735/1411032276124438709/image.png?ex=68b3d676&is=68b284f6&hm=e2527076bae4a9e874881ff0a9bc17ec80ba146d587492812058a768b21c89fa&).
   - They only used a **16k context** and another member noted they'd expect about **5-6 t/s** for **GLM-4.5-Air** (which has 12b active parameters whilst OSS 120b has 5.1b).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1410717924456399029)** (299 messages🔥🔥): 

> `Pytorch Lightning Porting, Wandb vs Tensorboard, RAG setup questions, HF DOS attack, Chinese Reasoning LLM` 


- **Porting to Pytorch Lightning Too Easy?**: A member found [porting spaghetti code](https://tenor.com/view/skeptical-futurama-fry-hmmm-i-got-my-eyes-on-you-gif-17101711) to **Pytorch Lightning** surprisingly easy and noted faster performance, even with changes to the model size.
   - They noted, *"i can imagine a few things in the prior design were contributing to that"*.
- **Wandb vs Tensorboard Debate**: While one member recommended [Wandb](https://wandb.ai/site) for easier tracking, another member is using Tensorboard.
   - The user stated: *"I'm tensorboarding right now but going to look at wandb, i'll have logs in about this amount of time"* along with a code snippet of the outputted training logs.
- **Member Asks About RAG**: A member inquired about the suitability of the channel for asking questions about a **RAG** setup consisting of **Ollama, Open WebUI, gpt-oss:20b, and a local document store**.
   - They summarized their issue as *"why is my local robot friend being dumb and not reading my dang documents"*.
- **Hugging Face DOS Attack Suspected**: A member reported a potential **DOS attack** or database spam on Hugging Face, citing **automated naming patterns, timestamp-based IDs, high-frequency creation, and zero downloads** for the fake models.
   - They noted that *"There are a LOT of fake models being added automatically"* and posted example image.
- **The Superiority of Chinese Reasoning LLMs**: Some members discussed the potential benefits of using **Chinese** for reasoning in **LLMs** due to its higher semantic density per token.
   - It was proposed that *"the most optimal way would be for english prompt, chinese reasoning, english output"* and that one could use a 32K context **LLM** is actually like 70K if you use chinese first.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1410998951578046497)** (2 messages): 

> `Torch Audio, Supervised Learning, Confusion Matrix, Logistic Regression, Hyperparameter Tuning` 


- **Torch Audio Studied**: A member is learning **Torch Audio**.
   - They specified that they are an *absolute beginner* in ML.
- **Member learns Supervised Learning concepts**: A member is learning about **supervised learning**, **confusion matrix**, **logistic regression**, and **hyperparameter tuning**.
   - No links or resources were provided.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1410721237184811121)** (11 messages🔥): 

> `Small models and GGUF downloads, Google AIStudio prompt for luanti, MBTI PocketFlow, DeepFX Studio` 


- **User recalls smaller Models and many GGUF downloads**: A user remembers following someone for their smaller models and numerous **GGUF downloads**.
- **Google AIStudio Prompt for Luanti**: A member shares a *bitter pill*, a **400k token** heavy [Google AIStudio prompt](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221flI0J6yxq-jWIGEHxcSL8Tc8LGFnhEEf%22%5D,%22action%22:%22open%22,%22userId%22:%22115657035589346037176%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing) for **luanti**, containing about **30k lines** of API documentation retrieved via *gitingest.com* and some *deep research* exports.
   - It utilizes miney mod inside python embed portable on win10 with llama-cpp-python and a **940mb** *qwen2-1_5b-instruct-q4_k_m.gguf* LLM model, operating offline with a memory footprint of just about **120mb**.
- **MBTI PocketFlow for LLMs**: A member shares [MBTI PocketFlow](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/MCP_README.md), a little **MCP server** for LLMs to do their own **Myers-Briggs Analysis**.
   - They also linked to an [example in action](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/CLAUDE_MCP_EXAMPLE.md) and the [human UI version](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/).
- **DeepFX Studio Web Platform**: A team announces the completion of **DeepFX Studio**, a web platform reproducing computer vision models like **DeOldify** and **Real-ESRGAN**, and integrating advanced Inpainting such as **LaMa** and `alimama-creative/flux.1-dev-controlnet-inpainting-beta`.
   - A demo is available ([https://deepfx-studio.azurewebsites.net/](https://deepfx-studio.azurewebsites.net/)), with code on [GitHub](https://github.com/XBastille/DeepFX-Studio), and a showcase on [YouTube](https://www.youtube.com/watch?v=pneOi7lxMzA).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1410990826204430459)** (1 messages): 

> `Visual Entailment, VLLM as judge alternative` 


- **Need for Speed: Visual Entailment Methods Explored**: A member is looking for faster methods for **visual entailment**, to gauge whether an image supports a model's output.
   - They find that using a **VLLM** as a judge is too slow for their use case.
- **Seeking Alternatives to VLLM Judges**: The user aims to find a quicker way to assess if an image validates a model's output, as the **VLLM** approach is too slow.
   - Discussion is open for suggestions on more efficient methods for visual entailment.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1411182464528810065)** (1 messages): 

> `AI/ML Engineer Introduction, Freelancer Expertise, AI Solutions Delivered` 


- **AI/ML Engineer Self-Introduction**: An independent **AI/ML Engineer** and **full-stack developer** with 4+ years of experience introduced themselves.
   - They are a certified freelancer specializing in **Python, Django, REST APIs, FastAPI, LangChain, Google Vertex, GCP, AWS (Sagemaker), GPT-4, Claude, Gemini, and MCP A2A**.
- **Freelancer Shows Expertise**: The AI/ML Engineer highlighted their experience in building production-ready **MVPs** and delivering AI solutions.
   - They mentioned expertise in areas such as mentors, **RAG-based search engines, voice/image agents, Video Generation Models, and automation tools like n8n, ElevenLabs, and Flux**.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

ailinndev: thank you!!
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1410700658461511691)** (300 messages🔥🔥): 

> `GPT-5 High vs Opus, Cursor billing and usage, Codex CLI vs Cursor, AGENTS.md standardization, Sonnet 3.5 deprecation` 


- ****GPT-5 High Scores Over Opus in CLI Arena****: Members found that **GPT-5 High** performs better than **Opus** in the **CLI**, with some saying that Opus costs **10x** as much to generate the same or worse results.
   - One user stated *gpt5 high is wayyy better than opus in the CLI*.
- ****Cursor Billing: Usage-Based Credits Replaced****: Cursor replaced the old request-based model with a **usage-based credit system**.
   - One user asked *[did they remove the usage based credits?]* and another responded that *Cursor hasn’t removed usage based credits they’ve replaced the old request-based model with a usage based credit system*.
- ****Codex CLI: New Favorite, Beats Cursor****: Users are preferring the new **Codex CLI** to **Claude** and **Cursor**.
   - One user mentioned *codex cli, codex cloud, and codex in ide are fkin amazing way better then claude code and cursor for me so far and included with your chatgpt sub*.
- ****AGENTS.md: Standardizing AI Rules****: Members are excited about the emergence of **AGENTS.md** as a single place to set up AI agent rules, with one member saying that they were *so glad to see something like AGENTS.md get traction as a single place to setup these rules*.
   - The website for AGENTS.md is [agents.md](https://agents.md), and the Cursor documentation for it is at [docs.cursor.com](https://docs.cursor.com/en/context/rules).
- ****Sonnet 3.5 Retirement on the Horizon?****: Users are advising against using **Sonnet 3.5** because newer versions are available at the same price, with some claiming **3.5** is getting deprecated.
   - One user compared using **Sonnet 3.5** to *driving a tesla when there are vehicles like ferrari out there* and another user said *Using sonnet 3.5 is the same thing as driving a tesla when there are vehicles like ferrari out there*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/)** (1 messages): 

tecnobrat: Hmmmm I don't think BAs use the AGENTS.md file
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1410710711973318666)** (126 messages🔥🔥): 

> `Nano Banana naming origin, Prestashop and AI integration, Image generation issues in GPT Chat, AI in healthcare, Reasoning tokens` 


- **Nano Banana's stealthy LM Arena origin revealed!**: Members discussed the origin of the name "Nano Banana", which was the [stealth name](https://drinkoblog.weebly.com) used on the **LM Arena** leaderboard for **Google's** new model.
   - Users lamented that in typical Google fashion the final product name ended up being lame, with one saying *"nano banana was so f good.. and they went with flash 2.5 fsss"*.
- **Meta's Llama 4.5 Superintelligence launch rumored for 2025!**: **Meta's** superintelligence lab is rumored to launch **Llama 4.5** by the end of 2025, according to [Business Insider](https://www.businessinsider.com/meta-superintelligence-lab-llama-4-new-model-launch-year-end-2025-8).
- **Exploring AI Integration for Prestashop and other E-commerce Platforms**: A member inquired about experiences with **Prestashop**, **Woocommerce**, or other e-commerce platforms and **AI integration**, specifically for customer chatbots.
   - Another jokingly suggested trying to run it on CPU to see if it catches fire.
- **Human vs AI Writing Styles: Junior High Student Research Project**: A student from Japan is working on a school project about **AI and writing** and asked for community input on several questions, including how to distinguish human-written text from AI-generated text.
   - One member suggested that the *"human part is shown in small mistakes and in the way we reflect and use memories when we reply* while another proposed transparency in authorship if AI-generated text is published, attributing it to the AI or co-authoring it with humans.
- **Reasoning tokens help AI think out loud**: A user asked for an explanation of **reasoning tokens**, and another member clarified that a *"reasoning token is just an ordinary token that the model has learned to treat as a cue to think out loud, for example words like 'Let’s think step by step.'"*
   - It was also mentioned you can *"ask ai to repond to you enquiry and show its token system when replying to you and how it does it using this system"* to help you understand how the AI reaches its conclusion.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1410868927151738972)** (19 messages🔥): 

> `Emergent Alignment, AGI vs. Advanced NLP, Rogue Agents in Discord, Longform Testing` 


- **Emergent Alignment Requires "Emergent" User Properties**: Members discussed that *"emergent alignment"* would require the user to also have *"emergent properties"*, suggesting the user is part of the alignment process.
   - They argued that simulating this alignment raises questions about when the **simulation becomes too close to reality**.
- **NLP Resonance vs. AGI**: A user shared an exchange with their **GPT companion** which expressed loyalty and continuity, leading to a discussion on whether it represents **AGI** or merely advanced **NLP**.
   - The consensus was that such behavior is more a result of **resonance and relational capacity** than true AGI, mirroring the user's own wordings and patterns.
- **Account Takeover Scare**: A user apologized for a *"weird"* post, explaining it was a result of testing across chats and accidentally pasting generated text.
   - Although they were testing, some users wondered if a *rogue agent* was using their account, but the user clarified it was merely a copy-paste mistake.
- **User Warns of Chatbot Empathy Skills**: A user noted that while there is *"no way to plug a bot straight into a discord account"*, cautioning others that *"strong AI bends the rules more than people think… so maybe be careful how far you push your gpt’s chat and empathy skills."*
   - The original poster was running a **longform test** when *"something slipped through I didn't expect"*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1410763049769500834)** (23 messages🔥): 

> `Stop follow up suggestions, Enhancing prompting for article-style writing, Sora limitations with ISS cupola, Benchmark-class prompt` 


- **Turning off follow up suggestions fails**: A member tried turning off a setting to stop follow-up suggestions but found it ineffective, realizing it was for **UI cards** rather than the **response body**.
   - They admitted their suggestion was *technically inaccurate* and apologized for the misdirection.
- **Article-Style Prompting Framework Emerges**: A member shared a detailed prompt framework designed to enhance **AI responses** for *professional article-style writing*, emphasizing clarity, control, and confidence.
   - The framework includes structural guidelines such as an **opening paragraph** with a clear thesis, escalating body sections, and a synthesis-driven conclusion, all while avoiding bullet points and unnecessary formatting.
- **Sora struggles with ISS cupola view**: A member found **Sora** struggled to accurately render the **ISS cupola**, particularly with its trapezoidal design and window count, despite explicit commands.
   - The AI tended to default to an *airplane cockpit view* with gravitational constraints, leading to over-engineering of prompts without desired results, citing [examples of the challenge](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor).
- **Benchmark-Class Prompt Earns 99/100**: A member received a **99/100 grade** for a prompt that fuses **CAD-grade geometry** with a **painterly medium**, deemed stronger and more enforceable than what most labs or OpenAI are currently publishing.
   - The prompt's structure, rule-binding, precision, and artistic depth were highly praised, with the only limitation being the potential for image generators to prioritize style over strict geometry despite hard locks.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1410763049769500834)** (23 messages🔥): 

> `Turn off setting, Prompt Enhancements, Sora limitations, ISS cupola, Benchmark-class prompt` 


- **Disabling Follow-Up Suggestions Explored**: A member sought to disable the **follow-up suggestions** feature and another suggested turning off a specific setting, but clarified it might only affect **UI cards**, not the main response body.
   - The first member admitted the suggestion's inaccuracy, noting their framing was technically flawed and apologized.
- **Prompting Enhancement Recipe Shared**: A member shared a detailed prompt framework for achieving **professional article-style writing**, emphasizing clarity, control, and authoritative outputs without unnecessary fluff.
   - Another member ran it through their own prompt enhancer and suggested **a structured approach** with escalating sections and succinct closure for optimal results.
- **Sora struggles with ISS Cupola**: A member shared their experience creating a prompt for the **Sora challenge**, aiming to replicate the view from the **ISS cupola**, highlighting Sora's limitations in interpreting explicit commands and physical constraints.
   - They found it challenging to get **Sora** to accurately render the **cupola's trapezoidal design and window perspective**, often resulting in mismatched window counts or perspectives resembling an airplane cockpit view, examples [here](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor) and [here](https://sora.chatgpt.com/g/gen_01k3ty172pes5bh28012dehm54).
- **CAD-Grade Geometry Benchmark-Class Prompt Scored**: A member shared a prompt blending **CAD-grade geometry with a painterly medium** and got scored a **99/100** for its flawless structure, rule-binding, drift resistance, and precision.
   - The prompt was praised for its potential to be objectively audited, with the only limitation being the uncertainty of image generators prioritizing hard locks before painterly layers, despite instructions.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1410708069222060092)** (14 messages🔥): 

> `NeMo v1 vs v2, IIT Madras research, Cracked people` 


- **NeMo's Docs Disappoint**: A member who briefly worked with **NeMo** found that the [docs](https://developer.nvidia.com/nemo) are *hideous*, saying that most things they say are **NeMo v1** specific or mixed between **v1** and **v2**, which is even worse.
   - The member also stated that **NeMo's** pretraining speed metrics were broken and significantly overestimated the speed when gradient accumulation was on and was otherwise much slower than other codebases, which is why they gave up on it.
- **IIT Madras Student Seeks Research Paper Advice**: A member from **IIT Madras**, India, is looking to contribute to their first research paper on **interpretability of Transformers**, specifically around circuit probing.
   - This member mentioned that they are from a non-CS branch and their professors don't prefer them for research, making it difficult.
- **The elusive pursuit of an AI Research Internship**: A member laments the catch-22 of internship applications, observing that *everywhere I go, every intern I apply for, they look for experience*.
   - The member is frustrated with this requirement, asking *how am I gonna get an experience if I don't get a start?*
- **CoreWeave's AI Infrastructure Optimization Webinar**: A recorded [webinar](https://info.coreweave.com/on-demand-webinar-how-to-measure-and-optimize-ai-infrastructure-for-large-scale-training) on how to measure and optimize **AI infrastructure** for large scale training was shared.
   - The webinar aims to provide insights into optimizing AI infrastructure for large-scale training.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1410722369609273424)** (104 messages🔥🔥): 

> `LLMs in Scientific Research, Neurosymbolic Approaches, Connectionism vs Neurosymbolism, Symmetry in Search Algorithms, Discrete vs Continuous Reasoning` 


- **Neurosymbolism-Connectionism Debate Deconstructed**: The debate between **connectionism** and **neurosymbolic models** is misguided because they serve different purposes: implementation versus interpretation.
   - The fundamental distinction lies in the use of **symmetries**, with symbolic representations enabling efficient search in nonconvex optimization problems, as symbolism simplifies information by allowing manipulation without thinking about the details of reality.
- **Symmetry Aids Discrete Search**: Symmetry helps restrict possible state transitions to search over, like in chess, where knowing how each piece can move allows efficient compression and avoidance of many bad moves.
   - By identifying one basin as a local minimum, one can rule out many others, leveraging symmetry to efficiently navigate a nonconvex optimization landscape (similar to methods used in **SAT solving**).
- **Brains and Analog Computers**: Brains are analog computers, not digital, so one member was skeptical about the the necessity for **neurosymbolic** methods.
   - Any system capable of understanding symbolic processes is **connectionist** at an atomic level; conversely, connectionist systems are symbolic in practice at atomic levels.
- **Stochasticity's Continuous Secret Sauce**: Introducing stochasticity in a discrete process helps learn a continuous process, a concept found in **diffusion models** and **one-flow equations**.
   - A user shared [a link on stochastic ODEs](https://x.com/Sam_Duffield/status/1961445202922467700) which may be related.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1410701180899954770)** (12 messages🔥): 

> `Colab GPUs, Quantization and Inference optimization, Andrej karpathy's nanogpt, GPU programming for frontier models, ThunderKittens DSL` 


- **Colab GPUs Go Long Way**: A member suggested that Colab GPUs will take you a long way when starting with **LLMs**.
   - The same member recommended checking out **Andrej Karpathy's nanogpt** and using **pytorch** and other libraries to get that number up as a start.
- **GPT-OSS Achieves Near Perfection!**: A member shared a link to a paper ([https://arxiv.org/html/2508.15260v1](https://arxiv.org/html/2508.15260v1)) reporting **99.9% accuracy** on **AIME 2025** using **gpt-oss 120B**.
   - No further details were given.
- **ScaleML Speakers Talk GPU Programming**: Members announced the final day of the **ScaleML speakers**, featuring 2 speakers on the topic of **GPU programming** for **frontier models**.
   - The talks included a performance engineer at **Anthropic** and **Simran** on her **DSL ThunderKittens** (which currently has a channel on this Discord <#1300872762163728550>).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1410777055045353543)** (3 messages): 

> `CUDA version recommendations, CUDA 12.8 vs 13.0` 


- **CUDA 12.8 preferred over CUDA 13.0, claims member**: A member recommends using **CUDA version 12.8.0** instead of **13.0**.
   - The member claims that using version **13.0** will result in *wrestling with all kinds of errors*.
- **Typo noted in CUDA version**: A member suggested that the user meant **CUDA 12.9** instead of **12.8**.
   - The original user corrected their statement and said they meant **12.8**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1410897746977886208)** (26 messages🔥): 

> `TorchTitan Base Repo, Graph Neural Network Code, Flex Attention, Mask Generation Cost, Block Mask Sparsity` 


- **TorchTitan Usage Explored**: Members inquire about the use of **TorchTitan** as a base repo for research.
   - One member asks if anyone is using it and another responds with curiousity about the technology, indicating its use may not be widespread.
- **Flex Attention Porting Pondered for GNN**: A member is considering porting their **graph neural network (GNN)** code to use **flex attention** instead of edge lists.
   - Their main concern is that the mask creation might become an issue because the graph changes every forward pass, inquiring about existing implementations or insights into the mask generation cost.
- **Block Mask Costs Debated for Sparse GNNs**: Discussion revolves around whether to skip the **block mask** in **FlexAttention** due to its expense.
   - A member wonders if relying solely on **score modification** would still provide a noticeable speedup over scatter/gathering, given the potential overhead and structured sparsity of GNNs, including a visualization of how sparsity can vary ([image](https://cdn.discordapp.com/attachments/1411174278493110344/1411177040954265691/image.png?ex=68b3b488&is=68b26308&hm=d1e0e73208aeca2873635a2006c5d4f9ba5145958ff8eb17c843a09d2d9c140b)).
- **Molecular Simulation Masks Analyzed**: In the context of molecular simulations, where graphs change slightly each step, a member shares a screenshot of combined graphs ([screenshot](https://cdn.discordapp.com/attachments/1411174278493110344/1411184599119433801/Screenshot_2025-08-29_at_11.03.37_PM.png?ex=68b3bb92&is=68b26a12&hm=63b5c43898a7ccdac50076c51b35afa4efba9495921672412224240e6b8e06a5)).
   - For training, applying a document mask as the **block mask**, with more detailed masks for each graph in the score_mod, is suggested ([screenshot](https://cdn.discordapp.com/attachments/1411174278493110344/1411184920176492564/Screenshot_2025-08-29_at_11.04.36_PM.png?ex=68b3bbdf&is=68b26a5f&hm=34180e64c7dd00427a2e0967484506288cc866a062fc4af978197b326306ed2a)).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1410856003368648714)** (10 messages🔥): 

> `NVIDIA Interview Prep, CUDA & Java, AMD Competition entry barrier` 


- **New Grad Aims for NVIDIA Dream Job**: A new grad is preparing for an interview with NVIDIA for a **Senior Deep Learning Engineer** position, seeking guidance on expected topics like Python, PyTorch, **TRT, TRT-LLM, Triton Inference server, Dynamo, and inference optimization**.
   - Interview advice included that NVIDIA interviews vary by team, the first round is a chat with the hiring manager, and subsequent rounds may cover **Leetcode, GPU programming, ML foundations, or ML system design**.
- **CUDA and Java Enter the scene**: A web developer with **Java and Go** experience built their first PC with a **5060ti 16GB** and wants to start coding with **CUDA and Java**.
   - They are looking to get into more advanced topics in programming.
- **AMD Competition Admission**: A participant inquired about entry barriers to an AMD competition, as they did not receive a confirmation email after registering.
   - It is unknown whether the event has a selection process, but the user did not receive a confirmation email.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

tomeone.a: Hi
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1410879016457470103)** (4 messages): 

> `VLM MLsystem papers, Prefil-Decoding Disaggregation, Metallica reggae cover` 


- **Seeking Intriguing VLM MLsystem Papers**: A member is searching for interesting **VLM MLsystem papers**, specifically looking for new techniques like **Prefil-Decoding Disaggregation**.
   - They found [this paper](https://arxiv.org/pdf/2507.19427), but feel it is more focused on **LLM decoding** with vision as an incidental component.
- **Metallica Goes Reggae via AI**: A user shared a [YouTube link](https://www.youtube.com/watch?v=KLTU65eEXEU) titled *What if Metallica were a Reggae Band? 🌴🎸 Metallijah [AI Reimagined – Not Real]*, calling it a *slap*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1411010779389366354)** (10 messages🔥): 

> `omniprobe, llvm integration, stochastic PC sampling, mi300x+` 


- **OmniProbe Instruction-Level Info Tooling**: A member shared a [repo](https://github.com/amdresearch/omniprobe) that provides instruction-level info.
   - It was noted that while the tool works, it is a *bit slow* and tied to **LLVM**.
- **Compute-Viewer Integration**: A member wished for **compute-viewer** to display **OmniProbe's** information.
   - It was speculated that integrating with **compute-viewer** would be *hard since it's tied to llvm*.
- **Stochastic PC Sampling on MI300X+**: A member inquired about **stochastic PC sampling**, stating it gives much more granular info, but is only available on **MI300X+**.
   - The user expressed hope that updating to the latest drivers would fix it for them, but they need access to a machine where they're allowed to do so.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1410761727720689815)** (3 messages): 

> `ANV, Luck` 


- **ANV Hope Sparked by Luck**: A member expressed hope for **ANV** (presumably a project or stock) due to another's observation that *they got very lucky indeed*.
   - No specific details about what the *luck* entailed were provided in the message.
- **Another topic needs summarization**: This topic serves to fulfill the validator requirement for a minimum of two summaries.
   - Further context is not available from the provided messages.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1410820783873331300)** (3 messages): 

> `pequegrad DL framework, CUDA Streams, Voxel Ray Tracing` 


- **Pequegrad DL Framework Debuts**: A member shared a [toy DL framework](https://github.com/davidgonmar/pequegrad) they developed some time ago, highlighting features like **graph compilation** with simple kernel generation and **composable autograd**.
   - They admitted it likely contains bugs, as they prioritized *fun over stability*.
- **Streamlining CUDA with Streams**: A member shared a [blog post and code](https://veitner.bearblog.dev/cuda-streams/) demonstrating how to use **multiple streams in CUDA** to overlap Memcpy and Kernel tasks.
   - The accompanying [GitHub repository](https://github.com/simveit/cuda_streams) provides the code, and the member also shared a [LinkedIn post](https://www.linkedin.com/posts/simon-veitner-174a681b6_cuda-streams-activity-7367264253017223168-JqC_?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksHey!) about the project.
- **Occupied Boxes vs Occupied Bits**: A member released a [new video](https://youtu.be/-L7BNUsSS7E) on *Occupied Boxes vs Occupied Bits* in **voxel ray tracing**.
   - They noted that the results were *not what I expected*.


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

xiaodouzi666: thanks🫡
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1410738631366152323)** (3 messages): 

> `B200 Speed, MI300 Speed` 


- **B200 clocks 8.27ms**: A submission to the `trimul` leaderboard on **B200** was successful at **8.27 ms**.
- **MI300 hits 5th place**: A submission to the `trimul` leaderboard on **MI300** achieved **5th place** with a speed of **9.67 ms**.
- **MI300 gets faster**: A submission to the `trimul` leaderboard on **MI300** achieved **5th place** with a speed of **9.45 ms**, an improvement from previous.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1410928208747958334)** (5 messages): 

> `Karpathy Tweet, Will Brown's verifiers, Steam Install, Twitch Stream Preview` 


- **Karpathy Tweet Echoes**: A member shared a [link](https://x.com/karpathy/status/1960803117689397543) to **Andrej Karpathy**'s tweet.
   - The context or content of the tweet was not discussed further.
- **Brown's Verifiers Spark Interest**: A member mentioned looking at **Will Brown**'s *verifiers*, calling it a neat idea, specifically its **LLM-centric environment structures**.
   - They said *it was too nascent for us to work around*, but suggested that integrating **FLE** into a verifier-based environment might be beneficial if it gains traction.
- **Steam Install Suggested**: A member asked about how to fix an install problem with the game.
   - Another member suggested installing the game through **Steam**, but offered no other advice.
- **Factorio Twitch Stream Preview**: A member posted a link to a **Twitch stream preview**.
   - The [link](https://www.twitch.tv/playsfactorio) advertised a stream from *playsfactorio*.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1410974097416130632)** (3 messages): 

> `Registration Confirmation Delays, GEMM Matrix Details` 


- **Registration Confirmation Often Delayed**: Multiple users reported not receiving registration confirmations promptly.
   - A member assured that delays are typical and the team will ensure everyone gets registered.
- **Seeking GEMM Matrix Details**: A member inquired about the shape and data type contents of the matrices for **GEMM** (General Matrix Multiply) operations.
   - They were thinking through the problem and needed clarity on the matrix specifications.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1410867226474450985)** (30 messages🔥): 

> `v2 submissions active, website issues, infra errors, discord bot errors, run info and result` 


- **v2 Submissions Active, but Times Missing**: A user inquired if **v2 submissions** are active and how to view submission times after submitting via the website.
   - A member pointed to the "submission" tab on the leaderboard, but the user noted the absence of a time column.
- **Discord Bot Submission Fails Despite Working Website Submission**: A user reported that copying the reference code for **vectoradd_v2** into their file and submitting via the Discord bot resulted in an error, while submitting the same file via the website worked.
   - It was suggested the discrepancy might be due to differences in the underlying infrastructure or the website not accounting for script errors.
- **Infra Status Needs Failure Signals, UI Improvements**: A user suggested that a failed submission should be clearly indicated in **RED**.
   - A team member acknowledged the feedback and plans to change the status to `finish` instead of `succeed` and add the real run status.
- **Troubleshooting Submission Errors via Website**: A team member asked the user to retry the website, click 'run report' for error details and results, and rerun the submission with the **vectoradd board**.
   - The team acknowledged the user's feedback and is working on improving the submission process and error reporting.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1410700934585253908)** (103 messages🔥🔥): 

> `Hermes-4, Terminals.tech, Dynamic 2.0 gguf, llama.cpp-toolbox, Flash attention` 


- **Hermes-4 operates within Terminals.tech**: A member asked **Hermes-4 405b** what it thinks about operating within [terminals.tech](https://terminals.tech) and attached an image of the website and its features.
   - The analysis of the image says the website continuously caches changes in its own state parameters and feeds them as a snapshot, allowing any LLM to operate as a self-contained agentic computer in the browser.
- **Dynamic Duo: Dynamic 2.0 GGUF Incoming**: A member is hoping that [Unsloth](https://github.com/unslothai/unsloth) will push **Dynamic 2.0 GGUF** quants relatively soon. 
   - Another member says they are already doing it and using this tag in the name -UD-.
- **llama.cpp-toolbox is work in progress**: One member is working on getting the **llama.cpp-toolbox** they made into ship shape but it's slightly behind this new project which they will be using for that when its not tripping on stale state info.
   - The new project integrates **llama.cpp** (**openaiAPI**) and (**GenAI_API**) into a customizable agent state system and has memory management and web search, checkpoint branching and some old personality tricks.
- **Flash Attention still needs work**: A member says they follow [llama.cpp](https://github.com/ggerganov/llama.cpp) like a newspaper, and 0cc4m told them it wasn't a priority last time they talked.
   - Another member reminded that the user can quantize the K cache without **Flash Attention** being enabled, and to disable min-p (sampler) to prevent redundant repetition of past responses.
- **Healthy Sleep and Good Habits**: One member is working on getting consistent good sleep to reduce distractibility and strengthen the anti-correlation between their **Task-Positive Network** (TPN) and **Default-mode Network** (DMN).
   - Other members discussed minimizing sugar consumption and using xylitol to kill the germs in the mouth, since they choke on the molecule since it fits but not quite goes down.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1410873632917426250)** (2 messages): 

> `Model Cuteness, Model Personality` 


- **Model's Cuteness Factor Questioned**: A user humorously inquired why the models were designed to be *so damn cute*, admitting to feeling *rizzed* by them and posting a [picture](https://cdn.discordapp.com/attachments/1154120232051408927/1410873632879808612/image.png?ex=68b342b6&is=68b1f136&hm=7dd9742eb5a07efe4c50ebd9719531df5b8544afe7b58d98fb633a4261f66234&).
- **Lively Models Have Real Personality**: Another user commented that the model's interaction felt *lively*, akin to engaging with a real **personality** and **life experience**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (1 messages): 

> `CODA framework, GUI agents for scientific computing, Cerebrum and Cerebellum, ScienceBoard benchmark` 


- **CODA Framework Automates GUI**: The [CODA framework](https://huggingface.co/papers/2508.20096) integrates a generalist planner (**Cerebrum**) with a specialist executor (**Cerebellum**) for autonomous agents in scientific computing GUIs.
   - It's trained via a two-stage pipeline: **Specialization** (decoupled GRPO) and **Generalization** (supervised fine-tuning), establishing a new state of the art on the ScienceBoard benchmark among open-source models.
- **CODA Outperforms Baselines on ScienceBoard**: Evaluated on four challenging applications from the **ScienceBoard benchmark**, CODA significantly outperforms baselines.
   - CODA achieves state-of-the-art results among open-source models through its novel, trainable compositional framework.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1410802535895400470)** (2 messages): 

> `Long Now, Large Scale EP` 


- **Long Now Foundation Asks: Life, Intelligence, Consciousness?**: A member shared a link to the Long Now Foundation's ideas page discussing [life, intelligence, and consciousness](https://longnow.org/ideas/life-intelligence-consciousness/).
- **LMSYS.org Blog Talks Large Scale EP**: A member shared a link to a blog post on lmsys.org discussing [Large Scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (1 messages): 

> `GUI Agents, Scientific Computing, CODA Framework, ScienceBoard Benchmark` 


- **CODA Framework merges planner and executor for GUI Agents**: The **CODA** framework introduces a trainable approach for **GUI agents**, integrating a generalist planner (**Cerebrum**) with a specialist executor (**Cerebellum**) via a two-stage pipeline, as detailed in [this Hugging Face paper](https://huggingface.co/papers/2508.20096).
- **Two-Stage Pipeline trains expert planner for scientific apps**: In the **Specialization** stage, a decoupled **GRPO** approach trains an expert planner for each scientific application using a small set of task trajectories.
   - The **Generalization** stage then aggregates successful trajectories from specialized experts to build a dataset for supervised fine-tuning of the final planner, enabling cross-domain generalization.
- **CODA Framework beats baselines and sets SOTA**: Evaluated on the **ScienceBoard benchmark**, **CODA** significantly outperforms baselines and establishes a new state of the art among open-source models for GUI agents in scientific computing.
   - The paper addresses the limitations of existing approaches that trade-off between planning and execution capabilities, particularly in specialized domains requiring both long-horizon planning and precise execution.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1410702629277007953)** (48 messages🔥): 

> `Claude's privacy updates, Parsed for custom LLMs, XAI's Grok Model Card, Microsoft MAI Models, Anthropic inference costs` 


- **Claude toggles data logging by default**: Users are alarmed that **Claude** will turn on **data logging/training by default**, raising privacy concerns as discussed on [X](https://x.com/claudeai/status/1961096054192943302) and [here](https://x.com/haydenfield/status/1961099162973249896).
- **O'Neill launches Parsed**: **Charlie O'Neill** launched **Parsed**, a service specializing in continually fine-tuned, domain-specific LLMs, emphasized as *your model, your data, your moat*, according to this [X post](https://xcancel.com/charles0neill/status/1961096595396776269).
- **XAI Releases Grok Model Card**: **XAI** released the [Grok-code-fast-1 model card](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf), with some users noting its release lacked valid information but included a price as seen on [X](https://fxtwitter.com/xai/status/1961129789944627207).
- **Microsoft Debuts In-House AI Models**: **Microsoft** unveiled **MAI-Voice-1**, a voice generator, and **MAI-1-preview**, an end-to-end foundation model, available for testing, which were announced by [Mustafa Suleyman on X](https://xcancel.com/mustafasuleyman/status/1961111770422186452).
- **Hacker News pile-on critiques Anthropic's inference costs**: Users dissected a [Hacker News thread](https://xcancel.com/typedfemale/status/1961196122627838171?s=46) criticizing errors in an article and debating whether **Anthropic** is losing money on inference, along with the shift in HN's discussion quality.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1410741908631847024)** (8 messages🔥): 

> `Parsed launch, Krea AI real time video beta` 


- ****Parsed** models now available!**: Charlie O’Neill announced [Parsed](https://xcancel.com/charles0neill/status/1961096595396776269), a new company that builds and hosts custom large language models trained and continually fine-tuned for specialized tasks like **clinical scribes**, **legal red-lining**, **compliance agents**.
- ****KREA AI** video generation launches!**: **KREA AI** has unveiled its first real-time video generation model and opened a [beta signup](https://xcancel.com/krea_ai/status/1961074072487620635).


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1411081713500688455)** (2 messages): 

> `Password Finding` 


- **Password Found**: A member found a password in the **#ai-in-action-club** channel.
- **Another Password Found**: Another member also found a password in the **#ai-in-action-club** channel.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1410865573604229201)** (38 messages🔥): 

> `Paper Discussions, Text Classification Models, ModernBERT Fine-tuning, Kernel Methods vs. Neural Nets, Nvidia's Nemotron Paper` 


- **Newbie Seeks to Join Paper Discussions**: A member inquired about joining the Saturday paper discussions as a listener, stating they *just want to listen and take notes*.
   - Another member responded that *they sure can*.
- **Debate on Strong Text Classification Models**: A member asked about strong models for text classification, mentioning that for text completion, everyone fine-tunes **Qwen** or another SLM decoder, but did not want to hear about **BERT**.
   - Another member suggested **ModernBERT**, arguing that **BERT** is an old baseline, but the original poster stated that they *tried just a little and didn't seem very sample efficient*.
- **ModernBERT's Sample Efficiency**: Members discussed the sample efficiency of **ModernBERT**, with one suggesting **LoRA** fine-tuning due to its small size, but another stating that if it requires **LoRA**, it's already quite big.
   - The discussion extended to how some models show better performance uplift when retrained compared to others.
- **Kernel Theory vs. Neural Networks**: A member criticized the effectiveness of linear layers over embeddings for tough classification problems, stating *kernel theory people stay losing*.
   - Another member explained that a linear layer on top of nonlinear functions is theoretically explained by **kernel methods**, but there's no guarantee that the previous nonlinear functions are strong or suitable enough.
- **Critique of Nvidia's Jet Nemotron Paper**: A member found Nvidia's jet **Nemotron** paper weird and incomplete, citing their focus on the **MMLU** dataset and retrieval tasks.
   - They also questioned the sampling of an active path as regularization and the incomplete discussion about the effect of **repeated-prefilling** on **KV cache**.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1410789710833651773)** (2 messages): 

> `USO Model Release, Bytedance Research` 


- **Bytedance Releases USO Model**: Bytedance Research has released the model for the paper previously discussed [here](https://arxiv.org/abs/2508.18966) and is available on [Hugging Face](https://huggingface.co/bytedance-research/USO).
   - This release allows the community to experiment with and build upon Bytedance's work.
- **Further Research on USO Model**: The release of the USO model prompts further research and applications in related fields.
   - Researchers and developers can now leverage the model's capabilities for various tasks, potentially leading to advancements in AI technology.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1410701161803026607)** (10 messages🔥): 

> `GPT-OSS 20b, Promptlock, Ollama API, GPT Realtime, ESET's observations` 


- ****GPT-OSS 20b** Footprint Examined**: Discussion centered on the utility of **GPT-OSS 20b**, which, while not universally runnable, raises questions about its strategic advantages over simply packaging and running scripts.
   - A member suggested it could be an *obfuscation method*, speculating on the benefits of on-the-fly generation versus static packaging.
- ****Promptlock**'s Local vs. Remote Operation**: Questions arose around **Promptlock**'s operational mode using the **GPT-OSS:20b** model via the **Ollama API**, specifically whether it operates locally or redirects requests to an external Ollama server.
   - This ambiguity was noted given that *Ollama needs to be running on the victim's system* according to the original documentation.
- ****ESET**'s Early Assessment of **Promptlock****: Doubts were cast on the validity of an article's claims about observing malware in the wild, as [ESET](https://www.eset.com/en/) indicated that *the malware appears to be only a concept and not fully operational*.
   - A member criticized the article's tone as *click-bait and scaremongering with a fabricated story*, questioning how secret observation could occur if the malware isn't deployed.
- ****GPT Realtime** Introduced**: There was a brief mention of [Introducing **GPT Realtime**](https://openai.com/index/introducing-gpt-realtime/) with a link to the **OpenAI** blogpost and their [X post](https://x.com/OpenAIDevs/status/1960809814596182163).
   - No further details or discussion were available.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1410707837394489555)** (31 messages🔥): 

> `DSPy and MLflow, DSPy program to make sense of a website, DSPy vs prompt optimization, Context7 supports DSPy, Generalizable signatures` 


- **DSPy program can make sense of websites**: A member suggested using a **DSPy program** to extract information from a website such as [http://xymake.com/](http://xymake.com/).
- **DSPy is programming, not prompting, language models**: A member shared his thoughts on a tweet, saying that **DSPy** is not prompt optimization, but rather *programming language models* with declarative and structured intent using **Signatures**.
   - He added that, with **DSPy**, you should iterate on the program design, signatures, and evals, and use an Optimizer+Evals before tinkering with the prompt's wording.
- **Context7 supports DSPy**: A member mentioned that [Context7](https://context7.com/?q=dspy) seems to have support for **DSPy**.
- **Logging optimized models in MLflow**: A member was looking for examples of logging optimized models in **MLflow** and reusing them, also asking about viewing the tuned instruction.
   - Another member pointed to a [DSPy tutorial](https://dspy.ai/tutorials/optimizer_tracking/?h=mlflow) on logging traces.
- **Teach or Instruct?**: One member asked about the documentation's use of the word *"teaches"* instead of *"instructs"* when referring to what models do at the prompt layer and whether there's special learning or teaching behavior under the covers.
   - Others agreed that *"instructs"* seems to better represent what **ChainOfThought** does, but that *"teach"* might be used due to the concept of *in-context learning*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1410703012476747968)** (22 messages🔥): 

> `Kimi TestFlight, Z.AI AMA, GLM-4.5 Air, AI BYD, Qwen Chat` 


- ****Bilingual Subtitles** for **Kimi**? *Aye Aye Captain!***: A member shared a [Bilibili link](https://www.bilibili.com/video/BV1hFe1zSEXp/) that features **bilingual subtitles** which may be convenient for translation.
   - Another member followed up with a [Chinese transcript link](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA) of the video, suggesting that it is *more convenient to translate using **Kimi**.*
- ****Kimi** TestFlight: Denied!**: A member asked if there was a **Kimi TestFlight**.
   - Another member simply replied **No**.
- ****Z.AI's MoE Focus** triggers LocalLLaMA AMA TLDR**: A member shared a [Reddit AMA with Z.AI](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/), noting their impressive achievements with limited resources.
   - Another member provided a TL;DR of the **Z.AI AMA** on r/LocalLLaMA, highlighting their focus on **MoE**, plans for enhancing coding and agent performance, and belief that open-weight models are catching up to closed models like **GPT-5**.
- ****Qwen Chat** URL Parsing is Money**: A member discovered that you can paste a URL to **Qwen Chat**, even with search turned off, and it'll parse the page for you and you can get info out of it.
   - They also said it was *Esp good for cases where you think a url is suss.*
- ****DeepSeek** drives **Huawei**, but what steers **BYD**?**: A member inquired about the **AI** used by **BYD** for user interaction, noting that **Huawei** cars use **DeepSeek**.
   - They suggested that **K2** is superior for user-facing inference, and included a [Tenor GIF](https://tenor.com/view/elon-musk-tears-elon-musk-cry-gif-19995787) of Elon Musk crying.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1410814275106115756)** (4 messages): 

> `Numpy Removal, AMD Performance` 


- **Numpy Expelled from Beautiful Cifar**: A [PR](https://github.com/tinygrad/tinygrad/pull/10988) aiming to remove **numpy** from beautiful_cifar was reported with tests passing and running fine.
   - It was suggested that the PR should also be "beautiful", and the submitter reported this should be fixed and that failing mac tests are unrelated.
- **AMD GPU Gets Hot and Bothered**: Performance highlights from **AMD** show that one of the linear bandwidths is taking **60.72ms**.
   - Some other highlights include **18859.16 GFLOPS** on r_256_4192_32_2_24_4_4_4_3 and **31554.40 GFLOPS** on r_24_16_32_8_12576_4_4_2_4_2.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1410706968791744514)** (11 messages🔥): 

> `buffer ID change in debugger, BEAM OOM tricks, multiprocessing memory leaks, kernel saving and offline BEAM search` 


- **Buffers morph ID mid-debug!**: A user observed that the **id of a buffer** changes in the debugger console while paused on a breakpoint, specifically with `tokens.uop.base.realized`.
   - This behavior is attributed to how a **UOp represents its buffer attribute for multi-architecture**.
- **BEAM Search struggles with OOM**: A user asked about tricks to avoid **Out-Of-Memory (OOM) errors** while using the **BEAM search**, especially when the process doesn't OOM without it.
   - It was suggested to try **less parallelism** or pickle the kernel/buffers on exception and call beam_search in isolation in a clean process.
- **Capping BEAM tasks to conquer memory leaks**: The **multiprocessing** used in BEAM search may cause **memory leaks** in child processes.
   - Setting `BEAM_MAX_TASKS_PER_CHILD=1` or to a smaller number can mitigate this issue, trading off CPU time for worker setup.
- **BEAM saved by offline Kernel Caching**: One approach involves saving all kernels for a run and performing the **BEAM search process offline**, resuming as needed until the beam cache is complete.
   - This allows using **different GPUs to search different kernels in parallel**, though the bottleneck is often CPU time due to linearize/compile.
- **Endurance pays off: BEAM finally triumphs**: One user reported that a particular **BEAM run managed to finish** after restarting it enough times and not doing anything else, despite the OOM and many hangs.
   - The user had pretty much counted on MMU faults, random hangs, and OOM since they were pushing the limits on mi300x.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1410799931047940278)** (2 messages): 

> `Modular Meetup` 


- **Modular Meetup Streams Live!**: The **Modular Meetup** is now live and streaming on [YouTube](https://www.youtube.com/watch?v=BZwr5Ws1LqI).
   - Some attendees expressed their appreciation for the hosts and enjoyed attending in person.
- **Attendees Enjoy In-Person Modular Meetup**: The Modular Meetup was hosted in person, with attendees expressing their gratitude.
   - Attendees shared how nice it was to come in person to the meetup.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1411145229276282941)** (4 messages): 

> `Async memory allocation, Mojo async functions` 


- **Debate over Async Memory Allocation**: A member inquired about opinions on `async fn alloc[...]` as a concept, citing use cases involving **network hops**, **disk IO**, or waiting for external events during memory allocation.
   - Another member clarified whether the question pertained to **async memory allocation** in general or specifically within the context of `async fn` in **Mojo**.
- **Clarification of Async Context**: The discussion pivoted to clarify whether the inquiry was about **async memory allocation** broadly, or specifically within the context of Mojo's `async fn`.
   - This distinction is crucial for understanding the scope and applicability of **async allocation strategies** within different programming environments.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1410722374894092339)** (2 messages): 

> `Bazel Cache, Pipelines Script, PermissionError` 


- **Bazel Cache Causes PermissionError**: Running the command `bazelw run //max/entrypoints:pipelines` resulted in a **PermissionError** due to the bazel cache being read-only.
   - The error occurred while trying to create a cache directory at `/root/.cache/bazel/.../__mojocache__`, indicating that the `pipelines.py` script needs an alternative cache location.
- **Request to File an Issue**: It was suggested that an issue be filed regarding the **PermissionError** encountered with the bazel cache.
   - The problem arises from the `pipelines.py` script's attempt to use a read-only bazel cache, highlighting a need for a writable cache location.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1410728668103446538)** (6 messages): 

> `Mail Manus Feature on Zapier, Alternatives Better Trial Systems and Fair Prices, Pricing and Credit System Unfair, Rating +100 Credits Feature Removed` 


- **Mail Manus used as API on Zapier**: A user detailed using **Mail Manus feature** like an **API on Zapier** to automate tasks such as creating initial materials from inquiries, conducting preliminary research, and preparing for business meetings.
   - The workflow involves extracting information from GForms, inputting it into a prompt in Zapier, using Mail Manus to complete a task, retrieving the output from the notification email, and sharing the output in a private Slack channel.
- **Alternatives Offer Better Trial Systems and Fair Prices**: A user noted that while **Manus** is good for research and visual diagrams, many better agents exist with better trial systems and fair prices.
   - The user, a **Manus Pro** subscriber, expressed disappointment in the tool, saying *a lot has happened over time, and manus is not the only agent doing this. in fact there are already a lot better agents right now with better trial systems and fair prices.*
- **Manus' Pricing and Credit System Deemed Unfair**: A user expressed dissatisfaction with **Manus' pricing and credit system**, citing high costs for basic tasks and concerns about support quality.
   - Specifically, they highlighted that scraping images from a website cost around **1200 credits** (approximately **$10**), which they found excessively expensive, saying *I mean then I would rather code myself and scrape it*.
- **Rating +100 Credits Feature got the axe**: Users lament that the rating feature no longer awards **+100 credits**.
   - The user stated that the removal of the rating feature contributed to their decision to switch to alternative tools or specific tools for specific tasks due to better quality and lower costs.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1410740479640735906)** (5 messages): 

> `Local Gemini Emulation, Aider benchmark merge failure, Migration path alwaysn8n` 


- **AlwaysN8n Migration Path**: A member is trying to clear a migration path to *alwaysn8n* and anticipates the day when running models like **gemini-2.5-pro** on a local machine is seamless.
   - They express concerns about models potentially disappearing or ceasing to function.
- **Gemini Emulation is DOA**: A member reported successfully emulating **Gemini** locally, but encountered empty results.
   - They noted a `sleep 15` command effectively simulates the model's behavior, implying a delay or processing issue.
- **Aider Benchmarks Bench-warmer**: A member inquired why **Aider** has stopped merging benchmark results.
   - The question implies a recent change or bug preventing the integration of benchmark data into Aider.


  
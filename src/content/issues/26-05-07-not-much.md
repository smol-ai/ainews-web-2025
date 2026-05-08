---
id: MjAyNS0x
title: not much happened today
date: '2026-05-07T05:44:39.731046Z'
description: >-
  **OpenAI** released **GPT-Realtime-2**, a voice model with **GPT-5-class
  reasoning**, tool use, interruption handling, and extended context windows up
  to **128K tokens**, achieving top scores on **Big Bench Audio** and
  **Conversational Dynamics** benchmarks. They also launched a **Chrome plugin
  for Codex** enabling browser control and multitasking, and introduced
  **GPT-5.5 with Trusted Access for Cyber** for secure defensive workflows and
  red teaming. **Anthropic** introduced **Natural Language Autoencoders** for
  interpreting model activations as human-readable text, aiding interpretability
  and debugging, while **Goodfire** proposed a neural geometry research agenda
  focusing on **manifolds** as primitives for neural network behavior. Anthropic
  also announced **The Anthropic Institute** to advance AI safety and economic
  resilience research.
companies:
  - openai
  - anthropic
  - goodfireai
  - scale-ai
models:
  - gpt-realtime-2
  - gpt-5.5
  - codex
topics:
  - voice-models
  - streaming-translation
  - transcription
  - benchmarking
  - context-windows
  - browser-automation
  - cybersecurity
  - interpretability
  - neural-geometry
  - manifolds
  - ai-safety
  - rlhf
people:
  - micahcarroll
  - milesbrundage
  - ryanpgreenblatt
---


**a quiet day.**

> AI News for 5/6/2026-5/7/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**OpenAI Voice, Codex, and Cybersecurity Releases**

- **GPT-Realtime-2 and new audio stack**: OpenAI released **GPT-Realtime-2** in the API, described as its most capable voice model with **GPT-5-class reasoning**, tool use, interruption handling, and longer conversations; it ships alongside **GPT-Realtime-Translate** for streaming translation across **70+ input languages / 13 output languages** and **GPT-Realtime-Whisper** for low-latency streaming transcription [@OpenAI](https://x.com/OpenAI/status/2052438194625593804). OpenAI says ChatGPT voice updates are still forthcoming [@OpenAI](https://x.com/OpenAI/status/2052438197695877316). Artificial Analysis reports GPT-Realtime-2 reaches **96.6% on Big Bench Audio**, leads its Conversational Dynamics benchmark at **96.1%**, expands context from **32K to 128K**, and keeps audio pricing unchanged [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2052486470469140777). Scale AI also placed GPT-Realtime-2 at #1 on its Audio MultiChallenge S2S leaderboard, with instruction retention rising from **36.7% to 70.8% APR** versus GPT-Realtime-1.5 [@ScaleAILabs](https://x.com/ScaleAILabs/status/2052451341071683732).
- **Codex gets browser control**: OpenAI shipped a **Chrome plugin for Codex** on macOS and Windows, letting Codex operate across background tabs without taking over the user’s browser; it can use plugins where possible, Chrome for logged-in sites, and combine tools for workflows like debugging browser flows, checking dashboards, research, or CRM updates [@OpenAI](https://x.com/OpenAI/status/2052480800004956323). The dev team emphasized browser DevTools, multi-tab parallelism, and web-app testing as key use cases [@OpenAIDevs](https://x.com/OpenAIDevs/status/2052481136971125158).
- **Cyber-specific GPT-5.5 access**: OpenAI announced **GPT-5.5 with Trusted Access for Cyber** for defensive workflows and a limited-preview **GPT-5.5-Cyber** for authorized red teaming, pentesting, and validation under enhanced verification and account controls [@cryps1s](https://x.com/cryps1s/status/2052508963409998283). Separately, Micah Carroll said OpenAI found instances of accidental **CoT grading** in previous RL runs after building a scanner, but did not find clear evidence those instances degraded CoT monitorability [@MicahCarroll](https://x.com/MicahCarroll/status/2052451995467018427).

**Anthropic, Interpretability, and AI Safety Tooling**

- **Natural Language Autoencoders**: Anthropic introduced **Natural Language Autoencoders**, a method for translating model activations into human-readable text so researchers can inspect “thought-like” internal representations rather than only sparse features or supervised probes [@AnthropicAI](https://x.com/AnthropicAI/status/2052435436157452769). Miles Brundage/ML-powered commentary framed NLAs as complementary to probing and dictionary learning, noting they revealed planning behavior and helped identify training-pipeline translation bugs; open-model NLAs are available on Neuronpedia [@mlpowered](https://x.com/mlpowered/status/2052446867037020402). Ryan Greenblatt cautioned that early tests did not recover “internal CoT” on single-forward-pass math cases, suggesting limitations or missing activation locations [@RyanPGreenblatt](https://x.com/RyanPGreenblatt/status/2052458229624672549).
- **Goodfire’s neural geometry agenda**: Goodfire launched a research series arguing neural networks “think in shapes,” with **manifolds** as a core primitive for interpreting and controlling behavior [@GoodfireAI](https://x.com/GoodfireAI/status/2052420446910644616). The thread contrasts manifold-level structure with SAE-style feature shattering, includes examples where steering along a learned manifold preserves coherent world-model behavior, and teases work on unsupervised manifold discovery and in-context geometry [@GoodfireAI](https://x.com/GoodfireAI/status/2052420594193650167). Goodfire also linked the agenda to scientific discovery, citing reverse-engineering of a scientific foundation model to uncover biomarker structure in a curved manifold [@GoodfireAI](https://x.com/GoodfireAI/status/2052468622103085107).
- **Anthropic safety infrastructure**: Anthropic shared the research agenda for **The Anthropic Institute**, focused on economic diffusion, threats/resilience, AI systems in the wild, and **AI-driven R&D** with human visibility and control [@AnthropicAI](https://x.com/AnthropicAI/status/2052385812881228218). It also moved **Petri**, its open-source interactive behavioral-evals tool, to Meridian Labs as an independent project [@AnthropicAI](https://x.com/AnthropicAI/status/2052494460966019137), and opened its security bug bounty publicly on HackerOne [@AnthropicAI](https://x.com/AnthropicAI/status/2052466175540629965).

**Agents, RL Environments, and Coding Workflows**

- **Prime Intellect Lab and Ramp Fast Ask**: Prime Intellect opened **Lab** out of beta as a full stack for building RL environments/evals, evaluating, post-training, deploying, and serving agents [@PrimeIntellect](https://x.com/PrimeIntellect/status/2052225145725698102). Ramp Labs used Prime Intellect to train **Fast Ask**, a small RL-trained subagent for spreadsheet QA that reportedly scores **+4% exact-match over Opus** at **Haiku-level latency** [@RampLabs](https://x.com/RampLabs/status/2052448843099254956); Prime says it outperformed Opus 4.6 while running faster and cheaper [@PrimeIntellect](https://x.com/PrimeIntellect/status/2052465182014840987).
- **Hermes Agent momentum**: Nous/Teknium shipped **Hermes Agent v0.13.0** with multi-agent orchestration via Kanban, enforced goal completion with `/goal`, disk-usage optimizations, custom LLM providers, and custom gateway channels [@Teknium](https://x.com/Teknium/status/2052495174404874714). Earlier updates added agent-free cron jobs via Hermes Gateway for programmatic recurring tasks [@Teknium](https://x.com/Teknium/status/2052219963591762194), blank-slate profiles with `--no-skills` [@Teknium](https://x.com/Teknium/status/2052351650279645590), and Lightpanda as a machine-native browser backend with Chrome fallback [@lightpanda_io](https://x.com/lightpanda_io/status/2052369346928758861).
- **Cursor orchestration and PR workflows**: Cursor introduced `/orchestrate`, a skill that recursively spawns planner, worker, and verifier agents via the Cursor SDK; internally it reportedly cut skill token use by **20%** while improving evals and reduced backend cold-start time by **80%** [@cursor_ai](https://x.com/cursor_ai/status/2052432778743210127). Cursor 3 also added an integrated PR review experience with diffs, commits, comments, review status, a file tree, and skill quick-action pills [@cursor_ai](https://x.com/cursor_ai/status/2052489387305488609).
- **Agent infra patterns**: LangGraph is adding **delta channels**, storing checkpoint history as diffs to control storage bloat for long-context agents [@sydneyrunkle](https://x.com/sydneyrunkle/status/2052344141963555312). Deep Agents added sandbox backends for provider-agnostic isolated execution across Daytona, Modal, Runloop, and LangSmith, with an **auth proxy** pattern to keep credentials out of prompt-injectable sandboxes [@sydneyrunkle](https://x.com/sydneyrunkle/status/2052459962169966752).

**Models, Benchmarks, and Inference Systems**

- **xAI, Zhipu, Zyphra, DeepSeek ecosystem**: xAI made **Image Generation Quality Mode** available on the xAI API after powering more than **300M images** in Grok, claiming better realism, text rendering, and creative control [@xai](https://x.com/xai/status/2052193877675983031). Zhipu published the **GLM-5V-Turbo technical report**, highlighting CogViT dual-teacher distillation, multimodal multi-token prediction, multimodal coding/tool use, and RL across 30+ task categories [@Zai_org](https://x.com/Zai_org/status/2052426777654387168). Zyphra’s **ZAYA1-8B** was described as AMD-trained, using under **1B active parameters**, large-scale RL, and a test-time method called **Markovian RSA** [@kimmonismus](https://x.com/kimmonismus/status/2052346978240205249). Antirez also released **DS4**, a specialized inference engine for **DeepSeek v4 Flash** built on llama.cpp/GGML lineage [@antirez](https://x.com/antirez/status/2052405820235678175).
- **Google model and API updates**: Google AI Studio announced **Gemini 3.1 Flash-Lite** as its most cost-efficient model for high-volume agentic tasks, translation, and simple data processing [@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2052453828272812310). Google also evolved the **Gemini Interactions API** from role-based `user/model` messages to typed **steps** such as `user_input`, `thought`, `function_call`, `tool_call`, and `model_output`, targeting richer multi-step agent workflows [@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2052487438967140700). Gemma 4’s MTP/speculative decoding was reported to deliver up to **3× faster** on-device inference [@googlegemma](https://x.com/googlegemma/status/2052468624657654194), with independent vLLM tests showing large throughput gains and **129 tok/s** on simple generation on an RTX Pro 6000 [@bnjmn_marie](https://x.com/bnjmn_marie/status/2052286398707687650).
- **Sequence models and coding evals**: Aviv Bick and Albert Gu introduced **Raven**, a fixed-state sequence model that learns which finite memory slots to update, aiming to fix persistence failures in SSMs and sliding-window attention and outperform prior linear models at **16× training sequence length** [@avivbick](https://x.com/avivbick/status/2052438903924396377), [@_albertgu](https://x.com/_albertgu/status/2052442144879862003). Scale released the **SWE Atlas Refactoring** leaderboard, testing whether agents can restructure code without regressions; **Claude Opus 4.7 with Claude Code** leads [@ScaleAILabs](https://x.com/ScaleAILabs/status/2052434456510878021). Arena’s longitudinal analysis says open models have largely closed the Text Arena gap, with the proprietary lead now around **+30 Arena points**, though expert prompts remain harder [@arena](https://x.com/arena/status/2052455463573426452).

**AI Infrastructure, Health, Robotics, and Applied Products**

- **Compute and infrastructure**: Anthropic’s SpaceX/xAI compute deal remained a major theme: Dario Amodei called the SpaceX partnership “visionary engineering + Claude” [@Mononofu](https://x.com/Mononofu/status/2052212359536496961), while Simon Willison highlighted that Anthropic reportedly gets **Colossus 1**, xAI keeps the larger **Colossus 2**, and Colossus 1 has environmental controversy [@simonw](https://x.com/simonw/status/2052436629365948920). Lambda closed a **$1B senior secured credit facility** to expand AI factories [@LambdaAPI](https://x.com/LambdaAPI/status/2052373882963972496), AMD promoted **MI350P PCIe** with **144GB HBM3E** and up to **2299 TFLOPS MXFP4** [@AMD](https://x.com/AMD/status/2052373018400219648), and Ai2 brought new NSF OMAI compute online with **NVIDIA Blackwell Ultra** systems from a **$152M** NSF/NVIDIA investment [@allen_ai](https://x.com/allen_ai/status/2052403904139169940).
- **Google Health and medical AI**: Google is turning Fitbit into the **Google Health** app on May 26, combining Fitbit tracking with Google services and a Gemini-powered **Google Health Coach** [@googlehealth](https://x.com/googlehealth/status/2052392762255761701). Google says Health Premium will be included in AI Pro and Ultra plans [@shimritby](https://x.com/shimritby/status/2052439569136767291), and announced **Fitbit Air**, a screenless wearable with up to one-week battery and $99.99 preorder pricing [@Google](https://x.com/Google/status/2052501704155775481). Separately, Glass Health launched an ambient scribing API at **$0.85/hour** for transcription plus token-priced note generation [@GlassHealthHQ](https://x.com/GlassHealthHQ/status/2052385429010121130).
- **Robotics and local agents**: Perplexity released **Personal Computer** in a new Mac app, letting agents operate across local files, native Mac apps, web, and Perplexity servers, including remote initiation from iPhone and always-on Mac mini setups [@perplexity_ai](https://x.com/perplexity_ai/status/2052445405754040816). NVIDIA Robotics highlighted Hugging Face’s Reachy Mini “agentic robotics app store” and **Isaac GR00T N** integration with LeRobot workflows [@NVIDIARobotics](https://x.com/NVIDIARobotics/status/2052446013949149649). EO-1 is now available through the standard LeRobot policy interface for robot-control training/eval/deploy workflows [@SongHaomin92651](https://x.com/SongHaomin92651/status/2052360599703867415).

**Top tweets by engagement**

- **OpenAI GPT-Realtime-2 API launch** — **11.7K** engagement [@OpenAI](https://x.com/OpenAI/status/2052438194625593804)  
- **Anthropic Natural Language Autoencoders** — **10.1K** engagement [@AnthropicAI](https://x.com/AnthropicAI/status/2052435436157452769)  
- **Claude Mythos helped Firefox fix more security bugs in April than prior 15 months** — **9.7K** engagement [@alexalbert__](https://x.com/alexalbert__/status/2052468573516513762)  
- **OpenAI Codex Chrome plugin** — **7.7K** engagement [@OpenAI](https://x.com/OpenAI/status/2052480800004956323)  
- **Goodfire neural geometry research agenda** — **5.1K** engagement [@GoodfireAI](https://x.com/GoodfireAI/status/2052420446910644616)  
- **Sam Altman on voice as a high-context AI interface** — **5.0K** engagement [@sama](https://x.com/sama/status/2052462271667028211)  
- **xAI Image Generation Quality Mode API** — **4.5K** engagement [@xai](https://x.com/xai/status/2052193877675983031)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.6 27B Local Inference and Quantization

  - **[2.5x faster inference with Qwen 3.6 27B using MTP - Finally a viable option for local agentic coding - 262k context on 48GB - Fixed chat template - Drop-in OpenAI and Anthropic API endpoints](https://www.reddit.com/r/LocalLLaMA/comments/1t57xuu/25x_faster_inference_with_qwen_36_27b_using_mtp/)** (Activity: 1798): **A recent **llama.cpp** MTP PR ([#22673](https://github.com/ggml-org/llama.cpp/pull/22673)) enables Qwen 3.6 27B’s built-in multi-token prediction tensors for speculative decoding; the poster converted MTP-capable GGUF quants ([HF](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)) and reports **~`2.5×` faster generation** on an M2 Max 96GB, reaching **`28 tok/s`** with `--spec-type mtp --spec-draft-n-max 3`. They also published fixed Jinja chat templates ([HF](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)) and provide `llama-server` settings for OpenAI/Anthropic-compatible local serving with `q8_0` KV cache and up to **`262144` context**; recommendations emphasize `q8_0-mtp` as the best speed/quality quant, avoiding `q4_0` KV beyond `64k`, and note that Qwen3.6-27B only uses KV cache in **`16/65` layers** due to hybrid linear attention, reducing KV memory ~`4×`. A commenter reports on an **RTX Pro 6000 Max-Q** that Qwen 3.6 “2.7B” Q8 increases from **`36 tok/s` to `78 tok/s`** with MTP, at ~`20%` slower prompt processing, with no observed output-quality degradation; the post also warns that **vision currently crashes llama.cpp when combined with MTP**.** Commenters broadly frame this as part of a major recent acceleration in local inference, making consumer-hardware agentic coding more viable. One technical question asks whether `turbo3`/`turbo4` was merged separately or is part of the MTP PR.

    - A user benchmarked `qwen 3.6 2.7B Q8` on an **RTX Pro 6000 MaxQ** and reported generation increasing from `36 tok/s` to `78 tok/s` with **MTP**, roughly a `2.17x` speedup. They noted an approximately `20%` prompt-processing slowdown, but said output quality appeared unchanged, making the tradeoff favorable for generation-heavy workloads.
    - One commenter asked whether the speedup depends on the recent `turbo3`/`turbo4` merge or is specifically part of the **MTP PR**, highlighting that the implementation path matters for reproducing the claimed inference gains.
    - There was a technical comparison question against **Qwen 3.6 Dflash** variants and low-bit `iq3_XS` quantizations. The commenter reported usually fitting `256k` context into `16GB` VRAM and asked whether these quants can also support `256k` context without `mmproj`, indicating interest in KV-cache/context-length feasibility across quant formats.

  - **[Quality comparison between Qwen 3.6 27B quantizations (BF16, Q8_0, Q6_K, Q5_K_XL, Q4_K_XL, IQ4_XS, IQ3_XXS,...)](https://www.reddit.com/r/LocalLLaMA/comments/1t53dhp/quality_comparison_between_qwen_36_27b/)** (Activity: 820): **The post benchmarks **Qwen 3.6 27B** GGUF quantizations on a deliberately odd PGN-to-SVG chess-rendering task, testing board-state tracking, piece placement, orientation, and last-move highlighting with identical `llama.cpp` sampling settings (`temp=0.6`, `top_p=0.95`, `top_k=20`, `ctx=65536`). The author reports **BF16/Q8_0** as essentially correct, **Q6_K** showing placement degradation, **Q5_K_XL/Q4_K_XL/IQ4_XS** still usable, **IQ3_XXS** mostly correct but with wrong board orientation, and **Q2_K_XL** structurally broken despite correct piece positions; full outputs are posted at [qwen3-6-27b-benchmark.vercel.app](https://qwen3-6-27b-benchmark.vercel.app/). For local 16 GB VRAM use, they prefer **IQ4_XS**, reporting about `pp 100 tps` / `tg 8 tps` on vanilla `llama.cpp`, improved to roughly `pp 760 tps` / `tg 22 tps` using **TheTom's TurboQuant** fork with `-ngl 99`, `turbo4/turbo2` KV-cache quantization, and context limited below ~`75k`.** The main technical caveat raised in comments is that the evaluation appears to be **single-run**, so stochastic variance could make individual quantization results outliers; commenters still noted that the observed degradation trend broadly matches expectations.

    - Several commenters questioned whether the quantization comparison used **single-run evaluations or repeated trials**, noting that LLM outputs can vary enough that *“one run is not enough”* and may produce misleading conclusions from statistical noise or outlier generations. They still observed an apparent expected trend of **quality degradation as quantization becomes more aggressive**, but wanted multiple samples per quant level to support the findings.
    - One technically substantive takeaway was that **4-bit quantization appears to remain the practical sweet spot**, with **3-bit quants still described as usable** despite common skepticism. A commenter argued that above roughly **5-bit**, users may often gain more by moving to a larger/better model rather than preserving extra precision on a smaller one, citing comparisons like `122B UD-Q3_K_XL` versus `35B IQ4_NL`.

  - **[Qwen3.6 27B uncensored heretic v2 Native MTP Preserved is Out Now With KLD 0.0021, 6/100 Refusals and the Full 15 MTPs Preserved and Retained, Available in Safetensors, GGUFs and NVFP4s formats.](https://www.reddit.com/r/LocalLLaMA/comments/1t5yajb/qwen36_27b_uncensored_heretic_v2_native_mtp/)** (Activity: 530): ****llmfan46** released **Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved** on Hugging Face, claiming `KLD = 0.0021`, `6/100` refusals, and preservation/retention of the full `15` native MTP heads across [Safetensors](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved), [GGUF](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF), [NVFP4](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4), [NVFP4-GGUF](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-GGUF), [NVFP4-MLP-only](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-MLP-Only), and [GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) variants. The post says the release includes benchmarks and that all variants were checked for full MTP retention; the author’s full model list is [here](https://huggingface.co/llmfan46/models).** Commenters requested additional deployment-oriented quantization support, especially `Q4_K_XS` for `16GB` systems, and asked whether MTP works with TurboQuant-compressed KV cache or could be applied to Gemma 4 dense models. One technical concern was that if the MTP draft heads were trained on the original refusal-aligned model while only the base was fine-tuned, MTP acceptance may degrade or *“fight the heretic”* specifically on newly unlocked refusal/tail-behavior cases despite the low aggregate `KLD = 0.0021`.

    - A key concern was whether preserving the full `15` MTP heads is actually beneficial after an uncensoring/heretic fine-tune: if the draft heads retain the original refusal distribution while the base model was modified, speculative decoding may “fight” the newly unlocked outputs. One commenter noted that the reported **KLD `0.0021`** indicates the base stayed close overall, but may not capture *tail behavior* on refusal/unlocked prompts, making **MTP acceptance rate on heretic cases** the more important validation metric.
    - Users asked for deployment-specific quantization details, including a **`Q4_K_XS` GGUF** target to fit `16GB` VRAM while retaining useful context, and whether preserved MTP remains compatible with **TurboQuant-compressed KV cache**. Another hardware-focused question flagged that **NVFP4 + MTP on Blackwell** may currently be blocked by CUDA/tooling support, with the commenter saying the stack appears “dead in the water until a new CUDA version is released.”
    - There were implementation questions around multimodal packaging and stability: commenters noted the inclusion of `mmproj` files and asked whether crashes related to **PR `#22673`** are still present. Another asked whether the same MTP-preservation approach could apply to a future **Gemma 4 dense** model, implying interest in portability of native MTP heads across architectures/fine-tunes.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Limits Raised via SpaceX Compute

  - **[Doubled Rate Limits for Claude Code](https://www.reddit.com/r/ClaudeCode/comments/1t5hs98/doubled_rate_limits_for_claude_code/)** (Activity: 3901): ****Anthropic** says a new compute-capacity partnership with **SpaceX**, plus other recent compute deals, enabled higher usage limits across **Claude Code** and the **Claude API** ([announcement](https://www.anthropic.com/news/higher-limits-spacex)). Effective immediately, **Claude Code Pro/Max** no longer has the prior *peak-hours limit reduction*, and **Opus-model API rate limits** are being “substantially” raised.** Top comments were mostly non-technical reactions: surprise/skepticism about whether the announcement is real, plus speculation that the SpaceX/Anthropic tie-up reflects Elon Musk’s rivalry with Sam Altman.


  - **[SpaceX Conpute Deal - Double Limits](https://www.reddit.com/r/ClaudeAI/comments/1t5htq1/spacex_conpute_deal_double_limits/)** (Activity: 1931): ****Anthropic announced a compute partnership with SpaceX** to “substantially increase” capacity, alongside other compute deals, and is immediately changing limits: removing **peak-hours limit reductions** for **Claude Code Pro/Max** and **substantially raising API rate limits for Opus models** ([Anthropic announcement](https://www.anthropic.com/news/higher-limits-spacex)). The post does not specify exact new rate-limit numbers or the nature of the SpaceX compute arrangement.** Comments are skeptical that higher limits will materially improve usable capacity, with one noting users may simply hit weekly caps faster and another comparing Claude unfavorably to OpenAI Codex usage economics. There’s also concern that any improvement may be temporary and regress within weeks or months.

    - Several commenters argue that a raw compute-capacity deal would not materially improve **Claude Chat** unless Anthropic also changes product-level throttles: *“A usage limit increase that doesn't change the weekly limit is practically useless.”* The key technical/product distinction raised is between backend compute availability and enforced per-user weekly quota policy.
    - One comparison frames Anthropic’s quota pressure against **OpenAI Codex** pricing/usage: a user claims *“$20 on codex gets you infinitely more usage than Claude,”* suggesting Anthropic may be reacting to user churn caused by stricter effective compute limits. The discussion implies that any short-term limit relaxation may be temporary if demand again saturates available capacity.


### 2. AI Lab Corporate Governance Drama

  - **[Sam Altman texts Mira Murati. November 19, 2023. [This document is from Musk v. Altman (2026).]](https://www.reddit.com/r/OpenAI/comments/1t5tn1n/sam_altman_texts_mira_murati_november_19_2023/)** (Activity: 5431): **The post references an image/document titled **“Sam Altman texts Mira Murati. November 19, 2023”**, allegedly from **Musk v. Altman (2026)**, but the linked Reddit gallery was inaccessible due to **403 Forbidden**, so the actual text-message contents could not be verified or summarized. No technical claims, model details, benchmarks, implementation facts, or litigation-document substance were available from the provided post metadata.**


  - **[xAI will be dissolved as a separate entity.](https://www.reddit.com/r/singularity/comments/1t5q5jm/xai_will_be_dissolved_as_a_separate_entity/)** (Activity: 2116): **The image is a **non-technical screenshot of an X.com post** attributed to **Elon Musk**, claiming that **xAI would be dissolved as a separate company** and folded into “**SpaceXAI**,” described as AI products from SpaceX: [image](https://i.redd.it/tzexewkj2lzg1.jpeg). No implementation details, model changes, infrastructure plans, or product roadmap are provided in the post/title, so the significance is primarily **corporate-structure/contextual**, not technical.** Comments frame the move as consistent with Musk’s prior desire to combine AI work with his other companies, while skeptics characterize it as potentially moving unprofitable AI efforts into SpaceX, a profitable/government-contract-supported entity.



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
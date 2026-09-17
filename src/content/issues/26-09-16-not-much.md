---
id: MjAyNS0x
title: not much happened today
date: '2026-09-16T05:44:39.731046Z'
description: >-
  A 30-day local deployment test of **Unsloth Qwen3.8-27B-UD-Q4_K_XL** on **RTX
  5070 Ti + RTX 4070 Super** GPUs showed strong coding-agent and image/UI task
  performance but faced challenges with reasoning mode consuming up to 50%
  context and degraded speed compared to Qwen 3.6. Mitigations included subagent
  control and loop detection. Commenters highlighted that **FP8/Q8
  quantization** offers better stability and reliability than Q4, with contexts
  up to 262k tokens. Another post benchmarked **UkisAI Swift-Qwen3.8-27B**, a
  fine-tuned model reducing reasoning tokens by 40%, achieving comparable
  quality with faster completion times. Discussions included deployment on 16GB
  GPUs and quantization improvements by **ISTA** and **ByteShape**.
companies:
  - unsloth
  - ukisai
  - bottlecap-ai
  - hugging-face
  - ista
  - byteshape
models:
  - qwen3.8-27b
  - qwen3.6-27b
  - swift-qwen3.8-27b
  - thinkingcap-qwen3.6-27b
topics:
  - quantization
  - long-context
  - model-fine-tuning
  - reinforcement-learning
  - performance-optimization
  - gpu-optimization
  - model-deployment
  - loop-detection
  - reasoning
people: []
---


**a quiet day.**

> AI News for 9/15/2026-9/16/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap



---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.8-27B Local Optimization Benchmarks

  - **[I ran Qwen 3.8 27B locally for 30 days, here are the results](https://www.reddit.com/r/LocalLLM/comments/1whqwdq/i_ran_qwen_38_27b_locally_for_30_days_here_are/)** (Activity: 578): **A 30-day local deployment test of **Unsloth Qwen3.8-27B-UD-Q4_K_XL** reported `845.1 tok/s` mean prompt processing, `73.8 tok/s` mean generation, and MTP acceptance `0.481` (`674/1401`) on a dual-GPU setup later identified as **RTX 5070 Ti + RTX 4070 Super**. The author found the model production-usable for coding-agent workloads and strong on image/UI tasks, but noted major operational costs from reasoning mode: up to ~`50%` context consumed by reasoning, occasional attempted `60k`-token reasoning traces, degraded speed vs Qwen 3.6, poisoned/repeated tool calls at `100k+` context, and fragile cache reuse in `llama.cpp`. Their mitigations included enforced subagents, per-subagent reasoning-level control, non-naive loop detection with deletion of bad tool-call context, and using `--spec-type draft-dflash,ngram-mod`, which they measured as ~`20%` faster than MTP+ngram on their hardware.** Commenters focused on reproducibility and harness dependence: one asked which agent harness supports these fixes, while another reported **millions of tokens** on Qwen 3.8 27B at **FP8** up to nearly `262k` context with few tool-call/looping issues, arguing that Q4 quantization likely worsens looping and that FP8/Q8 has a clear stability benefit.

    - Several commenters focused on quantization and long-context stability: one reported generating **several million tokens** with **Qwen 3.8 27B at FP8** with “no issues with tool calls” and rare looping, running contexts up to nearly `262k` tokens with auto-compaction. They observed that looping appears much earlier at `Q4`, but can be partly mitigated at the harness level; the practical takeaway was that **FP8/Q8 provides a clear reliability benefit** if the hardware can support it.
    - A technical question challenged how portable the reported fixes are across agent harnesses, noting that many behaviors are **harness-bound**. The commenter specifically mentioned using `zcode` with subagents and `hermes`, and asked which harnesses were used because tool calling, compaction, subagent orchestration, and loop prevention may depend heavily on implementation details.
    - Hardware and deployment constraints came up briefly: one user asked for the hardware configuration, while another reported switching to `ukisai/Swift-Qwen3.8-27B-GGUF` and running it on an **RTX 5090**, describing “swift thinking” as impressive. Another asked whether **subagents still make sense when parallel connections cannot be served**, highlighting that agent architectures may lose much of their benefit if the serving stack is strictly serial.

  - **[Cut Qwen3.8-27B Reasoning Tokens by 40% -- 3.8 'ThinkingCap' benchmarked!](https://www.reddit.com/r/LocalLLaMA/comments/1wh5elt/cut_qwen3827b_reasoning_tokens_by_40_38/)** (Activity: 374): **The post benchmarks **[UkisAI](https://ukisai.com/)'s Swift-Qwen3.8-27B**—not BottleCap's ThinkingCap—as a fine-tune aimed at reducing Qwen 3.8 27B “overthinking” by penalizing reasoning-marker tokens via RL and using a transfer component related to **[BottleCap AI's ThinkingCap-Qwen3.6-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B)**. In the author's Aider coding eval using `Q8_0`, Swift-Qwen3.8-27B achieved roughly comparable quality to Qwen3.8-27B while cutting completion tokens from `12,547` to `7,301`, seconds/case from `1,481` to `750`, and total tokens/solve from `19.3k` to `12.1k`, with Pass1 `30.8%` vs `27.1%` and Pass2 `75.7%` vs `77.6%`. A UkisAI creator clarified that the model was *not* trained on ThinkingCap traces, linked their methodology post ([Reddit](https://www.reddit.com/r/LocalLLaMA/s/SbzAnLuqiU)), and said a **Qwen 3.8 Flash Next** variant is planned.** Commenters focused on deployment: one suggested asking **ISTA** or **ByteShape** to produce high-quality quantizations, arguing an `IQ3` build could make it a strong assistant/coding model for `16GB` GPUs. Another shared an already-outdated **NInfer** artifact for Swift-Qwen3.8-27B on Hugging Face ([knoopx/Swift-Qwen3.8-27B-NInfer](https://huggingface.co/knoopx/Swift-Qwen3.8-27B-NInfer)) and noted it may need migration to the newer v3 weight-profile architecture.

    - A **UkisAI lab model creator** clarified that the model was *not* trained on ThinkingCap traces, arguing that using **Qwen 3.6 27B traces** would likely degrade performance because it conflicts with Alibaba’s RL improvements in **Qwen 3.8 27B**. They also noted a forthcoming **Qwen 3.8 Flash Next** release with no thinking-reduced variant, and pointed to the training-methodology discussion in their [model/post explanation](https://www.reddit.com/r/LocalLLaMA/s/SbzAnLuqiU).
    - One commenter suggested running **ISTA** or **ByteShape** quantization suites on the model, claiming they offer strong performance-per-filesize tradeoffs and could compound well with the reduced-thinking-token behavior. They specifically highlighted the potential for a strong assistant/coding setup on `16GB` GPUs using a high-quality `IQ3` quant.
    - Several users identified **endless reasoning loops** as a more important bottleneck than raw speed for **Qwen 3.8 27B**, with one reporting persistent looping even at `Q8` despite switching to newer Jinja templates and adjusting thinking settings. Another noted that Chinese reasoning models often struggle to decide when to stop generating, making lower token prices less meaningful unless reasoning-length control—such as Qwen 3.8 27B’s reasoning restriction parameter—actually works reliably.

  - **[Radeon AI Pro R9700 w/ Qwen3.8-27B Q8 hitting 90.8toks](https://www.reddit.com/r/LocalLLM/comments/1wh7ywq/radeon_ai_pro_r9700_w_qwen3827b_q8_hitting_908toks/)** (Activity: 340): **The [benchmark screenshot](https://i.redd.it/5uv494fx4qph1.png) shows **Qwen3.8-27B** on a **Radeon AI Pro R9700** using `Q8_0`, reporting `90.8 tok/s` generation, `1,413.7 tok/s` prefill, `370 ms` TTFT, batch `1`, `30` input / `400` output tokens, and a listed `262,144`-token context with `49.3 GB` VRAM usage. The post credits the [`llama-cpp-rdna-boosts`](https://github.com/stew675/llama-cpp-rdna-boosts) repo for making the setup practical, while linking the full LocalMaxxing run [here](https://www.localmaxxing.com/en/runs/cmu2x80ei068ulq01ec0aaxd4).** Commenters questioned the title/claim because a `Q8` 27B model is roughly `29 GB` by itself and an `F16` KV cache for `256 KiB` context would not fit on a `32 GB` card; the screenshot’s `49.3 GB` VRAM figure reinforces that concern. Another commenter suggested an alternative **MXFP4 vLLM/Radiance** build as faster: https://codeberg.org/ggz14/radiance-vllm-mxfp4

    - Several commenters challenged the VRAM feasibility of the title: **Qwen3.8-27B at Q8_0 is estimated around `29GB` just for weights**, so adding a **`256 KiB` K/V context at F16** would exceed a single **`32GB` Radeon AI Pro R9700**. The reported **`49.3GB VRAM`** usage suggests the run was not on one card, and a later comment indicates it may have been using **`3x R9700`**, making the headline misleading for single-GPU expectations.
    - One commenter recommended an alternative **MXFP4 vLLM build** claimed to be faster for this workload: [radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4). The suggestion implies that lower-precision MXFP4 inference may provide better throughput than the reported **Q8** configuration, especially for large Qwen models constrained by VRAM bandwidth/capacity.

  - **[Voodoo Dynamic Quant - Now MIT Licensed](https://www.reddit.com/r/LocalLLaMA/comments/1wgszma/voodoo_dynamic_quant_now_mit_licensed/)** (Activity: 412): **The image ([chart](https://i.redd.it/bdbwr3v4imph1.png)) is a dark-themed benchmark comparison for **“Voodoo Dynamic Quant - Now MIT Licensed”**, showing `Torch KLD`, `llama.cpp KLD`, and `llama.cpp PPL` versus GGUF model size in MB across **Voodoo**, **Unsloth**, and **llama.cpp** quantization variants. In context, the post announces an MIT-licensed toolset for Voodoo Dynamic Quant, which uses **gradient descent over per-tensor quantization gates** to choose GGUF quant levels under a target filesize, optimizing KL divergence against a BF16 reference checkpoint. The plotted results support the author’s claim that Voodoo is especially competitive at aggressive low-size quantization levels, while the post notes **Unsloth Dynamic 3.0** may still perform better at mid/high quant levels.** Comments were broadly positive about open-sourcing the method and suggested maintainers such as **Bartowski** might adopt it for public quants. One commenter criticized the GitHub README as AI-written/over-marketed and asked for clearer technical wording.

    - A commenter asked how **Voodoo Quant** can use gradient descent when quantization levels are discrete rather than continuous, specifically questioning the claim that it “runs all the quant levels of a model at the same time, for every tensor” and lets optimization pick levels for a target filesize. The key technical issue raised is how discrete quant choices are represented in a differentiable objective, since arbitrary gradient steps cannot directly move between quantization levels.
    - Another commenter reported testing a very similar quantization-layout optimization approach on **Gemma 3 1B** and found it computationally prohibitive: a single optimization step on a **6000 Pro** took about `40 minutes` at `batch=128`, with uncertain convergence. They also noted that calibration/training context length materially affects optimal quant layouts, saying layouts optimized at `4k` context differed significantly from those at `200k`, implying long-context calibration may be necessary but expensive.
    - There was a request for the method to be picked up by established quantization maintainers such as **Bartowski** (`u/noneabove1182`), suggesting the main practical value may come from integrating Voodoo Dynamic Quant into existing community quantization pipelines rather than remaining a standalone research repo.


### 2. Open-Weight Frontier Race and DeepSeek RSI

  - **[China's open-weight AI models are now just 4 months behind frontier US offerings, Mozilla report claims — models still lag in some benchmarks but are drastically cheaper to use](https://www.reddit.com/r/LocalLLaMA/comments/1wi32jg/chinas_openweight_ai_models_are_now_just_4_months/)** (Activity: 645): **A Mozilla analysis reported via [Tom’s Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinas-open-weight-ai-models-are-now-just-4-months-behind-frontier-us-offerings-mozilla-report-claims-models-still-lag-in-some-benchmarks-but-are-drastically-cheaper-to-use) claims leading **Chinese open-weight models** are now only about `4 months` behind frontier U.S. systems, while remaining materially cheaper to run. The report notes these models still underperform top U.S. offerings on some benchmarks, but their cost/performance profile could make them attractive for production deployments where “good enough” capability matters more than absolute frontier performance.** Commenters framed the current generation as already past a practical “good enough” threshold, with interest shifting toward lower inference prices, agentic reliability, RL-based refinement for code/voice quality, and fine-tuning. Some argued U.S. GPU export restrictions are the main remaining constraint on Chinese model progress, while others interpreted the `4-month` gap as evidence that frontier capabilities such as GPT/Astra-like systems may diffuse quickly.

    - Commenters highlighted that recent open-weight models may have crossed a practical *“good enough”* threshold for many workflows, shifting the priority from raw capability to **cost reduction**, better **agentic reliability**, and targeted post-training such as RL for improved *“taste in voice and code.”* The discussion frames the next competitive axis as cheaper inference and refinement rather than only benchmark leadership.
    - A technically relevant contrast was drawn between **open-weight/local deployment** and closed frontier APIs such as **Claude**, with commenters arguing that local models can be used in security-sensitive environments where external API calls are unacceptable. This was presented as a practical advantage independent of benchmark parity: open models may lag in some metrics but offer deployability, auditability, and control that closed models do not.

  - **[DeepSeek engineer relections on RSI - burying my talent to yesterday](https://www.reddit.com/r/LocalLLaMA/comments/1wgii3h/deepseek_engineer_relections_on_rsi_burying_my/)** (Activity: 635): **A **DeepSeek engineer** argues in a translated [WeChat post](https://mp.weixin.qq.com/s/zk0KxuLzhmMJ4LPYW_OHMA) that AI has moved from doc/code-assist to autonomously reading `CUDA`/`PTX`/`SASS`, profiling per-instruction stalls, and optimizing GPU operators, predicting AI-written kernels may match or exceed expert human work within `6–12 months`. They claim authorship of DeepSeek v4.1’s main attention operator—specifically **MQA attention with `head_dim = 512`**, excluding the top-k token indexer—and frame the near-term role shift as moving from hand-writing operators to “piloting” AI agents that generate and tune them. The post also raises a technical education concern: AI-assisted lab completion may erode core engineering skills like abstraction, system design, and full-stack reasoning, potentially increasing the rate at which poorly designed code is produced.** Commenters largely focused on the labor and governance implications: senior engineers said this AI transition feels larger than prior tooling shifts, but that being better at using AI than peers may preserve short-term employability. Others highlighted the geopolitical inversion: **OpenAI/Anthropic** often argue they must build AGI before China does, while this DeepSeek engineer argues open, cheap access is needed to prevent corporate-controlled “Cyberpunk 2077”-style AI inequality.

    - A commenter distilled the original DeepSeek engineer’s technical claim: in low-level GPU work—writing CUDA/PTX/SASS attention kernels—AI has moved from assistant to potentially outperforming expert humans in under a year. They cite the engineer’s expectation that model-assisted systems may surpass their own operator/kernel-writing ability within `6–12 months`, shifting the human role from direct implementation to supervising AI agents that generate and optimize kernels.
    - One technical correction noted that the translated term **“operator”** should likely be read as **CUDA kernel**, especially in the context of Attention implementations and GPU optimization. This matters because the discussion is specifically about low-level kernel engineering—CUDA/PTX/SASS performance work—not generic ML “operators” at a framework abstraction level.
    - The comments highlight a skills-development concern: if students use AI to complete programming and systems labs, they may fail to build durable engineering abilities such as abstraction, system design, debugging intuition, and cross-stack understanding. The technical worry is not merely job replacement, but that AI could enable mediocre engineers to ship flawed systems at `10x` speed without acquiring the expertise needed to evaluate or maintain what agents produce.

  - **[Hey, Meta. Where's those Muse Spark weights?](https://www.reddit.com/r/LocalLLaMA/comments/1whqm2c/hey_meta_wheres_those_muse_spark_weights/)** (Activity: 503): **The [image](https://i.redd.it/9ka4k65h6uph1.png) is a **meme/non-technical criticism** of **Meta** for not releasing promised **Muse Spark open weights** after more than a month, despite the poster noting Spark has moved from `1.2` to `1.3`. The post frames the delay against Zuckerberg’s argument that model releases cannot be delayed “even a month” in competition with Chinese open models, asking whether Meta will release the originally promised `1.2` weights or a newer current version.** Comments are broadly distrustful and cynical: users compare the situation to **Grok**, where newer versions remain closed while only older versions are open, and joke that Meta’s infinity logo implies an indefinite wait.

    - Commenters contrasted **Meta’s unreleased Muse/Spark weights** with **xAI’s Grok release pattern**, noting that *“Grok 4.6 (4.7 upcoming)”* exists while only **Grok 1 and Grok 2** have been open-released, implying a widening lag between frontier closed models and published weights.
    - A technically relevant explanation linked to Mark Zuckerberg’s post on X: [x.com/finkd/status/2099997096896274533](https://x.com/finkd/status/2099997096896274533). The quoted rationale says labs face liability if models cause harm, and claims **Meta delayed Muse for several months** specifically to work on *“safety and security”* and build stronger security foundations before release.


### 3. Apple Local AI and Server Ambitions

  - **[Apple Foundation Models: local AI natively on MacOS 27](https://www.reddit.com/r/LocalLLaMA/comments/1wh5fpa/apple_foundation_models_local_ai_natively_on/)** (Activity: 368): **The post says **Apple Foundation Models (AFM)** are available locally on **macOS 27** and can be invoked from Terminal with `fm chat`, framing this as a native, hardware-optimized local-AI path for Apple devices. A technical commenter reports two Neural Engine–optimized releases: finetunes of **Gemma `3B` dense** and **`20B` MoE**, with the `3B` model allegedly reaching **`85+ tok/s` on an M4 Pro with `24GB` RAM**, running primarily on the **Apple Neural Engine** rather than MLX/GPU, and intended for Apple Intelligence/app-level APIs.** Commenters are skeptical of capability: the `3B` model is described as *not good for agentic work*, and the `20B` MoE is expected to trail **Qwen** models in quality. The perceived value is less SOTA performance and more **power efficiency, native integration, and developer APIs** inside the Apple ecosystem.

    - Commenters noted Apple appears to have released **two Apple Foundation Models optimized for the Mac Neural Engine**, reportedly fine-tuned from **Gemma** variants: a `3B` dense model and a `20B` MoE model. One user reported the `3B` is not strong for agentic workflows and expects the `20B` MoE to trail stronger open models like **Qwen**, but emphasized Apple’s likely goal is **power-efficient local inference and OS/app integration** rather than frontier-model competitiveness.
    - A concrete performance datapoint was shared: the models can run entirely on the **Apple Neural Engine** and may not require **MLX**, with one user reporting **`85+ tokens/sec` on an M4 Pro with `24GB` RAM**. The technical value is framed around exposing native APIs so developers can add Apple Intelligence-style local AI features without shipping their own inference stack.
    - Discussion also touched on model format lock-in: one commenter speculated about a converter from **MLX** or **GGUF** into Apple’s native model format, but questioned whether this is technically feasible or intentionally restricted by Apple’s ecosystem design. Another user who tested the macOS 27 beta described the use case as “simple-ish on-device” personalization/context tasks, saying it is substantially better than old Siri but not intended to compete with downloadable open-weight or frontier models.

  - **[Apple May Return to Server Market With Nvidia Technology](https://www.reddit.com/r/LocalLLaMA/comments/1why9ao/apple_may_return_to_server_market_with_nvidia/)** (Activity: 448): ****Apple** is reportedly evaluating an externally sold AI inference server using future **M8-series Apple Silicon**, with a tentative **2029** timeframe and possible cancellation before launch, per [MacRumors](https://www.macrumors.com/2026/09/16/apple-may-return-to-server-market/). The system could use **Nvidia NVLink Fusion** for chip-to-chip/inter-accelerator networking, potentially to scale beyond Apple’s internal Private Cloud Compute-style interconnects, positioning it against datacenter AI platforms for on-prem model serving rather than training-heavy workloads.** Commenters were skeptical due to Apple’s prior abandonment of **Xserve** and the cylindrical Mac Pro era, arguing enterprise buyers prioritize long-term platform stability comparable to **x86 + CUDA** backward compatibility. Another major concern was OS support: commenters argued the product would be “dead in the water” for non-Apple datacenters unless Apple officially supports **Linux** rather than requiring Darwin/macOS-derived infrastructure.

    - Commenters emphasized that **datacenter buyers prioritize long-term platform stability** over hardware novelty, citing Apple's discontinuation of **Xserve in 2011** and the later **Mac Pro “trash can”** transition as examples of ecosystem rug-pulls. One technically substantive comparison was that **CUDA code written nearly `20 years` ago can still run with little or no modification** across old and current Nvidia GPUs, which commenters argue is a key reason **x86 + Nvidia** remains dominant in professional and server workloads.
    - Several commenters argued that any Apple server effort would be “dead in the water” for external datacenters unless Apple provides **official Linux support** rather than requiring Darwin/macOS-derived environments. The view was that a revived **Xserve-like system** with supported Linux could be competitive against Nvidia-oriented datacenter platforms such as **GB300**, but without Linux compatibility it would be unattractive to most non-Apple infrastructure operators.
    - One thread referenced Apple's historically strained relationship with **Nvidia**, particularly the overheating/failure issues around early **Intel/Nvidia unibody MacBooks**, as a potential obstacle to renewed collaboration. The technical concern is less about feasibility and more about whether Apple and Nvidia can sustain a supportable hardware/software partnership for enterprise deployments.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



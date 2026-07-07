---
id: MjAyNS0x
title: not much happened today
date: '2026-07-06T05:44:39.731046Z'
description: >-
  **Tencent** released **Hy3**, a **295B MoE** open-weight model with **21B
  active parameters**, **192 experts**, and **256K context** supporting **MTP
  speculative decoding**. It runs natively on **vLLM** with optimizations for
  **NVIDIA** and **AMD** hardware, achieving up to **2.95x** speedups and
  latency reductions. Hy3 competes closely with **GLM-5.2** in the open model
  space. **AutomationBench-AA** leaderboard evaluates agents on **657 tasks**
  across **40 SaaS apps**, with **Claude Fable 5** leading, followed by **Opus
  4.8**, **Gemini 3.5 Flash**, and **GPT-5.5 xhigh**. Open models lag behind,
  with **GLM-5.2 max** best at **27.8%**. New domain-specific capability indices
  highlight cost-performance tradeoffs. Research on persistent agent memory
  includes **A-TMA** improving conflict accuracy and **ReContext** enhancing
  long-context inference without retraining.
companies:
  - tencent
  - nvidia
  - amd
  - nous-research
  - hugging-face
  - artificial-anlysiis
  - dair-ai
models:
  - hy3
  - glm-5.2
  - claude-fable-5
  - opus-4.8
  - gemini-3.5-flash
  - gpt-5.5-xhigh
  - glm-5.2-max
topics:
  - mixture-of-experts
  - model-quantization
  - speculative-decoding
  - inference-speed
  - agent-evaluation
  - long-context
  - memory-optimization
  - cost-efficiency
  - benchmarking
  - multi-domain-evaluation
people:
  - eliebakouch
  - shunyuyao12
  - vllm_project
  - teortaxestex
  - tinygrad
  - mbusigin
  - artificialanlys
  - fchollet
  - omarsar0
---


**a quiet day.**

> AI News for 7/04/2026-7/06/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Tencent Hunyuan’s Hy3 Release and the Open-Weight Frontier**

- **Hy3 lands as a serious open model**: Tencent released **Hy3** under **Apache 2.0**, a **295B MoE** with **21B active parameters**, **192 experts / top-8 routing**, **GQA**, **256K context**, and a **3.8B MTP layer** for speculative decoding. Multiple posts framed it as competitive with much larger systems on reasoning, coding, and agentic tasks, with particular emphasis on reliability improvements like tool-calling stability and anti-hallucination work [@eliebakouch](https://x.com/eliebakouch/status/2074011171661701466), [@HuggingPapers](https://x.com/HuggingPapers/status/2074024501201813797), [@ShunyuYao12](https://x.com/ShunyuYao12/status/2074151389945827744).
- **Inference support was unusually day-0 mature**: [@vllm_project](https://x.com/vllm_project/status/2074147504254517529) said Hy3 runs natively in **vLLM** from launch with tool-call and reasoning parsers, **MTP speculative decoding**, and validated support on **NVIDIA and AMD**. A follow-up detailed Tencent production kernels now upstreamed into vLLM main, including load-balanced decode scheduling and fused FP8 MoE serving, with reported gains of **up to 2.95x** on mixed-length decode and latency reductions of roughly **24% TTFT** and **17% TPOT** versus default backends [@vllm_project](https://x.com/vllm_project/status/2074147506875969754). Community reaction was strong enough that [@Teknium](https://x.com/Teknium/status/2074264567803531589) quickly made Hy3 free on Nous Portal for two weeks.
- **Broader open-model context**: Hy3 was immediately compared against **GLM-5.2**, with some posters arguing Tencent has now joined the very top tier of open-source labs if the benchmark and vibe-test results hold [@teortaxesTex](https://x.com/teortaxesTex/status/2074012467886178725), while others still maintained **GLM-5.2** as the best currently usable open-weight model in practice [@__tinygrad__](https://x.com/__tinygrad__/status/2074206866641752190), [@mbusigin](https://x.com/mbusigin/status/2074238100251799998). The net takeaway: the open frontier is compressing fast, and the competition is increasingly about deployment robustness rather than just raw leaderboard deltas.

**Agent Benchmarks, Harnesses, and Long-Running Memory**

- **AutomationBench-AA adds a more realistic agent eval**: [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074194764510208230) launched an independent leaderboard for Zapier’s **AutomationBench**, evaluating agents across **657 tasks** and **40 simulated SaaS apps** with both objectives and guardrails. **Claude Fable 5** led at **48.6%**, narrowly ahead of **Opus 4.8** at **48.5%**, with **Gemini 3.5 Flash** at **42.6%** and **GPT-5.5 xhigh** at **42.1%**. More interesting than the ranking: every model still breaks business rules, and Gemini looked notably strong on **objective-per-guardrail-violation** and **cost efficiency**. Open weights remain meaningfully behind, with **GLM-5.2 max** the best listed open model at **27.8%**.
- **Capability indices are becoming multidimensional**: Artificial Analysis also introduced six domain-specific indices—**Finance & Accounting, Legal, Healthcare & Medical, Strategy & Ops, Engineering, Economics**—to move past single scalar model scores [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074299714699469221). The headline was familiar—**Claude Fable 5** plus **Opus 4.8 fallback** leads—but the more useful insight is how sharply rankings reshuffle by domain and how steep the price/performance frontier has become. This aligns with [@fchollet](https://x.com/fchollet/status/2074242671103889799), who argued that reporting benchmark scores without **cost per task** is increasingly meaningless.
- **Memory and retrieval remain bottlenecks for persistent agents**: Two papers got traction here. First, **A-TMA** tackles “ghost memory,” where stale and current facts are retrieved together in long-running assistants; on the LTP benchmark, adding it to Graphiti reportedly improves conflict accuracy by **+0.240 absolute** [@omarsar0](https://x.com/omarsar0/status/2074121191846261022). Second, **ReContext** is a training-free long-context inference harness that replays model-internal evidence right before answer generation, improving evidence utilization across eight 128K datasets [@dair_ai](https://x.com/dair_ai/status/2074178316819677238). Combined with **BlockSearch** for million-token in-context retrieval [@dair_ai](https://x.com/dair_ai/status/2074117920133898707), the theme is clear: better memory behavior is increasingly being engineered at inference time, not just trained in.

**Anthropic’s J-Space / Global Workspace Results**

- **Mechanistic interpretability took center stage**: Anthropic released research claiming a **global-workspace-like internal structure** in Claude, centered on a small subset of activations they call **J-space** [@AnthropicAI](https://x.com/AnthropicAI/status/2074185348142280912), [@AnthropicAI](https://x.com/AnthropicAI/status/2074185387577094398). The core claim is not chain-of-thought extraction, but identification of a privileged internal representational substrate that appears available for report, modulation, and flexible reasoning. Anthropic also shipped a Neuronpedia demo for open-weight models [@AnthropicAI](https://x.com/AnthropicAI/status/2074185390060110138).
- **Why researchers cared**: Interpretability researchers treated this as stronger evidence for a model “working memory” or internal workspace than prior public work, even if they disagreed with the framing. [@NeelNanda5](https://x.com/NeelNanda5/status/2074193936588148891) called it the best evidence yet for a working-memory-like mechanism. [@Jack_W_Lindsey](https://x.com/Jack_W_Lindsey/status/2074215950602379388) argued understanding this privileged space could be key to LLM cognition. Posts also highlighted practical safety angles: the workspace can reportedly surface hidden concepts, detect prompt injections, and expose internal sabotage-related features before they are verbalized [@mlpowered](https://x.com/mlpowered/status/2074190714100146483), [@LiorOnAI](https://x.com/LiorOnAI/status/2074198891990548940), [@omarsar0](https://x.com/omarsar0/status/2074264122330612223).
- **But the “consciousness” language was contested**: Anthropic’s public framing invited strong pushback. Supporters said the results suggest a functional analog of **access consciousness** rather than phenomenal consciousness [@BorisMPower](https://x.com/BorisMPower/status/2074201312531734567), while critics argued the company was overclaiming by conflating privileged latent activation with consciousness [@AlanCowen](https://x.com/AlanCowen/status/2074265992570736919). Even some sympathetic takes emphasized the bigger story is a new **intervention point** for auditing and steering models, not philosophy.

**Inference, Serving, and Systems Efficiency**

- **Speculative decoding remains hot infrastructure**: [@lmsysorg](https://x.com/lmsysorg/status/2074176669108367549) added **DSpark** to SGLang for confidence-driven, variable-length verification. The pitch is that under high load it avoids verifying every draft token, improving the throughput/latency tradeoff relative to fixed-budget speculative methods; DeepSeek-V4-Pro reportedly reached **383.7 tok/s at batch=1 on B300**. Microsoft also discussed prompt-level optimization of **GPT-5.5** in the GitHub Copilot harness to improve latency and token efficiency after launch [@code](https://x.com/code/status/2074178799512539571), [@pierceboggan](https://x.com/pierceboggan/status/2074180737147027757).
- **Inference efficiency is increasingly the strategic bottleneck**: [@jon_durbin](https://x.com/jon_durbin/status/2074169183835685351) argued that inference, not training alone, is now “the whole game,” because every data pipeline, RL loop, and agent runtime ultimately cashes out as test-time compute. That perspective also showed up in lower-level kernel work: Chutes reported major speedups for **MiniMax MSA** and **GatedDeltaNet-2**, including **~7x** sparse-attention training improvements on **RTX Pro 6000 / SM120** and better fused FP8 kernels [@jon_durbin](https://x.com/jon_durbin/status/2074119835366134188).
- **Infra releases beyond model serving**: Cloudflare launched **Workers Cache**, a regionally tiered cache in front of Worker entrypoints configured via standard HTTP headers [@Cloudflare](https://x.com/Cloudflare/status/2074117419728007181). OpenAI shipped **GPT-Realtime-2.1-mini**, bringing reasoning and tool use to the mini realtime line at the same price as the prior mini, alongside claimed **25%+ p95 latency reductions** from caching improvements [@OpenAIDevs](https://x.com/OpenAIDevs/status/2074255408013955466), [@OpenAIDevs](https://x.com/OpenAIDevs/status/2074255420831735824).

**World Models, Speech, and Document AI**

- **MIRA is a notable world-model demo**: General Intuition and Kyutai, with Epic Games, introduced **MIRA**, a playable multiplayer world model for Rocket League trained on **10k hours** of bot-collected data [@gen_intuition](https://x.com/gen_intuition/status/2074104524596457706). It runs in real time at **20 fps**, and posts highlighted a **5B-parameter** model running an entire 2v2 match on a single **NVIDIA B200**, with no explicit physics or rendering engine [@TheRundownAI](https://x.com/TheRundownAI/status/2074184559768277398). This was one of the clearest signals that video/world-model work is moving from toy demos toward interactive simulators.
- **Speech remains highly competitive**: AssemblyAI released **Universal-3.5 Pro Realtime**, a streaming STT model with **4.1% WER** on AA-WER Streaming and contextual priming that can be updated mid-call without reconnecting [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074160133702402314). On the TTS side, Artificial Analysis said **Speechify Simba 3.2** now leads its Speech Arena at **1233 Elo**, ahead of Gemini 3.1 Flash TTS, Sonic 3.5, and Inworld Realtime TTS 1.5 Max, while also being the cheapest among top-ranked models [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074265309985570890).
- **Document-context pipelines are becoming multimodal by default**: LlamaIndex and LanceDB described a retrieval pipeline for messy PDFs that separates **pages, chunks, and extracted assets** into linked multimodal tables, reporting **82% any-page-hit@5** and **74% answer accuracy** on a labeled ESG-report benchmark [@lancedb](https://x.com/lancedb/status/2074153945631457663), [@llama_index](https://x.com/llama_index/status/2074170470119752084). This pairs with Jerry Liu’s broader argument for a dedicated “document context layer” for agents [@jerryjliu0](https://x.com/jerryjliu0/status/2074165277634253106).

**Top tweets (by engagement)**

- **Anthropic’s global workspace paper** dominated engagement, with the primary announcement on Claude’s internal workspace/J-space far above everything else [@AnthropicAI](https://x.com/AnthropicAI/status/2074185348142280912).
- **Tencent Hy3** was the biggest pure model-release story, especially among technical accounts discussing open-source competitiveness and deployment [@teortaxesTex](https://x.com/teortaxesTex/status/2074012467886178725), [@ShunyuYao12](https://x.com/ShunyuYao12/status/2074151389945827744).
- **MIRA’s playable world model** was the standout multimodal/system demo [@gen_intuition](https://x.com/gen_intuition/status/2074104524596457706).
- **Will Depue’s “Stargate for Data”** thread was the most substantive strategy post, arguing that data collection—not compute alone—becomes the binding constraint and potential moat for frontier labs [@willdepue](https://x.com/willdepue/status/2074178395462848800).
- **John Carmack’s memory-system thread** drew significant technical interest by arguing inference hardware could exploit deterministic access patterns and much cheaper memory tiers than HBM for large-model serving [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2074248758422864226).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Large Open-Weight MoE Model Releases

  - **[longcat 2.0 (1.6T, ~48B active) weights are now open under MIT license](https://www.reddit.com/r/LocalLLaMA/comments/1unyvnz/longcat_20_16t_48b_active_weights_are_now_open/)** (Activity: 638): ****LongCat 2.0** weights are now open under the **MIT license** via announcements from [elie](https://x.com/eliebakouch/status/2073690402503487902) and [ModelScope](https://x.com/ModelScope2022/status/2073710226365165679), with technical details in the [LongCat 2.0 blog post](https://longcat.chat/blog/longcat-2.0/). The model is a very large **MoE** system with `1.6T` total parameters and roughly `48B` active parameters per inference; commenters note the released weights occupy about `3.55 TB` in **BF16** and `2.05 TB` in **FP8**.** Commenters emphasized the practical deployment burden from the multi-terabyte weight size, and noted that **Meituan**—described as China’s Groupon/Uber Eats analogue—reportedly trained it on fully domestic Chinese chips, prompting discussion about the geopolitical/market significance.

    - Commenters highlighted the scale and deployment footprint of **LongCat 2.0**: `1.6T` total parameters with approximately `48B` active parameters, implying a sparse/MoE-style architecture. One user noted the released weights require about `3.55 TB` in **BF16** and `2.05 TB` in **FP8**, which is important for anyone planning local storage or inference infrastructure.
    - A technical point raised was that **Meituan** reportedly trained the model on `100%` domestic Chinese chips, which commenters framed as significant for AI hardware supply-chain independence. This is especially notable given Meituan’s role as a major Chinese internet company comparable to a mix of Groupon and Uber Eats rather than a traditional AI lab.
    - Several users focused on the permissive **MIT license** and planned benchmarking against frontier open models such as **Qwen** and **DeepSeek**. The combination of `1.6T` total parameters, only `~48B` active parameters, and open weights suggests the model may be practical to compare with other high-end MoE open models if inference tooling supports its architecture efficiently.

  - **[New open  model from Tencent Hy: Hy3 (295B total 21B active - apache 2.0)](https://www.reddit.com/r/LocalLLaMA/comments/1uoozt4/new_open_model_from_tencent_hy_hy3_295b_total_21b/)** (Activity: 604): ****Tencent** released the non-preview **Hy3** model collection on [Hugging Face](https://huggingface.co/collections/tencent/hy3), described as a `295B`-parameter MoE with `21B` active parameters, now under **Apache 2.0** rather than the prior restrictive community license. Commenters highlight a linked benchmark chart and claim the release shows *“pretty impressive claimed gains over HY3-Preview”*, potentially making it relevant for high-end local/home inference setups if real-world performance matches reported results.** The main discussion is positive around the license change: commenters view moving from a geographically/restrictively limited license to **Apache 2.0** as the most important improvement, especially alongside Tencent’s recent Apache-licensed translation models.

    - Commenters highlight that **Tencent Hunyuan Hy3** is a large MoE-style release at **`295B` total parameters with `21B` active**, and one user notes the claimed benchmark gains over **HY3-Preview** appear substantial enough that, if they transfer to real workloads, it could be relevant for “high end home setups.” There is interest in practical inference availability, especially **GGUF quantizations**, which would determine whether local deployment is feasible.
    - A technically important licensing change was noted: Tencent reportedly moved from a more restrictive “community” license that limited usage in regions such as **South Korea, the UK, and the EU** to **Apache 2.0**. Commenters view this as significant because it enables broader commercial and research reuse, consistent with some of Tencent’s recent Apache-licensed translation models.
    - One commenter frames Hy3 as a potential alternative to **Qwen** and **MiniMax**, implying interest in whether its benchmark and real-world performance can compete with the current leading open-weight Chinese model families.

  - **[New model: GigaChat3.5-432B-A28B (with day-0 GGUF support!)](https://www.reddit.com/r/LocalLLaMA/comments/1uotkm7/new_model_gigachat35432ba28b_with_day0_gguf/)** (Activity: 439): ****Sberbank/ai-sage** released **GigaChat3.5-432B-A28B** on Hugging Face in both [instruct](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B) and [base](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B-base) variants, plus day-0 [GGUF weights](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B-GGUF); `llama.cpp` support is available via PR [ggml-org/llama.cpp#25342](https://github.com/ggml-org/llama.cpp/pull/25342), not yet master. Commenters quote the model card as a **custom MoE** replacing prior **GigaChat 3.1 Ultra 700B** with a ~`40%` smaller model that is stronger on code/math/agentic tasks, uses ~`4×` less KV cache/token, fits >`2×` more context in the same memory, and improves generation throughput by ~`20%`. Architecturally, it reportedly uses a **hybrid MLA + GatedDeltaNet linear-attention** stack plus **two MTP heads**, with claimed greedy decoding speedups of ~`1.5×` for one head and up to `2.2×` for two.** Top technical caveats were that benchmark comparisons to **DeepSeek 3.2** may be a weak reference point versus current frontier models, and that GigaChat3.5 is a **non-reasoning** model, so benchmark interpretation should account for that. One commenter praised the release openness—base model plus intermediate/open-weight checkpoints for a model of this size—while noting the training dataset remains undisclosed.

    - Commenters note that benchmark comparisons should account for **GigaChat3.5-432B-A28B** being a *non-reasoning* model, making comparisons against reasoning/frontier models potentially misleading. One user questioned using **DeepSeek 3.2** as a reference point, calling it “~year behind the frontier models,” while another argued non-reasoning baselines are increasingly rare and should be evaluated separately.
    - The release is viewed as unusually open for a model of this size: commenters highlighted that **intermediate checkpoints and the base model are open-weighted**, which they described as “top 10% of openness of models on HF.” The main missing artifact called out was the **exact training dataset**, which limits full reproducibility and data-contamination analysis.
    - A detailed architecture excerpt says **GigaChat 3.5 Ultra** is ~`40%` smaller than **GigaChat 3.1 Ultra 700B** while improving code, math, and agentic performance, using ~`4×` less KV-cache per token, fitting `>2×` more context in the same memory, and improving throughput by ~`20%`. The model uses a custom MoE hybrid attention design combining **MLA** with **GatedDeltaNet** linear-attention layers, plus **Multi-Token Prediction** with two heads, reportedly giving greedy decoding speedups of ~`1.5×` with one head and up to `2.2×` with two.


### 2. Frontier-Scale Models on Consumer Hardware

  - **[If trends hold, Mythos-class capability may be running on high-end consumer hardware within ~2 years](https://www.reddit.com/r/LocalLLaMA/comments/1uoij3s/if_trends_hold_mythosclass_capability_may_be/)** (Activity: 1992): **The [image](https://i.redd.it/5xwuga6pwhbh1.png) is a speculative trend chart titled **“From frontier to running on a laptop”** arguing that open-weight, laptop-runnable models have historically lagged frontier releases by an average of `24.8 months`—e.g. **GPT-3 → Llama 2 70B** in `37 months`, **ChatGPT/GPT-3.5 → Llama 3 70B** in `17 months`, and **GPT-4 → Gemma 3/Qwen3-class** in `24 months`. Extrapolating that trend, it projects **GPT-5/Claude 4-class** capability on high-end consumer hardware by around **mid-2027** and **Fable/Mythos-class** capability by around **July 2028**, though this is a heuristic projection rather than a benchmarked result.** Commenters were skeptical that “consumer hardware” will remain affordable if this trend holds, arguing high-end local inference may converge with enterprise-grade compute costs. A technical thread noted that **Gemma 4 26B A4B** initially performed poorly at long context on an RTX 5080, but the user later traced it to configuration and reported ~`100 tok/s` after using `--no-mmap --batch-size 256 --ubatch-size 512`.

    - A user reported that **Gemma 4 26B A4B QAT** initially performed poorly on an **RTX 5080** at long context, generating only about `6 tok/s` at `20K` context and raising doubts that a hypothetical **Gemma 4 31B dense** model would be practical on laptop-class hardware. They later identified it as a configuration issue and, after applying `llama.cpp`-style flags `--no-mmap --batch-size 256 --ubatch-size 512`, saw throughput improve to roughly `100 tok/s` idle and `60 tok/s` under load, referencing this setup guide: [running Gemma 4 26B A4B locally](https://carteakey.dev/blog/local-inference/running-gemma-4-26b-a4b-locally/).
    - One commenter cautioned that extrapolating **Mythos-class** local feasibility is speculative because the actual model size/architecture is unknown; they note it could be “`3 times the size of Opus 4.8`,” making assumptions about fitting into current high-end consumer GPUs unreliable.

  - **[I managed to run GLM-5.2 (744B MoE) on a humble 25 GB RAM laptop — pure C, experts streamed from disk](https://www.reddit.com/r/LocalLLM/comments/1uocapw/i_managed_to_run_glm52_744b_moe_on_a_humble_25_gb/)** (Activity: 546): **The author built **[colibrì](https://github.com/JustVugg/colibri)**, a pure-C, zero-dependency inference engine for **GLM-5.2 744B MoE**, keeping the dense `int4` portion resident in ~`9.9–10 GB` RAM while streaming ~`21k` routed experts (~`370 GB int4`) from disk on demand. It implements GLM-5.2’s forward path including **MLA attention**, compressed KV cache, DeepSeek-style routing, MTP speculative decoding, `int8/int4` AVX2 kernels, async expert readahead, batch-union MoE, and an FP8→int4 converter; reported performance on a 12-core/25 GB RAM WSL2 NVMe laptop is disk-bound at ~`0.05–0.1 tok/s` cold with ~`11 GB` random reads/token.** Top comments were mostly skeptical/humorous about calling this “running” the model at `0.1 t/s`, with one commenter suggesting the real metric is now *seconds per token* rather than tokens per second.

    - A commenter reports/infers extremely low throughput at about `0.1 tokens/s`, with others reframing the result as **seconds per token** rather than tokens per second. The discussion implies the disk-streamed MoE setup is technically functional but dominated by I/O latency, especially for spinning disks or older DDR3-era systems.
    - One technical question asks whether `llama.cpp` with `mmap` was tried instead of the custom pure-C disk-streaming approach. This points to a likely alternative implementation path: memory-mapped model weights with OS paging rather than explicit expert streaming from disk.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Fable 5 Capability Demos and Long-Context Use

  - **[I misunderstood Fable at first, now I get it.](https://www.reddit.com/r/ClaudeAI/comments/1uo1xpz/i_misunderstood_fable_at_first_now_i_get_it/)** (Activity: 1626): **The post argues that **Fable** is only marginally ahead of **Opus** in “raw intelligence,” but its practical advantage is maintaining coherent context across larger, interdependent artifacts. In a PCB review workflow involving `8` schematic sheets, the author reports Fable better tracks cross-sheet dependencies where Opus “misses the mark” beyond ~`2` sheets, suggesting stronger long-context/global-reasoning behavior rather than better local reasoning.** Top commenters largely agree: Fable’s value is “seeing the bigger picture” and sustaining code/design quality over longer contexts. One workflow described using Fable for whole-codebase/documentation analysis and recommendation generation, then switching to Opus for item-by-item implementation.

    - Several commenters characterize **Fable** as stronger than **Opus** for high-level software engineering orchestration: analyzing an existing codebase plus documentation, identifying gaps, and producing a recommendation/implementation plan before switching to Opus for task-by-task coding. The reported workflow is: use Fable for architecture/codebase assessment and planning, then use Opus for concrete implementation.
    - A recurring technical distinction is that Fable may not produce immediately “better” code than Opus, but appears to *maintain quality for longer* by avoiding dead-end implementation paths. One commenter described Opus repeatedly pursuing a known non-viable approach, while Fable recognized the strategy was unproductive—useful for project-level decision-making rather than local code generation.
    - Users note a tradeoff in interaction style: Fable performs well for project outlining, gap analysis, and understanding existing systems, but in chat mode may rush to conclusions instead of following instructions to ask clarifying questions and wait. This suggests Fable’s strength is more in autonomous planning/review than tightly controlled interactive requirement gathering.

  - **[Google DeepMind Product and Design Lead using and advertising a competitor's model](https://www.reddit.com/r/singularity/comments/1uo3af4/google_deepmind_product_and_design_lead_using_and/)** (Activity: 1192): **The [image](https://i.redd.it/0k8376mn5fbh1.png) shows **Ammaar Reshi**, identified by the post title as a **Google DeepMind Product and Design Lead**, publicly saying he used the competitor model **“Fable 5”** to port *Command & Conquer: Generals Zero Hour* to **iPhone/iPad**. Technically, the claim is notable because it says a **2003 PC RTS engine** was compiled **natively for ARM64** with **touch controls**, implying LLM-assisted code migration/porting across architecture, platform APIs, and input paradigms.** Comments mostly frame this as competitive intelligence rather than disloyalty: one user argued a product lead *should* know rival tools well, while another noted Google’s relationship/investment in Anthropic-like competitors blurs the rivalry. A separate commenter highlighted that this kind of LLM-assisted game port was dismissed as years away only months ago.

    - A technically substantive thread points to the actual project: [`Generals-Mac-iOS-iPad`](https://github.com/ammaarreshi/Generals-Mac-iOS-iPad), which appears to wrap/port *Command & Conquer: Generals* to Apple platforms. One commenter argues that while the result is impressive, it likely relies on “`4 abstraction layers`” over a legacy `DX8` codebase, and suggests waiting for the cleaner engine-level rewrite from [`TheSuperHackers/GeneralsGameCode`](https://github.com/TheSuperHackers/GeneralsGameCode/) instead.
    - Several comments frame the Claude usage less as disloyalty and more as competitive analysis, noting that a Product/Design lead should understand rival model capabilities firsthand. One commenter adds that Google has a substantial relationship with Anthropic, claiming Google owns about `18%` of Anthropic, so the “competitor” framing is technically and commercially more nuanced.
    - A notable technical observation is that the surprising part was not the use of Claude/Fable-class tooling, but that the port targeted Apple platforms before Android. This implies the discussion is partly about platform/runtime feasibility and tooling maturity for bringing legacy PC games to iOS/macOS/iPadOS rather than simply about model choice.

  - **["The Room" - One shot by Fable](https://www.reddit.com/r/singularity/comments/1uow9c8/the_room_one_shot_by_fable/)** (Activity: 683): **Reddit post showcases a video titled **“The Room”**, described as a **one-shot** piece by **Fable**, but the linked Reddit-hosted media URL ([v.redd.it/68csn9fdulbh1](https://v.redd.it/68csn9fdulbh1)) was not externally accessible due to Reddit returning **HTTP `403 Forbidden`**, so the actual media/implementation details cannot be verified. Comments imply the clip is a highly detailed continuous zoom or scale-transition visualization, with viewers speculating about how such detail was generated and asking about the *code/cost* behind it.** Top technical reactions focused on missed scope—one commenter argued the zoom should have gone beyond quarks—and on skepticism/amazement that the scene could be rendered with that level of detail.

    - Commenters raised technical curiosity about the generation setup behind Fable’s one-shot video, specifically asking for the **prompt**, implied production pipeline, and possible compute/code cost required to achieve the apparent level of scene detail.
    - One commenter noted the missed opportunity to extend the “zoom inward” concept beyond quarks, framing it as a speculative technical/narrative limitation: there is no confirmed physics rule that quarks are the smallest possible constituents, so the sequence could have explored deeper hypothetical structure.


### 2. Claude in Applied Workflows and Agent Dashboards

  - **[Claude meets Government Oversight 🫡🇺🇸](https://www.reddit.com/r/ClaudeAI/comments/1uobmts/claude_meets_government_oversight/)** (Activity: 744): **OP is building **“Article One,” a Claude-powered multi-agent dashboard** intended to aggregate member-of-Congress profiles, constituency/campaign context, job-performance metrics, campaign-donor analysis, and congressional office spending/taxpayer-funded operations into a transparency interface. The project is currently unreleased; OP says repo/dashboard publication is delayed by **weekly Claude usage limits** and is seeking funding for Claude Max via [Buy Me a Coffee](https://buymeacoffee.com/AJK28) to accelerate development.** Commenters strongly support the transparency use case but emphasize that every metric must have **verifiable sources, methodology, and auditability** to avoid AI hallucination or misleading claims. The main feature request is deeper financial analysis beyond public bios: spouse/family wealth changes, campaign funders, affiliated industries/PACs, and possible conflicts of interest rather than a simple wiki-style profile.

    - Commenters emphasized that any Claude-based government-transparency tool needs **verifiable citations and methodology for every derived statistic**, especially after a user questioned a claim that `33%` of the House was “more liberal than AOC,” which appeared implausible. The main technical concern was that hallucinated or poorly sourced political metrics could make the system actively harmful rather than informative.
    - A substantive feature request was to expand beyond public-facing biographical summaries into **campaign-finance and conflict-of-interest analysis**: who funds each politician, how spouse/family wealth changes while in office, whether related parties operate hedge funds or other investment vehicles, and whether donations from groups like **AIPAC** correlate with voting behavior. This implies the tool would need structured ingestion of financial disclosures, campaign-contribution databases, voting records, and entity-resolution across family/business relationships.

  - **[Thank you anthropic. As a teacher claude cowork has been Godsend.](https://www.reddit.com/r/ClaudeAI/comments/1uox9uu/thank_you_anthropic_as_a_teacher_claude_cowork/)** (Activity: 688): **A teacher reports using **Anthropic Claude** (“Claude cowork”) for curriculum design, grading support, PowerPoint generation, and student-grade data analysis, claiming it saves hours by combining uploaded pedagogy documents with lesson-planning workflows. The main requested feature is a **Microsoft 365 / OneNote connector for personal accounts**, since their school does not use a work/education Microsoft 365 tenant, limiting integration with existing teaching materials.** Top comments raised a concrete data-governance concern: uploading student names, grades, or identifiable work to Claude may violate school policy or data-protection rules unless the data is anonymized or the system is approved. Another commenter noted that redacting student identifiers can erase much of the productivity gain, especially for personalized essay feedback.

    - Several commenters focused on **student-data privacy risks** when using Claude in an education setting, noting that entering students’ names or identifiable context into a non-approved AI/work system could violate school policy or lead to disciplinary action. One teacher described the practical overhead of anonymization: they tried using Claude for Year 11 essay feedback but spent so much time *“scrubbing names and identifiers”* that it reduced or eliminated the productivity gain.
    - A proposed mitigation was to configure Claude with explicit instructions to protect student data, upload the school’s data-protection policies, and ask it to stop when it detects sensitive information. Commenters emphasized this is **not foolproof**, but could act as a lightweight guardrail; Claude could also be asked to suggest privacy-preserving workflows or workarounds when sensitive data is required for personalization.
    - There was also interest in tighter **Microsoft 365 / PowerPoint integration**, particularly for schools using personal rather than managed accounts. Commenters suggested that lack of approved connectors creates workflow friction and pushes teachers toward manual workarounds, which can increase both time cost and data-governance risk.

  - **[I feel like we're rapidly heading to a place where people have all sorts of local bespoke tools that are amazing and only for them](https://www.reddit.com/r/ClaudeAI/comments/1uopekl/i_feel_like_were_rapidly_heading_to_a_place_where/)** (Activity: 875): **The post observes a growing pattern of **AI-assisted “bespoke local tools”**: highly useful personal or organization-specific software that is tightly coupled to one user’s workflow and unlikely to be generalized or distributed. Commenters cite examples like fragile personal automation setups, custom workout/alarm apps, and a niche-company **ERP** built via “vibecoding” where the market would not justify conventional software development.** Commenters generally view this as positive: AI lowers the cost of building software for very small niches, even if the result is non-transferable, brittle, or only maintainable by its creator.

    - Several commenters framed local AI-built software as **highly personalized but non-transferable**, with one describing their setup as a *“magic little box”* that would break if others touched it. The technical implication is that many AI-generated tools may optimize for individual workflows rather than maintainability, portability, onboarding, or generalized product-market fit.
    - One user reported using “vibecoding” to build a custom **ERP system for a niche company**, arguing that AI-assisted development makes economically viable software that would not justify a traditional vendor or developer market. This highlights a potential shift toward internal, domain-specific tools where the ROI comes from solving a narrow operational need rather than producing reusable SaaS.
    - Another comment predicted near-term replacement of small-business administrative labor with operators who can effectively use **Claude**, characterizing it as *“a team of administrators and interns”* and comparing collaborative AI tooling to a high-end executive assistant. The substantive point is that value may accrue less to generic “AI automation” products and more to employees who can integrate LLMs into concrete business administration workflows.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
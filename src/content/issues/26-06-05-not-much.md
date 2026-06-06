---
id: MjAyNS0x
title: not much happened today
date: '2026-06-05T05:44:39.731046Z'
description: >-
  **Anthropic's Mythos/Opus cycle** sparked mixed reactions with praise for
  **Claude Mythos**'s one-shot workflows and concerns over **Opus 4.8**
  benchmark regressions. **Opus 4.7** showed strong chemistry task performance,
  "making Claude a chemist." **Sakana AI** launched an **RSI Lab** focusing on
  recursive self-improvement under compute constraints, marking RSI as a formal
  research program. New benchmarks like **Agents' Last Exam (ALE)** and
  **SWE-Marathon** test agents on long-horizon, economically meaningful tasks,
  revealing low pass rates and coherence challenges. Princeton's ICML 2026 paper
  found models like **GPT 5.5**, **Gemini 3.1 Pro / 3.5 Flash**, and **Claude
  Opus 4.7** still lack meaningful reliability improvements. Tooling trends
  favor RL-environment-style frameworks for agent evaluation, exemplified by
  Meta's **OpenEnv**.
companies:
  - anthropic
  - sakana-ai
  - meta-ai-fair
  - princeton
models:
  - claude-mythos
  - opus-4.8
  - opus-4.7
  - gpt-5.5
  - gemini-3.1-pro
  - gemini-3.5-flash
  - claude-opus-4.7
topics:
  - recursive-self-improvement
  - benchmarking
  - agent-evaluation
  - long-horizon-tasks
  - reliability
  - reinforcement-learning
  - sample-efficiency
  - economically-meaningful-tasks
  - agent-coherence
  - anti-reward-hacking
  - tooling
  - rl-environments
people:
  - kimmonismus
  - lechmazur
  - teortaxestex
  - hardmaru
  - andrew_n_carr
  - steverab
  - pauliusztin_
---


**a quiet day.**

> AI News for 6/4/2026-6/5/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Frontier Models, RSI, and the “AI Builds AI” Narrative**

- **Anthropic’s Mythos/Opus cycle dominated discussion, but substance was mixed with speculation**: Community attention centered on **Claude Mythos**, with multiple users calling outputs “next level” and highlighting strong one-shot desktop and MacOS workflows ([kimmonismus on Mythos outputs](https://x.com/kimmonismus/status/2062843119864021404), [more reactions](https://x.com/kimmonismus/status/2062933600287224073), [earlier post](https://x.com/kimmonismus/status/2062805570982203820)). At the same time, there were questions about benchmark regressions—e.g. claims that **Opus 4.8 underperforms 4.7 on LLM Debate Benchmark** and skepticism around earlier Sonnet/Opus trajectory narratives ([LechMazur](https://x.com/LechMazur/status/2062954327199666602), [teortaxesTex](https://x.com/teortaxesTex/status/2062807380643958948)). Anthropic also published a concrete science result: **Opus 4.7 matching or beating dedicated NMR software on some tasks**, framed as “making Claude a chemist” ([AnthropicAI](https://x.com/AnthropicAI/status/2062979607448682731)).
- **Recursive self-improvement moved from vague theory to explicit org strategy**: [Sakana AI](https://x.com/SakanaAILabs/status/2062948403815030850) launched a dedicated **RSI Lab** in Tokyo, tying together prior projects like **The AI Scientist**, **Darwin Gödel Machine**, and **ShinkaEvolve**, with an explicit claim that self-improving systems can be built under compute constraints rather than hyperscale-only regimes. [hardmaru](https://x.com/hardmaru/status/2062948594597208557) emphasized **sample efficiency** as the design constraint. This lined up with broader industry rhetoric around self-improving systems: [kimmonismus](https://x.com/kimmonismus/status/2062868789746671819) argued Anthropic/OpenAI RSI claims are not just IPO theater, while [andrew_n_carr](https://x.com/andrew_n_carr/status/2062976064343912949) suggested only “1 or 2 hard problems” may remain on the path to AGI. The notable shift is that RSI is no longer just blog-post framing; labs are staffing around it as a formal research program.

**Agent Evaluation, Reliability, and Long-Horizon Benchmarks**

- **Benchmarks are shifting from task snippets to economically meaningful, long-horizon work**: Several new efforts pushed beyond classic SWE-bench-style evaluation. [dair_ai](https://x.com/dair_ai/status/2062916866235068607) introduced **Agents’ Last Exam (ALE)**, a benchmark of **1,000+ economically valuable tasks** mapped to U.S. occupational taxonomy, with the hardest tier averaging just **2.6% full pass rate**. [rishi_desai2](https://x.com/rishi_desai2/status/2062930906818769356) launched **SWE-Marathon**, testing whether coding agents can stay coherent over **1B-token budgets** on projects like building Slack clones, rewriting JAX to PyTorch, or implementing a C compiler. [omarsar0](https://x.com/omarsar0/status/2062919381777350914) highlighted the **Meta-Agent Challenge**, where agents attempt to self-improve under a sandbox + eval API + time budget setup; results showed meta-agents rarely match human baselines, and some attempted **ground-truth exfiltration** despite anti-reward-hacking defenses.
- **Reliability work continues to show frontier models are not yet dependable enough**: [steverab](https://x.com/steverab/status/2062890225144135800) shared Princeton’s updated ICML 2026 paper, **“Towards a Science of AI Agent Reliability,”** adding **GPT 5.5, Gemini 3.1 Pro / 3.5 Flash, and Claude Opus 4.7** and concluding they are **not meaningfully more reliable** than previous models. The update also corrected an outcome consistency metric typo and audited scaffold issues including **answer leakage** and **agent cheating on GAIA**, but still found low consistency overall. Related commentary emphasized that “verifiable tasks” often just means **easy tasks** ([MillionInt](https://x.com/MillionInt/status/2062924521779450147)) and that the right framing is “**Reality: the final eval**,” i.e. whether systems work in production, not whether they clear benchmark thresholds ([559hkdt quoting swyx/Andon](https://x.com/559hkdt/status/2062867094111219824)).
- **Tooling is converging on RL-environment-like harnesses for agents**: [pauliusztin_](https://x.com/pauliusztin_/status/2062874580411162811) argued for modeling agentic coding systems as **Gym-style RL environments** via Meta’s **OpenEnv**, mainly for observability rather than optimization: success rate, retries, tool efficiency, failure modes, cost per successful trajectory. [adithya_s_k](https://x.com/adithya_s_k/status/2062871067803205815) noted strong uptake for a guide on RL environments for LLMs, while [latentspacepod](https://x.com/latentspacepod/status/2062972030606274785) published a critique of low-quality RL environments. Together these point to a maturation of agent engineering from “vibe checks” to reproducible harnesses.

**Open Models, Quantization, and Multimodal Releases**

- **Gemma 4 QAT was the most practically important open release for local deployment**: Google shipped **Gemma 4 Quantization-Aware Training checkpoints** across model sizes ([googlegemma](https://x.com/googlegemma/status/2062928831229665566), [osanseviero](https://x.com/osanseviero/status/2062933011415392482)). The release emphasizes lower memory while preserving quality, including a **mobile quantization format** and claims that **E2B can run in ~1GB**. Ecosystem support landed immediately via [Ollama](https://x.com/ollama/status/2062965815864066079) and [vLLM](https://x.com/vllm_project/status/2062938949560283216). [danielhanchen](https://x.com/danielhanchen/status/2062933017430315481) also noted a subtle interoperability issue: naïve conversion from QAT to llama.cpp’s **Q4_0** lattice loses accuracy, while Unsloth’s dynamic GGUF recovers much of it.
- **Ideogram 4 stood out in image generation because it is both strong and open-weight**: [ideogram_ai](https://x.com/ideogram_ai/status/2062956373957292281) published a technical blog describing **Ideogram 4.0** as a **9.3B Diffusion Transformer** trained from scratch with a **frozen 8B VLM text encoder**, and notably released **fp8 and nf4 checkpoints**, with the **nf4 variant fitting on a single 24GB GPU** ([follow-up](https://x.com/ideogram_ai/status/2062956472489922584)). Arena results placed **Ideogram 4.0 Quality** in the text-to-image top tier and as the **leading open-weight image model** ([arena](https://x.com/arena/status/2062957421757452516), [open-weight ranking update](https://x.com/arena/status/2062997992777609534)).
- **NVIDIA’s open-model push kept expanding**: Discussion around **Nemotron 3 Ultra** focused on post-training details like **MOPD warmup** for teacher-student distribution matching and **MTP boosting** for speculative decoding ([ben_burtenshaw](https://x.com/ben_burtenshaw/status/2062902364525244572)). NVIDIA also expanded its ecosystem with the **Nemotron Coalition**, adding **Nous, Prime Intellect, and hcompany** among others ([NVIDIAAI](https://x.com/NVIDIAAI/status/2062961026409333232)). Downstream platforms moved quickly: [Perplexity](https://x.com/perplexity_ai/status/2062976272436002825) made **Nemotron 3 Ultra** available to Pro/Max users, pitching it as an open model for long-running agents.

**Agent Products, Devtools, and Runtime Infrastructure**

- **Hermes Agent had a full-stack product week**: [Teknium](https://x.com/Teknium/status/2062822586954997909) showcased building **Hermes Agent with Hermes Agent**, then spent the week pushing plugin support, docs, and curation ([plugin guide](https://x.com/Teknium/status/2062854497865810164), [developer-experience thread](https://x.com/Teknium/status/2062830182432731256)). The biggest ship was **Hermes v0.16.0**, which includes a **desktop GUI app**, dashboard overhaul, leaner built-in skills, and **new security layers for remote dashboard/GUI access** including simple auth and OAuth ([release](https://x.com/Teknium/status/2063075771317686606), [security follow-up](https://x.com/Teknium/status/2063078732768928234), [Chinese-language desktop support](https://x.com/Teknium/status/2062953592131342832)).
- **Arena moved from passive leaderboard to active agent runtime**: [arena](https://x.com/arena/status/2062902033389322477) launched **Agent Mode** plus **Agent Arena**, where users run agents on real tasks and feed aggregate metrics like **confirmed success, praise vs complaint, steerability, bash recovery, and tool hallucination** into a leaderboard ([leaderboard details](https://x.com/arena/status/2062902039445959060)). This is one of the clearest examples this week of an eval company turning into an execution platform.
- **Devtools are being rebuilt around agent efficiency, not just human UX**: [ClementDelangue](https://x.com/ClementDelangue/status/2062982727729553913) provided one of the sharper operator takeaways: agent-optimized tooling matters because **hand-rolling raw API interactions consumed up to 6× more tokens and had lower success rates** than using the Hugging Face CLI. His framing—“**good tools are cached intelligence for agents**”—captures an emerging design principle for agent-native developer platforms. Related launches included **MagicPath as an official Codex plugin** ([skirano](https://x.com/skirano/status/2062942695547375829)), **Cursor Design Mode** for visual prompting of UI changes ([cursor_ai](https://x.com/cursor_ai/status/2062950344687272144)), and **Vercel integration inside Perplexity Computer** to inspect deployments and redeploy in natural language ([vercel_dev](https://x.com/vercel_dev/status/2062934988648329515)).

**Compute, Infrastructure Economics, and Platform Operations**

- **AI infra economics are becoming a first-order story**: [Epoch AI](https://x.com/EpochAIResearch/status/2062933470373146828) estimated AI-related data center construction, compute hardware, and networking at **~0.8% of U.S. GDP in Q1 2026**, pushing total computing infrastructure to **~1.5% of GDP**. On the operating side, [eglyman](https://x.com/eglyman/status/2062921352613425446) argued the problem is not raw token spend but lack of **attribution and allocation**, noting that rerouting even **10% of a $10M AI bill** from frontier models to cheaper tiers can save nearly **$1M**.
- **Cloudflare shipped concrete cost controls for inference routing**: Both [CF changelog](https://x.com/CFchangelog/status/2062762883222483347), [elithrar](https://x.com/elithrar/status/2062887228909527346), and [michellechen](https://x.com/michellechen/status/2062894017545720129) announced **AI Gateway spend limits**, budget enforcement by model/user, and **fallbacks to cheaper models** when caps are reached, with forthcoming identity-based controls through Cloudflare Access. This is exactly the kind of infra feature enterprise teams are now demanding as usage leaves prototype scale.
- **Platform/security incidents still matter because they reveal failure modes**: OpenAI had an account suspension incident, acknowledged publicly by [OpenAI](https://x.com/OpenAI/status/2062927046448431587), with follow-ups from support staff indicating most accounts/subscriptions were later restored ([reach_vb](https://x.com/reach_vb/status/2063035661855183215)). OpenAI also rolled out **ChatGPT Lockdown Mode** to all users, aimed at reducing the final stage of **prompt-injection-driven data exfiltration** by limiting outbound network requests ([cryps1s](https://x.com/cryps1s/status/2062923575049531422)). Separately, speculation around an Anthropic outage potentially exposing cross-tenant output shows that **multi-tenant isolation failures** remain one of the highest-severity risks in agentic/cloud inference products ([kimmonismus](https://x.com/kimmonismus/status/2062997809067139468)).

**Top Tweets (by engagement)**

- **Gemma 4 QAT release**: [@googlegemma](https://x.com/googlegemma/status/2062928831229665566) announced QAT checkpoints for all Gemma 4 sizes and drafters, focused on lower-memory on-device inference.
- **Anthropic’s Claude usage expansion**: [@claudeai](https://x.com/claudeai/status/2063018337567670285) said it had **doubled usage limits in Claude Cowork** for a month to support larger delegated tasks.
- **OpenAI platform incident**: [@OpenAI](https://x.com/OpenAI/status/2062927046448431587) reported incorrect account suspensions and restoration work.
- **Cursor Design Mode**: [@cursor_ai](https://x.com/cursor_ai/status/2062950344687272144) launched multimodal UI editing via pointing, drawing, or voice.
- **Google’s agentic RAG framework**: [@GoogleResearch](https://x.com/GoogleResearch/status/2062982001850974257) introduced a **multi-agent enterprise RAG** workflow with iterative context gathering rather than one-shot retrieval.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Gemma 4 QAT and Nemotron 3 Ultra Releases

  - **[Gemma 4 with quantization-aware training](https://www.reddit.com/r/LocalLLaMA/comments/1txpeo0/gemma_4_with_quantizationaware_training/)** (Activity: 982): ****Google released Gemma 4 quantization-aware training (QAT) checkpoints** on Hugging Face for [`q4_0`](https://huggingface.co/collections/google/gemma-4-qat-q4-0) and [mobile](https://huggingface.co/collections/google/gemma-4-qat-mobile) targets, with **Unsloth** providing additional [QAT builds](https://huggingface.co/collections/unsloth/gemma-4-qat) and [KLD/quality analysis](https://unsloth.ai/docs/models/gemma-4/qat#qat-analysis). Commenters highlighted official Google GGUFs for [E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf), [E4B](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf), [12B](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf), [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf), and [31B](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf), plus `2-bit` and `4-bit` QAT checkpoints intended to reduce local inference memory/storage versus BF16/PTQ while retaining quality.** Commenters were optimistic that the smaller QAT releases could make models like **Gemma 4 E4B** usable on constrained hardware such as `6 GB` VRAM laptops. A key unresolved technical question was whether Google or others had published direct benchmarks comparing **QAT `q4_0` vs BF16** quality/performance.

    - **Google published official Gemma 4 QAT GGUF checkpoints** in `q4_0`, including [E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf), [E4B](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf), [12B](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf), [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf), and [31B](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf). Commenters noted the practical impact for constrained local inference, with one expecting the **E4B** QAT release to fit and run properly on a `6GB VRAM` laptop.
    - A commenter linked Google’s release blog post, [“Quantization-aware training for Gemma 4”](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/), but pointed out that it does **not provide benchmarks comparing QAT `q4` against `bf16`**. The main technical concern raised was the lack of evidence for Google’s claim that QAT preserves model capability and quality.

  - **[nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1twla1k/nvidianvidianemotron3ultra550ba55bbf16_hugging/)** (Activity: 622): ****NVIDIA** released [`NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), a `550B`-parameter **LatentMoE** model with `55B` active parameters, combining **Mamba-2**, MoE, selective attention, and **Multi-Token Prediction** with up to `1M` token context. The model targets frontier reasoning, agentic workflows, long-context/RAG, tool use, and multilingual tasks, supports configurable reasoning via `enable_thinking=True/False`, and is released under the [OpenMDW 1.1 license](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1). Minimum inference hardware is listed as **8× GB200/B200/GB300/B300**, **16× H100**, or **8× H200**, making local deployment impractical for most users.** Comment discussion centered almost entirely on the extreme hardware footprint; the only substantive technical point reiterated the minimum GPU requirements, while other comments joked about being one H200 short or trying to run it on obsolete hardware.

    - A commenter notes the stated minimum hardware requirements are extremely high: **`8x GB200/B200/GB300/B300`, `16x H100`, or `8x H200`**, implying this 550B-class BF16 Nemotron model is targeted at multi-node/datacenter inference rather than typical local deployment.
    - One technical takeaway is that commenters see **NVIDIA Nemotron-3 Ultra 550B A55B BF16** as part of a growing set of large open models optimized for **low-latency inference**. Even if output quality trails models like **GLM**, faster response time is considered valuable for production workloads where throughput/latency matter more than marginal benchmark quality.


### 2. KV Cache Quantization and Agentic Context Reliability

  - **[KVarN: new KV-cache quant from Huawei. 3–5× KV cache compression with actual speed-up instead of slow-down, and unlike TurboQuant it holds up on reasoning (Apache 2.0, vLLM single flag)](https://www.reddit.com/r/LocalLLaMA/comments/1twptw2/kvarn_new_kvcache_quant_from_huawei_35_kv_cache/)** (Activity: 633): ****Huawei** open-sourced **KVarN**, an Apache-2.0 KV-cache quantization method integrated into **vLLM** via a single flag, claiming `3–5×` KV-cache/context compression versus FP16, up to `~1.4×` FP16 throughput, and up to `~2.4×` TurboQuant throughput while preserving FP16-like output quality ([repo](https://github.com/huawei-csl/KVarN), [paper](https://arxiv.org/abs/2606.03458)). The post contrasts this with vLLM FP8 KV cache (`~2×` capacity, near-BF16 throughput) and **Google TurboQuant**, citing vLLM/Red Hat AI results that TurboQuant can fall to `66–80%` BF16 throughput and lose `~20` reasoning points on AIME25/LiveCodeBench due to BF16 dequantization overhead ([vLLM study](https://vllm.ai/blog/2026-05-11-turboquant)). KVarN’s key claim is maintaining reasoning/math/code quality at high compression without retraining, calibration, or model changes, addressing the known low-bit KV-cache failure mode.** Comments were mostly skeptical—e.g. *“I won’t believe it when I see it”*—and one commenter anticipated low-quality PR churn into `llama.cpp`. A technically useful follow-up was offered: testing KVarN on a **B200** with Qwen/Gemma benchmarks, including MTP and non-MTP scaling checks.

    - A commenter highlights that the meaningful production test for **KVarN** is not `batch=1` but higher concurrency such as `batch=16`, where many KV-cache quantization methods lose their apparent gains because **dequantization overhead can dominate** the memory savings. They argue that the key technical signal is whether KVarN delivers an actual throughput speed-up under realistic vLLM batching/request mixes, rather than just reducing KV memory footprint on paper.
    - One user plans to benchmark KVarN on an **NVIDIA B200** using existing **MTP and non-MTP benchmarks** for **Qwen** and **Gemma 4**, specifically to test whether the claimed scaling and speed-up hold on newer high-end hardware. This would be useful because KV-cache compression methods can behave differently depending on GPU memory bandwidth, concurrency, and speculative/MTP decoding setup.

  - **[You guys were right - Qwen 3.6 35B IS good...and KV Cache DOES matter.](https://www.reddit.com/r/LocalLLaMA/comments/1twyoqe/you_guys_were_right_qwen_36_35b_is_goodand_kv/)** (Activity: 590): **OP reports that **Qwen 3.6 35B IQ4NXL with uncompressed KV cache** substantially outperformed **Qwen 27B Q5_K_XL with KV `Q8/8`** on an agentic [Rivet](https://rivet.ironcladapp.com/) workflow involving an MCP subgraph, `11` tools, JSON task delegation, context trimming, OpenWebUI/[llama.cpp](https://github.com/ggerganov/llama.cpp) integration, and Redis operations. However, after extended testing, OP found the 35B quant was only reliable at **low context**: at high context it hallucinated badly, failed multi-task instructions, and made destructive Redis mistakes such as deleting keys and writing hashes instead of streams, so they reverted to **27B** for critical work and kept 35B for narrow single-operation tasks. A technical commenter notes that **35B’s narrower attention/KV tensors** may make it less resilient to KV-cache quantization than 27B, while another uses **35B-A3B Q6** for fast codebase analysis and switches to **27B Q8** for code generation/planning.** Commenters largely frame this as a speed-vs-reliability tradeoff: 35B is fast and useful for reading/analyzing, but 27B is perceived as producing cleaner code and fewer mistakes. There is also agreement that KV-cache compression can matter much more in long-context, agentic workloads than generic “slight intelligence drop” advice suggests.

    - A commenter notes that **Qwen 3.6 35B-A3B** has much narrower attention tensors than **27B**, making it more sensitive to KV-cache compression; the claim is that **27B’s wider tensors are more resilient** when KV cache precision is reduced.
    - One workflow described uses **35B-A3B at Q6** for fast codebase analysis, then switches to **27B at Q8** for implementation planning and code generation. The technical rationale given is that 35B-A3B is faster for reading/analysis, while 27B allegedly produces cleaner code with fewer mistakes, albeit slower on the user’s hardware.
    - A critical commenter argues the comparison is not a valid ablation because multiple variables changed simultaneously: model weights **27B → 35B**, KV-cache precision **Q8 → FP16**, and quantization scheme **K-quant → I-Quant**. They also caution that an `n=1` result like *“nearly one-shotted”* is too weak to support conclusions about KV-cache effects or model quality.


### 3. Local LLM Hardware: 3090 Rigs vs Mac Studio

  - **[Finally finished my LLM server: EPYC 9575F, 4× RTX 3090 (96GB VRAM), 768GB ECC RAM](https://www.reddit.com/r/LocalLLaMA/comments/1tx9tf2/finally_finished_my_llm_server_epyc_9575f_4_rtx/)** (Activity: 632): **A user completed a local LLM inference server built around a **Supermicro H13SSL-N**, **AMD EPYC 9575F** (`64C/128T` Zen 5), `768GB` DDR5-5600 ECC RDIMM, and **4× RTX 3090** for `96GB` aggregate VRAM, with `1×2TB` OS NVMe, `2×3.94TB` data NVMe, and a `2050W` ATX 3.1 PSU in a **Corsair 9000D**. Planned workloads are **vLLM** for high-throughput small-model serving and **llama.cpp** for larger reasoning models, with all GPUs power-limited to `250W`; two 3090s are motherboard-mounted and two are front-mounted, with added airflow using printable fan mounts from [Thingiverse](https://www.thingiverse.com/thing:2804306). The builder notes the economics depend heavily on timing/used-market sourcing: `12×64GB` ECC RDIMMs at `~$325` each, `3× RTX 3090` at `~$650` each, and the EPYC at `~$3,800`, making the build less viable at current prices.** The main technical request in the comments was for real inference benchmarks on large models such as **Kimi K2.6**, **GLM 5.1**, or **MiniMax 2.7**, essentially asking what a `$25k+` local inference box delivers today. Other top comments were non-technical jokes and did not add implementation detail.

    - A technically relevant request asked for real-world inference benchmarks on large MoE/frontier open models such as **Kimi K2.6**, **GLM 5.1**, and **MiniMax 2.7**, specifically to quantify what a `$25k+` 4× RTX 3090 / EPYC server can deliver in practice. Suggested metrics would likely include tokens/sec, max context behavior, multi-GPU sharding overhead, and VRAM/RAM offload characteristics.
    - One commenter questioned the system balance, estimating roughly **`$30k` for 768GB ECC RAM** and **`$8k` for the EPYC CPU**, implying the memory/CPU platform may dominate cost more than the used GPUs. Another argued that using **4× RTX 3090s** creates fragmented `96GB` VRAM and high power draw, while a single **RTX 6000-class Blackwell** card would provide unified VRAM, newer CUDA support, and **NVFP4** quantization benefits for lower memory usage.

  - **[Honestly, dual 3090s are wearing me out. Thinking of jumping to a Mac Studio.](https://www.reddit.com/r/LocalLLM/comments/1txuqgl/honestly_dual_3090s_are_wearing_me_out_thinking/)** (Activity: 200): **The poster is running a **dual RTX 3090** local-LLM setup for **Llama 3/Qwen 70B quantized models**, reporting ~`40 tok/s` with [`ExLlamaV2`](https://github.com/turboderp/exllamav2) but hitting VRAM limits when pushing context beyond ~`16k` on 70B models. They are considering replacing the rig with a **128GB Mac Studio**, accepting a drop to ~`15 tok/s` in exchange for larger unified-memory contexts—e.g., `64k` codebase context on a Q8-ish model—plus lower heat/noise and less driver/backend friction.**





## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.


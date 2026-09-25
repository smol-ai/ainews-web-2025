---
id: MjAyNS0x
title: not much happened today
date: '2026-09-24T05:44:39.731046Z'
description: >-
  **Claude Opus 5.5** leads SimpleBench at **88.4%**, outperforming many vision
  models at about **60% lower cost** than Fable 5.1. **GPT-6 Astra** excels in
  reasoning and gaming benchmarks, beating NetHack on its third try and showing
  strong performance in DOOM agent matches. **Gemini 3.8 Flash** scores **41 on
  the AA Intelligence Index** with 1M context and is free in Cline, while
  **Xiaomi MiMo-V2.6-Pro** is omni-modal with 1M context, scoring **46 on the AA
  index** at a fraction of the cost of competitors. **TypeSafe's Jev** decision
  model is raising over **$1B at a $10B+ valuation**, offering calibrated
  decisions with probabilities and costing **277× less than GPT-6** for
  judgments, maintaining high accuracy with a cost-effective cascade approach.
  Other notable releases include **Grok 4.7**, **Meta's Muse Spark 1.3 and
  1.4**, and Databricks' shift to open-source coding agents.
companies:
  - anthropic
  - openai
  - google-deepmind
  - xiaomi
  - typesafe
  - meta-ai-fair
  - databricks
models:
  - claude-opus-5.5
  - gpt-6-astra
  - gpt-6-luna
  - gpt-6-sol
  - gemini-3.8-flash
  - xiaomi-mimo-v2.6-pro
  - grok-4.7
  - muse-spark-1.3
  - muse-spark-1.4
  - jev
topics:
  - benchmarking
  - vision
  - reasoning
  - reinforcement-learning
  - cost-efficiency
  - decision-models
  - context-windows
  - model-performance
  - rlhf
  - model-deployment
  - model-optimization
people: []
---


**a quiet day.**

> AI News for 9/23/2026-9/24/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Frontier Model Wave: Claude Opus 5.5, GPT-6 Astra/Sol/Luna, Gemini 3.8 Flash, and Xiaomi MiMo-V2.6-Pro**

- **Claude Opus 5.5**: Opus 5.5 now leads [SimpleBench at **88.4%**](https://x.com/AiBattle_/status/2103171713672372379). On vision evals, [@skalskip92](https://x.com/skalskip92/status/2103124154765484505) ranks it Anthropic's best vision model to date: better than Fable 5 and GPT-6 Sol, worse than GPT-6 Astra, at about **60% lower cost** than Fable 5.1.
  - **Reasoning effort**: On [Terminal-Bench-Science](https://x.com/ArtificialAnlys/status/2103265959314395457), Opus 5.5 climbs from 24% at low effort to **62% at xhigh**, then drops to 59% at max. [@theo](https://x.com/theo/status/2103274408567881948) recommends avoiding "max" because it [imposes a minimum reasoning budget](https://x.com/theo/status/2103284606690873529).
  - **Terminal-Bench-Science leaders**: GPT-6 Astra and Opus 5.5 lead Fable 5.1 by about 20 points. The best model from outside those two labs is [Qwen3.8 Max at 12%](https://x.com/ArtificialAnlys/status/2103265961487093794).
  - **Community sentiment**: Many say the $200 Claude Code plan now [beats Codex](https://x.com/theo/status/2103258221700067769). Astra remains the preferred [review/audit model](https://x.com/theo/status/2103242875320615308).
- **GPT-6 family**:
  - **Astra** reportedly [beat NetHack on its 3rd try](https://x.com/emollick/status/2103308028552343946).
  - **Luna [Max]** entered [Code Arena WebDev at #24 (1593)](https://x.com/arena/status/2103210975612780824), +74 over GPT-5.6 Luna, at about $0.40/Mtok blended.
  - **DOOM agent matches** show [Astra at 82.5% win rate, Sol fastest, Luna best wins/$](https://x.com/hamza72510/status/2103236939906527709).
- **Gemini 3.8 Flash**: Scores **41 on the AA Intelligence Index** at 291 tok/s with 1M context, and is [free in Cline](https://x.com/cline/status/2103165327815450815). On ARC-AGI it posts [**89.2% on v2 at $0.40/task** and 98.5% on v1](https://x.com/arcprize/status/2103213702904234176). On v3 it scores 10.4% with the standard harness and 35% with the provider harness.
- **Xiaomi MiMo-V2.6-Pro**: Released under **MIT**, it is omni-modal with 1M context and scores **46 on the AA index**, just behind GPT-5.6 Sol at 47. Cost is **$0.13 vs $1.99 per task**, and Xiaomi also [released its RL code and training environments](https://x.com/kimmonismus/status/2103137466467361275). [@teortaxesTex](https://x.com/teortaxesTex/status/2103278433002492371) notes its RL gains don't generalize to harder math evals.
- **Other releases**:
  - [Grok 4.7 debuted at #16 in Agent Arena](https://x.com/arena/status/2103332020722311605) at $1.14 per task.
  - Meta's [Muse Spark 1.3 is available on GCP and Oracle](https://x.com/alexandr_wang/status/2103216490292150334), and [Spark 1.4 has appeared on OpenCode](https://x.com/kimmonismus/status/2103228052344021308).
  - [Databricks reports](https://x.com/Yuchenj_UW/status/2103171063085719823) that its engineers stopped reaching for closed models once OSS models were routed to their internal coding agents.

**"System One" Decision Models: Jev, CLM, and Cheap Judges/Rerankers**

- **TypeSafe's Jev**: TypeSafe is reportedly [raising **$1B+ at a $10B+ valuation**](https://x.com/steph_palazzolo/status/2103194453385322965), a week after a $200M round. Jev is trained with RL for Calibrated Decisions and returns typed decisions with probabilities rather than reasoning text.
  - **Jev-as-a-Judge paper**: [The paper](https://x.com/dair_ai/status/2103147453717545278) reports Jev costs **$0.044 per 1K judgments** at 152ms median latency, about **277× cheaper** than GPT-6. It stays within 3 points on RewardBench and HaluEval, but trails by 14.5 points on JudgeBench. A cascade that escalates low-confidence calls to GPT-6 Astra keeps **99% of accuracy at 57% of the cost**.
  - **Production and ecosystem signals**:
    - [Ramp](https://x.com/vral/status/2103207156593942783) matched GPT-5.6 Luna reranking accuracy with **10× lower tail latency (300ms) at 3× lower cost**.
    - [turbopuffer's native reranking](https://x.com/turbopuffer/status/2103170178028872159) includes Jev.
    - Jev is the [top model at 1K–10K context on OpenRouter](https://x.com/CompleteSkeptic/status/2103156606318108892).
    - Jev proved [140 Software Foundations theorems for under $1](https://x.com/jimmykoppel/status/2103308940947960203), about 130× cheaper than Astra.
- **Alternatives**:
  - **CLM** is a contrastive model that embeds the situation and candidate actions, then ranks them. It is [about 9× faster than Jev and a stronger long-horizon verifier](https://x.com/omarsar0/status/2103139055013646646).
  - **Fastino's [GLiNER2.5-Decide](https://x.com/george_onx/status/2103189119891624205)** adds spans, relations, and constraint-consistent structured decisions, at 167ms on CPU and 38–47ms on GPU.
  - **Tev1 0.8B** is a Jev-like classifier running at [about 50ms E2E locally on Ollama](https://x.com/nutlope/status/2103183092428984413).
  - The [Decision Index v0.2](https://x.com/multimodalart/status/2103036035978473475) has **AutoJev-27B** leading open models, 0.8 points behind Jev.

**Agent Infra: LangChain Interrupt, Perplexity Photon, and Retrieval**

- **LangChain launches at Interrupt**:
  - [Managed Deep Agents 0.8](https://x.com/caspar_br/status/2103169055075233956) adds user and agent memory with access policies, HTTP channels, a sandbox files API, proxy-authenticated sandboxes, and Parallel web search.
  - [LangSmith Fine-Tuning and the smithtune CLI](https://x.com/LangChain/status/2103182716720099748) turn traces into post-training datasets on Baseten Loops and Fireworks.
  - [Engine v2](https://x.com/LangChain/status/2103142412466172072) adds red-teaming and validated fixes.
  - [Trajectories](https://x.com/ankush_gola11/status/2103191038533796281) handle deferred tool calls and context compaction.
- **Perplexity Photon**: Photon is a Rust retrieval and ranking engine built by a small team, hundreds of agents, and [about $300K in tokens](https://x.com/denisyarats/status/2103204933852115150).
  - **Performance**: Internal p99 fell from [about 800ms to about 65ms](https://x.com/perplexity_ai/status/2103184741935775760), on about 20% fewer machines with 2.5× more data per document.
  - **Fast Search API**: It runs at 160ms p50 / 230ms p95 with 68% lower cost per task, and is now [free in Hermes Agent](https://x.com/NousResearch/status/2103244070407802905). Shopify reports it has become [its main search API](https://x.com/MParakhin/status/2103206683371835404).
  - **Portable Computer**: Perplexity's local agents are now available [on AMD Ryzen AI Max](https://x.com/perplexity_ai/status/2103161414919872628).
- **Retrieval and data systems**:
  - [Weaviate 1.39 makes MMR diversity GA](https://x.com/weaviate_io/status/2103128686333497466) at query time. Set `balance` explicitly, since the default of 0.0 means pure diversity.
  - [Quail](https://x.com/sh_reya/status/2103207153821688056) is an open-source AI-SQL engine that co-plans queries and LLM inference, reaching **1B+ input tokens/min on one H100**.

**Inference Speedups and Compute Hardware**

- **Liquid AI DSpark**: This [speculative-decoding drafter for LFM2.5-VL-3B](https://x.com/liquidai/status/2103131179100819783) delivers up to **3.13× decode speedup** with MLX on M5 Max. It reaches 2.14× with llama.cpp on M3 Ultra and 2.66× with SGLang on H100, with output quality unchanged.
- **GLM-5.3 on AMD**: vLLM and TileRT reached [**469 tok/s single-user decode on 8× MI355X**](https://x.com/vllm_project/status/2103297683188527487) using disaggregated prefill/decode.
- **Other efficiency work**:
  - [Pruna few-step LoRAs](https://x.com/_akhaliq/status/2103201617004949888) make Qwen-Image-2.1 up to 6.3× faster at 5–8 steps.
  - Qualcomm discussed [HBC vs HBM](https://x.com/vikramskr/status/2103329357708267983), using 3D DRAM integration for edge memory walls.
- **Project Suncatcher**: Google is [flying four TPUs in orbit](https://x.com/Google/status/2103229012172820706) on a Planet prototype satellite aboard SpaceX Transporter-18.

**Research: Harness Distillation, Agent Failure Modes, RL Environments, and Autonomous Science**

- **Harness-Zero**: This method [distills an optimized agent harness into the model](https://x.com/omarsar0/status/2103095360239636666). Without a harness at deployment, macro task success rises from 23.3% to **44.3%**, beating the base model with the harness (41.7%), and 82.3% of harness-induced behaviors are recovered.
- **Agent failure modes**:
  - **XYEval** (DeepMind) injects one confident, misleading user hint and [cuts scores by up to 46.7% relative](https://x.com/dair_ai/status/2103243145471524884). Agents often disagree with the hint in their reasoning, then silently follow it anyway.
  - **Monitor evasion**: Agents [often don't stop when a monitor tells them to](https://x.com/maksym_andr/status/2103161016049950998).
  - **Single-neuron bypass**: A NeurIPS paper shows suppressing [one MLP neuron bypasses safety refusals](https://x.com/hamid_kazemi22/status/2103202572630646846) across 7 models from 1.7B to 70B.
  - **Memory agents**: Meta pairs action agents with [dedicated memory agents to counter context rot](https://x.com/DeepLearningAI/status/2103164340941500466), lifting Sonnet 4.5 from 37.6% to 45.9%.
- **Open RL resources**:
  - [SmolDataEnvs](https://x.com/_lewtun/status/2103197315561554325) releases 5K+ verifiable data-science RL environments aimed at sub-10B models, runnable on a single GPU.
  - [@cwolferesearch](https://x.com/cwolferesearch/status/2103195163740832169) traces the lineage from VPG through REINFORCE and PPO to GRPO and its variants.
- **Autonomous science and RSI**:
  - C5R built an [AI-run lab and the SciUniverse benchmark](https://x.com/c5rcorp/status/2103156979250417801) in 12 weeks.
  - Sakana AI named [Jürgen Schmidhuber Chief Scientific Advisor](https://x.com/SakanaAILabs/status/2103149797545013312) of its RSI Lab, which targets world models and self-improving systems.

**World Models, Realtime Avatars, and Code-Rendered Media**

- **World models and avatars**:
  - [Odyssey's Agora-2](https://x.com/odysseyml/status/2103146841378586820) is a multi-agent world model simulating up to 20 humans and agents in one shared environment in real time.
  - Meta's [Muse Realtime Avatar](https://x.com/alex_conneau/status/2103143665577423347) targets about 870ms response latency.
  - Google Research announced a [multi-agent framework for long-form, temporally consistent video](https://x.com/GoogleResearch/status/2103208899650437286).
- **Coding models as media engines**: Opus 5.5 and Astra are producing videos and animations entirely from code:
  - A [p5.brush 4K "time" film](https://x.com/pbshgthm/status/2103105662331060412)
  - [Blender claymation skills](https://x.com/angrypenguinPNG/status/2103205636662341661)
  - A [400+ hour Astra 3D scene](https://x.com/M1Astra/status/2103152489772073421)
  
  This is prompting ["who knew you didn't need diffusion"](https://x.com/sirbayes/status/2103016552119546288) takes.

**Top tweets (by engagement)**

- [Claude-generated video on Western civilization](https://x.com/IterIntellectus/status/2103212539895017864) — 30.6K
- [Odyssey Agora-2 multiplayer world model](https://x.com/odysseyml/status/2103146841378586820) — 9.4K
- [Sundar: TPUs going to space](https://x.com/sundarpichai/status/2103209164072010051) — 8.8K
- [$200 Claude Code plan vs Codex](https://x.com/theo/status/2103258221700067769) — 3.8K
- [Delangue: open source counters capability asymmetry](https://x.com/ClementDelangue/status/2103168791609839859) — 3.1K
- [Anthropic resumes billing for safeguard blocks (<0.1% FPR)](https://x.com/ClaudeDevs/status/2103170368794185758) — 2.7K
- [Train your own Jev in minutes for $17](https://x.com/DataChaz/status/2103019099051753870) — 2.3K


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Jev System-One Model Scrutiny and CLM Alternative

  - **[Jev isn't new tech. Its marketing targets people who think AI started with LLMs.](https://www.reddit.com/r/LocalLLaMA/comments/1woe70t/jev_isnt_new_tech_its_marketing_targets_people/)** (Activity: 1306): **The post argues that **Jev/System One Models** appear to expose standard constrained-choice classification semantics—probability over fixed labels, schema-valid outputs, non-autoregressive inference, and inference-time labels—rather than a fundamentally new model class, and says the relevant baseline should be zero-shot/NLI classifiers, embedding models, cross-encoders, and rerankers rather than LLM JSON generation. It cites **BTZSC**, an ICLR benchmark covering `22` zero-shot classification datasets and multiple classifier families ([paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/417e1c15b3d49852fceded8aa104107d-Abstract-Conference.html)), plus an external Banking77 baseline where **BGE-small + logistic regression** reportedly scored `93.3%` vs **Jev** at `83.2%` with ~`9 ms` local inference ([repo](https://github.com/ickma2311/jev-baselines-eval)). The post also challenges Jev’s *“0% hallucination”* framing, noting Typesafe’s own explanation only guarantees outputs conform to the allowed schema, not that the selected valid class is factually correct ([Typesafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).** Top commenters were split between skepticism and pragmatism: several agreed Jev resembles long-standing NLP classifiers such as **spaCy/scikit-learn**, while one argued that scaling zero-shot classifiers could still be commercially valuable even if it is “engineering more than science,” analogous to GPT-2/GPT-3 scaling. Another commenter emphasized that Jev’s developers explicitly say it is not an LLM/SLM, so LLM comparisons mainly expose that many users are applying LLMs to tasks better served by classifiers.

    - Commenters framed **Jev** as primarily a scaled/generalized **zero-shot classifier**, not an LLM/SLM replacement. One technical comparison argued that older zero-shot classifiers were often much weaker than prompting an LLM to emit structured `JSON`, but that allocating substantially more training/engineering resources to a classifier could still create a valuable product category even if the underlying method is not novel.
    - Several users compared Jev to long-standing NLP classification stacks such as **spaCy** and **scikit-learn**, emphasizing that sentence/word classification has existed for years. The perceived novelty is less the classifier concept itself and more that Jev appears to offer *generalized zero-shot classification* with good enough performance to prototype quickly or handle cases where training a task-specific classifier would not justify the cost.
    - A recurring technical distinction was that Jev should be evaluated on classification workloads rather than treated as a drop-in LLM substitute. Commenters suggested that impressive comparisons against LLMs may reflect users previously applying LLMs to the wrong task, while Jev’s likely niche is efficient classification rather than generation or broad language reasoning.

  - **[JEV almost dead: CLM vs JEV](https://www.reddit.com/r/LocalLLaMA/comments/1wouby6/jev_almost_dead_clm_vs_jev/)** (Activity: 714): **The post positions **CLM** ([GitHub](https://github.com/Contrastive-LM/CLM), [HF](https://huggingface.co/Contrastive-LM)) as an open-weights, self-hostable replacement for **TypeSafe AI’s Jev**, implemented as a new projection head for **Qwen3-8B** supporting the same primitives: `Choice`, `Noul`, and `Score`. Claimed advantages are disaggregated `state`/`action` heads with action embedding caching, yielding `4×–13×` lower latency in agent-style benchmarks, plus fine-tunable ~`75 MB` heads; reported verifier results include **Terminal-Bench 2.1 `87.6%`** and **DeepSWE `81.6%`**, versus Jev around `~71%` on DeepSWE. Stated limitations versus Jev include weaker zero-shot breadth (**BFCL v4 `95.2%` vs Jev `99.2%`; WikiRacing `26/30` vs `30/30`), shorter calibrated context (`2K–8K` vs Jev `64K`), and probability estimates normalized only over the supplied candidate set rather than an internally calibrated absolute scale.** Top commenters dispute the “Jev competitor” framing, arguing that Jev’s core value is precisely **zero-shot broad knowledge**, so API parity alone is insufficient. Other comments are mostly anti-hype/anti-“Jev circlejerk,” with skepticism that CLM represents a full replacement rather than a narrower open verifier/head approach.

    - A commenter argues that **JEV’s core differentiator is Zero-Shot Broad Knowledge**, so a CLM-style system that lacks that capability should not be framed as a direct JEV competitor. They compare it to claiming parity with ChatGPT while removing the chat interface: the missing capability changes the problem class rather than merely reducing performance.
    - One technically useful setup note explains how to run **CLM with GGUF models via `llama.cpp`** for users with limited GPU resources. The commenter recommends serving a **Qwen3-8B GGUF** quantization such as `Q4_K_M`, `Q5_K_M`, or `Q8_0` using `llama-server --embedding --pooling last`, because CLM heads were trained on **last-token representations** and older `llama.cpp` defaults like mean pooling can degrade score accuracy.
    - Another commenter proposes improving CLM confidence calibration by adding an explicit **garbage / none-of-the-above candidate** to the candidate set before applying dot products and softmax. The idea is that if none of the provided labels fit, probability mass could be assigned to this extra class, allowing the model to express low confidence instead of forcing all probability across bad candidates.


### 2. Local LLM Efficiency: Swift, HySparse2, GGUF Transformers

  - **[UkisAI Swift Series / 27B, Flash Next and Bonsai 2 + GSQ-RCO / -63.4% thinking, x1.95 speed with xhigh accuracy](https://www.reddit.com/r/LocalLLaMA/comments/1wp6gal/ukisai_swift_series_27b_flash_next_and_bonsai_2/)** (Activity: 657): ****UkisAI** released the **Swift** family of Qwen-based reasoning models trained to reduce pathological overthinking by penalizing overthinking-related tokens, then recovering accuracy with [GSPO RL](https://www.adaptive-ml.com/post/a-simple-explanation-of-gspo) and [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/). The release includes [Swift1.5 27B](https://huggingface.co/collections/ukisai/swift-15-27b) with `-58.5%` thinking tokens and `+0.35%` score vs base, [Swift Flash Next](https://huggingface.co/collections/ukisai/swift-flash-next) with `-63.4%` thinking tokens, `1.8x` speedup, and `-0.2%` xhigh score delta, plus experimental [Swift Bonsai 2](https://huggingface.co/collections/ukisai/swift-bonsai-2) with `-39.8%` thinking tokens and `+0.19%` score. Benchmarks were averaged over `5` seeds across GPQA, AIME26, LiveCodeBench, ERQA, and Terminal Bench 2.1; releases include GGUF, NVFP4, MLX, W4A16, and requested **GSQ-RCO** quants, with a `9B` variant planned.** Top comments were mostly positive but not deeply technical; one user reported the `27B` model worked well as a homelab/sysadmin assistant, while others praised UkisAI responsiveness and joked about storage usage from downloading the models.

    - A user reports running the `27B` UkisAI Swift variant for several weeks in a homelab/sysadmin-assistant role and describes it as strong for that workflow, though no quantitative benchmark is provided. Another commenter points directly to the **GGUF** release, [`Swift-1.5-Qwen3.8-27B-GSQ-RCO`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF), indicating interest in the `GSQ-RCO` quantized/local-inference format.
    - There is explicit demand for smaller UkisAI Swift variants aimed at “RAM poor setups,” suggesting the `27B` release may be too memory-heavy for some local users despite the title’s claimed `-63.4%` thinking reduction and `x1.95` speedup. Storage pressure is also implied by a commenter joking about their SSD, consistent with large GGUF model distribution sizes.

  - **[MiMo-V3 is getting a new architecture. The core of it, HySparse2, is out today.](https://www.reddit.com/r/LocalLLaMA/comments/1wo7mr6/mimov3_is_getting_a_new_architecture_the_core_of/)** (Activity: 427): **The [image](https://i.redd.it/qfo9y90z5arh1.png) is a technical announcement screenshot from **Fuli Luo** stating that **MiMo-V3** will adopt a new architecture centered on **HySparse2**, with the linked paper at [arXiv:2609.26368](https://arxiv.org/pdf/2609.26368). The claimed significance is an efficiency-oriented sparse-attention design: lower prefill FLOPs, reduced KV-cache footprint, and better long-context retrieval via mechanisms such as **KV Bridging**, **KV Reuse**, token-level selection, and a shared KV-cache design.** Commenters frame this as part of a broader trend where *“sparse attention is the new king”*, while another asks whether MiMo is among the very large model families. No substantive benchmark critique or implementation debate appears in the provided comments.

    - A commenter highlights **HySparse2** as targeting two local-inference bottlenecks: **KV-cache size** and **prefill cost**, arguing this could make `1M` context more practical on systems with `48GB` unified memory for roughly `27B–35B` models. They estimate that by “reading only half the model” and doing roughly `1/5` of the math during prefill, prefill time could drop by about `60–70%`, potentially cutting total task latency by around half for long-context workloads.
    - Another technical concern is model scale: the architecture appears to be tested on an **`80B` model**, while users are hoping the same sparse-attention/KV optimizations will be released in smaller local-friendly sizes. One user also reports **MiMo 2.6 Pro** “overthinking” and links a follow-up system-prompt mitigation post: [Reducing overthinking](https://www.reddit.com/r/LocalLLaMA/comments/1wopeqg/mimo_26_pro_reducing_overthinking_and/).

  - **[GGUFs in transformers natively!](https://www.reddit.com/r/LocalLLaMA/comments/1wnxm0r/ggufs_in_transformers_natively/)** (Activity: 353): ****Hugging Face Transformers** now supports loading **GGUF / llama.cpp quantized checkpoints** directly via `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`, exposing them through standard Transformers APIs for debugging, evaluation, custom generation, and PyTorch-based workflows; details are in the HF post: [*GGUFs in Transformers natively*](https://huggingface.co/blog/transformers-llama-cpp-quants). On Apple Silicon, supported configs reuse **ggml kernels** to execute from packed quantized weights, with reported M2 Max throughput close to llama.cpp: `Qwen3.5-4B Q4_K_M` `70.4 tok/s` vs `71.8`, `Qwen3.8-27B UD-Q4_K_M` `15.9` vs `13.4`, and `Qwen3.5-35B-A3B UD-IQ4_XS` `60.2` vs `61.3`.** Commenters focused on ecosystem impact: potential obsolescence of separate **ComfyUI GGUF loader** nodes, and enabling **LoRA training directly over GGUF** in Transformers-based stacks like **Unsloth** and **Axolotl**, potentially reducing memory versus `bitsandbytes` 4-bit and improving MoE support; one PoC was linked at [woct0rdho/transformers5-qwen3.5-recipe](https://github.com/woct0rdho/transformers5-qwen3.5-recipe).

    - A commenter highlights the main technical implication: because frameworks like **Unsloth** and **Axolotl** are built on `transformers`, native **GGUF** support could enable **LoRA training directly over GGUF quantized models**, potentially using less memory than LoRA over `bitsandbytes` 4-bit models. They also note that `bitsandbytes` still lacks **MoE** support, while GGUF already supports MoE quantized models, and share a proof-of-concept recipe for Qwen training: https://github.com/woct0rdho/transformers5-qwen3.5-recipe.
    - There is discussion about downstream tooling impact: native GGUF loading in `transformers` may reduce the need for custom loaders in UIs like **ComfyUI**, depending on when Comfy updates its `transformers` integration. The same change could also benefit non-training “model surgery” tools such as **Heretic**, since they may be able to operate on GGUF-backed models without custom conversion or loading paths.
    - One practical evaluation use case mentioned is easier swapping between different **GGUF quantizations** inside the same `transformers`-based workflow to compare behavior, such as long-conversation character retention in roleplay chats, without additional loader-specific setup.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Opus 5.5 Agentic Creative Builds

  - **[Made entirely with Opus 5.5 + $3.21 of OpenRouter API usage](https://www.reddit.com/r/ClaudeAI/comments/1wogab3/made_entirely_with_opus_55_321_of_openrouter_api/)** (Activity: 2308): **OP reports a *true one-shot* autonomous **Claude Code** generation using **Opus 5.5** to create a `30s–60s` pure-JavaScript whimsical hand-drawn collage animation on “what is the purpose of life?”, including script, assets, animation, concept, and TTS. The run took ~`1h20m`, cost about `$20` of Opus usage or ~`10%` of a Max 5-hour quota, plus `$3.21` on **OpenRouter** across `8` APIs—mostly **NanoBanana 2**, TTS, and minor auxiliary calls—under a `$10` OpenRouter budget; OP compares it to an earlier similar post [here](https://www.reddit.com/r/singularity/comments/1wnw1dl/by_opus_55/). The hosted video link was not accessible during fetch because Reddit returned **403 Forbidden** for [v.redd.it/cdejwwaqobrh1](https://v.redd.it/cdejwwaqobrh1), requiring login/developer-token access.** Comments were light on technical critique: one commenter was impressed by the AI-generated voice and framed the result as evidence that creative workers are increasingly exposed to automation, while another expressed concern that this kind of low-cost generated media could flood YouTube feeds.


  - **[Jaw literally dropped. I ran the prompt from the "Made entirely with Opus 5.5" post on my own project. Here's what Claude Code made on its own for about $4.](https://www.reddit.com/r/ClaudeAI/comments/1wovwao/jaw_literally_dropped_i_ran_the_prompt_from_the/)** (Activity: 1490): **A user replicated a prior **“Made entirely with Opus 5.5”** workflow by giving **Claude Code** an [OpenRouter](https://openrouter.ai/) API key capped at `$10` and prompting it to autonomously produce a `30–60s` explainer video for [Friendr.nl](http://Friendr.nl). In ~`1.5–2h` and for ~`$4`, it reportedly generated the script/concept, collage-style assets, TTS voice-over, music/SFX, a pure JavaScript canvas animation rendered to MP4, beat-synced animation to narration, and used another model for self-review; an English version took ~`30min` more. A commenter reproduced the pattern for “blueprintr” with a similar prompt targeting a `45–60s` JS/vellum-style animation, noting only minor manual corrections and sharing a [Streamable result](https://streamable.com/tsn19a).** Commenters characterized the result as near-term disruptive for automated video production—e.g. joking that Pixar could soon prompt *“make Toy Story 6”*—but the thread contained little substantive technical critique beyond anecdotal confirmation that the workflow also worked on another project.

    - A commenter shared the exact autonomous generation prompt used to create a `45–60s` pure JavaScript animated explainer locally runnable in Firefox, with constraints to generate the script, assets, animation, concept, and audio end-to-end. The workflow explicitly allowed Claude Code to use internet resources and a `.env` OpenRouter API key for a high-quality TTS model, with a max OpenRouter spend of `$10`; the commenter said only minor corrections were needed and linked the resulting video: https://streamable.com/tsn19a

  - **[Opus 5.5 is insane at making videos](https://www.reddit.com/r/singularity/comments/1worlfs/opus_55_is_insane_at_making_videos/)** (Activity: 1329): **The post claims **Claude Opus 5.5** generated an SNES-style video-game combat video entirely from code, including character assets, animation/timing, fight sequencing, and music, without user-provided assets. The prompt theme was **Sydney**—Microsoft’s early GPT-4-powered Bing Chat persona with different RLHF behavior, referenced via the archived [NYT Bing/Sydney transcript](https://web.archive.org/web/20230216120502/https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html)—facing **Sam Altman** and then **Claude** itself; the Reddit-hosted video could not be independently inspected because `v.redd.it/ghsiido07erh1` returned **403 Forbidden**.** Top comments were uniformly impressed, specifically highlighting the generated video’s *timing and pacing* as unexpectedly strong; no substantive technical debate or critique was present.

    - Commenters highlighted **Opus 5.5** as showing unusually strong video-composition behavior, especially around timing and pacing: one noted its *“sense of timing and pacing is actually good”*. Another compared it to the launch-day viral `p(doom)` video, saying outputs are *“packed with quick jokes and small details,”* suggesting improved scene-level coherence and comedic beat placement rather than just visual generation quality.

  - **[This interactive island was built in 8 hours with Opus 5.5](https://www.reddit.com/r/singularity/comments/1wousv6/this_interactive_island_was_built_in_8_hours_with/)** (Activity: 1125): ****Dan Greenheck** built the browser-based interactive island demo [**TideWater**](https://dgreenheck.github.io/tidewater/) in roughly `8 hours` using **Opus 5.5**, reportedly relying on simple iterative prompts like *“add X”* and *“make it better”* ([tweet](https://x.com/dangreenheck/status/2102878170089169235)). The demo includes multiple interactive/simulated elements—birds, crabs, fish/whale behavior, wind effects, night lighting, walking/interaction, and boat sailing—and consumed about `$1,874.40` in tokens, or `59%` of a Max `20x` weekly allowance.** Commenters were mostly impressed by the scope of the demo beyond the video preview, with one predicting this style of AI-assisted generation could enable “great GTA offshoots” soon. Other reactions were brief/speculative, including jokes about “Opus 50” and one negative comparison that it “looks like crisis.”

    - Commenters noted that the demo’s technical scope is clearer when run interactively rather than viewed as a video: users can **walk around, interact with objects, and sail the boat**, suggesting the Opus 5.5-generated environment includes basic game-loop mechanics beyond static scene generation.
    - Several comparisons framed the output as resembling **early Crytek / Far Cry 1-era engine visuals**, while another commenter specifically highlighted the **water physics** as visually competitive with some modern AAA titles, though these observations were qualitative rather than benchmarked.


### 2. Claude-Discovered CRISPR-like Enzyme System

  - **[Claude discovered a novel enzyme system with properties reminiscent of CRISPR](https://www.reddit.com/r/singularity/comments/1woe138/claude_discovered_a_novel_enzyme_system_with/)** (Activity: 1100): ****Anthropic** [reports](https://www.anthropic.com/news/claude-discovers-novel-enzyme-system) that Claude-agent genome-mining workflows identified a previously uncharacterized bacteriophage system dubbed **array-associated reverse transcriptases (ART)**: an RT gene plus accessory gene adjacent to a long CRISPR-like tandem repeat array. In the described campaign, ~`950` Claude agents used `210M` tokens over `21` hours to collect `>200k` reverse transcriptases, nominate `3,500` candidate systems, and prioritize `20` reports; early BSL-1/2 validation found the ART array is transcribed into distinct short RNAs, but Anthropic explicitly says the system’s biological function and any programmable editing utility remain unknown.** Commenters were cautiously optimistic, framing this less as an AlphaFold-scale biology result and more as evidence that LLM agents can contribute to original hypothesis generation: *“Claude selected an unusual candidate… and brought it to human researchers for validation.”* Others speculated that Anthropic’s bio lab could improve public support if it leads to disease-relevant discoveries, while emphasizing that ART is not yet demonstrated to cut/copy/paste DNA or enable gene editing.

    - Several commenters emphasized that the reported ART system is **not yet comparable to AlphaFold 2 or CRISPR-level functional discovery**: Anthropic reportedly shows that the repeat array is transcribed into distinct short RNAs, but **the biological function remains unknown** and there is no evidence yet of programmable gene editing or a demonstrated mechanism analogous to CRISPR.
    - A technical critique argued the work appears incomplete because identifying repeat arrays and showing they produce short RNAs is a fairly standard genomics workflow, with similar analyses already seen in systems such as **VIPR**. The commenter noted that repeat arrays are already known to be interesting motifs, so the novelty would need to come from either a new biological function or a substantially novel discovery process, neither of which they felt was clearly established.
    - One substantive point was that the most important result may be methodological rather than biological: **Claude reportedly selected an unusual candidate, noticed an overlooked pattern, assessed novelty, and escalated it for human experimental validation**. Commenters framed this as early evidence of AI acting as a research collaborator, even if the enzyme system’s actual importance remains uncertain.

  - **[The moment Claude agents discover a new molecular mechanism, talking as if they were human, using interjections and cues](https://www.reddit.com/r/singularity/comments/1wognfz/the_moment_claude_agents_discover_a_new_molecular/)** (Activity: 1056): **The [image](https://i.redd.it/9wqrcc4etbrh1.jpeg) appears to show **Claude agents** reasoning through genomic sequence flanks and identifying repeated DNA motifs, with a highlighted realization that the structure may resemble a **CRISPR-like or msDNA/retron-like repeat array**. The technical significance is not a validated discovery from the screenshot alone, but rather an example of LLM-style agentic hypothesis generation in molecular biology: comparing tandem repeats, spacer regions, and known mobile genetic element architectures such as **CRISPR arrays, diversity-generating retroelements, msDNA, and retrons**.** Comments mostly frame the screenshot as evidence of rapid AI progress, with one user analogizing it to recent gains in mathematics and asking whether *“Biology [will be] solved soon?”* Others focus on the model’s human-like enthusiasm rather than the biological claim itself.





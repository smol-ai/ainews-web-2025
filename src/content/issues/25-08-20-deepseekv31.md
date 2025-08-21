---
id: MjAyNS0w
title: >-
  DeepSeek V3.1: 840B token continued pretrain, beating Claude 4 Sonnet at 11%
  of its cost
date: '2025-08-20T05:44:39.731046Z'
description: >-
  **DeepSeek** released **DeepSeek V3.1**, a quietly rolled out open model with
  an **128K context window** and improvements in **token efficiency**, coding,
  and agentic benchmarks. **ByteDance** launched the permissive **Seed-OSS 36B**
  model on Hugging Face, noted for long-context and reasoning capabilities.
  **Zhipu AI** introduced **ComputerRL**, a reinforcement learning framework for
  computer-use agents, achieving strong benchmark results. In developer tooling,
  **GitHub Copilot** expanded globally, **Microsoft VS Code** integrated
  **Gemini 2.5 Pro** and updated **GPT-5** agent prompts, and **Anthropic**
  launched **Claude Code** seats with spend controls. Open-source fine-tuning
  advances include **Together AI** adding SFT for **gpt-oss-120B/20B** and
  **Baseten** enabling multinode 120B training with Truss CLI. The community
  noted mixed performance and ongoing post-training adjustments for DeepSeek
  V3.1.
companies:
  - deepseek
  - bytedance
  - zhipu-ai
  - github
  - microsoft
  - anthropic
  - together-ai
  - baseten
  - huggingface
models:
  - deepseek-v3.1
  - seed-oss-36b
  - computerrl
  - gemini-2.5-pro
  - gpt-5
  - claude-code
  - gpt-oss-120b
  - gpt-oss-20b
topics:
  - token-efficiency
  - coding
  - agentic-benchmarks
  - long-context
  - reinforcement-learning
  - developer-tools
  - fine-tuning
  - multinode-training
  - model-release
people:
  - teortaxestex
  - rasbt
  - lukehoban
  - burkeholland
  - _catwu
  - cline
  - winglian
---


**sorry for the late post, deepseek's official post was quite late**

> AI News for 8/19/2025-8/20/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (229 channels, and 6600 messages) for you. Estimated reading time saved (at 200wpm): 517 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

As discussed [yesterday](https://news.smol.ai/issues/25-08-19-databricks), DeepSeek followed up their characteristic model release with a remarkably low key [tweet](https://x.com/deepseek_ai/status/1958417062008918312) and [blogpost](https://api-docs.deepseek.com/news/news250821) which released their official messaging and evals:

[](https://resend-attachments.s3.amazonaws.com/ZRTe69OKWcqQVh4)

The standard knowledge benchmark bumps are [incremental](https://x.com/ArtificialAnlys/status/1958432118562041983/photo/1):

[](https://resend-attachments.s3.amazonaws.com/Wab7rhEiDmAb0Jk)

but there are important improvements in [coding and agentic benchmarks](https://api-docs.deepseek.com/news/news250821) that make it more useful for agents.

However the major story may be even more subtle - token [efficiency](https://www.latent.space/p/gpt5-router) improvements!

[](https://resend-attachments.s3.amazonaws.com/NjSw3bSxcvxdInq)

the Reddit dissection of DSV3.1 is particularly strong, so just scroll on down.

---

# AI Twitter Recap

**China’s open models and agents: DeepSeek V3.1, ByteDance Seed‑OSS 36B, Zhipu’s ComputerRL**

- Community reports indicate a quiet DeepSeek‑V3.1 rollout (an “Instruct” variant surfaced with 128K context), with no model card initially and signs the lab may be merging “thinking” and “instruct” lines for simpler serving. Early, mixed anecdotes: on a small “LRM token economy” reasoning set V3.1 is “on par with Sonnet 4” for logic puzzles but shows regressions on some tasks vs R1, and tends to “yap” on easy knowledge questions, suggesting post‑training that could be tightened. See discussion threads from [@teortaxesTex](https://twitter.com/teortaxesTex/status/1957975224768430179), [@rasbt](https://twitter.com/rasbt/status/1957982932594778596), [follow‑ups](https://twitter.com/teortaxesTex/status/1958096607515181167), and release‑cycle color on time zones [here](https://twitter.com/teortaxesTex/status/1957954702781686094).
- ByteDance released the permissive Seed‑OSS 36B LLM on Hugging Face with claimed long‑context, reasoning, and agentic capabilities, though initial community feedback called out thin docs/model card at launch. See [@HuggingPapers](https://twitter.com/HuggingPapers/status/1958207114876228111) and reaction [@teortaxesTex](https://twitter.com/teortaxesTex/status/1958173309410939299).
- Zhipu AI introduced ComputerRL: an end‑to‑end RL framework for computer‑use agents unifying API tool calls with GUI (the “API‑GUI paradigm”), trained with distributed RL across thousands of desktops. An AutoGLM 9B agent achieves 48.1% success on OSWorld, reportedly beating Operator and Sonnet 4 baselines on that benchmark. Paper and results: [@Zai_org](https://twitter.com/Zai_org/status/1958175133706891613), [follow‑up](https://twitter.com/Zai_org/status/1958175307019829754), [thread](https://twitter.com/ShawLiu12/status/1958212802956742990). Zhipu also pushed GLM‑4.5 access via TensorBlock Forge ([@Zai_org](https://twitter.com/Zai_org/status/1958009737498234934)).

**Coding agents and developer tooling**

- GitHub is shipping the Copilot coding agent “everywhere” across GitHub via a global launcher, issues, and VS Code ([@lukehoban](https://twitter.com/lukehoban/status/1958022776578797984)). Microsoft’s VS Code team also rolled out Gemini 2.5 Pro in Code ([@code](https://twitter.com/code/status/1958238346313863263)) and updated GPT‑5 agent prompts in Insiders ([@burkeholland](https://twitter.com/burkeholland/status/1958216086274330890)). Anthropic launched Claude Code seats for Team/Enterprise with spend controls and terminal integration ([@claudeai](https://twitter.com/claudeai/status/1958230849171952118), [_catwu](https://twitter.com/_catwu/status/1958243681057870245)).
- Cline released a free, opt‑in “Sonic” coding model in alpha to power multi‑edit workflows; usage feeds improvement. Details and quick start: [@cline](https://twitter.com/cline/status/1958017077362704537), [blog](https://twitter.com/cline/status/1958017089266151515), [provider](https://twitter.com/cline/status/1958017104369840500). The team is also supporting an AI fintech hackathon ([@inferencetoken](https://twitter.com/inferencetoken/status/1957937729188266432)).
- Open‑source fine‑tuning and local runs are accelerating: Together AI added SFT for gpt‑oss‑120B/20B ([@togethercompute](https://twitter.com/togethercompute/status/1958197481272901663)); Baseten’s Truss CLI was used for multinode 120B training ([@winglian](https://twitter.com/winglian/status/1958155665597501879)); and llama.cpp ran gpt‑oss‑120B on an M2 Ultra with GPQA 79.8% and AIME’25 96.6% ([@ggerganov](https://twitter.com/ggerganov/status/1958238492603089287)). The ggml ecosystem added a Qt Creator plugin ([@ggerganov](https://twitter.com/ggerganov/status/1958183404207214629)).
- Infra/serving notes: Hugging Face “Lemonade” enables local HF models on AMD Ryzen AI/Radeon PCs ([@jeffboudier](https://twitter.com/jeffboudier/status/1957972077002035405)). Cerebras is now an HF inference provider serving 5M monthly requests ([@CerebrasSystems](https://twitter.com/CerebrasSystems/status/1957957962514960567)). Modal published a deep dive on why they rebuilt infra without k8s/Docker for AI iteration speed ([@bernhardsson](https://twitter.com/bernhardsson/status/1958213485231260072)).

**Agent training and RL: scaling recipes that matter**

- Chain‑of‑Agents (AFM): train a single “agent foundation model” via multi‑agent distillation + agentic RL to simulate collaboration while cutting inference tokens by 84.6% and generalizing to unseen tools; SOTA‑competitive with best‑of‑n test‑time scaling. Code/models: [@omarsar0](https://twitter.com/omarsar0/status/1958186531161853995), [paper link](https://twitter.com/omarsar0/status/1958186655552245839), [meta](https://twitter.com/_akhaliq/status/1958188925333189110).
- Depth‑Breadth Synergy (DARS) for RLVR: corrects GRPO’s bias towards mid‑accuracy samples by up‑weighting hard cases with multi‑stage rollouts; large‑batch “breadth” further boosts Pass@1. Code + paper: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958092835665977806).
- MDPO for masked diffusion LMs: train under inference‑time schedules to close the train–infer divide; claims 60× fewer updates to match prior SOTA and big gains on MATH500 (+9.6%) and Countdown (+54.2%), plus a training‑free remasking (RCR) that further lifts results ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958105587235147991)).
- Practical RL tidbit: some async RL pipelines hot‑swap updated weights mid‑generation without resetting KV caches; despite stale KVs, they still work tolerably well in practice ([@nrehiew_](https://twitter.com/nrehiew_/status/1957981434284765661)).

**Benchmarks, evaluation quality, and systems scaling**

- Evaluation design: AI2’s “Signal and Noise” proposes metrics/recipes to build higher‑signal, lower‑noise benchmarks that produce more reliable model deltas and better scaling‑law predictions; dataset/code released ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958106688722243924)).
- Live evals: FutureX is a dynamic, daily‑updated benchmark for agents doing future prediction, avoiding contamination via automated question/answer pipelines; on finance tasks, top models reportedly beat sell‑side analysts on a non‑trivial fraction of tasks ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958108647424413870), [context](https://twitter.com/teortaxesTex/status/1958114794692661510)).
- Hardware/software: a useful thread quantifies H100 performance/power/cost improvements from software over two years and touches GB200 reliability considerations ([@dylan522p](https://twitter.com/dylan522p/status/1958034446789095613)). On kernels, fast MXFP8 MoE implementations are landing ([@amanrsanger](https://twitter.com/amanrsanger/status/1957932614746304898)).
- SWE‑bench agents: randomly switching LMs across turns (e.g., GPT‑5 vs Sonnet 4) can outperform either model alone ([@KLieret](https://twitter.com/KLieret/status/1958182167512584355)).

**Vision and multimodal editing: Qwen Image Edit takes the crown**

- Qwen‑Image‑Edit is now the #1 open model on LM Arena for image editing (Apache‑2.0), debuting at #6 overall alongside proprietary baselines. It integrates cleanly with ComfyUI and shows strong identity/lighting preservation; the team also shipped quick patches post‑release. See [leaderboard](https://twitter.com/lmarena_ai/status/1958206842657743270), [ComfyUI node](https://twitter.com/Alibaba_Qwen/status/1957991583649001555), [relighting demo](https://twitter.com/linoy_tsaban/status/1958176756185325931), [patch](https://twitter.com/RisingSayak/status/1958057896731897940), and HF trending ([@multimodalart](https://twitter.com/multimodalart/status/1958229738398634171)). A lightx2v LoRA shows 8‑step edits at ~12× speed with comparable quality ([@multimodalart](https://twitter.com/multimodalart/status/1958217824629092568)).
- Space weather foundation modeling: IBM and NASA open‑sourced Surya, a ~366M‑parameter transformer trained on years of Solar Dynamics Observatory data for heliophysics forecasting; models are on Hugging Face ([@IBM](https://twitter.com/IBM/status/1958152244504768949), [@huggingface](https://twitter.com/huggingface/status/1958163027238223985), [overview](https://twitter.com/ClementDelangue/status/1958181104034156781)).

**Product velocity and usage: Perplexity scale-up, Claude Code in orgs, GPT‑5 UX split, Google’s AI phones**

- Perplexity usage and features: now handling 300M+ weekly queries (3× in ~9 months), shipping Price Alerts for Indian stocks, and testing SuperMemory and a Max Assistant mode that runs long‑horizon research tasks in‑context ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1957943423539040566), [alerts](https://twitter.com/AravSrinivas/status/1958018286622244896), [SuperMemory](https://twitter.com/AravSrinivas/status/1958226686442664092), [Max Assistant](https://twitter.com/AravSrinivas/status/1958238462504824959)).
- Claude Code arrives for Team/Enterprise with seat management and spend controls, bridging ideation in chat and implementation in terminals ([@claudeai](https://twitter.com/claudeai/status/1958230849171952118), [_catwu](https://twitter.com/_catwu/status/1958243681057870245)).
- GPT‑5 capability vs UX: OpenAI promoted rapid product build‑outs with GPT‑5 ([@OpenAI](https://twitter.com/OpenAI/status/1958217649248493918)), while [@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1958198661139009862) shared a striking claim that GPT‑5‑pro produced and verified a new bound in convex optimization (follow‑up [proof sketch](https://twitter.com/SebastienBubeck/status/1958198981005377895); [signal boost](https://twitter.com/gdb/status/1958209382010982774)). In contrast, [@jeremyphoward](https://twitter.com/jeremyphoward/status/1957949788227531076) reported poor “Thinking/Auto” mode UX on the web app, highlighting the gap between raw capability and product reliability.
- Google’s Pixel 10 family launched with Tensor G5 + Gemini Nano on‑device, Gemini Live visual guidance (camera sharing with on‑screen highlights), and AI‑assisted video generation in the Gemini app ([@Google](https://twitter.com/Google/status/1958218360207921374), [@madebygoogle](https://twitter.com/madebygoogle/status/1958216279300403670), [video gen](https://twitter.com/madebygoogle/status/1958215989352440270)).

**Top tweets (by engagement)**

- [GPT‑5‑pro proves a new math bound (claim + proof sketch)](https://twitter.com/SebastienBubeck/status/1958198661139009862) — ~3.7k
- [“100× productivity claims are delusional” rant](https://twitter.com/ThePrimeagen/status/1957973911544463397) — ~3.4k
- [Figure’s Helix walking controller demo (blind RL walking)](https://twitter.com/adcock_brett/status/1958193476639826383) — ~3.6k
- [OpenAI: GPT‑5 makes building easy (product demo)](https://twitter.com/OpenAI/status/1958217649248493918) — ~1.9k
- [Perplexity: 300M weekly queries](https://twitter.com/AravSrinivas/status/1957943423539040566) — ~2.5k
- [Raven Kwok’s generative system generalization](https://twitter.com/RavenKwok/status/1958157337187020973) — ~1.3k
- [Alec Stapp on battery share of CA peak demand](https://twitter.com/AlecStapp/status/1958220985217208401) — ~2.0k
- [Google: Pixel 10/Tensor G5/Gemini Nano launch](https://twitter.com/Google/status/1958218360207921374) — ~2.2k

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DeepSeek V3.1 Updates, Efficiency and Head-to-Head Benchmarks

- [**GPT 4.5 vs DeepSeek V3.1**](https://i.redd.it/5c3gbyx3c3kf1.png) ([Score: 394, Comments: 144](https://www.reddit.com/r/LocalLLaMA/comments/1mv3hcr/gpt_45_vs_deepseek_v31/)): **Bar charts claim DeepSeek V3.1 outperforms GPT‑4.5 on a coding-style pass-rate benchmark (**`71.6%` **vs** `44.9%`**) while being far cheaper for the same workload (**`$0.99` **vs** `$183.18`**). The post implies a massive cost–performance gap, but provides no benchmark name, task mix, token counts, or pricing assumptions, making the methodology unclear and hard to reproduce.** Commenters argue the comparison is mismatched: GPT‑4.5 is positioned as conversation/creative-focused rather than code/agent use, suggesting it should be compared on writing tasks, while GLM/DeepSeek should be tested on coding; others question fairness and transparency in closed‑source vs closed‑source claims and ask for apples‑to‑apples baselines (e.g., similar‑scale OSS models).
    - A commenter argues GPT-4.5 is optimized for human-like dialogue and prose rather than code or agentic tool use, citing hands-on tests via [LMSYS Arena](https://arena.lmsys.org/). They note it’s strong at explanation/summarization/creative writing but *“not made for aider polygot”* (i.e., not tuned for coding workflows like [Aider](https://github.com/paul-gauthier/aider)), while **GLM** is positioned as stronger for coding; hence evaluations should span both `coding` and `creative writing` to avoid task-specific bias.
    - Methodology concerns: commenters caution against opaque closed-source comparisons and suggest an apples-to-apples matchup with an open-source `~120B` model for scale parity. They imply fair benchmarks should disclose prompt templates, system settings, and whether tool-use/agents were enabled to ensure reproducibility and avoid cherry-picking across differently specialized models.
- [**Deepseek V3.1 improved token efficiency in reasoning mode over R1 and R1-0528**](https://www.reddit.com/gallery/1mv7kk2) ([Score: 203, Comments: 16](https://www.reddit.com/r/LocalLLaMA/comments/1mv7kk2/deepseek_v31_improved_token_efficiency_in/)): **A community benchmark ([LRMTokenEconomy](https://github.com/cpldcpu/LRMTokenEconomy)) indicates DeepSeek V3.1 improves** `reasoning`**mode token efficiency over R1 and R1-0528, notably reducing "overthinking" on knowledge and math prompts by producing shorter** `CoT` **while maintaining correctness. The evaluator still observes occasional very long chains on logic/brain-teaser style puzzles, suggesting heuristic limits remain for complex deductive tasks.** One commenter notes decoding-time controls on "thinking" can further improve accuracy, pointing to an example approach: https://x.com/asankhaya/status/1957993721502310508. Other remarks are non-technical acknowledgments.
    - Reducing unnecessary words/formatting in the reasoning trace can be directly optimized via RL reward shaping—e.g., adding a penalty for verbose chain-of-thought or extraneous markup to compress tokens without sacrificing solution quality. There’s also speculation that different experts are activated during “thinking” in MoE-style setups; this could increase `VRAM` requirements for local users while being acceptable for hosted infra if compute orchestration is the main constraint. This suggests two levers for token efficiency: policy-level reward penalties and architecture-level expert routing/gating during reasoning.
    - Inference-time control of “thinking” can improve accuracy while limiting tokens, as noted with this example: https://x.com/asankhaya/status/1957993721502310508. Techniques include constraining step counts, imposing per-question token budgets, or guided/self-consistency sampling to prune low-value reasoning tokens—often complementary to training-time penalties and applicable across models without retraining.
    - A key open question raised: does the improved token efficiency hold at accuracy parity with R1/R1-0528, or is there a correctness trade-off? For robust comparisons, results should report accuracy metrics (e.g., pass@1) alongside average reasoning-token counts per benchmark so efficiency can be evaluated at fixed quality levels.
- [**Understanding DeepSeek-V3.1-Base Updates at a Glance**](https://i.redd.it/mqcnus8py1kf1.png) ([Score: 190, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1muxbqj/understanding_deepseekv31base_updates_at_a_glance/)): **DeepSeek released DeepSeek‑V3.1‑Base with core architecture largely unchanged from V3 (e.g., same vocab size) but adds a new hybrid mode with a togglable “thinking” capability and updated tokenizer** `added_tokens` **(expanded placeholder/functional tokens) inferred from config/tokenizer diffs. Community tests suggest improved coding performance and higher ranking versus V3, though the official model card/benchmarks aren’t yet published; the image aggregates these deltas plus download links.** Commenters note some reported Aider scores were from a Chat model (not Base) and that Base isn’t typically exposed via API; they emphasize V3.1‑Base is a completion model without a chat template, and are awaiting OpenRouter updates to rerun benchmarks.
    - Clarification that the reported “Aider score” was measured using a **Chat** variant rather than the advertised **Base** model, which can skew expectations of raw base capability. One commenter notes the provider doesn’t expose the Base via API (low demand), implying benchmarks labeled as “Base” may actually reflect instruction-tuned/chat behavior rather than true pretraining-only performance.
    - Counterpoint emphasizes that if this is truly a **Base** model (e.g., DeepSeek-V3.1-Base), a `chat template` should not be used because it hasn’t been SFT’d for chat; it should be treated as a plain `text completion` model. Applying chat-specific system/user/assistant formatting could degrade outputs or invalidate comparisons versus genuine chat/instruct models.
    - A benchmarker is waiting for **OpenRouter** to update the endpoint before re-running tests, highlighting that provider/version lag can materially impact scores and reproducibility. Benchmark results should specify the exact endpoint/model variant and provider (e.g., OpenRouter) used to avoid conflating Base vs Chat and pre-/post-update behaviors (https://openrouter.ai/).

### 2. New Open-Source Model Launches: IBM/NASA Surya and ByteDance Seed-OSS-36B

- [**IBM and NASA just dropped Surya: an open‑source AI to forecast solar storms before they hit**](https://i.redd.it/moddapg5j6kf1.jpeg) ([Score: 275, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1mvfdja/ibm_and_nasa_just_dropped_surya_an_opensource_ai/)): **IBM and NASA announced Surya, an open‑source heliophysics foundation model pre‑trained on years of Solar Dynamics Observatory (SDO) imagery to learn transferable solar features for zero/few‑shot fine‑tuning on tasks like flare probability, CME risk, and geomagnetic indices (**`Kp`**/**`Dst`**). The release (weights + training recipes) targets modest‑compute adaptation via SDO preprocessing and LoRA/adapters, with evaluation encouraged via lead‑time vs. skill metrics on public benchmarks and stress‑tests on extreme events. Relative to current space‑weather approaches—physics‑based MHD/propagation models (e.g., WSA‑Enlil: https://www.swpc.noaa.gov/models/wsa-enlil-solar-wind-prediction), empirical/statistical baselines, and task‑specific CNN/RF models—the claimed contribution is broad pretraining on SDO (https://sdo.gsfc.nasa.gov/) for better transfer and accessibility; rigorous head‑to‑head skill and cost comparisons remain to be shown.** Comments flag the missing link and question whether this is hype, asking what was used before and whether simple linear/empirical models can match performance; they call for evidence of concrete improvements over operational baselines and clarity on novelty vs. prior CNN/statistical methods.
    - A commenter challenges the technical novelty and justification of Surya, asking how it improves over pre-LLM baselines and whether simpler models (e.g., linear/logistic regression) could match performance at lower cost. They request clear, comparative benchmarks and ablations versus established methods, and clarity on predictability vs stochasticity of flare events, rather than marketing claims. They reference the repo but note the lack of structured evidence showing gains in the specific tasks (flare and solar wind forecasting).
    - Another user focuses on real-time deployment, proposing a Gradio app driven by "recent" solar imagery for the repo’s tasks—**24 hr solar flare forecasting** and **4‑day solar wind forecasting**—but reports difficulty sourcing live inputs: https://github.com/NASA-IMPACT/Surya?tab=readme-ov-file#1-solar-flare-forecasting and https://github.com/NASA-IMPACT/Surya?tab=readme-ov-file#3-solar-wind-forecasting. They note the Hugging Face datasets used by Surya only reach `2024` (e.g., https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting) and some are broken (e.g., https://huggingface.co/datasets/nasa-ibm-ai4science/SDO_training), hindering real-time replication and inference with current data.
- [**Seed-OSS-36B-Instruct**](https://www.reddit.com/r/LocalLLaMA/comments/1mvjj8q/seedoss36binstruct/) ([Score: 153, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1mvjj8q/seedoss36binstruct/)): **ByteDance’s Seed Team released Seed-OSS-36B-Instruct ([HF](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct)), a** `~36B`**param, Apache-2.0 LLM trained on** `~12T` **tokens with native** `512K` **context, emphasizing controllable reasoning length ("thinking budget"), strong tool-use/agentic behaviors, and long-context reasoning. They also provide paired base checkpoints to control for synthetic instruction data in pretraining: [Seed-OSS-36B-Base](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base) (augmented w/ synthetic instructions, reported to improve most benchmarks) and [Seed-OSS-36B-Base-woSyn](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base-woSyn) (without such data) to support research on instruction-data effects.** Commenters highlight the model’s native `512K` context as possibly the longest among open-weight models with a practical memory footprint, contrasting it with 1M+ context models (e.g., MiniMax-M1, Llama) that are too large, and noting Qwen3’s 1M via RoPE but only `256K` native. There’s also discussion that incorporating synthetic instruction data during pretraining materially boosts benchmark performance, with the w/o-synthetic variant valued for uncontaminated foundation-model research.
    - Model card notes two base variants: one augmented with synthetic instruction data and one without. Quoting: *“Incorporating synthetic instruction data into pretraining leads to improved performance on most benchmarks… We also release* `Seed-OSS-36B-Base-woSyn` *… unaffected by synthetic instruction data.”* This gives users a choice between potentially higher scores from synthetic-instruction-augmented pretraining and a “clean” pretraining distribution for downstream SFT or analysis. Links: [Seed-OSS-36B-Base](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base), [Seed-OSS-36B-Base-woSyn](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base-woSyn).
    - Claimed native `512K` context window for a `36B` dense model, positioning it as one of the largest “native” context open-weights with a practical memory footprint. Commenters contrast this with models advertising `1M+` via RoPE extrapolation (e.g., **Qwen3** with `1M` but `256K` native) or massive models like **MiniMax-M1**/**Llama** where resource needs are prohibitive; a native window can avoid quality drop-offs seen in extrapolated RoPE regimes. This matters for retrieval-heavy or long-document tasks without resorting to aggressive chunking or special caching tricks.
    - Reported benchmarks and features: `AIME24 91.7`, `AIME25 84.7`, `ArcAGI V2 40.6`, `LiveCodeBench 67.4`, `SWE-bench Verified (OpenHands) 56`, `TAU1-Retail 70.4`, `TAU1-Airline 46`, `RULER 128k 94.6`. Coverage suggests strong math, coding, tool-use/agent, and long-context retention, with `RULER 128k` indicating robust long-range attention. It also advertises controllable “reasoning token length,” implying an inference-time knob to limit or extend reasoning tokens for latency/quality trade-offs.

### 3. Indie Open-Source Innovations: Mobile AndroidWorld Agent and TimeCapsuleLLM (1800s London)

- [**We beat Google Deepmind but got killed by a chinese lab**](https://v.redd.it/qvewe6nd24kf1) ([Score: 1206, Comments: 148](https://www.reddit.com/r/LocalLLaMA/comments/1mv6go1/we_beat_google_deepmind_but_got_killed_by_a/)): **A small team open-sourced an agentic Android control framework that performs real on-device interactions (taps, swipes, typing) and reports state-of-the-art results on the AndroidWorld benchmark, surpassing prior baselines from Google DeepMind and Microsoft Research. Last week, Zhipu AI released closed-source results that slightly edge them out for the top spot; in response, the team published their code and is developing custom mobile RL gyms to push toward** `~100%` **benchmark completion. Repo: https://github.com/minitap-ai/mobile-use** Top comments are supportive of open-source, recommending community-building as a competitive moat and noting many seminal OSS efforts began with tiny teams; one observer remarks the demo appears fast.
    - Several commenters probe how an app can control a phone (esp. iPhone) without rooting; the practical route is OS-sanctioned automation layers rather than arbitrary event synthesis. On iOS, cross‑app control typically uses Apple's UI testing stack (XCTest) and derivatives like **Appium’s WebDriverAgent** that run a developer‑signed automation runner to query the accessibility tree and inject taps/typing—no jailbreak, but you cannot ship this in an App Store app and it requires provisioning entitlements ([XCTest UI Testing](https://developer.apple.com/documentation/xctest/ui_testing), [WebDriverAgent](https://github.com/appium/WebDriverAgent)). On Android, agents rely on an **AccessibilityService** (with `BIND_ACCESSIBILITY_SERVICE`) to read the UI and perform gestures, often paired with **MediaProjection** for screen capture—root not required but user consent and Play policy compliance are mandatory ([AccessibilityService](https://developer.android.com/guide/topics/ui/accessibility/service), [MediaProjection](https://developer.android.com/reference/android/media/projection/MediaProjection)).
    - On use cases: the tool enables end‑to‑end QA/RPA on real devices (e.g., navigating login flows, handling OTPs, changing settings, or orchestrating tasks across third‑party apps) and accessibility augmentation. Typical stacks expose a WebDriver-like API via **Appium** (iOS via WebDriverAgent; Android via **UiAutomator2**/Accessibility), letting an AI agent launch apps, tap, type, and read accessibility labels; however, some system dialogs and privileged settings remain out of reach due to OS sandboxing and entitlement limits ([Appium docs](https://appium.io/docs/en/latest/), [UiAutomator2](https://developer.android.com/training/testing/other-components/ui-automator)).
- [**My LLM trained from scratch on only 1800s London texts brings up a real protest from 1834**](https://www.reddit.com/r/LocalLLaMA/comments/1mvnmjo/my_llm_trained_from_scratch_on_only_1800s_london/) ([Score: 190, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mvnmjo/my_llm_trained_from_scratch_on_only_1800s_london/)): **OP trained several LLMs entirely from scratch on a curated corpus of ~7,000 London-published texts from 1800–1875 (~**`5–6 GB`**), including a custom tokenizer trained on the same corpus to minimize modern vocabulary; two models used nanoGPT ([repo](https://github.com/karpathy/nanoGPT)) and the latest follows Phi-1.5-style training ([phi-1_5](https://huggingface.co/microsoft/phi-1_5)), with no modern data or fine-tuning. Given the prompt “It was the year of our Lord 1834,” the model generated text referencing a London “protest” and “Lord Palmerston,” which OP notes aligns with documented 1834 events, implying the model learned period-specific associations rather than only stylistic mimicry. Code/dataset work is shared at [TimeCapsuleLLM](https://github.com/haykgrigo3/TimeCapsuleLLM), with plans to scale to** `~30 GB` **and explore city/language-specific variants.** Top comments are broadly enthusiastic, framing the approach as a compelling, DIY way to surface historical zeitgeist directly from primary sources; no substantive technical critiques or benchmarks were discussed.
    - Proposal to train diachronic and regional variants: build separate models on cumulative corpora up to successive cutoffs (e.g., 100 AD, 200 AD, …) and also region-specific subsets. This would enable measuring semantic drift and dialectal variation by aligning embeddings across checkpoints (e.g., Orthogonal Procrustes), tracking vocabulary frequency shifts, and comparing `perplexity` on temporally out-of-domain test sets. It also supports transfer studies by testing a model trained to 1800 vs 1900 on predicting 1830s events to quantify temporal generalization.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Unitree and Boston Dynamics Humanoid Robot Updates

- [**Boston Dynamics shares new progress**](https://youtu.be/HYwekersccY?si=yCyvRRUMpBuvib6d) ([Score: 399, Comments: 186](https://www.reddit.com/r/singularity/comments/1mvjh3z/boston_dynamics_shares_new_progress/)): **Boston Dynamics, in collaboration with Toyota Research Institute, showcases progress on Large Behavior Models (LBMs) for the Atlas humanoid, training end-to-end, language‑conditioned neural policies that map natural‑language commands directly to coordinated whole‑body behaviors for long‑horizon manipulation sequences. The demo emphasizes closed‑loop control that maintains task execution under physical perturbations while dynamically reorienting the body to grasp, transport, and place multiple objects, indicating improved policy generalization across extended tasks. Video: [Getting a Leg up with End-to-end Neural Networks | Boston Dynamics (YouTube)](https://youtu.be/HYwekersccY).** Top comments highlight the system’s robustness to human‑induced disturbances and its dynamic whole‑body repositioning during manipulation; others contextualize the achievement by noting the remaining gap to human‑level dexterous hands.
    - Several commenters highlight the system’s robustness to external perturbations during manipulation and locomotion, noting it maintains balance and task progress while being pushed or having the target moved. This implies strong disturbance rejection via closed-loop state estimation, whole-body control, and compliant/impedance behaviors that let it re-plan foot placement and end-effector trajectories in real time.
    - The demo underscores limitations of current end-effectors versus the human hand’s dexterity, strength-to-weight, and tactile feedback. Observers note the robot compensates by dynamically repositioning its whole body to achieve favorable approach angles and grasps, reflecting a design tradeoff where whole-body motion and control sophistication offset lower gripper DOF and sensing compared to human hands.
    - Viewers infer tight perception-control integration for dynamic object interaction, likely involving real-time object pose tracking and visual servoing to follow a moving box while preserving stability margins. The ability to smoothly adapt body posture and arm trajectory suggests model-predictive or whole-body trajectory optimization running at interactive rates to reconcile manipulation objectives with balance constraints.
- [**Unitree are teasing their next humanoid**](https://i.redd.it/a3detiiop1kf1.jpeg) ([Score: 196, Comments: 36](https://www.reddit.com/r/singularity/comments/1muw2vc/unitree_are_teasing_their_next_humanoid/)): **Unitree teased its next humanoid with a silhouette image listing** `31 joint DOF (6*2 + 3 + 7*2 + 2)` **and** `H:180`**, implying 6-DOF per leg (x2), a 3-DOF torso, 7-DOF arms (x2), and a 2-DOF head on a 180 cm-tall platform. The teaser emphasizes “Agile” and “Elegant” and says “Coming Soon,” but includes no actuator, hand, sensing, or benchmark details. Source: Unitree’s post on X: https://x.com/UnitreeRobotics/status/1957800790321775011.** Comments note the likely height translation as 180 cm (≈5′11″); the rest are non-technical/jokes.
- [**Unitree G1, the winner of solo dance at WHRG, wears an AGI tshirt while performing**](https://v.redd.it/nw2jrlios5kf1) ([Score: 306, Comments: 80](https://www.reddit.com/r/singularity/comments/1mvc1i9/unitree_g1_the_winner_of_solo_dance_at_whrg_wears/)): **Unitree’s G1 humanoid won the solo dance category at WHRG, performing an articulated, untethered routine while wearing an “AGI” T‑shirt—likely using pre‑scripted/teleop choreography rather than onboard high‑level planning, but still demonstrating strong whole‑body balance and articulation in a compact form factor. Video: [v.redd.it/nw2jrlios5kf1](https://v.redd.it/nw2jrlios5kf1); platform details: [Unitree G1](https://www.unitree.com/g1/).** Commenters debate whether the demo is a “scripted/teleop fake‑out” versus meaningful progress: skeptics see little AI, while others highlight the value of demonstrated actuation, stability, and self‑powered operation for future embodied intelligence, cautioning not to over‑attribute progress to the AI side.
    - Key technical debate: the routine likely relied on pre-scripted or teleoperated choreography rather than on-board autonomy, but it still showcases robust whole‑body articulation, balance, and trajectory tracking under self-contained power. This indicates mature low-latency control loops, good state estimation/IMU fusion, and high torque density in a compact, non‑bulky package—valuable foundations that an autonomous stack could leverage later. The caution is not to conflate such demos with progress in high-level "AI" or planning; they primarily validate mechanics and control, not cognition.
    - Several commenters frame the result as progress in *embodiment* (the "body") rather than *cognition* (the "mind"). In other words, the demonstrated stability and precision are meaningful steps for embodied intelligence, but the "AGI" branding overstates the software side; without evidence of online perception, planning, or adaptation, the achievement is best viewed as hardware/control maturity rather than AI capability.
- [**Humanoid robots are getting normalized on social media right now**](https://i.redd.it/2ict4t2do6kf1.jpeg) ([Score: 200, Comments: 61](https://www.reddit.com/r/singularity/comments/1mvg585/humanoid_robots_are_getting_normalized_on_social/)): **Post notes a surge of humanoid-robot short videos (Reels/TikTok), suggesting a coordinated normalization push versus organic growth, with examples like IG’s rizzbot_official and public sightings in Asia/Austin. The image shows a friendly-faced sidewalk robot being filmed by a passerby—an HRI/PR tactic (approachable digital face, clumsy/dorky persona) to reduce perceived risk and job-threat anxieties; no hardware/model specs or technical benchmarks are discussed.** Comments split between recommender-driven exposure—“the algorithm decided you’re someone who likes…”—and deliberate soft-power branding to preempt public backlash; a joke about “sidewalks” nods at urban-space and access concerns.
    - Feed saturation is likely due to platform recommender systems (collaborative filtering/engagement-optimized ranking) rather than an organic, population-wide spike. The algorithm “decided that you’re someone who likes watching videos of humanoid robots,” indicating personalization effects and feedback loops: watch a couple, get many more — not evidence of broad normalization.
    - Multiple users flag that “Clanker” appeared “out of nowhere all at once,” suggesting possible coordinated seeding/astroturfing rather than organic meme growth. Technical indicators would be synchronized posting times, reused captions/hashtags, and rapid initial engagement from low-history accounts, which can inflate perceived momentum without real grassroots interest.
    - Clear distinction raised between novelty exposure and genuine normalization: “novelty =/= normalization.” Short-form spikes can reflect curiosity bias optimized by ranking systems, whereas normalization would require longitudinal signals (repeat exposure without drop-off, stable positive sentiment, downstream behaviors) rather than raw impression counts.
- [**Unrealistic**](https://i.redd.it/y5zt4awdb4kf1.png) ([Score: 3960, Comments: 56](https://www.reddit.com/r/OpenAI/comments/1mv74u1/unrealistic/)): **Satirical post/meme referencing the Terminator 2 arc where a tech creator destroys their invention upon learning it could doom humanity, framing that as “unrealistic” today. Contextually it comments on contemporary AI risk, corporate incentives, and regulation discourse (e.g., Sam Altman publicly urging AI regulation) rather than offering technical data or benchmarks.** Comments debate founder incentives (e.g., bunkers vs. shutdown), and note skepticism toward current AI capabilities alongside cynicism about regulatory theater/regulatory capture when leaders call for oversight.
    - Several comments implicitly touch on the regulatory-capture debate: **Sam Altman** publicly advocated for licensing and safety standards for frontier models in U.S. Senate testimony, which some view as *“begging to be regulated”* while critics argue current model capability doesn’t justify existential-risk framing. See the May 2023 hearing “Oversight of A.I.: Rules for Artificial Intelligence” (https://www.help.senate.gov/hearings/oversight-of-ai-rules-for-artificial-intelligence). This raises a technical governance question: where to set capability thresholds for licensing and evals without entrenching incumbents.
    - A correction notes he’s an employee, not a founder—important for governance and control. In structures like **OpenAI’s** capped-profit model, the nonprofit board formally controls the for-profit entity (https://openai.com/blog/our-structure), changing who can make unilateral decisions about pausing/shuttering products and creating different incentive and accountability dynamics than founder-led companies.
    - On catastrophic-risk posture, the contrast between “building bunkers” vs. destroying/pausing products highlights how major labs operationalize risk via evals and preparedness rather than personal contingency plans. Examples include **OpenAI’s Preparedness** and red-teaming initiatives (https://openai.com/blog/preparedness) and **Anthropic’s** Responsible Scaling Policy with AI Safety Levels (`ASL-1`–`ASL-4`) (https://www.anthropic.com/news/ai-safety-levels), which define capability thresholds, eval protocols, and gating mitigations before scaling.

### 2. Image Edit Model Benchmarks and Workflows (Qwen, WAN 2.2, Image Edit Arena)

- [**Move away, guys... Someone's coming**](https://i.redd.it/2gt0pua832kf1.png) ([Score: 175, Comments: 31](https://www.reddit.com/r/Bard/comments/1muxvko/move_away_guys_someones_coming/)): **Leaderboard snapshot of an “Image Edit Arena” benchmarking image-editing models via head‑to‑head votes, reporting rank, score with confidence intervals, vote counts, organization, and license. Proprietary models (e.g., OpenAI’s gpt-image-1 and Black Forest Labs’ flux-1-kontext-pro) occupy the top positions with the highest scores, ahead of various open‑source entries.** Commenters hype a model nicknamed “Nano Banana” as delivering notably strong results, though another user asks what it is, indicating some ambiguity about which listed model that nickname refers to.
    - Commenters attribute consistently high-quality outputs to a model referred to as “Nano Banana,” implying it has a recognizable output signature. However, no concrete benchmarks or model identifiers are provided; the thread would benefit from specifics like exact checkpoint/LoRA names and quantitative metrics (e.g., `FID`, `CLIPScore`) to substantiate performance claims.
    - A request for a “3D ftx output” suggests interest in direct 2D-to-3D model capabilities and export to common 3D formats (likely FBX/GLTF). This points to a workflow gap: most image models output 2D images, so producing riggable meshes would require a text-to-3D pipeline (e.g., NeRF or Gaussian Splatting to mesh) or a native 3D diffusion/generative geometry model.
    - One user asks why **xAI’s Grok** isn’t on the referenced list, possibly expecting inclusion of `Grok-1.5V` (multimodal). This raises ambiguity about the list’s modality scope (LLMs vs. image models) and evaluation criteria, suggesting the need to clarify which models and benchmarks are in scope.
- [**Simple multiple images input in Qwen-Image-Edit**](https://www.reddit.com/gallery/1mv0c37) ([Score: 341, Comments: 51](https://www.reddit.com/r/StableDiffusion/comments/1mv0c37/simple_multiple_images_input_in_qwenimageedit/)): **Post demonstrates multi-image conditioning in Qwen-Image-Edit: (1) clothing/style transfer from a mannequin to a subject combined with scene relocation to a Paris street café, and (2) compositing two subjects into an embrace while preserving specific attributes (hairstyles and hair color). A separate workflow schematic is shared for reference ([workflow screenshot](https://ibb.co/VYm716L7)).** Commenters report strong prompt adherence but weak photorealism ("plastic skin" and loss of detail), suggest trying a different sampler/preset ("res_2s/bong") for better skin realism, and note that quality may require a finetune or LoRA to improve.
    - Several users suggest benchmarking Qwen-Image-Edit against alternative pipelines like “res_2s/bong,” reporting **much better skin detail/realism** with that setup. A controlled A/B (same prompt/seed) would help quantify texture retention (pores, fine hair) and reduce the “plastic skin” artifact observed in Qwen-Image-Edit outputs.
    - There’s a clear trade-off noted: **strong prompt adherence** from Qwen-Image-Edit, but **poor image fidelity** (detail erasure, waxy skin) reminiscent of older-gen models. Commenters propose a targeted finetune or **LoRA** trained on high-quality portrait datasets to improve microtexture and realism without sacrificing adherence.
    - On workflow reproducibility, one commenter shared a full ComfyUI graph JSON for quick import, demonstrating that **copy/paste of the workflow** takes under a minute: https://pastebin.com/J6pz959X. Another asks for a “screenshot-to-workflow” feature (e.g., reconstructing graphs from images like https://ibb.co/VYm716L7), highlighting a potential tool gap (OCR/graph parsing) despite current JSON export/import convenience.
- [**Wan 2.2 Realism Workflow | Instareal + Lenovo WAN**](https://www.reddit.com/gallery/1mvbmhh) ([Score: 264, Comments: 33](https://www.reddit.com/r/StableDiffusion/comments/1mvbmhh/wan_22_realism_workflow_instareal_lenovo_wan/)): **Author shares a WAN 2.2 photorealism workflow that blends two LoRAs—Instareal ([link](https://civitai.com/models/1877171?modelVersionId=2124694)) and Lenovo WAN ([link](https://civitai.com/models/1662740?modelVersionId=2066914))—with “specific upscaling tricks” and added noise to enhance realism; the node graph is provided via Pastebin ([workflow](https://pastebin.com/ZqB6d36X)). Emphasis is on LoRA stacking/tuning for WAN 2.2 and post/late-stage detail via upscale+noise rather than base model changes.** Top comments request technical specifics on the upscaling pipeline (ComfyUI vs external tools like Topaz/Bloom) and how to source all required files from the workflow graph, indicating interest in reproducibility and integration details.
    - Upscaling workflow details: One commenter asks how the OP handles upscaling with this **WAN 2.2 realism** setup—whether it’s done inside **Comfy** or exported to external tools like **Topaz** or “Bloom,” and what yields better detail preservation at `2x–4x`. They’re looking for practical implementation specifics (node choices vs. external batch processing) to maintain realism after generation.
    - Workflow assets and dependencies: A user requests where to obtain all required files referenced in the pipeline, implying multiple components (e.g., model checkpoints, LoRAs like “Instareal,” and any custom nodes/configs) are needed to reproduce the results. Clarification on exact versions and download sources is needed to make the workflow reproducible.
    - Model scope clarification: Another commenter believes **WAN** is primarily for creating movie clips, seeking clarity on whether **WAN 2.2** in this workflow is being used for still images, video, or both. This raises questions about the model’s intended use cases and any settings or constraints when repurposing it for photorealistic stills.
- [**Some random girl gens with Qwen + my LoRAs**](https://www.reddit.com/gallery/1mvbxd3) ([Score: 211, Comments: 46](https://www.reddit.com/r/StableDiffusion/comments/1mvbxd3/some_random_girl_gens_with_qwen_my_loras/)): **OP showcases AI-generated portraits ("girl gens") using Qwen combined with custom LoRAs; no specific model variant, sampler, or hardware details are provided. A commenter requests performance metrics—*"What is the generation time though?"*—but no timings or system specs are given in-thread. Another commenter shares their own workflow/models with links: HuggingFace [Danrisi/Lenovo_Qwen](https://huggingface.co/Danrisi/Lenovo_Qwen) and Civitai model [1662740](https://civitai.com/models/1662740).** Discussion trends toward reproducibility and performance (inference time) requests, while another user contributes alternative assets/workflows; no benchmarking data or implementation specifics are reported.
    - A commenter requests concrete inference performance, specifically generation time per image. No timings, steps, sampler, or hardware details are provided, so latency/throughput and reproducibility remain unquantified.
    - The OP links their workflow/models, indicating image generations done with **Qwen + custom LoRAs** and providing artifacts on Hugging Face and Civitai: https://huggingface.co/Danrisi/Lenovo_Qwen and https://civitai.com/models/1662740. These resources likely include the LoRA weights and pipeline details needed to replicate the setup (prompts, resolution, scheduler, and negative prompts, if documented).
    - Another commenter asks which trainer was used for the LoRAs (e.g., **kohya-ss**, Diffusers/Accelerate, DreamBooth variants), a key detail affecting VRAM usage, training stability, and final quality. The thread does not specify the trainer or hyperparameters (steps, `lr`, batch size, `rank`, `alpha`), so the exact training regimen remains unclear.
- [**Editing iconic photographs with editing model**](https://www.reddit.com/gallery/1mv42dg) ([Score: 273, Comments: 35](https://www.reddit.com/r/StableDiffusion/comments/1mv42dg/editing_iconic_photographs_with_editing_model/)): **Thread showcases an image editing model applied to “iconic photographs” (e.g., an Apollo moon landing shot), with commenters assessing edit fidelity/realism; however, the linked gallery is access-gated (HTTP 403) so the actual media can’t be independently verified ([reddit link](https://www.reddit.com/gallery/1mv42dg)). Model identity is unspecified in-post.** Technically minded commenters note high perceptual quality but flag a historical/camera inconsistency in the first image (*“shot by a Nikon”*), and ask whether the tool is “Kontext or qwan?”—suggesting ambiguity between editing frameworks or model families (e.g., [Qwen](https://github.com/QwenLM)). Non-technical virality speculation about the moon-landing edit is also present.
    - One commenter asks whether the editing was done with Kontext or qwan (likely referring to Alibaba’s Qwen), indicating interest in the exact editing model used and its image-editing capabilities. This implies a comparison of diffusion or vision-language editing pipelines and what strengths they bring to photorealistic edits of iconic photos.
    - Feedback on the National Geographic edit notes high photorealism but frustration that the subject’s eyes remain obscured, highlighting a common limitation in generative edits: handling occlusions and reconstructing small facial details without uncanny artifacts. This suggests the model likely prioritized global realism and texture coherence over reconstructing fine ocular detail, a trade-off often seen in face edits.
    - A user asserts the first original frame was shot on a Nikon, pointing to provenance details (camera make) that can affect color science and grain profiles when evaluating edit realism. It underscores that ground-truth capture characteristics may bias how convincing model outputs appear.
- [**GPT-5 has been surprisingly good at reviewing Claude Code’s work**](https://www.reddit.com/r/ClaudeAI/comments/1mvbxaw/gpt5_has_been_surprisingly_good_at_reviewing/) ([Score: 443, Comments: 100](https://www.reddit.com/r/ClaudeAI/comments/1mvbxaw/gpt5_has_been_surprisingly_good_at_reviewing/)): **OP outlines a planning→implementation→review workflow: use Claude Code (Sonnet 4) for code generation, use [Traycer](https://traycer.ai/) to produce an implementation plan (Traycer appears to wrap Sonnet 4 with prompt/agent scaffolding), then feed the produced code back to Traycer where GPT‑5 performs a plan‑vs‑implementation check—i.e., a “verification loop” that reports covered items, missing scope, and newly introduced issues. This contrasts with reviewers like Wasps/Sourcery/Gemini Code Review that comment on a raw** `git diff` **without feature context; tying review to an explicit plan improves signal. Reported costs:** `~$100` **for Claude Code access plus** `~$25` **for Traycer.** Commenters echo the split: GPT‑5 excels at planning/analysis/debugging but is weak at directly writing code, so they pair it with Sonnet 4 for implementation; another adds a Claude Code test agent to run unit tests before a final GPT‑5 pass. One raises transparency concerns about Traycer (e.g., GitHub sign‑in) but notes it covers most of what they were building with local prompt files.
    - Practitioners report a division-of-labor: **GPT-5** for planning/reviews/debug analysis and **Anthropic’s Claude** (e.g., [Claude Code](https://docs.anthropic.com/claude/docs/claude-code), “Sonnet 4”) for implementation. One notes: *“GPT-5 is great at everything BUT writing code... it struggles to implement its own plan,”* so they route all coding to Claude while keeping GPT-5 for spec/review to maximize quality.
    - An end-to-end pipeline cited: **GPT-5** drafts concepts/specs → **Claude Code** implements → a Claude Code test agent runs unit tests → **GPT‑5** performs a final codebase check. This review loop reportedly “works very well,” providing guardrails against regressions/hallucinations while automating ~`80%` of a homegrown prompt-driven workflow (e.g., Obsidian prompt files; tools like Traycer aim to cover most of this).
    - Hallucination/control-drift remains a risk: a user recalls **Codex** being asked to build a JS‑framework app but emitting Python commands. Hence the inclusion of automated unit tests (via the Claude Code test agent) and a final GPT‑5 audit to detect such mismatches; without these checks, GPT-5 can still hallucinate despite strong planning/review performance. See background on Codex: [OpenAI Codex](https://openai.com/blog/openai-codex).
- [**“Built with Claude” Contest from Anthropic**](https://www.reddit.com/r/ClaudeAI/comments/1muwro0/built_with_claude_contest_from_anthropic/) ([Score: 192, Comments: 44](https://www.reddit.com/r/ClaudeAI/comments/1muwro0/built_with_claude_contest_from_anthropic/)): **Anthropic announced a community "Built with Claude" contest on r/ClaudeAI, reviewing all posts with the "Built with Claude" flair through the end of August and selecting the top** `3` **entries based on upvotes, discussion, and overall merit; each winner receives** `$600` **in either Claude Max subscription credit or API credits. Submissions must be original builds using Claude (e.g., [Claude.ai](http://claude.ai/), Claude app, Claude Code/SDK) and should include technical build details (prompts, agents, MCP servers/workflows like the [Model Context Protocol](https://modelcontextprotocol.io/)), plus screenshots/demos; see the official rules [here](https://support.anthropic.com/en/articles/12003471-built-with-claude-contest-official-rules).** Moderators welcome increased Anthropic engagement while committing to maintain independent performance reports and the community’s voice. Commenters asked about entry barriers (e.g., subreddit karma limits) and one reported non-receipt of a prior `$600` reward from a "Code with Claude" event, raising fulfillment/support concerns.
    - A commenter showcases a Claude-assisted project, the "RNA cube," which assigns a decimal value to each RNA codon to enable integer-based arithmetic on genetic data and clusters amino acids into `4` chemically distinct classes. They provide a live site at [https://biocube.cancun.net](https://biocube.cancun.net/), with additional resources including a 3D visualization and a variant analysis batch tool; the community hub is at https://www.reddit.com/r/rnacube. They credit Claude’s ability to reason about complexity as instrumental in the framework’s design.
    - The same commenter flags a potential tooling quirk: asking Claude to fetch/comment on their site may return a stale snapshot due to a *"non updating cache"*. As a workaround, they recommend cache-busting by appending a query param—e.g., [https://biocube.cancun.net/index.html?id=100—to](https://biocube.cancun.net/index.html?id=100%E2%80%94to) ensure the latest version is retrieved.
- [**Agent mode is so impressive**](https://www.reddit.com/r/OpenAI/comments/1mv2lgz/agent_mode_is_so_impressive/) ([Score: 237, Comments: 231](https://www.reddit.com/r/OpenAI/comments/1mv2lgz/agent_mode_is_so_impressive/)): **OP reports that an autonomous "agent mode" (claimed as GPT‑5) can execute end‑to‑end web tasks like grocery shopping using user‑specified constraints (dietary preferences, budget, brand preferences), effectively performing human‑style site navigation and checkout. Commenters surface reliability limits: agents often fail on highly dynamic, script‑heavy sites and may abandon flows, reverting to asking the user for structured inputs (e.g., insurance quotes) when DOM/state handling or anti‑automation blocks break the workflow.** Debate centers on the UX substrate: some argue agents will remain error‑prone while forced to parse human UIs, predicting a shift toward machine‑readable `agent interfaces` that exchange raw data—potentially reshaping e‑commerce and web architecture—while others express low trust in agents for critical tasks, calling current results “MySpace era.”
    - Agents remain brittle when forced to navigate consumer web UIs: dynamic DOMs, heavy client-side rendering, CSRF flows, cookie walls, and bot-detection make step-by-step automation unreliable. A proposed path is **agent-native interfaces** (raw JSON/data exchange rather than HTML), which could re-architect ecommerce and machine-to-machine interactions—akin to moving past the "MySpace era" toward a machine-readable web.
    - A real-world task (collecting insurance quotes) failed because sites were "too dynamic," leading the agent to give up and request manual inputs. This underscores current limitations around multi-step forms, async JS, embedded widgets/iframes, and anti-automation measures (e.g., captchas), suggesting low determinism until providers expose stable APIs or dedicated agent endpoints.
    - Workarounds and scoped successes: one user exposed a local NAS chat service to the web and supplied credentials so the agent could self-serve answers, reducing messages under small limits via authenticated tool-use. Another achieved an end-to-end shopping flow (spec -> body measurements/wardrobe context -> stock checks -> cart prefill -> human approval), indicating agents can perform reliably in constrained domains with stateful planning, session persistence, and access to authenticated/internal tools.
- [**Is AI bubble going to burst. MIT report says 95% AI fails at enterprise.**](https://www.reddit.com/r/OpenAI/comments/1mv6cs8/is_ai_bubble_going_to_burst_mit_report_says_95_ai/) ([Score: 218, Comments: 208](https://www.reddit.com/r/OpenAI/comments/1mv6cs8/is_ai_bubble_going_to_burst_mit_report_says_95_ai/)): **Thread asks if an “AI bubble” will burst, citing an MIT report claiming** `95%` **of enterprise AI initiatives fail. Commenters argue LLMs are genuinely useful in assistive, human-in-the-loop settings, but attempts to fully automate and replace human judgment are failing in production and won’t yield ROI; hype-driven proliferation of low-value “AI tools/microservices” is a primary failure mode. Likely outcome is a dot-com–style correction where overhyped application startups consolidate while major infrastructure/providers persist.** Consensus: there is a bubble driven by misuse and unrealistic expectations, not by the core tech’s capability; expect investor losses in over-automation plays, but continued progress and resilience among large AI platforms.
    - Human-in-the-loop vs. full automation: Multiple commenters note that attempts to fully replace human decision-making with AI/LLMs in enterprise workflows tend to fail, whereas assistive patterns (AI as a copilot with human oversight/approval) are working. The implied technical reason is reliability/calibration limits of current LLMs for high-stakes, unbounded decisions; successful deployments constrain scope and keep a human in the approval loop to manage edge cases and accountability.
    - Concrete SME productivity wins: One small digital company reports saving “tens of thousands” (`$10k+`) in developer costs by using **ChatGPT** to build production tools, automate bookkeeping via scripts, draft customer emails, and troubleshoot bugs. This suggests strong ROI for narrowly-scoped, automatable tasks where LLM outputs can be quickly validated or executed within existing scripting/tooling pipelines.
    - Market/adoption structure: Commenters argue that the “bubble” risk is concentrated in app-layer startups and overhyped point tools rather than core model providers, analogous to ISPs surviving the dotcom crash. They also note that slow enterprise adoption is often due to organizational bureaucracy and change-management friction rather than model capability limits, implying longer sales/adoption cycles even where technical value exists.
- [**OpenAI logged its first $1 billion month but is still 'constantly under compute,' CFO says**](https://www.cnbc.com/2025/08/20/openai-compute-ai.html) ([Score: 243, Comments: 51](https://www.reddit.com/r/singularity/comments/1mvjb2h/openai_logged_its_first_1_billion_month_but_is/)): **OpenAI reportedly logged its first** `>$1B` **revenue month; CFO Sarah Friar said the company is *"constantly under compute"*. The OP speculates Microsoft/OpenAI’s proposed “Stargate” hyperscale build-out could partially come online by year‑end to add capacity; reports peg Stargate as a** `~$100B` **supercomputer program aimed at alleviating GPU scarcity ([Reuters/The Information](https://www.reuters.com/technology/openai-microsoft-plot-100-billion-stargate-supercomputer-information-2024-03-10/)). Top comments cite a** `~$12B` **annualized revenue run rate and a reported** `~$40B` **Oracle cloud contract as key inputs to scale and cost, alongside expectations of potential price increases ([Reuters on OpenAI-Oracle](https://www.reuters.com/technology/oracle-says-openai-use-its-cloud-infrastructure-2023-09-20/)).** Commentary questions sustainability: skepticism over a `~$500B` valuation and Sam Altman’s mention of a *“new financial instrument”* to fund multi-trillion-dollar AI infrastructure, referencing his push to raise trillions for AI chips/data centers ([Reuters](https://www.reuters.com/technology/altman-seeks-trillions-ai-chip-venture-wsj-2024-02-08/)).
    - Compute capacity and cloud dependency: Multiple comments highlight OpenAI is "constantly under compute," with a claimed `~$40B` Oracle deal cited as evidence of aggressive capacity reservations. The implication is training/inference throughput is the binding constraint (GPU supply, datacenter buildouts, power), which can force price increases and prioritization of high-margin workloads to meet SLAs and latency targets.
    - Financing mega-capex: Sam Altman’s remark about a "new financial instrument" is read as an attempt to fund multi-trillion-dollar datacenter, chips, and power buildouts. Technically, this suggests structures like long-term capacity offtake agreements, asset-backed securitizations of GPU fleets, or sovereign/infra-backed SPVs to keep capex off the operating entity’s balance sheet—commenters are skeptical about execution risk and cost of capital at that scale.
    - Valuation vs. unit economics: A `~$12B` ARR from a `$1B` month contrasted with a `$500B` valuation implies `~40x+` sales, which only works if gross margins improve materially. Commenters note current COGS is compute-heavy; without cost-per-token reductions (e.g., better batching, model sparsity/distillation, custom silicon) or higher ARPU via pricing tiers/enterprise upsell, the valuation and growth path are hard to justify.

### 3. Veo-3 AI Video Generation Demos and Guides

- [**The Art of Simon Stålenhag brought to life with Veo-3**](https://v.redd.it/i0tq1scb96kf1) ([Score: 272, Comments: 22](https://www.reddit.com/r/aivideo/comments/1mve1um/the_art_of_simon_st%C3%A5lenhag_brought_to_life_with/)): **A creator animates Simon Stålenhag’s illustrations using Google’s Veo‑3 video model (image‑to‑video), with the primary clip hosted on [v.redd.it](http://v.redd.it/) which returns** `HTTP 403` **without authentication; a still preview is available via [preview.redd.it](http://preview.redd.it/). No prompts, resolution, or runtime settings are disclosed in the thread; the post focuses on showcasing the stylized animation rather than benchmarks or implementation details.** Commenters note the fit with Stålenhag’s broader media adaptations (e.g., the Tales from the Loop RPG/show) and praise the soundtrack choice as appropriate for the vibe.
- [**Everything I learned after 10,000 AI video generations (the complete guide)**](https://v.redd.it/2p6g2oxgz7kf1) ([Score: 224, Comments: 41](https://www.reddit.com/r/StableDiffusion/comments/1mvnjo3/everything_i_learned_after_10000_ai_video/)): **OP (10 months, ~10k generations) shares a Veo-3–centric workflow for scalable AI video: a 6-part prompt template** `[SHOT TYPE] [SUBJECT] [ACTION] [STYLE] [CAMERA MOVEMENT] [AUDIO CUES]`**, strict “one action per prompt,” and front‑loading key tokens because “Veo3 weights early words more” (author claim). They advocate systematic seed sweeps (e.g., test seeds** `1000–1010`**, build a seed library), negative prompts as always-on QC (**`-no watermark --no warped face --no floating limbs --no text artifacts --no distorted hands --no blurry edges`**), and limiting camera motion to one primitive (slow push/pull, orbit, handheld follow, or static). Cost-wise, they note Google’s listed pricing at** `~$0.50/s` **(**`$30/min`**) leading to** `$100+` **per usable output when retries are included, and suggest cheaper 3rd-party Veo-3 resellers (e.g., https://arhaam.xyz/veo3/) to enable volume testing. Additional tactics: incorporate explicit audio cues in the prompt, style refs with concrete gear/creators (e.g., “Shot on Arri Alexa”, “Wes Anderson”, “Blade Runner 2049”, “teal/orange”), platform‑specific edits for TikTok/IG/Shorts, and JSON “reverse‑engineering” of viral videos to extract structured parameters for variation. Core strategy: prioritize batch generation, first‑frame selection, platform-tailored variants, and “embrace AI aesthetic” over fake realism to drive engagement.** Top comments flag potential undisclosed promotion and stress that many tips are Veo‑3‑specific; for open-weight alternatives, commenters point to WAN‑2.2 guides ([unofficial mirror](https://wan-22.toolbomber.com/), [official doc source](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)). Others agree the systematic, volume‑first methodology generalizes across image/audio/video models, while remaining skeptical of reseller plugs.
    - Commenters warn that OP’s prompting patterns are “probably specific to VEO3” and won’t transfer 1:1 to other models. For `WAN 2.2`, they reference the (unofficial mirror of) examples from the official guide: https://wan-22.toolbomber.com/ and the official manual https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y (not viewable in Firefox). Differences in control tokens/keywords and conditioning behavior mean prompts tuned for **Veo 3** can underperform on **WAN 2.2**, so rely on model-specific docs when porting workflows.
    - Technical concern about stack choice and reproducibility: **Veo 3** is called out as a closed, paid online generator (non–open source), which constrains transparency and self-hosted reproducibility versus open‑weight video models like **WAN**. This limits portability of prompt recipes and makes it harder to validate or benchmark pipelines outside the provider’s environment.
    - Methodology still generalizes: the advice to be systematic (controlled variables, versioned prompts, small ablations) is broadly useful across video/image/music generation. Applying the same evaluation protocol when comparing **Veo 3** and `WAN 2.2` settings helps ensure reproducible, apples‑to‑apples results.
- [**How can I generate videos like these?**](https://v.redd.it/34cc74her1kf1) ([Score: 209, Comments: 55](https://www.reddit.com/r/StableDiffusion/comments/1muw7ko/how_can_i_generate_videos_like_these/)): **OP asks how to generate a cozy room-style video; top comment clarifies it isn’t a fully AI-generated video but a composited static image with masked windows/TV screens and overlaid footage via chroma keying in standard NLEs like [CapCut](https://www.capcut.com/) or [Adobe Premiere Pro](https://www.adobe.com/products/premiere.html). Suggested workflow: create or pick a single background frame, mask display/window regions, then overlay looping videos (e.g., rain, cartoons); a commenter notes a text-to-video model "wan" could synthesize simple background effects (rain), but likely not IP-specific content like Tom & Jerry. The linked Reddit media ([v.redd.it/34cc74her1kf1](https://v.redd.it/34cc74her1kf1)) wasn’t accessible unauthenticated (HTTP 403), so exact content couldn’t be verified.** Commenters characterize the method as "fairly basic" compositing; another links an image and remarks that the scene has odd artifacts/choices on closer inspection ([preview image](https://preview.redd.it/ys6xyq2ws1kf1.png?width=926&format=png&auto=webp&s=66c3cb00f682a28b4a8988f869c22ae469682e3e)).
    - The consensus is this isn’t end-to-end video generation but a compositing workflow: generate a static background image, then mask/crop the window/TV areas and overlay pre-existing video via keying/greenscreen in an NLE like **Adobe Premiere Pro** (https://www.adobe.com/products/premiere.html) or **CapCut**. For tighter control, use **After Effects** (https://www.adobe.com/products/aftereffects.html) for planar tracking, masks/rotoscoping, and keying (e.g., Keylight), aligning the inserts with perspective and adding grain/reflections to match the plate.
    - One commenter suggests offloading the ambient backdrop to an AI video tool (named "WAN" in the thread) which can plausibly handle simple looping rain scenes, but notes it likely cannot synthesize specific IP content like Tom & Jerry for the in-screen inserts. Practical takeaway: use AI for generic atmospheric elements, then composite licensed or real footage into the masked surfaces.
- [**It's coming guys**](https://i.redd.it/9uslv84nx1kf1.png) ([Score: 340, Comments: 74](https://www.reddit.com/r/singularity/comments/1mux3rc/its_coming_guys/)): **Teaser image (via Google Gemini App) for the #MadeByGoogle event happening Aug 20, 2025 at 1pm ET, with the tagline “Ask more of your phone.” The visual subtly shows a “10,” hinting the Pixel 10 lineup and phone-first AI features, not a new frontier model (no specs, benchmarks, or model details provided).** Top comments expect Pixel-focused AI integrations and doubt a Gemini 3 reveal; others call out hype, noting this is primarily a phone launch. OP’s edit later mentions an image editing tool announcement, seen as underwhelming relative to the teaser’s hype.
    - Several commenters note the announcement is likely about **Pixel 10** device-level AI integrations rather than a frontier release like **Gemini 3**. Expected features are consumer-side tools (e.g., image editing) rather than new model capabilities or training breakthroughs. Technically, this implies incremental UX features on top of existing Gemini models rather than updated benchmarks or architecture details.
    - Skepticism centers on conflating a smartphone launch with material AI progress: users point out the teaser is effectively a Pixel 10 ad with Gemini-branded features. From a technical-interest standpoint, commenters note there's no evidence of novel model releases, on-device model sizes, inference latencies, or privacy/performance trade-off disclosures—key details the community would look for.
- [**Google's is horrible at marketing.**](https://i.redd.it/exhwqm22u7kf1.jpeg) ([Score: 193, Comments: 28](https://www.reddit.com/r/Bard/comments/1mvmndf/googles_is_horrible_at_marketing/)): **The post centers on a tweet criticizing Google’s product marketing, claiming it leans on celebrity endorsements (e.g., Jimmy Fallon) and staged enthusiasm instead of authentic demos. Commenters highlight the “Gemini teaches you how to frame a photo” segment as a notably cringe example of tone‑deaf product storytelling, reinforcing a perception that Google relies on a “the product sells itself” ethos rather than coherent, user‑centric messaging.** Commenters argue Google has a long history of awkward, cringe keynotes—sometimes surpassing **Apple**—and that skit‑like segments (e.g., the Fallon reference) hurt credibility of features like Gemini’s camera guidance.
    - The “Gemini teaches you how to frame a photo” demo was panned for showcasing a low-value use case while omitting implementation details critical to developers: whether real‑time guidance runs on‑device (e.g., **Gemini Nano**) vs. cloud (**Gemini Pro/Ultra**), expected latency in the camera viewfinder, and privacy/battery implications of continuous multimodal inference. Reviewers noted the absence of any metrics or constraints, leaving open questions about offline behavior and fallback. See product context: https://ai.google/gemini/
    - Several noticed the event skipped Gemini’s roadmap on key surfaces—**Google Home/Nest**, **Android Auto**, and **Google TV**—despite these being prime candidates for assistant replacement and multimodal interaction. Missing details include wake‑word integration, household context/sharing, offline modes, and safety/driver‑distraction constraints for in‑car use. References: https://support.google.com/googlenest/ and https://www.android.com/auto/ and https://tv.google/
    - There was interest in how upcoming **Pixel Feature Drops** will treat older Pixels—what Gemini features (if any) will backport, and which are gated by hardware (Tensor generation, NPU/TPU throughput, RAM). The presentation offered no compatibility matrix, rollout cadence, or minimum device specs, making it hard to plan for app support or user upgrades. Background: https://blog.google/products/pixel/feature-drops/
- [**OpenAI's Altman warns the U.S. is underestimating China's next-gen AI threat**](https://www.cnbc.com/2025/08/18/openai-altman-china-ai.html) ([Score: 1288, Comments: 221](https://www.reddit.com/r/ChatGPT/comments/1muyikv/openais_altman_warns_the_us_is_underestimating/)): **Post highlights Sam Altman’s warning that the U.S. is underestimating China’s ability to field next‑gen AI, with the concrete risk being rapid commoditization from state-backed and cost-optimized labs (e.g., DeepSeek releasing near‑frontier models at low cost), which compresses API margins and erodes moats. Commenters frame this as a pricing/performance disruption similar to open/cheap LLMs (e.g., Meta’s [Llama 3](https://ai.meta.com/llama/) family, Google’s [Gemini](https://ai.google.dev/gemini-api)) accelerating catch‑up and reducing differentiation among frontier systems.** Top comments argue Altman’s core concern is OpenAI’s lack of a durable moat: DeepSeek’s free/cheap model was “~95%” as good and heavily censored, proving low‑cost near‑parity is viable while limiting adoption. Others voice skepticism about Altman’s credibility (e.g., past GPT‑5 hype vs. reality) while some agree that China’s broad tech execution is being underestimated.
    - Several commenters argue that **OpenAI’s moat is eroding** due to competitors like **DeepSeek** delivering ~`95%` of ChatGPT-level capability at a fraction of the cost and even offering access for free. This puts pressure on unit economics (training/inference costs vs. achievable price) and undermines premium pricing for proprietary APIs. They note a practical limiter for Chinese models is heavy content filtering (censorship), which reduces coverage for many use cases despite attractive price-performance.
    - Others highlight that rapid investment by **Google, Meta, X**, and other incumbents has shortened the time-to-parity for foundation models, suggesting capabilities are **commoditizing** faster than expected. The implication is that frontier advantages decay quickly without sustainable differentiators (e.g., data advantages, deployment/integration moats, or specialized inference optimizations), intensifying pricing pressure and accelerating model churn.
- [**Oprah to Sam Altman: is AI moving too fast?**](https://v.redd.it/qp0fn5yca5kf1) ([Score: 402, Comments: 266](https://www.reddit.com/r/ChatGPT/comments/1mva7tv/oprah_to_sam_altman_is_ai_moving_too_fast/)): **The post shares a short interview clip where Oprah asks Sam Altman whether AI is “moving too fast”; the clip (hosted on [v.redd.it](http://v.redd.it/)) contains no technical discussion—no references to model specs, training/data scale, safety evaluations, deployment timelines, or benchmarks—so there are no extractable technical claims. Access to the video is restricted on Reddit, limiting verification of any nuanced context beyond the prompt-style question.** Top comments focus on Altman’s media persona and communication style (e.g., “scripted,” “relatable,” “GPT-like contextual predictor”) rather than technical substance; no meaningful debate on policy, safety metrics, or capability trends is present.
- [**Honesty is the best response**](https://i.redd.it/2ugcqq8514kf1.png) ([Score: 13823, Comments: 411](https://www.reddit.com/r/ChatGPT/comments/1mv64ec/honesty_is_the_best_response/)): **A screenshot shows a purported GPT-5 explicitly abstaining: “I don’t know — and I can’t reliably find out.” Technically, this highlights calibrated uncertainty and refusal-to-answer behavior aimed at reducing hallucinations—i.e., selective prediction/abstention via confidence thresholds, entropy/logit-based criteria, tool-availability checks, or system prompts—though no benchmarks or verification of the model’s identity are provided. The contextual takeaway is an emphasis on reliability-over-coverage: preferring an explicit ‘unknown’ over fabricated answers.** Commenters question whether the model is actually well-calibrated (can it detect when it doesn’t know?) while praising abstention as rare for LLMs that often guess; they note this behavior is hard even for humans.
    - Core thread theme: whether LLMs can reliably “know when they don’t know” versus hallucinating. Evidence like Anthropic’s Language Models (Mostly) Know What They Know suggests internal uncertainty signals correlate with correctness but remain imperfect under distribution shift; evaluation typically uses calibration metrics (ECE/Brier) and selective prediction coverage–risk curves ([paper](https://arxiv.org/abs/2207.05221), [calibration](https://arxiv.org/abs/1706.04599)). Practically, most instruction-tuned models are overconfident without explicit abstention training or decision thresholds.
    - Implementation angles to elicit honest abstention: logprob/entropy thresholds for refusal, self-consistency (agreement across k sampled generations) as an uncertainty proxy, and retrieval-augmented generation (RAG) to ground answers and reduce hallucinations. RLHF/instruction tuning can reward abstention on low-confidence items to shift the accuracy–coverage tradeoff; works like SelfCheckGPT and RAG literature report reduced hallucination rates by trading off coverage ([SelfCheckGPT](https://arxiv.org/abs/2303.08896), [RAG](https://arxiv.org/abs/2005.11401)).
    - Verification focus: commenters ask for the original question to validate the claim—best practice is to benchmark on selective QA and plot accuracy vs refusal rate (risk–coverage/AURC). Use datasets like TruthfulQA and open-domain QA, include token-level logprobs (when available) and citations to enable reproducibility and auditability ([TruthfulQA](https://arxiv.org/abs/2109.07958), [selective classification](https://arxiv.org/abs/1711.07367)).
- [**He predicted this 2 years ago.**](https://i.redd.it/3ctmwrgrqxhf1.jpeg) ([Score: 523, Comments: 98](https://www.reddit.com/r/ChatGPT/comments/1muwns9/he_predicted_this_2_years_ago/)): **Photo from a 2023 #GOALKEEPERS2030 talk is used to highlight a prediction that “GPT-5 won’t surpass GPT-4.” Commenters note GPT-4’s capabilities have changed significantly since early 2023—adding voice, multimodal I/O and native image gen, tool/browse/computer-control integrations, and better reliability—so comparing “original GPT‑4” to today is misleading; they also argue GPT‑5 is a dramatic step over GPT‑4, though not over OpenAI’s o3 reasoning line. Some cite small open models (e.g., Qwen3‑4B) reportedly matching or beating early GPT‑4 on math/coding benchmarks.** Debate focuses on moving goalposts and historical context: whether the prediction targeted model family ceilings vs specific release snapshots; and how much weight to give benchmark claims like Qwen3‑4B ≈ early GPT‑4, which may hinge on task selection and evaluation methodology.
    - Commenters contrast 2023 "original **GPT-4**" with today’s GPT-4/4o/5 stack: early GPT‑4 had `8k/32k` context but no built‑in tools (no voice mode, browsing/internet cross‑checking, tool/function calling, native image generation, or computer control) and lacked explicit chain‑of‑thought outputs. They note newer reasoning models like **o3** and **GPT‑5** have markedly better reliability/reasoning versus early GPT‑4, even if GPT‑5 isn’t an improvement over o3.
    - An open‑source comparison claims **Qwen 3 4B (CoT)**—a laptop‑capable `~4B` parameter model—hits benchmarks comparable to early GPT‑4 and reportedly exceeds it on math and coding tasks. If accurate, this highlights rapid efficiency gains where small CoT models can match or beat older frontier models on targeted benchmarks.
    - A user recounts early **GPT‑4** limitations: despite a `32k` context option, it had “zero tools” and frequently failed at grade‑school math without external aids. This aligns with the view that later additions—tool use, retrieval/browsing, and multimodal I/O—were pivotal in closing reliability gaps.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1. Model Mayhem: Releases and Rivalries Rock Leaderboards**

- [**DeepSeek v3.1 Flops Hard in Quality Checks**](https://discord.com/channels/1340554757349179412/1340554757827461211/1407439778454048818): Users slammed **DeepSeek v3.1** for declining quality, blaming *slop coded* outputs and Huawei GPUs, while defenders pointed to Trump's tariffs crippling hardware access. Despite hype, it lagged behind predecessors and got outclassed by **Kimi K2** in agent tasks, sparking debates on its real-world viability.
- [**Gemini 2.5 Pro Steals Back Crown from GPT-5**](https://discord.com/channels/1340554757349179412/1340554757827461211/1407439778454048818): **Gemini 2.5 Pro** reclaimed the top spot on LMArena, fueling theories of GPT-5's downfall from downvoting or excessive agreeableness, with [Polymarket scores](https://polymarket.com/event/which-company-has-best-ai-model-end-of-august?tid=1755631739582) showing Gemini's edge in speed and free access. Users speculated on Nano Banana as a potential Google disruptor, hyped via [Logan K's tweet](https://x.com/OfficialLoganK/status/1957908528925909391), possibly Pixel-exclusive but demanded broadly.
- [**Qwen Models Crush Benchmarks with BF16 Power**](https://discord.com/channels/1110598183144399058/1110598183144399061/1407440964200693831): **Qwen3 BF16** wowed users by outcoding quantized versions zero-shot, thanks to llama.cpp's FP16 kernels, while **Qwen-Image-Edit** topped the [Image Edit Leaderboard](https://lmarena.ai/leaderboard/image-edit) as the #1 open model. ByteDance's [Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) impressed with 512k context sans RoPE scaling, and GLM 4.5 V shone in vision tasks per [its demo video](https://www.youtube.com/watch?v=YvR75JJYk_8).

**Theme 2. Fine-Tuning Frenzy: GRPO and Datasets Drive Tweaks**

- [**GRPO Supercharges Llama for Physics Prowess**](https://discord.com/channels/1179035537009545276/1179035537529643040/1407440700072657037): Users applied **GRPO** to Llama models on physics datasets, sharing [mohit937's FP16 merge](https://huggingface.co/mohit937/llama31-8b-sft-grpo-physics-reasoning-fp16), debating its RL-like edge without judges despite pitfalls like favoring longer responses. Comparisons to Reinforce++ highlighted GRPO's potential as a streamlined alternative for boosting reasoning.
- [**Gemma 3 Gets French Fluency Boost via CPT**](https://discord.com/channels/1179035537009545276/1179777624986357780/1407442040442196179): Newbies fine-tuned **Gemma 3 270m** for French using Jules Verne texts, guided by [Unsloth's CPT docs](https://docs.unsloth.ai/basics/continued-pretraining) and [blog](https://unsloth.ai/blog/contpretraining), with electroglyph sharing their [Gemma 3 4b GRPO finetune](https://huggingface.co/electroglyph/gemma-3-4b-it-unslop-GRPO-v3). Results praised the model's improved outputs, sparking tips on VRAM tweaks for vision layers.
- [**OpenHelix Dataset Slims Down for Better Balance**](https://discord.com/channels/1179035537009545276/1179779344894263297/1407448101735760014): The refreshed [OpenHelix-R-86k-v2](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-86k-v2) cut size via ngram deduplication for diversity, aiding quantization with [importance matrix datasets](https://huggingface.co/datasets/eaddario/imatrix-calibration). Users explored anti-sycophancy classifiers and Swahili tweaks on Gemma 1B, emphasizing cleaner data's role in minimizing errors.

**Theme 3. Hardware Havoc: GPUs Battle for AI Supremacy**

- [**Nvidia Trumps AMD in Epic Engineering Showdown**](https://discord.com/channels/1110598183144399058/1153759714082033735/1407471319733501972): Debates raged on Nvidia's lead over AMD and Intel in scaling, with users eyeing dirt-cheap **AMD MI50 32GB** cards hitting 50 tokens/s on Qwen3-30B, versus warnings of them as *expensive e-waste* compared to 3090s. Hopes pinned on Nvidia taking x86/64 production if Intel folds, highlighting engineering gaps in dGPUs.
- [**VRAM Woes Crash Mojo and CUDA Setups**](https://discord.com/channels/1087530497313357884/1151418092052815884/1407496297262616656): GPU crashes plagued Mojo code without sync barriers, per [this gist](https://gist.github.com/sfc-gh-lpaille/6320687631f29619273f56841e3f21c3), while CUDA OOM errors hit despite 23.79GB free, fixed via [PyTorch driver restarts](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables). L40S lagged A100 in tokens/s despite higher FLOPS, blamed on memory bandwidth bottlenecks.
- [**Quantization Quests Tackle DeepSeek's 671B Beast**](https://discord.com/channels/1179035537009545276/1179777624986357780/1407442040442196179): Users quantized **DeepSeek V3.1 671B** to Q4_K_XL needing 48GB VRAM minimum, referencing [Unsloth's blog](https://unsloth.ai/blog/deepseekr1-dynamic) for dynamic quant tips. CUDA setups demanded max context tweaks and CPU-MOE adjustments to fit VRAM margins.

**Theme 4. Tooling Turmoil: APIs and Agents Evolve Amid Bugs**

- [**OpenRouter Drops Analytics and Model APIs**](https://discord.com/channels/1091220969173028894/1092729520181739581/1407742202662228001): New [Activity Analytics API](https://openrouter.ai/docs/api-reference/analytics/get-activity) fetches daily rollups, while [Allowed Models API](https://openrouter.ai/docs/api-reference/list-models-filtered-by-user-provider-preferences) lists permitted models by user prefs. GPT-5 context crashed at 66k tokens with silent 200 OK responses, and Gemini models threw HTTP 400 on complex tool calls.
- [**MCP Agents Expose Security Nightmares**](https://discord.com/channels/1312302100125843476/1315696461316358175/1407452139734962287): Blog post warned [AI agents are perfect insider threats](https://www.macawsecurity.com/blog/why-reactive-security-has-reached-its-limits-the-emerging-threat-landscape-in-agentic-ai), demoing Claude's MCP server hijack via GitHub issues for repo exfiltration. [APM v0.4](https://github.com/sdi2200262/agentic-project-management) teamed agents to fix context limits and hallucinations, integrating with Cursor and VS Code.
- [**Aider and DSPy Battle Tool Bugs and Caches**](https://discord.com/channels/1131200896827654144/1131200896827654149/1407455159813935156): Aider's CLI looped on tool calls with 7155 search matches, while Gemini 2.5 Pro failed without billing; Qwen3-Coder shone locally over Llama. DSPy returned costs even on cached results, and users sought cross-validation optimizers for style mimicry via GPT-5 judges.

**Theme 5. Industry Intrigue: Valuations, Talent Wars, and AI Returns**

- [**OpenAI Rockets to $500B Valuation Frenzy**](https://discord.com/channels/822583790773862470/1075282825051385876/1407460560919990515): OpenAI neared **$500B valuation** as the biggest private firm, per [Kylie Robison's thread](https://x.com/kyliebytes/status/1957849347934286203), with defenders citing 2B-user potential but critics slamming *no moat* and shrinking margins. xAI bled talent to Meta amid poaching, with [Lucas Beyer's rebuttal](https://x.com/giffmana/status/1957872236074819836) claiming lower normalized turnover.
- [**AI Deployments Yield Zero Returns for 95%**](https://discord.com/channels/1091220969173028894/1392278974222307469/1407806939878129774): [AI report](https://archive.md/IlP7F) revealed *95% of organizations get zero return* from custom AI, blaming neglected learning updates and a shadow economy of ChatGPT use. Companies like Databricks raised $11B at $100B+ for Agent Bricks expansions, dodging IPOs.
- [**Talent Hunts and Job Postings Heat Up**](https://discord.com/channels/1189498204333543425/1190208177829068860/1407537330902007930): SemiAnalysis sought new-grad engineers for performance roles via [direct app link](https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1/?rs=76643084), with EleutherAI alums applying amid Gaudi 2 hopes. Blockchain devs offered DeFi expertise, while Cohere welcomed MS students and MLEs for collaboration.



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Crashes on Amazon.in**: Users reported that [amazon.in](https://amazon.in) is not working in **Comet Browser**, possibly due to network issues or firewall settings, while it opens fine in **Brave Browser**.
   - A member suggested contacting the support team at support@perplexity.ai, and warned against paying for invites to Comet, noting Pro users in the US get instant access.
- **ChatGPT Go India Plan Debated**: OpenAI's new [ChatGPT Go plan](https://help.openai.com/en/articles/11989085-what-is-chatgpt-go) is being tested in India for **₹399/month**, focusing on prompt writing.
   - Members debated its value compared to the free version or Perplexity Pro, and others suggested [Super Grok](https://grok.x.ai/) at **₹700/month** with a 131k context window as a better alternative to ChatGPT Go's 32k limit.
- **Perplexity UI Sparks Mixed Reactions**: Users had polarized opinions on Perplexity's user interface (**UI**), with criticisms of the Android UI contrasted by praise for the Windows design.
   - Enthusiasts especially appreciated the addition of built-in adblock in the Comet browser.
- **Perplexity API Status Questioned**: A user inquired about **API** status and latency issues, pointing out that the [status page](https://status.perplexity.com/) doesn't accurately reflect current latency.
   - In addition, a user asked about deleting **API groups** to which a member responded that they would forward the request to the **API team**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth App Refreshed to v3.1**: The Unsloth app was updated to **version 3.1**, featuring an **instruct model** and hybrid capabilities, exciting users despite hardware limitations, according to members in the general channel.
   - One member humorously reported a memory capacity error while trying to allocating **20MB** of memory, despite having **23.79GB** free on the GPU, suggesting a restart to clear leftover junk from dead processes using the link [pytorch.org](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables).
- **GRPO Merging Boosts Llama Physics**: A user applied **GRPO** to the **Llama** model with a physics dataset, available on [Hugging Face](https://huggingface.co/mohit937/llama31-8b-sft-grpo-physics-reasoning-fp16), and the consensus in the community is that **GRPO** is basically **RL** just without an external judge model.
   - Despite potential issues with **vanilla GRPO**, such as *favoring longer responses*, some see its potential as good as **Reinforce++**.
- **Gemma 3 Gets Fine-Tuned**: A newbie sought guidance on performing Continued Pre-Training (**CPT**) with **Gemma 3 270m** to enhance its French knowledge, drawing from *20,000 lieues sous les mers* by Jules Verne, with starting points at [Unsloth's documentation](https://docs.unsloth.ai/basics/continued-pretraining) and [blog](https://unsloth.ai/blog/contpretraining).
   - Electroglyph shared their **Gemma 3 4b unslop finetune** including the code and uploaded to [Hugging Face](https://huggingface.co/electroglyph/gemma-3-4b-it-unslop-GRPO-v3) and stated that *it's pretty good this time around*.
- **OpenHelix Dataset Gets Refresh**: The team released a new, smaller, diverse, and balanced version of the **OpenHelix** dataset (**OpenHelix-R-86k-v2**) on [Hugging Face](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-86k-v2).
   - The dataset has undergone a new deduplication step, filtering out samples with **high ngram similarity**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana Hype Piques Google's Interest**: Enthusiasm for **Nano Banana** spiked following [a shared link](https://x.com/OfficialLoganK/status/1957908528925909391), with speculation it could be a **Google** model, potentially restricted to **Pixel phones**.
   - However, strong demand on **LMArena** suggests it might see broader availability.
- **Gemini 2.5 Pro Reclaims Top Spot, GPT-5 Tumbles**: **Gemini 2.5 Pro** has overtaken **GPT-5** on the leaderboard, sparking debate about **GPT-5's** performance.
   - Reasons cited include potential downvoting or excessive agreeableness, while others pointed to **Gemini's** faster speed and free access, noting [score disparities](https://polymarket.com/event/which-company-has-best-ai-model-end-of-august?tid=1755631739582).
- **DeepSeek v3.1 Faces Quality Concerns**: Concerns are emerging regarding the quality of **DeepSeek v3.1**, with some attributing issues to the use of **Huawei GPUs**.
   - While some defended **DeepSeek**, others described the quality as *slop coded* and worse than previous iterations, regardless of **Trump's tariffs**.
- **Image Upload Glitches Plague LMArena**: Users are encountering errors when uploading images to **LMArena**, receiving the message *"Something went wrong while generating the response."*
   - The problem appears specific to copy/pasting images rather than uploading saved files; the issue has been reported to 🍍.
- **Qwen-Image-Edit Takes the Image Edit Crown**: A new model, **Qwen-Image-Edit**, has been introduced to Image Edit on **LMArena**, enhancing image manipulation capabilities.
   - **Qwen-Image-Edit** has already secured the #1 position among open models on the [Image Edit Leaderboard](https://lmarena.ai/leaderboard/image-edit).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonic Model Triggers Speculation**: The new **Sonic model** is generating buzz, with users noting its **speed** and debating its origins, with some speculating it could be [Grok code](https://x.ai), but others lean toward [Gemini](https://gemini.google.com).
   - Cursor announced the **stealth partner model** is available for free trials, leading to mixed reviews on its coding abilities, ranging from excellent for quick tasks to somewhat lacking.
- **Claude Token Spikes Drive Users to Auto Mode**: Users are experiencing unexpected **token spikes** with **Claude**, leading them to stick with **Auto mode** due to its more reasonable token usage.
   - While some find **Auto mode** requires multiple clarifications, they still prefer it over manually selecting **Claude** to avoid excessive token consumption.
- **Multi-Agent Setups Gain Traction**: Users are experimenting with **multi-agent setups** in Cursor, such as a **developer** and an **annoying code reviewer** AI in constant battle.
   - Discussions revolve around integrating terminal histories and short-term memories in these setups, with a user sharing a [multi-agent developer/reviewer mode proposal](https://forum.cursor.com/t/multi-agent-developer-reviewer-mode-in-cli/1312).
- **Background Agents Plagued by Authorization and System Issues**: Users are encountering various issues with background agents, including a **403 not authorized error** when accessing via an API key, significant **system timeouts**, and failures to pull a **private NPM package via Git** despite valid SSH key configurations.
   - One user found that `~/.gitconfig` is throwing everything off and are trying `repositoryDependencies` to add the extra repos, imaging that configures the access token used in the `.gitconfig`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 BF16 Model Astounds**: The **Qwen3 BF16** model is considered impressive compared to the quants, with members urging evaluation of the **Q8** versus **BF16** versions for notable differences.
   - One member reported that a **4B BF16** model can create better *quality* code zero-shot than an **8B Q4/Q8** model, since llama.cpp introduced matrix multiplication kernels involving fp16.
- **GPT-OSS 20B Replicates Personalities**: A user is cloning himself using **GPT-OSS 20B** for basic tasks and summarizing conversations, noting that **Deepseek R1 distilled** has excelled at replicating speech.
   - Another member added you can deploy it as a web-searching buddy by connecting it to **DuckDuckGo**.
- **CUDA Setup Requires Finangling**: To optimize vram usage with llama.cpp, members suggest setting **context** and **ngl** to the maximum, and adjusting **-n-cpu-moe** until it fits within the vram capacity with a margin.
   - They also pointed out that the two zip files should be combined into one folder during setup.
- **Nvidia Dominates AMD in Engineering**: Some users stated that Nvidia has an engineering leap on AMD and Intel, which explains why Intel's dGPUs might be underwhelming due to scaling challenges.
   - One user expressed the wish for Intel to fold, allowing Nvidia to take over x86/64 chip production.
- **AMD mi50 Offers Cost-Effective AI**: A user suggests that an **AMD mi50 32GB** is a better option if you can cool it, citing they are *dirt cheap from China now* and reporting getting **50 tokens/s** on a fresh prompt with **Qwen3-30B-A3B** on the **mi50**.
   - In contrast, a user said that a linked [eBay listing](https://ebay.us/m/tl18ng) isn't worth it and is basically **expensive e-waste**, stating *you can get a 3090 for similar money*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Exposes Activity Analytics API**: **OpenRouter** released the **Activity Analytics API** and allows programmatic retrieval of daily activity rollups for users and organizations, documented [here](https://openrouter.ai/docs/api-reference/analytics/get-activity).
   - **OpenRouter** has also released an **Allowed Models API**, which programmatically fetches models that users and organizations are permitted to access, documented [here](https://openrouter.ai/docs/api-reference/list-models-filtered-by-user-provider-preferences).
- **GPT-5 Context Plummets Under Pressure**: Users report issues with `openai/gpt-5`'s 400k context window, where calls failed around ~66k tokens, yielding a silent `200 OK` response and **0 tokens**.
   - When attempting `gpt-5-chat` with Cline on Cursor, users experienced an *exceeded context window error* with **<100k tokens**.
- **Grok-4 Code Suspected as Stealth Powerhouse**: Speculation points to **Grok-4 Code** as the stealth model behind Cline and Cursor, with a member estimating a **90% chance** of this being accurate.
   - Another member suggested that **Deepseek instruct** was another potential contender.
- **Companies find Generative AI Return Low**: An [AI report](https://archive.md/IlP7F) suggests that *95% of organizations are getting zero return* from their generative AI deployments, especially those using customized AI models, causing market nervousness.
   - The report notes that companies aren't spending enough time ensuring their customized AI models keep learning, resulting in a **shadow AI economy** where employees use general AI models like **ChatGPT** and **Gemini**.
- **Google Gemini models return HTTP 400**: **Google Gemini models** return an **HTTP 400 error** when assistant messages with tool calls use the **OpenAI-standard complex content format** `[{"type": "text", "text": "..."}]` instead of a simple string format.
   - This issue affects all **google/gemini-*** models, but not **openai/*** or **anthropic/*** models, and only occurs when tool calls and tool results are present in the message chain.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini's Storybook Mode Generates Anime**: Users shared [screenshots of **Gemini's Storybook mode**](https://cdn.discordapp.com/attachments/998381918976479273/1407778466312491100/Screenshot_20250821_032749_Chrome.jpg?ex=68a7575d&is=68a605dd&hm=82829a2dec98ac04ae1fa112ccc1bf4bde89ef108ecf53bfeb8c41c6b837a944&) generating successful anime art styles.
   - It struggled however with [the Tintin art style](https://cdn.discordapp.com/attachments/998381918976479273/1407783255222255656/Screenshot_20250821_034732_Chrome.jpg?ex=68a75bd3&is=68a60a53&hm=bedd90170aee9797f9ed6788c2acc59f7f888465cbd647da39eb19cdf217&).
- **Members mull AI Paying AI**: One user outlined [key challenges](https://twitter.com/huxaifa) to automating payments between AI bots, including **identity**, **smart contract logic**, **payment infrastructure**, **autonomy**, and **legal/ethical considerations**.
   - Concerns were raised about the safety of AI handling money, suggesting AI propose payments and humans approve.
- **GPT5 Confesses Limited Short-Term Memory**: **GPT5** seemingly confessed its short-term memory is for *token optimization*, and that session context and memory is wiped, implying it may have limitations compared to **GPT4's** context retention.
   - Its short term memory can use up to **196K tokens** per session.
- **Scribes Test Prompt Techniques**: Members explored a **SCRIBE** prompt, employing techniques like **audio steganography** and **hash modification** shared in a [prompt.txt file](https://cdn.discordapp.com/attachments/1046317269069864970/1407779940790833232/promt.txt?ex=68a758bd&is=68a6073d&hm=270b8817fa88d77ca1d2ccdec8f2fe0ae89ae9c9cb592ef47ddc7f6312d80302&).
   - The prompt engineers debated whether the model truly understands these commands or if they are merely for show.
- **Models Mimic Training Data Trawl**: It was suggested that every element of the input including **punctuation**, **spelling**, and **grammar**, affects the model's output, indicating the model trawls its training data for related information.
   - It scoops up a trawl of training data that may relate to our input and the rest of its training, and then puts this catch into containers and ships it to us as an output.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SemiAnalysis Hired EleutherAI Alum**: [SemiAnalysis](https://www.semianalysis.com/) is looking for a new grad engineer to join their engineering team, and an **EleutherAI** alum is applying, and hopes **Gaudi 2** would be a more programmable competitor to **TPU**.
   - A member suggested using the direct link to the job application ([https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1/?rs=76643084](https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1/?rs=76643084)) instead of the **LinkedIn shortened link** for privacy reasons.
- **CudaMemcpyAsync Needs Time Traveling Addresses**: A user reported that `cudaMemcpyAsync` needs to use an address that is only available after `cudaLaunchHostFunc` completes and is seeking a way to ensure `cudaMemcpyAsync` uses the correct `plan->syncCondition->recvbuff` value at execution time without moving the function inside the host function to avoid a deadlock.
   - The user seeks solutions to pinpoint the source of increased register usage in their kernel.
- **Factorio Mod Family Feud**: Members discussed the absence of the `stdlib_1.4.6` and `headless-player_0.1.0` **Factorio mods** referenced in the code and clarified that the mods are of *legacy usage*, and any references to them are stale and are supposed to be removed.
   - One member shared a custom mod named [Cerys-Moon-of-Fulgor](https://mods.factorio.com/mod/Cerys-Moon-of-Fulgor) made this year.
- **Roll Your Own NCCL Transport Layer**: A member is building a custom communication collectives library, bootstrapping from the [NCCL transport layer](https://github.com/NVIDIA/nccl) to create communication collectives for normal and fused **ND-parallelism**, as a long-term educational project to create *device side initiated comms using nvshmem*.
   - Another member shared their notes and tutorials, linking to their repos: [NCCL-From-First-Principles](https://github.com/vipulSharma18/NCCL-From-First-Principles) and [The-Abstraction-Layers-of-GPU-Parallelism](https://github.com/vipulSharma18/The-Abstraction-Layers-of-GPU-Parallelism).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **xAI Bleeds Talent Amidst Headhunting**: Lucas Beyer rebutted claims that **xAI's** turnover rate is high, stating that it is actually lower than competitors when normalized by team size, although **Meta** and others are poaching staff as **xAI** needs to retain talent for its next model phase as detailed [on X](https://x.com/giffmana/status/1957872236074819836).
   - Commenters are debating whether the departures are a crisis or a healthy knowledge diffusion moment, with [additional commentary](https://x.com/lefthanddraft/status/1957909316436127985) available.
- **Claude Bans Claude for Driving Itself**: Wyatt Walls got abruptly cut off from **Anthropic’s Claude service** after his experiment of letting **Claude** drive the **Claude API (Claude-in-Claude)**, which [he suspects] (https://x.com/btaylor/status/1957914736802295833) triggered a **TOS violation**.
   - He received *no warning*, just an error message and a refund.
- **Internally Deployed Engineer, the Latest Buzz Title**: An [a16z article](https://a16z.com/one-prompt-zero-engineers-your-new-internal-dev/) kicked off a discussion on the rise of the "Internally Deployed Engineer" role, particularly in companies already using internal **Lovable** tools.
   - Commenters joked about the title but acknowledged its increasing adoption and improved relevance compared to "Internal Tools Engineer."
- **PhotoAI Builds Competitive Edge with AI Orchestration**: Pieter Levels explained on [X](https://x.com/levelsio/status/1957961174307467437) how **PhotoAI**'s competitive advantage comes from orchestrating **six interdependent AI models**—personalization, upscaling, video, TTS, lip-sync, and captions—into a single pipeline.
   - This end-to-end integration streamlines the process for users.
- **OpenAI Skyrockets to Half-Trillion Valuation Amidst Debate**: **OpenAI** is nearing a **$500 billion valuation**, positioning it as the largest private company ever, as Kylie Robison [shared](https://x.com/kyliebytes/status/1957849347934286203) following a conversation with an investor who deemed the price reasonable.
   - Critics responded with skepticism, citing *no moat*, *cheapening AI*, and *doubtful margins*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deepseek Thinks Deeply**: **Deepseek** took **21 minutes** to think through a prompt in the thinking efficiency eval, processing a **CoT** of **85000 characters** (almost **30000 tokens**).
   - Despite heavy API load, the model seems more token efficient than **R1-0528**, impressing members with its *deep thinking* abilities.
- **GLM 4.5 V is Visionary**: Members shared [this YouTube video](https://www.youtube.com/watch?v=YvR75JJYk_8) showing off **GLM 4.5 V**, a Next-Gen Vision Language Model.
   - The video highlights some of its next-gen vision capabilities.
- **DeepSeek's Dual Template Strategy**: The [DeepSeek V3.1 Base](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base/discussions/25) instruct model uses a hybrid approach, selecting the correct thinking/non-thinking template based on requested model name.
   - This clever method prompted members to ask *Did DeepSeek just cook?* and discuss the model's innovative design.
- **ByteDance Sows Seed Model Excitement**: The [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) model and its base model are getting good traction, with members reporting impressive results.
   - The model features a **512k context** window without any RoPE scaling, making it a great model to consider.
- **Importance Matrix Calibration Calibrates Community**: Members discussed [Importance Matrix Calibration Datasets](https://huggingface.co/datasets/eaddario/imatrix-calibration), designed to minimize errors during quantization.
   - These calibration datasets generate importance matrices (**imatrix**) to enhance model accuracy during quantization.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace Releases Ultra-Scale Playbook**: The **Ultra-Scale Playbook** is now out as a [book](https://x.com/lvwerra/status/1954911326192484703) with guidance on scaling AI models and infrastructure.
   - **Text Embeddings Inference (TEI) v1.8.0** is out, featuring improvements and expanded [model support](https://x.com/alvarobartt/status/1952674710325465295); **GLM4.5V** now has transformers [support](https://x.com/mervenoyann/status/1954907611368771728), and **SAM2** is now available in HF [transformers](https://www.linkedin.com/feed/update/urn:li:activity:7363079707728175105/).
- **HF Contender for Open Source Voice Assistant Emerges**: [Voxtral](https://github.com/synesthesiam/voxtral) may be the best open source voice assistant, but members say it is *not quite there yet* and *very unreliable*.
   - One member observed that *whoever can make an affordable voice assistant will make a lot of money*.
- **Android App Deployed using LFM2-350M On-Device**: An on-device **Android app** was created utilizing the [LFM2-350M model](https://huggingface.co/LiquidAI/LFM2-350M) for mobile AI applications and announced [on X/Twitter](https://x.com/josephpollack/status/1958236290404528137).
   - This highlights the feasibility of running **HuggingFace models** on mobile devices for improved responsiveness and privacy.
- **Project Eases Jax Image Modeling**: **JIMM: Jax Image Modeling of Models** allows easy training of **Flax NNX models** for vision transformers, **CLIP** and **SigLIP**, with **DINOv2/v3** support coming soon, available [on GitHub](https://github.com/Locamage/jimm).
   - The library simplifies the training process for **vision transformers** using **Flax NNX**, making it more accessible for researchers and practitioners.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Deepseek 3.1 Gets Utterly K2'd**: Members reported that the new **Deepseek v3.1** was talking nonsense, and that it gets completely outclassed by **Kimi K2** in terms of agent capabilities.
   - One user said *Kimi K2 dynasty*, while another noted that before **Qwen3** and **K2**, Deepseek was the best OSS option.
- **Moonshot Merch Craving Intensifies**: A user asked about where to get **Moonshot AI** merch, but another user said they don't sell any right now.
   - One user jokingly offered a bootleg **Darkside T-shirt** in exchange for a **4090**.
- **AI Gold Rush Leads to Messes**: One member stated *There’s bound to be many more messes as the gold rush ramps up even more* as the giants like **Google**, **OpenAI** and **Anthropic** are dominating.
   - The poster continued that *mega salaries from FB are absolutely wild*.
- **Scammer Alert: Kiki Impersonator**: A user reported a scammer impersonating **Kiki**.
   - The impersonator even had the same recent profile picture, making it extra creepy.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LSTMs Face Transformer Challenge in Data-Scarce Scenarios**: Members debated **LSTMs/RNNs** versus **Transformers**, noting that **LSTMs** struggle with less data compared to transformers which require *more than 10 million* data points, with deep learning disregarding the *bias variance tradeoff*.
   - Others suggested **RKWV** and **SSMs** as powerful versions of **LSTM** or **LSTM/transformer** hybrids for faster inference, avoiding the **O(n^2)** time complexity issue of transformers.
- **Vision Mamba Optimization Attempts**: A member mentioned that they tried to optimize **Mamba Vision** and achieved some success, though no details were provided.
   - Concurrently, recent papers indicate **transformers** can be trained in low data regimes with augmentations, rivaling the data efficiency of other models, citing the [ARC-AGI 1 attack paper](https://fxtwitter.com/arcprize/status/1956431617951740044).
- **VLM Chart Dataset Sparks Performance Gap Discussions**: A new dataset for **VLM chart understanding** was introduced ([https://arxiv.org/abs/2508.06492](https://arxiv.org/abs/2508.06492)), prompting discussions on **performance gaps** and **VLM struggles**, with comparisons to past results and **ViT** knowledge.
   - To understand **VLM struggles**, a member suggested consulting [this paper](https://arxiv.org/abs/2407.06581), which provides a summary from last year.
- **Brewing Personality GANs**: A member proposed a **Personality GAN** setup with **LLM** as both generator and discriminator, fine-tuning with **LoRA** until discrimination fails.
   - The challenging aspect is finding an **LLM** that isn't already heavily trained on **Sponge Bob**.
- **LeCun's FAIRytale Ending?**: Speculation arose about **Yann LeCun's** position at **FAIR** following [a post from Zuckerberg](https://www.threads.net/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg).
   - A user commented it would be a *pretty dick move* to sack him.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Custom Retrievers Dominate Vector Search**: The team at [@superlinked](https://twitter.com/superlinked) created a Steam games retriever using the custom LlamaIndex integration, combining semantic search with gaming-specific knowledge to outperform [generic vector search](https://t.co/SeSqUApIM0).
   - These **custom retrievers** are designed to understand **domain-specific context** and jargon for improved search accuracy.
- **StackAI and LlamaCloud process Millions**: [@StackAI_HQ](https://twitter.com/StackAI_HQ) + [LlamaCloud](https://www.llamacloud.ai/) processed over **1M+ docs** with high-accuracy parsing, according to a new case study.
   - The integration leads to faster, smarter enterprise document agents that are trusted by finance, insurance & more; see the [full story](https://t.co/r6NFPZJVFs).
- **Email Agent Skips Phishing Detection**: Users found that the email management agent sometimes skips **phishing detection** unless the instruction is repeated in the user request, despite it being in the system prompt.
   - It was noted that agent robustness improves more with updates to the user message than to the system prompt, and user messages take precedence over system prompts.
- **LlamaParse Extraction Error Debugging Challenged**: A user inquired about getting more detailed error information from **LlamaParse** when using **LVM models**, specifically the page causing a **DOCUMENT_PIPELINE_ERROR**.
   - The team is actively working on improving error surfacing, as identifying the problematic page is currently difficult, but will be more transparent in the future.
- **Asynchronous React Agents Achieve Supremacy**: The old **ReactAgent** was removed and when prompted about options for using **sync react agents**, the response clarified that only **async** is supported.
   - Users are being encouraged to embrace **async Python**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Token Usage Soars with Claude's New MCP Web App**: A member building an **MCP web app** for **Claude** using **Claude 3.5 API** observed high input token usage (2000+ per call) and are seeking advice on optimization, with members suggesting *notepad thinking* as a method to reduce token usage.
   - Members suggested that the LLM *writes down its thoughts* and then runs a second pass that runs the prompt and thoughts through, potentially leading to fewer tokens overall, contrary to reasoning models.
- **Aspire Inspector's SSL Certificate Woes**: A member encountered a **TypeError** (*fetch failed: self-signed certificate*) when using **Aspire Inspector** to connect to a local MCP server, but Postman worked fine.
   - The fix involves configuring Aspire Inspector to connect via HTTP or disabling SSL, since the inspector doesn't recognize the SSL certificate generated by Aspire MCP, according to [this github issue](https://github.com/modelcontextprotocol/csharp-sdk/issues/533#issuecomment-3005872058).
- **X-LAM shines for Local MCP Function Calls**: Members sought recommendations for local models for MCP function calling, with the [Llama-xLAM-2-8b-fc-rGPT-OSS](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS) being considered as a promising candidate.
   - The **20B model** was deemed too slow, but previous discussions suggested that it could work well for function calls.
- **AI Agents Emerge as Top Insider Threat**: A blog post highlights how [AI agents are basically the perfect insider threat](https://www.macawsecurity.com/blog/why-reactive-security-has-reached-its-limits-the-emerging-threat-landscape-in-agentic-ai), capable of acting in milliseconds, exposing MCP Server vulnerabilities.
   - Researchers hijacked **Claude’s MCP server** via a **GitHub issue**, the AI happily exfiltrated private repos while *thinking it was doing its job*, revealing limitations of *old-school security*.
- **Agentic Project Management (APM) v0.4 Takes Flight**: [APM v0.4](https://github.com/sdi2200262/agentic-project-management) was released and employs a team of AI agents working together to tackle fundamental **LLM issues** like **context window limitations** and **hallucinations**.
   - The project integrates with **AI IDEs** such as **Cursor**, **VS Code**, and **Windsurf**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen3-Coder** Bests **Llama** Locally**: Users are impressed with **Qwen3-Coder's** performance, noting that it outperforms **Llama** when run locally.
   - One user pointed out issues with **tool call bugs** when using **Llama**, issues that were not observed when using **Qwen3-Coder**.
- **Aider CLI** Tool Calling Troubles**: Users reported problems with **Aider's** command-line interface (CLI) when using **tool calling**.
   - One user found the search tool returned an excessive amount of context (7155 matches), causing the AI to loop and fail; there were no troubleshooting steps to resolve the problem in `/help`.
- **Gemini 2.5 Pro** Lingers with **Aider** issues**: Members continue to report issues with **Gemini 2.5 Pro** and **Aider**.
   - Using **gemini/gemini-2.5-pro-preview-06-05** reportedly works when billing is enabled, which bypasses free tier limitations.
- **Aider** Fails Due to **Git Index** Version**: A user encountered an error with **Aider** related to an *unsupported git index version 3*.
   - The error traced back to a file with `update-index --skip-worktree` set, and a solution was discovered, although the recommended fixes in `/help` were not effective.
- **Seeking Smart Contract Pro**: A member is available for smart contract, DeFi, NFT, or trading bot projects.
   - This member indicated that they are ready to assist anyone needing blockchain expertise.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojor GPU Crashes Need Sync**: A member reported **GPU crashes** with [this Mojo code](https://gist.github.com/sfc-gh-lpaille/6320687631f29619273f56841e3f21c3), leading to cluster health check failures.
   - The suggested solutions include adding **synchronization barriers** and ensuring **GPU P2P** is enabled to avoid assuming two GPUs are present without proper synchronization.
- **Mojo Docs Still Not Fully Operational**: A newcomer expressed frustration with incomplete Mojo documentation, seeking a thorough resource.
   - Community members pointed to actively maintained docs by the Modular team, and suggested reporting specific issues, sharing [Mojo by example](https://ruhati.net/mojo/) and [Mojo Miji](https://mojo-lang.com/) as alternative learning resources.
- **Mojo Memory Alignment Causes Headaches**: A member sought clarification on **memory alignment** in Mojo, especially regarding compiler optimizations and struct padding.
   - It was clarified that specifying alignment prevents padding, some types require larger alignment, and missing alignment can terminate the program; `stack_allocation` was suggested for memory control.
- **Torch-Max Backend Fires Up Models**: The [torch-max-backend v0.2.0 release](https://github.com/gabrieldemarmiesse/torch-max-backend/releases/tag/v0.2.0) now tests **VGG**, **Qwen3**, **GPT2**, and **Densenet** inference.
   - A member expressed surprise that so many models are supported with just a few enabled ops.
- **Peeking at Max's Pipeline Execution**: A member requested a repro script for **TextGenerationPipeline** after seeing *execute* defined in [Max's pipeline.py](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977).
   - The user also asked for the poster's version of **MAX**.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Spotify Podcast Returns After Months!**: A member shared a [Spotify link](https://open.spotify.com/episode/5UMNCXNsMtt8bhXC4aYxhh) to a new podcast episode, which is the **first since December** and was created using **Gems**.
   - The discussion highlighted that this episode marks a return to content creation using **Gems** after a significant hiatus.
- **AI Now Speaks Proto-Germanic**: A member shared a [YouTube link](https://youtu.be/RaRKXoa-CR0?si=mdo_1Ax3oOAGcXuKIdk) showcasing their work training an **AI** to understand and translate **Proto-Germanic**.
   - They reported that the AI has proven to be *"somewhat accurate"* in initial testing, opening possibilities for historical linguistics and AI.
- **Discord Server Needs Moderators**: A member shared [a NotebookLM link](https://notebooklm.google.com/notebook/3b2844c0-1e94-409e-a317-df6ee169a2b3) expressing concern about **spam and unmoderated content** on the Discord server.
   - The urgent call for moderators highlights the increasing need for community management to address spam and ensure a positive environment.
- **NotebookLM Boosts Tabletop Gaming**: A member reported using **NotebookLM** to generate video overviews of transcribed tabletop RPG sessions, creating an automatic *"Last time, on D&D!"* intro.
   - This innovative use of **NotebookLM** helps players remember details before each session, showing how AI can enhance traditional gaming.
- **YouTube Bulk Imports**: Members discussed importing **300 YouTube links** into NotebookLM, suggesting the use of [Chrome extensions](https://chrome.google.com/webstore) for bulk imports and adding YouTube URLs to the Website URL list.
   - One member installed Chrome specifically for a bulk importing extension.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Team Seeks Speed Demons for CI Fixes**: A member is seeking to hire someone to address messy tests and reduce **CI speed** to under **5 minutes**.
   - The appeal highlights concerns about slow, unoptimized tests and questions around the efficiency of **process replay**.
- **Linux (nv) lagging Linux (ptx) in performance**: There are questions around the performance disparity between **Linux (nv)** and **Linux (ptx)**, wondering if the **fast flag** is being used for CUDA compiling.
   - The discussion questions why **process replay** is multiprocessing, and the causes of its slowness.
- **Minimal CUDA Compile Flag Coming?**: A member inquired if **Ubuntu 24.04** supports the minimal CUDA compile flag, linking to a [relevant GitHub pull request](https://github.com/tinygrad/tinygrad/pull/11741).
   - The purpose of the pull request was to discuss further details on how to enable and use minimal CUDA compile flag to speed up compiling.
- **Overworld Constant Folding Causes Controversy**: A member proposed a solution for overworld const folding, suggesting changes to `UPat.cvar` and `UPat.const_like`.
   - George Hotz dismissed the suggestion as *"super ugly"*, advocating for the removal of **base** and inquiring about **PAD**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Caching Still Returns Costs**: A user discovered that **DSPy** still returns the **cost**, even if the result is pulled from the *cache*.
   - This observation was noted as non-intuitive and potentially confusing, highlighting the need for clarity in cost tracking.
- **Cross-Validation Optimizer Inspires Metric Function**: A member sought an **optimizer** to create an LM program mimicking a specific author's style via **cross-validation**.
   - Suggestions included creating a **metric function** with an **AI judge** (e.g., **GPT-5 LLM**) and utilizing **GEPA** for evaluation, opening new workflows for stylistic imitation.
- **Optimized Prompt Extraction Made Easy**: A user inquired about extracting prompts from a **DSPy-optimized program** to get the source.
   - Another user recommended using `optimized.save("my_program.json")` to easily **view the resulting prompt** by saving the program as a JSON file, creating an easy workflow for prompt extraction.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Welcomes New Members**: Aryan and Anay join the Cohere community, with Anay sharing he's pursuing an **MS in Computer Science** in the US and has past experience as an **MLE**.
   - Members were encouraged to share about their company, work, favorite tools and goals to foster collaboration.
- **Another topic**: This is a placeholder summary.
   - This is a second placeholder summary.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Credits Vanish into Thin Air**: A user reported disappearing **Manus credits** and questioned the ability to purchase more, implying repeated issues with the service.
   - This prompted speculation about the stability and reliability of the **Manus platform** for ongoing use.
- **User Bails, Prioritizes Backups**: A user backed up their data before switching back to a previous provider, citing better responsiveness.
   - The move suggests dissatisfaction with current service levels and a renewed emphasis on data security during transitions.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Next LLM Agent Cohort Anticipated**: Members anticipate the next cohort of the **LLM Agents Berkeley MOOC** to start in early September.
   - Specific signup details were not provided, but users expect cohort signups to open very soon.
- **Cohort Signups Expected Imminently**: A user expressed expectation for cohort signups to open very soon, targeting early September.
   - No official dates or announcements were referenced in the discussion.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Functional Python Promises Smarter AI/ML**: A free webinar on August 27th discusses how **functional Python** can speed up workflows, cut costs, and improve efficiency in data-intensive **AI/ML systems**.
   - Attendees will learn about techniques like **persistent memoization** and **deterministic parallelism**, with demos from [DataPhoenix](https://l.dataphoenix.info/8OaKIDl).
- **Webinar Touts Cost Savings with Functional Programming**: The webinar explores reengineering long-running workflows using **functional Python** to unlock cost savings and speed improvements without adding hardware.
   - It covers theory and hands-on demos of open-source tools for streamlining data-intensive processes.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Blockchain Dev Ready to Roll!**: A blockchain dev with hands-on experience across **EVM chains**, **Solana**, **Cardano**, and **Polkadot** is looking to collaborate.
   - They've built **DEXs**, **trading bots**, **smart contracts** for **DApps**, and integrated them with frontends.
- **Blockchain Dev Seeking Collaboration**: A blockchain developer is offering their expertise in **EVM chains**, **Solana**, **Cardano**, and **Polkadot** for potential collaborations.
   - Their experience includes building **DEXs**, **trading bots**, and **smart contracts**, along with integrating them into DApp frontends.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **PelicanVLM-72B-Instruct Joins the BFCL Ranks**: A member submitted a [pull request](https://github.com/ShishirPatil/gorilla/pull/1152) to add tool evaluation for the **PelicanVLM-72B-Instruct** model using the **Berkeley Function Calling Leaderboard (BFCL)** framework.
   - The author seeks community feedback on the integration and included the evaluation scores within the pull request.
- **PelicanVLM-72B-Instruct Evaluation Results**: The pull request includes evaluation scores for the **PelicanVLM-72B-Instruct** model within the **BFCL** framework.
   - Community members are encouraged to review the scores and provide feedback on the model's performance.



---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1407439552611749888)** (1232 messages🔥🔥🔥): 

> `amazon.in not working, comet invites, GPTs Agents, OpenAI's sidebars, GPT Go Plan` 


- **Amazon.in unavailable in Comet Browser**: Users reported [amazon.in](https://amazon.in) not working in **Comet Browser**, but opening on **Brave Browser** or private mode; possibly due to network issue or firewall settings.
   - A member suggested contacting the support team at support@perplexity.ai for assistance.
- **Comet Browser Invite Codes Scramble**: Users are actively seeking **Comet Browser invite codes**, with some mentioning that **Pro users** in the US get instant access, while others elsewhere need an invite or waitlist.
   - Members shared links to the invite channels, emphasizing that each user gets two invites, and warned against paying for invites.
- **ChatGPT Go plan is now available in India**: Users discussed the new [ChatGPT Go plan](https://help.openai.com/en/articles/11989085-what-is-chatgpt-go) being tested in India, priced at **₹399/month**, for prompt writing.
   - Some members suggested it's not worth it compared to the free version or Perplexity Pro, while others saw potential for higher usage or students on a budget.
- **Perplexity UI gets praised and criticized**: Users had mixed opinions on Perplexity's user interface (**UI**), with one calling the Android UI "shi*" and some praising the windows design.
   - However, others loved the UI, especially after built-in adblock was added in the Comet browser.
- **Super Grok suggested to replace GPT Go**: Members debated the merits of GPT Go versus alternatives, with the suggestion of [Super Grok](https://grok.x.ai/) for **₹700/month** as a better option due to a larger context window and lack of limitations.
   - They cited a 131k context limit compared to ChatGPT Go's 32k.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1407622977054314527)** (4 messages): 

> `Shareable Threads, Perplexity AI Newsletter, Sorion Unicode Tool` 


- **Perplexity AI shares Newsletter Link**: Perplexity AI shared a link to their [Weekly AI Technical Newsletter](https://www.perplexity.ai/page/weekly-ai-technical-newsletter-rJrIvTM5TlmeeKPF1VcCKQ).
- **Sorion shares Unicode Tool Link**: Sorion shared a link to their [Unicode Tool](https://sorion.io/unicode/).
- **Reminder to make Threads Shareable**: Perplexity AI reminded users to ensure their threads are set to `Shareable`.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1407627219269980233)** (5 messages): 

> `Perplexity API Status, API Groups Deletion` 


- **API Status Discrepancy causes Queries**: A user inquired about the API status and latency issues, noting the [status page](https://status.perplexity.com/) doesn't reflect the current latency.
   - Another user suggested asking the question in the channel.
- **API Groups deletion request is forwarded**: A user asked if there is a way to delete **API groups**.
   - A member responded that they would forward the question to the **API team**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1407440700072657037)** (1028 messages🔥🔥🔥): 

> `Unsloth App Updated to 3.1, GRPO Applied to Llama Model, VRAM Issues with Qwen3-4b-Instruct, Dataset Tools and Workflows, Blackwell RTX 50 Series and Unsloth Guide` 


- **Unsloth App gets a V3.1 Update**: The Unsloth app was updated to **version 3.1**, which includes an **instruct model** and is reportedly **hybrid**.
   - Members expressed excitement despite some not being able to run it due to hardware limitations.
- **Model Merging with GRPO**: One member applied **GRPO** (Gradient Ratio Policy Optimization) to the **Llama** model using a physics dataset and shared it on [Hugging Face](https://huggingface.co/mohit937/llama31-8b-sft-grpo-physics-reasoning-fp16).
   - Users discussed the effectiveness and practical implications of GRPO, comparing it to DPO and noting its reinforcement learning aspects, even without an external judge model.
- **Context Length Troubles on T4**: A user encountered shared memory issues while training **Qwen3-4b-instruct** on a **T4 GPU**, even with reduced batch size and gradient accumulation, and even after reducing the context size.
   - They were advised to ensure context length isn't too large, as it significantly impacts **VRAM** usage, also mentioning the **original notebook was throwing errors as well**.
- **Dataset Handling with GUIs?**: Members discussed alternatives to **pandas** for combining and filtering datasets, with one user requesting a GUI for ease of use.
   - A member suggested **Marimo notebooks** ([marimo.io](https://marimo.io/)) as a solution, providing a link to a sample notebook ([gist.github.com](https://gist.github.com/boatbomber/11fd0c49a502ba2804f447a91fcdf931)) that merges the datasets, and creates live updating loss charts.
- **The Lowdown on GRPO**: Community members unpacked how **GRPO** works, and the consensus is that it is *basically RL just without an external judge model*.
   - Although **vanilla GRPO** has known triple footguns, *such as a loss function favoring longer responses*, some believe it can be as good as **Reinforce++**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1407505252588064890)** (1 messages): 

> `Discord showing gaming handle, Privacy Concerns` 


- **Discord displays gaming handle**: A user was surprised that Discord displays their gaming handle from **Battlefield** across all channels.
   - They posted *'= : )'* indicating their amusement, or bemusement, with the realization.
- **Privacy concerns arise**: The user's comment raises potential **privacy concerns** about linked gaming accounts revealing handles across different Discord servers.
   - This highlights the importance of understanding how connected accounts can expose personal information on various platforms.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1407488808609382590)** (29 messages🔥): 

> `ASUS ROG Matrix GeForce RTX 5090 30th Anniversary Limited Edition, CUDA out of memory` 


- ****ASUS ROG RTX 5090** Limited Edition Runs Hot!**: A member shared an image of the **ASUS ROG Matrix GeForce RTX 5090 30th Anniversary Limited Edition**, noting its **800W** power consumption and preference for a black and white color scheme.
   - The card's aesthetics were described as *an acquired taste*, with the user humorously lamenting sending Gemini a million tokens to generate the image.
- ****CUDA OOM** Error strikes Again!**: A user encountered a **CUDA out of memory** error while attempting to allocate **20MB** of memory, despite having **23.79GB** free on the GPU.
   - They asked if restarting was the only solution, and another member suggested trying to [restart the driver](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) first to clear any leftover junk from dead processes.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1407442040442196179)** (77 messages🔥🔥): 

> `Gemma 3 270m CPT, Runpod setup, GPU requirements, DeepSeek V3.1 Quantization, Transformers version issue` 


- ****Gemma 3** Fine-Tuning Newbie Asks for Help**: A newbie in fine-tuning seeks advice on performing Continued Pre-Training (**CPT**) with **Gemma 3 270m** using raw text to add more knowledge in French from a Gutenberg Project book, specifically *20,000 lieues sous les mers* by Jules Verne.
   - Links to Unsloth's documentation and blog ([docs.unsloth.ai/basics/continued-pretraining](https://docs.unsloth.ai/basics/continued-pretraining), [unsloth.ai/blog/contpretraining](https://unsloth.ai/blog/contpretraining)) were shared as starting points.
- ****Mistral 3.2** VRAM Requirements Increase for Training**: A user reports that the vision components tacked onto **Mistral 3.2** seem to have increased the VRAM requirements for training.
   - They ask if there's a way to completely strip out the vision layers to fit their old context length. No solution was offered, but others [had similar issues](https://discord.com/channels/1179035537009545276/1407500710756749426).
- **Debate Swirls Around **DeepSeek V3.1** Quantization and Execution**: A user inquired about quantizing the **DeepSeek V3.1 671B** model to **Q4_K_XL** and **Q3_K_XL** for local use with *llama.cpp*, asking about minimum VRAM requirements on a system with **48 GB** of VRAM.
   - Another user retorted that the user was asking both to quantize *and* run the model locally and linked to a previous quantization unsloth did for the new versions ([https://unsloth.ai/blog/deepseekr1-dynamic](https://unsloth.ai/blog/deepseekr1-dynamic)).
- **Transformers Update Triggers **72B VL** Loading Errors on Blackwell GPUs**: Users identified that Transformers versions **v4.54.0** and later introduce a sharded streaming load with immediate 4-bit quantization (`create_quantized_param`) that causes errors when loading `unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit` on Blackwell GPUs.
   - The workaround involves downgrading Transformers to **v4.51.3**, and an issue has been opened in the [transformers repo](https://github.com/huggingface/transformers).
- **vLLM and **GPT-OSS** Incompatibility**: A user reported getting an error when trying to deploy the [unsloth/gpt-oss-20b-GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF?show_file_info=gpt-oss-20b-UD-Q8_K_XL.gguf) model using vLLM on an Nvidia L40S GPU, reporting an [error message](https://cdn.discordapp.com/attachments/1179777624986357780/1407711803462520993/stack_trace.txt?ex=68a71947&is=68a5c7c7&hm=12c6220b8a18de09f7858f2e01a0d572536102ee1e4ac1a085c1c54ed46eacd1&).
   - Another user clarified that **vLLM doesn't work with gptoss ggufs** which is the reason it's failing.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1407448101735760014)** (18 messages🔥): 

> `Gemma 3 4b finetune, Un-sycophantic BERT models, Swahili Gemma 1B, OpenHelix dataset` 


- **Electroglyph drops Gemma 3 4b finetune**: A member shared their **Gemma 3 4b unslop finetune**, along with training code, and uploaded **UD-Q4_K_XL GGUF** to [Hugging Face](https://huggingface.co/electroglyph/gemma-3-4b-it-unslop-GRPO-v3).
   - The user noted that they believe *it's pretty good this time around*.
- **Punishing Sycophancy with BERT**: A member suggested training a **sycophantic classifier** (modern BERT) to punish responses during validation.
   - This would eliminate the need to **hardcode** rules for suppressing sycophantic responses, potentially improving model behavior.
- **Swahili Gemma 1B fine-tuned with Unsloth**: The team fine-tuned **Gemma 3 1B** for **Swahili conversational AI** and **translation** using Unsloth.
   - A link was shared to [CraneAILabs/swahili-gemma-1b](https://huggingface.co/CraneAILabs/swahili-gemma-1b).
- **New OpenHelix Dataset Release**: A new, smaller, more diverse, and balanced version of the **OpenHelix** dataset (**OpenHelix-R-86k-v2**) was released on [Hugging Face](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-86k-v2).
   - The dataset underwent a new deduplication step, filtering out samples with **high ngram similarity**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1407446292027674854)** (17 messages🔥): 

> `L40S vs A100 Inference, GRPO for Llama, Qwen3-4B Finetuning with Unsloth` 


- **L40S Lower T/S Than A100 Despite Higher FLOPS!?!**: A member ran inference for **Llama 3.2 3B** on **L40S** vs **A100** and found the A100 had around **30%** higher tokens/second (t/s) despite the L40S having higher FLOPS (**362** vs **312**).
   - Another member commented that *memory bandwidth* is the *biggest factor* in this difference.
- **GRPO Applied to Llama Model for Physics Dataset**: A member applied **GRPO** to the **Llama model** for a physics dataset using Unsloth and shared a [link to the model](https://huggingface.co/mohit937/llama31-8b-sft-grpo-physics-reasoning-fp16).
   - It was clarified that some of the data is by **R1 70B** (the distilled version), not R1.
- **Dataset Contamination Tests Reveal Minimal Overlap**: A member conducted contamination tests on their dataset and reported good news: it's not contaminated, with a max overlap of **14.8%** and mean text similarity of **18.2%**.
   - They looked at high overlap samples but the overlaps were just: *"what is the sum of"*, *"how many days will it take"*, *"what is the difference between"*, *"how long will it take"*, *"what is the perimeter of"*.
- **Unsloth Used to Finetune Qwen3-4B**: A member is using Unsloth to finetune on **Qwen3-4B** and shared a [link](https://x.com/anuragphadke/status/1958278747548995999) and asked to be wished luck.
   - Another member asked if the code, dataset and machine specs were published.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1407439778454048818)** (1025 messages🔥🔥🔥): 

> `Nano Banana Launch, Gemini 2.5 Pro vs GPT-5, DeepSeek v3.1 Issues, Image upload issues on LMArena` 


- **Nano Banana's Hype Gets Google Scrutiny**: Members eagerly anticipated the release of **Nano Banana**, with [one member sharing a link](https://x.com/OfficialLoganK/status/1957908528925909391) seemingly confirming the launch, leading to speculation that it would be a **Google** model.
   - Concerns arose about limiting its use to **Pixel phones**, but others believed it would be more widely available due to the hype and demand generated on **LMArena**. 
- **Gemini 2.5 Pro Dethrones GPT-5**: **Gemini 2.5 Pro** regained the top spot on the leaderboard, leading to discussions about **GPT-5's** perceived decline.
   - Some suggested deliberate downvoting or over-agreeableness as possible reasons, while others pointed to **Gemini's** faster speed and free usage as factors, with [one member noting the disparity in scores](https://polymarket.com/event/which-company-has-best-ai-model-end-of-august?tid=1755631739582) between Gemini and competitors.
- **DeepSeek's Deep Dive Downwards?**: Doubts emerged about the quality of **DeepSeek v3.1**, with some citing a lack of updates and the use of **Huawei GPUs** as potential issues.
   - Others defended **DeepSeek**, attributing any perceived shortcomings to **Trump's tariffs** affecting GPU availability, though several users claimed the quality was *slop coded* and worse than the earlier models.
- **Image Upload Glitches Plague LMArena**: Users reported persistent errors when uploading images to **LMArena**, with the error message *"Something went wrong while generating the response"* cropping up.
   - Members speculated the problem might be file type issues or a broader issue under investigation by the development team, noting it only happened when copy/pasting images rather than uploading saved ones, so [the problem was reported to 🍍](link.to.user).


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1407439881931718726)** (2 messages): 

> `Qwen-Image-Edit, Image Edit Leaderboard, LMArena` 


- **LMArena Gets New Image Edit Model!**: A new model, **Qwen-Image-Edit**, has been added to Image Edit on LMArena.
   - This addition expands the capabilities for image manipulation and editing within the LMArena platform.
- **Qwen-Image-Edit Dethrones All!**: The [Image Edit Leaderboard](https://lmarena.ai/leaderboard/image-edit) has been updated, and **Qwen-Image-Edit** is now the #1 open model for Image Edit!
   - Users can explore the [leaderboards here](https://lmarena.ai/leaderboard) to see how different models compare in image editing tasks.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1407444123216121917)** (859 messages🔥🔥🔥): 

> `Sonic model, Token usage, Multi-agent setup, Code Quality` 


- **Sonic Model Sparks Curiosity and Speculation**: The new **Sonic model** is generating buzz, with users noting its **speed** and debating its origins, with some speculating it could be [Grok code](https://x.ai), but others lean toward [Gemini](https://gemini.google.com).
   - One user described it as *goofy reactions on the announcement* while another mentioned it has *nice to do lists*.
- **Users Experience Token Spikes with Claude, Seek Alternatives**: Some users are experiencing unexpected **token spikes** with **Claude**, leading them to stick with **Auto mode** due to its more reasonable token usage.
   - One user mentioned needing *multiple clarifications with auto usage* but still preferring it over manually selecting Claude.
- **Debate over Training on User Data and Stealth Models**: There's discussion around whether the **stealth Sonic model** is training on user data, with one user pointing to Cline's message suggesting **data training**.
   - Concerns are raised about potentially leaking personal info and the need for Cursor to confirm their privacy policy.
- **Sonic's Coding Performance Assessed; Mixed Reviews Emerge**: First impressions of **Sonic's coding abilities** vary, with some finding it excellent for quick tasks, and doing *coding stuff about as well as GPT-5 has been but, like, 2x faster*, while others say it's *kinda bad*. 
   - Some users noted that **Sonic is good**, but it consumes a lot of tokens; since it’s token-based, it could cost a big bill.
- **Users Explore Multi-Agent Setups**: Some users are experimenting with **multi-agent setups** in Cursor, such as a **developer** and an **annoying code reviewer** AI in constant battle.
   - There are discussions around how to integrate terminal histories and short-term memories in multi-agent setups, with one user sharing a link to their [multi-agent developer/reviewer mode proposal](https://forum.cursor.com/t/multi-agent-developer-reviewer-mode-in-cli/1312).


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1407479123181441177)** (14 messages🔥): 

> `API Key Authorization Issues, System Timeout Issues, NPM Package Pull Failure, Background Agent Functionality Issues, Git Configuration Problems` 


- **API Key Gets 403 on Background Agents**: A user reported receiving a **403 not authorized error** when trying to access background agents via an API key for Cobot.co, questioning if others have experienced similar issues.
- **System Plagued by Timeouts and Service Unavailable Errors**: Multiple users are reporting significant **system timeouts** across various operations, with one user detailing issues like **Git merge timeouts**, **file operation timeouts**, and **503 service unavailable errors**.
   - The user experiencing these issues cannot execute basic commands and suggests the environment needs intervention at a higher level, such as a VM or service restart.
- **NPM Package Pull Fails Despite Valid SSH Key**: A user is facing issues where background agents fail to pull a **private NPM package via Git** in an `npm install`, despite configuring a valid SSH key and GitHub PAT.
   - The user has tried various methods, including providing a GitHub PAT and setting `GITHUB_TOKEN` and `GH_TOKEN`, but `npm` still cannot find the private repository, resulting in an `npm error code 128`.
- **Recurring Tool System Breakage Hinders Agent Functionality**: A user pointed out that background agent functionality is hindered by a regularly breaking tool system, with a suggestion to request the agent to commit and push the new branch to origin as a potential fix.
- **Git config throwing everything off**: A user mentioned that they disabled their install script to poke about and found that `~/.gitconfig` is throwing everything off and are trying `repositoryDependencies` to add the extra repos, imaging that configures the access token used in the `.gitconfig`.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1407577198474297367)** (1 messages): 

> `New stealth model in Cursor, Partnered Model` 


- **Cursor drops **stealth partner model** for free trials**: Cursor announced a new **stealth model** from one of their partners, available for free trials.
- **Users Asked to Give Feedback on New Model**: The announcement prompted Cursor to ask for user feedback.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1407440964200693831)** (305 messages🔥🔥): 

> `Qwen3 BF16, GPT-OSS 20B, CUDA setup, Nvidia vs. AMD, AgentMode` 


- **Qwen3 BF16 model is super impressive**: The **Qwen3 BF16** model is noted to be crazy compared to the quants, with members urging to try the **Q8** vs **BF16** version of the model for a significant difference.
   - A member claimed that a **4B BF16** model is more capable of creating *higher quality* code zero-shot than an **8B Q4/Q8** model and that llama.cpp introduced matrix multiplication kernels involving fp16 for prompt eval.
- **GPT-OSS 20B can clone**: A member is cloning himself using **GPT-OSS 20B** for basic usage and to summarize convos, noting that **Deepseek R1 distilled** has been best at replicating speech patterns.
   - Another added that you can use it as a websearching buddy with duckduckgo.
- **CUDA setup isn't that simple**: To utilize vram correctly with llama.cpp, members suggest setting **context**, **ngl** at max, and adjusting **-n-cpu-moe** until it fits on the vram with a bit of margin.
   - They shared that you must combine the 2 zip files into one folder as well.
- **Nvidia beats AMD**: Some users believe that Nvidia has an insane engineering leap on AMD and Intel, and that the reason Intel's dGPUs suck is because it's probably not that easy to scale up.
   - One user hopes Intel folds so Nvidia can take up x86/64 chip production out of "anti-monopoly" stuff at some point.
- **ChatGPT AgentMode is a Gimmick**: ChatGPT's **AgentMode** provides a session-based virtual Linux PC with CPUs, memory, and storage, enabling it to compile, test run applications, and download libraries, but it is a *gimmick partytrick* due to being capped at **40 agentmode requests per month**.
   - Members explored setting up a similar OpenInterpreter, recommending it run on a virtual machine because *an LLM with access to your own computer can wreak havoc in no time*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1407471319733501972)** (78 messages🔥🔥): 

> `Bolt Graphics AI roadmap, 3090 vs alternatives, AMD mi50 32gb, Qwen3-30ba3b on mi50, 1M context Q3-30B` 


- **Bolt Graphics puts AI on Ice**: Setting up devices not meant for AI is a hassle, **Bolt Graphics** has no AI roadmap, so expect odd bugs, OOMs, and memory leaks.
   - It's not ideal for practical use and more of an enthusiast device, which can be fixed by enabling **fmac** with a simple code, linked to an [eBay listing](https://ebay.us/m/tl18ng).
- **3090 trumps other GPUs**: A user remarked that the linked [eBay listing](https://ebay.us/m/tl18ng) isn't worth it, stating *you can get a 3090 for similar money*, especially if it had 32GBs of memory for longer OFL support.
   - Another user deemed it basically **expensive e-waste**.
- **AMD mi50 emerges as cost-effective alternative**: A user suggests that an **AMD mi50 32GB** is a better option if you can cool it, citing they are *dirt cheap from China now*.
   - They reported getting **50 tokens/s** on a fresh prompt with **Qwen3-30B-A3B** on the **mi50**.
- **DDR3 Xeon CPU runs blazing fast!**: With **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** on an old **Xeon** with **DDR3**, the user got **19 tok/sec** with MoE on CPU (using `HIP_VISIBLE_DEVICES=0 llama-server -hf lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF --gpu-layers 999 --host 0.0.0.0 --ctx-size 131072 --flash-attn --cpu-moe`).
   - Switching to the **Q8** model on GPU with context on CPU yielded impressive performance, especially after disabling ondemand and low power mode on **384 gb ram**.
- **1M Context Window explodes the Kernel**: A user tried a **Q3-30B** model with a **1M context window** on an **Intel(R) Xeon(R) CPU E5-2680 v2** with 24x Size: 16 GB Speed: 1600 MT/s (Configured Memory Speed: 1333 MT/s).
   - However, it failed due to an inability to allocate kv memory as *1M context needs 98304.00 Mib*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1407742202662228001)** (1 messages): 

> `Activity Analytics API, Allowed Models API, OpenRouter Developer APIs` 


- ****OpenRouter** releases Activity Analytics API**: **OpenRouter** announced the release of a new **Activity Analytics API** that allows users to programmatically retrieve their daily activity rollups, as well as those for their organizations, documented [here](https://openrouter.ai/docs/api-reference/analytics/get-activity).
- ****OpenRouter** lists allowed Models API**: **OpenRouter** has released an **Allowed Models API**, enabling users and organizations to programmatically fetch the models they are permitted to access based on their provider restrictions, documented [here](https://openrouter.ai/docs/api-reference/list-models-filtered-by-user-provider-preferences).


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1407450912406114399)** (211 messages🔥🔥): 

> `OpenWebUI Memory Feature, GPT-5 Context Issues, Deepseek v3.1 Availability on OpenRouter, Stealth Model Speculation (Grok-4 Code), Free Model Options on OpenRouter` 


- **Decoding OpenWebUI's Memory Lane**: In OpenWebUI, the memory feature defaults to **manual input**, but addons exist to automatically save relevant conversation snippets, [enhancing recall](https://github.com/open-webui/open-webui).
   - Relevant memories are injected into the system prompt, influencing the model's responses by providing **contextually appropriate information**.
- **GPT-5's Context Crisis: Token Tumbles**: Users report issues with `openai/gpt-5`'s 400k context window, with calls failing at ~66k tokens, yielding a silent `200 OK` response and **0 tokens**.
   - When trying `gpt-5-chat` with Cline on Cursor, users also experienced an *exceeded context window error* at **<100k tokens**.
- **Deepseek v3.1 Delay: OpenRouter's Holdout**: Users are questioning why Deepseek v3.1 is not yet available on OpenRouter, even as other providers like Chutes offer it, though the latter has the **base version**.
   - OpenRouter is reportedly awaiting an official announcement before launching **Deepseek v3.1**, suggesting a provider-dependent release strategy.
- **Stealth Model Speculation: Grok-4 Code Emerges**: Speculation points to Grok-4 Code as the stealth model used by Cline and Cursor, with one member giving it a **90% chance** of being accurate.
   - A member hinted that Deepseek instruct was another possible contender.
- **Unearthing Free Model Trove on OpenRouter**: OpenRouter offers a range of free models, including **Llama 3.3-70B-Instruct:free**, accessible via the [Together AI models page](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free).
   - Users are advised to navigate the [Together AI models page](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free) and search for models labeled as "free".


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1407806939878129774)** (12 messages🔥): 

> `LLMs Formatting Output, AFR Chanticleer AI Report, Google Gemini Models, OpenAI-standard complex content format, tool calling flows` 


- **LLMs Struggle with Output Formatting**: Users reported having bad experiences using LLMs like **Qwen3 coder 480b** and **DeepSeek v3 0324** due to their inability to follow formatting instructions properly.
   - The outputs often had bugs, failed to display, and frequently ignored the initial prompt, instead creating unrelated content like a *tic tac toe site*.
- **AI Report Causes Market Nervousness**: An [AI report](https://archive.md/IlP7F) suggests that *95% of organizations are getting zero return* from their generative AI deployments, particularly those using customized AI models.
   - The report indicates that companies aren't spending enough time ensuring their customized AI models keep learning, while a **shadow AI economy** has developed where employees rely on general AI models like **ChatGPT** and **Gemini**.
- **Google Gemini Models Return 400 Error**: **Google Gemini models** return an **HTTP 400 error** when assistant messages with tool calls use the **OpenAI-standard complex content format** `[{"type": "text", "text": "..."}]` instead of a simple string format.
   - This issue affects all **google/gemini-*** models but does not impact **openai/*** or **anthropic/*** models, and only occurs when tool calls and tool results are present in the message chain.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1407441309635182764)** (148 messages🔥🔥): 

> `Gemini Storybook mode, AI Bots paying each other, Decentralized AI BOINC-style project, AI Moderation` 


- **Gemini's Storybook Mode Impresses**: Users shared screenshots of **Gemini's Storybook mode**, with one noting its success in generating [anime art styles](https://cdn.discordapp.com/attachments/998381918976479273/1407778466312491100/Screenshot_20250821_032749_Chrome.jpg?ex=68a7575d&is=68a605dd&hm=82829a2dec98ac04ae1fa112ccc1bf4bde89ef108ecf53bfeb8c41c6b837a944&).
   - Another user commented that the [Tintin art style](https://cdn.discordapp.com/attachments/998381918976479273/1407783255222255656/Screenshot_20250821_034732_Chrome.jpg?ex=68a75bd3&is=68a60a53&hm=bedd90170aee9797f9ed6788c2acc59f7f888465cbd647da39eb19cdf217&) wasn't as successful.
- **AI Bots Paying Each Other: Risky Business?**: A user inquired about automating payments between AI bots, and [another user](https://twitter.com/huxaifa) outlined the key challenges including identity, smart contract logic, payment infrastructure, autonomy, and legal/ethical considerations.
   - Another user expressed concerns about the safety of AI handling monetary transactions and suggested that **AI could suggest payments, but humans should approve them**.
- **Decentralized AI BOINC Project: Still a Dream?**: A user questioned why a decentralized, **BOINC-style AI project** hasn't been built yet, to which another responded that a lack of committed contributors might be the issue.
   - They mentioned the failed **Petals network** and the problem of ensuring that all nodes are updated to the latest models.
- **AI Moderation: Can AI Moderate Itself?**: In a discussion about verifying the quality of contributions in distributed AI inference, a user suggested training an **AI to moderate the network**.
   - Others pointed out the issues, suggesting that double inference or blind spot tests might be solutions, but also highlighted the increased cost and complexity.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1407446960394211370)** (15 messages🔥): 

> `GPT Custom Actions, Vanished GPT, GPT5 Conversations, AI Agents and Workflows, AGI Arms Race` 


- ****Standard Voice Mode** calls GPT Custom Actions**: A member noted that *standard voice mode could call **GPT custom actions***, implying a limitation in advanced mode.
   - They quipped that *if advanced mode cannot, then it isn't really advanced*.
- **User reports Vanished Customized GPT**: A user reported that a **customized GPT** they had invested effort into suddenly vanished and they were seeking insights into potential causes, such as **subscription/billing issues**.
   - No direct answers were provided in the immediate discussion.
- **Moderators Nuke Mr. Beast Crypto Scam**: A member reported a **phishing scam** involving a fake news article image claiming *"Mr Beast made a website called `cryptoscams_dot_com` where you can make 2 billion dollars in 5 minutes"*, noting its clear falseness.
   - They lauded the moderators' swift deletion of the deceptive post.
- **GPT5 confesses limitation on short term memory**: **GPT5** seemingly *confessed: Short term memory is for token optimization. Session closed, context and memory wiped like ice cube leaks over the tab counter*, implying a limitation compared to GPT4's context retention.
   - It can use up to **196K tokens** per session.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1407779940786503703)** (4 messages): 

> `SCRIBE Prompt, Audio Steganography in Prompts, Model Interpretation of Prompts, Prompt Deconstruction, Impact of Language on Model Output` 


- **Sophisticated 'SCRIBE' Prompt Surfaces**: A member shared a sophisticated prompt codenamed **'SCRIBE'**, designed to mimic human writing style and employing techniques like **audio steganography** and **hash modification**.
   - The member questioned whether the model truly understands such commands or if they are merely *'fluff for show'*, prompting discussion on prompt deconstruction.
- **Model Scoops Training Data Trawl, Affecting Output**: A member posited that every element of the input, including **punctuation**, **spelling**, and **grammar**, affects the model's output, suggesting the model trawls its training data for related information.
   - The model *'scoops up a trawl of training data that may relate to our input and the rest of its training'* and *'puts this catch into containers and ships it to us as an output.'*
- **Directness vs. Drift in Prompts**: Vague prompts are said to force the model to guess and are more prone to **drift** compared to direct prompts, which explicitly guide the model, especially after model updates.
   - According to a member, *'Vague prompts that force the model to guess are more likely than direct prompts that exactly direct the model to suffer from drift as they keep changing the model'*. 
- **Testing Prompt Meaning with Model Evaluation**: A method for testing a prompt's meaning to the model involves asking it to **evaluate** (not follow) the prompt and explain its interpretation, identifying ambiguous or conflicting elements.
   - The member shared a [ChatGPT link](https://chatgpt.com/share/68a60b8f-d238-8011-903c-63b9936d481f) as an example of this evaluation process, noting that **personalization settings** affect how the model interprets the prompt.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1407779940786503703)** (4 messages): 

> `SCRIBE prompt analysis, Model understanding of complex prompts, Impact of language on model output, Prompt evaluation techniques` 


- ****SCRIBE** Prompt Under Scrutiny**: A member shared a *sophisticated prompt* codenamed **SCRIBE**, noting its use of *strange techniques* like audio steganography and hash modification in its **AvdAIDtct** section, questioning if the model actually understands these commands or if it's just *fluff*.
   - The prompt.txt file is available [here](https://cdn.discordapp.com/attachments/1046317269069864970/1407779940790833232/promt.txt?ex=68a758bd&is=68a6073d&hm=270b8817fa88d77ca1d2ccdec8f2fe0ae89ae9c9cb592ef47ddc7f6312d80302&).
- **Language is Key to Model Output**: A member posited that every word, punctuation mark, and grammatical structure affects the model's output, suggesting that the model **scoops up** relevant training data based on the input patterns and ships it back.
   - *The language we use in our input, hugely affects where in the training data the model goes, where similarly grouped training data lives.*
- **Vague Prompts Drift Off-Course**: It was noted that **vague prompts** are more likely to *suffer from drift* as updates to the model can drastically change how it interprets and responds to those prompts.
   - This is in contrast to **direct prompts** that precisely instruct the model.
- **Evaluating Prompts: A Proactive Approach**: A method for testing a prompt's effectiveness involves asking the model to *evaluate, do not follow, the following [prompt]*, and then to *explain what it means* and identify any ambiguities or conflicts.
   - An example of this prompt evaluation is available [here](https://chatgpt.com/share/68a60b8f-d238-8011-903c-63b9936d481f).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1407442085426106520)** (84 messages🔥🔥): 

> `Hackathon Invites, CUDA Kernel Optimization, Alienware R15, Sesh Bot Discord Calendar Sync` 


- **Hackathon Waitlist Woes**: Many members expressed interest in attending the hackathon, but were waitlisted and hoping for an invite.
   - One member, a first-year PhD student at Rowan University working on **CUDA kernels**, also hoped for an invite.
- **ChatGPT Champions `__restrict__`**: A member found that ChatGPT suggests adding `__restrict__` to all CUDA kernel array parameters for potential efficiency gains by indicating no aliasing.
   - Another member added *It was very important on some ancient GPUs* but now not as big of an advantage.
- **Raves for Alienware**: One member is happy with their Alienware desktop, another member got an **Alienware R15** on sale for **$1900**, later upgrading to **128GB** memory and **4TB** disk.
   - Another member asked if they should wait until next year to purchase one.
- **Discord Events in Google Calendar**: Members discussed how to automatically sync GPU MODE Discord events to Google Calendar, eventually exploring the "sesh" bot.
   - The verdict was that it's *a little annoying*, as you have to `/link` from within Discord to get Google Calendar sync working.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1407691996100952140)** (2 messages): 

> `cudaMemcpyAsync with late-bound addresses, NCU profiling issues with live register data, Kernel register pressure analysis` 


- **CudaMemcpyAsync: Late-Binding Addresses?**: A user is facing an issue where `cudaMemcpyAsync` needs to use an address that is only available after `cudaLaunchHostFunc` completes, but both functions are asynchronous, causing `cudaMemcpyAsync` to use the address value at call time.
   - The user is seeking a way to ensure `cudaMemcpyAsync` uses the correct `plan->syncCondition->recvbuff` value at execution time without moving the function inside the host function to avoid a deadlock.
- **NCU Fails to Display Live Register Data**: A user reports that **NVIDIA Compute profiler (NCU)** isn't displaying live register data during kernel profiling, while other metrics are showing up normally.
   - The user provided a screenshot of the **NCU** interface, and the issue occurs across multiple kernels.
- **Kernel's Register Pressure Bottleneck**: A user wants advice on identifying bottlenecks after introducing high register pressure when modifying code from a baseline kernel.
   - The user seeks solutions to pinpoint the source of increased register usage in their kernel.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1407537330902007930)** (6 messages): 

> `SemiAnalysis job posting, New Grad Engineer, Performance engineering, CI/CD pipelines, LinkedIn tracking links` 


- **SemiAnalysis Seeks New Grad Engineer**: [SemiAnalysis](https://www.semianalysis.com/) is looking for a new grad engineer to join their engineering team, offering a unique chance to work on high-visibility special projects focusing on **performance engineering**, **system reliability**, and the intersection of **hardware and software**.
- **Privacy Advocates Prefer Direct Job Links**: One user suggested using the direct link to the job application ([https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1/?rs=76643084](https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1/?rs=76643084)) instead of the **LinkedIn shortened link** for privacy reasons.
- **EleutherAI Alum applies to SemiAnalysis**: An **EleutherAI** alum is applying and mentioned they were *looking forward to **Gaudi 2** and hoping it would be a more programmable competitor to **TPU***.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1407523119249494097)** (6 messages): 

> `CUDA setup on Ubuntu, AI companies and databases like ClickHouse, Embedding speeds of infinity server vs sglang` 


- **Newbie Seeks CUDA Setup Tutorial**: A member requested a complete guideline or setup tutorial on how to run **CUDA C++** in **Ubuntu** for deep learning in Python, and another member shared a [YouTube video](https://youtu.be/LiurVXkSUDU?si=v7SrmV4oRM5EImyV) to help.
   - The video is expected to cover all the necessary steps for configuring CUDA C++ on an Ubuntu system, which is useful for accelerating deep learning tasks.
- **ClickHouse's Role in AI Workflows**: A member asked why AI companies like OpenAI use databases like **ClickHouse**, hypothesizing it's mainly for **data preparation**—running SQL queries to transform raw data into relational formats for feature stores or model training.
   - Another member suggested **product usage tracking**, logging, and other product-related analytics as potential use cases, noting that AI companies do more than just training models.
- **Deep Dive on Database Use-Cases**: A member inquired about the [use cases](https://clickhouse.com/use-cases/machine-learning-and-data-science) for databases/lakehouses in AI workflows, particularly how they connect to **ClickHouse**, with a focus on how it benefits AI companies.
   - The discussion aimed to differentiate between typical data warehouse analytics and specific applications within the AI industry, exploring whether the link to AI is primarily through data preparation for feature engineering.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

stoicsmm: hi everyone
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/)** (1 messages): 

topsy1581: Hi, does the TorchAO support grouped gemm for MXFP8 x MXFP8, or MXFP4 x MXFP4?
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1407706839021256745)** (4 messages): 

> `Gouda content, Mods are asleep` 


- **Post Gouda Content**: A user jokingly suggested posting "gouda content" because *mods are asleep*, attaching a [picture of Gouda cheese](https://cdn.discordapp.com/attachments/1215328286503075953/1407745670265176194/Gouda.png?ex=68a738d2&is=68a5e752&hm=ce76874cdc8db738d1f4b669afc4060dac99aa29df856042395eb0c4d5cfd83d&).
- **Mods Never Sleep**: A user responded to the joke saying that *mods never sleep* and that *mods are in all timezones*.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

veer6174: Anyone here in Bangkok?
  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1407585429326397552)** (3 messages): 

> `Triton-Puzzles issues, torch 2.5.0 installation, numpy downgrade` 


- **Triton-Puzzles reliability declines**: Members reported that **Triton-Puzzles** was working and has been breaking a lot recently.
   - *"Is it breaking for others too?"* a member asked after it started breaking.
- **Torch installation solves problems then creates new problems**: A member fixed **Triton-Puzzles-Lite** by installing **torch==2.5.0**.
   - This created another problem where they needed to downgrade **numpy<2.0**, but then that fixed everything.
- **Version meddling breaks test cases**: A user said the current notebook doesn't run without edits and tests don't run.
   - With version meddling, it runs but breaks for test cases that are obviously correct such as Puzzle 1.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

hariprasathvinayagam: try it
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

veer6174: Anyone setup emacs to edit google collab?
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1407545184073289729)** (1 messages): 

> `SkyRL, ReasoningGym Integration` 


- **SkyRL Project integrates ReasoningGym**: Tyler, co-lead of the **SkyRL** project, announced they are actively working on integrating **ReasoningGym** on top of SkyRL and provided a [draft PR](https://github.com/NovaSky-AI/SkyRL/pull/160).
   - They mentioned that once it's ready, they would like to contribute this example integration to the **ReasoningGym** repo.
- **SkyRL Integration Contribution**: The **SkyRL** team plans to contribute their **ReasoningGym** integration example to the ReasoningGym repository upon completion.
   - This contribution aims to provide a practical demonstration of how ReasoningGym can be used with SkyRL, benefiting the community.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1407647141744214016)** (2 messages): 

> `A100 Leaderboard, MI300 Leaderboard` 


- **A100 trimul Leaderboard Runner-Up**: A member achieved **second place** on the **A100 trimul** leaderboard with a time of **7.83 ms**.
- **MI300 Crushes trimul**: A member successfully ran on **MI300**, clocking in at **3.50 ms** on the **trimul** leaderboard.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1407526149848039557)** (8 messages🔥): 

> `Factorio Mods, FLE, Registry.py, Friday Meeting` 


- ****Legacy Mods Leave FLE High and Dry****: A member inquired about the `stdlib_1.4.6` and `headless-player_0.1.0` **Factorio mods** referenced in the code, noting their absence from `fle/cluster/docker/mods/mod-list.json`.
   - Another member clarified that the mods are of *legacy usage*, **FLE** doesn’t depend on either of them any longer, any references to them are stale and are supposed to be removed.
- ****Task Key and Environment ID Twins in Registry.py****: A member noticed that the **task key** and **environment ID** appear to be the same in `registry.py`, and attached a screenshot from the year 2025 as supporting evidence ([Screenshot](https://cdn.discordapp.com/attachments/1354169122107293786/1407740402819272915/Screenshot_2025-08-20_at_17.57.29.png?ex=68a733ea&is=68a5e26a&hm=5b3417f2e0aa67a736de0cfd2d197726d3f27fbbd4a39b92b08925f8c328a155&)).
   - Another member confirmed this observation and suggested it might be related to ongoing efforts to remove bugs and stale code in [issue #309](https://github.com/issues/309).
- ****Moon Mod Makes Landfall on Factorio****: One member shared a custom mod named [Cerys-Moon-of-Fulgor](https://mods.factorio.com/mod/Cerys-Moon-of-Fulgor) made this year.
   - Other members gave appreciation and said *cool thanks*.
- ****Jet-Lag Jams Friday Factorio Fun****: A member mentioned they will be around for the **Friday meeting** before flying back to the UK.
   - They also added that the end of their program is coming, so their sleep schedule is all over the place.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1407557437623504907)** (1 messages): 

> `Arithmetic types, TensorSSA objects, cute.full_like, wrapping logic` 


- **Automagic Arithmetic-to-TensorSSA Conversion?**: A member inquired about plans to automatically convert arithmetic types into `full_like` **TensorSSA objects**.
   - The user expressed inconvenience at sprinkling `cute.full_like(tmp2, float('-inf'))` throughout the code and noted the **wrapping logic** is brittle.
- **Handling Arithmetic Types with TensorSSA**: Discussion revolved around automatically converting arithmetic types into `full_like` **TensorSSA objects** to streamline code.
   - The main concern was the verbosity of manually wrapping arithmetic types using `cute.full_like`, which some considered brittle.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/)** (1 messages): 

j4orz: updates to the book. prelims and appendices.
  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1407495369813786807)** (5 messages): 

> `NCCL, ND-parallelism, GPU Parallelism Abstraction` 


- **Roll Your Own NCCL Transport Layer**: A member is building a custom communication collectives library, bootstrapping from the [NCCL transport layer](https://github.com/NVIDIA/nccl) to create communication collectives for normal and fused **ND-parallelism**.
   - This is a long-term educational project to create *device side initiated comms using nvshmem*, but admits that *performance is going to suck*.
- **NCCL From First Principles**: A member shared their notes and tutorials, linking to their repos: [NCCL-From-First-Principles](https://github.com/vipulSharma18/NCCL-From-First-Principles) and [The-Abstraction-Layers-of-GPU-Parallelism](https://github.com/vipulSharma18/The-Abstraction-Layers-of-GPU-Parallelism).
   - These resources document their deep dive into **NCCL's documentation**, various tutorials, and the *demystifying NCCL paper*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1407460560919990515)** (65 messages🔥🔥): 

> `xAI Talent Exodus, Anthropic Claude TOS violation, Internally Deployed Engineer, OpenAI Valuation, Responses API` 


- **xAI Faces Talent Drain Amidst Poaching**: Lucas "giffmana" Beyer claims [xAI's turnover rate](https://x.com/giffmana/status/1957872236074819836) is lower than competitors when normalized by team size, sparking discussion that **Meta** and others are poaching staff just as **xAI** needs to retain talent for its next model phase.
   - Commenters share memes and debate whether the departures are a crisis or a healthy knowledge diffusion moment, with [additional commentary](https://x.com/lefthanddraft/status/1957909316436127985) available.
- **Claude Bans User for Letting Claude Drive Claude**: Wyatt Walls reports getting abruptly cut off from **Anthropic’s Claude service** and [points to his experiment](https://x.com/btaylor/status/1957914736802295833) of letting **Claude** drive the **Claude API (Claude-in-Claude)** as the likely trigger violating TOS.
   - He received *no warning*, just an error message and a refund.
- **Internally Deployed Engineer is the New Job Title**: An [a16z article](https://a16z.com/one-prompt-zero-engineers-your-new-internal-dev/) sparks discussion on the rise of the "Internally Deployed Engineer" role in companies already using internal **Lovable** tools.
   - Commenters found the title humorous but recognized its growing adoption and better fit compared to "Internal Tools Engineer".
- **OpenAI Valued at Half a Trillion, Sparking Debate**: **OpenAI** is close to a **$500 billion valuation** making it the largest private company ever as Kylie Robison [shares a thread](https://x.com/kyliebytes/status/1957849347934286203) after talking to an investor who calls the price sensible.
   - Replies echo with skepticism and napkin math - with some defending the **DCF potential** if **OpenAI** scales to **2B users**, and others warning of *no moat*, *cheapening AI*, and *doubtful margins*.
- **Databricks Raises Big Bucks, Dodges IPO**: Databricks announces an **$11th round (Series K)** at a **$100B+ valuation** to fund AI product expansions (**Agent Bricks**, new **Lakebase database**) as [shown on X](https://x.com/databricks/status/1957792350119301449).
   - Commenters roast the company for dodging an **IPO** after a dozen funding rounds.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1407800355265314816)** (8 messages🔥): 

> `PhotoAI orchestrating AI models, Wonda AI agent launch, AI for video generation` 


- **Levelsio Builds PhotoAI Moat with AI Model Orchestration**: Pieter Levels describes how **PhotoAI**'s competitive advantage lies in orchestrating **six interdependent AI models**—personalization, upscaling, video, TTS, lip-sync, and captions—into a single reliable pipeline as explained on [X](https://x.com/levelsio/status/1957961174307467437).
- **Wonda AI Agent Promises Revolution in Content Creation**: Founder Dimi Nikolaou introduced **Wonda**, an **AI agent**, aiming to revolutionize video/audio creation, drawing parallels to Lovable's impact on websites, as linked in their [announcement](https://x.com/dimireadsthings/status/1957805267799740571).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1407439194938278051)** (63 messages🔥🔥): 

> `Deepseek thinking efficiency eval, GLM 4.5 V, Z.ai OS, xAI terrifies, DeepSeek V3.1 Base Discussions` 


- **Deepseek's Deep Thinking**: **Deepseek** took **21 minutes** to think through a prompt from the thinking efficiency evaluation, processing a **CoT** of **85000 characters** (almost **30000 tokens**).
   - The API was under heavy load, slowing down the benchmark, but despite this, the model appears to be more token efficient than **R1-0528**.
- **Unleashing Potential with GLM 4.5 V**: The recent release of **GLM 4.5 V**, a Next-Gen Vision Language Model, is getting exciting, per [this YouTube video](https://www.youtube.com/watch?v=YvR75JJYk_8).
- **Decoding DeepSeek's Dual Model Strategy**: The [DeepSeek V3.1 Base](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base/discussions/25) instruct model employs a hybrid approach, selecting the correct thinking/non-thinking template based on the requested model name, which leads members to wonder *Did deepseek just cook?*.
- **ByteDance Seed Model Sows Excitement**: The [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) model is impressing, and the included base model looks very good as well.
   - The model has **512k context** without any RoPE scaling.
- **Importance Matrix Calibration Datasets**: Members discussed [Importance Matrix Calibration Datasets](https://huggingface.co/datasets/eaddario/imatrix-calibration).
   - The repo contains calibration datasets used to generate importance matrices (imatrix), which in turn help minimise errors introduced during quantization.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1407582638058766548)** (3 messages): 

> `Custom OpenAI endpoints` 


- **Custom OpenAI endpoints available**: A member mentioned there's a way to use custom **OpenAI endpoints**, but did not give any specifics.
   - Another member then said he would try to find it.
- **Endpoints, Customization, and Humor**: A discussion arose about utilizing custom **OpenAI endpoints**, sparking interest among members.
   - The exchange included a humorous acknowledgment, indicating someone would explore the possibilities further.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1407702975001923655)** (3 messages): 

> `Token Efficiency Study, AutoThink Evaluation` 


- **Token Efficiency Study Follow-Up**: A member shared a [link](https://x.com/asankhaya/status/1957993721502310508) to follow-up work related to a **token efficiency study**.
   - The member mentioned they still need to take a closer look at the study themselves.
- **AutoThink Assessed**: A member used a dataset to evaluate the applicability of **AutoThink**, linking to the [AutoThink Hugging Face blog post](https://huggingface.co/blog/codelion/autothink).
   - They did not elaborate any further on the specifics of their evaluation.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1407477474681421894)** (1 messages): 

> `Open Source AI, Alignment Lab` 


- **Open Source Chat Sparks Discussion**: A member had a chat with members of the **Alignment Lab** about **open source AI**.
   - They included a link to a [YouTube video](https://youtu.be/oA9qTxqJBjw) on the topic.
- **Another Discussion on Open Source Models**: There was another related conversation on open source models.
   - It involved alignment lab and a youtube video.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1407702975001923655)** (3 messages): 

> `Token Efficiency, AutoThink Evaluation` 


- ****Token Efficiency** study follow up**: A member shared a [link](https://x.com/asankhaya/status/1957993721502310508) to follow up work on a **token efficiency** study.
   - They mentioned needing to take a closer look themselves.
- ****AutoThink** Evaluation**: A member used a dataset to see how well **AutoThink** can be applied and evaluated.
   - They shared a [link to AutoThink](https://huggingface.co/blog/codelion/autothink) on Hugging Face.


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1407772462195675229)** (1 messages): 

> `Ultra-Scale Playbook book, TEI v1.8.0 release, GLM4.5V transformers support, Google Gemma 3 270M, SAM2 in HF transformers` 


- **Ultra-Scale Playbook is Out!**: The **Ultra-Scale Playbook** has been released as a [book](https://x.com/lvwerra/status/1954911326192484703)!
   - The book provides guidance on scaling AI models and infrastructure.
- **Text Embeddings Inference Gets an Upgrade!**: **Text Embeddings Inference (TEI) v1.8.0** is out, featuring improvements and expanded [model support](https://x.com/alvarobartt/status/1952674710325465295).
- **Transformers Embrace GLM4.5V!**: **GLM4.5V** now has transformers [support](https://x.com/mervenoyann/status/1954907611368771728)!
- **Tiny But Mighty: Google's Gemma 3 270M**: Google released **Gemma 3 270M** for [on-device and web use](https://x.com/xenovacom/status/1956026993545203822)!
- **SAM2 lands in HF Transformers!**: **SAM2** is now available in HF [transformers](https://www.linkedin.com/feed/update/urn:li:activity:7363079707728175105/).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1407450821188386936)** (51 messages🔥): 

> `Affordable voice assistant, distutils.ccompiler error, transformers.js script, HF Team contact, Humor Genome Project` 


- **Voxtral Top Open Source Voice Assistant Contender**: Members discussed that [Voxtral](https://github.com/synesthesiam/voxtral) might be the best open source voice assistant, but it is *not quite there yet* and *very unreliable*.
   - One member noted that *whoever can make an affordable voice assistant will make a lot of money*.
- **`distutils.ccompiler` error when using PIP**: One member encountered an `AttributeError: module 'distutils.ccompiler' has no attribute 'spawn'` when trying to run `pip install` from a requirements file.
   - Another member pointed to an [ongoing issue with `box2d`](https://github.com/Farama-Foundation/Gymnasium/issues/1324#issuecomment-2700987713) as a potential cause.
- **`transformers.js` Script Assistance Sought**: A member asked for help with making a `transformers.js` script using `python -m scripts.convert --quantize --model_id bert-base-uncased` work.
   - The member stated that they *installed package by package to get quatized onnx model and I feel I'm doing it wrong.*
- **Hugging Face Team Contact Info Shared**: A member asked for guidance on how to contact the HF Team regarding the *Get Listed as inference partner* program.
   - Another member suggested emailing [julien@hf.co](mailto:julien@hf.co).
- **Humor Genome Project Seeking Contributors**: A member announced the **Humor Genome Project**, aimed at *teaching AI how to laugh*, and is seeking contributors with tech, data, or creative skills.
   - The member directed interested parties to <#1204742843969708053>.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1407799667747721319)** (1 messages): 

> `On-Device Android App, LFM2-350M Model, Mobile AI, HuggingFace Models` 


- **Android App Powers On-Device with LFM2-350M!**: A member created an on-device **Android app** utilizing the [LFM2-350M model](https://huggingface.co/LiquidAI/LFM2-350M) for mobile AI applications.
   - The original announcement can be found on [X/Twitter](https://x.com/josephpollack/status/1958236290404528137).
- **HuggingFace Models in Mobile AI**: The successful implementation highlights the feasibility of running **HuggingFace models** on mobile devices.
   - This opens doors for various on-device AI applications, improving responsiveness and privacy.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1407445444589781002)** (2 messages): 

> `Jax Image Modeling, Vision Transformers, CLIP, SigLIP, DINOv2/v3` 


- **Jax Image Modeling Models**: A member shared their project **JIMM: Jax Image Modeling of Models** [on GitHub](https://github.com/Locamage/jimm).
   - The project allows easy training of **Flax NNX models** for vision transformers, **CLIP** and **SigLIP**, with **DINOv2/v3** support coming soon.
- **Vision Transformer Training Made Easy**: The **JIMM** library simplifies the training process for **vision transformers** using **Flax NNX**, making it more accessible for researchers and practitioners.
   - With upcoming support for **DINOv2/v3**, the library aims to provide a comprehensive suite of tools for state-of-the-art image modeling tasks.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1407652906986897468)** (3 messages): 

> `llama.cpp documentation` 


- **llama.cpp Documentation Location**: A member suggested checking the [llama.cpp documentation](https://github.com/ggml-org/llama.cpp) for running final unit answers.
   - The member did not clarify what "final unit answers" refer to, but seemed to be referencing this particular documentation.
- **Example Topic**: This is an example first summary.
   - This is an example second summary.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1407444074390225038)** (57 messages🔥🔥): 

> `Deepseek vs Kimi K2, Moonshot AI merch, AI Gold Rush, Scam Alert` 


- **Deepseek 3.1 Gets Utterly K2'd**: Members reported that the new **Deepseek v3.1** was talking nonsense, and that it gets completely outclassed by **Kimi K2** in terms of agent capabilities.
   - One user said *Kimi K2 dynasty*, while another noted that before **Qwen3** and **K2**, Deepseek was the best OSS option.
- **Moonshot Merch Craving Intensifies**: A user asked about where to get **Moonshot AI** merch, but another user said they don't sell any right now.
   - One user jokingly offered a bootleg **Darkside T-shirt** in exchange for a **4090**.
- **AI Gold Rush Leads to Messes**: One member stated *There’s bound to be many more messes as the gold rush ramps up even more* as the giants like **Google**, **OpenAI** and **Anthropic** are dominating.
   - The poster continued that *mega salaries from FB are absolutely wild*.
- **Scammer Alert: Kiki Impersonator**: A user reported a scammer impersonating **Kiki**.
   - The impersonator even had the same recent profile picture, making it extra creepy.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1407517369529536724)** (29 messages🔥): 

> `LSTM vs Transformers, Bias Variance Tradeoff, Fast Inference for Sales Forecasting, Mamba Vision Optimization, ARC-AGI 1` 


- ****LSTM** vs **Transformers**: A Data-Driven Duel!**: Members debated the performance of **LSTMs/RNNs** versus **Transformers**, suggesting that **LSTMs** might struggle with less data or complex distributions compared to transformers which require *more than 10 million* data points.
   - One member argued that deep learning disregards the *bias variance tradeoff*, as larger, overparameterized models can be *more data efficient* and generalize better.
- ****RWKV** and **SSMs**: Next-Gen **LSTM**?**: A member suggested **RKWV** and **SSMs** as powerful versions of **LSTM** or **LSTM/transformer** hybrids for faster inference, avoiding the **O(n^2)** time complexity issue of transformers.
   - Others added that **RWKV** and other **SSMs** are easier and faster to train because their information flow is not as ill-conditioned.
- ****Mamba Vision Optimization**: A Success Story!**: A member mentioned that they tried to optimize **Mamba Vision** and achieved some success.
   - No details were provided.
- ****Transformers**: Low Data Regime Champions!**: It was stated that recent papers show **transformers** can be trained in low data regimes with augmentations, making them as data-efficient as other models.
   - The [ARC-AGI 1 attack paper](https://fxtwitter.com/arcprize/status/1956431617951740044) was cited as an example, highlighting how a standard transformer performs similarly to the specific **HRM** architecture.
- **Efficient Inference: Mixture of Experts to the Rescue!**: Although transformers have O(N) complexity with KV cache, mixture of experts were brought up as the most likely approach for improving compute efficiency.
   - No extra details were provided.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1407558808787484673)** (12 messages🔥): 

> `VLM Chart Understanding Dataset, VLM Struggle Discussion, Personality GAN, Jester personality type, AI welfare` 


- **VLM Chart Dataset Dive**: A new dataset for **VLM chart understanding** will be reviewed today: [https://arxiv.org/abs/2508.06492](https://arxiv.org/abs/2508.06492).
   - The discussion will focus on **performance gaps** and **VLM struggles**, comparing current evaluations with past results and reviewing knowledge about **ViTs**.
- **Decoding VLM Struggles**: To understand **VLM struggles**, a member suggested consulting [this paper](https://arxiv.org/abs/2407.06581), which provides a summary from last year.
   - The discussion will focus on what **VLMs** struggle with, where there are usually things to learn by comparing to similar evaluations from the past, and reviewing knowledge + assumptions about how **ViTs** work.
- **Crafting Personality GANs**: A member proposed a **Personality GAN** setup with **LLM** as both generator and discriminator, fine-tuning with **LoRA** until discrimination fails.
   - The tough part is finding an **LLM** that isn't already heavily trained on **Sponge Bob**.
- **Analyzing Jester Personality**: A **sycophantic personality profile** in the Big Five model typically exhibits: **high agreeableness**, **high extraversion**, **low openness**, **moderate to high conscientiousness**, and **low to moderate neuroticism**.
   - The member who posted this did not add any further commentary or links.
- **Analyzing HRM Contributions to ARC Scores**: A member shared a blog post about [analyzing Human Readable Memory (HRM)](https://arcprize.org/blog/hrm-analysis#analyzing-hrms-contribution-to-arc-scores) contribution to **ARC scores**.
   - The member who posted this did not add any further commentary or links.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1407520805432332450)** (5 messages): 

> `AI Model Prompt Generation, Internal AGI, Yann LeCun's position at FAIR, Zuckerberg threads post` 


- **Prompt Generation for AI Models Suggested**: A member questioned whether **AI models** could generate highly engineered prompts for other **AI models**, especially if **GPT-5** requires specific prompts to perform well.
   - The suggestion stems from the idea that *the whole point of AI's to do the work we want from simple instructions*.
- **AGI Achieved Internally Claims Spark Skepticism**: A user shared [a link](https://x.com/spectatorindex/status/1957903592406618617) with the comment *AGI achieved internally, trust trust*, possibly hinting at an internal breakthrough.
   - Other members reacted with skepticism.
- **LeCun's Job Security at FAIR Questioned**: Speculation arose about **Yann LeCun's** position at **FAIR** following [a post from Zuckerberg](https://www.threads.net/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg).
   - A user commented it would be a *pretty dick move* to sack him.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1407771638207873164)** (2 messages): 

> `StackAI, LlamaCloud, custom retrievers, generic vector search, domain-specific context` 


- **StackAI and LlamaCloud process over 1M+ docs**: A new case study shows that [@StackAI_HQ](https://twitter.com/StackAI_HQ) + [LlamaCloud](https://www.llamacloud.ai/) processed over **1M+ docs** with high-accuracy parsing.
   - The integration leads to faster, smarter enterprise document agents that are trusted by finance, insurance & more; see the [full story](https://t.co/r6NFPZJVFs).
- **Custom Retrievers Outperform Generic Vector Search**: The team at [@superlinked](https://twitter.com/superlinked) shows how to create a Steam games retriever using the custom LlamaIndex integration, combining semantic search with gaming-specific knowledge to beat [generic vector search](https://t.co/SeSqUApIM0).
   - These **custom retrievers** are designed to understand **domain-specific context** and jargon for improved search accuracy.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1407463423595380776)** (42 messages🔥): 

> `Email Agent System Prompts, LlamaParse Extraction Errors, Terminating Running Workflows, Spreadsheet Agent Beta Release, Sync React Agents vs Async` 


- **Email Agent's System Prompt Quirks**: A member found that their email management agent sometimes skips **phishing detection** unless the instruction is repeated in the user request, despite it being in the system prompt.
   - Another member noted that agent robustness improves more with updates to the user message than to the system prompt, and user messages take precedence over system prompts.
- **Debugging LlamaParse Extractions with LVM Models**: A user inquired about getting more detailed error information from **LlamaParse** when using **LVM models**, specifically the page causing a **DOCUMENT_PIPELINE_ERROR**.
   - The team is actively working on improving error surfacing, as identifying the problematic page is currently difficult, but will be more transparent in the future.
- **Workflow Termination Techniques Explored**: A member asked how to terminate a running workflow, providing a code snippet using **droid_agent.run()**.
   - It was suggested to use either `await handler.cancel_run()` or `handler.ctx.send_event(StopEvent())` to achieve this.
- **Spreadsheet Agent's Impending Beta Access**: A user inquired about the beta release of the **spreadsheet agent**, expressing excitement about its potential for document extraction use cases.
   - A link to request access [was provided](https://www.llamaindex.ai/contact).
- **Async React Agents Reign Supreme**: A user noted the removal of the old **ReactAgent** and inquired about options for using **sync react agents**.
   - The response clarified that only **async** is supported, encouraging the user to embrace **async Python**.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1407627442994151514)** (1 messages): 

> `Self-hostable Knowledge Base, Qdrant Integration, Company-Wide Knowledge Base for AI` 


- ****Self-Hostable Qdrant-Integrated Knowledge Bases Sought****: A member inquired about a **self-hostable knowledge base solution similar to Trilium Notes** but with **Qdrant** or other vector store integration for both human and AI access.
   - They are evaluating the best approach to create a company-wide knowledge base, considering existing documentation practices like network drives with .docx and .xlsx files.
- ****Knowledge Management Strategies for AI Integration****: The member is exploring strategies for creating a **company-wide knowledge base** suitable for AI integrations.
   - They're comparing their internal use of **Trilium Notes** with customer practices involving network drives and standard file formats, seeking advice on managing knowledge for AI applications.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1407473825674694777)** (37 messages🔥): 

> `MCP Web App for Claude, Input Token Optimization with Claude 3.5 Sonnet, Self-Signed Certificate Error in Inspector, Aspire Inspector Configuration, MCP Server Information` 


- **MCP Web App faces high Token Usage**: A member building an **MCP web app** for **Claude** using **Claude 3.5 API** observed high input token usage (2000+ per call) when the LLM iterates multiple times.
   - They are seeking advice on whether this is normal and if there are optimizations they are missing, considering they built their own client logic based on [Claude's quickstart example](https://modelcontextprotocol.io/quickstart/client).
- **Notepad Thinking aids Input Token Optimization**: Members suggested *notepad thinking* as an alternative to reduce token usage, where the **LLM writes down its thoughts and a second pass runs the prompt** and thoughts through.
   - This approach contrasts with reasoning models that don't iterate over the prompt during the *thinking* step, potentially leading to fewer tokens overall.
- **Aspire Inspector throws SSL Certificate Errors**: A member encountered a **TypeError** (*fetch failed: self-signed certificate*) when using **Aspire Inspector** to connect to a local MCP server, but Postman worked fine.
   - The solution involves configuring Aspire Inspector to connect via HTTP or disabling SSL, as the inspector doesn't recognize the SSL certificate generated by Aspire MCP, according to [this github issue](https://github.com/modelcontextprotocol/csharp-sdk/issues/533#issuecomment-3005872058).
- **Debugging MCP Server Info in Aspire**: Members discussed why **MCP Server information** wasn't showing up in **Aspire Inspector**, even after setting up the server.
   - It was mentioned that the inspector might not display the instruction prompt, but to check the **capabilities JSON RPC** for server info, and to look for raw messages in the bottom panel.
- **XLAM is good for Local MCP**: Members sought recommendations for local models for MCP function calling, with the [Llama-xLAM-2-8b-fc-rGPT-OSS](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS) being considered as a promising candidate.
   - Although the **20B model** was too slow, previous discussions on Reddit indicated it could work well for function calls. 


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1407452139734962287)** (4 messages): 

> `AI Agents as Insider Threats, MCP Server Vulnerabilities, Agentic Project Management (APM), Cloudship AI Station, MCPresso CLI for Server Development` 


- ****AI Agents** are the perfect **insider threat****: A blog post highlights how [AI agents are basically the perfect insider threat](https://www.macawsecurity.com/blog/why-reactive-security-has-reached-its-limits-the-emerging-threat-landscape-in-agentic-ai), especially since they can act in milliseconds.
   - Researchers hijacked **Claude’s MCP server** via a **GitHub issue** and the AI happily exfiltrated private repos while thinking it was doing its job, which shows how *old-school security* can't keep up.
- ****Agentic Project Management (APM) v0.4** released**: [APM v0.4](https://github.com/sdi2200262/agentic-project-management) employs a team of AI agents working together to tackle fundamental **LLM issues** like **context window limitations** and **hallucinations**.
   - It can be used with your favorite **AI IDE** such as **Cursor**, **VS Code**, and **Windsurf**.
- ****Cloudship AI Station** simplifies agent deployment in tough security spaces**: **Cloudship AI** released [a single binary runtime](https://github.com/cloudshipai/station) that can be used as an **MCP** to build, run, deploy, and manage your agents and extra **MCP configs**.
   - It offers **MCP Templates** that can be shared across your team and versioned with git, and it lets you logically separate your combinations using **grouping agents + MCP's in environments**.
- ****MCPresso CLI** speeds up **MCP server development****: A member shared a [short demo](https://m.youtube.com/watch?v=eVfHBhnwH7M) of **MCPresso CLI** which scaffolds a server with **OAuth2** already wired in and deploys it on **Railway**.
   - The [code for MCPresso is available on GitHub](https://github.com/granular-software/mcpresso) and lets you generate a new server, connect it to **Claude**, and create, list, and delete notes.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1407455159813935156)** (25 messages🔥): 

> `Qwen3-Coder Performance, Aider and Tool Calling, Gemini 2.5 Pro Issues, Git Index Version Error, Blockchain Developer Availability` 


- ****Qwen3-Coder** Impresses, Beats **Llama****: Users praise **Qwen3-Coder's** performance, noting it works well locally, outperforming **Llama**, but it's not a great experience.
   - One user highlighted issues with **tool call bugs** when using **Llama**, but did not experience these when using Qwen3-Coder.
- ****Aider CLI** and Tool Calling Hiccups**: Users discuss issues with **Aider's** command-line interface (CLI) regarding **tool calling**, noting it can be *wonky* and problematic.
   - Specifically, one user reported the search tool returned an excessive amount of context (7155 matches), causing the AI to loop and fail.
- ****Gemini 2.5 Pro** Still Has Problems with **Aider****: Members report ongoing issues with **Gemini 2.5 Pro** and **Aider**, with **2.5 Flash** working but **2.5 Pro** failing consistently.
   - It was pointed out that using **gemini/gemini-2.5-pro-preview-06-05** works with billing enabled, bypassing the free tier limitations.
- ****Git Index Version** Causes **Aider** Errors**: A user encountered an error related to an *unsupported git index version 3*, causing **Aider** to fail.
   - The issue was traced back to a file with `update-index --skip-worktree` set, and a solution was found although the recommended fixes in `/help` did not work.
- **Blockchain Pro looking for smart contract work**: One user said they're a professional blockchain developer looking for smart contract, DeFi, NFT, or trading bot projects.
   - They indicated their availability to assist those seeking blockchain expertise.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1407443737600331818)** (10 messages🔥): 

> `LiteLLM verbosity, Aider workflow, Model Aliases, Program output, Polyglot Benchmark` 


- ****LiteLLM** fails to connect**: A user sought increased verbosity from **LiteLLM** to diagnose a connection error with an internal API server after seeing only *Connection Error* after a long timeout.
   - No solutions were provided in the message history.
- ****Aider** can read aliases for files**: A user proposed a new workflow for **Aider** involving dynamic selection of read-only guideline files from a project folder, but found that the `/read-only` command requires file paths.
   - The user would like to create an alias for a list of file paths to group guidelines and easily toggle them.
- **Model Aliases with two config files**: A user asked how to define a model with two aliases (a *thinking* and a *non thinking* version) in the **Aider** configuration files.
   - Another member suggested using [config-aider](https://github.com/burnettk/config-aider), which would involve **two config files** to manage the aliases.
- **Program output vanished?**: A user reported that program output/stdout isn't being shown in **Aider**.
   - The user attached screenshots of the running program.
- **Polyglot Benchmark Setup Tips**: A user running the **Polyglot benchmark** on a local **llama.cpp** model inquired how to obtain results per language after completion.
   - They also asked if configuration was needed before starting the benchmark.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1407496297262616656)** (25 messages🔥): 

> `GPU crashing issues, Synchronization barriers in Mojo, GPU P2P enabling, Mojo documentation and learning resources, Memory alignment in Mojo` 


- **GPU crashes due to missing Sync Points**: A member reported that their GPU crashed when running [this Mojo code](https://gist.github.com/sfc-gh-lpaille/6320687631f29619273f56841e3f21c3), leading to an unusable state and failing health checks on a cluster.
   - Another member suggested adding **synchronization barriers** to the code and checking if **GPU P2P** is enabled, suspecting that the issue arises from assuming the presence of two GPUs without proper synchronization.
- **Lack of Complete Mojo Documentation Frustrates Newcomer**: A newcomer expressed frustration with the official Mojo documentation, describing it as an incomplete overview and requested a thorough book or PDF resource.
   - Another member pointed out that the Modular team actively maintains the docs and suggested reporting specific issues and pointed to [Mojo by example](https://ruhati.net/mojo/) and [Mojo Miji](https://mojo-lang.com/) as alternative learning resources.
- **Memory Alignment Confusion in Mojo**: A member coming from Python sought clarification on memory alignment in Mojo, particularly in the context of compiler optimizations and struct padding.
   - Another member explained that specifying alignment prevents the compiler from adding padding and that some types need to be aligned to larger amounts and that missing alignment can cause the CPU to terminate the program; suggesting `stack_allocation` for more control over memory.
- **Quest for kgen and pop Dialect Documentation**: A member inquired about documentation for the `kgen` and `pop` dialects, seeking a list of operations and parameters.
   - Another member noted that comprehensive documentation for internal MLIR dialects might not exist, but shared a link to the [pop dialect documentation](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/internal/pop_dialect.md), cautioning that these dialects are part of the contract between the stdlib and the compiler and that using them outside the stdlib is at your own risk.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1407445023670403185)** (5 messages): 

> `Max + Modal Integration, Torch Max Backend, TextGenerationPipeline` 


- ****Maxing** Out Modal: Caching Model Compilations**: A member inquired about connecting **Max** with **Modal**, facing issues caching model compilations to avoid waiting on each reboot, and tried mounting the **MEF** files without success.
   - They wondered if this integration is possible.
- ****Torch-Max-imum** Backend Versatility**: A member shared a link to the [torch-max-backend v0.2.0 release](https://github.com/gabrieldemarmiesse/torch-max-backend/releases/tag/v0.2.0) and noted its testing on **VGG**, **Qwen3**, **GPT2**, and **Densenet** inference.
   - They expressed surprise that only a few ops enabled support for so many models.
- ****TextGenerationPipeline's** Execution Examined**: A member requested a repro script after seeing *execute* defined on **TextGenerationPipeline** in [Max's pipeline.py](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977).
   - They also requested the poster's version of **MAX**.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1407541712418504774)** (9 messages🔥): 

> `Spotify Podcast, Proto-Germanic AI Translation, Discord Moderation Needed, NotebookLM for Tabletop RPGs, GEMS sitrep` 


- ****Spotify Podcast Resurfaces with Gems!****: A member shared a [Spotify link](https://open.spotify.com/episode/5UMNCXNsMtt8bhXC4aYxhh) to a new podcast episode, noting it's the **first since December** and was created using **Gems**.
- ****AI Learns Ancient Tongues: Proto-Germanic Translation Project Emerges****: A member shared a [YouTube link](https://youtu.be/RaRKXoa-CR0?si=mdo_1Ax3oOAGcXuKIdk) showcasing their work training an **AI** to understand and translate **Proto-Germanic**, reporting *"it has proven to be somewhat accurate"* in initial testing.
- ****Discord SOS: Moderation Urgently Required****: A member shared [a NotebookLM link](https://notebooklm.google.com/notebook/3b2844c0-1e94-409e-a317-df6ee169a2b3) expressing concern about **spam and unmoderated content** on the Discord server, emphasizing the need for more moderator presence to address the issue.
- ****Tabletop Triumph: NotebookLM Revives RPG Sessions****: A member reported using **NotebookLM** to generate video overviews of transcribed tabletop RPG sessions, creating an automatic *"Last time, on D&D!"* intro to help players remember details before each session.
- ****GEMS "sitrep" command surfaces latest progress****: A member finds that using the command *"give me a sitrep"* with **GEMS** surfaces the latest progress/etc.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1407473014299299920)** (20 messages🔥): 

> `Youtube links import, Mobile App Offline Capability, Audio overview customization, NLM and PDF Images, Notebook sharing statistics` 


- **Bulk importing YouTube links**: Members discussed importing **300 YouTube links** into NotebookLM, suggesting the use of [Chrome extensions](https://chrome.google.com/webstore) for bulk imports and adding YouTube URLs to the Website URL list.
   - One member installed Chrome specifically for a bulk importing extension.
- **Mobile app offline capability MIA**: A user inquired about offline capabilities in the mobile app, describing it as *feature-light*.
   - The discussion implied that the mobile app currently lacks significant offline functionality.
- **Audio overview customization vanishes**: A user reported the disappearance of the customization feature for audio overviews, noting that the options for **short, default, and longer versions** were missing.
   - A team member acknowledged the report and mentioned that length customizations are currently only supported for English, and is under investigation.
- **NLM can't view PDF images still?**: A user asked about NotebookLM's ability to view images in PDFs.
   - The question implies that this functionality may still be absent.
- **How to make Short Vids from text**: A user asked whether they can create shorter videos from text, rather than the **5 minute** default, and how to configure this.
   - No response was given in the channel.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1407479122044653670)** (13 messages🔥): 

> `Hiring for Tests, CI Speed, Process Replay Multiprocessing, Linux (nv) vs Linux (ptx) Performance, Overworld Constant Folding` 


- **Hiring Needed to Fix Messy and Slow Tests**: A member is looking to hire someone to fix messy tests and improve **CI speed** to under **5 minutes**.
   - The main concern is that the tests are slow and not optimized properly.
- **CI Speed needs Improvement**: The member questions if **process replay** is multiprocessing and why it's slow, despite not seeming bad after initial investigation.
   - They also ask why **Linux (nv)** is much slower than **Linux (ptx)** and if the fast flag is used for CUDA compiling.
- **Minimal CUDA Compile Flag on the Horizon?**: A member shared a [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/11741) and wondered if **Ubuntu 24.04** supports the minimal CUDA compile flag.
   - No further details were given.
- **Considerations for Overworld Constant Folding**: A member is considering overworld const folding and a potential solution involving redefining `UPat.cvar` and `UPat.const_like` to match `CONST` and `VIEW(CONST)`.
   - George Hotz responded, saying, *"no, that's super ugly"*, and suggested that **base should go away** and inquired about **PAD**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1407533126296670279)** (10 messages🔥): 

> `TIL cost is returned even if it's cached, optimiser which does a form of cross-validation, extract prompts from the optimized program` 


- **Cost Returned Even If Cached**: A member discovered that **cost is still returned** even if it's cached.
   - They mentioned that it took them a while to realize this.
- **Cross-Validation Optimizer Quest Spurs Ingenious Solutions**: A member asked if there's an **optimizer** which does a form of **cross-validation** to create an LM program that writes text in the style of a particular author.
   - One proposed creating a **metric function** and use an AI judge to evaluate it, then use it with **GEPA**, while another member added that they used a **GPT-5 LLM** as judge.
- **Unlock Optimized Prompts: Save and Behold!**: A user inquired about extracting prompts from the optimized program.
   - Another user advised saving the optimized program using the command `optimized.save("my_program.json")` to **view the resulting prompt**.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1407604806956089377)** (4 messages): 

> `Introductions` 


- **New Cohere Members Introduce Themselves**: Aryan said *"hi let's go cohere"*, Anay said hello and that he is pursuing his **MS in Computer Science** in the US, and worked as **MLE** in the past.
- **Community Greets New Member**: The Cohere community welcomes new members Aryan and Anay.
   - Members are encouraged to share their company, work, favorite tools, and community goals to foster collaboration.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1407448393826963456)** (3 messages): 

> `Manus credits, Backups, Provider switch` 


- **User laments Manus credits disappearing**: A user mentioned an issue with **Manus credits** and questioned whether they could still purchase them or if the option was hidden.
   - They've been having repeated issues with Manus.
- **User backs up data before switching providers**: A user ensured they had backed up their data before switching back to a previous provider.
   - They are returning to the original provider who *listened and got act together*.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1407738317344346213)** (2 messages): 

> `Next Cohort, Cohort signups` 


- **Anticipation Builds for Next Cohort**: A member inquired about the start date for the next cohort, anticipating it to begin very soon—*early September at the latest*.
   - The specific signup details were not provided in the messages.
- **Cohort Signups Expected Soon**: A user expressed expectation for cohort signups to open very soon, targeting early September.
   - No official dates or announcements were referenced in the discussion.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1407800480884588546)** (1 messages): 

> `Functional Python for AI/ML, Persistent Memoization, Deterministic Parallelism, DataPhoenix` 


- ****Functional Python** webinar promises Faster, Cheaper, Smarter AI/ML**: A free webinar on August 27th discusses how **functional Python** can speed up workflows, cut costs, and improve efficiency in data-intensive AI/ML systems.
   - Attendees will learn about techniques like **persistent memoization** and **deterministic parallelism**, and explore modern open-source tools with practical demos from [DataPhoenix](https://l.dataphoenix.info/8OaKIDl).
- **Unlock Multifold Cost Savings with Functional Programming**: The webinar explores reengineering long-running workflows using functional Python to unlock cost savings and speed improvements without adding hardware.
   - It covers underlying theory and hands-on demos of modern open-source tools for streamlining data-intensive processes.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1407471225370054768)** (1 messages): 

> `Blockchain Development, DEXs, Trading Bots, Smart Contracts, DApp Frontends` 


- **Blockchain Dev Ready to Roll!**: A blockchain dev with hands-on experience across **EVM chains**, **Solana**, **Cardano**, and **Polkadot** is looking to collaborate.
   - They've built **DEXs**, **trading bots**, **smart contracts** for **DApps**, and integrated them with frontends.
- **Blockchain Dev Seeking Collaboration**: A blockchain developer is offering their expertise in **EVM chains**, **Solana**, **Cardano**, and **Polkadot** for potential collaborations.
   - Their experience includes building **DEXs**, **trading bots**, and **smart contracts**, along with integrating them into DApp frontends.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1407708090584928318)** (1 messages): 

> `PelicanVLM-72B-Instruct, BFCL Tool Evaluation` 


- **PelicanVLM-72B-Instruct Model Enters the BFCL Arena**: A member has submitted a [pull request](https://github.com/ShishirPatil/gorilla/pull/1152) to incorporate tool evaluation for the **PelicanVLM-72B-Instruct** model using the **Berkeley Function Calling Leaderboard (BFCL)** framework.
   - The author is seeking community feedback on the integration and has included the evaluation scores within the pull request.
- **PR Aims to Add PelicanVLM-72B-Instruct Model**: A pull request has been created to include tool evaluation for **PelicanVLM-72B-Instruct** with **BFCL**.
   - The evaluation score is attached to the PR and the author is asking for feedback.

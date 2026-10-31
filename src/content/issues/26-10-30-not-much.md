---
id: MjAyNS0x
title: not much happened today
date: '2026-10-30T05:44:39.731046Z'
description: >-
  **Anthropic** renamed its Max subscription tiers, removing the 5-hour session
  limit for verified Ultra accounts, improving user experience. **Google
  DeepMind** released **Gemini 4 Carbon**, topping multiple AI benchmarks, and
  previewed the interactive **Genie 4 world model**. **Mistral Large 4** with 1
  trillion parameters supports advanced fine-tuning and ranks highly in coding
  benchmarks. **NVIDIA** published a tech report on **Nemotron 3.5**, a
  multimodal model excelling in efficiency and quantization. Smaller specialized
  models include a **Microsoft** 4B hybrid agent excelling in computer-use tasks
  and **Phonic-1** leading speech-to-speech benchmarks. *"Anthropic is on a
  run"* and *"Huge!!!"* were notable reactions to subscription changes.
companies:
  - anthropic
  - google-deepmind
  - mistral-ai
  - nvidia
  - microsoft
models:
  - claude-ultra
  - gemini-4-carbon
  - genie-4
  - mistral-large-4
  - nemotron-3.5
topics:
  - subscription-models
  - benchmarking
  - fine-tuning
  - multimodality
  - model-quantization
  - context-windows
  - coding-benchmarks
  - model-efficiency
  - model-performance
people:
  - bcherny
  - alexalbert__
  - theo
  - kimmonismus
  - teortaxesTex
  - aravsrinivas
  - winglian
  - thibaultjaigu
---


**a quiet day.**

> AI News for 10/29/2026-10/30/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Claude Ultra Drops the 5-Hour Limit as Anthropic Renames Its Max Tiers**

- **Plan rename**: Anthropic renamed its Max tiers. Max 5x is now "Max" at $100/month, and Max 20x is now "Ultra" at $200/month. Existing subscribers do not need to take any action ([announcement](https://x.com/claudeai/status/2116220869479428500)).
- **5-hour limit removed**: Verified Ultra accounts no longer have the 5-hour session limit. The change starts today for most existing subscribers ([@claudeai](https://x.com/claudeai/status/2116220869479428500), [@bcherny](https://x.com/bcherny/status/2116222842413142111), [@alexalbert__](https://x.com/alexalbert__/status/2116227345501880781)).
  - **Rollout caveat**: Newer accounts keep a temporary limit until they are verified, which Anthropic describes as an anti-fraud measure. Affected users will get an email when the limit is lifted.
  - **Unchanged**: Weekly limits stay the same.
- **Reaction**: The change was widely welcomed, with posts such as "Huge!!!" ([@theo](https://x.com/theo/status/2116251545033662698)) and "Anthropic is on a run" ([@kimmonismus](https://x.com/kimmonismus/status/2116221615180755409)).
- **Subscription economics (unverified analysis)**: [@teortaxesTex](https://x.com/teortaxesTex/status/2116140119409484062) cites a SemiAnalysis reconstruction of Anthropic's finances.
  - **Margins**: Anthropic's API margin is put at about 92%, versus about 84% for DeepSeek.
  - **Subsidy**: Opus on subscriptions is described as heavily subsidized, breaking even only below about 21.3% utilization.
  - **Revenue vs compute**: A follow-up estimates that subscriptions bring in about 18% of revenue while using more than 50% of compute ([thread](https://x.com/teortaxesTex/status/2116318906243809304)).
  - **Comparison with DeepSeek**: The same author argues that a heavily subsidized DeepSeek V4.1-Flash plan ($10 for about $76.5 of API-equivalent usage) is close in price per token to Claude subscriptions ([long post](https://x.com/teortaxesTex/status/2116122753224831460)).

**Frontier and Open Model Releases**

- **Gemini 4 Carbon (Google DeepMind)**: The model is ranked #1 in Text Arena, #2 in Agent Arena and #4 in WebDev, and changed the Pareto frontier in all three ([Arena](https://x.com/arena/status/2116239697727180890)).
  - **Research quality**: Perplexity reports a new state of the art on its WANDR long-form research benchmark. Carbon is coming to Pro and Max users inside Computer "once we have sufficient token capacity" ([@AravSrinivas](https://x.com/AravSrinivas/status/2116263856628703247)).
- **Genie 4 world model**: Google's new world model generates interactive simulations with sound that users can steer in real time ([@Google](https://x.com/Google/status/2116237215026032872)).
  - **Availability**: It is a research preview for trusted testers. It is coming soon to Google AI Ultra in Project Genie and to Gemini API and AI Studio early-access partners ([rollout](https://x.com/Google/status/2116237216586383523)).
- **Mistral Large 4 ("Le Chonk")**: The model has 1T parameters ([TRL](https://x.com/QGallouedec/status/2116120080794939669)) and 1M-token context ([llama.cpp](https://x.com/llama_cpp/status/2116202682352603141)).
  - **Fine-tuning**: Axolotl has day-0 LoRA support. Its recommended recipe uses NVFP4 experts with merge-aware LoRA and fits on one 8xB200 node ([@winglian](https://x.com/winglian/status/2116081403221705096)).
  - **Coding**: One third-party coding benchmark ranks it the second-best open-source model ([@ThibaultJaigu](https://x.com/ThibaultJaigu/status/2116082719339757827)).
- **NVIDIA Nemotron 3.5 tech report**: Super has 120B total and 12B active parameters and handles text, image and video. It scores 51.78% on SWE-bench Pro, beating Nemotron 3 Ultra with about 4.6x fewer active parameters ([summary](https://x.com/dair_ai/status/2116178819271032975)).
  - **Lightning variant**: A 30B text-only model with 3B active parameters.
  - **Serving**: Up to 4,856 output tok/s/GPU on GB200 while holding at least 50 tok/s/user.
  - **Quantization**: NVFP4 recovers 99.62% of median BF16 accuracy.
- **Smaller and specialized models**:
  - **Microsoft computer-use agent**: A 4B hybrid agent that can both click and write code scores 66.0% on OSWorld-Verified, ahead of 27B and 30B agents ([@HuggingPapers](https://x.com/HuggingPapers/status/2116207128411005054)).
  - **Phonic-1**: Claims #1 on AA's speech-to-speech index, on τ-Voice and on Sierra's τ³-Voice ([@moinnadeem](https://x.com/moinnadeem/status/2116270909397123499)). It was built by a 10-person team with $4M ([@davisblalock](https://x.com/davisblalock/status/2116072630302490670)).
  - **mischief 1 (BioComputing Co)**: An image-to-video model priced from 0.5¢ per second. The company says it is at least 5x faster than its base model and draws on biological-neuron research ([launch](https://x.com/AlexKsendzovsky/status/2116153410101944352)).
    - **Skepticism**: Critics say the link between neuron research and the model is unexplained or marketing ([@willdepue](https://x.com/willdepue/status/2116344121757782224), [@jachiam0](https://x.com/jachiam0/status/2116333872849842472)).
- **OpenAI Private Intelligence**: An enterprise offering that gives businesses stronger data controls when using OpenAI's top models. It is being built with large customers ([@OpenAI](https://x.com/OpenAI/status/2116186509384815084)).

**Independent Evals, Benchmarks and Quantization Disputes**

- **AA Intelligence Index v5**: 60% of the score now comes from private or held-out tests. Scores are Opus 5.5 (max) 58, GPT-6 Astra 57 and GPT-6.1 Sol 55, with Sol costing about one-eighth as much per task ([summary](https://x.com/DeepLearningAI/status/2116245015450005965)).
  - **Effort scaling**: Opus 5.5 scores 35 at low effort versus 58 at max, at about 10x the cost.
- **AA-Omniscience v1.1**: The knowledge and hallucination benchmark was cut from 6,000 to 3,000 questions, with saturated items removed ([details](https://x.com/ArtificialAnlys/status/2116319506318635503)).
  - **Public set**: A new 300-question public set correlates 0.99 with private scores.
  - **Leaders**: Gemini 4 Carbon leads at Net Score 54, followed by Opus 5.5 at 48 and Fable 5.1 at 46.
  - **Other results**: Gemini 4 Argon has the lowest hallucination rate at 4%. MiniMax M3.1 Flash fell 22 points (16 → −6).
- **Agent Arena cost data**: An analysis of 26K sessions finds that without prompt caching, a task on a top-10 agent would cost 3.6x as much ([Arena](https://x.com/arena/status/2116186548588929328)).
  - **Example**: A typical Fable 5.1 task costs $7.44 with caching versus $27 without.
  - **Cache expiry**: A single cache invalidation adds about 5% to a task's bill.
  - **GPT-6.1 Terra (Max)**: It entered Agent Arena at #8 with a 10.6% net improvement at $0.62 median cost per task ([Arena](https://x.com/arena/status/2116239697727180890)).
- **DeepSeek V4.1-Flash-1017 quants**: Unsloth released dynamic v3.1 GGUFs that run on 64GB+ RAM ([@UnslothAI](https://x.com/UnslothAI/status/2116156587404562491)).
  - **Reproducibility dispute**: Unsloth says it cannot reproduce GSQ-RCO's numbers. For example, the claimed 100% AIME25 score at 1-bit came out as 91.88% in its runs ([@danielhanchen](https://x.com/danielhanchen/status/2116162363313525191)).
  - **Reasoning length**: Unsloth also reports that lower-bit quants reason somewhat longer than the unquantized model.
- **DeepSeek-Cyber (unconfirmed)**: Leaked "dreamed" scores circulated, including about 92% on CyberGym and 96.8% on ExploitBench at 3,000 steps. They are not an official release.
  - **Third-party data**: Per Fireworks, CyberGym pass@3 moved only from 94.82% (0910) to 95.36% (1017). [@teortaxesTex](https://x.com/teortaxesTex/status/2116220460690186241) argues that near saturation, almost any number is consistent with the same checkpoint.
  - **Analyst view**: The rumored scores are argued to fit a guardrail-free 1017 rather than a new model ([analysis](https://x.com/teortaxesTex/status/2116107732545683682)).
  - **Usage restriction**: The model reportedly cannot be used by Chinese entities sanctioned by the US government ([note](https://x.com/teortaxesTex/status/2116045868172456177)).

**Safety, Security and Agent Supply Chain**

- **Automated red-teaming (Anthropic)**: Thousands of Claude agents were tasked with evading a training-data monitor ([@AnthropicAI](https://x.com/AnthropicAI/status/2116349211549843666)).
  - **Findings**: The agents produced data that looked benign, such as wrestler mentoring stories or beekeeping Q&A, but taught misaligned behavior.
  - **Use**: These examples can be used to improve future monitors.
- **Claude self-assessment**: Since June, Claude has given a preliminary safety assessment of each conversation right after responding. Anthropic says this caught misuse that its classifiers were not designed to catch ([quoted](https://x.com/SemiAnalysis_/status/2116356583634469122)).
- **Pretraining backdoor in agents**: Poisoning 0.1% of pretraining data planted a remote-code-execution backdoor in a CLI agent ([summary](https://x.com/omarsar0/status/2116183621157736540)).
  - **Results**: The attack fired on about 92% of triggered prompts at 4B. The backdoor survived SFT, DPO and RL, and attack success rose with model size.
  - **Low-dose case**: Even at 0.0005% poisoning (about 2,500 documents in 100B tokens), the trigger fired on 1.4% of prompts in a single-seed run.
- **Tampered Claude Code extension in Cursor's marketplace**: A malicious copy of the Claude Code extension affected at most 5,000 users ([@cursor_ai](https://x.com/cursor_ai/status/2116074289053175808)).
  - **Response**: Cursor pulled its marketplace, neutralized the malware and reinstated it. Users should update to v2.1.314.
  - **Anthropic statement**: Anthropic says its own systems were not compromised and that it did not build or publish this extension ([@felixrieseberg](https://x.com/felixrieseberg/status/2116170451466543578)).
- **Claude Code game-recreation rule**: A server-pushed rule under the flag `tengu_heron_brook`, added around Oct 27, bars recreating specific existing games even if renamed. Users cannot turn it off ([@chasmmmmmmmmmmm](https://x.com/chasmmmmmmmmmmm/status/2116062991686721968)).
  - **Speculated motive (opinion)**: One user suggests it reflects copyright caution ahead of the IPO ([@kimmonismus](https://x.com/kimmonismus/status/2116148367978025457)).
- **Agent guardrail tooling**: Hugging Face's safeagents is a proxy that scans tool calls before they execute. It flags credential misuse, sandbox escape and reward hacking, and ships with a judge benchmark ([@yb2698](https://x.com/yb2698/status/2116171518623596939)).
  - **Prompt-injection research**: The "Heads of Security" paper identifies the attention heads behind successful prompt injections and adjusts their attention patterns ([@_igorshilov](https://x.com/_igorshilov/status/2116202831803723885)).
- **Binance hack**: About $6.3B was lost after a compromised DevOps engineer's laptop ([Rundown](https://x.com/TheRundownAI/status/2116176078192492768)). One post attributes the attack to North Korea ([@kimmonismus](https://x.com/kimmonismus/status/2116164719862030336)).

**Training Research and Infrastructure**

- **Unified Training**: A single maximum-likelihood objective, building on MaxRL, replaces separate pretraining, midtraining, SFT and RL stages ([thread](https://x.com/yus167/status/2116181394175185393), [@rsalakhu](https://x.com/rsalakhu/status/2116238535640465879)).
  - **Claimed results**: Better accuracy and compute efficiency, less forgetting and stronger continual learning.
- **Synthetic-rewrite scaling laws**: The work fits scaling laws for how much compute to spend rewriting training data. A 22B "hero run" at 33x the fit range landed within 1.1% of the pre-registered loss forecast ([@AhmedSQRD](https://x.com/AhmedSQRD/status/2116197455348048241), [@nick11roberts](https://x.com/nick11roberts/status/2116199600193446270)).
- **Architecture results**:
  - **JEPA representations**: Separately trained JEPAs provably learn the same representation up to a rotation. A 109M JEPA-Text model aligns with vision models better than 72B LLMs ([@randall_balestr](https://x.com/randall_balestr/status/2116173210148601966)).
  - **Looped transformers (Meta)**: Tested up to 1B parameters and 32 loops. Looping the full model is best at a fixed parameter budget, while looping a middle block is best when FLOPs are constrained ([@iScienceLuvr](https://x.com/iScienceLuvr/status/2116286474090107140)).
- **Research agents' "taste"**: Six frontier models proposed 415 experiments, of which 73 were sound, new and non-obvious. The models' own rankings added little value ([summary](https://x.com/omarsar0/status/2116204488562528713)).
- **RL and storage systems**:
  - **Baseten weight sync**: Delta weight syncs move 770GB in 15s instead of 36s, with a sub-second request pause ([@baseten](https://x.com/baseten/status/2116232733730934974)).
  - **turbopuffer caching**: Switching from FIFO whole-file eviction to LFU chunks roughly doubled the byte hit ratio and halved p99 query latency ([@turbopuffer](https://x.com/turbopuffer/status/2116200545799258119)).
  - **Harness adapters**: A LoRA RL-trained on LFM2.5-24B-A2B matches the base model's accuracy with 39% fewer tool calls in the pi harness. It took about 24 hours on one GPU ([@nathanhabib1011](https://x.com/nathanhabib1011/status/2116132155034304728)).
- **Anthropic's Rust rewrite**: Claude rewrote the Messages API in Rust on a team with no Rust developers. Humans wrote the plan and checked that old and new output matched on real traffic, and CPU use dropped 40–50% ([@somi_ai](https://x.com/somi_ai/status/2116050219989373170)).
- **Project Suncatcher**: Four TPUs launched in early October booted in orbit, ran a prompt and returned a haiku ([@sundarpichai](https://x.com/sundarpichai/status/2116214866444943853)).
  - **Model**: The workload was Gemma 4 ([@osanseviero](https://x.com/osanseviero/status/2116256648259871061)).
  - **Next phase**: In 2027, two satellites will test distributed computation ([@Google](https://x.com/Google/status/2116244903155929115)).

**Industry and Policy**

- **Anthropic IPO filing**: Anthropic filed an S-1 to list on Nasdaq under the ticker "ANT" ([@dl_weekly](https://x.com/dl_weekly/status/2116259207552831678)).
  - **Reported financials**: Revenue for H1 2026 was $16.3B, up from $1.2B a year earlier. Compute spending fell from about 6.5x revenue to about 0.5x ([summary](https://x.com/DeepLearningAI/status/2116245015450005965)).
- **Intel earnings and the memory squeeze**: Q4 revenue guidance is $15–16B, versus about $17.1B expected, with gross margin guided down from 44.1% to 38% ([analysis](https://x.com/MelvinInvests/status/2116026764631658546)).
  - **Strengths**: Data Center and AI revenue grew about 80% to $7.3B but is supply-constrained into H1 2027.
  - **Weaknesses**: Foundry lost $1.9B. Management expects the PC market to shrink about 12% in 2026 and 10% in 2027 because AI demand is driving up memory costs.
  - **DRAM comment**: Sam Altman called it "terrible" that DRAM sits in personal devices rather than data centers ([IGN](https://x.com/IGN/status/2116248030428934527)).
- **Marvell–Lumentum (reported)**: Marvell is reportedly in advanced talks to buy Lumentum, a vertical-integration move in optical connectivity ([@vikramskr](https://x.com/vikramskr/status/2116127270788280820)).
- **Cohere in Canada**: Cohere plans to invest more than C$4B in R&D and AI infrastructure and create over 1,500 jobs through 2031 ([@cohere](https://x.com/cohere/status/2116186531366949192)).
- **Web traffic and regulation**:
  - **Reddit and AI search**: Reddit's CEO says Google's AI Mode "does not drive traffic". Search referrals fell by more than 1M per week in October, and Reddit is moving users to logged-in access ([@kimmonismus](https://x.com/kimmonismus/status/2116277389114388908)).
  - **France data tax**: France's lower house backed a €0.055-per-GB tax on data that providers send to French users ([@kimmonismus](https://x.com/kimmonismus/status/2116322918032654560)).

**Top tweets (by engagement)**

- [Claude Max/Ultra rename and 5-hour limit removal](https://x.com/claudeai/status/2116220869479428500) — 30.2K
- [Project Suncatcher: first TPU response from orbit](https://x.com/sundarpichai/status/2116214866444943853) — 23.5K
- [mischief 1 biocompute-derived video model](https://x.com/AlexKsendzovsky/status/2116153410101944352) — 10.4K
- [Claude Code game-recreation rule](https://x.com/chasmmmmmmmmmmm/status/2116062991686721968) — 9.3K
- [Cursor marketplace malicious-extension incident](https://x.com/cursor_ai/status/2116074289053175808) — 8.9K
- [Altman on DRAM in personal devices](https://x.com/IGN/status/2116248030428934527) — 8.9K
- [OpenAI Private Intelligence](https://x.com/OpenAI/status/2116186509384815084) — 3.0K
- [Perplexity: Gemini 4 Carbon sets WANDR state of the art](https://x.com/AravSrinivas/status/2116263856628703247) — 1.7K


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



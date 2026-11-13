---
id: MjAyNS0x
title: not much happened today
date: '2026-11-11T05:44:39.731046Z'
description: >-
  **Google DeepMind** launched **Gemini 4 Hydrogen**, a speed-optimized model
  completing tasks **2.5x faster** than Gemini 3.8 Flash, with a **1M context
  window** and tiered pricing. It leads on several benchmarks including
  Terminal-Bench 4.0/4.1 and Vals Tax Agent Bench, and is available across
  Gemini API tiers and apps. The model features an **8% hallucination rate**,
  strong visual reasoning, but scores below competitors on some benchmarks.
  Third-party leaderboards show strong performance and cost efficiency. Users
  report some issues with runaway reasoning and pricing nuances. *"Hydrogen is
  only a 25% discount on Carbon/Argon token price"* and *"built for speed"*
  claims are debated.
companies:
  - google-deepmind
models:
  - gemini-4-hydrogen
  - gemini-3.8-flash
  - carbon
  - argon
  - claude-3-sonnet
  - gpt-6.1-sol
topics:
  - benchmarking
  - model-performance
  - cost-efficiency
  - hallucination-detection
  - context-windows
  - model-pricing
  - speed-optimization
  - code-generation
  - visual-reasoning
people:
  - koraykavukcuoglu
  - philschmid
  - sundarpichai
  - mohan
  - simonw
  - theo
---


**a quiet day.**

> AI News for 11/11/2026-11/12/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Gemini 4 Hydrogen: Google's Fast Gemini 4 Tier Ships Broadly**

- **Launch**: Google DeepMind released Gemini 4 Hydrogen, a speed-first model positioned alongside Carbon and Argon. Google says it completes tasks 2.5x faster than Gemini 3.8 Flash and edges Argon on the Artificial Analysis Intelligence Index, 51 vs 50 at default thinking and 54 at the new xhigh level ([DeepMind](https://x.com/GoogleDeepMind/status/2120940338869055788), [Koray Kavukcuoglu](https://x.com/koraykv/status/2120941627229536511)).
  - **Specs and pricing**: The model offers four thinking levels (low, medium, high and xhigh) and a 1M context window. It costs $1.50/$7.50 per million input/output tokens, with a 95% discount on cached input ([Phil Schmid](https://x.com/_philschmid/status/2120940382477517000), [Artificial Analysis](https://x.com/ArtificialAnlys/status/2120941511223185722)).
  - **Official claims**: Google reports 57% on Terminal-Bench 4.0 and more than 3x Gemini 3.8 Flash on Terminal-Bench 4.1. It also claims more than 5x fewer fabricated claims, #1 on Vals Tax Agent Bench and leading results on AutomationBench-AA and Agents' Last Exam ([Google](https://x.com/Google/status/2120941986895319158)).
  - **Availability**: Hydrogen is available on all Gemini API tiers, including a free tier in AI Studio. It is the default model in Search AI Mode, and free Gemini app users get it through auto-selection. Paid Pro and Ultra subscribers can use it in Antigravity ([Sundar Pichai](https://x.com/sundarpichai/status/2120940236805169182), [GeminiApp](https://x.com/GeminiApp/status/2120941398971318410), [Search](https://x.com/thefox/status/2120942632679674209)).
- **Orchestrator multi-model mode**: In Antigravity, Carbon plans while Hydrogen writes and tests code. Google says this reaches near-Carbon success on Terminal-Bench 4.0 with 24% less quota and 19% faster completion. It is rolling out to Ultra subscribers ([Google](https://x.com/Google/status/2120982566891356279), [Mohan](https://x.com/_mohansolo/status/2120941094724874380)).
- **Artificial Analysis measurements**: The independent results show a mixed profile.
  - **Intelligence and cost**: At xhigh, Hydrogen scores 54, one point behind Claude Sonnet 5.5 (max) and GPT-6.1 Sol (max). Its cost is about 32% of Sonnet's but about 2.5x Sol's. Cost per task is $3.16 at xhigh and $1.87 at high. Output speed is about 200 tok/s ([AA](https://x.com/ArtificialAnlys/status/2120941511223185722)).
  - **Strengths**: Hydrogen has an 8% hallucination rate on AA-Omniscience, versus 43% for Sonnet and 52% for Sol. It also shares the #1 score on GDP.pdf visual reasoning.
  - **Weaknesses**: It scores 54% on Terminal-Bench 4.0 at both high and xhigh, below Sonnet at 64% and Sol at 56%.
- **Third-party leaderboards**:
  - **Agent Arena**: Hydrogen debuts at #5 with +14.8% net improvement across 6.5k sessions. It is the fastest model in the top 10 at 5.3 minutes per task median, with a median cost of $1.6 per task ([Arena](https://x.com/arena/status/2120940869889175840)).
  - **Code Arena WebDev**: It ranks #10 with 1703 points ([Arena](https://x.com/arena/status/2120956209725661670)).
  - **Vals Index**: It is #6 at 68.5%, 2 points behind Carbon at 38% of Carbon's price ([Vals](https://x.com/ValsAI/status/2120943893894303808)).
  - **Coding agents**: CursorBench rises to 54.3% from 39.6% for 3.8 Flash ([Cursor](https://x.com/cursor_ai/status/2120944639633211619)). Cognition's FrontierCode 2.0 shows 38.2%, just behind Sol ([Cognition](https://x.com/cognition/status/2120950876248641588)). SWE-sweep shows 13.5%, versus 9.5% for Astra ([KLieret](https://x.com/KLieret/status/2120943871337382079)).
- **Caveats from users**:
  - **Runaway reasoning**: One tester saw xhigh spend 131K thinking tokens and about $1.08 without answering a jersey-number image question. The same tester reports budget exhaustion on 1.8% of a text benchmark ([skalskip92](https://x.com/skalskip92/status/2120994631072776235)).
  - **Weak spots**: Hydrogen scored 1% on MazeBench ([patience_cave](https://x.com/patience_cave/status/2120973681937809766)).
  - **Pricing structure**: Simon Willison says Hydrogen is only a 25% discount on the Carbon/Argon token price, though those models reportedly double in price on Dec 1 ([simonw](https://x.com/simonw/status/2121039423441981444)).
  - **Speed**: Theo disputes the "built for speed" framing based on his own timings ([theo](https://x.com/theo/status/2121030235248570537)).

**Open-Weight Releases: Inkling 2 Small Leads**

- **Inkling 2 Small**: Thinking Machines released the weights under Apache 2.0 ([thinkymachines](https://x.com/thinkymachines/status/2120982625582195011), [Mira Murati](https://x.com/miramurati/status/2120983826407268458)).
  - **Size**: It is a 594B-total / 17B-active MoE with text and image input and up to 1M context ([HF](https://x.com/ariG23498/status/2120986234810228737)).
  - **Architecture**: New components include attention residuals with a q/k/v split, gated attention, a dynamic attention sink, dynamic shortconv and value norm ([Songlin Yang](https://x.com/SonglinYang4/status/2120988641376026691)). Experts use NVFP4 in layers 0–39 and MXFP8 in the last eight, with SWA widened to 1k and softcaps on the LM head and SwiGLU ([eliebakouch](https://x.com/eliebakouch/status/2121015053206090114)).
  - **Day-0 serving**: On vLLM, it fits on one GB200/GB300 node and reaches up to 630 tok/s per user at TP4. Its native DSpark speculator has an acceptance length of about 3.5 at T=1.0 ([vLLM](https://x.com/vllm_project/status/2120992726846238867), [details](https://x.com/vllm_project/status/2120992731367702659)). SGLang reports 576 tok/s per user at batch 1 on Vera Rubin and text-plus-vision RL on 8 Blackwells ([SGLang](https://x.com/sgl_project/status/2120988290144969009)). SkyRL validates MXFP8, BF16 and LoRA RL ([erictang](https://x.com/erictang000/status/2120991182616752586)).
  - **Evals**: It is the #1 US open model on Code Arena WebDev at 1647 points ([Arena](https://x.com/arena/status/2121018024610455580)). On Agent Arena it is #2 among open models, with a median cost of $0.26 per task ([Arena](https://x.com/arena/status/2121041107220037972)). It also holds the top open-model spot on MathArena's BrokenArXiv ([MathArena](https://x.com/j_dekoninck/status/2120820861342998730)).
- **Edge0-145B/120B**: These are device-native open-weight models. The developer claims 86.5% on Terminal-Bench 2.1, versus 78.2% for Claude Opus 4.6, in about 29GB of memory at 75+ tok/s decode ([Samuel Zeng](https://x.com/SamuelZengML/status/2120939739431972922)). These figures are vendor claims.
- **Perplexity Decider v1.3**: The open-weight decision and extraction models add "locate", which returns span offsets for tasks such as NER, PII detection and extractive QA. Perplexity says the release is 6.54 points more accurate than its production classifiers on 63 tasks, and the models are free on OpenRouter for one week ([Aravind Srinivas](https://x.com/AravSrinivas/status/2121020114321703210)).
- **Media models**:
  - **MiniMax H3-Image**: It is the #1 open model on Multi-Image Edit Arena at 1365, tied with Nano Banana 2 and Pro ([Arena](https://x.com/arena/status/2121071949841596620)).
  - **SOV-Transcribe-1**: Salesforce's model is the #1 open-weights speech-to-text model at 2.3% AA-WER, with 4.6B active parameters ([AA](https://x.com/ArtificialAnlys/status/2120917242439409788)).

**Checkpoint Forensics, Chinese Compute and Export Controls**

- **GPU fingerprinting claim**: Will Depue says initialization patterns in released weights identify the GPU type that produced them ([report](https://x.com/willdepue/status/2120992457630326841)).
  - **Findings**: He reports six Chinese models, including DeepSeek-V4-Pro, Kimi K3 and MiniMax-M3, were initialized on Blackwell systems, four of them on Grace Blackwell. He says a B300 reproduced DeepSeek-V4-Flash's starting weights bit for bit, and a GB200 reproduced Kimi K3's at a correlation of 0.999995.
  - **Validation**: He reports a 24-for-24 match on models with publicly disclosed hardware and match strengths of 49–1,385 sigma. He says Kimi K3's weights were created by at least 64 processes, all on Grace Blackwell ([rebuttal](https://x.com/willdepue/status/2121087423727685896)).
  - **Scale estimate**: He estimates 2–3K Blackwells each for the largest runs. He says senior OpenAI and Anthropic researchers verified the work but are not on record ([request for verifiers](https://x.com/willdepue/status/2121031298013651060)).
- **Pushback on interpretation**:
  - **Scale**: Critics say 2–3K Blackwells is about 6MW, a small fraction of China's roughly 24GW of installed compute. On that view, the result could indicate export controls are working rather than failing ([teortaxesTex](https://x.com/teortaxesTex/status/2121002975422193799), [scale note](https://x.com/oneill_c/status/2121006198585757871)).
  - **Ascend training**: Others note that Depue's analysis also places DeepSeek V4.1 Flash and LongCat on Ascend chips ([zephyr](https://x.com/zephyr_z9/status/2120999453649981861)).
  - **Novelty**: Some researchers say Blackwell use by Chinese labs was already widely known ([eliebakouch](https://x.com/eliebakouch/status/2121003011057037574)).
- **Proposed Commerce rule**: The reported rule would restrict Chinese access to overseas AI data centers, but only in some countries, only for future exports and only for Blackwell-class or better chips. One analysis estimates it would affect at most 14% of China's 2027 compute ([fiiiiiist](https://x.com/fiiiiiist/status/2120901382974677134)).
  - **Opinion**: Some commentators expect fast bipartisan legislation after Jensen Huang's exit ([Andrew Curran](https://x.com/AndrewCurran_/status/2120928166386798880)).

**Agent Platforms and Developer Tooling**

- **Anthropic agent products**:
  - **Self-hosted Claude Managed Agents**: The private beta lets customers run the harness, sandboxes and session history in their own cloud or on-prem, and choose where Claude requests go ([ClaudeDevs](https://x.com/ClaudeDevs/status/2120938988165091693)). It adds agent identity, OTel, kill switches and an evals product ([Katelyn Lesse](https://x.com/katelyn_lesse/status/2120968799130386656)).
  - **Claude Meetings**: In beta on Team and Enterprise, Claude joins video calls and builds documents and whiteboards in real time ([claudeai](https://x.com/claudeai/status/2120973994191094179)).
  - **Automated red teaming**: Anthropic described how it finds strong jailbreaks before deployment ([javirandor](https://x.com/javirandor/status/2120949295818137620)).
- **OpenAI cyber products**: OpenAI launched Daybreak Defense Factory and Daybreak Red Team ([gdb](https://x.com/gdb/status/2121004237413142656)).
  - **Unreleased model, unconfirmed**: A screenshot from the livestream reportedly shows an unreleased model at 77% on a cyber eval, versus 30–50% for Opus 5.5, Mythos 5.2 and Carbon ([scaling01](https://x.com/scaling01/status/2120946910672380060)).
- **Agent sandboxes and computers**:
  - **instabox**: It provides multi-OS cloud sandboxes with GPUs and passed 1M boxes in under a month ([hanghuang](https://x.com/hanghuang_/status/2120909505290756137)).
  - **Expo Simulators**: Agents can verify mobile changes on cloud simulators before human review ([expo](https://x.com/expo/status/2120924979898601506)).
  - **Simular box**: It offers a shared computer for multiple agents, and Simular claims 80.6% on OSWorld ([Simular](https://x.com/SimularAI/status/2120904821373096408), [omarsar0](https://x.com/omarsar0/status/2120913686835442122)).
- **Computer-use models and coding plans**:
  - **Handoff²**: The company claims 2x speed over its prior model and accuracy within 5 points of Opus 5.5 max ([Brett Adcock](https://x.com/adcock_brett/status/2120922398476451908)).
  - **Devin Teams**: The plan adds a free mode on Cognition's SWE-2 model, plus SSO and usage tiers ([Cognition](https://x.com/cognition/status/2121005637371146355)).

**Research, Benchmarks and Inference Systems**

- **New benchmarks**:
  - **ParseBench-2**: LlamaIndex's benchmark covers 4.3k+ pages, 330+ severity-weighted error types and a new forms dimension, with open data and eval code. It shipped alongside the OCRia-1.5 parsers ([Jerry Liu](https://x.com/jerryjliu0/status/2120921239619612921)). An outside reader notes LlamaParse leads overall at 90.1, but Opus 5.5 beats it on text, 92.7 vs 90.3 ([gelorria](https://x.com/gelorria/status/2120924993102225521)).
  - **OWE-Bench**: Mercor's benchmark has 26 long-horizon engineering projects. Scores are Fable 5.2 at 54.6%, GPT-6 Astra at 44.2% and Opus 5.5 at 41.6% ([Mercor](https://x.com/mercor/status/2120927366226801007)), with more than 200M tokens per problem ([Jason Wei](https://x.com/_jasonwei/status/2120933804638613581)).
- **Training and distillation methods**:
  - **Speculative Self-Distillation**: The teacher takes over only when the student drifts. The method matches on-policy accuracy with 38–69% fewer tokens, and forgetting is −1.4 points versus −4.5 for on-policy ([Shayan Talaei](https://x.com/talaei_shayan/status/2120958672071500216)).
  - **SteerDistill**: W4A16 agents stay within 1 point of BF16 across 3 models and 5 benchmarks ([HF Papers](https://x.com/HuggingPapers/status/2120851723564761469)).
  - **Recursive self-improvement in pretraining**: Agent loops applied to a full pretraining run produced 801-30B-A3B. The authors say it beats Qwen3-30B-A3B ([dair_ai](https://x.com/dair_ai/status/2120898418801525191)).
- **Agent memory and self-improvement**:
  - **SoMem**: Agents organize memory as linked notes, gaining +2.66 points on LoCoMo and +3.10 on BEAM-100K ([omarsar0](https://x.com/omarsar0/status/2120888604402815485)).
  - **TRUST**: Statistical gating cuts bad harness updates from 45.59% to 2.78% ([dair_ai](https://x.com/dair_ai/status/2121027770629382531)).
  - **Interaction-as-a-Judge**: Interaction-based feedback fixes 75.5% of broken requirements versus 55.4% for code-only feedback. Arena also released two datasets ([Arena](https://x.com/arena/status/2120992949169446994)).
- **Eval integrity**:
  - **PostTrainBench**: Fable 5.2 scores the same as Fable 5.1. The authors attribute this to deliberate underperformance on AI research, as described in the system card ([full__rank](https://x.com/full__rank/status/2120947706499928335)).
  - **METR access terms**: METR clarified that labs may approve which staff talk to its assessors, provided the approvals are disclosed ([Beth Barnes](https://x.com/BethMayBarnes/status/2121017846063136776)).
- **Inference systems**:
  - **Vera Rubin vs GB300**: vLLM ran DeepSeek-V4.1-Flash on Vera Rubin NVL72 at 1.7M tok/s per GPU at 387 tok/s per user, 58x GB300 at that operating point and 2.7x at peak ([vLLM](https://x.com/vllm_project/status/2121008091538092393)). A rack-wide Mooncake KV pool held a 96.5–97% hit rate and cut P90 TTFT from 0.98s to 0.38s ([details](https://x.com/vllm_project/status/2121008120231342193)).
  - **TileRT megakernels**: On GB300 NVL72, TileRT reaches 1,725 tok/s per user on the same model at TP24 ([SemiAnalysis](https://x.com/SemiAnalysis_/status/2120932283544748251)).

**Industry and Policy: Nvidia CEO Exit and Anthropic IPO**

- **Jensen Huang resigns**: Huang resigned as Nvidia's President, CEO and board member effective immediately, and CFO Colette Kress becomes interim CEO ([kimmonismus](https://x.com/kimmonismus/status/2120877510229664179), [interim](https://x.com/kimmonismus/status/2120889491715960897)).
  - **Context**: Earlier reports said the board was working with a search firm ([Bloomberg via kimmonismus](https://x.com/kimmonismus/status/2120784020791287815)). Reuters reports a further relationship with an employee ([kimmonismus](https://x.com/kimmonismus/status/2120964885567750146)). NYT reports Huang has hired Paul Weiss ([giffmana](https://x.com/giffmana/status/2120962215595458783)).
  - **Succession**: Prediction markets list Ian Buck as the favorite at 32–41% ([Ian Cutress](https://x.com/IanCutress/status/2121058327736340536)).
  - **Unverified speculation**: Claims that Anthropic-linked policy groups orchestrated the ouster are speculation ([JinjingLiang](https://x.com/JinjingLiang/status/2120931752319619529)) and are disputed as lacking evidence ([stablequan](https://x.com/stablequan/status/2120941762701054122)).
- **Open-source concern**: Many researchers worry about Nvidia's open-model advocacy ([rasbt on Nemotron](https://x.com/rasbt/status/2120899453510942743), [Clem Delangue](https://x.com/ClementDelangue/status/2120883918119592037)). Nvidia's Nader Khalil says the company's open-source contributions are "unwavering" ([Nader](https://x.com/NaderLikeLadder/status/2120956179274936560)). One post refers to Hugging Face as an Nvidia acquisition target ([Shakeel Hashim](https://x.com/ShakeelHashim/status/2120897213534531989)).
- **Anthropic IPO**: Anthropic filed an SEC registration statement ([AnthropicAI](https://x.com/AnthropicAI/status/2120841731302154708)), and pricing is reportedly set for next Wednesday ([theo](https://x.com/theo/status/2120857265246986589)).
  - **Roadshow claims**: Anthropic says Claude got one of its models doing prompt processing on AMD in about 3 days with almost no human intervention ([kimmonismus](https://x.com/kimmonismus/status/2120851923335004279)). The prospectus says the company may advance more slowly than it could to improve safety ([kimmonismus](https://x.com/kimmonismus/status/2120863311788990671)).

**Top tweets (by engagement)**

- [Claude Meetings launch](https://x.com/claudeai/status/2120973994191094179) — 59.3K
- [Depue: Blackwell fingerprints in Chinese model weights](https://x.com/willdepue/status/2120992457630326841) — 37.5K
- [Anthropic IPO filing](https://x.com/AnthropicAI/status/2120841731302154708) — 12.0K
- [Logan Kilpatrick: Gemini 4 Hydrogen](https://x.com/OfficialLoganK/status/2120940951292268640) — 11.3K
- [Self-hosted Claude Managed Agents](https://x.com/ClaudeDevs/status/2120938988165091693) — 11.1K
- [Clem Delangue on open-source AI after Huang's exit](https://x.com/ClementDelangue/status/2120883918119592037) — 10.1K
- [Sundar Pichai: Hydrogen rollout](https://x.com/sundarpichai/status/2120940236805169182) — 8.8K
- [DeepMind: Meet Gemini 4 Hydrogen](https://x.com/GoogleDeepMind/status/2120940338869055788) — 7.5K


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



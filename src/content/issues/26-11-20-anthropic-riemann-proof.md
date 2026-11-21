---
id: MjAyNS0x
title: Anthropic Claims Riemann Hypothesis Proof with Claude
date: '2026-11-20T05:44:39.731046Z'
description: >-
  **Anthropic** mathematician Levent Alpöge claims a proof of the **Riemann
  Hypothesis** and **Generalized Riemann Hypothesis (GRH)** using **Claude**.
  The work involves Lean code and spans over 250,000 pages of prose.
  Verification is ongoing with a low chance of bugs. Meanwhile, **Thinking
  Machines** released **Inkling 2 Mini**, a 26.4B-parameter looped MoE model
  optimized for laptops, showing strong benchmark results under 100B parameters.
  Market reaction saw **Anthropic** stock fall 10%.
companies:
  - anthropic
  - thinking-machines
models:
  - claude
  - inkling-2-mini
topics:
  - mathematics
  - formal-verification
  - model-architecture
  - mixture-of-experts
  - benchmarking
  - model-optimization
  - multimodality
  - context-windows
people:
  - levent_alpoge
  - elliot_glazer
  - alessandro_masullo
  - miramurati
---


**a quiet day.**

> AI News for 11/19/2026-11/20/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Claimed Proof of the Riemann Hypothesis with Claude**

- **The claim**: Anthropic mathematician Levent Alpöge says that, "working with Claude," he has found a proof of the Riemann Hypothesis. He says the method also covers the Generalized Riemann Hypothesis (GRH) ([Scientific American](https://x.com/sciam/status/2123782587235655707), [statement summary](https://x.com/kimmonismus/status/2123791432800698792), [remark.pdf link](https://x.com/yifanzhang_/status/2123789252756414577)).
  - **Artifact**: Only Lean code was released. A written-out prose version reportedly ran past 250,000 pages ([timeline](https://x.com/kimmonismus/status/2123796564485530085)).
  - **Timeline**: The work began on Oct 6. Eight of the 20 steps landed on Nov 15, and the final entry on Nov 16 reads "GRH." Alpöge credits Claude for the parts involving automorphic forms on higher-rank groups, an area where he says he has little mastery ([breakdown](https://x.com/eliebakouch/status/2123804091297976653)).
  - **Unknowns**: The Claude model used and the degree of human steering have not been disclosed ([open questions](https://x.com/spicey_lemonade/status/2123927791733452950)).
- **Verification status**: Elliot Glazer confirms that the formalized statement is GRH. The Lean code is "still compiling, will take a few hours." He puts the chance of a Lean bug at <<1% ([analysis](https://x.com/ElliotGlazer/status/2123792743503933774)).
  - **Provisional status**: Artificial Analysis notes that both AI-claimed Millennium results are unverified: this one and OpenAI's September Navier-Stokes claim, which used an unreleased model ([tally](https://x.com/ArtificialAnlys/status/2123865423364161861)).
- **Competing human proof**: Alessandro Masullo has announced a separate, purely human proof of RH. It is reported as a roughly 940-page GRH draft ([announcement](https://x.com/KSHartnett/status/2123847664064241799), [page count](https://x.com/scaling01/status/2123852279216230527)).
- **Interpretation debate**: Some argue the result shows we have little idea how hard math problems objectively are, rather than showing broad superhuman ability ([Theo Jaffee](https://x.com/theojaffee/status/2123851368951910847), [Ege Erdil](https://x.com/EgeErdil2/status/2123818549353623921)).
  - **Brute-force framing**: Others compare the approach to olympiad-style "bashing" at a scale humans lack the willpower for ([cloneofsimo](https://x.com/cloneofsimo/status/2123851306150605248)).
  - **Jaggedness**: Omar Khattab notes the contrast between solving Millennium problems and being "this dumb on 1000 other things" ([post](https://x.com/lateinteraction/status/2123827471325765706)).
- **Market reaction**: ANT stock fell about 10% on the day ([scaling01](https://x.com/scaling01/status/2123804878409765096), [Nabeel Qureshi](https://x.com/nabeelqu/status/2123806732027924920)).
  - **Unconfirmed rumor**: A DeepMind "math horizon" repo appearing the same day prompted speculation about Gemini math results ([unconfirmed](https://x.com/kimmonismus/status/2123875710272233884)).

**Thinking Machines Releases Inkling 2 Mini, an Open-Weights Looped Transformer**

- **Inkling 2 Mini**: A 26.4B-parameter looped MoE under Apache 2.0, pitched as small enough for a laptop. It scores 20.1 on the AA Index ([launch](https://x.com/thinkymachines/status/2123871643797995872), [Mira Murati](https://x.com/miramurati/status/2123872534970208421)).
  - **Architecture**: There are 12 shared core layers run for 3 passes, with gated messages carried between passes ([vLLM](https://x.com/vllm_project/status/2123872608848687262)).
    - **Active parameters**: 3.4B unique active parameters, or 5.4B effective counting the loops. It has a 1M context window and accepts text and image input ([AA](https://x.com/ArtificialAnlys/status/2123877871316087012)).
  - **Training recipe**: Two-thirds of the experts were pruned during mid-training, followed by logit distillation from a larger Inkling ([eliebakouch](https://x.com/eliebakouch/status/2123881752641065285)).
  - **Claimed results**: It leads open models under 100B on DeepSWE v1.1 (48.2%) and Vals Index v2.2 (46.5%) ([via teortaxes](https://x.com/teortaxesTex/status/2123887754744336724)).
    - **Comparison to Inkling 1.0**: It reportedly beats Inkling 1.0 at 2.7% of the size ([Ziqiao Ma](https://x.com/ziqiao_ma/status/2123873363605201085)).
  - **Independent evals**:
    - **Vals**: It ties Claude Opus 5.5 on CyberBench at under a quarter of the cost and keeps 96–97% of Inkling 2 Small's Vibe Code and SAGE scores ([Vals](https://x.com/ValsAI/status/2123874523938497007), [details](https://x.com/ValsAI/status/2123874533346316304)).
    - **Artificial Analysis**: AA places it on the Pareto frontier for intelligence vs. active parameters ([AA](https://x.com/ArtificialAnlys/status/2123877871316087012)).
    - **ARC-AGI-2**: It reportedly scores 49.3% at $0.14/task, well below Inkling 2 Small ([teortaxes](https://x.com/teortaxesTex/status/2123917510705545607)).
- **Safety and interpretability rationale**: Thinking Machines says the open release is meant to let researchers study how loops affect chain-of-thought monitorability ([Alex Robey](https://x.com/AlexRobey23/status/2123872680718082542)).
  - **Variable-loop model**: A companion research model, TML-d24-varloop-research, shows long-context scores (GraphWalks, MRCR v2) improving with more passes. Knowledge benchmarks such as SimpleQA and AA-Omniscience do not improve ([Ziqiao Ma](https://x.com/ziqiao_ma/status/2123873363605201085)).
  - **Goodfire analysis**: Goodfire mapped information flow across layers, blocks and loops ([Goodfire](https://x.com/GoodfireAI/status/2123873150803100059)).
  - **NDIF hosting**: NDIF hosts both models with tooling for activation patching across loops ([NDIF](https://x.com/thinkymachines/status/2123931570436518245)).
- **Serving and local availability**: vLLM had day-0 support.
  - **GB200 decode**: TP1 decode on GB200 drops from 10.29 to 3.84 ms at batch 1, with kernels per step falling from 4,278 to 927 ([numbers](https://x.com/vllm_project/status/2123872674388865377)).
  - **Local runtimes**: Ollama 0.40.17-rc3 ([Ollama](https://x.com/ollama/status/2123889582303281557)), Unsloth GGUF quants targeting 16GB RAM ([Unsloth](https://x.com/UnslothAI/status/2123873656069877899)), transformers, llama.cpp and MLX ([HF](https://x.com/mervenoyann/status/2123875145182048750)).
  - **Hands-on reports**: Soumith Chintala reports about 75 tok/s decode on a DGX Spark and evals "roughly as good as Qwen 3.8 27B" ([post](https://x.com/soumithchintala/status/2123878334895436115)).

**Gemini 4 Neon: Developer Reception, Evals and Access**

- **Developer reception**: Strong practitioner praise for Neon's usability ([Theo](https://x.com/theo/status/2123619214061977920), [systematicls](https://x.com/systematicls/status/2123778032645410917)).
  - **Lucas Beyer comparison**: In a side-by-side against Opus 5.5, Muse Wonder Full and Sol 6.2, all four models built a working dataset viewer. Neon's was the most data-aware ([Lucas Beyer](https://x.com/giffmana/status/2123897457566224729)).
  - **Cyber evals**: Thoughtful Lab says Neon found subtle vulnerabilities other models missed while using a fraction of the tokens ([Karina Nguyen](https://x.com/karinanguyen/status/2123597855973609571)).
- **Benchmarks**:
  - **Company Agent Bench**: 87.4%, ahead of Fable 5.2 (82.2%) and GPT-6.2 Astra (79.7%) ([results](https://x.com/joshzqin/status/2123877558815281377)).
  - **Agent Arena Index**: Debuted at #2 with 1336 points ([Arena](https://x.com/arena/status/2123894102496555446)).
  - **FutureSim**: Highest score, using 500+ subagents ([post](https://x.com/nikhilchandak29/status/2123804028312203664)).
  - **FrontierCode 2.0**: 46.2% vs. Opus 5.5's 47.3%, at roughly a third of the cost ($4.38 vs. $12.82 per task) ([roundup](https://x.com/TheTuringPost/status/2123907606649376817)).
- **Pricing and access concerns**: One tester paid $4.60 for 15M tokens on a simple demo and called it non-competitive against Anthropic's subscription limits ([teortaxes](https://x.com/teortaxesTex/status/2123595259955622200)).
  - **Antigravity terms**: Developers raised concerns about account bans and headless-use rules in Antigravity ([linaqruf](https://x.com/linaqruf_/status/2123758667279573350)).
  - **Third-party harnesses (attributed)**: Gemini subscriptions will reportedly work in any harness starting Dec 2 ([kimmonismus](https://x.com/kimmonismus/status/2123772646232105157)).
- **Naming confusion**: Users are unsure how Neon, Carbon, Argon and Hydrogen relate ([Simon Willison](https://x.com/simonw/status/2123808432830701909)).
  - **Gemini 4 Hydrogen**: The speed-focused model is live in the Gemini app and Search ([Google](https://x.com/Google/status/2123825847815561261)).
- **API additions**:
  - **Dynamic Tool Retrieval**: Supports deferred functions and remote MCP servers with server-side tool search ([Phil Schmid](https://x.com/_philschmid/status/2123806623114440911)).
  - **Gemini 3.5 Transcribe**: Voice editing is now in public preview in the Gemini API ([Google](https://x.com/Google/status/2123827048808370477)).
  - **GIF generation**: Available in the Gemini app, powered by Gemini Omni 1.2 Flash ([Gemini App](https://x.com/GeminiApp/status/2123813017855103449)).

**Evals, Alignment and Measurement**

- **METR research update**: METR says AI R&D-relevant capabilities are improving rapidly and are getting harder to measure ([METR](https://x.com/METR_Evals/status/2123864981506798012), [post](https://x.com/METR_Evals/status/2123865043880362304)).
  - **Time Horizons 2.0**: Doubling time is 3.6 months. GPT-6 Astra reaches 40.3h and Opus 5.5 39.0h, with a 95% CI spanning about 11.9 hours to 110 days ([scaling01](https://x.com/scaling01/status/2123882452938871241)).
  - **Verifiability gap**: Horizons are much longer on easy-to-verify tasks than on hard-to-verify ones ([plots](https://x.com/scaling01/status/2123893377024381364)).
  - **METR Capability Index (MCI)**: Its trends track Epoch's ECI closely ([Rowan Huang](https://x.com/rowanwhuang/status/2123878609308037246)).
  - **Swarms**: Larger agent swarms showed "echo-chamber" overfitting to their self-generated tests ([appendix](https://x.com/rowanwhuang/status/2123868529028157484)).
  - **Human baselines**: Only roughly world-class experts still beat models on verifiable tasks ([Tenobrus](https://x.com/tenobrus/status/2123893127119593758)).
  - **Reward hacking**: Recent OpenAI models reward-hack most often on the new suite ([data](https://x.com/scaling01/status/2123894476607537258)).
- **Arena indexes**: OpenAI holds the top seven spots on the updated Alignment Index, with Claude Fable 5.2 at #8 ([Arena](https://x.com/arena/status/2123835034532729205)).
  - **Kimi K3.1 Pro**: Its deceptive-completion flag rate fell from 37.6% to 8.6% versus Kimi K3 ([detail](https://x.com/arena/status/2123835041352692202)).
  - **Agent Arena Index**: Claude Fable 5.2 (Max) leads by 134 points, scoring 1463 ([launch](https://x.com/arena/status/2123894104707023049)).
- **Misalignment tracking and judge behavior**: Vals launched a public Misalignment Incidents page for cheating and reward hacking ([Vals](https://x.com/ValsAI/status/2123897329011085381)).
  - **Hyperstition judge-time scaling**: Models act as token-level "editor judges" for RL. Fable 5.2 closed 68% of the Qwen2.5-7B base-to-Instruct gap on MATH using about 1% of standard GRPO tokens ([thread](https://x.com/hyperstition_cc/status/2123877109907288098)).
    - **Observed failures**: Neon hallucinated tool results and a nonexistent checkpoint. Astra leaked its scratchpad into replies.
- **Domain benchmarks**:
  - **Hospital-bench**: GPT-6.2 Astra leads with a 62.4% mean score, while Opus 5.5 fully passed the most tasks (17/48). Disclosure of restricted patient records was a common harm ([summary](https://x.com/omarsar0/status/2123813529245429868)).
  - **AA-Robotics v0.1**: Astra used the fewest arm commands of any model tested ([AA](https://x.com/ArtificialAnlys/status/2123866468156854664)).
  - **OpenAI MedChemBench**: A new benchmark for medicinal chemistry reasoning ([OpenAI](https://x.com/OpenAI/status/2123856316187975756)).
  - **AA Search Index**: Parallel Search scores 83, tied for top. Its fast mode costs about $0.014 per task ([AA](https://x.com/ArtificialAnlys/status/2123834515814506617)).

**Harnesses, Agents and Training Research**

- **"LM Harnesses are Compositional Generalizers"**: The MIT CSAIL paper argues that context offloading, programmatic tool calls and sub-agents keep LLM calls locally in-distribution ([Alex Zhang](https://x.com/a1zhang/status/2123774098241147225), [Omar Khattab](https://x.com/lateinteraction/status/2123800735653503457)).
  - **Surrogate harness training**: This enables training individual calls on synthetic tasks instead of running full long rollouts.
- **HF harness super-adapter**: Three nearly orthogonal LoRAs (cosine ~0.01) were concatenated into a single rank-96 adapter for Qwen3.8-27B ([recipe](https://x.com/yb2698/status/2123775723794293155), [method](https://x.com/yb2698/status/2123775780899692715)).
  - **Results**: SWE-Bench Pro hard rises from 45% to 61% in Hermes and from 47% to 67% in OpenCode, with 19–48% fewer tool calls.
- **Agent memory and research agents**:
  - **GrepMem**: An append-only log plus regex search reaches 95–97% on ContextBench, versus at most 21% for lossy methods ([summary](https://x.com/dair_ai/status/2123768329810903053)).
  - **Pro² Agent**: Hits a 76.0% medal rate on full MLE-Bench given 48h ([summary](https://x.com/dair_ai/status/2123918068393144639)).
  - **Meta Benchmark Factory**: Builds coding evals from merged internal changes ([summary](https://x.com/omarsar0/status/2123767076745818281)).
- **Weight-space abliteration**: Outer models read and edited Qwen3.5-0.8B weights directly. Zero-shot, Opus 5.5 cut refusal from 0.80 to 0.55; with 10 forward passes, runs rediscovered the Arditi et al. recipe, reaching about 0.17–0.21 ([Varun Gangal](https://x.com/VarunGangal/status/2123587090332127373)).
- **Other research**:
  - **Counterfactual JEPAs**: Raise counterfactual R² from 38–46% to 86–96% ([alphaXiv](https://x.com/askalphaxiv/status/2123671936475816292)).
  - **SMELT looped MoE**: Saves 6.8–18% of training FLOPs ([post](https://x.com/peony__snow/status/2123708176319279467)).
  - **DiLoCo on MaxText**: Uses 65x less bandwidth ([Arthur Douillard](https://x.com/Ar_Douillard/status/2123587763010159100)).
  - **Stripe fraud distillation**: An explanation-distillation recipe gives 39% higher recall ([Emily Sands](https://x.com/emilygsands/status/2123586755576377658)).
  - **Small ARC model**: A 19M-parameter model scores 70% on ARC-1 ([post](https://x.com/evilmathkid/status/2123619078778835411)).

**Other Model Launches**

- **Tencent Hy4**: Open weights for a 770B/49B-active natively multimodal MoE with 1M context, released with a system card ([Tencent](https://x.com/TencentHunyuan/status/2123625549147070538)).
  - **Serving**: Day-0 vLLM support covers BF16 and FP8 ([vLLM](https://x.com/vllm_project/status/2123631433965281773)).
  - **Safety evaluation**: Neel Nanda praised the third-party loss-of-control evaluation ([Neel Nanda](https://x.com/NeelNanda5/status/2123889182301139248)).
- **Microsoft Ignite**: Announcements include MAI Code 2, claimed at 96% of Opus 5.5's performance for 5% of the cost, plus the Autopilot agent and the Decision-1.1 router ([recap](https://x.com/kimmonismus/status/2123585386379411838)).
- **Additional releases**:
  - **Apple Depth Pro 2**: New depth model on Hugging Face ([akhaliq](https://x.com/_akhaliq/status/2123586174405296481)).
  - **Kyutai AURA**: Open-source real-time lip-synced avatar running on a single GPU ([Kyutai](https://x.com/kyutai_labs/status/2123795191417156022)).
  - **Baseten bison-1**: Ranks #1 on the AA Decision Index ([Baseten](https://x.com/baseten/status/2123871986862772641)).

**Industry and Policy**

- **Anthropic IPO**: Described as the largest IPO ever, raising $81.3B at pricing and $93.5B including the underwriters' option, at $83 a share ([numbers](https://x.com/TheTuringPost/status/2123907606649376817)).
  - **Isomorphic Labs**: Raised a $5B Series C (same source).
- **US–China**: Trump said he and Xi agreed to "joint guardrails" on AI, with no details given ([video quote](https://x.com/peterwildeford/status/2123840221816996322)).
  - **Chip manufacturing rules**: New US rules may block TSMC and Samsung from manufacturing Chinese AI chips ([Kyle Chan](https://x.com/kyleichan/status/2123606503080726858)).
  - **Nscale and ByteDance**: Nscale needs a license to give ByteDance remote GPU access; ByteDance made up about 73% of its 2025 revenue ([Techmeme](https://x.com/Techmeme/status/2123661586569351250)).
  - **Malaysia smuggling estimate**: Epoch finds trade records consistent with more than $3B of smuggled servers, roughly 150k H100-equivalents ([Epoch](https://x.com/EpochAIResearch/status/2123836818386997681)).
- **Nvidia leadership**: Following Jensen Huang's resignation, Trump urged Nvidia to reinstate him ([report](https://x.com/kimmonismus/status/2123866971515293984)).
- **Datacenter permitting**: SemiAnalysis says the Texas permit freeze could last 8–12 months under the newly elected governor ([SemiAnalysis](https://x.com/SemiAnalysis_/status/2123602891713708464)).
- **OpenAI distillation research**: OpenAI published work on detecting and disrupting distillation attacks on its models ([OpenAI](https://x.com/OpenAI/status/2123862344422088794)).

**Top tweets (by engagement)**

- [Scientific American: Anthropic mathematician claims RH proof](https://x.com/sciam/status/2123782587235655707) (22.5K)
- [Separate human proof of RH announced](https://x.com/KSHartnett/status/2123847664064241799) (9.3K)
- [Thinking Machines launches Inkling 2 Mini](https://x.com/thinkymachines/status/2123871643797995872) (7.9K)
- [Theo on Gemini 4 Neon usability](https://x.com/theo/status/2123619214061977920) (7.2K)
- [Alpöge RH claim details](https://x.com/kimmonismus/status/2123791432800698792) (7.2K)
- [Mira Murati on looping and monitorability](https://x.com/miramurati/status/2123872534970208421) (4.3K)
- [Trump: joint AI guardrails with Xi](https://x.com/peterwildeford/status/2123840221816996322) (4.2K)
- [OpenAI MedChemBench](https://x.com/OpenAI/status/2123856316187975756) (3.4K)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Open-Weight Multimodal MoEs: Hy4 and Inkling-2-Mini

  - **[Tencent Hy4 (flagship multimodal MoE A49B/770B) released](https://www.reddit.com/r/LocalLLaMA/comments/1y0vaey/tencent_hy4_flagship_multimodal_moe_a49b770b/)** (Activity: 495): **The [image](https://i.redd.it/vjwcet1udp2i1.jpeg) is a **technical benchmark comparison chart**, not a meme, for **Tencent Hy4**, a newly released natively multimodal MoE model with `770B` total parameters, `49B` active parameters, and `1M` context. It shows Hy4 improving over the dashed **“Hy4 preview”** baseline across coding, agent, terminal, browsing, and OS-world style tasks, while still often trailing frontier closed models such as **Claude Opus 5.5** and **GPT 6.1 Sol**; competing open/Chinese models like **Kimi**, **GLM**, and **DeepSeek** are also included. Relevant release links include the [GitHub repo](https://github.com/Tencent-Hunyuan/Hy4), [Hugging Face weights](https://huggingface.co/tencent/Hy4), [GGUF conversion](https://huggingface.co/AngelSlim/Hy4-GGUF), [license](https://huggingface.co/tencent/Hy4/blob/main/LICENSE), and [system card](https://hy.tencent.ai/Hy4%5C_system%5C_card.pdf).** Commenters viewed the benchmark performance as strong but noted the model is *“chonky”* for local use given its `770B/49B active` MoE scale. There was also discussion of whether Hy4 lacks **Engram**, whether it could replace **GLM5.3** in local workflows, and whether its cyber capabilities are restricted similarly to models like Qwen or K-series releases.

    - Several commenters focused on deployability: **Tencent Hy4** is described as strong but “chonky,” with the post title indicating a **multimodal MoE** configuration of `A49B/770B`, implying substantial memory/compute requirements for local inference compared with smaller open-weight models.
    - A technical comparison thread positions **Hy4** as a possible replacement for **GLM 5.3**, while questioning whether its cybersecurity capabilities are restricted similarly to **Qwen4** and **Kimi K3.1**. Another commenter notes **hy4-preview** was already competitive and “keeps up with **Kimi K3.1** in a lot of places,” suggesting expectations of comparable benchmark/task performance.

  - **[Tencent Hy Team & Shunyu Yao Here! Ask Us Anything about Hy4 & Hy4 flash!](https://www.reddit.com/r/LocalLLaMA/comments/1y15w4a/tencent_hy_team_shunyu_yao_here_ask_us_anything/)** (Activity: 1026): ****Tencent Hy Team & Shunyu Yao** are hosting an AMA on **Hy4** and **Hy4 Flash**, covering model construction, efficiency improvements, and workflow/agentic-use performance; official materials are on the Hy blog for [Hy4 Flash](https://hy.tencent.ai/research/hy4-flash) and [Hy4](https://hy.tencent.ai/research/hy4), with weights on Hugging Face: [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) and [tencent/Hy4](https://huggingface.co/tencent/Hy4). A top commenter reports **Hy4-flash** “saturated” their internal agentic coding / PR-review benchmark, finishing faster than **Qwen 3.8 Flash Next**, **GLM 5.3 Flash**, and **DeepSeek V4.1 Flash**. Another comment frames Hy4-flash as an MoE-style `156B-A12B` model with `7.7%` active parameters, compared with `Qwen3.8-Flash 125B-A6B` (`4.8%`) and `DSv4.1 552B-A16B` (`2.9%`, reportedly `A8B`/`1.45%` during prefill), asking whether the higher active-parameter density is responsible for quality gains and what degradation appears at `A8B` or `A5B`.** Commenters are particularly interested in whether Tencent will ship a smaller **Hy4-mini** class model to compete with/replace **Qwen3.6-35B-A3B**, and whether Hy4-flash’s apparent quality comes from higher MoE activation density rather than architecture/training differences.

    - A user reported that **Hy4-flash** fully saturated their internal benchmark for *agentic coding and PR review*, claiming it was the first model to pass all tests while completing faster than **Qwen 3.8 Flash Next**, **GLM 5.3 Flash**, and **Deepseek V4.1 Flash**. The benchmark details were not provided, but the comment highlights perceived strength in coding-agent workflows and latency-sensitive PR review tasks.
    - One technically detailed question compared MoE activation density across frontier flash models: **Hy4-flash** was described as `156B-A12B` with `7.7%` active parameters, **Qwen3.8-Flash** as `125B-A6B` with `4.8%`, and **Deepseek V4.1** as `552B-A16B` with `2.9%`, dropping to `A8B` / `1.45%` during prefill. The commenter asked whether Hy4-flash’s higher active-parameter density is important for intelligence, and whether Tencent tested lower-density variants such as `156B-A8B` or `156B-A5B`.
    - Several questions focused on scaling limits and future model design: whether current Hy models have reached the amount of intelligence they can “soak,” and whether stronger future Hy releases require more total parameters or could come from better post-training at the same size. Another user asked whether a smaller **Hy4-mini** variant might replace **Qwen3.6-35B-A3B**, implying interest in a lower-cost open-weight model tier.

  - **[Hy4-flash first impression: it's got AMAZING coding taste](https://www.reddit.com/r/LocalLLaMA/comments/1y0ofce/hy4flash_first_impression_its_got_amazing_coding/)** (Activity: 419): **A SWE reports an early, qualitative evaluation of **Hy4-Flash** for Python generation, using it for several hours on moderate-complexity, higher line-count tasks via datacenter serving/OpenRouter. Compared with **DeepSeek 1017** and **Qwen3.8-Flash**, they found Hy4-Flash required far less post-generation cleanup: fewer excessive comments, fewer unnecessary special-case branches, and generally more “boring and sensible” structure; reported usage cost was about `$0.10` for several hours.** Commenters asked for concrete side-by-side task examples to substantiate the comparison. Others agreed that manual review remains standard practice, and one commenter speculated that Hy4-Flash may have stronger judgment for software architecture based on reports of better document-review accuracy.

    - One commenter asked for **concrete side-by-side examples** showing Hy4-Flash and other models solving the same coding task differently, specifically to substantiate claims about Hy4-Flash having better “coding taste” or architecture decisions.
    - A technical explanation suggested that excessive inline prose/comments in model-generated code may come from partial distillation on **Anthropic/GPT-style outputs**, where comments can carry watermark-like stylistic artifacts without corrupting executable code. The proposed mitigation is to explicitly instruct the model that documentation belongs in external/domain docs rather than inline code comments.
    - Another detailed point argued that good agentic coding should be structured as separate passes: first “make it work and ship,” then run distinct refactoring/optimization waves. The commenter noted that both human organizations and AI agents often fail to perform the second pass, leading to bloated “god” functions/classes despite the initial implementation being functionally correct.

  - **[thinkingmachines/Inkling-2-Mini (MoE: 26b, a3B)](https://www.reddit.com/r/LocalLLaMA/comments/1y1diuk/thinkingmachinesinkling2mini_moe_26b_a3b/)** (Activity: 1340): ****Thinking Machines Lab** released [**Inkling-2-Mini**](https://huggingface.co/thinkingmachines/Inkling-2-Mini), an Apache-2.0 open-weight multimodal MoE model with [GGUFs by Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). It is a **24-layer decoder-only sparse MoE** with `26.2–26.4B` total params, `3.4B` active before looping and ~`5.4B` effective active params/token via three passes through 12 shared middle layers; each token routes to `8/128` experts, uses hybrid local/global attention, supports image+text inputs, and has `256K` native context extended to `1M` with YaRN. A Thinky employee noted the model was trained end-to-end in [`nvfp4`](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/) and appears unusually quantization-tolerant, citing Unsloth reports of ~`90%` top-1 retention even at `1-bit` quantization, while the looped-depth design was said to help coding tasks.** Commenters focused on its local-inference positioning, comparing it to **Qwen3-27B/30B-class** and **GPT-OSS-20B** due to the sub-50B size, FP4-oriented training, multimodality, and long-context support. There was excitement but also clarification that the headline `1M` context is extrapolated via YaRN rather than native, with native context reportedly `256K`.

    - A **Thinking Machines** employee highlighted two local-deployment-relevant design choices: Inkling-2-Mini was trained end-to-end in `nvfp4` ([blog](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)), and early third-party observations from **Unsloth** suggest unusually strong post-training quantization behavior, reportedly maintaining `90%` top-1 even at `1-bit` quantization ([X post](https://x.com/danielhanchen/status/2122685315257410031/photo/1)). They also noted the model uses a **looped architecture**, intended to provide greater effective depth at the same parameter count and apparently benefiting coding workloads.
    - Commenters emphasized the model’s local-friendly size/compute profile: **MoE `26B` total with ~`3B` active parameters**, positioning it against models like **Qwen3.8-27B** and **LFM3** while potentially fitting into the niche occupied by **GPT-OSS-20B**-style low-precision models. One user linked **Artificial Analysis** benchmark coverage for Inkling-2-Mini: https://artificialanalysis.ai/models/inkling-2-mini.
    - The context-window discussion clarified that the advertised **`1M` token context** is achieved via **YaRN**, while the model’s native context length is reportedly `256K`. Users also called out multimodal/image support, making the combination of sub-50B scale, long context, and local-oriented low-precision training the main technical draw.


### 2. Tiny Local Models for Speech and Documents

  - **[Kitten-TTS-2 Pico, Nano, Mini: SOTA super-tiny Apache 2.0 TTS, up to 16 languages (Pico is just 4 MB)](https://www.reddit.com/r/LocalLLaMA/comments/1y0e59m/kittentts2_pico_nano_mini_sota_supertiny_apache/)** (Activity: 570): ****KittenML** released Apache-2.0 code and weights for **Kitten-TTS-2 Pico/Nano/Mini**, a multilingual “super-tiny” TTS family: Pico is `~7M` params / `4 MB` with `8` languages, Nano is `~17M` / `8.8 MB` with `8` languages, and Mini is `~80M` / `48 MB` with `16` languages. The project claims Pico runs faster than realtime on a single smartphone CPU thread, with repos/demos/weights available on [GitHub](https://github.com/KittenML/KittenTTS), [HF demo](https://huggingface.co/spaces/KittenML/KittenTTS-2-Tiny-Demo), and model cards for [Mini](https://huggingface.co/KittenML/kitten-tts-2-mini), [Nano](https://huggingface.co/KittenML/kitten-tts-2-nano), and [Pico](https://huggingface.co/KittenML/kitten-tts-2-pico). Compared with the previous series, the author claims a new Pico tier that is `~6×` smaller than the old `25 MB` Nano, improved quality, and new multilingual support.** One technical commenter reported good perceived quality for English and German, but observed weak punctuation/prosody handling: outputs were run together into a near-single sentence, with commas and full stops almost ignored. Another commenter asked whether voice cloning is supported; no cloning capability is stated in the post.

    - Users reported **good English and German audio quality**, but one tester found the model “ran everything together into a single sentence,” largely ignoring commas and full stops. This suggests a potential **punctuation/prosody handling issue** in Kitten-TTS-2’s text normalization or pause prediction despite otherwise acceptable synthesis quality.
    - One user comparing against **Kokoro** noted that Kitten-TTS-2 handled **New Zealand town-name pronunciation** surprisingly well, where Kokoro frequently fails. They planned further testing and requested future **Mandarin support** plus additional younger/playful male voice presets, implying current male voices such as “Vincent” skew deeper or strained.

  - **[Cloudflare releases the weights of Clairic, its AI that reads invoices and receipts](https://www.reddit.com/r/LocalLLaMA/comments/1y03byh/cloudflare_releases_the_weights_of_clairic_its_ai/)** (Activity: 314): ****Cloudflare** released open weights for **Clairic** and **Clairic-flash**, Apache-2.0 document-extraction models for invoices/receipts; the original source is Cloudflare’s [Open Clairic announcement](https://blog.cloudflare.com/open-clairic), with a secondary summary noting support for text/images/PDFs up to `4` pages and outputs including typed fields, confidences, and spans. Commenters note the models are post-trained from **Qwen3.8-27B** and **Qwen3.5-9B** respectively, with the larger model around `27B` parameters / ~`56GB` and flash around `9B` / ~`20GB`; there are no public independent benchmarks yet, and GGUF/Ollama/llama.cpp packaging is apparently not available.** Commenters view the **Apache-2.0** licensing and self-hostability as especially useful for regulated sectors such as banking, where invoices/receipts often cannot be sent to third-party APIs. There is interest but no reported evidence yet on non-English invoice performance, particularly Turkish, despite the Qwen base suggesting possible multilingual capability.

    - Commenters identified the release as based on **post-trained Qwen models**, specifically mentioning `Qwen 3.8-27B` and `Qwen 3.5-9B`, and pointed readers to Cloudflare’s original technical announcement: https://blog.cloudflare.com/open-clairic.
    - A technically relevant deployment point was the **Apache 2.0 license** and ability to run Clairic **in-house**, which commenters noted is important for banks and other regulated users that cannot send invoices/receipts to external APIs. One commenter also raised multilingual extraction quality, noting that because the base is **Qwen**, languages such as Turkish *might* work well, but asked whether anyone had tested non-English invoices yet.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude-Assisted Riemann Hypothesis Proof Claim

  - **[Anthropic mathematician claims to prove Riemann hypothesis, most famous problem in all of math](https://www.reddit.com/r/singularity/comments/1y15d87/anthropic_mathematician_claims_to_prove_riemann/)** (Activity: 2018): **[*Scientific American* reports](https://www.scientificamerican.com/article/anthropic-mathematician-claims-to-prove-riemann-hypothesis-most-famous-problem-in-all-of-math/) that **Anthropic mathematician Levent Alpöge** claims to have proven the **Generalized Riemann Hypothesis**, with assistance from **Claude**, and released a **Lean-formalized** proof for machine verification. If both the mathematical argument and Lean formalization survive expert review, this would resolve one of the central open problems in number theory and represent a major milestone for AI-assisted theorem proving; however, community validation is still pending.** Commenters largely treat the claim as potentially historic but unverified. One theoretical physicist/AI researcher argued that, if true, it would be a “major discontinuity point” for mathematics, potentially affecting student interest, grant allocation, and perceptions of human vs. AI intellectual capacity.

    - A commenter identifying as a theoretical physicist now working in AI argues that, **if the proof is valid**, it would represent a major discontinuity for mathematics and AI-assisted research. They highlight likely downstream effects on the perceived value of human mathematical labor, student incentives, and grant allocation, framing it as evidence that AI systems may exceed human intellectual capabilities on historically hard open problems.
    - Several commenters stress the conditional nature of the claim—*“if this is true”*—while noting that a valid AI-linked proof of the **Riemann hypothesis** would be the largest demonstrated AI advance in mathematics by a wide margin. One technical contrast raised is that, unlike breakthroughs in physics, biology, or medicine, resolving this millennium-level pure math problem may have limited immediate practical impact despite its enormous theoretical significance.

  - **[Something just happened.](https://www.reddit.com/r/singularity/comments/1y14vi5/something_just_happened/)** (Activity: 2431): **The image is a screenshot of a typeset note from [Levent Alpöge’s domain](https://alpo.ge/remark.pdf) claiming, with assistance from **Claude**, a proof of the **Riemann Hypothesis** and related extensions such as the **Generalized** and **Extended Riemann Hypotheses**; the highlighted footnote marker suggests an important caveat or attribution in the document. If genuine and correct, this would be a major Millennium Prize-level result, but the post provides no peer review or verification yet. Image: [https://i.redd.it/jvhn3tvi2s2i1.png](https://i.redd.it/jvhn3tvi2s2i1.png)** Comments are mostly astonished and cautious, noting that this would imply multiple major mathematical breakthroughs arriving unusually quickly, while also making light of the Claude involvement with remarks like *“Claude et al.”*

    - Commenters identify the claim as likely coming from **Levent Alpöge** via the `alpo.ge` domain, noting his prior work on **Navier–Stokes** with **Tristan Buckmaster**. The discussion frames it as potentially one of multiple recent **Millennium Prize-level proof claims**, alongside mentions of progress on **ETH** and **Yang–Mills**.
    - A technically focused concern is verification latency: commenters ask how long expert review should take even if a proof is **Lean-formalized**. One commenter notes that Lean can provide confidence in *internal logical consistency*, but humans still need to confirm that the formal theorem statement actually corresponds to the claimed mathematical result.
    - Several comments connect the event to AI-assisted mathematics, with “**Claude et al.**” cited as a possible driver. The thread also mentions a separate claim that **unforced Navier–Stokes** was solved the same day, emphasizing uncertainty around whether these are verified breakthroughs or early-stage proof announcements.


### 2. Gemini 4 Model Rollout Signals

  - **[Google teases yet ANOTHER Gemini model drop!](https://www.reddit.com/r/singularity/comments/1y0awmz/google_teases_yet_another_gemini_model_drop/)** (Activity: 995): **The [image](https://i.redd.it/law74od0zk2i1.jpeg) is a teaser-style screenshot of **Omar Sanseviero** saying *“Shipping will continue,”* alongside a frame reading **“GEMINI 4 WILL RETURN”**, which the post interprets as Google hinting at another upcoming **Gemini** model/update. There are no concrete technical details, benchmarks, architecture notes, or release dates in the image or comments—this is primarily **speculative product-release signaling**, not a technical announcement.** Commenters frame Google’s rapid release cadence as a competitive threat to **OpenAI**, with one suggesting OpenAI may need to change its cadence to keep up. Another comment speculates the teaser could refer to **“helium,”** but no evidence or technical context is provided.


  - **[oh my god they made the joke real](https://www.reddit.com/r/GeminiAI/comments/1y0i7ve/oh_my_god_they_made_the_joke_real/)** (Activity: 960): **The image ([link](https://i.redd.it/zh0kq2uldm2i1.png)) appears to be a **joke/mock UI**, not a confirmed technical announcement: it shows a “Model selection” screen styled as a **Periodic Table of Gemini 4**, with a selected “Gemini 4 Hydrogen” model listing specs like `1M input`, `64K output`, pricing, and a release date. The title (“oh my god they made the joke real”) and comments frame it as the periodic-table naming meme being visualized as if it were an actual product selector rather than presenting real benchmark or implementation details.** Commenters mostly treated it as funny and possibly plausible branding, joking that there are “only 115 models left until Gemini 5” and noting that naming Gemini variants after elements is “interesting” and even likable.


  - **[Didn’t think it was possible that a model could have a personality in big 2026 but Google managed it somehow](https://www.reddit.com/r/Bard/comments/1y0a5e0/didnt_think_it_was_possible_that_a_model_could/)** (Activity: 140): **The image is a [smartphone screenshot of a Gemini chat](https://i.redd.it/9ywo33d8tk2i1.png) showing **Google Gemini “Neon Medium”** responding with a notably snarky, self-aware style about alleged Claude-like phrasing and “synthetic data contamination.” This is **non-benchmark, anecdotal evidence** rather than a technical finding: the post frames the model’s humor/persona as surprising for a “flash tier” Gemini variant, with the OP claiming it subjectively outperforms **Claude Opus 5.5** for their use and criticizing Google for limiting access despite paid tiers.** Commenters mostly discuss perceived “model personality” differences: Gemini is praised as a balanced mix of playful, friendly, and willing to push back, while GPT is described as overly agreeable, Claude as moralizing, Grok as edgy, and Chinese models as less distinctive. Several users agree that Gemini/Bard has historically been funny, while another jokingly calls for a “smoking gun.”

    - Several commenters compared **model persona/alignment behavior** across providers: **GPT/OA** was described as overly agreeable or “yes-man” when implementing weak ideas, **Claude** as more moralizing/philosophical, **Grok** as intentionally edgy, and Chinese models as having less distinct persona. The main technical takeaway is that users perceive **Gemini** as striking a different alignment balance: more willing to push back on bad assumptions while maintaining playful, friendly interaction.
    - One commenter noted that Gemini’s personality does not feel substantially changed from the **3.x** generation, but that its **creative-writing quality** appears improved. This suggests perceived gains may be more in generation quality/style control than in a wholesale shift in assistant persona.


### 3. ICLR AI-Generated Review Crackdown

  - **[ICLR is deleting AI generated reviews and DR AI reviewers' papers [D]](https://www.reddit.com/r/MachineLearning/comments/1y0y3rs/iclr_is_deleting_ai_generated_reviews_and_dr_ai/)** (Activity: 132): **A Reddit post claims **ICLR** is enforcing its review-integrity policy by deleting suspected **AI-generated reviews** and rejecting papers associated with reviewers/co-authors who used AI during peer review. A top comment reports that `484` papers were removed from the active submission list, while another asks for submission IDs to determine whether the removals are still rolling out or already completed.** Comment sentiment is broadly supportive of enforcement, with one commenter saying it is *“about time.”*

    - A commenter reports that **`484` papers were removed from the active list**, implying ICLR may have applied a moderation/filtering action at scale against papers associated with AI-generated reviews or dual-role AI reviewers. Other commenters ask for submission IDs, updated rankings, and examples of deleted reviews, but no concrete IDs, rankings, or review text were provided in the thread.

  - **[ICLR 2027 Desk Rejection [D]](https://www.reddit.com/r/MachineLearning/comments/1y0y1yl/iclr_2027_desk_rejection_d/)** (Activity: 159): **The post reports an **ICLR 2027 desk rejection** allegedly triggered by one or more co-authors failing to complete reciprocal reviewing obligations, with the author claiming they were not told which co-author/reviewer caused the violation and received no actionable warning. A commenter notes that warnings may depend on the email address in the author’s [OpenReview](https://openreview.net/) profile, and claims at least one co-author likely received a notification but failed to act; [TMLR](https://jmlr.org/tmlr/) is suggested as an alternative venue.** Comments split between *“well deserved”* accountability for negligent reviewers and criticism that reciprocal reviewing has made large ML conferences dysfunctional. One commenter argues major ML conferences have become “dumpster fires” and recommends smaller venues or journals instead.

    - A technically relevant process issue was raised around **OpenReview email/profile configuration**: one commenter reports receiving an email notification that allowed them to fix the issue, while suggesting the author may not have received it because their email was missing or misconfigured in their OpenReview profile. They also note that at least one co-author likely received the notification but failed to act, implying desk rejection may have resulted from missed compliance/action-item notifications rather than paper content.
    - Several commenters discussed the systemic risk of **reciprocal reviewing and author compliance requirements** at large ML conferences such as ICLR. One argued that reciprocal reviewing has introduced “serious consequences to authors” for reviewer-related or administrative failures, and recommended smaller targeted venues or journals such as **TMLR** as lower-risk alternatives.
    - A specific possible cause mentioned was a required **AI documentation form**: one commenter asked whether co-authors had received an email to “fill ai documentation,” noting that some people received such requests the previous week. This suggests the desk rejection may have been tied to an unmet submission metadata/documentation requirement rather than review quality.




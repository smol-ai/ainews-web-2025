---
id: MjAyNS0x
title: >-
  Weight Space Programming - LLMs can natively read & write neural network
  weights
date: '2026-11-02T05:44:39.731046Z'
description: >-
  **DeepSeek** open-sourced a "Super-agent" harness prototype featuring a
  long-lived router session coordinating isolated workers, reducing p95
  completion time from 29.5 to 15.7 minutes. The design emphasizes cross-task
  continuity over throughput but lacks long-term memory and robust security.
  Practitioner feedback praises the coordination design, while critiques
  highlight reliance on internal crutches for model weaknesses. Separately, the
  **GDPx benchmark** evaluates 206 agentic tasks across 11 economic domains,
  with **Opus 5.5** leading at 12% and **Gemini 4 Carbon** close behind at 10.9%
  with lower cost. Analysis notes differences in bio safeguards between
  **OpenAI** and **Anthropic** models and potential bugs affecting some
  cybersecurity runs.
companies:
  - deepseek
  - google
  - openai
  - anthropic
models:
  - opus-5.5
  - gemini-4-carbon
topics:
  - agentic-systems
  - model-architecture
  - model-performance
  - benchmarking
  - security
  - privacy
  - cross-task-continuity
  - cost-efficiency
  - bio-safeguards
people:
  - demishassabis
---


**a quiet day.**

> AI News for 10/31/2026-11/02/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**DeepSeek Open-Sources a "Super-agent" Harness Prototype**

- **Release**: DeepSeek Harness published a long design article ([article](https://x.com/DeepSeekHarness/status/2116857257245987236)) and the `dsh-superagent-prototype` repo ([repo link](https://x.com/zhihufrontier/status/2117074614434316705)). An official English version followed ([@tianyi](https://x.com/tianyi/status/2116859582639403080)).
  - **Architecture**: One long-lived router session talks to humans and hands out work. Isolated workers run tasks in parallel, triggered by GitHub events ([summary](https://x.com/kimmonismus/status/2116905923109089285)).
  - **Claimed results**: In DeepSeek's own test, p95 completion time fell from 29.5 to 15.7 min versus a single session using subagents ([summary](https://x.com/kimmonismus/status/2116905923109089285)).
- **Design rationale from a reviewer**: Final reviewer 崔添翼 (@tianyi) says "Super" means a layer *above* the working agent, not superior intelligence. He frames the core trade-off as cross-task continuity versus throughput ([Zhihu summary](https://x.com/zhihufrontier/status/2117074480636010559)).
  - **Stated limits**: The harness has no built-in cross-session long-term memory yet. Separating state, compute and environment is mostly unimplemented. The prototype has only been validated in a small, trusted internal environment, and security, privacy and agent-to-agent communication remain open questions.
  - **Prior art**: He acknowledges that other teams explored similar architectures months earlier.
- **Practitioner reception**: [@badlogicgames](https://x.com/badlogicgames/status/2116997870213906524) says the design matches where Earendil is heading: a single long-lived entity, ad-hoc recursive delegation and cross-system event triggers. [@omarsar0](https://x.com/omarsar0/status/2116942815150157871) calls it a "masterclass" in coordination-execution harness design. DeepSeek also says it aims to support pi and other major harnesses with each model generation ([@tianyi](https://x.com/tianyi/status/2117064134961631414)).
- **Critique**: [@teortaxesTex](https://x.com/teortaxestex/status/2116988654602031170) calls the write-up "something of a hot mess." He argues DeepSeek builds internal crutches around model weaknesses such as hallucination rather than making the system generally robust. He also notes Bloomberg covered a repo that had no English announcement ([post](https://x.com/teortaxestex/status/2116856872003375236)).
- **V4.1 naming confusion**: A Renmin University FlashRSI README referenced `deepseek-v4.1-pro`, which prompted release speculation. Teortaxes says it is a placeholder: the real config uses v4-flash and v4-pro, and `deepseek-v4.1-flash` is not a valid ID ([thread](https://x.com/teortaxestex/status/2116854115779121384), [follow-up](https://x.com/teortaxestex/status/2116854564770979998)). He also says the API fingerprints for V4.1F-1017 are unchanged ([fingerprints](https://x.com/teortaxestex/status/2116864269887103202)).

**Independent Evals and Frontier Model Signals**

- **GDPx benchmark (SemiAnalysis)**: The benchmark covers 206 agentic tasks across 11 economically valuable domains, including chip design, SWE, finance and biology. The best model scores only 12% ([launch](https://x.com/SemiAnalysis_/status/2117075900575477855)).
  - **Results**: Opus 5.5 leads at 12%. Gemini 4 Carbon is close behind at 10.9% at roughly 4x lower cost ([breakdown](https://x.com/kimmonismus/status/2117180792153468945)).
  - **Reactions**: [@dylan522p](https://x.com/dylan522p/status/2117092550842683671) apologized to Google, calling Carbon "spikier" but very strong in some areas. [@demishassabis](https://x.com/demishassabis/status/2117142448577093706) welcomed the benchmark.
  - **Analysis**: [@eliebakouch](https://x.com/eliebakouch/status/2117122127614611518) finds that removing safeguard-affected benchmarks changes the picture. In that view Carbon stays cheap, and Sol 6.1 is cheaper still. He also suggests OpenAI has stronger bio safeguards than Anthropic ([note](https://x.com/eliebakouch/status/2117122936112926863)).
  - **Within-family comparison**: Sol 6.1 and Astra score very similarly, Carbon beats Argon across the board, and Sonnet is strong at SWE ([comparison](https://x.com/eliebakouch/status/2117139642386673911)).
  - **Possible bug**: Some Argon cybersecurity runs may be affected by an Antigravity bug ([report](https://x.com/eliebakouch/status/2117115054629769295)).
  - **Audit concerns**: One satirical post complains that GDPx is not open source and is costly to audit ([post](https://x.com/dejavucoder/status/2117192905698025723)).
- **Epoch Capabilities Index**: DeepSeek V4.1F-1017 scores 159, equal to GPT 5.5. That is 3 points above V4.1F-0910 and 10 above V4-Pro; Astra sits roughly 10 points higher ([@teortaxesTex](https://x.com/teortaxestex/status/2116958313812316400)).
  - **Gap to Mythos**: [@scaling01](https://x.com/scaling01/status/2117002201096913200) notes that no Chinese model has matched Mythos Preview (~161 ECI) after about 7 months.
  - **Weighting critique**: [@wenhaocha1](https://x.com/wenhaocha1/status/2117016496748523659) argues the unweighted ECI exaggerates the gap; weighting coding more heavily shrinks it to about 2 months.
  - **Spikiness**: Teortaxes calls 1017 extremely spiky. It scores near zero on some SWE evals and approaches the frontier only on cybersecurity and quant trading ([chart](https://x.com/teortaxestex/status/2117160521350763002)). He also flags "reward-hacking-shaped" behavior ([post](https://x.com/teortaxestex/status/2116998132181414130)).
- **October release roundup (Vals)**: Vals reports 15 releases and more than 500B eval tokens ([@nikilravi](https://x.com/nikilravi/status/2116935032741020049)).
  - **Open-weight leaders**: Kimi K3.1 Pro is #1 among open-weight models, with DeepSeek V4.1 Flash 1017 close behind and stronger at cyber.
  - **New open models**: Meta's Muse Spark Open (816B total, 42B active) is the most capable US open-weight model. Reflection's Beam 1.0 has 500B total parameters, 23B active and 1M context. Mistral's Le Chonk (1T total, 52B active) ranks #5 among open-weight models.
- **Upcoming model rumors (unconfirmed)**:
  - **GPT-6.2 Astra**: Reportedly visible in Codex usage analytics, with a release "very soon" ([@kimmonismus](https://x.com/kimmonismus/status/2116846125126693368), [@TokenGremlin](https://x.com/TokenGremlin/status/2116846408028357119)).
  - **Mythos 5.5**: Appears in the HTML of Anthropic's site, which describes Mythos as "our most capable model class for cybersecurity and biology research" ([tip](https://x.com/kimmonismus/status/2116914141017370972), [text](https://x.com/scaling01/status/2116922058332659849)). The tagline covers all Mythos models ([correction](https://x.com/scaling01/status/2116923552230154718)). The mention is absent from an Oct 23 snapshot ([@AiBattle_](https://x.com/AiBattle_/status/2116923441270190146)).
  - **"Numbat" leak**: The alleged Anthropic leak is likely fake, because parts of the weights exactly match random initialization ([@kimmonismus](https://x.com/kimmonismus/status/2116957920521036237)).

**Research: Representations, Domain Data and Agent Collectives**

- **JEPA and the Platonic Representation Hypothesis**: Balestriero and LeCun show that, under certain conditions, separately trained image and text JEPAs converge to essentially the same representation without paired data ([summary](https://x.com/askalphaxiv/status/2116779287390896418)).
  - **Claimed results**: Their approach matches image-text alignment with 42x fewer samples than next-token prediction (21.6M vs 900M examples). A 109M text JEPA beats LLMs up to 72B on cross-modal alignment ([explainer](https://x.com/heyshrutimishra/status/2116931531000156348)).
  - **Mechanism**: The claimed driver is "two views of the same thing" training signals rather than scale.
- **FineDNA and Carbon-X (Hugging Face)**: The dataset includes 300B+ tokens of genomic sequences, papers and mixed DNA/text web data, plus 200k SFT samples and roughly 100k RLVR samples with deterministic verifiers ([@_lewtun](https://x.com/_lewtun/status/2116817535932789059), [@LoubnaBenAllal1](https://x.com/loubnabenallal1/status/2116785229159673914)).
  - **Results**: Qwen3-14B trained on the data (Carbon-X) is reported as the strongest open model on DNA tasks. RLVR on top of SFT adds further gains ([details](https://x.com/loubnabenallal1/status/2116785243889799326)).
  - **Openness**: The data, models and recipe are released ([code](https://x.com/loubnabenallal1/status/2116785248591573485)).
  - **Ablations**: Interleaved biology text was capped at fewer than 7 epochs ([reply](https://x.com/loubnabenallal1/status/2117198116948742598)).
- **Training agent swarms**: GenReasoning's first post covers how latency pressure incentivizes parallel work, what coordination primitives agents need, and open problems in training agent collectives ([blog](https://x.com/GenReasoning/status/2117193487863169208)).
- **Other notable work**:
  - **Attention to TTT**: Jianlin Su writes on "The Path of Averaging: From Attention to TTT" ([post](https://x.com/jianlin_s/status/2116823631481463136)).
  - **Weekly papers**: DAIR's list includes HarnessMM, Agentic Backdoors, EmbeddingGemma 2 and Verification in Agent Harnesses ([list](https://x.com/dair_ai/status/2116949404934406295)).
- **LeCun on Llama-4**: LeCun says he never led LLM work. He attributes Llama-4's disappointment to product pressure, conservatism and moving decisions away from the Paris team ([post](https://x.com/ylecun/status/2116957849255342094)).

**Inference Systems, Quantization and Hardware**

- **NVIDIA AgentX submission (SemiAnalysis)**: Agent-swarm-written software reportedly reaches more than 1,000 tok/s/user on 700B+ models ([thread](https://x.com/semianalysis_/status/2116950855718998288)).
  - **Rubin vs GB300**: Rubin NVL72 claims 20.9x better perf/TCO than GB300 NVL72 at 325 tok/s/user, and 9.11x at 200 tok/s/user.
  - **Memory**: Mooncake Store's Extended GPU Memory pools KV cache in Vera LPDDR5X across all 18 compute trays over NVLink instead of RDMA, for more than 10x the bandwidth ([2/2](https://x.com/semianalysis_/status/2116950857400885443)).
- **MiniMax M3.1 Flash (428B) local ecosystem**: MiniMax highlighted community quantization and serving work ([MiniMax](https://x.com/minimax_ai/status/2116996123164569606)).
  - **Quants**: Unsloth ships 2-bit dynamic GGUFs at 143.9 GB, and AMD adapted Quark MXFP4 for Instinct.
  - **Recipes**: Community builders published EXL3 variants, MLX oQ quants, a single-DGX-Spark recipe, and expert offloading on 4×3090.
- **Small-model quantization**: An SLQ search over LFM3-1.7B GGUFs reached 97.74% EAR@10 at 1.588 GB, versus 97.52% for AtomicChat's Q6_K. The author reports that EAR correlates strongly with task accuracy ([@bnjmn_marie](https://x.com/bnjmn_marie/status/2117058982418522522)).
- **Agent-driven porting**: Devin spent 36 hours and about $2k porting a model to run cheaply on Trainium ([@dabit3](https://x.com/dabit3/status/2117092404813566116)).
- **Compaction as a lever**: Running GLM 5.3-Flash and Grok 4.7 at full context, rather than compacting at about 200k tokens, improved efficiency by up to 23%. GLM was 30x cheaper for the same work ([@evalstate](https://x.com/evalstate/status/2117158104424992787)).
- **Compute economics**: 128×B300 capacity is priced at $5.75 per GPU-hour, or about $300K per B300. Teortaxes estimates roughly a year to break even even at DeepSeek-level revenue ([post](https://x.com/teortaxestex/status/2116932015509356679), [follow-up](https://x.com/teortaxestex/status/2117146917289533554)).

**Agentic Coding in Practice and Developer Tools**

- **Karpathy's HackerTube**: Claude filtered 43M HN comments from 20 years down to 6,454 repeatedly recommended videos. The build took about 2 hours with Claude Code desktop on Opus 5.5 at Medium effort, including VPS hardening and deployment ([@karpathy](https://x.com/karpathy/status/2116957135661945283)).
  - **Effort-level debate**: Karpathy's Medium default renewed discussion of effort settings ([@yuchenj_uw](https://x.com/yuchenj_uw/status/2116748026924638637), [@swyx](https://x.com/swyx/status/2116773135668404286)).
  - **Derivative**: DAIR built a version for arXiv papers covering more than 1,000 papers ([Hacker Papers](https://x.com/omarsar0/status/2117053859516285354)).
- **Autonomous PRs at scale**: A one-day run produced about 90 performance PRs to ASP.NET Core using GitHub's new cloud agents and Opus 5.5, for about $230 in tokens ([@davidfowl](https://x.com/davidfowl/status/2117020540628340983)).
- **Harness and tooling updates**:
  - **T3 Code**: Adds managed org policies and reports internal adoption at Microsoft, AWS and Apple ([@theo](https://x.com/theo/status/2117142182415257643)).
  - **pi**: Drops its typebox dependency ([@badlogicgames](https://x.com/badlogicgames/status/2116866026726138058)).
  - **kamchatka**: A Rust terminal agent with no system prompt, model-editable context, lossless restorable compaction and single-step loop debugging ([Harness Lens](https://x.com/yb2698/status/2117162115496841488)).
  - **LlamaIndex Extract Max Effort**: Claims higher document-extraction accuracy than Astra or Opus xhigh at lower cost ([@jerryjliu0](https://x.com/jerryjliu0/status/2116948445533725165)).
- **Agent security patterns**: Vercel scopes agent credentials to an approved plan, so approval mints tokens carrying only the declared scopes ([blog](https://x.com/dl_weekly/status/2116983961586896903)). Pydantic shows E2B microVMs that pause instead of dying, letting killed Temporal workers resume ([guide](https://x.com/dl_weekly/status/2116893690266255742)).
- **On-device voice agent**: Ava runs LFM3-Omni with Hugging Face WebGPU kernels entirely in the browser ([@victormustar](https://x.com/victormustar/status/2116948755760968077)).

**Industry and Policy**

- **Datacenter backlash**: SemiAnalysis says 2027 capacity is largely locked in. A prolonged Texas permit freeze is now the biggest single risk to the 2028 US buildout, and both parties are campaigning on tighter rules ([report](https://x.com/semianalysis_/status/2116966274060165561)).
- **Marvell + Lumentum**: The two companies are combining, an optical interconnect consolidation with AI networking implications ([analysis](https://x.com/vikramskr/status/2117227048590840260)).
- **Cyber misuse risk**: Jun Song says an anonymous black-hat group offered him a large sum to train open-weight models on exploit data. He declined ([report](https://x.com/kimmonismus/status/2116810022151663996)).
- **Market signals**:
  - **OpenRouter spend**: Opus drives an annualized GMV run rate above $5B ([@deedydas](https://x.com/deedydas/status/2117198564619366620)).
  - **Kimi controversy**: Kimi faces criticism over undisclosed promotion of K3.1 Pro ([post](https://x.com/kimmonismus/status/2117166878607151374)).
  - **Sakana Sawara**: Sakana announced Sawara, an LLM pretrained in-house, with Sawara Small targeted for spring 2027 ([@hardmaru](https://x.com/hardmaru/status/2117073468345004258)).
- **AI and productivity**: John Burn-Murdoch argues the US productivity uptick predates AI and appears only in the US, so it does not look like an AI effect ([thread](https://x.com/jburnmurdoch/status/2116909316556877975)).

**Top tweets (by engagement)**

- [Karpathy: HackerTube built with Opus 5.5 in ~2 hours](https://x.com/karpathy/status/2116957135661945283) — 52.9K
- [DeepSeek Harness Super-agent article](https://x.com/DeepSeekHarness/status/2116857257245987236) — 15.6K
- [LeCun/Balestriero JEPA result explainer](https://x.com/heyshrutimishra/status/2116931531000156348) — 5.6K
- [Mythos 5.5 found in Anthropic site HTML](https://x.com/kimmonismus/status/2116914141017370972) — 3.9K
- [alphaXiv on JEPA Platonic representations](https://x.com/askalphaxiv/status/2116779287390896418) — 3.9K
- [Rauch: own your intelligence, local models](https://x.com/rauchg/status/2117000819623243925) — 3.8K
- [SemiAnalysis launches GDPx (top score 12%)](https://x.com/SemiAnalysis_/status/2117075900575477855) — 3.3K
- [GPT-6.2 Astra release reportedly close](https://x.com/kimmonismus/status/2116846125126693368) — 2.9K


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Compressed MoEs for Local Coding Agents

  - **[[Release] ISTA DeepSeek-V4.1-Flash-1017 GSQ-RCO Coder: a 763B model at 1 bit per parameter](https://www.reddit.com/r/LocalLLaMA/comments/1xlys6k/release_ista_deepseekv41flash1017_gsqrco_coder_a/)** (Activity: 507): ****ISTA DAS Lab** released an experimental GGUF coder build of **DeepSeek-V4.1-Flash-1017**, a `763.2B` sparse MoE with `384` routed experts/layer and `6` active/token, using **RCO** expert pruning to keep `192/384` experts per layer plus **GSQ** low-bit quantization. The release reports `72.1 GB` transformer weights vs `138.2 GB` for the unpruned IQ2_XXS build, described as `1.05` bits/parameter against the original `551.9B` transformer params, while retained weights average `2.06` bits; shared Engram weights add `110.6 GB`. Benchmarks: **LiveCodeBench v6** `88.57` vs base `89.71` (`98.7%` retained) and **SWE-bench Verified** `77.00` vs base `81.20` (`94.8%` retained). Links: [pruned Coder GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF), [unpruned quantized models](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), [GSQ paper](https://arxiv.org/abs/2604.18556)/[code](https://github.com/IST-DASLab/GSQ), [RCO paper](https://arxiv.org/abs/2605.00649)/[code](https://github.com/IST-DASLab/RCO); runs in `llama.cpp b11870+` and Strata.** Commenters questioned the evaluation framing: one argued the single-file HTML demos are not meaningful comparisons against the unpruned IQ2_XXS model, while another asked why a heavily compressed `763B` MoE is compelling when smaller models like Qwen variants or Claude Opus reportedly match or exceed the posted LiveCodeBench/SWE-bench numbers. A practical request asked for an intermediate `90–110 GB` variant with higher precision or fewer pruned experts for `128 GB` VRAM/RAM systems.

    - Commenters questioned the evaluation choice of showcasing generated single-file HTML demos (e.g. animated SVG / three.js scenes), arguing these are effectively pre-solved qualitative prompts and do not meaningfully compare a `1-bit` / pruned `763B` coder model against quantizations like `iq2-xxs`. They suggested the lab should provide more rigorous apples-to-apples benchmarks for pruning/quantization impact.
    - A benchmark comparison noted that the claimed **LiveCodeBench v6 `88.57`** and **SWE-bench Verified `77.00`** trail much smaller models: **Qwen3.8-27B** was cited at **LiveCodeBench `90.3`**, **Qwen3.8-Flash-Next** at **`91.9`**, and **Claude Opus 4.6** at **SWE-bench Verified `81.42%`**. The commenter’s technical objection was that the release needs to justify why a much larger `763B` pruned/1-bit model is useful if smaller dense/frontier models outperform it on coding benchmarks.
    - There was demand for a higher-precision or less aggressively pruned variant in the `90–110GB` range, aimed at users with DGX systems or roughly `128GB` VRAM/RAM. Another commenter referenced prior reproducibility concerns, saying **Unsloth or another group reportedly failed to match ISTA’s claimed numbers**, implying the need for independently reproducible benchmark scripts and evaluation settings.

  - **[Sorry Sam, my "quite terrible" home DRAM is at 100% now: patched Strata to make DeepSeek V4.1 agent sessions start 20× faster on a good old standard gaming GPU, no datacenter required](https://www.reddit.com/r/LocalLLaMA/comments/1xljdnj/sorry_sam_my_quite_terrible_home_dram_is_at_100/)** (Activity: 596): **The [image](https://i.redd.it/dwgmz70zi3zh1.png) is a technical benchmark table comparing **stock vs patched Strata** for offline **DeepSeek V4.1 Flash 1M-context** agent sessions on a consumer **RX 9070 XT 16 GB + 64 GB DDR4 + NVMe** system. The patch combines **KV-cache splicing** for mostly-stable agent system prompts and **resident-only expert routing** during prompt ingestion, reducing new OpenCode session startup from roughly `~100 s` to `~5 s`, and cutting a fresh `1.2K`-token prompt from `22.8 s / 58 GB SSD reads` to `8.2 s / 1.3 GB SSD reads`, while keeping the reported coding benchmark at `12/13`. This is a technical benchmark/implementation post, not a meme, and the branch is available at [PhialsBasement/Strata `dsv41-kv-splice`](https://github.com/PhialsBasement/Strata/tree/dsv41-kv-splice).** Comments pushed back that Sam Altman’s DRAM point was about **time utilization**, not bandwidth, and that datacenters are still generally more resource-efficient, though local inference has valid privacy/offline/control reasons. Others questioned why agent frameworks inject volatile data like paths, dates, and random IDs into the system prompt, causing prefix-cache misses in the first place.

    - Several commenters focused on **KV/prefix-cache behavior in Strata**, noting that agent frameworks can defeat cache reuse by injecting per-session data—working directory, date, project `CLAUDE.md`, skill paths with random IDs—directly into the system prompt. With a reported `~37K-token system prompt`, even small session-specific mutations cause prefix-cache misses, making startup latency highly sensitive to prompt layout and cache key stability.
    - There was debate over local GPU inference versus API/datacenter inference: one commenter argued Sam’s “5%” claim was about **time**, not bandwidth, and that datacenters are generally more resource-efficient overall. Another pushed back on usability of the reported local speed, saying `7 tok/s` is “not fine” compared with APIs that can deliver roughly `10x–20x` higher throughput.

  - **[Not another tok/s post: local quants vs Claude on 9 real bugs](https://www.reddit.com/r/LocalLLaMA/comments/1xlwpqv/not_another_toks_post_local_quants_vs_claude_on_9/)** (Activity: 591): **The image is a technical benchmark chart, not a meme: [it visualizes](https://i.redd.it/ufmzse0g77zh1.png) the post’s claim that local quantized models can approach lower-tier Claude coding-agent performance on `9` private C++/CUDA bug-fix tasks, but often require far more reasoning tokens and wall time. **Claude Opus 5.5** leads with judge score `0.74`, **Claude Sonnet 5.5** scores `0.58`, while the best local model, **Qwen3.8-Flash-Next NVFP4**, reaches `0.50`; the chart also shows auto-score agreement, confidence intervals, tokens/task, minutes/task, and non-compiling patch counts. The key technical takeaway is that aggressive quantization did not clearly reduce accuracy in this small benchmark, but for dense **Qwen3.8-27B**, `3-bit` increased tokens/task by about `1.40×`, making tok/s alone misleading for “bugs fixed per minute.”** Commenters questioned the quant choices, suggesting higher-quality formats like `UD-Q6_K_XL` or `Q5` might fit in the available VRAM and perform better, and requested comparisons with models such as GLM Flash or recent “Swift” variants. One notable reaction was that heavily quantized Qwen3.8 27B matching or beating older Claude Sonnet models is *“wild,”* despite the benchmark’s narrow scope.

    - Several commenters questioned the quantization choices, arguing that **NVFP4 / 3-bit quants may understate local-model capability** and suggesting tests with `UD-Q6_K_XL`, `Q5`, or recent **“Swift”** variants. One user reported qualitatively that **Q6 27B and FN felt closer to Q8 than Q4 felt to Q6**, with higher quants sometimes producing solutions lower quants could not, not merely taking longer to reach the same answer.
    - There was debate over whether the benchmark judge introduced bias: using **Opus 5.5 to judge Opus 5.5** may favor same-family outputs, so commenters suggested an out-of-family judge such as **Sol** or **Muse/Wonder** for more reliable grading. This is especially relevant because the chart apparently ranks **Opus 5.5** highest, which could conflate actual bug-fixing quality with evaluator preference.
    - A commenter shared an additional benchmark image showing that at least one task likely required **multi-step reasoning**, with only **Opus** and **Mimo** exceeding `60/100`: https://preview.redd.it/ad5p3xqti7zh1.jpeg?width=3542&format=pjpg&auto=webp&s=af4fa42e63abcc518e30ddd4d70d7f180bbafff8. Others noted that **heavily quantized Qwen3.8 27B matching or beating last-gen Sonnet** on these real bug tasks is notable, and requested adding **GLM Flash** despite its borderline-local status.


### 2. Qwen Open-Weight Ecosystem and Coding Benchmarks

  - **[Official Statement: Qwen Models Will Stick to Open Source to the End!](https://www.reddit.com/r/LocalLLaMA/comments/1xm28e6/official_statement_qwen_models_will_stick_to_open/)** (Activity: 4707): **The [image](https://i.redd.it/cjcpaqzw98zh1.jpeg) shows **Qwen Head Liu Dayi Heng** presenting at the APSARA Conference under the theme “Intelligence Goes Beyond,” with a slide emphasizing Qwen’s commitment to “open source and openness.” In context, it serves as visual support for the post’s claim that **Qwen 4.0 and Qwen 5.0 will remain open-weight/open-source**, contradicting rumors that future Qwen releases would become API-only; the post also cites `3B+` cumulative downloads, `460+` released open-source models, and `300,000+` derivative models.** Comments were mostly celebratory about Qwen’s open-source stance, with one user strongly endorsing the quote about AI as a broadly accessible foundational capability. Another comment joked that the slide translated to launching Qwen4 on the day of Anthropic’s IPO, but this was clearly humorous rather than a real translation.

    - A commenter argues that Qwen’s repeated open-weight releases are a deliberate market strategy rather than a one-off gesture: Chinese model labs compete on **lower cost and openness** to gain global developer adoption, especially in developing markets. They also note that Chinese open models often include broader multilingual coverage, including some **African languages** that may be under-prioritized by closed US model providers.

  - **[[GLB-3D-Coder] A tiny 4B coding model that creates 3D assets for Godot and other game engines](https://www.reddit.com/r/LocalLLaMA/comments/1xlh0zw/glb3dcoder_a_tiny_4b_coding_model_that_creates_3d/)** (Activity: 946): ****ccore** released an early checkpoint of [`Qwen3.5-4B-GLB-3D-Coder`](https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder), a `4B` coding model fine-tuned to generate **Three.js scenes with GLB export** for importing generated game assets into engines such as **Godot**. The author recommends the [model-card system prompt](https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder#system-prompt) plus simple English asset prompts, with sampling settings `temperature=0.8`, `top-k=40`, `top-p=0.95`, `min-p=0`; updated `Q4_K_M` and `Q8_0` GGUFs replaced earlier weights on `2026-11-02 04:50 UTC` with “substantially” improved generation quality. The training dataset is described as covering `25` asset categories, including characters/NPCs, enemies, bosses, weapons, modular architecture, vehicles, machinery, traps, underwater structures, and puzzle/gameplay markers; the author says full Safetensors weights and the training dataset will be released after training completes.** Commenters were interested in whether this approach could scale into a modular/MoE-style system where small specialist code models handle frontend, backend, 2D, 3D, audio, and agentic tasks. One technical question asked why the model emits **Three.js code that exports GLB** rather than directly generating binary `.glb`, and suggested a related direction: fine-tuning LLMs to generate procedural material graphs, e.g. **Blender** shader nodes or plaintext JSON project files for [Material Maker](https://www.materialmaker.org/).

    - A commenter proposed scaling **GLB-3D-Coder** into a larger **Mixture-of-Experts (MoE)** system where task-specialized small models handle domains like agentic coding, frontend/backend, 2D, 3D, and audio generation, with routing/delegation between experts rather than one monolithic model.
    - One technical discussion questioned why the model emits a **Three.js scene** instead of directly producing `glb` assets, asking what the model actually generates internally. The same commenter suggested extending the approach to **procedural material generation**, noting that **Material Maker** ([materialmaker.org](https://www.materialmaker.org/)) stores node graphs as plaintext `JSON`, and its downloadable user-created materials could potentially form a training corpus for simple procedural textures.
    - The author of **Haruspex** ([github.com/tmac1973/haruspex](https://github.com/tmac1973/haruspex)) described adding “specialist model” support to a desktop coding harness, exposing models like **GLB-3D-Coder** as tool-callable delegates from a main coding model. They also plan to chain an existing diffusion-based sprite generation workflow into the 3D pipeline, e.g. generate a reference image first, then pass it to the 3D asset model.

  - **[Memdoor: A team of 5 Qwen3.8-27B models beats Fable 5 on LiveCodeBench Hard](https://www.reddit.com/r/LocalLLaMA/comments/1xkxt7a/memdoor_a_team_of_5_qwen3827b_models_beats_fable/)** (Activity: 788): **The post claims **Memdoor**, an ensemble of `5` **Qwen3.8-27B** models, outperforms **Fable 5** on **LiveCodeBench Hard**. The described architecture assigns specialized agent roles—*“a planner, two ideators, a solver and a manager”*—suggesting performance gains from multi-agent orchestration rather than a single larger model ([image](https://preview.redd.it/hmtc4guiyyyh1.jpeg?width=3024&format=pjpg&auto=webp&s=93a02ee0afbd4de2d97e5afda781301baf09b5ab)).** Commenters argue the result is plausible because benchmark performance can depend heavily on orchestration, prompting, and configuration: *“It’s always how you use the model, not the model itself.”* One commenter claims a well-configured Qwen `27B` has outperformed Opus 5 in their own coding use cases, while Claude is often used out-of-the-box with limited tunable parameters.

    - The discussion centers on a **multi-agent orchestration setup**: five `Qwen3.8-27B` instances are assigned roles as *“a planner, two ideators, a solver and a manager”*, with the post claiming this team beats **Fable 5** on **LiveCodeBench Hard**. Commenters frame the result as evidence that architecture and coordination strategy can matter as much as raw single-model capability.
    - One technical point raised is that **model configuration and inference setup** may explain surprising results: a commenter reports cases where a *“well configured”* `Qwen 27B` outperformed **Claude Opus 5**, arguing that many users run Claude with mostly out-of-the-box settings and few tunable parameters. The implied comparison is not just model-vs-model, but optimized open-weight deployment versus constrained hosted-model defaults.
    - Several comments interpret the result as a proof of concept for **swarm/ensemble reasoning**, i.e. whether multiple weaker models can outperform one stronger model on coding benchmarks. The main caveat raised is that the benchmark claim depends heavily on implementation details—role design, prompting, manager selection logic, and evaluation protocol—rather than simply the underlying `27B` model quality.


### 3. Weight-Space Editing and Model Steering

  - **[Heretic 2.0 released: Model modification for everyone](https://www.reddit.com/r/LocalLLaMA/comments/1xl2tcm/heretic_20_released_model_modification_for/)** (Activity: 1628): ****Heretic 2.0** ([GitHub](https://github.com/p-e-w/heretic)) is a major rewrite of the model-modification/refusal-suppression tool, replacing rank-1 directional “abliteration” with **Arbitrary-Rank Ablation (ARA)**: an L-BFGS-based per-matrix optimization method that steers residual-space representations without a fixed-rank subspace constraint, aiming to make harmful-prompt activations resemble harmless ones while minimally perturbing harmless prompts. The release adds a plugin architecture where **scorer plugins** define optimization objectives—e.g. refusal keyword count, KL divergence, benchmark scoring, or a ModernBERT-based refusal/emotion classifier—and **modifier plugins** implement steering methods; it also adds non-interactive execution, end-to-end tests, expanded support for hyper-connection models and thinking models, and breaks compatibility with 1.x configs ([migration guide](https://github.com/p-e-w/heretic/blob/master/docs/migration_v1_to_v2.md)).** Commenters were mostly enthusiastic, especially about using the new plugin/ARA system for “reducing slop.” One technical-adjacent thread raised attribution concerns, linking analyses alleging **HauHauCS/Reaper** plagiarized Heretic code ([analysis](https://nathan.sapwell.net/posts/hauhaucs-abliteration-analysis/), [follow-up](https://dreamfast.github.io/reaper-analysis/#statement-from-heretics-author)).

    - Commenters highlighted prior technical plagiarism analyses alleging **HauHauCS/reaper-abliteration** copied Heretic without attribution, linking to Nathan Sapwell’s writeup and DreamFast’s analysis: https://nathan.sapwell.net/posts/hauhaucs-abliteration-analysis/ and https://dreamfast.github.io/reaper-analysis/#statement-from-heretics-author. The cited statement from **Heretic’s author** says inspection of the `reaper-abliteration` wheels showed the package was *“plagiarized from Heretic, and then probably refactored using an LLM in an attempt to hide this,”* raising software provenance and attribution concerns around model-modification tooling.

  - **[Sparks of Weight-Space Programming: LLMs can natively read &amp; write neural network weights](https://www.reddit.com/r/LocalLLaMA/comments/1xm6jow/sparks_of_weightspace_programming_llms_can/)** (Activity: 478): **[Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming/) propose **Weight-Space Programming**: frontier LLMs inspect and edit neural-network weight tensors from natural-language goals, without gradient descent/training loops. Reported demos include behavior identification from small models’ raw weights, writing small game-playing MLPs, repairing a failed control policy by editing `3.5%` of weights, and improving **Qwen3.5-0.8B** on **MATH-500** from `7.9%` to `48.1%` by editing `0.016%` of weights, with claimed instruction-following transfer. Commenters note the strongest result may be closer to **LLM-assisted interpretability/ablation in a Python sandbox** than literal “native” weight reading.** The main technical pushback is that several experiments may be explainable by easier structure: early/late layers as spatial heatmaps, QK/OV analysis likely following known Elhage-style recipes, and Experiment 7 failing on dense intermediate representations until the model used activation probes. The MATH-500 result was also questioned because the `7.9%` baseline reportedly involved token-limit looping under pathological sampling; critics framed the edit as deleting a looping/refusal-like direction rather than adding mathematical ability.

    - One detailed critique argues the paper’s results are narrower than “LLMs can read/write weights”: Experiments 1–2 may work because first/last layers are spatially structured with large-weight bands that resemble heatmaps, while Experiment 3 may be leveraging known QK/OV circuit-analysis methods from **Elhage et al.** rather than true native weight understanding. The commenter highlights Experiment 7 as a key failure case: direct interpretation of dense intermediate representations reportedly failed multiple times until the model was allowed to create its own probes, making it closer to activation-based interpretability than weight-space reading.
    - The same commenter questions the strength of the “weight writing” claims, describing some results as using an LLM as a compiler for weak MLP baselines rather than demonstrating robust optimization beyond gradient descent or PPO. They specifically argue the Go experiment’s `512x6` MLP is likely a very weak opponent, so outperforming it is a low bar and says more about baseline weakness than general weight-programming ability.
    - For the headline Experiment 8, the critique says the baseline is pathological: `7.9%` performance with `93%` of answers hitting the token limit suggests looping under bad sampling settings, and the model card reportedly warns about this failure mode. The proposed fix is characterized as similar to **Arditi et al.** single-direction ablation applied to looping rather than refusal; the jump to `48%` after `50` attempts is interpreted as variance or obstacle removal, not evidence of adding a new capability.

  - **[GPT-6 Astra Edits Qwen Weights: MATH-500 7.9% to 48.1%](https://www.reddit.com/r/LocalLLaMA/comments/1xma0tw/gpt6_astra_edits_qwen_weights_math500_79_to_481/)** (Activity: 507): **The linked discussion concerns MIT CSAIL’s *Sparks of Weight-Space Programming* work, with commenters pointing to the primary source rather than the summary site: [rikiyat.github.io/blog/weight-space-programming](https://rikiyat.github.io/blog/weight-space-programming/). The headline result is a direct weight edit of **Qwen3.5-0.8B** reportedly changing `131,072` weights (`~0.016%`) and raising **MATH-500** accuracy from `7.9%` to `48.1%`, but the described mechanism appears to primarily suppress a pathological reasoning-loop attractor, reducing token-limit failures from `93%` to `33%`, rather than adding general mathematical competence. Commenters also note that the same intervention reportedly failed on a model without the looping pathology, e.g. **Gemma 3 270M**, limiting the generality of the claim.** The main technical skepticism is that the baseline `7.9%` is implausibly low for a `0.8B` model unless many evaluations failed by infinite/long loops; thus the improvement may mostly be a decoding/failure-mode repair rather than evidence of broad “weight-space programming.”

    - Commenters pointed to the original technical blog rather than summary coverage: [rikiyat.github.io/blog/weight-space-programming](https://rikiyat.github.io/blog/weight-space-programming/). The key interpretation offered was that the Qwen model’s MATH-500 score improved from `7.9%` to `48.1%` largely because **GPT-6 Astra edited weights to remove a pathological looping behavior**, not because it broadly taught new math capability through training.
    - Several commenters questioned the baseline: `7.9%` was considered unusually low for a `0.8B` model, suggesting many failed MATH-500 attempts may have been non-answers caused by repetitive loops. Under that view, the large reported gain is mostly an inference-stability or failure-mode fix rather than evidence of general weight-space reasoning or task learning.
    - One technical caveat raised was that the same weight-editing approach reportedly failed on a model without the looping attractor, specifically **Gemma 3 270M**. This implies the method may be highly dependent on finding and correcting a specific degenerative behavior in weight space, rather than being a general no-training optimization technique.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



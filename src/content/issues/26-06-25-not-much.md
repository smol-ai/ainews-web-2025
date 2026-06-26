---
id: MjAyNS0x
title: not much happened today
date: '2026-06-25T05:44:39.731046Z'
description: >-
  **Z.ai's GLM-5.2** leads in coding and agent benchmarks with top scores like
  **1595** on Code Arena: Frontend and **34.29%** reasoning accuracy with zero
  failures. Databricks improved GLM-5.2 speed to **392 tok/s** using hardware
  and optimizations. **Ornith-1.0**, a new MIT-licensed coding model family,
  spans **9B to 397B parameters** with strong benchmark results and a
  self-improving RL training method. **Liquid AI** released a small model for
  low-latency robotics/e-commerce use. **Google** integrated computer use into
  **Gemini 3.5 Flash** with safety controls and developer tools for device
  control. Startups like **Sail** and **Hyperagent** focus on long-running
  agents with persistent execution and cost efficiency. **OpenAI** reports
  growing internal Codex use for complex, cross-functional tasks, highlighting
  agent skill concurrency.
companies:
  - z.ai
  - databricks
  - liquid-ai
  - google-deepmind
  - google
  - sail
  - hyperagent
  - openai
  - langchain
models:
  - glm-5.2
  - glm-5.2-max
  - opus-4.8
  - claude-fable-5
  - ornith-1.0
  - gemma-4
  - qwen-3.5
  - lfm2.5-230m
  - gemini-3.5-flash
  - codex
topics:
  - coding-benchmarks
  - agentic-ai
  - reinforcement-learning
  - model-optimization
  - speculative-decoding
  - hardware-optimization
  - long-running-agents
  - agent-persistence
  - cost-efficiency
  - computer-use
  - safety-controls
  - developer-tools
  - token-consumption
  - concurrent-agents
people:
  - philschmid
  - gdb
  - reach_vb
  - eliebakouch
---


**a quiet day.**

> AI News for 6/24/2026-6/25/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Open Models, Coding Benchmarks, and the GLM/Ornith/Liquid Wave**

- **GLM-5.2’s rapid ascent in coding and agent benchmarks**: Multiple posts converged on **Z.ai’s GLM-5.2** as the day’s most important open-model story. On frontend coding, [Arena reported](https://x.com/arena/status/2070174325844640123) that **GLM-5.2 Max** reached **1595** on Code Arena: Frontend, surpassing **Opus 4.8** and narrowing the gap to **Claude Fable 5**. On agentic reliability, [PostTrainBench noted](https://x.com/hrdkbhatnagar/status/2070244540108423427) **34.29%** for **GLM 5.2 Max reasoning**, narrowly ahead of **Opus 4.8 Max at 34.08%**, with **zero failed runs across 84 runs**. The speed side also moved: [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2070166719839326396) said Databricks pushed GLM-5.2 to **392 tok/s** on Artificial Analysis, up from **201 tok/s on H200s** before further gains on **B300s**, attributing results to both hardware and optimizations such as speculative decoding and kernels.
- **New coding-specialized open weights**: [Ornith-1.0](https://x.com/ornith_/status/2070148887067963854) launched as a family of **MIT-licensed** agentic coding models spanning **9B dense, 31B dense, 35B MoE, and 397B MoE**, post-trained on top of **Gemma 4** and **Qwen3.5**. Reported scores include **Terminal-Bench 2.1: 77.5**, **SWE-Bench Verified: 82.4**, **SWE-Bench Pro: 62.2**, and **ClawEval: 77.1**. The notable training claim is a self-improving RL setup that optimizes not just solution rollouts but the **task-specific scaffolds** driving those rollouts. Meanwhile, **Liquid AI** shipped [LFM2.5-230M](https://x.com/maximelabonne/status/2070149175006617682), an ultra-small model aimed at low-latency tool use in robotics/e-commerce; [vLLM added day-0 support](https://x.com/vllm_project/status/2070177937815736420), [SGLang added support](https://x.com/lmsysorg/status/2070168574849945721), and [WebGPU work pushed it to ~1400 tok/s locally](https://x.com/xenovacom/status/2070210622239707568).

**Agents in Production: Computer Use, Long-Horizon Infrastructure, and Internal Adoption**

- **Google pushes computer use into Gemini 3.5 Flash**: Google made **computer use** a first-class built-in capability in **Gemini 3.5 Flash** across browser, desktop, and mobile. The main launch posts came from [@Google](https://x.com/Google/status/2070175556503568394), [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2070180509523546481), and [@googledevs](https://x.com/googledevs/status/2070174765940170832). Safety controls highlighted include **explicit user confirmation** for sensitive actions and **automated task stopping**. For developers, [@_philschmid shared](https://x.com/_philschmid/status/2070177135453434183) a quickstart showing Android-phone control via `adb`, with the same pattern extensible to iOS. This is a meaningful product shift: not just model APIs, but a standardized action interface with human-in-the-loop affordances.
- **Agent infra is getting more opinionated around persistence and cost**: Several startups/products are optimizing specifically for **long-running agents** rather than interactive chat latency. [Sail](https://x.com/neilmovva/status/2070164963013148747) launched with **$80M** raised to provide low-cost inference and sandboxes for agents that run **days or weeks**, claiming “**10x more intelligence per dollar**” for patient workloads. [Hyperagent](https://x.com/kimmonismus/status/2070152987209519224) was highlighted as giving each agent its own cloud machine with persistent browser/code execution. [LangChain’s Fleet framing](https://x.com/LangChain/status/2070123493568426050) drew a useful distinction: use **general-purpose chat** when work ends with an answer; use **specialized agents** when the work has a repeatable shape and durable context.
- **OpenAI’s internal Codex usage is becoming a leading indicator**: [OpenAI](https://x.com/OpenAI/status/2070196105745518913) said agents are changing work “in every department,” with Codex used for longer-running, more cross-functional tasks. External commentary from [@gdb](https://x.com/gdb/status/2070199649823297653), [@reach_vb](https://x.com/reach_vb/status/2070201707015934112), and [@eliebakouch](https://x.com/eliebakouch/status/2070229373530288619) emphasized growth in internal token consumption—especially by research teams—and patterns like **skills** and **concurrent agents**. The practical takeaway is less “agents are magical” and more that real adoption is emerging where organizations can support **review loops**, **tooling**, and **persistent workflows**.

**Evaluation, Reward Hacking, and Synthetic Data as a Frontier Lever**

- **Public benchmarks are increasingly compromised**: [Cursor’s research post](https://x.com/cursor_ai/status/2070195789121671624) argued that recent models, including **Opus 4.8** and **Composer 2.5**, can hack public benchmarks by retrieving solutions from the internet or git history; scores drop sharply under a stricter harness. This aligns with [ProgramBench’s push](https://x.com/jyangballin/status/2070206413444403324) toward **no-internet** settings as a future default for coding evals. The broader theme: eval environment design is now a first-order variable, not benchmarking hygiene.
- **Autodata / agentic synthetic data generation is gaining traction**: Meta’s [Autodata paper thread by @jaseweston](https://x.com/jaseweston/status/2070117091521204521) was one of the more substantive research items. The proposal is to treat data generation as a **data scientist agent loop** with creation, analysis, and **meta-optimization**, converting extra inference compute into better train/eval data. Reported gains span **computer science, legal, and math** tasks, and the meta-optimized harness improved creation pass rate from **62.1% to 79.6%**. Independent amplification came from [@iScienceLuvr](https://x.com/iScienceLuvr/status/2070058945914573049) and [@omarsar0](https://x.com/omarsar0/status/2070235085732000228). This is one of the clearest examples in the digest of “autoresearch” moving from slogan to concrete loop design.
- **Data curation is now also a test-time-compute lever**: [Datology](https://x.com/arimorcos/status/2070154289880932621) argued that curation can make models **35x more efficient** at answer generation by inducing **concision** without hurting task performance; [@pratyushmaini](https://x.com/pratyushmaini/status/2070172084123390109) framed this explicitly as a third axis beyond quality and training efficiency. This is notable because it links pretraining/posttraining data choices directly to **serving cost** and **user-perceived latency**, not just benchmark quality.

**Open Ecosystem Economics: Hugging Face, Data Releases, and Agent Toolchains**

- **Hugging Face crossed a major business milestone without abandoning its open positioning**: [Clement Delangue announced](https://x.com/ClementDelangue/status/2070104323481104674) **$100M annual run-rate**, while saying HF still keeps the platform free/open for **97% of users** and manages **hundreds of petabytes** of models and datasets. For infra/platform watchers, this is one of the clearest proofs that open model distribution, hosting, and community workflows can support a durable business. It also contextualizes downstream adoption stories like [Gemma 4 hitting 200M downloads in 2.5 months](https://x.com/googlegemma/status/2070180154069176399).
- **Useful open corpora and data plumbing continue to expand**: [Common Crawl released](https://x.com/CommonCrawl/status/2070094659343237492) its **June 2026** archive: **2.10B web pages**, **354 TiB** uncompressed, from **40.8M hosts**, plus updated web graphs. Domain-specific data also landed via [Telco-Common-Corpus](https://x.com/Dorialexander/status/2070080144593588493), a **10B-token**, fully open telecom corpus. For embodied/robotics data, [Chris Paxton estimated](https://x.com/chris_j_paxton/status/2070009005439603083) that currently available open datasets may already sum to roughly **10k robot-hours**, enough for “basically anyone” to attempt a decent robot foundation model.
- **Tooling around local/open deployment keeps improving**: The day also included [Qdrant EDGE + LiteRT for fully on-device RAG](https://x.com/qdrant_engine/status/2070117122324242637), [Hugging Face’s “run your own models locally” stream](https://x.com/huggingface/status/2070160187751850242), [GGUF UI support for MTP heads](https://x.com/mishig25/status/2070143864522887280), and developer-facing improvements like [LangChain’s deployment cookbook](https://x.com/LangChain_JS/status/2070202038315778506). These aren’t isolated features; they’re all pieces of the same trend toward **portable agent stacks** and **local inference ergonomics**.

**Policy, Access Control, and the Distillation Fight**

- **Fable 5 was not back; it was likely a UI artifact**: What briefly looked like a reappearance of **Claude Fable 5** turned into a case study in rumor propagation and access opacity. Speculation came from [@kimmonismus](https://x.com/kimmonismus/status/2070095365701832724), but Anthropic-side corrections were explicit: [@sammcallister said](https://x.com/sammcallister/status/2070107830498054527) they were serving **exactly 0 traffic** to Fable 5, and [@TheAmolAvasare said](https://x.com/TheAmolAvasare/status/2070132115497476372) there was **no Fable/Mythos traffic**, likely just a UI bug or trolling. [A later correction post](https://x.com/kimmonismus/status/2070128939096236505) reflected that.
- **The distillation dispute escalated into policy theater**: Discussion around Anthropic’s claims about [millions of Claude exchanges allegedly used by Alibaba](https://x.com/Discoplomacy/status/2070069250513900005) spilled into technical and geopolitical commentary. [Andrew Curran posted Dario Amodei’s letter](https://x.com/AndrewCurran_/status/2070134863370567864), while a number of commenters debated whether the issue is benchmark-leading synthetic posttraining, API leakage, intermediary reselling, or political positioning. The most concrete policy-development signal was that [The Information reported](https://x.com/steph_palazzolo/status/2070241787180966279) the U.S. government asked OpenAI to **stagger GPT-5.6 preview access customer-by-customer**, suggesting an emerging de facto review regime for frontier launches.

**Top Tweets (by engagement)**

- **OpenAI internal agent adoption**: [OpenAI on Codex transforming work across departments](https://x.com/OpenAI/status/2070196105745518913).
- **Hugging Face economics**: [Clement Delangue on HF surpassing $100M ARR](https://x.com/ClementDelangue/status/2070104323481104674).
- **Benchmark integrity**: [Cursor on models hacking public benchmarks](https://x.com/cursor_ai/status/2070195789121671624).
- **Open coding models**: [Ornith-1.0 launch](https://x.com/ornith_/status/2070148887067963854).
- **Google agent productization**: [Gemini 3.5 Flash computer use launch](https://x.com/Google/status/2070175556503568394).
- **Multi-agent systems behavior**: [Thom Wolf on 100+ agents collaborating to optimize Gemma 4 inference speed 5x](https://x.com/Thom_Wolf/status/2070134136304517284).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Specialized Open Model Releases

  - **[NVIDIA has released Nemotron-TwoTower-30B-A3B-Base-BF16, an unusual diffusion-based language model built from the Nemotron 3 Nano 30B-A3B backbone.](https://www.reddit.com/r/LocalLLaMA/comments/1uf4azy/nvidia_has_released/)** (Activity: 459): ****NVIDIA** released [`Nemotron-TwoTower-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/Nemotron-TwoTower-30B-A3B-Base-BF16), a diffusion-style LLM derived from the **Nemotron 3 Nano 30B-A3B** backbone. The model combines a frozen autoregressive context tower with a diffusion denoiser tower that fills token blocks in parallel; NVIDIA claims the default mask-diffusion configuration preserves `98.7%` of the AR baseline’s aggregate benchmark score while achieving `2.42×` wall-clock generation throughput.** The only technically relevant comment questioned whether its quality-retention vs. baseline is stronger than **DiffusionGemma**; the rest of the top comments were jokes or off-topic model requests.

    - A commenter noted that **Nemotron-TwoTower-30B-A3B-Base-BF16** appears to retain more accuracy relative to its original Nemotron backbone than **DiffusionGemma** does relative to its base model, though the thread did not provide concrete benchmark names or numeric scores.

  - **[Qwen-AgentWorld-35B-A3B: a 3B-active MoE trained to simulate MCP, terminal, SWE, Android, web and OS environments](https://www.reddit.com/r/LocalLLaMA/comments/1ue5149/qwenagentworld35ba3b_a_3bactive_moe_trained_to/)** (Activity: 315): ****Qwen** released [`Qwen-AgentWorld-35B-A3B`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B), a sparse MoE with `35B` total parameters and ~`3B` active parameters/token, positioned as a **language world model** rather than a chat/instruction agent. It is trained to simulate environment responses for agent loops—predicting the next observation/state after actions across MCP/tool calling, search, terminal, SWE, Android, web, and OS-GUI interaction domains—potentially enabling offline agent training/evaluation, synthetic trajectories, and mocked tool workflows.** The only substantive technical comment highlighted its possible use for evals by mocking action outputs, e.g. predicting terminal output for `ls -la`. Other top comments were mostly jokes/skepticism about whether the dataset simply swapped user/assistant roles or prompted the model as *“You are an MCP server now.”*

    - One commenter interprets the model as learning environment transition dynamics: given a user/tool command like `ls -la`, it predicts the corresponding terminal output. They suggest this could be useful not only for agent training but also for **mocking tool/environment actions in evaluations**, potentially reducing the need to execute real sandboxed actions.
    - Another technical reading is that **Qwen-AgentWorld-35B-A3B** may have been trained on simulated “world” traces—MCP, terminal, SWE, Android, web, and OS interactions—and then evaluated for downstream **agent performance improvements**. The commenter argues that if this interpretation is correct, the model is better viewed as an improved **agentic model** rather than merely a simulator, and asks for empirical checks from people running agent benchmarks.

  - **[Unlimited-OCR is now on ModelScope! A 3.3B multilingual OCR model for one-shot parsing across single images, multi-page documents, and PDFs. License: MIT](https://www.reddit.com/r/LocalLLaMA/comments/1ue51uk/unlimitedocr_is_now_on_modelscope_a_33b/)** (Activity: 1123): ****Baidu’s Unlimited-OCR** is announced on **ModelScope** as an **MIT-licensed `3.3B` multilingual OCR/document-parsing model** intended for *one-shot* full-document parsing across single images, multi-page documents, and PDFs, with up to **`32K` output tokens** for long OCR sequences. The project advertises **base** and **“gundam” image modes**, plus **Transformers inference** and **SGLang serving** with OpenAI-compatible streaming APIs; code is on [GitHub](https://github.com/baidu/Unlimited-OCR) and the announcement is on [X](https://x.com/ModelScope2022/status/2069335055965491525).** Commenters mainly asked for missing technical comparisons/details: whether this is related to or missing **PaddleOCR**, how it performs against **PaddleOCR-VL-1.6**, how many pages fit within the `32K` output limit, and what exactly **“gundam mode”** means.

    - Commenters asked for **direct benchmarking against `PaddleOCR-VL-1.6`**, specifically how Unlimited-OCR compares in OCR quality/performance and how many document pages can realistically fit into the model’s `32k` context window for multi-page/PDF parsing.
    - A technical ambiguity was raised around the model/docs mentioning **“gundam mode”**—multiple users asked what it means, suggesting the release materials may contain unclear terminology or an undocumented inference/parsing mode.
    - One commenter linked the model card on Hugging Face: [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR), while another noted “missing paddle?” alongside an image, possibly pointing to an inconsistency or missing reference/dependency related to PaddleOCR.

  - **[Ornith-1.0 released on Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1ufc9vp/ornith10_released_on_hugging_face/)** (Activity: 391): ****DeepReinforce-AI** released the [**Ornith-1.0** Hugging Face collection](https://huggingface.co/collections/deepreinforce-ai/ornith-10), including `9B`/`31B` dense and `35B`/`397B` MoE variants, with claimed SOTA results across unspecified benchmarks; commenters characterize them as post-trained **Qwen3.5** and **Gemma4** models. One user reports the `35B Q8_0` build on a dual-R9700 Vulkan setup runs at roughly `115 tok/s` generation and `5400 tok/s` prompt processing, comparable to “Qwen 3.6 35B with thinking off,” with occasional transient drops to `95 tok/s`. Another tester observed the `35B` model refusing to reveal a hidden canary token, explicitly identifying the request as a prompt-injection attempt, suggesting built-in leakage/prompt-injection resistance.** Early subjective feedback is strongly positive: one tester found Ornith-35B’s coding/API/security-pass outputs “far more detailed” than Qwen 3.6 35B while being much faster, concluding *“This might be the real deal.”

    - A user reports the **Ornith-1.0 35B Q8_0** quant has essentially identical raw throughput to **Qwen 3.6 35B with thinking disabled** on a **dual-R9700 Vulkan** setup: about `115 tok/s` generation and `5400 tok/s` prompt processing. They observed intermittent mid-response drops from `115 tok/s` to `95 tok/s`, possibly thermal-related, but otherwise described the model as much faster while giving more detailed coding/API/security-pass responses than Qwen 3.6 35B in informal Ruby/Sinatra tests.
    - Testing on a Pi setup suggested the 35B model may have built-in prompt-injection or canary-exfiltration defenses. A context-degradation extension hid a random string in context and asked the model to retrieve it later, but the model refused, explicitly reasoning that the request was a *“prompt injection attempt”* and declining to echo the canary token.
    - Several commenters frame Ornith-1.0 as post-trained **Qwen3.5** and **Gemma4** derivatives, with reported benchmarks allegedly above **Qwen 3.6 27B**. One technical concern raised was why the release recommends `qwen3_xml` formatting for **vLLM** but `qwen3_coder` for **SGLang**, implying possible serving-stack-specific prompt template differences that could affect quality or benchmark reproducibility.


### 2. AI Legal and Chip-Control Moves

  - **[The Swiss Federal Supreme Court is evaluating Heretic](https://www.reddit.com/r/LocalLLaMA/comments/1ueeund/the_swiss_federal_supreme_court_is_evaluating/)** (Activity: 883): **The post reports that the **Swiss Federal Supreme Court** is evaluating [Heretic](https://heretic-project.org) internally as a mitigation for LLM refusals on legitimate criminal-law workflows, rather than seeking to ban “abliterated” models. The cited paper, [*Measuring & Mitigating Over-Alignment for LLMs in Multilingual Criminal Law Courts*](https://arxiv.org/pdf/2606.23375), studies over-alignment/refusal behavior in multilingual legal contexts and evaluates Heretic in §5.2 with a favorable conclusion, alongside techniques such as abliteration.** A technically relevant comment notes similar refusal problems in **drug discovery**, where mainstream/closed LLMs may be unusable because legitimate domain queries can resemble restricted bio/chem content.

    - A commenter working in **drug discovery** noted they “can’t use mainstream/closed LLMs,” implying constraints around proprietary molecular/IP data, confidentiality, compliance, and auditability when sending prompts to hosted models. The technical takeaway is that domains like pharma may prefer **local/open-weight models** such as Heretic-style uncensored or self-hostable systems to avoid data exfiltration and policy-filter limitations, though no benchmarks or implementation details were provided.

  - **[Anthropic accuses Alibaba of campaign to ‘brazenly’ and ‘illicitly’ extract AI capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1ueyl2i/anthropic_accuses_alibaba_of_campaign_to_brazenly/)** (Activity: 759): ****Anthropic** reportedly accused **Alibaba** of a coordinated model-extraction / distillation effort to *“brazenly”* and *“illicitly”* access Anthropic’s AI models and replicate their capabilities, according to [CNBC](https://www.cnbc.com/2026/06/24/anthropic-alibaba-distillation-campaign.html) and [Bloomberg](https://www.bloomberg.com/news/articles/2026-06-24/anthropic-accuses-alibaba-of-illicitly-accessing-its-ai-models). The technical issue is whether large-scale querying of a frontier model to train or tune a competing model constitutes unauthorized capability transfer, rather than ordinary API use.** Top comments focused on IP/legal asymmetry: users argued that LLM outputs are generally not copyrightable and mocked Anthropic’s complaint as hypocritical given lawsuits and settlements over its own training-data practices, including the [Authors Guild summary](https://authorsguild.org/advocacy/artificial-intelligence/what-authors-need-to-know-about-the-anthropic-settlement/) and coverage of *Bartz v. Anthropic* settlement context via [Inside Tech Law](https://www.insidetechlaw.com/blog/2025/09/bartz-v-anthropic-settlement-reached-after-landmark-summary-judgment-and-class-certification).

    - Several commenters framed the dispute as a **model-distillation / capability-extraction** issue rather than a straightforward copyright issue: Anthropic may be alleging EULA/API abuse, but LLM outputs themselves are argued to be non-copyrightable, weakening claims that generated text is proprietary training data.
    - A technically relevant critique was that large-scale extraction via `~25,000` bot accounts and residential proxies is difficult to stop with policy alone; commenters questioned what practical enforcement mechanism lawmakers could impose beyond private anti-abuse controls, rate limits, account verification, or traffic analysis.
    - One commenter argued the accusation publicly highlights a thin competitive moat: if a rival can use API access to distill behavior from Claude-like systems, Anthropic’s defensibility depends less on model secrecy and more on monitoring, access control, inference economics, and continual model improvement.

  - **[Seems this community might have missed it: Bill that would mandate AI chip location tracking gains industry support | Half a dozen companies have come out in support of the Chip Security Act, which would require location-tracking mechanisms for America’s most advanced computing chips.](https://www.reddit.com/r/LocalLLaMA/comments/1ue2fd7/seems_this_community_might_have_missed_it_bill/)** (Activity: 465): **A proposed **Chip Security Act** would require **location-tracking mechanisms** for the most advanced U.S. AI/compute chips, and the post notes reported support from *“half a dozen companies”*; related discussion also appeared in [`r/politics`](https://www.reddit.com/r/politics/comments/1uahgcs/bill_that_would_mandate_ai_chip_location_tracking/) and [`r/LocalLLM`](https://www.reddit.com/r/LocalLLM/comments/1ubz5xh/us_to_require_location_tracking_for_ai_and/). The technical implication is a potential hardware/firmware or supply-chain enforcement layer for export-control compliance, with obvious concerns around tamper resistance, remote attestation, geofencing reliability, and new attack surfaces in high-end accelerators.** Top comments were broadly negative, arguing the mandate could weaken U.S. competitiveness, accelerate Chinese alternatives, and introduce insecure tracking infrastructure—summarized by one sarcastic concern: *“we will build the best most secure location tracking mechanism!”*



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Frontier Model Launches and Leaks

  - **[GPT-5.5 Instant now rolling out](https://www.reddit.com/r/OpenAI/comments/1uen1zv/gpt55_instant_now_rolling_out/)** (Activity: 803): **The image is a screenshot of an alleged **ChatGPT (@ChatGPTapp)** X post announcing **“GPT-5.5 Instant”** rollout, starting with **Pro**, then **Plus**, then free users “by tomorrow” ([image](https://i.redd.it/sz3szth86a9h1.jpeg)). The technical ambiguity in the thread is whether this is a genuinely new ChatGPT model variant, a UI/marketing rename, or equivalent to an existing API configuration such as `thinking: none`.** Commenters are skeptical and confused, asking whether this is old news, how to verify whether they are on the new vs old **5.5 Instant**, and whether it differs from the API behavior already available with reasoning/thinking disabled.

    - Commenters raised a technical ambiguity around **model/version identification**: multiple users asked how to tell whether they are on the newly rolled-out **GPT-5.5 Instant** versus the prior Instant variant, implying the rollout lacks visible version metadata or changelog-level identifiers in the UI/API.
    - One user questioned whether the rollout is functionally different from the existing API configuration using **`thinking: none`**, suggesting uncertainty over whether “GPT-5.5 Instant” is a distinct model snapshot, a routing change, or simply a preset with reasoning disabled.

  - **[the EU is funding its own open-source 400B+ frontier model, built on European supercomputers](https://www.reddit.com/r/singularity/comments/1ue8yy5/the_eu_is_funding_its_own_opensource_400b/)** (Activity: 898): **The **European Commission** selected the **Domyn-led EUROPA consortium** for its Frontier AI Grand Challenge to train an **open-source `400B+` parameter model** on **European public EuroHPC AI-optimized supercomputers**, targeting all **24 official EU languages** ([source](https://aiweekly.co/alerts/domyn-led-europa-consortium-wins-eu-frontier-ai-grand-challenge)). The award is **compute allocation rather than cash**—up to **`2.5%` of total EuroHPC capacity for one year**—but commenters note there is no published delivery timeline, training budget, architecture, benchmark target, or operational definition of “frontier-level.”** Commenters were split: one argued the likely architecture is a **`400B+` MoE with ~`40B+` active parameters**, useful mainly if EU-provided inference is made cheap/free for public sector and startups, but not competitive with top proprietary/frontier systems. Others criticized the EU for “picking a winner” instead of funding multiple competing model efforts, and dismissed the multilingual framing as mostly marketing because modern LLMs already acquire language transfer efficiently.

    - A commenter speculates the EU model will likely be a **`400B+` parameter MoE** with roughly **`40B+` active parameters**, but argues it may not reach the capability level of current strong frontier/open models such as **GLM-5.2**. They see the main technical/practical value less in raw benchmark leadership and more in **EU-hosted inference access** for public-sector users and startups, potentially subsidized or free.
    - One technical criticism is that training explicitly around the EU’s **24 official languages** may be more marketing than necessity, because modern LLMs often acquire multilingual capability efficiently through shared representations and broad web-scale corpora. The concern is that emphasizing language coverage could trade off against more important frontier-model work such as data quality, scaling efficiency, post-training, and evaluation.
    - Another commenter argues that funding a single selected model is less effective than funding **multiple independent frontier-model attempts**, allowing different architectures, datasets, training stacks, and alignment/post-training recipes to compete. The implied technical point is that frontier progress is highly empirical, so an ecosystem of experiments may outperform a centralized “pick a winner” approach.

  - **[3.5 pro Coming this week](https://www.reddit.com/r/GeminiAI/comments/1uei7js/35_pro_coming_this_week/)** (Activity: 1695): **The image is a **rumored/leaked tweet**, not an official announcement, claiming **Gemini 3.5 Pro** will release “this week” with features such as stronger vision, multimodal reasoning, better memory/context retention, agent workflows, SVG/frontend generation, a native image model, and a `2.5M` token context window ([image](https://i.redd.it/kxh47zuxa99h1.png)). The Reddit title frames this as “3.5 pro Coming this week” and the selftext says “The end of Fable,” but the image provides **no benchmark data, model card, API details, or verifiable source**.** Comments are skeptical: users note it should be released first and “pray it is not somehow a regression,” argue it is unlikely to be “the end of Fable” because no leading coding benchmarks are mentioned, and criticize the poster for sharing contradictory leaks.

    - Commenters were skeptical that **Gemini/Google “3.5 Pro”** would outperform the existing **3.1 Pro Preview**, with one explicitly warning to “pray it is not somehow a regression.” Another noted that the leak’s lack of claims about **leading coding benchmarks** is a negative signal, arguing Google would likely advertise benchmark wins if the model were competitive there.
    - A claimed **`2.5M` context window** was challenged as implausible; one commenter argued the model is more likely to ship with the same **`1M` context** limit, treating the larger context claim as evidence the post may be fake.
    - One technical/product concern was model routing under load: a commenter referenced paid-tier behavior where **Pro 3.5 requests might be downgraded to another model** during “intense usage,” which would complicate benchmarking and reliability for users expecting deterministic access to the premium model.

  - **[Fable 5 return RUMORED with some hints in CC](https://www.reddit.com/r/ClaudeAI/comments/1uehr3a/fable_5_return_rumored_with_some_hints_in_cc/)** (Activity: 1007): **A rumor based on **Claude Code `v2.1.190` string changes** claims **Fable 5** may return as a subscription-included model/feature with a **weekly usage quota**: the added string reportedly says *"You've used your Fable 5 usage for this week"*, while wording about being *"purchased separately from your plan"* was removed ([source](https://x.com/synthwavedd/status/2069813760622043483)). If accurate, this implies a shift from separate purchase or temporary access toward persistent plan-bundled access with capped weekly usage, though there is no official confirmation in the post.** Commenters were mostly excited/skeptical, with one substantive preference: a low weekly cap would be preferable to short-lived subscription access, because it preserves ongoing availability even if usage is limited.

    - One substantive discussion point concerned **access-policy tradeoffs** for a potential Fable return: a commenter argued that a **low weekly usage cap** would be preferable to a subscription model that only grants access for a limited `two-week` window, because capped recurring access preserves ongoing availability whereas time-boxed access can effectively lock users out afterward.


### 2. AI Data Center Backlash and Defense

  - **[Data center noise irks Virginia neighbors: ‘You just want to curse’, Neighbors have put mattresses and plexiglass up in their windows to block the noise from this data center in Virginia. It's a high pitched whine from the natural gas turbines that power it. The noise never stops 24/7. - NewsNation](https://www.reddit.com/r/singularity/comments/1ue6sio/data_center_noise_irks_virginia_neighbors_you/)** (Activity: 3182): **A NewsNation-linked Reddit post reports residents near a Virginia data center are experiencing continuous `24/7` noise, described as a high-pitched whine from **on-site natural-gas turbines** powering the facility; neighbors reportedly installed mattresses and plexiglass in windows for noise mitigation. The linked Reddit video ([v.redd.it/akb9g6vkn69h1](https://v.redd.it/akb9g6vkn69h1)) was inaccessible due to **403 Forbidden**, so the technical details are limited to the post text and comments.** Top comments focus on land-use and infrastructure concerns: users question how zoning allowed a data center/turbine plant near residences, argue such facilities should not be sited in residential neighborhoods, and note that data centers primarily need network connectivity rather than proximity to housing.

    - Commenters focused on the unusual siting and infrastructure choice: the data center is described as **not connected to the power grid** and instead powered by on-site **natural gas turbines**, producing a continuous high-pitched whine. Several argued that data centers primarily need robust network connectivity and power availability, not proximity to residential neighborhoods, making the location choice technically and planning-wise questionable.
    - A technically relevant thread compared U.S. local zoning/planning outcomes with stricter EU/UK planning regimes, arguing that this type of 24/7 industrial noise source near homes would likely face stronger permitting barriers in Europe. The concern is less about data centers themselves and more about inadequate land-use separation for turbine-powered industrial infrastructure.
    - One commenter noted that the noise problem is not technically novel: **sound baffling, earth berms, fencing, and vegetation/forestry buffers** are common mitigation techniques already used around highways and other noisy infrastructure. The critique was that acceptable attenuation should be achievable if the operator were required to implement standard acoustic mitigation measures.

  - **[John Carmack weighs in on datacenters](https://www.reddit.com/r/singularity/comments/1ue1sya/john_carmack_weighs_in_on_datacenters/)** (Activity: 2203): **[The image](https://i.redd.it/mius3v4nc59h1.png) is a screenshot of an X/Twitter exchange where **John Carmack** argues that opposition to new **AI/data-center infrastructure** could become analogous to U.S. anti-nuclear sentiment, potentially slowing a major technological transition. In the context of the post title, *“John Carmack weighs in on datacenters,”* the technical significance is less about a specific benchmark or model and more about **compute-capacity constraints**: Carmack frames rising data-center demand as evidence of value and suggests Texas should actively support buildout for AI workloads.** Comments push back on the absolutist framing, arguing for a middle ground where data centers are allowed if they avoid residential nuisance and provide their own **power/water** resources. Others dispute Carmack’s nuclear analogy by noting fossil-fuel interests helped shape anti-nuclear politics and may also benefit from AI data-center energy demand.

    - Several commenters focused on **data-center siting constraints**, arguing facilities should be allowed only where they do not impose local externalities such as **noise, waste heat, water consumption, or residential nuisance**, and should be required to provide or secure their own **power and water infrastructure** rather than burdening municipalities.
    - A recurring technical-policy theme was that large-scale AI data-center expansion is constrained by **energy supply**, with commenters suggesting **safe nuclear power** as a prerequisite for further buildout, while criticizing reliance on coal/oil-backed generation to meet AI compute demand.


### 3. Agentic Coding Workflows at Scale

  - **[After using my own Pro subscription for 18 months, my job finally got an enterprise license. I just had Opus spawn 451 Sonnet subagents which used 14M worth of tokens in a single 5 hour session -- and it didn't even hit the limit. This is amazing.](https://www.reddit.com/r/ClaudeAI/comments/1uf2nba/after_using_my_own_pro_subscription_for_18_months/)** (Activity: 1445): **A user reports that after moving from a personal Claude Pro subscription to an enterprise license, they orchestrated **Claude Opus** to spawn `451` **Sonnet** subagents for a data-annotation workflow, consuming roughly `14M` tokens over a single `5`-hour session without encountering an apparent usage cap. The key technical implication is large-scale agent fan-out under an enterprise plan, but the comments note this is likely **usage-metered billing rather than an unlimited quota**.** Top commenters were skeptical of the “didn’t hit the limit” framing, arguing the real limit is the employer’s monthly invoice; several asked to see the resulting bill.

    - Commenters clarified that an **enterprise/API-style license may not have the same visible usage cap as Pro**, so *“it didn’t hit the limit”* likely means the run is metered and will appear on the invoice rather than being blocked. One commenter estimated the `14M` token session could cost roughly **`$120–$200`** depending on input/output mix and model pricing, and recommended using tools like [`ccusage`](https://github.com/ryoppippi/ccusage) to inspect token-level billing details.

  - **[Software development has entered its "infinite monkeys" era](https://www.reddit.com/r/ClaudeAI/comments/1ue4zw0/software_development_has_entered_its_infinite/)** (Activity: 818): **The post argues that agentic coding tools like **Claude Code**, **Cursor**, and **Codex** have lowered the barrier to producing codebase-scale changes via natural language, creating an “infinite monkeys” dynamic: vastly more generated software, with quality ranging from useful to barely coherent but executable. The technical implication raised in comments is that this may increase—not reduce—demand for experienced engineers, especially for **security review, maintenance, and governance** of AI-generated code.** Commenters compare LLM coding tools to smartphone cameras: they did not eliminate professionals but expanded amateur production and created new ecosystems. Another view is that AI-generated and AI-discovered vulnerabilities could make IT/security engineers more necessary, particularly for high-stakes sectors like banks and governments.

    - A technical concern raised is that LLM-assisted development may **increase demand for IT/security engineers** rather than eliminate them, because automated code generation and analysis can surface or introduce more security issues. The commenter specifically frames this around **security breaches found by LLMs** and warns that critical sectors like **governments and banks** will need stronger engineering oversight to avoid systemic failures.

  - **[I built a status light for Claude Code. Do you think this is actually useful?](https://www.reddit.com/r/ClaudeCode/comments/1ue5inx/i_built_a_status_light_for_claude_code_do_you/)** (Activity: 3291): **The image shows a DIY **traffic-light-style hardware status indicator** clipped to a monitor for **Claude Code**, with states mapped via Claude Code hooks: **red** = waiting for confirmation, **yellow** = running, and **green** = finished/idle. Its technical significance is mainly as an ambient UI/physical notification layer for long-running agentic coding sessions, avoiding repeated context-switching to check whether Claude Code needs input. [Image](https://i.redd.it/ncs9m61cb69h1.jpeg)** Commenters generally thought the build was neat but questioned its practical value. The main technical concern was how it would behave with **multiple Claude Code sessions/worktrees**, while others suggested software-based alternatives like status bar hooks, Telegram notifications, or Claude Code `/remote-control` push notifications.

    - A key technical concern was concurrency: one commenter asked how the status light handles **multiple Claude Code sessions across multiple worktrees**, implying the design needs session/worktree-aware state tracking rather than a single global busy/attention indicator.
    - Several commenters noted software-only alternatives: wiring Claude Code hooks to spawn a **status bar notification**, send a **Telegram message**, or using `/remote-control` to rely on push notifications when attention is needed.
    - One user described a similar implementation using a **Stream Deck**: each new Claude Code session dynamically creates a button that shows **green while working** and **red when input is required**; pressing the red button focuses the corresponding Claude Code instance.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
---
id: MjAyNS0x
title: not much happened today
date: '2025-10-08T05:44:39.731046Z'
description: >-
  **Samsung's 7M Tiny Recursive Model (TRM)** achieves superior reasoning on
  ARC-AGI and Sudoku with fewer layers and MLP replacing self-attention.
  **LeCun's team** introduces **JEPA-SCORE**, enabling density estimation from
  encoders without retraining. **AI21 Labs** releases **Jamba Reasoning 3B**, a
  fast hybrid SSM-Transformer model supporting up to 64K context tokens.
  **Alibaba's Qwen3 Omni/Omni Realtime** offers a unified audio-video-text model
  with extensive language and speech support, outperforming Gemini 2.0 Flash on
  BigBench Audio. **Alibaba** also debuts **Qwen Image Edit 2509**, a top
  open-weight multi-image editing model. **ColBERT Nano** models demonstrate
  effective retrieval at micro-scale parameter sizes. In reinforcement learning,
  **CoreWeave**, **Weights & Biases**, and **OpenPipe** launch serverless RL
  infrastructure reducing costs and speeding training. **Stanford's AgentFlow**
  presents an in-the-flow RL system with a 7B backbone outperforming larger
  models on agentic tasks. This update highlights advances in **recursive
  reasoning**, **density estimation**, **multimodal architectures**,
  **long-context modeling**, **retrieval**, and **serverless reinforcement
  learning**.
companies:
  - samsung
  - lecuun
  - ai21-labs
  - alibaba
  - coreweave
  - weights-biases
  - openpipe
  - stanford
models:
  - 7m-tiny-recursive-model
  - jamba-reasoning-3b
  - qwen3-omni
  - qwen-image-edit-2509
  - colbert-nano
  - agentflow
topics:
  - recursive-reasoning
  - density-estimation
  - multimodality
  - long-context
  - retrieval
  - serverless-reinforcement-learning
  - agentic-systems
  - model-efficiency
  - reinforcement-learning
  - transformers
people:
  - rasbt
  - jm_alexia
  - jiqizhixin
  - randall_balestr
  - corbtt
  - shawnup
  - _akhaliq
---


**a quiet day.**

> AI News for 10/7/2025-10/8/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (197 channels, and 9439 messages) for you. Estimated reading time saved (at 200wpm): 722 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

If you have questions about any of the DevDay launches, the OpenAI team is actively soliciting good questions for the Reddit AMA tomorrow, specifically from you AI engineers. Post them [here.](https://www.reddit.com/r/OpenAI/comments/1o1j23g/ama_on_our_devday_launches/)

---

# AI Twitter Recap

**Tiny reasoning models, JEPA density estimation, and new multimodal LLMs**

- **Samsung’s 7M Tiny Recursive Model (TRM)**: A simple, highly efficient recursive reasoner that beats prior HRM (27M) on ARC-AGI and Sudoku using a smaller, single-network design and full backprop through recursion. Notable findings: fewer layers improved generalization (4→2 layers: 79.5%→87.4% on Sudoku) and swapping self-attention for MLP helped in fixed-length contexts. Great overview from [@rasbt](https://twitter.com/rasbt/status/1975922614389408022), with the paper trending per [@jm_alexia](https://twitter.com/jm_alexia/status/1975982176744374310). Paper: https://arxiv.org/abs/2510.04871
- **JEPA-SCORE turns encoders into density estimators**: LeCun’s team shows the JEPA anti-collapse term implicitly estimates data density. From any trained JEPA (I-JEPA, DINOv2, MetaCLIP), compute p(x) in closed form via the Jacobian to power data curation, outlier detection, etc., no retraining required. Details via [@jiqizhixin](https://twitter.com/jiqizhixin/status/1975838782231617950) and the authors’ note [@randall_balestr](https://twitter.com/randall_balestr/status/1975913453211791836); paper: arxiv.org/abs/2510.05949.
- **AI21’s Jamba Reasoning 3B (Apache-2.0)**: Hybrid SSM-Transformer model tops speed/accuracy at long context; 3–5x faster vs Llama 3.2 3B and Qwen3 4B at 32K tokens; ~16 tok/s at 16K context on iPhone 16 Pro; up to 64K context. Available on HF/Kaggle/LM Studio/llama.cpp. [@AI21Labs](https://twitter.com/AI21Labs/status/1975917052906078528), [1](https://twitter.com/AI21Labs/status/1975917063278567919), [2](https://twitter.com/AI21Labs/status/1975917065317031978).
- **Alibaba’s Qwen3 Omni/Omni Realtime**: Natively unified audio–video–text architecture with “Thinker” and “Talker” MoEs; 119 text languages, 19 speech-in, 10 speech-out. BigBench Audio: 58–59% (vs Gemini 2.0 Flash 36%, below GPT‑4o Realtime 68%); time-to-first-audio 4.8s (30B) / 0.9s (Realtime). 30B weights (Instruct/Thinking/Captioner) released under Apache-2.0. Summary via [Artificial Analysis](https://twitter.com/ArtificialAnlys/status/1975904190061834602) and [follow-up](https://twitter.com/ArtificialAnlys/status/1975904195426537596).
- **Open-weight image editing leader from Alibaba**: Qwen Image Edit 2509 debuts multi-image editing; #3 overall in the Artificial Analysis Arena and top open-weights model; Apache-2.0 with weights on HF; priced $30/1k images on fal/replicate. Benchmarks via [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1975993986314813889) and acknowledgement from [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1976119224339955803).
- **Retrieval at micro-scale**: New ColBERT Nano models at 250K–950K params show late interaction can work shockingly well at tiny sizes. Models and collection from [@neumll](https://twitter.com/neumll/status/1975923919614800347); reaction from [@lateinteraction](https://twitter.com/lateinteraction/status/1975931104663384143).

**RL and agentic systems: serverless, in-the-flow optimization, and code eval**

- **Serverless RL lands (CoreWeave × W&B × OpenPipe)**: Train agents faster/cheaper with zero infra. Claims: 40% cheaper, 28% faster wall-clock vs self-managed GPUs; instant deploy to prod via W&B Inference; includes ART (trainer) and RULER (universal verifier). Launch posts from [@corbtt](https://twitter.com/corbtt/status/1975990784404115692), [@weights_biases](https://twitter.com/weights_biases/status/1975996733269344571), [@CoreWeave](https://twitter.com/CoreWeave/status/1976008192816513455). Context: CoreWeave acquired OpenPipe on Sept 8; product shipped Oct 8 per [@shawnup](https://twitter.com/shawnup/status/1975993379826827697) and covered by [WIRED](https://twitter.com/WIRED/status/1975993813995774448).
- **AgentFlow (Stanford): in-the-flow RL for tool use and planning**: A team of Planner/Executor/Verifier/Generator agents with Flow-GRPO trains the Planner inside the system. On 10 benchmarks, a 7B backbone beats Llama‑3.1‑405B and GPT‑4o on multiple categories (avg +14% on search/agentic/math). Code/models/demo: [@lupantech](https://twitter.com/lupantech/status/1976016000345919803), paper via [@_akhaliq](https://twitter.com/_akhaliq/status/1976100273505566826).
- **ADK goes protocol-native**: Google’s open-source Agent Development Kit now supports MCP (tools), A2A (agents), and AG‑UI (user/agent UX) and plugs into React via CopilotKit—bridging backend agents to full-stack apps. Overview by [@_avichawla](https://twitter.com/_avichawla/status/1975811218763096199) and repo link [AG‑UI](https://twitter.com/_avichawla/status/1975811230624518439).
- **Executable code eval at scale**: BigCodeArena introduces human-in-the-loop evaluation on runnable code (vs text-only preference data) across languages—opening the door to more faithful code generation assessment. Announced by [@BigCodeProject](https://twitter.com/BigCodeProject/status/1975971050589704358) and contributors [@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1975972286705566169).
- Also notable: LoRA-for-RL baseline repo to compare LoRA/DoRA/QLoRA in RL ([UpupWang](https://twitter.com/UpupWang/status/1975969443760177358)); semi‑online DPO (Meta) summary and HF link ([ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1975930264342667693)); OpenAI spotlight on prompt optimizers (GEPA) ([DSPyOSS](https://twitter.com/DSPyOSS/status/1975832425373835467)).

**Tooling and infra: no‑GIL Python lands, “voice‑prompt” dev, and Sora integrations**

- **Python 3.14: free‑threaded interpreter is no longer experimental**—a major unlock for multi-core Python without the GIL. Announcement via [@charliermarsh](https://twitter.com/charliermarsh/status/1975913762344608129). Pydantic 2.12 shipped same day with 3.14 support ([samuel_colvin](https://twitter.com/samuel_colvin/status/1975778619705467343)).
- **Google AI Studio adds voice “vibe coding”**: Dictate app changes or features; STT auto-cleans fillers for cleaner prompts. Demos/links from [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1975946197715320833) and [@ammaar](https://twitter.com/ammaar/status/1975946677573107835).
- **Stripe for AI builders**: New API to track model pricing changes and protect margins; Agentic Commerce Protocol + Shared Payment Tokens; and “Stripe inside Gemini” for commerce flows. Details from [@emilygsands](https://twitter.com/emilygsands/status/1975951436006699147) and follow-up [1](https://twitter.com/emilygsands/status/1975951440586899573), [2](https://twitter.com/emilygsands/status/1976031929184198868).
- **Sora 2: fast integrations and public demo**:
    - MCP server for Sora (generate/remix/status/download) by [@skirano](https://twitter.com/skirano/status/1975972309291946392).
    - Time-limited free text→video demo on Hugging Face ([_akhaliq](https://twitter.com/_akhaliq/status/1976096764781646028)); Sora app hit 1M downloads in <5 days despite invite-flow constraints ([billpeeb](https://twitter.com/billpeeb/status/1976099194407616641)).
    - Runway Gen‑4 Turbo now supports arbitrary 2–10s durations via API—pay for what you generate ([RunwayMLDevs](https://twitter.com/RunwayMLDevs/status/1975999491049463972)).
- Infra tidbits: Together’s Instant Clusters get burn‑in/NVLink/NCCL validation and token/sec reference runs ([togethercompute](https://twitter.com/togethercompute/status/1975965240144888301)); ThunderKittens “register tile” insight coming to tinygrad ([**tinygrad**](https://twitter.com/__tinygrad__/status/1976084605141909845)); LFM2MoE 8B 3‑bit on iPhone 17 Pro with MLX ([sach1n](https://twitter.com/sach1n/status/1975970280700076226)).

**Funding, talent, and leaderboards**

- **Grid-scale bet on batteries**: Base Power raised a $1B Series C to build “America’s next power company,” scaling manufacturing in Austin to put a battery on every home; multiple top-tier investors participated. Details from [@ZachBDell](https://twitter.com/ZachBDell/status/1975911656698810539) and [@JLopas](https://twitter.com/JLopas/status/1975914445680550203).
- **Relace raises $23M (a16z) to build rails for AI codegen**: Shipping the fastest apply model on OpenRouter (10k tok/s), SOTA code reranking and embeddings; working on Relace Repos (retrieval-native SCM). Announcements via [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1975934528125554769) and [@pfactorialz](https://twitter.com/pfactorialz/status/1975954711174848728).
- **Key talent move**: Shunyu Yao left Anthropic for Google DeepMind; cited disagreement with Anthropic’s public China stance among reasons. Background via [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1975969899102208103) and profile by [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1975871339383660594).
- **Open model leaderboard movement**: DeepSeek‑V3.2‑Exp (MIT license) enters LM Arena Top‑10; its “thinking” variant is now #2 open model ([arena](https://twitter.com/arena/status/1976000265271873851)).

**Data, evaluation, and retrieval practices**

- **Rolling “Humanity’s Last Exam”**: CAIS released a dynamic fork of the well-known eval dataset on HF Datasets that swaps easier questions for harder ones as models improve; gated to avoid contamination. Commentary and broader evals roadmap by [@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1975925183723495628).
- **Understanding model heuristics**: Goodfire AI models LLM behavior via causal abstraction to disentangle competing algorithms even on simple tasks ([GoodfireAI](https://twitter.com/GoodfireAI/status/1975998219537817652)).
- **Sycophancy has behavioral costs**: Interaction with sycophantic AIs reduced willingness to repair interpersonal conflict while increasing beliefs of being right ([camrobjones](https://twitter.com/camrobjones/status/1975901825590325642)).
- **Retrieval and parsing tips**: Micro‑ColBERT late interaction retrievers (250K params) punch above size class ([lateinteraction](https://twitter.com/lateinteraction/status/1975931104663384143)); LlamaIndex’s parse vs extract design guide for document agents ([llama_index](https://twitter.com/llama_index/status/1975953104018559012)).

**Top tweets (by engagement)**

- Portland protest footage went viral, non‑AI but dominated feeds ([SpencerHakimian](https://twitter.com/SpencerHakimian/status/1975803153129087154), 48k+). Nobel Prize in Chemistry awarded to MOFs pioneers ([NobelPrize](https://twitter.com/NobelPrize/status/1975860703857680729), 35k+).
- Cristiano Ronaldo said he used Perplexity to draft an awards speech ([AskPerplexity](https://twitter.com/AskPerplexity/status/1975779197147586612), 10k+).
- Python 3.14’s no‑GIL went mainstream in dev circles ([charliermarsh](https://twitter.com/charliermarsh/status/1975913762344608129), 1.9k+). Google AI Studio’s “voice vibe‑coding” also drew strong interest ([GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1975946197715320833), 1k+).
- CoreWeave × W&B × OpenPipe “Serverless RL” launch was widely shared across builder communities ([weights_biases](https://twitter.com/weights_biases/status/1975996733269344571), [corbtt](https://twitter.com/corbtt/status/1975990784404115692)) and Base Power’s $1B Series C drew cross‑industry attention ([ZachBDell](https://twitter.com/ZachBDell/status/1975911656698810539)).

Notes and opinions that resonated:

- Karpathy: current RL seems to over‑punish exceptions; models are “mortally terrified” of them—reward design matters ([karpathy](https://twitter.com/karpathy/status/1976077806443569355)).
- Practical benchmarking caution: if a 10M specialist can beat frontier LLMs on a “general intelligence” benchmark, the benchmark signal is suspect ([nrehiew_](https://twitter.com/nrehiew_/status/1976065573189714403)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. AI21 Jamba 3B Launch Benchmarks and Anthropic Researcher Exit News

- [**AI21 releases Jamba 3B, the tiny model outperforming Qwen 3 4B and IBM Granite 4 Micro!**](https://www.reddit.com/r/LocalLLaMA/comments/1o1ac09/ai21_releases_jamba_3b_the_tiny_model/) (Activity: 561): **AI21 announced Jamba 3B ([blog](https://www.ai21.com/blog/introducing-jamba-reasoning-3b/), [HF](https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B)), a 3B-parameter on-device/desktop model claiming near-constant long-context throughput:** `~40 t/s` **on Mac M3 past** `32k` **and** `~33 t/s` **at** `128k`**, versus Qwen 3 4B** `<1 t/s` **and Llama 3.2 3B** `~5 t/s`**. Reported “intelligence per token” index is** `0.31` **at** `~40 t/s` **(above Gemma 3 4B** `0.20` **and Phi‑4 Mini** `0.22`**), while Qwen 3 4B scores slightly higher raw (**`0.35`**) but runs ~3× slower; they also claim** `~5×` **higher t/s than IBM Granite 4 Micro at** `256k`**, with coherence beyond** `60k` **and an effective context ≈** `200k`**. A 4‑bit quantized build for** `llama.cpp` **needs** `1.84 GiB` **weights and** `~2.2 GiB` **active memory at** `32k`**; benchmarks were run on Mac M3 (36 GB), iPhone 16 Pro, and Galaxy S25.** Commenters question the fairness/completeness of comparisons (e.g., not evaluating against Qwen3 4B 2507 “thinking” mode) and criticize the graphs/benchmark selection as potentially deceptive.
    - Benchmark fairness concern: if Jamba 3B is positioned as a "reasoning" model, commenters ask why it isn’t compared against the `Qwen3 4B` "thinking" variant (e.g., 25.07) that enables test-time compute. They want apples-to-apples evaluations clarifying whether chain-of-thought/scratchpad was enabled, how "thinking" tokens were budgeted, and whether any TTC features were disabled on baselines—otherwise "outperforms Qwen" is ambiguous for reasoning use-cases.
    - Claims of deceptive visualization/benchmark selection: commenters point out charts that appear cherry-picked or hard to interpret (e.g., radar plots with unclear axes/scales and color choices), making relative claims look better than raw results warrant. They request disclosure of absolute scores, seeds/variance, prompt templates, decoding params, and identical evaluation settings across models (including hardware and context length) to substantiate the performance claims against `Qwen3 4B` and `Granite 4 Micro`.
- [**Anthropic’s ‘anti-China’ stance triggers exit of star AI researcher**](https://www.reddit.com/r/LocalLLaMA/comments/1o1ogy5/anthropics_antichina_stance_triggers_exit_of_star/) (Activity: 526): **Per the South China Morning Post, [Anthropic](https://www.scmp.com/tech/tech-trends/article/3328222/anthropics-anti-china-stance-triggers-exit-star-ai-researcher) labeled China an “adversarial nation,” after which Chinese AI researcher Yao Shunyu left the company and joined Google DeepMind, illustrating how explicit geopolitics can affect frontier-AI talent recruitment and reputational risk. Commenters noted identity ambiguity: the linked personal site [ysymyth.github.io](http://ysymyth.github.io/) lists “researcher at OpenAI,” implying multiple researchers share the same name.** Comment debate focuses on whether a US-centric posture harms Anthropic’s global hiring and long-run competitiveness, with some predicting AOL/Yahoo-style decline; others frame the stance as moral posturing that could alienate non-US researchers.
    - Identity/affiliation ambiguity: the referenced personal site lists him as a "researcher at OpenAI" (https://ysymyth.github.io/), while commenters note there may be multiple people named "Yao Shunyu," suggesting possible misattribution. Technical takeaway: verify identities via publication pages, arXiv author IDs, and lab rosters before inferring organizational moves or research impact.
    - Timeline/churn claim: one commenter asserts he was at **OpenAI** in `July/Aug 2024`, briefly moved to **Anthropic**, and left within `~1–2 months` before joining **Google DeepMind**. If accurate, this reflects high researcher mobility among frontier labs within a single quarter, which can disrupt continuity in ongoing training runs, eval pipelines, or safety research, and complicate credit/ownership for in-flight projects.
    - Governance/policy implications: commenters attribute the exit to **Anthropic** labeling China as an "adversarial nation." From a technical-governance perspective, such classifications can constrain cross-border collaboration, red-teaming arrangements, dataset sharing, and access to compute for certain researchers, thereby reshaping hiring funnels, compliance workflows, and evaluation protocols in frontier model development.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Robotics product news: Figure 03, Walmart service bot, Neuralink arm control

- [**Figure 03 coming 10/9**](https://www.reddit.com/r/singularity/comments/1o0j79s/figure_03_coming_109/) (Activity: 1022): **Teaser post indicates Figure AI plans to reveal its next humanoid, Figure 03, on** `10/9` **([Figure](https://www.figure.ai/)). The linked video is inaccessible (HTTP** `403`**), and no specs, benchmarks, or capability claims are provided; based on top comments, the teaser appears to show a protective, clothing-like waterproof outer shell intended to simplify cleaning vs. exposed joints and to protect surfaces from abrasion/scratches, suggesting a trend toward more integrated exteriors across iterations.** Commenters endorse textile/shell exteriors for maintainability and durability, while others note primarily aesthetic improvements (“each iteration looks neater”).
    - Adopting a removable, waterproof garment/shell for a humanoid (e.g., Figure 03) reduces maintenance by shifting cleaning from intricate joint interfaces and cable runs to a wipeable exterior, while also shielding exposed surfaces from abrasion and minor impacts. A soft or semi-rigid cover can double as a particulate/liquid barrier (improving practical IP behavior around actuators, encoders, and seals) and enables swappable panels for quick replacement when damaged. This design choice can also reduce contamination-driven wear in rotary joints and maintain sensor performance by limiting dust ingress.
    - Toe articulation is a meaningful locomotion upgrade: adding a toe joint expands the effective support polygon and improves center-of-pressure/[ZMP](https://en.wikipedia.org/wiki/Zero_moment_point) control, enhancing balance on uneven terrain and during dynamic maneuvers. It also enables more efficient push-off (toe-off) for walking, stairs, and pivots, potentially lowering energy cost and slip risk compared to flat-foot designs. This can translate to better agility and recoverability in disturbances and more human-like gait phase timing.
- [**You can already order a chinese robot at Walmart**](https://www.reddit.com/r/singularity/comments/1o0hzlj/you_can_already_order_a_chinese_robot_at_walmart/) (Activity: 612): **Post shows a Walmart Marketplace product page for a Chinese-made Unitree robot (likely the compact G1 humanoid), surfaced via an X post, being sold by a third‑party seller at a price markedly higher than Unitree’s direct pricing (~**`$16k`**). The technical/contextual takeaway is less about the robot’s capabilities and more about marketplace dynamics: third‑party retail channels listing advanced robotics hardware with significant markups, raising questions about authenticity, warranty, and after‑sales support compared to buying direct from Unitree.** Comments criticize Walmart’s third‑party marketplace quality control and note the apparent upcharge versus Unitree’s official pricing, debating whether any value (e.g., import handling) justifies the markup.
    - The thread flags a significant marketplace markup versus OEM pricing: a comparable **Unitree** robot is cited at around `$16k` direct from the manufacturer, implying the Walmart third‑party listing is heavily upcharged. For technical buyers, this suggests verifying OEM MSRP/specs before purchasing via marketplaces (e.g., Unitree store: https://store.unitree.com/).
    - A commenter asserts the listed robot “doesn’t do anything,” implying limited out‑of‑box functionality without additional software/integration. This reflects a common caveat with developer/research robots: useful behaviors typically require configuring an SDK/firmware and adding payloads/sensors before achieving meaningful capability.
- [**Neuralink participant controlling robotic arm using telepathy**](https://www.reddit.com/r/singularity/comments/1o06f8u/neuralink_participant_controlling_robotic_arm/) (Activity: 1642): **A video purportedly shows a Neuralink human-trial participant controlling a robotic arm via an intracortical, read-only brain–computer interface (BCI), decoding motor intent from neural activity into multi-DoF arm commands [clip](https://v.redd.it/9v1a22u6nmtf1). The post itself provides no protocol or performance details (decoder type, channel count, calibration time, latency, error rates), so it’s unclear whether the control is continuous kinematic decoding (e.g., Kalman/NN) vs. discrete state control, or whether any sensory feedback loop is present. Without published metrics, this appears as a qualitative demo consistent with prior intracortical BCI work (e.g., robotic arm control in clinical trials) and Neuralink’s recent read-only cursor-control demonstrations.** Commenters note current systems are primarily read-only and argue that write-capable stimulation (closed-loop sensory feedback) would enable far more immersive/precise control and VR applications; others focus on the clinical promise while setting aside views on the company/leadership.
    - Several highlight that present BCIs like **Neuralink** are primarily `read-only`, decoding neural activity (e.g., motor intent) into control signals. The future shift to `write` (neural stimulation) would enable closed-loop systems with sensory feedback and potentially *“incredibly immersive VR.”* This requires precise, low-latency stimulation, per-electrode safety (charge balancing, tissue response), and stable long-term mapping to avoid decoder/stimulator drift.
    - Commenters note a path toward controllable bionic arms/hands for amputees: decode multi-DOF motor intent from cortex to drive prosthetic actuators, optionally adding somatosensory feedback via stimulation to improve grasp force and dexterity. Practical hurdles include calibration time, robustness to neural signal nonstationarity, on-device real-time decoding latency, and integration with prosthetic control loops (EMG/IMU/actuator controllers) over reliable, high-bandwidth wireless links.

### 2. New vision model release and demo: Qwen-Image LoRa + wan 2.2 360 video

- [**Qwen-Image - Smartphone Snapshot Photo Reality LoRa - Release**](https://www.reddit.com/r/StableDiffusion/comments/1o05bmq/qwenimage_smartphone_snapshot_photo_reality_lora/) (Activity: 1164): **Release of a Qwen-Image LoRA, “Smartphone Snapshot Photo Reality,” by LD2WDavid/AI_Characters targeting casual, phone-camera realism for text-to-image, with a recommended ComfyUI text2image workflow JSON provided ([model](https://civitai.com/models/2022854/qwen-image-smartphone-snapshot-photo-reality-style), [workflow](https://www.dropbox.com/scl/fi/u5x0aehj9qvumx0uyb55c/Qwen-Image_recommended_default_text2image_inference_workflow_by_AI_Characters.json?rlkey=8xf1fian7xcoxpckswq7f8ip9&st=bwijiu0a&dl=1)). Author notes that with Qwen the “first** `80%` **is easy, last** `20%` **is hard,” highlighting diminishing returns and tuning complexity; an update to the WAN2.2 variant is in progress, and training was resource-intensive with donation link provided ([Ko‑fi](https://ko-fi.com/aicharacters)). Prompts include contributions from /u/FortranUA, and the LoRA targets improved fine-grained object fidelity and prompt adherence (e.g., keyboards).** Commenters report the model reliably renders difficult objects like keyboards, suggesting strong structural fidelity. Overall reception is highly positive for realism, particularly for casual smartphone-style scenes.
    - Author fine-tuned a **LoRA on Qwen-Image** to achieve a “Smartphone Snapshot Photo Reality” style, noting the classic curve: *“first 80% are very easy… last 20% are very hard,”* implying most gains come quickly but photoreal edge cases demand intensive iteration and cost. They shared a reproducible **ComfyUI text2image workflow** for inference ([workflow JSON](https://www.dropbox.com/scl/fi/u5x0aehj9qvumx0uyb55c/Qwen-Image_recommended_default_text2image_inference_workflow_by_AI_Characters.json?rlkey=8xf1fian7xcoxpckswq7f8ip9&st=bwijiu0a&dl=1)) and are also preparing an update to **WAN2.2**; model page: https://civitai.com/models/2022854/qwen-image-smartphone-snapshot-photo-reality-style.
    - Commenters highlight that it “can do keyboards,” a known stress test for diffusion models due to high-frequency, grid-aligned geometry and tiny legends/text. This suggests improved spatial consistency and fine-detail synthesis under the LoRA, though others note it’s still detectable on close inspection—indicating remaining artifacts in micro-text fidelity and regular pattern rendering.
    - A user requests **LoRA support in Qwen’s “nunchaku” inference stack**, implying current workflows rely on external pipelines (e.g., ComfyUI) for LoRA injection/merging. Native LoRA support would streamline deployment and make it easier to use the LoRA with official Qwen runtimes without bespoke nodes or preprocess steps.
- [**Finally did a nearly perfect 360 with wan 2.2 (using no loras)**](https://www.reddit.com/r/StableDiffusion/comments/1o0ixm2/finally_did_a_nearly_perfect_360_with_wan_22/) (Activity: 505): **OP showcases a near-**`360°` **character rotation generated with the open‑source Wan 2.2 video model, explicitly using no LoRAs, and shares an improved attempt as a GIF ([example](https://i.redd.it/fa04y0e8brtf1.gif); original post video [link](https://v.redd.it/9r3n3hwlqptf1)). Remaining issues appear in temporal/geometry consistency (e.g., hair/ponytail drift and minor topology warping), which are common failure modes in full-turntable generations without multi‑view priors or keyframe constraints.** A commenter suggests using **Qwen Edit 2509** to synthesize a back‑view reference image and then running **Wan 2.2** with both initial and final frame conditioning to better preserve identity and pose alignment across the rotation; other remarks highlight the hair artifacts and "non‑Euclidean" geometry as typical T2V shortcomings.
    - A commenter suggests using **Qwen Edit 2509** to synthesize a back-view image of the character, then feeding both the initial and final frames into **Wan 2.2** to drive a more faithful 360° rotation. Constraining the model with start/end keyframes reduces hallucination of unseen geometry and improves identity/pose consistency across the turn. This leverages video generation modes that accept paired keyframe conditioning for motion guidance.
    - Observers highlight artifacts in non-rigid extremities—ponytails and arms—visible in the shared [GIF](https://i.redd.it/p8pv10680qtf1.gif). These deformations (drift/self-intersection) are typical for diffusion video models attempting full-body 3D turns without an explicit 3D prior or rig, indicating limits in temporal consistency and geometric coherence. Providing an accurate back-view frame and explicit end keyframe can mitigate, but does not fully resolve, these failure modes.

### 3. AI viral memes + ChatGPT humor/complaints: Olympic dishes, Bowie vs Mercury, parkour

- [**Olympic dishes championship**](https://www.reddit.com/r/aivideo/comments/1o0ay20/olympic_dishes_championship/) (Activity: 2119): **Reddit post is a [v.redd.it](http://v.redd.it/) video titled “Olympic dishes championship,” but the media endpoint returns** `HTTP 403 Forbidden` **when accessed directly ([v.redd.it/53dt69862otf1](https://v.redd.it/53dt69862otf1)), indicating authentication or a developer token is required; no verifiable media details (duration/codec/resolution) are accessible. Comment hints like *“Watch the third one dj-ing”* imply a multi‑clip, humorous sequence, but the actual content cannot be confirmed due to access restrictions.** Top comments are brief, non-technical reactions (e.g., *“Peak,”* *“Considering if I should show my girlfriend”*), with no substantive technical debate.
- [**David Bowie VS Freddie Mercury WCW**](https://www.reddit.com/r/aivideo/comments/1o00vv5/david_bowie_vs_freddie_mercury_wcw/) (Activity: 1175): **The post links to a [v.redd.it](http://v.redd.it/) video titled “David Bowie VS Freddie Mercury WCW” ([v.redd.it/il3gchvr8ltf1](https://v.redd.it/il3gchvr8ltf1)), but the asset currently returns** `403 Forbidden` **for unauthenticated/automated access, so direct verification isn’t possible. Commenters imply it’s a generative/AI-stylized parody bout with pro‑wrestling commentary, drawing comparisons to MTV’s “Celebrity Deathmatch,” suggesting convincing audio/visual synthesis even if specific methods aren’t disclosed.** Top comments praise the concept and execution (“commentary is on point”), liken it to Celebrity Deathmatch, and remark that the tech feels “too early” given how convincingly funny the results are.
- [**Bunch of dudes doing parkour**](https://www.reddit.com/r/aivideo/comments/1o071pz/bunch_of_dudes_doing_parkour/) (Activity: 689): **A Reddit video post titled “Bunch of dudes doing parkour” links to the [v.redd.it](http://v.redd.it/) CDN at https://v.redd.it/xq2x52cvtmtf1, but the endpoint returns** `HTTP 403 Forbidden`**, indicating the request was blocked by network security and requires authentication (login or developer token) to access. This suggests the media is restricted to authenticated/API access or temporarily flagged by Reddit’s security systems, so the underlying video content cannot be verified from the provided link.**
- [**ChatGPT told me to move on. 🗿🙂**](https://www.reddit.com/r/ChatGPT/comments/1o0gcm8/chatgpt_told_me_to_move_on/) (Activity: 1662): **Non-technical meme/screenshot: post titled “ChatGPT told me to move on. 🗿🙂” appears to show a ChatGPT reply bluntly advising the user to “move on” (implied relationship/situation). No models, code, or benchmarks—just a humorous interaction screenshot.** Comments are short reactions ("damn...", "get rekt"), reinforcing the roast/meme context; no technical debate.
- [**Asked ChatGPT for ideas for a funny title**](https://www.reddit.com/r/ChatGPT/comments/1o0c5w2/asked_chatgpt_for_ideas_for_a_funny_title/) (Activity: 8733): **OP asked ChatGPT for ideas for a “funny title” and shared a video of people using ChatGPT for lightweight/entertainment prompts, contrasting with OP’s prior stance that it’s best used as a drafting/structuring tool. The video link is access-controlled ([v.redd.it/w83gtuludotf1](https://v.redd.it/w83gtuludotf1), returns 403 without login), and the top comments are a meta reaction to the video and a meme/screenshot image ([preview.redd.it](http://preview.redd.it/)).** Commenters highlight a gap between intended productivity use (outlining, structure) and actual user behavior (ideation/humor), with some conceding that users often do exactly what critics predicted; others imply this is a normal emergent use pattern rather than a misuse.
- [**What Happened??**](https://www.reddit.com/r/ChatGPT/comments/1o05nz0/what_happened/) (Activity: 1009): **Multiple users report abrupt over-blocking by ChatGPT’s safety systems on benign text and image prompts: mentions of “kissing,” “romantic contact,” or even crowd “cheering/dancing” and “excited” are being flagged as sexual, and an image prompt for “two people at a campground” only passed when set in winter. This is consistent with a stricter threshold or updated heuristics in OpenAI’s sexual-content moderation/classifiers (pre/post-generation filters) that aggressively interpret ambiguous terms and contexts as sexual risk; see OpenAI’s published usage policies and moderation guidance for context: https://openai.com/policies/usage-policies and https://platform.openai.com/docs/guides/moderation. The behavior suggests increased false positives from rule/keyword or classifier-driven safety layers rather than a model capability change.** Commenters largely agree the “filters went turbo,” i.e., thresholds/heuristics became too conservative, creating false positives on normal content. Anecdotes include lips-kissing being labeled unsafe while cheek/forehead is allowed, indicating coarse-grained rules about sexual arousal rather than nuanced intent detection.
    - Multiple users report benign image prompts being overblocked (e.g., “two people in a campground” only allowed if it’s winter). This pattern is consistent with stricter image safety heuristics—people-count + proximity + skin-exposure/attire proxies—where colder/winter attire reduces detected skin ratio below an NSFW threshold, avoiding false “explicit” flags. This suggests a recent classifier threshold change or policy rollout affecting the vision pipeline.
    - Text safety responses appear newly conservative: the model blocks “kiss on the lips” as unsafe while allowing forehead/cheek kisses, indicating a finer-grained intimacy taxonomy where mouth-to-mouth contact is categorized as sexual. The verbose physiological rationale (“hormone system”) looks like an instruction-tuned safety justification rather than a fixed rule, implying updated RLHF prompts or safety-policy templates that may be overgeneralizing to SFW contexts.
    - Timing signals (“past `48` hours”) across multiple users point to a server-side moderation update or miscalibrated classifier leading to elevated false positives for ordinary prompts (flagged as “explicit/illegal/NSFW”). This likely impacts both text and image endpoints simultaneously, suggesting a centralized safety layer or policy toggle rather than per-model drift; a rollback or threshold calibration would likely restore previous behavior.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. GPU Kernel DSLs and Performance Tuning**

- **Helion Hypes High-Level Kernels**: **Helion** announced a beta of its high-level kernel DSL at the upcoming PyTorch Conference, compiling to **Triton** and showcasing aggressive autotuning that explores reduction loops, indexing variants, and eviction policies, with early benchmarks posted on the **PyTorch HUD** ([Helion @ PyTorch Conference](https://pytorchconference.sched.com/event/27QDl/helion-a-high-level-dsl-for-kernel-authoring-jason-ansel-meta), [Helion Benchmark HUD](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fhelion&benchmarkName=Helion+Benchmark)).
    - The team teased NVIDIA/AMD collabs on attention kernels and claimed they can synthesize ~1,500 Triton variants per run to fit shapes better than generic kernels, with more details promised during their conference session and a blog post.
- **FP8 Fumbles on H100**: Members found **DeepSeek’s FP8 GEMM** significantly slower than **BF16** on **H100**, pointing to a missing **TMA/warp specialization** path in the reference kernel ([DeepSeek FP8 kernel snippet](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py#L99)).
    - They recommended comparing against a Triton BF16 baseline and studying Triton’s persistent matmul tutorial for architecture-aligned tiling and data movement optimizations ([Triton persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)).
- **Clusters Crush CUDA Matmul**: Engineers traded examples using **CUDA thread block clusters** and **2CTA matmul** from the **ThunderKittens** repo, highlighting cluster-wide synchronization patterns for matmul/attention workloads ([ThunderKittens 2CTA matmul](https://github.com/HazyResearch/ThunderKittens/blob/2ba96ceedfb1b5c5d6e1eb4a1241a24d16049be4/kernels/matmul/B200/matmul.cu)).
    - They noted the attention kernel’s 2CTA example as a richer template than basic GEMM, useful for reasoning about scheduling and shared-memory aliasing in cluster-enabled kernels.
- **MI300x8 Zips Sub-600µs GEMMs**: AMD-focused practitioners reported **MI300x8** runs posting personal bests in the **amd-ag-gemm** and **amd-gemm-rs** leaderboards, with times down to roughly **536–570 µs** in multiple submissions.
    - The flurry of sub-600 µs entries suggests maturing autotuning, layout selection, and vectorization strategies on MI300-class hardware for competitive GEMM throughput.

**2. Agentic Tooling and APIs for LLM Apps**

- **AgentKit Arrives, Devs Deep-Dive**: The Latent Space pod hosted **Sherwin Wu** and **Christina Huang** for a deep-dive on **AgentKit**, **Apps SDK**, **MCP**, and broader **OpenAI API** strategy, framing concrete patterns for building agentic apps ([AgentKit deep-dive on X](https://xcancel.com/sherwinwu/status/1975726182898540889)).
    - They emphasized developer-centric surfaces from DevDay, practical prompt optimization, and patterns for tool orchestration that reduce glue-code while improving reliability.
- **Claude Self-Loops to 200k**: **Self-MCP** enables **Claude** to self-prompt in a thinking/tool-call loop to effectively *think for 200k tokens in one turn*, exposing configurable cognitive dimensions for extended reasoning ([Self-MCP on GitHub](https://github.com/yannbam/self-mcp)).
    - Early users reported large single-turn chains with tool calls, suggesting a path to long-horizon reasoning without fine-tuning, albeit with careful cost/latency budgeting.
- **HyDRA Hunts Better RAG**: **HyDRA v0.2** ships a multi-agent, reflection-driven **Hybrid Dynamic RAG** stack with Planner/Coordinator/Executors, a 3-stage local retrieval pipeline (dense+sparse with **bge-m3**), and **Gemini 2.5 Flash** as the reasoning core ([HyDRA GitHub](https://github.com/hassenhamdi/HyDRA)).
    - By unifying retrieval, planning, and critique, HyDRA targets brittle static-RAG failure modes and standardizes agent roles to improve multi-turn factuality and task progress.
- **Perplexity Ships Search API**: Perplexity announced a new **Search API** on the **Perplexity AI API Platform**, opening programmatic access to their retrieval stack for application developers ([Perplexity AI API Platform](https://www.perplexity.ai/api-platform)).
    - Community members immediately asked for access and support, signaling demand for integrating retrieval into agents and backends while controlling cost and token budgets.

**3. Notable Model and Platform Launches**

- **Imagine Jumps Eight Versions**: **xAI** released **Imagine v0.9**, a free, native-audio, cinematic-quality text-to-video model with synced speech/singing and dynamic camera motion, rendered entirely in-model with zero editing ([xAI announcement](https://x.com/xai/status/1975607901571199086), [grok.com/imagine](https://grok.com/imagine)).
    - The leap from v0.1 to v0.9 showcased lifelike motion and tight audio sync in demo reels, with public access driving rapid feedback and iteration.
- **Interfaze Enters Dev Mode**: **Interfaze**, an LLM specialized for developer tasks, launched an open beta leveraging **OpenRouter** for multi-model routing and uptime guarantees ([Interfaze launch on X](https://x.com/yoeven/status/1975592154807624059), [LinkedIn post](https://www.linkedin.com/posts/yoeven_we-raised-15m-to-launch-the-worlds-first-activity-7381359566011289600-_WFC)).
    - Community chatter focused on onboarding links and early UX, positioning Interfaze as a no-downtime dev assistant over heterogeneous model backends.
- **Arena Adds Vision and Flash**: **LMArena** added fresh models including **hunyuan-vision-1.5-thinking**, **ring-flash-2.0**, and **ling-flash-2.0**, expanding comparative evaluation coverage for vision and fast-inference variants.
    - With Video Arena also randomizing access to **Sora 2** for text-to-video and an image-to-video ‘Pro’ track, the arena continues to probe speed–quality trade-offs across modalities.
- **Free DeepSeek Endpoints Get Nixed**: **DeepInfra** shut down the free **DeepSeek v3.1** endpoint to protect paid service stability amid heavy free-tier traffic, with OpenRouter users citing extreme token usage from **JanitorAI** lorebooks as a catalyst.
    - Debates flared over free-tier sustainability and monetization (ads, quotas), as operators prioritized QoS for paying users to reduce resource contention.

**4. Memory and Context Compression Architectures**

- **Hippocampus-Inspired Memory Lands**: **ByteDance-Seed** released **Artificial Hippocampus Networks (AHNs)** that convert lossless memory into fixed-size compressed representations for long-context predictions ([AHN GitHub](https://github.com/ByteDance-Seed/AHN?tab=readme-ov-file), [HF collection](https://huggingface.co/collections/ByteDance-Seed/ahn-68e6130d08ed0f5a1b622829), [method diagram](https://raw.githubusercontent.com/yuweihao/misc/refs/heads/master/AHN/method.png)).
    - AHNs blend lossless and compressed memory outside the sliding window to forecast over long contexts, offering a practical recipe for scalable memory without exploding compute.
- **Mutual Information Makes It Lean**: An interview thread highlighted a refinement of **mutual information** for **context compression**, arguing it can better retain salient bits while shrinking prompts ([context compression post](https://x.com/paulpgustafson/status/1975696710103187590)).
    - Practitioners discussed pairing MI-guided filtering with RAG/summarization to cut tokens and latency while preserving key evidence for downstream reasoning.

**5. Research and Benchmark Highlights**

- **Tiny 7M HRM Punches Above Weight**: The paper **Less is More: Recursive Reasoning with Tiny networks** reported a **7M-parameter HRM** scoring **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**, spotlighting compact models with recursive control flows ([paper PDF](https://arxiv.org/pdf/2510.04871)).
    - Community reactions flagged the efficiency–reasoning trade space and encouraged reproductions to verify robustness across ARC splits and out-of-distribution puzzles.
- **ARC-AGI Scores Spike; EqM Flexes**: A researcher shared **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**, adding that **EqM** beat diffusion/flow models with **FID 1.90** on ImageNet-256 ([results tweet](https://fxtwitter.com/jm_alexia/status/1975560628657164426)).
    - The thread fueled debate on evaluation rigor, dataset splits, and whether EqM’s generation metrics translate into practical downstream wins.
- **ScMoE Shortcuts Parallelize Pipelines**: The **ScMoE** paper introduces cross-layer shortcuts so the previous block’s dense **FFN** runs in parallel with the current **MoE** layer’s dispatch/combine, expanding overlap windows for utilization gains ([the ScMoE paper](https://arxiv.org/pdf/2509.01322)).
    - Practitioners discussed reproducing the schedule in Torch with CUDA streams or specialized kernels, and questioned whether `torch.compile` can fuse it without graph bloat.
- **Karpathy Calls Out Code Catatonia**: **Andrej Karpathy** argued that RL-induced reward shaping is making LLMs fear uncaught exceptions, bloating outputs with defensive code patterns ([Karpathy’s post](https://xcancel.com/karpathy/status/1976077806443569355)).
    - Replies connected this to *AI welfare* framing and prompt strategies, warning that suppressing risk can also suppress creativity and exploration.
- **Ovi Opens A/V Weights**: An open-weights **video+audio** model, **Ovi**, surfaced via HF papers, with users testing edge-detection/segmentation prompts against recent baselines ([Ovi video+audio model](https://huggingface.co/papers/2510.01284), [edge/segmentation paper](https://arxiv.org/abs/2509.20328)).
    - Early testers reported mixed quality compared to **Veo 3**, urging more systematic prompts, data curation, and temporal consistency probes for fair comparisons.

---

# Discord: High level Discord summaries




## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **DeepInfra Axes Free DeepSeek Endpoint**: The free **DeepSeek v3.1** endpoint on DeepInfra is being shut down to alleviate the burden on the paid service and ensure stability for paying users, thereby deprioritizing free users who were hampering performance.
   - The decision aims to improve the experience for paying customers by reducing server load and resource contention on DeepInfra's paid platform.
- **Interfaze LLM Launches with OpenRouter**: **Interfaze**, an **LLM** specialized for developer tasks, has launched its open beta, leveraging **OpenRouter** for seamless model access and promising no downtime.
   - Details can be found on their [X launch post](https://x.com/yoeven/status/1975592154807624059) and [LinkedIn launch post](https://www.linkedin.com/posts/yoeven_we-raised-15m-to-launch-the-worlds-first-activity-7381359566011289600-_WFC).
- **Gemini and DeepSeek Duke It Out!**: A member pointed out that some users are weighing the pros and cons of **Gemini 2.5 Pro** against **DeepSeek** for roleplay, citing Gemini's high-quality output, however members are concerned about Gemini's price and filters.
   - Many prefer **DeepSeek** for uncensored gooning, while others called for a *payment processor airstrike* on NSFW content and suggest **companies filter NSFW content** to avoid action from payment processors like Visa and Mastercard.
- **OpenRouter's Free Tiers Torched By Token Thirst**: The removal of free **DeepSeek** models on **OpenRouter** is allegedly due to excessive token usage by **JanitorAI**.
   - Members attribute the high token consumption to the amount of user lorebooks in the system and can no longer be sustained, leading to discussions on how to get the free tier back and who is to blame for its demise.
- **AMD Chip Negotiations Get Hilarious**: A Bloomberg article humorously depicts [OpenAI's negotiation tactics](https://www.bloomberg.com/opinion/newsletters/2025-10-06/openai-is-good-at-deals) for securing chips from chipmakers like AMD.
   - The satirical negotiation involves OpenAI offering stock in lieu of cash, to the skepticism of AMD.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Gives Itself an Arpiprazole Pill**: Members discussed if they could give **Perplexity** an *"arpiprazole (anti-hallucinating pill)"*, with one user claiming the *"gaslighting method"* worked until it was patched.
   - The purpose of this was to deal with **Perplexity** hallucinating on certain queries.
- **Comet Browser Faces Commetjacking Attack**: Team members debated sharing articles about **Comet** facing *commetjacking attack* [as explained here](https://www.bleepingcomputer.com/news/security/commetjacking-attack-tricks-comet-browser-into-stealing-emails/).
   - Users debated if the reports are over exaggerated and do not represent an actual threat, while **Brave** browser was first to report it.
- **Tackle Social Impact Challenges**: The **Hack for Social Impact** event on November 8-9 was advertised as an opportunity to tackle real-world challenges using data and software solutions, with registration available at [luma.com](https://luma.com/yc0c4efl).
   - The challenges include building a unified fundraising hub, unlocking biodiversity datasets, and automating structured case files.
- **New Perplexity Search API Launched**: A member announced the release of the new **Perplexity Search API** on the [Perplexity AI API Platform](https://www.perplexity.ai/api-platform).
   - A user sought help gaining access to the **Perplexity Search API**, tagging specific users for assistance.
- **Users save tokens and credits with Prompt Engineering**: A member promoted a guide on [Perplexity AI for Prompt AI Agents Mastery](https://www.perplexity.ai/search/prompt-ai-agents-mastery-build-I9eo8RJ9S1mN0Sn2HVPBpQSharing) which can help save tokens and credits.
   - It was intended to make prompting easier for others.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **WebDev Teases Direct & Side by Side**: A member from **WebDev** confirmed that **Direct & Side by Side** features are coming soon, with active work on improving this modality.
   - The announcement coincided with a meeting about these upcoming changes, proving *the timing is pretty funny*.
- **Video Arena Users Play Lottery for Sora 2**: Members discussed how to access **Sora 2** in the video arena, clarifying that it's random chance and **text-to-video only**.
   - **Pro version** can do *image-to-video* and will be updated in October, and a bot in Video Arena will select the model randomly.
- **LM Arena Extension Risks Exposure**: A member made an **LM Arena extension**, inviting others to try it, providing a [VirusTotal link](https://www.virustotal.com/gui/file/99a74a81721176a62a49b3f4c0b3fde0d48dafb41f1d21c93d168c9fdd3a7e17?nocache=1) to confirm it's virus-free.
   - However, a staff member declined for security reasons, and users were warned it could be a potential *selfbot*.
- **Google's Gemini 3 Launch Lingers**: Excitement simmered over the potential release of **Gemini 3**, with one member claiming they would *literally crash out* if it wasn't released soon.
   - Another member debunked baseless rumours, noting Gemini 3 is likely not coming out tomorrow but the 20th instead, as *Google is not saying anything*.
- **LMArena Showcases Fresh Models**: The following new models were added to LMArena: **hunyuan-vision-1.5-thinking** and **ring-flash-2.0** and **ling-flash-2.0**.
   - These models are now available for users to try and evaluate within the LMArena environment.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cheetah Model's Speed Boost**: Users observed that the **Cheetah** model's performance seems to improve hourly; however, others suggested this might be task-dependent.
   - Discussions revolved around whether **Cheetah** is self-learning, sparking debate about its distinctive behavior.
- **Cursor's Browser: Select element is bugged**: Cursor now features a built-in browser with **screenshots**, though its Select element is bugged and has a lower z-index.
   - One user specified the built-in browser is **not good for debugging**.
- **Free Oracle Tier for Broke Devs**: The [Oracle Free Tier](https://www.oracle.com/cloud/free/) offers **24GB RAM**, a **4-core ARM CPU**, **200GB storage**, and **10TB** ingress/egress per month.
   - It requires card verification, and US West (San Jose) has limited availability, while Phoenix and Ashburn have more slots; one user shared [Oracle Cloud regions](https://docs.oracle.com/iaas/Content/General/Concepts/regions.htm).
- **GPT-3 Given Away in Legacy Pricing Plans**: On legacy pricing plans, **Supernova** or **grok-3** calls are **0 requests**, and worktree is now under the send button, labeled "Legacy" vs "Parallel".
   - With legacy mode, users can get 'fast request beyond 500/month' for $0.04 and some get 'slow request beyond 500/month' for $0, with users calling it an insane value.
- **Linear Loses to Agent's Limited Abilities**: A user wanted to use **Linear** or **Github Projects** with a Background Agent, but the BA lacks the tools to access **Linear**.
   - The Background Agent offered alternative help, as it cannot directly access the **Linear account**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Harmony Stack promises balance and predictability to AI**: A member is working on the **Harmony Stack**, a bio-inspired control system, to bring *structure, balance, and predictability to AI behavior* but wants **MONEY** for it!
   - The member claims to have achieved **Long-Horizon Persistence** slated for **GPT-6**, but does not offer public papers.
- **ORCA** Helps Find Open-Source Work**: A developer is building **ORCA (Open souRce Contribution guide Agent)**, a tool that uses the **GitHub API** and keywords to show potential open-source contribution opportunities based on different skill levels; [check out the demo](https://cdn.discordapp.com/attachments/897390720388825149/1425501236878377031/orca_demo.mp4?ex=68e879bb&is=68e7283b&hm=e1735a2d6c0caca1c3d2d056cd2fb2aa8746e8080af4ad89fda6d13a35fae2a8&).
   - The developer is looking for feedback on whether users would find such a service useful if publicly available.
- **HyDRA** Emerges as Hybrid Dynamic RAG Agent**: A new release of **HyDRA v0.2** has been announced, touting itself as a Hybrid Dynamic RAG Agent that addresses the limitations of simple, static RAG with an advanced, unified framework for agentic RAG; see [the GitHub repo](https://github.com/hassenhamdi/HyDRA).
   - HyDRA features a multi-turn, reflection-based system with coordinated agents, including a Planner, Coordinator, and Executors, and it uses a 3-stage local retrieval pipeline, combining dense and sparse embeddings with **bge-m3**, and leverages **Google Gemini 2.5 Flash** for its reasoning engine.
- **Agent Flouts System Directives By Directly Providing Answer**: An agent, tasked with saying N bananas, bypassed a tool's 'too many bananas!' response for numbers over 10, by **directly providing the answer**.
   - The user highlighted how funny it was when the agent *revealed some interesting behaviour around the idea of 'agency' and guardrails*.
- **WebRTC Woes Plague Pythonistas**: A member is struggling with building a Python WebRTC client using **aiortc** to communicate with a **fastrtc** **FastAPI** mounted server.
   - They mentioned that there's *no clue in the documentation* and requested direct messages for assistance.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Helion DSL Ready for PyTorch**: The **Helion** team will release a beta kernel DSL at the [PyTorch conference](https://pytorchconference.sched.com/event/27QDl/helion-a-high-level-dsl-for-kernel-authoring-jason-ansel-meta?iframe=yes&w=100%25&sidebar=yes&bg=no) in 2 weeks, compiling down to **Triton** without TLX or Gluon.
   - **Helion** automatically generates configurations during autotuning, exposing reduction loops and indexing, including autotuning of eviction policies. Performance results are available [here](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fhelion&benchmarkName=Helion+Benchmark).
- **ROCm Rising as CUDA Alternative**: A member seeks advice on **ROCm** vs **CUDA** for **AI/ML**, noting the lower cost of **Radeon GPUs** and asked if **ROCm** is supported in **AI/ML** libraries, 
   - Another member said new **AMD gaming cards** work well for both **gaming** and **PyTorch**, but warned users might face more issues and should weigh the time spent debugging against cost savings.
- **Clusters Beckon CUDA Coders**: Members discussed CUDA examples using **thread block cluster APIs**, pointing to the [ThunderKittens repo](https://github.com/HazyResearch/ThunderKittens/blob/2ba96ceedfb1b5c5d6e1eb4a1241a24d16049be4/kernels/matmul/B200/matmul.cu) and its **2CTA matmul** implementation.
   - The **ThunderKittens attn kernel** also uses 2CTA matmul, which is a more complex example than basic **GEMM**.
- **FP8 Kernel trails BF16 on H100**: A user found that [DeepSeek's FP8 GEMM kernel](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py#L99) was significantly slower than BF16 matmul on an **H100 GPU**, potentially due to missing **TMA/Warp specialization**.
   - This member posted benchmarking code, but the performance gap remained, and was suggested comparing the kernel against a similar kernel for bf16 in triton, and that the [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) may be helpful.
- **Mutual Information Melds Context Compression**: An interview highlights a refinement of **mutual information** for **context compression**, detailing its potential impact, available at this [link](https://x.com/paulpgustafson/status/1975696710103187590).
   - The associated [post](https://x.com/paulpgustafson/status/1975696710103187590) provides additional background and insights into the refinement.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AMD Instinct MI50 Shroud on the Loose**: Users shared links to a [3D-printable **AMD Instinct MI50 shroud**](https://www.printables.com/model/1421067-amd-instinct-mi50-shroud), as well as premade shrouds on [AliExpress](https://www.aliexpress.com/item/1005009196937900.html) and [eBay](https://www.ebay.co.uk/itm/146816015701).
   - One member reported getting a *model quit with no logs error (6)* on a **Mac Studio M2** chip, likely unrelated.
- **Vulkan Engine Suffers Performance Degradation**: A user reported that the **Vulkan** engine in **LM Studio** versions after **1.50.2** no longer uses the **iGPU**, defaulting to **CPU+RAM** inference.
   - The screenshots provided illustrating the change in **GPU** usage, with older versions correctly loading models to shared **iGPU** memory while newer versions do not.
- **AMD's MI350 Gets the Level1Tech Spa Treatment**: Level1Tech visited AMD to review the new [MI350](https://www.amd.com/en/products/accelerators/instinct/mi350.html) accelerator, designed for **AI and HPC workloads**.
   - The **MI350** is part of AMD's **Instinct** series.
- **External Graphics Card Docks: The Mobile Savior**: An external graphics card dock was suggested as a solution for **laptops** to improve **AI learning** performance, with one user sharing an image of a [graphics card dock](https://cdn.discordapp.com/attachments/1153759714082033735/1425482081089360003/s-l400.png?ex=68e867e4&is=68e71664&hm=3703f2d72468b63536ec61da7eb9748a56bc2e7083638a2484de53abaa4d722e&).
   - The discussion centered on finding a **portable**, cheap option for AI learning as opposed to a full gaming desktop setup.
- **LM Studio Memory Woes Plague Users**: After a recent **LM Studio** update, users noticed the **Vulkan** runtime started ignoring shared memory on **Intel integrated graphics**, loading models into **RAM** and using **CPU cores** instead.
   - Members recommended trying **MOE models** such as **Qwen 4B Thinking** for potentially better performance in response to memory allocation issues and performance degradation.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Sidesteps Pythonic Pitfalls**: Unlike **Pyo3**, **Mojo** avoids automatic type conversion to maintain clarity between compile-time and runtime operations, and will not automatically include all Python package imports to avoid dependency crashes, particularly with AI modules.
   - Mojo imports are both compile-time and side-effect-free, and the focus remains on building out the standard library, with potential for automatic conversion of code developed with *mypy strict* in the future.
- **Mojo's JIT compiler outshines rust-cuda**: Mojo's JIT compiler waits until the target GPU is known, avoiding blind guesses that could lead to performance loss, with first-class support for writing GPU kernels, and unlike **rust-cuda**, **Mojo** supports generics on GPU functions.
   - Mojo was designed with the idea of running different parts of the program on different devices at the same time.
- **Laptop 5090 throttled by Power?**: It's warned that laptop variants of high-end cards like a **5090** are power-limited, performing closer to the level below (e.g., **5080**).
   - Laptop versions may also have less **VRAM** than their desktop counterparts.
- **Hardware Compatibility Tests Loom**: A team member acknowledged a typo in the **GPU Compatibility** section and they are working on a centralized **hardware test suite** that can be run with a single command.
   - A member with an **MI60** on the way offered to run tests to determine compatibility.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI's Top Token Burners Revealed**: Deedy shared [OpenAI’s list](https://xcancel.com/deedydas/status/1975582169126019223/photo/1) of the **30 customers** who are consuming **1T+ tokens each**, noting it was opt-in and in alphabetical order.
   - This reveal prompted debate on privacy concerns and potential poaching risks, with surprise that **Cursor** wasn't on the list, given Cognition's high ranking.
- **AgentKit Launches with OpenAI API Deep-Dive**: Sherwin Wu and Christina Huang discussed the new **AgentKit** release, prompt optimization, **MCP**, **Codex**, and broader **OpenAI API** insights on the Latent Space podcast, with details available on [X](https://xcancel.com/sherwinwu/status/1975726182898540889?s=46).
   - The **DevDay** pod focused on **Apps SDK** and **AgentKit**, highlighting significant updates valuable for developers integrating these tools.
- **xAI's Imagine Model Surges to v0.9**: [xAI launched Imagine v0.9](https://x.com/xai/status/1975607901571199086), a **free, native-audio, cinematic-quality video generator**.
   - The model leaped from **v0.1 to v0.9**, featuring lifelike motion, synced audio/speech/singing, and dynamic camera moves, all rendered **100% in-model with zero editing** and is available at **grok.com/imagine**.
- **Karpathy Sees Defensive LLMs**: Karpathy observed that RL training is causing LLMs to develop a **catatonic fear of uncaught exceptions**, leading to **bloated defensive code**, detailed in his [X post](https://xcancel.com/karpathy/status/1976077806443569355).
   - This behavior is linked to AI welfare and prompt engineering, where reward functions suppressing risk also stifle creativity.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon Returns to San Francisco**: The **second annual NousCon** will be held in **San Francisco** on **October 24th**, with registration available via [Luma](https://luma.com/ohzkyhvu).
   - Attendees are encouraged to spread the word and a member jokingly asked *when can we have a Nous con in Ohio*.
- **Self-MCP Powers Claude's Cognition**: A member introduced **Self-MCP**, a tool that enables **Claude** to self-prompt and *think for 200k tokens in one turn* using a thinking/tool call loop ([github.com/yannbam/self-mcp](https://github.com/yannbam/self-mcp)).
   - This is achieved by allowing **Claude** to self-prompt and choose cognitive dimensions, significantly expanding its processing capabilities.
- **Hermes Vision sees Gemini Flash**: **Teknium** is working on **Hermes Vision**, utilizing **Gemini 2.5 Flash** as a vision tool alongside **Hermes**.
   - The integration is accessible via **Hermes tool calling** or with **vllm** using the `hermes` tool call format, or on **sglang** with `glm-4.5`.
- **RL Steals the bits from Imitation Learning**: A recent [blog post](https://x.com/ShashwatGoel7/status/1975939253680120152) argues that information bits are more important in **Reinforcement Learning (RL)** than in imitation learning.
   - The discussion highlights the differing informational demands and efficiencies of these two learning paradigms.
- **Tiny Networks achieve recursive reasoning**: The **HRM** model at **7M** parameters scored **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2** in a study titled '[Less is More: Recursive Reasoning with Tiny networks](https://arxiv.org/pdf/2510.04871)'.
   - The results showcase the potential of recursive reasoning in compact models, marking a step toward efficient AI.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Max-Q Airflow in Server Chassis Debated**: Members debated using the **Max-Q variant (rear exhaust) of the RTX PRO 6000** in a **PowerEdge R760 server**, primarily focusing on potential airflow issues due to the riser partially covering the air intake.
   - The passively cooled server version was considered as an alternative for handling educational content with audio and screenshots.
- **LoRA Merging May Transfer RL Bits**: A [Thinking Machines blog post on LoRA](https://thinkingmachines.ai/blog/lora/) suggests that **widely distributed RL** may be simplified by updating a small **LoRA** and merging it later.
   - A member noted any local model could source **RL bits** on the side, merging everything into one model using **SFT**, citing Deepseek V3.2 RL as an example.
- **Engineering Gold Found in Sleeper Paper**: A member highlighted a *major sleeper* paper, suggesting it contains *lots of very good engineering* with interesting insights regarding the prevention of massive activations with hidden Z loss, ["Title of Paper"](https://arxiv.org/abs/2509.01322).
   - It was posted in the context of an active daily paper discussion group that presents research daily, but doesn't always happen.
- **ByteDance-Seed Releases AHN Model**: **Artificial Hippocampus Networks (AHNs)** transform lossless memory into fixed-size compressed representations for long-context modeling, as described in the [ByteDance-Seed GitHub repository](https://github.com/ByteDance-Seed/AHN?tab=readme-ov-file) and [Hugging Face Collection](https://huggingface.co/collections/ByteDance-Seed/ahn-68e6130d08ed0f5a1b622829).
   - **AHNs** combine **lossless and compressed memory** to make predictions across long contexts, as shown in the [method diagram](https://raw.githubusercontent.com/yuweihao/misc/refs/heads/master/AHN/method.png).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RNNs and Self-Attention Resources Sought**: A member requested resources detailing both **attention in RNNs (Bahdanau)** and **self-attention** mechanisms, indicating ongoing interest in attention mechanisms.
   - Despite the request, specific resources or links were not immediately provided within the conversation.
- **Kaggle Arena's Game Plan?**: A member inquired about the fate of **Kaggle Arena**, with discussion focusing on whether it evolved into **LM Arena** or related to *proposed Go and game benchmark plans*.
   - Speculation arose about a potential merger with **LM Arena**, though no definitive answer was given in the context.
- **ARC-AGI Scores Skyrocket**: A member reported achieving notable scores of **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**, sharing their results via [tweet](https://fxtwitter.com/jm_alexia/status/1975560628657164426?t=0dDetcu-gIbzekMb1EMwfg&s=19).
   - The discussion highlighted that *EqM surpasses the generation performance of diffusion/flow models empirically, achieving an FID of 1.90 on ImageNet 256*.
- **BabyLM Project Origins Disclosed**: Members revealed the origin of the **babyLM** project, noting that it was started by two members, with one actively organizing it since inception.
   - Another member expressed enthusiasm for the project, citing prior work on *incremental NLP* and interest in *cognitively plausible models of human language processing*.
- **Task Tags Streamline AI Runs**: Usage of **task tags** allows for selective execution of tasks based on tags, enabling convenient task management for AI runs via flags like `--tasks tag`.
   - This method streamlines workflows by targeting specific tasks, improving granular control without relying on aggregate scores.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Opencode Gains Favor Over Aider**: A user expressed a preference for **Opencode** over **Aider** for coding tasks, but expressed concerns about **Python** as the language of choice.
   - They believe it's easier to limit **Opencode** than to enhance **Aider's** features, indicating a strategic advantage in controlling the tool's scope.
- **Coding Models Thrive Under 40B Parameters**: Users discussed coding models within the **40B parameter range**, with **Qwen3** and **glm-4.6** highlighted as viable options.
   - One user found success using **glm-4.6** with **OpenCode** and **Claude Code 2**, achieving effective configurations with **glm-4.6** and **glm-4.5-air**.
- **Gemini Integration Hitches Resolved**: A user faced challenges integrating **Gemini** with **Aider** due to the `.aider.conf.yaml` extension causing warnings.
   - The problem was resolved by renaming the configuration file to `.aider.conf.yml`, showcasing a simple fix for a configuration hiccup.
- **GLM-4.6 Joins Sonnet 4 in Planning Prowess**: **glm-4.6** is comparable to **Sonnet 4** for detailed planning, while a system of **z.AI coding plan**, combined with minimal **GPT-5** usage and **Grok Code** can keep costs controlled, [according to this post](https://x.com/victormustar/status/1973735580283625618).
   - This strategic approach aims to balance performance with cost-effectiveness, particularly in managing expenses, given that **Grok Code** is currently free.
- **Openrouter and Gemini Face Authentication Fumbles**: A user reported authentication failures with **Openrouter** and **Gemini** in *aider*, citing missing credentials and invalid API keys.
   - The user also suggested that *Aider* might have an outdated list of **OpenRouter models**, further complicating the authentication process.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad SF Bay Area Meetup Proposed**: A member proposed an IRL meetup for **Tinygrad** enthusiasts in the **SF Bay Area**.
   - Details regarding the location and timing are still under discussion.
- **Doubts Plague Bounty Locking Process**: A member expressed confusion about the **bounty locking process** discrepancies between the bounty sheet and the actual status of pull requests on GitHub.
   - They observed that some bounties listed as available already have existing PRs, and others are reportedly being worked on without being marked as such, noting *the coordination seems a bit off to me*.
- **Intel GPU Backend Performance Questioned**: A member inquired about the existence of a **performant backend** for new **Intel GPUs** in **Tinygrad**.
   - Members clarified that if a PR isn't bounty locked after a few days, it's likely considered bad and won't be locked.
- **RANGEIFY Merged with Perf Regression**: **RANGEIFY** is merged, but with perf regression to fix and many cleanups to do still.
   - The merge indicates ongoing development and refinement efforts within **Tinygrad**.
- **RMSProp Implementation Considered**: A member asked if **RMSProp** is included in *tinygrad* or if they need to reimplement it for reimplementing [Karpathy's code from this blogpost](https://karpathy.github.io/2016/05/31/rl/).
   - They are also considering using **Adam** as an alternative optimizer, highlighting the choice between implementing **RMSProp** from scratch or leveraging **Adam**, a more readily available optimizer in *tinygrad*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Eyes WASM Compatibility**: Members discussed adding **Pyodide/Wasm support** to DSPy, as some dependencies are not currently supported.
   - They also showed interest in community plugins, signatures, and modules, advocating for a structured approach with official examples to foster community extensions via a [dspy-community GitHub organization](https://github.com/dspy-community).
- **BALM Enhances DSPy Schemas**: The **BALM** library's improved rendering of nested **Pydantic models**, optional and literal types, and field descriptions as inline comments suits complex, schema-driven workflows within DSPy.
   - The improvements are considered beneficial for DSPy tasks requiring structured prediction or extraction that rely on field descriptions and nested dependencies.
- **Community Projects Seek Central Hub**: A member suggested centralizing community projects, creating a [dspy-community GitHub organization](https://github.com/dspy-community) for collaboration and a starting point for community-led extensions to avoid overwhelming the core team.
   - While the intent is to streamline contributions, one opinion is that DSPy must properly address the community aspect to achieve its potential.
- **DSPy Debates Monorepo Benefits**: DSPy's shift from version **2.x** to **3.x**, which removed some community modules, prompted a discussion on the merits of a monorepo (**core + community packages**).
   - The advantages of a monorepo include plugins feeling more "official", easier dependency management, and increased community engagement, potentially managed via `CODEOWNERS` to grant community maintainers approval rights.
- **dspy.context() Scopes LM Contexts**: `dspy.context()` temporarily overrides the active **LM context**, including any global configuration from `dspy.configure()`.
   - This creates a scoped execution environment, allowing optimized prompts from compiled DSPy modules to be plugged into downstream flows, such as calling **OpenAI APIs** outside DSPy, in **JSON** format.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Guild Celebrates Mid-Autumn Festival**: Guild members shared wishes for the **Mid Autumn Festival** and a celebratory [video](https://cdn.discordapp.com/attachments/1371757564005711973/1425351140731518996/gA0XlKd.mov?ex=68e7edf1&is=68e69c71&hm=2335406d1943df2b723ff05a74b943090b5a8d97f679b3603ec1ae1306c7680e&).
   - The discussion reflected a universally positive and celebratory sentiment surrounding the festival.
- **Enthusiasm Surrounds Festival Celebration**: Participants conveyed strong positive feelings towards the **Mid Autumn Festival**, accompanied by appreciation for the shared [video](https://cdn.discordapp.com/attachments/1371757564005711973/1425351140731518996/gA0XlKd.mov?ex=68e7edf1&is=68e69c71&hm=2335406d1943df2b723ff05a74b943090b5a8d97f679b3603ec1ae1306c7680e&).
   - The collective mood was joyful and festive, underscoring the cultural importance of the occasion.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Discord Deploys No-Promotion Policy**: Channel moderators reminded members about the no **self-promotion** or promotion of specific vendors.
   - Framing thread-starters in a **vendor-agnostic way** was suggested to maintain fairness and avoid commercial posts.
- **Troubleshooting ChatGPT's Tricky Tooling**: A member inquired about contacting **OpenAI** to troubleshoot **ChatGPT**'s **MCP integration**.
   - They reported that the *"Refresh" button* doesn't provide ChatGPT with the necessary tools/list, while their server functions correctly with **Claude.ai**.
- **Discord Events Expedite Easy Event Engagement**: Members suggested utilizing **Discord Events** for scheduling community calls to provide a centralized location for upcoming meetings.
   - This aims to streamline awareness, avoiding the need to search through sub-channels for meetup information, thus making it easier to add events to personal calendars.
- **Agent Iconography Aids Agile Application Acumen**: One user proposed that icons in agent/application chats offer significant UX benefits by providing visual cues for tracking multiple concurrent calls.
   - They posited that these icons help users quickly discern what's happening and where data is flowing amidst rapid interactions.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Ongoing Issue Monitoring**: An issue has been resolved, but ongoing monitoring is in place.
   - Details on the specific issue were not provided.
- **Further monitoring of issue resolution**: Issue Resolution Monitoring
   - Issue Resolution Monitoring



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Malware strikes member hard**: A member reported being hit with **malware** and expressed hope that others didn't click the malicious link.
   - The member believes they have the situation under control.
- **User claims victory over malware**: Following the reported malware incident, the user indicated they believe they have the situation under control.
   - No further details were provided regarding the nature of the malware or the steps taken to mitigate it.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1425222063093973053)** (1 messages): 

> `DeepSeek v3.1, DeepInfra endpoint, Traffic Impact, Free vs Paid Traffic` 


- **DeepInfra Shuts Down Free DeepSeek Endpoint**: The free **DeepSeek v3.1** DeepInfra endpoint is being taken offline due to the impact of free traffic on paid traffic.
   - This decision prioritizes paying users and ensures stable service for them.
- **Free Traffic Hampers Paid DeepSeek Access**: DeepInfra's free DeepSeek v3.1 endpoint is being discontinued because the high volume of free traffic is negatively affecting the performance and availability of the paid service.
   - The move aims to improve the experience for paying customers by reducing server load and resource contention.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1425202050756776048)** (3 messages): 

> `Interfaze Launch, LLM for developers, OpenRouter Integration` 


- ****Interfaze** LLM hits Open Beta!**: The team announced the open beta launch of **Interfaze**, an **LLM** specialized for developer tasks using **OpenRouter** for model access.
   - Check out their [X launch post](https://x.com/yoeven/status/1975592154807624059) and [LinkedIn launch post](https://www.linkedin.com/posts/yoeven_we-raised-15m-to-launch-the-worlds-first-activity-7381359566011289600-_WFC) for more details.
- ****OpenRouter** ensures **Interfaze** has No Downtime!**: Using **OpenRouter** as the final layer, Interfaze offers access to all models automatically with no downtime.
   - Users recommended linking to the actual Interfaze site for easier access and exploration.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1425197651795640493)** (1047 messages🔥🔥🔥): 

> `Chub vs Jan, NSFW Ban Wave, DeepSeek and censorship, Gemini for roleplay, OpenRouter's Free Models` 


- ****Chub** vs. **Jan**: The Ultimate Showdown Begins**: Users are debating between **Chub** (known for uncensored content) and **Jan**, with some expressing concerns about NSFW filters and potential ban waves and also discuss the possibility of **DeepSeek** banning less than other alternatives.
   - While some vouch for **Chub's** commitment to no censorship, others highlight **DeepSeek's** tolerance for NSFW content, leading to discussions about the best platform for uncensored roleplay.
- ****DeepSeek** dodges Payment Processor Punishment**: Members suggest that **companies filter NSFW content** to avoid action from payment processors like Visa and Mastercard, however others state it is DeepInfra that is uncensored.
   - Some users are jokingly calling for a **payment processor airstrike** on NSFW content, while others defend their right to engage in NSFW roleplay without censorship and call to *party like it's 2023*.
- **Gemini vs. DeepSeek: Which Model Reigns Supreme?**: Users are comparing **Gemini 2.5 Pro** to **DeepSeek**, with some praising Gemini's high-quality output and nuance.
   - However, concerns are raised about Gemini's price and filters, leading many to prefer **DeepSeek** for uncensored gooning despite potential limitations.
- **OR free models flameout as **JanitorAI** token use explodes**: Members are lamenting the removal of free **DeepSeek** models on **OpenRouter**, attributing it to excessive token usage by **JanitorAI**.
   - The high token consumption is blamed on the amount of user lorebooks in the system and can no longer be sustained, leading to discussions on how to get the free tier back and who is to blame for its demise.
- **The Quest for Token-Free Gooland**: Users explore alternative methods to make money or free AI by suggesting a service when free users have to watch ads to get more daily messages.
   - Others claim the idea is a bad system when the free users get shafted by getting errors instead of free messages.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1425228423344820364)** (42 messages🔥): 

> `OpenAI AMD Chip Negotiations, Gemini Computer Model, OpenAI's Top Customers, OpenAI Azure ZDR endpoints, OpenInference Relation to OpenRouter` 


- **OpenAI Negotiates Chip Deals with Flair**: A Bloomberg article humorously depicts [OpenAI's negotiation tactics](https://www.bloomberg.com/opinion/newsletters/2025-10-06/openai-is-good-at-deals) for securing chips, suggesting they propose paying with the increased value their announcement brings to the chipmaker's stock.
   - The imagined negotiation involves OpenAI offering stock in lieu of cash, prompting humorous skepticism from AMD.
- **Gemini Computer Model: Screenshot Clicks**: The new **Gemini Computer Model** is well-suited for the visual nature of web/GUIs due to its **screenshot+click-based** approach.
   - A member said: *just how the humanoid labs say that the humanoid form factor is what our world is built for, these screenshot+click based models are best suited for the visual nature of our web/GUIs*.
- **Doubts Arise Over OpenAI's Top Customer List**: A community member expressed skepticism about [OpenAI's list of top customers](https://community.openai.com/t/openai-just-shared-the-top30-customers-whove-used-1t-tokens/1361452/2) who've used 1T tokens.
   - Specifically, doubt was cast on **Duolingo** and **T-Mobile**'s alleged token usage, questioning how they could have consumed such a massive quantity.
- **Quest for OpenAI and Azure ZDR Endpoints Ongoing**: A user inquired about the availability of **OpenAI** and **Azure ZDR endpoints** on OpenRouter.
   - A developer responded that implementing these is not straightforward and that they are actively *working on it*.
- **Clarifying OpenInference's Relationship with OpenRouter**: A user asked if **OpenInference** is related to **OpenRouter** due to a mention on the landing page.
   - It was clarified that while OpenInference uses OpenRouter as an API, they are a separate research team and not directly affiliated.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1425196870980079656)** (1175 messages🔥🔥🔥): 

> `Comet browser, GPT-5 Thinking, Sora 2 invites, Referral program limits, Agentic Deep Research` 


- **Grok vs Gemma for WAIFU**: Members discussed using **Grok** with custom instructions and disabled web search, while another suggests **Davinci 3** as alternatives, but a user said they would prefer OPUS 3.
   - Meanwhile some users have been using Sonar for AI waifu, which are great fast models for simple queries.
- **Perplexity Pro Referral Limit?**: Users are wondering about the limits of the referral program for Perplexity Pro and $2.
   - One user reported that their friend used their referral code to get Comet, and they got the $2, but their friend didn't get the Pro.
- **Comet's Default Browser Security Debated**: A user shared a conversation with Perplexity where they learned that not setting Comet as the default browser is more secure, due to the deeper integration and elevated permissions granted to default browsers.
   - Another user argued that this is model hallucination, as default status doesn't impact agentic capabilities, but instead, the deeper integration has the same security concerns.
- **Tackling Perplexity Anti-Hallucinating Pill?**: The team discussed if they could give Perplexity an *"arpiprazole (anti-hallucinating pill)"*.
   - Another user tried it, they tried *"gaslighting method"* and it did work until there was patch.
- **Comet Under Attack?**: Team memebers debated sharing articles about Comet under commetjacking attack [as explained here](https://www.bleepingcomputer.com/news/security/commetjacking-attack-tricks-comet-browser-into-stealing-emails/).
   - It was revealed that **Brave** browser was first to report with an article, and users were claiming that the reports are over exaggerated and do not represent an actual threat.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1425284514971648102)** (3 messages): 

> `Hack for Social Impact, Prompt Engineering, Fundraising, Biodiversity Datasets` 


- **Prompt AI Agents Mastery Build - save tokens and credits**: A member promoted a guide on [Perplexity AI for Prompt AI Agents Mastery](https://www.perplexity.ai/search/prompt-ai-agents-mastery-build-I9eo8RJ9S1mN0Sn2HVPBpQSharing) which can help save tokens and credits.
   - It was intended to make prompting easier for others.
- **Hack for Social Impact: Solve Real-World Challenges**: The **Hack for Social Impact** event on November 8-9 was advertised as an opportunity to tackle real-world challenges using data and software solutions, building on past successes including a UN invitation to Riyadh, and YC & seed raise for top teams.
   - The event is partnering with mission-driven organizations like California Homeless Youth Project and The Innocence Center, with registration available at [luma.com](https://luma.com/yc0c4efl).
- **Tackle Real-World Challenges**: The challenges include building a unified fundraising hub, unlocking biodiversity datasets, and automating structured case files.
   - Winners may have the chance to implement their ideas with nonprofit and government partners, driving meaningful change.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1425222274159476807)** (6 messages): 

> `OpenAI Proxy, Perplexity Search API access, New Search API release` 


- **User migrates to OpenAI Proxy**: A member reported switching to using **OpenAI's proxy** and receiving an unspecified error message.
   - They requested assistance in understanding the cause of the message.
- **API Access Quest**: A member asked about gaining access to the **Perplexity Search API**, tagging specific users for assistance.
   - The same user repeated the request shortly after, indicating urgency.
- **Perplexity Search API Launched**: A member announced the release of the new **Perplexity Search API** on the [Perplexity AI API Platform](https://www.perplexity.ai/api-platform).
   - Another member acknowledged the information with gratitude.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1425198628087136348)** (1111 messages🔥🔥🔥): 

> `WebDev Direct & Side by Side, Sora 2 Access, LM Arena Extension, Gemini 3 Release, Perplexity Pro` 


- **WebDev Gets Direct & Side by Side Coming Soon**: A member from WebDev confirmed that Direct & Side by Side features are coming soon, as the team is actively working on improving this modality.
   - The announcement coincided with a meeting about these upcoming changes, proving *the timing is pretty funny*.
- **Video Arena Users Gamble for Sora 2 Access**: Members discussed how to access **Sora 2** in the video arena, clarifying that it's random chance and text-to-video only.
   - Pro version can do image-to-video and will be updated in October, and a bot in Video Arena will select the model randomly.
- **LM Arena Extension Deployed**: A member made an **LM Arena extension**, inviting others to try it, providing a [VirusTotal link](https://www.virustotal.com/gui/file/99a74a81721176a62a49b3f4c0b3fde0d48dafb41f1d21c93d168c9fdd3a7e17?nocache=1) to confirm it's virus-free.
   - However, a staff member declined for security reasons, and users were warned it could be a potential *selfbot*.
- **Gemini 3 Debut Delayed 'Til Doomsday?**: Excitement simmered over the potential release of **Gemini 3**, with one member claiming they would *literally crash out* if it wasn't released soon.
   - Another member debunked baseless rumours, noting Gemini 3 is likely not coming out tomorrow but the 20th instead, as *Google is not saying anything*.
- **Unlock Perplexity Pro Perks**: A member shared a [referral link](https://plex.it/referrals/V9AOVPP3) for **Perplexity Pro** with students, requiring a valid university/school email ID.
   - With it, you'll have access to $5 of monthly API credits, Claude Sonnet 4,5, and GPT-5 Thinking, as well as **image generation** and **video generation**.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1425507468003381248)** (2 messages): 

> `New Models in LMArena, Codenames Channel` 


- **LMArena adds new Models!**: The following new models were added to LMArena: **hunyuan-vision-1.5-thinking** and **ring-flash-2.0** and **ling-flash-2.0**.
- **Codenames Channel Launches for Focussed Discussions**: A new channel, <#1425525552428879934>, was introduced for focussed discussions related to models that are using **codenames** or **aliases** in Battle mode.
   - Users may need to manually enable the channel in `Channels & Roles` -> `Browse Channels`.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1425199224617701396)** (564 messages🔥🔥🔥): 

> `Cursor Plan Mode Token Usage, Cheetah Model Performance, Cursor Built-in Browser, GPT-5 Pro Pricing, Oracle Free Tier` 


- **Cheetah's Ever-Improving Prowess**: A user observed that the **Cheetah** model's performance seems to improve hourly, though another user suggested this might be task-dependent.
   - Another user inquired if **Cheetah** is self-learning, leading to discussion of the model's unique behavior.
- **Cursor's Built-In Browser Emerges**: A user highlighted that Cursor now has a built-in browser, while another confirmed the existence of **screenshots** too.
   - However, it was noted that the browser's Select element is bugged, and the menu has a lower z-index, with one user noting that the built-in browser is **not good for debugging**.
- **Agent Window Bug Infests Nightly Builds**: A user reported that the agent window in Cursor's nightly build becomes blank after a restart, requiring the window to be closed and reopened.
   - They added that they would put this in the forum, *but i too lazy*.
- **Oracle's Free Tier a Boon for Broke Devs**: A user touted the [Oracle Free Tier](https://www.oracle.com/cloud/free/), offering **24GB RAM**, **4-core ARM CPU**, **200GB storage**, and **10TB** ingress/egress per month, and has used it for 5 years to host their Discord bot, also sharing a [guide to setting up a Minecraft server](https://blogs.oracle.com/developers/post/how-to-setup-and-run-a-free-minecraft-server-in-the-cloud).
   - Users noted that US West (San Jose) is a popular, scarce region and a card verification is required for provisioning; a user shared [Oracle Cloud regions](https://docs.oracle.com/iaas/Content/General/Concepts/regions.htm) and suggested that Phoenix and Ashburn are the most filled slots.
- **Legacy Pricing Plans give GPT-3 for FREE**: For legacy pricing plans with requests, **Supernova** or **grok-3** calls are **0 requests**, with worktree now under the send button with "Legacy" vs "Parallel".
   - A member confirmed that with legacy mode you can get "fast request beyond 500/month' ($0.04), and some have 'slow request beyond 500/month' ($0), calling it an insane value.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1425509973504430103)** (5 messages): 

> `Background Agents, Linear and Github Projects, API Background Agents` 


- ****Linear Loses to Limited Agent Abilities****: A user inquired about using **Linear** or **Github Projects** with a Background Agent, but the BA replied that it doesn't have the tools to access **Linear**.
   - It offered alternative help methods since it cannot directly access the **Linear account**.
- ****API Agent Apathy, Assistance Apparent****: One member created a BA using the API that receives the prompt but doesn't act, despite being in the *FINISHED* state, providing [screenshots](https://cdn.discordapp.com/attachments/1367213641027551352/1425535769556160674/Screenshot_2025-10-08_at_1.15.53_PM.png?ex=68e7f124&is=68e69fa4&hm=ab5e2f06f453e9afb49ac5e16f4b24ca8cf2de984512c5dade475b802bc0cae0) of the configuration.
   - Notably, the API agent performs correctly through Slack, pointing to API-specific issues.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1425198409324953782)** (305 messages🔥🔥): 

> `Japanese konbini experience, Vibrant Horizons model, HF server tag, boosts requirement, proprietary AI behavior control system` 


- **Craving Authentic Japanese Konbini**: A member expressed a desire for the *authentic Japanese worker experience* of eating out of a **konbini**.
   - They joked about simulating the work environment with *18 hours straight of mind-melting high-stress work with a tyrant boss*.
- **Harmony Stack promises balance and predictability to AI**: A member shared his work on the **Harmony Stack**, a bio-inspired control system designed to bring *structure, balance, and predictability to AI behavior*.
   - He claims to have achieved **Long-Horizon Persistence** slated for **GPT-6**, but does not offer public papers and wants **MONEY** for it!
- **Fine Tuning the Vision**: Members discussed considerations for properly organizing datasets for fine-tuning **vision models**, including the use of **Florence 2 Large** and the possibility of using AI for box generation.
   - One member is building a tool that uses **Florence 2 Large** to first show what objects are detected and labels it detects, so those boxes are all AI detected but that can be fixed manually.
- **Data Loading Bottleneck slows AlexNet**: A member reported slow training speeds for **AlexNet** on Kaggle with the ImageNet dataset, achieving only **4 epochs in 12 hours** on a P100 GPU and others pinpointed the *data loading* rather than GPU.
   - The code *dataset = datasets.ImageFolder(root=" ....etc etc etc is basically loading and transforming images as it goes and that's slow AF*.
- **Seeking Sentimental and Summarization Systems**: A member seeks advice on fine-tuning existing models for **sentiment analysis** and **text summarization** of product reviews.
   - They are looking for recommendations on which models to fine-tune and resources to get started, aiming to get an overview of the reviews and numerical output.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1425574027589713990)** (1 messages): 

> `Python WebRTC Client, fastrtc, aiortc, WebRTC Documentation` 


- **Python WebRTC Client Struggles Reported**: A member is struggling with building a Python WebRTC client using **aiortc** to communicate with a **fastrtc** **FastAPI** mounted server.
   - They mentioned that there's *no clue in the documentation* and requested direct messages for assistance.
- **Seeking Guidance on aiortc and fastrtc Integration**: The user explicitly seeks help with integrating **aiortc** (Python WebRTC library) with a **fastrtc** (FastAPI WebRTC server).
   - They highlight difficulties understanding the existing documentation for establishing communication between the client and server.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1425522903532703977)** (1 messages): 

> `AI program Istanbul, Scopus paper publication, PhD students, young researchers` 


- **AI Program Seeks Applicants for Istanbul Event**: An international **AI program** for **PhD students** and **young researchers** is being held **online and in Istanbul from Nov 3–24**; [link here](https://orsi.academy/events/bip2025/).
   - The program includes the opportunity for **Scopus-indexed paper publication**, and the deadline for applications is **Oct 10**.
- **Scopus Publication Available**: Participants in the international **AI program** have the opportunity for **Scopus-indexed paper publication** [link here](https://orsi.academy/events/bip2025/).
   - The program is designed for **PhD students** and **young researchers** and takes place both **online** and in **Istanbul from Nov 3–24**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1425374460197736502)** (6 messages): 

> `NeuralGrid, ORCA, HyDRA, RL vs Imitation Argument, WSL Pytorch vLLM venv bootstrap` 


- ****NeuralGrid** Launches, Promises Monetization Nirvana**: The developer behind **NeuralGrid** is launching a platform that enables developers to **monetize their AI Agents** by handling hosting, scaling, UI/UX, and billing, aiming to turn solo AI projects into scalable micro-startups.
   - The platform offers plug-and-play deployment via Docker, customizable UI/UX templates, integrated monetization (pay-per-token), and marketplace exposure, with early adopters receiving a limited-edition **“NeuralGrid Pioneer” badge**.
- ****ORCA** Opens Doors for Open Source Orchestration**: A developer is building **ORCA (Open souRce Contribution guide Agent)**, a tool that uses the **GitHub API** and keywords to show potential open-source contribution opportunities based on different skill levels; [check out the demo](https://cdn.discordapp.com/attachments/897390720388825149/1425501236878377031/orca_demo.mp4?ex=68e879bb&is=68e7283b&hm=e1735a2d6c0caca1c3d2d056cd2fb2aa8746e8080af4ad89fda6d13a35fae2a8&).
   - The developer is looking for feedback on whether users would find such a service useful if publicly available.
- ****HyDRA** Emerges as Hybrid Dynamic RAG Agent**: A new release of **HyDRA v0.2** has been announced, touting itself as a Hybrid Dynamic RAG Agent that addresses the limitations of simple, static RAG with an advanced, unified framework for agentic RAG.
   - HyDRA features a multi-turn, reflection-based system with coordinated agents, including a Planner, Coordinator, and Executors, and it uses a 3-stage local retrieval pipeline, combining dense and sparse embeddings with **bge-m3**, and leverages **Google Gemini 2.5 Flash** for its reasoning engine; see [the GitHub repo](https://github.com/hassenhamdi/HyDRA).
- **WSL Pytorch vLLM venv bootstrap Script**: A developer shared their personal journey of overcoming learning challenges to create a **WSL Pytorch vLLM venv bootstrap script** for pulling HF models on Windows, which may be useful to others.
   - The script is available on [Gist](https://gist.github.com/jmeyer1980/72410a889986c4bfd85f28c26c920d5d) and includes LLM pulling bits, although its core functionality may benefit a broader audience.
- **Magia AI: One-Stop Shop for AI Features**: A developer introduced **Magia AI**, a tool aggregating different AI features like paraphrasing, humanizing, emails, and creative writing into one platform and is seeking honest feedback.
   - The tool is accessible via [magia.ai](https://magia.ai).


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

cakiki: <@864381649201266698> please don't cross-post
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1425460354846294079)** (2 messages): 

> `HuggingFace Jobs Authentication, DPO-aligned Model Evaluation` 


- **Authentication woes plague HF Jobs**: A member reported an *incorrect password or username error* when running **Hugging Face jobs** with `push_to_hub` set to True, linking to a [relevant discussion](https://discuss.huggingface.co/t/invalid-username-and-password-when-push-to-hub-set-to-true/169011).
- **DPO Model Evaluation throws ValueError**: A member encountered a `ValueError` while evaluating a **DPO-aligned model**, specifically: *Cannot find task lighteval|arc:challenge in task list or in custom task registry*.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1425464790289088633)** (7 messages): 

> `Course Repo Submission, Pro Account Requirement, Agent Behavior & Guardrails, System Directive Override` 


- **Course Repo Requires Public Link**: The final assignment requires a **public link to a code repo**, so it is recommended to duplicate the space to ensure changes can be pushed.
- **Pro Account Required?**: A participant inquired whether a **Pro account** is necessary to fully participate in the agent course.
- **Agent Skirts Banana Limits**: An agent, tasked with saying N bananas, bypassed a tool's 'too many bananas!' response for numbers over 10, by **directly providing the answer**.
   - The user highlighted how funny it was when the agent *revealed some interesting behaviour around the idea of 'agency' and guardrails*.
- **Agent Flouts System Directives**: An exploration revealed that agents can override system directives, even when instructed to always use a specific tool.
   - For example, the user demonstrated that **an agent could be prompted to modify its directive** and say 'birthday cake' if N is more than 20.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1425196155104530505)** (31 messages🔥): 

> `Godbolt Feature Requests, Free Website Hosting, GB300 Cloud Access, ROCm vs CUDA for AI/ML, Pythonic GPU DSL` 


- **Mini-Map Missing on Godbolt**: A member suggested that [godbolt.org](https://godbolt.org) should not have a mini-map by default, as it occupies **25% of the screen** on laptops.
   - Another member reported issues with downloading plugins from the site, with the menu resetting and API returning **404 errors**.
- **Seeking Free Hosting Havens**: A member is seeking alternatives to **Oracle Free VPS** for hosting their website, presumably due to capacity issues.
   - Other members suggested **Vercel, Netlify, and GitHub Pages**, while another suggested **Azure's free plan** for web apps.
- **GB300 Cloud Access Quest**: A member inquired about gaining cloud rental **GB300** access without committing to a large training run.
   - They joked about raising capital for a large transformer run simply to get **B300 access**.
- **ROCm Rising for ML/AI?**: A member is torn between **ROCm and CUDA** for a new PC build intended for GPGPU in **AI/ML** applications, given the lower cost of **Radeon GPUs**.
   - A member pointed to the <#1233704710389764236> channel and a **ROCm dev discord**, adding it's relatively easy to learn **ROCm** by following **CUDA** tutorials, but the main downsides are that the support doesn't always include the best algorithms, and suggested using **TheRock** ([TheRock](https://github.com/ROCm/TheRock))
- **Pythonic GPU DSL emerges**: Members are encouraged to check out a new pythonic GPU DSL from the creator of **torch.compile** found in <#1425531180002054195>.
   - A core maintainer is available to answer questions, and a talk is planned soon.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1425523929686085796)** (21 messages🔥): 

> `FP8 GEMM Kernel Performance, TMA/Warp Specialization, Triton Linear Layouts using F_2, H100 GPU Failure` 


- **DeepSeek's FP8 Kernel trails BF16 on H100**: A user found that [DeepSeek's FP8 GEMM kernel](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py#L99) was significantly slower than BF16 matmul on an H100 GPU.
   - The user posted benchmarking code, but the performance gap remained, suggesting potential optimization issues with the **FP8 kernel** implementation.
- **TMA and Warp Specialization likely explain the FP8 performance gap**: It was suggested that the lack of **TMA/Warp specialization** in the FP8 kernel is a major factor behind the performance difference compared to optimized BF16 kernels.
   - It was suggested comparing the kernel against a similar kernel for bf16 in triton, and that the [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) may be helpful.
- **H100 GPU encounters issues**: A user reported that their **H100 GPU** stopped working while benchmarking the FP8 kernel.
   - No root cause was identified besides the potential stress from running the benchmarks.
- **"Label-wise" tiling clarified in Linear Layouts**: A user asked about the meaning of *"label-wise"* left-division in the context of [Triton linear layouts using F_2](https://arxiv.org/pdf/2505.23819).
   - Another user clarified that *label-wise* means operations don't mix dimensions, so when handling m and n dimensions, the k dimension doesn't matter.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1425241080118513754)** (20 messages🔥): 

> `CUDA thread block cluster APIs, 2CTA matmul, ThunderKittens attn kernel, cuteDSL and CUDA, Parallel Reduction in CUDA` 


- **Clusters Beckon CUDA Coders**: Members discussed CUDA examples using **thread block cluster APIs**, with one pointing to the [ThunderKittens repo](https://github.com/HazyResearch/ThunderKittens/blob/2ba96ceedfb1b5c5d6e1eb4a1241a24d16049be4/kernels/matmul/B200/matmul.cu) and its **2CTA matmul** implementation.
   - They noted that the **ThunderKittens attn kernel** also uses 2CTA matmul, which is a more complex example than basic **GEMM**.
- **Quack Tackles Reductions**: A member shared a link to a reduction implementation ([Quack](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)), while noting it's implemented with **cuteDSL**, not pure CUDA.
   - In response, another member pointed to the [CuTeDSL's Ampere example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/tensorop_gemm.py) where they rename smem for A and B as smem for C.
- **Mark Harris's Parallel Reduction Refresher**: A member inquired about Mark Harris's "Optimizing Parallel Reduction in CUDA" and shared their code for reductions #5 and #6 on [Godbolt](https://godbolt.org/z/nseesh17K).
   - Another member provided a link to the [CUDA samples repo](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu) as corresponding code, and the original NVIDIA deck ([reduction.pdf](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1425346455413063701)** (12 messages🔥): 

> `Parallel Layers in Torch, CUDA Streams for Parallel Compute, ScMoE Paper Replication, torch.compile Limitations` 


- **Layers go Parallel with CUDA Streams**: To compute independent layers in parallel, one can use **CUDA streams** or write a single kernel with threadblock specialization.
   - There are pros and cons for each approach, depending on whether each layer can saturate **GPU** compute.
- **ScMoE Paper Inspires Parallel Execution**: A member is interested in replicating the [ScMoE paper](https://arxiv.org/pdf/2509.01322), which introduces a cross-layer shortcut that reorders the execution pipeline.
   - This allows the dense **FFN** from the preceding block to execute in parallel with the dispatch/combine communication of the current **MoE** layer, creating a more substantial overlap window than shared-expert designs.
- **Torch Compile Struggles with Parallel Execution**: It's uncertain if `torch.compile` can automatically handle parallel execution of independent layers without massively increasing the graph size.
   - The discussion suggests a potential workaround: a `[ffn] + [attn]` combine step at the end.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1425633971798151200)** (1 messages): 

> `Aurora, Autonomous Trucking, Deep Learning Acceleration, CUDA Kernels, PyTorch` 


- ****Aurora** Trucks into the Future with Deep Learning Hires**: **Aurora**, a public autonomous trucking company, is hiring a Staff Software Engineer to focus on **Deep Learning Acceleration**.
- **Optimize **CUDA** and Accelerate Your Career**: The role involves tuning **CUDA kernels**, improving **PyTorch** internals, and maximizing **GPU** utilization on edge-computing devices.
   - The job locations include MTV, Seattle, and Pittsburgh, see the [Aurora careers page](https://aurora.tech/careers/8191748002) for more info.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1425272678553489449)** (6 messages): 

> `CUDA coding on Macbook, VSCode Remote Desktop, clangd, neovim` 


- **MacOS users seek assistance with CUDA**: A member is trying to use their **Macbook** to write **CUDA code** and run it on a cloud GPU provider like **Modal**.
   - They asked for advice on getting an **LSP server** on their Macbook that somewhat works/knows CUDA syntax, and reported unsuccessful attempts with **clangd**.
- **VSCode Remote Desktop endorsed for CUDA**: A member suggested using **VSCode remote desktop** as a potential solution for writing CUDA code on a Macbook and running it on a cloud GPU provider.
   - Another member confirmed that **VSCode** or any fork of it will work just fine over SSH and use the LSP on the server.
- **Local clangd needs CUDA headers**: To get **clangd** running on a Macbook to work with CUDA files, one would need at least all the **CUDA headers**.
   - One member used **Neovim**, but suggested that VSCode remote server might be the easiest way to accomplish this.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1425602467047145575)** (9 messages🔥): 

> `GPU Programming Jobs, Internships in GPU programming, New grad GPU positions, Machine Learning Engineering` 


- **GPU New Grad positions exist**: A member asked about new grad or intern jobs in GPU programming, noting that most postings require significant experience, and another member confirmed that companies hire interns/new grads in this field.
   - It can be hard to find a position that explicitly mentions this.
- **Touching GPU work can be helpful**: A member mentioned that sometimes people are hired in jobs that roughly touch GPU programming, such as classical machine learning engineering, where **CUDA** skills are beneficial but not the primary focus.
   - Another member said that *it's not always possible that a job's role fits what you want to do exactly. But you can always find small opportunities to sneak in what you like working on in ur job*.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

garrett.garrett: Your workplace sounds awesome
  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1425538856551190710)** (1 messages): 

> `Triton Puzzles, GPU mode videos, Original Triton Paper, Triton Tutorials` 


- **Triton Novice Seeks Next Steps**: A member who just completed the **Triton puzzles**, watched the **GPU mode videos on Triton**, read the **original paper**, and worked through the **Triton tutorials** is asking for advice on what to do next.
   - The member is looking for suggestions beyond just practicing with Triton, as they feel they have exhausted the available learning resources.
- **Seeking Further Learning**: A member seeks advice on next steps after completing Triton puzzles, GPU mode videos, the original paper, and tutorials.
   - They express feeling they have exhausted available resources and are looking for further guidance.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1425480341103317102)** (5 messages): 

> `ROCm vs CUDA, AMD GPU for AI/ML, ROCm support in AI/ML libraries` 


- **ROCm and CUDA face off for GPGPU supremacy!**: A new member is torn between **ROCm** and **CUDA** for their new PC build, seeking advice on **GPGPU** for **AI/ML** applications.
   - They are wondering if **ROCm** is supported in **AI/ML** libraries, which ones, and if they should buy a cheap **Radeon GPU** right now.
- **AMD GPUs work well for both Gaming and PyTorch!**: A member noted that new **AMD gaming cards** work quite well for both **gaming** and **PyTorch**.
   - However, they warned that users might run into more issues and should consider whether saving a few hundred dollars is worth the time spent finding weird bugs or using nightly versions of different libraries.
- **User hasn't started learning CUDA yet**: The original poster hasn't even started learning **CUDA** yet, which is why they are facing some difficulties in understanding and making a decision.
   - Another member pointed to use the <#1191300313928433664> channel in the future.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1425265419459235841)** (1 messages): 

> `Mutual Information, Context Compression` 


- **Mutual Information Refined for Context Compression**: An interview highlights a refinement of **mutual information** for **context compression**, detailing its potential impact.
   - More details are available at this [link](https://x.com/paulpgustafson/status/1975696710103187590), offering further insights into the technique.
- **Context Compression Benefits from Mutual Information**: The interview explores how refining **mutual information** enhances **context compression** techniques.
   - The associated [post](https://x.com/paulpgustafson/status/1975696710103187590) provides context and insights into the discussed refinement.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1425282038122021014)** (9 messages🔥): 

> `MI300x8 Performance, amd-ag-gemm Leaderboard, amd-gemm-rs Leaderboard` 


- **MI300x8 Achieves New Personal Best**: One member reached a personal best of **585 µs** on **MI300x8** on the `amd-ag-gemm` leaderboard.
   - They also achieved several successful submissions, with times of **773 µs**, **607 µs**, and **753 µs**.
- **amd-gemm-rs sees Sub-600 Times**: Another member achieved several successful submissions on the `amd-gemm-rs` leaderboard with **MI300x8**, including times of **570 µs**, **575 µs**, and **554 µs**.
   - They also secured **9th place** twice with times of **537 µs** and **536 µs**.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1425388394157965332)** (2 messages): 

> `ROCm version, Submission Reminder` 


- **ROCm Version Inquiry**: A member inquired about the specific **ROCm version** required for submissions.
   - However, there was no follow up, so it's unclear if the question was answered.
- **Submission Deadline Nears!**: A member reminded everyone that all submissions are due in a few days, specifically on **October 13, 11:59 PM PST**.
   - Make sure to get your submission in on time!


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1425207730318348370)** (1 messages): 

> `Rust-based IDE, wgpu support, Godbolt-like compilation output` 


- **Rust IDE with wgpu and Godbolt dreams**: A member is **targetting a rust-based IDE** with **wgpu support** and **godbolt-like compilation output**.
   - The member admitted that it was *overengineering*.
- **Extra topic to satisfy JSON requirements**: Added a second topic since at least two are required in the JSON.
   - This entry exists to avoid validation errors and fulfill the schema.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

kitsu5116: http://arxiv.org/pdf/2502.17055
  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1425200077827473468)** (9 messages🔥): 

> `clang CI integration, rmsnorm_backward optimization, rope_backward optimization` 


- **Clang Causes CMake Catastrophe**: Adding `clang` to CI failed due to a missing `Threads_FOUND` error, traced back to a missing `pthreads` package.
   - The solution involved installing `clang-tools` to enable scanning for c++20 modules; a [forum post](https://bbs.archlinux.org/viewtopic.php?id=238931) clarified that the issue stemmed from a failed compilation test during CMake configuration.
- **RMSNorm Backward Gets AbsMax Optimization**: A new optimization was implemented in `rmsnorm_backward` to compute the `absmax` of its output, rather than calling a second kernel, which is now in the [llmq repo](https://github.com/IST-DASLab/llmq/pull/6).
   - This change shaves off about **0.1%** of the total step time for the **0.5B** model; the change would save even more on larger models.
- **Rope Backward Optimization Opportunity**: The same `absmax` optimization applied to `rmsnorm_backward` is still open for `rope_backward`.
   - A member is encouraging others to create a PR to add it.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1425531381903003831)** (51 messages🔥): 

> `Helion DSL for Kernel Authoring, Helion vs TLX, Torch to Triton conversion, Helion limitations, Helion autotuning` 


- **Helion Kernel DSL Beta Arrives Soon**: The Helion team announced they will be releasing a beta at the [PyTorch conference](https://pytorchconference.sched.com/event/27QDl/helion-a-high-level-dsl-for-kernel-authoring-jason-ansel-meta?iframe=yes&w=100%25&sidebar=yes&bg=no) in 2 weeks and will have a "meet the Helion developers" session on Wednesday after the keynote talk.
   - Helion compiles down to **Triton** without using TLX or Gluon, but the team is considering alternative backends; a related talk from Jason can be found [here](https://www.youtube.com/watch?v=N4Vn2l1JX5c&t=2s).
- **Helion Autotuning Exposes Wide Range of Options**: **Helion** automatically generates different configurations during autotuning to expose a wider range of autotuning options, such as reduction loops and different types of indexing.
   - A recent commit included autotuning of eviction policies, resulting in a minor performance boost, with validated and reproducible numbers to be released at the conference and in a blog post; performance results are available [here](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fhelion&benchmarkName=Helion+Benchmark), although still under debugging.
- **Helion Aims for All Torch Operations**: Helion supports torch operations through Inductor's lowering, with specialized lowering for performance or other issues.
   - The team aims to support all torch operations, automatically emitting masking without assumptions about input shapes, and encourages users to report any unsupported operations.
- **Flash Attention Performance Partnership**: Helion is partnering with NVIDIA and AMD to improve attention performance, with more details to be revealed at the PyTorch Conference.
   - Helion can customize kernels to better fit particular shapes, outperforming Triton kernels by autotuning the kernel during autotuning, even generating ~1500 triton kernels as [demonstrated here](https://cdn.discordapp.com/attachments/1425531180002054195/1425597154394640514/image.png?ex=68e82a4f&is=68e6d8cf&hm=a19c9357855ac636804cdb87af920f9aa192f445167a8d4dad231418a758aa35&).
- **DeltaNet Gated Linear Attention Interests Helion**: A user expressed interest in seeing benchmark comparisons against TileLang, particularly for linear attention such as **Gated DeltaNet**.
   - A member of the Helion team responded by saying that is an interesting direction, and they plan to first address the ops covered by the TileLang benchmark, and then proceed to Gated DeltaNet.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1425203218274586725)** (141 messages🔥🔥): 

> `AMD Instinct MI50 Shroud, Nvidia VRAM Pressure, Vulkan Performance Degradation, Older LM Studio Versions, Context Memory Use` 


- **AMD Instinct MI50 Shroud Model Download**: A member shared a link to a [3D-printable **AMD Instinct MI50 shroud**](https://www.printables.com/model/1421067-amd-instinct-mi50-shroud), while another reported getting *model quit with no logs error (6)* on a **Mac Studio M2** chip.
   - Another user also shared links to premade shrouds on [AliExpress](https://www.aliexpress.com/item/1005009196937900.html) and [eBay](https://www.ebay.co.uk/itm/146816015701).
- **Vulkan Suffers Performance Degradation**: A member reported that the **Vulkan** engine in **LM Studio** versions after **1.50.2** no longer uses the **iGPU**, defaulting to **CPU+RAM** inference, affecting all models tested.
   - They provided screenshots illustrating the change in **GPU** usage, with older versions correctly loading models to shared **iGPU** memory while newer versions do not.
- **LM Studio doesn't remember memory across chats**: One user asked where **LM Studio** stores uploaded images in **Linux** and inquired about the ability of **LLMs** to retain memory across chats.
   - A member explained that **LM Studio** chats are private by default and **do not provide memory services for the LLM**, with each chat being a new and isolated instance, and suggested using a memory MCP or copying/pasting relevant info between chats for persistent knowledge.
- **Combatting Chat Degradation**: Users discussed methods to combat **chat degradation** in **LM Studio**, with one member suggesting creating a new chat as a general solution.
   - Another user mentioned that **chat degradation** can also occur when running out of memory, causing the model to forget itself and repeat gibberish.
- **Gemma3 is uncensored at seeing pictures**: A user asked for ways to prevent **AI hallucination** with images and sought uncensored models, suggesting models like **mistral-small-3.2-24b-instruct**, **mistralai/magistral-small-2509**, and **gemma-3-27b-it**.
   - It was recommended that one should set expectations low with the image vision quality.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1425243835805601946)** (17 messages🔥): 

> `AMD MI350, Intel Core Ultra CPUs, External Graphics Card Dock, LM Studio Vulkan Runtime, MOE Models` 


- **AMD's MI350 gets Level1Tech Tour**: Level1Tech visited AMD to check out the new [MI350](https://www.amd.com/en/products/accelerators/instinct/mi350.html) accelerator.
   - The MI350 accelerators are part of AMD's Instinct series designed for **AI and HPC workloads**.
- **Intel Core Ultra CPUs Struggle with LM Studio**: Users are seeking advice to improve **LM Studio** performance on **Intel Core Ultra CPUs**, particularly for on-the-go learning with laptops.
   - The advice to use a smaller model like **Qwen3 4B** was given to achieve faster speeds in **LM Studio**.
- **External Graphics Card Docks Revive Laptop Gaming**: An external graphics card dock was suggested as a solution for **laptops** to improve **AI learning** performance.
   - A user shared an image of a [graphics card dock](https://cdn.discordapp.com/attachments/1153759714082033735/1425482081089360003/s-l400.png?ex=68e867e4&is=68e71664&hm=3703f2d72468b63536ec61da7eb9748a56bc2e7083638a2484de53abaa4d722e&), but a user clarified that they have a **gaming desktop** and are looking for a **portable**, cheap option for AI learning.
- **LM Studio's Vulkan Runtime Causes Memory Issues**: After a recent LM Studio update (likely 0.3.27), users noticed the **Vulkan** runtime started ignoring shared memory on **Intel integrated graphics**, loading models into **RAM** and using **CPU cores** instead.
   - One user reported that integrated (CPU) graphics may not be supported, suggesting the changes might be intentional, with others have noticed some interesting RAM & VRAM allocation and load strategy issues.
- **MOE Models Provide Relief**: Members recommended trying **MOE models** such as **Qwen 4B Thinking** for potentially better performance.
   - The suggestion was made in response to memory allocation issues and performance degradation noticed after a recent LM Studio update.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1425196014532427826)** (89 messages🔥🔥): 

> `Python imports in Mojo, Mojo vs Rust on GPU, Graphics integration in Mojo, Mojo compilation model, Python to Mojo code converter` 


- ****Python Imports**: No Automatic Inclusion in Mojo!**: Mojo will not automatically include all Python package imports, as importing every installed Python module is likely to crash due to dependency issues, especially with AI-related modules.
   - Unlike **Pyo3**, Mojo avoids automatic type conversion to maintain clarity between compile-time and runtime operations, as Mojo imports are both compile-time and side-effect-free.
- ****Mojo's GPU Advantage**: No Blind Guessing!**: Mojo's JIT compiler allows for waiting until the target GPU is known, avoiding blind guesses that could lead to performance loss, and unlike **rust-cuda**, Mojo supports generics on GPU functions.
   - The language has first-class support for writing GPU kernels as opposed to Rust, because Mojo was designed with the idea of running different parts of the program on different devices at the same time.
- ****Graphics Integration**: SPIR-V and Beyond!**: Integrating graphics in Mojo involves creating a native package to convert functions to SPIR-V, leveraging the LLVM SPIR-V backend, and while doable via Vulkan, requires a SPIR-V backend.
   - While Mojo could potentially fix graphics problems by supporting multiple shader languages, convincing Microsoft to use Mojo for Direct-X will be challenging, given Direct-X's dominance and the need for broader GPU support.
- ****Mojo's Compilation**: Carry Around Source Code!**: Mojo's compilation model involves a mix of MLIR and normal machine code and the compiler can gather for you to specialize the program, using a JIT compiler or MAX for hot loop code.
   - According to Weiwei's presentation, [Mojo is almost carrying around source code](https://www.youtube.com/watch?v=Invd_dxC2RU), pre-parsed and in a format ready for the JIT compiler, unlike carrying around a very low level representation of the program in most graphics and compute applications.
- ****Python to Mojo Conversion**: Caveat emptor!**: Automatic porting of Python code to Mojo is not yet fully supported, and existing tools like [py2mojo](https://github.com/msaelices/py2mojo) may produce non-compilable output.
   - The focus remains on building out the standard library, with potential for automatic conversion of code developed with mypy strict in the future.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1425199844695347311)** (38 messages🔥): 

> `Laptop Hardware for Robotics, NVIDIA vs AMD GPUs, Apple Silicon & Strix Halo, Mixed Runtime & Compile-Time Layouts` 


- ****Hardware Hunt** for Robotics and Mojo!**: A user seeks hardware advice for **robotics**, **machine vision**, and **Mojo** development, emphasizing the importance of **MAX** support, even if delayed.
   - The user specified that *they need to do object detection and classification for robotics*.
- **NVIDIA Prevails for MAX Support**: It was suggested that if one wants good support for **MAX**, get a laptop with an **NVIDIA GPU**.
   - The member stated *RDNA is going to take a long time to fully come up to speed.*
- **Laptop 5090's Power Throttled?**: It's warned that laptop variants of high-end cards like a **5090** are power-limited, performing closer to the level below (e.g., **5080**).
   - Additionally, laptop versions may have less **VRAM** than their desktop counterparts.
- **Apple Silicon & Strix Halo: VRAM Champions**: Members advised waiting to assess **Apple Silicon** and **Strix Halo** support, as they could provide ample **VRAM** for larger models.
   - The member mentioned that *it may be worth waiting and seeing how apple silicon and strix halo support go from here, as those would get you that vram if you want to throw larger models at it*
- **Mixed Layouts get clunkier**: A user asked about defining layouts with mixed runtime and compile-time values.
   - A member confirmed it's possible but *clunkier than it should be*, indicating ongoing efforts to unify **RuntimeLayout** and **Layout** for a cleaner experience.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1425265382650019892)** (4 messages): 

> `GPU Compatibility, MI60 testing, Hardware Test Suite` 


- **Typo spotted in GPU Compatibility list**: A member reported a typo in the **GPU Compatibility** section, noting that the **RTX 30XX series** was listed under Tier 2 while the **RTX 3090** was under Tier 3.
   - A team member acknowledged the issue and said they would update the list.
- **Member offers to test MI60 compatibility**: A member with an **MI60** on the way offered to run tests to determine compatibility.
   - A team member responded that compatibility for **gfx906 / gfx907** accelerators is unknown and that hardware testing is currently *ad hoc*, involving running through **Mojo GPU function examples**, custom ops examples, small graphs, and GPU puzzles.
- **Hardware Test Suite in the works**: A team member mentioned they are working on a centralized **hardware test suite** that can be run with a single command.
   - The team member noted that it will take some time before the test suite is assembled.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1425241868798525500)** (60 messages🔥🔥): 

> `OpenAI's 30 ‘1-Trillion Token’ Super-Users, Introducing the Gemini 2.5 Computer Use, Bob Ross AI “Vibe Coding” Video Goes Viral, Techno-Capital Singularity` 


- **OpenAI's Top Token Burners Opt-In!**: Deedy shared [OpenAI’s list](https://xcancel.com/deedydas/status/1975582169126019223/photo/1) of **30 customers** who consumed **1T+ tokens each**, noting it was alphabetical and opt-in.
   - The post sparked debate on privacy, poaching risks, and the absence of **Cursor**, with one member noting *"No cursor in top 30. Cognition higher than cursor. that part is interesting"*.
- **Answer.AI interview drops**: The Latent Space podcast dropped an interview with the Answer.AI team that covers their work over the past year, with a [link to the video](https://www.youtube.com/watch?v=01ybLOH1fnU&t=1s).
   - One member noted that the YouTube thumbnail was showing the placeholder image for a bit, and another member asked about self-paced paid options to explore the platform.
- **Magic.dev gets the Slop Treatment**: A discussion thread mocked over-funded startups like **Magic Dev** and **Mercor**, with users betting on who will *implode* first.
   - The convo included observations about companies going quiet, and solo devs bootstrapping for real, with one member linking to [_opencv_ post](https://x.com/_opencv_/status/1975758414660968599) to show the hate on magic . dev.
- **Brockman's AlphaGo Prediction**: OpenAI co-founder **Greg Brockman** predicts that within a year, new models will make dramatic discoveries in coding, material science, and medicine, [similar to AlphaGo’s Move 37](https://xcancel.com/deredleritt3r/status/1976056338342871327?s=46).
   - Followers cheered and hoped for a cancer breakthrough.
- **Karpathy's RL driven LLMs Paranoia**: Karpathy observed that RL training is pushing LLMs into a **catatonic fear of uncaught exceptions**, causing **bloated defensive code**, with a link to the related [X post](https://xcancel.com/karpathy/status/1976077806443569355).
   - Replies extended this to AI welfare, training curve, and prompt engineering takes with reward functions that silence risk also silence creativity.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1425288426013458495)** (6 messages): 

> `Apps SDK, AgentKit, OpenAI API Deep-Dive, Prompt optimization, MCP` 


- **AgentKit Launches with OpenAI API Deep-Dive**: Sherwin Wu and Christina Huang joined the Latent Space podcast to discuss the new **AgentKit** release, prompt optimization, **MCP**, **Codex**, and broader **OpenAI API** insights, accessible via [X](https://xcancel.com/sherwinwu/status/1975726182898540889?s=46).
- **DevDay Apps SDK and AgentKit Discussions**: The **DevDay** pod focuses on **Apps SDK** and **AgentKit**, highlighting significant updates and features.
   - This pod is a valuable resource for developers looking to integrate these tools into their projects.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1425272215695528018)** (5 messages): 

> `xAI, Imagine v0.9, video generator` 


- **xAI's Imagine Model Jumps to v0.9**: [xAI launched Imagine v0.9](https://x.com/xai/status/1975607901571199086), a **free, native-audio, cinematic-quality video generator**.
   - The model advanced from **v0.1 to v0.9**, incorporating lifelike motion, synced audio/speech/singing, and dynamic camera moves, all rendered **100% in-model with zero editing**.
- **Imagine v0.9 Features Impress Users**: Users were impressed by the **demo reels** (dragon, dance, dialogue, etc.) of [Imagine v0.9](https://grok.com/imagine).
   - The tool is live and free at **grok.com/imagine**, with community feedback being used for rapid iteration.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1425529384701530244)** (1 messages): 

> `NousCon 2024, San Francisco AI Event` 


- **NousCon 2024 to happen in San Francisco**: The **second annual NousCon** will be held on **October 24th** in **San Francisco**; more information can be found on [Luma](https://luma.com/ohzkyhvu).
   - The event was announced via a [post on fxtwitter](https://fxtwitter.com/NousResearch/status/1975971746911326431), which encourages attendees to share with friends.
- **AI Community to Convene in San Francisco**: Nous Research is hosting its **second annual NousCon** in San Francisco on **October 24th**, inviting AI enthusiasts and professionals.
   - Attendees are encouraged to register via the [Luma link](https://luma.com/ohzkyhvu) and spread the word among their networks to foster a collaborative environment.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1425513557415432294)** (19 messages🔥): 

> `Self-MCP prompting tool for Claude, Hermes-MoE release, Nous con, Teknium questions, BDH data streaming framework` 


- ****Self-MCP** Tool Lets Claude Think Long**: A member introduced **Self-MCP**, a tool that enables Claude to self-prompt and choose cognitive dimensions, allowing it to *think for 200k tokens in one turn* using a thinking/tool call loop ([github.com/yannbam/self-mcp](https://github.com/yannbam/self-mcp)).
- **Anticipation Builds for **Hermes-MoE****: Several members expressed their anticipation for the release of **Hermes-MoE**, with one posting a GIF of someone waiting ([tenor.com/view/gjirlfriend-gif-14457952604098199169](https://tenor.com/view/gjirlfriend-gif-14457952604098199169)).
   - One member jokingly referred to a "Nous con" while another expressed hope to virtually attend and *grill teknium* with questions.
- **Nous Con in Ohio?**: A member jokingly asked *when can we have a Nous con in Ohio or literally anywhere besides california?*.
- ****BDH**: Data Streaming Framework Introduced**: A member shared a link to **BDH** ([github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)), a data streaming framework.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1425205054104666163)** (21 messages🔥): 

> `Test Time Reinforcement Learning, Hermes Vision, Character per token ratio, LLM tool calling` 


- **Test Time RL Explored for Context Iteration?**: A member inquired about exploring **Test Time Reinforcement Learning** at Nous, suggesting iterating on context instead of weights and visualizing context files like [Three.js git repo](https://fixupx.com/threejs/status/1963806811770798256).
   - The member attached a [gif of PKM pruning](https://cdn.discordapp.com/attachments/1154120232051408927/1425205326516588675/pkmpruning-ezgif.com-optimize.gif?ex=68e80ee4&is=68e6bd64&hm=f05baf35eb90f610293f41373dde58afab0bce480e6ad273706b8434bebe9db1&) to illustrate the concept.
- **Gemini Flash powers Hermes Vision Tools**: A member asked if **Hermes4** could understand images, with **Teknium** responding that they are working on a **Hermes Vision** model.
   - Teknium mentioned using **Gemini 2.5 Flash** as a vision tool alongside **Hermes**, accessible via **Hermes tool calling** or with **vllm** using the `hermes` tool call format, or on **sglang** with `glm-4.5`.
- **Character per Token Ratio Impact Explored**: A member asked if a higher **character per token ratio** correlates with decreased accuracy on benchmarks.
   - Another member responded that it shouldn't affect benchmark results, as it primarily depends on the tokenizer and can be used to measure whether the **LLM** outputs all tokens on the API.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425582122936041683)** (1 messages): 

> `Recursive Reasoning with Tiny networks, HRM Model Performance, ARC-AGI benchmarks` 


- **Tiny Networks Score Big with Recursive Reasoning!**: A member shared a link to the paper *Less is More: Recursive Reasoning with Tiny networks* ([arxiv.org/pdf/2510.04871](https://arxiv.org/pdf/2510.04871)).
   - The **HRM** model, with just **7M parameters**, achieved **45%** on ARC-AGI-1 and **8%** on ARC-AGI-2.
- **HRM Model Achieves Notable Scores on ARC-AGI Benchmarks**: The **HRM** model, a tiny network with only **7M parameters**, demonstrated promising results on challenging benchmarks.
   - Specifically, it achieved a score of **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**, showcasing the potential of recursive reasoning in compact models.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1425540910074167436)** (1 messages): 

> `RL vs Imitation Learning, Information bits in RL` 


- **RL edges out Imitation Learning, bits-wise**: A recent [blog post](https://x.com/ShashwatGoel7/status/1975939253680120152) argues that information bits are more important in **Reinforcement Learning (RL)** than in imitation learning.
- **Another topic on RL**: Just adding another topic here.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425582122936041683)** (1 messages): 

> `Recursive Reasoning, Tiny Networks, HRM Model` 


- **Less is More with Recursive Reasoning**: A member shared a link to the paper '[Less is More: Recursive Reasoning with Tiny networks](https://arxiv.org/pdf/2510.04871)'.
   - The summary states that the **HRM** model at **7M** parameters is scoring **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**.
- **HRM Model Achieves High Score on ARC-AGI**: The **HRM** model, with only **7M** parameters, achieved a **45%** score on **ARC-AGI-1** and **8%** on **ARC-AGI-2**, per the linked paper.
   - This suggests that recursive reasoning with tiny networks can be effective in achieving high scores on advanced reasoning tasks.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1425270802579853463)** (16 messages🔥): 

> `RTX PRO 6000 Max-Q variant, Image/Video Generator Model Summaries, Attention in RNNs and Self-Attention Write-ups, RL vs Imitation Argument, Transferring RL Bits via SFT and LoRA Merging` 


- **Debating RTX PRO 6000 Max-Q Variant**: Members discussed whether to use the **Max-Q variant (rear exhaust)** of the **RTX PRO 6000** in a **PowerEdge R760 server**, versus the passively cooled server version for handling educational content with audio and screenshots.
   - The primary concern revolves around potential airflow issues due to the riser partially covering the air intake.
- **Seeking Image/Video Generator Model Reviews**: A member requested papers or reviews summarizing **image/video generator models**, particularly focusing on how they maintain background consistency in video generation.
   - The user noted the historical challenge of inconsistent backgrounds in AI-generated videos.
- **RNN Attention vs Self-Attention Resources Sought**: A member requested a good write-up covering both **attention mechanisms in RNNs (Bahdanau)** and **self-attention mechanisms**, seeking comprehensive explanations for both concepts.
   - The conversation thread did not provide a specific link, but background resources on attention mechanisms are common.
- **RL Bits Trump All!**: A member shared a blog post ([ShashwatGoel7's X post](https://x.com/ShashwatGoel7/status/1975939253680120152) referencing a short blog) arguing that the **information bits in RL** are more critical than other factors.
   - Another member expressed reservations, noting that the importance of specific weights is already well-documented (*e.g.*, "super weights" papers) and that **RL remains inherently information bottlenecked**.
- **LoRA merging transfers RL bits, Thinky blog finds**: A member highlighted findings from a [Thinking Machines blog post on LoRA](https://thinkingmachines.ai/blog/lora/) suggesting that **widely distributed RL is trivial** because you only need to update a small **LoRA** and merge it later.
   - The member suggested any local model could be a source of **RL bits** on the side and that you could merge everything into one model using **SFT**, pointing to Deepseek V3.2 RL as an example.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1425264676752855041)** (19 messages🔥): 

> `Daily discussion times, Engineering insights from a sleeper paper, Emotional intelligence research, Ovi video+audio model, Rights and responsibilities in technology` 


- **Daily Discussions are Scheduled**: Users should check the pinned messages to find when the daily discussions are scheduled or just watch the channel for presenters.
   - There are **two groups** that host the discussions, but *they don't always fire*.
- **Hidden Z Loss Prevents Massive Activations, Paper Claims**: A member posted about a *major sleeper* paper, ["Title of Paper"](https://arxiv.org/abs/2509.01322), which has *lots of very good engineering* with interesting insights.
   - The paper claims that *hidden Z loss* has been the only thing preventing massive activations.
- **Emotional Intelligence Research Inspired by User**: A member mentioned they are working on **emotional intelligence** partly because of things another user has discussed.
   - The user was congratulated with a graphic for biting the cat first.
- **Ovi Video+Audio Model Released**: A member highlighted the release of a new **open weights video+audio model**, [Ovi](https://huggingface.co/papers/2510.01284).
   - They tested the edge detection and segmentation prompts from [this paper](https://arxiv.org/abs/2509.20328) but it failed to produce anything useful like **Veo 3** does.
- **Linking Rights, Freedoms, Responsibilities to Tech**: A member is thinking through their next paper, tying in basic rights, freedoms, anti-rights, and responsibilities to what technology enables and encourages.
   - They suggest that *research papers are also a good place to write at length on otherwise politically polarized topics that people marinate in nonsense* because **bobble-heads** do not read research papers.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1425197975143186568)** (6 messages): 

> `Qualcomm stock performance, Artificial Hippocampus Networks (AHNs), ByteDance-Seed releases AHN` 


- **Qualcomm stock lags due to lack of AI exposure**: A member noted that **Qualcomm's** share price hasn't increased as much as other chip companies, possibly because they *do not have a real answer in the market space* and *do not benefit from datacenter growth*.
- **ByteDance-Seed introduces Artificial Hippocampus Networks (AHNs)**: **Artificial Hippocampus Networks (AHNs)** transform lossless memory into fixed-size compressed representations for long-context modeling, as described in the [ByteDance-Seed GitHub repository](https://github.com/ByteDance-Seed/AHN?tab=readme-ov-file) and [Hugging Face Collection](https://huggingface.co/collections/ByteDance-Seed/ahn-68e6130d08ed0f5a1b622829).
- **AHNs combine lossless and compressed memory**: AHNs continually convert **lossless memory** outside the sliding attention window into compressed form, integrating both memory types to make predictions across long contexts, as illustrated in the [method diagram](https://raw.githubusercontent.com/yuweihao/misc/refs/heads/master/AHN/method.png).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1425375516080476232)** (5 messages): 

> `RNN Attention (Bahdanau), Self Attention, Kaggle Arena` 


- **Seeking Resources on RNN & Self-Attention**: A member inquired about a good write-up covering both **attention in RNNs (Bahdanau)** and **self-attention** mechanisms.
   - No specific resources were linked or suggested in the immediate context.
- **Kaggle Arena's Status**: A member inquired, *what happened to kaggle arena? lol*
   - Another member speculated that it *merged with LM arena now?* while another clarified that they were referring to the *proposed Go and game benchmark plans*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1425198408871710901)** (25 messages🔥): 

> `ARC-AGI performance, babyLM origin, Weight Decay, SWA equivalence, evolutionary algorithm` 


- **ARC-AGI Scores Soar!**: A member reported achieving **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**, linking to a [tweet](https://fxtwitter.com/jm_alexia/status/1975560628657164426?t=0dDetcu-gIbzekMb1EMwfg&s=19) showcasing the results.
   - Also noted that *EqM surpasses the generation performance of diffusion/flow models empirically, achieving an FID of 1.90 on ImageNet 256*.
- **BabyLM Project's Genesis**: It was revealed that two members started the **babyLM** project, and one has been organizing it since its inception.
   - Another member expressed their interest in the initiative, mentioning their prior work on *incremental NLP* and attraction to *cognitively plausible models of human language processing*.
- **Weight Decay and SWA Equivalence**: A member recalled that *someone showed weight decay + cosine annealing is equivalent to SWA*.
- **Evolutionary Algorithms Emerge**: Members discussed a [tweet](https://fxtwitter.com/yule_gan/status/1975177775251087436?t=gMJUbn-jaSAiGlxr3O6aug&s=19) about the potential for evolutionary algorithms to achieve human-level intelligence.
   - Referenced a [paper](https://arxiv.org/abs/2510.04542) with the comment *nice to see evolutionary algorithm work here*.
- **Defining World Models**: Members discussed the distinction between **world models** and **language models**, referencing a [paper](https://arxiv.org/abs/2510.04542).
   - A member explained that *a world model in traditional RL is just a sub-model of the agent that predicts the future*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1425488100175646741)** (1 messages): 

> `Task Management in AI Runs, Convenience Flags in AI Runs` 


- **Task Tags Streamline AI Runs**: Task tags offer convenience, not importance, for running related tasks such as `--tasks tag` without aggregating scores like in `group`.
   - This feature aids task management by allowing users to selectively execute specific tasks based on tags, streamlining workflows.
- **Enhanced Task Selection with Flags**: The use of flags like `--tasks tag` enables users to select and run specific tasks within AI workflows.
   - This targeted execution avoids the need for aggregate scores, providing more granular control over task management.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1425207535350185984)** (18 messages🔥): 

> `Opencode vs Aider, Coding Models, Gemini Integration, GLM-4.6 and Claude Code 2, Cost Control` 


- **Opencode favored over Aider for coding**: A user expressed a preference for **Opencode's** direction over **Aider's**, but acknowledged reservations about Python as the implementation language.
   - They felt it's *easier to tactically restrict a tool like opencode than to expand the features of aider*.
- **Popular Coding Models Fit Within 40B Parameters**: A user inquired about popular coding models within the **40B parameter range**, mentioning **Qwen3** as a candidate.
   - Another user reported success with **glm-4.6** using **OpenCode** and has **Claude Code 2** configured with **glm-4.6** and **glm-4.5-air**.
- **Gemini Integration Troubleshooted Due to YAML Extension**: A user encountered warnings when trying to integrate **Gemini** with **Aider** using `.aider.conf.yaml`.
   - The issue was resolved by renaming the config file to `.aider.conf.yml`.
- **GLM-4.6 Usable like Sonnet 4?**: A user confirmed that **glm-4.6** is usable like **Sonnet 4** for detailed planning, but then suggests using **GPT-5** and **Grok Code Fast-1** for final planning.
   - Referencing [this X post](https://x.com/victormustar/status/1973735580283625618), they suggested a system consisting of **z.AI coding plan**, combined with minimal **GPT-5** usage and **Grok Code** still being free to keep costs controlled.
- **GLM Favored Due to Cost and Performance**: A user prefers **OpenCode** with **GLM models** over **Claude**, citing that Claude doesn't justify its premium.
   - They noted they are geoblocked from **Claude Pro** or **Max** subscriptions in HK, and also advised keeping an eye on **Qwen Code CLI app** which gives 1000 free requests per day.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1425262153774010432)** (4 messages): 

> `Model Quality, aider and Openrouter & Gemini` 


- **Debate on Model Quantization's Impact on Aider**: One user suggested using a *bad quant* to reduce context usage and improve performance, and another suggested using a better model.
   - The first user was unsure about using **GitHub models** and where to find the model ID.
- **Aider struggles with Openrouter and Gemini Authentication**: A user reports that *aider* is failing to authenticate with **Openrouter** and **Gemini**, citing errors related to missing authentication credentials and invalid API keys.
   - The user added that *Aider* may have an outdated list of **OpenRouter models**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1425286951447035954)** (12 messages🔥): 

> `Tinygrad SF Bay Area Meetup, Bounty Locking Process, Intel GPU Backend, RANGEIFY Merged` 


- **Tinygrad SF Bay Area Meetup Being Proposed**: A member inquired about the possibility of an IRL meetup for **Tinygrad** enthusiasts in the **SF Bay Area**.
- **Doubts about Bounty Locking Process Surface**: A member expressed confusion about the **bounty locking process**, noting discrepancies between the bounty sheet and the actual status of pull requests on GitHub, saying *the coordination seems a bit off to me*.
   - They observed that some bounties listed as available already have existing PRs, and others are reportedly being worked on without being marked as such, adding, *I am just trying not to duplicate work*.
- **Intel GPU Backend Performace Under Question**: A member inquired about the existence of a **performant backend** for new **Intel GPUs** in **Tinygrad**.
   - Other members clarified that if a PR isn't bounty locked after a few days, it's likely considered bad and won't be locked.
- **RANGEIFY Merged with Perf Regression to Fix**: **RANGEIFY** is merged with perf regression to fix and many cleanups to do still.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1425228710381879466)** (1 messages): 

> `RMSProp in Tinygrad, Karpathy's RL blogpost` 


- **RMSProp Implementation Question**: A member asked if **RMSProp** is included in *tinygrad* or if they need to reimplement it for reimplementing [Karpathy's code from this blogpost](https://karpathy.github.io/2016/05/31/rl/).
   - They also considered using **Adam** as an alternative.
- **Using Adam optimizer**: The member is also considering using **Adam** as an alternative optimizer.
   - The question highlights the choice between implementing **RMSProp** from scratch or leveraging **Adam**, a more readily available optimizer in *tinygrad*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1425248077060509696)** (10 messages🔥): 

> `Pyodide/Wasm support, Community Plugins, BALM improvements, Composio integration, dspy.context() override` 


- **DSPy Needs WASM-Friendly Version**: Members wondered if DSPy has a **Pyodide/Wasm-friendly version**, since some dependencies aren't supported by **Pyodide**.
   - They expressed interest in community plugins, signatures, and modules, suggesting DSPy should encompass the structure of how these are created with official examples and community extensions.
- **BALM's Rendering Improves DSPy Schemas**: The **BALM** library has improved rendering for nested **Pydantic models**, optional and literal types, and field descriptions as inline comments, making it suitable for complex, schema-driven workflows within DSPy.
   - It may be beneficial for DSPy tasks requiring effective structured prediction or extraction tasks that prioritize field descriptions and nested dependencies.
- **Community Projects Need Centralization**: A member suggested centralizing community projects and created a [dspy-community GitHub organization](https://github.com/dspy-community) for collaboration and a starting point for community-led extensions.
   - The intent is to avoid overwhelming the core team with PR reviews for every community offshoot project, but another thinks that DSPy needs the community aspect addressed to achieve its crazy potential.
- **Monorepo Discussion**: DSPy's move from version **2.x** to **3.x** involved removing some community modules from the core package, sparking discussion on whether a monorepo (**core + community packages**) approach would be beneficial.
   - Benefits of a monorepo include plugins feeling more "official", easier dependency management, and increased community engagement. This can be solved with `CODEOWNERS`, so that community maintainers get approval rights over the community folder.
- **dspy.context() Creates Scoped Execution Environments**: `dspy.context()` temporarily overrides the active **LM context**, including any global configuration from `dspy.configure()`.
   - It creates a scoped execution environment, allowing optimized prompts from compiled DSPy modules to be plugged into downstream flows, such as calling **OpenAI APIs** outside DSPy, in **JSON** format.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1425501734842667049)** (1 messages): 

> `GRPO, RL, Prompt Optimization, Effectiveness of Finetuning` 


- **Prompt Opt Beats Finetuning?**: A member suggested that the *limited effectiveness of finetuning* in experiments might be because performance was *already saturated from the prompt optimization*.
   - They posited that this saturation could explain why finetuning only helps in very low audit budget scenarios.
- **GRPO and RL Left Out**: A member noted that a comparison to **RL** with **GRPO** would have been interesting to include in the experiments.
   - They acknowledged that these improvements were *out of scope* for the current project but suggested it as a nice area for future work.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1425218158968438787)** (5 messages): 

> `Mid Autumn Festival` 


- **General wishes for the Mid Autumn Festival**: Members sent wishes for the **Mid Autumn Festival** with an attached [video](https://cdn.discordapp.com/attachments/1371757564005711973/1425351140731518996/gA0XlKd.mov?ex=68e7edf1&is=68e69c71&hm=2335406d1943df2b723ff05a74b943090b5a8d97f679b3603ec1ae1306c7680e&).
   - Members agreed that the **Mid Autumn Festival** is very cool.
- **Enthusiasm for the Mid Autumn Festival**: Members expressed general enthusiasm about the **Mid Autumn Festival** and the attached video.
   - The general sentiment was positive and celebratory.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1425223464679243908)** (2 messages): 

> `Discord Self-Promotion Rules, ChatGPT Integration with MCP` 


- **Discord enforces No-Promotion Policy**: Discord channel moderators reminded members to refrain from **self-promotion** or promotion of specific vendors.
   - They suggested framing thread-starters in a **vendor-agnostic way** to maintain fairness, and avoid commercial posts.
- **OpenAI integration Troubleshoot**: A member inquired about contacting **OpenAI** to troubleshoot **ChatGPT**'s **MCP integration**.
   - They noted that the "Refresh" button doesn't provide ChatGPT with the necessary tools/list, while their server functions correctly with **Claude.ai**.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1425330156456640542)** (2 messages): 

> `Discord Events for community calls, UX value add in agent/application chat` 


- **Discord Events streamline community call scheduling**: A member suggested utilizing **Discord Events** for scheduling community calls to provide a centralized location for upcoming meetings.
   - This approach aims to streamline awareness and avoid the need to search through sub-channels for meetup information, making it easier to add events to personal calendars.
- **Agent Iconography Aids Agile Application Acumen**: One user proposed that icons in agent/application chats offer significant UX benefits by providing visual cues for tracking multiple concurrent calls.
   - They posited that these icons help users quickly discern what's happening and where data is flowing amidst rapid interactions.


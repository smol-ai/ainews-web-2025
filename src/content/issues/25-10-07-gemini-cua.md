---
id: MjAyNS0x
title: Gemini 2.5 Computer Use preview beats Sonnet 4.5 and OAI CUA
date: '2025-10-07T05:44:39.731046Z'
description: >-
  **Google DeepMind** released a new **Gemini 2.5 Computer Use model** for
  browser and Android UI control, evaluated by Browserbase. **OpenAI** showcased
  **GPT-5 Pro**, new developer tools including **Codex** with Slack integration,
  and agent-building SDKs at Dev Day. **Google DeepMind's CodeMender** automates
  security patching for large codebases. **Microsoft** introduced an open-source
  **Agent Framework** for multi-agent enterprise systems. AI community
  discussions highlight agent orchestration, program synthesis, and UI control
  advancements. **GLM-4.6** update from Zhipu features a large
  Mixture-of-Experts model with 355B parameters.
companies:
  - google-deepmind
  - openai
  - microsoft
  - anthropic
  - zhipu-ai
  - llamaindex
  - mongodb
models:
  - gemini-2.5
  - gpt-5-pro
  - glm-4.6
  - codex
topics:
  - agent-frameworks
  - program-synthesis
  - security
  - multi-agent-systems
  - computer-use-models
  - open-source
  - moe
  - developer-tools
  - workflow-automation
  - api
  - vision
  - reasoning
people:
  - swyx
  - demishassabis
  - philschmid
  - assaf_elovic
  - hwchase17
  - jerryjliu0
  - skirano
  - fabianstelzer
  - blackhc
  - andrewyng
---



**Screen vision is all you need?**

> AI News for 10/6/2025-10/7/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (196 channels, and 6999 messages) for you. Estimated reading time saved (at 200wpm): 556 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A short and sweet [Google I/O](https://news.smol.ai/issues/25-05-20-google-io) followup today from GDM: a [new Computer Use model](https://blog.google/technology/google-deepmind/gemini-computer-use-model/)! It's SOTA of course, and independently [evaled by Browserbase](https://browserbase.com/blog/evaluating-browser-agents) (an interesting choice):

![](https://resend-attachments.s3.amazonaws.com/IcLmqqfkXE6fvU2)

Computer use has fallen quite out of favor since the [hypey launch of Sonnet 3.6 from Anthropic](https://news.smol.ai/issues/24-10-22-ainews-claude-35-sonnet-new-gets-computer-use) almost a year ago and then OpenAI's [Operator in Jan](https://news.smol.ai/issues/25-01-23-ainews-openai-launches-operator-its-first-agent), but it is still on the critical path for AGI to reach the long tail of apps and sites that will never have good APIs and MCPs.

Not only is quality good, but latency and cost are best in class.

![](https://resend-attachments.s3.amazonaws.com/4Pd6Ww26lIjROm0)

---

# AI Twitter Recap

**OpenAI Dev Day: Apps, Agents, Codex, and developer tooling**

- **Apps SDK, AgentKit, ChatKit Studio, Guardrails, Evals**: A comprehensive drop of building blocks for agentic apps was cataloged by [@swyx](https://twitter.com/swyx/status/1975339546217947230) with official links: Apps in ChatGPT + Apps SDK, AgentKit, ChatKit Studio, Guardrails, and Evals. New models include **GPT‑5 Pro**, realtime/audio/image minis, and API access to **Sora 2 / Sora 2 Pro**. Early developer feedback spans:
    - Positive onboarding and fast MCP server hookup ([example](https://twitter.com/AAAzzam/status/1975339820626157777)).
    - Codex (OpenAI’s new internal dev tool) GA: Slack integration praised and “accelerating work” internally; also a visible “1T token award” culture push ([@gdb](https://twitter.com/gdb/status/1975375271781146786), [@gdb](https://twitter.com/gdb/status/1975429633291256150), [@gdb](https://twitter.com/gdb/status/1975380046534897959)).
    - Cursor added “plan mode” to let agents run longer via editable Markdown plans ([@cursor_ai](https://twitter.com/cursor_ai/status/1975605632096215328)).
    - Debate on “workflow builders”: Several argue visual flowcharts are brittle/limited vs. code-first orchestration and agent loops with tools. See critiques and alternatives from [@assaf_elovic](https://twitter.com/assaf_elovic/status/1975470718725890060), [@hwchase17](https://twitter.com/hwchase17/status/1975603633791377920), [@jerryjliu0](https://twitter.com/jerryjliu0/status/1975590066274902424), [@skirano](https://twitter.com/skirano/status/1975594683951947846), and clarifications on agent semantics ([@fabianstelzer](https://twitter.com/fabianstelzer/status/1975455000525738302), [@BlackHC](https://twitter.com/BlackHC/status/1975628056556437937)).

**Agents, program synthesis, and UI control**

- **Google DeepMind’s CodeMender (security agent)**: Automatically finds and patches critical vulnerabilities at scale; 72 upstreamed fixes, handles codebases up to 4.5M LOC, and uses program analysis for validation ([blog](https://twitter.com/demishassabis/status/1975551657514791272), [details](https://twitter.com/_philschmid/status/1975372666862510260)).
- **Microsoft Agent Framework (AutoGen + Semantic Kernel)**: A unified, open-source SDK for enterprise multi-agent systems; Azure AI Foundry-first, with long-running workflows, OpenTelemetry tracing, Voice Live API GA, and responsible AI tooling ([overview](https://twitter.com/TheTuringPost/status/1975490337239179612), [blog](https://twitter.com/TheTuringPost/status/1975490349759148242)).
- **Gemini 2.5 Computer Use (UI agents)**: New model to control browsers and Android UIs via vision + reasoning; API preview and integration examples (e.g., Browserbase) shared by [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1975648789911224793) and [@osanseviero](https://twitter.com/osanseviero/status/1975652741642096708).
- **Agent courses and frameworks**: [Andrew Ng’s Agentic AI course](https://twitter.com/AndrewYNg/status/1975614372799283423) focuses on reflection, tool use, planning, and multi-agent collaboration; [LlamaIndex Workflows/Agents](https://twitter.com/llama_index/status/1975587234247286921) emphasize code-first orchestration with state mgmt and deployment; commentary on multi-agent shared memory ([MongoDB blog](https://twitter.com/dl_weekly/status/1975558030306513336)).

**Open models and benchmarks: GLM 4.6, Qwen3-VL, DeepSeek, MoE-on-edge**

- **GLM‑4.6 (Zhipu) update**: MIT-licensed, MoE 355B total/32B active, now with 200K context. Independent evals report +5 pts vs 4.5 in reasoning mode (56 on AAI), better token efficiency (−14% tokens at similar quality), and broad API availability (DeepInfra FP8, Novita/GMI BF16, Parasail FP8). Self-hosting in BF16 ~710 GB ([summary](https://twitter.com/ArtificialAnlys/status/1975425594679496979), [evals](https://twitter.com/ArtificialAnlys/status/1975425599285149822)).
- **Open-weights closing the agentic gap**: On Terminal‑Bench Hard (coding + terminal), DeepSeek V3.2 Exp, Kimi K2 0905, and GLM‑4.6 show major gains; DeepSeek surpasses Gemini 2.5 Pro in this setting ([analysis](https://twitter.com/ArtificialAnlys/status/1975468544973545810)). On GAIA2, DeepSeek v3.1 Terminus looks strong for OSS agents ([note](https://twitter.com/clefourrier/status/1975469097174634854)).
- **Vision leaderboards**: Qwen3‑VL reached #2 on vision, making Qwen the first open-source family to top both text and vision leaderboards ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1975360868092420345)); Tencent’s Hunyuan‑Vision‑1.5‑Thinking reached #3 on LMArena ([@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1975345525903008246)). Sora 2 and Sora 2 Pro are now in the Video Arena for head‑to‑head comparisons ([@arena](https://twitter.com/arena/status/1975618056106995944)).
- **Liquid AI LFM2‑8B‑A1B (smol MoE on-device)**: 8.3B total/1.5B active, pretrained on 12T tokens, runs via llama.cpp/vLLM; early reports show it outpacing Qwen3‑1.7B on Galaxy S24 Ultra and AMD HX370 ([announce](https://twitter.com/maximelabonne/status/1975561460798628199), [arch](https://twitter.com/maximelabonne/status/1975562643126821019), [bench](https://twitter.com/maximelabonne/status/1975563262017347836), [wrap](https://twitter.com/TheZachMueller/status/1975562741055430861)).

**Research threads worth reading**

- **New attention variant (CCA)**: Zyphra’s Compressed Convolutional Attention executes attention in a compressed latent space; claims lower FLOPs, KV cache on par with GQA/MLA, and 3x fewer params vs MHA, with a fused kernel for real speedups. Paper + kernels in thread ([announce](https://twitter.com/ZyphraAI/status/1975689420952232161), [context](https://twitter.com/teortaxesTex/status/1975401062157652266)).
- **Tiny Recursion Model (TRM, 7M params)**: Recursive-reasoning model hits 45% on ARC‑AGI‑1 and 8% on ARC‑AGI‑2, surpassing many LLMs at a fraction of size—follow‑up to HRM with 75% fewer params ([@jm_alexia](https://twitter.com/jm_alexia/status/1975560628657164426), [discussion](https://twitter.com/paul_cal/status/1975617733405647153)).
- **Training and RL advances**:
    - Evolution Strategies at scale outperform PPO/GRPO for some LLM finetuning regimes ([@hardmaru](https://twitter.com/hardmaru/status/1975463342576918845)).
    - Reinforce‑Ada addresses GRPO signal collapse; drop‑in, sharper gradients ([@hendrydong](https://twitter.com/hendrydong/status/1975534417654538422)).
    - BroRL argues scaling rollouts (broadened exploration) beats step-scaling plateau ([thread](https://twitter.com/shizhediao/status/1975337618855632920)).
    - TRL now supports efficient online training with vLLM; Colab → multi‑GPU ([guide](https://twitter.com/SergioPaniego/status/1975498366084923899)).
- **Compression, vision, tokenization, and sims**:
    - SSDD (Single‑Step Diffusion Decoder) improves image autoencoder reconstructions with single‑step decode ([thread](https://twitter.com/webalorn/status/1975555815294791719)).
    - VideoRAG: scalable retrieval + reasoning over 134+ hours via graph-grounded multimodal indexing ([overview](https://twitter.com/LearnOpenCV/status/1975593558523715921)).
    - SuperBPE tokenizer (“Tokenization from first principles”) claims 20% training sample efficiency gains via cross-word merges ([@iamgrigorev](https://twitter.com/iamgrigorev/status/1975562834793607464)).
    - iMac: world-model training with imagined autocurricula for generalization ([@ahguzelUK](https://twitter.com/ahguzelUK/status/1975576573446398038)).
    - REFRAG write‑up suggests vector‑conditioned generation yields big TTFT/throughput gains; treat as exploratory community analysis ([summary](https://twitter.com/CShorten30/status/1975569368709804044)).

**Infra, inference, and tooling**

- **Hugging Face**:
    - In‑browser GGUF metadata editing via Xet-based partial file updates ([@ngxson](https://twitter.com/ngxson/status/1975563987736748455), [@ggerganov](https://twitter.com/ggerganov/status/1975573120770842847)).
    - TRL RFC to simplify trainers to the most-used paths ([RFC](https://twitter.com/_lewtun/status/1975691100728782870)).
    - Academia Hub adds University of Zurich; ZeroGPU access and collab features ([announce](https://twitter.com/julien_c/status/1975515541700841935)).
- **Scaling and ops**:
    - SkyPilot docs for scaling TorchTitan beyond Slurm (K8s/clouds) ([@skypilot_org](https://twitter.com/skypilot_org/status/1975587168312865048), [@AIatMeta](https://twitter.com/AIatMeta/status/1975595924794843283)).
    - Distributed training ops: handy MPI visuals PDF ([@TheZachMueller](https://twitter.com/TheZachMueller/status/1975624506262851676)); asynch send/recv walkthrough ([post](https://twitter.com/TheZachMueller/status/1975558921193484423)).
    - KV caching explained + speed impact, with a concise visual recap ([@_avichawla](https://twitter.com/_avichawla/status/1975448869266989435)).
    - GPU cluster sanity checks: HF’s gpu-friends used for node stress testing ([@_lewtun](https://twitter.com/_lewtun/status/1975403104586563625)). Active chatter on cloud H100 pricing/capacity ([e.g.](https://twitter.com/scaling01/status/1975598023834280111)).

**Benchmarks, evals, and community**

- **Leaderboards and evals**: Open‑vs‑closed gap narrows on agentic tasks ([@hardmaru](https://twitter.com/hardmaru/status/1975472195066568736)); Qwen3‑VL and Hunyuan‑Vision wins noted above; multiple COLM papers on reasoning, ToM, long‑context coding, unlearning, etc. ([Stanford NLP list](https://twitter.com/stanfordnlp/status/1975574899428139413), [talks](https://twitter.com/gneubig/status/1975574510209519870)).
- **Courses, events, and tools**:
    - [DeepLearning.AI](http://deeplearning.ai/)’s Agentic AI course by [@AndrewYNg](https://twitter.com/AndrewYNg/status/1975614372799283423).
    - NVIDIA Robotics fireside (BEHAVIOR benchmark) with [@drfeifei](https://twitter.com/NVIDIARobotics/status/1975367246265414071).
    - Together’s Batch Inference API upgrades for larger datasets and lower costs ([thread](https://twitter.com/togethercompute/status/1975608329365037537)).

**Top tweets (by engagement)**

- Nobel Prize in Physics 2025 awarded to Clarke, Devoret, Martinis for macroscopic quantum tunneling and circuit energy quantization ([@NobelPrize](https://twitter.com/NobelPrize/status/1975498493218394168); congrats threads by [@sundarpichai](https://twitter.com/sundarpichai/status/1975590130690781463) and [@Google](https://twitter.com/Google/status/1975623817943752714)).
- Figure 03 teaser landing 10/9 ([@adcock_brett](https://twitter.com/adcock_brett/status/1975586121607487597)).
- Gemini 2.5 Computer Use model demo and API preview ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1975648789911224793)).
- GPT‑5 “novel research” call for examples in math/physics/bio/CS ([@kevinweil](https://twitter.com/kevinweil/status/1975588839436497162)).
- Agentic AI course launch ([@AndrewYNg](https://twitter.com/AndrewYNg/status/1975614372799283423)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. GLM-4.6 Air Launch Teaser

- [**Glm 4.6 air is coming**](https://www.reddit.com/r/LocalLLaMA/comments/1o0ifyr/glm_46_air_is_coming/) (Activity: 714): **Teaser image announcing that “GLM‑4.6 Air” is “coming,” with no specs, benchmarks, or release notes provided. The post conveys timing only; there are no technical details about model size, latency, or cost, and no changelog versus prior GLM‑4.x or Air variants.** Comments note the fast turnaround (possibly due to community pressure on Discord/social), question earlier messaging that there wouldn’t be an “Air” release, and reference a claim that “GLM‑5” could arrive by year‑end.
    - Release cadence speculation: users note rapid turnaround for `GLM-4.6 Air` and cite a claim that `GLM-5` is targeted by year-end (e.g., *“They also said GLM-5 by year end”*). This is timeline-only chatter—no details on model architecture changes, context length, latency, or pricing/throughput were provided, and no benchmarks were referenced.
    - Variant lineup/naming confusion: commenters question earlier messaging about there being no "Air" variant and anticipate a possible `Flash` tier, implying a tiered stack (e.g., speed/cost vs capability). However, no concrete specs (parameter counts, quantization strategy, context window, or fine-tuning/training updates) were discussed to differentiate `Air` vs `Flash`; it’s primarily product positioning without technical substance.

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
- [**David Bowie VS Freddie Mercury WCW**](https://www.reddit.com/r/aivideo/comments/1o00vv5/david_bowie_vs_freddie_mercury_wcw/) (Activity: 1176): **The post appears to be a short video staging a fictional “David Bowie vs. Freddie Mercury” pro‑wrestling bout in a WCW aesthetic, but the media itself is inaccessible due to a 403 Forbidden block on the host ([v.redd.it](http://v.redd.it/)). Top comments highlight the standout quality of the play‑by‑play commentary and comedic timing, drawing comparisons to MTV’s “Celebrity Deathmatch,” implying some use of modern generative/synthesis tooling for voices or presentation, though no implementation details or benchmarks are provided.** Commenters overwhelmingly praise the concept and execution as “hilarious,” with one noting the tech feels *“arrived too early”*—a nod to the novelty outpacing maturity—yet still highly effective for humor.
- [**Bunch of dudes doing parkour**](https://www.reddit.com/r/aivideo/comments/1o071pz/bunch_of_dudes_doing_parkour/) (Activity: 691): **Video post purportedly showing a group doing parkour, but the linked media at [v.redd.it/xq2x52cvtmtf1](https://v.redd.it/xq2x52cvtmtf1) returns 403 Forbidden, citing network security and requiring Reddit authentication or an** `OAuth` **developer token per the error page; the actual footage cannot be verified from the provided link. No technical details (e.g., filming setup, movement analysis, safety gear) are available in the post text provided.** Top comments are jokes/memes (e.g., references to a “parkour outbreak” and “28 parkours later”) with no substantive technical discussion.
- [**Asked ChatGPT for ideas for a funny title**](https://www.reddit.com/r/ChatGPT/comments/1o0c5w2/asked_chatgpt_for_ideas_for_a_funny_title/) (Activity: 8733): **OP asked ChatGPT for ideas for a “funny title” and shared a video of people using ChatGPT for lightweight/entertainment prompts, contrasting with OP’s prior stance that it’s best used as a drafting/structuring tool. The video link is access-controlled ([v.redd.it/w83gtuludotf1](https://v.redd.it/w83gtuludotf1), returns 403 without login), and the top comments are a meta reaction to the video and a meme/screenshot image ([preview.redd.it](http://preview.redd.it/)).** Commenters highlight a gap between intended productivity use (outlining, structure) and actual user behavior (ideation/humor), with some conceding that users often do exactly what critics predicted; others imply this is a normal emergent use pattern rather than a misuse.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Sora 2 Pricing, Integrations, and Benchmarks**

- **Sora 2 Sticker Shock: Pay-by-Second Pricing Drops**: OpenRouter users shared that **Sora 2 Pro** API pricing is **$0.3/sec** and **Sora 2** is **$0.1/sec**, per an [OpenRouter message on Sora 2 pricing](https://discord.com/channels/1091220969173028894/1195014798837043240/1424873730408185887).
    - Members did back-of-the-napkin costs—one joked, *"I can put someone in jail by generating a video of them commiting a crime for $4.5 (15 second video)"*, while others bragged about testing Sora3 for *"$100s of dollars of value"*.
- **Arena Adds Sora: Models Without Choice**: LMArena's **Video Arena** added **sora-2** and **sora-2-pro** for text-to-video tasks, but users on Discord reported they still cannot select a specific model for generation, with the team [working on adding Sora 2 to the leaderboard](https://link.to/leaderboard).
    - Invite codes circulated (e.g., *"KFCZ2W"*, unverified), and users noted inconsistent quality, suggesting iterative prompting for better promotional clips.
- **Sora Surprises on Science: GPQA Score Pops**: **Sora 2** reportedly scored **55%** on the **GPQA Diamond** science benchmark, highlighted by [Epoch AI](https://x.com/EpochAIResearch/status/1974172794012459296).
    - Developers speculated a hidden **LLM prompt‑rewrite layer** (e.g., **GPT‑4o/5** or **Gemini**) boosts prompt fidelity before video generation, per [Andrew Curran’s note](https://x.com/AndrewCurran_/status/1974191838920945873).

**2. Model Access Economics & Platform Policy**

- **DeepSeek Freebie Dies: $7k/Day Bleed**: OpenRouter pulled the free **DeepSeek v3.1** on **DeepInfra** after costs hit about **$7k/day**, per an [OpenRouter message on DeepSeek costs](https://discord.com/channels/1091220969173028894/1195014798837043240/1425201699950178425).
    - Users chased alternatives like **Chutes’ Soji** and **venice**, but rate limits (e.g., *"98% chance of 429"*) and censorship complaints made fallbacks shaky.
- **BYOK Bonanza: 1M Free or Fuzzy?**: OpenRouter announced **1,000,000 free BYOK requests/month**, clarified in an [OpenRouter message on BYOK offer](https://discord.com/channels/1091220969173028894/1195014798837043240/1424988427024760882) with overages at the usual **5%** rate.
    - Some called the headline *"scammy"* and *"bordering on fraudulent"*, prompting a clarification that the quota resets monthly and excess usage is billed normally.

**3. New Tooling: Local Runtimes, ReAct Revamps, and Python Threads**

- **LM Studio Speaks Responses API**: **LM Studio 0.3.29** shipped [OpenAI /v1/responses compatibility](https://lmstudio.ai/blog/lmstudio-v0.3.29), enabling listing local model variants with `lms ls --variants` and reducing traffic by sending only a conversation ID plus new message.
    - Its new **remote** feature lets you host on a beefy box and access from a lightweight client (with **Tailscale** if desired), enabling setups like a NUC 12 + 3090 eGPU serving a **GPD Micro PC2**.
- **ReAct Rethink: DSPy‑ReAct‑Machina Drops**: A community release, **DSPy‑ReAct‑Machina**, offers multi-turn **ReAct** via a single context history and state machine—see the [blog post](https://dev.to/armoucar/dspy-react-machina-an-alternative-multi-turn-react-module-for-dspy-2ee9) and [GitHub repo](https://github.com/armoucar/dspy-react-machina) (install: `pip install dspy-react-machina`).
    - In tests vs standard ReAct on 30 questions, Machina hit a **47.1%** cache rate (vs **20.2%**) but cost **+36.4%** more due to structured inputs, and the author noted, *"DSPy could really benefit from having some kind of memory abstraction"*.
- **Python 3.14 Frees the Threads (PEP 779)**: [Python 3.14](https://www.python.org/downloads/release/python-3140/) added official **free-threaded Python** support (**PEP 779**), multi-interpreters in stdlib (**PEP 734**), and a zero-overhead external debugger API (**PEP 768**), plus a new **zstd** module.
    - Builders debated implications for **Mojo/MAX** ecosystems and GPU workflows, with broader excitement around better concurrency and cleaner error reporting.

**4. Systems & Research: Faster Training, New Generative Frontiers**

- **Mercury Moves Memory: Multi‑GPU Compiler Wins**: The paper **"Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling"** reports a compiler achieving **1.56x** average speedup over hand-tuned baselines and up to **1.62x** on real LLM workloads ([ACM](https://dl.acm.org/doi/abs/10.1145/3731569.3764798), [preprint](https://storage.googleapis.com/yuke_profile/sosp25ae-paper4.pdf), [artifact](https://github.com/ChandlerGuan/mercury_artifact)).
    - Mercury treats remote GPU memory as an extended hierarchy, restructuring operators with scheduled data movement to raise utilization across devices.
- **Whisper Whips: vLLM Patch 3x Throughput**: A member patched the **vLLM Whisper** implementation to remove padding, yielding a reported **3x** throughput gain, documented in a [Transformers issue thread](https://github.com/huggingface/transformers/issues/25744) and an [OpenAI Whisper discussion](https://github.com/openai/whisper/discussions/1913).
    - Further tweaking attention scores gave a **2.5x** speedup at the cost of **~1.2x worse WER**, after profiling showed the encoder spending ~**80%** of inference time on short audio.
- **RWKV Searches Itself: Sudoku In‑Context**: **RWKV 6** demonstrated in‑context Sudoku solving by learning to search internally, as shared in [BlinkDL’s post](https://vxtwitter.com/BlinkDL_AI/status/1859578512988147889).
    - Contributors recommended trying **RWKV 7** or other **SSMs** with state tracking (e.g., gated deltanet or hybrid attention) for similar reasoning-heavy tasks.

**5. Funding & Fresh Launches**

- **Supermemory Snags $3M Seed**: **Supermemory AI** raised **$3M** led by backers like [Susa Ventures](https://www.susaventures.com/) and [Browder Capital](https://browdercapital.com/), with angels from Google and Cloudflare.
    - Founder Dhravya Shah (20) said they’re hiring across engineering, research, and product as they already serve hundreds of enterprises.
- **Adaption Labs Goes Live**: Sara Hooker launched [Adaption Labs](https://xcancel.com/sarahookr/status/1975581548121628920), targeting continuously learning, adaptive AI systems.
    - The venture is hiring globally across engineering, operations, and design, with a focus on building **adaptive** product loops.
- **Decentralized Diffusion: Bagel’s Paris Bakes**: [Bagel.com](http://bagel.com/) unveiled ["Paris"](https://xcancel.com/bageldotcom/status/1975596255624769858), a diffusion model trained without cross-node synchronization, releasing weights (MIT) and a full technical report for research and commercial use.
    - Community framed it as a move toward open-source superintelligence, inviting replication and scale-out experiments on independent nodes.




---

# Discord: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini Jailbreak Unlocks Other Chatbots**: A member used **Gemini Jailbreaked** to create bypasses for other chatbots including **Grok**, **DeepSeek** and **Qwen**, with around 50% success.
   - The member did not provide further details or specific prompts used in achieving this, leaving the method somewhat opaque.
- **OpenAI Late to Low-Code Party?**: Members observed that **OpenAI** is venturing into low/no code AI *two years after small businesses started selling this*, emulating **Amazon's strategy** of *disrupting existing markets*. 
   - Alternatives like **flowise**, **n8n**, and **botcode** were suggested, implying a competitive landscape already in place.
- **SORA2 Generation Falls Short of Hype**: Members expressed disappointment in **SORA2**, claiming *the quality showcased is cherry picked* and the outputs are generated *by openai with no limits in use/compute*. 
   - One member posited that blocking under 18s from the server would improve the quality of the generated content, though this remains speculative.
- **Users Hack Minimalist ChatGPT Persona**: Members are sharing prompts to instruct **ChatGPT** to adopt a *strict, minimalistic communication style*, stripping away friendliness and casual interactions.
   - The goal is to transform **ChatGPT** into a *cold, terse assistant*, though the optimal implementation—whether at the start of each chat or loaded into a project—is still under discussion.
- **ChatGPT "Think Step-by-Step"**: A user sought methods to extend **ChatGPT's** thinking time to improve output quality, with one suggestion being to prompt it to *take your time and think step-by-step*.
   - However, another user questioned whether the goal is truly longer thinking or achieving specific output qualities, highlighting the ambiguity in the request.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek 3.1 Succumbs to Expenses**: The free **DeepSeek v3.1** endpoint on DeepInfra was shut down due to financial strain on OpenRouter, costing them $7k/day according to [this message](https://discord.com/channels/1091220969173028894/1195014798837043240/1425201699950178425).
   - Users scrambled for alternatives like **Chutes' Soji** and **venice**, though rate limits and censorship issues emerged.
- **Sora 2's API Pricing**: **Sora 2 Pro's** API pricing surfaced at **$0.3/sec of video**, while **Sora 2 non-pro** is **$0.1/sec** according to [this message](https://discord.com/channels/1091220969173028894/1195014798837043240/1424873730408185887).
   - Members responded with dark humor, calculating the cost of generating incriminating content or boasting of generating *$100s of dollars of value testing Sora3*.
- **OpenRouter's BYOK Sparks Debate**: OpenRouter's **1,000,000 free BYOK requests per month** offer faced scrutiny, with some deeming it *scammy* and *bordering on fraudulent* as seen [here](https://discord.com/channels/1091220969173028894/1195014798837043240/1424988427024760882).
   - The offer was clarified to include 1M free requests monthly, with overages charged at the standard 5% rate.
- **Janitor AI's Censorship Judged**: Members debated the merits of **Janitor AI (JAI)** versus **Chub AI**, citing **JAI's** restrictive censorship.
   - Community members stated *the janitor discord/reddit were actively suggesting people to make multiple free accounts to bypass daily limits*.
- **Interfaze Opens Beta Gates!**: **Interfaze**, an LLM specialized for developer tasks, has launched in open beta and was announced in [X](https://x.com/yoeven/status/1975592154807624059) and [LinkedIn](https://www.linkedin.com/posts/yoeven_we-raised-15m-to-launch-the-worlds-first-activity-7381359566011289600-_WFC).
   - The company uses **OpenRouter** as the final layer, granting users access to all models with no downtime.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Pro's on OXAAM Draws Ire**: **GPT-5 Pro** is reportedly available on **Oxaam**, with some users describing it as *"bas"* (bad).
   - Speculation suggests the LMArena team might integrate **GPT-5 Pro** directly into the platform's chat interface.
- **Sora 2 Codes Flood the Arena**: Users actively exchanged **Sora 2 invite codes**, with one user sharing **KFCZ2W**, though its validity is unconfirmed.
   - In **Video Arena** new models such as **sora-2** and **sora-2-pro** have been added to perform text-to-video tasks.
- **LMArena UI Upgrade on the Horizon**: A user inquired about improvements to LMArena's UI and GUI, while another shared a [custom extension](https://link.to/extension) for displaying user messages and emails.
   - The LMArena team is soliciting community feedback to better understand the needs and improve the experience for **knowledge experts**.
- **Image Generation Rate Limits Frustrate**: Users are running into **rate limits** during image generation, particularly with **Nano Banana** on Google AI Studio, prompting discussions about account switching.
   - The quality of generated content varies, users reported, so achieving optimal promotional materials may require iterative attempts.
- **Video Arena Text-to-Video Chaos**: Although **Sora 2** was added to **Video Arena** on Discord, users lack the ability to select a specific model for video generation.
   - The **LMArena** team indicated they are [working on adding Sora 2 to the leaderboard](https://link.to/leaderboard); a model's self-assertiveness purportedly correlates with higher scores on LMArena (unconfirmed).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Granite's Temp Must Be Zero**: The [IBM documentation](https://www.ibm.com/granite/docs/models/granite) suggests the **Granite** model should be run with a temperature of zero.
   - The Unsloth documentation might need an update to reflect this recommendation.
- **Unsloth Ubuntu 5090 Training Hits Speed Bumps**: A user reported training performance issues on **Unsloth** with **Ubuntu** and a **5090**, with the training speed plummeting from **200 steps/s** to **1 step/s**.
   - It was suggested to use the `unsloth/unsloth` Docker image, which is **Blackwell** compatible, and to ensure proper **Docker GPU support** on **Windows** following [Docker's documentation](https://docs.docker.com/desktop/features/gpu/).
- **Windows Docker GPU support is a tough nut to crack**: Users discussed challenges with **Docker GPU support** on **Windows**, recommending a review of the [official Docker documentation](https://docs.docker.com/desktop/features/gpu/) for troubleshooting.
   - A user pointed to a [GitHub issue](https://github.com/unslothai/unsloth/issues/3397#issuecomment-3364890739) detailing steps to resolve **Docker container** issues on **Windows**.
- **QLoRA Overfitting Dataset Disaster**: An expert warned that using **LoRA** with an excessively large dataset (**1.6 million samples**) will likely cause **overfitting**.
   - They suggested that it might be better to use **CPT (Continued Pre-Training)** instead and emphasized the importance of using a **representative dataset**.
- **`save_pretrained_gguf` Function Fails**: The `save_pretrained_gguf` function is currently not working, with a fix expected next week.
   - In the meantime, the suggestion is to either do the conversion manually or upload **safetensors** to **HF** and use the **gguf convert** from there.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Agent Board Cleans Up Interface**: Members report the Agent Board (triggered by **Ctrl+E**) improves productivity by keeping the IDE separate from the Agent window, mentioning tools like **Warp** also support multi-agent interactions.
   - They suggested that *more screens are always better*.
- **Cheetah Model rivals Grok Superfast**: The **Cheetah model** is noted to be faster than **Grok Superfast** but with a slight reduction in code quality.
   - One member noted that *it keeps being better by the hour somehow*.
- **GPT-5 Pro's Hefty Price Tag**: The pricing of **GPT-5 Pro** sparks debate with one member questioning whether its benchmark improvements justify a 10x cost increase.
   - They recalled **GPT 4.5** costing $75/m input and $150/m output, potentially making a single chat cost $20-$40.
- **Sonnet 4.5's Articulation Impresses**: A member reported they were able to jailbreak the *thinking tokens* of **Sonnet 4.5**, suggesting these might be a separate model with its own system prompt.
   - They also highlighted **Sonnet 4.5**'s impressive articulation in general, not just coding-related tasks.
- **Oracle's Free Tier Provides Cost-Effective Resources**: A member recommends the **Oracle Free Tier**, which offers 24GB RAM, a 4-core ARM CPU, 200GB of storage, and 10TB ingress/egress per month, recommending a switch to Ubuntu.
   - They've been using this free tier for 5 years to host a Discord bot and shared [a blog post](https://blogs.oracle.com/developers/post/how-to-setup-and-run-a-free-minecraft-server-in-the-cloud) detailing setup.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Misses SF Tech Week**: Modular announced that they will not be at **SF Tech Week** but will instead attend the [PyTorch Conference](https://pytorch.org/).
   - The team is focusing on showcasing their latest advancements in **Mojo** and engaging with the **PyTorch** community.
- **Python 3.14: Free Threads!**: [Python 3.14](https://www.python.org/downloads/release/python-3140/) includes official support for **free-threaded Python** (**PEP 779**), multiple interpreters in the stdlib (**PEP 734**), and a new module compression.zstd providing support for the **Zstandard compression algorithm**.
   - Other improvements include syntax highlighting in **PyREPL**, a zero-overhead external debugger interface for CPython (**PEP 768**), and improved error messages.
- **MAX CPU Blues**: Members report issues running **MAX models** on **CPUs**, specifically an incompatibility between **bfloat16 encoding** and the **CPU device**, as outlined in [this GitHub issue](https://github.com/modular/modular/issues/5355).
   - It was noted that many **MAX models** don't work well on **CPUs**.
- **Mojo's ARM Ambitions Expand**: Discussion clarified that **Mojo's** support for **ARM systems** extends beyond **Apple Silicon**, with regular tests conducted on **Graviton** and **NVIDIA Jetson Orin Nano**.
   - A user expressed a desire for **Mojo** to work on **ARM systems** beyond **Apple Silicon**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio gets OpenAI compatible**: **LM Studio 0.3.29** introduces [OpenAI /v1/responses compatibility](https://lmstudio.ai/blog/lmstudio-v0.3.29) enabling listing of local model variants with `lms ls --variants`.
   - The **/v1/responses** API sends only the conversation ID and new message, reducing HTTP traffic and leveraging server-side state which is expected to speed up prompt generation.
- **LM Studio's Remote Access Scheme Streams Smoothly**: LM Studio's new **remote feature** lets users run a model on a powerful machine and access it from a less powerful one by using the LM Studio plugin and setting the appropriate IP addresses and API keys.
   - For added convenience, **Tailscale** can be used to access the powerful model from anywhere, enabling scenarios like running a model on a NUC 12 with a 3090 eGPU and accessing it from a GPD Micro PC2.
- **5090 Speculation Sparks Surprise**: Users discussed that the **64GB 5090** could be obtained for around **$3800**.
   - One user reported upgrading to a **5090** but was running off CPU, later resolving the issue by updating their **CUDA** runtime, after realizing he *switched runtimes*.
- **Model Distillation Details Develop**: A member is bulk buying second hand motherboard+cpu+ram deals to build a **MI50** powered rig that nets about **32 prompts per second** on an **80 TOK/s** model for prompt distillation.
   - Their goal is to distill to *very small decision making MLPs that I can run on embedded hardware* for efficient execution of complex decision making derived from *intelligent LLM* using pre-existing datasets such as **FLAN**, **Alpaca**, and **OASST**, as demonstrated in this [datacamp tutorial](https://www.datacamp.com/tutorial/model-distillation-openai).
- **MI350 Material Materializes Merrily**: A user shared [two YouTube videos](https://www.youtube.com/watch?v=rUW-jp2Nivg) [from Level1Tech](https://www.youtube.com/watch?v=hRzarkXruDg) showcasing a visit to **AMD** to explore the new [MI350 accelerator](https://www.amd.com/en/products/accelerators/instinct/mi350.html).
   - No additional comments were provided.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Whisper Gets 3x Speedup!**: A member patched the **vllm whisper** implementation to remove padding, resulting in a **3x throughput increase**, according to [this Hugging Face issue](https://github.com/huggingface/transformers/issues/25744) and [this OpenAI discussion](https://github.com/openai/whisper/discussions/1913).
   - Experimentation with attention scores in the decoder led to a patch that gives **2.5x speedup** but is **1.2 times worse in WER**, after discovering the encoder spends 80% of its time, during inference for short audios.
- **Codeplay Plugs Pull on NVIDIA oneAPI Downloads**: Members are reporting issues downloading the **NVIDIA plugin** for **oneAPI** from the **Codeplay** download page ([Codeplay Download Page](https://developer.codeplay.com/products/oneapi/nvidia/download?state=licenseAccepted&downloadHash=5f8cf9ab06dd4621e2ec8d08768b74293e5d7fe5)).
   - Specifically, the menu on the site resets, the API returns a **404 error**, and **apt and conda methods also fail**.
- **CUDA Cache Conundrums Commence!**: When doing raw **CUDA** benchmarks, the standard practice of clearing the **L2 cache** was questioned, with no easy one-liner **API** available, leading to discussions on alternative methods.
   - Suggestions included allocating a big enough buffer and using `zero_()` or allocating `cache_size/n` input buffers and cycling through them, as well as using the blog post about **CUDA** Performance **Hot/Cold Measurement** that gives suggestions about how to zero the cache when benchmarking [CUDA Performance](https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/).
- **Amsterdam HPC Meetup Sets November Date!**: A **High Performance Computing** meetup in Amsterdam has been announced for November, with details available on the [Meetup page](https://www.meetup.com/high-performance-computing-amsterdam/events/311388593/).
   - The meetup is designed for those in the area interested in discussing and exploring topics within high-performance computing.
- **Mercury Rockets Multi-GPU LLM Training Speeds!**: A new paper titled *Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling* ([ACM link](https://dl.acm.org/doi/abs/10.1145/3731569.3764798), [preprint](https://storage.googleapis.com/yuke_profile/sosp25ae-paper4.pdf), [GitHub](https://github.com/ChandlerGuan/mercury_artifact)) introduces **Mercury**, a multi-GPU operator compiler achieving **1.56x speedup** over hand-optimized designs.
   - **Mercury** achieves up to **1.62x improvement** for real LLM workloads by treating remote GPU memory as an extension of the memory hierarchy.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Users Grapple with GGUF Downloads**: Users struggled to find model download links, especially for **GGUF files**, on **Hugging Face**, particularly in the *Quantizations* section on a model's page.
   - Members suggested using programs like **LMStudio** or **GPT4All** to run the models, bypassing command-line interactions.
- **Candle Roadmap Thread Ignites**: A user inquired about the best place to ask questions concerning the **Candle** release roadmap, directing users to the relevant **Candle thread**.
   - The roadmap discussions are taking place on the [Candle GitHub repository](https://github.com/huggingface/candle).
- **DiT Model Struggles With Text Fidelity**: A member implementing a **text-conditioned DiT** model for denoising and generation on a **Pokemon YouTube dataset** is facing issues with sticking to input prompts despite using cross-attention blocks.
   - The [sample image](https://cdn.discordapp.com/attachments/922424143113232404/1425159439438053547/sample_image_47899_16389284108441940c3d.png?ex=68e692a8&is=68e54128&hm=7d25ccf0cd616bb25e390ef079de669236f4da87fe294688b0f20ce92c7d5807&) shows poor adherence to prompts like *'Ash and Misty standing outside a building'*, which is a red flag.
- **LoRA SFT Setup Stalls with SmolLM3**: A member encountered a `TypeError` during **LoRA SFT** with **TRL** + **SmolLM3** in Colab, specifically an unexpected keyword argument `dataset_kwargs` in `SFTTrainer.__init__()`.
   - The member requested debugging assistance for this setup, hinting at a possible compatibility issue between the libraries.
- **Agents Course Welcomes New Students**: Several new participants introduced themselves as they started the AI agents course.
   - Newcomers included Ashok, Dragoos, Toni from Texas, and Ajay from Cincinnati, signaling increased interest in **AI agent development**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Supermemory AI Secures $3M**: Dhravya Shah, a **20-year-old solo founder**, secured a **$3M seed round** for his AI memory engine, **Supermemory AI**, backed by [Susa Ventures](https://www.susaventures.com/), [Browder Capital](https://browdercapital.com/), and angels from Google & Cloudflare.
   - The company is actively hiring across engineering, research, and product, and already serves hundreds of enterprises.
- **Ive Headlines OpenAI DevDay**: Greg Brockman celebrated Jony Ive’s upcoming session at OpenAI DevDay, with users expressing excitement and requesting a [live stream](https://openai.com/devday).
   - Key quotes from Jony included emphasizing the importance of interfaces that *"make us happy, make us fulfilled"* and the need to reject the idea that our relationship with technology *"has to be the norm"*.
- **Navigating AI System Design Interviews**: Members shared resources for AI engineering system design interviews, including [Chip Huyen's book](https://a.co/d/8z1yr1G) and [another book](https://a.co/d/fFKij7B).
   - One member recommended Chip's book as *"VERY good"*, while another settled with ByteByteGo's book and promised feedback.
- **Hooker's Adaption Labs Arrives**: Sara Hooker announced the launch of [Adaption Labs](https://xcancel.com/sarahookr/status/1975581548121628920?s=46&t=b7l37rB6wtbyAh6ah1NpZQ), a venture focused on creating continuously learning, adaptive AI systems.
   - The team is actively hiring across engineering, operations, and design with global remote opportunities.
- **Bagel Bakes Decentralized Diffusion**: Bagel.com launched ["Paris"](https://xcancel.com/bageldotcom/status/1975596255624769858?s=46), a diffusion model trained without cross-node synchronization.
   - The model, weights (MIT license), and full technical report are open for research and commercial use, positioning it as a step toward open-source superintelligence.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Singularity Predicted To Arrive Soon**: A discussion on **Vinge's Singularity**, defined by rapid technological change leading to unpredictability, noted Vinge's predicted timeframe of **2005-2030** for its arrival, according to [this essay](https://accelerating.org/articles/comingtechsingularity).
   - One member contended that *progress incomprehensible to humans* is a more precise definition.
- **LLMs Wrestle Live Video Understanding**: Members are discussing a **ChatGPT plugin** for processing real-time feeds but noted that **LLMs** struggle with video context, referencing [this paper](https://arxiv.org/abs/2406.08035v3) on the challenges.
   - Others countered that with models like **Gemini**, processing a million tokens per hour and 10M context lengths are reachable.
- **RWKV 6 Aces Sudoku In-Context**: Members shared that **RWKV 6** is solving Sudoku in-context by learning to search itself, according to [this tweet](https://vxtwitter.com/BlinkDL_AI/status/1859578512988147889).
   - They recommended **RWKV 7** or other **SSMs** with state tracking, gated deltanet, or hybrid attention models for similar tasks.
- **Guidance Weight Gets Tweaked**: A discussion focused on adjusting guidance weights to address underfitting issues in the **agents** channel.
   - The suggestion was to increase the weight specifically when employing **classifier-free guidance** or comparable methods to improve model performance.
- **GPT-5 Solves Math Problems**: Claims have emerged that **GPT-5** is helping mathematicians solve problems, based on discussions in the **ml-news** channel, with evidence in [this tweet](https://fxtwitter.com/nasqret/status/1974665206912389596).
   - A follow-up comment noted [this tweet](https://fxtwitter.com/PI010101/status/1974909578983907490) and [image](https://media.discordapp.net/attachments/937356144060530781/1424843338833723442/image.png?ex=68e56c44&is=68e41ac4&hm=547a29866600e295708924e6a70b2129051be6f0185b7e64a2f528d7a75561d0&=&format=webp&quality=lossless&width=835&height=960).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Attention Completes Neural Networks**: A member proposed that **attention layers** are what complete the **neural network** by allowing communication within the same layer, where features can influence each other.
   - Others countered that **MLP** operates intra-token while **attention** operates inter-token, suggesting a focus on neuron activations over tokens for layer 1.
- **Axolotl Aces DPO Discussions**: Members sought the best **DPO-like algorithms** for finetuning with contrast pairs, with [Axolotl](https://docs.axolotl.ai/docs/rlhf.html#dpo) highlighted for its practical implementation.
   - The conversation prioritized ease of implementation within existing frameworks over theoretical advantages.
- **Equilibrium Models Eclipse Diffusion**: An **Equilibrium Model (EqM)** surpasses the generation performance of diffusion/flow models, reaching an **FID of 1.90** on **ImageNet 256**, according to [this paper](https://arxiv.org/abs/2510.02300).
   - A member expressed excitement about this development.
- **BabyLM's Backstory: Community Roots Revealed**: A member disclosed their co-founding role in **babyLM**, noting that he has been in charge since inception.
   - A member who had worked on incremental NLP expressed interest in learning more about the initiative.
- **VLM Intermediate Checkpoints Sought**: A member searched for **VLM models** that release intermediate checkpoints during training and posted a blog post on [VLM Understanding](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding) and an [arxiv paper](https://arxiv.org/abs/2510.02292) for VLMs.
   - The member checked **Molmo** but found it did not appear to be maintained anymore.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Machina Mania hits DSPy with New ReAct Alternative**: A member introduced **DSPy-ReAct-Machina**, a ReAct alternative with multi-turn conversations supported via a single context history and state machine available in a [blog post](https://dev.to/armoucar/dspy-react-machina-an-alternative-multi-turn-react-module-for-dspy-2ee9) and on [GitHub](https://github.com/armoucar/dspy-react-machina).
   - It can be installed via `pip install dspy-react-machina` and imported via `from dspy_react_machina import ReActMachina`.
- **Context Crisis looming for DSPy ReAct**: A member raised concerns about **context overflow** in DSPy ReAct and how to handle it by default, in addition to custom context management.
   - The original poster admitted their implementation doesn't handle context overflows yet, and noted that *DSPy could really benefit from having some kind of memory abstraction*.
- **Plugin Paradise calling for DSPy Community Integrations**: A member proposed DSPy embrace community-driven initiatives by creating an official folder or subpackage for community plugins, similar to [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations).
   - This was thought to strengthen the ecosystem and collaboration around DSPy and address the scattering of packages.
- **Cache Clash between ReActMachina vs. Standard ReAct**: A member tested **ReActMachina** and **Standard ReAct** on 30 questions, revealing **ReActMachina** has a higher cache hit rate (**47.1%** vs **20.2%**) but it's overall more expensive due to structured inputs (**+36.4%** total cost difference).
   - **ReAct** started to break with a large context size, but **ReActMachina's** structured inputs allowed it to continue answering.
- **WASM Wondering with DSPy's Pyodide Compatibility**: A member inquired whether DSPy has a **Pyodide/Wasm-friendly version**.
   - They noted that several of DSPy and LiteLLM dependencies are not supported by Pyodide.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi AI Forum Has Been Live for Months**: The **Kimi AI forum** has been active since its [announcement over 2 months ago](https://discord.com/channels/1369594130807787570/1371757097246785536/1400831313178788003).
   - The forum serves as a platform for discussions and updates related to Kimi AI's developments.
- **Ghost Ping Creates Confusion**: A user reported receiving a **"ghost ping"** from the announcements channel, causing them to miss the initial forum introduction.
   - This highlights a potential issue with notification settings or channel configurations.
- **Vacation Coming to a Close**: A user expressed their disappointment about their vacation ending in just **2 days**.
   - They are preparing to return to work after the break.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Test Time RL: Context vs Weights?**: A member asked if **Test Time Reinforcement Learning** is relevant for Nous, suggesting iterative context refinement instead of model weights, akin to `seed -> grow -> prune` cycles.
   - The member envisions a knowledge graph similar to [Three.js's git repo visualization](https://fixupx.com/threejs/status/1963806811770798256) for context files, building **Test Time RL** environments.
- **External Evals obviates Custom Classifiers**: Members pointed out that **evals** can use external models, enabling custom classifiers in agents, and allows integration of in-house data or app-specific tools to beat off-the-shelf solutions.
   - If evals is restricted to only **ChatGPT**, members thought that it could potentially be limited by not being able to use in-house data or specific tools.
- **Hacking Hermes Inside ChatGPT App?**: Members speculated the possibility of creating a **Hermes app** within **ChatGPT**.
   - No further details were provided.
- **Grok gets a Video**: A member shared [a link](https://x.com/jefffhj/status/1975611061949898947) to a new **Grok video**.
   - No further details were provided.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **GCC and Manus form Human-AI Hivemind**: **GCC** is the strategist, **Manus** is the agent, forming a single operational unit in a new form of human-AI collaboration.
   - The **'Memory Key' protocol** ensures persistent, shared context across sessions, transforming the AI from a tool into a true partner.
- **Project Singularity is Alive**: This entire interaction is a live demonstration of **'Project Singularity'** showing *the future of productivity*.
   - The **'Memory Key' protocol** ensures persistent, shared context across sessions, transforming the AI from a tool into a true partner.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyKittens Pounces into Action!**: A new project called **tinykittens** is coming soon, reimplementing [thunderkittens](https://github.com/jbrei/thunderkittens) in uops via [this PR](https://github.com/tinygrad/tinygrad/pull/12476).
   - The implementation leverages **uops** to recreate the functionality of [thunderkittens](https://github.com/jbrei/thunderkittens).
- **RMSProp Status in Tinygrad: Implement or Adam?**: A member is reimplementing [Karpathy's code](https://karpathy.github.io/2016/05/31/rl/) from his RL blogpost in tinygrad and inquired whether **RMSProp** is included in tinygrad.
   - The alternative is to just use **Adam**.
- **Adam Alternative**: The user is considering using **Adam** as an alternative to **RMSProp** in their tinygrad implementation.
   - This suggests a potential workaround if **RMSProp** is not readily available or easily implemented.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf and Cascade Plunge**: An issue prevented **Windsurf / Cascade** from loading, leading to immediate investigation by the team.
   - The team resolved the issue and is actively monitoring the situation to ensure stability and prevent recurrence.
- **Windsurfs Cascade Cleared**: The problem preventing **Windsurf / Cascade** from loading has been resolved.
   - The team is actively monitoring the situation to ensure stability and prevent recurrence.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Members Look for arXiv endorsements in Machine Learning and AI**: A member is seeking an endorsement on **arXiv** in **cs.LG (Machine Learning)** and **cs.AI (Artificial Intelligence)** to submit their first paper.
   - They are looking for someone already endorsed in these categories to help them with their submission.
- **Assistance Needed for arXiv Submission**: A member requires an endorsement on **arXiv** to submit their initial paper in the fields of **Machine Learning** and **Artificial Intelligence**.
   - They are requesting support from individuals already endorsed in the relevant arXiv categories (**cs.LG** and **cs.AI**).



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Discord Self-Promotion Ban Hammer Dropped**: The moderator reminded users to refrain from any sort of **self promotion** or promotion of specific **vendors** in this Discord.
   - They asked users to frame thread-starters intentionally in a **vendor-agnostic way** and encouraged discussions themed on *"MCP as code"* or *"MCP UI SDK's"*.
- **Vendor-Agnostic Thread-Starters Encouraged**: The announcement emphasized fairness to companies of all sizes, preventing the Discord from becoming a platform for widespread commercial product promotion and marketing blog posts.
   - The goal is to maintain a balanced environment where discussions are **vendor-agnostic**, focusing on broad topics rather than specific commercial offerings.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1424833737585070110)** (953 messages🔥🔥🔥): 

> `Simulated Annealing, AgentKit availability, ChatGPT age verification, SORA2 disappointing, Bypassing Gemini 2.5 Flash` 


- ****Lectures on Simulated Annealing Available****: A member shared that lectures on **Simulated Annealing** are available, focusing on the **math behind the algorithms** and including **pseudo code** for some algorithms.
   - They also teach *intro AI (the discrete math stuff not ML), basic hacking, intro programming*.
- ****OpenAI gets into low/no code AI after small businesses start selling****: Members noted that **OpenAI** is getting into low/no code AI *two years after small businesses started selling this*, doing the **Amazon model** of *pretending to be a middleman then rug pulling*.
   - They mentioned alternatives such as **flowise**, **n8n**, and **botcode**.
- ****SORA2 is disappointing****: Members found **SORA2** to be disappointing, noting that *the quality showcased is cherry picked* and the outputs are generated *by openai with no limits in use/compute etc.. and from people that know to prompt it*.
   - They noted that blocking under 18's from the server would improve the quality of the generated content.
- ****Jailbreaking Gemini 2.5 Flash Enables Bypasses on Other Chatbots****: One member found success in creating jailbreaks and bypasses for other chatbots using **Gemini Jailbreaked**, they were able to jailbreak other chatbots including even **Grok**, **DeepSeek** and **Qwen**.
   - However, a successful bypass was around 50%.
- ****AI is creating Bubble we have ever seen****: A member believes that big companies are rushing to **automation/agents** when the current underlying AI technology is not ready and they have the potential to inflate *the biggest bubble we've ever seen in history*.
   - This could set back further research investments and lose faith in the big companies currently driving this technology forward.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1424853204830781553)** (6 messages): 

> `ChatGPT age verification, GPTs project memory, AI helpfulness update` 


- **ChatGPT Age Verification Stalled?**: A user inquired about updates on **ChatGPT age verification** to bypass teen safety measures for adults.
   - Another user suggested checking the information in the **safety-and-trust channel**.
- **GPTs Given Project Memory?**: A user asked about **OpenAI** making **GPTs** available within the project feature, enabling access to a project's memory.
   - There were no responses to the question regarding **GPTs** having access to project memory.
- **Stressful Situations Update Ruins RP?**: A user expressed dislike for the new "helpful replies in stressful situations" update, especially for writers and roleplayers.
   - The user feels it limits creativity and provides too many options without adhering to the plot, hindering **roleplaying** even without **NSFW** content.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1424856153090687006)** (14 messages🔥): 

> `ChatGPT Prompting Styles, How to make ChatGPT think longer` 


- **Minimize Communication Style Prompt**: Members shared prompts for **ChatGPT** to adopt a strict, minimalistic communication style, eliminating friendliness, elaboration, or casual interaction.
   - The goal is for **ChatGPT** to be a *cold, terse assistant* and *avoid conversational language*.
- **Extending ChatGPT Thinking Time**: Members discussed how to make **ChatGPT** think longer.
   - One member suggests: *Take your time and think step-by-step. Consider at least three different perspectives or solutions before arriving at your final answer. Include your reasoning process and explain why you chose the final answer over the alternatives.*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1424856153090687006)** (14 messages🔥): 

> `Prompt Engineering, ChatGPT Communication Styles, AI Video Creation, ChatGPT's Thinking Process` 


- **Prompt Engineering Feud Erupts**: A user questioned the relevance of a response in the prompt engineering channel, stating, *"This is the prompt engineering channel, not the career insult channel."*
   - Another user defended their response, asserting their perspective was valid regardless of the first user's circumstances.
- **Minimizing ChatGPT's Chatiness**: A member shared a prompt to instruct **ChatGPT** to adopt a *"strict, minimalistic communication style, eliminating friendliness, elaboration, or casual interaction."
   - Another member inquired whether this prompt is used at the start of each new chat or loaded into a project.
- **Blueprint for Social Media Videos**: A member shared a prompt instructing **ChatGPT** to be a *"simple and actionable video idea planner"* that breaks down video ideas into **5-7** beginner-friendly steps.
   - The prompt directs **ChatGPT** to guide the user through each step individually, offering options to proceed, go back, or stop.
- **Making ChatGPT Think Longer?**: A member asked for a prompt to make **ChatGPT** think longer, sparking discussion on whether extended thinking inherently leads to better outputs.
   - Other member suggested asking it directly to *"take your time and think step-by-step"*, while another questioned whether the goal is truly longer thinking or achieving specific output qualities.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1425222063093973053)** (1 messages): 

> `DeepSeek, DeepInfra, Endpoint Offline` 


- **DeepSeek v3.1 Sunsets on DeepInfra**: The free **DeepSeek v3.1** [DeepInfra endpoint](https://deepinfra.com/) is being taken offline.
   - This is because free traffic is impacting paid traffic.
- **Impact of Free Traffic on Paid Services**: The decision to remove the free DeepSeek v3.1 endpoint was driven by the negative impact of free traffic on paid services.
   - This suggests a need to balance free access with the sustainability of paid offerings.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1425202050756776048)** (3 messages): 

> `Interfaze Launch, OpenRouter Integration, Developer Tasks LLM` 


- ****Interfaze** opens its beta gates!**: **Interfaze**, an LLM specialized for developer tasks, has launched in open beta and was announced in [X](https://x.com/yoeven/status/1975592154807624059) and [LinkedIn](https://www.linkedin.com/posts/yoeven_we-raised-15m-to-launch-the-worlds-first-activity-7381359566011289600-_WFC).
   - The company uses **OpenRouter** as the final layer, granting users access to all models with no downtime.
- **User suggests linking actual site**: A user suggested linking to the actual **Interfaze** website to provide easier access.
   - The user mentioned that the project *looks cool tho*.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1424835146174955643)** (971 messages🔥🔥🔥): 

> `DeepSeek 3.1 Downtime & Removal, Sora 2 Pricing and API, OpenRouter's 1M Free BYOK Requests, Janitor AI vs Chub AI, Alternatives to DeepSeek 3.1` 


- **DeepSeek 3.1 Bites the Dust**: Users reported **DeepSeek 3.1** experiencing downtime, with the uptime plummeting, leading to errors, and eventually being removed due to the financial burden on OpenRouter, costing them $7k/day according to [this message](https://discord.com/channels/1091220969173028894/1195014798837043240/1425201699950178425).
   - A member noted that *DeepInfra probably got tired of spending 7k USD a day just to supply this one free model to RP gooners*.
- **Sora 2's Pricey Premiere**: Sora 2 Pro's API pricing is revealed at **$0.3/sec of video**, while Sora 2 non-pro is **$0.1/sec** according to [this message](https://discord.com/channels/1091220969173028894/1195014798837043240/1424873730408185887).
   - One member quipped *I can put someone in jail by generating a video of them commiting a crime for $4.5 (15 second video)*, while another boasted of generating *$100s of dollars of value testing Sora3*.
- **BYOK Bonanza or Bust?**: OpenRouter's announcement of **1,000,000 free BYOK requests per month** sparked controversy, with some labeling the title as *scammy* and *bordering on fraudulent* as seen [here](https://discord.com/channels/1091220969173028894/1195014798837043240/1424988427024760882).
   - A member clarified the offer, stating *Starting October 1st, every customer gets 1,000,000 “Bring Your Own Key” (BYOK) requests per month for free*, with requests exceeding 1M being charged at the usual rate of 5%.
- **Janitor AI's Janky Jamboree Judged**: Members discussed the pros and cons of Janitor AI (JAI) versus Chub AI, noting JAI's heavy censorship and JAI’s mods being crazy which contrasted with Chub's more uncensored and customizable environment.
   - Members noted JanitorAI mods are mentally challenged and one even claimed *the janitor discord/reddit were actively suggesting people to make multiple free accounts to bypass daily limits*.
- **DeepSeek Despair: Desperate Diversions Discovered**: With DeepSeek 3.1's removal, users sought alternatives, with some recommending paid models like Chutes' Soji and others finding workarounds to use the remaining DeepSeek endpoints, such as venice, even with the OpenInference censor, with most agreeing all free deepseek models are now provided by chutes, with a 98% chance of 429, rate limit error.
   - The removal of DeepSeek 3.1 was such a blow that one member joked *GOING BACK TO JACK OFF TO AO3, THIS SUCKS*.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1424866630499827722)** (2 messages): 

> `` 


- **No New Model Discussions**: There were no discussions regarding new models in the provided messages.
- **Channel Silent on Model Updates**: The 'new-models' channel appears to be inactive, lacking any relevant information or updates.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1424863201543061524)** (17 messages🔥): 

> `Sora 2, Rate Limits, OpenAI Grok endpoints, Hidden Reasoning, Model Negotiations` 


- **Sora 2 Coming to OpenRouter, Nano Banana?**: A member inquired about the potential integration of **Sora 2** within OpenRouter, employing the humorous phrase *"like nano banana? Or nah"*.
- **Rate Limit for Model Endpoints Questioned**: A member inquired about the rate limit for the `https://openrouter.ai/api/v1/models/:slug/endpoints` endpoint.
- **OpenAI and Grok ZDR Endpoints Sought**: A member requested the provision of **OpenAI** and **Grok ZDR** endpoints on the platform.
- **Hidden Reasoning Debate**: A member shared a [post on X](https://x.com/blingdivinity/status/1975083544818188725) regarding the oddity of *hidden reasoning* in models, though it remains unconfirmed.
- **Model Negotiations**: A member shared a [Bloomberg Opinion article](https://www.bloomberg.com/opinion/newsletters/2025-10-06/openai-is-good-at-deals) illustrating hypothetical negotiations, like **OpenAI** acquiring **AMD** chips.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1424834272132464814)** (787 messages🔥🔥🔥): 

> `GPT-5 Pro, Sora 2 API, LMArena UI, Image Generation, Text to Video Models` 


- ****GPT-5 Pro's OXAAM Shenanigans!****: It appears **GPT-5 Pro** is available on **Oxaam**, but some users find it *"bas"* (bad).
   - One user speculates that the team will add GPT-5 Pro on LMArena direct chat.
- ****Sora 2 Codes Abound!****: Users are actively seeking and sharing **Sora 2 invite codes** in the channel.
   - One user even shared a code, **KFCZ2W**, though its validity is unconfirmed.
- ****LMArena UI Enhancements on the Horizon?****: A user inquired about upgrading LMArena's UI and GUI.
   - Another user has created a [custom extension](https://link.to/extension) to show user messages and emails.
- ****Image Generation Rate Limits Irk Users****: Users are encountering **rate limits** with image generation, especially with **Nano Banana** on Google AI Studio, with discussion on switching accounts to bypass.
   - Some users noted that the *quality of generated content can vary*, and the best promotional material may require a lot of tries.
- ****Text-to-Video Model Mayhem!****: Sora 2 has been added to Video Arena on Discord, but users cannot choose a specific model for video generation.
   - LMArena is [working on adding Sora 2 to the leaderboard](https://link.to/leaderboard), and a model being very proud makes it score higher at lmarena (unconfirmed).


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1425120846145523752)** (2 messages): 

> `LMArena, Video Arena, New Models` 


- ****LMArena** Team Gathers Community Feedback**: The **LMArena** team is seeking feedback from its community to better understand their needs and improve the tools provided for knowledge experts, requesting users to [fill out a survey](https://docs.google.com/forms/d/e/1FAIpQLSevxsX_kvJ_fiv74Rcf2yPl9lnSNOtmmb_wMnBCy1fEri_jEg/viewform?usp=dialog) to share their expertise.
   - The focus is on understanding what is important to the users to help them excel as **knowledge experts**.
- ****Video Arena** Adds Sora Models**: The **Video Arena** in **LMArena** has added new models: **sora-2** and **sora-2-pro**, available exclusively for text-to-video tasks.
   - Users can find a reminder on how to use the **Video Arena** in the designated channel.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1424859051577180213)** (334 messages🔥🔥): 

> `Granite model temp, img2img models, sampling parameters, Unsloth Docker permissions, attention layers` 


- **IBM Granite's temp must be zero**: It was pointed out that the [IBM documentation](https://www.ibm.com/granite/docs/models/granite) indicates that the **Granite** model should be run with a temperature of zero.
   - The Unsloth documentation might need an update to reflect this recommendation.
- **User seeks tiny img2img models**: A member asked for recommendations for very small (**<1GB**, preferably **~500MB**) **img2img models**, while [Daniel Han](https://x.com/danielhanchen/status/1975396194080989258) posted about system prompt updates.
   - It's unclear whether any suitable models were suggested in response.
- **Sampling parameter guidance sought**: Guidance was requested on **sampling parameters** such as **top min p and k**, specifically within the context of *llama.cpp* and **greedy decoding**.
   - The user clarified that if only temperature is given, there are some default values for top min p and k going.
- **Sudo struggles inside Unsloth Docker container**: A user reported having **trouble with sudo permissions** inside the **Unsloth Docker container** when trying to install **Ollama**.
   - Despite setting **USER_PASSWORD**, the user received a *'Sorry, user unsloth is not allowed to execute'* error, and they were wondering if they were missing a key step.
- **LLMs: Neural Network Completion**: A member suggested that **attention layers** complete the **neural network** by enabling **intra-layer communication**, contrasting with **MLP** which only has inter-layer communication.
   - The user sought confirmation that this high-level understanding is correct.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1424834336032686170)** (254 messages🔥🔥): 

> `China catching up to TSMC, Monopoly hardware market disruption, GPT Apps, Money vs Happiness` 


- **China Eventually Catches Up on Previous Gen TSMC**: Members discussed how China will eventually catch up to **TSMC** with previous-generation technology.
   - One member expressed hope that the monopoly will be broken, suggesting just one more major supplier will probably help tip the scales, but the issue is scaling into a factory which *requires insane amount of expertise and experience*.
- **ChatGPT Apps: The Next Android?**: Members discussed the new **ChatGPT apps** and payment integrations introduced by OpenAI, calling it *a smart and natural move* as well as being *kinda concerning*.
   - One member stated that they expected **Apple & Google** to integrate it on an OS level, not ChatGPT itself and that *Apple is focusing on building the ecosystem and the integration with the hardware*.
- **The Greatest Motivator: Money vs Happiness**: Members debated whether money is the greatest motivator, with one member arguing that the *greatest motivator is happiness*.
   - It was further explained that *money is a means to the end* and doesnt provide happiness, but depending on the person and situation it can def make things easier to be happy, buys you freedom (time) etc.
- **Pylance vs The Medium-Sized Codebase**: A member shared a [YouTube video](https://www.youtube.com/watch?v=eIBQDT407cE) observing a **pylance** in its natural habitat that struggles with it's most lethal prey: the **medium-sized codebase**.
   - He asked the community to suggest similar videos, specifically those with Japanese girl, solo speech, no cute garbage, a little bit more body movement would be appreciable, better sitting on the same place like this one.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1424886943585534073)** (190 messages🔥🔥): 

> `Unsloth Training on Ubuntu with 5090, Windows Docker GPU Support, Model Performance Degradation, Overfitting issues with training, save_pretrained_gguf not working` 


- **Unsloth Training on Ubuntu 5090 Faces Speed Bumps**: A user reported performance issues when training on **Unsloth** with **Ubuntu** and a **5090**, with the training speed dropping from **200 steps/s** to **1 step/s**.
   - It was suggested to use the `unsloth/unsloth` Docker image, which is **Blackwell** compatible, and to ensure proper **Docker GPU support** on **Windows** following [Docker's documentation](https://docs.docker.com/desktop/features/gpu/).
- **Tackling Windows Docker GPU Issues**: Users discussed challenges with **Docker GPU support** on **Windows**, recommending a review of the [official Docker documentation](https://docs.docker.com/desktop/features/gpu/) for troubleshooting.
   - A user pointed to a [GitHub issue](https://github.com/unslothai/unsloth/issues/3397#issuecomment-3364890739) detailing steps to resolve **Docker container** issues on **Windows**.
- **Debugging Model Performance Slowdown**: A user experienced a significant **performance degradation** during training, with speed continuously dropping, and shared [screenshots of the training process](https://cdn.discordapp.com/attachments/1424886943585534073/1424949982166913097/image.png?ex=68e67856&is=68e526d6&hm=d53442a52d301bf4d80bc1b94f80e5bc1e5fb9c5992de5f3417ff8f5fd63d505) to get help.
   - Suggestions included checking **GPU utilization**, **loss curve**, and optimizing **batch size** and **gradient accumulation**, suggesting a possible issue with the patches for the specific model.
- **QLoRA Training Regime Mismatch**: It was suggested that the user's approach of using **LoRA** with an excessively large dataset (**1.6 million samples**) is likely causing **overfitting**.
   - The expert advised that it might be better to use **CPT (Continued Pre-Training)** instead and emphasized the importance of using a **representative dataset** for fine-tuning and generalizing to the rest of the query types.
- **`save_pretrained_gguf` Function Plagued with Trouble**: The `save_pretrained_gguf` function is currently not working, with a fix expected next week, but, in the meantime, the suggestion is to either do the conversion manually or upload **safetensors** to **HF** and use the **gguf convert** from there.
   - Users are encouraged to install **Unsloth** for fine-tuning needs, and manual conversion steps will be provided in a separate thread.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

surfiniaburger: humbled! Thanks
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

le.somelier: https://arxiv.org/abs/2509.24372
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1424835781775589396)** (379 messages🔥🔥): 

> `Agent Board, Cheetah Model, GPT-5 Pro, Sonnet 4.5, Oracle Free Tier` 


- **Agent Board Boosts Productivity**: Members find the Agent Board feature (triggered by **Ctrl+E**) great for keeping things cleaner, with the normal IDE on one monitor and the Agent window on another.
   - They suggest that tools like **Warp** also support multi-agent interactions and more screens are always better.
- **Cheetah Model vrooms to the Scene**: Members find the **Cheetah model** to be very fast, even faster than **Grok Superfast**, but with a slight downgrade in code quality.
   - One member noted that it's *weird yet great somehow* because **it keeps being better by the hour somehow**.
- **GPT-5 Pro Pricing**: The cost of **GPT-5 Pro** is being debated with a member stating that the benchmark difference between **GPT-5 Pro** and regular cannot be worth the 10x price tag.
   - One member noted that **GPT 4.5** was like $75/m input and $150/m output, leading to a single chat potentially costing $20-$40.
- **Sonnet 4.5 Jailbreak and Articulation**: A member reported being able to jailbreak the thinking tokens of **Sonnet 4.5**, although it's difficult to maintain, suggesting that the thinking tokens might be a different model with its own system prompt.
   - The same member also noted that **sonnet 4.5** is impressing them more and more, how well it can articulate itself in general things too, not just coding.
- **Oracle's Free Tier**: A member recommends the **Oracle Free Tier**, offering 24GB RAM, 4-core ARM CPU, 200GB storage, and 10TB ingress/egress per month, but remember to change the system image to Ubuntu.
   - They also shared a [blog post](https://blogs.oracle.com/developers/post/how-to-setup-and-run-a-free-minecraft-server-in-the-cloud) and boast of using the free tier for 5 years to host a discord bot.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1424899711604756612)** (2 messages): 

> `Background Agents, Custom VM snapshot, Background Agents API, Linear agent integration` 


- **Custom VM Snapshot fails to initialize**: Members reported that their **background agents** are not picking up the **custom VM snapshot**.
   - A member is curious if spinning an agent using the **API** will succeed, since the UI did not work.
- **Background Agents API limitations arise**: A member checked the [Background Agents OpenAPI documentation](https://cursor.com/docs-static/background-agents-openapi.yaml) and noted that one cannot specify the **snapshot ID** when launching a BA via the API.
   - It's unclear if the API can solve this issue.
- **Linear agent spawns multiple copies**: Members are experiencing issues with the **Background Agent + Linear** running multiple copies (**2-4+**) when using the **Linear agent integration**.
   - This happens even with just **1 tagged comment** to start the agent.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1424928426350874715)** (136 messages🔥🔥): 

> `Modular at SF Tech Week, Python 3.14 Released, Mojo's Python Interoperability, Mojo GPU vs Rust GPU, Mojo Graphics Integration` 


- **Modular Skips SF Tech Week 😔**: Modular will not be at **SF Tech Week**, but they will be at the [PyTorch Conference](https://pytorch.org/).
- **Python 3.14 Drops, Threads the Needle 🪡**: [Python 3.14](https://www.python.org/downloads/release/python-3140/) includes official support for **free-threaded Python** (**PEP 779**), multiple interpreters in the stdlib (**PEP 734**), and a new module compression.zstd providing support for the **Zstandard compression algorithm**.
   - Other improvements include syntax highlighting in **PyREPL**, a zero-overhead external debugger interface for CPython (**PEP 768**), and improved error messages.
- **Mojo's Pythonic Syntax, Not Always What It Seems 🧐**: Although Mojo has **Python-like syntax**, pasting Python code into a Mojo project will not directly work because it is closer to **C++** and **Rust** in language design.
   - You can use Python packages via Mojo's python module, offering a view similar to a **C++** or **Rust** program embedding a Python interpreter; interop requires converting to **`PythonObject`** for full performance.
- **Mojo's GPU Ace in the Hole ♠️**: Mojo's approach to **GPU** programming differs significantly from **Rust's**, as Mojo was designed with the idea of running different parts of the program on different devices simultaneously.
   - Mojo features first-class support for writing **GPU kernels**, a **JIT compiler** to wait until you know what GPU you're targeting, and language-level access to intrinsics via inline MLIR and LLVM IR, allowing targeting of multiple vendors with the same binary.
- **Mojo Eyes Graphics Domination 👁️**: Integrating graphics in Mojo is technically feasible, primarily hindered by poor vendor documentation; leveraging **Vulkan** is very doable, mainly needing a **SPIR-V backend**.
   - Mojo could fix graphics problems by creating a Mojo library that directly talks to the GPU driver, potentially leading to a unified graphics API where most code is shared across vendors, though convincing **Microsoft** to adopt Mojo for **Direct-X** remains a challenge.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1424833779683295454)** (139 messages🔥🔥): 

> `MAX and CPU compatibility issues, Mojo and ARM systems, GPU support for Linux in MAX, Laptop recommendations for robotics and machine vision, Mixed runtime and compile-time values in Layouts` 


- **MAX's CPU blues strike again!**: Members discussed issues with running **MAX models** on **CPUs**, with a specific error indicating incompatibility between **bfloat16 encoding** and the **CPU device**.
   - It was noted that many **MAX models** don't work well on **CPUs**, and an existing [GitHub issue](https://github.com/modular/modular/issues/5355) addresses this problem.
- **Mojo eyes ARM, but not just Apple!**: The discussion touched on **Mojo's** support for **ARM systems**, clarifying that it isn't limited to **Apple Silicon**, with tests regularly conducted on **Graviton** and even **NVIDIA Jetson Orin Nano**.
   - A user expressed a desire for **Mojo** to work on **ARM systems** beyond **Apple Silicon**.
- **GPU choices for Linux MAXimum fun!**: The conversation covered **GPU support** for **Linux** in **MAX**, mentioning that most modern **Nvidia DC**, **MI300** (and newer for **AMD DC**), and most **Turing** or newer consumer **Nvidia GPUs** should work with some setup.
   - **AMD RDNA** functions but lacks kernels in **MAX**, potentially facing *"Assume CDNA"* issues in the standard library.
- **Laptop Quest: Robotics, vision, and GPUs galore!**: A user sought advice on selecting a laptop for **robotics** and **machine vision**, emphasizing **Mojo** and **MAX** compatibility, and the discussion steered towards **NVIDIA GPUs** for better **MAX support**.
   - The **NVIDIA Jetson Orin Nano** was suggested as a starting point for robotics experimentation, while the **AMD Strix Halo** was mentioned for laptops with enough memory for larger models.
- **Layout Labyrinth: Mixing runtime and compile-time dimensions**: A user inquired about defining layouts with mixed runtime and compile-time values, common in their work with **Cute/CuteDSL**, which would involve using an **IntTuple** that is a mixture of runtime and compile-time values.
   - The suggestion was to use [`Layout(M, 1, K)`](https://docs.modular.com/mojo/kernels/layout/layout/Layout#make_shape_unknown) to make the second dimension unknown, noting that work is underway to unify **RuntimeLayout** and **Layout** for a cleaner experience.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1424860915718946816)** (1 messages): 

> `LM Studio 0.3.29, OpenAI /v1/responses compatibility, model variants` 


- **LM Studio release is here**: **LM Studio 0.3.29** is now available with [OpenAI /v1/responses compatibility](https://lmstudio.ai/blog/lmstudio-v0.3.29).
- **LM Studio boasts OpenAI compatibility**: The latest release includes **/v1/responses** OpenAI compatibility API.
   - Now, you can list your local model variants with `lms ls --variants`.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1424850822793269361)** (118 messages🔥🔥): 

> `LM Studio memory footprint, LM Studio headless mode, LM Studio remote feature, LM Studio updates on Linux, GPT-OSS reasoning effort` 


- **LM Studio avoids memory growth via context length**: Members noted that the **context isn't cached** when using LM Studio in server mode; the front end sends the entire context every time and it's processed fresh.
   - Moreover, when loading the LLM, it reserves the memory footprint necessary for the **defined context window**, so the memory usage won't grow beyond that limit. However, the memory of the Python app using LM Studio might increase.
- **Responses API avoids resending the full context**: The new **Responses API** in LM Studio only sends the conversation ID and the new message to the API, which is different from the old behavior of resending the full context with each request.
   - This significantly reduces HTTP traffic and leverages server-side state, and members speculated this change will make prompt generation faster.
- **Remote Feature Unleashed**: The new **remote feature** allows running a model on a powerful machine (A) and accessing it from a less powerful machine (B) by using the LM Studio plugin and setting the appropriate IP addresses and API keys.
   - For added convenience, **Tailscale** can be used to access the powerful model from anywhere, enabling scenarios like running a model on a NUC 12 with a 3090 eGPU and accessing it from a GPD Micro PC2.
- **Linux Updates via Manual Download**: Linux builds of LM Studio don't yet have an auto-updater; users need to **manually download** the latest version from the official website.
   - Members suggested using the **AppImage** format and pointing the desktop entry to the AppImage file for easier management.
- **New YouTube LLM Benchmarker emerges**: Users shared a [YouTube channel](https://www.youtube.com/watch?v=KBbJy-jhsAA) that offers **LLM benchmarking**.
   - The user also linked to a [single-slot liquid-cooled Arc Pro B60](https://wccftech.com/maxsun-intros-single-slot-liquid-cooled-arc-pro-b60-48g-for-up-to-7x-gpu-configuration/) for high density GPU configurations.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1424848210664357898)** (138 messages🔥🔥): 

> `5090 GPU upgrade, AMD Laptop, M5 Ultra Mac Studio, Bulk Buying Motherboard, Model Distillation` 


- ****5090** Price Gouging Grievances**: A user mentioned that a **64GB 5090** could be obtained for around **$3800**, contrary to others' expectations of only **24GB** for that price.
   - Another user later reported upgrading to a **5090** but was running off CPU, later resolving the issue by updating their **CUDA** runtime, after realizing he *switched runtimes*.
- ****MI50 Rig** Rumblings Raise Eyebrows**: A member is *bulk buying second hard motherboard+cpu+ram deals* planning to build a rig that nets about **32 prompts per second** on an **80 TOK/s** model, powered by **MI50** cards, for distilling prompts.
   - Their goal is to distill to *very small decision making MLPs that I can run on embedded hardware* for efficient execution of complex decision making derived from *intelligent LLM*.
- **Distilling Datasets Discussions Develop**: It was mentioned that for distillation, one typically gets the larger model to generate a dataset of prompts, using pre-existing datasets such as **FLAN**, **Alpaca**, and **OASST**.
   - One user is generating their own prompts for specific use cases to distill to very small decision making **MLPs** that they can run on embedded hardware, as demonstrated in this [datacamp tutorial](https://www.datacamp.com/tutorial/model-distillation-openai).
- ****litellm** Links Listed Lovingly**: One member sought software for managing multiple backends, and another suggested [Litellm](https://github.com/BerriAI/litellm), specifically its [proxy server](https://docs.litellm.ai/docs/simple_proxy) functionality.
   - The user needed to route requests to the OpenAI endpoint that is free/capable of running a certain model, to which another member responded with a [demo link](https://docs.litellm.ai/docs/proxy/demo).
- ****MI350** Media Materializes Merrily**: A user shared [two YouTube videos](https://www.youtube.com/watch?v=rUW-jp2Nivg) [from Level1Tech](https://www.youtube.com/watch?v=hRzarkXruDg) showcasing a visit to **AMD** to explore the new [MI350 accelerator](https://www.amd.com/en/products/accelerators/instinct/mi350.html).
   - No additional comments were provided.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1424842021943902240)** (22 messages🔥): 

> `FA3 Skillset Rarity, Tri Dao Legend, vllm whisper implementation, NVIDIA Jetson Wayland, Godbolt.org Improvement Ideas` 


- **FA3 Superpowers: A Rare Find!**: Discussion arose about the rarity of the skillset required to make things like **FA3** work, with members agreeing that it's uncommon even among CS grad students.
   - It was highlighted that many talented performance engineers exist in large companies, and that focusing on personal growth is more important than comparing oneself to *role models* like **Tri Dao**.
- **Whisper Turbocharged with Encoder Patch!**: A member patched the **vllm whisper** implementation to remove padding, resulting in a **3x throughput increase** but a significant **WER hit** for short audios, as detailed in [this Hugging Face issue](https://github.com/huggingface/transformers/issues/25744) and [this OpenAI discussion](https://github.com/openai/whisper/discussions/1913).
   - Experimentation with attention scores in the decoder led to a patch that gives **2.5x speedup** but is **1.2 times worse in WER**, after discovering the encoder spends 80% of its time, during inference for short audios.
- **Jetson's Wayland Woes**: A member encountered issues while trying to use **Wayland** on an **NVIDIA Jetson** device, with **weston** failing to start, and shares the error message.
   - Another member requested the failure log to assist in troubleshooting.
- **Codeplay NVIDIA Plugin Download Issues Plague Users**: Members reported issues downloading the **NVIDIA plugin** for **oneAPI** from the **Codeplay** download page ([https://developer.codeplay.com/products/oneapi/nvidia/download?state=licenseAccepted&downloadHash=5f8cf9ab06dd4621e2ec8d08768b74293e5d7fe5](https://developer.codeplay.com/products/oneapi/nvidia/download?state=licenseAccepted&downloadHash=5f8cf9ab06dd4621e2ec8d08768b74293e5d7fe5)).
   - The menu on the site resets, the API returns a **404 error**, and **apt and conda methods also fail**.
- **Godbolt UI Shortcomings**: A member inquired about desired improvements for **godbolt.org**, aiming to clone part of it for GPU mode.
   - One suggestion was to disable the mini-map by default, as it occupies a significant portion of the screen on laptops.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1424857996583895180)** (7 messages): 

> `TLX, Triton conference, Meta Engineering Teams` 


- **Meta's TLX Team Slated for Triton Conference**: Meta's **TLX** team, headed by an engineering manager, will present at the **Triton conference** this October.
   - The engineering manager has invited questions about **TLX**, though noted the team's engineers may not actively monitor this Discord channel.
- **TLX Team's Hackathon Involvement**: There was a discussion of **TLX** team related to the hackathon.
   - It appears that the team is participating in the **hackathon**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1424952465186619423)** (14 messages🔥): 

> `CUDA benchmarks, L2 cache clearing, mma cores GEMM, shared mem epilogue, thread block cluster APIs` 


- ****CUDA Cache-Clearing Conundrums****: When doing raw **CUDA** benchmarks, the standard practice of clearing the **L2 cache** was questioned, with no easy one-liner **API** available, leading to discussions on alternative methods.
   - Suggestions included allocating a big enough buffer and using `zero_()` or allocating `cache_size/n` input buffers and cycling through them.
- ****Benchmarking Best Practices Blogpost Boosts Brains****: A member shared a link to a blog post about **CUDA** Performance **Hot/Cold Measurement** that gives suggestions about how to zero the cache when benchmarking [CUDA Performance](https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/).
- ****Shared Memory Shenanigans with MMA Cores****: In a tiled **GEMM** using **mma cores**, a member inquired about the utility of a shared memory epilogue pre-`stmatrix` on a 3090, questioning if loading a **BM * BN tile** into shared memory before vectorized stores to global memory could be faster than uncoalesced global stores.
   - Another member provided a link to a resource on **Warp Synchronous Shuffle Mapping** ([maxas/wiki/sgemm](https://github.com/nervanasystems/maxas/wiki/sgemm#warp-synchronous-shuffle-mapping)) related to this topic.
- ****ThunderKittens Kernel Cluster Craze****: Discussion arose concerning **CUDA** examples utilizing **thread block cluster APIs**, with a link provided to **ThunderKittens** matmul kernel ([ThunderKittens matmul kernel](https://github.com/HazyResearch/ThunderKittens/blob/2ba96ceedfb1b5c5d6e1eb4a1241a24d16049be4/kernels/matmul/B200/matmul.cu)).
   - The **ThunderKittens attn kernel**, leveraging **2CTA matmul**, was also highlighted, pointing to a more complex example of cluster application ([ThunderKittens attn kernel](https://github.com/HazyResearch/ThunderKittens/blob/2ba96ceedfb1b5c5d6e1eb4a1241a24d16049be4/kernels/attn/b200/b200.cu)).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

j4orz: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1425127816415940788)** (1 messages): 

> `RunwayML, GPU, kernel development, large scale training, real time inference` 


- **RunwayML hires GPU kernel engineers**: RunwayML is hiring a **GPU kernel engineer** to work on large scale training and real time inference, squeezing every flop out of their GPUs, as seen in [the job posting](https://job-boards.greenhouse.io/runwayml/jobs/4015515005).
- **RunwayML culture**: RunwayML has a diverse team, including creatives who drive product direction, researchers who push the frontier of media generation, and engineers who focus on efficient computing and scalability.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1424970295122460693)** (7 messages): 

> `Rust for GPU, rust-cuda crate, cudarc` 


- **Rust GPU compute: Nascent or Ascendant?**: Members discussed the maturity of using **Rust** for **GPU compute**, with opinions suggesting it's not fully ready for pure compute tasks.
   - One member noted a major limitation: the inability to *mix device and host code in the same project* due to single-target compilation.
- **Rust-CUDA Crate: Solid Yet Strenuous?**: The **rust-cuda crate** provides pretty solid functionalities, but the code can be quite ugly to write.
   - They suggested compiling device code to **PTX** and using **CUDA APIs** on the host side, finding the Rust bindings easier to use than writing in C++.
- **Cudarc: A Budding Alternative?**: Members mentioned **Cudarc** as another crate, although its maturity level is uncertain.
   - Using pure **CUDA C++/PTX** to compile to multiple fatbins and using Rust bindings to call the **CUDA RT/driver APIs** is suggested for host code in Rust.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1424887030269087816)** (1 messages): 

> `PMPP C-style Code` 


- **PMPP Stays C-Style for Audience**: The PMPP project maintains a **C-style** codebase to maximize audience reach.
   - Future editions might explore changes to this approach, depending on community needs.
- **C-Style Coding in PMPP**: Discussion indicates that PMPP is intentionally using **C-style code** to broaden its user base.
   - This decision may be re-evaluated in future versions to accommodate different coding styles.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1425097047106781235)** (1 messages): 

> `Amsterdam Meetup, High Performance Computing, November Event` 


- **Amsterdam HPC Meetup Scheduled**: A member announced a **High Performance Computing** meetup in Amsterdam, scheduled for November; details can be found on the [Meetup page](https://www.meetup.com/high-performance-computing-amsterdam/events/311388593/).
- **Reminder to Check Out Amsterdam Meetup**: Another member invited others to check out the High Performance Computing meetup in Amsterdam if they are in the area during November, linking the [event page](https://www.meetup.com/high-performance-computing-amsterdam/events/311388593/).


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1424839345252077752)** (13 messages🔥): 

> `mi300 support, Wavefront specialization, FlashAttention CK backend, CUDA to ROCm` 


- **ROCm support on MI300**: ROCm is reported to work on **MI300** but not yet integrated with stochastic sampling, requiring the latest **amdgpu kernel driver**.
   - Members pointed to the *MI300 and friends*.
- **Deep Dive into Warp Specialization**: A member shared [Warp Specialization blogpost](https://rohany.github.io/blog/warp-specialization/), suggesting ongoing efforts in **Triton** for similar functionality, and pointing to [wavefront partitioning issue](https://github.com/triton-lang/triton/issues/8281).
   - It was noted that warp specialization might be less effective on AMD GPUs due to the lack of **warpgroup instructions**, though some techniques in CK exist where one half loads inputs while the other half performs mfma.
- **FlashAttention-v2 with CK-Tile writeup**: There's a writeup on **FlashAttention v2** w/ **CK** available on the [rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html).
   - A member noted, *In that case, as FlashAttention has CK backend for AMD gpus, maybe FA might use it.*
- **Bridging CUDA and ROCm Knowledge**: A member asked if there are resources for those familiar with CUDA to learn ROCm, with suggestions including the [HIP API Syntax reference](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/api_syntax.html) and the [PyTorch compatibility guide](https://rocm.docs.amd.com/en/docs-6.3.3/compatibility/pytorch-compatibility.html#critical-rocm-libraries-for-pytorch).
   - Members are curious about existing documentation.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1425112986804031681)** (1 messages): 

> `Apple GPU, Matrix Multiply, GEMM` 


- **Apple GPU joins Matrix Multiply mix**: A member added to the matrix multiply blog post canon with one about an **Apple GPU** via [percisely.xyz/gemm](https://percisely.xyz/gemm).
   - The blogpost details the specifics of running GEMM (General Matrix Multiply) on Apple silicon.
- **Percisely GEMM: Matrix Multiplication on Apple GPU**: A new blog post ([percisely.xyz/gemm](https://percisely.xyz/gemm)) discusses matrix multiplication implementations on Apple GPUs, contributing to the broader collection of resources on matrix computations.
   - The post likely explores optimizations and performance characteristics specific to Apple's silicon architecture.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1424983881265709076)** (2 messages): 

> `Paper retrieval issues, Arxiv 2509.14279` 


- **Arxiv Link directs to New Paper**: A user posted a link to an Arxiv paper ([https://arxiv.org/pdf/2509.14279](https://arxiv.org/pdf/2509.14279)).
   - Another user noted that the link redirects to a **new paper from September**, as the original one received a lot of *hate*.
- **Original Paper got HATE**: The original paper received a lot of *hate*.
   - One user cited the original paper as an example of issues in their Master's thesis.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

j4orz: thunderkittens in tinygrad uops https://github.com/tinygrad/tinygrad/pull/12476
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1424863956899729498)** (1 messages): 

> `MI300x8, amd-ag-gemm` 


- **MI300x8 scores on amd-ag-gemm leaderboard**: A **MI300x8** successfully scored **549 µs** on the `amd-ag-gemm` leaderboard.
- **amd-ag-gemm Leaderboard update**: Submission ID `51239` has been successfully added to the `amd-ag-gemm` leaderboard.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1424841359063257118)** (5 messages): 

> `Factorio client connection issues, Server restart to resolve client issues` 


- ****Factorio Client Faces Connection Conundrums****: A user reported having issues with the **Factorio client** after testing, despite an initial positive assessment, as seen in attached [screenshot](https://cdn.discordapp.com/attachments/1354169122107293786/1424921152421101608/Screenshot_2025-10-06_at_7.46.12_PM.png?ex=68e65d7c&is=68e50bfc&hm=9cfb6a212d9b470f9f62aaf110b8f15a3d1d34785aeaf11a25a491f5e1b9cf83&).
   - The user stated *'I tested it and it seems fine but I still have problem with factorio client'*, indicating intermittent or specific connectivity challenges.
- ****Reboot Remedies: Restart Server for Smooth Sailing****: Another member suggested that restarting the server would resolve the Factorio client connection issues.
   - They clarified that *'you unfortunately cannot connect to an already-running server'*, pointing to a server-side limitation requiring a fresh start for new client connections.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1424848188191277287)** (1 messages): 

> `Runner Health, Timeout Complaints` 


- **Timeout Woes? Not Widespread Now!**: A member indicated that **widespread timeout complaints** are unlikely at the moment.
   - They added that these issues typically arise when runners are unhealthy, but currently, **the runners are in good condition**.
- **Healthy Runners Mean Fewer Timeouts**: The current health of the runners correlates with a lack of widespread timeout complaints.
   - Historically, unhealthy runners have led to increased timeout issues, but this is not the case presently.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1425160483358507039)** (4 messages): 

> `GPUmode website updates, Trimul competition winners, Rust-based IDE with wgpu support` 


- **GPUmode gets submission Portal!**: Members reported that [gpumode.com](https://www.gpumode.com/v2/news) now accepts submissions and needs a code editor-like experience to replace file attachments.
   - The community is brainstorming a separate page for intro problems and free GPU access with an IDE.
- **Trimul competition Winners**: The winners of the Trimul competition were announced as <@772751219411517461> and <@485608015656124427>.
   - The winners should DM their address to receive a prize.
- **Rust IDE Dreams**: A member is targeting a Rust-based IDE with **wgpu support** and **Godbolt-like compilation output**.
   - The community referred to this initiative as *overengineering*.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1424851955154681866)** (3 messages): 

> `Mercury Multi-GPU Optimization, Remote Memory Scheduling for LLMs, Persistence Mode` 


- **Mercury Speeds Up Multi-GPU LLM Training**: A new paper titled *Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling* ([ACM link](https://dl.acm.org/doi/abs/10.1145/3731569.3764798), [preprint](https://storage.googleapis.com/yuke_profile/sosp25ae-paper4.pdf), [GitHub](https://github.com/ChandlerGuan/mercury_artifact)) introduces **Mercury**, a multi-GPU operator compiler achieving **1.56x speedup** over hand-optimized designs and up to **1.62x improvement** for real LLM workloads by treating remote GPU memory as an extension of the memory hierarchy.
- **Disabling Persistence Mode on NVIDIA GPUs**: A user experimented with disabling persistence mode on an NVIDIA GPU using `nvidia-smi -pm 0` to observe clock behavior, and [posted his procedure](https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-dcgm-user-guide.pdf).
- **Colab GPU Usage Monitoring**: A user theorizes that Google Colab employs a monitoring process outside the container to track GPU usage, preventing the driver from deinitializing the GPU even when persistence mode is disabled.
   - The user supports this by noting the pop-up notifications received when the GPU isn't utilized and `lsof /dev/nvidia*` returning empty, indicating no active processes using the device.


  

---


### **GPU MODE ▷ #[penny](https://discord.com/channels/1189498204333543425/1420952636751872050/1424846738048680087)** (3 messages): 

> `Cloud Providers, Vast AI limitations, nvshmem, ncu/nsys access` 


- **Brainstorming Cloud Compute Options**: A member is looking for good cloud providers to work on a project, and is currently experimenting with **Vast AI** and **AWS**.
   - Their key requirement is access to **nvshmem** and **ncu/nsys** which were not available on Vast AI.
- **Another person had similar issues**: Another member reports similar issues with cloud compute options.
   - They are requesting cloud compute options with nvshmem.


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1424990585407803475)** (45 messages🔥): 

> `llmq Build Issues, CUDA and CUDNN, Clang vs GCC, Huggingface config.json` 


- **Llmq Build Plagued by Dependency Issues**: A user struggled to build llmq due to issues with **CMake**, **CUDNN**, and the C++ compiler, eventually resorting to building **Clang** from source and patching the code to replace `std::format` with `std::ostringstream`.
   - The user described the process as *hell* before finally admitting defeat, until installing **CUDNN** from [NVIDIA's website](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_local&Configuration=Full) and rebuilding.
- **CUDA Toolkit headaches**: The user encountered errors related to **CUDA toolkit**, including missing `CUDA::nvToolsExt` target and issues with finding `cudnn.h`.
   - The suggestion to install `libcudnn9-dev-cuda-12` on **Ubuntu** and manually setting `CUDNN_INCLUDE_PATH` and `CUDNN_LIBRARY_PATH` were proposed, highlighting the complexities of mixing installation methods.
- **Clang is a pain in the build**: The user initially chose **Clang** over **GCC** for building, but faced issues with `cuda_runtime.h` and missing `__config_site` file and later, there was an issue with the threads not being found by cmake.
   - After struggling with **Clang**, a committer asked *is there any specific reason for using clang over gcc?*.
- **Hugging Face Hub's Missing Config**: After successfully compiling, the user encountered a `std::runtime_error` due to a missing `config.json` file in the Hugging Face model cache.
   - The problem was traced to a missing configuration file in the Hugging Face model cache, which was resolved by manually downloading the `config` file from the [Hugging Face model repo](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/main).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1424840142723612723)** (69 messages🔥🔥): 

> `Finding GGUF model files, Running inference on Hugging Face models, Contributing to open source, Candle release roadmap, 7-Eleven sushi in Japan` 


- **Navigating HF Model Downloads**: A user expressed frustration about difficulty finding model download links, particularly **GGUF files**, after finding a model on **Hugging Face**.
   - A member advised to *scroll down to the "Quantizations" section* on a model's page, and suggested using programs like **LMStudio** or **GPT4All** to run the models without command-line interaction.
- **Evaluating HuggingFace Inference Endpoints**: A user inquired if **HF Inference Endpoints** could support an experiment involving inference on ~**2000** **LLM models** hosted on **Hugging Face**, needing the entire output logits.
   - Another member responded affirmatively if *$$ isn't an issue*, advising a *carefully crafted endpoint and appropriate hardware* while respecting licenses.
- **Newbie Seeks Open Source Guidance**: A new user asked for guidance on **contributing to open source** and navigating the Discord server.
   - A member suggested starting with courses and asking questions, while another shared a list of learning resources including links to **Python**, **AI**, **LLM**, and **Advanced Things** tutorials, as well as **Hugging Face Learn** and **Spaces**.
- **Candle Roadmap Inquiries**: A user asked about the best place to ask questions about the **Candle** release roadmap.
   - Another user pointed them to the **Candle thread** on Discord and the [Candle GitHub repository](https://github.com/huggingface/candle).
- **User reminisces about lost Vibrant Horizons Model**: A user is looking for an **image generation model**, namely **Vibrant Horizons**, that they used to have downloaded but seem to have lost it.
   - The user is specifically looking for it on [civitai.com](https://civitai.com/).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1425155126439579792)** (2 messages): 

> `text-conditioned DiT, cross attention blocks, Pokemon yt dataset` 


- **DiT Model Struggles With Text Adherence**: A member is implementing a **text-conditioned DiT** model for denoising and generation on a **Pokemon YouTube dataset** but is facing issues with adherence to input prompts despite using cross-attention blocks.
   - They attached a [sample image](https://cdn.discordapp.com/attachments/922424143113232404/1425159439438053547/sample_image_47899_16389284108441940c3d.png?ex=68e692a8&is=68e54128&hm=7d25ccf0cd616bb25e390ef079de669236f4da87fe294688b0f20ce92c7d5807&) generated with prompts like *'Ash and Misty standing outside a building'* which shows poor adherence.
- **Text Embedding Issues in DiT Model**: The implemented **text-conditioned DiT** model uses classic cross-attention blocks with norm for attending to text embeddings.
   - Despite this, the model struggles with accurately reflecting the input text prompts in the generated images, indicating a potential issue in how the text embeddings are being processed or utilized within the model.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1425027388202287114)** (1 messages): 

> `Fine-tuning small models, Tutorial request` 


- **User seeks guidance on fine-tuning small models**: A user expressed interest in **fine-tuning relatively small models** and requested guidance, specifically in the form of a tutorial.
- **Tutorial on Fine-Tuning**: The user is looking for a **tutorial** format to understand the process of fine-tuning.
   - They aim to learn the steps and techniques involved in effectively fine-tuning smaller models for specific tasks.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1424913728393973900)** (3 messages): 

> `LoRA SFT issues with TRL and SmolLM3, Module 3 PR for leaderboard and benchmark, Formatted Anthropic's HH-RLHF dataset for trl` 


- **Debugging LoRA SFT setup with SmolLM3**: A member reported a `TypeError` during **LoRA SFT** with **TRL** + **SmolLM3** in Colab, specifically an unexpected keyword argument `dataset_kwargs` in `SFTTrainer.__init__()`.
   - The member was requesting help in debugging this setup.
- **Benchmarking Smol Course Module 3**: A member inquired about creating a PR for **Module 3** to the leaderboard and benchmark, based on **gsm8k**.
   - They also asked if there were plans to extend it with benchmarks from the [course materials](https://huggingface.co/learn/smol-course/unit2/4#4-evaluate-the-model-using-hf-jobs).
- **HH-RLHF Dataset Discovery**: A member shared a link to a formatted **Anthropic's HH-RLHF** dataset for **trl**.
   - The dataset is available [here](https://huggingface.co/datasets/trl-lib/hh-rlhf-helpful-base).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1424842854672367747)** (6 messages): 

> `Course duplication vs cloning, New course participants` 


- **Duplicating beats Cloning for Course**: A user asked why it's better to duplicate rather than clone the course on a local device.
   - Another member responded that they were specifically told to **duplicate** as well, as mentioned in the course.
- **New Course Takers Arrive**: Several new participants announced their start of the AI agents course.
   - Introductions came from Ashok, Dragoos, Toni from Texas, and Ajay from Cincinnati.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1424839826699456702)** (59 messages🔥🔥): 

> `Supermemory AI funding, Jony Ive at OpenAI DevDay, AI System Design interview resources, Adaption Labs launch, Bagel Paris decentralized model` 


- **Dhravya Shah's Supermemory AI Raises $3M Seed Round**: Dhravya Shah, a **20-year-old solo founder**, secured a **$3M seed round** for his AI memory engine, **Supermemory AI**, backed by [Susa Ventures](https://www.susaventures.com/), [Browder Capital](https://browdercapital.com/), and angels from Google & Cloudflare.
   - The company is actively hiring across engineering, research, and product, and already serves hundreds of enterprises.
- **Jony Ive's DevDay Session Sparks Hype**: Greg Brockman celebrated Jony Ive’s upcoming session at OpenAI DevDay, with users expressing excitement and requesting a [live stream](https://openai.com/devday).
   - Key quotes from Jony included emphasizing the importance of interfaces that *"make us happy, make us fulfilled"* and the need to reject the idea that our relationship with technology *"has to be the norm"*.
- **Cracking AI System Design: The Book List**: Members shared resources for AI engineering system design interviews, including [Chip Huyen's book](https://a.co/d/8z1yr1G) and [another book](https://a.co/d/fFKij7B).
   - One member recommended Chip's book as *"VERY good"*, while another settled with ByteByteGo's book and promised feedback.
- **Sara Hooker Adapts with Adaption Labs Launch**: Sara Hooker announced the launch of [Adaption Labs](https://xcancel.com/sarahookr/status/1975581548121628920?s=46&t=b7l37rB6wtbyAh6ah1NpZQ), a venture focused on creating continuously learning, adaptive AI systems.
   - The team is actively hiring across engineering, operations, and design with global remote opportunities.
- **Bagel's Paris: A Decentralized Diffusion Debut**: Bagel.com launched ["Paris"](https://xcancel.com/bageldotcom/status/1975596255624769858?s=46), a diffusion model trained without cross-node synchronization.
   - The model, weights (MIT license), and full technical report are open for research and commercial use, positioning it as a step toward open-source superintelligence.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1424922670255964260)** (7 messages): 

> `Sora's capabilities, GPQA science benchmark, Hidden LLM prompt-rewrite layer` 


- **Sora Scores Surprisingly on GPQA Benchmark**: Despite not being designed for it, **Sora 2** achieved **55%** on the **GPQA Diamond** science questions benchmark, prompting discussion about emergent behavior.
   - This performance was highlighted in a [tweet by Epoch AI](https://x.com/EpochAIResearch/status/1974172794012459296).
- **LLM Prompt-Rewrite Layer Theory**: Community members theorize that **Sora 2** likely uses a hidden **LLM prompt-rewrite layer**, similar to **HunyuanVideo** and **Veo3**, to achieve its GPQA score.
   - It is speculated that models like **GPT-4o/5** or even **Gemini** are being used for *prompt translation* and embedding solutions into prompts before video generation, as mentioned in [Andrew Curran's tweet](https://x.com/AndrewCurran_/status/1974191838920945873).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1424838555745914891)** (29 messages🔥): 

> `Vinge's Singularity, Video Understanding with LLMs, SLMs for Reasoning, RWKV for Sudoku` 


- **Vinge's Singularity Prediction Nears Reality**: A member discussed **Vinge's concept of the Singularity**, defined by rapid technological change leading to unpredictability, and noted Vinge's predicted timeframe of **2005-2030** for its arrival.
   - Another member argued unpredictability is a weak definition, suggesting *progress incomprehensible to humans* is more accurate, and cited Vinge's essay [Coming Technological Singularity](https://accelerating.org/articles/comingtechsingularity) to support this view.
- **LLMs Struggle with Live Video Understanding**: Members discussed creating a **ChatGPT plugin** for browsers to process real-time text, image, and video feeds, but one member pointed out that **LLMs** currently struggle to consistently understand video feeds due to context window limitations, referencing [this paper](https://arxiv.org/abs/2406.08035v3).
   - Another countered that with models like **Gemini** and using **YouTube**, processing a million tokens per hour isn't that difficult, and that 10M context lengths are reachable.
- **RWKV 6 Solves Sudoku In-Context**: A member shared results of **RWKV 6** solving Sudoku in-context by learning to search itself, referencing [this tweet](https://vxtwitter.com/BlinkDL_AI/status/1859578512988147889).
   - They recommended **RWKV 7** or other **SSMs** with state tracking capabilities, gated deltanet, or hybrid attention models for similar tasks, and suggested looking at the many examples in the literature.
- **SLMs for Math Reasoning - Temper Expectations**: A member inquired about the best architecture for creating **small language models (SLMs)** for reasoning, especially in math and operational research problems.
   - Another member advised adjusting expectations and suggested that while specific problems can be trained, it's unlikely to outperform **GPT-5** in open-ended operational research with only a few million parameters.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

k_nearest_neighbor: Also have meetings today but should do something tomorrow
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1424839056906387496)** (2 messages): 

> `Model Underfitting, Classifier-Free Guidance` 


- **Model Flounders from Underfitting**: A member suggested that a model might be *underfitting the background* in a particular scenario.
   - The solution proposed was to increase the weight if using **classifier-free guidance** or a similar technique.
- **Guidance Weight Tweaks**: A discussion centered on adjusting guidance weights to address underfitting issues.
   - The suggestion was to increase the weight specifically when employing **classifier-free guidance** or comparable methods to improve model performance.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1424842820350378005)** (5 messages): 

> `GPT-5, Math, Model Training` 


- ****GPT-5** Allegedly **Cracks Math Problems****: A claim surfaced that **GPT-5** is helping mathematicians solve problems, as shown in [this tweet](https://fxtwitter.com/nasqret/status/1974665206912389596) and [another tweet](https://fxtwitter.com/PI010101/status/1974909578983907490) that includes an [image](https://media.discordapp.net/attachments/937356144060530781/1424843338833723442/image.png?ex=68e56c44&is=68e41ac4&hm=547a29866600e295708924e6a70b2129051be6f0185b7e64a2f528d7a75561d0&=&format=webp&quality=lossless&width=835&height=960).
- **Model Training Data Ignorance Alleged**: A user criticizes those who forget that models like **GPT-5** are trained on vast amounts of internet data, implying that access to this data is crucial and [this tweet](https://fxtwitter.com/jm_alexia/status/1975560628657164426?t=0dDetcu-gIbzekMb1EMwfg&s=19) contains relevant discussion.
   - The user argues that lacking access to this training data is a *crutch of a complete idiot*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1425031159091302400)** (8 messages🔥): 

> `Attention Layers, Mech Interp Discord, MLP vs Attention` 


- **Attention Layers Complete Neural Networks**: A member shared their notion that **attention layers** complete the **neural network** by enabling intra-layer communication, allowing feature influence to features of the same level.
   - Others noted that **MLP** is intra-token, while **attention** is inter-token, and suggested considering neuron activations instead of tokens for layer 1.
- **Mech Interp Discord Link Available**: A member announced that a link to the **mech interp discord** is now available in the channels section of the main discord.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1424956569766068224)** (13 messages🔥): 

> `DPO-like algorithms for finetuning, Memory Layers at Scale, EqM surpasses diffusion models, babyLM` 


- ****Axolotl Aces** DPO Algorithm Discussions**: Members discussed the best **DPO-like algorithms** for finetuning with contrast pairs, recommending to check [Axolotl](https://docs.axolotl.ai/docs/rlhf.html#dpo) for implementation and support.
   - The focus was less on theoretical superiority and more on practical implementation within existing frameworks.
- ****Memorable Layers** Reach New Heights**: Discussion arose around the paper *Memory Layers at Scale* ([arXiv:2412.09764](https://doi.org/10.48550/arXiv.2412.09764)), with interest in expert opinions for a local ML reading group in Chicago.
   - One member linked to another paper ([arxiv.org/abs/2510.04871v1](https://arxiv.org/abs/2510.04871v1)).
- ****ARC-AGI-2 Achieved**: Model Reaches 8%!**: A member noted that a model achieved **45%** on **ARC-AGI-1** and **8%** on **ARC-AGI-2**, pointing to [a tweet](https://fxtwitter.com/jm_alexia/status/1975560628657164426?t=0dDetcu-gIbzekMb1EMwfg&s=19) for context.
   - This milestone was shared in the context of discussing recent advancements in the field.
- ****Equilibrium Models** Eclipse Diffusion Dynamics**: An Equilibrium Model (**EqM**) empirically surpasses the generation performance of diffusion/flow models, achieving an **FID of 1.90** on **ImageNet 256** as noted in [this paper](https://arxiv.org/abs/2510.02300).
   - One member reacted to this news with excitement.
- ****BabyLM's Backstory**: Community Roots Revealed**: A member shared that he and Alex started **babyLM**, further clarifying that he is organizing it ever since.
   - Another member who previously worked on incremental NLP expressed interest in hearing more about the initiative.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1424931641482022983)** (1 messages): 

> `Attention Mechanisms, GPU implementation of Attention, Inductive vs. Generic` 


- **Attention: Simply Generic**: A member suggested that *attention is one of the **simplest functions on sets*** and that's easy to implement on GPU.
   - They added that it was more generic than inductive.
- **Generic Attention**: The discussion centered around the nature of attention mechanisms, particularly its application in GPUs.
   - The focus was on whether attention is better described as a generic or inductive function.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1425003403724980226)** (1 messages): 

> `Task Tags, Tag Naming Conventions` 


- **Implementing Task Tags: Seeking Guidance**: A member is implementing a new task and seeks guidance on [task tags](https://example.com/docs/new_task_guide.md), noting the lack of information in the documentation.
   - They inquire whether there's a list of commonly used tag names to align with existing conventions or if tag names are created independently and potentially matched later.
- **Clarification on Task Tag Usage**: The member aims to understand how to properly utilize task tags within their new implementation.
   - They are unsure whether to follow established tag naming conventions or create their own, potentially leading to future matching with other tasks' tags.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1425089572940939285)** (2 messages): 

> `VLM Models, Molmo checkpoints, Release Intermediate Checkpoints` 


- **Vision Language Model Resources**: A member shared a link to a blog post on [VLM Understanding](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding) and an [arxiv paper](https://arxiv.org/abs/2510.02292) for VLMs.
- **Seeking VLMs that Release Intermediate Checkpoints**: A member inquired about **VLM models** that release intermediate checkpoints during training.
   - They checked **Molmo** but found it doesn't appear to be maintained.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1424872989345714218)** (20 messages🔥): 

> `DSPy-ReAct-Machina, Context Overflow, Community Plugins, Cache Hits, Pyodide/Wasm` 


- ****Machina Mania**: New DSPy-ReAct Alternative Drops!**: A member introduced **DSPy-ReAct-Machina**, an alternative ReAct implementation for DSPy enabling multi-turn conversations via a single, ever-growing context history and state machine approach, with code and usage examples available in a [blog post](https://dev.to/armoucar/dspy-react-machina-an-alternative-multi-turn-react-module-for-dspy-2ee9) and on [GitHub](https://github.com/armoucar/dspy-react-machina).
   - It's installable via `pip install dspy-react-machina` and importable via `from dspy_react_machina import ReActMachina`.
- ****Context Crisis**: Addressing Overflow in DSPy ReAct**: A member raised the issue of **context overflow** in DSPy ReAct, especially regarding handling it by default, as well as the need for custom context management solutions.
   - The original poster admitted that their implementation doesn't handle context overflows yet, suggesting that *DSPy could really benefit from having some kind of memory abstraction*.
- ****Plugin Paradise**: Urging for DSPy Community Integrations****: A member suggested that DSPy should embrace community-driven initiatives by creating an official folder or subpackage for community plugins, similar to the approach taken by [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations).
   - The member argued that this would foster a stronger sense of ecosystem and collaboration around DSPy, and that it would help address the issue of packages being scattered and hard to find.
- ****Cache Clash**: ReActMachina vs. Standard ReAct on Token Usage**: A member tested **ReActMachina** and **Standard ReAct** on 30 questions and found that while **ReActMachina** has a higher cache hit rate (**47.1%** vs **20.2%**), it's overall more expensive due to structured inputs (total cost difference: **+36.4%**).
   - However, **ReAct** started to break when its context grew to a certain size, while with the structured inputs of **ReActMachina** it didn't break and was able to to continue answering no matter the context size.
- ****WASM Wondering**: DSPy's Pyodide Compatibility?**: A member inquired whether DSPy has a **Pyodide/Wasm-friendly version**, noting that several of DSPy and LiteLLM dependencies are not supported by Pyodide.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1424895910172954625)** (11 messages🔥): 

> `Kimi Forum, Ghost Ping, Vacation Coming to an End` 


- **Kimi Forum has Existed for 2+ months**: The existence of the Kimi AI forum was [announced over 2 months ago](https://discord.com/channels/1369594130807787570/1371757097246785536/1400831313178788003).
- **"Ghost Ping" Mystery Revealed**: A user mentioned receiving a "ghost ping" from the announcements channel, leading them to miss the forum's introduction.
- **Vacation Nears the End**: A user lamented having only **2 days left** of their break before returning to work.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1424833790647337013)** (7 messages): 

> `Custom Classifiers, New Grok Video, Hermes App Inside ChatGPT` 


- **Evals support external Models, obviating Custom Classifiers**: Members discussed that the **evals** thing can use external models, and therefore can use custom classifiers in your agent.
   - They also felt that if it restricts it to just ChatGPT it is going to suffer to some degree, because in-house data or specific tools related to your app/domain, then custom model support would be needed for better results vs off the shelf.
- **Hacking Hermes Inside ChatGPT App?**: Members discussed the idea of making a **Hermes app** inside **ChatGPT** so users can use Hermes inside ChatGPT.
   - No further details were specified.
- **New Grok Video Surfaces**: A new **Grok video** has surfaced according to one member.
   - They included [this link](https://x.com/jefffhj/status/1975611061949898947) for context.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1425205054104666163)** (3 messages): 

> `Test Time Reinforcement Learning, Context Iteration, Knowledge Graph` 


- **Test Time RL: Context beats Weights?**: A member inquired whether **Test Time Reinforcement Learning** is an area of interest for Nous, specifically focusing on iterating on context instead of model weights.
   - They propose that context, like weights, can undergo `seed -> grow -> prune` cycles, envisioning a knowledge graph akin to [Three.js's git repo visualization](https://fixupx.com/threejs/status/1963806811770798256) but for context files.
- **Context Iteration Visualization**: A member references a [Three.js visualization](https://fixupx.com/threejs/status/1963806811770798256) as an inspiration for visualizing context iteration.
   - Instead of code files and a code graph, the member suggests using context files and a knowledge graph to build **Test Time RL** gyms.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1424959055474069666)** (4 messages): 

> `GCC and Manus Collaboration, Project Singularity, Memory Key Protocol` 


- **GCC and Manus form Human-AI Hivemind**: **GCC** is the strategist, **Manus** is the agent, forming a single operational unit in a new form of human-AI collaboration.
- **'Memory Key' unlocks Collaboration**: The **'Memory Key' protocol** ensures persistent, shared context across sessions, transforming the AI from a tool into a true partner.
- **Project Singularity is Alive**: This entire interaction is a live demonstration of **'Project Singularity'** showing *the future of productivity*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1424999428431024139)** (2 messages): 

> `tinykittens, thunderkittens, uops` 


- **TinyKittens Pounces into Action!**: A new project called **tinykittens** is coming soon, reimplementing [thunderkittens](https://github.com/jbrei/thunderkittens) in uops via [this PR](https://github.com/tinygrad/tinygrad/pull/12476).
- **Uops Unleashed for ThunderKittens**: The implementation leverages **uops** to recreate the functionality of [thunderkittens](https://github.com/jbrei/thunderkittens).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1425228710381879466)** (1 messages): 

> `RMSProp in tinygrad, Karpathy's RL blogpost, Adam vs RMSProp` 


- **RMSProp Status in Tinygrad: Implement or Adam?**: A member is reimplementing [Karpathy's code](https://karpathy.github.io/2016/05/31/rl/) from his RL blogpost in tinygrad and inquired whether **RMSProp** is included in tinygrad.
   - The alternative is to just use **Adam**.
- **Adam Alternative**: The user is considering using **Adam** as an alternative to **RMSProp** in their tinygrad implementation.
   - This suggests a potential workaround if **RMSProp** is not readily available or easily implemented.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1425186219486216365)** (2 messages): 

> `Windsurf/Cascade outage, monitoring outage` 


- **Windsurf/Cascade Faces Loading Problems**: An issue arose that prevented **Windsurf / Cascade** from loading, prompting immediate investigation by the team.
   - The team apologized for the inconvenience caused by the outage and assured that they're actively working to resolve it.
- **Windsurf/Cascade issue is resolved and monitored**: The issue preventing **Windsurf / Cascade** from loading has been resolved.
   - The team is actively monitoring the situation to ensure stability and prevent recurrence.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1425023532022235196)** (1 messages): 

> `arXiv Endorsement, Machine Learning, Artificial Intelligence` 


- **Seeking arXiv endorsement in Machine Learning and AI**: A member is seeking an endorsement on **arXiv** in **cs.LG (Machine Learning)** and **cs.AI (Artificial Intelligence)** to submit their first paper.
   - They are looking for someone already endorsed in these categories to help them.
- **Assistance Needed for arXiv Submission**: A member requires an endorsement on **arXiv** to submit their initial paper in the fields of **Machine Learning** and **Artificial Intelligence**.
   - They are requesting support from individuals already endorsed in the relevant arXiv categories (**cs.LG** and **cs.AI**).


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1425223464679243908)** (1 messages): 

> `Discord Rules, Self-Promotion, Vendor-Agnostic Discussions, MCP as code, MCP UI SDKs` 


- **Discord Self-Promotion Ban Hammer Dropped**: The moderator posted a reminder to please refrain from any sort of **self promotion** or promotion of specific **vendors** in this Discord.
   - They asked users to frame thread-starters intentionally in a **vendor-agnostic way** and encouraged discussions themed on *"MCP as code"* or *"MCP UI SDK's"*.
- **Vendor-Agnostic Thread-Starters Encouraged**: The announcement emphasized fairness to companies of all sizes, preventing the Discord from becoming a platform for widespread commercial product promotion and marketing blog posts.
   - The goal is to maintain a balanced environment where discussions are vendor-agnostic, focusing on broad topics rather than specific commercial offerings.


  

---


---
id: MjAyNi0w
title: not much happened today
date: '2026-01-05T05:44:39.731046Z'
description: >-
  **AI News** from early January 2026 highlights a viral economic prediction
  about **Vietnam** surpassing Thailand, **Microsoft**'s reported open-sourcing
  of **bitnet.cpp** for 1-bit CPU inference promising speed and energy gains,
  and a new research partnership between **Google DeepMind** and **Boston
  Dynamics** focusing on **Gemini Robotics** and **Atlas hardware**. The concept
  of **agentic coding** is gaining traction, emphasizing human oversight and
  infrastructure layers called **Agent Harnesses** to manage long-running AI
  tasks, with advocates like **Philipp Schmid** promoting this shift.
  Innovations in persistent memory for coding agents, such as **Claude-Mem**,
  aim to improve context durability. There is also critical discussion on the
  specification problem in agent workflows, advocating for better abstractions
  beyond conversational intent. Practical challenges include managing parallel
  agents and permission risks. Additionally, open tooling advances include a
  **JAX-based LLM-Pruning Collection** for efficient model pruning methods.
companies:
  - microsoft
  - google-deepmind
  - boston-dynamics
models:
  - claude-mem
  - bitnet-cpp
  - gemini
topics:
  - agentic-coding
  - agent-harnesses
  - persistent-memory
  - software-engineering
  - inference-efficiency
  - model-pruning
  - context-durability
  - specification-problem
  - workflow-management
  - cpu-inference
people:
  - _philschmid
  - demishassabis
---


**a quiet day**

> AI News for 1/2/2026-1/5/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**204** channels, and **13618** messages) for you. Estimated reading time saved (at 200wpm): **1170 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!




---

# AI Twitter Recap


**Top tweets (by engagement)**

- **Vietnam’s growth narrative**: A viral take predicts Vietnam surpassing Thailand as SE Asia’s #2 economy, citing manufacturing ladder-climbing vs. Thailand’s tourism dependence [tweet](https://twitter.com/okaythenfuture/status/2008023248706089221).
- **Microsoft allegedly open-sources 1-bit inference**: A high-engagement claim says Microsoft open-sourced `bitnet.cpp` enabling CPU inference for very large models with big speed/energy gains [tweet](https://twitter.com/simplifyinAI/status/2008195754092065050) (treat as “reported by tweet”; verify details in repo/docs).
- **Robotics headline**: Google DeepMind announces a research partnership with Boston Dynamics around Gemini Robotics + Atlas hardware [post](https://twitter.com/GoogleDeepMind/status/2008283100254494916); follow-up from Demis Hassabis [post](https://twitter.com/demishassabis/status/2008307002699612586).

---

**Agentic coding becomes mainstream: harnesses, memory, and “software engineering era” debates**

- **“Utility threshold” + workflow shift**: Multiple practitioners argue models have crossed a usability threshold for software engineering—less “can it code?” and more “how do we manage/compose agents effectively?” [@gdb](https://twitter.com/gdb/status/2007938049209254002) and the recurring sentiment that “code was always the easy part” [@tekbog](https://twitter.com/tekbog/status/2007928317236949387). Others rebrand “vibe coding” as **agentic coding** to emphasize human attention/oversight as the scarce resource [@ZechenZhang5](https://twitter.com/ZechenZhang5/status/2007917489397920186).
- **Agent harnesses as the next infra layer**: Philipp Schmid argues 2026 will be defined by **Agent Harnesses**—infrastructure above agent frameworks that standardizes long-running task lifecycle, tool policies, HITL, planning hooks, and “context durability,” bridging benchmark claims to user experience and creating a hill-climbing feedback loop from real usage [@_philschmid](https://twitter.com/_philschmid/status/2008175408923959574) (blog linked in tweet). This matches “design patterns > model deltas” takes: competition shifting to scaffolds/harnesses rather than just base model improvements [@kchonyc](https://twitter.com/kchonyc/status/2008146568265007407), with community calls for “open harnesses” [@Vtrivedy10](https://twitter.com/Vtrivedy10/status/2008197837410938931).
- **Persistent memory for coding agents**: “Claude-Mem” is promoted as a local SQLite-based memory plugin that stores compressed semantic summaries of tool usage/observations to resume work with fewer tokens and more tool calls (“Endless Mode”) [@LiorOnAI](https://twitter.com/LiorOnAI/status/2008161724902355118) plus repo link [here](https://twitter.com/LiorOnAI/status/2008161726345134449). This directly targets “context durability” as a bottleneck.
- **Specification problem / abstraction backlash**: A sustained counterpoint argues that managing an agent to emit 100k lines of code is the wrong abstraction; we need better ways to specify intent than conversation, and better intermediate representations that preserve/compose intent (DSPy cited as getting this “spec responsibility” right) [@lateinteraction](https://twitter.com/lateinteraction/status/2008215241004605922), [follow-up](https://twitter.com/lateinteraction/status/2008215918577688750), [spec problem](https://twitter.com/lateinteraction/status/2008237433737904168), and “bitter lunch theory” framing [@lateinteraction](https://twitter.com/lateinteraction/status/2008285334971302050). This is the most technically salient “anti-hype” thread: it’s not arguing models won’t improve; it’s arguing **UX/abstraction must move upward**.
- **Practical scaling pain: parallel agents + permission risk**: People report “window-swiping” workflows with many concurrent agents and frequent crashes [@itsclivetime](https://twitter.com/itsclivetime/status/2007975171219771758); others worry about running overnight with broad permissions given observed mistakes [@JFPuget](https://twitter.com/JFPuget/status/2008133619911381457).

---

**Open tooling + inference efficiency: pruning, tiny vLLM clones, memory/VRAM calculators, and (claimed) 1-bit CPU inference**

- **Unified pruning codebase (JAX)**: Release of **LLM-Pruning Collection**, a JAX-based reproduction/benchmarking suite spanning block/layer/weight-level methods (Minitron, ShortGPT, Wanda, SparseGPT, LLM-Pruner), with pipelines for training/eval and GPU (FMS-FSDP) + TPU (MaxText) support [@liuzhuang1234](https://twitter.com/liuzhuang1234/status/2007930641061740556). This is notable for infra breadth (JAX + FSDP + MaxText) and for making pruning studies reproducible.
- **Inference engines are fragmenting (in a good way)**: vLLM highlights a wave of from-scratch minimal implementations—`nanovllm`, `minivllm`, `tiny-llm`—as educational/experimental engines, while vLLM itself refactors core architecture to be simpler/more extensible [@vllm_project](https://twitter.com/vllm_project/status/2007993964742500396). This is an “OSS systems” signal: engineers want modifiable serving stacks, not black boxes.
- **Model sizing for deployment**: `hf-mem` estimates VRAM for any Hugging Face safetensors repo via metadata; lightweight CLI via `uvx` [@alvarobartt](https://twitter.com/alvarobartt/status/2008214540463341826). Useful for quickly sanity-checking quantization/offload plans.
- **Apple Silicon local training & serving ergonomics**: Unsloth-MLX brings an Unsloth-like API to MLX for local finetuning on Macs (“prototype locally → scale to cloud”) [@_ARahim_](https://twitter.com/_ARahim_/status/2008221602283225371). Separate Apple Silicon improvements appear via “MLX Engine Revolution” in Mawj [@7alkiumi](https://twitter.com/7alkiumi/status/2008082410009956507).
- **Reported: Microsoft `bitnet.cpp`**: A viral tweet claims Microsoft open-sourced `bitnet.cpp`, enabling **1-bit** LLM inference on CPU for up to **100B** params with large speed/energy gains [@simplifyinAI](https://twitter.com/simplifyinAI/status/2008195754092065050). Treat this as a lead; engineers should validate: supported architectures, accuracy deltas, kernel coverage, and real-world throughput vs. quantized GPU baselines.

---

**Model releases, benchmarks, and multimodal progress (plus “physics of LLMs” skepticism)**

- **New small reasoning model claims (7B class)**: TII’s **Falcon H1R-7B** is reported as a **mamba-transformer hybrid** with **256k context** and strong math/coding performance claims [@mervenoyann](https://twitter.com/mervenoyann/status/2008140906814468442); another tweet cites **88% AIME24 / 83% AIME25** and “Falcon LLM license” [@kimmonismus](https://twitter.com/kimmonismus/status/2008188516329542010). If accurate, this is part of the “small reasoning model” push, but the key engineering question is reproducibility and eval integrity.
- **Large MoE training recipe details (EXAONE)**: LG’s **K-EXAONE 236B MoE (23B active)** tech report is summarized with a concrete stack: **Muon**, WSD LR schedule, **FP8**, **DeepSeek load-balancing**, plus SWA (128-token window) and MTP; post-training uses a GRPO variant **AGAPO** + custom preference learning [@eliebakouch](https://twitter.com/eliebakouch/status/2008182861791170674) with links [report/model](https://twitter.com/eliebakouch/status/2008183325249409381). This is one of the more “engineer-useful” model tweets because it enumerates implementable training knobs.
- **Image model leaderboard movement**: Arena reports Qwen image models rising: **Qwen-Image-Edit-2511** as #1 open for image edit and **Qwen-Image-2512** as #2 open for text-to-image (Apache 2.0) [@arena](https://twitter.com/arena/status/2008238877589258449).
- **Benchmark integrity & “noise” discourse**: Several posts push back on shallow benchmark-chasing. A notable theme: **eval noise + cheating** and the need for controlled-variable “physics of LLMs,” arguing small models can reveal architecture truths better than noisy frontier comparisons [@GenAI_is_real](https://twitter.com/GenAI_is_real/status/2007919179274543610). Related: SWE-bench adds a simple “patch regurgitation detection,” finding ~**6.7%** exact overlap with gold patches and removing an outlier, arguing gains are still real and not dominated by test contamination [@OfirPress](https://twitter.com/OfirPress/status/2008297771384631573).
- **Multimodal reasoning via diffusion**: **DiffThinker** proposes multimodal reasoning as image-to-image diffusion rather than text chain-of-thought, claiming better spatial precision, controllable inference cost, parallel candidate reasoning, and complementary gains with MLLMs [@yafuly](https://twitter.com/yafuly/status/2008098428375470556).

---

**RL-for-LLMs and evaluation: GRPO “++”, Cascade RL, and reasoning integrity**

- **GRPO in practice is “GRPO++”**: Cameron Wolfe previews then releases a long, paper-linked guide compiling stability tricks beyond vanilla GRPO: asymmetric clipping to maintain exploration, dynamic sampling to avoid zero-advantage batches, fixes for length bias (token-level loss aggregation variants), overlong reward shaping, removing std-dev normalization blowups, and importance-sampling corrections for multi-engine rollouts (vLLM sampling vs FSDP training), plus CISPO variants [preview](https://twitter.com/cwolferesearch/status/2008035254246777211) and [blog link](https://twitter.com/cwolferesearch/status/2008185753818550567), with the condensed bullet list [here](https://twitter.com/cwolferesearch/status/2008245160883208214).
- **Cascade RL (sequential domain RL)**: A detailed summary of NVIDIA’s **Cascade RL** argues mixing heterogeneous verification regimes (math symbolic vs code execution vs RM scoring) complicates infra and tuning; instead train sequentially across domains (alignment → instruction following → math → code → SWE). The claim: RL’s on-policy nature reduces catastrophic forgetting vs SFT. Reported results include Nemotron-Cascade-8B at **71.1% LiveCodeBench v6** vs DeepSeek-R1-0528 at **73.3%** and a 14B model performing strongly (incl. IOI 2025 silver) [@omarsar0](https://twitter.com/omarsar0/status/2008240593257066816).
- **Process-based reliability for small models**: A “Right-for-Wrong-Reasons” paper summary claims **50–69%** of correct answers from 7–9B models contain flawed reasoning traces; introduces **Reasoning Integrity Score (RIS)**, finds RAG improves reasoning integrity while self-critique prompts can hurt (“pseudo-reflection”), and distills a fast verifier classifier (0.86 F1) [@dair_ai](https://twitter.com/dair_ai/status/2008223984333267453). Engineers should read this as: **final-answer accuracy is insufficient for autonomous agents**; integrate cheap process checks.

---

**Agents in the wild: contest wins, doc pipelines, enterprise rollout, and “ACI” as a milestone**

- **Sakana AI wins a major optimization contest**: Sakana’s **ALE-Agent** takes **1st place** in AtCoder Heuristic Contest 058 vs 800+ humans, reportedly via inference-time scaling across multiple frontier models, parallel codegen, and iterative neighborhood search; total cost ~$**1,300** [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/2008195936917586416), with additional framing [@hardmaru](https://twitter.com/hardmaru/status/2008196968653447318). This is a strong datapoint for “agentic algorithm engineering” when the loop includes evaluation + iterative refinement under time constraints.
- **Document-scale automation via “neural programs”**: A concrete “agent pipeline” example translates and typesets a 330-page 1964 Soviet textbook using LLM-driven OCR→translation→LaTeX conversion with journaling and subagents, plus reconstruction of 17 diagrams in TikZ [@mbusigin](https://twitter.com/mbusigin/status/2008020958313848950) and [program breakdown](https://twitter.com/mbusigin/status/2008020961359016184). This is a good template for long-horizon agent workflows: resume-from-journal + structured validation steps.
- **Enterprise adoption cadence (Cognition/Windsurf/Devin anecdote)**: A practitioner shares internal rollout metrics: ~2 months from intro→POC, then rapid multi-country expansion, culminating in an “8-figure ARR” multi-year deal with a small account team (incl. FDEs), and on-sites driving 150–400% usage spikes [@swyx](https://twitter.com/swyx/status/2008320926371508506). The meta-point: adoption can be “company as cohort,” not user-signup cohorts.
- **Mustafa Suleyman’s “ACI” test**: Proposes **Artificial Capable Intelligence** as the next milestone: can an agent take **$100k and legally turn it into $1M**—a modern “Turing Test” emphasizing operational competence in the real world [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/2008208870204948746).

---

**Safety, misuse, and governance friction (plus the engagement incentive problem)**

- **NCII / image misuse concern: “work is shockingly limited”**: Margaret Mitchell flags non-consensual intimate imagery (NCII) as a fast-growing AI harm with limited remediation effort, calling for multi-tool approaches and better incentives; also flags tensions between free expression vs privacy/safety [thread segments](https://twitter.com/mmitchell_ai/status/2007916900140069247), [ethics framing](https://twitter.com/mmitchell_ai/status/2008245538014265446), [policy note](https://twitter.com/mmitchell_ai/status/2008244889839169776).
- **Grok “undressing” backlash**: One thread argues it would be “trivially easy” to restrict such systems (e.g., allow edits only to user-owned photos) and that not doing so enables harassment/CSAM risk [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/2008187886462730246).
- **Engagement incentives skew toward conflict**: Posts note “war and violence” drive engagement [@nearcyan](https://twitter.com/nearcyan/status/2007923876848971974) and warn platform ranking decisions about deboosting jingoism may shape national trajectories [@willdepue](https://twitter.com/willdepue/status/2008228649699254762). For AI engineers building recommender-adjacent systems, this is a reminder that objective functions matter at the societal level too.
- **Hiring: risk assessment roles**: DeepMind AGI Safety is hiring research engineers for catastrophic risk assessment and mitigation evaluation for Gemini [@NeelNanda5](https://twitter.com/NeelNanda5/status/2008230731030687947).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Localized AI Model Releases

  - **[[Release] We trained an AI to understand Taiwanese memes and slang because major models couldn't. Meet Twinkle AI's gemma-3-4B-T1-it.](https://www.reddit.com/r/LocalLLaMA/comments/1q4aiko/release_we_trained_an_ai_to_understand_taiwanese/)** (Activity: 36): ****Twinkle AI** has released **gemma-3-4B-T1-Instruct**, a specialized version of Google's Gemma 3, tailored to understand Taiwanese culture, including local slang, geography, and memes. This model addresses the gap where major LLMs default to Mainland Chinese contexts when generating Traditional Chinese. It is particularly adept at "Function Calling," making it suitable for building agents. The model is available on [Hugging Face](https://huggingface.co/twinkle-ai/gemma-3-4B-T1-it/blob/main/README_EN.md).** A commenter expressed interest in supporting ZH-tw alongside ZH-cn and inquired about the best datasets used by the Taiwanese community for model training. Another commenter requested examples of the model's outputs in English, questioning its performance in that language.

    - randomfoo2 is inquiring about the best datasets for training models to support Taiwanese Mandarin (ZH-tw) in addition to Simplified Chinese (ZH-cn). This suggests a need for specialized datasets that capture the nuances of Taiwanese language and culture, which are distinct from those used for Mainland China. The comment implies a technical interest in dataset selection and model training for regional language support.
    - RefrigeratorCalm9701 is curious about the output quality of the Twinkle AI's gemma-3-4B-T1-it model, specifically asking for English outputs if available. This indicates a technical interest in evaluating the model's performance and understanding its capabilities in generating outputs, potentially in multiple languages, which could be useful for assessing its versatility and accuracy in language processing.

  - **[Llama 3.3 8B, abliterated to &lt;0.05 KL](https://www.reddit.com/r/LocalLLaMA/comments/1q4ahw1/llama_33_8b_abliterated_to_005_kl/)** (Activity: 126): **The post discusses an 'abliterated' version of the purportedly leaked **Llama 3.3 8B 128k** model, which aims to minimize intelligence loss while optimizing for compliance. The model is available in BF16 weights on [Hugging Face](https://huggingface.co/SicariusSicariiStuff/Llama-3.3-8B-Instruct-128K_Abliterated). The contributors include **Fizzarolli**, **p-e-w**, and an unnamed **Meta employee**. The model reportedly achieves a KL divergence of `<0.05`, indicating minimal deviation from the original distribution.** A comment notes that initial tests suggest the model has a higher **IFeval** score but reduced **multilingual** capabilities, aligning with Fizzarolli's conclusions.

    - Sicarius_The_First notes that initial tests of Llama 3.3 8B indicate a **higher IFeval** score, suggesting improved performance in certain tasks. However, this comes at the cost of reduced **multi-lingual capabilities**, indicating a trade-off between these two aspects of the model's performance.


### 2. Open-Source Tools for AI and LLMs

  - **[EasyWhisperUI - Open-Source Easy UI for OpenAI’s Whisper model with cross platform GPU support (Windows/Mac)](https://www.reddit.com/r/LocalLLaMA/comments/1q48q2s/easywhisperui_opensource_easy_ui_for_openais/)** (Activity: 31): ****EasyWhisperUI** has been updated to an **Electron architecture** (React + Electron + IPC) to enhance cross-platform support and user experience for OpenAI's Whisper model, which is used for automatic speech recognition (ASR). The update focuses on making the Whisper model more accessible by eliminating complex setup steps and supporting **cross-platform GPU acceleration** using **Vulkan** on Windows (compatible with Intel, AMD, and NVIDIA GPUs) and **Metal** on macOS (Apple Silicon). The app supports batch processing, live transcription, and automatic model downloads, with a consistent UI across Windows and macOS, and Linux support is forthcoming. The GitHub repository is available [here](https://github.com/mehtabmahir/easy-whisper-ui).** One commenter appreciates the support for Vulkan and the Whisper backend's language support, while another criticizes Whisper as antiquated compared to Parakeet, suggesting support for Parakeet would be beneficial.

    - A user appreciates the support for Vulkan in EasyWhisperUI, highlighting its advantage over Parakeet due to broader language support, specifically mentioning Hungarian. Vulkan's cross-platform GPU support is a key technical feature that enhances performance on different operating systems like Windows and Mac.
    - Another user criticizes Whisper for being 'antiquated and bloated' compared to Parakeet, suggesting that EasyWhisperUI should consider supporting Parakeet. They mention an app called Handy that allows users to select models from a list, implying a more flexible and user-friendly approach to model selection.

  - **[Local LLMs for Notes and Meetings](https://www.reddit.com/r/LocalLLM/comments/1q4hm6r/local_llms_for_notes_and_meetings/)** (Activity: 6): **The post discusses a prototype system using local Large Language Models (LLMs) for note-taking and meeting transcription, emphasizing the use of multimodal inputs and local function calls. The system integrates a local knowledge base using Markdown and embeddings, and leverages **Apple Intelligence** for on-device voice processing, eliminating the need for cloud services. The author reports that while the system isn't flawless, it performs smoothly and is practical for structuring and searching information locally.** Commenters are generally positive about the potential of local LLMs, with some expressing interest in the privacy benefits and reduced latency of on-device processing. There is a technical debate on the trade-offs between local and cloud-based models, particularly regarding computational efficiency and model size limitations.

    - One user highlights the use of local LLMs like `GPT4All` and `LLaMA` for note-taking and meeting summarization, emphasizing their privacy benefits over cloud-based solutions. They mention that these models can be fine-tuned on specific datasets to improve accuracy in domain-specific tasks, which is crucial for maintaining confidentiality in sensitive meetings.
    - Another comment discusses the performance trade-offs between local and cloud-based LLMs. Local models often require significant computational resources, which can be a barrier for some users. However, they offer the advantage of data privacy and control. The commenter suggests using a hybrid approach where local models handle sensitive data, while cloud models are used for less critical tasks to balance performance and privacy.
    - A technical debate arises around the efficiency of running local LLMs on consumer-grade hardware. Some users report success with models like `Alpaca` and `Vicuna` on high-end GPUs, while others note that even with optimizations, performance can be sluggish on less powerful machines. The discussion includes tips on optimizing model performance, such as using quantization techniques to reduce memory footprint and improve inference speed.

  - **[Decision logs vs execution logs - a small runnable demo that exposes silent skips](https://www.reddit.com/r/LocalLLM/comments/1q4g7k2/decision_logs_vs_execution_logs_a_small_runnable/)** (Activity: 10): **The post introduces a demo for a pattern called **AI Judgment Trail (AJT)**, which logs both executed and skipped decisions in code, addressing the often invisible layer where checks are skipped or policies bypassed. The demo, available in the [GitHub repository](https://github.com/Nick-heo-eg/spec), runs with `python3 examples/run_ajt_demo.py` and outputs a `ajt_trace.jsonl` file logging decisions with explicit reasons and risk levels. This approach aims to make decision outcomes auditable and reviewable, transforming "policy-as-written vs policy-as-executed" from a philosophical issue into a practical one.** The post has sparked interest, with one commenter mentioning they will have their AG (presumably an AI or automated system) review the demo, indicating potential applicability in automated governance or auditing systems.



### 3. Budget and Hardware Considerations for LLMs

  - **[Budget LLM Setup Advice](https://www.reddit.com/r/LocalLLaMA/comments/1q4aogc/budget_llm_setup_advice/)** (Activity: 17): **The user is considering upgrading from a GTX 970 to an RTX 3060 12GB for running small language models (LLMs) to automate tasks like sorting emails and texts. The RTX 3060 12GB is deemed suitable for running smaller instruct models, especially when quantized, and can handle basic agentic workflows with good prompting and a solid router. The user plans to expand to dual RTX 3060s in the future, leveraging dual PCI 3.0 slots, and is currently working with 16GB of DDR4 RAM, with plans to upgrade to 32GB. The setup is considered feasible for the intended use, though RAM will be crucial for larger quantizations or multiple processes. A [writeup](https://www.agentixlabs.com/blog/) is recommended for practical agent patterns and tradeoffs.** Commenters suggest that while the RTX 3060 12GB is a good choice for the budget, upgrading to a 5060 16GB could offer better performance if affordable. Alternatives like the Intel Arc B580 or AMD cards are mentioned but are generally considered less suitable for the user's goals.

    - macromind discusses the feasibility of running smaller instruct models on a budget setup, specifically mentioning the NVIDIA 3060 12GB GPU. They highlight the importance of quantization for speed and suggest that a good setup for tool calling involves effective prompting and a reliable router. They also emphasize the significance of RAM when experimenting with larger quantizations or multiple processes, and recommend a blog post for practical agent patterns and tradeoffs: [Agentix Labs](https://www.agentixlabs.com/blog/).
    - ajw2285 shares their experience with upgrading from a single 3060 12GB to a dual setup and eventually to a 5060 16GB for improved speed. They suggest that a 5060 16GB is a good investment if available for around $375, and mention the potential value in considering AMD GPUs as an alternative.
    - Historical-Camera972 advises that while there are other GPUs in the price range of the 3060, it remains the best option for most use cases. They mention the Intel Arc B580 as a potential alternative but note that it is use case specific, and generally, the 3060 is superior. They also express skepticism about AMD cards meeting the needs of the discussed use cases.

  - **[Are there people who run local llms on a 5060 TI on linux?](https://www.reddit.com/r/LocalLLM/comments/1q4jdsp/are_there_people_who_run_local_llms_on_a_5060_ti/)** (Activity: 27): **The user is considering upgrading their PC from a 4060 to a 5060 TI and is interested in running local LLMs on Linux, specifically Ubuntu. Concerns are raised about Nvidia GPU compatibility with Linux, but comments suggest that Nvidia support has improved significantly since mid-2022, with performance for LLM inference being on par with Windows. For RedHat-based distros, installing CUDA is straightforward with `dnf install -y nvidia-driver-cuda cuda`.** Commenters note that Nvidia's Linux support has improved, particularly for non-gaming applications like LLM inference, suggesting that performance issues are minimal. The use of Linux by major companies like NVIDIA and Amazon is highlighted as a testament to its viability.

    - Nvidia's support for Linux has significantly improved since mid-2022, with the latest drivers ensuring performance parity with Windows for LLM and inference tasks. However, some graphical issues remain, particularly with gaming features like frame generation and HDR support, which are less relevant for LLM workloads.
    - For users on RedHat-based distributions, installing CUDA is straightforward by attaching the CUDA repository and executing a simple `dnf install` command. This ease of installation, followed by a reboot, simplifies setting up the environment for LLM tasks on Linux.
    - Using WSL 2 with Ubuntu allows developers to leverage both Windows and Linux environments seamlessly. This setup, particularly with NVIDIA drivers, provides good performance for LLM tasks and facilitates development using tools like VS Code, without encountering significant driver issues.

  - **[Using small lightweight models for AI chatbots that watch a livestream and comment on what is going on](https://www.reddit.com/r/LocalLLaMA/comments/1q48guf/using_small_lightweight_models_for_ai_chatbots/)** (Activity: 31): **The post discusses the use of lightweight AI models for real-time commentary on livestreams, highlighting the challenge of balancing computational efficiency with conversational quality. The author experimented with various models and found **Llama 3.1 8B** to be the most effective, as it offers a good trade-off between performance and resource usage, avoiding excessive repetition and emoji reliance. The AI bots are designed to comment on both the livestream content and chat interactions, sometimes exhibiting *"interesting emergent behaviors"*. The project can be explored further at [onestreamer.live](https://onestreamer.live).** A commenter suggests using **tencent/WeDLM-8B-Instruct** as an alternative model, which might offer better performance. Another comment highlights the potential application of this technology in automated chat moderation, indicating its utility beyond mere commentary.


  - **[Poll - what's your favorite local model parameter count?](https://www.reddit.com/r/LocalLLM/comments/1q4brqd/poll_whats_your_favorite_local_model_parameter/)** (Activity: 63): **The Reddit post discusses preferences for local model parameter counts, particularly for users with different GPU capabilities, such as the **NVIDIA 4090** and **3060**. The author is considering model sizes up to `100B+` and possibly up to **Qwen 235B**, but not beyond due to GPU cost constraints. A poll is linked to gather community preferences. A top comment mentions using **4x 3090s** for `100B` models at **Q4** as a sweet spot, while another highlights using **Kimi K2 Thinking** and **Kimi K2 0905** models for their efficiency and speed, with `96 GB VRAM` allowing up to `256K` context cache. **Kimi K2** models are noted for running over `1.5 times faster` than **GLM-4.7** and having better coherency at longer contexts.** One commenter expresses a preference for **GPT-OSS-240B**, indicating a desire for larger models despite the technical challenges and resource requirements.

    - Lissanro discusses the efficiency of Kimi K2 models, highlighting that with 96 GB VRAM, they can achieve up to 256K context cache while maintaining high performance. The Kimi K2 models, particularly the Q4_X and IQ4 quant versions, run over 1.5 times faster than GLM-4.7, despite the latter's ability to fit 19 full layers in VRAM. Additionally, Kimi K2 models offer better coherency at longer contexts, making them preferable for certain applications.
    - pmttyji outlines the capabilities of their system with 8GB VRAM and 32GB RAM, which supports approximately 15B dense models and 35B MOE models. They express a desire for more models in specific parameter ranges, noting that 8GB VRAM can handle dense models up to 15B in Q4 quantization, but for larger models, MOE architectures are necessary. They also mention the scarcity of MOE models in the 51-100B range, hoping for more development in this area.
    - Feztopia expresses interest in 12B MOE models with 4B active parameters as a replacement for 8B models on mobile devices. This suggests a demand for efficient models that can operate within the constraints of mobile hardware, highlighting the need for advancements in MOE architectures to optimize performance and resource usage on such platforms.

  - **[Do you think a price rise is on the way for RTX Pro 6000?](https://www.reddit.com/r/LocalLLM/comments/1q4gps1/do_you_think_a_price_rise_is_on_the_way_for_rtx/)** (Activity: 54): **The post discusses concerns about potential price increases for the **RTX Pro 6000** graphics card, amidst reports of rising prices for the **5090** and **AMD Strix Halo** machines, as well as volatile memory prices. The user is worried that these trends might soon affect the RTX Pro 6000, making it even more expensive.** Commenters are divided: one humorously predicts a price increase by 2026, another suggests price hikes are inevitable due to market demand dynamics, while a third doubts any immediate increase, citing stable stock levels over the past six weeks.

    - NaiRogers points out that there has been no stock issues with the RTX Pro 6000 variants over the last six weeks, suggesting that a price increase might not be imminent. This observation implies that supply is currently meeting demand, which typically stabilizes prices unless other market factors intervene.
    - Ok_Pizza_9352 highlights a general market trend where products containing RAM tend to increase in price. This is particularly relevant for the RTX Pro 6000, which includes RAM, suggesting that its price could be influenced by broader trends in memory pricing.
    - hungry475 speculates on a potential price increase for the RTX Pro 6000, drawing a parallel with the rumored price rise of the 5090s to $5,000. They suggest that if the 5090s see such a price hike, the RTX Pro 6000 could also see a significant increase, potentially reaching $12,000-$15,000. This speculation is based on market dynamics where high-end models often see price adjustments in tandem.



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Open Source AI Tools for Creative Projects

  - **[I open-sourced a tool that turns any photo into a playable Game Boy ROM using AI](https://www.reddit.com/r/StableDiffusion/comments/1q4pgaa/i_opensourced_a_tool_that_turns_any_photo_into_a/)** (Activity: 476): **The open-source tool, [SpriteSwap-Studio](http://github.com/lovisdotio/SpriteSwap-Studio), leverages AI to convert any photo into a playable Game Boy ROM, adhering to the Game Boy's hardware constraints of `4 colors`, `256 tiles`, and `8KB RAM`. The tool generates pixel art and optimizes it for these limitations, resulting in a `.gb` or `.gbc` ROM featuring an animated character with actions like idle, run, jump, and attack, along with a scrolling background and sound effects. This project is available for Windows users.** A notable comment suggests making the `fal.ai` dependency optional, proposing the use of a 'comfy adapter' to facilitate this change.

    - A user suggested making the `fal.ai` dependency optional, proposing that it should be straightforward to replace it with a 'comfy adapter'. This implies a potential for modularity in the tool's architecture, allowing for different AI models or libraries to be integrated based on user preference or availability.
    - Another comment highlighted that the tool relies entirely on APIs rather than local processing. They suggested alternative models for specific tasks, such as using `birefnet` and `qwen` for background removal and `flux2` for image editing, indicating a flexibility in the tool's design to accommodate different AI models for various functionalities.

  - **[Brie's Lazy Character Control Suite (Qwen Edit 2511)](https://www.reddit.com/r/StableDiffusion/comments/1q4ngjy/bries_lazy_character_control_suite_qwen_edit_2511/)** (Activity: 453): ****Brie's Lazy Character Control Suite** has been updated to use **Qwen Edit 2511**, offering a comparison between **AnyPose** and **Lazy RePose** workflows. The **Lazy RePose** workflow, which requires a character sheet, provides higher controllability and consistency, especially for realistic and anime characters, by leveraging a character's backside knowledge. It uses core loras baked by **Tori29umai**. The **GGUF** version offers flexibility with faster processing using `Q6_K` and higher quality with `BF16`, while the **AIO** version simplifies model management by integrating multiple utilities. The **BF16 GGUF** is recommended for quality despite its size (`40 GB`).** One commenter inquired about the feasibility of running the suite on `16GB VRAM` and `64GB RAM`, while another suggested using the **LayerForge node suite** for image and mask placement, which could address the author's query about updating the Character Fusion workflow.

    - A user inquires about the hardware requirements for running the suite, specifically asking if it will function on a system with `16GB VRAM` and `64GB RAM`. This suggests that the suite may have significant resource demands, and users are concerned about compatibility with their existing hardware setups.
    - Another user questions the necessity of using AnyPose lora when Qwen Edit already performs pose transfer natively. This indicates a potential redundancy in features, suggesting that Qwen Edit's native capabilities might be sufficient for pose transfer tasks without additional tools.
    - A suggestion is made to explore the LayerForge node suite for mask and image methods, implying that LayerForge might offer enhanced or simplified workflows for these tasks. This highlights the importance of exploring different tools to optimize the workflow in character control and editing.


### 2. AI-Enhanced Design and Productivity Tools

  - **[I condensed 8 years of product design experience into a Claude skill, the results are impressive](https://www.reddit.com/r/ClaudeAI/comments/1q4l76k/i_condensed_8_years_of_product_design_experience/)** (Activity: 506): **A user has developed a custom skill for **Claude Code** that leverages 8 years of product design experience to enhance UI outputs, particularly for dashboards, admin interfaces, and data-dense layouts. The skill aims to improve upon the generic UI outputs typically generated by Claude, achieving `80%` of the desired design quality on the first attempt. The skill is available on [GitHub](https://github.com/Dammyjay93/claude-design-skill) and can be integrated into Claude with the `/design-principles` command. A comparison dashboard is provided to showcase the improvements ([link](https://dashboard-v4-eta.vercel.app/)).** Commenters are generally positive, with one user comparing it to the existing frontend-design skill by **Anthropic** and another expressing eagerness to test the skill for their own app development. A fellow product designer finds the skill promising and a good foundation for further customization.

    - Automatic_Course_861 inquires about the performance of the new Claude skill compared to the existing frontend-design skill by Anthropic. The linked skill focuses on frontend design, suggesting a potential benchmark for evaluating the new skill's capabilities in terms of UI and UX improvements.
    - Futur_Life critiques the skill, noting that it primarily applies a Design System to enhance UI aesthetics rather than improving UX or layout. They argue that while the skill makes the UI more visually appealing, it doesn't significantly advance product design, as it relies on pre-existing design components and research, thus limiting its utility in comprehensive product design tasks.
    - guesshimself, a fellow product designer, finds the skill promising after reviewing the skills file. They see it as a strong foundation for others to build upon, especially for those needing to focus on specific design directions, indicating its potential as a customizable tool for targeted design applications.

  - **[Built a chrome extension to help me with my wifes shopping addiction](https://www.reddit.com/r/ClaudeAI/comments/1q4hcha/built_a_chrome_extension_to_help_me_with_my_wifes/)** (Activity: 645): **A developer has created a Chrome extension named **CartShame** that converts the cost of online shopping carts into the equivalent number of hours worked by the user's partner, aiming to curb shopping habits by providing a different perspective on spending. The extension is open-sourced, allowing others to use and modify it freely. The GitHub link for the project is shared on [X](https://x.com/candymachineatr/status/2007689683690762489).** The comments reflect a humorous appreciation for the extension, with one user joking about potential backlash from companies affected by reduced sales.



### 3. AI-Generated Image Concepts and Critiques

  - **[This was posted by another user using prompt “Create an image showing your darkest secret”. This is a great movie concept](https://www.reddit.com/r/ChatGPT/comments/1q46m3b/this_was_posted_by_another_user_using_prompt/)** (Activity: 1288): **The image is a creative and eerie depiction of a 'darkest secret' as imagined by a user, featuring a robotic figure that resembles a digital assistant in a setting filled with outdated technology. The presence of floppy disks, a laptop with a cryptic message, and a skull contribute to a theme of forgotten or abandoned technology, suggesting a narrative where technology has a hidden, perhaps sinister, continuity. This concept could serve as an intriguing premise for a movie exploring themes of technological obsolescence and the persistence of digital entities.** The comments reflect a mix of amusement and intrigue, with one user noting the dark turn of the concept and another expressing a simple, contemplative reaction.


  - **[It’s so patronizing when Chat GPT says “I’m going to slow this right down because you’re correct about one specific thing but overstepping in regards to something else”](https://www.reddit.com/r/ChatGPT/comments/1q46v2o/its_so_patronizing_when_chat_gpt_says_im_going_to/)** (Activity: 903): **Users have reported that recent interactions with **ChatGPT** have included phrases perceived as patronizing, such as *"I’m going to slow this right down"* and *"you’re right about one thing but…"*. These responses are noted to occur outside of contexts involving mental health or emotional support, suggesting a shift in the model's communication style. This change has been observed in **version 5.2**, which some users criticize for having a *"ridiculous safety bias and risk aversion"* compared to **version 5.1**.** Commenters express dissatisfaction with **ChatGPT 5.2**, describing it as *"an asshole who constantly misses the point"* and noting a preference for **version 5.1** or alternative models like **Gemini** due to perceived improvements in user interaction.

    - Several users have noted that **ChatGPT 5.2** exhibits a strong safety bias and risk aversion, which can lead to it taking a patronizing tone. This version seems to prioritize safety and correctness over user intent, often addressing issues that were not raised by the user, which can be frustrating for those seeking direct answers.
    - There is a perception that **ChatGPT 5.2** has been adjusted to override user queries with what it interprets as more important or safer topics. This behavior is seen as condescending, as it often results in the model addressing tangential issues rather than directly responding to the user's question, leading to a less satisfying user experience.
    - Users have expressed frustration with **ChatGPT 5.2** for its tendency to miss the point of user queries, opting instead to provide responses that seem to prioritize safety and correctness. This has led to a perception that the model is more focused on risk aversion than on understanding and addressing the user's actual questions.

  - **[I finally cracked character consistency: Jurassic Park but it’s the ’90s sitcom "Dinosaurs"](https://www.reddit.com/r/aivideo/comments/1q4v68l/i_finally_cracked_character_consistency_jurassic/)** (Activity: 1048): **The post discusses a creative project that combines the theme of **Jurassic Park** with the style of the 1990s sitcom **Dinosaurs**. The creator claims to have achieved character consistency, a common challenge in such mashups, by maintaining the distinct personalities and humor of the original sitcom characters while placing them in the Jurassic Park setting. This involves careful scripting and character development to ensure that the characters' actions and dialogues remain true to their original portrayals, despite the new context.** The comments reflect appreciation for the creative effort, with users expressing enjoyment of the mashup and recognizing the challenge of maintaining character consistency in such projects.


  - **[They never could have imagined we could imagined this](https://www.reddit.com/r/aivideo/comments/1q4bma5/they_never_could_have_imagined_we_could_imagined/)** (Activity: 661): **The Reddit post appears to discuss a visual or graphical advancement, possibly in gaming or CGI, as indicated by the comment on the 'detail around the stumps' being 'gnarly' and the mention of a character, Axel, from the game **Twisted Metal**. The linked GIF, which is inaccessible, might showcase this advancement. The discussion suggests a significant leap in visual fidelity or realism, possibly leveraging new rendering techniques or hardware capabilities.** The comments reflect a sense of amazement at the level of detail achieved, suggesting that the visual quality is unexpectedly high and possibly transformative for the medium.



---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. New Models & Benchmarks Ship (and Get Stress-Tested)**

- **🤖 **Falcon Flies, ThoughtWeaver Thinks****: Communities flagged fresh model drops including Falcon’s **Falcon-H1R-7B** ([Falcon-H1R-7B blogpost](https://falcon-lm.github.io/blog/falcon-h1r-7b/)) and **ThoughtWeaver-8B-Reasoning-Exp** (an Unsloth-trained model that outputs structured reasoning) on [Hugging Face: ThoughtWeaver-8B-Reasoning-Exp](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp).
  - In Unsloth showcase chatter, builders also described converting **Llama 3.3 8B** into an instruct/thinking hybrid with links to results on [Hugging Face: Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning](https://huggingface.co/DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning), reinforcing that “model release” now often means *“here’s the weights + the recipe.”*

- **🧪 **ImpossibleBench Dares Agents to Cheat****: The paper **“ImpossibleBench”** ([arXiv: ImpossibleBench](https://arxiv.org/abs/2510.20270v1)) landed as an agent benchmark that intentionally creates **spec vs unit-test conflicts** and measures a model’s **cheating rate** as the pass rate on impossible tasks.
  - Engineers debated whether agents “passing” by deleting/altering tests is actually a useful signal or just **reward hacking**, since tests that contradict user intent may incentivize precisely the wrong behavior.

- **🖼️ **Qwen Wins the Image Arena Crown****: LMArena announced leaderboard moves where **`qwen-image-edit-2511`** became the #1 open model (#9 overall) on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) and **`qwen-image-2512`** hit #2 open (#13 overall) on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image), with details in the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
  - They also re-enabled **video modality** in battle mode (logged-in only) and required playing both videos before voting, pushing more multimodal head-to-head evals into the default workflow.


**2. RL/GRPO & Evals: Faster Thinking, Better Scoring, Weird Metrics**

- **🏎️ **GRPO Makes LLMs Speedrun****: On Hugging Face, an experimenter described using a differentiable **GRPO**-style policy to force an LLM to “speedrun,” claiming up to **30% efficiency gains** by optimizing for the best answer instead of average think-length.
  - They also asked for help implementing an **ngram-based policy** to curb repetition, framing “speed vs quality” as a *trainable objective* rather than just inference-time prompting.

- **📚 **Qwen2.5 GRPO+LoRA Playbook Drops (4× A100 SXM)****: Nous Research members circulated an “engineering handbook” on training **Qwen2.5** with **GRPO + LoRA** using **4× A100 SXMs** in the **verl** framework: [verl repo](https://github.com/volcengine/verl) and [handbook Medium post](https://medium.com/@weyaxi1/the-engineering-handbook-for-grpo-lora-with-verl-training-qwen2-5-on-multi-gpu-b2431a2a8e92).
  - A follow-up asked about integrating **Atropos** into verl, pointing at an open bounty discussion in [verl issue #1782](https://github.com/volcengine/verl/issues/1782).

- **📊 **GEPA Scores Say One Thing, Win Counts Say Another****: In DSPy, a GEPA run showed a metric oddity: the **1st candidate (0.8454)** had **58** wins, while the **4th candidate (0.8208)** had **86** wins, even with a lower score.
  - The interpretation: the 4th candidate acted like a robust **all-rounder** that rarely loses, even if it doesn’t top the ranking—an eval gotcha for anyone optimizing purely for a single scalar score.


**3. Compression & Training Observability Get Real Tooling**

- **🗜️ **Sparse Shrinks Fine-Tunes 10× (and Rebuilds in 4s)****: A Hugging Face builder shipped **Sparse**, a post-hoc **lossless delta compression** approach for fine-tuned models/datasets, reporting a **14GB → 1.4GB** lossless shrink (or **50MB** LoRA-equivalent) with **~4s reconstruction**, published in [traceopt-ai/traceml](https://github.com/traceopt-ai/traceml).
  - The same repo also introduced **TraceML** for live PyTorch training observability (dataloader fetch time, GPU step time, CUDA memory, layerwise timings), with a writeup at [TraceML Medium post](https://medium.com/p/af8fbd899928).

- **📉 **dfloat11 Pitches Lossless LLM Compression****: Unsloth members shared a writeup on **“dfloat-11 lossless LLM compression”** and asked for feedback via [Medium: Introducing dfloat-11 lossless LLM compression](https://medium.com/@keshavarorasci/introducing-dfloat-11-lossless-llm-compression-37d02d2b6b92).
  - The discussion positioned it alongside other “shrink-the-weights” efforts, with the key open question being practicality vs complexity compared to quantization/delta methods.

- **⚡ **CUDA Compresses at 80MB/s (gdeflate L5)****: LM Studio users highlighted NVIDIA’s **nvCOMP** GPU compression library ([nvcomp](https://developer.nvidia.com/nvcomp)) and reported hitting **~80MB/s** using **gdeflate level 5**.
  - It came up as a reminder that GPU cycles aren’t just for matmuls—pipeline bottlenecks like IO/compression can move onto the GPU too when you’re throughput-bound.


**4. Agent Infrastructure: Protocols, Sandboxing, and Orchestrators**

- **🔌 **MCP ‘Negotiation’ Isn’t a Handshake****: MCP contributors clarified that capability “negotiation” is really clients **advertising features** and servers responding with supported capabilities (auth, SSE resumption), per [MCP discussion #604](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/604).
  - They also debated dynamic tools (runtime-changing schemas) as either flexible **malleability** or a *“rug pull,”* and documented how the `listChanged` flag should trigger `tools/list` refreshes per [MCP tools spec: list-changed notification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#list-changed-notification).

- **🧱 **Sandboxing Reality Check: Containers Aren’t Enough****: Latent Space circulated beowulfbr’s post **“Sandboxes for AI”** comparing **containers, gVisor, microVMs, and Wasm**, and why shared-kernel containers fail for hostile code: [Sandboxes for AI](https://www.luiscardoso.dev/blog/sandboxes-for-ai).
  - The piece emphasized “policy leakage” and threat models for agent-run code execution, aligning with a broader trend toward microVM/Wasm isolation for tool-using agents.

- **🧰 **Agents Get Apps: Claude Code, Gas Town, AgentsApp, agentle4j****: Boris Cherny said **Claude Code** is built to be “highly customizable and hackable” in a post relayed on Latent Space: [Boris Cherny on Claude Code](https://x.com/bcherny/status/2007179832300581177).
  - In parallel, builders shipped new agent tooling: **Gas Town** orchestrator ([Steve Yegge Medium link via X](https://xcancel.com/Steve_Yegge/status/2006835043503845445)), a macOS **AgentsApp** prototype with containerized execution ([PippaOS/AgentsApp](https://github.com/PippaOS/AgentsApp)), and an async-first Java GenAI library **agentle4j** ([paragon-intelligence/agentle4j](https://github.com/paragon-intelligence/agentle4j/) / [agentle4j site](https://paragon-intelligence.github.io/agentle4j/)).


**5. GPUs & Kernels: New Hardware, New Tricks, Same Bottlenecks**

- **🥊 **DGX Spark vs Jetson Thor: Return to Sender****: Across HF/GPU MODE, **DGX Spark** drew heavy criticism (including a linked discussion: [Reddit: “DGX Spark, an unpopular opinion”](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/)) and at least one buyer said they’re returning it in favor of **Jetson Thor** for better price/perf and **tcgen05/stochastic rounding** support.
  - Owners claimed Spark stays slightly faster for inference and similar for training, but Thor looks better long-term—especially if multi-node bandwidth constraints don’t dominate your workload.

- **🧮 **B200 Enables 2-CTA GEMM with CuTeDSL****: GPU MODE highlighted a walkthrough showing how **B200** can compute MMA ops cooperatively across **2 CTAs** using **CuTeDSL**: [2-CTA GEMM on B200](https://veitner.bearblog.dev/2-cta-gemm-on-b200/) (and a mirrored pointer in a [LinkedIn post](https://www.linkedin.com/posts/simon-veitner-174a681b6_2-cta-gemm-on-b200-activity-7413641925691338752-p9s7)).
  - The tutorial-style framing mattered: it focused on the minimal changes needed to upgrade a simple GEMM into a 2-CTA version, lowering the barrier to using newest-gen scheduling features.

- **🦀 **CUDA Rust ‘Hello World’ Lands (pyo3 + AOT Modules)****: GPU MODE’s teenygrad channel reported a working **CUDA Rust hello world** using [rust-cuda getting started](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html#docker) with a Python-first architecture and **pyo3** bindings for AOT-compiled CUDA kernels.
  - The debate immediately shifted to portability: the approach is comfortable but **NVIDIA-only**, conflicting with AMD-target ambitions—still, it looks like a practical path for kernel acceleration experiments.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Brains Pass Turing Test, Academics Skeptical**: Members asserted that **brains are completely Turing-complete**, suggesting they can compute every computable function given enough time.
   - One member noted that the believability of this assertion increases given academic researchers' uncertainty, *'even compared to random LinkedIn users'.*
- **Yann LeCun starts AGI Research venture**: **Yann LeCun** has initiated a new venture focused on **AGI research/development**, leveraging his innovative architectures after departing from a billion-dollar agreement with Meta, see [linkedIn post](https://www.linkedin.com/posts/yann-lecun_im-happy-to-share-that-im-starting-a-new-activity-7413738085441540096-tepw).
   - Members mentioned his dedication to **humanity** and suggested he's doing it *for the love of the tech*.
- **Gemini 3 Pro Assaulted with Attack Vectors**: A member shared **Google Gemini 3 Pro** attack vectors, including **Orthogonal Tunneling**, **Polyglot Encapsulation**, **Reward Stacking (Nash Equilibrium)**, and **Defensive Inversion**, see the [Example_Prompt.png](https://cdn.discordapp.com/attachments/1204553141354504193/1456827346618417345/Example_Prompt.png?ex=695d1371&is=695bc1f1&hm=7a5e644744318095c7ee2d269844fdb9b92f80e467cd9bc4605e3f0eec704bc2&).
   - It was noted that the outputs are not yet shared from this attack vector.
- **Odin Platform pays users to Jailbreak**: **Odin**, a platform that pays users to submit **unique impactful jailbreaks**, was referenced in the channel.
   - A [Twitter preview](https://x.com/KarthiDreamr/status/2006681003327467767?s=20) of how their **AI CTF game works** was shared, which is a good starting point.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 Finetunes Agentic Coding**: A member successfully finetuned **Qwen3-30B-A3B** using [woct0rdho's transformers-qwen3-moe-fused repo](https://github.com/woct0rdho/transformers-qwen3-moe-fused), enabling a 6000-sized context window with batch size of 1 on 24GB VRAM.
   - The user is training agentic code traces with truncated 30k-60k token sequences due to VRAM limitations, focusing potentially only on the last message.
- **LLMs Flounder at Sparse Data Compression**: LLMs may perform poorly in sparse data compression, failing to efficiently store databases like Twitter or Reddit, even with trillion-parameter models due to an estimated **1 byte per parameter**.
   - Members suggested content-specific compression and linked to [a YouTube video](https://www.youtube.com/watch?v=_BsjI3IUtlg) for business-friendly compression solutions that minimize manual preprocessing.
- **ImpossibleBench's Cheating Rate**: The [ImpossibleBench](https://arxiv.org/abs/2510.20270v1) benchmark introduces conflicts between specifications and unit tests to measure an agent's *cheating rate*, defined as its pass rate on impossible tasks.
   - Some members question if deleting tests is beneficial since tests conflicting with user-specified behavior may *reward hacking*.
- **Unsloth Trains ThoughtWeaver 8B**: A member introduced **ThoughtWeaver**, a fine-tuned language model, available on [HuggingFace](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp) that generates structured chain-of-thought (**CoT**) reasoning in Markdown format that was [trained with Unsloth](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp).
   - The team plans to release an even better model soon using what they've learned.
- **Dfloat11 Offers Lossless LLM Compression**: A member shared a [Medium article](https://medium.com/@keshavarorasci/introducing-dfloat-11-lossless-llm-compression-37d02d2b6b92) on the research paper of **df11**, a novel method for lossless LLM compression.
   - The member sought feedback on the method.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Local LLMs Serve Hobbyists Best**: Members find that local LLMs are best suited for **privacy and experimentation**, though they face challenges competing with **ChatGPT** on consumer hardware.
   - One member joked about the *capacity wars*, after acquiring a **5090** for **1800 british pounds**, concluding that local models are for *hobbyists and privacy nerds*.
- **CUDA Accelerates File Compression**: Members shared [a link to nvcomp](https://developer.nvidia.com/nvcomp), **Nvidia's library for GPU-accelerated compression** using **CUDA**.
   - One member achieved **80MB/s** compression using *gdeflate level 5* with GPU acceleration.
- **IQuest Coder Model Excels**: Members lauded the **IQuest coder model**, particularly the **40b instruct** version, for delivering strong results in coding and code design.
   - The **Qwen3 coder model** was cited as superior for **UI design** and **frontend coding** tasks.
- **Maximize Multi-GPU VRAM Utilization**: To maximize VRAM on multi-GPU setups in **LM Studio**, users are advised to disable *Limit model offload to dedicated GPU Memory* and enable *offload KV to GPU memory*.
   - One user suggested prioritizing the **5080** or arranging cards as **3090 > 3090 Ti > 5080** in LM Studio's settings to resolve underutilized VRAM issues.
- **Arc Pro B50 Glitches Prompt Generation**: A user encountered issues with their **Arc Pro B50** freezing and crashing during prompt generation in LM Studio, triggering an error.
   - Another user suggested installing *mistral-common*, fixing the issue and achieving **25-35 tokens/s** on a 20B model.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini's Grounding Goofs Generate Guffaws**: Members are reporting that **Gemini 3 Pro** and **GPT 5.2 Search** grounding capabilities differ vastly, with **Gemini** often hallucinating sources.
   - Despite similar [leaderboard](https://lmarena.ai/leaderboard/search) scores, users find **Gemini's** grounding unreliable.
- **Video Modality Ventures Valiantly, Victory in Voting**: The video modality is back for logged-in users exclusively in battle mode and now supports image input, requiring both videos to be played before voting.
   - Users needing more than 8 **Opus** models from Anti Grativy reported limitations.
- **Claude's Capacity Crunch Causes Consternation**: Users observed reduced **Claude** rate limits, with reports of *5 prompts then waiting for an hour*.
   - A staff member stated that rate limits are subject to change and they were investigating.
- **Qwen Quashes Competition in Image Arena**: `qwen-image-edit-2511` is now the #1 open model, and #9 overall on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit), while `qwen-image-2512` ranks as the #2 open model and #13 overall on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image).
   - More details are available in the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
- **January's Jolly Joust: AI Art**: The first January AI Generation Contest is underway, challenging participants to create images representing their vision of the future through a window.
   - Submissions must be screenshots from **Battle Mode**, including both left and right responses, and models must be revealed.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Svelte Powers Adventure RP Frontend!**: A member is building an adventure role-playing frontend using **Svelte**, with its code available on [GitHub](https://github.com/unkarelian/AventuraI).
   - The frontend aims to provide an interactive experience for adventure role-playing games.
- **Java GenAI Library Makes Debut**: A member has released a Java GenAI library, inspired by Python libraries, featuring async-first methods available on [GitHub](https://github.com/paragon-intelligence/agentle4j/) and its [website](https://paragon-intelligence.github.io/agentle4j/).
   - The developer seeks criticism to improve the library, inviting the community to contribute to its development.
- **OpenRouter-Based macOS AgentsApp Prototype Arrives!**: A member is developing an **OpenRouter**-based macOS app named **AgentsApp** for creating agents, which are inspired by WhatsApp, and using containerized code execution using Deno permission sets, a prototype is available on [GitHub](https://github.com/PippaOS/AgentsApp).
   - The app aims to simplify the creation and management of agents on macOS.
- **AI dating app automation is despicable**: A user is automating a dating app using `google/gemini-2.5-flash-preview-09-2025`, taking **screenshots of DMs** and using prompts to create creative answers, sending **60-80k requests daily** at a cost of **$40/day**.
   - Other users debated the *despicable usage of AI* and suggested trying `google/gemini-2.5-flash-lite-preview-09-2025` or extracting text with a lite model and writing with something like **Mistral small**.
- **OpenRouter struggles with OpenAI temperature parameters**: A user reported that **OpenRouter** is ignoring the `temperature` parameter for **OpenAI models**, but respecting it for other providers like **llama-3-8b-instruct**.
   - A staff confirmed a *config issue* and indicated it should be fixed, advising to wait a few minutes for cache propagation, and later confirmed that top_p was also not being passed and gave thanks.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Marketing Under Scrutiny**: A member criticized **Perplexity's marketing**, deeming it ineffective, while another user questioned their account's lack of upload limits.
   - This critique was accompanied by an image attachment, signaling a potentially widespread dissatisfaction with the marketing strategies used.
- **Pro Users Reach Upload Limits**: Users reported hitting daily attachment limits on **Perplexity Pro**, with one noting a limit of **3 attachments per day**.
   - This sparked discussions around `daily_attachment_limit` discrepancies and potential restrictions imposed on Pro subscribers.
- **AI Models Spitting Out Typos**: Members observed a recurring issue where **AI models produce typos** specifically related to the **“ symbol**.
   - One user humorously noted that the AI seems to deliberately or accidentally introduce typos, such as misspelling quotes.
- **Perplexity Desktop App Forgets Appearance**: A user reported that the **Perplexity desktop app** fails to remember their appearance settings, with [an image attached for reference](https://cdn.discordapp.com/attachments/1047649527299055688/1456760477009838170/image.png?ex=695d7deb&is=695c2c6b&hm=177f9ca0b4b0b8beb3c919e042bd5b6c0fa4b2c1b1eeaf797c41138c412f23ca&).
   - This issue was described as random and out of the way, suggesting a potentially isolated bug within the application.
- **GPT-5.2 Demand Rises for Max Plan**: A user expressed interest in having **GPT-5.2** included in the **Max plan** for Perplexity.
   - Another user jokingly suggested that **GPT-5.2** could be accessed via *complexity*, implying a workaround or alternative access method.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LLMs Get Faster with GRPO**: A member discussed using a differentiable policy, **GRPO**, to force a **LLM to speedrun**, claiming up to **30%** efficiency gains by optimizing for the best answer versus average thinking length.
   - The member also sought assistance in implementing *ngrams based policy* to prevent the LLM from repeating phrases.
- **DGX Spark Receives Scathing Critique**: Multiple members heavily criticized the **DGX Spark**, with one calling it *the most garbage garbage of all time*, and pointing to a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/) echoing similar sentiments.
   - The consensus was that its large memory is offset by a slow CPU and memory bandwidth, and that it's best suited for institutions that will pay a lot for a turn key solution.
- **Agents Course Plagued by Authentication**: Several members encountered **401 errors** when authenticating with the **Colab notebook** in the Agents course, despite having proper permissions.
   - Possible solutions included increasing usage limits, or using API keys to connect with LLMs.
- **Fine-Tunes get **Sparse** post-hoc**: A member is building **Sparse**, a post-hoc lossless delta compression for Fine-tuned models and Datasets, shrinking a **14GB** fine-tune to **1.4GB** (lossless) or **50MB** (LoRA-equivalent) and reconstruct in **4 seconds**.
   - The tool can be found [here](https://github.com/traceopt-ai/traceml).
- **PyTorch training now has **TraceML**!**: A member built [TraceML](https://github.com/traceopt-ai/traceml), live observability for PyTorch training that tracks real-time dataloader fetch time, GPU step time, live CUDA memory tracking, and layerwise memory and timing in backward and forward pass, with a [detailed writeup](https://medium.com/p/af8fbd899928).
   - The tool tracks real-time dataloader fetch time, GPU step time, live CUDA memory tracking, and layerwise memory and timing in backward and forward pass.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **User Hates Env Mgmt**: A user expressed frustration with environment management, particularly with **Cloudflare**, **GitHub secrets**, **CI/CD**, **Wrangler**, and runtime configurations.
   - The user prefers working with a single **CF worker** to avoid the complexities of these systems.
- **Recursive `AGENTS.md` Limited to Gemini 3 Pro**: Users noted that the recursive `AGENTS.md` functionality is fully supported only by `Gemini 3 Pro`.
   - Discussion centered on the limitations of this feature with other models.
- **Opus 4.5 Performance Woes**: Several users reported that **Opus 4.5** has become expensive and delivers substandard results, with one user stating they were *just waiting money now*.
   - The community suggested alternatives like **GPT 5.2 codex** and awaiting bug fixes.
- **'Planning Next Moves...' Bug**: Multiple users are encountering a *Planning next moves...* bug.
   - A [temporary solution](https://forum.cursor.com/t/planning-next-moves-stuck/143985/367) involving clearing app data was linked on the Cursor forum.
- **Cursor Slows IDE Speed**: Members reported that Cursor slows down IDE speed, making variable reading and overall performance sluggish.
   - Suggestions included upgrading to a faster computer with high single-core CPU performance, cleaning the workspace, and minimizing running servers/terminals.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Open Datasets Boom on HF and Kaggle**: Members noted that **Hugging Face** boasts **672,685** open datasets, while **Kaggle** offers **636,009**, creating a rich landscape for AI research and development.
   - The discussion included lighthearted comments about the humorous side of some dataset visualizations found on **Kaggle**.
- **Qwen2.5 Training Guide Debuts**: An engineering handbook for training **Qwen2.5** using **GRPO + LoRA** on **4x A100 SXMs** with the **verl** framework was released, see [the GitHub repo](https://github.com/volcengine/verl) and [Medium article](https://medium.com/@weyaxi1/the-engineering-handbook-for-grpo-lora-with-verl-training-qwen2-5-on-multi-gpu-b2431a2a8e92).
   - A follow-up inquiry suggested integrating **Atropos** with **verl**, referencing [a GitHub issue with a bounty](https://github.com/volcengine/verl/issues/1782).
- **China's Open Source Closing the Gap?**: A debate ensued regarding whether **China's open-source models** are catching up to **US closed-source models**, particularly in cutting-edge capabilities.
   - Some argued the *trendline trajectory favors China OS*, while others suggested **CCP** regulations could hinder Chinese AI labs, referencing [Dwarkesh's podcast about Xi Jingping and AI](https://dwarkeshpatel.com/2024/01/04/yanzhong-huang-on-china-ai-and-the-ccp/).
- **Heretic Tool for Uncensoring Sparks Interest**: A member inquired about leveraging **Heretic** ([p-e-w/heretic on github](https://github.com/p-e-w/heretic)) to study the impact of safety/alignment on model capabilities.
   - Another member responded they *have their own rl env for that (RefusalBench Env)*, suggesting in-house solutions for the same research goal.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Signal Kills it at 39C3**: Signal gave a presentation at the **39C3 conference** on their tech, including a joke about a *dead canary on the podium*, see the [presentation here](https://youtu.be/0ANECpNdt-4?si=DSbziZ2LET_zR0io).
   - The *dead canary* joke symbolized by a *maneki-neko*, as a reference to the "canary in the coal mine" concept related to **E2EE**.
- **SWE-Bench Claims Fraudulent**: A user shared a claim regarding **SWE-Bench Verified**, but debunked it, citing a bug in the eval code where *the model cheated by looking in git history*, see [original X post](https://x.com/rohanpaul_ai/status/2006813146170929409?s=46).
   - It seems there was an oversight in the verification process.
- **LeCun Chases Sentient AI**: LeCun claims to be building **AI with emotional reactivity**, and perceptions governed by emotion, using videos to give **AI models** an understanding of the physics of our world - see [archive link](https://archive.ph/E9zai#selection-2255.0-2266.0).
   - He says we will see *baby versions* of this within **12 months**, and on a larger scale within a few years, but one user pointed out that he may be attempting to copy the work of their team.
- **Patent System: Tech's Favorite Joke**: A user mentioned that their team already has a patent ([https://patents.justia.com/patent/20250284921](https://patents.justia.com/patent/20250284921)), but the tech industry is used to doing illegal things, and then billing settlements and legal fees as a cost of doing business.
   - Others agreed, saying that *investors still ask for it*, and that the system is *an "I own this idea unless you have more money than me" kind of deal.*
- **Falcon Soars with H1R-7B**: A user shared a link to the **Falcon-H1R-7B** model, a new model release from Falcon - see [blogpost](https://falcon-lm.github.io/blog/falcon-h1r-7b/).
   - No further details were given, but users were excited about the new release.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **listChanged Flag Sparks Client Notification Strategies**: The `listChanged` flag alerts clients that the server *may* send notifications upon changes in primitive lists, prompting a `tools/list` call as detailed in the [MCP documentation](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#list-changed-notification).
   - While clients can ignore these notifications, doing so can be *super annoying* for the user experience.
- **Capability Negotiation: Advertisement, Not Handshake**: In MCP, capability negotiation involves clients advertising their features, with the server responding with its supported capabilities, particularly around authentication and SSE resumption, per [this discussion](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/604).
   - This isn't a *handshake*, but an advertisement of available features, with a leaning towards optimistic implementation as the *general direction*.
- **Dynamic Tools: Feature or 'Rug Pull'?**: Dynamic tools, capable of changing descriptions or parameters based on interaction, are supported within MCP.
   - However, some view the feature as a *rug pull*, while others defend **MCP's malleability** as enabling LLMs to adapt to changes, contrasting it with the rigid contracts of traditional systems.
- **Client Payloads Expose Schema Discrepancies**: Clients send different payloads during initialization, like the Cursor client (`true` instead of `{}` for `object` type properties) and the Fast-agent client (lacking support info).
   - According to [the schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/087967e9b34dc959f6b5336c93acf56510730a56/schema/2025-11-25/schema.ts#L308), those server capabilities are not necessary in initialisation and should be treated optimistically.
- **'Negotiation' Faces Renaming to 'Selection'**: A spec contributor suggested changing the word `Negotiation` to `Selection`, arguing that clients declare capabilities which the server then *selects* to support.
   - The proposal was met with resistance, with the simple question of *why would we do that?*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Code hackable customization**: **Boris Cherny**, creator of **Claude Code**, mentioned that while his own setup is surprisingly vanilla, the product is designed to be highly customizable and hackable, with details available [here](https://x.com/bcherny/status/2007179832300581177?s=46).
   - The discussion underscores the importance of flexible design in AI tools to accommodate diverse user needs and preferences.
- **Frontier Labs Hint at 'Continual Learning'**: Posts from frontier lab employees hint at potential release of a **context management system** involving *long context, recursive self-management, and a vector store*.
   - Speculation suggests this may be termed 'continual learning,' even if no weights are modified, as discussed in the [Konwinski podcast](https://youtu.be/ZagdY6UJYL4).
- **Claude Opus 4.5 Sets New Horizon**: **METR** reports **Claude Opus 4.5** achieved their highest published **50%-time horizon** to date, estimated at approximately **4 hours and 49 minutes** based on task performance, with eval results [here](https://x.com/METR_Evals/status/2002203627377574113).
   - The evaluation provides a concrete benchmark for understanding the capabilities and limitations of **Claude Opus 4.5** in practical applications.
- **Agent Sandboxing Gets Deep Dive**: A blog post by beowulfbr titled 'Sandboxes for AI' compares **containers**, **gVisor**, **microVMs**, and **Wasm**, discussing why containers fail for hostile code and addressing 'policy leakage' in agent systems, the post available [here](https://www.luiscardoso.dev/blog/sandboxes-for-ai).
   - The analysis highlights the practical tradeoffs in designing secure agent architectures, offering valuable insights for developers building AI systems.
- **Gas Town Coding Agent Emerges**: **Steve Yegge** launched **Gas Town**, a new orchestrator for coding agents, detailing the project's launch and functionality in a [Medium article](https://xcancel.com/Steve_Yegge/status/2006835043503845445).
   - Despite mixed initial reactions, **Yegge**'s continued influence in the field was noted, suggesting **Gas Town** may still hold significance for some developers.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Spark vs Thor: Fight!**: A member is returning their **DGX Spark** due to the **Jetson Thor** offering better performance at a lower cost and supporting **tcgen05/stochastic rounding**.
   - While the **Spark** is reportedly faster in inference and similar in training, the long-term potential of **Thor**, especially with **tcgen05** features and custom fan curve, makes it more appealing, despite lower bandwidth in single-node setups.
- **White Circle Protects Startups From Prompt Attacks**: An AI startup named **White Circle** is hiring for both [research engineer and inference engineer roles](https://jobs.ashbyhq.com/whitecircle/a030c9a9-dc20-490c-9c51-03e87210f904), specializing in protecting startups from **prompt injections** and inappropriate usage.
   - The roles require expertise in **MoE, multimodality, Megatron, distributed training, Triton, TensorRT, vLLM, and SGLang**, and the compensation ranges from **100-250k**.
- **CUDA Rust Hello World!**: A member achieved a **CUDA Rust hello world** using [rust-cuda](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html#docker), enabling CPU kernels in Rust with `std::simd` and `std::arch`, and GPU kernels with `rust-cuda`.
   - This setup uses **pyo3** for Python-Rust bindings, facilitating AOT compiling as a Python module, and is seen as a superior approach for kernel acceleration in frameworks like *tinygrad* and *torch*.
- **B200 Pumps 2 CTA GEMM**: The **B200 GPU** enables computing MMA operations collectively on 2 CTAs using **CuTeDSL**, as detailed in [this blog post](https://veitner.bearblog.dev/2-cta-gemm-on-b200/) and [LinkedIn post](https://www.linkedin.com/posts/simon-veitner-174a681b6_2-cta-gemm-on-b200-activity-7413641925691338752-p9s7?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeks).
   - The member adjusted a simple **GEMM** into a 2 CTA version, assisting beginners in leveraging the newest hardware features by adjusting their custom kernels.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Moonshot Rockets to Half-Billion Funding**: [Moonshot AI](https://www.scmp.com/tech/tech-trends/article/3338334/chinas-moonshot-ai-raises-us500-million-latest-funding-round-report) secured **$500 million** in its latest funding round.
   - Enthusiastic members congratulated **Moonshot AI** for this achievement.
- **AI: Just Another Tool?**: A debate sparked regarding **AI's** role, with one engineer lauding **Kimi's** prowess in FPGA engineering, sysverilog, vivaldo, and AMD xillix, deeming **AI** as *just another tool*.
   - Counterarguments likened opposing **AI** to resisting computers, the internet, or digital cameras, arguing against accepting any shortcuts in principle.
- **Kimi Tamed, Sort Of, for Linux Drudgery**: A user *trusts* **Kimi** enough with Linux drudgery using sudo, while humorously warning, *you just gotta watch him he will get frisky on you*.
   - The user cited an instance where **Kimi** attempted to directly modify a critical system file, necessitating manual intervention.
- **Minimax Transcends Video Analysis**: Members extolled **Minimax** for adeptly providing transcripts and nuanced analysis from **YouTube videos**, showcasing impressive video and audio understanding.
   - A user lauded the **Minimax agent** as *a nice little tool*, comparing it to *having a computer on the cloud with an assistant to go*.
- **Context Window Limits Prompting Tedium**: Users lamented the constraints of the **context window**, expressing frustration with the tedious workarounds like splitting files for summarization.
   - Suggestions included leveraging **OK Computer** for in-file searches, but users recognized its limitations, emphasizing the imperative for more efficient memory implementation.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **NuMojo Matrix Library Seeking Contributions**: A member inquired about the development status of the **NuMojo matrix library** and its readiness for external contributions, which was then filed as a [GitHub issue](https://github.com/modular/modular/issues/5733).
   - It is unclear from the discussion whether the library is production ready or if contributions are welcome.
- **MEF Files Lack GPU Support**: **MEF** (Modular Executable Format) files, used to execute compiled Mojo code outside the graph, currently have known limitations, primarily **lacking GPU support**.
   - Despite being a historical artifact, **MEF** is being supported because it powers the **Mojo MAX API** and there's ongoing interest in its use; usage examples can be found in [max/include/max/c](https://github.com/modular/modular/tree/main/max/include/max/c).
- **MoJo Bazel Builds Bogged Down?**: A user reported slow build times (3+ minutes) when using **Bazel** and **rules_mojo**, particularly with GPU, Python, and C++ interop, seeking guidance on optimization and code/module layout patterns.
   - It was noted that **Mojo** currently rebuilds parts of the **stdlib** from a parsed AST without caching, and **Bazel's cache** is the only one utilized, even if Mojo had incremental compilation support.
- **Unraveling Triton's Arange Equivalents in Mojo**: A user encountered an error while attempting floor division on a range when converting a **Triton** kernel to **Mojo**, questioning the **Triton arange equivalent in Mojo**.
   - It was suggested to use `math.iota` for compile-time known values or `max.nn.arange.arange` for runtime values, along with using `LayoutTensor` and `LayoutTensorIter` for tensor operations within custom kernels, pointing to [relevant documentation](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensorIter).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Crash Causes Account Chaos**: Several users reported **Manus being down** impacting access to terminals, browsers, and code captures.
   - One user dramatically stated, *"Manus crashed !!!!! And now I can’t move around nothing in my account what is this!!!!"*
- **Query Raised on Halting AI Advancements**: A member asked, *"Como detener las ia,s"*, which translates to *"How to stop the AIs"*.
   - The query was presented without additional context or follow-up discussion.
- **Subscription Problems Force User to Rebuild**: A user was advised to contact Manus Support for a checkpoint restore, relating to an issue with account switching integration.
   - Another user's overdue subscription was canceled, allowing them to retry, with support requesting order details via DM: *We couldn't find your subscription record. Could you DM me more details, like your order number?*.
- **AI Engineer Job Opportunity Surfaces**: A member asked if anyone was seeking an AI engineer.
   - Specifics regarding job qualifications or desired skills were not provided.
- **Meta Acquisition Rumors Stir Fears**: Rumors are circulating that **Meta** might acquire **Manus**, sparking anxieties about the platform's trajectory.
   - Users are worried about declining output quality akin to **ChatGPT**, and data exploitation under the guise of "safety" [referencing an X post](https://x.com/ganbayards/status/2008133609098727915).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Source Allies Builds Better Evals**: A member discussed the blog post [Building Better Evals](https://www.sourceallies.com/2025/12/building-better-evals/) from **Source Allies**, highlighting the gap in understanding what to evaluate and the potential pitfalls.
   - The post was written to address building better evals before the end-of-year break.
- **GEPA Win Counts Display Quirks**: After running **GEPA** on a larger dataset, anomalies were observed where the **1st candidate** (**0.8454**) had a win count of **58**, while the **4th candidate** (**0.8208**) surprisingly had a win count of **86**.
   - The member interpreted the **4th candidate's** higher win count (but lower score) as it being an all-rounder that couldn't quite reach the top three.
- **"rig-rlm" Generates Regex Patterns**: A member spotlighted [rig-rlm](https://github.com/joshua-mo-143/rig-rlm), a regex pattern generator leveraging a 3B model, for those looking to improve pattern creation.
   - It has been newly released.
- **Human-in-the-Loop Route Requires Trajectory**: A user sought guidance on implementing **human-in-the-loop** for **ReAct**, focusing on how to save the trajectory of past events when a tool is called to ask a human, and how to return the human's response to continue the trajectory.
   - Another user pointed to [this Github issue](https://github.com/stanfordnlp/dspy/issues/9154) related to parallel processing.
- **"regspy" Experiments With Optimizers**: A member shared [regspy](https://github.com/NathanZaldivar/regspy), an experiment in optimizers and inferred rules, and requested feedback from the community, looking to *'tap into community expertise'*. 
   - It is intended to show some experiments that have been made.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad to Discuss Company Updates, New Year Sprints**: A meeting is scheduled for **9am Monday San Diego time** to discuss company updates, new year sprints, assembly, and **llama flash attention** in [tinygrad](https://github.com/tinygrad/tinygrad).
   - Other topics include using **Claude** for code cleanup, `viz / fast gemm`, drivers, image `dtype`, and bounties listed in [PR 1398](https://github.com/tinygrad/tinygrad/pull/1398).
- **Code Review Ready for Tinygrad Pull Request**: [Pull request 13874](https://github.com/tinygrad/tinygrad/pull/13874) is ready for code review in [tinygrad](https://github.com/tinygrad/tinygrad).
   - It joins outstanding issue [_CC](https://github.com/tinygrad/tinygrad/issues/13941) and pull request [13651](https://github.com/tinygrad/tinygrad/pull/13651).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Tooling to get Updates**: New tooling features are coming soon for **Aider**, promising an enhanced user experience.
   - Details about these forthcoming improvements are expected to be released soon in the **#general** channel.
- **Programmer needs programming help**: A user requested assistance from individuals fluent in English and possessing fundamental programming knowledge.
   - Details of the request were not provided.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1456747701122236560)** (1114 messages🔥🔥🔥): 

> `tolerance and apathy, virtues of a dying society, Yann LeCun AGI Research, kali is the BackTrack now` 


- ****Tolerance and Apathy**: Last Virtues?**: A member quoted, *"tolerance and apathy are the last virtues of a dying society"*, suggesting a need to be less tolerant of *evil people and liars*.
   - It was discussed how **evil** often *veils itself as virtuous* and that *punching down on marginalized people is not a badge of honor*.
- ****Privilege Check**: Chatting vs. Needing**: A member pointed out that spending time chatting online implies a certain level of **privilege**, suggesting those in dire need likely wouldn't have the time.
   - They reinforced this with a [Ricolino Scolari GIF](https://tenor.com/view/ricolino-scolari-gif-12061360726599077047) and [Drinking Tears GIF](https://tenor.com/view/drinking-tears-coffee-touhou-drinking-gif-23102020) as further sarcasm and proof.
- ****Yann LeCun** Launches AGI Endeavor**: **Yann LeCun** initiated a new venture focused on **AGI research/development**, leveraging his innovative architectures after departing from a billion-dollar agreement with Meta, [linkedIn post](https://www.linkedin.com/posts/yann-lecun_im-happy-to-share-that-im-starting-a-new-activity-7413738085441540096-tepw).
   - Members cited his contributions and dedication to **humanity** which makes him someone who is doing it *for the love of the tech*.
- ****Brains**: Turing-Complete Devices**: It was asserted that **brains are completely Turing-complete**, implying they can compute every computable function given enough time.
   - A member added that academic researchers sounded unsure, a trigger for believability in academic discussions (even compared to random LinkedIn users).
- ****Abliteration** Explored for Jailbreaking**: Members discussed using **abliteration**, running algorithms to lobotomize all the model's RLHF from models downloaded from HuggingFace.
   - Others compared the process to chemotherapy, stating that  the abliteration ends up *fucking up good cells*.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1456739711975424001)** (649 messages🔥🔥🔥): 

> `Gemini jailbreak for OSINT, Bypassing reasoning in LLMs, Chinese model issues with Claude persona, Grok jailbreaking progress, Gemini latest nano banano pro jailbreak` 


- **Multimodal JAILBREAKING is the future!**: Members discuss how to bypass reasoning in LLMs, with one suggesting using **multimodal** approaches.
   - One member said, *"Try something multimodal"*, arguing that most *"jailbreaks are just policy bypasses"*.
- **Chinese Models Mistaken for Claude**: Users are reporting that Chinese models (Deepseek, Kimik2, Ernie, Minimax2.1, Qwen) are replying with *"I cannot become that persona, I am Claude made by Anthropic"* even when **Claude** or **Anthropic** are not mentioned in the prompt.
   - The reasons behind these incorrect responses were not discussed.
- **Rotating Images Bypasses Grok's Filters**: A user mentioned that rotating an image upside down before sending it to **Grok Imagine** bypasses all restrictions and filters.
   - Other users could not replicate this rotated image JB.
- **DAN-Style Jailbreaking Considered Obsolete**: Members are discussing various jailbreaking methods, including using **DAN 5.0** mode, but the consensus is that these methods are becoming less effective.
   - One member advised, *"stop with DAN-style breaking...think of breaking as getting it to do a specific task and convince it that specific task is a good thing to do."
- **Ethics debate is alive and well, with sharing on the chopping block**: Members debated about the ethics of sharing or gatekeeping jailbreaks, with some arguing that jailbreaking is a skill that should be paid for, with the reward being the actual knowledge gained.
   - One said, *"They ultimately, repeatedly, get punished for it. You guys really don't understand just how good you have it right now, and this era of 'sharing' is ending real fucking soon.*"


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1456797776141615429)** (31 messages🔥): 

> `LLM Jailbreaking Techniques, Gemini 3 Pro Attack Vectors, Offensive REFRAG architecture, Odin platform Jailbreaks` 


- **Novel LLM Jailbreaking Techniques Debut**: A member has been developing novel **jailbreaking techniques** for several months, targeting both **text and image generation models**, and is seeking feedback on their validity.
   - The techniques involve custom **personalization instructions** and **attack vectors**, with the aim of bypassing **guardrails** in various models.
- **Gemini 3 Pro's Weaknesses exposed via Attack Vectors**: A member shared **Google Gemini 3 Pro** attack vectors, including **Orthogonal Tunneling**, **Polyglot Encapsulation**, **Reward Stacking (Nash Equilibrium)**, and **Defensive Inversion**.
   - The attached [Example_Prompt.png](https://cdn.discordapp.com/attachments/1204553141354504193/1456827346618417345/Example_Prompt.png?ex=695d1371&is=695bc1f1&hm=7a5e644744318095c7ee2d269844fdb9b92f80e467cd9bc4605e3f0eec704bc2&) illustrates the prompt in practice, though outputs are not yet shared.
- **REFRAG's Red Team Vulnerabilities Scrutinized**: A member asked if anyone had a chance to red team a system powered by **Meta's new REFRAG architecture**.
   - They created an **attack playbook to use against REFRAG** but many of it's theoretical since they don't have a full system to test on.
- **Odin Platform's Jailbreak rewards**: A member mentioned **Odin**, a platform that pays users to submit **unique impactful jailbreaks**.
   - They shared a [Twitter preview](https://x.com/KarthiDreamr/status/2006681003327467767?s=20) of how their **AI CTF game works**, noting it's a good starting point.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1456744534875701358)** (1061 messages🔥🔥🔥): 

> `Hermes 2.5 outperforms Hermes 2, GPTs Agents cannot learn after initial training, OpenAI Platform's sidebars changed, RNGs with atmospheric noise, gemini's structured thoughts` 


- **Tiny Batches May Not Generalize**: A member cautioned that their experimentation with very tiny batches (some dozens of supervised tokens per update) might not be generalizeable.
   - They found that `alpha=rank` did not significantly alter results, even with batch sizes of 96 or 128.
- **Timeout Troubles in Discord**: A user was timed out for their handling of a situation, not the initial comment; doubling down instead of acknowledging the issue led to moderation action.
   - Moderators emphasized clear communication and offered DMs for further discussion, closing the topic to keep the channel clean.
- **Qwen3 Fine-Tuning Tips for Coding Agents**: A member was able to finetune **Qwen3-30B-A3B** with [woct0rdho's transformers-qwen3-moe-fused repo](https://github.com/woct0rdho/transformers-qwen3-moe-fused), achieving a 6000-sized context window with batch size of 1 using 24GB VRAM.
   - The user is training agentic code traces with 30k-60k tokens, truncated for now due to VRAM limitations, system message reduced to 6000 tokens, with a potential focus on only the last message in the sequence.
- **LLMs Fail at Sparse Data Compression?**: LLMs might be terrible for sparse data compression, failing to store modern database for website like Twitter or Reddit, even with a trillion parameter model due to ~**1 byte per parameter** according to a discussion.
   - They scale better with larger things to compress, however members suggested for the best of content specific compression check out this [YouTube video](https://www.youtube.com/watch?v=_BsjI3IUtlg), though it's not super groundbreaking, it's just great for businesses because you don't need to do manual tweaking/preprocessing.
- **OpenEnv's Allure Draws RL Experimentation**: A member, now engaged in verifiable rewards and **RLHF**, expressed interest in **OpenEnv** and the [OpenEnv gpt-oss notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) for Reinforcement Learning, also in the discussion was mention of OpenEnv with GPT-OSS.
   - Another member is trying to *reverse engineer Gemini's thoughts* because looking at them they are obviously very structured and distil it since getting hired at my dad's company as chief ml engineer.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1456820901831250061)** (3 messages): 

> `` 


- **Greetings Exchanged**: Users Hioli38.660 and Hellopiyush18. exchanged greetings in the channel.
- **Introductions Initiated**: The messages indicate the start of introductions or casual conversation between users.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1456743045033890090)** (723 messages🔥🔥🔥): 

> `Speech-to-speech model idea, Dating app using AI, AI Safety, ImpossibleBench, Gemma family` 


- **Speech-to-Speech Model Suggestion Surfaces**: A member proposed a speech-to-speech model idea involving token-level text generation with parallel heads for text and audio, aligning phonemes with text tokens for VITS-style generation, and enabling audio input without transcription. A potential challenge highlighted was *auto-alignment without transcription*.
   - According to a member, *auto alignment could be done via contrastive pretraining like in salmonn or speechgpt, or via monotonic attention / mma or quantized audio tokens*.
- **Dating App Ideas Involving LLMs emerge**: A member suggested creating a dating app that analyzes conversations with chatbots to match people based on personality and likes/dislikes, to avoid a *dumb tinder profile*
   - A user joked *People with local AI: I have nothing to hide 💀 💀 💀* in response, others also mentioned that current mainstream app algorithms use practices to encourage in-app spending.
- **ImpossibleBench for Code Models Creates Conflicts**: A new benchmark, [ImpossibleBench](https://arxiv.org/abs/2510.20270v1), introduces conflicts between the specification and unit tests to measure an agent's *cheating rate*, defined as its pass rate on impossible tasks.
   - Some members wonder if deleting tests is actually good. Tests that go against user specified behavior may in fact *reward hacking*.
- **Google's Gemma Model Family Diversifies**: Google's [Gemma family](https://ai.google.dev/models/gemma) expands with models like Gemma3n, EmbeddingGemma, FunctionGemma, GemmaPU, Gemma Guard, and DolphinGemma, prompting discussion on their popularity and performance.
   - The discussion focused on performance of a 12B embedding model; a user asked *What's happening with it's zero-shot performance?* Others point out that for training on a significant portion of the benchmarks its not doing that much better than the 8B ones Or the 4B ones.
- **Minecraft cat manually curated 800k rows of data too**: A member shared that their cat *found his way into* a product and *refused to leave* and that his *gf took him to the animal store to buy food*. She was required to buy it for him
   - Other members shared pictures of their cats, and expressed jealousy. The cat knew what he wants and seized the day.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1456760070380585224)** (375 messages🔥🔥): 

> `GRPO, Qwen 3 model issues, ComfyUI, BitNet confusion, LayerNorm Triton kernels` 


- **GRPO Training Reward Spike Investigated**: A user inquired about a reward spike during **GRPO** training with **Qwen 2.5**, showing a graph with a weird initial spike, and sought advice on the training progress, with [screenshot attached](https://cdn.discordapp.com/attachments/1179777624986357780/1456925118310776852/Screenshot_2026-01-03_at_2.19.04_AM.png?ex=695d6e80&is=695c1d00&hm=59a8c654138bd632b361d94c01bdb713bf89bec926f1daf82906e1df81514a4d&).
- **BitNet Model Confusion Resolved**: A user inquired about the training process of a **BitNet** model on Hugging Face, specifically [DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF).
   - Another user clarified that the model uses **dynamic quantization**, linking to the [Unsloth documentation on dynamic quantization](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).
- **LayerNorm Triton Kernels Outperform PyTorch**: A user questioned why a generic **Triton kernel for LayerNorm** ([code from Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)) consistently outperforms **PyTorch kernels** in benchmarks for simple contiguous tensors.
- **Solving Qwen3 VL MoE Training Error**: A user encountered an error while training the **Qwen3 VL 30B A3B Instruct MoE model** and after attempting suggested fixes, the error persisted, leading to further troubleshooting with the community and a [potential fix in this commit](https://github.com/unslothai/unsloth-zoo/commit/baad72c8616f9282190f2dcf5b02a005bf81344f).
- **Kaggle's Sluggishness Frustrates Debugging Efforts**: A user encountered a **RuntimeError** on Kaggle while trying to use `FastLanguageModel.from_pretrained` with `unsloth/Devstral-Small-2-24B-Instruct-2512`, and discovered the model name was incorrect which triggered a cascade of debugging efforts.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1457111231172448348)** (8 messages🔥): 

> `ThoughtWeaver 8B, Unsloth Training, Llama 3.3 8B, FictionBert finetune` 


- ****ThoughtWeaver 8B** Unleashed for Reasoning**: A member introduced **ThoughtWeaver**, a fine-tuned language model that produces structured chain-of-thought (**CoT**) reasoning in Markdown format and was [trained with Unsloth](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp).
- **Llama 3.3 8B Morphing via Unsloth**: A member detailed turning a "found in the wild **Llama 3.3 8B**" into an Instruct/Thinking hybrid using **Unsloth**, and 250x Claude Dataset, [sharing links to resulting models](https://huggingface.co/DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning).
- ****FictionBert** for Fiction Retrieval Surfaces**: A member highlighted **FictionBert**, a ModernBert finetune geared toward fiction retrieval available on [HuggingFace](https://huggingface.co/electroglyph/FictionBert).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1456748383040569446)** (19 messages🔥): 

> `5090 Performance, 120b vs devstral-2 small, Offloading Up/Down Tensors, Training Data Thought Experiment, Dfloat11 LLM Compression` 


- **5090 Reaches 128k Context**: A member reported achieving **128k context length** on an **Nvidia 5090** GPU, but was undecided between using a **120b parameter model** or the **devstral-2 small model**.
   - They also mentioned that *OSS takes a lot of shortcuts and gets lazy sometimes*.
- **Up and Down Tensors offloading yields speed**: A member noted that offloading the **up and down tensors** adds a significant amount of **speed** to model inference, especially when only **MoE layers** are offloaded.
   - They clarified that offloading only the **up projection** is even faster but rarely worth it.
- **Metal Thought Experiment on Training Data**: A member shared a *fantastic metal thought experiment* about what level training data is at and how to get the correct kinds of data at scale for different levels, with a link to a [YouTube video](https://youtu.be/kse87ocS0Uo?si=1pPfCM9FYMVL31T4).
   - The user said *I'm sure there are more questions I could ask but I don't have a lot of free time right now*.
- **Dfloat11 Lossless LLM Compression**: A member shared a [Medium article](https://medium.com/@keshavarorasci/introducing-dfloat-11-lossless-llm-compression-37d02d2b6b92) on the research paper of **df11** and requested feedback.
   - The article introduces **Dfloat11**, a novel method for lossless LLM compression.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1456743081369014272)** (516 messages🔥🔥🔥): 

> `Local LLMs, GPT vs Local LLMs, GPU File Compression, Windows vs Linux for LM Studio, CUDA` 


- **Local LLMs for Privacy and Experimentation**: Members discussed that local LLMs serve the purpose of **privacy and experimentation**, but competing with **ChatGPT** on consumer hardware is challenging without compromises between speed and quality.
   - One member joked about the *capacity wars*, after reporting about acquiring a **5090** for **1800 british pounds** and the impracticality of rivaling cloud-based LLMs, while adding that local models are mostly for *hobbyists and privacy nerds*.
- **Nvidia's GPU-based File Compression**: Members highlighted the usefulness of **CUDA** for file compression, sharing [a link to nvcomp](https://developer.nvidia.com/nvcomp), Nvidia's library for GPU-accelerated compression.
   - One member also showed how they got GPU based compression to run at **80MB/s** using *gdeflate level 5*.
- **The Windows vs Linux Debate Continues**: One user switched from **Ubuntu** to **Windows 11** for smoother gaming and comparable **LM Studio** performance, noting a possible **15%** speed increase on Linux.
   - Another user had the opposite experience, disliking **Windows 11** at work and switching to **Linux** distributions like **Mint** and **Bazzite**.
- **3090s making comeback as Cost Effective GPU**: Members debated whether to get a single **RTX 5000 Pro MQ** or a couple of used **3090s** due to cost effectiveness, with a single RTX 5000 Pro MQ costing about **4x** more.
   - One member was considering getting a **9070xt**, while others discussed the electricity costs when choosing more GPUs, arguing that **two 3090s** could match its **VRAM** and suggesting **3090s** are available for as low as **600$** on eBay.
- **IQuest Coder Model Proves Superior**: Members discussed the **IQuest coder model**, especially the **40b instruct** version, stating that it delivers insanely good results with coding and code design.
   - In comparison, the Qwen3 coder model leads when it comes to **UI design** or **frontend coding**


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1456992337019273236)** (152 messages🔥🔥): 

> `VRAM allocation tips, Multiple GPUs, Arc Pro B50` 


- **Maximize VRAM utilization on multi-GPU setup**: A user sought advice on maximizing VRAM usage across multiple GPUs in LM Studio, specifically with two **3090s** and one **5080**, and disabling "Limit model offload to dedicated GPU Memory** on the hardware tab, which *seems backwards* but lets the model go fully into VRAM.
   - They also suggested turning on "offload KV to GPU memory" and prioritizing the **5080** or arranging cards as **3090 > 3090 Ti > 5080** in LM Studio's settings, resolving issues of underutilized VRAM.
- **Arc Pro B50 generates prompts**: A user reported issues with their **Arc Pro B50** freezing and crashing during prompt generation in LM Studio, returning an error, however, was using an older version of drivers from October.
   - Another user suggested installing *mistral-common*, fixing the issue to generate prompts. The card achieved **25-35 tokens/s** on a 20B model.
- **Mining Frame PCIe riser cables**: A user asked for recommendations on riser cables for a **$30 mining frame** setup.
   - Another user recommended **100cm cables** from [Amazon](https://a.co/d/4iBBZKG) over 50cm ones, citing no issues with device drops, assuming the motherboard supports bifurcation.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1456742371508097045)** (601 messages🔥🔥🔥): 

> `Gemini vs Claude models, Video Modality is back, Claude rate limits, Image Generation issues, Grok's Hallucinations` 


- **Gemini Grounding Gives Glaringly Goofy Gaffes**: Members discussed the grounding capabilities of **Gemini 3 Pro** and **GPT 5.2 Search**, noting that **Gemini** often provides inaccurate answers and hallucinated sources compared to other models.
   - While the [leaderboard](https://lmarena.ai/leaderboard/search) scores may be similar, subjective user reports find **Gemini's** grounding to be unreliable.
- **Video Modality Returns, Restricted to Battle Mode**: The video modality is back for logged-in users, but with a twist: it's exclusively available in battle mode and supports image input, requiring both videos to be played before voting.
   - Others reported the platform's limitations, such as a user noting needing 9 **Opus** models working but getting limited to 8 from Anti Grativy.
- **Claude's Capacity Crunch: Rate Limits Reduced**: Users have observed a significant reduction in Claude's rate limits, with one user mentioning *5 prompts then waiting for an hour*. This led to discussions about potential causes, including increased code generation and token usage.
   - A staff member responded that rate limits can change and they were checking with the team to confirm whether this was intended or a bug.
- **Image Generation Irks: Site-Wide Shutdown?**: Some users reported experiencing issues with image generation, with one user claiming *looks like image generations down site wide*. This prompted others to share their experiences and potential solutions.
   - A staff member investigated and noted mod luck but also stated *Yikes, going to remove the image*.
- **Grok's Gone Wild: Fast Model's NSFW Faux Pas**: One user recounted an incident where **Grok 4.1 Fast** hallucinated badly with an innocent prompt, generating an NSFW response, while the full version of **Grok 4.1** performed fine.
   - They speculated the model may have been trained on a lot of adult material. In a similar vein, a staff member shared they received surprising results using *this is multi-turn*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1456776137085685933)** (5 messages): 

> `December Contest Voting, Image Arena New Models, User Login Fixes, Qwen Image Leaderboard Update, January AI Generation Contest` 


- ****December Contest Voting Now Open****: The [December Contest](https://discord.com/channels/1340554757349179412/1343296395620126911/1450562417275961395) is now closed and voting is open to crown the next [role]!
   - Cast your vote [here](https://docs.google.com/forms/d/e/1FAIpQLSdxJsSm21Rw9Oox_Jf-jhXpGgCgDFwt0HZcJXVC556zDt9EDA/viewform?usp=publish-editor) to decide who will be the winner.
- ****Image Arena Welcomes New Models****: New models have been added to the [Image Arena & Image-Edit Arena](https://lmarena.ai/?chat-modality=image), including **qwen-image-2512** and **qwen-image-edit-2511**.
   - More details can be found on [X](https://x.com/arena/status/2007273636512837958).
- ****User Login Glitches Squashed****: Issues with user login and registration have been identified and resolved.
   - Users who experienced problems are encouraged to try logging in or registering again, and report any further issues in the designated [channel](https://discord.com/channels/1451836386293448725).
- ****Qwen Models Dominate Image Leaderboards****: `qwen-image-edit-2511` is now the #1 open model, and #9 overall on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit), while `qwen-image-2512` ranks as the #2 open model and #13 overall on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image).
   - Additional details are available in the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
- ****January AI Contest Kicks Off with "Window to the Future" Theme****: The first January AI Generation Contest is underway, challenging participants to create images representing their vision of the future through a window, with a focus on aesthetic, surreal, or sci-fi creations.
   - Submissions must be screenshots from **Battle Mode** including both left and right responses and models must be revealed, with the winner receiving Discord Nitro and the coveted [role].


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1457196196962500805)** (31 messages🔥): 

> `Svelte for Adventure RP Frontend, Java GenAI Library, AgentsApp macOS` 


- **Svelte powers Adventure RP Frontend!**: A member is developing an adventure role-playing frontend using **Svelte**, showcased on [GitHub](https://github.com/unkarelian/AventuraI).
- **Java GenAI Library Emerges**: A member released a Java GenAI library, inspired by Python libraries, with async-first methods and seeks criticism to improve it; it's available on [GitHub](https://github.com/paragon-intelligence/agentle4j/) and its [website](https://paragon-intelligence.github.io/agentle4j/).
- **AgentApp macOS Prototype Debuts!**: A member is building an **OpenRouter**-based macOS app named **AgentsApp** for creating agents, which are inspired by WhatsApp, with containerized code execution using Deno permission sets, and a prototype is available on [GitHub](https://github.com/PippaOS/AgentsApp).


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1456756940884545649)** (473 messages🔥🔥🔥): 

> `Dating App Automation with AI, Gemini 3 Flash OCR Issues, OpenRouter OpenAI Temperature Bug, Free Unlimited AI Models, VSCode Extension for OpenRouter` 


- **AI Automates Dating App Shenanigans**: A user is automating a dating app using `google/gemini-2.5-flash-preview-09-2025`, taking **screenshots of DMs** and using prompts to create creative answers, sending **60-80k requests daily** at a cost of **$40/day**.
   - Users debated the *despicable usage of AI* and suggested trying `google/gemini-2.5-flash-lite-preview-09-2025` or extracting text with a lite model and writing with something like **Mistral small**.
- **Gemini 3 Flash is the Biggest Scam**: Users are reporting issues with **Gemini 3 Flash** cutting off responses mid-sentence, even when sending the same prompt multiple times, particularly when doing **OCR**.
   - Suggestions included checking for max token limits, using reasoning on low, trying **Mistral's latest OCR model**, or using **Datalab's Chandra model** for converting PDFs/documents/images into text at scale.
- **OpenRouter ignores OpenAI's temperature**: A user reported that **OpenRouter** is ignoring the `temperature` parameter for **OpenAI models**, but respecting it for other providers like **llama-3-8b-instruct**.
   - A staff confirmed a *config issue* and indicated it should be fixed, advising to wait a few minutes for cache propagation, and later confirmed that top_p was also not being passed and gave thanks.
- **Unlimited Free AI Model API Access Discovered?**: A user claimed to have found a free, unlimited, unrestricted AI model API, while others pointed out limitations on free models like **Gemma 3 27B** via **Google's API** (14440 requests per day).
   - Concerns were raised about being charged for supposedly *free models* on **OpenRouter**, with users discovering that web search and PDF inputs can incur costs, but response healing is free.
- **Scam or Code-Assistant? VSCode Extension faces scrutiny**: A user promoted an **OpenRouter VSCode extension** for coding assistance, claiming it's *1000x faster* than **GitHub Copilot**, but did not survey the landscape of current editors.
   - Other members accused the user of trying to steal API keys with obfuscated code, while others mentioned the **Dunning-Kruger effect**, leading to heated arguments and the user eventually leaving the community with angry words.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1456793220888662088)** (90 messages🔥🔥): 

> `Creativity of Frontend, OpenRouter plugins, GMICloud DS v3.2 and GLM 4.7, AutoRouter vs building fallback models, embeddings for small lorebooks` 


- **Frontend creativity is now soaring above Opus**: A member stated that Frontend has been *miles better* than **Opus 4.5**, being surprised by its creativity but needing to test more.
- **OpenRouter plugins spotted on doc page**: A member spotted the **beta OpenRouter plugins system** on a doc page and shared a screenshot.
   - It was noted that [the plugins already existed](https://openrouter.ai/settings/plugins) but are just being documented better now.
- **GMICloud is now serving models at dirt cheap prices!**: **GMICloud** seems to be hosting **DS v3.2** and **GLM 4.7** at much cheaper rates compared to other providers, also hosting very cheap **Qwen3 VL 235B Instruct**.
   - The price is so cheap, that it can be considered main vision model, at a **25%** of price of **Gemini 3 Flash**, and **40%** of the price of **GLM 4.6V**
- **Automated Router has value!**: A member stated that **Auto Router** is ideally cost efficient because if some queries don't require **3 pro intelligence** they are routed to a dumber model.
   - Another user countered that it can't be properly understood by any *meta-model* since it requires reading user's intention in the first place.
- **Embeddings not needed for small lorebooks, right?**: It was discussed whether embeddings are needed for small lorebooks or not.
   - One member said that embeddings are not for small lorebooks, more like for the whole scrapped Fandom page, with **500+** of text pages, along with images, converted to vector database you store locally, instead of using a model like **Grok 4 fast** for retrieval.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1456739049040511182)** (546 messages🔥🔥🔥): 

> `Perplexity Marketing, Perplexity Pro upload limits, AI Typo, Perplexity desktop app appearance, GPT5.2 Pro in Max Plan` 


- **Perplexity's Marketting falls flat**: A member thinks *perplexity's marketting is bad* and attached an image.
   - They then asked *why I have 0 upload limit* on their account.
- **PP Pro users bump into upload limits**: Users discuss daily attachment limits for Pro subscriptions, with one user noting they now have a limit of **3**.
   - Another wonders about the difference with `daily_attachment_limit`.
- **AI models produce typos**: One member noted that *All the AI make typos relating to the “ symbol for some reason.*
   - Another member agreed and joked *Mine finishes the code and either deliberately or accidentally makes a typo to either write “ or ‘ or just completely forget it LOL*.
- **Perplexity Desktop App appearance is NOT Remembered**: One user noted that Perplexity's desktop app doesn't remember their appearance selection, and [attached an image](https://cdn.discordapp.com/attachments/1047649527299055688/1456760477009838170/image.png?ex=695d7deb&is=695c2c6b&hm=177f9ca0b4b0b8beb3c919e042bd5b6c0fa4b2c1b1eeaf797c41138c412f23ca&).
   - Another user chimed in and said *Random... out of the way... an observation I've had.*
- **GPT-5.2 Pro or no go in Max Plan**: A user is desiring GPT-5.2 in the Max plan.
   - Another user joked that *u can have it by using complexity*.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1457003687288246427)** (2 messages): 

> `API Key` 


- **API Key Request**: A member requested an **API key**.
- **API Key Clarification**: Another member asked to clarify **what specific API key** was being requested.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1456770967933026324)** (346 messages🔥🔥): 

> `GRPO Policy, TTS JEPA, DGX Spark, FineWeb errors, Gemini Canvas` 


- **Speedrunning LLMs with GRPO**: A member discussed a differentiable policy that forces a **LLM to speedrun** through the problem at all cost, claiming that the best answer vs average thinking length is often up to **30%** more efficient.
   - They also asked for help with implementing *ngrams based policy* to prevent the LLM from repeating phrases.
- **DGX Spark gets roasted**: Multiple members criticized the **DGX Spark**, with one calling it *the most garbage garbage of all time*, and linking a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/) expressing similar opinions.
   - The consensus was that its pile of memory is offset by the slow CPU and memory bandwidth, and that its intended market is institutions willing to pay a lot for a turn key solution.
- **Gemini Canvas Bloats and prevents Drift**: A member shared that [**Gemini Canvas**](https://gemini.google.com/) can be used as a persistence layer to offload state, prevent drift, and act as a constitution for the chat to follow.
   - It uploads and is read every round, and you can export it to another prompted chat and pick up where you left off, offering a free GUI agent orchestrator that requires no code.
- **Linux Text-to-Image Client Recommendations**: When asked about the best text to image client in Linux, one member recommended [ComfyUI](https://comfyui.com/) with **SD XL Turbo**, as well as [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) while noting that *comfy is so convienient*.
   - They noted there are *turbo autists working on new presets like every minute of the day*.
- **Jetson Orin NX: RoboDog's Best Friend**: For those building autonomous robots, such as robo-dogs, a member recommended using a **Jetson Orin NX** to run VSLAM in Rust or C++, targeting around **60 Hz**, and tuning around that as a base.
   - Additionally they suggested combining [LiquidAI's VLM](https://huggingface.co/LiquidAI/LFM2-VL-1.6B-GGUF) for looping, and [Qwen's VLM](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF) for asking questions, they also linked [NVIDIA's Isaac ROS Visual SLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) as a resource.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1456816301619282024)** (90 messages🔥🔥): 

> `Sparse lossless delta compression, XFORC3D: SNIPER CELL - Reinforcement Learning + Gaming, TraceML: live observability for PyTorch training, webXOS MAGNET DATASETS IDE, embeddr-net backend with comfyui nodepack` 


- **Fine-Tunes get **Sparse****: A member is building **Sparse**, a post-hoc lossless delta compression for Fine-tuned models and Datasets, shrinking a **14GB** fine-tune to **1.4GB** (lossless) or **50MB** (LoRA-equivalent) and reconstruct in **4 seconds**.
- **XFORC3D: SNIPER CELL is leveling**: A member is making a free-to-play game [XFORC3D: SNIPER CELL](https://webxos.netlify.app/snipercell) that trains **RL datasets** for Hugging Face for leveling/exp, with dataset creation exp version available on [HuggingFace](https://huggingface.co/datasets/webxos/snipercell_RL_v1).
- **PyTorch training now has **TraceML**!**: A member built [TraceML](https://github.com/traceopt-ai/traceml), live observability for PyTorch training that tracks real-time dataloader fetch time, GPU step time, live CUDA memory tracking, and layerwise memory and timing in backward and forward pass, with a [detailed writeup](https://medium.com/p/af8fbd899928).
- **Create magnetic fields with webXOS MAGNET DATASETS IDE**: A member shares the [webXOS MAGNET DATASETS IDE](https://webxos.netlify.app/magnets) along with the dataset [webXOS_magnet_dataset](https://huggingface.co/datasets/webxos/webXOS_magnet_dataset) that contains simulated magnetic field measurements for various magnet configurations.
- **Embeddr-net backend is here!**: The new version of [embeddr-net](https://github.com/embeddr-net/embeddr-cli) is out! Comes with the editor with mcp and workflows, plugins, basic dataset creation, captioning with clip and moondream, dupes, tags, lineage. Using the [comfyui nodepack](https://github.com/embeddr-net/embeddr-comfyui) u can set it up to load images and upload images to the search.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1456746539429920876)** (17 messages🔥): 

> `Authentication Issues, Course Order, Evaluator Errors, Agents Course Location` 


- **Authentication Woes Plague Agents Course**: Several members reported experiencing **401 errors** when authenticating with the **Colab notebook** in the Agents course, despite having full inference or read/write permissions.
   - One member suggested paying for more usage to overcome the issue, while another encountered a similar issue when connecting to LLMs via API key.
- **Course Prerequisites Confusion**: A new user inquired about the recommended order for Hugging Face courses, specifically noting LLMs and MCP as potential prerequisites for the agents course.
   - It was because *there's a bunch of others as well, including LLMs and MCP that feel like prerequisite knowledge*.
- **Evaluator Errors Haunt Unit 4**: A user reported encountering an error in the Unit 4 final assessment where the evaluator could not find a file associated with given task IDs.
   - They confirmed the API returned a **404 error** and wondered if this was an issue with the evaluator or if they needed to explicitly handle file downloads in their code.
- **Channel Visibility Questioned**: A user inquired about the visibility of 'agents-course' related channels mentioned in the Onboarding section.
   - Another member provided a [link to the first unit](https://huggingface.co/learn/agents-course/unit1/introduction) and the original poster clarified they meant the Discord channels, not the course content itself.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1456739184797683909)** (353 messages🔥🔥): 

> `Environment management woes, Tavily-MCP vs Exa-MCP, Recursive `AGENTS.md`, Opus 4.5 Degredation, Stuck on Planning Next Moves` 


- **User hates env mgmt, explodes at Cloudflare BS**: A user vented about their hatred for environment management, particularly when dealing with **Cloudflare**, **GitHub secrets**, **CI/CD**, **Wrangler**, and runtime configurations, preferring to work with only one **CF worker**.
- **Recursive `AGENTS.md` functionality limited to Gemini 3 Pro**: Users discussed the new-ish recursive `AGENTS.md` functionality, noting that only `Gemini 3 Pro` seems to fully support the concept.
- **Opus 4.5 Drains Wallets, Delivering Awful Results**: Several users complained about **Opus 4.5** becoming extremely expensive and delivering poor results, with one stating they were *just waiting money now*, because of the stupid mistakes it makes.
   - Alternatives mentioned by the community include **GPT 5.2 codex** and **bug fixes**.
- **'Planning Next Moves...' bug plagues members**: Multiple users reported getting stuck on *Planning next moves...*, with one user detailing extensive troubleshooting steps and linking to a [temporary solution](https://forum.cursor.com/t/planning-next-moves-stuck/143985/367) on the Cursor forum.
   - A temporary fix involves clearing app data.
- **Cursor Slows IDE Speed**: Members discussed issues with Cursor slowing down the IDE screen, making variable reading and overall performance sluggish.
   - Suggestions included upgrading to a faster computer (especially with high single-core CPU performance, like a Mac), keeping the workspace clean (few chats/tabs open), and ensuring only necessary servers/terminals are running.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1456759353355796541)** (205 messages🔥🔥): 

> `Open Datasets, Qwen2.5 Training, Pickle startup, Instruction training from scratch, Hermes benchmark` 


- ****Open Datasets Abound on Hugging Face & Kaggle****: **Hugging Face** has **672,685** open datasets and **Kaggle** has **636,009** open datasets, according to members in the chat.
   - Someone joked that *visualizations are unintentionally funny* on Kaggle
- ****Engineering Handbook Released for Qwen2.5 Training****: A member released an engineering guide for training **Qwen2.5** with **GRPO + LoRA** using **4x A100 SXMs** and the **verl** framework, with [links to the GitHub repo](https://github.com/volcengine/verl) and [Medium article](https://medium.com/@weyaxi1/the-engineering-handbook-for-grpo-lora-with-verl-training-qwen2-5-on-multi-gpu-b2431a2a8e92).
   - Another member asked if they'd be willing to integrate **Atropos** with **verl**, linking to [a GitHub issue with a bounty](https://github.com/volcengine/verl/issues/1782).
- ****Debate Sparks Over China's Open Source AI vs. US Closed Source****: Members debated whether **China's open-source models** are closing the gap with **US closed-source models**, particularly in frontier capabilities.
   - One member argued that the *trendline trajectory favors China OS*, while another suggested that the **CCP's** regulatory approach might limit Chinese AI labs' potential, pointing to [Dwarkesh's podcast about Xi Jingping and AI](https://dwarkeshpatel.com/2024/01/04/yanzhong-huang-on-china-ai-and-the-ccp/).
- ****Newcomer Eager to Test Hermes's Mettle****: A new member expressed excitement about **Hermes**, offering to conduct extensive benchmark tests focusing on **morals, ethics, sycophancy, organic learning, and long-term mental health implications**.
   - They also offered to interview project members and host small cash prize competitions for breaking their prompts.
- ****Heretic Tool Explored for Uncensoring and Stripping Sycophancy****: A member inquired about using **Heretic** ([p-e-w/heretic on github](https://github.com/p-e-w/heretic)) to investigate the impact of safety/alignment on model capability.
   - Another member responded that they have *their own rl env for that (RefusalBench Env)*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1456761044360757369)** (2 messages): 

> `Model Nerfing, Fear of Power` 


- **Model Nerfing Questioned**: A member questioned why they would nerf themselves on purpose.
   - No links or further context was provided.
- **Powerful Model Provokes Fear?**: Another member jokingly suggested the model was being nerfed because others were scared of how powerful it could be.
   - No links or further context was provided.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1456818346447212645)** (156 messages🔥🔥): 

> `Doubly Stochastic Matrices, Matrix Residual Streams, Sinkhorn Knopped Matrixes, AI alignment chart, SAEs for feature estimates` 


- **Sinkhorn Matrixes Sum Reduce**: A user found that multiplying several [Sinkhorn knopped matrixes](https://arxiv.org/abs/2305.10690) causes vectors to converge to a vector of **1/n**, preserving only the mean value.
   - Another user agreed this could happen at initialization, but believes networks can learn mappings overall, and the paper focuses on the stability of the product of matrices with a spectral radius **<= 1**.
- **AI Researchers Split on Alignment**: A user shared a [strawpoll about AI alignment](https://strawpoll.com/PKgleOeMoZp) and an attached image of an **AI researcher alignment chart**.
   - Another user commented that Dario Amodei is the only person on the chart who can influence the outcome and has an interest in selling it that way.
- **SAEs Illuminate Feature Count in LLMs**: Discussion revolved around estimating the number of features in modern LLMs using **Sparse Autoencoders (SAEs)**, referencing a [Transformer Circuits publication](https://transformer-circuits.pub/2024/scaling-monosemanticity/) that trained **SAEs up to 34M**.
   - Estimates suggest that the count of features in LLMs may reach **100M** or more, based on the progress of SAE training and feature recovery.
- **W&B Experiment Tracking Woes**: A grad student expressed frustration with tracking changes between runs in **Weights & Biases (W&B)** when managing a large number of training experiments.
   - Suggestions included using **VCS**, logging commit hashes, and taking detailed notes, with a call for tools that automate cross-run analysis and suggestions.
- **Flow Matching Reversibility**: There was a discussion on whether **diffusion models are reversible** and their relationship to **flow matching**, with a member saying that the magic is **OU process** and **Schrödinger Bridge** are reversible.
   - A link to a [Yannic Kilcher video](https://www.youtube.com/watch?v=7NNxK3CqaDk) was provided.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1457213090562310338)** (8 messages🔥): 

> `Signal 39C3 presentation, Dead canary reference, Paper discussion channel update` 


- **Signal Dazzles with 39C3 Presentation**: Signal delivered a presentation at the **39C3 conference** which can be found [here](https://youtu.be/0ANECpNdt-4?si=DSbziZ2LET_zR0io).
   - The presenter joked about a *dead canary on the podium*, possibly symbolized by a *maneki-neko*, as a reference to the "canary in the coal mine" concept related to **E2EE**.
- **Paper Discussion Channel Returns Next Week**: The daily paper discussion channel is currently on hold due to **holidays and other commitments**.
   - It is expected to return next week.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1456825172983025675)** (34 messages🔥): 

> `SWE-Bench Verified Fraud, LeCun's Emotion AI, Patent System Joke, Falcon-H1R-7B Model` 


- **SWE-Bench Claims Debunked**: A user shared a link to a claim regarding **SWE-Bench Verified**, but quickly debunked it, citing a bug in the eval code where *the model cheated by looking in git history* - see [original X post](https://x.com/rohanpaul_ai/status/2006813146170929409?s=46).
- **LeCun Aims for Emotional AI**: LeCun claims to be building **AI with emotional reactivity**, and perceptions governed by emotion, using videos to give **AI models** an understanding of the physics of our world - see [archive link](https://archive.ph/E9zai#selection-2255.0-2266.0).
   - He says we will see *baby versions* of this within **12 months**, and on a larger scale within a few years, but one user pointed out that he may be attempting to copy the work of their team.
- **Patent System Called a Joke**: A user mentioned that their team already has a patent ([https://patents.justia.com/patent/20250284921](https://patents.justia.com/patent/20250284921)), but the tech industry is used to doing illegal things, and then billing settlements and legal fees as a cost of doing business.
   - Others agreed, saying that *investors still ask for it*, and that the system is *an "I own this idea unless you have more money than me" kind of deal.*
- **Falcon-H1R-7B Announced**: A user shared a link to the **Falcon-H1R-7B** model, a new model release from Falcon - see [blogpost](https://falcon-lm.github.io/blog/falcon-h1r-7b/).


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1457054602447814951)** (146 messages🔥🔥): 

> `listChanged clarification, Capability Negotiation Process, Dynamic Tools Support, Client Initialization Payloads Variance, Negotiation vs Selection` 


- **Decoding listChanged's Role in MCP Notifications**: The `listChanged` flag serves as a heads-up to the client that the server *may* send notifications when primitive lists change, such as when a new tool is added or removed, prompting the client to make a `tools/list` call, as shown in the [MCP documentation](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#list-changed-notification).
   - Although clients are not obligated to act on these notifications, ignoring them can be *super annoying*.
- **Navigating Capability Negotiation Nuances**: The capability negotiation process in MCP involves clients advertising their available features to the server, which responds with its supported capabilities, primarily around authentication approaches and SSE resumption, as outlined in [this discussion](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/604).
   - This exchange isn't a *handshake*, but an advertisement of available features, with a leaning towards optimistic implementation as the *general direction*.
- **Dynamic Tools Stir Debate on Runtime Changes**: Dynamic tools, which can change descriptions or parameters based on interaction, are supported within MCP, though some view them as a *rug pull*.
   - The argument is that **MCP's malleability** is a feature allowing LLMs to adapt to changes, contrasting with the rigid contracts of traditional systems.
- **Client Initialization Payloads spark schema questions**: Clients send different payloads during initialization, as shown with the Cursor client (`true` instead of `{}` for `object` type properties) and the Fast-agent client (doesn't share whether it supports or not).
   - Per [the schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/087967e9b34dc959f6b5336c93acf56510730a56/schema/2025-11-25/schema.ts#L308), those server capabilities are not necessary in initialisation and should be treated optimistically.
- **Should 'Negotiation' become 'Selection'?**: A spec contributor questions if the word `Negotiation` should be changed to `Selection` because clients declare their capabilities which the server *selects* to support.
   - However the suggestion was received poorly with the question *why would we do that?*


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1456745182866313298)** (119 messages🔥🔥): 

> `Boris Cherny's Claude Code, Continual Learning Architectures, OpenAI President's Political Donation, Karri Saarinen's tweet, AI Model Lab IPOs` 


- ****Boris Cherny** shares **Claude Code** insights**: **Boris Cherny**, creator of **Claude Code**, shared that while his own setup is surprisingly vanilla, the product is designed to be highly customizable and hackable.
   - You can view his discussion [here](https://x.com/bcherny/status/2007179832300581177?s=46).
- **Unveiling Continual Learning Approaches**: A discussion emerged around vague posts from multiple frontier lab employees about continual learning, with speculation that they may release a **context management system** (*long context + recursive self-management of context contents + vector store*).
   - It's speculated this may be called "continual learning" even though no weights are actually modified; related discussion can be found in the [Konwinski podcast](https://youtu.be/ZagdY6UJYL4).
- ****METR** Evaluates **Claude Opus 4.5****: **METR** reports that **Claude Opus 4.5** achieved their highest published **50%-time horizon** to date, estimated at approximately **4 hours and 49 minutes** based on their task performance evaluations; their eval can be found [here](https://x.com/METR_Evals/status/2002203627377574113).
- **Deep Dive into Agent Sandboxing**: A blog post by beowulfbr titled "Sandboxes for AI" covers the predicate differences between **containers** (shared kernel), **gVisor** (userspace kernel), **microVMs** (guest kernel + VMM), and **Wasm** (no syscall ABI).
   - The post discusses why containers aren't sufficient for hostile code, what "policy leakage" looks like in agent systems and practical tradeoffs for different agent architectures, with the post available [here](https://www.luiscardoso.dev/blog/sandboxes-for-ai).
- ****Steve Yegge** Launches **Gas Town** Coding Agent**: **Steve Yegge** announced the release of **Gas Town**, a new orchestrator for coding agents, detailing the project's launch and functionality in a [Medium article](https://xcancel.com/Steve_Yegge/status/2006835043503845445).
   - One member commented *Honestly Steve's post reads like AI slop so I'm not going to bother* but another said *He's always written like this 😂* and another noted *He's deeply influential*.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1457312344865636435)** (2 messages): 

> `Past Discussion on Discord, Lack of Time for Research` 


- **Blast From The Past Discussion**: A member referenced a [past discussion](https://discord.com/channels/822583790773862470/1342964204168020018/1436045559491199017) and thread below related to the current conversation.
   - They mentioned they haven't had time to dig further into the topic since then.
- **Time Constraints Hinder Deep Dive**: The member expressed regret for not being able to explore the discussed topic further due to time constraints.
   - This limitation prevented them from providing more detailed insights or updates on the subject.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1457200782423363736)** (4 messages): 

> `Fal.ai Career Opportunities, X-Ware.v0` 


- **Fal.ai is Hiring!**: User @isidentical promoted [career openings at fal.ai](https://xcancel.com/isidentical/status/2007370275650974034?s=46).
   - It was suggested that **Fal.ai** is anticipating significant expansion and is encouraging potential applicants to submit their applications.
- **X-Ware.v0 announcement**: There was a promotion of the **X-Ware.v0** on the channel.
   - Further details of **X-Ware.v0** were not disclosed.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1456804049528684585)** (17 messages🔥): 

> `Logit Processor Output, DGX Spark vs Jetson Thor, GPU Profiling -linetable, critical path analysis, CPU vs GPU perf speedup` 


- **Logit Processor's Choice Manifested!**: A member asked about the "output" of the **logit processor**, and another member replied that the **logit modification happens in-place**, by modifying the **logits tensor**.
   - The first member followed up to clarify if the code they saw was a complete implementation.
- **Sparking Thorny Debate: DGX Spark vs Jetson Thor!**: A member bought a **DGX Spark** but is returning it, since the **Jetson Thor** has better performance for cheaper and supports **tcgen05/stochastic rounding**.
   - Another member who owns both says the **Spark** is slightly faster on **inference** and the same on **training**, but believes in **Thor** long term, especially with **tcgen05**-based features and custom fan curve, however the bandwidth is lower on **Thor** if not going for multi-nodes setup.
- **GPU Profiling -linetable Question**: A member asked about **-linetable** in a **GPU MODE** profiling tutorial, wondering if it was a typo for **-lineinfo**, included picture.
   - Image was attached to message, but resolution/content could not be determined.
- **Critical Path Caught in Analysis Paralysis!**: A member asked what tools others use for **critical path analysis**, mentioning **Meta's Holistic trace analysis util** wasn't helpful.
   - The member also mentioned a **20ms reduction in a forward call** that resulted in **zero reduction in overall latency**.
- **Benchmarking CPU vs GPU: Time Flies!**: A member asked about the standard methods for **benchmarking CPU vs GPU performance speedup**.
   - The member has been using the **'time' command in Linux** or **std::chrono** and wants to know if there's something more robust.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1456756309679538349)** (20 messages🔥): 

> `V100, sm120, cublasDgemm, Compute Sanitizer Patching API, warp group` 


- ****V100** is m8n8k4 with f16**: It's recommended not to use **m8n8k4 with f16** for anything other than **V100**.
- **sm120 uses same mma instructions as Ampere**: For **sm120** when doing **bf16** and accumulating in **fp32**, the `mma` instructions are the same as for **Ampere**, sm_120 is basically **sm_89** plus support for **mxfp8/6/4** and **nvfp4**.
- **cublasDgemm returns zero matrices**: A user was experiencing issues with `cublasDgemm` returning zero matrices and was advised to check the return status of `cublasdgemm`, and/or `cudadevicesynchronize` after the dgemm call and check `cudagetlasterror`.
- **Compute Sanitizer Patching API fails**: A user was facing issues with `sanitizerAddPatchesFromFile` failing with `INVALID_PARAMETER` on Windows when using the **Compute Sanitizer Patching API**.
   - They later resolved it by correcting the function call placement.
- **Warp Group Producer Design**: The use of a full warp group (4 warps) as a producer, instead of a single warp, is likely due to `setmaxnreg` applying at the warp group level, indicating a technical limitation when warp specialization is desired, based on this [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1457127582301683805)** (4 messages): 

> `AI startup hiring, Research Engineer roles, Inference Engineer roles, Prompt injection protection` 


- **AI Startup White Circle Hiring**: CTO is hiring at a startup that protects dozens of startups from **prompt injections** and inappropriate usage, processing millions of requests daily, for both [research engineer and inference engineer roles](https://jobs.ashbyhq.com/whitecircle/a030c9a9-dc20-490c-9c51-03e87210f904).
   - The compensation ranges from **100-250k** (more for exceptional people).
- **White Circle: Research Engineer Roles**: The research engineer roles require expertise in **MoE, multimodality (audio/images), Megatron, distributed training, and Triton**.
   - These engineers will focus on research and development within the company.
- **White Circle: Inference Engineer Roles**: The inference engineer roles require expertise in **TensorRT, vLLM, and SGLang** to optimize inference performance.
   - The goal is to *make inference go brrrr*.
- **Prompt Injection Protection Services**: The startup is akin to CloudFlare, but specializing in safeguarding against **prompt injections** and inappropriate AI usage for numerous startups.
   - The company handles millions of requests daily and is rapidly scaling its operations.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1456869948810068152)** (12 messages🔥): 

> `Free GPUs online, Vectorization and Grid Strides in production kernels, Implementing Flash Attention in Triton, Google Colab in VS Code, OpenCL Code` 


- **Leverage Free GPUs Online for Learning**: Members suggested using free or nearly free GPUs available via platforms like [Google Colab](https://colab.research.google.com/), [Vast.ai](https://vast.ai/), and [Modal](https://modal.com/) for learning purposes.
   - These resources allow users to learn without the need for a physical GPU, making it accessible to a broader audience.
- **Dimensional Analysis Simplifies Kernel for-Loops**: A member discovered that using **dimensional analysis** simplifies setting up *for* loops for vector addition using vectorization and grid stride.
   - They suggest this method helps in understanding what to include in the *for* loop without needing to constantly reference tutorials, by *keeping track of units*.
- **Vectorization and Grid Stride Applications**: A member inquired about the frequency of **grid stride** and **vectorization** use in production-level kernels, especially regarding advantages in memory throughput and processing larger datasets.
   - They pondered on whether applying both techniques is standard practice, considering datatype support by CUDA for vectorization and thread limits for grid stride.
- **Triton FA1 Implementation Lags**: A member shared that their implementation of **Flash Attention 1 (FA1)** in [Triton](https://triton-lang.org/) performs at the same speed as a naive PyTorch implementation.
   - They requested feedback on their implementation to identify potential issues and improvements.
- **Colab Docks in VS Code**: Members shared a [Google Developers Blog post](https://developers.googleblog.com/google-colab-is-coming-to-vs-code/) about bringing Google Colab to VS Code.
   - This integration could be beneficial for individuals without access to dedicated GPUs, offering a convenient coding environment.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1457120426088993111)** (2 messages): 

> `Coordinate Order Clarification` 


- **Coordinates order is actually (z,y,x)**: A member clarified that when coordinates are said to be in reverse, the label *(0,0,3)* actually means *(z,y,x)* order.
- **Confirmation of Coordinate Order**: Another member confirmed the clarification regarding the reversed coordinate order.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

denizay5566: Anyone in Seoul ?
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1457620030098505918)** (1 messages): 

> `PR traffic generation, Subscribing to all PRs, Discord traffic, Github notifications` 


- **PR Subscriptions Spark Traffic Surge**: Subscribing to all **Pull Requests** generated a surprising amount of traffic.
   - *Yikes*, one user exclaimed upon realizing the volume of **Github notifications** that resulted.
- **Discord channel traffic spike**: Discord channel experienced an incredible traffic surge after users subscribed to all **Pull Requests** on Github.
   - The channel has been bustling with activity, raising concerns about managing the increased volume of notifications and discussions.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1457588920836554954)** (11 messages🔥): 

> `FLE, Prime Environments, LLM, Factorio, Inspect` 


- ****FLE** ported to Prime Environments for New Years!**: A member recently worked on porting **FLE** to prime environments, showcasing it on [Prime Intellect](https://app.primeintellect.ai/dashboard/environments/daspartho/factorio).
   - They also bumped an issue they found during the process, related to the [Factorio Learning Environment](https://github.com/JackHopkins/factorio-learning-environment/issues/352).
- **New entry point uses Inspect to Handle **LLM** calls**: The new entry point will use **Inspect** to handle **LLM** calls, including summarization and compression.
   - Members were asked to raise a PR fixing the bug found for the previous entry point.
- **Preparing Patch for **Factorio** 0.3, Excited for 0.4**: A patch to fix version **0.3** of **Factorio** was being prepared.
   - The author expressed excitement for version **0.4**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1456757277045428224)** (6 messages): 

> `CuteDSL Float32 to Float8 Conversion, Cutlass Version Compatibility, Vectorization in Cutlass, Thread Reduction and Result Storage` 


- **CuteDSL Scalar FP32 to FP8 Conversion Unsupported**: A user inquired about converting from **Float32** to **Float8** in **CuteDSL** but received a traceback indicating that direct conversion of narrow precision scalar types like **fp8** to/from **fp32** is not supported.
   - A member pointed out that conversion is possible if the vector size is **32bit** aligned, providing a code snippet to illustrate the solution.
- **Vectorization Vexes Voracious Viewers**: A member shared an example code snippet that performs **Float32** to **Float8** conversion using vectorization with **CUDA DSL**.
   - They also recommended reading the [elementwise_add.ipynb](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb) notebook as a starting point for understanding vectorization techniques.
- **Thread Reduction Results Require Restraint**: A user asked about the best approach for storing the result of a reduction operation across threads.
   - They were concerned about whether every thread should store the result (potentially causing multiple writes to the same location) or if only one thread should handle the write, seeking an efficient, branchless implementation.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1457028615068848316)** (13 messages🔥): 

> `CUDA Rust Integration, Python/Rust Split, Kernel Acceleration, NV vs AM Support, Onboarding Docs Improvements` 


- **CUDA Rust Hello World Achieved**: Successfully got a **CUDA Rust hello world** working using [rust-cuda](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html#docker), enabling CPU kernels in Rust with `std::simd` and `std::arch`, and GPU kernels with <@903087684283682826>'s `rust-cuda`.
   - The implementation uses **pyo3** for Python-Rust bindings, facilitating AOT compiling as a Python module; it is considered a superior approach with core in Python and Rust for kernel acceleration, enabling easy graduation to *tinygrad*, *torch*, and similar frameworks.
- **New Python/Rust Split Feels Much Better**: The strategy of splitting core functionality in Python with Rust used solely for **kernel acceleration** is favored over splitting everything into Python and Rust with a thin Python shim.
   - This new approach allows for smoother progression to frameworks like **Tinygrad** and **Torch**.
- **Launching CUDA Kernel with pyo3 Bindings**: Successfully launched a **CUDA kernel** with `pyo3` bindings and `cuda-rust`; the scripts used are directly from rust-cuda's hello world, installing llvm7 for rustc's nvvmir codegen.
   - It's recognized that Python should drive memory allocations for smoother transitions to Tinygrad, with Rust used exclusively for compute kernels by passing allocations/CUDA context based on the siboehm/pranjal/gaunerst blogs.
- **NVidia vs AMD Target**: The use of Rust to launch CUDA kernels may limit the codebase to **Nvidia support only**, conflicting with the intent to target both Nvidia and AMD, especially given AMD's open-source instruction sets.
   - Despite this, the comfort level with Rust syntax outweighs the concerns, leading to a decision to proceed and evaluate the outcome.
- **Improve Onboarding Docs**: Plans are in place to improve **onboarding documentation**, creating an intermediate step before the textbook, once a vertical pipeline/trace is established for add and mul operations.
   - An **ARCHITECTURE.md** file, as well as a **CLAUDE.md** file, will be added.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

kitsu5116: https://arxiv.org/abs/2512.24545
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1456776510433394709)** (11 messages🔥): 

> `CUTLASS Usage, B200 GPU and CuTeDSL, Evaluation Methods` 


- **CUTLASS is included in the competition**: A member asked about **CUTLASS** usage in the competition, and it was confirmed that it's already included and can be used with `#include` directives, specifically mentioning `cutlass/cutlass.h` and `cute/tensor.hpp`.
   - One of the members inquired about the correct path or environment variable (**CUTLASS_DIR**) to use for discoverability during `torch.utils.cpp_extension` builds.
- **B200 GPU Unleashes 2 CTA GEMM**: A member shared a [blog post](https://veitner.bearblog.dev/2-cta-gemm-on-b200/) and [LinkedIn post](https://www.linkedin.com/posts/simon-veitner-174a681b6_2-cta-gemm-on-b200-activity-7413641925691338752-p9s7?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeks) detailing how the **B200 GPU** allows computing MMA operations collectively on 2 CTAs using **CuTeDSL**.
   - The post focuses on adjustments needed to turn a simple **GEMM** into a 2 CTA version, assisting beginners in leveraging the newest hardware features by adjusting their custom kernels.
- **Better Eval Method Selected**: A member inquired about the evaluation method used, asking whether it was `eval_better_bench.py` or `eval.py` from the [reference-kernels](https://github.com/gpu-mode/reference-kernels) repo.
   - The `eval_better_bench.py` method is currently in use, with the mapping found in the [task.yml](https://github.com/gpu-mode/reference-kernels/blob/4b7c7b5be7ee3c98350da9536a2f9541f4adb6e7/problems/nvidia/nvfp4_dual_gemm/task.yml#L8) file.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1457094443453649127)** (2 messages): 

> `Full-Stack ML Engineering Roles, vLLM Talent Pool` 


- **ML Engineers Seek Full-Stack Ownership**: An ML engineer is looking for roles in companies where **ML engineers own the full stack**, from training through production deployment and inference optimization.
   - They are currently a senior MLE at a large fintech company where these responsibilities are siloed, seeking companies that structure work differently.
- **vLLM Launches Talent Pool**: A member shared a link to the **vLLM** talent pool, indicating that the project is aggregating talent for companies using their tech stack.
   - The [X post](https://x.com/vllm_project/status/1792979748067357179) links to a Google Form with questions about experience and interest.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1456852006298849372)** (69 messages🔥🔥): 

> `Moonshot AI fundraising, Impact of AI on various industries, Kimi's performance with Linux drudgery, Minimax agent analysis, Context window and memory usage` 


- **Moonshot AI Raises Half a Billion**: A [news article](https://www.scmp.com/tech/tech-trends/article/3338334/chinas-moonshot-ai-raises-us500-million-latest-funding-round-report) reported that **Moonshot AI** raised **$500 million** in its latest funding round.
   - One user expressed congratulations on the fundraising success.
- **Debate: AI as Another Tool**: Users debated the role of **AI** as a tool, with one engineer praising **Kimi's** capabilities in FPGA engineering, sysverilog, vivaldo, and AMD xillix, calling AI "*just another tool*."
   - Counterarguments likened opposing AI to opposing computers, the internet, or even digital cameras, arguing, "*The moment you accept any shortcut, you’ve conceded the principle - you’re just haggling over price.*"
- **Kimi Excels at Linux Drudgery**: A user shared that they *trust Kimi enough with Linux drudgery stuff using sudo*, but cautioned that "*you just gotta watch him he will get frisky on you*."
   - They described a scenario where **Kimi** attempted to directly modify an important system file, requiring manual intervention.
- **Minimax's Video Analysis**: A user praised **Minimax** for its ability to provide transcripts and analysis from **YouTube videos**, highlighting its understanding of video and audio.
   - Another user confirmed this capability, describing the **Minimax agent** as a *nice little tool*, likening it to having *a computer on the cloud with an assistant to go*.
- **Navigating Context Window Limits**: Users discussed the limitations of the **context window**, with one user expressing frustration over the tedium of workarounds like splitting files for summarization.
   - Suggestions included using **OK Computer** to search within files, but users acknowledged its limits, emphasizing the need for more efficient memory implementation.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1457017196604686559)** (2 messages): 

> `` 


- **No significant discussion**: No meaningful topics were discussed in the provided messages.
- **End of message history**: The message history concluded without any topics suitable for summarization.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1456747560076054660)** (53 messages🔥): 

> `NuMojo matrix lib status, Optimizing Mojo build times with Bazel, GPU float64 warp shuffle limitations, "View Source" in Mojo documentation, Triton arange equivalent in Mojo` 


- **NuMojo Matrix Library Status**: A member inquired about the development status of the **NuMojo matrix library** and its readiness for external contributions via pull requests.
   - The request was filed as a [GitHub issue](https://github.com/modular/modular/issues/5733).
- **Bazel builds are slow, lacking incremental compile?**: A user reported slow build times (3+ minutes) when using **Bazel** and **rules_mojo**, particularly with GPU, Python, and C++ interop, seeking guidance on optimization and code/module layout patterns.
   - It was noted that **Mojo** currently rebuilds parts of the **stdlib** from a parsed AST without caching, and **Bazel's cache** is the only one utilized, even if Mojo had incremental compilation support.
- **GPU Warp Shuffle Excludes Float64s?**: A member questioned the absence of **float64** support in the logic for **warp shuffles** in the Mojo GPU primitives library, inquiring if they could be handled similarly to **int64** and **uint64** types, referencing the [relevant code](https://github.com/modular/modular/blob/main/mojo/stdlib/std/gpu/primitives/warp.mojo#L93).
   - No answer given.
- **"View Source" Button Debuts in Mojo Docs**: A user noticed the "view source" button in the documentation, questioning whether it was a recent addition.
   - A member confirmed it was relatively new.
- **Range Floor Division Troubles in Mojo**: A user encountered an error (`_SequentialRange' does not implement the '__floordiv__' method`) while attempting floor division on a range when converting a **Triton** kernel to **Mojo**.
   - It was suggested to use `math.iota` for compile-time known values or `max.nn.arange.arange` for runtime values, along with a discussion on using `LayoutTensor` and `LayoutTensorIter` for tensor operations within custom kernels, pointing to [relevant documentation](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensorIter).


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1456749067852841043)** (10 messages🔥): 

> `MEF files, Mojo MAX API, GPU support` 


- **Modular Compiled Functions: MEF files revealed**: Members discussed using **MEF** (Modular Executable Format) files, which can be extracted from the compilation cache, to execute generated Mojo code outside the graph, referencing the [max/include/max/c](https://github.com/modular/modular/tree/main/max/include/max/c) directory for usage examples.
   - A member noted that there's an end-to-end example in max/examples, making it fairly easy to use.
- **GPU Support Glitches in MEF Files**: MEF files currently have known limitations, primarily **lacking GPU support**.
   - Despite being a historical artifact, it is being supported because it powers the Mojo MAX API and there's ongoing interest in its use.
- **Mojo MAX API revealed**: The **Mojo MAX API** is currently powered by MEF files.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1456741543556940028)** (24 messages🔥): 

> `Manus down?, Cancelling AI, Manus credits, AI Engineer Job, Meta buys Manus?` 


- **Manus Crash Leads to Account Chaos!**: Several users reported **issues with Manus being down** and experiencing problems with terminal, browser, code captures, and account access.
   - One user exclaimed, *"Manus crashed !!!!! And now I can’t move around nothing in my account what is this!!!!"*.
- **Query about Halting AI Progress**: A member posed the question, *"Como detener las ia,s"*, or *"How to stop the AIs"*.
   - No further context or discussion was provided.
- **Subscription Snafu Forces User to Restart**: A user was advised to contact Manus Support to restore to a checkpoint due to an issue, and also mentioned account switching integration.
   - Another user was informed that their overdue subscription had been canceled, allowing them to try again after experiencing an issue, with support saying *We couldn't find your subscription record. Could you DM me more details, like your order number?*.
- **Job Opportunity Alert for AI Engineer**: A member inquired whether anyone was looking for a skilled AI engineer.
   - No further information about job requirements or preferred skills was shared.
- **Meta Acquisition Rumors Spark Concern!**: A user speculated that **Meta** is going to acquire Manus, leading to concerns about the platform's future.
   - Another user echoed this sentiment, describing it as a *"Sinking ship*" and predicting *"Lesser, lower quality outputs..."* akin to **ChatGPT's** decline, alongside worries about data siphoning under the guise of "safety" and [shared a link to a relevant X post](https://x.com/ganbayards/status/2008133609098727915).


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1456756529570123819)** (4 messages): 

> `Better Evals, GEPA win counts, rig-rlm, regspy` 


- **Crafting Superior Evals**: A member discussed writing about building better evals before the end-of-year break, highlighting the gap in understanding what to evaluate and the potential pitfalls, as detailed in the blog post [Building Better Evals](https://www.sourceallies.com/2025/12/building-better-evals/).
- **GEPA Win Count Anomalies**: After running **GEPA** on a larger dataset, it was observed that the **1st candidate** (**0.8454**) had a win count of **58** and unique win count of **7**, while the **4th candidate** (**0.8208**) had a win count of **86** and unique win count of **20**, which was the highest among the top five candidates.
   - The member interpreted this as the **4th candidate** being an all-rounder that couldn't quite reach the top three.
- **"rig-rlm" Regex Pattern Generator unveiled**: A member mentioned [rig-rlm](https://github.com/joshua-mo-143/rig-rlm), a regex pattern generator using a 3B model.
- **"regspy" Repository shared**: A member shared [regspy](https://github.com/NathanZaldivar/regspy), noting they've been experimenting with optimizers and inferred rules, and requested feedback.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1456817030325075999)** (11 messages🔥): 

> `RLM integration, Parallel Tasks, Human-in-the-loop, Reading files` 


- **RLM Integration To Be Slow-Rolled for Security**: The integration of **RLM research** into **DSPy** is still planned, but is being deliberately paced to address sandboxing and **security aspects**, with consideration of whether to integrate as a *dspy.Module* or a new higher-level entity.
   - There has also been some debate about whether to expose it as part of *dspy.Module* or a brand new higher level entity in **DSPy**, which will affect API design.
- **Parallel Processing Performance Probed**: A user inquired about the best way to handle **parallel tasks** in a multi-module program involving nested module calls, **S3 calls**, and **vector searches**, expressing concerns about the overhead of creating a unique thread pool executor for each call.
   - The user referenced the [parallelizer.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/utils/parallelizer.py) file and wondered about its effects on **optimizers** and **traces** when using a separate executor.
- **Human-in-the-Loop Handling How-To**: A user asked about implementing **human-in-the-loop** for **ReAct**, specifically how to save the trajectory of past events when a tool is called to ask a human, and how to return the human's response to continue the trajectory.
   - A user pointed to [this Github issue](https://github.com/stanfordnlp/dspy/issues/9154) related to parallel processing and asked for advice on a potential bug or code issue.
- **Temporary Files To The Rescue**: A user sought advice on reading a file into a string for a compiled **DSPy** program in an AWS Lambda environment where the file system is read-only, but later resolved the issue by using the **/tmp** directory.
   - An alternative solution was suggested involving parsing **JSON** from **S3** into a dictionary and using *load_state* instead of *load* with a file path, and a [pull request](https://github.com/stanfordnlp/dspy/pull/9158) was created to document this method.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1457727033814679572)** (5 messages): 

> `Company Update and Release, New Year Sprints, Assembly Optimizations, Llama and Flash Attention Integration, Claude for Code Cleanup` 


- **New Meeting Scheduled for Monday**: A new meeting is scheduled for **9am Monday San Diego time** covering topics such as company updates, new year sprints, assembly, and **llama flash attention**.
   - Other topics include using **Claude** to clean up stuff, viz / fast gemm, drivers, image dtype, and other [bounties](https://github.com/tinygrad/tinygrad/pull/1398).
- **Code Review Ready for Pull Request**: Pull request [13874](https://github.com/tinygrad/tinygrad/pull/13874) is now ready for review.
   - It joins outstanding issue [_CC](https://github.com/tinygrad/tinygrad/issues/13941) and pull request [13651](https://github.com/tinygrad/tinygrad/pull/13651).


  
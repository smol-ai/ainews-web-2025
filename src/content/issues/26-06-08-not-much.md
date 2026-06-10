---
id: MjAyNS0x
title: not much happened today
date: '2026-06-08T05:44:39.731046Z'
description: >-
  **FrontierCode** benchmark by **Cognition** highlights the challenge of coding
  tasks with the best model, **Opus 4.8**, scoring only about **13%** on the
  hardest subset, indicating coding is less solved than benchmarks suggest. The
  trend toward using **loops** as a control metaphor for coding agents is
  prominent, with emphasis on clear goals, verification, and iteration, though
  some experts caution about overreliance on loops. Agent ergonomics are
  improving with observability dashboards, sandbox environments, and workflow
  tools from **ClaudeDevs**, **MagicPath**, **LangSmith**, and **Modal**.
  **Kimi** by **Moonshot** released major updates including a stronger coding
  agent and a desktop agent product supporting up to **300 local sub-agents**.
  **Google** advanced efficient local deployment with upgrades to **Gemma 4**
  checkpoints.
companies:
  - cognition
  - frontiercode
  - moonshot
  - google
  - claudedevs
  - magicpath
  - langsmith
  - modal
models:
  - opus-4.8
  - gemma-4
topics:
  - coding-evaluation
  - agent-control
  - verification
  - agent-ergonomics
  - sandbox-environments
  - local-inference
  - workflow-optimization
  - cli-tools
  - plugin-integration
  - persistent-memory
people:
  - swyx
  - dzhng
  - claudecode
  - bcherny
  - reach_vb
  - omarsar0
  - gneubig
  - hamelhusain
  - angaisb_
---


**a quiet day.**

> AI News for 6/5/2026-6/8/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Coding Agents, Loops, and the Shift from “Passing Tests” to Mergeable Software**

- **FrontierCode raises the bar on coding evals**: Cognition introduced **FrontierCode**, a new benchmark explicitly targeting whether code is actually **mergeable**, not merely unit-test passing. Tasks were built with open-source maintainers, with each taking **40+ hours** and evaluated on dimensions like regression safety, cleanliness, scope, test correctness, and maintainability. The headline result is that the best model, **Opus 4.8**, scores only about **13%** on the hardest subset—far below the 50%+ regime common on SWE-Bench-style evals, suggesting coding is much less “solved” than popular benchmarks imply ([Cognition announcement](https://x.com/cognition/status/2064061031912288715), [Scott Wu’s summary](https://x.com/ScottWu46/status/2064073699368800475), [swyx breakdown](https://x.com/swyx/status/2064081945567580323), [theo’s questions on variance/reproducibility](https://x.com/theo/status/2064126021088215385), [Cognition response](https://x.com/cognition/status/2064215347503452649)).
- **“Loops” are becoming the dominant agent-control metaphor—but with caveats**: The day’s loudest practical theme was that coding agents should be given **clear goals, verification criteria, and iteration structure** rather than one-shot prompts. Popular examples include [dzhng’s “don’t use loops, design state machines”](https://x.com/dzhng/status/2063931263312892406), [Claude Code’s retrospective on auto mode, routines, and verification](https://x.com/ClaudeDevs/status/2064032814392352816), [bcherny’s thread](https://x.com/bcherny/status/2064034799711588805), [OpenAI Codex tips on outcome-first prompting](https://x.com/reach_vb/status/2064028260070215772) and [Approve-for-me defaults](https://x.com/reach_vb/status/2064044955421769755), plus [LangChain OSS “rubrics”](https://x.com/sydneyrunkle/status/2064034061165682931). But several practitioners pushed back on naïve loop hype: [Omar Sar0](https://x.com/omarsar0/status/2064024230396604469) and [Graham Neubig](https://x.com/gneubig/status/2064011013637234728) emphasized that human checkpoints remain essential outside easily verifiable domains, while [Hamel Husain](https://x.com/HamelHusain/status/2064019243990188259) joked about muting the word entirely.
- **Agent ergonomics are improving around verification and orchestration**: Product changes across the stack reflect this shift. [ClaudeDevs added observability dashboards for MCP connector developers](https://x.com/ClaudeDevs/status/2064072801062121906), including adoption, latency, and error views. [MagicPath launched a Builder plan](https://x.com/skirano/status/2064035120483352776) for external-agent workflows and multiplayer canvas editing. [LangSmith Sandboxes](https://x.com/LangChain/status/2064030008738296065) and [Modal’s sandbox scaling story](https://x.com/AmplifyPartners/status/2063998736703856737) point toward the same infrastructure trend: agents need isolated, inspectable, long-running environments.
- **Practical usage patterns are settling**: The strongest operator advice converged on measurable outcomes, bounded autonomy, and thread hygiene. [Angaisb_ warned against overlong Codex threads degrading performance](https://x.com/Angaisb_/status/2064103464142065852), while [reach_vb reported success with single-thread context accumulation](https://x.com/reach_vb/status/2064115851503059418). That mismatch itself is useful signal: current agent performance is still strongly shaped by **harness behavior and workflow choices**, not just base-model quality.

**Model Releases, Local Inference, and Serving Stack Upgrades**

- **Kimi shipped both a stronger coding agent and a desktop agent product**: Moonshot released a major update to **Kimi Code**, its open-source coding agent, adding **one-line CLI install**, drag-and-drop **video as coding context**, ACP support, plugins, and IDE integration ([announcement](https://x.com/KimiDevs/status/2063981516708024369)). It also launched **Kimi Work**, a desktop agent product with up to **300 local sub-agents**, browser-use via extension, finance-focused tool access, and persistent memory ([product launch](https://x.com/Kimi_Moonshot/status/2063990409903112344), [desktop availability](https://x.com/crystalsssup/status/2063992904209842215)).
- **Google pushed hard on efficient local deployment**: Gemma got several notable upgrades. New **QAT Gemma 4** checkpoints reportedly preserve performance while using **~4x less memory**, with **Gemma 4 E2B** fitting in about **1GB** using a mobile quantization format ([@_philschmid](https://x.com/_philschmid/status/2063990553826439378)). Separately, **Gemma 4 MTP** was merged into **llama.cpp**, enabling faster decoding when paired with QAT checkpoints ([Gemma team](https://x.com/googlegemma/status/2064030477628182814)). [llama.cpp also added video input support](https://x.com/osanseviero/status/2063985470489448887), expanding local multimodal use cases.
- **Open-source/open-weight competition remains intense**: [Artificial Analysis reported MiniMax-M3 at 55 on its Intelligence Index](https://x.com/ArtificialAnlys/status/2064066303863005254), which would make it the leading open-weights model once weights are released. M3 adds **native multimodality** and a **1M token context window**, with strong GPQA/MMMU-Pro numbers but notable abstention on hallucination-sensitive evals. Meanwhile [norpadon announced Apple-hardware-optimized quantized Qwen3.5 checkpoints](https://x.com/norpadon/status/2064040631479976240).
- **Serving infrastructure is broadening from text LLMs to world models and omni models**: **vLLM-Omni 0.22.0** added day-0 support for **NVIDIA Cosmos 3 world models**, robot serving APIs, TTS models such as **Qwen3-TTS** and **VoxCPM2**, faster image/video serving, and broader quantization/hardware coverage ([release](https://x.com/vllm_project/status/2064013506882703421)). This reflects a broader trend toward generalized multimodal serving rather than text-only inference stacks.

**Benchmarks, Evaluation Methodology, and Real-World Agent Measurement**

- **Agent evaluation is moving from synthetic tasks to in-the-wild telemetry**: Arena launched **Agent Arena**, a leaderboard based on over **1M real-world sessions**, using **causal tracing** rather than voting to estimate treatment effects of orchestrators/harnesses across five signals: **confirmed success, praise vs complaint, steerability, bash recovery, and tool hallucination** ([overview](https://x.com/arena/status/2064021507681276234), [methodology thread](https://x.com/ml_angelopoulos/status/2064028763697127844)). Whether the methodology fully holds up remains to be seen, but it’s one of the clearest attempts yet to benchmark deployed agents using actual usage traces.
- **Specialized benchmarks keep proliferating into new output domains**: Hugging Face and Mecado released **CADGenBench**, a benchmark for generating and editing **engineering-grade 3D CAD parts** from drawings or STEP modifications, with metrics covering geometry, topology, interface compatibility, and CAD validity ([launch thread](https://x.com/MikushRab/status/2063999885796614522), [Thom Wolf summary](https://x.com/Thom_Wolf/status/2064029993638764672)). This is a meaningful shift: evaluation is expanding beyond text/code into structured artifacts where correctness is physical and geometric.
- **A recurring thesis: good benchmarks become training pipelines**: [Ofir Press argued](https://x.com/OfirPress/status/2063990430350340575) that the best benchmarks are scalable and rooted in **real-world crawled data sources**, making them useful not just for measurement but also for data generation. That view shows up implicitly in both FrontierCode and Agent Arena: benchmarks are no longer static scoreboards; they are becoming **feedback loops for product and RL improvement**.

**Google, Apple, and the Consumer AI Platform Race**

- **Google expanded AI packaging, Search, and developer surfaces**: Google announced a more capable **NotebookLM** with agentic chat, stronger reasoning, and more output formats for Ultra subscribers ([launch](https://x.com/NotebookLM/status/2064016460964585549)). It also cut **Google AI Plus** pricing from **$7.99 to $4.99/month** while doubling storage to **400GB** ([pricing update](https://x.com/NewsFromGoogle/status/2064066310393209100)). On the platform side, [Google highlighted a major Search upgrade](https://x.com/Google/status/2064034586762354893), including multimodal search and **Gemini 3.5 Flash** as the new default in AI Mode.
- **Apple’s WWDC AI story centered on integration, not frontier leadership**: Commentary around WWDC focused on a rebuilt **Siri AI** with on-screen awareness, app actions, personal context, and better voice interaction, alongside concerns about **EU availability** and hardware gating ([kimmonismus live thread](https://x.com/kimmonismus/status/2064059964709388774), [regional limitation note](https://x.com/kimmonismus/status/2064047278105464868)). A technically notable detail came from [awnihannun](https://x.com/awnihannun/status/2064202168618422396): Apple’s on-device model is reportedly a **20B-parameter query-routed architecture** that loads experts from NAND into RAM once per query, a nonstandard design optimized for device constraints.

**Research Directions: Continual Learning, Agent Training, and Optimization Debates**

- **Anthropic framed one core blocker for AI in science as infrastructure mismatch**: Its new science blog argues AI has advanced faster in coding than biology because biological databases and tooling were not designed for agent use; the bottleneck is less raw intelligence than **agent-compatible scientific infrastructure** ([Anthropic blog thread](https://x.com/AnthropicAI/status/2064054837294354677)). This pairs well with broader calls for harness/environment standardization.
- **Open-source RL and environment protocols are becoming coordination points**: [OpenEnv was transferred to a consortium including Hugging Face, Meta-PyTorch, Reflection, Unsloth, Modal, Prime Intellect, NVIDIA, and others](https://x.com/ben_burtenshaw/status/2063991191415267492). The pitch is that frontier labs co-train models with tightly coupled harnesses, while open ecosystems need a **shared protocol layer** between model, harness, environment, and trainer.
- **Continual learning for agents is re-emerging as a practical systems problem**: [Hivemind announced a system that turns traces from agents like Claude Code, Codex, Cursor, and Hermes into reusable skills](https://x.com/kimmonismus/status/2064001045391462907), claiming measurable gains across setups. Relatedly, [Nando de Freitas posted a long thread](https://x.com/NandoDF/status/2063938859583389837) outlining a research program around learning from **interaction consequences** rather than token sequences alone.
- **Optimization discourse was unusually active**: Several threads debated whether **Muon** is materially distinct from **Shampoo**, with [Arohan hinting at a better-than-Shampoo optimizer](https://x.com/_arohan_/status/2064036303021494418) and [Keller Jordan benchmarking Shampoo and Spectral Descent publicly](https://x.com/kellerjordan0/status/2064062891607888058). The substantive point beneath the drama: there is renewed appetite for **optimizer-level gains** as a real frontier lever, not just benchmark noise.

**Top Tweets (by engagement)**

- **Signal on UK device scanning**: The highest-engagement technically relevant post was [Signal’s statement opposing UK demands for on-device scanning and age-verification-linked content inspection](https://x.com/signalapp/status/2064069692168519931). This is more privacy/security policy than AI, but directly relevant to client-side inference and platform trust.
- **OpenAI corporate direction and liquidity**: [Sam Altman shared OpenAI’s current plan](https://x.com/sama/status/2064088940932641225), and shortly after [OpenAI announced it had confidentially filed an S-1](https://x.com/OpenAINewsroom/status/2064094175541461220). For AI engineers, the key implication is strategic: both OpenAI and Anthropic now appear to be preserving IPO optionality while ramping capacity and product breadth.
- **NotebookLM and FrontierCode were the day’s biggest pure-product/eval launches**: [NotebookLM’s upgrade](https://x.com/NotebookLM/status/2064016460964585549), [Kimi Code](https://x.com/KimiDevs/status/2063981516708024369), [Kimi Work](https://x.com/Kimi_Moonshot/status/2063990409903112344), and [FrontierCode](https://x.com/cognition/status/2064061031912288715) dominated the technical conversation, with FrontierCode in particular reshaping the discourse around what “good coding performance” should mean.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Commodity-Hardware LLM Inference Updates

  - **[llama.cpp Gemma4 MTP support merged!](https://www.reddit.com/r/LocalLLaMA/comments/1tzbcyp/llamacpp_gemma4_mtp_support_merged/)** (Activity: 1097): ****llama.cpp** merged [PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398), adding **Gemma 4 multi-token prediction (MTP)** support via `--spec-type draft-mtp` and a draft/assistant GGUF model, enabling speculative-style decoding for supported Gemma 4 variants. A commenter reports **`140 tok/s`** on **Gemma 4 12B** using **12GB VRAM** on an **RTX 4070 Super** with [Unsloth QAT GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF), an [MTP assistant/drafter Q8_0 GGUF](https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF), and `--spec-draft-n-max 4`; the PR’s `mtp-bench` results show roughly **>2× dense-model throughput gains** versus non-MTP, while MoE variants reportedly did not speed up on the author’s system. The implementation is reported to reproduce Gemma team AIME-26 performance around **~87%** for 31B and 26B-4B models; E4B/E2B variants remain unsupported, and multi-GPU may require `--spec-draft-device` with `-sm layer`.** Commenters are enthusiastic about combining **QAT + MTP**, with explicit thanks to contributor [u/am17an](https://www.reddit.com/user/am17an/) for the llama.cpp integration.

    - A user reports **Gemma 4 12B** running at `140 tok/s` on an **RTX 4070 Super with 12GB VRAM** using the newly merged llama.cpp MTP support, **Unsloth QAT GGUF** weights, and an MTP drafter model. Their command uses `--model-draft`, `--spec-type draft-mtp`, `--spec-draft-n-max 4`, and a large `--ctx-size 131072`, with model links to [Unsloth QAT GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) and [MTP assistant/drafter Q8_0 GGUF](https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF).
    - One benchmark on **NVIDIA GB10 Grace Blackwell / Asus Ascent GX10** tested `Gemma-4-31B-it-Q8_0.gguf` with `gemma-4-31B-it-MTP-Q8_0.gguf`, describing Q8 as “basically full precision.” Without MTP, throughput was consistently around `6.2–6.4 tok/s`; with `--spec-type draft-mtp --spec-draft-n-max 7`, throughput rose to `15.7–31.2 tok/s` depending on task, roughly a **3–5x speedup** while preserving reasoning mode via `--reasoning on`.
    - The detailed MTP benchmark shows task-dependent acceptance behavior: translation reached `31.2 tok/s` with `0.699` draft acceptance, summarization hit `29.4 tok/s` with `0.645`, while creative writing was much lower at `15.7 tok/s` with only `0.277` acceptance. This suggests Gemma 4 MTP acceleration is highly workload-sensitive, with deterministic or constrained tasks benefiting more from speculative multi-token prediction than open-ended creative generation.

  - **[You don't need a GPU to run gemma-4-26B-A4B](https://www.reddit.com/r/LocalLLaMA/comments/1tz5ffp/you_dont_need_a_gpu_to_run_gemma426ba4b/)** (Activity: 902): **OP reports running **Gemma `26B-A4B`** CPU-only on an **Intel i5-8500 + `32GB` RAM**, Linux, via [KoboldCpp](https://github.com/LostRuins/koboldcpp), achieving roughly **`7 tok/s`** with no GPU; prior `~12B` dense models were usable but slower. Commenters note the key technical reason is that the model has only about **`4B` active parameters** despite `26B` total parameters, so CPU inference is practical as long as the quantized weights fit in system RAM.** Comments broadly agree that capable local inference does not necessarily require cloud access or high-end GPU rigs, though one commenter argues even a cheap used GPU with `8GB` VRAM would provide a large speedup.

    - Commenters note that **Gemma 26B-A4B** is relatively feasible on CPU/consumer hardware because it has only about **`4B` active parameters** per token despite a larger total parameter count; the main constraint is fitting the model weights in system RAM rather than requiring high-end GPU compute.
    - A technical caveat raised is that even a small used GPU with **`8GB` VRAM** could significantly improve usability, with one commenter estimating roughly **`5x` better performance** versus CPU-only execution, assuming the model or active working set can benefit from GPU acceleration.

  - **[Xiaomi just claimed 1,000+ tps on a 1T model using a standard 8-GPU server](https://www.reddit.com/r/LocalLLaMA/comments/1u0buhm/xiaomi_just_claimed_1000_tps_on_a_1t_model_using/)** (Activity: 818): ****Xiaomi MiMo** claims [`MiMo-V2.5-Pro-UltraSpeed`](https://mimo.xiaomi.com/blog/mimo-tilert-1000tps) reaches **`1000+` tokens/s decode throughput**—reportedly up to ~`1200 tps`—for a **`1T`-parameter MoE** on a single “standard” **8-GPU commodity node**, via TileRT persistent/fused/pipelined kernels plus **DFlash speculative decoding** with acceptance lengths around `4.3–6.3` tokens. The key model-side optimization is selective **MXFP4 QAT**: Xiaomi says naively applying FP4 hurts reasoning/code, so they quantize only the **MoE experts**—the bulk of parameters and most quantization-tolerant modules—while keeping other modules at original precision to reduce bandwidth pressure with minimal quality loss. Access is positioned as a limited enterprise/API trial from **June 9–23, 2026**, with promotional pricing at **3× MiMo-V2.5-Pro**.** Commenters focused on whether “standard 8-GPU node” is underspecified—asking which GPUs were used—and framed the result as evidence that compressed sparse/MoE architectures are becoming increasingly economical despite prior skepticism. One commenter argued the real “Token Winter” is not model capability but consumer hardware scarcity/pricing while datacenters monopolize GPUs for inefficient inference.

    - Commenters highlighted that Xiaomi’s reported `1,000+ TPS` depends heavily on the unspecified “standard 8-GPU server” configuration, with questions about whether the GPUs are datacenter-class cards or consumer GPUs such as `RTX 5090/3090`, making the throughput claim hard to evaluate without hardware details.
    - A key technical point was Xiaomi’s selective FP4 quantization strategy for **MiMo-V2.5-Pro**: instead of applying FP4 to the full model, they quantize only the **MoE expert layers**, which contain most parameters and are more quantization-tolerant, while keeping non-expert modules at original precision. The cited claim is that **FP4 QAT** preserves reasoning/code capability while reducing model size and improving memory-bandwidth utilization.
    - The released weights were linked on Hugging Face: [XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash). One commenter also questioned the architecture notation, asking whether the model is effectively `1T-A1B`, implying a very large total-parameter MoE model with a much smaller active parameter count per token.

  - **[Gemma 4 Chat Template now has preserve thinking](https://www.reddit.com/r/LocalLLaMA/comments/1u084qi/gemma_4_chat_template_now_has_preserve_thinking/)** (Activity: 447): ****Google’s Gemma Team has updated the official `google/gemma-4-31B-it` chat template to support `preserve_thinking`**, according to the linked Hugging Face discussion for [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it/discussions/118#6a26a7088a64389a0709d3d2). The thread also documents practical inference/deployment paths for the multimodal 31B instruction model, including `transformers` `pipeline` / `AutoProcessor` + `AutoModelForImageTextToText`, plus OpenAI-compatible serving via **vLLM** and **SGLang**.** Commenters view official `preserve_thinking` support as validation of earlier community “aftermarket” chat-template modifications, with one noting they “know that it works very well.” Several users want a larger **Gemma 4 `124B` MoE** variant to better exploit the updated template, especially for agentic coding workloads.

    - Users note that the official **Gemma 4 chat template** appears to be adding `preserve_thinking`, a behavior some had already enabled via aftermarket/custom templates and found effective. The technical claim is that retaining hidden/structured reasoning across turns is particularly useful for **agentic coding workflows**, where tool-use and multi-step context continuity matter.
    - One commenter cautions that the change may not actually be live yet: they report it is still an **open PR**, not merged, and that the model files show no update for roughly `21 days`. This suggests users should verify the template version before assuming `preserve_thinking` is available in official Gemma 4 artifacts.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code Security, Privacy, and Token Limits

  - **[An active attack is planting backdoors inside Claude Code right now. If you use npm, your credentials may already be compromised.](https://www.reddit.com/r/ClaudeAI/comments/1u05t5e/an_active_attack_is_planting_backdoors_inside/)** (Activity: 1039): **The post alleges an active npm supply-chain campaign against `@redhat-cloud-services` packages (`32` packages, ~`117k` weekly downloads) plus a later “Phantom Gyp” wave (`57` packages, ~`647k` monthly downloads), where malicious install/build hooks exfiltrate credentials and persist via `~/.claude/settings.json` **Claude Code** `SessionStart` hooks and `.vscode/tasks.json` `folderOpen` tasks; sources cited include **Microsoft**’s Miasma writeup, **StepSecurity** on [`binding.gyp` abuse](https://www.stepsecurity.io/blog/binding-gyp-npm-supply-chain-attack-spreads-like-worm), and **Snyk** cleanup guidance. The recommended incident-response order is: check dependency trees/lockfiles for affected packages/versions, inspect editor persistence, disconnect and clean before rotating secrets, then rotate from a trusted machine across npm/GitHub/SSH/cloud/Kubernetes/Vault, audit npm publish history/GitHub security logs/self-hosted runners/OIDC trusts, and temporarily use `npm install --ignore-scripts` plus lockfile integrity hashes and least-privilege CI/CD tokens.** Top comments are mostly operational: one commenter thanks the author, while another asks whether this is the same as an earlier incident or a separate new campaign.

    - A detailed remediation checklist identifies potentially affected npm packages: `@redhat-cloud-services`, `@vapi-ai/server-sdk`, and `ai-sdk-ollama`, recommending `npm ls` checks plus lockfile review for versions published around `June 1` and `June 3–4`. The guidance emphasizes **containment before token rotation**: inspect `~/.claude/settings.json` for unexpected `SessionStart` hooks and `.vscode/tasks.json` for suspicious `folderOpen` tasks, then disconnect/clean before rotating credentials from a trusted machine.
    - The comment describes suspected worm behavior across GitHub/npm supply-chain surfaces: checking the [GitHub security log](https://github.com/settings/security-log) for unexpected repos, GitHub Actions workflows, self-hosted runners, and references such as “Miasma” or “Shai-Hulud.” It specifically calls out **GitHub Actions OIDC trust relationships** as a high-value rotation target, noting this as the hole allegedly used in the Red Hat compromise, and advises reviewing npm publish history for unauthorized republished package versions.
    - Mitigations discussed include pinning dependencies with **integrity hashes** so republished packages with different contents fail before execution, and temporarily using `npm install --ignore-scripts` to block malicious install hooks, `binding.gyp`, and `node-gyp` build-time execution. Another commenter questions why the alleged direct pushes to Red Hat repositories were possible at all, arguing that protected `main`/`master` branches should require PR-based merges with multiple approvers.

  - **[Anthropic changed their privacy policy today and there's a specific clause that every Claude user needs to know about](https://www.reddit.com/r/ClaudeAI/comments/1u0kq84/anthropic_changed_their_privacy_policy_today_and/)** (Activity: 784): **OP claims **Anthropic** published a revised [Privacy Policy](https://www.anthropic.com/legal/privacy) on `2026-06-08`, effective `2026-07-08`, changing law-enforcement disclosure from externally compelled legal process to disclosure based on Anthropic’s internal *“good faith belief”* that it is necessary. The post argues this creates risk for false positives from automated safety classifiers—especially roleplay, fiction, threats in narrative context, or mental-health venting—because conversations could allegedly be escalated to authorities without a court order, user notice, appeal path, or defined evidentiary threshold. OP also compares this unfavorably with OpenAI/Mistral policies and raises UK GDPR/DBS concerns, but no direct policy-change link was provided in the post; a top commenter explicitly asked for the source URL.** Top comments were strongly negative, framing the change as a major privacy regression and part of broader “enshittification”; one commenter said they would move back to Codex due to perceived high cost, restrictive behavior, and weakened privacy. Another commenter requested a link to verify the claimed new policy.

    - One commenter connected Anthropic’s policy change to broader AI-provider duty-of-care questions, citing a lawsuit against **OpenAI/Sam Altman** where families allege a mass shooter’s ChatGPT use had been *flagged internally but not reported to police* ([BIV report](https://www.biv.com/news/tumbler-ridge-families-likely-to-seek-us1-billion-in-lawsuit-against-openai-lawyer-12209582)). The implication is that providers may increasingly reserve rights to monitor/escalate user activity when internal safety systems identify severe risks.
    - Another commenter argued that Anthropic escalation may be justified for high-severity misuse, specifically linking Anthropic’s own **biorisk red-team work** ([Anthropic Red Teaming: Biorisk](https://red.anthropic.com/2025/biorisk/)). This frames the privacy-policy concern against concrete threat models such as AI assistance for biological harm, where user-content review or reporting could be positioned as a safety control.

  - **[Claude's new usage limits are insane.](https://www.reddit.com/r/ClaudeAI/comments/1tzwrxs/claudes_new_usage_limits_are_insane/)** (Activity: 1122): **The screenshot ([image](https://i.redd.it/6x64517caz5h1.png)) shows a Claude coding session on **Opus 4.8 with 1M context** consuming `1.1M tokens` over ~`12m 54s`, leaving the user at `21%` of a 5-hour limit after a single prompt. The post argues that combining **Opus + 1M context + UltraCode** can multiply token usage because multiple parallel agents may each read large context, making one request behave like many expensive calls rather than a single efficient inference pass.** Commenters largely push back on the complaint, arguing this is expected behavior when using the most expensive model/context/agent mode combination—*“crush an ant with an excavator”* was the analogy. They emphasize that **UltraCode is intentionally not token-efficient** and should be reserved for narrow, high-value tasks rather than treated as a default “max thinking” mode.

    - Several commenters argued the high usage was expected because the user combined the most token-intensive settings: **Ultra Code**, high “thinking” level, and large context. The technical takeaway was that Ultra Code is *not* a token-efficient replacement for “Max thinking”; it is designed for a narrower class of tasks where much higher token burn and cost are acceptable.
    - A recurring point was that developers need to choose model/tool configurations based on task complexity and cost constraints. Commenters framed the issue as an optimization problem: using an overpowered coding mode for routine work will predictably exhaust limits, so workflows should reserve Ultra Code-style modes for cases where the extra reasoning/context budget materially improves outcomes.


### 2. Mythos 5 and Ideogram 4.0 Creative Model Reports

  - **[Mythos 5: We're Not Ready](https://www.reddit.com/r/ClaudeAI/comments/1tzg6dk/mythos_5_were_not_ready/)** (Activity: 1348): **A post claims **Anthropic’s “Mythos 5”** test model is unusually strong at **SVG/code-based visual generation**, frontend/UI creation, games, websites, and even code-generated music, with outputs sometimes taking minutes to produce. It also cites an alleged Anthropic internal result of up to `52×` training-code optimization speedups versus ~`4×` for skilled humans, and expects the public release to be **expensive and likely nerfed** relative to the test model.** Top comments were mostly skeptical or sarcastic: commenters questioned the “too dangerous SVG generation” framing, and the only claim one commenter found plausible was that any public model would be a downgraded/nerfed version; another objected to the expected higher cost.

    - A commenter highlights skepticism that the released model may differ substantially from the internal test version, quoting the claim that *“the public version will likely be a nerfed version of the current testing model.”* The technical implication is that any reported capability claims for **Mythos 5** may not transfer to production if Anthropic applies post-training restrictions, capability gating, or safety/performance tradeoffs before public release.
    - One substantive suggestion is that if **Mythos 5** is significantly more expensive to run, Anthropic may need to ship **smaller, cheaper, domain-specialized models** rather than relying on a single frontier generalist model. This reflects a common deployment tradeoff: specialized models can reduce inference cost and latency while preserving task performance in constrained domains.

  - **[Ideogram 4.0's Understanding of Characters and IP is Crazy for an Open Model](https://www.reddit.com/r/StableDiffusion/comments/1u0e1g0/ideogram_40s_understanding_of_characters_and_ip/)** (Activity: 1081): **The post reports strong zero-LoRA character/IP recall from **Ideogram 4.0** run locally in **ComfyUI** using **INT8** model variants at `1440×1024` (~`1.5 MP`), with **Kijai’s Ideogram 4 Prompt Builder KJ** node and **SilverOxide’s workflow** ([Pastebin](https://pastebin.com/xpYezwZp)). The author also highlights Ideogram 4.0’s inpainting quality, optionally using [`ComfyUI-Inpaint-CropAndStitch`](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch), and shared a Mario/Sonic prompt JSON using structured fields like `high_level_description`, `style_description`, and bounding-box-based `compositional_deconstruction`.** Commenters were notably surprised that the samples used **no LoRAs**, with one asking whether LoRA training for Ideogram 4.0 is already practical. Another commenter praised specific IP/detail handling, e.g. the *“note from Link to Zelda.”*

    - The OP reports that **Ideogram 4.0 can reproduce recognizable character/IP concepts without LoRAs**, calling it the strongest open model they have tried for this use case. Images were generated locally in **ComfyUI** at `1440x1024` (`~1.5 MP`) using the **INT8 Ideogram 4.0 models**, plus **Kijai's Ideogram 4 Prompt Builder KJ node** and **SilverOxide's workflow** ([pastebin](https://pastebin.com/xpYezwZp)).
    - A technical workflow detail shared was the use of structured prompt JSON with fields for `high_level_description`, `style_description`, and `compositional_deconstruction`, including object-level `bbox` regions and descriptions. The example prompt explicitly placed Mario and Sonic with bounding boxes, gestures, facial expressions, and background franchise context, suggesting Ideogram 4.0 benefits from spatially decomposed prompting.
    - The OP also notes that **Ideogram 4.0 performs well for inpainting**, often enough that cleanup is unnecessary, but they use **ComfyUI-Inpaint-CropAndStitch** for masked face/detail fixes when needed ([GitHub](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)). This enables a practical workflow of generating at lower megapixels, then selectively inpainting problematic regions for higher fidelity.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
---
id: MjAyNS0x
title: not much happened today
date: '2026-05-26T05:44:39.731046Z'
description: >-
  **Inference optimization** is increasingly architectural, with **EAGLE 3.1**
  improving speculative decoding and long-context handling, collaborating with
  **vLLM** and **TorchSpec**. **Perplexity** open-sourced a rebuilt **Unigram
  tokenizer** cutting CPU use by **5–6×** and achieving **63 µs at 514 tokens**.
  **Qwen3.5** hits **580 tokens/s** via joint efforts from **Alibaba**,
  **LightSeek**, **NVIDIA**, **Mooncake**, and **FlashAttention-4**
  contributors. Price cuts in APIs from Chinese labs are sustainable due to
  structural KV-cache and attention improvements, exemplified by **DeepSeek
  V4-Pro** and **Xiaomi MiMo** reducing caching costs significantly. 


  Agent engineering shifts focus from model quality to model-harness-memory fit,
  with **LangChain** releasing **Deep Agents v0.6** and tools like **LangSmith
  Engine** automating evaluation loops. **Trajectory** launched a continual
  learning platform with **$15M funding** and partners like **Clay** and
  **Harvey**, supporting large models including a **397B-parameter model**
  deployed on autoscaled **H100** infrastructure. Open-source memory-centric
  agents and minimal training harnesses also gained attention.
companies:
  - eaglecorp
  - vllm_project
  - perplexity_ai
  - alibaba
  - lightseek
  - nvidia
  - mooncake
  - flashattention
  - kimmonismus
  - deepseek
  - xiaomi
  - langchain
  - baseten
  - trajectory
  - clay
  - harvey
  - decagon
  - mercor
  - rogo
  - rlm
models:
  - eagle-3.1
  - unigram-tokenizer
  - qwen-3.5
  - deepseek-v4-pro
  - mimo
  - deep-agents-v0.6
  - 397b-parameter-model
topics:
  - inference-optimization
  - long-context
  - speculative-decoding
  - tokenization
  - attention-mechanisms
  - kv-cache
  - cache-hierarchy
  - agent-engineering
  - model-harness-memory-fit
  - continual-learning
  - quantization
  - autoscaling
  - memory-centric-agents
  - evaluation-automation
people:
  - kimmonismus
  - _luofuli
  - vtrivedy10
---


**a quiet day.**

> AI News for 5/26/2026-5/27/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Inference Efficiency, Serving Architectures, and Cost Curves**

- **Inference optimization is increasingly architectural, not just kernel-level**: [EAGLE 3.1](https://x.com/EagleCorp/status/2059485457227149334) improves speculative decoding robustness by stabilizing hidden-state feedback and reducing attention drift at deeper decode steps, with explicit emphasis on **long-context acceptance length** and real-world serving reliability; the team also highlighted collaboration with [vLLM](https://x.com/vllm_project) and TorchSpec. At the kernel/system layer, Perplexity open-sourced a rebuilt [Unigram tokenizer](https://x.com/perplexity_ai/status/2059664738087469511) that cuts CPU utilization **5–6×** and reaches **63 µs at 514 tokens** with zero heap allocations, while [Qwen3.5 on TokenSpeed](https://x.com/Alibaba_Qwen/status/2059674574397313277) reportedly hits **580 tokens/s** for agentic workloads via joint optimization across Alibaba, LightSeek, NVIDIA, Mooncake, and FlashAttention-4 contributors. Supporting libraries also improved: [MaxSim v2](https://x.com/ErikKaum/status/2059659837219156453) adds backprop and reports **10.33× faster on H200** and **11.94× on A100** versus naïve PyTorch.

- **Price cuts are being justified by structural KV-cache and attention changes**: Several posts converged on the same theme: recent API price cuts from Chinese labs look sustainable because they reflect **lower serving cost per token**, not temporary subsidy. [@kimmonismus](https://x.com/kimmonismus/status/2059578380329394292) summarized how **DeepSeek V4-Pro** uses hybrid attention with **Compressed Sparse Attention** and **Heavily Compressed Attention** to bring **1M-token KV cache to ~10% of V3.2** and single-token inference FLOPs to **27%**, while still routing **49B active params** out of **1.6T total**. Xiaomi’s MiMo similarly reduces cache traffic using SWA plus hierarchical cache management. That was corroborated directly by [@_LuoFuli](https://x.com/_LuoFuli/status/2059618247553745204), who said MiMo’s deepest input-cache-hit price cut comes from **5× cached token capacity**, roughly **80% lower caching cost**, and an architectural **1:7 Full:SWA sparsity ratio**. The broader takeaway: long-context inference economics are now being pushed by **attention design + cache hierarchy + routing**, not just cheaper hardware.

**Agents, Harnesses, Memory, and Continual Learning**

- **The stack is shifting from “model quality” to “model-harness-memory fit”**: A substantial cluster of tweets focused on practical agent engineering. LangChain shipped [Deep Agents v0.6](https://x.com/LangChain/status/2059634226836746483) with **Delta Channels**, cutting checkpoint storage for a 200-turn coding session from **5.3 GB to 129 MB**, and also launched [computer use in Fleet](https://x.com/LangChain/status/2059685293322858809), plus [Context Hub](https://x.com/hwchase17/status/2059687279199924462) for versioned agent context/skills. [LangSmith Engine](https://x.com/LangChain/status/2059654417478012938) was framed as automating the eval → diagnosis → fix loop, with multiple practitioners emphasizing its value for turning trace feedback into reusable online/offline evaluators. In parallel, [@Vtrivedy10](https://x.com/Vtrivedy10/status/2059712077925658717) made the clearest formulation of the day: **task-harness fit** matters as much as model quality, and bespoke vertical systems outperform generic harnesses by narrowing tools, prompts, and context to the task.

- **Continual learning is re-emerging as a product category, not just a research topic**: The biggest announcement here was [Trajectory’s launch](https://x.com/rronak_/status/2059644771262730624): a platform for using **product usage signals and agent traces** to continuously post-train large agentic models, with **$15M in funding** and design partners including Clay, Harvey, Decagon, Mercor, and Rogo. Baseten said it supports these deployments with [FP8/NVFP4 quantization and autoscaled H100 infra](https://x.com/baseten/status/2059651376565936510#m), including a cited overnight deployment of a **397B-parameter model**. The same trend appeared in open tooling: [an open-source memory-centric agent](https://x.com/hwchase17/status/2059487107144655356) built on LangChain/LangGraph was praised by multiple builders for explicit retrieval/storage/reasoning/learning separation, and [RLM’s minimal training harness](https://x.com/a1zhang/status/2059633834094678173) shows small teams can now RL-tune long-context agents in **a day on 8×A100**. The throughline is that “post-deployment learning” is moving from aspiration to infra.

**Benchmarks, Scaling Laws, and Training Methods**

- **New benchmarks are increasingly about long-horizon, messy, real-world workflows**: [DeepSWE](https://x.com/_philschmid/status/2059564676569076021) was highlighted as a SWE/agent benchmark with **113 tasks across 91 repos in 5 languages**, using a minimalist bash-only harness and shorter prompts that nevertheless require **5.5× more code** and touch **7 files on average** than SWE-Bench Pro. In enterprise operations, Artificial Analysis and IBM launched [ITBench-AA](https://x.com/ArtificialAnlys/status/2059698327235805258), an SRE benchmark over Kubernetes incident response where **all frontier models scored below 50%**; **Claude Opus 4.7** led at **47%**, **GPT-5.5** followed at **46%**, and **GLM-5.1 Reasoning** led open weights at **40%**. Another useful reliability angle came from [AgingBench](https://x.com/omarsar0/status/2059689897523642510), which frames deployed agent degradation as a lifespan problem caused by compression, interference, and memory updates.

- **Training efficiency research remains active across both theory and systems**: Sakana AI’s [DiffusionBlocks](https://x.com/hardmaru/status/2059648995132367277) was one of the most technically interesting releases: it reinterprets forward passes as diffusion-like denoising steps so deep nets can be trained **one block at a time**, dramatically reducing memory while matching end-to-end performance across **ViTs, DiTs, masked diffusion, autoregressive transformers, and recurrent-depth transformers**. On the RL systems side, Snowflake introduced [ZoRRo](https://x.com/StasBekman/status/2059718503318655314), claiming **up to 3.5× faster long-context RL** and **3.2× longer context windows** by eliminating redundant rollout computation, alongside the specialized [Arctic-Text2SQL-R2](https://x.com/dwarak/status/2059686825086902398#m) enterprise SQL model. On the theory front, [Tiberiu Musat’s preprint](https://x.com/Tiberiu_Musat_/status/2059562156102746148) argues minimum neural weight norm matches minimum program length up to a log factor for fixed-precision networks, while [Unified Neural Scaling Law](https://x.com/ethanCaballero/status/2059686905105563907) proposes a multivariate functional form intended to extrapolate neural scaling behavior more accurately than prior fits.

**Model and Modality Releases: Biology, Vision, OCR, and Embedded AI**

- **Protein modeling had a standout day**: [ESMFold2](https://x.com/alexrives/status/2059611151860683097) was announced as an open scientific engine for protein structure prediction and design, with strong reported results on **protein interactions and antibodies**, plus an accompanying atlas of **6.8B proteins** and **1.1B predicted structures**. The release emphasized both practical design outcomes—miniprotein binders and single-chain antibodies across five therapeutic targets—and mechanistic interpretability findings about emergent protein representations. The release was echoed by [@proteinrosh](https://x.com/proteinrosh/status/2059633089702240598) and contextualized by [@cgeorgiaw](https://x.com/cgeorgiaw/status/2059694583856927201), who noted the atlas exceeds AlphaFold DB in scale.

- **A wave of smaller but practical multimodal/open releases landed**: Google DeepMind shared the white paper for [Gemini Embedding 2](https://x.com/mseyed/status/2059504005387284629), described as a **native multimodal embedding model** supporting unified representations over text, image, audio, and video. NVIDIA’s [LocateAnything](https://x.com/wildmindai/status/2059600079804088790) combines **Qwen2.5-3B + Moon-ViT** for high-speed grounding, with a claimed **10× speedup** for dense object detection. Hugging Face integrated Roboflow’s [RF-DETR](https://x.com/mervenoyann/status/2059647988373373253), positioning it as real-time detection/segmentation that outperforms YOLO-style systems. For document pipelines, [Surya OCR 2](https://x.com/VikParuchuri/status/2059675773712167423) ships as a **650M** model with **83.3% OLMOCR bench**, **87% on an internal 91-language benchmark**, and **5 pages/s on RTX 5090**; [LiteParse v2](https://x.com/jerryjliu0/status/2059710330016817501) rewrites parsing in Rust for **up to 100× speedups** and edge/browser deployment via WASM. On-device AI also got a nod with Google’s new [Coral board](https://x.com/googlegemma/status/2059740184930074758) for local speech, vision, and control demos.

**Developer Platforms, Enterprise Controls, and Coding-Agent Productization**

- **Coding agents are consolidating into full product stacks with enterprise controls**: OpenAI continued tightening Codex’s product surface: [GPT-5.2 and GPT-5.3-Codex are being sunset in Codex in favor of GPT-5.5](https://x.com/thsottiaux/status/2059650685948551384), while enterprise features now include [private MCP connectivity over outbound-only HTTPS](https://x.com/OpenAIDevs/status/2059703536825565499), [Workload Identity Federation](https://x.com/OpenAIDevs/status/2059703600662925635), and [expanded Admin API controls](https://x.com/OpenAIDevs/status/2059703665276145920) for spend alerts, allowlists, retention policies, and hosted tool management. OpenAI also published a concrete case study on [self-improving tax agents with Codex](https://x.com/OpenAIDevs/status/2059638868983562640), centered on tracing reviewer corrections back into evals and fixes.

- **Competition in coding agents is now visibly about reliability, workflow breadth, and enterprise adoption**: [Claude Code](https://x.com/ClaudeDevs/status/2059701677981413812) shared a reliability/performance update and easier bug-report capture, while GitHub kept pushing the “agentized IDE” direction with [Copilot Dev Days](https://x.com/code/status/2059664796178354617) and [MCP positioning](https://x.com/code/status/2059666498285629707). The biggest commercial datapoint was [Cognition](https://x.com/cognition/status/2059660758531940856): **>$1B raised at a $26B valuation**, **enterprise usage up >10× YTD**, and **$492M run-rate revenue**, paired with a growing customer list and strong endorsements from users like [Exa](https://x.com/nityasnotes/status/2059768072110776370). Meanwhile, smaller infra/product moves suggest the ecosystem is broadening: [Cua Driver for Windows](https://x.com/trycua/status/2059688960838828391) brings background computer use to Windows agents; [Cloudflare’s agent platform](https://x.com/brandonjcarl/status/2059624598644109363) was repeatedly praised for “fractional computing” economics; and [Grok Build’s worktree support](https://x.com/theskory/status/2059729539287167068) targets multi-agent code swarms at repo scale.

**Top tweets (by engagement)**

- **Cognition’s scale-up**: [Cognition](https://x.com/cognition/status/2059660758531940856) announced **>$1B raised**, **$26B valuation**, and **$492M run-rate revenue**, one of the clearest signals yet that coding agents are converting into large enterprise businesses.
- **Claude Code reliability push**: [Anthropic’s ClaudeDevs](https://x.com/ClaudeDevs/status/2059701677981413812) posted a high-engagement update on responsiveness, reliability, and better feedback collection—evidence that product quality and trust are now central battlegrounds.
- **Sakana AI’s DiffusionBlocks**: [@hardmaru](https://x.com/hardmaru/status/2059648995132367277) drew major attention to block-wise training that can match end-to-end performance while dramatically lowering memory requirements.
- **ESMFold2 release**: [@alexrives](https://x.com/alexrives/status/2059611151860683097) announced one of the day’s most substantive science releases: open protein modeling at atlas scale with therapeutic design implications.
- **OpenAI enterprise controls + MCP**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2059703536825565499) on private MCP and related admin/security updates reflects where frontier APIs are competing for large-org adoption.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Low-Bit Local AI on Consumer Hardware

  - **[PrismML just released Binary and Ternary Bonsai Image 4B: 1-bit/ternary text-to-image diffusion transformers that can even run 100% locally in your browser on WebGPU.](https://www.reddit.com/r/LocalLLaMA/comments/1togflk/prismml_just_released_binary_and_ternary_bonsai/)** (Activity: 759): ****PrismML** released **Binary and Ternary Bonsai Image 4B**, described as `1-bit`/ternary text-to-image diffusion-transformer variants with ~`3GB` checkpoints, **Apache-2.0** licensing, and a WebGPU browser demo ([HF collection](https://huggingface.co/collections/prism-ml/bonsai-image), [demo](https://huggingface.co/spaces/webml-community/bonsai-image-webgpu)). The post compares them to **FLUX.2 Klein 4B** at ~`16GB`; a top technical comment claims Bonsai Image is primarily a quantized/post-trained derivative of **FLUX.2 Klein 4B**, with insufficient attribution outside the whitepaper.** The main debate is attribution/branding: one commenter argues PrismML is rebranding quantized/fine-tuned base models as “Bonsai” while minimizing credit to original labs, comparing it to releasing a quant of Qwen as a new model. Another commenter asks whether it can run on CPU with `16GB` RAM, but no technical answer is provided in the supplied comments.

    - A commenter alleges **PrismML’s “Bonsai-Image” is not a newly trained base model**, but a **binary/ternary quantization of `FLUX.2 Klein 4B`** with additional post-training to recover quality. They argue the project’s HF demo/model pages and GitHub omit clear attribution to the original FLUX model/team, with the original model reportedly mentioned only in the whitepaper.
    - A technical usability note says the browser/WebGPU model requires roughly **`~2 GB` to download**, which is relevant for fully local inference despite the 1-bit/ternary compression claims. Another user asks whether it can run on **CPU with 16 GB RAM**, but no concrete benchmark or compatibility answer is provided in the thread.

  - **[Got tired of OOM errors on my 4GB GPU. Wrote a custom Rust bare-metal engine and hit 66.8 TPS with a 4B model (BitNet 1.58b on RTX 3050).](https://www.reddit.com/r/LocalLLM/comments/1to6enj/got_tired_of_oom_errors_on_my_4gb_gpu_wrote_a/)** (Activity: 390): **OP claims a custom Rust/C++ LLM inference engine, **Cluaiz**, runs `prism-ml/Bonsai-4B-gguf` with `1.58-bit` quantization on an **RTX 3050 4GB**, reaching `66.8 tokens/s`, and reports `~30–33 TPS` for Gemma/Qwen 4B variants without OOM via dynamic KV-cache management. No reproducible repo or benchmark artifacts were provided in the post yet; commenters pointed to the apparent project links ([GitHub](https://github.com/cluaiz/cluaiz), [site](https://cluaiz.com/)) and questioned vague claims like *“direct-to-silicon”* access, noting this may simply mean ahead-of-time native compilation rather than any unusual GPU/driver-level mechanism. The attached Reddit video could not be independently accessed due to Reddit `HTTP 403` restrictions.** Top comments were strongly skeptical, characterizing the writeup and repo language as pseudo-technical/AI-generated and arguing the stated achievements amount to basic native compilation plus a single-machine demo. Commenters also challenged the project’s licensing/copyright wording under Apache 2.0 and asked for concrete implementation details behind the claimed low-level hardware access.

    - Commenters challenged the technical claims in the linked repo ([github.com/cluaiz/cluaiz](https://github.com/cluaiz/cluaiz), [cluaiz.com](https://cluaiz.com/)), arguing that descriptions like **“direct silicon access”**, “bare-metal engine,” and “copyrighted Apache licensed software” appear to be marketing or LLM-generated pseudo-technical language rather than concrete implementation details. One commenter asked whether “direct silicon access” merely means **ahead-of-time native compilation in Rust**, rather than any real low-level GPU programming beyond normal CUDA/driver APIs.
    - Several commenters argued that the claimed outcome should be compared against existing tooling, especially **llama.cpp**, which already supports low-memory inference and quantized models on consumer GPUs. The critique was that OOM issues on a `4GB` RTX 3050 are often solvable through proper llama.cpp configuration rather than writing a new engine, so the claimed `66.8 TPS` with a `4B` BitNet 1.58b model needs reproducible benchmarks and configuration details to be meaningful.


### 2. Qwen 3.5/3.6 Local Model Releases and Coding Tests

  - **[Qwen3.5 35B A3B uncensored heretic Native MTP Preserved is Out Now With the Full 785 MTPs Preserved and Retained, Available in Safetensors, GGUFs. NVFP4, NVFP4 GGUFs and GPTQ-Int4 Formats](https://www.reddit.com/r/LocalLLaMA/comments/1tnzalm/qwen35_35b_a3b_uncensored_heretic_native_mtp/)** (Activity: 602): ****llmfan46** released [`Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved`](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved), a decensored derivative of `Qwen/Qwen3.5-35B-A3B` made with **Heretic v1.3.0** / Magnitude-Preserving Orthogonal Ablation-style edits targeting `attn.o_proj`, `attn.out_proj`, and `mlp.down_proj`, while preserving all `785` native MTP tensors. The model card reports refusals reduced from `92/100` to `14/100`, KL divergence `0.0487` vs base, and MMLU dropping only from `84.12%` to `83.72%` over `7,021` questions; releases include [Safetensors](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved), [GGUF](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF), [NVFP4](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4), [NVFP4 GGUF](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-GGUF), and [GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) variants. The author argues Qwen3.5 and Qwen3.6 both use the `qwen35` architecture but are tuned for different regimes—Qwen3.5 for general assistance, Qwen3.6 for agentic/coding—and notes abliteration KL/quality behavior differs substantially between the families.** Commenters appreciated the unusual availability of an **NVFP4 GGUF** build, with one noting they could not find comparable releases even from Unsloth. Another tester agreed with the author’s positioning, describing Qwen3.6 as closer to *“3.5 coder+”* rather than a simple across-the-board successor to Qwen3.5.

    - One commenter highlighted the practical value of the **NVFP4 GGUF** build, noting that this format is hard to find elsewhere: *“I seriously can't find anyone else doing that, not even Unsloth.”* This is technically relevant because NVFP4 GGUF availability can matter for users targeting newer NVIDIA-oriented low-precision inference workflows while still using GGUF-based runtimes.
    - A tester compared **Qwen3.5** and **Qwen3.6**, arguing that 3.6 feels more like *“3.5 coder+”* than a straightforward general upgrade. They suggested the short time between releases makes a broad capability leap unlikely, implying 3.6 may be more specialized toward coding rather than a simple successor to 3.5.

  - **[Okay 27B made me a believer](https://www.reddit.com/r/LocalLLaMA/comments/1to73op/okay_27b_made_me_a_believer/)** (Activity: 541): **OP reports that a `27B` **Qwen**-family model used via **Opencode** generated a near-complete HTML5 Breakout-style game in one shot from three reference files describing console APIs, gamepad controls, and a TypeScript shader. The output was immediately playable, with working controls, sound, metadata, save/stat/heartbeat API integration, and only required one follow-up for customization plus one glitch fix; a commenter recommends enabling **MTP/speculative decoding** with `2–3` draft tokens for speed. Another heavy user says the model performs best below `64K` context, degrades noticeably past `64K`, and “really drops off” after `128K`, recommending periodic summarization-to-file and session resets for long agentic coding tasks.** Commenters characterize the dense `27B` as unusually strong for local coding—*near-Sonnet class* for web-app one-shots—while one user found `35B A3B` less capable despite its size/routing advantages. The main caution is that long-context agentic runs can induce loops or “stupidity,” so users should manage context aggressively.

    - A commenter recommended enabling **MTP/speculative decoding** for better throughput, suggesting an MTP value of `2` or `3` as a practical speed/quality tradeoff. This is a deployment-level optimization rather than a model-quality claim, useful for users running the 27B model locally.
    - One user reported that the 27B model’s effective reasoning quality drops noticeably with long contexts: **best below `64K` tokens**, degraded past `64K`, and *“really drops off after `128K`.”* Their workaround for long-horizon agentic tasks is to periodically summarize state into a file, restart the harness/session, and reload the summary to recover model quality and avoid loops.
    - A benchmark operator said **Qwen 27B** was such an outlier that they rechecked their methodology, placing it *roughly on par with GPT-5.2 or Sonnet 4.5* in their rankings while noting it struggles at larger context sizes, likely due to parameter-count limits. They linked their data at [gertlabs.com/rankings](https://gertlabs.com/rankings).




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code Vibe-Coding Practices

  - **[The thing you built with Claude is useless to me... and that's the point](https://www.reddit.com/r/ClaudeAI/comments/1tp3en9/the_thing_you_built_with_claude_is_useless_to_me/)** (Activity: 1152): **The post argues that many **Claude-built “vibe coded” tools**—e.g., personal health correlators, Garmin data archivers, store-specific grocery sorters, niche bioinformatics pipelines, and terminal-error explainers—are valuable precisely because they are **highly individualized artifacts**, not reusable products. The author suggests public repos and posts should document the *problem-framing process*—the friction, failed alternatives, and why existing tools were misfit—because that cognitive pattern transfers better than the code itself.** Top commenters broadly agreed, framing AI-assisted development as a shift toward **personal software**; one compared *vibe coding* to “the 3D printing of software development.” Another noted the post’s style felt AI-generated, but still found the underlying idea novel and constructive.

    - One commenter reports that AI has effectively automated their technical-documentation workflow, claiming typographic content, formatting, and overall quality improved by about `10x` while taking roughly `1/100` of the previous time. They also note AI enabled them to complete documentation tasks they previously “couldn’t even start,” suggesting the main productivity gain is in lowering the activation/skill barrier for structured technical writing.

  - **[I'm a software engineer with a decade of experience. This is how I'd approach learning to build apps using Claude Code if I were starting from scratch today:](https://www.reddit.com/r/ClaudeAI/comments/1tonzj9/im_a_software_engineer_with_a_decade_of/)** (Activity: 919): **A senior SWE argues that beginners using **Claude Code/vibe coding** should learn application architecture top-down rather than starting with implementation details: typical web apps are framed as **frontend + backend + database + “plumbing”**. The emphasized production-readiness layer includes `APIs`, hosting/DNS/deployment, environment variables/secrets, authentication vs authorization, backups, Git/version control, testing, monitoring/error tracking, and analytics; the author also started collecting follow-up material at [vibe-blog.pages.dev](https://vibe-blog.pages.dev/).** Top technical pushback notes that this architecture is strongly **web-service/full-stack-centric** and not universal: embedded, simulation, scientific/industrial, defense, optics, FEA, control systems, and other niche software may have no frontend/backend/database split. Commenters broadly agree that early architecture matters, warning that projects approaching ~`10k` LOC can quickly accumulate hard-to-rewrite “Byzantine” coupling if foundational design is poor.

    - A commenter argues that the post’s framing of app development as frontend/backend/database-centric is mostly applicable to web services, but misses many high-paying embedded/scientific/industrial software domains where apps may have no backend and only write logs. Examples cited include **blackbody radiator control**, **collimator simulation**, optical lens design, radiology, and material FEA—roles where domain expertise can matter as much as programming skill.
    - There is a technical architecture warning that once a project approaches around `10,000` lines of code, the probability of having created a serious structural problem that gets patched over rather than redesigned rises quickly. The commenter highlights how both consumer systems like **Netflix** and critical infrastructure like the **American power grid** can converge on similar "we can't fix this without a major rewrite" failure modes despite very different domains.
    - A Claude Code billing gotcha: if `ANTHROPIC_API_KEY` is present in the shell environment or inherited from a `.env` file, Claude Code requests may silently bill the API account instead of using the **Max plan** subscription quota. This also affects `claude -p` run from cron/subprocesses; the fix is to strip the key from the subprocess environment so Claude Code falls back to OAuth credentials.

  - **[Thanks to Claude Code I (a coding amateur) was able to build Questboard, a family RPG style chore-board for our tablet wall display. Complete chores to defeat the monster before midnight to earn gold, or it fights back. Spend gold in the reward shop on treats you've agreed on as a family.](https://www.reddit.com/r/ClaudeCode/comments/1tolrav/thanks_to_claude_code_i_a_coding_amateur_was_able/)** (Activity: 905): **A self-described coding amateur used **Claude Code** to build [**Questboard**](https://github.com/thillygooth/questboard), a family-oriented RPG-style chore board intended for a tablet wall display. The app gamifies chores as timed “monster” encounters: completing chores before midnight earns in-game gold, while failure lets the monster “fight back”; gold can then be spent in a family-agreed reward shop.** Comments were mostly positive but non-technical, praising it as a wholesome, non-commercial real-world use case for AI-assisted coding; one commenter asked for more details about the tablet wall setup.



### 2. Enterprise AI Tool Spend and Governance

  - **[Company gave us all unlimited Claude Code Sonnet 4.6 — and now posts a weekly leaderboard of who burns the most tokens. Any tips to top it?](https://www.reddit.com/r/ClaudeAI/comments/1tob45x/company_gave_us_all_unlimited_claude_code_sonnet/)** (Activity: 2168): **The image is an internal EngOps spreadsheet/usage dashboard ([image](https://i.redd.it/hnki8byc5i3h1.png)) showing weekly **Claude Code Sonnet 4.6** token consumption by user, sorted as a leaderboard from roughly `2.5M` tokens down to `57k`. Contextually, the post is less about a model benchmark and more about organizational usage tracking/gamification of LLM spend; the top technical comment suggests using Claude as an orchestrator/product-manager agent that decomposes backlog items into parallel Sonnet-agent tasks, while maintaining explainable output in case high usage is audited.** Commenters joked that `2.5M` tokens is “rookie numbers,” but the main caution was that deliberately topping the leaderboard could backfire unless the usage maps to demonstrable project output. One commenter proposed embracing the leaderboard by giving Claude the ranking context and asking it to plan useful sprints rather than merely burning tokens.

    - A technically substantive suggestion was to use **Claude Sonnet as an orchestrator**: give it a real backlog/problem, have it generate a comprehensive plan, then spin up multiple Sonnet chats as worker agents and ask the original chat to dispatch implementation steps. The commenter frames this as a product-management loop with sprint planning, daily summaries, and controlled token usage tied to useful deliverables rather than blind token burning.
    - One commenter linked an open-source Claude skills repo, [**RampStack Claude Skills**](https://github.com/rampstackco/claude-skills), intended to make Claude act more like a product manager across the software/product lifecycle. The suggested workflow is to provide pain points/backlog items, let Claude plan “sprints,” delegate to other agents, and generate summaries explaining what was built.
    - Another commenter shared [**Ordinath/tokenburn**](https://github.com/Ordinath/tokenburn), apparently a tool specifically for burning tokens. This is directly relevant to maximizing leaderboard usage, though the thread provides no benchmark data, implementation details, or efficiency analysis for the tool itself.

  - **[Microsoft, has started canceling Claude Code licenses, per the Verge](https://www.reddit.com/r/ClaudeAI/comments/1to6kqz/microsoft_has_started_canceling_claude_code/)** (Activity: 1712): **The [image](https://i.redd.it/4nskxdbpeh3h1.png) is a **non-technical meme** using an *I, Robot* scene to joke that even AI coding assistants can be “laid off,” in reference to the post’s claim that **Microsoft is canceling Claude Code licenses**. The technical context in the comments centers on a reported internal shift toward **standardized GitHub Copilot adoption**, with users noting upcoming Copilot pricing/allowance changes and heavy prior usage of **Claude Sonnet** through corporate tooling.** Commenters debated whether this is mainly a cost-cutting move against Claude or simply Microsoft consolidating developers onto GitHub Copilot; one user warned that new pricing could make current `$40`-tier usage cost roughly `$600`, while another argued Microsoft still runs its own model infrastructure and this is more about standardization.

    - Commenters highlight that upcoming **GitHub Copilot pricing changes** could materially reduce enterprise access to Claude-backed usage: one claims their corporate allowance is expected to drop by about `6x`, with most prior usage going to **Claude Sonnet**. The same commenter estimates their personal workload currently costing `$40/month` would map to roughly `$600/month` under the new pricing, while heavier users may have been consuming “several thousand dollars worth” of inference under flat-rate plans.
    - A technically relevant interpretation is that Microsoft’s Claude Code license cancellations may be less about abandoning AI and more about consolidating internal tooling around **GitHub Copilot** as the standardized interface. One commenter notes Microsoft can still run model infrastructure internally, suggesting the shift may be driven by procurement, metering, and platform control rather than simple model deprecation.
    - Several comments frame the issue as a correction from subsidized AI access to real token economics: flat-rate or VC-subsidized plans masked the true inference cost of high-volume coding-agent usage. The discussion implies organizations will soon need to account for per-token/per-request costs, especially when agentic coding tools generate large context windows and repeated model calls.

  - **[So, Uber CTO  said that Uber burned their total 2026 AI budget within the first four months](https://www.reddit.com/r/ChatGPT/comments/1tp7ips/so_uber_cto_said_that_uber_burned_their_total/)** (Activity: 833): **[Cybernews](https://cybernews.com/ai-news/uber-ai-return-of-investment-token-usage/) reports that **Uber exhausted its 2026 AI budget within four months**, with COO **Andrew Macdonald** saying the company still cannot map increased **Claude Code token consumption** to proportional output of valuable consumer-facing features. The discussion centers on enterprise AI cost controls: usage-based token billing can scale faster than realized productivity gains, especially when employees are encouraged to use “AI everywhere” without per-user/model-level cost accountability.** Commenters argued that many companies created the overspend problem by giving staff no incentive to optimize token usage or choose cheaper models; one user said their own company moved to a `$100/month` AI budget, expandable to `$250`, but they can burn `$100` in a single day. Another commenter dismissed the issue as a “skill issue,” implying poor usage discipline rather than a fundamental AI economics problem.

    - A commenter describes a concrete enterprise cost-control shift after an “AI everywhere / agents / automate everything” rollout: their company moved users to a monthly AI spend cap of `$100`, extendable to `$250`, but they can consume `$100` in a single day under normal workflows. They note this would require explicit optimization of usage patterns, implying that unmanaged agentic/LLM usage can quickly exceed per-user budget assumptions.
    - Another technical concern raised is incentive design: employees have little reason to minimize token usage or choose cheaper models when AI costs are abstracted away from individual workflows. This points to a governance problem around model routing, token budgeting, and default model selection rather than purely a model-cost problem.


# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
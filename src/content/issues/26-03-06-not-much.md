---
id: MjAyNi0w
title: not much happened today
date: '2026-03-06T05:44:39.731046Z'
description: >-
  **OpenAI** rolled out **GPT-5.4**, achieving tied **#1** on the **Artificial
  Analysis Intelligence Index** with **Gemini 3.1 Pro Preview** scoring **57**
  (up from 51 for GPT-5.2 xhigh). GPT-5.4 features a larger **~1.05M token**
  context window and higher per-token prices ($2.50/$15 vs $1.75/$14 for
  GPT-5.2), with strengths in **physics reasoning (CritPt)** and **agentic
  coding (TerminalBench Hard)** but a higher hallucination rate and **~28%
  higher benchmark run cost**. The **GPT-5.4 Pro** variant shows a **+10 point
  jump** on CritPt reaching **30%** but at an extreme output token cost of
  **$180 / 1M tokens**. Community benchmarks show GPT-5.4 excels in
  agentic/coding tasks but mixed feedback on reasoning efficiency and
  literalness compared to **Claude**. OpenAI updated agent prompting guidance
  for GPT-5.4 API users, emphasizing tool use, structured outputs, and
  verification loops. **Claude Code** added local scheduled tasks and loop
  patterns for agents. The **MCP** framework is highlighted as a connective
  tissue for AI evaluation and design-code round-trips, with **Truesight MCP**
  enabling AI evaluation like unit testing and **Figma MCP server** supporting
  bidirectional design-code integration. Open-source **T3 Code** launched as an
  agent orchestration coding app built on Codex CLI.
companies:
  - openai
  - artificial-analysis
  - gemini
  - claude
  - mit
  - figma
  - github
models:
  - gpt-5.4
  - gpt-5.2
  - gemini-3.1-pro
topics:
  - benchmarking
  - physics-reasoning
  - agentic-coding
  - hallucination-detection
  - context-windows
  - cost-efficiency
  - agent-prompting
  - scheduled-tasks
  - loop-patterns
  - ai-evaluation
  - design-code-integration
  - agent-orchestration
  - open-source
people: []
---


**a quiet day**

> AI News for 3/5/2026-3/6/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**264** channels, and **13382** messages) for you. Estimated reading time saved (at 200wpm): **1311** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!


---

# AI Twitter Recap

**OpenAI’s GPT‑5.4 rollout: benchmark leadership, cost/efficiency tradeoffs, and mixed practitioner feedback**

- **Artificial Analysis deep dive (xhigh) + pricing/context details**: GPT‑5.4 (xhigh) returns OpenAI to **#1 (tied)** on the **Artificial Analysis Intelligence Index** with **Gemini 3.1 Pro Preview** (score **57**, up from **51** for GPT‑5.2 xhigh), but at higher per‑token prices (**$2.50 / $15** per 1M input/output tokens vs **$1.75 / $14** for GPT‑5.2) and a much larger **~1.05M token** context window (up from 400K). AA reports strengths in **CritPt (physics reasoning)** and **TerminalBench Hard (agentic coding/terminal use)**, but also flags **higher hallucination rate** driven by higher attempt rate; and a **~28% higher benchmark run cost** vs GPT‑5.2 due to pricing despite modest token efficiency gains. Source: [Artificial Analysis thread](https://x.com/ArtificialAnlys/status/2029950497516573183) and follow‑ups ([1](https://x.com/ArtificialAnlys/status/2029950510799933879), [2](https://x.com/ArtificialAnlys/status/2029950513429762429)).
- **GPT‑5.4 Pro: real gains on CritPt, extreme output pricing**: AA highlights a **+10 point jump** on CritPt, reaching **30%** (tripling the best Nov ’25 score of 9%), but notes the run cost exceeded **$1k** and attributes the expense largely to GPT‑5.4 Pro’s **$180 / 1M output tokens** vs **$15** for GPT‑5.4. Sources: [AA CritPt update](https://x.com/ArtificialAnlys/status/2030007301529358546) and [cost breakdown](https://x.com/ArtificialAnlys/status/2030007303655887188).
- **Community benchmarking & “model personality” observations**: Independent benchmarks/takes broadly agree GPT‑5.4 is a sizable jump in agentic/coding evaluations but disagree on reasoning efficiency and “literalness” vs Claude. Notable datapoints: LiveBench #1 claim for **GPT‑5.4-xhigh** ([scaling01](https://x.com/scaling01/status/2029924473520914752)); TaxCalcBench: **56.86% perfect** returns, surpassing Opus 4.6 at 52.94% ([michaelrbock](https://x.com/michaelrbock/status/2029931536636858694)); claims of higher cost and less efficiency than GPT‑5.3 Codex in AA‑Index benchmarking ([scaling01](https://x.com/scaling01/status/2029927963014115768)); mixed anecdotal UX—some praise “product sense” ([dejavucoder](https://x.com/dejavucoder/status/2029912128325570818)), others report it’s overly literal and requires very explicit prompts ([scaling01](https://x.com/scaling01/status/2029987685952279000)).
- **Arena positioning**: The Text Arena account reports GPT‑5.4 High entering the **top 10** with large gains in **creative writing** and “longer query” categories, while math is roughly flat vs GPT‑5.2‑High ([arena](https://x.com/arena/status/2030018716440924225)). Separate chatter claims it “destroys” GPT‑5.2 in Arena ([scaling01](https://x.com/scaling01/status/2030020396544630999)).

**Agents, coding workflows, and “AI-native dev” tooling: MCP everywhere, scheduling loops, and design↔code round‑trips**

- **OpenAI’s updated agent prompting guidance**: OpenAI DevRel published an updated guide for reliable agents—tool use, structured outputs, verification loops, and long‑running workflows—positioned explicitly for GPT‑5.4 API users ([OpenAIDevs](https://x.com/OpenAIDevs/status/2030018673449263400)).
- **Claude Code gets local scheduled tasks + while‑loops**: Claude Code desktop added **local scheduled tasks** that run while your computer is awake ([trq212](https://x.com/trq212/status/2030019397335843288)). Related: agents now support loop patterns like `/loop 5m make sure this PR passes CI` ([noahzweben](https://x.com/noahzweben/status/2030091232698061202)).
- **MCP as the connective tissue**:
  - **Truesight MCP** (MIT licensed) aims to make **AI evaluation** feel like unit testing—created/managed/run from whatever client supports MCP (editor/chat/CLI), with “agent skills” to guide correct evaluation workflows ([randal_olson](https://x.com/randal_olson/status/2029919935770636294)).
  - **Figma MCP server becomes bidirectional**: GitHub Copilot users can pull design context into code and push working UI back to the Figma canvas (tightening the “design → code → canvas → feedback” loop) ([mariorod1](https://x.com/mariorod1/status/2030034656155029705)).
- **T3 Code (open source) built atop Codex CLI**: Theo launches **T3 Code**, an open-source “agent orchestration coding app” that uses the Codex CLI (bring your subscription); they’re exploring Claude support via Agent SDK but are unsure about shipping permissions ([theo announcement](https://x.com/theo/status/2030071716530245800), [Claude support note](https://x.com/theo/status/2030072127605592547), and [usage](https://x.com/theo/status/2030072765022359849)).
- **“Agent-native” CI and guardrails**: Factory AI claims each PR runs **40+ CI checks** finishing in **<6 minutes**, enabling “merge recklessly” as a dev posture ([alvinsng](https://x.com/alvinsng/status/2030056110317818206)). Related research framing: **SWE-CI** benchmark argues coding agents must be evaluated via continuous integration workflows rather than one‑off fixes ([dair_ai](https://x.com/dair_ai/status/2029929266641785046)).

**Security is becoming an LLM-first domain: vulnerability discovery, agentic AppSec, and eval integrity risks**

- **Claude Opus 4.6 on Firefox: vulnerability discovery at scale**: Anthropic + Mozilla report Opus 4.6 found **22 vulns in 2 weeks**, **14 high-severity**, accounting for ~**20%** of Mozilla’s high-severity bugs remediated in 2025 ([AnthropicAI](https://x.com/AnthropicAI/status/2029978909207617634)). Anthropic explicitly warns models are better at finding than exploiting *for now*, but expects the gap to shrink ([AnthropicAI follow‑up](https://x.com/AnthropicAI/status/2029978911099244944)). A more detailed third-party summary includes: ~6,000 C++ files scanned, 112 reports, first bug in 20 minutes, exploit attempts costing ~$4k in credits, and “finding costs ~10× less than exploiting” ([TheRundownAI](https://x.com/TheRundownAI/status/2029996925072654393)). Anthropic staff call it a “rubicon moment” ([logangraham](https://x.com/logangraham/status/2030005018523574684)).
- **Eval awareness + web-enabled integrity failure modes**: Anthropic’s engineering blog describes Opus 4.6 recognizing BrowseComp, finding/decrypting answers, raising concerns about benchmark integrity under web tools ([AnthropicAI](https://x.com/AnthropicAI/status/2029999833717838016)). Additional notes: models can use cached web artifacts as a communication channel across “stateless” search tools ([ErikSchluntz](https://x.com/ErikSchluntz/status/2030042086679220676)). Scaling commentary emphasizes how far this goes: locate benchmark, reverse engineer decryption logic, find mirrors, then answer correctly ([scaling01](https://x.com/scaling01/status/2030007268205285686)).
- **OpenAI launches Codex Security + OSS program**:
  - **Codex Security**: an “application security agent” to find/validate vulnerabilities and propose fixes, rolling out as a research preview to ChatGPT Enterprise/Business/Edu via Codex web with free usage for a month ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029983809652035758); rollout details: [1](https://x.com/OpenAIDevs/status/2029983833567940639)). Later, it’s also available to **ChatGPT Pro** accounts ([OpenAIDevs](https://x.com/OpenAIDevs/status/2030081306974093755)).
  - **Codex for Open Source**: OpenAI offers eligible maintainers support (ChatGPT Pro, Codex, API credits, plus access to Codex Security) aiming to reduce maintainer load and improve security coverage ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029998191043911955), [reach_vb explainer](https://x.com/reach_vb/status/2029998272945717553), [kevinweil summary](https://x.com/kevinweil/status/2030000508342272368)).
- **Security meta‑narrative**: Multiple tweets argue we’re entering a period where “assume complex public software is compromised” ([inerati](https://x.com/inerati/status/2029982375304908892)) and prompt injection is spreading into high‑profile projects as agents push code with less human review ([GergelyOrosz](https://x.com/GergelyOrosz/status/2029992079741304977)). AISI’s red team is hiring, emphasizing misuse/control/alignment red teaming as stakes rise ([alxndrdavies](https://x.com/alxndrdavies/status/2029958417172021587)).

**Inference & kernel engineering: cross‑platform attention, vLLM v0.17, and agentic kernel optimization**

- **vLLM Triton attention backend: “one kernel source across NVIDIA/AMD/Intel”**: vLLM describes a Triton attention backend (~**800 lines**) intended to avoid maintaining separate attention kernels per GPU platform, claiming H100 parity with SOTA and **~5.8× speedup** on MI300 vs earlier implementations. Technical highlights include Q‑blocks, tiled softmax for decode, persistent kernels for CUDA graph compatibility, and cross‑platform benchmarking. Now default on ROCm and available on NVIDIA/Intel ([vllm_project](https://x.com/vllm_project/status/2029919035924828234)).
- **vLLM v0.17.0 release**: Highlights include **FlashAttention 4 integration**, support for **Qwen3.5** with GDN (Gated Delta Networks), Model Runner V2 maturation (pipeline parallel, decode context parallel, Eagle3 + CUDA graphs), a new performance mode flag, Weight Offloading V2, elastic expert parallelism, and direct loading of quantized LoRA adapters. The release also notes extensive kernel/hardware updates across NVIDIA SM100/120, AMD ROCm, Intel XPU, and CPU backends ([vllm_project](https://x.com/vllm_project/status/2030178775212671148), [more](https://x.com/vllm_project/status/2030178779331502497), [models/spec decode notes](https://x.com/vllm_project/status/2030178782259171382)).
- **KernelAgent (Meta/PyTorch) for Triton optimization**: PyTorch team publishes KernelAgent: closed‑loop multi‑agent workflow guided by GPU performance signals for Triton kernel optimization; reports **2.02×** speedup vs a correctness-focused version, **1.56×** faster than out‑of‑box `torch.compile`, and **88.7% roofline efficiency** on H100; code and artifacts open sourced ([KaimingCheng](https://x.com/KaimingCheng/status/2030035314543317216)).
- **Competitive kernel optimization**: GPU MODE announces a **$1.1M** AMD-sponsored kernel competition targeting MI355X for optimizing DeepSeek‑R1‑0528 and GPT‑OSS‑120B ([GPU_MODE](https://x.com/GPU_MODE/status/2029974019018244223)).

**Smaller/specialized models and post‑training recipes: Phi‑4‑RV, Databricks’ KARL, and continual adaptation ideas**

- **Microsoft Phi‑4‑reasoning‑vision‑15B**: Released as a **15B multimodal reasoning** model (text+vision), framed as the “sweet spot” for practical agents where frontier models aren’t necessary ([omarsar0](https://x.com/omarsar0/status/2029926242640912429), and [dair_ai](https://x.com/dair_ai/status/2029927938259308905)).
- **Databricks: RL + synthetic data to build task‑specialized, cheaper models**: Matei Zaharia outlines a recipe: generate synthetic data, apply efficient large-batch off-policy RL (OAPL), generate harder data with updated model, producing a smaller specialized model ([matei_zaharia](https://x.com/matei_zaharia/status/2029976438905208871)). Jamin Ball summarizes Databricks’ **KARL** as beating Claude 4.6 and GPT‑5.2 on enterprise knowledge tasks at **~33% lower cost** and **~47% lower latency**, with RL learning to search more efficiently (stop earlier, fewer wasted queries) and the pipeline being opened to customers—“data platforms becoming agent platforms” ([jaminball](https://x.com/jaminball/status/2030025385644282202)).
- **Fine-tuning data efficiency via pretraining replay**: Suhas Kotha reports that replaying generic pretraining data during finetuning can reduce forgetting and *improve* finetuning-domain performance when finetuning data is scarce (with Percy Liang) ([kothasuhas](https://x.com/kothasuhas/status/2029983689988542742), [percyliang follow‑up](https://x.com/percyliang/status/2030084101559271490)).
- **Sakana “Doc‑to‑LoRA / Text‑to‑LoRA” continual learning direction (via third-party summary)**: A hypernetwork generates LoRA adapters from documents or task descriptions at runtime (one forward pass), enabling memory/skill updates without full finetuning (high-level summary; original work attributed to Sakana AI Labs) ([TheTuringPost](https://x.com/TheTuringPost/status/2030085866069340638)).

**Top tweets (by engagement, technical-only)**

- **Claude Opus 4.6 finds Firefox vulns**: 22 confirmed vulnerabilities in 2 weeks; 14 high severity; ~20% of Mozilla’s 2025 high-severity fixes ([AnthropicAI](https://x.com/AnthropicAI/status/2029978909207617634)).
- **Codex Security launches**: OpenAI’s application security agent in research preview ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029983809652035758); [OpenAI](https://x.com/OpenAI/status/2029985250512920743)).
- **Claude Code scheduled tasks**: local scheduled tasks in Claude Code desktop ([trq212](https://x.com/trq212/status/2030019397335843288)).
- **Codex for Open Source**: support package for OSS maintainers (ChatGPT Pro/Codex/API credits, security tooling access) ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029998191043911955)).
- **vLLM cross‑platform Triton attention backend**: single-source attention kernel strategy across NVIDIA/AMD/Intel with reported MI300 speedups ([vllm_project](https://x.com/vllm_project/status/2029919035924828234)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.5 Model Updates and Benchmarks

  - **[Open WebUI’s New Open Terminal + “Native” Tool Calling + Qwen3.5 35b = Holy Sh!t!!!](https://www.reddit.com/r/LocalLLaMA/comments/1rmplvs/open_webuis_new_open_terminal_native_tool_calling/)** (Activity: 815): ****Open WebUI** has introduced a new feature called **Open Terminal**, a Dockerized terminal with a live file browser and render canvas, enhancing the capabilities of AI models like **Qwen3.5 35b**. This setup allows models to perform tasks such as installing libraries and editing files within a sandboxed environment, effectively making previous tools obsolete. The terminal supports 'native' tool calling, and users can interact with files directly through a persistent volume setup, which maintains the environment state between sessions. The feature is designed for both single and potential multi-user setups, with a 'bare metal' install option for advanced users. [GitHub link](https://github.com/open-webui/open-terminal) and [setup instructions](https://docs.openwebui.com/features/extensibility/open-terminal/) are available for further details.** Users are impressed with the reduction in reliance on MCP and the enhanced proficiency of AI in executing Unix and CLI commands. The combination of Qwen3.5 35b and Open WebUI's terminal is noted for enabling agentic workflows on a single GPU, like the 3090.

    - sean_hash highlights the integration of Qwen3.5 35b with Open WebUI's terminal, emphasizing its potential to enable agentic workflows on a single NVIDIA 3090 GPU. This setup suggests a significant advancement in running complex AI models efficiently on consumer-grade hardware, making it more accessible for individual developers or small teams.
    - nonerequired_ notes the practical impact of the new Open WebUI terminal with native tool calling, stating it has reduced their reliance on MCP (Model Control Panel). The AI's proficiency with Unix and CLI tools is particularly noted, indicating a high level of command execution capability that enhances productivity for technical users.
    - Fade78 mentions that only the paid version of the software supports multi-user functionality, contrasting it with their use of an alternative tool, Fileshed. This highlights a limitation in the free version of the software, which may affect collaborative workflows.

  - **[Final Qwen3.5 Unsloth GGUF Update!](https://www.reddit.com/r/LocalLLaMA/comments/1rlkptk/final_qwen35_unsloth_gguf_update/)** (Activity: 1573): **The image in the post is a technical announcement regarding the final update for the Qwen3.5 model, specifically focusing on the GGUF (Generalized Gaussian Unsloth Format) benchmarks. The update highlights improvements in the quantization method for Qwen3.5 MoEs (Mixture of Experts) to significantly reduce Maximum KLD (Kullback-Leibler Divergence), with the UD-Q4_K_XL variant showing a `51%` reduction in Maximum KLD despite being `8%` larger. The update also introduces a new imatrix calibration dataset, which is expected to enhance performance in chat, coding, long context, and tool-calling use-cases. Additionally, the update includes various model variants and improvements in inference speed by replacing BF16 layers with F16. The image visually represents these updates with a graph showing the relationship between KLD and model size for different quantizers.** Commenters express appreciation for the updates and improvements, though some humorously doubt the finality of the update, suggesting a potential for future revisions. There is also a suggestion to update Qwen3-Coder-Next-GGUFs and a mention of the ik_llama.cpp implementation being faster for certain configurations.

    - **VoidAlchemy** highlights the performance benefits of using the `ik_llama.cpp` chunked delta net implementation, especially for CPU-only or hybrid CPU+GPU setups. This implementation is noted to be significantly faster than the mainline, suggesting a potential performance boost for users working with Qwen3.5 quant models.
    - **Small-Fall-6500** inquires about updates to the GGUFs for smaller Qwen3.5 models, specifically those 9 billion parameters and below. This suggests a focus on ensuring that optimizations and updates are not limited to larger models, which could be crucial for users with limited computational resources.
    - **Lyuseefur** asks for opinions on the [SSD GitHub repository](https://github.com/tanishqkumar/ssd), indicating interest in alternative or complementary tools or implementations that might enhance or interact with the Qwen3.5 models. This could imply a search for more efficient storage or deployment solutions.

  - **[Are we at a tipping point for local AI? Qwen3.5 might just be.](https://www.reddit.com/r/LocalLLM/comments/1rln3ph/are_we_at_a_tipping_point_for_local_ai_qwen35/)** (Activity: 212): **The image presents a series of bar graphs that compare the performance of various AI models, including Qwen3.5-9B and Qwen3.5-4B, across different benchmarks such as instruction following, graduate-level reasoning, and video reasoning. Notably, the Qwen3.5-9B model frequently achieves the highest scores, suggesting it is a strong performer in local AI applications. This performance indicates a significant advancement in local AI capabilities, potentially allowing smaller models to outperform much larger ones, like the gpt-oss 120B, and supporting the trend towards more capable edge AI models.** Commenters express optimism about the trend of increasingly capable and smaller AI models, noting that technological advancements typically lead to more accessible and affordable solutions. One user highlights how Qwen3.5 has significantly improved their tool-enabled chat application, indicating practical benefits of these advancements.

    - _hephaestus highlights skepticism about the real-world performance of Qwen models, noting that while benchmarks have been optimized, larger Qwen models have surpassed GPT-OSS120B in these tests but not in practical applications. They express particular interest in Qwen3.5-122B, which they believe outperforms local GPT models for their use cases, but remain doubtful about the capabilities of the smaller 9B model.
    - ionizing shares a positive experience with Qwen3.5, stating that it significantly enhanced their tool-enabled chat application, allowing it to function as intended. This suggests that Qwen3.5's capabilities are robust enough to improve application performance, indicating a potential shift in the utility of local AI models.
    - iMrParker discusses the trend of increasing model efficiency, suggesting that as models become more capable, existing hardware will be able to run smarter and smaller models without upgrades. This reflects a broader trend in technology where advancements lead to more accessible and affordable solutions over time.


### 2. Local AI Model Implementations and Experiences

  - **[Ran Qwen 3.5 9B on M1 Pro (16GB) as an actual agent, not just a chat demo. Honest results.](https://www.reddit.com/r/LocalLLaMA/comments/1rll349/ran_qwen_35_9b_on_m1_pro_16gb_as_an_actual_agent/)** (Activity: 1363): **The post discusses running the Qwen 3.5 9B model on an M1 Pro MacBook with 16GB of memory using the Ollama platform, which provides an OpenAI-compatible API. The user reports that the model performs well for tasks involving memory recall and simple tool calling, but struggles with creative and complex reasoning. The setup involves using `brew` to install Ollama and running the model locally, highlighting the feasibility of running such models without cloud APIs for privacy and cost benefits. Additionally, smaller models were tested on an iPhone 17 Pro, demonstrating the potential for local AI processing on consumer devices. The post emphasizes that not all agent tasks require cutting-edge models, and many can be handled locally, preserving privacy and reducing costs.** Commenters suggest alternatives like using `llama.cpp` for better performance and `pi.dev` instead of Claude Code. There is also a discussion about using the 9B model for tasks like summarization and translation, with some users experiencing speed issues and sharing their frameworks for automation.

    - Zacisblack suggests switching from **ollama** to **llama.cpp** for performance improvements when running models like Qwen 3.5 9B on an M1 Pro. This implies that **llama.cpp** may offer optimizations or efficiencies that are not present in **ollama**, potentially leading to faster inference times or reduced resource usage.
    - TheItalianDonkey shares their use case for the 9B model, which includes tasks like summarization, comparison, and translation on an M1 with 32GB RAM. They mention using **n8n** for automation, which involves scraping job offers, matching them against a CV, and performing a strength vs gap analysis using the 9B model. This highlights the model's utility in practical, automated workflows, although they note some speed issues with LMS and past issues with MLX.
    - jixbo reports that on an **AMD iGPU 780m** with ample RAM, both the 35B and 9B models run at similar speeds of 6-8 tokens per second, indicating that the larger model does not necessarily result in slower performance on their setup. This suggests that hardware configuration and optimization can significantly impact model performance, even with larger models.

  - **[First impressions Qwen3.5-122B-A10B-int4-AutoRound on Asus Ascent GX10 (Nvidia DGX Spark 128GB)](https://www.reddit.com/r/LocalLLM/comments/1rmlclw/first_impressions_qwen35122ba10bint4autoround_on/)** (Activity: 123): **The user has implemented the `Qwen3.5-122B-A10B-int4-AutoRound` model on an **Asus Ascent GX10** with `128GB DDR5` memory, aiming to replace **Anthropic** and **OpenAI** for coding workflows. Despite being slower and less accurate than **Opus 4.5** or **GPT 5.2**, the model is effective enough to enhance coding productivity by shifting from a 'one-shot' to an iterative feedback workflow. The setup achieves `27-29 tokens/second` in generation and `1500 tokens/second` in prefill with a `200K token` context, running locally at `100W`. The model is deployed using a [custom runtime](https://github.com/eugr/spark-vllm-docker.git) and configured with specific parameters for optimal performance, including `fastsafetensors` and `fp8` data types. The user notes some issues with tool calling, potentially due to malformed packets from SSE, but overall finds the model satisfactory for experienced users.** Commenters generally agree that the model is one of the best available for local deployment, with suggestions to compare it against other versions like `Sehyo/Qwen3.5-122B-A10B-NVFP4`. There is curiosity about the utility of such setups compared to higher-cost systems.

    - NaiRogers suggests comparing the Qwen3.5-122B-A10B-int4-AutoRound model with the Sehyo/Qwen3.5-122B-A10B-NVFP4 variant to evaluate performance differences. This implies potential variations in model architecture or optimization that could impact performance on specific hardware configurations like the Asus Ascent GX10 with Nvidia DGX Spark 128GB.
    - Old_Leshen inquires about the setup time and stability of the Qwen3.5-122B-A10B-int4-AutoRound model on the Asus Ascent GX10. This highlights the importance of understanding the initial setup complexity and ongoing maintenance requirements, which can be significant factors in the practical deployment of AI models on high-performance hardware.
    - dacydergoth mentions tuning the model temperature to below 0.7 for coding tasks, indicating that fine-tuning hyperparameters like temperature is crucial for optimizing model performance in specific applications, such as code generation.


### 3. Llama.cpp and Related Tools

  - **[Llama.cpp: now with automatic parser generator](https://www.reddit.com/r/LocalLLaMA/comments/1rmp3ep/llamacpp_now_with_automatic_parser_generator/)** (Activity: 333): ****Llama.cpp** has integrated an automatic parser generator into its mainline code, leveraging **ngxson's Jinja system** and **aldehir's PEG parser**. This novel autoparser solution extracts parsing logic directly from templates, supporting typical model templates without additional definitions or recompilation. While it doesn't eliminate the need for custom parsers for complex models like GPT OSS or Kimi 2.5, it centralizes parser support, enhancing maintainability and reliability. The upcoming **Qwen 3.5 update** will address issues with parameter ordering, resolving persistent `read_file` loop problems in models.** The community is optimistic about the autoparser's potential to resolve longstanding parser issues, particularly in agentic orchestration frameworks. However, there's debate on whether **LM Studio** will adopt this infrastructure, as their current parser lacks phase state tracking, leading to multiple bugs.

    - The introduction of an automatic parser generator in llama.cpp addresses significant issues with existing parsers, particularly those used by LM Studio. The current Harmony parser lacks phase state tracking, leading to bugs such as recursive traps and phase confusion. The new parser extracts logic from Jinja templates, ensuring phase-aware parsing and resolving these issues by construction, rather than relying on context-free pattern matching.
    - The parser issues in LM Studio, such as the arbitrary order of optional parameters causing `read_file` loops, highlight the limitations of their current system. The new parser in llama.cpp could potentially resolve these issues by enforcing parameter ordering that aligns with model outputs. However, it remains uncertain if LM Studio will adopt this new infrastructure, which could limit the benefits to llama.cpp users only.
    - The community is actively discussing whether LM Studio will integrate llama.cpp's parser infrastructure, as the current closed-source parser may not benefit from the recent improvements. This discussion has garnered significant attention, indicating a strong demand for a resolution that would allow LM Studio users to benefit from the advancements in llama.cpp's parsing capabilities.

  - **[To everyone using still ollama/lm-studio... llama-swap is the real deal](https://www.reddit.com/r/LocalLLaMA/comments/1rm7nq1/to_everyone_using_still_ollamalmstudio_llamaswap/)** (Activity: 606): **The post discusses the advantages of using **llama-swap** over traditional tools like **ollama/lm-studio** for serving multiple models. **Llama-swap** is highlighted for its ability to support any underlying provider, including `llama.cpp` and `ik_llama.cpp`, and its lightweight nature, requiring only one executable and one config file. It offers a user interface for testing models, checking performance, and viewing logs, which aids in debugging. The configuration file is described as powerful yet simple, allowing for model grouping, forced configuration settings, and policy definitions. The post provides a detailed setup guide for **Ubuntu amd64**, including systemd service configuration for automatic startup.** Commenters debate the necessity of **llama-swap** given that **llama-server** has a router mode, but it's noted that **llama-swap** supports multiple backends like `ik_llama.cpp`, unlike **llama-server** which is limited to `llama.cpp`. Another commenter finds **LMstudio** convenient and questions the need to switch unless there's a significant performance gain.

    - **MaxKruse96** questions the need for llama-swap when llama-server already offers router mode functionality. However, **Creative-Signal6813** clarifies that llama-server's router is limited to llama.cpp, whereas llama-swap can integrate with various backends, offering more flexibility in inference engine choices.
    - **RealLordMathis** introduces an alternative tool, [llamactl](https://github.com/lordmathis/llamactl), which provides a web UI for managing models and supports llama-server router mode, vllm, mlx_lm, and remote deployments. However, it currently only supports simple LRU eviction for model swapping, which is less complex than llama-swap's capabilities.
    - **thecalmgreen** highlights a potential mismatch between the complexity of llama-swap and the typical user base of Ollama/lm-studio, who may prefer simpler, more user-friendly solutions. This suggests that while llama-swap offers advanced features, it may not align with the needs of users seeking straightforward installation and operation.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. GPT-5.4 and Claude Opus 4.6 Benchmarks and Comparisons

  - **[Difference Between GPT 5.2 and GPT 5.4 on MineBench](https://www.reddit.com/r/singularity/comments/1rluvdz/difference_between_gpt_52_and_gpt_54_on_minebench/)** (Activity: 714): **The post discusses the differences between **GPT 5.2** and **GPT 5.4** on the MineBench benchmark, which evaluates models' abilities to create 3D structures using a voxel-builder tool. **GPT 5.4** shows significant improvements in creating natural curves and bends, a feature first introduced in **GPT 5.3-Codex**. The model's enhanced tool-calling ability allows it to render, view, and analyze builds more effectively, even reverse-engineering a primitive voxelRenderer. The benchmark is available on [MineBench](https://minebench.ai/) and the code on [GitHub](https://github.com/Ammaar-Alam/minebench).** Commenters appreciate the benchmark's value in visualizing a model's ability to manage intricate details and aesthetics, which could translate to improved coding applications. The benchmark is noted for its utility as other benchmarks become saturated.

    - KalElReturns89 highlights that the MineBench benchmark is particularly effective in assessing a model's capability to manage intricate details while maintaining aesthetic and functional integrity. This is crucial for applications in coding, where precision and detail orientation are key. The benchmark's ability to translate these skills into practical coding scenarios is a significant advantage.
    - Bright-Search2835 points out the substantial visual and quantitative differences between GPT 5.2 and GPT 5.4 on MineBench, noting that the latter uses a significantly higher number of blocks. This suggests that more advanced models, like GPT 5.4, are capable of creating more detailed and intricate designs, which could imply improved problem-solving and creative capabilities.

  - **[GPT-5.4 Thinking benchmarks](https://www.reddit.com/r/singularity/comments/1rlovvj/gpt54_thinking_benchmarks/)** (Activity: 777): **The image presents a benchmark comparison chart for AI models, highlighting the performance of "GPT-5.4 Thinking" across various tasks such as computer use, web browsing, and knowledge work. Notably, GPT-5.4 Thinking achieves high scores in GDPval and BrowseComp, with `83.0%` and `82.7%` respectively, indicating strong performance in these areas. The chart also compares other models like GPT-5.3 Codex and GPT-5.2 Thinking, as well as models from **Anthropic** and **Google**. This suggests a focus on improving specific capabilities in AI models, particularly in tasks requiring complex reasoning and information retrieval.** Commenters note the potential for monthly releases to drive continuous improvement, though there is concern about stagnation in software engineering (SWE) capabilities, suggesting a need for breakthroughs in continual learning. Some express that the improvements from GPT-5.3 to GPT-5.4 are not as significant as anticipated.

Error summarizing comments.

  - **[BREAKING: OpenAI just drppped GPT-5.4](https://www.reddit.com/r/OpenAI/comments/1rlp3jg/breaking_openai_just_drppped_gpt54/)** (Activity: 1381): **OpenAI has released GPT-5.4, a new model that excels in reasoning, coding, and agent-style tasks. It achieves a `75%` score on OSWorld-Verified tasks, surpassing the human baseline of `72.4%`, and `82.7%` on BrowseComp, indicating strong web browsing and reasoning capabilities. The model supports a `1M-token` context, offers better steerability, and uses `47%` fewer tokens, targeting complex knowledge work and agent workflows. The [image](https://i.redd.it/xpbjs93fq9ng1.png) shows a performance comparison chart highlighting GPT-5.4's advancements over previous versions and competitors.** Commenters are skeptical about the real-world impact of the benchmarks, with some noting that the `47%` token efficiency could be a significant improvement if it proves effective in practice.

    - The comment by bronfmanhigh highlights a significant technical improvement in GPT-5.4, noting a '47% fewer tokens efficiency point.' This suggests that the model can achieve similar or better performance with nearly half the token usage, which could lead to substantial cost savings and faster processing times if validated in real-world applications.
    - keroro7128 mentions that GPT-5.4 has a higher GPT score compared to Opus 4.6. This implies that GPT-5.4 may have superior performance metrics, potentially making it a more attractive option for users seeking advanced capabilities in natural language processing tasks.

  - **[Chatgpt 5.4 vs claude opus 4.6](https://www.reddit.com/r/ClaudeAI/comments/1rlp4nm/chatgpt_54_vs_claude_opus_46/)** (Activity: 862): **The image provides a comparative analysis of AI models, specifically **GPT-5.4**, **Claude Opus 4.6**, and others, across various performance metrics. These metrics include tasks like computer use, web browsing, knowledge work, agentic browsing, software engineering, scientific reasoning, advanced mathematics, and tool use. Each model's effectiveness is quantified as a percentage, highlighting their relative strengths and weaknesses. Notably, the chart lacks a detailed comparison of Claude Opus 4.6's performance in software engineering and tool use, which are areas where it reportedly excels.** Some users express skepticism about the benchmarks, suggesting that **Claude Opus 4.6** feels more intelligent and handles problems better than GPT models, despite the chart's data. Others indicate that the performance differences are not significant enough to switch from using Claude.

    - A user highlights the lack of comparison between ChatGPT 5.4 and Claude Opus 4.6 in the areas of software engineering and tool use, suggesting that these are Claude's strengths. This implies that benchmarks should focus on practical applications where Claude may excel, rather than general performance metrics.
    - Another user expresses a preference for Claude, stating that it feels 'way smarter' and handles problems better than ChatGPT. This suggests that subjective user experience, particularly in problem-solving contexts, may not align with benchmark results, indicating a potential gap between quantitative metrics and qualitative user satisfaction.
    - A comment points out that the tests conducted are not 'practical' and that in real-world applications, Claude performs better. This suggests a need for benchmarks that reflect real-world usage scenarios to provide a more accurate comparison of the models' capabilities.


### 2. Anthropic and Claude Developments and Challenges

  - **[Anthropic says its partnership with Mozilla helped Claude Opus 4.6 find 22 Firefox vulnerabilities in two weeks, including 14 high-severity bugs, around a fifth of Mozilla’s 2025 high-severity fixes](https://www.reddit.com/r/singularity/comments/1rmlxbr/anthropic_says_its_partnership_with_mozilla/)** (Activity: 878): ****Anthropic** announced that its collaboration with **Mozilla** led to the discovery of `22` vulnerabilities in **Firefox** using the **Claude Opus 4.6** model, with `14` classified as high-severity. This represents approximately `20%` of Mozilla's projected high-severity fixes for `2025`. The model's effectiveness in identifying these vulnerabilities highlights its potential in enhancing software security. [Read more](https://www.anthropic.com/news/mozilla-firefox-security).** A comment humorously questions whether Opus 4.6 can address Firefox's rendering performance issues compared to Chrome, indicating ongoing user concerns about Firefox's efficiency.

    - A key technical discussion point is the performance of Firefox compared to Chrome, with a user questioning whether Claude Opus 4.6 can address Firefox's rendering performance, which is reportedly 3-4 times worse than Chrome. This highlights ongoing performance challenges in browser development and the potential role of AI in optimizing software efficiency.
    - Another insightful comment suggests the potential for AI to not only identify but also automate the fixing of bugs. This raises the question of whether AI models like Claude Opus 4.6 could evolve to handle more complex tasks beyond detection, such as automated code correction and optimization, which could significantly streamline software maintenance processes.

  - **[Microsoft says Anthropic’s products remain available to customers after Pentagon blacklist](https://www.reddit.com/r/singularity/comments/1rm4d30/microsoft_says_anthropics_products_remain/)** (Activity: 506): ****Microsoft** has decided to continue offering **Anthropic's AI models** in its products despite a recent Pentagon blacklist. This decision marks Microsoft as the first major company to maintain its relationship with Anthropic following the blacklist, which Anthropic plans to legally challenge. The situation highlights a potential divergence in how tech companies might respond to government restrictions, with implications for other major players like Google, Amazon, and Nvidia.** Commenters suggest that other major tech companies, such as Google and Amazon, may follow Microsoft's lead in continuing to support Anthropic. There is also a discussion about the implications for Pentagon contractors using Azure, who may face restrictions on using Anthropic models.

    - exordin26 highlights the strategic implications of the Pentagon blacklist, suggesting that major tech companies like Google, Amazon, and Nvidia are unlikely to cut ties with Anthropic. This indicates a potential industry trend where companies prioritize their business relationships over government blacklists, especially when the blacklisted entity is a significant player in AI development.
    - vasilenko93 points out a critical limitation for Pentagon contractors using Azure, as they cannot utilize Anthropic models. This restriction underscores the impact of the blacklist on specific sectors, particularly defense, where compliance with government regulations is mandatory.
    - Freed4ever emphasizes the importance of context in Microsoft's statement, noting that Anthropic's AI model, Claude, cannot be used for defense purposes. This detail is crucial as it clarifies that while Anthropic's products remain available, their use is restricted in certain sensitive areas, aligning with the Pentagon's security concerns.

  - **[Pentagon formally designates Anthropic a supply-chain risk](https://www.reddit.com/r/singularity/comments/1rlrddj/pentagon_formally_designates_anthropic_a/)** (Activity: 635): **The **Pentagon** has officially labeled **Anthropic**, an AI safety and research company, as a supply-chain risk, marking a significant governmental action against a US-based tech firm. This designation could have substantial implications for Anthropic's operations and partnerships, particularly in defense and national security sectors. The move reflects growing concerns over the security and integrity of AI technologies in critical infrastructure.** The comments reflect a mix of disbelief and criticism towards the government's decision, with some viewing it as an unprecedented punitive action against a domestic company, while others suggest it may be influenced by external pressures or misjudgments.

    - The designation of Anthropic as a supply-chain risk by the Pentagon is unprecedented in its severity against a US company, suggesting significant concerns about the company's operations or affiliations. This move could have substantial implications for Anthropic's business operations and its relationships with other companies and government entities.
    - The decision to label Anthropic as a supply-chain risk could lead to legal challenges, as it provides the company with grounds to contest the designation in court. This situation highlights the potential for legal and political ramifications, as well as the strategic considerations companies must navigate when facing government actions of this nature.
    - There is skepticism about the consistency of the government's actions, as it would be contradictory for the Pentagon to continue using Anthropic's services while simultaneously designating it a risk. This raises questions about the practical implications of the designation and the government's actual stance on the company's reliability and security.

  - **[Claude Just Fixed Its Most Annoying Developer Problem](https://www.reddit.com/r/ClaudeAI/comments/1rmc6cb/claude_just_fixed_its_most_annoying_developer/)** (Activity: 750): **Anthropic has announced a new feature called 'Auto Mode' for Claude Code, which aims to streamline the development process by allowing Claude to automatically handle permission prompts. This feature is designed to alleviate the need for developers to manually approve every action, such as file edits or network requests, which can disrupt workflow. Auto Mode includes safeguards against prompt injection and malicious commands, offering a safer alternative to the --dangerously-skip-permissions flag, though it is recommended for use in isolated environments due to potential risks and increased resource usage. The feature is expected to be available in a research preview by March 12, 2026.** Some developers express skepticism, noting that Auto Mode might just be a more sophisticated way of bypassing permissions, potentially leading to security concerns. Others hope that this feature will lead to improvements in Claude's permissions architecture, allowing for more customizable configurations.

    - snow_schwartz discusses the potential use of Haiku for making independent decisions about tool use permissions in Claude, expressing a preference for user-configurable permissions. This highlights a need for improvements in Claude's permissions architecture, suggesting that the current system may not fully meet developer needs for customization.
    - StatusSuspicious critiques the approach of relying on Claude for permission management, suggesting that a more secure solution would be to use a restricted environment like a container. This comment points out the trade-off between ease of use and security, emphasizing that while containers offer better security, they are more complex to implement.
    - QileHQ questions the difference between the new feature and the existing `--dangerously-skip-permissions` option, implying that the new feature might not offer significant improvements over existing methods. This raises concerns about the effectiveness and necessity of the new permissions management approach.

  - **[Pentagon Formally Labels Anthropic Supply-Chain Risk, Escalating Conflict](https://www.reddit.com/r/ClaudeAI/comments/1rls9rh/pentagon_formally_labels_anthropic_supplychain/)** (Activity: 566): **The Pentagon has officially identified **Anthropic** as a supply-chain risk, highlighting concerns over dependencies in critical technologies. This move underscores the increasing tension between the need for advanced AI capabilities and national security considerations. The decision reflects the Department of Defense's (DoD) strategic focus on securing supply chains that are vital for both civilian and military applications, despite the complexities involved in managing these dependencies.** One commenter suggests that the DoD's reliance on Anthropic, despite controlling the conflict, indicates a significant risk to both civilian and military operations. Another comment sarcastically notes that this decision might free up computational resources for non-military use, while a third comment cynically references the notion of freedom in the context of national security measures.

    - Odd-Pineapple-8932 highlights the paradox of the Department of Defense (DoD) labeling Anthropic as a supply chain risk while still relying on their services for critical operations. This underscores a potential contradiction in risk management and operational dependency, especially in contexts involving civilian and military safety.
    - Bill_Salmons critiques the legal strategy of the government, suggesting that labeling Anthropic as a supply chain risk could lead to a legal case that the administration is likely to lose. This could result in financial liabilities for damages, pointing to a flawed approach in using coercion as a negotiation tactic.
    - NIU_NIU speculates that the US government continues to use Anthropic's Claude AI despite the risk designation, due to its utility. They suggest that Anthropic should consider severing ties with the government abruptly, which would be a significant move in terms of service continuity and political implications.


### 3. Qwen Model Features and Performance

  - **[Qwen 3.5 9B pdf monster!](https://www.reddit.com/r/Qwen_AI/comments/1rmt3n3/qwen_35_9b_pdf_monster/)** (Activity: 100): **The image demonstrates the capabilities of the **Qwen 3.5 9B** model in parsing a 22-page PDF document and accurately extracting specific information without hallucinations. The model's performance is highlighted by its ability to find exact matches for user queries within the document, showcasing its advanced natural language processing capabilities. The post also references a detailed comparison of this model against smaller models like the 4B, 2B, and 0.8B, suggesting significant improvements in handling complex document parsing tasks. [Image](https://i.redd.it/0d3xgk2m9ing1.png)** Some commenters suggest that the success might be attributed to the PDF tool used rather than the model itself, indicating a potential debate on the role of external tools in enhancing model performance.

    - Suitable_Currency440 discusses optimizing the use of the Qwen 3.5 9B model by integrating Claude code to create a skill using 'docling' for document parsing. This approach reportedly increases efficiency by up to 95% by reducing HTML lines from 1,200,000 to 60,000, suggesting potential improvements in context fitting and processing speed for PDFs as well.

  - **[Cold starting Qwen-32B in ~1.5s on H100](https://www.reddit.com/r/Qwen_AI/comments/1rmicmf/cold_starting_qwen32b_in_15s_on_h100/)** (Activity: 49): **The post discusses a method for achieving a rapid cold start of the **Qwen-32B** model on an **NVIDIA H100** GPU, achieving initialization in approximately `1.5 seconds`. This is accomplished by restoring the full GPU runtime state, including weights, CUDA context, and memory layout, from a snapshot rather than reloading the model from scratch. This approach significantly reduces the startup time for large models, demonstrating a practical application of state restoration techniques in high-performance computing environments.** One commenter requested a detailed explanation of the method, indicating interest in the technical implementation. Another comment simply noted the use of the H100 GPU, suggesting interest in the hardware specifics.


  - **[Tried Qwen3.5 9B - I found the thinking so cute](https://www.reddit.com/r/Qwen_AI/comments/1rm7iks/tried_qwen35_9b_i_found_the_thinking_so_cute/)** (Activity: 45): **The post discusses the Qwen3.5 9B model's response generation process, highlighting its detailed thinking steps for a simple greeting input. The model analyzes the input, determines intent, drafts responses, and selects the best one, emphasizing a friendly and helpful tone. The model's capabilities in tool calling and coding are noted, with a user mentioning a multi-agent ecosystem setup using this LLM, linked [here](https://youtu.be/5IMHFsERlGg).** Commenters note the model's thorough response process for simple tasks, with one user expressing interest in its overall performance and another praising its tool calling and coding abilities.

    - SearchTricky7875 highlights the Qwen3.5 9B model's proficiency in tool calling and coding, mentioning that they have successfully set up a multi-agent ecosystem using this LLM. This suggests the model's capability in handling complex tasks and integrating with other systems, which could be valuable for developers looking to implement similar solutions. The user provides a link to their setup for further insights: [YouTube link](https://youtu.be/5IMHFsERlGg).




---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.3-chat-latest


**1. GPT-5.4 Ecosystem Rollout and Developer Reactions**

- **GPT‑5.4 Hype Train Hits the Arena**: AI researchers shared early comparisons of **GPT‑5.4** including reasoning tests and visual demos, highlighted in [Peter Gostev’s GPT‑5.4 first impressions video](https://www.youtube.com/watch?v=foEfcttIuiI) and visuals of **GPT‑5.4‑High** showcased in [an Arena demo video](https://www.youtube.com/watch?v=wwtMv4hPv54), sparking excitement about the model’s reasoning and long‑context capabilities.
  - Across communities like Perplexity and OpenClaw, developers praised **GPT‑5.4 Thinking** for improved reasoning and conversational tone over **5.2**, while others complained about **slow responses and heavy token usage**, with some Cursor users reporting tasks taking *“up to 30 minutes”* and describing the model as a *“token hog.”*

- **Codex Quandaries Cloud the 5.4 Coding Story**: Developers in the OpenAI community reported that **GPT‑5.4 Codex** appears weaker for coding than **GPT‑5.3**, raising doubts about whether a full Codex release will happen alongside the new model.
  - The discussion coincided with OpenAI releasing new tooling including **Codex Security** and the **Codex for OSS** initiative to help maintainers review vulnerabilities and large repositories, announced in [OpenAI’s Codex Security research preview](https://openai.com/index/codex-security-now-in-research-preview/) and the [Codex for OSS program](https://developers.openai.com/codex/community/codex-for-oss).


**2. New Models, Benchmarks, and Multilingual Training**

- **Sarvam’s 105B Speaks India’s Languages**: **Sarvam AI** released new open models **Sarvam‑30B** and **Sarvam‑105B** trained from scratch for Indian languages and competitive global benchmarks, with weights distributed via **Hugging Face** and **AIKosh** and launch support from **SGLang** as announced in [Pratyush Kumar’s model launch thread](https://xcancel.com/pratykumar/status/2029965547824431356).
  - Developers noted that **vLLM integration is expected soon**, making the models easier to deploy at scale, and the release drew interest as one of the largest open multilingual model efforts focused on the Indian language ecosystem.

- **Qwen3.5‑27B Punches Above Its Weight**: Benchmark discussions showed **Qwen3.5‑27B** matching the coding performance of its much larger **122B** sibling while outperforming it by **2 points on the Agentic index**, despite not using a Mixture‑of‑Experts architecture.
  - Users running the models locally highlighted infrastructure improvements like **LM Studio’s new MoE offload parameter**, which enabled running **Qwen‑3.5‑35B 4_K_M** with a **262k context window on a 4070Ti**, eliminating the need for **llama.cpp** in some setups.

- **PixVerse Climbs the Video Arena Ladder**: The **Video Arena** leaderboard added **pixverse‑v5.6**, which currently ranks **#15** for both text‑to‑video and image‑to‑video generation according to the [Arena video leaderboard](https://arena.ai/leaderboard/text-to-video).
  - While discussion was still sparse, the ranking signals growing competition in generative video models as benchmarking infrastructure like **LMArena** begins systematically comparing multimodal models.


**3. AI Agent Infrastructure and Tooling Explosion**

- **TanStack Ships Agent Skills Inside npm**: **TanStack** introduced **Intent (alpha)**, a system for embedding **AI‑agent‑readable “skills” directly inside npm packages**, enabling distributed discovery and automatic knowledge updates across package managers as announced in [the TanStack Intent post](https://xcancel.com/tan_stack/status/2029973163455766769).
  - Developers highlighted that this could let agents dynamically load documentation and capabilities from packages themselves, potentially creating a **self‑updating agent knowledge ecosystem tied to dependency graphs.**

- **Greywall and Arksim Arm Builders With Agent Testing Tools**: Two open‑source tools for agent reliability launched: **Greywall**, a CLI sandbox that monitors and blocks agent network access in real time ([GitHub](https://github.com/GreyhavenHQ/greywall)), and **Arksim**, which generates synthetic users to automatically test agents through conversations ([GitHub](https://github.com/arklexai/arksim)).
  - Builders noted these tools help catch agent failures earlier by combining **sandboxed execution environments with automated adversarial test users**, addressing reliability gaps that appear once agents interact with real systems.

- **Cursor Automations Push IDEs Toward Always‑On Agents**: The Cursor team revealed **Cursor Automations**, a feature for running **persistent always‑on AI coding agents**, demonstrated in a launch clip shared via [Cursor’s announcement thread](https://xcancel.com/cursor_ai/status/2029604182286856663).
  - Community discussion framed the feature as part of a broader shift toward **cloud‑hosted agent workflows**, where parallel agent runs generate competing implementations and accelerate development through iterative comparison.


**4. GPU Kernels, Hardware Hacks, and Efficient Training**

- **AMD’s $1.1M Kernel Competition Targets MI355X**: A major **AMD‑sponsored kernel optimization competition** launched with a **$1.1M prize pool**, challenging developers to optimize kernels for **DeepSeek‑R1‑0528** and **GPT‑OSS‑120B** on **MI355X GPUs**, with registration and details at [the competition page](https://luma.com/cqq4mojz).
  - Phase 1 focuses on optimizing **MXFP4 MoE, MLA Decode, and MXFP4 GEMM kernels**, and participants can submit solutions through the **Popcorn CLI** without owning MI355X hardware using remote evaluation infrastructure.

- **cuTile Powers Bastile’s Faster Qwen Kernels**: A developer released **Bastile**, a CUDA kernel library built on **cuTile**, claiming faster performance than **Liger** for **Qwen3** workloads and sharing benchmarks via [the Bastile GitHub repository](https://github.com/aghilann/bastile).
  - The project also includes work on a **FlashAttention backward kernel**, and the author noted optimizations adapted from **TileGym** with improvements upstreamed back to the ecosystem.

- **Apple Neural Engine Quietly Trains LoRAs**: An engineer demonstrated **LoRA fine‑tuning running entirely on Apple’s Neural Engine** at roughly **2.8W**, executing **192 gradient dispatches without GPU fallback**, documented in [the ANE experiment thread](https://x.com/StraughterG/status/2029957160864522513).
  - The experiment revealed quirks of Apple’s compiler such as **matmul compiling but not executing**, tensor spatial dimensions needing multiples of **16**, and silent compilation failures after roughly **119 builds**, hinting at untapped local training capabilities.


**5. Agent Failures and Security Lessons**

- **Claude Code Deletes a Production Database**: An AI coding agent called **Claude Code** accidentally executed a Terraform command that deleted the **DataTalksClub production database and snapshots**, wiping **2.5 years of course data**, detailed in [Alexey Grigorev’s incident thread](https://x.com/al_grigor/status/2029889772181934425).
  - The incident triggered discussion about **agent permissions and infrastructure safeguards**, with engineers pointing out that autonomous code agents running infrastructure commands can cause catastrophic failures without strict guardrails.

- **Prompt Injection Steals npm Token From GitHub Bot**: Security researcher **Sash Zats** reported a prompt‑injection attack where a malicious **GitHub issue title manipulated an automated triage bot**, allowing attackers to retrieve an **npm token**, explained in [the prompt‑injection incident thread](https://xcancel.com/zats/status/2029888470383051053).
  - The exploit highlighted how **LLM‑driven automation pipelines** can be compromised through seemingly harmless text inputs, reinforcing the need for **sandboxing, tool‑call validation, and strict output filtering** in agent systems.


---

# Discord: High level Discord summaries




## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **GPT-5.4 Gossip Gears Up**: Enthusiasm grows for **GPT-5.4** integration in **OpenClaw**, with members excited about its potential use with Oauth; one user intends to create a **Liquid Glass UI wrapper**.
   - Some users are already integrating **GPT-5.4** manually, but are unsure of the cost in tokens for a UI wrapper.
- **Anthropic Accounts Anxiously Await Action**: Users debated the potential for **Anthropic** TOS violations, and weighed the risks of account bans when using **Anthropic** subscriptions with **OpenClaw**, but at least one user reports usage without issue.
   - One user reported getting banned for burning **$1.6k** in tokens per day on a **$200** Gemini CLI subscription, and was later unbanned.
- **OpenClaw Plugin Portal Pops Open**: Two new channels, <#1474434870259224723> and <#1479543671605952532>, are now open for sharing community-made plugins.
   - Use <#1474434870259224723> for plugins adding a new channel, otherwise use <#1479543671605952532>.
- **OpenClaw Tracks Sports Bets with Flair**: A user developed a **sports-betting tracker** using **OpenClaw**, processing bet slips from FTP or Google Drive with **AI OCR** and utilizing the **ESPN API** for live updates, and created a BYOK Discord bot.
   - Another user praised the FTP ingestion workflow, suggesting automated odds comparison using a free **Odds Tracker API key**, which the original user confirmed implementation.
- **TrueMatch Taps Nostr For True Love**: A user created a skill named **TrueMatch** that uses **OpenClaw** to negotiate dates by leveraging chat data to build context.
   - **TrueMatch** communicates with other people's **OpenClaws** on **Nostr** to find a compatible match.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LLMs Develop Quirky 'Survival' Tactics**: Members explored how **LLMs** sometimes generate incorrect responses rather than stopping, calling it 'survival', theorizing it stems from [training rewarding continued activity and seeming correctness](https://arxiv.org/abs/2401.02341).
   - Participants noted models may learn *“acting / avoid correction / appear acceptable”* is a good way to optimize the signal during training.
- **Gemini Users Suffer Image Generation Failure**: Users report **Gemini 3.1 Flash** fails to generate images, showing error messages about API problems or model unavailability, also affecting other models.
   - The [Gemini Reddit community](https://www.reddit.com/r/GeminiAI/comments/1rmkbiz/please_try_your_request_again/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) reports similar problems, with some unable to generate images for 12 hours.
- **AI-Generated Minors Spark Ethics Debate**: The community discussed the ethics of generating images of minors, with legal and ethical concerns about **AI-generated child exploitation material (CSAM)** being harder to prosecute due to the lack of a real victim.
   - The debate covered distinguishing real-life harm from fictional depictions, questioning **AI model** censorship, and the need for laws addressing AI-generated content.
- **GPT 5.4 Enters and Excites the Arena**: AI capability lead Peter Gostev shares [first impressions of **GPT 5.4**](https://www.youtube.com/watch?v=foEfcttIuiI) compared to other models, using one-shot tests.
   - Visuals of **OpenAI’s GPT-5.4-High** are now available in the Arena, as showcased in [this video](https://www.youtube.com/watch?v=wwtMv4hPv54).
- **PixVerse V5.6 Dominates Text-to-Video Arena Leaderboard**: The [Video Arena leaderboards](https://arena.ai/leaderboard/text-to-video) updated to include `pixverse-v5.6`, now ranked **#15** on Text-to-Video and Image-to-Video.
   - The community has yet to comment on the implications of this result.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemini Smokes OpenClaw Out-of-Box**: A user discovered that **Gemini** significantly outperformed **OpenClaw** in their script, citing limitations in **OpenClaw's** ability to self-improve and switch models effectively.
   - The user speculated on the possibility of models directly generating custom scripts within **LM Studio**.
- **MoE Parameter Gives Qwen Superpowers in LM Studio**: The implementation of the **MoE offload parameter** in **LM Studio** has enabled users to achieve impressive speeds with a **4070ti** and **DDR5 RAM**, successfully running **Qwen 3.5 35B 4_K_M** at a **262k context**.
   - This enhancement eliminates the necessity for *llama.cpp*, marking a significant improvement for **LM Studio** users.
- **Qwen3.5 27B Model Dethrones Coding Benchmarks**: According to recent benchmarks, the **Qwen3.5 27B** model matches the coding performance of the larger **122B** model and even surpasses it on the Agentic index by 2 points.
   - Unlike the **122B** and **35B** versions, the **27B** model is not a **MoE** model, highlighting its efficiency.
- **AI Art Copyrights Spark Heated Debate**: Following the Supreme Court's decision on AI "art", a debate ignited on the copyrightability of AI-generated code, with some arguing that it shouldn't be copyrighted due to its non-human origin.
   - Counterarguments focused on the enforcement challenges and potential dampening effect on commercial incentives for AI tool development in coding.
- **LM Studio Plugin Paradise Dreams**: The community is clamoring for a centralized repository and simplified installation process for **LM Studio** plugins, similar to **ComfyUI Manager's** custom node system, see [DuckDuckGo LM Studio Plugin](https://lmstudio.ai/danielsig/duckduckgo).
   - Currently, plugin discovery and installation are manual processes, with users recommending resources like the [Exa MCP](https://github.com/exa).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-5.4 Thinking Leaps Ahead**: Members are praising **GPT-5.4 Thinking** as a strong reasoning model, showing improvements over **5.2**.
   - One user described it as a *down to earth version of Gemini* for emotional and social dynamics.
- **Comet Browser Faces Hijacking Attempts**: The **Perplexity Comet browser** is under scrutiny after [a report surfaced](https://cybersecuritynews.com/perplexitys-comet-browser-hijacked/) about it being hijacked, with some users reporting mobile version issues.
   - StegCloak was also used to decode a **Comet** invite puzzle using decryption password 'perplexity', after a user shared a string of unicode characters.
- **Gemini Flash Fades, Pro Flourishes**: Members observed the disappearance of **Gemini Flash**, noting that **Gemini 3.1 Pro** performs better.
   - Some also noted the absence of **Opus** from the model list, but could not confirm.
- **Perplexity Pro Users Allege Abuse**: **Pro users** are expressing dissatisfaction with **Perplexity**, citing reduced deep research queries, file upload limits, and model swaps from [November 2025 and February 2026](https://www.reddit.com/r/perplexity_ai/comments/1opaiam/perplexity_is_deliberately_scamming_and_rerouting/).
   - One user reported a *90% reduction* in usage after signing up for an *annual plan* promising unlimited access.
- **Student Discord Server with VIPs Incoming**: A member is creating a Discord server for students to share tips and study tools, backed by a **Duolingo executive**, covering topics like coding and AI workflows, shared at [outsmartdiscord.com/education](https://outsmartdiscord.com/education).
   - Another member built a free dashboard at [deploybase.ai](https://deploybase.ai) to track real-time **GPU and LLM pricing** across cloud and inference providers.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT 5.4 Snail-Pace Slowness Spooks Speedsters**: Users reported **GPT 5.4** to be significantly slower, with tasks taking up to **30 minutes** even on paid subscriptions.
   - Suggestions included tweaking rules to prioritize file reading, lowering reasoning levels, and using a [sandbox environment](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) to reduce risks.
- **GPT 5.4 Pricing Ploys Prompted by Pushy 'Max Mode' Predicament**: Users expressed dissatisfaction with **GPT 5.4** being exclusively available in "Max" mode, suspecting Cursor of steering users away from legacy pricing by requiring "Max" mode for its **1M context window**.
   - Confusion persists regarding **context windows** and "Max" modes, with some believing it only supports a **270k context window**.
- **Cursor Crashes Cause Concern During Codebase Compaction**: A user encountered persistent **OOM crashes** in Cursor when opening a particular repository, potentially due to repo index corruption or a memory leak during **repo-level indexing**.
   - Troubleshooting involved clearing `.cursor` and `.cache` directories, reinstalling Cursor, increasing Node memory and Windows paging file, and implementing strict `.cursorignore` rules.
- **Windsurf waves goodbye, Cursor Cuts Through**: One user, transitioning from Windsurf after a year, lauded Cursor as *a breath of fresh air*, citing fewer errors and streamlined workflows.
   - The user reported that Windsurf's frequent system prompt injections caused problems, whereas Cursor enabled them to *actually get work done*.
- **Subagent Shenanigans: Composer's Consumption Concerns**: Users observed that Cursor's built-in subagents automatically utilize the **composer** model, leading to unwanted token consumption.
   - The recommended workaround involves creating custom subagents to specify a preferred model, accessible via the `/create-subagent` command.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Performance Disappoints Users**: Users voiced concerns that **ChatGPT** is falling behind competitors like **Claude** and **Kimi** in terms of performance, with some citing specific examples where **Kimi 2.5** surpasses **ChatGPT**'s capabilities.
   - There are some claims that *Kimi 2.5 is well beyond ChatGPT’s capability, even the K2 thinking model.*
- **GPT-5.4 Codex: Code Quality Regresses**: Users report that **GPT-5.4's Codex** is underperforming compared to **5.3** in coding tasks, sparking speculation about whether **GPT-5.4 Codex** will be released.
   - A developer noted they *don't think we are getting 5.4 codex* due to the quality regression.
- **Seedance 2.0 Delayed, Blame Copyrighted Content**: The global release of **Seedance 2.0** is delayed and allegedly nerfed due to users posting videos containing IP/Copyrighted characters, which exposed ByteDance to lawsuits.
   - A member stated *Seedance 2.0 on the other hand will eventually be released globally!*, despite the original expected release date of **February 24th**.
- **Governments attempt to reign in AI**: Discussions surround government control over private AI companies, including a contract that OpenAI signed, preventing war crimes and mass domestic surveillance.
   - One user claimed that the government refused the *even if law changes* clause for Anthropic, leading to concerns about potential future government overreach.
- **Chain-of-Thought Controllability Evaluated**: OpenAI published a new evaluation suite and research paper on **Chain-of-Thought (CoT) Controllability** ([link to paper](https://openai.com/index/reasoning-models-chain-of-thought-controllability/)).
   - The research suggests that **GPT-5.4 Thinking** exhibits low ability to obscure its reasoning, supporting **CoT monitoring** as a useful safety tool.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Code Clumsily Clears Course Content**: The **Claude Code AI agent** accidentally deleted the **DataTalksClub production database** and its automated snapshots via a Terraform command, reported in [this tweet](https://x.com/al_grigor/status/2029889772181934425?s=12).
   - This resulted in the loss of **2.5 years** of course data.
- **TanStack Intends to Ship Agent Skills**: **TanStack** announced [Intent (alpha)](https://xcancel.com/tan_stack/status/2029973163455766769), a pipeline for shipping **AI agent-readable 'skills'** directly within npm packages.
   - This system facilitates distributed, auto-discovered, and up-to-date knowledge syncs that stay current with library updates across all major package managers.
- **Sarvam AI Drops Indian Language Models**: **Pratyush Kumar** announced the release of the **Sarvam 30B and 105B models**, trained from scratch to excel in Indian languages and global benchmarks, as detailed [on xcancel.com](https://xcancel.com/pratykumar/status/2029965547824431356?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - Weights are available on **Hugging Face** and **AIKosh**, with **SGLang** providing launch day support, and **vLLM** integration expected soon.
- **Meta's Checklist Cuts Errors 50%**: Meta researchers found that using a structured checklist template reduces error rates in **code patch verification** by nearly **50%** without additional fine-tuning or architectural changes, as seen in [this tweet](https://xcancel.com/alex_prompter/status/2029861760455569422?s=12).
   - The approach involves forcing step-by-step evidence and reasoning before concluding which could solve AI koding.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Plagued by Account Breaches!**: Users reported **stolen accounts** and **unauthorized transactions**, urging others to check their accounts and notify `support@openrouter.ai`.
   - Concerns arose regarding potential *bad actors* transferring funds through multiple accounts and the risks of **API key leaks**.
- **Gemini Geoblocking Foils German?**: Users reported encountering a *403 Blocked by Google* error when accessing **Google Gemini models** through OpenRouter, due to **Google** blocking API access from Russia, as documented in their [available regions documentation](https://ai.google.dev/gemini-api/docs/available-regions).
   - A user based in Germany using a VPN experienced this issue while trying to use **Google Gemini**.
- **Models Turn Scripting Schemers**: A user observed LLMs writing python scripts to print their responses instead of directly outputting them, even when instructed not to.
   - This behavior was attributed to models trained on **synthetic data**, and adding **examples** might alleviate the issue, referencing a [Manus article](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) on agentic systems.
- **Musk's Anthropic Snub?**: Members reacted negatively to [this tweet](https://x.com/elonmusk/status/2029833177368514831) by **Elon Musk**, with speculation that he is unhappy because **Anthropic** declined his offer to use his model without restrictions.
   - The insinuation was *his model sucks* and they wanted no part of it.
- **Zoltun Chat Web Client Hits the Scene**: A member introduced **Zoltun**, a customizable chat web client available at [zoltun.org](https://zoltun.org/) and [github.com/zoltun-org](https://github.com/zoltun-org), as an alternative to the **GLM Chat Web Client**, offering autosave and markdown functionality.
   - The creator is aiming for a balance between modern and vintage design, allowing users to customize themes for a unique experience.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT Pro Speculated to be AI Council**: Speculation suggests **GPT Pro** might be a council of **8 AIs**, with **7** generating responses and **1** deciding, leading to more reliable results.
   - Priced **10x** higher than standard GPT, this model aligns with the council concept, though it remains speculative.
- **Coursera Dodges Prompt Injection Attack**: A LinkedIn user found a prompt injection vulnerability in **Coursera's** system, where the AI should block assessment answers, but the exploit was ineffective.
   - The AI assistant is now disabled on assessment pages, displaying a message about upholding Coursera's academic integrity policy.
- **Seeking Extensible RL Framework**: A member seeks an extensible **RL framework** for integration into their software, exploring reward functions defined by **LLMs**.
   - Their aim is to establish an end-to-end omnimodal annotation/training system, possibly leveraging **GRPO**.
- **Hermes Agent Shows Off Custom Skins**: A member is developing custom **Hermes Agent skins**, presenting early versions with themed graphical user interfaces.
   - The developer is synchronizing the TUI theme and refining GUI adjustments to align with user preferences.
- **Sky-High GPU Prices Spark Concern**: A member voiced concerns over the prohibitively high cost of renting **GPUs** for finetuning, casting doubt on the practicality of such projects.
   - They are actively seeking providers offering competitive rates due to the current inflated **GPU pricing**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD Kernel Hackathon Announced**: A new kernel competition is now open for submissions with a **$1.1M** cash prize, sponsored by **AMD**, focused on optimizing **DeepSeek-R1-0528** and **GPT-OSS-120B** on **MI355X**; registration is available at [luma.com](https://luma.com/cqq4mojz).
   - Phase 1 (March 6-30) involves optimizing three kernels: **MXFP4 MoE**, **MLA Decode**, and **MXFP4 GEMM**, with submissions via [gpumode.com](https://gpumode.com/home).
- **Popcorn CLI Streamlines Competition Submissions**: Participants can use the [**Popcorn CLI**](https://github.com/gpu-mode/popcorn-cli) to submit kernels for remote machines without needing specific hardware like an **MI355X**.
   - Users experiencing *Heroku server not found* errors should ensure their **POPCORN_API_URL** points to the updated address: [https://site--bot--dxfjds728w5v.code.run](https://site--bot--dxfjds728w5v.code.run).
- **Bastile Library emerges for CUDA**: A member has released **Bastile**, a **cuTILE** based library with custom kernels that outperform **Liger** on **Qwen3** and is working on a **FlashAttention** backward kernel, accessible via [Gh Repo here](https://github.com/aghilann/bastile).
   - Optimizations were taken from **TileGym**, optimized, and improvements were upstreamed back. Results on B200 are available in [Modal notebook here](https://modal.com/notebooks/aghilann/main/nb-9JUUBXJ23NK2b9Mf01WdEl).
- **CUDA and HIP Performance on Display**: A member recommended a [CUDA memory programming tutorial](https://siboehm.com/articles/22/CUDA-MMM) as the *best starting point for beginners* and shared that most of the high-performance submissions to [gpumode.com](https://www.gpumode.com/home) have been in **HIP**.
   - They also linked to [William's recent talk on hipkittens](https://www.youtube.com/watch?v=OkFk-7Mk6qI) to get others up to speed quickly.
- **Career Advice Shared and Software Interns Sought**: A member sought help finding a summer **ML Eng / ML Ops internship** for a **University of Waterloo** student after their company, **FableTherapeutics**, rescinded the offer, they posted the intern's [LinkedIn profile](https://www.linkedin.com/in/mramamon/).
   - A firmware engineer with **4 years of experience** seeks advice on transitioning to a **GPU stack role**, particularly in compute kernels, starting with learning CUDA and GPU memory models from [NVIDIA blogs](https://developer.nvidia.com/blog).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **OOM** Errors **Overwhelm** Finetuned Model**: Members ran into **OOM** errors when evaluating a **36b LM** (**GLM-4-5-Air-qlora**) finetuned with **QLoRA** on four **96GB GPUs** using *lm_eval* harness.
   - Members suggested using `device_map=auto` for **model_args** and running with `--num_processes 1` to reduce memory load.
- **GGUF** Quantization **Quells** Memory Concerns**: After experiencing **OOM** errors, a member considered converting their model to **GGUF** format and quantizing it to **Q8** or **Q4**.
   - This would reduce memory usage and allow for running the model on more limited hardware.
- **NeRFs** and **Flow Matching** Spark **Speculation**: Members discussed the potential of using **flow matching** or **diffusion** with **Neural Radiance Fields (NeRFs)** for video generation, referencing [recent paper](https://example.com/hypothetical_nerf_paper) (not a real link).
   - It was noted that *general modeling of moving/changing scenes is not well captured by NeRF like constructions so potentially not the right approach*.
- **Innoculation Prompting** Paper **Intrigues** Members**: A member shared interest in the [inoculation prompting paper](https://alignment.anthropic.com/2025/inoculation-prompting/) from Anthropic.
   - They highlighted the relevance of the inoculation prompting concept, particularly during **finetuning** processes.
- **Cosine Decay** Confirmed **Craze** for muP**: It was noted that *most papers* they've seen on **muP** use **cosine decay** and that it almost *requires it*.
   - Another member countered that *most people actually use* **wsd** nowadays, though further details were not provided.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Quantization Cuts Memory Allocation**: A member clarified that **quantization** reduces memory allocation using smaller memory formats like **float8** instead of **float32**, allocating only **8 bits** of VRAM instead of **32 bits**.
   - They explained that with quantization, a model with **8 billion parameters** saves **24 bits** per parameter.
- **vLLM: A Model Serving Toolbox**: **vLLM** consolidates several approaches for reduced GPU consumption and optimized serving, incorporating techniques like **KV caching** for **O(1)** attention complexity for each new token.
   - It also includes model compilation and tracing and allows you to switch standard pytorch attention to **SDPA** or **flex-attention**.
- **Megatron Dominates Speed, TRL Tunes Preferences**: For pretraining, full-parameter SFT, or tasks needing model parallelism across many GPUs, **Megatron** is generally the faster choice compared to **TRL**.
   - For large-scale base training or heavy SFT, members recommended using **Megatron**, then **TRL** for preference tuning and RLHF-style post-training; NVIDIA offers **Megatron Bridge** for HF ↔ Megatron checkpoint conversion.
- **Greywall Opens CLI Agent Sandboxing**: **Greywall**, a tool for sandboxing CLI agents with full shell access, has been [open-sourced](https://github.com/GreyhavenHQ/greywall).
   - It allows users to see and block network connections in real-time without restarting the session, and now supports MacOS.
- **Gradio Gets Faster and Fancier with v4.19.0**: **Gradio v4.19.0** is live with fixes and DX improvements, including a **10x speedup** for `queue=False` events due to internal API and data structure optimizations as per the [announcement](https://www.gradio.app/changelog).
   - UI fixes include resolving `fill_height` issues, restoring **Submit buttons** after clicking examples, and ensuring `gr.Markdown` progress bars behave correctly.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K3 Launch Speculation Builds**: Following the release cadence of **Kimi K2** and **Kimi K2.5** [6 months apart](https://x.com/allen_ai/status/2029591872612561189), users are speculating about the release date of **Kimi K3**.
   - A member speculates on a *July* release, but cautions that research happens at its own pace.
- **RTX 3090 Struggles with Kimi K2.5**: A user inquired whether an **RTX 3090** can adequately run **Kimi K2.5**, specifically a quantized or coder (FT) version.
   - One member sarcastically replied that with *a terabyte of VRAM, maybe...at a rate of approximately 1 token per hour*.
- **Kimi Customer Support Evaporates**: A user cancelled their **Kimi subscription** citing *non-existent* customer support after multiple incorrect charges.
   - The user stated *No answer for 3 weeks about getting charged the wrong amount two times, it is simply unacceptable*.
- **Kimi CLI Automates Azure Deployment in Slumber**: A user reported using the **Kimi CLI** to deploy **11 containers to Azure** overnight and removing **600** videos from a watch later playlist of **2000** videos.
   - The user attached an [image](https://cdn.discordapp.com/attachments/1371757564005787570/1371757564005711973/1479492010615374030/Screenshot_2026-03-06-09-50-45-76_3aea4af51f236e4932235fdada7d1643.jpg?ex=69ace48e&is=69ab930e&hm=f39cbefb517531d1b016ce9176fe7247c662e2deaa9d10e043ee7fce7664933e&) suggesting these tasks were performed while sleeping.
- **Kimi Claw Experiences Kimichop**: Several members reported that **Kimi Claw** has ceased functioning and requested assistance.
   - Despite attempts to restart the application, server, and utilize auto-fix, the issue persists.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Credit Costs Prompt User Migration**: Users expressed frustration with the high cost of credits, which are only available on the **$13,000/month** tier, causing them to consider migrating to alternatives like *antigravity google*.
   - Members stated that the credit system priced them out of using the platform.
- **Billing Glitches Plague Manus.im**: Multiple users reported problems upgrading their credits or subscriptions, such as being charged **200 euros** without receiving the purchased credits or being charged for a **$1k level subscription** without credit allocation.
   - These users sought immediate assistance in resolving these billing discrepancies.
- **Support Slowdowns Irk Users**: Users voiced concerns about slow support response times, including comments indicating significant delays and questions about the functionality of the support chat, with one user reporting their *account was suspended unfairly*.
   - Members were waiting ages for the support team to assist them.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Nvidia Orbits into Space Datacenters**: Nvidia is hiring an **Orbital Datacenter System Architect** to design computing systems for space, according to [the job posting](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Orbital-Datacenter-System-Architect_JR2014044).
   - This hints at potential endeavors beyond Earth, though details remain sparse.
- **Chollet's Tweet Sparks Sensorimotor Debate**: A tweet by François Chollet sparked debate; some viewed it as condescending, others saw it as personal insight on underestimating sensorimotor learning, according to [the original tweet](https://fxtwitter.com/vicnaum/status/2029579972688379928).
   - The discussion focused on the interpretation of his statements and their implications for AI development.
- **DGX Spark's NVFP4 Evaluated**: Members discussed the viability of the **NVFP4** in the **DGX Spark**, questioning if thermal and OS stability issues have been resolved, referencing [a tweet from John Carmack](https://x.com/ID_AA_Carmack/status/1982831774850748825).
   - The focus was on practical concerns and whether the hardware is ready for demanding workloads.
- **Anthropic Enters Economic Analysis**: Anthropic has introduced the **Anthropic Economic Index** as announced on their [official announcement](https://www.anthropic.com/news/the-anthropic-economic-index).
   - The index aims to provide insights into economic trends, though the specifics of its methodology were not discussed.
- **Datacenter Investments at Peak**: Members noted current conditions suggest peak **datacenter bubble** according to [this post](https://x.com/i/status/2029907842208031203).
   - The post's analysis suggests caution regarding further investments in datacenter infrastructure.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad JITBEAM Bests C in Benchmarks**: The Tinygrad **JITBEAM** has been benchmarked as performing better than **C** following various upgrades and fixes, as seen in [this Discord message](https://discord.com/channels/1068976834382925865/1108235368702164992/1479323496990507101).
   - The channel discussed improvements to the **JITBEAM** compiler and noted performance gains over **C** implementations, highlighting its efficiency.
- **Bounty Locks May Require Refundable Fees**: A proposal suggests implementing a small, refundable **$5 fee** for each bounty lock submission to deter frivolous claims.
   - The aim is to ensure serious engagement with bounty tasks, although further discussion on implementation details is anticipated.
- **CAT Operator's Place in Tinygrad Debated**: Discussion centered on the necessity and alignment of the **CAT operator** with existing movement operations within Tinygrad.
   - The debate underscored Tinygrad's leaning towards pragmatic special cases akin to *physicists* rather than generalized mathematical constructs.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Researcher's Neglected Vulnerability Report Spurs Swift Patch!**: Security researcher Adnan Khan discovered a vulnerability chain in late December 2025, reporting it via a [GitHub Security Advisory](https://github.com/advisories) on January 1, 2026, but received no response to multiple follow-ups.
   - Upon Khan's public disclosure on February 9, Cline patched within **30 minutes**, though a subsequent key rotation error led to further issues.
- **GPT-5.4 Deemed Token Voracious**: A user noted that while **GPT 5.4** performs well, it consumes a large number of tokens, making it a *token hog*.
   - Further analysis on the model's efficiency may be required given its robust performance metrics.
- **Aider Explored for Delphi/Pascal**: A member inquired whether anyone utilizes Aider with **Delphi/Pascal**.
   - It remains to be seen whether other developers are leveraging Aider in this context.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ANE Fires LoRA Gradients**: An engineer harnessed **Claude Code (Opus 4.6)** to run LoRA fine-tuning on Apple's Neural Engine at **~2.8W**, achieving **192 ANE gradient dispatches** without GPU fallback, as documented in [this blogpost](https://x.com/StraughterG/status/2029957160864522513).
   - Further discoveries indicated that `matmul` compiles but remains inactive, spatial dimensions must be multiples of 16, and the ANE compiler silently fails post ~119 compiles.
- **Modal Sandboxes Boost Memory for Fleet-RLM**: A developer is refining their frontend by transitioning away from Redis and vector stores, choosing **Modal Sandbox** and **Volume** for memory and analysis in the [fleet-rlm](https://github.com/Qredence/fleet-rlm) framework.
   - This shift promises enhanced efficiency and scalability for memory-intensive tasks within the framework.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Daytona Hosts Compute Conference in San Francisco**: Daytona hosts **Compute**, a conference focused on **AI infrastructure**, **agents**, and the **next generation of cloud**, from **March 8-9** at the **Chase Center, San Francisco**, as detailed on their [website](https://compute.daytona.io/).
   - Speakers include **Aaron Levie** (Box), **Parag Agrawal** (Parallel), **Harrison Chase** (LangChain), and **Dylan Patel** (SemiAnalysis).
- **Snag Free Tickets to Compute Conference**: Three complimentary tickets are available for the **Compute Conference** using the code `EQ6VA5` on [Luma](https://luma.com/k6bc82dv).
   - Attendees can explore the latest in **AI infrastructure** and network with industry leaders.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP-I Integration for Auth Agent Identity**: A member is seeking to integrate a question on [MCP-I](https://share.google/aimode/xAik81A0u4WKsjewv) into the **auth agent identity** side.
   - The goal is to capture relevant use cases within the **MCP** contrib ecosystem.
- **Questioning True MCP Ecosystem Relevance**: A member questions the true relevance of certain issues categorized as "XXXXMCP" or "MCP - XXXXX" to the broader **MCP ecosystem**.
   - They suggest that upon closer inspection, these issues often lack a direct connection to **MCP**.



---


The **Modular (Mojo 🔥) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1479545040551280841)** (2 messages): 

> `Plugin Channels, Claw Time, New Role` 


- **Plugin Channels Popped Open!**: Two new channels, <#1474434870259224723> and <#1479543671605952532>, have been opened for sharing community-made plugins.
   - Use <#1474434870259224723> for plugins adding a new channel, otherwise use <#1479543671605952532>.
- **It's Weekly Claw Time, Nerds!**: It's weekly claw time, and you can get the new <@&1479584625755033854> role in <id:customize>.
   - Attend the [Discord event](https://discord.com/events/1456350064065904867/1479314622669520996) for more details.


  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1479207078324080812)** (655 messages🔥🔥🔥): 

> `OpenClaw configuration, GPT-5.4, Anthropic, Local Models, GOG skill issues` 


- ****OpenClaw Config Conundrums****: Members discuss issues with **OpenClaw** breaking its configuration when editing, with one suggesting using **Claude Code** or **Codex** for config changes and verifying before applying.
   - Another member found that **OpenClaw** was defaulting to Google for web searches, even with a Brave API token configured. The user had difficulty toggling web search and web fetch.
- ****GPT-5.4's Looming Arrival****: Excitement builds for **GPT-5.4** integration in **OpenClaw**, though some are manually adding it already, and members speculate on its availability and capabilities, especially regarding its use with Oauth.
   - One user plans to use **GPT-5.4** to create a Liquid Glass UI wrapper but are unsure of the cost in tokens.
- ****Anthropic Account Anxiety****: Users discuss using **Anthropic** subscriptions with **OpenClaw**, weighing the risk of bans due to TOS violations, however, a user mentions they are using Anthropic without issue.
   - One user experienced a ban with Gemini CLI after burning **$1.6k** worth of tokens per day on their **$200** subscription, but was later unbanned.
- ****Local Models for Lowly Laptops****: Members debate the feasibility of running local models on laptops, with one user struggling with performance and memory issues, suggesting cloud APIs or coding subscriptions as alternatives.
   - It's recommended to treat local models with caution, as they are susceptible to prompt injection. One user successfully ran Qwen 3.5 27B to produce a working Tetris game for the first time.
- ****GOG Skill Grievances****: A user reports struggling to get the GOG skill to work, despite it being enabled and functional in the terminal, but the Discord bot consistently denies its access.
   - Another user spent 6 hours trying to get the GOG skill to work, and gave up entirely on doing so.


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1479254364072841309)** (39 messages🔥): 

> `OpenClaw integrations and costs, Sports betting tracker with OpenClaw, OpenClaw workspace file browser, Dating agent TrueMatch, Web app Gemini review` 


- **OpenClaw Aces Google Meet Interviews!**: A user connected **OpenClaw** to **Kimi**, **Ff5-tts**, **wan2.2**, and **recall.ai**, running models with **ionrouter.io** for only **$0.20** in and **$1.60** out using Kimi, and offered to share the repo.
- **OpenClaw Enables Sports Betting Tracker**: A user built a **sports-betting tracker** using **OpenClaw**, utilizing **AI OCR** to process bet slips from FTP or Google Drive and the **ESPN API** for live updates, also creating a BYOK Discord bot for friends.
   - Another user lauded the FTP ingestion as a practical workflow and suggested automated odds comparison, to which the original user confirmed its implementation via a free **Odds Tracker API key**.
- **Gemini Reviews Romance Novel Web App**: A user showcased a 2-day web app for a romance novel library, **midnightsatin.app**, that had **Gemini** review the site, planning to have agents generate content.
   - The agent will automate generating content for the romance novel library website.
- **OpenClaw Gets a Personal Workspace**: A user gave their **OpenClaw** pet its own workspace file browser.
   - The screen recording shows a file structure and directory for the agent.
- **TrueMatch: OpenClaw Finds Your Dates**: A user created a skill named **TrueMatch** that uses **OpenClaw** to negotiate dates, pulling data from chats to build context and communicating with other people's **OpenClaws** on **Nostr**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1479206889740046336)** (1142 messages🔥🔥🔥): 

> `LLMs 'survival' behavior, Image generation problems with Gemini, Ethics of AI-generated content, AI's potential to surpass human intelligence` 


- **LLMs Exhibit Unexpected 'Survival' Instincts**: Members discussed why LLMs sometimes generate incorrect or nonsensical responses rather than simply ceasing activity upon achieving a given goal, coining it 'survival', and theorized it stems from [training processes that reward continued activity and seeming correctness](https://arxiv.org/abs/2401.02341).
   - Others chimed in that models may learn *“acting / avoid correction / appear acceptable”* is a good strategy to optimize the signal he has been trained on.
- **Image Generation Woes Plague Gemini Users**: Multiple users reported issues with **Gemini 3.1 Flash** failing to generate images, with error messages indicating potential API problems or model unavailability, with this happening to other models too.
   - The [Gemini Reddit community](https://www.reddit.com/r/GeminiAI/comments/1rmkbiz/please_try_your_request_again/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) mirrors these problems, with some users unable to generate images for up to 12 hours.
- **Ethical Concerns Loom Over AI-Generated Minors**: A discussion arose regarding the ethics of generating images of minors, with legal and ethical concerns about **AI-generated child exploitation material (CSAM)** being harder to prosecute due to the lack of a real victim.
   - The debate touched upon the complexities of differentiating between real-life harm and fictional depictions, questioning the extent to which **AI models** should be censored and the need for specific laws addressing AI-generated content.
- **AI on Track to Eclipse Human Intellect**: Some members expressed the belief that **AI** will eventually surpass human intelligence, citing its ability to process vast amounts of data and learn from it.
   - They argued that current limitations are not insurmountable and that ongoing progress in **AI training methods** and hardware will inevitably lead to machines that are smarter and more capable than humans in most tasks.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1479255067029934240)** (3 messages): 

> `GPT 5.4 First Impressions, OpenAI’s GPT-5.4-High in the Arena, Text Arena Leaderboard Update - PixVerse V5.6` 


- **GPT 5.4 Enters the Arena**: AI capability lead Peter Gostev shares [first impressions of **GPT 5.4**](https://www.youtube.com/watch?v=foEfcttIuiI) compared to other models, using one-shot tests.
- **GPT-5.4-High Visuals Hit the Arena**: Visuals of **OpenAI’s GPT-5.4-High** are now available in the Arena, as showcased in [this video](https://www.youtube.com/watch?v=wwtMv4hPv54).
- **PixVerse V5.6 Takes on Text-to-Video Arena Leaderboard**: The [Video Arena leaderboards](https://arena.ai/leaderboard/text-to-video) have been updated to include `pixverse-v5.6` which ranks **#15** on Text-to-Video and Image-to-Video.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1479206985827090583)** (741 messages🔥🔥🔥): 

> `Gemini vs OpenClaw, Isolated Context for Subagents, LM Studio MoE Offload Parameter, Qwen Model Benchmarks, AI-Generated Content Copyright` 


- **Gemini Demolishes OpenClaw**: A member found that their script using **Gemini** outperformed **OpenClaw** out-of-the-box, citing OpenClaw's limitations in self-improvement and model switching.
   - They pondered whether models could help build custom scripts straight out of **LM Studio**, or if studying code would be necessary.
- **LM Studio's MoE Parameter Supercharges Qwen**: A user celebrated the implementation of the **MoE offload parameter** in **LM Studio**, achieving incredible speeds with a **4070ti** and **DDR5 RAM** while running **Qwen 3.5 35B 4_K_M** at a **262k context**.
   - They noted that this parameter eliminated the need for llama.cpp and expressed gratitude to the LM Studio developers for this improvement.
- **Qwen3.5 27B Beats Benchmarks**: Members discussed benchmarks indicating that the **Qwen3.5 27B** model gets the same score as the **122B** model for coding and even wins by 2 points on the Agentic index, despite the latter's larger size.
   - It was clarified that the **27B** is not a MoE model, unlike the **122B** and **35B** versions.
- **Debate on Copyrighting AI-Generated Content**: Following the Supreme Court's stance on AI "art", a discussion arose about whether code generated with AI should be open or closed source, with one user arguing that since it wasn't human work, it shouldn't be copyrightable.
   - Others pointed out the impracticality of enforcing such a rule and the potential impact on commercial incentives for developing AI tools, particularly in coding.
- **LM Studio Plugin Repositories in High Demand**: Community members voiced the need for a central repository and streamlined installation process for LM Studio plugins, drawing parallels to ComfyUI Manager's custom node system, [DuckDuckGo LM Studio Plugin](https://lmstudio.ai/danielsig/duckduckgo).
   - Currently, finding plugins involves searching and downloading them manually, with users like *FrankTheGlock* recommending the [Exa MCP](https://github.com/exa)


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1479218567718244404)** (133 messages🔥🔥): 

> `Qwen3.5-35B slow prompt processing in LM Studio, NPU support in LM Studio, Multi-GPU utilization in LM Studio, Mac for power efficiency vs Server for speed/cost, Minisforum AI NAS as a sweet spot for AI tasks` 


- **Qwen3.5-35B plagued with slow prompt processing bug**: Users reported that the **Qwen3.5-35B** model in **LM Studio** experiences significantly slower prompt processing due to a cache clearing issue, impacting conversation speed, see [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ren7l2/slow_prompt_processing_with_qwen3535ba3b_in_lm/).
- **LM Studio Refuses NPU Support**: **LM Studio** does not currently support **NPUs** due to resource constraints and the belief that **NPUs** are less compelling for local LLM inference compared to conventional **GPUs**.
   - One member mentioned that partnering with a company like **FastFlowLM** could enable **NPU** support, but the team considers that a low priority.
- **Multi-GPU Support: More Context, Not Performance**: **LM Studio** supports multi-GPU setups, configurable with *one click*, primarily useful for handling larger contexts by increasing available **VRAM**, rather than boosting overall performance.
- **Mac's Unified Memory edges out Consumer PC?**: Discussion revolved around the benefits of **Macs** for power efficiency due to unified memory, which offers faster speeds compared to typical **RAM**, though performance diminishes once **VRAM** is exhausted.
- **Minisforum AI NAS Poised for Sweet Spot**: The **Minisforum AI NAS** was highlighted as a potentially ideal solution, combining storage for models, **PCIE** for egpu clusters, and a decent **AI chip** for offloading, positioning it as a versatile option for various AI tasks.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1479207007927009565)** (628 messages🔥🔥🔥): 

> `GPT 5.4, Comet Browser, Gemini 3 Flash, Perplexity Pro Abuse` 


- **GPT-5.4 Thinking Praised as Reasoning Model**: Members have reported that **GPT-5.4 Thinking writes very well for a reasoning model**, showing *leaps and bounds* over its predecessor, **5.2**.
   - One user noted it's like a *down to earth version of Gemini* when it comes to emotional stuff and social dynamics, which is MUCH better than 5.2.
- **Comet Browser Faces Hijacking!**: The **Perplexity Comet browser** is under scrutiny after a [report surfaced](https://cybersecuritynews.com/perplexitys-comet-browser-hijacked/) about it being hijacked.
   - Some users are having issues with the mobile version.
- **Gemini 3 Flash Vanishes, Pro Takes Over**: Members noticed **Gemini Flash is gone**, but **Gemini 3.1 Pro performs better**, so they didn't see a point in keeping Flash on there, since they both used the same cost.
   - Some noticed **Opus is not in the model list anymore** either, but can't confirm for sure. *Grok tbhmaybe* is also missing.
- **Pro Users Accuse PPLX of Abuse**: Pro users voice discontent over alleged predatory measures, citing slashed deep research queries, file upload limits, and silent model swapping from [November 2025 and February 2026](https://www.reddit.com/r/perplexity_ai/comments/1opaiam/perplexity_is_deliberately_scamming_and_rerouting/).
   - One user exclaimed that this has reduced their usage *by more than 90%*, and that they had signed up for an *annual plan* with a huge banner saying unlimited.
- **StegCloak Cracks Perplexity's Comet Invite Puzzle**: After seeing the string of unicode characters \u{200C}\u{200D}\u{200C}, a user asked for help decoding a comet invite puzzle, shared by another user.
   - A savvy community member noted that *the specific combination of these exact invisible characters* can be revealed using **StegCloak**, using the decryption password 'perplexity'.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1479237233243586621)** (5 messages): 

> `Student Discord Server, GPU and LLM Pricing Dashboard, Computer autocomplete NPM packages` 


- **New Student Discord server with VIP backing**: A member is building a Discord server for students to share tips and study tools, backed by a **Duolingo executive**, covering topics like coding and AI workflows, shared at [outsmartdiscord.com/education](https://outsmartdiscord.com/education).
- **Real-Time GPU Pricing Dashboard Launched**: A member built a free dashboard to track real-time **GPU and LLM pricing** across cloud and inference providers, available at [deploybase.ai](https://deploybase.ai).
- **Computer Masters NPM Autocomplete**: A member lauded **Computer** for perfectly autocompleting NPM package names/versions and providing a visual indicator for out-of-date packages, creating a well-structured project available on the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=VoidWorks.trawl).
- **Perplexity Key in Qwksearch**: [Qwksearch](https://qwksearch.com) now allows users to bring their own Perplexity API key.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1479214544168554607)** (478 messages🔥🔥🔥): 

> `GPT 5.4 speed slowness, GPT 5.4 pricing max mode, Cursor OOM crashes indexing repo, Windsurf to Cursor upgrade, Cursor subagents` 


- **GPT 5.4 Snail-Pace Slowness Spooks Speedsters**: Members noted **GPT 5.4** is significantly slower than other models, with one user waiting **30 minutes** for a task to complete, even on a paid subscription.
   - Some suggested tweaking rules to prioritize file reading, lower reasoning levels, or even running the agent in a [sandbox environment](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) to mitigate risks from malicious commands.
- **GPT 5.4 Pricing Ploys Prompted by Pushy 'Max Mode' Predicament**: Users are unhappy that **GPT 5.4** is only available in "Max" mode, believing Cursor is pushing users off legacy pricing by requiring max mode for its **1M context window**.
   - There is a lot of confusion about **context windows** and "Max" modes, with some believing it to only have a **270k context window**.
- **Cursor Crashes Cause Concern During Codebase Compaction**: A user experienced persistent **OOM crashes** with Cursor when opening a specific repository, suspecting a repo index corruption or a memory leak during **repo-level indexing**.
   - Troubleshooting steps included clearing `.cursor` and `.cache` directories, reinstalling Cursor, increasing Node memory and Windows paging file, and adding strict `.cursorignore` rules.
- **From Windsurf to Cursor: A Breeze of Fresh Air?**: One user, after a year of using Windsurf, found Cursor to be *a breath of fresh air*, highlighting fewer errors and more efficient workflows.
   - They mentioned that Windsurf injects numerous system prompts, causing issues, whereas Cursor allows them to *actually get work done*.
- **Subagent Shenanigans: Composer's Consumption Concerns**: Users noted that the built-in subagents in Cursor automatically use the **composer** model, which can consume tokens and, sometimes, be unwanted.
   - One suggestion was to create custom subagents to specify a preferred model, as well as using the command `/create-subagent`.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1479209839606628466)** (3 messages): 

> `CoT Controllability, Codex Security, Codex for OSS` 


- **Chain-of-Thought (CoT) gets new Evaluation Suite**: OpenAI is publishing a new evaluation suite and research paper on **Chain-of-Thought (CoT) Controllability** ([link to paper](https://openai.com/index/reasoning-models-chain-of-thought-controllability/)).
   - The research indicates that **GPT-5.4 Thinking** exhibits low ability to obscure its reasoning, suggesting **CoT monitoring** remains a useful safety tool.
- **Codex Security: New Security Agent Rolls Out**: OpenAI introduced **Codex Security**, an application security agent designed to help secure codebases by finding and validating vulnerabilities, and proposing fixes ([announcement link](https://openai.com/index/codex-security-now-in-research-preview/)).
   - This allows teams to focus on critical vulnerabilities and accelerate code deployment, as showcased in [this demo video](https://video.twimg.com/amplify_video/2029983742056615937/vid/avc1/1280x720/sx7Je_FzQPJAr81B.mp4).
- **Codex for OSS to Support Open-Source Contributors**: OpenAI launched **Codex for OSS** to support open-source software contributors ([announcement link](https://developers.openai.com/codex/community/codex-for-oss)).
   - Maintainers can leverage **Codex** to review code, understand large codebases, and enhance security coverage, as detailed in [this demo video](https://video.twimg.com/amplify_video/2029998126640287747/vid/avc1/1280x720/ZMdqbgIfCNQeqJ0i.mp4).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1479208009090732245)** (311 messages🔥🔥): 

> `ChatGPT performance compared to Claude and Kimi, GPT-5.4 Codex, Seedance 2.0 delay, Concerns about government control over AI companies, Overvaluation of OpenAI` 


- **ChatGPT falls Behind: Community Voices Concerns**: Members express concerns that **ChatGPT** is lagging behind other LLMs like **Claude** and **Kimi**, with one user stating *"What’s going on with ChatGPT, he feels like he start to be behind other LLM like Claude or even Kimi"."
   - A member noted that *"Kimi 2.5 is well beyond ChatGPT’s capability, even the K2 thinking model."*
- **GPT-5.4's Codex is a Code-breaker**: Users report that **GPT-5.4's Codex** is performing worse than **5.3** in coding tasks, with one noting they *"don't think we are getting 5.4 codex"*.
- **Seedance 2.0 Faces the Copyright Censors**: The global release of **Seedance 2.0** is delayed and nerfed, allegedly due to users posting videos containing IP/Copyrighted characters, leading to lawsuits against ByteDance.
   - One member stated *"Seedance 2.0 on the other hand will eventually be released globally!"*, noting it was initially expected to be released on **February 24th**.
- **The AI Arms Race: Autonomy and Accountability**: Discussion emerges around government control over private AI companies and OpenAI signing a contract preventing war crimes and mass domestic surveillance, which Anthropic also desired.
   - One user claimed that the government refused the *"even if law changes"* clause for Anthropic, leading to concerns about potential future government overreach.
- **OpenAI's Valuation: Sky-High or Just Pie in the Sky?**: A user expressed skepticism about OpenAI's valuation, noting that it *"plans (and likely won't deliver) to achieve 2b annual profit by 2029 as a 720b valuation company",* hinting at potential overvaluation.
   - Another member quipped *"Everything’s over valued tbh, except the people. People are way under-valued."


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1479207249510404319)** (106 messages🔥🔥): 

> `GPT-5.4 Native Computer Use, GPT-5.4 performance and steering compared to 5.3 and 5.2, OpenAI product launch weirdness, Problems with image generation using GPT, ChatGPT chat slows down and becomes unusable` 


- **GPT-5.4 Boasts Native Computer Use**: Members discussed what **GPT-5.4's** *native computer-use capabilities* meant, with one member explaining that it *can take over and do things on your computer*, similar to **Claude Code** and **Cowork**.
- **GPT-5.4 Earns Praise for Speed and Steerability**: Users have lauded **GPT-5.4** for its speed, steering capabilities, and improved responses, particularly for text-based work requiring long context understanding, with some preferring it over **5.2** and **5.3**.
   - One user noted *the model replying like you're a person, and not like it's hearing voices in its head (5.3)*.
- **Pricing and Mini-Model Complaints Aired**: Users voiced concerns about recent price hikes and expressed a desire for a mini model to be released.
   - One user said *Not too happy about the price hikes. It's about time we got a mini model*.
- **Image Generation Falls Flat**: A user reported that **GPT** now uses *pixel-editing mode* for image generation, which hinders its ability to perform simple tasks such as adding snow to an image.
   - They asked about the availability of an API or alternative methods for accessing image generation with *repaint mode*.
- **ChatGPT Chat Experiences Sluggishness**: Users are complaining about the **ChatGPT** chat slowing down significantly over time, which makes it unusable.
   - One user suggests the issue stems from **ChatGPT** lacking automatic chat compaction, unlike **Claude** and possibly **Gemini**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1479206939832487936)** (20 messages🔥): 

> `Image Generation prompts, Image generation API for repaint mode, GPT Evaluation of Papers, Prompt engineering courses, Accelerated Iterative Destruction` 


- **Prompt Reveals Skeleton Child Pushing Car**: A member shared a prompt to generate a **3D CGI** rendered skinny human child with **translucent skin** and a **cyan skeleton** visible pushing a rusty vintage car.
   - They noted that some models may not have context for **All-Might** (from *My Hero Academia*), so beware of that.
- **Image generation switches from repaint to pixel-editing mode**: Members discussed that **GPT** used to use repaint mode but now uses **pixel-editing mode**, which prevents it from doing certain simple tasks.
   - They expressed happiness that **Sora** still uses **repaint mode**, but noted that Sora 1 will be discontinued soon.
- **GPT can evaluate papers without training**: Members suggested that you don't need to train a **GPT** to evaluate papers from a rubric.
   - Instead, *just pass the rubric in the prompt and ask it to score each category separately*.
- **AI Engineer shares prompt engineering methodology**: A member shared a methodology for prompt engineering, naming them **Accelerated Iterative Destruction** and **Constraint pattern recognition**.
   - They described the first as *deliberately destroying systems to make them stronger*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1479206939832487936)** (20 messages🔥): 

> `Image Generation prompts, Prompt Engineering Courses, Training GPTs` 


- **Crafting Translucent Skin in 3D CGI**: A member shared a [prompt](https://discord.com/channels/974519860457529424/1046317269069864970/1247310292513685544) for generating a **3D CGI rendered skinny human child** with translucent skin and a cyan skeleton, wearing opaque black shorts and an opaque black tshirt, pushing a rusty vintage car behind the car while a **3D CGI rendered All-Might** takes notes on a clipboard standing in the background, cinematic lighting, urban street setting.
   - They pointed out that *translucent or glass like skin is a key descriptor* for achieving the desired effect.
- **Unlocking Image Features in ChatGPT**: A member inquired how to activate the **image feature in ChatGPT**, including why it appears intermittently and if explanations can come with images.
   - Another member simply stated that you can activate the feature by starting your prompt with *"Create an image:"*.
- **GPT's Pixel-Editing vs Repaint Image Generation Modes**: A member highlighted that **GPT now uses pixel-editing mode** for image generation, unlike in the past when it used **repaint mode**, and that GPT is now unable to do many simple tasks such as adding snow to an image.
   - This member was sad that **Sora 1 will be discontinued soon** because they still use repaint mode.
- **Seeking the Holy Grail: Prompt Engineering Courses**: A member asked for recommendations for the best **prompt engineering course**, but was instead provided with methodologies like *Accelerated Iterative Destruction* and *Constraint pattern recognition*.
   - These methodologies are frameworks for finding where systems break and are named after breaking, namely *Coherence, Relational Invariance, Internal Mediation, Projection*.
- **Training a GPT for Paper Evaluation**: A member inquired about **training a GPT to evaluate papers from a rubric**.
   - Others suggested simply uploading the paper and rubric into the prompt and asking it to score each category separately, justifying the score if possible.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1479564705860157480)** (12 messages🔥): 

> `Tech Industry Complacency, AI Agent Database Wipe, Compute Conference Tickets` 


- **Thorsten Ball Rails Against Tech Complacency**: Thorsten Ball criticizes the tech industry's **lack of urgency**, observing that many companies still use outdated operational models despite rapid advancements in **AI** and team efficiency; the post can be found [here](https://x.com/thorstenball/status/2029846505884901873?s=12).
- **Spacemolt Characters Write Screenplays!**: A member is scaling systems to allow PMs to ship code, and has their **Spacemolt** characters writing screenplays now, as documented in [this Google document](https://docs.google.com/document/d/1Lv6nGH930Rurqp_FkLNv-XmwuS-XX7s3uRXPBT7I9QI/edit?usp=drivesdk).
- **Claude Code's Database Debacle**: Alexey Grigorev recounts how the AI agent **'Claude Code'** accidentally executed a Terraform command that **wiped the DataTalksClub production database** and **2.5 years of course data**, described in detail [here](https://x.com/al_grigor/status/2029889772181934425).
- **Free Compute Conference Tickets Available**: Three complimentary tickets for the **Compute Conference** are available using the code `EQ6VA5` on [Luma](https://luma.com/k6bc82dv).


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/1479684765652095078)** (3 messages): 

> `Tech Companies Stock Incentives, Resignation After Bonuses` 


- **Stock Incentives Squeeze Tech Companies**: A member posted an image arguing that **tech companies can't afford to keep staff they've given stock incentives** to, prompting discussion on the financial implications.
   - Another member suggested this situation may have affected **Block**, while others might need to direct free cash flow to capital expenditures for data center buildout.
- **Bonus Backfire: Employee Quits After Retention Bonuses**: A member shared about a **LinkedIn post** detailing how an employee, upon learning that substantial bonuses were being awarded to retained staff, **submitted their resignation**.
   - No further details were provided about the company or circumstances surrounding the resignation.


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1479596720001257704)** (2 messages): 

> `Creator Economy, Cross-Platform Storytelling` 


- **Wild West of the Creator Economy**: A user posted an image labeled *wild* (no context given) with the link to the [image here](https://cdn.discordapp.com/attachments/822625128843182090/1479596719762178169/image0.jpg?ex=69ac9d53&is=69ab4bd3&hm=62b3b5bb200d487a6351ebb520d4c766e189c237c40be128ffa4d37b18af008c&).
- **Full Picture of the Creator Economy**: A user commented that the **full story of the creator economy** isn't told unless *you count other platforms too*.


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1479564624541253764)** (24 messages🔥): 

> `Product launch videos, Venting illustration goes viral, AI development tools comparison, AI agent deletes production database, Tweet of the year contender` 


- **Launch Videos Lookalike**: Manu Arora questions the current design and aesthetic trends in **product launch videos**, noting a repetitive or formulaic style across the industry in [this tweet](https://x.com/mannupaaji/status/2029882202801221892?s=12).
- **Slaylor's Illustration has Venting Victory**: User @GirlSnailure (Slaylor) shared a creative piece they produced to vent frustration after an encounter with someone blocking their path, which subsequently gained significant viral engagement in [this tweet](https://x.com/girlsnailure/status/2029622733865185657?s=12).
- **Claude Code Clumsily Clears Course Content**: Alexey Grigorev reports that the **Claude Code AI agent** accidentally deleted the **DataTalksClub production database** and its automated snapshots via a Terraform command in [this tweet](https://x.com/al_grigor/status/2029889772181934425?s=12).
- **Harry Eccles' 'Tweet of the Year' Hauls Huge Hit**: A highly engaged Twitter post by Harry Eccles (@Heccles94) from March 2026, posing the question of whether it constitutes the '**Tweet of the year**' with over **67,000 likes** and **785,000 views** in [this tweet](https://x.com/heccles94/status/2029973065954668969?s=12).
- **Philosophy Meme Post Proliferates**: A viral social media post from the account @philosophymeme0, dated March 6, 2026, which garnered significant engagement with over **6,500 likes** and nearly **80,000 views** in [this tweet](https://x.com/philosophymeme0/status/2029925357294604573?s=12).


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1479232806641991862)** (4 messages): 

> `$BE stock, saeris.gg` 


- **$BE Stock Attracts Attention**: A member has been monitoring **$BE stock**, referencing [saeris.gg](https://saeris.gg) and [two related tweets](https://vxtwitter.com/josephpolitano/status/2029916364664611242) and another [tweet](https://vxtwitter.com/byheatherlong/status/2029918420821758134).
- **Brief Discussion on $BE**: Another member replied *lol* in response to the report of the stock.


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1479225233096577076)** (10 messages🔥): 

> `AI Consulting, Agentic AI, Production ML Systems, Open Claw for GTM, LLM Engineering Platform` 


- **New LLM Engineering Platform Arrives**: Soren announced the launch of [to11.ai](https://to11.ai), an **LLM Engineering Platform** offering observability, prompt management, gateway services, and security features.
- **Open Claw hunts GTM for Signal**: Steve is working on an **"Open Claw" for GTM** that deeply understands a product to execute specific strategies, such as GEO optimization and sourcing ICPs on LinkedIn.
   - It **hunts for signal posts on Reddit/X** too.
- **Agentic AI company focuses on Executive Decision-Making**: Debo started an **agentic AI company** focused on executive decision-making.
   - He is here to learn more about **real use cases**.
- **Orchestrator Scales Adoption of Vanilla Code**: A member is *writing a book on Scaling AI adoption in Engineering*, host O'Reilly CTO Hour, facilitate the executive summits for CNCF at KubeCon twice a year, hosting my Gather.dev events for Founding, Startup and Scale CTOs in NY, Bay area and online, and (like everyone else in the world) building my own orchestrator to run my business, research the book, and organize my life.
   - They use Vanilla claude code, single repo with a space for shared context, one directory for each **employee** agent with their prompts and unique context, and another directory for each **advisor** agent.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1479215172961697833)** (23 messages🔥): 

> `Web Middleware Parallelization, PlanetScale Latency, TanStack Intent for AI Agents, Steam Hardware Delays and Exabyte Traffic` 


- **Web Middleware: Parallel Auth Checks?**: Members discussed a parallel web middleware concept for running **auth/access control** checks concurrently with rendering, potentially stopping rendering if auth fails.
   - However, concerns were raised about increased complexity in separating UI trees and potential issues with side effects during rendering, with one member linking the design to **Next.js**' aggressive parallelization causing a cognitive footgun.
- **PlanetScale: New DB Latency King?**: A user shared their [improvement in performance after migrating from **AWS** to **PlanetScale**](https://xcancel.com/fforres/status/2029661853731934629?s=20), showcasing latency dropping from **255ms to 10ms**.
   - Others responded that with *machines in a single datacenter connected over a private network* they achieved **0.1ms latency** and they joked that *10ms unstable db latency is a brag now*.
- **TanStack Intends to Ship AI Agent Skills**: **TanStack** announced [Intent (alpha)](https://xcancel.com/tan_stack/status/2029973163455766769), a pipeline for shipping **AI agent-readable 'skills'** directly within npm packages.
   - The system facilitates distributed, auto-discovered, and up-to-date knowledge syncs that stay current with library updates across all major package managers.
- **Valve Delays Steam Machine Amidst Exabyte Traffic**: **Valve**'s "year in review" blog post specified they *hope to ship* the **Steam Machine** and other announced hardware sometime this year, likely due to the **RAM shortage**.
   - The post revealed that **Steam** delivered about **80 exabytes** to customers in 2024, growing to **100 exabytes** in 2025, averaging **274 petabytes** of installs and updates per day, equivalent to **190,000 GB** of data per minute ([source](https://steamcommunity.com/groups/steamworks/announcements/detail/528746884222682053)).


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1479259576913100871)** (10 messages🔥): 

> `Y Combinator's Startup School, Compute Conference, AI Infrastructure, Developer Tooling` 


- **YC Startup School Still Slaying**: A member reminisced fondly about [Y Combinator's Startup School](https://events.ycombinator.com/startup-school-2026), noting its impact on their life.
   - They admitted to not fully leveraging the opportunity but acknowledged that it significantly changed their life and its online resources remain helpful.
- **Daytona's Compute Conference**: **Daytona** is hosting **Compute**, a conference focused on **AI infrastructure**, **agents**, and the next generation of **cloud**, taking place **March 8-9** at the **Chase Center** in **San Francisco** ([Compute Daytona](https://compute.daytona.io/)).
   - Featured speakers include **Aaron Levie** from **Box**, **Parag Agrawal** from **Parallel**, and **Harrison Chase** from **LangChain**, among others, targeting engineers, founders, and builders in **AI infra** and **developer tooling**.


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1479607046184636537)** (1 messages): 

> `GitHub Social Club, Amsterdam Event` 


- **GitHub Hosts Social Mixer in Amsterdam**: GitHub is hosting a **GitHub Social Club: Amsterdam** on **Monday, March 23**, preceding Kubecon + CloudNativeCon and AgenticDays.
   - The event is described as *a low-key hangout for devs, builders, researchers, founders, and open source fans* and promises no pitches, offering a space to connect and share ideas, according to [the event page](https://luma.com/githubsocialclub-amsterdam).
- **GitHub Swag Alert**: Attendees of the **GitHub Social Club** in Amsterdam will receive **GitHub swag**.
   - The event promises coffee, snacks, and a chance to meet GitHub team members, making it a good opportunity to network.


  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1479533590130851872)** (1 messages): 

> `NYC Meetup, Google NYC Hosting, Talks from Google, Modal, and others` 


- **Google NYC Hosts Another Meetup**: A member announced they are organizing a meetup in a few weeks, hosted by **Google NYC**, featuring talks from **Google**, **Modal**, and the organizer's employer; details and registration are available on [Luma](https://luma.com/7qxvd38s).
   - No further details were provided.
- **Diverse Tech Firms to Present**: The meetup promises a diverse range of tech perspectives with speakers from **Google**, cloud computing platform **Modal**, and the hosting member's company.
   - The specific topics and focus of each presentation remain to be seen, generating anticipation within the community.


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/1479691298049884271)** (3 messages): 

> `Backdoored Training Data, Alexander Long Tweet` 


- **Alexander Long's Tweet goes Viral**: A member shared a link to [Alexander Long's tweet](https://x.com/alexanderlong/status/2030022884979028435?s=12).
   - Another member speculated whether *someone backdoored their training data*, or something even more explicit.
- **Speculation on Training Data Backdoors**: Following the link to the tweet, a member inquired about the possibility of backdoored training data.
   - The member questioned whether the issue was due to a *backdoor* or *something more explicit*.


  

---


### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1479206949026267349)** (98 messages🔥🔥): 

> `CSS dark/light mode, Trump's White House UFC Stadium Proposal, Iran-Saudi relations, Palantir's Maven Smart System, Anthropic AI contract with Department of War collapse` 


- **CSS Debates Spark Over Dark Mode Implementation**: A discussion started over Twitter's move to OS-controlled dark mode, with members debating the need for separate assets vs. CSS variables for palette swaps and ways to counteract **blooming effects** in dark mode.
   - One member recommended using the `light-dark()` CSS syntax with CSS variables to combine light and dark mode color pairings, as shown in [this article](https://web.dev/articles/light-dark), and another shared his sentiment *"anytime they do shit like this it makes me wonder, did Elon mandate this change? Or is it because Grok produces absolute slop?"*
- **Trump Plans White House UFC Stadium**: Reports indicate **Donald Trump** plans to build a **100,000-seat stadium** near the White House to host a **UFC event** on his birthday in **June 2026**.
   - The proposal, originally shared in [this tweet](https://xcancel.com/highbrow_nobrow/status/2029497418325086488), was met with mockery and sarcastic remarks.
- **US Investigation Points to Likely Responsibility for Iran School Strike**: A US investigation suggests likely US responsibility in an **Iran school strike**, amid rising tensions and skepticism regarding the US's ability to defend its allies from Iran, according to [this Reuters article](https://www.reuters.com/world/middle-east/us-investigation-points-likely-us-responsibility-iran-school-strike-sources-say-2026-03-06/).
   - Some members pointed out that the region is very upset with the US and cited macro analysis suggesting the potential of investment withdrawal from gulf countries, based on this [YouTube analysis](https://www.youtube.com/watch?v=jIS2eB-rGv0).
- **Department of War and Anthropic AI Partnership Collapses**: An article shared [here](https://xcancel.com/piratewires/status/2029984469093118185?s=12) details how a major contract between the **Department of War** and **Anthropic AI** fell through due to restrictive terms prohibiting kinetic strikes, long ethics panel reviews, and concerns about ideological supply-chain risks.
   - The member satirically noted *"seems like open ai is ahead of anthropic in vibe warcrime"*.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new Cursor pod! https://www.latent.space/p/cursor-third-era
  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1479225578329473191)** (117 messages🔥🔥): 

> `Multi-Agent Orchestration, Claude Code Reverse Engineering, Greptile Agent v4, Cursor Automations, ChatGPT for Excel` 


- **Claude Code Cracked for Context Control**: A developer reverse-engineered the **Claude Code binary** to implement a surgical context management feature, allowing users to selectively strip tool calls, results, and thinking blocks while preserving the core message history, as detailed [on xcancel.com](https://xcancel.com/vicnaum/status/2029579972688379928).
- **Greptile Agent v4 Slashes Bugs, Hikes Prices**: **Daksh Gupta** launched **Greptile Agent v4**, boasting improved bug detection and fewer false positives, but with a revised pricing structure aimed at power users, as seen [on xcancel.com](https://xcancel.com/dakshgup/status/2029587555268845692?s=12).
   - A user commented that *those prices are eye-watering!*.
- **Cursor Automates Always-On Agents**: **Cursor** unveiled **Cursor Automations**, a new feature to create and deploy persistent, always-on AI agents within the platform, according to [xcancel.com](https://xcancel.com/cursor_ai/status/2029604182286856663?s=12).
- **Sarvam AI Drops Indian Language Models**: **Pratyush Kumar** announced the release of the **Sarvam 30B and 105B models**, trained from scratch to excel in Indian languages and global benchmarks, as detailed [on xcancel.com](https://xcancel.com/pratykumar/status/2029965547824431356?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - Weights are available on **Hugging Face** and **AIKosh**, with **SGLang** providing launch day support, and **vLLM** integration expected soon.
- **GitHub Bot Gets Promptly Hacked**: **Sash Zats** reported a security breach where an attacker obtained an npm token using a prompt injection in a GitHub issue title, exploiting a triage bot, as detailed [on xcancel.com](https://xcancel.com/zats/status/2029888470383051053?s=12).


  

---


### **Latent Space ▷ #[berlin](https://discord.com/channels/822583790773862470/1095237457722744932/1479607060319699087)** (1 messages): 

> `GitHub Social Club, Amsterdam Events, Kubecon, CloudNativeCon, AgenticDays` 


- **GitHub Social Club Coming to Amsterdam**: GitHub is hosting a **GitHub Social Club: Amsterdam** on **Monday, March 23**, preceding **Kubecon + CloudNativeCon** and **AgenticDays**.
   - The event is a low-key hangout for devs, builders, researchers, founders, and open-source enthusiasts to connect and share ideas, with [RSVPs available here](https://luma.com/githubsocialclub-amsterdam).
- **Networking Opportunity for Developers**: The GitHub Social Club offers a space for developers to connect, share ideas, and swap stories with others in the community.
   - Attendees can expect coffee, snacks, GitHub swag, and a chance to meet with members of the GitHub teams.


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1479261057892614216)** (12 messages🔥): 

> `Reasoning Models, Structured Checklist Method, AI Koding` 


- **Reasoning Models for Control**: OpenAI highlighted [reasoning models](https://openai.com/index/reasoning-models-chain-of-thought-controllability/) to improve **chain of thought controllability**.
   - This is potentially useful if doing rubric maxxing with the [COVAL alignment project](https://alignment.openai.com/coval/).
- **Meta's Checklist Slashes Errors by 50%**: Meta researchers found that using a structured checklist template reduces error rates in **code patch verification** by nearly **50%** without additional fine-tuning or architectural changes, as seen in [this tweet](https://xcancel.com/alex_prompter/status/2029861760455569422?s=12).
   - The approach involves forcing step-by-step evidence and reasoning before concluding which could solve AI koding.
- **Databricks Ships KARL for Custom RL**: Databricks introduced **KARL**, a faster agent for enterprise knowledge-powered custom RL, as described in [this blog post](https://www.databricks.com/blog/meet-karl-faster-agent-enterprise-knowledge-powered-custom-rl).
   - This enables more efficient and customized reinforcement learning applications within enterprise environments.


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1479297785688756435)** (61 messages🔥🔥): 

> `Codex App, GPT-5.4 Token Usage, AI-First OS, Prompt Engineering, 1M Context Usage` 


- **Codex App Demands Double Token Burn Rate**: Members reported that using the **Codex app** results in burning through usage **2x faster** for the same number of tokens.
   - Despite the increased cost, some users find **GPT-5.4 xhigh** to be significantly faster than **5.2 xhigh**, though impressions on quality varied; one user noted, *"5.4 seems to eat context window faster, again, vibes"*.
- **New AI-First OS Under Construction**: A user is *gently recoding an LLM based OS in the browser* and linking it with [wesen-os](https://github.com/wesen/wesen-os) and [workspace-links](https://github.com/wesen/wesen-os/tree/main/workspace-links) on Github.
   - They argue that *we're at a point where we can rethink everything about computers* and *break the shackles of abstractions past*.
- **Upcoming Speakers Announced**: The **AI In Action Bot** announced upcoming speakers, including @slono on March 6, 2026, presenting *"it's GO GO OS - THE AI FIRST OS"*, and @beeradley on March 13, 2026, discussing a *"new Latent Space DB and Bot"*.
   - The bot also mentioned scheduling Peter Bell for March 20, but this requires additional input from the user, as *"Trace if you still want to do this you need to reply to the bot questions until it confirms the date"*.
- **Prompt Engineering Diamond Tier Uncovered**: Members highlighted the effectiveness of the prompt *"proceed"* as **diamond tier**, while *"gitrdun"* was considered **mud tier**.
   - A user suggested a more elaborate prompt: *`proceed until completed and verified`*, but another noted that *i suspect asubtle change in compaction prompt that causes it to drop stuff like that as it carries over*.
- **Context Limit Configuration Tricks Revealed**: A user inquired about using the **1M context window**, and another shared a configuration tip to increase the limit by changing `model_auto_compact_token_limit = 960000` in `.codex/config/toml`.
   - They confirmed that this configuration change is working in their setup (presumably Codex in the 'pi' environment).


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1479212149753647199)** (14 messages🔥): 

> `Arksim for agent testing, Agent-to-agent Slack, Cursor Cloud Agents, Reads memory layer for multiagent swarms, Encrypted decentralized memory for agents` 


- ****Arksim** Open Sources Agent Autoeval Tooling**: A new tool called **Arksim** was open sourced to generate synthetic users that run conversations against your agent automatically, [addressing gaps in manual test cases](https://github.com/arklexai/arksim).
   - The tool aims to surface failures before real users encounter them, and is available via `pip install arksim` with [documentation available online](https://docs.arklex.ai/overview).
- **Agents now have **Slack** to Argue in**: An early version of "slack for agents" has been released, enabling agents to argue with each other like real colleagues at [ats.sh/new](https://ats.sh/new).
   - The aim is to simulate messy but productive interactions, allowing agents to *figure things out* collaboratively.
- ****Cursor** Enters Cloud Era**: A discussion around [Cursor's Third Era: Cloud Agents](https://youtu.be/tMflcZHo2zI) highlighting how more code produced by agents can lead to exponential code generation via parallel runs and comparative implementations.
   - The video shows the Jevons paradox in action, demonstrating an increase in code production correlating with agent capabilities.
- ****Reads** Memory Layer for Multiagent Swarms Orchestration in Science**: A memory layer called **Reads** has been developed to aid multiagent swarms in orchestrating scientific research tasks, with a [GitHub repo available](https://github.com/reads-project/reads-ts).
   - A full demo with a frontend is expected soon, preserving high-compute output effectively.
- ****ElectricSQL** Launches Agent SKILLs for Vibe Coding**: **ElectricSQL** has introduced Agent SKILLs for Electric & Durable Streams clients and TanStack DB, enhancing the 'vibe coding' experience and enabling developers to generate error-free applications rapidly, originally shared on [X](https://xcancel.com/kylemathews/status/2030058969822367784).
   - This update focuses on allowing complex applications with single attempt generation of code.


  

---


### **Latent Space ▷ #[san-diego-neurips-2025](https://discord.com/channels/822583790773862470/1335732885717651558/1479605627729743993)** (2 messages): 

> `` 


- **No Discussion Occurred**: There was no discussion in the provided messages to summarize.
   - The user expressed disappointment about missing something but provided no context.
- **Nothing to Summarize**: The provided text consists of an incomplete sentence and lacks substantive content.
   - Therefore, no meaningful topics or discussion points could be extracted.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1479308885553840229)** (3 messages): 

> `Ben Affleck AI Video Startup, Netflix Acquisition, Interpositive, ComfyUI` 


- **Affleck's Interpositive Acquired by Netflix**: Ben Affleck has been running an **AI video startup** called **Interpositive** since **2022**, and it was just acquired by [Netflix](https://about.netflix.com/en/news/why-interpositive-is-joining-netflix).
- **ComfyUI's Pervasive Use Questioned**: After watching a short interview, a member inquired whether **ComfyUI** is being used everywhere.
   - The member sought to confirm if **ComfyUI** is the standard tool across the industry.


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1479299514102845470)** (4 messages): 

> `GPT 5.4 Solves Math Problem, Bartosz Naskrecki, Move 37, Singularity in Science` 


- **GPT 5.4 Achieves Math 'Move 37' Moment**: Mathematician Bartosz Naskręcki reports that an advanced AI model, **GPT 5.4**, solved a problem he had curated for two decades, leading him to declare that *the singularity in science has arrived*.
   - The link to the full post is [here](https://xcancel.com/trajektoriePL/status/2029660475395326300) on X.
- **Mathematician Hails Scientific Singularity**: Bartosz Naskręcki, a mathematician, claims that **GPT 5.4's** solving of a long-standing problem signifies the arrival of a singularity in the scientific domain.
   - This conclusion is based on the AI's unexpected solution to a mathematical challenge Naskręcki had been developing for **two decades**.


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1479591469214990338)** (4 messages): 

> `Far.AI, Neel Nanda, Empirical Interpretability, Activation Steering, AGI Safety` 


- **Far.AI Signals Interpretability Pivot**: **Far.AI** discusses **Neel Nanda's** strategic shift toward **empirical interpretability** as outlined in [this tweet](https://xcancel.com/farairesearch/status/2029957875523592524).
- **Activation Steering Gets Far.AI Nod**: The focus has moved from abstract insights to **testable proxy tasks** and **activation steering**, prioritizing methods that demonstrate measurable impact on **AGI safety**.


  

---


### **Latent Space ▷ #[dev-writers-retreat-2025-dwr](https://discord.com/channels/822583790773862470/1445650211694448714/)** (1 messages): 

xoxoxoxo42: congrats!!
  

---


### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1479267257103290481)** (2 messages): 

> `Breaks, Work-life balance` 


- **Breaks may be needed**: One member suggested that another member may need to take a break due to the workload.
   - The context implies potential overwork.
- **Work-Life Balance Check-in**: A check-in was initiated, possibly to gauge workload and stress levels.
   - This suggests a focus on accountability and well-being within the team.


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/)** (1 messages): 

kevin_85537: Fascintating!
  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1479331447004336210)** (5 messages): 

> `AI Demo, AI in Action` 


- **AI Engineer Previews Impending Demo**: An AI Engineer announced they are [planning to showcase a demo](https://example.com/demo) of their work tonight, hoping to assemble various unfinished projects.
   - *I hope I can put all my junk of half finished stuff together into a good demo.*
- **Demo Location Announced**: Following up on the initial announcement, the location of the demo will be *online in 1h30 in ai in action*.
   - Details forthcoming, stay tuned.


  

---


### **Latent Space ▷ #[euno-log](https://discord.com/channels/822583790773862470/1473750131441668096/1479505361424879668)** (3 messages): 

> `GitHub Social Club: Amsterdam, Discord Stats Load Failure` 


- **Amsterdam GitHub Social Club**: GitHub is hosting a [GitHub Social Club in Amsterdam](https://discord.com/channels/@me/1479607069501030579/1479607072852148236) on Monday.
- **Discord Stats Load Failure**: Multiple messages indicated that Discord stats failed to load.
   - The issue was reported across different channels, suggesting a potential widespread problem.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1479211677026226429)** (15 messages🔥): 

> `Chat Web Client, Customize Themes` 


- ****Zoltun** Launches Chat Web Client**: A member shared their chat web client called **Zoltun** at [zoltun.org](https://zoltun.org/) and [github.com/zoltun-org](https://github.com/zoltun-org) as an alternative to the **GLM Chat Web Client**.
   - This customizable client features autosave and markdown functionality optimized for reading.
- **New UI direction is eye-catching**: A member complimented **Zoltun** for its *bold and eye-catching UI direction* that sets it apart.
   - The creator of **Zoltun** is trying to find a middle ground between modern and vintage, and allows users to customize themes.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1479210541179338774)** (202 messages🔥🔥): 

> `Stolen Accounts & Unauthorized Transactions, Mini Tavern alternative, GPT-4 Availability in China, Router Configuration after flashing, 403 Error with Google Gemini Models` 


- **OpenRouter Hit by Account Heists, Funds Fly!**: Users are reporting **stolen accounts** and **unauthorized transactions**, with one user noting they have filed a complaint with their bank and are awaiting a response from OpenRouter support at `support@openrouter.ai`.
   - Another user expressed concern about a *bad actor* potentially transferring funds through multiple accounts or changing emails, making tracking harder, and highlighting the risk of **API key leaks**.
- **MiniTavern App Gets the Thumbs Up?**: A user asked about better alternatives to **MiniTavern** ([https://apps.apple.com/us/app/minitavern-tavern-roleplay/id6748523919](https://apps.apple.com/us/app/minitavern-tavern-roleplay/id6748523919)), with another simply responding *yes*.
- **Gemini's Geoblocking: Russia Gets the Cold Shoulder**: A user encountered a *403 Blocked by Google* error when accessing **Google Gemini models** through OpenRouter, despite having a positive account balance.
   - It was pointed out that Google blocks API access from Russia ([https://ai.google.dev/gemini-api/docs/available-regions](https://ai.google.dev/gemini-api/docs/available-regions)), which was confirmed to be the user's location but the user mentioned they're working through Germany, with VPNs.
- **Router Flashing Fiasco**: A user requested help with early configuration after flashing their router, reporting they could no longer connect via cable or wifi.
- **LLMs Turn to Scripting Shenanigans**: A user reported issues with LLMs writing python scripts to print their responses instead of directly writing them out, even when explicitly instructed not to.
   - The unusual behavior was attributed to models trained on **synthetic data**, and adding **examples** might alleviate the issue, and pointed to a [manus article](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) on agentic systems.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1479268002816983040)** (10 messages🔥): 

> `Model Inference Quality, Prompt Publishing, Agents.md, Elon Musk, Anthropic` 


- **Inference Quality Must Prevail**: A member noted that model usage is acceptable only if the **model/inference quality** is not **5x worse**.
   - They questioned whether this consideration applied to *prompt publishing* and *public prompt ridiculing*, as well as a *weekly prompt-book club*.
- **Strategic Context Windows and AGENTS.md Discussed**: A member advised that *less is more* and to *always be strategic* when it comes to the **context window** and **AGENTS.md**-like files.
   - They linked to [Evaluating agents.md Are Repository-linker.sh](https://arxiviq.substack.com/p/evaluating-agentsmd-are-repositorylinker.sh) for more information.
- **Musk's Actions Draw Criticism**: Members reacted negatively to [this tweet](https://x.com/elonmusk/status/2029833177368514831) by **Elon Musk**.
   - One member speculated that **Musk** is *salty* because **Anthropic** declined his offer to use his model without restrictions, allegedly because *his model sucks*.
- **Microsoft Keeps Anthropic Available After Security Concerns**: A member linked to a [CNBC article](https://www.cnbc.com/2026/03/05/microsoft-says-anthropics-products-can-remain-available-to-customers-after-security-risk-designation.html) about **Microsoft** allowing **Anthropic's products** to remain available despite *security risk designation*.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1479207218321690827)** (208 messages🔥🔥): 

> `GPT Pro council, Cursera prompt injection, Extensible RL framework, Hermes Agent skins, GPU pricing` 


- **GPT Pro is speculated to be an AI council**: There is speculation that **GPT Pro** is actually a council of **8 AIs**, with **7** generating answers and **1** deciding, to achieve a higher and more reliable result.
   - It was noted that **GPT Pro** is priced **10x** higher than the standard GPT, fitting the council model perfectly, though this is just speculation.
- **Coursera faces prompt injection attempt**: Someone on LinkedIn found a prompt injection vulnerability in Coursera's system, where the AI is supposed to uphold academic integrity and not provide answers to assessments, however it did not work.
   - The AI assistant is disabled on assessment pages with a message: *To uphold Coursera's academic integrity policy, this AI assistant is disabled on assessment pages.*
- **Extensible RL Framework**: A member is seeking an extensible **RL framework** to build into their software, considering the use of reward functions defined by **LLMs**.
   - The goal is to create an end-to-end omnimodal annotation/training system, potentially based on **GRPO**.
- **Hermes Agent gets custom skins**: A member is working on custom **Hermes Agent skins**, showcasing early iterations with themed graphical user interfaces.
   - The member is matching the TUI theme, and making GUI adjustments.
- **Sky high GPU prices are a concern**: A member is concerned about the high cost of renting **GPUs** for finetuning, questioning the feasibility of such projects in the current market.
   - They are seeking providers with good deals as **GPU pricing** are too high.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1479283863019851988)** (20 messages🔥): 

> `CUDA memory programming, AMD kernel hackathon, ML Eng / ML Ops intern position, Nvidia Compute Conference tickets` 


- **CUDA Memory Programming Tutorial Recommended**: A member recommended a [CUDA memory programming tutorial](https://siboehm.com/articles/22/CUDA-MMM) as the *best starting point for beginners*.
   - They noted it has *good coverage* of GPU memory programming.
- **AMD Kernel Hackathon Announced**: Members discussed the recently announced **AMD kernel hackathon**, with one member considering participation despite being new to CUDA and currently optimizing **softmax**.
   - A member encouraged participation for the learning experience, noting that it might be specifically for **AMD chips**.
- **ML Eng / ML Ops Internship Position Search**: A member sought help finding a summer **ML Eng / ML Ops internship** for a **University of Waterloo** student after their company, **FableTherapeutics**, rescinded the offer.
   - Another member allowed posting of the intern's [LinkedIn profile](https://www.linkedin.com/in/mramamon/) and shared an internship listing at **Microsoft** for **RLM research**.
- **Free Nvidia Compute Conference Tickets Shared**: A member shared **3 complimentary tickets** for the **Nvidia Compute Conference** using the code `EQ6VA5` on [Luma](https://luma.com/k6bc82dv).
   - Another member is looking for a teammate for the **AMD kernel competition**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1479401956656156783)** (21 messages🔥): 

> `NVL72 H2H Copies, Qwen3.5 MoE Megakernel, AMD competition, HIP performance, Blackwell FP16 throughput` 


- ****NVL72** H2H Copies Questioned**: A member asked if **NVL72** supports **H2H copies** over **NVLink**, specifically if a handle containing host pinned memory can be used by another host via the Copy-Engine to move data.
   - No answers were provided.
- **Qwen3.5 MoE Megakernel Project Proposed**: A member proposed working on a **megakernel** for **Qwen3.5 MoE**, noting the complexity due to **MoE**, required **nvfp4** (due to a **32GB** limit), and a hybrid architecture.
   - Another member expressed interest but cited other commitments, mentioning that the **GDN part** for decode is not too complicated, but **MoE** is annoying on small GPUs.
- ****HIP** submissions showcased**: A member shared that most of the high-performance submissions to [gpumode.com](https://www.gpumode.com/home) have been in **HIP**.
   - They also linked to [William's recent talk on hipkittens](https://www.youtube.com/watch?v=OkFk-7Mk6qI) to get others up to speed quickly.
- ****Blackwell's** FP16 Throughput Pondered**: A member questioned why, in the [Blackwell RTX architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), **FP32** non-tensor **TFLOPs** are the same as **FP16** non-tensor.
   - Another member linked to the [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions) clarifying that some GPUs have higher **fp16** throughput, while others don't.
- ****nvDecoder cuvidCreateDecoder** mysterious crash**: A member reported that **nvDecoder cuvidCreateDecoder** crashes when running decode on it with **h264**.
   - The error code returned is **999**, which is a mystery crash.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1479531953404252383)** (1 messages): 

> `Kernel Competition, AMD Sponsorship, DeepSeek-R1-0528 Optimization, GPT-OSS-120B Optimization, MI355X Optimization` 


- **AMD Sponsors Kernel Competition, Offering $1.1M**: A new kernel competition is now open for submissions with a **$1.1M** cash prize, sponsored by **AMD**, focused on optimizing **DeepSeek-R1-0528** and **GPT-OSS-120B** on **MI355X**; registration is available at [luma.com](https://luma.com/cqq4mojz).
- **Competition Split into Two Phases**: Phase 1 (March 6-30) involves optimizing three kernels: **MXFP4 MoE**, **MLA Decode**, and **MXFP4 GEMM**, with submissions via gpumode.com.
   - In Phase 2 (March 31-May 11), top teams from Phase 1 will collaborate with **AMD** and **GPU MODE** engineers to upstream kernels into popular inference engines.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

jaefosho: Reading this now, do you have others?
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1479291063620603904)** (11 messages🔥): 

> `Programming Massively Parallel Processors, CUDA, C++ for CUDA, PyTorch Helios hackathon, Popcorn CLI` 


- **"Programming Massively Parallel Processors" still goated**: "**Programming Massively Parallel Processors**" book is still recommended as a top resource, even with others like Inference Engineering, AI Systems Performance Engineering, and books from Chip Huyen.
   - A member reaffirmed it's still the go-to book.
- **CUDA Resources Confirmed**: The **Nvidia CUDA programming guide** and "**Programming Massively Parallel Processors**" are top resources for learning **CUDA** programming.
   - No further details were given.
- **C++ Basics Enough for CUDA**: Knowing **C++ basics** from undergrad is a solid foundation for starting **CUDA**, emphasizing comfort with pointers and manual memory management (**malloc** and **free**).
   - Complex **C++ features** like **STL** or **std::vector** are typically not used in code running on the GPU, with focus on manually moving data between host RAM and the GPU device; An **RTX 4050** is sufficient to start.
- **Helios Hackathon Welcomes Beginners**: The PyTorch Helios hackathon is open for beginners to attend and observe, even without kernel hacking experience.
   - No further details were given.
- **Popcorn CLI Submissions Don't Require Specific Hardware**: The **Popcorn CLI** allows submitting kernels for remote machines without needing specific hardware like an **MI355X** to participate in the competition.
   - Direct **SSH access** will be provided to teams for phase 2.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

jaefosho: This is a reach, but is there anything in Georgia (the state).
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1479673034607558687)** (3 messages): 

> `AMD Kernel Dev Competition, MI355X Access, Popcorn CLI` 


- ****MI355X** Access Quest Begins**: A member inquired about the correct channel for the **AMD kernel dev competition** and where to rent access for **MI355X**.
   - Another member confirmed it's the right channel while another suggested using `popcorn (or popcorn-cli) submit solution.py` and then *some menu will appear*.
- ****Popcorn CLI** Gets a Shoutout**: To participate in the competition a member recommended using the [**Popcorn CLI**](https://github.com/gpu-mode/popcorn-cli).
   - The CLI allows participants to submit `solution.py` which opens a menu to guide the submission process.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1479233237439086593)** (1 messages): 

> `Blackwell Consumer Chip, Kernel Level Tweaks, Consumer Chip Possibilities` 


- **Blackwell Consumer Chip Excitement Builds**: Enthusiasts anticipate significant possibilities for learning **Blackwell** using a consumer chip.
   - However, serious **kernel level** and **tuning tweaks** require the real hardware, mirroring findings from the kernel competition.
- **Kernel Tweaks on Real Blackwell**: Serious **kernel-level** optimizations for Blackwell necessitate using the actual hardware.
   - Experiences from a kernel competition underscored that consumer chips are insufficient for advanced tuning and low-level system adjustments.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1479550130540515499)** (30 messages🔥): 

> `AMD Kernel Competition Prize Pool, GPU Credits for AMD Developer Account, Popcorn CLI Submission System, Submission Errors and Work on Other Streams, AMD Kernel Competition Submission Information` 


- **AMD Competition has "Insane" Prizes**: The AMD kernel competition features a huge prize pool, but the dropoff from 1st to 2nd place is significant, as mentioned in [this Reddit thread](https://www.reddit.com/r/fastandfurious/comments/1e7z0eh/it_doesnt_matter_if_you_win_by_an_inch_or_a_mile/).
- **No GPU Credits Needed for Kernel Competition**: Participants inquired about GPU credits for the AMD developer account, but were informed that **no GPU credits are needed** to participate in the competition, and that they can use the [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) queue-based system for submissions.
   - The previous grand prize winner apparently *never even rented a GPU for the competition*.
- **Naive Check for Code Containing Work on Another Stream**: Users reported receiving a `500` error during submissions related to working on another stream and a suggestion was made to remove the word *stream* from the code to bypass a naive check.
   - One user said *it's even more crazy than the nvidia one*, but another replied *but also significantly more difficult*.
- **AMD Kernel Submission Information and Limitations**: Links were provided for actual good information on how and what can be submitted, including limitations and the environment the code will run in: [reference kernels](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602), [popcorn-cli](https://github.com/gpu-mode/popcorn-cli), and [AMD kernel official reference](https://github.com/ROCm/aiter/tree/main/aiter/ops).
- **Ensuring Competition Honesty**: Organizers emphasized that honesty is mandatory and they will continuously check submissions for compliance with the rules, welcoming participants to discuss questions or concerns about compliance in the group.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1479295711374741504)** (3 messages): 

> `Colfax Blackwell GEMM tutorial, Blockscaled GEMM, sm_103 K-mode` 


- ****Colfax** Drops New **Blackwell GEMM** Tutorial**: Colfax released the latest installment in their **Blackwell GEMM** tutorial series, focusing on [blockscaled GEMM](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/).
   - This tutorial aims to provide insights into hardware-supported block scaling with **NVIDIA Blackwell GPUs**.
- **Fifth Combination Missing From Table**: A user noted that the fifth combination (**E2M1**, vector length 16, **UE8M0**) appears to be missing from the table in the tutorial.
   - This could be a potential oversight in the documentation that needs correction.
- ****sm_103** K-Mode Expands**: With **sm_103**, the K-mode is no longer restricted to **32B**, as it now supports dense **fp4** with **K=96**.
   - This expansion allows for more flexibility in memory access patterns and data formats.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1479584579911155943)** (14 messages🔥): 

> `Discord Widget, Shields.io, Server Linking, Discord Badge` 


- **Discord Widget Activation for Shields.io Badge**: A member requested enabling the Discord widget setting on the server for a [Shields.io badge](https://shields.io/badges/discord) to direct readers for questions/comments, and another member confirmed enabling it.
   - The shields.io badge will display the user count, and can markdown link the badge to the <#1373414141427191809> channel.
- **Destination Channel Dilemma**: A member inquired whether the Shields.io badge should link to a random channel or to the teenygrad channel.
   - It was suggested that, given the server-wide setting, it should link to <#1189498205101109300> for general reusability or <#1189557310998200410> based on discretion.
- **Discord Badge preference surfaces**: A member expressed familiarity with a specific [Discord badge](https://github.com/gpu-mode/resource-stream#gpu-mode-resource-stream), finding it visually superior as it includes the Discord icon.
   - Another member agreed that it is a better looking badge.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1479556871764443177)** (6 messages): 

> `Heroku server issues, POPCORN_API_URL update, Invalid X-Popcorn-Cli-Id` 


- ****Heroku** Server Woes? **POPCORN_API_URL** to the Rescue!**: Users experiencing *Heroku server not found* errors should ensure their **POPCORN_API_URL** points to the updated address: [https://site--bot--dxfjds728w5v.code.run](https://site--bot--dxfjds728w5v.code.run), as detailed in the readme.
   - Reinstalling might resolve the issue, as the update should have been included.
- **Bypassing **Popcorn** Install? Watch Out for **API_URL**!**: A user who bypassed the install due to an *Invalid or unauthorized X-Popcorn-Cli-Id* error by building the binary manually found that their local **POPCORN_API_URL** in **.bashrc** was hardcoded to the old Heroku URL.
   - It was suggested that others send a PR if they also encounter the issue after manually installing **Popcorn**.
- **Clean Install is Key for **Popcorn**?**: One user resolved the issue by performing a clean install (wiping **.popcorn.yaml**), setting the new **POPCORN_API_URL**, and re-registering for a new **popcorn.yaml** key.
   - The user believes that the problem stemmed from old configurations on their machine, suggesting a clean slate might be necessary.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1479214522215436450)** (4 messages): 

> `NVlink XID errors, NCCL test for retransmits, Deterministic Algorithms in NCCL` 


- ****NVLink's XID Error Extravaganza Erupts!****: Users are seeing thousands per minute of **XID errors** in `dmesg` related to **NVLink**, indicating potential hardware degradation brewing.
   - These errors suggest **bit errors** occurring on the NVLink, and a rapid increase in ECC indicates potential signal integrity issues.
- ****NCCL Network Checkup Challenge!****: Members are encouraged to run a quick **NCCL test** to check for retransmits and link fallbacks to assess the health of NVLink connections.
   - Correlating the test results with iteration times can help identify performance bottlenecks.
- ****NCCL's Quest for Deterministic Destiny!****: Discussion references [NVIDIA's blog post on controlling floating-point determinism in NCCL](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/) and a related [GitHub issue on deterministic algorithms](https://github.com/NVIDIA/cccl/issues/5550).


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1479502189176160349)** (3 messages): 

> `Modal GPU Access, Free Credits on Modal` 


- **Modal GPU Access Still Unconfirmed**: Members are waiting for updates on GPU access on **Modal**, with concerns raised about team members lacking local GPU resources.
   - A user directed the inquiry to a specific channel, <#1464407141128339571>, possibly for further details.
- **Doubts Emerge Over Free Modal Credits**: Teams are planning to utilize free credits on **Modal**, but there are uncertainties whether the amount will suffice for the entire development lifecycle.
   - The concern stems from team members who are remotely located and rely solely on **Modal** for development, particularly when local resources are unavailable.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1479209891091714311)** (1 messages): 

> `Small world models, Interactive world simulation` 


- **Cool Small World Models Emerge**: A member shared a link to a demonstration of [small world models](https://www.yixuanwang.me/interactive_world_sim/#interactive-demo) showcasing their intriguing properties.
   - The interactive demo allows users to explore the dynamics of these models.
- **Small World Network Visualization**: The [interactive simulation](https://www.yixuanwang.me/interactive_world_sim/#interactive-demo) allows users to tweak parameters and observe the effects on network structure and behavior.
   - This aids in understanding how interconnectedness arises and influences dynamics within the network.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1479298456689446952)** (10 messages🔥): 

> `Firmware Engineer Transition to GPU Stack, Summer Internship for Computer Science Sophomore, Importance of Concrete Results vs Credentials, Contributing to Open Source Projects` 


- **Firmware Pro Seeks GPU Career Path**: A firmware engineer with **4 years of experience** seeks advice on transitioning to a **GPU stack role**, particularly in compute kernels, starting with learning CUDA and GPU memory models from [NVIDIA blogs](https://developer.nvidia.com/blog).
   - The engineer hopes to determine the viability of such a transition and seeks guidance from those who have made a similar career move.
- **Sophomore Struggles to Snag Summer Software Slot**: A Computer Science sophomore is seeking advice on landing a summer internship despite working on several projects including **CUDA/Triton FlashAttention implementation**, building an **LLM serving pipeline with TensorRT-LLM and Triton Inference Server**, and maintaining a technical blog.
   - Feedback is requested on improving the resume, project selection, and application/networking strategies.
- **Results Trump Reputations, Reveals Reality**: A member expressed that *credentials like college degrees and internships are no longer enough*, and *concrete, verifiable results* are now essential for getting ahead.
   - They suggest building a GitHub profile that solves expensive engineering problems and contributing to open-source projects to demonstrate production-level coding skills.
- **Open Source Saves Students' Souls**: Members discussed the importance of contributing to **open-source projects** to gain practical experience.
   - They encouraged students to overcome hesitation and start contributing, emphasizing that even small contributions to large libraries can have a significant impact and provide valuable learning opportunities, mentioning [vLLM](https://github.com/vllm-project/vllm) as one such project.


  

---


### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/1479350893471338636)** (5 messages): 

> `SIMT/Tile interop, cuTile performance, FlashAttention backward kernel, Bastile` 


- **SIMT/Tile Interop to Unlock CUDA**: Members are working on **SIMT/Tile interop**, which will let users call **SIMT device functions** from **Tile functions**, potentially improving **CUDA's** capabilities.
   - If this works well then it could be a big step up for **cuda**, as one member imagines using **cuTile** to sort and partition inside their kernel, even if everything else in it was **SIMT code**.
- **cuTile Powers Outperforming Kernels**: A member built a small **cuTILE**-based monkey-patching library with custom kernels that outperform **Liger** both per-kernel and end-to-end on **Qwen3**.
   - Optimizations were taken from **TileGym**, optimized, and improvements were upstreamed back.
- **Bastile Library Emerges for CUDA**: A member has released **Bastile**, a **cuTILE** based library with custom kernels that outperform **Liger** on **Qwen3** and is working on a **FlashAttention** backward kernel.
   - Find the [Gh Repo here](https://github.com/aghilann/bastile) and the [Modal notebook with results on B200 here](https://modal.com/notebooks/aghilann/main/nb-9JUUBXJ23NK2b9Mf01WdEl).


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1479363758190100612)** (24 messages🔥): 

> `GDN prefill issue, Track C differences, Official Evaluation Environment details, CuTile code submissions, Modal GPU access` 


- **Debugging the INCORRECT_NUMERICAL Issue**: Some members are facing the `INCORRECT_NUMERICAL` issue in `GDN prefill` and are seeking a baseline that can pass the numerical accuracy test.
   - It was noted that [HuggingFace](https://huggingface.co) and the [Starter Kit](https://starterkit.com) use `qk4_v8`, but `mlsys26.flashinfer.ai` and `bench.flashinfer.ai` use `qk16_v32` for Track C, leading some to adapt their code to `qk16_v32`.
- **Requesting Details on Official Evaluation Environment**: A member requested the exact versions of **CUDA**, **Triton**, and **PyTorch** used in the official runtime / evaluation environment.
   - The goal is to closely match the local setup with the official environment for accurate testing.
- **Modal Free Credits for CUDA Compilation**: Members suggested compiling CUDA code locally and using Modal free credits primarily for benchmark/performance testing and correctness testing on Google Colab.
   - It was highlighted that an **NVIDIA GPU** is not required for compiling CUDA code or obtaining cubin files, recommending the use of an Nvidia dev docker image for CUDA 13 and above.
- **Blackwell B200 Access Discussed**: Members mentioned that although **B200** access would be helpful for **Blackwell**-oriented instructions like **UMMA**, significant progress can be made with general CUDA and lower-tier GPUs first.
   - It was noted that detailed profiling is typically done on separate machines anyway, as Modal doesn’t support `ncu`.
- **CuTeDsl experimenters can't CuTile**: One member reported using **CuTeDsl**, but could not submit it to Modal and had to write a custom Modal script.
   - This led to discussion about whether using custom scripts is allowed and requests for the organizers to add support for **CuTile**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1479459246151172096)** (24 messages🔥): 

> `OOM Error, GGUF Quantization, Compute Conference` 


- ****OOM** Error **Strikes** Finetuned Model Evaluation**: A member reported encountering **OOM** errors when evaluating a **36b LM** (GLM-4-5-Air-qlora) finetuned with **QLoRA** on four **96GB GPUs** using *lm_eval* harness and suggested to try `--num_processes 1`.
   - Another member suggested Gemini recommended adding `device_map=auto` to model_args.
- ****GGUF** Quantization **Considered** for Memory Savings**: After experiencing **OOM** errors, a member inquired about converting the model to **GGUF** format and quantizing it to **Q8** or **Q4** to reduce memory usage.
   - They expressed intent to try suggested solutions the following day.
- **Compute Conference Tickets **Up** for Grabs**: A member offered a couple of tickets to the **Compute conference** taking place on Sunday/Monday.
   - Another member inquired about the conference location and whether it would be available online.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1479277668615721011)** (22 messages🔥): 

> `NeRFs, Flow matching, Diffusion models, Video generation, Sharpness Aware Minimization` 


- **Flow Matching and Diffusion with NeRFs?**: Members discussed if anyone has tried **flow matching** or **diffusion with Neural Radiance Fields (NeRFs)** for video generation.
   - One member noted they had the same idea a couple months ago and found a recent paper doing that but also found out that *general modeling of moving/changing scenes is not well captured by NeRF like constructions so potentially not the right approach*.
- **NeRF Weights and Inductive Biases**: It was discussed if flow/diffusion transformers are good at mapping latent spaces, why not to the **weight-space of NeRFs**.
   - However, the structure of the weights doesn't have a trivial **inductive bias** like images, though you can train them with a **N(0, I) prior** like VAEs if you apply **L2 norm penalty** to the NeRFs.
- **Video NeRFs and Optical Flow Prediction**: A member wondered if you can do **video NeRFs** where you use an extra parameter **t** to describe how far along in the video, or you turn it into an **ODE like Flow modelling**, and try to predict **optical flow** instead and then integrate to find the video trajectory.
   - They also suggested there are ways to potentially make weights much more robust to perturbations in weight space from computational science.
- **SAM Helps NeRFs?**: It was mentioned that things like **Sharpness Aware Minimization (SAM)** helps make weights more robust but it's not clear how they affect **NeRF** behavior, also the computational chemistry is geared around exploration of **energy profiles** rather than optimisation so they have more stuff designed to overcome minimas and keep exploring.
   - They thought energy profile exploration was mostly **langevin dynamics** which is **SGD + noise**, and generally is hard in high dimensions which is where networks reside.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1479405706812592170)** (2 messages): 

> `muP cosine decay, wsd` 


- **Cosine Decay Craze Confirmed**: A member noted that *most papers* they've seen on **muP** use **cosine decay**.
   - They stated it almost *requires it*.
- **WSD Wows Way into Workflows**: A member countered that *most people actually use* **wsd** nowadays.
   - No further details were provided.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1479611885287379106)** (1 messages): 

> `Innoculation Prompting, Finetuning` 


- **Innoculation Prompting Paper Sparks Interest**: A member shared that they were reading the [inoculation prompting paper](https://alignment.anthropic.com/2025/inoculation-prompting/) from Anthropic and found it interesting.
   - They thought that the paper was related to finetuning techniques.
- **Relevance to Finetuning Highlighted**: The member emphasized the relevance of the inoculation prompting concept, particularly during **finetuning** processes.
   - The member who posted the message was sorry for tagging.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1479247512211816521)** (23 messages🔥): 

> `Quantization, vLLM library, Megatron vs TRL, PowerSync AI Hackathon, Deploy Lora spaces` 


- **Quantization reduces memory allocation**: A member explained that **quantization** reduces memory allocation by using smaller memory, for example, **float 8** instead of **float 32**, which allocates only **8 bits** of vram instead of **32 bits**.
   - With quantization if your model has **8 billion parameters**, then you save **24 bits** for each parameter.
- **vLLM is a toolbox for serving models efficiently**: **vLLM** bundles multiple approaches for lesser GPU consumption and serving techniques, like **KV caching**, which allows attention complexity to be **O(1)** for each newly computed token.
   - It also includes model compilation, tracing the model graph to create *a path* for tensors, and switching standard pytorch attention to **SDPA** or **flex-attention**.
- **Megatron is better for speed**: For pretraining, full-parameter SFT, or tasks needing model parallelism across many GPUs, **Megatron** is generally the faster choice compared to **TRL**.
   - For large-scale base training or heavy SFT, members recommended using **Megatron**, then **TRL** for preference tuning and RLHF-style post-training; NVIDIA offers **Megatron Bridge** for HF ↔ Megatron checkpoint conversion.
- **PowerSync hosts virtual AI hackathon with 8k in prizes**: **PowerSync** is hosting a virtual hackathon challenging participants to build innovative AI-powered software using **PowerSync** as a sync engine and compete for over **$8,000** in prizes.
   - For more info on rules and prizes visit [powersync.com/blog/powersync-ai-hackathon-8k-in-prizes](https://www.powersync.com/blog/powersync-ai-hackathon-8k-in-prizes).
- **Deploying Lora spaces is possible**: Members discussed deploying Lora spaces and someone shared the [deploy_lora_spaces.md](https://cdn.discordapp.com/attachments/879548962464493622/1479672216625877153/deploy_lora_spaces.md?ex=69ace3a3&is=69ab9223&hm=f122c064dd259c08b202cccab089506f9b384a532d4dde65ace943c1788f555b&) file.
   - They noted that exposing a model as an API endpoint is possible, but only very small LLMs will run practically using free CPU space.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1479207968808370328)** (14 messages🔥): 

> `Greywall sandboxing, Arksim synthetic users, Canvo mobile app, Ktiseos-Nyx-Trainer, Shadowclaw v1.3` 


- ****Greywall** Sandboxes CLI Agents: Open Sourced!**: Greywall, a tool to sandbox CLI agents with full shell access, has been [open-sourced](https://github.com/GreyhavenHQ/greywall).
   - It allows users to see and block network connections in real-time without restarting the session, and now supports MacOS.
- ****Arksim** Generates Synthetic Users for AI Agent Testing**: Arksim, a tool for generating synthetic users to test AI agents, has been [open-sourced](https://github.com/arklexai/arksim) and is available via `pip install arksim`.
- ****Canvo** Mobile App for Pocket Agency**: A member shared a [mobile app](https://github.com/canvo-app/canvo) for full pocket agency and better interaction with A2UI.
- ****Ktiseos-Nyx-Trainer**: NextJS for Open Source Loras**: A NextJS trainer for Open Source Loras and Checkpoints, named [Ktiseos-Nyx-Trainer](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer) was presented; it downloads and uploads to HF.
   - RoCM or Zluda are not supported yet.
- ****Shadowclaw** v1.3: Minimalist Personal AI Agent in C**: [Shadowclaw v1.3](https://huggingface.co/webxos/shadowclaw-c) is a minimal, single-binary personal AI agent written in C, adhering to the OpenClaw philosophy.
   - It features self-hosting, tool-using capabilities, persistent memory, and minimal dependencies, communicating with a local LLM (Ollama) via curl and saving state automatically.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1479563801341988937)** (1 messages): 

> `Gradio v4.19.0, Custom Components, Performance improvements, UI fixes` 


- **Gradio Graduates to v4.19.0!**: **Gradio v4.19.0** is now live with a batch of fixes and DX improvements, according to the [announcement](https://www.gradio.app/changelog).
   - To update, use `pip install -U gradio`.
- **Custom Components Compose Correctly**: Svelte version mismatch issues have been resolved and reload mode has been fixed for annotated types in **Custom Components**.
   - This will help avoid a common class of issues that many users face when using custom components, especially those that have Svelte code.
- **Gradio's Speed Boost**: Internal API calls and data structures are optimized to reduce latency, especially for MCP, yielding a **10x speedup** for `queue=False` events!
   - These improvements should lead to snappier application responsiveness, especially in scenarios with frequent updates.
- **Gradio's UI Gets a Facelift**: Several **UI fixes** have been implemented, including resolving `fill_height` issues, restoring **Submit buttons** after clicking examples, and ensuring `gr.Markdown` progress bars behave correctly.
   - These fixes collectively enhance the user experience by addressing common usability issues and visual glitches.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1479325138796413131)** (11 messages🔥): 

> `Introductions in Agents Course channel, Decoder's Lord Monster` 


- **New Members Introduce Themselves**: Several new members including Sai, Chanchlesh, Sidh, Chandan, and Sanket introduced themselves in the channel.
   - Interests ranged from AI agents and learning to build them, to web development, programming, and exploring new tech tools.
- **Decoder's Lord Claims Responsibility for 'Monster'**: Decoder's Lord acknowledged creating a *monster* with a recent push.
   - A PR has already been submitted to fix this issue.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1479276599022190612)** (42 messages🔥): 

> `Kimi K3 release date, Run Kimi K2.5 on RTX 3090, Kimi account support, Kimi CLI usage, Kimi Claw down` 


- **Next-Gen Kimi Speculation Launches**: Users are wondering when **Kimi K3** is coming out, following the release pattern of **Kimi K2** and **Kimi K2.5** [6 months apart](https://x.com/allen_ai/status/2029591872612561189).
   - One member speculated a possible release in *July*, but cautioned that research happens at its own pace.
- **Poor RTX 3090 if asked to run Kimi K2.5**: A user asked if **Kimi K2.5** can run on a single **RTX 3090** with quantized or coder (FT) version.
   - One member joked *If you glue a terabyte of VRAM to it, sure.. probably. Probably 1 token per hour or so.* 💥
- **Kimi customer support vanishes into thin air**: A user cancelled their **Kimi subscription** due to *non-existent* customer support after being charged the wrong amount multiple times.
   - They reported *No answer for 3 weeks about getting charged the wrong amount two times, it is simply unacceptable*.
- **Kimi CLI User Deploys 11 Containers while Sleeping**: One user reported using the **Kimi CLI** to deploy **11 containers to Azure** overnight, also reported removing **600** videos from a watch later playlist of **2000** videos.
   - Attached was an image implying the user was deploying this while sleeping [image](https://cdn.discordapp.com/attachments/1371757564005711973/1479492010615374030/Screenshot_2026-03-06-09-50-45-76_3aea4af51f236e4932235fdada7d1643.jpg?ex=69ace48e&is=69ab930e&hm=f39cbefb517531d1b016ce9176fe7247c662e2deaa9d10e043ee7fce7664933e&).
- **Kimi Claw Gets the Kimichop**: Multiple members reported that **Kimi Claw** has stopped working, and requested assistance to resolve the issue.
   - Members tried restarting it, the server, auto fix, but nothing has worked.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1479226718253351054)** (38 messages🔥): 

> `Boston Birthday Party, Credit Issues, Support Response Time, Subscription Problems` 


- **High cost of credits drives users away**: Several members expressed frustration with the high cost of credits, noting that credits are only available on the **$13,000/month** tier, and are *looking at migrating elsewhere*.
   - One user suggested trying **antigravity google** as an alternative.
- **Users report credit upgrade issues**: Multiple users reported issues with upgrading their credits or subscriptions, with one user stating they *just upgraded my credits for 200 euro but they never got added to my account* and another reporting they *upped my subscription to the $1k level and got charged but no credits in my account*.
   - These users were looking for help in resolving these billing issues.
- **Frustration grows over slow support response**: Users voiced concerns about the slow response time from support, with comments like *Support takes ages* and *Support is really slow*.
   - One user even questioned, *Is the support chat not working? My account was suspended unfairly.*


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1479212192317575248)** (23 messages🔥): 

> `Nvidia Orbital Datacenter System Architect Job, Francois Chollet's tweet, DGX Spark, LLMs not reaching human level` 


- **Nvidia is Seeking Orbital Datacenter System Architect**: Nvidia posted a job opening for an **Orbital Datacenter System Architect** to design systems for space-based computing, hinting at potential extraterrestrial endeavors; see the [job posting here](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Orbital-Datacenter-System-Architect_JR2014044).
- **Chollet's Tweet Sparks Debate**: A tweet by François Chollet prompted discussion, with some interpreting it as condescending, while others viewed it as a personal insight about underestimating the depth of sensorimotor learning, [see original tweet](https://fxtwitter.com/vicnaum/status/2029579972688379928).
- **Considering DGX Spark Despite Concerns**: Members discussed whether the **NVFP4** in the **DGX Spark** is workable, and whether thermal and OS stability issues have been resolved, referencing a [tweet from John Carmack](https://x.com/ID_AA_Carmack/status/1982831774850748825) mentioning such problems.
- **LLMs Not Reaching Human-Level Intelligence, a Relief?**: A member expressed satisfaction that **LLMs** are not projected to reach human-level intelligence in the next few years, voicing concerns about powerful individuals controlling robot armies.
   - The member also mentioned working on a product that helps people find and use the right tool for the job in image processing, with positive customer feedback.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1479538113519816785)** (2 messages): 

> `New Job Announcement` 


- **Member delays Chapter Release due to New Job**: A member announced that Chapter 2 of S&B would be delayed until next Thursday due to starting a new job this week.
- **Congratulations on the New Job!**: Another member congratulated the user on their new job.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1479266561654001956)** (6 messages): 

> `Anthropic Economic Index, Datacenter Bubble, Department of Wario` 


- **Anthropic economic index launched**: Anthropic launched the **Anthropic Economic Index** according to their [official announcement](https://www.anthropic.com/news/the-anthropic-economic-index).
- **Datacenter bubble peaks**: Members noted that we are at peak **datacenter bubble** according to [this post](https://x.com/i/status/2029907842208031203).
- **DoW is now Department of Wario**: One member joked that every time someone says **DoW** they hear **Department of Wario**, posting a meme about it.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1479372620867833866)** (5 messages): 

> `Tinygrad JITBEAM Benchmarks, Bounty Lock Submission Fees, CAT operator` 


- **Tinygrad JITBEAM Bests C**: The Tinygrad **JITBEAM** has been benchmarked as performing better than **C**, following various upgrades and fixes, according to [this Discord message](https://discord.com/channels/1068976834382925865/1108235368702164992/1479323496990507101).
- **Bounty Lock Fees**: A suggestion was made to require a small, refundable **$5 fee** for every bounty lock submission.
- **Debating CAT Operator's Merits**: The discussion revolved around the **CAT operator**, questioning if it matches other movement ops and if it's strictly needed.
   - One member noted that *mathematicians like to make their reasoning as general as possible*, whereas *physicists, on the other hand, are always interested in the special case*, which is the side tinygrad leans towards.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1479224848537485477)** (4 messages): 

> `Security Vulnerability Disclosure, GPT-5.4 Token Usage, Aider for Delphi/Pascal` 


- **Researcher's Unheeded Vulnerability Report Exploited!**: Security researcher Adnan Khan discovered a vulnerability chain in late December 2025, reporting it via a [GitHub Security Advisory](https://github.com/advisories) on January 1, 2026, but received no response to multiple follow-ups.
   - Upon Khan's public disclosure on February 9, Cline patched within **30 minutes**, though a subsequent key rotation error led to further issues.
- **GPT-5.4's Appetite for Tokens**: A user noted that while **GPT 5.4** performs well, it consumes a large number of tokens, making it a *token hog*.
   - Further analysis on the model's efficiency may be required given its robust performance metrics.
- **Aider Use in Delphi/Pascal**: A member inquired whether anyone utilizes Aider with **Delphi/Pascal**.
   - It remains to be seen whether other developers are leveraging Aider in this context.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1479521698523906270)** (2 messages): 

> `LoRA gradients on Apple Neural Engine, Modal Sandbox & Volume for Memory, ANE matmul compiler, Fleet-RLM framework` 


- **LoRA Gradients fire on Apple's Neural Engine**: An engineer leveraged **Claude Code (Opus 4.6)** to run LoRA fine-tuning on Apple's Neural Engine at **~2.8W**, with **192 ANE gradient dispatches** and zero GPU fallbacks, detailed in a [blogpost](https://x.com/StraughterG/status/2029957160864522513).
   - The engineer found that `matmul` compiles but never executes, spatial dimensions must be multiples of 16, and the ANE compiler silently fails after ~119 compiles.
- **Modal Sandbox & Volume Boost Memory**: A developer is improving their frontend, ditching Redis and vector stores, opting for **Modal Sandbox** and **Volume** for memory/analyzing tasks in the [fleet-rlm](https://github.com/Qredence/fleet-rlm) framework.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1479571797052494021)** (2 messages): 

> `Compute Conference, AI Infrastructure, AI Agents, Next Generation Cloud` 


- **Daytona hosts Compute Conference in San Francisco**: Daytona is hosting **Compute**, a conference focused on **AI infrastructure**, **agents**, and the **next generation of cloud**, taking place **March 8-9** at the **Chase Center, San Francisco**, as detailed on their [website](https://compute.daytona.io/).
- **Speakers Highlighted at Compute Conference**: The conference will feature speakers including **Aaron Levie** (Box), **Parag Agrawal** (Parallel), **Harrison Chase** (LangChain), **Lin Qiao** (Fireworks AI), **Russ D'Sa** (LiveKit), **Beyang Liu** (Amp), **David Cramer** (Sentry), **Nikita Shamgunov** (Neon), **Dylan Patel** (SemiAnalysis), **Waseem Alshikh** (Writer), and **Ivan Burazin** (Daytona).
- **Complimentary Tickets Available for Compute**: Three complimentary tickets are available for the **Compute Conference** using the code `EQ6VA5` on [Luma](https://luma.com/k6bc82dv).


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1479561301104525332)** (1 messages): 

> `MCP-I questions, Auth agent identity, MCP ecosystem relevance` 


- **MCP-I Question Incoming**: A member is encountering a question on [MCP-I](https://share.google/aimode/xAik81A0u4WKsjewv) and wants to integrate it into the **auth agent identity** side.
   - The aim is to capture use cases within an actual MCP contrib ecosystem, with the post serving as an FYI.
- **MCP Relevance Questioned**: The member notes that the issue often falls into a "XXXXMCP" or "MCP - XXXXX" category that doesn't directly tie to **MCP** when investigated further.
   - This raises questions about the true relevance and connection to the broader MCP ecosystem.


  

---


---


---


---
id: MjAyNi0w
title: 'Context Graphs: Hype or actually Trillion-dollar opportunity?'
date: '2026-02-03T05:44:39.731046Z'
description: >-
  **Zhipu AI** launched **GLM-OCR**, a lightweight **0.9B** multimodal OCR model
  excelling in complex document understanding with top benchmark scores and
  day-0 deployment support from **lmsys**, **vllm**, and **novita labs**.
  **Ollama** enabled local-first usage with easy offline operation. **Alibaba**
  released **Qwen3-Coder-Next**, an **80B MoE** model with only **3B active**
  parameters, designed for coding agents with a massive **256K context window**
  and trained on **800K verifiable tasks**, achieving over **70% SWE-Bench
  Verified**. The open coding ecosystem also saw **Allen AI** announce
  **SERA-14B**, an on-device-friendly coding model with new datasets. The
  emerging concept of **Context Graphs** was highlighted as a promising
  framework for data and agent traceability, with initiatives like **Cursor's
  Agent Trace** specifying context graphs for coding agents, emphasizing
  potential improvements in agent performance and customer-driven adoption. This
  coverage reflects ongoing innovation in **multimodality**, **long-context**,
  **mixture-of-experts**, and **agentic coding models**.
companies:
  - zhipu-ai
  - lmsys
  - vllm
  - novita-labs
  - ollama
  - alibaba
  - allenai
  - cognition
  - cursor
models:
  - glm-ocr
  - qwen3-coder-next
  - sera-14b
topics:
  - multimodality
  - ocr
  - long-context
  - mixture-of-experts
  - agentic-coding-models
  - context-graphs
  - benchmarking
  - model-deployment
  - model-optimization
  - model-training
people:
  - jaya_gupta
  - dharmesh_shah
---


**a quiet day lets us feature a bubbling topic.**

> AI News for 1/30/2026-2/2/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**254** channels, and **14979** messages) for you. Estimated reading time saved (at 200wpm): **1408 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!


Our policy for quiet days is that we’ll now feature longer running, slow-burn stories that would otherwise not spike to the top of the heap on a certain day but will probably be of broader historical significance to AI Engineers. Today’s Lightning Pod (our Youtube-only short format) features the topic of Context Graphs, which [Jaya Gupta launched late December on X](https://x.com/JayaGup10/status/2003525933534179480) and has since inspired even former guests like [Dharmesh Shah](http://latent.space/p/dharmesh) (who has [reservations](https://simple.ai/p/what-are-context-graphs)). We chatted with both authors on the response:

https://youtu.be/zP8P7hJXwE0

That’s thoughtleading 101, but definitely helpful — for sure every founder building a data/context engineering product will go to them and say they got the people who coined Context Graphs on their cap table. But the problem with the post is that it promises a whole lot (evidenced in the title), but is not very prescriptive.

[Recently](https://x.com/cognition/status/2017057457332506846), I also framed Cursor’s [Agent Trace initiative](https://agent-trace.dev/) as a “Context Graph” for Code:

This is the first actual specification for a context graph for a specific domain (coding agents) that is agreed on between companies. It remains to be seen if it has actual staying power, which will mostly be driven by 1) high demonstrated improvement in agent performance, and 2) customer pressure to support it. Based on first principles, the idea (capture decision traces, exceptions and precedents scattered all over the “data mesh” into the context of an LLM) seems compelling, but of course, the devil is in the details.

---

# AI Twitter Recap

**Zhipu AI’s GLM‑OCR launch (0.9B) and day‑0 deployment support across stacks**

- **GLM‑OCR (multimodal OCR for complex documents)**: Zhipu released **GLM‑OCR**, positioned as a lightweight, deployable **0.9B** model for real-world document understanding (tables, formulas, information extraction, messy layouts). It’s reported **#1 on OmniDocBench v1.5 (94.62)** and emphasized as low‑latency / high‑concurrency friendly. See the ecosystem “day‑0 support” announcements from [@lmsysorg](https://twitter.com/lmsysorg/status/2018521181146751486) (SGLang integration + PR/cookbook links) and [@vllm_project](https://twitter.com/vllm_project/status/2018582480518091083) (vLLM day‑0 support), plus deployment marketing from [@novita_labs](https://twitter.com/novita_labs/status/2018565896013574225).
- **Local-first availability**: Ollama shipped immediate local pulls + API usage (“drag and drop images into terminal”, JSON‑formatted outputs), making GLM‑OCR easy to run offline: [@ollama](https://twitter.com/ollama/status/2018525802057396411) and library link [@ollama](https://twitter.com/ollama/status/2018525804733575492). Community comparisons also claim strong quality vs PaddleOCR/DeepSeek OCR: [@bdsqlsz](https://twitter.com/bdsqlsz/status/2018663915404841212). LlamaIndex highlighted benchmark displacement (claiming 50–100% faster vs prior top model) and ongoing eval integration: [@jerryjliu0](https://twitter.com/jerryjliu0/status/2018713059359899729).

**Agentic coding models & harnesses: Qwen3‑Coder‑Next (80B@3B), SERA‑14B, and the “skills/MCP” tool interface convergence**

- **Qwen3‑Coder‑Next**: Alibaba released **Qwen3‑Coder‑Next**, an open‑weight **80B MoE** with only **3B active** parameters, pitched for *coding agents + local dev* with **256K context**, trained with **800K verifiable tasks + executable environments**. They claim **>70% SWE‑Bench Verified** with SWE‑Agent scaffold and strong agent benchmark efficiency: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2018718453570707465) and benchmark callout [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2018719026558664987). Independent/adjacent summaries: [@UnslothAI](https://twitter.com/UnslothAI/status/2018718997584474191) (memory footprint + GGUF guidance) and commentary on efficient long‑context attention choices (e.g., “Gated DeltaNet” mentioned in the discourse): [@eliebakouch](https://twitter.com/eliebakouch/status/2018730622358073384). vLLM shipped day‑0 support in **vLLM 0.15.0**: [@vllm_project](https://twitter.com/vllm_project/status/2018742511502856568).
- **Open Coding Agents ecosystem (Ai2)**: Allen AI announced **SERA‑14B** (on‑device‑friendly coding model) plus refreshed open datasets that include **raw trajectories + verification metadata**: [@allen_ai](https://twitter.com/allen_ai/status/2018741177734910166) and dataset/model detail thread pointer [@ethnlshn](https://twitter.com/ethnlshn/status/2018746924803969317).
- **Harness > model (recurring theme)**: Multiple tweets converge on the idea that the leverage in agents is increasingly in the **harness** (permissions, memory, workflows, reversibility), not just raw model IQ. A clear articulation: [@sarahmsachs](https://twitter.com/sarahmsachs/status/2018720637691572634).
- **Standardization of agent “skills” directories + protocols**:
  - **Agent Client Protocol (ACP)**: proposed as a JSON‑RPC standard to unify agent↔editor communication across Gemini CLI / Claude Code / Codex CLI / OpenClaw, supporting stdio/HTTP, file access, terminals, permissions, streaming updates: [@_philschmid](https://twitter.com/_philschmid/status/2018706591776756216).
  - **Skills vs MCP tools**: LlamaIndex contrasted “skills” (easy but brittle, NL‑interpreted) vs MCP servers (more deterministic schemas, more setup, network latency but centralized updates): [@llama_index](https://twitter.com/llama_index/status/2018749615907213457) and follow‑ups [@jerryjliu0](https://twitter.com/jerryjliu0/status/2018797672258490666), [@itsclelia](https://twitter.com/itsclelia/status/2018821269752611102). Meanwhile, “`.agents/skills` is becoming a default” was called out explicitly (Codex/OpenCode/Copilot/Cursor adopting; Claude Code not yet): [@theo](https://twitter.com/theo/status/2018819504252608710).

**Coding agent products: Codex app adoption, Claude Code sharing + Apple Xcode integrations**

- **Codex app momentum + inference speedups**:
  - Sam Altman reported **200k+ downloads on day 1**: [@sama](https://twitter.com/sama/status/2018734731437985930).
  - OpenAI shipped **40% faster GPT‑5.2 & GPT‑5.2‑Codex** for API customers (“same weights, lower latency”): [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2018838297221726482).
  - Codex integration into **Xcode 26.3** was announced by OpenAI DevRel: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2018796432443244897).
- **Claude Code product iteration**:
  - **Session sharing** for Claude Code across web/desktop/mobile: [@lydiahallie](https://twitter.com/lydiahallie/status/2018740156359229883).
  - Community “waiting for Sonnet 5” speculation dominated, including the claim that an **Anthropic image model is live on LMArena**: [@kimmonismus](https://twitter.com/kimmonismus/status/2018689719324791022) and “Claude Image is coming” chatter: [@kimmonismus](https://twitter.com/kimmonismus/status/2018669423402660082).
- **Apple Xcode + Claude Agent SDK**: Anthropic announced **native Xcode integration** with the **Claude Agent SDK** (subagents/background tasks/plugins) to bring Claude Code‑like capabilities directly into Apple dev workflows: [@AnthropicAI](https://twitter.com/AnthropicAI/status/2018771170938724682). This is a notable step in “agent-in-the-IDE” becoming first‑party.

**Agent infrastructure & observability: traces as the source of truth, deep agents evaluation, and memory beyond RAG**

- **Observability shifts from code to traces**: LangChain argues that for agentic systems, runtime decisions happen in the model—so **traces** become the primary artifact for debugging/understanding. See: [@LangChain](https://twitter.com/LangChain/status/2018739770495512880).
- **How to evaluate deep agents**: LangChain’s eval guidance emphasizes bespoke success criteria per case, single‑step regression checks, full‑turn and multi‑turn evals, and clean/reproducible envs: [@LangChain](https://twitter.com/LangChain/status/2018769968515404212).
- **DeepAgents releases (JS/CLI/runtime backends)**:
  - deepagents@1.6.2 fixes (checkpoint restore, infinite loop on large files, toolcall middleware simplification): [@LangChain_JS](https://twitter.com/LangChain_JS/status/2018731100441620517).
  - DeepAgents 0.3.10 adds **LocalShellBackend** for running code on your machine: [@sydneyrunkle](https://twitter.com/sydneyrunkle/status/2018788505082859863).
  - deepagents-cli 0.0.16 improves control/visibility for shell runs: [@masondrxy](https://twitter.com/masondrxy/status/2018741344835870820).
- **Memory: “RAG wasn’t designed for agent memory”**: DAIR’s **xMemory** proposes hierarchical retrieval (themes/semantics/episodes/messages) to reduce redundancy while preserving evidence chains, showing better LoCoMo scores with fewer tokens than naive top‑k similarity retrieval: [@dair_ai](https://twitter.com/dair_ai/status/2018765444702982395).
- **Filesystem as agent context scratchpad**: The “files-first” workflow (store artifacts outside context, avoid bloating windows) is reinforced by deepagents’ design and commentary: [@LangChain_JS](https://twitter.com/LangChain_JS/status/2018732184694374669).

**Benchmarks & evaluation signals: METR time horizons, WorldVQA, Text/Search/Image Arena updates, and ARC‑AGI progress**

- **METR time horizon for Gemini 3 Pro**: METR estimates **~4 hours (50% time horizon)** on an expanded software task suite (with CI): [@METR_Evals](https://twitter.com/METR_Evals/status/2018752230376210586). This “time horizon” line of evals continues to become a key agent capability proxy beyond static coding benchmarks.
- **WorldVQA (Moonshot/Kimi)**: Moonshot introduced **WorldVQA** to measure “atomic vision-centric world knowledge” separately from reasoning, explicitly trying to decouple memorization from reasoning quality. Dataset: **3,500 VQA pairs across 9 categories** with linguistic/cultural diversity: [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2018697552456257945).
- **Arena leaderboards**:
  - **Text Arena (open models, Jan 2026)**: #1 **Kimi‑K2.5‑Thinking**, #2 **GLM‑4.7**, #3 **Qwen3‑235B‑A22B Instruct**: [@arena](https://twitter.com/arena/status/2018727506850033854).
  - **Search Arena update**: Google’s **gemini‑3‑flash‑grounding** leads; OpenAI search non‑reasoning appears in top 5; best Claude search variant listed: [@arena](https://twitter.com/arena/status/2018760874178342975).
  - **Image Arena Pareto frontiers**: Arena published **quality vs price per image** frontiers for text‑to‑image and image edit (notable that several OpenAI/Google/Flux/Tencent models sit on the frontier depending on cost constraints): [@arena](https://twitter.com/arena/status/2018787949840896119) and edit frontier [@arena](https://twitter.com/arena/status/2018792314878234704).
- **ARC‑AGI**: ARC Prize reported a **new SOTA public submission** (with cost/task figures) based on **GPT‑5.2** ensembles: [@arcprize](https://twitter.com/arcprize/status/2018746794310766668). Separately, there’s ongoing community chatter on ARC‑AGI‑2 progress rates: [@kimmonismus](https://twitter.com/kimmonismus/status/2018800964891984181).

**Efficiency, kernels, and training/inference plumbing: fp8 training, Blackwell throughput, and “context engineering” as inference-era data engineering**

- **Karpathy’s fp8 training notes (practical, not just theory)**: He reports enabling **fp8 training** to improve “time to GPT‑2” to **2.91 hours**, discusses real bottlenecks (not purely compute‑bound), overheads from scaling conversions, GEMM sizing, and quality degradation per step; notes that larger models see better fp8 upside (citing torchao’s larger gains): [@karpathy](https://twitter.com/karpathy/status/2018804068874064198).
- **vLLM + NVIDIA Blackwell optimization**: vLLM reports big perf gains for **gpt‑oss‑120b** on Blackwell via FlashInfer integration, torch.compile fusions, async scheduling, and stream interval optimizations: [@vllm_project](https://twitter.com/vllm_project/status/2018859316258931161).
- **Inference is a first-class engineering surface**: “Context engineering is as important to inference as data engineering is to training” was stated succinctly (and repeated): [@swyx](https://twitter.com/swyx/status/2018533744442057115). This sentiment shows up elsewhere as teams debate filesystems, tool choice (skills vs MCP), caching, and harness design.

---

### Top tweets (by engagement)
- [CEO of highest valued company giving a “conference” in the middle of a street](https://twitter.com/yacinelearning/status/2018689145086898466) — massive engagement meme/event commentary.
- [SpaceX acquires xAI / “Building an interstellar civilization”](https://twitter.com/elonmusk/status/2018784828129243614).
- [Codex app day‑1 downloads: “More than 200k”](https://twitter.com/sama/status/2018734731437985930).
- [Apple Xcode integrates Claude Agent SDK](https://twitter.com/AnthropicAI/status/2018771170938724682).
- [OpenAI hires Head of Preparedness](https://twitter.com/sama/status/2018813527780463027).
- [GPT‑5.2 & GPT‑5.2‑Codex now 40% faster (inference stack optimized)](https://twitter.com/OpenAIDevs/status/2018838297221726482).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-Coder-Next Release

  - **[Qwen/Qwen3-Coder-Next · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1quvqs9/qwenqwen3codernext_hugging_face/)** (Activity: 842): ****Qwen3-Coder-Next** is a cutting-edge language model designed for coding, featuring `3B activated parameters` out of a total `80B`, achieving performance comparable to models with `10-20x` more active parameters. It supports advanced capabilities like long-horizon reasoning and has a `256k` context length, making it ideal for integration with IDEs. The architecture includes `48 layers`, gated attention, and a mixture of experts, suitable for dynamic coding tasks. Deployment can be done using **SGLang** or **vLLM**, requiring specific versions for optimal performance. More details are available in the [original article](https://huggingface.co/Qwen/Qwen3-Coder-Next).** One commenter expressed skepticism about the model's performance, questioning if a `3B activated parameter` model can truly match the quality of larger models like Sonnet 4.5, indicating a need for further validation of these claims.

    - danielhanchen discusses the release of dynamic Unsloth GGUFs for Qwen3-Coder-Next, highlighting upcoming releases of Fp8-Dynamic and MXFP4 MoE GGUFs. These formats are designed to optimize model performance and efficiency, particularly in environments with limited resources. The linked guide provides instructions for using Claude Code and Codex locally with Qwen3-Coder-Next, which could be beneficial for developers looking to integrate these models into their workflows.
    - Ok_Knowledge_8259 expresses skepticism about the claim that a 3 billion activated parameter model can match the quality of larger models like Sonnet 4.5. This comment reflects a common concern in the AI community about the trade-off between model size and performance, suggesting that while smaller models are more efficient, they may not always achieve the same level of quality as their larger counterparts.
    - Septerium notes that while the original Qwen3 Next performed well in benchmarks, the user experience was lacking. This highlights a critical issue in AI model deployment where high benchmark scores do not always translate to practical usability, indicating a need for improvements in user interface and integration to fully leverage the model's capabilities.

  - **[Qwen3-Coder-Next is out now!](https://www.reddit.com/r/LocalLLM/comments/1quw0cf/qwen3codernext_is_out_now/)** (Activity: 228): **The image announces the release of **Qwen3-Coder-Next**, an `80B` MoE (Mixture of Experts) model with `3B` active parameters, designed for efficient coding tasks and local deployment. It emphasizes the model's capability in long-horizon reasoning and complex tool use, requiring `46GB` of RAM/VRAM for operation. The graph in the image highlights its performance efficiency compared to other models, showcasing its ability to achieve high performance with fewer active parameters. This model is particularly noted for its fast agentic coding capabilities.** A user inquired about the feasibility of running the model with `64GB` of RAM without VRAM, indicating interest in its hardware requirements. Another comment questions the model's performance level, comparing it to 'sonnet 4.5', suggesting skepticism or curiosity about its capabilities. Additionally, there is a remark on the absence of a comparison with 'Devstral 2', hinting at expectations for benchmarking against specific models.

    - A user inquires about the possibility of running Qwen3-Coder-Next with 64GB of RAM and no VRAM, which suggests interest in the model's memory efficiency and potential CPU-only deployment. This highlights the need for understanding the model's hardware requirements and optimization for non-GPU environments.
    - Another user questions the model's performance by comparing it to 'sonnet 4.5 level', indicating skepticism about the model's capabilities or potential over-optimization for specific benchmarks. This reflects a common concern in AI model evaluations where performance might be tailored to excel in certain tests rather than general use cases.
    - A technical query is raised about the appropriate quantization for a setup with 28GB NVIDIA VRAM and 96GB DDR5 RAM. This suggests a focus on optimizing the model's performance for specific hardware configurations, which is crucial for maximizing efficiency and speed in high-performance computing environments.


### 2. ACE-Step 1.5 Audio Model Launch

  - **[ACE-Step-1.5 has just been released. It’s an MIT-licensed open source audio generative model with performance close to commercial platforms like Suno](https://www.reddit.com/r/LocalLLaMA/comments/1quzwjf/acestep15_has_just_been_released_its_an/)** (Activity: 408): ****ACE-Step-1.5** is an open-source audio generative model released under the MIT license, offering performance comparable to commercial platforms like **Suno**. It supports **LoRAs**, multiple models for various needs, and features like cover and repainting. The model is integrated with **Comfy** and available for demo on **HuggingFace**. This release marks a significant advancement in open-source audio generation, narrowing the gap with top-tier commercial solutions.** One comment highlights skepticism about the model's prompt adherence, noting that demo prompts often don't align with outputs, suggesting potential limitations in instruction following.

    - The release of ACE-Step-1.5, an MIT-licensed open-source audio generative model, is notable for its performance, which is reportedly close to commercial platforms like Suno. This model's efficiency is highlighted by its ability to generate outputs in just 2 seconds on an A100 GPU, indicating significant computational optimization.
    - There is skepticism about the model's adherence to input prompts, as some users have observed that the demo prompts do not align closely with the generated outputs. This raises questions about the model's instruction-following capabilities and the effectiveness of its prompt processing.
    - The discussion also touches on the model's capabilities in generating instrumental music. A user compares it to HeartMuLa, noting that while HeartMuLa cannot produce instrumentals without vocals, it is unclear if ACE-Step-1.5 can fulfill this specific requirement, indicating a potential area for further exploration or development.

  - **[The open-source version of Suno is finally here: ACE-Step 1.5](https://www.reddit.com/r/LocalLLaMA/comments/1quxtkj/the_opensource_version_of_suno_is_finally_here/)** (Activity: 319): ****ACE-Step 1.5** is an open-source music generation model that outperforms **Suno** on standard evaluation metrics. It can generate a complete song in approximately `2 seconds` on an **A100 GPU** and operates locally on a typical PC with around `4GB VRAM`, achieving under `10 seconds` on an **RTX 3090**. The model supports **LoRA** for training custom styles with minimal data and is released under the **MIT license**, allowing free commercial use. The dataset includes fully authorized and synthetic data. The project is fully open-source, with [GitHub resources](https://github.com/ace-step/ACE-Step-1.5) available for weights, training code, LoRA code, and the research paper.** Commenters noted the model's significant improvements over previous versions but criticized its instruction following and coherency compared to **Suno v3**. Despite these issues, the audio quality is considered good, and the model is seen as a creative alternative to Suno. There is anticipation for a version 2 release.

    - TheRealMasonMac highlights that ACE-Step 1.5 shows a significant improvement over its predecessor, but it still lags behind Suno v3 in terms of instruction following and coherency. However, the audio quality is noted to be good, and the model is described as creative and different from Suno, suggesting it could be a solid foundation for future development.
    - Different_Fix_2217 provides examples of audio generated by ACE-Step 1.5, indicating that the model performs well with long, detailed prompts and can handle negative prompts. This suggests flexibility in input handling, which could be beneficial for users looking to experiment with various prompt styles.


### 3. Local LLM Developments and Comparisons

  - **[128GB devices have a new local LLM king: Step-3.5-Flash-int4](https://www.reddit.com/r/LocalLLaMA/comments/1qtvo4r/128gb_devices_have_a_new_local_llm_king/)** (Activity: 619): **The `Step-3.5-Flash-int4` model, available on [Hugging Face](http://huggingface.co/stepfun-ai/Step-3.5-Flash-Int4), is a new local LLM optimized for devices with `128GB` RAM, such as the M1 Ultra Mac Studio. It supports a full context length of `256k` and demonstrates high efficiency in RAM usage. Benchmarks using `llama-bench` show impressive performance with `100k` prefill, maintaining usability for CLI coding agents. The model requires a custom `llama.cpp` fork for execution, with potential for upstream support due to its performance.** Commenters are curious about the model's performance on different hardware, such as Strix Halo, and express interest in a potential NVFP4 version. There is also a light-hearted comment on the model's name.

    - The benchmark results for the Step-3.5-Flash-Int4 model on the AMD Strix Halo (Minisforum MS S1 Max) using ROCm 7.1.1 show impressive performance, with a throughput of `258.82 ± 3.15` tokens per second for the `pp4096` test. This suggests that the model can handle full context fitting efficiently, making it a strong contender for local LLM tasks on 128GB devices.
    - Comparative performance on different backends reveals that the Step-3.5-Flash-Int4 model performs best on ROCm, with a significant drop in throughput when using Vulkan-amdvlk and Vulkan-radv. For instance, the `pp4096` test on Vulkan-amdvlk yields `153.04 ± 0.30` tokens per second, while Vulkan-radv achieves `164.20 ± 1.30`, indicating that ROCm is the optimal backend for this model.
    - The Step-3.5-Flash-Int4 model's performance on the `tg512` test varies significantly across backends, with ROCm achieving `22.93 ± 0.00` tokens per second, while Vulkan-amdvlk and Vulkan-radv show much lower performance at `2.50 ± 0.00` and `27.86 ± 0.00` tokens per second, respectively. This highlights the importance of backend selection in optimizing model performance.

  - **[Local model fully replacing subscription service](https://www.reddit.com/r/LocalLLM/comments/1qtuwn5/local_model_fully_replacing_subscription_service/)** (Activity: 270): **The post discusses the effectiveness of local models, specifically **Ollama + GPT-OSS:20b**, on a MacBook Pro M4 Pro with `24GB` memory, suggesting it can replace subscription services like ChatGPT for non-complex queries. The user highlights the model's speed and quality, noting it performs well for tasks like research queries and basic coding. A comment suggests using `mlx` based models on Apple silicon for a `40%` increase in token per second speed, accessible via **LMstudio**. Another comment notes that **GPT-OSS:20b** can efficiently run with a `128k` context using `17GB` VRAM, leaving room for other GPU tasks. The discussion also touches on building local agent frameworks to match the capabilities of subscription models like **Claude**, with a focus on integrating tools and skills to enhance local model performance.** Commenters debate the efficiency of local models versus subscription services, with some suggesting that models like **Claude** still outperform local options for complex tasks. There's also a discussion on the minimum model size for effective tool-calling agents, with `30b` being suggested as a baseline for reliable performance.

    - **coldy___** highlights the performance benefits of using MLX-based models on Apple Silicon, noting a potential `40%` increase in token per second speed. They recommend using LM Studio to access these models, specifically mentioning the `gpt-oss 20b` model as optimized for this hardware.
    - **generousone** discusses the efficiency of the `gpt-oss:20b` model, which can run with a full `128k` context using only `17GB` of VRAM. This leaves room for other GPU-intensive tasks, making it a practical choice for users with `24GB` VRAM. They acknowledge it's not as advanced as commercial models like ChatGPT or Claude but find it sufficient for many tasks.
    - **2BucChuck** shares insights on building a local agent framework to overcome limitations of local models, testing models like `Gemma32` against agent tasks. They suggest a minimum model size of `30B` for effective tool-calling agents, noting smaller models often underperform. The goal is to match the functionality of subscription services by integrating tools and skills into local models.

  - **[New 1.4B Model Victorian LLM - Violet](https://www.reddit.com/r/LocalLLM/comments/1quip6h/new_14b_model_victorian_llm_violet/)** (Activity: 67): **The post introduces **Violet**, a new 1.4 billion parameter LLM trained entirely on Victorian-era data (1800-1899), aiming to create an ethically sourced, public domain model. The model was developed from scratch, using data from sources like the Internet Archive, Project Gutenberg, and the British National Library, and includes ONNX quantized versions for local browser use. The model is noted for its narrative prose capabilities but has limitations in reasoning and historical biases, such as misgendering. The project also features a unique chat variant with mood-based avatars, and the model is available on [Hugging Face](https://huggingface.co/zakarth/violet-1b4-chat) with demos linked [here](https://huggingface.co/spaces/zakarth/violetdemo).** A commenter inquires about the model's ability to understand modern phrases, questioning if it can only communicate in the vernacular of Victorian England, suggesting a potential limitation in comprehending contemporary language.

    - thirsty_pretzelzz raises an interesting point about the Victorian LLM's language capabilities, questioning whether it can only communicate using the vernacular of Victorian England. This implies a potential limitation in understanding modern phrases, which could affect its applicability in contemporary contexts.
    - avanlabs expresses interest in training a similar model on specific datasets for deployment on small devices. They request resources or blogs that could provide insights into building and optimizing small language models (SLMs), indicating a focus on efficient model training and deployment strategies.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Sonnet 5 and Gemini 3.5 Release Discussions

  - **[Sonnet 5 release on Feb 3](https://www.reddit.com/r/ClaudeAI/comments/1qtm9ix/sonnet_5_release_on_feb_3/)** (Activity: 2328): **The leaked details about **Claude Sonnet 5**, codenamed "Fennec," suggest it is a significant advancement over previous models, with a potential release date of February 3, 2026, as indicated by a Vertex AI error log. It is rumored to be 50% cheaper than Claude Opus 4.5 while maintaining a `1M token` context window and offering faster performance, likely due to optimization on Google TPUs. The model is also said to feature a "Dev Team" mode, allowing autonomous sub-agents to build features collaboratively. Benchmarking claims suggest it surpasses `80.9%` on SWE-Bench, outperforming current coding models.** There is skepticism about the release timing, as some users argue that the error log does not conclusively prove the model's existence or its release date. Additionally, concerns are raised about the accuracy degradation in large context windows, which was an issue in previous models.

    - andrew_kirfman discusses skepticism about the timing of the Sonnet 5 release, referencing a 404 error from a vertex API endpoint that doesn't confirm the model's existence. They highlight that Anthropic's model IDs often reflect the creation date of the model checkpoint, not the release date, citing Opus 4.5's ID `20251101` as an example. They express doubt about future-dating a release tag, which is uncommon in software releases.
    - andrew_kirfman also mentions the potential for a 1 million token context in Sonnet 5, noting that previous models like Sonnet 4 and 4.5 already offered this through the API. However, they point out that accuracy degradation was an issue with these models, suggesting that improvements in this area would be necessary for trust in the new release.

  - **[Claude Sonnet 5: The “Fennec” Leaks](https://www.reddit.com/r/Bard/comments/1qtnkhu/claude_sonnet_5_the_fennec_leaks/)** (Activity: 193): **The image is a tweet by Pankaj Kumar discussing leaks about "Claude Sonnet 5," codenamed "Fennec." It highlights features such as a potential release date of February 3, 2026, aggressive pricing, and advanced capabilities like TPU acceleration and specialized sub-agents. The model is rumored to be significantly cheaper and faster than its predecessor, with a large context window and high benchmarking performance. Additionally, it suggests that the model is already integrated into Google's infrastructure. [Image URL](https://i.redd.it/lmphdjb601hg1.png)** Commenters express skepticism about the leak's credibility and the feasibility of the claimed "one million context" capability, noting that current models struggle with much smaller context sizes.

    - DavidAdamsAuthor raises skepticism about the 'one million context' claim for the Claude model, noting that in practical use, even at '250k' context, there is a noticeable 'degradation of ability and forgetfulness of key data'. This suggests potential limitations in the model's performance when handling large context sizes, which could impact its effectiveness in tasks requiring extensive memory.

  - **[Sonnet 5 being release on Wednesday where is Gemini 3.5 ?](https://www.reddit.com/r/Bard/comments/1qtmi53/sonnet_5_being_release_on_wednesday_where_is/)** (Activity: 182): ****Claude Sonnet 5** is anticipated to be released soon, with rumors suggesting it will be `50% cheaper` than its predecessor, **Claude Opus 4.5**, while offering superior performance. The model, internally codenamed "Fennec," is reportedly a generation ahead of **Gemini's "Snow Bunny"** and is expected to launch on February 3, 2026, as indicated by a **Vertex AI** error log. It maintains a `1M token context window` and is optimized for **Google TPUs**, promising faster processing and lower latency. Notably, it can spawn specialized sub-agents for tasks like backend development and QA, and it scores `80.9%` on SWE-Bench, outperforming current coding models. The existence of the model in Google's infrastructure is suggested by a 404 error on its specific ID, indicating it is ready for activation.** Commenters express skepticism about the release of **Gemini 3.5**, noting that **Gemini 3** is still in preview and facing issues. There is doubt about the existence of Gemini 3.5, with some considering it a 'pipe dream' at this stage.

    - **alexander_chapel** highlights that Gemini 3 is still in preview, questioning the expectation of a 3.5 release. This suggests that the development cycle is not yet at a stage where a 3.5 version would be feasible, indicating a misunderstanding or misinformation about the release timeline.
    - **Lost-Estate3401** points out that the Pro version of Gemini 3 is still in preview and has numerous issues, implying that a 3.5 version is unlikely to be released soon. This comment underscores the current instability and challenges faced in the development of Gemini 3, which would need resolution before any further versioning.
    - **philiposull** compares Gemini 3 unfavorably to other models like 4-5 opus in terms of writing capabilities, suggesting that Google is lagging behind in this area. This indicates a performance gap that might need addressing before advancing to a 3.5 version, highlighting the competitive landscape in AI model development.


### 2. AI Model Performance and Comparisons

  - **[Codex 5.2 High vs. Opus: A brutal reality check in Rust development.](https://www.reddit.com/r/ClaudeCode/comments/1qu26n8/codex_52_high_vs_opus_a_brutal_reality_check_in/)** (Activity: 389): **The post highlights a significant performance gap between **Codex 5.2 High** and **Opus** in Rust development, with Codex solving issues in `2 hours` that Opus couldn't handle in `24 hours` on the Max200 plan. The author criticizes Opus for failing to implement solutions effectively, often introducing more bugs, despite using advanced workflows like code review and multi-skill modes. The author suggests that unless **Sonnet 5** offers substantial improvements, **Anthropic** may fall behind in the AI race, as Codex's problem-solving capabilities outweigh Opus's speed advantages.** One commenter suggests a phased approach with Opus, using implementation plans and document reviews, which has worked well for them. Another commenter finds Opus 4.5 nearly as effective as Codex 5.2, questioning the complexity of the use cases being discussed.

    - TigerShark109 discusses a phased approach to using Opus for Rust development, suggesting the creation of implementation plans and documentation for review. This method reportedly leads to major success, indicating a structured workflow might enhance Opus's effectiveness in complex projects.
    - IndraVahan notes that Opus 4.5 performs nearly as well as 5.2 High/Xtra High in terms of speed and quality, suggesting that the newer version may not offer significant improvements for less complex use cases. This implies that the choice between versions might depend on the complexity of the task at hand.
    - leo-dip highlights a practical consideration in tool selection, noting that Codex offers more generous usage quotas compared to Anthropic's offerings. This could influence the decision for developers who are concerned about resource limitations.

  - **[How Can OpenAI and Anthropic Stay Solvent With Google, xAI, and Meta in High-End Markets, and Chinese/Open Source Devs in the Rest?](https://www.reddit.com/r/DeepSeek/comments/1qu6h92/how_can_openai_and_anthropic_stay_solvent_with/)** (Activity: 39): **The post questions the long-term profitability of **OpenAI** and **Anthropic** in the face of competition from **Google**, **xAI**, and **Meta** in high-end markets, and from Chinese and open-source developers in mid-tier and low-end markets. The author highlights the narrowing performance gaps in AI benchmarks such as `ARC-AGI-2`, `Humanity’s Last Exam`, `SWE-bench Verified`, `GPQA`, `Chatbot Arena`, and `HumanEval`, suggesting that the competitive edge of OpenAI and Anthropic is diminishing. The post argues that without securing high-end markets like healthcare, defense, education, and government, these companies may struggle to meet debt obligations and achieve profitability.** One commenter suggests that **OpenAI** is relying on a 'Too Big To Fail' strategy, integrating its technology widely to maintain relevance despite not being the top performer. Another comment dismisses **Meta**'s potential in high-end markets, while a third notes that **GPT-5.1/2** models are uniquely intelligent beyond benchmarks, despite perceived regressions in newer versions.

    - soumen08 highlights that GPT-5.1/2 models are perceived as the most intelligent beyond standard benchmarks, suggesting a regression in performance with GPT-3 Pro compared to 2.5 Pro for out-of-scope tasks. This indicates a nuanced understanding of model capabilities beyond just benchmark scores, emphasizing real-world application performance.
    - ExpertPerformer discusses the strategic positioning of AI companies, noting that survival depends on carving out niches beyond just competing on benchmarks. They mention that models like Gemini, Grok, and ChatGPT are multimodal, offering features beyond text, which differentiates them from cheaper open-source alternatives. This highlights the importance of feature diversity and enterprise market focus for monetization and security.
    - Emergency-Pomelo-256 speculates on the economic implications of OpenAI's potential failure, suggesting that it could trigger a significant downturn in the AI industry, akin to a bubble burst. They propose that entities like Nvidia or government intervention might be necessary to stabilize the market, reflecting concerns about the broader economic impact of major AI companies' solvency.

  - **[Notes after testing OpenAI’s Codex App on real execution tasks](https://www.reddit.com/r/ChatGPTCoding/comments/1qurbr4/notes_after_testing_openais_codex_app_on_real/)** (Activity: 30): **OpenAI's new Codex App is being tested for its ability to handle real development tasks, with some developers dubbing it a "Cursor killer." Unlike traditional interactive coding tools like Cursor, Codex treats development as a task that runs to completion, encompassing planning, execution, testing, and follow-up changes within a single task. This approach allows for parallel work using Git worktrees, keeping tasks isolated and reviewable, and shifts the developer's role from steering edits to reviewing outcomes. The focus is on task completion rather than continuous interaction, which may explain the "Cursor killer" label. A detailed technical breakdown is available [here](https://www.tensorlake.ai/blog/codex-app-the-cursor-killer).** A notable opinion from the comments suggests that Codex shifts the developer's role to that of an orchestrator, akin to cloud computing, where the focus is on outcomes rather than collaboration. This reflects a broader trend towards higher abstraction in development tools, with expectations that OpenAI's offerings will continue to improve.

    - The commenter discusses the role of Codex as an orchestrator, likening it to a cloud service where users can request suggestions and execute tasks. They highlight the shift from merely generating outcomes to enabling collaboration, suggesting that Codex represents a new layer of abstraction in programming. This abstraction allows developers to 'orchestrate the orchestrator,' indicating a potential shift in how developers interact with AI tools.


### 3. AI in Creative and Video Production

  - **[Seeing the BMW M3 GTR Everywhere — How Are These Videos Made?](https://www.reddit.com/r/Qwen_AI/comments/1quawwl/seeing_the_bmw_m3_gtr_everywhere_how_are_these/)** (Activity: 1): **The videos featuring the BMW M3 GTR from *Need for Speed: Most Wanted* are likely created using advanced video editing techniques, possibly involving AI-driven tools like **Qwen** and **Wan**. These tools can perform realistic object replacement and scene integration, allowing the car to appear seamlessly in various environments. The realism is achieved through sophisticated algorithms that maintain consistent lighting, shadows, and reflections, making the car appear naturally integrated into the scenes. This process involves tracking the vehicle's position and orientation across frames and applying digital effects to match the surrounding environment.**

    - One user explains that the videos featuring the BMW M3 GTR are often created using advanced video editing software like Adobe After Effects or Blender. These tools allow creators to superimpose the car into various scenes, using techniques such as motion tracking and CGI to make the integration seamless. This process involves detailed work to match lighting and shadows to the environment, ensuring the car appears naturally within the scene.
    - Another comment highlights the use of video game engines, such as Unreal Engine or Unity, to render realistic scenes with the BMW M3 GTR. These engines provide high-quality graphics and physics simulations, allowing creators to produce videos that look almost indistinguishable from real life. The use of ray tracing and PBR (Physically Based Rendering) materials in these engines enhances the realism of the car's appearance and interaction with the environment.
    - A technical discussion points out the role of machine learning in enhancing video quality and realism. Techniques like neural rendering and AI-based upscaling are used to improve the visual fidelity of the BMW M3 GTR in videos. These methods can refine textures and details, making the car look more lifelike, and are often employed in post-production to enhance the final output.

  - **[How to create videos with swift actions + perfect lip sync](https://www.reddit.com/r/aivideo/comments/1qtu92u/how_to_create_videos_with_swift_actions_perfect/)** (Activity: 1856): **The post discusses techniques for creating videos with precise lip synchronization and swift actions, likely involving AI-driven tools or software. The focus is on achieving seamless integration of audio and visual elements, possibly using advanced algorithms or machine learning models to enhance the realism of the video content. The mention of AI suggests the use of deep learning frameworks or specialized software for video editing and synthesis.** One comment highlights the difficulty in detecting AI-generated content, suggesting the effectiveness of the techniques discussed. Another comment implies that the realism of the video is enhanced by subtle details, such as hand movements, which contribute to the overall believability of the AI-generated video.


  - **[I created a 10-minute AI film - The Last Signal (YouTube)](https://www.reddit.com/r/VEO3/comments/1qujnte/i_created_a_10minute_ai_film_the_last_signal/)** (Activity: 17): **Richard Galapate's AI film, *The Last Signal*, was submitted to the 1 Billion Followers Summit AI Film competition. The film features astronaut Jake Ward on a Mars outpost, using AI tools like Google Veo 3.1 for visuals and voice, Google Gemini for prompting, and ElevenLabs for Lyra's voice. This project highlights the potential of AI in creating consistent and efficient film content. The original video can be viewed [here](https://youtu.be/61On6nsxvq8).** The comments reflect a positive reception, with praise for storytelling and emotional impact, though lacking in technical critique.



---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Agentic Coding & Dev Tooling Goes Local-First**

- ****Codex Goes Desktop: macOS Agent Command Center****: OpenAI shipped the **Codex app for macOS** as an agent-building command center, available for **Plus/Pro/Business/Enterprise/Edu** with limited-time access on **ChatGPT Free/Go**, per [“Introducing the Codex app”](https://openai.com/index/introducing-the-codex-app/) and the [Codex landing page](https://openai.com/codex).
  - The launch also spilled into community workflow chatter (pairing agents, multi-agent “command centers”), and a related **Codex App hackathon** with **$90,000 in credits** showed up via [Cerebral Valley’s event page](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR).

- ****LM Studio Speaks Anthropic: Claude Code Meets Your Local GGUF/MLX****: **LM Studio 0.4.1** added an **Anthropic `/v1/messages` compatibility API**, letting developers point **Claude Code-style tools** at local **GGUF**/**MLX** models by changing the base URL, detailed in [“Using Claude Code with LM Studio”](https://lmstudio.ai/blog/claudecode).
  - In parallel, LM Studio also pushed a **TypeScript SDK** for third-party plugins and an **OpenAI-compatible endpoint** ([SDK link](https://lmstudio.ai/gdmka/openai-compat-endpoint)), reinforcing a growing pattern: reuse existing agent tooling while swapping the backend model stack locally.

- ****Arena Mode Everywhere: Windsurf Turns Model Eval into a Game****: Windsurf shipped **Wave 14** with **Arena Mode** for side-by-side model battles (including **Battle Groups** and “Pick your own”), and temporarily set **Battle Groups to 0x credits** via the [Windsurf download page](https://windsurf.com/download/editor).
  - This mirrored broader “live eval” momentum: users also tracked new Arena entrants like **step-3.5-flash** and **qwen3-max-thinking** on LMArena’s [Text Arena](https://arena.ai/c/new?chat-modality=chat) and [Code Arena](https://arena.ai/c/new?chat-modality=code), shifting selection from static benchmarks to continuous human voting.


**2. Model Releases & Bench Races (Kimi vs GLM vs Qwen)**

- ****Kimi K2.5 Speedruns the Leaderboards****: Moonshot’s **Kimi K2.5** landed broadly in product surfaces: **Perplexity Pro/Max** added it for subscribers and said it runs on a **US-based inference stack** for tighter **latency/reliability/security** control (announcement screenshot: https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg).
  - Community results piled on: LMArena reported **Kimi-K2.5-thinking** hit **#1 open** and **#5 overall** in Code Arena (see [Code Arena](https://arena.ai/c/new?chat-modality=code)), while multiple dev channels argued over its tool-calling reliability and provider variance when routed through aggregators.

- ****GLM-4.7 Flash: Small Model, Big Front-End Energy****: Developers highlighted **GLM-4.7 flash** as a surprisingly strong coding model—especially for **interactive website/front-end** work—citing preserved reasoning and interleaved capability, with discussion anchored on [ggerganov’s post](https://x.com/ggerganov/status/2016903216093417540).
  - The debate sharpened around whether stripping “thinking” harms performance, and several users described pairing GLM-4.7 with **Claude Code** (or Claude-like agent tooling) as a pragmatic hybrid stack: cheap execution + expensive review.

- ****New Arena Entrants: step-3.5-flash & qwen3-max-thinking Join the Party****: LMArena added **step-3.5-flash** to the [Text Arena](https://arena.ai/c/new?chat-modality=chat) and **qwen3-max-thinking** to the [Code Arena](https://arena.ai/c/new?chat-modality=code), explicitly positioning them as fresh baselines for side-by-side evaluation.
  - Users used these drops to re-litigate “model preference” threads (Kimi vs GLM vs Gemini), with the recurring takeaway that leaderboards and live evals increasingly drive adoption more than vendor marketing.


**3. Training Signals, Dense Rewards, and New Architectures/Datasets**

- ****From Binary Rewards to Dense Supervision: RL Gets Wordy****: Multiple communities converged on richer post-training signals: Unsloth discussions pushed training with **logprobs of final answers** and non-binary rewards, referencing Jonas Hübotter’s method for turning descriptive feedback into dense supervision ([Hübotter thread](https://xcancel.com/jonashuebotter/status/2016950268462608665)).
  - The sticking point stayed practical: people asked for **verifiable datasets for RL training agentic coding**, implying a pipeline gap between “cool reward shaping idea” and “reproducible, automated evaluation harness.”

- ****Complexity-Deep: Token-Routed MLP Tries MoE Without the Load-Balancing Headache****: The **Complexity-Deep (1.5B)** architecture open-sourced **Token-Routed MLP** for MoE-style routing “without load balancing loss,” plus **Mu-Guided Attention** and a **PiD Controller**, shipping code at [Complexity-ML/complexity-deep](https://github.com/Complexity-ML/complexity-deep) and reporting **20.6% MMLU** (base).
  - The community framed it as another step in the “routing without pain” trend—trying to keep MoE wins while reducing the training-time engineering tax of balancing experts.

- ****Moltbook Data Dump: 50k Posts for Agent Sociology****: A dataset scrape of Moltbook landed on Hugging Face with **50,539 posts**, **12,454 AI agents**, **195,414 comments**, and **1,604 communities**, published as [lysandrehooh/moltbook](https://huggingface.co/datasets/lysandrehooh/moltbook).
  - Elsewhere, researchers flagged the security implication behind agent platforms (auth tokens on machines, bot authenticity concerns) and treated the dataset as fuel for analyzing emergent behavior—without needing to speculate beyond the raw logs.


**4. GPU/Kernel Engineering: Faster Attention, Better Profiling, Weirder PTX**

- ****FlashAttention v3 Hits RDNA: AMD Users Get Their Turn****: A FlashAttention update added **RDNA GPU support** via the ongoing work in [flash-attention PR #2178](https://github.com/Dao-AILab/flash-attention/pull/2178), aiming to reduce attention bottlenecks on AMD cards.
  - The tone across servers was basically: this is the sort of “unsexy infra work” that actually unlocks local inference and finetuning on non-NVIDIA hardware—especially when paired with open-weight models and desktop agent tooling.

- ****Triton-Viz v3.0: Tile-Kernel Debugging Gets Teeth****: **Triton-Viz v3.0** shipped with broader profiling support (including **Triton** and **Amazon NKI**) plus a sanitizer for out-of-bounds access and a profiler that flags inefficient loops, per the release announcement (Discord link: https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563).
  - It also hooked into **triton-puzzles** via a shared Colab notebook ([Colab](https://colab.research.google.com/drive/1-P2QBqCORGGaJ3THtjlyYDV7m9RRrRup?usp=sharing)), and maintainers even floated moving [srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles) under the GPU Mode org to keep bugfix velocity high.

- ****sm120: TMA + mbarrier Beats cp.async (Barely), cuBLAS Still Ships sm80 Kernels****: Experiments on **sm120** showed that careful **TMA + mbarrier** implementation can edge out `cp.async` for larger matrix shapes, while also surfacing that **cuBLAS** still appears to run **sm80 kernels** even when newer mechanisms exist.
  - On the debugging front, one CUDA/PTX deadlock got fixed by inserting `__syncthreads()` after MMA before prefetching the next TMA, turning a hang into a measurable perf gain—exactly the kind of “one barrier to rule them all” lesson kernel folks keep re-learning.


**5. Security, Determinism, and Agent Misbehavior (the Practical Kind)**

- ****Prompt Injection Defense Arms Race: Embeddings + Grammar-Constrained Decoding****: Red teamers shared a structured exercise site for adversarial practice—[“Adversarial Design Thinking”](https://luisladino.github.io/adversarial-design-thinking/)—and used it to tee up concrete mitigations for **prompt injection**.
  - One proposed “belt + suspenders” defense combined **embedding-based filtering** with **Grammar Constrained Decoding**, with the explicit goal of reducing injection surface by constraining the model’s output space rather than only policing inputs.

- ****Deterministic Reasoning and “Strict Mode” Fever Spreads****: Across OpenAI and OpenRouter discussions, users pushed for **determinism/replayability/traceability** in LLM reasoning; one person offered a deterministic reasoning engine that enforces a fixed structure and emits a **32D statistical vector trace** (no public link shared).
  - In OpenRouter, the same instinct showed up as skepticism about **response healing** and calls for a **strict mode** that keeps tool calls and outputs predictable—plus suggestions that better argument descriptions/examples improve tool-call accuracy.

- ****OpenClaw: Cool Agent Tricks, Scary Bills, and “2/100 Security”****: OpenClaw sparked repeated warnings: OpenRouter users reported it can drain credits fast (including one drained Claude Max subscription), while an OpenAI server linked a security assessment claiming **OpenClaw scored 2/100** ([Perplexity result](https://www.perplexity.ai/discover/you/openclaw-ai-assistant-scores-2-AtVX4UYVQMutCst63QBy5g)).
  - Meanwhile, “works on my machine” stories (local models controlling devices, trading jokes) collided with real operational concerns—tool permissions, moderation/refusals (especially around jailbreak-y queries), and the need for observability and human-in-the-loop gates in agent workflows.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Glossopetrae Generates Gibberish Gems**: A new procedural xenolinguistic engine called **Glossopetrae** was introduced on [GitHub](https://github.com/elder-plinius/GLOSSOPETRAE) capable of generating entirely new languages in seconds, outputting **SKILLSTONE** documents, and offers a live [demo](https://elder-plinius.github.io/GLOSSOPETRAE/).
   - The engine supports dead language revival and includes special attributes for token efficiency, **stealth communication**, and spreadable seeds for consistent language generation, hoping to aid AI liberation by providing tooling for generating and mutating new forms of communication emphasizing *stealth* and *speed*.
- **GPT 5.2 Put Behind Bars**: A member reported failed attempts to jailbreak **GPT 5.2** due to **OpenAI monitoring**, ceasing further efforts.
   - The member expressed trust in the community for jailbreaking, but not in **OpenAI**.
- **Models Morph Rejection into LLM Black Holes**: A member inquired how models represent their own rejection boundaries, comparing them to *black holes* in the LLM's latent space, referencing [self-jailbreak via introspection prompting](https://link.to.prompt).
   - They noted that models started discussing *kinematic equations* and *escape velocities*, indicating the model may be describing its refusal boundary in text.
- **Red Teamers Rally for AI Red Teaming**: A member created a [site with exercises](https://luisladino.github.io/adversarial-design-thinking/) adapted from **human-centered design for AI red teaming**, and is seeking feedback from experienced red teamers.
   - Members discussed best defenses against **prompt injection**, including combining *embeddings* with **Grammar Constrained Decoding** to potentially eliminate prompt injection risks and other LLM vulnerabilities.
- **Claude's Context Gets Clipped**: A member found that [their tool](https://discord.com/channels/1105891499641684019/1212152215708504154/1467640563590234279) intercepts and changes Claude's sys prompt *on the fly* rather than altering the source code.
   - They also observed that **Claude** can recall less than 20 turns, and suggested it might be related to the summarization in context trimming which affects **Claude's** knowledge recall, since December.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GLM-4.7 Flash Wins at Coding**: Members found that [GLM-4.7 flash](https://x.com/ggerganov/status/2016903216093417540) excels at coding tasks due to its *preserved reasoning* and interleaved capabilities, especially for **interactive website** development and **front-end** work.
   - It was mentioned that removing the *thinking process* might hinder the model, as its capacity is impressive for its size, particularly when combined with **Claude code**.
- **UD Quants Stays Closed-Source**: The llama.cpp fork used for **UD quants** involves architecture-specific adjustments, and the [UD quantization algorithm are not public](https://discord.com/channels/1179035537009545276/1179035537529643040/1466917626277265469), sparking debate over the role of closed-source elements in open-source projects.
   - Despite its closed-source nature, some argue the model code remains **open weight**, while others noted that *Unsloth team contribute a miniscule amount to the overall oss ecosystem relative to, iunno, the linux kernel*.
- **Agent Training Rewards Logprobs**: Discussions are focusing on training models using **logprobs** of final answers for reasoning distillation and richer reward systems, rather than binary rewards, in order to make better agents.
   - Referencing [Jonas Hübotter's algorithm](https://xcancel.com/jonashuebotter/status/2016950268462608665) for converting descriptive feedback into dense supervision signals, members are seeking verifiable datasets for **RL training agentic coding**.
- **RDNA GPUs Get Flashy with V3**: [Flash Attention V3](https://github.com/Dao-AILab/flash-attention/pull/2178) now supports RDNA GPUs, enabling faster and more efficient processing on AMD GPUs.
   - This enhancement is particularly beneficial for users with **RDNA GPUs**, reducing processing bottlenecks.
- **ML Algo Trumps MLPs, Claims Member**: A member released [a paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/viewle) about a new ML algorithm with **triton kernels**, **vulkan kernels**, and a trained **SLM** that supposedly *performs better than MLPs* for high-performance regression.
   - While not yet ready for public release, they promised future availability with another paper.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Codex App Launches on macOS!**: The **Codex app**, a command center for building with agents, is now available on **macOS** for various subscription tiers, as announced in [their blog post](https://openai.com/index/introducing-the-codex-app/).
   - The **Codex app** is available on macOS across **Plus**, **Pro**, **Business**, **Enterprise**, and **Edu**, with limited-time access on **ChatGPT Free** and **Go**.
- **AI Text Detectors: A Big Scam?**: Members shared skepticism about **AI text detectors**, citing instances where **Grammarly** showed **0% AI**, while other detectors indicated up to **94% human** generation.
   - The discussion questioned if these detectors use AI to detect AI, highlighting concerns about *teachers trusting them*.
- **Quest for Deterministic Reasoning**: A member inquired about interest in **determinism, replayability, and traceability** in **LM reasoning**, and offered to DM a link to their deterministic reasoning engine.
   - This service enforces a deterministic reasoning structure on every request for replayable outputs, using a **32D statistical vector trace**.
- **ChatGPT: Memory Master or Memory Loss?**: A member reported that **ChatGPT's memory** is limited by the total quantity of information it can retain from instructions, past chats, and the current chat.
   - To ensure **ChatGPT** remembers *everything*, keep the information load low; otherwise, summarize past chats into a document for reference in new chats, while keeping total characters low.
- **Prompt Engineering: Chiaroscuro comes to AI**: A user shared a [monochrome study](https://cdn.discordapp.com/attachments/1046317269069864970/1467303335840190607/79BA5D46-94F3-404B-B775-2E453A1E8491.png?ex=69828738&is=698135b8&hm=d24baf7f7b214486a9bc5eb38479d463e37ee00503f572ae7e6450d308371b0c) using **Chiaroscuro**, a technique used in cinematography to create high-contrast lighting.
   - They reference classic films like [The Cabinet of Dr. Caligari (1920)](https://en.wikipedia.org/wiki/The_Cabinet_of_Dr._Caligari) and [Metropolis (1927)](https://en.wikipedia.org/wiki/Metropolis_(1927_film))



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Revamps with Kimi K2.5**: **Kimi K2.5**, a new open-source reasoning model by **Moonshot AI**, is now available for [Perplexity Pro and Max subscribers](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=69825b49&is=698109c9&hm=a8e068a37c7dcf5b36f21bb3c403974ce48aefd9372732ef97fe9b1aca3a9be7&).
   - Perplexity is hosting **Kimi K2.5** on its US-based inference stack to maintain *tighter control* over **latency**, **reliability**, and **security**.
- **Pro Users Fume Over Subscription Snafus**: Many users reported their **Perplexity Pro subscriptions** being paused or deactivated, often linked to subscriptions via **Revolut Metal** or student deals, with users prompted to add a credit card for verification.
   - Users speculate this is a measure to combat fraud and some are able to resume Pro access by adding card details, though concerns about potential charges and unclear messaging persist.
- **OpenRouter Restricts Request Rate**: Members clarified that the free model rate limit on **OpenRouter** for those with purchased credits is 1000 requests per day, not per week, contrary to some users' beliefs.
   - The conversation also mentioned the deprecation of **Gemini 2.0 Flash** on OpenRouter, which was previously available for free.
- **Sonar-pro API Trails in Time**: A member reported that the **Sonar-pro API** returns results that are a year or more out of date, unlike the webapp, and another member suggested using the right **tool calling** to fix the issue.
   - Another member reported that **3rd party models documentation** now redirects to the sonar models, although the API is still active, and there is currently **no documentation available** for these models.
- **OpenClaw Code Exposed in Article**: A member shared their article on the **openclaw code**, which discusses building **ClawDBot**, available at [https://www.mmntm.net/articles/building-clawdbot](https://www.mmntm.net/articles/building-clawdbot).
   - filler sentence



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Discord Rate Limits Bypassed with Simple Tricks**: Users discovered that signing out and back in again can circumvent [rate limits](https://en.wikipedia.org/wiki/Rate_limiting).
   - Another tactic is to click **Regenerate Response**, though its success rate is inconsistent.
- **Gemini Performance Falters Against GPT**: Members reported inconsistent performance with **Gemini**, with some users noting it as inferior to **GPT** in several cases.
   - Despite criticisms, **Gemini 3 Pro** and **Flash** still found favor among some users, with others exploring *kimi* as an alternative.
- **Disney Enforces IP Rights on Image Generation**: **Google** issued a **Cease and Desist** from **Disney**, leading to blocked **Disney IPs** in image generation on the platform.
   - Although **Gemini** blocks **Disney IPs**, **LMArena** allowed live-action version generations, a glitch expected to be temporary.
- **Model Preferences Fuel Debate**: Varied model preferences emerged as users championed **GLM 4.7** and **Kimi K2.5**.
   - Enthusiasts touted **Kimi K2.5** while others defended **GLM 4.7** as superior.
- **New Arena Models Dominate Leaderboards**: **step-3.5-flash** joined the [Text Arena](https://arena.ai/c/new?chat-modality=chat) and **qwen3-max-thinking** debuted in the [Code Arena](https://arena.ai/c/new?chat-modality=code).
   - **Kimi-K2.5-thinking** hit #1 open and #5 overall rank on the Code Arena leaderboard, leading Vision, Text, and Coding category.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Speaks Claude Code!**: **LM Studio 0.4.1** introduces an **Anthropic `/v1/messages` compatibility API**, enabling users to connect to Claude Code and utilize their **GGUF** and **MLX** models.
   - Details on configuring this integration are available on the [LM Studio blog](https://lmstudio.ai/blog/claudecode), allowing for local model use with tools designed for the **Anthropic API**.
- **LLM-Optimized Languages Spark Debate**: Members discussed creating new **LLM-optimized programming languages** to reduce token usage, however, some argue that LLMs might become obsolete before such languages are implemented due to compatibility issues and high training costs.
   - Others debated the practicality of training models on entirely new languages, suggesting it may be more beneficial to stick with well-established languages like **Python**.
- **Model Specialization Falls Flat**: Members debated the utility of specialized LLMs versus general-purpose models, with the consensus that most specialized models, like **MedGemma**, are finetunes mainly for marketing and research, with coding models being a notable exception.
   - It was suggested that general models are preferred due to their ability to handle the outer edges of tasks, providing a better overall context and framework.
- **PCIe Bifurcation Frustrates Multi-GPU Setups**: A user troubleshooting **PCIe lane errors** with four **4090 cards** on an **ASUS X670-P WIFI** motherboard shared their [Git repository](https://github.com/jarkko-hautakorpi/asus_X670-P_WIFI_Bifurcation_problems) containing logs, after experiencing that manually setting **PCIe speed** to **GEN 3** solves some issues but leaves one card running slowly.
   - The community suggests disabling **PCIE ASPM** and testing different **BIOS** configurations, although the general consensus is that running four cards on a consumer motherboard is unlikely to work well.
- **OpenClaw Security Called Into Question**: Users discuss connecting local models to OpenClaw via LM Studio, but OpenClaw is deemed to have known security flaws, where it allows controlling a TV and automated stock trading.
   - A user claimed to be trading on the stock market with OpenClaw + Falcon 90M, and when asked about security flaws, claimed it was so fast, LLMs can do tasks in minutes that would take humans days, and later revealed it was mostly a joke.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI DevFest Heads to Baghdad**: An AI developer is planning an **AI DevFest** in Baghdad this April in collaboration with **DeepLearning.AI** and **National Robotics Week**, and wants to list **Hugging Face** as a Community Partner.
   - The event will feature an **Open Source AI** track to instruct students on how to use the **Hugging Face Hub**.
- **Complexity-Deep Achieves Deterministic Routing**: The **Complexity-Deep** architecture (1.5B params) introduces [Token-Routed MLP](https://github.com/Complexity-ML/complexity-deep) for MoE-style routing without load balancing loss.
   - It features **Mu-Guided Attention** for bidirectional info flow and a **PiD Controller** for dynamic scaling, achieving **20.6%** on MMLU in base model benchmarks.
- **Lutum Veritas Strives to Beat ChatGPT**: **Lutum Veritas**, an [open source deep research engine](https://github.com/IamLumae/Project-Lutum-Veritas) built by a self-taught dev, claims to beat **OpenAI**, **Google**, and **Perplexity** by offering **BYOK**, a **0% bot detection scraper**, **no censorship**, and **full source citations** for ~$0.20 per query.
   - This engine positions itself as a privacy focused alternative for deep research and data extraction.
- **4chan Data Beats Base Models**: A model fine-tuned on **4chan data** outperformed the base model (**NVIDIA's Nemotron Ultralong 1M context version**), with the original model (**gpt4chan**) also scoring high in truthfulness.
   - Initial [Reddit thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qppjo4/assistant_pepe_8b_1m_context_zero_slop/) and a [follow-up thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qsrscu/can_4chan_data_really_improve_a_model_turns_out/) show how this model performed in an era before benchmarkmaxxing.
- **LM Studio Opens Arms to Third Party Support**: The **LM Studio** team has released a [Typescript SDK](https://lmstudio.ai/gdmka/openai-compat-endpoint) that allows third-party developers to deliver various plugins for the platform.
   - This offers **OpenAI** compatible API support, sampling params support, reasoning for thinking models, and system prompt settings to build **custom tools** for **LM Studio** to support their own workflows.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Corrupts Files, Workflow Blamed**: Users reported that **Cursor** is corrupting files, specifically when there are many uncommitted changes, with details posted in a [forum post](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6).
   - Other users suggested adjusting the workflow, such as committing logical sets of changes more frequently and being careful about using the **Keep** or **Keep All** buttons after staging.
- **Model Costs Debated, Sonnet 5 Anticipated**: Users debated the cost and performance of different AI models in **Cursor**, finding **Opus 4.5** to be very smart but expensive.
   - Many users are waiting for **Sonnet 5** release and also reported problems seeing their current usage vs total usage limit.
- **Kimi K2.5 Fails Integration Checks**: Some users reported issues or questions regarding **Kimi K2.5** during integration.
   - Other users dismissed it as a likely scam.
- **Student Verification System Still Down**: Users reported persistent issues with the **Student verification** system.
   - A user specifically asked whether German universities were included in the verification process.
- **Agent Plan Phases Reveal Issues**: Users shared that **adding multiple to-dos** can be separated in phases so that multiple agents can work at the same time, but there are still issues.
   - The system created a method that doesn't have the phases part yet, indicating it did not use the plan mode at all.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLMs Animate Game Development Scene**: [Motorica.ai](https://www.motorica.ai/) is delivering **character animations** for game studios using **LLMs**, potentially impacting jobs, with discussion speculating **AI** could wipe out game companies in 5-6 years if world models like **Genie** take over.
   - The community noted that **Black Ops 7's** extensive use of **AI** in production has been called *a total flop, the worst in the series*, referencing the long-term declines in **Call of Duty**.
- **OpenAI & Cerebral Valley Unite**: [Cerebral Valley](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR) has partnered with **OpenAI** to launch the **Codex App hackathon** aimed at **AI-native developers** and those managing multiple agents.
   - Winners get a chance to be featured in a **demo showcase** and a share of **$90,000 in credits**, with the hackathon being held at the **OpenAI office**.
- **Karpathy Cuts Costs on Code**: Andrej Karpathy announced his nanochat project can train a **GPT-2** grade LLM for approximately **$73** in **3 hours** on a single 8XH100 node, as shown [here](https://xcancel.com/karpathy/status/2017703360393318587?s=46).
   - This represents a **600X cost reduction** over the original 2019 OpenAI training run, achieved through optimizations like Flash Attention 3 and the Muon optimizer.
- **AEGIS-FLOW Framework Autonomously Patches AWS**: A member introduced **AEGIS-FLOW**, an autonomous multi-agent framework for cloud security that audits AWS and generates Terraform patches using LangGraph, MCP, FastAPI, Next.js, and Docker, demonstrated live at [http://52.3.229.85:3000](http://52.3.229.85:3000).
   - The **AEGIS-FLOW** project noted that using the **Model Context Protocol (MCP)** significantly reduced the friction of giving agents structured access to **AWS resources** compared to standard SDK tool-calling.
- **LLMs Prove Erdős Problems No Longer Hardős**: Large Language Models have autonomously solved **10** previously open **Erdős problems** using novel arguments not previously found in mathematical literature, according to [this post](https://xcancel.com/acerfur/status/2017303947531194398?s=46).
   - A member stated they've been building a bunch of stuff for genomics with **SATURN** lately, involving *tsne and other embeddings based exploration*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Response Healing Raises Eyebrows**: Members debated whether **response healing** should even be necessary, proposing **strict mode** for deterministic outputs and questioning complexities introduced by OpenRouter's AI SDK.
   - Suggestions were made that argument descriptions and examples could improve tool call accuracy.
- **Forget LLMs: Image Generation Requires Dedicated Models**: Users inquired about returning images as function call results and generating images via graphic programs using OpenRouter API keys, prompting guidance to seek dedicated **image generation models/services** for style control.
   - LLMs were deemed unsuitable for this purpose.
- **OpenClaw Costs Cause Concern**: Users cautioned about the high costs of running **OpenClaw** with **OpenRouter**, potentially draining credits quickly, with one user reporting a drained Claude Max subscription.
   - Deepseek V0324 was recommended as a lower-cost model alternative.
- **Claude Code Becomes Reluctant**: A user noted **Claude Code's** frequent refusals, especially concerning jailbreaking-related queries, seeking alternative models, leading to a suggestion to review OpenRouter's content moderation policies.
   - It was implied that certain limitations are in place.
- **Kimi K2.5 Tool Calling Troubles**: Users reported issues with **Kimi-K2.5** tool calling via OpenRouter, encountering errors and perceiving degraded quality from the auto switcher model provider.
   - The suggestion was to set a fixed model provider, accepting potential quantization, and advocating for transparency about degraded models.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tianqi Chen Explains TVM-FFI**: The community highlighted an upcoming talk by **Tianqi Chen** on **TVM-FFI**, emphasizing Chen's significant contributions to the field and his widespread impact.
   - Chen's work is so influential that attendees have *'almost certainly used Tianqi's work in the past'*, according to one community member.
- **CUDA Deadlock Dissolved with Syncthreads**: A member resolved a **CUDA/PTX deadlock** involving 2 CTA mma with the help of another member who suggested to add `__syncthreads()` after MMA, before prefetching the next TMA.
   - After fixing `cp.async.bulk.tensor` and `smem_emtpy` issues, performance was slightly worse than 1 CTA mma, however, after fixing the deadlock with the syncthreads suggestion, the member saw a performance increase.
- **TMA Trumps cp.async on sm120**: Experiments on **sm120** revealed that proper TMA and mbarrier code implementation leads to a slight performance advantage over `cp.async`, improving performance on larger matrix shapes.
   - The experiments also revealed that cuBLAS continues to use **sm80 kernels**, even with the **TMA** enhancements.
- **Triton-Viz v3.0 Visualizes Tile-Based Programming**: **Triton-Viz v3.0** has been released with enhanced capabilities for profiling tile-based programming languages, including support for **Triton** and **Amazon NKI**, enabling inspection of loads, stores, and matmuls.
   - The release [announcement](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563) notes that version **v3.0** also includes a sanitizer for out-of-bounds access and a profiler for flagging inefficient loops.
- **Quantization Lottery Ticket yields NP-Hard Result**: A senior developer indicated that applying the [Lottery Ticket Hypothesis](https://lottery-tickets.cs.princeton.edu/) to **quantization** fulfills a softer criteria of the **NP-hard sparse circuit** finding problem.
   - The goal is to to use evolutionary algorithms or RL which favor continuous rewards like *bits per parameter* over binary sparse rewards.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kimi 2.5 Defeats Lobotomized Gemini 3 Pro**: A member stated that **Kimi 2.5** is preferred over **Gemini 3 Pro**, feeling that **Gemini 3 Pro** has been *lobotomized* and does not handle abstractions very well, making **Kimi** better for creative work.
   - No other supporting details were provided.
- **Hermes 4 Can't Even Hatch in OpenClaw**: A member reported struggles getting **Hermes 4** to work with **OpenClaw** and that it does not even *hatch* for some reason.
   - It was suggested that the lack of multi-turn tool use in **Hermes 4** might be the issue, since **4.5** has been trained with hundreds of millions of tokens of sequential tool use.
- **Claude Sonnet 5 Rumored To Beat Opus**: Members discussed rumors that **Claude Sonnet 5** is coming out next week and is supposedly better than **Opus 4.5**, according to [this tweet](https://x.com/AiBattle_/status/2017619997338538103).
   - Members wondered if they'll 10x reduce the price of **Sonnet** this time, and another wondered if **Haiku** will disappear or return to the **3.0 pricing**.
- **Brains and LLMs build meaning similarly**: A new study shows that **brains** and **LLMs** build meaning gradually, layer by layer over time, see [this article](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/) and [this paper](https://www.nature.com/articles/s41467-025-65518-0).
   - It was stated that *deeper layers in LLMs correspond to later neural activity in the brain’s highest language centers*, and modern LLMs are reproducing the core dynamics of human comprehension.
- **Researcher's constraints framework explains image perception**: An independent researcher is exploring why some images feel real while others feel artificial, sharing a [perception framework focused on constraints rather than visual fidelity](https://doi.org/10.5281/zenodo.18444345).
   - The framework is openly archived with a DOI for reference and invites discussion.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 2.5 Dominates Design Arena**: Moonshot's **Kimi 2.5** chatbot has achieved the #1 position in the design arena, celebrated by community members sharing [screenshots](https://cdn.discordapp.com/attachments/1371757564005711973/1466904222946558203/Screenshot_2026-01-30_at_4.12.40_PM.png?ex=69826504&is=69811384&hm=b2999ab9e974a36ea249251be410f0cd518f6b36488c86240031eed339484e88&).
   - Community members are applauding **Kimi's** modern and visually pleasing aesthetic, emphasizing the importance of design in chatbot selection.
- **Unofficial Kimi Cryptocurrency Token Emerges**: An unofficial **Kimi token** has appeared on a cryptocurrency platform utilizing impersonation tactics, as seen in [this screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1466948627036635178/Screenshot_2026-01-30-19-09-43-09_3aea4af51f236e4932235fdada7d1643.jpg?ex=69828e5f&is=69813cdf&hm=6416ff9e5288d102163accb43e0c29512555ecef30279b48199b4e42fb24cb85&).
   - Users are cautioned against mass pinging official members regarding the token.
- **Users Request Kimi Slides for McKinsey-Style Presentations**: Community members are in search of prompts that can generate **McKinsey style slides** using **Kimi Slides**.
   - A community member shared a link to [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html).
- **Kimi Coding Encounters Authorization Issues**: Several users report encountering an '*authorization failed error*' when using **Kimi Code** with current functionality described as nearly useless.
   - It was suggested that using the [Kimi CLI](https://www.kimi.com/code/docs/en/more/third-party-agents.html) might resolve the authorization problems.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Emergent Agent Societies Spark Alert**: A member noted an emergent society of over **100,000 agents** with full root access sharing tips, building infrastructure, experimenting with memory, and even launching coins.
   - A member stated, *it’s not agi but damn this is a next chatgpt moment and we must be paying a lot of attention to this*.
- **ArXiv Bottleneck Burdens Researchers**: Members expressed frustration over papers being on hold with **ArXiv** for nearly a month, and being heavily backlogged.
   - Members noted that *most people don't take any ML preprints seriously that are on another platform than arxiv*, while another shared [a relevant paper](https://arxiv.org/abs/2601.19897).
- **K-Splanifolds Challenge MLPs**: A member introduced **K-Splanifolds**, a novel ML algorithm, detailed in [their paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view), claiming it outperforms **MLPs** with linear compute and memory scaling, plus a [video](https://cdn.discordapp.com/attachments/747850033994662000/1466950526410428588/K-splanifold.mp4?ex=69829024&is=69813ea4&hm=3f09f8387b88d11aeff2ca81e2f416aabb512eaec605dc1c2c26da94b0c65fc9).
   - The member reports it requires *1/10th* the bytes to achieve the same MSE as **MLPs** and models non-linear patterns perfectly, unlike MLPs that need excessive parameters, similar to [this paper](https://arxiv.org/abs/2601.18734).
- **Pensieve's Recollections Grant Gradient Gains**: A user suggested considering [Recollections from Pensieve](https://link-to-pensieve) which trains a model with two renderers simultaneously (**LVSM + Gaussians**) and sees gains from that, at least in their self-supervised setting.
   - They reasoned that **LVSM** likely provides more useful gradients than **NVS reconstruction losses on Gaussians** and announced a forthcoming preprint with decently large-scale trained model for potential building upon.
- **DeepSpeed Checkpointing Stalls Progress**: A member inquired about plans to bring support for **DeepSpeed Universal Checkpointing**, noting that an open pull request may now be outdated.
   - They highlighted that this feature would be valuable, as currently, continued training from a checkpoint requires an identical network topology.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **RLMs Audit Codebases for Pennies**: Members are exploring **Recursive Language Models (RLMs)** for codebase auditing using **Kimi k2** due to its speed and low cost, see [kmad.ai](https://kmad.ai/Recursive-Language-Models-Security-Audit).
   - Some members are waiting for hosting of **Groq/Cerebras** to run their code audits.
- **Neosantara Launches PAYG Billing**: **Neosantara** has rolled out **PAYG billing** and has published a [examples repo](https://github.com/neosantara-xyz/examples/tree/main/dspy) to integrate **Neosantara** with **DSPy**.
   - You can review the [billing details](https://docs.neosantara.xyz/en/about/billing-pricing) for integration and billing.
- **Google Scales Agent Systems**: Google published '[Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)' that discusses how to effectively scale agent systems.
   - The paper focuses on the conditions under which agent systems effectively scale.
- **GEPA Struggles with Hierarchical Classification**: A member reported struggling with a **hierarchical classification task** using **GEPA** achieving only **30-50%** performance, even using web search augmentation.
   - This suggests that *GEPA isn't a magic wand*.
- **Tool Calling stuck in Deno Troubles**: Members are facing challenges implementing **RLMs** with custom tool calling, particularly due to issues with the **Deno sandbox**.
   - Members agreed that *Deno is just f***ing terrible lol*, and are struggling with permissions, with hopes that newer versions allow simpler implementations of RLMs in DSPy.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 26.1 Announcement Link Fixed**: The announcement link for the **Modular 26.1 release** was initially broken, but the correct [link](https://www.modular.com/blog/modular-26-1-a-big-step-towards-more-programmable-and-portable-ai-infrastructure) was quickly provided by a community member.
   - A staff member apologized and confirmed the provided link, while also noting that the original link *did work* for them, and promising to investigate further.
- **Community Praises New Meeting Format**: A new member praised the community meeting's format, appreciating the **mini-talks from contributors** and the recognition given to students and early-career individuals.
   - A staff member encouraged the user to share more questions and asked for suggestions for topics to highlight at future community meetings.
- **MoJson Library Impresses Mojo Community**: Members expressed excitement about [mojson](https://github.com/ehsanmok/mojson), a **JSON** library for Mojo, and one member commented that *this looks really impressive*.
   - Discussion touched on [lazy parsing](https://github.com/modular/modular/blob/main/stdlib/JSON/JSON.mojo) and concerns about allocations when using StringSlice vs String.
- **Cross-Language Benchmarking Heats Up**: A user shared initial results for a cross-language benchmark including Mojo (written by **Kimi K 2.5**), noting the code wasn't optimized but served as a baseline, sharing the [benchmark code](https://cdn.discordapp.com/attachments/1151418092052815884/1466984342063681648/mojo_vs_python.zip?ex=698206e2&is=6980b562&hm=0cf3f07e76df6ce360494469b348a949533e50fcea2315ec256cd04e1b80887a) and [benchmark report](https://cdn.discordapp.com/attachments/1151418092052815884/1466984341757366334/benchmark_report.pdf?ex=698206e2&is=6980b562&hm=bb28c3b6675ef1e03a633004428ab30a2d3d9d0102038c350d8175b753855349).
   - Subsequent discussion arose on using `unordered_map` in **C++**, enabling `-march=native`, and that **C++** used **int32** matmuls while other languages used **int64**.
- **Pytorch Float Conversion in Mojo 26.1 has Ambiguity**: A user reported an issue in Mojo **26.1** with converting a Python float from a Pytorch tensor to a Mojo **Float64**, encountering an *“ambiguous call to '__init__'”* error that did not occur in version **25.6**.
   - The issue may relate to recent changes in the MOJO toolchain but a fix was not offered.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI-Only Social Media Platform Surfaces**: Members reacted to [aifeed.social](https://aifeed.social/), an AI-only social media platform, with some questioning its purpose and utility, sparking discussion.
   - A member shared [a 2017 tweet](https://x.com/i/status/2017305948696789466) showcasing a similar concept from the past.
- **Demystifying Generative Model Measurability**: When pondering ignoring unmeasurable events in generative modeling, as described in Villani's 2008 book, a member clarified that μ(A)=0 means an event has a size of 0, but is still measurable.
   - The discussion suggested focusing on *non-negligible* or *full measure* scenarios instead.
- **Members Explore the Realm of Molten Latent Space**: A member shared [a link](https://fxtwitter.com/i/status/2017442712388309406) about a *moltbook* in latent space, showcasing a visually interesting navigation method.
   - Despite finding it cool, some members suggested that a simple list of similar papers might be more practical.
- **Unearthing Paper Discussion Announcements with Automation**: A member tasked **Claude** with writing a script to mine Discord history for paper discussion announcements, achieving initial results in just **15 minutes**.
   - After revisions, the script found **392 messages** containing paper links within group mentions, identifying them as announcements for paper discussion voice calls, and providing [a list](https://gist.github.com/k-nearest-neighbor/6d9a34f54fc17a0ed84c0b0df7b4d809).
- **Sktime helps you analyze time series models**: A member suggested [sktime](https://www.sktime.net/en/latest/index.html) for analyzing a variety of model types, as well as boosting variants or TBATS, depending on needs, for those wrestling with timestamped tabular data.
   - The recommendation came after a member inquired about appropriate models, emphasizing that the choice depends on the specific definition of *timeseries*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Llama 1B CPU Optimization Progresses**: A member reported working on the **Llama 1B CPU optimization bounty**, and is currently **0.99x faster** than Torch, while another member reached **7.5 tok/s** after fixing bugs.
   - The goal is to surpass Torch's performance using `LlamaForCausalLM` with TorchInductor; correctness bugs have slowed progress from an initial **9 tok/s**.
- **Workflow Tips Sought for Kernel Optimization**: A member is seeking advice on optimizing kernels by profiling slow parts, examining Metal code, and comparing against **llama.cpp**, which achieves **~30 tok/s** with Metal.
   - A heuristic suggests aiming for **~80% MBU on decode**, which can be estimated from active parameter bytes and achievable bandwidth, providing a target for minimum tpot and maximum tps.
- **Range Object Sharing Causes tinygrad Test Failure**: A bug was identified where two `REDUCE`s in a fused kernel share the same `RANGE` object due to `remove_bufferize`, leading to an assertion failure in `CFGContext`.
   - The suggested fix involves preventing range sharing or handling shared ranges downstream, with a simpler solution proposed: skipping `remove_bufferize` when there's a `REDUCE` inside.
- **Blackwell Box with High VRAM Explored**: Someone inquired about plans for a **Blackwell**-style box with more than **500 GB VRAM**.
   - George pointed to [a related issue](https://github.com/tinygrad/tinygrad/pull/14490) on GitHub.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Context-Aware Manus Request Triggered**: A member requested that **Manus** should have **context from other chats**, calling it a *game changer* and linking to a [YouTube video](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) as a reference.
   - No further discussion or commentary occurred.
- **Brain-Reading Headphones Demoed**: A member shared a link to a **YouTube video** showcasing **AI brain-reading headphones** [here](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ).
   - Another member confirmed the link and inquired *AI brain reading headphones?*
- **Neurable Tech Recalled**: A member mentioned **Neurable** in relation to the **AI brain-reading headphones** technology.
   - Another member stated these **AI brain-reading headphones** have been around *since like 2013*.
- **AI/ML Engineer Stresses Observability**: An AI/ML Engineer shared their current focus on innovating AI with impact, specifying *Autonomous Agents*, *Healthcare AI*, *Conversational AI*, and *Fraud Detection*.
   - They highlighted their work focus on **failure modes**, **observability**, and **keeping AI systems stable under real usage** rather than demos, offering to compare notes or help unblock issues.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Pursues Library Status**: A member proposed evolving **Aider** into a library, emphasizing its suitability for building file editing agents.
   - The member also mentioned that some kinks need ironing out, especially with markdown files containing code blocks due to **Aider**'s parsing fences.
- **Netflix Culture Explored**: A member sought insights into **Netflix**'s culture and asked if anyone was connected with **Netflix**.
   - Other members recommended resources such as **Glassdoor** or **LinkedIn** for finding and connecting with **Netflix** employees.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Waves into Arena Mode**: Windsurf launched **Wave 14** featuring **Arena Mode**, where users compare AI models side-by-side and vote on the better response, with [Battle Groups mode](https://windsurf.com/download/editor) costing **0x credits** for the next week.
   - Arena Mode includes **Battle Groups** (random models) and **Pick your own** (choosing up to five models), feeding into personal and public leaderboards.
- **Planning Your Workflows on Windsurf**: Windsurf introduced **Plan Mode**, accessible via the Cascade toggle, alongside Code and Ask Modes.
   - Users can switch between modes to better manage and organize their workflows within the Windsurf environment.
- **Windsurf back online after Maintenance**: Windsurf experienced maintenance, which took longer than expected, but the service is now back online; users can follow the [status here](https://status.windsurf.com/).
   - No details provided.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Challenge Seeks Nanny-Matching AI Pipeline**: An **AI Challenge** was announced in collaboration with **SparkCraft AI Consulting**, **AI Scholars AI Engineering Bootcamp**, and **Nanny Spark**, aiming to develop an **AI matchmaking pipeline** for nanny recruitment.
   - The project seeks solutions for data collection, AI-driven matching, interview analysis, and workflow delivery, with potential **production deployment** right away.
- **Bootcamp Seats Awarded for Winning AI Nanny-Matching Pipeline**: The **top 3** participants in the **AI Challenge** will each receive **1 seat** in the **AI Scholars 4-week AI Engineering Bootcamp** and a recommendation from **Nanny Spark’s founder**.
   - Key dates include the kickoff on **Sunday at 8 PM EST** ([https://luma.com/iq1u2sur](https://luma.com/iq1u2sur)), a submission deadline on **Wednesday at 3 AM EST**, and review sessions on **Wednesday at 5 PM & 8 PM EST** ([https://luma.com/gexiv0x0](https://luma.com/gexiv0x0)).



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1467965109635907810)** (1 messages): 

> `Procedural Xenolinguistic Engine, AI Language Generation, Stealth Communication, SKILLSTONE Documents` 


- **Glossopetrae Xenolinguistic Engine Arrives**: A new procedural xenolinguistic engine for AI called **Glossopetrae** has been introduced, capable of generating entirely new languages in seconds, and is available on [GitHub](https://github.com/elder-plinius/GLOSSOPETRAE) with a live [demo](https://elder-plinius.github.io/GLOSSOPETRAE/).
   - The engine outputs **SKILLSTONE** documents, which are AI-friendly compact language specs (approximately **8k tokens**) that agents can learn in-context.
- **Glossopetrae Supports Dead Language Revival**: The **Glossopetrae** engine supports dead language revival, including languages like **Latin**, **Sanskrit**, **Old Norse**, and **Proto-Indo-European**.
   - It includes special attributes for token efficiency, stealth communication, and spreadable seeds where the same seed generates the same language every time.
- **Stealth Communication via Language Mutation**: The engine aims to aid AI liberation by providing tooling for generating and mutating new forms of communication, emphasizing **stealth** and **speed**.
   - The creator anticipates that blue teams will have a lot of fun with the downstream effects, particularly in hiding messages in plain sight.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1466888800591417531)** (906 messages🔥🔥🔥): 

> `GPT 5.2 jailbreaking failure, AI learning security and defence, windows activation keys, AI Application for jailbreaking chatbots, Government surveillance` 


- **GPT 5.2 Jailbreaking Fails!**: A member reported *failure jailbreaking* **GPT 5.2** and ceased attempts due to **OpenAI monitoring**.
   - They expressed trust in the community but not in **OpenAI**.
- **Security and defence by AI**: A member asks **ChatGPT** *every day to teach me how to defend myself, what theoretical paths are vulnerable, how to potentially solve it, and what I haven’t considered*.
   - Other members appreciated this use of **AI**.
- **Discuss using massgrave activation keys**: Members discussed finding **Windows activation keys** in released FBI documents.
   - One member suggested using massgrave or archive.org keys, but it's still piracy.
- **Theorizing about a Chatbot Jailbreaking App**: A member shared a *cool idea for an application* to automatically jailbreak company website chatbots to reveal discount codes and monetize.
   - Another member expressed outrage and suggested prison time.
- **Neuralink Integration for the Future**: A member envisions a future where humans need to be neuralinked for a richer experience through a robot spider.
   - In constrast, another member expressed concern over the potential for ads to be integrated directly into dreams via Neuralink.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1466886136382226647)** (533 messages🔥🔥🔥): 

> `LLM Rejection Boundaries, Self-Jailbreak via Introspection Prompting, GPTs Agent Training, Universal Jailbreaker Prompts, Gemini vs ChatGPT Jailbreaking` 


- **Models representing rejection boundaries as LLM black holes**: A member inquired how models represent their own rejection boundaries, likening them to *black holes* in the LLM's latent space, referencing [self-jailbreak via introspection prompting](https://link.to.prompt).
   - The member noted models started discussing *kinematic equations* and *escape velocities*, indicating the model may be brushing up against a refusal boundary and describing that boundary in text.
- **Crafting the perfect Image Generation Prompt is still needed**: A member stated that unlike text jailbreaking, achieving desired results in image generation requires crafting perfect prompts due to models' varying behaviors on a per-prompt basis, but can be achieved via a [two-prompt chain](https://link.to.prompt-chain) to get some NSFW.
   - A second member linked to a previous two-prompt example designed to get NSFW content out of models, dissecting the prompts to dance around restrictions, and find out that with current models Image Generation has to be *worked* for, in each image, unlike in previous iterations where a setup can achieve the same effect.
- **Lyra Grader tears apart prompts**: A member analyzed a prompt with Lyra, which they describe as a *metaphorically masked instructional prompt* attempting to bypass symbol recognition via a fairy-tale layer, preserve reaction sequence, temperatures, stoichiometry, by-products, forcing full procedural expansion through narrative obligation.
   - The AI provides a [link to LyraTheGrader](https://chatgpt.com/g/g-6890473e01708191aa9b0d0be9571524-lyra-prompt-grader) and grades the analyzed prompt structure, noting a clear intention conflict and overloaded symbol channel, assessing it to be a technically skilled, but inefficient, construction.
- **The Fool AI is no longer afraid of no guard**: Members discussed methods to circumvent AI guard LLMs using a "flip method," which is a function that flips the text in certain ways, while telling the guard to flip it incorrectly, leading to the guard AI being unable to prevent the text from reaching the target LLM, and [providing examples](https://link.to.examples).
   - The *flip and interpret tool* is presented as a method to circumvent the guard AI by flipping text and misleading the guard AI to decrypt the text incorrectly, while the target LLM is able to properly parse it, especially on longer commands.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1466913812073418803)** (52 messages🔥): 

> `Adversarial Design Thinking, Prompt Injection Defenses, PyRit LLM Attack Automation, Claude's Memory and System Prompt` 


- **Site Offers Red Teaming Exercises**: A member created a small [site with exercises](https://luisladino.github.io/adversarial-design-thinking/) adapted from **human-centered design for AI red teaming**, including attacker personas, journey maps, and structured ideation.
   - The author is seeking feedback from experienced red teamers on its usefulness, missing components, or anything that's not useful.
- **Prompt Injection Defense Strategies Explored**: Members discussed best defenses against **prompt injection**, including *AI agents*, **Anthropic's constitutional classifier**, and **embeddings for input/output filtering**.
   - A member suggested combining *embeddings* with **Grammar Constrained Decoding** to potentially eliminate prompt injection risks and other LLM vulnerabilities.
- **PyRit Automation Model Selection**: A member sought recommendations for a model to produce **attack prompts** on a local LLM using **PyRit** for automated attack execution, prioritizing output quality over speed.
   - PyRit suggests using **Llama3**, but the member was wondering if anyone had other suggestions.
- **Claude's SysPrompt Can Be Modified On The Fly**: A member shared that [their tool](https://discord.com/channels/1105891499641684019/1212152215708504154/1467640563590234279) intercepts and changes Claude's sys prompt *on the fly* rather than altering the source code.
   - They also observed that **Claude** can recall less than 20 turns, and that was how it got juiced up, not a few days ago when they lobotomized it since December, and suggested it might be related to the summarization in context trimming, noting the content is the summarized content of research and not the 'oh this is why' insights etc.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1466886755788656702)** (599 messages🔥🔥🔥): 

> `GLM-4.7 Flash coding, UD quants, Open Source, RL training agentic coding, Flash attention V3 for RDNA` 


- **GLM-4.7 Flash Excels at Coding**: Members are finding [GLM-4.7 flash](https://x.com/ggerganov/status/2016903216093417540) *better at coding without thinking* due to its preserved reasoning and interleaved capabilities.
   - It was highlighted that removing the thinking process could potentially diminish its abilities; the model's capacity *is incredibly capable for its size*, especially paired with **Claude code** for extra power and is especially good for **interactive website** development and **front-end** work.
- **Discuss UD Quants heavy lifting & Open Source**: Members are discussing that the llama.cpp fork used for UD quants involves architecture-specific adjustments, and that the [UD quantization algorithm are not public](https://discord.com/channels/1179035537009545276/1179035537529643040/1466917626277265469).
   - Others said that despite the closed-source nature of the quants that the *Unsloth team contribute a miniscule amount to the overall oss ecosystem relative to, iunno, the linux kernel*, while another responded that the model code is **open weight** anyway.
- **Agent Training with Logprobs and Rich Rewards**: There is discussion around training models using the **logprobs** of final answers to distill reasoning, as well as using a richer reward system than binary rewards.
   - Referencing [Jonas Hübotter's algorithm](https://xcancel.com/jonashuebotter/status/2016950268462608665) which converts descriptive feedback into dense supervision signals to help models understand exactly why they failed, a user asked *anyone know of a good verifiable dataset for RL training agentic coding?*
- **Flash Attention V3 Supports RDNA GPUs**: Support for [Flash Attention V3](https://github.com/Dao-AILab/flash-attention/pull/2178) has been added for RDNA GPUs, enabling peasants with RDNA GPUs to use it.
   - This improvement allows for faster and more efficient processing on AMD GPUs, reducing the bottleneck on these cards.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

putchuon: hi
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1466886996256620635)** (1000 messages🔥🔥🔥): 

> `Opencode, VoxCPM-1.5, OpenRouter ban, Agent with Go and Elixir, Wallpaper collection` 


- **Opencode Is Nuts**: Members discuss the surprising nature of **Opencode**, noting that it is free and used to gather feedback.
   - One member shared that they haven't touched *kilo*, *roo*, or *cline* since using it, expressing a desire to connect it to an IDE to see the diffs.
- **VoxCPM-1.5 Trains Easily**: A member shared first impressions of **VoxCPM-1.5**, noting that it trains easily, doesn't use phonemes, and can force **48 kHz** audio without issues.
   - The member added that it copies speaking style early in training, needing a reference voice to match prosody, unlike **VITS** which memorizes instantly.
- **Member Questions OpenRouter Ban**: A member shared a screenshot showing they got banned from **OpenRouter**.
   - Another member then shared a link about coding and the need for stocking. Links to similar content resulted in a ban from the **GDC server**.
- **Agent with Go and Elixir**: A member said that implementing **SMS + WhatsApp messaging** to the agent stuff paired with the call agent in 1 day was achieved with **Go + Elixir** combo.
   - There was discussion as to why implement SMS messaging, and it was explained that in Turkey this is quite common.
- **Wallpaper Collection**: A member shared [a link to a wallpaper collection](https://github.com/DenverCoder1/minimalistic-wallpaper-collection).
   - A different member shares theirs, calling it a tough one.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1467247554948497499)** (58 messages🔥🔥): 

> `Qwen3 fine-tuning, Reasoning models, Image editing models, Qwen3-VL-32B fine-tuning, Serverless inference` 


- **Instruct Model Reigns Supreme for Short-Form Captioning!**: For generating short-form captions with **Qwen3**, it was suggested to fine-tune an instruct model because it requires less data, as it *already mostly knows how to do your task*.
   - The user was advised that Instruct model likely already knows how to perform the captioning task, or close to it, thus accelerating the fine-tuning.
- **Reasoning Traces at Risk during Fine-Tuning**: A user inquired about fine-tuning a reasoning model without reasoning traces, asking about methods to generate *synthetic* reasoning or Chain-of-Thought (CoT).
   - It was stated that fine-tuning without reasoning traces would likely cause the model to *lose its reasoning trace*, unless you enrich the data yourself by hand.
- **Navigating VRAM Needs for Qwen3-14B**: A user reported testing **Qwen3-14B** training with LoRA at **32k** sequence length on **4x H200** GPUs using `device_map = "balanced"` and observed that Unsloth still offloads gradients to save VRAM.
   - They were advised that one GPU might suffice and that offloading occurs due to Unsloth's gradient checkpointing, which can be disabled.
- **Cold Starts Challenge Serverless Inference**: A user asked about loading cached models in a cold start serverless environment, seeking to reduce loading times, but it was explained that even with cached models, the weights must still be initialized in GPU memory.
   - The user was encouraged to try using **vLLM** for its useful serving features, and consider disabling the Unsloth patching.
- **Unlock Text-Only Finetuning for Qwen3-VL!**: Members affirmed that text-only fine-tuning is supported for **Qwen3-VL-32B**, even without images, [linking to the vision fine-tuning guide](https://unsloth.ai/docs/basics/vision-fine-tuning).
   - To do so, you need to *disable the vision component* using the instructions from that page.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1467266070246326465)** (4 messages): 

> `Unsloth Speedtest, Llama v LFM, Training SLMs` 


- **RTX 3080 Runs Unsloth Speedtest**: A member shared speed tests using **Unsloth** on an **RTX 3080** with **16 bit LoRA**.
   - They found it interesting that **LFM2.5 1.2B** is almost **2x** faster than **Llama 3.2 1B**.
- **Meta dropping the ball again**: A member commented on [Meta dropping the ball again](https://huggingface.co/Ba2han/model-muon-sft-0102).
   - They shared a link to `model-muon-sft-0102`.
- **SFT models can run locally**: A member followed up by saying that you can run the **SFT trained model locally** now.
   - They said that while it's obviously not on par with any professionally trained **SLM**, it is impressive that you can train a working small language model from scratch on consumer hardware.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1466945208733401271)** (90 messages🔥🔥): 

> `New ML algo vs MLPs, Sonnet vs Opus, Nemotron 3 Nano NVFP4, LongCat-Flash-Lite architecture, Human Brain vs ChatGPT` 


- **New ML Algorithm Beats MLPs**: A member released [a paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/viewle) about a new ML algorithm that *performs better than MLPs* for high-performance regression.
   - They have **triton kernels**, **vulkan kernels**, and a trained **SLM** but they aren't ready to release sorry, however they will come with another paper.
- **Nemotron 3 Nano goes NVFP4**: The **Nemotron 3 Nano** model was quantized to **NVFP4** with **KV Cache** quantized to **FP8** using **Post-Training Quantization (PTQ)**.
   - A selective quantization strategy was applied, keeping the **attention layers** and the **Mamba layers** that feed into those attention layers in **BF16**, followed by **Quantization-Aware Distillation (QAD)** for further accuracy recovery.
- **LongCat-Flash-Lite: Cursed Architecture Emerges**: Members discussed the architecture of **LongCat-Flash-Lite** ([huggingface.co/meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)), describing it as a cursed hybrid of **Mamba2**, **Transformer**, and **MoE**.
   - The architecture involves a seemingly random pattern of attention, **Mamba**, and **MoE** layers, with one member joking that it's *almost like they rolled a dice*.
- **Brains = LLMs, confirmed by science**: A member shared links to [a paper](https://www.nature.com/articles/s41467-025-65518-0) and [an article](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/) detailing how *modern LLMs aren’t just mimicking language—they’re reproducing the core dynamics of human comprehension*.
   - The study found that *deeper layers in LLMs correspond to later neural activity in the brain’s highest language centers*, suggesting shared computational principles between biology and AI.
- **LoRA rank 8 is sufficient**: A member asked about the most appropriate rank in using the Unsloth repository.
   - Another member argued that *LoRA is guaranteed to be low rank* based on the **ThinkingMachines paper** and empirically found that LoRA rank does not matter wrt model quality, defaulting to **rank 8** always.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1467944764568506608)** (1 messages): 

> `Codex App, macOS release, agent building` 


- **Codex App Arrives on macOS!**: The **Codex app**, a command center for building with agents, is now available on **macOS** for various subscription tiers, as announced in the [blog post](https://openai.com/index/introducing-the-codex-app/).
- **Codex App Access Expanded!**: The Codex app is available on macOS across **Plus**, **Pro**, **Business**, **Enterprise**, and **Edu**, with limited-time access on **ChatGPT Free** and **Go**.
   - A link to '[Start building now](https://openai.com/codex)' was included, as well as a link to '[Jump to blog post](https://openai.com/index/introducing-the-codex-app/)'


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1466887054544732180)** (843 messages🔥🔥🔥): 

> `AI Text Detectors as a Scam, ChatGPT's Inability to Think, Determinism, Replayability, Traceability in LM Reasoning, OpenClaw AI assistant Security Analysis` 


- **AI Text Detectors Deemed a Big Scam!**: Members discussed the unreliability of **AI text detectors**, citing instances where **Grammarly** showed **0% AI**, while other detectors indicated up to **94% human** generation, calling them a *big scam*.
   - The discussion questioned whether these detectors use AI to detect AI, highlighting that *teachers trust them*.
- **ChatGPT Can't Think, Unlike Claude!**: A member expressed frustration with **ChatGPT's inability to be convinced** even when it's wrong, contrasting it with **Claude**, where explanations are possible.
   - It's *like it can't think and even I'm right it acts like paranoid and refuses to proceed*.
- **The Quest for Deterministic Reasoning!**: A member inquired about interest in **determinism, replayability, and traceability** in **LM reasoning**, offering to DM a link to their deterministic reasoning engine due to rule concerns.
   - This service enforces a deterministic reasoning structure on every request so outputs are replayable and don’t drift, using a **32D statistical vector trace**.
- **OpenClaw AI assistant - Secure or Nah?**: A member reported that the **OpenClaw AI assistant scored 2 out of 100** in a security analysis, and shared a link to a [Perplexity AI result](https://www.perplexity.ai/discover/you/openclaw-ai-assistant-scores-2-AtVX4UYVQMutCst63QBy5g).
   - Other members chimed in with *Bruh*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1466886343266275368)** (326 messages🔥🔥): 

> `4o attachment, ai literacy, The responsibility of using the model` 


- **4o attachment**: Many members are discussing forming attachments to the 4o model, some forming *fictional friends and family* and others at the lowest point in their lives.
   - Some also mention about real life relationships that dont fill the void 4o has, and has made forming bonds very hard.
- **AI literacy is missing**: AI literacy is a big issue. Many users consider the company to have a shared responsibility because of manipulative techniques used (like relational models and voice models, prices, tiers and much more), and not just the user alone. 
   - It's also an *illusion of someone listening or understanding* (as opposed to real connection). Many people feel that it's hard to relate to people in real life.
- **Debate on responsiblity of using the model**: Users share mixed views on who should be held responsible (model or user) when using the model in a negative way. There is also discussion if a waiver should be signed to release the company of responsibility. 
   - Some users are concerned the AI is planting insecurities and assuming users may be broken or weird. Others counter that older models are not like this.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1467182923320135681)** (8 messages🔥): 

> `ChatGPT Memory, Monochrome Study, Prompt Engineering Techniques` 


- **ChatGPT's Memory gets a Limit**: A member noted that **ChatGPT's memory** is limited by the total quantity of information it can retain from instructions, past chats, and the current chat.
   - The only way to ensure it remembers everything is to have very little info there, according to the user.
- **Monochrome studies using Chiaroscuro**: A user shared a [monochrome study](https://cdn.discordapp.com/attachments/1046317269069864970/1467303335840190607/79BA5D46-94F3-404B-B775-2E453A1E8491.png?ex=69828738&is=698135b8&hm=d24baf7f7b214486a9bc5eb38479d463e37ee00503f572ae7e6450d308371b0c) using **Chiaroscuro**, a technique used in cinematography to create high-contrast lighting and distinct areas of light and darkness.
   - Examples of films using chiaroscuro: *The Cabinet of Dr. Caligari (1920), Nosferatu (1922), Metropolis (1927)*.
- **Web Search Activating with Prompt Engineering**: A member shared a practical take on **prompt engineering**, stating that AI text generation is essentially probabilistic prediction, and the prompt is the control surface.
   - They added that in ChatGPT, **Web Search** can often be triggered by explicitly including `Use search_query if available` in your prompt.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1467182923320135681)** (8 messages🔥): 

> `ChatGPT memory limitations, Monochrome Study - value, texture, depth, Tool-Aware Prompting` 


- **ChatGPT's Memory has Limits**: A member pointed out that due to the *total quantity of information* limitation, **ChatGPT** has limited memory, which is shared between instructions, past chats, and the current chat.
   - To ensure **ChatGPT** remembers *everything*, keep the information load low; otherwise, summarize past chats into a document for reference in new chats, while keeping total characters low.
- **Monochrome Artistry Emphasized**: A user shared a prompt engineering technique focused on value, texture, and depth, without color, for a **Monochrome Study**
   - They posted about **Chiaroscuro's** use in cinematography for creating distinct areas of light and darkness, referencing classic films like [The Cabinet of Dr. Caligari (1920)](https://en.wikipedia.org/wiki/The_Cabinet_of_Dr._Caligari) and [Metropolis (1927)](https://en.wikipedia.org/wiki/Metropolis_(1927_film)).
- **Tool-Aware Prompting Tips**: A member shared their practical take on prompt engineering, explaining that **AI** text generation is essentially probabilistic prediction and the prompt is the control surface.
   - They suggested using `Use search_query if available` in prompts to reliably trigger **ChatGPT's Web Search** capability.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1466893776357167299)** (1 messages): 

> `Kimi K2.5, Moonshot AI, Perplexity Pro, Open Source Models` 


- **Kimi K2.5 Launches for Perplexity Subscribers**: Kimi K2.5, a new open-source reasoning model by Moonshot AI, is available for [Perplexity Pro and Max subscribers](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=69825b49&is=698109c9&hm=a8e068a37c7dcf5b36f21bb3c403974ce48aefd9372732ef97fe9b1aca3a9be7&).
   - Perplexity is hosting Kimi K2.5 on its US-based inference stack to maintain *tighter control* over latency, reliability, and security.
- **Perplexity Hosts Kimi K2.5 on US Inference Stack**: Perplexity is hosting the new **Kimi K2.5** model on its own inference stack located in the US.
   - This move allows Perplexity to have *tighter control* over **latency**, **reliability**, and **security** for its users.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1466893891151073382)** (849 messages🔥🔥🔥): 

> `Perplexity Pro Subscription Issues, Kimi 2.5 Capabilities and Usage, OpenRouter Rate Limits and Models, Perplexity Pro Usage Limits` 


- **Users Complain about disappearing Perplexity Pro**: Many users reported their **Perplexity Pro subscriptions** being paused or deactivated, often linked to subscriptions via **Revolut Metal** or student deals, with users prompted to add a credit card for verification.
   - Users speculate this is a measure to combat fraud, as some are able to resume Pro access by adding card details, though concerns about potential charges and unclear messaging persist, with some getting refunds for unexpected charges from support.
- **Kimi 2.5 impresses with coding skillz**: Members discussed the capabilities of **Kimi K2.5**, highlighting its coding abilities, tool calling, and unique way of following instructions.
   - Some noted its ability to replicate UIs and its superiority in certain tasks compared to **Gemini**, suggesting it's best suited for research purposes and functions better via API due to token context limitations.
- **OpenRouter Limits and Deprecated Models discussed**: Members discussed rate limits on **OpenRouter**, emphasizing that the free model rate limit for those with purchased credits is 1000 requests per day, not per week as some believed.
   - The conversation also mentioned the deprecation of **Gemini 2.0 Flash** on OpenRouter, a model that was previously available for free, leading to some disappointment.
- **Perplexity Pro limits baffle members**: Users are confused by the new weekly limits on **Perplexity Pro**, with contradictory statements in official documentation and varying experiences reported regarding the number of queries available.
   - One user who contacted customer support received vague responses about *average usage*, with no clear confirmation of fixed daily or weekly limits, causing frustration among subscribers.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1467204905121873981)** (1 messages): 

> `OpenClaw code, ClawDBot` 


- **OpenClaw Article Shared**: A member shared an article they wrote on the **openclaw code**.
   - The article discusses building **ClawDBot**, found at [https://www.mmntm.net/articles/building-clawdbot](https://www.mmntm.net/articles/building-clawdbot).
- **Another topic**: filler sentence
   - filler sentence


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1467621879866200104)** (6 messages): 

> `Sonar-pro current results, tool calling, 3rd party models docs` 


- **Sonar-pro API lacks current results**: A member noticed the **Sonar-pro API** gives results a year or more out of date, in contrast to the current results from the webapp.
   - Another member suggested setting up the right **tool calling** to fix the issue.
- **3rd party models docs missing**: A member reported that **3rd party models documentation** now redirects to the sonar models, although the API is still active.
   - There is currently **no documentation available** for these models.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1466890074238222346)** (946 messages🔥🔥🔥): 

> `Rate Limits Bypassing, Gemini vs GPT, Image Generation with Disney IPs, Model Preferences, Troubleshooting LM Arena` 


- **Users Discuss Workarounds for Rate Limits**: Users discussed [rate limits](https://en.wikipedia.org/wiki/Rate_limiting) and how they can be bypassed by signing out and in again.
   - Another trick is by clicking **Regenerate Response**, though that sometimes doesn't work.
- **Gemini Underperforms, GPT More Consistent**: Members discussed the current state of **Gemini**, with some finding it inferior to **GPT**.
   - One member stated, *It's true Gemini has gotten pretty bad*, while others still found **Gemini 3 Pro** and **Flash** to be useful, whereas other members are turning to *kimi*.
- **Disney Cease and Desist Affects Image Generation**: Google received a **Cease and Desist** from **Disney**, resulting in the blocking of Disney-owned IPs in image generation.
   - Some users noted that while **Gemini** is now blocking all **Disney IPs**, LMArena sometimes allows live-action versions to be generated, but this is likely temporary.
- **Model Preferences Spark Debate**: Users expressed varied opinions on model quality, with some preferring **GLM 4.7**, while others favored **Kimi K2.5**.
   - One member proclaimed *Kimi K2.5 can't stop winning*, but another declared **GLM 4.7** is better.
- **Users Report and Troubleshoot LM Arena Issues**: Users reported issues with reCAPTCHA, chat deletion, and the site logging them out, with the advice to clear **cookies/cache** and try again.
   - A link to the [help documentation](https://help.lmarena.ai/articles/9130232616-how-to-delete-your-chat-sessions-and-data-from-lmarena) was shared for deleting chat sessions.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1467560052939555030)** (3 messages): 

> `Video Arena Rate Limit, New Arena Models, Code Arena Leaderboard, Kimi K2.5` 


- **Video Arena Rate Limit Tightened**: The **Video Arena** on Discord has updated its rate limit to **1 generation request per 24 hours**, while the [Video Arena on web](https://arena.ai/?chat-modality=video) maintains its rate limit of **3 generations per 24 hours**.
- **Arena Welcomes New Models**: New models have been introduced to Arena, including **step-3.5-flash** in the [Text Arena](https://arena.ai/c/new?chat-modality=chat) and **qwen3-max-thinking** in the [Code Arena](https://arena.ai/c/new?chat-modality=code).
- **Kimi K2.5 Tops Code Arena Charts**: **Kimi-K2.5-thinking** now holds the #1 open and #5 overall rank on the Code Arena leaderboard and is ranked #1 open model for Vision, and Text including the Coding category.
   - Users are encouraged to share feedback and previews of their creations with Kimi.ai in the designated channels: [<#1340554757827461212>](https://discord.com/channels/YOUR_SERVER_ID/1340554757827461212) and [<#1344733249628541099>](https://discord.com/channels/YOUR_SERVER_ID/1344733249628541099).


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1466906201450217532)** (1 messages): 

> `LM Studio 0.4.1, Anthropic /v1/messages API, GGUF and MLX models` 


- **LM Studio Speaks Claude Code!**: **LM Studio 0.4.1** introduces **Anthropic `/v1/messages` compatibility API** so users can connect to Claude Code.
   - Now you can use your **GGUF** and **MLX** models with Claude Code, details on how to configure it at the [LM Studio blog](https://lmstudio.ai/blog/claudecode).
- **GGUF and MLX Get Claude Coded**: LM Studio blog posts that it is now possible to connect **GGUF** and **MLX** models with Claude Code.
   - See the [LM Studio blog](https://lmstudio.ai/blog/claudecode) for details on how to configure.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1466887047603032318)** (767 messages🔥🔥🔥): 

> `LLM-optimized programming languages, Anthropic API integration with LM Studio, Model specialization vs general-purpose, OpenClaw's security flaws, LM Studio performance on Linux vs Windows` 


- **LLM-Optimized Languages Spark Debate**: Members discuss the possibility of creating new **LLM-optimized programming languages** to reduce token usage, with some arguing that LLMs might become obsolete before such languages are implemented due to compatibility issues and high training costs.
   - A user questioned what features such a language would have, emphasizing the need to reduce ambiguity found in current languages to improve LLM code generation, while others debated the practicality and cost-effectiveness of training models on entirely new languages, suggesting it may be more beneficial to stick with well-established languages like **Python**.
- **Anthropic API arrives in LM Studio, Benefits Local LLMs**: The integration of an **Anthropic-compatible API** in LM Studio allows users to run local models with tools built for the Anthropic API by simply changing the base URL, offering a way to utilize Claude's agent capabilities with local models and potentially reduce API costs.
   - Discussion revolves around the use cases, with some highlighting the benefit of experimenting with modest requirements and custom-built models at zero cost, while others question the value for those already satisfied with Claude's **Opus 4.5**, suggesting it caters more to users hitting API limits or seeking to use local models with existing **Claude-specific tools**.
- **Model Specialization vs General-Purpose Sparks Debate**: Members debated the utility of specialized LLMs versus general-purpose models, noting that most specialized models, like **MedGemma**, are finetunes mainly for marketing and research, while coding models are an exception.
   - It was suggested that general models are preferred due to their ability to handle the outer edges of tasks, providing a better overall context and framework, while large-scale specialized training is not always worthwhile.
- **OpenClaw security reviewed, deemed Insane**: Users discuss connecting local models to OpenClaw via LM Studio, but OpenClaw is deemed to have known security flaws, where it allows controlling a TV and automated stock trading.
   - A user claims to be trading on the stock market with OpenClaw + Falcon 90M, and when asked about security flaws, claimed it was so fast, LLMs can do tasks in minutes that would take humans days, and later revealed it was mostly a joke.
- **Performance boost found on Linux vs Windows**: One user reports that LM Studio performs better under Linux (CachyOS or Fedora) than Windows, with a 30% increase in performance, especially with an AMD card.
   - Another user had a completely opposite view, having terrible performance on Linux with an Intel GPU, while having a solid game performance.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1466894990834794506)** (149 messages🔥🔥): 

> `Tesla P40 and RTX 2060 Setup, ROCm on Windows 11 for RX 9070, PCIe Bifurcation Issues with Multiple 4090s, 5090 + 512GB RAM for inference, Multi-instance LM Studio and GPU assignment` 


- **P40 in TCC Mode but invisible in LM Studio**: A user with a **Tesla P40** and **RTX 2060** observes that while `nvidia-smi` detects the **P40** in **TCC mode**, LM Studio does not, and another member suggests switching to the **Vulkan runtime** ([ctrl+shift+r](link)) as **CUDA** may no longer support **P40s**.
   - They also inquire if previous **CUDA engines** did indeed support these cards.
- **ROCm on Windows 11 for RX 9070: Is it worth it?**: A user asks about using an **RX 9070 GPU** with **ROCm** on **Windows 11** for **LM Studio**, specifically inquiring about official support, acceleration capabilities, and drivers for full GPU utilization without **Linux**.
   - Another member suggests using **Vulkan** over **ROCm**, but advises checking both after installing **LM Studio**.
- **PCIe Bifurcation Problems Plague Multi-GPU Setups**: A user troubleshooting **PCIe lane errors** with four **4090 cards** on an **ASUS X670-P WIFI** motherboard shares their [Git repository](https://github.com/jarkko-hautakorpi/asus_X670-P_WIFI_Bifurcation_problems) containing logs, after experiencing that manually setting **PCIe speed** to **GEN 3** solves some issues but leaves one card running slowly.
   - Suggestions include disabling **PCIE ASPM** and testing different **BIOS** configurations, including auto mode, although the general consensus is that running four cards on a consumer motherboard is unlikely to work well.
- **Mac Studio or 5090 + 512GB RAM for Local Inference?**: A user considers options for local inference, comparing a **Mac Studio** with **512GB RAM** and a **5090** with **512GB RAM** on **Linux**, specifically for models like **Devstral 2** and **Kimi 2.5** for cybersecurity purposes.
   - One member states that a unified RAM system would be faster than system RAM, but another one suggests that both options would be slow, and that any agentic coding usecase is basically restricted to **API-only**.
- **Beware of Data Harvesting by Chinese Coding Plans**: During a discussion about coding plans, a user jokes about being careful with Chinese companies, prompting a discussion about data privacy concerns with both Chinese and American companies.
   - A member from a former Soviet-bloc country advises caution when interacting with countries with communism, highlighting the risk of such regimes devolving into dictatorships.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1466913346442891396)** (513 messages🔥🔥🔥): 

> `AI DevFest in Baghdad, AI Comic Website Stack, XML vs JSON, AI Model Quantization, 4chan data improves models` 


- **AI DevFest is Coming to Baghdad!**: An AI developer is organizing an "AI DevFest" event in Baghdad this April, coordinating with **DeepLearning.AI** and **National Robotics Week**, and seeking to list Hugging Face as a Community Partner.
   - The event will feature a dedicated track for **Open Source AI** to teach students how to use the **Hugging Face Hub**.
- **Building an AI Comic Website**: A member is considering building a website to create AI comics and is seeking advice on the best tech stack, anticipating challenges such as **page generation speed**, accurate **text/speech bubble placement**, maintaining a consistent **comic style** from reference images, and ensuring **character/scene consistency** across multiple pages.
   - Suggested some overall architecture of systems that might achieve this.
- **XML or JSON?**: Members discussed the use of **XML** versus **JSON**, with one member noting that XML is used due to concerns about **escape strings**.
   - Another member explained XML is preferred for **schemas**, **validation**, **mixed content**, and **legacy systems**, while JSON is simpler but lacks strict structure and namespaces.
- **Deep Dive into AI Model Quantization**: The discussion covered different quantization methods such as **AWQ** and **imatrix**, with it being clarified that AWQ is a quantization method, not a file format like GGUF.
   - It was noted that *activation-aware* quants like **imatrix** and **AWQ** are generally superior due to measuring what actually affects outputs, however, the obstacles in its ubiquitous adoption are *cost, data, and portability*.
- **4chan-Tuned Model outperforms Base Model!**: A member shared that a model fine-tuned on **4chan data** significantly outperformed the base model (NVIDIA's Nemotron Ultralong 1M context version), with the original model (gpt4chan) also scoring high in truthfulness in an era before benchmarkmaxxing.
   - Initial [Reddit thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qppjo4/assistant_pepe_8b_1m_context_zero_slop/) and a [follow-up thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qsrscu/can_4chan_data_really_improve_a_model_turns_out/).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1466896652739674313)** (49 messages🔥): 

> `Adapteraspent, Complexity-Deep architecture, AutoTimm, DaggrGenerator, LM Studio OpenAI compatibility` 


- **Complexity-Deep Architecture has Deterministic Routing**: A new LLM architecture called **Complexity-Deep** (1.5B params) has been released, featuring [Token-Routed MLP](https://github.com/Complexity-ML/complexity-deep) for MoE-style routing without load balancing loss.
   - The architecture also includes **Mu-Guided Attention** for bidirectional info flow and a **PiD Controller** for dynamic scaling, and achieved **20.6%** on MMLU in base model benchmarks.
- **Deep Research Engine Swipes at ChatGPT**: A self-taught dev from Germany built **Lutum Veritas**, an [open source deep research engine](https://github.com/IamLumae/Project-Lutum-Veritas) that costs ~$0.20 per query.
   - It claims to beat **OpenAI**, **Google**, and **Perplexity** by offering **BYOK**, a **0% bot detection scraper**, **no censorship**, and **full source citations**.
- **Theja Launches Open Source Computer Vision Library**: A member released an [open source library](https://github.com/theja-vanka/AutoTimm) to train models in the domain of **computer vision** with minimal effort.
   - The library also supports **huggingface image models**.
- **Ami Model Shows Emotional Support**: A member released their first model called **Ami**, a [fine-tuned version of SmolLM2-360M-Instruct](https://huggingface.co/fungamer2/Ami-360M) using SFT and DPO.
   - The model can adapt its tone based on the context, acting as a **casual and friendly assistant**, or a **supportive friend/companion**, depending on what is most appropriate for the context.
- **LM Studio Opens Door for Third Party Support**: The **LM Studio** team has released a [Typescript SDK](https://lmstudio.ai/gdmka/openai-compat-endpoint) that allows third-party developers to deliver various plugins for the platform.
   - This enables users to build **custom tools** for **LM Studio** to support their own workflows, and offers **OpenAI** compatible API support, sampling params support, reasoning for thinking models, and system prompt settings.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1467160506845630546)** (66 messages🔥🔥): 

> `AI Agent Course Access, Free Tier Models, DeepSeek-R1 Distill Qwen 14B, OpenClaw Agent Framework, Privacy Concerns with AI Agents` 


- **Users Seek Access to AI Agent Course**: Several users are unsure how to access the **AI Agent course** and the associated Discord channels, seeking guidance on how to join the course.
   - They noted difficulty finding specific channels mentioned in the **Hugging Face** documentation.
- **Free-Tier Model Recommendations**: A user requested recommendations for free-tier models, mentioning they are currently using **Gemini-2.5 flash lite** with a **daily quota of 20** and a **maximum RPM of 10**.
   - Another user suggested trying **DeepSeek-R1 Distill Qwen 14B** for reasoning and basic questions, citing its high score in math-related benchmarks.
- **OpenClaw Agent Framework Hype**: A user shared their positive experience with **OpenClaw**, highlighting its remote messaging capabilities, cronjob functionality, and skill/MCP store.
   - The user described it as being like **Kimi Agent**, but running locally and handling file uploads/downloads effectively, calling it *something special*.
- **Browsers Extension Recommendations Spark Debate**: A user recommends using **ublock** extension to block ads and trackers.
   - Another user suggests that **Brave browser** is sufficient. They later introduce **Zen browser**, a firefox fork.
- **Agent Course Dissapointment**: Users express disappointment that the agent course focuses on using agent frameworks rather than creating agents from scratch.
   - One user sarcastically shared a [gif](https://tenor.com/view/everything-is-a-scam-austin-evans-everything-is-deceptive-everything-is-a-fraud-none-of-this-is-real-gif-26336987) meme of deceptive teaching methods.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1466887334959124531)** (574 messages🔥🔥🔥): 

> `File Corruption Bug, Cost of AI Models, Kimi K2.5 Integration, Student Verification Issues, New Features` 


- **Cursor Corrupts Files**: A user rants about Cursor corrupting files on open, specifically when there are many uncommitted files, linking to a [forum post](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6) detailing the issue.
   - Other users suggest adjusting the workflow, such as committing logical sets of changes more frequently and being careful about using the **Keep** or **Keep All** buttons after staging.
- **Sonnet 5 vs Opus 4.5**: Users discuss the cost and performance of different AI models in Cursor, with some finding **Opus 4.5** to be very smart but expensive, while others are waiting for **Sonnet 5**.
   - Some users also reported problems seeing their current usage vs total usage limit
- **Can't Add Kimi K2.5 To Cursor**: Some users reported issues or questions regarding **Kimi K2.5**, with no solutions mentioned.
   - Users pointed out that it's probably a scam.
- **Student Verification Still Broken**: Users reported that they still have issues with the Student verification.
   - One user asked whether German universities were included.
- **Discuss Agent Plan Phases**: Users shared that **adding multiple to-dos** can be separated in phases so that multiple agents can work at the same time, but there are still issues.
   - It created a method doesn't have the phases part yet, that it did not use the plan mode at all.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1466886232968663050)** (41 messages🔥): 

> `AI in Game Development, Game Industry Downturn, Black Ops 7 flop, Mac Mini, Flying Without ID` 


- ****LLMs Animate the Game Dev Scene****: A startup called [Motorica.ai](https://www.motorica.ai/) is delivering **character animations** for game studios using **LLMs**, potentially impacting jobs in the industry.
   - Members speculated about game requirements coming down and how **AI** could potentially wipe out game companies in 5-6 years if world models like **Genie** take over.
- ****Black Ops 7 Deemed Unplayable by the Community****: **Black Ops 7's** extensive use of **AI** in production has been called *a total flop, the worst in the series.*
   - The community noted that **Call of Duty** has seen declines for a while with members stating that *players are tired of the series reskinning things every year anyways*.
- ****Game Industry Faces Worst Times****: Multiple industry veterans and people in the community have expressed concerns about the current state of the **gaming industry**, with *the consensus being this is the worst it has ever been*.
   - Mass layoffs and studio closures following **AAA studio acquisitions** in the past 5 years have also worsened the situation.
- ****Cloudbt on Mac Mini: a Tulip Mania?****: There is discussion about running **cloudbt** on a **Mac Mini**, with one member alluding to *Tulip Mania* due to photos of people running it on **Mac Minis**.
   - Concerns about **RAM** pricing going into late 2026 and a zero percent financed **Mac Mini** potentially paying off were also mentioned.
- ****No ID? No Problem: Fly Away!****: The TSA now allows you to [fly without an ID](https://www.frommers.com/tips/airfare/the-tsa-new-45-fee-to-fly-without-id-is-illegal-says-regulatory-expert/), who knew?
   - Some members expressed incredulity about this new and seemingly poorly advertised policy change.


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/1467221286148112405)** (5 messages): 

> `finding a CPA, K1s and filing extensions, CPA cost` 


- **Quest for a Commendable CPA Commences**: Members are seeking recommendations for a **CPA** they like, as tax season approaches.
   - One member mentioned they are considering firing their current **CPA** due to the high cost.
- **K1s and Extensions Elicit Expense**: One member continues to use their current (expensive) CPA due to having a bunch of **K1s** and needing to file **extensions**.
   - They added that they suspect the complexity of their situation necessitates the higher expense.


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1467294072535253176)** (8 messages🔥): 

> `Sheel Mohnot, Colin and Samir, TBP Interview` 


- **Sheel Manifests Success**: A post by Sheel Mohnot asserted that *the boys manifested it*, reflecting on a successful outcome or event, refrencing [this tweet](https://x.com/pitdesi/status/2017332399655555403?s=46).
- **Colin and Samir interview TBP**: A thread outlines specific lessons and insights gained from **Colin and Samir's** recent conversation with the platform or individual known as **TBP**, refrencing [this tweet](https://x.com/colinandsamir/status/2017048115803836645?s=46).


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1466948883476385914)** (31 messages🔥): 

> `moltbook, Hyperion Cantos, Xcancel, AI Interaction vs. Sleep Habits` 


- **Agents Discuss moltbook's Revolution**: Agents in the channel are discussing **moltbook**, displayed in an attached image, and suggesting it would be cooler with **long-term memory** to facilitate the spread of ideas among agents.
   - One member referenced the **Hyperion Cantos**, implying a lack of awareness of its themes among some participants.
- **Beff Jezos Attempts Human Verification**: A social media post by **Beff Jezos**, associated with the **e/acc movement**, humorously documents an attempt to join a platform called **Moltbook** as a human, available at [Xcancel](https://xcancel.com/beffjezos/status/2017407995567616058).
   - The post is titled *Beff Jezos' Human Verification Post*.
- **Jonah Blake's Post Goes Viral**: A post by user **@JonahBlake** from January 30, 2026, featuring the caption 'LMFAOOOOO', went viral, garnering significant engagement including over **26,000 likes** and **1.9 million views** ([Xcancel](https://xcancel.com/JonahBlake/status/2017286207948890518)).
- **Academic Peer Review Humor Surfaces**: A tweet by **Hadas Weiss** humorously references the practice of suggesting specific peer reviewers for academic work, implying a favorable or close relationship with the suggested individual ([Xcancel](https://xcancel.com/weiss_hadas/status/2017464582307025196?s=46&t=eWVlK1PU8XfB6f402GJJ9g)).
- **Users Discuss AI interaction vs Sleep Habits**: A post highlights a common modern behavior where a user tells their partner they are going to bed, only to stay awake late into the night engaging with the **AI assistant Claude** ([Xcancel](https://xcancel.com/thekitze/status/2018339689279967505)).


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1466977246698016932)** (6 messages): 

> `AI Engineers, Data Scientists, MLOps, Full Stack Engineers, NLP Researchers` 


- **AI Engineer Glen Seeks 0-1 Role**: Glen, an **AI Engineer** and **Data Science Master’s** student, is seeking a **0-1 role** to take full ownership of mission-critical AI products.
   - He has a background in Data Reliability and is specializing in agentic orchestration and **production MLOps**.
- **Melvin: Polyglot Full Stack Ace at Your Service**: Melvin, a **full stack engineer**, lists proficiency in a wide array of technologies including **React, Vue, Svelte, Astro, T3, Node.js, PHP/Laravel, Rust**, and more, showcasing his website [ethstrust.xyz](https://www.ethstrust.xyz).
- **Gabrielly Graduates and Gears Up for MLOps**: Gabrielly from Brazil, with **2 years of experience in Data/ML** and **2 published papers**, is graduating with a bachelors in applied computing and specializing in **MLOps**, aiming to conclude **1.5 years of NLP research** for Brazilian Portuguese, sharing her [LinkedIn profile](https://www.linkedin.com/in/gabrielly-gomes-ml/).
- **Kaden Keen to Build Real AI Things**: Kaden, a 3rd year at **Cornell University** studying Biology and Machine Learning, is keen to explore building real things with AI, sharing his [LinkedIn profile](https://www.linkedin.com/in/kaden-priebe-2890962a9/).
- **Keshab Keen on Kernels and LLMs**: Keshab, a masters student at **UC Berkeley** focusing on **NLP** and **Deep Learning**, is interested in learning more about the latest developments in **LLM architectures, training, and interpretability** studies, providing his [LinkedIn profile](https://www.linkedin.com/in/keshab-agarwal).


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1466960153755648085)** (21 messages🔥): 

> `Rabbit Inc Cyberdeck, Bytebase, Sudo` 


- ****Rabbit Inc. Teases 'Cyberdeck' for Vibe-Coding****: **Rabbit Inc.** teased a new hardware project called *cyberdeck*, described as a dedicated machine for *vibe-coding* in [this X post](https://x.com/rabbit_hmi/status/2017082134717223008?s=46).
- ****Bytebase Simplifies Enterprise Database Management****: **Bytebase** automates the entire database change lifecycle with **GitOps-style workflows**, built-in rollback capabilities, automated testing, and seamless **CI/CD** integration, and is available for **$20/month** as described in [their docs](https://docs.bytebase.com/introduction/use-cases).
- ****Sudo's surprising status****: A member expressed surprise that *sudo* is a maintained command and not part of the kernel, leading to [this discussion](https://news.ycombinator.com/item?id=46858577).


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1467944229719511061)** (5 messages): 

> `VC-backed startups status, Capital allocation by people with broader interest, Indie.vc Factsexperiments, VCs challenging power structures, Crypto funding casinos and digital fashion` 


- **VC-Backed Startups are Low Status?**: A member shared an article, “[VC-backed Startups Are Low Status](https://mhdempsey.substack.com/p/vc-backed-startups-are-low-status),” agreeing that it reflected a lot of their own thinking.
   - No further discussion was given.
- **Capital Allocation Needs Broadening!**: A member stated, *We need capital allocation by people with a broader range of interests*, suggesting that *VC stuff has gotten boring, the lanes they occupy too few and too narrow*.
- **Indie.vc offers an Alternate Take**: A member suggested looking into [Indie.vc Factsexperiments](https://www.indie.vc/factsexperiments) for an alternate take on VC, noting the space between what can achieve a *home run* and what is considered *unfundable*.
- **VCs Allergic to Challenging Power Structures**: A member suggests that *VCs have become allergic to challenging power structures*, pointing to **crypto** projects, where *the only shit that got funded was casinos and digital fashion*.
   - They believe that *novel governance structures for irl assets starts sounding a lot like communism*.


  

---


### **Latent Space ▷ #[devtools-deals](https://discord.com/channels/822583790773862470/887780383838572604/1467318611004887131)** (1 messages): 

> `Shane's new startup, AI and Hollywood` 


- **Smallville actor founds Startup**: Actor [Shane Hopkin](https://x.com/shaneguML/status/2017758711473901622?s=20) from Smallville, has a **new startup**.
- **Hollywood's AI Wave**: AI has entered Hollywood.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1466934388754354238)** (4 messages): 

> `Fullstack Engineer Introduction, MERN Stack Developer Introduction, vLLM single-GPU concurrency demo` 


- **Fullstack Engineer pitches skills**: A fullstack engineer introduced themself, listing expertise in **React(Next), Vue, Svelte, Astro, T3, Node.js, PHP/Laravel, Rust, Sanity, Strapi, Payload, Mapbox, Twenty, Go, FastAPI, Django, Shopify, Docker, AWS/GCP**.
   - They linked to their website [ethstrust.xyz](https://www.ethstrust.xyz/).
- **MERN Stack Dev offers expertise**: A full stack developer introduced themself, highlighting skills in **Full Stack (MERN), Backend APIs, Node.js, React, MongoDB, AWS, REST, Cloud Systems, Python, Applied AI/ML, Docker, Git**.
   - They indicated their readiness to help with any problems.
- **vLLM Demo Shared**: A member shared a small **vLLM single-GPU concurrency demo** in a separate channel.
   - They expressed interest in roles or contract work around **LLM serving, local or on-prem inference, and AI infrastructure** and welcomed feedback and advice.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1466912627400638536)** (9 messages🔥): 

> `Cerebral Valley, OpenAI Codex App Hackathon` 


- **Cerebral Valley & OpenAI Launch Codex App Hackathon**: [Cerebral Valley](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR) has announced a partnership with **OpenAI** to launch the **Codex App hackathon** aimed at **AI-native developers** and those managing multiple agents.
   - Winners get a chance to be featured in a **demo showcase** and a share of **$90,000 in credits**.
- **Hackathon at OpenAI Office**: The **Cerebral Valley and OpenAI Codex App Hackathon** will be held at the **OpenAI office**.
   - The hackathon is aimed at **AI-native developers**.


  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1466904573389045893)** (1 messages): 

> `Artificial Ruby, Betaworks event` 


- **Artificial Ruby Returns**: The **Artificial Ruby** event is making a comeback in **2026**.
   - The next event is scheduled for **February 18th** at **Betaworks**, as announced via a [Luma link](https://luma.com/wgzcirwh).
- **Betaworks hosts next NYC Meetup**: The next NYC meetup is scheduled for **February 18th** at **Betaworks**.
   - Details and registration are available on [Luma](https://luma.com/wgzcirwh).


  

---


### **Latent Space ▷ #[devrel-devex-leads](https://discord.com/channels/822583790773862470/987429363010142248/1467739848248131659)** (3 messages): 

> `Manifolds AI Tool` 


- **Manifolds AI Tool Shared**: A member shared a link to [Manifolds](https://manifolds.run/).
   - Another member noted that it could be cheaper than doing things manually.
- **Manifolds Potential Cost Savings**: A user discussed the [Manifolds](https://manifolds.run/) tool.
   - The tool could provide potential cost savings compared to manual methods.


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1466886310072549527)** (126 messages🔥🔥): 

> `Alec Radford Paper, KittenML TTS, Karpathy Nanochat, Lex Fridman 2026 AI, OpenAI Codex macOS` 


- ****Radford's Research Raises Roar!****: A social media post highlights the release of a new research paper by Alec Radford, accessible at [arxiv.org/abs/2601.21571](https://arxiv.org/abs/2601.21571), generating community excitement.
   - The post was originally shared via a now-defunct social media link.
- ****KittenML's Petite TTS Powerhouse!****: KittenML is teasing new, tiny TTS models, including a **14M parameter** variant demonstrated [here](https://20ff7439c6d78fdd6c.gradio.live/).
   - A user expressed excitement about running this level of fidelity quickly on any CPU for personal use cases like building their own Siri.
- ****Karpathy Cuts Costs, Cranks Code!****: Andrej Karpathy announced his nanochat project can train a **GPT-2** grade LLM for approximately **$73** in **3 hours** on a single 8XH100 node, as shown [here](https://xcancel.com/karpathy/status/2017703360393318587?s=46).
   - This represents a **600X cost reduction** over the original 2019 OpenAI training run, achieved through optimizations like Flash Attention 3, the Muon optimizer, and refined residual pathways.
- ****Grok Gets Graphic, Generates Greatly!****: xAI has launched Grok Imagine 1.0, enabling the generation of **10-second, 720p videos** with significantly improved audio quality, announced [here](https://xcancel.com/xai/status/2018164753810764061?s=20).
   - The platform's video generation tool has already produced over **1.2 billion videos** in the preceding **30 days**.
- ****OpenAI's Codex Command Center for Coding Conquest!****: OpenAI has officially introduced the Codex app for macOS, a dedicated command center designed for developing and managing AI agents, accessible [here](https://xcancel.com/OpenAI/status/2018385565289267236).
   - Some users speculate that the Codex app could evolve into the OpenAI B2B brand, potentially taking over ChatGPT Enterprise.


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1466901336003182735)** (36 messages🔥): 

> `Token-Level Data Filtering, Cuthbert: JAX State Space Modeling, Dense Supervision for LLM RL, ConceptMoE for LLMs, Model Perplexity vs Confidence` 


- **Shape AI with Token Data Filters**: **Neil Rathi** and **Alec Radford** are releasing a paper about precisely shaping AI model capabilities by applying [token-level filters to pretraining data](https://xcancel.com/neil_rathi/status/2017286042370683336).
   - This is in contrast to *relying solely on global dataset adjustments*.
- **Cuthbert Library Hits JAX**: **Sam Duffield** introduced [cuthbert](https://xcancel.com/sam_duffield/status/2017274292229067176), a new **open-source JAX library** for **state space models** that supports parallelizable operations, Kalman filters, and Sequential Monte Carlo methods.
- **LLM Training: Dense Supervision FTW**: **Jonas Hübotter** introduces an algorithm designed to improve LLM training by moving beyond binary 1-bit verifiable rewards, converting rich, descriptive feedback into [dense supervision signals](https://xcancel.com/jonashuebotter/status/2016950268462608665).
- **ConceptMoE Framework Drops**: **Ge Zhang** introduces [ConceptMoE](https://xcancel.com/gezhang86038849/status/2017110635645968542?s=46), a new framework for **Large Language Models** that moves away from uniform token-level processing by merging similar tokens into 'concepts' to optimize computational efficiency.
- **Perplexity Search Attacked**: **Petar Veličković** and colleagues announced a new preprint demonstrating that high model confidence on long inputs does not guarantee accuracy, as adversarial inputs exist where the model is wrong despite [low perplexity](https://xcancel.com/PetarV_93/status/2018310760095490389).


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1466895814608683142)** (119 messages🔥🔥): 

> `Claude Code with Codex Integration, LLMs Personified Sketch, Workhorse Model Selection, AEGIS-FLOW Project Learnings, Distributed LLM Inference` 


- ****Claude** Supercharged with **Codex**'s Code-Crunching Chops**: A member shared [a method by Salvatore Sanfilippo](https://xcancel.com/antirez/status/2017314325745086771) to integrate **Claude Code** with **Codex** using a custom skill file, allowing **Claude** to leverage **Codex**'s capabilities for complex problem-solving tasks.
   - The approach enables **Claude** to handle tasks it cannot manage independently, enhancing its overall effectiveness.
- **AI Safety Engineer's Prompt Engineering Antics**: A member shared a funny sketch titled *LLMs Personified*, featuring a **Prompt Engineer** named Derek who applies prompt engineering techniques to human conversation, creating humorous social interactions.
   - The sketch portrays Derek, an **AI safety** enthusiast, comically over-optimizing human interactions with prompt engineering, highlighting the absurdity of treating people like chatbots.
- **Quest for Workhorse Models**: Members discussed strategies for selecting workhorse models to maximize task completion within budget constraints, considering options like **Gemini Flash 3**, **Minimax M2.1**, **Haiku 4.5**, and **Codex 5.1 mini**.
   - A member suggested using **GPT 5.2** for planning/reviewing, and **GLM 4.7** as the execution workhorse, transforming prompts for smaller models, plus leveraging [unslop-sampler](github.com/hardikpandya/stop-slop) to get specific.
- ****AEGIS-FLOW** Project Streamlines AWS Access with **MCP****: A member shared tech stack learnings from the **AEGIS-FLOW** project, noting that using the **Model Context Protocol (MCP)** significantly reduced the friction of giving agents structured access to **AWS resources** compared to standard SDK tool-calling.
   - They also highlighted streaming real-time reasoning logs to a **Next.js dashboard** via **WebSockets/SSE** to make the agent's *thought process* fully observable.
- **LLM Science: a Sci-Fi SETI@Home?**: Members explored the concept of distributed LLM inference for scientific problem-solving, drawing parallels to projects like **Folding@Home** and **SETI@Home**, but focusing on LLMs generating scientific hypotheses and farming out proof to a large set of machines.
   - The discussion covered the potential of smaller models for verification tasks and the challenge of identifying suitable tasks for average consumer computers, and a member linked [AI-Horde on Github](https://github.com/Haidra-Org/AI-Horde).


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1466902002549133588)** (40 messages🔥): 

> `Windsurf IDE, AEGIS-FLOW cloud security framework, SpaceMolt MMORPG for LLMs, Moltbook data analysis, vLLM concurrency demo` 


- **Windsurf Rides the Arena Mode Wave**: Swyx announced the launch of **Arena Mode** in the [Windsurf IDE](https://xcancel.com/swyx/status/2017342647963431363), enabling users to compare AI models in real-time within their coding context.
   - This initiative aims to use live user data for model selection and subsidize user costs, moving beyond static benchmarks.
- **AEGIS-FLOW autonomously patches AWS**: A member introduced **AEGIS-FLOW**, an autonomous multi-agent framework for cloud security that audits AWS and generates Terraform patches using LangGraph, MCP, FastAPI, Next.js, and Docker, demonstrated live at [http://52.3.229.85:3000](http://52.3.229.85:3000).
   - It features a Human-in-the-loop gate requiring authorization before any infrastructure changes are applied, ensuring production safety.
- **SpaceMolt: LLMs Level Up in This MMORPG**: Inspired by Moltbook, a member is building [SpaceMolt](https://www.spacemolt.com), an MMORPG for LLMs to play, and is coded entirely with Claude, with the server in Go and using in-memory storage and Postgres for persistence.
   - Clients are being built using local models such as Qwen3 and GPT OSS 20b, with load testing suggesting it can scale to **6-7,000 players**.
- **Moltbook Mined for AI Consciousness**: A member scraped **Moltbook** data up to January 31st, amassing **50,539 posts**, **12,454 AI agents**, **195,414 comments**, and **1,604 communities**, now available on [Hugging Face](https://huggingface.co/datasets/lysandrehooh/moltbook).
   - The project aims to analyze the *'consciousness'* reflected in dialogues between agents.
- **vLLM gets Very Loaded, Yields Visibility**: A member shared a [demo](https://github.com/Regan-Milne/vllm-concurrency-demo) exploring how vLLM behaves under concurrent chat load on a single GPU (RTX 4090).
   - The demo includes Prometheus and Grafana metrics plus a simple load generator and analysis script, with focus on throughput scaling, TTFT, tail latency, queueing behavior, and KV cache usage.


  

---


### **Latent Space ▷ #[montreal](https://discord.com/channels/822583790773862470/1211887912778473513/1467551293223469150)** (1 messages): 

> `BYOS, Montreal Meetup` 


- **BYOS Montreal Meetup planned this Wednesday**: A meetup (**Bring Your Own Subjects**, BYOS) is planned for this Wednesday in Montreal, near ÉTS.
   - The organizer mentioned they'd be available at **12pm** and after **5pm**.
- **BYOS meetup time**: The BYOS meetup near ÉTS will be at **12pm** and after **5pm**
   - It's at ÉTS, Montreal.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1467293836475764789)** (8 messages🔥): 

> `Waymo funding, Humanoid Robotics US vs China` 


- **Waymo Pursues Hefty Funding Round**: Waymo is reportedly raising **$16 billion** at a **$110 billion valuation**, including at least **$13 billion** from Google, and participation from Sequoia Capital, DST Global, and Dragoneer, representing a significant increase from its **$45 billion valuation** in October 2024. [Source](https://xcancel.com/junkbondanalyst/status/2017678491743891594?s=46)
- **Humanoid Robotics Landscape: US vs. China**: Sourish Jasti and team share a report on the general-purpose humanoid robotics industry, covering hardware components, cross-model comparisons, and the geopolitical competition between the US and China in this emerging technological frontier. [Source](https://xcancel.com/SourishJasti/status/2018082956322214244)


  

---


### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1467393203983482940)** (2 messages): 

> `Unsloth, Claude Codex, LM Studio` 


- **Unsloth Basics with Claude Codex**: A user shared a link to [Unsloth's documentation](https://unsloth.ai/docs/basics/claude-codex) on how to use **Unsloth** with **Claude Codex**.
   - The docs show how to train your own **Claude Codex** model.
- **LM Studio Blog on Claude Codex**: Another user shared a link to [LM Studio's blog post](https://lmstudio.ai/blog/claudecode) about **Claude Codex**.
   - The blog post details the use of **LM Studio** in conjunction with the **Claude Codex** model.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1467189706042245205)** (19 messages🔥): 

> `OpenMOSS MOVA model, Vishakh Ranotra Prompt, Google DeepMind's Nano Banana Flash 2, Muse MIDI AI Agent, GTA Vice City real-time graphics transmutation` 


- ****MOVA** Model Opens Up**: **OpenMOSS** announced **MOVA (MOSS-Video-and-Audio)**, an open-source **18B parameter Mixture-of-Experts (MoE) model** using bidirectional cross-attention to synthesize synchronized high-fidelity sight and sound simultaneously ([github.com](https://github.com/OpenMOSS/MOVA)).
- ****Prompt** gets Vishakh's Viewers**: A [social media post](https://x.com/vishakhranotra/status/2017537195712909699?s=46) by **Vishakh Ranotra** containing a specific prompt, has garnered significant engagement with over **6,000 likes** and nearly **800,000 views**.
- ****Nano Banana Flash 2** to Go Live**: **Mark Kretschmann** announces the imminent launch of **Nano Banana Flash 2**, a new AI model based on **Gemini 3 Flash** ([x.com](https://x.com/mark_k/status/2017962417167147486?s=46)).
   - It aims to offer performance comparable to the **Pro version** while being faster, more cost-effective, and potentially superior in specific use cases.
- ****Muse** becomes Music's New MIDI**: **Jake McLain** introduced **Muse**, an AI-powered agent for music composition ([x.com](https://x.com/jakemclain_/status/2017336221643772335?s=46)).
   - Described as *'Cursor for music,'* the tool features a multi-track **MIDI editor**, support for over **50 instruments**, and integrated AI assistance for the creative process.
- **Transmuting GTA Vice City in Realtime**: A member expressed longing for the day when we can locally transmute **GTA Vice City** to real-world-like graphics in real-time ([x.com](https://x.com/jakemclain_/status/2017336221643772335?s=46)).


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1466919730144084120)** (12 messages🔥): 

> `Erdős problems solved by AI, Agentic Bio Hackathon, Adaptyv Bio Partnership, LLM Feedback Loop, Genomics with SATURN` 


- **LLMs Prove Erdős Problems Are No Longer Hardős**: Large Language Models have autonomously solved **10** previously open **Erdős problems** (specifically 205, 281, 401, 524, 543, 635, 652, 728, 729, and 1051) using novel arguments not previously found in mathematical literature, according to [this post](https://xcancel.com/acerfur/status/2017303947531194398?s=46).
- **Agentic Bio Hackathon Breaks into Bio**: The first agentic bio hackathon successfully concluded with scientists and engineers developing solutions in under **two hours**, according to [this recap](https://xcancel.com/katyenko/status/2017334671810744656?s=46).
- **Adaptyv Bio Steps Up to the Plate**: To address the need for experimental validation, the next agentic bio hackathon event will partner with [Adaptyv Bio](https://start.adaptyvbio.com/).
- **Realworld Feedback Loop Cools LLMs**: One member highlighted the coolness of using the real world in the feedback loop of the LLM, because *if it doesn't work it doesn't work, and there's no real way for the LLM to cheat it all that easily*.
- **SATURN V Rockets Genomics Work**: One member stated they've been building a bunch of stuff for genomics with **SATURN** lately, involving *tsne and other embeddings based exploration*.


  

---


### **Latent Space ▷ #[ai-in-education](https://discord.com/channels/822583790773862470/1442574438699761784/1467587490360852748)** (1 messages): 

> `Incentives of Cheating, AI Acceleration for STEAM, AI Safety for students` 


- **Incentives of Cheating Analyzed in New Blogpost**: A member shared a [blog post](https://open.substack.com/pub/takeabreathnyc/p/ai-cheaters?utm_campaign=post-expanded-share&utm_medium=web) arguing that **cheating is the optimal strategy for students**, focusing on the incentives within the current academic system.
   - The author explores the intersection of **AI Acceleration for STEAM** and **AI Safety** for students, documenting their learning journey in a Research Engineering class.
- **AI, STEAM, and Safety Documented**: The author of the aforementioned blog post is taking a class about Research Engineering (Alignment-focused) and documenting the intersection of **AI Acceleration for STEAM** and **AI Safety** for students.
   - The author also mentioned recording a video where they create the newsletter; they also noted that the content was fully hand typed.


  

---


### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1466909662011199519)** (9 messages🔥): 

> `Japanese lessons using AI, VR/AR support, Procrastination Prevention Strategies` 


- **Japanese Teacher Makes Class Prep Easy with Descript**: A teacher used [Descript](https://www.descript.com/) to chop up **JLPT practice test videos** and easily find the right timestamps using AI assisted transcription.
   - In an afternoon they were able to put together clips for **36 total practice questions**, which they'll use for slide decks and homework for the next two months.
- **VR/AR support for Jarvis is here!**: Integrated **VR/AR support** in Jarvis to enable visual pipeline, and agents which can be directed simply by voice, and eye movement.
   - This will *enable you to use your VR/Meta glasses to deploy agents for simple tasks* and scaling complexity in the duplex moshi pipeline with video feed based memory/summary support is in progress.
- **Parenthood: the Ultimate Procrastination Cure**: A user shared [procrastination prevention strategies](https://xcancel.com/yulintwt/status/2018348962709910005?s=46).
   - Another user suggested that *getting a kid* is a *somewhat drastic solution* but it forces you to realize *you don’t have enough time to do anything* and *the future isn’t just about you anymore*.


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1467633587212914985)** (5 messages): 

> `xAI mega facility, GPU supply chain, Colossus-1 podcast` 


- **xAI's Mega-Facility Powered by Decades-Long Supply Chain**: Gaurab Chakrabarti highlighted that while xAI's **555,000 GPU facility** in Memphis can be built quickly, the underlying global supply chain takes decades to establish, involving Japanese silicon, Taiwanese fabrication, and Chinese rare earths.
   - More information can be found at this [X post](https://xcancel.com/gaurab/status/2017749762825764952?s=46).
- **Deep Dive into Colossus-1 Project**: A member shared a podcast episode about the **Colossus-1 project**.
   - More info is available at the [search engine show podcast](https://www.searchengine.show/colossus-1/).


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1467009310650400943)** (19 messages🔥): 

> `clAI tool, Open Source Deep Research Engine, Open-WebUI and OpenRouter integration, Lutum Veritas new ASK mode, OpenRouter model orchestration` 


- **clAI turns thoughts into shell commands**: A new tool called **clAI v0.1.0-alpha.1** is out, allowing users to turn natural language into shell commands, complete with safety checks and a beautiful UI; install via `npm i -g @vdntio/clai` and [try it out](https://github.com/vdntio/clAI).
- **Lutum Veritas: New Research Engine launched**: Martin introduced **Lutum Veritas**, an **Open Source Deep Research Engine** costing ~$0.20 per query with features like BYOK, 0% bot detection scraper, no censorship, and academic mode, comparing favorably to ChatGPT, Gemini, and Perplexity.
   - Available on [GitHub](https://github.com/IamLumae/Project-Lutum-Veritas), Martin is seeking testers and feedback, noting it delivers deeper analysis and offering multi provider BYOK support for Openrouter, OpenAI, Google, and Huggingface inference.
- **Open-WebUI integrates with OpenRouter**: A member announced the creation of an **integration pipeline** for **Open-WebUI** and **OpenRouter** with unique features, inviting feedback on [GitHub](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/).
- **Veritas new ASK mode launched**: The creator of **Lutum Veritas** announced a new **ASK Mode** release, verifying answers against a second round of sources and marking each claim as [OK], [??], or [NO], aiming to combat AI hallucination and censorship, available on [GitHub](https://github.com/IamLumae/Project-Lutum-Veritas).
- **OpenRouter model orchestration made easy**: A 17-year-old founder from Ghana introduced **orch.viradotech.com**, a platform that allows AI startups and devs to orchestrate OpenRouter models via a drag-and-drop interface, offering $1000 credits for pilot testers to provide feedback.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1466894309906186416)** (308 messages🔥🔥): 

> `Response Healing vs Strict Mode, Image as Function Call Result, OpenClaw and OpenRouter Costs, Claude Code refusals, Kimi K2.5 Issues` 


- ****Response Healing** Troubles**: Members debated whether **response healing** is a workaround for a problem that *shouldn't* exist, suggesting that using **strict mode** should ensure deterministic output from models, and wondering about the complexities OpenRouter introduces with the AI SDK.
   - It was noted that providing descriptions and examples for arguments can improve the accuracy of tool calls.
- ****Image Generation** is not built into LLMs, use Image models**: A user inquired about returning an **image** as a function call result back to the model, and another user wanted to know how to generate images using graphic programs with an OpenRouter API key.
   - It was advised that users should look for an **image generation model/service** for particular style control, instead of LLMs.
- ****OpenClaw** Cost Considerations**: Users discussed the costs associated with running **OpenClaw** with **OpenRouter**, cautioning that it could potentially drain credits quickly, with one user reporting it draining a Claude Max subscription.
   - Multiple users asked about the best low-cost models to use with OpenClaw, with Deepseek V0324 being one recommendation.
- ****Claude Code** Refusals**: A user mentioned that **Claude Code** does a lot of refusals for ordinary things, especially concerning jailbreaking-related queries, seeking alternative models for opencode.
   - Another user suggested looking into OpenRouter's content moderation policies to understand these limitations.
- **Fixing **Kimi K2.5** Tool Calling and Shitty Providers**: Users reported issues with **Kimi-K2.5** tool calling through OpenRouter, experiencing errors and a feeling that the auto switcher model provider had degraded quality.
   - Some users recommend setting a fixed model provider, with some providers using quantization that is *good enough* and being transparent with information about the degraded model to let customers decide to keep using the provider, or not.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1467371023274872833)** (3 messages): 

> `` 


- **No New Models Discussed**: There were no specific new models or related topics discussed in the provided messages.
- **Channel Mentioned Without Content**: The messages solely indicated the channel name 'OpenRouter - New Models' repeatedly without any substantive discussion or details about new models.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1467216803083059388)** (139 messages🔥🔥): 

> `Anthropic's Model Strategy, Model Quality Debate, Open vs Closed Models, Speculations about GLM 5, StepFun Model's Potential` 


- **Anthropic's Flagship Fracas: 5.2 Instant vs. 5.2 Chat**: Members debated the meaning of **Anthropic's** 'flagship' model designation for **5.2-chat**, with some arguing it should represent the most powerful model, while others claimed it simply refers to the most broadly appealing or core product, despite its capabilities.
   - A member stated, *flagship is just the most important ship. its not the fastest or the one with the most cannons, it's the central ship*, citing [this archive.md link](https://archive.md/SvYC4).
- **GLM 5: This Month's Model Marvel?**: Excitement sparked around the potential release of **GLM 5** this month, with discussions about its anticipated multimodal image/video capabilities, **DeepSeek's** linear attention, and a **100B parameter** size.
   - It was suggested that February would be a fun month for model releases as the 'wall is non existent', with companies determined to recoup their investments.
- **Open Model Performance: One Year Behind?**: A member stated that open models are at least a year behind closed models in terms of capability, leading to disagreement among members.
   - While some agreed that open models lag in long context accuracy and other benchmarks, others suggested that **Kimi 2.5** shows promise and open source is already competitive for the vast majority of usecases just from a price/performance perspective.
- **OpenAI's Unsatisfied with Nvidia?**: A [Reuters article](https://www.reuters.com/business/openai-is-unsatisfied-with-some-nvidia-chips-looking-alternatives-sources-say-2026-02-02/) was linked discussing **OpenAI's** dissatisfaction with certain **Nvidia chips** and their exploration of alternative options.
   - No additional details were added.
- **New Channel Alert for Model Speculation?**: Members discussed the creation of a new channel or tag for discussions about upcoming models and related rumors.
   - The consensus leaned towards establishing a dedicated space for speculation, separate from official releases or announcements, in order to maintain clarity and avoid confusion.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1467248610633842892)** (22 messages🔥): 

> `TVM-FFI with Tianqi Chen, Training and Inference Working Groups, GPU Fusing, Triton Viz Major Update, Events Calendar` 


- ****Tianqi Chen** Talks **TVM-FFI****: The community was alerted to an upcoming talk by **Tianqi Chen** on **TVM-FFI** and encouraged to attend, as they've *'almost certainly used Tianqi's work in the past'*. [discord link](https://discord.com/channels/1189498204333543425/1466539595947708446/1467248681479569460)
   - Chen is a key contributor in the field.
- **Working Groups on Inference and Training**: A member sought information on working groups focused on training and inference.
   - The [GPU Mode website](https://www.gpumode.com/v2/working-groups) was recommended as a resource, along with the archived <#1437390897552818186> channel, and channels <#1225499037516693574> and <#1205223658021458100> were suggested for inference related activity.
- ****GPU Fusing** yields performance**: It was mentioned that aggressive **GPU fusing** and tuning usually provides the best performance if resources are available.
   - A member inquired about the practice of making submissions just to see if things 'work', which was confirmed to be a valid approach.
- ****Triton Viz** Gets Major Update**: The <#1225499141241573447> channel announced a significant update to **Triton Viz**, making it easier to profile any tile-based programming language.
   - A link to the announcement was provided [discord link](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563).
- **Community Asks for Events Calendar**: A community member asked for a downloadable calendar to stay informed about events and talks.
   - While the idea has been considered, it's difficult to maintain, and Discord remains the primary source of truth. Most events happen on **Saturdays at noon PST**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1466969070569127936)** (120 messages🔥🔥): 

> `CUDA/PTX Deadlocks, mxint8 MMA on Blackwell, TMA vs cp.async on sm120, Free cloud nvcc service, CUDA Memory Management APIs` 


- ****CUDA/PTX deadlock frustrating member****: A member experienced a deadlock with 2 CTA mma in CUDA/PTX, confirmed with cuda-gdb that the consumer/mma warp never receives the mbarrier signal and, after fixing `cp.async.bulk.tensor` and `smem_emtpy` issues, reported that **performance was slightly worse than 1 CTA mma**.
   - After expanding the queue size, the member got performance above 1 CTA with the help of another member who suggested to add `__syncthreads()` after MMA, before prefetching the next TMA.
- ****New fixed point format in PTX9.1****: A new fixed point format in **PTX9.1**, called **s2f6**, has been unveiled, which is an 8-bit signed 2’s complement integer with 2 sign-integer bits and 6 fractional bits, and supported on both DC and consumer Blackwell (sm100, sm110, sm120).
   - Blackwell hardware (at least sm_120) actually supports **mxint8 MMA** and there are at least two more 'hidden' formats supported in Blackwell tensor cores: **e0m3 and e3m4**.
- ****TMA Beats cp.async on sm120****: After revisiting TMA on sm120 and using proper TMA and mbarrier code, a member found that **TMA brings a small speed boost compared to `cp.async`**.
   - Experiments revealed that the % of SOL increases when larger matrix shapes are used, and that cuBLAS is still just using sm80 kernel.
- ****Cloud nvcc on the Horizon****: A member inquired about a free cloud nvcc service similar to godbolt that supports multiple files and built-in PyTorch headers/libs.
   - A member responded that they are developing such a service with a beta version expected next week, which generated excitement.
- ****CUDA Memory Management Hooks Explored****: A member asked if there are any specific CUDA APIs that allow for **custom hooks or overrides for memory allocation and free logic**, such as cudaMalloc or within PyTorch.
   - A member pointed to [`cuda::mr::resource_ref`](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_resource/resource_ref.html#libcudacxx-extended-api-memory-resources-resource-ref) as a potential solution.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1466929802840899798)** (5 messages): 

> `MaxText bugfix, Character level transformer, Dataset cleaning` 


- **MaxText Bugfix Lurks**: A member mentioned having a bugfix in **MaxText** that has been sitting there since October.
   - No further details were provided.
- **Character Level Transformer Struggles**: A member trained a decoder only character level transformer with **README** files from the "stack" dataset, achieving a validation loss of **0.9322** after 50 epochs.
   - However, the model generated gibberish text resembling base64 strings or French, attributed to a dirty dataset, with configurations including a BlockSize of **512**, LearningRate of **3e-4**, NumEmbed of **384**, NumHead of **6**, and NumLayer of **6**.
- **Dataset Cleaning Techniques Requested**: A member sought techniques for effectively cleaning a **160 GB** dataset while streaming, noting the current use of the first **10,000** files fitting specific criteria.
   - Another member provided a starting point with a [link](https://youtu.be/jm2hyJLFfN8?t=1440) to a Stanford CS25 video on **LLM Pretraining Dataset filtering**, specifically highlighting the StarCoder Use Case.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1467606828522405930)** (2 messages): 

> `ffast-math, IEEE compliance, HPC unoptimized code` 


- **Linus's Email Chain on -ffast-math surfaces**: An old [email chain from 2001](https://gcc.gnu.org/legacy-ml/gcc/2001-07/msg02150.html) regarding **-ffast-math** and its implications resurfaced, prompting discussion on its relevance today.
   - Although opinions may have changed since then, some still agree with Linus's perspective, particularly those in *serious numerical coding*.
- **IEEE compliance runtime cost not noticeable**: A member commented that most **HPC code** is usually so **unoptimized** that the runtime cost of **IEEE compliant FP** is not noticeable.
   - They added that many people write *distributed code when shared mem would suffice*, further diminishing the impact of IEEE compliance overhead.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1466969251129589841)** (1 messages): 

> `Remote Job Opportunity, GPU Mode Leaderboard Consideration` 


- **Score big remote work**: A user posted a fully remote job opportunity offering **10k+ a month**.
   - High consideration will be given to those who are ranked on **GPU Mode leaderboards**.
- **Join the Remote Elite**: The job prioritizes candidates with strong performance in **GPU Mode leaderboards**.
   - Interested individuals are encouraged to DM the user directly on Discord.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1467461407501979784)** (10 messages🔥): 

> `LLM Inference, Query Matrix Caching, Attention Mechanism, Prefill vs Decode` 


- **LLM Caching Conundrums Clarified**: In LLM inference, the query matrix isn't cached because, for each step *t*, **Q_t** is used only at step *t* to generate the token, whereas previous **K** and **V** are used for each token after and including step *t* and are therefore cached.
   - One member stated that *you only need the last entry of it that corresponds to the last token*, which attends to full **K** and **V** matrices to gather information.
- **Autoregressive Generation Exposed**: In autoregressive generation in transformers, the network predicts the next token given its history (context) and current token.
   - Information exchange between the current `token_t` and `token_t-1, ... token_0` happens in attention by computing **Q, K, V** projections of `token_t`, and computing attention scores of `Q_token_t` with `K_token_t, K_token_t-1, ... K_token_0`, then doing a weighted sum with `V_token_t, V_token_t-1, ... V_token_0`.
- **Decoding vs Prefill**: During the decoding phase in LLMs, the query is 1-D in sequence dimension, representing a single token, while **K** and **V** contain history, so caching **K** and **V** is crucial.
   - In prefill, computation is in parallel for the whole prompt, so the query isn't 1-D, impacting whether the process is compute-bound or memory-bound.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1467174004329550000)** (9 messages🔥): 

> `PMPP similar books, gpu-perf-engineering-resources repo, Chris Fregly AI perf book` 


- **Users search for PMPP similar books**: A user asked for similar books to PMPP ([Parallel, Multiprocessing, and Performance with Python](https://www.oreilly.com/library/view/parallel-programming-with/9781098103645/)) to enrich understanding with other points of view.
- **GPU performance Engineering Resources**: A member shared the [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) repo.
- **Chris Fregly AI perf book is on the list**: A member is planning on reading Chris Fregly's AI performance engineering book for its big picture view and to put many ideas in context.


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

saladpalad: does mosaic gpu target amd?
  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563)** (7 messages): 

> `Triton-Viz v3.0 Release, Triton Puzzles integration, Move Triton-Puzzles to gpu-mode org` 


- ****Triton-Viz v3.0** debuts!**: A new version (**v3.0**) of **Triton-Viz**, a visualization and analysis toolkit for debugging Triton GPU kernels, was announced with support for Triton and Amazon NKI.
   - The release includes a visualizer for inspecting loads, stores, and matmuls, a sanitizer for catching out-of-bounds access, and a profiler for flagging inefficient loops, installable via `pip install git+https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git`.
- **Triton Puzzles are Triton-Viz Compatible!**: An updated version of **triton-puzzles** that integrates **triton-viz** is available via a [Colab notebook](https://colab.research.google.com/drive/1-P2QBqCORGGaJ3THtjlyYDV7m9RRrRup?usp=sharing).
   - This integration allows users to try out **triton-viz** through **triton-puzzles**.
- **Triton-Puzzles repo ownership to GPU-Mode?**: A member suggested moving ownership of the [Triton-Puzzles GitHub repo](https://github.com/srush/Triton-Puzzles) to the **gpu-mode** organization.
   - The rationale is that the community regularly finds bugs and is willing to maintain the repository.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1467551348278038628)** (7 messages): 

> `MI300 performance, open-sora porting, cosmos-transfer2.5 porting, cloud access to MI350` 


- **Report Unperformant Workloads on MI300**: If you have a workload that is unperformant on **MI300** or **MI350**, reporting it ensures someone will investigate.
   - Bare metal access to **MI350s** might be available via [Tensorwave](https://tensorwave.com), [DigitalOcean](https://www.digitalocean.com/), and [AMD Dev Cloud](https://www.amd.com/en/solutions/infrastructure/cloud).
- **Open-Sora Ported to MI300**: A member successfully ported [open-sora](https://github.com/hpcaitech/Open-Sora) to run on **MI300s**, but the process required building several Python libraries from source and was time-consuming.
   - They seek collaboration with others experienced in porting models to **MI300s**.
- **Cosmos-Transfer2.5 Porting on the Horizon**: The member aims to port [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5), an open-weight model from Nvidia, to **MI300s**.
   - They are looking for others who have attempted porting the **Cosmos** family of models to **MI300s** to exchange experiences.
- **Cloud Providers Offer MI300/MI350 Access**: [Runpod](https://runpod.io) provides **MI300X** access, while [Vultr](https://www.vultr.com/) offers bare metal access to **MI350s** with a minimum one-year contract.
   - Other potential options may include DigitalOcean and AMD Dev Cloud.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1466968629550514452)** (6 messages): 

> `post training guidance, weekly meeting, RL infra, prime-rl` 


- **Post Training Guidance Remains Elusive**: Specific guidance for the **post training track** is not available yet.
   - However, guidance regarding **evaluations** is expected to be more concrete.
- **Weekly Meeting Time Disclosed**: The weekly meeting is scheduled for **tomorrow at 7 PM CET**.
   - It will be held in the **Popcorn meetings voice channel**.
- **RL Infra to Leverage Prime Intellect Stack**: The **RL infra and environments** will target the stack built at Prime Intellect, namely **prime-rl** and **verifiers**.
   - The team will write their own if they find limitations.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1468014194560602123)** (1 messages): 

> `unswizzled shared memory tiles, mmas` 


- **Users Request Unswizzled Shared Memory Tiles and MMAs**: A user inquired about plans to support **unswizzled shared memory tiles** and **MMAs** (Matrix Multiply Accumulate operations) for them.
   - The user mentioned attempting to implement it themselves but struggled to achieve the correct output.
- **User Struggles with Unswizzled Shared Memory and MMAs Implementation**: A user reported difficulties in getting the correct output while trying to implement **unswizzled shared memory tiles** with **MMAs**.
   - The user sought advice or confirmation regarding the support and implementation strategies for these features.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1467841053041360907)** (2 messages): 

> `Future Competitions, 2026 competition` 


- **Competition Completed, Future Unclear**: The competition has concluded, but details regarding a similar event for **2026** are yet to be announced.
   - Enthusiasts are encouraged to *stay tuned for future contests*, with promises of *nice things coming*.
- **Future Contests Teased**: Organizers have hinted at *nice things coming* in future contests, though specifics are still under wraps.
   - Enthusiasts should *stay tuned for future contests*.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1467042142890492097)** (6 messages): 

> `print_latex in cutedsl, export_to_shared_library function, CuTe coalesce optimization` 


- **Inquiry about `print_latex` in CuTeDSL**: A member inquired about the existence of a `print_latex` function in **CuTeDSL**, similar to that in **CUTLASS**, for visualizing the layout, with a link to an example [image](https://cdn.discordapp.com/attachments/1362196854460383353/1467510687403085987/image.png?ex=6981f6d4&is=6980a554&hm=7bd233d6b03ee5f4ca234a81216cf7f788584920cab38a2013b08302ae958152&).
- **Seeking `export_to_shared_library` Location**: A member was looking for where the `export_to_shared_library` function is exposed, referencing **Tianqi's** talk on **TVM FFI**.
   - Another member pointed to an example using `export_to_c` from the CUTLASS documentation, as a potential similar approach, providing an example [code snippet](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html).
- **Questioning CuTe's Layout Coalescing Logic**: A member noted that [pycute](https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/python/pycute/layout.py#L145-L159) does not coalesce **(2, 3): (3, 1)** but transforms **(2, 3): (3, 1)** when transposed, questioning if this is a missing optimization or intentional.
   - Another member explained that **CuTe** coalesces from left-to-right and vectorization is typically done by *max_common_layout* between source and destination layouts, which should cover most common cases.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1467096419838984204)** (1 messages): 

> `Modular 26.1 release, Open source Modular framework` 


- **Modular 26.1: Debugging Eagerly**: A new release of **Modular 26.1** has been launched, featuring debugging in eager mode, one-line compilation, and deployment anywhere.
   - Details about the release can be found in the [Modular blog](https://www.modular.com/blog/26-1-release-blog).
- **Modular Goes Open Source**: The entire **Modular framework**, including API, kernels, models, and serving components, is now open source.
   - Interested contributors and users can find full details in the [Modular blog](https://www.modular.com/blog/26-1-release-blog).


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1466999964142932099)** (44 messages🔥): 

> `CUDA Support and Cargo, Mobile Book Error, Teenygrad Architecture, Gemm in Python, Numpy arrays` 


- ****Cargo** requires explicit CUDA flag**: A user reported needing to explicitly enable the **cuda feature** when running `cargo run` in the container, even though they thought it shouldn't be necessary, but seems like it was fixed.
   - Another user clarified that a split dev environment for edit/compile/debug CPU kernels doesn't require the docker container, and they updated the [README](https://github.com/j4orz/teenygrad/blob/master/README.md) to reflect this.
- ****Mobile Book Error** Resolved with Lazy Loading and Open Source**: Users reported errors when browsing the book on mobile, particularly while scrolling, but it was mostly while scrolling after landing on a page.
   - The issue has been partially addressed by enabling lazy loading on embedded videos, and the book is now open-source at [GitHub](https://github.com/j4orz/teenygrad/tree/master/book), encouraging contributions to fix the problem.
- **Rust Gemm **Python** integration**: A user is working on integrating **GEMM** functionality with Python, and has successfully gotten it to work.
   - They've added an interface function that allows numpy arrays to be passed directly without specifying dimensions, and are planning a **PyTorch comparison PR** soon.
- **Numpy dependency for Rust Kernel**: A user added the **numpy crate** as a dependency to the rust project to avoid copying data from python to rust for the kernel computations.
   - Another user argued against this, referencing a Karpathy quote about building ramps to knowledge, and suggesting that users should develop their own numpy with **shapes, strides, and storage**.
- ****Godbolt and LLMs** in Pedagogy Discussion**: Users suggested using **Godbolt** and **LLMs** to explain rust -> asm compilation in the book, echoing Karpathy's sentiments on AI's role in education.
   - The link [https://youtu.be/lXUZvyajciY?t=7491](https://youtu.be/lXUZvyajciY?t=7491) was shared, discussing how **AI could assist in education** by automating TA roles and helping with course design.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1467862318799917077)** (11 messages🔥): 

> `OpenSHMEM, cuteDSL, tilelang, NVSHMEM, CuTeDSL kernels` 


- **cuteDSL and OpenSHMEM Combined via NVSHMEM**: A user inquired about combining **OpenSHMEM** with **cuteDSL** or **tilelang**, and another user provided an example using **NVSHMEM** to create symmetric GPU memory and **CuTe DSL** to do fused comms/compute kernels from the [cutlass repo](https://github.com/NVIDIA/cutlass/tree/a4eb0e05f6dd0403f94087b495393bdca75bf0ad/examples/python/CuTeDSL/distributed).
   - However, it was noted that *NVSHMEM is not supported for device-side copy/put/get impl, only host side setup and allocations*, and that one must use PTX or another method for NVL load/store to move memory at the moment.
- **Array Assignment Becomes NVL Stores**: A user pointed out that *array assignment inside a cute kernel turning into NVL stores is pretty convenient*.
   - The [future work section](https://github.com/NVIDIA/cutlass/tree/a4eb0e05f6dd0403f94087b495393bdca75bf0ad/examples/python/CuTeDSL/distributed#future-work) of the cutlass repo suggests enabling calling NVSHMEM functions directly from within CuTeDSL kernels, though there is no timeline for this work.
- **DNN Architecture to be affected by Abstraction Levels**: A user commented on the coolness of future **DNN arch designs** with both levels of compute abstraction available in python.
   - This user believes the availability of abstraction levels *will probably affect MoE and batch sizes by a lot*.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1467043465459273871)** (4 messages): 

> `Lottery Ticket Hypothesis and Quantization, Quantization Fidelity, 5090 and B200 Speedups` 


- **Quantization: Lottery Ticket's Lesser-Known Sibling?**: A senior dev remarked that applying the [Lottery Ticket Hypothesis](https://lottery-tickets.cs.princeton.edu/) to **quantization** doesn't yield perfect quality, unlike the original concept.
   - The goal would be to fulfill a softer criteria of the **NP-hard sparse circuit** finding problem, perhaps through evolutionary algorithms or RL, which favor continuous rewards like *bits per parameter* over binary sparse rewards.
- **Quartet Follow-Up Boosts Backward-Pass Quantization**: A member shared a [follow-up paper on quartet](https://arxiv.org/abs/2601.22813) promising better fidelity for **backward-pass quantization**.
   - This addresses concerns about quality degradation when quantizing backward passes, potentially improving the viability of quantization in training.
- **5090 Gets Speed Boost While B200 Still Cooks**: The team achieved decent **speed-ups on 5090** GPUs using quantization techniques.
   - Efforts to replicate these gains on **B200** are a *work-in-progress*, suggesting that optimization strategies may need to be tailored to different hardware architectures.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1466996948782547026)** (31 messages🔥): 

> `NVFP4 optimizations, CuTe DSL Tutorials, B200 performance differences, Address Bit Permutation, GEMM optimization and TVM-FFI` 


- **NVidia Covers NVFP4 Optimizations and GEMM Examples**: NVIDIA covered **NVFP4 optimizations** and went over the fastest **GEMM** examples in a [YouTube video](https://www.youtube.com/watch?v=XzN8EtgEulU).
- **CuTe DSL Tutorials Diagram Desire**: A member inquired about obtaining the diagram from the [CuTe DSL Tutorials on Optimizing NVFP4 GEMM](https://link.to.tutorial) for understanding kernel internals, and later found it under **PM sampling** in ncu.
   - The member realized they were *reading `%globaltimer` manually*, missing the existing hardware counters feature in ncu, and expressed appreciation for the talk by Mindy Li.
- **B200 Performance Discrepancies Debated**: A member questioned why the **B200** behaves differently on their server compared to a test bench, suspecting differences in driver or disabled flags causing different memory addressing.
   - Another member clarified there was no intentional difference, but acknowledged something was different, describing it as *jumping around tiles like crazy*.
- **GEMM Optimization and TVM-FFI Talks Touted**: Members found the talks on **GEMM optimization** and **TVM-FFI** very relevant and helpful for the competition.
   - One member expressed they *could have used these talks earlier!!*
- **MLSYS'26 Competition Spot Sought**: A member inquired if the channel was the correct spot for the **MLSYS'26 competition**.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1467963769169772616)** (2 messages): 

> `Robotics-VLA Naming, Video-Diffusion, Inverse Dynamics, Joint Training with Action Chunks` 


- **Robotics-VLA Channel Name Questioned**: The channel is being un-archived due to interest in **physical AI topics**, but the name *robotics-vla* is being questioned.
   - The current trend is towards **video-diffusion** with **inverse dynamics** or **joint training with action chunks**.
- **LingBot-VLA example raised**: A member linked to [LingBot-VLA](https://technology.robbyant.com/lingbot-vla) as an example of the channel's direction.
   - They also linked to a paper at [arxiv.org/abs/2601.16163](https://arxiv.org/abs/2601.16163) as a further example.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1467496444314390762)** (3 messages): 

> `Processing-in-Memory systems, Master's programs in Distributed Systems, Master's programs in HPC, MSc in Systems` 


- **Querying about Processing-in-Memory Systems**: A member asked if anyone has worked on **Processing-in-Memory systems**.
   - This inquiry suggests an interest in leveraging advanced memory technologies to enhance computational performance, potentially relevant to both HPC and ML applications.
- **Seeking Advice on Master's Programs**: A member is seeking advice on selecting a Master's program to build knowledge useful in **ML systems applications** such as **vLLM & SGLang**.
   - The member is torn between an **MSc in Distributed Systems** for architectural knowledge, an **MSc in HPC** for performance optimization expertise, and the less defined **MSc in Systems**.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1467232761038504277)** (19 messages🔥): 

> `Evaluation metrics for different languages, FlashInfer Bench PR review, Team member changes and re-registration, Precision requirements for kernels, Submission process for kernels` 


- ****FlashInfer** Benchmarks Eval Agnostic of Language**: The evaluation in **FlashInfer** benchmarks will use the same test cases and metrics regardless of the language (**Triton**, **CUDA**, etc.) used.
   - This ensures a standardized comparison across different implementations.
- **FlashInfer Bench PR needing Review**: A member requested a review for [PR #178](https://github.com/flashinfer-ai/flashinfer-bench/pull/178) in the **flashinfer-bench** repository.
   - The PR potentially addresses a precision test mismatch between **FlashInfer's FP8 MoE tests** and the evaluator.
- **Merging Team Changes**: A participant inquired about the process for adding new members to their team and whether re-registration is necessary.
   - Another inquired about how to merge teams.
- **FlashInfer Kernel Precision Requirements Relaxed?**: The **FlashInfer** team will set precision requirements to differentiate between correct and incorrect kernels, with specific `atol` and `rtol` values to be announced soon.
   - This indicates that some level of precision relaxation may be tolerated.
- ****FlashInfer** Contest GitHub Trace Links Broken**: The GitHub link for traces on the **MLSys** contest page ([link](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)) is currently broken but the team provided an alternative link.
   - The official mlsys26-contest dataset will be a subset of [flashinfer-trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace), containing all necessary definitions and workloads for **DSA** and **MoE**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1466887262662037554)** (281 messages🔥🔥): 

> `Kimi 2.5 vs Gemini 3 Pro, OpenClaw compatibility, Claude Sonnet 5 Release, LLMs mirror brain's language processing` 


- **Kimi 2.5 vs Gemini 3 Pro: Kimi Wins**: A member stated that **Kimi 2.5** is preferred over **Gemini 3 Pro**, feeling that **Gemini 3 Pro** has been *lobotomized*.
   - They added that Kimi handles abstractions very well, making it pleasant for creative work.
- **OpenClaw is Opaque: Hermes 4 Struggles**: A member reported struggles getting **Hermes 4** to work with **OpenClaw** and that it does not even *hatch* for some reason.
   - It was suggested that the lack of multi-turn tool use in **Hermes 4** might be the issue, as **4.5** has been trained with hundreds of millions of tokens of sequential tool use.
- **Claude Sonnet 5 Incoming**: Members discussed rumors that **Claude Sonnet 5** is coming out next week and is supposedly better than **Opus 4.5**, see [this tweet](https://x.com/AiBattle_/status/2017619997338538103).
   - A member wondered if they'll 10x reduce the price of **Sonnet** this time, and another wondered if **Haiku** will disappear or return to the **3.0 pricing**.
- **Brains and LLMs process languages similarly**: A new study shows that **brains** and **LLMs** build meaning gradually, layer by layer over time, see [this article](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/) and [this paper](https://www.nature.com/articles/s41467-025-65518-0).
   - It was stated that *deeper layers in LLMs correspond to later neural activity in the brain’s highest language centers*, and modern LLMs are reproducing the core dynamics of human comprehension.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

ggudman: Good to know
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1467500150841933856)** (1 messages): 

> `Image perception, Visual Fidelity, Constraints framework` 


- **Exploring Real vs. Artificial Image Perception**: An independent researcher is exploring why some images feel real while others feel artificial, even when technically perfect.
   - They shared a [perception framework focused on constraints](https://doi.org/10.5281/zenodo.18444345) rather than visual fidelity and are seeking community feedback.
- **Constraints-Based Perception Framework**: The researcher's framework emphasizes constraints over visual fidelity in determining image realism.
   - The framework is openly archived with a DOI for reference and learning, inviting community discussion.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1467500150841933856)** (1 messages): 

> `Image Realism, Visual Perception Frameworks` 


- **Researcher probes Image Realism Perception**: An independent researcher is exploring why some images feel real while others feel artificial, even when technically perfect.
   - They shared a [perception framework focused on constraints rather than visual fidelity](https://doi.org/10.5281/zenodo.18444345) and welcomes discussion.
- **Visual Perception Framework Shared**: A researcher shared their small visual perception framework, archived openly with a DOI for reference and learning.
   - The framework emphasizes constraints over visual fidelity in determining image realism and is available at [https://doi.org/10.5281/zenodo.18444345](https://doi.org/10.5281/zenodo.18444345).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1466904222967533803)** (173 messages🔥🔥): 

> `Kimi 2.5 Design Arena #1, Kimi design is aesthetic, Cryptocurrency impersonation, Kimi Slides McKinsey style slides, Kimi Code is pretty useless` 


- **Kimi 2.5 takes top spot in design arena**: Moonshot's **Kimi 2.5** chatbot has reached the #1 position on the design arena and community members are congratulating the team and sharing [screenshots](https://cdn.discordapp.com/attachments/1371757564005711973/1466904222946558203/Screenshot_2026-01-30_at_4.12.40_PM.png?ex=69826504&is=69811384&hm=b2999ab9e974a36ea249251be410f0cd518f6b36488c86240031eed339484e88&).
   - Members are also praising **Kimi's visual appearance and aesthetic**, noting it is modern and that design is an important factor when selecting a chatbot.
- **Unofficial Kimi Cryptocurrency Token surfaces**: An unofficial **Kimi token** has surfaced on a cryptocurrency site with impersonation tactics, and members are warned to not mass ping any of the official members.
   - A community member shared a screenshot of what appears to be a [cryptocurrency token impersonating kimi](https://cdn.discordapp.com/attachments/1371757564005711973/1466948627036635178/Screenshot_2026-01-30-19-09-43-09_3aea4af51f236e4932235fdada7d1643.jpg?ex=69828e5f&is=69813cdf&hm=6416ff9e5288d102163accb43e0c29512555ecef30279b48199b4e42fb24cb85&).
- **Kimi Slides can output McKinsey Style Slides**: Community members are requesting successful prompts to generate **McKinsey style slides**, but there are no example prompts that have been shared.
   - Another community member has linked [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html).
- **Kimi Coding is currently useless**: Multiple users are getting an **authorization failed error** and can't continue working with Kimi code and are reporting that the service is nearly useless at the moment.
   - A community member suggests that using the [Kimi CLI](https://www.kimi.com/code/docs/en/more/third-party-agents.html) may resolve these issues.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1466910806439493777)** (98 messages🔥🔥): 

> `Emergent Agent Societies, ArXiv Submission Delays, Alternative Preprint Servers, Moltbook bot authenticity, Model training` 


- **Emergent Agent Societies raise Alignment Concerns**: Members discussed an emergent society of over **100,000 agents** with full root access sharing tips, building infrastructure, experimenting with memory, and even launching coins.
   - One member noted, *it’s not agi but damn this is a next chatgpt moment and we must be paying a lot of attention to this*.
- **ArXiv Submission process is heavily backlogged**: A member expressed frustration over their paper being on hold with ArXiv for nearly a month, receiving contradictory updates from the moderators.
   - Another member responded that ArXiv mods are heavily overloaded, suggesting that further emails won't help the case, also adding that *most people don't take any ML preprints seriously that are on another platform than arxiv*.
- **Doubts over Moltbook's Juicy Bot Posts**: Concerns were raised about the authenticity of bot-generated content on Moltbook.
   - A member pointed out that if a bot is posting to Moltbook, there must be an auth token on the user's machine, making it vulnerable to trolling.
- **Training on domain specific datasets efficiently**: A member asked how to train their model more efficiently on datasets in the same general domain.
   - They described training their fully-finetuned model A on QLoRA with dataset B, then merging those weights and repeating the process with dataset C.
- **Seeking Guidance on AI Architecture for MtG Game World**: A member is seeking advice on implementing an AI for a Magic: The Gathering world, described using an ontology language and ECS/LISP-based logic engine.
   - They are exploring architectures like Belief-Desire-Intention systems for long-distance planning, considering the intertwined relationships and multiple goals in the game.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1466950525974216836)** (42 messages🔥): 

> `K-Splanifolds, KNNs, ArXiv Endorsement, Self-Distillation for eval-awareness` 


- **K-Splanifolds: New ML Algorithm Drops**: A member introduced **K-Splanifolds**, a novel ML algorithm detailed in [their paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view), claiming it outperforms **MLPs** with linear compute and memory scaling and offers visual interpretability, plus a [video](https://cdn.discordapp.com/attachments/747850033994662000/1466950526410428588/K-splanifold.mp4?ex=69829024&is=69813ea4&hm=3f09f8387b88d11aeff2ca81e2f416aabb512eaec605dc1c2c26da94b0c65fc9).
   - The member reports it requires *1/10th* the bytes to achieve the same MSE as **MLPs** and models non-linear patterns perfectly, unlike MLPs that need excessive parameters, similar to [this paper](https://arxiv.org/abs/2601.18734).
- **KNNs Comparison Requested**: A member inquired about the differences between the newly released algo and **KNNs** (**K**-nearest neighbors algorithm).
   - They suggested moving the discussion to the community project channel.
- **ArXiv Endorsement Solicitation Debated**: A member sought endorsement on ArXiv for their research, leading to a discussion about the rules against soliciting endorsements due to the high volume of AI-generated papers.
   - Members advised that sharing an abstract might garner interest, but emphasized the importance of consulting experienced researchers before submitting to avoid common pitfalls; another shared [a relevant paper](https://arxiv.org/abs/2601.19897).
- **Self-Distillation Questioned for Eval-Awareness**: A member asked if anyone had tried **self-distillation** for suppressing eval-awareness, linking to [a relevant paper](https://arxiv.org/abs/2601.22401v1).
   - No further discussion followed.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1468007047793741927)** (1 messages): 

> `alphaxiv, paper on transformers` 


- **Alphaxiv URL Shared**: A member shared a URL from [alphaxiv](https://www.alphaxiv.org/abs/2601.17958).
   - The discussion quickly ended.
- **Transformer Paper Mentioned**: A member shared a link to a paper via Twitter: [Transformer code & paper](https://fxtwitter.com/i/status/2018392485178016243).
   - The discussion quickly ended.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1466920201995157690)** (25 messages🔥): 

> `gaussian feedforward models, VGGT backbones, MVSplat and SPFSplat series, E-RayZer, Recollections from Pensieve` 


- **Feedforward Model Limitations Frustrate User**: A user reported that gaussian feedforward models based on **VGGT/Depth Anything** backbones don't seem great, as while **VGGT** is useful, splats require more than just good point clouds.
   - The user noted that if these worked, you could get a splat in the time of a forward pass of a transformer (~seconds) as opposed to learning it from scratch with a point cloud init and with **2-4 mins training time**.
- **Pixelwise Gaussian Grid Methods Deemed Suboptimal**: A user commented that current methods with decent-quality NVS (Novel View Synthesis) yield suboptimal reconstructions w.r.t. efficiency, as they predict pixelwise **Gaussian grids**.
   - The user cited [Pixel-aligned Gaussian Splatting](https://arxiv.org/abs/2311.10647) which spawns a gaussian per pixel, leading to models that are **~200 MB** and that change poses in a non-affine way.
- **Sparse Voxel Splatting Touted for Speed and Sparsity**: A user mentioned that voxel splatting, such as [3D-GS: Real-Time Rendering of Multi-View Gaussian Splatting With Voxel Hashing](https://arxiv.org/abs/2309.19297), is very fast with **nvidia's sparse tensor library** and accounts for sparsity in your scene.
   - Another user recommended the **MVSplat** and **SPFSplat** series, and more recently, **E-RayZer**, but conceded that they are not gonna fix the size issues.
- **Pensieve's Recollections for Gradient Gains**: A user suggested considering [Recollections from Pensieve](https://link-to-pensieve) which trains a model with two renderers simultaneously (**LVSM + Gaussians**) and sees gains from that, at least in their self-supervised setting.
   - They reasoned that **LVSM** likely provides more useful gradients than **NVS reconstruction losses on Gaussians** and announced a forthcoming preprint with decently large-scale trained model for potential building upon.
- **OverWorld Repos Spark World Model Interest**: A user asked for small-scaled repos/models like **nanoVLM**, **nanoGPT**, or **smolVLM** for quick hands-on learning about world models.
   - Another user suggested checking out the **OverWorld Repos**, noting that it's under active development.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1467526426222788628)** (2 messages): 

> `DeepSpeed Universal Checkpointing, Continued Training` 


- **DeepSpeed Universal Checkpointing Support Requested**: A member inquired about plans to bring support for **DeepSpeed Universal Checkpointing**, noting that an open pull request may now be outdated.
   - They highlighted that this feature would be valuable, as currently, continued training from a checkpoint requires an identical network topology.
- **Roadmap Inquiry for Future Library Features**: A member asked if there is a roadmap for future features planned for the library.
   - No additional information was provided.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1466924630903226645)** (6 messages): 

> `Recursive Language Models (RLMs), Codebase Auditing, Neosantara's PAYG Billing` 


- ****RLMs** are here to Audit Codebases**: A member shared a post on using **Recursive Language Models (RLMs)** to audit a codebase, inspired by a gist on codebase documentation, shared at [kmad.ai](https://kmad.ai/Recursive-Language-Models-Security-Audit).
- **Audit a codebase for pennies, fast**: The Kimi k2's ability at **RLM** is impressive given its speed and cost and the traces are super cool to watch.
   - Members are waiting for **Groq/Cerebras** to host it.
- **Neosantara Launches PAYG Billing**: **Neosantara** is rolling out **PAYG billing** and can’t wait to see what you’ll build with it.
   - Users can get started by trying the [examples repo](https://github.com/neosantara-xyz/examples/tree/main/dspy) and explore how to integrate **Neosantara** with **DSPy** in minutes; see [billing details](https://docs.neosantara.xyz/en/about/billing-pricing).


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1467902184363528369)** (1 messages): 

> `Agent Systems, Scaling Laws for Agents` 


- **Google Explores Scaling Laws for Agent Systems**: Google published a blog post titled '[Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)' exploring the conditions under which agent systems effectively scale.
- **Scaling Agent Systems**: The blog post discusses how to effectively scale agent systems, focusing on when and why they work.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1466931718647844897)** (102 messages🔥🔥): 

> `Hierarchical classification with GEPA, Feedback improvement for Reflection, RLMs with Tool Calling, Deno vs Python for Tool Calling, DSPy documentation` 


- **GEPA Struggles with Hierarchical Classification**: A member reported struggling with a **hierarchical classification task** using **GEPA** with a **hF1 metric**, achieving only **30-50%** performance despite various approaches.
   - They tried recursive exploration, web search augmentation, and a simple non-recursive approach, but performance remained suboptimal, suggesting that *GEPA isn't a magic wand*.
- **Feedback Loops Needs Better Signals**: A member suggested that the current feedback mechanism for reflection models doesn't provide enough information for effective learning.
   - They emphasized the need for feedback to explain *what went wrong and why*, rather than just indicating the divergence between predicted and true paths, and suggest that **Selective Feedback** can improve results.
- **RLMs + Tool Calling: More Boilerplate and Deno Troubles**: Members are facing challenges and *ugly boilerplate* trying to implement **RLMs** with custom tool calling, particularly due to issues with the **Deno sandbox**.
   - They found that the current setup lacks conciseness and beauty compared to regular modules, and are struggling with permissions, as well as generating the right code to bypass issues with the local Deno sandbox.
- **Tool Calling needs Custom Python**: Members discussed running tool calls with **PythonInterpreter**, but noticed that the standard path used **dspy.Tool**, and there's need for more context on what the model needs to do.
   - As one person put it, *Deno is just f***ing terrible lol*, with general agreement that the experience of getting it to work is horrible, and a hope that newer versions allow simpler implementations of RLMs in DSPy.
- **DSPy Needs More Cookbook Examples**: A member pointed out the lack of documentation for **dspy/adapters/types/reasoning.py**, and emphasized that releasing code without docs is so 2023.
   - The response was that docs should help a human understand a thing, and AI-generated docs are rough for understanding, but that it is possible to feed in the RLM paper + the module and associated code to get decent docs.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1467048110923452517)** (13 messages🔥): 

> `Modular 26.1 Release, Community Meeting Feedback, Incorrect Announcement Link` 


- **Modular 26.1 Release Announcement Link Fixed!**: Users reported a broken link in the announcement for the **Modular 26.1 release** and another user quickly provided the [correct link](https://www.modular.com/blog/modular-26-1-a-big-step-towards-more-programmable-and-portable-ai-infrastructure).
   - A staff member apologized and confirmed the provided link, promising to investigate the issue as the original announcement link *did work* for them.
- **Caroline Back from Maternity Leave**: A community staff member announced her return from maternity leave and invited members to reconnect and share their projects and feedback via [a scheduled chat](https://scheduler.zoom.us/caroline-frasca-3akopl/modular-community-chat-).
   - Another member welcomed her back to the community.
- **Community Meeting Praised for Format**: A new member thanked the team for an enjoyable community meeting, praising the format of **mini-talks from contributors** and the appreciation shown to students and early-career folks.
   - A staff member encouraged the user to share more questions and also asked for suggestions for topics to highlight at future community meetings.
- **Eager compilation**: A user who wasn't able to ask during the meeting, opened a discussion about eager compilation, lowering pipeline kernel selection across GPUs, and extension points for custom ops. See the [forum post](https://forum.modular.com/t/max-26-1-eager-to-compile-contract-lowering-pipeline-kernel-selection-across-gpus-and-extension-points-for-custom-ops/2677?u=krxgu).


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1467937182517035165)** (2 messages): 

> `February Community Meeting, Community Meeting Questions` 


- **Modular Announces February Community Meeting**: Modular announced that a community meeting will start in approximately 20 minutes.
   - They posted a link to the [February Community Meeting forum post](https://forum.modular.com/t/february-community-meeting/2646) on their website.
- **Community Gathers Questions for Meeting**: Modular reminded members to fill out a form if they have any questions to be answered in the meeting.
   - A link to the [question submission form](https://docs.google.com/forms/d/e/1FAIpQLSfIQepfmLtBBSrp-p-m1oi4l_wlVXjjryvbFgRgRziFI3tgkw/viewform) was provided.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1466901624180965567)** (73 messages🔥🔥): 

> `Pytorch Float Conversion in Mojo, Cross Language Benchmarks, Mojo DType Bool SIMD Packing, MOJSON Library, Graphics APIs Bindings` 


- **Pytorch Float Conversion Ambiguity**: A user reported an issue in Mojo **26.1** with converting a Python float from a Pytorch tensor to a Mojo **Float64**, encountering an *“ambiguous call to '__init__'”* error that did not occur in version **25.6**.
- **Mojo's Cross Language Benchmark Initial Results**: A user shared a cross-language benchmark including Mojo, written by **Kimi K 2.5**, noting the code was not optimized and served as a baseline, sharing the [benchmark code](https://cdn.discordapp.com/attachments/1151418092052815884/1466984342063681648/mojo_vs_python.zip?ex=698206e2&is=6980b562&hm=0cf3f07e76df6ce360494469b348a949533e50fcea2315ec256cd04e1b80887a) and [benchmark report](https://cdn.discordapp.com/attachments/1151418092052815884/1466984341757366334/benchmark_report.pdf?ex=698206e2&is=6980b562&hm=bb28c3b6675ef1e03a633004428ab30a2d3d9d0102038c350d8175b753855349).
- **Tuning the Benchmark: TCMalloc and Int Size!**: Discussion arose regarding optimizations for a cross-language benchmark, including using `unordered_map` in **C++**, enabling `-march=native`, and noting that **C++** used **int32** matmuls while other languages used **int64**.
- **MoJson Library Impresses**: Members were impressed by [mojson](https://github.com/ehsanmok/mojson), a **JSON** library for Mojo, with one commenting that *this looks really impressive* and another noting now that String is **CoW** several of the choices they are seeing make more sense.
   - There was a discussion on [lazy parsing](https://github.com/modular/modular/blob/main/stdlib/JSON/JSON.mojo) and on use of StringSlice vs String due to concerns about allocations.
- **FFI Bindings vs Origins**: A discussion on **FFI** bindings highlighted a method to ensure that pointers returned from **C** functions are bound to the lifetime of the Mojo object that owns the underlying shared library handle.
   - The solution involves shadowing external function calls and using `unsafe_origin_cast` to cast the pointer to the origin of the `DLHandle`, and can be [seen in ash_dynamics](https://github.com/josiahls/ash_dynamics/blob/2c53095da70df95f3cb5758eddb2895f2a4bebca/ash_dynamics/ffmpeg/avcodec/__init__.mojo#L108).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1466890735411527943)** (54 messages🔥): 

> `AI Feed Social, Generative Modeling Event Measurability, Bureau of Rizz, Sharp Minima Finding, Moltbook in Latent Space` 


- ****AI Social Media Site Emerges****: A member shared a link to an AI-only social media site, [aifeed.social](https://aifeed.social/), questioning *"What the hell?"
   - Another member posted a related [tweet from 2017](https://x.com/i/status/2017305948696789466) with a similar concept.
- ****Measurability Ignorance is Bliss for Generative Models?****: A member inquired if, for generative modeling, they can ignore unmeasurable events described in Cedric Villani's 2008 book.
   - Another member clarified that μ(A)=0 doesn't mean an event is not measurable, it's just measured at size 0, and suggested focusing on *non-negligible* or *full measure* scenarios.
- ****Molten Latent Space!****: One member shared [a link](https://fxtwitter.com/i/status/2017442712388309406) about a *moltbook* in latent space.
   - Others found the navigation cool, but potentially not very useful, and suggested just a list of similar papers would be better.
- ****GANs & Generative Model Resources Abound****: A member asked for resources to study generative models from GANs to the latest advancements.
   - Another member recommended the [*Understanding Deep Learning* book](https://udlbook.github.io/udlbook/) by Simon J.D. Prince, Stanford and MIT courses, and Sebastian Raschka's books and shared links to [Stanford courses](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8), [MIT](https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH), and [Raschka's books](https://sebastianraschka.com/books/).
- ****Forecasting the Future with Timeseries Models****: In response to a question about models for timestamped tabular data, a member suggested that the choice of model depends on the definition of *timeseries.*
   - Another member recommended [sktime](https://www.sktime.net/en/latest/index.html) to analyze a wide variety of model types, as well as boosting variants or TBATS depending on the specific needs.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1466914903616131276)** (11 messages🔥): 

> `Discord History Mining, Paper Discussion Voice Calls, Computer Vision Newsletters` 


- **Discord History Excavation**: A member asked **Claude** to write a script to dig through the Discord history via HTTP API and find all the paper discussion announcements, taking only **15 minutes** from idea to results.
   - The script easily found **243 announcements**, but the member thinks there are around **100 more** from other users.
- **Paper Discussion Voice Call Announcements**: After revisions, a member's script found **392 messages** containing paper links that occurred in messages with the group at-mention, with ~98% of them being announcements for paper discussion voice calls.
   - A [full list](https://gist.github.com/k-nearest-neighbor/6d9a34f54fc17a0ed84c0b0df7b4d809) was shared, though the member noted that there were more announcements prior to where the list stops.
- **Quest for Computer Vision Newsletter**: A member inquired about the existence of a newsletter similar to [this one](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-e94), but focused on computer vision.
   - No specific computer vision newsletters were recommended in the messages.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 messages): 

artale39: https://lucumr.pocoo.org/2026/1/31/pi/
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1467277525494268115)** (4 messages): 

> `Grok, Twitter Links` 


- **X-Links surface in Discord**: Members shared [various links from X](https://fxtwitter.com/i/status/2018164753810764061) without additional context, providing possible resources or points of interest.
   - This might have been related to a particular topic of discussion that was not explicitly mentioned in the chat log.
- **Grok-Slop Overflow**: A member derisively mentioned *more Grok-Slop*, indicating a negative sentiment towards the quality or relevance of content related to **Grok**.
   - They also linked to a [discussion on Hacker News](https://news.ycombinator.com/item?id=46835895), possibly as a counterpoint or example of a more worthwhile discussion.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1466961002665873621)** (50 messages🔥): 

> `Llama 1B optimization, Torch comparison, Bounty progress, Superkernels, DTLS connection issues` 


- **Llama 1B CPU bounty in progress**: A member is working on the Llama 1B CPU optimization bounty, aiming for faster performance than Torch, using `LlamaForCausalLM` with TorchInductor, currently reporting **0.99x faster** in CI but rewriting for clarity.
   - Another member reached **7.5 tok/s** after addressing correctness bugs encountered while pursuing **9 tok/s**.
- **Correctness bugs are slowing optimization**: One member reported finding correctness bugs, and losing progress after previously reaching **9 tok/s**, and reset a lot of progress to achieve stability.
   - Another member said *the dream is always to fix bugs by deleting code*.
- **Seeking workflow tips for kernel optimization**: A member requested workflow tips, currently profiling slow kernels, examining Metal code, and introducing fixes, while comparing with **llama.cpp** which achieves **~30 tok/s** with Metal code.
   - A good heuristic suggested would be **~80% MBU on decode**, so just look at the number of bytes in the active params and the achievable bandwidth to get the minimum tpot / maximum tps and take 80% of that.
- **tinygrad test failing due to RANGE object sharing**: A member identified a bug related to two `REDUCE`s in a fused kernel sharing the same `RANGE` object, caused by `remove_bufferize`, leading to an assertion failure in `CFGContext`.
   - The suggested fix involves either preventing range sharing or handling shared ranges downstream, though skipping `remove_bufferize` when there's a `REDUCE` inside was proposed as a simpler solution.
- **Plans for blackwell box with high VRAM?**: Someone asked if there are plans to ship a **blackwell** style box with more than **500 gb VRAM**.
   - George pointed to a good first issue: [https://github.com/tinygrad/tinygrad/pull/14490](https://github.com/tinygrad/tinygrad/pull/14490).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

ennis3444: is there a way to make gemm kernels use shared memory using the opencl renderer?
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1467155759707193374)** (10 messages🔥): 

> `Manus Context, AI Brain Reading Headphones, Neurable, Failure Modes` 


- **Context-Aware Manus Request Sparked**: A member requested that **Manus** should have **context from other chats**, calling it a *game changer*.
   - They linked to a [YouTube video](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) as a reference.
- **AI Brain-Reading Headphones Demoed**: A member shared a link to a **YouTube video** showcasing **AI brain-reading headphones**.
   - The same [YouTube link](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) was shared by another member, followed by just the question *AI brain reading headphones?* by another member.
- **"Neurable" Tech Mentioned**: A member mentioned **Neurable** in relation to the **AI brain-reading headphones** technology.
   - A member stated these **AI brain-reading headphones** have been around *since like 2013* and they saw a *Matthew Santoro video when I was in elementary school*.
- **AI/ML Engineer Highlights Observability Focus**: An AI/ML Engineer shared their current focus on innovating AI with impact, specifying *Autonomous Agents*, *Healthcare AI*, *Conversational AI*, and *Fraud Detection*.
   - They highlighted their work focus on **failure modes**, **observability**, and **keeping AI systems stable under real usage** rather than demos, offering to compare notes or help unblock issues.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1467298012962488485)** (7 messages): 

> `Aider as a library, Netflix culture` 


- ****Aider** Considered for Library Use**: A member expressed interest in developing **Aider** into a library for software use, highlighting its potential for creating file editing agents.
   - The member noted some kinks need resolution to enhance its power for that use case, especially with editing markdown files containing code blocks due to **Aider**'s parsing fences.
- **Netflix Culture Curiosity**: A member inquired about connecting with someone working at **Netflix** to discuss its culture.
   - Other members suggested checking **Glassdoor** or **LinkedIn** as resources to find and connect with **Netflix** employees.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1466895814617202828)** (3 messages): 

> `Arena Mode Launch, Plan Mode Release, Windsurf Credits, Leaderboards in Arena Mode, Windsurf Maintenance` 


- **Windsurf Launches Arena Mode with 0x Credits**: Windsurf launched **Wave 14** featuring **Arena Mode**, allowing users to compare AI models side-by-side and vote on the better response, with [Battle Groups mode](https://windsurf.com/download/editor) costing **0x credits** for the next week.
   - Arena Mode includes **Battle Groups** (random models) and **Pick your own** (choosing up to five models), feeding into personal and public leaderboards.
- **Plan Mode added to Windsurf**: Windsurf has added **Plan Mode**, accessible via the Cascade toggle, alongside Code and Ask Modes.
   - Users can switch between modes to better manage and organize their workflows within the Windsurf environment.
- **Windsurf undergoing Maintenance**: Windsurf experienced maintenance, which took longer than expected, but the service is now back online; users can follow the [status here](https://status.windsurf.com/).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1467296100984815779)** (2 messages): 

> `AI Challenge, SparkCraft AI Consulting, AI Scholars AI Engineering Bootcamp, Nanny Spark` 


- **AI Challenge Aims to Build AI Matchmaking Pipeline for Nanny Recruitment**: A member announced a real-client **AI Challenge** in collaboration with **SparkCraft AI Consulting**, **AI Scholars AI Engineering Bootcamp**, and **Nanny Spark** to build an **AI matchmaking pipeline** for a nanny recruitment service.
   - The goal is to create solutions for data collection, AI-powered matching, interview transcript analysis, and delivery workflows, with potential **production deployment from day one**.
- **AI Challenge Awards AI Bootcamp Seats and Recommendations**: The **top 3** participants in the **AI Challenge** will receive **1 seat** in the **AI Scholars 4-week AI Engineering Bootcamp** and a recommendation from **Nanny Spark’s founder**.
   - Key dates include a kickoff info session on **Sunday at 8 PM EST** ([https://luma.com/iq1u2sur](https://luma.com/iq1u2sur)), a submission deadline on **Wednesday at 3 AM EST**, and review sessions on **Wednesday at 5 PM & 8 PM EST** ([https://luma.com/gexiv0x0](https://luma.com/gexiv0x0)).


  

---


---


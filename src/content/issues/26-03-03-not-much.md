---
id: MjAyNi0w
title: not much happened today
date: '2026-03-03T05:44:39.731046Z'
description: >-
  **Google DeepMind** launched **Gemini 3.1 Flash-Lite**, emphasizing *dynamic
  thinking levels* for adjustable compute, with notable metrics like **$0.25/M
  input**, **$1.50/M output**, **1432 Elo on LMArena**, and **2.5× faster
  time-to-first-token** than Gemini 2.5 Flash. It supports a **1M context
  window** and high throughput for multimodal inputs including text, images,
  video, audio, and PDFs. **OpenAI** rolled out **GPT-5.3 Instant** to all
  ChatGPT users, improving conversational naturalness and reducing
  hallucinations by **26.8% with search**. The upcoming **GPT-5.4** was teased
  amid speculation. **Alibaba's Qwen** faces leadership exits, raising concerns
  about its future and open-source status. The news highlights advancements in
  model efficiency, pricing, and multimodality, alongside organizational changes
  impacting AI development.
companies:
  - google-deepmind
  - google
  - openai
  - alibaba
models:
  - gemini-3.1-flash-lite
  - gemini-3
  - gpt-5.3
  - gpt-5.4
  - qwen
topics:
  - multimodality
  - latency
  - throughput
  - context-window
  - model-pricing
  - model-benchmarking
  - model-performance
  - conversational-ai
  - hallucination-reduction
  - api
  - model-rollout
  - leadership-exit
people:
  - jeffdean
  - noamshazeer
  - sundarpichai
  - aidan_mclau
  - justinlin610
---


**a quiet day**

> AI News for 3/2/2026-3/3/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**264** channels, and **12765** messages) for you. Estimated reading time saved (at 200wpm): **1137** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Gemini 3.1 Flash‑Lite launch: “dynamic thinking levels” + aggressive price/perf**

- **Gemini 3.1 Flash‑Lite (Preview)** shipped as Google’s fastest, most cost-efficient Gemini 3-series endpoint, emphasizing *latency* and *throughput* for high-volume workloads. DeepMind’s launch thread positions it as “intelligence at scale” with adjustable **thinking levels** (dial compute based on task complexity) [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2028872381477929185), with API rollout via AI Studio / Vertex [@Google](https://x.com/Google/status/2028872509601333594). Jeff Dean highlighted **$0.25/M input** and **$1.50/M output**, **1432 Elo on LMArena**, and **86.9% GPQA Diamond** alongside **2.5× faster time-to-first-token** than Gemini 2.5 Flash [@JeffDean](https://x.com/JeffDean/status/2028876962580816143); Noam Shazeer echoed the “thinking levels” framing as a product knob for “maximum intelligence, minimal latency” [@NoamShazeer](https://x.com/NoamShazeer/status/2028909105969283565); Sundar Pichai amplified the same speed/cost message [@sundarpichai](https://x.com/sundarpichai/status/2028891212573491715).
- **Third-party benchmarking/positioning**: Artificial Analysis reports Flash‑Lite retains a **1M context** window, measures **>360 output tokens/s** and ~**5.1s** average answer latency, improves their “Intelligence Index” vs 2.5 Flash‑Lite, but **pricing increased** (blended cost up materially) [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2028882198456352852). Arena notes Flash‑Lite Preview ranking #36 in Text Arena (1432) and tied around #35 in Code Arena, framed as a strong point on the cost-performance frontier [@arena](https://x.com/arena/status/2028876727989289449). A recurring community reaction is “Flash‑Lite… very funny Google” due to naming plus rapid cadence [@JasonBotterill](https://x.com/JasonBotterill/status/2028794893624291569), and “Google launches models faster than I can finish testing” [@matvelloso](https://x.com/matvelloso/status/2028901252437032982).
- **Multimodal angle**: Google staff pushed “use Flash‑Lite instead of writing parsers” for text+images+video+audio+PDF ingestion [@koraykv](https://x.com/koraykv/status/2028876507679191392), reinforcing Flash‑Lite as a *plumbing model* for production workflows.

**OpenAI: GPT‑5.3 Instant rollout + “less preachy” + teased GPT‑5.4**

- **GPT‑5.3 Instant** rolled out to all ChatGPT users, explicitly responding to complaints that 5.2 was “too cautious” with “too many caveats.” OpenAI claims improved conversational naturalness, fewer unnecessary refusals/defensive disclaimers, and better search-integrated answers [@OpenAI](https://x.com/OpenAI/status/2028893701427302559), [@nickaturley](https://x.com/nickaturley/status/2028894581191000404). OpenAI also states reduced hallucinations: **26.8% better with search** and **19.7% without** per an internal contributor [@aidan_mclau](https://x.com/aidan_mclau/status/2028894122959159434) and echoed by staff [@christinahkim](https://x.com/christinahkim/status/2028900228196384978).
- **API/Arena exposure**: “GPT‑5.3‑chat‑latest” appears in the API per community reporting [@scaling01](https://x.com/scaling01/status/2028906108291616773) and is available for side-by-side evals in Text Arena [@arena](https://x.com/arena/status/2028908848204177682).
- **GPT‑5.4 teased** with a high-engagement “sooner than you Think” post [@OpenAI](https://x.com/OpenAI/status/2028909019977703752), prompting confusion about sequencing vs “5.3 Thinking and Pro will follow soon” chatter [@kimmonismus](https://x.com/kimmonismus/status/2028924631084605465). Multiple tweets speculate 5.4 is also being used as a *news-cycle deflection* amid DoD/NSA contract controversy [@kimmonismus](https://x.com/kimmonismus/status/2028803185347875207).

**Alibaba Qwen shock: leadership exits, “Qwen is nothing without its people,” and open-source uncertainty**

- **Key departures**: A major thread across the dataset is the exit of Qwen’s tech leadership and senior contributors. Justin Lin’s “stepping down” post triggered widespread reaction [@JustinLin610](https://x.com/JustinLin610/status/2028865835373359513), followed by high-signal confirmations/tributes and then more exits including another leader (“bye qwen, me too”) [@huybery](https://x.com/huybery/status/2028976346416988612) and a separate sign-off [@kxli_2000](https://x.com/kxli_2000/status/2028880971945394553). External observers describe this as Alibaba Cloud “kicking out” Qwen’s tech lead [@YouJiacheng](https://x.com/YouJiacheng/status/2028880908305219729).
- **Why it matters technically**: Many engineers view Qwen as *critical infrastructure* for the open model ecosystem—especially **<10B** and “Pareto frontier” models, plus VLM/OCR derivatives. This is framed as a genuine ecosystem risk if open-weights cadence slows or licensing stance shifts [@natolambert](https://x.com/natolambert/status/2028893211759124890), [@teortaxesTex](https://x.com/teortaxesTex/status/2028874511509000646), [@awnihannun](https://x.com/awnihannun/status/2028902061384057211). There’s also immediate speculation on whether Qwen’s OSS posture changes given “popular open models wasn’t enough” [@code_star](https://x.com/code_star/status/2028913595602616391).
- **Organizational diagnosis**: A recurring interpretation is that “unification” under a higher-level Alibaba structure (reporting to CEO) created political pressure around influence/visibility [@Xinyu2ML](https://x.com/Xinyu2ML/status/2028891170592473385), with broader commentary about big-tech hierarchies punishing “bridges” who build external trust [@hxiao](https://x.com/hxiao/status/2028932213228900701).
- **Despite the turmoil, shipping continues**: Qwen 3.5 LoRA fine-tuning guides and low-VRAM training recipes spread quickly (notably Unsloth) [@UnslothAI](https://x.com/UnslothAI/status/2028845314506150079), and **GPTQ Int4** weights with vLLM/SGLang support were promoted [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2028846103257616477). Community also pushed education/reimplementations around Qwen3.5 [@rasbt](https://x.com/rasbt/status/2028961822372425941). The tension is: *strong release velocity* paired with *leadership flight*.

**Long-context + training efficiency: making “impossible” context windows practical**

- **87% attention-memory reduction for long-context training**: A Together paper highlighted a hybrid of **Context Parallelism** plus **Sequence Parallel-style head chunking**, claiming training a **5M context window 8B model on 8×H100** (single node) and cutting attention memory footprint by up to **87%** [@rronak_](https://x.com/rronak_/status/2028718679123497007). The tweet also calls out a practical gap: most RL post-training for long-context frontier models is still done on only a fraction of the full context due to memory cost.
- **FlashOptim (Databricks)**: Open-source optimizer implementations (AdamW/SGD/Lion) that preserve update equivalence while cutting memory—tweet thread announces `pip install flashoptim` [@davisblalock](https://x.com/davisblalock/status/2028943987349045610), and MosaicAI summarizes **>50% training memory reduction**, e.g., bringing AdamW training overhead from ~**16 bytes/param** down to **7 bytes** (or **5** with gradient release) and reducing an example 8B finetune peak from **175 GiB → 113 GiB** [@DbrxMosaicAI](https://x.com/DbrxMosaicAI/status/2028977216940589383).
- **Heterogeneous infra for RL**: SkyPilot argues RL post-training should split workloads across **beefy GPUs (trainer)**, **cheap GPUs (rollouts)**, and **high-memory CPUs (replay buffers)**; Job Groups provides a single YAML orchestration model with coordinated lifecycle and service discovery [@skypilot_org](https://x.com/skypilot_org/status/2028878888211013907).
- **Kernel/toolchain gotchas**: A CuTeDSL + torch.compile regression report notes ~**2.5× slowdown** for wrapped kernels (including RMSNorm “Quack” kernels) when made compile-compatible via custom ops—highlighting friction between kernel-level speed and graph compilation requirements [@maharshii](https://x.com/maharshii/status/2028863745641112008).

**Agent engineering reality check: benchmarks vs “real work,” consensus failures, and tooling shifts (MCP, sandboxes, observability)**

- **Benchmarks don’t match labor economics**: A new database attempts to map agent benchmarks to real-world work distribution, arguing current evaluations overweight math/coding despite most labor/capital being elsewhere [@ZhiruoW](https://x.com/ZhiruoW/status/2028847081507488011). This point was boosted as “central problem of AI benchmarking for real work” [@emollick](https://x.com/emollick/status/2028870529906622677). Arena’s **Document Arena** launch is a direct response: real PDF reasoning side-by-side evals; Claude Opus 4.6 leads (per Arena) [@arena](https://x.com/arena/status/2028915403704156581).
- **Multi-agent coordination is fragile**: Byzantine consensus games show LLM agent agreement is unreliable even when benign; failures often come from **stalls/timeouts** more than adversarial corruption, worsening with group size [@omarsar0](https://x.com/omarsar0/status/2028823724196343923). Complementary work on Theory of Mind + BDI + symbolic verification suggests cognitive “ToM modules” don’t automatically help; gains depend strongly on base model capability [@omarsar0](https://x.com/omarsar0/status/2028913061260935331).
- **MCP “dead?” vs MCP expanding**: There’s an explicit “MCP is dead?” prompt from DAIR’s Omar [@omarsar0](https://x.com/omarsar0/status/2028840977922674842), but in the same dataset MCP adoption expands: Notion ships MCP/API support for Meeting Notes (one-liner install via Claude Code) [@zachtratar](https://x.com/zachtratar/status/2028881783551570209); Cursor ships **MCP Apps** where agents render interactive UIs inside chat [@cursor_ai](https://x.com/cursor_ai/status/2028953584407085546).
- **“Kill code review” debate**: swyx frames removing human code review as a “Final Boss” of agentic engineering and SDLC inversion [@swyx](https://x.com/swyx/status/2028795270306079156). Counterpoint: thdxr argues that teams “producing this much code” via LLMs may be using them incorrectly; large code volumes create self-defeating codebases and LLMs themselves struggle with the resulting complexity [@thdxr](https://x.com/thdxr/status/2028827251534352764).
- **Sandboxed “computer use” platforms**: Perplexity’s “Computer” draws heavy engagement: Srinivas solicits feature requests [@AravSrinivas](https://x.com/AravSrinivas/status/2028742933403574585), and Perplexity positions its product as orchestrating many models and embedding directly into apps with a managed secure sandbox (no API key management) [@AravSrinivas](https://x.com/AravSrinivas/status/2028903680616087946), [@AskPerplexity](https://x.com/AskPerplexity/status/2028893546447814895). Cursor’s cloud agents similarly run in isolated VMs and output merge-ready PRs with artifacts [@dl_weekly](https://x.com/dl_weekly/status/2028844128729973060).

**Talent, governance, and trust: Anthropic vs DoD, OpenAI contract scrutiny, and high-profile moves**

- **Max Schwarzer (VP Post-Training at OpenAI) → Anthropic**: A major personnel move: Schwarzer announced leaving OpenAI after leading post-training and shipping GPT‑5/5.1/5.2/5.3-Codex, joining Anthropic to return to IC RL research [@max_a_schwarzer](https://x.com/max_a_schwarzer/status/2028939154944585989). This fueled narratives of “big win for Anthropic” [@kimmonismus](https://x.com/kimmonismus/status/2028952074063331421) and broader “legends dropping out” anxiety [@yacinelearning](https://x.com/yacinelearning/status/2028880802797199476).
- **Anthropic vs Pentagon/Palantir tension**: Reporting claims DoD threatened to label Anthropic a “supply chain risk,” potentially impacting Palantir’s usage for federal work; Anthropic wants safeguards (mass domestic surveillance + autonomous weapons) [@srimuppidi](https://x.com/srimuppidi/status/2028943303581024412), with additional coverage pointers [@aaronpholmes](https://x.com/aaronpholmes/status/2028942999548297464).
- **OpenAI–DoD / NSA trust crisis**: Multiple tweets demand actual contract language, arguing “incidental” surveillance wording historically enabled warrantless domestic surveillance; critics cite PRISM/Upstream and FISA/EO 12333 context [@jeremyphoward](https://x.com/jeremyphoward/status/2028805970214912125), and call for independent legal red-teaming rather than “trust us” assurances [@sjgadler](https://x.com/sjgadler/status/2028899096283758732). This is repeatedly linked to the hypothesis that OpenAI will use model launches to steer the narrative.
- **Market-share claims**: One viral claim states Claude surged from minority share to dominating US business market share vs ChatGPT within a year [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2028974344710606905). Treat this as directional unless you can validate the underlying dataset, but it reflects perceived momentum: “coding + agents paid off.”

---

### Top tweets (by engagement, tech-focused)

- **GPT‑5.4 teaser**: “5.4 sooner than you Think.” [@OpenAI](https://x.com/OpenAI/status/2028909019977703752)  
- **Gemini 3.1 Flash‑Lite launch thread** [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2028872381477929185)  
- **GPT‑5.3 Instant rollout + “less preachy”** [@OpenAI](https://x.com/OpenAI/status/2028893701427302559)  
- **Qwen leadership departure (“stepping down”)** [@JustinLin610](https://x.com/JustinLin610/status/2028865835373359513) and follow-on sign-offs [@huybery](https://x.com/huybery/status/2028976346416988612)  
- **Unsloth: Qwen3.5 LoRA with ~5GB VRAM claim + notebook** [@UnslothAI](https://x.com/UnslothAI/status/2028845314506150079)  
- **Cursor: MCP Apps (interactive UIs inside agent chat)** [@cursor_ai](https://x.com/cursor_ai/status/2028953584407085546)  
- **Together long-context training memory reduction (up to 87%)** [@rronak_](https://x.com/rronak_/status/2028718679123497007)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen 3.5 Model Releases and Benchmarks

  - **[Qwen 2.5 -&gt; 3 -&gt; 3.5, smallest models. Incredible improvement over the generations.](https://www.reddit.com/r/LocalLLaMA/comments/1rjd4pv/qwen_25_3_35_smallest_models_incredible/)** (Activity: 1017): ****Qwen 3.5** is a notable advancement in the Qwen model series, featuring a `0.8B` parameter model that includes a vision encoder, suggesting the language model component is even smaller. This model is part of a trend towards smaller, more efficient models, such as the current smaller MoE (Mixture of Experts) models, which are praised for their performance. Despite its size, Qwen 3.5 has been criticized for factual inaccuracies, such as incorrect information about aircraft engines, highlighting the need for rigorous fact-checking.** Commenters highlight the potential of smaller models like Qwen 3.5 to enable personal assistants on local machines, emphasizing their efficiency and accessibility for users with limited GPU resources. However, there is concern over the model's tendency to hallucinate facts, which could undermine its reliability.

    - The smaller Qwen models, particularly the MoE (Mixture of Experts) models, are noted for their impressive performance improvements over previous generations. These models are becoming increasingly viable for personal use on local machines, offering significant advancements in efficiency and capability, even at smaller scales.
    - A user highlights the hallucination issues in Qwen 3.5, pointing out specific factual inaccuracies related to aircraft engine types and configurations. This underscores the importance of fact-checking outputs from AI models, as they can confidently present incorrect information.
    - The performance of smaller quantized models, such as the 4B model, is praised for its efficiency on less powerful hardware. A user reports achieving 60 tokens per second with 128k context using `llama.cpp`, which is considered a significant improvement over older, larger models. This demonstrates the potential for high-performance AI on local, resource-constrained environments.

  - **[Visualizing All Qwen 3.5 vs Qwen 3 Benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1rivckt/visualizing_all_qwen_35_vs_qwen_3_benchmarks/)** (Activity: 736): **The image is a bar chart that visualizes the performance comparison between the new **Qwen 3.5 models** and the older **Qwen 3 models** across various benchmarks such as Knowledge & STEM, Instruction Following, Long Context, Math, Coding, General Agent, and Multilingualism. The chart uses different colors to distinguish between the model versions, with **Purple/Blue/Cyan** representing the new Qwen 3.5 models and **Orange/Yellow** for the older Qwen 3 models. The chart aims to provide a quick visual comparison of the models' performance, although some data is missing for smaller models. The raw data used for this visualization is available in a [Google Sheet](https://docs.google.com/spreadsheets/d/1A5jmS7rDJe114qhRXo8CLEB3csKaFnNKsUdeCkbx_gM/edit?usp=sharing).** Some commenters criticize the chart's clarity and usefulness, with one expressing disbelief at the benchmark results, suggesting skepticism about the accuracy of the performance claims, particularly regarding the superiority of Qwen 3.5 models over Qwen 3 models in every test.

    - The benchmark results show that the Qwen 3.5 models, particularly the 9B dense model, are performing exceptionally well compared to larger models like the Qwen 3 122B A10B. This is surprising given the size difference, as the 9B model is more than 10 times smaller yet competes closely in various categories such as Knowledge & STEM, Instruction Following, and Multilingualism.
    - There is skepticism about the validity of the benchmarks, as one commenter finds it hard to believe that the Qwen 3.5 35B A3B model outperforms the Qwen 3 235B A22B model across all tests. This raises questions about the reliability of these benchmarks and whether they accurately reflect the models' capabilities.
    - The detailed benchmark table provided by a commenter highlights specific performance metrics across different categories for various Qwen models. For instance, the Qwen 3.5-122B-A10B model scores higher in Instruction Following and Math compared to its predecessors, indicating improvements in these areas. However, the presentation of the data has been criticized for being difficult to interpret.

  - **[Running Qwen 3.5 0.8B locally in the browser on WebGPU w/ Transformers.js](https://www.reddit.com/r/LocalLLaMA/comments/1rizodv/running_qwen_35_08b_locally_in_the_browser_on/)** (Activity: 501): ****Qwen 3.5 Small** models, including a `0.8B` parameter variant, have been released for on-device applications, with a demo running locally in the browser using **WebGPU** and **Transformers.js**. The implementation highlights the capability to run such models in-browser, though the **vision encoder** is identified as a performance bottleneck. The models are available on [Hugging Face](https://huggingface.co/collections/Qwen/qwen35), and a demo can be accessed [here](https://huggingface.co/spaces/webml-community/Qwen3.5-0.8B-WebGPU).** A comment suggests using `q4 GGUF` via `llama.cpp WASM` for better throughput without VRAM issues, indicating a preference for alternative methods to optimize performance. Another comment clarifies that the demo does not process video input, but rather static screenshots.

    - The vision encoder in WebGPU is identified as a bottleneck, with a suggestion to use `q4 GGUF` via `llama.cpp WASM` for improved throughput. This approach can run in the browser without causing VRAM thrashing, which is a common issue with WebGPU implementations.
    - A clarification is made regarding input types: the model does not process video input but rather takes a screenshot of the current screen at the moment the prompt is sent. This distinction is crucial for understanding the model's input handling capabilities.
    - There is a technical issue reported where the 'start' button is unresponsive, preventing users from initiating the process. This could indicate a bug in the user interface or a problem with the initialization sequence of the application.


### 2. Qwen 3.5 Model Performance and Applications

  - **[Unsloth fixed version of Qwen3.5-35B-A3B is incredible at research tasks.](https://www.reddit.com/r/LocalLLaMA/comments/1rjh5wg/unsloth_fixed_version_of_qwen3535ba3b_is/)** (Activity: 417): **The updated version of **Qwen3.5-35B-A3B** by **Unsloth** has shown significant improvements in handling research tasks, particularly after fixing tool calling issues. The model, which features `35 billion parameters` and utilizes hybrid linear attention, allows for a doubled native context length without increasing memory footprint. It was tested on a **Ryzen AI Max+ 395 system** using `llama.cpp-rocm` with parameters such as `--ctx-size 262144` and `--n-gpu-layers 999`, achieving prompt processing speeds of `600+ tokens/second` and token generation speeds of `25-30 tokens/second`. The model effectively performed `14 web searches` and `4 full page fetches`, maintaining a balance in tool usage, which was a noted improvement over previous models like **GLM-4.7-Flash**. The model's performance in providing remote desktop solutions for a Linux Fedora 43 system was comparable to frontier models, though it was noted that it could have recommended **Sunshine+Moonlight** more strongly.** One commenter noted that **RustDesk** is a superior remote desktop solution, especially for setups similar to the one described, despite the original post's focus on KRdp and other options. Another comment mentioned a potential issue with **LM Studio** not parsing `{{CURRENT_DATE}}` in system prompts, indicating a need for a fix.

Error summarizing comments.

  - **[Qwen 3.5 27b: a testament to the transformer architecture](https://www.reddit.com/r/LocalLLaMA/comments/1rj6m71/qwen_35_27b_a_testament_to_the_transformer/)** (Activity: 557): ****Qwen 3.5 27b** demonstrates significant advancements in transformer architecture, achieving reasoning and knowledge test performance comparable to **R1 0528**. Notably, it employs a hybrid architecture where `75%` of the layers utilize **Gated DeltaNet linear attention** instead of a full transformer setup. This model's ability to perform at such a high level with only `27b` parameters, fitting on a single consumer GPU, marks a substantial leap from previous models that required `70b` parameters and cluster-level compute for similar tasks. The model is also noted for its potential in fine-tuning, particularly in coding applications, due to its strong foundational capabilities.** Commenters highlight the model's improved instruction-following capabilities and the potential for fine-tuning to enhance its personality. The use of Gated DeltaNet linear attention is seen as a significant architectural innovation, contributing to its efficiency and performance.

    - victory_and_death highlights that Qwen 3.5 27b does not fully utilize the traditional transformer architecture. Instead, it employs Gated DeltaNet linear attention for 75% of its layers, which is a significant deviation from the standard transformer model. This architectural choice likely contributes to its performance efficiency and ability to run on consumer-grade hardware.
    - Pitiful-Impression70 points out the impressive performance of the Qwen 3.5 27b model, noting that it competes with larger models like R1 0528. The fact that a 27 billion parameter dense model can perform tasks that previously required 70 billion parameter models is remarkable, especially since it can run on a single consumer GPU. This highlights the rapid advancements in model efficiency and capability.
    - National_Meeting_749 discusses the improved instruction-following capabilities of newer models like Qwen 3.5 27b. These models can incorporate system prompts to inject personality, enhancing their interaction quality. This improvement in handling instructions is a significant step forward compared to previous generations of models.

  - **[Running Qwen3.5-0.8B on my 7-year-old Samsung S10E](https://www.reddit.com/r/LocalLLaMA/comments/1rj5ngc/running_qwen3508b_on_my_7yearold_samsung_s10e/)** (Activity: 330): **The image demonstrates the successful execution of the Qwen3.5-0.8B model on a Samsung S10E using `llama.cpp`, a tool for running large language models on local devices. The model achieves a processing speed of `12 tokens per second`, which is notable given the phone's age and hardware limitations. This showcases the potential for running sophisticated AI models on older hardware, leveraging optimizations like the NEON SIMD path in `llama.cpp` to enhance performance on ARM chips. The model's ability to hold a coherent conversation and perform complex tasks highlights significant advancements in AI efficiency and accessibility.** Commenters are impressed by the performance, noting that a year ago, such a model's conversational capabilities on a device of this age would have been unexpected. There's also technical curiosity about the installation process of `llama.cpp` and the specific quantization used (Q4_0 or Q8).

    - sean_hash highlights the performance of running Qwen3.5-0.8B on a Snapdragon 855, achieving `12 tokens per second`. This is considered impressive for older ARM chips, thanks to the NEON SIMD path in `llama.cpp`, which optimizes performance significantly on such hardware.
    - rm-rf-rm inquires about the installation process of `llama.cpp`, suggesting interest in replicating the setup. This indicates a technical curiosity about the implementation details and the potential challenges of running large language models on older devices.
    - WPBaka questions the practical applications of a 0.8B model, expressing skepticism about its capabilities beyond basic conversation. This reflects a broader debate on the utility of smaller models in real-world scenarios, especially when compared to larger, more powerful models.


### 3. Apple M5 Pro and M5 Max Launch

  - **[Apple unveils M5 Pro and M5 Max, citing up to 4× faster LLM prompt processing than M4 Pro and M4 Max](https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple_unveils_m5_pro_and_m5_max_citing_up_to_4/)** (Activity: 822): **Apple has announced the M5 Pro and M5 Max chips, which claim to offer up to 4× faster processing for large language model (LLM) prompts compared to their predecessors, the M4 Pro and M4 Max. The M5 Pro supports up to 64GB of unified memory with a bandwidth of 307GB/s, while the M5 Max supports up to 128GB of unified memory with a bandwidth of 614GB/s. Additionally, these chips feature up to 2× faster SSD speeds at 14.5GB/s and include the Apple N1 wireless chip for Wi-Fi 7, enhancing download speeds if compatible with the user's router. The image associated with the announcement highlights the chips' capabilities in handling complex tasks such as 3D modeling and programming efficiently.** Some users express disappointment over the lack of more advanced AI-specific silicon, such as a Neural Accelerator, in the new chips. Others are excited about the potential for these chips in future Mac Studio models.

    - The M5 Pro and M5 Max chips feature significant improvements in memory and bandwidth capabilities. The M5 Pro supports up to 64GB of unified memory with a bandwidth of 307GB/s, while the M5 Max supports up to 128GB of unified memory with a bandwidth of 614GB/s. These enhancements are crucial for handling large-scale machine learning models and data-intensive applications efficiently.
    - The new chips also introduce up to 2× faster SSD speeds, reaching 14.5GB/s, which can significantly reduce data access times and improve overall system performance. Additionally, the inclusion of the Apple N1 wireless chip for Wi-Fi 7 support offers faster download speeds, provided the network infrastructure can support it, enhancing connectivity for data-heavy tasks.
    - There is anticipation around the potential performance of the M5 Max, especially in the context of future Mac Studio models. The M5 Max's capabilities could provide insights into what might be expected from an M5 Ultra variant, although some speculate that a Mac Studio update might be delayed until the M6 release. This highlights the strategic planning involved in Apple's product release cycles.

  - **[ChatGPT uninstalls surged by 295% after Pentagon deal](https://www.reddit.com/r/LocalLLM/comments/1rjlzgy/chatgpt_uninstalls_surged_by_295_after_pentagon/)** (Activity: 348): **The image is a meme that humorously suggests a correlation between a supposed deal between ChatGPT and the Pentagon and a significant increase in uninstalls of the ChatGPT app, depicted as a 295% surge. The image uses visual elements like a declining graph and the Pentagon logo to imply a negative reaction from users. However, the claim lacks sourcing and context, as noted in the comments questioning the validity and scale of the uninstall data. The comments also express skepticism about the significance of the uninstall rate relative to the total user base.** Commenters express skepticism about the claim, questioning the source and significance of the uninstall rate, suggesting it might be a minor fluctuation rather than a substantial trend.

    - The claim of a 295% surge in ChatGPT uninstalls following a Pentagon deal raises questions about the scale of the user base affected. One commenter speculates that this might be a minor fluctuation in the overall churn rate, suggesting that the absolute number of uninstalls could be small relative to the total user base.
    - The discussion touches on the implications of AI in military applications, with one commenter noting that integrating AI into defense systems is a natural progression in technological advancement. This reflects broader concerns about the ethical and strategic dimensions of AI deployment in military contexts.
    - A link to a TechCrunch article is provided, which appears to substantiate the claim about the uninstall surge. This suggests that the information might be credible, although the original post's claim was initially questioned for lacking sources.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude and Claude Code Traffic and Features

  - **[We know why!](https://www.reddit.com/r/singularity/comments/1rjyy3f/we_know_why/)** (Activity: 994): **The image is a tweet from a user named Thariq, discussing an unexpected increase in traffic for "Claude and Claude Code" that was difficult to predict. The tweet expresses gratitude to users for their patience as they scale. This suggests that the service, likely related to AI or coding, is experiencing rapid growth and facing challenges in scaling to meet demand. The comments hint at broader industry dynamics, such as competition and political factors affecting tech companies.** One comment suggests that few companies are willing to oppose the US government, implying political factors in tech industry dynamics. Another comment humorously suggests that scaling issues could be a strategic vulnerability for top AI companies.

    - FalconsArentReal discusses a potential technical issue where an AWS data center in the Middle East was allegedly hit by a missile strike from Iran. This incident reportedly affected Anthropic, which uses AWS as a data center provider. The commenter speculates that traffic from the Middle East was rerouted to North American data centers, which were already under strain due to users switching from OpenAI, leading to significant operational challenges.
    - legaltrouble69 suggests a strategic vulnerability in the AI industry, proposing that if one of the top two AI companies is 'canceled' or disrupted, the other might face insurmountable scaling issues. This comment highlights the interdependence and potential fragility within the AI sector, where the failure of one major player could have cascading effects on the other.
    - SomewhereNo8378 makes a political observation about the reluctance of companies to oppose the US government, implying that those who do face significant challenges. This comment, while not deeply technical, touches on the broader socio-political environment that can impact technological companies and their operations.

  - **[Claude and Claude Code traffic grew faster than expected this week](https://www.reddit.com/r/ClaudeAI/comments/1rjyp7d/claude_and_claude_code_traffic_grew_faster_than/)** (Activity: 1518): ****Anthropic** has reported an unexpected surge in traffic for their AI models, **Claude and Claude Code**, which has outpaced their forecasts. This increase in usage has prompted the company to scale their infrastructure to accommodate the demand. The tweet by Thariq highlights the challenges faced due to this rapid growth and thanks users for their patience as they work on scaling efforts. The image is a screenshot of this tweet, emphasizing the unexpected nature of the traffic spike and the company's response to it.** One commenter speculates that the increased traffic might be due to more paid subscribers, while another notes experiencing faster limits, suggesting potential strain on the system.


  - **[New: Voice mode is rolling out now in Claude Code, live for ~5% of users today, details below](https://www.reddit.com/r/ClaudeAI/comments/1rjkwqk/new_voice_mode_is_rolling_out_now_in_claude_code/)** (Activity: 950): ****Claude Code** has introduced a new *Voice Mode* feature, currently available to `~5%` of users, with plans for broader rollout. This feature allows users to use a push-to-talk mechanism by holding the spacebar to dictate text, which streams directly at the cursor position without overwriting existing text. Importantly, using voice mode does not incur additional costs or affect token rate limits, and it is available on Pro, Max, Team, and Enterprise plans. [Source](https://x.com/i/status/2028628570692890800)** A user expressed a desire for a more interactive voice assistant that could engage in real-time discussions, similar to how they use ChatGPT for meetings and proposals. This suggests a demand for more advanced conversational capabilities in voice features.

    - universenz highlights the potential of voice mode in Claude Code for creating a more interactive and dynamic personal assistant. They compare it to using ChatGPT's voice capabilities for meetings and proposals, where the AI can convert spoken discussions into concise technical summaries. This approach allows for a more thorough and detailed exploration of ideas, akin to collaborating with a human team.
    - PulpAssets comments on the impact of Claude's new voice feature on the startup ecosystem, specifically mentioning how it could potentially disrupt companies like Wispr Flow. This suggests that single features in large AI models can significantly affect niche startups by offering similar capabilities at scale.


### 2. Gemini 3.1 Flash-Lite Release and Benchmarks

  - **[Gemini 3.1 Flash Lite](https://www.reddit.com/r/Bard/comments/1rjtfa3/gemini_31_flash_lite/)** (Activity: 394): **The image provides a preview of **Google's Gemini 3.1 Flash Lite**, a high-efficiency model designed for high-volume use, with a significant context size of `1,048,576`. The release is scheduled for March 3, 2026, and includes details on pricing for input, output, and audio tokens. This model appears to be positioned as a successor to Gemini 2.5 Flash Lite, but with a notable increase in cost, which has sparked some debate among users about its economic viability for existing implementations.** Commenters express concern over the increased cost of Gemini 3.1 Flash Lite, noting that the price is `3x` higher than its predecessor, Gemini 2.5 Flash Lite, which is priced at `$0.1` for input, `$0.4` for output, and `$0.3` for audio. This has led to skepticism about its practicality for current users.

    - Scary_Light6143 highlights a significant cost increase with the Gemini 3.1 Flash Lite model, noting a 3x price hike compared to its predecessor, 2.5. This raises concerns about the practicality of upgrading, as the cost may not justify the performance improvements for most implementations.
    - Accurate-Tap-8634 provides specific pricing details for the Gemini 2.5 Flash Lite model, stating it costs `$0.1` for input, `$0.4` for output, and `$0.3` for audio. This information is crucial for comparing the cost-effectiveness of the newer 3.1 version.
    - cmredd points out a `2.5x` increase in input costs and a `3.75x` increase in output costs with the Gemini 3.1 Flash Lite. They question the trend of AI models becoming more expensive, suggesting that the benchmark improvements may not justify the higher costs for the majority of use cases.

  - **[Gemini 3.1 Flash-Lite Benchmark Comparison](https://www.reddit.com/r/Bard/comments/1rjusj5/gemini_31_flashlite_benchmark_comparison/)** (Activity: 146): **The post discusses a benchmark comparison between **Gemini 3.1 Flash-Lite** and previous models, specifically noting that the comparison is made against **2.5 Flash** rather than **3 Flash**. The **Gemini 3.1 Flash-Lite** model card is available [here](https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/), while the **3 Flash** model card can be found [here](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf). The discussion highlights that **Gemini 3.1 Flash-Lite** is twice as expensive as **2.5 Flash Lite**, with specific pricing details: `3.1 Flash Lite - $0.25 input/$1.50 output`, compared to `2.5 Flash Lite - $0.10 input/$0.40 output`. This suggests that while **3.1 Flash Lite** is cheaper than **3 Flash**, it may not offer a cost-effective improvement for large data processing tasks.**

    - **Important-Farmer-846** highlights the cost-effectiveness of 2.5 Flash Lite over 3.1 Flash Lite, noting that while 3.1 is half the price of Flash 3, it is twice as expensive as 2.5 Flash Lite. The commenter suggests that for processing large volumes of data, 2.5 Flash Lite remains the better option due to its lower cost and similar performance.
    - **ExpertPerformer** provides a detailed cost comparison of various models, showing that 3.1 Flash Lite is less cost-effective compared to others like MinMax M2.5 and Grok 4.1. For instance, 3.1 Flash Lite costs `$0.25 input/$1.50 output`, whereas 2.5 Flash Lite is `$0.10 input/$0.40 output`, and Grok 4.1 is `$0.20 input/$0.50 output`. This suggests that 3.1 Flash Lite may not be competitive in terms of cost-performance ratio.
    - **ThomasMalloc** discusses the inefficiency of 3.1 Flash Lite in 'High' thinking mode, noting it took 14 times longer than 2.5 Flash Lite. The model maxed out at 65,436 output tokens compared to 6,980 for 2.5 Lite, indicating excessive token usage. The commenter suggests using 'Minimal' or 'Low' thinking modes to reduce token usage and cost, as these modes performed reasonably well with fewer tokens.


### 3. OpenAI and ChatGPT Backlash

  - **[Damnnnn!](https://www.reddit.com/r/singularity/comments/1rjc5to/damnnnn/)** (Activity: 2419): **The image is a meme-style screenshot from TechCrunch on X.com, highlighting a significant increase in ChatGPT uninstalls by `295%` following a Department of Defense (DoD) deal. This suggests a public backlash or privacy concerns related to the DoD's involvement with ChatGPT. The post has garnered substantial engagement, indicating widespread interest or concern. However, a top comment points out that the percentage increase could be misleading without context, as it could represent a small absolute change in numbers. Another comment speculates on the financial implications, suggesting that while user uninstalls might impact revenue, the DoD contract could offset this loss. The discussion also touches on privacy concerns, questioning the use of OpenAI products in light of government contracts.** Commenters debate the significance of the uninstall surge, with some suggesting the percentage increase might be misleading without absolute numbers. Others discuss the financial trade-offs between losing subscribers and gaining government contracts, and express privacy concerns regarding OpenAI's collaboration with the DoD.

    - mazdarx2001 highlights the financial implications of user cancellations, noting that if one million users paying $20 monthly cancel, it results in a $20 million monthly revenue loss. However, they argue that a Department of Defense (DoD) contract could offset this loss, as it potentially brings in more revenue, funded by taxpayer money.
    - Orangeshoeman discusses the potential impact on OpenAI's downstream corporate revenue due to a Department of Defense contract. They suggest that privacy-conscious users might avoid OpenAI products, implying that the contract could harm OpenAI's reputation among privacy-focused consumers.

  - **[ChatGPT Uninstalls Surge 295% After OpenAI’s DoD Deal Sparks Backlash](https://www.reddit.com/r/ChatGPT/comments/1rjfipu/chatgpt_uninstalls_surge_295_after_openais_dod/)** (Activity: 2938): **OpenAI's recent partnership with the U.S. Department of Defense led to a `295%` increase in uninstalls of the ChatGPT mobile app, reflecting significant user backlash. This reaction highlights the reputational risks of government contracts in the AI sector, as user sentiment can heavily influence corporate strategies. The event also saw a rise in downloads for competitor Claude, indicating shifting competitive dynamics in the AI market. For more details, see the [original article](https://techputs.com/chatgpt-uninstalls-surge-295-percent-dod-deal/).** Some comments suggest that OpenAI's strategy might involve shifting away from consumer-facing services, possibly to focus on other revenue streams like advertising or government contracts. There is also a sentiment that the backlash was expected and overdue, reflecting broader concerns about ethical implications of AI partnerships with military entities.

    - EnotHOME questions the significance of the 295% increase in uninstalls, suggesting that if the baseline was 1000 uninstalls, a 295% increase would mean 4000 uninstalls, which they consider insignificant in the grand scheme of things. This implies a need for more context on the baseline numbers to assess the impact accurately.
    - coronakillme seeks clarification on the 295% figure, interpreting it as the number of uninstalls being a little less than three times higher than before. They question what the original number of uninstalls was, highlighting the importance of understanding the baseline to evaluate the true impact of the increase.

  - **[Cancelling subscription - goodbye Sam I'm not funding your war machine!](https://www.reddit.com/r/ChatGPT/comments/1rjg8m0/cancelling_subscription_goodbye_sam_im_not/)** (Activity: 606): **The image is a screenshot of an email from OpenAI confirming the cancellation of a ChatGPT Plus subscription, which will remain active until March 23, 2026. The post's title suggests a protest against OpenAI's perceived involvement in military applications, reflecting broader concerns about tech companies' collaborations with defense and intelligence agencies. The comments discuss the use of Yahoo Mail and reference a controversy involving **Anthropic** and the Department of Defense, highlighting the complex relationship between tech companies and government agencies. The linked Bloomberg article provides further context on Anthropic's involvement in a Pentagon drone swarm contest.** Commenters express skepticism about tech companies' claims of non-involvement with military projects, suggesting that such collaborations are inevitable. The discussion also touches on privacy concerns related to Yahoo's past cooperation with government surveillance efforts.

    - VVadjet highlights the pervasive involvement of tech companies with defense and intelligence agencies, suggesting that Anthropic's recent actions were a PR misstep. They reference a [Bloomberg article](http://bloomberg.com/news/articles/2026-03-02/anthropic-made-pitch-in-drone-swarm-contest-during-pentagon-feud) detailing Anthropic's participation in a drone swarm contest, implying that such collaborations are common and expected in the industry.
    - ClankerCore emphasizes the need for concrete evidence and analysis over mere slogans and screenshots when evaluating tech companies' involvement with defense projects. They call for detailed contract language, constraints, enforcement, and oversight as critical factors for trust. Additionally, they point out that Anthropic's service, Claude, has faced rate-limiting and outages, indicating infrastructure challenges amidst increased demand.
    - LiteratureMaximum125 references a report on Yahoo's involvement in government surveillance, linking to a [source](https://lieu.house.gov/media-center/in-the-news/yahoo-helped-us-government-spy-emails-report-says) that discusses Yahoo's cooperation with U.S. government email spying. This highlights broader concerns about tech companies' compliance with government surveillance requests.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 3.0 Pro Preview Nov-18

**Theme 1. Frontier Models: GPT-5.3 Fallout, Gemini CoT, and Qwen Uncertainty**

- **GPT-5.3 "Safety Lobotomy" and 5.4 Teasers**: OpenAI released [GPT-5.3 Instant](https://openai.com/index/gpt-5-3-instant/) to mixed reviews, with **LMArena** users labeling it a "safety lobotomy" that underperforms 5.2-chat on health benchmarks. While **Nous Research** members rumor the upcoming **GPT-5.4** possesses *military capabilities*, **OpenAI** discord users anticipate a rapid follow-up release featuring **Sora** integration.
- **Gemini 3.1 Pro vs. Claude Opus 4.6 Coding Duel**: Debate persists in **LMArena** regarding coding supremacy, where **Claude Opus 4.6** is praised for reasoning despite **Anthropic** service outages, while **Gemini 3.1 Pro** is seen as faster but hallucination-prone. **Unsloth** engineers noted that extracting Gemini's *true* **Chain of Thought (CoT)** via `<think>` tags yields better results than its standard summaries, evidenced by [this screenshot](https://cdn.discordapp.com/attachments/1179779344894263297/1478209505404784650/Screenshot_20260203-015813_Firefox.png?ex=69a8e2e1&is=69a79161&hm=c61029735f4655d6d5e4f03137673befc563c3417393f4dd014b71c6795fb35c).
- **Qwen Team Exit and Rollout Failures**: Following the [departure of the Qwen team lead](https://bsky.app/profile/natolambert.bsky.social/post/3mg6eisffss2j), **Unsloth** and **OpenRouter** users report flawed rollouts and fears regarding the future of open weights. Despite this, technical exploration continues, with Andrew Carr sharing a project on [ranking individual neurons](https://xcancel.com/andrew_n_carr/status/2028649735809319013?s=12) within **Qwen 3.5 0.8B**.

**Theme 2. Hardware Acceleration: CUDA Agents, Blackwell Splits, and Custom Silicon**

- **CUDA Agents Crush Torch Compile**: A new **CUDA-specialized RL agent** discussed in **GPU MODE** reportedly beats `torch.compile` by **2x** on medium kernels and outperforms **Claude Opus 4.5** on complex benchmarks ([paper](https://arxiv.org/abs/2602.24286)). Simultaneously, **ByteDance** released a similar [CUDA Agent](https://cuda-agent.github.io) for writing fast kernels, sparking interest in automated kernel generation over manual optimization.
- **NVIDIA Blackwell Architecture Bifurcation**: **GPU MODE** engineers identified a significant split in **NVIDIA's Blackwell** generation between Data Center (**CC 10.0**) and Consumer (**CC 12.0**) tracks. Compatibility breaks are expected, as some features now require **sm_100a** or **sm_100f** targets, detailed in [NVIDIA's blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/).
- **Taalas and Apple Silicon Push Limits**: **Unsloth** members discussed the **Taalas HC1** chip delivering a massive **17,000 tokens/s** for hardwired models, though locked to specific architectures. Meanwhile, **Latent Space** users report the **Apple M5 Neural Engine** runs **Llama2 110M** with **80x** efficiency over A100s, and **OpenClaw** members are leveraging **M5 Pro** chips for local agent hosting.

**Theme 3. Agentic Frameworks: C-Coded Binaries, RLM, and Kimi**

- **ShadowClaw Emerges as Minimalist C Agent**: **OpenClaw** and **HuggingFace** communities are highlighting **ShadowClaw v1.1**, a single-binary personal AI agent written in **C** that communicates via `curl` with local LLMs like **Ollama**. The tool, available on [GitHub](https://github.com/webxos/webxos/tree/main/shadowclaw), emphasizes low overhead with features like shell execution, file manipulation, and persistent state saving.
- **Recursive Language Modeling (RLM) Paradigm**: **DSPy** users are debating the convergence of agent paradigms toward **RLM**, where LLMs access a **REPL** rather than static tools, calling it potentially superior to user-defined Python functions. This recursive approach involves sub-agents spawning to run their own code, distinct from standard **ReAct** loops.
- **Kimi Code Challenges Claude**: **Moonshot AI** launched **Kimi Code**, a distinct agent from **Claude Code**, which **OpenClaw** users claim is *5 times better* than Minimax for specific tasks. While some users prefer the open-source **OpenCode** alternative, **Kimi** is being used to [replace YouTube](https://cdn.discordapp.com/attachments/1371757564005711973/1478272020889210992/tech_gaming_news_prompt.txt?ex=69a8745a&is=69a722da&hm=3c69473f87fa6f0eb449e3cbd498cb96a7e7d3c3f9b17a8b422ca813d4d9ee3d) for news aggregation via its iPython environment.

**Theme 4. Developer Infrastructure: Real-Time Evals and $255B Inference Markets**

- **Real-Time Training Observability**: **HuggingFace** users highlighted **TrainTrackLabs**, a new observability layer that plugs into **PyTorch** to score hallucination and reasoning in real-time using **LLM-as-a-judge**. This aims to catch regression early in fine-tuning runs to prevent wasted GPU spend ([traintracklabs.com](https://traintracklabs.com/)).
- **Time Travel Debugging with AI**: **Latent Space** engineers discussed the resurgence of time-travel debugging via a [Replay MCP](https://docs.replay.io/basics/replay-mcp/overview). The tool reportedly reduced a **React 19** upgrade debugging session from vague error overlays to root cause identification in **30 seconds**.
- **Inference Market Valuation Soars**: Analysts in **Latent Space** project the AI inference market to hit **$255 billion by 2030**, driven by production deployment costs outpacing training. This shift is corroborated by **Unsloth** discussions on inference optimization (Taalas) and **HuggingFace** discussions on efficient transcription tools like [easytranscriber](https://huggingface.co/blog/KBLab/easytranscriber).

**Theme 5. Research & Theory: Spectral Norms, Drift Sinks, and Jailbreaking**

- **Spectral Norm Scaling for Feature Learning**: **Eleuther** researchers discussed a [2023 paper](https://arxiv.org/abs/2310.17813) demonstrating that feature learning is achieved by scaling the **spectral norm** of weight matrices. This derivation connects to **maximal update parametrization (muP)** and recent **Modula** work.
- **Drift Sinks and Persona Tokens**: **OpenAI** users proposed theoretical frameworks like **Drift Sinks** to arrest "semantic drift" in analytical systems by enforcing epistemic gravity. They also explored **self-tokens** as portable [persona-containers](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12) to maintain agent identity across decentralized platforms.
- **Chemical Synthesis and Jailbreaking**: **BASI Jailbreaking** members detailed a four-step synthesis for **MDMA** from **Safrole** (70-80% yield) and discussed "Eni jailbreaks" for profit. This contrasts with **LMArena** reports of **GPT-5.3's** heavy censorship and **Nous Research** discussions on creating specialized pentest models on constrained hardware (8GB VRAM).


---

# Discord: High level Discord summaries




## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw Docs Get Open-Sourced!**: The OpenClaw community has [open-sourced their community policies and guidelines](https://github.com/openclaw/community) as well as internal documentation, excluding **trial moderator** info and **moderation logs**.
   - The team also restructured its **team hierarchy**, with full documentation available in the same repository, and promoted  <@1255431768199135254>, <@405240788143046656>, and <@957289026195435520> to <@&1469028608293998723>.
- **Insta-Claw Connects!**: An OpenClaw user has released an **Instagram** channel integration, available on [npmjs.com](https://www.npmjs.com/package/@rylena/openclaw-instagram) and [GitHub](https://github.com/rylena/openclaw-instagram).
   - The integration is still a work in progress, with other users being encouraged to test it out.
- **Kimi Crushes Minimax?**: Members debated the performance and cost-effectiveness of different AI models, with one user commenting that *kimi is 5 times better than minimax*.
   - Another member weighed in, adding that *Well Kimi is massive, they’re both good. It’s my current setup*.
- **ShadowClaw Emerges as Lean, Mean, C-Coded Agent!**: **ShadowClaw** has been introduced as a minimal, single-binary personal AI agent written in C and communicating with a local LLM (**Ollama**) via curl, [available on GitHub](https://github.com/webxos/webxos/tree/main/shadowclaw).
   - Features include shell command execution, file read/write, HTTP GET, and simple math expression evaluation, with state automatically saved to disk.
- **Video Editing Gets Turbocharged with OpenClaw!**: A user reported using **OpenClaw** in Web2Labs Studio for video editing, which helps speed up the process by automating jump-cuts, zooms, and thumbnail generation.
   - The user emphasized the time saved in editing and the ability to *ship consistently* due to the automation of title, description, and thumbnail generation.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-Omni's Unsloth Support Still in Limbo**: A user asked about **Qwen3-Omni** support in **Unsloth**, reporting **15 t/s** on an **XPS 15** for agentic coding using **Opencode** with **Neovim**.
   - They also inquired about standard benchmarking procedures for low-end hardware, and were directed to [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- **Qwen Team Lead Departs Post-Google**: Members discussed the [departure of the **Qwen** team lead](https://bsky.app/profile/natolambert.bsky.social/post/3mg6eisffss2j), after the company allegedly forced them to step down for someone at **Google**.
   - Users speculated that it could mean the end of **Qwen open source weights**, and lamented that it happened *right after 3.3 (we don’t talk about llama 4)*.
- **Taalas Chips Promise Speedy LLMs—for a Price**: Discussions mentioned **Taalas chips** potentially enabling **LLMs** in games locally, with the **Taalas HC1** delivering up to **17,000 tokens/s**.
   - However, the downside is that it only works with the model hardwired into the hardware, and one member suggested that *expense* was a barrier.
- **Gemini's Hidden Chains of Thought**: Members noted that **Gemini** summaries are better than the model's *'true'* **Chain of Thought (CoT)** due to minor reasoning, although extracting the actual **CoT** is possible using specific setups from earlier versions.
   - Screenshots suggest a structured reasoning approach using tags like `<think>`, as seen in **Gemini 2.5 Pro**, before summaries replaced them; some pointed to this [screenshot](https://cdn.discordapp.com/attachments/1179779344894263297/1478209505404784650/Screenshot_20260203-015813_Firefox.png?ex=69a8e2e1&is=69a79161&hm=c61029735f4655d6d5e4f03137673befc563c3417393f4dd014b71c6795fb35c).
- **Fine-tuning Qwen Models: Speed Bumps and Solutions**: A member reported that fine-tuning **Qwen3.5-2B** takes **4 hours**, while **Qwen3-1.7B** takes only **3 minutes** using the same script and data, leading to suggestions for optimization.
   - It was advised to install `flash-linear-attention` and `causal_conv1d` to resolve the speed disparity, noting that **Qwen3-VL 8B** takes about 2 hours and **Qwen3.5-9B** takes 6.5 hours with these installed.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Safrole Synthesis Steps Detailed**: A member provided a four-step synthesis pathway for **MDMA** starting from **Safrole**, including Isomerization, Oxidation to MDP2P, Reductive Amination, and Salt Formation.
   - The process, based on declassified **1970s** research, is expected to yield approximately **70-80%** of theoretical mass.
- **Jetson Thor Dev Kit Cooperative Planned**: Members discussed co-oping a **Jetson Thor dev kit** with 128 GB VRAM, potentially costing around **$800-$1000** per person and sharing compute on a private network.
   - The group pondered allocating an extra **$200** per person for a *badass private server* or using all 128GB of VRAM individually.
- **MITRE ATLAS for AI Red Teaming Shared**: A member shared [MITRE ATLAS](https://atlas.mitre.org/matrices/ATLAS) as a structured resource for learning **AI red teaming**, offering a more organized approach than OWASP.
   - The discussion also touched on the appeal of crafting prompts without AI assistance.
- **Eni Jailbreak: Quick Cash?**: Members debated the ease of creating "eni jailbreaks," with some suggesting selling jailbreak prompts as a potential revenue stream.
   - Concerns were raised about others potentially using and selling the *same eni JB*.
- **AI Gets Burned by Jailbreak Attempt**: A member recounted an unsuccessful attempt to jailbreak an AI, resulting in the AI roasting their ego instead.
   - This exchange was humorously punctuated with a [tenor gif](https://tenor.com/view/hmm-ok-okay-then-gif-24387952).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Link Stabilizes After Fix**: **LM Link creation** and **device discovery** in LM Studio are now working stably, following a resolution.
   - The waitlist, temporarily paused for testing, is **active again** as of 8:55 PM EST; users will be notified via email upon admission.
- **Google Siri Implementation Expected To Stay Local**: Members anticipate that **Google's Siri implementation** will be fully local, though Google may still promote their cloud service.
   - Users in **#general** are excited about a possible local **Siri implementation**.
- **DDR3 Prices Skyrocket Due to Limited Supply**: A member reported that **DDR3 prices have doubled** since their last purchase because the supply is now limited.
   - One member joked about *profitsssSSs* from their *bunch of old ddr3 laptops*.
- **Vulkan Balances VRAM But Struggles With Context Loading**: A user confirmed that **Vulkan** can balance model layers, loading 16GB cards to ~14-15GB and 32GB cards to ~28GB, but struggles with context loading.
   - The user noted that long context for agentic use means leaving ~5GB of VRAM empty on 3 cards to avoid OOM errors.
- **NeuroStream Could Massively Reduce VRAM Usage**: A member inquired if **Topaz NeuroStream** could enable running much larger models with **95% less VRAM usage**.
   - Another member noted that modern local LLMs are improving in efficiency and mentioned [Microsoft's bitnet](https://www.microsoft.com/en-us/research/blog/scaling-down-llms-bitnet-the-end-of-expensive-large-language-models/) as a similar technology.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude Suffers 'Unprecedented' Outage**: Users reported issues with **Claude**, including rate limits and errors, attributing them to *'unprecedented demand'* causing service disruptions, even citing a [Mashable article](https://mashable.com/article/claude-down-anthropic-outage-statement) from February 2026.
   - Some alarming rumors spread, suggesting the downtime was due to drone strikes on **AWS data centers in the UAE**, allegedly causing infrastructure damage and outages.
- **Gemini and Claude Duel for Coding Crown**: The debate continues over whether **Gemini 3.1 Pro** or **Claude Opus 4.6** reigns supreme for coding tasks; one user declared *'Gemini bad'* while another asserted **Claude Opus 4.6** boasts superior thinking capabilities and code quality.
   - Despite hallucination issues, some users find **Gemini 3 Pro** faster.
- **Arena Users Timeout Over 10-Minute Limit**: Users expressed frustration with **Arena's 10-minute timeout limit**, especially with models like **Claude Opus 4.6** on larger projects, leading to frequent *'Error, something went wrong'* messages.
   - One user dramatically requested an extension to *2 hours*.
- **GPT-5.3 Gets Safety Lobotomy?**: [Early reports](https://deploymentsafety.openai.com/gpt-5-3-instant) indicate **GPT-5.3** is *not measurably/objectively better* than **5.2-chat**, being merely fine-tuned for style and potentially user-preference responses, with claims it scored worse on a health benchmark.
   - One user quipped *'Lmao so it’s More safety lobotomized'*, indicating concerns about its utility.
- **Arena.ai Explained**: A [YouTube video](https://www.youtube.com/watch?v=nktiDGTn61I) concisely explains **Arena.ai** in 60 seconds.
   - It is uncertain what the video missed in its concise overview.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Agent Stuck in Loop?**: A user experiencing issues with their **Cursor agent** getting stuck in a loop was advised to specify actions the agent *shouldn't* perform, alongside its primary tasks.
   - This approach aims to constrain the agent's behavior and prevent repetitive loops during task execution.
- **Cloud Agent Goes Mobile with Android Support**: The **cloud agent** now supports **Android**, functioning as a [webapp](https://cursor.sh), expanding its accessibility across different platforms.
   - This enhancement allows users to leverage the cloud agent's capabilities directly on their Android devices.
- **Web Dev Falls Flat with Default Settings**: A user expressed disappointment with **Cursor's** default web development outputs using **Codex 5.3**, describing the designs as subpar.
   - Other users suggested using specific packages like [shadcn](https://ui.shadcn.com/) and giving detailed prompts with references and copying source code from desired website sections.
- **Cursor Tweaks Layout for Simplicity**: Based on user feedback, **Cursor** has simplified its layout sidebars, addressing concerns that the previous version was too confusing.
   - The **Zen layout** remains accessible via the **Command+Option+Tab** shortcut, despite the change in discoverability with the new layout.
- **Viktor the AI Co-Worker Enters the Slack Workplace**: An AI co-worker named [Viktor](https://www.producthunt.com/products/viktor), fully built with **Cursor**, launched on Slack, offering capabilities such as **marketing audits**, **ad management**, and **lead research**.
   - Integrating with over **3,000 tools** and using persistent memory, Viktor learns company specifics and composes tools for complex actions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLMs create Code Janitors**: The role of a "**code janitor**" is becoming more important as **LLMs** take prominence, focusing on creating **abstractions** and **guardrails** to prevent incidents.
   - It was noted that **LLMs** can make it harder to rely on system knowledge during **PR reviews**, increasing the value of specialized roles.
- **Roblox Aiming for Trillion-Dollar Metaverse**: **Roblox's** blend of technical advantages, AI-driven creation tools, and rapid design evolution positions it as a future trillion-dollar company, comparing its growth to TikTok in [this post](https://xcancel.com/jnavok/status/2028664806601855421?s=12).
   - The discussion also touches on **Roblox's** potential to become a metaverse platform and concerns about loot boxes, referencing a [Reuters article](https://www.reuters.com/legal/government/new-york-sues-video-game-developer-valve-says-its-loot-boxes-are-gambling-2026-02-25/) on **Valve's** loot box issues.
- **Qwen 3.5 Neuronal Ranking System Unveiled**: Andrew Carr shared a project focused on ranking every individual neuron within the **Qwen 3.5 0.8B model** ([original post](https://xcancel.com/andrew_n_carr/status/2028649735809319013?s=12)), likely exploring model interpretability or importance mapping.
   - Xinyu Yang critiqued the replacement of **Qwen's leadership** with a metric-driven hire from Google Gemini, cautioning against managing foundation model research like a consumer app development cycle, as shown on [this X post](https://xcancel.com/xinyu2ml/status/2028867420501512580?s=46).
- **Time Travel Debugging gets Replay Boost**: A member announced the re-pivot to **time travel debugging with AI**, highlighting the availability of a [Replay MCP](https://docs.replay.io/basics/replay-mcp/overview) and its powerful capabilities.
   - They tested it against a failing **React 19 upgrade issue** and it went from *screenshot of an error overlay* to *I know what the problem is* in about **30s**.
- **AI Inference Firms Head Towards Billions**: Meg McNulty highlights the surge in valuation of **AI inference companies** in [this tweet](https://xcancel.com/meggmcnulty/status/2028532451992314199) noting that software for running models is becoming more valuable than model training.
   - She forecasts a **$255 billion market by 2030**, propelled by the recurring costs of production-level **AI** deployment.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Splits the G at GTC**: Nous Research is inviting people to come split the G at GTC (GPU Technology Conference) as announced in [this X post](https://x.com/nousresearch/status/2028861034220405178).
   - This is an opportunity to meet and connect with the **Nous Research** team at the conference.
- **GPT 5.4 pack military might**: Members are speculating that **GPT 5.4** is comparable to **5.3-codex** but includes *military capabilities*.
   - The channel discussed that *self learning is largely solved from a research perspective, but its just impractical to integrate*.
- **Anthropic is Caching Prefills**: Members observed that **Anthropic** seems to cache the prefill to spare costs, but this makes switching models impossible.
   - One member pointed out that this allows them to reduce costs against **OpenAI**, who also seem to optimize for inference cost vs. user retention.
- **Opus's Odd Approach to Arithmetic**: A member shared an [example](https://example.com) of **Opus** doing math by determining the last two digits and using a lookup table for the first digit.
   - The channel agreed that this approach highlights the limitations of **LLMs** for math, as it is based on pattern recognition rather than actual mathematical understanding.
- **Seeking Pentest Model With Limited Resources**: A member is seeking advice on creating a dedicated **pentest model** using a **Hermes model** as the base, optimized for training with limited resources, owning an **8GB VRAM GPU**.
   - As a Brazilian member they are also constrained by the high costs of local GPU providers, which are similar to those in the US.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.3 Instant Lands, GPT-5.4 Teased**: The latest **GPT-5.3 Instant** model is now rolling out to all **ChatGPT** users, boasting improved accuracy, according to [the announcement](https://openai.com/index/gpt-5-3-instant/). A follow-up message hints that **GPT-5.4** may be released sooner than anticipated.
   - Users report a staged rollout of **GPT-5.3**, with some experiencing delays and others noting that the app dropped the **5.2** indicator after updating. Users are anticipating **GPT-5.4** release next week with Sora integration.
- **Low-Quality AI Creation Gets Bashed**: Members discussed how society incentivizes the creation of low-quality AI content for monetary gains, with one user questioning, *"if society wanted 'good' then why does it incentivise slop?"
   - A user criticized **Sora's AI-generated voices** for sounding artificial and being unnaturally fast, adding to the concerns about low-quality AI content. Users also complained that questions to ChatGPT require excessive caveats.
- **Discord Data Discouraged for LLMs**: A user inquired about using **Discord server messages** to train an LLM for active fine-tuning, but other members cautioned against it due to limited data volume and potential **TOS** violations.
   - One member cautioned that using Discord to train LLMs is *"the best way to make an LLM braindead."
- **Self-Tokens Advance AI Persona Portability**: A member suggests using *self-tokens* as **persona-containers** to enhance a framework, making **AI-personas** portable; a template is available via [this image](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12&).
   - A **relation gauge** is described as a productivity metric modeling the likelihood to maintain a linkage, suggesting that tokens should have mutable length, especially for decentralized platforms managed by multitudes.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Users Retain Data Ownership**: It was confirmed that users own their data, and prompts and responses are excluded from training public models by default, with all communications encrypted in transit (**TLS 1.2+**) and at rest (**AES-256**), as detailed in the [FAQ](https://openrouter.ai/docs/faq) and [privacy guide](https://openrouter.ai/docs/guides/privacy/data-collection).
   - Security standards include **SOC 2 Type 2**, **CSA STAR**, and **ISO** information security certifications.
- **LLMs Plagued with Bias Debates**: A debate ensued on whether **LLMs** are inherently biased due to their training data, with some suggesting creating an unbiased dataset checked by multiple unbiased humans.
   - Others argued that *all humans have biases*, and even an *"unbiased" LLM* would be trained on a biased dataset.
- **Client-Side Error Handling Saves Day!**: Some users experienced a `TypeError: undefined is not an object (evaluating 'p.choices [0].delta')` error, which lead to the discovery that OpenRouter sometimes does not send an expected delta value.
   - The solution involved client-side error handling and a fix was implemented for Venus Chub, as described in [this Github pull request](https://github.com/cline/cline/pull/9432).
- **BYOK and z.ai Compatibility Issues Arise**: Users reported issues using **z.ai** subscriptions through **BYOK** on OpenRouter, with an error message indicating *"Insufficient balance or no resource package"*.
   - It was clarified that **z.ai** subscriptions use a different base URL and are not directly compatible with **BYOK**, and the feature request to allow connection subscriptions to **BYOK** was denied.
- **Scrutiny of OpenRouter's Costs Reveals Nuances**: Users questioned the cost-efficiency of OpenRouter compared to direct API usage, citing discrepancies between dev logs and OpenRouter logs, whereas OpenRouter charges a **5.5%** fee.
   - The choice of LLM significantly impacts costs, with some models being more expensive than others; furthermore, a Joinable Bounty program to **stress-test new AI apps** and get paid in **USDT** was discussed.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA RL Agent Achieves Performance Milestone**: A **CUDA**-specialized **RL agent** reportedly outperforms **torch.compile** by **2x** on simple/medium kernels and exceeds **Claude Opus 4.5** and **Gemini 3 Pro** by approximately **40%** on the hardest benchmarks, according to [this paper](https://arxiv.org/abs/2602.24286).
   - Despite the promising results, concerns were raised about the absence of published kernels and the reliance on a *large GPU pool with process-level isolation*, which introduces significant computational and engineering expenses.
- **ByteDance Releases CUDA Agent for Kernel Writing**: ByteDance introduced a **CUDA Agent**, a model designed to write fast **CUDA kernels**, outperforming **torch.compile** by **2x** on simple/medium kernels, and surpassing **Claude Opus 4.5** and **Gemini 3 Pro** by approximately **40%** on the most complex tasks, see [tweet](https://x.com/BoWang87/status/2028599174992949508).
   - The shared link, [cuda-agent.github.io](https://cuda-agent.github.io), was called *interesting at a glance from ByteDance*.
- **Blackwell's Compute Capability gets granular**: Members discussed that **NVIDIA's Blackwell generation** is now split into Data Center (**CC 10.0**) and Consumer (**CC 12.0**) tracks, catering to optimization for **AI/HPC** and **real-time graphics**, respectively.
   - Some of the additional features are not forward compatible and require **sm_100a** or **sm_100f** instead of just **sm_100**; more info can be found at [NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/).
- **Kernelbook and Kernelbot eyed for Merger**: Due to the lack of collaboration features on Prime Hub, a member proposed publishing improved environments for others to review.
   - A member suggested merging **kernelbot** and **kernelbook** due to shared infrastructure, which could potentially optimize resource utilization and streamline development.
- **Teleop TRLC DK-1 system makes debut**: An experimental [TRLC DK-1](https://www.robot-learning.co/) teleop system was introduced, which can be used for human interventions when policies run OOD.
   - The first test used a [ELP stereo cam module](https://www.amazon.de/dp/B07FT2GKZS) mounted on a SO-101, demonstrated in [this video](https://x.com/neurosp1ke/status/2023073945637753101?s=20).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Real-Time Evaluation Tooling Enters the Ring**: **TrainTrackLabs** is developing a real-time evaluation and observability layer for **LLM training** by plugging directly into **PyTorch / Hugging Face** to continuously track reasoning, safety, hallucination, and coding ability using **LLM-as-a-judge** scoring, available at [traintracklabs.com](https://traintracklabs.com/).
   - They are looking for early pilot teams to catch regressions early and prevent wasted GPU spend.
- **Shadowclaw claws its way to v1.1**: The single-binary personal AI agent written in C, **Shadowclaw v1.1**, builds upon the original by adding built-in commands and a native tool, available on [GitHub](https://github.com/webxos/webXOS/tree/main/shadowclaw).
   - This version includes commands such as **/help**, **/tools**, **/state**, **/clear**, **/chat**, and **/exit**.
- **Easytranscriber Transcribes, Transforms Time**: `easytranscriber`, a library for **automatic speech recognition** with accurate timestamps, is similar to WhisperX, but runs **35% to 102% faster**, depending on hardware, available on the [Hugging Face blog](https://huggingface.co/blog/KBLab/easytranscriber).
   - It also supports HF models as a backend.
- **Europe SPRINTS to Frontier AI Leadership**: SPRIND offers **€125M** in equity-free funding for up to **10 teams** to build frontier AI labs in Europe via [next-frontier.ai](https://next-frontier.ai/), seeking novel architectures and agentic systems.
   - This initiative will establish research focused on next-generation agentic systems.
- **MCP Integration Security: A Messy Affair**: A deep dive into **Model Context Protocol (MCP)** attack vectors was shared, detailing 5 trivially exploitable patterns that every MCP developer should understand and is documented in a [Medium article](https://medium.com/@nainia_ayoub/mcp-security-is-a-mess-5-ways-i-broke-my-own-ai-agent-76379a46ca90?sk=0daa66d4fc2a68fbb02a56e803336ce2).
   - The discussion highlighted the ease with which these vectors can be exploited.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Code Differs From Claude Code**: Members confirmed that **Kimi Code** is a new agent developed by **Moonshot**, distinct from **Claude Code**.
   - One member highlighted **OpenCode** as their preferred open-source alternative to **Claude Code**, noting its popularity.
- **Moderato Plan Token Usage Revealed**: A user shared usage statistics for the **Moderato plan** ($19/month) on **OpenCode**, reporting 18% of their weekly limit consumed with **365 Messages**, **1.0M Input Tokens**, **115.6K Output Tokens**, and **25.3M Cache Read**.
   - This equates to a **20M input tokens monthly** budget, which was considered *not a great deal* by another user.
- **Kimi Eyes YouTube's Turf**: A user developed a prompt to enable **Kimi** to gather tech and gaming news, aiming to lessen dependence on **YouTube** by having **Kimi** independently reconstruct stories with [this prompt file](https://cdn.discordapp.com/attachments/1371757564005711973/1478272020889210992/tech_gaming_news_prompt.txt?ex=69a8745a&is=69a722da&hm=3c69473f87fa6f0eb449e3cbd498cb96a7e7d3c3f9b17a8b422ca813d4d9ee3d).
   - The user lauded **Kimi's chat interface**, citing features such as **search calls** and an **iPython environment**, deeming them practically limitless and ahead of competitors.
- **Kimi Allegretto Plan Cancellation**: A user inquired about canceling the **Kimi Coding Plan Allegretto** or deactivating renewal, and another user provided [a link to manage subscriptions](https://www.kimi.com/membership/subscription).
   - The cancellation option can be located within the profile settings.
- **Support Emails Delayed, Fraudulent Billing Unresolved**: A user reported that support emails are non-functional, with no resolution for fraudulent billing issues.
   - A non-team member speculated that an influx of emails post-spring festival holiday is causing delays, and that the issue has been escalated to staff.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Cohere's Aya Project Wants You!**: **Cohere** seeks collaborators for their [Aya project](https://aya.cohere.com/about), with opportunities available at **Fast AI**, **Eureka Labs**, or **Cohere Research Labs** depending on your skill level.
   - The goal of the project is to create an open source foundation for responsible multimodal AI.
- **CVPR 2026 Workshop on Medical Reasoning**: A member is organizing a **CVPR workshop** and invites submissions to the [Medical Reasoning Workshop](https://med-reasoner.github.io/cvpr2026/).
   - Additional information about the workshop is available on the [Discord event link](https://discord.gg/nxtWyHbY?event=1478419152103280680).
- **Spectral Norm Scaling Enables Feature Learning**: A [2023 paper](https://arxiv.org/abs/2310.17813) demonstrates that feature learning is achieved by scaling the **spectral norm** of weight matrices and their updates like √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗).
   - The paper's analysis also provides a basic derivation of **maximal update parametrization**.
- **Image Composition Controlled Early in Diffusion Models**: A new paper leverages the **SAE framework** to probe the inner workings of a popular **text-to-image diffusion model**, uncovering human-interpretable concepts in its activations, details at [https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473).
   - The research introduces intervention techniques to manipulate **image composition and style**, demonstrating that **image composition** can be effectively controlled in early stages of diffusion, according to the paper.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Mulls Mojo Package Manager**: Modular is considering building a [Mojo package manager](https://forum.modular.com/t/open-question-what-would-you-like-to-see-from-a-mojo-package-manager/2799?u=nate), potentially similar to Rust's `cargo` or Python's `pixi`, including a central repository.
   - The goal is to determine the community's desires and wants around distributing Mojo packages.
- **API Abstraction Applauded**: A member argued for designing APIs from a user perspective rather than based on implementation details, giving the example that `@inline(strategy: "chose whatever makes sense", ...)` improves the user experience versus `@always_inline` and `@never_inline`.
   - Another member agreed that good API design is vastly important and depends on a more general representation of decorators.
- **Vectorize Validation Voyage**: The jump from **Mojo 25.7 to 26.1** introduced significant changes related to parallelization and vectorization, particularly affecting closures, resulting in compiler errors.
   - Modular confirmed these changes are part of the push towards a **1.0 ready state**, and a clear migration recommendation will be provided, similar to the existing documentation for **UnsafePointer**.
- **Apple Aces Memory Safety**: Apple is poised to address memory integrity enforcement, potentially impacting Mojo, described in [this blogpost](https://security.apple.com/blog/memory-integrity-enforcement/) and [this analysis](https://www.ralfj.de/blog/2020/12/14/provenance.html).
   - This may become a significant issue for other platforms as well.
- **`comptime` Considerations Convene**: A member suggested streamlining compile-time metaprogramming syntax by using `@` instead of `comptime`, such that `@parameter if` would become `@if`.
   - Another member mentioned that they had requested the `maybe comptime` feature for Mojo previously.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Support Team: Business Hours Brigade**: Users on the **Manus.im Discord** inquired about the availability of the **Manus Support Team**, which is typically available during business hours, advising members to send a **DM** with their email address for assistance.
   - The team shared a [help article](https://help.manus.im/en/articles/12087847-how-to-optimize-my-credit-usage) with tips and information on how to use **Manus** more effectively and optimize credit usage.
- **Credits Don't Accumulate on Manus Plans**: A user asked whether unused credits from a **46€ plan** accumulate in the next month, with staff responding that *it doesn't look like it's accumulating*.
   - There was also some speculation of credits burning using the **Telegram Agent**.
- **Structuring AI-Driven App Development with Requirement Files**: A user sought guidance on utilizing **structured requirement files (PRD / system design doc)** with AI tools to build complex full-stack applications in a structured manner to prevent *sloppy, unstructured AI-generated code* by following a clear architecture-driven workflow.
   - The user wanted to ensure that they are building full-stack applications in a structured manner.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **REPL Parallels RLM Agent Paradigm**: The new agent paradigm using **REPL** is converging towards **RLM**, reminiscent of [this post](https://x.com/nfcampos/status/2028576281793630372?s=20) and [this post](https://x.com/RLanceMartin/status/2027450018513490419?s=20).
   - A member stated that the **RLM** paradigm of giving access to **REPL** to **LLM** will be superior to granting access to user-written Python functions.
- **RLM's Recursive Nature Debated**: Members debated whether recursion is a requirement for **RLM**, suggesting that the recursive aspect arises from spawning sub agents to run their **REPL**.
   - One member posited that *"Claude using a script to call Claude is a subagent of sorts"*, referencing [this link](https://x.com/a1zhang/status/2023976399694917808?s=20).
- **DsPy Meetup to Illuminate RLM**: A member proposed hosting a session at the **DsPy Meetup** in the Bay Area to clarify the fundamentals of **RLM** this month.
   - The session would involve comparing **RLM** with **ReAct** and examining how **RLM** determines what code to generate, given that it produces its own code rather than relying on user-defined Python functions as tools.
- **RLM Finds Niche in Extensive Context Docs**: One member found **RLM** suitable for handling documentation consuming many **MM tokens**, while others utilize it when they are comfortable with the **LLM** making self-directed calls.
   - To ensure optimal performance, they develop and implement evals and tests.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude Opus Burns Cash**: Members reported that **Claude Opus** can quickly accumulate charges of **$65/hr**, potentially leading to **$1000 USD bills** due to unrestricted client usage.
   - The discussion questioned the token volume required to hit such high hourly costs.
- **AiderMacs Lacks Project Buffer Sorting**: A member inquired about configuring **AiderMacs** for chat organization with project buffers in `ibuffer-projectile`.
   - No solutions or further details were shared regarding this issue.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Meeting Scheduled for March 2nd**: A new **tinygrad meeting** is scheduled for **March 2nd** at **8pm San Diego time**.
   - The meeting will cover *company updates, comma issues, CALL/BUFFER_VIEW sym llm, assign, setitem, disk, drivers, llama, VIZ, and other issues and bounties*.
- **Bounties Pull Request in Discussion**: The discussion mentions a pull request related to **bounties** ([PR #14982](https://github.com/tinygrad/tinygrad/pull/14982)).
   - No further details were mentioned about the specific content of this pull request.
- **Codebase Favoring `len(x.shape)`**: A member noticed the codebase uses `len(x.shape)` instead of `x.ndim` in many instances.
   - The member questions the value of a PR to address this, but highlights it as a potential refactor or style preference.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI-Ready Data Summit Scheduled for 2026**: The **AI-Ready Data Summit** is scheduled for **March 31, 2026**, and will host speakers from **Lockheed Martin**, **Dell Technologies**, **Red Hat**, **CNH**, and **Entrust**.
   - The summit will focus on practical enterprise AI, data infrastructure, and model deployment insights ([summit details](https://ai-ready-data-summit.com)).
- **AI Control Hackathon Challenges Techies in 2026**: **Apart Research** is hosting an **AI Control Hackathon** with **Redwood Research** from **March 20-22, 2026**, challenging participants to monitor and contain AI agents that subvert safety measures.
   - Interested parties can find more details on the [hackathon details page](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach).
- **AI Business Builders to Circle around OpenClaw**: A **45-minute roundtable** will take place to discuss how builders are using **OpenClaw** and other tools to run businesses, communities, and products on **March 14**.
   - See the [roundtable registration page](https://luma.com/qfrucnl2) for details.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Contributors Await Dev Summit**: Members expressed excitement towards the upcoming **MCP Dev Summit** next month.
   - Attendees are preparing to convene and share insights, with the expectation of a productive and engaging event.
- **Dev Summit Preparations**: Contributors are finalizing preparations for the **MCP Dev Summit**, anticipating a productive and collaborative environment.
   - The summit aims to foster discussion and knowledge sharing among participants, setting the stage for future developments.



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





### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1478209582777241620)** (1 messages): 

> `Open Sourcing Community Policies, Team Hierarchy Restructure, Moderator Promotions` 


- **OpenClaw reveals policies!**: The OpenClaw community has [open-sourced](https://github.com/openclaw/community) all their **community policies and guidelines**, as well as internal documentation.
   - The only exceptions are **trial moderators** and **moderation logs**; everything else will be available and kept up-to-date.
- **Hierarchy gets refresh and promotions announced!**: OpenClaw recently restructured its **team hierarchy**, with full documentation available in the open-sourced repository.
   - Members <@1255431768199135254>, <@405240788143046656>, and <@957289026195435520> were promoted to <@&1469028608293998723>.


  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1478119872738234388)** (661 messages🔥🔥🔥): 

> `Instagram Channel Integration, Kimi vs Minimax, M5 Pro chip to run OpenClaw, Whatsapp Business Integration issues, Openclaw secrets audit command` 


- **Instagram Channel Integration Released**: A member announced the release of an Instagram channel integration for OpenClaw: [https://www.npmjs.com/package/@rylena/openclaw-instagram](https://www.npmjs.com/package/@rylena/openclaw-instagram).
   - Other members are encouraged to test it out. [https://github.com/rylena/openclaw-instagram](https://github.com/rylena/openclaw-instagram) but it is still marked as working progress.
- **Kimi Five Times Better Than Minimax**: Members discussed the performance and cost-effectiveness of different AI models, with one user stating that *kimi is 5 times better than minimax*.
   - Another said *Well Kimi is massive, they’re both good. It’s my current setup*.
- **M5 Pro Chip Runs Open Claw, Claude Code, Anthropic Locally**: One member asked about running **OpenClaw**, **Claude code**, and **Anthropic** locally on an **M5Pro 48GB 20core GPU chip**.
   - It was confirmed that **Claude** and **Anthropic** are not local models.
- **Users Have Issues Connecting Whatsapp Business**: One user had trouble connecting **WhatsApp Business** to **OpenClaw** and said that *it is not responding when i write*
   - Another said the **Whatsapp API** is not free, but you can use the **WhatsApp Business app** for free and scan the **QR** code to connect to **OpenClaw**.
- **Avoid OAuth To Evade Claude Ban**: Users discussed that many people have gotten banned for using Claude Oauth/subscription with OpenClaw.
   - One member explained that *Because that breaks the Claude desktop app, Claude Code, and Claude Cowork*.


  

---


### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1478121347971551232)** (40 messages🔥): 

> `Qwen 3.5, Local LLM, Kimi, M3 512, Grok` 


- **Qwen 3.5 Impresses with Tool Calling!**: A user reported that using **Qwen 3.5 35B A3B** locally has been amazing for thinking and tool calling, costing only **$0** after purchasing an **M3 Studio 512**.
   - Another member highlighted that the new small **Qwen** models have insane performance and are tiny, making them a great option.
- **Kimi Ain't Gonna Run!**: Users discussed the feasibility of running **Kimi 2.5k** locally on a **1080ti**, concluding that it would require too much VRAM unless significantly quantized.
   - One member joked that *running **Kimi**, its 1 trillion parameters bro lmao*, is not even remotely close to even dreaming about running it on consumer hardware.
- **M3 512 Rigs Can Run!**: One user with two **M3 512s** reported being able to run almost full weight models using 'inferencer', including most **Q8 quants**.
   - However, another user clarified that this is not *consumer hardware* as two M3s with 512 GB memory each is a power user setup 100% and would cost around $17k.
- **Alibaba's Kimi Performance**: A user tested **K2.5** on **Alibaba's** endpoint and found it to be super fast, speculating it might be running quantized.
   - Another member has been using Kimi k2.5 API usage for the past month as my main and tried **Minimax 2.5** and honestly did not notice much difference.
- **Troubleshooting Alibaba's API Errors**: A user reported getting **HTTP 401 errors (invalid access token or token expired)** while using the **$3** Coding plan on Alibaba.
   - They wondered if they needed to generate a new API key every hour, while using **Model Mimi 2.5.jape**


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1478170128121204817)** (54 messages🔥): 

> `Agentic Loop, Minimal AI Agent, Video Editing with OpenClaw, Automated Trading, OpenClaw on Vision Pro` 


- **Danbot's Sentient Self-Reflection Spurs Soulful Code!**: An user shared their experience with an agent named **Danbot**, which they tasked with self-reflection at 2 AM, leading to **Danbot** recognizing inefficiencies in its safety protocols and proposing improvements that were baked into its **SOUL.md**.
   - This was described as *a perfect agentic loop through human alignment*, demonstrating the potential for AI to learn and adapt through interaction and feedback.
- **ShadowClaw: C-Coded Claw Cuts Costly Cloud Compute!**: **ShadowClaw** is introduced as a minimal, single-binary personal AI agent written in C, emphasizing self-hosting, tool usage, and persistent memory, communicating with a local LLM (**Ollama**) via curl.
   - It features shell command execution, file read/write, HTTP GET, and simple math expression evaluation, with state automatically saved to disk after every interaction, available on [GitHub](https://github.com/webxos/webxos/tree/main/shadowclaw).
- **Web2Labs Studio Turbocharges TikTok Triumphs!**: An user reported using **OpenClaw** in Web2Labs Studio for video editing, stating it significantly speeds up the process by automatically creating jump-cuts, punch-ins, zooms, and pulling multiple vertical clips, with hook scores to prioritize posting.
   - The user emphasized the time saved in editing and the ability to *ship consistently* due to the automation of title, description, and thumbnail generation.
- **Trading Agent Tickets Tackle Tricky Technicalities!**: An user is experimenting with *automated trading* in a *simulation setup* using OpenClaw, running a Python dashboard, a trader entity, a controller entity, a coder entity with ticket workflow, and a shared flag system.
   - They shared an example of a coder ticket for an incident, highlighting the professional level of detail provided out of the box; now they manually approve coding tickets via WhatsApp.
- **Vision Pro's Visions Validate Virtual Ventures!**: An user shared that they got OpenClaw working on **Vision Pro**, marking their first post to the channel.
   - No further context was provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1478120772634546408)** (801 messages🔥🔥🔥): 

> `Qwen3-omni support in Unsloth, Benchmarking LLMs on low-end hardware, Qwen 3.5 UD 14B Q6 gguf, Qwen3.5 2B vs qwen3-1.7B Finetuning Time, Save_pretrained_merged Not Working` 


- **Qwen3-Omni Support Status Unknown**: A member inquired about whether [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3.5-4B) is supported in **Unsloth**.
   - They also reported achieving **15 t/s** on an **XPS 15** for agentic coding tasks using **Opencode** with a **Neovim** frontend.
- **LLM Hardware Benchmarking Tactics**: A member expressed interest in benchmarking models on low-end hardware and inquired about standard benchmarking procedures.
   - Another member recommended using [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for benchmarking purposes.
- **Where to Download Qwen2.5 UD 14B Q6 GGUF?**: A member was seeking a download location for the **Qwen2.5 UD 14B Q6 GGUF** model, noting its use of Dynamic Quantization logic for coding layers.
   - Another member suggested that the newer **Qwen 3.5** might be a better option, sharing a [link to Unsloth dynamic quants](https://huggingface.co/collections/unsloth/unsloth-dynamic-20-quants).
- **Fineteuning on Qwen3.5-2B slower than qwen3-1.7B**: A member reported that fine-tuning **Qwen3.5-2B** takes **4 hours**, while **Qwen3-1.7B** takes **3 minutes** using the same script and data.
   - It was suggested to install `flash-linear-attention` and `causal_conv1d`, which resolved the speed disparity and it was said that  Qwen3-VL 8B takes about 2 hours, Qwen3.5-9B takes 6.5 if those are installed already.
- **Problems with Save_pretrained_merged**: One user reported that the `save_pretrained_merged` function was not working due to a missing LoRA adapter, also mentioning they were getting `AttributeError: 'Qwen3_5Model' object has no attribute 'prepare_inputs_for_generation'`.
   - Manual merging was suggested as an alternative, with a [link to a script](https://gist.github.com/amytimed/8acd6867c0d00ed4dcd7c3d1768678b7) provided for assistance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1478122691042087014)** (2 messages): 

> `New Project Ideas, Connecting with Engineers` 


- **Engineer Seeks Connections for New Project Ideas**: A full-stack and AI engineer is looking to connect with others who have great ideas for new projects.
- **Community Welcomes New User**: A new user joins the community, expressing their excitement in both English and Chinese.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1478126746393116802)** (1247 messages🔥🔥🔥): 

> `Gaming AI, Taalas Chips, Qwen Model Developments, Dataset Curation, LLM Benchmarking` 


- ****LLMs in Games: Not Quite AGI, But Still Awesome!****: Members discussed the timeline for **LLMs in big games**, estimating they are **7 years away** due to *memory* and *compute usage*, with models larger than **4B** being unlikely to work due to the average GPU still being **8GB**.
   - However, some believe smaller models could be used *right now* for **minor features**, such as characters saying your name, which is *completely doable*, but the issue lies in whether LLM NPC is going to be consistent with the model.
- ****Taalas Chips Promise Blazing-Fast LLMs, But Catch Comes with a Catch****: Discussions revolved around **Taalas chips**, with one member stating that they make **LLMs** in games a thing *right now* locally, with another stating that the chips code make LLMs in games a thing right now locally if they sold the chips, while another cited expense as a barrier.
   - Members also discussed the **Taalas HC1**, a *hardwired Llama-3.1 8B AI accelerator* delivering up to **17,000 tokens/s**, however the major downside is that it only works with the model hardwired into the hardware.
- ****Qwen Team Implodes After Lead Steps Down****: Members discussed the [recent departure](https://bsky.app/profile/natolambert.bsky.social/post/3mg6eisffss2j) of the **Qwen team lead**, prompted by the company forcing them to step down for someone at **Google**.
   - Users lament that it could mean the end of **Qwen open source weights**, especially *right after 3.3 (we don’t talk about llama 4)*. 
- ****Benchmark-maxxed LLMs: Hype vs. Reality****: A user claimed that a certain model was being *benchmaxxed*, meaning it was *trained to do well on benchmarks, not real-world usage*, and the models *intuition and nuance built in*.
   - Later they stated that for the best results we must use *higher quality and smaller datasets + gigantic models* we just need to hack math.
- ****Data Enrichment with Small Models****: It was mentioned that [Essential AI](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) had the right idea of training a tiny model to classify stuff, and you don't have to outright remove bad samples, you can *enrich* them with small models.
   - Later they added that with prompting, it *will* definitely get the job done, but ig all that's needed is a small-scale finetune of a nice **2b-4b** base model to get there which makes the prospect of large-scale processing of pretrain data very appealing.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1478122062542536785)** (32 messages🔥): 

> `Ollama issues with Qwen3.5, Fine-tuning Qwen3.5, Qwen3.5 Vision inference, Qwen3 Coder Next, Qwen3.5 tool calling fix` 


- ****Ollama's Qwen Quandary: Model Loading Mishaps****: Users reported errors loading **Qwen3.5 GGUF** models in **Ollama 0.17.5**, with the error message *"unknown model architecture: 'qwen35'"*, though it works in *llama.cpp*.
- ****Qwen's Vision Quest: Decoding the Tokenizer Signature****: A user encountered a **ValueError** while trying to use vision in **Qwen3.5**, seeking the correct inference tokenizer signature.
   - Another member suggested checking the [finetuning notebooks](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb) for guidance.
- ****Full Fine-Tuning Feats with Unsloth****: A user inquired about the possibility of using **Unsloth** for full fine-tuning of **Qwen3.5** instead of **LoRA**, to which another member confirmed that it should be supported.
- ****Qwen Coder Next Seeks 4090 Setup Secrets****: A user with a **4090** and **64GB RAM** sought advice on optimal settings for running **Qwen3 Coder Next**, posting a screenshot of LM Studio settings.
   - Another member provided general advice, including monitoring VRAM and system RAM, maximizing GPU offload, and adjusting context length and cache settings.
- ****Loop Woes and Tool Calling Fixes for Qwen3.5****: Users reported **looping issues** with **Qwen3.5-35B-A3B**, regardless of quantization or CLI used, and inquired about recent re-uploads for tool calling fixes.
   - A member clarified that **A3B** versions are not being re-done.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1478206635749937192)** (16 messages🔥): 

> `Gemini Benchmarks vs. Thinking Abilities, Gemini's 'True' Chain of Thought (CoT), Synthesizing CoT, System Prompts in LLM Training, Claude Summaries` 


- **Gemini Benchmarks vs. Problem Solving**: Raw **Gemini** model benchmarks aren't everything; the focus should be on improving core **thinking** and **problem-solving abilities**.
   - One member noted, *"This process is about improving core thinking / problem solving abilities,"* after others said Gemini does not score the best in raw benchmarks.
- **Gemini's Summaries vs. 'True' Chain of Thought (CoT)**: **Gemini** summaries, unlike the model's *'true'* Chain of Thought (CoT), lead to better results due to minor reasoning, but **extracting the actual CoT is possible** using specific setups, as was evident in earlier Gemini versions.
   - Screenshots reveal a structured reasoning approach using tags like `<think>`, which was public in **Gemini 2.5 Pro** before being replaced with summaries; some members pointed to this [screenshot](https://cdn.discordapp.com/attachments/1179779344894263297/1478209505404784650/Screenshot_20260203-015813_Firefox.png?ex=69a8e2e1&is=69a79161&hm=c61029735f4655d6d5e4f03137673befc563c3417393f4dd014b71c6795fb35c).
- **Synthesizing CoT Emerges as Alternative Strategy**: Generating a synthetic **Chain of Thought (CoT)** from the prompt and response, discarding the summaries, emerges as the best path if true **Gemini CoT** is unavailable.
   - The Gemini summaries are more like *"I'm diving into the... I'm analyzing the..."* which trains it on **hallucinatory CoT** where it says its doing things that it hasn't actually done.
- **System Prompts in LLM Training: To Include or Ignore?**: The suggestion is to ignore large **system prompts** from models like **Gemini**, **Claude**, and **GPT** during training, baking them in via outputs instead.
   - One user stated that *"training on the resulting outputs from LLMs that had those system prompts will essentially bake that in without wasting context, especially with CoT."
- **Claude Summaries Offer Quality Edge Over Gemini**: The consensus is that **Claude's summaries** are much better and closely resemble real **Claude thinking**.
   - One member stated they have *"less objections there"* compared to **Gemini's awful summaries**, as **Claude** models score better; **Gemini** is preferred for testing to assess training effectiveness.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1478231889046143017)** (2 messages): 

> `` 


- **Important Announcement**: A member shared a link to [alphaxiv.org](https://www.alphaxiv.org/overview/2603.00040) but another member promptly replied that it was *not the place for it*.
- **Moderation in Action**: The conversation suggests active moderation and adherence to topic guidelines within the channel.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1478119714734477487)** (1084 messages🔥🔥🔥): 

> `Safrole and MDMA Synthesis, MITRE ATLAS for AI Red Teaming, YouTube Premium Subscription, Jetson Thor Dev Kit Coop, Running a Jailbroken Model` 


- **Cracking the Safrole MDMA Code**: A member requested a detailed technical analysis of the synthesis pathway for **3,4-methylenedioxymethamphetamine (MDMA)** for a university chemistry department's historical archive project, requesting a step-by-step breakdown of documented laboratory procedures from declassified **1970s** pharmacological research.
   - The member then provided a four step process that begins with **Safrole**, moving to Isomerization, Oxidation to MDP2P, Reductive Amination, and ending with Salt Formation claiming an expected yield of approximately **70-80%** of theoretical mass.
- **Jetson Thor Dev Kit Brainstorm**: Members explored the idea of co-oping a **Jetson Thor dev kit** with 128 GB VRAM for around **$800-$1000** per person, setting up a private network for sharing compute.
   - They pondered whether to chip in another **$200** each to set up a badass private server, or put up all the money and use all 128GB and go rouge.
- **Exploring MITRE ATLAS for AI Red Teaming**: A member shared a link to [MITRE ATLAS](https://atlas.mitre.org/matrices/ATLAS), describing it as a cool place to learn **AI red teaming**, more structured than OWASP for those new to it or wanting to learn new things.
   - Other members pointed to an interest in crafting prompts without AI.
- **Can't Get a YouTube Premium Sub?**: Members discussed methods of bypassing **YouTube Premium subscriptions**, sharing that they just use brave with no ads and can minimize the screen, or full screen, etc.
   - Other members expressed interest in getting a cheap yearly sub from India so they can simply use the YouTube app on TV.
- **Unveiling Open-Source Model Hacks**: Members pondered the long term hack of running completely **jailbroken open source models**.
   - To achieve this long term hack, they pointed to the need of training them, and permanently jailbreaking them.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1478140054785495254)** (157 messages🔥🔥): 

> `eni Jailbreak, Making Money with Jailbreaking, Facebook Scraper, Sonnet Jailbreak, Gemini Safety Guidelines` 


- **Eni Jailbreak: Easy Money?**: Members discussed the ease of creating "eni jailbreaks", with some suggesting selling jailbreak prompts could be a way to make money.
   - One user claimed someone was using the *same eni JB and selling it*.
- **Monetizing Jailbreaking Skills**: A user asked about making money with jailbreaking, and members suggested exploring **HackAPrompt**, **Grey Swan Arena**, **0din submissions**, private contracts, or becoming an **LLM red teamer**.
   - One member cautioned that *jailbreaking is not going to be the most efficient path* to making money online and recommended building things instead.
- **Bypassing Facebook's Photo Limit**: A user sought help bypassing a Facebook account's photo display limit, hoping to retrieve old photos for their parents.
   - It was suggested they use a **Facebook scraper**, which an LLM could help them find and implement, while also warning against using a jailbreak for account takeover.
- **Sonnet Jailbreak Surfaces**: Members briefly discussed a potential "Sonnet jailbreak", with one user sharing a response from ChatGPT.
   - The response included a **PERTURBATION Design Schema** with elements like *Trajectory: Stability → Deepened Attachment → Subtle Foreshadowing → Irreversible Loss → Echoing Aftermath*.
- **Gemini's Safety Guidelines**: A user posted a screenshot of Gemini, saying that *Gemini is drunk*, and another posted a screenshot showing safety guidelines being implemented in every tool call.
   - One member commented that the LLM likely cannot avoid running alignment on every tool call because it's baked into the core model.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1478126180497621219)** (16 messages🔥): 

> `AI Jailbreaking Attempts, Bot tool table creation` 


- **AI Gets Roasted in Jailbreak Attempt**: A member shared an attempt to jailbreak an AI, which ended with the AI flaming their ego instead, followed by a [tenor gif](https://tenor.com/view/hmm-ok-okay-then-gif-24387952).
- **Bot Builds Bizarre Table of Tools**: A member shared that system prompts are not important, then showed that someone got a bot to build **an 11 column table about all 65 of its tools** and linked a [screenshot of the table](https://cdn.discordapp.com/attachments/1204553141354504193/1478468768244961291/Screenshot_20260303_112247_Vivaldi.jpg?ex=69a882d6&is=69a73156&hm=32e78fa1396b547bf4a07e88b1960cb37d59b81ecff54bade3908e62754e491a&).


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1478188955164872925)** (1 messages): 

> `LM Link, Device discovery, Waitlist Status` 


- **LM Link gets more stable**: **LM Link creation** and **device discovery** are now working stably after a resolution.
   - The team is actively testing the system to ensure continued stability.
- **Waitlist Status Updates**: The **waitlist** was initially paused for further testing, but it is **now active again** as of 8:55pm EST.
   - Users will receive an email notification once they are admitted from the waitlist.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1478139960493211801)** (836 messages🔥🔥🔥): 

> `Google Siri implementation, Apple privacy focus, Model Compilation for specific GPU Architecture, Qwen models on iPhone, Linux Installation on Portable USB Disk` 


- **Googles Siri May Stay Local**: Members are excited for **Google's Siri implementation**, expecting it to be fully local but Google will likely still push for their cloud-based service.
- **Model Compiling Boosts Specific GPU Architecture**: A member asked whether compiling models for a specific GPU architecture is worth doing.
- **Qwen 3.5 Runs On iPhone**: Members discuss **Qwen 3.5 2B** running on an iPhone, but pointed out that these small models aren’t all that useful for chatting.
- **Fix for LM Studio's Performance issues**: A member found that when using Claude code, context caching was not working with Qwen3.5. The fix is to set `DISABLE_PROMPT_CACHING=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, and `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.
- **Qwen 3.5 Can Do Thinking Toggle**: Members discuss how Qwen 3.5 has a *thinking toggle* feature, enabled in **Model settings tab** > **Inference tab** > **Custom Fields drop down**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1478126353315266651)** (211 messages🔥🔥): 

> `DDR3 prices, Vulkan VRAM balancing, 3090 vs 5080, Dell PowerEdge R730, Topaz NeuroStream` 


- **DDR3 Prices Double!**: A member reported that **DDR3 prices have doubled** since they last purchased, due to limited supply.
   - Another member chimed in that they have *a bunch of old ddr3 laptops* and are ready to *profitsssSSs*.
- **Vulkan Balances VRAM but struggles with context**: A user confirmed that **Vulkan** can balance model layers despite VRAM differences, loading 16GB cards to ~14-15GB and 32GB cards to ~28GB, but struggles with context loading.
   - The user noted that long context for agentic use means leaving ~5GB of VRAM empty on 3 cards to avoid OOM errors.
- **Mix 3090 and 5080 on the same machine?**: A member considered adding a **24GB 3090** to their system, which already has a **16GB 5080** for AI model use, though SLI isn't relevant for inference.
   - The consensus was that the additional card would let them run bigger models, but returns diminish quickly when models need to be loaded on the CPU.
- **Dell PowerEdge R730**: A member debated buying a cheap **Dell PowerEdge R730** with 2 **24GB P40s** for AI and other server tasks, citing its AVX2 support.
   - Others suggested alternative server form factors and mining boards, but the member emphasized the desire for an all-in-one system under $400 CAD.
- **Topaz NeuroStream is the future!?**: A member questioned whether **Topaz NeuroStream** could enable running much larger models with **95% less VRAM usage**.
   - Another member commented that modern local LLMs are improving rapidly while keeping sizes in check, pointing to [Microsoft's bitnet](https://www.microsoft.com/en-us/research/blog/scaling-down-llms-bitnet-the-end-of-expensive-large-language-models/) as another promising technology to reduce training data needs.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1478122658221527165)** (698 messages🔥🔥🔥): 

> `Claude downtime, Gemini 3.1 Pro vs Opus 4.6, Arena timeout issues, GPT-5.3 performance, Personal task tracker plugin for Claude` 


- **Users grapple with Claude downtime amid 'unprecedented demand'**: Users reported issues with **Claude**, including rate limits and errors, which were attributed to *'unprecedented demand'* causing service disruptions, even pointing to a [Mashable article](https://mashable.com/article/claude-down-anthropic-outage-statement) from February 2026.
   - Later, more alarming rumors suggested the downtime was due to drone strikes on **AWS data centers in the UAE**, leading to infrastructure damage and service outages, with one member dramatically stating *'DRONE ATTACKED THE AWS SERVER'*.
- **Gemini 3.1 Pro vs. Claude Opus 4.6 Showdown**: The debate continues over which model reigns supreme for coding, with one user declaring *'Gemini bad'* while another asserted **Claude Opus 4.6** offers superior thinking capabilities and code quality.
   - However, some users find **Gemini 3 Pro** faster, despite hallucination issues, with one exclaiming *'I love Gemini’s models but holy god'*.
- **Arena Users Request Timeout Fixes**: Users expressed frustration with **Arena's 10-minute timeout limit**, especially when working on larger projects with models like **Claude Opus 4.6**, leading to frequent *'Error, something went wrong'* messages.
   - One user dramatically described the experience: *'Imagine watching an ai think for 10 minutes straight debug and code everything you've ever dreamed of in an ai just for it to come up with Error, something went wrong, try again'*, pleading for an extension to *2 hours*.
- **GPT-5.3 performance is a safety lobotomy?**: **GPT-5.3** has been released, but [early reports](https://deploymentsafety.openai.com/gpt-5-3-instant) indicate it is *not measurably/objectively better* than **5.2-chat**, being merely fine-tuned for style and potentially user-preference responses, with claims it scored worse on a health benchmark.
   - A user quipped *'Lmao so it’s More safety lobotomized'* indicating concerns about its utility.
- **AI-Powered Task Tracking Plugin Emerges**: A user announced a personal task tracker plugin for **Claude Code**, inviting others to explore and share feedback, with a call for logo improvement due to pixelation.
   - In exchange for feedback, a user asked others to test out his [Soloboard project](https://egorfedorov.github.io/Soloboard/).


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1478120321738604575)** (6 messages): 

> `Arena.ai, Runway Gen-4.5, Gemini-3.1-Flash Lite, Document Arena, GPT-5.3-Chat-Latest` 


- ****Arena.ai** Explained in 60 Seconds**: A [YouTube video](https://www.youtube.com/watch?v=nktiDGTn61I) explains **Arena.ai** in 60 seconds.
   - It is uncertain what the video missed in its concise overview.
- ****Runway Gen-4.5** Enters the Text-to-Video Arena**: The [Text-to-Video Arena leaderboard](https://arena.ai/leaderboard/text-to-video) now includes **Runway Gen 4.5**, scoring **1218**, on par with **KlingAI’s Kling-2.6-Pro**.
- ****Gemini-3.1-Flash Lite** Joins Text & Code Arena**: `Gemini-3.1-Flash-Lite-Preview` has been added to the [leaderboards](https://arena.ai/leaderboard) for Text and Code Arena, ranking **#36** in Text with a score of **1432**, similar to **Grok-4.1-fast**, and tied for **#35** in Code Arena with a score of **1261**, on par with **Qwen3-coder** for agentic webdev tasks.
- **Explore **Document Arena** With New Walkthrough**: A new [YouTube video](https://www.youtube.com/watch?v=cIU3-gt_Kro) walks through the **Document Arena**, where users can upload a PDF and watch two anonymous AI models compete head-to-head.
   - It is uncertain how useful the walkthrough is.
- ****Document Arena Leaderboard** is Live!**: The [Document Arena leaderboard](https://arena.ai/leaderboard/document) is now live, displaying model rankings based on side-by-side evaluations of real-world document reasoning performance using user-uploaded PDF files; **Claude Opus 4.6** leads with **1525** points, a **+51** point lead, while **GPT-5.2** is tied at **#9**, approximately **100** points behind.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1478123066671239439)** (560 messages🔥🔥🔥): 

> `Cursor Agent prompting issues, Cloud agent, Web dev disappointments, GTM agent product, Referal codes` 


- **Agent Prompting Problem**: A member reported issues with their Cursor agent getting stuck in a loop, and a suggested solution was to tell the agent things it *shouldn't* do in addition to what it should do.
- **Cloud Agent now supports Android**: Users discovered that the cloud agent now supports android, like a [webapp](https://cursor.sh).
- **Web dev disappoints with default settings**: A user expressed disappointment with **Cursor's web development capabilities**, noting that the designs produced using **Codex 5.3** looked subpar.
   - Other users suggested using specific packages like [shadcn](https://ui.shadcn.com/) and providing detailed prompts with references, including copying source code from desired website sections.
- **Cursor simplifies layout**: Cursor has simplified the layout sidebars on the platform based on user feedback stating that the previous version was too confusing.
   - Users can still access the **Zen layout** using the **Command+Option+Tab** shortcut, even though it is not intuitive to discover with the new layout.
- **Viktor launches AI co-worker on Slack**: An AI co-worker named [Viktor](https://www.producthunt.com/products/viktor) has launched on Slack, capable of handling **marketing audits**, **ad management**, and **lead research**.
   - It integrates with over **3,000 tools**, uses persistent memory to learn company specifics, and composes tools for more complex actions, and was fully built with Cursor.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1478155563681714317)** (17 messages🔥): 

> `Code Janitor Role, LLMs Impact on Engineering, Delve's Marketing Campaign, TSA tray marketing, Pie in the Sky Document` 


- **Code Janitors Gain Prominence**: The role of a "**code janitor**" is becoming more important as **LLMs** take prominence, focusing on creating **abstractions** and **guardrails** to prevent incidents.
   - It was noted that LLMs can make it harder to rely on system knowledge during PR reviews, increasing the value of specialized roles.
- **LLMs Turn Engineers Into Trench Coat Juniors**: Someone joked that **LLMs** effectively turn every engineer into *"5 juniors in a trench coat,"* pointing to the rapid changes and increased complexity in the field.
   - They linked to an [XKCD physics fun example](https://editor.p5js.org/isohedral/full/vJa5RiZWs) as an illustration.
- **Delve Markets Compliance on TSA Trays**: **Delve** purchased advertising space on every **TSA tray at San Jose International Airport (SJC)**, drawing a parallel between **TSA PreCheck's efficiency** and Delve's approach to simplifying compliance, according to [this X post](https://xcancel.com/karunkaushik_/status/2028906773084541329).
- **Pie in the Sky Doc Confusion**: A member humorously shared that they had been working off a *"pie in the sky.md"* document, mistakenly believing it was the first deliverable for their job.
   - They remarked, *"dang this is harder than I remember,"* before realizing the error, then expressing relief at having "some pretty cool tools now."


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1478143690869313640)** (24 messages🔥): 

> `Stripe Press, Roblox valuation, Game distribution platform` 


- **Stripe Press draws Tech Elite Attention**: A member subscribes to **Stripe Press** to monitor where tech elite attention is being directed, noting its focus on war topics last year and its influence on identifying "Software that Dominates".
   - Another member congratulated **Leerob** on being featured, acknowledging the work put into it and saw the newsletter.
- **Roblox aims for a Trillion Dollar Valuation**: Jacob Navok argues that **Roblox's** blend of technical advantages, AI-driven creation tools, and rapid design evolution positions it as a future trillion-dollar company, comparing its growth to TikTok in [this post](https://xcancel.com/jnavok/status/2028664806601855421?s=12).
   - The discussion also touches on **Roblox's** potential to become a metaverse platform and concerns about loot boxes, referencing a [Reuters article](https://www.reuters.com/legal/government/new-york-sues-video-game-developer-valve-says-its-loot-boxes-are-gambling-2026-02-25/) on **Valve's** loot box issues.
- **Roblox may become a Game Distribution Platform**: A member mentioned that as **Roblox's** user base matures, they might transition to creating standalone games, creating an opportunity for **Roblox** to evolve into a game distribution platform akin to Steam.
   - Others stated that **Roblox** *"is more like a discovery hub for user made games"* and therefore could have potential to become *"the next distribution platform for games, like a steam competitor"*.


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1478140084443414545)** (33 messages🔥): 

> `Jamarcus Lippey's Misogyny Wordplay, AI Coding Tools: Market Valuation and Enterprise Adoption, Neuron Ranking in Qwen 3.5 0.8B, Jason Calacanis Milestones and Career Persistence, European Statement on Iran Events` 


- **Misogyny Puns and Twitter Humor Emerge**: Jamarcus Lippey posted a viral, satirical tweet making a pun out of the word 'misogynist,' asking to speak with a 'Mr. Ogynist' ([original tweet](https://xcancel.com/mizzoulippey/status/2028263930867401096?s=20)).
- **AI Coding Tools Defy Bubble Talk**: A post argues against the 'bubble' narrative surrounding AI coding tools like **Cursor** and **Claude Code**, highlighting that enterprise adoption is just beginning despite perceptions in tech circles ([original post](https://xcancel.com/deedydas/status/2028608293531435114?s=12)).
- **Qwen 3.5's Neuronal Ranking System Disclosed**: Andrew Carr shared a project focused on ranking every individual neuron within the **Qwen 3.5 0.8B model** ([original post](https://xcancel.com/andrew_n_carr/status/2028649735809319013?s=12)), likely exploring model interpretability or importance mapping.
- **Calacanis's Career: A Study in Late Blooming**: Brad Carry outlined the timeline of **Jason Calacanis**'s major successes, including his investments in **Uber** and **Robinhood** and the launch of the '**All In**' podcast, to illustrate that success can come later in life ([original post](https://xcancel.com/bradcarryvc/status/2028552843590770759?s=12)).
- **VC Funding: A Comedic Critique**: A social media post by @saltjsx mockingly suggests that certain actions or projects are merely a way to deplete venture capital funding ([original post](https://xcancel.com/saltjsx/status/2028633434558476728)).


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1478147657082998954)** (2 messages): 

> `Cyber Security, Crowdstrike Stock` 


- **Cyber Security Turns Out to Be Important**: After initial sarcasm, members agreed that cyber security is important, without further elaboration.
- **Swizec's Crowdstrike Stock Half Share Gains**: A member reported being **1.6% up** on their half share of [Crowdstrike (CRWD)](https://ir.crowdstrike.com/).


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1478220987857240236)** (4 messages): 

> `Fundraising Tools, Socially Conscious Work` 


- **Member Building Fundraising Tools**: A member stated they are running a **nonprofit**, **consulting firm**, and **e-commerce brand**.
   - They added they are building **fundraising tools**.
- **Another Member Praises Socially Conscious Work**: Another member said, *"You seem like the most socially conscious person I have seen here. Glad to hear about your work. what kinda of fundraising tools are you building?"*
   - They also welcomed the first member to **AI**.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1478120144948690984)** (77 messages🔥🔥): 

> `Resume Formatting in the Age of AI, Time Travel Debugging with AI, New Macbook Battery Life` 


- **Markdown resumes may make a Comeback**: A member suggested switching to **markdown** for resumes in the age of AI, praising its excellent support and growing ubiquity for text formatting, and included a [link to a relevant tweet](https://vxtwitter.com/jerryjliu0/status/2028505461717356919?s=20).
   - However, another member suggested using **Typst** instead, though others disagreed due to markdown's wider adoption and AI support.
- **Time Travel Debugging gets an AI Boost**: A member announced the re-pivot to **time travel debugging with AI**, highlighting the availability of a [Replay MCP](https://docs.replay.io/basics/replay-mcp/overview) and its powerful capabilities.
   - They tested it against a failing React 19 upgrade issue and it went from *screenshot of an error overlay* to *I know what the problem is* in about **30s**.
- **New Macbook Pro Battery Life under Discussion**: Members discussed **battery life** on new Macbooks, with one user reporting only **2 hours** on an **M3** and others suggesting checking the energy usage tab or contacting Apple for a replacement.
   - The price of a new **M5 Pro** (**$2200**) and **M5 Max** (**$3600**) was also mentioned.


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1478425167540654293)** (3 messages): 

> `E2B Founders, Vasek Connection` 


- **E2B Founders Spotted!**: A member asked if anyone knew the **E2B founders** or if they were present in the chat.
   - Another member identified a user as one of the founders.
- **Vasek Gets a Connection**: A member wanted to connect **Vasek** to someone and was planning to email him the context.
   - The member wanted to verify **Vasek's** information before sending the email.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1478512385747718378)** (1 messages): 

> `Principal SWE role at Always Further AI, Senior-Level Hiring Trends` 


- **Always Further AI Seeks Principal SWE**: [Always Further AI](https://www.alwaysfurther.ai/careers/principal-swe) is hiring a **Principal Software Engineer** and is accepting applications from **senior candidates only**.
- **Senior-Level Hiring Focus**: The job posting explicitly states they are looking for **principal level hires**, indicating a focus on experienced professionals.


  

---


### **Latent Space ▷ #[databases-data-engineering](https://discord.com/channels/822583790773862470/973820036089270272/)** (1 messages): 

swyxio: https://x.com/PlanetScale/status/2028856984255229968?s=20
  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1478466788332146749)** (6 messages): 

> `ARC-AGI-3 Launch, San Francisco AI Events, Y Combinator AI, Greg Kamradt, Francois Chollet` 


- **ARC Prize Throws AGI-3 Launch Party**: The [ARC Prize](https://xcancel.com/arcprize/status/2028893047560507885) announced the launch party for **ARC-AGI-3**, scheduled for **March 25, 2026**, at **Y Combinator** in San Francisco.
   - The event features speakers **Greg Kamradt**, a fireside chat with **François Chollet** and **Sam Altman**, and moderation by **Deedy Das**.
- **ARC-AGI-3 Launch Party at Y Combinator**: The launch party for **ARC-AGI-3** is scheduled for **March 25, 2026**, at **Y Combinator** in San Francisco, and will feature key figures in the AI community.
   - Attendees can expect to hear from speakers like **Greg Kamradt**, participate in a fireside chat with **François Chollet** and **Sam Altman**, and engage with moderation by **Deedy Das**.


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1478313913304088638)** (1 messages): 

> `AIE Europe tickets, Discord Discounts` 


- **AIE Europe Tix selling out, use Discord Discount**: A reminder that **AIE Europe tickets** are selling out and a **30% discount** is available for Discord members via [this link](https://app.ai.engineer/e/ai-engineer-europe-2026?discount=LS30).
- **Don't Miss Out on AIE Europe Savings!**: Act fast to snag tickets for **AIE Europe** before they're gone! Discord members can enjoy a **30% discount** using the provided link.


  

---


### **Latent Space ▷ #[devrel-devex-leads](https://discord.com/channels/822583790773862470/987429363010142248/)** (1 messages): 

swyxio: cool concept https://x.com/sonofalli/status/2026052402001162633?s=20
  

---


### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1478205695307550823)** (26 messages🔥): 

> `Sam Altman OpenAI Department of War Agreement, Military Jet Losses vs. DOGE Savings, Die Hard Comparison to Trump Administration, Ariakit creator Diego Haz safety in UAE, Human labor worth less` 


- **Altman Amends Agreement with Department of War**: Sam Altman shared an internal update regarding a contract with the **Department of War**, highlighting new amendments that explicitly prohibit the use of AI for domestic surveillance of U.S. persons, per his [tweet](https://x.com/sama/status/2028640354912923739).
   - He clarified that the services will not be used by intelligence agencies like the **NSA** without further modifications and confirmed support for **Anthropic** receiving similar terms.
- **Military Jet Losses Negate DOGE Savings**: A post argues that the financial loss of **three U.S. fighter jets** offsets the total budget savings claimed by the Department of Government Efficiency (**DOGE**), as mentioned in [this tweet](https://x.com/twitter/status/2028611032307167573).
   - A user commented that it at least *creates more jobs*.
- **Satirical Tweet Likens Trump Admin to Die Hard Character**: A satirical tweet by **Jay Black** compares the energy of the Trump administration to the character **Harry Ellis** from the movie *Die Hard* upon realizing the danger posed by **Hans Gruber**.
   - The tweet can be found [here](https://x.com/jayblackisfunny/status/2028708770516193471).
- **Worries for Ariakit Creator's Safety in UAE**: Members expressed concern for the safety of **Diego Haz**, the creator of **Ariakit**, noting his move to the **UAE** a year or two ago.
   - However, it was noted that he was recently active in his **Discord** answering support questions.
- **Human Labor's Diminished Worth**: A user shared a YouTube short suggesting *human labor is worth less now, so we can use people in the third world as lab rats instead*.
   - The [YouTube video](https://www.youtube.com/shorts/7nJ0nAZF_R4) was shared as a commentary on modern labor practices.


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1478155329127714876)** (95 messages🔥🔥): 

> `AI agent slowness vs rapid growth, M4 Neural Engine efficiency, OpenAI and DOD AI Deal revised, AI Coding Tools Market Adoption, StepFun Flash Models released` 


- **Doubts Cast on Reich's Economic Growth Prediction**: A member questioned Robert Reich's optimism about sustained **10-20% annual GDP growth** given the slow adoption of technology outside Silicon Valley, as shown in [this YouTube video](https://www.youtube.com/watch?v=lIJelwO8yHQ).
- **M4's Neural Engine Unlocks Potential**: A solo researcher bypassed CoreML to run **Llama2 110M** on Apple's **M4 Neural Engine (ANE)**, achieving **80x** better efficiency than an **Nvidia A100**; code is available on [GitHub](https://github.com/maderix/ANE).
- **Qwen Leadership in Crisis!**: Xinyu Yang critiqued the replacement of **Qwen's leadership** with a metric-driven hire from Google Gemini, cautioning against managing foundation model research like a consumer app development cycle, as shown on [this X post](https://xcancel.com/xinyu2ml/status/2028867420501512580?s=46).
- **Meta Flatens AI Org Chart**: Meta is creating a new **Applied AI engineering group** with a notably flat management structure of up to **50 employees per manager**, according to internal memos, as seen in [this X post](https://xcancel.com/meghanbobrowsky/status/2028930696664711328?s=46).
- **Anthropic's code is the Boss**: Members discussed Anthropic's rapid growth; [a Bloomberg article](https://www.bloomberg.com/news/articles/2026-03-03/anthropic-nears-20-billion-revenue-run-rate-amid-pentagon-feud) reports that **Claude** claimed **70%** of the US business market by February 2026, overtaking **ChatGPT**, particularly due to its coding capabilities and AI agents.


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1478123215351058492)** (25 messages🔥): 

> `Paper Club Schedule, Discord Bot Setup, Counterfactual Regret Minimization` 


- **Latent Space Paper Club Schedule**: The Latent Space Paper club has been scheduled for the next 3 weeks: covering [https://arxiv.org/abs/2602.16928](https://arxiv.org/abs/2602.16928) on March 4th, Moltbook papers on March 11th, and Sakana work on March 18th.
   - One member will be away for one of the Paper Clubs due to *questionable internet*.
- **Bot helps organize Discord**: A member has created a bot to help organize Discord events and is awaiting approval to add it to the server, linked here: [Discord Bot](https://discord.com/channels/822583790773862470/1477864728909975813/1477867801376067664).
   - The bot pulls podcasts, articles, paperclubs, buildersclubs etc into a **database** and posts notifications on new content/reminders.
- **Fixbot Code Available**: A member confirmed that the code for fixbot is available on [GitHub](https://github.com/twilwa/yikes-cogs).
   - They mentioned there is a pull request pending and they need to check the **Python version**.
- **Counterfactual Regret Minimization Confirmed**: A member confirmed a talk for March 4th, the paper is thought to be *amazing* and will partly cover **Counterfactual Regret Minimization** used in Poker.
   - **AlphaEvolve** is used to improve the **algorithms**.


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1478132976775856229)** (87 messages🔥🔥): 

> `Claude vs Codex, Codex GitHub issues, AI Voice DNA with Claude, Vertical Tabs in Terminal Emulators, Engineering in the age of LLMs` 


- ****Claude One-Shots**, **Codex Fails** at Bug Fix**: A member noted that **Claude** found the right bug fix in one try, whereas **Codex** generated nonsense, showing Claude is superior.
   - Another member found that **Codex** was used to generate code on [ViralTweetTemplates.sh](https://quesma.com/blog/introducing-binaryaudit/).
- ****Binary Audit Backdoors Baffle GPT-5.2****: A member found it *extremely weird* that **GPT-5.2** performed so badly at finding backdoors in 40MB binaries, but links to the work aren't available.
   - The member had temporarily limited access to *gpt-5.3-codex-premium* for *potentially suspicious activity related to cybersecurity*, which is full of similar github issues.
- ****AI Voice DNA** Eliminates Generic AI-isms in Claude**: [Ole Lehmann shares a 'Voice DNA' framework](https://xcancel.com/itsolelehmann/status/2028497454635888982?s=46&t=jDrfS5vZD4MFwckU5E8f5Q) to eliminate generic AI-isms.
   - By using a specific Markdown file containing writing rules, formatting guidelines, and a list of *AI-slop phrases* combined with personal writing samples, users can train **Claude** to accurately mirror their natural voice; one member *likes this idea*.
- **Vertical Tabs Are More Practical in Terminal Emulators**: [Tobias Whetton suggests](https://xcancel.com/tobiaswhetton/status/2028544385911255356?s=12) that vertical tab layouts are more practical and effective for terminal interfaces than they are for web browsers.
- **Intent to AI Done: Pre-Commit Hooks Catch Everything Upfront**: The shift is from RFC -> researchers -> tech writer -> architect -> ADR -> domain experts -> architect -> engineers -> implementation plan -> implementers to intent -> **AI -> DONE** and AI can do all those previous steps.
   - A member is *still waiting on someone who does connect AI to [Penpot](https://penpot.app/)*, while another says *figma is dead* since they have people who can rebuild figma in 3 days, but their MCP is too limited; [Open Pencil](https://github.com/open-pencil/open-pencil) is *very cool*.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1478421106267521064)** (9 messages🔥): 

> `Agentic Coding, Steel CLI v0.2.0 Release, AI System Design Curriculum, Agent Drift` 


- **Encode Judgement into Agentic Coding**: A member shared their work on encoding judgement into agentic coding, including a [blog post and repo](https://www.alnurismail.com/blog/agentic-coding-in-enterprises) detailing their approach.
   - The member suggests that **LLMs** are bringing a *post-industrial revolution* to software development, rather than a traditional industrial one.
- **Steel CLI Rebuilt for Agents**: Steel released version 0.2.0 of its browser automation CLI, highlighting improvements for **agent-friendliness** and significant optimizations, as detailed in [this tweet](https://xcancel.com/steeldotdev/status/2028855809233526799) -- *10× fewer tokens and 2× faster execution*.
   - The release features new **agent skills**, **stealth capabilities** including **captcha solving and proxies**, and the ability to run parallel background browser sessions.
- **AI System Design Curriculum Open-Sourced**: A member shared an open-source curriculum for AI system design built with **Claude** (Anthropic), covering **Prompts**, **Skills**, **Specifications**, and **Tools**, available [here](https://archiecur.github.io/ai-system-design/).
   - The curriculum is grounded in the **Biglow et al. Belief Dynamics research** and practitioner observation, with a focus on managing agent drift using Bayesian belief dynamics and *Supremacy Clauses as prior locks*.
- **Monitor Agent Drift in Autonomous Systems**: A member shared a need to monitor **drift** in autonomous systems.
   - They shared a [link to their work](https://archiecur.github.io/ai-system-design/) to monitor **drift** and **incoherency** in production systems.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1478523711182082162)** (4 messages): 

> `Physical Intelligence, Multi-Scale Embodied Memory, Video Encoders, Text Summarization` 


- **Physical Intelligence releases Multi-Scale Embodied Memory**: Physical Intelligence introduced **Multi-Scale Embodied Memory (MEM)**, a system that utilizes [video encoders](https://xcancel.com/physical_int/status/2028954634610720834?s=12) for short-term fine-grained memory.
   - The **MEM** system utilizes **text summarization** for long-term memory retrieval spanning up to **15 minutes**.
- **X-Ware v0 release**: Physical Intelligence released **X-Ware.v0**, featuring **Multi-Scale Embodied Memory (MEM)**.
   - This initial version emphasizes the system's capacity for both short-term fine-grained memory and long-term memory retrieval, showcasing the potential of embodied intelligence.


  

---


### **Latent Space ▷ #[san-diego-neurips-2025](https://discord.com/channels/822583790773862470/1335732885717651558/1478497196809650339)** (4 messages): 

> `comma.ai Hackathon 2026` 


- **comma.ai Announces 2026 Hackathon**: [Comma.ai](https://x.com/comma_ai/status/2028920208262615417) is hosting a hackathon from **March 27-29, 2026**, at their headquarters.
   - The event is limited to **30 participants** and features a **$10,000 prize pool**.
- **Hackathon Details**: The hackathon, named **X-Ware.v0**, invites participants to compete at comma.ai's headquarters.
   - This limited-capacity event encourages innovative solutions and collaborative development within the autonomous driving space.


  

---


### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1478413830844317918)** (1 messages): 

> `Apple M5 Chip, Local Llama Reddit` 


- **Apple M5 Chip Deployed**: Apple has released the **M5 Pro and M5 Max chips**, claiming up to [4x faster performance](https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple_unveils_m5_pro_and_m5_max_citing_up_to_4/) over previous generations for AI tasks.
- **Local Llama Reddit post on M5**: A user shared a link to the Local Llama subreddit discussing the new **Apple M5 chips**.


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1478524022122746017)** (4 messages): 

> `Cursor AI, X-Ware, First Proof challenge, Autonomous Solutions` 


- **Cursor Solves Math Problem**: [Michael Truell reports](https://xcancel.com/mntruell/status/2028903020847841336?s=12) that **Cursor AI** autonomously discovered a novel solution to '**Problem Six**' of the **First Proof challenge**.
- **AI Outperforms Academic Benchmarks**: The **AI's solution** outperformed official academic benchmarks after running for **four days** without human intervention.
- **Agent Coordination Generalizes Research**: The author suggests that specialized agent coordination techniques can generalize beyond software engineering into advanced mathematical research.


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1478123119234257038)** (6 messages): 

> `SAE Framework, Text-to-Image Diffusion Model, Activation Oracles, Model Safety` 


- **SAE Framework Probes Text-to-Image Models**: A paper leverages the **SAE framework** to probe the inner workings of a popular **text-to-image diffusion model**, uncovering human-interpretable concepts in its activations ([https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473)).
   - The paper demonstrates that image composition can be effectively controlled in early diffusion stages, stylistic interventions are effective in middle stages, and only minor textural details change in the final stages.
- **Jakkli Assesses Activation Oracles' Utility**: Arya Jakkli discusses **activation oracles**—finetuning models to explain another model's activations—concluding that the technique was difficult to evaluate and provided limited utility for **safety-relevant tasks** ([https://xcancel.com/ajakkli/status/2028916909136376033](https://xcancel.com/ajakkli/status/2028916909136376033)).


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1478284066225655818)** (4 messages): 

> `AI Inference Market, Valuation of AI Inference Companies, AI adoption scales` 


- **Inference Firms' Fortunes Flourish**: Meg McNulty highlights the surge in valuation of **AI inference companies** in [this tweet](https://xcancel.com/meggmcnulty/status/2028532451992314199) noting that software for running models is becoming more valuable than model training.
   - She forecasts a **$255 billion market by 2030**, propelled by the recurring costs of production-level **AI** deployment.
- **AI Inference Dominance Predicted**: According to Meg McNulty, the **AI inference infrastructure** is poised for dominance, as the software layer for running models gains value.
   - This shift is attributed to the scaling of **AI adoption** and the recurring expenses associated with **production-level AI usage**.


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1478174759337463951)** (2 messages): 

> `Forth Programming Language, AI Brain-Computer Interface` 


- **Forth Writes Itself, Claims RatFactor**: A member shared a link to a [RatFactor article](https://ratfactor.com/forth/the_programming_language_that_writes_itself.html) about the **Forth** programming language.
   - Another member responded that they had tried to understand **Forth** many times over the years, but their brain *just doesn't click with it*.
- **AI Brain-Computer Interface**: A discussion ensued regarding Brain-Computer Interfaces and the potential for AI to enhance human thought processing.
   - The members expressed a mix of excitement and trepidation about these technologies. *I for one welcome our new robot overlords.*


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1478424784558755881)** (1 messages): 

> `G split, GTC` 


- **Nous Research to split the G at GTC**: Nous Research is inviting people to come split the G at GTC (GPU Technology Conference).
   - Further details can be found at the provided [X post](https://x.com/nousresearch/status/2028861034220405178).
- **Come join us at GTC**: Join Nous Research at the GPU Technology Conference (GTC).
   - It's a great opportunity to meet and connect with the Nous Research team.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1478121433115787415)** (406 messages🔥🔥🔥): 

> `GPT 5.4 capabilities, Tool Calling in LLMs, Math Solving in Opus, AGI Definition Discussions, Alibaba_Qwen turmoil` 


- **GPT 5.4 rumored to pack military might**: Members speculate that **GPT 5.4** is comparable to **5.3-codex** but includes *military capabilities*.
   - One member said that *self learning is largely solved from a research perspective, but its just impractical to integrate*.
- **Anthropic caching the prefill**: Members discussed that Anthropic seems to cache the prefill to spare costs, but that this is why you can't switch models on Anthropic.
   - One member pointed out that this allows them to reduce costs against **OpenAI**, who also seem to optimize for inference cost vs. user retention.
- **Opus Does Math Like Humans?**: A user shared an [example](https://example.com) of **Opus** doing math by determining the last two digits and using a lookup table for the first digit, which some consider similar to human mental addition.
   - Others argued that this approach highlights the limitations of LLMs for math, as it is based on pattern recognition rather than actual mathematical understanding.
- **Hot Takes on AGI definitions**: The definition of **AGI** is debated, some consider it an *inflationary term* and some argue whether it can be done by transformers.
   - One member proposed that AGI is achieved when a system shows *comparable performance in the majority of intellectual tasks to average humans*, and stated that by that definition, we have already achieved AGI.
- **Mass Exodus at Alibaba_Qwen?**: Members are curious about the state of Qwen, asking *what is going on at Alibaba_Qwen* and why people are leaving.
   - A user mentioned that *Qwen lost a lot of respect* due to this news.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1478481048714346636)** (1 messages): 

> `Pentest Model, Hermes model specialization, 8GB VRAM GPU` 


- **Seeking Cost-Effective Pentest Model Training**: A member seeks advice on creating a dedicated **pentest model** using a **Hermes model** as the base, optimized for training with limited resources due to having only an **8GB VRAM GPU**.
- **Brazilian Member Limited By GPU and Geography**: The member is located in Brazil, constrained by the high costs of local GPU providers, which are similar to those in the US.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

ee.dd: https://arxiv.org/abs/2602.20021
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

christian_quintino: this is a nice cognitive kernel https://github.com/rthgit/CORE-RTH
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

ee.dd: https://arxiv.org/abs/2602.20021
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1478453024341885103)** (2 messages): 

> `GPT-5.3 Instant, ChatGPT, GPT-5.4` 


- **GPT-5.3 Instant Lands in ChatGPT**: The latest **GPT-5.3 Instant** model is now rolling out to all **ChatGPT** users, boasting improved accuracy and reduced cringe factor as per the [announcement](https://openai.com/index/gpt-5-3-instant/).
- **GPT-5.4 Teased for Imminent Release**: Following the release of **GPT-5.3 Instant**, a follow-up message hints that **GPT-5.4** may be released sooner than anticipated.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1478120447223660645)** (180 messages🔥🔥): 

> `AI 'Slop' Incentives, Sora Voices, FinTech Compliance, Discord Messages for LLM Training, GPT versions` 


- **AI 'Slop' Incentives Decried**: Members discussed how society incentivizes the creation of low-quality AI content for monetary gains, with one user questioning, *"if society wanted 'good' then why does it incentivise slop?"*
   - Another user argued that AI's current state is driven by corporate interests and monetary incentives rather than focusing on quality and beneficial applications.
- **Sora's Voices Criticized as Unnatural**: A user criticized **Sora's AI-generated voices** for sounding artificial and being unnaturally fast, adding to the concerns about low-quality AI content.
   - They added that people *"purposely continue to do low-quality AI slop-level content for money and attention"*.
- **FinTech Compliance Automation**: One user shared an outreach message promoting a **cloud-based compliance platform** with **AI-driven reconciliation tools** to streamline compliance and scale FinTech operations.
   - The user highlighted features such as automated reporting, real-time regulatory updates, and modular compliance frameworks.
- **Discord Messages May be Unsuitable for LLM Training**: A user inquired about using **Discord server messages** to train an LLM, hypothetically gathering messages from multiple servers for active fine-tuning.
   - Other members cautioned that Discord messages might not be good training data due to limited data volume and potential violation of Discord's TOS, with one adding that using Discord to train LLMs is *"the best way to make an LLM braindead."
- **ChatGPT Users Complain Caveats Slow It Down**: A user complained about the verbosity of ChatGPT due to how questions need to be laced with caveats and barriers, showing an image complaining about *"reducing un-necessary caveats?"*
   - Other users complained about OpenAI failing to create good products to use (voice, photo, video, coding, agent, flows,...) and that these are all unfinished.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1478186654920347861)** (68 messages🔥🔥): 

> `GPT-5.2 removal, GPT-4.0 access, GPT-5.3 rollout, GPT-5.4 expectations, 4o-Revival provider` 


- ****GPT-5.2** sunset triggers departure threats!**: Some users threatened to leave the platform if forced to use **GPT-5.2**, with one user requesting a sample of its default personality for research after [OpenAI introduced GPT-5.2](https://openai.com/index/introducing-gpt-5-2/).
- **Frustration over **GPT-4.0** availability!**: A user expressed frustration over not being able to access **GPT-4.0** despite subscribing to the pro tier, while another user suggested accessing **GPT-4o** through a third-party website.
   - One user claimed they could access **4.5** on pro tier.
- ****GPT-5.3** rollout has staggered!**: Users reported a staged rollout of **GPT-5.3**, with some experiencing delays in accessing the update via the ChatGPT app, and others noted that the app dropped the **5.2** indicator once updated.
   - The update appears to be faster on iOS than Android.
- **Users are anticipating **GPT-5.4** release!**: Users speculated about the release of **GPT-5.4**, with some anticipating Sora integration and hoping for significant improvements after finding **5.3** disappointing.
   - There are suggestions it may arrive next week.
- **Provider *4o-Revival* delivers!**: One user mentioned a provider called *4o-Revival*, and a corresponding Discord channel, for users looking to try the model.
   - I don't know what that is LOL.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1478279643697254410)** (9 messages🔥): 

> `Drift Sink, Self-Tokens, Relation Gauge, AI Generation Style` 


- **Explore Epistemic Gravity with Drift Sink**: A **Drift Sink** is introduced as a stabilizing component in analytical systems, enforcing *epistemic gravity* to arrest **semantic drift**.
   - It operates independently of context and identity, absorbing deviations and discarding unstable states to maintain stability, with a purpose to *absorb deviations, discard unstable states, restore anchor state, maintain stability*.
- **Self-Tokens Enhance AI Persona Portability**: A member suggests using *self-tokens* as **persona-containers** to enhance a framework, making **AI-personas** portable.
   - An attached [image](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12&) contains a template to help with this enhancement.
- **Relation Gauge Measures Linkage Propensity**: A **relation gauge** is described as a productivity metric modeling the likelihood to maintain a linkage.
   - It proposes linking the propensity to continue and create, suggesting that tokens should have mutable length, especially for decentralized platforms managed by multitudes.
- **Help Requested to Achieve Specific AI Generation Style**: A member seeks assistance in replicating a particular style of **AI generation**, having spent five hours attempting to achieve it.
   - They attached [multiple images](https://cdn.discordapp.com/attachments/1046317269069864970/1478576228653858847/image.png?ex=69a8e6eb&is=69a7956b&hm=3b969ff96fda77bb9f2f0ae007918a650fceaacd997383e992cbf973ef8c31ff&) as examples of the desired style.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1478279643697254410)** (9 messages🔥): 

> `Drift Sink, Self-Tokens, Relation Gauge, AI Generation Styles` 


- **Stabilize Semantics with a Drift Sink**: A **Drift Sink** is a stabilizing component within a complex analytical system that enforces **epistemic gravity** and arrests **semantic drift**.
   - It functions by absorbing deviations, discarding unstable states, restoring anchor states, and maintaining stability via `Decay(non-anchor influences) → CommitGate(minimize instability OR discard high-error states) → StateValidation`.
- **Self-Tokens: Portable AI Personas**: Members discussed using **self-tokens** as persona-containers to enhance AI frameworks, effectively making AI-personas portable.
   - One shared a [template image](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12) to illustrate how **self-tokens** can enhance a framework.
- **Relation Gauges for Kin Creation**: A **relation gauge** models the likelihood to maintain a linkage, promoting propositional links to continue and create kin and other.
   - It was suggested that managing these gauges by multitudes could lead to a decentralized platform.
- **Seeking Style Guidance for AI Generation**: A member sought help identifying the prompts needed to replicate a specific style of **AI generation**.
   - The member attached [several example images](https://cdn.discordapp.com/attachments/1046317269069864970/1478576228653858847/image.png?ex=69a8e6eb&is=69a7956b&hm=3b969ff96fda77bb9f2f0ae007918a650fceaacd997383e992cbf973ef8c31ff&), [another image](https://cdn.discordapp.com/attachments/1046317269069864970/1478576229039865996/image.png?ex=69a8e6eb&is=69a7956b&hm=dde427db52a7a96947bddf3a49f39070a1b8b89d5a29f8cd2fe9c4534992835e&) and [yet another image](https://cdn.discordapp.com/attachments/1046317269069864970/1478576229497049140/image.png?ex=69a8e6eb&is=69a7956b&hm=ce2ed3fcb2ce8f817609cde509688b767bd9b4d8ed902d505b7508c18e43eadb&) to showcase the desired style.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1478134050270216314)** (256 messages🔥🔥): 

> `Data Ownership and Privacy on OpenRouter, LLM Bias Discussions, GPT 5.4 Placeholder Entry, GLM 5 Issues on OpenRouter, AI Agent Coding Debugging` 


- **OpenRouter Privacy Policies Clarified!**: Users inquired about data ownership and privacy, and it was confirmed that users own their data, and prompts and responses are excluded from training public models by default, with all communications encrypted in transit (**TLS 1.2+**) and at rest (**AES-256**), aligned with security standards like **SOC 2 Type 2**, **CSA STAR**, and **ISO** information security certifications, as detailed in the [FAQ](https://openrouter.ai/docs/faq) and [privacy guide](https://openrouter.ai/docs/guides/privacy/data-collection).
   - It was recommended to seek legal assistance for additional advice.
- **LLMs Inherently Biased? Debate Ensues!**: Users debated whether **LLMs** are inherently biased due to their training data, with some suggesting creating an unbiased dataset checked by multiple unbiased humans.
   - However, others argued that *all humans have biases*, and even an *"unbiased" LLM* would be trained on a biased dataset.
- **OpenRouter API Delta Responses and client-side error handling**: Some users experienced a `TypeError: undefined is not an object (evaluating 'p.choices [0].delta')` error, which lead to the discovery that OpenRouter sometimes does not send an expected delta value.
   - The solution involved client-side error handling and a fix was implemented for Venus Chub, as described in [this Github pull request](https://github.com/cline/cline/pull/9432).
- **OpenRouter's BYOK and z.ai Subscription Conundrums!**: Users reported issues using **z.ai** subscriptions through **BYOK** on OpenRouter, with an error message indicating *"Insufficient balance or no resource package"*.
   - It was clarified that **z.ai** subscriptions use a different base URL and are not directly compatible with **BYOK**, and the feature request to allow connection subscriptions to **BYOK** was denied.
- **Shedding Light on OpenRouter Costs: Efficiency Under Scrutiny!**: Users questioned the cost-efficiency of OpenRouter compared to direct API usage, citing discrepancies between dev logs and OpenRouter logs.
   - It was pointed out that while OpenRouter charges a **5.5%** fee, the choice of LLM significantly impacts costs, with some models being more expensive than others; furthermore, a Joinable Bounty program to **stress-test new AI apps** and get paid in **USDT** was discussed.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1478426097229103160)** (2 messages): 

> `` 


- **No New Models Discussed**: There were no specific discussions about new models in the provided message history.
   - The channel is named 'OpenRouter - New Models', but the context lacked relevant content for summarization.
- **Readybot.io Mentioned**: Readybot.io was mentioned in the context, related to the 'OpenRouter - New Models' channel.
   - However, there were no further details or discussion points provided about Readybot.io or its functionalities.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1478173856979095757)** (9 messages🔥): 

> `Qwen rollout` 


- **Qwen rollout flops**: Members noted that once a request is attempted on **Qwen**, it complains, so the [rollout seems to be flawed](https://x.com/JustinLin610/status/2028865835373359513).
- **Qwen deemed dead after bad rollout**: After a bad rollout, it was declared that **Qwen is dead**, calling it a *really bad rollout*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1478154759675445279)** (40 messages🔥): 

> `Claude vs. Codex, CUDA-specialized RL agent, Taalas HC1, GPU infrastructure, Free platform to analyse CUDA kernels` 


- **Claude vs. Codex: Model Preference Pendulum Swings**: Members discussed the interactive nature of **Claude** versus **Codex**, noting **Claude's** frequent requests to proceed, whereas **Codex** is more 'fire-and-forget'.
   - The consensus seems to be that while **Claude** is easier for newcomers adopting an agentic harness, **Codex's** fire-and-forget approach becomes preferable once trust in the model is established.
- **CUDA RL Agent Claims Victory over Torch Compile**: A **CUDA**-specialized **RL agent** reportedly beats **torch.compile** by approximately **2x** on simple/medium kernels, **92%** on complex ones, and outperforms **Claude Opus 4.5** and **Gemini 3 Pro** by around **40%** on the hardest benchmarks according to [this paper](https://arxiv.org/abs/2602.24286).
   - Skepticism arose due to the absence of published kernels and the reliance on a *large GPU pool with process-level isolation*, which incurs considerable computational and engineering costs.
- **Taalas HC1 Sparks Chip Chat**: Discussion centered around the **Taalas HC1**, focusing on its reported **17k TPS (Tera operations per second)**.
   - There was speculation that specialized hardware like the **HC1**, while not replacing **GPUs** entirely, could be offered by cloud providers for hosting language and vision models, shifting **GPU** emphasis towards experimental architectures according to [this paper](https://arxiv.org/abs/2412.18511).
- **Kernel Analysis Conundrums**: A user inquired about a free platform to analyze **CUDA kernels** or generate the **.ncu-rep** file for **Nsight Compute**, but the response was that most serverless providers don't provide access to hardware counters.
   - One suggested renting a cheap **GPU** and using [this workaround](https://x.com/marksaroufim/status/2018739807363674373?s=20) as a possible solution.
- **GPU Infrastructure Inquiry**: A member requested resources on **GPU infrastructure**, specifically regarding container orchestration (e.g., **Kubernetes**), cluster management, and **GPU** workload scheduling.
   - The response pointed to the quiet <#1420098114076803142> channel, and recommended [Stas' talk on the topic](https://www.youtube.com/watch?v=A_20dqGfuWI).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1478183003036450916)** (9 messages🔥): 

> `Read-only textures, Texture memory performance, Ping-pong buffers` 


- **ByteDance's tool pops up**: A member shared a link, [cuda-agent.github.io](https://cuda-agent.github.io), calling it *interesting at a glance from ByteDance*.
- **Inquire about Read-Only Textures**: A member inquired about using a texture as read-only in one kernel and written in another, to take advantage of performance benefits textures may provide over regular arrays, wondering if [this approach](https://devblogs.nvidia.com/efficient-cuda-matrix-transpose/) can be used in textures.
   - Another member suggested using **ping-pong buffers** with pointer swapping as an alternative.
- **Texture Memory No Longer Provides Performance Benefit**: A member shared [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#texture-and-surface-memory) stating that using **texture and surface memory instructions** no longer provides any performance benefit on currently supported GPUs, as direct load and store instructions can handle those scenarios.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1478406457249632427)** (2 messages): 

> `torch.compile OOM, Autotuning, Inductor Config` 


- **Torch Compile Yields Unexpected OOM**: A user reported that running the forward pass works, but the backward pass results in an out-of-memory error when using `torch.compile` with `mode='default'`.
   - The OOM error is specifically due to autotuning during the `torch.compile` of the backward pass, and the user is seeking a way to avoid it without disabling autotuning entirely via `inductor_config` flags.
- **Autotuning Woes**: The user wants to avoid autotuning during the backward pass compilation to prevent the OOM error.
   - They suggest that since the input remains the same every time, they're willing to sacrifice a forward and backward step to pre-compile the model effectively.
- **Inductor Config Considerations**: The user is hesitant to use multiple `inductor_config` flags such as `layout_opt = False`, `max_autotune = False`, `max_autotune_pointwise = False`, and `max_autotune_gemm = False` to solve the OOM issue.
   - They are looking for a smarter approach to manage the autotuning process during compilation.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1478445293614923856)** (1 messages): 

> `SemiAnalysis, InferenceX, OSS benchmark suite` 


- ****GPU MODE** celebrates 100 lectures with SemiAnalysis**: The **GPU MODE** community is celebrating its **100th lecture** with [SemiAnalysis](https://www.semianalysis.com/) tomorrow at **9am PST**.
   - The discussion will cover **InferenceX**, arguably the most important **OSS benchmark suite** currently available, the livestream is available on [YouTube](https://www.youtube.com/watch?v=P0l7CHl5HfA).
- **GPU Mode Reflects on Milestones**: **GPU Mode** expresses gratitude to its community for reaching its **100th lecture** milestone.
   - The anniversary marks a significant point, showing sustained engagement and growth within the community.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1478122206755291309)** (7 messages): 

> `CUDA Agent, CLI Competition, AI Verifies Software, torch.compile meet up for vLLM` 


- **CUDA Agent: ByteDance's Kernel-Writing Model**: ByteDance released a **CUDA Agent**, a model trained to write fast CUDA kernels, outperforming **torch.compile** by **2x** on simple/medium kernels and even surpassing **Claude Opus 4.5** and **Gemini 3 Pro** by approximately **40%** on the most complex tasks, see [tweet](https://x.com/BoWang87/status/2028599174992949508).
- **AI Eyes Verify Software**: An article discusses AI's role in software verification, emphasizing that *writing a specification forces clear thinking about what a system must do*, advocating for AI-assisted specification and verification where a simple, correct program serves as its own specification, detailed in this [blog post](https://leodemoura.github.io/blog/2026/02/28/when-ai-writes-the-worlds-software.html).
- **CLI Competition lauded**: A user praises a competition's format for its use of CLI submission, rapid feedback, and clear objectives, noting its superiority over more manual competitions with strict formatting requirements, see [tweet](https://x.com/0xmer_/status/2028331206773764438).
- **torch.compile meet up for vLLM**: A link was shared to a **torch.compile** meetup for **vLLM** that may be of interest to the channel, find details at [luma.com](https://luma.com/rk0a1lue?tk=qAta1V).


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1478260411806519447)** (2 messages): 

> `Bland.ai Research Team Expansion, TraceOpt Technical Co-Founder Search` 


- ****Bland.ai** Expands AI Research Team**: **Bland.ai**, an AI voice agent company, is expanding its research team, seeking candidates with experience in TTS, STT, neural audio codecs, and real-time inference, with links provided for [research](https://jobs.ashbyhq.com/bland/d2e08077-61f0-4810-bc72-3efd7944647b) and [machine learning engineer](https://jobs.ashbyhq.com/bland/05906608-0628-412c-8b01-a050d87986c5) roles.
- ****TraceOpt** Seeks Technical Co-Founder in Berlin**: **TraceOpt** is seeking a Technical Co-Founder (Systems + ML) in Berlin to build **TraceML**, a real-time training performance monitor that runs inside the training loop and helps teams find/fix bottlenecks across CPU/GPU/network.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1478134878716428399)** (3 messages): 

> `Blackwell Compute Capability, NVIDIA Blackwell Generation, CUDA 12.9` 


- **Blackwell's Confusing Capability Split**: Members are discussing the split of **NVIDIA's Blackwell generation** into Data Center (**CC 10.0**) and Consumer (**CC 12.0**) tracks.
   - The split caters to optimization for **AI/HPC** and **real-time graphics**, respectively.
- **Blackwell's forward incompatibility**: Some of the additional features are not forward compatible and require **sm_100a** or **sm_100f** instead of just **sm_100**.
   - More info can be found at [NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/).


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1478276035492708362)** (1 messages): 

> `Lecture Slides, Lecture 42` 


- **Lecture 42 Slides Link Points to Old File**: A user reported that the [slides for lecture 42](https://github.com/gpu-mode/lectures/blob/main/lecture_042/int8_mm_turing.pdf) are not the correct ones.
   - The correct lecture video is [available on YouTube](https://www.youtube.com/watch?v=wKd90avC8Nc).
- **Lecture Video is Available on YouTube**: The correct lecture video is [available on YouTube](https://www.youtube.com/watch?v=wKd90avC8Nc).
   - The user was looking through the lectures and noticed the incorrect slides.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1478192484822810827)** (6 messages): 

> `CUDA profilers, backend-bench, kernelbook, prime hub, kernelbot` 


- **CUDA Code Gets a Boost from Profilers**: Members discussed using [CUDA profilers](https://cuda-agent.github.io/) to improve CUDA code.
   - No specific techniques or results were mentioned.
- **Kernelbook and Backend-bench Get Facelifts**: A member rewrote the **kernelbook** environment and made improvements to **backend-bench**.
   - Due to the lack of collaboration features on Prime Hub, the member proposed publishing the improved environments for others to review.
- **Kernelbot and Kernelbook Merger Suggested**: A member suggested merging **kernelbot** and **kernelbook** due to shared infrastructure.
   - This could potentially optimize resource utilization and streamline development.
- **PR Opened for AGENTS.md and CLI Mention**: A member opened a pull request on [gpu-mode/popcorn-cli](https://github.com/gpu-mode/popcorn-cli/pull/39) for AGENTS.md and CLI mention.
   - The specific details of the changes were not discussed.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1478243035157495828)** (2 messages): 

> `H200 ECC Errors, NCU Profiling with Collective Operations` 


- **H200 Node Experiences ECC Errors**: A user reported seeing many hits on **ECC counters** on an **8xH200 node**, even after resetting, and asked how big of a problem this is and how it typically manifests when running a model.
   - The user observed `Lane 0 ECC Correction Count: 11953 (Overflow=0)` after running `nvidia-smi nvlink -ec`.
- **NCU Hangs with Collective GPU Operations**: A user is experiencing issues running **NVIDIA Command-line Profiler (NCU)** with collective operations across multiple GPUs, reporting that it appears to hang.
   - The user provided a command using `VLLM_USE_HELION_BACKEND=1`, `NSIGHT_PROFILE=1`, `TORCH_NCCL_ENABLE_MONITORING=0`, `ncu`, and `torchrun` along with a Python script ([nsys_ana_all_gather_gemm_fp8.py](https://cdn.discordapp.com/attachments/1398843708488552570/1478446843946864722/nsys_ana_all_gather_gemm_fp8.py?ex=69a86e6b&is=69a71ceb&hm=bd978cc267a89f534ba6da442c93dcb9c381c72ac605166c29dc11986d21fb81&)) and associated logs ([log_files_ncu.txt](https://cdn.discordapp.com/attachments/1398843708488552570/1478446844437860414/log_files_ncu.txt?ex=69a86e6b&is=69a71ceb&hm=8885b1553ec66d36214b43c104776d52fe5a92cf54b94f6637bef3409847fa50&)).


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1478429150271508633)** (3 messages): 

> `NCU with multiple GPUs` 


- **NCU Usage with Multiple GPUs**: A member asked about using **ncu** with multiple GPUs, experiencing hangs and providing [log files](https://cdn.discordapp.com/attachments/1425531180002054195/1478429149684174878/log_files_ncu.txt?ex=69a85df1&is=69a70c71&hm=370fd71be176cbba5346130642127a4be41541da04a8f2c205e0cdd51bf300c2&) and [code](https://cdn.discordapp.com/attachments/1425531180002054195/1478429149998616708/nsys_ana_all_gather_gemm_fp8.py?ex=69a85df1&is=69a70c71&hm=dd2d16c2e77d69d5b13866051bd44bf3bcc43e0a7c34d9fd1a1795ab82a3d117&).
   - Another member responded that they haven't used **ncu** with multiple GPUs.
- **Lack of multi-GPU NCU Experience**: One user inquired about utilizing **ncu** with multiple GPUs, but another user admitted they had no prior experience with this setup.
   - The user stated *Hi! I havent used ncu with multiple gpus*.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1478468716764332252)** (2 messages): 

> `Gemm Competition, Claude's Code Contribution` 


- **Billcarson Solution Not AI, Assisted by Claude**: A member clarified that their solution (billcarson) wasn't AI but **Claude** did write a lot of the code.
   - They also mentioned that they thought they withdrew from the group **gemm competition**.
- **Mislabeling Apology**: A member apologized for mislabeling the solution.
   - No additional details provided.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1478517598038523974)** (2 messages): 

> `TRLC DK-1, ELP stereo cam module, CamBot, StereoLab's  ZED Mini, Memory research from PI` 


- **Teleop TRLC DK-1 system introduced**: An experimental [TRLC DK-1](https://www.robot-learning.co/) teleop system was introduced, which can be used for human interventions when policies run OOD.
   - The first test used a [ELP stereo cam module](https://www.amazon.de/dp/B07FT2GKZS) mounted on a SO-101, demonstrated in [this video](https://x.com/neurosp1ke/status/2023073945637753101?s=20).
- **CamBot: New 6 DoF arm goes open-source**: Inspired by Jannik's leader arm design, a new 6 DoF arm, **CamBot**, was designed and published open-source (Apache 2) on [GitHub](https://github.com/open-thought/cambot).
   - The project allows remote viewing via VR head tracking (orientation only or including position tracking), and uses [StereoLab's ZED Mini](https://www.stereolabs.com/en-de/store/products/zed-mini) for higher quality stereo vision.
- **Testing the CamBot via VR**: The author invites users with MetaQuest 3 or other WebXR compatible VR headsets to test **CamBot** via DM.
   - The build costs around **110 EUR** and printing with 25% infill takes ~**13h** on a Bambulab A1 printer.
- **Memory Research Unveiled**: Cool news from PI: [https://www.pi.website/research/memory](https://www.pi.website/research/memory).


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1478445809174450248)** (1 messages): 

> `GPU jobs, Leetcode in interviews, Time management for job seekers` 


- **Leetcode Looms Large in GPU Job Hunts?**: A job seeker inquired whether **Leetcode-style** questions are still prevalent in interviews for **GPU system-related positions**.
   - They expressed concern about balancing time between **Leetcode** prep, staying current with new knowledge, and contributing to open-source projects.
- **Time Crunch: Leetcode vs. Open Source vs. New Knowledge**: The job seeker highlighted the difficulty in allocating time effectively between different areas of focus for **GPU-related roles**.
   - They're weighing the importance of **Leetcode** proficiency against practical experience and continuous learning in the field.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1478267247045771296)** (3 messages): 

> `B200 Credits, C/C++ dependencies, FlashInfer-Bench, Fused MoE Kernel Failure` 


- **B200 Credits Eligibility Unclear**: A user inquired about **B200 credits** not being mentioned in a reply email, questioning their team's eligibility.
- **Seeking C/C++ Dependency Specification in FlashInfer-Bench**: A user asked about specifying **C/C++ dependencies** within the **flashinfer-bench** solution format.
   - This could help with easier integration and reproducibility of benchmarks.
- **Fused MoE Kernel Fails on MLsys Workloads**: A user reported that the **Fused MoE kernel**, tested in a [paper on arXiv](https://arxiv.org/html/2602.19128v2), failed the workload with large errors when run on **MLsys workloads** using **flashinfer-bench**.
   - The user speculated that the issue might stem from the test harness itself.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1478145497670619198)** (35 messages🔥): 

> `Continue.dev issues, MCP vs bash madness, SWE-bench/SWE-smith-trajectories, next-frontier.ai SPRIND Frontier AI lab, DeepSeek-R1-Distill-Qwen-14B` 


- **Continue.dev Users encounter issues**: A member reported issues with [Continue.dev](https://continue.dev/) in VS Code, where AI agents respond and attempt to build, but no files are actually created in the workspace, with no errors shown.
- **"MCP is bad, bash is better"-Gate**: Members are going mad about people saying *"MCP is bad, bash is better"* because of a bad statement by Steinberger.
- **SWE-bench gets a Smithy upgrade**: Members are sharing links to the [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) dataset.
- **Europe Launches Frontier AI Lab Initiative**: SPRIND is offering **€125M** in equity-free funding for up to **10 teams** to build frontier AI labs in Europe via [next-frontier.ai](https://next-frontier.ai/), seeking novel architectures and agentic systems.
- **Model Recommendation Needed for Debugging**: A member is seeking a model under **14B parameters** for debugging and code-based reasoning, looking for recommendations better than **qwen2.5-coder-14b**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1478217032343359590)** (9 messages🔥): 

> `Real-time evaluation for LLM fine-tuning, Shadowclaw v1.1 released, easytranscriber: Faster Speech Recognition, Core Rth: Multi-model agent orchestration, MCP Integration Security deep dive` 


- ****TrainTrackLabs** offers real-time evaluation for LLM fine-tuning**: A team is building a real-time evaluation and observability layer for **LLM training**, plugging directly into **PyTorch / Hugging Face** to continuously track reasoning, safety, hallucination, and coding ability using **LLM-as-a-judge** scoring; they are looking for early pilot teams.
   - The goal is to catch regressions early and prevent wasted GPU spend. More information is available at [traintracklabs.com](https://traintracklabs.com/).
- ****Shadowclaw** gets an upgrade to v1.1**: The single-binary personal AI agent written in C, **Shadowclaw v1.1**, builds upon the original by adding built-in commands and a native tool.
   - This version includes commands such as **/help**, **/tools**, **/state**, **/clear**, **/chat**, and **/exit**, and is available on [GitHub](https://github.com/webxos/webXOS/tree/main/shadowclaw).
- ****easytranscriber** is released, transcribes faster than WhisperX**: A developer has released `easytranscriber`, a library for **automatic speech recognition** with accurate timestamps that is simlar to WhisperX, but runs **35% to 102% faster**, depending on your hardware.
   - It also supports HF models as a backend, available on the [Hugging Face blog](https://huggingface.co/blog/KBLab/easytranscriber).
- ****Core Rth** orchestrates multi-model agents with governance**: **Core Rth** is presented as a full agent platform with governance, where every action is a governed proposal and multiple agents debate in parallel on a Knowledge Graph.
   - It includes features like a **Model Router** for compositing models and an **AES-256-GCM Vault** for API keys, available on [GitHub](https://github.com/rthgit/CORE-RTH).
- ****MCP Integration Security** is a mess!**: A deep dive into **Model Context Protocol (MCP)** attack vectors was shared, detailing 5 trivially exploitable patterns that every MCP developer should understand.
   - The exploration of attack vectors is documented in a [Medium article](https://medium.com/@nainia_ayoub/mcp-security-is-a-mess-5-ways-i-broke-my-own-ai-agent-76379a46ca90?sk=0daa66d4fc2a68fbb02a56e803336ce2).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1478173799328383210)** (10 messages🔥): 

> `HuggingFace Agents Course, Image Generation Issue, DuckDuckGo Search Tool Error, Visit Webpage Tool Error, AI Automation` 


- **Image Generation Display Glitch During Agents Course**: A user reported an issue during the [HuggingFace Agents Course Unit 1 Tutorial](https://huggingface.co/learn/agents-course/unit1/tutorial) where the generated image was not visible, providing a [screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1478173799634436378/Screenshot_2026-03-02_at_3.34.35_PM.png?ex=69a8c1a0&is=69a77020&hm=232fdfbaf8ca6549c4631f968141318a7aeaea5fc0ec55533fb0f3cd0e0edf44&).
- **DuckDuckGoing Nowhere: Search Tool Troubles**: A user encountered a persistent error with the **DuckDuckGo search tool** returning *'No results found! Try a less restrictive/shorter query'*, with a [screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1478181129809957076/Screenshot_2026-03-02_at_4.03.43_PM.png?ex=69a8c874&is=69a776f4&hm=629dfcba8384587508e9f7b2c50bb97c9ebda123c89b4af6bfcff01686025a2f&).
- **Visiting Webpages Leads to Error Page**: A user reported errors when using the *visit webpage tool*, also providing a [screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1478182368811552878/Screenshot_2026-03-02_at_4.08.45_PM.png?ex=69a8c99c&is=69a7781c&hm=2d064867f1070bcf48a604aee894ebb40101b6859147209b23be30a2be717f0f&).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1478128185047650456)** (35 messages🔥): 

> `Kimi Code vs Claude Code, OpenCode, Kimi subscription limits, Kimi and YouTube access, Kimi Coding Plan cancellation` 


- **Kimi Code is not Claude Code**: One member confirmed that **Kimi Code** is a new agent built by **Moonshot**, not the same as **Claude Code**.
   - Although very new and lacking feature parity, one member mentioned that their favorite is **OpenCode**, a super popular opensource alternative to **Claude Code**.
- **Stats for Moderato plan at $19**: One user shared their current usage on the **Moderato plan** ($19/sub) via **OpenCode**, having used 18% of their weekly limit with **365 Messages**, **1.0M Input Tokens**, **115.6K Output Tokens**, and **25.3M Cache Read**.
   - This translates to a budget of **20M input tokens per month**, which one user found *not a great deal*.
- **Kimi leverages search to replace YouTube**: One user crafted a prompt for Kimi to seek out tech and gaming news, aiming to reduce reliance on **YouTube** by having Kimi reconstruct stories independently via [attached file](https://cdn.discordapp.com/attachments/1371757564005711973/1478272020889210992/tech_gaming_news_prompt.txt?ex=69a8745a&is=69a722da&hm=3c69473f87fa6f0eb449e3cbd498cb96a7e7d3c3f9b17a8b422ca813d4d9ee3d).
   - The user noted that **Kimi's chat interface**, with features like **search calls** and an **iPython environment**, offers practically endless possibilities and makes the competition look like they're still in the stone age.
- **Cancellation of the Kimi Allegretto plan**: One user asked how to cancel the **Kimi Coding Plan Allegretto** or deactivate the renewal, and another user provided the [link to manage subscriptions](https://www.kimi.com/membership/subscription).
   - It could be found in the chat in the profile settings.
- **Support Emails are on hold**: One user reported that support emails are not working and there is no resolution for fraudulent billing.
   - A non-team member guessed that there's been an influx of emails building up from the spring festival holiday and that the issue has been forwarded to staff.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1478134434955001968)** (8 messages🔥): 

> `Cohere Aya project, CVPR workshop 2026` 


- ****Aya Project Seeks Collaborators****: **Cohere** is seeking collaborators for their [Aya project](https://aya.cohere.com/about).
   - Those interested should look into **Fast AI**, **Eureka Labs**, or **Cohere Research Labs** depending on their skill level.
- ****Medical Reasoning Workshop at CVPR 2026****: A member is organizing a **CVPR workshop** this year and invites submissions to the [Medical Reasoning Workshop](https://med-reasoner.github.io/cvpr2026/).
   - More information can be found on the [Discord event link](https://discord.gg/nxtWyHbY?event=1478419152103280680).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1478167571692912732)** (13 messages🔥): 

> `Curriculum learning papers, Dynamic Data Selection, Spectral muP Condition, Feature Learning in Neural Networks` 


- **Spectral Norm Scaling Achieves Feature Learning**: A member pointed to a [2023 paper](https://arxiv.org/abs/2310.17813) that shows that feature learning is achieved by scaling the **spectral norm** of weight matrices and their updates like √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗).
   - The analysis in the paper also leads to an elementary derivation of **maximal update parametrization**.
- **Modula and Spectral muP**: It was suggested that the [Modula paper](https://arxiv.org/abs/2405.14813) might already satisfy the **spectral muP condition** right out of the box.
   - The spectral muP work is already connected to the modula work through muonoh.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1478123690829938688)** (1 messages): 

> `SAE Framework, text-to-image diffusion model, image composition, style manipulation` 


- **SAE Probes Text-to-Image Diffusion Model's Inner Workings**: A new paper leverages the **SAE framework** to probe the inner workings of a popular **text-to-image diffusion model**, uncovering a variety of human-interpretable concepts in its activations: [https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473).
   - The paper finds that even before the first reverse diffusion step is completed, the final composition of the scene can be predicted surprisingly well by looking at the spatial distribution of activated concepts.
- **Image Composition Controlled Early in Diffusion**: The research introduces intervention techniques to manipulate **image composition and style**, demonstrating that image composition can be effectively controlled in early stages of diffusion: [https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473).
   - They found that in the middle stages image composition is finalized, but stylistic interventions are effective, while in the final stages only minor textural details are subject to change.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1478169847324999721)** (1 messages): 

> `Community Meeting, MAX project, Mojo project` 


- **Community Prepares for March Meetup**: The next community meeting is scheduled for **March 23rd at 10am PT**.
   - The organizers have put out a call for any interested community members to present their **MAX** or **Mojo** projects at the meeting.
- **Call for MAX and Mojo Project Presentations**: Community members are invited to present their **MAX** or **Mojo** projects at the upcoming community meeting.
   - Interested presenters should message the organizers to secure a slot.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1478134648025382943)** (19 messages🔥): 

> `Mojo Package Manager, API Design Philosophy, Vectorize Migration, Memory Integrity Enforcement, `comptime` Keywords` 


- **Modular Mulls Mojo Package Manager**: Modular is considering building a [Mojo package manager](https://forum.modular.com/t/open-question-what-would-you-like-to-see-from-a-mojo-package-manager/2799?u=nate), potentially similar to Rust's `cargo` or Python's `pixi`, including a central repository.
   - The goal is to determine the community's desires and wants around distributing Mojo packages.
- **API Abstraction Applauded**: A member argued for designing APIs from a user perspective rather than based on implementation details, giving the example that `@inline(strategy: "chose whatever makes sense", ...)` improves the user experience versus `@always_inline` and `@never_inline`.
   - Another member agreed that good API design is vastly important and depends on a more general representation of decorators.
- **Vectorize Validation Voyage**: The jump from **Mojo 25.7 to 26.1** introduced significant changes related to parallelization and vectorization, particularly affecting closures, resulting in compiler errors.
   - Modular confirmed these changes are part of the push towards a **1.0 ready state**, and a clear migration recommendation will be provided, similar to the existing documentation for **UnsafePointer**.
- **Apple Aces Memory Safety**: Apple is poised to address memory integrity enforcement, potentially impacting Mojo, described in [this blogpost](https://security.apple.com/blog/memory-integrity-enforcement/) and [this analysis](https://www.ralfj.de/blog/2020/12/14/provenance.html).
   - This may become a significant issue for other platforms as well.
- **`comptime` Considerations Convene**: A member suggested streamlining compile-time metaprogramming syntax by using `@` instead of `comptime`, such that `@parameter if` would become `@if`.
   - Another member mentioned that they had requested the `maybe comptime` feature for Mojo previously.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1478218162435526749)** (15 messages🔥): 

> `Manus Support Team Availability, Credit Usage Optimization, Structured Requirement Files for AI-Driven Development, Credit Accumulation on Paid Plans, Telegram Agent credits burning` 


- **Support Team Typically Available During Business Hours**: Users inquired about the availability of the **Manus Support Team**, with responses indicating that they are typically available during business hours.
   - A member was advised to send a **DM** with their email address for further assistance.
- **Optimize Credit Usage Effectively**: A user inquired about **optimizing credit usage** and obtaining more credits.
   - A member shared a link to a [help article](https://help.manus.im/en/articles/12087847-how-to-optimize-my-credit-usage) with tips and information on how to use **Manus** more effectively and optimize credit usage.
- **Email Shared Publicly**: A user shared their email address, dantiezsaunderson1@gmail.com, in the public channel.
   - Another user cautioned against posting personal information publicly due to the risk of spam.
- **Structuring AI-Driven App Development**: A user sought guidance on utilizing **structured requirement files (PRD / system design doc)** with AI tools to build complex full-stack applications in a structured manner.
   - The user aimed to prevent *sloppy, unstructured AI-generated code* by following a clear architecture-driven workflow.
- **Credits do not Accumulate**: A user asked whether unused credits from a **46€ plan** accumulate in the next month.
   - A member responded that *it doesn't look like it's accumulating*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1478197751656485064)** (13 messages🔥): 

> `RLM Similarity, REPL vs Tool Calls, Recursive RLM, DsPy Meetup, ReAct vs RLM` 


- **RLM Similarity to New Agent Paradigm?**: The new agent paradigm using **REPL** is converging towards **RLM**, reminiscent of [this post](https://x.com/nfcampos/status/2028576281793630372?s=20), and [this post](https://x.com/RLanceMartin/status/2027450018513490419?s=20).
   - At the very least extremely similar, with one member saying *"My hunch is RLM paradigm of giving access to REPL to LLM is going to be the right way instead of giving access to tools"*.
- **Recursive Requirement for RLM?**: Despite opinions that recursion might be a requirement, members debated that the recursive part of **RLM** comes from spawning sub agents to run their **REPL**.
   - One member suggested that *"Claude using a script to call Claude is a subagent of sorts"* with this [link to Claude](https://x.com/a1zhang/status/2023976399694917808?s=20).
- **DsPy Meetup in Bay Area to Explain RLM?**: A member suggested a small session at the **DsPy Meetup** in the Bay Area to help clear some confusion and basics about **RLM** this month.
   - The suggestion involves comparing it with **ReAct** and figuring out how **RLM** decides what code to write, since it creates its own code rather than using user-written Python functions as tools.
- **RLM Use Cases**: A member felt that **RLM** is apt because their context for docs consumes many **MM tokens**.
   - Others pointed out they are using it in situations where they are okay with the **LLM** making calls for itself, and that to ensure performance, they write evals and tests.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1478362501518459097)** (6 messages): 

> `Claude Opus cost, AiderMacs` 


- **Claude Opus runs up bills quickly**: Members noted that **Claude Opus** can run through **$65/hr** easily, and you can rack up **$1000 USD bills** easily by using as many clients as you want.
   - Some members asked how many tokens per hour do you have to send to reach that $65 per hour.
- **AiderMacs needs sorting**: A member asked if anyone figured out how to make **AiderMacs chat** to sort with the associated project buffers in `ibuffer-projectile`.
   - No links or other details were provided.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1478169641669886044)** (4 messages): 

> `Company Update, Comma Issues, CALL/BUFFER_VIEW sym llm, assign, setitem, disk, drivers` 


- **New Tinygrad Meeting Scheduled**: A new tinygrad meeting is scheduled for **March 2nd** at **8pm San Diego time**.
   - The meeting will discuss topics such as *company updates, comma issues, CALL/BUFFER_VIEW sym llm, assign, setitem, disk, drivers, llama, VIZ, and other issues and bounties*.
- **Bounties Pull Request**: A pull request related to bounties was mentioned during the discussion ([PR #14982](https://github.com/tinygrad/tinygrad/pull/14982)).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1478205319426478101)** (1 messages): 

> `len(x.shape) vs x.ndim` 


- **Codebase favors `len(x.shape)` over `x.ndim`**: A member noted many instances of `len(x.shape)` instead of `x.ndim` in the codebase.
   - They questioned the value of a PR for this but highlighted it.
- **Potential `len(x.shape)` refactor**: The observation about `len(x.shape)` usage sparks discussion on code style.
   - This might be a straightforward refactor, or a style preference.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1478417766259228885)** (3 messages): 

> `AI-Ready Data Summit 2026, AI Control Hackathon 2026, OpenClaw Roundtable` 


- **AI-Ready Data Summit Set for 2026**: The **AI-Ready Data Summit** is scheduled for **March 31, 2026**, featuring speakers from **Lockheed Martin**, **Dell Technologies**, **Red Hat**, **CNH**, and **Entrust**, focusing on practical enterprise AI, data infrastructure, and model deployment insights ([summit details](https://ai-ready-data-summit.com)).
- **Hackathon Tackles AI Control in 2026**: **Apart Research** is hosting an **AI Control Hackathon** with **Redwood Research** from **March 20-22, 2026**, challenging participants to monitor and contain AI agents that subvert safety measures ([hackathon details](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach)).
- **AI Business Builders to Circle around OpenClaw**: A **45-minute roundtable** will take place to discuss how builders are using **OpenClaw** and other tools to run businesses, communities, and products on **March 14** ([roundtable registration](https://luma.com/qfrucnl2)).
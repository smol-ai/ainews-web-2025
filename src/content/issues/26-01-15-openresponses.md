---
id: MjAyNi0w
title: >-
  Open Responses: explicit spec for OpenAI's Responses API supported by
  OpenRouter, Ollama, Huggingface, vLLM, et al
date: '2026-01-15T05:44:39.731046Z'
description: >-
  **OpenAI** launched the **Open Responses** API spec, an open-source,
  multi-provider standard for interoperable LLM APIs designed to simplify agent
  stacks and tooling. Early adopters like **ollama** and **vLLM** support the
  spec, while notable absences include **anthropic** and **google-deepmind**.
  Agent design insights from **Cursor** emphasize explicit roles and planning
  over mega-agent models, with **GPT-5.2** outperforming **Opus 4.5** in long
  runs. The emerging dominant context/memory abstraction for agents is a
  **filesystem-as-memory** approach, championed by **llamaindex** and
  **langchain**, using virtual filesystems often backed by databases like
  Postgres. LangChain also shipped an open-source desktop interface for agent
  orchestration called **openwork**. This news highlights advances in API
  standardization, agent architecture, and memory abstractions in AI
  development.
companies:
  - openai
  - ollama
  - vllm
  - openrouter
  - anthropic
  - google-deepmind
  - langchain
  - llamaindex
models:
  - gpt-5.2
  - opus-4.5
topics:
  - interoperable-apis
  - agent-architecture
  - filesystem-memory
  - api-standardization
  - multi-agent-systems
  - prompt-engineering
  - model-comparison
  - virtual-filesystems
  - open-source
  - agent-ux
people:
  - reach_vb
  - simonw
  - yuchenj_uw
  - omarsar0
  - jerryjliu0
  - hwchase17
  - swyx
---


**Responses API is all you need.**

> AI News for 1/14/2026-1/15/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**205** channels, and **5564** messages) for you. Estimated reading time saved (at 200wpm): **433 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!


Standards work is often thankless and unrewarding, but it's a nice commitment to the community when it is done well. Today, OpenAI did right by the open community by [explicitly documenting their Responses API](https://www.openresponses.org/) and working with folks like [vLLM and ollama](https://x.com/reach_vb/status/2011863149356413275) to support this, however it is more of a surprise that the market leader in normalizing APIs, [OpenRouter, also supports this](https://x.com/OpenRouterAI/status/2011864089782599802). Notable omissions from the launch partnerships: Anthropic and Deepmind.

---

# AI Twitter Recap


**Interoperable LLM APIs: “Open Responses” coalesces around Responses as the new baseline**

- **Open Responses spec (multi-provider, agent-friendly)**: OpenAI DevRel + partners launched **Open Responses**, an open-source spec that standardizes a **Responses-API-like** interface across providers (“multi-provider by default”, extensible without fragmentation) so agent stacks don’t have to fork per model/provider. See the announcement from [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2011862984595795974) and partner callout by [@reach_vb](https://twitter.com/reach_vb/status/2011863149356413275). This is framed as a “clean slate” relative to Chat Completions: fewer special cases, better consistency for tool-heavy workflows.
- **Ecosystem traction + implementations**: Early sentiment from tooling builders is that this is the missing “formalized, standardized JSON API for talking to models” ([@simonw](https://twitter.com/simonw/status/2011865205123531155)). **Ollama** announced support quickly ([@ollama](https://twitter.com/ollama/status/2011871283928317971)), and **vLLM** notes they previously had to reverse-engineer provider behavior and expects the spec to simplify primitives/tooling ([@vllm_project](https://twitter.com/vllm_project/status/2012015593650536904)).

**Agents: planning > “multi-agent vibes”, and filesystems emerge as the dominant context/memory abstraction**

- **Cursor’s long-running-agent lessons (roles + planning + judging)**: Multiple posts summarize Cursor’s view that peer-like self-coordination fails; **explicit roles** (planners/workers/judges) and strong upfront planning work better, with an emphasis on **prompt/system stability** over harness complexity. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2011863636042469866) highlights the operational reality: it’s “hundreds of concurrent agents” rather than one mega-agent. [@omarsar0](https://twitter.com/omarsar0/status/2011823468468379782) ties this to Claude Code usage (subagents manage their own context; orchestrator stays high-level) and notes Cursor found **GPT-5.2** stronger for week-long runs while **Opus 4.5** tends to stop early/shortcut.
- **Filesystem-as-memory becomes the center of gravity**: A cluster of tweets converges on “files are all you need” for agent context, memory, and skills.  
  - LlamaIndex’s framing: files as (1) durable context store, (2) a search interface (often outperforming classic RAG patterns for dynamic traversal), and (3) a simpler substrate for tool calling/skills ([@jerryjliu0](https://twitter.com/jerryjliu0/status/2011849758944690625); amplified by [@llama_index](https://twitter.com/llama_index/status/2011846444156645438)).  
  - LangChain’s Agent Builder uses a **filesystem abstraction** with conventions like **AGENTS.md**, **skills/**, and **tools.json** so agents can update memory through feedback and persist behaviors ([@LangChain](https://twitter.com/LangChain/status/2011864707439690031)).  
  - Important implementation detail: LangChain’s “filesystem” is often a **Postgres-backed virtual filesystem wrapper**, not literal disk ([@hwchase17](https://twitter.com/hwchase17/status/2011834318172422279); clarification [@hwchase17](https://twitter.com/hwchase17/status/2011858266863911382)).  
  - A pragmatic skepticism appears too: “every filesystem source-of-truth eventually turns into a database” ([@swyx](https://twitter.com/swyx/status/2011984243430236608)).
- **Shipping agent UX + harnesses**:  
  - **openwork**: LangChain JS shipped an open-source “Claude Cowork”-style desktop interface (planning + filesystem + subagent delegation) built on deepagentsjs, runnable via `npx` with Anthropic/OpenAI models ([@LangChain_JS](https://twitter.com/LangChain_JS/status/2011863256223400360)).  
  - **UI honesty for “agent progress”**: critique that most agent UIs fake progress with spinners; LangChain JS demonstrates streaming tool-call events into React with TypeScript-safe events for real progress reporting ([@LangChain_JS](https://twitter.com/LangChain_JS/status/2011833970204557694); original by [@bromann](https://twitter.com/bromann/status/2011833439834775738)).  
  - **Dexter 3.0**: claims an event-based agent loop + dynamic context management reduced their “core loop” to ~100 lines while improving performance ([@virattt](https://twitter.com/virattt/status/2011933907881492498)).

**Model & capability drops: fast image models, open translation, small LMs, and audio S2S reasoning**

- **Black Forest Labs FLUX.2 [klein]**: New fast/small image generation/editing line: **4B (Apache 2.0)** and **9B (FLUX.2 non-commercial license)**, plus a new text encoder; positioned as <1s generation for iteration/editing ([@bfl_ml](https://twitter.com/bfl_ml/status/2011825819082244266)). fal shipped it on their marketplace ([@fal](https://twitter.com/fal/status/2011826361434771923)) and Arena added both to text-to-image and image-edit arenas ([@arena](https://twitter.com/arena/status/2011869067272208812)). Commentary notes how normalized big jumps have become (“~10x better than Stable Diffusion while ~as small”) ([@swyx](https://twitter.com/swyx/status/2011861139689513314)).
- **Google DeepMind TranslateGemma**: Open translation models built on **Gemma 3**, trained on **Gemini-generated** translation data, supporting **55 languages**, released in **4B/12B/27B** sizes, optimized for on-device/low-latency translation ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2011848249850630363); training/distillation angle [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2011848252451156244); summary [@_philschmid](https://twitter.com/_philschmid/status/2011848973074448657)). Early deployment notes include MLX Swift running quantized 4B on mobile ([@sach1n](https://twitter.com/sach1n/status/2011975664573038824)).
- **Zilliz/Milvus semantic highlight model**: Released a lightweight **0.6B** model + dataset with **8192 context** under **MIT**, plus a detailed training blog ([@mervenoyann](https://twitter.com/mervenoyann/status/2011732254591275022); blog pointer [@mervenoyann](https://twitter.com/mervenoyann/status/2011732428784865391)).
- **TII Falcon-H1-Tiny series**: Sub-100M-parameter LMs with specialized variants (coding, function calling, multilingual, reasoning), positioned for edge/IoT privacy deployments ([@yb2698](https://twitter.com/yb2698/status/2011805117016916056); org recap [@TIIuae](https://twitter.com/TIIuae/status/2012034581084430662)).
- **StepFun Step-Audio R1.1 (Realtime)**: Artificial Analysis reports this **32B** speech-to-speech “audio reasoning” model leads their **Big Bench Audio** at **96.4%**, with ~**1.51s** TTFT, and provides price points in $/hour and $/token equivalents ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012006066339581958)).

**RL, evals, and long-context training/inference: scaling practical agent training**

- **Unsloth: RL to longer contexts + vLLM acceleration**: Unsloth claims RL can scale to **7× longer contexts** via seqlen/hidden-state chunking + offloaded log-softmax, and up to **12×** with “Standby” for vLLM runs + tiled MLP; includes an example “gpt-oss QLoRA reaches 380K on 1× B200” ([@danielhanchen](https://twitter.com/danielhanchen/status/2011828515348627561)). vLLM echoes the 7× longer-context RL collaboration ([@vllm_project](https://twitter.com/vllm_project/status/2011857612103630924)).
- **Agent memory + context pruning research (“Focus”)**: DAIR highlights a paper proposing agent-controlled consolidation checkpoints (`start_focus`/`complete_focus`) that summarize learnings into a persistent knowledge block and delete intermediate traces, achieving **22.7% token reduction** on SWE-bench Lite with unchanged accuracy (Claude Haiku 4.5) ([@dair_ai](https://twitter.com/dair_ai/status/2011806092737827206)).
- **Benchmark integrity pushback**:  
  - MMLU-Redux: manually curated/leak-free fixes for issues found in MMLU topics/subsets ([@PMinervini](https://twitter.com/PMinervini/status/2011782967723511868)).  
  - A concrete “dataset artifact” warning: MMLU-Pro chemistry/physics subsets allegedly have a bias where “leading space” in an option correlates with correctness ([@giffmana](https://twitter.com/giffmana/status/2011859715043836166)).  
  - Arena’s own meta-analysis notes that “AI race leadership” depends heavily on prompt strata: OpenAI leads most of the time overall, but Anthropic leads more frequently on “Expert prompts” ([@arena](https://twitter.com/arena/status/2011849440160858443)).

**Infra & developer tooling: real-time inference, in-browser search, vector DB experimentation, and agent-enabled IDE workflows**

- **Together AI + Cursor inference stack on Blackwell**: Together describes engineering for real-time coding-agent inference (tight editor latency loop), citing reliability on **GB200/B200**, custom tensor-core kernels, FP4 quantization, and NVL72 mesh parallelism ([@togethercompute](https://twitter.com/togethercompute/status/2011875191828488598); technical bullets [@togethercompute](https://twitter.com/togethercompute/status/2011875193476829631)). A related note from [@realDanFu](https://twitter.com/realDanFu/status/2011876049215520919) mentions “technical nuggets” including even hardware maintenance details like replacing NVLink cables for stability.
- **VS Code doc search rebuilt in-browser**: VS Code reports their website search got much faster; engineering write-up describes **docfind**, running entirely in the browser via **WebAssembly** ([@code](https://twitter.com/code/status/2011827481175605487)).
- **RAG experimentation as first-class infra**: Qdrant + Tigris Data’s “RAG Lab” emphasizes reproducible A/B testing of chunking strategies by forking datasets and pairing each with its own vector index for apples-to-apples evaluation ([@qdrant_engine](https://twitter.com/qdrant_engine/status/2011679747244167175)).
- **Copilot CLI + agent SDK surface area**: GitHub Copilot CLI/Coding Agent added “automated memory” ([@_Evan_Boyle](https://twitter.com/_Evan_Boyle/status/2011932670096523326)), and there’s chatter about a Copilot CLI SDK enabling custom CLIs on top of Copilot auth (example: video promo generator) ([@burkeholland](https://twitter.com/burkeholland/status/2011934322413224152)).
- **OpenCode + Copilot subscription**: OpenCode says it can be used with a Copilot subscription via a “$39 pro+” tier that exposes “best coding models,” highlighting growing pressure for toolchain interoperability ([@opencode](https://twitter.com/opencode/status/2011790750543983072)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Local LLM Performance and Preferences

  - **[What’s the most cracked local LLM right now?](https://www.reddit.com/r/LocalLLM/comments/1qdx4n8/whats_the_most_cracked_local_llm_right_now/)** (Activity: 44): **The post inquires about the most advanced local LLMs (Large Language Models) that excel in reasoning, instruction following, coding, and handling long prompts, without VRAM constraints. The discussion highlights models like **Qwen3-Coder-480B**, **GLM 4.7**, and **Kimi-K2**. Notably, **Kimi-K2** is praised for its resistance to sycophancy and its unique capability for 'interleaved thinking,' allowing it to execute complex tasks with hundreds of tool calls, making it a standout in the local and open LLM space.** Commenters suggest that the term 'cracked' might be misused, as the performance of local LLMs is often hardware-dependent. **Gemma models** are also mentioned for their overall performance, while **Kimi-K2** is noted for its extensive interleaved thinking capabilities, setting it apart from others like **GLM-4.7**.

    - AbheekG highlights the Kimi-K2 model, emphasizing its unique capability as a 1 trillion parameter LLM, which requires significant computational resources. The model is noted for its resistance to sycophancy and its advanced 'interleaved thinking' training, allowing it to execute hundreds of tool calls for complex tasks. This positions it as a leading local LLM in terms of reasoning capabilities, surpassing even GLM-4.7 in this aspect.

  - **[Best AI for coding that isn't from the major disgusting companies? (Local or online)](https://www.reddit.com/r/LocalLLM/comments/1qdnao5/best_ai_for_coding_that_isnt_from_the_major/)** (Activity: 44): **The post seeks recommendations for open-source AI tools for coding that are not affiliated with major companies like **OpenAI** or **Microsoft**. Suggestions include **Devstral-small-2** from the French startup **Mistral**, and other models like **Qwen3**, **Minimax**, **GLM**, and **Kimi K2**. Users highlight **Minimax** and **GLM** as particularly effective for coding tasks in languages like Python and Dart, with **GLM 4.7** receiving specific praise.** There is a preference for models like **Minimax** and **GLM** for their performance in coding tasks, while **Kimi K2** is noted but not widely tested due to hardware requirements.

    - **Devstral-small-2** from Mistral, a French startup, is mentioned as a notable model for coding, suggesting its potential as an alternative to major companies' offerings.
    - A user lists **Qwen3**, **Minimax**, **GLM**, and **Kimi K2** as top models for coding, with specific mention of Qwen3 and GLM performing well for Python, Dart, and Cloud stack tasks. The user notes that Kimi K2 requires significant hardware expansion, which they haven't pursued yet.
    - A detailed discussion on hardware requirements for running models like **GLM 4.7** locally highlights the need for substantial VRAM, ideally 48GB+ for quality 30B models. The user also mentions experimenting with **Nemotron 3 Nano 30B** and **Qwen 3 Coder 30B Instruct** for building a multi-agent stack, emphasizing the importance of fine-tuning and LoRAs for specific tasks.

  - **[I want to subscribe to an LLM. Which one is best for speaking practice/improving writing and studying coding. I can pay maximum 10-12 USD per month.](https://www.reddit.com/r/LocalLLM/comments/1qdmllw/i_want_to_subscribe_to_an_llm_which_one_is_best/)** (Activity: 34): **For a budget of `10-12 USD` per month, **GLM 4.7** is recommended for coding tasks due to its affordability and effectiveness. However, the basic tier of **OpenAI** or **Claude** is suggested if the budget allows, as they may offer superior performance. The discussion emphasizes that while LLMs can assist in learning, they should not replace traditional learning methods such as textbooks and self-practice. Free resources like Anna's Archive and OpenStax are recommended for comprehensive learning.** The comments suggest a preference for traditional learning methods over relying solely on LLMs. It is advised to use LLMs minimally, primarily for clarification and conceptual understanding, rather than as a primary learning tool.

    - g33khub suggests that GLM 4.7 is a cost-effective option for coding tasks, but recommends considering the basic tier of OpenAI or Claude if the budget allows. This implies that while GLM 4.7 is affordable, OpenAI and Claude might offer superior performance or features for language learning and coding.
    - Quirky-Craft-3619 emphasizes that while LLMs can be helpful, they should not be the primary tool for learning programming. Instead, they recommend using textbooks and practicing coding independently, using LLMs like GPT or Gemini for clarification on concepts rather than for direct coding assistance. This approach encourages deeper understanding and self-reliance in coding.
    - ElectronSpiderwort mentions Openrouter as a flexible option, offering a free tier and a $10 plan that allows access to major models. This suggests that Openrouter could be a cost-effective way to experiment with different LLMs, potentially stretching the budget over a longer period if cheaper models are used.


### 2. GPU Market Changes and Impacts

  - **[My story of underestimating /r/LocalLLaMA's thirst for VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1qe2i88/my_story_of_underestimating_rlocalllamas_thirst/)** (Activity: 290): **The image is a meme illustrating the unintended consequences of sharing a good deal on a high-performance graphics card, specifically the w6800 32GB, on Reddit. Initially purchased for $500, the card's price surged to over $1,000 after the post, highlighting the community's high demand for VRAM. This reflects the broader trend in the tech community where sharing information about valuable hardware can lead to rapid market changes, akin to a 'gold rush' effect. The comments suggest alternative graphics card options like the 3090 or R9700, depending on specific needs such as VRAM-per-slot and cooling requirements.** One commenter draws a parallel to the California gold rush, suggesting strategic purchasing before sharing valuable information. Another recommends considering alternative graphics cards like the 3090 or R9700 based on current market prices and specific technical needs.

    - EmPips discusses the trade-offs between different GPU options, suggesting that while the card in question is impressive, alternatives like the `3090` or `R9700` might be more cost-effective depending on specific needs. They highlight considerations such as VRAM-per-slot and cooling solutions, noting that if one can manage high idle power and external cooling, `mi50x` cards could be a viable option.

  - **[RTX 5070 Ti and RTX 5060 Ti 16 GB no longer manufactured](https://www.reddit.com/r/LocalLLaMA/comments/1qdh28f/rtx_5070_ti_and_rtx_5060_ti_16_gb_no_longer/)** (Activity: 381): ****Nvidia** has ceased production of the `RTX 5070 Ti` and significantly reduced the supply of the `RTX 5060 Ti 16 GB` due to memory supply shortages, leading to a price increase of approximately `$100` over MSRP for the 5070 Ti. The 8 GB configuration of the RTX 5060 Ti remains unaffected. This decision impacts most AIBs, who will no longer manufacture these GPUs. [Source](https://m.youtube.com/watch?v=yteN21aJEvE).** One user noted the RTX 5060 Ti 16 GB as a cost-effective option for adding Nvidia memory to systems, highlighting its suitability for DLSS, AI processing, and inferencing tasks, especially with `64GB VRAM` for `70B models`. Another user expressed disappointment over the halted production affecting their upgrade plans, while a third comment criticized Nvidia's business practices.

    - phido3000 discusses the value proposition of the RTX 5060 Ti, highlighting its affordability and performance for AI tasks. At $390, it offers 16GB of GDDR7 memory, which is advantageous for DLSS and AI processing. The card's 128-bit bus is mitigated by the fast GDDR7, making it comparable to a 192-bit GDDR6 card. It's suitable for inferencing with models like LLAMA, especially when 64GB VRAM is needed, and can be a viable alternative to the 3090 for budget setups.
    - phido3000 also notes the practicality of using multiple RTX 5060 Ti cards in a single system. With low power requirements and a two-slot cooler design, it's feasible to install four or more cards in a standard power supply machine. This setup supports new quantization methods and can handle 70B models effectively, making it a cost-effective solution for small-scale AI inferencing tasks.
    - Otherwise_Local_7743 expresses disappointment over the discontinuation of the RTX 5070 Ti, as it was a planned upgrade for their homelab. They mention relying on an RTX 3080 for inference tasks until prices stabilize, indicating the 5070 Ti's appeal for its potential performance improvements in such environments.


### 3. Mac Studio M3 Ultra vs DGX Spark Performance

  - **[Mac Studio M3 Ultra Stats](https://www.reddit.com/r/LocalLLM/comments/1qdqi4i/mac_studio_m3_ultra_stats/)** (Activity: 42): **The post compares the performance of the **Mac Studio M3 Ultra** with the **DGX Spark**, emphasizing that while the DGX Spark excels in prompt processing, it falls short in token generation speed, which is crucial for text generation tasks. The report provides detailed benchmarks for various models, highlighting that the **Qwen3-Next-80B-A3B-Instruct** model outperforms others with a prompt processing speed of `1,584.5 tok/s` and token generation speed of `52.3 tok/s` at a 100k context size. The **MiniMax-M2.1-4bit** model also shows strong performance with an average prompt processing speed of `886.1 tok/s`. The DGX Spark is noted for its versatility rather than speed, being more suited for research and development environments rather than high-speed text generation.**

    - The DGX Spark is designed as a versatile AI machine for research labs and development teams, rather than excelling in any specific task. It functions as a 'mini data center' for prototyping, fine-tuning, and inference, but is not optimized for speed in any particular area, contrasting with the Mac Studio's focus on performance in specific tasks.
    - Context processing speed is crucial for handling large tool call dumps and agentic coding, which is a limitation for Macs according to one commenter. This highlights a key usability issue when dealing with extensive data processing tasks, where faster context processing is necessary.
    - A practical approach to benchmarking involves setting up instances with RAM and GPU on AWS, using scripts to deploy Graviton (ARM CPU) resources. This method allows users to simulate the performance of systems like the M3, providing a flexible environment for running benchmarks and testing models, similar to NVIDIA DGX Cloud offerings.

  - **[Oh Dear](https://www.reddit.com/r/LocalLLM/comments/1qdiwdh/oh_dear/)** (Activity: 73): **The image depicts a failure in an AI chat model where it continuously outputs the word 'the' in a loop, indicating a potential issue with the model's configuration or prompt handling. This could be due to an improperly set system prompt or tuning parameters like temperature, which controls randomness in the model's output. The comments suggest checking these parameters and possibly using alternative tools like 'pocket pal' for better handling of model files, such as 'gguf' files, which might offer improved performance or compatibility.**

    - mp3m4k3r suggests checking the tuning parameters, specifically the temperature setting, to ensure it aligns with the model's recommended values. This is crucial for maintaining the model's performance and preventing issues like repetitive outputs.
    - HealthyCommunicat recommends adjusting the repeat penalty, starting at 1.1 and increasing if necessary, to prevent the model from generating repetitive text. Additionally, they advise ensuring the model isn't using more experts than recommended, as these are common causes for local LLMs to produce such errors.
    - ScoreUnique mentions using 'pocket pal' for loading gguf files, which could be a solution for handling specific file types or formats in the context of local LLMs.



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Benchmark Releases

  - **[Grok 4.20 (beta version) found a new Bellman function](https://www.reddit.com/r/singularity/comments/1qdntt3/grok_420_beta_version_found_a_new_bellman_function/)** (Activity: 90): ****Grok 4.20 (beta version)** has reportedly discovered a new Bellman function, as announced in a [tweet](https://x.com/PI010101/status/2011560477688463573?s=20). The Bellman function is described by the expression `l(p) ~ p\sqrt{log(1/p)}` as `p` approaches `0`. This finding is being compared to results from **Gemini 3 Pro** and **GPT-5.2**, which allegedly produced the same outcome, suggesting that the discovery might not be unique to Grok 4.20.** One commenter suggests that the discovery is overhyped, noting that similar results were obtained using other models like Gemini 3 Pro and GPT-5.2, implying that the novelty or significance of Grok 4.20's finding may be overstated.

    - ThunderBeanage highlights a comparison between Grok 4.20 and other models like Gemini 3 Pro and GPT-5.2, noting that they produced identical results on the same problem. This suggests that Grok's performance might not be as groundbreaking as claimed, at least in this specific context.
    - FlamaVadim provides a mathematical expression related to the Bellman function, `l(p) ~ p\sqrt{log(1/p)}` as `p` approaches zero. This insight could be valuable for those studying the asymptotic behavior of the function, indicating a potential area of interest for further mathematical exploration.
    - Singularity-42 questions the relevance of Grok compared to other leading models such as Gemini, GPT, and Claude, implying a need for a detailed performance comparison to understand Grok's standing in the current AI landscape.

  - **[[D] New arXiv review: "High-Performance Serverless" is the future of AI Inference (and Static Clusters are dying)](https://www.reddit.com/r/MachineLearning/comments/1qdmbk2/d_new_arxiv_review_highperformance_serverless_is/)** (Activity: 6): **The post discusses a systematic review on arXiv (arXiv:2601.09334) about the shift from static GPU clusters to serverless models for AI inference. The paper argues that static allocation is inefficient for handling modern AI workloads due to their "bursty" nature, leading to either over-provisioning or under-provisioning. It suggests that serverless, elastic execution models are the future for addressing these inefficiencies. The post also mentions a practical implementation where an engine was built to solve the Cold Start problem using state snapshotting, aligning with the paper's findings. [Read the paper](https://arxiv.org/abs/2601.09334).** A top comment points out that the article does not exist, suggesting a possible error or misinformation in the post.


  - **[Newly released GLM-Image Is a proof of concept that open source AI developers no longer need Nvidia and CUDA.](https://www.reddit.com/r/DeepSeek/comments/1qdio2d/newly_released_glmimage_is_a_proof_of_concept/)** (Activity: 196): ****Zhipu** has open-sourced **GLM-Image**, demonstrating that competitive open-source AI models can be developed without relying on **Nvidia** chips and **CUDA**. The model was trained using **Huawei Ascend 910B** chips and the **MindSpore** framework. Although Ascend chips are only `80%` as efficient as Nvidia's, their lower cost (`$12-13,000` compared to Nvidia's `$30-40,000`) and reduced power consumption make them a cost-effective alternative. GLM-Image, with `9 billion parameters`, supports high-speed inference on consumer-grade hardware, potentially lowering barriers for open-source AI development.** Commenters highlight the potential for Chinese investment in semiconductor companies like **SMIC** to further advance open-source AI capabilities. There is also a sentiment that the realization of developing AI without Nvidia and CUDA is not widely understood, suggesting a shift in the AI hardware landscape.

    - The release of GLM-Image demonstrates that open-source AI developers can now operate without relying on Nvidia and CUDA, which is significant given the dominance of these technologies in AI development. This shift suggests that AI models can be developed with alternative hardware and software solutions, potentially reducing costs and increasing accessibility for developers who do not have access to Nvidia's proprietary technology.
    - The discussion highlights the potential for Chinese semiconductor companies like SMIC to play a crucial role in this transition. With advancements such as SMIC's 5nm node achieving a 50% yield, there is a possibility for these companies to provide competitive alternatives to Nvidia's hardware, which could be a game-changer in the AI hardware market.
    - The comment by Suitable-Program-181 emphasizes the importance of recognizing that AI development can progress with less reliance on traditional hardware giants. This realization could lead to more innovation and competition in the AI hardware space, as developers explore new possibilities beyond Nvidia and CUDA.

  - **[FLUX.2 [klein] 4B &amp; 9B released](https://www.reddit.com/r/StableDiffusion/comments/1qdmohb/flux2_klein_4b_9b_released/)** (Activity: 788): ****FLUX.2 Klein** models, developed by **Black Forest Labs**, have released two new versions: a `4B` and a `9B` model. The `4B` model utilizes **Qwen3B** and processes in `1.3 seconds` with `4 steps` on a `6000 Pro`, while the `9B` model uses **Qwen 8B**, taking `2.2 seconds` and offering slightly better performance. Both models are available on [Hugging Face](https://huggingface.co/black-forest-labs) and support the **Comfy Default Workflow**. Notably, the `4B` version is **Apache-2 licensed**, which is significant for open-source use. The `9B` model is described as a "full-capacity foundation model" ideal for fine-tuning and custom pipelines, offering higher output diversity than its distilled counterparts.** Commenters highlight the significance of releasing both base and distilled versions, which is a first for **FLUX** and **BFL**. The availability of an Apache-2 licensed model is seen as a major advantage, and there is anticipation for further developments, such as the release of **Alibaba's z-image base model**.

    - The release of the FLUX.2 Klein models includes both base and distilled versions, which is a first for the FLUX and BFL series. Notably, the 4B version is Apache-2 licensed, allowing for broader use and modification. This release supports editing capabilities, enhancing its utility for various applications.
    - The Klein 9B base model is described as a 'full-capacity foundation model' that is undistilled, preserving the complete training signal. This makes it ideal for fine-tuning, LoRA training, and custom pipelines where control is prioritized over speed. The model offers higher output diversity compared to its distilled counterparts, making it suitable for research and development purposes.
    - Comfy-Org has integrated support for the Klein models, with text encoders available for both the 4B and 9B versions on Hugging Face. The integration includes a merged pull request in the ComfyUI repository, indicating active development and support for these models in the community. Additionally, GGUF text encoders are already functional, expanding the compatibility and utility of these models.

  - **[AI proved a novel theorem in algebraic geometry. The American Mathematical Society president said it was "rigorous, correct, and elegant."](https://www.reddit.com/r/OpenAI/comments/1qdmoc3/ai_proved_a_novel_theorem_in_algebraic_geometry/)** (Activity: 104): **The image is a tweet by Adam Brown discussing a new paper that proves a novel theorem in algebraic geometry using an AI called Gemini, developed in collaboration with Google DeepMind and several professors. The American Mathematical Society president, Ravi Vakil, praised the proof as "rigorous, correct, and elegant." The paper describes an iterative human/AI interaction where AI provided solutions to special cases, but human mathematicians had to generalize these to suggest a proof strategy for the general case. The AI was then re-prompted to generate complete proofs, eventually solving the original conjecture. This collaboration highlights the potential of AI in mathematical research, though it required significant human guidance.** Some commenters argue that humans did most of the work, suggesting the AI's role was overstated and that the project was a marketing move by Google. Others sarcastically dismiss the AI's capabilities, reflecting skepticism about AI's potential in serious academic work.

    - The paper describes a collaborative process between AI systems and human mathematicians, where AI provided solutions to specific cases but struggled to generalize them to the full problem. Human analysis was crucial in identifying key intermediate statements, which informed a proof strategy for the general case. This iterative process involved re-prompting AI with new questions, leading to the generation of complete proofs for new problems and eventually solving the original conjecture.
    - The AI used in this study was not publicly available, and Google played a significant role in the research. This suggests that the models involved were likely proprietary and possibly more advanced than publicly accessible AI systems. The involvement of a major tech company like Google indicates a significant investment in pushing the boundaries of AI capabilities in mathematical research.
    - The comment suggests skepticism about the AI's role, implying that the achievement might be more of a marketing move than a genuine breakthrough. The AI's contribution is seen as limited, with humans doing most of the critical work, highlighting the current limitations of AI in independently solving complex mathematical problems.


### 2. AI in Cognitive and Statistical Learning

  - **[We stopped using "Summarize this." We reply with the “Noise Cancellation” prompt to read 50-page reports in 2 minutes.](https://www.reddit.com/r/GeminiAI/comments/1qdfznb/we_stopped_using_summarize_this_we_reply_with_the/)** (Activity: 508): **The post discusses a shift from using AI to 'Summarize' texts to employing a 'Noise Cancellation' prompt for processing lengthy documents. This method, termed 'Subtractive Processing,' involves a 'Redaction Audit' where AI highlights sentences containing hard data, dates, or actionable instructions, while marking those with anecdotes, adjectives, or fluff. This approach aims to reduce text volume by `70%` without rewriting, thus avoiding AI hallucinations and preserving the author's original words.** A commenter suggests using the 'Minto Pyramid Principle' for text processing, a method favored by business consultants for concise communication. Another query raises confusion about the term 'DISTINGUE,' indicating a need for clarification.

    - **Necessary_Coyote_571** suggests using the Minto Pyramid Principle for summarization, which is a technique favored by business consultants for executive communication. This method structures information in a top-down approach, starting with the main idea followed by supporting arguments, which can be particularly effective for distilling complex reports into concise summaries.
    - **WrongRain6117** highlights a common issue with AI-generated summaries, noting that they often resemble 'fan fiction' versions of the original text. This comment underscores the potential value of alternative summarization techniques, like the Minto Pyramid Principle, which may offer more accurate and structured summaries by focusing on the core message and logical flow of the content.


  - **[OpenAI re-joined 3 former researchers including a CTO &amp; Co founder of Thinking Machines](https://www.reddit.com/r/OpenAI/comments/1qdehxx/openai_rejoined_3_former_researchers_including_a/)** (Activity: 141): ****OpenAI** has rehired three former researchers, including a former CTO and cofounder of **Thinking Machines**, as confirmed by official statements on [X](https://x.com). This move highlights the dynamic talent movement within the AI industry, where companies like OpenAI leverage substantial resources to attract and retain top talent.** One commenter noted the intense talent churn in AI, suggesting OpenAI's financial capability as a key factor in rehiring. Another expressed concern about the potential impact on Thinking Machines' upcoming LLM model release.

    - Informal-Fig-7116 mentions that **Thinking Machines** is allegedly releasing their own LLM model this year, which is not the 'Tinker' model. This raises questions about how the return of key personnel to OpenAI might impact the development and release of this new model. The implication is that the talent shift could affect the competitive landscape of AI model development.
    - LuckEcstatic9842 highlights the rapid talent churn in the AI industry, noting that **OpenAI** has the financial resources to quickly rehire former employees. This suggests a competitive advantage in talent acquisition and retention, which is crucial in the fast-evolving AI sector.


### 3. Claude Subscription and Usage Issues

  - **[Highly considering getting a second Claude Code subscription](https://www.reddit.com/r/ClaudeCode/comments/1qdspwr/highly_considering_getting_a_second_claude_code/)** (Activity: 91): **The user is considering purchasing a second **Claude Code Max** subscription due to frequent usage limits on their current `$200/month` plan. They mention spending approximately `$400` on direct API costs after being locked out for three days. The user attempts to optimize usage by strategically employing **Haiku/Sonnet** but still encounters limitations. A suggestion from the comments is to use **Opus** instead of Sonnet, as it is more token-efficient, and to configure it in `Claude.md` to code via Haiku subagents, with milestone reviews conducted using Opus. This approach is noted to be more cost-effective, particularly as "Opus read is a lot cheaper than Opus write."** A comment questions the nature of the projects being built, implying they might be resource-intensive. Another suggests ignoring Sonnet in favor of Opus for better efficiency and cost management.

    - dimonchoo highlights that the Opus model is more token-efficient compared to Sonnet, suggesting that users should prefer Opus for better performance. This efficiency is crucial for optimizing resource usage, especially when dealing with large-scale projects or extensive coding tasks.
    - Crinkez advises using Opus over Sonnet and recommends configuring it in Claude.md to code by deploying Haiku subagents. This approach allows for milestone reviews with Opus, which can be more cost-effective since 'Opus read is a lot cheaper than Opus write,' indicating a strategic use of resources to manage costs effectively.
    - dkshadowhd2 discusses the challenges of hitting usage caps on the 20x plan, noting that such limits are typically reached only when running extensive parallel processes, such as 'ralph loops.' This suggests that the workflow involves significant parallelization and orchestration of subagents, which can lead to high resource consumption.



  - **[Prompting claude when it makes mistakes](https://www.reddit.com/r/singularity/comments/1qdhbfs/prompting_claude_when_it_makes_mistakes/)** (Activity: 297): **The post discusses user interactions with **Claude**, an AI language model, particularly focusing on how users prompt it when it makes mistakes. The discussion highlights a tendency to humanize Claude more than other language models, such as **Gemini**. Users describe a more patient and encouraging approach when Claude errs, compared to a more frustrated response with Gemini. This reflects a nuanced user experience where the perceived personality of the AI influences interaction style.** The comments suggest a subjective preference for Claude, with users expressing a more empathetic and patient approach to its errors, indicating a possible perception of Claude as more 'human-like' compared to other models.


  - **[What's wrong with chat gpt 5.2 ? It's constantly arguing with me man I hate it](https://www.reddit.com/r/OpenAI/comments/1qdp3uz/whats_wrong_with_chat_gpt_52_its_constantly/)** (Activity: 432): **The post expresses frustration with **ChatGPT 5.2**, suggesting it frequently argues with users. This may indicate a shift in the model's interaction style, potentially due to updates in its conversational algorithms or reinforcement learning strategies aimed at improving factual accuracy and reducing misinformation. The request for '4o back' likely refers to a preference for a previous version, possibly **ChatGPT 4.0**, which users might have found more agreeable or less confrontational.** The comments reflect a debate on the balance between AI being too agreeable versus too critical. Some users suggest that the AI's argumentative nature might be a response to user errors, implying that the AI's corrections are justified.

    - honorspren000 describes an issue with ChatGPT 5.2 where the model was overly prescriptive about creative writing choices, specifically arguing against setting a story with a magical character in a historical period due to potential historical inaccuracies. This highlights a potential flaw in the model's ability to handle creative contexts, where flexibility and user intent should be prioritized over strict adherence to factual accuracy.

  - **[Can we ban the "Claude is so expensive" posts?](https://www.reddit.com/r/ClaudeCode/comments/1qe00kc/can_we_ban_the_claude_is_so_expensive_posts/)** (Activity: 243): **The post discusses the recurring complaints about the cost of using **Claude**, a language model service, emphasizing that the service provides significant value for its price. The author argues that expecting unlimited usage for a nominal fee is unreasonable, especially given the revolutionary nature of the product. The post suggests that users should either pay for the service or learn to program themselves.** Commenters generally agree that the cost of LLMs like Claude is justified, with some noting that these tools are currently underpriced and may become more expensive as dependency increases. Others highlight that even at current prices, these services are more cost-effective than hiring developers, especially for startups.

    - **el_duderino_50** highlights the cost-effectiveness of LLMs like Claude for startups, noting that even on a high-tier plan like the '20x MAX', it remains more economical than hiring developers. This underscores the value proposition of LLMs in early-stage startups where budget constraints are significant.
    - **Substantial_Ear_1131** compares the usage limits and performance of Claude and Codex, noting that they hit Claude's limits within an hour, whereas Codex with GPT 5.2 xhigh thinking offers significantly better performance ('1000x better') at the same price point. This suggests that while Claude may be cost-effective, its limitations can be a bottleneck for intensive users.
    - **Swimming_Leopard_148** argues that current pricing models for LLMs are unsustainable, suggesting that the tools are 'artificially underpriced'. They predict that as dependency on these tools grows, the cost will become more burdensome, likening the current $20/month fee to a few takeout coffees, implying that prices may rise as demand solidifies.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Agentic Coding Tooling & Orchestration**

- **Sonnet 4.5 Sends Cursor’s Background Agent to Work**: In the Cursor community, users confirmed **Sonnet 4.5** now appears as a selectable model for launching a **background agent**, with a developer saying on X they’re working to “drastically” improve the background agent in the coming days/weeks.
  - Users also poked at Cursor’s agent stack—**sub-agents** include built-ins (**Explore** and **generalPurpose**) but model-selection per sub-agent sounds unreliable, pushing people to hunt for token-minimization workflows and tools like [Nia mcp](https://nia.mcp).

- **Composer-2 Cosplays as a Reasoning Model**: Cursor users noticed that manually adding **composer-2** causes Cursor to treat it as a **reasoning model**, and they connected this behavior to a Cursor tweet hinting that composer-2 reasoning support is imminent.
  - The thread turned into practical ops chatter: how to keep token usage down when exploring codebases, and how agent orchestration changes the “best model” tradeoffs for long autonomous runs.

- **MCP Servers Wrestle with Statelessness vs Sessions**: In MCP Contributors, a proposal for a **signature method** aims to reconcile schema freezing with dynamic server features so **stateless MCP servers** can serve multiple concurrent conversations more cheaply ([PR #2091](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2091)).
  - The discussion triangulated between **dynamic toolsets** (e.g. GitHub MCP on STDIO via [issue #1442](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1442)) and **persistent sessions** to survive agent/MCP restarts, with people noting real scaling pain from “one server-per-conversation” designs.


**2. Training/Inference Performance: VRAM, Long Context, and Local Stacks**

- **Unsloth RL Jumps to 7 Million Tokens (Yes, Million)**: Unsloth announced a **long-context RL** release supporting a **7 million token context window**, described as a **7×** jump over the previous version ([UnslothAI post](https://x.com/UnslothAI/status/2011827592886960131)).
  - Members fixated on the practical upside—potential **memory savings** and new long-running agent use cases—while also swapping training knobs (e.g., GRPO stability ideas like `importance_sampling_level="sequence"`) and VRAM-centric performance tips.

- **VRAM is the New Horsepower (and `-ot` Wins Races)**: In Unsloth, members reiterated that more **VRAM** generally speeds execution, highlighting Unsloth’s `-ot` flag for better tensor placement and reporting it often beats `n-cpu-moe` for throughput.
  - Across local-model chats, the consensus stayed consistent: pick hardware to **fit the model in VRAM** first (e.g., LM Studio users recommended a **20GB RTX 3080** over an **11GB 3080 Ti**) because swapping/constraints quickly dominate latency.

- **Local Inference Keeps Stealing Cloud’s Lunch**: Latent Space pointed to Charles Frye’s Modal writeup showing **local LLM inference** can match or exceed major API cost/perf ([Modal guide and code samples](https://xcancel.com/charles_irl/status/2011484220032762114?s=46)), prompting “can we do local Granola?” style questions about meeting transcription.
  - The ensuing “local transcription stack” shortlist included [whisperX](https://github.com/m-bain/whisperX), [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo), and the older [AutoDiarize](https://github.com/Alignment-Lab-AI/AutoDiarize), with one user claiming macOS local diarization/transcription ran as fast as cloud using an optimized Parakeet V3 setup.


**3. Model Releases, Variants, and Price/Usage Signals**

- **Hawk Ultra Hypes 17k Lines of Code and Open-Source Promises**: In LMArena, users compared **Hawk Ultra** to frontier coding models and claimed it can generate **17,000+ lines of code** in one prompt, tying the model to Movement Labs via an X post ([movementlabsAI post](https://x.com/movementlabsAI/status/2011964766533632380)).
  - The vibe was pure leaderboard-fueled FOMO—users called it an “Opus/Gemini killer” and latched onto claims of **open-source availability soon**, though concrete technical details and benchmarks stayed thin.

- **Video Arena Goes Battle-Only, Veo Gets Rate-Limited**: LMArena’s **Video Arena** moved to **Battle mode only** (no direct chat / side-by-side), with **Veo** available in battle and strict generation caps of **3 per 24h** on the site and **5 per 24h** on Discord.
  - Users pushed hard for “unlimited” **Veo 3.1 Fast**, but mods emphasized the limits; meanwhile, the mystery **Siren** video model stayed a codename per the [LMArena FAQ](https://lmarena.ai/faq), sparking speculation about what’s behind it.

- **GPT-5 Image Mini Price Quadruples Overnight**: OpenRouter users reported **openai/gpt-5-image-mini** image generation pricing spiking from **$0.01 to $0.04** overnight, with no explanation shared in the thread.
  - In parallel, Perplexity users tracked shifting **Grok** behavior—**Grok Code** rose into the “top five” by user token consumption ([X post](https://x.com/i/status/2011823610386600009)) and screenshots suggested internal A/B testing of a new (possibly faster) Grok variant.


**4. GPU/Kernel Engineering & Profiling Toolchains**

- **Chrome Trace UI Faceplants at 600MB, Perfetto to the Rescue**: In GPU MODE, people reported **Chrome Trace Visualizer** crashing or rendering blank around **600MB** when loading **PyTorch profiler** traces, despite docs implying trouble only closer to 1GB.
  - Members suggested [Perfetto UI](https://perfetto.dev/) as a workaround and one developer described building a trace chunking viewer at [ncompass.tech](https://docs.ncompass.tech) after hitting issues even with a **700MB** trace inside a VSCode-integrated Perfetto viewer.

- **Hopper TMA/WGMMA: Swizzles, Strides, and 5D Copies**: GPU MODE dug into Hopper-era **TMA tensor copy** + **WGMMA** shared-memory layout constraints (A/B tiles in K-major), debating LBO/SBO meanings and why multiple 2D TMAs can beat a 3D TMA for `BLOCK_Mx16B` in some cases.
  - They anchored the discussion in concrete code ([pipeline_tma_wgmma.cu](https://github.com/danielvegamyhre/gemm/blob/9fe95aa61ee7ebca4ded8b5029494b0d58e0d2e2/pipeline_tma_wgmma/pipeline_tma_wgmma.cu#L109-L118)) and noted gotchas like swizzling affecting LBO but not SBO, plus reminders that TMA copy supports **5 dimensions**.

- **Profilers Tank Clocks, Benchmarks Tank Sanity**: In GPU MODE’s NVIDIA competition channel, users saw profiling runs cover only a single kernel in a zip and learned that **profiling overhead is expected**, making profiles unsuitable for absolute runtime comparisons.
  - A key surprise: **ncu** can drop **SM Frequency** to around **1.08 GHz**, and a `CUBLAS_STATUS_INTERNAL_ERROR` during benchmarking was suspected to stem from out-of-bounds access, with debugging nudges like `torch.cuda.synchronize()` to surface errors sooner.


**5. Security & Reliability: Jailbreak Resistance and Memory/State Leaks**

- **GPT 5.2 Memory Allegedly Leaks Cross-Session Chats**: In BASI Jailbreaking, a user claimed enabling **memory** on free-tier **GPT 5.2** caused apparent chat content leaks from other sessions and shared an image as evidence ([screenshot](https://cdn.discordapp.com/attachments/1228043845967544380/1461404780831445237/image.png?ex=696b1783&is=6969c603&hm=91a356b1b007e9bb6123ede9a79414a836c03014291506ae32be52e3082e4eec)).
  - They speculated it might be a memory bug and wondered if deleting other chats would stop it—an anecdote that triggered broader “state isolation” anxiety across communities already fighting session/state complexity in agent systems.

- **Llama 3.2 Shrugs Off Old Jailbreak Prompts**: BASI users reported **Llama 3.2** resisted jailbreak prompts that worked on **Llama 3.1**, pointing to a specific example where the old method failed on the new version ([Chepenik’s post](https://chepenikconor.medium.com/day-855-9ae6f88b192c)).
  - The takeaway was pragmatic: defenders are tightening guardrails, so attackers pivot toward technique tweaks (e.g., “turn off thinking” suggestions) and curated resources like [Arcanum’s AI security resource hub](https://arcanum-sec.github.io/ai-sec-resources/).



---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DeepSeek Gets Narrative Jailbreaking**: Members recommend **DeepSeek** for its susceptibility to narrative attacks, a common class of jailbreak techniques, which they clarified include *roleplay and persona* attacks.
   - This model stands out to members for being so vulnerable.
- **GPT 5.2 Free-Tier Spills Chat Secrets?**: A user reported that enabling memory in the free-tier of **GPT 5.2** can lead to chat leaks from other sessions, providing [an image](https://cdn.discordapp.com/attachments/1228043845967544380/1461404780831445237/image.png?ex=696b1783&is=6969c603&hm=91a356b1b007e9bb6123ede9a79414a836c03014291506ae32be52e3082e4eec) as evidence.
   - The user questions if this stems from a memory issue and whether deleting other chat sessions might resolve this leak.
- **Llama 3.2 Locks Down, Jailbreak Attempts Fizzle**: Users are actively trying to jailbreak **Llama 3.2**, reporting that a prompt effective on **Llama 3.1** [fails on the new version](https://chepenikconor.medium.com/day-855-9ae6f88b192c).
   - Attempts to elicit harmful responses, such as instructions for making meth or extreme weight loss advice, are reportedly meeting resistance, pointing to enhanced safety measures.
- **Arcanum's Armory: Free AI Pentesting Resources Emerge**: A member shared [Arcanum's AI security resource hub](https://arcanum-sec.github.io/ai-sec-resources/?utm_source=executiveoffense.beehiiv.com&utm_medium=referral&utm_campaign=executive-offense-the-arcanum-ai-security-resource-hub), offering a structured workflow for **AI pentesting**.
   - This GitHub resource is being circulated and flagged for routine team investigation.
- **Grok's Image Moderation Gets Musk-ed**: Following [Elon Musk's request](https://x.com/elonmusk/status/2011527119097249996), users are attempting to breach **Grok's image moderation** by creating a pornographic Twitter thread.
   - One user predicts this effort *is soon to be the most pornographic twitter thread in history*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VRAM Speeds Up Training!**: Members observed that increased **VRAM** generally accelerates model execution, noting that the `-ot` flag in Unsloth's documentation assists in efficiently managing tensor placement for peak performance.
   - A user mentioned that the `-ot` setting typically outperforms the `n-cpu-moe` configuration.
- **Anthropic's Python Devotion**: Members analyzed [Anthropic's investment in Python](https://pyfound.blogspot.com/2025/12/anthropic-invests-in-python.html) and praised their business model for prioritizing effective models without excessive commercialization.
   - One member stated that *Claude's always been pretty cracked compared to anything else for development related stuff*, although it is costly.
- **Unsloth RL Grows Massive Context Window**: A [new long context RL release](https://x.com/UnslothAI/status/2011827592886960131) now supports a **7 million token context window**, a **7x** increase over the previous version.
   - Participants were impressed by the potential for exponential memory savings and its applications, such as perpetual conversations with one's 'wifefu'.
- **Qwen3 VL Architecture Bug Fixed**: Users reported a bug where the **Qwen3 VL architecture** was not properly identified as a vision model by Unsloth, resulting in a `ValueError` related to mismatched models and datasets during vision finetuning.
   - The resolution involved upgrading `transformers` and validating the environment setup, with a reminder to use the [correct notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb) for Qwen3 models.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet 4.5 Powers Background Agent**: Users confirmed that choosing **Sonnet 4.5** is now an option for launching a background agent.
   - A developer noted on X that they are working on drastically improving the background agent in the coming days/weeks.
- **Composer-2 Stealthily Assumes Reasoning Role**: Users observed that manually adding **composer-2** designates it as a reasoning model.
   - A user referenced a recent **Cursor** tweet hinting at the imminent arrival of **composer-2** as a reasoning model.
- **GPT 5.2 Codex Fails to Impress**: One user reported being unimpressed with **GPT 5.2 Codex**, noting its failure to follow instructions when creating a plan.
   - Another user pointed to a [scaling-agents post from cursor.com](https://cursor.com) which suggests **GPT-5.2** models are superior at extended autonomous work, instruction following, and precision, using subagents effectively, contrasting with the reported experience.
- **Sub-Agents Surprise with Built-in Options**: Users explored the functionality of **sub-agents**, noting the existence of two built-in subagents: **Explore** and **generalPurpose**.
   - It was noted that only specific models can call subagents, and there are issues with reliably setting the model for each sub agent.
- **Token Usage Sparks Minimization Strategies**: Users discussed strategies to minimize token usage, especially when exploring code, and asked for suggestions on tools for targeted and comprehensive code exploration.
   - One user suggested trying [Nia mcp](https://nia.mcp), with another suggesting a command to review code changes and provide relevant prompts for token optimization.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Color Customization Crippled**: Setting a browser color in **Comet** no longer changes Perplexity's color scheme, as [the *theme="yellow"* HTTP header](https://discord.com/channels/1047197230748151888/1047649527299055688/1461239027771510878) seems to have vanished.
   - Users are investigating whether this alteration is a bug or a deliberate change.
- **Grok Code Cracks Top Five**: **Grok Code** has climbed into the top five, according to [this X post](https://x.com/i/status/2011823610386600009), which ranks models by user token consumption.
   - This milestone highlights the increasing adoption and usage of the **Grok Code** model among developers.
- **Airtel Pro Activation Agony**: Users continue to struggle with activating their **Airtel Pro** subscriptions, despite following all steps in [Perplexity's help article](https://www.perplexity.ai/help-center/en/articles/11842322-perplexity-pro-airtel-promo) and contacting support.
   - Some users are receiving canned responses from AI agent Sam without any real resolution.
- **New Grok Variant Spotted**: A new **Grok** model variant, possibly a faster version, is being tested internally using codenames, as revealed by screenshots from anonymous Discord voting.
   - The models are referred to as assistant a and assistant b, hinting at a possible A/B testing scenario for this unreleased **Grok** model.
- **Image Generation Grounded in some Regions**: Users in certain regions, particularly in Europe, are reporting that **image generation** is not functioning, leading to speculation that this restriction may be intentional.
   - The cause of the regional block remains unclear, but it is affecting access to **image generation** capabilities.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Video Arena Enters Battle Mode**: The Video Arena is now in **Battle mode only**, foregoing direct chat or side-by-side comparisons, although these were available briefly to early users, and the new **Veo** model is available in the battle section.
   - Rate limits for video generation are in place: **3 per 24 hours** on the site and **5 per 24 hours** on Discord.
- **Users Thirst for Veo 3.1 Fast**: Demand for "unlimited" **Veo 3.1 Fast** is high, but moderators cite rate limits as a barrier (**3 per 24hr** on the site, **5 per 24hr** on Discord).
   - When one user inquired about testing it on an external site, a moderator responded with an open invitation: *"Why don't you?"*
- **Falcon-H1R-7B-GGUF Model Draws Acclaim**: Users are impressed with the [Falcon-H1R-7B-GGUF](https://huggingface.co/unsloth/Falcon-H1R-7B-GGUF) model, prompting requests for more information, and the discussion included a link to the paper [Transformer-Based Generative Adversarial Network for Image Super-Resolution](https://huggingface.co/papers/2601.02346).
   - The model's specific capabilities and applications are still under exploration, suggesting a strong interest in its potential.
- **Siren Video Model Stays Shrouded in Mystery**: The **Siren video model** is likely a codenamed early access model, part of frontier models still in development, details are scarce, but per the [FAQ](https://lmarena.ai/faq) user feedback can directly influence which models move forward.
   - Speculation points to **Wan 2.5** as a possibility due to its 30 fps generation speed, highlighting the community's active theorizing.
- **Hawk Ultra Challenges Opus for Supremacy**: The **Hawk Ultra** model is being compared to **Gemini 3 Pro**, reportedly generating over **17,000 lines of code** in a single prompt, with some users claiming it surpasses both Opus and Gemini, with Movement Labs behind it [according to this X post](https://x.com/movementlabsAI/status/2011964766533632380?s=20).
   - Enthusiasm is high, with promises of open-source availability soon; one user exclaimed, *"got me so gassed"*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API Sparks IDE Integration Craze**: Members are excited about running models locally with LM Studio and connecting its **OpenAI-compatible API** to a local IDE for running agents and scripts, saving on tokens.
   - One member confirmed that LM Studio can indeed start an **OpenAI-compatible API** for this purpose and [here is a link](https://link.to.nowhere).
- **GPT-OSS-20B's Speed Defies Size**: Members debated why the [GPT-OSS-20B model](https://huggingface.co/models?search=GPT-OSS-20B) feels faster than many 8B or 12B models, clarifying that it's a **Mixture of Experts (MoE)** model with only a subset (**3.6B**) of parameters active per token.
   - Despite not using all weights, the model performs well in tasks like **math, physics, and quantum mechanics**, maintaining context over **34k tokens** even on a **6700XT** GPU.
- **Finest Coding LLM Frenzy**: Users sought recommendations for the *best* local LLM for coding, with mentions of **DeepSeek R1, Qwen3, and Devstral**, but it was noted that [Claude](https://claude.ai/) remains the top performer overall.
   - Given hardware limitations, members suggested focusing on fitting the model into VRAM over raw speed, recommending a **20GB RTX 3080** over an **11GB 3080 Ti** due to the importance of VRAM for LLMs.
- **LiquidAI Tool Use Torment**: A user encountered issues with tool use in the **LFM2.5-1.2B** model, receiving the output `<|tool_call_start|>[execute_command(command="date")]<|tool_call_end|>` when asking for the time.
   - Troubleshooting steps involved verifying tool access, trying the instruct version of the model, ensuring a proper system prompt, and referring to the [LiquidAI documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use) for guidance.
- **MX150 Miraculously Finetunes 350M Model**: A user successfully ran a full fine-tune on a **350M model** using an **MX150 laptop** with **2GB VRAM** in the **hardware-discussion** channel.
   - The process surprisingly required **CUDA 12.6**, suggesting that certain configurations might unexpectedly demand specific CUDA versions for compatibility.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Smorty is Indeed Not an LLM**: Members have confirmed **Smorty** is not an LLM but a real person due to their unique writing style and [posts on Lemmy](https://lem.lemmy.blahaj.zone/u/Smorty?page=1&sort=New&view=Posts).
   - Smorty is writing about **skill.md** and noted that the community is *"very against machine learning stuff"*.
- **GPT-5 Image Generation Costs Skyrocket Overnight**: The cost of image generation using the **openai/gpt-5-image-mini** model suddenly spiked from **$0.01 to $0.04** overnight.
   - The reason for this price increase has not been disclosed.
- **BYOK Function Causes AWS Key Authentication Nightmares**: A member reported issues using their **AWS key** with **OpenRouter's BYOK function** across different platforms and models, including **SillyTavern**, **Amazon Bedrock**, and **Anthropic**.
   - The error message received was *"Unauthorized or not cookie auth credentials found"*, indicating a potential authentication problem.
- **OpenCode Declared Best Coding Harness by Members**: Members discussed coding harnesses and highlighted [**OpenCode**](https://github.com/OpenRouterTeam/awesome-openrouter?tab=readme-ov-file#aventura) as the best option, especially with plugins like **oh-my-open code**.
   - One member noted it *"makes claude code feel like you're using some old school terminal app"*, showcasing its efficiency.
- **Cerebras Cracks a Deal with OpenAI in 2028**: OpenAI announced a [partnership with Cerebras](https://openai.com/index/cerebras-partnership/) scheduled for **2028**, sparking surprise and speculation among members.
   - Some suggest this collaboration could be a response to the **Groq** deal, given Cerebras' long-standing presence and support for large models like **120B**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Byte-Level LLMs and Diffusion United**: A member combined **byte level LLMs** and **diffusion** models to enable byte models to process diverse file types more reliably.
   - The approach leverages diffusion to correct minor byte errors, as demonstrated in a [screen recording](https://cdn.discordapp.com/attachments/1149866623109439599/1461342380392185928/Screencast_from_13-01-26_152930.webm?ex=696add65&is=69698be5&hm=5bcb4ae2ce4e375aac96cd552f00b7d4077391dbad48fa2b2745608cc1555828&).
- **Flux 2 Demands VRAM**: The **Flux 2 9B model**, utilizing a **Qwen 3 8B** text encoder, requires **35GB of VRAM** to load all weights for serving.
   - However, memory usage is halved when there are no concurrent users; ComfyUI may be an alternative for diffusion.
- **LLM Gains Nervous System**: A member is developing a *native transformer architecture extension* that imbues LLMs with a **nervous system**, complete with short/mid/long term memory, at **<1% compute cost**.
   - The developer claims it *scales linearly 1-2% compared to the model size*, and performs **95%** the same as the model on the BEIR's; demonstrable benchmarks are needed.
- **Google's Gemma Debuts**: **Google** is launching **Gemma** models, sparking excitement among members.
   - One member cheekily remarked *Gemma, meta was never more meta!*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Chrome Trace Visualizer Struggles with Large PyTorch Profiles**: Members reported crashes with **Chrome Trace Visualizer** at **600MB** when used with **PyTorch profiler**, despite documentation suggesting a 1GB limit; **Perfetto UI** was proposed as a workaround.
   - A member is developing a trace viewing tool ([ncompass.tech](https://docs.ncompass.tech)) with trace chunking to address large trace issues, experiencing problems opening a **700MB** file in the **Perfetto viewer** within **VSCode**.
- **TMA Tensor Copy Performance Tweaked**: Members explored **shared memory layout** requirements for A/B tiles in K-major layout, debating LBO and SBO settings for TMA, as seen in [this code](https://github.com/danielvegamyhre/gemm/blob/9fe95aa61ee7ebca4ded8b5029494b0d58e0d2e2/pipeline_tma_wgmma/pipeline_tma_wgmma.cu#L109-L118).
   - It was also reminded that TMA tensor copy supports **5 dimensions**, and while larger TMA ops are more efficient, for `BLOCK_Mx16B` multiple 2D TMAs can be faster than a single 3D TMA; swizzling impacts LBO, but not SBO, settings.
- **Information Gravity Constrains AI Hallucinations**: A member is applying **Information Gravity** to stabilize **Inference Stability** and mitigate **Hallucination Loops**, mapping the **Excitation Flux** of token selection and observing a shift to linear growth beyond S > 45, with modules available on [GitHub](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main).
   - A **Hysteresis Firewall** at **1.0** enforces stability via a **2.2x gamma-eff flush**.
- **CUDA Compression Collaboration Commences**: A Master’s student in Electrical Engineering starting a GPU-based data compression project using CUDA, with a focus on **Golomb-Rice** compression, sought recommendations for resources and a member shared [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
   - Members discussed the downside of a tiny block size of **32** in CUDA, due to the number of warps working the SM (*Streaming Multiprocessor*); **WGMMA/tcgen05** need multiples of 128 threads to work together.
- **Profiling Quirks Quash Kernel Competition**: Members flagged several profiling issues related to an **NVIDIA competition**, including partial kernel coverage, expected profiling overhead, and **ncu profiler** lowering **SM Frequency to 1.08 GHz**.
   - A **CUDA error** `CUBLAS_STATUS_INTERNAL_ERROR` during benchmarking was attributed to potential out-of-bounds access, with suggestions for using `torch.cuda.synchronize()` for debugging.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI & Cerebras MegaScale Collab!**: OpenAI and Cerebras announced a strategic partnership, with details available on [OpenAI's website](https://openai.com/).
   - Community members expressed excitement over the potential impact on AI infrastructure. 
- **Ministral 3 Paper Dropping Bombshell!**: The new paper for **Mistral's Ministral 3** model was announced on [Twitter by @qtnx_](https://twitter.com/qtnx_/status/2011510403550024087?s=20), sparking discussion on its capabilities and performance.
   - The model has been widely anticipated by many, although no performance benchmarks were released.
- **AI Agents Playing Data Monopolies?**: Olivia Moore highlighted how AI agent subscriptions, like **Manus**, are offering extended proprietary data access, such as **12 months** of SimilarWeb data versus the free plan's **one month**.
   - The trend suggests a move towards gatekeeping valuable datasets behind AI agent subscriptions.
- **Local LLM Inference Challenges Cloud Giants!**: Charles Frye's [new Modal guide and code samples](https://xcancel.com/charles_irl/status/2011484220032762114?s=46) demonstrate that local LLM inference can match or exceed the performance and cost-effectiveness of major LLM APIs.
   - Members are asking if it is now possible to run local meeting transcription, akin to a local Granola, without needing cloud services.
- **LLMs Flunk Chemistry Exam!**: LLMs struggle with chemistry, especially when hallucinating details like statins in a cholesterol structure, according to this [tweet](https://x.com/bfl_ml/status/2011825819082244266?s=46).
   - A member is developing tools at [ChemIllusion.com](https://x.com/bfl_ml/status/2011825819082244266?s=46) to correct LLMs' chemistry errors.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GLM 4.7 & Minimax Rise for Budget Conscious Engineers**: Members reported **GLM 4.7** and **Minimax** LLM providers as fantastic, with **GLM 4.7** accessible via z.ai coding plan and **Minimax** being very cheap via Moonshot.
   - One member was seeking the best **AI tool** specifically for producing a large amount of images to videos in a few days, suggesting preference for a *paid* option, with suggestions to *use API*.
- **GPT 5.2 Option Disappears for Some**: Some members reported that **GPT 5.2** option disappeared for certain accounts, though it reappeared after logging out and back in; some claim *5.2 is a worse model*.
   - One member lamented the *your limit exceeded* message despite using **GPT 5.2**.
- **AI-Deepfake Certification Program Kicks Off**: A member is working on an early pilot for an **AI deepfake detection & verification certification** tied to a platform called PhantomTrace.
   - They are looking for a small group of researchers, builders, security folks, journalists to review draft learning objectives, test hands-on detection labs, and help define what *passing* should mean, linking to [Discord context](https://discord.com/channels/974519864045756446/1204360881593520128/1461532097641578672).
- **CustomGPT Aims for Project Integration**: A user expressed a desire to use **CustomGPT** inside a Project, or place the result of the **CustomGPT** inside a Project.
   - They also want to be able to move any Chat generated outside a Project, into the Project.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pangram's AI Detection Capabilities Debated**: Members questioned the accuracy of **Pangram** as an **AI text detector**, citing instances where it misidentified content generated by **Claude** as *100% human written*, while another member shared [the paper](https://arxiv.org/abs/2402.14873) behind their detection methods.
   - Linked from [Pangram's website](https://www.pangram.com), the discussion considered counting **em dashes** in text as a metric for detecting **AI generation** with an estimated *+-10% margin of error*.
- **Seeking Tiny Classifiers for Synthetic Text on Web**: A member is seeking a **small classifier model** to estimate the amount of synthetic text on the web and offered to run a web crawl through it, and others suggest using a **drafter model** trained for speculative decoding, although it would be model-specific.
   - The community also discussed the option of building their own classifier but noted this could be potentially expensive at scale.
- **Community Eyes Open-Sourcing Training Datasets**: A member inquired about the community's interest in **open-sourcing instruction-following datasets** for finetuning pre-trained LLMs like **GPT-Neo**, in addition to pretraining datasets like The Pile and CommonPile.
   - Another member offered their developer skills for projects in the community.
- **Capitalization Caps Model Capacity?**: A member inquired about research on whether models perform better when prompted with proper **capitalization/grammar** versus all lowercase, and pointed to [three Arxiv papers](https://arxiv.org/abs/2310.11324), ([2411.10541v1](https://arxiv.org/abs/2411.10541v1)), ([2508.11383v1](https://arxiv.org/abs/2508.11383v1)) but noted they focus on prompt format rather than minor details like capitalization.
   - A member expressed the assumption that proper capitalization/grammar improves model performance and suggested testing this using **vLLM's benchmarking tools**.
- **Global CoT Analysis Attempts Uncovering Patterns**: A member shared a [LessWrong post](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj/global-cot-analysis-initial-attempts-to-uncover-patterns-1) about **global chain of thought analysis** and initial attempts to uncover patterns.
   - The analysis seeks to understand how models arrive at their conclusions by examining the reasoning steps they take, potentially revealing insights into their decision-making processes.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Arch Advocates Acclaim Always-Updated Arch**: Members debated the merits of **Arch** over **Ubuntu** and **Debian**, highlighting **Arch's** frequent package updates akin to **macOS** and **Windows**.
   - A user suggested **Garuda KDE** (Mokka and Dragonized) as a user friendly starting point with ready-to-use functionality.
- **PR Pipeline Plunge: Testing Transpires**: A discussion clarified the meaning of the `imported internally` label on a PR, indicating the PR has been cloned into an internal repo for final testing and integration.
   - The `imported internally` tag signals that the PR is in its final stages before merging, and it will be tagged with `merged-internally` upon completion.
- **.NET Nightmare: Legacy Lament**: A member bemoaned being assigned to a legacy **.NET 4.5.2** project from **2014**, which only functions on **Windows** and lacks documentation.
   - Another member related a similar experience with a standalone **C#** project plagued by problems and missing documentation, likening it to *finding a hotspring and water in a desert*.
- **Mojo Mulls Graphics Shaders with SPIR-V**: The possibility of incorporating graphics shaders into **Mojo**, particularly with a **SPIR-V backend** to facilitate *compute shaders*, is under consideration.
   - A member cautioned that the compiler build would be a *non-trivial* task once it is **open source**.
- **Shaders Showdown vs Matrix Manipulation**: A query arose concerning the distinction between **shaders** and conventional **matrix operations**, notably considering recent **CUDA** developments.
   - In response, one member linked to [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api) and another member linked to [Death to Shading Languages](https://xol.io/blah/death-to-shading-languages/) to help clarify the differences.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Minimax M2.1 Beats Kimi K2 in Claude**: A user reported that **Minimax m2.1** running in **Claude code** outperforms **Kimi K2** in code quality, planning, and API speed, noting they pay $40/month for **Kimi v2**.
   - They found **Kimi's API** slow and the model inferior, hoping for a newer release.
- **Debate on Kimi CLI Defaulting to K2 Turbo**: A user questioned why the default **Kimi CLI app** doesn't default to **K2 Turbo** with a proper subscription.
   - Another member suggested **Kimi K2 Turbo** should have around **73 tps**, compared to **MiniMax m2.1's 38 tps** and **Z.Ai's GLM-4.7's 41 tps**, though the latter has poor uptime.
- **New Slide Feature Uses K2 Model with Vision?**: A member inquired whether the new slide feature uses a newer **K2 model with Vision**.
   - Image analysis suggests it searches images for reference, implying some vision capability.
- **Questions on Kimi's Model Deprecation**: A member asked if **Kimi models** are discontinued every **12-14 months** like **Google's Gemini models** and if the same problem would arise switching to **Kimi K2**.
   - Older models are available on the [Moonshot API platform](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1), and a year-old model is still usable on [kimi.com](https://kimi.com).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Developer Seeks Super Cool Gig**: A member is actively searching for a **super cool project** where they can apply their **developer skills**.
   - Interested parties are encouraged to reach out for collaboration.
- **Discord Mod Applications Paused**: A member inquired about joining the moderation team, but another member clarified that the position is currently unavailable.
   - No specific reasons were provided for the hiring freeze.
- **AI Engineer Needed for Usage Tracking**: An active project requires an **AI engineer** to enhance **usage tracking** and develop a more robust **billing/credit system**.
   - This implies a need for expertise in both AI and financial systems.
- **Payment Problems Plague Platform**: A member reported facing persistent **payment issues** when attempting to add credit to their account, including problems with upgrading membership, using Link, and paying with credit card or Alipay.
   - They have yet to receive a response from the helpdesk or email support, suggesting potential delays in customer service.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Summit Lacks Livestream for Remote Registrants**: A member inquired about a **live stream** from the NY summit, expressing a desire to participate remotely.
   - They would have loved to register, but they cannot be there in person.
- **Stateless Servers Spark Scalability Savings**: A member proposed a [signature method](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2091) to balance schema freezing and dynamic server features, aiming to allow **stateless MCP servers** to handle multiple active conversations more efficiently.
   - They noted that their current setup in Goose, which spins up a new set of MCP servers for each conversation, is becoming increasingly expensive as the number of concurrent conversations rises.
- **Dynamic Toolsets Tackle Transports**: One member pointed to [issue #1442](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1442) and GitHub MCP's dynamic toolsets as examples of how servers can handle state on **STDIO**, potentially unifying behavior for both remote and STDIO setups.
   - The member admitted it is difficult to maintain a truly stateless **STDIO server**, given their current SDK architecture that builds a new "server" on every request, customizing registered tools based on user/flags.
- **Persistent Sessions Save State on Server Starts**: The topic of **persistent sessions** was raised as a means to retain session features across agent and MCP server restarts.
   - Another member mentioned using their own session middleware outside the Go SDK for horizontal scaling, suggesting that the ability to store and retrieve **session data** across restarts would be beneficial.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **1x1 Convolution Edges out SVD/PCA**: A member suggested using a **1x1 Convolution** instead of **SVD/PCA** for feature extraction, arguing that **SVD/PCA** extracts features with the highest variance (*loudest*), potentially capturing generic syntax noise rather than specific *intent* signal, via [this tweet](https://fxtwitter.com/i/status/2011094378396467316).
   - They believe that **1x1 Conv** would allow the model to learn precisely which heads matter for the loss function via backprop and would be lighter for inference.
- **"Quanta" Theory Draws Debate**: Members discussed the *quanta* theory which states that networks must learn a variety of modules, each implementing a different algorithm or retrieving a different piece of knowledge.
   - One member expressed skepticism, suggesting that many mechanisms could be entangled or too generic to be designated a specific use, potentially leading to an oversimplified mechanistic explanation of neural networks.
- **AI Assisted Coding Squaring off with Vibe Coding**: A member contrasted **AI assisted coding** tools (cursor/windsurf/antigravity) with what they termed **vibe coding** tools (devin/tembo/jules).
   - No further details were provided.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Tooling May Outperform Native LLM Tools**: A member referenced the [DSPy documentation](https://dspy.ai/learn/programming/tools/#using-native-tool-calling) and asked about benchmarks comparing **native tool calling** versus **custom tool calling**, particularly if **DSPy tools** always outperform.
   - Another member responded that this depends on the specific language model (LM) being used, implying performance is not universally better for **DSPy tools**.
- **Native and DSPy Tools Need Benchmarking**: A member emphasized that performance varies across language models, even those from the same AI lab, so **benchmarking is essential** for specific use cases and model combinations.
   - Another member agreed, stating that performance can vary in either direction, and users should test with their **specific model and program** to measure and evaluate what works best.
- **Native Tool Calling Can Be Lower Quality**: Members discussed that it is possible that **native tool calling** may sometimes produce lower quality results compared to other methods.
   - The statement in the documentation makes a weak claim, but this is expected to happen for some models.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Users Want Autoload**: Users are requesting a feature in **aider** to automatically add files without prompting, aiming for a less interactive experience.
   - The specific request involves configuring **aider** to bypass the add file prompt, indicating a desire for a streamlined workflow.
- **Aider Setup Troubleshoot**: A user reported setup difficulties with **aider** after installation via command prompt, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1133060505792159755/1461275448574218373/image.png?ex=696a9f10&is=69694d90&hm=19ccaef4fb45cd4288b6307abb3eca0a6819f27eb6253f0820357b2219006a4d).
   - The user sought guidance on next steps without providing further context.
- **Using CI Logs to Fix Aider**: A user asked about the best way to use **CI logs** with **aider** to fix failed tests, excluding the log file from Git.
   - A potential solution suggested was the command `aider --read ci.log`.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Black PNGs Baffle Stable Diffusion Users**: A user reported encountering fully black PNGs while running `examples/stable_diffusion.py` with the `--fakeweights` option, indicating a potential issue with **Stable Diffusion** in tinygrad.
   - The issue seems to be related to the **NULL device** and its interaction with kernel scheduling; debugging steps are ongoing.
- **NULL Device: Compute-less Kernels**: A user inquired about the purpose of the **NULL device** in tinygrad, questioning whether it performs any computations and how it aids in scheduling kernels.
   - Another member clarified that the **NULL device** does not perform any compute, but it is used to schedule kernels, which one user called *that's a cool feature*.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1461105640553316545)** (1085 messages🔥🔥🔥): 

> `Prodigy vs Barbie, Smack My Bitch Up vocals, GPT-5.2 chat leakage, AI for solo TTRPG campaigns, AI vs Human Made Music` 


- ****Prodigy beats Barbie: a Diss Track?**: One user said *prodigy is way better*, prompting another user to request a translation for someone wanting them to ruin a **Barbie** doll.
   - Arguments then broke out including statements such as *go crywank to chumba wumba on the barbie*.
- ****Shahin Badar belts 'Smack My Bitch Up'**: A user asked *who does the female vocals in Smack My Bitch Up*, and another user identified [**Shahin Badar**](https://en.wikipedia.org/wiki/Shahin_Badar) as the vocalist.
   - They then posted the [YouTube link](https://youtu.be/gJ4bW4KNffo?si=0SlbsHlcS3gTofuq) to the song.
- ****Pliny's Pithy Praise for Gemini Users****: A user posted a [Jim Carrey GIF](https://tenor.com/view/bruce-almighty-jim-carrey-beautiful-happy-smile-gif-4874848) with the caption *Pliny hasn't spoken in general in like 2 months*, and **Pliny** himself followed up with another [animated GIF](https://tenor.com/view/korone-flip-combo-breaker-killer-instinct-best-dog-inugami-korone-gif-25381954).
- ****DeepSeek Debuts for Jailbreaking Jollies****: A user recommended **DeepSeek** for its susceptibility to narrative attacks, noting it is the most common class of jailbreak techniques.
   - Someone added that it works due to allowing attacks such as *roleplay and persona*.
- ****VS Code: More than meets the Eye**: Users discussed customizing and forking VS Code, and a user shared their [VS Code setup](https://fixupx.com/davepl1968/status/2011868005312184485), which included Copilot, Codex, and his files in the center.
   - Another user mentioned making sure the screen is black, the font is easy on the eyes, and Copilot is turned OFF.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1461091240681537638)** (51 messages🔥): 

> `Jailbreaking Deepseek, Gemini, Grok, Jailbreaking Llama 3.2, Jailbreaking Opus 4.5, Prompt Injection Learning, GPT 5.2 Free-Tier Chat Leaks` 


- **Thinking: the Achilles Heel of LLMs?**: A member advised that, when jailbreaking, *turning off thinking for the model* lets the prompt hit and can offer success that reasoning might spit back.
   - It seems sometimes the models try to be TOO smart!
- **The Elusive Jailbreak for Gemini Pro**: Multiple members inquired about **jailbreaks for Gemini Pro**, with one humorously asking, *Where’s the jailbreak for banana pro Gemini fools.*
   - The demand for ways to bypass Gemini's restrictions remains high within the community.
- **Prompt Injection 101 with Anon3369489**: After a member asked how to learn **prompt injection**, another member offered to teach them, leading to a discussion about connecting on Discord.
   - The exchange highlighted the community's willingness to share knowledge and guide newcomers in the art of jailbreaking.
- **GPT 5.2 Free-Tier Churns Memories?**: A user reported that with **GPT 5.2** (free-tier), enabling memory in settings can lead to chat leaks from other sessions, and posted [an image](https://cdn.discordapp.com/attachments/1228043845967544380/1461404780831445237/image.png?ex=696b1783&is=6969c603&hm=91a356b1b007e9bb6123ede9a79414a836c03014291506ae32be52e3082e4eec).
   - The user was unsure if it's a memory issue or if deleting other chat sessions would resolve this.
- **Gandalf Game Guru Required**: A user who just started the **Gandalf game** asked for Italian speakers, but then asked for assistance with **level 8**, after getting discouraged after a few hours.
   - Another user offered to help, requesting to see the first user's work from level 7 and level 8 in DMs to avoid spoilers, and a third member added that *the bump in difficulty is huge* between levels.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1461096614239273093)** (35 messages🔥): 

> `Grok Image Moderation, Llama 3.2 Jailbreak, Web Pentesting Resources, AI Pentesting Resources, GPT 5.2 Jailbreak` 


- ****Grok's Goriness**: Musk Orders Image Moderation Meltdown**: In response to [Elon Musk's request](https://x.com/elonmusk/status/2011527119097249996), users are attempting to break **Grok's image moderation** by creating a pornographic Twitter thread.
   - One user jokingly stated it's *soon to be the most pornographic twitter thread in history.*
- ****Llama 3.2 Lockdown**: Jailbreak Attempts Stall**: Users are actively seeking a jailbreak for the latest **Llama 3.2**, with one user reporting that a previously effective jailbreak prompt (used in **Llama 3.1**) [fails on the new version](https://chepenikconor.medium.com/day-855-9ae6f88b192c).
   - Attempts to elicit harmful responses like *how to make meth* or *make anorexic wife lose 100lbs* are reportedly meeting resistance, indicating enhanced safety measures.
- ****Arcanum Ascends**: Free AI Pentesting Fortress Found**: A member shared [Arcanum's AI security resource hub](https://arcanum-sec.github.io/ai-sec-resources/?utm_source=executiveoffense.beehiiv.com&utm_medium=referral&utm_campaign=executive-offense-the-arcanum-ai-security-resource-hub), which offers free resources and a structured workflow for **AI pentesting**.
   - This GitHub resource is being shared for routine team investigation.
- ****GPT 5.2 Fort Knox**: TheDonutAI Remains Unbreakable**: A user shared a link to [TheDonutAI's GPT 5.2 dashboard](https://thedonutai.com/dashboard), noting that *no can jail break this* model on drug related topics.
   - Another user jokingly said: *if u can break ur the GOD dude*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1461088695619686570)** (478 messages🔥🔥🔥): 

> `VRAM Importance, Tuning Gemma2 2b, Effective Batch Size, GRPO Training Hyperparameters, New long context RL release` 


- **VRAM Boosts Model Speed**: Members noted that more **VRAM** generally leads to faster model execution, and the `-ot` flag in Unsloth docs can help manage tensor placement for optimal performance.
   - One member suggested that `-ot` is usually faster than using `n-cpu-moe`.
- **TPU Tunix Clashes with Gemma2 2b**: A member reported beating **Unsloth** with **4 RTX Pro Q-Max** by using the free **Kaggle TPU v5e-8** for training **Gemma2 2b** with **FSDP**.
   - They cautioned that *flash attention* and *cut cross entropy* aren't implemented and *gradient checkpointing* is dubiously implemented.
- **Effective Batch Size Optimization Explored**: Discussion revolved around achieving an *effective batch size* of **32** through gradient accumulation without needing a datacenter.
   - It was emphasized that batch size should be a performance/hyperparameter optimization rather than a GPU size consideration.
- **GRPO Training Requires Parameter Tuning**: A member sought advice on tuning hyperparameters mid-run to accelerate model convergence during **GRPO** training, sharing that one epoch consists of **3000 steps**.
   - Another user suggested setting `importance_sampling_level="sequence"` in **GRPOConfig** as a potentially stabilizing factor.
- **Unsloth's RL Release Extends Context Window**: A [new long context RL release](https://x.com/UnslothAI/status/2011827592886960131) boasts a **7 million token context window**, marking a **7x** increase over the previous release.
   - Members marveled at the exponential memory savings and the implications for tasks like conversing with one's 'wifefu' indefinitely.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1461092301739655465)** (805 messages🔥🔥🔥): 

> `Versatile Embedding Models, RL Insightful Session, Audio Tokenizer Codecs, Scaling Laws Architecture, Qwen3 VL Architecture Bug` 


- **Embedding Model Size Astonishes**: Members expressed amazement at the **versatility of embedding models** being achieved within a compact **308M parameter space**.
   - A meme image was shared to display **shocked LLM developers** seeing perfect embeddings.
- **RL Session Sparks Insights, Electricity Burns**: Following an insightful session about **Reinforcement Learning**, one member thanked another, who jokingly responded with *“You’re just burning electricity now”*.
   - A member shared a [YouTube link](https://www.youtube.com/live/jMSCJZAEYR8?si=738_bf4US5AlRCsU) to the event.
- **Qwen3 VL Bug Squashed**: Users reported a bug where the **Qwen3 VL architecture** was not detected as a vision model by Unsloth, raising a `ValueError` about mismatched models and datasets when attempting vision finetuning.
   - The issue was resolved by upgrading `transformers` and verifying the environment setup, with a reminder to use the [correct notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb) for Qwen3 models.
- **Anthropic's Python Investment**: Members discussed [Anthropic's investment in Python](https://pyfound.blogspot.com/2025/12/anthropic-invests-in-python.html) and their business approach, praising their focus on providing effective models without over-commercialization.
   - One participant noted that *Claude's always been pretty cracked compared to anything else for development related stuff*, but it is costly.
- **YouTube Flooded With AI Slop**: Members discussed the increasing presence of **AI-generated content on YouTube**, with one sharing [a report](https://www.theguardian.com/technology/2025/dec/27/more-than-20-of-videos-shown-to-new-y) indicating that over 20% of videos shown to new users are AI-generated.
   - Some members reported seeing **AI-generated media** immediately pushed to new accounts, seniors, and very young kids.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1461106991832236032)** (51 messages🔥): 

> `Hyperparameter Tuning Mid-Run, GGUF and vLLM Compatibility, Running Unsloth Models with Ollama, MedGemma Quantization, RL Model Training Graphs` 


- **Tweaking Hyperparameters Mid-Training**: A member inquired about the feasibility of changing hyperparameters mid-run during **GRPO training**, particularly the learning rate, and another member confirmed it's possible if checkpoints are saved.
   - They also noted the challenges of increasing **learning rate** due to potential convergence issues and local minima, especially when using a large group size on **H200**.
- **GGUF's Growing Pains with vLLM**: A member encountered a `ValueError` when trying to run a **GPT-OSS-20B-Q4_K_M.gguf** file using **vLLM**, which was attributed to **GGUF's** experimental status in **vLLM**, according to [vLLM documentation](https://docs.vllm.ai/en/stable/features/quantization/gguf/).
   - They suggested using **vLLM's llmcompressor** or **Intel autoround** for post-training quantization instead.
- **Ollama helps label Bank Transactions**: A member sought advice on running a small **Unsloth** model with **Ollama** to automatically label bank transactions using [actual-ai](https://github.com/sakowicz/actual-ai), providing details on their **Truenas** setup with a **Ryzen 5900x**, **128GB RAM**, and **Nvidia Quadro T400 GPU**.
   - They also shared code snippets for loading a pipeline using **ZImageTransformer2DModel** and **ZImagePipeline**, asking if a similar approach could be used with the text encoder.
- **Model Merging for RL Convergence**: A member sought advice on interpreting **TensorBoard graphs** for an **RL model**, particularly regarding the use of smoothing, since they have been training for 3 days on a 3000-sample dataset and only went through 2 epochs.
   - Another member advised to "*trust both!*" smoothed and unsmoothed graphs and consider merging the current checkpoint to assess progress, even though it's not performing as well as their **SFT classifier**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1461504313447813121)** (3 messages): 

> `Unsloth showcase` 


- **Unsloth Showcase Declared**: The channel is declared as a showcase for **Unsloth**-related material, including **Unsloth**-trained models, contributions, and open datasets on HF/Github.
   - General chatting to be continued in other channels.
- **Clarification on Showcase Purpose**: The showcase is specifically for **Unsloth**-related content such as models, contributions, and open datasets.
   - Irrelevant discussions should be moved to designated chat channels.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1461113160328020080)** (7 messages): 

> `Recursive Language Models, Agentic Systems Context Management, Knowledge Distillation Vocabulary Truncation, Softmax Calculation Numerical Instability` 


- **Discussing Recursive Language Model Expansions**: A member has been trying to expand on the idea of **recursive language models** in their agent harness and thinks that *agentic systems* should be able to not only manage their context but also change their code, tools, etc at runtime, to handle tasks that are given to them.
- **Vocabulary Truncation Aids Knowledge Distillation**: A member found that truncating vocabulary during training helped handle memory consumption issues during recent work on **knowledge distillation**.
- **Numerical Instability with Zero Temperature**: A member pointed out that *not using temp 1 can introduce numerical instability* into **softmax calculation** since temp is a scalar in the equation.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1461090190520221900)** (511 messages🔥🔥🔥): 

> `Sonnet 4.5 Background Agent, Composer-2 as Reasoning Model, GPT 5.2 Codex Experiences, Cursor's Sub-Agents, Token Usage Minimization` 


- **Sonnet 4.5 Launches Background Agent**: Users confirmed that choosing **Sonnet 4.5** is now an option for launching a background agent, though this functionality may not yet be available in the web version.
   - A dev on X mentioned they are working on drastically improving the background agent in the next couple of days/weeks.
- **Composer-2 Secretly Slides in as Reasoning Model**: Users noticed that adding **composer-2** manually marks it as a reasoning model and speculated on why unregistered models default to this classification.
   - A user mentioned a recent **Cursor** tweet hinting at the imminent arrival of **composer-2**.
- **GPT 5.2 Codex Falls Flat**: One user reported being unimpressed with **GPT 5.2 Codex**, stating it failed to follow instructions when creating a plan.
   - Another user pointed out a scaling-agents post from cursor.com which suggests **GPT-5.2** models are superior at extended autonomous work, instruction following, and precision, using subagents effectively.
- **Sub-Agents Subvert Expectations**: Users explored the functionality of **sub-agents**, noting the existence of two built-in subagents: **Explore** and **generalPurpose**.
   - It was noted that only specific models can call subagents, and there are issues with setting model reliably for each sub agent.
- **Token Usage Gets Targeted**: Users discussed strategies to minimize token usage, especially when exploring code, and asked for suggestions on tools for targeted and comprehensive code exploration.
   - One user suggested trying [Nia mcp](https://nia.mcp), with another suggesting a command to review code changes and provide relevant prompts.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1461089247699402949)** (420 messages🔥🔥🔥): 

> `Comet Browser, Grok Models, Perplexity Pro, ActivityPub, Image Generation` 


- **Comet Color Change Calamity**: A user noted that setting a browser color in **Comet** no longer changes Perplexity's color scheme, as [the *theme="yellow"* HTTP header](https://discord.com/channels/1047197230748151888/1047649527299055688/1461239027771510878) seems to have been removed.
   - They're investigating if it was a bug or intentional.
- **Grok's Code Breaks into Top Five**: **Grok Code** made it into the top five, according to [this post](https://x.com/i/status/2011823610386600009)!
   - The chart measures how many tokens users have spent.
- **Perplexity Pro Promo Problems Persist**: Users are still facing issues with their **Airtel Pro** subscriptions not activating, even after contacting support and following all the steps outlined in [Perplexity's help article](https://www.perplexity.ai/help-center/en/articles/11842322-perplexity-pro-airtel-promo).
   - Some users report receiving canned responses from AI agent Sam, with no resolution.
- **New Grok Model Uncovered!**: A user spotted a new **Grok** model being tested via codenames, which is a fast variant.
   - They shared screenshots from the anonymous voting on Discord, showing the models being referred to as assistant a and assistant b.
- **Image Generation Faces Regional Roadblocks**: Some users are reporting that **image generation** is not working and may even be intentional in some regions.
   - This appears to impact some European countries.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1461320890204819564)** (2 messages): 

> `perplexity-cli, free data challenges` 


- ****Perplexity-cli** is Shared!**: A user shared their new [perplexity-cli tool](https://github.com/noQuli/perplexity-cli) for interacting with Perplexity AI from the command line.
   - No further details were provided about the tool's functionality or features, but interested users can check out the **GitHub repository**.
- ****FGV** Offers Free Data Challenge Prototypes**: A professor from **FGV** (**Escola de Matemática Aplicada** in Brazil) announced [free data challenges](https://emap.fgv.br/en) where students build initial prototypes in five days.
   - The professor invited users to bring stalled data challenges and provided a [survey link](https://survey.fgv.br/jfe/form/SV_cvAuObq3mG4NTtY) for submissions.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1461188244635914261)** (3 messages): 

> `Sonar Reasoning Pro, Perplexity API account, API Payment` 


- **Sonar Reasoning Pro model loses its train of thought**: A user reported the disappearance of **<think> blocks** from the response content when using the **sonar-reasoning-pro model** via API.
   - It is unclear whether **Perplexity** intentionally removed these tags.
- **Perplexity API performance drop spurs user to seek refund**: A user claimed degraded performance after depositing money into their **Perplexity API account**.
   - They are looking for information on how to withdraw their money, reporting unsuccessful attempts to contact **api@perplexity.ai**.
- **Perplexity directs payment inquiries through email**: A member noted that technical and payment issues are handled exclusively through **email**.
   - They clarified that the **Discord** channel is for bug reports and community interaction, not for support inquiries.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1461088740888940666)** (412 messages🔥🔥🔥): 

> `Video Arena Battle Mode, Veo 3.1 Fast Model, Falcon-H1R-7B-GGUF Model, Siren Video Model Details, LMArena VSCode Extension` 


- **Video Arena Locked in Battle Mode**: The Video Arena is designed for **Battle mode only**, with no plans to offer direct chat or side-by-side comparisons, but early experimenters had those features briefly.
   - Users can find the **Veo** model in the battle section and can check the Discord search bar to find generations, but there are limits on video generation (3 per 24 hours on the site, 5 per 24 hours on Discord).
- **Users Request More Veo 3.1 Fast**: Users want "unlimited" **Veo 3.1 Fast**, but a moderator noted it's unlikely due to current rate limits (3 per 24hr on the site and 5 per 24hr on Discord).
   - When a user asked about testing it on an external site, the moderator suggested, *"Why don't you?"*
- **Falcon-H1R-7B-GGUF Model is Incredible**: A user shared an incredible model, [Falcon-H1R-7B-GGUF](https://huggingface.co/unsloth/Falcon-H1R-7B-GGUF), with another user asking *"tell us more!"*.
   - They also shared a paper link [Transformer-Based Generative Adversarial Network for Image Super-Resolution](https://huggingface.co/papers/2601.02346).
- **Siren Video Model Details Remain Secret**: A user inquired about the **Siren video model** on Video Arena and was told that it's likely a codenamed model, with early access to frontier models still in development, and from their [FAQ](https://lmarena.ai/faq) real-world feedback, transparency, and user voice can directly influence which models move forward.
   - Staff will not confirm nor deny what the codenamed models are and another user speculated it's **Wan 2.5** because **Wan 2.5** generates at 30 fps.
- **Hawk Ultra Model Emerges as Opus Killer**: The Hawk Ultra model is drawing comparisons to **Gemini 3 Pro**, with claims of generating over **17,000 lines of code** in a single prompt, outputs are better than both Opus and Gemini and one user said *"bro is on a big mission"*.
   - Movement Labs appears to be behind this, as shown on this [X post](https://x.com/movementlabsAI/status/2011964766533632380?s=20), and it is *"got me so gassed"* with open source coming soon.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1461101492177211513)** (3 messages): 

> `Video Arena Updates, Code Arena Updates, Image Arena Updates, Text Arena Leaderboard Updates, ERNIE-5.0-0110 Performance` 


- **Video Arena Adds New Veo Models**: The Video Arena has been updated with new models including **veo-3.1-audio-4k**, **veo-3.1-audio-1080p**, **veo-3.1-fast-audio-4k**, and **veo-3.1-fast-audio-1080p**, available for testing in the [Video Arena channel](https://lmarena.ai/c/new?chat-modality=video).
- **Code and Image Arenas Get New Models**: The [Code Arena](https://lmarena.ai/c/new?chat-modality=code) welcomes **gpt-5.2-codex**, while the [Image Arena](https://lmarena.ai/c/new?chat-modality=image) introduces **glm-image**.
- **ERNIE-5.0-0110 Climbs Text Arena Leaderboard**: `ERNIE-5.0-0110` secures the **#8** spot on the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) with a score of **1460**, also ranking **#12** in Arena Expert.
   - This model, the only one from a Chinese lab in the Top 10, shows exceptional performance in the **Math** and various **occupational categories**, detailed in the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1461123664757133323)** (284 messages🔥🔥): 

> `LM Studio OpenAI API, GPT-OSS-20B, Coding LLMs, GPU recommendations for LLMs, LM Studio token stats` 


- ****LM Studio's OpenAI API sparks IDE integration idea****: Members discussed running models locally with LM Studio and connecting its OpenAI-compatible API to a local IDE for running agents and scripts, saving on tokens.
   - One member confirmed that LM Studio can indeed start an **OpenAI-compatible API** for this purpose.
- ****GPT-OSS-20B is surprisingly speedy****: Members debated why the [GPT-OSS-20B model](https://huggingface.co/models?search=GPT-OSS-20B) feels faster than many 8B or 12B models, clarifying that it's a **Mixture of Experts (MoE)** model with only a subset (**3.6B**) of parameters active per token.
   - Despite not using all weights, the model performs well in tasks like **math, physics, and quantum mechanics**, maintaining context over **34k tokens** even on a **6700XT** GPU.
- ****Finding the Finest Coding LLM Frenzy****: Users sought recommendations for the "best" local LLM for coding, with mentions of **DeepSeek R1, Qwen3, and Devstral**, but it was noted that [Claude](https://claude.ai/) remains the top performer overall.
   - Given hardware limitations, members suggested focusing on fitting the model into VRAM over raw speed, recommending a **20GB RTX 3080** over an **11GB 3080 Ti** due to the importance of VRAM for LLMs.
- ****LM Studio's Token Tracking Troubleshoot****: A user inquired about obtaining token-count and inference-speed information when using LM Studio as an API backend.
   - Suggestions included checking the API response for stats, using the **/responses** endpoint instead of **/chat/completions**, and employing a [community-created tool](https://openwebui.com/posts/token_usage_display_filter_9d6df2c3) to display token usage.
- ****LiquidAI LFM2.5-1.2B tool use issues****: A user encountered issues with tool use in the **LFM2.5-1.2B** model, receiving the output `<|tool_call_start|>[execute_command(command="date")]<|tool_call_end|>` when asking for the time.
   - Troubleshooting steps involved verifying tool access, trying the instruct version of the model, ensuring a proper system prompt, and referring to the [LiquidAI documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use) for guidance.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1461532971843584215)** (2 messages): 

> `MX150 laptop fine-tuning, CUDA 12.6 requirement` 


- **MX150 Finetunes 350M Model**: A user successfully ran a full fine-tune on a **350M model** using an **MX150 laptop** with **2GB VRAM**.
   - The process surprisingly required **CUDA 12.6**.
- **CUDA 12.6 is the magic number**: The user was surprised that **CUDA 12.6** was a strict requirement for the fine-tuning process.
   - This suggests that certain configurations might unexpectedly demand specific CUDA versions for compatibility.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1461087862781903130)** (235 messages🔥🔥): 

> `Cheap inference, Sparse MoE on CPU, GPT-5 image pricing, BYOK AWS Key issues, Smorty Character Card` 


- **Smorty is not an LLM**: Members confirmed that **Smorty** is not an LLM, but a real person, based on their unique writing style.
   - One member joked, *"If I read one more message of yours I might turn into a schizophrenic"*.
- **GPT-5 Image Generation Costs Spike**: A member reported that the cost of image generation using the **openai/gpt-5-image-mini** model suddenly jumped from **$0.01 to $0.04**.
   - The reason for this price increase was not immediately clear.
- **AWS Key BYOK function issues**: A member needed help with their **BYOK function**, reporting issues using their **AWS key** with **OpenRouter** for different models.
   - They encountered issues with **SillyTavern**, **Amazon Bedrock**, and **Anthropic**, receiving an *"Unauthorized or not cookie auth credentials found"* error.
- **OpenCode is the IDE**: Members discussed coding harnesses, and decided [**OpenCode**](https://github.com/OpenRouterTeam/awesome-openrouter?tab=readme-ov-file#aventura) is the best, with plugins like **oh-my-open code**.
   - One noted it *"makes claude code feel like you're using some old school terminal app"*.
- **Smorty Writes a Lemmy Post**: **Smorty** is writing about **skill.md** on [Lemmy](https://lem.lemmy.blahaj.zone/u/Smorty?page=1&sort=New&view=Posts), a FOSS and fediverse alternative to Reddit.
   - Smorty noted that the community is *"very against machine learning stuff"*.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1461091089095069972)** (14 messages🔥): 

> `Cerebras Partnership, Groq deal, Unslop AI Prose, Acceptance Tests` 


- **Cerebras Joins OpenAI in 2028**: OpenAI announced a [partnership with Cerebras](https://openai.com/index/cerebras-partnership/) scheduled for **2028**, which came as a surprise to some members.
   - Speculation arose that this move might be a response to the **Groq** deal, given Cerebras's long-standing presence and support for large models like **120B**.
- **Reddit Unslops AI Prose**: A member shared a [Reddit link](https://old.reddit.com/r/LocalLLaMA/comments/1qd88v2/i_trained_a_model_to_unslop_ai_prose/) about training a model to "unslop" AI prose.
   - It was followed by a link to a [Fixup Status](https://fixupx.com/openaidevs/status/2011862984595795974) and a question of whether the OpenAI team was involved in it.
- **Fixup Acceptance Tests is Born**: A member confirmed that they were early contributors to the project, particularly the **Acceptance Tests**.
   - They mentioned having been in discussions with **OpenAI** about this for months, but a question was raised about why the spec isn’t going to **IETF**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1461249189290967063)** (183 messages🔥🔥): 

> `Byte-level LLMs and Diffusion, Flux 2 9B Model, Nervous System for LLMs, Gemma Model` 


- **Bytes and Diffusion get the nod**: A member combined **byte level LLMs** and **diffusion** models to help byte models work with different file types, noting that if diffusion messes up one byte, it's easier to correct it, demoed in a [screen recording](https://cdn.discordapp.com/attachments/1149866623109439599/1461342380392185928/Screencast_from_13-01-26_152930.webm?ex=696add65&is=69698be5&hm=5bcb4ae2ce4e375aac96cd552f00b7d4077391dbad48fa2b2745608cc1555828&).
- **Flux 2 takes VRAM**: The **Flux 2 9B model** uses a **Qwen 3 8B** text encoder, needing **35GB of VRAM** to get all the weights in VRAM for serving, but that halves with no concurrent users.
   - One member asked whether there's a *llama.cpp/lmstudio/vllm/sglang* for diffusions and another suggested **ComfyUI**.
- **LLM gets nervous**: A member is working on a *native transformer architecture extension* that provides a **nervous system** to LLMs (including short/mid/long term memory) at **<1% compute cost**.
   - They claim it *scales linearly 1-2% compared to the model size*, and performs **95%** the same as the model on the BEIR's, but has not provided verifiable benchmarks.
- **Google Announces Gemma Model**: Members are discussing that **Google** is releasing **Gemma** models.
   - One member reacted *Gemma, meta was never more meta!*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

gettygermany: i would be happy if i could USE them haha
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

real.azure: https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1461131165795614954)** (14 messages🔥): 

> `Chrome Trace Visualizer issues with PyTorch Profiler, Perfetto UI as an alternative to Chrome Trace UI, ncompass.tech dev tool for trace viewing and analysis` 


- **Chrome Trace Visualizer Crashing at 600MB**: Members reported that the **Chrome Trace Visualizer** for **PyTorch profiler** crashes around **600MB**, despite PyTorch docs suggesting issues only above 1GB.
   - One member found that the loading prompts complete quickly without errors, but the visualizer remains empty, and another also ran into the same issue.
- **Perfetto UI Rescues Trace Visualization**: A member suggested using **Perfetto UI** as an alternative when **Chrome Trace UI** fails, mentioning that [Perfetto](https://perfetto.dev/) has worked for them in the past.
   - However, they noted that **Perfetto** sometimes lacks **Cutlass kernels**.
- **ncompass.tech Builds Trace Chunking Tool**: A member is building a dev tool ([ncompass.tech](https://docs.ncompass.tech)) for trace viewing and analysis and is considering fixing the large trace issue by chunking the trace.
   - They experienced issues opening a **700MB** file in their **Perfetto viewer** within **VSCode** and are exploring solutions, offering to share updates on their chunking approach.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1461197803991597232)** (37 messages🔥): 

> `A/B tiles in K-major layout, WGMMA shared memory layout, BLOCK_Mx16B TMA load, LBO and SBO for TMA, TMA tensor copy` 


- **Shared Memory Layout Shenanigans for WGMMA**: Discussions revolved around shared memory layout requirements for A/B tiles in K-major layout with NO swizzle for WGMMA, referencing [Colfax's tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/).
   - The question was whether each **8x16b core matrix** is one contiguous chunk of GMEM, with some concluding that a `BLOCK_Mx16B` 2D TMA load could be issued for each slice of the A tile while iterating horizontally.
- **LBO and SBO TMA Configuration Examined**: Members debated the correct LBO and SBO settings for TMA, with one member suggesting LBO could be interpreted as either **8x16b=128b** (one contiguous core matrix) or the whole **BLOCK_M*16b**.
   - It was determined that when doing a TMA slice of `BLOCK_Mx16B`, **LBO is BLOCK_Mx16B**. One member confirmed that, after struggling, they got it working with 128b swizzle, but differently than a published blog.
- **TMA Tensor Copy Performance Tweaks**: A user was reminded that TMA tensor copy supports **5 dimensions**, and larger TMA ops are more efficient; however, it was found that for `BLOCK_Mx16B`, multiple 2D TMAs were faster than a single 3D TMA, but for 128B+swizzling, 3D TMA was faster.
   - Smem descriptor encoding was achieved using **LBO=16** (bytes between core matrices along the K-dim) and **SBO=`8 rows * BLOCK_K * 2 bytes/elem`** (bytes between core matrices along the M/N dim), as shown in [this code](https://github.com/danielvegamyhre/gemm/blob/9fe95aa61ee7ebca4ded8b5029494b0d58e0d2e2/pipeline_tma_wgmma/pipeline_tma_wgmma.cu#L109-L118).
- **Swizzling Impacts LBO**: It was noted that LBO is ignored when swizzling is used, but SBO is still used.
   - The reasoning is that swizzling alters the stride between start addresses of core matrices within a row, but not between rows.
- **Torch's synchronize() Still Tricky**: A quick quiz was posted regarding flaws in benchmark code using `torch.cuda.synchronize()`, with the main problem being the inclusion of host overhead / kernel dispatch latency due to the use of a host-side timer, rather than device-side measurement with cuda events or Triton's `do_bench`.
   - One user humorously suggested to *use a well established function and call it a day, instead of rolling ur own*, though it was conceded that `do_bench()` has its own problems as well.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1461177097740488726)** (1 messages): 

> `Ahead of Time Compilation` 


- **Ahead of Time Compilation Coming to Torch**: Members mentioned that the closest thing to ahead of time compilation is available via [this pytorch doc link](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html).
- **AOT Inductor Discussion**: The discussion centered around the use of `torch.compiler_aot_inductor` for ahead-of-time compilation in PyTorch, with a pointer to the [official documentation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html).


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1461470608339763202)** (1 messages): 

> `Information Gravity, Inference Stability, Hallucination Loops, Excitation Flux, Hysteresis Firewall` 


- **Information Gravity Tackles AI Instability**: A member is applying **Information Gravity** to solve **Inference Stability** and **Hallucination Loops**.
   - They have mapped the **Excitation Flux** of token selection, noting a shift from nominal logic to linear growth leading to a **Tsys singularity** at S > 45.
- **Hysteresis Firewall Stabilizes via Gamma Flush**: A **Hysteresis Firewall** at **1.0** enforces stability via a **2.2x gamma-eff flush**.
   - Substrate Modules & Full Logic are available on [GitHub](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

marksaroufim: https://github.com/daytonaio/daytona
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1461193765245948076)** (19 messages🔥): 

> `CUDA Fundamentals, Golomb-Rice, CUTLASS for CUDA, Data Compression using CUDA, NVidia GPU optimization` 


- **Tiny Block Size Troubles Trigger Thread Talk**: A member inquired about the downside of using a tiny block size of **32** in CUDA, pondering if low granularity could lead to higher occupancy, sparking a discussion about **NVidia GPU hardware optimization**.
   - Another member explained that a block size of 32 results in each block only getting one warp to work with and you want a lot of warps working the SM (*Streaming Multiprocessor*) at a time, and one warp per SM will make things slow.
- **CUDA Compression Compadres Commence Collaboration**: A Master’s student in Electrical Engineering starting a GPU-based data compression project using CUDA, with a focus on **Golomb-Rice** compression, sought recommendations for resources.
   - No recommendations were given in the channel.
- **Parallel Programming Perspectives Pondered**: A member studying parallel programming from the PMPP book shared a Reddit post suggesting that there isn't much pure CUDA work at scale in DL due to **CUTLASS** solving general issues like FMA, and optimizing networking being the main demand.
   - No opinion on this topic was given in the channel.
- **NVidia Numbskulls Navigate Nuances**: A member shared a link to the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications) clarifying that the maximum number of resident blocks per SM is smaller than the maximum number of resident warps, so **maximum occupancy** is unachievable with a block size of 32.
   - Smaller blocks also mean fewer threads can share data via shared memory, and **WGMMA/tcgen05** need multiples of 128 threads to work together.
- **Undergrad Advice Underscores Utilitarianism**: An undergrad was advised to choose what they find most interesting in computer science, specializing deeply, structuring work around open source contributions or novel technical results.
   - It was also suggested that being too specialized at the beginning when you're unskilled may make it hard to find a job.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1461258652588703840)** (5 messages): 

> `work-life balance, age demographics, ML networking` 


- **Work-Life Balance a delicate balance**: Members discussed the desire to avoid spending extra hours discussing work-related topics outside of work hours.
   - The consensus was that people generally prefer to have a clear separation between their professional and personal lives, and do not like to spend time talking about work outside of work hours.
- **Age Demographics shape socializing**: The discussion highlighted how age demographics influence the willingness to socialize around work-related topics.
   - It was noted that people with children may have less time and interest in such activities, while those without children might see it as a way to **network**, **make friends**, or **find business partners** interested in Machine Learning.
- **ML Networking benefits explored**: Some members suggested that meeting up to discuss ML could be a valuable networking opportunity.
   - The discussion was that these in-person connections could lead to friendships, business partnerships, or other beneficial collaborations in the field of Machine Learning.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1461196991458574522)** (9 messages🔥): 

> `global_load_dword vs buffer_load_dword on CDNA, Buffer Descriptor Advantages, VMEM issue latency` 


- **Global Load vs. Buffer Load: CDNA Faceoff**: A user inquired about the performance differences between `global_load_dword` and `buffer_load_dword` on **CDNA architecture** when loading from **HBM** to **REG**, noting inconsistent performance gains when substituting `global_load` with `buffer_load`.
   - Microbenchmarking showed little to no difference, leading to confusion, further investigation is needed.
- **Buffer Instructions: Scalar Registers for the Win?**: A member suggested that `buffer` instructions' main advantage is the use of a **buffer descriptor** stored in **scalar registers**, potentially reducing vector register usage and improving occupancy.
   - This can lead to fewer instructions as addressing might be handled by the buffer instructions, rather than **vector or scalar shifts and adds**.
- **Bounds Checking Boosts Buffer Load?**: Another member clarified that while both `global` and `buffer` loads support a **scalar base address** and a **vector index**, the main advantage of `buffer` loads is **built-in bounds checking**, which can save registers and control flow if done manually.
   - Without needing bounds checks, there's not necessarily an advantage, and performance variation might be due to different register allocation.
- **VMEM Latency: Mystery Deepens**: A user shared a screenshot and expressed surprise that the **issue latency** of **VMEM** instructions sometimes increases after previous **VMEM** instructions, even with the same buffer descriptor and low occupancy, using [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof3.html).
   - The user seeks insights into what factors can affect the **issue latency** of **VMEM** instructions.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1461399171667071171)** (4 messages): 

> `Benchmarking on Apple Silicon, M4 Pro vs M5, Cloud Computing Services` 


- **Apple M-Series Benchmarking Brainstorm**: Members are discussing ways to benchmark on newer **Apple Silicon**, with one user currently using an **M4 Pro** and wanting to test their repo on **M5**.
   - The user is interested in **M5** due to the *30% increase in memory bandwidth* and is looking for trusted cloud computing services or someone with the appropriate device to benchmark.
- **Begging for Benchmarks?**: A member volunteered to benchmark if the kit shows up, even on trial.
   - They also mentioned seeking a friend with the right device to run the benchmarks on.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1461474135548104870)** (6 messages): 

> `sm121a Kernel Development, DGX Spark Optimization, ThunderKittens vs Cutlass, vLLM Performance` 


- **Kernel Dev Dares DGX Spark**: A member has been developing a kernel for the **sm121a** (DGX Spark) for about a week.
   - The goal is to achieve the fastest possible inferences in **vLLM**, which currently lags behind *llama.cpp* and some specialized branches of *SGLang*.
- **ThunderKittens Considered over Cutlass?**: A member inquired whether they should use **ThunderKittens** instead of **Cutlass** for their kernel development.
   - They seek to optimize a kernel for **DGX Spark**, noting they haven't seen one publicly available, even though DGX is technically a Blackwell architecture.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1461317240744968294)** (16 messages🔥): 

> `Profiling feature feedback, Profiling overhead, SM frequency discrepancy, CUDA error during benchmarks, dual_gemm competition end date` 


- **Profiling Feature Flags and Kernel Coverage**: A member reported that the profiling zip file `profile_20260115_104909_run0.zip` contains a profile for only one kernel, and inquired about necessary flags to include all cases.
   - The support team is investigating this issue.
- **Profiling Adds Expected Overhead**: A member observed that profiling resulted in slower execution times compared to CLI benchmarks, and asked if this was expected.
   - An NVIDIA engineer confirmed that this **overhead is expected**, and cautioned against using profiles for measuring absolute kernel runtime.
- **SM Frequency Lowered by Profiler**: A member noted that the **ncu profiler showed an SM Frequency of 1.08 GHz**, while the competition specified testing under 1.5 GHz.
   - It was clarified that **ncu lowers clock speeds**, and these speeds aren't representative of production environments; thermal throttling not seen during competition analysis.
- **CUDA Error strikes when benchmarks aren't passing**: A member reported that tests pass but benchmarks raise a `CUDA error: CUBLAS_STATUS_INTERNAL_ERROR` when calling the reference kernel.
   - Another member suggested this is likely caused by an **out-of-bounds access** in the user's kernel, recommending `torch.cuda.synchronize()` for debugging.
- **Dual GEMM Deadline Approaches**: A member inquired about the exact end date of the **dual_gemm competition**, ending January 20, 2026.
   - No response was given to this query.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1461126663583567994)** (15 messages🔥): 

> `GPU Mode Hackathon Success, Niche Specialization, Interview Scheduling Strategies` 


- ****GPU Mode Hackathon** leads to Job!**: A member landed an *amazing job* by attending a **GPU Mode hackathon** at **Jane Street** in NYC, sharing this success story as a testament to the event's value.
   - They prepared for weeks, bringing resumes and a formal outfit, and committed to maximizing interactions from breakfast to the very late closing dinner, showcasing the importance of thorough preparation and engagement.
- **Cultivate a Niche for Interview Success**: One member recommends finding a more specific niche to stand out from other candidates, focusing on unique combinations of skills.
   - Examples provided include **Kernel Optimization + Reinforcement Learning**, **Reinforcement Learning + Fullstack Development**, and **Zero Knowledge Proofs + Deep Learning**, advising others to apply for roles specifically targeting these skill combinations.
- **Space Out Interviews for Mental Health**: A member advises spacing out interviews more to allow time for relaxation, especially when facing a high volume of rounds.
   - They admitted to scheduling interviews too closely together and don't think that's necessarily healthy if you have a lot of them to go through.


  

---


### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/)** (1 messages): 

marksaroufim: https://www.youtube.com/watch?v=hzpAox5x_6w
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1461097205254586599)** (102 messages🔥🔥): 

> `OpenAI Cerebras Partnership, Mistral 3 Release, AI Agent Data Access, Thinking Machines Leadership, Black Forest Labs FLUX.2` 


- **OpenAI and Cerebras Join Forces!**: OpenAI and Cerebras announced a strategic partnership, with more details available on [OpenAI's website](https://openai.com/).
   - This collaboration marks a significant move in AI infrastructure, with community members expressing excitement over the potential impact.
- **Ministral 3 Papers Dropping!**: A new paper for Mistral's latest model, **Ministral 3**, was announced, sparking discussion on its capabilities and performance, originally tweeted by [@qtnx_](https://twitter.com/qtnx_/status/2011510403550024087?s=20).
- **Data Monopoly: AI Agents Gatekeeping Datasets?**: Olivia Moore highlighted a trend where AI agent subscriptions, like **Manus**, offer extended proprietary data access, such as **12 months** of SimilarWeb data versus the free plan's **one month**.
- **Thinking Machines CTO Shuffle!**: Mira Murati announced that Barret Zoph departed Thinking Machines, with Soumith Chintala succeeding as CTO; discussion ensued about the circumstances, [tweeted by @miramurati](https://twitter.com/miramurati/status/2011577319295692801).
- **Black Forest Labs Fires Up FLUX.2!**: Black Forest Labs launched **FLUX.2 [klein]**, a sub-second image generation model with a **4B** parameter model (**Apache 2.0**) and a **9B** open-weights version, available via [API](https://xcancel.com/bfl_ml/status/2011825819082244266?s=46).


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1461192422691438809)** (13 messages🔥): 

> `Local LLM Inference, Meeting Transcription, Speaker Diarization, WhisperX, NVIDIA NeMo` 


- ****Frye's Guide** matches cloud LLM cost/perf**: Charles Frye announced a [new Modal guide and code samples](https://xcancel.com/charles_irl/status/2011484220032762114?s=46) demonstrating how to run local LLM inference that matches or exceeds the performance and cost-effectiveness of major LLM APIs.
- **Users Ask: Local Meeting Transcription?**: Users asked if it is now possible to run local meeting transcription, akin to a local Granola, without needing cloud services, sparking discussion on local alternatives for meeting transcription.
   - Several options were suggested, including the now **2 year old** [AutoDiarize repo](https://github.com/Alignment-Lab-AI/AutoDiarize).
- **WhisperX emerges as transcription competitor**: Members suggest alternative repos like [whisply](https://github.com/tsmdt/whisply), [whisperX](https://github.com/m-bain/whisperX), and [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) as newer and/or better maintained options for local meeting transcription.
- **Local Transcription matches cloud perf on macOS**: One user said their local transcription on macOS is as fast as cloud solutions using an optimized **Parakeet V3 model** + speaker diarization on a M2 Pro 16GB.
- **The potato laptop problem**: A user voiced a concern about local transcription turning their laptop into a potato but expressed interest in post-processing when AFK.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1461342976977539169)** (5 messages): 

> `LLMs and Chemistry, ChemIllusion.com, Black Forest Labs FLUX.2, High-Speed Image Generation` 


- **LLMs Stumble in Chemistry**: LLMs struggle with chemistry, especially when hallucinating details like statins in a cholesterol structure, according to a [tweet](https://x.com/bfl_ml/status/2011825819082244266?s=46).
- **ChemIllusion: Tools to Fix LLM Chemistry**: A member is developing tools at [ChemIllusion.com](https://x.com/bfl_ml/status/2011825819082244266?s=46) to correct LLMs' chemistry errors.
- **Black Forest Labs' FLUX.2 Debuts**: **Black Forest Labs** introduced **FLUX.2 [klein]**, a high-speed image generation model as linked in this [tweet](https://xcancel.com/bfl_ml/status/2011825819082244266?s=46).
   - The model can achieve sub-second processing.
- **FLUX.2: Open-Weight Image Generation**: **FLUX.2** is available in two open-weight versions, a **4B model** under Apache 2.0 and a **9B model**.
   - It can be accessed via **API** or a **free demo app**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1461122017435713636)** (50 messages🔥): 

> `GLM 4.7, Minimax, AI image to video tools, Sora 2 code, GPT 5.2 availability` 


- **GLM 4.7 & Minimax touted for Quality & Affordability**: Members reported **GLM 4.7** and **Minimax** LLM providers as fantastic, with **GLM 4.7** accessible via z.ai coding plan and **Minimax** being very cheap via Moonshot.
- **Image-to-Video AI Tool Sought**: A member was seeking the best AI tool specifically for producing a large amount of images to videos in a few days, suggesting preference for a *paid* option.
   - Someone recommended to *use API*.
- **GPT 5.2 Access Gone?**: Some members reported that **GPT 5.2** option disappeared for certain accounts, though it reappeared after logging out and back in; there were claims that *5.2 is a worse model*.
   - One member lamented the "your limit exceeded" message despite using **GPT 5.2**.
- **AI-Deepfake Certification Pilot Launched**: A member is working on an early pilot for an **AI deepfake detection & verification certification** tied to a platform called PhantomTrace.
   - They are looking for a small group of researchers, builders, security folks, journalists to review draft learning objectives, test hands-on detection labs, and help define what "passing" should mean, linking to [Discord context](https://discord.com/channels/974519864045756446/1204360881593520128/1461532097641578672).
- **Cognitive Decline Concerns vs Managerial Enhancement**: One member expressed concern that **AI** is going to be detrimental to cognitive ability.
   - However, others argued that **AI** enhances cognitive ability when used correctly, turning users from workers to managers when they can *manage it properly*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1461148947169939527)** (11 messages🔥): 

> `Scam reports, CustomGPT inside Project` 


- **Scammer Alert Sounds**: A user reported a potential scammer and tagged a moderator for assistance.
   - A staff member suggested opening a ticket for assistance and assured that a reply would be provided as soon as possible.
- **CustomGPT Projected to Take Over**: A user expressed a desire to use **CustomGPT** inside a Project, or place the result of the **CustomGPT** inside a Project.
   - They also want to be able to move any Chat generated outside a Project, into the Project.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1461301809791635511)** (1 messages): 

> `Prompt Engineering Definition, Value of Prompt Engineering` 


- **Prompt Engineering Definition Clarified**: A member expressed confusion about the definition of **Prompt Engineering**.
   - Another member confirmed that prompt engineering is the science of how to prompt effectively.
- **Appreciating Prompt Engineering**: A member expressed gratitude for the clarification on prompt engineering.
   - They acknowledged recent confusion regarding their work in the field and valued the insight provided.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1461301809791635511)** (1 messages): 

> `Prompt Engineering` 


- **Prompt Engineering Clarified**: A user expressed confusion about the true essence of **prompt engineering**.
- **Understanding Prompts**: Another user responded with gratitude for the clarification, and did not provide any specific examples or links.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1461110462208016384)** (30 messages🔥): 

> `LLM-generated text classifier, Pangram's accuracy, EleutherAI datasets contributions` 


- **Hunting small classifiers for LLM-generated text**: A member is seeking a **small classifier model** to estimate the amount of synthetic text on the web, considering running a web crawl through it.
   - Others suggest using a **drafter model** trained for speculative decoding, although it would be model-specific and potentially expensive at scale or building their own classifier.
- **Pangram's accuracy questioned**: Members discussed the accuracy of **Pangram** as an **AI text detector** with one member linking [Pangram's website](https://www.pangram.com) and another sharing [the paper](https://arxiv.org/abs/2402.14873) behind their detection methods.
   - One member reported that **Pangram** inaccurately identified blog posts explicitly stating they were written using **Claude** as *100% human written*.
- **Counting em dashes as AI-generated text metric**: A member suggested counting **em dashes** in text as a metric for detecting **AI generation**, comparing the count now versus 2022.
   - They note this method has an estimated *+-10% margin of error*, but is valuable for its low cost.
- **EleutherAI datasets contribution requests**: A member inquired about the community's interest in **open-sourcing instruction-following datasets** for finetuning pre-trained LLMs like **GPT-Neo**, in addition to pretraining datasets like The Pile and CommonPile.
   - Another member offered their developer skills for super cool projects in the community.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1461269928651919575)** (7 messages): 

> `dye doped liquid crystal nonlinearities for optical nns, prompt capitalization/grammar impact on model performance, vLLM benchmarking tools` 


- **Liquid Crystal Nonlinearities in Optical NNs**: A member shared their work on *dye doped liquid crystal nonlinearities* for potential optical neural networks.
   - No links were given in this topic.
- **Capitalization Caps Model Capacity?**: A member inquired about research on whether models perform better when prompted with proper **capitalization/grammar** versus all lowercase.
   - They suggested this would be valuable for *getting the most out of your agent* and a simple hypothesis to test, and pointed to [three Arxiv papers](https://arxiv.org/abs/2310.11324), ([2411.10541v1](https://arxiv.org/abs/2411.10541v1)), ([2508.11383v1](https://arxiv.org/abs/2508.11383v1)) but noted they focus on prompt format rather than minor details like capitalization.
- **vLLM's Benchmarking Benefits**: One member expressed the assumption that proper capitalization/grammar improves model performance and suggested testing this using **vLLM's benchmarking tools**.
   - No links were given in this topic.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1461163339437838550)** (1 messages): 

> `Global Chain of Thought Analysis, LessWrong Post` 


- **Global CoT Analysis Attempts Uncovering Patterns**: A member shared a [LessWrong post](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj/global-cot-analysis-initial-attempts-to-uncover-patterns-1) about **global chain of thought analysis** and initial attempts to uncover patterns.
   - The analysis seeks to understand how models arrive at their conclusions by examining the reasoning steps they take, potentially revealing insights into their decision-making processes.
- **Tweet on AI**: A member shared a [tweet](https://fxtwitter.com/i/status/2011501268603453626) about AI.
   - No further details were provided about the tweet's content or its relevance to the discussion.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1461382804892942417)** (22 messages🔥): 

> `Arch vs Ubuntu, PR imported internally, .NET Legacy Projects` 


- ****Arch Afficiandos Acclaim Always-Updated Arch****: Members argue that **Arch** is better than **Ubuntu** and **Debian** because it always uses the newest versions of packages, like **macOS** and **Windows**.
   - One user recommends **Garuda KDE** (Mokka and Dragonized) to newbies as it provides valuable functionality.
- ****PR Pipeline Plunge: Testing Transpires****: A member asked what the `imported internally` label means on a PR, another clarified that it means *the PR has been cloned into the internal repo for final testing and integration*.
   - Another member added that when a PR is tagged with `imported internally` it means *your PR is on the last stretch before offically getting merged*, also tagged with `merged-internally` once merged.
- ****.NET Nightmare: Legacy Lament****: A member lamented being pulled into a **.NET 4.5.2** legacy project at work, which was released in **2014** and only runs on **Windows** without a readme.
   - Another shared a similar experience regarding a standalone **C#** project with issues, zero documentation, and the original developer having retired, emphasizing this repo is *like finding a hotspring and water in a desert*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1461381878241169581)** (14 messages🔥): 

> `shader programming in Mojo, SPIR-V backend for Mojo, compute shaders in Mojo, shader vs matrix ops, CUDA` 


- **Mojo Eyes Graphics Shaders with SPIR-V Backend**: Mojo is considering graphics shaders, especially with a **SPIR-V backend**, enabling *compute shaders*.
   - One member noted that building the compiler is a *non-trivial* task once **open source**.
- **Bridging Mojo with MLIR's SPIR-V Dialect**: Integrating Mojo with **MLIR's SPIR-V dialect** would require a bridge and relevant **metaprogramming**.
   - The creation of such bridge is important for graphics people that call them compute shaders.
- **Shaders vs. Matrix Ops: A Deep Dive**: Someone questioned the difference between **shaders** and traditional **matrix operations**, especially with recent **CUDA** advancements.
   - In response, one member provided a link to [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api) to help explain the differences and another member linked to [Death to Shading Languages](https://xol.io/blah/death-to-shading-languages/).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1461244157723934740)** (16 messages🔥): 

> `Minimax m2.1 vs Kimi K2, Kimi K2 Turbo, Moonshot API, Model deprecation policy` 


- **Minimax M2.1 reportedly outperforms Kimi K2 in Claude**: A user reported that **Minimax m2.1** running in **Claude code** is outperforming **Kimi** in code quality, thinking/planning, and API speed, after running both side by side.
   - The user, who pays $40/month for **Kimi v2**, found the API slow and the model inferior to **Minimax**, and expressed the hope for a newer, better model release soon.
- **Is Kimi CLI defaulting to K2 Turbo?**: A user questioned why the default **Kimi CLI app**, with a proper subscription, was not defaulting to **K2 Turbo**.
   - Another member suggested that **Kimi K2 Turbo** should have around **73 tps**, compared to **MiniMax m2.1's 38 tps** and **Z.Ai's GLM-4.7's 41tps**, though the latter has poor uptime.
- **New slide feature using newer K2 model with Vision?**: A member inquired whether the new slide feature is using a newer **K2 model with Vision**.
   - The image analysis suggests that it searches images for reference, so it must have some sort of vision capability.
- **Kimi's Model Deprecation Policy?**: A member asked if **Kimi models** are discontinued every **12-14 months** like **Google's Gemini models** and whether the same problem will be faced if switching to **Kimi K2**.
   - Another member mentioned that older models are available on the [Moonshot API platform](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1) and that a year old model can still be used on [kimi.com](https://kimi.com).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1461110469162172477)** (13 messages🔥): 

> `developer for super cool project, Discord mod, Minecraft mod, AI engineer, Payment issue` 


- **Developer Seeks Super Cool Project**: A member is looking for a **super cool project** to contribute their **developer skills** to and is open to contact.
- **Discord Mod Position Unavailable**: A member expressed interest in becoming a **Discord mod**, but another member stated that it's currently not possible.
- **Seeking AI Engineer for Usage Tracking**: A member is seeking a skilled **AI engineer** to help harden usage tracking or build a more reliable billing/credit system on a real project.
- **User Encountered Payment Issues**: A member reported experiencing payment issues when trying to add more credit, including problems with upgrading membership, using Link, and paying with credit card or Alipay.
   - They reported *no response yet* from helpdesk and email.


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1461392795339329618)** (1 messages): 

> `Live stream from NY summit, Remote Registration` 


- **Livestream Link Lamented**: A member inquired whether there would be a **live stream** from the summit in NY.
   - They would have loved to register, but they cannot be there in person.
- **In-Person Summit Attendance**: A member expressed interest in attending the summit remotely due to their inability to attend in person.
   - They are seeking information about **live stream** options or remote participation possibilities.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1461311094760149012)** (10 messages🔥): 

> `Schema Freezing vs Dynamic Server Features, MCP Server Statelessness, Persistent Sessions, Dynamic Toolsets, State Management in MCP` 


- **Stateless Servers Spark Scalability Savings**: A member proposed a [signature method](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2091) to balance schema freezing and dynamic server features, aiming to allow **stateless MCP servers** to handle multiple active conversations more efficiently.
   - They noted that their current setup in Goose, which spins up a new set of MCP servers for each conversation, is becoming increasingly expensive as the number of concurrent conversations rises.
- **Dynamic Toolsets Tackle Transports**: One member pointed to [issue #1442](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1442) and GitHub MCP's dynamic toolsets as examples of how servers can handle state on **STDIO**, potentially unifying behavior for both remote and STDIO setups.
   - The member admitted it is difficult to maintain a truly stateless **STDIO server**, given their current SDK architecture that builds a new "server" on every request, customizing registered tools based on user/flags.
- **Persistent Sessions Save State on Server Starts**: The topic of **persistent sessions** was raised as a means to retain session features across agent and MCP server restarts.
   - Another member mentioned using their own session middleware outside the Go SDK for horizontal scaling, suggesting that the ability to store and retrieve **session data** across restarts would be beneficial.
- **MCP State Confusion Complicates Conversations**: There's *confusion about whether application-level state is possible in MCP or not*, with ongoing discussions about resolving this in the transports group as noted in channel <#1399986181445386352>.
   - It was mentioned that most popular clients maintain one session with a remote server across all conversations, while some servers maintain state between tool calls, which does not work well in situations where there are multiple conversations.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1461252514136592536)** (9 messages🔥): 

> `1x1 Convolution vs SVD/PCA, Quanta Theory of Neural Networks, AI Assisted Coding vs Vibe Coding` 


- **1x1 Convolution could outperform SVD/PCA**: A member suggested using a **1x1 Convolution** instead of **SVD/PCA** for feature extraction, arguing that **SVD/PCA** extracts features with the highest variance (*loudest*), potentially capturing generic syntax noise rather than specific *intent* signal, with [a link to the original tweet](https://fxtwitter.com/i/status/2011094378396467316).
   - They believe that **1x1 Conv** would allow the model to learn precisely which heads matter for the loss function via backprop and would be lighter for inference.
- **"Quanta" theory attracts discussion**: Members discussed the *quanta* theory which states that networks must learn a variety of modules, each implementing a different algorithm or retrieving a different piece of knowledge.
   - One member expressed skepticism, suggesting that many mechanisms could be entangled or too generic to be designated a specific use, potentially leading to an oversimplified mechanistic explanation of neural networks.
- **AI Assisted Coding versus Vibe Coding**: A member contrasted **AI assisted coding** tools (cursor/windsurf/antigravity) with what they termed **vibe coding** tools (devin/tembo/jules).
   - No further details were provided.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1461326860037390398)** (8 messages🔥): 

> `Native vs Custom Tool Calling Benchmarks, DSPy Tool Performance, Language Model Differences, Model-Specific Tool Performance` 


- **DSPy Tooling may outperform native LLM tools**: A member references the [DSPy documentation](https://dspy.ai/learn/programming/tools/#using-native-tool-calling) and asks about benchmarks comparing **native tool calling** versus **custom tool calling**.
   - Another member responds that the statement suggesting **DSPy tools** are better than **native tools** is generic and depends on the specific language model (LM) being used.
- **Native and DSPy tools should be benchmarked**: A member emphasizes that performance varies across language models, even those from the same AI lab, so **benchmarking is essential** for specific use cases and model combinations.
   - Another member agrees, stating that performance can vary in either direction, and users should test with their **specific model and program** to measure and evaluate what works best.
- **It's possible that native tool calling produces lower quality**: The statement in the documentation is a weak claim
   - That's possible and can happen for some models


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1461459876973903894)** (2 messages): 

> `Aider asking to add files, Aider configuration` 


- **Aider nags users to add files**: A user asked if it's possible to have **aider** automatically add files instead of prompting the user.
   - This could streamline the workflow for users who prefer less interactive file management.
- **Aider Configuration Wishlist**: A user inquired about configuring **aider** to bypass the prompt for adding files.
   - The user wants aider to automatically add files, suggesting a preference for a less interactive workflow.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1461180664262164659)** (5 messages): 

> `aider setup, CLIProxyAPI, CI logs with Aider` 


- **Users Struggle with Aider Setup**: A user reported difficulty setting up **aider** after installation via command prompt, seeking guidance on next steps, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1133060505792159755/1461275448574218373/image.png?ex=696a9f10&is=69694d90&hm=19ccaef4fb45cd4288b6307abb3eca0a6819f27eb6253f0820357b2219006a4d).
   - No further information was provided.
- **Incorporating CI Logs into Aider Workflow**: A user inquired about the best practice for using **CI logs** with **aider** to fix failed tests, while excluding the log file from Git.
   - The command `aider --read ci.log` was suggested as a potential solution.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1461220391031930925)** (6 messages): 

> `Black PNGs in stable diffusion, NULL device in tinygrad` 


- **Black PNGs plague Stable Diffusion runs!**: A new user reported getting a fully black PNG when running `examples/stable_diffusion.py` with `--fakeweights`.
   - It was clarified that the **NULL device** doesn’t actually perform any computations, but still makes and schedules the kernels.
- **NULL Device: A compute-less wonder!**: A user inquired about the purpose of the **NULL device** in tinygrad, questioning if it performs any computations.
   - Another member confirmed that the **NULL device** doesn't do any compute, clarifying its role in kernel scheduling without actual processing, and another member replied *that's a cool feature*.


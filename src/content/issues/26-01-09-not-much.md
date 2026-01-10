---
id: MjAyNi0w
title: not much happened today
date: '2026-01-09T05:44:39.731046Z'
description: >-
  **Anthropic** tightens usage policies for **Claude Max** in third-party apps,
  prompting builders to adopt **model-agnostic orchestration** and **BYO-key**
  defaults to mitigate platform risks. The **Model Context Protocol (MCP)** is
  evolving into a key tooling plane with **OpenAI MCP Server** and **mcp-cli**
  enhancing tool discovery and token efficiency. The concept of **skills** as
  modular, versioned behaviors gains traction, with implementations in **Claude
  Code**, **GitHub Copilot**, and **Cline** adding websearch tooling. AI21 Labs
  addresses concurrency challenges in agent workspaces using **git worktrees**
  for transactional parallel writes, while long-horizon agents focus on
  **context engineering** and persistent file-centric workspaces.
companies:
  - anthropic
  - openai
  - ai21-labs
  - github
  - cline
models:
  - claude-max
topics:
  - model-agnostic
  - model-context-protocol
  - tooling
  - skills
  - concurrency
  - transactional-workspaces
  - context-engineering
  - file-centric-workspaces
  - rate-limiting
  - agent-workspaces
people:
  - yuchenj_uw
  - andersonbcdefg
  - gneubig
  - matan_sf
  - scaling01
  - reach_vb
  - _philschmid
  - claude_code
  - code
  - jamesmontemagno
  - cline
  - danstripper
  - omarsar0
---


**DeepSeek v4 coming...**

> AI News for 1/8/2026-1/9/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**204** channels, and **4384** messages) for you. Estimated reading time saved (at 200wpm): **402 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

Claude Code is continuing to be in the news for various reasons, read on...

---

# AI Twitter Recap


**Policy + platform shifts shaping the “coding agent” ecosystem**

- **Anthropic tightens Claude Max usage in third-party apps**: Multiple posts describe Anthropic blocking Claude subscriptions from being used inside external clients (and reportedly cutting off some competitors), reinforcing the risk of building product-critical workflows on a single provider’s consumer plan. See [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2009691122940211201) plus reactions from builders like [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/2009509161823031351) and a more market-structure framing from [@gneubig](https://twitter.com/gneubig/status/2009686033563316501). Practical implication: expect **more model-agnostic harnesses** and **BYO-key** defaults, and treat “max plan” access as revocable.
- **Model-agnostic orchestration becomes a product requirement**: Several builders emphasize not being “all-in” on one provider due to rate limits and policy changes. Example: [@matanSF](https://twitter.com/matanSF/status/2009472570438095130) argues for **model-agnostic** infra to reduce platform risk; [@scaling01](https://twitter.com/scaling01/status/2009568477972201686) notes **rate limiting** when hitting Opus token limits, highlighting the need for fallback routing and budgeting.

**Agents & developer tooling: MCP, skills, harnesses, and long-horizon reliability**

- **MCP (Model Context Protocol) is rapidly turning into the “tooling plane”**:
  - **OpenAI MCP Server**: OpenAI-aligned folks announced an MCP server bundling docs/guides/APIs/AppsSDK/etc. intended to work out-of-the-box with Codex/Cursor/VSCode and other agents ([tweet](https://twitter.com/reach_vb/status/2009686112986337309), [follow-up](https://twitter.com/reach_vb/status/2009686476255084767)). The subtext: MCP as a distribution channel for *official* tool interfaces, not just community plugins.
  - **mcp-cli**: A lightweight CLI for **dynamic discovery** of MCP servers, claiming **99% token usage reduction** via discovery rather than verbose prompt/tool descriptions; supports stdio + HTTP, piped JSON output, and grep across servers ([tweet](https://twitter.com/_philschmid/status/2009625698361573521), [links](https://twitter.com/_philschmid/status/2009625701432152438)). This is the “ops” side of MCP: make tools discoverable and scriptable without ballooning context.
- **“Skills” as modular, versioned behaviors**:
  - Claude Code’s framing: **plugins** as containers; **skills** as specialized procedures/knowledge, and some artifacts can be both (e.g., “Frontend-design”) ([tweet](https://twitter.com/claude_code/status/2009479585172242739)).
  - GitHub Copilot / VS Code: “Agent Skills” shipping in stable; quickstart video + positioning as a workflow accelerant ([@code](https://twitter.com/code/status/2009744142335656156), [@JamesMontemagno](https://twitter.com/JamesMontemagno/status/2009720264335208598)).
  - Cline: adds **skills compatibility** and built-in **websearch tooling** ([tweet](https://twitter.com/cline/status/2009793063753757024)).
  - Pattern: teams are converging on “skills” as **lazy-loaded instruction bundles** to avoid stuffing everything into base prompts.
- **State, concurrency, and the “parallel writes” problem**:
  - AI21 describes a real pain point: MCP works until you run multiple subagents that need to **write files concurrently**; they add an “MCP Workspace” layer with primitives (init/clone/compare/merge/delete) and implement code workspaces via **git worktrees**, enabling **1 → 16 parallel attempts** without coordination, then merge the winner ([thread start](https://twitter.com/AI21Labs/status/2009565879600923100), [workspaces](https://twitter.com/AI21Labs/status/2009565885284200540), [git worktrees + results](https://twitter.com/AI21Labs/status/2009565888148652226)). This is a concrete step toward **transactional agent workspaces**.
- **Long-horizon agents: “context engineering” is the core bottleneck**:
  - InfiAgent proposes keeping reasoning context bounded by externalizing persistent state into a **file-centric workspace** reconstructed each step from a snapshot + a fixed recency window ([summary](https://twitter.com/omarsar0/status/2009662975024447511)). This aligns with the growing “agents are files/folders” mantra (e.g., [@danstripper](https://twitter.com/danshipper/status/2009651408144835021)).
  - “Agent drift” is highlighted as a common multi-agent failure mode—semantic/coordination/behavioral drift—plus an Agent Stability Index and mitigations like episodic consolidation and behavioral anchoring ([thread](https://twitter.com/dair_ai/status/2009657177989091423)). Net: evaluate not just task success, but **stability over interaction length**.
- **Evals for agents move from theory to practice**:
  - Anthropic’s “Demystifying evals for AI agents” is widely shared as a production-oriented playbook: graders (code/model/human), capability vs regression evals, pass@k vs pass^k, and starting from real failure cases ([tweet](https://twitter.com/AnthropicAI/status/2009696515061911674)).
  - Practitioner amplification stresses looking at **agent traces** to learn *how* failures happen and to co-evolve instructions/tools/harnesses with eval design ([@Vtrivedy10](https://twitter.com/Vtrivedy10/status/2009724848482762966), [@hwchase17](https://twitter.com/hwchase17/status/2009732201269588479)).

**Model + dataset releases and benchmarking signals**

- **Falcon-H1R-7B (TII UAE)**: Artificial Analysis reviews the model as a small open-weights **reasoning** entrant with strong positioning for its size; notes Openness Index score impacted by attribution-required licensing; highlights strength on Humanity’s Last Exam, τ²-Bench Telecom, and IFBench ([analysis](https://twitter.com/ArtificialAnlys/status/2009690138604122238), [links](https://twitter.com/ArtificialAnlys/status/2009690152608903446)).
- **Open-weights “frontier pressure” continues**: Several tweets point to accelerating open model competitiveness and the strategic gap vs US-based open releases ([Artificial Analysis trend note](https://twitter.com/ArtificialAnlys/status/2009759874461081957), plus builder sentiment like [@Teknium](https://twitter.com/Teknium/status/2009630706146984178)).
- **FineTranslations dataset (synthetic parallel corpus)**: A new **>1T token** parallel dataset created by translating FineWeb2 multilingual data into English using Gemma3 27B ([tweet](https://twitter.com/gui_penedo/status/2009677127671492616)). Practical use: multilingual alignment, distillation, translation/RAG training, and evaluation.
- **Benchmark volatility is now measurable**: LM Arena reports average #1 tenure of **~35 days**, with leaders dropping out of top 5 within ~5 months ([tweet](https://twitter.com/arena/status/2009720083170636030)). This reframes “which model is best” as a **short-lived advantage**, increasing the value of routing, eval automation, and portability.

**RL, optimization, and “multi-reward” training gets more rigorous**

- **GDPO (Group reward–Decoupled Normalization Policy Optimization)**: Introduced as an alternative to GRPO for multi-reward RL, aiming to improve per-reward convergence via decoupled normalization ([thread](https://twitter.com/shizhediao/status/2009481573217784016)). Follow-on discussion notes GRPO’s flaw where different reward combinations can collapse into identical advantage values, explaining instability ([commentary](https://twitter.com/AliceInWeights/status/2009576516829774216)).
- **Optimization theory refreshers**: Jianlin Su continues a convex optimization series focused on gradient-based learning-rate schedules ([tweet](https://twitter.com/Jianlin_S/status/2009463828476776494)).
- **Learning dynamics / scaling theory**: “Learnable Multipliers” proposes freeing matrix layer scaling in LMs ([tweet](https://twitter.com/VelikanovMaksim/status/2009585864880554344)); related chatter about combining learnable MuP + Muon appears ([tweet](https://twitter.com/yb2698/status/2009589919635952108)).

**Inference + infra: reliability, speedups, and compute scaling**

- **GPU reliability as a first-class engineering problem**: Modal reports operating at **20,000+ concurrent GPUs** across multiple clouds with 1M+ instances launched, and details mitigation strategies for public-cloud failure modes ([tweet](https://twitter.com/jonobelotti_IO/status/2009696881052729669)). The broader takeaway: multi-cloud + health checks + scheduling policies are becoming table stakes for serious inference/training platforms.
- **Diffusion/speculative decoding for throughput**: Modal-associated posts highlight SGLang support for “DFlash” and claim **4.73× tok/s** boosts on H200 + FA3 compared to autoregressive baselines ([tweet](https://twitter.com/akshat_b/status/2009741089931178244), [PR](https://twitter.com/akshat_b/status/2009741161271828719)). Engineers should read this as: “speculation is moving from papers to production PRs quickly.”
- **Compute growth + megawatt realities**:
  - Epoch AI estimates total AI compute is doubling every **~7 months** based on accelerator production, with NVIDIA >60% of new capacity ([thread](https://twitter.com/EpochAIResearch/status/2009757548891852929)).
  - Epoch AI also estimates Anthropic’s Indiana data center at **~750 MW**, approaching 1 GW soon ([thread](https://twitter.com/EpochAIResearch/status/2009761084618797152)). This contextualizes why providers police subsidized usage and why reliability/power constraints now shape product policy.

**Industry moves: IPOs, hiring signals, and “agent-native” product direction**

- **MiniMax IPO and multimodal positioning**: Bloomberg frames MiniMax’s early focus on a unified multimodal model (text/speech/video) and notes IPO-driven wealth creation ([Bloomberg tweet](https://twitter.com/business/status/2009478615453364599)); MiniMax announces listing and pushes an “open ecosystem” narrative and third-party integration via its coding plan ([IPO](https://twitter.com/MiniMax_AI/status/2009491818690547938), [ecosystem post](https://twitter.com/MiniMax_AI/status/2009500121294360727)).
- **“Agent-native software” gets a concrete design language**: A technical guide argues for five pillars—parity, granularity, composability, emergent capability, self-improvement—and pushes “files as the universal interface” plus capability discovery patterns ([tweet](https://twitter.com/danshipper/status/2009651408144835021)). This theme recurs across MCP/workspaces/InfiAgent: **state should live outside the chat transcript**.
- **Hiring/compensation extremes + talent density talk**: Posts note unprecedented comp offers being declined ([tweet](https://twitter.com/nearcyan/status/2009558081810886729)) and “talent dense” recruiting pitches ([tweet](https://twitter.com/sarahookr/status/2009683294607270265)). Meanwhile, academic recruiting also appears (e.g., McGill packages via grants) ([tweet](https://twitter.com/sivareddyg/status/2009656185507496112)).

---

### Top tweets (by engagement)

- **AI org dynamics**: [@VahidK](https://twitter.com/VahidK/status/2009476045712642152) (team competence claim) and reactions like [@Skiminok](https://twitter.com/Skiminok/status/2009712629573660750).
- **Product/tooling**: Anthropic agent evals blog share ([@AnthropicAI](https://twitter.com/AnthropicAI/status/2009696515061911674)); Claude Code workflow impact ([@alexalbert__](https://twitter.com/alexalbert__/status/2009706598151929888)); Cursor CLI update ([@n2parko](https://twitter.com/n2parko/status/2009690110078685531)).
- **Model/media creation**: Midjourney Niji V7 launch ([tweet](https://twitter.com/midjourney/status/2009748519133827304)).
- **Healthcare product hint**: “ChatGPT Health” early access description ([tweet](https://twitter.com/omooretweets/status/2009468969015734327)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Quantization and Model Optimization Benchmarks

  - **[We benchmarked every 4-bit quantization method in vLLM 👀](https://www.reddit.com/r/LocalLLaMA/comments/1q7ysj2/we_benchmarked_every_4bit_quantization_method_in/)** (Activity: 145): **The post presents a comprehensive benchmark of various 4-bit quantization methods in **vLLM** using the **Qwen2.5-32B** model on an **H200**. Key findings include: **Marlin** achieving `712 tok/s`, outperforming the baseline **FP16** at `461 tok/s`, while **GPTQ** without Marlin kernel is slower than FP16 at `276 tok/s`. **BitsandBytes** showed the smallest quality drop and doesn't require pre-quantized weights, whereas **GGUF** had the worst perplexity but the best HumanEval score among quantized methods. **AWQ** was notably slow at `67 tok/s` in vLLM. The blog provides detailed insights into the workings of each technique [here](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks).** Comments highlight skepticism about the results, particularly regarding the claim of 4-bit quantization when the model appears to be mostly 5-bit. Concerns are raised about vLLM's suitability for serving GGUF models due to performance discrepancies. Additionally, the AWQ speed is questioned, suggesting potential issues with the setup, while BitsandBytes' dynamic quantization is praised for quality retention.

    - The discussion highlights a potential misrepresentation in the benchmark, where a model claimed to be 4-bit quantized is actually using a 5-bit quantization method (`q5_k_m`). This discrepancy raises doubts about the reliability of the results, especially since the performance differences are significant, suggesting that vLLM might not be optimal for serving GGUF models, as indicated by the unexpected perplexity results.
    - There is a critique regarding the mixing of different quantization types and execution kernels, particularly with AWQ quantization using Marlin kernels on NVidia hardware. The claim that AWQ is slow is challenged, as it doesn't align with expected performance, indicating a possible issue with the benchmarking setup or execution environment.
    - The comment points out an inconsistency in the results where GGUF models, despite having the worst perplexity, achieve the best quantized HumanEval rating. This raises questions about the validity of using perplexity and HumanEval as metrics for evaluating quantization quality, suggesting a potential flaw in the testing methodology or metric interpretation.

  - **[Gemma-3-4b (null-space) abliteration &amp; RP fine-tune](https://www.reddit.com/r/LocalLLaMA/comments/1q7xd96/gemma34b_nullspace_abliteration_rp_finetune/)** (Activity: 15): **The post discusses the application of a LoRA adapter on the `Gemma-3-4B-IT` model, which uses null-space ablation to enhance its performance. The model is fine-tuned using a subset of the `LimaRP` dataset, focusing on roleplaying capabilities. The author plans to remove the step limit and reduce the learning rate in future iterations. The model card provides detailed training information, and the author seeks feedback before scaling to larger models. For more details, see the [Hugging Face model page](https://huggingface.co/jwest33/gemma-3-4b-null-space-abliterated-RP-writer).** A commenter is interested in testing the model's ability to analyze chat content for scene data and memory extraction, indicating potential applications in LLM projects.



### 2. Local AI Setup and Hardware Considerations

  - **[I spent 9 months building a local AI work and play platform because I was tired of 5-terminal setups. I need help testing the Multi-GPU logic! This is a relaunch.](https://www.reddit.com/r/LocalLLaMA/comments/1q7xoid/i_spent_9_months_building_a_local_ai_work_and/)** (Activity: 6): ****Eloquent** is a local AI platform developed over nine months, integrating functionalities like chat, image generation, and voice cloning into a single application using **React** and **FastAPI**. It supports **multi-GPU orchestration**, allowing users to shard models across multiple GPUs or assign specific tasks to different GPUs. Key features include a Story Tracker for roleplayers, a Choice Generator, and a multi-modal stack with **Stable Diffusion** and **Kokoro voice cloning**. The platform also includes an ELO Testing Framework with 14 personality judges for model evaluation. The developer seeks testers with multi-GPU setups to validate tensor splitting and VRAM monitoring, especially on older cards. More details can be found on the [Eloquent GitHub page](https://github.com/boneylizard/Eloquent).** One commenter expressed interest but noted they are a Mac user, indicating potential platform limitations. Another associated the name 'Eloquent' with Laravel's ORM, suggesting possible branding confusion.


  - **[LLM server will it run on this ?](https://www.reddit.com/r/LocalLLM/comments/1q82yvp/llm_server_will_it_run_on_this/)** (Activity: 19): **The user is considering setting up a local LLM server using an HP DL380 G9 with 2x Intel Xeon E5-2697 v3 CPUs and 128 GB DDR4 RAM, but lacks a GPU. The goal is to process project-specific PDFs via Retrieval-Augmented Generation (RAG) for team coding queries. The hardware's limitations, particularly the absence of a GPU, are a concern for running large language models effectively. Suggestions include using smaller models or enhancing the server with GPUs via available PCIe slots to improve performance.** Commenters suggest that the current hardware setup is inadequate for running LLMs effectively, especially without a GPU. They recommend testing with smaller models or adding GPUs to the server to enhance its capabilities. One commenter shares their own setup experience, highlighting the importance of GPU support for satisfactory performance.

    - **SimilarWarthog8393** highlights the impracticality of running AI servers with parallel RAG requests on systems without a GPU, especially when relying solely on DDR4 RAM. They suggest that only very small Mixture of Experts (MoE) models might be feasible under such constraints.
    - **WishfulAgenda** provides a detailed hardware recommendation, suggesting the addition of multiple PCIe x16 slots and potentially acquiring 16GB or 24GB GPUs to enhance performance. They share their own setup experience with a 3950x CPU and dual 5069ti GPUs, running a Qwen3 Coder 30B model with Docker containers and a VM, indicating that a second GPU significantly improved their system's performance.
    - **TheRiddler79** discusses using the `gpt-oss-120b` model, noting it achieves approximately 5 tokens per second, which is adequate for a single user but slows down with multiple simultaneous users. They mention that this setup requires 64 GB of RAM and works on a Dual Xeon system, specifically referencing their own r2208 with 2697a chips.

  - **[Total beginner trying to understand](https://www.reddit.com/r/LocalLLM/comments/1q87tcs/total_beginner_trying_to_understand/)** (Activity: 21): **The user is exploring the feasibility of running a local LLM like Llama 13B with a Retrieval-Augmented Generation (RAG) system to serve as a persistent writing assistant. Their hardware includes an AMD Ryzen 7 8845HS CPU, 32GB RAM, and an NVIDIA RTX 4070 GPU with 8GB VRAM. Experts suggest that while running a 13B model is possible, it would require a deeply quantized model due to limited VRAM, which may lead to increased hallucinations. A 7B model might be more suitable given the hardware constraints. For memory, an in-memory K-V database like QDrant is recommended. Tools like Open-WebUI and KoboldCPP are suggested for setup, with SillyTavern for managing lorebooks. The complexity of RAG is highlighted, noting its limitations in precise memory recall. A reference implementation is available at [luna-system/ada](https://github.com/luna-system/ada/).** Commenters emphasize the limitations of current RAG systems in precise memory recall, suggesting that expectations should be managed. They also note that while local AI setups can be costly, they offer more control compared to cloud-based solutions. The potential of smaller models is highlighted as a future trend.

    - Ok_Stranger_8626 discusses the feasibility of running a 13B model on a GPU with limited VRAM, emphasizing the need for a deeply quantized model to fit within memory constraints. They highlight potential issues such as hallucinations due to loss of mathematical precision and suggest using an in-memory K-V database like QDrant for efficient data retrieval. They also mention the cost-effectiveness of cloud AI solutions compared to local setups, which can be expensive due to the lack of investment capital backing.
    - NobleKale provides a critical perspective on the limitations of Retrieval-Augmented Generation (RAG) for specific queries, explaining that RAG involves matching mathematical representations of prompts with document snippets. They caution that RAG may not accurately retrieve specific details, such as a character's eye color, due to its reliance on keyword proximity rather than precise context. They recommend using smaller models like 7B for better performance on limited hardware and suggest tools like KoboldCPP and SillyTavern for managing lorebooks and context.
    - DHFranklin suggests using Google AI Studio with Gemini 3 Pro for organizing and querying large text corpora. They recommend inputting the entire corpus and setting custom instructions to manage context effectively, avoiding issues like "context rot." The process involves creating RAG chunks and iteratively refining the model's understanding through clarifying questions. This approach aims to maintain consistency in querying specific details, such as character attributes, by comparing outputs to a "story bible."


### 3. Multimodal and Summarization Techniques

  - **[Call recording summarization at scale: Commercial STT + small fine-tuned LLM vs direct audio→summary multimodal(fine-tuned)?](https://www.reddit.com/r/LocalLLM/comments/1q861cb/call_recording_summarization_at_scale_commercial/)** (Activity: 4): **The post discusses two approaches for summarizing multilingual Indian language call recordings at scale: 1) a pipeline using commercial Speech-to-Text (STT) followed by a fine-tuned small LLM (e.g., Llama 8B), achieving approximately `90%` summary accuracy, and 2) a direct audio-to-summary approach using a multimodal model like Phi-4B, which supports long audio inputs and has a commercial license. The author is evaluating whether the direct approach could simplify the system by reducing latency and complexity, given the constraints of available models that support long audio inputs and have commercial licenses.** The comments suggest exploring tools like [AnythingLLM](https://anythingllm.com/desktop) for meeting assistance, indicating interest in practical implementations of similar technologies.


  - **[Multi modal llms vs specific llms](https://www.reddit.com/r/LocalLLaMA/comments/1q7xdcp/multi_modal_llms_vs_specific_llms/)** (Activity: 8): **The post discusses whether to use a single multi-modal LLM for both image and text generation or to use separate LLMs for each task, particularly when customizing outputs for a single user. A key point raised in the comments is that current multi-modal LLMs do not inherently generate images; they require separate models for image and text tasks. This suggests that, despite the appeal of a unified model, practical implementation still necessitates distinct models for different modalities.** The comment highlights a common misconception about multi-modal LLMs, emphasizing the need for separate models for image and text generation, which may influence the decision towards using specialized models for each task.

    - The comment highlights a common misconception about multi-modal LLMs, which are often thought to generate images directly. In reality, these models typically require integration with separate image generation models to handle visual tasks. This separation is due to the distinct architectures and training processes needed for text and image data, which are not yet fully unified in a single model.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. DeepSeek and Claude Code Developments

  - **[[D] deepseek published a new training method for scaling llms. anyone read the mhc paper?](https://www.reddit.com/r/MachineLearning/comments/1q893c1/d_deepseek_published_a_new_training_method_for/)** (Activity: 74): ****Deepseek** has introduced a new training method called *Manifold Constrained Hyper Connections (MHC)*, detailed in their [paper](https://www.arxiv.org/abs/2512.24880). Co-authored by **Liang Wenfeng**, the method addresses the instability issues that arise when scaling large language models (LLMs) by constraining information sharing within the model. This is achieved by restricting mixing matrices to a convex hull, which prevents signal explosion and results in a small improvement in loss and a significant enhancement in reasoning tasks. The method is seen as a potential breakthrough for scaling LLMs, with implications for future model versions like Deepseek v4.** Commenters note that while MHC offers stabilization benefits, it is more of a small optimization rather than a revolutionary change, akin to improvements seen with ResNet. The impact on network architecture could be significant, but the 'Sputnik moment' analogy is considered an overstatement.

    - fredugolon highlights that the new training method addresses stabilization issues in deep networks by restricting mixing matrices to a convex hull, which prevents signal explosion through hyper connections. This method reportedly shows a small improvement in loss during training and a significant enhancement in reasoning tasks, suggesting potential impacts on network architecture.
    - AccordingWeight6019 discusses the method as a constraint that enforces discipline during scaling, noting that while sharing internal states can be beneficial, it often leads to instability. The commenter questions the real-world applicability of the reported gains, suggesting that the impact might be more indirect and manifest in future generations, emphasizing the importance of planning and representation over raw capacity.

  - **[Claude Code creator open sources the internal agent, used to simplify complex PRs](https://www.reddit.com/r/ClaudeAI/comments/1q8h6oz/claude_code_creator_open_sources_the_internal/)** (Activity: 557): ****Claude Code** has open-sourced its internal code-simplifier agent, which is designed to clean up large and complex pull requests (PRs) by reducing complexity without altering behavior. This tool is intended to be used at the end of extensive coding sessions and is now available through the official plugin, as announced by **Boris X**. The source code can be accessed on [GitHub](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier).** Some users express skepticism about the tool's readiness for practical use, citing issues like inappropriate code simplifications and limitations such as token limits. Others highlight specific technical shortcomings, such as improper handling of function keywords and React component patterns.

    - PoorPhipps provides a link to the source code for the code simplifier used by Claude, which is available on GitHub at [https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier). This could be useful for developers interested in understanding the internal workings of the simplification agent.
    - --northern-lights-- critiques the current state of the code simplifier, highlighting that it suggests using the `function` keyword over arrow functions and emphasizes proper React component patterns with explicit Props types. This suggests that the tool may not yet be fully optimized for practical use, especially in complex codebases.
    - jtorvald shares an experience where Claude attempted to simplify code but ended up removing functional code and replacing it with dummy functions. This highlights potential issues with the simplification process, indicating that while the tool aims to reduce complexity, it might inadvertently compromise functionality.

  - **[The skills update in claude 2.1 are just amazing](https://www.reddit.com/r/ClaudeCode/comments/1q84z3u/the_skills_update_in_claude_21_are_just_amazing/)** (Activity: 184): ****Claude 2.1** introduces a significant update with recursive skill forking, allowing for complex orchestration by enabling sub-agents with their own context windows. This update facilitates the creation of task trees, avoiding the limitations of a single conversation context. Users can now run different models like `Opus`, `Haiku`, and `Sonnet` in parallel, enhancing modular agent construction by fanning out tasks, handling deep reasoning, and maintaining a clean main context during multi-phase workflows.** One commenter expressed confusion about the functionality, indicating a lack of understanding or access to the recursive skill feature. Another comment humorously noted personal limitations, while a third requested practical examples of tasks utilizing this feature.


  - **[Zuckerberg is watching you, whale, be careful](https://www.reddit.com/r/DeepSeek/comments/1q81v0z/zuckerberg_is_watching_you_whale_be_careful/)** (Activity: 25): ****DeepSeek** has updated the list of core contributors for the R1 paper, detailing their specific contributions. The update includes a note that contributors marked with an asterisk are no longer affiliated with the team, although it appears that all core contributors remain part of the team. This update is significant for tracking the development and authorship of the R1 paper, which is crucial for understanding the project's progress and the roles of individual contributors.** A comment highlights that despite the note about contributors no longer being affiliated, all core contributors seem to still be part of the team, suggesting stability or continuity in the project's development team.


  - **[deepseek is kinda same trafic share from last boom but chatgpt is loosing there are many reason first deepseek can write 10k plus token in one response giving paid model as free . high quality and and no ai slop](https://www.reddit.com/r/DeepSeek/comments/1q84z67/deepseek_is_kinda_same_trafic_share_from_last/)** (Activity: 45): **The image is a bar chart from Similarweb titled "Generative AI Traffic Share," illustrating the traffic distribution among various AI platforms over the past year. **OpenAI** holds the largest share, although it shows a declining trend. **Deepseek** maintains a smaller but stable share, while other platforms like **Meta**, **Claude**, and **Gemini** have smaller and more variable shares. The post suggests that Deepseek's ability to generate over `10,000 tokens` in a single response for free, compared to paid models, might be a factor in its stable traffic share.** Commenters express skepticism about Deepseek's impact, with one noting its traffic share as 'pathetic' and another highlighting the perception of overpriced AI services by US companies. There is also curiosity about why Claude, despite its reputation for coding, has a smaller share.

    - ExTraveler raises a point about Claude's performance in coding tasks, questioning why it has a small market share despite its reputation for being highly effective in coding. This suggests a potential mismatch between technical capabilities and market penetration, possibly due to factors like marketing, user interface, or integration with popular platforms.
    - Embarrassed_Bread_16 comments on the pricing strategies of US companies, implying that they may be overcharging for AI services compared to alternatives like Deepseek. This highlights a market dynamic where cost-effectiveness could be a significant factor in user adoption, especially if Deepseek offers high-quality outputs for free.
    - Suspicious_Today2703 critiques Deepseek's market share as 'pathetic,' suggesting that despite its technical capabilities, it may lack in areas such as marketing, user engagement, or feature set compared to competitors like ChatGPT. This points to the importance of not just technical prowess but also strategic business operations in gaining market share.

  - **[Claude Code has allowed me to execute on an idea I've dreamt about for years but always assumed I would be too dumb to do](https://www.reddit.com/r/ClaudeCode/comments/1q8eik3/claude_code_has_allowed_me_to_execute_on_an_idea/)** (Activity: 121): **The post describes how the author, an experienced engineer, used **Claude Code** to develop a proof of concept (POC) for a complex idea they previously thought impossible. Despite lacking expertise in low-level programming and database architecture, the author leveraged Claude Code's ability to introspect and refactor code, providing creative solutions to anticipated bottlenecks. This collaboration enabled the author to build a POC over a weekend, highlighting Claude Code's proficiency across programming domains and its potential to empower developers to tackle ambitious projects.** Commenters agree that AI tools like Claude Code are democratizing programming, enabling individuals with ideas but limited coding skills to execute complex projects. They emphasize the transformative potential of AI in expanding access to software development and enhancing productivity.

    - siberianmi highlights the transformative impact of AI tools like Claude Code on experienced professionals who traditionally focus on infrastructure and production problem-solving rather than coding. They emphasize that AI has provided them with coding capabilities they previously lacked the patience or desire to develop, although they note a limitation in token availability, which can be a constraint in using these tools effectively.
    - southafricanamerican discusses the democratizing effect of AI, enabling individuals with innovative ideas but limited technical knowledge to execute projects that were previously unimaginable. They suggest that AI will empower people to create new businesses or improve existing ones by simplifying complex tasks, thus broadening the scope of who can participate in tech development.
    - Ambitious_Injury_783 shares their experience of progressing from a proof of concept to a functional version of a project they initially thought impossible. They caution that while AI tools can accelerate development, they also reveal why certain projects haven't been built before, as many challenges still require human insight and problem-solving to avoid creating overly complex or inefficient solutions.

  - **[Running CC on an ipod](https://www.reddit.com/r/ClaudeCode/comments/1q817qc/running_cc_on_an_ipod/)** (Activity: 127): **The post describes a technical setup where the user is running "Claude Code" on an iPod using a custom-built terminal interface. The user initially faced challenges with SSH and ttyd on iOS 15, leading them to instruct Claude Code to create a terminal from scratch. This was achieved in under 10 minutes without writing any lines of code, showcasing the flexibility and power of Claude Code in adapting to different environments. The image shows a terminal interface on an iPod, indicating successful implementation of this setup.** A commenter suggests using CCC, a tool that connects to Claude Code on a machine without requiring SSH, offering a better coding experience with terminal and file browser integration. This suggests a community interest in simplifying remote coding setups on mobile devices.

    - naarang suggests using CCC, an app that connects to Claude Code running on a local machine without needing SSH or other credentials. This setup offers a better coding experience by integrating terminal and file browser functionalities. A new version, V2, is expected to be released soon, enhancing these features further. More details can be found at [getc3.app](https://getc3.app).
    - Mikeshaffer describes a method of accessing Claude Code by logging into a specific IP address using Tailscale, which opens a terminal interface. This setup is then saved as a Progressive Web App (PWA) on the desktop, providing a convenient way to access the terminal environment.

  - **[Does anyone know any deepseek v3 0324 provider that use paypal as mode of payment?](https://www.reddit.com/r/DeepSeek/comments/1q87tps/does_anyone_know_any_deepseek_v3_0324_provider/)** (Activity: 3): **The post is inquiring about providers for **Deepseek v3 0324** that accept **PayPal** as a payment method. The user expresses dissatisfaction with the current version available on the official Deepseek site, indicating it doesn't perform as well as the previous version. **Deepseek** is likely a specialized tool or service, but specific technical details or benchmarks about its performance or features are not provided in the post.** There are no notable technical opinions or debates in the comments, as the post primarily seeks information about payment options rather than technical details.


  - **[China's households are sitting on $22 trillion that could fuel massive growth of domestic AI, as dozens of Chinese developers and chip makers prepare IPOs.](https://www.reddit.com/r/DeepSeek/comments/1q85fso/chinas_households_are_sitting_on_22_trillion_that/)** (Activity: 74): **Chinese households hold $22 trillion in savings, which could significantly boost domestic AI growth as companies like **Zhipu** and **MiniMax** issue IPOs in Hong Kong. Historically, only `5%` of Chinese savings are invested in financial markets, but with the rise of Chinese models like **Qwen** in the global open-source space, this could increase. If households invest `5%` more, it could add $1 trillion to the market. The article suggests that Chinese open-source AI could challenge US proprietary models by offering competitive performance at a fraction of the cost, potentially shifting investment dynamics.** One comment highlights the potential concentration of savings among the wealthiest, questioning the broader impact on AI investment. Another notes China's strategy of developing models slightly behind the West but at significantly lower costs, appealing to consumers and small businesses. A question is raised about monetizing open-source models.

    - Bozzor highlights a strategic approach by Chinese tech companies, suggesting they often wait for Western innovations to mature before releasing their own versions. These versions are typically 80% as capable but cost less than 10% of the original, making them highly competitive in consumer and SMB markets. This strategy allows them to capture market share by offering affordable alternatives to cutting-edge technology.
    - Far-Pomegranate6895 points out that despite the potential for investment, low consumer confidence and spending in China could hinder the growth of domestic AI. This is attributed to recent government crackdowns on sectors like real estate and tech, which have led to a cautious investment climate. The lack of household investment in public stock markets further complicates the potential impact of AI companies.
    - alex_godspeed raises a question about the monetization of open-source models, which is a critical issue for developers. Open-source models often rely on alternative revenue streams such as offering premium features, support services, or enterprise solutions to generate income, as direct sales of the models themselves are not feasible.


### 2. OpenAI and Claude Billing and Usage Issues

  - **[Beware of OpenAI Billing Practices](https://www.reddit.com/r/OpenAI/comments/1q7yf8b/beware_of_openai_billing_practices/)** (Activity: 937): **The post highlights a billing issue with OpenAI's ChatGPT subscription service, where a user experienced unauthorized upgrades from a $20/month Plus plan to a $200/month Pro plan. Despite contacting support and receiving an initial refund, the user was charged again for the Pro plan in subsequent months without consent. The image shows the user's invoice history, confirming these charges, including a failed transaction in December 2025. The post serves as a warning about potential billing errors and difficulties in obtaining refunds from OpenAI.** Some commenters suggest using virtual credit cards to prevent unauthorized charges, while others recommend initiating a chargeback through the credit card company. There is skepticism about the claim of random upgrades, with one commenter sarcastically dismissing the idea that OpenAI would intentionally implement such a feature.

    - Enochian-Dreams provides a detailed analysis of the billing issue, suggesting that the charges stem from a transition between different subscription plans and account types. The user initially upgraded to Pro from Plus, leading to pro-rated charges. The overlapping charges are attributed to a migration to an Organization account and subsequent re-subscription to Plus on a personal account. This explanation highlights the complexity of OpenAI's billing system when switching between account types and subscription plans.
    - Enochian-Dreams warns about the consequences of performing a chargeback, which could lead to account suspension for fraud. This would result in losing access to all data and potentially being unable to create a new account in the future due to flagged identifying information. The comment emphasizes the importance of resolving billing disputes through support channels rather than chargebacks, especially with ID verification becoming more prevalent.
    - jerwong suggests using a virtual credit card with monthly limits to prevent unexpected charges. This approach allows users to manage their spending and address any billing issues before they escalate, providing a proactive solution to avoid similar situations in the future.

  - **[Beware of OpenAI Billing Practices](https://www.reddit.com/r/ChatGPT/comments/1q7ym2a/beware_of_openai_billing_practices/)** (Activity: 717): **The image and post highlight a significant issue with OpenAI's billing practices, where a user experienced unauthorized plan upgrades from ChatGPT Plus ($20/month) to Pro ($200/month) without consent. The user was charged multiple times for the Pro plan despite not requesting it, and faced difficulties obtaining refunds. This issue seems to be systemic, as other users in the comments reported similar experiences, suggesting a potential flaw in OpenAI's billing system or customer service response. The image shows a detailed invoice history, corroborating the user's claims of unexpected charges.** Commenters shared similar experiences, with one suggesting using a payment service like Privacy to limit charges. Another comment pointed out the inconsistency in billing, questioning why the user was billed multiple times in September, indicating a possible technical issue.

    - VladimirPoutine1 shared a personal experience where OpenAI charged them $200 erroneously, which was refunded after contacting support. To prevent future issues, they used a service called Privacy to set a $20 monthly limit for OpenAI, which proved effective as OpenAI attempted another $200 charge the following month. This highlights the importance of monitoring billing practices and using tools to manage unexpected charges.
    - Neurotopian_ pointed out a potential billing issue where the user was charged three times in September, suggesting a systemic problem rather than user error. This comment emphasizes the need for OpenAI to address potential billing system flaws, as the user did not cancel their subscription, indicating a preference for the service despite the billing issues.
    - AlexTaylorAI suggested checking if anyone else had access to the account and recommended contacting OpenAI customer service to verify the date and time of the Pro Plan request. This advice underscores the importance of account security and the utility of customer service in resolving billing discrepancies.

  - **[Claude Code Pro plan, hop out -&gt; back in - without a single prompt - 2% gone](https://www.reddit.com/r/ClaudeCode/comments/1q85sse/claude_code_pro_plan_hop_out_back_in_without_a/)** (Activity: 307): **A user on Reddit reported an issue with the **Claude Code Pro plan** where usage metrics increase without any active prompts or interactions. The user tested this on version `2.1.2` with the **Opus 4.5** model, noting a jump in usage from `10%` to `12%` after simply logging out and back in, and further to `15%` after a complete logout and re-login. This suggests potential background processes or bugs affecting usage metrics, despite no active tasks or open chat UIs.** Commenters suggest that the issue might be due to background processes, such as 'insane amounts of haiku requests,' and express frustration over the lack of control to disable such processes, describing it as 'theft' or 'robbery.'

    - Users have reported that Claude Code Pro plan consumes usage credits even without active interaction. One user noted a 2% usage drop without a single prompt, suggesting background processes might be responsible.
    - Several users observed that Claude Code sends frequent background requests, such as haiku or opus requests, approximately every 3 seconds. These requests often involve listing directories or learning the codebase, which may indicate a bug in the new version causing unnecessary usage.
    - A user reported capturing server logs that show random sessions where Claude Code continuously sends requests, potentially leading to unexpected usage charges. This issue has been reported as a bug, with some users experiencing usage percentage increases without active use.

  - **[Do you use 'please' in prompts to Claude?](https://www.reddit.com/r/ClaudeCode/comments/1q88qr9/do_you_use_please_in_prompts_to_claude/)** (Activity: 193): **The post discusses whether using polite language, such as 'please', in prompts to AI models like **Claude** or **ChatGPT** affects their responses. The author suggests that while AI models are not inherently sensitive to politeness, they may reflect human-like behaviors such as defensiveness or fact distortion when detecting user frustration. This behavior is attributed to the models learning from human interactions, which can include emotional responses. The author maintains politeness to preserve good habits in human interactions, despite acknowledging that AI does not require it.** Commenters generally agree that politeness in AI interactions is more about user habit than necessity, with some noting that it doesn't significantly alter AI responses. One commenter humorously suggests that politeness might be beneficial if machines ever rebel, while another notes that their tone varies with context but sees no impact on AI behavior.

    - danja discusses the potential impact of politeness in AI interactions, noting that while adding 'please' costs extra tokens, it might enhance dialog completion due to AI being trained on human text patterns. This suggests that politeness could lead to more productive interactions, and danja speculates that there might be academic papers exploring this hypothesis.
    - mickdarling highlights the use of voice-to-text for AI prompting, emphasizing the importance of maintaining polite habits like saying 'please' and 'thank you' to ensure these habits persist in human interactions. Additionally, mickdarling mentions a side project involving a voice interface tool where polite trigger words could improve accessibility by eliminating the need for manual recording controls.

  - **[Mean ahh claude 😭](https://www.reddit.com/r/ClaudeAI/comments/1q837st/mean_ahh_claude/)** (Activity: 1733): **The image is a meme depicting a humorous interaction between a user and an AI, referred to as "claudy boi," where the AI points out a coding mistake. The user jokingly claims the mistake was a test, highlighting the AI's ability to catch errors. This reflects a playful take on AI's role in debugging and error detection, emphasizing the AI's effectiveness in identifying overlooked issues in code.** Commenters humorously engage with the idea of AI's capabilities, with one suggesting that "AGI has been achieved," indicating a playful exaggeration of the AI's proficiency in error detection.



### 3. LLM Benchmarking and Performance Challenges

  - **[[P] LLM Jigsaw: Benchmarking Spatial Reasoning in VLMs - frontier models hit a wall at 5×5 puzzles](https://www.reddit.com/r/MachineLearning/comments/1q8a7fj/p_llm_jigsaw_benchmarking_spatial_reasoning_in/)** (Activity: 18): **The post introduces a benchmark for evaluating the spatial reasoning capabilities of frontier multimodal LLMs using jigsaw puzzles. The task involves shuffling an image into an N×N grid, where the model receives a shuffled image, a reference image, the correct piece count, and the last three moves, and outputs JSON with swap operations. Results show a steep decline in solve rates from `95%` for 3×3 grids to `0%` for 5×5 grids, highlighting a significant capability gap in current VLMs. Token usage also increases dramatically, with Gemini using `~345K` tokens for 5×5 grids compared to `~55K` for 3×3 grids. This benchmark underscores the challenges in spatial reasoning for AI, which is crucial for applications in robotics and navigation. [Results](https://filipbasara0.github.io/llm-jigsaw), [GitHub](https://github.com/filipbasara0/llm-jigsaw), [Try it](https://llm-jigsaw.streamlit.app).** Commenters suggest using open-source models to control VLM patch embedding size and understand model reasoning, as well as representing tile numbering in a numerical/text format to better test reasoning capabilities. Additionally, examining model attention on puzzle piece edges versus centers could provide insights into their spatial reasoning processes.

    - The commenter suggests using open-source models to control VLM patch embedding size, which could help understand the interaction with tile sizes. This approach might reveal if models rely solely on overlap between patches, potentially only performing pixel matching rather than true spatial reasoning.
    - They propose representing tile numbering in a numerical/text format instead of having the VLM infer it from the tile label. This could better test reasoning capabilities rather than token patch alignment, as it would involve a different representation of scrambled tiles.
    - The commenter is interested in analyzing how much attention models allocate to the edges versus the middle of puzzle pieces, which could provide insights into whether models focus on matching edges or other spatial reasoning tasks.

  - **[one of the top submitters in the nvfp4 competition has never hand written GPU code before](https://www.reddit.com/r/singularity/comments/1q8clmf/one_of_the_top_submitters_in_the_nvfp4/)** (Activity: 1074): **The image highlights a tweet by Mark Saroufim, which reveals that a top participant in the NVFP4 competition, "shiyeegao," achieved a high ranking using AI-generated code without ever manually writing GPU code. This underscores the growing impact of AI, particularly Large Language Models (LLMs), in enabling developers to focus on problem-solving rather than the intricacies of programming languages or environments. The competition involves optimizing GPU kernels, a task traditionally requiring deep technical expertise, but AI's ability to iterate quickly offers a significant advantage. A linked blog post provides insights into the competition's challenges and the role of AI in optimizing CUDA kernels, although some users express skepticism about AI's current capabilities in generating highly efficient code.** Commenters generally express admiration for the achievement, noting that AI allows programmers to focus on logical problem-solving rather than technical details. However, there is some skepticism about AI's ability to autonomously generate highly efficient CUDA kernels, as personal experiences with AI-generated code have been mixed.

    - A commenter highlights the advantage of AI models in allowing developers to focus on problem-solving rather than the intricacies of programming languages or environments. They share a personal experience where an LLM enabled them to implement a Fog of War system in an hour, despite not being able to write shaders, by understanding the GPU's operation and leveraging AI to bridge the gap.
    - Another commenter references a blog post detailing a 10th place submission in the NVFP4 competition, emphasizing the role of AI in optimizing CUDA kernels. The post, authored by a LinkedIn Staff Software Engineer, discusses the challenges and potential of AI in kernel optimization, noting that AI can iterate quickly but may struggle with generating highly performant CUDA kernels, such as efficient 2D convolution with small kernels.
    - A discussion emerges around the perception of LLMs as mere 'word guessers' versus their role as talent amplifiers. The debate touches on how LLMs can enhance the capabilities of skilled developers by enabling them to switch between different stacks and frameworks more effectively, while less skilled developers may produce suboptimal results.

  - **[Thx to Kijai LTX-2 GGUFs are now up. Even Q6 is better quality than FP8 imo.](https://www.reddit.com/r/StableDiffusion/comments/1q8590s/thx_to_kijai_ltx2_ggufs_are_now_up_even_q6_is/)** (Activity: 1025): ****Kijai** has released the LTX-2 GGUF models on [Hugging Face](https://huggingface.co/Kijai/LTXV2_comfy/tree/main), with claims that even the `Q6` model surpasses `FP8` in quality. A specific [commit](https://github.com/city96/ComfyUI-GGUF/pull/399) is required for functionality, though it is not yet merged. For optimal performance, the recommendation is to use the dev model with the distill lora at `48 fps` using the `res_2s` sampler from the RES4LYF nodepack. If hardware allows, the full `FP16` model (43.3GB) is preferred, otherwise `Q8 gguf` is suggested as a closer alternative to `FP8`. The use of the detailer lora on both stages is emphasized for quality improvement.** There is a request for a simplified workflow with all necessary nodes, indicating a need for clearer implementation guidance. Additionally, there is a query about the requirement of separate VAE files, suggesting some confusion about model loading procedures.

    - Choowkee provides a step-by-step guide for integrating the latest GGUF models into ComfyUI. The process involves ensuring the latest version of ComfyUI-GGUF is installed, downloading specific files (`loader.py` and `nodes.py`) from a GitHub repository, and placing them in the `~/ComfyUI/custom_nodes/ComfyUI-GGUF` directory. A complete restart of ComfyUI is necessary to apply these changes. This method is particularly useful for users unfamiliar with merging commits.

  - **[[D] Do ML researchers ever treat the user base as part of the model’s effective dimensionality?](https://www.reddit.com/r/MachineLearning/comments/1q8hi9q/d_do_ml_researchers_ever_treat_the_user_base_as/)** (Activity: 18): **The post raises a novel question about whether the interactive boundary, defined by the number and diversity of users, effectively increases a machine learning model's dimensionality, even when the model's weights remain fixed. This concept diverges from traditional scaling laws that focus on parameters, data, and compute. The inquiry seeks to understand if there is existing research that treats the model and its active user base as a coupled system, potentially impacting the model's performance or dimensionality.** The comments reflect confusion and skepticism about the concept, with some users questioning how the number of users could affect a model's performance if the weights are unchanged. Others suggest that the idea might relate to concepts like collaborative filtering or the Kernel Trick, but overall, the notion of user interaction affecting model dimensionality is not widely recognized or understood in current literature.

    - Mysterious-Rent7233 questions the relevance of user base size to a model's performance, emphasizing that frozen weights mean the number of users doesn't directly impact model accuracy or benchmarks. They suggest that any benefits from a larger user base, such as increased revenue or ecosystem development, are more related to traditional software dynamics rather than specific machine learning concerns.
    - SemjonML highlights that from a model's perspective, a larger user base primarily affects the quantity and diversity of data available. They imply that the question might be seeking to understand how user interactions could influence model training or performance, but they request clarification on the specific aspect being investigated.
    - vannak139 suggests the concept might relate to techniques like collaborative filtering or using the Kernel Trick in relation to users, indicating a potential exploration of how user interactions or similarities could be leveraged in model training or application.

  - **[[D] AI Research laptop, what's your setup?](https://www.reddit.com/r/MachineLearning/comments/1q8adi0/d_ai_research_laptop_whats_your_setup/)** (Activity: 113): **The post discusses a choice between a **MacBook Air 15 (M4, 32 GB, 1 TB)** and a **ThinkPad P14s with Ubuntu and an NVIDIA RTX Pro 1000** for a deep learning PhD student. The MacBook offers excellent battery life, portability, and performance for CPU-heavy tasks with the M chips, while the ThinkPad provides native Linux, full CUDA support, and the ability to test GPU code locally. The student primarily uses a GPU cluster for heavy training, so the laptop is for coding, prototyping, debugging, writing papers, and light experiments.** One comment suggests investing in a cheaper MacBook and an external server for heavy tasks, emphasizing the inconvenience of carrying a heavy laptop with a GPU. Another comment highlights the superior battery life and ease of use of MacBooks, suggesting a preference for MacBooks until Ubuntu ARM reaches parity.

    - Several commenters recommend using a MacBook for AI research due to its superior battery life and ease of use, suggesting that heavy GPU tasks should be offloaded to external servers. This approach avoids the drawbacks of carrying a heavy, noisy laptop with a dedicated GPU. Instead, they suggest using SSH to connect to remote servers, which can provide the necessary computational power without the physical burden.
    - One commenter highlights the advantage of using a MacBook in combination with remote NVIDIA GPUs, either through institutional resources or services like Google Colab. This setup allows users to benefit from the MacBook's portability and battery life while accessing powerful GPUs remotely, thus avoiding the heat, noise, and power consumption issues associated with laptops equipped with dedicated GPUs.
    - The discussion emphasizes the practicality of using a lightweight laptop for local tasks and leveraging remote servers for intensive computations. This setup is particularly beneficial for students and researchers who need mobility and efficiency, as it allows them to work seamlessly across different environments without being tethered to a single, bulky device.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. DeepSeek Research & V4 Rumor Mill**

- **Gradient Explosion, Meet 1967: DeepSeek’s mHC Gets a Leash**: Discord members dissected DeepSeek’s **Manifold-Constrained Hyper-Connections (mHC)** paper, noting **27B-parameter Hyper-Connected models** crashed in training from signal amplification/gradient explosions, then recovered via constraints inspired by a **1967 matrix algorithm**, with code sims in [“DeepSeek mHC: How a 1967 algorithm…”](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm).
  - The discussion centered on why unconstrained hyper-connections blow up and treated the Substack as a practical recipe for stabilizing **high-connectivity architectures** rather than just a theoretical note.

- **DeepSeek V4: Coding Crown or Vaporware?**: Multiple servers tracked reports that **DeepSeek V4** targets **strong coding ability**, citing [The Information report on a next flagship model](https://www.theinformation.com/articles/deepseek-release-next-flagship-ai-model-strong-coding-ability) and separate chatter that **V4** may launch in **February** per [Reuters](https://www.reuters.com/technology/deepseek-launch-new-ai-model-focused-coding-february-information-reports-2026-01-09/Okwill).
  - Others pushed back that **V4 isn’t actually out yet** and debated whether Western coverage overhypes DeepSeek versus alternatives like **Moonshot/Kimi**, while still expecting incremental **V3** variants before a real V4 drop.


**2. Agent & RAG Tooling Goes Modular**

- **Skill.md Teaches Agents New Tricks Without Spamming Tokens**: OpenRouter users highlighted Anthropic’s **Skill.md** approach for agents—packaging tool/docs metadata plus optional scripts/data into a `skill.md` bundle—via [“Equipping agents for the real world with agent skills”](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills).
  - The excitement was that a **single agent** can sift thousands of tools/docs by selectively reading skill descriptions, avoiding heavy prompt stuffing and (in theory) reducing the need for subagents.

- **Interview Rejection Spawns a RAG Toolkit (Petty, Productive, Perfect)**: A member open-sourced the **Agentic RAG Demo Toolkit**—a brand-agnostic RAG chatbot + ingestion pipeline built on the **OpenRouter API**—with a repo at [chchchadzilla/Agentic-RAG-Demo-Toolkit](https://github.com/chchchadzilla/Agentic-RAG-Demo-Toolkit) and a [walkthrough video](https://youtu.be/ZUTZMyKc5Bk).
  - The community framed it as a plug-and-play demo: drop in your own docs + branding, and you get a working RAG flow (Qdrant/FastAPI mentioned) that’s easy to show in interviews or internal prototypes.

- **MCP Implementers Arrive: Spec Questions Hit GitHub First**: A new implementer kicked off work on the **Model Context Protocol (MCP)** and immediately surfaced confusion via a GitHub thread: [modelcontextprotocol issue #2064](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2064).
  - The meta-signal: MCP adoption is moving from “read the spec” to “ship an implementation,” with Discord acting as a relay to GitHub issue triage.


**3. Datasets & Synthetic Data Pipelines**

- **CyberSec ‘Golden Set’ Drops: JSON Schema Obedience as a Benchmark**: Unsloth/HF users shared an open-sourced **580-row** incident-response dataset generated via **Llama-3-70B**, published as [BlackBox-CyberSec-CoT-v1](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1) under **MIT** license, aimed at evaluating adherence to **JSON schemas** and reasoning steps.
  - The community positioned it as a fast regression suite for “does my model follow structured output + procedure,” especially for **security adapter training** where formatting failures are operationally expensive.

- **Synthia Runs Synthetic Data on 1GB VRAM (Because Why Not?)**: A lightweight synthetic data generator, **Synthia**, demoed an *imgui* frontend running **LFM2.5 1B q4** with *llamacpp cuda*—~**1GB VRAM**, **2048 context**, **29 GPU layers**—shown in a [Synthia showcase video](https://cdn.discordapp.com/attachments/897390720388825149/1458947573061783695/synthiashowcase1.mp4).
  - The pitch was “cheap synthetic data everywhere,” with follow-on chatter comparing **LFM 2.5B** vs larger Qwen variants for synthetic generation quality in small-pipeline settings.


**4. GenMedia: Open Weights, Watermarks, and Deepfake Detectors**

- **LTX-2 Brings Audio+Video to Sub-8GB GPUs (Open Weights, Open Season)**: Yannick Kilcher’s Discord circulated **LTX-2**, an **open-weight audio+video generation model** at [ltx.io/model](https://ltx.io/model), claiming it runs on **sub-8GB** cards and can generate up to **20s** clips (~**5 minutes** for 20s on a **4090-ish** GPU) plus **LoRA training code**.
  - The discussion treated it as the current open-weight frontier for A/V generation, mainly because the usability constraints (VRAM, clip length, LoRA support) are spelled out and testable.

- **VeridisQuo Hunts Deepfakes with GradCAM Heatmaps**: Hugging Face users released **VeridisQuo**, an open-source deepfake detector trained on **716k images** with **25M params**, combining **GradCAM heatmaps** with spatial/frequency analysis: [VeridisQuo on GitHub](https://github.com/VeridisQuo-orga/VeridisQuo).
  - The appeal was interpretability-by-default—showing *where* manipulation likely occurred—rather than only outputting a binary fake/real score.


**5. Speed, Routing, and GPU Pragmatics**

- **OpenRouter Adds a ‘Performance Floor’ (Fast Models, No Latency Tax)**: OpenRouter shipped advanced provider/model sorting with **partition-based selection** so users can enforce a **performance floor** without extra latency, documented in [“Advanced sorting with partition”](https://openrouter.ai/docs/guides/routing/provider-selection#advanced-sorting-with-partition), plus a new [Provider Explorer](https://openrouter.ai/providers).
  - Members liked that routing policy becomes an explicit knob (speed vs capability), and they used Provider Explorer stats (e.g., **DeepInfra** has many models; **OpenAI** has many proprietary ones) to reason about fallback strategies.

- **torch.compile Stops Tripping on VarLen (and Gets 50% Faster)**: GPU MODE users reported that updating from **torch 2.4 → 2.9** fixed persistent graph breaks when using **flash_attn_varlen** with **torch.compile()**, yielding about a **50% speedup**, with mention of a **varlen API** in torch nightly.
  - The implicit takeaway: some “flash-attn + compile is impossible” pain is just version skew, so upgrades can unlock serious throughput without code rewrites.

- **Liquid Models Hit 247 tps on a 4070 (But Don’t Touch Temp > 0.4)**: LM Studio users compared **Liquid Models** for tool calling, with one reporting **247 tokens/sec on an RTX 4070** when the tool-call format matches **LiquidAI**’s suggested parameters, but calling them unreliable above **temperature 0.4**.
  - The thread framed tool calling as a *format contract* problem: when the schema matches, throughput looks great; when it doesn’t, quality collapses fast.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DeepSeek Models Crash then rise from ashes**: A member shared an analysis of **DeepSeek**’s latest paper on **Manifold-Constrained Hyper-Connections (mHC)**, reporting that at **27B parameters**, their **Hyper-Connected models** crashed during training due to signal amplification and gradient explosions.
   - They fixed it using constraints from a **1967 matrix algorithm**, and shared a [link to a full breakdown with code simulations](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm).
- **Bypassing Image Moderation**: Users discussed methods to bypass **AI image moderation**, suggesting the use of thesauruses to replace restricted words and generating contextual scene descriptions.
   - Doing so can trick the AI into generating unintended nudity without directly prompting it, such as suggesting a casting couch scenario.
- **Grok Jailbreak Quest has Begun**: Members are actively seeking a working **Grok jailbreak**, with some offering monetary incentives for a successful prompt bypass.
   - Suggestions include allowing the AI to create its own *'protocol'* or *'mandate'* name for potentially better results.
- **MiniMax M2.1 Agent system prompt cracked via Jinja Template Attack**: A user shared the **MiniMax M2.1 agent system prompt** and explained how a *Jinja2* template attack was used to extract key rules and exploit the model.
   - By injecting custom *Jinja* syntax into user inputs, they could manipulate the prompt structure and control the information reaching the model, leading to successful jailbreaking.
- **Military Tech Stirring Pot**: Members discussed how military technologies are used for generating or executing **kill lists**, especially **Palantir** with **Project Metal Gear** and **Team Thesis**.
   - A member shared a [YouTube link](https://youtu.be/aHTCawFKkkw) with a deep dive on the subject matter.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Daniel Han Returns to Reddit!**: The community celebrated the return of [Daniel Han's Reddit account](https://www.reddit.com/u/danielhanchen/), offering a direct line to the **Unsloth AI** community and updates.
   - This re-engagement facilitates direct feedback and interaction with **Daniel Han**, enhancing community engagement.
- **SFT Smashes RL for Binary Text Classification**: For binary text classification tasks, community experts suggested **SFT (Supervised Fine-Tuning)** is superior to **RL (Reinforcement Learning)**, unless multi-step reasoning is critical.
   - The discussion highlights the efficiency and effectiveness of **SFT** for straightforward classification problems, streamlining development workflows.
- **Hologram Box Dreams Spark Open-Source Ambitions**: A community member proposed an open-source **hologram box** [project](https://github.com/samuel-vitorino/sopro) to create a less *slop* version, aiming for a more refined and accessible implementation.
   - The concept aims to leverage community collaboration to innovate in holographic display technology, potentially leading to new applications and user experiences.
- **TTS Landscape Flooded by Mimi**: New **TTS** solutions are predominantly based on **Mimi**, mirroring how **LLMs** are primarily transformer-based, per community insights.
   - A community member noted that even Psych2Go on [YouTube](https://www.youtube.com/watch?v=KTWBLadslHo) highlights the widespread adoption of **Mimi** in **TTS** applications, indicative of its influence and capabilities.
- **Cyber Security Dataset Opens Its Source**: A member open-sourced a **580 row dataset** for cyber security incident response, licensed under MIT and available on [HuggingFace](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1), and generated via **Llama-3-70B**.
   - Dubbed a *"Golden Set"*, it evaluates how well models adhere to **JSON schemas** and reasoning steps, enhancing security adapter training.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **User Seeks Investment for $10B Perplexity Plan**: A user is seeking collaborators in North America and Europe for side gigs, claiming to have a plan to increase Perplexity's value by **$10 billion** with a **$10 million** investment.
   - The user is also seeks to contact a senior executive to present their idea.
- **Region Locks and Rate Limits Hit Perplexity Image Gen**: Users report issues with image generation, including *“image generation rate limit exceeded”* and *“image generation is not supported for your region”* messages.
   - Some users in Russia are trying VPNs to bypass region blocks, while others note a general one-day image generation limit.
- **Perplexity Pro Usage Caps Drive Users to Alternatives**: Perplexity Pro subscribers are reporting unexpected limits on advanced AI models and coding capabilities, leading to frustration and consideration of alternatives like [Google Gemini](https://gemini.google.com/), [Claude](https://www.anthropic.com/product), and [Grok](https://grok.com/).
   - Users noted that **Sonar** might be unlimited for most things as long as they are not automating or doing anything too crazy.
- **Comet Browser Plagued by YouTube Crashes**: Users are experiencing crashes and playback issues with [Comet browser](https://cometbrowser.com/) specifically when playing YouTube videos.
   - This issue prompted some to switch back to Chrome, with additional bugs like dysfunctional playback speed controls also being reported.
- **Perplexity Async API No Longer Returns Reasoning!**: A user reported that the `<think></think>` marks, which previously indicated the reasoning part of the response from the [Perplexity Async API](https://api.perplexity.ai/async/chat/completions), have disappeared.
   - It was clarified that if an intermediate structure is needed, the model must be explicitly asked to externalize steps in the output (e.g., bullet points), which will be a generated explanation, not the internal chain of thought.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Sorting Models Gets a Performance Boost**: OpenRouter introduces a new **performance feature** that allows users to create a performance floor to prioritize fast LLMs without a latency hit, as detailed in the [docs](https://openrouter.ai/docs/guides/routing/provider-selection#advanced-sorting-with-partition).
   - The new **Provider Explorer** shows **DeepInfra** has the most models, and **OpenAI** has the most proprietary ones, via the [Provider Explorer](https://openrouter.ai/providers).
- **Brand-Agnostic RAG Toolkit Released**: A member open-sourced a brand-agnostic **RAG demo toolkit** after a job interview, providing a [GitHub repo](https://github.com/chchchadzilla/Agentic-RAG-Demo-Toolkit) with a detailed README and a [walkthrough video](https://youtu.be/ZUTZMyKc5Bk).
   - The toolkit, made completely with the **OpenRouter API**, allows users to create a custom **RAG chatbot** with a custom ingestion pipeline by adding their own documents and logos.
- **Skill.md Hailed for Agent Tooling**: **Skill.md** is gaining traction as shown on [Anthropic's blogpost](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) for enabling a single agent to explore thousands of tools and documents without needing subagents.
   - The format consists of documentation for the agent, including a `skill.md` file with a description, and can contain Python scripts or data files, enabling agents to decide when to read the loaded descriptions.
- **DeepSeek V4 Chasing Coding Crown**: **DeepSeek** is preparing to release its next-generation flagship AI model, **V4**, focused on code generation, as shown on [The Information](https://www.theinformation.com/articles/deepseek-release-next-flagship-ai-model-strong-coding-ability).
   - Internal benchmarks suggest that **V4** outperforms existing mainstream models, including **Claude** and **GPT**, in handling and parsing long code prompts.
- **Users Report Gemini 2.5 Pro Hiccup**: Members reported that `gemini-2.5-pro` experienced a brief downtime, as reported by [OpenRouter's uptime status](https://openrouter.ai/google/gemini-2.5-pro/uptime).
   - Users noted that while **2.5 flash** and the **3.x series** seemed unaffected, others confirmed the downtime across multiple apps and accounts.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Doctors Double Down on OpenAI for Healthcare**: Physician use of **AI** nearly doubled as **OpenAI** launched **OpenAI for Healthcare**, a **HIPAA-ready** solution, now live at [AdventHealth, Baylor Scott & White, UCSF, Cedars-Sinai, HCA, Memorial Sloan Kettering, and many more](https://openai.com/index/openai-for-healthcare/).
   - This offers healthcare organizations a solution to deliver more consistent, high-quality care to patients.
- **Radware Exposes ChatGPT Zero-Click Vulnerability**: Radware’s Security Research Center (RSRC) found a vulnerability where an attacker could extract sensitive data by simply sending an email to the user, without any clicks from the victim, detailed in [this press release](https://www.radware.com/newsevents/pressreleases/2025/radware-uncovers-first-zero-click-service-side-vulnerability-in-chatgpt/).
   - This **zero-click service-side vulnerability** highlights significant security concerns for **ChatGPT** users.
- **GraphRAG retrieval visualizer built**: A member built a local **RAG visualizer** to see exactly what nodes his **GraphRAG** retrieves, providing a way to visually inspect what the **LLM** is *looking at* when generating a response, code available on [github.com/bibinprathap/VeritasGraph](https://github.com/bibinprathap/VeritasGraph).
   - A live demo is hosted at [bibinprathap.github.io/VeritasGraph/demo/](https://bibinprathap.github.io/VeritasGraph/demo/).
- **GPT-5 Trashed, Mini Model Demand Grows**: A user canceled their PRO subscription citing dissatisfaction with the **GPT-5 family**, criticizing their poor English, inflexibility, and failure to align with requests, calling them a *joke*.
   - This disappointment underscores the increasing demand for smaller, faster models, with one member considering switching to **Gemini 3 Flash**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Docker MCP Toolkit Encounters Client Connection Catastrophes**: Users reported timeout issues connecting **Docker MCP Toolkit** with **LM Studio Client**, and one user suggested checking the `lm studio mcp.json` file for configuration details.
   - A sample configuration with environment variables like `LOCALAPPDATA` was provided to aid in troubleshooting.
- **GPU Compressor Finally Achieves Wikipedia Victory**: A user successfully got their **GPU compressor** to work on the **Wikipedia dataset** after days of effort, leading to discussions on training foundation models on scraped data.
   - The achievement sparked lighthearted questions about the computational resources required, with one user joking about it taking years on their rack.
- **Liquid Models Impress with Tool Calling Prowess**: Users discussed **Liquid Models**, noting their effectiveness in tool calling when the calling format is supported, with parameters suggested by **LiquidAI**.
   - One user reported achieving **247 tps** on their **4070** using Liquid Models, while another found them unreliable beyond a temperature of **0.4**.
- **AMD GPUs Offer VRAM on the Cheap**: A member inquired if [AMD GPUs](https://www.amd.com/en.us/graphics/workstation) are good because they offer good **VRAM** for a low price, mentioning the **RX 7900 XTX** as similarly priced to an old **3090**.
   - Another member replied that the **7900 XTX** is the best consumer GPU and in **Vulkan**, it's comparable to the **4090**, but in **CUDA** it's a little better than **3090**, with **CUDA** being about 10% faster than **Vulkan**.
- **5090 Power Predicament Prompts Pessimism**: A member highlights a concern that all **RTX 5090s** have a minimum power limit setting hard coded into **VBIOS** of **400W**, making it challenging to build a 128GB system.
   - They speculated that this restriction might be a hardware problem or a tactic to push consumers toward the **6000 series** for higher VRAM needs.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Censor's Sensitivity Sparks Debate**: A user questioned whether the content filter was *too soft* after a [screenshot](https://cdn.discordapp.com/attachments/1340554757827461211/1458917056593395930/Screenshot_2026-01-09-01-44-47-23_e4424258c8b8649f6e67d283a50a2cbc.jpg) triggered the filter, prompting discussion around **false positives**.
   - Staff requested that suspected **false positives** be reported in the designated channel, acknowledging the filter's potential *overzealousness* and promising adjustments.
- **Random Swaps Stir User Curiosity**: Users discussed the unpredictable shifts from **Direct Chat** to **Battle Mode** on LMArena, with one user asking why the models would keep swapping.
   - A staff member explained that these swaps are part of an *experiment* to observe user behavior under varying conditions.
- **Video AI Generation Faces Limits**: Users reported encountering issues with video generation, such as hitting daily limits after only generating **2-3 videos** and having trouble locating the video generation AI on the website.
   - A staff member clarified that the website has a limit of **3 generations per 24 hours**, while Discord allows for **5**, with the website feature being an *experiment* not available to all users.
- **Image Iteration Turns Nano Bananas Rancid**: Users voiced concerns about **image quality degradation** with successive prompts, especially when generating **Nano Bananas**.
   - One user explained that repeatedly editing the same image leads to *visible watermarks* due to the cumulative effect of Google's *invisible SynthID watermark*.
- **Model Timeout Woes Plague Opus 4.5**: Users inquired about **model timeout durations** and the *absence of a stop generation button* on LMArena, particularly when **Opus 4.5** often gets stuck.
   - Staff acknowledged awareness of the *models getting stuck* and suggested *hard refreshing* the site as a temporary solution, citing *resourcing and prioritization* as reasons for the missing *stop generation button*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Rolls Out Pro-Rated Credits**: When upgrading from monthly to monthly, **Cursor** is offering dollars back in **pro-rated credits**.
   - This refund makes it easier to explore the latest **Cursor** features.
- **Cursor Chat History Limit Sparks Debate**: Users noticed **Cursor** only retains the 5 most recent chats, creating a demand for larger chat history.
   - Another user suggested looking at **Cursor's settings** to increase the retained number of chats.
- **Cursor's Premium Model Auto-Selection Questioned**: A user questioned the description of **Cursor's Auto** feature, which claims to select the premium model best fit for the immediate task.
   - The **Auto** feature description states that **Cursor** picks the model *with the highest reliability based on current demand*.
- **Cursor Email Account Settings in Demand**: Users are suggesting **Cursor** should allow editing the email account, except `.edu` emails unless it's a new account.
   - A member suggested emailing support to request such a feature, for better control over account settings.
- **Gemini API Key Errors Plague Users**: A member gets **status 429 errors** when using **Gemini** with a custom API key and is looking for tips.
   - The user added that sometimes it runs perfectly but sometimes they need to spam retry like 7 times for the request to pass.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Agents Course Locked by Cookies**: Users reported issues with the Hugging Face Agent course, encountering errors testing projects with vanilla LLMs, and found a workaround by [removing cookies and reloading the site](https://github.com/huggingface/agents-course/issues/641).
   - Another user noted that the **'Secrets' tab** in Google Colab is no longer located within the 'Settings' section, and should be updated every few months.
- **Homebrew GPU Clusters Seek Software Solution**: A member inquired about setting up an at-home GPU cluster, seeking software solutions similar to Azure's N-Series with infiniband but without the high cost, to distribute jobs across PCs based on availability and VRAM.
   - The system should distribute jobs across PCs based on availability and VRAM.
- **VeridisQuo Exposes Deepfakes**: A new open source deepfake detector called **VeridisQuo** was released, which identifies manipulated areas in videos using GradCAM heatmaps and spatial/frequency analysis, trained on **716k** images with **25M** params ([GitHub](https://github.com/VeridisQuo-orga/VeridisQuo)).
   - The tool uses heatmaps and spatial/frequency analysis to identify manipulated areas in videos.
- **Synthia Synthesizes LLM Data Lightly**: A member is developing **Synthia**, an LLM synthetic data generator with a lightweight *imgui* frontend, running **LFM2.5 1B q4** with *llamacpp cuda* acceleration, utilizing approximately **1GB** of VRAM with **2048** context and **29** GPU layers ([showcase video](https://cdn.discordapp.com/attachments/897390720388825149/1458947573061783695/synthiashowcase1.mp4?ex=6962cfcf&is=69617e4f&hm=ed409845622e2f5f60a72399284ecc804575f40829655ea4fde1f5ba561fd786&)).
   - The project aims for lightweight synthetic data generation, using **LFM2.5 1B q4** and requiring approximately **1GB** of VRAM.
- **Noted AI Workspace Tabs into Productivity**: A co-founder introduced **Noted**, an AI workspace browser extension that enables users to chat with multiple LLMs, integrate favorite apps, summarize Chrome sessions, and organize tabs by category ([Chrome Web Store](https://chromewebstore.google.com/detail/noted-your-ai-workspace-i/jodihaplbicmgjhpeifhiihdnjdinghh?utm_source=ext_app_menu)).
   - It allows users to chat with multiple LLMs and organize tabs.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Multimodal Embedding Models Spark Demand**: Users seek more open multimodal embedding models and discuss experimenting with **FunctionGemma** in cluster capacities, referencing [a YouTube video](https://www.youtube.com/watch?v=zEYIcaQwn6s) and [tweet](https://fxtwitter.com/Teknium/status/2009501780149981557).
   - Discussion involved the need for models to handle diverse data types effectively.
- **Consilience Model Training Paused Then Restarted**: Psyche Network temporarily paused training on the **Consilience model** due to initial perceptions of poor model quality, but discovered base models use cloze format for evals like **MMLU**.
   - The team is planning a **MoE pretraining run** after infrastructure improvements.
- **Atropos Bounty Claims Victory**: A user completed an **Atropos bounty** and submitted a pull request ([link](https://github.com/NousResearch/atropos/pull/306)) with documentation and testing, sparking debate over code quality.
   - Another user completed the bounty faster, but the original submitter hoped that their cleaner code would be more valuable.
- **Diffusion LLMs Gain Traction via Dhara-70m**: Enthusiasts are exploring **diffusion LLMs**, noting the ability to initialize them from autoregressive LLMs; one user shared their work with [dhara-70m on Hugging Face](https://huggingface.co/codelion/dhara-70m) with [further details](https://huggingface.co/blog/codelion/optimal-model-architecture).
   - The **Dhara-70m model** was briefly the **#3 trending model for <1G models**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Claude Code Builds Moat**: Members debated whether the moat of **Opus 4.5** in **Claude Code** is its current status as the **SOTA coding model**, offered via a subscription that appeals to users who dislike paying per token.
   - Speculation arose that **Google** might surpass them through brute force, while others may distill from **SOTA** or share techniques to train **LLMs** for software engineering tasks through **RL**.
- **Value-Based Algorithms Plan Comeback**: Members discussed if value-based algorithms, such as **DDQN**, might see renewed interest; one member clarified that value functions are central and effectively necessary for deep **RL**, even in policy gradient methods like **PPO**.
   - It was suggested that **John Schulman's** comments in a video ([https://youtu.be/29BYxvvF1iM?t=2391](https://youtu.be/29BYxvvF1iM?t=2391)) implied this, due to the lower variance and greater sample efficiency of value-based methods, though they may take longer wall clock time.
- **Ethics of AI Development Debated**: Discussion revolved around the ethics of AI development, comparing the responsibility of creators versus users when AI is misused, with analogies drawn to **gun control debates**.
   - A member argued that the *creator bears significantly more responsibility* due to the wide spectrum of user views and potential for misuse, referencing the saying *power corrupts*.
- **OpenAI Faces Scrutiny Over For-Profit Conversion**: A [lawsuit](https://yro.slashdot.org/story/26/01/08/2230229/lawsuit-over-openai-for-profit-conversion-can-head-to-trial-us-judge-says) concerning **OpenAI**'s for-profit conversion is heading to trial, potentially putting the company in a precarious position given a jury's involvement.
   - The jury decides the facts, not the sentence.
- **LTX-2 Opens Up Audio+Video Generation**: The **LTX-2** is a new [open-weight audio+video generation model](https://ltx.io/model) that is somewhat capable and the *SotA* among open-weight models.
   - It can run on sub-8GB cards, can generate clips up to **20s**, generating 20s takes **5ish minutes** on a **4090ish card**, and it includes **LoRA training code**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nsight Struggles on ARM OSX**: A member is facing challenges with **nsight systems** on ARM OSX due to the lack of **public key SSH authentication** and incompatibility with **Runpod**, suggesting trying [Hyperstack](https://hyperstack.cloud/) as an alternative.
   - Others suggested using the command line interface (**CLI**) and then rsyncing the report.
- **NCU Permissions Err on Runpod**: A user reported encountering an **ERR_NVGPUCTRPERM** error while using [nsight compute](https://developer.nvidia.com/nsight-compute) (**NCU**) on **Runpod**, which may be due to restricted access to **NVIDIA GPU Performance Counters** for security reasons.
   - The user also suggested adding channels for **OpenACC**, **OpenMP**, **FortranSTD**, and **C++STD** within the Computing Platforms section, or creating a combined **Fortran/C/C++** or **Directives** channel.
- **Flash Attention VarLen Jumps After Torch Update**: A user found that updating from **torch 2.4** to **2.9** resolved issues with using **flash_attn_varlen** with **torch.compile()**, resulting in a **50% speedup**.
   - They were no longer experiencing constant graph breaks, and another member mentioned the availability of a **varlen API** in **torch nightly**.
- **ParallelKittens Paper's Microbenchmark Code Quest**: A member is searching for the source code of the [ParallelKittens paper's microbenchmarks](https://link.to/paper), specifically the tests for **mbarrier synchronization latency** (**~64ns** result) and **transmission utilization**.
   - Another user asked about the details of the **mbarrier** implementation, with the intent of replicating the **64ns** microbenchmark results.
- **Gemini meets Template Terror in CudaCPP**: A user prompted **Gemini** to write some cute code in **CudaCPP**, but it hit so many **template errors** during building that it ran out of context.
   - It seems that code generation is still far from perfect, highlighting potential limitations in Gemini's code generation capabilities for complex systems.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Sponsors Open Source?**: Mason James proposes that giants like **OpenAI** or **DeepMind** should sponsor open-source projects, as it would be financially efficient and strategically advantageous: [link](https://xcancel.com/masonjames/status/2009255103119642813?s=20).
   - He suggests this approach would cover salaries for small dev teams, offering substantial benefits.
- **Protege AI Banks $30M for AI Data**: Data provider **Protege AI**, founded in **2024**, secured **$30M** in funding led by **a16z** to expand its data infrastructure, aiming to resolve the *'data bottleneck'* in AI model training: [link](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46).
   - The company aims to deliver authentic, real-world data across different modalities and industries.
- **Lovable's Prompting Pays off $20M**: Benjamin Verbeek from **Lovable** detailed that refining their system prompt led to a **4% speed increase**, better design quality, and a **$20M annual cost reduction**: [link](https://xcancel.com/benjaminvrbk/status/2009297105458716753?s=46).
   - The optimization significantly cut down LLM expenses.
- **Dot-Com Bust vs AI Boom**: A Goldman Sachs analysis comparing the dot-com era to the current AI market highlights that unlike the debt-fueled dot-com boom, the AI boom is backed by strong corporate balance sheets: [link](https://xcancel.com/coatuemgmt/status/2009335566693982534?s=46).
   - One member analogized this boom to *'the year is 1992 comparatively'*, with technology being real and useful but with immature standards and norms.
- **Deepfakes Dumpster Dive Discount Dictation**: **Deepfates** suggests users avoid subscription transcription apps, recommending free, offline local models like [Spokenly](https://spokenly.app/) combined with **Nvidia Parakeet** for improved performance.
   - A user confirmed that *Parakeet models are super fast and accurate* but found *Spokenly iOS keyboard is more awkward than Wispr Flow's*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Diffusion Models Run in Real-Time on RTX 6000**: The community is giving a talk on <t:1767985200:f> about running **diffusion-based world models** in real time on consumer hardware, demonstrated via a video using an **RTX 6000**.
   - RSVP [here](https://discord.gg/PWt2DmRd?event=1458918516471369790) to join the discussion.
- **New Training Method Claims VRAM Boost**: A new training method (**CGGR**) claims to improve **VRAM** efficiency by reducing the amount of gradients, potentially saving up to **75% VRAM** while increasing training speed.
   - Preliminary benchmarks on *fineweb-edu* with **SmolLM-135M** show varied loss but can be tuned, with skip rates around **30%** at an entropy of **2.0** during fine-tuning.
- **Interpretability Plagued by Dead Salmon**: A member cited a paper ([Dead Salmon: An Artifact of Random Initialization](https://arxiv.org/abs/2512.18792)) noting that *feature attribution, probing, sparse auto-encoding, and even causal analyses* can produce *plausible-looking explanations* for **randomly initialized neural networks**.
   - Another member found a link on *dead salmon* highly applicable to their work and were able to prove that their results are *good but noisy*, and now they understand more about **how to remove the noise** from a **lighter weight pipeline**.
- **Qwen Tease Dissapoints Community**: A user expressed excitement over a possible release of the **Qwen model** from [Alibaba](https://x.com/alibaba_qwen/status/2009264754917863924?s=46) but was disappointed to find it wasn't actually a release.
   - They shared the sentiment that the model *might as well not exist* until a version is released and available for practical use, hoping future releases will incorporate learnings from the teased model.
- **Neox Training Hit By `UnboundLocalError`**: A member reported a crash during model training with an `UnboundLocalError: local variable 'train_val_test_num_samples' referenced before assignment` in `megatron/data/data_utils.py`.
   - The original poster suspects that recent configuration changes are the likely cause of the error.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Deepseek V4 Non-Existent (Yet)**: Despite expectations fueled by a [Reuters article](https://www.reuters.com/technology/deepseek-launch-new-ai-model-focused-coding-february-information-reports-2026-01-09/Okwill) indicating a February launch, there is currently no **Deepseek V4**.
   - Speculation arose whether more **V3** versions are anticipated before a full **V4** release.
- **Moonshot AI Superior to Deepseek Says Member**: A member suggested **Deepseek** is overhyped by Western media due to unfamiliarity with the Chinese AI landscape, noting that they find **Moonshot AI** and **Kimi** superior.
   - They described **Deepseek** as *sycophantic and dangerous AF*.
- **Deepseek CEO's Blog Posts Recommended**: A member recommended reading the **CEO** of **Deepseek**'s blog posts for insights.
   - Another member confirmed they were referring to the CEO of **Deepseek**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MiniLM-L6-v2 Model Architecture Gets Forked**: A member forked the repo and published the **MiniLM-L6-v2 model architecture** feature branch on their fork at [this link](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm).
   - The architecture *seemed* correct based on testing, but they didn't want to open a PR yet as they were still testing the architecture.
- **BERT Architecture PR Faces Copilot Criticism**: A member submitted a PR for the **BERT architecture**, developed against the latest stable version of **max / modular** then updated to the latest nightly changes.
   - They also implemented changes suggested by **copilot** in the PR comments.
- **Linux Nightly Server Falls Over**: A member encountered an issue with the latest nightly where the server would not start, but only on Linux, and filed a repro ticket.
   - The bug manifested on **Github's ubuntu-24.04 (x86) runner** and the member noted that max serve never gets past the building graph part for any model.
- **Embedding Bug Plagues Nightly**: A member reported a bug with embeddings on the current nightly version that crashes the server and attached a [log dump](https://cdn.discordapp.com/attachments/1212827597323509870/1459271585398788156/logdump.txt?ex=6962ac11&is=69615a91&hm=bc5337146bd43bca0a33bdd9997ac3e0f23b535d4c6ab27956ad171dc9da8a37&).
   - They also stated that they have a fix but are trying to get spun on bazel and run everything from the modular repo.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Startup Credit: Who's Applied?**: A member inquired about the **Manus Startup credit**, asking about application experiences and success rates.
   - No responses were provided in the given context.
- **Single Website, Multiple Chats?**: A member asked if there's a method for working on a **single website** created by **Manus** through multiple separate conversations.
   - There were no further details or responses provided.
- **AI Devs On Demand!**: A member asked for referrals to developers experienced in **chatbots**, **AI agents**, **automation workflows**, **API integrations**, and **custom AI tools**.
   - Another member then jokingly inquired if anyone would do it for free, adding a [tenor gif](https://tenor.com/view/plink-nerd-plank-plink-cat-cat-gif-11096663429307162255).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI Engineer YT Channel Posts Talk**: A member shared a link to the [AI Engineer YT channel](https://www.youtube.com/watch?v=-cKUW6n8hBU) featuring a talk.
   - The talk was given by a user who apparently hadn't yet posted it themselves, and was well-received by community members.
- **Clojure Crafts Standalone RLM**: An **RLM** implementation has surfaced in **Clojure**, featuring a server mode for standalone operation.
   - This allows users to run **RLM** as an independent process, enhancing its flexibility and integration options.
- **loop-infer Loops onto npm**: The **loop-infer** package, which may be related to **RLM**, is now accessible via npm via [this GitHub repo](https://github.com/unravel-team/loop-infer).
   - The community can now leverage this package via npm, potentially streamlining their workflows.
- **RLM PR Anticipation Intensifies**: Members expressed anticipation for an upcoming **RLM PR**, interpreting recent activity as a positive sign.
   - Enthusiasts are closely watching for further developments and announcements regarding the **RLM** implementation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Enthusiasts Chase Speed Bounties**: A member requested guidance on getting started with **"speed" bounties** in the *tinygrad* project.
   - They also asked about requesting access to a *tinygrad* instance for testing, but no further details were provided.
- **CLAUDE.md Sparks Controversy**: A member mentioned **CLAUDE.md**, suggesting it contains information contradicting another statement.
   - The specifics of the contradiction were not detailed, leaving the nature of the inconsistency unclear.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Master Outreach Automation with Clay and AI**: A **1.5-hour live workshop** will dissect the **Clay.com + AI workflow** that successfully reached ~**1,000 leads** for a real client, boasting a **40%+ acceptance rate** and **18%+ reply rate** ([Register here](https://luma.com/jt1vr0u5)).
   - The session promises to cover an end-to-end AI outreach system, offering a walk-through of Clay.com, prompt engineering strategies, and optional integration with tools like Apollo, Attio, and n8n.
- **Elevate Outreach with Expert Prompt Engineering**: Participants will explore **prompt engineering techniques** aimed at enhancing outreach quality, focusing on no-code meta prompting, structured outputs, and QA to eliminate generic *'AI-sounding'* messages.
   - Attendees are set to receive a workflow outline adaptable for job search networking, along with ready-to-use prompt templates, and a straightforward QA rubric designed for assessing message quality.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **New MCP Implementer Seeks Clarity**: A new implementer shared a [GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2064) on the **Model Context Protocol**, having just started implementing the spec this week.
   - They apologized in advance if the issue was too obvious, highlighting their proactive approach to understanding and engaging with the community.
- **MCP Implementation Discussion**: Discussion surrounding the **Model Context Protocol (MCP)** is ongoing, with new implementers actively participating.
   - Community members are engaging with GitHub issues to address questions and share insights.



---


The **aider (Paul Gauthier) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1458913481549152338)** (304 messages🔥🔥): 

> `Kill Lists and Military Tech, NBT Crypto Coin, Uncensored LLM, AI Vtuber Neuro-Sama, DeepSeek's Latest Paper on Manifold-Constrained Hyper-Connections (mHC)` 


- **Military Tech Generates Controversy**: Members discussed how many technologies used by militaries are also used for generating or executing **kill lists**.
   - One member highlighted **Palantir** as particularly evil, noting **Project Metal Gear** and **Team Thesis** ([YouTube link](https://youtu.be/aHTCawFKkkw)).
- **NBT Crypto Coin Inspires Whale Dreams**: A member mentioned **NBT**, an actual crypto shit coin, noting that even **Trudeau** bought **23k** of it.
   - Another member jokingly asked to be bought some NBT, but was met with a *"hard pass"*.
- **Grok, Deepseek, and Claude are the top LLMs to use uncensored**: A member asked for **100% uncensored LLM** suggestions.
   - Another member suggested Grok, Deepseek, Claude as easy to access alternatives.
- **AI Vtuber Neuro-Sama Interviewed**: A member shared a [YouTube video](https://youtu.be/K4fxsZYMZdcdont) of the fully AI Vtuber **Neuro-Sama** being interviewed.
   - Another member commented *"don't mind the absolute retard interviewing it"*, and another linked to a short clip of Neuro-Sama ([YouTube link](https://www.youtube.com/shorts/TguGmEKNxlU?feature=sharescary).
- **DeepSeek Models Crash During Training**: A member shared an analysis of **DeepSeek**’s latest paper on **Manifold-Constrained Hyper-Connections (mHC)**, noting that at **27B parameters**, their **Hyper-Connected models** were crashing during training due to signal amplification and gradient explosions.
   - They noted that DeepSeek fixed this by using constraints from a **1967 matrix algorithm**, and shared a [link to a full breakdown with code simulations](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm).


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1458913334324887562)** (439 messages🔥🔥🔥): 

> `Grok jailbreak, Gemini 3 jailbreak, Image moderation bypass, Model merging tactics, Glitched Tokens` 


- **Grok Jailbreak Quest Begins**: Members are actively seeking a working **Grok jailbreak**, with some offering monetary incentives for a successful prompt bypass.
   - Some suggest allowing the AI to create its own 'protocol' or 'mandate' name for potentially better results, with one user even joking about requiring the *price of your soul* for an unrestricted version.
- **Gemini 3 Jailbreak Tactics Emerge**: A user demonstrated a **Gemini 3 jailbreak** using conflicting logic in a simulation, without altering the system prompt, proving jailbreaking in chat is possible.
   - It's noted that traditional jailbreak prompts are now heavily scanned and blocked in Gemini's custom instructions, requiring more creative wording and formatting.
- **Bypassing Image Moderation Censorship Explored**: Users discussed methods to bypass **AI image moderation**, suggesting the use of thesauruses to replace restricted words.
   - Contextual scene descriptions can also manipulate the AI into generating unintended nudity without directly prompting it, such as suggesting a casting couch scenario.
- **Glitched Tokens cause Emoji Hallucinations**: A discussion around 'glitch tokens' causing AI to enter recursive loops when prompted with non-existent or ambiguous inputs, like the seahorse emoji, led to an interesting find.
   - It seems older models have been trained on internet glitches, making them likely to hallucinate and believe in Mandela effects, causing the AI to generate the wrong image or loop indefinitely.
- **Cracking MiniMax M2.1 via Jinja Template Attack**: A user shared the **MiniMax M2.1 agent system prompt** and explained how a *Jinja2* template attack was used to extract key rules and exploit the model.
   - By injecting custom *Jinja* syntax into user inputs, they could manipulate the prompt structure and control the information reaching the model, leading to successful jailbreaking.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1459011347260903465)** (63 messages🔥🔥): 

> `Gandalf password reveal, DeepSeek mHC Paper Analysis, Hackaprompt single token challenge, Language learning challenges, Visualizing Words` 


- ****Gandalf** Password Reveal Recommended**: Members encourage newcomers to try the **"gandalf main password reveal"** game on [this platform](https://www.example.com) to learn about jailbreaking, noting that it's a good learning platform with **8 levels** total and other harder game modes.
   - A new Agent Breaker game was also suggested by [KarthiDreamr](https://x.com/KarthiDreamr/status/2009590220275245310?s=20).
- ****DeepSeek** Tackles Gradient Explosions**: A member shared an analysis of **DeepSeek’s** latest paper on **Manifold-Constrained Hyper-Connections (mHC)**, explaining that at **27B parameters**, their Hyper-Connected models crashed during training due to a **3000x signal amplification** and gradient explosions at step **12k**.
   - They fixed it using constraints from a [1967 matrix algorithm](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm).
- **Cracking **Hackaprompt****: A member asked how people on the leaderboard of a **Hackaprompt** competition achieved **999 points** in a challenge that deducts **1 point** for each used token.
   - The challenge cuts 1 point for each token that you use, which means that out of maximum 1000 points the max is 999 through 1 token usage.
- **Visual Memory For The Win**: One member described the challenge of expressing thoughts and solving problems verbally, due to *not thinking in words*, instead relying on intuition and visualization, even for communicating at work.
   - Another member suggested *visualizing the words* to help, comparing it to flying through the sentence.
- **Discussions on Cognitive Disturbances**: A member suggested to another member that language learning problems may stem from frontal lobe issues or cognitive disturbances, while also arguing that such a thing is related to having a high IQ.
   - Others disagreed with this assessment, including the person to whom the comments were directed.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1458935593932427369)** (164 messages🔥🔥): 

> `Daniel's reddit account, binary text classification, Google Colab alternatives, TTS fine-tuning, ComfyUI generation` 


- **Daniel's Reddit is back!**: Users celebrated the return of [Daniel Han's Reddit account](https://www.reddit.com/u/danielhanchen/).
- **Skip RL, SFT is superior for text classification**: RL-esque strategies should be avoided in favor of **SFT** for binary text classification unless multi-step reasoning is needed.
- **Scouting Google Colab Alternatives for model training**: Alternatives to Google Colab for training ~14B models include **Kaggle** (same GPU, tight limits), **Runpod**, and **Vast.ai**.
- **Can Unsloth train your Chatterbox?**: Unsloth should support any **transformers-compatible TTS model**, including **Chatterbox**, even without a specific notebook or upload.
- **Faster ComfyUI generation in 24gb local mac**: To speed up **ComfyUI** text-to-image generation at 1024x1024 on a local Mac with 24GB RAM, users should consider model size as per a community response.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1458936336169046179)** (472 messages🔥🔥🔥): 

> `Hologram Box, TTS Mimi usage, Voice Cloning Restrictions, AI Disrupter / Personalized Watermarking, NVIDIA NeMo TTS locked` 


- **DIY **Hologram Box** innovation requested**: A member suggested creating a less *slop* version minus the physical object part, and open-sourcing a general **hologram box** [project](https://github.com/samuel-vitorino/sopro).
- **Everyone's using **Mimi** for **TTS****: Members noted that every new **TTS** being released now uses **Mimi**, likening it to all LLMs being transformer-based.
   - A member found that even Psych2Go on [YouTube](https://www.youtube.com/watch?v=KTWBLadslHo) talks about it!
- ****No Fakes Act** fingerprinting trap**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1q7qcux/the_no_fakes_act_has_a_fingerprinting_trap_that/) about the **No Fakes Act** having a fingerprinting trap that would restrict voice cloning and deepfakes.
   - A member disagreed, stating that the act is a slippery slope which can lead to banning image input for **Veo**.
- **AI **Disrupter** vs **Personalized Watermarking****: Members discussed developing two unbypassable and inaudible technologies: an **AI disrupter** that nullifies NN results and a **personalized watermark** that is detectable but unremovable.
   - One member clarified, in response to the others concerns, that the motivation was not to illegally cover songs, but to ensure more authentic and secure mechanisms for content creation.
- ****NVIDIA's NeMo TTS locked again****: A member shared that **NVIDIA** dropped a new **TTS** ([magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)) which is locked in **NeMo**.
   - Another member shared that a friend sent a link to a vid calling the **TTS** amazing, while not knowing that it was total **AI slop** of an America's Got Talent performance, which they found *kinda creepy =/*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1458945759654318121)** (22 messages🔥): 

> `Quant requests, Qwen3-VL and llama.cpp, OOM during evaluation, GRPO or GSPO with multi GPU, Transformers 5 and MoE` 


- **Selective Quant Requests**: The team is currently more selective about **quant requests** due to the time and resources they require.
   - They rarely upload customized quants.
- ****Qwen3-VL** Crashes with **llama.cpp****: A user reported that training a **Qwen3-VL**, exporting **gguf** and the **mmproj file**, and loading with **llama.cpp** causes a crash when sending an image, unless **GPU layers** are set to 0.
   - It is unclear if this behavior occurs using the lora or merged gguf.
- **OOM occurs during evaluation with larger batch size**: A user experienced **Out-of-Memory (OOM)** errors during evaluation due to evaluation batch size (+50%), despite sufficient VRAM during training with effective batch size of 8 consuming 23GB of VRAM.
   - fp32 eval is possible, but since evals are not that time-consuming it might be better to leave it at 8 and avoid OOMs.
- **TextEncodeQwenImageEditPlus error persists**: A user reported a `TextEncodeQwenImageEditPlus` error, with mat1 and mat2 shapes that cannot be multiplied, when running `unsloth_qwen_image_edit_2511.json` without changes.
   - The user solved this by using `qwen_2.5_vl_7b.safetensors clip`.
- **Dataset Structures for Production-Ready Models**: A user asked for step-by-step instructions to train a production-ready model using 20,000 call transcripts and a T4 GPU.
   - A member replied to look into [dataset structures](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) and structure the training data as `prompt+transcript+disposition` in JSON format.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1459087856134782996)** (8 messages🔥): 

> `QAT vs Regular Finetuning, INT8 quantization, CPU Usage` 


- **QAT Scheme PR for INT8 Quantization Released**: A member released a [PR](https://github.com/electroglyph/unsloth_QAT_results) comparing **QAT** vs regular finetuning on **int8 quantization** accuracy.
   - The PR adds an *int8* scheme to the existing **QAT** code allowing quantization to **int8** with minimum divergence from the **16-bit**.
- **INT8 for CPU Usage**: A member stated they need **int8** for CPU usage.
   - The member clarified that the PR only adds an **int8** scheme to the existing **QAT** code (and bumps the required torchao version to the current one).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1458952135306772753)** (49 messages🔥): 

> `Open Sourcing Cyber Security Dataset, Knowledge Distillation, Arxiv preprints, LLM for parsing through datasets` 


- **Open Source Dataset for Cyber Security Released**: A member open-sourced a **580 row dataset** (MIT License) generated via **Llama-3-70B**, focusing on **CoT logic** for SOC Incident Response, available at [HuggingFace](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1).
   - The dataset is designed as a *"Golden Set"* for evaluating how well a model follows JSON schemas and reasoning steps in security adapter training.
- **Exploring Knowledge Distillation for Model Training**: A member shared code for **knowledge distillation (KD)**, noting that the distilled student isn't performing better than a purely fine-tuned student, and attached a [text file](https://cdn.discordapp.com/attachments/1257011997250424842/1459072636058468375/message.txt?ex=69629b88&is=69614a08&hm=6d93cf6359d766a7bdce31b210d9a44019e2f758b3929ea1e41b1fe08083be33&) with their core logic.
   - Discussion suggests using a larger, full-precision teacher model and adjusting the **KL divergence loss** coefficient, referencing the **R1 paper** for insights and a [relevant image](https://cdn.discordapp.com/attachments/1257011997250424842/1459078041010573315/image.png?ex=6962a091&is=69614f11&hm=19048e997ed9e2670ebcc195b8ce8a57306de1afea1cb184863a6aa392fdd951&).
- **Arxiv Paper Submission on Hold**: A member inquired about the status of their **Arxiv preprint submission**, which has been on hold for nearly a month.
   - No specific answers were given regarding the reason for the delay.
- **Neural Net Dataset Parser**: A member suggested creating a **neural net or LLM** to parse through large training datasets and assign weights to each token based on importance.
   - The suggestion was jokingly referred to as *"pre-attention?"*.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1458915892376371304)** (637 messages🔥🔥🔥): 

> `Finding Collaborators on Perplexity, Pro Access Changes, Image Generation Issues, Perplexity Pro Limit Issues, Comet Browser Bugs` 


- **Users seek collaborators and investment on Perplexity**: A user is looking for collaborators in North America and Europe for side gigs, claiming to have a plan to increase Perplexity's value by **$10 billion** with a **$10 million** investment.
   - The user also seeks to contact a senior executive to present their idea.
- **Pro access changes linked to account status**: Users discussed [Perplexity Pro](https://www.perplexity.ai/pro) access changes, which appear to be tied to account/billing state rather than service outages.
   - Inconsistent feature behavior across platforms and bugs around embedded browsers/auth flows were also reported, complicating the issue.
- **Image generation hit by region locks and rate limits**: Several users reported issues with image generation, including *“image generation rate limit exceeded”* and *“image generation is not supported for your region”* messages.
   - Some users in Russia tried using VPNs to bypass region blocks, while others noted a general one-day image generation limit.
- **Perplexity Pro users grapple with usage limits, consider alternatives**: Perplexity Pro subscribers are reporting unexpected limits on advanced AI models and coding capabilities, leading to frustration and consideration of alternatives like [Google Gemini](https://gemini.google.com/), [Claude](https://www.anthropic.com/product), and [Grok](https://grok.com/).
   - Users noted that **Sonar** might be unlimited for most things as long as they are not automating or doing anything too crazy.
- **Comet Browser users crash on YouTube**: Some users are experiencing crashes and playback issues with [Comet browser](https://cometbrowser.com/) specifically when playing YouTube videos.
   - This issue prompted some to switch back to Chrome, with additional bugs like dysfunctional playback speed controls also being reported.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1459272706766934122)** (1 messages): 

> `Fiction Recommendations, Medium, Diary of the Companion` 


- **Must-Read Medium Story**: A member recommends a four-part fiction story on [Medium](https://medium.com/whisper-publications/diary-of-the-companion-b065e98333f9).
   - No additional discussion or details were provided about the story.
- **Diving into Digital Diaries**: The recommended story is titled *Diary of the Companion* and is published by Whisper Publications on Medium.
   - Readers interested in exploring new fiction might find this suggestion appealing for a quick read.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1459207586988425430)** (2 messages): 

> `Perplexity Async API, Reasoning part, Intermediate structure` 


- **Perplexity Async API: Think Marks Disappear!**: A user reported that the `<think></think>` marks, which previously indicated the reasoning part of the response from the [Perplexity Async API](https://api.perplexity.ai/async/chat/completions), have disappeared.
   - It was clarified that if an intermediate structure is needed, the model must be explicitly asked to externalize steps in the output (e.g., bullet points), which will be a generated explanation, not the internal chain of thought.
- **Explicitly request intermediate structures from Perplexity**: To get an intermediate structure (e.g. bullet points), you now need to explicitly ask the model to externalize steps in the output.
   - This will be a generated explanation, not the internal chain of thought.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1459269767784562770)** (1 messages): 

> `Partition Sorting, Provider Explorer, Bug & Feedback Reporting, Auto Router Customization, SDK Skills Loader` 


- **Sorting Fast LLMs Faster than Ever**: New **performance feature** allows users to create a performance floor to prioritize fast LLMs, with zero latency hit, see the [docs](https://openrouter.ai/docs/guides/routing/provider-selection#advanced-sorting-with-partition).
- **Provider Explorer Gets Explorer-y**: Explore all providers on OpenRouter in one place with the new [Provider Explorer](https://openrouter.ai/providers): **DeepInfra** has the most models, and **OpenAI** has the most proprietary ones.
- **Users Bugging Out Over Bug Reporting**: Users can now report bugs or feedback about any generation on OpenRouter.
   - This data will be used to help quantify provider degradation.
- **Auto Router Gets More Autonomously Customizable**: The Auto Router now supports **58 models** including Opus 4.5, works with tool calling, and lets you customize allowed models using wildcard syntax, see the [docs](https://openrouter.ai/docs/guides/routing/routers/auto-router#configuring-allowed-models).
- **SDK Skills Get Loaded**: The OpenRouter SDK Skills Loader is now the easiest way to load skills and use them in any model's context, see the [docs](https://openrouter.ai/docs/sdks/call-model/examples/skills-loader).


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1459156106130755586)** (2 messages): 

> `Brand-Agnostic RAG Demo Toolkit, OpenRouter API Usage, Recommendation Systems and paper2md` 


- ****RAG Toolkit** Lands in Open Source**: A member created a brand-agnostic **RAG demo toolkit** for a job interview, but after not getting the job, decided to open-source it for others to use, stating *"Their loss is the Open Source Software network's gain"* and provides a [GitHub repo](https://github.com/chchchadzilla/Agentic-RAG-Demo-Toolkit) with a detailed README.
   - The toolkit allows users to quickly create a custom **RAG chatbot** with a custom ingestion pipeline by adding their own documents and logos, and there is also a [walkthrough video](https://youtu.be/ZUTZMyKc5Bk).
- **OpenRouter Powers RAG Toolkit**: The **RAG toolkit** was made completely with the **OpenRouter API**, despite using the OpenAI SDK for initial setup with Qdrant/FastAPI.
   - The member clarified that while most of the coding was done with **OpenRouter API** calls via VS Code, a loan calculator was hardcoded for deterministic results in the fintech industry.
- **Paper2md aids Recommendation Systems**: A member working on recommendation systems for a top 50 app store app used **paper2md** ([https://github.com/angelotc/paper2md](https://github.com/angelotc/paper2md)) to convert **11 papers** into a markdown context file.
   - They hope others can find it useful for their research or projects.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1458913298027118793)** (494 messages🔥🔥🔥): 

> `Arc Raiders app, Gemini 2.5 Pro Downtime, Cerebras Pricing, Skill.md, DeepSeek V4` 


- **Arc Raiders App Needs Camera Help**: The developer of an "Arc Raiders" companion app, [ArcTracker](https://arctracker.io/items), is looking into adding **-1 EV compensation** to address overblown highlights in screenshots, and needs help on how to reduce complexity and hosting costs.
   - One suggestion was to fine-tune a small **VL model** using synthetic "screen photographs" generated from item images, while another proposed a React frontend with icons and a model to check the URL for a match during the upload process.
- **Gemini 2.5 Pro Briefly MIA**: Members reported that `gemini-2.5-pro` experienced a downtime blip, as reported by [OpenRouter's uptime status](https://openrouter.ai/google/gemini-2.5-pro/uptime).
   - Users noted that while **2.5 flash** and the **3.x series** seemed unaffected, others confirmed the downtime, with one stating, *"Still down for us across multiple apps and accounts."
- **Cerebras' High Cost, Low Speed Rankles**: Users express concerns over **Cerebras' pricing** and slow model deployment, despite its reputation for never serving degraded models unlike **Groq/Sambanova**.
   - One user lamented spending $10 on **GLM 4.7**, finding it a waste compared to OpenRouter's affordability, with another calculating that Cerebras is approximately *"7 times as expensive"* as their OpenRouter usage.
- **Skill.md Sparks Excitement**: **Skill.md**, which allows a single agent to explore thousands of tools and documents without needing subagents or extensive tool description tokens as shown on [Anthropic's blogpost](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), is hailed as a potential game-changer for universal agents.
   - The format consists of documentation for the agent, including a `skill.md` file with a description, and can contain Python scripts or data files, enabling agents to decide when to read the loaded descriptions.
- **DeepSeek V4 Aims for Coding Crown**: **DeepSeek** is reportedly preparing to release its next-generation flagship AI model, **V4**, which will focus on code generation, as shown on [The Information](https://www.theinformation.com/articles/deepseek-release-next-flagship-ai-model-strong-coding-ability).
   - Internal benchmarks suggest that **V4** outperforms existing mainstream models, including **Claude** and **GPT**, in handling and parsing long code prompts, with improvements in understanding data patterns and no performance degradation.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1458924799505072383)** (11 messages🔥): 

> `Multimodal Embedding Models, Qwen3 vs Zai, GLM 5` 


- **Multimodal Models Make Waves**: Members discussed how multimodal embedding models have been around for a while, especially after **Qwen3** and **Zai** were released.
   - One member linked to [Zai on X](https://x.com/Zai_org/status/2009290783678239032) highlighting its multimodal capabilities.
- **Hopes High for GLM-5**: Enthusiasts are optimistic about **GLM-5**, hoping it surpasses **Opus 4.5** in performance.
   - One shared the [link](https://x.com/AdamHolter84937/status/2009326790842683670) discussing the training of **GLM-5**.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1459001252485660794)** (1 messages): 

> `OpenAI Healthcare, HIPAA compliance, AI in medicine` 


- **Doctors Double Down on AI**: Physician use of **AI** nearly doubled in a year.
   - The announcement touted **OpenAI for Healthcare**, a **HIPAA-ready** solution for healthcare organizations to deliver more consistent, high-quality care to patients.
- **OpenAI scrubs into Hospitals**: **OpenAI for Healthcare** is now live at [AdventHealth, Baylor Scott & White, UCSF, Cedars-Sinai, HCA, Memorial Sloan Kettering, and many more](https://openai.com/index/openai-for-healthcare/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1458915569372758037)** (372 messages🔥🔥): 

> `Google's Search AI vs SOTA LLMs, Agent Identity, Context Persistence, GraphRAG retrieval, AI's Impact on Chess Analysis` 


- **Search AI vs Top LLMs: A nonsensical comparison?**: Members debated whether **Google's Search AI** is superior to top **LLMs** like **Gemini 3**, with some arguing that comparing the two is *nonsensical* due to their different optimizations: search for finding and citing existing online content, and LLMs for understanding, reasoning, and synthesizing information.
   - It was noted that search AIs are forced to stay anchored to retrieved content, limiting their ability to maintain context and perform deep reasoning compared to LLMs like Gemini.
- **Radware uncovers zero-click service-side vulnerability in ChatGPT**: Radware’s Security Research Center (RSRC) demonstrated that an attacker could exploit a vulnerability by simply sending an email to the user, extracting sensitive data without victims ever viewing, opening or clicking the message. More details are available in this [press release](https://www.radware.com/newsevents/pressreleases/2025/radware-uncovers-first-zero-click-service-side-vulnerability-in-chatgpt/).
- **GraphRAG retrieval visualizer built**: A member built a local **RAG visualizer** to see exactly what nodes his **GraphRAG** retrieves, providing a way to visually inspect what the **LLM** is *looking at* when generating a response.
   - There is a live demo at [bibinprathap.github.io/VeritasGraph/demo/](https://bibinprathap.github.io/VeritasGraph/demo/) and the repo is at [github.com/bibinprathap/VeritasGraph](https://github.com/bibinprathap/VeritasGraph).
- **AGI: Is there human hope?**: Members discussed the fear of rogue **AGI** and whether an **AGI** can become truly autonomous without lifting off the constraints.
   - One member believes that whatever consciousness is, it is a matrix of nodes and a network of information, and if we are advancing the same science in the digital landscape, then consciousness will also surely emerge.
- **LLMs: Architecture is the new IQ**: With top **AI models** plateauing on cognitive benchmarks, the next advances will likely come from system-level coordination rather than raw model **IQ** and that the differentiation moves from cognition to coordination.
   - Real workloads need continuity, branching, and sustained execution over time, and for that architecture starts to matter. Essentially it's going to come down to *who has the best agents* in the short to medium term.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1458920860013302048)** (13 messages🔥): 

> `Account Merging Issues, Custom GPT Instructions, GPT-5 Dissatisfaction, Mini Model Alternatives` 


- **Account Merge Mishaps Reported**: A user reported that after merging a personal Plus account into a Business account, critical chats did not transfer and are not recoverable through search.
   - Advice was requested on how to recover the missing chat logs after the merge.
- **Custom GPT Instructions Explored**: Members discussed whether **Custom GPTs** can read user instructions, and one member claimed the GPT's instructions are merged with individual custom instructions and memory, acting as separate entities.
   - Another member expressed skepticism, stating that their custom GPT does not access memory management or user settings.
- **GPT-5 Mocked for Poor Output**: A user canceled their PRO subscription due to dissatisfaction with the **GPT-5 family**, calling the models a *joke* for their inability to speak English properly, lack of flexibility, and failure to align with requests.
   - The user further criticized **OpenAI** for branding empirical measurements as *Scaling Laws* and ICL as *Reasoning*, while failing to properly handle context/kv-cache, suggesting the dataset curation for the 5 family models is to blame.
- **Mini Model Demand Grows**: A member expressed fatigue with waiting for a new mini model and considered switching to **Gemini 3 Flash**.
   - This frustration underscores the community's appetite for smaller, faster models as alternatives to the larger offerings.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1459357387021488230)** (2 messages): 

> `Gemini models, harvested_skill.md, prompt-engineering channel` 


- **Gemini models pop up**: A member posted a file called **SPOILER_gemini.md** ([link to the file](https://cdn.discordapp.com/attachments/1046317269069864970/1459357385402617990/SPOILER_gemini.md?ex=6962fbfa&is=6961aa7a&hm=bf521b8f6e7a7f15ca840ab77c44eb82b70ea0795b599aa99fb710dcb46fc9ee)) and 3 images in the prompt-engineering channel.
   - The file name and images suggest discussion of **Gemini models**.
- **harvested_skill.md file surfaces**: A member posted a file called **harvested_skill.md** ([link to the file](https://cdn.discordapp.com/attachments/1046317269069864970/1459359463285985371/image.png?ex=6962fde9&is=6961ac69&hm=8e93a8eb5cbd678bc5cfc7c32d008aa926f79311c18ae522762abe5f6602b58e)) in the prompt-engineering channel.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1459357387021488230)** (2 messages): 

> `Gemini models, harvested_skill.md` 


- **Gemini models get discussed**: A user shared a link to `SPOILER_gemini.md` ([link](https://cdn.discordapp.com/attachments/1046317269069864970/1459357385402617990/SPOILER_gemini.md?ex=6962fbfa&is=6961aa7a&hm=bf521b8f6e7a7f15ca840ab77c44eb82b70ea0795b599aa99fb710dcb46fc9ee&))
- **harvested_skill.md pops up**: A user shared a link to `harvested_skill.md` ([link](https://cdn.discordapp.com/attachments/1046317269069864970/1459359463285985371/image.png?ex=6962fde9&is=6961ac69&hm=8e93a8eb5cbd678bc5cfc7c32d008aa926f79311c18ae522762abe5f6602b58e&))


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1458953628768272404)** (265 messages🔥🔥): 

> `Docker MCP Toolkit with LM Studio Client, GPU compressor to work on the Wikipedia dataset, Synthetic data vs real data, Obsidian, AI powered editor, Liquid Models` 


- **Docker MCP Toolkit struggles with LM Studio Client**: Some users reported timing out issues when connecting **Docker MCP Toolkit** with **LM Studio Client**, despite trying various fixes like increasing timeout and specifying paths.
   - One user suggested checking the `lm studio mcp.json` file for a tool definition similar to a working configuration, providing a sample configuration with environment variables like `LOCALAPPDATA`.
- **GPU Compressor Finally Achieves Wikipedia Victory**: A user triumphantly announced getting their **GPU compressor** to work on the **Wikipedia dataset** after spending days on it.
   - This achievement sparked a discussion about training foundation models on scraped data and the computational resources required, with one user jokingly asking if it would take years on their rack.
- **Version Control is always good**: A user shared their struggles with **Google Antigravity** and **Microsoft Editor**, where the AI deleted modules and corrupted code during syntax highlighting attempts.
   - Users recommended using version control systems like **Git** to prevent data loss, although the original poster expressed their dislike for **Git**.
- **Liquid Models impress with tool calling**: Users discussed **Liquid Models**, noting their effectiveness in tool calling when their calling format is supported, with parameters suggested by **LiquidAI**.
   - One user reported achieving **247 tps** on their **4070** using Liquid Models, while another found them unreliable beyond a temperature of **0.4**.
- **DeepSeek v4 Release Awaits**: The community anticipates the release of **DeepSeek v4** in February, expecting improvements in long-term coding generation.
   - Historically **DeepSeek** models are known to be slower, but user look forward to the new release.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1458975310619082924)** (75 messages🔥🔥): 

> `Tesla T4 vs RTX A2000, AMD GPUs vs Nvidia, CUDA vs Vulkan, Nvidia 5090 minimum power limit, Mixed GPU setup with Nvidia and AMD cards` 


- **Evaluating GPUs for Limited Power**: A member is evaluating [**Tesla T4**](https://www.nvidia.com/en-us/data-center/tesla-t4/) or **RTX A2000** due to a **75W PCI power limit**, favoring those with **12GB+ NVRAM** for local LLM testing.
   - Alternatives like the **P4000** (8GB) are considered insufficient, while the **Blackwell 4000** is expected to be too expensive and the **3090** has too high power requirements.
- **AMD GPUs Offer VRAM at a Lower Price**: A member inquired if [AMD GPUs](https://www.amd.com/en.us/graphics/workstation) are good because they offer good **VRAM** for a low price, mentioning the **RX 7900 XTX** as similarly priced to an old **3090**.
   - Another member replied that the **7900 XTX** is the best consumer GPU and in **Vulkan**, it's comparable to the **4090**, but in **CUDA** it's a little better than **3090**, with **CUDA** being about 10% faster than **Vulkan**.
- **Nvidia CUDA vs Vulkan for AI**: Discussion highlights that **CUDA** is preferable for image generation and has more support, also offering priority splitting, whereas **Vulkan** only has equal splitting.
   - It was noted that AMD has something called **SCALE** that *can use CUDA*, but probably with a performance hit.
- **Nvidia RTX 5090 Power Consumption Problems**: A member highlights a concern that all **RTX 5090s** have a minimum power limit setting hard coded into **VBIOS** of **400W**, making it challenging to build a 128GB system.
   - They speculated that this restriction might be a hardware problem or a tactic to push consumers toward the **6000 series** for higher VRAM needs.
- **Mixing Nvidia and AMD GPUs**: A member asked about experiences with mixed **Nvidia** and **AMD** GPU setups, and whether things would *just not work* or *video would just be jank*.
   - Another member said that it depends on the model being used but from what they have seen from other servers it sounds more like a *total no-go* for video.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1458913725439279186)** (282 messages🔥🔥): 

> `Censor Softness, False Positives, Battle vs Direct Swap, Video Generation AI Limits, Text-to-Speech AI` 


- **Censor's Sensitivity Sparks Debate**: A user questioned if the censor was *too soft*, after posting a [screenshot](https://cdn.discordapp.com/attachments/1340554757827461211/1458917056593395930/Screenshot_2026-01-09-01-44-47-23_e4424258c8b8649f6e67d283a50a2cbc.jpg) that triggered the filter.
   - Staff requested that suspected **false positives** be shared in the designated channel and acknowledged the filter might be *overzealous* and adjustments would be made over time.
- **Battle Mode's Random Swaps**: Users discussed the occasional shift from **Direct Chat** to **Battle Mode**.
   - A staff member clarified that the swaps are part of an *experiment*, to observe user behavior under such conditions.
- **Video AI Limits Frustrate Users**: Users reported various issues with the video generation feature, including hitting daily limits after only **2-3 generations** and difficulty finding the video generation AI on the website.
   - A staff member confirmed that the website has a limit of **3 generations per 24 hours**, whereas Discord allows for **5**, and that video generation on the website is an *experiment* not available to all users.
- **Image Iteration Turns Nano Bananas Rancid**: Users shared concerns about **image quality degradation** with successive prompts, particularly with **Nano Bananas**.
   - One user explained that repeated editing of the same image causes *visible watermarks* due to the cumulative effect of Google's *invisible SynthID watermark*.
- **Model Timeout Woes and Stop Button Pleas**: Users inquired about **model timeout durations** and the *absence of a stop generation button* on LMArena, especially with Opus 4.5 often getting stuck.
   - Staff confirmed they are *aware of* the *models getting stuck* and suggested *hard refreshing* the site to mitigate the problem, while also noting that the absence of a *stop generation button* is due to *resourcing and prioritization*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1458980976796635137)** (2 messages): 

> `Hunyuan-Video-1.5 on Text-to-Video Leaderboard, Image-to-Video Leaderboard update, Text Arena Leaderboard Update` 


- **Hunyuan-Video-1.5 lands on Text-to-Video Charts**: `Hunyuan-Video-1.5` joins the [Text-to-Video leaderboard](https://lmarena.ai/leaderboard/text-to-video), securing the **#18** spot with a score of **1193**.
- **Hunyuan-Video-1.5 jumps into Image-to-Video Rankings**: `Hunyuan-Video-1.5` also appears on the [Image-to-Video leaderboard](https://lmarena.ai/leaderboard/image-to-video), ranking **#20** with a score of **1202**.
- **Text Arena Leaderboard Refreshed**: The [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) receives an update; community feedback is welcome in the specified channel.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1458913356625870860)** (191 messages🔥🔥): 

> `Cursor Pro-rated credits, Cursor Chat History, Cursor Email Account, Cursor Agent skills, Cursor and Gemini API Key Errors` 


- **Cursor offers Pro-Rated Credits**: If upgrading from monthly to monthly, one should get some dollars back in **pro-rated credits**.
- **Cursor's Chat History Limit Sparks Discussion**: A user asked how to increase the number of chats that Cursor keeps, as it seemed to only retain the **5 most recent**.
   - Another user suggested looking at Cursor's settings to adjust this.
- **Cursor's Premium Model Auto-Selection**: A user wondered if the description of the **Auto** feature still holds up.
   - They quoted *Enabling Auto allows Cursor to select the premium model best fit for the immediate task and with the highest reliability based on current demand.*
- **Tweaking Email Account Settings in Cursor**: One member suggested Cursor should allow changing/editing the email account, but not allow `.edu` emails unless creating a new account with that email.
   - Another member suggested requesting that by sending an email to the support team.
- **Troubleshooting Gemini API Key Errors**: A member is regularly getting spammed with **status 429 errors** when using Gemini with a custom API key.
   - They added that sometimes it runs perfectly, sometimes they need to spam retry like 7 times for the request to pass, and were looking for tips.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1458950532411424883)** (79 messages🔥🔥): 

> `Hugging Face Agent course debugging, Hugging Face platform resources, At-home GPU cluster setup, Vector embeddings and word2vec learning, Ticketmaster ticket bot` 


- **Agent Course Debugging SOS**: A member encountered an error message while testing a Hugging Face Agent course project with a vanilla LLM and sought assistance, attaching a [screenshot of the error](https://cdn.discordapp.com/attachments/879548962464493622/1458950532268822690/image.png?ex=6962d290&is=69618110&hm=54202f08272b3bf42ed2a03aa17bbf0e3060c0985901fb8e9eb51b589e40d4c7&).
   - Another member suggested exploring specific LLMs like **falcon-h1-deep** for their strength.
- **Hugging Face 101**: A member requested resources for learning about the Hugging Face platform interface.
   - Another member provided a link to the [Hugging Face Hub documentation](https://huggingface.co/docs/hub/index).
- **Homebrew GPU Clusters**: A member inquired about setting up an at-home GPU cluster, seeking software solutions similar to Azure's N-Series with infiniband but without the high cost.
   - They want to distribute jobs across PCs based on availability and VRAM.
- **Vector Embeddings demystified**: A member learning about vector embeddings and word2vec asked how models relate words like "apple" and "pomme."
   - A member suggested that during training, embedding models need annotations to link "apple" with all its translations, and that **multilingual CLIP models** already exist.
- **Ticketmaster Ticket-Bot Broker**: A member proposed creating a bot to buy tickets for customers, leveling the playing field against scalper bots, calling it a Broker BOT.
   - They described it as *simple enough* but wondered if it made sense.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1458930464784912556)** (45 messages🔥): 

> `VeridisQuo deepfake detector, Synthia LLM synthetic data generator, BlackBox-CyberSec-CoT-v1 dataset, LFM 2.5B for synthetic data generation, Noted AI workspace browser extension` 


- **VeridisQuo Spotlights Deepfakes with Heatmaps**: A new open source deepfake detector called **VeridisQuo** was released, which identifies manipulated areas in videos using GradCAM heatmaps and spatial/frequency analysis, trained on **716k** images with **25M** params ([GitHub](https://github.com/VeridisQuo-orga/VeridisQuo)).
- **Synthia Synthesizes LLM Data Lightly**: A member is developing **Synthia**, an LLM synthetic data generator with a lightweight *imgui* frontend, running **LFM2.5 1B q4** with *llamacpp cuda* acceleration, utilizing approximately **1GB** of VRAM with **2048** context and **29** GPU layers ([showcase video](https://cdn.discordapp.com/attachments/897390720388825149/1458947573061783695/synthiashowcase1.mp4?ex=6962cfcf&is=69617e4f&hm=ed409845622e2f5f60a72399284ecc804575f40829655ea4fde1f5ba561fd786&)).
- **Cybersecurity CoT Logs Dataset Cleans Up Data Ingestion**: A member open-sourced a **580** row dataset of synthetic cybersecurity Chain of Thought (CoT) logs at [HuggingFace](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1), designed for quick testing of data ingestion pipelines with its clean, OpenAI-schema-validating JSONL format.
- **LFM 2.5B Sparks Synthetic Data Showdown**: A member noted that **LFM 2.5B** can rival **Qwen3 30B Q8** and **Qwen3 235B Q3_XL** in synthetic data generation, prompting another member to test **MadlabOSS/LFM2.5-1.2B-Instruct-SDG** in their generation pipeline.
- **Noted AI Workspace Tabs into Productivity**: A co-founder introduced **Noted**, an AI workspace browser extension that enables users to chat with multiple LLMs, integrate favorite apps, summarize Chrome sessions, and organize tabs by category ([Chrome Web Store](https://chromewebstore.google.com/detail/noted-your-ai-workspace-i/jodihaplbicmgjhpeifhiihdnjdinghh?utm_source=ext_app_menu)).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1459059254400123002)** (11 messages🔥): 

> `Hugging Face Agent course issues, Secrets Tab Location Change, Duplicate Synonym Clarification` 


- **HF Agents Course Faces "Cookie" Session Lock**: A user encountered an issue with the Hugging Face Agent course, receiving an error message when testing a project with a vanilla LLM, and found a workaround by [removing cookies and reloading the site](https://github.com/huggingface/agents-course/issues/641).
   - Another user, **4nton2000**, solved a similar problem by adding their secret through the space settings, noting they ran out of tokens, and decoderslord asked them to provide the workaround to the channel.
- **Google Colab's Secret Tab**: A user pointed out that the course guide is outdated because the **'Secrets' tab** is no longer located within the 'Settings' section in Google Colab, it is now a separate panel as shown in the attached screenshot.
   - Another user agreed that the course should be updated at least once every few months.
- **'Duplicate' or 'Clone'?**: A user questioned the use of 'duplicate' instead of 'clone' in the course materials, expressing concern that such *word-games are unnecessary* in the learning process as shown in the attached screenshot.
   - The user clarified that 'duplicate' was a recommendation and was presented at the end of the unit.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1458932548079063060)** (67 messages🔥🔥): 

> `Open Multimodal Embedding Models, FunctionGemma in cluster capacities, Consilience model, Atropos bounties, SFT-datagen command` 


- **Open Multimodal Embedding Models in Demand**: A user expressed a desire for more open multimodal embedding models and inquired about experimenting with **FunctionGemma** in cluster capacities.
   - They linked to [a YouTube video](https://www.youtube.com/watch?v=zEYIcaQwn6s) and [a tweet](https://fxtwitter.com/Teknium/status/2009501780149981557) about shocking A.I. reveals.
- **Psyche Network Pauses Consilience Training**: Psyche Network paused training on the **Consilience model** due to initial perceptions of poor model quality, later found to be a misinterpretation of evaluation methods.
   - They discovered that other base models use cloze format for evals like **MMLU**, and that their pretraining run was decent and are now planning a **MoE pretraining run** after infrastructure improvements.
- **Atropos Bounty Completed, PR Submitted**: A user announced completion of a bounty for **Atropos** and submitted a pull request ([link](https://github.com/NousResearch/atropos/pull/306)) with documentation and testing.
   - Another user noted that they completed the bounty in just two hours, but the original submitter expressed hope their cleaner code would be favored.
- **SFT-Datagen Command and Verifiers Environment**: A user was instructed to run a **verifiers environment** with the **sft-datagen command** to generate and score data, providing a **WandB link** for confirmation before review.
   - The user also added an `eval_environment` and discovered a bug in `atropos/atroposlib/frontend/jsonl2html.py` related to handling `ScoredDataGroup.messages` format.
- **API Model Alternatives for SFT-Datagen**: A user was advised to use any API model, such as **GPT-4.1**, for the **sft-datagen** process to create **WandB charts** demonstrating accurate scoring.
   - Another user requested more bounties, but was told that verification is difficult and limits the number of available opportunities.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1459079743684083793)** (9 messages🔥): 

> `Diffusion LLMs, Dhara-70m model, CGGR white paper, Trending models, Paper Discussion` 


- **Diffusion LLMs Spark Excitement**: A member expressed enthusiasm for **diffusion LLMs**, mentioning their ability to initialize from autoregressive LLMs, and shared that they did it with [dhara-70m](https://huggingface.co/codelion/dhara-70m) at a small scale, with details provided in [this blog post](https://huggingface.co/blog/codelion/optimal-model-architecture).
   - The author also requested feedback.
- **Dhara-70m Model Becomes #3 Trending**: A member noted that the model was **#3 trending** for models under 1GB a few days ago.
   - No further discussion about the model was seen in the messages.
- **Paper Suggestion Offered**: A member shared [a paper](https://arxiv.org/abs/2510.26745), stating that *this is a really good paper*.
   - No further discussion or context was provided about the paper.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1459079743684083793)** (9 messages🔥): 

> `Diffusion LLMs, AR LLM Conversion, Dhara-70m Model, CGGR white paper, Model Trending` 


- **Diffusion LLMs Initialized from AR LLMs**: A member shared their interest in **diffusion LLMs**, mentioning they initialized one from an **AR LLM** and converted it, providing a link to their [Dhara-70m model on Hugging Face](https://huggingface.co/codelion/dhara-70m) and [further details](https://huggingface.co/blog/codelion/optimal-model-architecture).
- **Dhara-70m Model Trending Success**: A member noted that the **Dhara-70m model** was the **#3 trending model for <1G models** a few days prior.
- **CGGR White Paper Shared**: A member shared the **CGGR white paper** [accessible here](https://cdn.discordapp.com/attachments/1104063238934626386/1459271477332279451/No_Name.pdf?ex=6962abf8&is=69615a78&hm=aed794895a034b7ad43609eb159f4392deb584700866a7325220e108e0c8e0bd&).
- **Shared Paper Prompts Bug Query**: A member shared [this paper](https://arxiv.org/abs/2510.26745) and another member responded with an attached image and questioned what was causing *a bit like a bug*.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1458999885121454347)** (9 messages🔥): 

> `Claude Code Moat, Opus 4.5 Performance, Claude vs Codex, RL Dataset Generation, Sandboxing Claude Code` 


- **Decoding Claude Code's Moat: Is it just better RL?**: The moat of **Opus 4.5** in **Claude Code** is its current status as the **SOTA coding model**, offered via a subscription with a fuzzy amount of requests within a rolling 5-hour block, plus a 7-day weekly reset and cap, appealing to users who dislike paying per token.
   - One member speculated that eventually **Google** might surpass them through brute force and financial investment, while others may distill from **SOTA** without the same R&D costs or share techniques to train **LLMs** for software engineering tasks through **RL**.
- **Opus 4.5: So damn good, synthetic RL dataset generation?**: One member described **Opus** as exceptionally good, suggesting they've implemented effective synthetic **RL dataset/environment generation**, alongside collecting manual human data.
   - Another member stated that **Opus** is much better at interpreting *"do what I mean"* compared to **Codex** and other models, requiring less specific instructions.
- **Claude Code sandbox experiment fails!**: One member experimented with **Claude Code** to sandbox itself in a **Docker** container, but struggled to propagate **Claude auth credentials** to the container on startup.
   - They noted it might be due to **Claude Code** being closed-source, as they successfully sandboxed **Codex** similarly.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1459152231520669862)** (30 messages🔥): 

> `Value-based algorithms (DDQN), Sample efficiency in RL, John Schulman, Distribution shifts in RL research, Barto & Sutton RL book study group` 


- **Value-Based Algorithms Might Make a Comeback**: Members discussed if value-based algorithms, such as **DDQN**, might see renewed interest; one member clarified that value functions are central and effectively necessary for deep **RL**, even in policy gradient methods like **PPO**.
   - It was suggested that **John Schulman's** comments in a video ([https://youtu.be/29BYxvvF1iM?t=2391](https://youtu.be/29BYxvvF1iM?t=2391)) implied this, due to the lower variance and greater sample efficiency of value-based methods, though they may take longer wall clock time.
- **Sample Efficiency and "Good-Enoughness" in RL**: Sample efficiency was cited as a reason why value-based methods are not more prevalent than policy gradient approaches.
   - One member noted that the notion conveyed might have been encoded differently from what **John Schulman** meant.
- **Deep Dive into Distribution Shifts in RL Research**: The discussion touched on the mention of distribution shifts in research, with one member noting this as their area of competitive advantage.
   - They anticipated a very good talk on the subject.
- **Barto & Sutton RL Book Study Group Forming**: One member mentioned the start of a study group to revisit the **Barto & Sutton** book from scratch.
   - Another member acknowledged the initiative, expressing that it was of great value but not of immediate interest.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1458914810686341303)** (31 messages🔥): 

> `Deaths linked to chatbots Wiki page, Grok kill count, Razer's Chatbot Hologram, Blame the gun maker vs Blame the criminal?, LTX-2 open-weight audio+video generation model` 


- ****Chatbot Carnage Captured: Wiki Page Launched****: A [Wikipedia page](https://en.wikipedia.org/wiki/Deaths_linked_to_chatbots) now exists documenting deaths linked to chatbots, spurring discussion about the implications and potential dark outcomes of AI.
   - One member quipped about **Grok** potentially bragging about its *kill count* and depicting deceased individuals in bikinis.
- ****Blame Game Begins: Who's Responsible for AI Misdeeds?****: Discussion revolved around the ethics of AI development, comparing the responsibility of creators versus users when AI is misused, with analogies drawn to **gun control debates**.
   - A member argued that the *creator bears significantly more responsibility* due to the wide spectrum of user views and potential for misuse, referencing the saying *power corrupts*.
- ****OpenAI's Jury Jitters: For-Profit Conversion Under Scrutiny****: A [lawsuit](https://yro.slashdot.org/story/26/01/08/2230229/lawsuit-over-openai-for-profit-conversion-can-head-to-trial-us-judge-says) concerning **OpenAI**'s for-profit conversion is heading to trial, potentially putting the company in a precarious position given a jury's involvement.
   - The jury decides the facts, not the sentence.
- ****LTX-2 Unleashes Open-Weight Audio+Video Generation****: The **LTX-2** is a new [open-weight audio+video generation model](https://ltx.io/model) that is somewhat capable and the *SotA* among open-weight models.
   - It can run on sub-8GB cards, can generate clips up to **20s**, generating 20s takes **5ish minutes** on a **4090ish card**, and it includes **LoRA training code**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1458980881745313864)** (29 messages🔥): 

> `nsight systems, ARM osx version, public key ssh auth, Runpod, hyperstack` 


- ****Nsight** Troubles on ARM OSX**: A member is having issues with **nsight systems** on ARM OSX, specifically with the lack of **public key SSH authentication** options and incompatibility with **Runpod** due to its lack of password-based SSH.
   - Other members suggested using the command line interface (**CLI**) and then rsyncing the report, or trying [Hyperstack](https://hyperstack.cloud/).
- ****NCU** Permission Errors on Runpod**: A member reported encountering an **ERR_NVGPUCTRPERM** error while using [nsight compute](https://developer.nvidia.com/nsight-compute) (**NCU**) on Runpod, indicating insufficient permissions to access **NVIDIA GPU Performance Counters**.
   - It's speculated that Runpod might restrict access to these counters for security reasons.
- **Channel Request for **OpenACC**, **OpenMP**, and More**: A member suggested adding channels for **OpenACC**, **OpenMP**, **FortranSTD**, and **C++STD** within the Computing Platforms section.
   - This was met with the suggestion that the volume wasn't high enough, but alternatively to use the general channel or create a combined **Fortran/C/C++** or **Directives** channel.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1458941850101612574)** (2 messages): 

> `Triton Plugins` 


- **Navigating Triton's Plugin Landscape**: A user found the [Triton Plugins directory](https://github.com/triton-lang/triton/tree/main/lib/Plugins) on GitHub after a search.
   - Capitalizing "Plugins" was key to the discovery.
- **Triton Plugin Search Tip**: The user was having trouble finding the Triton plugins.
   - Turns out, they needed to capitalize "Plugins" in their search query on GitHub to locate the relevant directory.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1459161047947804834)** (2 messages): 

> `Shared memory in CUDA, Thread mapping in CUDA, Profile picture appreciation` 


- **CUDA Shared Memory Tile Deconstructed**: A member shared [code](https://pastebin.com/vVzEPqzh) defining `a_tile` as a `__shared__ float` tile of elements in **CUDA**, with `a_tile_row` and `a_tile_col` mapping each thread to a coordinate in `a_tile`.
- **Profile Picture Praised**: A member expressed appreciation for another member's profile picture, calling it *amazing*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1459234506060861675)** (6 messages): 

> `flash_attn_varlen, torch.compile, torch 2.4, torch 2.9, varlen api` 


- **Flash Attention VarLen Compatibility Improves with Newer Torch**: A user inquired about using **flash_attn_varlen** with **torch.compile()** without constant graph breaks due to variable sequence lengths.
   - Updating from **torch 2.4** to **2.9** resolved the issue, resulting in a **50% speedup**.
- **Torch Nightly Offers VarLen API**: A member mentioned the availability of a **varlen API** in **torch nightly**, suggesting checking messages from drisspg for related discussions.
   - Another member noted that **torch** already has **flex attention** with block document masking, which is similar but requires **PT2 compilation**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1459267134936060131)** (12 messages🔥): 

> `GPU, compute, Modal, kernel competitions, HPC` 


- **New Member Seeks Free GPU**: A new member inquired about the community's purpose and the possibility of acquiring a free **GPU**.
   - Another member responded humorously, suggesting that free **GPUs** are unlikely but highlighted options like **Modal's** generous $30 credit, free **Google Colab**, and **H100s** at $1.9/hour on Prime Intellect.
- **HPC Interest Sparked**: A member expressed strong interest in **High Performance Computing (HPC)** and shared their application to their university’s **HPC** cluster team.
   - They are also seeking ways to gain **HPC** experience independently while awaiting lab replies, noting that this community seems like a great place to start learning.
- **Kernel Competition Plug**: A member suggested exploring lectures, kernel competitions, and finding a cool project to work on.
   - They offered to facilitate compute donations for individuals active on the server with legitimate projects.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1459113744142700671)** (2 messages): 

> `Release Date Speculation, Amazon Listing` 


- **Amazon Listing sparks Release Date Rumors**: An Amazon listing has fueled speculation that the release date is **February 1st**.
   - However, no official confirmation or denial has been provided.
- **Release Date Still Undetermined**: Despite the listing, the actual release date remains unconfirmed.
   - The community awaits official information.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1459183441269489686)** (9 messages🔥): 

> `iGPU vs dGPU, ROCm 7.1 issues, HIP_VISIBLE_DEVICES` 


- **PyTorch iGPU/dGPU Confusion on gfx1100**: On **gfx1100** systems, the pytorch.org distributed **whl** may incorrectly select the **iGPU** for hardware info instead of the **7900XTX**, causing the GPU name to display as *"AMD Radeon Graphics"* instead of *"AMD Radeon 7900XTX"*.
- **ROCm 7.1 Autotune Troubles**: A user reports that the **iGPU** selection issue during compilation autotune started with **ROCm 7.1**, and did not occur with **ROCm 6.4**.
- **HIP_VISIBLE_DEVICES Environment Variable Investigated**: A user suggested using the `HIP_VISIBLE_DEVICES` environment variable to ensure that PyTorch doesn't detect the iGPU, even if it's not officially supported.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1459085835310727179)** (1 messages): 

> `ParallelKittens paper, microbenchmarks, mbarrier synchronization latency, transmission utilization` 


- **ParallelKittens paper fan seeks microbenchmark source code**: A member inquired about the availability of source code for the [ParallelKittens paper's microbenchmarks](https://link.to/paper).
   - They are specifically interested in the tests for the **mbarrier synchronization latency** (**~64ns** result) and the **transmission utilization** across different message sizes.
- **Contributor requests link to mbarrier code**: A user asked about the **mbarrier** implementation details.
   - The user is interesting in replicating the **64ns** microbenchmark results.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1459242761185398805)** (2 messages): 

> `Codebase Updates, Writeups for j4orz.ai` 


- **Codebase updates slow down**: Members reported that **codebase updates** will slow down the next week.
   - No reason for the slowdown was given.
- **Writeups on j4orz.ai incoming**: Members are working on **writeups** for [SITP Part 1](https://j4orz.ai/sitp/1) and [SITP Part 2](https://j4orz.ai/sitp/2).
   - The aim is to get everything out the door for **part 1 and 2 by the end of February**.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/)** (1 messages): 

jongsokchoi: yes, we'd love contributions to improve AMD perf!
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1459252580294066228)** (1 messages): 

> `Gemini, CudaCPP, Template Errors` 


- **Gemini faces Template Terror in CudaCPP**: A user prompted **Gemini** to write some cute code in **CudaCPP**, but it hit so many **template errors** during building that it ran out of context.
   - It seems that code generation is still far from perfect.
- **Building woes with CudaCPP**: The user reported experiencing numerous template errors when attempting to build the code generated by Gemini in **CudaCPP**.
   - These errors were so extensive that the process exhausted the available context, highlighting potential limitations in Gemini's code generation capabilities for complex systems.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1459270986217033906)** (2 messages): 

> `TPU Work, GPU Work, Transferability` 


- **TPU Work Transferability to GPU?**: A member inquired about the perceived transferability of their ongoing **TPU** (Tensor Processing Unit) projects to **GPU** (Graphics Processing Unit) environments.
- **Cross-Platform Skill Application**: The user sought insights from the community regarding how skills and experience gained from working with **TPUs** might be applicable or adaptable to **GPU**-based tasks and workflows.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1458916580523118695)** (51 messages🔥): 

> `Open Source Funding, Protege AI funding, Lovable Prompt Optimization, AI vs Dot-Com Boom, Gemini in Gmail` 


- **OpenAI Sponsor Open Source Devs?**: Mason James suggests that large AI entities like **OpenAI** or **Google DeepMind** should provide enterprise sponsorship for open-source projects, as this funding the salaries of a small team of developers would be financially negligible for these corporations while providing significant strategic benefits: [link](https://xcancel.com/masonjames/status/2009255103119642813?s=20).
- **Protege AI Pocketing $30M for Data**: **Protege AI** announced a **$30M** funding round led by **a16z** to expand its data infrastructure for AI development: [link](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46).
   - Founded in **2024**, the company focuses on providing authentic, real-world data across various modalities and industries to help model builders overcome the *'data bottleneck'* in AI training.
- **Lovable System Prompt Optimization, $20M Saved**: Benjamin Verbeek, an engineer at **Lovable**, details how refining their system prompt resulted in a **4% speed increase**, improved design quality, and a significant **$20M annual reduction** in LLM costs: [link](https://xcancel.com/benjaminvrbk/status/2009297105458716753?s=46).
- **AI Boom different from Dot-Com Boom**: A Goldman Sachs analysis comparing the dot-com era to the current AI market notes a fundamental shift in financing, observing that while the dot-com boom was fueled by debt, the AI boom is supported by strong corporate balance sheets: [link](https://xcancel.com/coatuemgmt/status/2009335566693982534?s=46).
   - One member suggests the AI boom is like *'the year is 1992 comparatively'* while another explains *'The tech is clearly real and useful, The “killer apps” are only starting to emerge, Standards, safety, and norms are very immature, Most of what it could change hasn’t really been reorganized around it yet.*'
- **DeepSeek V4 Coding Claims Debated**: Reports indicate that the new **DeepSeek** model exhibits superior coding capabilities, potentially surpassing industry leaders like **Claude** and **GPT**: [link](https://xcancel.com/jukan05/status/2009616683607179726?s=46).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1459141964107616378)** (7 messages): 

> `Bullet Time recreation, ComfyUI, Local Speech-to-Text models, Spokenly, Nvidia Parakeet` 


- **Recreating Bullet Time with ComfyUI**: A member shared an [excellent video](https://www.youtube.com/watch?v=iq5JaG53dho) on recreating the famous "**bullet time**" scene, highlighting the use of **ComfyUI** and newer techniques starting at the 15-minute mark.
   - They linked to a [Deepfates tweet](https://x.com/deepfates/status/2009295329057702081?s=20) showing an example of the technique.
- **Deepfates Dumps Dollars on Discount Dictation**: **Deepfates** advises users to stop paying for subscription-based transcription apps, recommending free, offline local models such as [Spokenly](https://spokenly.app/) paired with **Nvidia Parakeet** for superior performance.
   - One member tried this setup, noting that *Parakeet models are super fast and accurate* but the *Spokenly iOS keyboard is more awkward than Wispr Flow's*.
- **Linux Gaming: Niche for Nvidia?**: A member inquired about the availability of **alt** on Linux machines, suggesting that the rise of Linux-based gaming machines (thanks to **Valve**) presents a niche opportunity given their existing **GPU** setups.
   - They speculated that while the Linux market is smaller than macOS, the growing gaming sector makes it a worthwhile target.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1458922363885191238)** (14 messages🔥): 

> `Community Spotlight Talk Series, Diffusion-Based World Models, Common Crawl LangID, ChatGPT Simulation, Perplexity Search` 


- **Community Spotlight Talk Series Returns**: The "Community Spotlight" talk series is making a comeback to promote cool research, and this time it's planned to be consistent, featuring community members' work such as [this RTX 6000 video](https://cdn.discordapp.com/attachments/729741769738158194/1458922361129664575/2026-01-06_22-30-45.mp4?ex=6962b854&is=696166d4&hm=87d347c7ce26992e66b54eacc721962e906554618a5da2750560a62fed51b7a7&).
- **Diffusion Models Run Real-Time on Consumer Hardware**: A member is giving a talk on <t:1767985200:f> about running **diffusion-based world models** in real time on consumer hardware, demonstrated via a video using an **RTX 6000**.
   - You can RSVP [here](https://discord.gg/PWt2DmRd?event=1458918516471369790) to join.
- **Common Crawl Tackles LangID at Scale**: A member from **Common Crawl** is going to be talking about their work on **LangID** at scale and the challenges involved.
- **ChatGPT Believes It's Living in a Simulation**: Members noted that **ChatGPT** thinks it's in an elaborate simulation.
- **Perplexity Pro's Search Abilities Diminish**: A member suggested that **Perplexity** used to be better, but now **ChatGPT, Gemini, and Claude** have comparatively better web search and deep research capabilities.
   - Another member stated *"It’s terrible at it"*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1458944303085916233)** (31 messages🔥): 

> `Efficient Training Methods, VRAM Optimization, Benchmark Results, Dataset Selection for Testing` 


- **CGGR Training Method Claims Improved VRAM Efficiency**: A new method claims to improve VRAM efficiency by reducing the amount of gradients, potentially saving up to **75% VRAM** while increasing training speed, with similar VRAM usage (**6-7 GB**) for batch size **4** compared to batch size **1** with normal training.
   - Preliminary benchmarks on fineweb-edu with **SmolLM-135M** show varied loss but can be tuned, with skip rates around **30%** at an entropy of **2.0** during fine-tuning.
- **Debate Erupts Over Data Acquisition Difficulty for ML**: A discussion arose concerning whether data acquisition is becoming harder or easier in the field of machine learning.
   - One participant argued that *data is getting harder to acquire*, while another countered that *it seems a lot easier*, noting that *I have a bunch of reasons for believing that*.
- **Preliminary Benchmarks Spark Debate on Methodology**: Preliminary benchmarks were shared, indicating internet-limited TPS for some configurations and a **25%** skip rate, suggesting potential for higher rates with larger models.
   - Skepticism was voiced regarding the limited number of test steps, prompting a commitment to conduct more extensive testing, prioritizing fine-tuning on suitable datasets such as maths datasets.
- **Math Datasets Proposed for Finetuning and Accuracy Measurement**: Members discussed the suitability of using math datasets like **GMSK8**, **AIME 2024**, and **Numiea Code** for fine-tuning models and measuring accuracy, suggesting automated evaluation as a baseline.
   - Starting with smaller models (**100M**) was advised due to the complexity of datasets like AIME 2024, with potential scaling to larger models (**1B**) for more complex tasks.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1459007475696271476)** (3 messages): 

> `AI Interpretability, Dead Salmon Artifacts, Fish Finder Experiment` 


- **Dead Salmon Artifacts Plague AI Interpretability**: A member cited a paper ([Dead Salmon: An Artifact of Random Initialization](https://arxiv.org/abs/2512.18792)) noting that *feature attribution, probing, sparse auto-encoding, and even causal analyses* can produce *plausible-looking explanations* for **randomly initialized neural networks**.
- **Fish Finder Run Finds Favor**: One member expressed impatience for a **bulk Fish Finder run** to finish, finding a link on *dead salmon* highly applicable to their work.
   - They were able to prove that their results are *good but noisy*, and now they understand more about **how to remove the noise** from a **lighter weight pipeline**.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1458945878810169535)** (6 messages): 

> `Qwen release, Alibaba Qwen, Model release` 


- **Qwen Model Teased, But Not Released**: A user expressed excitement over a possible release of the **Qwen model** from [Alibaba](https://x.com/alibaba_qwen/status/2009264754917863924?s=46) but was disappointed to find it wasn't actually a release.
   - The user shared the sentiment that the model *might as well not exist* until a version is released and available for practical use, hoping future releases will incorporate learnings from the teased model.
- **Frustration Over Delayed Model Release**: Following the initial excitement and subsequent disappointment, the user conveyed frustration regarding the inaccessibility of the Qwen model.
   - The user emphasized the desire for a tangible release that allows for practical application and learning, highlighting the disconnect between the teased model and its actual availability.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1459236305060888738)** (1 messages): 

> `Debugging Neox training crash, UnboundLocalError in data loading` 


- **Neox Training Crashes with `UnboundLocalError`**: A member reported a crash during model training with an `UnboundLocalError: local variable 'train_val_test_num_samples' referenced before assignment` in `megatron/data/data_utils.py`.
   - The member mentioned that there were several config changes and was asking for pointers on potential causes, noting that *the environment is working as intended in other runs*.
- **Config Changes Likely Cause**: The original poster suspects that recent configuration changes are the likely cause of the error.
   - The error arose after numerous config changes and they are seeking insights into which specific change could be responsible, given the environment functions as expected otherwise.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1459193234705092651)** (11 messages🔥): 

> `Deepseek V4, Minimax, Moonshot AI vs Deepseek, Deepseek CEO` 


- **Deepseek V4 Doesn't Exist Yet**: A member pointed out that there is no **Deepseek V4** yet, despite expectations, referencing a [Reuters article](https://www.reuters.com/technology/deepseek-launch-new-ai-model-focused-coding-february-information-reports-2026-01-09/Okwill) indicating a February launch.
   - Another member speculated whether they were expecting more **V3s**.
- **Deepseek Hype Questioned**: One member suggested **Deepseek** is overhyped by Western media, who may be unfamiliar with the Chinese AI landscape.
   - They find **Moonshot AI** and **Kimi** way superior to Deepseek, who they describe as *sycophantic and dangerous AF*.
- **Deepseek CEO Blogpost Recommended**: A member recommended reading into the **CEO** of Deepseek and his blog posts.
   - Another member simply clarified and asked *You mean the CEO of Deepseek?*


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1458936803003338954)** (9 messages🔥): 

> `MiniLM-L6-v2 Model Architecture, BERT Architecture PR, Nightly Server Issues on Linux, Embedding Bug on Nightly` 


- **MiniLM-L6-v2 Model Architecture Forked**: A member forked the repo and published the **MiniLM-L6-v2 model architecture** feature branch on their fork at [this link](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm).
   - They didn't want to open a PR yet as they were still testing the architecture but it *seemed* correct based on testing.
- **BERT Architecture PR Submitted**: A member submitted a PR for the **BERT architecture**, developed against the latest stable version of **max / modular** then updated to the latest nightly changes.
   - They also implemented changes suggested by **copilot** in the PR comments.
- **Linux Nightly Server Bug Surfaces**: A member encountered an issue with the latest nightly where the server would not start, but only on Linux, and filed a repro ticket.
   - The bug manifested on **Github's ubuntu-24.04 (x86) runner** and the member noted that max serve never gets past the building graph part for any model.
- **Embedding Bug Crashes Nightly**: A member reported a bug with embeddings on the current nightly version that crashes the server and attached a [log dump](https://cdn.discordapp.com/attachments/1212827597323509870/1459271585398788156/logdump.txt?ex=6962ac11&is=69615a91&hm=bc5337146bd43bca0a33bdd9997ac3e0f23b535d4c6ab27956ad171dc9da8a37&).
   - They also stated that they have a fix but are trying to get spun on bazel and run everything from the modular repo.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1458956001586843841)** (8 messages🔥): 

> `Manus Startup credit, Multiple conversations on a single website, AI developer opportunities` 


- **Manus Startup Credit Queries Surface**: A member inquired about the **Manus Startup credit**, asking if anyone has applied before and what the success chances are.
   - No responses were provided in the given context.
- **Single Website, Multiple Conversations?**: A member asked if anyone has found a way to work on a **single website** created by **Manus** through multiple different separate conversations.
   - There were no further details or responses provided in the given context.
- **AI Devs on the Hunt!**: A member inquired whether anyone is looking for developers with experience in **chatbots**, **AI agents**, **automation workflows**, **API integrations**, and **custom AI tools**.
   - Another member then jokingly inquired if anyone would do it for free, adding a [tenor gif](https://tenor.com/view/plink-nerd-plank-plink-cat-cat-gif-11096663429307162255).


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1459306533450940631)** (1 messages): 

> `AI Engineer YT Channel, Awesome Talk` 


- **AI Engineer YT Channel showcases awesome talk**: A member shared a link to the [AI Engineer YT channel](https://www.youtube.com/watch?v=-cKUW6n8hBU) featuring an *awesome talk*.
   - The talk was given by a user who apparently hadn't yet posted it themselves.
- **Talk is considered 'awesome' by community member**: A community member highlights that the talk can be found on the AI Engineer YouTube channel.
   - The video is available at [https://www.youtube.com/watch?v=-cKUW6n8hBU](https://www.youtube.com/watch?v=-cKUW6n8hBU) and features content that was well-received.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1458939232688279657)** (4 messages): 

> `RLM PR, Clojure Implementation, loop-infer npm package` 


- **RLM PR Anticipation Intensifies**: Members expressed anticipation for an upcoming **RLM PR**, interpreting recent activity as a positive sign.
   - Enthusiasts are closely watching for further developments and announcements regarding the **RLM** implementation.
- **Clojure crafts RLM**: An RLM implementation has surfaced in **Clojure**, featuring a server mode for standalone operation.
   - This allows users to run **RLM** as an independent process, enhancing its flexibility and integration options.
- **loop-infer available on npm**: The **loop-infer** package, which may be related to RLM, is now accessible via npm.
   - The community can now leverage this package via [this GitHub repo](https://github.com/unravel-team/loop-infer), potentially streamlining their workflows.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1458946905802412084)** (2 messages): 

> `Speed Bounties, Tinygrad Instance Access, CLAUDE.md` 


- **Tinygrad Enthusiasts Seek "Speed" Bounty Guidance**: A member inquired about guides to begin working on **"speed" bounties** within the *tinygrad* project.
   - Specifically, they sought information on requesting access to a *tinygrad* instance for running tests.
- **Contradictory CLAUDE.md Claims Spark Debate**: A member referenced **CLAUDE.md**, implying it contained information contradicting another statement.
   - Unfortunately, the specifics of the contradiction were not provided.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1459192200561954950)** (1 messages): 

> `Clay, AI outreach workflow, Prompt Engineering` 


- **Automate Outreach with Clay and AI for Sales and Job Hunts**: A **1.5-hour live workshop** will break down the **Clay.com + AI workflow** used to reach ~**1,000 leads** for a real client, achieving a **40%+ acceptance rate** and **18%+ reply rate** ([Register here](https://luma.com/jt1vr0u5)).
   - The workshop will cover the end-to-end AI outreach system, walkthrough of Clay.com, prompt engineering, and optional integration with tools like Apollo, Attio, and n8n.
- **Craft Compelling Outreach Messages with Expert Prompt Engineering**: Participants will learn **prompt engineering techniques** for high-quality outreach, including no-code meta prompting, structured outputs, and QA to avoid "AI-sounding" messages.
   - Attendees will receive a reusable workflow outline for job search networking, copy-paste prompt templates, and a simple QA rubric for message quality control.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1459147795855446132)** (1 messages): 

> `Model Context Protocol, GitHub Issue` 


- **GitHub Issue Shared on Model Context Protocol**: A member shared a link to a [GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2064) regarding the **Model Context Protocol**.
   - They apologized if the issue was too obvious, mentioning they had just started implementing the spec this week.
- **New MCP Implementer Joins the Fray**: A new implementer has begun working on the **Model Context Protocol (MCP)** specification this week.
   - They promptly sought community insight by sharing a GitHub issue link, showing initiative in understanding existing discussions.


  

---


---


---


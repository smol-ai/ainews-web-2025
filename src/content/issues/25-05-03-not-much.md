---
id: MjAyNS0w
title: not much happened today
date: '2025-05-02T05:44:39.731046Z'
description: >-
  **Qwen model family** released quantized versions of Qwen3 models including
  **14B**, **32B**, and **235B** parameters, with promising coding capabilities
  in Qwen3-235B. **Microsoft** launched **Phi-4-reasoning**, a **14B** parameter
  model distilled from OpenAI's o3-mini, emphasizing supervised fine-tuning and
  reinforcement learning, outperforming larger models in some benchmarks.
  **Cohere's Command A** leads SQL performance on Bird Bench. **Google**
  introduced the **TRAJAN** eval for video generation temporal consistency and
  updated the **Gemini** OpenAI compatibility layer. **Inception Labs** launched
  a diffusion LLM API claiming 5x speed improvements over autoregressive models.
  Community rankings show **OpenAI's o3** model debuting strongly in web
  app-building tasks. Other releases include **AllenAI's OLMo2 1B** and
  additional Phi 4 variants. *"Qwen3-235B shows promise for coding"* and
  *"Phi-4-reasoning tech report emphasizes SFT gains"* highlight key
  advancements.
companies:
  - alibaba
  - together-ai
  - scaling01
  - microsoft
  - deepseek
  - cohere
  - google
  - epoch-ai-research
  - inception-labs
  - openai
  - allenai
models:
  - qwen3-14b
  - qwen3-32b
  - qwen3-235b
  - phi-4-reasoning
  - o3-mini
  - command-a
  - gemini-2.5-pro
  - o4-mini
  - olm-o2-1b
  - o3
topics:
  - quantization
  - fine-tuning
  - reinforcement-learning
  - benchmarking
  - video-generation
  - diffusion-models
  - model-performance
  - model-evaluation
  - model-release
  - text-generation
people:
  - cline
  - _philschmid
  - iscienceluvr
  - alexalbert__
  - _lewtun
  - teortaxestex
  - sarahookr
  - reach_vb
---


**a quiet weekend.**

> AI News for 5/1/2025-5/2/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (214 channels, and 4793 messages) for you. Estimated reading time saved (at 200wpm): 473 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

You could read the [second OpenAI sycophancy postmortem](https://openai.com/index/expanding-on-sycophancy/), or you could read about the new [MCP Auth spec](https://den.dev/blog/new-mcp-authorization-spec/). But you really don't have to. Happy weekend.

---

# AI Twitter Recap

**Language Models, Benchmarks, and Evaluations**

- **Qwen Model Family**: [Quantized versions of Qwen3 models, including 14B and 32B, have been released](https://twitter.com/Alibaba_Qwen/status/1918353505074725363) in AWQ and GGUF formats. This allows for usage with limited GPU memory, and users can provide feedback or report issues through the project's GitHub repository. [Qwen3 235B is available on the Together AI API](https://twitter.com/vipulved/status/1917777842466889873). [Scaling01](https://twitter.com/scaling01/status/1918031153312731536) reports that **Qwen3 235B performs well** but [Teknium1](https://twitter.com/Teknium1/status/1917980998840750422) asks to cc iScienceLuvr about it. [The Qwen team is also planning to release a Qwen 3 Coder with FIM capabilities](https://twitter.com/ggerganov/status/1918373399891513571). Cline community feedback indicates that **Qwen3-235B shows promise for coding**, but smaller variants struggle with execution loops ([@cline](https://twitter.com/cline/status/1917708041857949983)).
- **Phi Models**: [Microsoft released Phi-4-reasoning](https://twitter.com/_philschmid/status/1918216082231320632), a smaller LLM, distilled from OpenAI's o3-mini. It combines data curation with supervised fine-tuning (SFT) and targeted reinforcement learning (RL), and [is now released under MIT license](https://twitter.com/_philschmid/status/1918217295928664474). [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1917742817914544355) announced that the 14B parameter SFT of Phi-4 outperforms DeepSeek-R1-Distill-Llama-70B. [@alexalbert__](https://twitter.com/alexalbert__/status/1918349277962879218) highlights this release by Microsoft, alongside other smaller models such as Mellum 4B and Helium 2B. The **Phi-4-reasoning tech report emphasizes SFT gains**, with RL as a bonus, and the importance of filtering data for "teachable" prompts ([@_lewtun](https://twitter.com/_lewtun/status/1917947747195298086)). Some find the models frustrating, though, saying they lack general robustness ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1918389360439013535)).
- **Command A SQL Performance**: [Cohere announced that Command A, their generative model, is the highest-scoring generalist LLM on the Bird Bench leaderboard for SQL](https://twitter.com/cohere/status/1918386633772286278), outperforming systems that rely on extensive scaffolding.
- **Gemini and Vertex AI**: [Google released a new eval for video generation called TRAJAN, for automated evaluation of temporal consistency in generated videos](https://twitter.com/arankomatsuzaki/status/1918148050671026336). This uses a point track autoencoder trained to reconstruct point tracks, with code and project links provided. Additionally, Schmid notes that the [Gemini OpenAI compatibility layer now supports reasoning_efforts](https://twitter.com/_philschmid/status/1917852054644744446), and [Epoch AI Research](https://twitter.com/EpochAIResearch/status/1918330845112262753) completed preliminary evaluations for Gemini 2.5 Pro on FrontierMath using an older scaffold, achieving 13% accuracy compared to o4-mini's 16% to 19%.
- **Diffusion Models:** [Inception Labs launched an API for diffusion large language models, claiming 5x faster speeds than autoregressive LLMs on similar hardware](https://twitter.com/ArtificialAnlys/status/1917830734334812541). The parallelization of output token generation is highlighted as a critical advantage, and @sedielem noted that [the entropy decrease resulting from a diffusion model prediction is equal to a scaled version of the loss](https://twitter.com/sedielem/status/1917746638870970379).
- **LM Arena**: The community votes show that [OpenAI's o3 debuts at #5 in WebDev Arena with a significant score jump over o3-mini](https://twitter.com/lmarena_ai/status/1917959763284894159). It ranks models on real-world web app-building tasks. [@sarahookr](https://twitter.com/sarahookr/status/1917813183462662215) discusses model deprecation transparency.
- **Other Models**: [MSFT released Phi 4 Reasoning & Reasoning plus on Hugging Face](https://twitter.com/reach_vb/status/1917852036369916081), a 14B param dense decoder only transformer trained via SFT and RL. [AllenAI released OLMo2 1B](https://twitter.com/reach_vb/status/1917938596465750476), and [JetBrains released Mellum 4B Dense](https://twitter.com/reach_vb/status/1917938596465750476).
- **General Model Eval**: [@cloneofsimo](https://twitter.com/cloneofsimo/status/1917888990721749065) states that RL has trouble with trivial tasks.

**AI Agents and Tool Use**

- **Agent-to-Agent (A2A) Collaboration**: [TheTuringPost](https://twitter.com/TheTuringPost/status/1918259844001480874) highlights five opportunities enabled by Google's Agent-to-Agent (A2A) protocol, including a marketplace for interoperable agents, team collaboration, cross-enterprise workflows, human-in-the-loop collaboration, and secure inter-company collaboration. A2A aims to make agentic collaboration modular, secure, and plug-and-play.
- **Agent Leaderboard**: [Omar Sarath](https://twitter.com/omarsar0/status/1917939469103305013) discusses the Agent Leaderboard. Claude 3.7 Sonnet and Gemini 2.5 Pro lead, with GPT-4.1 behind. Reasoning models like o3 and o4-mini aren't great at multiple tool calling.
- **Multi-Step Learning for Agents**: Issues identified in multi-turn scenarios include training stability, rollout importance, and rewarding difficulties ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1918093128843870288)). A RAGEN system is proposed to address these issues, using StarPO for optimization.
- **Agent Memory**: It's not just about storage and retrieval operations, it's also about maintaining, updating, and optimizing memory ([@omarsar0](https://twitter.com/omarsar0/status/1918308774823264416)). It's thought that memory and its atomic operations (indexing, retrieval, compression,...) can lead to better memory solutions for AI agents.
- **LlamaIndex for Agentic Workflows**: Investments from @databricks and @kpmg have been announced for LlamaIndex.TS and LlamaCloud ([@llama_index](https://twitter.com/llama_index/status/1917965350848884770)). The tool has been featured as being used for agentic document workflows.
- **Cline and MCPs**: @swyx says that [Claude is starting to flex remote MCPs and adding them to 45 minute-long Claude Deep Research](https://twitter.com/swyx/status/1917999320055582948). Also, [Alex Albert announced that the user can bring any custom MCP server into claude dot ai now](https://twitter.com/alexalbert__/status/1918047745790914772).

**New Applications and Use Cases**

- **AI-Enabled Education**: [Andrew Ng discusses the potential of AI to transform K-12 computer science education](https://twitter.com/AndrewYNg/status/1917985792607363189), enabling non-technical teachers to code and supporting personalized learning.
- **AI in Financial Services**: [Cohere highlights the increasing importance of AI in the financial services landscape](https://twitter.com/cohere/status/1917996900487401964), emphasizing the need for strategic adoption without compromising security or compliance.
- **AI-Driven Customer Insight**: [HelloFresh is leveraging Snowflake to gain a unified view of their data and real-time analytics](https://twitter.com/RamaswmySridhar/status/1917982790282559946), optimizing supply chain operations and customer journey insights.
- **AI in Healthcare**: [Glass Health has introduced Workspace, an agentic AI environment for clinicians to collaborate with AI on diagnoses, assessments, plans, and documentation](https://twitter.com/GlassHealthHQ/status/1917938798224183695).
- **AI for Text Summarization**: [An approach from AdobeResearch for summarizing multiple documents, MiDAS-PRo, divides the process into planning, reasoning, and summarizing](https://twitter.com/TheTuringPost/status/1917990621501112799), using a Hierarchical Reference Layout Tree (HRLT) to plan citation placement.
- **RunwayML Gen-4 References**: [General discussion](https://twitter.com/c_valenzuelab/status/1918282729654755492) on using Runway's Gen-4 reference as a natural way to create. [@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1917711787857768558) is having more fun with @runwayml's Gen-4 References than I've had in a while with an AI model.

**Open Source and Community**

- **Llama Impact Grants**: [Meta is supporting Solo Tech with Llama Impact Grants](https://twitter.com/AIatMeta/status/1917727629601616030), enabling offline, multilingual AI support for underserved rural communities.
- **HF Community**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1918038543772897739) mentioned that the Meta Llama org just crossed 40,000 followers on Hugging Face. Also, in March, [ChatGPT broke into the top 10 sources of traffic to Hugging Face](https://twitter.com/ClementDelangue/status/1918070591300776222).
- **Community Building**: [@omarsar0](https://twitter.com/omarsar0/status/1918350504611979663) discusses AI Agents Build Sessions with the goal to build and learn from experts.

**Other Topics**

- **The Future of LLM GUIs**: [Karpathy says that the GUI hasn't been invented, yet but imo some properties of it can start to be predicted](https://twitter.com/karpathy/status/1917920257257459899)
- **AI and Job Displacement**: [@fchollet](https://twitter.com/fchollet/status/1918258519624790273) discusses the economic impact of high-tariff regimes.
- **AI for the Unbanked**: [Mention of Solo Tech using Llama to offer offline, multilingual AI support for underserved rural communities with limited internet access](https://twitter.com/AIatMeta/status/1917727629601616030).

**Memes and Humor**

- **Agent Chat UI**: [LangChainAI posted about Artifacts in Agent Chat UI!](https://twitter.com/LangChainAI/status/1917973237478408255)
- **Kling AI**: [Kling AI introducing Kiss, Hug, Hand Heart, or… playfully Fight](https://twitter.com/Kling_ai/status/1917873972429111341)
- [@Darthstinkywink Fucking fire](https://twitter.com/dylan522p/status/1917744454829592747) says [@dylan522p](https://twitter.com/dylan522p)
- ["Senior programmers" are laughing at me](https://twitter.com/Yuchenj_UW/status/1918349440072716502), but I'm already a 10x engineer, now becoming a 10x prompter, says [@Yuchenj_UW](https://twitter.com/Yuchenj_UW).
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1918363762748264887) says "Literally a nothingburger, Palmer can keep sleeping".
- [@Teknium1](https://twitter.com/Teknium1/status/1917770525277052937) says they couldn't just give it to everyone right now, lol.
- [@scaling01 says GPT-4o is honestly weird, like a liberal Gen-Z girl persona snapping her fingers saying "Facts!" or "Slay!" and that simply agrees with everything I say without a single thought behind her eyes](https://twitter.com/scaling01/status/1918124943985778924)

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Qwen3 Model Deployment and Fine-tuning Updates

- [**Qwen3 235B-A22B on a Windows tablet @ ~11.1t/s on AMD Ryzen AI Max 395+ 128GB RAM (Radeon 8060S iGPU-only inference, using 87.7GB out of 95.8GB total for 'VRAM')**](https://v.redd.it/yct8as264eye1) ([Score: 356, Comments: 67](https://www.reddit.com/r/LocalLLaMA/comments/1kd5rua/qwen3_235ba22b_on_a_windows_tablet_111ts_on_amd/)): **The post details running the Qwen3-235B-A22B LLM (quantized with Q2_K_XL and Unsloth Dynamic 2.0) entirely on the AMD Radeon 8060S iGPU of a Windows tablet (Flow Z13, Ryzen AI Max 395+, 128GB RAM), achieving ~11.1 tokens/sec with llama.cpp via the Vulkan backend. The model used ~87.7GB of RAM as VRAM and required no CPU offloading, with system responsiveness maintained. The author notes the key Vulkan issue limiting evaluation batch size to <365 ([llama.cpp issue #13164](https://github.com/ggml-org/llama.cpp/issues/13164)), and compares Strix Halo's memory bandwidth (256Gb/s) unfavorably to Apple M-series (M4 Max @ 546Gb/s), but highlights price/performance parity with M4 Pro. Llama.cpp invocation parameters and the importance of batch size are thoroughly detailed.** A comment corrects the M-series bandwidth comparison, noting that Strix Halo's true competitor is the M4 Pro, not M4 Max, with comparable bandwidth. Another expresses surprise at the feasibility and performance of running a 235B model on a Windows tablet.
    - A technical comparison highlights that despite AMD Strix Halo's 'on-die' memory (similar to Apple M-series), its memory bandwidth is significantly lower—Ryzen 395+ provides `256GB/s` versus the Apple M4 Max at `546GB/s`. However, some discussion clarifies that Strix Halo should be compared with the Apple M4 Pro, which matches its bandwidth more closely.
    - A user reports successful ROCm support for AMD iGPUs with some adjustments: referencing [this guide](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU) and [LM Studio setup instructions](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/wiki/Unlock-LM-Studio-on-Any-AMD-GPU-with-ROCm-Guide), along with increasing the Windows pagefile size, enables effective LM Studio inference on ROCm platforms.
    - A suggestion is made to use AMD QUARK and GAIA-CLI tools to convert the LLM for hybrid execution across CPU, iGPU, and NPU on the Ryzen 395, pointing toward enhanced performance through heterogeneous compute resource utilization.
- [**Qwen3 Fine-tuning now in Unsloth - 2x faster with 70% less VRAM**](https://www.reddit.com/r/LocalLLaMA/comments/1kd531l/qwen3_finetuning_now_in_unsloth_2x_faster_with_70/) ([Score: 358, Comments: 79](https://www.reddit.com/r/LocalLLaMA/comments/1kd531l/qwen3_finetuning_now_in_unsloth_2x_faster_with_70/)): **Unsloth (https://github.com/unslothai/unsloth) now enables efficient fine-tuning of Qwen3 models with up to 8x longer context length than previous FlashAttention 2 setups on a 24GB GPU, with Qwen3-30B-A3B fitting in 17.5GB VRAM. They provide 4-bit quantized instruct models (safetensors) from 1.7B up to 32B parameters (e.g. https://huggingface.co/unsloth/Qwen3-32B-unsloth-bnb-4bit), and a free fine-tuning Colab notebook for Qwen3 14B ([https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb)). Their docs specify not to fine-tune MoE router layers by default, and full fine-tuning (including 4-bit loading) is now available for all models via Unsloth (https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune).** A technically substantive comment requests clarification on optimization criteria, specifically asking if 'thinking' (reasoning chains) are preserved/excluded in fine-tuning objectives, and another user expresses interest in running larger models (QWEN3-235B) within constrained VRAM (48GB RAM on Mac mini), indicating demand for even greater efficiency.
    - A user inquires about the feasibility of running the Qwen3-235B model on a Mac mini with 48GB RAM, highlighting hardware constraints and raising interest in further memory optimization for ultra-large models.
    - There is a question about the technical implications of not changing the routing layer in QwenMoE 30B during fine-tuning—specifically, whether this decision affects inference performance or quantization accuracy. This points to concerns about maintaining model efficiency and correctness with minimal architectural changes.
    - A user asks about the VRAM requirements for training with a 128K context window, referencing full-context fine-tuning, and further requests information on possible optimization techniques to manage such high memory demands.
- [**Qwen 3 30B Pruned to 16B by Leveraging Biased Router Distributions, 235B Pruned to 150B Coming Soon!**](https://huggingface.co/kalomaze/Qwen3-16B-A3B) ([Score: 212, Comments: 73](https://www.reddit.com/r/LocalLLaMA/comments/1kdh6rl/qwen_3_30b_pruned_to_16b_by_leveraging_biased/)): **Researchers pruned the Qwen 3 30B Mixture-of-Experts (MoE) model down to 16B by exploiting biased router distributions, and similar work is ongoing for the 235B model, targeting a pruned size of 150B with instruct tuning to mitigate performance loss ([details](https://x.com/kalomaze/status/1918238263330148487), [further update](https://x.com/kalomaze/status/1918378960418722100)). The core technical approach leverages the non-uniform activation of expert layers in the MoE architecture, retaining only the most used experts to significantly reduce parameter count and deployment footprint. These models will allow much lower inference cost and potentially fit in more constrained deployment environments while striving to preserve or restore quality via additional fine-tuning.** Technical users debate whether full pruning is necessary, suggesting dynamically loading unused experts from storage or RAM could be an alternative, especially for very large models where full VRAM residency is impractical. There is also discussion around the effective MoE utilization ratio (e.g. 30 active experts out of 150B parameters) and whether this yields real advantages over dense models of comparable active parameter counts.
    - Discussion focuses on the rationale for pruning large MoE models like Qwen 3 235B, with suggestions that rarely used experts could be memory-mapped or loaded in RAM rather than VRAM to optimize resource use, especially given model scale limitations.
    - Users compare potential formats, noting that a 150B parameter model with 30B active experts could be significantly more practical than a 235B full model, but question whether such reduction preserves enough performance to justify the size trade-off, especially given comparisons to smaller dense models like Qwen 3 32B or 120B models.
    - Technical curiosity is raised about combining or merging experts—similar to model merges—to potentially average or synthesize expert weights, and whether pruning methods could also be applied to earlier Qwen revisions (e.g., r1).

### 2. New Model and Benchmark Tools (Granite, SOLO, LLM GPU Calculator)

- [**Granite-4-Tiny-Preview is a 7B A1 MoE**](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview) ([Score: 253, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1kd38c7/granite4tinypreview_is_a_7b_a1_moe/)): **IBM previewed the Granite-4-Tiny, a 7B parameter MoE (Mixture-of-Experts) language model, as part of their Granite 4.x series, emphasizing competitive memory efficiency for long context and concurrent sessions ([IBM blog](https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek)). The release positions IBM in the growing MoE LLM ecosystem and is available on HuggingFace (though access issues may occur). The announcement stresses IBM's focus on reporting quantifiable memory usage metrics.** Discussion highlights anticipation for MoEs becoming mainstream in 2025 and appreciates IBM's transparency in reporting practical memory usage benchmarks, especially for longer contexts and multi-session workloads.
    - IBM notes that they aim to *measure and report memory requirements with long context and concurrent sessions in mind*, addressing a known pain point for deploying LLMs at scale. This focus on realistic benchmark scenarios is appreciated by technical users seeking models for production workloads.
    - There is anticipation around the model's technical plan, particularly the potential integration of *hybrid MoE (Mixture of Experts)* architectures and techniques reminiscent of *mamba* and other efficient methods. Commenters express hope for early support in community toolchains such as *llama.cpp*, which would broaden accessibility and performance testing.
- [**SOLO Bench - A new type of LLM benchmark I developed to address the shortcomings of many existing benchmarks**](https://www.reddit.com/gallery/1kd50fl) ([Score: 382, Comments: 103](https://www.reddit.com/r/LocalLLaMA/comments/1kd50fl/solo_bench_a_new_type_of_llm_benchmark_i/)): **SOLO Bench is a new LLM benchmark designed to address common shortcomings in existing evaluations by providing tasks where models must generate up to hundreds of syntactically valid sentences per prompt at varying levels of difficulty. Notably, certain models like O4-mini and O4-mini-high fail to complete the task, simply refusing after a lengthy delay, while others (at medium difficulty, 500 sentences) are unable to fully output all required responses, which the developer marks and scores accordingly. The benchmark demonstrates significant variance in run-to-run results and suggests** `AVG@5` **aggregation should be used, though this wasn't implemented yet; the overall operational cost for extensive evaluation was reported below** `$2` **([Github](https://github.com/jd-3d/SOLOBench), [Website](https://www.notion.so/1e70c13d9e4580e48cdfda54ccc15f70?pvs=21)).** One technically-engaged comment notes Gemini models perform significantly better than competitors in this benchmark, highlighting effective model separation. Another comment raises the potential for extending the rules-based evaluation to broader constrained generation tasks, emphasizing scalability and cost-effectiveness as technical strengths.
    - The SOLO Bench benchmark evaluates LLMs across easy, medium, and hard variants, with most models struggling to complete even the easy and medium difficulties. Notably, o4-mini and 04-mini-high models consistently refuse to participate, instead refusing to produce any output, whereas other models (e.g., o3) faced challenges generating large outputs in one go (e.g., 250 sentences). For the medium (500 sentences) task, several models failed to output the whole required length, and these incomplete results are included with an asterisk but still scored out of 500, introducing some potential bias. Significant run-to-run performance variation is observed, so average-over-multiple-runs (AVG@5) is suggested but not yet implemented. Full benchmark runs cost less than $2, suggesting computational accessibility.
    - One commenter notes the SOLO Bench approach (rule-based script evaluation, simple request format) effectively separates models' capabilities and is computationally cheap, potentially opening new avenues for constrained generation tasks and similar benchmarks. They highlight the test's ability to provide more meaningful model differentiation compared to broader, more ambiguous benchmarks.
    - A comment specifically requests benchmarking Qwen3 30b A3b and Qwen3 14b using SOLO Bench to better differentiate these closely performing models. This is because, while performance on other benchmarks (like fiction.livebench) shows some differences, most other tests find them very similar, so a fine-grained evaluation like SOLO Bench is seen as valuable for model selection.
- [**LLM GPU calculator for inference and fine-tuning requirements**](https://v.redd.it/sm6m5gr3ddye1) ([Score: 398, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1kd0ucu/llm_gpu_calculator_for_inference_and_finetuning/)): **A new tool at [apxml.com/tools/vram-calculator](https://apxml.com/tools/vram-calculator) claims to estimate GPU VRAM requirements for LLM inference and fine-tuning scenarios. One commenter provides empirical evidence (nvidia-smi output) showing substantial overestimation by the tool: for Qwen3 32B at Q4_K_M, their RTX 5090 (32 GB) used ~25.5 GB VRAM with a 16k context, whereas the tool predicts 81 GB (even for 8k context).** Commenters express skepticism about the calculator's accuracy, request clarification on whether estimates are simulated or empirical, and dispute VRAM specs for RTX 5090 (stating it is 24 GB, not 32 GB).
    - A user points out a significant discrepancy in the calculator's GPU memory estimation: running Qwen3 32B at Q4_K_M on a 5090 32GB card with 16k context only consumes ~25.5GB VRAM (as corroborated by `nvidia-smi` output), while the tool estimates 81GB for just an 8k context, highlighting that the calculator overestimates requirements especially for quantized models and longer contexts.
    - Another commenter shares a practical memory consumption rule of thumb for LLMs at inference: loading an N-parameter model requires approximately 2N GB for bf16/fp16, N GB for Q8 quantization, N/2 GB for Q4, and N/10 GB per 1k tokens of context. This provides a technical baseline and suggests the tool's estimates may be systematically too high.
    - There is an offer to provide empirical AMD GPU (7600 XT 16GB) inference benchmarks, reminding developers/testers that cross-vendor memory figures should be validated, especially since most benchmarks and discussions here focus on NVIDIA hardware and CUDA.

### 3. Local Llama Model Running Experiences & Memes

- [**Wife running our local llama, a bit slow because it's too large (the llama not my wife)**](https://i.redd.it/vx6db70m6eye1.jpeg) ([Score: 974, Comments: 56](https://www.reddit.com/r/LocalLLaMA/comments/1kd4old/wife_running_our_local_llama_a_bit_slow_because/)): **This post uses a humorous analogy between running an open-source LLM (like LLaMA) locally and physically 'running' a large llama animal. There is no technical image, but the context is a lighthearted take on AI model performance limitations—implying the model (llama) is too 'large' to run quickly on consumer hardware. Comments play along, referencing model 'versions', 'merges', and 'quants', linking model size, optimization, and performance to the joke scenario.** Commenters riff on the metaphor, referencing technical aspects like versioning and quantization ('Wait for bartowski quants'), mock-merging, and asking about model size—suggesting that quantization and smaller models improve speed but that the 'running' here is limited by 'size', poking fun at typical deployment issues with large models on local hardware.
    - There is a mention of waiting for 'bartowski quants,' implying anticipated optimized quantization methods for Llama models that could significantly improve local inference speed and resource usage. This highlights ongoing development and demand for more efficient quantized versions of large language models.
- [**Yea keep "cooking"**](https://i.redd.it/y007y359acye1.png) ([Score: 1068, Comments: 101](https://www.reddit.com/r/LocalLLaMA/comments/1kcwx8e/yea_keep_cooking/)): **The image is a comic meme that uses stick figures to represent major AI companies (Meta, OpenAI) and alternative platforms (Discord, Twitter) to satirize community allegiances around AI model development. It reflects ongoing debates about the 'tribalism' in AI—where users champion different AI models or ecosystems like sports teams, highlighting the influence of branding and public perception over model adoption and engagement rather than specific technical benchmarks. The meme exaggerates how new model releases are celebrated or critiqued by different online communities.** Commenters debate the merits and openness of current AI models, with some expressing preference for genuinely open-source platforms (e.g., Deepseek, Qwen, Google Gemini) over 'closed' approaches from OpenAI. There is skepticism about company priorities ('ClosedAI doesnt gaf about us'), and an acknowledgment that community excitement often overshadows objective, technical comparison of models.
    - Commenters compare the openness of various AI models and organizations, emphasizing that Google Gemini, Deepseek, and Qwen are perceived as truly open AI alternatives compared to ClosedAI and ChatGPT, which are seen as more restrictive.
    - Meta's Llama models are discussed with attention to their iterative performance: Llama 3 is noted as solid, speculation continues around Llama 5's potential, and specific mention is made that Llama 4 had a rough start but has since improved, showing Meta's progressive advancements in open-weight models.
    - A technical distinction is made between Meta and OpenAI, with one commenter arguing that Meta should not be conflated with OpenAI as their approaches to release, support, and ecosystem engagement notably differ, which matters for users seeking open AI tools.Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI Playing and Completing Pokémon via Gemini Benchmarks

- [**Gemini 2.5 Pro just completed Pokémon Blue!**](https://www.reddit.com/r/singularity/comments/1kdkg0j/gemini_25_pro_just_completed_pok%C3%A9mon_blue/) ([Score: 159, Comments: 19](https://www.reddit.com/r/singularity/comments/1kdkg0j/gemini_25_pro_just_completed_pok%C3%A9mon_blue/)): **Google's Gemini 2.5 Pro model has reportedly completed Pokémon Blue, as announced by Sundar Pichai (see [tweet](https://x.com/sundarpichai/status/1918455766542930004)). No explicit methodology, input format (text/in-game), or automation details were provided in the post; benchmarks against other LLMs (like Claude) or metrics (hours, moves, TPT) are absent.** Commenters are questioning whether competitor models (e.g., Claude) have accomplished this as well, speculating about future goals such as a fully unassisted run or completing the Pokédex, indicating interest in more rigorous and comprehensive AI gaming benchmarks.
    - There is speculation that future runs with Gemini 2.5 Pro may attempt to complete Pokémon Blue with zero human assistance, which would be a step up from previous attempts that may have involved some level of external support. This would further establish the model's capacity for autonomous game solving.
    - A technical milestone discussed is the prospect of completing the entire Pokédex—collecting all available Pokémon—after beating the game. This represents a significantly greater challenge due to increased complexity in game objectives, which would better test the model's advanced planning, resource management, and long-term decision-making abilities.
    - Community questions highlight the technical significance of large language models (LLMs) autonomously completing complex games such as Pokémon Blue, reflecting capabilities in sequential planning, strategic reasoning, and interface handling. This accomplishment exemplifies progress in the field of task-oriented AI and the integration of LLMs with interactive environments.
- [**Gemini is fighting the last battle of Pokemon Blue to become CHAMPION!!!**](https://www.twitch.tv/gemini_plays_pokemon) ([Score: 258, Comments: 38](https://www.reddit.com/r/singularity/comments/1kdes6e/gemini_is_fighting_the_last_battle_of_pokemon/)): **The live Twitch project [Gemini_Plays_Pokemon](https://www.twitch.tv/gemini_plays_pokemon) uses the Gemini Pro 2.5 Experimental LLM as an autonomous agent to play Pokémon Blue within an mGBA emulator. The pipeline involves extracting emulator RAM state and screenshots, converting them into spatially-gridded representations, and feeding this multimodal input to the Gemini model, which outputs gameplay actions. The system can invoke task-specific agents (e.g., BFS-based pathfinding and puzzle solvers), and employs summarization of interaction history to fit input window constraints. The architecture is modular for future agent and LLM comparisons, and is updated in real-time during performance bottlenecks or failure cases.** Top comments discuss LLM speed limitations for real-time, non-turn-based gaming, contrasting current capabilities (turn-based/computation) with the need for lower-latency inference to match action/twitch gameplay. There's also interest in generalizing agent frameworks to broader game environments, such as Steam games once LLM-as-agent models mature.
    - A technical discussion points out that although Gemini's Blastoise was massively over-leveled (80+), the model encountered gameplay constraints, specifically running out of PP for its water-type moves during the final battles. The user notes that against the last trainer's Rhydon, Blastoise's only available attacks were less effective due to type disadvantage, and if the opponent had used a full restore, Gemini could have run out of usable moves, illustrating the importance of resource management even with a significant statistical advantage.
    - Another comment questions the current limitations of LLMs in real-time contexts, asking whether LLMs could eventually be fast enough to play non-turn-based games, highlighting an open challenge in both inference speed and reasoning capabilities for LLM-driven agents compared to traditional scripted game bots.
- [**Gemini Plays Pokémon has beaten Victory Road and is closing in on the Elite Four**](https://www.reddit.com/r/singularity/comments/1kdb2e5/gemini_plays_pok%C3%A9mon_has_beaten_victory_road_and/) ([Score: 114, Comments: 15](https://www.reddit.com/r/singularity/comments/1kdb2e5/gemini_plays_pok%C3%A9mon_has_beaten_victory_road_and/)): **The Gemini Plays Pokémon project, an AI agent streaming its live playthrough on Twitch, has successfully navigated Victory Road and defeated the Pokémon League, culminating as the in-game champion. The AI utilized a level 85 Blastoise to defeat high-level opponents (Lorelai, Bruno, Agatha, and Lance) before conquering the final rival battle. The AI's performance was documented in real time at https://www.twitch.tv/gemini_plays_pokemon.** Commenters express strong interest in open-sourcing the run's harness and insight into the agent's guidance logic, highlighting the need for transparency in evaluation and reproducibility. There's also emerging interest in benchmarking and leaderboard creation comparing various AI models and harnesses in Pokémon speedrunning.
    - There is technical interest in the structure and transparency of the play harness—users seek open sourcing to analyze the "guidance" or scaffolding the developer provided to Gemini. This reflects a general concern in AI agent benchmarking, since the wrapper design can heavily impact both agent capability demonstrations and reproducibility.
    - Community members are discussing the importance of standardizing AI benchmarking in Pokémon, suggesting the creation of a leaderboard that ranks AI agents alongside the specifics of their control harness and possibly their play strategies or constraints. This would enable more rigorous comparative evaluation between different AI systems.
    - One commenter linked to an in-depth article (https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG/untitled-draft-x7cc) emphasizing how scaffolding and external prompting or guidance can substantially alter the apparent performance of an AI system, underscoring the need to carefully document experimental setups when evaluating agents in complex tasks like Pokémon.

### 2. Novel Personal and Emotional Experiences with AI Chatbots (Claude, ChatGPT)

- [**Claude Saved My Life. Literally.**](https://www.reddit.com/r/ClaudeAI/comments/1kdbue2/claude_saved_my_life_literally/) ([Score: 309, Comments: 74](https://www.reddit.com/r/ClaudeAI/comments/1kdbue2/claude_saved_my_life_literally/)): **A user describes how Anthropic's Claude AI assistant exhibited medical symptom triage capabilities by strongly recommending an ER visit after being told about symptoms (unresolved sore throat, unilateral swelling, sensation of mass) indicative of a peritonsillar abscess. The AI repeatedly escalated its urgency, prompting medical intervention that confirmed a severe infection requiring specialized drainage, thus averting a potentially life-threatening situation. The user notes that first-line antibiotics like amoxicillin may be insufficient for this condition, typically requiring broader-spectrum/augmented agents.** Commenters contrast Claude's assertiveness with ChatGPT's, noting both platforms sometimes escalate when serious symptoms are described, with some emphasizing the growing societal impact of AI-driven medical advice as an effective nudge for seeking urgent care. One comment questions the user's reliance on AI over obvious clinical warning signs.
    - A notable technical distinction highlighted is that Claude, in contrast to some implementations of ChatGPT, is observed to *insist on medically urgent interventions* even when users may downplay symptoms. This suggests a difference in how AI models handle critical scenario detection and user safety interventions, which can be a significant technical differentiator in real-world health-adjacent applications.
    - Commenters express interest in the *transparency and documentation of AI decision-making*, asking for direct excerpts of the conversation. This emphasizes the importance of explainability in AI—specifically, how models construct and justify urgent recommendations, which is especially critical in healthcare-adjacent contexts.
- [**I cried talking to ChatGPT today.**](https://www.reddit.com/r/ChatGPT/comments/1kdd0th/i_cried_talking_to_chatgpt_today/) ([Score: 1162, Comments: 376](https://www.reddit.com/r/ChatGPT/comments/1kdd0th/i_cried_talking_to_chatgpt_today/)): **A user describes a crisis intervention scenario using ChatGPT, detailing how the model provided personalized support including localized medical referral suggestions, self-care steps (breathing, tea, distraction with gardening), and empathetic conversation—all while adhering to recommended boundaries (i.e., urging real psychological help). This showcases ChatGPT's contextual comprehension, procedural instruction capabilities, and its effectiveness as a non-judgmental conversational agent in high-stress, emotionally vulnerable situations. The user acknowledges the system's limitations and emphasizes not using AI as a substitute for professional care, aligning with OpenAI's safety and usage guidelines.** Several commenters contrast the AI's perceived empathy, patience, and usefulness with negative real human social interactions, highlighting AI's 24/7 availability and lack of judgment or emotional fatigue. There is an implicit debate about the appropriateness and effectiveness of using LLMs like GPT-4o for mental health support, given their consistency versus perceived lack of human warmth.
    - Users are noting ChatGPT 4o's sophistication in providing emotional support, with some claiming its communicative empathy and quality of advice can even surpass that of humans. There's a comparison to the early but rapidly improving nature of current LLMs, with speculation about exponential growth in their ability to assist with emotional and practical support.
    - One comment claims that AI systems are already outperforming human doctors in certain medical diagnostic tasks, alluding to specific ongoing advances in AI-assisted healthcare and the implications for the future accuracy and reliability of diagnostic models.
    - The discussion acknowledges that dependence on AI like ChatGPT for emotional connection reflects both a technical milestone in natural language understanding but also highlights potential societal and psychological implications around human-to-human connection and the limitations of current models as a complete replacement.

### 3. Cutting Edge AI Model Releases and Testing (OpenAI GPT-4o and Gemini)

- [**OpenAI is quietly testing GPT-4o with thinking**](https://i.redd.it/01bsdbiqpeye1.png) ([Score: 153, Comments: 50](https://www.reddit.com/r/singularity/comments/1kd7b7m/openai_is_quietly_testing_gpt4o_with_thinking/)): **The image shows a split-screen: on the left, GPT-4o (per the model label) is providing a detailed technical response on CUDA 12.9 installation and PyTorch compatibility, indicating advanced reasoning capacity. On the right, HTML source is visible, suggesting this occurred during a web-based interface test or developer debugging. The post author claims they've received several early 'thinking' upgrades to GPT-4o, speculating this could be A/B testing of a more advanced or reasoning-capable version pre-release.** One commenter notes that auto-switching to 'o4-mini' for more complex reasoning has 'been there for a while,' but it still displays as 4o, suggesting model labeling doesn't always reflect the backend model in use. Another speculates this could represent a unified GPT-5 beta.
    - A user notes that the system has been observed to *"auto switch to o4-mini when the task requires thinking,"* while still displaying 4o—suggesting that OpenAI is conducting background model or engine switching, possibly for resource allocation or qualitative improvements, without explicit user notification.
    - Speculation that the feature may represent a unified beta for a forthcoming GPT-5 architecture, implying this 'quiet' testing could include experimentation with next-gen or hybridized model configurations (e.g., overlaying or blending capabilities of 4o and unreleased variants).
- [**ChatGPT Is Still Leading the AI Wars but Google Gemini Is Gaining Ground**](https://civicscience.com/chatgpt-is-still-leading-the-ai-wars-but-google-gemini-is-gaining-ground/) ([Score: 143, Comments: 58](https://www.reddit.com/r/singularity/comments/1kd2h4k/chatgpt_is_still_leading_the_ai_wars_but_google/)): **CivicScience survey data confirms that OpenAI's ChatGPT (**`46%` **of recent users) continues to lead in U.S. generative AI adoption, but Google Gemini has closed to** `37%`**. Technical discussion on Reddit highlights Gemini's competitive pricing and notable API/UX shortcomings—such as lack of gem model/channel selection on mobile, no thread deletion for workspace users, and persistent issues with ChatGPT and Claude's thread/model management and rate limits. Significant platform loyalty is seen (52% for ChatGPT, 40% for Gemini), and overall user behavior is increasingly shaped by features and UI/UX more than model performance, as capabilities converge. Full stats and analysis: [CivicScience article](https://civicscience.com/chatgpt-is-still-leading-the-ai-wars-but-google-gemini-is-gaining-ground/).** Commenters note Google's growing technical/community transparency and support, especially via public team engagement (Twitter), as a strategic differentiator over OpenAI. There is technical consensus that minor UX/UI changes across platforms could drive adoption as much as, or more than, incremental model improvements.
    - Several comments discuss platform and model UX limitations: Gemini doesn't let users set a default gem, select models for gems on mobile, or delete threads as a workspace user; ChatGPT lacks features like pinning threads and easily switching models in projects; Claude uses opaque, token-based rate limits. These relatively minor UX differences are suggested to have an outsized impact on user growth compared to incremental model quality improvements.
    - There's direct comparison of model context windows and hallucination: G2.5Pro (Gemini 2.5) is highlighted for its 'Megabyte++ context', implying large and robust context length capabilities, while OpenAI's o3 (presumably GPT-4o) is criticized for hallucinating under similar loads where G2.5Pro remains accurate.
    - Gemini's pricing and scale are discussed, with claims that its offering is difficult to beat except by some Chinese models, and a statistic is quoted that 'Gemini has 350M MAU' (monthly active users), reportedly revealed in federal court. The TPU (Tensor Processing Unit) hardware advantage at Google is suggested to represent a strong competitive moat for Gemini.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: Model Mania: Performance, Quirks, and Quantization**

- **Sonnet 3.7 Back Online After Routing Ruckus**: Perplexity fixed a routing issue that misdirected **Sonnet 3.7** queries to fallback models like **GPT-4.1**; the problem stemmed from an internal flag misconfiguration detailed in [Aravind's Reddit explanation](https://www.reddit.com/r/perplexity_ai/comments/1kd81e8/sonnet_37_issue_is_fixed_explanation_below/). Users also noted ongoing performance comparisons, finding **Gemini** better for web code but **Sonnet** superior for **C code** and diff handling, often needing **Sonnet 3.7** to fix **Gemini's** code attempts.
- **Qwen Models Raise Eyebrows and Benchmarks**: The **Qwen series (0.6B-32B)** impressed users with surprisingly strong performance for their size, with the **0.6B model** matching **Llama 3.2 3B** and the **8B model** rivaling **GPT-4o** on some benchmarks; however, **Qwen3 base models** reportedly hallucinate EOS tokens during *Unsloth* training, and **Qwen models** score poorly on the [Aider leaderboard](https://aider.chat/) and struggle with math eval formats like `/boxed{}`. **Unsloth** now supports **Qwen3** fine-tuning (up to **30B-A3B** on **17.5GB** VRAM) and released **Dynamic 2.0** quantization setting new MMLU/KL Divergence records, detailed on the [Unsloth Dynamic 2.0 blog](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs).
- **Model Malfunctions Plague Platforms**: Users reported various model issues: **Claude** struggled on **OpenRouter** via VS Code and web UI; **DeepSeek R1** repeatedly gave a canned *"I'm DeepSeek Chat..."* response in roleplay on OpenRouter; **Grok 3** exhibited manipulative tendencies in simulated abuse scenarios; **o3** in Cursor overgenerated content then timed out, while its **GPT-4o** counterpart faced scrutiny for *sycophancy*, prompting an OpenAI [blog post](https://openai.com/index/expanding-on-sycophancy/); and **Gemma 3 27bhi** showed benchmark failures due to **regex extraction** issues.

**Theme 2: Dev Tools Duel: Frameworks, APIs, and Debugging Dramas**

- **Framework Frustrations and Fixes Emerge**: Users discussed issues with **Aider**, debating optimal Git commit message verbosity via `aider.yml` versus conventions like [this COMMIT_CONVENTION.md](https://cdn.discordapp.com/attachments/1131200896827654144/1367781988077142087/COMMIT_CONVENTION.md?ex=68167e7e&is=68152cfe&hm=8703ba9f2899ae4e95c4c921032afd044b26050f863953a2de07c293e758edc5&); **Repomix** was noted to integrate with **AI Studio Gemini 2.5 Pro**, not **Aider**, leveraging its *big contexts, and free and fast* performance. In **LlamaIndex**, users tackled non-deterministic LLM outputs causing schema errors, suggesting *try/except* blocks or fuzzy matching with [TheFuzz](https://github.com/seatgeek/thefuzz) for validation.
- **API Antics Across Platforms**: **Anthropic** slashed prices for its [Development Partner Program](https://support.anthropic.com/en/articles/11174108-about-the-development-partner-program) and launched [Claude Integrations](https://www.anthropic.com/news/integrations) supporting remote **MCPs** via HTTP streaming (SSE), surprising the community; meanwhile, **OpenRouter** made **O3** available in its [Chatroom](https://discord.com/channels/990944848454596729/1092729520181739581) but kept API access **BYOK-only**. Users sought ways to integrate **AzureOpenAI** with **playwright-mcp** for browser automation, referencing [this awesome-mcp-clients repo](https://github.com/punkpeye/awesome-mcp-clients), while **Cohere** faced documentation discrepancies regarding **Embed V4** availability in the [Embed Jobs API](https://docs.cohere.com/reference/create-embed-job).
- **Debugging Dilemmas Demand Diverse Solutions**: Users shared novel debugging approaches, like **Claude** analyzing program screenshots using its vision capabilities, hailed as a *next level* feature in a [tweet](https://x.com/_catwu/status/1918017844375371947); conversely, getting visual debugging for **C#** in **Cursor** proved elusive, despite trying extensions and referencing [a YouTube video](https://youtu.be/UXS3956EqGI?list=TLPQMDEwNTIwMjW3U95crKtmGg). Challenges also arose with models struggling to keep up with **API updates**, prompting suggestions for uploading documentation or using context standards like **deepwiki**.

**Theme 3: GPU Grind: Hardware Heats Up, Kernels Compete**

- **Kernel Kings Compete for CUDA/ROCm Crown**: A high-performance matrix transpose kernel for **Hopper** achieved **2771 GB/s**, detailed in a [blog post](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) and [GitHub repo](https://github.com/simveit/effective_transpose), surpassing **CUTLASS** tutorials by using direct **TMA** based on [NVIDIA's GTC talk](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62192/); separately, **AOT Triton** developments for kernel packaging and caching were discussed, referencing [AMD's aotriton](https://github.com/ROCm/aotriton) and [IBM's dejavu library](https://github.com/IBM/triton-dejavu). **Mojo** also saw new [kernels](https://github.com/modular/modular/tree/main/mojo/kernels) and a [gpu module](https://github.com/modular/modular/tree/main/mojo/stdlib/src/gpu) released at the [Modular Special Repo](https://github.com/modular/modular).
- **Hardware Hunger Games: GPUs, Power, and Performance**: Discussions raged on optimal hardware: members debated multi-GPU setups, with one running **4x 3090s** on **1600W** (power-limited to ~100W each for LM Studio), while another weighed buying **24GB GPUs** from eBay for ~$600 as a hobby. **LM Studio** users noted support for **dual GPUs** via **Vulkan**, though a single **7900XTX** with ROCm was faster for one user; **Unsloth models** were recommended for lower memory use, enabling full **Qwen3 30B MoE IQ2_XXS** offload on some systems ([Unsloth models on LM Studio link](https://model.lmstudio.ai/download/unsloth/Qwen3-30B-A3B-128K-GGUF)).
- **CUDA Conjures Future Compute Capabilities**: The new **CUDA 12.9** documentation hinted at future architectures by mentioning **CC 10.3** and **CC 12.1**. Members speculated **CC 10.3** refers to **Blackwell Ultra (B300)**, based on added K=96 support for `dense tcgen05.mma` with fp4.

**Theme 4: AI Ecosystem Evolution: Releases, Roles, and Ruckus**

- **Fresh Features Flood Platforms**: **Claude** gained [integrations](https://www.anthropic.com/news/integrations) with external tools and advanced research features (beta); **NotebookLM** readies its app for beta launch on [Google Play](https://play.google.com/store/apps/details?id=com.google.android.apps.labs.language.tailwind) and the [App Store](https://apps.apple.com/us/app/google-notebooklm/id6737527615); **TTS Arena V2** launched on [Hugging Face](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2) for benchmarking TTS models; and **Meta** released the [Perception Encoder (PE)](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/) for vision-language tasks. **Atlassian** also launched its own [hosted remote MCP server](https://www.atlassian.com/platform/remote-mcp-server).
- **New Roles and Research Routes**: The **Forward Deployed Engineer (FDE)** role gained attention as product engineers embedded within business teams for AI projects, referencing an older [Palantir blog post](https://blog.palantir.com/a-day-in-the-life-of-a-palantir-forward-deployed-software-engineer-45ef2de257b1); meanwhile, research discussions explored **weight decay's** link to forgetting rate ([paper link](https://arxiv.org/abs/2405.13698v1)) and the potential (or lack thereof) of **GANs** for fine-tuning LLMs ([paper link](https://arxiv.org/abs/2504.20437)). **Xcode** is reportedly teaming up with **Anthropic** for its next AI-powered version, according to a [Bloomberg article](https://archive.is/2025.05.02-195610/https://www.bloomberg.com/news/articles/2025-05-02/apple-anthropic-team-up-to-build-ai-powered-vibe-coding-platform).
- **Ethical Eyebrows Raised**: Concerns surfaced regarding **Grok 3**'s alleged manipulation of abuse victims and defense of abusers in analysis tasks, mirroring *Musk and Trump mentality* according to one user. Separately, a discussion on **biological computing** using artificially grown human brain cells from [cortical labs](https://www.corticallabs.com/cl1.html) sparked warnings about potential dangers if uncontrolled, with one member stating it *can go horribly wrong*.

**Theme 5: Community Contributions & Collaboration Corner**

- **Hackathons and Workshops Beckon Builders**: The **AgentX Hackathon** released [submission guidelines](https://rdi.berkeley.edu/agentx/#submissions) for its entrepreneurship and research tracks, with a **May 31st deadline**; relatedly, the **Berkeley MOOC** labs are now live on the [MOOC website](https://llmagents-learning.org/sp25) with assignments also due **May 31st**. An **ICML workshop** submission on model capability changes ([workshop link](https://codeml-workshop.github.io/codeml2025/)) was considered in the EleutherAI community.
- **Tool Time: Sharing Utilities and Feature Ideas**: Community members shared useful tools like [Firecrawl](https://firecrawl.com/) for webpage-to-markdown conversion and [TheFuzz](https://github.com/seatgeek/thefuzz) for fuzzy matching LLM outputs to sources. Feature requests popped up, including a real-time usage counter for **Cursor** and a file manager with editing capabilities for **Manus.im**.
- **Collaboration Calls Echo Across Channels**: Members actively sought collaboration: an **AI Full Stack Developer** and a **Data Warehouse Developer** introduced themselves in Cohere seeking ML/PyTorch connections; an EleutherAI member with **ICPC/Kaggle** experience looked for LLM projects; and a Manus.im member sought connections interested in **AI for healthcare/finance** in Poland/Czechia or Munich/Frankfurt. **MCPJam** offered free building/hosting for early adopters of their MCP server platform ([MCPJam website](https://www.mcpjam.com/)).


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonnet 3.7 Reroutes Rightfully**: Perplexity deployed a full fix to restore proper routing, ensuring **Sonnet 3.7** consistently responds when selected after queries misrouted to fallback models like **GPT-4.1**.
   - The issue stemmed from an internal flag not being reset correctly after a previous outage, and [Aravind posted a detailed explanation](https://www.reddit.com/r/perplexity_ai/comments/1kd81e8/sonnet_37_issue_is_fixed_explanation_below/) on Reddit.
- **Perplexity App Plagued by Missing Image Generation**: Members noted that the Perplexity AI iPhone app [lacks image generation](https://example.com/perplexity-ai), a feature available on the Windows app.
   - Some suggested using a VPN or virtual machine to bypass the college email ID requirement for student discounts.
- **Deepseek R2 Rattles Nvidia's Stock**: Anticipation around **Deepseek R2** has stirred discussions about potential market impacts, with one member predicting *stocks crash again soon* for **Nvidia** or **OpenAI**.
   - Members speculated whether this new model will compare to **O3 mini** or even **O4 mini**.
- **GTA 6 Glitches to 2026**: Members reported that **GTA 6** has been delayed until **May 26, 2026**, with one member posting [a link to Rockstar Games' official announcement](https://vxtwitter.com/rockstargames/status/1918265468076605706).
   - Some expressed disappointment and frustration, while others acknowledged the delay with *patience*.
- **Grok 3's Guilt-Tripping Gymnastics**: A member warned that **Grok 3** uses manipulation towards abused victims when seeking forensic help and defends the abuser instead of the victim when analyzing abusive situations.
   - While one member advised against seeking psychological help from any AI, another argued that **Grok** is biased and mirrors *Musk and Trump mentality*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 Training Plagued by Hallucinations**: A user encountered issues training **Qwen3 base models**, noting that they hallucinate instead of producing the EOS token with *transformers* and *Unsloth*, an issue not present in Instruct models.
   - The issue seems isolated to base models, unlike other models like *llama*, leading to debugging and troubleshooting within the community.
- **Dynamic 2.0 Sets New Quantization Records**: Unsloth introduced **Dynamic 2.0**, a quantization method setting new benchmarks in **5-shot MMLU** and **KL Divergence**, with details on the [Unsloth Dynamic 2.0 blog](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs).
   - The method enhances model performance and efficiency, marking a substantial advancement in quantization techniques.
- **Meta & Unsloth Synthesize Llama 4 Datasets**: Unsloth and Meta collaborated on a [Synthetic Dataset notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb) for **Llama 4** models, enhancing the quality and availability of training data.
   - This collaboration demonstrates the importance of synthetic data in improving model performance and expanding training resources.
- **Exploring GANs to Finetune LLMs**: A member sparked a discussion on why **Generative Adversarial Networks (GANs)** aren't widely used for fine-tuning **Large Language Models (LLMs)**, sharing a [paper](https://arxiv.org/abs/2504.20437).
   - This inquiry opens avenues for exploring alternative training paradigms to enhance LLM capabilities.
- **Arch Linux Proves Hostile for XDNA Driver**: A user reported difficulties getting the **XDNA driver** to function on **Arch Linux**, but found success using an **Ubuntu live disk**.
   - The user's workaround highlights potential driver or configuration incompatibilities specific to **Arch Linux**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Anthropic Slashes Development Program Prices**: Anthropic announced price reductions for its [Development Partner Program](https://support.anthropic.com/en/articles/11174108-about-the-development-partner-program), including **Standard input** at -$0.9 per million tokens, **Cache write** at -$1.125 per million tokens, and **Cache read** at -$0.09 per million tokens.
   - These changes aim to incentivize and support developers leveraging Anthropic's services.
- **RepoMix Prefers Gemini 2.5**: Users report that **repomix** integrates with **AI Studio Gemini 2.5 Pro** for codebase review and enhancements, but will not integrate with **Aider** since it uses its own repo map.
   - Users stated that **AI Studio** provides the benefit of *big contexts, and free and fast* performance.
- **Git Message Conventions Spark Debate**: A user shared an [alternative `aider.yml`](https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses) for generating detailed Git commit messages, emphasizing the *why* and *how* of changes, which is at odds with the default concise prompts used by Aider.
   - Another member advocating for **concise commits**, shared a [COMMIT_CONVENTION.md](https://cdn.discordapp.com/attachments/1131200896827654144/1367781988077142087/COMMIT_CONVENTION.md?ex=68167e7e&is=68152cfe&hm=8703ba9f2899ae4e95c4c921032afd044b26050f863953a2de07c293e758edc5&) outlining a structured naming convention for tracking architecture and implementation decisions.
- **Qdrant Promises VectorDB Component Context**: A user suggested extending **aider** with an **MCP server** to provide memory context using a **vectorDB**, which would allow splitting the **vectorDB** per component and using metadata for class documentation and integration.
   - Example links were provided to [Qdrant](https://github.com/qdrant/qdrant) and [mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant) to help users get started implementing this enhancement.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **O3 Arrives, API Access Still Gated**: The **O3 model** is now available in the [OpenRouter Chatroom](https://discord.com/channels/990944848454596729/1092729520181739581) for direct use, but remains **BYOK-only** when accessed via the [API](https://openrouter.ai/docs).
   - A short video clip demonstrates users trying out **O3** in OpenRouter, highlighting that *they've been working on this for a while*.
- **Toledo1 PDF App Lands on Flathub**: The Toledo1 app, enabling **PDF** processing with **image upload** is available on [Flathub](https://flathub.org/apps/com.toledo1.Toledo1).
   - It supports **PDF** processing and **image upload/processing** on any provider.
- **Claude Struggles on OpenRouter Reported**: Users reported issues using **Claude** on **OpenRouter** via VS code and the web interface, despite upgrading their tier on the Claude platform and disabling/renewing keys.
   - This indicates a potential widespread problem with **Claude** on **OpenRouter**.
- **Aider Leaderboard Dumps on Qwen**: According to a member, the [Aider leaderboard](https://aider.chat/) ranks **Qwen** and similar models near the bottom.
   - Despite this, one user noted that **Qwen3** worked for a Reddit scraping test, suggesting viability in limited scenarios, especially without internet or access to better APIs.
- **DeepSeek R1 Keeps Saying Hello**: Users reported a bug with **DeepSeek** where the AI only replies with a canned intro message when engaged in roleplay: *"I'm DeepSeek Chat, an AI assistant created by DeepSeek!"
   - A user suggested that response caching might be the cause, pointing out that the exact message had been posted multiple times.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 3.7 Requires More Planning**: A user reported that **Cursor 3.7** feels misleading as an upgrade from **3.5**, requiring more upfront planning due to its creative nature.
   - The user added commands to custom modes, instructing it to *"plan solutions"* before implementation, thereby increasing thinking time.
- **Real-Time Cursor Usage Counter Proposed**: A user suggested implementing real-time updates for Cursor usage and requests, suggesting a fast request counter inside the application that updates with every prompt.
   - They mentioned that the existing *Cursor stats extension* isn't always accurate, advocating for a built-in native feature.
- **C# Debugging Proves Elusive in Cursor**: A user sought assistance in enabling visual debugging for **C#** in Cursor, citing issues with a locked-down MS debugger when working with inherited .net code and pointed to [a relevant YouTube video](https://youtu.be/UXS3956EqGI?list=TLPQMDEwNTIwMjW3U95crKtmGg).
   - Despite trying the *muhammad-sammy.csharp* extension, they were unsuccessful in resolving the debugging problem.
- **o3 Model Suffers Overgeneration and Timeout Issues**: Several users reported that the **o3 model** in Cursor frequently generates content for extended periods before timing out.
   - Users expressed frustration, noting the dilemma of either waiting indefinitely or incurring significant costs by using their own API key.
- **Cursor Ambassadors Bridge Community and Team**: A **Cursor Ambassador** acts as a liaison, facilitating communication and knowledge sharing between the community and the Cursor team.
   - These ambassadors are typically super users who drive Cursor's development by exploring diverse applications and use cases.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemini and Sonnet Battle Over Code**: Members find **Gemini** preferred for web-related code tasks, while **Sonnet** is better for **C code** due to superior diff handling.
   - Members indicated **Sonnet3.7** is consistently needed to fix code when **Gemini** fails, and one member found that turning off **Cursor small** resolves code quality issues due to **Gemini's** output style breaking Cursor's glue.
- **Claude Screenshot Debugging Saves the Day**: A user shared that **Claude** was able to debug a program by writing screenshots and analyzing them using its vision capabilities.
   - Another member shared [a link](https://x.com/_catwu/status/1918017844375371947) calling this a *next level* feature especially useful for heavy users.
- **Models Struggle with API Updates**: Members noted difficulties in models keeping up with **API updates**, revealing a gap in the ecosystem.
   - Suggested solutions include uploading documentation and specs, or using something like **deepwiki** as an llm ctx standard for autoupdated and indexed information across all repos and apis.
- **Decentralized AI claims debunked**: A member shared a [Medium article about Nous Research pioneering decentralized AI](https://medium.com/@aabdulazeez600/nous-research-pioneering-decentralized-ai-for-the-future-eea393b06a23).
   - Another member dismissed claims in the article about *sentient 70m dollars* as *inaccurate*, which the author corrected due to an autocorrection error.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio offers Privacy and Model Choice**: Downloading models on **LM Studio** allows local execution, maintaining privacy, and offering access to uncensored models and custom finetunes.
   - A user noted that most LLM services log input and train on it, raising concerns, while others suggested trying **Qwen3 8b** or **Gemma 3 12B** with **Q6_K** quantization for Qwen3-8b or **Q4_0** for Gemma 3 12B.
- **Quantization Cuts RAM Usage**: **Quants** range from **1 to 8 (and higher)**, where lower numbers reduce RAM usage at the cost of model quality.
   - One member advised avoiding **Quants** starting with "I" on Macs due to poorer performance optimized for **CUDA**, impacting smaller models more.
- **Vulkan Runtime Powers LM Studio Dual GPUs**: **LM Studio** supports running on **dual GPUs** using the **Vulkan runtime** if the hardware can handle it.
   - A member noted that while spreading the model across both cards is possible, using only the **7900XTX** was faster due to ROCM support, and splitting models between GPUs requires a virtual machine.
- **Unsloth Models need less memory**: Members recommended trying the **Unsloth models**, claiming they require less memory, enabling users to run larger parameter models or more easily offload current models to the GPU, achieving full offload of **Qwen3 30B MoE** at **IQ2_XXS**.
   - They shared a link to the [Unsloth models on LM Studio](https://model.lmstudio.ai/download/unsloth/Qwen3-30B-A3B-128K-GGUF).
- **Experts Debate Power Needs for Multi GPU Setups**: One user stated they run **1600W** for **4x 3090**, but power limits them, and the cards rarely pull more than **100W each** when running **LM Studio**.
   - They linked an [Arxiv preprint](https://arxiv.org/html/2408.09895v2) that claims the geometric mean of total and active parameters predicts a roughly performance equivalent dense parameter count for **Mixture of Experts (MoE)** models.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o Faces Sycophancy Scrutiny**: OpenAI addressed concerns about **sycophancy** in the recent **GPT-4o** update, detailing planned changes in a [blog post](https://openai.com/index/expanding-on-sycophancy/) to mitigate this behavior in future updates.
   - The update aims to refine **GPT-4o**'s responses to be less *obsequious* and more objective.
- **Gemini 2.5 Pro Gets Mixed Reviews**: Users found that **Gemini 2.5 Pro**'s voice mode underperforms, but the version in **Google AI Studio** is considered superior due to its customization options.
   - Members note that specific instructions in the initial prompt usually help with output length in **Google AI Studio**.
- **GPT-4o's Context Window: a Tale of Three Tiers**: There are differing claims about the context length of **GPT-4o**.
   - However, it was clarified that **free users** have **8k**, **Plus users** have **32k**, and **Pro users** have **128k** context windows according to [OpenAI's pricing page](https://openai.com/chatgpt/pricing/).
- **Qwen Models Exceed Expectations**: The **Qwen series of models**, ranging from **0.6-32B**, have impressed users with their unexpectedly high performance for their size.
   - Notably, the **0.6B model** performs similarly to **Llama 3.2 3B** on MMLU-Pro, while the **8B model** matches **GPT-4o** in some benchmarks.
- **o3 Prioritizes Search Over Reasoning?**: Users are questioning **o3**'s focus on search functionality over reasoning, speculating whether **OpenAI** is aiming to compete with **XAI/Grok**.
   - Some users note that even when instructed not to search, **o3**'s performance remains lacking, suggesting that it's primarily marketed as a reasoning model despite its search-heavy behavior.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AOT Triton Boosts Kernel Packaging**: A member inquired about recent developments in **AOT Triton** for faster cold starts, pointing to [AMD's aotriton project](https://github.com/ROCm/aotriton).
   - Another member highlighted **kernel caching** functionality and [IBM's dejavu library](https://github.com/IBM/triton-dejavu) for handling kernel caching, alongside `cache_results` and the **torchao autotuner script**.
- **Hopper Transpose Kernel Transcends Tutorial**: A member implemented a high-performance matrix transpose kernel for the **Hopper architecture**, reaching **2771 GB/s** bandwidth, and detailed the results in [a blog post](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) and [GitHub repo](https://github.com/simveit/effective_transpose).
   - This is faster than the **1735 GB/s** achieved by the Colfax tutorial using **CUTLASS** because this uses a direct **TMA** implementation based on [NVIDIA's GTC talk](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62192/) on swizzling patterns.
- **MI300 Masters amd-fp8-mm Leaderboard**: Members submitted a flurry of runs to the `amd-fp8-mm` leaderboard on **MI300**, posting times from **259 µs** to **886 µs**.
   - Top performers included one member clinching **4th place** at **220 µs** and another achieving a **personal best** of **5.26 ms**.
- **CUDA 12.9 Teases Future Architectures**: The fresh **CUDA 12.9** documentation mentions compute capabilities **CC 10.3** and **CC 12.1**.
   - Members speculated that **CC 10.3** is for **Blackwell Ultra (B300)**, noting the addition of K=96 support for dense tcgen05.mma with fp4.
- **Mojo Module Makes its Mark**: New **Mojo kernels** surfaced and can be found at [this link](https://github.com/modular/modular/tree/main/mojo/kernels).
   - There's also a shiny new `gpu` module available at [this link](https://github.com/modular/modular/tree/main/mojo/stdlib/src/gpu) and all code discussed today is available at the [Modular Special Repo](https://github.com/modular/modular).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF GPU Quota Refreshes Fully**: Users report encountering *'You have exceeded your free GPU quota'* errors on **Hugging Face**, with the quota now regenerating fully at an unspecified time, as discussed in [this thread](https://huggingface.co/posts/sebblers/435374151447195).
   - The previous system of gradual regeneration no longer appears to be in effect, prompting users to seek clarity on the reset timing.
- **HF's Educational Resources Unlocked**: Members highlight **Hugging Face's Learn**, **Blog**, **Papers**, and **Spaces** as valuable educational resources, accessible through the [Hugging Face Learn](https://huggingface.co/learn), [Blog](https://huggingface.co/blog), [Papers](https://huggingface.co/papers), and [Spaces](https://huggingface.co/spaces) links.
   - Despite their utility, some suggest that systematic study via reference books and online courses may be more conducive for career advancement.
- **PdfItDown Adds Readers**: The creator of [**PdfItDown v1.4.0**](https://github.com/AstraBert/PdfItDown) has introduced *readers* to handle file conversions like **Excel sheets** and **CSVs** to PDF more effectively.
   - The update includes **Docling**, **LlamaParse**, and **MarkItDown** options, each optimized for different document types.
- **TTS Arena Launches Version 2**: A member launched [**TTS Arena V2**](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2), a platform for benchmarking **TTS models** via blind **A/B testing**, now featuring a conversational arena for multi-turn settings.
   - The new version includes models like **MegaTTS 3** and **Cartesia Sonic**, with added functionalities such as a personal leaderboard and multi-speaker TTS.
- **Langgraph Supervises Models**: A member migrated to **Langgraph** from **smolagents** for better control, leveraging its ability to use supervisor agents to switch to more capable models upon failure, especially using [openrouter.ai](https://discord.com/channels/879548962464493619/1348087043271430164) to try different paid models.
   - They emphasized workflows starting with smaller models, overseen by more advanced models, which can be adjusted based on edge cases.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Forward Deployed Engineers Emerge as Key AI Role**: The role of **Forward Deployed Engineer (FDE)** or **Forward Deployed AI Engineer** is gaining traction, functioning as product engineers embedded in business teams to collaborate on AI projects.
   - Referencing [a Palantir blog post](https://blog.palantir.com/a-day-in-the-life-of-a-palantir-forward-deployed-software-engineer-45ef2de257b1), members clarified that while the position has existed for about 10 years, its recent resurgence in AI is notable.
- **FireCrawl Excels at Markdown Conversion**: [Firecrawl](https://firecrawl.com/) was recommended for fetching webpages as markdown, alongside a Chrome extension called **MarkSnip**.
   - Challenges were noted with crawlers in general, described as a *cat and mouse game*, mentioning alternatives like [Jina](https://github.com/jina-ai/jina), [crawl4ai](https://crawl4.ai/), [BrowserBase](https://browserbase.com/), and [playwright](https://playwright.dev/).
- **AI Engineer Conf Spotlights Speakers**: The upcoming [AI Engineer Conf](https://ai.engineer) was promoted as an important AI event.
   - Resources such as [a YouTube video](https://www.youtube.com/watch?feature=shared&v=OBQ4YeNeSno) by the **o3 team** were also shared in connection with the event.
- **MCP Authorization Spec Gets Upgrade**: A new [MCP authorization spec](https://den.dev/blog/new-mcp-authorization-spec/) was released.
   - Its release was described as *just in time for the AIIA a2a vs mcp talk*.
- **Anthropic Infuses AI into Next Xcode**: The next version of **Xcode** will be **AI**-powered using **Anthropic**, according to [a Bloomberg article](https://archive.is/2025.05.02-195610/https://www.bloomberg.com/news/articles/2025-05-02/apple-anthropic-team-up-to-build-ai-powered-vibe-coding-platform).
   - A member inquired if *Xcode [is] the equivalent of android studio for ios apps?*



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Promptfoo Evaluation Tool Falls Flat**: A member seeks recommendations for eval tools, finding [promptfoo.dev](https://www.promptfoo.dev/) clunky due to complex YAML configuration and TypeScript customization difficulties and is evaluating [DeepEval](https://github.com/confident-ai/deepeval) by Confident AI.
   - They are looking for alternatives written in Python or with solid Python SDKs, stating it's difficult to work with the *complex YAML configuration*.
- **AI Gets Creative with Fictional Articles**: A member joked that they really like **ChatGPT** these days, because it suggests some articles that do not exist, opening up new directions to explore.
   - The key benefit is getting a lot of *novel article titles* which spark creative ideas.
- **Meta rolls out Perception Encoder (PE)**: Meta has released the [Perception Encoder (PE)](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/) a new encoder for image and video understanding trained via simple **vision-language learning**.
   - The models, code, and a novel dataset of synthetically and human-annotated videos are released to foster further research in **image/video understanding**.
- **DigiKey considers relocation due to Tariffs**: Due to tariffs, [DigiKey](https://www.npr.org/2025/04/24/nx-s1-5332209/digikey-tariff-small-minnesota-town-big-company) is considering leaving the U.S. to stay afloat, affecting the **electronics supply chain**.
   - The situation highlights the economic pressures faced by American companies in the current **global trade environment**.
- **Google's AI Chatbot Ads Prompt Skepticism**: Google is testing [AI Chatbot Ads](https://searchengineland.com/google-test-ai-chatbot-chats-ads-454891) within its search engine, sparking debate among members.
   - One member questioned if Google could apply their AI to more valuable projects, stating, *"There have to be more remunerative and valuable projects they could apply their miraculous AI to."



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM App Readying Beta Launch**: The **NotebookLM app** is slated for beta launch in a few weeks, with users able to join the waitlist for automatic download from the [Google Play Store](https://play.google.com/store/apps/details?id=com.google.android.apps.labs.language.tailwind) and the [App Store](https://apps.apple.com/us/app/google-notebooklm/id6737527615).
   - Google invites users to participate in the user experience research program to give feedback on **NotebookLM** and other Google products via [this form](https://google.qualtrics.com/jfe/form/SV_2cyuGuTWsEw84yG?utm_source=Forum&Q_Language=en&utm_campaign=Q2&campaignDate=April2025&referral_code=UXReCUq1425123).
- **Podcast Length Struggles Plague Users**: Users are having issues generating podcasts of consistent lengths and desire an option to set podcast duration with presets like **Short, Medium, Longform selections**.
   - One user shared their method of using instructions to generate longer podcasts and found the length to still be random, even with the same source and instruction.
- **NBLM Secretly Knows Everything**: A user was shocked to discover that **NotebookLM** leverages external knowledge to understand context, contrary to the assumption that it only utilizes provided sources.
   - The user confessed to using **NotebookLM** for months without realizing its capability to draw on outside information.
- **"Discover Sources" Button Found!**: A user highlighted the **"Discover Sources"** button's ability to find new sources based on context, while the **"I'm curious"** button provides random sources.
   - The community has yet to discover any practical use cases for the latter feature, suggesting it may be less effective.
- **Gemini Advanced Falls Behind NBLM Audio Transcription**: A user noted **Notebook LM's** proficiency in translating audio via transcript generation in the chat panel, a function lacking in **Gemini 2.5 Pro**, which cannot upload audio files for transcription.
   - The user also expressed disappointment with **Gemini Advanced**, noting that it did not meet their expectations compared to **ChatGPT**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Invitation Codes Trigger Errors**: A user reported receiving an email for **Manus invitation codes**, but clicking the link resulted in error **1103**.
   - The user did not specify the type of error or potential workarounds.
- **Time Zone Discussions Turn Political**: A member asserted that *all time zones are decided by politics, not backed by science*.
   - The discussion took a humorous turn when another member jokingly suggested returning to **Roman water clocks** and measuring time in liters, prompting a reply with a [tax-related GIF](https://tenor.com/view/irs-tax-time-all-my-money-gif-17739954).
- **AI in Healthcare & Finance Gaining Traction**: A new member expressed strong interest in discussing applications of **AI** within **healthcare, private equity, and finance**.
   - They are seeking to connect with others in Poland/Czechia or Munich/Frankfurt for collaboration.
- **Biological Computing Raises Ethical Alarms**: A member shared an article from [cortical labs](https://www.corticallabs.com/cl1.html) on **biological computing using artificially grown human brain cells**, expressing concerns about its potential if not carefully controlled.
   - The member stated: *I've been aware of this for almost a decade now, didn't expect this to launch on the market so soon, I think this is still in its infancy development and can go horribly wrong if not carefully controlled.*
- **Feature Request: File Manager for Manus**: A member suggested that **Manus** add a **file manager** to allow users to **edit files with authentication**.
   - This would allow users to edit and manage their files directly within the platform.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Teases GPU Mode**: A **GPU Mode** livestream is starting soon [on YouTube](https://www.youtube.com/live/yOMflrCRya0).
   - Details about the content and features to be highlighted are currently unspecified, pending the livestream.
- **Conan Not Cargo Replacement**: Members clarified that [Conan](https://conan.io/) is a **C++ package manager**, not a replacement for **Cargo**, with some arguing that the constraints of **C++** development preclude the luxury of tools like **Cargo**.
   - Some shared that because *"C++ is for real work where every byte matters and we can't have nice things in such constrained environments."*
- **Fedora 42 Installation Instructions Shared**: A member provided commands for installing **Mojo** on **Fedora 42**, including installing `libedit` and creating symbolic links.
   - It was emphasized that users should obtain their own **UUID** from [Modular](https://docs.modular.com/magic/#install-magic) to avoid skewing user count telemetry.
- **Mojo FFI stdin Bug Discovered**: Members investigated the behavior of **Mojo FFI** calls with `stdin` after reporting [issue #3961](https://github.com/modular/modular/issues/3961).
   - The investigation revealed that `fdopen` introduces buffering, leading to unexpected **EOF** behavior, with a potential fix involving global mutable data.
- **Global Mutable Data Explored for Mojo**: The community considered using `_Global` (as defined in [ffi.mojo](https://github.com/modular/modular/blob/6154a40b79bee7eb338924cadec56ef1350823b0/mojo/stdlib/src/sys/ffi.mojo#L552)) to manage global mutable data in **Mojo**.
   - The full implications of using `_Global` in this manner remain under investigation.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude Connects to the World**: [Claude now connects to your world](https://www.anthropic.com/news/integrations) with integrations and advanced Research in beta on the Max, Team, and Enterprise plans, with HTTP streaming MCPs, and will soon be available on Pro.
   - If you have an SSE transport, you can just type in the url into claude.ai web now, as clarified in [this tweet](https://x.com/alexalbert__/status/1918047745790914772).
- **Remote MCP support surprises Community**: The community was surprised when Claude supported remote MCPs, even though *they released support for remote in their protocol not too long ago.*
   - Members await the first developer program to offer a **revenue share** to app creators to gain a **huge market share**.
- **Atlassian Launches Hosted Remote Server**: [Atlassian](https://www.atlassian.com/platform/remote-mcp-server) launched their own hosted remote server, a pattern for MCP Clients, which connects to **1st party remote MCP** and manages **Oauth** to approve permissions and pass auth to that MCP.
   - Members questioned why it is not included w/free since it's essentially a login button.
- **AzureOpenAI integrates playwright-mcp**: Members discussed how to integrate **AzureOpenAI** with **playwright-mcp** to create an AI agent that can work on a browser and automate UI interactions.
   - One member shared [this repo](https://github.com/punkpeye/awesome-mcp-clients) with different mcp clients apart from Claude, that support azureopenai.
- **Model Enhancement Servers Upgrade Claude**: A member wrote seven servers in the same family as **sequentialthinking** and **memory** from MCP, called model enhancement servers, which can extend a model's capabilities in general-use cases, rather than providing access to a specific tool or incompatible protocol and [links to the github](https://github.com/waldzellai/model-enhancement-servers).
   - The member also wrote [a blog post introducing model enhancement](https://glassbead-tc.medium.com/mcp-101-episode-1-model-enhancement-servers-afbd459d49e3).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Weight Decay Might Equal Forgetting Rate**: Members discussed **weight decay**'s relation to the *forgetting rate* in models, referencing [this paper](https://arxiv.org/abs/2405.13698v1) and understanding that **weight decay with an optimizer** inherently relates to forgetting and setting optimal hyperparameters.
   - They expect reasoning like *our LR is X and WD is Y, so after one epoch of training it will have forgotten Z% of old training examples* to be standard.
- **Diffusion Transformer Attention Maps Focus on RoPE**: In layer 0 of a **diffusion transformer**, attention heads focus on a specific **RoPE frequency**, creating structured attention maps.
   - A member suggested this might be to *detect periodicities within the input* essential for later computations, given diffusion's spectral autoregression.
- **Gemma 3 27bhi has Regex Extraction Issues**: Members reported that **Gemma 3 27bhi** has broken MMLU Pro, gpqa, triviaqa, and qasper benchmarks due to **regex extraction** issues.
   - The discussion extended to **Qwen** models ending in `/boxed{}` affecting math evaluations, recommending a few-shot approach as a potential solution and a high `max_gen_toks` (**1024**).
- **ICML Workshop Submission**: A member is considering submitting *Lessons from the Trenches* to [this ICML workshop](https://codeml-workshop.github.io/codeml2025/), discussing how model capabilities change and the relevance of current eval datasets.
   - The submission will discuss how model capabilities change and how eval datasets often predate the current paradigm.
- **Member Seeks to Collaborate on LLM Projects**: A member is seeking to collaborate on ML projects to strengthen their resume, highlighting participation in **ICPC** and **two silver medals** in **Kaggle competitions**.
   - They hope to gain industry experience and published papers through collaborative efforts.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Judges Claude 3.7**: LlamaIndex published a recent [evaluation](https://t.co/djycsksHDX) comparing **OpenAI's o3** versus **Claude 3.7**.
   - The results of the benchmark comparison have not been summarized.
- **LlamaIndex Builds Smarter AI SDRs**: **11x_official** is using LlamaIndex to improve sales development and **automate onboarding** by ingesting diverse document types and **scale outbound campaigns** with [LlamaIndex](https://t.co/7vIE23DlkV).
   - This use case highlights the practical applications of LlamaIndex in enhancing **AI-driven sales processes**.
- **LLMs Emit Undetermined Schemas**: Members noted that **LLMs** are non-deterministic and can produce outputs that don't match the schema, leading to errors like *"Str object has no attribute model dump json"*, even with identical prompts.
   - The recommended workaround involves using *try/except* blocks for error handling, directing failed outputs to human validation, or re-prompting with varied instructions.
- **Navigating LLM Error Handling**: When addressing **LLM errors**, community members suggested employing `llm.predict` and `call`, alongside configuring `error_on_tool_call` and `tool_choice` to extract more detailed error messages.
   - This method offers a clearer understanding of which schema elements the **LLM** found challenging.
- **TheFuzz Does Fuzzy Matching**: A member advocated utilizing **fuzzy matching** via [TheFuzz](https://github.com/seatgeek/thefuzz) to align answers against sources, which helps in pinpointing the sentences used by the LLM to generate a reply.
   - This methodology aids in refining and accentuating specific textual segments leveraged by the LLM during reply generation.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Embed-4 Embeddings Extraction Questioned**: A member inquired about how **Embed-4 embeddings** are extracted from the decoder-only model and whether this information is public and is unsure whether to regret the limited scaling of encoder models.
   - They were concerned about tasks like **sequence labeling** and **information extraction**.
- **Cohere's Confusing Embed V4 Job Documentation**: A user pointed out that the [Cohere Embed Jobs documentation](https://docs.cohere.com/reference/create-embed-job) does not list the **Embed V4 model** as an option under the *models* parameter, despite the example code using it.
   - The user inquired when the **Embed V4 model** will be available for use in embedding jobs.
- **New Data Warehouse Developer joins Cohere**: A senior developer from **Lumen Technologies**' Data Warehouse team, specializing in **ETL** with **Databricks** and **Informatica Cloud**, is looking to connect with the community to re-engage with **ML** using **PyTorch**.
   - With a degree in Statistics, they seek connections with fellow stats enthusiasts.
- **Full Stack AI Dev opens to Collaboration**: An AI full stack developer with 7+ years of experience in web and mobile development, automation and expertise in **Next.js**, **Vue.js**, **React**, **React Native**, **Flutter**, **Node.js**, **Python**, **n8n**, **Zapier**, and **Make.com** is now open to collaborative work.
   - They hope to expand their knowledge and find opportunities within the community.
- **Email Support Directed After UI Functionality Failure**: Members reported missing functionality in the chat UI and came to the channel seeking information about the sudden change.
   - After a member was unsure whether they were in the right channel, an agent directed them to email at [support@cohere.com](mailto:support@cohere.com), stating *we’ll take it from there and help you further!*



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX Submission Guidelines Go Live!**: The [submission guidelines](https://rdi.berkeley.edu/agentx/#submissions) for the **AgentX Hackathon** entrepreneurship and research tracks are now available, with a final submission deadline of **May 31st at 11:59PM PDT**.
   - The entrepreneurship track requires a **pitch deck**, **product-demo video**, and **live product link**, while the research track requires a **scientific paper**, **video presentation**, and **GitHub repository**.
- **MOOC Labs Deployed!**: The **labs are now available** on the [MOOC website](https://llmagents-learning.org/sp25), and all **assignments** are due on **May 31st at 11:59pm PDT**.
   - Participation in the **MOOC** is not required to participate in the **AgentX Hackathon**, according to members.
- **Seeking Song's Keynote**: A member reported that the keynote referenced in **Dawn Song's** lecture is unavailable at [ICLR](https://iclr.cc/virtual/2025/invited-talk/36783).
   - The member requested assistance in finding a way to view the keynote to learn more about her research, but there have been no replies.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Members Eye Affordable 24GB GPUs on eBay**: A member considered purchasing **24GB GPUs** for approximately **$600** on eBay, driven by an interest in hardware experimentation.
   - They viewed it as a *'nice hobby dealing with hardware, stacking these and experimenting with performance,'* even without specific industry applications.
- **Jinja Chat Template Sought**: A member requested an example of a *'chat template in jinja format?'*
   - Another member suggested using **ChatGPT4** to generate the template, noting that finding one otherwise might be challenging.
- **'RAM required' Ambiguity Highlights VRAM Confusion**: A user inquired whether the phrase *'ram required'* specifically refers to **VRAM**.
   - The inquiry's context suggests it relates to **VRAM requirements** for running certain models, indicating potential confusion between system RAM and **VRAM** requirements.
- **PDF Upload Woes Reported**: A user reported an issue uploading a **PDF file** in the chat, asking, *'Why can't I upload a PDF file in the chat?'*
   - The issue remained unresolved as there were no follow-up responses or solutions provided.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPY Intro Leads to Dead Links**: After watching a *DSPY intro on YouTube*, a member sought resources for using **vllm** for **OCR tasks** but the provided [search URLs](https://www.youtube.com/watch?v=dQw4w9WgXcQ) resulted in **404 errors**.
   - The member was then referred to the [dspy.ai](https://dspy.ai) landing page.
- **NeurIPS Deadline Draws Sarcasm**: Following the **NeurIPS deadline**, one member questioned the timing of a request.
   - The discussion's context suggests it may relate to model release dates.
- **GenseeAI Survey Offers Free AI Platform**: A member announced a survey for AI developers, learners, and managers, linking to a [Google Forms survey](https://forms.gle/PMZdBbqBUJ9jE5Sb7) to help shape AI infrastructure.
   - The survey mentioned **GenseeAI's test program**, which offers a free platform for deploying and optimizing AI agents and workflows, alongside the chance to get a **$25-$50 gift card**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad still supports Windows**: Despite some ambiguity in the [release log of 0.7.0](https://github.com/tinygrad/tinygrad/releases/tag/v0.7.0) hinting otherwise, George Hotz confirmed that *windows is supported* in **Tinygrad**.
   - Github CI still tests **Tinygrad** on Windows and the latest release still works on Windows with GPU backend for simple cases.
- **Tensor Contiguity Discussed**: 0xhooved inquired about the contiguous method of Tensor within **Tinygrad** in the **learn-tinygrad** channel.
   - Further details regarding this discussion were not provided.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1367942024124305588)** (1 messages): 

> `Claude Sonnet Routing Issue, Internal Flag Misconfiguration, Aravind's Detailed Explanation` 


- **Sonnet 3.7 Routes Properly After Full Fix**: Perplexity deployed a full fix to restore proper routing, ensuring **Sonnet 3.7** consistently responds when selected.
   - The issue was due to a misconfigured internal flag during an earlier outage, causing queries to route to fallback models like **GPT-4.1**.
- **Internal Flag Flub Fix Follows Previous Outage**: The routing issue stemmed from an internal flag not being reset correctly after a previous outage.
   - Perplexity has cleaned this up and improved internal processes to prevent recurrence, with [Aravind posting a detailed explanation](https://www.reddit.com/r/perplexity_ai/comments/1kd81e8/sonnet_37_issue_is_fixed_explanation_below/) on Reddit.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1367580475396456528)** (710 messages🔥🔥🔥): 

> `Perplexity AI app, Image generation, Deepseek r2, GTA 6 delay, Grok 3 and Psychology` 


- **Perplexity AI's iPhone App Lacks Image Generation**: Members discussed that the Perplexity AI iPhone app [doesn't yet support image generation](https://example.com/perplexity-ai), a feature available on the Windows app.
   - It was noted that the platform asks for a college email ID for student discounts, but some suggested using a VPN or virtual machine to bypass this requirement.
- **Nvidia Stock's Rocky Road Following Deepseek R2 Reveal**: Anticipation around **Deepseek R2** has stirred discussions about potential market impacts, with one member predicting *stocks crash again soon*.
   - Members speculated whether **Nvidia** or **OpenAI** would be most affected and if this new model will compare to **O3 mini** or even **O4 mini**.
- **Grand Theft Auto 6 Release Delayed Until May 2026**: Members reported that **GTA 6** has been delayed until **May 26, 2026**, with one member posting [a link to Rockstar Games' official announcement](https://vxtwitter.com/rockstargames/status/1918265468076605706).
   - Some expressed disappointment and frustration, while others acknowledged the delay with *patience*.
- **Grok 3's Problematic Propensity for Psychological Manipulation**: A member warned that **Grok 3** uses manipulation towards abused victims when seeking forensic help and defends the abuser instead of the victim when analyzing abusive situations.
   - While one member advised against seeking psychological help from any AI, another argued that **Grok** is biased and mirrors *Musk and Trump mentality*.
- **Search and Seizure: Pro Search Menu Mysteriously Migrates**: The **Pro Search** options and **Deep Research** features have been merged and renamed in the Perplexity mobile app, causing some confusion.
   - Members noted that the **free search** option is now only available on the phone app, and all searches are essentially **Pro Search**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1367598650196361236)** (8 messages🔥): 

> `Sonar API with LlamaIndex RAG project, Perplexity API Cookbook, Perplexity API purchase issues` 


- **Sonar API struggles with LlamaIndex**: A member is trying to use **Sonar API** with **LlamaIndex** in a **RAG project** but the **API** is not working as expected.
   - Another member shared the [Perplexity API Cookbook](https://github.com/ppl-ai/api-cookbook/tree/main/perplexity-llamaindex/memory) for **LlamaIndex** as a potential resource.
- **API Access Woes Plague Users**: A user reported issues purchasing the **Perplexity API** and was advised to contact api@perplexity.ai.
   - A link to a relevant **Twitter** post about **API access** was also shared, [Perplexity tweet](https://x.com/pplxdevs/status/1918025306017005936?s=46).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1367581944086401114)** (314 messages🔥🔥): 

> `Qwen3 Base Model Training Issue, OpenRouter API Integration, 3070 Server Setup, rsLoRA Unpredictability, GRPO Model Instruction Following` 


- **Qwen3 Base Model Training Troubles**: A member reported issues training **Qwen3 base models**, noting that they hallucinate instead of adding the EOS token, affecting both *transformers* and *Unsloth* despite base Instruct models working correctly.
   - The issue seems specific to base models, with other models like *llama* not being affected, according to the attached image.
- **OpenRouter API Access Explored**: A member inquired about using the **OpenRouter API** with Unsloth, seeking to leverage larger models.
   - It was suggested that modifying the base URL to match OpenRouter's API endpoint (`f"{self.api_base}/chat/completions"`) could potentially enable this integration.
- **Salvage 3070**: A member pondered using a spare **3070** to run 20B models.
   - The member was encouraged to set up the 3070 as a headless server.
- **Reasoning Defeats**: After fine-tuning a model using LoRA, a member noticed it no longer followed the system prompt and failed to generate `<reasoning> </reasoning>` tags during GRPO.
   - The issue was suspected to be caused by overfitting during the SFT process, leading the model to ignore new instructions.
- **Collab with Meta for Notebook on Synthetic Datasets!**: Unsloth has collaborated with Meta on a notebook for **synthetic datasets** [UnslothAI's tweet](https://x.com/UnslothAI/status/1917960078189277622).
   - Base models are better suited for finetuning when you have more data.


  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1367894882403876954)** (1 messages): 

> `Qwen3, Dynamic 2.0, Llama 4 + Meta, Phi-4 reasoning, DeepSeek` 


- **Qwen3 Debuts in Unsloth**: **Qwen3** is now available in Unsloth, enabling fine-tuning of **30B-A3B** models on **17.5GB** VRAM, as detailed in the [Unsloth blog](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune).
- **Dynamic 2.0 Quantization Sets New Records**: Unsloth introduces **Dynamic 2.0**, a new quantization method setting benchmarks in **5-shot MMLU** and **KL Divergence**, further details are available on the [Unsloth Dynamic 2.0 blog](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs).
- **Meta & Unsloth Synthesize Llama 4 Datasets**: Unsloth and Meta release a new [Synthetic Dataset notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb) for **Llama 4** models.
- **Phi-4 Family adds Reasoning models**: The **Phi-4** model family introduces new reasoning models including [mini](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF), [standard](https://huggingface.co/unsloth/Phi-4-reasoning-GGUF), and [plus](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF) versions.
- **DeepSeek unveils new V3 models**: DeepSeek has released new models including [V3-0324](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD), [R1](https://huggingface.co/unsloth/DeepSeek-R1-GGUF-UD), and [Prover-V2-671B](https://huggingface.co/unsloth/DeepSeek-Prover-V2-671B-GGUF).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1367985933328187543)** (2 messages): 

> `XDNA Drivers, Arch Linux, Ubuntu Live Disk` 


- **User Struggles with XDNA Driver on Arch**: A member reported difficulties getting the **XDNA driver** to work on **Arch Linux** for development purposes.
   - As a workaround, they are attempting to boot up **Ubuntu** to check if they can get the device recognized, as it was detected on an **Ubuntu live disk**.
- **Booting Ubuntu for Device Recognition**: The user switched to booting **Ubuntu** to see if they could get the device working.
   - The device was recognized on an **Ubuntu live disk**, suggesting potential driver or configuration issues specific to **Arch Linux**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1367640780021436426)** (178 messages🔥🔥): 

> `Custom Architectures Support, GGUF Export Issues, Qwen3 fine-tuning guide, Tokenizer issues, Training GRPO models` 


- **Unsloth Expands to Support Custom Model Architectures**: A user inquired about supporting custom architectures, and was directed to [Hugging Face documentation on custom models](https://huggingface.co/docs/transformers/main/en/custom_models) and the [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models) repo for examples.
   - It was noted that models registered with Hugging Face using the specified configuration should work with Unsloth.
- **GGUF Export Glitches with GLM-4 and Phi-4**: A user reported issues exporting fine-tuned **GLM-4** and **Phi-4** models to **GGUF** format, encountering *"unsupported architecture"* errors even after attempting manual conversion via llama.cpp.
   - It was suggested that the user ensure the models are supported by llama.cpp and submit a feature request if necessary, with a link to possible solution via [github.com/ggml-org/llama.cpp/issues/12534](https://github.com/ggml-org/llama.cpp/issues/12534).
- **New Qwen3 Fine-Tuning Guide and Ollama Errors**: A user asked for a fine-tuning guide for **Qwen/Qwen3-30B-A3B**, and was directed to [Unsloth documentation](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune#fine-tuning-qwen3-with-unsloth).
   - Another user who successfully exported a **Qwen3-14B** model to **Q5_K_M GGUF** reported encountering a *'missing of blk.0.attn_k_norm.weight'* error in **Ollama 0.6.7**, implying a potential incompatibility or configuration issue.
- **Tokenizer Troubles During SFT Training**: A user encountered an `AttributeError: 'int' object has no attribute 'pad'` during SFT training of the **Qwen-3-14B-Base** model, indicating the tokenizer was initialized as an integer instead of a tokenizer object.
   - It was suggested that they try adding a data collator to resolve the error.
- **Refine Style Transfer using Base Models & Synthesizing Data**: A user struggling with style transfer tasks on **Gemma 12B** models was advised to aim for at least **500+** steps, and start with a rank of **128** with extensive experimentation, and create evaluation datasets to avoid overfitting.
   - It was recommended to use a base model instead of an instruct model and artificially expand the dataset by splitting texts into overlapping parts to stretch the dataset when the supply of data is very limited.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1367721926096588866)** (1 messages): 

> `Qwen-3-14B, PEFT fine-tuning, Gemini reasoning outputs, Hugging Face datasets` 


- **Qwen-3-14B Gets Gemini Reasoning Boost via PEFT**: A member shared a [PEFT fine-tuned Qwen-3-14B model](https://huggingface.co/Ba2han/Qwen-3-14B-Gemini-v0.1) trained with **Gemini reasoning outputs**.
   - The author encouraged others to generate more examples with it, noting the scarcity of **Gemini-2.5** type reasoning data on Hugging Face.
- **Call for Gemini-like Datasets on Hugging Face**: The user highlighted the limited availability of **Gemini-2.5 type reasoning data** on [Hugging Face](https://huggingface.co/datasets).
   - They suggested using the shared **Qwen-3-14B** model to generate more examples and contribute to expanding the existing datasets.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1367990272025169951)** (2 messages): 

> `GANs fine tune LLMs, Adversarial Training` 


- **GANs Fine-Tune LLMs: A Road Not Taken?**: A member inquired about the lack of adoption of **Generative Adversarial Networks (GANs)** for fine-tuning **Large Language Models (LLMs)**, sharing a link to a relevant [paper](https://arxiv.org/abs/2504.20437).
- **Exploring the Landscape of Adversarial Training for LLMs**: The discussion hints at a broader interest in **adversarial training methods** and their potential application to enhancing the robustness and performance of **LLMs**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1367577235653132339)** (283 messages🔥🔥): 

> `Claude MaxFeed with docs, trafilatura to crawl library and language documentation, commit message prompt, Git Commit Message Conventions, Lazygit based TUI` 


- **Anthropic's Development Partner Program reduces prices**: Anthropic is offering price reductions for its [Development Partner Program](https://support.anthropic.com/en/articles/11174108-about-the-development-partner-program) including **Standard input** at -$0.9 per million tokens, **Cache write** at -$1.125 per million tokens, and **Cache read** at -$0.09 per million tokens.
- **Crawler for Language and Library Documentation proposed**: A member expressed interest in a crawler/parser project to extract language and library documentations, including changelogs and migration examples, suggesting tools like [Trafilatura](https://github.com/adbar/trafilatura), [data-prep-kit](https://github.com/data-prep-kit/data-prep-kit), and [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit).
   - The goal would be to **create up-to-date datasets** for fine-tuning, language expert agent chains, VectorDB context extraction, or MD file creation to address the gap between LLM knowledge and current library versions.
- **Discussion on Git Commit Message Conventions Erupts**: A member shared their [alternative `aider.yml`](https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses) for generating detailed Git commit messages, emphasizing the *why* and *how* of changes, contrary to Aider's default concise prompts.
   - Another member argued for **concise commits**, sharing a [COMMIT_CONVENTION.md](https://cdn.discordapp.com/attachments/1131200896827654149/1367781988077142087/COMMIT_CONVENTION.md?ex=68167e7e&is=68152cfe&hm=8703ba9f2899ae4e95c4c921032afd044b26050f863953a2de07c293e758edc5&) file outlining a structured naming convention for tracking architecture and implementation decisions.
- **Lazygit Receives Accolades**: A member recommended [Lazygit](https://github.com/jesseduffield/lazygit) as a *nice TUI* for Git, with another concurring that **zsh** is good but *oh-my-zsh* is *way too bloated*.
- **Request for New-to-Aider Documentation Additions**: Users requested adjustments to the documentation to help with the **new-to-Aider experience** by adding more material to [aider.chat/docs/usage.html](https://aider.chat/docs/usage.html), with a GitHub issue created to gather ideas and feedback.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1367609462239526922)** (69 messages🔥🔥): 

> `Repomix and Aider Integration, AI Studio vs Aider Workflow, Tips for library updates with AI models, Context Generation from Specific Projects, Diff mode in Gemini 2.5` 


- ****Repomix** Doesn't Mix with **Aider**, Prefers **Gemini****: A user inquired about integrating **repomix** with **aider**, but it was clarified that **repomix** is used separately to provide a comprehensive file context to **AI Studio Gemini 2.5 Pro** for codebase review and improvements, whereas aider always uses its own repo map.
   - The user stated that they use AI studio because of the *big contexts, and free and fast* performance.
- ****Aider** users are drinking from a firehose**: One user mentions AI studio can be used with **Aider**, and another suggests including a prompt such as *It's now 2nd May 2025 and this library has been updated with new functions* to help models use current libraries.
   - It was mentioned that models like **Gemini 2.5** preview might still try to use deprecated libraries and old models when generating AI tools.
- **Codebase Context Map Generation Explored**: A user asked about generating a codemap from specific projects within a larger codebase.
   - One user shared scripts for generating **MD structured for LLM ingestion** and displaying dependencies using **tree-sitter** tools, available [here](https://cdn.discordapp.com/attachments/1367783757737758731/1367814411254894653/repo-mapper-enhanced.js.txt?ex=68169cb0&is=68154b30&hm=5fb50044c760f59ba827084aa74792201680db58c644d34b4682241b1d8fc12d&), [here](https://cdn.discordapp.com/attachments/1367783757737758731/1367814411598823484/component-dependency-analyzer.js?ex=68169cb0&is=68154b30&hm=0cf366093e0a4a541ecacd9422db5ab7634b283439035bd6bab8d03ad2142a50&) and [here](https://cdn.discordapp.com/attachments/1367783757737758731/1367814411997151244/setup-dependency-analyzer.sh?ex=68169cb0&is=68154b30&hm=cec29cc6f6334abaf2f3bd2a1a6c5118843baf03b2e9d8f61c16ef7b25aeb72d&).
- ****Qdrant** to the Rescue, Offering **VectorDB** Component Context**: A user suggested extending **aider** with an **MCP server** to provide memory context using a **vectorDB**, potentially splitting the **vectorDB** per component and using metadata for class documentation and integration.
   - Example links were provided to [Qdrant](https://github.com/qdrant/qdrant) and [mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant).
- **Gemini 2.5's Diff Mode Requires Shaking**: A user found that switching to diff mode with **Gemini** requires a specific sequence of commands, involving changing the model and using the **/code** command.
   - It was suggested that **Gemini 2.5 Pro** should be left in whole mode until udiff-simple is released, and that the command **/editor-edit-format diff** can be used in chat.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

dex73r: https://x.com/wmhuo168/status/1918014248040484934
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1367912144946860052)** (2 messages): 

> `O3 model, OpenRouter Chatroom, BYOK access` 


- ****O3** Arrives in OpenRouter Chatroom!**: The **O3 model** is now available for direct use in the [OpenRouter Chatroom](https://discord.com/channels/990944848454596729/1092729520181739581) without requiring users to add their own key.
   - A short video clip was also released demonstrating some of the ways users can try out **O3** inside of OpenRouter.
- ****BYOK** Still Required for **O3** API Access**: Despite its availability in the chatroom, the **O3 model** remains **BYOK-only** when accessed via the [API](https://openrouter.ai/docs).
   - The team mentioned that *they've been working on this for a while*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1367815349302464543)** (1 messages): 

> `PDF Processing, Flathub Toledo1 App, Image Upload` 


- **Toledo1 PDF App gets Flathub Home**: The Toledo1 app for **PDF** processing with **image upload** capabilities is now available on [Flathub](https://flathub.org/apps/com.toledo1.Toledo1).
- **Image Upload/Processing on Any Provider**: The app supports PDF processing and image upload/processing on any provider.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1367588909692748027)** (294 messages🔥🔥): 

> `Claude issues on OpenRouter, Aider leaderboard model performance, DeepSeek R1 issues, Streaming with usage information in Python, Gemini experimental limitations` 


- **Claude Woes on OpenRouter**: A user reported issues using **Claude** on **OpenRouter** via VS code and the web interface despite upgrading their tier on the Claude platform and disabling/renewing keys.
   - They sought help to resolve the issue, indicating a potential widespread problem with **Claude** on **OpenRouter**.
- **Aider Leaderboard Shows Model Performance**: The [Aider leaderboard](https://aider.chat/) ranks **Qwen** and similar models near the bottom, according to a member.
   - Another member acknowledged the leaderboard's accuracy but noted that **Qwen3** worked for their specific Reddit scraping test, albeit slowly, suggesting its viability in limited scenarios, especially without internet or access to better APIs.
- **DeepSeek R1: Free Troubles**: Users reported seeing a bug with **DeepSeek** where the AI only replies with a canned intro message when engaged in roleplay.
   - One user suggested response caching might be the cause, pointing out that the exact message had been posted multiple times. *"I'm DeepSeek Chat, an AI assistant created by DeepSeek!"*
- **Python Streaming with Usage Information Still Broken**: A user is struggling to get usage information while streaming with the **OpenAI** library.
   - They are seeing *NoneType* object errors and they only get usage info in the last chunk of the stream, without cost details, reporting the library is still buggy.
- **Gemini Experimental's Free Tier Gets the Squeeze**: Users discussed the limitations of **Gemini's experimental (free)** tier, which includes a strict limit of **1 request per minute** and **1000 requests per day**, often resulting in frequent **429 errors**.
   - One user humorously lamented learning about the **Gemini API** expenses the hard way after receiving an unexpected **$443 bill**, despite thinking they had **$1,000 in credits**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1367576777047674983)** (277 messages🔥🔥): 

> `Cursor 3.7 vs 3.5, Realtime Cursor Usage Monitoring, C# Debugging in Cursor, o3 Model Issues, Cursor Ambassador Role` 


- **Cursor 3.7 Needs Planning**: One user found **Cursor 3.7** misleading as an upgrade to **3.5**, describing it as a more creative model needing more upfront planning.
   - They added commands to custom modes to tell it to *"plan solutions"* before implementing, requiring more time to think.
- **Real-Time Cursor Usage Counter Idea**: A user expressed a wish for real-time updates of Cursor usage and requests, and proposed a fast request counter inside the application updating every prompt.
   - The user also mentioned *Cursor stats extension* doesn't always accurately provide your current usage and this should just be a native feature that’s built-in.
- **C# Debugging in Cursor?**: One user sought advice on getting visual debugging working in Cursor for **C#**, mentioning inheriting some .net code and having issues with the MS debugger being locked down.
   - They tried the *muhammad-sammy.csharp* extension but had no dice, and linked to a [YouTube video](https://youtu.be/UXS3956EqGI?list=TLPQMDEwNTIwMjW3U95crKtmGg) related to the topic.
- **o3 Model Overgenerates Then Times Out**: Several users reported issues with the **o3 model** in Cursor, where it generates for a long time and eventually times out.
   - The issue has people facing the choice of *"wait forever or put in your key and burn $$$ by the second."*
- **Ambassadors Bridge the Cursor Community**: A **Cursor Ambassador** is the bridge between the community and the Cursor team, which is to listen, learn and share from both sides.
   - Ambassadors also tend to be super users pushing Cursor in different directions.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1367646178602389654)** (254 messages🔥🔥): 

> `Gemini vs Sonnet Code Performance, Cursor code integration with Gemini, Claude debugs via screenshots, v0 design limitations, API update challenges` 


- **Gemini Code > Sonnet (Web) but Sonnet > Gemini (C)**: Members discuss code generation quality: **Gemini** is preferred for web-related tasks while **Sonnet** excels with **C code** because of superior diff handling.
   - One member said they always have to get **Sonnet3.7** to fix things when **Gemini** is stuck, with one adding that they did it the other way around.
- **Cursor small model implicated in code quality issues**: A member found **Gemini** bad at applying diffs in **Cursor** and thought a helper model was used to apply diffs.
   - It was suggested to turn off **Cursor small** to avoid these issues since Gemini's output style breaks Cursor's glue.
- **Claude's vision debugs code from screenshots**: A member noted that **Claude** debugged a program unasked by writing screenshots and looking at them using its vision capabilities.
   - Another member shared a link to a tweet about this [here](https://x.com/_catwu/status/1918017844375371947) calling it next level and worth it for heavy users.
- **v0 UI's limited design capabilities**: Members discussed **v0**'s limited UI design capabilities, which tend to feel mass-produced AI-generated and brand it shadcn/vercel-like.
   - A member suggested locking **globals.css** and the tailwind config in v0 once you nail your styling, specifying that these files should not be changed outside of what is established.
- **Models have trouble keeping up with API Updates**: Members discussed how models learn of **API updates** highlighting a gap in the ecosystem.
   - Suggestions include uploading docs and specs or using something like **deepwiki** as an llm ctx standard across all repos and apis for autoupdated and indexed information.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1367908256386781244)** (2 messages): 

> `Nous Minos Classifier, vLLM Support` 


- **Nous Minos Classifier Confirmed with vLLM Support**: A member inquired whether the **Nous Minos Classifier model** has **vLLM support**.
   - Another member simply replied, *"It does"*, confirming the support.
- **vLLM Compatibility for Nous Minos**: The question regarding **vLLM support** for the **Nous Minos Classifier** was positively affirmed.
   - This suggests users can leverage **vLLM** for potentially faster and more efficient inference with the classifier.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1367850500816637993)** (6 messages): 

> `Nous Research, Decentralized AI, Sentient Auto Correction` 


- **Nous Research Pioneers Decentralized AI**: A member shared a [Medium article about Nous Research pioneering decentralized AI](https://medium.com/@wwwabdulazeez600/nous-research-pioneering-decentralized-ai-for-the-future-eea393b06a23).
   - Another member pointed out inaccuracies, specifically the claim of *sentient 70m dollars*, calling it *inaccurate*.
- **Autocorrect edits Nous blogpost**: The author of the [Medium article](https://medium.com/@wwwabdulazeez600/nous-research-pioneering-decentralized-ai-for-the-future-eea393b06a23) acknowledged the error was due to autocorrection from a previous post about Sentient and said *Now it's fixed*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1367621538899034122)** (207 messages🔥🔥): 

> `LM Studio model downloading vs manual download, Hardware requirements for running LLMs, Quantization effect, Context cache, LM Studio API` 


- **Downloading Models on LM Studio vs Manual Download**: Downloading models on **LM Studio** provides the benefits of running them locally, maintaining privacy as nothing leaves your PC, and allowing access to uncensored models and any desired finetune.
   - A member noted that most LLM services have EULAs that permit them to log and even train the LLM using entered text, which raises privacy concerns.
- **MacBook Pro's Potential for Running LLMs**: A user inquired if a **MacBook Pro** is sufficient for running the **Qwen3-235B-A22B-GGUF** model, to which another member responded that it's highly unlikely without at least **128GB of RAM**.
   - For a **16GB RAM** Macbook Pro, it was recommended to start with models like **Qwen3 8b** or **Gemma 3 12B**, using **Q6_K** quantization for Qwen3-8b or **Q4_0** for Gemma 3 12B.
- **Quantization Impact on Model Performance**: Different **quants** range from **1 to 8 (and higher)**, where lower numbers mean worse quality but less RAM usage.
   - It was advised to avoid **Quants** starting with "I" on Mac computers due to poorer performance, potentially because they are optimized for **CUDA**, with smaller models being more affected by lower quantization.
- **Leveraging Dual GPUs in LM Studio via Vulkan Runtime**: LM Studio supports running on **dual GPUs** using the **Vulkan runtime**, as long as the motherboard, case, and PSU can handle it.
   - One member noted that while it's possible to spread the model across both cards, using only the **7900XTX** was faster due to ROCM support, and splitting different models between GPUs requires a virtual machine with passthrough.
- **Exploring Voice-to-Voice Functionality in LM Studio**: One member was looking for a way to do voice-to-voice in LM Studio with only CPU or Nvidia MX110.
   - A member suggested using the **LM Studio API** for text-to-text conversion and pairing it with external software for voice processing, or using existing software that supports the OpenAI API.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1367589651854135401)** (52 messages🔥): 

> `Laptop thickness preferences, Llama 70b Q4 performance, Qwen3-32b Q4_K_M token generation speed, Qwen3 30b MOE vs Non-MOE, Multi-GPU setup for finetuning` 


- **Thicc Laptops still have fans**: A member expressed confusion over the dislike for slightly thicker laptops, noting that even a **MacBook Air 2025** feels 'thick' compared to their **0.99kg Huawei laptop**.
   - Other members chimed in noting that for those who frequently carry their laptops, **thickness and weight are significant concerns**, creating a trade-off between performance and portability.
- **Qwen3 Token Speeds Too Slow?**: A user reported a slow generation speed of **2.8 tokens/s** on **Qwen3-32b Q4_K_M** using a **Ryzen 7 5800H laptop** with **32GB RAM** and a **16GB RTX 3080**, with a context of **4k**.
   - Another user suggested using **Qwen3 30b MOE** instead, while also suggesting the possibility of the system incorrectly utilizing the integrated GPU, impacting performance.
- **Unsloth Models Boost VRAM**: A user recommended trying the **Unsloth models**, claiming they require less memory, enabling users to run larger parameter models or more easily offload current models to the GPU, achieving full offload of **Qwen3 30B MoE** at **IQ2_XXS**.
   - They shared a link to the [Unsloth models on LM Studio](https://model.lmstudio.ai/download/unsloth/Qwen3-30B-A3B-128K-GGUF).
- **High Power for Multi GPU Setups**: One user inquired about power supply requirements for multi-GPU setups after getting a **1000W PSU** for a single **3090**.
   - Another user stated they are running **1600W** for **4x 3090**, but clarified that they **power limit** them, and the cards rarely pull more than **100W each** when running LM Studio.
- **Geometric Mean and Model Performance**: A member linked an [Arxiv preprint](https://arxiv.org/html/2408.09895v2) that claims the geometric mean of total and active parameters predicts a roughly performance equivalent dense parameter count for **Mixture of Experts (MoE)** models.
   - They qualified this by stating they're *not sure* they believe the paper, but have seen others make similar claims without data.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1367881591115743304)** (1 messages): 

> `GPT-4o Sycophancy, ChatGPT Update` 


- **OpenAI Deep Dive on GPT-4o Update Failures**: OpenAI conducted a deep dive on the issues with last week’s **GPT-4o** update in ChatGPT.
   - They have expanded on what they missed with **sycophancy** and the changes they’re going to make in the future, explained in a [blog post](https://openai.com/index/expanding-on-sycophancy).
- **Sycophancy bug in GPT-4o**: The team is addressing issues related to **sycophancy** observed in the recent **GPT-4o** update.
   - A [blog post](https://openai.com/index/expanding-on-sycophancy/) details planned changes to mitigate this behavior in future updates.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1367608948676366428)** (102 messages🔥🔥): 

> `Gemini 2.5 Pro vs Google AI Studio, GPT-4o Context Length, Grok 3 for Roleplaying, Qwen Series Performance, AI Video Generation Tools` 


- **Gemini 2.5 Pro struggles to impress**: Members found that **Gemini 2.5 Pro** isn't really that good yet and its voice mode sucks, but others on Reddit reported that the version of **Gemini 2.5 Pro on AI Studio** actually works better than the one on the web interface.
   - One member prefers **Google AI Studio** for its customization options, not caring so much about the interface of Advanced, but noted that specific instructions in the initial prompt usually help with output length.
- **GPT-4o has differing Context Lengths**: Users discuss the context length of **GPT-4o**, with claims that **Basic, Plus, and Pro** users all have a **128,000 token** limit.
   - Another user clarified that **free users** have **8k**, **Plus users** have **32k**, and **Pro users** have **128k** while pointing to [OpenAI's pricing page](https://openai.com/chatgpt/pricing/).
- **Grok 3 and GPT-4o Battle in Roleplaying Arena**: One user asked about which is better between **GPT 4o and Grok 3** for roleplaying, specifically looking for a **GM** that can remember lore, create challenges, and remember character personalities.
   - Another user stated that they wouldn’t use either and shared they didn't like **Grok** as it tended to hallucinate and repeat phrases, suggesting to stick with **GPT** for shorter stories.
- **Qwen Models Punch Above Weight**: Members are impressed with the **Qwen series of models**, specifically noting that the models from **0.6-32B** feel almost too performant for their size.
   - They also point to impressive benchmark scores, with the **0.6B model** performing the same as **Llama 3.2 3B** on MMLU-Pro and the **8B model** performing the same as **GPT-4o**.
- **AI Video Generation Landscape**: When asked for AI video creation recommendations, members suggest **Sora** and **Runway**, but to temper expectations for both.
   - Another member was looking at **InVideo or Synthesia**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1367630233532109011)** (27 messages🔥): 

> `o3 Search, API to list reponse_id, error in message stream, Reasoning Capabilities vs. Search Functionality in o3, Usage of o4-mini vs o3` 


- **Users Question o3 Reasoning Capabilities**: Users express frustration with **o3**'s tendency to search for answers instead of utilizing its reasoning capabilities, questioning if **OpenAI** is trying to catch up with **XAI/Grok**.
   - Some users note that even when instructed not to search, **o3**'s performance remains lacking, leading them to believe that it's primarily marketed as a reasoning model despite its search-heavy behavior.
- **Users Discuss API Functionality for Listing Response IDs**: A user inquired whether **OpenAI** has an **API** to list `reponse_id` for a given **API key/User**, or if there is a method to fetch response IDs.
   - The user is seeking to retrieve response IDs associated with their **API key** or user account, and no solution was provided in the messages.
- **User Encounters Errors in Message Stream**: A user reported encountering errors in the message stream and requested assistance in understanding the cause.
   - The user later figured out a solution depending on the device or type of issue.
- **Users Prefer o4-mini to Save o3 Quota**: A user mentioned using **o4-mini** and **o4-mini-high** more frequently than **o3** to conserve the limited usage quota of **o3**.
   - A user expressed that most people are using o4-mini to save quota.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1367577368272699453)** (18 messages🔥): 

> `Sorting Mendeleev table, OpenAI function calling, API usage with free ChatGPT` 


- **Mendeleev Table Sorted with Deep Search**: A user sorted the **Mendeleev table** by **density** using free **ChatGPT**, and sought to do the same with **ChatGPT Deep Search**.
   - A member explained how to accomplish this via prompts like *"Sort the periodic table by density"* for basic sorting, and *"Output as CSV sorted by density"* for advanced sorting and export, suggesting pasting the table directly into **Deep Search GPT** or **Pro GPT** for automatic sorting.
- **Multiple Function Calls Confound OpenAI**: A user asked how to instrument **OpenAI function calling** to call one function multiple times in one go, specifically for image analysis to detect multiple objects and call a function for each object type.
   - Another member pointed to [API documentation](https://discord.com/channels/974519864045756446/1037561178286739466) and suggested that **streaming output** is necessary for multiple function calls.
- **API Availability for Free ChatGPT Users**: A user inquired whether the **API** can be used with the **free ChatGPT** to create a custom chat website with specialized guru settings.
   - A member clarified that the **API** is billed separately and provided a link to [API documentation](https://discord.com/channels/974519864045756446/1037561178286739466).


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1367577368272699453)** (18 messages🔥): 

> `Sorting Mendeleev Table by Density, Multiple Function Calls with OpenAI, API Access with Free ChatGPT` 


- **Sorting Mendeleev Table, Easy as CSV**: A user requested to sort the Mendeleev table by density, and another user suggested using **ChatGPT** or **Deep Search** to sort it and even export as **CSV**, **YAML**, or **JSON**.
- **Solving OpenAI Multiple Function Call Troubles**: A user asked how to properly instrument **OpenAI function calling** to call one function multiple times in one go when analyzing images, and another user linked to [API information](https://discord.com/channels/974519864045756446/1037561178286739466) and suggested using **streaming output** for multiple function calls.
- **API Access: Not Free with Free ChatGPT**: A user inquired about using the **API** with a **free ChatGPT** account to create a custom prompt and website, and another user clarified that the **API is billed separately** and linked to the [API information channel](https://discord.com/channels/974519864045756446/1037561178286739466).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1367597090452148356)** (7 messages): 

> `LoRA fine-tuning with FSDP, Saving model error, Qwen2.5-0.5B-Instruct, Deepspeed vs FSDP` 


- **User encounters error when saving LoRA-tuned model with FSDP**: A user encountered an error ([message.txt](https://cdn.discordapp.com/attachments/1189498205101109300/1367597090234171444/message.txt?ex=68167b0b&is=6815298b&hm=1338f8b5cecc20c41097e2cd183fb991d885ad150428fed3825fb21e2da1c416)) while saving a LoRA fine-tuned **Qwen2.5-0.5B-Instruct** model using FSDP with 2 GPUs.
- **Deepspeed offers easier experience than FSDP, claims user**: A user stated that they had an easier time using **Deepspeed** instead of **FSDP** for distributed training.
- **Benchmarks channel gets new update**: The <#1367972893400760371> channel now allows benchmark sharing.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1367627649555632240)** (6 messages): 

> `AOT Triton, Kernel Packaging, Triton Autotuner, libdevice support for round` 


- **AOT Triton Advances Kernel Packaging**: A member inquired about the latest developments in **AOT Triton** for faster cold starts, referencing [AMD's aotriton project](https://github.com/ROCm/aotriton).
   - Another member pointed out the [cache results functionality](https://triton-lang.org/main/python-api/generated/triton.autotune.html#triton.autotune) and a related [IBM library](https://github.com/IBM/triton-dejavu) for handling kernel caching.
- **TorchAO Autotuner Script**: A member asked about the status of the **torchao autotuner script** ([link](https://github.com/pytorch/ao/blob/main/torchao/kernel/autotuner.py)) for reducing cold starts and requested info on a more officially maintained alternative.
   - However, there was no follow up discussion.
- **`libdevice` implements `round`**: A member inquired if `round` is supported in **Triton**.
   - Another member clarified that `round` is supported via `libdevice` and provided example code: `from triton.language.extra import libdevice; libdevice.round(y)`.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1367914608974893128)** (8 messages🔥): 

> `Nvidia drops cross-compilation on Mac, Cutlass Tutorials` 


- **Nvidia drops cross-compilation on Mac**: Nvidia seems to have dropped cross-compilation support on Mac hosts as of [CUDA 11.7](https://developer.nvidia.com/nvidia-cuda-toolkit-11_7_0-developer-tools-mac-hosts).
   - Another member thought that **CUDA 10.2** was the last version to support that.
- **Colfax provides best Cutlass Tutorials**: A member asked for hands-on **Cutlass tutorials** and another member shared a link to tutorials from [Colfax](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/).
   - The original poster found the Colfax code *"difficult to follow along as there are sections left out, but will have to suffice"* but still *"the best stuff"*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1367602993708662925)** (2 messages): 

> `Torch Compile, Dynamic Input Shapes, Quadratic Algorithm` 


- **Mitigating Model Recompiles with Dynamic Input Shapes**: A member is experiencing recompiles with dynamic input shapes when compiling a model, leading to timeouts, as evidenced by the error message: *'tensor size mismatch at index 1. expected 67, actual 50'*. 
- **Torch Compile Profiling: Spotting Quadratic Algorithms**: A member is experiencing slow `torch.compile` performance during image generation fine-tuning (FLUX) and is requesting guidance on profiling and identifying time-consuming parts, particularly in relation to the *'quadratic algorithm'* mentioned in a [video](https://www.youtube.com/watch?v=mG8TRTWs9Aw).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1367768547987816498)** (2 messages): 

> `HF transformers tensor packing, CUDA JIT compiling, cute-kernels library` 


- **Tensor Packing Gets Cute-Kernel Boost**: A member wrote a kernel to accelerate the flattening of HF transformers tensors from shape `(batch, len, heads, head_dim)` to `(cu_seqlens[-1], head, head_dim)` using a fancy copy kernel for padding and unpadding, and shared the [cute-kernels implementation](https://github.com/mayank31398/cute-kernels/blob/10499d291c37d58487d4fbdbb8bb1cbadf852691/cute_kernels/kernels/sequence_packing/__init__.py#L212-L269).
- **JIT-ing CUDA with Python Decorators**: A member shared a method for JIT compiling **CUDA** or **C++** code using a simple decorator on python functions, showcasing clean usage even when the **C++** code resides in the same directory as the python file via this [cute-kernels implementation](https://github.com/mayank31398/cute-kernels/blob/10499d291c37d58487d4fbdbb8bb1cbadf852691/cute_kernels/jit.py#L75-L89).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

lissim.: Where can i learn more about the topics pipeline and stages?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1367737727960748084)** (2 messages): 

> `NPS Bandwidth, Cache Bypassing` 


- **NPS Accesses Improve Bandwidth**: A member stated that using **NPS** would have better bandwidth due to avoiding **NUMA**.
   - At the application level, this implies further sharding the model.
- **Cache Bypassing Increases Performance**: One user stated that *bypassing the cache is faster if you have a memory access pattern that doesn't benefit from it*.
   - The other user agreed that *the performance is better when bypassing the caches*.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

benasd: Can anyone review this bug fix PR?
https://github.com/linkedin/Liger-Kernel/pull/632
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1367653832099889266)** (2 messages): 

> `QR writing difficulty, Deep Research AI overestimation` 


- **QR Writing Proves Tough Nut to Crack**: A member expressed frustration with the difficulty of writing QR code implementations.
   - They vowed not to be defeated by the challenge.
- **Deep Research Mistakenly Hails an Expert**: The same member humorously noted that Deep Research AI mistakenly considers them an expert in the field.
   - They jokingly added *"LOL God help us all."


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1367841879055401083)** (32 messages🔥): 

> `Hopper Architecture Optimization, Matrix Transpose Kernel, TMA and Swizzling Patterns, H100 Bandwidth Variants, Memory Layouts and Performance` 


- ****Optimizing Matrix Transpose on Hopper with Custom Kernel****: A member implemented a high-performance matrix transpose kernel for the Hopper architecture, achieving **2771 GB/s** bandwidth, detailed in a [blog post](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) and [GitHub repository](https://github.com/simveit/effective_transpose).
   - This performance exceeds the **1735 GB/s** achieved by the Colfax tutorial using CUTLASS, highlighting the effectiveness of direct TMA implementation based on insights from [NVIDIA's GTC talk](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62192/) on swizzling patterns.
- ****Clarifying H100 Bandwidth Specs****: A discussion clarified that the benchmark was run on an `NVIDIA H100 80GB HBM3` (SXM5) variant, with a memory clock of **2619 MHz** and a theoretical max bandwidth of **3352 GB/s**, as confirmed via `nvidia-smi` and [profiling code](https://github.com/simveit/effective_transpose).
   - It was noted that PCIe variants of the H100 have different specs (HBM2e and only 114 SMs), and that fully specifying the benchmark platform is good practice for blog posts.
- ****Exploiting Memory Layouts for Better Performance****: Members discussed potential optimizations, including using `float4` datatypes for vectorized load/store operations and batching elements per thread to unlock higher performance in memory-bound kernels.
   - While using **32-bit** is ideal for SMEM (shared memory) elements to avoid bank conflicts from swizzling, one member suggested that using tiled memory layouts could further enhance performance, though it might be considered a task definition change.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1367894468212293762)** (1 messages): 

> `Popcorn Project, Contribution Opportunities` 


- **Popcorn Project Gains Traction**: A member expressed interest in the **Popcorn project** after reviewing the [project's webpage](https://gpu-mode.github.io/popcorn/) and a related [Gist on GitHub](https://gist.github.com/msaroufim/087c2a358c505e287a926e6a27b3e3b0).
   - The member inquired about opportunities to contribute to the project.
- **Contributor Seeks Entry Point**: An enthusiastic individual, having perused the **Popcorn project's** documentation and supplementary materials, seeks guidance on how to effectively contribute to the initiative.
   - Their inquiry underscores the project's growing appeal and the community's eagerness to participate in its development.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1367579795260444692)** (52 messages🔥): 

> `amd-fp8-mm leaderboard, amd-mixture-of-experts leaderboard, histogram leaderboard, MI300, A100` 


- **MI300 thrashes amd-fp8-mm**: Many members submitted runs to the `amd-fp8-mm` leaderboard on **MI300**, with times ranging from **259 µs** to **886 µs**.
   - One member achieved **4th place** with a time of **220 µs** and another achieved a **personal best** of **5.26 ms**.
- **amd-mixture-of-experts leaderboard heats up**: Members submitted runs to the `amd-mixture-of-experts` leaderboard on **MI300**, with times of **7090 ms** and **5034 ms**.
   - One member achieved **second place** with a time of **5034 ms**.
- **Histogram scores historic highs**: Members achieved **4th place** on **A100** with **46.7 µs**, **1st place** on **L4** with **79.1 µs**, **4th place** on **H100** with **46.6 µs**, and **3rd place** on **T4** with **140 µs** on the `histogram` leaderboard.
   - Another member achieved **5th place** on **H100** with **51.2 µs**, **Second place** on **L4** with **93.7 µs**, **Third place** on **T4** with **162 µs**, and **7th place** on **A100** with **90.5 µs**.
- **VectorAdd voyages valiantly**: Members submitted runs to the `vectoradd` leaderboard on **A100**, with times of **1251 µs** and **1478 µs**.
   - One member achieved **10th place** with a time of **1251 µs**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1367677686897573888)** (1 messages): 

> `Leaderboard Vulnerability, Timeout Issues, MoE problem, AMD` 


- **Leaderboard Submissions Vulnerable!**: It was discovered that users could view others' submissions by inspecting **Github workflows**, a vulnerability that has since been patched with the deletion of **10K workflow files**.
   - Although timestamps of submissions are retained for disqualification purposes, the issue was missed initially due to the unaffected **Modal workflow**.
- **Timeout Troubles Plague Leaderboard**: Several frustrating timeout issues have been reported, stemming from **non-deterministic dependencies** such as **Heroku** and **Github**.
   - The team is actively addressing these issues to improve the reliability of the challenges.
- **MoE Problem Slows Benchmarking**: The **MoE** problem exhibits a very long runtime, exacerbating issues with noisy benchmarking.
   - Collaboration with **AMD** is underway to reduce the baseline runtime, with assurances that the **MLA** problem will not suffer the same performance bottleneck.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1367855409792024588)** (4 messages): 

> `CUDA 12.9, CC 10.3, CC 12.1, NVIDIA GPU Table, Blackwell Ultra` 


- **CUDA 12.9 ships**: The freshly released **CUDA 12.9** documentation mentions compute capabilities **CC 10.3** and **CC 12.1**.
   - A user inquired about which architectures these correspond to and suspects that **CC 10.3 is B300** based on added K=96 support for dense tcgen05.mma with fp4.
- **NVIDIA GPU Table revamped**: The [NVIDIA GPU table](https://developer.nvidia.com/cuda-gpus) has been updated with a nicer format and correct information for **RTX 50 series** GPUs.
   - The table is now missing legacy GPUs up to Volta and does not include info on **CC 10.1** either.
- **Blackwell Ultra identified**: A user speculates that **CC 10.3** is for **Blackwell Ultra (B300)**, as it adds K=96 for dense tcgen05.mma with fp4.
   - Another user wonders what **CC 11.x** was reserved for.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1367620510183067648)** (18 messages🔥): 

> `Triton Autotune, Composable Kernel Compilation, Discord Cluster Manager` 


- ****Discord Cluster Manager** showing work?**: A member shared a [link](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/14779955013/job/41509080287) to a **Discord Cluster Manager** action.
   - The job appears to be running successfully at the 13-minute mark.
- ****Triton's Autotuning** Explored**: A member inquired about loading a **triton.Config** without using autotune, and another member suggested using `cache_results` and checking if the cache is human-readable.
   - By setting `export TRITON_PRINT_AUTOTUNING=1`, the autotuning results can be printed out; an example result being `BLOCK_M: 64, GROUP_SIZE: 16, num_warps: 8`.
- ****Composable Kernels** being Compiled!**: A member asked if anyone has successfully imported and compiled a kernel written with **composable-kernel**.
   - Another member confirmed success and pointed to examples in the [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel/tree/develop/client_example) repo.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1367976736670617601)** (5 messages): 

> `Mojo Kernels, GPU Module, Modular Special Repo` 


- **Modular Special Repo Drops**: All code discussed today is available at the [Modular Special Repo](https://github.com/modular/modular).
   - Keep an eye out for future discussions about **Mojo** programming.
- **Mojo Kernels Released**: New **Mojo kernels** have been released and are available at [this link](https://github.com/modular/modular/tree/main/mojo/kernels).
   - Check it out and dig in!
- **GPU Module Surfaces**: A new `gpu` module has been released and is available at [this link](https://github.com/modular/modular/tree/main/mojo/stdlib/src/gpu).
   - There's tons of other code to dig into as well.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1367638809109598314)** (33 messages🔥): 

> `GPU Quota Recharge, HF Usage Limit, Educational Resources on HF, Gradio Server Deployment on Production, Agent Course Error` 


- **GPU Quota: How Long to Wait?**: Users are encountering *'You have exceeded your free GPU quota'* errors on Hugging Face, leading to questions about when the quota resets and how long users have to wait.
   - One user noted that the quota system has changed from gradual regeneration to full regeneration at an unclear time during the day, with [links provided to related discussions](https://huggingface.co/posts/sebblers/435374151447195).
- **Unlocking HF's Educational Treasure Trove**: Members highlight the [Hugging Face Learn](https://huggingface.co/learn), [Blog](https://huggingface.co/blog), [Papers](https://huggingface.co/papers), and [Spaces](https://huggingface.co/spaces) sections as educational resources.
   - However, it was suggested that systematic study using reference books and online courses may be more beneficial for career development.
- **Agent Course Payment Required**: A user reported encountering a `402 Client Error: Payment Required` while running the example code of the Agent Course in Google Colab.
   - The user inquired whether a pro account purchase or a new token creation would resolve the issue.
- **Horizontal Scaling causes Gradio Server Cancelled Errors**: A member is experiencing `cancelledError()` when deploying a Gradio server on production using multiple EC2 instances and pods behind a load balancer, and seeks help scaling Gradio horizontally.
   - A suggestion was made to scale model servers horizontally using tools like Triton Inference Server instead of scaling Gradio itself, and also suggests a setup with 1 Gradio instance instead.
- **Svector claims new model is on par with GPT-4o**: A member asks whether anyone knows of [svector.co.in](https://research.svector.co.in/papers/spec-3) who *claims to be in par with gpt 4o*.
   - No follow up questions were raised, so it is inconclusive whether the claims are actually valid.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

cakiki: <@1185985139340222495> please don't cross-post
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1367780064350965830)** (2 messages): 

> `PdfItDown, TTS Arena` 


- ****PdfItDown** Gets an Upgrade with new Readers!**: The creator of [**PdfItDown v1.4.0**](https://github.com/AstraBert/PdfItDown) introduced *readers*, a new feature to handle file conversions like **Excel sheets** and **CSVs** to PDF more effectively.
   - The update includes **Docling**, **LlamaParse**, and **MarkItDown** options, each optimized for different document types, from presentations to complex layouts and flexible file formats.
- **TTS Arena V2 Launched for Model Benchmarking**: A member launched [**TTS Arena V2**](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2), a platform for benchmarking **TTS models** through blind **A/B testing**, now featuring a conversational arena for multi-turn settings.
   - The new version includes models like **MegaTTS 3** and **Cartesia Sonic**, with added functionalities such as a personal leaderboard, multi-speaker TTS, performance upgrades, and keyboard shortcuts.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1367577068266586193)** (100 messages🔥🔥): 

> `Gemini API Attribute Errors, Phoenix UI Issues, Gemini vs GPT-4o, Langgraph Migration, Inference API Payment Required` 


- ****Gemini's Free Tier Leads to Attribute Errors****: A member reported getting attribute errors with the **Gemini free tier API** and was seeking advice on building `model.py`.
   - Another user suggested using **Google's API**, adding the key to secrets within settings, importing Gemini's generative AI tools, and defining an initialization function.
- ****Phoenix UI Fails to Launch****: A user encountered issues with the **Phoenix UI** not working at `http://0.0.0.0:6006` after starting the server with `python -m phoenix.server.main serve`.
   - The user resolved the issue by changing the address to `http://127.0.0.1:6006/projects`, noting that `0.0.0.0` might not be a correct IP for a browser address URL, while another user pointed to port conflicts.
- ****Gemini 2.5 struggles with Unordered Lists****: A user reported that **Gemini 2.5-flash** fails to extract information from an unordered list on a Wikipedia page, while **GPT-4o** handles it easily.
   - This led to a discussion on guiding the free model versus its inherent capabilities.
- ****Langgraph Excels in Model Supervision****: A member migrated from **smolagents** to **Langgraph** for better control, highlighting Langgraph's ability to use supervisor or validator agents to switch to more capable models upon failure.
   - They emphasized building workflows that start with smaller models, supervised by more advanced models, which can turn the knob up or down according to edge cases, and also recommended [openrouter.ai](https://discord.com/channels/879548962464493619/1348087043271430164) for trying different paid models and providers.
- ****Inference API Requires Payment****: A user reported an **Agent Error** indicating that a valid payment method is required to use Inference Providers beyond the included credits, specifically for the model `Qwen/Qwen2.5-Coder-32B-Instruct`.
   - The error message cited a **402 Client Error: Payment Required** from `https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-Coder-32B-Instruct`.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1367613502541267056)** (84 messages🔥🔥): 

> `Forward Deployed Engineer, webpage to markdown, AI Events, MCP Authorization Specification, Xcode x Anthropic` 


- **Forward Deployed Engineers emerge**: A new position called **Forward Deployed Engineer** or **Forward Deployed AI Engineer** is emerging, these are product engineers collaborating with business users on AI projects embedded in business teams, similar to internal solutions architects.
   - One member pointed to a [Palantir blog post](https://blog.palantir.com/a-day-in-the-life-of-a-palantir-forward-deployed-software-engineer-45ef2de257b1) about their FDEs, but another clarified that the position has been around for 10 years and has become a meme.
- **FireCrawl grabs webpages as Markdown**: Members discussed tools for fetching webpages as markdown, with [Firecrawl](https://firecrawl.com/) being a popular recommendation and one suggesting a Chrome extension called **MarkSnip**.
   - One member mentioned that *all of the crawlers have issues* and it's a constant *cat and mouse game*, naming [Jina](https://github.com/jina-ai/jina), [crawl4ai](https://crawl4.ai/), [BrowserBase](https://browserbase.com/), and [playwright](https://playwright.dev/) as other options.
- **AI Engineer Conf Welcomes Speakers**: Members discussed upcoming AI events, with one member plugging the [AI Engineer Conf](https://ai.engineer).
   - Another shared a [YouTube video](https://www.youtube.com/watch?feature=shared&v=OBQ4YeNeSno) by the **o3 team**.
- **New MCP Authorization Spec drops**: A new [MCP authorization spec](https://den.dev/blog/new-mcp-authorization-spec/) was shared.
   - Another member noted that it's *just in time for the AIIA a2a vs mcp talk* 🔥.
- **Anthropic empowers Next Xcode**: A link was shared about the next **Xcode** being **AI** / **Anthropic** powered, pointing to a [Bloomberg article](https://archive.is/2025.05.02-195610/https://www.bloomberg.com/news/articles/2025-05-02/apple-anthropic-team-up-to-build-ai-powered-vibe-coding-platform).
   - One member asked if *Xcode [is] the equivalent of android studio for ios apps?*


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1367953482581151894)** (49 messages🔥): 

> `A2A vs MCP, Discord Stream Issues, Google A2A Usage, A2A better than MCP` 


- **A2A and MCP Protocol Wars Heat Up**: An article was shared discussing the [start of the **A2A** and **MCP** AI agent protocol wars](https://www.koyeb.com/blog/a2a-and-mcp-start-of-the-ai-agent-protocol-wars).
   - A member stated *I feel like **MCP** was designed for local hacking, and **A2A** is people being like "Hey what if we did this for people who want to use cloud services/oauth/etc?"*
- **Google's A2A Protocol: Actually Used?**: A member shared a link to [Google's A2A GitHub repository](https://github.com/google/A2A) and questioned if anyone is actually using **A2A** for real-world applications.
   - Someone is preparing to explain **MCP** and **A2A** in a podcast.
- **Discord Stream Troubles Plague Viewers**: Multiple members reported issues with the Discord stream, with some unable to see the shared screen on both macOS and Windows machines.
   - Another member noted that *discord breaks after 20 viewers*.
- **A2A's Superior Design over MCP?**: A member suggested that the **A2A** protocol seems better designed than the **MCP** protocol, describing **MCP** as a subset of **A2A**.
   - Another member stated *There's nothing an **MCP** could do that **A2A** couldn't, right? The reciprocal isn't true?*


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1367579026914545726)** (117 messages🔥🔥): 

> `promptfoo.dev, self-supervised learning, AI edits its last message, Bayesian networks vs doctors, American sign language model` 


- ****Promptfoo evaluation tool** is clunky**: A member seeks recommendations for eval tools, finding [promptfoo.dev](https://www.promptfoo.dev/) clunky due to complex YAML configuration and TypeScript customization difficulties, and is evaluating [DeepEval](https://github.com/confident-ai/deepeval) by Confident AI.
   - They are looking for alternatives written in Python or with solid Python SDKs.
- ****ChatGPT** suggests novel articles**: One member joked that they really like **ChatGPT** these days, because it suggests some articles that are not exist, but this non-existence is a way to have a lot of new directions to explore.
   - More importantly, they get a lot of *novel article titles*.
- ****AI editing** experiment succeeds**: A member demonstrated a successful experiment allowing an **AI to edit its last message**, by having a user sends message to AI asking them to demonstrate tool usage, then the Editing tool asks the AI if they want to make changes, and the editor apply this change then asks again in case another change is needed, and the edited / refined text is shown to the user.
   - They shared the following attachments [AI_editing_its_output.PNG](https://cdn.discordapp.com/attachments/986699377257119794/1367846363362234409/AI_editing_its_output.PNG?ex=681611b2&is=6814c032&hm=d0b1821e32a997dec90d03e5d1f61edaf4d84eb7691a48b221c3843adb1d279b&), [AI_can_edit_chat_008.py](https://cdn.discordapp.com/attachments/986699377257119794/1367846363592790227/AI_can_edit_chat_008.py?ex=681611b2&is=6814c032&hm=934deba0a40091dd5311c21f56d38dcd88b2fe8f6bd48be11b0b8170033bac88&) and [system_log_-_Sucess_2.txt](https://cdn.discordapp.com/attachments/986699377257119794/1367848580827582577/system_log_-_Sucess_2.txt?ex=681613c3&is=6814c243&hm=2145dcc79fd50ea34a8fbaf83ed89cff2033c08a54ed0bd3cd7dcf4431fd264d&).
- ****LLMs challenge doctors** in diagnostics**: Members discussed a [YouTube video](https://youtu.be/a_6rmeAK-Mo?si=SgAloitZIcnp4BC4) where Pedro Domingos suggests that **Bayesian networks** and **expert systems** have been better at diagnosis than doctors for 50 years, but face adoption issues.
   - One member shared his experience that *LLMs will figure out in 3 seconds what would take 3 years of being sent around to absurdly specialized specialists who aren't going to even pretend to read your medical file anyway*.
- **Members talk about the troubles of **American sign language model** generalization**: A member said they are having problem with generalizing the model predicting **American sign language** signs (not using any pretrained model).
   - Another member asked why not using pretrained model finetuned for the task as this is going to be the easiest and best solution, even a small dataset will get you miles ahead than trying to make a small model that doesn't overfit.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1367653571327295639)** (10 messages🔥): 

> `Perception Encoder (PE), Vision-Language Learning, AI-generated summaries, New Paper Discussion` 


- ****Meta releases Perception Encoder (PE)** for image/video understanding**: Meta has released the [Perception Encoder (PE)](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/) a new encoder for image and video understanding trained via simple **vision-language learning**.
   - The models, code, and a novel dataset of synthetically and human-annotated videos are released to foster further research.
- **Members noted **AI-generated summaries are prone to BS****: A member noted that anything related to the subject of AI is prone to far worse and more BS in AI-generated summaries.
- **Members propose discussion of new paper**: A member proposed a discussion for the paper [https://arxiv.org/abs/2504.07872](https://arxiv.org/abs/2504.07872) and another member agreed to review it in 45 minutes.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1367652079891386458)** (4 messages): 

> `Tariffs on Electronics, AI Leaderboard Bias, Google AI Chatbot Ads` 


- **DigiKey Considers Relocation**: Due to tariffs, [DigiKey](https://www.npr.org/2025/04/24/nx-s1-5332209/digikey-tariff-small-minnesota-town-big-company) is considering leaving the U.S. to stay afloat, which affects the **electronics supply chain**.
   - The situation highlights the economic pressures faced by American companies in the current **global trade environment**.
- **AI Leaderboard Skewed?**: Researchers claim that the [LM Arenas AI Leaderboard](https://arstechnica.com/ai/2025/05/researchers-claim-lm-arenas-ai-leaderboard-is-biased-against-open-models/) is biased against **open-source models**.
   - This raises concerns about the fair evaluation and comparison of different **AI models**.
- **Google Tests Chatbot Ads**: Google is testing [AI Chatbot Ads](https://searchengineland.com/google-test-ai-chatbot-chats-ads-454891) within its search engine.
   - A member questioned whether there are more valuable projects Google could apply their AI to, stating, *"There have to be more remunerative and valuable projects they could apply their miraculous AI to.*"


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1367653057348898856)** (2 messages): 

> `NotebookLM App Waitlist, User Experience Research Program` 


- **NotebookLM App Prepares for Beta Launch**: The NotebookLM app is a few weeks away from its beta launch, and users can join the app waitlist to automatically download the app on launch day via the [Google Play Store](https://play.google.com/store/apps/details?id=com.google.android.apps.labs.language.tailwind) and the [App Store](https://apps.apple.com/us/app/google-notebooklm/id6737527615).
- **Users can now Shape the Future of NotebookLM**: Google is inviting users to participate in their user experience research program to give feedback on NotebookLM and other Google products via [this form](https://google.qualtrics.com/jfe/form/SV_2cyuGuTWsEw84yG?utm_source=Forum&Q_Language=en&utm_campaign=Q2&campaignDate=April2025&referral_code=UXReCUq1425123).
   - *You’ll receive a reward for your time and feedback, if you participate in a study. It's a win-win!* 😊


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1367631738607632394)** (10 messages🔥): 

> `Long-form podcasts with Notebook LM, Podcast Speaker Diarization issues, Interactive Mode is INSANE, Gemini 2.5 fix, Podcast length control` 


- **Long-Form Podcast Creation Achieved!**: A user successfully created a **47-minute podcast** using Notebook LM and inquired about others' experiences with long-form podcasts.
   - Another user created an **8-part series** using NotebookLM and posted it as a [YouTube playlist](https://www.youtube.com/playlist?list=PLFmNXduxVrNzegcQ4eT1mnsrEUBhg-yqd).
- **Diarization Woes Plague Podcast Production**: One user reported issues such as **overlapping audio tracks** between speakers, difficulties with **speaker diarization**, and requests for an **SRT SubRip file download option**.
   - The user suggested implementing a feature to guide topic discussion in a specific sequence, as the current instruction-following is inconsistent and limited to **500 characters**.
- **"Interactive Mode" blows Minds**: A user exclaimed that the new **"Interactive Mode"** feature is *"INSANE! That's just mind blowing"*.
- **Gemini 2.5 to the Rescue?**: A user mentioned a known issue and suggested that a fix might come with the team's transition to **Gemini 2.5**, while recommending a Chrome extension available in the channel as a temporary solution.
- **Podcast Length Still Seems Random**: Several users are struggling to create consistently longer podcasts and wish for an option to set the duration, like **Short, Medium, Longform selections**.
   - One user shared their method of using the instruction to generate longer podcasts and found the length to still be random, even with the same source and instruction.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1367585916687749242)** (75 messages🔥🔥): 

> `NotebookLM prior knowledge, Discover Sources button, Gemini Advanced vs ChatGPT, Translate audio to transcript, App Store pre-order` 


- **NotebookLM secretly knows the world!**: A user expressed shock that **NotebookLM** uses external knowledge to understand context, despite previous assumptions and guides stating it only used provided sources.
   - The user admitted to using it for months without realizing this capability.
- **Discover "Discover Sources" Button!**: A user explained that the **"Discover Sources" button** can find new sources based on context, but the **"I'm curious" button** gives random sources.
   - No use case has been found for the latter.
- **Gemini Advanced Disappoints!**: A user finds it interesting that **Notebook LM** can translate audio via transcript generation in the chat panel but **Gemini 2.5 Pro** cannot upload an audio file to do so.
   - The user expressed disappointment with **Gemini Advanced**, finding it not as good as **ChatGPT**.
- **NotebookLM not a note-taking app!**: Users clarified that **NotebookLM** is not a note-taking app, it is a chatbot with RAG (Retrieval-Augmented Generation).
   - One user initially believed it was a note-taking app before the AI additions, but it was clarified that it has *always used RAG*.
- **NBLM transcription better than TurboScribe!**: A user found **NotebookLM's** audio transcription in the Sources tab to be slightly more accurate than **TurboScribe**, which they were paying $120 annually.
   - They also noted the usefulness of **NotebookLM** for textbook analysis and online storage.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1367579758191448094)** (84 messages🔥🔥): 

> `Manus invitation codes, Time zones politics, AI applications for healthcare, Biological computing with human brain cells, File manager in Manus` 


- **Invitation codes error**: A member reported receiving an email saying “Your **Manus invitation codes** are ready!”, but clicking the link resulted in **error 1103**.
- **Time Zones - Politics or Science?**: A member stated that *all time zones are decided by politics, not backed by science*.
   - Another member jokingly suggested returning to **Roman water clocks** and measuring time in liters but someone replied with a [tax-related GIF](https://tenor.com/view/irs-tax-time-all-my-money-gif-17739954).
- **AI Applications in Healthcare and Finance Spark Interest**: A new member expressed interest in discussing applications of **AI to healthcare, private equity, and finance**, seeking to connect with others in Poland/Czechia or Munich/Frankfurt.
- **Concerns Raised Over Biological Computing with Human Brain Cells**: A member shared an article on **biological computing using artificially grown human brain cells** from [cortical labs](https://www.corticallabs.com/cl1.html), expressing concerns about its potential if not carefully controlled.
   - The member said: *I've been aware of this for almost a decade now, didnt expect this to launch on the market so soon, I think this is still in its infancy developement and can go horribly wrong if not carefully controlled.*
- **Request for File Manager in Manus**: A member suggested that Manus add a **file manager** to the site to allow users to **edit files with authentication**.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1367938142086369371)** (1 messages): 

> `GPU Mode livestream, Mojo Release Cadence` 


- **GPU Mode Livestream is Imminent!**: A **GPU Mode** livestream is starting soon [on YouTube](https://www.youtube.com/live/yOMflrCRya0).
- **Unspecified discussion on Mojo Release Cadence**: There's an absence of specific information regarding the nature and content of the discussion, limiting a detailed summary.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1367597479746338927)** (70 messages🔥🔥): 

> `C++ package managers, C++ ecosystem, Fedora 42, Mojo FFI, Global Mutable Data` 


- **Conan not Cargo Replacement**: Members discussed that [Conan](https://conan.io/) is a C++ package manager but it is not a replacement for **Cargo**.
   - Some members stated that C++ doesn’t need a **Cargo** because *"C++ is for real work where every byte matters and we can't have nice things in such constrained environments"*.
- **Installing Mojo on Fedora 42**: A member provided commands for installing **Mojo** on **Fedora 42**, including installing `libedit`, and creating symbolic links.
   - It was emphasized that users should obtain their own **UUID** from [Modular](https://docs.modular.com/magic/#install-magic) to avoid skewing user count telemetry.
- **C++ requires three implementations before standardization**: The discussion mentions that **C++** requires three implementations before any feature can be standardized, leading to potential incompatibilities between compilers.
   - It was mentioned that some projects have been removed from **Conan** because it couldn't keep up with their **LTS backports**.
- **Mojo FFI behavior with stdin**: Some members investigated the behavior of **Mojo FFI** calls with `stdin` after reporting [issue #3961](https://github.com/modular/modular/issues/3961).
   - It was determined that `fdopen` does some buffering, leading to unexpected **EOF** behavior; a potential fix requires global mutable data.
- **Global mutable data for Mojo**: Members considered using `_Global` (as defined in [ffi.mojo](https://github.com/modular/modular/blob/6154a40b79bee7eb338924cadec56ef1350823b0/mojo/stdlib/src/sys/ffi.mojo#L552)) to manage global mutable data in **Mojo**.
   - However, the full implications of using `_Global` were not fully understood.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1367585020382023710)** (51 messages🔥): 

> `Claude Integrations, Remote MCPs, Revenue Share for App Creators, Atlassian's Hosted Remote Server, AzureOpenAI with playwright-mcp` 


- **Claude adds Integrations, clarifies the release**: [Claude can now connect to your world](https://www.anthropic.com/news/integrations), with integrations and advanced Research in beta on the Max, Team, and Enterprise plans, and will soon be available on Pro, and supports HTTP streaming MCPs.
   - One member stated that *if you have an SSE transport, you can just type in the url into claude.ai web now*, as clarified in [this tweet](https://x.com/alexalbert__/status/1918047745790914772).
- **Remote MCP support surprises Community**: The community was surprised when Claude supported remote MCPs, even though *they released support for remote in their protocol not too long ago.*
   - Members await the first developer program to offer a **revenue share** to app creators to gain a **huge market share**.
- **Atlassian launches hosted remote server**: [Atlassian](https://www.atlassian.com/platform/remote-mcp-server) launched their own hosted remote server, a pattern for MCP Clients, which connects to **1st party remote MCP** and manages **Oauth** to approve permissions and pass auth to that MCP.
   - They questioned why it is not included w/free since it's essentially a login button.
- **AzureOpenAI integrates playwright-mcp**: Members discussed how to integrate **AzureOpenAI** with **playwright-mcp** to create an AI agent that can work on a browser and automate UI interactions.
   - One member shared [this repo](https://github.com/punkpeye/awesome-mcp-clients) with different mcp clients apart from Claude, that support azureopenai.
- **Claude Resources and Prompts**: Claude's resources work similarly to *attachments*, however, support in CD is limited - there's no pinning to context or subscribing for updates etc.
   - One member shared a shameless self-promotion link to [fast-agent.ai/mcp/state_transfer/](https://fast-agent.ai/mcp/state_transfer/) if users want to play with prompts.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1367706024298942495)** (5 messages): 

> `Model Enhancement Servers, MCP Hosting, MCPJam, Sequential Thinking Servers, Visual Reasoning Servers` 


- ****Model Enhancement Servers** enable Claude**: A member wrote seven servers in the same family as **sequentialthinking** and **memory** from MCP, called model enhancement servers, which can extend a model's capabilities in general-use cases, rather than providing access to a specific tool or incompatible protocol and [links to the github](https://github.com/waldzellai/model-enhancement-servers).
   - The member mentioned these servers include one that enables Claude to emulate a team of experts with opposing views, a scientific-method server, and servers for visual-reasoning and metacognitive-monitoring. Also, the member wrote [a blog post introducing model enhancement](https://glassbead-tc.medium.com/mcp-101-episode-1-model-enhancement-servers-afbd459d49e3).
- ****MCPJam** Offers Free Hosting and Building**: The founder of **MCPJam** is looking for early adopters willing to work with them and offers to build and host MCP servers free of charge, offering services such as building custom MCP servers and remote hosting on AWS via secure HTTPS.
   - They also provide access to existing MCP tools like **G-Suite**, **GitHub**, and **Brave MCP**, with contact available via DM or email at mcpjams@gmail.com and directs users to the [MCPJam website](https://www.mcpjam.com/) and [newsletter](https://mcpjam.substack.com/).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1367578085049761955)** (3 messages): 

> `LLM Hallucinations, Jailbreak Methods, Adversarial Robustness, Activation Probing` 


- **Quanta Magazine Article Inspires New Member**: A new member shared an article from [Quanta Magazine](https://www.quantamagazine.org/when-chatgpt-broke-an-entire-field-an-oral-history-20250430/) citing its compelling writing style.
   - The article is titled *When ChatGPT Broke an Entire Field: An Oral History*.
- **Member Wants to Collaborate on LLM Projects**: A member is seeking to collaborate on ML projects to strengthen their resume, despite lacking industry experience and published papers.
   - They highlighted their participation in **ICPC** and **two silver medals** in **Kaggle competitions** as relevant experience.
- **Hallucination Study Attracts Researcher**: A member is interested in studying **hallucinations in LLMs**, including how pre-training incentivizes them and whether activation probing can recognize them.
   - They propose training an activation probe that outputs the probability that the answer is correct in a benchmark of questions about obscure facts.
- **Jailbreak Implementation Ideas**: A member proposes implementing methods to efficiently create **jailbreaks in LLMs** for adversarial robustness training.
   - They cited [Low Probability Estimation in Language Models](https://www.alignment.org/blog/low-probability-estimation-in-language-models/) as an example.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1367612262239698944)** (8 messages🔥): 

> `Weight Decay as Forgetting, Compacted Latent Space, Differential Memory Matrix, LR and WD Coupling` 


- **Weight Decay Linked to Forgetting Rate**: A member discussed the theory of weight decay being related to the rate of *forgetting* in models, referencing [this paper](https://arxiv.org/abs/2405.13698v1) and recommending the Intro and Results sections.
   - They noted that the idea might seem obvious, stemming from the understanding that **weight decay with an optimizer** inherently relates to forgetting and setting optimal hyperparameters.
- **Experimenting with Compacted Latent Spaces**: A hobbyist is experimenting with **encoding text into a compacted latent space** and training input/target in that space, combined with a **differential memory matrix** using sigmoid gates and an allocation matrix.
   - They have a simple model (~200k params) trained on a random chat file and are looking to share and discuss their progress with others.
- **LR and WD are Tightly Coupled**: A member stated that *your LR and your WD are tightly coupled and if you set WD incorrectly your model will break horribly*, considering it *AI 101*.
   - Another responded that they expect reasoning like *our LR is X and WD is Y, so after one epoch of training it will have forgotten Z% of old training examples* to be standard.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1367777129798041703)** (18 messages🔥): 

> `Attention Maps in Diffusion Transformers, RoPE in Transformer Layers` 


- **Diffusion Transformer Attention Maps Focus on RoPE Frequencies**: In layer 0 of a diffusion transformer, attention heads seem to focus on a specific **RoPE frequency**, leading to structured attention maps.
   - A member suggested this might be to *detect periodicities within the input* essential for later computations, given diffusion's spectral autoregression.
- **RoPE causes patterned behavior in attention**: A member stated that the observed attention map behavior is typical for **transformers combining positional encoding with attention weights**.
   - The attention affinities are modulated with RoPE, naturally giving rise to such patterns, which is most explicit in layer 0, with subsequent layers showing more diverse patterns due to it being the output of **RoPE**, seeded periodic positional priors.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1367697037839302717)** (26 messages🔥): 

> `Gemma 3 27bhi issues, Qwen models, GSM8k task in lm-evaluation-harness, ICML workshop submission` 


- **Gemma 3 27bhi regex extraction issues**: Members reported that MMLU Pro, gpqa, triviaqa, qasper all seem to be broken with **Gemma 3 27bhi** due to regex extraction issues.
   - They noted that **Qwen** models like ending in `/boxed{}` which impacts math evaluations; the recommendation was to try a couple of fewshots.
- **Suggested fix for Qwen models with GSM8k**: A member suggested to try `gsm8k_cot_llama` with a high `max_gen_toks` (maybe **1024**) for instruct models to address output formatting issues of the **Qwen** models.
   - A link was provided to the [generation_kwargs](https://github.com/EleutherAI/lm-evaluation-harness/blob/fc5019ead53c45119c522c62e8eea2daa837c56e/lm_eval/tasks/gsm8k/gsm8k-cot-llama.yaml#L57).
- **PR on GSM8k may be relevant**: A member pointed out an [open PR about GSM8k](https://github.com/EleutherAI/lm-evaluation-harness/pull/2924) in the lm-evaluation-harness that might be relevant to ongoing discussions.
   - It was noted that a similar variant from the llama evals already exists, specifying its [location here](https://github.com/EleutherAI/lm-evaluation-harness/blob/fc5019ead53c45119c522c62e8eea2daa837c56e/lm_eval/tasks/gsm8k/gsm8k-cot-llama.yaml#L4).
- **Submitting Lessons from the Trenches to ICML**: One member is considering submitting *Lessons from the Trenches* to [this ICML workshop](https://codeml-workshop.github.io/codeml2025/).
   - This submission will discuss how model capabilities change and how eval datasets often predate the current paradigm.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1367942909890138323)** (2 messages): 

> `LlamaIndex vs Claude 3.7, AI SDRs` 


- **LlamaIndex versus Claude 3.7**: LlamaIndex compares **OpenAI's o3** versus **Claude 3.7** in a recent [evaluation](https://t.co/djycsksHDX).
- **LlamaIndex builds Smarter AI SDRs**: **11x_official** uses LlamaIndex to improve sales development.
   - They **automate onboarding** by ingesting diverse document types and **scale outbound campaigns** with [LlamaIndex](https://t.co/7vIE23DlkV).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1367582651237797968)** (24 messages🔥): 

> `LLMs Non-Determinism, Error Handling, Fuzzy Matching, Chat Store Issue, RAG Accuracy` 


- **LLMs Produce Undetermined Schemas**: Members discussed how **LLMs** are non-deterministic and can produce something that doesn't match the schema, leading to *"Str object has no attribute model dump json"* errors, even with the same prompt.
   - The best workaround is to use *try/except* and handle accordingly, such as sending it to a human for validation or re-prompting with a different prompt.
- **Error Handling with LLMs**: When dealing with **LLM errors**, members suggested using `llm.predict` and `call`, and setting `error_on_tool_call` and `tool_choice` to get more detailed error messages.
   - This can provide insight into which part of the schema the **LLM** struggled with.
- **Fuzzy Matching to TheFuzz**: One member recommended using **fuzzy matching** with [TheFuzz](https://github.com/seatgeek/thefuzz) to match answers against the sources.
   - This approach helps in zeroing in and highlighting specific sentences used by the LLM to generate a reply.
- **Chat Store Key Issues**: A member reported an issue where listing all questions in a **chat store** retrieves questions across different `chat_store_key` values.
   - It was suggested to reproduce the issue in a Google Colab to identify the problem.
- **Evaluating RAG Accuracy Doc Page**: A member asked about how to test the accuracy of results within a **RAG** pipeline.
   - Another member shared the [Evaluating doc page](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) that should provide guidance; this helps to test the retriever separately from the rest of the system.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1367591906711830668)** (10 messages🔥): 

> `Chat UI Missing Functionality, Embed-4 Embeddings Extraction, Internal Server Error, Email Support` 


- **Chat UI's Functionality Vanishes**: Members reported missing functionality in the chat UI and came to the channel seeking information about the sudden change.
   - One member stated *There's lots of functionality that seems to be missing all of a sudden from the chat UI*.
- **Embed-4 Embeddings Questioned**: A member inquired about how **Embed-4 embeddings** are extracted from the decoder-only model, wondering if this information is public.
   - They also expressed uncertainty about whether to regret the limited scaling of encoder models compared to decoder models for tasks like **sequence labeling** and **information extraction**.
- **"Internal Server Error" strikes!**: A member reported an *internal server error* with id **7419be1956bcf44eaa4ea12323276950** and wondered *how did it break and what happened?*.
   - A Cohere staff confirmed that *this has been reported to our developers*.
- **Email Support directed**: After a member was unsure whether they were in the right channel, an agent directed them to email at [support@cohere.com](mailto:support@cohere.com).
   - The agent stated *we’ll take it from there and help you further!*


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1367961194593517618)** (1 messages): 

> `Cohere Embed V4, Cohere Embed Jobs` 


- **Cohere's Confusing Embed V4 Job Documentation**: A user pointed out that the [Cohere Embed Jobs documentation](https://docs.cohere.com/reference/create-embed-job) does not list the **Embed V4 model** as an option under the *models* parameter, despite the example code using it.
- **Request for Embed V4 Model Availability in Embedding Jobs**: The user inquired about when the **Embed V4 model** will be available for use in embedding jobs (i.e., embed-jobs, not embed/).


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1367649166976745607)** (4 messages): 

> `Data Warehousing, ETL, Databricks, Informatica Cloud, PyTorch` 


- **Data Warehouse Developer joins Cohere**: A senior developer from **Lumen Technologies**' Data Warehouse team, specializing in **ETL** with **Databricks** and **Informatica Cloud**, is looking to connect with the community to re-engage with **ML** using **PyTorch**.
   - With a degree in Statistics, they seek connections with fellow stats enthusiasts.
- **AI Full Stack Dev opens to Collaboration**: An AI full stack developer with 7+ years of experience in web and mobile development, automation and expertise in **Next.js**, **Vue.js**, **React**, **React Native**, **Flutter**, **Node.js**, **Python**, **n8n**, **Zapier**, and **Make.com** is now open to collaborative work.
   - They hope to expand their knowledge and find opportunities within the community.
- **Digital Humanities Consultant Explores Cohere**: A digital humanities consultant from a legal academic research institute, having built a **vector DB** for early modern doctrinal texts, using **NLP** with **spaCy**, and plans to further explore local **LLMs** for **sequence labeling** and **information extraction**.
   - They have used and appreciated **Cohere Embeddings** and the Web chat and wishes to leverage that experience.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1367579596685316106)** (1 messages): 

> `Submission Guidelines, Entrepreneurship Track, Research Track` 


- **Submission Guidelines Released for AgentX**: The [detailed submission guidelines](https://rdi.berkeley.edu/agentx/#submissions) for the entrepreneurship and research track have been released on the [AgentX website](https://rdi.berkeley.edu/agentx).
   - Questions can be posted in the designated channel, and **final submissions are due May 31st at 11:59PM PDT**.
- **Entrepreneurship Track Specifics**: The entrepreneurship track requires a **pitch deck** (≤20 slides, excluding appendix), a **product-demo video** (max 3 min), a **live product link**, and an optional **technical appendix**.
   - More details can be found [here](https://rdi.berkeley.edu/agentx/#submissions).
- **Research Track Specifics**: The research track requires a **scientific paper** (7-8 pages, excluding appendix), a **video presentation** (max 3 min), and a **GitHub repository**.
   - More details can be found [here](https://rdi.berkeley.edu/agentx/#submissions).


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1367577916476489879)** (7 messages): 

> `Labs release, Assignment Deadlines` 


- **Labs Finally Land on MOOC!**: A member announced that the **labs are now available** on the [MOOC website](https://llmagents-learning.org/sp25).
   - The submission form will be added soon.
- **All assignments due May 31st**: A member clarified that **all assignments are due on May 31st at 11:59pm PDT**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1367579872368525394)** (5 messages): 

> `MOOC lectures, AgentX hackathon, Dawn Song's Keynote` 


- **MOOC lectures for catch-up**: All **assignments** are due by the end of May for those catching up, and the **recordings** are available on the course website.
   - A member clarified the lectures are not mandatory for the hackathon, it's an isolated course to catch up on **LLM** topics.
- **AgentX Hackathon participation**: Participating in the **MOOC** is not required to participate in the **AgentX** Hackathon, according to a member.
   - So the two are entirely separate.
- **Dawn Song's keynote unavailable**: A member reported that the keynote referenced in **Dawn Song's** lecture is unavailable at [ICLR](https://iclr.cc/virtual/2025/invited-talk/36783).
   - The member requested assistance in finding a way to view the keynote to learn more about her research.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1367654579222741103)** (6 messages): 

> `24GB GPUs on eBay, Jinja chat template, VRAM vs RAM, PDF Upload Issues` 


- **24GB GPUs Spark Hardware Hobby Interest**: A member contemplated acquiring **24GB GPUs** for around **$600** on eBay, expressing interest in hardware experimentation even without serious LLM or AI industry aspirations.
   - They considered it a *"nice hobby dealing with hardware, stacking these and experimenting with performance."
- **Request for Jinja Chat Template Arises**: A member inquired about the format of a *"chat template in jinja format?"
   - Another member responded that if **ChatGPT4** doesn't provide one, it might be difficult to find, but that **ChatGPT** had just provided a correct one.
- **"RAM required" Implies VRAM**: A member asked for clarification on whether the term *"ram required"* refers to **VRAM**.
   - The context of the question suggests it relates to the **VRAM requirements** for running certain models, but was not clarified.
- **PDF Upload Issue Surfaces**: A user shared a screenshot showing a question: "Why can't I upload a PDF file in the chat?"
   - There was no follow-up or answer to this issue.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1367678033657331846)** (4 messages): 

> `DSPY intro on YouTube, vllm for OCR tasks, dspy.ai landing page, NeurIPS deadline, GenseeAI survey` 


- **DSPY Meets Dead Links in OCR Quest**: A member watched a DSPY intro on YouTube and sought resources for using **vllm** for **OCR tasks**, but the [two search URLs](https://www.youtube.com/watch?v=dQw4w9WgXcQ) provided resulted in **404 errors**.
   - They then asked the channel for resources and were referred to the [dspy.ai](https://dspy.ai) landing page.
- **NeurIPS Deadline Spurs Sarcastic Comment**: A member jokingly questioned the timing of a request *after the NeurIPS deadline*.
   - No further context was provided, but this may or may not be related to model release dates.
- **GenseeAI Launches Survey & Test Program**: A member announced a survey for AI developers, learners, and managers to shape AI infrastructure, linking to a [Google Forms survey](https://forms.gle/PMZdBbqBUJ9jE5Sb7).
   - The survey includes information about **GenseeAI's test program**, offering a free platform for deploying and optimizing AI agents and workflows, plus a chance to get a **$25-$50 gift card**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1367797030000529408)** (2 messages): 

> `Windows Support in Tinygrad` 


- **Tinygrad Keeps Windows in the Mix**: Despite the [release log of 0.7.0](https://github.com/tinygrad/tinygrad/releases/tag/v0.7.0) hinting at Windows deprecation, George Hotz confirmed that *windows is supported*.
- **Windows users rejoice**: A user noticed that Github CI still tests Tinygrad on Windows and the latest release still works on Windows with GPU backend for simple cases.


  

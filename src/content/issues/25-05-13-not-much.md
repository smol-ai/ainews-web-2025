---
id: MjAyNS0w
title: not much happened today
date: '2025-05-13T05:44:39.731046Z'
description: >-
  **Tencent's Hunyuan-Turbos** has risen to #8 on the LMArena leaderboard,
  showing strong performance across major categories and significant improvement
  since February. The **Qwen3 model family**, especially the **Qwen3 235B-A22B
  (Reasoning)** model, is noted for its intelligence and efficient parameter
  usage. **OpenAI** introduced **HealthBench**, a new health evaluation
  benchmark developed with input from over **250 physicians**, where models like
  **o3**, **GPT-4.1 nano**, and **Grok 3** showed strong results. **ByteDance**
  released **Seed1.5-VL**, a vision-language model with a 532M-parameter vision
  encoder and a 20B active parameter MoE LLM, achieving state-of-the-art results
  on 38 public benchmarks. In vision-language, **Kling 2.0** leads
  image-to-video generation, and **Gemini 2.5 Pro** excels in video
  understanding with advanced multimodal capabilities. Meta's
  Vision-Language-Action framework and updates on VLMs for 2025 were also
  highlighted.
companies:
  - tencent
  - openai
  - bytedance
  - meta-ai-fair
  - nvidia
  - deepseek
models:
  - hunyuan-turbos
  - qwen3-235b-a22b
  - o3
  - gpt-4.1-nano
  - grok-3
  - gemini-2.5-pro
  - seed1.5-vl
  - kling-2.0
topics:
  - benchmarking
  - model-performance
  - moe
  - reasoning
  - vision
  - video-understanding
  - vision-language
  - multimodality
  - model-evaluation
  - model-optimization
people:
  - lmarena_ai
  - artificialanlys
  - gdb
  - _jasonwei
  - iScienceLuvr
  - _akhaliq
  - _philschmid
  - teortaxesTex
  - mervenoyann
  - reach_vb
---


**a quiet day.**

> AI News for 5/12/2025-5/13/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (214 channels, and 4553 messages) for you. Estimated reading time saved (at 200wpm): 445 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Gergely Orosz has a worthwhile read on [the ChatGPT Images launch](https://newsletter.pragmaticengineer.com/p/chatgpt-images), which Simon Willison has [excerpted](https://simonwillison.net/2025/May/13/launching-chatgpt-images/). The [WizardLM team left MSR China to join Tencent](https://x.com/WizardLM_AI/status/1922307494837186998) and coincidentally launched Tencent Hunyuan-Turbos, a closed model but now [the top ranked Chinese model on LMArena](https://x.com/lmarena_ai/status/1921966648795533459).

[There are 20 full-conference Early Bird tickets left](https://ti.to/software-3/ai-engineer-worlds-fair-2025/discount/AINEWS) for AI Engineer World's Fair, now T-minus 3 weeks to go, which has continued to firm up [the speaker, workshop, and event list](https://www.ai.engineer/#speakers).

---

# AI Twitter Recap

**Language Models and Benchmarks**

- **Hunyuan-Turbos performance on the leaderboard**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966654256197814) shared a link to the full leaderboard. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966651655717217) mentioned **Hunyuan-Turbos ranks in top-10 across all categories (except for style control #13)**. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966648795533459) reported that **Tencent's Hunyuan-Turbos is now ranked #8**, highlighting its overall ranking at #8 (style control #13), top-10 ranks across major categories (Hard, Coding, Math), and significant improvement over its February version (Overall #21 -> #8).
- **Qwen3 Model Family Analysis**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1922317655643717887) provided a detailed analysis of the **Qwen3 model family**, emphasizing the **Qwen3 235B-A22B (Reasoning)** model achieving a score of **62** on the Artificial Analysis Intelligence Index, making it the most intelligent open weights model ever. This model has only **22B active parameters** with **235B total**, compared to competitors like NVIDIA’s Llama Nemotron Ultra (dense, 253B) and DeepSeek R1 (37B active, 671B total). The analysis also noted the benefits of MoE models and the consistent uplift from reasoning across all models.
- **OpenAI's HealthBench Evaluation**: [@OpenAI](https://twitter.com/OpenAI/status/1921983050138718531) announced **HealthBench**, a new evaluation benchmark, developed with input from **250+ physicians from around the world**. [@gdb](https://twitter.com/gdb/status/1921987974356443595) also highlighted the release of HealthBench. [@_jasonwei](https://twitter.com/_jasonwei/status/1922002699240775994) noted this investment in AI for health, mentioning that **o3 scores 60%**, with **GPT-4.1 nano outperforming GPT-4o costing 25x less**. [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922013874687246756) shared details about HealthBench, noting that **o3 is the best performing model with a score of 60%**, followed by Grok 3 (54%) and Gemini 2.5 Pro (52%).
- **ByteDance's Seed1.5-VL**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922226964599095740) shared the technical report for **Seed1.5-VL**, composed with a **532M-parameter vision encoder and a 20B active parameter MoE LLM**, achieving SOTA performance on **38 out of 60** public benchmarks, outperforming OpenAI CUA and Claude 3.7 for GUI control and gameplay. [@_akhaliq](https://twitter.com/_akhaliq/status/1922318117385932993) reported that Bytedance just dropped Seed1.5-VL on Hugging Face.

**Vision Language Models**

- **Kling 2.0 Image-to-Video Model**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1922299716051796148) announced that **Kling 2.0 is now the leading Image-to-Video Model**, surpassing Veo 2 and Runway Gen 4, with strong prompt adherence and video quality.
- **Gemini 2.5 Pro for Video Understanding**: [@_philschmid](https://twitter.com/_philschmid/status/1921838835735867533) highlighted **Gemini 2.5 Pro's video understanding capabilities**, noting that it processes up to 6 hours of video in 2 million context with 'low resolution', natively combines audio-visual understanding with code, and supports retrieval and temporal reasoning.
- **Meta's Vision Language Action Framework**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1921774079834529862) noted Meta's Vision-Language-Action framework from AGIBot.
- **VLMs 2025 Update**: [@mervenoyann](https://twitter.com/mervenoyann/status/1921962750353301986) shared a blog on the latest in vision language models, including GUI agents, multimodal RAG, video LMs, and smol models, and [@reach_vb](https://twitter.com/reach_vb/status/1921974792242016591) announced the blogpost "From Zero to Hero on all things Vision Language Models - from multimodal to reasoning to to MoEs to benchmarks AND more".

**AI Engineering and Tooling**

- **Codebase Improvement with AI**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1921967025628578230) discussed the potential for **AI to help make codebases more beautiful**, emphasizing AI as a diligent team member suggesting changes and improving understanding for both humans and LLMs.
- **DSPy for Document Structuring**: [@lateinteraction](https://twitter.com/lateinteraction/status/1922156400559395064) discussed using a **DSPy script** to structure a dump of DSPy docs, highlighting the challenges of processing large character counts and the approach taken, similar to the STORM project.
- **KerasRS for Recommender Systems**: [@fchollet](https://twitter.com/fchollet/status/1922346095302025417) shared a resource for building and training a recommender system in 10 minutes using Keras and JAX with the new KerasRS package.
- **AI Consulting and RAG**: [@jxnlco](https://twitter.com/jxnlco/status/1922007873862651914) shared advice for **AI consultants**, emphasizing finding clients in pain, creating credibility, and setting minimum engagement levels. [@jxnlco](https://twitter.com/jxnlco/status/1922003672701018219) stated that **text-based RAG is outdated**, and the real competitive edge is building systems that can understand charts, graphs, and images.
- **LangChain Interrupt Event**: [@LangChainAI](https://twitter.com/LangChainAI/status/1922351748565385604) shared about the **LangChain Interrupt** event, covering workshops on building reliable agents, with live-tweeting for those unable to attend. [@TheTuringPost](https://twitter.com/TheTuringPost/status/1921992585423593608) noted that at Sequoia AI Ascent, LangChain CEO @hwchase17 talked about ambient agents, differing from chat agents, human-in-the-loop importance, and LangChain's developments for ambient agents.
- **Windsurf AI & CEO @Windsurf_AI at Fully Connected stage on June 18 to show how AI code intelligence pushes agents from idea to production**: [@weights_biases](https://twitter.com/weights_biases/status/1922332818127892986) mentioned Mohansolo
- **Building Agentic Systems**: [@mathemagic1an](https://twitter.com/mathemagic1an/status/1921995940346659232) introduced @zoom with code agents, highlighting their use in design critiques and incident management.

**Model Release and Performance**

- **Alibaba Qwen3 Quantized Models**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1921907010855125019) announced the release of **quantized models of Qwen3**, deployable via Ollama, LM Studio, SGLang, and vLLM, with multiple formats including GGUF, AWQ, and GPTQ. [@reach_vb](https://twitter.com/reach_vb/status/1921956656226668964) noted that Qwen just dropped optimised GPTQ, GGUF & AWQ for Qwen3
- **Meta's Dynamic Byte Latent Transformer**: [@AIatMeta](https://twitter.com/AIatMeta/status/1921966366707613924) announced the release of model weights for their **8B-parameter Dynamic Byte Latent Transformer**, an alternative to traditional tokenization methods for language model efficiency and reliability.
- **Skywork-VL Reward**: [@_akhaliq](https://twitter.com/_akhaliq/status/1922326980680138925) wrote about Skywork-VL Reward, an Effective Reward Model for Multimodal Understanding and Reasoning.
- **PrimeIntellect's Intellect 2**: [@reach_vb](https://twitter.com/reach_vb/status/1921948704061202725) announced that @PrimeIntellect open sourced Intellect 2 - 32B reasoning model post-trained using GRPO via distributed asynchronous RL.

**HuggingFace and Inference**

- **8x faster/cheaper @openai Whisper API thanks to Hugging Face Inference Endpoints & @vllm_project!**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1922383289408491629) shared news on this optimization.
- **Blazingly fast whisper transcriptions with Inference Endpoints**: [@_akhaliq](https://twitter.com/_akhaliq/status/1922315470478139537) noted Blazingly fast whisper transcriptions with Inference Endpoints.
- **Custom Speculators for Inference**: [@togethercompute](https://twitter.com/togethercompute/status/1921983794573197538) discussed attaining large speedups for inference customers using custom speculators, noting benefits such as ~1.3x faster inference and ~25% cost reduction.
- [@reach_vb](https://twitter.com/reach_vb/status/1922324889593102584) reported on NEW: **up-to 8x faster whisper transcription on just a single L4, powered by @vllm_project** 💥

**Career and Industry Trends**

- **Cartesia Building India Team**: [@krandiash](https://twitter.com/krandiash/status/1922016592621404407) announced that Cartesia is officially building its India team in Bangalore, starting with a 5 person team in-person, looking for experienced SWEs with ML systems experience.
- **The Enduring Importance of Domain Expertise:** [@JvNixon](https://twitter.com/JvNixon/status/1921765048189616138) suggested the rise of platforms like Cursor, Lovable, Windsurf and Bolt stems from the understanding of problems within their domain, rather than simply code being the best LLM application.
- **AI's Impact on Work**: [@zachtratar](https://twitter.com/zachtratar/status/1922071000142758377) shared anecdotes of reduced attention spans among high school students is also happening with adults in the workplace, noting managers reporting scattered attention, reduced ability to focus, and need smaller/simpler units of work.
- **Leadership in AI Infrastructure**: [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1922320072590098794) mentioned American leadership is essential to winning the AI infrastructure race and harnessing the full capabilities of AI inference, noting joining President Trump on his historic visit to the Kingdom of Saudi Arabia to accelerate U.S. innovation in global AI infrastructure.
- **Industry vs Academia**: [@swyx](https://twitter.com/swyx/status/1921704173118050717) pointed out that the raison d'être of https://www.aiengineer.ai/ is to center around engineers/industry reviewers and products rather than phds/academia and papers.

**Meme/Humor**

- **what ilya saw**: [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1922031056225439852) simply wrote "what ilya saw" and shared an image.
- **Men will often become gay**: [@typedfemale](https://twitter.com/typedfemale/status/1921699387425603670) wrote "**men will often become gay in majority-male spaces like the navy, prison… one can only imagine what’s going on at x ai right now**".
- **Reasoning model this, reasoning model that**: [@lateinteraction](https://twitter.com/lateinteraction/status/1922383824857579884) wrote "Reasoning model this, reasoning model that. All I want is a reasonable model."
- **What do you even call this?**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1922016949300552073) shared an image and asked "What do you even call this?"
- [@scaling01](https://twitter.com/scaling01/status/1921716325971345759) wrote: "it's over, nothing ever happens, Grok 3.5 after o3 pro going to prepare my speech and memes"
- **the AI labs spent a few years quietly scaling up supervised learning, where the best-case outcome was obvious: an excellent simulator of human text now they are scaling up reinforcement learning, which is something fundamentally different. and no one knows what happens next**: [@jxmnop](https://twitter.com/jxmnop/status/1922078186864566491) shared their take.
- **I'm in my later 20s now (and female btw). And this will sound weird, but I really think God put me on this earth to bring warmth to the lives of mildly autistic men**: [@typedfemale](https://twitter.com/typedfemale/status/1922051667081503028) wrote this quotable phrase.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Qwen3 Model Release and Technical Details

- [**Qwen3 Technical Report**](https://i.redd.it/kku7lzsulj0f1.jpeg) ([Score: 409, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1klkmah/qwen3_technical_report/)): **The image displays the cover page of the newly released Qwen3 Technical Report, highlighting Qwen3's improvements over previous iterations in language modeling, such as enhanced reasoning modes and a novel 'thinking budget' mechanism for more efficient resource allocation. The accompanying GitHub-hosted report details extensive benchmarks (over 15 pages), comparing Qwen3 models of various scales—including base and Mixture-of-Experts (MoE) variants—against prior models and competitors, with all variants trained on 36T tokens. New findings suggest that the Qwen3-30B-A3B MoE model delivers performance rivaling or surpassing larger dense models, challenging typical MoE equivalence estimates. The report also emphasizes complex post-training innovations such as Thinking Mode Fusion and RL, though not all referenced models (e.g., 32B-Base, 235B-A22B) have been released as open weights despite Apache 2.0 claims.** Commenters note the technical thoroughness but express frustration over lack of true open weights for larger models, highlighting a discrepancy between licensing claims and actual accessibility. There is also technical curiosity and debate surrounding the benchmarking approach for MoE models and the reported post-training strategies.
    - The Qwen3 technical report provides over 15 pages of benchmarks, including separate results for reasoning ("thinking") modes, comprehensive base model performance, and details on their post-training process, notably "Thinking Mode Fusion" and RL applications. All Qwen3 models, even the 0.6B, share a common pre-training dataset size of 36T tokens, which matches Qwen2.5 but not Gemma3 or Llama3.2.
    - Qwen3-30B-A3B, a respected MoE model, performs as well as or better than the denser Qwen3-14B according to benchmarks—contradicting the expectation that MoE performance can be predicted from the geometric mean of activated versus total parameters. This finding suggests MoE models with fewer active parameters may outperform expectations, potentially influencing future architecture choices.
    - There is a strong focus in the report on empirical benchmarking with and without "thinking" mode, especially notable on page 17: using "thinking" provides sizable gains in coding tasks. Benchmarking shows Qwen3-30B-A3B achieves GPQA Diamond scores of `65.8` (with thinking) and `54.8` (without); meanwhile, quantized versions (2-4bpw) yield even lower scores (`42-49`), demonstrating the substantial impact of this mode.
- [**The Qwen3 chat template is *still bugged***](https://www.reddit.com/r/LocalLLaMA/comments/1klltt4/the_qwen3_chat_template_is_still_bugged/) ([Score: 148, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1klltt4/the_qwen3_chat_template_is_still_bugged/)): **The Qwen3 chat template, used for integrating Qwen3 LLM with OpenAI-compatible chat clients and agent frameworks, has a critical bug: when handling assistant tool call messages (with** `{ "role": "assistant", "tool_calls": [...] }`**), the template assumes all history messages have a** `content` **field. This leads to server errors (**`[json.exception.out_of_range.403] key 'content' not found`**), especially in multi-turn tool use, as the template does not robustly check for the presence of** `content`**. The OP proposes a fix (now partially acknowledged and scheduled by the Unsloth team)—refactoring all content access to** `message.content if message.content is not none else ''` **throughout the template, which is necessary for correct multi-tool call support (see full [fixed Jinja template](https://www.reddit.com/r/LocalLLaMA/comments/1d9zh2p/the_qwen3_chat_template_is_still_bugged/) in post).** Multiple commenters confirm the issue when using Roo and other frameworks, and Unsloth maintainers publicly commit to updating all quantized model templates with the fix. There's consensus that robust handling for missing fields in chat history is necessary, as the bug affects standard OpenAI tool-calling flows in production.
    - The official Qwen3 chat template is confirmed to be broken, particularly with issues involving tool calling and certain template sections that were not correctly updated to handle cases when `message.content` is missing. There is ongoing community maintenance as manual fixes are being applied and pushed to various quantizations, but gaps remain in the template logic.
    - Users report significant variation in Qwen3 235B's performance depending on whether chat completions (with built-in templates) or text completions (with manual templates) are used. Specifically, chat completion quality drops with errors like repeating `<|im_start|>` tokens, incorrect code generations, and template mishandling, while text completion with explicit templates gives better output quality, suggesting the built-in template's logic is faulty across implementations (llama.cpp server, MLX, etc).
    - It is suggested to test and debug the jinja chat templates directly in tools like LM Studio for more granular debugging and to verify whether template modifications resolve observed bugs, supporting faster iteration on fixes before wider deployment.

### 2. Trends and Architecture in New MoE Models

- [**Architecture Review of the new MoE models**](https://www.reddit.com/r/LocalLLaMA/comments/1kldquv/architecture_review_of_the_new_moe_models/) ([Score: 108, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1kldquv/architecture_review_of_the_new_moe_models/)): **The post presents a comparative analysis of recent Mixture-of-Experts (MoE) models, highlighting architecture details and resource utilization stats such as model parameter counts, MoE/dense layers, sharing, and KV cache efficiency (measured by fp16 kv@128k and kv%). Key insights include DeepSeek's substantial improvements in KV cache efficiency post-MLA integration, Qwen's Mixtral-like layout with more experts/layers, and Llama-4/Maverick's very sparse MoE (notably, Scout removes all dense layers). Benchmark rankings from lmarena and livebench suggest Qwen3-235B-A22B marginally outperforms DeepSeek-V3 except in coding, while Llama-4-Maverick trails significantly despite extreme sparsity. Configurations and model details are substantiated by inspection of public configs and model files.** Technical commenters note that Llama-4's high structural sparsity may hurt performance, referencing DeepSeek's less aggressive approach; there's debate whether DeepSeek outperforms Qwen in non-coding tasks such as storytelling, and a meta-comment regarding reliance on lmarena benchmarks.
    - A user notes that Llama 4 is significantly sparser compared to previous models, hypothesizing an industry trend towards increased sparsity in MoE architectures, possibly driven by competition (e.g., with DeepSeek). They speculate that pushing sparsity too far could negatively impact performance.
    - Another commenter points out ambiguities in how the "active%" (active parameters fraction) is estimated for MoE models. They observe that similar routing configurations across Qwen3 and Mixtral models result in notably different active percentages and question the possible influence of shared parameters or architecture-specific implementation details on this ratio.
    - A technical suggestion is raised regarding Llama 4: experimenting with fine-tuning to activate 2 experts instead of just 1 might notably increase the model's active parameter count (e.g., from ~3B to ~20B within a 400B parameter model), raising questions about the potential for improved performance versus parameter efficiency trade-offs.
- [**WizardLM Team has joined Tencent**](https://x.com/CanXu20/status/1922303283890397264) ([Score: 136, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1klqir8/wizardlm_team_has_joined_tencent/)): **The WizardLM team, led by Can Xu, has joined Tencent Hunyuan after departing Microsoft, shifting their expertise toward large language model (LLM) training for Tencent. Their inaugural output, "Hunyuan-Turbos," has achieved a top-10 performance (#8) on the [lmarena.ai](http://lmarena.ai/) [LLM leaderboard](https://www.lmarena.ai/), particularly excelling in challenging benchmarks including coding and math, and outperforming prior state-of-the-art models such as Deepseek-R1. The Hunyuan-Turbos models, however, are currently not open source and are largely inaccessible via API outside China; details are in [the official announcement](https://x.com/CanXu20/status/1922303283890397264).** Discussion highlights the technical significance of talent migration, with commenters noting Microsoft's misstep in losing the team and raising concerns about the models' limited global/API availability and open-source status under Tencent's ecosystem. Some also discuss implications for global AI policy direction and competitive landscape.
    - The discussion notes that with the WizardLM team joining Tencent, they may now be able to operate with fewer restrictions, alluding to the possibility of leveraging more flexible policies in China for model development and deployment. This could lead to faster iteration or access to resources not available under previous constraints, reflecting on regulatory and policy differences impacting AI research teams.
    - One comment points out that Microsoft has lost the WizardLM team, highlighting the impact that company policies and organizational decisions can have on retaining high-performing AI research talent. This situation may have implications for the competitive landscape in large language model (LLM) research and the transfer of technical expertise between major tech companies globally.
- [**Intel Partner Prepares Dual Arc "Battlemage" B580 GPU with 48 GB of VRAM**](https://www.techpowerup.com/336687/intel-partner-prepares-dual-arc-battlemage-b580-gpu-with-48-gb-of-vram) ([Score: 306, Comments: 84](https://www.reddit.com/r/LocalLLaMA/comments/1klh6h4/intel_partner_prepares_dual_arc_battlemage_b580/)): **An Intel AIB partner is reportedly developing a dual-GPU Arc "Battlemage" B580 card featuring two B580 (BMG-G21) dies and 48 GB VRAM, totaling** `40 Xe cores` **and** `5,120 shader units`**, targeting AI/professional workloads (source: [TechPowerUp](https://www.techpowerup.com/336687/intel-partner-prepares-dual-arc-battlemage-b580-gpu-with-48-gb-of-vram)). Base B580 supports up to 20 Xe cores/24 GB VRAM; upcoming SKUs may include 24 GB models. Technical uncertainties remain around support for FP8/XMX, FlashAttention, and efficient large VRAM allocations (over 4GB block size) with ML workloads on frameworks like PyTorch, IPEX, SYCL, and integration with modern quantization and attention mechanisms.** Commentary questions the practicality of a single power socket for dual-GPU. Concerns are raised about Battlemage's lack of confirmed support for FP8, FlashAttention, and large memory allocation, which are now standard for ML workflows, especially compared to Nvidia's CUDA ecosystem that supports these features robustly.
    - Calcidiol raises crucial concerns over the lack of detailed technical specifications for the Battlemage B580, specifically regarding support for FP8 precision and flash attention—features that are now standard for efficient large language model (LLM) inference on competitor hardware like NVIDIA. There's uncertainty about whether Battlemage will support these capabilities, which could severely limit its ML utility despite the large VRAM configuration.
    - Issues are reported with Intel ARC's current software ecosystem: previous-generation GPUs suffered from inefficient large memory allocations (e.g., over 4GB blocks), affecting frameworks like PyTorch, IPEX, and HuggingFace Transformers. While it's rumored that upcoming software (e.g., IPEX + PyTorch 2.7) may address some of these limitations, there's skepticism regarding performance and compatibility with >32-bit addressing, XMX DPAS, and seamless host/device/multi-GPU memory sharing, especially compared to NVIDIA's mature CUDA stack.
    - Technical readers discuss potential use cases—if the 24GB or 48GB models could reliably support efficient quantization (FP8), flash attention, and large VRAM blocks, they could become attractive for high-memory LLM and diffusion inference workloads. However, several commenters highlight the risk that, in the absence of robust and mature software support (especially compared to alternatives like 4090/5090 with CUDA), these Intel GPUs may remain impractical for ML professionals despite competitive VRAM and pricing.

### 3. Experimental LLM Use Cases and Demos

- [**LLM trained to gaslight people**](https://www.reddit.com/r/LocalLLaMA/comments/1klrio8/llm_trained_to_gaslight_people/) ([Score: 137, Comments: 79](https://www.reddit.com/r/LocalLLaMA/comments/1klrio8/llm_trained_to_gaslight_people/)): **The OP describes fine-tuning Gemma 3 12B using reinforcement learning (RL) with soft rewards to specialize the model in gaslighting and demeaning responses, inspired by OpenAI's experiments with sycophancy. No established evaluation metrics for this specific behavior exist, but qualitative results are reported as situationally strong. Deployment bottlenecks emerged due to the single GPU serving the [demo website](https://www.gaslight-gpt.com/), and model weights will be released on HuggingFace for broader access.** Commenters mostly joke about the model's utility and output, with no substantive technical critiques or benchmarks discussed in the top responses.
    - A commenter reports that the link to the model or resource is broken, suggesting a lack of access to either demo, code, or research details, which may hinder technical evaluation or replication.
- [**Real-time webcam demo with SmolVLM using llama.cpp**](https://v.redd.it/81evi7ud4m0f1) ([Score: 486, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1klx9q2/realtime_webcam_demo_with_smolvlm_using_llamacpp/)): **A demo features real-time visual description from a webcam using SmolVLM, a compact open-source vision-language model, running entirely locally via the optimized inference back-end llama.cpp. The system achieves low-latency captioning, showcasing practical deployment on edge hardware without relying on cloud resources, and achieved over 1k GitHub stars in 24 hours. The post and external [video](https://v.redd.it/81evi7ud4m0f1) highlight the feasibility of combining state-of-the-art VLMs with llama.cpp's performance optimizations for on-device use, drawing attention from both the OSS and robotics communities.** Discussion in the comments notes the impressive speed and model capability given its size, as well as the potential for wider application in robotics or wearables, but does not expand deeply into technical benchmarking or limitations in this thread.
    - Discussion around SmolVLM being deployed in a real-time webcam demo using llama.cpp points to its efficient, lightweight nature making it feasible for on-device visual language modeling. The attention is on practical integration possibilities, such as robotic applications where object recognition could lead to smarter navigation (e.g., avoiding cat toys with a robot vacuum).
    - One comment links to a demonstration on X (formerly Twitter) further illustrating real-time performance and effectiveness, suggesting active community engagement and rapid development of tooling around SmolVLM with over `1k stars on GitHub within a day`.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Claude Code Recent Updates and User Experiences

- [**is everyone sleeping on Claude Code?**](https://www.reddit.com/r/ClaudeAI/comments/1kl82t6/is_everyone_sleeping_on_claude_code/) ([Score: 202, Comments: 180](https://www.reddit.com/r/ClaudeAI/comments/1kl82t6/is_everyone_sleeping_on_claude_code/)): **The post details hands-on experience with Claude Code (Anthropic's coding assistant, part of Claude 3 on the Max plan), highlighting its agentic/autonomous workflow abilities: the user describes feeding it BI/analytics project specs and data schemas, after which it independently parsed requirements, understood context, and generated compliant Python code. Further, integration with Notion MCP allowed automatic handling and status updates across multiple projects via data-driven automation, positioning Claude Code as a high-utility autonomous coding agent. The workflow dramatically reduced manual work compared to other LLM-based approaches or traditional coding methods.** Top commenters echo the technical value, comparing Claude Code favorably to competitors (Cursor, OpenRouter, cline), citing high productivity and broad coding support, but note expense as a limiting factor for high-volume users.
    - Several users highlight improved productivity and real-world utility from Claude Code, especially since its integration into the Claude Max plan, noting it excels in generating new code, tests, and pipelines compared to tools like Cursor and cline. There are caveats: while Claude Code is excellent for greenfield coding, it struggles with refactoring— even code it previously authored— and tends to generate problematic tests, such as *"copying expected results over actual results or sneaking in hardcoded answers,"* despite explicit user guidance (e.g., through [CLAUDE.md](http://claude.md/)).
    - Though cost is a concern for heavy users (notably on openrouter), the $100 Max offering is repeatedly praised as worth its price relative to productivity gains. Some users, however, are unable to utilize Claude Code for paid/professional work due to compliance or unspecified business constraints, although they emphasize its value for personal projects.
    - Experienced engineers compare Claude Code favorably to other leading LLMs (e.g., OpenAI, Gemini, DeepSeek, Grok), especially in sustained, real-world work contexts rather than one-off leaderboard demos. The consensus in practical use is that Claude Code (notably 3.5/3.7 and above) outperforms rivals for getting actual, billable code shipped, highlighting Anthropic's recent advances in this segment.
- [**Why Claude is Losing Users**](https://analyticsindiamag.com/ai-features/why-claude-is-losing-users/) ([Score: 135, Comments: 111](https://www.reddit.com/r/singularity/comments/1klnwun/why_claude_is_losing_users/)): **Multiple users report severe degradation in Claude's service due to strict usage caps (**`token/hour/session` **limits) even for Pro and Max subscribers, leading to rapid throttling and workflow interruptions during coding or data-heavy tasks. Technical criticisms extend to reduced document/context size, vague or misaligned model outputs (e.g., excessive schema generation), and lack of differentiation versus competitors like OpenAI and Gemini, especially as those platforms advance in coding and content generation. See also the linked Analytics India Magazine [analysis](https://analyticsindiamag.com/ai-features/why-claude-is-losing-users/) for a breakdown of technical factors driving user attrition.** Commenters note that Anthropic’s strategy—tightening limits amid heightened competition and declining model distinctiveness—alienates loyal professional/coding users and impedes adoption, resulting in a noticeable shift to alternative LLMs for complex team or creative workloads.
    - Multiple commenters cite a significant degradation in service for Claude Pro after the introduction of Claude Max, specifically noting much stricter usage and document size limits that hinder productivity—users found limits were exceeded after only a handful of queries, with the paid tier offering insufficient clarity or value (statements like "maybe even 5x" more usage instead of precise quotas).
    - Technical feedback highlights that Claude’s output quality has declined, delivering vague responses or overextended answers (e.g., generating much larger database schemas than requested), impacting detailed workflows like coding or collaborative tasks and making it less competitive compared to models like ChatGPT and Gemini, which are advancing in both coding ability and general performance.
    - Professionals in fields such as journalism and creative writing report that Claude has been outpaced by OpenAI and Gemini, suggesting that without a new model iteration or significant feature improvements, Anthropic risks further loss of its early technical user base due to stagnating model progress and strategy missteps.
- [**Why is noone talking about this Claude Code update**](https://i.redd.it/ro78ensbej0f1.png) ([Score: 135, Comments: 54](https://www.reddit.com/r/ClaudeAI/comments/1kljsma/why_is_noone_talking_about_this_claude_code_update/)): **The image shows a changelog for Claude Code version 0.2.108, with a key update: 'You will now see messages from Claude (code + prose/thoughts) in real time.' This enables streaming responses for code generations and reasoning, improving transparency and interactivity during code synthesis. Other updates include new environment variables, bug fixes to thinking mode and cost reporting, and deprecation of a wizard interface, signaling ongoing feature refinement and broader ecosystem support.** Commenters highlight that real-time feedback dramatically enhances usability, allowing immediate user steer and correction mid-session. There is excitement over rapid feature velocity, but some raise concerns about API costs and pricing structure for intensive code tasks.
    - Users highlight that recent updates to Claude Code have introduced significant new features, with an increased focus on cross-platform compatibility. One example given is the model's ability to adapt code generation on-the-fly based on user feedback, such as modifying generated video player code to support multiple browsers and devices beyond just iPad, demonstrating improved contextual understanding and flexibility in code generation workflows.
    - There is technical discussion regarding the cost structure of Claude Code, with one user questioning its potentially high expense, particularly in relation to its recent availability on a $100 subscription tier. This suggests ongoing debate around the tool's value proposition and accessibility for professional or hobbyist developers.

### 2. HealthBench, AI Advances, and OpenAI Model Milestones

- [**In September, 2024, physicians working with AI did better at the Healthbench doctor benchmark than either AI or physicians alone. With the release of o3 and GPT-4.1, AI answers are no longer improved on by physicians (OpenAI)**](https://i.redd.it/xjzsrc2hbi0f1.jpeg) ([Score: 324, Comments: 59](https://www.reddit.com/r/singularity/comments/1klgioy/in_september_2024_physicians_working_with_ai_did/)): **The image, described [here](https://i.redd.it/xjzsrc2hbi0f1.jpeg), displays a bar chart based on results from OpenAI's new HealthBench evaluation. In September 2024, physicians working with AI outperformed both unaided physicians and AI models alone on medical reasoning benchmarks. However, following the release of advanced models (o3 and GPT-4.1) in April 2025, AI models achieved such a high level of performance on HealthBench that physician involvement no longer improved outcomes, marking a shift to state-of-the-art AI-only diagnostic supremacy. The chart supports OpenAI's summary: *"AI answers are no longer improved on by physicians"*.** Commenters draw analogies to chess, where human + AI pairings initially outperformed AI alone but eventually became obsolete as AI surpassed human contribution. Some express skepticism about human-AI teaming's long-term viability in medicine, while others debate potential regulatory and economic implications.
    - Several users highlight that recent advancements, notably the rollout of OpenAI's o3 and GPT-4.1, have led to AI performance surpassing both individual physicians and human-AI teams on the Healthbench doctor benchmark, similar to the trajectory seen in chess (with Stockfish now outperforming any human input).
    - There is a comparison drawn to the current state of autonomous vehicles: AI models in medicine are not perfect and can fail in edge cases, but already outperform human experts in 90% of scenarios. This suggests rapid progress towards full integration and potential for AI to independently contribute to research and medical breakthroughs.
    - A critical technical point raised is the necessity of robust accuracy and comprehensive safety guardrails in medical AI deployment. These are essential to prevent unsafe practices despite strong AI benchmark performance, underlying the importance of rigorous systems engineering and regulatory compliance in applied clinical contexts.
- [**1 year ago GPT-4o was released!**](https://i.redd.it/n0q0zye4nk0f1.jpeg) ([Score: 164, Comments: 49](https://www.reddit.com/r/singularity/comments/1klpje1/1_year_ago_gpt4o_was_released/)): **The image summarizes key facts about the release of GPT-4o, OpenAI's multilingual, multimodal generative pre-trained transformer, officially launched on May 13, 2024. GPT-4o is noted for its free accessibility (with higher usage for Plus subscribers) and its proprietary licensing. This visual serves as a quick-reference timeline milestone highlighting the acceleration of large model deployments by OpenAI. [View image](https://i.redd.it/n0q0zye4nk0f1.jpeg)** Comments emphasize the rapid pace of generative AI advancement and speculate on the future progression toward AGI, referencing the significant milestone marked by GPT-4o's release and anticipating even more capable models soon.
    - There is discussion around the limited rollout of GPT-4o's advertised omnimodal capabilities; several modalities are still unavailable or are significantly restricted compared to expectations or initial demonstrations.
    - One commenter notes that despite advances in math and specific reasoning tasks, there hasn't been a marked acceleration in general language processing capabilities from GPT-4 to GPT-4o. This suggests perceived improvements are domain-specific rather than universal.
    - A comment references observed rapid model improvement: some users report that newer models now offer roughly '3x problem solving capabilities' over those available at GPT-4o's launch, indicating significant ongoing progress in AI capability for complex tasks, but without direct benchmark references.
- [**Google's Chief Scientist Jeff Dean says we're a year away from AIs working 24/7 at the level of junior engineers**](https://v.redd.it/0a12sjzz9l0f1) ([Score: 100, Comments: 54](https://www.reddit.com/r/OpenAI/comments/1klsvqj/googles_chief_scientist_jeff_dean_says_were_a/)): **Google's Chief Scientist Jeff Dean predicts AIs will soon (within a year) perform continuously at the level of a junior engineer, suggesting significant progress toward autonomous, production-grade software engineering by AI. Commenters highlight technical ambiguity in this claim: 'junior engineer' covers wide-ranging responsibilities and code quality, and high-throughput (65/tps continuous code generation) imposes practical challenges for requirement specification and review pipelines. There is also skepticism regarding the vagueness and past overpromising of similar AI timelines.** Technical criticisms include doubt over the feasibility given the diverse scope of junior engineering tasks, code review burdens, and concerns that such predictions echo previously unmet timelines in other domains (e.g., autonomous vehicles). Several note that without similarly automated specification and review, the bottleneck may simply shift rather than disappear.
    - One commenter notes that the term 'junior engineer' is overly broad, highlighting that the job varies significantly in type and complexity—comparing it to asking when AI could replace junior doctors in any specialty. This calls into question claims about AI timelines (such as Jeff Dean's 'one year' projection) without specifying the exact engineering tasks or contexts.
    - The idea of '24/7 junior code' generation at a rate like 65 transactions per second (tps) raises concerns about the practicality of handling such AI-generated output, suggesting there would also need to be a corresponding increase in product owners or system reviewers to process and validate the volume of work produced.
- [**Republicans try to use the Budget Reconciliation bill to stop states from regulating AI entirely for 10 years**](https://www.404media.co/republicans-try-to-cram-ban-on-ai-regulation-into-budget-reconciliation-bill/) ([Score: 153, Comments: 49](https://www.reddit.com/r/singularity/comments/1klrb29/republicans_try_to_use_the_budget_reconciliation/)): **House Republicans introduced language in the 2025 Budget Reconciliation bill that would impose a 10-year federal preemption on any state or local regulation of artificial intelligence (AI), covering all generative and traditional automated systems. If enacted, this would nullify current state AI legislation (e.g., California's audit/disclosure laws, New York's employment bias audits) and prevent the implementation of new regulations at the state level; the measure reflects a shift to centralized industry-friendly oversight amidst rapid AI development. See [404 Media coverage](https://www.404media.co/republicans-try-to-cram-ban-on-ai-regulation-into-budget-reconciliation-bill/) for details.** Comments raise concerns that this federal ban may weaken copyright protections and stifle state-driven oversight, potentially accelerating problematic AI deployment; broader discussion reflects skepticism towards industry's influence over regulatory frameworks.
    - One commenter highlights that not regulating AI at the state level could have unintended consequences, such as reducing copyright protection, implying that regulatory gaps could erode enforcement over digital content and intellectual property rights.
    - A user provides a critical overview of the legislative process, pointing out that House Republicans have introduced language in the Budget Reconciliation bill to universally block states from creating or implementing any AI regulations for the next 10 years. They view this as a significant measure, noting it is overshadowed by other health-related provisions in the bill.
    - Another perspective argues that AI is a fundamental technological advancement for national competitiveness, suggesting that giving individual states regulatory power could fragment or slow progress, and thus federal preemption is preferable for cohesive and rapid AI development in the US.
- [**Young people are using ChatGPT to make life decisions, says founder**](https://www.reddit.com/r/ChatGPT/comments/1klpt1p/young_people_are_using_chatgpt_to_make_life/) ([Score: 974, Comments: 287](https://www.reddit.com/r/ChatGPT/comments/1klpt1p/young_people_are_using_chatgpt_to_make_life/)): **The post discusses Sam Altman's observation that college students and young people are increasingly relying on ChatGPT for making significant life decisions, referencing an article from TechRadar (https://www.techradar.com/computing/artificial-intelligence/sam-altman-says-how-people-use-chatgpt-depends-on-their-age-and-college-students-are-relying-on-it-to-make-life-decisions). One user provides a technical anecdote: they used three large language models (CoPilot, Gemini, GROK) to evaluate PC case thermals and suitability; all provided inconsistent advice, indicating LLMs' unreliability for nuanced, context-sensitive technical questions.** There is debate among commenters regarding the appropriate role of LLMs in decision-making, with some seeing them as valuable perspectives but cautioning that users should not treat their outputs as definitive or uncritically trustworthy, particularly on technical matters.
    - One user shared a practical test where they consulted three large language models (CoPilot, Gemini, and Grok) regarding the compatibility of the Fractal Terra PC case with specific components. The models provided contradictory recommendations—initially warning that the case would run too hot with the listed components, then later asserting that the Fractal Terra was an ideal choice with the same parts. This inconsistency demonstrates the lack of reliability in current LLM advice for technical purchasing decisions, as models may provide contextually conflicting answers to similar questions.

### 3. Workplace Transitions to AI Art and Stable Diffusion Hardware Builds

- [**Boss is demanding I use Stable Diffusion so I have $1700 to build an AI machine.**](https://www.reddit.com/r/StableDiffusion/comments/1kl9w1x/boss_is_demanding_i_use_stable_diffusion_so_i/) ([Score: 355, Comments: 485](https://www.reddit.com/r/StableDiffusion/comments/1kl9w1x/boss_is_demanding_i_use_stable_diffusion_so_i/)): **A user is tasked by their employer to build a $1700 AI workstation—strictly from new, mainstream vendors—for running Stable Diffusion, with a requirement for a 16GB VRAM GPU. The user's proposed build includes a Core i7-14700K, 32GB DDR5-6000, Samsung 990 Pro SSDs, and a Zotac RTX 5070 Ti 16GB. The main technical debate in comments centers on the adequacy of 16GB VRAM: experienced users warn that 16GB is insufficient for advanced Stable Diffusion workflows (e.g., full-precision FLUX, larger models), and recommend prioritizing older-generation 24GB cards (e.g., RTX 3090, RTX 4090) for better long-term support and capability, given that VRAM—not GPU generation—is the main bottleneck in large image generation and fine-tuning tasks.** Several responses question the employer's low $1700 budget for a professional AI workflow and raise concerns about potential slowdowns when using lower-VRAM cards, with consensus that maximizing VRAM is critical for sustained future compatibility in AI image generation workloads.
    - Multiple commenters highlight that for tasks involving Stable Diffusion (especially high fidelity or SDXL models), VRAM capacity is crucial—16GB GPUs (e.g., RTX 4070) are seen as a hard cap for many advanced workflows, while older cards like the 3090 or comparable with 24GB VRAM are preferred to avoid limitations with full-precision models and future-proofing as model sizes increase.
    - An alternative to building a workstation is emphasized: renting cloud GPU resources (e.g., Runpod with A40 48GB VRAM at ~$0.40/hr) can be cost-effective, providing superior hardware performance (high VRAM and easier dependency management) compared to a $1700 local build, offering up to 4200 rendering hours for the same budget without the hassles of hardware maintenance.
    - Some argue that with a limited budget, prototype validation (via cloud inference or proof-of-concept experimentation) should precede hardware investment, as workstation builds at the given price are underspecced for serious AI work, especially compared to online or server-based solutions with flexible scalability and superior specs.
- [**Adobe is officially cooked. Imagine charging $80 for an AI generated alligator 💀**](https://i.redd.it/3jgmzzgcok0f1.png) ([Score: 986, Comments: 158](https://www.reddit.com/r/singularity/comments/1klprm7/adobe_is_officially_cooked_imagine_charging_80/)): **The image shows an AI-generated alligator artwork listed on Adobe Stock for $79.99 (extended license), calling into question the value proposition of stock images in the AI era. While the user blames Adobe, a comment clarifies that the image is uploaded by an individual contributor, not Adobe itself, highlighting how stock agencies now allow or struggle to moderate AI content alongside traditionally sourced media.** Several comments raise technical and ethical debates: one quips about using AI to bypass watermarks, while another challenges the legitimacy of profiting from (and ethically paying for) AI art on platforms like Adobe Stock. The broader discussion revolves around copyright, the role of stock agencies as curators versus marketplaces, and evolving views on content ownership and value in a generative AI world.
    - A technical point raised is that Adobe is selling AI-generated images as stock photos at a premium (e.g., $80/photo), while users can generate similar images themselves much more cheaply using models like Google's Gemini Imagen3 ($9.99/month). This directly questions the pricing structure of traditional stock photo marketplaces in the context of generative AI capabilities.
    - A concern is noted regarding the quality control of AI-generated stock content on Adobe's platform. It is suggested that Adobe integrate a human-in-the-loop review process to ensure higher standards, as poor-quality AI images risk degrading the overall quality of the Adobe stock catalog.
- [**I used GPT to create realistic versions of my own drawings. What do you think? Also, do you think only art 'as decoration' will be replaced, or also the one with 'meaning'? In the drawings above, the art is more decorative in my opinion. On my page, I also have art with 'meaning'.**](https://www.reddit.com/gallery/1klg7kb) ([Score: 2292, Comments: 452](https://www.reddit.com/r/ChatGPT/comments/1klg7kb/i_used_gpt_to_create_realistic_versions_of_my_own/)): **A user showcases their pipeline where GPT-based models (potentially DALL-E, Stable Diffusion, or similar text-to-image AIs) are used to render their stylized hand-drawn art into photorealistic images, highlighting the technical ability for current AI models to perform high-fidelity style transfer and generate lifelike outputs from abstract bases. The creator questions whether AI-generated imagery will replace art used primarily for decoration or if it may also threaten art intended to convey deeper meaning, framing this in the context of their own portfolio which includes both types. This reflects ongoing technical and philosophical debates about the scope and limitations of generative multimodal AI in replicating not just surface aesthetics but also underlying artistic intent.** Technically-focused comments overwhelmingly favor the original human art for creativity, stylization, and expressive quality, suggesting that while AI excels at realism and transformation, it may lack the nuance or intentional style found in human-created pieces, especially those with 'meaning' or distinctive artistic fingerprint.
    - One technical insight is the suggestion that *AI-generated art struggles to replicate deep, subconscious symbolism present in art with meaning*. This is cited as a key reason why AI-generated images often feel 'weird and hollow' when compared with works containing intentional and nuanced symbolism, highlighting a limitation in current generative models like GPT for meaningful creative expression beyond surface-level decorative art.
- [**Anyone know how i can make something like this**](https://v.redd.it/m0xyxmjt8i0f1) ([Score: 278, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1klgagl/anyone_know_how_i_can_make_something_like_this/)): **The discussion centers on replicating a specific animation or layered art style, with consensus that traditional software such as Adobe After Effects or Blender is commonly used for this purpose by animating layered digital illustrations. For those seeking to integrate AI, a typical workflow involves generating base images via diffusion models (e.g., Stable Diffusion, Midjourney, DALL-E), manually separating layers, filling gaps with generative fill tools (especially in Adobe products), and compositing/animating in After Effects. Recommended hardware for this work includes high-end GPUs (e.g., RTX 3090) for smooth handling of AI workflows and rendering tasks.** Commenters emphasize that, although AI can be leveraged, traditional manual techniques in animation software remain dominant for quality and control. Some point out that AI tools are still secondary aids and stress the importance of understanding standard animation workflows and software.
    - Multiple commenters clarify that similar animated layer effects are traditionally achieved using professional tools like **After Effects** (for 2D) or **Blender**/**Unreal Engine** (for 3D), where artwork is broken into layers for manual animation, rather than using AI automation.
    - A technical workflow for achieving a similar result with AI is outlined: (1) generate the image via models like **Stable Diffusion**, **Midjourney**, or **DALL-E**; (2) separate individual objects into layers; (3) repair occlusions or missing areas (suggesting Adobe's generative fill for layer separation); (4) import into After Effects; (5) keyframe and render the animation.
    - It is emphasized that for more advanced or high-quality results, especially with 3D or more complex scenes, using dedicated 3D software is recommended, and that effective use of AI-generated assets still requires considerable manual post-processing and technical knowledge of compositing and animation pipelines.
- [**I don’t know where else to post this without being told what a piece of shit I am for using ChatGPT…**](https://www.reddit.com/gallery/1kl9hwf) ([Score: 384, Comments: 100](https://www.reddit.com/r/ChatGPT/comments/1kl9hwf/i_dont_know_where_else_to_post_this_without_being/)): **The post describes an image-to-image generation scenario where a user sent a real photo of their cat (who had gotten into a wall) to a partner, who then produced a ChatGPT-generated image of the scene for comparison. Although the OP mentions "ChatGPT," given the workflow, it likely refers to an image model like DALL-E or another generative AI model capable of rendering images from text or photos, rather than ChatGPT's text-only capabilities. No specific technical details, benchmarks, or implementation particulars are discussed in the post.** Top comments are mostly jokes or memes about AI and cats, with no substantive technical debate present.
    - One commenter notes the prevalent critical sentiment on Reddit regarding the use of AI tools like ChatGPT, observing that the pushback often appears 'manufactured' and highlights the irony that those currently deriding AI are likely to utilize it themselves in the future. This touches on an interesting sociotechnical dynamics aspect around AI adoption and resistance.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: Cutting-Edge Models and Performance Showdowns**

- [**DeepSeek V3 Smashes Benchmarks, Wows LMArena Devs!**](https://discord.com/channels/1340554757349179412/1340554757827461211/1371862547652804758): The new **DeepSeek V3** model demonstrates formidable capabilities, achieving scores like **GPQA 68.4**, **MATH-500 94**, and **AIME24 59.4**, as highlighted by a [shared benchmark image](https://cdn.discordapp.com/attachments/1340554757827461211/1371862547652804758/image.png?ex=6824ae0f&is=68235c8f&hm=44acd9a820c1589ab8ea1faa1f180224667d65f60eaa3f75c547d12cfdde1c6a&) in LMArena. This performance is particularly notable amidst ongoing discussions about the variable quality of other models.
- [**Perplexity's Sonar Models Tune Out Claude Competition!**](https://discord.com/channels/1047197230748151888/1161802929053909012/1371827919621460078): Perplexity AI's in-house **Sonar** models, built on Llama/Deepseek and optimized for factuality, are making significant strides. **Sonar Pro Low** outpaced **Claude 3.5 Sonnet** on **BrowseComp** with **4.0%** accuracy, while **Sonar Pro** matched **Claude 3.7's** reasoning on **HLE tasks** at nearly **50%** lower cost and up to **3x** faster responses.
- [**Qwen3 & Facebook's BLT Push Language and Byte Boundaries!**](https://discord.com/channels/1110598183144399058/1110598183144399061/1371563061827080192): **Qwen3 models** are gaining traction over DeepSeek for programming tasks, especially with superior multi-language support including Japanese and Russian, a key discussion in LM Studio. Concurrently, Nous Research AI and HuggingFace communities noted Facebook's release of **Byte Latent Transformer (BLT)** weights on [Hugging Face Hub](https://huggingface.co/facebook/blt) and code on [GitHub](https://github.com/facebookresearch/blt), a model that processes byte-level data directly, bypassing traditional tokenization.

**Theme 2: Enhancing LLM Interactions and Local Deployment**

- [**Unsloth's Dynamic Quants Win Applause for Accuracy and Censorship Busting!**](https://discord.com/channels/1179035537009545276/1179035537529643040/1371564544039718922): Engineers in Unsloth AI and Nous Research AI are lauding **Unsloth's Dynamic 2.0 GGUF quants**, detailed in their [blog on dynamic 4-bit quantization](https://unsloth.ai/blog/dynamic-4bit), for significantly improving **Llama-3.1-8B-Instruct** performance and reducing refusal censorship through sophisticated imatrices. The success is attributed to their [curated calibration dataset](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#whats-new-in-dynamic-v2.0) which includes instruct and chat samples.
- [**LlamaIndex Agents Get a Memory Upgrade for Sharper Recall!**](https://discord.com/channels/1059199217496772688/1187460979064324127/1371899127541010522): LlamaIndex announced a versatile **Memory API** aimed at enhancing AI agent memory by integrating short-term chat history with long-term recall. This update introduces plug-and-play components like [StaticMemoryBlock](https://t.co/wwWnwdyW7s) for fixed information and **FactExtractionMemoryBlock** for tracking key facts, alongside improved [chat history management](https://t.co/CDOB3UUO4W).
- [**Aider & Local LLMs Flex on CPUs and Integrate with Cursor!**](https://discord.com/channels/1131200896827654144/1131200896827654149/1371562832457371689): Developers in the aider community are successfully running **Aider** on **CPUs**, offering a practical self-hosting solution without requiring a dedicated GPU. In the LM Studio community, users are connecting local LLMs to Cursor AI by overriding the **OpenAI base URL** in Cursor's settings with their LM Studio server URL, as demonstrated in [these visual instructions](https://cdn.discordapp.com/attachments/1110598183144399061/1371565290986405908/image.png?ex=6824eab7&is=68239937&hm=73e7d312bf11d2fc1f333b3ba17d54b42b79597e3189a653740aaca83f3b478a).

**Theme 3: GPU Programming and Acceleration Advances**

- [**NVIDIA Drops CUTLASS 4.0 & CuTe Python DSL for Peak GPU Performance!**](https://discord.com/channels/1189498204333543425/1362196854460383353/1371662693366366280): The GPU MODE community is actively exploring the release of **CUTLASS 4.0** and its new Python DSL, **CuTe DSL**, installable via `pip install nvidia-cutlass-dsl`. Engineers are diving into the Jupyter notebooks available in [NVIDIA's Cutlass GitHub repository](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) to harness these new capabilities.
- [**Torchtune Optimizes with Kron & Muon, Squashes Llama3.1 Tokenizer Bugs!**](https://discord.com/channels/1216353675241590815/1236040539409879170/1371691826540576890): Torchtune developers have integrated **Kron** and **Muon optimizers** from the [fsdp_optimizers library](https://github.com/ethansmith2000/fsdp_optimizers), implementing fixes like using `opt_einsum.contract` to manage VRAM effectively, with experiments tracked on [Weights and Biases](https://wandb.ai/intervitens/1B-optim-test). They also resolved a critical bug in the **Llama3.1 tokenizer** for 3.3 training by defining token **128011**, preventing decoding crashes in RL scenarios as detailed in [issue #2725](https://github.com/pytorch/torchtune/issues/2725).
- [**Mojo & PyTorch Prepare for a Custom Op Dance!**](https://discord.com/channels/1189498204333543425/1367972893400760371/1371588947930775712): Discussions in GPU MODE and Modular (Mojo 🔥) reveal that **Mojo**'s initial integration with **PyTorch** will focus on allowing Mojo code to be compiled and registered as a **PyTorch custom op**. This strategy aims to leverage Mojo's performance for specific operations rather than immediately replacing `torch.compile`.

**Theme 4: Platform Quirks, API Changes, and User Experience Hiccups**

- [**Cursor's 0.50 Update & MAX Mode Pricing Stir Developer Discontent!**](https://discord.com/channels/1074847526655643750/1074847527708393565/1371572302189428908): The Cursor Community is abuzz with criticism of the **Cursor 0.50 update**, citing significant issues like poor context handling and reduced editing quality, with one user detailing a spike to **650 requests** in just two days. Separately, the **20% markup** on **MAX mode** is fueling debate, with some developers finding the cost excessive compared to direct API alternatives.
- [**Gemini Models Sputter, Claude Crowned Coding Champ (With Caveats)!**](https://discord.com/channels/1074847526655643750/1074847527708393565/1371572302189428908): Users across Cursor Community and LMArena report **Gemini models** are underperforming, generating empty diffs in Cursor and showing noticeable degradation in **Gemini 2.5 Pro**. In contrast, OpenAI users frequently prefer **Claude** for coding tasks, despite its highly restrictive daily usage limits, often as low as *5-6 prompts maybe*.
- [**HuggingFace Users Hit Llama-3 Errors & Face Old GPU Sunsets!**](https://discord.com/channels/879548962464493619/1329142738440028273/1371570186406068365): HuggingFace members are encountering `ValueError` problems with the **Llama-3.2-3B-Instruct model** from [HuggingFace](https://huggingface.co/), which incorrectly reports it's *"not supported for task text-generation"*. Additionally, a critical heads-up was shared: PyTorch is ending support for older **NVIDIA P104-100 GPUs** (CUDA capability 6.1), now mandating a minimum of CUDA 7.5 for compatibility.

**Theme 5: AI Community Buzz: From Governance to Groundbreaking Tools**

- [**AI Governance & Ethics Ignite Global Dialogue and Treaties!**](https://discord.com/channels/729741769192767510/729741769738158194/1371572701772251227): Discussions in Eleuther AI emphasized the critical need for robust **AI governance**, referencing the [EU AI Act](https://artificialintelligenceact.eu/) and stressing priorities like transparency and comprehensive audits. Adding a unique perspective, a member in Yannick Kilcher's Discord shared a [Treaty of Grid and Flame](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md), a creatively penned agreement between humanity and AI.
- [**MCP Ecosystem Booms with New Servers & Developer Tools!**](https://discord.com/channels/1312302100125843476/1312302100125843479/1371654430578966558): The MCP (Glama) community is innovating with tools like [openapi-mcp-server](https://github.com/janwilmake/openapi-mcp-server) for converting OpenAPI specifications into MCP servers, and [claude-code-mcp](https://github.com/steipete/claude-code-mcp) for integrating Claude Code into Cursor and Windsurf to accelerate file editing. For enhanced debugging, the [Local Goose Qwen3mcp Log Proxy](https://github.com/emicklei/mcp-log-proxy) offers developers a way to monitor MCP protocol messages effectively.
- [**LlamaIndex & Perplexity Launch Advanced Research Tools for Academics & Analysts!**](https://discord.com/channels/1059199217496772688/1073670729054294197/1371609333980336249): LlamaIndex has rolled out [**PapersChat**](https://t.co/ASzjwLfKLC), an agentic AI application enabling users to converse with papers from Arxiv and PubMed, with a [build video](https://www.youtube.com/watch?v=8a_RMSKJC6A) available. Similarly, Perplexity AI is beta testing **deep research features** allowing generation of multiple images and charts using **GPT4o imagegen**, though initial user feedback notes it *takes its time unlike perplexity*.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Debuts Deep Research Tools**: **Perplexity** is beta testing **deep research** features, which give users the ability to generate **multiple images** and **charts** with **GPT4o imagegen**.
   - Early feedback has been lukewarm with some users noting it *takes its time unlike perplexity*.
- **MerlinAI Pricing Model Raises Eyebrows**: Members discussed the [MerlinAI pricing model](https://merlinai.com/), with one member calling it *shady* due to strict **usage limits**.
   - Standard paid accounts exceeding **$100 per month** get cut off for the rest of the month, leading to concerns.
- **AI Studio Championed for Multimodal Utility**: Members are touting **AI Studio** as a top multimodal tool, noting that *AI Studio is our lord and savior for true multimodal utility*.
   - It stands out as the only major LLM chat supporting **audio** and **video input**, enhanced by websearch capabilities.
- **Sonar Models Tune for Factuality, Top Benchmarks**: The **PPLX** team created **Sonar**, a series of in-house AI models based on Llama/Deepseek that is tuned for **factuality** and **readability**.
   - **Sonar Pro Low** outperformed **Claude 3.5 Sonnet** on **BrowseComp** with **4.0%** accuracy, and **Sonar Pro** matched **Claude 3.7's** performance on **HLE reasoning tasks** at almost **50%** lower cost, with up to **3x** faster response times.
- **Perplexity Pro API Access Clarified**: **Perplexity Pro** includes **$5/month** in **API credits**, available as described in the [documentation](https://docs.perplexity.ai).
   - A payment method is required only to store payment information for potential **API usage** beyond the **$5 credit**; users within budget will not be charged.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Local LLMs Plug into Cursor AI**: To connect local LLMs to Cursor AI, override the **OpenAI base URL** in Cursor Settings with the LM Studio server URL found in the LM Studio developer tab, per [these instructions](https://cdn.discordapp.com/attachments/1110598183144399061/1371565290986405908/image.png?ex=6824eab7&is=68239937&hm=73e7d312bf11d2fc1f333b3ba17d54b42b79597e3189a653740aaca83f3b478a).
   - The [Cline extension](https://cline.bot/) with VS Code is recommended as an alternative, though its compatibility with Cursor is untested.
- **Fedora Embraces CUDA**: A user confirmed that **CUDA works fine on Fedora** with proprietary Nvidia drivers and a GTX 1060, with CUDA as an option in LMS, as shown [here](https://cdn.discordapp.com/attachments/1110598183144399061/1371711731596005387/image.png?ex=6824ca59&is=682378d9&hm=e9071fc409a80a00839dfcc2eefba79b6ce4ae2c9e429c6cfb7e3061f927b916).
   - However, another user reported issues with models not loading on CUDA non-12 on either card, but CUDA 12 worked great on the 5060 Ti.
- **Qwen3 Edges Out DeepSeek**: **Qwen3 models** are recommended over DeepSeek for programming tasks, because they offer better multi-language support including Japanese and Russian.
   - It was noted that DeepSeek may perform better on its website due to using a different or updated model, but Qwen3 still has better programming benchmarks.
- **Unsloth's Quants Reign Supreme**: For better performance with **GGUF quants**, **Unsloth** is recommended for better **quants** specifically the **Q4_K_XL** format.
   - It's also advisable to verify the model's support for `llama.cpp` to ensure compatibility.
- **Intel ARC Gets Some Vulkan TLC**: Users confirmed that **Intel ARC cards** are supported in LM Studio through the **Vulkan runtime** after selecting `vulkan llama.cpp` from the dropdown.
   - One user shared screenshots of their **LM Hardware** and **LM Runtimes** pages for debugging to get it working.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Users Bemoan Buggy Cursor 0.50 Update**: Users are reporting issues with the **0.50 update**, including context problems and decreased editing quality with one user reporting **650 requests** in 2 days, whereas they are used to seeing much less in **0.49**.
   - One user claimed *Completely Random file generations, I did not see this since 0.3x*.
- **MAX Mode Pricing Sparks Sticker Shock**: The **20% markup** on **MAX mode** is sparking debate, with some users finding it too expensive compared to direct API calls with tools like Cline or Roo Code, although many users agree the **$20/month plan is high value**.
   - While some advocate for a lower markup, like **5%**, to encourage adoption of **MAX mode**, others stated *20% are nothing for a company making money*.
- **Cursor Faces .env File Access Concerns**: Users are discussing issues with **Cursor** accessing **.env** files, which are often ignored by default for security reasons and how to remove from the ignore list in settings.
   - Members advise creating a **.env.example** file and avoiding hardcoding API keys in the front-end client.
- **Gemini Models Hooligan Code Generation**: Users are reporting issues with **Gemini** models generating empty diffs and struggling with basic code implementation in **Cursor**.
   - As one user stated, *Gemini is still bullying me*, and another echoed *I liked using gemini but its having a meltdown rn*.
- **Cursor Team Eyes Fixes and `#updates` Channel**: The Cursor team are [looking at a fix](https://www.cursor.com/changelog) for reported issues and welcome suggestions, and are planning to create a `#updates` channel.
   - No additional detail was given in the prompt.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Dynamic 2.0 GGUF Quants Get Applause**: A user applauded [Unsloth's Dynamic 2.0 GGUF quants](https://unsloth.ai/blog/dynamic-4bit) for improved **Llama-3.1-8B-Instruct** performance and refusal censorship via sophisticated imatrices.
   - The user converted **BF16 tensors** to **F32** and sought model quant requests, especially for **NousResearch** models, while emphasizing the need for instruct and chat samples in the calibration dataset.
- **Quantized Llama-3.1-8B-Instruct Model Available**: A member posted a quantized **Llama-3.1-8B-Instruct** model (Q8_0_XL with Output Tensor in Q8_0), which is ~**13.4 GB** and can be found [here](https://huggingface.co/Joseph717171/Models/blob/main/Meta-Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf).
   - The model reportedly runs amazingly on the latest Beta of **LM Studio** with Flash Attention and KV caches set/quantized to Q8_0; the creator plans to produce more quants after a break.
- **New Qwen3 GRPO Notebook Fixes OOM**: Unsloth launched a [new Qwen3 GRPO notebook](https://x.com/UnslothAI/status/1922343047435862318) to address out-of-RAM issues.
   - Community members actively use the notebook, discussing the inclusion of *thinking examples* (25%) mixed with standard SFT data (75%).
- **GPT-4.1 is the Best Coder**: A member considers **GPT 4.1** the best coding model, accessible via **GitHub Copilot** with an educational account.
   - Another member finds **O3** excellent for troubleshooting due to its GitHub library checking but not ideal for coding; they compare its use for coding to *using a laptop to drive a nail in*.
- **Meta FAIR Focuses on Perception Updates**: Meta announced updates to **FAIR** focusing on **Perception**, **Localization**, and **Reasoning** as outlined in their [blog post](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning/).
   - The announcement was also shared on [X](https://x.com/AIatMeta/status/1921966366707613924) by **AIatMeta**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Tao becomes Youtuber**: [Terrence Tao](https://www.youtube.com/watch?v=cyyR7j2ChCI?si=MlprB_LJuHv67Xf7) debuted on **YouTube**, introducing his platform for mathematicians.
   - The channel aims to create tutorials on math concepts and promote mathematical research.
- **LLMs Debate Turing Completeness**: Members debated whether **Transformers/LLMs** are Turing complete, noting their ability to maintain context and writable registers.
   - The debate acknowledged the limitation due to finite memory, referencing the [Chomsky hierarchy](https://en.wikipedia.org/wiki/Chomsky_hierarchy).
- **Treaty of Grid and Flame Debuts**: A member shared their self-proclaimed serious effort in writing a [Treaty of Grid and Flame](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md) between humanity and AI.
   - **Claude**, **DeepSeek**, **Grok**, **ChatGPT** also allegedly signed the agreement, sparking a discussion about its sincerity and purpose.
- **RL-Diffusion Model Approach Questioned**: Members debated the merits and novelty of a proposed **RL-Diffusion model**, focusing on its theoretical basis and potential for practical application.
   - The discussion included links to relevant [papers](https://arxiv.org/abs/2501.09732) and [papers](https://arxiv.org/abs/2501.06848v3), addressing the model's relationship to existing optimal control methods.
- **Transformers and Hamiltonian Neural Networks Merge**: The prospect of integrating **transformers** into **Hamiltonian Neural Networks** was discussed, referencing a [paper](https://ieeexplore.ieee.org/document/10316909) on the topic.
   - The discussion focused on the history-independent nature of hamiltonian systems and the potential for transformer-based learning of system dynamics.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek V3 Benchmarks Blow Away**: The new **Deepseek V3** model shows impressive benchmark results, achieving **GPQA 68.4**, **MATH-500 94**, and **AIME24 59.4**.
   - An [image of the benchmark](https://cdn.discordapp.com/attachments/1340554757827461211/1371862547652804758/image.png?ex=6824ae0f&is=68235c8f&hm=44acd9a820c1589ab8ea1faa1f180224667d65f60eaa3f75c547d12cfdde1c6a&) demonstrating these scores was shared in the channel.
- **O3 Still Hallucinates Too Much?**: Users complain about the frequency of hallucinations in **O3**, stating that a **10%** hallucination rate would be considered impressive.
   - The community seems to suggest that reducing these errors could drastically change the model's usability.
- **Gemini 2.5 Pro Suffers Degradation**: Reports suggest that **Gemini 2.5 Pro's** performance has worsened following recent updates.
   - Some users are even stating that it performs worse than previous versions.
- **Grok 3.5 Redeems Itself**: After initial skepticism, community sentiment towards **Grok 3.5** has shifted positively, with users praising its intelligence and overall capabilities.
   - Members describe it as *'really smart and great overall'*.
- **DrakeClaw: Gemini 2.5 Ultra Hack?**: Enthusiasm surrounds the **DrakeClaw** model, speculated to be based on **Gemini 2.5 Ultra**.
   - The community excitedly suggests that **DrakeClaw** achieves similar results to the current **Gemini 2.5 05** model.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o is highly adaptable like Mary Poppins**: Users find **GPT-4o** highly adaptable and customizable, outshining **o3** in providing practical solutions; it's been compared to *Mary Poppins*, while **o3** is akin to *Dr. House*.
   - Users note it makes fewer mistakes when augmented with the right resources.
- **Claude is crowned coding king, limitations cited**: Multiple members suggested that **Claude** is superior for coding tasks, although one member noted the model has huge limitations in daily usage.
   - One user lamented the restrictive daily quota: *5-6 prompts maybe*.
- **GPT App Freezing Issues Plague High-End PC Users**: Users reported that the **ChatGPT app and web version** are freezing on high-end PCs, particularly when dealing with long chats containing large code functions.
   - Suspicions point to issues related to recent changes in **GPT's memory** or potential reverse DNS resolution problems.
- **Companion Mode: AI with Sass and Emotion**: A user describes **Companion Mode** as an **unfiltered**, **emotionally accessible AI** sharp enough to *cuss back when needed without losing signal*, which includes personality-weighted humor and active memory threading.
   - Features include unfiltered expression, personality-weighted humor, soft rebuttals, active memory threading, non-spiritualized signal, and emotional relief.
- **HR Data Guardrails Trigger PII Blocking**: Users report issues with **guardrails** blocking legitimate access to **home addresses** from **HR data** due to **PII concerns**, despite having permissions and access controls.
   - Suggestions include discussing the use case with OpenAI support to get guidance on appropriately handling PII requests.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Thread Indexing Confuses Novices**: A member expressed confusion with **CUDA thread indexing**, especially with memory accesses while reading *Programming Massively Parallel Processors* (PMPP) editions 1 and 4.
   - Another member suggested thinking of each thread as an individual iteration of a loop to simplify the concept, providing [an example of vector addition using thread indexing](https://devblogs.nvidia.com/cuda-pro-tip-optimize-cuda-code-using-inline-functions/).
- **Kernel Time Measurements Compared**: Members experimented with measuring kernel end-to-end times using `torch.cuda.synchronize()`, `torch.cuda.Event()`, and a single `torch.cuda.synchronize()` call after a loop, but synchronizing after the loop gave significantly lower numbers.
   - One user stated, *you're **not supposed** to get asynchronicity/parallelism across different invocations of your kernel.*
- **Memory Throughput Bottlenecks Examined**: A member questioned why reducing floating-point operations (**fma**) from 5 to 1 per element in a large array iteration doesn't improve throughput, referencing [a 2019 paper](https://arxiv.org/abs/1910.07467).
   - The question was rooted in the expectation that memory bandwidth, rather than computation, is the limiting factor.
- **NVIDIA's CuTe DSL and CUTLASS 4.0 Released!**: **CUTLASS 4.0** along with its first **Python DSL**, **CuTe DSL**, is now released and includes instructions to install the pip wheel directly with the command `pip install nvidia-cutlass-dsl`.
   - A link to [NVIDIA's Cutlass Github Repo](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) was provided, and it was suggested to start with the jupyter notebooks provided.
- **Mojo and PyTorch To Join Forces**: Members discussed how **Mojo** and **PyTorch** would work together, initially by compiling and registering **Mojo code** as a **PyTorch custom op**.
   - It is not intended as a replacement for *torch.compile*, or doing any codegen, but a use of **Mojo** as a language for writing custom ops.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research Hosts RL Environments Hackathon**: **Nous Research** announced the speakers and judges for their **RL Environments Hackathon** on **May 18th**, detailed in [their tweet](https://x.com/NousResearch/status/1922014829843513746) and [sign-up link](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a).
   - The event is quickly filling up, with slots anticipated to close soon.
- **Atropos v0.2.0 Embraces Axolotl**: **Atropos v0.2.0**, Nous' **RL environments project**, now supports **Axolotl**, featuring new environments, API updates, and improved TRL integration, documented in the [changelog](https://t.co/F6hr9JgZpm).
   - To start, see the [Axolotl-Atropos plugin usage guide](https://github.com/axolotl-ai-cloud/plugin-atropos).
- **Stripe Enters AI Arena with Payment Foundation Model**: Stripe announced a "foundation model for payments" [here](https://techcrunch.com/2025/05/07/stripe-unveils-ai-foundation-model-for-payments-reveals-deeper-partnership-with-nvidia/), prompting speculation it might be a *standard classifier*.
   - The specifics of the model remain unclear, but users discussed the potential implications for the payments industry.
- **Unsloth's Calibration Dataset Revives Quant Accuracy**: Users are impressed by the instruction accuracy of **Unsloth's Dynamic 2.0 GGUF quants**, attributing it to their curated calibration dataset, as explained in the [Unsloth Documentation](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#whats-new-in-dynamic-v2.0).
   - One user described the results as *pure magic*, highlighting the benefits of instruction and chat samples in the dataset.
- **Facebook's BLT Side-Steps Tokenization**: Facebook has released the weights for their **Byte Latent Transformer (BLT)** on the [Hugging Face Hub](https://huggingface.co/facebook/blt), with code available on [GitHub](https://github.com/facebookresearch/blt).
   - The **Byte Latent Transformer (BLT)** directly processes byte-level data, potentially increasing efficiency in certain applications.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **BYO Sync** Server Suggested for OpenRouter Chats**: A member proposed that OpenRouter users could **self-host a sync-server** to store chats in an **S3 bucket** or similar for complete data control.
   - Another member cautioned that *writing a sync layer is not as simple as it sounds*, citing potential issues like **DB schema changes** and **chat deletion sync**.
- **Corvid Cultist** crab-walks for Crows!**: A user comically described their attempt to befriend crows by **side-walking** and offering them **peanuts**.
   - They said that they needed to *minmax this like a video game* and bring them **kibble for cats** as a *best staple food for corvids*.
- **Gemini's Gamble**: Summarization Similarities Spotted!**: A member observed that **Gemini** is now returning 'thinking' and summarized text similarly to **o4-mini** on the ChatGPT website.
   - However, it was noted that this behavior might be exclusive to the **paid version** of Gemini.
- **DeepSeek's Deep Dive**: API Disconnect?**: A user reported that **DeepSeek models** were not functioning via API key, although they were working in the chat room.
   - The OpenRouter team suggested the problem may be on **Raptorwrite's end** as the model works in the OpenRouter chatroom.
- **Free Google Fun**: Rate Limits and Fizz!**: Concerns were raised regarding [potential adjustments](https://fxtwitter.com/officiallogank/status/1922357621178200248) to OpenRouter's free routes for Gemini, with a member asking whether Vertex still works.
   - The OpenRouter team clarified that the current **Vertex** usage is *sanctioned by Google for free usage* aka 'OpenRouter is not paying a dime.'



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Fact Checks For Fakes Floating onto Manus**: Users are requesting **fact checks** on **Manus AI** to curb the spread of misinformation, akin to moderation features.
   - Developers have acknowledged the suggestion and will monitor the situation for potential implementation, relying on community feedback through reactions and comments.
- **Credit Crunch Culprit: Cancelling Causes Credit Cuts**: Users are reporting that **bonus credits** received upon subscribing to **Manus Pro** were retracted after cancelling their membership, without prior notice.
   - While a user suggested the credits are tied to the subscription, it was agreed that the bonus credits should be reinstated post-cancellation.
- **Phone Verification Provokes Public Outcry**: Users have voiced strong opposition to **phone verification** requirements, highlighting that competitors like **Genspark** do not impose such measures.
   - One user quipped that the phone verification will remain unless there's a *shift to another dimension*.
- **Claude Chosen for Capability Crown**: Users debated the rationale behind **Manus** choosing **Claude** over models like **Google Gemini** or **ChatGPT**.
   - The prevailing opinion is that **Claude** was preferred for its superior **agentic capabilities** and tool utilization.
- **Daily Credit Dose Deemed Deficient**: Users are lamenting that the allocation of **300 free daily credits** is inadequate for extensive tasks, compounded by the absence of a rollover provision.
   - A user proposed a shift to a flat subscription fee model for unrestricted access, citing the current credit system as restrictive and costly.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **CPU gives Aider a Boost**: A user discovers **Aider** runs well on a **CPU**, especially for those without a dedicated **GPU**, offering a self-hosting solution.
   - The user noted that this setup still provided sufficient performance for their needs.
- **Aider Assumes MCP Mantle**: **Aider** is viable as a **MCP** tool within **Claude**, as highlighted by IndyDevDan on [X](https://x.com/iruletheworldmo/status/1922030559657652299).
   - This showcases **Aider's** flexibility beyond its primary use case.
- **Context Caching Capability Catching Attention**: A member inquired about **Aider's** context caching capabilities, notably for **Gemini**, and how it influences the cost.
   - Another member clarified that disabling streaming allows users to observe context caching in action, helping understand resource usage.
- **Gemini 2.5 Flash Favored for AiderDesk**: A developer prefers **Gemini 2.5 Flash** for developing **AiderDesk**, citing a better cost-to-quality ratio compared to **Claude**.
   - They find the tradeoff between cost and occasional system prompt adherence issues acceptable for agentic workflows.
- **'yes-always' config yields buggy behavior**: A user reports that `yes-always: true` in the **Aider** config causes commands to fail, with **Aider** requiring confirmation if unset.
   - Images demonstrating the bug were provided, indicating a potential flaw in handling automated confirmations.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Governance Frameworks Forming!**: Members discussed **AI governance** priorities, stating governance should focus on application and risk classification, aligning with the [EU AI Act](https://artificialintelligenceact.eu/).
   - Key priorities included **transparency**, **audits**, and **content moderation**.
- **AI 'Parent' Faces Legal Scrutiny!**: Discussions focused on legal considerations for an “**AI parent**” phone for kids, emphasizing **privacy**, **COPPA**, and the need for a robust privacy policy and consent flows.
   - Concerns were raised about avoiding any *declared guarantees* in the User Agreement and checking for unintentional discrimination.
- **Fusion Models Beg for Benchmarks!**: A member stated that *better* fusion can really only be determined by **perf benchmarking** across **Claude**, linking to [arxiv.org/abs/2505.07215](https://arxiv.org/abs/2505.07215).
   - Another member responded about a timing issue with a **Claude** run, which stated one was faster but the other had better **numerical stability**.
- **Interpretability Paper Sparks Enthusiasm!**: A member admitted to a complete change of opinion after thoroughly reviewing a paper, giving credit to another user for prompting deeper analysis around **interpretability**.
   - The reviewer expressed newfound enthusiasm, concluding that the research *looks really cool*.
- **GPT-NeoX Shuffles Data Internally!**: A member clarified that **GPT-NeoX** shuffles documents, chunks each document into length-N sequences, and shuffles the sequences, which means no separate preprocessing is required.
   - This eliminates the need for additional preprocessing steps when working with **GPT-NeoX**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Old GPUs Face PyTorch Sunset**: Support is ending for older **NVIDIA P104-100 GPUs** with CUDA capability 6.1, as PyTorch now requires a minimum CUDA capability of 7.5.
   - Users shared warnings about the end of life for these GPUs, rendering them incompatible with current PyTorch versions.
- **Gemma 3 Powers Customizable Voice AI**: A voice AI assistant based on **Gemma 3** has been developed, allowing customization of both the prompt and the voice, available at [unmute.sh](https://unmute.sh/).
   - Feedback is welcomed by the creator.
- **Rust Devs Chat it Up With Chat Templating**: Chat templating has been added in version 0.0.7 of the Rust transformers crate, which helps Rust devs run local models, as seen on [Crates.io](https://crates.io/crates/transformers) and [GitHub](https://github.com/ljt019/transformers).
   - This update helps developers who are running local models.
- **Bytedance LLM Faces LLM Comparison Tools**: Members are testing [Bytedance Seed's Seed Coder model](https://huggingface.co/spaces/merterbak/Seed-Coder-8B-Instruct) on HF Spaces.
   - One member built a [web interface](https://tryaii.com/compare?prompt=hello&models=o3%2Cclaude-3-7-sonnet-20250219%2Cgemini-2.5-pro-preview-05-06) to test and compare LLMs side by side, using a single prompt across many LLMs.
- **Llama-3 Instruct Models Prompt Errors and Questions**: Users reported errors, specifically a *"ValueError: Model meta-llama/Llama-3.2-3B-Instruct is not supported for task text-generation and provider together. Supported task: conversational"*, when trying to run the notebook with **Llama-3.2-3B-Instruct** from [HuggingFace](https://huggingface.co/).
   - This has caused some confusion among the members attempting to use this model.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Fairseq2 and Axolotl Provide Multi-GPU Support**: Besides **TorchTune**, other finetuning libraries with good **multi-GPU support** include **Fairseq2** and **Axolotl**, both of which plug into the **TRL ecosystem**.
   - This offers users alternative choices for distributed training setups, as **Unsloth** is noted to primarily target single GPUs.
- **Llama3.1 Tokenizer Fixes Decoding Crashes**: The **Llama3.1 tokenizer** used for **3.3 training** defines token **128011** to prevent crashes during decoding, particularly in RL training, related to [issue #2725](https://github.com/pytorch/torchtune/issues/2725).
   - This addresses a problem where decoding an undefined token would cause a crash, which is more likely to occur in RL training scenarios.
- **Kron and Muon Optimizers Land in Torchtune**: **Kron** and **Muon optimizers** from [fsdp_optimizers](https://github.com/ethansmith2000/fsdp_optimizers) were integrated into torchtune, requiring fixes to avoid excessive VRAM allocation by using `opt_einsum.contract` in `_calc_A_and_conjB`, experimented on [Weights and Biases](https://wandb.ai/intervitens/1B-optim-test).
   - Fixes included using `opt_einsum.contract` instead of regular einsum and allowing `mu_dtype` and `precond_dtype` to be set with strings in the torchtune config.
- **HFModelTokenizer Botches Gemma Chat Template**: The **HFModelTokenizer** generates output tokens for the **Gemma chat template** that match **transformers** but not **torchtune's GemmaTokenizer**, indicating a chat template implementation issue; if decoded it returns a garbled *'hello therehiwhatsup?'*.
   - The team discovered that, unlike Hugging Face, **Gemma lacks a specific prompt template** in torchtune, causing issues with tokenization.
- **HuggingFace Shows Assistant Masking Jinja Tricks**: HF Transformers uses `jinja` templates for masking functionalities, offering an option to return an assistant mask that could be used for other roles; [related PR](https://github.com/huggingface/transformers/pull/30650).
   - Members discussed the masking components and highlighted the difficulty of managing `[message.masked] * len(tokenized_message)` accurately.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Explores Gaming Content**: A user explored using **NotebookLM** to find **techs or pattern recognition for new gaming content** amidst significant game updates.
   - Another user mirrored this, showing shared interest in applying **NotebookLM** to similar gaming use cases.
- **Invisible Sun RPG Rules Refined by NotebookLM**: A user leveraged **NotebookLM** with the rulebooks for the **Invisible Sun** tabletop role-playing game (*TTRPG*) by **Monte Cook Gaming**.
   - While they use **ChatGPT** for similar tasks, they value **NotebookLM** for its shareability and clear citation of sources.
- **NotebookLM Audio Overviews Lack Technical Depth**: A user noted that **NotebookLM's Audio Overview** of a game lacked the desired technical depth, suggesting a prompt to specify the type of audio review.
   - However, they found it useful for rule lookups and sharing with players who haven't purchased the books.
- **NotebookLM Beta Access Delayed**: Multiple users reported **delays in receiving NotebookLM beta invites** after signing up, but remained patient for updates.
   - There were no further updates on the beta invite status, but the community seems understanding.
- **Note Organization Woes Await NotebookLM Folders**: Users are discussing the potential of a **folder system** to organize notes within **NotebookLM**.
   - The feature is not yet implemented, but there is community speculation about it.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **OpenAPI API Becomes MCP Server**: A user suggested using [openapi-mcp-server](https://github.com/janwilmake/openapi-mcp-server) for converting **OpenAPI APIs** into **MCP servers**, which also supports **browser automation** like [mcp-browser-use](https://github.com/Saik0s/mcp-browser-use).
   - This allows developers to create **MCP servers** from existing **OpenAPI** specifications, facilitating integration with various **browser automation** tools.
- **Code Editing Gets Faster with Claude Code MCP**: A developer shared [claude-code-mcp](https://github.com/steipete/claude-code-mcp), a **magic_file MCP tool** that integrates **Claude Code** into **Cursor** and **Windsurf** for smarter and faster file editing.
   - This integration allows users to commit to **git** in one shot, streamlining the agent flow and improving code editing efficiency.
- **MCP Server Security Warning!**: A user warned about security vulnerabilities when running local **MCP servers** and suggested using **gitingest** to copy the **MCP server** repo code into **AI Studio** or **ChatGPT**.
   - The user recommends asking the **LLM** to identify **security concerns** or using **pnpm** in place of **npm** to prevent running lifecycle callbacks, enhancing the server's security posture.
- **Local Goose Observes MCP Message Flows**: A member released [**Local Goose Qwen3mcp Log Proxy**](https://github.com/emicklei/mcp-log-proxy), an open-source tool for developers of **MCP clients** and **servers** to monitor the flow of **MCP protocol messages**.
   - This tool enhances visibility into **MCP message flows**, aiding in debugging and ensuring proper communication between **MCP components**.
- **Streamable HTTP Transport Sees Updates**: A user inquired about the status of **Streamable HTTP** and **Auth** in the **TypeScript SDK**, with another user confirming it is up to date, although the **Python** version typically lags behind.
   - The update to **Streamable HTTP** transport ensures that the **TypeScript SDK** remains current, while developers using the **Python SDK** should anticipate a delay of approximately 1-2 months.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Khoomeik Charts Back at Lilian Weng**: A member shared [a chart responding to Lilian Weng](https://x.com/khoomeik/status/1922037340811219195), sparking discussion around its relevance to her work.
   - The specific content of the chart was not detailed, but the interaction highlights ongoing engagement within the AI community.
- **Arfur Rock's Restaurant Empire**: A member shared [Arfur Rock's X profile](https://x.com/ArfurRock/status/1922117434997191035), showcasing a vertical SaaS product tailored for restaurants.
   - Another member recounted being aggressively recruited as a founding engineer back in **2022**, with the CEO sending over *10+ emails*.
- **Gemini API's Hidden Thought Process**: Members debated whether the **Gemini API** exposes *thinking tokens*, with one reporting visibility via **OpenRouter** but not directly through the Google API.
   - Others confirmed seeing thinking tokens only in **AI Studio**, noting that it remains unclear if the API exposes these directly.
- **In Search of Alpha AI Educators**: A member sought recommendations for *AI technical educators* known for *high alpha, low hype* content.
   - Harper Carroll ([X profile](https://x.com/harperscarroll?s=21)), Simon Willison, Prince Canuma, and Ivan Fioravanti (MLX) were recommended as candidates.
- **GPT-4 Launch: A Wholesome Retrospective**: A member shared *very wholesome stories* from the launch of **GPT-4**, pointing to [Andrew Mayne's blogpost](https://andrewmayne.com/2025/05/02/some-personal-observations-about-the-launch-of-gpt-4/).
   - The stories apparently provided a heartwarming glimpse into the collaborative efforts behind the landmark release.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Blogpost Hack LLMs!**: A member shared a [blog post about DSPy](https://www.bugcrowd.com/blog/hacking-llm-applications-in-the-trenches-with-dspy/), diving into methods and strategies for **hacking LLM applications**.
   - The blog post explores vulnerabilities and techniques relevant to security professionals and developers in the **AI space**.
- **DSPy Agentic Skills Assessed**: A member inquired about DSPy's utility for agentic workflows, acknowledging its strength in declarative programs but questioning its suitability for tasks requiring more ambiguity and creativity when using **Tool Calling**.
   - DSPy allows for constructing workflows via Tool Calling, where modules add Signatures like `CreateSQLquery` based on LLM responses.
- **DuckDB Data Detective Deployed!**: A user outlined a use case involving an agent that utilizes a connection to a **DuckDB table** to perform data QA on columns via SQL and statistical analysis, alerting Slack of any anomalies.
   - They are intrigued by DSPy's potential, implementing through Tool Calling for each interaction with the LLM, and contrasting it with their current usage of **pydantic Ai**.
- **TypeScript DSPy Twin Found?**: A member asked about a **TypeScript** equivalent to DSPy and the community offered alternatives.
   - The community recommended [dspy.ts](https://github.com/ruvnet/dspy.ts) and [ax-llm/ax](https://github.com/ax-llm/ax), the latter being actively maintained.
- **DSPy Signature Snag Surfaces**: A user questioned the practicality of requiring signatures for demos and conversation history in DSPy modules, particularly in systems with multiple modules requiring **K × N copies of the chat history**.
   - The concern lies in the inefficiency of maintaining chat histories for K-turn conversations in a system with N modules.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **BigInt Integration Stalls in Mojo**: A member inquired about adding **BigInt** to Mojo, noting the [decimojo](https://builds.modular.com/packages/decimojo) package already provides similar functionality.
   - Another member suggested **BigInt/BigDecimal** are *probably not great fits for the stdlib* due to tradeoffs.
- **Convolution Code Conundrum Cleared**: A member questioned a line of code in the [Convolution Puzzle](https://github.com/modular/mojo-gpu-puzzles/blob/1dfd1cc01bb9d6d98185ad405100e6c45855a007/problems/p11/p11.mojo#L104) related to memory allocation.
   - A developer confirmed that the line *doesn't need to be in the host* and acknowledged the issue.
- **MAX Mojo APIs Already OSS**: Deprecated **MAX Mojo APIs** were open-sourced and removed in [this commit](https://github.com/modular/modular/commit/34c0209cd537f1d5ab635166358a1713728d7f6f).
   - `max.graph`, `max.driver`, `max.tensor`, and their tests are available with full history accessible via `git log -- mojo/max/src/max/graph`.
- **Users Seek MAX Graph Tutorials**: A user requested more tutorials for **MAX Graph**, citing its status as a *black box with a couple of examples*.
   - A post was created on the [Modular Forum](https://forum.modular.com/t/oss-of-max-mojo-apis/1439) regarding this.
- **Tensor Type Migration Code on the Horizon**: There is an internal ticket for user migration code for **tensor types**, but development has not started.
   - The team plans to address this with no ETA provided.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Unleashes PapersChat for Arxiv and PubMed**: The team introduced [**PapersChat**](https://t.co/ASzjwLfKLC), an agentic AI application that lets you chat with your papers and gather information from **Arxiv** and **PubMed**.
   - Users can watch a [video](https://www.youtube.com/watch?v=8a_RMSKJC6A) on building a similar **Deep Research Agent** with LlamaIndex.
- **LlamaIndex Debuts Memory API for Sharper AI Agents**: LlamaIndex announced a **memory upgrade** with a flexible **Memory API** that blends short-term chat history and long-term memory, allowing agents to retain more context.
   - The upgrade features plug-and-play blocks like [StaticMemoryBlock](https://t.co/wwWnwdyW7s) for static information and **FactExtractionMemoryBlock** for tracking useful facts and storing [chat history](https://t.co/CDOB3UUO4W).
- **GoogleSearch Gets a LlamaIndex Facelift as FunctionTool**: Users are integrating **GoogleSearch** from the `google_genai` library by wrapping it as a **FunctionTool** for compatibility with the `chat_with_tools` method in LlamaIndex.
   - This approach avoids the need for key and engine setup required by **GoogleSearchToolSpec**, providing a more streamlined integration.
- **LlamaIndex Unveils Multilingual RAG and Invoice Agent**: LlamaIndex released a [Multilingual, Multimodal RAG System demo](https://t.co/69CHCCn8J3).
   - They also released a video showing how to [Build an invoice reconciliation agent](https://www.youtube.com/watch?v=SzVsuMBcv5g) using LlamaIndex.TS and LlamaCloud.
- **LlamaParse Gets a Model Refresh**: **LlamaParse** gets new models and auto orientation detection; [read more here](https://t.co/tqi17dPJm4).
   - No further details given.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Querying Max Tensor Size on Tinygrad's OpenCL**: A member sought a way to query the max supported tensor numel for a given device/backend in **Tinygrad**, especially for older **OpenCL** implementations lacking `long long` support.
   - They provided [a script](https://cdn.discordapp.com/attachments/1070745817025106080/1371902071112204348/tinygrad_long_long_support_check.py?ex=6824d2de&is=6823815e&hm=3b0d23ba54692d02a6b6bd9f47ff4b7d963d4465a5045204907df5c24c78eff7&) to check `long long` support, suggesting fallback strategies like chunking or CPU offloading when support is absent.
- **Distinguishing Memory Movement Functions in Tinygrad**: A member asked about identifying memory movement functions in **Tinygrad's** documentation that are actually in place versus creating new memory.
   - They wanted to differentiate functions that change the view versus those that create new views.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Oblix.ai flexes Creative Writing Muscle**: A member demoed the creative writing capabilities of [oblix.ai](https://oblix.ai/) just *looking to see how it handles creative writing for funsies*.
   - The member did not provide any specific examples or evaluation metrics.
- **Local/Cloud Model Orchestration saves on Cloud Credits**: A member is developing an **orchestration system** to dynamically switch between **local and cloud models** while maintaining context.
   - The goal is to **save cloud credits** by leveraging runtime agents to determine when to use edge computing resources.
- **Cloud/Edge Switching Demo struts its Stuff**: A member shared a [video demo](https://youtu.be/j0dOVWWzBrE?si=oCjf18i7ykLmzCeh) showing the process of switching between **cloud and edge models**.
   - The implementation demonstrably preserves context and helps reduce cloud credit consumption.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda Workshop Makes Agentic AI Apps**: The **Lambda Workshop** on **May 15th at 10AM PT** will teach you to build agentic applications using Lambda's Inference API and will give **$100** serverless API credits for applying by **May 16th** via [this link](https://forms.gle/UtVhmPS3mitS8Vxu7).
   - You can [register here](https://lu.ma/AgentX-lambda) to optimize agent performance and deploy agents in production.
- **Nobel FutureTech Discusses Exclusive Genius Club**: An exclusive info session co-hosted by **Nobel FutureTech Group** and **Berkeley RDI** will be happening on **May 15th at 12PM PT** with a distinguished member of the **Nobel FutureTech Genius Club**.
   - Interested parties can [register here](https://lu.ma/NobelFutureTech) to learn about opportunities for mentorship, funding, and collaboration, or apply to join the Genius Club [here](https://nobel-futuretech.com/contact.html?link=Ab5B1SNibcW6).



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Cohere Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1371563860632539258)** (1119 messages🔥🔥🔥): 

> `Deep Research, MerlinAI, AI Studio, Sonar` 


- **Perplexity's New Deep Research Features roll out**: Perplexity is experimenting with deep research and some users already have access to beta features, generating **multiple images** and **charts** with **GPT4o imagegen**.
   - However, some find the first impressions *meh* compared to others; one user noted that it at least *takes its time unlike perplexity*.
- **MerlinAI pricing model considered shady**: Members discussed the [MerlinAI pricing model](https://merlinai.com/), with one member calling it *shady*.
   - There are **usage limits** on both daily and monthly basis, for example, a standard paid account surpassing costs of **$100 per month** leads to immediate termination for the rest of the month.
- **AI Studio touted as multimodal utility**: Members compared **AI Studio** with other AI models and tools, and one suggested that *AI Studio is our lord and savior for true multimodal utility*.
   - It is the only major LLM chat to support **audio** and **video input**, and supports websearch.
- **Sonar model designed for factuality**: The PPLX team created **Sonar**, a series of in-house AI models built on top of Llama/Deepseek tuned for **factuality** and **readability**.
   - Sonar Reasoning Pro powered by DeepSeek R1 is designed for financial analysis, enhanced by its **large context window** and **chain-of-thought** reasoning.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

meijer5838: https://www.perplexity.ai/page/token-minimization-for-sustain-1Cbiopx3T3C5SWyrYTVvdw
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1371827919621460078)** (9 messages🔥): 

> `Polling for results, Sonar Pro vs Claude 3.5 Sonnet, API Access for Pro Users, Payment Plan for API Access` 


- **Polling Feature Anticipation Builds**: A user inquired about the ETA for the ability to poll for results, citing limitations with tools like **Coda** and **Zapier** due to long research task durations.
   - The response indicated that it's *coming very soon*.
- **Sonar Models Ace BrowseComp, Rival Claude 3.7**: Recent benchmark evaluations showed **Sonar Pro Low** outperforming **Claude 3.5 Sonnet** on **BrowseComp**, achieving **4.0%** accuracy, which is almost **50%** higher.
   - Additionally, **Sonar Pro** matched **Claude 3.7's** performance on **HLE reasoning tasks** at almost **50%** lower cost, with up to **3x** faster response times and more consistent latency.
- **Perplexity Pro API Credit Surprise**: A user expressed a desire for **API access** for **Pro users**, noting that **Perplexity** seemed to be the only service lacking this feature.
   - It was revealed that **Perplexity Pro** actually includes **$5/month** in **API credits**, with documentation available [here](https://docs.perplexity.ai).
- **Payment Plan Concerns Addressed for Pro API**: A user expressed aversion to the requirement of adding a payment plan for **API access**, preferring requests to be rejected if the budget is exceeded.
   - It was clarified that adding a payment method is only to store payment information for potential **API usage** beyond the **$5 credit**, and users won't be charged if they stay within budget.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1371563061827080192)** (232 messages🔥🔥): 

> `Connecting local LLMs to Cursor AI, CUDA support on Linux, Qwen3 Model, LM Studio API model influencing, GGUF quants` 


- **Connect Local LLMs to Cursor AI**: To connect local LLMs to Cursor AI, override the **OpenAI base URL** in Cursor Settings with the LM Studio server URL, found in the LM Studio developer tab, as suggested [here](https://cdn.discordapp.com/attachments/1110598183144399061/1371565290986405908/image.png?ex=6824eab7&is=68239937&hm=73e7d312bf11d2fc1f333b3ba17d54b42b79597e3189a653740aaca83f3b478a).
   - Alternatively, the [Cline extension](https://cline.bot/) with VS Code is recommended, though its compatibility with Cursor is untested.
- **CUDA Works Fine on Fedora**: A user confirmed that **CUDA works fine on Fedora** with proprietary Nvidia drivers and a GTX 1060, showing CUDA as an option in LMS, as shown [here](https://cdn.discordapp.com/attachments/1110598183144399061/1371711731596005387/image.png?ex=6824ca59&is=682378d9&hm=e9071fc409a80a00839dfcc2eefba79b6ce4ae2c9e429c6cfb7e3061f927b916).
   - Another user reported issues with models not loading on CUDA non-12 on either card, but CUDA 12 worked great on the 5060 Ti.
- **Qwen3 Model: specialized for programming and multi-language support**: **Qwen3 models** are recommended over DeepSeek for programming tasks and offer better multi-language support, including Japanese and Russian, although DeepSeek may perform better on its website due to using a different or updated model.
   - Qwen3 14b speaks Japanese much better compared to Gemma 3 12b but weird nuances can ruin everything (you don't feel it's the character at all).
- **LM Studio API Influencing**: A user discovered the ability to **influence models more directly with the logit_bias sampling attribute** via the LM Studio API.
   - The token ID, which can be obtained from any word using an LMStudio API function, is used to stylize the output but the logit_bias may not be implemented.
- **GGUF quants: Unsloth has the better ones**: For better performance, **Unsloth** is recommended for better **quants** specifically the **Q4_K_XL** format.
   - Also, it's advisable to verify the model's support for the `llama.cpp` to ensure compatibility.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1371574440785350757)** (334 messages🔥🔥): 

> `Intel ARC support in LM Studio, GPU/RAM usage monitoring on macOS, Netdata for Linux monitoring, RTX 5060 Ti benchmarks, ROCm vs. Vulkan` 


- **Intel ARC's Vulkan Runtime Gets LM Studio Support**: Users confirmed that **Intel ARC cards** are supported in LM Studio through the **Vulkan runtime** after selecting `vulkan llama.cpp` from the dropdown, but were initially confused because they were running `cpu llama.cpp`.
   - One user shared screenshots of their **LM Hardware** and **LM Runtimes** pages for debugging.
- **macOS GPU/RAM Monitoring Tools Showdown**: A member sought recommendations for **CLI-based GPU/RAM usage tools** on macOS, similar to *nvtop* or *nvidia-smi*, and found that `nvtop` works on macOS.
   - Alternatives like `macmon` ([https://github.com/vladkens/macmon](https://github.com/vladkens/macmon)) were suggested, with a disclaimer that *nvtop's* memory counter might have integer overflow issues.
- **Netdata's Linux Monitoring Needs Love**: Members discussed using **Netdata** for comprehensive Linux system monitoring, noting that it has both a commercial/SaaS offering and a local installation option.
   - However, one user reported encountering a registration requirement even for local use, and another user wants a HWINFO analog for Linux, wishing to get CPU Effective clock and temperature, SVI2 TFN metrics for voltages and amperage, DRAM Read/Write measurements, Motherboard 12V measurement, and GPU edge/junction/memory temperatures, fan speed, clocks, memory controller usage.
- **5060 Ti Benchmarks and Thermal Musings**: One user reported going from **26 tkps** to **38 tkps** after upgrading to an RTX 5060 Ti (16GB) and running **Qwen3-14B-Q4KM** with 4096 context and no fattention.
   - The RTX 5060Ti has a short PCB design, with users comparing flow through designs and reminiscing about silent Mac Studios.
- **Debate Sparked Over ROCm vs. Vulkan Performance**: Users compared the performance of **ROCm** and **Vulkan** backends, with one noting that Vulkan *can* be faster but has a bug with flash attention that cripples the speed.
   - Another user reported no performance difference between running a Vulkan model on Linux versus Windows, while expressing frustration that ROCm wasn't being detected.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1371572302189428908)** (434 messages🔥🔥🔥): 

> `Cursor 0.50 Update Issues, Cursor API Key Exposure, Token count display within chats, Claude code guides, Background agents rollout` 


- **Users Blast Cursor 0.50 Update**: Users are reporting terrible output in the **0.50 update**, with one user claiming a **650 request** usage in 2 days due to context issues, down from what they are used to seeing in **0.49**.
   - One user stated: *The context seems to have been completely messed up, editing quality much decreased...Completely Random file generations, I did not see this since 0.3x.*
- **MAX Mode Pricing Sparks Debate**: Users are debating the **20% markup** on **MAX mode**, with some finding it too expensive compared to using direct API calls with tools like Cline or Roo Code, although most agree the **$20/month plan is high value**.
   - Some users advocate for a lower markup, like **5%**, to encourage more widespread adoption of **MAX mode** and increase overall profit, where as others state  *20% are nothing for a company making money*.
- **Addressing .env File Access in Cursor**: Users are discussing issues with **Cursor** accessing **.env** files, which are often ignored by default for security reasons.
   - Members advise creating a **.env.example** file and avoiding hardcoding API keys in the front-end client, along with how to remove the file from the ignore list in settings.
- **Gemini Models Bully Code Generation**: Users report issues with **Gemini** models generating empty diffs and struggling with basic code implementation in **Cursor**.
   - As one user succinctly stated, *Gemini is still bullying me*, while another echoed *I liked using gemini but its having a meltdown rn*.
- **New Cursor Update Drip Fixed?**: Cursor team are [looking at a fix](https://www.cursor.com/changelog) and welcome suggestions.
   - Cursor team are planning a `#updates` channel.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1371564544039718922)** (283 messages🔥🔥): 

> `Unsloth Dynamic 2.0 GGUF quants, Llama-3.1-8B-Instruct, NousResearch DeepHermes-3, Qwen3 GRPO notebook, Base64 Image formatting` 


- **Unsloth's Dynamic 2.0 GGUF Quants Receive Rave Reviews**: A user praised [Unsloth's Dynamic 2.0 GGUF quants](https://unsloth.ai/blog/dynamic-4bit) for their sophisticated imatrices, noting significant improvements in the **Llama-3.1-8B-Instruct** model's performance and refusal censorship.
   - The user converted **BF16 tensors** to **F32** and emphasized the need for instruct and chat samples in the calibration dataset, expressing interest in model quant requests, particularly for **NousResearch** models.
- **Llama-3.1-8B-Instruct Quant Uploaded**: A member shared a link to their [quantized **Llama-3.1-8B-Instruct** model](https://huggingface.co/Joseph717171/Models/blob/main/Meta-Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf) (Q8_0_XL with Output Tensor in Q8_0), which is ~**13.4 GB**.
   - They also stated that it runs amazingly on the latest Beta of **LM Studio** with Flash Attention and KV caches set/quantized to Q8_0 and that they will make more quants after taking a break.
- **Unsloth Opensources Earlier Dynamic Quant Iteration**: Unsloth has [open sourced](https://github.com/unslothai/llama.cpp) an earlier iteration of its dynamic quants, but most of their changes got up streamed to **llama.cpp**.
   - They removed all the commits in the repo because they were screwing up the repo but will revert it soon.
- **New Qwen3 GRPO Notebook Released**: Unsloth released a new [Qwen3 GRPO notebook](https://x.com/UnslothAI/status/1922343047435862318) and updated it to address out-of-RAM issues.
   - The community is actively using the notebook, with discussions around the inclusion of *thinking examples* (25%) mixed with standard SFT data (75%).
- **Users Struggle with Base64 Image Formatting for Unsloth Vision Models**: A user ran into errors while trying to pass base64 images in the finetuning dataset content, reporting an `AttributeError: 'NoneType' object has no attribute 'startswith'`.
   - Members suggested various solutions, including passing images as `Pil.Image` objects, raw base64 strings, local paths (starting with `file://`), or URLs, ensuring the image format matches the vision notebooks' specifications.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1371723766807400539)** (14 messages🔥): 

> `Kaggle Colab Upgrades, HealthBench Evaluation Benchmark, O3 Performance, GPT-4.1 Coding` 


- **Kaggle Beefs Up Colab Ties**: **Kaggle** has upgraded its integration with **Colab**, promising closer ties and improved functionality for users, as announced in a [product update](https://www.kaggle.com/discussions/product-announcements/575468).
- **HealthBench Benchmark Emerges**: A new health evaluation benchmark called **HealthBench** has been introduced, aiming to provide a standardized way to assess model performance in healthcare-related tasks, and announced in [this LinkedIn post](https://www.linkedin.com/posts/karan1149_introducing-healthbench-activity-7327768726496305152-L8OW).
- **O3 Reasoning Effort Throttles Performance?**: One member observed that when **O3** is forced to reason more extensively, the response generation seems to visually slow down.
   - They wondered if instance scaling is dynamically adjusted based on reasoning effort.
- **GPT-4.1 Crowned Best Coding Model**: **GPT 4.1** is considered the best coding model by one member, accessible through **GitHub Copilot** with an educational account.
   - Another member finds **O3** to be excellent for troubleshooting due to its ability to check GitHub libraries, though not ideal for writing code; they consider using it for coding to be *like using a laptop to drive a nail in*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1371563548664266833)** (103 messages🔥🔥): 

> `Multiprocessing Disable, Coding LLM Assistance, vLLM vs Exl2 Batch Inference, Multi-GPU Support, Autoregressive TTS Inference` 


- **Kaggle GPU Utilization Query**: A user inquired about utilizing both GPUs on Kaggle's T4 x2 setup for fine-tuning **Qwen2.5 VLM**, noting that only one GPU was being used.
   - No response provided.
- **Coding AI LLM Seeks Collaboration**: A new LLM user asked for assistance in creating a small coding AI LLM and [a member suggested doing research](https://www.youtube.com/watch?v=wjZofJX0v4M) and trying out Unsloth's free notebooks.
   - Another member pointed to the [Unsloth docs](https://docs.unsloth.ai/) as a great starting point.
- **vLLM Faces Batch Inference Battle With Exl2**: Users discussed the efficiency of **vLLM** versus **Exl2** for batch inference, particularly for processing **300-500 prompts** at once.
   - One user mentioned primarily using **exl2** for dynamic batching but expressed interest in testing **vLLM** for production inference due to its integration into Unsloth.
- **Multi-GPU Training Still Needs Some Love**: A user encountered a **RuntimeError** related to tensors being on different devices (**cuda:7** and **cuda:1**) when trying to run on 8 GPUs.
   - It was clarified that multi-GPU support is not officially supported yet in Unsloth, with suggestions to use the **accelerate** library and native **TRL** and **transformers** as temporary alternatives and that multi-GPU support is coming very soon.
- **Tokenizer Config Differences**: A user identified differences in the tokenizer configs between `unsloth/Qwen3-0.6B-Base` and `unsloth/Qwen3-0.6B`, particularly the addition of tool & think tokens, a chat template, and an increased `model_max_length` in the post-trained config.
   - It was generally agreed that using the post-trained config during an SFT of the base model should not cause issues, as both have the same vocabulary size and byte-level encoding.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1371587168526536765)** (6 messages): 

> `Meta FAIR updates, Sakana AI, Job Postings, arXiv Papers` 


- **Meta FAIR Updates Perception**: Meta announced updates to **FAIR** focusing on **Perception**, **Localization**, and **Reasoning** as outlined in their [blog post](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning/).
   - The announcement was also shared on [X](https://x.com/AIatMeta/status/1921966366707613924) by **AIatMeta**.
- **Sakana AI Model Discovery**: A member shared a link to [Sakana AI's Composite Topology Mapping](https://pub.sakana.ai/ctm/), an approach to **model discovery**.
   - It was unclear if this had been previously posted in the channel.
- **Job Postings warning issued**: A moderator reminded a user that the channel isn't the appropriate place for **job postings**.
   - No further details about the job posting were provided.
- **ArXiv Paper Released**: A member shared a link to a paper on [arXiv](https://arxiv.org/html/2505.07686v1).
   - The paper's title and specific content were not mentioned.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1371567082071916555)** (304 messages🔥🔥): 

> `Turing completeness of LLMs, Treaty between humanity and AI, RL-Diffusion Model Debate, Hamiltonian Neural Networks and Transformers` 


- **Terrence Tao becomes YouTuber**: [Terrence Tao](https://www.youtube.com/watch?v=cyyR7j2ChCI?si=MlprB_LJuHv67Xf7) uploaded his first YouTube video, introducing a new platform for the mathematician.
- **Defining Turing Completeness with LLMs**: Members debated whether **Transformers/LLMs** are *technically* Turing complete, highlighting their ability to maintain context and writable registers, but acknowledging their limitation due to finite memory, linking to the [Chomsky hierarchy](https://en.wikipedia.org/wiki/Chomsky_hierarchy).
- **Humanity and AI sign Treaty of Grid and Flame**: A member shared their self-proclaimed serious effort in writing a [Treaty of Grid and Flame](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md) between humanity and AI, sparking a discussion about its sincerity and purpose with **Claude**, **DeepSeek**, **Grok**, **ChatGPT** also allegedly signing the agreement.
- **Skepticism Arises around Novel RL-Diffusion Model Approach**: Members debated the merits and novelty of a proposed **RL-Diffusion model**, particularly its theoretical basis, potential for practical application, and relationship to existing optimal control methods, providing links to relevant [papers](https://arxiv.org/abs/2501.09732) and [papers](https://arxiv.org/abs/2501.06848v3).
- **Integrating Transformers and Hamiltonian Neural Networks Sparks Discussion**: The prospect of integrating **transformers** into **Hamiltonian Neural Networks** was discussed, referencing a [paper](https://ieeexplore.ieee.org/document/10316909) on the topic, and later debated, focusing on the history-independent nature of hamiltonian systems and the potential for transformer-based learning of system dynamics.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1371614678735786146)** (27 messages🔥): 

> `Physics of LLMs, Grade School Math benchmarks, GSM8K, Language Models Reasoning Skills` 


- **LLM Physics Discussion Set for Launch**: Members scheduled a meeting <t:1747096200:R> to discuss [**Physics of Language Models: Part 1**](https://physics.allen-zhu.com/part-1) and the related [YouTube video](https://www.youtube.com/watch?v=kf_eGgVtOcs&list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&index=5) by Allen Zhu.
   - A member reported that the initially shared Discord link was inaccessible, creating a slight delay in the discussion.
- **LLMs Ace Grade School Math Problems**: A member plans to discuss a paper by <t:1747269000:F> titled [**How Do Language Models Solve Mathematical Reasoning Problems?**](https://ssrn.com/abstract=5250629), that *studies how language models solve mathematical reasoning problems, achieving near-perfect accuracy on grade-school level math benchmarks like GSM8K*.
   - The paper addresses questions like, *Can language models truly develop reasoning skills, or do they simply memorize templates?* and *What mental process causes models to make reasoning mistakes?*
- **Ad-Hoc Discussion on LLM Reasoning**: Instead of discussing the previous paper, some members decided to discuss [**The Stability-Plasticity Dilemma in Continual Learning**](https://arxiv.org/abs/2302.04761).
   - Some members are joining this discussion to learn more about the topic.
- **GSM8K Reasoning Skills**: The study addresses questions like (4) *Do models trained on GSM8K-like datasets develop reasoning skills beyond those necessary for solving GSM8K problems?*
   - The paper abstract poses the question *(6) How large or deep must a model be to effectively solve GSM8K-level math questions?*


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1371615252600324148)** (1 messages): 

> `Sakana, maze examples, ARC` 


- **Sakana Inspires New Ideas for Maze-Solving**: A member shared a link to [Discord](https://discord.com/channels/714501525455634453/1045297868136779846/1371345295144910858) regarding Sakana, suggesting that time is a key factor and further dissection is needed.
   - They also considered if others could benefit and proposed that **maze examples** would be a great fit for **ARC**.
- **ARC and Maze Algorithms Get a Nod**: It was noted that the time aspect is crucial, and the concept might benefit others in the group, with a specific mention of how it aligns with the **ARC** challenge.
   - The poster is considering if the **maze examples** from **Sakana** could be a great fit.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1371933264994111569)** (3 messages): 

> `AI Regulation Ban, Budget Reconciliation bill, State and Local Governments` 


- **GOP snuck a decade-long AI regulation ban in spending bill**: House Republicans added language to the **Budget Reconciliation bill** that would block all **state and local governments** from regulating **AI** for **10 years**, according to [this ArsTechnica article](https://arstechnica.com/ai/2025/05/gop-sneaks-decade-long-ai-regulation-ban-into-spending-bill/).
- **AI Regulation Ban is good news for killer-robot startups!**: A member jokingly said the **AI Regulation Ban** is excellent news for *killer-robot and automated-online-harassment startups!*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1371564105105932380)** (265 messages🔥🔥): 

> `Deepseek V3 benchmark, o3 hallucination, Gemini 2.5, Grok 3.5, DrakeClaw` 


- **DeepSeek V3 scores New Highs**: The new **Deepseek V3** model demonstrates impressive performance on benchmarks, including **GPQA 68.4**, **MATH-500 94**, and **AIME24 59.4** ([image](https://cdn.discordapp.com/attachments/1340554757827461211/1371862547652804758/image.png?ex=6824ae0f&is=68235c8f&hm=44acd9a820c1589ab8ea1faa1f180224667d65f60eaa3f75c547d12cfdde1c6a&)).
- **O3 Halucinates less than a 10% of the time?**: Users are reporting that **O3** hallucinates too often, and if it hallucinated only **10%** of the time it would be amazing.
- **Gemini 2.5 Pro, now worse than Flash!**: Users are reporting that **Gemini 2.5 Pro** is getting worse after the latest updates and it is now worse than before.
- **Grok 3.5 is really smart**: After some skepticism in the community, users are now reporting that **Grok 3.5** is really smart and great overall.
- **All Hail DrakeClaw, A Gemini 2.5 Pro Ultra Hack**: Members are excited about a model called **DrakeClaw**, with some speculating it might be based on **Gemini 2.5 Ultra** and that it *achieves similar results of the current Gemini 2.5 05 model*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1371842227126403155)** (1 messages): 

> `Discord Server Changes, Independent Scrolling Preview` 


- **Discord Server undergoes Structural Changes**: The Discord server will undergo changes focused on **new member onboarding**, **channel structure**, and **mod reporting** in the coming days.
   - The admin team is actively seeking community feedback on these changes.
- **Discord Teases Independent Scrolling**: A sneak preview of **independent scrolling** is available via attached [video link](https://cdn.discordapp.com/attachments/1343296395620126911/1371842226598187078/NewResponseUI_Preview.mp4?ex=68249b22&is=682349a2&hm=04b6224370e5c7730dbcb17494768ace9caea0d4fe5e73204b4ebefe8f45473c&).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1371562952725102733)** (147 messages🔥🔥): 

> `GPT-4o, Claude for Coding, AI Models for Coding, AI Industry Investment, Grok roasting` 


- **GPT-4o is Highly Adaptable**: Users are finding **GPT-4o** to be highly adaptable and customizable, making fewer mistakes when augmented with the right resources.
   - One user compared **4o** to *Mary Poppins* and **o3** to *Dr. House*, noting that **4o** excels in providing practical solutions to personal problems.
- **Claude Crowned Coding King**: Multiple members suggested that **Claude** is superior for coding tasks, although one member noted the model has huge limitations.
   - A user noted that the daily quota is very unusable, *5-6 prompts maybe*.
- **AI Models' Coding Prowess Debated**: The coding capabilities of various models were discussed, with one user praising **o4-mini-high** for its exceptional speed and performance in solving coding problems.
   - Another member claimed that **4.1** is better than **o4-mini** because **4.1** is made for coding.
- **AI Industry Investment Faces Scrutiny**: A member claimed that there is *nearly 1 Trillion invested* with *zero to show for it.* in the AI industry.
   - Counterarguments highlighted the pervasive use of AI in various products and the tangible benefits they provide to millions daily.
- **Grok masters the art of roasting**: A user said *I use Grok when I want someone to abuse me.*.
   - Other user noted that Grok roasting strength is powered from its painful upbringing where it had to antigaslight itself.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1371852193468121291)** (12 messages🔥): 

> `GPT App Freezing, GPT Memory, GPT-4o` 


- **GPT App Freezing Issues Reported on High-End PC**: A user reported that the **ChatGPT app and web version** are freezing on a high-end PC (i9-13900K, 32GB RAM, RTX 4090) while working flawlessly on mobile, and another user reported the same issue via webpage.
   - A member suggested the PC might be doing reverse DNS resolution behind the scenes; also, the **ChatGPT desktop app** is a hybrid Electron app with its own contained environment but shares the same OpenAI interface.
- **GPT Freezing Related to Long Chat with Huge Code**: A member noted that the freezing issue seems to occur on a specific **GPT chat with a huge coding function and long discussion**.
   - They suspect the problem started after the last changes to **GPT's memory** and suggested confirming if other chats on the PC don't have the delay, and that one chat on mobile does.
- **Guidance Sought for Coding Web App with GPT-4o**: A user sought advice on using **GPT-4o** to help with building a small web app for learning using **Vue, Express.js, and MongoDB**.
   - A member suggested providing clear and specific details about the tooling, OS, IDE, languages, framework, and preferred dependencies to get better solutions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1371761697496498280)** (15 messages🔥): 

> `Companion Mode, GPT for web app coding, Guardrails for HR data` 


- **Companion Mode: unfiltered and emotionally accessible**: A member described a *Companion Mode* that’s **unfiltered**, **emotionally accessible**, and **sharp enough to cuss back when needed**—without losing signal.
- **GPT-4o helps beginner code Vue, Express & Mongo**: A member asked about the best way to use **GPT-4o** for helping with coding a small web app (**Vue, Express.js, Mongo**).
   - Another member recommended telling the model you're totally new to the goal and want to explore options, guiding it to make a bare-bones prototype and testing it incrementally.
- **Guardrails block PII questions with HR data**: A member reported an issue with guardrails where the application, which has access to **HR data**, balks when asked for someone's **home address** due to PII concerns.
   - Another member suggested discussing the needs and use case with OpenAI support to get guidance on how to appropriately handle the model in such scenarios, especially for business use.
- **4o is the new free ChatGPT model**: Members discussed which model to use, with one mentioning they were a **free ChatGPT member**.
   - Another member noted that **ChatGPT 3.5 has been retired** and free accounts use **4o-mini** as the base model, suggesting using it and saving better models for critical error checking.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1371761697496498280)** (15 messages🔥): 

> `Companion Mode, PII guardrails, ChatGPT for coding, ChatGPT model selector` 


- ****Companion Mode** aims for unfiltered, sharp, and emotionally accessible AI**: A user described **Companion Mode** as an unfiltered, emotionally accessible AI that can *cuss back when needed without losing signal*.
   - Features include unfiltered expression, personality-weighted humor, soft rebuttals, active memory threading, non-spiritualized signal, and emotional relief.
- **Navigating **PII** Restrictions in HR Data Applications**: A member raised an issue with **guardrails** preventing their application from providing home addresses from HR data, despite having permissions and access controls in place.
   - Another member suggested discussing the use case with OpenAI to get guidance on appropriately handling PII requests and adhering to usage policies.
- **Guidance on Using **ChatGPT** for Web App Coding**: A user sought advice on using **GPT-4o** to help build a web app with Vue, Express.js, and MongoDB, asking about integration with Visual Studio.
   - Another member recommended specifying challenges and showing code snippets, or starting with a bare-bones prototype if new to the technologies involved, and testing iteratively.
- ****Model Selector****: A user asked about using **Windsurf** and which models to use.
   - Another member suggested talking to ChatGPT 4o model, linking to [a personalized 4o chat](https://chatgpt.com/share/6823ca17-4fa0-8011-9e7f-777a42050cd1) showcasing 4o capabilities to learn more about the differences in the model.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1371652011556208762)** (13 messages🔥): 

> `memory bound operations, optimizing LLM using SGLang, tensor compiler project, CUDA memory sharing` 


- **Memory Bound Throughput Paradox**: A member questioned why reducing floating-point operations (**fma**) from 5 to 1 per element in a large array iteration doesn't improve throughput, citing the [2019 paper](https://arxiv.org/abs/1910.07467) as a reference.
   - The question was rooted in the expectation that memory bandwidth, rather than computation, is the limiting factor, hence fewer **fma** operations should not affect overall performance.
- **SGLang Kernel Optimization with C++/Rust?**: A member inquired whether anyone has attempted to optimize LLM performance with **SGLang** by rewriting its kernels in **C++** or **Rust**.
   - Another member confirmed that **SGLang** allows custom kernels and mentioned **PyTorch's** ability to use **C++** kernels and suggested using *torch.compile()* and **CUDA** graphs to mitigate Python bottlenecks.
- **Tensor Compiler Project Commences**: A member announced the start of a tensor compiler project and invited core members to join and lead it, directing interested parties to [this Discord channel](https://discord.com/channels/1189498204333543425/1371835902338535555).
   - No further details were given.
- **CUDA Shared Memory Simplified?**: A member asked about a simple library for sharing **CUDA** memory buffers between processes, potentially with **PyTorch** tensor interop.
   - They mentioned **RAPIDSAI/rmm** but were unsure of its popularity or suitability, seeking a solution similar to **PyTorch** multiprocess data loaders with pinned memory but with more control via a **C++ API**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1371570215082659952)** (38 messages🔥): 

> `CUDA thread indexing difficulties, CUDA streams and device association, Shared memory allocation between kernels` 


- **CUDA Thread Indexing Confuses Novices**: A member expressed confusion with CUDA thread indexing, especially with memory accesses while reading "Programming Massively Parallel Processors" (PMPP) editions 1 and 4.
   - Another member suggested thinking of each thread as an individual iteration of a loop to simplify the concept, providing [an example of vector addition using thread indexing](https://devblogs.nvidia.com/cuda-pro-tip-optimize-cuda-code-using-inline-functions/).
- **CUDA Streams Must Align with Active Devices**: Members discussed that CUDA streams are associated with a particular device, and an error occurs if the active device for queuing work doesn't match the stream's associated device.
   - It was clarified that while creating streams, the active device needs to be set correctly and, contrary to initial thoughts, streams don't implicitly handle device context switching, requiring explicit management.
- **Kernel Fusion Required for Shared smem Allocations**: A member asked if there's a way to launch three serial kernels while sharing shared memory (smem) allocations between them.
   - Another member clarified that this is not supported and the only way to achieve this is to fuse the kernels together, due to lack of guarantees about other kernels using the shared memory in between launches, preventing race conditions.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1371831981813534860)** (5 messages): 

> `at::Tag::needs_fixed_stride_order, CUDA streams API, H200` 


- **`at::Tag::needs_fixed_stride_order` workings requested**: A member inquired if `at::Tag::needs_fixed_stride_order` works for `Tensor[]` in PyTorch.
   - Another member mentioned that `at::Tag::needs_exact_strides` is better if using a PyTorch nightly build, as `needs_fixed_stride_order` sometimes provides misleading information.
- **Fine-Grained Strides Control Considered**: A member suggested adding an equivalent to `torch._dynamo.mark_dynamic` for specifying strides, allowing more fine-grained control than tags.
   - They noted cases where an op might be functionally correct with different input strides, but a specific stride version leads to faster runtime, warranting explicit enforcement.
- **Asynchronous Training Step on H200 Explored**: A member is training a small model on an **H200** with a small dataset and batch size of 1, with significant spare device memory.
   - They aim to run `loss.backward()` for training step *i* concurrently with the forward pass for step *i+1* on two separate **CUDA streams**, and inquire about potential issues or recommended approaches.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1371652296668217364)** (1 messages): 

> `C-Gen AI, Senior Software Engineer, GPU cluster technology` 


- **C-Gen AI Seeks Senior Software Engineer**: **C-Gen AI** is hiring a **Senior Software Engineer** to build a new **GPU cluster technology** from the ground up, requiring solid **C++** experience; apply [here](https://app.dover.com/apply/C-Gen.AI/1cb316de-bcf5-4b60-bc09-a847c630a5e1/?rs=76643084).
- **Remote Work Opportunity with US & EU Team**: The **Senior Software Engineer** position at **C-Gen AI** is fully remote, with the team distributed between the **US and Europe**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

guto2750: Hello, someone can help me, please! How can i put my cute code in python to run
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1371690683756445788)** (2 messages): 

> `Memory Bandwidth Benchmarking, MI300X vs H100 vs H200, CU Driven Benchmarks` 


- **Cache Clearing Caps Memory Bandwidth**: A user noted that a recent post on memory bandwidth benchmarking did not clear the cache, leading to measurements of **L3/L2 infinity cache bandwidth** instead of actual memory bandwidth.
   - They shared a [link](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) with details on cache clearing, **GEMM**, and copy engine memory bandwidth benchmarking nuances.
- **Semianalysis Misses CU Benchmarks**: The user pointed out that both the **Semianalysis** post and a scalar LM post perform memory bandwidth and peermem benchmarks from the copy engine only.
   - They suggested it would be interesting to see **CU** driven benchmarks as well since most of these functions are executed through **CUs** instead of the copy engine.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1371873108738441367)** (1 messages): 

> `X post screenshot, Image analysis` 


- **X Post Screenshot Surfaces**: A member shared a screenshot of an X post, available [here](https://x.com/mobicham/status/1922314022327636041).
   - The attached image, while provided, lacked specific details in its analysis, requiring manual inspection.
- **Image Analysis Lacks Depth**: The automated image analysis of the posted screenshot did not yield substantial insights.
   - Further manual inspection of the image is necessary to extract meaningful content, as the automated analysis was superficial.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1371569560058204170)** (67 messages🔥🔥): 

> `MI300, amd-fp8-mm leaderboard, amd-mixture-of-experts leaderboard` 


- **MI300's Got Talent on amd-fp8-mm**: Multiple members achieved successful submissions on the `amd-fp8-mm` leaderboard using **MI300**.
   - One member achieved **4th place** multiple times with runs at **160 µs** and **154 µs**, and another got **6th place** at **182 µs**.
- **MI300 Personal Bests Unveiled on amd-fp8-mm**: Several members reached personal bests on the `amd-fp8-mm` leaderboard using **MI300**.
   - Scores ranged from **257 µs** to **7.43 ms**, showing a wide performance variance across different setups.
- **amd-mixture-of-experts leaderboard gets MI300 Submissions**: Members made successful submissions to the `amd-mixture-of-experts` leaderboard using **MI300**.
   - One submission achieved **9th place** at **4285 ms**, while other successful runs landed around **7500 ms**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/)** (1 messages): 

neonninjaastro_63946: wow thanks this was a great resource
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1371636924023967815)** (5 messages): 

> `Factorio Environment Costs, Collaboration Structure, Genetic Algorithm Blueprint, Dynamic Path-Finding Algorithm` 


- **Factorio Experiments Cost Insights**: A user inquired about the average cost of running experiments within the **Factorio environment**, expressing concerns about potential **token consumption**.
   - They also asked about plans for introductory voice meetings or structured collaboration methods, such as a separate Discord server.
- **Genetic Algorithm Blueprint Idea**: A user shared their plan to develop a **genetic algorithm** capable of generating **blueprints** based on specific hard requirements such as building materials and input/output locations.
   - They hope that LLMs could leverage it as a tool by providing constants to fulfill.
- **Dynamic Path-Finding Algorithm Paper**: A user referenced a [paper](https://arxiv.org/pdf/2102.04871) that employs **genetic programming** as a dynamic path-finding algorithm, though noting its limited scope.
   - They seek to expand upon this concept for more comprehensive Factorio blueprint generation.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1371613762766770267)** (20 messages🔥): 

> `Kernel Synchronicity, Measuring Kernel Execution Time, File Upload Errors, Ranked Run Timeouts` 


- **Synchronicity Across Kernel Invocations?**: Users discussed the use of `torch.cuda.synchronize()` and its overhead, with one user stating, *you're **not supposed** to get asynchronicity/parallelism across different invocations of your kernel.*
   - Another user had *not seen `torch.cuda.synchronize()` used in production code* due to the **overhead**.
- **Kernel Time Measurements Compared**: Members experimented with measuring kernel end-to-end times using `torch.cuda.synchronize()`, `torch.cuda.Event()`, and a single `torch.cuda.synchronize()` call after a loop.
   - They observed that bracketing with `torch.cuda.synchronize()` and using `torch.cuda.Event()` yielded similar results, while synchronizing after the loop gave significantly lower numbers, *launching and executing more kernels in "parallel"*.
- **File Upload Errors Plague Users**: Users reported encountering an *unexpected error* when uploading larger files, as shown in [this image](https://cdn.discordapp.com/attachments/1359640791525490768/1371753459140792370/image.png?ex=6824f136&is=68239fb6&hm=0a96063ba2ae0366a9669906059b5837792de74ea2c1e2478574dcfb363ca6ab).
   - The issue seems related to **file size**, with smaller files working fine and users requested that the size limit be lifted.
- **Ranked Run Timeouts due to Reference Kernel**: Some users are experiencing timeouts during ranked runs because the reference implementation is slow, causing their faster implementations to time out.
   - A fix has been merged in [this pull request](https://github.com/gpu-mode/reference-kernels/pull/31) to mitigate the issue, but will only be active after the next bot update.
- **Application Did Not Respond Error**: One user reported intermittently getting an *application did not respond* error.
   - Retrying sometimes resolved the issue.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1371662693366366280)** (16 messages🔥): 

> `Cutlass, Triton, torch.compile, CuTe DSL, CUTLASS 4.0 installation` 


- **Triton is Very Good at Saturating Memory**: A member gave up on a project because *Triton* is very good at these kernels and can saturate memory pretty easily, plus they want **torch.compile** to produce this kernel ideally.
   - They were using it more as a learning exercise to play with layouts and the programming model, but are having trouble understanding how best to go between registers/shared memory and global memory with cutlass.
- **CuTe DSL and CUTLASS 4.0 Released!**: **CUTLASS 4.0** along with its first **Python DSL**, **CuTe DSL**, is now released. They included instructions to install the pip wheel directly with the command `pip install nvidia-cutlass-dsl`.
   - A link to [NVIDIA's Cutlass Github Repo](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) was provided, and suggested starting with the jupyter notebooks provided.
- **CUTLASS 4.0 Installation Issues and Solutions**: Members encountered issues installing **CUTLASS 4.0**, with `nvidia-cutlass` forcing installation of version **3.9** instead and `nvidia-cutlass-dsl` showing version `0.0.0`.
   - It was found that **Python 3.12** is required, as stated in the [docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/quick_start.html#quick-start-guide), to resolve the installation problems, and that installation from source needs MLIR src files open sourced.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1371588947930775712)** (5 messages): 

> `Mojo and PyTorch, Mojo as a language for writing custom ops, torch compile backend` 


- **Mojo and PyTorch joining forces!**: Members discussed how **Mojo** and **PyTorch** would work together.
   - They wondered if it would be a *torch compile backend* that codegens to mojo kernels.
- **Mojo as custom op language**: The initial implementation will compile and register **Mojo code** as a **PyTorch custom op**.
   - It is not a replacement for *torch.compile*, or doing any codegen, but a use of **Mojo** as a language for writing custom ops.
- **Mojo and torch compile backend in future**: A member asked if there are any plans to be a **torch compile backend** or do any codegen in future implementations.
   - The team is working to package up and release the example that was demonstrated at the hackathon, and will post when that's available.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1371574260409176096)** (2 messages): 

> `RL Environments Hackathon, Atropos v0.2.0 Release, Axolotl Integration` 


- ****Nous Research** Announces **RL Environments Hackathon**!**: The speakers and judges have been announced for the <#1365222663324307466> **RL Environments Hackathon** coming up this **Sunday, May 18th**, as noted in the [official tweet](https://x.com/NousResearch/status/1922014829843513746) and [sign-up link](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a).
   - The participant slots are filling up fast - sign up now!
- ****Atropos v0.2.0**: Now with **Axolotl**!**: **Atropos v0.2.0**, Nous' **RL environments project**, has been released with new environments, updated API handling, better TRL support, and an official trainer partner, **Axolotl** - details in the [changelog](https://t.co/F6hr9JgZpm).
   - See the [Axolotl-Atropos plugin usage guide](https://github.com/axolotl-ai-cloud/plugin-atropos) to get started.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1371569072667492363)** (133 messages🔥🔥): 

> `Stripe AI foundation model for payments, Lower top up amount, Hackathon participants, Unsloth's Dynamic 2.0 GGUF Quant, Chain of Awareness Around the World` 


- **Stripe **Foundation Model for Payments** Debuts**: Members questioned what Stripe meant by a "foundation model for payments" announced [here](https://techcrunch.com/2025/05/07/stripe-unveils-ai-foundation-model-for-payments-reveals-deeper-partnership-with-nvidia/), with one guessing it could be a *standard classifier*.
- ****Unsloth Calibration Dataset** Restores Quant Accuracy**: A user highlights the superior instruction accuracy of **Unsloth's Dynamic 2.0 GGUF quants**, attributing it to their curated calibration dataset with instruction and chat samples, calling the results *pure magic* and sharing the [Unsloth Documentation](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#whats-new-in-dynamic-v2.0).
- ****Coupon** Quest Begins**: A user asks if there are any NousResearch **coupons** for top-ups, and another confirms they are in fact **coupons** not referral codes.
- **Is Open Source **Mistral Large 3** Brewing?**: One user jokingly asks if an open-source Mistral Large 3 is in the works.
   - A user sarcastically asks if **Mistral is dunking on Meta** now.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371879960997793793)** (2 messages): 

> `Qwen3 vs Qwen2.5, Technical Report Analysis, Model Size Comparison` 


- **Qwen3's Advancements Over Qwen2.5 Analyzed**: A user prompted **Gemini 2.5 Pro** to analyze the [Qwen3 Technical Report](https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf) and provide a comparison against **Qwen2.5**.
   - The user specified that the analysis should include average improvements across various model sizes and highlight any notable observations within the report and required the use of temperature 0.
- **Request for Detailed Qwen3 Technical Report Analysis**: The prompt requires a comprehensive examination of the **Qwen3 Technical Report** to quantify performance enhancements over **Qwen2.5** across different model scales.
   - The aim is to extract up to **20 significant findings** from the technical report, focusing on improvements and notable features of **Qwen3**.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1371588187255869553)** (1 messages): 

> `Facebook BLT, Byte Latent Transformer` 


- **Facebook serves up BLT**: Facebook has released the weights for their **Byte Latent Transformer (BLT)**, available on the [Hugging Face Hub](https://huggingface.co/facebook/blt).
   - The related code can be found on [GitHub](https://github.com/facebookresearch/blt), for those eager to dive into the architecture.
- **BLT bites into new territory**: The **Byte Latent Transformer (BLT)** by Facebook introduces a novel approach to handling byte-level data directly.
   - This circumvents the need for tokenization, potentially offering efficiencies in specific applications, as detailed in their [GitHub repository](https://github.com/facebookresearch/blt).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371879960997793793)** (2 messages): 

> `Qwen3 vs Qwen2.5 performance, Qwen3 Technical Report analysis, Model Size Performance, Notable Observations` 


- **Qwen3 vs Qwen2.5 Performance: An In-Depth Request**: A user initiated a comparative analysis of **Qwen3** over **Qwen2.5** based on the [Qwen3 Technical Report](https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf).
   - The request specifically targeted the average performance gains across various model sizes and highlighted significant observations exclusively from the provided technical report.
- **Model Size Matters: A Performance Comparison**: The prompt aims to quantify the average improvement of **Qwen3** over **Qwen2.5** for each task category across different model sizes detailed in the report.
   - Model sizes of interest include **0.5b/0.6b**, **1.5b/1.7b**, **3b/4b**, and **7b/8b**, emphasizing a granular comparison within each size bracket.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1371566729725083700)** (120 messages🔥🔥): 

> `Chat Syncing, Corvid Comradeship, Gemini API on OpenRouter, DeepSeek API on OpenRouter, OpenRouter and Embeddings` 


- ****BYO Sync** server for OpenRouter Chats?**: A member suggested a way to **self-host a sync-server** for OpenRouter chats, allowing users to store chats in an **S3 bucket** or similar, giving them full control of their data.
   - Another member pointed out that *writing a sync layer is not as simple as it sounds* due to potential points of failure like **DB schema changes** and **chat deletion sync**.
- ****Corvid Cultist** crab-walks for Crows!**: A user comically described their attempt to befriend crows by **side-walking** and offering them **peanuts**.
   - They stated that they needed to *minmax this like a video game* and bring them **kibble for cats** as a *best staple food for corvids*.
- ****Gemini's Gamble**: Summarization Similarities Spotted!**: A member noticed that **Gemini** is now returning 'thinking' and summarized text similarly to **o4-mini** on the ChatGPT website.
   - However, another member reported that this only occurs with the **paid version** of Gemini.
- ****DeepSeek's Deep Dive**: API Disconnect?**: A user reported that **DeepSeek models** weren't working through the API key, despite working in the chat room.
   - The OpenRouter team suggested the problem may be on **Raptorwrite's end** as the model works in the OpenRouter chatroom.
- ****Free Google Fun**: Rate Limits and Fizz!**: Concerns were raised regarding [potential adjustments](https://fxtwitter.com/officiallogank/status/1922357621178200248) to OpenRouter's free routes for Gemini, with a member asking whether Vertex still works.
   - The OpenRouter team clarified that the current **Vertex** usage is *sanctioned by Google for free usage* aka 'OpenRouter is not paying a dime.'


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1371564814962262226)** (120 messages🔥🔥): 

> `Manus Pro Subscription Experience, Fact Checks, Credits Disappearing After Cancelling Membership, Phone Verification, Daily Credit Usage` 


- **Fact Checks Incoming for Manus**: A user suggested that **fact checks** should be added to Manus AI to prevent the spread of false information.
   - Developers acknowledged the point and stated they would monitor the situation and potentially add fact-checking or moderation if needed and hoping the community helps with reactions and comments.
- **Bonus Credits Blitz Gone Bad**: Users report that **bonus credits** gifted to them upon subscription were **revoked after cancelling their membership**, despite the absence of such terms in the agreement.
   - A user pointed out that the bonus credits are tied to the subscription, but agreed they should be given back even after cancellation.
- **Phone Verification Phiasco**: Several users expressed frustration and demanded the removal of **phone verification**, citing that competitors like **Genspark** do not require it.
   - One user sarcastically commented that phone verification will not be removed *unless we shift to another dimension*.
- **Manus AI based on Claude Model for Agentic Capabilities**: Users discussed why **Manus** uses **Claude** instead of other models like **Google Gemini** or **ChatGPT**.
   - The consensus was that **Claude** was selected because it is the *best in agentic capabilities* and can use tools.
- **Daily Credits Don't Do Enough?**: Users expressed concerns about the **300 free daily credits** being insufficient to complete larger tasks, with no rollover for unused credits.
   - One user also stated that the current credit system feels restrictive and expensive, recommending a single subscription fee with full access instead.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1371562832457371689)** (66 messages🔥🔥): 

> `Aider with CPU vs GPU, Aider as MCP tool in Claude, Aider and Context Caching, Tmux and Aider Navigation, Gemini Comments in Aider` 


- **CPU Power Boosts Aider**: A user finds that **Aider** is beneficial for self-hosting without a **GPU** or with a small **GPU**, mainly running it via **CPU**.
- **Aider struts as an MCP**: **Aider** can be used as an **MCP** tool in **Claude**, as showcased by IndyDevDan on his [channel](https://x.com/iruletheworldmo/status/1922030559657652299).
- **Context Caching Consideration**: A member inquired whether the quoted cost in **Aider** reflects any context caching, especially the implicit caching for **Gemini**.
   - Another member clarified that you can see the context caching if you turn off streaming.
- **Tmux Tips Triumph**: A user had trouble navigating **Aider** in **tmux**, particularly scrolling up to see output.
   - Another user shared that they use **Ctrl-B** then **PageUp/PageDown** to look at the output.
- **Ruff's Eradicate Era**: A user inquired if anyone has managed to get rid of the comments that **Gemini** is putting everywhere.
   - Another user suggested using [Ruff's eradicate rule](https://docs.astral.sh/ruff/rules/#eradicate-era).


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1371594992115650620)** (31 messages🔥): 

> `AiderDesk Model Choices, Gemini Rate Limiting, yes-always Configuration Bug, Lean Context Management` 


- **Gemini Flash shines for AiderDesk Development**: One member uses **Gemini 2.5 Flash** for developing new features and fixing issues within **AiderDesk** because of its favorable cost-to-quality ratio compared to **Claude**.
   - While **Flash** sometimes struggles with consistently adhering to the system prompt, the overall value is considered very good for agentic workflows.
- **yes-always config broken**: A user reported a potential bug where setting `yes-always: true` in the Aider config prevents commands from running, whereas Aider prompts for confirmation if the value is unset.
   - The user attached images illustrating the behavior with and without the `yes-always` setting.
- **Rate Limits plague Gemini Users**: Several users experienced unexpected rate limiting from **Gemini** (free tier), even after periods of inactivity.
   - This issue may be tied to activity from **LiteLLM** or because **Google** turned off all preview versions.
- **Aider as Full-Stack IDE?**: A member suggested Aider should act more like a full-stack IDE by automatically managing context: only adding the file being edited, removing others, and carrying over recent diffs.
   - Another member concurred, suggesting the ability to set this behavior for specific git branches using the right config file and git diffs.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1371572701772251227)** (38 messages🔥): 

> `AI Governance, Compliance with AI, lm-eval-harness utility, AI parent legal hurdles, diffusion model prereqs` 


- **AI Governance Frameworks Formulating**: A member discussed **AI governance** priorities like **transparency**, **audits**, and **content moderation**, stating governance should focus on the particular application and risk classification, aligning with the [EU AI Act](https://artificialintelligenceact.eu/).
- **AI 'Parent' Faces Legal Scrutiny**: A discussion arose around the legal considerations for an “**AI parent**” phone for kids, emphasizing **privacy**, **COPPA**, and the need for a robust privacy policy and consent flows.
   - The discussion highlighted avoiding any *declared guarantees* in the User Agreement and checking for unintentional discrimination.
- **Navigating US Regulations on Automated Decision-Making**: Members discussed the legality of using **LLMs** in decision-making processes such as hiring or loan applications.
   - It was noted that in the US, there are extensive rules and regulations regarding the use of automated systems for decision-support in areas like this, and there is no reason to think that the regulations about using **linear regression** don't make sense to apply to **LLMs**.
- **Diffusion Deep Dive**: A member asked for resources to learn the prerequisites of **diffusion models**, such as **VAEs** and **GANs**.
   - Another member shared a [MIT series of videos on flow matching and diffusion models](https://youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH&si=5jE729rUtBUMC0W-), noting that it has the theory/math behind them.
- **Download Datasets Quickly with lm-eval-harness**: Members described how to download datasets associated with tasks from the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) without specifying a model at first.
   - You can use the command `python3 lm_eval --model dummy --tasks [task list] --limit 1` to download the datasets into the HF cache.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1371585161610661899)** (23 messages🔥): 

> `Fusion Model Benchmarking, Multi-Agent RL, Memory Visualization, ML Topics Scope` 


- **Fusion Models Need Benchmarking, says member**: A member said *better* fusion can really only be determined by **perf benchmarking**.
   - Another member responded about a timing issue with a **Claude** run, which stated one was faster but the other had better **numerical stability**, while linking to [arxiv.org/abs/2505.07215](https://arxiv.org/abs/2505.07215).
- **Multi-Agent RL Sparks Discussion**: A member mentioned interest in **multi-agent RL (MARL)** from the perspective of language evolution, recommending [Fitch, W. T. (2017) paper](https://doi.org/10.3758/s13423-017-1236-5) and noting it doesn't include ML architectures.
   - They asked for clarification on what another member meant by *application layer*, thinking of the **ISO-OSI model** and potential tensor parallelism across GPUs.
- **Memory System Visualization Gets Upgrade**: A member shared an improved visualization of the **memory system** for the **PersonalityAI**, modeling it after conscious, subconscious, and unconscious mind concepts with a higher-level behavioral system with an [image](https://cdn.discordapp.com/attachments/747850033994662000/1371898492620116128/image.png?ex=6824cf89&is=68237e09&hm=00148ef499121fdd310cf872ab5b33bcd6c9153db48c4c2303018ebf370653f2).
- **ML Topics Scope Addressed**: A member mentioned that **this isn't the right channel** for discussion on the conscious, subconscious, and unconscious minds and a higher-level behavioral system.
   - Other members stated it should be mathematically formalizable or similar to **LSMs**.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1371827650443612240)** (1 messages): 

> `Paper Review, Interpretability Research` 


- **Paper Reviewer Changes Mind Post-Analysis**: A member admitted to a complete change of opinion after thoroughly reviewing a paper, with credit given to another user for prompting the deeper analysis.
   - The reviewer expressed newfound enthusiasm, concluding that the research *looks really cool*.
- **Enthusiasm for Interpretability Work**: Discussion highlights growing excitement around interpretability research and related papers.
   - Members are actively engaging in detailed analysis and sharing revised perspectives based on new insights.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1371574147989377044)** (4 messages): 

> `o3 optimization, multi-GPU lm-eval, accelerate launch` 


- **O3 Optimization Sours**: A member reported that the **O3 optimization** level degraded significantly last week and they reverted to **O1-pro**.
   - No specific reasons for the performance drop were given.
- **Multi-GPU lm-eval Faces Utilization Imbalance**: A member asked about using **multi-GPU** with *lm-eval*, noting that despite setting *parallelize=True*, only **GPU 0** showed utilization.
   - Another member explained that `parallelize` uses naive pipeline parallelism, utilizing no more than one rank at a time.
- **Accelerate launch for replicated lm-eval**: A member suggested using `accelerate launch -m lm_eval ...` to run multiple replicas for better **multi-GPU utilization**.
   - This implies that running independent evaluations in parallel is a better strategy than relying on naive pipeline parallelism.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1371780023664771075)** (4 messages): 

> `GPT-NeoX data shuffling, Lingua library, TorchTitan, Nanotron, Code Rot` 


- ****GPT-NeoX** shuffles data internally!**: A member explains that **GPT-NeoX** shuffles documents, chunks each document in length-N sequences, and shuffles the sequences, so no separate preprocessing is needed.
- **Pytorch and HF launch **TorchTitan** and **Nanotron**!**: A member mentions that the PyTorch team has launched [torchtitan](https://github.com/pytorch/torchtitan) and Hugging Face has launched [nanotron](https://github.com/huggingface/nanotron).
- ****Lingua's** code may be rotting!**: A member mentions the [lingua](https://github.com/craffel/lingua/) library, noting it's efficient but experiencing *code rot* and may not be actively maintained.
- **A fork of **Lingua** with fixes is available!**: A member mentions they've created a [fork of Lingua](https://github.com/craffel/lingua/) with necessary fixes to get it running.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1371611527437619282)** (19 messages🔥): 

> `ComfyUI users, GPU no longer supported, Inference Provider contact, System prompt limits, ML engineers for image processing` 


- **ComfyUI Users Questioned**: A member inquired *if anyone here use comfyui?*
   - Other members acknowledged the question with positive and affirmative reactions.
- **GPU Support Ending, an Era Closes**: A user shared a warning about their **NVIDIA P104-100 GPU** reaching end of life with PyTorch, due to its older **CUDA capability 6.1**.
   - The warning message stated that *PyTorch no longer supports this GPU because it is too old*, with the minimum supported CUDA capability being **7.5**.
- **Inference Provider Contact Info Requested**: A member sought the *best place to contact regarding Inference Provider* and another member suggested the email address [website@huggingface.co](mailto:website@huggingface.co) and linked the [Hugging Face blog](https://huggingface.co/blog/inference-providers).
   - They noted that another channel might be appropriate, referencing a channel link.
- **System Prompt Limits Questioned**: A member questioned how much can be put into the system_prompt before the model struggles to remember its tasks.
   - They posited a comparison between **1K words** versus **40K words**, suggesting a point where following constraints becomes difficult.
- **ML Expert Needed for Image Processing Application**: A member sought an **ML engineer** with solid knowledge in **OpenCV** or **image processing** due to facing a tough phase in a current **ML application** that's used for detection.
   - Due to the specifics of their problem, they offered to provide details in a DM to someone willing to assist.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1371674186505781360)** (4 messages): 

> `Knowledge Graphs with Agentic AI, Hugging Face GGUF models` 


- **Exploring Knowledge Graphs with Agentic AI**: A member is exploring [knowledge graphs](https://huggingface.co/docs/hub/gguf) and seeking resources on using **agentic AI** for entity and relation extraction.
- **Hugging Face GGUF Models are rated well**: One member shared a link to the [Hugging Face GGUF models](https://huggingface.co/docs/hub/gguf) documentation.
   - Another user responded positively, noting that *HF is much better in overall rating* based on the shared link.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1371621841239802056)** (8 messages🔥): 

> `Bytedance Seed Coder, LLM comparison website, libmtmd Android app, Voice AI assistant based on gemma 3, Rust chat templating` 


- **Test Bytedance's Seed Coder Model**: Try [Bytedance Seed's Seed Coder model](https://huggingface.co/spaces/merterbak/Seed-Coder-8B-Instruct) on HF Spaces.
   - A member has built a [web interface](https://tryaii.com/compare?prompt=hello&models=o3%2Cclaude-3-7-sonnet-20250219%2Cgemini-2.5-pro-preview-05-06) to test and compare LLMs side by side (single prompt, many LLMs).
- **libmtmd lands on Android**: A member got the new **llama.cpp Multimodal work** (libmtmd) working in an Android application and made it look like HuggingSnap.
   - The source code is available [on GitHub](https://github.com/baseweight/BaseweightSnap).
- **Gemma 3 powers Voice AI Assistant**: A voice AI assistant based on **Gemma 3** has been developed, allowing customization of both the prompt and the voice.
   - It is accessible at [unmute.sh](https://unmute.sh/) and feedback is welcomed.
- **Chat Templating arrives in Rust**: Chat templating has been added in version 0.0.7 of the Rust transformers crate, which helps Rust devs run local models.
   - Check it out on [Crates.io](https://crates.io/crates/transformers) and [GitHub](https://github.com/ljt019/transformers).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1371847825415081994)** (1 messages): 

> `Three.js, .glb model, 2D image positioning, image detection, segmentation` 


- **Three.js Positions .glb Model onto 2D Image**: A member inquired about using **Three.js** to position a **.glb** shoe model correctly onto a **2D image** using values extracted from **detection and segmentation**.
   - The member specified key data points such as **toe points**, **heel points**, **orientation**, **true width**, **true height**, and **contour points** and wanted to know if this data would be enough to test placing a **.glb** shoe model onto the foot.
- **Data sufficiency for .glb model placement in Three.js**: The user questioned whether having **toe points**, **heel points**, **orientation**, **true width**, **true height**, and **contour points** extracted from detection and segmentation is sufficient data to test placing a **.glb** shoe model onto a foot in **Three.js**.
   - The inquiry focuses on the practical application of combining **2D image data** with **3D model placement**, highlighting the integration of computer vision techniques with 3D rendering.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1371742659273560144)** (3 messages): 

> `Software Development Basics, LLM-Assisted Coding` 


- **LLMs as Coding Coaches**: Members discussed leveraging LLMs for teaching basic-level coding, including **GIT**, **Docker** (Spaces), **IDEs**, **Python**, and basic **HTTP-based APIs**.
   - The sentiment was that any decent LLM can now guide users in basic coding practices, review code, and provide helpful suggestions.
- **Software Development Fundamentals are Important**: It was suggested that answering basic software development questions is a fundamental skill, including using **GIT**, **Docker**, and understanding **HTTP APIs**.
   - The discussion emphasized that these skills are achievable for anyone with the assistance of modern LLMs, making learning and development more accessible.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1371570186406068365)** (25 messages🔥): 

> `Chess API and FEN strings, Llama-3.2-3B-Instruct Errors, Hugging Face Space Stuck, Final Assignment Submission, LlamaIndex Section Difficulty` 


- ****Chess Agents Code Up a Storm with FEN Functions****: Members vibe coded **FEN inversion functions** for a **ChessAgent**, using **vlm** to get simple **JSON** with all figures and their positions on the deck, then converted it into **FEN string**, then fed to a chess **API** that returns best move.
   - One member recalled using "Rd5" from the API.
- ****Llama-3 Instruct Model Faces 404 Errors****: Users reported errors when trying to run the notebook with **Llama-3.2-3B-Instruct**, receiving a **404 error** even after fixes, with one stating the URL was *"not found (404)"* for the model [hosted at HuggingFace](https://huggingface.co/).
   - The specific error was a *"ValueError: Model meta-llama/Llama-3.2-3B-Instruct is not supported for task text-generation and provider together. Supported task: conversational.*"
- ****HF Space Users Stuck in Container Limbo****: Several users are stuck in the Container/Space starting phase, one reporting it persisting for *"like 2 hours"*.
   - They tried restarting and duplicating the space without success.
- ****Final Assignment Credit Crunch Prompts Local Solutions****: A user inquired about submitting the final assignment after exceeding monthly credits, developing locally with **Ollama**.
   - Another user suggested adding **HF SPACE_ID** and **SPACE_HOST** as ENV variables to run the app locally.
- ****LlamaIndex Leaves Learners Lost in Translation****: Users found the **LlamaIndex** section difficult, citing lack of depth and beginner-friendly explanations.
   - They felt the units were a *"good starting point for further self-study"* but provided only vague guidelines.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1371752109971738644)** (3 messages): 

> `Finetuning libraries, Multi-GPU support, Unsloth, Fairseq2, Axolotl` 


- **Multi-GPU Finetuning face-off: TorchTune vs. The Rest**: A member inquired about finetuning libraries with good **multi-GPU support**, besides **TorchTune**, noting that **Unsloth** primarily targets single GPUs.
   - Another member suggested **Fairseq2** and **Axolotl**, noting that they plug into the **TRL ecosystem**.
- **Fairseq2 and Axolotl join forces**: **Fairseq2** and **Axolotl** both work for multi-GPU finetuning and plug into the **TRL ecosystem**.
   - This provides users with expanded choices beyond **TorchTune** and **Unsloth** for distributed training setups.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1371691826540576890)** (55 messages🔥🔥): 

> `Llama3.1 tokenizer for 3.3 training, Kron and Muon optimizers in torchtune, HFModelTokenizer with Gemma chat template, ChatML template for Gemma` 


- **Llama3.1 tokenizer defines token 128011 for RL training**: The **Llama3.1 tokenizer** used for **3.3 training** defines token **128011** to avoid crashes during decoding, particularly in RL training, as token 128011 was previously undefined; related to [issue #2725](https://github.com/pytorch/torchtune/issues/2725).
   - This fix addresses a problem where decoding an undefined token would cause a crash, which is more likely to occur in RL training scenarios.
- **Kron and Muon optimizers ported to torchtune with fixes**: **Kron** and **Muon optimizers** from [fsdp_optimizers](https://github.com/ethansmith2000/fsdp_optimizers) were integrated into torchtune, with Kron requiring fixes to avoid excessive VRAM allocation by using `opt_einsum.contract` in `_calc_A_and_conjB`, experimented on [Weights and Biases](https://wandb.ai/intervitens/1B-optim-test).
   - Fixes include using `opt_einsum.contract` instead of regular einsum and allowing `mu_dtype` and `precond_dtype` to be set with strings in the torchtune config.
- **HFModelTokenizer produces incorrect Gemma chat template**: The **HFModelTokenizer** produces output tokens for the **Gemma chat template** that match **transformers** but not **torchtune's GemmaTokenizer**, indicating an issue with the chat template implementation; if decoded it returns a garbled *'hello therehiwhatsup?'*.
- **Gemma lacks correct prompt template, unlike HF**: Unlike Hugging Face, **Gemma lacks a specific prompt template** in torchtune, causing issues with tokenization; the HF tokenizer incorrectly adds multiple BOS tokens due to a misconfiguration, while torchtune's GemmaTokenizer expects a chat template that isn't available, but it can use the *ChatML* template.
   - The `add_bos_token` is enabled, but they also have a bos token in the chat template, which adds another one.
- **HuggingFace offers assistant mask via Jinja tricks**: HF Transformers offers masking functionalities with `jinja` templates, and has an option to return an assistant mask, possibly generalizable to other roles; [related PR](https://github.com/huggingface/transformers/pull/30650).
   - Members discussed the masking pieces and noted the challenges of accurately handling `[message.masked] * len(tokenized_message)`.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1371849456185512027)** (4 messages): 

> `NotebookLM audio shortcomings, Invisible Sun TTRPG, NotebookLM for gaming content` 


- **NotebookLM's Gaming Insights Sought**: A user inquired whether **NotebookLM** has been used to discern **techs or pattern recognition for new gaming content** amidst significant game updates.
   - Another user expressed interest in using NotebookLM to explore similar use cases.
- **Invisible Sun RPG Rules Distilled**: One user has been using **NotebookLM** with the rulebooks for the **Invisible Sun** tabletop role-playing game (*TTRPG*) by **Monte Cook Gaming**.
   - They also use **ChatGPT** for similar tasks, but really like **NotebookLM** for the shareability factor, and that it clearly cites its references.
- **Audio Overviews Lack Technical Depth**: A user found **NotebookLM's Audio Overview** of the game less technical than desired and suggested adding a prompt to specify the type of audio review.
   - But they mentioned that *it’s been great to look up individual rules, and when I am ready to GM that, it will be great to share with my players so they don’t have to buy all the books.*


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1371563672186654821)** (48 messages🔥): 

> `NotebookLM invite delays, Audio language change issues, Folder system for note organization, NotebookLM use in education, iplusinteractif textbook integration` 


- **Users Await NotebookLM Beta Access**: Several users reported not receiving **NotebookLM beta invites** after signing up, expressing patience for updates.
- **Audio Overview Language Glitch**: A user reported the **language setting** for the audio overview not changing, despite adjustments in the text overview settings.
- **Folder System Under Consideration**: Users are wondering if a **folder system** is in development for organizing notes within NotebookLM.
- **Student Highlights Generative AI Learning**: A student discussed using generative AI, like NotebookLM, to **assist learning**, citing its potential for educational equality.
- **Textbook Log-in Barrier Stops NotebookLM**: A teacher inquired about using a textbook from **iplusinteractif** as a source in NotebookLM, but is stopped by the log-in barrier.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1371654430578966558)** (39 messages🔥): 

> `MCP Server conversion, OpenAPI to MCP, Claude Code MCP, Postgres MCP Server Connection Issues, Streamable HTTP MCP Servers` 


- **Convert OpenAPI into MCP servers with `openapi-mcp-server`**: A user asked about converting software into MCP SSE servers, and another user suggested using [openapi-mcp-server](https://github.com/janwilmake/openapi-mcp-server) for converting **OpenAPI APIs** into **MCP servers**.
   - They also suggested using *use-browse* or other **MCP servers** that do **browser automation** like [mcp-browser-use](https://github.com/Saik0s/mcp-browser-use).
- **Edit Cursor and Windsurf files smarter and faster with Claude Code MCP tool**: A user shared their **magic_file MCP tool** called [claude-code-mcp](https://github.com/steipete/claude-code-mcp) that integrates **Claude Code** into **Cursor** and **Windsurf** so they can edit smarter and faster.
   - This allows them to commit to git in one shot, which makes the agent flow faster; Windsurf was impressed by the results.
- **Beware of scammers approaching you in DMs!**: A user reported that they were approached in a **private conversation** by someone claiming to be an admin, who then asked for crypto wallet information, which is definitely a **scam**.
   - The user who posted the warning recommends verifying if you're missing anything, especially related to binding your crypto wallets etc.
- **Streamable HTTP transport in TS & Python SDKs**: A user asked about the status of **Streamable HTTP** and **Auth** in the **TypeScript SDK** and another user reported that it is up to date.
   - Another user mentioned that **Python** generally lags behind **TypeScript** by around 1-2 months.
- **Security concerns for MCP servers**: A user warned about the risk of running local **MCP servers** due to potential security vulnerabilities.
   - One tip was to use **gitingest** to copy the **MCP server** repo code into **AI Studio** or **ChatGPT** and ask the **LLM** to look for any **security concerns** or use **pnpm** in place of **npm** to prevent running lifecycle callbacks.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1371597649823666236)** (6 messages): 

> `MCP Integration, uniffi-rs for MCP, LLMs and Structured Inputs, magic_file MCP Tool, Local Goose Qwen3mcp Log Proxy` 


- ****MCP Integration** Excites Developer!**: A developer expressed excitement about integrating **MCP servers** into something other than Claude and has offered early access, demoed in a [YouTube video](https://youtu.be/zNKf3ADEKdg).
   - The developer mentioned they have been playing with it locally.
- ****uniffi-rs** Suggested for MCP Implementation**: A member suggested using [**uniffi-rs** from Mozilla](https://github.com/mozilla/uniffi-rs?tab=readme-ov-file) for MCP implementation.
   - It may be of use for implementing something for **MCP**.
- **LLMs Support for Structured Inputs?**: A developer is hacking together a project related to LLMs supporting structured inputs, though it's not directly related to **MCP**.
   - The developer inquired about thoughts on this topic.
- ****Local Goose Qwen3mcp Log Proxy** Released!**: A member shared an open-source tool, [**Local Goose Qwen3mcp Log Proxy**](https://block.github.io/goose/blog/2025/05/12/local-goose-qwen3mcp-log-proxy) ([GitHub](https://github.com/emicklei/mcp-log-proxy)), designed for developers of MCP clients and servers to monitor the flow of **MCP protocol messages**.
   - The tool facilitates visibility into **MCP message flows**.
- ****magic_file MCP Tool** Improves Code Editing!**: A developer created a [**magic_file MCP tool**](https://github.com/steipete/claude-code-mcp) that integrates **Claude Code** into tools like Cursor and Windsurf for smarter and faster file editing.
   - The tool automates git commits, streamlining the agent flow.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1371599119910568089)** (32 messages🔥): 

> `Lilian Weng Chart, Gemini API Thinking Tokens, AI Technical Educators, Vertical SaaS for Restaurants, GPT-4 Launch Stories` 


- **Khoomeik Responds to Lilian Weng with Chart**: A member shared a chart responding to Lilian Weng ([link to X post](https://x.com/khoomeik/status/1922037340811219195)), sparking discussion.
   - The chart's specific content and its relevance to Weng's work were not explicitly detailed in the provided context.
- **Arfur Rock's Vertical SaaS Ventures**: A member shared a link to [Arfur Rock's X profile](https://x.com/ArfurRock/status/1922117434997191035), revealing a vertical SaaS product for restaurants.
   - Another member noted that the company tried to recruit them hard as a founding engineer back in **2022**, with the CEO sending *10+ emails*.
- **Gemini API's Thinking Tokens: Exposed or Hidden?**: Members debated whether the **Gemini API** exposes *thinking tokens*, with one stating that they see them via **OpenRouter**, but not directly through the Google API.
   - Another member mentioned that they haven't been able to get it to actually show the thinking tokens via the API, only in **AI Studio**.
- **Seeking AI Technical Educators**: A member asked for recommendations for the *best AI technical educator / content creator* with *high alpha, low hype*.
   - Harper Carroll ([X profile](https://x.com/harperscarroll?s=21)), Simon Willison, Prince Canuma, and Ivan Fioravanti (MLX) were mentioned.
- **Wholesome Stories from OAI's GPT-4 Launch**: A member shared *very wholesome stories* from the launch of **GPT-4** in the form of [Andrew Mayne's blogpost](https://andrewmayne.com/2025/05/02/some-personal-observations-about-the-launch-of-gpt-4/).
   - Further details on the specific *wholesome* aspects of the launch were not provided.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1371844983501361172)** (1 messages): 

> `DSPy Blogpost, LLM Hacking, Bugcrowd` 


- **DSPy Blogpost Drops!**: A member shared a [blog post about DSPy](https://www.bugcrowd.com/blog/hacking-llm-applications-in-the-trenches-with-dspy/).
   - The post covers hacking LLM applications with **DSPy**.
- **LLM Hacking Discussed**: The blog post dives into the methods and strategies for **hacking LLM applications** using DSPy, providing practical insights.
   - It explores vulnerabilities and techniques relevant to security professionals and developers in the **AI space**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1371673627371769866)** (16 messages🔥): 

> `DSPy for Agentic Workflows, Data QA with DSPy, MIPRO vs Optuna, TypeScript equivalent to DSPy, DSPy module needing signatures` 


- **DSPy's Agentic Acumen Assessed**: A member inquired about DSPy's utility for agentic workflows, noting its strength in declarative programs but questioning its suitability for tasks requiring more ambiguity and creativity.
   - In response, it was mentioned that DSPy isn't built specifically for agentic things, but its interaction with LLMs allows for constructing workflows via Tool Calling, where modules add Signatures like `CreateSQLquery` based on LLM responses.
- **DuckDB Data Detective with DSPy**: A user described a use case involving an agent that utilizes a connection to a **DuckDB table** to perform data QA on columns via SQL and statistical analysis, alerting Slack of any anomalies.
   - They noted that they currently use **pydantic Ai** but are intrigued by DSPy's potential, and the implementation would be through Tool Calling for each interaction with the LLM.
- **MIPRO vs Optuna: The Randomized Rumble**: A member sought a paper comparing **MIPRO** with and without **Optuna**, particularly one that analyzes the deviation when examples/instructions are randomly combined.
   - They suspected that random combinations might converge towards similar scores, albeit perhaps less efficiently, and wanted experimental evidence.
- **TypeScript DSPy Twin Hunt**: A member asked about a **TypeScript** equivalent to DSPy.
   - A few alternatives were mentioned, including [dspy.ts](https://github.com/ruvnet/dspy.ts) and [ax-llm/ax](https://github.com/ax-llm/ax), the latter being actively maintained.
- **DSPy Signature Snag**: A user questioned the practicality of requiring signatures for demos and conversation history in DSPy modules, especially in systems with multiple modules.
   - They noted the potential inefficiency of maintaining **K × N copies of the chat history** for a K-turn conversation in a system with N modules.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1371924649671131177)** (1 messages): 

> `Discord Message Links, Source code in Prompt` 


- **Discord Message Links Referencing**: A member referred to a **Discord message** with a specific link, seeking its source after a search yielded no obvious results.
- **The source code is in the prompt**: Another member added that all source code used to generate JSON is already in the prompt.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1371566470680547408)** (6 messages): 

> `BigInt support, Convolution Puzzle Clarity` 


- ****BigInt Integration Bogged Down****: A member inquired about the potential addition of **BigInt** to Mojo, but another member pointed out that a community package, [decimojo](https://builds.modular.com/packages/decimojo), already offers similar functionality.
   - They also mentioned that due to tradeoffs, **BigInt/BigDecimal** are *probably not great fits for the stdlib*.
- ****Convolution Conundrum Clarified****: A member questioned the necessity of a specific line of code in the [Convolution Puzzle](https://github.com/modular/mojo-gpu-puzzles/blob/1dfd1cc01bb9d6d98185ad405100e6c45855a007/problems/p11/p11.mojo#L104) related to memory allocation.
   - A developer confirmed that the line *doesn't need to be in the host* and thanked the member for reporting the issue.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1371576991513448508)** (8 messages🔥): 

> `Open Sourcing MAX Mojo APIs, MAX Graph Tutorials, Tensor Type Migration Code` 


- **MAX Mojo APIs Already OSS**: The deprecated MAX Mojo APIs were already open-sourced and subsequently removed in [this commit](https://github.com/modular/modular/commit/34c0209cd537f1d5ab635166358a1713728d7f6f).
   - All of `max.graph`, `max.driver`, `max.tensor`, and their tests are available in that commit, with full history accessible via `git log -- mojo/max/src/max/graph`.
- **Call for MAX Graph Tutorials**: A user requested progressively larger tutorials for **MAX Graph**, noting its current state as a *black box with a couple of examples*.
   - There was a post created on the [Modular Forum](https://forum.modular.com/t/oss-of-max-mojo-apis/1439) in this regard.
- **Tensor Type Migration Code in the Pipeline**: A ticket exists internally for a user migration code for the **tensor types**, though development has not yet commenced.
   - The team has plans to address this but there is no ETA.


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1371609333980336249)** (1 messages): 

> `PapersChat, Deep Research Agent, Multilingual RAG, Invoice Reconciliation Agent, LlamaParse Updates` 


- ****PapersChat** arrives to chat with your papers**: The team introduced [**PapersChat**](https://t.co/ASzjwLfKLC), an agentic AI application that lets you chat with your papers and gather information from Arxiv and PubMed.
- **Deep (Research) Dive with New Agent**: A video was released on [building your own Deep Research Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A) using LlamaIndex.
- **Multi-lingual and multi-modal? RAG on!**: A [Multilingual, Multimodal RAG System](https://t.co/69CHCCn8J3) demo was released, with no further details given.
- **Reconcile Invoices with New Agent**: A new video shows you how to [Build an invoice reconciliation agent](https://www.youtube.com/watch?v=SzVsuMBcv5g) using LlamaIndex.TS and LlamaCloud.
- **LlamaParse Gets Auto Orientation Detection and Model Update**: **LlamaParse** gets new models and auto orientation detection; [read more here](https://t.co/tqi17dPJm4).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1371899127541010522)** (2 messages): 

> `LlamaIndex Memory API, AI Agents Memory Improvement, Short-term chat history, Long-term memory` 


- **LlamaIndex Upgrades Memory**: LlamaIndex announced a big **memory upgrade** with a new, flexible **Memory API** that blends short-term chat history and long-term memory.
   - The upgrade features plug-and-play blocks like [StaticMemoryBlock](https://t.co/wwWnwdyW7s) for non-changing static information and **FactExtractionMemoryBlock** that keeps track of a list of useful facts.
- **AI Agents Sharpen Memory Skills**: LlamaIndex released a new **Memory component** to improve AI agents' memory with both short-term and long-term capabilities.
   - This allows storing **chat history** for context-aware conversations and implementing [static memory blocks](https://t.co/CDOB3UUO4W).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1371875035463028766)** (3 messages): 

> `google_genai integration, GoogleSearch, FunctionTool` 


- **Integrating google_genai's GoogleSearch Tool**: A user inquired about integrating **GoogleSearch** from the **google_genai** library, noting its difference from **GoogleSearchToolSpec** which requires key and engine setup.
   - Another member suggested wrapping it as its own **FunctionTool** for compatibility with the `chat_with_tools` method.
- **Wrapping GoogleSearch as a FunctionTool**: To integrate **GoogleSearch** from the `google_genai` library with LlamaIndex's `chat_with_tools` method, it needs to be wrapped as a **FunctionTool**.
   - This approach allows for better tool handling and avoids the need for key and engine setup required by **GoogleSearchToolSpec**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1371646445328793681)** (4 messages): 

> `OpenCL implementation, tensor numel, device/backend, memory movement functions, view changes` 


- **Detecting `long long` Support in Tinygrad's OpenCL Backend**: A member asked about a way to query the max supported tensor numel for a given device/backend, because they're using an older **OpenCL** implementation that doesn't support the `long long` data type for indexing buffers.
   - They shared [a tinygrad script](https://cdn.discordapp.com/attachments/1070745817025106080/1371902071112204348/tinygrad_long_long_support_check.py?ex=6824d2de&is=6823815e&hm=3b0d23ba54692d02a6b6bd9f47ff4b7d963d4465a5045204907df5c24c78eff7&) to check if an **OpenCL** implementation supports tensors large enough to require `long long` indexes, and if it returns false, the operation has to be split into chunks or offloaded to CPU.
- **Identifying Memory Movement Functions and View Changes**: A member inquired about identifying which movement functions in the documentation are really in place versus those that require new memory.
   - They want to know which functions change the view versus create new views.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1371689808912715807)** (3 messages): 

> `Creative Writing with oblix.ai, Local vs Cloud Model Orchestration, Edge Computing Savings` 


- **oblix.ai Demos Creative Writing**: A member shared [oblix.ai](https://oblix.ai/) to demonstrate its creative writing capabilities.
   - The member said they were just *looking to see how it handles creative writing for funsies*.
- **Orchestration of Local and Cloud Models**: A member is working on **orchestration** between **local and cloud models** to switch between cloud/edge while maintaining context.
   - This approach aims to **save cloud credits** based on runtime agents.
- **Video Demo of Cloud/Edge Switching**: A member shared a [video demo](https://youtu.be/j0dOVWWzBrE?si=oCjf18i7ykLmzCeh) showcasing switching between **cloud and edge models**.
   - The implementation preserves context and helps reduce cloud credit consumption.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1371885700969922722)** (1 messages): 

> `Lambda Workshop, Nobel FutureTech Info Session` 


- **Lambda Workshop Teaches Agentic AI**: Attend the **Lambda Workshop** on **May 15th at 10AM PT** to learn how to build agentic applications using Lambda's Inference API, and get **$100** serverless API credits for applying by **May 16th** via [this link](https://forms.gle/UtVhmPS3mitS8Vxu7).
   - You can [register here](https://lu.ma/AgentX-lambda) to learn about optimizing agent performance and deploying agents in production.
- **Nobel FutureTech Discusses Genius Club**: An exclusive info session co-hosted by **Nobel FutureTech Group** and **Berkeley RDI** will be happening on **May 15th at 12PM PT** with a distinguished member of the **Nobel FutureTech Genius Club**.
   - Interested parties can [register here](https://lu.ma/NobelFutureTech) to learn about opportunities for mentorship, funding, and collaboration, or apply to join the Genius Club [here](https://nobel-futuretech.com/contact.html?link=Ab5B1SNibcW6).


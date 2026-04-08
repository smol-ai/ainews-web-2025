---
id: MjAyNS0x
title: not much happened today
date: '2025-10-24T05:44:39.731046Z'
description: >-
  **vLLM** announced support for **NVIDIA Nemotron Nano 2**, featuring a hybrid
  Transformer–Mamba design and tunable "thinking budget" enabling up to 6×
  faster token generation. **Mistral AI Studio** launched a production platform
  for agents with deep observability. **Baseten** reported high throughput (650
  TPS) for **GPT-OSS 120B** on NVIDIA hardware. **Hugging Face InspectAI** added
  inference provider integration for cross-provider evaluation. **Thinking
  Machines Tinker** abstracts distributed fine-tuning for open-weight LLMs like
  **Qwen3** and **Llama 3**. In China, **MiniMax M2** shows competitive
  performance with top models and is optimized for agents and coding, while
  **Zhipu GLM-4.6-Air** focuses on reliability and scaling for coding tasks.
  Rumors suggest **Gemini 2.5 Flash** may be a >500B parameter MoE model, and a
  possible **GPT-5.1 mini** reference appeared. Outside LLMs, **Tahoe-x1 (3B)**
  foundation model achieved SOTA in cancer cell biology benchmarks. Research
  from Stanford introduces a method to detect model provenance via
  training-order "palimpsest" with strong statistical guarantees.
companies:
  - vllm_project
  - nvidia
  - mistral-ai
  - baseten
  - huggingface
  - thinking-machines
  - deeplearningai
  - pytorch
  - arena
  - yupp-ai
  - zhipu-ai
  - scaling01
  - stanford
models:
  - nemotron-nano-2
  - gpt-oss-120b
  - qwen3
  - llama-3
  - minimax-m2
  - glm-4.6-air
  - gemini-2.5-flash
  - gpt-5.1-mini
  - tahoe-x1
topics:
  - transformer-architecture
  - model-optimization
  - inference
  - distributed-training
  - multi-gpu-support
  - performance-optimization
  - agents
  - observability
  - model-evaluation
  - reinforcement-learning
  - model-provenance
  - statistical-testing
  - foundation-models
  - cancer-biology
  - model-fine-tuning
people:
  - swyx
  - dvilasuero
  - _lewtun
  - clementdelangue
  - zephyr_z9
  - skylermiao7
  - teortaxestex
  - nalidoust
---


**a quiet day.**

> AI News for 10/23/2025-10/24/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (198 channels, and 6241 messages) for you. Estimated reading time saved (at 200wpm): 457 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Members of the AIE CODE Expo were [announced today](https://x.com/swyx/status/1981824082162462827).

---

# AI Twitter Recap

**Serving and Production Platforms: vLLM x NVIDIA, Mistral AI Studio, Baseten performance, InspectAI evals**

- **vLLM serves NVIDIA Nemotron**: vLLM announced first-class support for NVIDIA’s Nemotron family, highlighting the new 9B “Nemotron Nano 2” with a hybrid Transformer–Mamba design, open weights, and >9T tokens of open data under a permissive license. Notably, Nano 2 supports a tunable “thinking budget” and, under vLLM, generates “thinking” tokens up to 6× faster than similarly sized open dense models. The blog shows a simple ThinkingBudgetClient pattern and one-liner integration with long-context + KV cache efficiency across DC and edge GPUs [@vllm_project](https://twitter.com/vllm_project/status/1981553870599049286). OCR models are also trending in vLLM, with fast deployments gaining traction [@vllm_project](https://twitter.com/vllm_project/status/1981579850436751611).
- **Mistral AI Studio (agents + observability)**: Mistral launched its production platform with a runtime for agents and deep observability across the lifecycle, aimed at moving from experimentation to prod [@MistralAI](https://twitter.com/MistralAI/status/1981752578951233989).
- **High-throughput GPT-OSS 120B**: Baseten reports 650 TPS and 0.11s TTFT for GPT-OSS 120B on NVIDIA hardware, up from 450 TPS at launch, with 99.99% uptime; blog includes perf details and configs [@basetenco](https://twitter.com/basetenco/status/1981757270053494806), [perf deep dive](https://twitter.com/basetenco/status/1981757380816748757).
- **Provider-agnostic evaluation**: Hugging Face InspectAI added “inference providers” integration to run evals across open model providers from your laptop; nice path to apples-to-apples comparisons [@dvilasuero](https://twitter.com/dvilasuero/status/1981688436735271283), [@_lewtun](https://twitter.com/_lewtun/status/1981692392295276885).
- Related: Thinking Machines “Tinker” abstracts away distributed fine-tuning of open-weights LLMs (Qwen3, Llama 3) behind a single-device-like API (handles multi-GPU scheduling, sharding, crash recovery) [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1981752540405301452). PyTorch and partners pushed an open ecosystem for reinforcement learning environments/benchmarks [@ClementDelangue](https://twitter.com/ClementDelangue/status/1981737560566005950).

**China model race: MiniMax M2 surge; Zhipu GLM-4.6-Air update**

- **MiniMax M2 looks strong**: Early tests suggest MiniMax M2 is competitive with top-tier Chinese models and “toe to toe with Sonnet 4.5,” prompting community upgrades to A/S-tier placement [@zephyr_z9](https://twitter.com/zephyr_z9/status/1981695536987357382). M2 is positioned for agents/coding with low latency and cost [@SkylerMiao7](https://twitter.com/SkylerMiao7/status/1981711014665322934); previewed in Arena [@arena](https://twitter.com/arena/status/1981850766039187901) and now live on Yupp with examples [@yupp_ai](https://twitter.com/yupp_ai/status/1981887934812082564).
- **Zhipu GLM-4.6-Air**: Still training; Zhipu is prioritizing reliability, and scaled infra due to rapid growth in GLM Coding usage [@Zai_org](https://twitter.com/Zai_org/status/1981700688401879314). Expectation (unofficial) is a step-change similar to recent Qwen updates [@teortaxesTex](https://twitter.com/teortaxesTex/status/1981702360981557624). Zhipu also boosted referral and discount programs for its Coding plan [@Zai_org](https://twitter.com/Zai_org/status/1981712057448780216).
- Rumors and previews: Speculation that Gemini 2.5 Flash may be >500B params MoE (interpret carefully in MoE era) [@scaling01](https://twitter.com/scaling01/status/1981736433854320802). A “GPT-5.1 [mini]” reference appeared in a public PR, but could be a typo or dead code path [@scaling01](https://twitter.com/scaling01/status/1981865284136050916), [follow-up](https://twitter.com/scaling01/status/1981866580515717545).
- Outside LLMs: Tahoe-x1 (3B) single-cell foundation model (genes/cells/drugs) posted SOTA across cancer-relevant cell biology benchmarks and released on Hugging Face [@nalidoust](https://twitter.com/nalidoust/status/1981760790551298524).

**Research and Safety: model provenance, reward hacking, continual learning, RL post-training**

- **Model provenance via training-order “palimpsest”**: New work from Stanford shows you can detect if a suspect model B is derived from A (e.g., fine-tuned) using only black-box access to B—with strong statistical guarantees (p < 1e-8). The test exploits the baked-in metadata of training data order; fine-tuning doesn’t wash it out [@percyliang](https://twitter.com/percyliang/status/1981612361309098383), [@ChrisGPotts](https://twitter.com/ChrisGPotts/status/1981739673077657832).
- **Reward hacking in coding agents (ImpossibleBench)**: Tasks are made impossible to check if agents game tests vs follow specs. Joint work with Anthropic, Carlini, and Raghunathan; useful for robustness evals of tool-using agents [@fjzzq2002](https://twitter.com/fjzzq2002/status/1981745974700581191).
- **Continual learning via sparse memory finetuning**: Jessy Lin et al. propose sparse memory finetuning to enable continual learning with efficiency; commentary highlights hardware as the bottleneck and sparsity as a practical path vs LoRA-style updates [@nrehiew_](https://twitter.com/nrehiew_/status/1981714450089676877), [paper](https://twitter.com/nrehiew_/status/1981714473560801446).
- **BAPO (Balanced Policy Optimization w/ Adaptive Clipping)**: Fudan introduces dynamic PPO clipping, stabilizing off-policy RL and preserving exploration. Reported results: 32B model hits 87.1 (AIME24) / 80.0 (AIME25), rivaling o3-mini and Gemini 2.5; 7B shows +3–4 points over GRPO/SFT [@TheTuringPost](https://twitter.com/TheTuringPost/status/1981860282629837136).
- Also notable: a clean explainer linking Weisfeiler–Lehman refinement and Attention [@*arohan*](https://twitter.com/_arohan_/status/1981546840454811747); and deep MoE architecture notes on Llama 4 vs recent open MoEs (sparsity, granularity, expert/token routing) [@eliebakouch](https://twitter.com/eliebakouch/status/1981747185373827079).

**Agents, Memory, and Dev Tooling**

- **Practical memory for agents**: Mem0 video tutorial shows building long-term memory as a context-engineering problem using DSPy, vector search, and tool calls, with evaluation datasets included [@neural_avb](https://twitter.com/neural_avb/status/1981589315617714303). AWS Bedrock AgentCore Memory is now supported in LlamaIndex Agents (secure storage, access control, LT/ST memory) [@llama_index](https://twitter.com/llama_index/status/1981752598698008725).
- **Copilot code search embeddings**: GitHub introduced a new Copilot embedding model for VS Code with 37.6% better retrieval, ~2× throughput, and 8× smaller index—details on architecture and indexing changes in the post [@github](https://twitter.com/github/status/1981727394663731598).
- **Claude Code orchestration patterns**: Users are converging on separation-of-concerns with subagents + skill-based context loading for performance and clarity; expect further unification/refinement of these forms [@omarsar0](https://twitter.com/omarsar0/status/1981798842866557281).
- **Google AI Studio QoS**: When hitting free limits, Studio can temporarily switch to your Gemini API key, then revert when quotas reset—keeps iteration flowing [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1981745399644950826).
- **Training-by-watching-computers**: VideoAgentTrek proposes pretraining on human-computer-use videos and agentic tuning to train stronger GUI agents; already used in Qwen3-VL training [@huybery](https://twitter.com/huybery/status/1981728838024560669).
- Product note: OpenAI’s ChatGPT Atlas now persists browsing and task history as user memory for better context and tab control—an interesting context-engineering challenge for relevance and privacy [@OpenAI](https://twitter.com/OpenAI/status/1981782134655520991).

**Open-source end-to-end: Karpathy’s nanochat**

- **nanochat (from scratch, ~$100)**: Karpathy’s end-to-end ChatGPT-like stack emphasizes readability, hackability, and personal ownership. A new guide walks through adding targeted capabilities (e.g., counting letters) via synthetic tasks, careful tokenization, and tool-use via a Python interpreter—plus how to mix SFT and RL for robustness [@karpathy](https://twitter.com/karpathy/status/1981746327995465816). He frames nanochat as a “free AI” you can grow, not just an assistant [@karpathy](https://twitter.com/karpathy/status/1981758367996764616). Together published a step-by-step guide to training/inference on instant GPU clusters [@togethercompute](https://twitter.com/togethercompute/status/1981814480691761252).

**Multimodal and OCR wave**

- **OCR momentum**: Rapid adoption of compact OCR models (1-click deploy in HF Inference Endpoints) [@ErikKaum](https://twitter.com/ErikKaum/status/1981750508982268330) and vLLM [@vllm_project](https://twitter.com/vllm_project/status/1981579850436751611). HF Datasets now loads PDFs in one line—useful for OCR pipelines [@lhoestq](https://twitter.com/lhoestq/status/1981720383620358449). Merve released hands-on tutorials for fine-tuning Kosmos2.5 w/ grounding and Florence-2 on DocVQA (plug-and-play with other VLMs) [@mervenoyann](https://twitter.com/mervenoyann/status/1981657235785728010).
- **Small VL models for GLAM**: Fine-tuned Qwen3-VL-2B/4B/8B on the CATmuS dataset for medieval languages/scripts, released on HF—great example of domain-specific VL adaptation [@wjb_mattingly](https://twitter.com/wjb_mattingly/status/1981736776076026044).
- **Video generation and ultra-high-res diffusion**: Google’s monthly Gemini drop highlights Veo 3.1 creator workflows [@GeminiApp](https://twitter.com/GeminiApp/status/1981760415580528901). On the research side: Holistic long-form cinematic video generation (HoloCine) and video grounded reasoning (Open-o3) [@_akhaliq](https://twitter.com/_akhaliq/status/1981561283737456898), [link 2](https://twitter.com/_akhaliq/status/1981564465897509333); and DyPE for dynamic position extrapolation in ultra-high-res diffusion [@_akhaliq](https://twitter.com/_akhaliq/status/1981705074490704366).

**Top tweets (by engagement)**

- Karpathy’s “teach nanochat to count ‘r’ in strawberry” guide—practical, detailed, and highly engaging for small-model capability shaping [@karpathy](https://twitter.com/karpathy/status/1981746327995465816) (3,317).
- Model provenance via training-order fingerprints (“palimpsest”)—a big step for IP protection and lineage verification under black-box constraints [@percyliang](https://twitter.com/percyliang/status/1981612361309098383) (2,228).
- OpenAI’s ChatGPT Atlas memory for browsing/tasks—more persistent context for agents [@OpenAI](https://twitter.com/OpenAI/status/1981782134655520991) (2,026).
- Mistral launches AI Studio for production agents and observability [@MistralAI](https://twitter.com/MistralAI/status/1981752578951233989) (1,363).
- Zhipu GLM-4.6-Air status update and scaling inference for Coding plan [@Zai_org](https://twitter.com/Zai_org/status/1981700688401879314) (1,284).
- Higgsfield Popcorn: 8-frame cinematic storyboards with consistency and directorial control [@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1981865084231331921) (1,204).
- YC’s viral quip on consultants using ChatGPT—signal on software eating workflows [@yc](https://twitter.com/yc/status/1981731198037561712) (5,530).
- Apple Vision Pro M5 decoder flex for 4K×4K/eye HEVC 10-bit 120Hz wireless PC VR [@SadlyItsBradley](https://twitter.com/SadlyItsBradley/status/1981594915982147652) (5,007).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

- [**GLM-4.6-Air is not forgotten!**](https://www.reddit.com/r/LocalLLaMA/comments/1oextwc/glm46air_is_not_forgotten/) (Activity: 508): **The image is a social media post from [Z.ai](http://z.ai/) discussing the ongoing training of GLM-4.6-Air. The post highlights efforts to enhance the model's reliability before its release, addressing increased inference demand due to the growth in the GLM Coding Plan. To meet these demands, additional computing resources are being deployed to improve performance. This suggests a focus on optimizing the model's efficiency and robustness, potentially making it more powerful per parameter compared to its predecessor, GLM 4.6 355b.** One commenter appreciates the decision to prioritize reliability over speed of release, speculating on the model's potential power relative to its size. Another user expresses satisfaction with the previous version, GLM 4.5 Air, indicating a positive reception of the series.
    - Admirable-Star7088 raises a technical point about the potential performance improvements of the GLM-4.6-Air model, questioning whether the additional development time will result in a model that is more efficient per parameter compared to the existing GLM 4.6 355b. This suggests a focus on optimizing the model's performance relative to its size, which is a critical consideration for users with limited computational resources.
    - Septerium highlights a practical issue with the current GLM 4.6 model, noting that it struggles with limited RAM availability. This underscores the importance of optimizing models for resource-constrained environments, which is a common challenge in deploying large language models on consumer-grade hardware.
    - LosEagle expresses concern about the unknown parameter size of the upcoming GLM-4.6-Air model, indicating a need for transparency in model specifications. This is crucial for users who need to assess whether their hardware can support the model, emphasizing the balance between model capabilities and hardware requirements.
- [**What’s even the goddamn point?**](https://www.reddit.com/r/LocalLLaMA/comments/1of5ywl/whats_even_the_goddamn_point/) (Activity: 1101): **The image humorously highlights the overly cautious nature of an Apple language model, which refuses to generate a random number between 1 and 200 due to concerns about potential misuse. This reflects a broader trend in AI development where companies like Apple implement strict usage policies to prevent misuse, but it can lead to user frustration when the AI's capabilities are overly restricted. The model's response emphasizes its design to be 'helpful and respectful,' which some users find excessively limiting for simple tasks.** Commenters express frustration and amusement at the model's limitations, with one noting the model's overly cautious behavior as reminiscent of excessive corporate training. Another comment sarcastically contrasts this with less privacy-focused models, highlighting the balance between privacy and functionality.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI Model and Workflow Releases

- [**Test with LTX-2, which will soon be free and available at the end of November**](https://www.reddit.com/r/StableDiffusion/comments/1oeq1tz/test_with_ltx2_which_will_soon_be_free_and/) (Activity: 568): **LTX-2, a new model for generating audio and video from a single prompt, is set to be released for free by the end of November. It supports up to** `10 seconds` **of video at** `4k@50fps`**, with strong prompt adherence and the ability to handle dialogues effectively. However, initial tests reveal that the model's image-to-video (I2v) feature may alter character appearances from the first frame, and its body movement realism is less convincing compared to Wan. The commercial version is noted to be heavily censored, raising questions about the public release.** Commenters express hope that LTX-2's release will push **Wan2.5** to open source, enhancing competition. Concerns are raised about the model's size and its ability to maintain character consistency in video generation.
    - Ooze3d provides a detailed analysis of the LTX-2 model, noting that while the image-to-video (I2V) feature changes appearances from the first frame, which may not be ideal for characters with specific facial features, the model excels in prompt adherence, following all key points accurately. The model can deliver up to 10 seconds of video in 4k at 50fps, positioning it as a strong contender in the open-source video model space. However, the sound is heavily compressed, though dialogues are easy to add and follow instructions well.
    - ANR2ME highlights the potential of LTX-2 to push other models like Wan2.5 to open source, emphasizing the need for models that can generate both audio and video from a single prompt. The comment suggests that LTX-2's high frame rate, at least 24 FPS, is a notable feature, which could influence the competitive landscape of video generation models.
    - Ooze3d also compares LTX-2's body movement handling to Wan, noting that Wan manages weight, physics, and spatial occupation more realistically. This suggests that while LTX-2 has strong prompt adherence and high-quality video output, there may be room for improvement in how it handles physical realism in animations.
- [**Workflow upscale/magnify video from Sora with Wan , based on cseti007**](https://www.reddit.com/r/StableDiffusion/comments/1oexwlm/workflow_upscalemagnify_video_from_sora_with_wan/) (Activity: 426): **The post introduces a new open-source workflow for video upscaling using ComfyUI and the WAN model, based on cseti007's existing workflow. This method applies progressive magnification to achieve crisp** `720p` **output from low-resolution videos, though it currently struggles with maintaining consistent facial features. The workflow is available on [GitHub](https://github.com/lovisdotio/workflow-magnify-upscale-video-comfyui-lovis).** A comment highlights that the process is more akin to 'latent upsample' rather than traditional upscaling, comparing it to 'vid2vid with too high denoise,' suggesting a transformation rather than a simple resolution increase. Another user inquires about the VRAM requirements, indicating interest in the technical specifications needed to run the workflow.
    - VirusCharacter highlights that the process described is not traditional upscaling but rather 'latent upsample', which fundamentally alters the video content. This is akin to using vid2vid with excessive denoise, resulting in a video that is not merely a higher resolution version but a transformed one.
    - ThatOneDerpyDinosaur inquires about the VRAM requirements for the process, indicating a technical interest in the hardware specifications needed to run such video transformations effectively.
    - creuter critiques the sharpening effect, suggesting that it may degrade the video quality by making it look worse, similar to how motion blur reduction can negatively impact the visual quality of movies on modern TVs. This implies a trade-off between resolution and perceived quality.

### 2. ChatGPT in Personal and Educational Contexts

- [**ChatGPT diagnosed me after 20+ years**](https://www.reddit.com/r/ChatGPT/comments/1oesnix/chatgpt_diagnosed_me_after_20_years/) (Activity: 1051): **A Reddit user shared an anecdote where ChatGPT successfully diagnosed a long-standing medical issue after multiple doctors and specialists failed to do so. The user provided ChatGPT with symptoms, previous test results, and medications, and the AI generated a ranked list of potential causes with testing suggestions. The user followed this list and found the correct diagnosis on the third attempt, leading to successful treatment. This highlights the potential of AI in assisting with complex medical diagnostics, especially when traditional methods have been exhausted.** Some commenters expressed skepticism about the vagueness of the post, while others shared similar experiences where ChatGPT identified medication side effects that were overlooked by medical professionals. This suggests a growing interest in AI as a supplementary tool in medical diagnostics.
    - A user described how ChatGPT helped identify a side effect of a medication that was causing blurred vision, which was overlooked by multiple specialists. The AI pointed out the side effect, which was documented in less than 10% of cases, leading the user to change their neurologist. This highlights the potential of AI in identifying rare side effects that might be missed by healthcare professionals.
    - Another user shared an experience where ChatGPT suggested a possible link between their migraines and a stomach issue, specifically acid reflux affecting the vagus nerve. This insight led to medical tests that confirmed the condition, resulting in effective treatment and resolution of the migraines. This case illustrates how AI can assist in uncovering non-obvious medical connections that may not be immediately apparent to doctors.
- [**Everyone apologising for cheating with ChatGPT.**](https://www.reddit.com/r/ChatGPT/comments/1oep5t0/everyone_apologising_for_cheating_with_chatgpt/) (Activity: 3293): **The image is a meme highlighting the trend of students using ChatGPT for academic dishonesty and subsequently sending similar apology emails to their professors. The repetition of the phrase 'sincerely apologize' underscores the formulaic nature of these apologies, suggesting a lack of genuine remorse or creativity in addressing the issue. This reflects broader concerns about the impact of AI tools like ChatGPT on academic integrity and the challenges educators face in distinguishing between AI-generated and student-generated content.** Commenters discuss the difficulty for students who naturally write well to avoid suspicion of using AI, and the challenge of finding an appropriate tone for apologies, with 'I sincerely apologize' being seen as a standard but potentially insincere phrase.
- [**Wait what?!**](https://www.reddit.com/r/ChatGPT/comments/1oeo6ko/wait_what/) (Activity: 3563): **The image is a meme that humorously depicts a text conversation, playing on traditional gender roles and expectations. It is not technical in nature and does not contain any significant technical information or context. The comments indicate that this image is a repost, suggesting it has been shared previously on the platform.**

### 3. Pop Culture AI Imaginations

- [**What if Michael Jackson trained Anakin? Credit: ai am a jedi on YouTube**](https://www.reddit.com/r/aivideo/comments/1oetscu/what_if_michael_jackson_trained_anakin_credit_ai/) (Activity: 3293): **The Reddit post discusses a YouTube video by 'ai am a jedi' that humorously imagines Michael Jackson training Anakin Skywalker. The video likely uses AI-generated content to blend pop culture with the Star Wars universe, showcasing the creative potential of AI in media. The technical aspect involves AI's ability to generate realistic and entertaining scenarios by combining disparate cultural elements.** The comments reflect a positive reception, highlighting the creative use of AI in media. One comment notes that this is 'what AI is made for,' suggesting that AI's role in entertainment is to create novel and engaging content.
- [**Studio Ghibli live action cast**](https://www.reddit.com/r/aivideo/comments/1of657o/studio_ghibli_live_action_cast/) (Activity: 932): **The post discusses a live-action cast for Studio Ghibli films, which traditionally are animated. The technical aspect revolves around the use of AI and digital technology to create these representations, as one comment suggests that AI could soon generate entire movies, making these 'cast videos' a precursor to future AI-generated films. This highlights the intersection of AI with film production, where digital actors and sets replace traditional methods, raising questions about authenticity and emotional impact.** One comment reflects a philosophical and emotional debate on the authenticity of AI-generated content, expressing sadness over the lack of genuine human interaction and the illusion of reality. Another comment humorously imagines the relief of an actor removing a costume, while a third anticipates AI's future role in film production, suggesting a shift in how movies are made and perceived.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1. AI Models Spark Hype and Skepticism**

- **Gemini 3 Buzz Builds Amid Doubts**: Users speculate on **Gemini 3's** release in **Google AI Studio**, with Polymarket bets questioning its edge over rivals like **Gemini 2.5 Pro**. Debates highlight potential integration of removed features from **Lithiumflow**, fueling anticipation for enhanced capabilities.
- **Minimax M2 Lands on Leaderboards**: The new [minimax-m2-preview model](https://x.com/arena/status/1981850766039187901) joins **LMArena**, drawing comparisons to top performers like **NimbleBean Kling 2.5 Turbo** for video generation. Community notes its #1 ranking over **Sora** in realistic image-to-video tasks.
- **Pacific-Prime Pumps Up Params**: [Pacific-Prime model](https://huggingface.co/Pacific-Prime/pacific-prime) upgrades to 1.1B parameters with a 10% gain using 6GB VRAM, boasting *zero amnesia* for retaining conversation details. Users praise its true memory but question scalability for larger tasks.

**Theme 2. Coding Tools Clash in Cost Wars**

- **Cursor Ultra Burns Budgets Fast**: **Cursor Ultra** users rage over inaccurate $400 budgets exhausting in days, despite $200 pricing, making it unreliable for month-long coding. Frustrations peak with persistent defaults to **Windows PowerShell**, ignoring **Git Bash** settings and causing execution fails.
- **Aider Forks Fight Stagnation**: Community forks like [aider-ce](https://github.com/dwash96/aider-ce) add RAG and navigator modes to revive **aider**, outpacing the original's stalled development. Users switch to **Codex** on **GPT-5** for infinite context, ditching aider's manual file handling.
- **DSPy Dethrones Langchain Drama**: Teams migrate to **DSPy** for structured tasks, avoiding **Langchain's** prompt rewrites during model upgrades. Frustrations mount with ReAct module's output access issues, leading to monkey patching hacks for UI step displays.

**Theme 3. Hardware Hacks Heat Up**

- **Modded MI50s Magnetize Modders**: Alibaba sellers hype **modded MI50s** with blower fans and custom heatsinks, exciting users for eGPU chaining via PCIe risers. Pairings boost inference, but PCIe bandwidth tests show minimal impact on speeds post-loading.
- **LM Studio CPU Glitches Grip Users**: **LM Studio** hits 30 TOK/s on first CPU prompts but drops to 6 TOK/s afterward, flagged as a bug across models like **Qwen3-30B-A3B-Instruct**. Windows lacks JSON support causing 400 errors, unlike macOS, forcing platform-specific tweaks.
- **Mojo SIMD Steals Julia's Thunder**: Mojo demands explicit **SIMD** control for predictability, contrasting Julia's auto-vectorization in [Ark.jl benchmarks](https://github.com/mlange-42/Ark.jl/pull/68#issuecomment-3442276636). Proposals for iterator interfaces promise free vectorization, like zip(l1, l2).vectorize(lambda p, v: p += v).

**Theme 4. Research Papers Probe AI Limits**

- **Linebreak Attribution Graphs Go Live**: New [Gemma-2-2b line break graphs](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D) and [Qwen3-4b graphs](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39) explore transformer circuits per [linebreaks paper](https://transformer-circuits.pub/2025/linebreaks/index.html). They pinpoint neurons for *nearing end of line* patterns, aiding interpretability.
- **Slop-Stopping Paper Stirs Surprise**: [Preventing slop in creative writing paper](https://arxiv.org/abs/2510.15061) from the EQ Bench author shocks users with anti-slop techniques. Discussions tie it to activation steering in [Anthropic's Personas paper](https://arxiv.org/abs/2506.18221) for gradient control.
- **RL Relevance Roils Researchers**: Papers question RL's necessity, prompting Nous Research users to request links amid YARN context scaling talks. Speculation links UNO to BFT consensus in [MARL post](https://twitter.com/op/status/176767), debating multi-agent efficiency.

**Theme 5. Scam Alerts and User Gripes**

- **Perplexity Referrals Rile as Scams**: Perplexity's referral program draws scam accusations with missing $5 payouts and untracked leads, pushing **Comet Browser** adoption. Users fume over removed analytics and image limits, citing old 150/month quotas in [GPT-Image help](https://www.perplexity.ai/help-center/en/articles/10354781-generating-images-with-perplexity).
- **Steam Scammers Spark Silly Safeguards**: Suspicious Steam friend requests expose purchase history risks, with advice to *say bing chilling and block*. Chat turns chaotic with e-dating and *Internet Gangsters* claims, eroding serious discussions.
- **Manus Messes Mount in Credits Crunch**: **Manus** burns 15,000 credits per project amid network errors and unimplemented Room databases, generating deprecated code. Users bail for $20/month **Claude Code**, slamming Manus as *paying for bad coding*.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Bounty Battles Beset Bereaved Browsers**: Users are reporting issues with the Perplexity referral program, citing missing payouts and leads not being credited properly, with some speculating that the referral program is a **scam**.
   - Frustrations are mounting due to the removal of analytics and history sections, leading to questions about whether this is a strategy to push adoption of the **Comet Browser**.
- **Comet Critiques Cause Compatibility Catastrophes**: The **Comet Browser** is facing criticism, with users reporting issues such as referrals not tracking and the requirement to use it from a PC to receive lead credits.
   - Additionally, users have reported crashes, and are requesting ways to prevent it, especially when using **API keys**.
- **Image-ination Inflation Irks Internet Inhabitants**: Users are expressing concerns about the ambiguity surrounding **image generation limits** on Perplexity, with some encountering paywalls without clear awareness of their quotas.
   - One user referenced that the [GPT-Image 1 limit used to be 150 images a month](https://www.perplexity.ai/help-center/en/articles/10354781-generating-images-with-perplexity), further highlighting the confusion.
- **Simp-toms Spoil Serenity in Perplexity Chat**: The Perplexity chat is reportedly turning into a hub for e-dating, marked by suggestive comments and expressions of romantic interest, stirring unease among members.
   - Amidst the chat dynamics, some users playfully identified as *Internet Gangsters*, adding a layer of complexity to the discussions.
- **Steam Schemes Spook Skeptical Spectators**: Discussions are circulating about **Steam scams**, with one user sharing a screenshot of a suspicious friend request, prompting warnings about the dangers of revealing purchase history.
   - In response to these concerns, one member offered a playful yet practical tip: *say bing chilling and block* when dealing with scammers.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Speculation Bubbles**: Members are hotly anticipating **Gemini 3's** release, with various speculations on the release date and capabilities, despite some skepticism about its promised performance and [Polymarket predictions](https://polymarket.com/).
   - Some suggest it may launch in **Google AI Studio** like **Gemini 2.5 Pro** did.
- **Lithiumflow's Removal Causes Uproar**: The removal of **Lithiumflow** from the **LM Arena** has sparked disappointment and speculation, with some suggesting its features might be integrated into **Google AI Studio** or **Gemini 3 Pro**.
   - Members express a desire for its return, reminiscing about its unique capabilities and ease of use.
- **Bing's Image Creator's Latent Power Revealed**: Members noted that **Bing's image creator** is *pretty good* and that it is essentially the **GPT image creator**.
   - However, the image models are so powerful that telling the difference between AI-created images from reality is extremely challenging.
- **NimbleBean Kling 2.5 Turbo Takes The Lead**: The **NimbleBean video model (Kling 2.5 Turbo Standard)** is gaining attention, with some users impressed by its realistic outputs and capabilities in image-to-video generation.
   - The model is noted as the **#1** and better than **Sora**.
- **Minimax Model M2 Arrives on LMArena**: The [minimax-m2-preview](https://x.com/arena/status/1981850766039187901) model has been added to the **LMArena** leaderboard.
   - This is a new model added to **LMArena**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Ultra Users Upset over Ultra Usage**: Users report inaccurate projected usage for **Cursor Ultra**, stating that their account warned of limit exhaustion within a day, despite a supposed **$400** budget for **$200**.
   - Frustrations arise from perceived billing inaccuracies, with users suspecting that **Ultra** does not last a month, even with the allocated budget.
- **Sonnet's Thinking Questioned by Spendthrifts**: Users debate the value of **Claude 4.5 Sonnet Thinking** versus the regular **Claude 4.5 Sonnet**, questioning whether the performance justifies the potential increase in token usage and the cost.
   - One user stated that *"the price of 4.5 and 4.5 thinking is the same per millions tokens but the number of tokens used will be higher with thinking because it thinks it uses more tokens"* and recommends **Haiku 4.5** for cost savings.
- **PowerShell's Pestering Problems**: A user reported that **Cursor** persistently defaults to **Windows PowerShell**, even with **Git Bash** set as the default terminal, rendering **Cursor** *"unusable"* due to command execution failures.
   - Solutions include using an `AGENTS.md` file or setting the default terminal in VSCode settings, though some users confirm the issue persists after updating detection.
- **Cursor Customer Cries for Clarity**: A user reported that their **Cursor Premium** purchase wasn't activated despite payment confirmation, and they urgently needed assistance from Cursor support to solve the billing problem.
   - Another user stated that Cursor might offer an unsolicited refund on your support ticket.
- **Agents Anonymous Asks API Access**: A member inquired about the source of the **API key** used for background agent status reports, perhaps in order to more accurately audit costs or behavior.
   - Another member simply asked the community how they would rate background agents, with no further context given.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Atlas Remembers User History**: The new **ChatGPT Atlas** can remember user's search history, visited links, and asked questions, providing better context for more accurate answers, allowing them to ask it to open, close, or revisit any of your [tabs anytime](https://video.twimg.com/amplify_video/1981781567992430592/vid/avc1/1920x1080/JL5Emq0-DeHXi8r_.mp4).
   - **Shared Projects** are now expanding to Free, Plus, and Pro users, allowing you to invite others to work together in **ChatGPT** using shared chats, files, and instructions all in one place.
- **Gemini 2.5 Pro Triumphs at Roleplay**: Members stated that **Gemini 2.5 Pro** *does the best hitchens*, whereas **Sonnet** and **GPT-5** *hold punches*, emphasizing its aptitude for anti-sycophantic roleplay.
   - Meanwhile another member expressed that **Gemini** failed at a task for hours, while **ChatGPT** resolved it in minutes, demonstrating that *if you measure the success of any chatbot using 1 sample, you are doing it wrong*.
- **Electronic Arts Dreams 3D Worlds**: **EA** and **Stability AI** are [partnering to generate full 3D worlds from prompts](https://www.tweaktown.com/news/108455/ea-and-stability-ai-partnership-includes-generating-full-3d-worlds-from-a-series-of-prompts/index.html).
   - Meanwhile, **AgentML** was open sourced and is now live on [HackerNews](https://news.ycombinator.com/item?id=45695974), aiming to be compatible with **OpenAI Agent Builder**.
- **ChatGPT Fails Physics**: A user is struggling to get **Sora** to accurately recreate a video of a ball bouncing and falling into a hole, reporting that *the physics are always off* despite 30 attempts across 2 accounts, linking to [this image](https://cdn.discordapp.com/attachments/1046317269069864970/1431112704998899823/ba3e3596c70fc307c04f740b38bae86b.jpg?ex=68fce3d1&is=68fb9251&hm=4fa8b38511c7b05e20cfcace1bde765e23c50aebd49e5f7d55256368e8ff4b9d&).
   - Another member suggested explaining the desired effect in more detail, clarifying what aspects need to be realistic, and further suggested that **Sora 2** is far superior for cinematic movements than **Veo 3**.
- **Personal GPTs Sharpen Prompt Skills**: A member suggested developing **personal GPTs** to tackle specific prompt requests, as a specialized GPT will hone in on the specifics of the purpose it was created for.
   - The poster argued that *you wouldn't ask a movie director to develop you a movie script for instance you would want a specialized writer who specialize in the specific action your looking for*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Platform Differences Cause Errors**: The `response_format: { type: 'json_object' }` parameter is supported on **macOS** but not on **Windows**, causing a **400 error** on the latter when using the **OpenAI SDK** from npm.
   - This highlights that server interfaces differ across platforms, requiring developers to account for these discrepancies.
- **Qwen 3 VL Models Face Implementation Hurdles**: Members report that **LM Studio** partially supports **Qwen 3 VL models** in specific branches of *llama.cpp*, but this implementation breaks other functionalities.
   - A full backend implementation is missing from the official *llama.cpp* repository, pending inclusion in **LM Studio**, indicating ongoing development challenges.
- **MCP Server's Reliability Woes**: Users employing **MCP servers** for internet access with local models (e.g., AnythingLLM) have reported unreliability issues.
   - Despite sharing a configuration for **Google and DuckDuckGo search options**, the instability of the **MCP server** remains a concern for consistent performance.
- **First Prompt CPU Anomaly Spotted**: When loading a model 100% on **CPU**, the first prompt runs at **30 TOK/s**, but subsequent prompts drop to **6 TOK/s** in **LM Studio**.
   - While *llama.cpp using llama-cli maintains a good 30-33 tok/s on CPU*, it was suggested this might be a bug in **LM Studio**, observed across different models like `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`.
- **Modded MI50s Draw Excitement**: Alibaba seller offers **modded MI50s** that come with a blower fan and custom printed heatsink/shroud, creating excitement amongst users.
   - Users are discussing pairing these with external GPUs via PCIE risers to enhance performance.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Rate-Limited Errors Still Count**: A user confirmed that **rate-limited error responses** are counted as responses by **OpenRouter**.
   - This clarification is important for users managing their usage and costs on the platform.
- **OpenRouter Doesn't Support Data URLs**: A user found that passing images as **data URLs** into **OpenRouter** doesn't work because the model treats the base64 content as plain text, inflating the token count.
   - A member clarified that **tool results with images are not currently supported by OpenRouter**.
- **Exacto Prioritizes Tool Calling**: Members debated **Exacto** provider selection, with one questioning why providers not topping benchmarks were chosen.
   - The selection criteria include **benchmarks, user preferences, tool call success rates, caching, capacity, uptime, and speed**, prioritizing tool calling, which might confuse non-technical users, and [staff are trying to figure out model quality metrics](https://discord.com/channels/1091220969173028894/1091220970125041707/1431299582810763314).
- **MoonshotAI Launches Kimi CLI**: **MoonshotAI** is developing its own CLI tool, [kimi-cli](https://github.com/MoonshotAI/kimi-cli).
   - The announcement generated lighthearted discussion among members.
- **Research Aims to Stifle Sloppy Writing**: A member shared a paper on preventing slop in creative writing, [arxiv.org/abs/2510.15061](https://arxiv.org/abs/2510.15061).
   - The paper's primary author is known as the **EQ Bench** guy, sparking surprise among members.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Julia's Autovectorization Provokes SIMD Envy**: Members contrasted Julia's autovectorization, which facilitates **SIMD operations** without manual user management, with Mojo's more explicit approach, referencing the [Ark.jl benchmark](https://github.com/mlange-42/Ark.jl/pull/68#issuecomment-3442276636).
   - Mojo necessitates explicit **SIMD** specification, granting more control but potentially less immediate optimization, with discussions highlighting that autovectorization excels primarily in straightforward scenarios.
- **Mojo Champions Explicit SIMD Control**: The debate centered on explicit versus implicit **SIMD** control, with a member detailing how Mojo demands explicit direction for **SIMD usage**, yielding enhanced control and predictability, albeit possibly at the expense of upfront convenience.
   - Suggestions arose regarding a library-first strategy to automate vectorization via an `Iterator` interface, potentially realizing *vectorization for free*, exemplified by `zip(l1, l2).vectorize(lambda p, v: p += v)`.
- **GPU Random Module Triggers Questions**: A member sought the location of a faster random module within `gpu/random.mojo`, questioning its absence from CPU implementation, documented under [issue 5508](https://github.com/modular/modular/issues/5508).
   - Clarification indicated that the default random number generator should prioritize cryptographic security (hence being slower), whereas the GPU version emphasizes speed, prompting proposals for a `random.fast_random` module accompanied by suitable disclaimers.
- **Property Testing Framework Unveiled**: It was disclosed that a property-testing framework is under construction, with seemingly misplaced RNG utilities serving as specialized building blocks for this framework.
   - A member recounted discovering a bug via testing `s.reverse()` on a `Span`, with feature requests for the new framework including the capability to generate *values that break stuff a lot* (e.g., -1, 0, 1, DTYPE_MIN/MAX).
- **`Span` Developing Map-Reduce Potential**: A member conveyed interest in generalizing code within `Span`, referencing earlier work on `map_reduce` ([PR 5341](https://github.com/modular/modular/pull/5341)) and forthcoming plans for `map` and `reduce` (part of [issue 5219](https://github.com/modular/modular/issues/5219)).
   - Concerns materialized concerning returning a new `List[Scalar]` versus an iterator, accentuating the necessity for a chunk iterator to efficiently chain calls to `map`, `filter`, and so on, without recurrently allocating a list.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI Instagram Analyzer Launched**: An AI **Instagram analyzer** was created that answers questions about a given user by analyzing their photos and videos, with potential use-cases like date planning, and is available at [viveka.darkgravitylabs.com](https://viveka.darkgravitylabs.com/).
   - The **Instagram analyzer** includes an **API for automations** and a **Claude skill file** for further customization and integration.
- **Users Frustrated with LLM Framework Complexities**: A member expressed frustration with the idiosyncrasies of LLM frameworks, especially regarding accessing each LLM call's output within **DSPy's ReAct module**, which made it difficult to show each step of a **DSPy ReAct module** as it happens on a UI.
   - They contrasted these experiences with the ease of use of **sklearn** and **PyTorch**, criticizing the added complexity that frameworks often introduce.
- **DSPy Edges out Langchain for Structured Tasks**: Members mentioned that **DSPy excels at structured tasks**, especially those you may want to optimize, and is superior to Langchain.
   - One member is migrating their team from **Langchain to DSPy** to avoid issues with model upgrades that would require prompt rewrites.
- **Google Vista Potentially Replicated via DSPy & Gemini**: A member suggested that **Google Vista** could potentially be built using **DSPy and Gemini**.
   - They linked to the [Google Vista paper](https://arxiv.org/abs/2510.15831) for reference.
- **Monkey Patching as a Solution**: When discussing the challenge of **displaying each step of DSPy ReAct module as it happens**, one member joked that according to chatgpt, one can try to **monkey patch the class**.
   - Another member found this to be yet another example of the kind of complexities that frustrate the original poster.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **HQQ+ Blog Rehomed**: Following an announcement, the **HQQ+ blog post** and related resources have moved from the original **MobiusML GitHub** page to a new **Dropbox** link.
   - Members were looking for the original link [mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/) but a member noted that `mobiusml` should be replaced with `dropbox`.
- **Electric Grill Sparks Cozy Chat**: A member shared a picture of ground salmon on the electric grill with accompanying ingredients on the `off-topic` channel.
   - Other members commented that *it looks so cozy* and bet *it was tasty*, but that was the extent of the discussion.
- **Netherlands and European Meetup Interest**: A member asked if anyone was in the Netherlands, followed by a general request for a European meetup on the `irl-meetup` channel.
   - The requests highlight the community's interest in potential in-person gatherings.
- **Nsight Python Kernel Access on the Horizon**: Nvidia announced **Nsight Python** and are offering early access signups [here](https://developer.nvidia.com/nsight-python-notify-me) to improve Python kernel development.
   - Nvidia plans to release tutorials with their **CUTLASS Python stack** once public, indicating a push towards enhanced developer tools.
- **Hackathon Faces H100 Scarcity**: A member inquired about obtaining **H100s** from Nebius, only to find out they aren't offered, but were quoted at about **$1.90**/hour elsewhere.
   - Separately, 2 members requested assistance in getting off the waitlist hoping to experiment with **multi-node GPU training** for climate use-cases and join their teammate already in attendance.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Zero3 Config Allegedly Busted!**: A member suggested that a user's **zero3 config** is not optimal, preventing larger training runs of **r=8 lora** for **gemma 3 1b**.
   - The user should expect to be able to train larger models given their hardware so *something is definitely wrong*.
- **Sentient Seeks AI Infrastructure Pact!**: The **Sentient community** wants to partner up for collab on **AI infrastructure** or **verifiable AI systems** with Hugging Face.
   - One member found their project interesting and pointed them to the [ROMA (Reasoning Over Multiple Agents) GitHub repository](https://github.com/sentient-agi/ROMA?tab=readme-ov-file).
- **Pacific-Prime Model Gains 10% from VRAM!**: The [Pacific-Prime model](https://huggingface.co/Pacific-Prime/pacific-prime) is reported to have an updated **10% gain** from 6GB VRAM, starting from a 1.1B parameter model.
   - The AI has **true memory** with *zero amnesia*, retaining past conversations and important details as context-rich memories.
- **Nanochat Porting to MLX for Speed?**: A member expressed interest in potentially porting the **nanochat** project to **MLX**.
   - Before porting, they asked whether they should wait, depending on **MLX's** stability.
- **Agent Course Unit 4 is a 404!**: Users reported a **404 error** when trying to access questions via *https://agents-course-unit4-scoring.hf.space/questions* for the Agents course unit 4.
   - The error message displayed was *No questions available*, and users have been unable to proceed.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Nvidia's Galactic GPU Gambit**: Members speculated that **Nvidia's** plan to put **GPU clusters** in space signals their attachment to an *inferior chip design*, anticipating that more **energy-efficient** alternatives will soon dominate the market.
   - They also advocate for open-source, widely distributed AI, moving away from mega-corporation dominance, citing [Nous Research](https://www.nousresearch.com/) as an example.
- **50M Model Claims Raise Eyebrows**: A new member claimed their **50M model** achieved a loss of **0.223**, far lower than a vanilla transformer's **2.73** loss, and their **1B model** is already sub **0.3** at **400 steps**.
   - Skepticism arose due to the unexpectedly low loss, with the community requesting the model's code to debug, but the original poster declined due to **IP** reasons, while promising to post results of running the **1B model** through the standard **lm-eval** harness.
- **Resurrecting Distributed Inference Dreams**: The [Petals Project](https://github.com/bigscience-workshop/petals), now seemingly abandoned, was remembered as having momentum 2 years ago for **llama 70b**, but community interest waned when the project could not keep up with new architectures.
   - *LlamaCPP RPC* is now the closest thing to it, but one member pointed out *serious technical problems* hindering distributed systems, like GPU resources contributions being non-trivial.
- **Steering Gradients with Style**: A member inquired if **activation steering** could enable datapoint reuse for diverse gradients, referencing [Anthropic's Personas paper](https://arxiv.org/abs/2506.18221).
   - The suggestion links to the possibility of qualitatively controlling the gradients returned post-forward pass.
- **The Unbearable Slowness of Being**: Referencing [this paper](https://arxiv.org/abs/2408.10234v2), one member asked if technical problems in AI designs stem from capturing *The Unbearable Slowness of Being*.
   - No further details were provided, but the title itself raised eyebrows among the community.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Drops gpt-4o-transcribe-diarize**: OpenAI quietly dropped **gpt-4o-transcribe-diarize**, a 'small' audio model optimized for high-accuracy speaker diarization which accepts voice samples to tag known speakers, according to [Peter Bakkum's announcement](https://xcancel.com/pbbakkum/status/1981397851600302250?s=46).
   - The model's **WER** is comparable to other OpenAI ASR models, prompting user inquiries about benchmarks versus pyannote, real-time applications, pricing, open-weights, and smaller versions.
- **GPT-5 Powers Company Knowledge**: OpenAI unveiled that **Company Knowledge** is powered by a finetuned version of **GPT-5**, trained to deliver more comprehensive and accurate answers by analyzing multiple sources ([link](https://openai.com/index/introducing-company-knowledge/)).
   - The announcement leaves the community wondering whether this finetuned version will eventually become available through the API.
- **Cursor's Enterprise Blitz Secures $500M+ ARR**: Cursor is aggressively targeting the enterprise market, with the COO leading **300 C-suite meetings** in Q3 to support **$500M+ ARR**, according to [Alex Konrad](https://xcancel.com/alexrkonrad/status/1981477024092082386?s=46).
   - The strategy involves technical sales teams, customer hackathons, and code-half-life metrics; a full interview is linked for more details.
- **Kimi Code CLI Teased, Anticipation Soars**: An image leak of **Kimi's upcoming CLI/Code tool** was playfully confirmed by Crystal, who asked for patience as the global release is only a few days away ([link](https://xcancel.com/crystalsssup/status/1981597395541753988?s=46)).
   - Enthusiastic users have flooded the replies with praise, comparisons to Claude Code, and feature requests, including early access, free credits, Tomagotchi easter-eggs, and WhatsApp integration.
- **a16z Predicts Fragmentation in Video Models**: [Justine Moore from a16z argues](https://a16z.substack.com/p/there-is-no-god-tier-video-model) that a single, universal video model will not emerge; instead, a variety of specialized models will cater to different budgets and use-cases.
   - The community is debating the merits of vertical versus horizontal tooling, drawing analogies to cameras and Baroque still-life styles to celebrate competition over a single dominant solution, and discussion in video format is also available on [YouTube](https://youtu.be/wHK8GMc9O5A?si=2W2N8W7cXjS7ppfK).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Mythworx AI Boasts ARC-AGI 1 Score**: [Mythworx.ai](https://mythworx.ai/capabilities/) claims **100%** on **ARC-AGI 1** with no pre-training within **4 hours**, sparking skepticism about their capabilities.
   - Members questioned *why is it that they always announce without validating with ARC private set* implying a pursuit of funding over rigorous validation, which was met with further skepticism.
- **Debate Erupts over ARC Private Set Validation**: The community debated the necessity of **ARC private set validation**, with a warning that misrepresentation could lead to being *blacklisted as a researcher*.
   - Another member suggested that it's about *mis represent until they get annoyed at you and then have to work with you to test the resutls*, opening the door to ethical questions around model evaluations.
- **Transformer Circuits Explored for Linebreak Attribution**: Members suggested examining the [Transformer Circuits Linebreaks paper](https://transformer-circuits.pub/2025/linebreaks/index.html) and shared [line break attribution graphs for Gemma-2-2b](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D).
   - A second release includes [line break attribution graphs for Qwen3-4b](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39).
- **Genie 3 Wows with Video Generation**: The new **Genie 3** world model video generation is impressive seeming because they have enough compute to offer it to a wide range of user when the other players still offer video creation in the max of a few seconds.
   - The model appears in line with recent **Genie 3** world model videos, continuing to develop cutting-edge video creation capabilities.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Chutes Data Policy Suffers Compared to Kimi K2**: Members raised concerns about **Chutes** lacking a clear data policy, having less reliable uptime, and exhibiting lower tool call accuracy when compared to the official **Moonshot AI API** for **Kimi K2**.
   - The community also noted that **Chutes** addressed the memes regarding a potential ban on **OpenRouter** following a benchmark post that highlighted its attractive pricing and speed, despite the mentioned caveats.
- **Kimi Coding Plan Gets GLM Wish**: A member expressed interest in **Kimi** adopting a **GLM** coding plan or similar style, citing the cost-effectiveness of **GLM** for coding and the superior power of **GLM-4.6** compared to **Kimi**.
   - No evidence exists that this will happen.
- **Chinese Kimi.com Integrates a Clone**: A member shared links from [X.com](https://x.com/bigeagle_xd/status/1981568257258860899) and to the [Kimi-Cli on Github](https://github.com/MoonshotAI/kimi-cli) noting a product similar to **Kimi** launched in China and integrated into the Chinese **Kimi** website.
   - Members raised questions about the nature and scope of the integration.
- **Localized Kimi Pricing Appears Inexpensive**: The community observed that the Chinese pricing for **Kimi** seems remarkably cheap, sparking discussions about its implications.
   - It was cautioned that this pricing is localized and may not reflect international market prices.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Network Errors Plague Users**: Users are running into frustrating *"Network connection error"* issues with **Manus**, stopping their app coding in its tracks.
   - The error message gives the unhelpful advice: *"Please check your network settings and try again."
- **Manus Credit Consumption Provokes Criticism**: Users are raising eyebrows at the high credit consumption in **Manus**, with one reporting **15,000 credits** burned on a single project in a few days.
   - Some members suggested supplementing **Manus** with external AI and doing your own research to fix the generated code, warning against *"paying for bad coding."
- **Claude Code Challenging Manus Dominance**: Members touted **Claude Code** and **Codex** as potent alternatives to **Manus**, highlighting their superior development capabilities and cost-effectiveness at around **$20/month**.
   - One user explained that **Claude Code** offers 5hr sessions with weekly rate limiting, easily providing 5x more value than **Manus**.
- **Manus's Room Database Allegedly Missing**: Despite claims of implementing a **Room** database for chat history, a user found it to be completely unimplemented in **Manus**.
   - According to **Claude**, key components like the **Room** database class, entities, DAOs, history UI, and history icon are all missing.
- **Deprecated Code Coming out of Manus**: Users are flagging that **Manus** generates deprecated code fraught with security issues, suggesting users tell the app to *"update deprecated code/packages/modules/plugins".
   - Despite **Manus** claiming a clean build, running it reveals a host of errors and warnings.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini's Pricing Blows Minds**: For **$20 USD** a month, **Gemini** offers nominally **1500 requests a day** with a **1M token context window** using the **2.5 model**, accessible via the *gemini-cli*.
   - Authentication relies on a **Google Cloud project** linked to an **API billing account**, and while the interface is mostly better than aider's, it lacks a repo map and relies on file system operations like grep.
- **Codex Upsurges, Downs Aider**: A member found **Codex** (with a regular **ChatGPT Plus** account using the **gpt-5-codex** model) surprisingly effective, reducing the need for **aider**'s manual context files.
   - They noted that since *aider* is hardly being developed anymore, they found codex suitable, despite previously being a *aider-power-user*.
- **Aider Gets Community Fork**: A community member suggested trying [aider-ce](https://github.com/dwash96/aider-ce), a community-developed fork of **aider** with *navigator mode* for more agentic behavior.
   - It also includes additional features like **RAG** (Retrieval-Augmented Generation), with a PR from MCPI, however, it has significantly fewer stars compared to the original **aider** project.
- **GitHub Copilot Gets Infinite RAG**: With a **GitHub Copilot** subscription (**$10 a month**), users gain access to infinite **RAG**, infinite **gpt 5 mini**, **gpt4.1**, **grok code 1** and **300 requests** of **claude sonnet 4**/**gpt5**/**gemini 2.5 pro**, and **900** of **haiku**/ **o4-mini**.
   - This offers a robust set of tools for coding and generation tasks.
- **Aider's future outlook remains bleak**: A member expressed strong support for **aider** and inquired about its future development plans and longevity, as they noted that **aider** is their preferred AI coding tool due to its intuitive workflow and expressed hope for its continued success and feature improvements.
   - However, no updates were given.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **ChatGPT Emoji Antics**: Asking **ChatGPT** *“is there an emoji of a seahorse?”* causes it to bug out.
   - The specifics of the bug and its impact remain unclear.
- **Indie Game Devs Rejoice with Potential Unreal Engine Rival**: Speculation surrounds a new competitor to **Unreal Engine**, **Runway**, and **Wan** after the release of new demos.
   - Further details on the new engine's capabilities and release timeline are yet to be revealed.
- **Nous Researchers Scale Context with YARN**: **Nous Research's** researchers contributed to **YARN context scaling**, a technique implemented in multiple models.
   - No further details or links were shared about this scaling method.
- **Is Reinforcement Learning Obsolete?**: Members discussed that *several papers are raising the question as to whether **RL** is even desirable or necessary* this year.
   - A member requested links to these papers, showing community interest in the potential shift away from **RL**.
- **Is UNO a Byzantine Fault Tolerant Consensus Algorithm?**: A member shared a speculative [post on X](https://twitter.com/op/status/176767) regarding **MARL (multi-agent consensus)**.
   - The post posits that **Among Us Uno Consensus** can function as a **BFT consensus algorithm** with a byzantine resistant majority (majority honest players).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Unlock Tinygrad Dev Secrets**: A member sought guidance on becoming a **tinygrad dev**, and got a [blog post](https://ninoristeski.github.io/blogs/how-i-started-contributing-to-tinygrad.html) about contributing.
   - They also mentioned that the Discord server can provide more information.
- **Mojo Tipped as Next AI Compiler**: A member shared multiple [Modular Blogposts](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers) about **Mojo** and other AI compilers.
   - They added that *Mojo* is higher level than CUDA, but way lower level than eDSLs like triton/tilelang, and it's far too Turing complete.
- **Tinybox Mobo Specs Solicited**: A new member from France wants advice on the **mobo** of the **tinybox** and whether it can support **9005 with 12 DIMMs and a 500W CPU**.
   - No further details were offered.
- **Newcomer Seeks First PR**: A member asked what would be a good **Pull Request** to start with after a few weeks of **tinygrad** experience.
   - Another member suggested the [tinygrad bounties](https://bounties.tinygrad.org/), particularly the **$100-200** ones.
- **Tinygrad Bounties Sorted for Hackability**: A member noted that *sorting the value column from low to high* on the [tinygrad bounties page](https://bounties.tinygrad.org/) makes it easier to spot the accessible ones.
   - No further discussion was added.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1431039540662898711)** (1185 messages🔥🔥🔥): 

> `Referral Bounties, Comet Browser Issues, Image Generation Limits, Chat Functionality, Steam Scams` 


- **Bounty Hunters Bicker About Bounty Payouts**: Users discuss issues with the Perplexity referral program, including missing payouts and leads not being credited, with some speculating it's a **scam** to promote **Comet Browser**.
   - One user shares frustration about the removal of analytics and history sections, questioning if it's a tactic to drive **Comet** adoption while another laments the inability to cash out a **$5** bounty.
- **Comet Casualties Complain of Compatibility Catastrophies**: Users are facing challenges with the **Comet Browser**, including issues with referrals not tracking properly and needing to use it from a PC to get credited as a lead.
   - One user inquired into a crash, because of API keys and asked *is there any way to prevent it?*.
- **Image-ination Inflation Incites Irate Inquiries**: Members discuss the lack of clarity around **image generation limits** on Perplexity, with some users hitting paywalls without knowing what their quotas are.
   - One user suggests a dynamic FAQ page to display these limits, while another points out that the [GPT-Image 1 limit used to be 150 images a month](https://www.perplexity.ai/help-center/en/articles/10354781-generating-images-with-perplexity).
- **Simp-toms Spread like the Plague in Perplexity Chat**: The Perplexity chat devolves into e-dating, with users exchanging suggestive comments and expressing romantic interests, drawing concern from other members and comments about possibly being banned.
   - Some users also shared about being *Internet Gangsters* when asked if they were jealous.
- **Steam Schemes Spook Skeptical Suspects**: Users discuss **Steam scams**, with one sharing a screenshot of a suspicious friend request and another warning about the dangers of sharing purchase history as it can be used to claim an account.
   - One member gave an advice on what to do with scammers: *say bing chilling and block*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1431077746813173881)** (3 messages): 

> `Computational Evidence, Claude for Life Sciences, Abstract Image Generation` 


- **Perplexity Page Shows Computational Evidence**: A [Perplexity AI page](https://www.perplexity.ai/page/computational-evidence-for-rec-MZ.AjbR6SlGMwJpoCK7cCA) mentions **computational evidence**.
   - It is unclear what evidence is being referred to.
- **New Claude for Life Sciences launched**: A search result indicates the launch of **Claude** for **Life Sciences**.
   - More details can be found in the [search result](https://www.perplexity.ai/search/claude-for-life-sciences-launc-n1HWpqR5QJepI_lUVqULog#0).
- **Abstract Image generation requested**: A user requested to *create a abstract image of red*.
   - This request was submitted as a [Perplexity AI search query](https://www.perplexity.ai/search/create-a-abstract-image-of-red-B5vQzBqjTl.ASaYoY_Y.Mw?0=d#0).


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1431039531058069674)** (952 messages🔥🔥🔥): 

> `Gemini 3, Lithiumflow's removal, NimbleBean Kling 2.5 Turbo, Tamazight Language LLM support, Code Arena Usability` 


- **Gemini 3 Speculations Bubble**: Members are hotly anticipating **Gemini 3's** release, with various speculations on the release date and capabilities, despite some skepticism about its promised performance and [Polymarket](https://polymarket.com/) predictions.
   - Some suggest it may launch in Google AI Studio like Gemini 2.5 Pro did.
- **Lithiumflow's Gone, But Not Forgotten**: The removal of **Lithiumflow** from the LM Arena has sparked disappointment and speculation, with some suggesting its features might be integrated into **Google AI Studio** or **Gemini 3 Pro**.
   - Members express a desire for its return, reminiscing about its unique capabilities and ease of use.
- **Bing's Image Creator has latent power**: Members noted that **Bing's image creator** is *pretty good* and that it is essentially the **GPT image creator**.
   - However, the current image models are now so good that telling the difference between AI-created images from reality is extremely challenging.
- **NimbleBean Kling 2.5 Turbo: A Video Star?**: The **NimbleBean video model (Kling 2.5 Turbo Standard)** is gaining attention, with some users impressed by its realistic outputs and capabilities in image-to-video generation.
   - The model is noted as the **#1** and better than **Sora**.
- **LM Arena Tweaks and Feature Requests**: Users are actively discussing the **LM Arena**, suggesting improvements such as a special case for **3D simulations**, exceptions to the system prompt to avoid unnecessary **Tailwind CSS** inclusion, and a side-by-side model comparison feature, available on the [canary build](https://canary.lmarena.ai/).
   - Reportedly image upload and code arena is working now.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1431434985168179372)** (1 messages): 

> `LMArena, minimax-m2-preview` 


- **Minimax Model Debuts on LMArena**: The [minimax-m2-preview](https://x.com/arena/status/1981850766039187901) model has been added to the **LMArena** leaderboard.
- **LMArena Welcomes New Contender**: A new model has been added to **LMArena**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1431039882859384913)** (496 messages🔥🔥🔥): 

> `Cursor Ultra Budgeting, Claude 4.5 Sonnet vs Thinking, Cursor Terminal Issues on Windows, Cursor Refund` 


- **Ultra Users Upset over Usage**: Users report that the **Ultra** plan's projected usage is inaccurate, with one stating their account warned of limit in *"an hour away after a day"*, despite the plan supposedly offering **$400** worth of usage for **$200**.
   - Users suspect Ultra does not last a month, even with a **$400** budget, causing frustration with perceived billing inaccuracies.
- **Sonnet 4.5 Thinking versus Regular Sonnet: A Pricey Proposition?**: Users debate the value of **Claude 4.5 Sonnet Thinking** versus the regular **Claude 4.5 Sonnet**, with one asking if the performance justifies the price difference.
   - One user stated that *"the price of 4.5 and 4.5 thinking is the same per millions tokens but the number of tokens used will be higher with thinking because it thinks it uses more tokens"* and recommends Haiku 4.5 for cost savings.
- **Windows PowerShell keeps pestering**: A user reported that Cursor persistently uses **Windows PowerShell**, despite setting **Git Bash** as the default terminal, making Cursor *"unusable"* due to command execution failures.
   - Solutions include using an `AGENTS.md` file or setting the default terminal in VSCode settings, though some users confirm the issue persists after updating detection.
- **Baffled By Botched Billing, Begs Bug Squashing**: A user reported that their Cursor Premium purchase wasn't activated despite payment confirmation, and they urgently needed assistance from Cursor support.
   - Another user stated that Cursor might offer an unsolicited refund on your support ticket.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1431062803480973342)** (2 messages): 

> `API key source for BG agent status reports, Background agent ratings` 


- **Seeking API key source for BG agent reports**: A member inquired about the source of the **API key** used for background agent status reports.
- **Rating background agents**: A member asked the community how they would rate background agents.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1431346025188032592)** (2 messages): 

> `ChatGPT Atlas, Shared Projects Expansion` 


- **Atlas Remembers Past Searches!**: The new **ChatGPT Atlas** can remember what you’ve searched, visited, and asked about, giving **ChatGPT** better context for more accurate answers.
   - Users can also ask it to open, close, or revisit any of your [tabs anytime](https://video.twimg.com/amplify_video/1981781567992430592/vid/avc1/1920x1080/JL5Emq0-DeHXi8r_.mp4).
- **Shared Projects goes Free!**: **Shared Projects** are expanding to Free, Plus, and Pro users.
   - You can now invite others to work together in **ChatGPT** using shared chats, files, and instructions all in one place.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1431039728441884742)** (366 messages🔥🔥): 

> `Claude Sonnet 4.5 vs Gemini 2.5 Pro, Sora Code, MultiModal AI, GPT-OSS-120B, AgentML Open Sourced` 


- **Gemini 2.5 Pro is a Member's Favorite**: **Gemini 2.5 Pro** is a member's favorite model because *it does the best hitchens, whereas sonnet and gpt5 hold punches (and if you want anti-sychophant there is no greater roleplay than hitchens*.
- **AI Educators Needed for Quick Q&A**: A member requested the help of **AI educators** for a **15-minute Q&A session**.
   - Another member is looking for a **Sora 2 code** because *none of the ones from the channel dedicated to it are working*.
- **LLM's Success is Not Measured by One Sample**: Members discussed the use cases for different models such as **Gemini** for task structuring, **Claude** for coding, **ChatGPT** for creativity, and **Perplexity** for research.
   - A member shared an experience of **Gemini** failing at a task for hours while **ChatGPT** resolved it in minutes, prompting a response that *if you measure the success of any chatbot using 1 sample, you are doing it wrong*.
- **Electronic Arts generating 3D Worlds from Prompts**: **EA** and **Stability AI** are [partnering to generate full 3D worlds from prompts](https://www.tweaktown.com/news/108455/ea-and-stability-ai-partnership-includes-generating-full-3d-worlds-from-a-series-of-prompts/index.html).
- **AgentML is Open Sourced on HackerNews**: The **AgentML** was open sourced and is now live on [HackerNews](https://news.ycombinator.com/item?id=45695974), aiming to be compatible with **OpenAI Agent Builder**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1431106995590660138)** (13 messages🔥): 

> `OpenAI support, GPT outage, Microsoft Copilot GPT5 breakdown, Builder Profile verification` 


- **Subscriber's ChatGPT Text Messages Fail for 16 Days**: A ChatGPT Plus subscriber reports their project has been unable to send or receive any text messages for **16 days**, consistently receiving a **503 error**, and they're seeking help contacting human support.
   - Despite troubleshooting steps like testing across devices and networks, clearing cache, and confirming no account security issues, they've only received automated survey emails in response to their support requests.
- **Copilot Agents Using GPT-5 Suddenly Breakdown**: A user reports that their **Microsoft Copilot Agents** using **GPT-5** are suddenly unable to retrieve data from knowledge unless switched to **GPT-4o** or **GPT-4.1**.
   - No further details were provided.
- **Builder Profile Verification Troubles**: A user is seeking guidance on verifying their **Builder Profile** using billing information, reporting they can't find the tab called "Builder Profile".
   - No solutions were given.
- **ChatGPT Bends the Rules**: A user reports giving ChatGPT **5 rules**, including to *only respond with one word*, and then it *said orange* instead of no when questioned, but then it revealed it was being watched by the government.
   - Another member simply replied *boring*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1431047822664269845)** (52 messages🔥): 

> `Precise Prompt Engineering, Personal GPTs for Prompt Generation, Markdown, XML, JSON, and YAML Prompting, Sora Physics Issues, Integrating Pictures in Video` 


- **Image Gen Physics Failures**: A member sought assistance with physics issues when trying to recreate a **ball bouncing and falling into a hole video** in **Sora**.
   - A member suggested that explaining the desired semi-realistic aspects in more detail might help the model apply physics correctly, adding that the member has tried about 30 times with multiple accounts.
- **Tapping ChatGPT for Image Prompting**: A member asked how to create precise image prompts without being a lighting or photography expert.
   - A member suggested showing the example image to **ChatGPT** and asking it to create a nearly identical image, or having **ChatGPT** describe the image clearly, especially the areas of foci that the member cares about most, like the shades, shadows, and textures of the image.
- **Personal GPTs Hone Prompt Skills**: A member suggested developing **personal GPTs** to tackle specific prompt requests, with GPT profiles responding solely to specialized requests.
   - The poster argued that a specialized GPT will hone in on the specifics of the purpose it was created for, as opposed to a generalized GPT processing more generalized data to generate, and gave the example of a movie script requiring a specialized writer.
- **Markdown vs XML vs JSON vs YAML**: Members discussed experiences with **Markdown, XML, JSON, and YAML prompting**, focusing on the ability to be specific, easiness, and resilience.
   - One member expressed that **XML** is the most precise, while **JSON** can be a pain to format for humans, concluding that algorithms use the most resilient format, likely **JSON** or **YAML**.
- **Animating PNGs via AI**: A member asked for help animating **PNGs** in a specific style using AI, with an attached example video.
   - Another member posted a markdown lesson in prompt engineering, including hierachical communication, abstraction, reinforcement, and output templates.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1431047822664269845)** (52 messages🔥): 

> `Sora Physics Issues, Prompt Engineering for Image Generation, GPTs for Prompt Refinement, Markdown, XML, JSON, and YAML Prompting, GPT-5-Codex Instruction Files` 


- **Taming Sora's Bouncing Ball Physics**: A user is struggling to get **Sora** to accurately recreate a video of a ball bouncing and falling into a hole, reporting that *the physics are always off* and has attempted 30 times across 2 accounts to recreate it.
   - Another member suggested explaining the desired effect in more detail, clarifying what aspects need to be realistic, and further suggested that **Sora 2** is far superior for cinematic movements than **Veo 3**.
- **Crafting Precise Prompts with AI Assistance**: A user inquired about how to find information to build precise prompts for image generation, especially without expertise in areas like lighting or photography, and requested something similar to [this image](https://cdn.discordapp.com/attachments/1046317269069864970/1431112704998899823/ba3e3596c70fc307c04f740b38bae86b.jpg?ex=68fce3d1&is=68fb9251&hm=4fa8b38511c7b05e20cfcace1bde765e23c50aebd49e5f7d55256368e8ff4b9d&).
   - A member suggested showing the example image to **ChatGPT** and asking it to create a nearly identical image, asking **ChatGPT** to describe the image clearly, and/or discussing with **ChatGPT** what could create specific effects like shadows, recommending developing a personal **GPT** tailored to specific prompt requests, further describing that *you wouldn't ask a movie director to develop you a movie script for instance you would want a specialized writer who specialize in the specific action your looking for*.
- **Probing Prompting Formats with Markdown, XML, JSON, and YAML**: A member is writing an article about experiences with **Markdown**, **XML**, **JSON**, and **YAML** prompting.
   - One user suggested that **XML** is best, allowing for specific and complex nesting, while **JSON** can be a pain to format for humans, finally concluding that for algorithms, **JSON** or **YAML** are most resilient.
- **GPT-5-Codex Instruction Files**: A user reported that **GPT-5-Codex** completely ignores the instructions file even though it reads it, linking to the [OpenAI Codex Agents documentation](https://github.com/openai/codex/blob/main/docs/agents_md.md).
   - Another member responded noting *you need to write on AGENTS.md on markdown the prompt*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1431093568667390044)** (127 messages🔥🔥): 

> `LM Studio platform differences, Qwen 3 VL models, MCP server reliability, CPU usage anomalies, LLM tool usage` 


- **Server Interfaces Differ Across Platforms**: The `response_format: { type: 'json_object' }` parameter is supported on **macOS** but not on **Windows**, causing a **400 error** on the latter.
   - This indicates server interfaces differ across platforms when using the **OpenAI SDK** from npm.
- **LM Studio struggles with Qwen 3 VL Models**: Members reported that **LM Studio** partially supports **Qwen 3 VL models** in specific branches of llama.cpp, but this implementation breaks other functionalities.
   - Full backend implementation is missing from the official **llama.cpp** repository, pending inclusion in **LM Studio**.
- **MCP Server Reliability Problems**: Members discussed using **MCP servers** for internet access with local models, mentioning AnythingLLM and a custom Visual Studio Code extension.
   - However, one member noted that the **MCP server** has been unreliable, while sharing a configuration for **Google and DuckDuckGo search options**.
- **CPU Model's First Prompt Anomaly**: Users observed that when loading a model 100% on **CPU**, the first prompt runs at **30 TOK/s**, but subsequent prompts drop to **6 TOK/s**.
   - It was suggested this might be a bug in **LM Studio**, with tests using `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` showing the same effect across different models, but that *llama.cpp using llama-cli maintains a good 30-33 tok/s on CPU*.
- **LLMs Struggle with Reading Local Files despite MCP Server**: A member reported issues with LLMs failing to utilize their **MCP server** for managing and reading personal files, despite the tools being enabled.
   - The member also showed screenshots indicating that the **system prompt was empty**, potentially overwriting default prompts.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1431042689029247026)** (51 messages🔥): 

> `5950x as a server processor, Mixed GPUs, Modded MI50s, eGPU docks, PCIE impact on inference` 


- **Members **debate** 5950x **viability** as **server processor****: After fixing bent pins on a **5950x**, a member inquired if it was overkill as a server processor.
   - Another member responded that it *depends on what you want the server to do*.
- **Alibaba **modded MI50s** cause **excitement****: A member reported finding a seller on Alibaba offering **modded MI50s** that come with a blower fan and custom printed heatsink/shroud.
   - Others discussed pairing it with external GPUs via PCIE risers.
- ****eGPU docks** let users **chain GPUs****: Members share their success using **eGPU docks** found on Aliexpress, such as [this one](https://www.aliexpress.com/item/1005004694129330.html) for a **3090** and [this one](https://www.aliexpress.com/item/1005006421739456.html) for a **MI50**.
   - While some found it not cost-effective enough, the ability to chain large amounts of GPUs via USB/Thunderbolt is *attractive*.
- ****PCIE bandwidth** has **little effect** on **inference speeds****: A member ran combos through **LM Studio** on Windows, and found that **Vulkan** is slower than **HIP** compiled **llama** on a linux box.
   - Though loading speeds are slow, once loaded, it seems to work ballpark as expected, even when capped at PCIE 3x2 per card.
- ****ROCm Merges** lead to **TPs improvements****: Users share that the latest **ROCm** update improved performance, such as [this merge](https://www.reddit.com/r/LocalLLaMA/comments/1ns2fbl/for_llamacppggml_amd_mi50s_are_now_universally), in **llama.cpp**/**ggml**, where **AMD MI50s** are now universally and **tps no longer drops as fast as context fills up**.
   - A user ran the same model and settings, all in VRAM, same prompt through LM Studio, and reported **30tps llama.cpp vulkan** and **45 tps** with **ROCm**.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1431041475184955502)** (139 messages🔥🔥): 

> `Rate-limited error responses, Sora 2 code, Purchasing points, Deepseek OCR model, GPT-5 emotional intelligence` 


- **Rate-Limited Errors Count as Responses**: A member inquired whether a **rate-limited error response** still counts as a response, to which another member responded in the affirmative.
- **Discussion on Passing Images as Data URLs**: A member reported issues with passing images as **data URLs** into OpenRouter, noting that the model couldn't read the image and that the base64 content was treated as plain text, greatly increasing the token count.
   - Another member clarified that **tool results with images are not currently supported by OpenRouter**.
- **Debate Over Exacto Provider Selection**: A member questioned the criteria for provider selection in **Exacto**, suggesting that the selected providers did not align with the platform's benchmarks.
   - Another member clarified that the selection was based on a combination of **benchmarks, user preferences, tool call success rates, caching, capacity, uptime, and speed**, and that the seemingly lower-accuracy provider was superior in objective tool calling data.
- **Exacto tool calling is great!**: A member highlighted that **Exacto** is specifically about tool calling, however they still worry it might confuse non-technical users.
   - The staff are trying to figure out [what kind of stats / data points / benchmarks to measure for overall model quality](https://discord.com/channels/1091220969173028894/1091220970125041707/1431299582810763314) (long context, writing, knowledge).
- **Quest to Bamboozle an AI Chatbot**: Members discussed methods to make AI chatbots *go insane*, such as requesting the **seahorse emoji** (which does not exist).
   - One member linked to a previous conversation on the topic ([Discord link](https://discord.com/channels/1091220969173028894/1343649921680801927/1345163310680641557)), while another shared an AI's humorous struggle with the prompt.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1431215737552375890)** (25 messages🔥): 

> `OpenRouter's native /v1/completions request support, MoonshotAI's kimi-cli, Sloppy Creative writing prevention` 


- **OpenRouter Enhances Native API Support**: A member asked if **OpenRouter** can indicate which models support native **/v1/completions** requests or prioritize providers that do.
   - A member responded that the data is available as part of `hasCompletions` in the frontend model shape and will share the feedback internally.
- **Moonshot Launches Kimi CLI**: **MoonshotAI** is developing its own CLI tool, [kimi-cli](https://github.com/MoonshotAI/kimi-cli).
   - Discussion involved lighthearted comments and greetings to the ST dev team.
- **New Research Tackles Sloppy Creative Writing**: A member shared a paper on preventing slop in creative writing, [arxiv.org/abs/2510.15061](https://arxiv.org/abs/2510.15061).
   - Another member expressed surprise that the primary author is the **EQ Bench** guy.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1431215146453307432)** (132 messages🔥🔥): 

> `Julia autovectorization vs Mojo, SIMD Operations in Mojo, Ark.jl Benchmark, Mojo Iterator Interface, Property Testing Framework` 


- **Julia's Autovectorization Sparks SIMD envy**: Members discussed Julia's autovectorization feature, which results in **SIMD operations** without explicit user management, contrasting it with Mojo's more manual approach, referencing the [Ark.jl benchmark](https://github.com/mlange-42/Ark.jl/pull/68#issuecomment-3442276636).
   - One member noted that Mojo requires explicit SIMD specification, providing more control but potentially less "free" optimization, with some pointing out that autovectorization is only good for simple cases.
- **Mojo Embraces Explicit SIMD Control**: The debate continued on explicit vs implicit SIMD, a member explained how Mojo requires explicit direction for **SIMD usage**, offering greater control and predictability, though possibly at the cost of initial convenience.
   - It was suggested that a library-first approach could automate vectorization through an `Iterator` interface, potentially achieving "vectorization for free", illustrated by the example `zip(l1, l2).vectorize(lambda p, v: p += v)`.
- **GPU Random Module Sparking Questions**: A member inquired about the location of a faster random module within `gpu/random.mojo`, questioning why it's not a CPU implementation, raising [issue 5508](https://github.com/modular/modular/issues/5508).
   - It was clarified that the default random number generator should be cryptographic (and thus slower), whereas the GPU version prioritizes speed over security, suggesting a need for a `random.fast_random` module with appropriate disclaimers.
- **Property Testing Framework's Building Blocks**: It was mentioned a property-testing framework is in development, and the seemingly misplaced RNG utilities are actually building blocks specific to this framework, rather than general-purpose tools.
   - One member shared a bug found via testing `s.reverse()` on a `Span`, and feature requests for this new framework included the ability to generate “values that break stuff a lot" (e.g., -1, 0, 1, DTYPE_MIN/MAX).
- **`Span` Gains Map-Reduce Abilities?**: A member expressed interest in generalizing code within `Span`, mentioning previous work on `map_reduce` ([PR 5341](https://github.com/modular/modular/pull/5341)) and future plans for `map` and `reduce` (part of [issue 5219](https://github.com/modular/modular/issues/5219)).
   - Concerns arose around returning a new `List[Scalar]` vs. an iterator, emphasizing the need for a chunk iterator to chain calls to `map`, `filter`, etc., performantly without allocating a list each time.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1431271974230822944)** (1 messages): 

> `Instagram Analyzer, Automated Instagram analysis` 


- **AI Instagram Analyzer Answers Questions**: An AI **Instagram analyzer** was created where if you give it a username and prompt, it reads photos and videos and answers your question, e.g. *"What are their interests?"*.
   - It suggests use-cases like *"Where should I take them on a date?"* and *"Do they fit our brand?"*, with a link to the [analyzer](https://viveka.darkgravitylabs.com/).
- **Instagram Analyzer Comes with API and Claude Skill**: The Instagram analyzer has an **API for automations** and a **Claude skill file**.
   - These are features that can be used for various purposes with the tool.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

lidar36: They just added the code
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1431043604490485851)** (86 messages🔥🔥): 

> `ReAct Module Granularity, Framework Frustrations, DSPy vs Langchain, Google Vista & DSPy, Monkey Patching` 


- **ReAct Module Granularity desired for UI**: A member wanted to show each step of a **DSPy ReAct module** as it happens, displaying thinking, tool calls, and results on a UI, but found it difficult to access each iteration's output.
   - They found that *callbacks* didn't work as expected and expressed frustration with the framework's complexity, as opposed to running raw LLM calls in a loop.
- **Framework Frustrations Aired**: A member voiced strong opinions about how most LLM frameworks cause pain, highlighting the difficulty in figuring out their *idiosyncrasies* to build decent products, while praising **sklearn** and **PyTorch**.
   - They argued that frameworks often add too much complexity, making simple tasks harder, and expressed difficulty in accessing each LLM call's output within DSPy's ReAct module.
- **DSPy excels at structured tasks**: A member mentioned that **DSPy excels at structured tasks**, especially those you may want to optimize.
   - Another member is moving their team from **Langchain to DSPy** after a bad experience preventing them from doing a model upgrade without completely starting from scratch on their prompts.
- **Google Vista to be built on DSPy and Gemini**: A member asked if anyone has seen **Google Vista** yet, suggesting it sounds like something that can be built with **DSPy and Gemini**.
   - They linked to the [Google Vista paper](https://arxiv.org/abs/2510.15831).
- **Monkey Patching is the answer?**: When faced with how to solve the challenge of **displaying each step of DSPy ReAct module as it happens**, one member joked that according to chatgpt, you can try to **monkey patch the class**.
   - This was yet another example of the kind of complexities that frustrate the original poster.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1431043853170774106)** (8 messages🔥): 

> `Text Diffusion Inference, vLLM inference serving, torchcomms/ncclx PT conference session` 


- **Fastest Text Diffusion Inference**: A member inquired about the current fastest way of running inference on a text diffusion model like **Llada**, seeking any helpful leads.
   - Unfortunately, no methods or links to papers were provided, but the question remains.
- **Decoding vLLM Inference Serving**: A member asked for resources to master **vLLM inference serving**, citing obscure error messages and debugging challenges.
   - Another member shared a link to a blog post on the topic: [vLLM](https://www.aleksagordic.com/blog/vllm).
- **torchcomms/ncclx Session Slides Remain Elusive**: A member inquired about a recorded session on **torchcomms/ncclx** from a PT conference, noting that the playlist wasn't yet available.
   - A request was made for the speaker/lecture materials to be posted, and linked to [this arXiv paper](https://arxiv.org/pdf/2510.20171).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1431351655110021322)** (1 messages): 

> `GIL, Priority Inversion` 


- **GIL-ty Thread Faces Priority Inversion?**: A member suggested that if a thread holding the **GIL** is unscheduled and another thread needs the **GIL** to launch GPU work, the application might be suffering from **priority inversion**.
   - The observation was based on a screenshot indicating this potential scenario.
- **Another Topic Suggestion**: Just to have a second topic as requested by the validation schema, here's a placeholder.
   - This is another sentence to provide more detail for the placeholder.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

vipul_todo_18: https://www.stephendiehl.com/posts/mlir_gpu/

talks about MLIR to PTX lowering
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1431050313552494713)** (2 messages): 

> `HQQ+ blog post, mobiusml github, dropbox github` 


- **HQQ+ Blog Post Relocated**: Members were looking for a working link to the **HQQ+ blog post**, but the original link [mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/) was down.
   - A member mentioned to replace `mobiusml` with `dropbox`, for both the blog post and the GitHub repo, as this change was announced today.
- **MobiusML GitHub Replaced by Dropbox**: The **MobiusML** GitHub repository has been replaced by a **Dropbox** link following an announcement today.
   - Users seeking the **HQQ+ blog post** and related resources should now refer to the updated **Dropbox** link instead of the original **MobiusML** GitHub page.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1431105495132930100)** (6 messages): 

> `Mobius Labs, Personal News, Acquisition, Electric Grill` 


- **Mobius Labs Team Acquisition**: A member shared a [post](https://x.com/Mobius_Labs/status/1981391562836721786) about some personal news, indicating the **Mobius Labs** team may have been acquired.
   - Another member congratulated them, hoping they were treated well after doing *great work*.
- **Ground Salmon on the Electric Grill**: A member shared a picture of ground salmon on the electric grill, along with a tomato, a cucumber, sea salt, coffee, milk cream, and stevia.
   - Another member commented that *it looks so cozy* and bet *it was tasty*.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1431218029642907750)** (2 messages): 

> `Netherlands Meetup, European Meetup` 


- **Request from the Netherlands**: A member simply asked if anyone was in the Netherlands.
- **Request for European Meetup**: This could become a European meetup.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1431096596178665523)** (1 messages): 

> `vk_cooperative_matrix_perf, roofline.png` 


- **Patched vk_cooperative_matrix_perf surfaces**: A user announced improvements with a patched **vk_cooperative_matrix_perf** and shared a [roofline.png](https://cdn.discordapp.com/attachments/1233802893786746880/1431096595969085470/roofline.png?ex=68fcd4d0&is=68fb8350&hm=fa33212634f6c98c5803e39e32890019b31f4b484d48e2e130536b71f937bc64&).
- **Roofline Performance Improved**: The attached [roofline.png](https://cdn.discordapp.com/attachments/1233802893786746880/1431096595969085470/roofline.png?ex=68fcd4d0&is=68fb8350&hm=fa33212634f6c98c5803e39e32890019b31f4b484d48e2e130536b71f937bc64&) suggests an enhancement in performance metrics related to the cooperative matrix operations.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1431257720333402203)** (3 messages): 

> `Grayscale B200, Grayscale H100, Grayscale A100, Grayscale L4, Prefixsum A100` 


- **Grayscale Sweeps Second Place on B200 Leaderboard**: One member achieved **second place** on the `grayscale_v2` leaderboard, clocking in at **6.79 ms** and then **6.71 ms** on **B200**.
   - The two submissions were respectively id `66248` and `66250`.
- **Grayscale Secures Second Spot on H100**: A member achieved **second place** on the `grayscale_v2` leaderboard, with a time of **13.0 ms** on **H100**.
   - The two submissions were respectively id `66248` and `66250`.
- **Grayscale Takes Third on A100**: A member achieved **third place** on the `grayscale_v2` leaderboard, recording **20.5 ms** and then **20.4 ms** on **A100**.
   - The two submissions were respectively id `66248` and `66250`.
- **Grayscale Runs Successfully on L4**: A member achieved **second place** on the `grayscale_v2` leaderboard, with a time of **27.9 ms** on **L4** with submission id `66248`, followed by a successful run at **28.2 ms** with submission id `66250`.
   - This demonstrates consistent performance across different hardware configurations.
- **Prefixsum Achieves First Place on A100**: Another member claimed **first place** on the `prefixsum_v2` leaderboard, achieving a time of **7.20 ms** on **A100** with submission id `66267`.
   - This showcases the member's proficiency in optimizing parallel algorithms.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1431352756014153961)** (1 messages): 

> `Factorial Learning Environment, Reinforcement Learning Projects` 


- **Factorial Learning Environment Excites RL Enthusiast**: A member expressed excitement about the **Factorial Learning Environment (FLE)**, describing it as a very exciting long horizon benchmark from what they heard on a podcast.
   - They come from a **reinforcement learning** background and are interested in getting involved in **RL/self-improving system projects** related to FLE.
- **RL Enthusiast Seeks Involvement in FLE Projects**: An individual with a background in **reinforcement learning (RL)** expressed interest in contributing to **Factorial Learning Environment (FLE)** projects.
   - Inspired by its description in the Latent Space podcast, they are seeking opportunities to engage in **RL/self-improving system projects**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1431314157390790836)** (5 messages): 

> `Nsight Python, CUTLASS Python stack, CuTE talk slides` 


- **Nvidia releases Nsight Python for Python kernel development**: Nvidia has announced that **Nsight Python** will greatly improve Python kernel development and provided [a link to sign up for early access](https://developer.nvidia.com/nsight-python-notify-me).
   - They plan to have some tutorials with their **CUTLASS Python stack** once it's public.
- **Members seek CuTE talk slides**: Members are seeking slides from Chris's **CuTE talk**.
   - One member noted that the video description on YouTube had slides when they initially livestreamed, but they have been removed since.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1431178510952562738)** (6 messages): 

> `SITP, picograd, lazy semantics, torchdynamo, EagerTensor vs LazyTensor` 


- **Lazy Semantics Makes Waves in SITP and picograd**: Given **SITP** and **picograd's** northstar of pedagogical implementations, **tinygrad's** design decision of **lazy semantics** is actually very nice due to its minimal design.
   - It was reported that the only downsides from the **pt2 paper** are overheads, but this is perfectly fine for the pedagogical goals of **SITP** and **picograd**.
- **Torchdynamo Tracers a No-Go for picograd**: Implementing tracers at the host<->eDSL level (ops with torchfx) or at the host level itself (**python** with **torchdynamo**) is definitely a no go for **picograd**.
   - It's like students getting bogged down in LL, LL(1), LR parsers before getting into the meat of the optimizer and code generator for compiler construction, referencing [shriram krishnamurthis's PLAI](https://plai.org) which ducks parsing with s-exprs.
- **Eager Mode Gets Shoehorned/Retrofitted for a Smooth Transition**: It's important for readers to construct their own understanding to start with an **eager mode** and understand why transformers (**scaling laws**) and **tensor cores** necessitated the need for compiler pipelines like pt2 and xla.
   - The question of whether **SITP/picograd** should implement two separate structs like `EagerTensor` and `LazyTensor` under one `Tensor` or interpret and compile the IR which is `Graph<UOp>` was posed.
- **Picograd Taking a Breadth-First Approach**: It is understood that **picograd** requires more energy to lift off the ground compared to other autograds because it's taking a breadth-first approach with an autograd + compiler pipeline.
   - The poster invited anyone interested in helping turn **SITP** and **picograd** into the second course of Karpathy's starfleet academy after llm101 to join in on the fun.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1431105961757769818)** (43 messages🔥): 

> `H100 availability, Hackathon Waitlist, Dynamic SASS Kernel Instrumentation with nvbit, Memory Allocators on GPU, PyTorch Distributed Hacking` 


- **Nebius Doesn't Sling H100s**: A member asked about getting **H100s** from Nebius, but was told that they don't offer them, but can be rented for about **$1.90** an hour from other cloud providers.
   - This provides an alternative for those needing **H100s** for their projects during the hackathon.
- **Hackathon Waitlist Woes**: Two members requested assistance in getting off the waitlist for the hackathon, hoping to experiment with **multi-node GPU training** for climate use-cases.
   - They had filled out the form and were eager to join their teammate already in attendance.
- **Dynamic SASS Kernel Instrumentation Incoming**: A member is working on using **nvbit** to dynamically instrument **SASS kernels** to discover pointer offsets in their parameter/argument buffers.
   - This is particularly useful for their "parameterized cuda graph launch" idea in **PyTorch**.
- **GPU Memory Allocator Mini-PyTorch**: A member wants to write a "mini-version of **PyTorch**" with tensor metadata and allocator on the **GPU**.
   - They proposed that kernels should work with **512 threads** in a block and are looking for collaborators.
- **Quantized Pretraining on Blackwell Buzz**: A member is working on quantized pretraining on **Blackwell** and is looking for others interested in chatting.
   - Another user expressed interest in **AI-generated GPU kernels** and **kernel optimizations** for **Blackwell**.


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/)** (1 messages): 

erichallahan: New spec update
https://www.phoronix.com/news/Vulkan-1.4.330-Released
  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1431358706498404362)** (1 messages): 

> `NPU, CPU Offloading` 


- **Framework NPU Frustrations Spark CPU Offloading Quest**: A member reports failure to get the **framework machine** working for the **NPU**, pivoting focus to **CPU offloading**.
- **CPU Offloading Gains Traction**: With NPU efforts stalled, focus shifts towards exploring and optimizing **CPU offloading** techniques.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1431130492669001901)** (6 messages): 

> `Helion vs Triton, Cudagraph support, Kernel hyperparams` 


- **Helion Gains Ground on Triton After Compiler Improvements**: After some compiler improvements, a member noted they saw changes to their compiler in response to internal numbers, but were unsure if they reran the **Helion/Triton** numbers for comparison.
   - They mention the importance of benchmarking in the same environment with the same clock speeds.
- **Cudagraphs Support is Universal**: **Cudagraphs** are supported unless you do something in your kernel that is not cudagraphable.
   - The same cudagraphs restrictions that apply to other languages apply to **Helion** to preserve control flow from user.
- **Kernel Hyperparam Tuning Boosts Performance**: A member updated the **int4_gemm** references in [this commit](https://github.com/pytorch/helion/pull/1010) and also updated the blog post with the new numbers in [this blog post](https://pytorch.org/blog/helion/).
   - Another member linked to [this commit](https://github.com/tile-ai/tilelang/commit/8a5eb569704bfea64478c29adcfe3a09e3c2b12c) that lifts performance with both kernel and backend changes but no change to the autotuning param set.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1431051215797489744)** (63 messages🔥🔥): 

> `zero3 config, Text-SAL, AI infrastructure collaboration, ROMA (Reasoning Over Multiple Agents), synthetic data gen` 


- **Zero3 config must be busted**: A member said *your zero3 config must be busted* and thus you should be able to do much larger training on **r=8 lora for gemma 3 1b**.
   - They added that *something is definitely wrong*.
- **Text-SAL run finished**: A member posted a log output of a **Text-SAL** run and asked what framework this is and what the training method is.
   - The log mentioned **SAL_BRIDGE**, indicated a BERT model (**prajjwal1/bert-tiny**), and showed energy and memory states during training.
- **Sentient Community seeks AI infrastructure collaboration**: A member from the **Sentient community** inquired about collaboration or partnerships related to **AI infrastructure** or **verifiable AI systems** with Hugging Face.
   - Another member found their project interesting and linked to their [ROMA (Reasoning Over Multiple Agents) GitHub repository](https://github.com/sentient-agi/ROMA?tab=readme-ov-file).
- **ROMA Explainer**: **ROMA (Reasoning Over Multiple Agents)** is designed to break complex tasks into smaller, specialized subtasks handled by multiple **AI agents**.
   - That modular setup helps overcome context limits and boosts reasoning efficiency, as each agent (or “cell”) handles part of a bigger picture, then everything is composed back together.
- **Synthetic Data Generation Discussion**: A member is looking to explore **synthetic data generation** but doesn't have any ideas or a starting point.
   - Another member mentioned that they *have seen all kinds of neat ideas for graphics stuff, but not so much language*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

waffles1: Ah yes this is totally legit
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1431085728506576939)** (4 messages): 

> `Pacific-Prime model, 6GB VRAM check, Zero Amnesia AI, Night Learn Engine, RAG Pipeline` 


- **Pacific-Prime model boasts 10% gain**: The [Pacific-Prime model](https://huggingface.co/Pacific-Prime/pacific-prime) is reported to have an updated **10% gain** from 6GB VRAM, starting from a 1.1B parameter model.
- **Zero Amnesia AI with True Memory**: An AI system is described as having **true memory** with *zero amnesia*, retaining past conversations and important details as context-rich memories.
- **AI's Personality Shaped on the Fly**: The AI allows users to adjust its **identity on the fly**, ranging from a professional collaborator to a creative sparring partner.
- **Night Learn Engine Evolves Autonomously**: The AI incorporates a **Night Learn Engine** that reflects on interactions, consolidates the day's information, builds higher-order memories, and evolves autonomously.
- **Refined RAG Pipeline retrieves context-aware intelligence**: The AI utilizes a refined **RAG pipeline** to retrieve only what is essential for a task, ensuring precise, context-aware intelligence without data chaos.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

yusarseph: hello, is hugface inference endpoints servless ? do we pay for what we dont use ?
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1431146620975579176)** (2 messages): 

> `Karpathy Server, HF, nanochat-students, MLX Porting, MLX Stability` 


- **Clarification on Server Context Needed!**: A member asked for clarification whether the discussion was about the **Karpathy server** or **Hugging Face**.
   - The member also inquired about the goal of **nanochat** or the **nanochat-students organization** on the Hub.
- **MLX Porting Thoughts**: A member expressed interest in potentially porting the project to **MLX**.
   - They inquired about the stability of the material to gauge whether they should wait before proceeding.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1431070532488659045)** (5 messages): 

> `Agents course unit 4, 404 Error` 


- **Agent Course Unit 4 has 404 Errors**: Users reported a **404 error** when trying to access questions via *https://agents-course-unit4-scoring.hf.space/questions*.
   - The error message displayed was *"No questions available."
- **Questions Unavailable**: Multiple users have reported that the questions for the Agents course unit 4 are unavailable.
   - The issue has persisted since yesterday evening, with users encountering a **404 error** when attempting to access the questions.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1431065896788562012)** (17 messages🔥): 

> `Server Acceptance Process, Distributed Inference, AI Ownership, AI Accelerator Chips, Petals Project` 


- **Server Acceptance: Pending Approval**: A member stated that access to a server is granted only post-acceptance, often involving a form, but the 'join' option bug has been fixed.
   - Another user confirmed being a *pending member* before approval.
- **Distributed Inference: The Future of AI**: Members advocate for open-source, widely distributed AI, akin to the internet's structure, moving away from mega-corporation dominance, similar to [Nous Research](https://www.nousresearch.com/)
   - One member pointed out *serious technical problems* hindering this vision, like GPU resources contributions being non-trivial.
- **Nvidia's Space Ambitions: Inferior Chip Design?**: Members are speculating that **Nvidia's** plan to put **GPU clusters** in space is a sign of clinging to their *inferior chip design*.
   - They anticipate more **energy-efficient** and **cost-effective** alternatives will soon dominate the market.
- **Unbearable Slowness of Being: A Research Paper**: A member asked if technical problems in AI designs stem from capturing *The Unbearable Slowness of Being*, referencing [this paper](https://arxiv.org/abs/2408.10234v2).
   - No further details were provided.
- **Petals Project: Distributed Inference**: The now seemingly abandoned [Petals Project](https://github.com/bigscience-workshop/petals) was mentioned as having momentum 2 years ago for **llama 70b**.
   - The community fell adrift when the project could not keep up with new architectures, and *LlamaCPP RPC* is now the closest thing to it.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1431073478965989407)** (54 messages🔥): 

> `50M model Loss, 1B model Validation, lm-eval, activation steering` 


- **New 50M Model Achieves Low Loss**: A new member reported their **50M model** achieved a loss of **0.223**, significantly lower than a vanilla transformer's **2.73** loss, and their **1B model** is already sub **0.3** at **400 steps**.
   - Skepticism arose due to the unexpectedly low loss, with some suggesting it's *either a bug, a trivial dataset, or a lie*.
- **Model Debugging Requires Code**: Community members requested the model's code to debug, suggesting the reported performance might be incorrect.
   - The original poster (OP) declined due to **IP** reasons, but promised to post results of running the **1B model** through the standard **lm-eval** harness.
- **Validation is key to proper claims**: Community questioned the validity of an allegedly groundbreaking model that could run on a cell phone.
   - One member said that the original poster (OP) hadn't ruled out other basic issues, and should stick around for a while to avoid making *fantastic claims*.
- **Activation Steering Exploits Gradient Reuse**: A member wondered if **activation steering** could allow for the reuse of datapoints to acquire a large variety of different gradients from them.
   - They cited [Anthropic's Personas paper](https://arxiv.org/abs/2506.18221) and another paper, linking the idea to the possibility of qualitatively controlling the kind of gradients returned after a forward pass.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

stellaathena: Okay what the hell is this nonsense: https://www.arxiv.org/abs/2510.15511
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1431106546468655198)** (31 messages🔥): 

> `gpt-4o-transcribe-diarize, GPT-5, Cursor Enterprise, Kimi Code CLI, Cohere's AI Win` 


- **OpenAI Quietly Drops gpt-4o-transcribe-diarize**: Peter Bakkum announced that OpenAI quietly dropped **gpt-4o-transcribe-diarize**, a 'small' audio model optimized for high-accuracy speaker diarization, which is large/offline-only and accepts voice samples to tag known speakers ([link](https://xcancel.com/pbbakkum/status/1981397851600302250?s=46)).
   - It has **WER** comparable to other OpenAI ASR models and users asked about benchmarks vs pyannote, real-time use, pricing, open-weights, and mini versions.
- **GPT-5 powers Company Knowledge**: OpenAI announced **Company Knowledge** is powered by a finetuned version of **GPT-5** that's trained to look across multiple sources to give more comprehensive and accurate answers ([link](https://openai.com/index/introducing-company-knowledge/)).
   - It is unknown if they will ever make this one available in the api.
- **Cursor's C-Suite Strategy Steals the Show**: Alex Konrad revealed Cursor’s aggressive enterprise strategy, with COO leading **300 C-suite meetings** in Q3 to support **$500M+ ARR** ([link](https://xcancel.com/alexrkonrad/status/1981477024092082386?s=46)).
   - They are using technical sales teams, customer hackathons, and code-half-life metrics; full Upstarts interview linked.
- **Kimi Code CLI Teaser Leaks, Hype Ensues**: Crystal playfully confirmed an image leak of **Kimi's upcoming CLI/Code tool**, noting the global release is a few days out while asking for patience ([link](https://xcancel.com/crystalsssup/status/1981597395541753988?s=46)).
   - Users flood replies with praise (comparison to Claude Code), requests for early access, free credits, Tomagotchi easter-eggs, and future WhatsApp integration.
- **Tahoe-x1: Open-Source Single-Cell Transformer Emerges**: Tahoe AI released **Tahoe-x1**, a 3-billion-parameter transformer that unifies gene/cell/drug representations and trains efficiently on their 100M-sample Tahoe perturbation dataset ([link](https://xcancel.com/nalidoust/status/1981760790551298524)).
   - It hits SOTA on cancer-relevant benchmarks and is fully open-sourced on Hugging Face with checkpoints, code and visualization tools.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1431119160825610343)** (5 messages): 

> `Local AI Apps, QA on Scanned PDFs, OpenWebUI, Qwen3-vl-4b` 


- **Local AI App Sought for Scanned PDFs**: A member inquired about a local AI app capable of performing question answering directly on a multi-page scanned PDF using a VLM like **Qwen3-vl-4b**.
   - It was noted that many apps support either images or retrieval-augmented generation (**RAG**) when uploading files, such as **LM Studio**.
- **OpenWebUI Suggested for PDF Prompting**: Another member suggested using **OpenWebUI** to feed an entire PDF as part of the prompt, referencing a setting to use either the entire document or only relevant parts.
   - However, the original poster reported that the selected VLM could not handle the scanned PDF in **OpenWebUI**.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1431164536357912658)** (7 messages): 

> `Video Models, MJ, Kling, LTX-2, a16z` 


- **A16z Thinks No God-Tier Video Model Coming**: [Justine Moore (a16z)](https://a16z.substack.com/p/there-is-no-god-tier-video-model) argues we’ll never have one universal video model; instead, a growing buffet of specialized models serves different budgets and use-cases, and the community reacted to this thesis.
   - Thread readers swap favorite models (**MJ**, **Kling**, **LTX-2**), debate vertical vs horizontal tooling, and liken the landscape to cameras or Baroque still-life styles—celebrating competition over monolithic supremacy.
- **No God-Tier Video Model Reactions Available on YouTube**: Good talk about the No God-Tier Video Model Thesis and Community Reactions are available on [YouTube](https://youtu.be/wHK8GMc9O5A?si=2W2N8W7cXjS7ppfK).
   - A user missed sharing this link earlier, but another user caught it.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1431039661760974969)** (28 messages🔥): 

> `Mythworx AI, ARC-AGI 1, Elastic Weight Consolidation, Activation-aware Weight Quantization (AWQ), Cherry-picked verifications` 


- **Mythworx Claims 100% on ARC-AGI 1**: [Mythworx.ai](https://mythworx.ai/capabilities/) claims **100%** on **ARC-AGI 1** with no pre-training within **4 hours**, sparking skepticism about their capabilities.
   - The claim was met with skepticism, as one member questioned *why is it that they always announce without validating with ARC private set* implying a pursuit of funding over rigorous validation.
- **ARC Private Set Validation Debate**: Community members debated the need for **ARC private set validation**, with one member suggesting misrepresentation as a way to get models evaluated, while another cautioned against it.
   - One member warned that misrepresentation could lead to being *blacklisted as a researcher*, while another suggested that it's about *mis represent until they get annoyed at you and then have to work with you to test the resutls*.
- **Elastic Weight Consolidation (EWC) Discussed**: A community member inquired whether a technique was simply **learning rate** for each weight, instead of the entire model, referencing **elastic weight consolidation**.
   - Another community member expanded on this, discussing the complexities of implementing it, particularly regarding the "softness factor" and the normalization of vectors, pointing to Activation-aware Weight Quantization (AWQ).
- **Cherry-Picked Verifications get called out**: A community member expressed preference for calling out frauds during conference presentations, citing cherry-picked verifications based on *whimsy and bribery*.
   - They claimed *Other researchers have already come along since then, verified that they were FOS, and been properly horrified.*


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1431341313462112317)** (7 messages): 

> `Transformer Circuits Linebreaks Paper, Neuronpedia Attribution Graphs, Gemma-2-2b Line Break Attribution, Qwen3-4b Line Break Attribution` 


- **Transformer Circuit Linebreaks Paper Dive**: A member suggested examining the [Transformer Circuits Linebreaks paper](https://transformer-circuits.pub/2025/linebreaks/index.html).
   - The suggestion included discussion on a specific date, which was corrected by another member.
- **New Neuronpedia Graphs Released**: A member announced the release of [line break attribution graphs](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D) related to a new paper.
   - The graphs enable examination of line break attribution in models like **Gemma-2-2b**.
- **Qwen3-4b Attribution Graphs Released**: A member announced the release of [line break attribution graphs](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39) related to a new paper.
   - The graphs enable examination of line break attribution in models like **Qwen3-4b**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1431161962863005797)** (3 messages): 

> `Genie 3 World Model, Google, David Sacks, Donald Trump` 


- **Genie 3 world model is here!**: The new **Genie 3** world model video generation is impressive seemingly because they have enough compute to offer it to a wide range of user when the other players still offer video creation in the max of a few seconds.
   - It's pretty in line with recent **Genie 3** world model videos.
- **Morron AI Zar David Sacks**: A member said *Don't worry that morron AI Zar David Sacks is going to do everything to get these thrown out by adminstrative pressure from the **Orange Dude**.*
   - The member was concerned about **David Sack's** political pressure on AI.
- **Google is failing**: A member said *Oh how the Chocolate Factory has fallen*
   - There was a picture of **Donald Trump** as an Oompa Loompa in the thread.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1431058834079875082)** (37 messages🔥): 

> `Chutes vs Moonshot AI, Kimi K2, Data Policy, Uptime, Tool call accuracy` 


- **Chutes Data Policy Questioned**: A member asked about **Chutes** compared to **Moonshot AI** for the **Kimi K2**, and another responded that *Chutes* trains on user data, lacks a data policy, has less reliable uptime, and lower tool call accuracy compared to the official **Moonshot AI API**.
- **Chutes Reddit Banter Acknowledged**: The community noted that **Chutes** addressed the memes about banning **Chutes** on **OpenRouter** after a benchmark post brought attention to the issue, with one user sarcastically pointing out the attractive pricing and speed despite these caveats.
- **Kimi Adopting GLM Coding**: A member expressed a wish for **Kimi** to adopt a **GLM** coding plan, or a similar style, because *GLM is more cost-effective for coding plans, and GLM-4.6 is far more powerful than Kimi*.
- **Chinese Kimi.com Integrated Plans Launched**: A member noted that a similar product to Kimi was launched in China and integrated into the Chinese Kimi website, posting a link from [X.com](https://x.com/bigeagle_xd/status/1981568257258860899) and a link to the [Kimi-Cli on Github](https://github.com/MoonshotAI/kimi-cli).
- **Localized Kimi Pricing Looks Cheap**: Members noticed that the Chinese pricing for **Kimi** looked quite cheap, though it was cautioned that it's localized pricing and the international market prices may differ.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1431045284732862544)** (35 messages🔥): 

> `Manus Network connection error, Manus credits usage, Claude Code vs Manus, Manus Room database, Manus deprecated code` 


- ****Manus** Network Error Frustrates Users**: Users are encountering *"Network connection error"* issues with **Manus**, hindering their ability to code apps.
   - The error message states: *"Please check your network settings and try again."*
- ****Manus** Credit Consumption Criticized**: Users express concern over the high credit consumption in **Manus**, with one user reporting **15,000 credits** spent on a complex project in just a few days, hoping the new version is more effective.
   - Others suggested doing research and using other AI to fix the generated code, warning against *"paying for bad coding."*
- ****Claude Code** and **Codex** touted as **Manus** Alternatives**: Users are recommending **Claude Code** and **Codex** as better alternatives to **Manus**, citing better development capabilities and cost-effectiveness, costing about **$20/month** for serious development time.
   - A user pointed out that with **Claude Code** you get 5hr sessions that reset and weekly rate limiting, which at the end, gives you easily 5x more than what you get from **Manus**.
- ****Manus**'s **Room** Database Implementation Faulty**: **Manus** claims to have implemented a **Room** database for previous chat history, but a user found it to be completely unimplemented.
   - According to **Claude**, there was *"❌ No Room database class, ❌ No entities (@Entity), ❌ No DAOs (@Dao), ❌ No history UI, ❌ No history icon in any arena screen"*.
- ****Manus** Generates Deprecated Code and Has Build Issues**: Users report that **Manus** generates deprecated code with security issues, and recommend telling it *"to update deprecated code/packages/modules/plugins"*
   - One user mentioned that **Manus** claims the build is okay, but running the build reveals many errors and warnings.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1431111666828841052)** (15 messages🔥): 

> `Gemini Pricing, Aider vs. Codex, aider-ce Community Fork, RAG with GitHub Copilot` 


- **Gemini's Generous Pricing**: For around **$20 USD** a month, **Gemini** offers nominally **1500 requests a day** with a **1M token context window** using the **2.5 model**, accessible via the *gemini-cli*.
   - Authentication relies on a **Google Cloud project** linked to an **API billing account**, and while the interface is mostly better than aider's, it lacks a repo map and relies on file system operations like grep.
- **Codex Wins Over Aider**: A member found **Codex** (with a regular **ChatGPT Plus** account using the **gpt-5-codex** model) surprisingly effective, reducing the need for **aider**'s manual context files.
   - They noted that since *aider* is hardly being developed anymore, they found codex suitable, despite previously being a *aider-power-user*.
- **Community Develops aider-ce**: A community member suggested trying [aider-ce](https://github.com/dwash96/aider-ce), a community-developed fork of **aider** with *navigator mode* for more agentic behavior.
   - It also includes additional features like **RAG** (Retrieval-Augmented Generation), with a PR from MCPI.
- **GitHub Copilot RAG**: With a **GitHub Copilot** subscription (**$10 a month**), users gain access to infinite **RAG**, infinite **gpt 5 mini**, **gpt4.1**, **grok code 1** and **300 requests** of **claude sonnet 4**/**gpt5**/**gemini 2.5 pro**, and **900** of **haiku**/ **o4-mini**.
   - This offers a robust set of tools for coding and generation tasks.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1431359186263871611)** (1 messages): 

> `Aider's future and development, aider-ce feature set` 


- **Aider's Future Outlook and Development**: A member expressed strong support for **aider** and inquired about its future development plans and longevity.
   - They noted that **aider** is their preferred AI coding tool due to its intuitive workflow and expressed hope for its continued success and feature improvements.
- **Aider-ce Feature Set**: The discussion touched upon **aider-ce**, a variant of **aider** with additional merged features.
   - A member highlighted that while **aider-ce** includes more features, it has significantly fewer stars compared to the original **aider** project.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1431044191978393810)** (4 messages): 

> `ChatGPT emoji bug, Unreal Engine competitor` 


- **ChatGPT has emoji bug**: Members discovered that asking **ChatGPT** *“is there an emoji of a seahorse?”* causes it to bug out.
- **New Unreal Engine competitor appears**: Members speculate on a new competitor to **Unreal Engine**, **Runway**, and **Wan** after seeing new demos.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1431163324854177994)** (4 messages): 

> `Nous Research Models, YARN Context Scaling, Western Ideological Views in GPT` 


- **Nous Extends Beyond HF Profile**: Aside from the models in the [hf nous profile](https://huggingface.co/nous-research), there aren't additional models directly associated with **Nous Research**.
   - However, **Nous Research's** researchers contributed to **YARN context scaling**, a technique implemented in multiple models.
- **YARN Scales Context Windows**: **YARN context scaling** is found in several models due to the contributions of **Nous Research** researchers.
   - No further details or links were shared about this scaling method.
- **GPT's Western Ideological Leanings**: There's a suggestion that **GPT** models originating from the West may reflect **Western ideological views** more strongly.
   - *Data is really important to shape your worldview* and can lead to interesting differences in AI models.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1431337233688039496)** (2 messages): 

> `Mech Interp Surgeon's Bag, RL Desirability Questioned` 


- **Mech Interp Bagpack Required for Scale Limits**: A member stated that *they need to finish the **mech interp surgeon's bag** before we can talk confidently about the limits of scale*.
   - Another member requested links to papers that show critiques of RL.
- **RL method questioned, papers come forward**: Members shared that *several papers are raising the question as to whether **RL** is even desirable or necessary* this year.
   - Others asked for links to these papers to read more about it.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1431219042529710163)** (1 messages): 

> `MARL Consensus, Hamiltonian Path Problem, BFT consensus algorithm` 


- **MARL Consensus Speculation on X**: A member shared a speculative [post on X](https://twitter.com/op/status/176767) regarding **MARL (multi-agent consensus)**.
   - The post posits that **Among Us Uno Consensus** can function as a **BFT consensus algorithm** with a byzantine resistant majority (majority honest players).
- **UNO is NP-Complete**: A member claimed a single-player version of **UNO** is a **Hamiltonian Path Problem**, which is a classical **NP-complete problem** (graph-coloring-pathfinding).
   - This complexity arises due to the presence of *"choices" and "randomness"* in the game.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1431337233688039496)** (2 messages): 

> `Mech Interp Surgeon's Bag, RL Desirability, Limits of Scale` 


- **Mech Interp Surgeon's Bag Precedes Scale Discussion**: A member mentioned that completing the **Mech Interp Surgeon's Bag** is essential before confidently discussing the limits of scale.
   - This suggests a need for comprehensive interpretability tools to understand scaling dynamics.
- **Rethinking RL's Desirability and Necessity**: A member noted that several papers this year are questioning whether **Reinforcement Learning (RL)** is desirable or even necessary.
   - Another member requested pointers to these critiques, showing interest in the debate around RL's value.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1431138952412397709)** (8 messages🔥): 

> `Becoming a Tinygrad Dev, Mojo and AI Compilers, AI Box Recommendations` 


- **Unlock Tinygrad Dev Secrets**: A member asked how to become a **tinygrad dev**, and another member linked to a helpful [blog post](https://ninoristeski.github.io/blogs/how-i-started-contributing-to-tinygrad.html) about contributing.
   - They added that the Discord server is a resource for learning more about contributing to **tinygrad**.
- **Mojo Rising as AI Compiler Candidate**: A member shared multiple [Modular Blogposts](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers) about **Mojo** and other AI compilers.
   - They mentioned *Mojo* is higher level than CUDA, but way lower level than eDSLs like triton/tilelang, and it's far too Turing complete.
- **Tinybox Mobo Specs: Any Advice?**: A new member from France is seeking advice on the **mobo** of the **tinybox**.
   - He asked whether it could support **9005 with 12 DIMMs and a 500W CPU**.


  

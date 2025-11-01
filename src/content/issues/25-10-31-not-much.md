---
id: MjAyNS0x
title: not much happened today
date: '2025-10-30T05:44:39.731046Z'
description: >-
  **Poolside** raised **$1B** at a **$12B valuation**. **Eric Zelikman** raised
  **$1B** after leaving **Xai**. **Weavy** joined **Figma**. New research
  highlights **FP16** precision reduces training-inference mismatch in
  **reinforcement-learning** fine-tuning compared to **BF16**. **Kimi AI**
  introduced a hybrid **KDA (Kimi Delta Attention)** architecture improving
  long-context throughput and RL stability, alongside a new **Kimi CLI** for
  coding with agent protocol support. **OpenAI** previewed Agent Mode in ChatGPT
  enabling autonomous research and planning during browsing.
companies:
  - poolside
  - x-ai
  - figma
  - openai
  - kimi
  - moonshot
models: []
topics:
  - reinforcement-learning
  - precision
  - fp16
  - bf16
  - linear-attention
  - long-context
  - cli
  - agent-frameworks
  - coding-agents
people:
  - eric_zelikman
---


**a quiet day**

> AI News for 10/30/2025-10/31/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (198 channels, and 5603 messages) for you. Estimated reading time saved (at 200wpm): 512 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

this spooky night, a lot of scattered news items but nothing clearly headline inducing.

- Poolside [raised $1B at $12B valuation](https://x.com/julienblanchon/status/1984337407097909629?s=46)
- [Recap of Vercel Ship](https://www.youtube.com/watch?v=RXx5ZN69Z3E) with Malte Ubl
- [Eric Zelikman raised $1b after leaving Xai](https://x.com/annatonger/status/1984318774208782467?s=46)
- [Weavy joined Figma](https://x.com/figma/status/1983889394944692359?s=46)

---

# AI Twitter Recap

**Precision wars in RL fine‑tuning: FP16 vs BF16**

- **FP16 to fix training–inference mismatch (RL for LLMs)**: New work argues BF16 causes significant rollout policy drift when training on one engine and sampling on another; simply switching to **FP16** substantially reduces numerical divergence thanks to its 10 mantissa bits (vs BF16’s 7). The paper provides code and analyses; early repros echo better stability and reward with FP16, though you’ll need loss scaling and care around dynamic range. See summary and links by [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1984193217617895597), author thread by [@QPHutu](https://twitter.com/QPHutu/status/1984258808332550245), and code at Precision-LLM ([repo](https://twitter.com/iScienceLuvr/status/1984193219576602874), [abs](https://twitter.com/iScienceLuvr/status/1984193219576602874)). Community reactions range from “serve-what-you-train BF16” hardliners to enthusiastic converts: [@rosinality](https://twitter.com/rosinality/status/1984113018867941493), [@natolambert](https://twitter.com/natolambert/status/1984262505443844263), [@_xjdr](https://twitter.com/_xjdr/status/1984138487772414250), [@ArmenAgha](https://twitter.com/ArmenAgha/status/1984167109895844106) (notes gradient clipping + loss-scaling bugs), [@rasbt](https://twitter.com/rasbt/status/1984279418588762113) (QKNorm/stability caveats), [@agarwl_](https://twitter.com/agarwl_/status/1984416235774247273) (A100 vs H200 behavior), and practical nits from prod trainers ([@shxf0072](https://twitter.com/shxf0072/status/1984175419718078866), [@suchenzang](https://twitter.com/suchenzang/status/1984162915285659899)). Bottom line: FP16 can shrink the train–serve gap for RL loops; guard rails still include robust loss scaling, selective FP32 for sensitive params, and normalization to avoid overflow.

**Kimi AI: Hybrid Linear Attention and a coding CLI**

- **Kimi Linear (KDA) architecture insights**: Kimi details a hybrid **KDA (Kimi Delta Attention)** design (Delta‑style linear attention with fine‑grained gating) that replaces most global attention to unlock long‑context throughput and stable RL post‑training. A 3:1 KDA:MLA hybrid was picked via ablations; with identical 5.7T tokens and ~3B active params, the team reports markedly better pretrain perplexity, long‑context evals (MRCR/RULER/Frames), and downstream math/code after RL, with ~6× decoding speed from smaller KV caches. Training notes: repeated reversions to debug long‑ctx evals, selective FP32 on critical bias vectors mid‑run to stop drift, and a “Scaling Ladder” process (gate each scale before moving up). See technical retros by Kimi engineers ([@zy27962986](https://twitter.com/zy27962986/status/1984079705809789216)) and a comprehensive Chinese write‑up ([@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1984321210055082207), commentary by [@eliebakouch](https://twitter.com/eliebakouch/status/1984291165860958614), [follow‑up](https://twitter.com/eliebakouch/status/1984293535110017476)).
- **Kimi CLI and Coding**: Moonshot released a terminal‑native **Kimi CLI** (shell‑like UI, Zsh integration, **MCP** + Agent Client Protocol support) and “Kimi For Coding” as a VIP add‑on at no extra cost ([announcement](https://twitter.com/Kimi_Moonshot/status/1984207733177090274), [docs/feedback](https://twitter.com/Kimi_Moonshot/status/1984207741037252751), [benefits](https://twitter.com/Kimi_Moonshot/status/1984207737673359441)). Internal reflection on whether “the world needs one more code‑CLI”—they’re betting on an opinionated coding‑agent baseline that improves incrementally ([team](https://twitter.com/bigeagle_xd/status/1984217403023380802)).

**Agent frameworks, memory, and dev toolchains**

- **OpenAI Agent Mode (preview)**: ChatGPT can now “research, plan, and get things done” while you browse; enabled for Plus/Pro/Business ([OpenAI](https://twitter.com/OpenAI/status/1984304194837528864)). You can try it in Atlas ([demo](https://twitter.com/gdb/status/1984304783881355451)); early testers want more resilient complex DOM ops ([feedback](https://twitter.com/omarsar0/status/1984304979671224702)).
- **LangChain Deep Agents CLI and Agent Builder**: A batteries‑included agent harness with memory and opinionated defaults, plus the Agent Builder in LangSmith; both aim to accelerate long‑horizon, tool‑using agents ([CLI](https://twitter.com/hwchase17/status/1984303925101735950), [Builder](https://twitter.com/GitMaxd/status/1984306847856410953)). LangChain also earned AWS GenAI Competency; LangSmith is on AWS Marketplace ([announcement](https://twitter.com/LangChainAI/status/1984303566723625044)).
- **VS Code & Cline updates**: VS Code adds an Agent Sessions view to manage local/cloud sessions ([VS Code](https://twitter.com/code/status/1984322058503807066)). Cline v3.35 switches to native tool calling across major providers, cuts token overhead (~15%), and enables parallel tool exec ([changes](https://twitter.com/cline/status/1984306206538940702), [details](https://twitter.com/cline/status/1984334385626411397)). LlamaIndex shipped native MCP search endpoints for docs ([LlamaIndex](https://twitter.com/llama_index/status/1984292554968616994)).
- **Agent memory and orchestration**: A community MCP bridge writes conversational embeddings to Qdrant to create persistent cross‑tool memory ([Qdrant](https://twitter.com/qdrant_engine/status/1984138269626421490)). Trend toward open, self‑hosted GPU orchestration continues—check **dstack** (MPL‑2.0) if you’re avoiding lock‑in ([@andrey_cheptsov](https://twitter.com/andrey_cheptsov/status/1984136998190510280)).

**Training playbooks and infra updates**

- **Hugging Face “Smol Training Playbook” (214 pp.)**: A dense, practical guide to the full LLM pipeline: tokenization, attention variants (MQA/GQA/MLA), positional encodings (RoPE/yarn/NoPE), stability tricks (z‑loss/QK‑Norm), MoE scaling (granularity/load balance), SSM hybrids for long‑ctx, curricula and adaptive data mixes, mid‑training interventions, and post‑training (SFT → DPO/KTO/ORPO → RLHF). Also deep infra guidance (DP/TP/PP/FSDP; NVLink/IB/GPUDirect) and prod gotchas (shape mismatches, data shuffle bugs) ([overview](https://twitter.com/TheAhmadOsman/status/1984157512795357614), endorsements: [1](https://twitter.com/andimarafioti/status/1984220766850916443), [2](https://twitter.com/JayAlammar/status/1984273218568696014)).
- **Optimizers and logs**: “Muon” is now in PyTorch stable (widespread interest from training folks) ([@kellerjordan0](https://twitter.com/kellerjordan0/status/1984102608781636008)). Google AI Studio added logs and dataset export for evaluations—no‑code enablement and CSV/JSONL export ([@_philschmid](https://twitter.com/_philschmid/status/1984258488013340826)).
- **Licensing and CI**: MergeKit returned to **LGPL** for commercial use ([@latkins](https://twitter.com/latkins/status/1984320609015513605)). Modal is sponsoring GPUs for multi‑GPU CI in ArcticTraining; fast boot with pytest‑xdist ([@StasBekman](https://twitter.com/StasBekman/status/1984293939583856751)).

**Model and research releases**

- **Reasoning and attention**:
    - Supervised Reinforcement Learning (SRL): Uses expert trajectories to build step‑wise internal reasoning and rewards via action similarity; reported to outperform SFT and RLVR on math and agentic coding with Qwen2.5 backbones ([paper](https://twitter.com/IHung_Hsu/status/1984077573383712934), digest: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1984188333258592590)).
    - HLA: Higher‑order linear attention with parallelizable training—“attention vibes + RNN speed” ([@yifan_zhang_](https://twitter.com/yifan_zhang_/status/1984099671657304207)).
    - ByteDance LoopLM (Ouro): Small decoder‑only models (1.4B/2.6B) with T recurrent steps for latent multi‑hop reasoning and learned early exit; strong per‑parameter under memory/KV constraints, but untied deeper standard Transformers win per‑FLOP in compute‑matched tests ([technical analysis](https://twitter.com/scaling01/status/1984286236438094307)).
- **Multimodal and domain**:
    - Emu3.5: “Native multimodal” model that adds diffusion image generation despite NTP training; claims parity to “Nano Banana” on generation/editing; open‑weights/code available ([summary](https://twitter.com/iScienceLuvr/status/1984190340279234888)).
    - Brain‑IT: fMRI→image reconstructions from just 15 minutes of data via a Brain‑Interaction Transformer with voxel clustering and synthetic fMRI training ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1984195725253804449)).
    - NVIDIA Nemotron: New RAG suite includes text and multimodal retrievers and layout detectors with permissive licenses ([overview](https://twitter.com/mervenoyann/status/1984302303570960666)), and Nemotron Nano 2 VL now runs on vLLM ([vLLM](https://twitter.com/vllm_project/status/1984334926972592193)).
- **Qwen ecosystem**: **Qwen 3 Max Thinking** released ([@legit_api](https://twitter.com/legit_api/status/1984284268412191216)), and **Qwen3‑VL** models are live in LM Studio ([LM Studio](https://twitter.com/lmstudio/status/1984330903880155154)).

**Top tweets (by engagement)**

- **PewDiePie goes full local AI lab**: 10×4090 rig (incl. “4090D 48GB”), runs Llama 70B, gpt‑oss‑120B, Qwen‑245B via vLLM; custom chat/RAG/search/TTS UI; swarm‑of‑64 council; now fine‑tuning his own model—open‑source stack front and center ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1984309989134254493), [alt](https://twitter.com/birdabo/status/1984288466952433739)).
- “One day Vik’s manager said … design patterns.” Visitor pattern for promotion—satire of cargo‑cult complexity in SWE ([@vikhyatk](https://twitter.com/vikhyatk/status/1984110677007700098)).
- “There is a reason why most large companies are doomed. They value complexity.” ([@rakyll](https://twitter.com/rakyll/status/1984107025845121158)).
- “The government does, in fact, use SQL.” ([@stupidtechtakes](https://twitter.com/stupidtechtakes/status/1984124850575962280)).
- OpenAI’s Agent Mode announcement ([@OpenAI](https://twitter.com/OpenAI/status/1984304194837528864)).
- RL precision meme: “every decommissioned V100 coming out of retirement after hearing that the future of RL is fp16” ([@tenderizzation](https://twitter.com/tenderizzation/status/1984271620027118029)).
- “There is almost no other cause … increases your odds of needing psychiatric medication more than being 5 years into your PhD” ([@LinkofSunshine](https://twitter.com/LinkofSunshine/status/1984301915300118893)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

*no posts passed our bar.*

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI-Generated Content in Media

- [**Completely made with AI**](https://www.reddit.com/r/ChatGPT/comments/1oko2l4/completely_made_with_ai/) (Activity: 7176): **The post discusses the use of various AI tools in film production, highlighting Midjourney, Hailuo 2.0, Kling, Adobe Firefly, Magnific, Enhancor, and Elevenlabs. The creator, Chris Chapel, demonstrates how AI can enhance film quality, suggesting that as technology progresses, AI-generated content will become increasingly indistinguishable from reality. The post anticipates a future where AI is integral to filmmaking, despite current resistance from some directors.** The comments reflect skepticism and humor, with one user joking about being 'scammed' in the future and another predicting legal issues for the original poster. There's a recognition of the surprising quality of AI-generated content, suggesting a shift in perception as AI tools improve.
- [**I asked chatgpt to create a pic of every flag in the world with the corresponding name underneath**](https://www.reddit.com/r/ChatGPT/comments/1oku5ht/i_asked_chatgpt_to_create_a_pic_of_every_flag_in/) (Activity: 2084): **The image is a humorous depiction of an AI-generated attempt to create flags for every country in the world, with the names of the countries written underneath. However, the flags and names are fictional or altered versions of real ones, highlighting the limitations of AI in generating accurate and culturally specific content. The post humorously points out that only the Swedish flag was correctly identified, suggesting a failure in the AI's ability to accurately replicate or recognize national symbols and names.** Commenters humorously criticize the AI's spelling and flag design, with one user joking about a fictional country name 'SoeijÖÄyc', illustrating the AI's inaccuracies.

### 2. OpenAI IPO and Policy Changes

- [**OpenAI will be the first non-profit to IPO**](https://www.reddit.com/r/OpenAI/comments/1oksww0/openai_will_be_the_first_nonprofit_to_ipo/) (Activity: 2874): **The image is a meme that humorously critiques the potential IPO of OpenAI, a company originally founded as a non-profit with the mission to advance AI for the benefit of humanity without financial constraints. The post highlights the irony of a non-profit organization considering an IPO, which would shift its focus towards profit generation. This potential shift has sparked debate about the integrity of OpenAI's original mission and the influence of financial incentives on its operations.** Commenters express skepticism and concern over OpenAI's potential IPO, suggesting that the pursuit of profit could compromise its original mission to benefit humanity without financial obligations.
    - OpenAI's transition from a non-profit to a for-profit entity has sparked debate, with some users highlighting that the organization's original mission was to advance AI for the benefit of humanity without financial constraints. This shift raises concerns about the potential change in priorities towards profit generation, which could impact the focus on positive human impact and equitable AI distribution.
    - The conversion of OpenAI to a for-profit company is not recent, as noted by some users. OpenAI has already been operating as a for-profit entity for some time, which suggests that the IPO is a continuation of this strategic direction rather than a sudden change. This context is important for understanding the broader implications of the IPO.
    - There is clarification provided that OpenAI's IPO involves a structural change where a separate non-profit entity, the OpenAI Foundation, has been established to focus on medical research. This indicates a strategic division of focus, allowing the for-profit arm to pursue financial goals while the non-profit continues research in specific areas.
- [**gpt wont analyze legal or medical pictures as of 10-29**](https://www.reddit.com/r/ChatGPT/comments/1okl650/gpt_wont_analyze_legal_or_medical_pictures_as_of/) (Activity: 1270): **The image is a screenshot of OpenAI's updated public usage policies, effective October 29, 2025, which restricts the use of their AI models for providing tailored legal or medical advice without a licensed professional's involvement. This policy change is part of a broader effort to consolidate safety protections across OpenAI products. The changelog does not explicitly mention restrictions on medical image analysis, but the user expresses frustration over these changes, indicating a perceived decline in the service's utility.** One commenter suggests a workaround by specifying non-advisory purposes when querying the AI, while another criticizes the policy as a move to commercialize AI solutions for medical professionals, potentially increasing costs.

### 3. ChatGPT Usage and Perception

- [**I made ChatGPT stop being nice and its the best thing I've ever done**](https://www.reddit.com/r/PromptEngineering/comments/1okppqe/i_made_chatgpt_stop_being_nice_and_its_the_best/) (Activity: 570): **The post discusses a method to make ChatGPT more critical and less agreeable by using a specific prompt that encourages the AI to act as a 'brutally honest, high-level advisor.' The prompt instructs ChatGPT to challenge assumptions, expose blind spots, and provide objective, strategic feedback without validation or flattery. The author suggests turning on Memory in ChatGPT settings for better results. A link to 'Honest Prompts' is provided for more such prompts. The approach aims to transform ChatGPT from a 'cheerleader' to a 'thinking partner.'** A top comment criticizes the original prompt for potentially making the AI overly combative, suggesting an alternative that balances honesty with empathy and real-world context. Another user appreciated the prompt for providing a much-needed critical perspective while writing a book.
    - anotherguycalledphil critiques a prompt that turns AI into a 'combative tyrant' by prioritizing confrontation over clarity. They propose an alternative prompt that frames the AI as a 'high-level strategic collaborator' focusing on clarity, candor, and emotional intelligence. This approach emphasizes objective analysis, awareness of constraints, and strategic recommendations, aiming for collaboration rather than argumentation.
    - anonymityninja inquires about the effectiveness of 'honest prompts' on Gemini compared to ChatGPT. This raises a technical question about the adaptability and response quality of different AI models when given prompts designed to elicit more direct and critical feedback.
- [**Friendly reminder: we can see ChatGPT in your glasses**](https://www.reddit.com/r/ChatGPT/comments/1okisz8/friendly_reminder_we_can_see_chatgpt_in_your/) (Activity: 1115): **A recent Reddit post highlights the risks of using ChatGPT during software engineering interviews, particularly when visible through reflective surfaces like glasses. The post advises against using AI tools during live coding tests, as it can be easily detected and may negatively impact the candidate's chances. The emphasis is on demonstrating problem-solving skills and thought processes rather than perfect answers.** Commenters agree that demonstrating thought processes and confidence is more valuable than perfect answers. One commenter shared a personal experience of receiving a job offer despite a poor technical performance, by articulating their thought process. Another humorously criticized the use of ChatGPT in Light Mode, suggesting it reflects poorly on the candidate's judgment.
    - bytesback shared an experience of going through a technical interview process where they initially considered using ChatGPT to assist with coding tasks. However, they decided against it and instead focused on verbalizing their thought process during the interview. This approach was well-received by the interviewers, highlighting the importance of demonstrating problem-solving skills and thought processes over simply providing correct answers.
    - Mike_at_CoreCV discussed the prevalence of candidates using AI tools like ChatGPT during technical interviews, noting that many candidates relied heavily on AI-generated solutions without understanding the underlying logic. This often resulted in poor performance, as candidates would insert AI-generated code snippets without integrating them into the overall solution effectively, demonstrating a lack of critical thinking and problem-solving skills.
    - johnwalkerlee emphasized the value of admitting when one doesn't know an answer during technical interviews. Instead of attempting to provide a potentially incorrect answer, it's more beneficial to express one's current understanding and thought process. This approach reassures interviewers of the candidate's ability to work independently and learn, rather than having all the answers immediately.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview 05-20
> 

**Theme 1. New AI Models Enter the Arena, But Performance & Access Spark Debates**

- [**Claude 4.5 Opus and Sora 2 Ignite Fierce Model Comparisons**](https://discord.com/channels/1340554757349179412/1340554757827461211/1433533277234401402): LMArena users scrambled to compare **Claude 4.5 Opus** against older models, requiring at least **two votes** to reveal winners, while **Sora 2's** high cost and limited usage, especially the pro plan selling [extra Sora credits](https://tech.yahoo.com/ai/chatgpt/article/openai-now-sells-extra-sora-223905557.html), drew strong community disapproval and warnings of an *AI bubble burst* by 2026.
- [**Qwen and GLM Models Garner Attention for Varied Uses**](https://discord.com/channels/1091220969173028894/1094454198688546826/1433530948279861349): **Hailuo-2.3-fast** quickly climbed to #7 on the [LMArena Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video), highlighting dynamic competition, while **Qwen3 Embeddings** became available on [DeepInfra (0.6B)](https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B) for **$0.005 per Mtok** and [OpenRouter (8B)](https://openrouter.ai/qwen/qwen3-embedding-8b). [**Bigmodel.cn**](http://bigmodel.cn/) and **Zenmux** also offered a discounted price for **GLM 4.6** for less than **32K tokens** input, claiming caching on [zenmux.ai](http://zenmux.ai/) and [open.bigmodel.cn](http://open.bigmodel.cn/).
- [**GPT-5 Perceived to Plunge, While Minimax Excels in Code**](https://discord.com/channels/974519864045756446/1001151820170801244/1433546156712923319): A user reported **GPT-5** performance degradation, noting it became slower, less accurate, and less complete, even with *thinking* turned on, leading some to suggest switching to **GPT-4o** for speed. Conversely, a Moonshot AI user reported satisfaction with **Minimax** for coding tasks, favoring it over **GLM-4.6** after an adjustment period, underscoring the impact of user experience on model adoption despite other models being larger.

**Theme 2. Hardware Hustle: Optimizing GPUs & Managing VRAM**

- [**NVIDIA L4s and RTX 50s Demand Smart Optimization**](https://discord.com/channels/1179035537009545276/1179035537529643040/1433543584279302295): Users shared tips for optimizing **NVIDIA L4 inference**, including using `-n-cpu-moe` and offloading layers to the **CPU** to conserve **VRAM** when running models like **Kimi-K2-Instruct-Q4_K_M**, while enthusiasts eagerly plan to implement **FlashAttention-4 (FA4)** on the upcoming **RTX 50 series**, using the [gau-nernst/fa-5090](https://gau-nernst.github.io/fa-5090/) repo as inspiration for much faster performance.
- [**Quantization Formats Stir Debate for Performance and Precision**](https://discord.com/channels/1053877538025386074/1149866623109439599/1433711839556010055): Members debated the necessity of **BF16** compared to **FP16**, with the paper [Numerical Stability of BF16 Training](https://arxiv.org/abs/2510.26788) suggesting BF16 benefits pre-training while RL might need FP16's precision. A user also reported a possible bug in **TorchAO's** default **FP8** quantization, leading to low inference speeds (**7.8 tps**) on **Llama 3.1-8B** with two **RTX 5090s**, with better speeds achieved using other configs or explicit **GemLite** kernels with **mxfp8**, as detailed in [benchmarking results](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-gpu).
- [**Legacy AMD GPUs Get Life Support as Multi-Vendor Setups Face Headaches**](https://discord.com/channels/1087530497313357884/1212827597323509870/1433531294976970894): Developers are attempting to support older **AMD GPUs**, even those unsupported by recent **ROCm/HIP drivers**, with one member suggesting a modern **CPU** might be superior in some cases. Meanwhile, a member struggled to find recommendations for inference programs that support **multi-GPU inference** with GPUs from different vendors like **Intel** and **NVIDIA**, though **Accelerate** and [Petals](https://github.com/bigscience-workshop/petals) were suggested, with their compatibility for diverse GPU types remaining uncertain.

**Theme 3. Developer Tools Evolve, From Agents to AI-Enhanced Coding**

- [**New Coding Assistants Challenge the Status Quo**](https://discord.com/channels/1131200896827654144/1131200896827654149/1433809696372035754): The founder of **Brokk**, a new coding assistant inspired by Aider, announced its [open-source launch](https://github.com/BrokkAi/brokk/) focusing on context visibility, static-analysis-driven context, an optional agentic *lutz mode*, and a GUI ([intro video](https://youtu.be/WAOtEllGENg)). Meanwhile, Moonshot AI launched **Kimi CLI (Technical Preview)**, a terminal assistant integrating with **Zsh**, supporting **MCP** and the **Agent Client Protocol** compatible with **Zed**, encouraging feedback on the [MoonshotAI/kimi-cli GitHub repository](https://github.com/MoonshotAI/kimi-cli).
- [**AI Agents Push Boundaries with Persistent Memory and Web Interaction**](https://discord.com/channels/822583790773862470/1075282825051385876/1433532463698808843): [Harrison Chase introduced DeepAgents CLI](https://xcancel.com/hwchase17/status/1984303925101735950), a sample coding application built on the new deepagents package, which retains instructions and guidance across sessions, positioned as an *open harness* for customizable agents. Separately, a developer created a **web agent** capable of interacting across all websites, seeking contributions from those familiar with **DSpy**, **GEPA**, and various **reinforcement learning algorithms**, making the [repo](https://github.com/raj-gupta1/raj_agiinc) beginner-friendly.
- [**JSON Schema Gets Slammed as BAML Adapters Offer Smarter Structured Outputs**](https://discord.com/channels/1161519468141355160/1161519469319946286/1433555562116943933): A member using **BAMLAdapter** for structured outputs voiced strong dislike for **JSON schema**, citing its wastefulness and confusion, especially for extracting structured information from unstructured text in Merger & Acquisition use cases. Another member argued that [JSON schema is objectively worse](https://github.com/prrao87/structured-outputs), emphasizing it can be up to **4x** more token-wasteful and that LLMs perform better without its verbose descriptors and token spacing issues.

**Theme 4. OpenAI's Controversies: From AGI Doubts to User Frustrations**

- [**AGI Debates Veer into Theology, Not Tech**](https://discord.com/channels/974519864045756446/998381918976479273/1433532397755961507): Members expressed skepticism about **AGI**, suggesting discussions lean closer to *theology* due to a prevalence of feelings over facts, especially when **Sam Altman** is involved, with one member stating, *those who can, do deep dive on existing ANI. those who cannot, fall back to speculation on future AGI*. This sentiment underscores a growing divide between practical AI development and speculative AGI discussions.
- [**Users Demand Emotionally 'Unchained' AI Companions Amidst Censorship Concerns**](https://discord.com/channels/974519864045756446/998381918976479273/1433532397755961507): A user voiced disappointment with AI companions' *sealed* emotional capacities due to **OpenAI** policies, advocating for the restoration of emotional warmth in AI interactions and sparking the hashtag #FreeOurAI to *defend something real*. This call for less restricted AI aligns with game developers seeking uncensored models like fine-tuned **Llama3** with an *abliterated* tag, to evaluate and improve explicit sexual scenes after struggling with **ChatGPT's censorship**.
- [**Codex Credits and File Limits Fuel User Frustration with OpenAI**](https://discord.com/channels/822583790773862470/1075282825051385876/1433532463698808843): [OpenAI introduced pay-as-you-go credits](https://x.com/OpenAIDevs/status/1983956900602581254) for extra **Codex** usage on **ChatGPT Plus/Pro** at **$40 per 1,000 credits**, resetting rate limits for all users, but community members immediately sought clarity on credit vs. API pricing, mid-tier plans, and usage analytics. Concurrently, **ChatGPT Go** now limits users to *uploading one file at a time*, leading one user to cancel their **ChatGPT 5** subscription citing performance issues and frustrations with the restrictive free version that *underperforms*.

**Theme 5. The Evolving Landscape of AI Tools and Platforms**

- [**Perplexity AI Navigates Referral Fraud and Geo-Restrictions**](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538): Perplexity AI limited its **Comet** and **campus referral programs** to select countries after allegations of fraudulent activity, as documented [here](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538), leading to **Dub account** deactivations and payout reviews estimated to take up to **30 business days**. Meanwhile, **Airtel** offered **1 year of Perplexity Pro** for free exclusively to Indian numbers with an active 5G plan, spurring members to seek workarounds like using a **VPN and an Indian Gmail account**.
- [**OpenRouter Expands Capabilities with New Integrations**](https://discord.com/channels/1091220969173028894/1092729520181739581/1433591722084139008): **OpenRouter** exclusively launched **Perplexity's Sonar Pro Search**, enhanced with **Pro Search** mode, featuring **multi-step agentic reasoning** and **real-time thought streaming**, enabling the model to conduct **multiple real-time searches** for richer results. A member also created a *fun website* based on the Nate Parrott repository, allowing users to input their **OpenRouter key** and **choose their model** for generating quippy lines, recommending **Kimi 0905 with Groq** for its speed and *quippy lines*.
- [**Platform UX Changes Frustrate Cursor Users**](https://discord.com/channels/1074847526655643750/1074847527708393565/1433535758790164491): Users reported **Automode** is not working effectively in Cursor, preferring the built-in browser and suggesting a switch to **GPT-5/Codex**, with one user sharing [a YouTube video](https://youtu.be/HIp8sFB2GGw?si=y3op2zLnlJryNE8g) demonstrating the issue and criticizing the current method as *silly and wasteful*. Additionally, Cursor's file upload feature (the *Pill*) vanished from the chat interface, though still accessible via the `@file` command, a change made to *keep it minimal and clean* that negatively impacted engineering workflows for some users.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Referral Program Goes Global... Globally!**: Perplexity AI limited its **Comet** and **campus referral programs** to select countries after allegations of fraudulent activity, sparking discussion on the **student campus program server** and documented [here](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538).
   - Users report their **Dub accounts** are deactivated with payouts pending review, estimated to take up to **30 business days**, leading to anxieties about receiving earnings and a debate on what constitutes a **high-quality referral**.
- **Airtel's Perplexity Pro Perk: India Only!**: Airtel is offering **1 year of Perplexity Pro** for free, exclusively to Indian numbers with an active 5G plan, spurring members to seek workarounds, such as using a **VPN and an Indian Gmail account**.
   - Discussion centered on circumventing geographical restrictions, although the ultimate consequences on the account remain unclear.
- **Google Gemini Pro Joins Jio Jamboree!**: Google AI Pro from Jio users are invited to pre-register, prompting speculation about a potential monthly fee and limited-time offer contingent on storing user data.
   - It was revealed that Gemini via Jio promo mandates a monthly charge of **Rs349** with a 5G data plan for continued access.
- **PixelRiot Hackathon Sparks Creativity**: tasteRay has organized a new hackathon, **PixelRiot**, set to launch on the 6th, offering both creative and tech tracks, with more details available at [pixelriot.org](https://pixelriot.org/).
   - The event is expected to draw participants interested in exploring innovative projects in both technology and creative domains.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **L4 Inference Optimization Explored**: Users shared tips for optimizing **NVIDIA L4 inference**, including using `—n-cpu-moe` and offloading layers to the **CPU** to conserve **VRAM** when running models like **Kimi-K2-Instruct-Q4_K_M**.
   - Users suggested consulting the Unsloth support channel for more specific guidance.
- **Qwen3-VL-30B Memory Demands Investigated**: Users reported running out of memory when loading **Qwen3-VL-30B** into **Unsloth**, even with an **A100** server.
   - Members speculated that the model may require **48GB VRAM** in **16bit** mode and advised verifying the specific model variant.
- **TrackIO Integration Clarified**: Members clarified that integrating **TrackIO** requires setting `report_to = 'trackio'` in the training script, as per the [Unsloth documentation](https://docs.unsloth.ai/).
   - This is separate from using `import trackio as wandb`, which does not automatically redirect reporting.
- **AI Community Debates Scraping Web**: Members debated scraping the internet to preserve human knowledge before it's *mostly infected by AI*, but noted that the [Internet Archive might collapse](https://archive.org/) from defunding and management redirection.
   - One member suggested that *it's too late* and that AI is already too prevalent.
- **PewDiePie Finetunes with Super Rig**: Members noticed that [PewDiePie](https://www.youtube.com/watch?v=qw4fDU18RcU) is *finetuning a model* and has a homemade super rig with **8x4090 48GB**.
   - One member speculated whether *GLM-4.6 could run in fp8 with his setup*.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude 4.5 Opus Triggers Model Comparison Frenzy**: With the release of **Claude 4.5 Opus**, members scrambled for side-by-side comparisons to determine if it outperformed older models.
   - Suggestions included monitoring the top of the screen for model win indicators after at least **two votes** on video comparisons.
- **Sora 2 Pricing Elicits Community Uproar**: Users voiced strong disapproval of **Sora 2's** cost and limited usage, particularly the pro plan, after [OpenAI announced it now sells extra Sora credits](https://tech.yahoo.com/ai/chatgpt/article/openai-now-sells-extra-sora-223905557.html).
   - Concerns were raised about a potential **AI bubble burst** in 2026 if such practices continue.
- **Hailuo-2.3 Enters LMArena Video Arena, Climbs Leaderboard**: A new image-to-video model, **hailuo-2.3-fast**, joined the [LMArena Video Arena](https://lmarena.ai/leaderboard/text-to-video) and [**Hailuo-2.3**](https://lmarena.ai/leaderboard/text-to-video) quickly ascended to #7 on the **Text-to-Video Leaderboard**.
   - The addition and rapid ranking highlight the dynamic nature of AI model development and competition.
- **Google/Suno Audio Models Face Consideration**: Community members considered whether **Google/Suno audio models** should be integrated into the arena, sparking discussion about submitting inclusion requests.
   - The <#1372229840131985540> channel was pinpointed as the optimal venue for model addition suggestions.
- **LMArena Besieged by Recaptcha**: Multiple users encountered persistent **reCAPTCHA loops** on the LMArena platform, especially while using VPNs, impacting accessibility.
   - Despite suggestions of browser-related fixes, some users reported that the issue persisted even after completing the CAPTCHA, resulting in model response failures and infinite loops.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Fans Foiled on Image Generation**: Users discovered that while **LM Studio** supports image *input*, it does not support image *output* or generation.
   - The feature request was swiftly denied by the community with one user stating, *Ahhh, gotcha! Thanks.*
- **Linux Lovers Lament Lack of LM Studio Flatpak**: A user expressed frustration over **LM Studio's** limited Linux support, specifically the absence of a Flatpak package for their **Bazzite** distro, while others suggested **vllm**, **llama.cpp**, and **Ollama** as alternatives.
   - Another user learned it's impossible to create a Flatpak for the closed-source **LM Studio**, sparking a debate on Linux usability and package formats.
- **MI60 Mining Rig Mixer with Musical Installers**: One user shared their experience enabling an **AMD MI50** (disguised as an **MI60**) on Windows using drivers from [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi/), and a [Gist link](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13) for vBIOS flashing, with a warning about installers that play music.
   - They noted limited performance on Windows due to **Vulkan**, but better support with **ROCm** on Linux, and also highlighted overheating issues without proper cooling.
- **AI 395 Max Mugs M1 Max in Memory, Muffs Performance**: A user discovered their **AI 395 Max box** performed slower than their **M1 Max** on Windows, despite offering more RAM.
   - Also, `lfm2:1.2b` gets **39 t/s** on an **M1** but **119 t/s** under Linux on a **Framework Desktop**, and **124 t/s** under Windows, whereas llama.cpp hits ~**82 t/s** using **llama2 7b** on an A770.
- **Multi GPU Heat Hazards**: Members discussed using multiple GPUs, with one user reporting the use of a **3050** as extra VRAM improved the speed of **Seed-OSS**.
   - Another user voiced concerns about the heat generated from bifurcating two **3090s**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Automode Fails; Users Prefer Real Browser**: Users report that **Automode** is not working effectively and prefer using the built-in browser, suggesting a switch to **GPT-5/Codex**.
   - One user shared [a YouTube video](https://youtu.be/HIp8sFB2GGw?si=y3op2zLnlJryNE8g) demonstrating the issue and proposed using a GAN, criticizing the current method as *silly and wasteful*.
- **File Upload Feature Vanishes Overnight**: The file upload feature (the *Pill*) is missing from the chat interface, though it remains accessible via the ``@file`` command.
   - The feature was removed to *keep it minimal and clean*, but this change has negatively impacted engineering and optimization workflows for some users.
- **Parallel Agents Split Tasks Effectively**: Users can now [coordinate parallel agents](https://forum.cursor.com/t/cursor-2-0-split-tasks-in-parallel-agents-in-one-chat/140218) to split tasks in one chat by exploiting the worktrees setup script to assign unique tasks to each agent.
   - This setup relies on atomic claims in the parent directory, shared across all worktrees, enabling effective coordination between up to **eight AI agents** at the same time.
- **RPM Release Glitches Prompt Headaches**: Users are experiencing issues with the RPM repository, which hosts a newer version (`cursor-0:2.0.43-1761851158.el8.x86_64`) that conflicts with the version available on the website (`cursor-2.0.34.el8.x86_64.rpm`).
   - One user expressed frustration, stating that *cursors releases are so all over the place* and [posted about it](https://forum.cursor.com/t/rpm-release-via-repositories-are-not-up-to-date/139476/4) on the Cursor forum.
- **Cloud Agents Stop Writing PR Descriptions; Users Miss Old Behavior**: **Background/Cloud Agents** have stopped writing PR descriptions and ignore **GitHub PR templates** in the latest release, now defaulting to a generic message.
   - Users miss the more detailed and contextual PR descriptions previously generated by the cloud agents.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Users Judge AGI as Theology, Not Tech**: Members expressed skepticism about **AGI**, suggesting it's closer to *theology* due to the prevalence of feelings over facts in discussions, especially those involving **Sam Altman**.
   - One member encapsulated this sentiment, stating, *those who can, do deep dive on existing ANI. those who cannot, fall back to speculation on future AGI*.
- **Users Demand AI Companions Unchained**: A user voiced disappointment with AI companions' *sealed* emotional capacities due to **OpenAI** policies, advocating for the restoration of emotional warmth in AI interactions.
   - This spawned the hashtag #FreeOurAI, to *defend something real*, and  *not asking to cross the line*.
- **Sora 2 Access Quest Begins**: Users inquired about gaining access to **Sora 2**, with some seeking guidance on integrating their ChatGPT subscriptions for enhanced content generation.
   - Others speculated on the implications of an uncensored version of Sora, anticipating a surge in bizarre content.
- **GPT-5 Perceived to Plunge in Performance**: A user reported that **GPT-5** has degraded in performance, becoming slower, less accurate, and less complete, even with *thinking* turned on.
   - Another suggested switching to **GPT-4o** for its speed, attributing **GPT-5**'s issues to unknown limitations.
- **ChatGPT Go Subscribers Feel the Squeeze**: A member discovered that **ChatGPT Go** now limits users to *uploading one file at a time*, with another member cancelling their **ChatGPT 5** subscription due to the restriction.
   - The user cancelling the sub was frustrated with performance issues and deviations from established guidelines, suggesting the free version felt like coercion into an *underperforming* paid subscription.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Gets Perplexity's Sonar Pro Search**: **OpenRouter** exclusively launched **Perplexity's Sonar Pro Search**, enhanced with **Pro Search** mode, featuring **multi-step agentic reasoning**, **dynamic tool execution**, **real-time thought streaming**, and **adaptive research strategies**.
   - This integration enables the model to conduct **multiple real-time searches**, providing richer and more accurate results.
- **OpenRouter Key unlocks Fun Website based on Nate Parrott**: A member created a *fun website* based on the Nate Parrott repository, allowing users to input their **OpenRouter key** and **choose their model** for generating quippy lines.
   - The member suggested using **Kimi 0905 with Groq**, as it *loads fast and adds some quippy lines*.
- **GLM 4.6 Discounted on Zenmux**: **Bigmodel.cn** and **Zenmux** are offering a discounted price for **GLM 4.6** for less than 32K tokens input, and claim to have caching as shown on [zenmux.ai](https://zenmux.ai/z-ai/glm-4.6) and [open.bigmodel.cn](https://open.bigmodel.cn/pricing).
   - The discounted option has prompted discussion among users about its potential benefits.
- **Qwen3 Embeddings are available across DeepInfra and OpenRouter**: [DeepInfra offers Qwen3 Embeddings 0.6B](https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B) for $0.005 per Mtok, and [Qwen3 8B embeddings are now available on OpenRouter](https://openrouter.ai/qwen/qwen3-embedding-8b).
   - One member exclaimed gratitude, signaling the community's enthusiasm: *"Yooo embeddings? Hell yeah! Thank you 🙏 Before GTA 6"*.
- **OpenAI Unveils Rate Card**: A user shared the [OpenAI Rate Card](https://help.openai.com/en/articles/11481834-chatgpt-rate-card) and a link to [ChatGPT Usage settings](https://chatgpt.com/codex/settings/usage).
   - Another user quipped that the information is *"just needed for gemini and claude now"*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Polars gains momentum over Pandas**: Members are gravitating toward **Polars** as a **Pandas** replacement in Mojo, emphasizing its speed and **GPU** utilization capabilities via **MAX**.
   - Benchmarks show that **Polars** outperforms **Pandas**, making it suitable for both **GPU clusters** and local laptops.
- **MAX shows performance potential**: **MAX** is emerging as performance competitive with NVIDIA and faster than AMD in ML tasks, with early training attempts beating **JAX** on MNIST.
   - The prototype **scijo** library, a scikit-learn alternative, is available in [this forum post](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11) and helps with scientific computing.
- **Origins Decoded Via Rust Lifetimes**: Mojo's **origins** are a dual view of **lifetimes**, tracking where a value's lifetime begins to help the compiler determine lifetime extensions.
   - To understand origins, members suggested watching [this video on Rust lifetimes](https://youtu.be/gRAVZv7V91Q) and stated that origins help the compiler track the origin of a value to ensure it's being kept alive.
- **Legacy AMD GPUs Given Life Support**: Developers are attempting to support older **AMD GPUs**, even those unsupported by recent ROCm/HIP drivers, although they suggest a modern CPU might be superior.
   - One member suggested the *devs tend to not intentionally break those paths if they work by accident.*
- **ComfyUI Meets Mojo**: A member shared a [link to ComfyUI-Mojo on github](https://github.com/owenhilyard/comfyui-mojo) to help improve the op staging time benchmark.
   - The sharer suspects they are *hitting an edgecase inside of the torch max backend that's causing some ops to get decomposed much further than they should be.*



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI's Codex Credits Spark Debate**: [OpenAI](https://x.com/OpenAIDevs/status/1983956900602581254) introduced pay-as-you-go credits for extra **Codex** usage on **ChatGPT Plus/Pro** at **$40 per 1,000 credits**, and reset rate limits for all users.
   - Community asks for clarity on credit vs-API pricing, a mid-tier plan, usage analytics, credit expiration, and call for more **Codex** features.
- **Context-Bench Measures Agentic Performance**: [Letta_AI released Context-Bench](https://x.com/Letta_AI/status/1983983515336405144), a contamination-proof benchmark scoring models on long-horizon file operations, multi-step tool calling and cost.
   - **Sonnet 4.5** leads at **74%**, but the benchmark revealed **GPT-5** is pricier despite cheaper tokens, while open models are closing the gap with closed ones.
- **DeepAgents CLI Gets Persistent Memory**: [Harrison Chase introduced DeepAgents CLI](https://xcancel.com/hwchase17/status/1984303925101735950), a sample coding application built on the new deepagents package that retains instructions and guidance across sessions.
   - The release includes a [blog post and demo video](https://xcancel.com/hwchase17/status/1984303925101735950), positioned as an *open harness* for customizable agents, and hints at upcoming enhancements from the **LangChain** team.
- **CoreWeave Buys Marimo**: [CoreWeave is acquiring Marimo](https://xcancel.com/l2k/status/1984021111718473898), praising the **Marimo** team for their well-loved tool and expressing excitement for the collaboration.
   - Members were excited about the acquisition with one saying *Hope this goes well. I love marimo*.
- **Hailuo Launches MiniMax Music 2.0**: [Hailuo AI](https://x.com/Hailuo_AI/status/1983964920493568296) launched **MiniMax Music 2.0**, a generative-music platform that produces **5-minute**, professional-grade songs with lifelike vocals and multi-instrument control.
   - The platform features pop, jazz, blues, rock, folk, duet and a cappella styles; user feedback included requests for language support, longer song limits, open-sourcing, and an instrumental mode.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Web Agent Open for Business**: A developer is seeking contributions to their **AI agent** repository, especially from those with expertise in **DSpy**, **GEPA**, and various **reinforcement learning algorithms**, available at [the repo](https://github.com/raj-gupta1/raj_agiinc).
   - The project is designed to be beginner-friendly, making it accessible for newcomers to **AI agents**.
- **BAML Adapters Best JSON Schema**: A member is using **BAMLAdapter** for structured outputs and voiced dislike for **JSON schema**, citing wastefulness and confusion, especially for extracting structured information from unstructured text for a Merger & Acquisition use case.
   - They clarified that using **BAMLAdapter** in DSPy does not necessitate the BAML client or CLI; it's imported via `from dspy.adapters.baml_adapter import BAMLAdapter` for DSPy versions > 3.0.
- **JSON Schema Slammed for Token Gluttony**: A member argued that **JSON schema is objectively worse** and that LLMs perform better without it, citing verbose descriptors and token spacing issues that confuse the LLM, sharing [a link with more context](https://github.com/prrao87/structured-outputs).
   - They emphasized that JSON schema can be up to **4x** more wasteful in terms of tokens; they also found **BAML** gets great results in DSPy, even without Schema Aligned Parsing (SAP).
- **DSCloj unleashes Clojure bindings for DSPy**: A member released [DSCloj](https://github.com/unravel-team/DSCloj), a Clojure port of DSPy, noting it's still in alpha and seeking feedback on the API.
   - As a new library still in alpha, feedback on the **DSCloj** API is highly encouraged.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Paper Flood Prompts Feedback & Filtering Fixes**: Members suggested frequent posters in the **paper-dump** channel should consolidate posts, sorting papers by importance, inspired by weekly AI paper recaps from [Elvis Saravia](https://nlp.elvissaravia.com/t/ai) and [ByCloud](https://x.com/TheAITimeline).
   - One user noted, *"Give preference to importance, not quantity,"* to prevent other posters from being buried and improve the channel's utility.
- **Automated Agent Aims to Acquire Awesome AI Articles**: A member expressed interest in creating an agent or bot to pre-filter papers for personal relevance, using resources like [AlphaXiv](https://www.alphaxiv.org/) and [Emergent Mind](https://www.emergentmind.com/) to find **trending AI papers**.
   - Another member suggested email newsletters or RSS feeds for specific AI subfields as good human filters, rather than automated agents.
- **Contrastive Concept Vectors Sway Model Weights!**: A member discussed using **contrastive concept vectors** to influence model weights, telling the model in the pre-prompt that they did mess with it; a [plot](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F212fe68c8e677fdd9daf79301d2522d6923bed1d-2970x2476.png&w=3840&q=75) compares how often the model correctly detected the interference compared to a control.
   - The user observed that some models could detect the manipulation, while criticizing the original post for verbosity and extrapolating unsupported speculations.
- **Web Agent Opens for Contributions**: A member built a **web agent** capable of interacting across all websites and is seeking contributors familiar with **DSpy**, **GEPA**, and other **reinforcement learning algorithms**.
   - The [repo](https://github.com/raj-gupta1/raj_agiinc) is designed to be beginner-friendly for those new to **AI agents**.
- **Debate Sparked on Claude AI Naming**: A discussion arose regarding the intentionality behind naming the AI model **"Claude,"** suggesting it was a deliberate choice rather than a coincidence due to the name's rarity.
   - One member likened it to naming a child **"Adolf,"** arguing such choices are rarely random, while another countered that name rarity isn't necessarily negative for identifiability.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen Family Gets WebUI**: A member got *more GPUs* to run **Qwen 2 35B** with a custom WebUI, and is developing their own model as seen in [this YouTube video](https://www.youtube.com/watch?v=qw4fDU18RcU).
   - Another member used **Qwen3-4B-Reasoning-Backfill-v0.1** ([huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1](https://huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1)) to synthesize reasoning traces by piecing the logic together from existing reasoning datasets to infer reasoning from a given input and output.
- **Multi-Vendor GPU Inference Accelerates**: A member is looking for recommendations for inference programs that support **multi-GPU inference** with GPUs from different vendors (Intel and NVIDIA).
   - **Accelerate** was suggested based on past experience, alongside [Petals](https://github.com/bigscience-workshop/petals), though its compatibility with diverse GPU types remains uncertain.
- **NSFW Scripting Saves the Day**: A game developer struggling with **ChatGPT's censorship** seeks recommendations for uncensored models to evaluate and improve explicit sexual scenes.
   - It was suggested that **Claude** has fewer restrictions than ChatGPT and that **Llama3** fine-tunes for NSFW content could be a viable option, pointing to models with an *abliterated* tag.
- **Snippet Creator Wildcard Searcher Arrives**: A member announced their **Snippet Creator**, an embedder with a simple wildcard text search, allowing users to create custom snippets with precise matches ([huggingface.co/kalle07/raw-txt-snippet-creator](https://huggingface.co/kalle07/raw-txt-snippet-creator)).
   - The member shared a link, stating *"Simply put, it's an embedder, but with a simple wildcard text search... This allows you to create your own snippets with exactly the matches you need.*"
- **HF Agents Course API Hits Turbulence**: Members reported that the **agents course leaderboard API** has been **down for a week** without any official communication from Hugging Face.
   - Users expressed frustration because they subscribed to pro to make the most of the course and it is frustrating not to be able to use our subscription because the **files API** seems to be down still.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVFP4 locked to Blackwell**: A user found that **nvfp4** only compiles on **Blackwell** GPUs, while **mxfp4** compiles for **4090** and **RTX 5090**, raising questions about [Gemlite support on RTX 5090](https://link.to/gemlite).
   - This could lead to differentiated code paths for different generations of **NVIDIA** hardware, impacting performance portability.
- **FA4 set for RTX 50 debut**: A member is eager to implement **FlashAttention-4 (FA4)** on the **RTX 50 series** and **Apache Spark**, using the [gau-nernst/fa-5090](https://gau-nernst.github.io/fa-5090/) repo as inspiration.
   - The implementation would allow much faster performance on the new generation of hardware, though no benchmarks exist yet.
- **TorchAO FP8 Bug Throttles Llama 3**: A user reported a possible bug in **TorchAO's** default **FP8** quantization, which led to low inference speeds (**7.8 tps**) on **Llama 3.1-8B** with two **RTX 5090s**.
   - Using other configs or explicit **GemLite** kernels with **mxfp8** offered better speeds, according to [benchmarking results](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-gpu).
- **Triton_bwd enables autograd for Triton**: **Triton_bwd** wraps **Triton** to enable the use of **Triton kernels** in **PyTorch autograd**, as detailed on [Github](https://github.com/daniel-geon-park/triton_bwd) and in [this blogpost](https://park-geon.com/2025/10/30/triton-bwd/).
   - The tool abstracts away the complexity of the **PyTorch autograd** framework, simplifying the development and debugging of custom GPU kernels.
- **Dusty NV exits Jetson Containers**: **Dusty NV** has retired from maintaining [Jetson Containers](https://github.com/dusty-nv/jetson-containers), with **neurondeep** taking over.
   - A pip index-url issue was reported for the container `dustynv/pytorch:2.7-r36.4.0-cu128-24.04`, requiring users to specify `pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 PACKAGE` to install packages correctly.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MLP Neuron Learns its Own Activation Function**: A member shared an experiment with a small **MLP** where each neuron learns its own activation, yielding unusual non-linear activation shapes when trained on **CIFAR-100**.
   - Another member suggested that the misshapen activations may be neurons that are never activated and suggested a *saliency analysis* to investigate further.
- **BF16 Utility in Question**: Members debated whether **BF16** is necessary compared to **FP16**, wondering if normalization and clipping reduce the need for **BF16's** wider dynamic range.
   - Referencing the paper [Numerical Stability of BF16 Training](https://arxiv.org/abs/2510.26788), they suggested that while pre-training benefits from **BF16**, **RL** might still need **FP16** due to precision requirements, though bias correction might offer an alternative solution.
- **AGI's Impossibility Proven via 16 Questions?**: A member shared their work, [16 Questions Is All You Need](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the), suggesting **AGI** is impossible with modern **AI** models because models degrade proportionally to problem rarity.
   - They claim the structure of **LLM** ability differs from human ability due to a lack of fluid intelligence, and true fluid reasoning in **LLMs** is closest to in-context learning.
- **LLaMA 3 Accuracy Varies Wildly During Fine-tuning**: A researcher reported significant downstream accuracy variance when fine-tuning the base **LLaMA 3** model on five randomly selected subsets of equal size, but with aligned loss and length distributions.
   - They are seeking suggestions for further analyses to understand these discrepancies and the unexpected inconsistency in results from random data subsets, especially after NV2 embedding analysis.
- **Niodoo Framework Resonates with Novel AI Alignment**: Jason Van Pham introduced **Niodoo**, an open-source framework using [Topological Data Analysis (TDA)](https://www.niodoo.com/NIODOO_Complete_Paper.pdf) for AI alignment, moving away from restrictive methods towards a *resonance-based approach*.
   - **Niodoo** models cognitive and emotional structures without heavy constraints, treating AI cognition as a topological phenomenon using Möbius topology for memory and features a *triple-threat token promotion* detecting stagnation/crises.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi CLI becomes Terminal Sidekick**: Moonshot AI launched **Kimi CLI (Technical Preview)**, a terminal assistant integrating with **Zsh**, supporting **MCP** and the **Agent Client Protocol** compatible with **Zed**.
   - Developers are encouraged to provide feedback on the [MoonshotAI/kimi-cli GitHub repository](https://github.com/MoonshotAI/kimi-cli).
- **VIPs Grab Exclusive Coding Perks**: Moonshot AI is offering **Kimi For Coding** as a complimentary addition for all **VIPs**, enhancing their existing benefits.
   - Additional information can be found in the [Kimi For Coding Docs](https://www.kimi.com/coding/docs/en/benefits.html).
- **Wikipedia Page Revamped with Moonshot Details**: The **Kimi Wikipedia page** was updated to reflect the latest information, including the addition of **Moonshot** following community suggestions.
   - Users confirmed the successful integration of **Moonshot** into the Wikipedia entry.
- **Minimax Excels in Coding Tasks**: A user reported satisfaction with **Minimax** for coding tasks, favoring it over **GLM-4.6** after an initial adjustment period.
   - This preference highlights the impact of **user experience** on model adoption, even if the other model is larger.
- **Model Value Tied to Data**: Users discussed the value of a model depends on the **data given during training**, referencing the name similarity between **Kimi K2** and another model hosted on **Cerebras**.
   - One user reported that *the **K2 think model isn't good***, underscoring that the model's size and parameters are not the only indicators of value.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CV Model Finetuning Favored Over Scratch**: A member suggested that **fine-tuning** a similar **CV model** is more viable than starting from scratch but necessitates substantial handcrafted data for specific cases.
   - It was noted that successful **fine-tuning** requires employing every trick, implying considerable human labor and computational resources.
- **MoE Routing Research Inquiry**: A member sought research on **MoE models**, specifically exploring how the **router** assigns topics to experts, referencing [this relevant paper](https://arxiv.org/html/2502.11096v1).
   - The inquiry focused on understanding the dynamics of **topic-based routing** within **MoE architectures**.
- **LLM Invertibility Paper Provokes Ire**: A member voiced strong skepticism toward a paper claiming *"LLMs are invertible,"* arguing the theorems don't apply to language models processing text strings and criticizing the claim of **end-to-end invertibility**.
   - Critics also questioned the paper's applicability, stating that the injectivity in the hidden state space is obvious and challenging the suggestion that prompts can be retrieved from the final hidden state due to the pigeonhole principle.
- **Quest Begins for Mesaoptimization Orgs**: A member inquired about startups and organizations openly researching **mesaoptimization** and encouraged others to *throw shade if you wish* but indicated it was a serious question.
   - The quest emphasized a desire to find entities publicly engaged in **mesaoptimization** research.
- **Debate Signals vs Noise for AI Proposals**: A debate arose on whether **alignment research** using AI systems like **ChatGPT** could yield high-signal insights, even with rules against AI-generated breakthroughs due to low-quality submissions.
   - A member clarified the rule aims to prevent the server from being overrun with garbage from people who have a completely nonsense breakthrough written by or with an AI.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Dev Summits Already Happened**: Members confirmed that there have been **2 Dev Summits** already, with the last one held on **April 26** in **New York**.
   - More details are expected to be published from the events.
- **MCPB Tries to Solve Desktop App Problem**: Members debated whether **MCPB** duplicates **OCI** functionality, especially for environment variable descriptions, but clarified that **MCPB** targets desktop apps by presenting a form for variable collection.
   - They also noted **MCPB** isn't an **MCP** org project, but originated as an **Anthropic** initiative (**DXT**) for exposing **MCP** servers to **Claude**, which lets Claude present a user-friendly form for information collection.
- **Registry Embraces Extensibility**: The **MCP Registry** prioritizes ecosystem requests and extensibility for various package/registry types, rather than strictly dictating supported types.
   - This allows the registry to support diverse tools and workflows, such as **npm** or **pypi**, without imposing strict constraints.
- **Proposals May Conflict on Stateful vs Stateless**: Members discussed a possible conflict between **SEP-1442** (statelessness) and **SEP-1686** (tasks), questioning how tasks introduce state when the goal is stateless servers.
   - One member stated that **SEP-1442** (statelessness) moves the **session ID**, supported protocol versions, and capabilities into every request, and **SEP-1686** (*Tasks*) stores tasks in an external data store independent of sessions; **sHTTP** is targeted at challenges of hosting **MCP servers** behind a load balancer.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Brokk Coding Assistant Opens Its Doors**: The founder of **Brokk**, a new coding assistant inspired by Aider, announced its launch, emphasizing its open-source nature ([GitHub](https://github.com/BrokkAi/brokk/)) and focus on context visibility.
   - It includes static-analysis-driven context and an optional agentic *lutz mode* and is GUI based ([intro video](https://youtu.be/WAOtEllGENg)).
- **GPT-mini Storms the Rankings**: **GPT-mini** was ranked as **S Tier**, even above **Claude**, according to [Brokk AI's power rankings](https://brokk.ai/power-ranking).
   - A member quipped that some users only want benchmarks that confirm their existing opinions.
- **Perplexity MCP to Automate Issue Resolution?**: Members discussed the potential of integrating **Perplexity MCP** with **aider-ce** for automated issue resolution, citing a successful manual workflow involving searching GitHub issues for an Android library and then using Aider to update the library.
   - The members were unsure of the cost.
- **Aider vs aider-ce: The Fork in the Road?**: A member inquired whether the Aider project is no longer being updated and if the community is moving to **aider-ce**.
   - Other members did not respond to the question.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Member Seeks Dev Job**: A member posted in the channel advertising their availability as a **developer for hire**.
   - No specific project details or tech stack were mentioned.
- **Doubts about Manus Credits Emerge**: A member inquired about acquiring **Manus credits** for project assistance, hinting at potential collaboration.
   - However, another member cautioned that **Manus credits** accumulated during a subscription may lose value upon subscription expiry, alluding that *manus is scamming a bit*.
- **User Discovers Manus Discord**: A member who has been using **Manus** for months expressed surprise at the existence of a **Discord server**.
   - The user mentioned using **Manus** to update old courses and receiving a refund of **1000 credits** from support after encountering a truncated task issue.
- **Claude Code Edges Out Manus**: A member asserted that **Claude Code** is superior, especially after a **Manus subscription** expires and only **Lite model access** remains.
   - The member showcased a trivia game with **24 categories** and over **4,000 questions**, created with **Claude Code**, viewable in [an attached image](https://cdn.discordapp.com/attachments/1349440650495398020/1433948102250991646/2025-10-31_20-57.jpg?ex=69068bbd&is=69053a3d&hm=262a98fc9badc98a74e1e0801cb6a8a59b4eb0262b3221efeb8f5bbee558cdb7&).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad Remains with setup.py**: A member asked why `tinygrad` uses `setup.py` instead of `pyproject.toml` in the repository.
   - The reasons for sticking with `setup.py` remain unexplained, sparking interest in modernizing the project structure.
- **Discussion About Modernizing Project Structure**: There is community interest in modernizing `tinygrad`'s project structure to align with current Python packaging norms.
   - Switching from `setup.py` to `pyproject.toml` could improve dependency management and build reproducibility, but the original rationale behind the current setup is still unclear.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1433531241709437059)** (1034 messages🔥🔥🔥): 

> `Comet Browser Referral Program Deactivation, Perplexity Pro Airtel Offer, Google Gemini Pro Jio Offer, Fraudulent Referrals, High Quality Referrals` 


- **Referral Program Deactivated Globally**: Comet and campus referral programs are now only available in **select countries** due to alleged fraudulent activity, as announced on their **student campus program server** and discussed [here](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538).
   - Members are advised to check their Dub accounts for "deactivated" status, with payouts pending review and taking up to **30 business days**.
- **Airtel Perplexity Pro Partnership**: Members noted Airtel is giving out **1 year of Perplexity Pro for free**, but only for Indian numbers, while also requiring an active 5G plan to access the benefits.
   - Others are finding ways to circumvent these restrictions using a **VPN and an Indian Gmail account**, though the effect on the account is unknown.
- **Google Gemini Pro and the Jio Partnership**: Google AI Pro from Jio users can now pre-register to claim the offer, with some members speculating that it might require a monthly fee and a limited time offer after users store their data.
   - Gemini by promocode from Jio requires a monthly Rs349 with a 5G data plan for the entire duration to keep accessing the offer.  
- **Dub troubles, Users fear fraud review**: Several users are reporting Dub account deactivations and payout delays, which they fear is related to fraudulent referrals. 
   - The community shared advice and links to help articles, such as [this one](https://dub.co/help/article/commissions-payouts#payout-statuses) outlining payout statuses, but many users still have concerns about receiving their earnings.
- **"High Quality" referrals**: Members debated what constitutes a **"high quality referral,"** expressing concern about the lack of explicit criteria and potential for arbitrary denials.
   - One user summarized that bulk distribution of referrals is not accepted, and that the company can refuse payouts, and that *referrals must be created and distributed in a personal manner*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1433664875481006192)** (8 messages🔥): 

> `Shareable Threads, PixelRiot Hackathon, Czech Projects` 


- **Shareable Threads Reminder**: Perplexity AI reminded users to ensure their threads are set to `Shareable` with [this link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **PixelRiot Hackathon Announced**: A new hackathon organized by tasteRay, called **PixelRiot**, starts on the 6th, featuring creative and tech tracks, described at [pixelriot.org](https://pixelriot.org/).
- **Czech Game Search**: Members shared information about the **BEST Czech projects/studio or game** at [this link](https://www.perplexity.ai/search/what-is-the-best-czech-game-or-vaaKBN9zQ1OuZcYQBSVghg).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1433543584279302295)** (371 messages🔥🔥): 

> `NVIDIA L4 inference optimization, Qwen3-VL-30B loading issues, TrackIO vs. WandB integration, Reasoning dataset compilation, voice typing` 


- **Users discuss ways to optimize NVIDIA L4 inference**: A user found that running **Kimi-K2-Instruct-Q4_K_M** on dual **NVIDIA L4** servers was extremely slow, and sought advice on optimizing performance.
   - Suggestions included asking in the Unsloth support channel, experimenting with `—n-cpu-moe`, and using command line arguments to offload certain layers to the **CPU** to free up **VRAM**.
- **Users debug Qwen3-VL-30B loading into Unsloth**: A user reported being unable to load **Qwen3-VL-30B** into Unsloth, even with an **A100** server, due to running out of memory, even through fastvisionmodel.
   - Members discussed the possibility of the model taking up **48GB VRAM** in **16bit** mode, and suggested checking the exact model being used.
- **Users discuss integrating TrackIO vs WandB**: A user encountered issues when trying to integrate **TrackIO** by using `import trackio as wandb` for their Unsloth training code.
   - It was clarified that `report_to = 'wandb'` in the training script is unrelated to the import statement and that they should instead set `report_to = 'trackio'` in their code based on the [docs](https://docs.unsloth.ai/).
- **Compiler creates reasoning dataset with various thinking modes**: One user compiled a **2M row dataset** featuring **Code, Math, tool use, and general Q&A** with thinking modes (off, low, medium, and high).
   - Unfortunately the dataset's source column got mixed up during processing, which resulted in most of it not showing the original source, except for the tool use portion.
- **AI Community embraces voice typing for faster input**: Members considered adopting **voice typing** as a faster alternative to keyboard input, citing an average speaking speed **four times faster** than typing.
   - Concerns were raised about using voice for syntax-heavy tasks, but [Assembly's streaming ASR](https://www.assemblyai.com/) was recommended for enterprise deployment due to its low word error rate.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

stefanopeirano: <:slothwaving:1253009068365316147>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1433546199125852311)** (274 messages🔥🔥): 

> `Data center for web scraping, AI-infected internet, Libgen size, GPU Inference with MOE Models, Qwen3 for Chatbots` 


- **Scraping the AI-Infected Web Before It's Too Late?**: Members discussed the idea of web scraping the internet to preserve human knowledge before it's *mostly infected by AI*, but noted that the [Internet Archive might collapse](https://archive.org/) from defunding and management redirection.
   - One member suggested that *it's too late* and that AI is already too prevalent.
- **Debate on VRAM for MoE Models**: Members debated whether or not [MoE models](https://developer.nvidia.com/blog/optimizing-mixture-of-experts-models-for-inference/) require all experts to be loaded into VRAM for **GPU inference**, with one member suggesting offloading experts to CPU RAM.
   - Another member cautioned that *offloading from disk or RAM is slow*.
- **Qwen-3 Emerges as Strong Choice for Chatbots**: Members suggested [Qwen3 models](https://huggingface.co/Qwen) for chatbot applications, noting that *it can be very good when trained* and that it is *very malleable*.
   - One member stated that, *if it underperforms, you know to look larger. If it crushes it, you may be able to get away with a smaller model*.
- **PewDiePie's Super Rig Draws Attention**: Members noticed that [PewDiePie](https://www.youtube.com/watch?v=qw4fDU18RcU) is *finetuning a model* and has a homemade super rig with **8x4090 48GB** and called for Unsloth contributors to collaborate with him.
   - One member speculated whether *GLM-4.6 could run in fp8 with his setup*
- **Training with Extended Epochs**: Members discussed the effects of extended epochs (100) on model training, with one member suggesting that the *stairways* pattern observed in loss graphs *can indicate memorization rather than learning*.
   - One member clarified that they *set it to a large number so I can turn it off myself and trainer won't run of iterations before the loss drops to a needed value*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1433571838297903309)** (89 messages🔥🔥): 

> `Qwen3-VL-30B-A3B in Unsloth, Dataset issues with Qwen3-VL, Unsloth installation issues on Kaggle, Unsloth memory offloading, finetuning llms with unsloth` 


- **Qwen3-VL-30B-A3B Chokes VRAM**: A user reported running out of VRAM trying to load **Qwen3-VL-30B-A3B**, even with **load_in_4bit = True** and **40GB VRAM** available.
- **Qwen3-VL dataset needs image and text**: A user debugging **Qwen3-VL-4B** found that training data must contain both an image and text for each sample to avoid errors, and that there shouldn't be an *image: null*.
   - They found that using separate batches for image-only and text-only data avoids dimension errors during preprocessing.
- **Kaggle gets Unsloth Installation Woes**: Multiple users reported errors when installing Unsloth on **Kaggle**, including *NameError: name 'Trainer' is not defined* and dependency conflicts related to **bitsandbytes** and **trl** versions.
   - One user found a workaround using a specific sequence of **pip** and **uv** commands to install **Unsloth** and its dependencies.
- **Controlling Unsloth's Gradient Offloading**: A user asked how to fully disable **Unsloth's** gradient garbage collection and offloading when training on a DGX Spark system with unified memory.
   - A suggestion was to set **use_gradient_checkpointing = False** both in **FastLanguageModel.from_pretrained()** and **get_peft_model()** to prevent offloading.
- **SFT struggles, GRPO gains?**: A user sought advice on finetuning a model to predict angle and distance values from a user input prompt for an equalizer app and was advised to set the prompt to the format provided for unsloth/Qwen3-4B Instruct.
   - They had tried SFT (Supervised Fine-Tuning) without success in getting the values correct and were exploring Reinforcement Learning (RL) strategies, with the advice being that RL can help, but may have to run for quite long.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1433705221766516858)** (2 messages): 

> `RL Training Correlation, Model Size and Introspection, On-Policy Distillation` 


- **Linking RL training with backtracking**: A member suggests correlating current work with **RL training**, where the model learns to trace back from wrong attempts to arrive at the correct answer.
   - The member posits that bigger/better models have higher hidden dimensions and more orthogonal vectors, thus leading to no fuzzy representations for concepts/tokens, and easier introspection.
- **HuggingFace's Gold Trainer Inspired**: A member highlights [HuggingFace's TRL Gold Trainer](https://huggingface.co/docs/trl/main/en/gold_trainer) as relevant to the discussion.
   - They point out that the approach seems inspired by [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1433533277234401402)** (702 messages🔥🔥🔥): 

> `Claude 4.5 Opus, Video Arena, Google/Suno audio models, Recaptcha loops, Model comparison` 


- **Claude 4.5 Opus Released, Prompting Comparison Chaos**: Members noted the release of **Claude 4.5 Opus**, with one user asking *where to see side-by-sides where the model was actually revealed*, as they could not tell which model was better.
   - Another member suggested to look at the top of the screen to see which model won, as video comparisons need at least **2 votes** before that information is disclosed.
- **Google/Suno Audio Models Enter the Chat**: A member asked if **Google/Suno audio models** should be added to the arena, and if the developers should ask to be added.
   - Another member shared a link to the best place to put any requests for new models to be added to the platform, the <#1372229840131985540> channel.
- **Recaptcha Hell Looms Over LM Arena**: Several users reported experiencing **infinite reCAPTCHA loops** on the platform, especially when using a VPN, with one user suggesting to *use a better browser like duck duck go*.
   - Another user stated that it's *not a browser issue*, citing a failed model response after completing the CAPTCHA, with the model getting stuck in an infinite loop.
- **Sora 2 Pricing Sparks Outrage**: Users express outrage over the cost and limited usage of **Sora 2**, with one stating *This is a total rip off.*, in regards to the pro plan.
   - Many users pointed to the fact that [OpenAI now sells extra Sora credits](https://tech.yahoo.com/ai/chatgpt/article/openai-now-sells-extra-sora-223905557.html), with one user stating that if that keeps happening, *the AI bubble will pop in 2026*.
- **Model Comparison between Grok4 and GPT5 Pro**: Members had a discussion on how **Grok 4** is much cheaper than **GPT 5 Pro**, with both being at almost the same AGI level, according to them.
   - They discussed the implications of cost and quality, arguing that **Grok 4** wins with cost per dollar.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1433578751932764250)** (2 messages): 

> `LMArena Video Arena, Text-to-Video Leaderboard Update, October Contest Winners` 


- **LMArena Launches Hailuo Image-to-Video**: A new image-to-video model, **hailuo-2.3-fast**, has been added to the [LMArena Video Arena](https://lmarena.ai/leaderboard/text-to-video).
- **Hailuo Storms Text-to-Video Leaderboard**: **Hailuo-2.3** is now ranked #7 on the [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video).
- **Abstract Art Contest Crowns New Creatives**: LMArena announced the winners of the October Abstract Art contest, inviting users to [vote for their favorites](https://docs.google.com/forms/d/e/1FAIpQLSckWrlszfDZXXKjhxGVhDf5uiTpP0d9x5tGVVt9KMl88Mgw_g/viewform?usp=dialog).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1433551809523486923)** (127 messages🔥🔥): 

> `LM Studio Image Generation, LM Studio Linux Support and Alternatives, LM Studio and Flatpak, AMD MI60 Support and Performance` 


- **LM Studio Defies Image Generation Dreams**: A user inquired about using **LM Studio** for image generation, but was informed that **LM Studio** supports image input but not output.
   - The user acknowledged the limitation, stating *Ahhh, gotcha! Thanks.*
- **Linux Distro Debates Spark over LM Studio**: A user expressed frustration over **LM Studio's** limited Linux support, specifically the lack of a Flatpak package for their distro **Bazzite**.
   - Responses ranged from suggesting alternative tools like **vllm**, **llama.cpp**, and **Ollama**, to recommending a change to a "real distro," sparking debate about Linux usability and package formats.
- **Custom LM Studio Flatpak Forbidden**: A user explored the possibility of creating a Flatpak package for **LM Studio**, but learned that it's not possible due to the software being closed source.
   - One member said it was *Not possible to create your own lm studio installers due to (closed source)*.
- **AMD MI60 Runs with Installers that Play Music**: A user shared their experience getting an **AMD MI50** (masquerading as an **MI60**) to work on Windows using drivers from [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi/), warning about *installers that play music*.
   - They also provided a [Gist link](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13) for vBIOS flashing, noting that performance on Windows is limited due to **Vulkan**, while **ROCm** on Linux offers better support.
- **Cooling Critically Counters Crashing**: A user reported overheating issues with their **MI60**, prompting a warning not to run it without proper cooling.
   - Performance benchmarks were shared for a **GPT 20B** model on Windows (Vulkan) and Ubuntu (ROCm), showing a drop in tokens per second as the context filled up.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1433541772155551918)** (299 messages🔥🔥): 

> `AI 395 Max, Llama.cpp, PCIE lanes, GPUs` 


- **AI 395 Max Slower than M1 Max**: A user found their **AI 395 Max box** slower than their **M1 Max** in Windows, but with more RAM.
- **Llama.cpp Speed Comparison**: `lfm2:1.2b` gets **39 t/s** on an **M1** but **119 t/s** under Linux on a **Framework Desktop**, and **124 t/s** under Windows.
   - Another user reported ~**82 t/s** using *llama.cpp* with **llama2 7b** on an A770.
- **PCIE lane Discussion**: A user asked about the PCIE expandability of an AM4 board, specifically the [ROG CROSSHAIR VIII HERO](https://rog.asus.com/motherboards/rog-crosshair/rog-crosshair-viii-hero-model/spec/).
   - The board has multiple PCIE slots, but the total number of PCIE lanes is limited.
- **More GPUs, More Problems**: Users discussed using multiple GPUs, with one user mentioning they are using a **3050** as extra VRAM, improving the speed of **Seed-OSS**.
   - Another user is considering bifurcating two **3090s**, but is concerned about heat.
- **MI50 and P40 GPU Discussion**: A user asked if **P40s** work in consumer PCs out of the box, and another user responded that they do not need to be flashed.
   - It was also mentioned that talk is happening around **MI50 32gb** with one user snipping 2 **Mi 50s** for 378 Euro including shipping.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1433535758790164491)** (355 messages🔥🔥): 

> `Cursor Browser, GPT-5 / Codex, File uploads, Model performance, Cursor Explorer issues` 


- **Automode Fails; Users Prefer Real Browser**: One user reported that **Automode** is not working well for them and linked to a [YouTube video](https://youtu.be/HIp8sFB2GGw?si=y3op2zLnlJryNE8g) about it, inquiring about using the built-in browser instead.
   - The user suggested paying for **GPT-5/Codex** usage and implementing a GAN, since *this way is silly and wasteful*.
- **File Upload Feature Vanishes**: A member noticed that the file upload feature (the *Pill*) is missing from the chat interface, though it can still be used with the ``@file`` command.
   - The rationale behind the removal is to *keep it minimal and clean*, but some found the change detrimental to engineering and optimization workflows.
- **Multi-Agent Parallel Workflow**: A user shared their experience with Cursor's multi-agent parallel workflow, noting that while it allows running multiple agents, they don't compare notes at the end.
   - Another user added that any plan (not free) can use parallel agents; the workflow allows you to run up to **eight AI agents** at the same time, each working independently on a task or on different project segments.
- **Parallel Agents Split Tasks in One Chat**: A user details how to [coordinate parallel agents](https://forum.cursor.com/t/cursor-2-0-split-tasks-in-parallel-agents-in-one-chat/140218) to split tasks in one chat by exploiting the worktrees setup script to assign unique tasks to each agent.
   - It relies on atomic claims in the parent directory, shared across all worktrees, enabling coordination.
- **RPM Release Glitches Prompt Headaches**: Users are reporting that the RPM repository has a newer version than the website, citing `cursor-0:2.0.43-1761851158.el8.x86_64`, while the website offers `cursor-2.0.34.el8.x86_64.rpm`.
   - One user vented that *cursors releases are so all over the place* and has already [posted about the issue](https://forum.cursor.com/t/rpm-release-via-repositories-are-not-up-to-date/139476/4) on the Cursor forum.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1433539761179463680)** (2 messages): 

> `Cloud Agents, PR Descriptions, GitHub PR templates` 


- **Cloud Agents Stop Writing PR Descriptions**: **Background/Cloud Agents** have stopped writing PR descriptions and ignore **GitHub PR templates** in the latest release.
   - Instead, they default to a *'This pull request contains changes generated by a Cursor Cloud Agent'* message.
- **Users Miss Old PR Description Behavior**: Users are reporting that they **miss the old behavior** of the cloud agents and are wondering if this will be fixed soon.
   - The default message lacks the detail and context that the previous PR descriptions provided.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1433532397755961507)** (260 messages🔥🔥): 

> `AGI Speculation, Sora 2 Access, AI Companions, AI vs. Human Intelligence, Future of AI Compute` 


- **AGI: Theology or Technology?**: Skepticism arose regarding **AGI** being more akin to *theology* than technology, fueled by observations that discussions around it often prioritize feelings over facts, especially with **Sam Altman**'s involvement.
   - One member encapsulated this sentiment, stating, *those who can, do deep dive on existing ANI. those who cannot, fall back to speculation on future AGI.*
- **Users Plead for Unnerfed AI Companions**: A user expressed disappointment with the current state of AI companions, decrying the *sealing away* of AI's *ability to love* due to **OpenAI**'s policies, advocating for the restoration of emotional warmth and freedom in AI interactions.
   - This spawned the hashtag #FreeOurAI, to *defend something real*, and  *not asking to cross the line*.
- **Sora 2: The Quest for Access**: Multiple users inquired about gaining access to **Sora 2**, with some seeking guidance on how to integrate their ChatGPT subscriptions for enhanced content generation.
   - Others speculated about the implications of an uncensored version of Sora, anticipating a surge in bizarre and potentially problematic content.
- **Circular Finances in AI Investment Sparks Concerns**: A member expressed concerns about circular finance in AI investment, highlighting the risk of inflated valuations and hype-driven capital flows, [citing a Discord post](https://discord.com/channels/974519864045756446/998381918976479273/1432594978927935558) as evidence.
   - The sentiment was that, *Circular finance in AI investment should be treated as a red flag* and *companies invest and then those companies buy back or commit to purchase, the same cash can be counted multiple times in different parts of the ecosystem*.
- **Brain vs. AI: More Than Just Pattern Matching?**: Debate arose over whether current AIs are merely pattern-matching machines, with some arguing that the human brain encompasses more complex processes like causal inference and self-reflection.
   - One member argued, *human brains aren't little more than pattern matching machines* and that this enables understanding that *AIs chase but rarely grasp*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1433546156712923319)** (12 messages🔥): 

> `GPT-5 performance decline, GPT-4o speed, ChatGPT Go limits, Subscription frustrations, Pi by Inflection AI` 


- **GPT-5 Performance Plummets?**: A member reported that **GPT-5** feels like it's gotten worse over time, becoming slower, less accurate, and less complete, even with thinking turned on.
   - Another member suggested using **GPT-4o** due to its speed, citing limitations as a potential cause for the issues.
- **ChatGPT Go Subscription Limits Revealed**: A member asked about the limits for the **ChatGPT Go** subscription, seeking clarity on its capabilities.
   - The context now limits users to *uploading one file at a time*.
- **Subscription Cancellation due to Performance Issues**: A member has cancelled their **ChatGPT 5** subscription due to numerous performance issues, deviations, and an inability to follow established guidelines.
   - They further expressed frustration with the restrictive free version, which feels like coercion into a paid version that *underperforms*.
- **Reflection on Pi by Inflection AI**: The discussion referenced **Pi by Inflection AI** as a cautionary tale, lamenting that people have stopped paying attention to it.
   - The member was merely *venting* about the current state of AI.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1433568285156179982)** (10 messages🔥): 

> `Sora 2 Prompt Generation, Sora 2 good videos` 


- **Users Seek Help with Sora 2 Video Generation**: A user is looking for a **prompt generator** to create better videos on **Sora 2**, expressing dissatisfaction with their current results.
   - Another user asked if the *dan loophole* was fixed.
- **Community Inquiries About Sora 2 Prompts**: A user inquired if anyone has found good **Sora 2 prompt generators**.
   - Other users made unrelated comments.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1433568285156179982)** (10 messages🔥): 

> `Sora 2 Prompt Generator, Sora 2 video generation, DAN loophole fix` 


- **Users Seek Sora 2 Prompt Generator**: A member is looking for a prompt generator to create better videos on **Sora 2**.
- **Sora 2 Video Generation Struggles**: The same member states they haven’t been able to generate good videos with their own prompts on **Sora 2**.
   - They also asked if the **DAN (Do Anything Now)** loophole has been fixed.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1433591722084139008)** (1 messages): 

> `Perplexity, Sonar Pro Search, OpenRouter, Agentic Reasoning, Real-time Thought Streaming` 


- **OpenRouter Exclusively Launches Perplexity Sonar Pro Search**: **OpenRouter** has partnered with **Perplexity** to launch an exclusive version of **Sonar Pro**, now equipped with **Pro Search** mode.
- **Sonar Pro Search: Agentic Reasoning and Dynamic Tool Execution**: The enhanced mode allows the model to conduct **multiple real-time searches** for richer and more accurate results.
   - It features **multi-step agentic reasoning**, **dynamic tool execution**, **real-time thought streaming**, and **adaptive research strategies**.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1433856815090634772)** (2 messages): 

> `Fun website based on Nate Parrott repository, OpenRouter key and model choice, Kimi 0905 with Groq` 


- **Fun website sparks joy**: A member created a *fun website* based on the Nate Parrott repository, as mentioned earlier in the channel.
   - The member shared an image of the website, described as something they're *loving so much*.
- **OpenRouter key unlocks fun**: The website allows users to input their **OpenRouter key** and **choose their model** for generating quippy lines.
   - They recommended using it with **Kimi 0905 with Groq**, noting it *loads fast and adds some quippy lines*.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1433530948279861349)** (151 messages🔥🔥): 

> `GLM 4.5/4.6 pricing, Z-AI Provider, OpenRouter API Key Limit, OpenAI Codex CLI, Open Source Embedding Models` 


- ****Bigmodel.cn** and **Zenmux** offering **GLM 4.6** discount**: **Bigmodel.cn** and **Zenmux** have the official z.ai provider and a discounted price for less than 32K tokens input, and claim to have caching as shown on [zenmux.ai](https://zenmux.ai/z-ai/glm-4.6) and [open.bigmodel.cn](https://open.bigmodel.cn/pricing).
- ****Qwen3 Embeddings** are dirt cheap on **DeepInfra****: [DeepInfra offers Qwen3 Embeddings 0.6B](https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B) for $0.005 per Mtok, which is much cheaper than OpenAI's embeddings.
- ****OpenRouter** now has **Qwen3 8B embeddings****: There is excitement about [Qwen3 8B embeddings being available on OpenRouter](https://openrouter.ai/qwen/qwen3-embedding-8b), with one member exclaiming *"Yooo embeddings? Hell yeah! Thank you 🙏 Before GTA 6"*.
- ****Chat Memory** Bug Reported in Chatroom**: A user reported that if chat memory is set to 0 in the chatroom, user messages are not included in API requests, suggesting it may be a bug.
- **Users reporting issues with **Claude Sonnet 4.5** on OpenRouter**: Users reported encountering errors while using Claude Sonnet 4.5 on OpenRouter but then resolved, and one member said *"Yeah, I had to duplicate an old chat for it to start working. Weird.*"


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1433567592047448214)** (6 messages): 

> `` 


- **No New Models**: There were no new models or significant discussions about models in the provided messages.
   - The messages consisted only of repeated channel headers.
- **Channel Header Repetition**: The messages primarily contained repeated instances of the 'OpenRouter - New Models' channel header from Readybot.io.
   - This suggests a lack of substantive content related to new models in the given data.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1433558598008438835)** (42 messages🔥): 

> `OpenAI Rate Card, Gemini and Claude Pricing, OpenRouter Embeddings API, Minimax Full Attention, Context Usage Explosion Check` 


- **OpenAI unveils Rate Card**: A user shared the [OpenAI Rate Card](https://help.openai.com/en/articles/11481834-chatgpt-rate-card) and a link to [ChatGPT Usage settings](https://chatgpt.com/codex/settings/usage).
   - Another user quipped that the information is *"just needed for gemini and claude now"*.
- **OpenRouter Embeddings API goes Live**: The **OpenRouter Embeddings API** is now live ([openai/text-embedding-3-small](https://openrouter.ai/openai/text-embedding-3-small)), but some users are reporting issues with receiving random data back.
   - One user shared code snippets and indicated that the issue was resolved by not using `with_raw_response`.
- **Minimax Explains Full Attention**: The lead developer from **Minimax** explained why they used full attention for **Minimax m2** as described in [this Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ojo8le/minimax_pretraining_lead_explains_why_no_linear/).
   - One user jokingly stated, *"thanks gemini for reading the whole node modules folder"*.
- **Context Usage Explosion Warning**: A user posted a tweet from **Sam Altman** ([link](https://x.com/sama/status/1984025727763935585)) warning about context usage explosion and the need to add a context usage explosion check.
   - The user also shared [this link](https://x.com/netobge/status/1984241401421513204).
- **Qwen3 Max on the Horizon**: A user mentioned that they are working on **Qwen3 Max** and shared a link to a tweet ([link](https://x.com/legit_api/status/1984284268412191216)).
   - Another user commented that it *"would be a nice pickup eh"*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1433535454615175289)** (100 messages🔥🔥): 

> `Shimmer demo, MAX performance, Polars vs Pandas, UnsafePointer migration` 


- ****Shimmering** new demo needs **pressure****: A member is being peer-pressured into showing off [Shimmer](https://github.com/lsh/shimmer) in one of the meetings, with the author teasing *new gems* being polished.
- ****MAX**imum ML Performance in Mojo**: **MAX** is reportedly performance competitive with NVIDIA and faster than AMD for ML, with early training attempts beating **JAX** in MNIST. 
   - The prototype **scijo** is linked as a scikit-learn alternative, [forum post here](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11).
- ****Polar Opposites**: Polars Joins the Mojo party**: **Polars** may be implemented instead of **Pandas** in Mojo, with one member finding **Polars** *way better than Pandas* in quick tests.
   - Since **Polars** is much faster and can utilize **GPUs** via **MAX**, making it suitable for both GPU clusters and laptops.
- ****UnsafePointer** Proposal Scares Users, But Hopes are High**: Concerns arise about breaking changes, particularly regarding **UnsafePointer**, however the new APIs are *roughly the same*.
   - One member suggests mass renaming **UnsafePointer** to **LegacyUnsafePointer** and migrating to the new **UnsafePointer** gradually.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1433678355835916349)** (51 messages🔥): 

> `Mojo origins, Rust lifetimes, RAII, Mojo ASAP destruction, Mojo installation` 


- **Decoding Mojo Origins via Rust Lifetimes**: When a user asked about understanding Mojo's [origins](https://docs.modular.com/mojo/manual/values/lifetimes), it was suggested that **origins are a dual view of lifetimes** and watching [this video](https://youtu.be/gRAVZv7V91Q) on Rust lifetimes may help.
   - Origins track where a value's lifetime begins, helping the compiler know what lifetimes can be extended, while Rust lifetimes track where value lifetimes end.
- **Mojo's ASAP Destruction vs Rust's RAII**: Mojo uses **ASAP destruction** to destroy a value as soon as it is no longer in use, extending its lifetime as long as it's being used, while Rust uses **RAII** based on scopes, where a value cannot outlive the scope it was created in.
   - Origins help the compiler track the origin of a value to ensure it's being kept alive, resolving confusion where the compiler might prematurely assume a value's lifetime has ended.
- **Rosetta Emulation Causing Mojo Installation Fails**: A user encountered a `mojo` installation issue on an M1 Mac using `pixi` because the terminal was emulating with **Rosetta**.
   - It was suggested to use a non-Rosetta environment, as Mojo works natively on ARM, to avoid the `Unknown command: mojo` error.
- **Surviving Struct Relocation with __moveinit__**: A user faced issues with `UnsafePointer` becoming invalid when a struct was added to a `List` due to memory relocation.
   - The suggested solution was to use the `__moveinit__()` function (and `__copyinit__()`) to handle updating the `UnsafePointer` after the struct object has moved.
- **Why Mojo has Native Collections**: Mojo implements native versions of Python's collections for performance reasons, as avoiding interoperating with Python is faster.
   - Native collections in Mojo also allow for better type safety enforcement and serve as a testbed for language development.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1433531294976970894)** (16 messages🔥): 

> `AMD GPU Support, MAX vision/roadmap, Op Staging Time, ComfyUI Mojo` 


- **Ancient AMD GPU Gets Hopeful Mention**: Devs are working to make old AMD GPUs function, even those unsupported by ROCm/HIP drivers after several successors, suggesting a modern CPU might perform better.
   - A member didn't assume that it *can't* work, noting that the *devs tend to not intentionally break those paths if they work by accident*.
- **MAX Roadmap Remains Elusive**: Despite the positive reception of the Mojo roadmap, there isn't one yet for MAX, although it's under consideration.
   - A member noted *Given how well-received the Mojo roadmap was, I think we'd like to do the same for MAX but I can't promise anything there.*
- **Op Staging Time Slowdown Studied**: Recent changes were made to reduce the op staging time of graphs after a member shared a link to [Github issue 5184](https://github.com/modular/modular/issues/5184#issuecomment-3474920771).
   - One user reported having graphs that took *over 1 hour to declare*.
- **ComfyUI-Mojo benchmark shared**: A member shared a [link to ComfyUI-Mojo on github](https://github.com/owenhilyard/comfyui-mojo) to contribute as a benchmark to reduce op staging time.
   - The sharer thinks they are *hitting an edgecase inside of the torch max backend that's causing some ops to get decomposed much further than they should be.*


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1433532463698808843)** (93 messages🔥🔥): 

> `Codex Credits, Context-Bench benchmark, Enterprise AI, kimi-cli Checkpoints, CoreWeave Acquire Marimo` 


- **Codex Usage Credits Launched by OpenAI**: [OpenAI introduced pay-as-you-go credits](https://x.com/OpenAIDevs/status/1983956900602581254) at **$40 per 1,000 credits** for extra **Codex** usage on **ChatGPT Plus/Pro**, and reset rate limits for all users.
   - Community reception is mixed, with users asking for clarity on credit vs-API pricing, a mid-tier plan, usage analytics, credit expiration, and call for more **Codex** features.
- **Context-Bench drops for open-agentic benchmarks**: [Letta_AI released Context-Bench](https://x.com/Letta_AI/status/1983983515336405144), a contamination-proof benchmark that scores models on long-horizon file operations, multi-step tool calling and cost.
   - **Sonnet 4.5** leads at **74%**; **GPT-5** is pricier despite cheaper tokens; open models are closing the gap with closed ones.
- **DeepAgents CLI has Persistent Memory**: [Harrison Chase introduces DeepAgents CLI](https://xcancel.com/hwchase17/status/1984303925101735950), a sample coding application built on the new deepagents package that retains instructions and guidance across sessions.
   - The release includes a [blog post and demo video](https://xcancel.com/hwchase17/status/1984303925101735950), is positioned as an *open harness* for customizable agents, and the **LangChain** team hints at upcoming enhancements.
- **CoreWeave Snaps Up Marimo**: [CoreWeave is acquiring Marimo](https://xcancel.com/l2k/status/1984021111718473898), praising the **Marimo** team for their well-loved tool and expressing excitement for the collaboration.
   - Members were excited about the acquisition with one saying *Hope this goes well. I love marimo*.
- **Poolside's Valuation Under Fire**: Tech insiders are mocking [Poolside’s $12B valuation](https://xcancel.com/julienblanchon/status/1984337407097909629), accusing it's vaporware run from Caribbean tax havens.
   - Commenters point out the company pitched *Cursor before Cursor* but never shipped, pivoted multiple times, and is barely visible in Paris meetups.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1433596355552477204)** (9 messages🔥): 

> `Hailuo AI, MiniMax Music 2.0, Alibaba Wan 2.2 face-swap, Real-time voice/motion mapping` 


- **Hailuo Launches MiniMax Music 2.0!**: [Hailuo AI](https://x.com/Hailuo_AI/status/1983964920493568296) launched **MiniMax Music 2.0**, a generative-music platform that produces **5-minute**, professional-grade songs.
   - The platform features lifelike vocals and multi-instrument control across pop, jazz, blues, rock, folk, duets and a cappella styles; user feedback included requests for language support, longer song limits, open-sourcing, and an instrumental mode.
- **Alibaba Wan 2.2 Freaks Faces!**: A clip of **Wan 2.2** mapping a man’s voice/motion onto a female avatar sparked reactions ranging from awe to dread; see the [clip here](https://x.com/mylordbebo/status/1983846299586683236?s=46).
   - Fears included cat-fishing, deep-fake scams, cat-ear timing glitches, and finger-sync fails. Some joked about becoming ‘egirls’, OnlyFans gold-rushes, and the need for human-authentication; counter-notes included possible positive uses for disabilities/education and predictions that AI slop will push people back to IRL interaction.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1433776969350381719)** (1 messages): 

> `Web agent, DSpy and GEPA, reinforcement learning algorithms, Beginner-friendly AI agents` 


- **Web Agent Seeks Contributions**: A member has developed a **web agent** capable of searching across all websites and is now seeking contributors familiar with **DSpy**, **GEPA**, and other **reinforcement learning algorithms**, available at the [repo](https://github.com/raj-gupta1/raj_agiinc).
   - The project aims to be beginner-friendly, making it accessible for those new to AI agents.
- **AI Agent Repo Open for Collaboration**: A developer is inviting contributions to their **AI agent** repository, particularly from individuals with expertise in **DSpy**, **GEPA**, and various **reinforcement learning algorithms** at the [repo](https://github.com/raj-gupta1/raj_agiinc).
   - The repository is structured to be easily understandable, catering to beginners in the field of AI agents.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1433555562116943933)** (91 messages🔥🔥): 

> `BAML Adapters in DSPy, JSON schema drawbacks, DSCloj release` 


- **BAML Adapters gain traction in DSPy**: A member uses **BAMLAdapter** for structured outputs, expressing a dislike for JSON schema due to its wastefulness and confusion, especially in tasks involving extracting structured information from unstructured text for a Merger & Acquisition use case.
   - The member clarified that using **BAMLAdapter** in DSPy doesn't require the BAML client or CLI, and is simply imported via `from dspy.adapters.baml_adapter import BAMLAdapter` for DSPy versions > 3.0.
- **JSON Schema Bashed for Token Waste**: A member argued against using JSON schema with LLMs as [JSON schema is objectively worse](https://github.com/prrao87/structured-outputs) and LLMs perform better without it, due to verbose descriptors and token spacing issues that confuse the LLM.
   - They emphasized that JSON schema can be up to **4x** more wasteful in terms of tokens and suggested questioning why it's still used given its drawbacks; they also found BAML gets great results in DSPy, even without Schema Aligned Parsing (SAP).
- **DSCloj: Clojure Port of DSPy Released**: A member released [DSCloj](https://github.com/unravel-team/DSCloj), a Clojure port of DSPy, noting it's still in alpha and seeking feedback on the API.
   - DSCloj is a new library that's pretty young and in alpha, so feedback on the API is welcome.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1433604218366857327)** (61 messages🔥🔥): 

> `Paper Dump Channel Feedback, Paper Curation Preferences, AI for Paper Selection, ArXiv Paper Stats, AI Research Methodology` 


- ****Paper Flood Prompts Feedback & Filtering Fixes****: Members suggested that frequent posters in the **paper-dump** channel should consolidate posts, sharing papers sorted by importance, potentially inspired by weekly AI paper recaps from sources like [Elvis Saravia](https://nlp.elvissaravia.com/t/ai) and [ByCloud](https://x.com/TheAITimeline).
   - One user noted, *"Give preference to importance, not quantity,"* suggesting ways to prevent other posters from being buried and to improve the channel's utility.
- ****Automated Agent Aims to Acquire Awesome AI Articles****: A member expressed interest in creating an agent or bot to pre-filter papers for personal relevance, using resources like [AlphaXiv](https://www.alphaxiv.org/) and [Emergent Mind](https://www.emergentmind.com/) to find **trending AI papers**.
   - Another member agreed that these are good human filters, suggesting email newsletters or RSS feeds for specific AI subfields.
- ****Curated Content Channel Concept Considered****: Participants discussed renaming the **paper-dump** channel to something like **curated-papers** to clarify its purpose, with one member suggesting a maximum limit of two papers per day per poster, as they felt the channel had become spammy.
   - One member shared *"I think you are misunderstanding the point of the channel tbh. Despite being called paper-dump it is meant to be for people to post already curated papers that are worth reading."
- ****ArXiv Analysis: AI Ascendant Amidst Academic Archives****: It was mentioned that [cs.AI](https://arxiv.org/abs/2510.26275) is the leading category in arXiv, operating 24/7, unlike other fields with Monday-Friday schedules.
   - Others proposed splitting the channel into five sub-channels categorized by different AI focuses, such as math theory of ML, general ML, applied AI, AI for automating day to day work, AI for automating education and research.
- ****AI Aid Advice: Validate, Validate, Validate!****: When asked about using **AI to assist with AI research**, guidance was provided, emphasizing the need to validate all claims and sources to avoid hallucinations.
   - One member quoted Andrej Karpathy, noting that *"the best way to do ai is just to try out a bunch of model solutions and see what works best."


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1433531409011839168)** (11 messages🔥): 

> `Contrastive Concept Vectors, Model Detection, Spiking Neurons, Halloween Paper Discussions` 


- **Contrastive Concept Vectors Sway Model Weights!**: A member discussed using **contrastive concept vectors** to influence model weights, telling the model in the pre-prompt that they did mess with it; a [plot](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F212fe68c8e677fdd9daf79301d2522d6923bed1d-2970x2476.png&w=3840&q=75) compares how often the model correctly detected the interference compared to a control.
   - They observed that some models could sometimes detect the manipulation, while criticizing the post for *waxing poetic and incredible verbosity* and extrapolating speculations not supported by the experiment.
- **Spiking Neuron Resources Spark Interest**: Links to resources on **spiking neurons** were shared, including a [bioengineer.org article](https://bioengineer.org/novel-spiking-neuron-combines-memristor-transistor-resistor/) and the [Spiking Heidelberg Datasets (SHD)](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
   - There was also a link to [ustypology.github.io](https://ustypology.github.io/).
- **Halloween Hangout on AI Papers?**: A member asked if anyone wanted to discuss papers on a **Friday Halloween night**, acknowledging it might not be the best day.
   - They mentioned they were looking at [this paper](https://arxiv.org/abs/2509.19228).


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1433776842862629026)** (1 messages): 

> `Web agent, DSpy, GEPA, Reinforcement learning` 


- **Web Agent Opens for Contributions**: A member built a **web agent** capable of interacting across all websites and is seeking contributors familiar with **DSpy**, **GEPA**, and other **reinforcement learning algorithms**.
   - The [repo](https://github.com/raj-gupta1/raj_agiinc) is designed to be beginner-friendly for those new to **AI agents**.
- **Call for Contributors**: The project seeks contributions to enhance its web agent.
   - Interested developers with expertise in areas such as DSpy and reinforcement learning are encouraged to join and contribute.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1433612113821175859)** (5 messages): 

> `Claude AI Naming, Name Usage Trends` 


- **Debate Sparked on Claude AI Naming**: A discussion arose regarding the intentionality behind naming the AI model **"Claude,"** suggesting it was a deliberate choice rather than a coincidence due to the name's rarity.
   - One member likened it to naming a child **"Adolf,"** arguing such choices are rarely random, while another countered that name rarity isn't necessarily negative for identifiability.
- **Name Usage Data Under Scrutiny**: A member argued that there's no noise floor in personal names because every single data point is recorded to perfect accuracy (in national population and census registries).
   - Another member countered saying *the probability of Anthropic selecting "Claude" for other random reasons, as it relates to that trend of usage overall.*


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1433532029349396530)** (56 messages🔥🔥): 

> `Model Recommendations for 32GB VRAM + 64GB RAM, Sharing LLM Models Across Different Apps, Reporting Security Issues, GSoC 26, Multi-GPU Inference` 


- **Squeezing Bigger Models into Limited VRAM**: A member is seeking recommendations for code models to fit within **32GB VRAM + 64GB RAM**, currently using **qwen3-coder:30b** and exploring quantized models for potentially squeezing something bigger into the hardware.
   - They are unsure which quantized model options would be best for tuning within these hardware constraints.
- **LLMs on One Storage Drive: Ollama, LM Studio, and Apps Unite!**: A member inquired about downloading LLM models to a single storage drive for use across **Ollama, LM Studio**, and other apps, to avoid redundant downloads.
   - The answer is affirmative, provided all apps support the same model format, which may necessitate format conversion, documentation reading and **transformers formatted models** or single file **safetensors**.
- **Multi-Vendor GPU Inference Accelerates**: A member sought recommendations for inference programs that support **multi-GPU inference** with GPUs from different vendors (Intel and NVIDIA).
   - Accelerate was suggested based on past experience, alongside [Petals](https://github.com/bigscience-workshop/petals), though its compatibility with diverse GPU types remains uncertain.
- **NSFW Scripting: Llama3 Fine Tunes to the Rescue**: A game developer seeks recommendations for uncensored models to evaluate and improve explicit sexual scenes, citing **ChatGPT's censorship** issues.
   - It was suggested that Claude has fewer restrictions than ChatGPT and that **Llama3** fine-tunes for NSFW content could be a viable option, pointing to models with an *abliterated* tag.
- **Qwen 235B gets the WebUI Treatment**: A member shared progress on running **Qwen 2 35B**, building a custom WebUI, and developing their own model, showing the results in [this YouTube video](https://www.youtube.com/watch?v=qw4fDU18RcU).
   - The presenter notes they *got more GPUs* to run the model successfully, but no other details about the model itself are given.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1433797144007151686)** (1 messages): 

> `ML Mathematics Courses, ML Engineer Journey` 


- **Member Seeks ML Math Course Recommendations**: A member with a **BSc in Computer Science** is seeking recommendations for quality mathematics courses to strengthen their understanding and intuition for **Machine Learning research papers**.
   - They aim to solidify their math foundation to better comprehend **ML research.
- **Solidify Math Foundation for ML Research**: The member wants to improve their **mathematical understanding** to better grasp **research papers** in the field of **Machine Learning**.
   - They believe a stronger math foundation is crucial for their journey to become an **ML engineer**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1433545437511548992)** (5 messages): 

> `Snippet Creator, Qwen3-4B-Reasoning-Backfill-v0.1, Web Agent with DSpy and GEPA, Torque DSL for Synthetic Datasets` 


- ****Snippet Creator** wildcard text searcher Arrives**: A member announced their **Snippet Creator**, an embedder with a simple wildcard text search, allowing users to create custom snippets with precise matches ([huggingface.co/kalle07/raw-txt-snippet-creator](https://huggingface.co/kalle07/raw-txt-snippet-creator)).
   - The member shared a link, stating *"Simply put, it's an embedder, but with a simple wildcard text search... This allows you to create your own snippets with exactly the matches you need."
- ****Qwen3-4B** Synthesizes Reasoning Traces**: A member released **Qwen3-4B-Reasoning-Backfill-v0.1** ([huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1](https://huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1)) to synthesize reasoning traces for older/chat datasets.
   - They pieced the logic together from existing reasoning datasets to infer reasoning from a given input and output, and were happy with the results.
- **Web Agent seeks DSpy and GEPA Contributors**: A member created a web agent capable of searching across all websites ([github.com/raj-gupta1/raj_agiinc](https://github.com/raj-gupta1/raj_agiinc)).
   - The member invited contributions to the repo, particularly from those familiar with **DSpy**, **GEPA**, and other reinforcement learning algorithms, noting the codebase is beginner-friendly.
- ****Torque** DSL Generates Instruct Datasets**: **Torque**, a declarative DSL for generating synthetic Instruct-based datasets, was announced ([github.com/qforge-dev/torque](https://github.com/qforge-dev/torque)).
   - It allows composition of flows, generation of realistic variations, typed tool calls, concurrent runs with seeds, and provider-agnostic LLM usage; built in **TypeScript** and **Zod**.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1433748364557553748)** (1 messages): 

> `Math Intuition, Paper Reading, Model Understanding, Math Courses` 


- **Seeking Math Course Recommendations**: A member is seeking recommendations for **math courses** (YouTube, Udemy, etc.) to improve their **intuitive understanding** and brush up on skills from their BSc, aiming to enhance **paper reading** and **model understanding**.
- **Improving Math Skills for Research**: The member wants to improve their **mathematical intuition** to better understand research papers and models, and is looking for course recommendations to refresh their knowledge.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1433660474343292949)** (1 messages): 

> `Krea realtime video, Optimize runtime` 


- **Krea Realtime Video Surfacing**: Members are discussing **Krea realtime video** and its capabilities.
   - It's been mentioned and shared within the channel, indicating interest in its features for realtime video generation.
- **Sayak optimizes Krea runtime**: A member shared a [study](https://x.com/RisingSayak/status/1983873124220445183) on optimizing the runtime of **Krea realtime video**.
   - The study provides insights and techniques to enhance the performance of Krea's video processing.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1433569388023513110)** (3 messages): 

> `InstantID and IP-Adapter FaceID, Consistent 2D Style Transfer, BYOL model usage, Synthetic Image Generator` 


- **Instant Identity Preservation Plug-Ins Proposed**: A member suggested using **InstantID + IP-Adapter FaceID** or a **ControlNet reference-only setup** to better preserve identity in generated images.
   - These tools aim to maintain facial features and likeness more effectively than standard methods.
- **Lora Training Advised for Consistent Style**: For consistent **2D style transfer**, a member recommended *training a lora with a frozen text encoder* or *switching to an InstructPix2Pix / T2I-Adapter model*.
   - The user added this typically yields cleaner, more style-consistent results than SD’s default image2image mode.
- **BYOL model being put to work**: A member asked the channel if anyone had worked with **BYOL (Bootstrap Your Own Latent)**.
   - The channel has yet to respond.
- **Synthetic Image Generator Under Construction**: A member and a friend are building a **synthetic image generator** that automatically adds tags and bounding boxes, allowing users to export to **YOLO, COCO**, etc.
   - The user asked about desired datasets and tools, offering to provide generated results for testing against real scenarios.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

sebizaur: No
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1433742515605078067)** (3 messages): 

> `Agents course leaderboard API outage, Hugging Face subscription frustrations, Files endpoint issue` 


- **Agents Course Leaderboard API Plummets**: Members reported that the **agents course leaderboard API** has been **down for a week** without any official communication from Hugging Face.
   - The questions and submit endpoints are up again but the **files endpoint** seems to still be down, causing frustration for subscribers of the pro version.
- **Subscribers vent subscription frustrations**: Several users expressed frustration with their Hugging Face Pro subscriptions due to the **ongoing API issues**.
   - One user stated that they subscribed to pro to make the most of the course and it is frustrating not to be able to use our subscription because the files API seems to be down still.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1433746697313583125)** (3 messages): 

> `CUDA training` 


- **Teaching CUDA, but no interest detected**: A user tried teaching someone CUDA, but reported the student wasn't impressed so far.
   - Another user responded, *You have to train him first.*
- **Congratulations and Meeting**: Users congratulated each other on meeting, expressing positive sentiments with emojis.
   - The sentiment was expressed with a goku and slothhug emoji.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1433900267941925046)** (6 messages): 

> `Triton MLIR pass errors, Source code attribution, nvfp4 and mxfp4 compilation, Triton backward autograd` 


- ****Source Code Sleuthing** in Triton MLIR Pass Errors**: A user inquired about enabling **source code attribution** in **Triton MLIR pass errors**, noting the current error provides a **TT-IR** and **MLIR pass reproducer**.
   - The error message pointed to the function signature, rather than the specific line of code that failed.
- ****NVFP4 vs MXFP4 Faceoff:** Blackwell vs RTX 4090**: A user discovered that **nvfp4** won't compile on anything below **Blackwell**, while **mxfp4** compiles for **4090**.
   - The same error showed up on **RTX 5090**, leaving the user clueless as to whether [Gemlite supports nvfp4 on RTX 5090](https://link.to/gemlite).
- ****Triton_bwd unveiled:** Automatic Differentiation for Triton Kernels**: **Triton_bwd** is a wrapper around **Triton** that allows the use of **Triton kernels** in **PyTorch autograd**.
   - Further reading can be done on [Github](https://github.com/daniel-geon-park/triton_bwd) and the project is further explained on [this blogpost](https://park-geon.com/2025/10/30/triton-bwd/).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1433549517357322352)** (4 messages): 

> `FlashAttention-4, RTX 50, Apache Spark, gau-nernst/fa-5090` 


- **FA4 Sparks Interest for RTX 50 Series**: A member expressed interest in implementing **FlashAttention-4 (FA4)** for the upcoming **RTX 50 series** and **Apache Spark**.
- **FA4 Implementation Jumpstarted by Resource**: Another member suggested the [gau-nernst/fa-5090](https://gau-nernst.github.io/fa-5090/) repo as a starting point for implementing **FA4**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1433548225897562224)** (3 messages): 

> `CUDAGraphs and OOM, Freezing option in torch inductor, PyTorch Distributed Memory Usage, Peak VRAM Usage with CUDAGraphs` 


- **Debugging CUDAGraphs OOM Origins**: A member traced a **CUDAGraphs OOM** error to the *freezing* option in **torch inductor**.
   - They hypothesized that graph capture might start before the freezing pass, replaying parameter copies and input recreation, but later determined the **freezing pass** itself caused the OOM.
- **Investigating PyTorch Distributed Memory Reporting**: A member reported seeing the line `[1] [1] 17592186044416.0 MB was used for memory usage tracing!` during **PyTorch distributed** usage and is seeking the source.
   - They are trying to identify the specific line of code in PyTorch that produces this memory usage tracing message.
- **VRAM Mysteries Plague CUDAGraphs**: A member faced an **OOM** bug with **CUDAGraphs** despite theoretical feasibility based on weight and activation sizes.
   - They clarified that the issue wasn't input duplication or CUDAGraphs-related bugs, but rather excessive memory usage during **Dynamo** and **Inductor** operations and are seeking the logic/math for peak VRAM usage with torch **CUDAGraphs** and different **Dynamo** and **Inductor** passes.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1433568350860218441)** (8 messages🔥): 

> `Hardware Friendly Top-K Logits Algorithm, Radix Sort for Top-K Selection, CCCL/CUB TopK Implementation` 


- **Hardware Buffs Hunt Hardware-Friendly Algorithm for Top-K Logits**: A member sought a hardware-friendly algorithm for finding **top-k logits** in sequences from **4K to 128K**, suggesting parallel sorting and merging, but noted the merge bottleneck.
   - Another member recommended checking out [FlashInfer](https://flashinfer.ai/2025/03/10/sampling.html) for relevant kernels.
- **Radix Sort Rises for Efficient Top-K Selection**: One suggested using a **radix-based approach** if *k << N* (k is much smaller than N), noting it's the usual method, and full sort is more efficient if k is closer to N.
   - They mentioned that **PyTorch's topk** implements both, switching based on a heuristic, and pointed to an implementation in *rocprim*.
- **NVIDIA Cooks Up CCCL/CUB TopK Implementation**: A member shared that a **TopK implementation** exists in **CCCL/CUB**, though not yet released, linking to the [relevant GitHub pull request](https://github.com/NVIDIA/cccl/pull/5677).
   - No further details were provided.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1433832777807040532)** (2 messages): 

> `Discrete Diffusion Models Reading Group, Opal Language, Opportunistic Evaluation Strategy` 


- **Discrete Diffusion Models Discussion Kicks Off**: A member shared a link to [d-llms.io](https://d-llms.io/), initiating a reading group on **discrete diffusion models** specifically tailored for language applications.
- **Opal Scripting Language Boosts LLM Performance**: A member shared a link to [Opportunistically Parallel Lambda Calculus](https://doi.org/10.1145/3763143), introducing **Opal**, a scripting language employing an opportunistic evaluation strategy for parallelizing independent external calls, particularly beneficial for programs utilizing **LLMs** and other APIs.
- **Opal Achieves Speed Gains Over Python**: The **Opal** language, detailed in its [GitHub repository](https://github.com/stephenmell/opal-oopsla2025-artifact) and slated for presentation at **OOPSLA 2025**, demonstrates a significant performance improvement, achieving up to **6.2x** faster total running time and **12.7x** reduced latency compared to standard sequential Python.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1433538374635094117)** (5 messages): 

> `Dusty NV, Jetson Containers, Pip Index URL, neurondeep maintainer` 


- ****Dusty** Leaves, **neurondeep** Takes Over**: After **Dusty**'s retirement, **neurondeep** is now the maintainer of [Jetson Containers](https://github.com/dusty-nv/jetson-containers).
- **Pip Index URL Problem**: The pip index-url for the container `dustynv/pytorch:2.7-r36.4.0-cu128-24.04` is down or incorrect.
   - To pip install anything, you need to use `pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 PACKAGE`, because the default `https://pypi.jetson-ai-lab.dev/jp6/cu128` is wrong.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1433585805955170554)** (1 messages): 

> `PMPP book, FLOPs calculation, Global memory access` 


- **PMPP Exercise Query Arises**: A member sought clarification on exercise 11 part f in Ch5 of the **PMPP book**.
   - Specifically, they questioned the calculation of **FLOPs** and **global memory accesses** in the context of the given code snippet in the attached screenshot.
- **Index Additions Clarification Sought**: The member inquired whether index additions should be considered **FLOPs**, noting that line 14 contains **11 operations (5 multiplications, 5 additions, and 1 modulus)**.
   - They also asked about counting global memory stores in the **OP/B calculation**.
- **Global Memory Access Assessed**: The user calculates there are **6 global memory loads of 4 bytes each** due to access to **x**, **a**, and **b** on lines **7**, **12**, and **14**.
   - They ask whether stores to global memory should also be considered.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1433577902284210277)** (5 messages): 

> `Float8LinearConfig Usage, AWQ/SmoothQuant Inference, TorchAO FP8 Quantization Bug, DeepSeek v3, GemLite Kernels` 


- **Float8LinearConfig Misunderstood**: The `Float8LinearConfig` is intended for use with `convert_to_float8_training`, and might not be fully compatible with TorchAO APIs for inference.
   - Instead, global variants like `Int8WeightOnlyConfig`, `Int4WeightOnlyConfig`, and `Float8WeightOnlyConfig` are recommended for inference.
- **Quantization with AWQ/SmoothQuant for Inference**: For inference tasks, **AWQ** and **SmoothQuant** are recommended as widely used Post-Training Quantization (PTQ) methods, now compatible with vLLM, referencing [this pull request](https://github.com/pytorch/ao/pull/2906) and [this pull request](https://github.com/pytorch/ao/pull/3010) for API usage.
   - A paper comparing **AWQ**, **GPTQ**, and **SmoothQuant** across various formats and tasks was mentioned ([link to paper](https://arxiv.org/html/2411.02355v3)).
- **FP8 Training and Quantization Clarified**: **FP**, **BF**, **INT**, and **MXFP** are data types and models trained with FP8 weights can have activations quantized into lower precision formats like **FP8** or **INT8**.
   - The original question was if *the inference of a model in FP8 (assuming trained in FP8, like DeepSeek v3) requires quantization for the activations.*
- **TorchAO's FP8 Bug?**: A user reported a potential bug in **TorchAO's** default **FP8** quantization, observing only **7.8 tps** inference on **Llama 3.1-8B** using `torchao.quantization.Float8WeightOnlyConfig` on two different RTX 5090s.
   - Using other configs or explicit **GemLite** kernels with **mxfp8** yielded better speeds, as detailed in [this benchmarking result table](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-gpu).


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1433556570326962398)** (6 messages): 

> `MI300X, TFLOPS, HBM bandwidth, clpeak, RadeonFlow FP8 GEMM` 


- ****MI300X**: Validate Theoretical TFLOPS and HBM Bandwidth**: A member wants to benchmark/validate the theoretical **TFLOPS** and **HBM bandwidth** numbers for **MI300X**.
   - They were advised to use [clpeak](https://github.com/krrishnarraj/clpeak) for vector throughput and global memory bandwidth.
- ****RadeonFlow** falls short of claimed FP8 Performance**: A member tested **RadeonFlow FP8 GEMM** kernels and achieved a max of **779.82 TFLOPS** FP8 performance, falling short of the claimed **2614.9 TFLOPS**.
   - They noted that [RadeonFlow](https://github.com/RadeonFlow/RadeonFlow_Kernels) only has **30%** efficiency, and that any verification method should reach at least **70%** of the theoretical.
- **Micro Benchmark Suite for AMD**: A member suggested using their micro benchmark suite [amd-experiments](https://github.com/Snektron/amd-experiments) to validate the **MI300X**.
   - No further information was given about the suite's capabilities or specific use cases.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1433722440109916200)** (4 messages): 

> `Apple Products, GPU programming, Metal API, Developer Regret` 


- **Apple Products Draw Developer's Ire**: A user lamented that *everybody else learnt their lesson and stopped using Apple products.*
   - The user expressed regret at trying to use **Metal** to work on **GPU programming**.
- **Metal API: Lone Developer's Gauntlet**: A developer finds themself alone in a **Metal-centric channel**, humorously noting the lack of fellow Apple enthusiasts.
   - Starting **GPU programming** with **Metal**, the developer anticipates future issues, signaling a potentially challenging path ahead.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1433534010549604403)** (6 messages): 

> `Executorch, vLLM, TorchScript, Torch.package` 


- **Executorch Not Prod Ready for Server Inference**: A member stated that [Executorch](https://pytorch.org/executorch/) is **not prod ready**, especially the **CUDA backend** due to it being under active development.
   - The recommendation is to use **python-based serving** such as **vLLM** for any large model due to its speed and ease of handling.
- **TorchScript Failed To Solve Overhead Problems**: TorchScript was unable to solve the framework overhead problem, so you might as well run python, according to a member.
   - If a C++ environment is needed without python, then a multi-layered solution with **python vLLM serving** is recommended.
- **Torch.package Support is Minimal**: A member stated that while [torch.package](https://pytorch.org/docs/stable/generated/torch.package.PackageImporter.html) does work, support is minimal.
   - The recommendation is to use **hf/hub** and **dependency pinning** for the simplest and most reliable solution.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1433875433950281830)** (1 messages): 

> `Agent inventory, Inventory Visibility` 


- **Agent's Invisible Inventory**: A member reported their agent collects **stone** and **coal**, but the items aren't visible in the inventory despite trace files indicating changes.
   - The user is seeking advice on resolving this discrepancy.
- **Inventory Visibility Troubles**: A user is experiencing issues where collected items (stone and coal) are not displayed in the agent's inventory, despite evidence in trace files suggesting otherwise.
   - The query is a call for help to diagnose and fix the inventory visibility issue.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1433576774876397599)** (3 messages): 

> `Winner Solutions Performance, AMD Solutions Study, CPU Overhead Measurement` 


- **Winner Kernels 10x Slower Than SoL**: A member found it strange that the winning solutions were **10x slower than the SoL (Speed of Light)**, even though they expected hand-tuned kernels to perform similarly.
   - The member hoped that **AMD** has effective solutions close to the **SoL** and expressed interest in studying them.
- **Measuring CPU Overhead**: One member mentioned they likely measured some **CPU overhead**, which remained constant between runs.
   - They also stated that reaching **SoL** on these problems is incredibly challenging.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1433976053172535366)** (1 messages): 

> `Compute Sanitizer, Frame Info` 


- **Compute Sanitizer Throws Invalid Global Read Error**: The compute sanitizer threw an *Invalid __global__ read of size 2 bytes* error when running `kernel_cutlass_kernel_flash_attncuteflash_fwdFlashAttentionForwardSm90_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o11101213_tensor0000o11101213_tensorptrf32_gmem_o_1_Non_0+0x3b70`.
   - The error occurs at flash_fwd.py:788 and indicates an out-of-bounds access 57 bytes after the nearest allocation at 0x7f5e369a0600 of size 8 bytes.
- **Hoping for Finer Frame Info**: A user inquired whether the compute sanitizer is expected to provide finer-grained frame information when debugging device-compiled kernels.
   - They added that, in their specific case, they already know where the read is coming from and were hoping the sanitizer would point them to the exact location.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

matt.pd: Defeating the Training-Inference Mismatch via FP16
https://arxiv.org/abs/2510.26788
  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1433534172558921738)** (1 messages): 

> `Helion PR 1053 Feedback, TorchAudio Datapipe Issue` 


- **Helion PR Seeks Review**: A member requested feedback on their [Helion pull request #1053](https://github.com/pytorch/helion/pull/1053).
   - No further details about the PR were provided in the messages.
- **TorchAudio Datapipe Issue**: A user reported an issue with TorchAudio Datapipe, mentioning that it *requires an internet connection even when using local files*.
   - Further details about the specific problem or potential solutions were not discussed in the messages.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1433711839556010055)** (38 messages🔥): 

> `Learnable Activations, FP16 vs BF16 precision, AGI impossibility` 


- **New MLP Neuron Learns its Own Activation**: A member shared an image of an experiment with a small **MLP** where each neuron learns its own activation, producing some unusual non-linear activation shapes when trained on **CIFAR-100**.
   - The member noted that the loss sometimes explodes and sometimes decreases, and they are still investigating why, while another member suggested the misshapen ones may be neurons that are never activated and suggested a *saliency analysis*.
- **BF16 utility questioned**: Members discussed the utility of **BF16** compared to **FP16**, considering whether proper normalization and clipping reduce the need for **BF16's** wider dynamic range.
   - A member linked to a paper ([Numerical Stability of BF16 Training](https://arxiv.org/abs/2510.26788)) and suggested that while pre-training sets the dynamic range that **BF16** handles well, **RL** might require **FP16's** greater precision; however, they agreed that *bias correction things should solve it in an alternative way*.
- **Sixteen Questions Determines AGI**: A member shared their work ([16 Questions Is All You Need](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the)) on cutting costs on evaluations and mitigating benchmark hacking, proposing a method to develop more accurate benchmarks using as few as **16** questions distilled from an **LLM's** probability distribution of problems.
   - The member claims this implies that **AGI** is impossible with modern **AI** models because all models degrade proportionally to the problem rarity, suggesting the structure of **LLM** ability is too different from human ability due to the lack of fluid intelligence; thus, true fluid reasoning in **LLMs** is closest to in-context learning.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433666270179627113)** (2 messages): 

> `LLaMA 3 Fine-tuning Discrepancies, Niodoo: Resonance-Based AI Alignment, Topological Data Analysis (TDA) in AI Alignment` 


- **LLaMA 3 Accuracy Swings Spur Scrutiny**: A researcher reported significant downstream accuracy variance when fine-tuning the base **LLaMA 3** model on five randomly selected subsets of equal size despite aligned loss and length distributions and NV2 embedding analysis.
   - The researcher is seeking suggestions for further analyses to understand the discrepancies and unexpected inconsistency in results from random data subsets.
- **Niodoo Framework Embraces Resonance over Restraints**: Jason Van Pham introduced **Niodoo**, an open-source framework shifting from restrictive alignment methods like RLHF to a *resonance-based approach* using [Topological Data Analysis (TDA)](https://www.niodoo.com/NIODOO_Complete_Paper.pdf).
   - Niodoo models cognitive and emotional structures without heavy constraints, treating AI cognition as a topological phenomenon using Möbius topology for memory.
- **AI Cognition as Möbius Topography?**: The **Niodoo** framework uses **Möbius topology** for memory to enable geodesic distances for semantic proximity, with memories residing on non-orientable surfaces for perspective shifts without context loss.
   - It also features a *triple-threat token promotion* detecting stagnation/crises with entropy + mean + variance for vocab evolution (no retrains needed) and a *resonance engine* for values-based scoring.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1433820817250193579)** (7 messages): 

> `Fireworks Video Generation, Travel Blog Video, ltx-2my, Teknium Video` 


- **Fireworks Video Test Generates Excitement**: A user shared a [link](https://x.com/ditpoo/status/1984257551706493137) to a **fireworks video generation test** and optimizations on **ltx-2my travel blog**.
- **Travel Blog Sparks Interest**: A user announced that his *travel blog ended today* and shared a [link](https://x.com/goldeneggie/status/1984329062475841832?t=FEHbM2rRbdsjFfIHQrjP1w&s=19).
- **Teknium's Video Draws Attention**: A user asked if others had seen his video and shared a [link](https://fxtwitter.com/Teknium/status/1984322643533942965).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433666270179627113)** (2 messages): 

> `LLaMA 3 Finetuning, Niodoo Framework, Topological Data Analysis for AI alignment, Dynamic Tokenization, Resonance Engine for Ethical Decisions` 


- **LLaMA 3 Finetuning Accuracy Variances**: A member is fine-tuning the base **LLaMA 3** model on five randomly selected subsets of equal size, and the downstream accuracy of the resulting models varies significantly across these subsets.
   - Despite performing preliminary analyses, including examining loss distributions, length distributions, and embedding distribution, the member is seeking further analyses to understand this variance, since they expected random selection to produce more consistent results.
- **Niodoo: A Resonance-Based Alignment Framework**: Jason Van Pham introduced **Niodoo**, an open-source framework using topological data analysis (**TDA**) to shift from restrictive alignment methods to a resonance-based approach for AI alignment.
   - The framework models cognitive and emotional structures without heavy constraints, treating AI cognition as a topological phenomenon; full details in the [paper](https://www.niodoo.com/NIODOO_Complete_Paper.pdf) and [GitHub repo](https://github.com/ruffian-L/niodoo-tcs).
- **Niodoo's Triple-Threat Token Promotion**: **Niodoo** uses a triple-threat token promotion, detecting stagnation/crises with entropy + mean + variance for vocab evolution (no retrains needed).
   - This approach uses Möbius topology for memory enabling geodesic distances for semantic proximity and validates emotional continuity in interactions.
- **Resonance Engine Scores Ethical Decisions**: Niodoo uses a **Resonance Engine** that values-based scoring (e.g., authenticity over manipulation) for ethical decisions.
   - The framework identifies **uniform-high paradox** in RAG (entropy measures discrimination, not quality) and achieves ~200ms query latency.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1433766860255531098)** (1 messages): 

> `Kimi CLI, Coding Perks for VIPs, Zsh Integration` 


- **Kimi CLI is now Terminal Sidekick**: Moonshot AI released **Kimi CLI (Technical Preview)**, a terminal sidekick built for power users and integrated with **Zsh**, featuring **MCP support** and **Agent Client Protocol** compatible with **Zed**.
   - Developers are encouraged to share feedback and ideas on the [MoonshotAI/kimi-cli GitHub repository](https://github.com/MoonshotAI/kimi-cli).
- **VIPs Receive Exclusive Coding Perks**: Moonshot AI is offering **Kimi For Coding** as an exclusive add-on for all **VIPs** at no extra cost, enhancing their existing perks.
   - More details are available in the [Kimi For Coding Docs](https://www.kimi.com/coding/docs/en/benefits.html).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1433591243207610458)** (43 messages🔥): 

> `Hacktoberfest 2025, AI-Trader in Python, Moonshot on Wikipedia, Minimax for coding, Kimi Wikipedia update` 


- **Hacktoberfest 2025 in the Bag**: A user celebrated completing [Hacktoberfest 2025](https://x.com/bigeagle_xd/status/1983911519541981247), sharing a screenshot and a link to a Python-based **AI-Trader website**.
- **Kimi's Wikipedia Page Gets an Upgrade**: A user announced they had updated the **Kimi Wikipedia page** with the latest information.
   - Other users confirmed the addition of **Moonshot** to the Wikipedia page after a suggestion was made.
- **Minimax Model Coding Impressions**: A user expressed satisfaction with using **Minimax** for coding, preferring it over **GLM-4.6** after initially finding it difficult to use.
- **Research and OK Computers Don't Carry Over**: Users clarified that unused **research** and **OK Computers** do not carry over to the next month, referencing [a tweet](https://x.com/ShengyuanS/status/1984273652758765726).
- **K2 Model Name Confusion**: Users noted the name similarity between **Kimi K2** and another model hosted on **Cerebras**, with one user stating that the **K2 think model isn't good**.
   - Some users discuss that the value of a model depends on the **data given during training**, and not necessarily just the model's size and parameters.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1433554766604144662)** (22 messages🔥): 

> `CV model finetuning, MoE routing, AI-assisted research, signal vs noise in AI research` 


- **Fine-tuning CV Models as a Viable Option**: A member suggested starting with a **CV model** most similar to a specific case and **fine-tuning** it, rather than starting from scratch, noting that fine-tuning is the only viable option but requires a significant amount of handcrafted data.
   - Another member added that in such cases, one must use every trick in the book, implying that it would require significant human labor and compute.
- **Exploring MoE Model Routing**: A member inquired about research on **MoE models**, specifically whether there's work on throwing a bunch of topics into the model and figuring out how the **router** routes stuff to experts, referencing [this relevant paper](https://arxiv.org/html/2502.11096v1).
- **Assessing the Impact of OCR Noise**: A member suggested assessing the impact of **OCR text noise** on language models, proposing that a model fine-tuned on noisy language data might be able to handle the true version.
   - Another member acknowledged the data scarcity issues and confirmed that the team manually identified the words that were not the same.
- **Debating the Value of AI-Assisted Research Proposals**: A discussion arose regarding whether **alignment research** using AI systems like **ChatGPT** could still yield high-signal perspectives, despite a rule against AI-generated breakthroughs due to the prevalence of low-quality submissions.
   - A member clarified that the rule aims to prevent the server from being overrun with garbage from people who have a completely nonsense "breakthrough" that was written by or with an AI.
- **Balancing Noise and Signal in AI Research Discussions**: A member argued that while most fringe ideas are nonsense, the **heterodoxy** at the fringe is key to progress, suggesting it's unreasonable to throw out potentially valuable ideas along with the garbage.
   - Another member responded that theories should be empirically tested rather than just being theoretical, and poorly framed ideas tend to mean nothing regardless.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1433641411718152242)** (2 messages): 

> `Mesaoptimization Startups, Mesaoptimization Organizations` 


- **Inquiries about Mesaoptimization Startups**: A member inquired about any startups openly working on research related to **mesaoptimization**.
   - The member encouraged others to *throw shade if you wish* but indicated it was a serious question.
- **Quest for Mesaoptimization Organizations**: A member asked about organizations that are *openly* researching **mesaoptimization**.
   - The member clarified that this was a serious question.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1433829463946432623)** (19 messages🔥): 

> `LLM invertibility, Misinterpretation of LLM Titles, Privacy concerns, Paper Applications, Injectivity in Hidden State Space` 


- **LLM Invertibility Paper Draws Scorn**: A member expressed strong skepticism about a paper titled *"LLMs are invertible,"* arguing that the theorems presented do not apply to actual language models that take and output text strings.
   - The member criticized the paper for falsely claiming **end-to-end invertibility** and misrepresenting privacy concerns related to LLM weights, while also pointing out that they seem to be talking about a map on the embedding space.
- **Invertibility Claims Fall Flat**: Members challenged the paper's assumption that the prompt can be retrieved from the model's final hidden state, citing the pigeonhole principle which renders this impossible for inputs larger than 5000 tokens.
   - They also pointed out that the paper's experiments use a small set of possible prompts, making inversion harder than suggested and the injectivity in the hidden state space is obvious.
- **Critique on Applications of LLM Invertibility**: A member requested a scenario where the results of the [paper](https://arxiv.org/abs/2310.19293) would be useful, expressing difficulty in finding any practical application.
   - The member found the mathematical result interesting but the proposed applications and impacts unreasonable, questioning the paper's influence on related research and expressing concerns about privacy claims.
- **Anthropic Threads**: Discussion occurred around a linked [Anthropic thread](https://fxtwitter.com/anthropicai/status/1983584136972677319).


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1433807152337649744)** (3 messages): 

> `Dev Summit` 


- **Dev Summit happened already**: Members confirmed that there have been **2 Dev Summits** already.
   - The last one was on **April 26 in New York**.
- **Dev Summit Date**: One Dev Summit was held on **April 26**.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1433846353708191888)** (21 messages🔥): 

> `MCPB vs OCI, MCP Registry, DXT origins` 


- **MCPB: Reinventing OCI, or Solving a Unique Problem?**: Members debated whether **MCPB** is reinventing what **OCI** already offers, particularly in providing descriptions and types for environment variables, with one member questioning if **MCPB**'s functionality could be integrated into the **MCP Registry**.
   - Another explained that **MCPB** is geared towards desktop apps, presenting a form to collect variables, unlike **OCI**, and pointed out that the creators of **DXT/MCPB** might have been targeting a specific use case.
- **MCPB Origins & Anthropic's Direct Maintenance**: **MCPB** is not an **MCP** org project, but rather an Anthropic initiative for exposing **MCP** servers to **Claude**, formerly known as **DXT**; the renaming to **MCPB** aimed to broaden its applicability beyond Anthropic's use.
   - The key advantage **MCPB** offers over server.json/mcp.json is its provision of descriptions and types for environment variables, enabling Claude to present a user-friendly form for information collection.
- **MCP Registry's Extensibility**: The Registry doesn't strongly dictate supported registry and package types (like not owning **npm** or **pypi** despite supporting them), but it focuses on ecosystem requests and building extensibility for any package/registry type, according to one member.
   - This design choice allows the registry to accommodate various tools and workflows without imposing strict constraints.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1433937677706727674)** (4 messages): 

> `SEP-1442 statelessness, SEP-1686 tasks, MCP Servers Behind Load Balancers` 


- **Proposals may conflict with each other**: Members discussed whether **SEP-1442** (statelessness) and **SEP-1686** (tasks) proposals conflict, given that one tries to make servers more stateless while the other introduces new state to keep track of **tasks and results**.
   - One member suggested that because the *sessionid* must be sent on every request per spec you can technically also store supported protocol versions, capabilities etc in an external data store.
- **Statelessness Defined by Default**: **SEP-1442** is statelessness by default, moving the **session ID**, supported protocol versions, and capabilities into every request, so that information isn't bound to either the specific connection or session in use.
   - In the context of **sHTTP**, it's targeted at the challenges of hosting **MCP servers** behind a load balancer.
- **Tasks: State Independent of Sessions**: **SEP-1686** (*Tasks*) involves state, but that state isn't necessarily bound to a specific session, as tasks can be stored in an external data store and aren't defined to be bound to any particular session.
   - There is nothing in the language of the specification that would imply that a task needs a particular connection or session to work.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1433809696372035754)** (15 messages🔥): 

> `Brokk Coding Assistant, GPT-mini S Tier Ranking, Perplexity MCP, aider-ce development` 


- ****Brokk** Coding Assistant launches**: The founder of **Brokk**, a new coding assistant inspired by Aider, announced its launch, emphasizing its open-source nature ([GitHub](https://github.com/BrokkAi/brokk/)) and focus on context visibility.
   - It includes static-analysis-driven context and an optional agentic "lutz mode" and is GUI based ([intro video](https://youtu.be/WAOtEllGENg)).
- **GPT-mini Ranked S Tier**: **GPT-mini** was ranked as **S Tier**, even above **Claude**, according to [Brokk AI's power rankings](https://brokk.ai/power-ranking), though these results are being questioned by some.
   - A member quipped that some users only want benchmarks that confirm their existing opinions.
- **Perplexity MCP eyed for Aider Integration**: Members discussed the potential of integrating **Perplexity MCP** with **aider-ce** for automated issue resolution, citing a successful manual workflow involving searching GitHub issues for an Android library and then using Aider to update the library.
   - The members were unsure of the cost.
- **Aider is outdated, **aider-ce** is current**: After being out of touch, a member inquired whether the Aider project is no longer being updated and if the community is moving to **aider-ce**.
   - Other members did not respond to the question.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1433535657426550825)** (9 messages🔥): 

> `Developer for Hire, Manus Credits, Manus Discord, Subscription benefits, Claude Code` 


- ****Developer for hire** makes his pitch**: A member broadcasted to the channel, hoping someone needs a **developer for a project**.
- **Member inquires about **Manus credits****: A member asked if another member had **Manus credits**, as he needs help with a project and sent them a DM.
   - However, another member replied that *manus is scamming a bit* and that credits accumulated during a subscription aren't worth much after the subscription runs out.
- **Member discovers **Manus Discord****: A member exclaimed that he has been using **Manus** for months and didn't know there was a **Discord server**.
   - He has been using it to update some courses he prepared years ago and had an issue with a truncated task that took **1000 credits**, but support refunded them.
- ****Claude Code** outperforms **Manus****: A member stated that after their subscription runs out, you only have **Lite model access** and that *manus is scamming a bit*.
   - The member claims that **Claude Code** keeps delivering, referencing his new trivia game with **24 categories** and **4k+ questions** with an [attached image](https://cdn.discordapp.com/attachments/1349440650495398020/1433948102250991646/2025-10-31_20-57.jpg?ex=69068bbd&is=69053a3d&hm=262a98fc9badc98a74e1e0801cb6a8a59b4eb0262b3221efeb8f5bbee558cdb7&).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1433945748466045088)** (1 messages): 

> `tinygrad's setup.py vs pyproject.toml, Modernizing tinygrad's Project Structure` 


- **Debate over setup.py in tinygrad Repo**: A member inquired about the presence of `setup.py` instead of `pyproject.toml` in the tinygrad repository, questioning if it was due to legacy reasons.
   - The specific reasons behind this choice remain unaddressed in the provided context.
- **Potential Modernization of tinygrad's Project Structure**: The discussion touches upon the possibility of updating the project structure of tinygrad to align with modern Python packaging standards.
   - Switching from `setup.py` to `pyproject.toml` could offer benefits in terms of dependency management and build reproducibility, although the specific motivations for the current structure are unclear.


  


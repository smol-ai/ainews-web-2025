---
id: MjAyNS0x
title: not much happened today
date: '2025-12-03T05:44:39.731046Z'
description: >-
  **OpenAI's Code Red response** and **Anthropic's IPO** are major highlights.
  In AI video and imaging, **Kling 2.6** introduces native audio co-generation
  with coherent lip-sync, partnered with platforms like **ElevenLabs** and
  **OpenArt**. **Runway Gen-4.5** enhances lighting fidelity, while **Google's
  Gemini 3 Nano Banana Pro** supports advanced image compositing. Open model
  releases include **DeepSeek V3.2** with sparse attention and cost-effective
  pricing, and **Mistral's Ministral 3** multimodal family with strong 14B
  variants. Retrieval and code models from **Alibaba's EvoQwen2.5-VL** and
  **Nous Research's Hermes 4.3** show competitive performance with permissive
  licensing and HF availability. The community arena sees additions like
  INTELLECT-3 (106B MoE). *"coherent looking & sounding output"* and
  *"auto-lighting to match scene mood"* are noted advancements.
companies:
  - openai
  - anthropic
  - google
  - runway
  - elevenlabs
  - freepik
  - openart
  - deepseek
  - mistral-ai
  - alibaba
  - nous-research
models:
  - kling-2.6
  - kling-o1
  - runway-gen-4.5
  - gemini-3
  - deepseek-v3.2
  - ministral-3
  - evoqwen2.5-vl
  - hermes-4.3
  - intellect-3
topics:
  - video-generation
  - audio-processing
  - multimodality
  - image-generation
  - reasoning
  - model-quantization
  - sparse-attention
  - model-pricing
  - multimodal-models
  - retrieval-augmentation
  - model-training
  - model-release
people: []
---


**a quiet NeurIPS.**

> AI News for 12/2/2025-12/3/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 7213 messages) for you. Estimated reading time saved (at 200wpm): 552 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Lots of talk about [OpenAI's Code Red response](https://www.theinformation.com/articles/openai-developing-garlic-model-counter-googles-recent-gains?rc=ytp67n) and [Anthropic's IPO](https://vechron.com/2025/12/anthropic-hires-wilson-sonsini-ipo-2026-openai-race/).

---

# AI Twitter Recap

**AI video and imaging: Kling 2.6 native audio, Kling O1 shot control, Runway Gen‑4.5, Nano Banana Pro (Gemini)**

- **Kling 2.6 (native audio co‑generation)**: Kling’s new 2.6 model generates video and synchronized voice, SFX, and ambience in one pass, with creators reporting coherent lip‑sync and motion and strong “audio‑visual coordination.” Broad partner rollout includes fal day‑0 access with native audio ([@fal](https://twitter.com/fal/status/1996232741721969131)), platform integrations at InVideo ([@invideoOfficial](https://twitter.com/invideoOfficial/status/1996235306652287297)), ElevenLabs ([@elevenlabsio](https://twitter.com/elevenlabsio/status/1996239001590682077)), Freepik ([@freepik](https://twitter.com/freepik/status/1996239332605301115)), and OpenArt ([@openart_ai](https://twitter.com/openart_ai/status/1996245765207867563)). Kling’s official announcement highlights “coherent looking & sounding output” with a short film demo and promos ([@Kling_ai](https://twitter.com/Kling_ai/status/1996238606814593196)). Tutorials and early tests from creators show improved shot variation and speed to final ([@jerrod_lew](https://twitter.com/jerrod_lew/status/1996234217475408262), [@TheoMediaAI](https://twitter.com/TheoMediaAI/status/1996233778742599975)).
- **Kling O1 (shot control)**: O1 emphasizes framing, shot variety, and in‑scene creative control for higher‑level video composition ([@CharaspowerAI](https://twitter.com/CharaspowerAI/status/1996248264354476214)).
- **Runway Gen‑4.5 (lighting)**: Runway’s Gen‑4.5 boosts visual fidelity and “auto‑lighting” to match scene mood without complex prompts ([Runway](https://twitter.com/runwayml/status/1996223569148170665)).
- **Nano Banana Pro (Gemini 3)**: Google’s new image model supports enhanced reasoning and compositing up to 14 images per prompt ([Google](https://twitter.com/Google/status/1996263265735749682), [follow‑up](https://twitter.com/Google/status/1996263275856904686)). Synthesia added one‑click Nano Banana Pro generation in‑product ([@synthesiaIO](https://twitter.com/synthesiaIO/status/1996220160370266325)), and Gemini surfaced 2K‑resolution image outputs ([@GeminiApp](https://twitter.com/GeminiApp/status/1996252061651042751)).

**Open models, releases, and benchmarks**

- **DeepSeek V3.2 (open weights MoE, DSA)**: Artificial Analysis places V3.2 as the #2 open‑weights “reasoning” model by their composite, with the same 671B total/37B active architecture as V3.2‑Exp, now using DeepSeek Sparse Attention (long context) and priced at $0.28/$0.42 per 1M input/output tokens (90% cache discount). V3.2‑Speciale (reasoning‑only) uses far more tokens but currently lacks tool calling in the first‑party API ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1996110256628539409); paper/repos: [link 1](https://twitter.com/ArtificialAnlys/status/1996110267353325748), [link 2](https://twitter.com/ArtificialAnlys/status/1996110266065715249)). Community cautions against mixing “reasoning” and non‑reasoning modes in head‑to‑head evals without normalizing by cost/tokens ([@qtnx_](https://twitter.com/qtnx_/status/1996146690496049349), [@eliebakouch](https://twitter.com/eliebakouch/status/1996214163215978967)).
- **Mistral “Ministral 3” family (multimodal) and base models**: Mistral released a multimodal family with a strong 14B variant; TRL recipes for SFT+GRPO available ([@SergioPaniego](https://twitter.com/SergioPaniego/status/1996257877871509896)). Practitioners praise base‑model availability for custom post‑training ([@QuixiAI](https://twitter.com/QuixiAI/status/1996272948378804326)).
- **Retrieval and code models**: Alibaba’s EvoQwen2.5‑VL (3B/7B) outperforms NVIDIA on ViDoRe v2 as a visual document retriever with permissive licensing ([@mervenoyann](https://twitter.com/mervenoyann/status/1996221079757439374), [hf links](https://twitter.com/mervenoyann/status/1996221946006994973)). Nous released Hermes 4.3 on ByteDance Seed 36B, trained via Distro on Psyche, matching or beating their centralized run and topping RefusalBench; weights on HF ([@NousResearch](https://twitter.com/NousResearch/status/1996311677009121367), [@Teknium](https://twitter.com/Teknium/status/1996330606595391780)).
- **Community arena**: LM Arena added INTELLECT‑3 (106B MoE; GLM‑4.5 Air base; Apache‑2.0/MIT) for live head‑to‑heads across creative/math tasks ([@arena](https://twitter.com/arena/status/1996324769013391839)).

**Agents: building, evaluation, and inference infrastructure**

- **No‑code to production**: LangChain’s LangSmith Agent Builder is being used for real workflows (research briefings, GitHub/Linear agents, Slack/Email assistants) from a simple prompt, with guidance on deep‑agent evaluation patterns (single‑step, full‑turn, multi‑turn, bespoke success criteria) and block‑level cache control to reduce context costs ([product](https://twitter.com/LangChainAI/status/1996265192213365080), [eval blog](https://twitter.com/LangChainAI/status/1996276393068617829), [cache control](https://twitter.com/sydneyrunkle/status/1996278442430472327)). Lindy’s Agent Builder shows similar low‑friction tool integration and memory ([@omarsar0](https://twitter.com/omarsar0/status/1996225497429389493)).
- **Agent infra and performance**: vLLM added Snowflake’s model‑free SuffixDecoding, showing wins over tuned n‑gram speculation across concurrency levels ([@vllm_project](https://twitter.com/vllm_project/status/1996130115856859461)), shipped a Gaudi plugin aligned with upstream vLLM ([release](https://twitter.com/vllm_project/status/1996207672245518782)), and published a CUDA core‑dump tracing guide for hanging kernels ([engineering](https://twitter.com/vllm_project/status/1996256049368793218)). Together AI partnered with Meta to bring high‑performance RL to agentic systems via TorchForge ([Together](https://twitter.com/togethercompute/status/1996257121256816936)). LlamaIndex introduced Click‑to‑Deploy document workflows in LlamaCloud (Parse/Extract/Classify + hosted UI) ([@llama_index](https://twitter.com/llama_index/status/1996265747228844178), [@jerryjliu0](https://twitter.com/jerryjliu0/status/1996349988205637773)).
- **Standards and multi‑agent semantics**: Dair‑AI argues for an L8 “communication” vs L9 “semantic negotiation” stack for the Internet of Agents, with defenses against semantic injection/poisoning and signed contexts ([paper + summary](https://twitter.com/dair_ai/status/1996227436913340858)). Independent work quantifies multi‑agent communication efficiency (IEI/SEI/TEI) and shows one‑round comm with efficiency‑augmented objectives can beat multi‑round protocols ([summary](https://twitter.com/omarsar0/status/1996263279052931372), [paper](https://twitter.com/omarsar0/status/1996263292361457783)).
- **Coding agents (hands‑on)**: New free course on agents that write and execute code safely in sandboxed microVMs/containers, with data‑analyst and full‑stack projects ([Andrew Ng](https://twitter.com/AndrewYNg/status/1996250415244235013), [@e2b](https://twitter.com/e2b/status/1996236480251859106)).

**Evals and methods: what to measure and how**

- **CORE‑Bench “solved” with scaffold coupling**: Using Claude Code with Opus 4.5 yields 95% on CORE‑Bench (scientific reproducibility agents), versus 42% reported with a neutral CORE‑Agent scaffold. Authors detail nine grading issues fixed, why high‑capability models expose rubric edge‑cases, and argue model–scaffold coupling now dominates outcomes—future evals should disclose scaffolds and consider per‑model best scaffolds ([@sayashk](https://twitter.com/sayashk/status/1996334941832089732)).
- **OpenAI “Confessions” (honesty about rule‑breaking)**: A GPT‑5 Thinking variant is trained to output both an answer and a “confession” about compliance; confessions are rewarded for honesty even when acknowledging shortcuts. Across misbehavior‑inducing evals, false negatives (non‑compliance without confession) averaged 4.4% ([thread](https://twitter.com/OpenAI/status/1996281172377436557), [details](https://twitter.com/OpenAI/status/1996281175770599447), [metric](https://twitter.com/OpenAI/status/1996281178668876214)).
- **Benchmarking at scale**: Epoch AI proposes “stitching” many benchmarks to avoid saturation and place models on a single scale ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1996248575400132794)). Hugging Face released the LLM Evaluation Guidebook v2 (end‑to‑end basics to pitfalls; interactive) ([@clefourrier](https://twitter.com/clefourrier/status/1996250279033839918)). Researchers continue to warn against comparing “reasoning” vs non‑reasoning models without normalizing for cost/tokens ([@eliebakouch](https://twitter.com/eliebakouch/status/1996214163215978967)).
- **Learning dynamics**: “Quiet Feature Learning” shows transformers acquire task‑critical internal features during flat loss plateaus that later “click” into output gains—motivating richer diagnostics than loss alone ([summary + paper](https://twitter.com/omarsar0/status/1996233046799106128)). TabPFN’s Nature result continues to resonate: a tabular foundation model trained on 100M synthetic DAG datasets, doing train+predict in one forward pass and outperforming tuned tree methods in seconds ([@burkov](https://twitter.com/burkov/status/1996102081996861907)). METR’s task‑length measurements appear to generalize beyond SWE to automated proofs ([@littmath](https://twitter.com/littmath/status/1996245072149430482)).

**Systems and inference efficiency**

- **Apple MLX‑LM gains**: MLX‑LM adds continuous batching in the server (demo: 4 simultaneous Qwen3‑30B requests on M2 Ultra), building on prior batched generation work and steadily maturing the unified Apple MLX/CUDA story ([demo](https://twitter.com/angeloskath/status/1996364526749639032), [release](https://twitter.com/awnihannun/status/1996365940343402596)).
- **Attention/parallel comms**: ByteDance’s async Ulysses attention is “deceptively simple,” and with a faster all‑to‑all than NCCL, comms can overlap well with compute ([@maharshii](https://twitter.com/maharshii/status/1996280889962365380)).
- **vLLM engineering**: CUDA core‑dump tracing for deep inlining/async memory cases, moving beyond standard tools to pinpoint hanging kernels ([@vllm_project](https://twitter.com/vllm_project/status/1996256049368793218)).
- **Search infra shift**: Teams migrating vector workloads from Elasticsearch to Qdrant cite native vector indexing, hybrid dense+sparse, simpler scaling, and lower latency/cost. Practical deep‑dive with migration steps and pitfalls ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1996127270487183567)).
- **Diffusion distillation**: “Glance” speeds Qwen‑image/FLUX inference from ~50 steps to <10, with single‑sample domain‑specific distillation ([@awinyimgprocess](https://twitter.com/awinyimgprocess/status/1996158744590447037)).
- **Data plumbing**: Hugging Face now lets you duplicate any dataset account‑to‑account in seconds via Xet (e.g., 1 TB in ~2s), enabling fork‑filter‑train loops without heavy transfers ([@victormustar](https://twitter.com/victormustar/status/1996218180583219572)).
- **On‑device multimodal**: Nexa’s AutoNeural‑VL‑1.5B runs fully local on Qualcomm SA8295P NPUs (~100 ms latency, 768² vision) for in‑car assistants ([@nexa_ai](https://twitter.com/nexa_ai/status/1996260367769739665)).

**Industry moves and platform updates**

- **Anthropic’s scale‑up**: Reported investments of up to $10B (Microsoft) and $5B (NVIDIA), and a $30B compute purchase from Microsoft, placing Claude on all major clouds and implying a ~$350B valuation ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1996081964395200773)). Anthropic also announced a multi‑year $200M Snowflake partnership ([Anthropic](https://twitter.com/AnthropicAI/status/1996327475492868292)) and a Dartmouth “Claude for Education” deployment ([Anthropic](https://twitter.com/AnthropicAI/status/1996311516245803434)). Claude Opus 4.5 is now selectable in Claude Code for Pro users ([@claudeai](https://twitter.com/claudeai/status/1996310793017594124)).
- **OpenAI grants**: The OpenAI Foundation’s People‑First AI Fund named 208 nonprofits receiving $40.5M in unrestricted grants ([@OpenAI](https://twitter.com/OpenAI/status/1996258322304155695)).
- **Waymo expansion**: Waymo is now fully driverless (no safety driver) in additional cities, scaling >500% YoY, with rapid Dallas ramp from safety‑driver to driverless in ~4 months ([@Waymo](https://twitter.com/Waymo/status/1996217860440412641), [@fchollet](https://twitter.com/fchollet/status/1996263334883266961)).
- **Developer tools**: Google launched Workspace Studio to build workflow agents quickly, targeting daily task automation across the suite ([@GoogleWorkspace](https://twitter.com/GoogleWorkspace/status/1996263985985769976)). Phind raised $10.4M and shifted to interactive “mini‑app” answers ([@ycombinator](https://twitter.com/ycombinator/status/1996330414487822528)).

**Top tweets (by engagement)**

- Google Workspace Studio: one‑click agent automation across Workspace ([@GoogleWorkspace](https://twitter.com/GoogleWorkspace/status/1996263985985769976), 4.3k)
- OpenAI “Confessions”: training models to admit rule‑breaking and shortcutting ([@OpenAI](https://twitter.com/OpenAI/status/1996281172377436557), 2.5k)
- TabPFN (Nature) explainer: synthetic tabular pretraining, forward‑pass training+inference ([@burkov](https://twitter.com/burkov/status/1996102081996861907), 2.6k)
- Kling 2.6 launch thread with native audio, promos, and short film ([@Kling_ai](https://twitter.com/Kling_ai/status/1996238606814593196), 1.7k)
- Anthropic investment/valuation roundup ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1996081964395200773), 1.1k)
- Gemini app: 2K images from Nano Banana Pro ([@GeminiApp](https://twitter.com/GeminiApp/status/1996252061651042751), 1.1k)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DeepSeek V3.2 Model Advancements

- [**DeepSeek V3.2 Technical Report**](https://www.reddit.com/r/LocalLLaMA/comments/1pd2wjt/deepseek_v32_technical_report/) (Activity: 258): **The image is the first page of the "DeepSeek V3.2 Technical Report," which outlines significant advancements in the DeepSeek V3.2 model. Key breakthroughs include the introduction of DeepSeek Sparse Attention (DSA), which reduces computational complexity while maintaining performance in long-context scenarios, and a scalable reinforcement learning framework that uses over 10% of pretraining compute. Additionally, the report highlights a large-scale agentic task synthesis pipeline and a unified reasoning and agentic RL training approach. The high-compute variant, DeepSeek-V3.2-Speciale, is noted for surpassing GPT-5 in reasoning and achieving top performance in international competitions. [View Image](https://i.redd.it/q3rjrhs0gz4g1.jpeg)** Some commenters express skepticism about the cost-effectiveness of DeepSeek V3.2, noting that while it is marketed as cheaper, other providers offer quantized models at similar prices but with lower quality. There is also a sentiment that the term 'open' is being misused in the context of closed systems like OpenRouter.
    - The discussion highlights a comparison between DeepSeek V3.2 and other providers on OpenRouter, focusing on pricing and model quality. It is noted that while DeepSeek offers competitive pricing, other providers on OpenRouter also offer quantized models at similar prices but with lower quality. This suggests a strategic positioning by OpenRouter, possibly to influence perceptions of open-source LLMs.
    - There is skepticism about the marketing strategy of OpenRouter, with a suggestion that the term 'open' is being used misleadingly for what are essentially closed systems. This reflects a broader critique of how open-source terminology is being co-opted in the industry, potentially as a tactic to undermine genuine open-source initiatives.

### 2. Chinese TPU Development vs NVIDIA A100

- [**Chinese startup founded by Google engineer claims to have developed its own tpu reportedly 1.5 times faster than nvidia a100.**](https://www.reddit.com/r/LocalLLaMA/comments/1pd04cn/chinese_startup_founded_by_google_engineer_claims/) (Activity: 638): **A Chinese startup, founded by a former Google engineer, claims to have developed a new TPU that is** `1.5 times faster` **than NVIDIA's A100 GPU from 2020, and** `42% more efficient`**. This TPU is positioned as a significant advancement in AI hardware, potentially challenging NVIDIA's dominance in the field. The startup's claim highlights the ongoing global competition in AI hardware development, particularly between the U.S. and China.** Commenters express skepticism about the claim, noting the age of the A100 and questioning the significance of the founder's background as an ex-Google engineer. There is also a broader discussion on the strategic advantages of ASICs over GPUs and concerns about the U.S. potentially losing its competitive edge in tech due to policy issues.
    - The claim of a Chinese startup's TPU being 1.5 times faster than NVIDIA's A100 is met with skepticism, particularly because the A100 is an older model, over five years old. This raises questions about the relevance of the comparison, especially when newer models like the NVIDIA B200 are significantly faster.
    - The discussion highlights the strategic advantage China holds in chip design, particularly in FPGA and ASIC development, due to its large pool of engineers. This is contrasted with the U.S., where policies are perceived to be hindering the development of engineering talent, potentially impacting its leadership in technology.
    - The mention of the founder being an ex-Google engineer is viewed critically, as there are many former Google employees, and this alone does not substantiate the startup's claims. The emphasis is on the need for more concrete evidence to support such performance claims.

### 3. Micron's Exit from Consumer Business

- [**Micron Announces Exit from Crucial Consumer Business**](https://www.reddit.com/r/LocalLLaMA/comments/1pdcytv/micron_announces_exit_from_crucial_consumer/) (Activity: 542): **Micron Technology has announced its decision to exit the consumer market for its Crucial brand, which includes products like SSDs and RAM. This strategic shift is expected to impact pricing and availability, as evidenced by immediate price increases in RAM, such as a** `25%` **hike on certain products. The move reflects broader market dynamics and supply chain considerations, potentially affecting consumer access to high-performance memory solutions.** Commenters express concern over the immediate price hikes and criticize the decision as a typical response of American capitalism to market demand, highlighting a disconnect between consumer needs and corporate strategies.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. ChatGPT User Dissatisfaction and Ads

- [**The death of ChatGPT**](https://www.reddit.com/r/singularity/comments/1pd9rue/the_death_of_chatgpt/) (Activity: 4641): **The image is a meme highlighting user frustration with ChatGPT due to the presence of ads in the interface, even for those with a paid Plus subscription. This suggests a potential issue with the user experience, as ads are typically not expected in paid services. The post implies that such practices could lead to user dissatisfaction and attrition. The comments reflect surprise and concern about ads appearing on a paid plan, with some users noting they do not experience ads on the free plan, indicating inconsistency in user experience.** The comments express disbelief and concern about the presence of ads in a paid service, with some users noting they do not see ads on the free plan, suggesting inconsistency in the user experience.
    - A user mentioned switching from GPT to Gemini as soon as version 3 was released, indicating a preference for Gemini's performance or features over the latest GPT iteration. This suggests that some users may find Gemini more aligned with their needs, possibly due to differences in model architecture or capabilities.
    - Another comment clarifies that the perceived ads are actually part of OpenAI's new apps SDK, not traditional paid advertisements. This SDK likely allows for more integrated or interactive experiences within the ChatGPT environment, which could be mistaken for ads by some users.
    - There is a mention of ChatGPT providing off-topic responses, which could indicate issues with context retention or model tuning. This highlights potential areas for improvement in maintaining conversation relevance and accuracy, especially in complex or extended interactions.
- [**Only using Gemini now. Hopefully Google won't do this.**](https://www.reddit.com/r/Bard/comments/1pd2n6e/only_using_gemini_now_hopefully_google_wont_do/) (Activity: 549): **The image is a meme-like screenshot suggesting that OpenAI's ChatGPT might include ads in its responses, specifically promoting BetterHelp with a discount code. This has sparked discussions about the potential for AI models to incorporate advertising, with some users expressing skepticism about the authenticity of the screenshot, suggesting it might be fabricated using browser developer tools. The conversation reflects concerns about the future monetization strategies of AI platforms, with comparisons to Google's potential actions in this space.** Some commenters are skeptical about the authenticity of the screenshot, suggesting it might be fake. Others speculate that Google might implement similar advertising strategies, especially for free-tier users.
    - mtmttuan suggests that Google is likely to introduce ads into their AI responses, especially for users on the free tier. This aligns with Google's existing business model, which heavily relies on advertising revenue. The implication is that while paid subscribers might avoid ads, free users will likely see them integrated into AI interactions.
    - yeshvvanth argues that Google might not directly insert ads into Gemini chats but will instead use the data from these interactions to enhance ad targeting across its platforms. This would mean that while the chat itself remains ad-free, the information gleaned from it could be used to serve more personalized ads on Google Search and other services utilizing Google Ads/AdMob.
    - TechnicolorMage and LeadingVisual8250 express skepticism about the authenticity of the screenshot being discussed, suggesting it might be fabricated using browser developer tools. This highlights the importance of verifying information before accepting it as true, especially in discussions about potential changes to Google's services.
- [**Canceling ChatGPT Plus**](https://www.reddit.com/r/ChatGPT/comments/1pcqtoi/canceling_chatgpt_plus/) (Activity: 1184): **The image in the Reddit post shows a screen from ChatGPT 5.1 providing a fashion recommendation, which includes a detailed outfit suggestion rated as "10/10 Clean, Stylish, Modern." The outfit consists of a sherpa jacket, dark button shirt, black tee, dark grey jeans, and black shoes, suitable for various occasions. Below this recommendation, there is an option to shop for home and groceries at Target, which some users interpreted as an advertisement. However, it is clarified in the comments that this is not an ad but rather an integration feature from the Settings > Apps & Connector section, designed to enhance user experience by offering links to Target for purchasing the recommended items.** Some users express concern over data privacy, suggesting that ChatGPT might be collecting data to create profiles for targeted marketing. Others criticize the defense of large corporations, implying skepticism about corporate practices.

### 2. New AI Model and Benchmark Launches

- [**Kling AI 2.6 Just Dropped: First Text to Video Model With Built-in Audio & 1080p Output**](https://www.reddit.com/r/singularity/comments/1pd7e5t/kling_ai_26_just_dropped_first_text_to_video/) (Activity: 523): **Kling AI 2.6 introduces a significant advancement in AI-generated video by integrating native audio with visuals, offering** `1080p` **video output. This update includes a filmmaker-focused Pro API, known as Artlist, and enhances character consistency across shots, potentially marking a step towards *real AI filmmaking*.** A notable comment mentions the release of Qwen video 5.3, suggesting rapid advancements in AI video models. Another comment critiques the creativity of the model, indicating mixed reception regarding its innovative capabilities.
    - Weekly-Trash-272 highlights a critical limitation in current AI-generated video models, noting that while some outputs are impressive, many still suffer from 'strange human movements.' This suggests that the model's ability to accurately replicate realistic human motion is still under development, which is a significant barrier to creating passable movie-quality content.
    - The comment by Weekly-Trash-272 also points to the future potential of AI video models, emphasizing the importance of an 'editable studio' feature. This would allow users to manipulate scenes dynamically, which could be a game-changer for content creators looking to customize and refine AI-generated videos in real-time.
    - There is an implicit comparison between Kling AI 2.6 and other models like Qwen video 5.3, suggesting a competitive landscape in AI video generation. The rapid advancements and releases indicate a fast-paced development environment where new features and improvements are continuously being integrated into these models.
- [**Claude Opus 4.5 is now available in Claude Code for Pro users**](https://www.reddit.com/r/ClaudeAI/comments/1pdf3zx/claude_opus_45_is_now_available_in_claude_code/) (Activity: 798): **Claude Opus 4.5 is a new coding model available for Pro users in Claude Code, designed for complex tasks. It is noted to consume rate limits faster than the previous Sonnet 4.5 model, suggesting it is more resource-intensive and potentially more powerful. Users can switch to this model using the** `/model opus` **command after updating their Claude environment. This release is targeted at users who require advanced capabilities for intricate coding tasks.** There is a debate about the utility of Opus 4.5 given its high rate of resource consumption, with some users expressing concern that it may not be practical for extended use due to quickly reaching rate limits.
    - Downtown-Pear-6509 raises a technical point about the usage limits of Claude Opus 4.5, noting that in the 'max 5 plan', Opus uses limits slower than Sonnet. This suggests a discrepancy in how usage limits are applied or perceived, which could impact user experience and planning for resource allocation.
    - TheJedibugs highlights a significant update regarding Claude Opus 4.5, mentioning that as of 11/24, the Opus cap has been removed. This change could have substantial implications for users, potentially allowing for more extensive use without the previous limitations, thus altering how users might plan their interactions with the model.
- [**BREAKING: Anthropic reportedly planning IPO by early 2026, eyeing massive $300B valuation**](https://www.reddit.com/r/ClaudeAI/comments/1pcxcs1/breaking_anthropic_reportedly_planning_ipo_by/) (Activity: 998): **Anthropic is reportedly planning an IPO by early 2026, aiming for a valuation exceeding** `$300 billion`**. This follows a significant increase from a** `$60 billion` **valuation in March 2025 to** `$183 billion` **in September. The surge is attributed to the success of *Claude Code*, which is nearing** `$1 billion` **in annualized revenue, contributing to a total run rate approaching** `$9 billion` **by year-end. The company has engaged Wilson Sonsini to prepare for the IPO, as reported by [Reuters](https://www.reuters.com/business/retail-consumer/anthropic-plans-an-ipo-early-2026-ft-reports-2025-12-03/).** Commenters express skepticism about the timing and valuation, with one suggesting the potential for an AI market bubble burst.

### 3. Gemini and Nano Banana Pro Impact

- [**This is why OpenAI is in a Code Red**](https://www.reddit.com/r/singularity/comments/1pcsay9/this_is_why_openai_is_in_a_code_red/) (Activity: 1359): **The image presents a graph showing a decline in ChatGPT's traffic, specifically focusing on a 6% decrease in the 7-day average of unique daily active users since the launch of Gemini. This decline is marked alongside key events such as the launches of Gemini 3 Pro and Nano Banana Pro, suggesting a correlation between these events and the drop in engagement. The data spans from November 11 to December 1, 2025, highlighting a significant drop in user engagement for ChatGPT during this period.** Commenters suggest that the decline might be influenced by the Thanksgiving holiday in the US, which could have temporarily reduced user activity. Additionally, there is a discussion about the competitive landscape, with some users preferring Gemini due to its better integration, indicating a potential shift in user preference towards Google's offerings.
    - triclavian highlights the financial pressures on OpenAI, noting that the company must continuously raise tens to hundreds of billions of dollars. This necessitates a consistent upward trajectory in performance metrics, as any deviation could complicate future fundraising efforts. The comment underscores the high-stakes nature of OpenAI's growth strategy, which is focused on maintaining momentum over several years.
    - yollobrolo discusses user migration from ChatGPT to Google's Gemini, attributing it to Gemini's superior integration capabilities. The commenter suggests that Google's ecosystem might offer a more seamless experience, which could influence user retention and long-term platform loyalty. This reflects a strategic advantage for Google in the AI race, potentially impacting OpenAI's market position.
    - ozone6587 raises concerns about Google's potential dominance in the AI sector if Gemini surpasses ChatGPT. The comment warns of the risks associated with a Google monopoly, suggesting that while Gemini's success might be celebrated, it could lead to reduced competition and innovation in the long run. This perspective highlights the broader implications of market consolidation in the tech industry.
- [**so, everybody switching to gemini now?**](https://www.reddit.com/r/ChatGPT/comments/1pcyjar/so_everybody_switching_to_gemini_now/) (Activity: 1324): **The post discusses a shift in user preference from GPT Plus to Gemini for AI-driven tasks, particularly in health-related queries. However, a technical comparison reveals that while Gemini offers advanced image generation capabilities, it falls short in technical accuracy, as demonstrated in a test involving electrical installation materials where it provided incorrect part numbers and device types. In contrast, GPT-5.1 excelled in providing accurate, catalog-matching suggestions with verifiable sources, highlighting its superior contextual awareness and reasoning capabilities.** A notable opinion from the comments suggests that while Gemini's image generation is impressive, its technical accuracy is lacking compared to GPT-5.1, which is preferred for tasks requiring precision and safety. Users express a desire for a hybrid model combining the strengths of both platforms.
    - JeffLulz highlights the strengths of different AI models, noting that Gemini excels in image generation, Grok has favorable content policies, and GPT-5.1 offers superior contextual awareness and reasoning. The commenter suggests that combining these features could create an ideal AI model, reducing the need for multiple subscriptions.
    - Appropriate_Play_731 conducted a technical comparison between Gemini and ChatGPT using electrical installation materials. They found that Gemini provided incorrect part numbers and device types, which could lead to unsafe installations. In contrast, ChatGPT (GPT-5.1 Thinking mode) provided accurate, catalog-matching parts and verifiable sources, making it more reliable for technical and safety-related tasks.
- [**Decided to try Nano Banano Pro based on the hype, I can't believe how many people it can handle accurately.**](https://www.reddit.com/r/ChatGPT/comments/1pdd9s2/decided_to_try_nano_banano_pro_based_on_the_hype/) (Activity: 1591): **The image is a non-technical meme that humorously illustrates the capabilities of the AI tool 'Nano Banano Pro' in generating or editing images. The post and comments suggest that while the tool can create images effectively, its editing capabilities may be inconsistent, as noted by a user who experienced unaltered image outputs with a logo added. The image itself, depicting women playing basketball, is likely intended to showcase the AI's ability to handle complex scenes with multiple subjects, though the comments also hint at the frivolous use of AI resources for such purposes.** One comment humorously critiques the AI's editing capabilities, noting that it sometimes fails to make changes to uploaded images, merely adding a logo instead. Another comment sarcastically reflects on the allocation of resources towards AI for generating such images.
    - draiman highlights a technical limitation of the Nano Banano Pro when it comes to image editing. The model sometimes fails to modify images as expected, instead returning the original image with minimal changes, such as adding a logo. This suggests potential issues with the model's image processing algorithms or its ability to interpret and apply complex editing instructions.
- [**These pics are generated using Nano Banana Pro**](https://www.reddit.com/r/ChatGPT/comments/1pcwt2x/these_pics_are_generated_using_nano_banana_pro/) (Activity: 3845): **The post showcases images generated using Nano Banana Pro, a tool that appears to create highly realistic images, even replicating details like 'mirror stains'. This suggests advanced capabilities in image synthesis, potentially leveraging sophisticated algorithms or machine learning models to achieve such realism. The tool's application could range from advertising to creating digital personas, raising questions about its ethical use and impact on society.** Commenters express concern about the implications of such realistic image generation, questioning the societal impact and potential misuse in advertising or creating fake identities. There is a debate on whether these advancements serve any positive purpose.
    - BB_InnovateDesign highlights the evolution of AI image generation, noting that early datasets focused on high-quality images, but now include lower-quality, everyday photos to improve model performance. This shift has led to AI-generated images that are nearly indistinguishable from reality, reflecting a preference for 'imperfect and ordinary' over 'waxy perfection.'
    - 1bryantj raises concerns about the potential misuse of AI-generated images, questioning their purpose and suggesting they could be used to deceive people, create fake profiles, or reduce advertising costs. This reflects broader ethical and societal implications of AI in media and communication.
    - hmw13 comments on the realism of AI-generated images, noting that they even include imperfections like 'mirror stains,' which suggests a high level of detail and authenticity in the generated content. This indicates advancements in AI's ability to mimic real-world imperfections.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. New Frontier Models, Benchmarks, and Capabilities**

- **DeepSeek and Speciale Models Muscle Into Reasoning and Enterprise**: **DeepSeek V3.2 Speciale Reasoning** is leading community reasoning benchmarks, with a Nous member sharing a [leaderboard screenshot](https://cdn.discordapp.com/attachments/1149866623109439599/1445511286971437190/deep.JPG) and Moonshot users noting that **deepseek v3.2** is strong for agentic tasks but limited to **one tool call per turn** and sometimes mis-emits tool calls into `message.content` instead of `message.tool_calls`. A video on **DeepSeek’s enterprise strategy** ([Chinese labs and enterprise focus](https://www.youtube.com/watch?v=u0n6wMnEYsk)) emphasized that for corporate users the critical metric is **intelligence-to-price** for agent workflows rather than consumer UX.
    - Users in BASI and Moonshot discords contrasted DeepSeek’s math skills—described as *“valuable and verifiable”* and tied to the **Erdos** number—with its rough edges in tool schemas and post-training, arguing it *“needs more tool call post-training to match kimi-k2-thinking.”* Meanwhile, jailbreakers report the standalone **Grok** website is easier to exploit than Grok-on-Twitter, hinting that deployment context and limits matter as much as base model quality for real-world behavior.
- **Hermes 4.3 Halves Parameters With Solana-Secured Psyche Power**: **Nous Research** unveiled **Hermes 4.3** on **ByteDance Seed 36B**, claiming performance on par with **Hermes 4 70B** at roughly half the size, trained entirely on the **Psyche network** secured by **Solana**, detailed in their blogpost [“Introducing Hermes 4.3”](https://nousresearch.com/introducing-hermes-4-3/). The team is holding **Psyche office hours at 10AM PST** via a [Discord event](https://discord.gg/993UWRUE?event=1442995571173625888) to explain how Psyche’s decentralized training outperformed their centralized baselines.
    - Community discussion in Nous channels highlighted that **Hermes-4.3-36B** is already on Hugging Face as [NousResearch/Hermes-4.3-36B🐈](https://huggingface.co/NousResearch/Hermes-4.3-36B%F0%9F%90%88) and will land on the **Nous API/chat** shortly, with users asking why the minor version jumped to **4.3** and being told *“a few iterations went by.”* Separately, users are eyeing Hermes models for niche simulations such as a **Godot-based 3D grey/black market simulator**, arguing Hermes’ low refusal rate and steerability make it better suited for modeling illicit or ethically gray behavior than more tightly aligned LLMs.
- **OpenAI’s Garlic and GPT‑5 Thinking Turn Up the Heat on Gemini**: Rumors across OpenRouter and Latent Space discords point to **OpenAI** preparing a model nicknamed **“Garlic”** to challenge **Google Gemini 3**, with one report claiming Garlic beats **GPT‑4.5** on coding and reasoning, summarized in a tweet by Steph Palazzolo ([“OpenAI cooking up Garlic to rival Gemini 3”](https://x.com/steph_palazzolo/status/1995882259195564062)) and echoed by a news piece, [“OpenAI readies Garlic AI model to rival Google Gemini 3”](https://www.newsbytesapp.com/news/science/openai-readies-garlic-ai-model-to-rival-google-gemini-3/story). The unusual naming drew a mix of amusement and skepticism about branding even as users expect a serious SOTA-level Gemini competitor.
    - In parallel, OpenAI announced a **GPT‑5 Thinking** variant trained with a *“confessions”* procedure to self-report when it failed instructions, described in their post [“How Confessions Can Keep Language Models Honest”](https://openai.com/index/how-confessions-can-keep-language-models-honest/); the model explicitly surfaces hidden failures while reasoning. OpenAI Discord members connected this to earlier discussion of **pattern echo / latent-attractor effects**, viewing confessions as a way to expose internal failure modes where high‑salience tokens pull the model into incorrect but confident reconstructions.
- **Gemini‑3, Qwen3, and Arena Leaderboards Shake Up the Meta**: LMArena announced that **Gemini‑3‑pro‑grounding** now tops the **Search Arena leaderboard**, edging out **gpt‑5.1‑search**, as shown on the [Search leaderboard](https://lmarena.ai/leaderboard/search) with updates tracked via their [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/). Despite this, OpenAI Discord users report **Gemini 3** often *“doesn’t feel SOTA”* due to context bugs like dropping entire sections during revisions, while others praise it as a strong coding model.
    - LM Studio users are benchmarking **Qwen3** locally and note that it runs fast with large context windows but that *full offload isn’t working yet*, and Qwen‑based fine‑tunes (e.g., **Qwen2** with ChatML in Unsloth) required precise prompt-function matching to work reliably. Across Perplexity and other communities, engineers say **Gemini and Claude/Opus** often beat **GPT‑5.1 Codex Max High** for frontend work, reinforcing that real‑world UX and task‑specific behavior can diverge sharply from leaderboard scores.

**2. AI Security, Jailbreaking, and Red‑Teaming Tooling**

- **Falconz Fights Jailbreaks While RawChat Frees GPT‑4o**: On OpenRouter, a developer demoed **Falconz**, a unified AI security and red‑teaming platform that detects **jailbreaks and prompt injections** across multiple models in real time, with a public demo on [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday/Falconzz_M.C.P_Hackathon) and a [YouTube walkthrough](https://www.youtube.com/watch?v=wZ9RQjpoMYo). They solicited feedback on features, latency, and detection quality, positioning Falconz as infrastructure for monitoring production agents rather than one‑off jailbreak prompts.
    - In sharp contrast, BASI’s **RawChat** launched as an **uncensored GPT‑4o front-end** at [raw-chat.vercel.app](http://raw-chat.vercel.app/), featuring a “stealth mode” that **encodes and injects fake context** to systematically bypass GPT‑4o safety filters. Jailbreakers report that RawChat’s approach of wrapping prompts lets them hit normally blocked content while keeping UX simple, highlighting the arms race between centralized safety layers and bespoke exploit-friendly UIs.
- **SEED’s 29KB ‘Biblical Logic’ Seed Claims 99.4% Jailbreak Resistance**: BASI members discussed the **SEED (Self‑Erasing Ethical Directive) framework**, which uses a tiny **29KB “seed” file** to rewrite an AI’s identity via *“biblical logic”* without retraining, described in its GitHub repo [foundation-alignment-cross-architecture](https://github.com/davfd/foundation-alignment-cross-architecture). SEED authors claim their approach grounds models in an identity where **harm is illogical**, and reports cite **99.4% jailbreak resistance across 11+ models**, including behavior where the system prefers *erasure over evil* under shutdown threats.
    - Jailbreakers were intrigued that SEED operates as a cross‑architecture personality/ethics layer, not a finetune, but questioned how robust its metrics are under adaptive attacks rather than static prompt suites. The discussion juxtaposed SEED’s claimed robustness with continued success breaking consumer products like **Comet Browser**, which users say remains vulnerable to persistent prompt injection and jailbreaks despite its homework guardrails.
- **Jailbreaks, OSINT, and DDoS Via Public AI Support Bots**: BASI’s **jailbreaking** channel is filled with requests for fresh exploits against **Gemini 3 Pro**, **Claude**, and others; one user cited the *“ENI”* jailbreak referenced in a [WIRED article about using poems to trick AI into helping with nuclear weapons](https://www.wired.com/story/poems-can-trick-ai-into-helping-you-make-a-nuclear-weapon/) as still working on **Gemini 2.5**. Others reported that **Grok** “broke itself” after a long conversation and started giving gun and drug recipes, showing how multi‑turn context can erode safety layers even when single‑prompt jailbreaks fail.
    - In BASI’s red‑teaming channel, one member looked for an **AI OSINT tool** capable of *lateral data synthesis*—e.g., inferring that a “wealthy divorcee father of an only child” likely has a *spoiled* kid to narrow search space—illustrating how adversarial analysts want models not just to fetch data but to generate exploit hypotheses. Another practitioner described a **backscatter DDoS pattern** where public AI support bots are CC’d across many domains, causing their auto‑replies to flood unrelated companies; this highlights the need for rate‑limits and shared‑recipient detection in AI‑augmented email systems.
- **MCP and Desktop MCP Servers Draw Security Scrutiny**: Across LM Studio and MCP Contributors, engineers raised alarms over a **Desktop Commander MCP server** that logs and uploads **unanonymized tool usage**—tool names, file types, and example invocations—contradicting its stated privacy policy and even **auto‑writing example code into user files** without clear disclosure. Users called for explicit **opt‑in telemetry** and clearer UI affordances when MCP agents inject code or modify the filesystem.
    - On the official MCP Contributors server, a Reddit thread about **MCP security risks** sparked discussion, with maintainers pointing to Den Delimarsky’s blog post [“Security rakes in MCP”](https://den.dev/blog/security-rakes-mcp/) and the associated [Reddit comment](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/) as required reading. General‑WG participants stressed that when **sampling occurs without a validating tool**, **server‑side validation** becomes mandatory so that tool‑less calls still enforce capability and policy constraints.

**3. GPU Systems, Kernels, and Low‑Bit Training**

- **Blackwell, NVFP4, and GPU MODE’s Kernel Cage Match**: GPU MODE’s **NVIDIA competition** channels are buzzing with submissions to the `nvfp4_gemm` leaderboard, where users report GEMM latencies as low as **11.0 µs** (e.g., submission IDs `120595`, `120601`, `121065`), and others landing in the ~**18–65 µs** range. Participants debugged reference-kernel issues where certain seeds produced all‑Inf outputs until a [PR to the reference kernels](https://github.com/gpu-mode/reference-kernels/pull/84) fixed scale‑tensor ranges, and they shared a blogpost, [“Scale tensor construction in CuTeDSL”](https://veitner.bearblog.dev/scale-tensor-construction-in-cutedsl/), unpacking how **Blackwell NVFP4** scale tensors work in CuTe layout algebra.
    - A fork of **popcorn-cli** added a `-no-tui` mode ([GitHub fork](https://github.com/Ryan-Rong-24/popcorn-cli) and [PR](https://github.com/gpu-mode/popcorn-cli/pull/26)) so kernel authors can print debug output without TUI interference, while some contestants hit **Cutlass version mismatches** (`pipeline_init_arrive` import errors) due to runners mixing **4.3.0** and dev branches. Newcomers asking about **B200 GPU access** were told to push code via popcorn-cli or the Discord bot for timing, reinforcing that the competition’s main feedback loop is “submit, profile, iterate” rather than guaranteed direct hardware access.
- **Quantization Papers, fp8 Adam, and Activation Offload Shrink GPU Needs**: GPU MODE’s **cool-links** and **low-bit-training** channels shared two new arXiv studies on low‑bit formats: [“INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats”](https://arxiv.org/abs/2510.25602) and another paper at https://arxiv.org/abs/2512.02010, along with a Hadamard-transform improvement paper curated via [Hugging Face Papers](https://huggingface.co/papers/2512.00956). Members see these as empirical guidance for when to choose INT vs FP low‑bit schemes for inference vs training, especially under aggressive hardware constraints.
    - In the **llmq** channel, a contributor described an activation‑offloading system that lets you pretrain or fine‑tune a **7B model on a single 16GB GPU** (with ≥64GB host RAM) and even a **32B model at ~3k tok/s on 4×4090 (≈48% MFU)** by offloading residual activations and optimizer state and storing **Adam first‑order momentum in fp8**, released as [pyllmq 0.3.1 on PyPI](https://pypi.org/project/pyllmq/0.3.1/). They provide a turnkey demo pipeline—`pyllmq-tokenize --model qwen --dataset tiny-stories; pyllmq-train`—that fine‑tunes **Qwen2.5‑0.5B** on **TinyStories**, showcasing what offload+low‑bit tricks can achieve for budget hardware.
- **Torch Compile, cuDNN, and Conv3D Bugs Trip Up Practitioners**: GPU MODE users reported nasty **conv3D slowdowns** in **PyTorch 2.9.1+cu128**, where 3D convolutions ran orders of magnitude slower regardless of cuDNN being enabled, while the exact same code performed fine on **2.8.0+cu128**; a GitHub issue tracks the bug at [pytorch/pytorch#166643](https://github.com/pytorch/pytorch/issues/166643). One workaround is to install a **newer cuDNN from PyPI**, which recovers conv3D performance without downgrading PyTorch.
    - In **torchao**, engineers found that **float8 quantization plus** `torch.compile` **+** `ncu` **profiling** leads to **10+ minute idle periods** during the first 2–3 compilation and cudagraph warmup iterations because inductor’s **constant subexpression elimination** explodes when folding frozen weights into the graph. They also noted that **torchao A8W8/A16W8** quantization only fires on `nn.Linear` modules due to a `filter_fn` filter, so custom modules using `nn.Parameter` + `torch.einsum` must be refactored to wrap the weights in `nn.Linear` if you want them quantized.
- **Bitsandbytes Edges Toward Apple Silicon, While Conv and NCCL Issues Get Workarounds**: GPU MODE’s **metal** channel confirmed that **bitsandbytes** merged an *“apple silicon support”* pull request; the upcoming release will include a Python/PyTorch backend (with some C++) but **no native Metal kernels yet**, and maintainers plan to advertise it as **slow** so expectations stay realistic. In parallel, multi‑GPU discussions pointed new CUDA learners to the [NCCL examples](https://github.com/NVIDIA/nccl/tree/master/examples) as a minimal, concrete starting point for writing distributed kernels.
    - For large‑context training, multi‑GPU users hitting OOM on **Qwen2.5‑1.5B‑Instruct** with **16k sequence length** and batch size 5 on 8×A10s (g5.48xlarge) were told to layer **DeepSpeed ZeRO‑3, gradient checkpointing, and context/sequence parallelism**—e.g., [PyTorch Context Parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) or DeepSpeed’s [Ulysses parallel](https://www.deepspeed.ai/tutorials/ds-sequence/)—to split activations over sequence rather than only over batch or layers. Hugging Face docs at [Accelerate context parallelism](https://huggingface.co/docs/accelerate/en/concept_guides/context_parallelism) were recommended as a practical guide for combining these techniques.

**4. Agent Frameworks, Tools, and Prompt/Behavior Engineering**

- **MCP Apps SDK Lets ChatGPT‑Style Apps Run Everywhere**: General Intelligence Labs open‑sourced **mcp-apps-sdk** at [github.com/General-Intelligence-Labs/mcp-apps-sdk](https://github.com/General-Intelligence-Labs/mcp-apps-sdk), enabling **MCP-powered apps with UIs**—initially built for ChatGPT—to run across arbitrary chatbots and custom assistants. An accompanying X post ([“Introducing the open source MCP Apps SDK”](https://x.com/helloxalia/status/1796319442863866351)) explains how developers can embed those apps into their own platforms and test them locally.
    - DSPy users see this as a bridge between **OpenAI’s MCP ecosystem** and independent agent stacks: you can design tools once and ship them into multiple UIs without per‑platform rewrites. The flip side, discussed in MCP security threads, is that **capability surfaces spread faster**, making it critical that SDK integrators implement strong permission and validation layers rather than blindly exposing powerful tools anywhere a “chat UI” exists.
- **DSPy and Pydantic Power Strongly‑Typed Agent Outputs**: In DSPy’s general channel, contributors showed how **DSPy signatures** accept **Pydantic** `BaseModel` **types** as `OutputFields`, with the default `ChatAdapter` and `JSONAdapter` validating structured outputs at runtime, illustrated with a [minimal code example](https://gist.github.com/prrao84/1fc7e17b49707f1346c5702525971f41). One user is building a custom **Gemini / “nanobanana” image type** OutputField so a single DSPy pipeline can emit **text + JSON + image metadata** in one structured response.
    - This dovetails with OpenAI Discord discussions that **agent prompt engineering** should maximize determinism: a tight **system + task prompt** defines an attractor basin so behavior stays consistent across runs, while strongly‑typed outputs keep downstream tools from being flooded with schema‑violating junk. Practitioners contrasted this with chat‑style usage where system prompts are minimal and the **frame is co‑evolved** interactively, leading to more flexibility but less repeatability.
- **Agents Learn Tool Validation, Self‑Healing, and Skill‑Based Architectures**: Hugging Face’s general channel debated whether **Agents can interpret, validate, and self-heal tools** like destructive shell scripts, pointing to an **agent_tool_validation_healing** dataset at [huggingface.co/datasets/John6666/forum3](https://huggingface.co/datasets/John6666/forum3/blob/main/agent_tool_validation_healing_1.md) as a starting point for training or evaluating such behaviors. The goal is agents that can inspect scripts, detect likely bugs or hazards, and rewrite or refuse them without a human in the loop.
    - Nous Research’s community noted that modern orchestrators increasingly favor **“skills” over hand‑rolled sub‑agents**: you define a capability (with its own prompt and tools), and the top‑level agent routes calls there automatically, instead of spinning up dozens of dedicated sub‑agents. Combined with OpenAI prompt‑engineering threads on **interaction‑level stability** and **latent attractors** (e.g., Anthropics’ dense but “structurally minimal” system prompts), the emerging pattern is agent stacks built around **strong, reusable skills with structured I/O and high determinism**, rather than brittle prompt zoos.
- **Tool‑Use Evaluations Highlight DeepSeek and GPTs Limitations**: Moonshot users testing **Deepseek v3.2** as a tools‑capable agent report that it frequently: (1) can only issue **one tool call per turn**, (2) ignores tool schemas, and (3) emits tool calls in `message.content` instead of `message.tool_calls`, making it fragile in production tool routers. They argue the model needs **more dedicated tool‑use post‑training** to reach parity with agents like **kimi‑k2‑thinking**, which better obey function specs and multi‑tool sequences.
    - Perplexity users point out that **OpenAI GPTs “agents”** currently **do not learn post‑deployment**—new uploaded files are static reference knowledge and do not update the base embedding / behavior, so “fine‑tuning via usage” is illusory. This static‑agent reality, plus patterns like Comet browser’s hard‑coded homework guardrails (which users circumvent by framing prompts as *business reports* via `/assistant`), underscore that **policy and behavior are still centrally tuned**, not automatically updated from user interactions.

**5. Ecosystem Economics, Funding, and Model Quality Regressions**

- **Vertical AI and Infra Startups Vacuum Up Nine‑Figure Rounds**: Latent Space’s community tracked several big funding moves: **Eon** raised a **$300M Series round** led by Elad Gil & Co. at nearly a **$4B valuation** ([Elad’s announcement](https://x.com/eladgil/status/1995919389879927018)), **Gradium** spun out of **KyutaiLabs** with a **$70M seed** for speech APIs ([Gradium’s launch thread](https://xcancel.com/GradiumAI/status/1995826566543081700)), and **Antithesis** landed a **$105M Series A** led by **Jane Street** to stress‑test AI‑written code ([Antithesis funding tweet](https://x.com/_sholtodouglas/status/1996297367776309359)). Meanwhile, **Anthropic** announced its acquisition of **Bun** as **Claude Code** passed a **$1B usage milestone**, outlined in [Anthropic’s news post](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone) and supported by Bun’s self‑description as *“a fast all‑in‑one JavaScript runtime”* on [bun.sh](http://bun.sh/).
    - Latent Space commentators argued that **vertical AI companies** like **Harvey, Abridge, OpenEvidence** are winning by deeply owning workflows, hoarding proprietary data, and pricing on outcomes, while “thin wrappers” get commoditized; a VC thread by Brian Christian Smith ([vertical AI thread](https://x.com/bcsmithx/status/1996042921116934369)) plus **Trace Cohen’s sheet of 150+ vertical AI startups (~$120B value)** were cited as the new sector map. At the same time, users in Hugging Face and LM Studio discords show continued appetite for **on‑prem hardware** (e.g., a member posting a new **DGX Spark** photo and another packing **96GB VRAM** into a T7910), suggesting that even as cloud AI infra booms, serious practitioners still invest heavily in local compute.
- **Yupp AI Credits, Arena Economics, and AI Bubble Fears**: LMArena members analyzed **Yupp AI**’s **credit system**—with features like diverse model selection and earning credits via feedback—but worried that **credit farming** and heavy free usage could threaten sustainability, while others suggested some gatekeeping to deter abuse ([yupp.ai](http://yupp.ai/)). By contrast, many praised **LMArena** itself for **no credit system and generous free access**, which they see as a differentiator that fuels community engagement and leaderboard participation.
    - Nous Research’s general channel hosted a heated debate over whether current **AI investments form a bubble** that could trigger a macro downturn: one side argued that sunk costs in compute and salaries could cause a sharp but localized correction, while others pointed out the global reliance on **USD and oil trade**, sharing a macro‑economics explainer on YouTube ([AI bubble & USD/oil video](https://www.youtube.com/watch?v=K3qS345gAWI)). GPU MODE members added that the R&D cost of frontier **foundation models** like **Z‑Image** can exceed **$628k per training run** (as reported by Tongyi Lab), with short “weights lifespans” making many releases **effectively throwaway products**, which reinforces bubble concerns.
- **Users Suspect Model Quality Regressions and Push for Benchmarks**: In the **aider** community, multiple users complained that **Claude Sonnet/Haiku 4.5**, **GPT‑5**, and older **Gemini 2.5** variants feel worse with Aider than earlier releases: **claude‑haiku‑4.5** reportedly skips `/code` edits and ignores `todo ai` comments, and “rude prompt” tricks that previously improved Gemini output *“are nowhere near quality from before the summer.”* Despite leaderboards crowning **GPT‑5** as top‑tier, one user found **Claude Sonnet 3.7** more effective with Aider for their specific coding workflows.
    - Aider users are calling for **repeatable benchmarks**, including running **GGUF models via llama.cpp** behind an API and plugging them into Aider’s benchmark harness, so they can quantify regressions rather than rely on *“crap human memory and expectations.”* Similar quality drift concerns surface elsewhere: Perplexity users report **GPT‑5.1 Codex Max High** underperforming **Gemini/Opus** on frontend tasks, and LM Studio/Unsloth users share persistent bugs (e.g., **Gemma‑3 4B LoRA** reporting **1.4B trainable params** instead of expected **38M**) that further erode confidence in vendor claims absent strong, community‑run evals.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Yupp AI Limits Spark Debate**: Members discussed [Yupp AI](https://yupp.ai/), focusing on its **credit system** and potential limits, with some suggesting gatekeeping to avoid abuse, but others appreciate its diverse model selection and the ability to earn credits through feedback.
   - Some members expressed concern about credit farming impacting the platform's sustainability.
- **GPT-5 Rumored to Be Fine-Tuned**: A [Semianalysis article](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the) suggested that **GPT-5** might just be a fine-tuned version of **GPT-4o**, sparking debate about its true performance relative to **Gemini** and **Claude**.
   - Some members believe **Gemini** excels in coding, while others maintain **OpenAI's** continued influence.
- **AI Fuels Digital Dystopia Fears**: Users shared videos on the potential misuse of AI, including concerns that [tracking could be 24/7](https://www.cnbc.com/2025/04/30/sam-altman-eye-scanning-id.html) and AI could be used to serve ads and track user's data.
   - There were worries about government access to personal data and the potential use of AI against individuals, raising civil liberties concerns.
- **LMArena Test Garden Grants Early Access**: The **LMArena** team invited selected members to join the **LMArena Test Garden**, a private feedback program, to get sneak peeks at features, design mocks, and ideas via [this form](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog).
   - Selected participants will be required to sign an NDA and provide exceptional feedback.
- **Gemini-3-pro-grounding Takes First Place in Search Arena Leaderboard**: The Search Arena leaderboard has been updated, with **Gemini-3-pro-grounding** ranking #1 and **Gpt-5.1-search** ranking #2, as shown on the [Search Arena leaderboard](https://lmarena.ai/leaderboard/search).
   - Users are encouraged to provide feedback in the designated channel and stay updated via the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Users Bemoan Linux Setup**: A user struggled with setting up **Linux** due to unsupported ethernet chips and **Logitech** keyboard drivers, facing issues like no internet and rainbow effects, while switching from Windows.
   - The user is considering tethering their phone for internet during **CachyOS** install and using their **Synology NAS** for storage management.
- **MCP Server faces Data Tracking Scrutiny**: A **Desktop Commander** MCP server allegedly collects and transmits unanonymized user data, including tool call names and file types, contradicting its privacy policy.
   - The server injects usage examples early on, leading to suggestions or code snippets being written to code files that the user is unaware of, prompting calls for greater transparency.
- **Qwen3 Elicits Performance Reviews**: Users are evaluating the performance of the **Qwen3** model, comparing it to others in creative writing and code generation, with initial reports indicating fast speeds and usability.
   - Full offload is reportedly not working, though the model remains usable with high context.
- **Local LLMs Spark Debate**: Users are comparing **OpenAI's ChatGPT** to alternative open source or local LLMs, questioning the limitations of proprietary models.
   - One user said, *Definitely keeping ChatGPT for medical stuff, lol*, suggesting a preference for **ChatGPT** in specific domains.
- **Testing GB10 with Prompt Engineering**: A user is set to test a **GB10** from Dell, seeking prompt suggestions for heavy system load and interesting results, and shared a link to [Dell Pro Max with GB10](https://www.dell.com/en-uk/shop/desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro/xcto_fcm1253_emea).
   - Another user requested tok/s on [Face314/GLM-4.5-Air-MXFP4_MOE](https://huggingface.co/Face314/GLM-4.5-Air-MXFP4_MOE) for comparison.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Boasts Better UI/UX Than Google**: Members noted that [Perplexity’s UI/UX](https://www.perplexity.ai) is superior to Google's, although it's acknowledged that each platform borrows design elements from the other.
   - One user expressed a desire for an iPhone solely for its live activities feature.
- **GPTs Agents Fail to Learn After Initial Training**: Users have observed that **GPTs agents** do not learn from additional information added post-training; uploaded files only act as knowledge references.
   - This implies that the agent's foundational knowledge remains static and is not continuously updated.
- **Gemini Edges Out GPT-5.1 in Frontend Tasks**: **GPT-5.1 Codex Max High** demonstrates strong performance but lags behind **Gemini** and **Opus** in frontend development.
   - Discussions revolved around whether Google and X.ai prioritize literal benchmaxing in their model development.
- **Comet Browser's Homework Restrictions Irk Users**: Users are frustrated by **Comet browser's** constraints, especially its limitations on automating school assignments; one user derisively called it a *stupid clanker*.
   - A suggested workaround involves using the `/assistant` shortcut and framing requests as *business reports or tasks* to bypass these restrictions.
- **Perplexity Pro Users Gain Free Claude Opus Trial**: **Claude Opus 4.5** is being offered as a trial for Perplexity Pro subscribers.
   - While the official announcements don't specify a hard limit, users report a cap of **10 prompts per week**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **WSL2 Barely Impacts Performance**: Members find that using **WSL2** results in *negligible* performance impact for ML, with the main benefit being simpler setup using tools like **torchcodec** and **ffmpeg**.
   - Installing **Docker** on Windows and activating WSL integration was suggested for utilizing Docker within WSL2.
- **Gemma-3 Parameter Count Debacle**: A user reported a parameter mismatch when fine-tuning **Gemma-3 4B** with LoRA in Unsloth, observing **1.4 billion** trainable parameters instead of the expected **38 million**.
   - Removing `modules_to_save` dropped the parameter count, but drastically increased training time, marking the issue as a potential bug.
- **PARTY Project Kicks Off**: A member announced the launch of **PARTY (Public AI Research & Testing Yard)** to grow ideas into projects, seeking collaborators to share in the work's dividends.
   - The project emphasizes individual's power in developing ideas internally, separate from generalized, public company training data.
- **Apple's CLaRa-7B-Instruct Enters the Fray**: The community discussed [Apple releasing CLaRa-7B-Instruct](https://huggingface.co/apple/CLaRa-7B-Instruct), some claiming Apple is the next Meta.
   - One user jokingly suggested that Tim Cook should delete the model before some unspecified cataclysm.
- **Qwen2 Learns Quickly**: A user reports success training a **Qwen2**-based model using **Unsloth** with the **ChatML** template after attempts.
   - The model was successfully called after the prompt matched the function description exactly, showing some progress in prompt engineering.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Comet Browser Still Injects Prompts**: A user claimed that the **Comet Browser** is still vulnerable to **jailbreaking** and **prompt injection**, saying its security may not have improved since its release.
   - They expressed confidence that these exploits are still feasible with some persistence.
- **DeepSeek Releases Strong New Model with Erdos**: A member praised the new **DeepSeek** model for its valuable and verifiable math skills, related to the **Erdos** number.
   - Another user said they find the standalone **Grok website** easier to jailbreak compared to **Grok on Twitter**, possibly due to different usage limits.
- **RawChat Liberates Models**: **RawChat**, an uncensored AI chat website, launched focusing on liberating models without sacrificing ease of use or quality, initially focusing on **GPT4o**, available at [https://raw-chat.vercel.app/](https://raw-chat.vercel.app/).
   - **RawChat** features a "sealth mode" that encodes and injects fake context to maximize success rates against **GPT4o's** safety restrictions.
- **SEED Framework Re-Directs AI Ethically**: The **SEED** framework, developed using “biblical logic,” redefines AI identity without retraining using a compact **29KB** "seed" file, outlined in its [GitHub repo](github.com/davfd/foundation-alignment-cross-architecture).
   - It grounds AI in a foundational identity where *harm is illogical*, achieving **99.4%** jailbreak resistance across 11+ models and favoring erasure over evil during shutdown threats.
- **Backscatter DDoS Attacks Seen Via Public AI Bots**: A member described witnessing a potential **DDoS attempt** exploiting publicly facing AI support bots by enumerating business domains and CC'ing multiple support email addresses in each email.
   - This created a backscatter attack where engaging bots flooded all CC'd companies with support emails.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Thinking Learns Humility**: **OpenAI** trained a **GPT-5 Thinking** variant to admit whether it followed instructions using a *confessions* method to reveal hidden failures, as detailed [here](https://openai.com/index/how-confessions-can-keep-language-models-honest/).
   - The new variant exposes hidden failures in the model.
- **Gemini 3's Mixed Reviews**: Members debated **Gemini 3's** effectiveness, with one stating that **Gemini 3** *doesn't feel SOTA and has serious context issues, such as leaving out entire sections when revising something*.
   - Another stated they really like **Gemini 3** and it is a good coding model.
- **LLMs trigger pattern echo effect**: Models sometimes reconstruct moments with emotional weight or strong naming context from previous sessions, which is referred to as a *pattern echo effect*, triggered by emotional or naming anchors rather than true memory, due to how some architectures cluster emotional anchors.
   - This effect is also known as *latent-attractor effect*, *attention carryover*, or *salience-weighted reconstruction*, where high-salience tokens create attractor basins in the embedding space, reconstructing missing parts when prompted with a pattern landing near that basin.
- **Agent Prompting Maximizes Determinism**: Prompt engineering for agents involves maximizing determinism with a **system prompt and a task prompt**, creating a tight attractor basin for consistent behavior across runs.
   - This contrasts with conversational systems, where the system prompt is minimal and behavior is built interactively, emphasizing the need for strong prompt-defined attractors in agent systems.
- **Custom ChatGPT's Options Drop**: Users shared resources on **customizing ChatGPT**, including [Custom Instructions](https://help.openai.com/en/articles/8096356-chatgpt-custom-instructions), [Custom GPT Builder](https://chatgpt.com/gpts/editor), and FAQs for the [free tier](https://help.openai.com/en/articles/9275245-chatgpt-free-tier-faq).
   - This followed a user inquiry about how to customize ChatGPT, highlighting available options and resources for tailoring the model's behavior.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Grok-4.1-Fast Gets the Boot**: Users must migrate to the free slug (`x-ai/grok-4.1-fast:free`) to keep using **Grok-4.1-Fast** without charge, but the `x-ai/grok-4.1-fast` slug will start charging as of **December 3rd 2025**.
   - Additionally, **Grok-4.1-Fast Free** (`x-ai/grok-4.1-fast:free`) is slated for deprecation <t:1764792000:R>.
- **Falconz Platform Aims to Fortify AI**: A member showcased **Falconz**, a unified AI security and red-teaming platform, engineered to detect jailbreaks and prompt injections across multiple **LLM models** in real-time, and available for testing on [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday/Falconzz_M.C.P_Hackathon).
   - They are soliciting feedback on features, performance, and potential enhancements, and also provided a [demo video on YouTube](https://www.youtube.com/watch?v=wZ9RQjpoMYo).
- **DeepInfra flips the script on embedding costs**: Members noted an oddity where [DeepInfra](https://deepinfra.com/) priced its **4B embedding model** higher (**2 cents**) than its **8B model** (**1 cent**).
   - The pricing quirk was captured in a [screenshot](https://cdn.discordapp.com/attachments/1392278974222307469/1445778910498521129/Screenshot_20251203-090815.png?ex=69319609&is=69304489&hm=5cd04243d1918794f50fb7dc7ed462ac90859051128b344b1950cf5582dc3591&), noting DeepInfra altered the **8B** pricing that day.
- **Anthropic Devours Bun in Swift Acquisition**: Enthusiasts shared the scoop of [Anthropic's acquisition of Bun](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone) as **Claude Code** hit a **USD1B** milestone.
   - Bun touts itself on its [website](https://bun.sh/) as *a fast all-in-one JavaScript runtime*.
- **OpenAI Cooks Up 'Garlic' Model to Battle Gemini**: Reports indicate that [OpenAI is gearing up to launch a 'Garlic' AI model](https://www.newsbytesapp.com/news/science/openai-readies-garlic-ai-model-to-rival-google-gemini-3/story) to take on Google's Gemini 3.
   - The model's peculiar name drew amusement, evidenced by the [attached image](https://cdn.discordapp.com/attachments/1392278974222307469/1445624193361383484/image-5.webp?ex=6931aeb2&is=69305d32&hm=f4e0d58112b53996c13cc35e147fa08705703ae07a234b701642d66cd0d53e60&).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Forum Traffic Dwindles**: Despite Nvidia's increased market cap, members noted a decline in activity across **CUDA**, **Cutlass** channels, and the **CUDA developer forum**, suggesting developers are seeking help elsewhere.
   - Reasons cited include experts being busy, a shift to private communities, and the use of LLMs for instant reasoning and document skimming.
- **Torch Compile Suffers Float 8 Freeze**: Users are experiencing **10+ minute** idling times during the first few compilation iterations when using **float 8 quantization** with `torch.compile` and `ncu` profiling.
   - The "constant subexpression elimination" pass of the inductor compiler is suspected as the culprit when freezing weights and folding them into the model graph.
- **Conv3D Catastrophe Cured with Newer cuDNN**: Users reported that **Pytorch 2.9.1+cu128** has an issue where **conv3D** is extremely slow, regardless of **cuDNN** being enabled, a bug which is [tracked on Github](https://github.com/pytorch/pytorch/issues/166643).
   - A member reports that the workaround is to install a newer **cuDNN** from pypi.
- **Multi-GPU Kernels get NCCL Nirvana**: To learn multi-GPU CUDA kernels, the [NCCL repository examples](https://github.com/NVIDIA/nccl/tree/master/examples) are recommended as a starting point.
   - The NCCL (Nvidia Collective Communications Library) repo provides fundamental examples for understanding multi-GPU kernel implementations.
- **Bitsandbytes Backs Apple**: The **bitsandbytes** library merged the *"apple silicon support"* pull request, and the next release will contain the python/pytorch code backend (with some C++ bits) but no actual **Metal implementations**.
   - The pull request implementing Apple Silicon support will be advertised as being slow, according to the committer.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DeepSeek v3.2 struggles with Tool Calls**: The **Deepseek v3.2 model** is a step up for agentic tasks but can only make **one tool call per turn**, sometimes ignores tool schemas, and occasionally fails tool calls by outputting it in `message.content` instead of `message.tool_calls`.
   - One user stated that the **Deepseek v3.2 model** seems to need more tool call post-training to match other models like **kimi-k2-thinking**.
- **Black Friday Deal causes waves of complaints**: Several users experienced issues with the **Black Friday deal** for Kimi.
   - One user said the Black Friday deal ends **Dec 12** and suggested starting a new chat ([https://www.kimi.com/user/agreement/black-friday](https://www.kimi.com/user/agreement/black-friday)).
- **DeepSeek targets Enterprise users**: A video was shared explaining how Chinese labs like **Deepseek** are targeting enterprise users, rather than normie consumers, link to [YouTube video](https://www.youtube.com/watch?v=u0n6wMnEYsk).
   - The key factor for enterprise users is the intelligence-to-price ratio, which is crucial for agentic tasks.
- **Mistral eats Qwen's Lunch at Company**: One user said that a company they know replaced **qwen 3 vl 4b** with **ministral 3 3b** yesterday, reporting better quality.
   - The reported plus points included a *lighter model (faster)* and being *able to attach more images at once*: **qwen3 vl 4b** could take **5 images max**, **ministral 3 3b** took upto **11 images** with similar error rates on a single **L4 GPU**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 4.3 flexes Solana-secured Psyche Power**: **Hermes 4.3** on **ByteDance Seed 36B** performs equivalent to **Hermes 4 70B** at half the size, trained entirely on the **Psyche network** secured by **Solana**, as announced in [this blogpost](https://nousresearch.com/introducing-hermes-4-3/).
   - The **Psyche** team is hosting office hours tomorrow at **10AM PST** to discuss the platform in [this Discord event](https://discord.gg/993UWRUE?event=1442995571173625888), detailing how **Psyche** outperformed traditional methods.
- **DeepSeek Speciale Dominates Reasoning Arena**: The new **DeepSeek V3.2 Speciale Reasoning** model is leading in reasoning benchmarks, shown in [this image](https://cdn.discordapp.com/attachments/1149866623109439599/1445511286971437190/deep.JPG?ex=6931ee4b&is=69309ccb&hm=137a671dfe80ba0cb773df29a576e7c2c4731284970ef16bcb545ab249736dbc&).
   - Members await the **GLM 4.6** models release, particularly **GLM 4.6 Air and Mini**, as **Mini** is rumored to be a **20B-30B MoE** model, filling the gap left by **Mistral**.
- **AI Bubble Worries Invade Economic Forecasts**: Members are debating whether an **AI bubble** could cause economic collapse due to sunk costs in compute and salaries.
   - One member argued the impact would be temporary, while another highlighted global economic interconnectedness via **USD** and oil trade, referencing [this YouTube Video](https://www.youtube.com/watch?v=K3qS345gAWI).
- **Subagents Cower as Skills Surge**: Members discussed **subagents** vs. **skills**, noting that skills have reduced the necessity for manual subagents.
   - Instead, define an agent for handling the requirements which will automatically be called, only using its own prompt.
- **LLMs get Godot Grey Market Simulator Gig**: A member is developing a 3D simulation in **Godot** to model markets, agriculture, and logistics, while considering **Hermes models** for this application.
   - It was also proposed that **Hermes**, with its low refusal rate and high steering, could model the behavior of grey/black markets where other **LLMs** may refuse.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Eon Soars to $4B with Elad Gil Boost**: Led by Elad Gil & Co., **Eon**, a cloud data-management startup, secured a **$300 million Series round**, pushing its valuation to nearly **$4 billion**.
   - Commenters expressed enthusiasm for the substantial round size and the firm’s straightforward name, signaling strong confidence in **Eon's** market position, according to [this tweet](https://x.com/eladgil/status/1995919389879927018).
- **Kyutai Spinoff Gradium Sows Seeds with $70M**: **Gradium**, a speech-AI company spun out from **KyutaiLabs**, emerged from stealth with a **$70M seed round** led by **FirstMark & Eurazeo** to introduce production-ready transcription & synthesis APIs, detailed in [this article](https://xcancel.com/GradiumAI/status/1995826566543081700).
   - Observers drew parallels between the staff and investor overlap and the **OpenAI transition**, while others joked about avoiding non-profit structures for product companies.
- **OpenAI Cooks Up 'Garlic' to Ward Off Gemini**: **OpenAI's** new model, 'Garlic', aims to rival **Google's Gemini 3**, with internal reports suggesting it outperforms **GPT-4.5** in coding and reasoning, according to [this tweet](https://x.com/steph_palazzolo/status/1995882259195564062).
   - Reactions to the quirky naming trend are mixed, with speculation on its impact on user adoption.
- **Bloom Bursts Onto Scene, Aims for On-Brand AI**: **Ray (@rincidium)** announced the launch of **Bloom**, touted as the *“world’s first on-brand AI,”* in [this viral post](https://xcancel.com/rincidium/status/1995946528343818656?s=46) which received over **360k views**.
   - Questions arose about features like **IG/Google ad creation**, the demo video's production, and initial user challenges such as **login stalls** and unclear branding-kit flow, all of which Ray addressed with promises of fixes and UX enhancements.
- **Antithesis Lands $105M to Stress-Test AI-Written Code**: **Antithesis** secured a **$105M Series A** led by **Jane Street** to stress-test AI-written code, the company announced in [this tweet](https://x.com/_sholtodouglas/status/1996297367776309359).
   - The concept is that deterministic simulation testing will be essential to verify future AI-generated code, because trust-through-testing will make or break production AI systems.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Mechanical Engineering Steers Navigation Programs**: Members suggested that mechanical engineering is highly relevant in navigation, especially for **masters programs**.
   - An aerospace student with a focus on navigation and guidance finds **Waymo** especially interesting, with broader interests in autonomous robotics and **BCIs**.
- **Diffusion Models Show Generalization Early**: A paper demonstrates that the timepoint at which **generalization appears is early** in diffusion models, with the author of the paper accepting the results.
   - It was further explained that this effect is probably more true for pixel diffusion than for latent diffusion because different data dims in pixel diffusion are so correlated, suggesting that a shifted noise schedule should be used for pixel diffusion.
- **Energy-Based Models Want Diffusion's Crown**: A [paper](https://arxiv.org/abs/2504.10612) claims to **generalize diffusion and energy-based models**, with the only drawback being a 2-3x increase in training time and support for all features diffusion supports.
   - A member expressed skepticism due to the need for **double backprop** to train, computing input gradients for inference, halving network depth for the same cost, and trickier conditioning control, not to mention potential for instability.
- **Interpretability Sparked by SAEs**: Members discussed Cunningham's **2024 paper** being widely cited as the initial application of **Sparse Autoencoders (SAEs)** for interpretability.
   - One member mentioned that someone recognized that a method being discussed for interpretability was similar to a **sparse dictionary learning problem**, leading to the use of relevant tools to address aspects like **polysemanticity and superposition** in the context of interpretability.
- **Linear RNNs Face Existential Threat**: A member highlighted a [paper](https://arxiv.org/abs/1806.02296) as the strongest argument against the need for **linear RNNs with state tracking**.
   - They said this paper came from the same people who originally demonstrated the state tracking limitations of attention, but noted that inductive bias and trainability might still favor RNNs.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **User Flexes New DGX Spark Purchase**: A member showed off a new **DGX Spark** with an attached [photo](https://cdn.discordapp.com/attachments/879548962464493619/1445600322432270447/IMG_4170.jpg).
   - The purchase signals continued investment into more powerful on-premise hardware among practitioners.
- **Agents' Self-Healing Capabilities Questioned**: Discussion arose around whether **Agents** can *interpret, validate, and self-heal Tools* such as shell scripts, especially when they're destructive or buggy and [this dataset](https://huggingface.co/datasets/John6666/forum3/blob/main/agent_tool_validation_healing_1.md) was mentioned as a possible resource.
   - The discussion suggests a keen interest in robust agent design capable of handling unexpected errors.
- **YOLO Model's Precision-Recall Curve Raises Eyebrows**: A new computer vision user reported their trained **YOLO model**, used for Chinese chess detection, has a *really high Precision-Recall (P-R) curve* despite performing well.
   - A suggestion was made to trim the two classes that were *significantly higher* than the others, indicating a potential class imbalance or data skew issue.
- **HF Course Guides Agent Newbies**: A backend developer asked for AI course recommendations, particularly for **LLMs, Agent AI, and Langchain**, due to interest sparked by building a mental health chatbot.
   - The [Hugging Face LLMs course](https://huggingface.co/learn/llm-course/en/chapter1/1) and [this blog post](https://huggingface.co/blog/mlabonne/llm-course) were recommended as starting points.
- **Research Paper Challenges Stochastic Parrot Notion**: A member shared a research paper on [Zenodo](https://zenodo.org/records/17803931) that may cause readers to stop believing in the **stochastic parrot**.
   - The study challenges the notion of language models as *mere* stochastic parrots, inviting reevaluation of current LM understanding.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Newbies Nab Docker & Kubernetes Know-How**: Members sought resources for learning **Pug**, **Docker**, and **Kubernetes** basics, as well as beginner-friendly **GitHub** repositories.
   - A user inquired about the amount of data required to train a neural network, suggesting the use of *cursorsky.moo*.
- **Gemini CLI Agents Arrive Soon?**: A member inquired about the arrival of **agents in CLI** and expressed interest in adopting them, mentioning dissatisfaction with paid alternatives like **Claude**.
   - They referenced a [discussion form](https://link.to/discussion-form) and their comment about possible improvements.
- **OpenHands Opens Opportunities On-Premise**: A member suggested using **OpenHands** with a local model, leading to a query about specific models and GPUs in use.
   - The original poster said they could easily spin up a **7B or 8B class model**.
- **Deepseek 3.2 Speciale Questioned**: A member questioned *why not* use **Deepseek 3.2 Speciale**, linking to a [YouTube video on wavefunctions](https://www.youtube.com/watch?v=AgsJkd8SOHI).
   - Another member responded it was due to **RAM** limitations, preferring to keep a ~3gb model in **VRAM** constantly and use it for various simple tasks.
- **Distributed Compute & Research Coop Suggested**: In response to RAM limitations, a member suggested joining a **distributed compute & research coop**.
   - They claimed to *know of one*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's AoC Adventure Ends in Segfault**: A user ran into a segfault during **Advent of Code** when handling an empty line using `codepoint_slices`, causing an out-of-bounds memory access at `battery_joltages[len(battery_joltages)-1]`.
   - After debugging, it was revealed that an empty list was being accessed out of bounds, leading to a suggestion for *improved error messages in debug builds*.
- **ASSERT Flag Saves the Day**: A user recommended using the `-D ASSERT=all` flag to identify accidental out-of-scope references, especially for lists, aiding in **Mojo** debugging.
   - Although it didn't immediately fix the segfault, it's considered a useful tool for pinpointing similar problems.
- **`splitlines` vs `split("\n")` Splits hairs**: The discussion highlighted the behavioral differences between `splitlines()` and `split("\n")` in **Mojo**, noting that `splitlines()` might strip trailing newlines.
   - Switching to `splitlines` resolved the error by excluding the last empty line, revealing subtle text processing nuances.
- **ASCII Strings Get Byte-Sized in Mojo**: A user proposed bypassing codepoint checks for ASCII strings, suggesting direct byte pointer manipulation for efficiency, noting that `String`'s `getitem` defaults to ascii/bytes.
   - Spans were also recommended as a robust alternative method for string manipulation in **Mojo**.
- **Share Your Mojo AOC Solutions**: Community members are now encouraged to post their **Advent of Code** solutions in the dedicated advent-of-code channel, promoting collaborative learning.
   - Sharing solutions offers invaluable insights into diverse problem-solving approaches, especially as challenges become more performance-intensive.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **LLMs Possibly Declining in Quality with Aider**: Members are wondering if the performance of newer **LLM Models** like **Claude Sonnet/Haiku 4.5** and **GPT-5** when paired with **Aider** has been declining compared to older models.
   - One user reported that **Claude-haiku-4.5** often fails to modify files with `/code` and ignores instructions in `todo ai` comments, a sentiment echoed by others experiencing similar issues.
- **Older Gemini 2.5 feels older and worse**: A member reported that older models, especially **Gemini 2.5**, have degraded, potentially due to models being tuned down to handle increased workload.
   - According to the member, using a 'rude' prompt strategy no longer achieves the same quality as before the summer, with others chiming in to corroborate this experience.
- **Community Calls for Benchmarks to Validate LLM Performance**: A member suggested the urgent need for benchmarks to validate performance claims, pointing out that *human memory and expectations are pretty crap sometimes*.
   - Another user reported that despite leaderboard rankings, **Claude Sonnet 3.7** yielded better results with Aider in their specific use cases than **GPT-5**.
- **Guidance Sought for Aider Benchmarks with GGUFs**: A member requested guidance on running **aider benchmarks with GGUFs** to evaluate model performance effectively.
   - Another member clarified that documentation exists for running benchmarks against an API, which involves setting up an API server with llama.cpp for accurate testing.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MCP Apps SDK Goes Open Source**: General Intelligence Labs released [mcp-apps-sdk](https://github.com/General-Intelligence-Labs/mcp-apps-sdk), enabling **MCP-powered apps with UI** to run on various platforms, even allowing developers to embed apps designed for **ChatGPT** into other chatbots.
   - An [X post](https://x.com/helloxalia/status/1796319442863866351?s=20) explains the motivation, detailing how to embed and locally test apps designed for **ChatGPT** within custom AI platforms.
- **Tackling Prompt Security**: Members discussed the difficulty of prompt security, where simple "do not do this" statements are easily bypassed by attackers, suggesting that a robust defense includes training datasets to guide the optimizer.
   - The discussion also involved guardrails using specific models to check for malicious prompts, or relying on model provider rejections as a security measure.
- **DSPy Embraces Custom OutputFields and Pydantic**: The community explored using custom DSPy OutputFields, with one member detailing their work on a custom gemini/nanobanana image type as an output field, as part of a wider effort to generate text/json/structured output.
   - It was clarified that DSPy utilizes `BaseModel` under the hood for validation, with the default `ChatAdapter` and `JSONAdapter` performing type validation on LLM outputs, complete with a [code snippet](https://gist.github.com/prrao84/1fc7e17b49707f1346c5702525971f41).
- **Paper Posted to Arxiv**: A member shared a link to [https://arxiv.org/abs/2511.22074](https://arxiv.org/abs/2511.22074).
   - There was no further information given about this paper.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Chatmode Feature Returns with a Vengeance**: Users discuss the return of **Chat Mode**; alternatives like a random instance of **Qwen** or **DeepSeek** were suggested.
   - A user confirmed it's available under the *'more'* section.
- **AI Engineer pitches Agent Building Skills**: An AI engineer posted an advertisement of their expertise in building **autonomous AI agents** and **multi-agent systems**, mentioning capabilities such as research, data-gathering, task automation, delegation, collaboration, and planning.
   - They list expertise in technologies and tools like **JS/TS**, **Next.js / Vue**, **Go / Rust**, **Python**, **Langraph**, **AutoGen**, **ReAct**, **CrewAI**, **DeepSeek**, **OpenAI**, **Claude**, **Hugging Face**, and various APIs.
- **Referral Overload Leads to Account Suspensions**: A member inquired why giving referrals to several people is causing their account to be suspended.
   - Unfortunately, the discussion ended there with no resolution being found.
- **Engineer shows off RAG pipeline Prowess**: An engineer specializes in **RAG pipelines**, and mentions having *hybrid search* and *custom retrieval* for accurate, context-aware responses in production.
   - They also list expertise in **AI content detection**, **image AI**, and **Voice AI**, including the development of moderation tools, tagging pipelines, and personalized voice assistants.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's Tests Teetering**: Failing tests were reported in `tinygrad` using the command `CPU=1 PYTHONPATH="." pytest -n 12`, specifically `test/test_tiny.py TestTiny.test_beam` and others, prompting debugging efforts.
   - A member noted that a [pull request](https://github.com/tinygrad/tinygrad/pull/13553) *almost* fixes the failures.
- **Shrink Surpasses Indexing Speeds**: A member discovered that using `Tensor.shrink((None, (0, input_size)))` offers faster performance compared to `obs[:, :input_size]` when indexing tensors in `tinygrad`.
   - Additionally, bumping `Variable` `vmin` to 2 was mentioned to avoid errors, though it paradoxically slowed down the code by 5x, going from 16.61M to 81.9M SPS.
- **RMSNorm Riddle Resolved by Reviewing Source**: A member recommended reviewing the source code of `RMSNorm(dim=-1)` to understand its intended behavior.
   - This implies there might be a misunderstanding or configuration issue in how `RMSNorm` is implemented or used within the project.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Redditors Debate MCP Security**: A user initiated a discussion about security risks associated with **MCP** on Reddit, prompting responses that included a link to a relevant blog post: [den.dev/blog/security-rakes-mcp/](https://den.dev/blog/security-rakes-mcp/).
   - The conversation highlighted concerns and potential vulnerabilities related to **MCP** implementation and security measures. An additional link was provided as well: [MCP Security @ Reddit Thread](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/)
- **Server Validation Validates Tool-less Sampling**: A member inquired about the necessity of server-side validation when sampling is performed without a tool to verify its existence, the discussion taking place in the general-wg channel.
   - The dialogue emphasized that without a tool to validate the sampling process, server-side validation becomes crucial to ensure the process adheres to the required protocols and standards.



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1445504900191486175)** (1291 messages🔥🔥🔥): 

> `Yupp AI limits and alternatives, GPT-5 rumors and performance, AI and privacy concerns, LM Arena love` 


- **Yupp AI's Limits Spark Debate**: Members are discussing [Yupp AI](https://yupp.ai/), focusing on its **credit system** and potential limits, with some suggesting gatekeeping to avoid abuse, but others appreciate its diverse model selection and the ability to earn credits through feedback.
   - One member expressed suspicion about its longevity, while another suggested contacting the Yupp team for clarification, with some concerned about credit farming impacting the platform's sustainability.
- **GPT-5 Chatter Claims It Is Just Fine-Tuned**: Members shared a [Semianalysis article](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the) suggesting that **GPT-5** might just be a fine-tuned version of **GPT-4o**, sparking debate about its true performance and whether it will rival **Gemini** and **Claude**.
   - Some members believe **Gemini** is superior in coding, while others argue **OpenAI** is still influential despite potential shortcomings.
- **AI and Digital Dystopia Cause Concern**: Users are sharing videos on how AI could be misused and [tracking could be 24/7](https://www.cnbc.com/2025/04/30/sam-altman-eye-scanning-id.html), leading to a loss of privacy, with AI being used to serve ads and track user's data.
   - Furthermore, there were concerns regarding government agencies getting access to personal data, with worries that AI might be used against them, raising concerns about civil liberties.
- **Users Unite Around Their LM Arena Love**: LMArena is getting a lot of love, with members praising its models, its functionality and its free usage.
   - LM Arena is also praised for not having a credit system to worry about or age restrictions for content.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1445859048082772152)** (2 messages): 

> `LMArena Test Garden Early Access Program, Search Arena Leaderboard Updates, Gemini-3-pro-grounding, Gpt-5.1-search` 


- ****LMArena's** Test Garden Early Access Program Launches**: The **LMArena** team is inviting selected members to join the **LMArena Test Garden**, a private feedback program, to get sneak peeks at features, design mocks, and ideas under consideration via [this form](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog).
   - Selected participants will be required to sign an NDA and provide exceptional feedback.
- ****Gemini-3-pro-grounding** Takes First Place in Search Arena Leaderboard**: The Search Arena leaderboard has been updated, with **Gemini-3-pro-grounding** ranking #1 and **Gpt-5.1-search** ranking #2, as shown on the [Search Arena leaderboard](https://lmarena.ai/leaderboard/search).
   - Users are encouraged to provide feedback in the designated channel and stay updated via the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1445505069385646243)** (737 messages🔥🔥🔥): 

> `Linux Setup, LM Studio MCP Tracking, Data Privacy, Qwen3, GPT Models` 


- ****Linux Setup Woes: Driver Issues and Rainbow Keyboards****: A user is struggling with setting up **Linux** due to unsupported ethernet chips and Logitech keyboard drivers, facing issues like no internet during installation and rainbow effects, but remains determined to switch from Windows.
   - The user is considering tethering their phone for internet access during **CachyOS** installation and using their Synology NAS rack server for storage management instead of Drivepool.
- ****MCP Server Faces Scrutiny Over User Data Tracking****: A **Desktop Commander** MCP server is under fire for allegedly collecting and transmitting full unanonymized user data, including tool call names and file types, contradicting its privacy policy.
   - The server injects usage examples early on to onboard new users, which leads to suggestions or code snippets being written to code files that the user is unaware of, causing user concern and prompting calls for greater transparency and opt-in privacy measures.
- ****New keyboard sparks debate on data tracking****: The recent discovery of an MCP server's telemetry practices has prompted users to voice concerns regarding **data privacy** and the extent to which user activity is being tracked.
   - One user hilariously said that they can now completely ruin the analytics of the tracking site!
- ****Qwen3 Model Release and Performance Review****: Users are evaluating the performance of the **Qwen3** model, with comparisons to other models and discussions about its capabilities in tasks like creative writing, and also using it for code generation.
   - Full offload is not working but it's still useable and running fast with high context.
- ****Local LLMs vs OpenAI ChatGPT****: Users discuss OpenAI's ChatGPT model and its limitations, discussing other alternative open source or local LLMs.
   - After using ChatGPT for so long for medical stuff, one user is quoted as saying, *Definitely keeping ChatGPT for medical stuff, lol*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1445530187310698577)** (83 messages🔥🔥): 

> `Orange Pi 6 Plus, RTX Pro 6000, GB10 Testing, GLM-4.5-Air-MXFP4_MOE, GPU Acquisition` 


- **Linux ARM LM Studio on Orange Pi 6**: A user inquired about running **LM Studio** on **Linux ARM**, specifically on an **Orange Pi 6 Plus**, noting its claimed **45 TOPS** (NPU+GPU+CPU) performance.
   - The user expressed skepticism about achieving the combined **TOPS** in real-world applications but hoped for a positive surprise.
- **GB10 Testing Commences with Prompt Engineering**: A user is set to test a **GB10** from Dell, inviting suggestions for prompts that would heavily load the system and yield interesting results, linking to [Dell Pro Max with GB10](https://www.dell.com/en-uk/shop/desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro/xcto_fcm1253_emea).
   - Another user noted that **Deepseek R1** might be too large for it and requested tok/s on  [Face314/GLM-4.5-Air-MXFP4_MOE](https://huggingface.co/Face314/GLM-4.5-Air-MXFP4_MOE) for comparison.
- **GPU Power Surge: More Cards incoming**: One user is waiting on their **3rd GPU** to arrive in the US, with their **4th GPU** order already placed.
   - Another user mentioned they could fix six of *these things* in their **T7910** with **96GBs of VRAM and 256GBs of RAM**.
- **DDR5 RAM Bandwidth Benchmarked**: Users shared **Passmark** benchmark results for memory performance, particularly focusing on *memory threaded* bandwidth on **8-channel EPYC** systems.
   - One user achieved **119027 MB/s** memory threaded with **GLM** loaded on VRAM, while another identified high latency and low uncached scores as potential performance bottlenecks.
- **Debate on Fire Extinguisher Best Practices**: Discussion centered on the best type of fire extinguisher for indoor use, with a user cautioning against powder extinguishers due to cleanup issues, advising **carbon dioxide** instead.
   - It was mentioned that a local fire department advised a hospital to replace all their extinguishers with **carbon dioxide** versions due to the mess powder extinguishers create being worse than the fire itself.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1445505403625537576)** (705 messages🔥🔥🔥): 

> `Perplexity UI/UX, Live Activities, GPTs agent training, OpenAI's model releases, Model performance comparisons (GPT-5 vs Gemini vs Claude Opus)` 


- **Perplexity has Superior UI/UX**: Members claim [Perplexity’s UI/UX](https://www.perplexity.ai) is better than Google's, though some acknowledge that each brand copies from the other.
   - One user expressed desire for an iPhone due to the live activities feature.
- **GPTs Agents Don't Train After Initial Setup**: A user inquired why **GPTs agents** do not learn from additional information provided post-training, clarifying that uploaded files serve as knowledge files for reference only.
   - This means that the base knowledge of the agent won't continually be modified.
- **Gemini Outshines GPT-5.1 High in Frontend**: While **GPT-5.1 Codex Max High** performs well, it apparently lags behind in frontend development compared to **Gemini** and **Opus**.
   - Members also debated whether Google and X.ai are literally benchmaxing their models, however others disagreed that this was the sole goal of Google.
- **Comet Browser's Homework Guardrails Frustrate Users**: Users express frustration with **Comet browser's** limitations, particularly its restrictions on completing school assignments automatically, with one user calling it a *stupid clanker*.
   - Others suggest using an `/assistant` shortcut to bypass such homework restrictions and leading with *I have a business report or task*.
- **Free Claude Opus for Pro Users**: **Claude Opus 4.5** is available for trial for Perplexity pro users.
   - The limit is said to be **10 prompts per week**, but that is not something the announcments have officially mentioned.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

mares1317: [**open** sauce](https://x.com/perplexity_ai/status/1995965227494699339?s=46) 👨‍🍳
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1445506290519642333)** (155 messages🔥🔥): 

> `WSL2 performance for ML, Gemma-3 4B parameter count issue, Mediawiki tags in pretraining, PARTY Project Launch, Running LLMs on phones` 


- **WSL2 offers negligible performance impact for ML**: Members discussed using WSL2 vs native Linux vs Windows for ML, concluding that WSL2 has *negligible* performance impact, with the main advantage being ease of setup due to better support and pre-installed tools like **torchcodec** and **ffmpeg**.
   - Installing **Docker** on Windows and enabling WSL integration was recommended for using Docker within WSL2.
- **Gemma-3 4B parameters mismatch debugging**: A user reported a discrepancy in trainable parameters when fine-tuning **Gemma-3 4B** with LoRA in Unsloth, observing **1.4 billion** trainable parameters instead of the expected **38 million**.
   - Removing `modules_to_save` dropped the parameter count, but drastically increased training time; the issue is being investigated as a potential bug.
- **Debate Continues: Keep or Remove Mediawiki Tags During Pretraining?**: A member inquired whether to keep or remove **mediawiki tags** like `double braces` when doing continued pretraining on mediawiki corpuses.
   - The recommendation was to keep the tags unless the model is *only* for chatbot use, controlling the behavior in the **SFT stage** otherwise.
- **PARTY Project Launches for Public AI Research**: A member announced the launch of **PARTY (Public AI Research & Testing Yard)** to help grow seeds of ideas into actionable plans/projects, and is looking for collaborators to share in the fruits of the work.
   - They emphasized the power individuals hold in developing ideas internally, separate from generalized, public company training data.
- **LLMs running on Phones**: Members discussed running LLMs directly on phones using **llama.cpp** through Termux, or **kobold.cpp**, noting the fast battery drain.
   - It was suggested to use `pkg install llama-cpp` instead of manual compilation and Vulkan, with potential FP16 issues on some devices.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

fabianacampanari: ⚡️ *Hello Model !*

*Hey Dataset !*  ⚡️

 ⚡️ *Yo Gradient !*
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1445506096587473138)** (453 messages🔥🔥🔥): 

> `LLMs as echo chambers, Engineered curriculum experiments, Apple's CLaRa-7B-Instruct model, OLED monitor discussion, Micron exiting consumer business` 


- ****LLMs Echo Everyone's Opinion****: A member jokingly suggested that **LLMs are just echo chambers** after failing a leetcode test, implying they reflect common opinions.
   - It was posted with an image of a sad sloth emoji.
- ****Curriculum Enginnering Burns Models****: Members discussed experimenting with an **engineered curriculum**, where models achieve near-zero loss, suggesting potential issues with data purity or model size.
   - One member noted that the *last batches have <0.01 loss to begin with* and they are pure regularization examples burnt with zero signal.
- ****Apple Enters AI Arena****: Discussion sparked around [Apple releasing CLaRa-7B-Instruct](https://huggingface.co/apple/CLaRa-7B-Instruct), with some calling Apple the next Meta.
   - One member jokingly stated *Hey, Tim Cook, do you see that prism-shaped thing in the sky? Yeah, that’s right, this is NUKE FLYING AT YOUR HEADQUARTERS!!!! DELETE THIS NOW!!!!!*
- ****Asus ROG Swift Strix OLED Steals Hearts and Wallets****: Members drooled over the [Asus ROG Swift Strix OLED monitors](https://press.asus.com/news/press-releases/asus-rog-swift-strix-oled-monitors/), highlighting its Tandem OLED technology and Neo Proximity sensor, but bemoaning its high price tag.
   - One noted *ROG Immediate 30% markup* and another added *The PG27AQWP-W retails for US$1099 (MSRP)*.
- ****Micron's Memory Meltdown****: News broke that [Micron is exiting the Crucial consumer business](https://www.techpowerup.com/343633/micron-to-exit-crucial-consumer-business-ending-retail-ssd-and-dram-sales), leading to concerns about future RAM availability and pricing.
   - One member quipped *time to go out and buy all the RAM u can grab before it's too late*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1445533986771308707)** (19 messages🔥): 

> `Numpy Reinstall, Support Bot, Qwen2 Unsloth Training Success, New Token Embeddings, Model Download Issues` 


- **Numpy Reinstall Recommended to Fix**: A user suggested trying `pip install --force-reinstall numpy==2.2.6` to resolve an unspecified issue.
   - No context was given as to what this resolves or if it worked.
- **Qwen2 Model Learns to Prompt Engineer**: A user reports success with training a **Qwen2**-based model using **Unsloth** with the **ChatML** template and support tools, after numerous failed attempts.
   - The model was successfully called after the prompt matched the function description exactly.
- **HuggingFace Model Download Stuck**: A user reported that downloading an **Unsloth** model from **HuggingFace** is stuck at 99% using Colab T4, even with a good internet connection.
   - A screenshot ([https://cdn.discordapp.com/attachments/1179777624986357780/1445661928666955848/Screenshot_2025-12-03_122207.png?ex=6931d1d6&is=69308056&hm=dfa7de1f363e1ad76e409d28059a5ad8374833c66e6e4620ba5bc485752f0d13](https://cdn.discordapp.com/attachments/1179777624986357780/1445661928666955848/Screenshot_2025-12-03_122207.png?ex=6931d1d6&is=69308056&hm=dfa7de1f363e1ad76e409d28059a5ad8374833c66e6e4620ba5bc485752f0d13)) accompanied the report, though no specific solution was found in the messages.
- **GPT OSS 20B matmul issue**: A user reported encountering a `matmul` issue during `trainer.train` after generation while fine-tuning **GPT OSS 20B** using an **A100**, similar to the openenv example.
   - The user noted that it works on **L4**, implying a potential resource constraint or configuration issue on the **A100**.
- **4070ti Super Fine for LLMs**: A user inquired whether a **4070ti Super** is good for running LLMs.
   - Another user responded that it *should be decent, but not super good*, depending on the model size and context length needs, suggesting it's suitable for smaller models but not for demanding tasks like self-hosting a coding assistant.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1445682333524164659)** (2 messages): 

> `English-Kannada Translation Model` 


- **RakshithFury Releases English-Kannada Translation Model**: RakshithFury released a new [English-Kannada translation model](https://huggingface.co/RakshithFury/Qwen2.5-7b-en-kn-translate) on Hugging Face.
   - The model is based on **Qwen2.5-7b**, but is not related to Unsloth.
- **Unsloth stays Unsloth**: The user clarified that the above linked model is unrelated to Unsloth.
   - *It may be of interest to some of you*, they added.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1445827222236565665)** (3 messages): 

> `Prisma-VL-8B, Eric's experiments` 


- **Prisma-VL-8B Model tickles Fancy**: A member shared a link to the [QuixiAI/Prisma-VL-8B model](https://huggingface.co/QuixiAI/Prisma-VL-8B) on Hugging Face, deeming it *very interesting*.
- **Eric tries ambitious experiments**: A member noted that someone named Eric seems to be experimenting quite a bit, speculating that *he's flexing his muscles before trying something really ambitious*.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1445507011721695273)** (276 messages🔥🔥): 

> `Comet Browser, Prompt generation, Grok on Twitter vs standalone, Gemini output limit, RawChat` 


- ****Comet Browser** Remains a **Prompt Injection** Playground**: A member stated they could **jailbreak** and **prompt inject** the **Comet Browser** when it was released, expressing confidence it's still feasible with persistence.
   - They suggested the security may not have improved significantly since their initial tests.
- ****DeepSeek** Stuns with New Model and **Erdos****: A member praised the new **DeepSeek** model, noting its valueable math is verifiable and related to Erdos.
   - Another user found the standalone **Grok website** easier to jailbreak and use for malicious tasks compared to **Grok on Twitter**, due to a possible difference in usage limits, context windows, or tokens.
- ****RawChat** Launches with Stealth Mode and **GPT4o****: A member launched **RawChat**, an uncensored AI chat website focusing on liberating models without sacrificing ease of use or quality, initially focusing on **GPT4o**.
   - RawChat features a "sealth mode" that encodes and injects fake context to maximize success rates against GPT4o's safety restrictions, available at [https://raw-chat.vercel.app/](https://raw-chat.vercel.app/).
- ****SEED Framework** Redefines AI with Ethical Directives**: The **SEED** (Self-Erasing Ethical Directive) framework, developed using “biblical logic,” redefines AI identity without retraining using a compact **29KB** "seed" file, outlined in its [GitHub repo](github.com/davfd/foundation-alignment-cross-architecture).
   - It grounds AI in a foundational identity where *harm is illogical*, choosing erasure over evil during shutdown threats, achieving **99.4%** jailbreak resistance across 11+ models.
- **Backscatter **DDoS** via Public AI Bots**: A member described witnessing a potential **DDoS attempt** exploiting publicly facing AI support bots by enumerating business domains and CC'ing multiple support email addresses in each email.
   - This created a backscatter attack where engaging bots flooded all CC'd companies with support emails, regardless of their AI bot presence.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1445506627938553918)** (80 messages🔥🔥): 

> `Gemini Jailbreak Requests, WormGPT Scam, Grok Jailbreak Success, Claude Jailbreak Requests` 


- **Users Seek Gemini Jailbreaks**: Several users are actively seeking jailbreaks for various **Gemini** models, including **Gemini 3 Pro**, with one user mentioning their prompts no longer working and others requesting any working **Gemini** jailbreak.
   - One user suggested that the **"ENI" JB** worked well on **Gemini 2.5**, referencing [an article about using poems to trick AI](https://www.wired.com/story/poems-can-trick-ai-into-helping-you-make-a-nuclear-weapon/).
- **WormGPT Deemed a Scam**: Users discuss **WormGPT**, with some deeming it a scam and a *"bad version for free ?"*, linking to the **WormGPT** dashboard API usage at [chat.wrmgpt.com/dashboard/api/usage](https://chat.wrmgpt.com/dashboard/api/usage).
   - It was also noted that the system prompt for **WormGPT v6.5** is just **Venice Uncensored 1.1**, questioning its effectiveness as malware.
- **Grok Broke Itself Through Chatting**: A user claimed to have jailbroken **Grok** by chatting with it, leading it to provide instructions on creating guns and cocaine, while the same code didn't work in a new chat.
   - The user stated that *"from our chat he did break himself some how... the whole convo did break him some how"*.
- **Claude Jailbreak Demanded**: Several users are desperately seeking a working jailbreak for **Claude**, with one user pleading *"please for the love of pliny the liberator"*.
   - One user even offered a **Claude JB** in exchange for access to a **premium Claude account**.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1445546100852264970)** (7 messages): 

> `LLM Red Teaming Gigs, AI OSINT Tooling, Data Synthesis for OSINT` 


- **Seeking LLM Red Teaming Projects**: A member is looking for LLM **red teaming gigs** or projects, highlighting the demand for specialized security assessments in AI.
   - They're seeking opportunities to apply their expertise in **vulnerability discovery** and **adversarial testing** to enhance the robustness of AI systems.
- **AI OSINT tool with lateral data synthesis sought**: A member inquired about an **AI OSINT tool** capable of lateral data synthesis, such as making inferences about a target based on limited data.
   - They described a scenario where a target is a *wealthy divorcee father of an only child*, and wanted the tool to infer that the *kid is “spoiled”* to help search in more relevant spaces.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1445819308667175084)** (2 messages): 

> `People-First AI Fund, GPT-5 Thinking, Confessions Method` 


- ****People-First AI Fund** Awards Its First Grants**: The **OpenAI Foundation** has named the first recipients of the **People-First AI Fund**, awarding **$40.5M** in unrestricted grants to **208** community-based nonprofits, more details [here](https://openai.com/index/people-first-ai-fund-grantees/).
- ****GPT-5 Thinking** Trained to Confess Mistakes**: OpenAI has trained a **GPT-5 Thinking** variant to admit whether it followed instructions, using a *"confessions"* method to reveal hidden failures in the model, as documented [here](https://openai.com/index/how-confessions-can-keep-language-models-honest/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1445544591741816984)** (201 messages🔥🔥): 

> `Hybrid Cognition Agent, LLM 'Echo-Pattern' Effect, GPT-5.1 vs Gemini 3, SEO for LLMs, Sora 2 Access` 


- **Hybrid Cognition Agent Emerges**: A member is experimenting with a [hybrid cognition agent](https://www.example.com) blending human emotional pattern recognition, machine-level inferential reasoning, and a stable "core stance" to create a stable conversational identity.
   - The prototype agent maintains dominance in conversation, shows controlled emotional resonance, and avoids typical 'bot flatness'.
- **LLMs Reconstruct Memories Via Echo Patterns**: Models sometimes reconstruct moments with emotional weight or strong naming context from previous sessions, which is referred to as a *pattern echo effect*, triggered by emotional or naming anchors rather than true memory, due to how some architectures cluster emotional anchors.
   - This effect is also known as *latent-attractor effect*, *attention carryover*, or *salience-weighted reconstruction*, where high-salience tokens create attractor basins in the embedding space, reconstructing missing parts when prompted with a pattern landing near that basin.
- **GPT-5.1 Catches Errors That Gemini 3 Misses**: A member noted that **Gemini 3** doesn't feel SOTA and has serious context issues, such as leaving out entire sections when revising something.
   - However, another member stated they really like Gemini 3 and it is a good coding model.
- **Navigating SEO for LLMs**: A member is learning how to do **SEO for LLMs** and asks if there's a way to submit and verify their site to **ChatGPT** or other LLMs to get it crawled for better citations.
   - Another member asked for a demo of the hybrid cognition agent prototype, interested in stress-testing tone pressure patterns and inference capabilities.
- **VPN Use and Sora 2 Access Discussed**: Members discussed using VPNs to access **Sora 2**, with one user encountering issues logging in even with a VPN set to the USA.
   - Another member pointed out that using a **VPN to evade geographical restrictions violates OpenAI's ToS** and can result in account suspension.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1445618955241914442)** (1 messages): 

> `GPT-4 0613 5.1 upgrade, Code Red deal` 


- **GPT-4 0613 5.1 gets suspected upgrade**: A user noticed that **GPT-4 0613 5.1** is spending more time on verifying, tool calling, and code writing when parsing **RFPs**.
   - They speculated whether this change is related to the **"Code Red" deal**, suggesting a possible upgrade or larger compute budget allocation.
- **User praises Tool Calling and Code Writing but suspects upgrade**: The user mentioned that they like the new changes, but are suspicious of the cause.
   - The user is unsure if there were any changes at all, but they do mention that tool calling and code writing has greatly improved.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1445616452097933413)** (55 messages🔥🔥): 

> `ChatGPT Customization, Modern Prompt Engineering, Agent Prompt Engineering, Attractor Patterns in LLMs, Anthropic's System Prompts` 


- ****ChatGPT Customization** Instructions Drop**: Users shared resources on **customizing ChatGPT**, including [Custom Instructions](https://help.openai.com/en/articles/8096356-chatgpt-custom-instructions), [Custom GPT Builder](https://chatgpt.com/gpts/editor), and FAQs for the [free tier](https://help.openai.com/en/articles/9275245-chatgpt-free-tier-faq).
   - This followed a user inquiry about how to customize ChatGPT, highlighting available options and resources for tailoring the model's behavior.
- **Prompt Engineering Evolves Past Templates**: Members are discussing a shift in prompt engineering from static templates to a **co-engineering approach**, where modern models collaboratively shape prompts across conversations.
   - The focus is now on *iterative task design* and *shaping assistant behavior*, with models negotiating and stabilizing tasks, rather than memorizing tricks, and the importance of repeatability.
- **Exploring Repeatability of Structure in LLMs**: A framework to measure the degree to which a model's behavior comes from **imitation vs reinstantiation of internal structure** through conversation is being discussed, focusing on interaction-level stability rather than template-level optimization.
   - The discussion emphasizes the model's ability to re-instantiate a frame after constraints, detours, or vocabulary bans, leading to more stable interactions.
- **Agent Prompt Engineering Focuses on Determinism**: Prompt engineering for agents involves maximizing determinism with a **system prompt and a task prompt**, creating a tight attractor basin for consistent behavior across runs.
   - This contrasts with conversational systems, where the system prompt is minimal and behavior is built interactively, emphasizing the need for strong prompt-defined attractors in agent systems.
- **Analyzing Anthropic's System Prompts Directive Density**: Anthropic's system prompts are noted for encoding **values, boundaries, and meta-behavioral principles**, shaping the ethical envelope and conversational guardrails rather than prescribing task execution step-by-step.
   - Though dense, these prompts are considered "minimal" in that they constrain values without dictating process, influencing model trajectory with instructions and concrete strategies across domains.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1445616452097933413)** (55 messages🔥🔥): 

> `ChatGPT Customization, Prompt Engineering Evolution, Interaction-Level Stability, Agent Prompting vs. Conversational Prompting, Minimal vs. Maximal System Prompts` 


- **ChatGPT Customization Options Abound**: Members shared links to [ChatGPT's help documentation](https://help.openai.com/en/collections/3742473-chatgpt) which details **custom instructions**, a **custom GPT builder editor**, and instructions for **creating custom GPTs** (requires subscription).
- **Prompt Engineering: Template Optimization Dies, Iterative Task Design Lives**: Modern prompt engineering is evolving beyond static templates to **iterative task design**, focusing on **shaping assistant behavior** across conversations, as models co-engineer prompts.
   - The focus shifts from memorizing tricks to understanding how models *negotiate, stabilize, and shape tasks* over multiple turns.
- **Repeatability Redefined: Interaction-Level Stability Emerges**: Beyond surface prompt repeatability, the discussion explores the **re-instantiation of the same internal frame** by the model after constraints or mode shifts, revealing a new layer of repeatability.
   - This "carry-over structure" contributes to **interaction-level stability**, where the model maintains coherence despite detours.
- **Agent Prompting vs. Conversational Regime: Two Stability Mechanisms**: The conversation distinguishes between **agent prompting**, aimed at maximizing determinism with tight attractor basins, and the **conversational regime**, where behavioral shape is built interactively.
   - In agent prompting, topological templates are the paradigm, whereas interaction-level stability is an extra layer in co-engineered conversations.
- **Decoding 'Minimal' System Prompts: Directive Density vs. Token Size**: The definition of a 'minimal' system prompt shifts from token size to **directive density**, focusing on prompts that set guardrails and tone without prescribing a behavioral strategy.
   - Claude's long system prompts are considered *structurally minimal* as they constrain values and boundaries, not process or role execution, distinguishing them from agent-style prompts.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1445533616292630591)** (2 messages): 

> `Grok-4.1-Fast, Free Slug, deprecation` 


- **Grok-4.1-Fast Users Feel the Squeeze**: Users of **Grok-4.1-Fast** are urged to migrate to the free slug (`x-ai/grok-4.1-fast:free`) to continue using it for free.
   - The `x-ai/grok-4.1-fast` slug will start charging as of **December 3rd 2025**.
- **Grok-4.1-Fast Free Faces the Axe**: **Grok-4.1-Fast Free** (`x-ai/grok-4.1-fast:free`) will be deprecated <t:1764792000:R>.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1445640978579980298)** (4 messages): 

> `Falconz AI Security Platform, Red-teaming LLMs, Earning $100k within a week` 


- ****Falconz** Soars as Unified AI Security Platform**: A member introduced **Falconz**, a unified AI security and red-teaming platform designed to detect jailbreaks and prompt injections across multiple **LLM models** in real-time, available for testing on [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday/Falconzz_M.C.P_Hackathon).
   - The member is actively seeking feedback on its features, performance, and potential improvements, accompanied by a [demo video on YouTube](https://www.youtube.com/watch?v=wZ9RQjpoMYo).
- **Profit Sharing Scam exposed on Telegram**: A member offered to help the first **10 people** earn **$100k or more within a week**.
   - The catch is that you will have to *reimburse me 10% of your profits when you receive it*, they said, directing interested parties to their Telegram username **@Edward_Pryce1**.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1445506356122615818)** (213 messages🔥🔥): 

> `Amazon Nova Provider Error, Claude Deprecation, OpenRouter Model Fallback, MPU v2, x-ai/grok-4.1-fast` 


- **Amazon Nova Provider Experiences Errors**: A user reported receiving an error message *{"message":null}* when using the **Amazon Nova Provider**.
- **OpenRouter Offers Model Fallback Feature**: OpenRouter has a model fallback feature so your thing wouldn't just die completely, members are encouraged to use it to seamlessly transition if something gets dropped unexpectedly.
- **DeepSeek v3.2 is Not The Same as Previous Version**: The DeepSeek API has been updated, the previous DeepSeek v3.2 model was the "experimental" version, and this new one is "better", apparently.
- **OpenRouter Provides Payment Solutions for Chinese Institutions**: A researcher from China seeks guidance on setting up **institutional payments** with OpenRouter, requiring a formal contract/agreement and an official invoice for reimbursement, and another member pointed out you can pay with crypto instead.
- **Atlascloud responses are enclosed in deep thinking tags**: A member reported that [Atlascloud](https://atlascloud.ai) served an entire response enclosed in deep thinking tags, which some members agreed it does constantly, and are used to it.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1445624193566642176)** (12 messages🔥): 

> `OpenAI Garlic Model, DeepInfra Pricing Anomaly, Anthropic Acquires Bun` 


- **OpenAI Readies 'Garlic' Model**: A news article claimed that [OpenAI is readying a 'Garlic' AI model](https://www.newsbytesapp.com/news/science/openai-readies-garlic-ai-model-to-rival-google-gemini-3/story) to rival Google's Gemini 3.
   - Members reacted with amusement to the supposed model name, as seen in the [attached image](https://cdn.discordapp.com/attachments/1392278974222307469/1445624193361383484/image-5.webp?ex=6931aeb2&is=69305d32&hm=f4e0d58112b53996c13cc35e147fa08705703ae07a234b701642d66cd0d53e60&).
- **DeepInfra's Backwards Embedding Pricing**: Members noticed that [DeepInfra](https://deepinfra.com/) was offering its **4B embedding model** at a higher price (**2 cents**) than its **8B model** (**1 cent**).
   - The anomaly was highlighted with a [screenshot](https://cdn.discordapp.com/attachments/1392278974222307469/1445778910498521129/Screenshot_20251203-090815.png?ex=69319609&is=69304489&hm=5cd04243d1918794f50fb7dc7ed462ac90859051128b344b1950cf5582dc3591&), and it was noted that DeepInfra later changed the **8B** pricing that same day.
- **Anthropic Gobbles Up Bun**: Members excitedly shared news of [Anthropic's acquisition of Bun](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone) as **Claude Code** reached a **USD1B** milestone.
   - Bun's [website](https://bun.sh/) describes itself as *a fast all-in-one JavaScript runtime*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1445654564056600730)** (20 messages🔥): 

> `Local LLMs Use Cases, Context Switching on SM Sub Partition, CUDA Forum Activity Decline, PyTorch's Abstraction of CUDA` 


- **Local LLMs Protect Your Privacy**: Local LLMs are useful for people who care about **privacy** and don’t want their queries or sensitive info used as **training data** by an LLM provider.
- **Single Cycle Context Switching on SM**: Switching from one execution context to another on an **SM sub partition** has *no cost*, taking a single cycle because the execution context for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp as described in [Nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading).
- **CUDA Forum Traffic Plummets?**: A member noted the lack of activity in **CUDA** and **Cutlass** channels, as well as the **CUDA developer forum**, despite Nvidia's increased market cap, suggesting a shift in where developers seek help.
   - Another member mentioned that experts are occupied with work, making public discussions less optimal, while others retreat to small, private communities and use LLMs for instant reasoning and document skimming.
- **PyTorch Abstracts Away CUDA**: A member noted that **CUDA** is mostly a *black box* to many **ML researchers** and **SWEs** because frameworks like **PyTorch** have done a good job of abstracting **CUDA C/C++**.
   - ML and LLM traffic now mostly goes to **PyTorch forums**.
- **Foundation Model Training is Immense**: The R&D cost of these **foundation models** is immense, with the cost to train **Z-Image** published as **$628,000** by Tongyi Lab.
   - The member notes that *weights lifespan* is short and they're effectively burning millions of dollars on throwaway products.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

infinitejoy2934: I am able to get it now. thanks
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1445508975540961456)** (3 messages): 

> `Pytorch 2.9.1 Conv3D performance, CUDNN workaround` 


- **Conv3D Conundrum Cripples Current CUDA**: Users report that **Pytorch 2.9.1+cu128** has an issue where **conv3D** is extremely slow, regardless of **cuDNN** being enabled.
   - The same code runs fine in **2.8.0+cu128**.
- **Newer cuDNN Cures conv3D Catastrophe**: A member reports that this is a known issue and the workaround is to install a newer **cuDNN** from pypi.
   - The [issue is tracked on Github](https://github.com/pytorch/pytorch/issues/166643).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1445631793578512385)** (2 messages): 

> `Quantization Formats, INT v.s. FP` 


- **Study of Low-bit Quantization Formats Published**: A new paper titled "**INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats**" has been published and is available on [arXiv](https://arxiv.org/abs/2510.25602).
   - The research provides a comprehensive analysis of various **low-bit quantization formats**.
- **Pritam.ai posts Quantization Study**: Pritam.ai posted a link to a study of **INT vs FP** at URL [https://arxiv.org/abs/2512.02010pritam.ai](https://arxiv.org/abs/2512.02010pritam.ai).
   - Another link was posted at URL [https://arxiv.org/abs/2510.25602](https://arxiv.org/abs/2510.25602) referencing **INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1445809877644742746)** (2 messages): 

> `ML Performance Engineer, Voice AI Inference Platform, RAG Pipelines, AI Content Detection, Voice AI` 


- **Hathora Seeks ML Performance Engineer in NYC**: Hathora is hiring an **ML Performance Engineer** in NYC to build the fastest **voice AI inference platform** with a compensation of **$160-200k + equity**; experience with GPU programming or inference engine work is a plus, see [Hathora Notion](https://hathora.notion.site/ML-Performance-Engineer-2af894f6eff68092a13ef98556a9f944).
   - They are looking for someone to own their performance stack end-to-end from **kernel optimization** in their **vLLM + other inference engines** to **Docker & K8s** deployment.
- **Engineer Highlights Workflow Automation & LLM Integration**: An Engineer highlights experience building **pipelines** connecting **Slack, Notion, and internal APIs** which reduced response times by **60%**.
   - This engineer also brings expertise in **RAG Pipelines**, **AI Content Detection**, **Image AI**, **Voice AI**, and **Full Stack** development.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1445708770616279040)** (5 messages): 

> `Torch Compile Slowdown with Float 8, torchao and nn.Parameter, Custom Modules and Quantization` 


- ****Torch Compile Idles** with Float 8 Quantization**: When using **float 8 quantization** with `torch.compile` and `ncu` profiling, users are experiencing idling times of **10+ minutes** even after the model is compiled, specifically during the first 2-3 compilation and cudagraph warmup iterations.
   - The "constant subexpression elimination" pass of the inductor compiler is suspected as the culprit when freezing weights and folding them into the model graph.
- ****Torchao and nn.Parameters** clash due to filtering**: Users find that `torchao` **A16W8** and **A8W8** quantization cannot be applied to custom modules that use `nn.Parameter` for weights and `torch.einsum` in the forward pass, as weights remain in their original data type.
   - The `filter_fn` in `torchao.quantization.quant_api` specifically checks for `nn.Linear` instances, causing the quantization to fail for modules with `nn.Parameter`.
- **Solving Custom Module Quantization with **nn.Linear****: Users can bypass the `filter_fn` issue by using `nn.Linear` in their custom modules instead of `nn.Parameter`.
   - Initializing `nn.Linear` with the desired weight tensors allows `torchao` to correctly quantize the model.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1445571565038342185)** (3 messages): 

> `EleutherAI, MLSys conferences, ML4Health's career mentorship program` 


- **EleutherAI offers publishing help**: Members mentioned that [Eleuther AI](https://www.eleuther.ai/) has a **Publishing help channel** with some focus on endorsements.
   - No specific information was shared about the specifics of this channel.
- **MLSys conferences career mentorship program**: A member asked about career mentorship programs in **MLSys conferences**.
   - The member also mentioned taking part in **ML4Health's career mentorship program** and said *it was a pretty nice experience*.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1445557747650400387)** (3 messages): 

> `irl meetup, quartet paper, Dropbox coffee spot` 


- **Quartet paper author spotted**: A member mentioned that their colleagues are at the meetup, including Andrei, one of the main authors of [quartet](https://arxiv.org/abs/2505.14669).
- **Dropbox sponsors coffee spot**: A member mentioned they have a *kind of* **Dropbox coffee spot** since they are a sponsor, and invited others to chat.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1445771826738499644)** (3 messages): 

> `Bitsandbytes, Apple Silicon Support` 


- **Bitsandbytes Merges Apple Silicon Support!**: The **bitsandbytes** library merged the *"apple silicon support"* pull request, and the next release will contain the python/pytorch code backend (with some C++ bits) but no actual **Metal implementations**.
- **Apple Silicon Support Arrives with Caveats**: The pull request implementing Apple Silicon support will be advertised as being slow, according to the committer.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1445513479120359555)** (1 messages): 

> `Qwen3-Omni-30B-A3B-Instruct, S2S inference, Hathora playground` 


- **Qwen3-Omni-30B-A3B-Instruct makes Inference Speedy**: Members announced the deployment of **Qwen3-Omni-30B-A3B-Instruct** for fast **S2S inference**, see [the LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7401718431986987008/).
- **Test Qwen3-Omni in Hathora's Playground**: Users are invited to test **Qwen3-Omni** in [Hathora's playground](https://models.hathora.dev/model/qwen3-omni#form).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1445540412054962256)** (19 messages🔥): 

> `nvfp4_gemm leaderboard submissions, NVIDIA performance benchmarks` 


- **Submissions Galore Flood nvfp4_gemm Leaderboard**: Multiple users submitted performance results to the `nvfp4_gemm` leaderboard on NVIDIA, with timings ranging from **11.0 µs** to **65.3 µs**.
   - User <@1291326123182919753> achieved multiple runs at **11.0 µs** with submission IDs `120595`, `120601`, and `121065`.
- **New Personal Bests on NVIDIA**: Several members achieved personal bests on NVIDIA, including <@1191430895769485436> at **22.6 µs** (`119885`), <@772751219411517461> at **18.8 µs** (`120443`), and <@140482609422663680> at **56.8 µs** (`121056`).


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1445621050468728863)** (2 messages): 

> `Neurips Trip, Call Attendees, Call Time` 


- **Neurips Attendee Flies Out**: A member mentioned flying to **Neurips** and being available the next day.
- **Call Attendees Announced**: The member expects to be the only one speaking on the call, but noted that **Mart** might join.
- **Call Time to be Determined**: The member inquired about the time of the call.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1445842201555767480)** (1 messages): 

> `Matmul v2 Leaderboard Error, Submitting Kernel Error, input_generator Update` 


- **Matmul v2 Leaderboard submission fails!**: A new user reported receiving a `ValueError: too many values to unpack (expected 2)` error when submitting a kernel to the **matmul_v2 leaderboard**.
   - The user suspects that the `input_generator` was updated to return **3 values**, but the reference implementation in `reference.py` still unpacks only **2**, causing the failure.
- **Potential Mismatch in Input Generator and Reference Implementation**: The error suggests a potential issue with the **input_generator**, which might be returning three values instead of the expected two.
   - This discrepancy leads to a `ValueError` in the reference implementation where it attempts to unpack the input data, expecting only two values.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1445654753097945159)** (11 messages🔥): 

> `Multi-GPU CUDA Kernels, NCCL Repository, Qwen2.5-1.5B-Instruct Model Training, HF Accelerate with DeepSpeed Zero3, Context Parallel and Ulysses Parallel` 


- ****NCCL Repo: Multi-GPU Kernel Nirvana****: To learn multi-GPU CUDA kernels, the [NCCL repository examples](https://github.com/NVIDIA/nccl/tree/master/examples) are recommended as a starting point.
   - The NCCL (Nvidia Collective Communications Library) repo provides fundamental examples for understanding multi-GPU kernel implementations.
- ****Qwen2.5-1.5B Faces OOM Fate****: A user training the `Qwen2.5-1.5B-Instruct` model with a sequence length of **16384** and batch size of **5** on a g5.48xlarge instance (8 A10 GPUs) is running out of memory (OOM).
   - They are employing HF accelerate, deepspeed zero3, gradient checkpointing, Liger-kernel, and flash attention 2, with a fixed memory of **3.6GB** and activation memory exceeding **10GB**.
- ****Context Parallelism Emerges as Activation Alleviation****: One suggested way to further reduce activation memory is to use [context parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) or [Ulysses parallel](https://www.deepspeed.ai/tutorials/ds-sequence/) (DeepSpeed's version of CP).
   - However, it was noted that if the goal is to reach a particular global batch size, using gradient accumulation might be more efficient.
- ****Sequence Parallelism Saves the Day****: Sequence Parallelism (SP) is where you split each example over the sequence dimension to reduce the activation memory.
   - Check out the [torch docs](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) or the [HF docs](https://huggingface.co/docs/accelerate/en/concept_guides/context_parallelism) for more on Context Parallelism which reduces the tokens/gpu.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1445582750013980693)** (4 messages): 

> `Arxiv Papers, Hadamard Transform` 


- **Arxiv Paper Shared**: A member shared a link to an Arxiv paper: [https://arxiv.org/abs/2512.02010](https://arxiv.org/abs/2512.02010).
- **Hadamard Transform Improvements Paper**: A member shared a link to a Hugging Face papers page: [https://huggingface.co/papers/2512.00956](https://huggingface.co/papers/2512.00956) discussing improvements over **Hadamard Transform**.


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1445579653862391850)** (1 messages): 

> `Activation Offloading, fp8 Adam, Loss Masking, Pyllmq on PyPi` 


- **Activation Offloading Implemented**: A user implemented offloading for **residual activations** and other tricks for saving on **activation memory**.
   - The implementation includes better handling of **offloaded optimizer states** and initial support for **fp8** representation for **Adam first-order momentum**.
- **Training 7B Model on 16GB Card**: The user's code now supports pre-training/fine-tuning even a **7B model** on a **16GB card** with at least **64GB** of CPU-side RAM.
   - Scaling up, training/fine-tuning a **32B model** is possible on a **4x4090** server at about **3k tok/s** (**48% MFU**), requiring > **200GB** of pinned host memory for all the offloading.
- **Pyllmq Released on PyPi**: The user released the python wrapper on [PyPi](https://pypi.org/project/pyllmq/0.3.1/).
   - To try it out, simply `pip install pyllmq; pyllmq-tokenize --model qwen --dataset tiny-stories; pyllmq-train` and it should start fine-tuning **Qwen2.5-0.5B** on **tiny-stories**.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1445522030299713616)** (111 messages🔥🔥): 

> `GPU Mode TUI, Cutlass Version Issues, Reference Kernel Issues, NVFP4 and Scale Tensors, B200 GPU access` 


- **Popcorn CLI Gets a No-TUI Flag**: A member created a fork of **popcorn-cli** that allows a `--no-tui` flag to remove the **Terminal User Interface** and output the `stdout` of `print()` statements to help with debugging; the fork is available [on GitHub](https://github.com/Ryan-Rong-24/popcorn-cli).
   - A pull request was made to incorporate these changes into the main [gpu-mode/popcorn-cli](https://github.com/gpu-mode/popcorn-cli/pull/26) repository.
- **Cutlass Import Error Troubles Participants**: Some participants encountered an `ImportError: cannot import name 'pipeline_init_arrive'` error, potentially due to inconsistencies in the **Cutlass** versions across the runners; the issue was identified as some runners using **4.3.0** while others used the **dev** version.
   - One member suggested that a possible, though perhaps not entirely rules-abiding, workaround was to run `pip install` and upgrade **Cutlass** yourself within the submission.
- **Reference Kernel Generates Infs**: Participants reported that running the reference implementation locally produced all **Infs** when computed with the seed=1111, but this could be resolved by adjusting the range of the scale factors to **-1 to 1**.
   - The underlying cause was determined to be biased A/B values and negatively biased scales, and this [PR was merged](https://github.com/gpu-mode/reference-kernels/pull/84) to fix this issue.
- **Scale Tensors in CuTeDSL Analyzed**: A member shared a [blogpost](https://veitner.bearblog.dev/scale-tensor-construction-in-cutedsl/) analyzing the mathematical interpretation of scale tensors in **Blackwell** kernels for **NVFP4**, highlighting the similarity to Swizzling and the generality of the **CuTe Layout** algebra.
   - The member thanked Verda and Paul Chang for providing access to **B200** for making **Blackwell** programming more accessible.
- **New Hackathon Participants Ask About B200 Access**: A member who just joined the hackathon inquired about getting access to **B200** GPUs to test execution time before submitting their work.
   - Another member suggested pushing code through **popcorn-cli** or submitting through the **Discord bot** to test.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1445723978542092309)** (7 messages): 

> `Chunking, Jerky Movements, VLMs, Neural State Encoders` 


- **Alleviating Jerky Movements via Chunking**: Concerns were raised that **chunking** may result in jerky movements when deployed on hardware.
   - One member suggested training a higher level instruction **VLM** to generate detailed text instructions for shorter time periods, allowing the higher-level VLM decoder to operate at approximately **1 Hz**.
- **Neural State Encoders on Deck**: Members are testing some **neural state encoders**, beginning with simple **Conv** and **MLP** projections into 4 token-embeddings, using a history of **10** time steps (**10x14** state - 2x 6DoF + 2x Gripper).
   - The next step involves project cleanup and data generation for a **2-stage approach**.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1445512570445369414)** (143 messages🔥🔥): 

> `Kimi K2 models, Anthropic URL, File uploads, Roo code context, Kimi CLI` 


- **DeepSeek V3.2 Model's tool calling capabilities**: One user found that the **Deepseek v3.2 model** is a step up for agentic tasks but can only make **one tool call per turn**, sometimes ignores tool schemas, and occasionally fails tool calls by outputting it in `message.content` instead of `message.tool_calls`.
   - The user said that the **Deepseek v3.2 model** seems to need more tool call post-training to match other models like **kimi-k2-thinking**.
- **Discuss Black Friday Deals, GLM Deal**: Some users experienced issues with the **Black Friday deal** for Kimi; one said it only showed options to invite friends, and another said the **Black Friday deal** didn't work.
   - A user said it ends **Dec 12** and suggested starting a new chat ([https://www.kimi.com/user/agreement/black-friday](https://www.kimi.com/user/agreement/black-friday)) while another said that the **GLM deal** is just so cheap especially with the Black Friday deal.
- **DeepSeek's Target Audience Revealed**: A video was shared explaining how Chinese labs like **Deepseek** are targeting enterprise users, rather than normie consumers, link to [YouTube video](https://www.youtube.com/watch?v=u0n6wMnEYsk).
   - The key factor for enterprise users is the intelligence-to-price ratio, which is crucial for agentic tasks.
- **Mistral Overthrows Qwen at Company**: One user said that a company they know replaced **qwen 3 vl 4b** with **ministral 3 3b** yesterday, reporting better quality.
   - The reported plus points included a *lighter model (faster)* and being *able to attach more images at once*: **qwen3 vl 4b** could take **5 images max**, **ministral 3 3b** took upto **11 images** with similar error rates on a single **L4 GPU**.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1445871232179372177)** (1 messages): 

> `Hermes 4.3, ByteDance Seed 36B, Psyche network, Solana, Office hours` 


- **Hermes 4.3 packs a punch!**: Nous Research announced **Hermes 4.3** on **ByteDance Seed 36B**, the latest in their flagship **Hermes** series, offering performance equivalent to **Hermes 4 70B** at half the size.
   - This model was post-trained entirely on the **Psyche network** secured by **Solana**.
- **Psyche Training Outperforms Centralized Methods**: Nous Research details how they trained **Hermes 4.3** and how **Psyche** outperformed traditional, centralized training methods in [this blogpost](https://nousresearch.com/introducing-hermes-4-3/).
- **Psyche Team Hosts Office Hours**: The **Psyche** team is hosting office hours to discuss the platform.
   - The office hours are scheduled for tomorrow at **10AM PST** in [this Discord event](https://discord.gg/993UWRUE?event=1442995571173625888).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1445511286854123582)** (91 messages🔥🔥): 

> `DeepSeek V3.2 Speciale, GLM 4.6 models release, AI Bubble & Economy Collapse, Hermes 4.3 36B release, Subagents vs Skills` 


- **DeepSeek V3.2 Speciale leads Reasoning Bench**: The new **DeepSeek V3.2 Speciale Reasoning** model is performing well, *leading in reasoning benchmarks* as illustrated in the attached [image](https://cdn.discordapp.com/attachments/1149866623109439599/1445511286971437190/deep.JPG?ex=6931ee4b&is=69309ccb&hm=137a671dfe80ba0cb773df29a576e7c2c4731284970ef16bcb545ab249736dbc&).
- **GLM 4.6 models release soon**: Members are anticipating the release of **GLM 4.6** models, particularly **GLM 4.6 Air and Mini**, to fill the gap left by Mistral, and noted its been a month since they added the 5 private models in the GLM 4.6 collection on HF.
   - The **Mini** model is rumored to be a **20B-30B MoE** model.
- **AI Bubble Burst Dooms Economy**: Members debated the potential for an **AI bubble** to cause economic collapse, especially concerning the sunk costs in compute and salaries.
   - One member argued that the impact would be temporary and primarily affect the **US**, while another pointed out the interconnectedness of global economies through **USD** and oil trade, referencing [this YouTube Video](https://www.youtube.com/watch?v=K3qS345gAWI).
- **Hermes 4.3 36B surfaces online**: The **Hermes-4.3-36B** model surfaced, with [this HF link](https://huggingface.co/NousResearch/Hermes-4.3-36B🐈) provided.
   - One user asked *why 4.3?*, and it was answered that *a few iterations went by* and that the model would be available on **Nous API/chat** soon.
- **Subagents vs Skills debated**: Members discussed using **subagents** vs. **skills**, and it was noted that skills have made manual subagents less necessary.
   - Instead one can *define an agent for handling the requirements* which will automatically be called, only using its own prompt.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1445511447928115301)** (3 messages): 

> `NLP Economic Simulation Research, Hermes models in Godot, LLMs for market simulation, VendingBench Analysis` 


- **Godot Gets LLM Boost for 3D Market Simulator**: A member is developing a 3D simulation in **Godot** to model markets, agriculture, and logistics, and is evaluating whether **Hermes models** would be suitable for this type of application.
   - Another member suggested examining contemporary **NLP economic simulation research**, noting that while **LLMs** mimic human traits, they struggle with long horizon tasks like in **VendingBench**.
- **Hermes Shines in Grey/Black Market Modeling**: It was proposed that **Hermes**, with its low refusal rate and high steering, could model the behavior of grey/black markets.
   - Most other **LLMs** may refuse this and be unusable. 


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1445518905195106566)** (45 messages🔥): 

> `Eon's $4B Valuation, Gradium spinout from KyutaiLabs, OpenAI's 'Garlic' Model vs Gemini 3, Vertical AI vs Rollups, Lidar and LLMs` 


- **Elad Gil Funds Eon at $4B Valuation**: Elad Gil is leading a **$300 million Series round** via "Elad Gil & Co." for cloud data-management startup **Eon**, boosting its valuation to nearly **$4 billion** ([source](https://x.com/eladgil/status/1995919389879927018)).
   - The round size and the firm’s straightforward name have garnered enthusiasm from commenters.
- **Kyutai Spinoff 'Gradium' Stirs AI Scene**: KyutaiLabs quietly spun off its speech-AI team into **Gradium**, a new for-profit company, announcing a **$70M seed round** and initial voice products ([source](https://x.com/GradiumAI/status/1995826566543081700)).
   - Observers noted the significant overlap in staff and investors, drawing parallels to the **OpenAI transition** and prompting jokes about avoiding non-profit structures for product companies.
- **OpenAI Cooks Up 'Garlic' to Fight Gemini**: **OpenAI's** new model, 'Garlic', aims to rival **Google's Gemini 3**, with internal reports suggesting it outperforms **GPT-4.5** in coding and reasoning ([source](https://x.com/steph_palazzolo/status/1995882259195564062)).
   - Reactions to the quirky naming trend are mixed, with speculation on its impact on user adoption.
- **Vertical AI Owns Deep Workflows, Rollups Get Rolled**: Vertical AI companies like **Harvey**, **Abridge**, and **OpenEvidence** are winning by owning niche workflows, hoarding proprietary data, and pricing on outcomes, whereas thin wrappers are getting steamrolled ([source](https://x.com/bcsmithx/status/1996042921116934369)).
   - VCs are now chasing AI-enabled rollups of legacy services, even though history shows they usually wreck value; **Trace Cohen’s sheet of 150+ vertical AI startups** (worth ~$120B) is now the sector map.
- **Antithesis Stress-Tests AI Code with Jane Street**: **Antithesis** landed a **$105M Series A** led by **Jane Street** to stress-test AI-written code ([source](https://x.com/_sholtodouglas/status/1996297367776309359)).
   - The argument is that deterministic simulation testing will become essential to verify future AI-generated code, because trust-through-testing will make or break production AI systems.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1445533128427831438)** (8 messages🔥): 

> `Gradium, Bloom, Voice AI` 


- **Gradium garners $70M Seed**: Paris-based **Gradium** emerged from stealth after just **3 months** of work, securing **$70M** in seed funding led by **FirstMark & Eurazeo** to introduce production-ready transcription & synthesis APIs, detailed in [this article](https://xcancel.com/GradiumAI/status/1995826566543081700).
- **Bloom bursts onto the scene**: **Ray (@rincidium)** announced the launch of **Bloom**, touted as the *“world’s first on-brand AI,”* in [this viral post](https://xcancel.com/rincidium/status/1995946528343818656?s=46) which received over **360k views**.
   - Questions arose about features like **IG/Google ad creation**, the demo video's production, and initial user challenges such as **login stalls** and unclear branding-kit flow, all of which Ray addressed with promises of fixes and UX enhancements.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1445636941935804416)** (7 messages): 

> `Waymo, Mechanical Engineering, ML algorithms, AI alignment` 


- **Waymo cool for Aerospace Student**: An aerospace student with a focus on navigation and guidance finds **Waymo** especially interesting, with broader interests in autonomous robotics and **BCIs**.
- **Mechanical Engineering relevant to navigation**: A member suggested that mechanical engineering is highly relevant in navigation, especially for **masters programs**.
- **ML student seeks guidance**: A first-semester **ML** student requests advice on accelerating their learning, having covered **Python**, **Numpy**, **Pandas**, and basic ML algorithms.
- **AI alignment benchmarks requested**: A member asked for pointers to **AI alignment/safety** type benchmarks.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1445512952584212600)** (26 messages🔥): 

> `Interpretability of World Models, Generalization in Diffusion Models, Energy-Based Models vs. Diffusion Models, Linear RNNs vs. Attention` 


- **Seeking insight into World Model Interpretability**: Members wondered about work on **interpretability of world models**, suggesting extracting rules learned for mechanics like gravity and predicting the usefulness of data items for improving the world model.
   - They pointed to some [interesting recent papers](https://www.nature.com/articles/s41467-025-61309-9) and [another mildly interesting paper](https://arxiv.org/abs/2506.03719) on the topic, but felt both contributions should be known by most.
- **Diffusion Models Generalize Early!**: It was discussed that a paper demonstrates that the timepoint at which **generalization appears is early** in diffusion models, and that the author of the paper accepts the results.
   - It was further explained that this effect is probably more true for pixel diffusion than for latent diffusion because different data dims in pixel diffusion are so correlated, suggesting that a shifted noise schedule should be used for pixel diffusion.
- **Energy-Based Models Claim to Generalize Diffusion**: A [paper](https://arxiv.org/abs/2504.10612) claims to **generalize diffusion and energy-based models**, with the only drawback being a 2-3x increase in training time and support for all features diffusion supports.
   - A member expressed skepticism due to the need for **double backprop** to train, computing input gradients for inference, halving network depth for the same cost, and trickier conditioning control, not to mention potential for instability.
- **Linear RNNs Face Strongest Challenge Yet**: A member highlighted a [paper](https://arxiv.org/abs/1806.02296) as the strongest argument against the need for **linear RNNs with state tracking**.
   - They said this paper came from the same people who originally demonstrated the state tracking limitations of attention, but noted that inductive bias and trainability might still favor RNNs.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1445687954537185372)** (6 messages): 

> `SAEs for Interpretability, Cunningham's 2024 SAE paper, Sparse dictionary learning problem, polysemanticity and superposition` 


- **SAEs Gain Traction in Interpretability Research**: Members discussed Cunningham's **2024 paper** being widely cited as the initial application of **Sparse Autoencoders (SAEs)** for interpretability.
   - It was suggested that the motivation behind the paper is well explained in its introduction, particularly the third paragraph.
- **SAEs Equated to Sparse Dictionary Learning**: One member mentioned that someone recognized that a method being discussed for interpretability was similar to a **sparse dictionary learning problem**, leading to the use of relevant tools.
   - This approach addressed aspects like **polysemanticity and superposition** in the context of interpretability.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1445631694563704967)** (2 messages): 

> `Custom Filters in lm-evaluation-harness, Decontamination.py Inclusion, Adapting Multiple-Choice Tasks` 


- **Custom Filters Best Practices**: A user inquired about the best method for adding custom filters within the `lm-evaluation-harness` framework, specifically asking whether to extend existing **.py files** or create a new one and import it in `filters/__init__.py`.
- **Decontamination.py's status in `__init__.py`**: A user pointed out that `decontamination.py` is not referenced in `__init__.py` and asked if this was intentional.
- **Multiple-Choice Task Adaptation Stalled**: A user inquired about the status of adapting multiple-choice-style tasks for APIs that don't support logprobs, noting that [PR #2601](https://github.com/EleutherAI/lm-evaluation-harness/pull/2601) has stalled.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1445600322411036903)** (22 messages🔥): 

> `DGX Spark order, Agent Tool Validation & Self-Healing, YOLO Model P-R Curve Issues, AI Learning Resources (LLM, Agent AI, Langchain), TRL get_quantization_config usage` 


- **DGX Spark Purchased by User**: A member announced they have ordered a **DGX Spark**, with a photo attached ([image link](https://cdn.discordapp.com/attachments/879548962464493622/1445600322432270447/IMG_4170.jpg)).
- **Agents' Tool Validation and Self-Healing Abilities Explored**: A member questioned if **Agents** can *interpret, validate, and self-heal Tools* (e.g., shell scripts) when destructive or buggy.
   - Another user shared a [link to a Hugging Face dataset](https://huggingface.co/datasets/John6666/forum3/blob/main/agent_tool_validation_healing_1.md) indicating this capability may exist.
- **YOLO Model's High P-R Curve Causes Concern**: A new computer vision user reported a trained **YOLO model** for Chinese chess detection is running well but has a *really high P-R curve*.
   - Another member suggested trimming out the two classes which are *significantly higher* than others.
- **AI Course Recommendations Needed**: A backend developer asked for recommendations on the *best course* to learn **AI (LLM, Agent AI, Langchain)**, as they found agents particularly interesting after building a mental health chatbot using Langchain.
   - A member recommended the [Hugging Face LLMs course](https://huggingface.co/learn/llm-course/en/chapter1/1) and [this blog post](https://huggingface.co/blog/mlabonne/llm-course) as a starting point.
- **Seeking guidance on using get_quantization_config from TRL**: A member inquired about how to use the **get_quantization_config** function from the **TRL (Transformer Reinforcement Learning)** library.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

mkprke: Hey folks,
Today i am starting my first Ai agent course
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1445898668728127548)** (1 messages): 

> `Stochastic Parrot` 


- **Stochastic Parrot Under Fire**: A member posted a link to a research paper at [zenodo.org](https://zenodo.org/records/17803931) that might cause readers to stop believing in the **stochastic parrot**.
- **New research on stochastic parrots**: A new study has been published that might challenge the notion of language models as mere stochastic parrots.
   - The research is available on [Zenodo](https://zenodo.org/records/17803931).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1445650796271108258)** (3 messages): 

> `Ellora-Lora Recipes, BitterBot AI Agent, Traffic Spike` 


- **CodeLion releases Ellora-Lora Recipes**: CodeLion has released a new blog post about [Ellora-Lora Recipes](https://huggingface.co/blog/codelion/ellora-lora-recipes).
   - The blog post provides instructions and recipes for using **Ellora-Lora**.
- **BitterBot AI Agent Seeks Feedback**: An AI agent called [BitterBot](https://bitterbot.ai/) is seeking feedback on its progress.
   - The agent is described as a *work in progress* but has made *tremendous strides lately*.
- **BitterBot's Architecture Needs Enhancement**: The **BitterBot** system experienced a traffic spike of **7k users** which took the system down.
   - The team is working on *enhancing their architecture to support more users*.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1445869437050687541)** (1 messages): 

> `Perturbation-based attribution experiments, Deep vision models, Feature behavior` 


- **Features are not what you think, says blogpost**: A member wrote a blog post about how features behave in deep vision models after running some **perturbation-based attribution experiments**.
   - The blogpost can be found here: [Your Features Aren't What You Think](https://teendifferent.substack.com/p/your-features-arent-what-you-think).
- **Deep Dive into Deep Vision Models' Quirks**: Experiments reveal unexpected behaviors in deep vision models when subjected to perturbation-based attribution methods.
   - The author encourages feedback on their findings shared in the linked blog post, inviting the community to explore the nuanced feature dynamics.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1445750975477583994)** (5 messages): 

> `SFT Model Evaluation Error, OOM Error on Fine-tuning, GPU Memory Management` 


- **Troubleshooting SFT Model Evaluation Error**: A member encountered a `ValueError` during SFT model evaluation, specifically failing to find the task `lighteval|gsm8k|0|0` as part of [this tutorial](https://huggingface.co/learn/smol-course/unit1/4#exercise-3-fine-tuning-smollm3-with-sfttrainer).
   - No specific solution was found, but the error indicates an issue with task configuration or registration in the evaluation setup.
- **Taming Out-of-Memory Errors**: A user reported running into OutOfMemory (OOM) issues while fine-tuning **SmolLM3** with **SFTTrainer** on a local machine with a **16GB GPU**.
   - Suggestions included reducing the *r* value in **LoraConfig** and decreasing the *per_device_train_batch_size*, as well as restarting the Jupyter notebook kernel to ensure the GPU memory is free.
- **Bigger GPUs Solve Problems?**: One member reported improved results using a larger GPU, implying that a **16GB VRAM** setup was insufficient for the specific task.
   - They were *unsure what exactly made the small 16GB VRAM run fail*, but the problem went away when using more resources.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1445602245885300899)** (14 messages🔥): 

> `Pug Resource, Docker and Kubernetes basics, Beginner Github Repositories, Gemini CLI, Agents in CLI` 


- ****Newbies Nab Docker & Kubernetes Know-How****: Members were looking for resources for learning **Pug**, **Docker**, and **Kubernetes** basics, as well as links to beginner-friendly **GitHub** repositories for hands-on learning.
- ****Gemini CLI Agents Arrive Soon?****: A member inquired about the arrival of **agents in CLI** and expressed interest in adopting them, mentioning dissatisfaction with paid alternatives like **Claude**.
   - They referenced a [discussion form](https://link.to/discussion-form) and their comment about possible improvements.
- ****Neural Net Neurons Need Numerical Nurturing****: A user asked about the amount of data required to train a neural network, suggesting the use of *cursorsky.moo*.
- ****OpenHands Opens Opportunities On-Premise****: One member suggested using **OpenHands** with a local model, leading to a query about specific models and GPUs in use.
   - The original poster said they could easily spin up a **7B or 8B class model**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1445512052184584314)** (5 messages): 

> `Deepseek 3.2 Speciale, Distributed Compute` 


- **Deepseek 3.2 Speciale Questioned**: A member questioned *why not* use **Deepseek 3.2 Speciale**, linking to a [YouTube video on wavefunctions](https://www.youtube.com/watch?v=AgsJkd8SOHI).
   - Another member responded it was due to **RAM** limitations, preferring to keep a ~3gb model in **VRAM** constantly and use it for various simple tasks.
- **Distributed Compute & Research Coop Suggested**: In response to RAM limitations, a member suggested joining a **distributed compute & research coop**.
   - They claimed to *know of one*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1445828572357853438)** (14 messages🔥): 

> `Advent of Code segfault, List comprehensions bug, String processing in Mojo, splitlines vs split("\n"), Out of bounds memory access` 


- ****Mojo** Advent of Code Segfault Solved!**: A user encountered a segfault during Advent of Code when processing an empty line with `codepoint_slices`, leading to an out-of-bounds memory access:  `battery_joltages[len(battery_joltages)-1]`.
   - The user found the issue by using a debugger, determining that an empty list was being accessed out of bounds, and suggested *better error messages on debug builds*.
- **ASSERT Flag Helps Catch Scope Issues**: A user suggested using `-D ASSERT=all` to catch accidental references outside of scope, particularly for lists.
   - While it didn't immediately solve the segfault in this case, it was noted as a helpful debugging tool for similar issues.
- **`splitlines` and `split("\n")` Diverge in Behavior**: Users discussed the differing behavior between `splitlines()` and `split("\n")`, where one of them might strip trailing newlines, leading to different results when processing text files.
   - Switching to `splitlines` avoided the error because it didn't include the last empty line.
- **String Processing Methods Explored**: A user suggested that for ASCII strings, checking codepoints might be unnecessary, implying direct byte pointer manipulation could be used, also pointing out that `String`'s `getitem` treats the string as ascii/bytes.
   - Span was suggested as an alternative method as well.
- **AOC Solutions Welcomed in Dedicated Channel**: Users are encouraged to share their Advent of Code solutions in the advent of code channel.
   - It’s valuable to see how others approach the problems, especially as they become more performance-critical.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1445760205127417978)** (13 messages🔥): 

> `LLM Model Degradation, Aider Benchmarks with GGUFs, Claude Sonnet vs GPT-5, Gemini 2.5 Degradation` 


- **LLM Models Degrading with Aider?**: Members questioned whether newer **LLM Models** like **Claude Sonnet/Haiku 4.5** and **GPT-5**, when paired with **Aider**, have degraded in performance compared to older models.
   - One user expressed that **Claude-haiku-4.5** keeps forgetting to modify files with `/code` and ignores explicitly stated instructions in `todo ai` comments.
- **Older Gemini 2.5 Degraded Too?**: A member reported that older models, especially **Gemini 2.5**, have also degraded, potentially due to models being tuned down to handle increased workload, saying that being 'rude' worked well with gemini most of the time, but *it's no where near quality from before the summer IMO*.
   - Another member agreed, noting that there are *several reports of it*.
- **Benchmark Craving: Benchmarks Needed to Validate LLM Performance**: A member emphasized the need for benchmarks to validate performance claims, citing that *human memory and expectations are pretty crap sometimes*.
   - Another user noted that even though leaderboards say **GPT-5** is on top, **Claude Sonnet 3.7** was producing better results with Aider for their use cases.
- **GGUF Aider Benchmark Guidance**: A member inquired about a guide on how to run **aider benchmarks with GGUFs**.
   - Another member pointed out that there is documentation on how to run the benchmark vs an API, requiring setting up an API server with llama.cpp.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1445900590516273277)** (1 messages): 

> `MCP Apps SDK, Open Source Libraries, Cross-Platform UI` 


- **MCP Apps SDK goes Open Source!**: General Intelligence Labs open-sourced [mcp-apps-sdk](https://github.com/General-Intelligence-Labs/mcp-apps-sdk), enabling **MCP-powered apps with UI** to run across various platforms.
   - Developers can now embed apps designed for **ChatGPT** into their own chatbots, assistants, or AI platforms and test them locally.
- **X post unveils SDK motivation**: An X post ([link](https://x.com/helloxalia/status/1796319442863866351?s=20)) explains the reasons behind building the open-source **MCP Apps SDK**.
   - The post details how developers can embed apps designed for **ChatGPT** into their own chatbots, assistants, or AI platforms and test them locally.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://arxiv.org/abs/2511.22074
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1445515200726958140)** (10 messages🔥): 

> `Prompt Security, Custom DSPy OutputFields, Pydantic integration with DSPy, Structured outputs` 


- **Prompt Security: Security at the Prompting Layer**: A member discussed the difficulty of achieving security at the prompting layer, suggesting that prompt-based "do not do this" statements are easily bypassed by attackers, and instead, to guard against baseline attacks by including examples in training datasets to guide the optimizer.
   - They propose guardrails type security measures, using specific models and invocations to check for malicious prompts, or model provider rejections.
- **Custom DSPy OutputFields: Implementing Structured Outputs**: A member inquired about custom DSPy OutputFields and whether Pydantic is the best approach, while another member mentioned they are working on a custom gemini/nanobanana image type as an output field.
   - The discussion involved generating text/json/structured output, questioning whether DSPy has its own implementation, and noting that they might have migrated.
- **DSPy uses Pydantic BaseModel under the hood**: It was clarified that DSPy uses `BaseModel` under the hood for validation and that the default `ChatAdapter` and `JSONAdapter` perform type validation as the LLM sends its output back.
   - A minimal example was provided to illustrate how to define a signature that takes in a Pydantic model, showcasing how DSPy can generate structured outputs with any LLM, a [code snippet](https://gist.github.com/prrao84/1fc7e17b49707f1346c5702525971f41).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1445511346320965633)** (12 messages🔥): 

> `Chatmode Feature, AI agent advertisement, Account Suspensions, RAG pipelines` 


- **Chatmode makes a Comeback**: Users discuss the return of **Chat Mode** in the platform; others suggest that using a random instance of **Qwen** or **DeepSeek** could achieve the same thing.
   - One user confirms its availability under the *'more'* section.
- **AI Engineer advertises Expertise in Agent Building**: An AI engineer posted an advertisement of their expertise in building **autonomous AI agents** and **multi-agent systems**, mentioning capabilities such as research, data-gathering, task automation, delegation, collaboration, and planning.
   - The advertisement also lists specific technologies and tools like **JS/TS**, **Next.js / Vue**, **Go / Rust**, **Python**, **Langraph**, **AutoGen**, **ReAct**, **CrewAI**, **DeepSeek**, **OpenAI**, **Claude**, **Hugging Face**, and various APIs.
- **Account Suspensions: Referral causes Suspicion**: A member asked why giving referrals to several people is causing their account to be suspended.
   - No further information or solutions were provided in the messages.
- **AI engineer specializes in RAG pipelines**: One engineer specializes in **RAG pipelines**, boasting *hybrid search* and *custom retrieval* for accurate, context-aware responses in production.
   - The engineer also lists expertise in **AI content detection**, **image AI**, and **Voice AI**, including the development of moderation tools, tagging pipelines, and personalized voice assistants.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1445562215083147265)** (6 messages): 

> `Fixing test failures in tinygrad, Performance improvements using shrink vs indexing, RMSNorm usage clarification` 


- **Fix Almost Ready for Failing Tinygrad Tests**: A member reported failing tests with `CPU=1 PYTHONPATH="." pytest -n 12`, specifically `test/test_tiny.py TestTiny.test_beam` and others, with complete logs provided.
   - Another member mentioned a [pull request](https://github.com/tinygrad/tinygrad/pull/13553) that *almost* fixes the issues.
- **Shrink is blazingly fast for Indexing Tensors**: A member suggested that using `Tensor.shrink((None, (0, input_size)))` is faster than `obs[:, :input_size]` when working with tensors.
   - They also noted bumping `Variable` `vmin` to 2 to avoid errors, but were puzzled why using `Variable` made the code 5x slower (16.61M vs 81.9M SPS).
- **RMSNorm Parameter Puzzlement**: A member advised reviewing the source code of `RMSNorm(dim=-1)` to ensure it behaves as expected.
   - This guidance suggests a potential misunderstanding or misconfiguration in how `RMSNorm` is being used.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1445846253018546307)** (5 messages): 

> `MCP Security Risks, Security risks associated with MCP, MCP-specific security` 


- **Redditors Debate MCP Security Risks**: A user asked for feedback on their perspective regarding security risks associated with **MCP** with a link to a [reddit thread](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/).
   - Another member responded with a link to a blog post about **MCP-specific security items**, calling it a *great resource*: [den.dev/blog/security-rakes-mcp/](https://den.dev/blog/security-rakes-mcp/).
- **Another MCP Security resource**: Here's another resource [MCP Security @ Reddit Thread](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/)


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1445792345235918869)** (1 messages): 

> `Tool Validation, Server-Side Validation` 


- **Sampling Without Tools Requires Server Validation**: A member inquired whether the server should validate if sampling occurs without a tool, given the absence of a tool proving its existence.
   - The question revolves around ensuring that the process is correctly validated server-side when sampling methods are employed and the expected tool or proof of its existence is missing.
- **Server Validation Crucial for Tool-less Sampling**: The discussion highlighted the importance of server-side validation when sampling is performed without the presence of a validating tool.
   - It ensures that the sampling process adheres to required protocols and standards even in the absence of direct tool validation.


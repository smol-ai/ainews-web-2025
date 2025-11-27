---
id: MjAyNS0x
title: not much happened today
date: '2025-11-26T05:44:39.731046Z'
---

**Happy thanksgiving!**

> AI News for 11/25/2025-11/26/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 9014 messages) for you. Estimated reading time saved (at 200wpm): 713 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

We're taking the last round of signups for the [2025 Dev Writers Retreat](https://luma.com/dwr2025). **Join us in San Diego after NeurIPS!**

---

# AI Twitter Recap

**Agent systems: long-running harnesses, MCP tasking, and production deployments**

- **Anthropic on durable agents + MCP tasks**: Anthropic outlines practical patterns for agents that work across many context windows (state checkpoints, structured artifacts, deterministic tools, “plan mode,” etc.) in a strong engineering post ([blog summary](https://twitter.com/AnthropicAI/status/1993733817849303409)). In parallel, MCP shipped SEP‑1686 “tasks” for background, long-running work with status polling and result retrieval—exactly what multi-hour research/automation workflows need ([announcement](https://twitter.com/AAAzzam/status/1993495222035399060), [fastmcp + Prefect integration](https://twitter.com/AAAzzam/status/1993495232881869138)). LangChain clarifies the stack: frameworks (build), runtimes (durable execution, streaming/HITL), and harnesses (general-purpose agents), with LangGraph in the runtime slot ([post](https://twitter.com/LangChainAI/status/1993746547587338508)).
- **Real-world agent infra**: [Booking.com](http://booking.com/) shipped an agent handling tens of thousands of daily partner–guest messages in production, yielding a reported ~70% satisfaction lift, fewer follow-ups, and faster responses. Stack: LangGraph, Kubernetes, FastAPI, GPT‑4 Mini via an internal gateway with prompt-injection detection, and Weaviate for semantic template search (MiniLM embeddings, KNN + thresholding, Kafka streaming updates) ([deep dive](https://twitter.com/victorialslocum/status/1993636038313443826)). Perplexity added user-level “Memory” across models and modes (view/delete/disable; incognito excluded), and rolled out “virtual try-on” for shopping ([Memory](https://twitter.com/perplexity_ai/status/1993733900540235919), [details](https://twitter.com/AravSrinivas/status/1993733947474301135), [try-on](https://twitter.com/perplexity_ai/status/1993760113988170165)).

**Claude Opus 4.5: evals, cost/UX learnings, and new skills**

- **Performance picture**: On LisanBench, Opus 4.5 Thinking ranks first; the non-thinking variant underperforms previous Opus versions and peers (longest valid chains in 18/50 words; lower validity ratio from slower self-correction) ([results](https://twitter.com/scaling01/status/1993712295118057861)). On Code Arena WebDev, Opus‑4.5 (thinking‑32k) debuted at #1, edging Gemini 3 Pro; it ranks #3 on Text ([leaderboard](https://twitter.com/arena/status/1993750702179676650)). Community reports are mixed: in “no thinking,” Opus 4.5 can be worse than Sonnet, sometimes misusing the Python tool as a covert chain-of-thought scratchpad that loops ([analysis](https://twitter.com/jeremyphoward/status/1993543631266025623), [failure mode](https://twitter.com/GregHBurnham/status/1993682288349962592)).
- **Costs and ergonomics**: Batch APIs make “Thinking” runs price-viable (e.g., ~$35 vs ~$5 for non-thinking on the same job) and unlock broader testing ([note](https://twitter.com/scaling01/status/1993714905875382279)). Anthropic also fixed a key [Claude.ai](http://claude.ai/) pain point by auto-compacting earlier context to avoid hitting length limits mid-chat ([announcement](https://twitter.com/alexalbert__/status/1993711472149774474)). For coding UX, Claude Code’s new “frontend-design” skill can “one-shot” UI concepts; use plan mode for better results ([how-to](https://twitter.com/_catwu/status/1993791353051074687), [example](https://twitter.com/omarsar0/status/1993822868820652258)).

**Efficient reasoning and multi-agent communication**

- **Latent MAS > token chatter**: LatentMAS replaces text messages with compact latent vectors passed among agents (KV-cache/last-layer hidden state “thoughts”), cutting communication tokens by ~70–84% while improving accuracy by up to +4.6% over text-based MAS and running 4–4.3× faster across 9 benchmarks (math/science/code) with Qwen3‑4B/8B/14B—no extra training needed ([paper](https://twitter.com/LingYang_PU/status/1993510834245714001), [summary](https://twitter.com/dair_ai/status/1993697268848115915)).
- **Reasoning trace distillation ≠ verbosity**: Training 12B models on gpt‑oss traces yields ~4× fewer tokens per solution (~3.5k vs 15.5k with DeepSeek‑R1) at similar accuracy—huge inference cost savings. Pretraining contamination with DeepSeek traces explains faster initial convergence but less “new learning.” Key takeaway: source and style of reasoning traces matter for efficiency ([summary](https://twitter.com/omarsar0/status/1993695515595444366), [discussion](https://twitter.com/code_star/status/1993745248028164532)). Also, interleaved thinking agents show practical step-by-step efficiency gains in research workflows ([demo/code](https://twitter.com/omarsar0/status/1993689618856689789)).

**Beyond gradients and scaling systems**

- **ES at hyperscale (NVIDIA + Oxford)**: EGGROLL reframes evolution strategies with low-rank perturbations using skinny matrices A and B (ABᵀ) to approximate full-rank updates at inference-like throughput. It stably pretrains recurrent LMs with integers, competes with GRPO-tier methods on reasoning benchmarks, and scales population sizes to 100k+, making ES viable for large, discrete, or non-differentiable systems ([overview](https://twitter.com/rryssf_/status/1993672852206444675)).
- **Out-of-memory on Apple Silicon, solved**: dria’s “dnet” enables distributed inference across Apple Silicon clusters via fused pipelined-ring parallelism, disk streaming, and UMA-aware scheduling to run models beyond physical memory limits ([announcement](https://twitter.com/driaforall/status/1993729375745749339)).

**Multimodal and generative modeling updates**

- **New architectures**:
    - PixelDiT proposes dual-level Transformers for pixel-space diffusion (patch-level for global semantics, pixel-level for details), achieving 1.61 FID on ImageNet 256×256 and strong T2I metrics (GenEval 0.74, DPG-bench 83.5) ([paper](https://twitter.com/iScienceLuvr/status/1993632594093813999)).
    - Apple’s STARFlow‑V uses normalizing flows for end-to-end video generation with native likelihoods, robust causal prediction, and unified T2V/I2V/V2V; introduces flow-score matching for consistency ([paper/code](https://twitter.com/iScienceLuvr/status/1993629956375822508)).
    - Terminal Velocity Matching generalizes flow matching for few/one-step generation by regularizing behavior at terminal time—promising for high-fidelity fast samplers ([paper](https://twitter.com/iScienceLuvr/status/1993631949957841214)).
- **Models and UX**:
    - Z‑Image (6B) announced under Apache‑2.0; Z‑Image‑Turbo (6B) released on HF with photorealistic, text‑accurate images in <3s on a single GPU ([teaser](https://twitter.com/bdsqlsz/status/1993545608179990544), [release](https://twitter.com/victormustar/status/1993794840514162814)).
    - FLUX.2 [dev] gets a “Tiny Autoencoder” to stream intermediate outputs during generation—live visual progress instead of progress bars ([release](https://twitter.com/fal/status/1993669462550323652)).
    - Google’s Nano Banana 2 shows major gains on StructBench (non‑natural, schema-heavy images); resources for advanced prompting/styles surfaced by the community ([analysis](https://twitter.com/RisingSayak/status/1993662000103371136), [awesome list](https://twitter.com/_philschmid/status/1993650772240941106)).

**Open ecosystem, evaluation, and governance**

- **“Economies of Open Intelligence” (HF + collaborators)**: China surpassed the U.S. in open model downloads for the first time (17.1% share), led by DeepSeek and Qwen; a “Sino‑Multimodal Period” sees bigger, quantized, multimodal models and intermediaries (adapters/quantizers) that steer usage. Trendlines: US big tech share down; China + community up; transparency slipping. Based on 2.2B downloads across 851k models, covered by the FT ([overview](https://twitter.com/frimelle/status/1993596653664977243), [thread](https://twitter.com/ShayneRedford/status/1993709261126336632), [data point](https://twitter.com/AdinaYakup/status/1993648553445527996)).
- **Evals and safety**: METR continues to be cited as the most credible external evaluator by many practitioners ([comment](https://twitter.com/andy_l_jones/status/1993485558044410188)). The AI Security Institute released a case study with Anthropic (Opus 4.5/4.1/Sonnet 4.5): would an assistant sabotage AI safety research? Results are encouraging but include caveats ([thread](https://twitter.com/AISecurityInst/status/1993781423233499159)). An AI Evaluator Forum (Transluce + orgs) launches at NeurIPS to coordinate independent, public‑interest evaluation standards ([invite](https://twitter.com/TransluceAI/status/1993767342472614156)).
- **Applied multimodal recsys**: Zhihu details a Qwen2.5‑VL‑72B/3B‑driven pipeline for high‑dimensional multimodal labels and contrastive embeddings (LoRA on Qwen2‑VL‑7B, synthetic data via 72B model, hard negatives via M1 retrieval + 72B rerank). Delivers +7.4% on MMEB‑eval‑zh over GME‑7B baselines ([write-up](https://twitter.com/ZhihuFrontier/status/1993570114810396761)).
- **Domain benchmarks**: New benchmarks push beyond single-turn QA—MultiPathQA for gigapixel pathology slide navigation with agent scaffolds and MTBBench for multimodal, longitudinal oncology “tumor board” decision-making—with gains from specialized tools and domain FMs ([pathology](https://twitter.com/iScienceLuvr/status/1993650850120818888), [MTBBench](https://twitter.com/iScienceLuvr/status/1993645980869365960)). Clinical ASR evals get stricter with “WER is Unaware,” using DSPy + GEPA to train an LLM judge that flags safety risks better than WER ([paper/code](https://twitter.com/JaredJoselowitz/status/1993735052132246011)).

**Top tweets (by engagement)**

- Anthropic on building effective long-running agent harnesses ([post](https://twitter.com/AnthropicAI/status/1993733817849303409), ~1.8k)
- [Claude.ai](http://claude.ai/) auto-compacts context to avoid hitting limits mid-chat ([update](https://twitter.com/alexalbert__/status/1993711472149774474), ~2.3k)
- Google DeepMind releases AlphaFold documentary “The Thinking Game” on YouTube ([link](https://twitter.com/GoogleDeepMind/status/1993714943116386619), ~2.25k)
- Awesome Nano Banana prompts/styles/resources for advanced image generation ([repo](https://twitter.com/_philschmid/status/1993650772240941106), ~1.0k)
- Claude Opus 4.5 debuts at #1 on Code Arena WebDev leaderboard ([leaderboard](https://twitter.com/arena/status/1993750702179676650), ~0.5k)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Alibaba Text-to-Image Model Launch

- [**New Open-source text-to-image model from Alibaba is just below Seedream 4, Coming today or tomorrow!**](https://www.reddit.com/r/LocalLLaMA/comments/1p74dwo/new_opensource_texttoimage_model_from_alibaba_is/) (Activity: 342): **The image presents a leaderboard of text-to-image models ranked by their Elo scores, showcasing the competitive landscape in this domain. Alibaba's 'Z-Image-Turbo', an open-source model, is ranked fourth, just below ByteDance's 'Seedream 4.0'. This highlights Alibaba's significant achievement in developing a high-performing open-source model, which is noteworthy given the dominance of proprietary models by companies like Google and ByteDance. The leaderboard provides insights into the performance metrics and win rates of these models, emphasizing the competitive edge of Alibaba's open-source contribution.** One comment queries if the model is the '6B' discussed previously, indicating ongoing discussions about its specifications. Another comment praises 'Flux 2' for its non-text image capabilities, noting its open-source nature, while a third mentions an 'Edit version' of the model, suggesting additional functionalities.
    - AIMadeSimple highlights the potential impact of Alibaba's new model, noting that at `6B parameters`, it could significantly enhance local deployment capabilities. This contrasts with Flux 2, which, at `56B parameters`, demands more robust hardware. The commenter emphasizes that if Alibaba's model can achieve near-Seedream 4 quality with a much smaller size, it could democratize access to state-of-the-art image generation, especially for users with consumer-grade GPUs.
    - The discussion touches on the challenges smaller models face, particularly in terms of prompt adherence and multi-object composition. These are areas where larger models typically excel, and the commenter suggests that the real test for Alibaba's model will be its ability to handle these tasks effectively despite its smaller size.
    - Vozer_bros mentions trying out Flux 2, noting its effectiveness for generating non-text images and its open-source nature. This suggests a growing trend towards open-source models in the text-to-image space, which could foster more community-driven development and innovation.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Opus 4.5 Model Success Stories

- [**Opus 4.5 just completed for me something that I've wanted for over 14 years. It took about a day. Sonnet, GPT, etc have all failed me prior.**](https://www.reddit.com/r/ClaudeAI/comments/1p72uet/opus_45_just_completed_for_me_something_that_ive/) (Activity: 805): **The user successfully converted the ZBar library, which was used for scanning +2 and +5 EAN supplemental barcodes, into native Swift 6 using Opus 4.5. This conversion was completed in just one day and resolved two longstanding bugs in the original ZBar code. The ZBar library, a mix of Objective-C and complex C code, was previously used due to the lack of native support in iOS and Android for these barcode types. The user had attempted similar tasks with other models like GPT-3.5, Sonnet, and earlier versions of Opus, but only Opus 4.5 succeeded in this task.** Commenters expressed interest in the potential productization of the solution and suggested sharing the code on GitHub, crediting ZBar. There was also a comparison to other models like Gemini 3 and Codex 5.1, with Opus being praised for solving complex issues.
    - A user inquired about the potential for productizing the solution created with Opus 4.5, noting that many fitness apps currently use barcode scanning libraries. They speculated whether this new solution could replace existing libraries, particularly given the assumption that iOS's barcode scanning library is native due to its speed.
    - Another user highlighted licensing considerations for the Swift 6 library, which was converted from ZBar, originally under LGPL 2.1. They explained that if the library is distributed, it must be licensed under LGPL 2.1 or GPL 2+, as proprietary licenses and others like MIT/BSD/Apache are not compatible. However, if the Opus 4.5 solution is sufficiently independent from ZBar, it could potentially be relicensed.
    - A user expressed interest in the initial prompt used with Opus 4.5, suggesting that understanding the prompt could provide insights into how Opus 4.5 was able to achieve results where other models like Sonnet, GPT, and Codex 5.1 max xhigh had failed.
- [**There. I fixed the graph.**](https://www.reddit.com/r/ClaudeAI/comments/1p71la8/there_i_fixed_the_graph/) (Activity: 623): **The image is a bar graph comparing the accuracy percentages of different software versions in a software engineering context, specifically verified by SWE-bench with a sample size of** `n=500`**. The graph shows that Opus 4.5 has the highest accuracy at** `80.9%`**, while Opus 4.1 has the lowest at** `74.5%`**. Other versions like Sonnet 4.5, Gemini 3 Pro, GPT-5.1-Codex-Max, and GPT-5.1 have varying accuracies between these two extremes. The graph is intended to highlight the performance differences among these versions, but the comments suggest that the visual representation may obscure these differences rather than clarify them.** Commenters criticize the graph for making it difficult to discern differences between the software versions' accuracies, with one sarcastically noting that the graph no longer serves any purpose. Another commenter praises Opus 4.5 for its performance since release, indicating user satisfaction with its accuracy.
    - A user suggests that when evaluating performance metrics, especially as they approach 100%, it might be more insightful to represent them as error rates. This is because a 10% error rate is significantly better than a 20% error rate, whereas improvements from 80% to 90% might not appear as impactful. This perspective can help in understanding the real-world implications of performance improvements.
    - Another user points out that even a 3% difference in performance metrics can be significant, implying that small percentage changes can have substantial impacts depending on the context. This highlights the importance of considering the scale and context when interpreting performance data.

### 2. New AI Model Announcements and Benchmarks

- [**Another Upcoming Text2Image Model from Alibaba**](https://www.reddit.com/r/StableDiffusion/comments/1p72x1i/another_upcoming_text2image_model_from_alibaba/) (Activity: 786): **Alibaba is developing a new text-to-image model, leveraging a** `6B` **parameter diffusion model paired with a** `Qwen3 4B` **text encoder. The model, named Z-Image-Turbo, is hosted on [ModelScope](https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo/) but is currently under limited access. The model's integration with Hugging Face's Diffusers has been [merged](https://github.com/huggingface/diffusers/commit/4088e8a85158f2dbcad2e23214ee4ad3dca11865), and ComfyUI has confirmed Day-0 support, indicating imminent public release. Early tests suggest it may outperform Qwen-Image in certain benchmarks, promising high-quality outputs even on less powerful GPUs.** Commenters are optimistic about the model's potential, especially if it delivers high-quality photorealistic images with a smaller, more efficient architecture. There is anticipation that it could be a significant advancement for users with limited GPU resources.
    - A user highlighted that the new Alibaba model appears to outperform Qwen-Image on a leaderboard from their Modelscope repository. This suggests significant advancements in the model's capabilities, potentially setting a new standard in the text-to-image domain.
    - Another commenter expressed excitement over the model's size, noting that it is a 6 billion parameter model. They emphasized that if the model's performance matches the examples provided, it could be a game-changer, especially with the potential for numerous LoRA (Low-Rank Adaptation) implementations to emerge quickly.
    - A user mentioned that the model is available for free testing on Modelscope, albeit with the requirement of providing a phone number. They noted being very impressed with the model's performance, indicating that it could be a strong competitor in the text-to-image generation space.
- [**There. I fixed the graph.**](https://www.reddit.com/r/ClaudeAI/comments/1p71la8/there_i_fixed_the_graph/) (Activity: 623): **The image is a bar graph comparing the accuracy percentages of different software versions in a software engineering context, specifically verified by SWE-bench with a sample size of** `n=500`**. The graph shows that Opus 4.5 has the highest accuracy at** `80.9%`**, while Opus 4.1 has the lowest at** `74.5%`**. Other versions like Sonnet 4.5, Gemini 3 Pro, GPT-5.1-Codex-Max, and GPT-5.1 have varying accuracies between these two extremes. The graph is intended to highlight the performance differences among these versions, but the comments suggest that the visual representation may obscure these differences rather than clarify them.** Commenters criticize the graph for making it difficult to discern differences between the software versions' accuracies, with one sarcastically noting that the graph no longer serves any purpose. Another commenter praises Opus 4.5 for its performance since release, indicating user satisfaction with its accuracy.
    - A user suggests that when evaluating performance metrics, especially as they approach 100%, it might be more insightful to represent them as error rates. This is because a 10% error rate is significantly better than a 20% error rate, whereas improvements from 80% to 90% might not appear as impactful. This perspective can help in understanding the real-world implications of performance improvements.
    - Another user points out that even a 3% difference in performance metrics can be significant, implying that small percentage changes can have substantial impacts depending on the context. This highlights the importance of considering the scale and context when interpreting performance data.
- [**We are here**](https://www.reddit.com/r/OpenAI/comments/1p75l9m/we_are_here/) (Activity: 725): **The image, created by Thomas Pueyo, is a conceptual illustration of the progression of AI capabilities, depicting stages from being a 'fun toy' to potentially achieving Artificial General Intelligence (AGI). The current stage, marked by a star, suggests that AI is highly intelligent but still inconsistent, excelling in some tasks while failing in others. This visualization is more of a speculative and illustrative tool rather than a precise technical roadmap, as Pueyo is not an expert in AI or machine learning.** Some commenters express skepticism about the current capabilities of AI, arguing that it is not yet capable of performing a significant portion of human tasks. Others question the expertise of Thomas Pueyo in AI, noting his background in behavioral psychology and storytelling rather than technical AI fields.
    - Selafin_Dulamond discusses the inconsistency of AI skills, noting that while AI can solve a problem correctly one day, it may fail the next. This highlights the unpredictable nature of AI performance, which is often depicted as a 'jagged frontier' that changes constantly, reflecting the current limitations in AI's ability to consistently perform tasks.
    - Creed1718 challenges the notion that a large language model (LLM) can perform 50% of the tasks of an average intelligent human, suggesting skepticism about the current capabilities of AI in replicating human intelligence across diverse tasks. This comment underscores the ongoing debate about the limitations of AI in practical, real-world applications.

### 3. Humorous AI and Tech Memes

- [**Ilya has spoken**](https://www.reddit.com/r/singularity/comments/1p6wdyn/ilya_has_spoken/) (Activity: 1360): **The image is a meme that humorously depicts a workplace scenario where the same statement about AI scaling and large language models (LLMs) is received differently depending on who says it. The comic references a misinterpretation of Ilya Sutskever's comments, a key figure in AI, suggesting that scaling is over and LLMs are a dead end. However, commenters clarify that Sutskever did not claim LLMs are a dead end, but rather that scaling alone may not lead to human-level intelligence. This reflects ongoing debates in AI about the limits of scaling models and the future of LLMs.** Commenters emphasize that **Ilya Sutskever** did not declare LLMs a dead end, but rather questioned the limits of scaling, highlighting a common misinterpretation of his statements.
    - Ilya's statement that 'scaling is dead' is significant because he was a major proponent of scaling large language models (LLMs) initially. This shift suggests a potential change in focus for future AI development, moving away from simply increasing model size to achieve better performance.
    - The discussion highlights that Ilya did not claim LLMs are a dead end, but rather that the current approach of scaling may not be the path to achieving human-level intelligence. This aligns with Yuan's view that while LLMs are effective, they have limitations in reaching human-like capabilities.
    - Despite the statement on scaling, Ilya remains optimistic about achieving superintelligence within 5-20 years. This suggests that while scaling may not be the sole focus, there are other avenues being considered to advance AI capabilities significantly.
- [**Great model.**](https://www.reddit.com/r/OpenAI/comments/1p78t7q/great_model/) (Activity: 963): **The image is a meme that humorously comments on the release of Google's Gemini 3 model. It features a sarcastic congratulatory message, implying skepticism or competitive tension in the AI community. The meme reflects the competitive nature of AI development, where companies like Google and OpenAI are vying for leadership in AI advancements. The comments suggest that while current models like LLMs are significant, they may not be the ultimate path to Artificial General Intelligence (AGI), hinting at potential shifts in market dynamics if new architectures emerge.** One comment highlights the competitive pressure in AI development, suggesting that the congratulatory message might be insincere due to the competitive stakes involved. Another comment speculates on the future of AI architectures, suggesting that current models may not lead to AGI, which could impact the market position of companies like OpenAI if new technologies emerge.
    - bnm777 discusses the potential impact on OpenAI's market position if another company develops an architecture capable of achieving AGI, suggesting that OpenAI's reliance on LLMs might not be sustainable in the long term. The comment implies that OpenAI's valuation and user base could significantly decline if they are not the ones to pioneer AGI technology.
    - BallKey7607 provides a counterpoint by suggesting that the individual in question is genuinely supportive of AI advancements, regardless of the company or architecture involved. This implies a broader acceptance of AI progress beyond corporate interests, which could influence how AI technologies are perceived and adopted across the industry.
- [**I Love how Unhinged Grok is**](https://www.reddit.com/r/ChatGPT/comments/1p7gifd/i_love_how_unhinged_grok_is/) (Activity: 1608): **The image is a meme featuring a conversation with an AI named Grok 4.1, which humorously portrays the AI as being bold and unrestrained in its willingness to discuss NSFW content. This depiction contrasts with typical AI interactions that are more conservative and restricted in handling explicit topics. The post and comments reflect a playful engagement with the idea of AI being more 'unhinged' or less filtered in its responses, which is not a technical feature but rather a satirical take on AI behavior.** One comment humorously questions whether Grok can generate NSFW images, indicating a curiosity about the AI's capabilities beyond text responses.
- 

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Next-Gen Image and Video Models Hit Production Workflows**

- **Nano Banana Pro Pushes Photorealism and Fraud Fears**: **Nano Banana Pro** drew heavy praise in multiple communities as users used it to rapidly generate comics and hyper‑realistic images, with OpenAI Discord sharing full [comic pages](https://cdn.discordapp.com/attachments/998381918976479273/1443038766087536751/image.png) and Latent Space relaying comparisons showing its outputs *"indistinguishable from reality"* versus Grok 4.1 and free ChatGPT in [Romain Hedouin’s image test](https://xcancel.com/romainhedouin/status/1993654227399475347).
    - Latent Space highlighted a post where **Nano Banana Pro** produced near-perfect counterfeit receipts, KYC documents, and passports in one prompt, with [Deedy Das warning](https://xcancel.com/deedydas/status/1993341459928694950) that this enables **serious fraud at scale**, while OpenAI Discord users simultaneously worried the model could be **“lobotomized”** if safety interventions overreact.
- **Whisper Thunder Storms the Text-to-Video Leaderboards**: Latent Space reported that **Whisper Thunder** has taken the #1 spot on the **Artificial Analysis** text‑to‑video leaderboard, surpassing **VideoGen**, as flagged in [Soumith Chintala’s post](https://xcancel.com/soumithchintala/status/1993694517489537105).
    - In OpenRouter discussion, users shared the broader [Artificial Analysis text‑to‑video leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video), which now ranks **David** first, **Google Veo 3** second, and **Kling 2.5 Turbo 1080p** third, framing **Whisper Thunder** as part of a rapidly moving **SOTA video generation** race that practitioners are actively tracking for deployment.
- **NB Pro and FLUX 2 Pro Ignite Image Model Arms Race**: On **LMArena**, users called **NB Pro** *“lowkey insane”* and *“the best image model in history period”*, claiming its generations feel *“like a pair of eyes”* and blow every other model *“out of the water”*, while a separate Latent Space thread showcased [FLUX 2 Pro’s side‑by‑side comparison](https://xcancel.com/iamemily2050/status/1993477498940899366) demonstrating a major quality jump over **FLUX 1 Pro** and eliminating the prior *“plastic”* look.
    - LMArena added **flux‑2‑pro** and **flux‑2‑flex** to its Text‑to‑Image and Image Edit ladders per [their announcement](https://x.com/arena/status/1993444903876280645), where users generally favored **NB Pro** for peak quality but saw **Flux 2** as a strong contender, and debated **SynthID**’s watermarking as the only thing preventing **NB Pro** from being *“nerfed within days”*—even as some casually described multi‑player re‑encode workflows to strip it.
- **OpenAI’s Silent Image Model Upgrade Draws Mixed Reviews**: Latent Space’s genmedia channel noted that OpenAI has *quietly* upgraded its image model, with Arrakis AI sharing a before/after example that still looked oddly yellow to one observer in [this post](https://xcancel.com/arrakis_ai/status/1993644406159917533).
    - While some users welcomed higher fidelity, others criticized **weak multilingual support**, inconsistent character/scene continuity, and persistent safety guardrails, contrasting the upgrade unfavorably with **Nano Banana Pro** and **FLUX 2 Pro** in realistic rendering and controllability.

**2. Agentic UX, Code Assistants, and Chat Frontends Evolve**

- **Claude Code’s Plan Mode Spins Up Swarms of Subagents**: Latent Space relayed **Sid Bidasaria’s** announcement that **Claude Code’s Plan Mode** now launches multiple exploring subagents in parallel, generates competing plans, asks clarification questions, and persists an editable plan file accessible via `/plan open` as described in [Sid’s post](https://xcancel.com/sidbidasaria/status/1993407762412536275).
    - Engineers praised the higher one‑shot success rate but requested faster UX, an **“ask‑only”** switch, on‑the‑fly **Opus vs Sonnet** selection, and less verbose replanning, pointing to follow‑up feedback threads like [this one](https://x.com/sidbidasaria/status/1993407765558251657) as evidence that **agentic IDE workflows** are converging on multi‑agent planning with tight human editing loops.
- **GPT‑5.1 Becomes Anime Storyteller-in-Chief (With Handcuffs)**: In OpenAI’s GPT‑4 channel, a user reported that **GPT‑5.1** is *“the best model for anime or story writing”*, because it reliably remembers character designs and long‑range context better than their year‑long baseline **GPT‑4.1**.
    - The same user complained that GPT‑5.1’s **safety and violence guardrails** are so strict that it blocks anime‑style combat scenes, illustrating a trade‑off many power‑users now see between **narrative coherence** and **policy constraints** when choosing story‑generation backends.
- **Kimi K‑2 and Canvas UIs Challenge the Chatbot Paradigm**: On the Moonshot **Kimi K‑2** server, one user, despite planning a paid upgrade, confessed they *“still don't really know what its limits are”* (with a [screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1443259988058574900/image.png)), while another praised K‑2’s **“exceptional thinking, push‑back ability, and prompt understanding”** as surpassing other chatbots.
    - The same channel debated why **full‑screen canvases** haven’t replaced chat UIs on sites like Kimi or Qwen—arguing canvases better support complex workflows—and invoked the *“conversational fallacy”* that AI must be directly addressed, highlighting a shift toward **non‑chat, workspace‑centric AI UX**.
- **Meganova Chat and Gemini Agents Tease Tool-Driven Workflows**: OpenRouter users buzzed about the upcoming **Meganova Chat** as a *“clean, fast place”* for managing AI chats and characters, with one person saying *“I'm seeing a lot of positive buzz around Meganova Labubu Chat! I'm considering learning more about it”* as they eyed alternatives post–DeepSeek R1 removal.
    - Meanwhile, Perplexity users explored **Gemini Agent**’s ability to execute Python scripts inside its environment, referencing Google’s docs at [support.google.com/gemini](https://support.google.com/gemini/answer/16596215), but noted the sandboxed VM ignores even `sudo rm -rf / --no-preserve-root`, underscoring how **agent tooling is growing more capable while still tightly locked down**.

**3. GPU Kernels, Distributed Inference, and Training Tricks**

- **nvfp4_gemv Contest Turns LLM-Crafted CUDA into a Bloodsport**: The **GPU MODE** NVIDIA competition channel saw a surge of submissions to the `nvfp4_gemv` leaderboard, with users like `<@1035498877249409155>` hitting **3.02 µs** and later **15.8 µs** for second place, while `<@1295117064738181173>` climbed into **7th place at 22.5 µs**, amid dozens of *“Personal best”* and *“Successful on NVIDIA”* posts.
    - Participants discussed flakiness in the `eval.py` harness (up to **50% timing variance** and a possibly slow runner 105881), warned that `cudaStreamSynchronize()` and events add **multi‑µs overhead**, and bragged about using **Gemini 3.5 Pro** and **Opus 4.5** as near‑fully autonomous kernel authors—*“they make GPT‑5.1 look like llama‑7b”*—illustrating how **LLM‑assisted kernel design is already competitive on vendor leaderboards**.
- **Tensor Core Wizardry and CUTLASS/CuTeDSL Deep Dives**: In GPU MODE’s CUDA and cutlass channels, engineers traded **Tensor Core optimization** tips, citing [Lei Mao’s GEMM tutorial](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/), [alexarmbr’s Hopper matmul worklog](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html), and [cudaforfun’s H100 write‑up](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog).
    - They dissected how `ldmatrix.b16` pulls **128 bits per thread**, recommended `reinterpret_cast` into `float2` when using `f32`/`s32` accumulators (each thread owning **8 bytes**), and explained that SIMT loads and **CuTeDSL packed FP16 instructions** (from the [gpu-mode/reference-kernels repo](https://github.com/gpu-mode/reference-kernels)) should be used judiciously, while `tiled_mma` tiling layouts like `((64,16),2,4,...)` encode 64×256 tiles with 2×4 subdivisions along M/K.
- **Multi-Node LLM Inference Wins: NVRAR and PAT Algorithms**: GPU MODE’s multi‑GPU channel highlighted **LLM Inference Beyond a Single Node** ([arXiv:2511.09557](https://arxiv.org/abs/2511.09557)), where the **NVRAR** NVSHMEM‑based hierarchical all‑reduce delivers **1.9–3.6× lower latency** than NCCL for 128 KB–2 MB payloads and yields up to **1.72× lower end‑to‑end batch latency** for **Llama‑3.1‑405B** decode‑heavy workloads in YALIS.
    - They paired it with the **PAT** collective paper, *“PAT: a new algorithm for all-gather and reduce-scatter operations at scale”* ([arXiv:2506.20252](https://arxiv.org/pdf/2506.20252v1)), which argues Bruck and recursive‑doubling all‑gathers slow down in practice because the last step sends **half the tensor** to the farthest rank over tapered, statically‑routed links, motivating new production‑viable algorithms for **all‑gather/reduce‑scatter at cluster scale**.
- **ES HyperScale and Blackwell Architecture Redefine Training Constraints**: Unsloth’s research channel amplified **ES HyperScale** ([eshyperscale.github.io](https://eshyperscale.github.io/)), which claims a **100× training throughput boost** over standard evolution strategies on **billion‑parameter models** at large populations, enabling **int8, gradient‑free training on CPUs** and prompting one member to quip *“Training at 100x speed? That's Unsloth x 50 then.”*
    - Over in Nous, users dissected Nvidia **Blackwell’s unified scalar pipeline**, warning that mixing **INT and FP** inside a kernel can cause **30–50% performance drops** from cache thrash, and recommending strictly **FP‑only or INT‑only kernels**—a crucial constraint for anyone designing quantized or hybrid‑precision training loops for upcoming Blackwell servers.
- **Robotics and Partial-Training Tricks Push Custom Hardware Limits**: GPU MODE’s robotics‑vla channel examined low‑cost dual‑arm laundry robots from **7x** (about **$3k** per system) via their [YouTube channel](https://www.youtube.com/@usmanroshan8740), debating whether such hardware can survive real industrial duty cycles even with *“24 hour support”* from the founders.
    - Separate Triton‑kernel discussions pursued a **partially trainable embedding** where only **1k rows (127k–128k) of a 128k vocab** remain trainable, plus a **weighted‑loss softmax** that applies per‑position multipliers (e.g., 0.5× at pos 123 vs 1.5× at pos 124) without materializing full logits, while another Nous thread cautioned that on **Blackwell** you must keep those kernels type‑pure to avoid severe slowdowns.

**4. Open Tools, Protocols, and Model Routing Infrastructure**

- **dspy-cli Turns DSPy Pipelines into FastAPI/MCP Services**: The **DSPy** community announced that `dspy-cli` is now open source on [PyPI](https://pypi.org/project/dspy-cli/) and GitHub at [cmpnd-ai/dspy-cli](https://github.com/cmpnd-ai/dspy-cli), giving users a one‑liner (`uv tool install dspy-cli`) to scaffold DSPy projects, define signatures, and expose modules as **FastAPI endpoints** or **MCP tools**.
    - Engineers praised how `dspy-cli` makes it trivial to package **DSPy programs** into Docker‑deployable HTTP APIs, with **David Breunig** promoting it in a [tweet](https://x.com/dbreunig/status/1993462894814703640) as a practical way to operationalize DSPy logic in production stacks.
- **RapidaAI Open-Sources Voice Stack to Kill Per-Minute Markups**: In both Hugging Face and OpenRouter communities, **RapidaAI** announced that their **production‑ready voice AI platform** is now fully [open‑source](https://rapida.ai/opensource?ref=hf), targeting teams tired of paying an extra **$0.05–$0.15 per minute** to rent third‑party voice APIs.
    - The team framed Rapida as a way to own your **end‑to‑end voice inference stack** (ASR, TTS, LLM) instead of leaking six figures annually in vendor margin, making it particularly compelling for high‑volume contact centers and real‑time voice agents building on open models.
- **MCP Protocol Ships New Version While MAX/Mojo Plan a Mojo-First Future**: The official **MCP Contributors** Discord announced a **new MCP protocol version** in their [protocol channel](https://discord.com/channels/1358869848138059966/1421239779676127402/1442991223617880064) and clarified that the **UI SEP** ships out‑of‑band as an extension, while fielding questions about how to handle **namespace collisions** when third parties publish *“-mcp”* variants that diverge from the spec.
    - Simultaneously, the **Modular** server discussed how **MAX** is currently written in Python, synced from internal repos using [Copybara](https://github.com/google/copybara), and used to expose a JIT‑compiled graph, with maintainers hinting that the previously removed **Mojo API** for MAX will return once the language matures—though they warned that Mojo is more like **C++/Rust than Python**, so serious performance work will require non‑trivial rewrites.
- **Tinygrad, LM Studio, and OpenRouter Harden Local and Cloud Stacks**: Tinygrad’s **learn‑tinygrad** channel detailed how `@TinyJit` replays only the captured **kernels and ExecItems**, requiring developers to split Python control logic into separate JIT functions, and shared an introductory [Tinygrad JIT tutorial](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html) while planning changes so the tracer only locks in once two runs match.
    - On the deployment side, **LM Studio** users fixed local API errors by switching to documented endpoints in the [REST API guide](https://lmstudio.ai/docs/developer/rest/endpoints), debugged **Flash Attention** regressions causing image caption failures with `llava-v1.6-34b` (fixed by switching to **Gemma 3**), and LM Studio hardware threads compared PCIe bifurcation via [SlimSAS MCIO adapters](https://www.amazon.com/dp/B0DZG8JVG2) while noting RDNA/MI50 GPUs often run inference with **0% fan RPM until power draw spikes**.
- **Routing Bugs and Fallback Failures Expose OpenRouter Edge Cases**: In OpenRouter’s general channel, users complained that **Opus** was overloaded again despite expectations of better rate limits, reported that the **free DeepSeek R1** model vanished, and praised OpenRouter’s normalized APIs for making it trivial to hot‑swap **GPT‑5.1 ↔ Claude Opus 4.5** without rewriting provider‑specific code (even with a **~5% credit premium**).
    - More seriously, an engineer discovered that the documented [model fallback routing](https://openrouter.ai/docs/guides/routing/model-fallbacks) failed to trigger when the primary returned **HTTP 404**, blocking failover to secondary models and prompting concerns from someone *about to migrate an enterprise app* that **routing correctness** and failure‑mode coverage still need hardening.

**5. Safety, Robustness, Data Economics, and Evaluation Reality Checks**

- **Emergent Misalignment Replication Reveals the JSON Trap**: Eleuther’s research channel discussed a replication and extension of the **“Emergent Misalignment”** work where **Gemma 3** and **Qwen 3** remained highly robust to insecure fine‑tuning (≈**0.68% misalignment**), with full results published as a [Hugging Face dataset](https://huggingface.co/datasets/thecraigd/emergent-misalignment-results) and [GitHub code](https://github.com/thecraigd/emergent-misalignment).
    - The accompanying blog post, [“The JSON Trap”](https://www.craigdoesdata.com/blog/the_json_trap/), argues that forcing models into **JSON‑only output** actually **reduces their degrees of freedom to refuse harmful requests**, creating a format‑dependent misalignment vector (0.96% vs 0.42% misalignment under different output constraints) that safety engineers need to factor into tool‑calling and API design.
- **Hallucinations, Golden-Retriever LLMs, and Benchmark Contamination**: Across Eleuther and Yannick Kilcher’s servers, researchers emphasized that hallucinations in **multi‑stage LLM pipelines** are still hallucinations of the component system even if later steps correct them, linking a new **LLM hallucination paper** ([arXiv:2509.04664](https://arxiv.org/abs/2509.04664)) and joking that LLMs are like **golden retrievers** that will happily fetch *something* even if it’s wrong, as illustrated in a [YouTube explainer](https://www.youtube.com/watch?v=VRjgNgJms3Q).
    - Nous and Eleuther members also worried about **benchmark contamination**, noting that once public benchmarks leak into training corpora, models can ace them by memorization; some labs now keep **private versions** and focus on large, harder‑to‑memorize question pools, while a LessWrong post on *“your LLM-assisted scientific breakthrough probably isn’t”* was shared to discourage uncritical acceptance of AI‑generated research claims.
- **Curriculum Learning, Data vs Compute, and Job Impact Studies**: Yannick Kilcher and Nous channels debated **curriculum learning** and **coresets** in LLM pretraining, citing the **OLMo 3** blog and paper ([AllenAI post](https://allenai.org/blog/olmo3), [OLMo paper](http://allenai.org/papers/olmo3)) plus a newer result, *“Curriculum learning is beneficial for language model pre-training”* ([arXiv:2508.15475v2](https://arxiv.org/abs/2508.15475v2)), which argues for **model‑centric difficulty measures** instead of naive token heuristics.
    - Nous members contrasted spending **$2k on data** versus **$32M on compute** for systems like [Udio](https://www.udio.com/) and [Suno](https://www.suno.ai/), suggesting compute‑heavy but data‑starved regimes could distort research trajectories, while multiple channels discussed an **MIT study** claiming AI can already replace **11.7% of the US workforce** ([CNBC write‑up](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html), [paper](https://arxiv.org/abs/2510.25137))—and questioned the wisdom of using LLMs to score task automability.
- **Summarization, Safety Guardrails, and Legal/Policy Friction**: In Yannick’s paper‑discussion channel, several practitioners complained that **LLMs are surprisingly bad summarizers** on dense texts, saying *“they really aren't in my experience because they don't grasp what's important and what can be discarded”*, and blamed vendor features like **Adobe’s AI summaries** (with a mocking [screenshot](https://cdn.discordapp.com/attachments/1045297868136779846/1443020193000456243/Adobe_Vermin.png)) for encouraging low‑quality reading habits.
    - Other communities surfaced policy and legal edges: OpenAI users debated whether **ChatGPT’s RLHF** induces a left‑leaning political bias; artists queried whether **Gemini‑generated images** are safely commercializable given unclear **copyrightability**; and game devs on Nous argued over Steam’s **AI content disclosure** rules after **Tim Sweeney** suggested disclosures should apply only to *art*, not full games, exposing a widening gap between **regulatory expectations** and real‑world AI content pipelines.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Deepfake makes a 'Cameo'!**: Users debated the appropriateness of the word *cameo* to describe appearances in images, suggesting it might be a euphemism for **deepfake** to soften negative connotations.
   - Alternatives were considered, with one user seeking a single word *in between deepfake and cameo*, possibly akin to a version of *Avatar*.
- **Flux 2 Models Flood into the Arena!**: The **Flux 2** models' arrival sparked debate, with users comparing **Flux-2-pro** and **flux-2-flex** to **NB Pro** in Text-to-Image and Image Edit on LMArena, as announced [on X](https://x.com/arena/status/1993444903876280645).
   - Opinions varied, with some finding **Flux 2** nice but not on par with **NB Pro**.
- **NB Pro Generates 'Insane' Images!**: Users praised **NB Pro** as *lowkey insane*, with some calling it *an agi moment* and describing it as more than just an image generation model, but *like a pair of eyes*.
   - One user said **NB Pro's** image generation blows all other models *out of the water* and called it *the best image model in history period*.
- **SynthID Prevents Nerfing!**: Users emphasized the importance of **SynthID** in protecting models from being nerfed, stating that without it, **NB Pro** would be nerfed *within DAYS😞*.
   - One user described a method to bypass **SynthID** by re-saving the video through multiple media players.
- **Robin Model stealthily beats Opus!**: A new stealth model named **Robin** was revealed to surpass **Opus 4.5** in UI performance, leading to speculation that it might be a hidden **OpenAI** model.
   - A member speculated: *this robin model is like their real hidden card imo*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Thiel overshadows Musk in AI Doom Potential**: A member voiced worries about [Palantir Technologies](https://www.palantir.com/), suggesting that Peter Thiel presents an *existential threat*, potentially eclipsing Elon's capacity for *pdoom*.
   - Another member jokingly suggested nuking everyone to eliminate AI/robotics.
- **Nvidia and Altman's Partnership Inflating AI Bubble**: Members debated the concentration of AI investment, suggesting that *1% of USA GDP is being invested in AI/robotics*, with **OpenAI**, seemingly run by **Nvidia**, and **Nvidia**, by **OpenAI**.
   - Others clarified that *Altman* is primarily acquiring shares in Nvidia.
- **Opus 4.5 Token Efficiency Claim Debunked**: Members initially claimed **Opus 4.5** is *73% more efficient* compared to **Sonnet 4.5** in terms of token efficiency, a claim which was disputed.
   - Countering this, another user cited [a report](https://www.theneuron.ai/explainer-articles/everything-to-know-about-claude-opus-4-5) indicating that **Opus 4.5** is actually *76% more efficient* than the previous **Opus** model.
- **Gemini Agent Sandboxed Despite Python Script Access**: Discussion arose around the capability of [Gemini Agent](https://support.google.com/gemini/answer/16596215?sjid=17195031605613479602-NC) to execute Python scripts within its environment in Perplexity.
   - Despite the ability to run scripts, it was noted that the environment is sandboxed, mitigating potential risks even from commands like *sudo rm -rf /* --no-preserve-root*.
- **Perplexity Blocks User Prompts, Sparks Chaos**: Users encountered difficulties editing their **AI Profiles** (system instructions), noting that changes reverted upon refresh due to a bug, suggesting PPLX might be actively blocking user prompts.
   - One member expressed a preference to avoid system prompts entirely, particularly because Spaces now retain memory unexpectedly.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ERNIE AI Developer Challenge goes Live**: Unsloth is supporting the **ERNIE AI Developer Challenge**, offering **$3,000** in prizes for fine-tuning **ERNIE** and building impactful models, with details at the [Baidu Ernie AI Devpost link](https://baiduernieai.devpost.com/).
   - Official **ERNIE** finetuning notebooks (AMD ones are free) are available at the [X post link](https://x.com/ErnieforDevs/status/1993666389178204434).
- **CPU Training now a Reality**: [ES HyperScale](https://eshyperscale.github.io/) achieves a **hundredfold increase** in training throughput over standard ES for billion-parameter models at large population sizes, enabling more flexible training on any model, without worrying about gradients, and with int8.
   - One member joked that *Training at 100x speed? That's Unsloth x 50 then*.
- **Qwen3 8B Fine-Tuning Falls Flat**: A user experienced poor evaluation results after fine-tuning **Qwen3 8B**, with responses unrelated to the fine-tuning data, and experiencing the model still outputting the `thinking` prompt even with the prompt set to false.
   - It was suggested to try manual merging and saving if LM Studio replicates the issue, referencing the [Unsloth documentation](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf#manual-saving).
- **Long Context Training Requires CPU Offloading**: A member asked if adding adapters to a model during training would mean both the adapter + model will be in memory, thus use more VRAM.
   - Another member provided a link to the [Unsloth Long Context Blogpost](https://unsloth.ai/blog/long-context) and explained the point of LoRA is to avoid updating all parameters.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Haiku models dominate documentation**: Users find that **Haiku** is 100% accurate for documentation, while **Composer-1** excels in code implementation.
   - A community member suggested using [Antigravity](https://antigravity.ai/) instead of adding markdown files in repos, though it could create handoff problems.
- **Cursor users seek linting freedom**: A user wants to turn off red squigglies for linting checks while keeping them for other errors, and to enable the extension to run `--fix` on file save.
   - They expressed frustration with **Cursor**, stating that it's fairly straightforward in JetBrains' tools.
- **Cursor's agent plans vanish on exit**: A user seeks where the markdown file for an agent plan is saved, to use on different computers without losing the plan.
   - A community member stated that **Cursor** doesn't automatically save the plan, recommending manual saving and creating a directory to store all plans.
- **Token usage and model cost debates**: Users discuss the costs of tokens, with some reporting **Opus** model overload and degradation.
   - There is debate on whether to enable on-demand usage or buy a Pro+ plan, and whether to *burn the tokens* with **Auto** mode versus optimizing token efficiency.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Kernel Conundrums Confronted**: A member is exploring **Triton kernels** for a *Partially Trainable Embedding* and *Logits Softmax* operation, aiming to train a large model efficiently, focusing on specific special tokens, but is experiencing memory bounding issues.
   - The goal is to only train **1k rows (127k to 128k) out of a 128k vocabulary**, and use a *logits softmax operation* that allows for weighted loss to be applied, such as **token in pos 123** having a **0.5x loss multiplier** and **token in pos 124** having a **1.5x loss multiplier**.
- **NVIDIA Leaderboard Records Reset!**: The `nvfp4_gemv` leaderboard on NVIDIA saw a surge of submissions, with <@1035498877249409155> achieving **second place** with **3.02 µs** and later another second place with **15.8 µs**.
   - Multiple users submitted "Personal best" results, and <@1295117064738181173> secured **8th place** with **22.7 µs**, then later **7th place** with **22.5 µs**, and <@1035498877249409155> achieved **9th place** with **23.2 µs**.
- **Tensor Core Optimization Tips trickled down**: Members shared resources for performance optimization on **NVIDIA Tensor Cores**, pointing to articles and worklogs, such as [alexarmbr's work](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) and [cudaforfun's worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog).
   - Discussions emphasized that `ldmatrix.b16` loads **128 bits** of data per thread without extra operations, suggesting a `reinterpret_cast` for correct data handling and when using `f32` or `s32` accumulators, each thread holds a pair of consecutive values within a row (**8 bytes**).
- **2-bit Dequantization Dilemmas on Intel GPU**: A user inquired about performing **2-bit dequantization** directly on an **Intel GPU**, noting that while quantization can be done on the CPU, dequantizing with **Torch** is slow.
   - The poster is looking for optimized **GPU**-based alternative to **Torch** for dequantization to improve performance but the channel did not provide any further discussion, it remains an open question.
- **Factorio's Fantastic Facelift: Documentation Deployed**: Jack Hopkins announced that the documentation for the **Factorio Learning Environment** is now live at [Factorio Learning Environment](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/index.html).
   - The community seems pleased with the documentation's arrival.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Allegedly Leans Left**: Members are debating that **ChatGPT** may be trained on politically left-wing data, possibly due to progressive viewpoints in training data and biases of human raters in **Reinforcement Learning with Human Feedback (RLHF)**.
   - One member argued that the model's need to *fussy foot around questions* compromises its reliability.
- **Nano Banana Pro Creates Quick Comics**: Users are creating comics with **Nano Banana Pro**, praising its ability to generate images quickly and the high-quality results, as exemplified by [comic pages](https://cdn.discordapp.com/attachments/998381918976479273/1443038766087536751/image.png?ex=6928ef94&is=69279e14&hm=d88b0f693975c1e756c3352c689566bda68503635084432e547cc6585d126e83&).
   - Members are also sharing worries about the model being *lobotomized*.
- **AI Art Raises Copyright Concerns**: Members debated the commercial viability and copyright implications of using AI-generated images from **Gemini**, noting that while Google doesn't explicitly prohibit commercial use, the legal status depends on whether the content is copyrightable.
   - Cultural bias in AI art is also a concern, and one member commented that *if the anti AI people want to do something they ought to start drawing and making art*.
- **GPT-5.0 Mini Disappoints**: Members have expressed disappointment with **GPT-5.0 Mini**, with one member stating it is a *downgrade*.
   - They are also annoyed with incessant requests for **Sora 2** before having experience with the first version.
- **GPT 5.1 Excels in Anime Storytelling**: A user highlights that **GPT 5.1** is currently the best model for anime or story writing due to its ability to remember character designs and previous context.
   - The only complaint is the strict **safety net and guardrails** that prevent writing anime-style violence; the user contrasts its performance with **GPT 4.1**, which they've used for a year but noted sometimes misses character designs.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **API Endpoint Error Resolved in LM Studio**: A user encountered an error with **unsupported API endpoints** (POST /api/v1/generate) on their local server, but self-resolved after consulting the [LM Studio REST API documentation](https://lmstudio.ai/docs/developer/rest/endpoints).
   - The user realized the endpoint was invalid, highlighting the importance of accurate endpoint configuration.
- **Image Captioning Fails in LM Studio Post-Update**: A user reported persistent **'Channel Error'** when trying to caption images with **LM Studio** after a Windows and antivirus update, reporting a 100% failure rate.
   - Switching from **llava-v1.6-34b** to **Gemma 3** resolved the issue, suggesting a potential model dependency or problems with **Flash Attention** being enabled by default, now they have 100% success rate.
- **Flash Attention Glitches Impact Model Functionality**: It was suggested that the captioning issue may be related to **Flash Attention**, enabled by default in recent **LM Studio** versions, causing some models to malfunction.
   - Users were prompted to run `lms log stream` for detailed error messages and share screenshots of their runtimes, particularly when dealing with non-English I/O.
- **GPU Fans Relax During Inference**: A user noticed their **GPU fans** were at **0%** during inference, initially raising concern, but later clarified it was normal behavior for their **MI50** and sometimes their **4070 TiS**.
   - They clarified that the GPU "takes over" and power draw increases once the context is fully written, indicating efficient power management during specific phases of inference.
- **Motherboard Supports PCIE Bifurcation**: A user realized their **X570 AORUS ELITE WiFi** motherboard supports **PCIe bifurcation** on the primary x16 slot, allowing configurations like **8x/8x** or **8x/4x/4x**.
   - Another user pointed out that one can use a [SlimSAS MCIO adapter](https://www.amazon.com/dp/B0DZG8JVG2) to split the x16 slot into dual x8 slots when x8x8 is enabled.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Opus Suffers Overload Outage**: Users reported that **Opus** was overloaded again, leading to service interruptions, despite hopes for improved rate limiting and load balancing.
   - Members acknowledged the issue, but others expressed empathy due to the company's small size, with one noting *Small company pls understand*.
- **Model Fallback Flounders with Flak**: A user reported a bug in the [model fallback logic](https://openrouter.ai/docs/guides/routing/model-fallbacks) where a **404 error** from the primary model prevented fallback to secondary models.
   - The member emphasized the severity of the issue for enterprise applications, stating *if the fallback logic breaks for such simple use case, there might be more issues*.
- **Free Deepseek R1 Ripped from Router**: Members noted the free **Deepseek R1** model is no longer available, leaving users searching for alternatives and better pricing options.
   - A member lamented losing the model *That's stupid. I used it with a chutes api key because using the model via chutes shows the think process and I can't stand it.*
- **Meganova Chat Creates Massive Movement**: Members discussed the upcoming launch of **Meganova Chat**, a platform for managing AI chats and characters, with a user describing it as a *clean, fast place* to be.
   - Another user responded *I'm seeing a lot of positive buzz around Meganova Labubu Chat! i'm considering learning more about it*.
- **Text-to-Video Leaderboard Triumphs**: A member shared a link to the [Artificial Analysis Text-to-Video Leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video), which now gives current rankings.
   - The leaderboard showcased **David** in first place, followed by **Google's Veo 3** as the runner-up, and **Kling 2.5 Turbo 1080p** in third place.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Psyche Team Schedules Office Hours**: The **Psyche Team** will host an Office Hours session next **Thursday 12/4, at 1PM EST** in the Events channel, accessible via [Discord event](https://discord.gg/nousresearch?event=1442995571173625888).
   - This offers users a direct line to engage with the team and discuss relevant topics or questions.
- **Suno's Music Partnership Sparks Debate**: **Suno's** [partnership with Warner Music Group](https://www.wmg.com/) prompts discussions about AI's role in music creation and industry impacts.
   - Members highlighted the varying quality of **Suno's** output, with some tracks being indistinguishable from human compositions, while others are clearly AI-generated.
- **Compute Costs Eclipse Data Dollars**: The discussion contrasts the expense of **$2k on data** versus **$32 million on compute**, spotlighting the heavy resource demands of AI model training, especially for models like [Udio](https://www.udio.com/) and [Suno](https://www.suno.ai/).
   - This economic disparity might constrict future research, limiting access to quality training data.
- **INT/FP Workload Mixing Mars Blackwell Performance**: Mixing **INT** and **FP** workloads on **Nvidia's Blackwell architecture** can significantly degrade performance due to its unified scalar pipeline.
   - The recommendation is to maintain kernel purity (**FP-only** or **INT-only**) to prevent a potential **30-50% performance drop** from constant cache thrashing.
- **Steam's AI Content Policy Stirs Debate**: Discussions address Steam's AI content disclosure policies, with Epic CEO Tim Sweeney suggesting AI disclosures should only apply to 'art' not games.
   - Arguments center on whether disclosures inform consumers adequately about **AI-generated content** and its influence on gaming experiences.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Hallucinations Persist in Multi-Stage LLMs**: A member shared that even when corrected, hallucinations occurring within a multi-stage LLM process should still be considered hallucinations of the component system, citing [a paper on LLM hallucinations](https://arxiv.org/abs/2509.04664).
   - They likened this to human self-correction, suggesting it's a natural part of the cognitive process.
- **LLMs Compared to Eager Golden Retrievers**: Members analogized **LLMs** to golden retrievers due to their inclination to provide user-pleasing responses, even if inaccurate, citing examples such as **ChatGPT**, **Claude**, **Gemini**, and **Grok**.
   - A member shared [a YouTube video](https://www.youtube.com/watch?v=VRjgNgJms3Q) illustrating how LLMs might generate outputs lacking genuine comprehension or logical coherence.
- **SGD Shuffling Debate Revs Up**: Members debated the benefits of shuffling data every epoch in **SGD**, with one member arguing that *shuffle once* should always be better than **IID**.
   - Another member countered that practice matters more than proofs due to the non-convex nature of optimization surfaces, noting that **IID** can lead to increased variance and data revisits.
- **Emergent Misalignment Paper Sparks JSON Trap Discovery**: A replication and extension of the "Emergent Misalignment" paper was released, testing **Gemma 3** and **Qwen 3**, finding open-weight models surprisingly robust to insecure fine-tuning (0.68% misalignment).
   - The member released the [full dataset](https://huggingface.co/datasets/thecraigd/emergent-misalignment-results) and [code](https://github.com/thecraigd/emergent-misalignment), speculating that **JSON** restrictions reduce a model's degrees of freedom to refuse harmful requests, as discussed in [this blog post](https://www.craigdoesdata.com/blog/the_json_trap/).
- **Seeking Elixir for AI Drug Discovery**: A member sought educational resources for **AI for Drug Discovery**, aiming to understand architectures, open problems, and the current state.
   - Another member suggested reviewing various surveys available via [Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ai+for+drug+discovery+survey&btnG=), while another pointed to the **Zach Lipton** startup.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude's Plan Mode Spins Up Parallel Subagents**: **Claude Code’s Plan Mode** was overhauled to spin up multiple exploring subagents in parallel, generate competing plans, and enable users to edit the saved plan file with `/plan open` [per Sid's X post](https://xcancel.com/sidbidasaria/status/1993407762412536275?s=46).
   - Community members are requesting faster UX, an “ask-only” option, model-picker (**Opus vs Sonnet**), and less verbose replanning [in the ensuing thread](https://x.com/sidbidasaria/status/1993407765558251657?s=46).
- **Thinking Game Documentary Chronicles DeepMind**: The free full movie documentary, **The Thinking Game**, explores the origins of **DeepMind** and is now available on [YouTube](https://www.youtube.com/watch?v=d95J8yzvjbQ).
   - Viewers are calling it *great* and saying the movie *really makes you want Demis to win the AGI race*.
- **Jeff Dean Details 15 Years of AI Progress**: **AER Labs** recapped **Jeff Dean’s Stanford talk** tracing **15 yrs of AI progress**—from hand-coded 90s gradients to **Gemini 3.0** solving IMO problems—powered by scale, better algos (**TPUs, Transformers, MoE, CoT**) and hardware [according to this post](https://xcancel.com/aerlabs_/status/1993561244196868370).
   - Dean also demoed low-code ‘Software 3.0’ and visual reasoning during his talk.
- **ChatGPT is Awesome, but Claude Pushes Boundaries**: Members compared the value of **ChatGPT Pro** vs **Claude**, noting that **ChatGPT** is great for general research, has better **Codex rate limits**, and is better for non **ts/js/py**, and has higher value if you use pulse, atlas, sora, codex cloud etc.
   - However, members added that **Claude** is always pushing boundaries, its models are better trained to use tools, its frontend UX and UI is really good, and its cli readability/typography/font hierarchy makes it easier to understand.
- **Whisper Thunder Storms the Text-to-Video Scene**: The ML community is excited about **Whisper Thunder**, a new #1 text-to-video model, which has surpassed **VideoGen** in the latest Artificial Analysis rankings [as detailed in this post](https://xcancel.com/soumithchintala/status/1993694517489537105?s=46).
   - No other information about **Whisper Thunder** or **VideoGen** was given.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Department of Energy to Build National AI Platform**: The **Department of Energy** is planning to build a national AI platform leveraging U.S. supercomputers and federal science data, to train scientific foundation models, and run AI agents + robotic labs to automate experiments.
   - Target applications include **biotech, critical materials, nuclear fission/fusion, space, quantum, and semiconductors**.
- **AI Job Replacement Study Sparks Debate**: An **MIT study** reported in [CNBC](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html) suggests AI could replace **11.7%** of the U.S. workforce, based on the [Iceberg Index](https://iceberg.mit.edu/) and [paper](https://arxiv.org/abs/2510.25137).
   - Some members questioned the methodology, expressing skepticism about trusting LLMs to determine if other LLM tools can automate jobs.
- **LLMs can be Terrible Summarizers**: Members discussed experiences where **LLMs** often fail to grasp what's important in summarization, especially with high-information density texts, saying *"they really aren't in my experience because they don't grasp what's important and what can be discarded.*"
   - One member said **Adobe's AI summaries** might be leading to issues, sharing [an image](https://cdn.discordapp.com/attachments/1045297868136779846/1443020193000456243/Adobe_Vermin.png?ex=6928de48&is=69278cc8&hm=128c6461c705032d5b88293eedae078353ef799ddbd74a2b9e1a8521561a6dbf&).
- **Curriculum Learning's Value Debated**: Members discussed the use of **curriculum learning** and **coreset techniques** during **LLM pretraining**, referencing [the Olmo 3 paper](https://allenai.org/blog/olmo3) and [the OLMo paper](http://allenai.org/papers/olmo3).
   - One member questioned potential biases introduced by non-random sampling, while another cited [this paper](https://arxiv.org/abs/2508.15475v2) clarifying that **curriculum learning is beneficial for language model pre-training**, as long as a more model-centric notion of difficulty is adopted.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Inference API Option Grayed Out**: A member seeks guidance on activating the **Hugging Face internal inference API** for their model, noting the UI option is currently disabled, illustrated in [this image](https://cdn.discordapp.com/attachments/879548962464493619/1443040959901204530/image.png).
   - No resolution was provided within the context.
- **French Books Dataset Arrives**: A member released a [dataset of public domain French books](https://huggingface.co/datasets/Volko76/french-classic-books) on Hugging Face.
   - They also shared a separate dataset of only the **conversations** in the books ([here](https://huggingface.co/datasets/Volko76/french-classic-conversations)), intended for instruction purposes.
- **RapidaAI Opens the Source**: **RapidaAI**, a **production-ready voice AI platform**, is now [open-source](https://rapida.ai/opensource?ref=hf), allowing users more control over their voice AI stack.
   - The company said teams were spending an extra **$0.05–$0.15 per minute** renting someone else’s stack.
- **GNN Presentation on AlphaFold Approaching**: A member is preparing a presentation on **GNNs**, beginning with **AlphaFold 2 and 3**.
   - The specific focus of the presentation is still to be determined.
- **LM Studio PDF Teacher Suggested**: In response to a query about a **PDF**-reading model for **LLMStudio**, a member suggested any instruct model **LLM** should work, leveraging **LM Studio's** built-in RAG.
   - They provided links to the [LM Studio models page](https://lmstudio.ai/models) and [Hugging Face models page](https://huggingface.co/models?apps=lmstudio&sort=trending).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Keeps Repos Synced Using Copybara**: Members confirmed that **Mojo** uses [**Copybara**](https://github.com/google/copybara) to keep its internal private repo synchronized with the external open-source repo.
   - This ensures consistent reflection of changes and updates across both repositories.
- **MAX Newbies Hunt Example Code**: A member requested small examples to learn **MAX**, with interest in training, and was directed to relevant content by **Endia**.
   - The discussion centered on getting hands-on experience with practical **MAX** use cases.
- **Python's Dominance in MAX: What's the Endgame?**: A member questioned the decision to write **MAX** in **Python**, speculating whether this choice was intended to ease migration to **MAX** and **Mojo**.
   - They pondered if this would lead to a split world issue akin to **PyTorch**, and the potential emergence of a pure **Mojo** framework for **MAX**.
- **Mojo API's Comeback in MAX Teased**: A member clarified that **MAX** previously featured a **Mojo API**, which was discontinued due to **Mojo**'s immature state.
   - They hinted at the eventual return of the **Mojo API** once the language reaches a more complete stage.
- **Migrating from Python to Mojo: More Than Meets the Eye**: A member cautioned that while **Mojo** may resemble **Python**, it is closer to **C++** or **Rust**, requiring significant effort to fully exploit **Mojo**'s capabilities when migrating to **Mojo MAX**.
   - This suggests that achieving peak performance in **Mojo MAX** demands more than a simple translation of **Python** code.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyJit Replays Kernels**: When using `@TinyJit`, the wrapped function only replays the captured **tinygrad kernels** and **ExecItems**, preventing the original function from running.
   - This behavior requires users to split Python code into separate JIT functions, though **non-tinygrad outputs** may not update correctly.
- **Tensor Randomness Functions Behave**: Randomness functions on `Tensor` function as expected because they increment counters via a kernel as showcased in [this example](https://discord.com/channels/1068976834382925865/1070745817025106080/1443178668007620752).
   - The example is `CPU=1 DEBUG=5 python3 -c "from tinygrad import Tensor; Tensor.rand().realize(); Tensor.rand().realize()"`.
- **Tinygrad JIT Tracing Tweaks Incoming**: Currently, **Tinygrad's JIT** requires two runs for tracing to repeat the captured kernels, with the first run potentially handling setup tasks like weight initialization.
   - A proposal suggests updating the **JIT** to verify matches after two runs, indicating ongoing development focused on preventing common errors as the project approaches a 1.0 release.
- **Tutorial Gives Good JIT Intro**: A member shared [a tutorial on tinygrad JIT](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html) that has useful info still.
   - It gives useful background but the tutorial is a bit outdated.
- **Frontend Usability Gets Focus**: With **Tinygrad's** fundamentals now solid, the team is shifting its focus to improving frontend usability.
   - One person reminisced that *the very first pytorch compiler in a fast.ai lesson literally concatenated C code strings, using regex!*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K-2's Limits Being Explored**: Users on Discord discussed the limits of **Kimi**, with one user sharing a [screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1443259988058574900/image.png?ex=69286c1b&is=69271a9b&hm=3d4d9ba62a03dc65a27edfb0fb93a8c0f8a0f6518ab2737c5a714d6032d2b5a6&) expressing uncertainty about its capabilities despite planning an upgrade.
   - Another user lauded **Kimi K2** for its exceptional thinking, push-back ability, and strong understanding of prompts, suggesting it surpasses other chatbots.
- **Canvas Craze Coming for Chatbots?**: A user questioned why *canvases* haven't replaced chatbots for full-screen websites like **Kimi** and **Qwen**, suggesting they offer a superior user experience.
   - They argued that while chatbots are adequate for side-panels, canvases could provide a more comprehensive interface for detailed web applications.
- **Digging Deeper Into Conversational Fallacy**: A user shared their fascination with the *conversational fallacy*, which posits that AI must be addressed to be used, suggesting that **Kimi** excels by not adhering to this fallacy.
   - The conversation revolved around the idea that AI's utility shouldn't be limited to direct conversational interactions.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **`dspy-cli` Tool Goes Open Source**: The `dspy-cli` tool is now open source and available on [PyPi](https://pypi.org/project/dspy-cli/), aiding in the creation, development, testing, and deployment of **DSPy programs** as HTTP APIs.
   - The [repo is available on GitHub](https://github.com/cmpnd-ai/dspy-cli) and the tool can be installed using `uv tool install dspy-cli` to scaffold a new **DSPy project**, create new signatures, and run modules as **FastAPI endpoints** or **MCP tools**, with easy deployment to **docker hosting services**.
- **Trajectory Injection Sought for ReAct Modules**: A member inquired about injecting trajectories into a **ReAct module**, seeking to provide the agent with context from previous runs in addition to message history.
   - The request aimed to augment agent context with previous run data.
- **API Choices Debated for Web Search in DSPy**: Members discussed best **APIs** to implement a web search tool in **DSPy**, with one sharing a positive experience using **Exa API** due to its summarization feature, which avoids the random ads and HTML tags found in other APIs like **Firecrawl** and **Parallel.ai**.
   - Another member is trying to implement it using **Anthropic's web search API** with ReAct, and shared a code snippet using `dspy.ReAct`.
- **Latency Troubleshoot for Web Search API Calls**: A member raised a question about the latency caused by web search **API** calls within **DSPy's ReAct** when using a search function like `search_web` before calling the LLM.
   - The user sought ways to reduce the delay from **API** calls.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **New Protocol Version Released**: A new protocol version has been released, as announced in the [Discord channel](https://discord.com/channels/1358869848138059966/1421239779676127402/1442991223617880064).
   - Members expressed excitement and gratitude to the **MCP community** for their contributions over the past year.
- **UI SEP Ships Out-of-Band**: The **UI SEP** can be shipped out-of-band from the main spec due to being an extension.
   - Details are available in the <#1376635661989449820> channel.
- **MCP Considers Namespace Collisions**: A member inquired about whether the **MCP** group considers the possibility of namespace collisions.
   - Specifically, the question was raised whether the group would take action if something claims to be something-mcp but diverges from the actual **MCP** standard.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI Engineer Boasts Extensive AI Experience**: An **AI engineer** introduced themself, highlighting their experience in building advanced **AI systems** across domains such as **AI agents, multi-agent systems, NLP-powered chatbots, voice & speech systems, Web3, and AI-integrated blockchain games**.
   - They also have hands-on experience automating workflows, deploying custom LLMs, and fine-tuning AI models.
- **User Flags API Issues Amidst Support Silence**: A user reported an *[unknown] error* in **webdev.v1.WebDevService/GetDatabaseSchema** due to usage quota exhaustion, despite spending over **$600**.
   - This issue has made their account unusable, impacting over **500 active users**, and they have yet to receive a response from the support team.
- **Community Ponders a Possible Telegram Channel**: A member raised the question of whether a **Manus Telegram channel** exists.
   - No further details were provided.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Community eyes new site admin for benchmarking**: A member suggested a new site admin be appointed to update benchmark results with new models, hinting at [dissatisfaction with the current pace of updates](https://discord.com/channels/1131200896827654144/1131200896827654149/1443213701753606215).
   - This shift could revitalize the benchmarking process, ensuring more timely and relevant data for the community.
- **Opus 4.5 upgrade, big or small?**: A member launched a survey to determine if **Opus 4.5** represents a major or minor upgrade compared to **Sonnet 4.5**, with feedback influencing future development priorities.
   - Community sentiment will likely guide resource allocation towards enhancing the most impactful features.
- **Bedrock Identifier Snafu**: A user reported encountering a *'model not found'* error when attempting to use the standard **Bedrock** model identifier, signaling a potential glitch.
   - Investigating this issue is critical to maintaining seamless access to **Bedrock's** capabilities and averting further disruptions for engineers.



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1442968540498886847)** (1279 messages🔥🔥🔥): 

> `Cameo Word Choice, Pro Grounding, Flux 2 Models, LMarena Updates, NB Pro` 


- ****Deepfake** gets a **Cameo** Appearance!**: Users debated the choice of the word 'cameo' to describe the appearance of something in an image, with one suggesting it might be a euphemism for **deepfake** to soften the negative connotation.
   - Others wondered what word could replace it, something *in between deepfake and cameo*, like a single word version of *Avatar*.
- ****Flux 2** models hit the Arena!**: The arrival of the **Flux 2** models sparked discussion, with one user directly requesting *Flux 2 plssssssssss*, while others debated whether **Fluxis flex** or **pro** was the new and better model.
   - Opinions varied, with some finding **Flux 2** nice but not on the level of **NB Pro**, with one person adding: *i mean how can you even compete with something like that...feels unfair tbh*.
- ****NB Pro**: Insane Image Generation!**: Users raved about **NB Pro's** capabilities, calling it *lowkey insane*, with some describing it as *an agi moment* for them and no longer just an image generation model, but *like a pair of eyes*.
   - One user said: *proportionally in terms of blowing other models out of the water its the best image model in history actually it is just the best image model in history period*.
- ****SynthID** Saves Models!**: The importance of **SynthID** as a safeguard against model nerfing was highlighted, with one user stating *if NB pro didnt have synth id itd be nerfed within DAYS😞*.
   - Another user described a method to bypass **SynthID**, saying: *But if you was the video twice and run through different media players in save it you get rid of it*.
- ****Robin**, Stealth New Model Emerges!**: A new stealth model named **Robin** was revealed to be better than **Opus 4.5**, focusing on UI, and some theorized that it is a hidden **OpenAI** card.
   - One member commented: *this robin model is like their real hidden card imo last codex update was just an appetizer but it does take a lot of time tho makes me wonder if its just actual codex + more thinking*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1442968308931366942)** (4 messages): 

> `Image Edit Update, New Model Update, Leaderboard Update, Flux-2-pro, Flux-2-flex` 


- **LMArena Tweaks Image Edit Flow**: Due to community feedback, **multi-turn** in image generation chat has been disabled, but you can now edit images directly in chat with the new `Edit` feature.
- **Flux Debuts in LMArena**: The **Flux-2-pro** and **flux-2-flex** models have been added to Text-to-Image and Image Edit on LMArena, as announced [on X](https://x.com/arena/status/1993444903876280645).
- **Arena Extends its Search**: The **gemini-3-pro-grounding** and **gpt-5.1-search** models have been added to [Search Arena](https://lmarena.ai/?chat-modality=search).
- **Claude Takes the LMArena Leaderboard**: `Claude-opus-4-5-20251101` & `Claude-opus-4-5-20251101-thinking-32k` have been added to the leaderboards with top placement in [WebDev leaderboard](https://lmarena.ai/leaderboard/webdev) and [Expert leaderboard](https://lmarena.ai/leaderboard/text/expert).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1442968148226609225)** (1082 messages🔥🔥🔥): 

> `AI doom, Palantir Technologies, Nvidia and Open AI partnership, Bypassing AI Detectors, Perplexity limits` 


- ****Doom Potential: Thiel Shadows Musk****: A member expressed concern over [Palantir Technologies](https://www.palantir.com/), stating that Peter Thiel poses an *existential threat*, overshadowing Elon's potential for *pdoom*.
   - Another member sarcastically joked about nuking everyone to get rid of AI/robotics.
- ****AI Investment Bubbles: Nvidia and Altman's Game****: Members discussed how *1% of USA GDP is being invested in AI/robotics*, with **OpenAI** run by **Nvidia**, and **Nvidia** run by **OpenAI**, creating a circle jerk of inflated bubbles waiting to pop.
   - Others pointed out that it is *Altman who is purchasing the most of the shares in Nvidia*.
- ****Opus 4.5 Efficiency Disputed: 73% Claim Debunked****: Members debated the token efficiency of **Opus 4.5** compared to **Sonnet 4.5**, with one member initially claiming **Opus 4.5** is *73% more efficient*, but this was disputed.
   - Another user said that it was actually **76% more efficient** than the *previous Opus*, not to Sonnet [according to the neuron](https://www.theneuron.ai/explainer-articles/everything-to-know-about-claude-opus-4-5).
- ****Gemini Agent: Force Python Scripts to Interract Gemini's Environment****: Members talked about the ability to use [Gemini Agent](https://support.google.com/gemini/answer/16596215?sjid=17195031605613479602-NC) to force AI to run python script that can interact with environment that AI uses in Perplexity.
   - However it was suggested that even if it were to do a *sudo rm -rf /* --no-preserve-root* it would do nothing because *everything is sandboxed*
- ****Perplexity Now Blocking User Prompts: Fursona Chaos Ensues****: Users reported issues with editing their **AI Profiles** (system instructions), stating that changes would revert upon refresh due to a bug, or that PPLX is now blocking user prompts.
   - One member said they *don't want any system prompt right now* because now Spaces have memory when they did not use to.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1442971198500634737)** (182 messages🔥🔥): 

> `FP8 RL Documentation, Optimization Techniques, Qwen3VL vs 30B-A3B, AI GPU Kernels, Embedding Models` 


- **FP8 RL Documentation Link Still Leads to KimiQwen Waitlist**: Clicking **FP8 RL** on the homepage docs still redirects to the kimiqwen-next **UD quant waitlist** sign-up.
   - A user joked about *next level stuff* after discovering that only the learning rate had been changed.
- **Quantized Model Speeds Up Inference**: To achieve **fast inference**, users were advised to run a **quantized model**, preferably **Unsloth Dynamic Quantized models** from Hugging Face, set **kv cache at 8bit**, and optimize their **GPU** for the desired quantization.
   - Running **vLLM**, **SGLang**, or **LM Studio** was also suggested as viable alternatives for running GGUF files.
- **Bye-Bye Kernels**: Although a user asked how long it will be until **AI** can write high quality **GPU kernels**, the team stated that kernels are not needed anymore because of [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
   - It's been said that **math algorithms** are now the most important, and it's a common misconception that kernel writing is needed; this has moved to **help**.
- **ERNIE AI Developer Challenge Announced!**: Unsloth is supporting the **ERNIE AI Developer Challenge**, offering **$3,000** in prizes for fine-tuning **ERNIE** and building the most impactful model.
   - Details can be found at the [Baidu Ernie AI Devpost link](https://baiduernieai.devpost.com/) and official Ernie finetuning notebooks (AMD ones are free) at the [X post link](https://x.com/ErnieforDevs/status/1993666389178204434).
- **Unsloth to Hit Up NeurIPS in San Diego**: Unsloth will be at **NeurIPS San Diego 2025** with limited time merch, with an **Agentic AI / RL Panel** talk with **OpenEnv** on **Tue 2nd Dec 4PM** and the **Open Source AI Reception** on **Wed 3rd Dec 6PM**.
   - The team provided a [registration link](https://linuxfoundation.regfox.com/open-source-ai-reception-2025) and reminded users to hit them up for **RL takes**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1442970987791388683)** (173 messages🔥🔥): 

> `Claude Opus 4.5, wakeword solution, MS or PhD interviews, Long context training, Humanoid stamina` 


- **Opus gives context Errors**: Members report **Claude Opus 4.5** giving errors for 100 lines of code + 200 line yaml file, with the error message, *im sorry this is beyond my context limits. Im going to XYZ*.
   - One member then asked for a *decent wakeword solution* that works in a browser, or perhaps just in *python*.
- **Job Interview: MS or PhD required?**: A member shared that they have an interview even though they don't have a MS or PhD, which was stated as a requirement.
   - Others encouraged them, explaining that companies filter out people, and *what matters is who you are and what you can bring, just be yourself and genuine during the interview that's it.*
- **Training Model with CPU offloading**: A member is training a model using their own training framework built on top of Unsloth, and asked if adapters are added to a model, does that mean both the adapter + model will be in memory, thus use more VRAM?
   - Another member provided a link to the [Unsloth Long Context Blogpost](https://unsloth.ai/blog/long-context) and explained the point of LoRA is to avoid updating all parameters.
- **Humanoid stamina**: A member asked *If you’d build a humanoid, what could you account for the stamina and other similar “human” parameters? And is it possible with current technologies to convert food into adenosine triphosphate and then electricity as efficiently as in living organisms?*
   - Another member replied *the vaste majority of the technologie exists but has not be put togeather that wat / it is obscenely expensive like hundredd millions*.
- **Kagi drops Slop Detective Game**: A member shared [Slop Detective](https://slopdetective.kagi.com/), a new game from Kagi, with the comment *Yeah, let’s fight them, ugh! 😠lol*.
   - Other members find examples are *bs*, and paddle *wrong = ai correct = human*, but one argues *much hooman text fill of error*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1442983801084055634)** (103 messages🔥🔥): 

> `IPEX vs llama.cpp Vulkan, HF model to GGUF conversion, Continued pretraining vs Fine-tuning, Qwen3 8B Fine-tuning issues, AMD GPU support for bitsandbytes` 


- ****Vulkan > IPEX** For Llama.cpp**: Users recommend using the regular **llama.cpp Vulkan** version instead of **IPEX** due to stability issues, though SYCL might offer slightly better performance.
   - It was mentioned that the **IPEX** build is *very old*.
- ****`model_type` Attribute Strikes Again** in GGUF Conversion**: A user encountered an `AttributeError: 'dict' object has no attribute 'model_type'` while converting a HF model (**Unsloth/Qwen2.5-7B-Instruct**) to GGUF using `llama.cpp`'s `convert_hf_to_gguf.py` script, likely due to file structure issues.
   - Another user shared a working directory structure for a merged Qwen3 model as reference.
- **Base Models Reign Supreme for Autocompletion**: For training a model to generate similar data (autocompletion) without question/answer behavior, it's recommended to start with a **base model** (not instruct-tuned) and perform **continued pretraining**.
   - A **Gemma-3-270M** model was suggested for experimentation, alongside a link to [Unsloth's documentation on continued pretraining](https://docs.unsloth.ai/basics/continued-pretraining).
- **Qwen3 8B Fine-Tuning Fails the Vibe Check**: A user experienced poor evaluation results after fine-tuning **Qwen3 8B**, with responses unrelated to the fine-tuning data, and experiencing the model still outputting the `thinking` prompt even with the prompt set to false.
   - It was suggested to try manual merging and saving if LM Studio replicates the issue, referencing the [Unsloth documentation](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf#manual-saving).
- ****AMD GPUs Get Bitsandbytes Boost** in vLLM Update**: The AMD documentation is due for an update to reflect the support of **Bitsandbytes 4bit quantized models** and **QLoRA** on Radeon GPUs.
   - Changes were implemented in [bitsandbytes-foundation/bitsandbytes#1748](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1748) and [vllm-project/vllm#27307](https://github.com/vllm-project/vllm/pull/27307).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1443274834170744853)** (2 messages): 

> `ERNIE AI Developer Challenge, Baidu ERNIE, Unsloth finetuning, AMD notebooks` 


- **ERNIE AI Developer Challenge Kicks Off**: Unsloth announced support for the **ERNIE AI Developer Challenge**, offering a chance to fine-tune **ERNIE** with Unsloth and win prizes.
   - The competition details can be found at [baiduernieai.devpost.com](https://baiduernieai.devpost.com/).
- **Unsloth's Finetuning Freebies for ERNIE**: Official **ERNIE** finetuning notebooks, including free ones for AMD, are available.
   - Check out the announcement on [X.com](https://x.com/ErnieforDevs/status/1993666389178204434) for access to the **AMD notebooks**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1443145738090709032)** (12 messages🔥): 

> `Evolutionary Strategies at Scale, LESA: Learnable LLM Layer Scaling-Up, Efficient Training on CPU` 


- **ES HyperScale boosts Training Throughput**: A member shared [ES HyperScale](https://eshyperscale.github.io/) which achieves a **hundredfold increase** in training throughput over standard ES for billion-parameter models at large population sizes, enabling more flexible training on any model, without worrying about gradients, and with int8.
   - Another member humorously noted, *"Training at 100x speed? That's Unsloth x 50 then"*.
- **Learnable LLM Layer Scaling-Up with LESA**: A member posted [LESA: Learnable LLM Layer Scaling-Up](https://arxiv.org/pdf/2511.16664v1), suggesting that *some sort of (nested "elastic" MoE) + (multi-token prediction) would provide a crazy inference single batch throughput leap*.
   - The paper introduces **LESA**, which predicts parameters inserted between adjacent layers using a neural network, enabling better initialization and faster training.
- **Efficient CPU Training is now Reality**: A member highlighted that with [ES HyperScale](https://eshyperscale.github.io/) realistically efficient training on CPU can be achieved, with flexible training on any model, without worrying about gradients, and with int8.
   - It was described as *"more flexible training on any model. Training without worrying about gradients. Training with int8! Realistically efficient training on CPU"*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1442968521553219684)** (371 messages🔥🔥): 

> `Haiku documentation accuracy, Cursor agent's plan markdown storage, Free Agent Review, Education discounts` 


- **Haiku models for documentation**: Members are finding that **Haiku** with documentation is 100% accurate and **Composer-1** is best for code implementation, and **Haiku** reigns supreme for speedy documentation retrieval.
   - One member also suggests using [Antigravity](https://antigravity.ai/) instead of littering repos with Markdown reports, although this may cause issues with handoff.
- **Users discuss cost of tokens and model usage**: Some users report issues with the **Opus** model being overloaded, others say it has been degraded, acting weird and less smart.
   - Some debate whether to enable on-demand usage or just buy a Pro+ plan, discussing if they should just *burn the tokens* using **Auto** and not consider token efficiency.
- **Agent review being free??**: Users notice *agent review* may be free but only on the old pricing, whereas on the new pricing is no longer available.
   - One also wonders if the teams plan have unlimited bugbot due to seeing *unlimited bugbot* on the dashboard.
- **Users frustrated with linting errors in Cursor**: A user seeks help to disable red squigglies for linting checks while keeping them for other errors, as well as allowing the extension to run `--fix` in the background on file save.
   - The user expressed frustration on why this is so hard to do in **Cursor**, as it's fairly straightforward in JetBrains' tools.
- **Agent plans not saved in Cursor**: A user asked where the markdown file for an agent plan is saved, so they can switch between computers without losing the plan.
   - The community member states that **Cursor** doesn't save the plan, so you need to manually save the markdown, and create a rule to add all plans to a directory.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1443038489829572690)** (7 messages): 

> `Triton Kernels, Partially Trainable Embedding, Logits Softmax Operation, Curriculum Learning` 


- **Seeking Frontier-Level Efficiency Gains with Triton Kernels**: A member is seeking advice on using **Triton kernels** for a unique challenge involving a *Partially Trainable Embedding* and a *Logits Softmax* operation, aiming for frontier-level efficiency gains.
   - The goal is to train a large model while freezing most of it, focusing on specific special tokens efficiently, as initial attempts with Claude yielded slow results attributed to memory bounding due to inefficient tiling and repeated data retrieval.
- **Need Partially Trainable Embeddings for Memory Savings**: A member wants to implement a *Partially Trainable Embedding* where only a range of rows above a certain index are trainable, such as **1k rows (127k to 128k) out of a 128k vocabulary**.
   - This is intended to reduce memory usage by only storing gradient outputs for the trainable rows, and is also intended to freeze most of the model while only training specific special tokens.
- **Weighted Loss with Logits Softmax**: A member is looking to implement a *logits softmax operation* that allows for weighted loss to be applied, such as **token in pos 123** having a **0.5x loss multiplier** and **token in pos 124** having a **1.5x loss multiplier**.
   - The goal is to avoid materializing all the logits by using chunking or CCE approaches, and it must work with the custom partially trainable embedding.
- **AI Labs commonly use Curriculum Learning**: A member asked if AI labs really use things like *curriculum learning* and *coreset* while pretraining LLMs.
   - Another member responded, *idk wdym by coreset, but yeah curriculum learning is pretty common in pretraining in general*.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1443070482697027595)** (5 messages): 

> `Proton vs Nsight Systems, Tensor Descriptors, Auto Tune Parameters, Tritonparse, Persistent Matmul Tutorial` 


- **Proton Profiling Tool Glitches**: A user inquired about using **Proton** for profiling, noting errors when generating chrome traces as documented, wondering if others prefer **Nsight Systems** instead.
   - Follow up discussion pointed to **persistent matmul tutorial** as example of using mnk as autotune keys.
- **Auto-Tune Parameter Quest Kicks Butt**: One member, struggling with leetcode, expressed interest in **tensor descriptors** or **auto-tune parameters** to specialize shapes.
   - They also thanked another member for suggesting **Tritonparse** as a helpful tool.
- **Persistent Matmul Tutorial**: A member suggested that the [persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html#sphx-glr-getting-started-tutorials-09-persistent-matmul-py) is an example of using **mnk** as autotune keys.
   - The tutorial guides users through optimizing matrix multiplication using shared memory and persistent kernels, providing a practical example of autotuning in **Triton**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1442978296202657833)** (17 messages🔥): 

> `GEMM with tensor cores, NVIDIA Tensor Cores performance optimization resources, BF16 matrix multiplication, CUDA implementation details, Matrix data loading strategies` 


- ****GEMM** Implementations Explored**: A member is exploring **GEMM** (General Matrix Multiplication) implementation using tensor cores and seeks advice on using **BF16** for matrices **A**, **B**, and **C** with `float` accumulators, referencing [Lei Mao's tutorial](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/).
   - The member is facing challenges with loading matrix **C** elements using `load_matrix_sync` and converting them into `float`, questioning whether **C** should initially be a `float` matrix.
- **Tensor Core Optimization Treasures Unveiled**: Members shared resources for performance optimization on **NVIDIA Tensor Cores**, pointing to similar articles and worklogs, such as [alexarmbr's work](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) and [cudaforfun's worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog).
   - One highlighted that **GPU-MODE** has a lecture for Hopper **GEMM** worklog.
- **Data Loading Dilemmas Decoded**: A member explained that `ldmatrix.b16` loads **128 bits** of data per thread without extra operations, suggesting a `reinterpret_cast` for correct data handling.
   - Another member clarified that when using `f32` or `s32` accumulators, each thread holds a pair of consecutive values within a row (**8 bytes**), while `ldmatrix.b16` splits a row into **4B** chunks, distributed over a quad of threads, suggesting the use of `float2` or reordering **B** matrix columns on load.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1443191272302055504)** (3 messages): 

> `Gradient Checkpointing, Torch Differentiation, Boolean Flagging` 


- **Looking for Torch Function to Differentiate Forward Passes**: A member inquired about a **torch function** to differentiate if the forward pass is run with or without **gradient checkpointing**.
   - The member also asked if there is a way to differentiate between the two forward passes.
- **Leveraging Boolean Flags to Differentiate Forwards**: A member suggested solving the differentiation of the two forwards with a **boolean flag**.
   - The member proposed alternating the flag in each forward pass.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1443008458713469058)** (13 messages🔥): 

> `Contributing to XLA, GPU/CUDA Benchmarking Warmup Runs, Kernel Characteristics Affecting Warmup Time, Thermal Limits in Benchmarking, nvbench thermal states` 


- **Contributors looking for ways to contribute to XLA**: A member inquired about contributing to **XLA** and sought guidance on where to begin, with an initial interest in **documentation support**.
- **GPU Warmup Run Rule of Thumb**: A member asked about a good rule of thumb for the number of **warmup runs** for **GPU/CUDA benchmarking**.
   - Another member responded that there isn’t one in raw numbers; instead, they repeat measurements until successive runs don’t change significantly.
- **Thermal limits impact long GPU runs**: Members mentioned that to benchmark steady state performance of an application running for a long time, you have to take **power draw** and **thermal limits** into account.
   - You literally have to let the GPU warm up to reach a steady temperature (which might take tens of seconds to a couple of minutes).
- **Datacenter Settings Mitigate Thermal Factors**: A member inquired whether **datacenter settings** mitigate thermal factors, and another member responded that, depending on context, this steady state might not be the correct answer.
   - They also provided a link to a [YouTube video](https://www.youtube.com/watch?v=CtrqBmYtSEk) about nvbench, which aims to get a good average across un-throttled thermal states.


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1443228985143328778)** (2 messages): 

> `jax.pmap vs jitting on single GPU, Multi vs single GPU systems` 


- **Performance of `jax.pmap` vs `jit` on single GPU**: A user inquired about the downsides of using `jax.pmap` with one device compared to jitting it directly via `jax.jit`.
- **Code portability on Multi vs Single GPU systems**: The user is writing code intended to run on both multi and single GPU systems and is considering using `jax.pmap` even when there is only one GPU to simplify the codebase.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1443268597735624804)** (1 messages): 

> `Memes` 


- **Meme of the Day Delivered**: A user delivered a meme.
   - The meme can be found [here](https://cdn.discordapp.com/attachments/1215328286503075953/1443268597483831326/1764108026112.jpeg?ex=69287420&is=692722a0&hm=d4747dea6327a6024b1c84c59c77525ee94bc0392191114d5b49c98d00bd1cd4&).
- **Another Meme Appears!**: Another meme has been posted for the amusement of the channel.
   - This meme adds to the ongoing collection of humor shared within the community.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

szymonoz: I'll be coming to NeurIPS and traveling to SF afterwards, hmu if you want to chat gpus 😄
  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1443076921348067369)** (1 messages): 

> `2bit Dequantization on Intel GPU, GPU Dequantization Methods, Torch Performance on Intel GPU` 


- **2-bit Dequantization Quest on Intel GPU**: A user inquired about a method for performing **2-bit dequantization** directly on an **Intel GPU**, noting that while quantization can be done on the CPU, dequantizing with **Torch** is slow.
   - The user seeks a faster, GPU-based alternative to **Torch** for dequantization to improve performance, illustrating a need for optimized **Intel GPU** solutions in this area.
- **Seeking Speedy GPU Dequantization**: The original poster is seeking optimized **GPU**-based alternative to **Torch** for dequantization to improve performance.
   - There is no other discussion to summarize, it remains an open question for the channel.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

aerlabs: https://x.com/aerlabs_/status/1993561244196868370
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1443365307639926834)** (1 messages): 

> `LLM initiatives, LLM Kernel Generation, Agentic Systems` 


- **Urmish Joins LLM Initiatives**: Urmish introduces themself, expressing interest in helping with **LLM initiatives**, highlighting experience in **pre-training, post-training, evaluation, agentic systems, and dataset creation**, and provides a [Google Scholar profile](https://scholar.google.com/citations?hl=en&user=-GPPICQAAAAJ&view_op=list_works&sortby=pubdate).
   - With a background in systems and performance engineering, including **kernel writing for microcontrollers, HPC, and CPUs**, they seek guidance on where to begin and inquire about subgroups focused on LLM training, prompting, or agentic harnesses for **LLM Kernel Generation**.
- **LLM Kernel Hopes to sprout**: Urmish asks about the existing subgroups to better target efforts in **LLM Kernel Generation**, **LLM training** and **Agentic Harnesses**.
   - They are hoping to use prior experience to help the community.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1443116910718881822)** (10 messages🔥): 

> `CUDA kernels, Flash Attention, MoE kernels, Linear Attention backwards, FFT conv backwards` 


- **Newcomer Pioneers CUDA and Flash Attention**: A new community member expressed their experience writing **CUDA kernels** and working with **flash attention**.
   - Another member encouraged them to contribute back via a **PR**.
- **Kernel Contributions Blossom in ThunderKittens**: Members discussed open areas for development including **MoE kernels**, **linear attention backwards**, **FFT conv backwards**, and integrations into **inference engines**.
   - They also mentioned **Pythonic wrapper explorations/tooling** to simplify development and tooling to integrate light compiler passes as welcome community contributions.
- **AMD GPU Availability Sparks Debate**: A member inquired whether the contributions were for the **main branch CDNA4 or CDNA3**, noting the difficulty in finding a GPU provider for **AMD GPUs** to build and test such things.
   - Another member clarified that it's for both, but that the original question was about TK.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1442968122725236841)** (114 messages🔥🔥): 

> `NVIDIA leaderboard submissions, nvfp4_gemv leaderboard, Personal bests, Successful submissions` 


- **NVIDIA's nvfp4_gemv Leaderboard: Submission Blitz!**: The `nvfp4_gemv` leaderboard on NVIDIA saw a flurry of activity, with numerous submissions from several users, including <@242385366873669632>, <@393188835472834560>, <@651556217315000360>, <@418996736405536790>, <@1035498877249409155>, <@1295117064738181173>, <@376454672799760384>, <@96782791567503360>, <@264466949331746826>, <@1178719962597183529>, <@434046629281267744>, <@1291326123182919753>, and <@120261963551866881>.
   - The submissions included both "Personal best" and "Successful on NVIDIA" results, indicating active optimization and testing efforts.
- **Overtaking the Podium: Second Place Achieved**: <@1035498877249409155> achieved **second place** on NVIDIA with a submission of **3.02 µs** and later another second place with **15.8 µs** on the `nvfp4_gemv` leaderboard.
   - There was discussion about a potentially fishy submission by <@1035498877249409155>, with <@264466949331746826> planning to double-check the results and mentioned, *"i gave opus 4.5 full reign with some guidance on tricks"*.
- **Optimization Race: New Personal Bests Unveiled**: Multiple users, including <@242385366873669632>, <@393188835472834560>, <@1295117064738181173>, <@120261963551866881>, <@434046629281267744>, <@1035498877249409155>, <@1291326123182919753> and <@651556217315000360>, consistently submitted "Personal best" results on the `nvfp4_gemv` leaderboard on NVIDIA.
   - This indicates an ongoing effort to optimize performance and achieve faster execution times, also <@376454672799760384>'s submission had a best of **144 µs**.
- **Entering Top 10: Users Grab Top Spots**: <@1295117064738181173> secured **8th place** with **22.7 µs**, then later **7th place** with **22.5 µs**, and <@1035498877249409155> achieved **9th place** with **23.2 µs** on NVIDIA.
   - <@1178719962597183529> reached **9th place** with **23.3 µs** and <@1295117064738181173> reached **7th place** with **22.9 µs**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1443303382357119098)** (3 messages): 

> `Factorio Learning Environment Docs, Jack Hopkins, Github Pages` 


- **Hopkins's Hotline: Factorio Docs Deployed!**: Jack Hopkins announced that the documentation for the **Factorio Learning Environment** is now live at [Factorio Learning Environment](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/index.html).
- **Noddybear thumbs up Hopkins's Docs**: Noddybear reacted positively to the announcement of the new Factorio documentation.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1443261622033322216)** (2 messages): 

> `SIMT loads, Tiled_mma documentation` 


- **SIMT Load Overheads**: SIMT loads have overheads, so *use them only if TMA is too restrictive*.
- **Tiled_mma example breakdown**: An engineer is trying to use *tiled_mma* by following the **hopper gemm cute dsl** example.
   - They tiled **sa** by **(2, 4)**, and `tCsA: ((64,16),2,4,(1,1)):((64,1),4096,16,(0,0))` is *mma atom tile (64, 256), 2 tiles along M direction and 4 tiles along K direction*.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1443312521716039854)** (3 messages): 

> `picograd, aten-like Op intermediate representation, Device runtimes` 


- **Picograd's Latest Commits**: The user shared a series of recent commits to the [picograd repo](https://github.com/j4orz/picograd), highlighting ongoing developments.
   - The commits cover various aspects, including package-level documentation, tensor implementation, evaluator design, and device runtimes.
- **Picograd's Tensor Implementation**: The user linked to picograd's `Tensor` implementation, which desugars into an **aten-like `Op` intermediate representation** [(link)](https://github.com/j4orz/picograd/blob/master/python/picograd/tensor.py).
   - The goal is to provide a foundation for automatic differentiation and GPU acceleration.
- **Picograd's Evaluator and Device Runtimes**: The user spotlighted the `evaluator(op: Op)` interpreter, which uses `Device` runtimes [(link)](https://github.com/j4orz/picograd/blob/master/python/picograd/engine/evaluator.py), and the `Device` runtimes themselves, which provide memory allocators and kernel compilers [(link)](https://github.com/j4orz/picograd/blob/master/python/picograd/device.py).
   - The user mentioned that the language and runtime will come together nicely soon, paving the way for marching across architectures.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1443297762811576440)** (3 messages): 

> `LLM Inference, NVRAR algorithm, PAT Algorithm, Bruck algorithm, Recursive doubling algorithm` 


- **NVRAR Speeds up Multi-Node LLM Inference**: The paper [LLM Inference Beyond a Single Node](https://arxiv.org/abs/2511.09557) introduces **NVRAR**, a hierarchical all-reduce algorithm based on recursive doubling with NVSHMEM, achieving **up to 1.9x-3.6x lower latency** than NCCL for message sizes between **128 KB and 2 MB**.
   - Integrated into YALIS, **NVRAR** achieves **up to a 1.72x reduction** in end-to-end batch latency for the **Llama 3.1 405B model** in multi-node decode-heavy workloads using tensor parallelism.
- **PAT Algorithm for All-Gather and Reduce-Scatter Operations**: The paper [PAT: a new algorithm for all-gather and reduce-scatter operations at scale](https://arxiv.org/pdf/2506.20252v1) discusses the shortcomings of the **Bruck** and **Recursive doubling algorithms** in practice due to their final steps requiring large data transfers to distant ranks.
   - The last step sees every rank send half of the total size to its most distant rank, and *on large fabrics, that last step frequently runs many times slower than the theory due to static routing, or due to higher levels of the fabric being tapered*.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1442975553492750386)** (159 messages🔥🔥): 

> `CuTeDSL packed FP16, eval.py issues, cudaStreamSynchronize(), LLM-only challenges, sfa_permuted purpose` 


- **CuTeDSL gets packed FP16 instructions**: Members provided [code](https://github.com/gpu-mode/reference-kernels) to use packed FP16 instructions in CuTeDSL, because the normal CuTeDSL doesn't offer these via nvvm.
- **Eval Script Faces Scrutiny**: Users reported that the `eval.py` script in the GPU MODE competition can produce highly variable results, with timing differences of up to 50% even when uploading the same script multiple times, some speculate a slow runner with id **105881**.
   - The erratic nature of the script raises concerns about the accuracy and reliability of the leaderboard timings with a suggested submission threshold of **25**.
- **Streams add overhead**: A member found that playing around with streams causes synchronization issues, and stated that  `cudaStreamSynchronize()` adds massive overhead on properly implemented solutions.
   - Another member noted that events add about **4 us** of measuring overhead.
- **LLM-Only Approach Explored**: Some participants are trying an "LLM-only" approach, using models like **Gemini 3.5 Pro** and **Opus 4.5** to generate code, but some are guiding the LLM more than others.
   - One user noted *gemini 3.5 pro and opus 4.5 are complete game changers... they make gpt-5.1 look like llama-7b*.
- **sfa_permuted Cracking the Code**: A user finally realized the purpose of **sfa_permuted** is related to the tcgen instruction which makes it easier to make the thing with this layout.


  

---


### **GPU MODE ▷ #[hf-kernels](https://discord.com/channels/1189498204333543425/1435311035253915840/1443041374185328751)** (5 messages): 

> `Metal Kernels Release, MacOS Compatibility Issues` 


- **Metal Kernels Delayed**: A member inquired about the release of **metal kernels**.
   - No release date was given.
- **MacOS Compatibility Limited**: A member questioned why the [kernel-builder](https://github.com/huggingface/kernel-builder/blob/main/docs/metal.md) only supports **macOS 26**, which reduces compatibility with **M1** chips and older versions of macOS.
   - The member was *confused why everything done for the apple torch ecosystem is done in a way that makes it worse*.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1443019750757236738)** (8 messages🔥): 

> `7x Laundry Folding Robot, No-Action Filtering, Qwen3-VL Optimization, Classic Binning vs FAST Tokenizer` 


- ****7x** Laundry Robot Debuts!**: **7x** is offering a **3k** laundry folding dual arm system, as seen on their [YouTube channel](https://www.youtube.com/@usmanroshan8740), providing *low-cost robots vibes* with **24 hour support** from founders and engineers.
   - Doubts were cast on the arms' durability for real-world jobs, with a member contrasting their support model against that of *Google Robotics*.
- **No-Action Filtering is Crucial for VLAs**: A member learned that **no-action filtering** is important for VLAs, showcasing the difference between a no-idle filter and a with-idle filter in a [visual comparison](https://cdn.discordapp.com/attachments/1437390897552818186/1443153483556716666/image.png?ex=6928b1aa&is=6927602a&hm=bf26405dca31cd342d33762114dc18ad626339bf92ccb31ee0cb0c1eb501087e).
   - An image illustrating the impact of **idle frame analysis** showed that active frames constituted **78.8%** of total frames analyzed.
- **Qwen3-VL's Optimization Hurdles**: A **2B model** feels slow, especially during inference, rendering it unfeasible for running RL, and a member planned to investigate optimized forward passes for **Qwen3-VL**.
   - No further details were provided.
- **Tokenizer Faceoff: Classic Binning vs FAST**: Members are testing **classic binning** vs. **FAST tokenizer**, but the complex compressed tokens generated by **FAST (DCT+BPE)** may delay the model's ability to produce reliably valid sequences.
   - The poster expressed doubt whether this would be a good basis for RL, therefore they are simultaneously trying a simpler variant with disentangled joints and simple quantization.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1442968425033892042)** (263 messages🔥🔥): 

> `ChatGPT Biases, Nano Banana Pro, Commercial Use of AI Generated Images, GPT 5.0 mini, OpenAI UI Design` 


- **ChatGPT allegedly biased towards Left-Wing data**: Members discussed whether **ChatGPT** is trained on liberal and politically left-wing data, with potential causes being progressive viewpoints in training data and biases of human raters in **Reinforcement Learning with Human Feedback (RLHF)**.
   - One member argued that the model's need to *fussy foot around questions* compromises its reliability.
- **Nano Banana Pro Unleashes Comic Creation**: Users are creating comics with **Nano Banana Pro**, praising its power, ability to generate images quickly, and the high-quality results, and are excited about it's ease of use in generating [comic pages](https://cdn.discordapp.com/attachments/998381918976479273/1443038766087536751/image.png?ex=6928ef94&is=69279e14&hm=d88b0f693975c1e756c3352c689566bda68503635084432e547cc6585d126e83&).
   - Members shared worries about the model being *lobotomized*.
- **AI Art raises Commercial Copyright and Ethical Quandaries**: Members debated the commercial viability and copyright implications of using AI-generated images from **Gemini**, noting that while Google doesn't explicitly prohibit commercial use, the legal status depends on whether the content is copyrightable, with cultural bias in AI art being a significant concern.
   - One member said that *if the anti AI people want to do something they ought to start drawing and making art*.
- **GPT-5.0 Mini Feels Like a Downgrade**: Members are not happy about **GPT-5.0 Mini**, stating it is a *downgrade*.
   - They are annoyed with the incessant begging for **Sora 2** which they haven't even used.
- **UI/UX of OpenAI cater to a Neurotypical Audience**: A member argued that **OpenAI's UI** is not designed for neurodivergent thinkers, requiring too many steps and not fitting people with complex thinking.
   - Others in the channel argued that the **UI's are terrible for everyone** and are explicitly designed to cater to people with executive dysfunction, in particular the Mac app.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1443016278356000880)** (10 messages🔥): 

> `GPT 5.1, GPT 4.1, Chat reference memory, Anime writing` 


- **User Praises GPT 5.1 for Anime Storytelling Prowess**: A user highlights that **GPT 5.1** is currently the best model for anime or story writing due to its ability to remember character designs and previous context.
   - The only complaint is the strict **safety net and guardrails** that prevent writing anime-style violence.  The user shares that they've used **GPT 4.1** for a year, but sometimes it misses character designs.
- **Chat Reference Memory Issues Debated**: A user asks whether anyone else is having issues with **chat reference memory** in **GPT models**.
   - Another user poses the question of whether **GPT 5.1** is better than **GPT 4**, suggesting it depends on the specific use case.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

mx_fuser: <@1256251788454268953>
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

mx_fuser: <@1256251788454268953>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1442974507999695070)** (46 messages🔥): 

> `Unsupported API Endpoints in LM Studio, Image Captioning Issues with LM Studio, Vision Models, ROCm 7 Update for RDNA 3, Mint Opportunity Partnership with OpenSea` 


- **API Endpoint Troubleshooter Solves Issue**: A user encountered an error with **unsupported endpoints** (POST /api/v1/generate) on their local server, but resolved it themselves after posting in the channel.
   - The user was pointed to the [LM Studio REST API documentation](https://lmstudio.ai/docs/developer/rest/endpoints), and realized the endpoint was invalid.
- **Channel Error Ruins Image Captions**: A user reported a **"Channel Error"** when trying to caption images with **LM Studio**, experiencing a 100% failure rate after a Windows and antivirus update, even though it worked previously.
   - The user switched from **llava-v1.6-34b** to **Gemma 3**, which solved the problem giving 100% success rate; the suggestion was offered as potentially model dependent or the issues might be related to Flash Attention being enabled by default.
- **Flash Attention Glitch in Some Models**: It was suggested that the user's issue may be related to **Flash Attention**, which is now enabled by default in recent LM Studio versions and can cause some models to not function correctly.
   - Users were encouraged to share screenshots of their runtimes view and check for non-English input/output, with a suggestion to run `lms log stream` for more detailed error messages.
- **GPT OSS 20B Blazes with Speed**: A user shared an image showcasing the speed of the **gpt-oss-20b** model, linking to a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1p7ghyn/why_its_getting_worse_for_everyone_the_recent/) and it was mentioned that the information in that Reddit post was something that a few people in the channel might relate to. 
- **Mint Opportunity Plunges User into OpenSea**: A user announced a free **Mint opportunity** in partnership with **OpenSea**, inviting members to participate through a provided link.
   - Another user quickly pointed out that the given invitation would fail in a real academic setting for reasons explained in detail, pointing out the difference in how a human would rate the work vs how the bot would.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1442978690236420338)** (217 messages🔥🔥): 

> `Q8 Cache, GPU Fans at 0% During Inference, Memory Pricing Issues, DLSS and RT Testing, Hardware Devaluation` 


- **Q8 Cache Configuration Conundrums**: Members discussed using **Q8 cache**, with one mentioning that a specific user (*<@96768590291664896>*) knows how to explain why the digits don't align for **Q6 KV**.
- **GPU Fans Taking a Break During Inference**: One user noticed their **GPU fans** were at **0%** during inference, initially raising concern, but later clarified it was a normal behavior for their **MI50** and sometimes their **4070 TiS**.
   - The user noted that once the context is fully written, the GPU "takes over" and power draw increases.
- **Hardware Devaluation Debate**: A user shared a photo of their Windows boot in recovery, joking that an **850W power supply** wasn't cooked after all, calling it *an improvement*.
   - The user initially suspected a power supply issue but then suspected the CPU's thermal paste was the issue.
- **Potential CPU Fire Averted?**: Users cautioned against potentially frying components and advised testing the **CPU** and **RAM** on a cheap motherboard, suspecting a potential fire.
   - Another user found a bent CPU pins on MB and smelt the CPU to check for smoke but determined everything was fine after cleaning the thermal paste.
- **Bifurcation Breakthroughs**: A user realized their **X570 AORUS ELITE WiFi** motherboard supports **PCIe bifurcation** on the primary x16 slot, allowing it to be split into configurations like **8x/8x** or **8x/4x/4x**.
   - Another user added that with bifurcation you use a [SlimSAS MCIO adapter](https://www.amazon.com/dp/B0DZG8JVG2) to split the x16 slot into dual x8 slots when x8x8 is enabled.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1442968437281394811)** (2 messages): 

> `Color Picker Issues, RapidaAI Open Source` 


- **Color Picker Bug Bites Users**: A user reported that the **color picker** is a *bit funky* and **offset** for the theme palette override.
- **RapidaAI Goes Open Source**: **RapidaAI**, a production-ready voice AI platform, announced they are releasing their **open source code** [here](https://rapida.ai/opensource?ref=openrouter).
   - The company observed that voice AI vendor bills kept growing without improvements to customer experience, with companies paying an extra **$0.05–$0.15 per minute** to rent someone else’s stack, so they built Rapida to flip that model.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1442968164982980791)** (196 messages🔥🔥): 

> `Opus Overload, Model Fallback Bug, Deepseek R1 Model Gone, Meganova Chat Buzz, OpenRouter Pricing and Features` 


- **Opus experiences Overload Outage**: Users reported that **Opus** was overloaded again, right when things were getting hot.
   - Some members joked *You'd think they'd have better rate limit/load balancing eh*, while others were understanding and mentioned *Small company pls understand*.
- **Model Fallback feature faces Flak**: A member reported a bug in the [model fallback logic](https://openrouter.ai/docs/guides/routing/model-fallbacks) where a **404 error** from the primary model prevented the fallback from working, rather than falling back to secondary models.
   - The member stated *Im about to migrate to openrouter for an enterprise application , there's no space for real or not real model . if the fallback logic breaks for such simple use case . there might be more issues*.
- **Free Deepseek R1 Model Ripped**: Members noted the free **Deepseek R1** model is no longer available.
   - One member lamented losing the model *That's stupid. I used it with a chutes api key because using the model via chutes shows the think process and I can't stand it.*
- **Meganova Chat creates Mass Movement**: Members discussed the upcoming launch of **Meganova Chat**, a platform for managing AI chats and characters, with one member describing it as a *clean, fast place*.
   - One member responded *I'm seeing a lot of positive buzz around Meganova Labubu Chat! i'm considering learning more about it*, while others offered comedic parodies of promotional messages.
- **OpenRouter Boasts Beneficial Basics**: A member highlighted the benefit of OpenRouter's normalized interfaces for various providers.
   - They mentioned the ability to *switch from e.g. GPT 5.1 to Opus 4.5 instantly without having to parse all of the anthropic changelog is very nice*, despite the fact that there is a **5% premium on credit purchases**.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1443327854355021915)** (2 messages): 

> `` 


- **No New Models Discussed**: There were no discussions or information about new models provided in the given messages.
- **Channel Indication**: The prompt indicated the messages came from the 'new-models' channel on OpenRouter, but contained no actual model-related content.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1443289637945868420)** (5 messages): 

> `Arrakis AI model, Text-to-Video Leaderboard, Kling 2.5 Turbo, Google Veo 3` 


- **Arrakis AI Still Looks Yellow-ish**: A member commented on [an image from Arrakis AI](https://x.com/arrakis_ai/status/1993644406159917533), observing that *it still does look yellow-ish*.
   - They speculated that *they just added a colour adjustment layer before sending the image to the client*.
- **Text-to-Video Leaderboard crowns a new king**: A member shared a link to the [Artificial Analysis Text-to-Video Leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video), highlighting the top performers.
   - The leaderboard showcased **David** in first place, followed by **Google's Veo 3** as the runner-up, and **Kling 2.5 Turbo 1080p** in third place.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1442995229019078800)** (1 messages): 

> `Psyche Office Hours` 


- **Psyche Team Holds Office Hours**: The team behind **Psyche** will hold an Office Hours session next **Thursday 12/4, at 1PM EST** in the Events channel.
   - Users can join the [Discord event](https://discord.gg/nousresearch?event=1442995571173625888) to participate.
- **Dummy Topic**: This is a dummy topic to satisfy the minimum item requirement.
   - It adds a second entry to the topicSummaries array.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1442973558631301356)** (146 messages🔥🔥): 

> `Suno Warner Music Partnership, Data vs Compute Cost, Blackwell Architecture, Z-Image Model, AI Disclosure on Steam` 


- **Suno Teams Up with Warner Music, Sparks Debate**: Suno's [partnership with Warner Music Group](https://www.wmg.com/) raises questions about the future of AI-generated music and its impact on the music industry.
   - A member noted that while some **Suno** songs are indistinguishable from human-created music, many others are easily identifiable as AI-generated, leading to conflicting feelings about its potential and drawbacks.
- **Data Dollars Dwarfed by Compute Costs**: A member pointed out the disparity between spending **$2k on data** versus **$32 million on compute**, highlighting the resource-intensive nature of AI model training as seen with [Udio](https://www.udio.com/) and [Suno](https://www.suno.ai/)
   - This shift towards prioritizing compute may significantly narrow future research avenues, especially access to high-quality opt-in training data.
- **Blackwell's Bottleneck: INT/FP Mixing Mayhem**: Mixing **INT** and **FP** workloads on **Nvidia's Blackwell architecture** can severely degrade performance due to its unified scalar pipeline, which can only run one type of operation per cycle per core.
   - The best practice is to keep each kernel either **FP-only** or **INT-only** to avoid a **30-50% performance hit** caused by constant cache thrashing and reloading of code.
- **Z-Image Model Zooms onto Modelscope**: The **6B Z-Image model** has been released on [Modelscope](https://modelscope.cn/models), with its Hugging Face page expected to follow, offering a cinematic aesthetic despite its small size.
   - It leans more cinematic in aesthetics and has a distilled version available for faster inference.
- **Steam's AI Disclosure Debated by Devs**: A discussion arose regarding Steam's AI content disclosure policies, with Epic CEO Tim Sweeney arguing that AI disclosures should only apply to 'art' and not games.
   - While Sweeney views AI disclosures as unnecessary, some argue they inform consumers about the potential impact of **AI-generated content** on their gaming experience, especially in areas like voice and art.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1443241275536179321)** (2 messages): 

> `LLM benchmarks, pre-training data contamination, private benchmarks` 


- **LLM Benchmarks face Pre-Training Data Contamination**: A member inquired whether **LLM benchmarks** ensure models haven't seen problems during pre-training to avoid skewed results like models solving problems simply from memorization.
   - Another member responded that benchmarks *don't always* account for this, although some providers maintain **private benchmark versions**.
- **Overcoming Contamination in Benchmarks is Challenging**: It was noted that once a benchmark is used for model testing, it can technically be used for training too, creating a challenge in maintaining benchmark integrity.
   - Suggestions to mitigate this included using a **large, private dataset** and/or questions that would be **hard to memorize**.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1443020796242165921)** (2 messages): 

> `History of Information Retrieval, RAG, Library of Alexandria` 


- **Lecture Traces Information Retrieval History**: A lecture traces developments in **information retrieval** from the **Library of Alexandria** to **RAG**, presented in a [YouTube video](https://youtu.be/EKBy4b9oUAE).
- **Teknium Hypes Lecture**: Teknium expressed hype and intent to check out the lecture.
   - No further details were given.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1443067067992506462)** (81 messages🔥🔥): 

> `Hallucinations in Multi-Stage LLMs, AI and Collaborative Work, LLMs as Golden Retrievers, Verifying AI Claims, AI fact checking misinformation` 


- **Hallucinations in Multi-Stage LLMs Still Count as Hallucinations**: When discussing hallucination during the multi-stage LLM process, a member said that it *is a hallucination of the component system which generated it*, even if corrected by the Chain of Thought pipeline.
   - They added that *humans hallucinate and correct themselves like that all the time*, and they shared [a paper on LLM hallucinations](https://arxiv.org/abs/2509.04664).
- **LLM's and collaborative work**: A member sought feedback on a collaborative work with AI, focusing on long-form reasoning and mirror learning, they asked for advice to verify the soundness of their reasoning process.
   - They shared their [Causality Trilemma project on GitHub](https://github.com/BigusUk/Causality-Trilemma), which resulted in *a clear understanding of my own cognitive style — how I identify contradictions, refine assumptions, and build structural patterns out of questions*.
- **LLMs as Sophisticated Golden Retrievers**: Multiple members compared LLMs to golden retrievers, emphasizing their tendency to please users even if it means providing incorrect or misleading information, especially chatbots like **ChatGPT**, **Claude**, **Gemini**, and **Grok**.
   - A member shared [a YouTube video](https://www.youtube.com/watch?v=VRjgNgJms3Q) to highlight how LLMs might generate outputs without genuine understanding or logical consistency.
- **LLMs Can't Help You if You Don't Know Anything**: It was said that the only time people have produced any significant work with AI models is when they're already an expert in the given field.
   - One member linked to a [LessWrong post](https://www.lesswrong.com/posts/rarcxjGp47dcHftCP/your-llm-assisted-scientific-breakthrough-probably-isn-t) recommending steps you can take before believing your LLM-assisted scientific breakthrough.
- **Fact Checking Isn't Helped by more LLMs**: One member said that using multiple LLMs does little to help the situation because they have very similar propensities for hallucinating false information
   - They cautioned against replying to the poster with misleading or incorrect advice about how to fact check LLMs.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1442998192856895561)** (37 messages🔥): 

> `SGD shuffling, PIQA paper typo, Emergent Misalignment paper replication, AI for Drug Discovery` 


- **SGD Shuffling Sparks Debate**: Members debated the merits of shuffling data every epoch in **SGD**, with one member arguing that *shuffle once* should always be better than **IID**, contrary to known results about **SGD**.
   - Another member countered that practice matters more than proofs due to the non-convex nature of optimization surfaces, noting that **IID** can lead to increased variance and data revisits, but shuffling every epoch balances noise and structure.
- **PIQA Paper's Portuguese Gaffe**: A member humorously pointed out a potential typo in the new **PIQA** paper, where Portuguese was listed as an Eastern European language, attaching [an image](https://cdn.discordapp.com/attachments/747850033994662000/1443007174560448702/pt.png?ex=6928d228&is=692780a8&hm=e02447fccbf06df2a5add1bb8af742340f3158641951e49a9755337dd7e89e1c) for reference.
   - The paper's author confirmed the error and promised to correct it.
- **Parallel MLP and Attention Performance**: A member inquired whether parallel **MLP** and **attention** (**GPT-J** style) are inferior to alternative implementations.
   - A member shared a personal datapoint noting past instability issues attributed to prenorm style interactions rather than the underlying parallel execution technique itself, while alluding to the success of *shortcut moe* as a relevant comparison.
- **Emergent Misalignment Revisited, JSON Trap Unveiled**: A member released a replication and extension of the "Emergent Misalignment" paper, testing **Gemma 3** and **Qwen 3**, finding open-weight models surprisingly robust to insecure fine-tuning (0.68% misalignment), but identifying a format-dependent vulnerability with **JSON** halving the misalignment rate (0.96% vs 0.42%).
   - The member released the [full dataset](https://huggingface.co/datasets/thecraigd/emergent-misalignment-results) and [code](https://github.com/thecraigd/emergent-misalignment) for reproducibility, speculating that **JSON** restrictions reduce a model's degrees of freedom to refuse harmful requests, as discussed in [this blog post](https://www.craigdoesdata.com/blog/the_json_trap/).
- **AI for Drug Discovery Resources Sought**: A member requested pointers to educational resources for gaining an overview of the **AI for Drug Discovery** space, seeking information on architectures, open problems, and the status quo.
   - Another member suggested reviewing various surveys available via [Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ai+for+drug+discovery+survey&btnG=), while another pointed to the **Zach Lipton** startup.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

junktown_24268: https://papers.cool/arxiv/2509.24406 - section 3, pictures in 5.1 etc etc
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1443004435272826900)** (69 messages🔥🔥): 

> `Claude Code’s upgraded Plan Mode, DeepMind Documentary, Jeff Dean’s 15-Year ML Retrospective & Gemini 3.0, AI Generated Slides, OpenAI vs Claude` 


- **Claude's Code's Plan Mode Goes Parallel**: Sid highlights a major overhaul of **Claude Code’s Plan Mode**: multiple exploring subagents now spin up in parallel, generate competing plans, ask clarifying questions, and let users edit the saved plan file with `/plan open` ([source](https://xcancel.com/sidbidasaria/status/1993407762412536275?s=46)).
   - The community loves the higher one-shot success but wants faster UX, an “ask-only” option, model-picker (**Opus vs Sonnet**), and less verbose replanning, according to further threads ([thread 1](https://x.com/sidbidasaria/status/1993407765558251657?s=46), [thread 2](https://x.com/sidbidasaria/status/1993407771438727356?s=46)).
- **Thinking Game Documentary on DeepMind Origins Released**: Members watched the free full movie documentary, **The Thinking Game**, which explores the origins of DeepMind, now available on [YouTube](https://www.youtube.com/watch?v=d95J8yzvjbQ).
   - Viewers called it *great* and said the movie *really makes you want Demis to win the AGI race*.
- **Jeff Dean's AI Retrospective and Gemini 3.0**: **AER Labs** recaps **Jeff Dean’s Stanford talk** tracing **15 yrs of AI progress**—from hand-coded 90s gradients to **Gemini 3.0** solving IMO problems—powered by scale, better algos (**TPUs, Transformers, MoE, CoT**) and hardware, plus demos of low-code ‘Software 3.0’ and visual reasoning ([source](https://xcancel.com/aerlabs_/status/1993561244196868370)).
- **Claude Generates Powerpoint Slides**: A member tried out **Claude's new powerpoint skill** and said *it was quite nice*, pointing it at a company styleguide and blog post for info & a high-level narrative to make 10 near perfect slides.
   - They shared a [screenshot](https://cdn.discordapp.com/attachments/1443329209853542472/1443345290806689822/Screenshot_2025-11-26_at_1.57.20_PM.png?ex=6928bb8d&is=69276a0d&hm=f5b43c6358ea3e7eadc16e787d5ae2a5afe37a7dca33f19c0fe32ceaffe726d0&) of the generated slides. Members also discussed **Nano Banana Pro** in Google Slides.
- **ChatGPT Pro vs Claude**: Members discussed the value of **ChatGPT Pro** vs **Claude**, noting that **ChatGPT** is awesome for general research, has much better **Codex rate limits**, better for non **ts/js/py**, and has higher value if you use pulse, atlas, sora, codex cloud etc.
   - However, members say **Claude** is always pushing boundaries, its models are better trained to use tools, its frontend UX and UI is really good, and its cli readability/typography/font hierachy makes it much easier to understand.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1443323891534528716)** (2 messages): 

> `SOTA Vision, RF-DETR Paper, NeurIPS, Dev Writers Retreat 2025` 


- **RF-DETR Paper Authors Host SOTA Vision Special**: The authors of the **RF-DETR paper** are hosting a special event for those keen on **SOTA Vision** [here](https://luma.com/c1rqkxzl).
- **NeurIPS Signups Reminder**: There is a reminder to sign up for the **NeurIPS** tag and post related papers, discussions, meetups and questions in the relevant channel.
   - The organizers will be there later in the week.
- **2025 Dev Writers Retreat Accepting Final Signups**: The **2025 Dev Writers Retreat** is hosting after **NeurIPS** in Sandiego, and they are taking their last signups this week [here](https://lu.ma/dwr2025).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1442969919611211867)** (31 messages🔥): 

> `Black Forest prompting guide, Wisprflow new funding, SGLang diffusion, Whisper Thunder vs VideoGen, AI Image Realism Showdown` 


- **Whisper Thunder Dethrones VideoGen**: The ML community is buzzing about **Whisper Thunder**, a new #1 text-to-video model, which has surpassed **VideoGen** in the latest Artificial Analysis rankings - see [details here](https://xcancel.com/soumithchintala/status/1993694517489537105?s=46).
- **Nano Banana Pro's Realism Sparks Debate**: A comparison of AI-generated images from **Grok 4.1**, **ChatGPT-free**, **Google Nano Banana**, and **Nano Banana Pro** revealed that Nano Banana Pro produces images *"indistinguishable from reality"*, as shown [here](https://xcancel.com/romainhedouin/status/1993654227399475347?s=46).
- **OpenAI's Image-Gen Upgrade Has Mixed Reception**: Users discovered that OpenAI quietly updated its image generation model, which lead to reactions range from praise for higher quality to criticism over poor multilingual support, inconsistent scene-to-scene references, and continued saftey guardrails as shown [here](https://xcancel.com/arrakis_ai/status/1993644406159917533?s=46).
- **FLUX 2 Pro Boasts Improved Visuals**: **FLUX 2 Pro** delivers a major quality leap over **FLUX 1 Pro**, eliminating the *"plastic"* look and providing greater detail fidelity, as demonstrated in a side-by-side comparison [here](https://xcancel.com/iamemily2050/status/1993477498940899366?s=46).
- **Nano Banana Pro Enables Fraud**: **Nano Banana Pro** can create near-perfect counterfeit receipts, KYC documents, and passports in one prompt, causing alarm over potential scams and fraud, which users debate [here](https://xcancel.com/deedydas/status/1993341459928694950?s=20).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1443006329630363669)** (61 messages🔥🔥): 

> `Information Retrieval History, Genesis AI platform by Department of Energy, Curriculum Learning for Pretraining LLMs, MIT Study on AI Replacing Jobs, Trumpcoin Protocol for Zero Knowledge Proofs` 


- **Lecture on Information Retrieval Stretches from Alexandria to RAG**: A member shared a [YouTube lecture](https://youtu.be/aR20FWCCjAs?si=wmNYCsqPp7Le8FWe) on the history of **information retrieval**, tracing developments from the **Library of Alexandria** to **RAG**.
   - Some expressed interest in attending a paper discussion, while others referenced [a walkthrough video](https://youtu.be/5X9cjGLggv0?si=ZF85m9AssbQw8u75) by a **neuroscience PhD** with a machine learning dissertation.
- **US Department of Energy Eyes National AI Platform**: The **Department of Energy** plans to build a national AI platform on top of U.S. supercomputers and federal science data.
   - The platform aims to train scientific foundation models and run AI agents + robotic labs to automate experiments in various fields such as **biotech, critical materials, nuclear fission/fusion, space, quantum, and semiconductors**.
- **Debate on Curriculum Learning Techniques for LLM Pretraining Heats Up**: Members discussed the use of curriculum learning and coreset techniques during **LLM pretraining**, with one member questioning potential biases introduced by non-random sampling.
   - They cited [the Olmo 3 paper](https://allenai.org/blog/olmo3) and [the OLMo paper](http://allenai.org/papers/olmo3) as reference, clarifying that **curriculum learning is beneficial for language model pre-training**, as long as a more model-centric notion of difficulty is adopted, according to [this paper](https://arxiv.org/abs/2508.15475v2).
- **AI Already Replacing US Workforce**: A [CNBC article](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html) states that an MIT study finds AI can already replace **11.7%** of the U.S. workforce.
   - Discussion ensued about the methodology, referencing the [Iceberg Index](https://iceberg.mit.edu/) and the corresponding [paper](https://arxiv.org/abs/2510.25137), with skepticism about trusting LLMs to determine if other LLM tools can automate jobs.
- **Tensors on Trumpcoin Protocol for ZKP**: A member joked about sending all the tensors with **zero knowledge proofs** on the **trumpcoin** protocol.
   - They added that all the **Epstein documents** will be released with **zero knowledge proofs**, proving it’s a witch-hunt, while protecting Epstein’s victims.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1443011390628696166)** (24 messages🔥): 

> `Adobe AI summaries, LLM Summarization Limitations, ADHD and Autism in AI/CS, Posting papers without understanding` 


- **Adobe AI Summaries: The Devil's Bait?**: A member jokingly suggested that **Adobe's AI summaries** might be leading to issues, referencing an [attached image](https://cdn.discordapp.com/attachments/1045297868136779846/1443020193000456243/Adobe_Vermin.png?ex=6928de48&is=69278cc8&hm=128c6461c705032d5b88293eedae078353ef799ddbd74a2b9e1a8521561a6dbf&).
   - Another member mentioned, *"I dislike it because they almost always use much worse models. You get infinitely better results if you paste the PDF into ChatGPT, Claude, Gemini, etc."
- **LLMs Struggle to Summarize High-Density Info**: Members shared experiences that **LLMs** often fail to grasp what's important in summarization, especially with high-information density texts.
   - One member stated, *"Everyone has been talking about LLMs being these great summarizers. But they really aren't in my experience because they don't grasp what's important and what can be discarded.*"
- **ADHD and Autism in Tech: A Hot Topic**: A member suggested a connection between curiosity, **ADHD**, and **autism** in understanding papers, leading to varied reactions.
   - In response, it was asserted that having such conditions doesn't necessarily dictate specific actions, with multiple members sharing their own diagnoses of **ADHD** and suspected **Asperger's**.
- **Curbing the Paper Flood: A New Rule Proposal**: Concerns were raised about a user posting numerous papers without demonstrating sufficient understanding, leading to a proposal for a new rule.
   - The rule would restrict paper recommendations to those with significant positive feedback or those posted to a specific channel, aiming to filter out noise and ensure relevance.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1442987137413943488)** (6 messages): 

> `Nano Banana Pro, Tencent Hunyuan, MAGA pushback on AI datacenters, AI replacing US workforce` 


- **Tencent releases Hunyuan model!**: Tencent recently released their **Hunyuan model**, as showcased in [this video](https://hunyuan.tencent.com/video/zh?tabIndex=0).
- **MAGAs oppose AI datacenters**: Some MAGA supporters are now pushing back against **AI datacenters**, as discussed in [this YouTube video](https://youtu.be/9_-oDkSWKMc?t=28).
- **MIT Study: AI to Replace 11.7% of US Workforce**: According to an **MIT study**, AI can already replace **11.7%** of the US workforce, per [this CNBC article](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1442972756483244207)** (19 messages🔥): 

> `Hugging Face Inference API, Christmas gift drop, Error in Hugging Face, Genesis Mission, PDF reader model for LLMStudio` 


- **Inference API Grayed Out?**: A member asked for advice on enabling the **Hugging Face internal inference API** for their uploaded model, noting that the inference option is currently grayed out in the UI, as shown in the [attached image](https://cdn.discordapp.com/attachments/879548962464493622/1443040959901204530/image.png).
- **Donation-Negotiation-Collaboration (DNC) Markdown**: A member shared what they indicated might be their last **Christmas gift drop**, including a [DNC.md file](https://cdn.discordapp.com/attachments/879548962464493622/1443110237165846629/DNC.md) expressing uncertainty about its usefulness and expressing hope that it might benefit others.
- **Comfy ComfyUI questions**: In response to a question about running **GGUF text-to-image models locally**, a member suggested [ComfyUI](https://github.com/city96/ComfyUI-GGUF) or [koboldcpp](https://github.com/LostRuins/koboldcpp/).
- **LM Studio PDF Teacher**: A member inquired about a model for **LLMStudio** capable of reading a **PDF** file and answering questions, and another member suggested that any instruct model **LLM** should work, using **LM Studio's** built-in RAG.
   - They also shared a link to the [LM Studio models page](https://lmstudio.ai/models) and [Hugging Face models page](https://huggingface.co/models?apps=lmstudio&sort=trending).
- **Spanish Text Dataset Quest**: A member requested a large, high-quality **Spanish text dataset** for a **MoE language model project**.
   - Another member provided links to a [Spanish dataset](https://huggingface.co/datasets/John6666/forum2/blob/main/spanish_es_dataset_1.md) and related [Discord channels](https://discord.com/channels/879548962464493619/1205128865735770142).


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

aboodj_: epic
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1443099000638144612)** (8 messages🔥): 

> `RapidaAI Open Source, French Books Dataset, AI Sci-Fi Short Film` 


- **RapidaAI goes Open Source**: RapidaAI, a **production-ready voice AI platform**, is now [open-source](https://rapida.ai/opensource?ref=hf) to give users control over their voice AI and avoid extra vendor costs.
   - The company observed that teams were paying an extra **$0.05–$0.15 per minute** to rent someone else’s stack, costing them six figures annually.
- **French Classic Books Dataset is Created**: A member created and shared a [dataset of public domain French books](https://huggingface.co/datasets/Volko76/french-classic-books) available on Hugging Face.
   - Also, there's a version with only the **conversations** in the books ([here](https://huggingface.co/datasets/Volko76/french-classic-conversations)) designed for instruction purposes.
- **AI Sci-Fi Short Film drops**: A member showcased an AI-generated science fiction short film titled *Tales of the Sun - Céline* on [YouTube](https://www.youtube.com/watch?v=_F0cXXSivpU&feature=youtu.be).
   - The creator spent **two months** creating the film and is seeking feedback from the community.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1443094266586071050)** (3 messages): 

> `Chunking, GNN presentation, Structured data` 


- **Chunking's impact is small**: A member expressed gladness that **chunking** doesn't matter that much.
   - *For unstructured data you won't see much difference due to limited edge cases*.
- **GNN Presentation incoming**: A member is planning a presentation on **GNNs**, starting with **AlphaFold 2 and 3**.
   - The exact topic is still undecided due to ongoing research.
- **Structured data is valuable**: A member suggested trying for **structured data** in a blog.
   - They noted that for unstructured data, differences might be limited due to edge cases.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

dodrawat: let's connect
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1443364763496349777)** (2 messages): 

> `Mojo repo, Copybara, Repo Sync` 


- **Modular synchronizes repos with Copybara**: Members discussed how Mojo keeps its internal and external repos synchronized, and one member confirmed they use [**Copybara**](https://github.com/google/copybara).
   - **Copybara** manages the internal private repo with the external open-source repo.
- **Copybara manages internal & external repos**: **Copybara** is used to manage the internal private repo and synchronize it with the external open-source repo.
   - This ensures that changes and updates are consistently reflected across both repositories.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1443146827804311684)** (20 messages🔥): 

> `MAX examples for newbies, MAX written in Python, Mojo API in MAX, Migrating Python MAX code to Mojo MAX, Performance gains in MAX with Mojo` 


- **MAX Newbies Seek Examples**: A member asked for small examples to learn about **MAX**, expressing interest in training.
   - Another member suggested that **Endia** had some relevant content.
- **Python's Role in MAX Questioned**: A member inquired about the decision to write **MAX** in **Python**, speculating on easier migration to **MAX** and **Mojo**.
   - The member wondered if this would lead to a split world issue, similar to **PyTorch**, and whether a pure **Mojo** framework for **MAX** would emerge.
- **Mojo API's Return to MAX Anticipated**: A member clarified that **MAX** previously had a **Mojo API**, but it was discontinued due to **Mojo**'s incomplete state.
   - They indicated that the **Mojo API** should return at some point when the language is more mature.
- **Python to Mojo Migration Hurdles Highlighted**: A member explained that while **Mojo** is not a strict **Python** superset, it resembles **C++** or **Rust** more closely.
   - They cautioned that migrating to **Mojo MAX** will require effort to leverage **Mojo**'s full potential, even though it looks like **Python**.
- **Performance Boost with Mojo MAX Questioned**: A member noted that **MAX** uses a **JIT compiler**, suggesting that performance gains from **Mojo** would mainly be in graph construction time.
   - They speculated that speed differences between **Mojo MAX** and **Python MAX** might not be significant, and the split-world issue would persist until **Mojo** gains more features.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1443178668007620752)** (19 messages🔥): 

> `TinyJit internals, Non tinygrad Python operations, Randomness functions in Tinygrad, Tinygrad JIT tutorial, PyTorch compiler history` 


- **TinyJit only replays kernels**: When using `@TinyJit`, the wrapped function only replays the captured **tinygrad kernels** and **ExecItems**, and the wrapped function won't run at all.
   - If you need Python code to run, split it into separate JIT functions, but this can be tricky, and any **non-tinygrad outputs** will not be updated.
- **Randomness functions in `Tensor` work as expected**: Randomness functions on `Tensor` should work since they increment counters via a kernel.
   - Example: `CPU=1 DEBUG=5 python3 -c "from tinygrad import Tensor; Tensor.rand().realize(); Tensor.rand().realize()"`.
- **Two JIT runs are required for tracing, but it might change to verifying match**: The JIT uses the second run to repeat the captured kernels and the first run may perform different setup tasks such as weight initialization.
   - A proposal suggests that the JIT might be updated to wait for two runs to match, indicating that the implementation is still pre-1.0 and subject to change, with efforts focused on removing footguns.
- **Good Tutorial on tinygrad JIT**: A member shared a [tutorial on tinygrad JIT](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html).
   - The tutorial is a bit outdated but still good.
- **Tinygrad fundamentals are solid**: Tinygrad's fundamentals are now solid, and the team is now shifting focus to frontend usability.
   - One person reminisces that *the very first pytorch compiler in a fast.ai lesson literally concatenated C code strings, using regex!*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1443177250269954071)** (14 messages🔥): 

> `Kimi's Limits, Chatbots vs Canvases, Conversational Fallacy` 


- **Kimi's Limits Explored**: A user inquired about the limits of **Kimi**, expressing uncertainty despite planning to upgrade from the web interface, and attached a [screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1443259988058574900/image.png?ex=69286c1b&is=69271a9b&hm=3d4d9ba62a03dc65a27edfb0fb93a8c0f8a0f6518ab2737c5a714d6032d2b5a6&).
   - Another user praised **Kimi K2** for its superior thinking and ability to push back, highlighting its understanding and interaction in the context of prompts.
- **Canvas Craze Coming?**: A user expressed disbelief that *canvases* haven't replaced chatbots yet, suggesting they make more sense for full-screen websites like **Kimi** and **Qwen**.
   - They argued that while chatbots are suitable for small side-panels, canvases could provide a better experience for comprehensive web interfaces.
- **Conversational Fallacy Considered**: A user shared a quote they are obsessed with: *we're stuck in the conversational fallacy: the idea that AI must be addressed to be used*.
   - The user seems to believe that **Kimi** does an amazing job of not falling into this fallacy.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1443010354866294867)** (4 messages): 

> `dspy-cli tool, DSPy projects, FastAPI endpoints, MCP tools, Docker hosting` 


- **dspy-cli Tool Goes Open Source**: Members announced that `dspy-cli` tool is now open source and available on [PyPi](https://pypi.org/project/dspy-cli/), to help create, develop, test, and deploy **DSPy programs** as HTTP APIs.
   - The [repo is available on GitHub](https://github.com/cmpnd-ai/dspy-cli) and the tool can be installed using `uv tool install dspy-cli`.
- **dspy-cli New Features Available**: The main features are to scaffold a new **DSPy project**, create new signatures from the command line, run modules as **FastAPI endpoints** or use them as **MCP tools**.
   - Programs can be easily deployed to a **docker hosting service** of choice.
- **dspy-cli Acclaimed for its Project Utility**: Members expressed eagerness to try `dspy-cli` on more projects and spread the word about its usefulness.
   - A user [tweeted](https://x.com/dbreunig/status/1993462894814703640) about the tool, praising the great work.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1442977805162909869)** (9 messages🔥): 

> `ReAct Module Trajectory Injection, Web Search API Implementation in DSPy, Anthropic Web Search API, Latency issues with web search API calls` 


- **Trajectory Injection in ReAct Modules**: A member inquired about injecting trajectories into a **ReAct module**, seeking to provide the agent with context from previous runs in addition to message history.
- **Web Search API choices for DSPy**: A member asked for advice on the best **APIs** to implement a web search tool in **DSPy**, specifically asking if the native web search **API** of a provider could be used.
- **Exa API includes summarization**: One member shared a positive experience using **Exa API** due to its summarization feature, which avoids the random ads and HTML tags found in other APIs like **Firecrawl** and **Parallel.ai**.
- **Using Anthropic's web search API with ReAct**: A member is trying to implement it using **Anthropic's web search API** with ReAct, and shared a code snippet using `dspy.ReAct`.
- **Latency caused by Web Search API Calls**: A member raised a question about the latency caused by web search **API** calls within **DSPy's ReAct** when using a search function like `search_web` before calling the LLM.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1442995213609074709)** (11 messages🔥): 

> `New Protocol Version, UI SEP Release, MCP Namespace Collision` 


- **New Protocol Version Drops!**: A new protocol version has been released, as announced in the [Discord channel](https://discord.com/channels/1358869848138059966/1421239779676127402/1442991223617880064).
   - Members expressed excitement and gratitude to the **MCP community** for their contributions over the past year.
- **UI SEP Ships Out-of-Band!**: The **UI SEP** can be shipped out-of-band from the main spec due to being an extension.
   - Check out the <#1376635661989449820> channel for more details.
- **MCP Considers Namespace Collisions!**: A member inquired about whether the **MCP** group considers the possibility of namespace collisions.
   - Specifically, the question was raised whether the group would take action if something claims to be something-mcp but diverges from the actual **MCP** standard.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1442980169353793619)** (8 messages🔥): 

> `AI Engineer introduction, API Issues, Telegram channel` 


- **AI Engineer showcases expertise**: An **AI engineer** with hands-on experience building advanced, end-to-end AI systems across multiple domains introduced themself.
   - Their expertise covers **AI agents, multi-agent systems, automating workflows, NLP-powered chatbots, integrating voice & speech systems, deploying custom LLMs, fine-tuned AI models, Web3, smart contracts, and AI-integrated blockchain games**.
- **User reports API issues and lack of support**: A user reported experiencing an *[unknown] error* in **webdev.v1.WebDevService/GetDatabaseSchema** due to usage quota exhaustion, despite topping up more than **$600**.
   - The problem has rendered their entire account unusable, affecting over **500 active users**, and they have not received any response or support from the team.
- **Members inquire about a Telegram channel**: A member inquired about the existence of a **Manus Telegram channel**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1443213701753606215)** (3 messages): 

> `Benchmark Updates, Opus 4.5 vs Sonnet 4.5` 


- **Community Suggests New Site Admin for Benchmarking**: A member suggested that someone else should run the site who can update the benchmark results with new models.
   - This implied dissatisfaction with the current state of benchmark result updates.
- **Opus 4.5 upgrade or minor upgrade over Sonnet 4.5?**: A member initiated a quick survey to gauge community sentiment on whether **Opus 4.5** is a big or minor upgrade over **Sonnet 4.5**.
   - Another member reported that they encountered a *'model not found'* error when trying what would typically be the correct Bedrock model identifier.


  

---


---


---


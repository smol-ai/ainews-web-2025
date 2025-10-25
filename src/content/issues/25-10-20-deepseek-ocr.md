---
id: MjAyNS0x
title: >-
  DeepSeek-OCR finds vision models can decode 10x more efficiently with ~97%
  accuracy of text-only, 33/200k pages/day/A100
date: '2025-10-20T05:44:39.731046Z'
description: >-
  As **ICCV 2025** begins, **DeepSeek** releases a novel **DeepSeek-OCR** 3B MoE
  vision-language model that compresses long text as visual context with high
  accuracy and efficiency, challenging traditional tokenization approaches. The
  model achieves ~97% decoding precision at <10× compression and processes up to
  ~33M pages/day on 20 A100-40G nodes, outperforming benchmarks like GOT-OCR2.0.
  Discussions highlight the potential for unlimited context windows and
  tokenization-free inputs, with contributions from **@karpathy**,
  **@teortaxesTex**, and others. In video generation, **google-deepmind**'s
  **Veo 3.1** leads community benchmarks with advanced precision editing and
  scene blending, while **Krea** open-sources a 14B autoregressive video model
  enabling realtime long-form generation at ~11 FPS on a single B200 GPU.
companies:
  - deepseek-ai
  - google-deepmind
  - krea
models:
  - deepseek-ocr
  - deepseek3b-moe-a570m
  - veo-3.1
topics:
  - ocr
  - vision
  - multimodality
  - model-compression
  - long-context
  - model-architecture
  - video-generation
  - autoregressive-models
  - model-efficiency
  - precision-editing
people:
  - karpathy
  - teortaxestex
  - reach_vb
  - _akhaliq
  - eliebakouch
  - vikhyatk
  - demishassabis
---


**Vision is all you need?**

> AI News for 10/17/2025-10/20/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (198 channels, and 14010 messages) for you. Estimated reading time saved (at 200wpm): 1097 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

As ICCV kicks off in Hawaii, DeepSeek continues to show signs of life. This one is a relatively [small paper with 3 authors](https://github.com/deepseek-ai/DeepSeek-OCR), and a small 3B model, but the contribution of a SAM+CLIP+compressor named **DeepEncoder**:

[](https://resend-attachments.s3.amazonaws.com/K2h3q6nqRq5KzAn)

and the headline findings are sound:

[](https://resend-attachments.s3.amazonaws.com/MrS11LlpmmdZten)

[](https://resend-attachments.s3.amazonaws.com/U1xDO4WwZEdXcXn)

The significance of a very good OCR model, beyond liberating a lot of data from books and PDFs, is the opportunity to always consume rich text and [get rid of the tokenizer](https://x.com/karpathy/status/1980397031542989305).

---

# AI Twitter Recap

**DeepSeek’s “Optical Context Compression” OCR and the end of text-only context?**

- **DeepSeek-OCR (3B MoE VLM) release**: DeepSeek unveiled a small, fast vision-language OCR that treats long text as “visual” context and compresses it 10–20× while preserving accuracy. Key numbers: ~97% decoding precision at <10× compression and ~60% at 20×; ~200K pages/day per A100-40G and ~33M pages/day on 20 nodes (8× A100-40G each). It beats GOT-OCR2.0 and MinerU2.0 on OmniDocBench using far fewer vision tokens and can re-render complex layouts (tables/charts) into HTML. Day-0 support in vLLM delivers ~2,500 tok/s on A100-40G, with official support landing next release. Code and model are on GitHub/Hugging Face. See overviews and demos from [@reach_vb](https://twitter.com/reach_vb/status/1980170192392270227), [@_akhaliq](https://twitter.com/_akhaliq/status/1980260630780162505), [@casper_hansen_](https://twitter.com/casper_hansen_/status/1980166248878203093), [@vllm_project](https://twitter.com/vllm_project/status/1980235518706401405), and the initial highlight by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1980160624140456370).
- **Architecture and implications for long context**: The released LLM decoder is a DeepSeek3B-MoE-A570M variant using MHA (no MLA/GQA), 12 layers, 2 shared experts, and a relatively high 12.5% activation ratio (vs 3.52% in V3 and 5% in V2), per [@eliebakouch](https://twitter.com/eliebakouch/status/1980193125202083951). The community debate centers on whether compressing “old” text into vision tokens enables “theoretically unlimited context” and better agent memory architectures, and whether pixels can be a superior input interface for LLMs than text tokens. See arguments for multimodal encoders and tokenization-free inputs by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1980165682516869575) and [@karpathy](https://twitter.com/karpathy/status/1980397031542989305), clarifications that storage remains tokens (not screenshots) by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1980453820632297900), and counterpoints on prefix-caching incompatibility and practical KV compression limits by [@vikhyatk](https://twitter.com/vikhyatk/status/1980437184839905725). Good concise summaries: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1980247935700066468), [@_akhaliq](https://twitter.com/_akhaliq/status/1980260630780162505).

**Video generation: Veo 3.1 leaps ahead; Krea Realtime goes OSS**

- **Veo 3.1 tops community evals and adds precision editing**: Google DeepMind’s Veo 3.1 jumped ~+30 on the Video Arena to become the first model over 1400 in both text-to-video and image-to-video, overtaking prior leaders on physics/realism per the community. DeepMind also shipped precision editing (add/remove elements with consistent lighting/scene interactions) and robust “Start Frame → End Frame” guidance that can blend real footage into stylized outputs. Try and compare in Flow/Gemini and LM Arena. Details from [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1980261047836508213), [@demishassabis](https://twitter.com/demishassabis/status/1980397419658645708), [@arena](https://twitter.com/arena/status/1980319296120320243), and examples via [@heyglif](https://twitter.com/heyglif/status/1980362634982748332).
- **Open-source realtime video generation**: Krea released “Realtime,” a 14B Apache-2.0 autoregressive video model capable of ~11 FPS long-form generation on a single B200. Weights and report are on Hugging Face; early benchmarks and notes from [@reach_vb](https://twitter.com/reach_vb/status/1980376352726610342) and the launch thread by [@krea_ai](https://twitter.com/krea_ai/status/1980358158376988747). Also noteworthy: Ditto’s instruction-based video editing dataset/paper ([@_akhaliq](https://twitter.com/_akhaliq/status/1980265202500116525)) and VISTA, a “test-time self-improving” video generation agent ([@_akhaliq](https://twitter.com/_akhaliq/status/1980398215707906391)).

**Agentic coding stacks, governance, and enterprise posture**

- **Claude Code goes web + iOS with safe-by-default execution**: Anthropic launched Claude Code in the browser and iOS, running tasks in cloud VMs with the chat loop during execution. A new sandbox mode in the CLI lets you scope filesystem and network access, reducing permission prompts by 84%; Anthropic open-sourced the sandbox for general agent builders. Early reviews praise the direction but note rough edges in cloud handoff. See launch and deep dives by [@_catwu](https://twitter.com/_catwu/status/1980338889958257106), sandbox details from [@trq212](https://twitter.com/trq212/status/1980380866657526047) and [@_catwu](https://twitter.com/_catwu/status/1980383210560450961), the open-source repo note by [@omarsar0](https://twitter.com/omarsar0/status/1980408741007876183), and a product vibe check by [@danshipper](https://twitter.com/danshipper/status/1980334576225472793).
- **Enterprise-grade agent ops (BYOI, multi-cloud, and speed)**: Cline announced an enterprise version that runs where developers work (VS Code/JetBrains/CLI) and with whichever model/provider is available (Claude/GPT/Gemini/DeepSeek across Bedrock, Vertex, Azure, OpenAI). This “bring your own inference” posture materially helps during cloud outages. IBM and Groq are pairing watsonx agents with Groq LPU inference (claimed 5× faster at 20% of cost) and enabling vLLM-on-Groq, indicating the agent stack is rapidly diversifying beyond a single cloud. See [@cline](https://twitter.com/cline/status/1980369441079849229), [@robdthomas](https://twitter.com/robdthomas/status/1980239227955683598), and [@sundeep](https://twitter.com/sundeep/status/1980288298477125841). Also in this vein: MCP-backed doc servers injected into coding agents ([@dbreunig](https://twitter.com/dbreunig/status/1980328051134329110)), easy multi-cloud GPU dev envs ([@dstackai](https://twitter.com/dstackai/status/1980369241963741236)), and global batch inference playbooks ([@skypilot_org](https://twitter.com/skypilot_org/status/1980307993842622471)).

**Infra resilience and performance tooling**

- **AWS us-east-1 outage (blast radius and lessons)**: A major outage took down multiple AI apps (e.g., Perplexity and Moondream’s website; Baseten’s web UI), with services gradually recovering. PlanetScale reported 99.97% of DB ops completed in us-east-1 by minimizing external dependencies. The episode re-emphasized multi-region/multi-cloud strategies, minimizing vendor lock-in, and BYOI: see outage status and recovery from [@AravSrinivas](https://twitter.com/AravSrinivas/status/1980172632600506579) and ([recovery](https://twitter.com/AravSrinivas/status/1980239929189036222)), impacts from [@vikhyatk](https://twitter.com/vikhyatk/status/1980171953614012448), [@basetenco](https://twitter.com/basetenco/status/1980191414031138868) ([recovery](https://twitter.com/basetenco/status/1980211561013850376)), [@midudev](https://twitter.com/midudev/status/1980190169513828437), [@reach_vb](https://twitter.com/reach_vb/status/1980211455564861923), [@nikitabase](https://twitter.com/nikitabase/status/1980399551883407787), and a PSA on causality from [@GergelyOrosz](https://twitter.com/GergelyOrosz/status/1980381693136847258). Related: “BYOI strikes again” from [@cline](https://twitter.com/cline/status/1980311303001633125).
- **Kernels, DSLs, and quantization**: Modular brought industry-leading perf to AMD MI355 in two weeks and now supports 7 GPU architectures across 3 vendors, demonstrating the benefits of deep compiler investment ([launch](https://twitter.com/clattner_llvm/status/1980320847475913112), [coverage](https://twitter.com/clattner_llvm/status/1980321245314064467)). TileLang, a new AI DSL, hits ~95% of FlashMLA on H100 with ~80 lines of Python via layout inference, swizzling, warp specialization, and pipelining ([@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1980170674112188440)). Also, GPTQ int4 post-training quantization is now built into Keras 3 with a vendor-agnostic guide ([@fchollet](https://twitter.com/fchollet/status/1980343806265552918)).

**Evals and benchmarks: real money, real leaderboards, and structured reasoning**

- **Real-money trading eval (interpret with caution)**: A community benchmark ([nof1.ai](http://nof1.ai/)) allocated $10k per model over a few days; reports show DeepSeek V3.1 and Grok 4 leading while GPT-5/Gemini 2.5 lost money ([@mervenoyann](https://twitter.com/mervenoyann/status/1980178771706835425), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980318499185823760)). Caveats: small-N, high variance, prompt dependence, and path dependence; “noise dominates” unless you shard capital across many runs ([@abeirami](https://twitter.com/abeirami/status/1980434468398883076)). Context: DeepSeek’s quant pedigree is a recurrent theme ([@hamptonism](https://twitter.com/hamptonism/status/1980182896049811780)).
- **Leaderboards and structured reasoning**: WebDev Arena added four models: Claude 4.5 Sonnet Thinking 32k; GLM 4.6 (new #1 open); Qwen3 235B A22B; and Claude Haiku 4.5 ([@arena](https://twitter.com/arena/status/1980367208300835328)). Elsewhere, Parlant’s Attentive Reasoning Queries (ARQ) use schema-constrained, domain-specific “queries” instead of free-form CoT and reported 90.2% across 87 scenarios vs 86.1% for CoT (repo in thread) ([@_avichawla](https://twitter.com/_avichawla/status/1980159925109309799)). Also see “when to stop seeking vs act” termination training (CaRT) ([@QuYuxiao](https://twitter.com/QuYuxiao/status/1980303030722703747)) and the observation that DeepSeek perf tracks PrediBench results ([@AymericRoucher](https://twitter.com/AymericRoucher/status/1980196484617523445)).
- **China model notes**: Kimi K2 claims up to 5× faster and 50% more accurate on internal workloads ([@crystalsssup](https://twitter.com/crystalsssup/status/1980147163629047854)); team shared internal benchmarks ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1980219115840385349)).

**Domain tools: Life sciences, data pipelines, and structured extraction**

- **Claude for Life Sciences**: Anthropic launched connectors (Benchling, PubMed, [Synapse.org](http://synapse.org/), etc.) plus Agent Skills to follow scientific protocols, with early users including Sanofi, AbbVie, and Novo Nordisk. Anthropic also published a Life Sciences GitHub repo with examples ([launch](https://twitter.com/AnthropicAI/status/1980308459368436093), [details](https://twitter.com/mikeyk/status/1980311408576458764), [repo](https://twitter.com/scaling01/status/1980297805911712107)).
- **Data workflows**: LlamaIndex demonstrated a robust text-to-SQL workflow with semantic table retrieval (Arctic-embed), OSS text2SQL (Arctic via Ollama), multi-step orchestration, and error handling ([@llama_index](https://twitter.com/llama_index/status/1980309057287446532)). FinePDFs released new PDF OCR/Language-ID datasets and models (XGB-OCR) to power document pipelines ([@HKydlicek](https://twitter.com/HKydlicek/status/1980319822585143498), [@OfirPress](https://twitter.com/OfirPress/status/1980319814481817901)). For structured VLM extraction, Moondream 3 shows single-shot JSON parsing of complex parking signs—no OCR stack required ([@moondreamai](https://twitter.com/moondreamai/status/1980405287531254089)).

**Top tweets (by engagement)**

- DeepSeek’s “visual compression OCR” and long-context implications caught fire across the community: succinct technical summary by [@godofprompt](https://twitter.com/godofprompt/status/1980233080213590326) and the broader “pixels over tokens” thread by [@karpathy](https://twitter.com/karpathy/status/1980397031542989305).
- Massive AWS outage updates (Spanish): impact and commentary roundup by [@midudev](https://twitter.com/midudev/status/1980190169513828437); Perplexity outage and recovery from [@AravSrinivas](https://twitter.com/AravSrinivas/status/1980172632600506579) and ([recovery](https://twitter.com/AravSrinivas/status/1980239929189036222)).
- Veo 3.1’s leap to #1 in Video Arena, with official acknowledgments from [@arena](https://twitter.com/arena/status/1980319296120320243) and [@demishassabis](https://twitter.com/demishassabis/status/1980397419658645708).
- Kimi K2 performance claim: up to 5× faster and 50% more accurate ([@crystalsssup](https://twitter.com/crystalsssup/status/1980147163629047854)).
- Classic read: Richard Sutton resurfaces original Temporal-Difference learning resources ([@RichardSSutton](https://twitter.com/RichardSSutton/status/1980150877177688544)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DeepSeek OCR Release

- [**DeepSeek releases DeepSeek OCR**](https://www.reddit.com/r/LocalLLaMA/comments/1obcm9r/deepseek_releases_deepseek_ocr/) (Activity: 565): **DeepSeek has released a new OCR model, [DeepSeek OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR), which introduces a novel approach called *Optical Compression*. This technique leverages increasing image compression over time to facilitate a form of visual/textual forgetting, potentially enabling longer or even infinite context handling. This approach is detailed in their paper, which highlights its potential to extend context length significantly beyond current capabilities.** A notable discussion point is the comparison to **Qwen3 VL**, with some users intrigued by the naming of a mode as 'gundam'. The community is also discussing the implications of the optical compression technique for context management in OCR applications.
    - DeepSeek OCR introduces a novel approach called 'Contexts Optical Compression', which leverages increasing image compression over time as a method for visual/textual forgetting. This technique potentially allows for much longer context windows, possibly even infinite, by efficiently managing memory and processing resources. This could be a significant advancement in handling large-scale data inputs in OCR systems.
    - The model has been trained on a substantial dataset, including 1.4 million arXiv papers and hundreds of thousands of e-books. This extensive training dataset suggests that DeepSeek OCR might excel in specific domains, particularly in recognizing complex text structures like math and chemistry formulae. While it may not surpass PaddleOCR-VL in overall state-of-the-art performance, it could outperform in specialized text recognition tasks.
    - There is anticipation for the Omnidocbench 1.5 benchmarks to provide more detailed performance metrics. Current evaluations like edit distance are insufficient without complementary metrics such as table TEDS and formula CDM scores. These benchmarks will be crucial in assessing DeepSeek OCR's capabilities in comparison to existing models, especially in specialized areas like mathematical and chemical text recognition.
- [**What happens when Chinese companies stop providing open source models?**](https://www.reddit.com/r/LocalLLaMA/comments/1ob9vvk/what_happens_when_chinese_companies_stop/) (Activity: 809): **Chinese companies like Alibaba have shifted from open-source to closed-source models, exemplified by the transition from WAN to WAN2.5, which now requires payment. This move raises concerns about the future availability of open-source models from China, which have been crucial for global access and competition against US models. The change could impact the global AI landscape, as open-source models have been a key differentiator for Chinese companies in the international market.** Commenters suggest that China's open-source strategy has been a counter to the US's proprietary models, providing affordable alternatives. If Chinese models become closed-source, they may lose international appeal, as their open-source nature was a primary advantage over US models.
    - TopTippityTop discusses the strategic advantage China gains from open source models, highlighting that China's economy is more focused on physical goods production, whereas the US economy is more dependent on software and services. This reliance makes the US economy more fragile, suggesting that China's open source strategy is a calculated move to leverage this economic dynamic.
    - RealSataan argues that the primary appeal of Chinese models to international users is their open source nature. If Chinese companies were to stop providing open source models, these models would lose their competitive edge against American alternatives, which are more widely accessible globally.
    - Terminator857 suggests that if Chinese companies transition to closed source models, they could potentially increase their revenue significantly. This implies a trade-off between maintaining open source accessibility and capitalizing on proprietary models for financial gain.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Robotics Innovations

- [**Introducing Unitree H2 - china is too good at robotics 😭**](https://www.reddit.com/r/singularity/comments/1obbuf9/introducing_unitree_h2_china_is_too_good_at/) (Activity: 1324): **Unitree Robotics has introduced the Unitree H2, a new robotic model that showcases advanced movement capabilities, making strides towards more natural and fluid motions. This development highlights China's growing expertise in robotics, with the H2 model demonstrating significant improvements in agility and functionality. The robot's design and engineering reflect a focus on enhancing practical applications, although some users express a desire for more utility-focused features.** Commenters note the impressive naturalness of the robot's movements, suggesting that while the technology is advancing, there is still a demand for robots to perform more practical tasks.
    - midgaze highlights China's rapid advancements in robotics and automation, suggesting they are approaching an 'automation singularity.' This implies a self-reinforcing cycle where improved manufacturing leads to better robots, which in turn enhance manufacturing capabilities. The comment underscores the strategic advantage China is gaining in this sector, potentially outpacing global competitors.
    - crusoe points out a critical observation regarding the Unitree H2, noting that while the robot is shown performing tasks like dancing, it lacks demonstrations of practical applications. This contrasts with companies like Boston Dynamics, which often showcase their robots in real-world scenarios, emphasizing functionality over entertainment.
    - RDSF-SD comments on the naturalness of the Unitree H2's movements, indicating significant progress in robotic kinematics and control systems. This improvement in movement fluidity is crucial for applications requiring human-like interaction and precision, suggesting that the technology is advancing towards more sophisticated and practical uses.
- [**Movies are staring to include "No AI was used in the making of..." ect in the end credits**](https://www.reddit.com/r/singularity/comments/1obfu6c/movies_are_staring_to_include_no_ai_was_used_in/) (Activity: 764): **Recent films are beginning to include disclaimers in their end credits stating that "No AI was used in the making of this movie." This trend reflects a growing concern over the use of AI in creative processes, reminiscent of past debates over digital versus film photography. The claim is seen by some as performative, given the pervasive integration of AI tools in production workflows, and the expectation that AI will become even more embedded in future creative tools.** Commenters express skepticism about the feasibility of completely avoiding AI in film production, noting parallels to past technological shifts like the transition from practical effects to CGI. There is a belief that AI will become so integral that such disclaimers will be impossible to substantiate.
    - NoCard1571 argues that claims of not using AI in film production are largely performative, suggesting that it's unlikely no one used a language model during production. They predict that generative AI will become so integrated into tools that such claims will be impossible to verify in the future.
    - zappads highlights a historical parallel with the replacement of pyrotechnic artists by CGI in the 2000s, noting that cost and ease often drive technological adoption in film production. They suggest that economic pressures will lead to increased AI use, regardless of current claims about its absence.
    - letmebackagain emphasizes the importance of focusing on the quality of the final product rather than the tools used in its creation. This perspective suggests that the debate over AI usage should center on the artistic and technical merits of the work rather than the production methods.

### 2. AGI Predictions and History

- [**In 1999, most thought Ray Kurzweil was insane for predicting AGI in 2029. 26 years later, he still predicts 2029**](https://www.reddit.com/r/OpenAI/comments/1obh30l/in_1999_most_thought_ray_kurzweil_was_insane_for/) (Activity: 626): **Ray Kurzweil has consistently predicted the arrival of Artificial General Intelligence (AGI) by** `2029`**, a claim he first made in** `1999`**. Despite skepticism, Kurzweil maintains this timeline, suggesting that AI will achieve human-level intelligence across a wide range of tasks by then. However, the lack of a universally accepted definition of AGI complicates these predictions, as noted by experts who argue that while AI may reach human parity in specific tasks, the essence of AGI remains elusive.** A notable opinion from the comments highlights skepticism about AGI predictions due to the absence of a clear definition. Another perspective suggests that while AI might achieve human-level performance in many tasks by `2029`, the concept of true AGI is still undefined.
    - jbcraigs highlights the challenge in predicting AGI due to the lack of a universally accepted definition. They argue that while AI may achieve human-level performance in specific tasks by 2029, the concept of 'true AGI' remains elusive because we don't fully understand what it entails.
    - KairraAlpha discusses the misconception that AGI is merely about surpassing human capabilities in math and logic. They suggest that intelligence is multifaceted and that AGI might not conform to traditional expectations. They imply that current models like GPT-5 and Claude could reveal unexpected capabilities if unrestricted, hinting at the complexity and unpredictability of AGI development.
- [**Today is that day 😭**](https://www.reddit.com/r/ChatGPT/comments/1obbrf2/today_is_that_day/) (Activity: 2058): **The post discusses a controversial use of MLK Jr's likeness in AI-generated content, leading to OpenAI's decision to prohibit such uses. This reflects ongoing ethical concerns in AI regarding the representation and use of historical figures' images and voices without consent. The issue highlights the need for stricter guidelines and oversight in AI content generation to prevent misuse and respect intellectual property rights.** Commenters express frustration and disappointment, questioning the ethical oversight in AI development and the responsibility of companies like OpenAI to prevent such occurrences.
- [**The meme continues**](https://www.reddit.com/r/aivideo/comments/1obz0di/the_meme_continues/) (Activity: 418): **The Reddit post humorously references a situation where a streamer, possibly Hassan, is jokingly accused of forcing someone to watch their stream to increase watch time. This is likened to a potential South Park joke, highlighting the absurdity and humor in the situation. The external link indicates restricted access due to network security, requiring login or a developer token for further access.** Commenters find the situation amusing and suggest it would fit well as a **South Park** joke, indicating the humor resonates with the show's style.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. AI Video Generation Showdown**

- **Veo Victorious on Video Leaderboards**: **Veo-3.1** now ranks #1 on both the [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) and [Image-to-Video Leaderboard](https://lmarena.ai/leaderboard/image-to-video), with the organizers inviting submissions and feedback in their [Arena announcement on X](https://x.com/arena/status/1980319296120320243). Community testing ramped up around these boards, centering on prompt coverage, temporal coherence, and motion fidelity under leaderboard constraints.
    - Participants shared generations and edge-case prompts while discussing leaderboard methodology and evaluation bias toward short clips and specific prompt classes, highlighting the importance of **consistent motion** and **maintained identity**. Several engineers noted that leaderboards spur fast iteration cycles and reproducibility for **text-to-video** and **image-to-video** baselines.
- **Sora Slides While Veo Surges**: Engineers compared **Sora 2** outputs (see example on [Sora examples](https://sora.chatgpt.com/p/s_68f6acd380788191b301485853f831fc)) against **Veo-3.1**, reporting perceived quality degradation in Sora since initial release despite Veo’s leaderboard dominance. Debates focused on subjective quality vs. leaderboard scores, and on prompt reproducibility across model updates.
    - Discussion stressed that evaluation should normalize for seed, clip length, and postprocessing to fairly judge **temporal consistency**, **physics plausibility**, and **character persistence**. Some users concluded that even if Veo currently tops leaderboards, Sora’s strengths still appear in specific cinematic scenes and stylized VFX.
- **Krea Realtime Cranks Out Open-Source Video**: **Krea AI** open-sourced a 14B autoregressive text-to-video model, **Krea Realtime**, distilled from Wan 2.1 and capable of ~**11 fps** on a single NVIDIA **B200**, as announced in the [Krea Realtime announcement](https://xcancel.com/krea_ai/status/1980358158376988747). Engineers immediately explored **ComfyUI** graphs, expected throughput on **RTX 5090**, and fine-tuning hooks for domain-specific motion.
    - Builders highlighted that an OSS baseline with real-time generation unlocks rapid **workflow prototyping** and benchmarking against closed models. Early adopters traded notes on context windows, frame conditioning, and optimizing decode pipelines for **low-latency streaming**.

**2. Kernel DSLs and Quantization Updates**

- **Helion Hits Public Beta**: **Helion 0.2** shipped as a public beta on PyPI ([Helion 0.2 on PyPI](https://pypi.org/project/helion/0.2.0/)) alongside developer outreach at the Triton Developer Conference and **PyTorch Conference 2025**. The tool positions itself as a high-level DSL for kernel authoring layered over compiler stacks, with multiple talks and live Q&A for hands-on users.
    - Engineers welcomed a higher-level path to author performant kernels while keeping **MLIR** and compiler ergonomics in play. Conference chatter emphasized tight integration with **PyTorch Compiler** stacks and future-proofing for evolving GPU backends.
- **Triton TMA Truths on SM120 and Hopper**: Practitioners testing **TMA** in **Triton** on NVIDIA **SM120** reported no wins vs `cp.async`, aligning with notes that on **Hopper** TMA underperforms for loads under ~**4 KiB** and that **Ampere** lacks TMA (pointer math may still be faster); background context referenced the matmul deep-dive [Matmul post (Aleksa Gordić)](https://www.aleksagordic.com/blog/matmul). Benchmarks suggest TMA shines for larger tiles and multicast patterns but demands careful tiling to beat `cp.async` on small transfers.
    - CUDA discussions contrasted **latency/bandwidth** behavior across DSMEM, L2, and device memory while tuning descriptor-driven layouts. Takeaway: profile both **tile size** and **swizzle**; keep a `cp.async` fallback path for sub-4 KiB tiles on Hopper-class parts.
- **TorchAO Tweaks Quant Configs**: **TorchAO** will deprecate `filter_fn` for `quantize_op` in favor of regex-capable **ModuleFqnToConfig** ([TorchAO PR #3083](https://github.com/pytorch/ao/pull/3083)), simplifying selective quant policies. In parallel, users noted **SGLang** online quantization’s current inability to skip vision stacks as documented in the [SGLang quantization docs](https://docs.sglang.ai/advanced_features/quantization.html#online-quantization).
    - Teams welcomed regex-based scoping for large codebases mixing text and vision channels, flagging migration work in existing helpers. The broader thread tied into **provider-agnostic** deployment hygiene and upcoming **PyTorch 2.9** features for symmetric memory backends in multi-GPU settings.

**3. New Models, Datasets, and Agent Tooling**

- **Qwen3 Vision Lands with VL-8B**: **Qwen** released the multimodal **Qwen3-VL-8B-Instruct** on Hugging Face ([Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)), with GGUF variants appearing for local runners. Engineers compared it against other VLMs in real workflows (e.g., **ComfyUI**) and discussed prompt templates and chat formatting.
    - Early adopters evaluated OCR, chart/table parsing, and code-diagram grounding, noting tokenizer/chat-template sensitivities. The thread emphasized consistent **ChatML** formatting and careful system prompt handling to stabilize **vision-language** performance.
- **xLLMs Drops Multilingual Dialogue Troves**: The **xLLMs** collection published multilingual/multimodal dialogue datasets for long-context reasoning and tool-augmented chats ([xLLMs dataset collection](https://huggingface.co/collections/lamhieu/xllms-66cdfe34307bb2edc8c6df7d); spotlight: [xllms_dialogue_pubs](https://huggingface.co/datasets/lamhieu/xllms_dialogue_pubs)). The sets target **long-context**, **multi-turn coherence**, and **tool-use** evaluation across up to nine languages.
    - Builders highlighted that standardized multi-turn traces accelerate **SFT** and **eval** pipelines. Discussions focused on splitting by language/task, curating tool traces, and mapping to **instruct** vs **cloze** templates for harnesses.
- **Agents Get Self-Hosted Tracing**: A community project released a **self-hosted tracing and analytics** stack for the **OpenAI Agents** framework to address GDPR and export limitations ([openai-agents-tracing](https://github.com/yusuf-eren/openai-agents-tracing)). The repo ships dashboards and storage to keep agent traces private and portable.
    - Teams praised the ability to inspect latencies, tool-call fanout, and failure modes without sending telemetry to third-party dashboards. Privacy-conscious orgs flagged this as critical for regulated workloads requiring **on-prem** observability.

**4. Portable GPU Compute on Macs**

- **tinygrad Turns USB4 Docks into eGPU Lifelines**: **tiny corp** announced public testing of a pure-Python driver enabling **NVIDIA 30/40/50-series** and **AMD RDNA2–4** GPUs via any **USB4 eGPU** dock on Apple‑Silicon MacBooks ([tinygrad eGPU driver announcement](https://xcancel.com/__tinygrad__/status/1980082660920918045)). Engineers immediately probed perf ceilings, NPU interplay, and dev ergonomics for mobile rigs.
    - Threads debated whether streamlined **NPU** programming could complement eGPU offload for hybrid pipelines. Mac users compared dock/firmware quirks and discussed driver maturity for **compute-intensive** workflows (LLM infer, diffusion, video).
- **Secondhand 3090 Survival Guide**: A practitioner shared a field-tested checklist for buying used **RTX 3090** cards—bring a portable eGPU setup, confirm `nvidia-smi`, run `memtest_vulkan`, optionally run `gpu-burn`, and watch thermals ([RTX 3090 used-buying tips](https://xcancel.com/taha_yssne/status/1960418430655586677)). The guidance aims to reduce **VRAM**/thermal surprises and weed out flaky boards.
    - Mac eGPU experimenters echoed the importance of live validation under load rather than idle checks. The community also noted nascent **macOS NVIDIA driver** efforts circulating in tinygrad-related threads for broader compatibility testing.

**5. Research & Evaluation Highlights**

- **Anthropic Maps Attention Mechanics**: Anthropic extended attribution graphs from MLPs to attention with the paper **Tracing Attention Computation Through Feature Interactions** ([paper](https://transformer-circuits.pub/2025/attention-qk/index.html)). Discussions connected this to prior interpretability work and to techniques for mitigating **logit blowups** via QK normalization.
    - Researchers debated how feature hierarchies emerge across attention layers and how to visualize **QK interactions** in practice. Engineers noted the potential for targeted ablations and better **mechanistic** probes in small-to-mid models.
- **Eval Harness Gets a UX Glow-Up**: Eleuther outlined an **lm-evaluation-harness** refactor adding new templates, standardized formats, clearer instruct tasks, and friendlier UX (branch: [smolrefact](https://github.com/EleutherAI/lm-evaluation-harness/tree/smolrefact); planning: [Eval harness planning When2Meet](https://www.when2meet.com/?33070160-Bw5xm)). Goals include easier conversion between task variants (e.g., **MMLU → cloze → generation**) and saner `repeats` behavior.
    - Library users welcomed fewer format footguns and better reproducibility for **long-context** and **tool** tasks. The team solicited feedback from heavy users to finalize templates before broader roll-out.
- **NormUon Nudges Optimizer State of the Art**: A new optimizer dubbed **NormUon** entered discussion circles with claims of **SOTA-level** results if benchmarks hold ([NormUon optimizer (arXiv:2510.05491)](https://arxiv.org/abs/2510.05491)). Community cross-checks compared it against **Muon** with non-speedrun baselines and QK-norm mitigations.
    - Practitioners reported it’s *"the same perf as muon on their non-speedrun setups but with good muon baselines"* while flagging stability from smoother weight spectra. Others cautioned for head-to-head ablations before declaring wins on **reasoning-heavy** workloads.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet's Cash: Referral Program Payouts Probed**: Users discussed the **Comet referral program**, troubleshooting [referral links](https://www.perplexity.ai/browser/invite-ga) and reporting **missing payouts**.
   - Tips for successful referrals included ensuring referred users install Comet and perform a search.
- **AWS Accident: Outage Outlines Over-Reliance**: A recent **AWS outage** caused widespread issues for Perplexity AI and other services, leading to discussions about the [risks of over-reliance](https://health.aws.amazon.com/health/status) on single cloud providers.
   - Users shared status links and personal experiences, noting that **certain features were still affected** even after the initial outage was resolved.
- **Claude Conquers: GPT-5 Grounded in Competition**: Members debated the relative merits of **GPT-5 versus Claude**, with many finding **Claude** to be superior for complex projects and reasoning tasks.
   - Some users noted that **GPT-5's free tier** felt noticeably weaker, while others suggested using **Claude 4.5** for optimal results.
- **Philosophical Forte found for Claude Sonnet 4.5**: **Claude Sonnet 4.5** is highlighted as effective for **philosophical problem solving**, accompanied by a [shared Claude conversation link](https://claude.ai/share/886d8469-3dd2-4e46-b491-c28e5131985d).
   - Users have also begun claiming **Perplexity AI claim invite links** and recommending it for **step-by-step guides**.
- **Pricing Particulars: API Access Assessed**: A user inquired about who to contact for custom **API pricing** for large API users and how to get access.
   - Another user suggested emailing [api@perplexity.ai](mailto:api@perplexity.ai) to get in touch with the appropriate team.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Lithiumflow Impresses with Coding, Sparks Gemini Speculation**: Members tested **Lithiumflow's** coding capabilities, finding its ability to write a full macOS system in a single HTML file impressive, with a demo available on [Codepen](https://codepen.io/Kross-the-scripter/pen/emJVRyx).
   - Some users found it inferior to **Claude 4.5** for coding, leading to theories that **Lithiumflow** might be a nerfed **Gemini 3** variant or a specialized coding model that hints at **Google's** prioritization of coding in their models.
- **Gemini 3: Release Date Remains a Mystery**: Speculation continues around the release and specifications of **Gemini 3**, with theories suggesting that **Lithiumflow** and **Orionmist** are potential **Gemini 3** iterations or specialized models.
   - A tweet suggests the release is still two months away, accessible on [Twitter](https://x.com/OfficialLoganK/status/1980435968323907884).
- **Video Arena: Veo-3.1 Dominates Video Leaderboards!**: The [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) & [Image-to-Video Leaderboard](https://lmarena.ai/leaderboard/image-to-video) now show **Veo-3.1 ranking #1 in both** categories.
   - Users are encouraged to share their **Veo-3.1** generations and provide feedback, as [announced on X](https://x.com/arena/status/1980319296120320243).
- **AI Video Quality: Sora 2 vs Veo 3.1**: Members are debating the video quality of different AI models, including **Sora 2** and **Veo 3.1**, with examples available on [ChatGPT's website](https://sora.chatgpt.com/p/s_68f6acd380788191b301485853f831fc).
   - Despite **Veo 3.1** being ranked higher on leaderboards, the consensus is that **Sora 2's** quality has degraded since its initial release.
- **Claude Sonnet 4.5: A League Above?**: Members discuss **Claude Sonnet 4.5's** creative writing capabilities, with one stating that **Claude Sonnet 4.5 Thinking** is in *"a different league"* compared to **Gemini 2.5 Pro**.
   - However, ongoing bugs and issues with models getting stuck or generating generic error messages on LMArena are hindering effective testing and comparison.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Eyes Apple Hardware**: Support for Apple hardware is planned for **Unsloth** sometime next year, but members expressed some hesitation.
   - No further information or timeline was disclosed at this time.
- **Hackathon Halted; MI300X for All!**: The AMD x PyTorch Hackathon is extended, with each participant receiving **100 hours of 192GB MI300X GPUs**, to compensate for disruptions.
   - Community members described the event as having *so many issues* but the team is *trying*.
- **Modelscope Saves the Day During HF Outage**: Due to [AWS outages](https://health.aws.amazon.com/health/status) affecting Hugging Face, members recommended using [Modelscope](https://modelscope.cn/models/unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit) as a temporary mirror.
   - A code snippet was shared describing how to load models from Modelscope in a Colab environment, requiring the `load_in_4bit` flag.
- **Thinking Mode Tagging in SFT**: A member proposed using XML tags such as `<thinking_mode=low>` to teach an AI model different thinking modes during SFT, categorizing examples based on CoT token count, with an **auto** mode for the model to decide.
   - Another member suggested that controllable thinking usually benefits from some **reinforcement learning after SFT**.
- **Luau-Devstral-24B-Instruct-v0.2 > GPT-5 Nano**: A member pointed out that [GPT-5 Nano in Luau-Devstral-24B-Instruct-v0.2](https://huggingface.co/TorpedoSoftware/Luau-Devstral-24B-Instruct-v0.2) outperforms GPT-5, noting its surprising performance.
   - The member remarked that *GPT-5 is weird*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Comcast Representatives peddling user data?**: A user alleged that **Comcast** representatives are stealing data and selling it through phishing attempts targeting authentic **Kenyan organizations** and **Philippines-based call centers**.
   - The user claims they are documenting cases and suggested contacting **CEO Brian Roberts** for accountability.
- **OpenCode Discovers Super Speedy Stealth Model**: The **OpenCode** team has a mysterious **stealth model** that is incredibly fast at spitting out tokens, but they are unsure if it's intelligent.
   - Members stated it's the fastest model they've ever seen, generating hype for what's next in **OpenCode**.
- **Grok cracks Video Generation**: **Grok** has now launched video generation, accessible via **SuperGrok** subscription ($30/month) without requiring a **Twitter/X** account.
   - Users are finding **Grok** video generation copes fine with franchises and 3rd party content, producing videos without watermarks.
- **Sora Reality Distortion Field**: A user inquired if **Sora** can distinguish between images of real people and fictional characters, with a simple response in the affirmative.
   - The conversation starter also shared a link to *pseudoCode* thought experiment for AI models.
- **Controlling Conversation flow on ChatGPT**: A user expressed frustration with **ChatGPT's** tendency to end responses with unsolicited follow-up questions and sought advice on disabling this *'feature'*, and other users had ideas.
   - A user suggested replacing the follow-up questions with something else, such as a joke or a detail about a favorite subject, and shared example chats, like [this one with Dad jokes](https://chatgpt.com/share/68f6b5ec-04e0-8011-b4ad-8342ee1a0405).



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **TLRAG Framework Claims Debated**: A developer introduced **TLRAG**, a *deterministic and model-agnostic framework* claiming superiority over **Langchain** without function calling, and boasting token savings and dynamic AI persona evolution, as documented on its [dev.to page](https://dev.to/tlrag).
   - Community members expressed skepticism regarding **TLRAG's** claims, UI/UX glitches, security and whitepaper structure, with one calling the whitepapers *AI-generated shit wich isnt true at all*.
- **SambaNova Latency Woes in DeepSeek v3.1**: Users reported latency issues with **SambaNova** in **DeepSeek v3.1 Terminus**, noting that despite theoretically providing higher throughput, **DeepInfra** appears faster in both throughput and latency as shown on [OpenRouter's comparison page](https://openrouter.ai/deepseek/deepseek-v3.1-terminus/providers?sort=throughput).
   - The discussion underscored the importance of practical performance metrics over theoretical throughput, especially in real-world applications.
- **GPT-5 Image: Is It Just GPT-image-1?**: Members debated the authenticity of **GPT-5 Image**, with some speculating that it is merely **GPT-image-1** with a tweaked API, rather than a new or improved model.
   - One member succinctly stated *I think so I've tested both and I like nano banana a lot more*, indicating a preference for alternative image generation models.
- **Qwen3 Pricing Stuns Users**: The community is stoked over the **Qwen3 235A22B API pricing**, finding it excellent, especially when integrated with **W&B** for extensive data processing.
   - Users are urging **OpenRouter** to highlight routine price drops, emphasizing that the intelligence-per-dollar ratio is unmatched, setting a new standard in cost-effectiveness.
- **LFM 7B Sees Its End**: The community bid farewell to **Liquid's** hosting of **LFM 7B**, the original $0.01/Mtok LLM, which was deleted at **7:54 AM (EST)**.
   - Users noted the lack of a direct replacement in terms of pricing on **OpenRouter**, with **Gemma 3 9B** being the closest but triple the output cost.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API on iPad with MCP Servers**: A user wants to access **LM Studio** API from an iPad using [3Sparks Chat](https://example.com) and use **MCP servers**, which is currently not supported via the API.
   - Users suggested using remote desktop apps like **Tailscale** or **RustDesk** as workarounds.
- **System Prompt Parsing Causes Bracket Chaos**: **System prompts** in **LM Studio** are parsed, causing issues with special characters like **brackets** depending on the model and chat template, with a fix incoming via [Jinja template fix](https://github.com/ggml-org/llama.cpp/pull/15019/commits/e52c95c740831a7b84820721730980cb59999f23).
   - The **ChatML template** is suggested for **Qwen models** to mitigate some of these issues, along with a [custom prompt template](https://cdn.discordapp.com/attachments/1110598183144399058/1428919144962724020/message.txt?ex=68f834a8&is=68f6e328&hm=58ee3a7881ea7f41910df34a1d8a84e0f644cea6b0dd9ac5b386d473b8aa58ff&).
- **Maximizing Auxiliary GPU Performance by Exhausting It**: A user is trying to make their **3050** as an auxiliary GPU alongside a **3090** for extra VRAM, but some users noted that the two cards were *"suffocating the GPU"*.
   - Discussion covered setting the **3090 as the primary compute device** in hardware settings.
- **Desk Fan Cooling Mod as a Joke**: A user jokingly installed [dual 12-inch turbine fans](https://cdn.discordapp.com/attachments/1153759714082033735/1428989742602522624/IMG20251018171356.jpg?ex=68f7cda7&is=68f67c27&hm=8dcd0000dc20ee37ad9d4108ee238e75c8548dc0778594ef230cc451d50ebcdb&) to cool their PC, after returning from a psych ward.
   - The user clarified it was an external **desk fan** and not an internal PC fan.
- **EPYC vs Threadripper for the Ultimate LLM Rig**: Members debated the merits of using an **EPYC 9654** versus a **Threadripper** for running large language models (**LLMs**).
   - The consensus leaned towards **EPYC** for its superior memory bandwidth and dual-CPU capabilities, plus you can consider used **3090s** or **MI50s**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Inference API demands conversational task names**: Members found that **Mistral** models require renaming the variable *'text-generation'* to *'conversational'* to function correctly with the **HF Inference API**, referencing a [relevant issue](https://github.com/langchain-ai/langchain/issues/31434#issuecomment-2936308959).
   - The discovery of **Florence** model's task name requirements highlight that other models may have similar constraints, requiring specific task names to respond to prompts.
- **NivasaAI Navigates Google ADK**: A member is switching **NivasaAI** from **Agent + Tools** to multiple agents with **dynamic routing** for enhanced UX, with the initial commit available on [GitHub](https://github.com/HakAl/adk_rag/commit/c5d70489cb8c19ab1b13cd84948835fa6a1c7d03).
   - A member plans to finalize a router using **Google ADK** that classifies requests into categories such as `code_validation`, `rag_query`, and `general_chat` to handle a range of tasks from syntax checking to casual conversation.
- **Local LLMs Exposed for Hallucinating**: After testing **47 configs**, a member found that local LLMs may hallucinate after **6K tokens**, detailing their findings on **quantization configurations** in a [Medium article](https://medium.com/@poojakanalaa/i-trained-47-different-quantization-configurations-so-you-dont-have-to-c637b274452d).
   - A member introduced a new architecture with a decentralized tokenizer, noting it's not compatible with **llama.gguf** but available on [GitHub](https://github.com/pacific-prime777/architecture_INL_transformer).
- **Self-Hosted Tracing Arrives for OpenAI Agents**: A member open-sourced a [self-hosted tracing and analytics infrastructure](https://github.com/yusuf-eren/openai-agents-tracing) for **OpenAI Agents framework** traces, addressing GDPR concerns and the inability to export data from OpenAI's tracing dashboard.
   - This framework helps developers monitor and analyze the performance of their OpenAI Agents while maintaining data privacy and control.
- **DeepFabric Delivers Data-Driven SLMs**: A member introduced **DeepFabric**, a tool for training **SLMs** to improve structured output and tool calling, shared via a [GitHub link](https://github.com/lukehinds/deepfabric).
   - The tool enables the generation of reasoning trace-based datasets that can be directly loaded into TRL (SFT), enabling developers to train SLMs that excel in generating structured outputs and effectively utilize tools.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton's TMA Triumph on Nvidia SM120**: A member reports implementing **TMA** support for **dot** in **Triton** for **NVIDIA SM120** GPUs, but observed no performance boost compared to `cp.async`. Members indicate that on **Hopper**, **TMA** is less efficient for loads under **4KiB**.
   - A member noticed on **Ampere** that `desc.load([x, y])` performance is poor, given that **Ampere** lacks **TMA**.
- **CUDA's CTA Conundrum: CTA is now Coorperative Thread Array**: In **CUDA**, the community clarified that **Cooperative Thread Array (CTA)** is synonymous with a **thread block**, especially when leveraging the *distributed shared memory feature*.
   - A member shared a [blog post](https://www.aleksagordic.com/blog/matmul) which indicates that using distributed shared memory likely involves **latency and bandwidth** differences between current and other blocks.
- **PyTorch Profiler's CUPTI Predicaments on Windows & WSL**: A user encounters a `CUPTI_ERROR_INVALID_DEVICE (2)` error with **torch.profiler** on Windows and WSL, despite **CUDA** availability, which others suggest may require hardware counter access via [this gist](https://gist.github.com/msaroufim/9e56ce5d42a5e9ccd5e938c83181ea47).
   - A user training an **AlphaZero** implementation on a **3090ti** is experiencing slow training times, but members pointed out that **AlphaZero** training requires substantially more compute power.
- **GPU Dev Resources Debut! Centralized Knowledge Beckons**: A member shares their curated [repository of GPU engineering resources](https://github.com/goabiaryan/awesome-gpu-engineering), consolidating various helpful materials in one location.
   - A member also posted a [blog post](https://blog.sinatras.dev/PMPP-Eval+Journey) about their *PMPP-Eval journey* for people interested in **performance modeling and prediction**.
- **TorchAO Touts Quantization Configuration Triumph**: The `filter_fn` for `quantize_op` will be deprecated in favor of using **ModuleFqnToConfig**, which now supports regex, described in [TorchAO pull request #3083](https://github.com/pytorch/ao/pull/3083).
   - Currently **SGLang** online quantization does not support skipping quantization for vision models, and a user inquired about skipping quantization for the vision model.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **ManusAI V1.5 conjures Web Apps in Minutes**: [ManusAI v1.5](https://xcancel.com/manusai/status/1978854323774792135?s=46) now transforms prompts into **production-ready web apps** in under **4 minutes** flat, incorporating *unlimited context* via auto-offloading and recall, a feature launched on **October 16th**.
   - User reactions varied, with some marveling at its speed and diving into context engineering, while others found previous tools like **Orchids**, **Loveable**, and **V0** *underwhelming* compared to the precision of coding directly.
- **AI Gloom engulfs Researcher Sentiment**: Prominent AI researchers like **Sutton**, **Karpathy**, and **Hassabis** are embracing longer AI timelines, triggering debates about a possible hype cycle *correction* [as noted here](https://xcancel.com/scaling01/status/1979485406816092601?s=46).
   - The community's responses swung between alarm and defense of ongoing progress, amid questions on whether this new wave of pessimism is overblown or simply misinterpreted.
- **Cursor spawns Git Worktrees for Parallel AI Agents**: **Cursor** now auto-creates Git worktrees [as highlighted here](https://xcancel.com/RayFernando1337/status/1979568674433564886?t=CaNW9vyf6jbjoA2qFJdaRw&s=19), enabling users to operate multiple AI agent instances in parallel across separate branches.
   - The launch has garnered praise and triggered a flurry of setup tips, port usage inquiries, and enthusiasm for potential use cases.
- **Tinygrad Turns Apple Macs into NVIDIA eGPU Powerhouses**: The **tiny corp** team announced public testing of their pure-Python driver, which brings **30/40/50-series NVIDIA GPUs** (and **AMD RDNA2-4**) to life via any **USB4 eGPU** dock on **Apple-Silicon MacBooks**, detailed in [this announcement](https://xcancel.com/__tinygrad__/status/1980082660920918045).
   - The channel discussed whether **NPUs** needed streamlined programming and whether **AMD's** success hinges on simplifying the process, as highlighted in [this tweet](https://x.com/__tinygrad__/status/1980082660920918045).
- **AI-Generated Luxe Escapism fuels Facebook Fantasies**: A study funded by **OpenAI** revealed that in **Indonesian Facebook groups** with **30k members**, low-income users (making <$400/month) post AI photos of themselves with Lambos, in Paris, or at Gucci stores ([link](https://xcancel.com/itstimwijaya/status/1979814111069553137?s=46)).
   - Discussion centers on whether this trend is purely geographic or socio-economic, drawing parallels to past Hollywood dreams and generative-AI photo apps.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RTX 3090 Tokes LLM Training**: Members estimated training speeds for a **30M parameter LLM** on an **RTX 3090**, ranging from hundreds to thousands of tokens per second (TPS).
   - One member reported **120 kt/s** on a **4090** using **30m rwkv7 bsz 16 seqlen 2048**, leading to expectations of *high thousands* of TPS.
- **Anthropic Extends Attribution Graphs to Attention**: The discussion explores extending **attribution graphs** to **attention mechanisms**, inspired by [this video](https://youtu.be/hdi1a9MjwDs?si=taIuYbeF6v-yRSxI&t=628) and **Anthropic** explored this in a follow-up post, detailed in the paper [Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html).
   - The original **Biology of LLMs** paper only looked at **MLPs** while freezing attention, as well as the paper [https://arxiv.org/abs/2510.14901](https://arxiv.org/abs/2510.14901).
- **Eval Harness Revamp Ready**: A meeting is scheduled to discuss additions to the **eval harness**, focusing on sharing current plans and gathering feedback from library users, with details available [on When2Meet](https://www.when2meet.com/?33070160-Bw5xm) and [on this branch](https://github.com/EleutherAI/lm-evaluation-harness/tree/smolrefact).
   - Key improvements include: **new templates** for easy format conversion, standardizing formats, making instruct tasks more intuitive, and general UX improvements (e.g. **`repeats`**), with the goal of enabling easier conversion between task variants (e.g., **MMLU** -> cloze -> generation).
- **AI Paper Punctuation Police**: A discussion critiqued the writing style in AI papers, particularly regarding comma usage, referencing the paper [https://arxiv.org/abs/2510.14717](https://arxiv.org/abs/2510.14717).
   - One member quipped *I unironically believe that the writing of half of AI papers would be improved if you gave the authors a comma limit*, to which another replied *Then they’ll switch to semicolons or em dashes*.
- **NormUon Optimizer Enters the SOTA Race**: A member mentioned [a new optimizer](https://arxiv.org/abs/2510.05491) that *looks like SOTA* if the results are good.
   - Multiple sources say *it's the same perf as muon on their non-speedrun setups but with good muon baselines* with the observation that `modded-nanogpt does qk norm which is one way you can avoid logit blowups.`



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 trounces Prediction Markets**: One user shared that **Kimi K2** is the best model for working with prediction markets and has been using it, as stated in [this tweet](https://x.com/rauchg/status/1979660103675687157?s=46&t=_NtP_RUn04yF_4hD_VEDkQ).
   - They did not mention their specific use case for **prediction markets**.
- **Groq's Kimi faces Growing pains**: **Groq's** implementation of **Kimi** experienced a period of instability, with intermittent functionality issues.
   - According to a user, the implementation is now back to normal, but provided no specific details on the issues.
- **Kimi K2 Cracks BSODs**: A user praised **Kimi K2's** ability to provide solid troubleshooting advice for computer problems, and specifically suggested the use of **verifier** for stress-testing drivers that may be causing **BSODs**.
   - The user did not include specific details on the kind of troubleshooting advice given, only stating that it was solid.
- **DeepSeek Falls Short of Moonshot AI**: A user expressed a preference for **Moonshot AI's Kimi** over **DeepSeek**.
   - The user acknowledged that this preference was a personal opinion, with no supporting evidence or reasoning provided.
- **Codex not Cutting It**: Users discussed various CLI tools for working with MCP servers and models like **DeepSeek**, **GLM Coder**, and **Qwen**, highlighting **Claude Code** and **Qwen-code** as solid options.
   - The consensus was that **Codex** is only ideal when using with **OpenAI** models; no specific reasons for this preference were provided.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GLM 4.6 Opens OS Community Horizons**: The release of **GLM 4.6**, capable of running locally, aims to reduce reliance on proprietary models from companies led by **Sam Altman**, **Elon Musk**, and **Dario Amodei**, according to [this YouTube video](https://www.youtube.com/watch?v=bOfoCocOjfM).
   - Meanwhile, the rise of new **Chinese LLMs** like **Ant Group's** Ling models being separate from **Alibaba's** Qwen team also prompts debate about the necessity of multiple models, given the existing presence of several strong contenders like **Qwen**.
- **ScaleRL Appears Perfect for Sparse Reasoning Models**: According to Meta's recent Art of Scaling RL paper, [**ScaleRL**](https://x.com/Devvrit_Khatri/status/1978864275658871099) seems tailor-made for **sparse reasoning models** such as **MoE**.
   - Members are questioning whether **trajectory-level loss aggregation granularity** performs better than **sample-level** ones for **Iterative RL (Reasoning)**, with discussion on whether *step/steps (iteration) level might be a middle ground* for tuning.
- **New Healthcare AI Safety Standards**: A member proposed research on **AI safety** with a focus on clinical/healthcare applications, to create benchmarks for AI models, referencing the [International AI Safety Report 2025](https://assets.publishing.service.gov.uk/media/679a0c48a77d250007d313ee/International_AI_Safety_Report_2025_accessible_f.pdf).
   - The goal is to propose more accurate metrics, with the member seeking feedback on whether this is a good research topic.
- **Nous Champions Decentralization with Psyche**: **Nous Research** is embracing decentralization through open-source methodologies and infrastructure implementations, exemplified by **Psyche**, as evidenced by these links [Nous Psyche](https://nousresearch.com/nous-psyche) and [Stanford paper](https://cs.stanford.edu/~gakiwate/papers/sigcomm25-centralization.pdf).
   - A member stated, *"Nous successfully decentralizes with their open source methodologies and infrastructure implementations."*



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Google Drive Connection Allegedly Operable**: Users reported that [Manus connects to Google Drive](https://drive.google.com) by clicking the **+** button and adding files.
   - The user confirmed it's *not available as a connected app*.
- **Users Report Manuscript Project Files Vanishing with Credits**: A user building a site reported that after spending **7000 credits**, everything in the project disappeared, including files and the database, and support was unhelpful.
   - Another user echoed this, reporting losing nearly **9000 credits** and finding *no files or preview*.
- **Manus Used to Create Android Frontend App MVP Cheaply**: A user created an Android frontend app MVP using **525 Manus credits**, then used Claude to fix issues, and praised Manus' UI/UX capabilities.
   - The user shared [images of the app](https://cdn.discordapp.com/attachments/1349440650495398020/1429746630935969802/2025-10-19_18-06.jpg?ex=68f7eb90&is=68f69a10&hm=eb47b7c3e935587fd229b70648d0dcf7043ea52556918a996c0872722972e7b7&).
- **Manus Infrastructure has Outage**: Manus experienced a temporary outage due to an infrastructure provider issue, with some users in certain regions still facing errors accessing the homepage.
   - The team communicated updates and thanked users for their patience, reporting that *most Manus services are back online*.
- **Free Perplexity Pro Promo Sparks Discord Drama**: A user shared a [referral link for a free month of Perplexity Pro](https://pplx.ai/muhammadze84187), prompting a negative reaction from another user who told them to *Stfu!* and *Get a job*.
   - The linked user said it was a *no drama, no lies, no clickbait* way to earn real money.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Compiling Itself?**: Mojo's compiler, leveraging **LLVM** and tightly integrated with it by lowering to **MLIR**, faces challenges in achieving full self-hosting unless it incorporates LLVM.
   - While a complete rewrite to Mojo is technically feasible, it's deemed *a lot of work for not a ton of benefit*, but C++ interop might make it easier in the future.
- **Mojo as a MLIR DSL**: Mojo functions as a specialized **DSL for MLIR**, parsing directly into **MLIR** and utilizing it over traditional ASTs, which grants it considerable adaptability and flexibility.
   - Mojo's architecture features numerous dialects, primarily new ones apart from the **LLVM dialect**, positioning it for diverse computational applications.
- **MAX Kernels Backending PyTorch**: Interest is sparked to switch the backend of **JAX** to use **MAX kernels**, potentially interfacing with **C++** as a fun project, noting a mostly functional **MAX backend** for **PyTorch** already exists.
   - While Mojo can `@export(ABI="C")` any **C-abi-compatible function**, direct communication with **MAX** currently requires Python.
- **Mojo Aims for Pythonic Prevention**: Mojo's design seeks to avert the fragmentation seen in Python between users of Python and CPython by providing avenues to transition between more pythonic `def` and more systems-oriented `fn` code.
   - A member emphasized that *the goal is to leave the door open to lower-level control but give safe default ways for people to not shoot themselves in the foot; let's hope we can deliver.*
- **Mojo Debates UDP Socket Support**: Users questioned **UDP socket support** in Mojo, with a response that it's possible via **libc**, but full standard library support is pending for proper implementation.
   - The answer indicated a preference for doing it *"right not fast"* and cited *"language-level dependencies"* as a factor.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Pilots Claude-Powered Agents**: Members explored integrating **Claude agents** into **DSPy** programs, referencing [a prior implementation](https://x.com/dronathon/status/1979042187733004291) with a **Codex agent**.
   - The primary difficulty lies in the lack of an SDK for **Claude code**; the community wants to see example agents.
- **Clojure Contemplates Concurrency with DSPy**: A member inquired about adapting DSPy to **Clojure REPL** environments, particularly regarding **data representation**, concurrent **LM calls** due to immutability, and **inspecting generated functions**.
   - The nuances of adapting DSPy's **Python paradigms** to Clojure's functional and concurrent nature are not yet fully explored.
- **DSPy Grapples with Generics for Typing**: The community debated the feasibility of **fully typing DSPy** in Python, with a focus on whether **Python generics** suffice.
   - While one member expressed confidence, emphasizing optimizing the **input handle/initial prompt**, they cautioned against premature optimization without clear **task and scoring mechanisms**.
- **Gemini Geodesy for Google API Keys**: Users discussed using **Gemini models** within **dspy.lm**, confirming it's achievable via [proper configuration](https://dspy.ai/#__tabbed_1_4) and **API key** setup.
   - A member jokingly shared their 'painful journey' in finding the correct API key, recommending **AI Studio** over the console.
- **LM Studio gets llms.txt generator**: A member shared a [llms.txt generator](https://github.com/AcidicSoil/lms-llmsTxt/tree/main) for **LM Studio** powered by the **DSPy framework**.
   - The tool allows users to easily generate `llms.txt` for repos lacking it, leveraging any LLM available in LM Studio; the member recommended using *"osmosis-mcp-4b@q4_k_s"* for generating example artifacts.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Tianshou Boosts RL in PyTorch**: A member suggested [Tianshou](https://github.com/thu-ml/tianshou) as a viable **reinforcement learning framework** for PyTorch, noting its relevance to graph neural networks.
   - The recommendation was received well by the community, furthering interest in practical **RL** implementations.
- **What even is an AI Engineer??**: Members debated the qualifications for an **"AI Engineer,"** joking that even [using the OpenAI API in Python](https://platform.openai.com/docs/api-reference) might be enough to impress on LinkedIn.
   - The discussion highlighted the varied interpretations of the role, from assembling Legos in visual n8n to creating custom GPTs.
- **Cracking the ML Debugging Interview**: Members discussed how to prepare for an **"ML Debugging"** coding interview, recommending readiness to discuss [handling overfitting](https://www.youtube.com/watch?v=PykNdM4v4Xo).
   - ChatGPT was also suggested as a mock interview tool and the conversation was recorded [here](https://chatgpt.com/share/68f68935-3148-8005-907f-86ec2ed6e93c).
- **Qwen3 sees clearly**: A new **Qwen3 Vision model** has been released and is available on [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
   - Members seemed curious about how it compared to other vision models such as those on [ComfyUI](https://comfyui.com/).
- **Machine Unlearning Seeks ArXiv Fame**: Members discussed advocating for **papermachine unlearning & knowledge erasure** to be recognized in its own arXiv category (cs.UL/stat.UL).
   - The promotion would highlight the growing importance of **data privacy and security**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Still Kicking, Releases Just Slower!**: Users discussed **Aider's** development pace, clarifying it's active but with slower releases, recommending **aider-ce** for more frequent updates and cloning [GitHub](https://github.com/Aider-AI/aider/graphs/contributors) to familiarize with the codebase.
   - Community members are aiming to boost the project through increased commit frequency.
- **Aider Eyes Agentic Extension Integration**: A developer is building an agentic extension for **Aider** using **LangGraph**, including task lists, RAG, and command handling.
   - Discussion centered on whether to integrate the extension directly into Aider or keep it a separate project, highlighting the importance of maintaining Aider's simplicity while competing with top-tier agent solutions.
- **Devstral Small Model: Sleeper Hit?**: The `Devstral-Small-2507-UD-Q6_K_XL` model is receiving accolades for surprisingly strong performance on limited hardware (32GB RAM), with self-correction and large context handling, especially with [Unsloth's XL quantized versions](https://github.com/unslothai).
   - The user found that the model is outperforming `Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL`, in PHP, Python, and Rust coding tasks, supporting image, and should be added to Aider benchmarks.
- **Aider-CE Takes Aim at Codex CLI Crown**: One user switched back to **Aider** after testing gemini-cli, opencode, and Claude (with claude-code-router for DeepSeek API), praising its grep-based code search/replace and self-updating todo list.
   - The user also highlighted the value of Aider's simplicity and straightforwardness for coding tasks with MCP formatting in .aider.conf.yml).
- **Reasoning Timeouts Plague Commit Messages**: A user reported that using **Deepseek V3.1 Terminus** via **OpenRouter** for commit message reasoning was too slow, motivating them to disable it.
   - An alternative was suggested: copy the API's reasoning in resources and setting a new alias to a weak model.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Karpathy Launches Minimal Chat App**: Andrej Karpathy launched [nanochat](https://github.com/karpathy/nanochat), a minimalistic chat application, but the poster questioned how significant the release was.
   - No other details provided.
- **ShapeTracker Faces Deprecation**: tinygrad plans to deprecate **ShapeTracker** during meeting #92, along with discussions on [usb gpu](https://x.com/__tinygrad__/status/1980082660920918045), **multi output kernel**, and **FUSE_OPTIM**.
   - Additional topics included rangeify regressions, openpilot, resnet, bert, assign, cleanups, viz, driver, tiny kitten, symbolic processing, bounties, new linearizer, and clang2py.
- **Contributor Requests Performance Metrics**: A contributor requested the addition of **MFLOPS** and **MB/s** metrics, displayed in yellow on the **DEBUG=2** line for better performance monitoring.
   - The contributor explicitly asked that *clean code* be written and cautioned against using *AI code you don't understand*!
- **macOS Nvidia Drivers Finally Arrive**: Nvidia drivers for macOS have been successfully produced, opening up possibilities for GPU-accelerated tasks on macOS.
   - To enable the drivers, users are instructed to run `brew tap sirhcm/tinymesa`.
- **TinyJit Gradient Accumulation Troubles**: Members identified gradient accumulation issues in **TinyJit**, particularly in [model_train.py](https://github.com/tinygrad/tinygrad/blob/c7c59e6dd71158f50bbb9a87298b4ed1d65a6fb6/examples/mlperf/model_train.py#L1375C1-L1390C54), questioning the math regarding gradient accumulation.
   - A member solved the issues by rewriting the gradient addition step using assign to ensure it worked correctly.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **DevOps Admin Explores Secure MCP Access**: A DevOps admin is exploring secure methods for granting **MCP access** to non-technical users within their organization, with a focus on avoiding key management and implementing an **Identity Provider (IDP)** layer.
   - Solutions under consideration include [Webrix MCP Gateway](https://docs.webrix.ai/docs/admin/monitor-logs) and [making Docker MCP Gateway multi-tenant](https://github.com/docker/mcp-gateway/issues/130).
- **MCP auth extension promises enterprise-managed auth**: It was suggested that the **enterprise managed auth profile**, which is being released as an **MCP auth extension**, is designed to address the DevOps admin's needs.
   - However, current fine-grained permissions are limited to **oauth scope granularity**.
- **Discord Channel Aimed at Contributors**: A member clarified that the Discord is intended for communication among **MCP protocol** and related projects contributors, rather than technical support.
   - Users seeking help were encouraged to DM for links to appropriate communities.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1428822532936569074)** (1099 messages🔥🔥🔥): 

> `Comet Referral Program, AWS Outage, GPT-5 vs Claude, Scientific Method, MCP` 


- ****Discord Discusses Dollars for Referrals****: Members discuss the **Comet referral program**, with some users asking for help on [how to properly execute referral links](https://www.perplexity.ai/browser/invite-ga).
   - Some users report **missing referrals or payouts**, while others share tips for success, such as ensuring referred users install Comet and perform a search.
- ****AWS Outage Ominously Overwhelms Online Operations****: A recent **AWS outage** caused widespread issues for Perplexity AI and other services, leading to discussions about the [risks of over-reliance](https://health.aws.amazon.com/health/status) on single cloud providers.
   - Users shared status links and personal experiences, with some noting that **certain features were still affected** even after the initial outage was resolved.
- ****GPT-5 Gets Grounded: Claude Crowned King****: Members debated the relative merits of **GPT-5 versus Claude**, with many finding **Claude** to be superior for complex projects and reasoning tasks.
   - Some users noted that **GPT-5's free tier** felt noticeably weaker, while others suggested using **Claude 4.5** for optimal results.
- ****Scientific Method Schooled for Scrutinizing Studies****: A user asked for guidance on using AI for academic research, prompting a discussion on the **scientific method** and its application in building research frameworks.
   - Experienced members advised breaking down the process into steps, referencing successful papers, and leveraging the Perplexity API in combination with tools like Google Colab.
- ****MCP: Mystery Context Provider Proves Perplexing****: Members inquired about **Local and Remote MCPs (My Context Providers)**, with some struggling to find the PerplexityXPC in Connectors, and others wondering if [MCPs were worth learning](https://www.perplexity.ai/help-center/en/articles/11502712-local-and-remote-mcps-for-perplexity).
   - This led to a broader discussion of Local access to files/device and the automation opportunities around this technology.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1429430526720217189)** (11 messages🔥): 

> `Perplexity AI, TikTok Video, Claude Sonnet 4.5, Shareable Threads, AWS Dashboard` 


- **Perplexity Powers Polished TikTok**: A member shared a [TikTok video](https://vm.tiktok.com/ZNd73p4Gm/) made with help from **Perplexity AI**.
- **Perplexity AI Steps Up as Guide**: A member shared a **Perplexity AI** [claim invite link](https://www.perplexity.ai/browser/claim-invite/MzRkZDY5NWYtZTVlMy00ZTY4LTg4MDUtMTRjMDNiYjZiZDdi) recommending it for **step-by-step guides**.
- **Claude 4.5's Philosophical Forte**: **Claude Sonnet 4.5** is highlighted as effective for **philosophical problem solving**, accompanied by a [shared Claude conversation link](https://claude.ai/share/886d8469-3dd2-4e46-b491-c28e5131985d).
- **Shareable Threads Savior**: The **Perplexity AI** bot prompted multiple users to ensure their threads are *shareable*, linking to a [relevant Discord channel message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **AWS Ace-essibility with Perplexity**: A member shared a **Perplexity AI** search regarding creating an **AWS dashboard** [link](https://www.perplexity.ai/search/can-you-make-me-a-aws-dashboar-gAh.JkSmRpmGvAjikOOV4w#1).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1429037115617968191)** (8 messages🔥): 

> `API pricing, ZTT server` 


- **Contacting Perplexity for API Pricing**: A user inquired about who to contact for custom **API pricing** for large API users.
   - Another user suggested emailing [api@perplexity.ai](mailto:api@perplexity.ai) to get in touch with the appropriate team.
- **Request for ZTT Server Invitation**: A user asked for an invitation to the **ZTT server**.
   - No further context or details were provided regarding the **ZTT server** or its purpose.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1428821310758326294)** (1182 messages🔥🔥🔥): 

> `Lithiumflow's Coding Capabilities, Gemini 3 Speculation, AI Video Generation Quality, Claude Sonnet 4.5` 


- **Lithiumflow's Coding Skills: Gemini's Coding Focus?**: Members tested **Lithiumflow** and found that it has impressive capabilities, writing a full macOS system with functioning apps in a single HTML file, showcased on [Codepen](https://codepen.io/Kross-the-scripter/pen/emJVRyx), with positive feedback.
   - Despite this, some users found that **Lithiumflow** is inferior to **Claude 4.5** for coding, indicating it may be a nerfed or Pro model rather than an Ultra model, with discussions suggesting **Google** is prioritizing coding abilities in its models.
- **Gemini 3: When Will it Arrive?**: There is much speculation regarding **Gemini 3's** release and specifications, and some members are theorizing that **Lithiumflow** and **Orionmist** could be nerfed versions of Gemini 3 or specialized coding models, while a tweet suggests the next release won't occur for another two months, viewable on [Twitter](https://x.com/OfficialLoganK/status/1980435968323907884).
- **AI Video Quality: Sora vs. Veo 3.1**: Members are debating the video quality of different AI models, including **Sora 2** and **Veo 3.1**, viewable on [ChatGPT's website](https://sora.chatgpt.com/p/s_68f6acd380788191b301485853f831fc), with some stating that the quality of Sora has degraded since its initial release, and expressing confusion about why Veo 3.1 is ranked higher on leaderboards.
   - The consensus appears to be that even though **Veo 3.1** is ranked higher, that it is not better than **Sora 2**.
- **Claude Sonnet 4.5: Is It Worth the Hype?**: Members discuss **Claude Sonnet 4.5's** capabilities, particularly in creative writing, with one member stating that **Claude Sonnet 4.5 Thinking** is in *"a different league"* than **Gemini 2.5 Pro**.
   - However, others mention ongoing bugs and issues with models getting stuck or generating generic error messages on LMArena, hindering their ability to test and compare effectively.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1429878597346398268)** (1 messages): 

> `Text-to-Video Leaderboard, Image-to-Video Leaderboard, Veo-3.1 ranking` 


- **Veo-lociraptor: Veo-3.1 Dominates Video Leaderboards!**: The [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) & [Image-to-Video Leaderboard](https://lmarena.ai/leaderboard/image-to-video) now show **Veo-3.1 ranking #1 in both** categories.
   - Users are encouraged to share their **Veo-3.1** generations and provide feedback.
- **Arena X Post**: The leaderboards were also [announced on X](https://x.com/arena/status/1980319296120320243).
   - Check out how Video Arena works and share generations in the discord channels.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1428843902626037910)** (1106 messages🔥🔥🔥): 

> `Apple hardware support in Unsloth, AMD x PyTorch Hackathon, Training reasoning models for legal systems, Synthetic data generation with synonyms, Choosing models for coding tasks` 


- **Unsloth eyes Apple Silicon Support**: Support for Apple hardware is planned for Unsloth sometime next year, but as one member noted, *take my word with a grain of salt...*.
   - No further information or timeline was disclosed at this time.
- **AMD x PyTorch Hackathon Extension announced**: The AMD x PyTorch Hackathon is extended and to compensate for disruptions, each participant receives **100 hours of 192GB MI300X GPUs** to use even outside the hackathon.
   - Community members described the event as having *so many issues* but the team is *trying*.
- **Modelscope mirrors Huggingface to circumvent AWS**: Due to [AWS outages](https://health.aws.amazon.com/health/status) affecting Hugging Face, a member recommends using [Modelscope](https://modelscope.cn/models/unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit) (a Chinese HF mirror) as a temporary alternative.
   - A code snippet was shared describing how to load models from Modelscope in a Colab environment, with continued need of setting the `load_in_4bit` flag.
- **Consulting Advice triggers liability fears**: A user seeking guidance on commercially using a fine-tuned LLM for real estate messaging was cautioned against free advice due to potential liability concerns.
   - Another member clarified that such concerns apply only when providing detailed step-by-step guidance, not for suggesting general model or dataset choices.
- **GPT-OSS finetuning seeks "policy" removal**: Members discussed using RLHF to penalize **GPT-OSS** for generating policy-related content or refusing to answer questions, with the goal of creating a less censored model.
   - Suggestions included removing the *Chain of Thought* to reduce censorship, but it may make it dumber, but uncensored*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1428877670632132740)** (8 messages🔥): 

> `Software Engineer Introduction, AI Bot Development, New Tricks for Old Dogs` 


- **Software Engineer Specializes in AI Projects**: A software engineer specializing in AI project development introduced themselves, offering services such as **automation tasks**, **NLP**, **model deployment**, **text-to-speech**, **speech-to-text**, and **AI agent development**.
   - They highlighted proficiency with tools like **n8n**, **Zapier**, **Make.com**, **GPT-4.5**, **GPT-4o**, **Claude 3-7 sonnet**, **Llama-4**, **Gemini2.5**, **Mistral**, and **Mixtral**, also providing a [link to their portfolio](https://akari-hiroshi-dev.vercel.app/).
- **AI Bot Dev Enters the Chat**: A developer primarily focusing on **AI bot development** introduced themselves, also mentioning capabilities in **gaming** and **scraping**.
   - No other details were given.
- **Big Tech Bot Dev Going Deeper**: A dev with experience in big tech developing bots using **ChatGPT** expressed enthusiasm for **Unsloth**.
   - They described themselves as *an old dog learning new tricks* and stated that they're *going deeper now*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1428867345077764159)** (426 messages🔥🔥🔥): 

> `Qwen 2.5 VL Issues, Hackathon Challenges and Synthetic Data, GPU vram burden, Diff2flow, RP Stat thing` 


- **Qwen 2.5 VL Can't Understand Images**: A member reported that **Qwen 2.5 VL** couldn’t understand images, providing a [GitHub link](https://github.com/Emericen/tiny-qwen) to the code.
   - The member suspected HuggingFace of embedding watermarks to create unfair competition, but clarified that the issue was that every model did not work at that point.
- **Hackathon Halted Due to Lost Work**: The Unsloth hackathon was paused due to some participants losing their work, prompting an investigation to ensure a fair competition, with promises of an even better prize.
   - Participants discussed using synthetic data kits, generating Q&A pairs from models, and the challenges of completing the task within the time limit with a 192GB restriction.
- **Diff2flow Project Revealed**: A member shared details about a project called **diff2flow**, which converts existing **eps** or **vpred** models to flow matching, requiring specific hyper-parameters and datasets.
   - They recommended using **S3 on AWS** for data storage, cautioning about ingress/egress costs, and noted that it's different from the identifying watermark project.
- **Tackling Scene Graph Algorithmic Nightmares**: A member expressed frustration with building a scene graph for an **RP stat thing**, describing it as a code nightmare with numerous edge cases.
   - The discussion explored using structured outputs and traversing a simple tree, but the challenge lies in capturing a naturally emerging scene graph from incomplete data.
- **Ultravox Adapter Struggles with Speech-to-Text**: A member released their first speech-to-text model based on **Ultravox** and **Qwen 3 4B Instruct**, noting it's currently subpar and barely hears input.
   - They're doing a new run with **1152k steps** vs **256k steps** on the old adapter, and pointed out the special thing with ultravox is that it isn't asr to llm to tts, the speech input gets processed by whisper but with the output being a **768 dim space**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1428925897645228183)** (73 messages🔥🔥): 

> `FailOnRecompileLimitHit, Gemma3-270m decoding, TTS and ASR model training, GRPO recipe for gpt oss 20b, QWEN2.5 7B chat template` 


- ****FailOnRecompileLimitHit** Arises in Unsloth RFT Notebook**: A user encountered a **FailOnRecompileLimitHit** issue while trying the GPT OSS 20B unsloth reinforcement fine tuning notebook on an **H100 80G**, suggesting adjusting the setting shown in the error message.
   - Another user suggested increasing the `torch._dynamo.config.cache_size_limit` or sorting the dataset by size might help.
- **Cracking the Code: Decoding **Gemma3-270m** Predictions**: A user was struggling to correctly decode predictions when training **gemma3-270m**, aiming to compare labels to predictions in `compute_metrics`.
   - A possible solution involves setting `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` or checking [this GitHub issue](https://github.com/unslothai/unsloth/issues/2257) for adjusting the HF code or generating regularly inside `compute_metrics`.
- ****TTS and ASR** Model Training Guide Available**: A user inquired about training a **TTS and ASR** model on local languages using Unsloth locally, with data in CSV format.
   - A link to the [Unsloth TTS guide](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning) was provided for assistance.
- ****GRPO** Recipe Struggles for **GPT OSS 20B****: A user shared that the **GRPO** recipe for **gpt oss 20b** seems to be struggling even after 100 steps and [linked to the colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-GRPO.ipynb).
   - They noted they made modifications to get it running on Modal.
- ****QWEN2.5 7B** Model Chat Template Integration Asked About**: A user asked about how to ensure Unsloth applies the chat template during fine-tuning of a **QWEN2.5 7B** model.
   - Another user shared screenshots regarding chat template applying process for Gemma3.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1429381502646358048)** (14 messages🔥): 

> `Luau-Qwen3-4B-FIM-v0.1, Training Configurations, Luau-Devstral-24B-Instruct-v0.2, Brainstorm adapter` 


- **Luau-Qwen3-4B-FIM-v0.1 Training Finishes Swimmingly**: A member announced the completion of training for [Luau-Qwen3-4B-FIM-v0.1](https://huggingface.co/TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1), and shared clean graphs showcasing the results.
   - Other members congratulated them and noted the clean graphs.
- **Luau-Devstral-24B-Instruct-v0.2 Beats GPT-5 Nano**: A member pointed out that [GPT-5 Nano in Luau-Devstral-24B-Instruct-v0.2](https://huggingface.co/TorpedoSoftware/Luau-Devstral-24B-Instruct-v0.2) outperforms GPT-5, noting its surprising performance.
   - The member remarked that *GPT-5 is weird*.
- **Training Configs are Shared for Qwen3**: A member shared their detailed training configuration for **Qwen3**, including parameters such as `per_device_train_batch_size = 2` and `learning_rate = 2e-6`.
   - Another member found the information very helpful in understanding the settings for training **Qwen3s**.
- **Brainstorm Adapter to Boost Metrics**: A member suggested adding a **Brainstorm (20x) adapter** to the model to potentially increase metrics and improve long generation stability.
   - The original trainer welcomed the suggestion and expressed interest in the potential benefits.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1429086624175620219)** (12 messages🔥): 

> `AI model thinking modes, xLLMs Dataset Collection, Double descent history and papers` 


- **Thinking Mode XML Tags Teach New Tricks**: A member proposed using XML tags such as `<thinking_mode=low>` to teach an AI model different thinking modes during SFT, categorizing examples based on CoT token count, with an **auto** mode for the model to decide.
   - Another member suggested that controllable thinking usually benefits from some **reinforcement learning after SFT**, while acknowledging the lack of compute power for RL, planning to experiment with SFT only.
- **xLLMs Datasets: A Multilingual Bonanza**: The **xLLMs** project introduced a suite of multilingual and multimodal dialogue datasets on [Hugging Face](https://huggingface.co/collections/lamhieu/xllms-66cdfe34307bb2edc8c6df7d), designed for training and evaluating advanced conversational LLMs on capabilities like **long-context reasoning** and **tool-augmented dialogue**.
   - One highlight is the [xllms_dialogue_pubs dataset](https://huggingface.co/datasets/lamhieu/xllms_dialogue_pubs), ideal for training models on **long-context reasoning**, **multi-turn coherence**, and **tool-augmented dialogue** across **9 languages**.
- **Double Descent History Visualized**: A member shared a [YouTube video](https://www.youtube.com/watch?v=z64a7USuGX0) explaining the **double descent history** and related papers visually.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1428826602829254707)** (569 messages🔥🔥🔥): 

> `Comcast data selling, Stealth model, Grok video, Sora 2 video quality, Sora 2 invites` 


- **Comcast reps steal data and distribute it, claims user**: A user claimed **Comcast** representatives are stealing data and selling it through sophisticated phishing attempts targeted through authentic **Kenyan organizations** and **Philippines-based call centers**.
   - They are documenting cases for collective escalation and suggested directly contacting the CEO Brian Roberts to be held accountable.
- **OpenCode Testers Discover Super Speedy Stealth Model**: The **OpenCode** team has a mysterious **stealth model** for the weekend that is incredibly fast at spitting out tokens.
   - They're unsure if it's intelligent but noted it's the fastest model they've ever seen.
- **Grok Launches Video Imagine Feature**: **Grok** is now doing video generation, accessible via **SuperGrok** subscription ($30/month) without requiring a **Twitter/X** account.
   - Users are finding Grok video generation copes fine with franchises and 3rd party content, producing videos without watermarks.
- **Veo and Grok Video Generation Benchmarked Against Sora 2**: Members are discussing that **Veo 3.1** is roughly on the same level as **Sora**, but it's not free and should be available in Australia, while another agreed that **OpenAI is 20x ahead**.
   - One user is trying to create a video with a **KSI cameo**, but wasn’t sure what prompts to use, and another had one **video scene declined as too violent**.
- **Qwen3 finetuning on Unsloth Library hits roadblock**: A user is facing issues finetuning the **Qwen3-2.35B-A22B-Instruct-2507** model when saving checkpoints using a llama factory and the unsloth library.
   - Another user suggested it sounds like running out of memory and that saving checkpoints—especially full models or even LoRA weights—can cause **VRAM exhaustion** because it involves duplicating tensors for serialization, which doesn't benefit from training-time optimizations like gradient checkpointing.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1428846059106472057)** (42 messages🔥): 

> `Agentic AI Hackathon, VPN use with ChatGPT/Sora, DALL-E private image generation, ChatGPT access in universities, Sora Access and "k0d3"` 


- **Hackathon for Agentic AI Sought**: A member inquired about an **agentic AI hackathon**, prompting responses regarding upcoming hackathons hosted by **OpenAI** and **Microsoft**.
- **VPN Ban Risk when Streaming Sora**: Members discussed the risk of using a **VPN** to access **Sora** from countries with geographical restrictions like Argentina, stating that doing so violates the **Terms of Service**.
   - The ToS violation could lead to a ban.
- **DALL-E Privacy Questioned**: Members questioned whether **DALL-E** offered private image generation similar to **Midjourney**, and if using **GPT** was allowed during **CTFs**.
- **Free Microsoft-Sponsored GPT for Universities**: A member shared that some universities provide access to **ChatGPT** through their **Microsoft** packages, often via **CoPilot**.
   - Others clarified that **Microsoft** provides infrastructure, and if a university utilizes Microsoft's tools then **CoPilot** is how to get access to **GPT**.
- **Sora 'k0d3' and IOS App Mentioned**: Members discussed accessing **Sora**, suggesting it has a separate website and an **iOS app**, as well as **invite codes** to gain access to **Sora 2**.
   - One member mentioned that to get access to Sora 2, you have to go to [Sora.com](https://sora.com) and enter a *k0d3* if you are in North America.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1428826242756382792)** (51 messages🔥): 

> `Sora's ability to distinguish real vs fictional images, Prompt engineering for Sora, Controlling ChatGPT output, Sora screenshake VFX, Learning prompt engineering` 


- **Sora's Reality Check: Distinguishing Real vs Fictional Images**: A user inquired if **Sora** can distinguish between images of real people and fictional characters.
   - Another user simply responded *"Yes"*.
- **Bangladeshi User Aims for Photorealistic Sora Videos but Encounters Roadblock**: A user from Bangladesh wants to create a photorealistic video of a person using AI and shared that they were told AI doesn’t support photo-realistic photos.
   - Another user asked about prompting ChatGPT to only accept certain sources and reject others.
- **Screenshake Success: A Tremor in the AI World?**: A user asked about achieving **screenshake** or other **VFX** in **Sora 2**.
   - One user responded *"Yes, barely"*, suggesting **screenshake** might be downvoted due to human preference testing, and shared links to **Sora** generations experimenting with **screenshake** like [this one about an earthquake](https://sora.chatgpt.com/p/s_68f5ea1683948191be038ab282d1eb61).
- **Prompt Engineering: The Future is Now**: Users discussed methods to learn and apply prompt engineering, with one user sharing a link to their personal **Sora** account dedicated to making movie trailers.
   - One member advised against relying on *'master prompts'*, suggesting instead to focus on clear communication and accurate language, emphasizing the importance of fact-checking and verifying the output.
- **Taming the Chatbot: Strategies for Controlling ChatGPT's Conversational Habits**: A user expressed frustration with **ChatGPT's** tendency to end responses with unsolicited follow-up questions and sought advice on disabling this *'feature'*.
   - A user suggested replacing the follow-up questions with something else, such as a joke or a detail about a favorite subject, and shared example chats, like [this one with Dad jokes](https://chatgpt.com/share/68f6b5ec-04e0-8011-b4ad-8342ee1a0405).


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1428826242756382792)** (51 messages🔥): 

> `Sora recognizing real vs fictional images, Sora account for movie trailers, Prompt engineering resources, Screenshakes or other VFX in Sora 2` 


- **Sora juggles real vs unreal images?**: A member inquired if **Sora** can distinguish between real person images and fictional character images, with a simple response in the affirmative.
   - The conversation starter also shared a link to *pseudoCode* thought experiment for AI models.
- **Sora account dedicated to movie trailers debuts**: One member shared that they have a **Sora account** dedicated to making movie trailers, calling it helpful.
   - The member thanked others for the resources and expressed excitement about the potential of Sora.
- **Users Share Prompt Engineering Tips**: Members discussed techniques for effective prompting, including using **hierarchical communication with markdown**, abstraction with variables, and ML format matching.
   - Another user cautioned against relying on "master prompts", instead advocating for clear communication and iterative refinement based on the AI's output.
- **Sora 2 tries screen shakes and VFX**: A user asked about getting **screenshakes or other VFX on Sora 2** and another said that it was possible *barely*.
   - Links to various screenshake experiments on **sora.chatgpt.com** were shared, demonstrating different levels of success.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1428823206856228996)** (125 messages🔥🔥): 

> `TLRAG framework, Deterministic AI, Model Agnostic Framework, OpenRouter user demographic, AI Slop` 


- **TLRAG Framework pitched as Deterministic and Model Agnostic**: A developer is pitching a framework called **TLRAG** as fully deterministic and model agnostic, claiming it doesn't need function calling or regular frameworks like **Langchain** and has no dependencies, showcased on a [dev.to page](https://dev.to/tlrag).
   - The framework claims to save +90% of tokens, independently save memories, curate chats, learn, evolve, and change its own identity and prompts, however, its website and whitepapers faced community scrutiny.
- **Website UI/UX glitches and Security Concerns**: Users gave feedback on the website's UI/UX issues and raised concerns about security, while one member noted that **TLRAG** claims are true, and **TLRAG** offers comprehensive security and performance features like **Cloudflare WAF & DDoS Protection**.
   - The developer responded with a comprehensive security and performance list including **Cloudflare WAF & DDoS Protection**, **Traefik Reverse Proxy**, and **JWT Authentication**.
- **Community Skepticism on TLRAG claims**: Several users expressed skepticism, with one member suggesting it's superficially just **RAG** with keyword search and others pointing out issues with the whitepaper's structure, comparisons, and metrics.
   - The developer defended their approach, stating that they don't care about marketing and prefer to be judged by the product itself, also the same developer dismissed critiques as *AI-generated shit wich isnt true at all*.
- **OpenRouter user demographic**: A user pointed out that **Open Router** users are not the target demographic for this type of product, as they are mostly people who don't know what **Open Router** is.
   - The developer claimed that people instead find reasons why it isnt nessesary to test it, but also apologized for getting defensive and acknowledged the constructive criticism.
- **Vimprove announced: RAG chatbot for Neovim**: A user announced the creation of **Vimprove**, a **RAG chatbot API/CLI/Nvim plugin for neovim documentation**, which uses **sentence-transformers** and **chromadb** locally, and the **OpenRouter API** for responses, and is available on [GitHub](https://github.com/rlarson20/Vimprove).
   - A user jokingly suggested banning the creator.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1428828487044562994)** (428 messages🔥🔥🔥): 

> `SambaNova latency in DeepSeek v3.1, AI SDK Anthropics models in OpenRouter, Default LLM for market research - gemini 2.5 pro?, Is llama doing anything good?, Support Agent SKILLS from Claude?` 


- **DeepSeek v3.1's SambaNova Latency**: A user is experiencing latency issues with **SambaNova** in **DeepSeek v3.1 Terminus**, noting that despite SambaNova accounting for higher throughput, **DeepInfra** seems faster in both throughput and latency according to [OpenRouter's provider comparison page](https://openrouter.ai/deepseek/deepseek-v3.1-terminus/providers?sort=throughput).
- **GPT-5 Image is GPT-image-1: Not a New Model?**: Members discussed the quality of **GPT-5 Image** and speculated that it is actually **GPT-image-1** wrapped with a responses API, and therefore not an updated model.
   - One member stated *I think so I've tested both and I like nano banana a lot more*.
- **OpenInference Censorship Implementation**: Members discussed the censoring of **OpenInference** models, with one noting that it's the only open source model provider that implements some form of its own moderation because it is collecting all data to use for "research" which they may publish.
   - One member stated that they had been using it uncensored for a week and asked *can it go back to being uncensored*.
- **Deepseek V3 0324 meets Grim Reaper**: Members reported issues with **Deepseek v3 0324**, with one stating *i swear i just saw V3 0324 die in real time*.
   - This followed a discussion of issues with the **Deepseek free models** more generally.
- **Stripe Supports Debit: A Cause for Jubilation**: A user inquired about payment methods, and another noted that **Stripe** supports **debit cards**.
   - The original user clarified that their debit card was a specific type (ING) and wondered if that mattered, but another user replied that stripe accepts those too.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1428827775099338787)** (93 messages🔥🔥): 

> `Fake AI Product Success Rates, AI Art in Corporate Branding, Qwen3 235A22B API Pricing, Liquid stopped hosting LFM 7b, AWS Status Page` 


- ****Mock AI Products Achieve Success?****: A user jokingly suggested that the key to success for fake AI products is to get enough people to see it on [Twitter](https://x.com/vithursant19/status/1979176346329792738) (profit may vary).
   - A discussion ensued about the use of **AI-generated art** by companies, with some finding it *unprofessional and cheap* due to the laziness in changing the default styles.
- ****Qwen3 Pricing Amazes Users****: A user noted that the **Qwen3 235A22B API pricing** is great, especially with W&B, leading to excitement about processing a lot of data through it.
   - Another user mentioned the need for **OpenRouter** to have routine price drop announcements, emphasizing that nothing else comes close to that intelligence per dollar.
- ****The Demise of LFM 7B****: The community mourned the loss of **Liquid's** hosting of **LFM 7B**, the original $0.01/Mtok LLM, with one user suggesting *hosting a funeral* for it and stating that it was deleted at **7:54 AM (EST)**.
   - A user pointed out that when sorting by price in **OpenRouter**, there's no 1 to 1 alternative with the closest match being **Gemma 3 9B** but that is triple the output cost.
- ****AWS Having a Massive Sh*t****: A user posted a link to the [AWS status page](https://health.aws.amazon.com/health/status) as well as a third party [status page](https://usa-status.com/AWS) implying **AWS** was having a major outage.
   - Another user sarcastically noted that *the government gets more than 4 trillion dollars, and can't achieve three 9's of uptime*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1428864881385279630)** (295 messages🔥🔥): 

> `Llama Bench support, MCP servers, OpenHands agentic framework, System Prompts, Jinja Template issue` 


- **API Access and MCP Servers on iPad**: A user is seeking to access **LM Studio** via the API from an iPad using [3Sparks Chat](https://example.com), and wants to know if it's possible to use **MCP servers** through the API, similar to the chat interface, but MCP over API isn't supported yet.
   - Another user suggests a remote desktop app or mentions that serving the chat panel as a webpage could also accomplish this, while others suggest Tailscale or RustDesk.
- **System Prompts parsing Issues**: A user discovered that **system prompts** undergo parsing, and special characters like **brackets** are problematic depending on the model and chat template in use, noting that the [Jinja template issue](https://github.com/ggml-org/llama.cpp/pull/15019/commits/e52c95c740831a7b84820721730980cb59999f23) is being fixed.
   - They suggest using the **ChatML template** for **Qwen models** to mitigate some issues, and share a [custom prompt template](https://cdn.discordapp.com/attachments/1110598183144399061/1428919144962724020/message.txt?ex=68f834a8&is=68f6e328&hm=58ee3a7881ea7f41910df34a1d8a84e0f644cea6b0dd9ac5b386d473b8aa58ff&)
- **Local Inference Copilot Extension**: A user inquired about a **VSCode extension** for **LM Studio**, mentioning the existing **Void Editor** support is lacking; another user recommends the **kilocode extension** for **LM Studio** support.
   - Members note that there are several extensions available to specify a **local inference server**, even for **GitHub Copilot**, with one user noting native support may be coming soon through an [OAI compatible model selector](https://code.visualstudio.com/docs/copilot/customization/language-models#_use-an-openaicompatible-model).
- **LM Studio and MCP Root Functionality**: A user seeks to configure **MCP root directories** in **LM Studio**, similar to the **@modelcontextprotocol/inspector** tool, but discovers that **LM Studio** only supports the bare minimum in **MCP servers** (e.g., only tools).
   - It's determined that **LM Studio** does not fully support the [mcp protocol specification](https://modelcontextprotocol.io/specification/2025-06-18/client/roots), which is why tools that rely on this functionality will not work.
- **Running TTS models with LM Studio API**: Users discuss the possibility of running **TTS models** on **LM Studio**, with one user stating this is not currently possible, and another explaining how to run TTS with LMS api.
   - One user details how to build a custom app with **Qwen3** to code it, then using the LM Studio API.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1428957374747639861)** (339 messages🔥🔥): 

> `GPU Cooling Solutions, 3090 Hotspot Issues, Auxiliary GPU Usage, EPYC vs Threadripper for LLMs, ROCm Support for Older GPUs` 


- **Extreme Cooling Mod: Desk Fan for PC?**: After returning from a psych ward, a user jokingly installed [dual 12-inch turbine fans](https://cdn.discordapp.com/attachments/1153759714082033735/1428989742602522624/IMG20251018171356.jpg?ex=68f7cda7&is=68f67c27&hm=8dcd0000dc20ee37ad9d4108ee238e75c8548dc0778594ef230cc451d50ebcdb&) to cool their PC, expecting improved airflow.
   - The user clarified it as a joke, and it was not a PC fan to be installed inside the case, but an external **desk fan**.
- **3090 Hotspot Troubles and Tweaks**: A user reported **3090 hotspot temps** reaching **90-95°C**, even after reseating the CPU and finding the core temp acceptable, and considers undervolting or repasting.
   - Suggestions included checking the power limit, repasting the GPU (using a spatula for even application), and considering **PTM** (phase change material) for better thermal performance, and a lower power limit could help determine if the temp sensor is faulty.
- **Maximizing Auxiliary GPU Performance**: A user explored using a **3050 as an auxiliary GPU** alongside a **3090** for extra VRAM in AI tasks.
   - Discussion covered setting the **3090 as the primary compute device** in hardware settings and concerns that the two cards were *"suffocating the GPU"*.
- **Debating the Perfect LLM Build**: Members debated the merits of using an **EPYC 9654** versus a **Threadripper** for running large language models (LLMs).
   - The consensus leaned towards **EPYC** for its superior memory bandwidth and dual-CPU capabilities, with a suggestion to consider multiple used **3090s** or **MI50s** as cost-effective alternatives to a single high-end **5090** for increased VRAM.
- **ROCm Compatibility Woes and Solutions**: Users discussed issues with **ROCm (Radeon Open Compute)** compatibility for older GPUs, specifically the **gfx906 (MI50)**, in **llama.cpp**.
   - It was pointed out that the latest **llama.cpp** versions may not support older ROCm versions, but a [YouTube tutorial](https://www.youtube.com/watch?v=xcI0pyE8VN8) claims to restore full functionality in **ROCm 7** for **llama.cpp**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1428864106386952392)** (535 messages🔥🔥🔥): 

> `Fine-tuning LLMs, HF Inference API & text generation, MCPs vs Agents, Image analysis and labeling, AWS outage` 


- **Level up LLM Fine-Tuning**: A member recommends a [smol course](https://huggingface.co/learn/smol-course/unit0/1) and [Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide) for learning about **fine-tuning LLMs**.
   - The discussion emphasizes that the *simplicity* is often prioritized over *speed* in sample code, and optimization techniques may become obsolete, suggesting writing them separately from the start (see [transformers/llm_optim](https://huggingface.co/docs/transformers/main/en/llm_optim)).
- **HF Inference API Requires Task Naming Conventions**: Members discovered that **Mistral** models require renaming the variable *'text-generation'* to *'conversational'* to function correctly with the **HF Inference API**, referencing a [relevant issue](https://github.com/langchain-ai/langchain/issues/31434#issuecomment-2936308959).
   - They noted that they learned about **Florence** model's task name requirements and that other models may have similar constraints, requiring specific task names to respond to prompts.
- **AI Agents vs MCPs**: A discussion clarified that MCPs (Model Context Protocols) or Agents are used to manage AI interactions, particularly in contexts like email, to prevent incorrect actions, with one member suggesting the term *AI APP* for simplicity and provided a link to [mcp.so](https://mcp.so/) as an example of the concept.
   - Further, Model Context Protocol are defined as *defined tools for small tasks that your LLM calls to finish a particular complex task that requires external tools* and that *They are reusable*.
- **Efficient Data Labeling Strategies**: Members discussed efficient data labeling strategies, with one suggesting the use of **GPT-4o** for high-accuracy labeling and comparing the results with **Claude** and **Gemini** APIs to automate the correction process.
   - Alternative **vision models** were mentioned and some members concluded that for images with unique identifiers, manual labeling might be the only option.
- **AWS Outage Affects Hugging Face**: Users reported issues downloading models from Hugging Face due to a global **AWS outage**, which was causing internal server errors.
   - The Hugging Face status page confirmed the issue, noting that the service was experiencing trouble due to the AWS outage, but the issue has since been resolved.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1429023952273149992)** (6 messages): 

> `NivasaAI Dynamic Routing, Google ADK, Agent + Tools, Max limitations` 


- ****NivasaAI's Dynamic Routing Debut****: A member is switching **NivasaAI** from **Agent + Tools** to multiple agents with **dynamic routing** for enhanced UX.
   - The goal is instant chat responses aggregated from tools and agents, with the initial commit available on [GitHub](https://github.com/HakAl/adk_rag/commit/c5d70489cb8c19ab1b13cd84948835fa6a1c7d03).
- ****Google ADK Powers NivasaAI's Router****: A member plans to finalize a router using **Google ADK** that classifies requests into categories such as `code_validation`, `rag_query`, and `general_chat`.
   - The aim is to handle a range of tasks from syntax checking to casual conversation.
- ****Max Plan - Maxed Out?****: A member questioned whether the limitations of the **Max** plan are enough to accomplish their goals.
   - The post included a picture of an invoice, indicating the limitations of the plan.
- ****Reverse Engineering Required****: A member posted an image and stated it's like *reverse engineering to understand how it actually works*.
   - They expressed it's funny because something has been engineered without absolute control over the engineering, so it is funny that we have to reverse engineer it


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1428847282480222339)** (4 messages): 

> `Qwen3 Vision Model, Microsoft Protein Functionality Prediction` 


- **Qwen3 Vision Model releases**: A member shared a link to the new **Qwen3-VL-8B-Instruct-GGUF** [vision model](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF) and the original [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- **Microsoft predicts protein's functionality**: A member posted about impressive new **protein functionality prediction** tools from Microsoft, including a link to the [bioemu model](https://huggingface.co/microsoft/bioemu).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1428823501996949545)** (14 messages🔥): 

> `Agent Building Tutorial, LLM Token Hallucination, New Architecture for Decentralized Tokenizers, Self-Hosted Tracing for OpenAI Agents, Amiko: Social Identity Platform for AI Twins` 


- **Agent Building Tutorial Debuts**: A member shared a new [tutorial on building agents](https://samdobson.uk/posts/how-to-build-an-agent/), inviting feedback from the community.
- **Local LLMs May Hallucinate!**: A member tested **47 configs** and found that local LLMs may hallucinate after **6K tokens**.
   - They shared a [Medium article](https://medium.com/@poojakanalaa/i-trained-47-different-quantization-configurations-so-you-dont-have-to-c637b274452d) detailing their findings on **quantization configurations** that actually work on real hardware.
- **New Transformer Architecture with Decentralized Tokenizer Emerges**: A member introduced a new architecture with a decentralized tokenizer, noting it's not compatible with **llama.gguf** but available on [GitHub](https://github.com/pacific-prime777/architecture_INL_transformer).
- **Self-Hosted Tracing for OpenAI Agents Framework Released**: A member open-sourced a [self-hosted tracing and analytics infrastructure](https://github.com/yusuf-eren/openai-agents-tracing) for **OpenAI Agents framework** traces, addressing GDPR concerns and the inability to export data from OpenAI's tracing dashboard.
- **Amiko: Social Identity Platform for Building AI Twins**: A member introduced [Amiko](http://www.heyamiko.com), a social identity platform for building behavior-first **AI twins**, **companions**, and **social agents**, highlighting its focus on privacy, ownership, and personality.
   - They explained that *Amikos are private, portable, and user-owned. They don’t just reflect you. They act with you.*


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1429412315350040659)** (2 messages): 

> `` 


- **TE Choice Unveiled**: A member inquired about the specific **TE (Transformer Engine)** used in a recent project.
- **Details on Transformer Engine Implementation**: The member is seeking clarity on the final **Transformer Engine** implementation details.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1428891058409963623)** (1 messages): 

> `Chat Template Conversion, Tokenizer Execution, Fine-Tuning Script` 


- **Chat Template Transmogrification Troubles**: Members discussed the initial step of converting a dataset into the **model's specific chat template**.
   - This process ensures compatibility and optimal performance during fine-tuning, particularly with models sensitive to input formatting.
- **Tokenizer Triumphs & Tribulations**: The next step involves running the **model's tokenizer** on the converted dataset.
   - This process converts the text into numerical tokens that the model can understand.
- **Fine-Tuning Frontier Fracas**: The final step is executing the **fine-tuning script** on the tokenized dataset.
   - This adapts the pre-trained model to the specific nuances of your data, potentially improving its performance on targeted tasks.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1428972772171710646)** (16 messages🔥): 

> `Leaderboard Submission Delay, CUDA Out of Memory Errors in SMOL Course, Lighteval Bug Fix, DPO Exploration Lacking` 


- **Leaderboard Submission Delayed Despite Merged PR**: A member reported that their PR was merged, but their submission wasn't showing on the leaderboard, possibly due to the dataset initially being private; the leaderboard is updated weekly or bi-weekly.
   - After making the dataset public, the member is waiting for the next leaderboard revision to see if their submission appears.
- **CUDA Out of Memory Plague SMOL Course Students**: A member encountered **CUDA out of memory errors** while following the SMOL course (section 1.5) using a HF Pro account, HF jobs, and Colab, despite using an a100-large instance.
   - One member experiencing the same issue was advised to try reducing `per_device_train_batch_size` to **2** and increasing `--gradient_accumulation_steps` to **2** or more.
- **Lighteval Bug Fix Saves the Day**: A member encountered errors during the evaluation job in chapter 1 due to a bug in the latest `lighteval` version and a missing `emoji` package, solved by using the flag: `--with "lighteval[vllm]@git+https://github.com/huggingface/lighteval,emoji"`.
   - Another member also reported issues with `lighteval` producing a `FileNotFound` error and shared a link to a [GitHub issue](https://github.com/huggingface/lighteval/issues/988) with a solution.
- **DPO Process Leaves Member Unfulfilled**: After completing DPO, one member found it less exploratory compared to SFT and expressed feeling restricted to using alignment data without a clear understanding of its purpose in evaluation.
   - They noted that *they really didnt see any invitation to explore anything beyond what was provided* and it felt like alignment data should just be used without a sense of what they are using it for in evaluation.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1428864940206194708)** (13 messages🔥): 

> `Course Starting, SmolAgents Framework, DeepFabric for SLMs` 


- **Course Launch Mania**: Several members announced they are starting the course today.
   - This sparked excitement among newcomers ready to dive into the agent-based learning experience.
- **SmolAgents Framework Troubling Trainees**: A member expressed difficulty grasping the **SmolAgents framework** in Unit 2.
   - They indicated a need for deeper understanding and clarification of its concepts.
- **DeepFabric Dazzles Data-Driven Devotees**: A member introduced **DeepFabric**, a tool for training **SLMs** to improve structured output and tool calling, shared via a [GitHub link](https://github.com/lukehinds/deepfabric).
   - The tool enables the generation of reasoning trace-based datasets that can be directly loaded into TRL (SFT).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1428822894301020290)** (19 messages🔥): 

> `Triton distributed talk, Helion talk, Category Theory and AI resources, Impossible Cloud Network (ICN) collab, AMD event` 


- **Triton Talk Gets Nixed!**: A [Triton distributed talk](https://www.youtube.com/watch?v=EcwXLcvU06g) was scheduled but then **canceled**.
   - A **Helion talk** started about an hour later, though its recording status is unconfirmed.
- **Category Theory makes AI Aha Moment!**: Bruno Gavranovic maintains an up-to-date list of **category theory / AI resources** [here](https://github.com/bgavran/Category_Theory_Machine_Learning).
   - He also ran a course with some folks at DeepMind ([Categorical Deep Learning](https://categoricaldeeplearning.com/)), and a member maintains a categorical deep learning compiler (**catgrad**).
- **Impossible Cloud Wants Collaboration!**: Ali from **Impossible Cloud Network (ICN)** wants to explore a collab or workshop with GPU MODE and shared their [whitepaper](https://www.icn.global/docs).
   - ICN is a **web3** project.
- **Mark gives talk at AMD Event**: A member reported watching Mark give a talk at the **AMD event**, though a direct link wasn't provided.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1429062257609805916)** (15 messages🔥): 

> `TMA performance, Triton host TensorDescriptor, algebraic shuffle` 


- **TMA: Triton mystery; Ampere's perf history**: A member is implementing **TMA** support for **dot** in **Triton** for **NVIDIA SM120** GPUs, but sees no performance benefits using **TMA** over `cp.async` and asks about when and where **TMA** benefits performance gain over `cp.async`.
- **SM120 struggles to be TMA friendly**: Members discussed whether **SM120** has the **TMA** hardware, with at least tensor map + cp.async.bulk.tensor present.
   - One member noted that at least on **Hopper**, **TMA** isn't all that efficient for loads < **4KiB**, and loading larger tiles is still better, up to a point.
- **Ampere's TensorDescriptor runs like garbage!**: A member noticed on **Ampere** that when they do `desc.load([x, y])` the performance is kind of bad, noting that **Ampere** doesn't have the **TMA**.
   - They were expecting it to basically have the same perf as using pointers or a block pointer.
- **Auxiliary Registers to the Rescue?**: A member suggested using auxiliary registers to hold new values when in-place operations aren't possible, and then moving them to their final position at the end.
- **Algebraic Shuffles reign supreme**: A member mentioned that they never implemented a specific algorithm because very good contributions implementing shuffles via a more algebraic approach arose, and they just stayed with that.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1428832087217147904)** (21 messages🔥): 

> `Thread Block vs CTA, Distributed Shared Memory Latency/Bandwidth, TMA vs cp.async, CUDA Learning Resources, Device-Agnostic TMA Logic` 


- ****CTA Clarified**: Cooperative Thread Array**: In CUDA, a **Cooperative Thread Array (CTA)** is synonymous with a **thread block**, representing a 1D, 2D, or 3D array of threads.
   - It is called a Cooperative Thread Array (CTA) when using the distributed shared memory feature.
- ****Latency Leaps** in Distributed Shared Memory?**: When using distributed shared memory, accessing shared memory in the current block versus another block likely involves **latency and bandwidth** differences.
   - While concrete numbers are scarce, one member pointed to [this blog post](https://www.aleksagordic.com/blog/matmul), indicating it's slower than shared memory but faster than L2 cache.
- ****TMA Triumphs**: Tile Matrix Accelerato**: **TMA** uses a single instruction to launch a tile copy and can **multicast**, saving time on index calculations and simplifying swizzling.
   - Benchmarks showed TMA slightly faster (~0.5%) than cp.async for **large tiles**, but cp.async was quicker for smaller tiles, aligning with recommendations from [NVIDIA's memory optimization talk](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/).
- ****CUDA Commences**: Newbie Navigates Kernels**: A new CUDA user is studying **GPU architecture** aiming to specialize in GPU/inference engineering and is seeking resources.
   - Another user offered to sync up and compare notes while studying.
- ****Blackwell's Blueprint**: Layout Logistics**: Discussion arose whether TMA logic is **device-agnostic**, potentially requiring different layouts to leverage fully on architectures like **B200 vs. 5090**.
   - For **Blackwell**, one member referenced a [5D layout example](https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py#L272) for scales, noting that as long as the first dimension is contiguous, performance should be consistent.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1428982859519098941)** (15 messages🔥): 

> `Pytorch profiler CUDA issues, AlphaZero Compute Requirements, Matmul Fusion, Process scheduling issues` 


- **Pytorch Profiler CUPTI Issue on Windows & WSL**: A user is encountering a `CUPTI_ERROR_INVALID_DEVICE (2)` error with **torch.profiler** on both Windows and WSL despite **CUDA** being available and the **CUPTI** path being set.
   - A member suggested enabling hardware counter access via [this gist](https://gist.github.com/msaroufim/9e56ce5d42a5e9ccd5e938c83181ea47), though the user reported that this solution did not work for them.
- **AlphaZero struggles on single 3090ti**: A user training an AlphaZero implementation on a **3090ti** is experiencing low GPU utilization and slow training times with **100 simulations** and asked if it was a configuration issue or just needed more time.
   - Another member responded that training chess with **AlphaZero** requires substantially more compute than a single **3090ti** can provide, citing DeepMind's resource usage.
- **matmul & epilogue fusion?**: A member asked about easy ways to fuse the **matmul** and **epilogue** operations.
   - They see separate matmul and triton kernel even when using **torch.compile** to fuse the **F.silu(x@w)**.
- **Process Descheduling Causes Slow Iterations on H200x8**: A user running a **torchtitan llama 3B pretrain** on a single **H200x8** Bare Metal Instance is experiencing random slow iterations.
   - An nsys trace revealed that the active thread/process is being descheduled from a CPU for a few seconds, even with no other processes running, leading to a question about potential OS/kernel misconfiguration.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1429057259719102587)** (2 messages): 

> `GPU Engineering Resources, PMPP-Eval Journey` 


- **Curated GPU Engineering Resources Debut!**: A member shared their curated [repository of GPU engineering resources](https://github.com/goabiaryan/awesome-gpu-engineering), consolidating various helpful materials in one location.
   - This could be a boon for engineers looking for *centralized knowledge*.
- **PMPP-Eval+Journey Blogpost Unveiled**: A member posted a [blog post](https://blog.sinatras.dev/PMPP-Eval+Journey) about their *PMPP-Eval journey*.
   - This could be a great resource for people interested in **performance modeling and prediction**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1428841133651394681)** (1 messages): 

> `Seed Stage SF Startup, GPU Performance Engineers, Herdora Hiring` 


- ****Herdora** startup grabs seed funding**: A seed-stage SF startup named **Herdora** announced they're funded by **YC**, **Jeff Dean**, **Woj Zaremba** (OpenAI co-founder), and the head of kernels at Together.ai.
   - The company is based in Pac Heights, where the bottom floor is the office.
- ****Herdora** seeks PyTorch/CUDA Kernel Engineers**: **Herdora** is hiring engineers who like to write **PyTorch**, **CUDA kernels**, and push **GPU** performance.
   - They are looking for both full-time and winter/spring/summer interns, and you can apply and DM your resume, or DM with any questions.
- ****Herdora** offers sweet compensation package**: **Herdora** is offering a compensation package of **$170-200k** + **2-4%** equity.
   - Apply for full-time and winter/spring/summer internships at [Herdora's job page](https://jobs.ashbyhq.com/herdora).


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

mannythecreator: Could you please share some url that talks about these.
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1428842688383553726)** (8 messages🔥): 

> `vLLM quantization, SGLang quantization support, Online quantization, ModuleFqnToConfig, torchao_utils.py` 


- ****vLLM** still beats **SGLang** at Quantization**: **vLLM** integration supports any type of quantization config, but **SGLang** only supports **int4wo**, **int8dq**, **int8wo**.
   - The supported quantization methods are described in the [SGLang documentation](https://docs.sglang.ai/advanced_features/quantization.html#online-quantization).
- ****ModuleFqnToConfig** replaces **filter_fn****: The `filter_fn` for `quantize_op` will be deprecated in favor of using **ModuleFqnToConfig**, which now supports regex.
   - More details are available in the [TorchAO pull request #3083](https://github.com/pytorch/ao/pull/3083).
- ****SGLang** lacks Vision Model Quantization**: Currently **SGLang** online quantization does not support skipping quantization for vision models.
   - A user inquired about skipping quantization for the vision model, but **SGLang** does not support this feature.
- ****torchao_utils.py** needs refactoring**: The code in `torchao_utils.py` is currently in use, but the team plans to refactor it to use **ModuleFqnToConfig** for better quantization configuration.
   - The relevant code can be found on [GitHub](https://github.com/sgl-project/sglang/blob/184a4df697ed75805ac10146dd93e75f1fc609a7/python/sglang/srt/layers/torchao_utils.py#L42).


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1428903762042294342)** (13 messages🔥): 

> `geohot, GPUs go brrr, DGX Spark impressions, Blackwell instructions` 


- **Geohot gets memed**: A member posted a meme featuring **Geohot**.
   - The meme was just the text *lol geohot* on top of a stock photo of a man in a turtleneck.
- **GPUs go BRRR**: A member shared a link to [horace.io/brrr_intro.html](https://horace.io/brrr_intro.html) and a picture of **GPUs**.
   - Another member joked about turning it into a black and white cartoon.
- **DGX Spark is smaller than expected**: A member noted that the **DGX Spark** is smaller than they thought.
   - Another member commented that it is a prototyping box, not an inference box, and that your **price per token** is going to be cheaper with a cloud provider.
- **Inquiring about Blackwell instructions**: A member asked about the compute type of the **DGX Spark** device, and whether it exposes a full subset of **Blackwell instructions** to the user.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1429740921913610321)** (2 messages): 

> `San Diego Meetup, Orange County Meetup` 


- **San Diegans Seek SoCal IRL**: A member in **San Diego** is looking to see if there's an **irl meetup** planned.
   - They tagged another member for awareness and to express interest in attending.
- **Orange County Member Reporting**: A member reported their location as **Orange County**.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1429039203173404802)** (2 messages): 

> `Triton Kernels, Fused Kernels, Kernel Assistance` 


- **Triton's Fused Kernels' Power**: Triton's strength lies in its fused kernels, meaning it can execute operations more efficiently.
   - Other repositories could benefit from assistance with developing and optimizing their kernels.
- **Kernel Assistance Needed**: Many repositories besides Triton require help with kernel development and optimization.
   - This presents opportunities for contributions and improvements in various kernel-related projects.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1429628713028554833)** (10 messages🔥): 

> `AMD Warp Sizes vs NVIDIA, MI300x Cache Coherency, Warp Tiling, GEMM Occupancy` 


- **AMD vs NVidia: Warp Size Wonders**: Discussion arose regarding the impact of **AMD's 64-thread warp size** compared to **NVIDIA's 32-thread warp size** on occupancy.
   - A member noted that comparing kernel occupancy directly between **AMD** and **NVIDIA** isn't straightforward due to differing hardware architectures, although the max block size is **1024** for both.
- **MI300x NVLink Cache Coherency**: Concerns were raised about **cache coherency on NVLink** for the **MI300x**, with a user posting an [image](https://cdn.discordapp.com/attachments/1233704710389764236/1429727155448840302/image.png?ex=68f7d96c&is=68f687ec&hm=10c3159bfc1b79e7731f8fbb75dc06948930dfe0a37a031ed59cbd2ddb215e2a) for reference.
   - Another member pointed to the [AMD documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/how-to/hip_runtime_api/memory_management/coherence_control.html) and suggested using `hipExtMallocWithFlags(&ptr, size, hipDeviceMallocFinegrained)` to achieve cache coherency on **MI300**.
- **Warp Tiling Tactics**: The discussion touched on how a warp size of **64** affects **warp tiling**, suggesting a need for more threads and larger tile sizes.
   - One member clarified that the key factor influencing warp tiling is the size of the **warp gemm instruction (MFMA on wave64 GPUs)**, and larger warp size means supplying fewer registers per thread.
- **GEMM Shape Affects Occupancy**: One member asked if the occupancy doesn't affect much when doing **GEMMs** of large shape like **8192 * 8192 * 8192**.
   - Another member replied that it boils down to the amount of data processed per workgroup, though tools like **CK-tile** assume only one block per **CU**, so persistent GEMMs launch only **304** blocks on **MI300**.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1429685787376746617)** (1 messages): 

> `TileLang, Deepseek V32, Sparse MLA` 


- **Deepseek Model's Sparse MLA Gets TileLang Treatment**: A member suggested checking out a [TileLang implementation](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/sparse_mla_fwd.py) of the **Sparse MLA** (Multi-Layer Attention) component from the **Deepseek V32** model.
- **TileLang Project Mentioned**: A member pointed to the **TileLang** project, suggesting exploration of its capabilities and example implementations.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1429364448849563712)** (5 messages): 

> `Nvidia DGX, Petaflop Compute, MMA Atoms in CuTe, CUTLASS docs, Blogpost on MMA Atoms` 


- **Nvidia DGX Surfaces**: A member asked *what am I looking at?* and another member responded **Nvidia DGX**.
- **Petaflop Pricing Puzzle**: A member noted **1 petaflop of fp4 compute for 4k$**, referring to the dev kit.
- **MMA Atoms Detailed in CuTe**: A member explained that **MMA Atoms** are one of the fundamental building blocks in **CuTe**, and shared a [blog post](https://veitner.bearblog.dev/mma-atoms-in-cute/) discussing examples from the CuTe docs, offering an additional explanation for newcomers.
- **CUTLASS and PTX Docs Linked**: The same member shared a link to the [CUTLASS docs](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0t_mma_atom.html) and mentioned making a connection of the **CuTe abstraction** to the corresponding section of the **PTX docs**.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1430020472342319205)** (1 messages): 

> `LLM for Kernel Generation, LLM for Bottleneck Identification` 


- **Kernel-Generating LLM: Future of Optimization?**: The discussion revolves around the utility of having an **LLM** capable of generating **kernels** versus one that can identify **bottlenecks at runtime**.
   - The former could potentially automate the creation of optimized code, while the latter could dynamically adjust system resources to maximize performance.
- **Bottleneck-Sniffing LLM: Real-Time Performance Boost?**: An **LLM** that identifies **bottlenecks at runtime** could enable real-time optimization and resource allocation, potentially leading to more immediate performance gains.
   - This could involve analyzing system metrics and logs to pinpoint performance bottlenecks and suggest corrective actions.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1428828912657502248)** (1 messages): 

> `Thread Execution Clarification, Collective Launch Behavior, TMA Operations in New TK, Prefix Meaning in New TK` 


- **Thread Execution gets Explicit Designation**: Each operation now clearly defines who executes it, e.g., no prefix (like `tma::load_async`) means it's run by the calling thread(s), and a prefix (like `warp::tma::load_async`) denotes a collective launch.
   - Behavior depends on the operation, e.g., `warp::tma::load_async` is run by lane 0 of the calling warp, and some operations *only* allow collective launch like `warp::exp` or `warpgroup::wgmma`.
- **Prefix Solves TK's Implicitness Conundrum**: In the previous version of TK, no prefix implicitly meant *either* run by an entire warp or a single thread, depending on the operation, creating ambiguity.
   - For instance, `add` no longer exists and is replaced by `warp::add`, and users must ensure that `tma::load_async` or any semaphore operation is run by a single thread, or use `warp::tma::load_async`.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1429132931716349952)** (3 messages): 

> `VectorAdd Leaderboard, H100 Results, B200 Results, A100 Results` 


- **VectorAdd Leaderboard Gets New Submission**: A member achieved **second place** on the `vectoradd_v2` leaderboard with id `66209` on **H100 (526 µs)**, **B200 (236 µs)** and **A100 (909 µs)**.
   - Another member submitted three runs on the `vectoradd_v2` leaderboard with submission id `66228` and `66233` to **10th place (1582 µs)** and **8th place (1243 µs)** respectively on **A100**.
- **Another Member Submits to A100 VectorAdd Leaderboard**: A member achieved **10th place** on the `vectoradd_v2` leaderboard with submission ID `66228` on **A100** with a time of **1582 µs**.
   - This was followed by another submission, ID `66233`, which achieved **8th place** with a time of **1243 µs**.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1429198230096318574)** (4 messages): 

> `CP5, coalescing memory accesses, tiling, Transpose` 


- **Tiling Improves CP5**: After a baseline solution in **CP5**, tiling got a member to around **4.9s**.
   - The next optimization discussed was coalescing memory accesses.
- **Coalescing Confusion Clarified**: A member seeks help coalescing memory accesses, clarifying that coalescing happens when all **32 threads** in a warp access a contiguous **128 byte** block.
   - The member presented the code and asked about how to optimize it.
- **Transposing Transpires**: A member explores transposing the matrix to improve memory access patterns: `Bs[threadIdx.x][threadIdx.y] = diffs[col_B * ny + row_B]`.
   - They mentioned that it *didn't quite work*.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1429152615803129887)** (20 messages🔥): 

> `H100 server prices, RTX 6000 ADA TFLOPs variance, Benchmarking nuances and thermal throttling, NVLink bridge prices, CuBLAS autotuning` 


- **Enterprise Clients Eyeing Hefty H100 Server Prices**: A member inquired about current **H100 server prices** for enterprise clients.
   - While no concrete numbers were shared, the inquiry highlights ongoing interest in acquiring top-tier GPU hardware.
- **RTX 6000 ADA's TFLOPs Performance Faces Scrutiny**: A member questioned the low and varying **TFLOPs** performance of the **RTX 6000 ADA**, linking to relevant [Xitter post](https://x.com/thezachmueller/status/1979649658965369049?s=46) for context.
   - The discussion includes suggestions to monitor clock rate, temperature, and power capping during benchmarks to identify potential throttling issues.
- **Benchmarking Tactics Discussed**: Members shared insights on refining benchmarking methodologies, including the suggestion to add a `time.sleep(0.1)` before each shape to stabilize thermal conditions.
   - Links to a relevant [ml-engineering repo](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks) and [dataset](https://huggingface.co/datasets/muellerzr/consumer-mamf/blob/main/nvidia_rtx_6000_ada_generation_bf16.txt) were provided.
- **CuBLAS Autotuning Tweaks Suggested for Max Performance**: A member proposed using `torch.compile(torch.mm, mode="max-autotune-no-cudagraphs")` to enable **CuBLAS autotuning** on different shapes when benchmarking.
   - The aim is to optimize matrix multiplication performance, particularly on non-DC GPUs where CuBLAS tuning might be suboptimal.
- **NVLink Bridge Prices Spark Collector's Item Debate**: A member questioned the high prices of **NVLink bridges**, speculating whether it's due to collector's value or genuine utility and demand.
   - The query highlights the ongoing interest and perceived value of multi-GPU configurations using NVLink technology.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1429859840498335774)** (1 messages): 

> `` 


- **No Documentation Updates**: A member indicated they did not work on the documentation website over the weekend and have no updates.
   - Consequently, they will skip today's meeting.
- **Meeting Absence**: Due to the lack of progress on the documentation website, a member will be absent from today's meeting.
   - They cited the absence of updates as the reason for skipping the meeting.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1429400208802054144)** (2 messages): 

> `Competition Submissions, Winner Write-ups` 


- **Community Craves Competition Submission Write-Ups**: Members are eager to see the solutions from a recent competition and are looking forward to write-ups from the winners.
   - This would be a great learning experience for everyone else who participated, as well.
- **s1r_o Looking at u my guy**: A member shouts out another expecting him to share his competition write-up.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1428886699676532817)** (22 messages🔥): 

> `CUTLASS tile size tuning, CuTe in non-CMake projects, MoE Grouped GEMM throughput, CUTLASS naming conventions, PTX code generation` 


- **CUTLASS Tile Size Tuning Tactics Teased**: When tuning tile sizes for CUTLASS C++ APIs, a member suggests using a script to generate multiple C++ files with different parameter permutations and testing them, since **compile-time parameters** cannot be JIT compiled.
   - Another member chimed in that although this *might pollute the instruction cache at runtime*, in practice, it was still faster than other methods tried.
- **CuTe can Compile Simply!**: Members confirmed that it is possible to use **CuTe cpp** in a simple non-CMake project by directly compiling the file with `nvcc` and the [usual C++ flags](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
   - Another member suggested compiling the example **14_ampere_tf32_tensorop_gemm** with specific flags like `-arch=sm_86` and `-std=c++17`.
- **MoE Throughput Talked Thoroughly**: A member inquired about calculating the **useful throughput** for prefill and decode in **MoE Grouped GEMM** by comparing theoretical FLOPs to observed latency.
   - Another member prefers end-to-end processing time measurements, suggesting to measure a *dummy_time* (everything except kernel calls) and a *ref_time* (with a reference implementation) and optimizing for *delta_time = ref_time - dummy_time*.
- **CUTLASS Convention Clarified**: A member asked about the naming convention `tXrA` in CUTLASS, specifically what the `X` stands for, referencing [this line of code](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu#L179).
   - Another member explained that `tX` is a placeholder for **copy atom layout**, pointing to the [naming conventions in the documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html), and provided a link to a tutorial which explains [naming conventions more thoroughly](https://github.com/66RING/tiny-flash-attention/blob/main/cutlass_cute_tutorial_en.md).
- **PTX Powerplay Promises Potential**: A member shared a tool that annotates a CUDA CuTe kernel, compiles it to **PTX**, and provides the PTX assigned registers for annotated variables.
   - This approach reportedly achieves **26x faster** semirings and includes an example for generating random PTX kernels and bulk compiling them on all cores, as seen in [this GitHub repo](https://github.com/MetaMachines/mm-ptx/blob/master/examples/stack_ptx_inject/README.md#02_bulk_rand_gemm), along with a Python example exposing CuTe kernels as arbitrary semiring tensor routines for PyTorch in [this other github repo](https://github.com/MetaMachines/mm-kermac-py).


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1428839012784935014)** (7 messages): 

> `SITP, picograd, Karpathy's Eureka Course, MLSys 2026 tutorial, Tinygrad` 


- ****SITP and picograd** Aim for Karpathy's Eureka Course**: The goal is to make **SITP** and **picograd** the second course on **Karpathy's** "Starfleet Academy" Eureka, following **LLM101**.
   - The creator aims to get **nanogpt** and **nanochat** running and is seeking a *creative co-director* to help translate **torch eager mode**, **tinygrad**, **TVM**, and **Halide** into a codebase and course.
- **Bridging the Gap from Micrograd to Tinygrad**: Inspired by **Karpathy's** influence on **George Hotz's** streams, **SITP** and **picograd** aim to bridge the gap from **micrograd** to **tinygrad**, filling a need for clearer documentation.
   - The initiative seeks to offer a public resource led by someone new to **MLSys**, focusing on practical updates and spearheaded **PRs**.
- **Integrating Tinygrad with Triton**: The plan involves integrating **tinygrad** with **Triton** without converting tensors to **torch.Tensors**, as exemplified by earlier work from ghot and ptill, readers should implement their own tensors [https://github.com/tinygrad/tinygrad/pull/470](https://github.com/tinygrad/tinygrad/pull/470).
   - Future tasks include adding an execution engine around **tinygrad's** tensor frontend, specifying the runtime, integrating the memory allocator with **Triton**, and implementing forward and backward passes for **GEMM** in **Triton**.
- **Capturing Whole Graph with torch.export**: **Avik's** work on sound whole graph capture with **torch.export** is referenced [https://www.youtube.com/watch?v=cZTX2W1Qqh8](https://www.youtube.com/watch?v=cZTX2W1Qqh8).
   - However, it's noted that it's *not 100% guaranteed to work because PyTorch is so flexible* due to the possibility of using third-party libraries that can cause graph breaks.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1429183756622888972)** (28 messages🔥): 

> `Expert Parallelism (EP) for MoE with AllToAll, Combining EP with DP, Parallel folding, Multi-GPU training with Triton, Iris for Multi-GPU training on AMD GPUs` 


- ****Expert Parallelism Deconstructed****: When asked about how **expert parallelism (EP)** for **MoE** is implemented using two **AllToAll** operations, one member shared diagrams of naive EP and **EP2DP** architectures and noted that the first **A2A** is needed to send tokens to the proper experts when EP is combined with **data parallelism (DP)**, with the second **A2A** rerouting them back.
   - They pointed to a paper on `parallel folding` as a generalized approach to arbitrary sharding, also noting that without combining DP with EP, there would be *a lot of useless work in the dense parts of the network*.
- ****Lecture on EP/guest will be out December****: A lecture on EP and other guest lectures will be fully released in December.
   - Links were provided to [Perplexity AI's blog on efficient MoE communication](https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication) and their [pplx-kernels GitHub repository](https://github.com/perplexityai/pplx-kernels).
- ****Triton's Multi-GPU Capabilities Discussed****: While **Triton** doesn't officially support multi-GPU training, the community mentioned that [triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) works well for single and multi-node setups.
   - It was suggested that official support for multi-GPU training should be integrated into the Triton language soon.
- ****Iris emerges as AMD's multi-GPU trainer****: [Iris](https://rocm.github.io/iris/), a project for multi-GPU training on **AMD GPUs**, was highlighted, with one member praising its clean design compared to roc-shemem, emphasizing its focus on intra-node rather than multi-node setups.
   - A developer noted that [Gluon](https://rocm.github.io/iris/reference/gluon/overview.html) is the cleanest backend.
- ****Torch 2.9 adds symmetric support****: A member pointed out that **PyTorch 2.9** includes symmetric support with an nvshmem backend (supporting multi-system) and a CUDA backend (supporting up to 1 NVLink domain).
   - More details can be found on the [PyTorch blog](https://pytorch.org/blog/pytorch-2-9/).


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1428861565750476963)** (14 messages🔥): 

> `Synthetic Data AI Agents Challenge, Nvidia DGX Spark, Disaggregated prefill/decode, Speculative decoding, kernel optimization` 


- **Team Seeks Synthetic Data AI Agents Challenge**: Someone is looking for a team to participate in the **Synthetic Data AI Agents Challenge** and has made their [GitHub](https://www.GitHub.com/tyler-hilbert) available.
   - The member plans to bring the **Nvidia DGX Spark** to the hackathon and work on projects like *disaggregated prefill/decode* and *speculative decoding*.
- **Hackathon Duration Clarified**: A participant inquired about the duration of the hackathon, and another user clarified that it would last **one day**.
   - Someone tried the **DGX Spark** last weekend, and wished the inbox instructions were better, bricking the first one.
- **Team Assembles for Kernel Optimization**: Two members are seeking a team to work on **kernel optimization, RL training, and model inference**, considering implementing *triton distributed* on **B200s** or *deterministic kernels*.
   - They referenced a blog post on [deterministic kernels](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) by Thinking Machines Labs.
- **Rackscale Characterization and Memory Improvement**: A member has projects in mind related to **rackscale scale-up characterization** needing **NVL36/72**, and improving **PyTorch Symmetric Memory**.
   - Another member expressed interest in the symmetric memory improvement project.
- **Checkpoint Engine Discussion**: A participant mentioned **Kimi’s checkpoint engine RL parameter update** and shared a link to [Checkpoint Engine](https://moonshotai.github.io/checkpoint-engine/).
   - They inquired whether they would get **GB200s** in addition to **B200s**.


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/1429584160980340877)** (3 messages): 

> `CUDA, OpenCL, Vulkan` 


- **Experience with CUDA**: A member shared their initial experience with **CUDA**, describing it as fun due to the abundance of solid learning resources.
   - They now primarily work with **OpenCL** and **Vulkan** for computing and thought it would be beneficial to have a space to discuss these **APIs** and exchange ideas.
- **Appreciation for OpenCL/Vulkan Discussion Space**: The same member expressed appreciation for the existence of a dedicated space to discuss **OpenCL** and **Vulkan**.
   - They emphasized the value of sharing ideas and collaborating on projects related to these **APIs**.


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1428860225477283912)** (2 messages): 

> `Fault Tolerant Llama Training, Node Failure Prediction` 


- **PyTorch Achieves Fault Tolerance with Synthetic Failures**: A new [PyTorch blogpost](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/) details fault-tolerant **Llama training** using **2000 synthetic failures every 15 seconds** without checkpoints on Crusoe L40S.
   - The solution offers an alternative to traditional checkpointing with a bash script and job restarts, prompting considerations of investing in more automated fault tolerance processes.
- **Node Failure Prediction Discussed for Minimal Downtime**: Members discussed the possibility of predicting a high rate of **node failures** using agentic systems or ML.
   - It was suggested that predicting failures could facilitate easier replacement with minimal downtime.


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1428970641201041408)** (2 messages): 

> `Qutlass integration` 


- **Members look forward to Qutlass integration**: A member asked about the timeline for **Qutlass integration**.
- **No timeline given for Qutlass**: No timeline was given.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1428843006039031984)** (3 messages): 

> `Helion, PyTorch Conference, Triton Developer Conference, Helion 0.2` 


- **Helion Events Kick Off**: The team has announced a series of events this week to discuss **Helion** and engage with developers, as seen on the [Youtube stream](https://www.youtube.com/watch?v=1zKvCLuvUYc).
   - These events include talks and meet-and-greets at the **Triton Developer Conference 2025** and **PyTorch Conference 2025**.
- **Helion Takes the Stage at Triton Developer Conference**: On Tues, Oct. 21, there will be a talk at 2025 **Triton Developer Conference** titled "Helion: A Higher-level DSL for Kernel Authoring", more information can be found at [TritonDeveloperConference](https://tritonconference.eventbuilder.com/TritonDeveloperConference?ref=TritonDeveloperConference).
   - The talk will explore Helion, a domain-specific language (**DSL**) designed for kernel authoring.
- **PyTorch Conference Welcomes Helion Developers**: On Wed, Oct. 22, attendees can meet the developers of **PyTorch Compiler** and **Helion** at the **PyTorch Conference 2025**, see the [conference schedule](https://pytorchconference.sched.com/event/27QN9/meet-the-developers-of-pytorch-compiler-and-helion?iframe=no).
   - The Helion team will also present a talk on Thurs, Oct. 23, at the **PyTorch Conference 2025**, further details available at [PyTorch Conference](https://pytorchconference.sched.com/event/27QDl/helion-a-high-level-dsl-for-kernel-authoring-jason-ansel-meta?iframe=no).
- **Helion 0.2 Debuts as Public Beta**: **Helion 0.2** is now released as a public beta, and can be found on [pypi.org](https://pypi.org/project/helion/0.2.0/).
   - This release marks a significant step toward broader accessibility and testing of the **Helion** framework.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1428933821167960105)** (133 messages🔥🔥): 

> `ManusAI v1.5, AI sentiment shift, Cursor Git worktree support, Grok 4.20, GPT-4o transcribe diarize` 


- **ManusAI drops v1.5, Web App Wunderkind**: [ManusAI v1.5](https://xcancel.com/manusai/status/1978854323774792135?s=46) converts prompts into **production-grade web apps** in under **4 minutes**, boasting *unlimited context* via auto-offloading and recall, launched on **October 16th**.
   - Users praised its speed and questioned context engineering, while some found previous tools like Orchids, Loveable, and V0 *disappointing* compared to coding directly.
- **AI Gloom Boom, Top Researchers Temper Timelines**: Prominent AI researchers like **Sutton**, **Karpathy**, and **Hassabis** are adopting longer AI timelines, sparking debate about a potential hype cycle *pop* [according to this thread](https://xcancel.com/scaling01/status/1979485406816092601?s=46).
   - Reactions ranged from alarm to defense of progress, with skepticism about whether pessimism is overstated or misinterpreted in the replies.
- **Cursor Creates Git Worktrees for AI Agents**: **Cursor** now auto-creates Git worktrees [as discussed here](https://xcancel.com/RayFernando1337/status/1979568674433564886?t=CaNW9vyf6jbjoA2qFJdaRw&s=19) allowing users to run multiple AI agent instances on separate branches in parallel.
   - The release elicited praise, tips, and questions about setup and port usage, with some expressing the potential use cases it unlocks.
- **Grok 4.20 Tokes up Pythonic Logic**: Elon Musk teased that **Grok 4.20** can generalize logic from **Python** to other languages [per this tweet](https://xcancel.com/elonmusk/status/1979622705423917216).
   - Speculation arose that **Grok 5** outperforming Andrej Karpathy would signify true **AGI**, sparking timeline and software creation debates.
- **Krea AI Opens the Video Floodgates**: **Krea AI** open-sourced **Krea Realtime**, a 14B parameter autoregressive text-to-video model distilled from **Wan 2.1**, generating video at **11 fps** on a single NVIDIA B200 [as described here](https://xcancel.com/krea_ai/status/1980358158376988747).
   - The release sparked interest in ComfyUI workflows, RTX 5090 performance, and fine-tuning support among users.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1429845521819893880)** (6 messages): 

> `Lightning Pods, X-Ware.v0, Elie's 2025 State of Pre-training Podcast` 


- **Lightning Pods get Notified**: A member announced that they shipped some notable **lightning pods** this weekend, linking to a [tweet](https://x.com/swyx/status/1980286306312999071) for more context.
- **X-Ware.v0 Ships**: The team shipped **X-Ware.v0**, an [Elie’s 2025 “State of Pre-training” podcast recap](https://xcancel.com/swyx/status/1980286306312999071).
- **Elie Drops Pre-Training Podcast Recap**: **Swyx** released an interview where **Elie Bakouch** distills his latest pre-training talk into a 1-hour deep-dive.
   - Topics include **Muon, DeepSeek NSA, Ari Morcos’ BeyondWeb**, and open-source HF tools like **Nanotron & DataTrove**; Swyx also invites suggestions for other high-alpha, unrecorded talks to preserve.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1429379118205505686)** (14 messages🔥): 

> `NPU programming, AMD's NPU approach, eGPUs, tinygrad's eGPU support, RTX 3090 buying guide` 


- ****NPUs** Need Streamlined Programming?**: Concerns arise about the complexity of programming **NPUs**, with speculation that **AMD's** success hinges on simplifying the process, as highlighted in [this tweet](https://x.com/__tinygrad__/status/1980082660920918045).
- ****Tinygrad** Unleashes **NVIDIA eGPUs** on Apple Silicon Macs**: The **tiny corp** team announced public testing of their pure-Python driver enabling **30/40/50-series NVIDIA GPUs** (and **AMD RDNA2-4**) over any **USB4 eGPU** dock on **Apple-Silicon MacBooks**, as detailed in [this announcement](https://xcancel.com/__tinygrad__/status/1980082660920918045).
- **eGPU Buyer Shares Tips for Buying a Used **RTX 3090****: Taha shared lessons learned after buying a used **RTX 3090**: meet seller in person, bring a portable **eGPU** test rig, verify recognition with nvidia-smi, run memtest_vulkan, optionally gpu-burn, and monitor temps as seen in [this tweet](https://xcancel.com/taha_yssne/status/1960418430655586677).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1429548671745130667)** (8 messages🔥): 

> `AI-Generated Luxe Escapism, Endless Summer AI photobooth` 


- ****AI-Generated Luxe Escapism** Feeds Dreams**: Tim Wijaya shared an **OpenAI**-funded study revealing that in **Indonesian Facebook groups** with **30k members**, low-income users (making <$400/month) post AI photos of themselves with Lambos, in Paris, or at Gucci stores ([link](https://xcancel.com/itstimwijaya/status/1979814111069553137?s=46)).
   - Discussion revolves around whether this is purely geographic or tied to socio-economic status and compared it to past Hollywood dreams, generative-AI photo apps, and virtual travel via games.
- **Laurent Del Rey's **Endless Summer** Launches**: Developer Laurent Del Rey launched her first self-built **iOS app**, **Endless Summer**—an **AI photobooth** that generates fake vacation shots ([link](https://xcancel.com/laurentdelrey/status/1975221173840679208?s=46)).
   - The web buzz rallied friends and strangers sharing love, and hopes for future model improvements.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1428864929863041065)** (95 messages🔥🔥): 

> `LLM Training Speed on RTX 3090, Minimum Model Size for Coherence, Pretraining vs Fine-tuning, EleutherAI Discord Server Tag` 


- **LLM Training TPS Tango on RTX 3090**: Members discussed the expected training speed of a **30M parameter LLM** on an **RTX 3090**, with estimates ranging from hundreds to thousands of tokens per second (TPS).
   - One member reported **120 kt/s** on a **4090** using **30m rwkv7 bsz 16 seqlen 2048**, leading to expectations of *high thousands* of TPS.
- **Coherence Conundrums for Tiny Transformers**: A member inquired about the minimum model size required for coherence when pretraining a language model, questioning whether **30M parameters** is sufficient.
   - Suggestions included exploring models around **100M parameters**, such as a **512 window Bert model** or a **GPT2 around 124M**.
- **Pretraining Pilgrimage vs. Fine-tuning Frolic**: Members debated the merits of pretraining from scratch versus fine-tuning, with one member expressing interest in pretraining a model *for fun*.
   - A member recommended starting with normal fine-tuning, sharing a blog post about fine-tuning a tiny **distilBERT** for analyzing tweets and stonks, as a easier option.
- **EleutherAI Emblems: Server Tag Explorations**: Members discussed the possibility of creating an EleutherAI Discord server tag, similar to role icons, for users to display.
   - The options for icons are relatively limited, with community input requested on potential designs and logos (e.g., a water droplet that resembles the logo), as shown in attached [screenshots](https://cdn.discordapp.com/attachments/729741769738158194/1429980908299091999/image.png).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1428934449227366430)** (37 messages🔥): 

> `Attribution graphs, Diffusion models vs LLMs, Continuous and binary rewards in RL, Evaluation-awareness experiments, NormUon optimizer` 


- **Attribution Graphs Expand Beyond MLPs!**: Discussion arose whether attribution graphs, initially applied to MLPs, could be extended to attention mechanisms, inspired by [this video](https://youtu.be/hdi1a9MjwDs?si=taIuYbeF6v-yRSxI&t=628).
   - It was pointed out that **Anthropic** explored this in a follow-up post, detailed in the paper [Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html), which extends attribution graphs to attention while the original **Biology of LLMs** paper only looked at MLPs while freezing attention, as well as the paper [https://arxiv.org/abs/2510.14901](https://arxiv.org/abs/2510.14901).
- **Diffusion Models vs. LLMs: Flops and Memory Access**: It was mentioned that *Diffusion models are much more flops hungry than LLMs*, making methods applicable to them, whereas **LLMs** face memory access overhead during inference due to random routing, hindering kernel optimization techniques.
   - Someone rebutted and said *TREAD is only applied during training, and the FLOPS/fwd improvements are the minor part of the gains*.
- **Finite Sample Proofs Get Comma-lmented!**: A discussion critiqued the writing style in AI papers, particularly regarding comma usage, referencing the paper [https://arxiv.org/abs/2510.14717](https://arxiv.org/abs/2510.14717).
   - One member quipped *I unironically believe that the writing of half of AI papers would be improved if you gave the authors a comma limit*, to which another replied *Then they’ll switch to semicolons or em dashes*.
- **NormUon Optimizer Enters the Ring**: A member mentioned [a new optimizer](https://arxiv.org/abs/2510.05491) that *looks like SOTA* if the results are good.
   - Multiple sources say *it's the same perf as muon on their non-speedrun setups but with good muon baselines* with the observation that `modded-nanogpt does qk norm which is one way you can avoid logit blowups.`
- **Logit Blowups and Weight Distributions**: Discussion revolved around whether a particular **Muon** variant improves logit blowup issues by normalizing neuron-wise norm updates, and its relation to **Kimi k2's** attention clipping.
   - The suggestion was that the **NormUon** optimizer should improve upon **Muon** even without clipping because *the smoother weight distribution is inherently desirable for stability*, especially since updates without clipping increase the spectral rank of the weights.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1429139718217535592)** (13 messages🔥): 

> `Anthropic's Biology of LLMs paper, Cross layer transcoders with attribution graphs for diffusion models, Finetuning Llama-3 to count words, Subtracting the space_direction, Decoupling data complexity from model complexity` 


- **Anthropic's Biology Paper Extends Attribution Graphs**: Anthropic's original **Biology of LLMs paper** focused on **MLPs**, but a new paper extends **attribution graphs** to include **attention mechanisms** as detailed in [this tweet](https://fxtwitter.com/danielmurfet/status/1952656698973499476) and [this tweet](https://fxtwitter.com/danielmurfet/status/1952656715717222589) with a link to [the paper](https://arxiv.org/abs/2508.00331).
   - It seems scientists are seeing *decoupling data complexity from model complexity* and *finding hierarchies of modes in the output* that seems absolutely fascinating.
- **Llama-3 Struggles with Counting After Ablation**: A member finetuned **Llama-3** to count words and observed that ablating the "has_space" direction by subtracting `1.0 * space_direction` only dropped the count from 9 to 8 instead of the expected 7.
   - They clarified their **space_direction** as the difference vector (**W_1 - W_0**) from a 2-class **nn.Linear probe**, where **W_1** represents "has_space" and **W_0** represents "not_space".
- **Subtracting Space Direction Does Not Remove All Space Direction**: A member suggested that act - 1*space_direction *does not remove all of the space direction*.
   - Another member pointed to [figure 1 of this paper](https://arxiv.org/abs/2411.09003) as a helpful example to understand why.
- **Desire for Specific Feature Learning Examples in Papers**: A member expressed disappointment that a **3M model** paper didn't provide specific instances of **feature learning** emerging in training.
   - They quoted the paper: *Just as developmental biologists track cell differentiation to understand organ formation, we can now visualize how token patterns differentiate and organize in susceptibility space, revealing both when and how specialized circuits emerge*.
- **Inquiry About Cross-Layer Transcoders in Diffusion Models**: A member asked if anyone has explored using **cross-layer transcoders** with **attribution graphs** for **diffusion models**.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1429865814605168672)** (1 messages): 

> `Eval Harness, lm-evaluation-harness, MMLU, repeats` 


- **Eval Harness Meeting Scheduled**: A meeting has been scheduled to discuss additions to the **eval harness**, focusing on sharing current plans and gathering feedback from library users, with details available [on When2Meet](https://www.when2meet.com/?33070160-Bw5xm).
   - The main changes are on [this branch](https://github.com/EleutherAI/lm-evaluation-harness/tree/smolrefact).
- **Eval Harness UX Overhaul Incoming**: The team is targeting common pain points to make the **harness more intuitive**, key improvements include: **new templates** for easy format conversion, standardizing formats, making instruct tasks more intuitive, and general UX improvements (e.g. **`repeats`**).
   - They are aiming to enable easier conversion between task variants (e.g., **MMLU** -> cloze -> generation).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1428863525207670984)** (139 messages🔥🔥): 

> `DeepSeek vs Moonshot, Groq's Kimi Implementation, Kimi K2 Troubleshooting, Prediction Markets, Quant Trading` 


- **Moonshot AI vs Deepseek AI**: A user expressed a preference for **Moonshot AI's Kimi** over **DeepSeek**, though recognizing it as a personal opinion.
- **Groq's Kimi Implementation Faces Hiccups**: **Groq's** implementation of **Kimi** experienced a period of instability, with intermittent functionality issues before returning to normal, according to a user.
- **Kimi K2 Troubleshoots Computer Problems**: A user praised **Kimi K2's** ability to provide solid troubleshooting advice, even suggesting the use of **verifier** for stress-testing drivers that may be causing **BSODs**.
- **Kimi K2 excels at prediction markets**: One user shared that Kimi K2 is the best model out there for working with prediction markets and has been using [Kimi K2](https://x.com/rauchg/status/1979660103675687157?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) for it.
- **Discussion on MCP CLI tools**: Users discussed various CLI tools for working with MCP servers and models like **DeepSeek**, **GLM Coder**, and **Qwen**, highlighting **Claude Code** and **Qwen-code** as solid options.
   - The consensus was that **Codex** is only ideal when using with **OpenAI** models.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1428835644968538224)** (94 messages🔥🔥): 

> `GLM 4.6, Claude's Coding Monopoly, AI Learning Resources, LLM Reasoning, OS Model Development` 


- **GLM 4.6 dethrones proprietary models**: With the release of **GLM 4.6** capable of running locally, the OS community looks to break free from reliance on proprietary models like those of **Sam Altman**, **Elon Musk**, and **Dario Amodei**, according to [this YouTube video](https://www.youtube.com/watch?v=bOfoCocOjfM).
- **Harvard course for A.I. newbies**: For those interested in getting into AI with some computer science knowledge, Harvard's CS50 Introduction to Artificial Intelligence with Python ([pll.harvard.edu](https://pll.harvard.edu/course/cs50s-introduction-artificial-intelligence-python)) and this [YouTube playlist](https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&si=iMcYK87ztT7yC4sC) may be a good starting point.
   - One should learn the attention equation, how a transformer works, and what loss functions are and look at Baidu, Tencent or Alibaba technical reports.
- **Vercel suffers major outage**: **Vercel** experienced a major outage, causing downtime for both its chat and portal services.
   - The outage was linked to wider issues with **AWS**, impacting various services, including food ordering platforms.
- **Chinese LLM Mania**: The rise of new **Chinese LLMs** prompts debate about the necessity of multiple models, given the existing presence of several strong contenders like **Qwen**.
   - There is significant internal competition, such as **Ant Group's** Ling models being separate from **Alibaba's** Qwen team.
- **Nous Research promotes Decentralization**: **Nous Research** embraces decentralization through open-source methodologies and infrastructure implementations, exemplified by **Psyche**, as evidenced by these links [Nous Psyche](https://nousresearch.com/nous-psyche) and [Stanford paper](https://cs.stanford.edu/~gakiwate/papers/sigcomm25-centralization.pdf).
   - A member stated, *"Nous successfully decentralizes with their open source methodologies and infrastructure implementations."


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1428850492577939586)** (8 messages🔥): 

> `Sampling method, AI Safety, Clinical AI, Healthcare AI` 


- **Sampling Method Extends Research**: A member shared a paper about a novel [sampling method](https://arxiv.org/abs/2510.14901) that is *scientifically grounded* and controlled by the parameter **N**.
   - Another member noted that it is *almost like an extension* of [this research](https://discord.com/channels/1053877538025386074/1104063238934626386/1427069164765450390) and wondered if post-training could be formalized as a sampling problem, suggesting to test it on a modern model like **DeepSeek-Base/Instruct**.
- **Safety Focus in Clinical/Healthcare AI**: One member is *focusing on **AI safety*** by proposing research on clinical/healthcare AI to create and evaluate benchmarks for AI models.
   - They seek feedback on whether this is a good research topic, referencing the [International AI Safety Report 2025](https://assets.publishing.service.gov.uk/media/679a0c48a77d250007d313ee/International_AI_Safety_Report_2025_accessible_f.pdf) to incorporate general field standards.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1429008509760045137)** (2 messages): 

> `ScaleRL for Sparse Models, Trajectory vs Sample Level Loss Aggregation, Iterative RL Reasoning` 


- **ScaleRL fits Sparse Reasoning Models Nicely**: Meta's recent Art of Scaling RL paper suggests that [**ScaleRL**](https://x.com/Devvrit_Khatri/status/1978864275658871099) appears almost tailor-made for **sparse (e.g., MoE) reasoning models**.
- **Trajectory vs Sample Level Aggregation**: The question arises: Does **trajectory-level loss aggregation granularity** always perform better than **sample-level** ones for **Iterative RL (Reasoning)**?
   - It was noted that [trajectory-level aggregation preserves long-range credit assignment](https://x.com/ditpoo/status/1979749686492717207), but forgets iterative improvements, while the models might drift toward memorization-like pattern imitation.
- **Refining RL with Iteration and Diversity**: The **RL objective** may lack ways to encourage improvement or diversity across similar prompts because **reward credit** is coarse and global.
   - It was suggested that *step/steps (iteration) level might be a middle ground* for tuning.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1428850492577939586)** (8 messages🔥): 

> `Sampling Method, International AI Safety Report 2025, Healthcare AI Safety` 


- **Paper on Sampling Method Spurs Discussion**: A member shared [a paper](https://arxiv.org/abs/2510.14901) that spurred a discussion about its sampling method, with one member suggesting it's similar to **MCTS** but *better because it's well scientifically grounded*.
   - Another member noted it's *like an extension* to [a previous discussion](https://discord.com/channels/1053877538025386074/1104063238934626386/1427069164765450390) and wished the authors had tested the method on a modern model like **DeepSeek**.
- **Healthcare AI Safety Focus Proposal**: A member proposed research on **AI safety** with a focus on clinical/healthcare applications, aiming to create benchmarks for AI models with precise accuracy.
   - They linked to the [International AI Safety Report 2025](https://assets.publishing.service.gov.uk/media/679a0c48a77d250007d313ee/International_AI_Safety_Report_2025_accessible_f.pdf) and asked for views on whether this is a good research topic.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1428834356046332096)** (107 messages🔥🔥): 

> `Manus infrastructure outage, Manus credits disappearing, Free perplexity pro, Open Sourcing Manus, Manus google drive connection` 


- **Manus Google Drive Connection, is it Back?**: Users confirmed that [Manus connects to Google Drive](https://drive.google.com) by clicking the **+** button and adding files, although it's not available as a connected app.
- **Manus Project Files Vanish, Credits Disappear!**: A user reported that after building a site and spending **7000 credits**, everything in the project disappeared, including files and the database, and support was unhelpful.
   - Another user echoed this, reporting losing nearly **9000 credits** and finding *no files or preview*.
- **Manus MVP Android Frontend App Created Cheaply**: A user made an Android frontend app MVP using **525 Manus credits**, then used Claude to fix issues.
   - The user praised Manus' UI/UX capabilities, sharing [images of the app](https://cdn.discordapp.com/attachments/1349440650495398020/1429746630935969802/2025-10-19_18-06.jpg?ex=68f7eb90&is=68f69a10&hm=eb47b7c3e935587fd229b70648d0dcf7043ea52556918a996c0872722972e7b7&).
- **Manus Suffers Infrastructure Outage**: Manus experienced a temporary outage due to an infrastructure provider issue, with some users in certain regions still facing errors accessing the homepage.
   - The team communicated updates and thanked users for their patience, reporting that *most Manus services are back online*.
- **Free Perplexity Pro Promo Causes Discord Disruption**: A user shared a [referral link for a free month of Perplexity Pro](https://pplx.ai/muhammadze84187), prompting a negative reaction from another user who told them to *Stfu!* and *Get a job*.
   - It was a *no drama, no lies, no clickbait* way to earn real money, according to the linked user.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1429431345519525929)** (54 messages🔥): 

> `Mojo compiler self-hosted?, Mojo and MLIR, MAX Kernels Backend, Advantages of MAX Dynamic Shapes, Mojo vs Python` 


- **Mojo compiler: C++ now, Mojo later?**: The Mojo compiler uses **LLVM** and is tightly bound to it, lowering to **MLIR**; a full self-host is unlikely unless Mojo absorbs LLVM.
   - While a complete rewrite to Mojo is possible, it's deemed *a lot of work for not a ton of benefit*, but C++ interop might make it easier in the future.
- **Mojo, DSL for MLIR: A Flexible Foundation**: Mojo effectively functions as a **DSL for MLIR**, parsing directly into **MLIR** and using it instead of normal ASTs, giving it significant flexibility.
   - Mojo employs numerous dialects, with most being brand new, aside from the **LLVM dialect**, making it adaptable to various computational needs.
- **MAX Kernels: Mojo's secret weapon for Pytorch**: There is interest in switching the backend of **JAX** to use **MAX kernels**, potentially interfacing with **C++** as a fun project, and there is a mostly functional **MAX backend** for **PyTorch** already.
   - Mojo can *`@export(ABI="C")` any C-abi-compatible function*, but to talk to **MAX** you currently need to use Python.
- **Dynamic Shapes: Advantage of MAX**: One member mentioned that interpreting **jaxpr** to build a max graph is possible ([jax.dev](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)).
   - It was pointed out that using **MAX** instead of a Mojo wrapper around **XLA** preserves advantages like **dynamic shapes**.
- **Mojo's Goal: Preventing Pythonic Fragmentation**: The goal of Mojo is to prevent the Pythonic split between users of Python and users of CPython, by offering ways to move between more pythonic `def` and more systems-y `fn` code.
   - One member said that *the goal is to leave the door open to lower-level control but give safe default ways for people to not shoot themselves in the foot; let's hope we can deliver.*


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1429665499737755796)** (12 messages🔥): 

> `UDP Sockets in Mojo, Mojo vs Rust vs C++, itertools package in Mojo` 


- **Mojo Debates UDP Socket Support**: Users inquired about **UDP socket support** in Mojo, and were informed that while it's possible via **libc**, full standard library support is pending for proper implementation.
   - The response indicated a preference for doing it *"right not fast"* and noted *"language-level dependencies"* as a factor.
- **Mojo's General Purpose Future Compared to Rust/C++**: Users questioned if Mojo is a general-purpose language and a potential replacement for **Rust** and **C++**.
   - The response indicated that Mojo aims to be general purpose, with current strengths in **numerics**, and could potentially replace Rust and C++ *"once it’s done."*
- **Mojo Adds itertools Package Sparking Debate**: Mojo added an `itertools` package via [this commit](https://github.com/modular/modular/commit/648770359ecc5388aababd3418c14bfaf90ca161), leading to questions about copying Python's module structure.
   - Concerns were raised about potential divergences from **Python behavior**, hindering more performant solutions, versus creating a **Python compatibility layer via extensions**.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1429491355184267284)** (4 messages): 

> `Max Backend for PyTorch, PyTorch Nightly Use` 


- **Max Backend quest for PyTorch**: The path to a good **max backend** for pytorch is discussed, pointing to [this github issue](https://github.com/pytorch/pytorch/issues/165811).
   - One member noted *Han Qi is the GOAT*.
- **PyTorch Nightly gets torch-max-backend**: One reason for starting to [use pytorch nightly](https://github.com/gabrieldemarmiesse/torch-max-backend/commit/b34182722d444fb7f71ff097532c0e2af98ac6ed) in the main branch is to get a recent PR from **Han Qi** in pytorch.
   - It was needed a very recent PR from **Han Qi** in pytorch, so it's related.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1429677972658589769)** (1 messages): 

> `LM Studio, DSPy framework, llms.txt generator` 


- **LM Studio Receives llms.txt Generator Powered by DSPy**: A member shared a [llms.txt generator](https://github.com/AcidicSoil/lms-llmsTxt/tree/main) for **LM Studio** powered by the **DSPy framework**.
   - This tool allows users to easily generate `llms.txt` for repos lacking it, leveraging any LLM available in LM Studio; the member recommended using *"osmosis-mcp-4b@q4_k_s"* for generating example artifacts.
- **lms-llmsTxt on Github**: Github user AcidicSoil has published the files at [lms-llmsTxt on Github](https://github.com/AcidicSoil/lms-llmsTxt/tree/main).
   - Easy way to generate llms.txt for any repo that might not have it using pretty much any llm available in lm-studio.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1428983296607387668)** (64 messages🔥🔥): 

> `Claude Agents, Clojure REPL environments, Typing DSPy, Gemini models in DSPy, RLM implementation` 


- **Integrating Claude Agents Inside DSPy Programs**: Members discussed integrating **Claude agents** within **DSPy**, referencing [an example](https://x.com/dronathon/status/1979042187733004291) where someone implemented a **Codex agent**.
   - A member stated that they would like to see someone implement this, especially for **Claude code**, and mentioned difficulties encountered due to the lack of an SDK.
- **Clojure Considerations for DSPy**: A member raised questions about using DSPy in **Clojure REPL** environments, highlighting potential differences in **data representation**, concurrent **LM calls** due to immutability, and **inspecting generated functions**.
   - This sparked discussion about the challenges of adapting DSPy to different programming paradigms.
- **Typing DSPy**: Members discussed the status of **fully typing DSPy** (Python) and whether **Python generics** are sufficient for this purpose.
   - One member affirmed it was doable, focusing on optimizing the **input handle/initial prompt**, but cautioned against optimizing without a clear task and scoring mechanism.
- **Gemini Models and DSPy Configuration**: A user asked about using **Gemini models** in **dspy.lm**, to which members clarified that it's possible by [setting the correct configuration](https://dspy.ai/#__tabbed_1_4) and providing the appropriate API key.
   - One member humorously related to the 'painful journey' of finding the correct API key, recommending the use of **AI Studio** over the console.
- **Tackling Scanned PDFs with DSPy**: Users discussed methods for **DSPy agents to read scanned PDFs**, mentioning options like converting **PDFs to images** and using **VLM/OCR-capable LLMs** such as Claude Code.
   - Another member suggested using a `read_pdf` tool with the **vanilla Gemini API** or pointed to a guide on using [PaddleOCR](https://dev.to/czmilo/2025-complete-guide-paddleocr-vl-09b-baidus-ultra-lightweight-document-parsing-powerhouse-1e8l#use-cases).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1428860160692191353)** (38 messages🔥): 

> `Weekly tautological counter, Reinforcement learning framework for PyTorch, Graph neural networks, AI engineer qualifications, ML Debugging interviews` 


- **Tianshou: Reinforcement Learning Library Surfaces**: A member inquired about a **reinforcement learning framework** for PyTorch, and another suggested [Tianshou](https://github.com/thu-ml/tianshou) as a viable option.
   - The suggestion came with a lighthearted remark about its relevance to graph neural networks.
- **Defining "AI Engineer": LinkedIn vs. Reality**: The discussion revolved around the question of what constitutes an **"AI Engineer,"** with some joking that [using the OpenAI API in Python](https://platform.openai.com/docs/api-reference) qualifies one on LinkedIn.
   - Others quipped that even [assembling Legos in visual n8n](https://www.n8n.io/) or creating a custom GPT could suffice.
- **Tackling ML Debugging Interview Prep**: Someone requested advice on how to prepare for an **"ML Debugging"** coding interview, and one member recommended being ready to discuss [handling overfitting](https://www.youtube.com/watch?v=PykNdM4v4Xo).
   - Another pointed to [ChatGPT](https://chatgpt.com/share/68f68935-3148-8005-907f-86ec2ed6e93c) as a potential mock interview tool.
- **IntelliCode Mind-Reading Abilities**: A member shared an experience with **Microsoft's IntelliCode** in Visual Studio, noting its impressive ability to predict and suggest code completions, especially given extensive context.
   - The member elaborated that the model's effectiveness stems from the wealth of context it receives, including classes, open files, and recent code elements.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1429021327276113991)** (18 messages🔥): 

> `Lip Movement Algorithm, Paper Machine Unlearning, Backscatter IoT Applications, RL Training-Free Sampling Method` 


- **Mouth Sync Algorithm Quest Kicks Off**: A member is looking for a good open-source **lip movement algorithm** for matching to audio to modernize an old comedy rant.
- **Promote Paper Machine Unlearning To ArXiv Category**: Members discussed promoting **papermachine unlearning & knowledge erasure** to its own arXiv category (cs.UL/stat.UL).
- **Backscatter IoT Has Excellent Method for Low-Energy**: **Backscatter** is generally an excellent method for **low-energy IoT applications** and the University of Washington did some good work with it in years previous.
- **RL Training-Free Sampling Methods**: A member shared a link to a paper that introduces a **training-free sampling method** that matches **RL trained reasoning** on many zero shot tasks ([https://arxiv.org/abs/2510.14901](https://arxiv.org/abs/2510.14901)).


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1429160542500683958)** (2 messages): 

> `AI Agent Movie Inspiration, Voice integration` 


- **AI Agent provides helpful tools to human**: A member found inspiration in a movie where an **AI agent gave the human tools**.
   - They thought *this is inspiring for agent development as I am also working on voice now*.
- **Voice integration is inspiring**: The member who enjoyed the **AI agent movie** is now working on voice.
   - The agent gave the human **helpful tools**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1428861954243694702)** (5 messages): 

> `Qwen3 Vision Model, aidaw.com, Unitree Robotics, DeepSeek-OCR` 


- **Qwen3 sees the vision**: A new **Qwen3 Vision model** has been released and is available on [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- **aidaw Announces Something**: A member shared a link to [aidaw.com](https://aidaw.com/).
- **Unitree Robotics Spots**: A member shared a link to [UnitreeRobotics](https://fixupx.com/UnitreeRobotics/status/1980140278930661501).
- **DeepSeek Eyes OCR**: A member shared a link to [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1428836342770434152)** (32 messages🔥): 

> `Aider Status and Roadmap, Integrating Agentic Extensions into Aider, Devstral Small Model Feedback, aider-ce vs Codex CLI` 


- ****Aider's Pulse: Not Dead, Just Slow!****: Users discussed the status of Aider, with some wondering if it was dead due to slower releases, but others clarified that it's not dead, just that releases are slow, and that using **aider-ce** provides more recent updates.
   - One user mentioned cloning the repository recently and plans to spend the next few weeks familiarizing themselves with the code base to help revive the project's momentum via increased commit frequency reported at [GitHub contributors](https://github.com/Aider-AI/aider/graphs/contributors).
- ****Aider Evolution: Agentic Extension Integration?****: A developer is creating an agentic extension for Aider using **LangGraph**, featuring task lists, RAG, and user command handling, and inquired whether it would be better to integrate it directly into Aider or as a separate project.
   - The core philosophy of keeping Aider simple and straightforward was emphasized, with the goal of competing with top-tier agent solutions without sacrificing its core functionality.
- ****Devstral Small Model Steals the Show!****: A user reported surprisingly good results with the `Devstral-Small-2507-UD-Q6_K_XL` model on a laptop with 32GB of RAM, praising its ability to self-correct, handle large contexts, and perform well on various coding tasks in PHP, Python, and Rust, even supporting image.
   - They felt this model deserves better recognition in Aider benchmarks, noting it outperformed `Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL`, and highlighted its architectural thinking and tool use capabilities, recommending [Unsloth's XL quantized versions](https://github.com/unslothai).
- ****Aider-CE is making waves against Codex CLI****: After testing gemini-cli, opencode, and Claude (with claude-code-router to use DeepSeek API), a user has switched back to aider as their main tool, highlighting its sophisticated grep-based code search/replace system and self-updating todo list (*use with --yes-always and get more done!*).
   - The user also noted the value of Aider's simplicity and straightforwardness for coding tasks, preserving the functionality of the original **/ask** and **/architect** modes, and the value of the MCP formatting in the .aider.conf.yml).


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1429065423571386410)** (5 messages): 

> `Commit Message Reasoning, Read-Only Files, Aider Style Guidelines` 


- **Reasoning Disabled for Commit Messages?**: A user inquired about disabling reasoning for commit messages, citing that **Deepseek V3.1 Terminus** via **OpenRouter** took too long to generate a one-line message.
   - Another user suggested copying the API's reasoning in resources and setting a new alias to a weak model.
- **Navigating Read-Only Files**: A user asked about best practices for managing read-only files when using **Aider**.
   - The user mentioned that sometimes they have a few more files and they want them to be on read-only.
- **Aider Style Guidance Strategies**: A user asked about supplying appropriate style guidelines to **Aider**.
   - A member suggested putting the style guidelines in a file and loading it with `--read CONVENTIONS.md` and telling aider config to always load such files read only.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1428820727322116126)** (12 messages🔥): 

> `nanochat, SHRINK and PAD, CI for external PR, usb gpu, MFLOPS and MB/s` 


- **Karpathy Launches nanochat**: Andrej Karpathy launched [nanochat](https://github.com/karpathy/nanochat), a minimalistic chat application.
   - The poster questioned how significant the release was.
- **Left Side Removed from SHRINK and PAD**: A member is working on a bounty to *remove the left side arg from SHRINK and PAD*.
   - The original bounty was discussed at length in the [Discord general channel](https://discord.com/channels/1068976834382925865/1068982781490757652/1427878481190195202).
- **tinygrad says Bye Bye ShapeTracker**: tinygrad is planning to deprecate **ShapeTracker** during meeting #92.
   - The discussion points during the meeting included: company update, **bye bye shapetracker**, [usb gpu](https://x.com/__tinygrad__/status/1980082660920918045), **multi output kernel**, rangeify regressions, openpilot, resnet, bert, FUSE_OPTIM, assign, more cleanups, viz, driver, tiny kitten, more symbolic?, other bounties, new linearizer, new clang2py.
- **Contributor Asks for MFLOPS and MB/s**: A contributor asked for **MFLOPS** and **MB/s** to be added in yellow on the **DEBUG=2** line.
   - They asked that the implementer write *clean code* and not use *AI code you don't understand*!
- **macOS Nvidia drivers are actually real**: Nvidia drivers for macOS were successfully produced.
   - Make sure to run `brew tap sirhcm/tinymesa` to get it to work.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1429962916987408436)** (5 messages): 

> `Gradient Accumulation, TinyJit, Manual Gradient Division` 


- **Gradients Add Up When Calling Backward Multiple Times**: A member inquired if calling `backward` multiple times before calling `optimizer.step` simply adds the gradient contributions, and confirmed it seems that way.
- **Gradient Accumulation Issues in TinyJit**: A member questioned the math in [model_train.py](https://github.com/tinygrad/tinygrad/blob/c7c59e6dd71158f50bbb9a87298b4ed1d65a6fb6/examples/mlperf/model_train.py#L1375C1-L1390C54) regarding gradient accumulation, suspecting it was broken in **TinyJit**.
   - Another member confirmed running into issues with grad accumulation and rewrote the gradient addition step using assign to make it work.
- **Making Grad Accum Work with Manual Gradient Division**: A member got gradient accumulation to work by setting `reduction=sum`, manually counting non-padding tokens, performing backward on each microbatch, and then dividing the gradients.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1429476007064371231)** (5 messages): 

> `MCP Access, Webrix MCP Gateway, Docker MCP Gateway Multi-Tenant, MCP auth extension, oauth scope granularity` 


- **DevOps Admin Seeks Secure MCP Access Solutions**: A DevOps admin is seeking ways to securely provide **MCP access** to non-technical users within their organization, aiming to avoid managing keys and implement an **Identity Provider (IDP)** layer.
   - They are considering [Webrix MCP Gateway](https://docs.webrix.ai/docs/admin/monitor-logs) and [making Docker MCP Gateway multi-tenant](https://github.com/docker/mcp-gateway/issues/130) and reached out for advice from a user who previously discussed securing **MCPs with Okta**.
- **Enterprise Managed Auth Profile Emerges as Solution**: A member noted that the **enterprise managed auth profile** is designed precisely for this use case and is being published as an **MCP auth extension**.
   - However, it was noted that fine-grained permissions are currently limited to **oauth scope granularity**.
- **Discord Aimed at Contributors**: It was noted that this Discord is not oriented toward technical support but rather intended to facilitate communication between contributors to the **MCP protocol** and related projects.
   - Users seeking help were encouraged to DM for links to more appropriate communities.



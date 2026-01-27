---
id: MjAyNi0w
title: not much happened today
date: '2026-01-22T05:44:39.731046Z'
description: >-
  **Anthropic** launches "Claude in Excel Pro" with enhanced features.
  **OpenAI** reveals upcoming **Codex** agent loop and cybersecurity measures.
  **Google** boosts **Gemini App** quotas and partners with **Sakana AI** for
  advanced AI Scientist projects in Japan. **Cursor** introduces Agent Skills
  for dynamic context focus. **GPT-5.2 Pro** achieves **31%** on FrontierMath
  Tier 4, showing significant benchmark progress. **Baseten** raises **$300M**
  at a **$5B valuation** targeting high-performance inference. Discussions
  highlight math benchmarks as indicators of AI capability, uneven AGI progress,
  and the importance of reasoning and continual learning as future frontiers.
  Notable figures include *Sam Altman*, *François Chollet*, *Shane Legg*, and
  *Demis Hassabis*.
companies:
  - anthropic
  - openai
  - google
  - sakana-ai
  - cursor
  - baseten
  - epoch-ai-research
  - deepmind
models:
  - claude-3
  - codex
  - gemini
  - gpt-5.2-pro
topics:
  - benchmarking
  - reasoning
  - continual-learning
  - reinforcement-learning
  - model-performance
  - agentic-ai
  - security
  - model-training
people:
  - sama
  - fchollet
  - shane_legg
  - demishassabis
---


**a quiet day**

> AI News for 1/22/2026-1/23/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**206** channels, and **7161** messages) for you. Estimated reading time saved (at 200wpm): **579 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!




---

# AI Twitter Recap


**Top tweets (by engagement)**

- **Anthropic ships “Claude in Excel”**: Claude in Excel expands to Pro, with multi-file drag/drop, safer cell writes, and longer sessions via auto-compaction ([claudeai](https://twitter.com/claudeai/status/2014834616889475508)). Big engagement discussion about Microsoft 365 Copilot lagging ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014835455393726726)).
- **OpenAI roadmap + agent loop**: Sam Altman says Codex launches are coming and OpenAI is nearing a “Cybersecurity High” level with restrictions and later “defensive acceleration” ([sama](https://twitter.com/sama/status/2014733975755817267)). OpenAI publishes a technical deep dive into the **Codex agent loop / harness orchestration** ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2014794871962533970)).
- **Google AI Ultra limits boosted**: Gemini App daily quotas increased to **1,500 Thinking** + **500 Pro** prompts/day for Ultra members ([joshwoodward](https://twitter.com/joshwoodward/status/2014566936479437173)).
- **Sakana AI ↔ Google partnership + investment**: Sakana announces strategic partnership and funding from Google to combine **Gemini/Gemma** with Sakana’s “AI Scientist” / “ALE-Agent” work and to deploy in high-security domains in Japan ([SakanaAILabs](https://twitter.com/SakanaAILabs/status/2014686043711406355), [hardmaru](https://twitter.com/hardmaru/status/2014686852691918971), [JeffDean](https://twitter.com/JeffDean/status/2014716109216448975)).
- **Cursor launches Agent Skills**: First-class “Skills” for agents, emphasizing discovery + dynamic context focus ([cursor_ai](https://twitter.com/cursor_ai/status/2014753596223770841)).
- **FrontierMath jump**: **GPT-5.2 Pro hits 31% on FrontierMath Tier 4**, up from 19% previous best ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2014769359747744200)). Practitioners highlight usefulness and even benchmark issue-spotting ([gdb](https://twitter.com/gdb/status/2014859263701839963)).
- **Claude Code “run locally for free” how-to**: A popular tutorial claims running Claude Code-like workflows locally with open models, private + tool-enabled ([dr_cintas](https://twitter.com/dr_cintas/status/2014771670070747278)).
- **Baseten raises $300M** at **$5B valuation**, positioning around the “many-model future” and high-performance inference ([basetenco](https://twitter.com/basetenco/status/2014755013344792595), [tuhinone](https://twitter.com/tuhinone/status/2014755252244005273)).

---

**Frontier models, benchmarks, and the “capability” narrative**

- **Math as a leading indicator (FrontierMath + cross-benchmark correlations)**: Epoch reports **GPT-5.2 Pro = 31%** on FrontierMath Tier 4 (no overfitting claimed), a sizable step up from 19% ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2014769359747744200)). Separate Epoch analysis argues benchmark scores correlate strongly across domains (≈**0.68** across domains, ≈**0.79** within-domain), implying a latent capability factor behind “math/coding/reasoning” progress ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2014806095504785664)). Practitioners note concrete value: catching problem flaws/typos and even “pointing out a flaw” in a Tier 4 problem ([gdb](https://twitter.com/gdb/status/2014859263701839963), [GregHBurnham](https://twitter.com/GregHBurnham/status/2014774878591655984)).
- **AGI timelines vs product reality**: A recurring theme is “systems are uneven”: smart in formal domains, unreliable elsewhere. A widely shared quip captures this mismatch (“smarter than a PhD in math, dumber than an intern”) ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014564105194242452)). François Chollet stresses that progress is **vertical-specific** (especially in verifiable domains like code) because unlimited synthetic data makes memorization/operationalization easier there, and warns against extrapolating to all human tasks ([fchollet](https://twitter.com/fchollet/status/2014821042464948270)).
- **Reasoning + continual learning as the “real” frontier**: Reporting from interviews claims Shane Legg pegs **50% chance of “minimal AGI” by 2028** with Google’s definition including continuous learning/memory/world models ([kimmonismus](https://twitter.com/kimmonismus/status/2014697026890416586)). Follow-on notes say Demis Hassabis explicitly says DeepMind **has not solved continual learning**, and is exploring combinations of AlphaZero-like approaches with foundation models ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014785682309579119), [Hangsiin](https://twitter.com/Hangsiin/status/2014774897680253442)).
- **Model/arch discourse: MoE provenance fights**: A thread disputes the claim that DeepSeek MoE “built on Mixtral,” arguing the DeepSeek MoE paper appeared almost immediately after Mixtral’s arXiv release, Mixtral training details were sparse, and DeepSeek’s MoE is architecturally different/more sparse and cites **GShard** not Mixtral ([eliebakouch](https://twitter.com/eliebakouch/status/2014575628675092845)). Another framing calls DeepSeek a distinct “neoMoE” tree vs “oldMoE” ([kalomaze](https://twitter.com/kalomaze/status/2014659449219383367)).
- **Second-tier multimodal updates (China)**: A detailed Chinese-language review positions **Baidu ERNIE 5.0** as improved and stable but still costly (2T params, ~61K context) and “firmly second tier” vs top multimodal systems with large compute budgets ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2014606592826912840)).

---

**Agents and coding: from workflows → harnesses → skills**

- **OpenAI Codex “agent loop” becomes explicit**: OpenAI publishes how Codex orchestrates turns: assemble inputs → run inference → execute tools → feed results back until stopping, i.e., the agent harness as a first-class system component ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2014794871962533970)). This aligns with broader commentary that “training better models is only one axis; harness + experimentation can surprise” ([Hangsiin](https://twitter.com/Hangsiin/status/2014794375466033657)).
- **Workflows vs agents is collapsing into “skills / guidance / RLMs”**: A strong technical synthesis argues the agent/workflow boundary is less about “control flow in code” and more about **state representation**, **dynamic instruction selection**, and **who dictates composition**—with Replit’s “Decision-Time Guidance,” Skills, and **Recursive Language Models (RLMs)** as hybrid points on the design spectrum ([lateinteraction](https://twitter.com/lateinteraction/status/2014685012994515206)). DSPy community posts push RLMs as a practical method for “arbitrarily long prompts” by delegating to code + subcalls instead of summarization loss ([getpy](https://twitter.com/getpy/status/2014717862246756384)).
- **Cursor: Agent Skills shipped**: Cursor introduces **Skills** as discoverable specialized prompts/code and pitches them as also improving context focus via dynamic discovery ([cursor_ai](https://twitter.com/cursor_ai/status/2014753596223770841), [cursor_ai](https://twitter.com/cursor_ai/status/2014753597624598665)). This is echoed by the broader market trend: “make non-devs write code by calling it skills” ([kylebrussell](https://twitter.com/kylebrussell/status/2014689618617122883)).
- **Claude Code ecosystem keeps expanding (and copying wars)**: Multiple posts highlight rapid feature diffusion between tools (“cursor adopting popular features from claude code”) ([dejavucoder](https://twitter.com/dejavucoder/status/2014635509025526198)). Practical snippets: Claude tasks stored on filesystem (`~/.claude/tasks`) enabling multi-session/subagent collaboration via broadcasts ([dejavucoder](https://twitter.com/dejavucoder/status/2014584272183861407)). At the same time, pain points remain (e.g., absurd file download hacks via base64) ([dbreunig](https://twitter.com/dbreunig/status/2014540341526069738)).
- **Security posture is becoming a headline feature**: Sam Altman states OpenAI will increasingly constrain coding models for cybercrime and later pivot to **defensive acceleration** (helping patch bugs) as mitigation ([sama](https://twitter.com/sama/status/2014733975755817267)). One anecdote flags a potential security footgun: Codex Slack integration produced shareable task links accessible without auth in incognito (if accurate, it’s an urgent product-security issue) ([apsdehal](https://twitter.com/apsdehal/status/2014770563810758938)).
- **Enterprise “agents fail in production” reminder**: A long post claims **95% of enterprise AI pilots fail** (citing MIT research), emphasizing that production viability is about **authorization-aware retrieval**, **guardrails**, **monitoring**, and **auditability**, not demo capability ([victorialslocum](https://twitter.com/victorialslocum/status/2014654495301525683)).

---

**Inference + systems: vLLM, KV compression, storage, and infra maturity**

- **vLLM keeps becoming the open inference “substrate”**: vLLM positions itself as the bridge from open models to deployable inference, highlighting vLLM Studio workflows ([vllm_project](https://twitter.com/vllm_project/status/2014536660361584833)). A notable infra-engineering post documents a difficult **vLLM memory leak** debugging path (Python profilers → pmap → BPFtrace → GDB) traced to **UCX mmap hooks**; fix merged upstream ([vllm_project](https://twitter.com/vllm_project/status/2014630499231412477)).
- **System intelligence / routing**: vLLM announces a public beta of **vLLM-SR** (Semantic Router) on AMD, framing it as a “system intelligence” approach rather than monolithic models doing everything ([XunzhuoLiu](https://twitter.com/XunzhuoLiu/status/2014672307407704279)).
- **KV cache compression via distillation**: NVIDIA Research releases **Qwen3-8B-DMS-8x**, claiming **8× KV cache compression** with minimal overhead and only ~1K fine-tuning steps, outperforming token-importance eviction proxies; compatible with sparse attention methods ([p_nawrot](https://twitter.com/p_nawrot/status/2014770473289019709)).
- **Tooling for predictable deployment**: `hf-mem` estimates inference VRAM from Safetensors metadata without downloading weights, aiming to eliminate trial/OOM loops ([LiorOnAI](https://twitter.com/LiorOnAI/status/2014730309128855801)).
- **Storage and data-plane attention**: SkyPilot pushes “Volumes” for high-performance storage (AI checkpoints/data) as object stores aren’t always fit ([skypilot_org](https://twitter.com/skypilot_org/status/2014752751545381044)). Jina proposes a neat compression trick: convert embeddings to spherical coordinates pre-compression, claiming near-lossless reconstruction below float32 epsilon and ~1.5× storage savings ([JinaAI_](https://twitter.com/JinaAI_/status/2014753001387499927)).
- **GPU kernel evaluation for agents**: AMD AGI releases **Magpie**, an open-source kernel eval suite for correctness + performance across AMD/NVIDIA, designed for agent workflows; claims **3000× token efficiency** vs using GPU profilers alone, and plans tracing integrations with SGLang/vLLM ([realSharonZhou](https://twitter.com/realSharonZhou/status/2014722290865549649)). MLSys 2026 launches FlashInfer-Bench contest tracks (MoE/DSA/GDN) with separate human vs agent evaluation ([ye_combinator](https://twitter.com/ye_combinator/status/2014836302198472789)).

---

**Ecosystem + business: partnerships, pricing, and “value-sharing” debates**

- **Sakana AI ↔ Google: strategic partnership + funding (and controversy)**: Sakana frames the deal as combining Google infra/models (Gemini/Gemma) with Sakana’s research automation (AI Scientist, ALE-Agent) and pushing deployments in mission-critical domains requiring security/data sovereignty ([SakanaAILabs](https://twitter.com/SakanaAILabs/status/2014686043711406355)). Media echoes it (Nikkei/Bloomberg/etc.) ([nikkei](https://twitter.com/nikkei/status/2014637546563658172), [business](https://twitter.com/business/status/2014594583234027753)). A dispute emerges: one claim says it’s a small Google Cloud Japan compute deal and “DeepMind not involved” ([shaneguML](https://twitter.com/shaneguML/status/2014847946110783649)), while Sakana leadership publicly counters that DeepMind is indeed involved and tags Demis/Jeff Dean ([hardmaru](https://twitter.com/hardmaru/status/2014885853789884416)).
- **Baseten’s “many-model future” + fundraising**: Baseten raises **$300M at $5B** and argues inference is the bottleneck enabling millions of specialized models and reliable low-latency UX ([basetenco](https://twitter.com/basetenco/status/2014755013344792595), [tuhinone](https://twitter.com/tuhinone/status/2014755252244005273)).
- **Anthropic economics: inference cost pressure**: A report says Anthropic cut 2025 gross margin outlook to **40%** due to inference costs **23% higher than expected**, despite projected **$4.5B revenue** (~12× YoY) ([kimmonismus](https://twitter.com/kimmonismus/status/2014673235594641838)).
- **“Value-sharing” model for AI-enabled discoveries**: Reporting claims OpenAI’s CFO discussed deals taking a cut of customer profits/IP (starting with drug discovery), akin to Isomorphic Labs’ model ([kimmonismus](https://twitter.com/kimmonismus/status/2014643034089259103)). Some push back on sensationalism and incentives (e.g., you can’t both sell tokens and own discoveries without also eating compute cost) ([code_star](https://twitter.com/code_star/status/2014541663356772516), [paul_cal](https://twitter.com/paul_cal/status/2014692633730261339)).

---

**Multimodal + voice + video: quality leaps and tooling**

- **Voice is accelerating (open + low-latency)**: Teknium claims an open voice cloning HF demo is the closest to ElevenLabs quality they’ve seen in open models ([Teknium](https://twitter.com/Teknium/status/2014687269329031253)). NVIDIA releases **PersonaPlex**, an open-source real-time full-duplex conversational voice stack optimized for very low latency ([kimmonismus](https://twitter.com/kimmonismus/status/2014703479491854751)).
- **Video generation: controllability and arenas**: Runway Gen-4.5 I2V adds more precise “zoom into specified regions” control ([c_valenzuelab](https://twitter.com/c_valenzuelab/status/2014674372120785176)) and creators showcase short-film workflows ([Artedeingenio](https://twitter.com/Artedeingenio/status/2014693398502842731)). LMSYS Arena launches/expands **Video Arena** leaderboards (Veo, Sora 2, Kling, Hailuo, etc.) ([arena](https://twitter.com/arena/status/2014815916056576257)).
- **3D agents and world models for interactive environments**: Berkeley demo “VIGA” claims a multimodal agent that generates 3D/4D Blender scenes from images with no training ([HavenFeng](https://twitter.com/HavenFeng/status/2014765400563781777)). A smaller “world model you can play” demo appears on HF (Waypoint-1-Small, 2.3B) ([victormustar](https://twitter.com/victormustar/status/2014766391811826022)). Separately, “world models are next big wave for gaming/robotics” sentiment resurfaces ([kylebrussell](https://twitter.com/kylebrussell/status/2014529425983914098)).

---

**Security, trust, and integrity issues in the AI social layer**

- **Account compromises targeting AI insiders**: Multiple warnings indicate prominent accounts (Deedy Das; a Kimi researcher/“Crystal”) were hacked and used for phishing/scams, likely crypto-driven ([cloneofsimo](https://twitter.com/cloneofsimo/status/2014536638010163262), [ml_angelopoulos](https://twitter.com/ml_angelopoulos/status/2014543018137944486), [Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2014571513299796154), [Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014572557270450194)).
- **Misinformation / fake papers**: A fake “llama 4” arXiv paper is flagged as not actually Meta-authored ([TimDarcet](https://twitter.com/TimDarcet/status/2014626676798366006)).
- **Open-source “layers” framing**: A practical taxonomy distinguishes **open code** vs **open weights** vs **open training pipeline** (data + recipes + reproducibility), arguing teams must decide which layer they truly need ([TheTuringPost](https://twitter.com/TheTuringPost/status/2014630341349408928)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-TTS Model Release and Discussion

  - **[Qwen have open-sourced the full family of Qwen3-TTS: VoiceDesign, CustomVoice, and Base, 5 models (0.6B &amp; 1.8B), Support for 10 languages](https://www.reddit.com/r/LocalLLaMA/comments/1qjul5t/qwen_have_opensourced_the_full_family_of_qwen3tts/)** (Activity: 880): ****Qwen** has open-sourced the full family of **Qwen3-TTS** models, which includes VoiceDesign, CustomVoice, and Base models, with sizes of `0.6B` and `1.8B` parameters. These models support `10 languages` and are designed for tasks such as Voice Clone, Voice Design, and Custom Voice. The image provides a comparison chart of these models against others like MiniMax and SeedTTS, highlighting their performance across various metrics, where lower values indicate better performance. The models are available on [GitHub](https://github.com/QwenLM/Qwen3-TTS) and [Hugging Face](https://huggingface.co/collections/Qwen/qwen3-tts), with a [blog post](https://qwen.ai/blog?id=qwen3tts-0115) and [paper](https://github.com/QwenLM/Qwen3-TTS/blob/main/assets/Qwen3_TTS.pdf) detailing their capabilities.** Commenters appreciate the open-source release but express concerns about the models' dependency on Python and Nvidia GPUs, suggesting a need for support in other languages and platforms like llama.cpp or mistral.rs for broader accessibility.

    - **FullstackSensei** raises a technical concern about the current limitations of running Qwen models, highlighting the need for support in environments like `llama.cpp` or `mistral.rs` that can leverage GPU inference beyond just CUDA. This is particularly relevant given the rising costs of hardware and the desire for more accessible deployment options beyond Python and Nvidia GPUs.
    - **LetterRip** comments on the English voice outputs of Qwen3-TTS, noting that they seem to be influenced by Japanese Anime dubs. This suggests a potential bias in the training data, which could affect the naturalness and authenticity of the generated voices in English, especially if the training set was not diverse enough.
    - **silenceimpaired** discusses the performance of Qwen3-TTS, noting that while the samples are impressive, there is a concern about the frequency of certain outputs. This implies that while the model can produce high-quality audio, there might be consistency issues that need addressing to ensure reliable performance across different use cases.

  - **[Qwen dev on Twitter!!](https://www.reddit.com/r/LocalLLaMA/comments/1qjtyw8/qwen_dev_on_twitter/)** (Activity: 833): **The image is a Twitter post by **Chen Cheng**, announcing a new model with the tagline "Tiny model. Big personality" and a countdown, indicating an imminent release. The comments suggest that this might be related to a TTS (Text-to-Speech) model from a previous vLLM leak, with a link to a [Hugging Face collection](https://huggingface.co/collections/Qwen/qwen3-tts) that might be relevant. This suggests a new development in the field of TTS models, potentially offering significant improvements or features.** One comment humorously suggests that the new model might finally justify the investment in a high-end GPU like the `5090`, indicating high expectations for the model's performance.

    - ThePixelHunter discusses the current landscape of model sizes, noting that while smaller models are more accessible for local training on single GPUs, there is a lack of competition in the 50-120 billion parameter range. This range is ideal for enthusiasts with multiple high-end GPUs, such as a couple of 3090s or three 16GB cards, suggesting a gap in the market for larger, yet still locally trainable models.


### 2. Local LLM Development and Hardware Considerations

  - **[I gave my local LLM pipeline a brain - now it thinks before it speaks](https://www.reddit.com/r/LocalLLM/comments/1qkudvz/i_gave_my_local_llm_pipeline_a_brain_now_it/)** (Activity: 65): **The post discusses a significant update to a local LLM pipeline named Jarvis, soon to be called TRION, which now incorporates a self-developed Sequential Thinking MCP (Multi-Component Processor). This system, built with **Ollama**, **DeepSeek-R1**, and custom MCP servers, allows the AI to "think out loud" by breaking down complex questions into step-by-step reasoning, significantly reducing hallucinations. The AI dynamically decides when to use this deep thinking approach, providing instant answers for simple questions and detailed reasoning for complex ones. The project leverages a CIM (Causal Intelligence Module) framework developed by **u/frank_brsrk**. The implementation is open-source and available on [GitHub](https://github.com/danny094/Jarvis/tree/main).** Commenters appreciate the open-source nature of the project and express interest in experimenting with it. There is a sentiment that local LLMs will become more important as reliance on centralized AI providers diminishes.

    - GCoderDCoder discusses the integration of local LLMs with tools like 'roo code', 'vibe kanban', and 'MCPs' to automate workflows and reduce manual coding efforts. They highlight the importance of local LLMs in the context of increasing reliance on AI, contrasting it with commercial solutions like Anthropic's offerings. This reflects a broader trend towards developing independent, open-source AI solutions to maintain control and flexibility.
    - burn-n-die inquires about the system configuration used for running the local LLM pipeline. This is a critical aspect for technical readers interested in replicating or understanding the performance and scalability of such a setup. Details on hardware specifications, software environment, and any optimizations would be valuable for those looking to implement similar systems.

  - **[Someone is selling a Lamda Labs workstation with 4× RTX 2080 Ti =&gt; 4 x 11GB =&gt; 44GB VRAM. Is this machine well-supported by open source models? Is it fast enough?](https://www.reddit.com/r/LocalLLM/comments/1qjlzqt/someone_is_selling_a_lamda_labs_workstation_with/)** (Activity: 107): **A Lambda Labs workstation with 4× RTX 2080 Ti GPUs, totaling `44GB VRAM`, is being considered for purchase at `$2000`. This setup is generally well-supported by open-source machine learning frameworks, and the `44GB VRAM` is sufficient for most tasks. However, setting up the system properly could be challenging. An alternative suggestion is to build a 2x RTX 3090 rig, which might offer better performance and cost around `$2.5k`. The workstation is deemed capable of handling many open-weight LLM models, especially if it includes at least `32GB of system RAM`. The machine is suitable for exploring a wide range of machine learning projects, although not all require extensive VRAM.** There is a debate on whether to purchase the existing setup or build a new one with newer GPUs like the RTX 3090. Some argue that the existing setup is sufficient for learning and exploring ML models, while others suggest building a new rig for better performance and learning experience.

    - The workstation with 4× RTX 2080 Ti GPUs, totaling 44GB VRAM, is generally well-supported by most open-source frameworks. However, setting it up correctly can be challenging. The system's capability is sufficient for exploring many open-weight LLM models, especially if it includes at least 32GB of system RAM, which enhances its potential for various machine learning tasks.
    - A technical consideration is that Turing generation GPUs, like the RTX 2080 Ti, were initially thought to be incompatible with FlashAttention 2. However, recent updates indicate that this limitation has been addressed, as noted in the [FlashAttention Triton GitHub repository](https://github.com/egaoharu-kensei/flash-attention-triton). This expands the utility of the workstation for certain advanced ML tasks.
    - While the 4× RTX 2080 Ti setup is capable, some suggest that building a new system with 2× RTX 3090 GPUs might offer better performance and value. The RTX 3090 provides more VRAM per card and improved performance, potentially making it a more future-proof investment for machine learning projects.

  - **[OpenAI CFO hinting at "Outcome-Based Pricing" (aka royalties on your work)? Makes the case for local even stronger.](https://www.reddit.com/r/LocalLLaMA/comments/1qkiylw/openai_cfo_hinting_at_outcomebased_pricing_aka/)** (Activity: 419): ****OpenAI's CFO, Sarah Friar**, discussed a potential shift towards "outcome-based pricing" for large enterprise deals, particularly in high-value industries like pharmaceuticals. This model would involve OpenAI taking a share of the value created by their AI, such as a cut from a pharmaceutical company's profits if AI contributes to a major discovery. This approach is not intended for regular users or indie developers, and the initial reports suggesting a broader application were misleading. The concept raises discussions about the benefits of local AI deployment versus reliance on cloud-based services, drawing parallels to the energy sector's grid versus solar power debate.** Commenters highlight skepticism about OpenAI's potential pricing model, comparing it to the lack of royalties paid to data creators used in AI training. The analogy of local AI deployment to solar power is appreciated, emphasizing control over infrastructure to avoid future costs tied to value-based pricing.

    - The discussion highlights concerns about OpenAI's potential shift to 'Outcome-Based Pricing', which could involve royalties based on the revenue generated from using their models. This is compared to the current model where users pay based on usage, akin to how electricity is billed. The analogy suggests that such a pricing model could drive users to consider local or self-hosted solutions, especially as OpenAI's profitability grows and they seek higher profits.
    - The comment by WeMetOnTheMountain critiques the efficiency of OpenAI's models, suggesting that they consume a large number of tokens to maintain performance, which results in slower processing speeds. The commenter argues that alternative models like GLM or mini Max could potentially offer better results when implemented in a 'one loop dialectical circuit', indicating a preference for more efficient, possibly self-hosted, solutions.
    - Winter_Educator_2496 emphasizes the need for open-source alternatives to OpenAI's models, which could be hosted in the cloud but also switched to local hosting if necessary. This reflects a broader sentiment for more control and flexibility over AI tools, especially in light of potential pricing changes and the desire to avoid dependency on a single provider.


### 3. Hugging Face Model Releases and Trends

  - **[This Week's Hottest Hugging Face Releases: Top Picks by Category!](https://www.reddit.com/r/LocalLLM/comments/1qjqhja/this_weeks_hottest_hugging_face_releases_top/)** (Activity: 49): ****Hugging Face** has released several trending models across different categories this week. In text generation, the `zai-org/GLM-4.7-Flash` model, with `31B` parameters, is designed for fast and efficient text generation, boasting `124k` downloads. Its quantized counterpart, `unsloth/GLM-4.7-Flash-GGUF`, offers a `30B` parameter model optimized for local inference with `112k` downloads. In the image/multimodal category, `zai-org/GLM-Image` and `google/translategemma-4b-it` are notable for their capabilities in creative edits and multilingual tasks, respectively. For audio/speech, `kyutai/pocket-tts` and `microsoft/VibeVoice-ASR` provide compact TTS and multilingual ASR solutions. Other notable releases include `Lightricks/LTX-2` for image-to-video conversion and `stepfun-ai/Step3-VL-10B` for advanced reasoning in image-text-to-text tasks.** A technical debate has emerged regarding the performance of the `GLM-4.7 30B-A3B` model compared to the `Qwen3-Coder 30B-A3B` for programming tasks, with some users finding the latter superior.

    - A user compared the GLM-4.7 30B-A3B model to the Qwen3-Coder 30B-A3B model, noting that the latter performs better for programming tasks. This suggests that Qwen3-Coder may have optimizations or architectural advantages that make it more suitable for code-related applications. Further benchmarks or detailed evaluations would be needed to substantiate this claim and understand the specific areas where Qwen3-Coder excels.

  - **[Good local LLM for coding?](https://www.reddit.com/r/LocalLLM/comments/1qk9ked/good_local_llm_for_coding/)** (Activity: 62): **The user is seeking a local LLM for coding that can run on an `rx 6750 xt` GPU with `12GB` VRAM, considering models like **GLM 4.7 flash**. However, concerns about VRAM limitations suggest that `30B` parameter models, even when quantized to `q4`, may exceed the GPU's capacity. Recommendations include models like [VisCoder2-7B](https://huggingface.co/TIGER-Lab/VisCoder2-7B), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), and [NousCoder-14B](https://huggingface.co/NousResearch/NousCoder-14B), with **gpt-oss-20b** noted for its speed and reliability despite being heavily censored. It's suggested to use models under `10B` parameters or employ a coding MoE model with `llama.cpp` to offload some processing to system RAM.** There is a debate on the suitability of `30B` models for the user's GPU, with a consensus leaning towards using models under `10B` parameters due to VRAM constraints. The use of `llama.cpp` for offloading to system RAM is also discussed as a viable strategy.

    - Javanese1999 highlights several models for local coding tasks, including [VisCoder2-7B](https://huggingface.co/TIGER-Lab/VisCoder2-7B), which is described as a better version of Qwen2.5-Coder-7B-Instruct, and [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), noted for its speed even when exceeding VRAM capacity. The commenter prefers gpt-oss-20b for its reliability in light coding tasks despite its censorship in refusal prompts.
    - Used_Chipmunk1512 advises against using 30B models quantized to q4 due to GPU limitations, suggesting that models under 10B are more suitable for most users. This highlights the importance of considering hardware constraints when selecting a local LLM for coding.
    - RnRau suggests using a coding Mixture of Experts (MoE) model with the `llama.cpp` inference engine to offload some of the model's processing to system RAM, which can be a practical approach for handling larger models without overwhelming the GPU.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. OpenAI and Anthropic Developments

  - **[OpenAI says Codex usage grew 20× in 5 months, helping add ~$1B in annualized API revenue last month](https://www.reddit.com/r/singularity/comments/1qk6pbi/openai_says_codex_usage_grew_20_in_5_months/)** (Activity: 535): **OpenAI's Codex usage has surged 20 times over five months, contributing to an additional `$1 billion` in annualized API revenue, as reported by **Sarah Friar**, OpenAI's CFO. The company is experiencing a shift towards enterprise customers, with the revenue split moving from `70% consumer and 30% enterprise` to `60% consumer and 40% enterprise`, and is expected to reach a `50-50` balance by the end of the year. OpenAI aims to achieve `$20 billion` in annualized revenue by 2025, supported by cloud investments and infrastructure scaling.** A comment suggests skepticism about the profitability, estimating a cost of `$7 billion` to achieve the `$1 billion` revenue. Another comment highlights a shift in AI tools used by a financial services company, indicating competition in the B2B market with **Anthropic** and **OpenAI**.

    - BetImaginary4945 suggests that the cost of generating $1B in revenue for OpenAI might be as high as $7B, implying a significant expenditure on infrastructure, research, and development to support such rapid growth. This raises questions about the sustainability and profitability of OpenAI's business model in the long term.
    - balagachchy shares an insight from their experience at a multinational financial services company, noting a shift from using ChatGPT to Gemini and Claude Code for software engineering tasks. This highlights the competitive landscape in AI tools for enterprise use, where companies are exploring different solutions to meet their specific needs.
    - imlaggingsobad comments on the competitive dynamics between OpenAI and Anthropic in the B2B market, suggesting that while Anthropic is perceived as a leader, OpenAI's rapid growth and innovation could still make it a formidable competitor. This underscores the ongoing competition and potential for shifts in market leadership.

  - **[OpenAI CEO meets Middle East investors over potential $50B fundraising](https://www.reddit.com/r/OpenAI/comments/1qjrpbq/openai_ceo_meets_middle_east_investors_over/)** (Activity: 191): ****OpenAI** is reportedly in discussions with Middle Eastern sovereign wealth funds to raise a potential `$50 billion` in a new funding round, as confirmed by [CNBC](https://www.cnbc.com). The talks are still in preliminary stages, with no term sheets signed yet. **Sam Altman**, OpenAI's CEO, is currently in the UAE to engage in these discussions, highlighting the strategic importance of this potential investment for OpenAI's future growth and operational scaling.** A notable opinion from the comments suggests skepticism about OpenAI's financial strategy, questioning why the company isn't pursuing an IPO given its significant revenue, and criticizing its reliance on external capital to manage high operational costs.

    - AtraVenator highlights concerns about OpenAI's financial strategy, noting that despite having over `$20B` in annual revenue, the company is seeking additional external capital rather than moving towards a self-sustaining model. This raises questions about their high compute costs and reliance on external funding to cover these expenses.
    - The discussion touches on the potential risks of OpenAI going public, with NotABCDinFL suggesting that an IPO could lead to a 'massive rug pull' where institutional investors might cash out, leaving retail investors at a disadvantage. This reflects concerns about the stability and transparency of OpenAI's financial practices.
    - There is skepticism about OpenAI's leadership and strategic direction, with BeingComfortablyDumb questioning how the company has moved from having a first-mover advantage and significant market share to its current financial challenges. This implies a critique of the management's ability to capitalize on their early lead in the AI industry.

  - **[Anthropic's Claude Constitution is surreal](https://www.reddit.com/r/OpenAI/comments/1qjytb2/anthropics_claude_constitution_is_surreal/)** (Activity: 611): **The image discusses the use of the pronoun "it" for Anthropic's AI, Claude, and the potential for Claude to develop preferences for different pronouns, suggesting the emergence of functional versions of emotions or feelings from training on human-generated data. This is not a deliberate design choice by Anthropic, but it raises questions about the moral status of these emotional states. The text reflects ongoing debates in AI ethics about the implications of AI systems potentially developing human-like emotional states, which are not yet fully understood or intentionally designed.** Commenters note that this aligns with current research and extreme safety measures in the AI industry, emphasizing the surreal nature of these developments and the importance of humility in AI labs' claims.

    - br_k_nt_eth highlights that the Claude Constitution aligns with current research trends and extreme safety measures being tested in the AI industry, which sometimes negatively impact company reputations. This suggests that those familiar with advanced models would not find the approach surprising, as it reflects ongoing industry practices.
    - heavy-minium argues that the surreal aspect of Claude's Constitution is not unique to Claude but is inherent to any large language model (LLM). They point out that emotions are patterns in training data, and this phenomenon is unavoidable unless the model is either broken or too small. The commenter suggests that the relabeling of this characteristic is more about public relations than a technical breakthrough.
    - laystitcher emphasizes the importance of the cautious language used in the Claude Constitution, such as the word 'may,' which reflects the humility of AI labs in acknowledging the uncertainties in their developments. This cautious approach is seen as appropriate given the current surreal advancements in AI technology.

  - **[Microsoft is using Claude Code internally while selling you Copilot](https://www.reddit.com/r/ClaudeAI/comments/1qk4up5/microsoft_is_using_claude_code_internally_while/)** (Activity: 1276): ****Microsoft** is internally using **Claude Code** across various divisions like Windows and Teams, despite heavily investing in **OpenAI** and promoting **Copilot**. This internal use is sanctioned for all Microsoft repositories, indicating a significant investment of `$500M/year` with **Anthropic**. Interestingly, **Azure** sales teams receive quota credit for Anthropic sales, suggesting a strategic partnership. Despite **Claude Code** not outperforming in `95%` of benchmarks, developers report superior problem-solving capabilities, challenging the reliability of current benchmark tools. **Copilot** is priced at `$10/month/user`, whereas **Claude Code** is `$150` for enterprise use.** Commenters highlight the discrepancy between benchmark results and real-world performance, suggesting benchmarks may not fully capture tool effectiveness. The partnership between Microsoft and Anthropic is seen as strategic, with Claude's integration into various Microsoft products and services.

    - CurveSudden1104 highlights a discrepancy between benchmark results and real-world performance, noting that while Claude doesn't outperform in 95% of benchmarks, developers find it superior in problem-solving. This suggests that current benchmarks may not accurately reflect practical utility, indicating a potential gap between quantitative metrics and qualitative user experience.
    - morrisjr1989 points out that Claude's integration into Microsoft's ecosystem is part of a strategic partnership, with Claude being utilized in various Microsoft products like Copilot, Foundry, and Azure-hosted services. This integration underscores a collaborative approach rather than a competitive one, leveraging Claude's capabilities across multiple platforms.
    - UnknownEssence provides a cost comparison, noting that Copilot is priced at $10 per month per user, whereas Claude Code is significantly more expensive at $150 for enterprise use. This price difference highlights the distinct market positioning and target audiences for each product, with Copilot being more accessible to individual users and Claude Code catering to enterprise needs.

  - **[Claude’s eureka moment is not ending soon it looks like](https://www.reddit.com/r/ClaudeAI/comments/1qjlrgb/claudes_eureka_moment_is_not_ending_soon_it_looks/)** (Activity: 1377): **The image and post discuss the competitive landscape of AI coding agents, particularly focusing on **Claude**, a tool developed by **Anthropic**. The post suggests that **Gemini** has open-sourced their CLI in an attempt to compete with Claude, which is notably used by **Nvidia**. This highlights the ongoing race in AI development tools, with speculation about whether the market will consolidate around a few dominant players or remain diverse. The comments reflect a belief that AI will significantly transform programming, with some users noting their companies' exclusive use of Claude.** One comment suggests skepticism about the CEO's investment in the product's company, while another highlights a shift in programming paradigms, predicting that future programmers will rely heavily on AI tools.

    - sine120 argues that Claude Code should be open-sourced, suggesting it lacks unique features that justify keeping it proprietary. They mention that other frameworks like Opus could integrate Claude's capabilities, and by not open-sourcing, Anthropic might miss the chance to lead AI development, potentially allowing competitors like Google and Chinese labs to catch up. They emphasize that developers might prefer openness over marginal performance improvements.
    - itsdr00 highlights a significant shift in software development life cycles (SDLC) due to AI advancements, particularly with Claude Code. They note that some companies are restructuring their SDLC to leverage AI, implying that traditional methods are becoming obsolete. This reflects a broader industry trend where AI is increasingly integral to development processes, akin to a paradigm shift from older technologies like punch cards.


### 2. Gemini and AI Studio Issues

  - **[I’m honestly sick of this: Gemini Web vs AI Studio Context Window Mess](https://www.reddit.com/r/Bard/comments/1qkj31m/im_honestly_sick_of_this_gemini_web_vs_ai_studio/)** (Activity: 49): **The user reports a significant regression in the Gemini web/app's ability to handle large files since the update to **Gemini 3**. Previously, with **Gemini 2.5 Pro**, files containing `600k–800k` tokens could be processed without issues, retaining full context for queries. However, the current version rejects files over `100k` tokens and provides incomplete or incorrect responses. In contrast, **Gemini AI Studio** continues to handle the same large files effectively, suggesting the underlying model's capability remains intact but is not accessible in the consumer-facing app. This discrepancy raises concerns about potential limitations imposed on the web/app version, possibly misleading users about the product's capabilities.** Commenters express dissatisfaction with the Gemini web/app, noting that **AI Studio** is the only reliable platform for using Google's models effectively. Some users, even on the Pro plan, report receiving errors when uploading large documents, indicating a possible mismatch between advertised capabilities and actual performance.

    - A user mentions that AI Studio is the only viable platform for using Google models effectively, implying that other platforms like Gemini app and Antigravity do not meet their expectations despite having a subscription. This suggests potential issues with the usability or performance of these platforms compared to AI Studio.
    - Another user discusses the Pro plan, noting that they have not encountered issues with document processing. They suggest that if documents are too large in terms of tokens, the system might default to classic retrieval methods rather than processing the entire file, indicating a possible limitation in handling large documents.
    - A user on the Pro plan reports receiving an error after uploading a 20-page PDF, describing the situation as 'absurd.' This highlights potential limitations or bugs in the system's ability to handle larger documents, even for users on higher-tier plans.

  - **[AI Studio Rate Limits are out of control again...](https://www.reddit.com/r/Bard/comments/1qkztjy/ai_studio_rate_limits_are_out_of_control_again/)** (Activity: 67): **The post discusses recent issues with rate limits on **AI Studio**, where users, including those with Pro subscriptions, are experiencing frequent request denials. This is a change from previous usage patterns where limits were rarely hit. The user expresses frustration that their Pro subscription cannot be applied to AI Studio, which they find superior to the main site. Technical comments suggest that the rate limits might be due to dynamic prompt limits, increased GPU allocation for new training, or a higher user count. Additionally, **Gemini 2.5 Pro** has been rate limited for the first time, indicating possible resource constraints or strategic adjustments by the platform.** Commenters speculate that the rate limits could be due to increased demand or resource reallocation, with some suggesting desperation on the platform's part. Others report encountering internal errors, indicating potential technical issues beyond just rate limiting.

    - OneMisterSir101 suggests that the current rate limits on AI Studio might be due to dynamic prompt limits, which could be influenced by either GPU delegation to new training tasks or an increase in user count. This implies a resource allocation issue where computational resources are being stretched thin, potentially affecting performance and availability.
    - Undertaker1995 notes that Gemini 2.5 Pro has been rate limited for the first time, indicating a significant shift in resource management or demand. This could reflect a strategic decision by the platform to manage load or a response to increased usage, highlighting potential scalability challenges.
    - wildwriting reports encountering an 'internal error' message, despite attempting standard troubleshooting steps like reloading the page and restarting the browser. This suggests a deeper technical issue within the platform, possibly related to server-side problems or misconfigurations that are not resolved by client-side actions.

  - **[I'm sorry but Gemini is getting worse and worse](https://www.reddit.com/r/GeminiAI/comments/1qjrokj/im_sorry_but_gemini_is_getting_worse_and_worse/)** (Activity: 1301): **The post discusses a decline in the performance of **Gemini**, particularly in its memory capabilities and intelligence. Previously, the pro mode of Gemini could remember `30+ conversations` with a total of `180,000 words`, but recent updates have halved this memory capacity, leading to a perceived decrease in intelligence and reliability. The user expresses frustration, suggesting that **ChatGPT** might be a better alternative due to its longer and more conversational responses.** Commenters agree with the decline in Gemini's performance, noting increased issues with speculation and hallucination. There is skepticism about future updates, with one commenter cynically suggesting that any improvements will be short-lived.

    - The comment by Particular-Battle315 highlights a common lifecycle pattern in AI models where initial releases are powerful but get 'nerfed' over time. This is observed across companies like Anthropic, OpenAI, and Google, suggesting a strategic approach to model updates that may not be immediately apparent to all users.
    - Duchess430 discusses the potential for running large AI models on personal computers using specialized open-source models, which may outperform Gemini for specific tasks. They mention the GGUF (GPT-Generated Unified Format) as a method to optimize resource usage by splitting data between RAM and VRAM, allowing for running large models without high-end hardware.
    - rephil3 points out issues with Gemini, specifically its tendency to speculate and hallucinate, which are common problems in AI models that can affect their reliability and user trust.

  - **[Gemini about to get busy?](https://www.reddit.com/r/Bard/comments/1qk2yx6/gemini_about_to_get_busy/)** (Activity: 33): **The post discusses the potential impact of **ChatGPT** introducing ads on its user base, suggesting that this could lead to a significant migration of users to **Gemini**, especially as Gemini's models improve and integrate more deeply with **Google's** ecosystem. Concerns are raised about whether Gemini can handle a sudden influx of users without degrading the experience for its current base. A technical issue noted is Gemini's handling of conversations, where users report chats being replaced with 'sensitive query' messages, and the lack of a 'Projects' feature to maintain context, unlike **ChatGPT** and **Claude**.** Commenters debate Gemini's readiness to handle increased user load, with some arguing that **Google's** extensive experience and infrastructure, including recent investments in clean energy and data centers, position it well to scale effectively. Others highlight the technical shortcomings of Gemini, such as conversation management issues, as potential drawbacks.

    - Loud-Independent9041 highlights a significant issue with Gemini's conversation handling, where chats are sometimes replaced with 'a sensitive query' messages, disrupting the user experience. This contrasts with ChatGPT and Claude, which offer a 'Projects' feature to maintain context across conversations, a feature Gemini lacks, impacting its usability for continuous dialogue.
    - rollk1 points out Google's strategic positioning in the AI and data center space, emphasizing their acquisition of Intersect Power to support clean energy for data centers. This move, along with their existing Google Cloud infrastructure, positions them advantageously for scaling AI models, potentially outpacing competitors like OpenAI.
    - FalseAcadia4306 notes a potential increase in Gemini's user base, as evidenced by receiving a 'research queued' message for the first time, suggesting a surge in demand or usage that could be straining the system's capacity.


### 3. DeepSeek and Baidu's ERNIE 5.0 Innovations

  - **[DeepSeek-V3.2 Matches GPT-5 at 10x Lower Cost | Introl Blog](https://www.reddit.com/r/DeepSeek/comments/1qkoc53/deepseekv32_matches_gpt5_at_10x_lower_cost_introl/)** (Activity: 125): ****DeepSeek-V3.2** is an open-source AI model that reportedly matches the performance of **GPT-5** in mathematical reasoning tasks while operating at a cost 10 times lower, specifically `$0.028` per million tokens. The model utilizes a novel 'Sparse Attention' architecture, which contributes to its efficiency, achieving frontier-class performance with a total training cost of approximately `$5.5 million`, significantly less than the `$100M+` typically spent by major US tech companies. The model's architecture includes **DeepSeek Sparse Attention (DSA)** for efficient long-context processing and a refined **Mixture-of-Experts** approach, which activates only a subset of parameters per token, enhancing task-specific performance. For more details, see the [Introl Blog](https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage).** One comment suggests skepticism about the reported cost savings, noting that a significant portion of OpenAI's expenses may be attributed to executive salaries rather than direct model development costs.


  - **[Baidu's new ERNIE 5.0 is going hard after GPT and Gemini](https://www.reddit.com/r/DeepSeek/comments/1qkpxzm/baidus_new_ernie_50_is_going_hard_after_gpt_and/)** (Activity: 51): ****Baidu's ERNIE 5.0** is making significant strides in mathematical reasoning and technical problem-solving, ranking #2 globally on the LMArena Math leaderboard, just behind the unreleased GPT-5.2-High. It surpasses GPT-5.1 and Gemini 2.5 Pro in math and scores higher on specialized benchmarks like MathVista and ChartQA, particularly excelling in interpreting complex visual diagrams. In the 'VLMs Are Blind' benchmark, ERNIE 5.0 scored `77.3`, outperforming GPT-5-High's `69.6`. Additionally, ERNIE 5.0 offers a cost advantage, being nearly `90%` cheaper than OpenAI’s GPT-5.1 for similar token volumes, making it a competitive option in terms of pricing.**

    - ERNIE 5.0 is noted for its impressive scale with `2.4 trillion parameters`, significantly larger than competitors like DeepSeek's `671 billion` and Kimi K2's `1 trillion`. Despite its size, the quality of output is reported to be similar to other models, with particularly fast inference speeds. However, the model's strict system prompt alignments can make interactions feel restricted, though users can adjust the tone with specific prompts for better results.
    - The model offers a free web version with a `128k context window`, comparable to DeepSeek, which is a significant advantage for users needing extensive context handling. However, the default interaction tone is described as overly corporate, which can be modified with specific prompts to achieve more engaging interactions. This flexibility in tone adjustment is seen as a positive feature despite the initial restrictions.
    - A recent update to ERNIE 5.0, referred to as "5.0 Preview 1203", has reportedly improved the model's engagement and interaction quality, making it more fun and collaborative. This suggests that Baidu is actively iterating on the model to enhance user experience, potentially addressing earlier criticisms of restrictive interactions.

  - **[DeepSeek’s Quiet Technical Wins (That Nobody Talks About)](https://www.reddit.com/r/DeepSeek/comments/1qjob34/deepseeks_quiet_technical_wins_that_nobody_talks/)** (Activity: 85): ****DeepSeek** is recognized not only for its benchmark performance but also for its engineering innovations, which include *better routing for efficiency*, *cleaner long-context behavior*, and *faster token generation*. These features contribute to its distinctiveness in practical applications. Notably, DeepSeek employs **Mixture of Experts (MoE)** for smarter routing and introduces **Engram** to separate memory from reasoning, emphasizing architectural innovation over brute-force scaling.** Commenters highlight DeepSeek's unique 'thinking process' and its focus on architectural innovation, such as using MoE and Engram, as key differentiators from other AI models.

    - **Hey-Intent** highlights DeepSeek's architectural innovations, particularly the use of Mixture of Experts (MoE) for smarter routing and the introduction of Engram to separate memory from reasoning. This approach emphasizes sustainable AI progress through architectural improvements rather than brute-force scaling, which is a significant shift in AI development strategy.
    - **Fine_Effective4980** points out that DeepSeek's system-level efficiency, combining routing and token generation, results in a more responsive and stable user experience, especially with longer context. This efficiency is not captured in traditional benchmarks but is crucial for real-world applications, offering a smoother and more reliable workflow.
    - **Althalvas** notes that DeepSeek's R1 model provides a superior thinking process compared to other AI models, even when only using free versions. This suggests that DeepSeek's models may have a more refined approach to processing, which could be attributed to their architectural choices.


---

# AI Discord Recap

> A summary of Summaries of Summaries


## Gemini 3.0 Pro Preview Nov-18

**Theme 1. Hardware Limits and Kernel Hacking: B200s, ROCm, and Mobile Optimization**

- **FlashAttention-4 hits 71% utilization on B200**: Early benchmarks show **FlashAttention-4** reaching **1,605 TFLOPS/s** on an **NVIDIA B200 GPU** using BF16 inputs, capturing roughly 71% of the theoretical maximum. Engineers in the **GPU MODE** discord noted a lack of official documentation regarding specific fp4/fp8/fp16 specs, sparking debate over the hardware's true theoretical ceiling compared to leaked materials.
- **Developer abandons ROCm for CUDA**: A frustrated developer publicly documented their switch from **AMD's ROCm** to **NVIDIA** after purchasing a 5090, citing packaging failures, build issues, and a "hostile" ecosystem for consumer-facing hardware. The discussion highlighted that mid-grade NVIDIA hardware often outperforms AMD gear on specific kernels like **Conv3D** due to software maturity, referencing a [Reddit thread on performance regressions](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/).
- **Mobile GPU memory path splitting**: Engineers in **tinygrad** discovered that optimizing **L2 bandwidth** on mobile GPUs requires treating **textures** and **buffers** as distinct hardware pathways. Maximizing throughput involves strategically feeding one input as a texture and another as a buffer to saturate the available bandwidth, a technique critical for edge inference.

**Theme 2. Agentic Workflows: Cursor Sub-agents, Replit Control, and Aider TUI**

- **Cursor 2.4 wobbles while Sub-agents emerge**: While users reported **Composer 1** breaking into endless loops and **Cursor 2.4** causing significant lag on high-end PCs, savvy users found evidence of a **sub-agent** feature rollout. The system injects a **<subagent_delegation_context>** prompt to enable parallel task execution and better context handling, as detailed in the [Changelog video](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4?ex=69752b85&is=6973da05&hm=50e3cbf6432112dcbe36b0315b1645fd7d856c9d2ead97e639b2d8abcfa5b8f4&).
- **Replit Agent gets real-time brains**: Zhen Li published a technical breakdown of **Decision-Time Guidance** in the **Replit Agent**, replacing static rules with real-time control mechanisms for complex navigation. This architectural shift aims to reduce fragility in autonomous coding tasks, moving closer to adaptive "system 2" thinking, as described in [this blog post](https://xcancel.com/zhenthebuilder/status/2014393451442581688?s=46).
- **Aider eyes a TUI makeover**: The **aider** community is actively designing a **Terminal User Interface (TUI)** to allow message editing while browsing replies and rendering **Mermaid diagrams** directly in the terminal. Simultaneously, users are chaining **aider** for rapid context management with **Claude Code** for complex debugging to minimize token costs and leverage [aider's efficient file search](https://discord.com/channels/1131200896827654144/1131200896827654149/1464167385060872203).

**Theme 3. Model Architecture and Audio: Qwen3-TTS, NanoGPT Hacks, and GLM Speedups**

- **Qwen3-TTS clones voices at scale**: Alibaba released the **Qwen3-TTS** family, ranging from **0.6B to 1.8B** parameters, capable of high-quality voice cloning and supporting 10 languages. The release challenges proprietary models like ElevenLabs, with demos and weights available on [Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS).
- **NanoGPT gets a Difference Layer boost**: Researchers in **Eleuther** reported that replacing the **QKV linear layer** with a *difference layer*—`x = (self.a2(x) - self.b2(x)) * ...`—significantly improved **NanoGPT** performance on simple tasks. Others noted that switching activations from **GELU** to **SwiGLU** also provided a baseline boost, emphasizing the need for [stronger baselines](https://github.com/Eternalyze0/difference_layer) before claiming architectural supremacy.
- **GLM-4.7 Flash zooms on llama.cpp**: The **Hugging Face** community noted that **llama.cpp** updates have accelerated **GLM-4.7 Flash GGUF** inference by approximately **1.5x**. Users are advised to rebuild from source and grab fixed quants from [Unsloth's repo](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) to enable the `--flash-attn on` flag for optimal performance.

**Theme 4. Inference Engineering: Speculative Decoding, SRAM Specs, and MoE Memory**

- **Speculative decoding slows down vLLM**: Engineers debugging **Qwen3-VL** on **vLLM** found that enabling speculative decoding often hurts **Time To First Token (TTFT)** unless batch sizes are massive. The consensus is that small draft models introduce too much overhead for single-stream or low-batch inference, and standard [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics/) via Grafana are recommended for tuning.
- **Small MoEs starve in 8GB RAM**: Discussions in **Unsloth AI** concluded that running **Mixture of Experts (MoE)** models on 8GB RAM is largely futile because the active parameter count becomes too low to be useful. While **Qwen 2.5 3B** (dense) remains the king for low-memory coding, "small" MoEs like **LFM2** lack the corpus density to compete effectively.
- **Cerebras CS3 packs 41GB SRAM**: It was revealed in **OpenRouter** that each **Cerebras CS3** wafer-scale instance houses **41GB of SRAM**, designed to interconnect up to **2048 instances**. This massive on-chip memory allows for extremely high-bandwidth model execution, bypassing traditional HBM bottlenecks found in GPU clusters.

**Theme 5. Adversarial Attacks and Platform Instability**

- **Gemini 3 Pro jailbroken via ENI**: The **ENI jailbreak** technique, originally targeting Claude, was successfully ported to **Gemini 3 Pro** in AI Studio, with users reporting it "works like magic" even on the Flash variant. The exploit allows bypassing safety guardrails, detailed in a shared [GitHub methodology](https://github.com/pranrichh/Jailbreaks/blob/main/GEMINI-CLAUDE%20JAILBREAK.md).
- **Perplexity Pro limits pinch users**: **Perplexity** subscribers are reporting severe undocumented limits, with file uploads capped at three per day and research queries throttled to as low as **20 daily** for some (vs the expected 600). Users suspect aggressive **A/B testing** or financial tightening, further aggravated by [API 401 errors](https://discord.com/channels/1047197230748151888/1161802929053909012/1464341850809831519) despite valid credits.
- **Kimi AI hits capacity wall**: **Moonshot AI's Kimi** service is suffering widespread outages, with users facing constant "This mode is at capacity" errors and vanishing conversation histories. Speculation in the community points to a potential datacenter failure or API restrictions from upstream providers like Google [derailing the service](https://discord.com/channels/1369594130807787570/1371757564005711973/1464201099123888242).


## gpt-5.2


**1. Cursor 2.4 Subagents Rollout & Developer UX Fallout**

- **Subagents Ship Fast, Task Tool Ghosts Everyone**: **Cursor 2.4** introduced **subagents** for parallel task completion per the [Cursor Changelog](https://cursor.com/changelog) and a demo [video](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4), but users noticed injected **`<subagent_delegation_context>`** that tells the model to call a **Task tool that isn’t available**.
  - Community speculation pegged this as an **incomplete rollout** (prompting paths shipped ahead of backend), and some users suspected subagents silently **fall back to Composer 1**, worsening latency and “*planning next moves*” hangs.

- **Composer Crash Derby: Loops, Lag, and the Great Downgrade**: Users reported **Composer 1** as “completely broken,” including **endless chat loops** and crashes, with workarounds like downgrading to **Cursor 2.3** (notably on **macOS Big Sur 11.7.3**) and filing reports via the [Cursor bug forum](https://forum.cursor.com/c/support/bug-report/6).
  - Separately, **Cursor 2.4** drew complaints of severe **lag/unresponsiveness** and frequent crashes even on high-end machines, fueling criticism that releases feel **premature** and hard to trust for daily engineering work.

- **Billing Roulette: Token Counts, Auto Mode, and DIY Telemetry**: Cursor users flagged **usage/billing discrepancies** (missing dollar amounts, unexpected bonus credits, and limits not triggering despite heavy use), and some suspected **Auto mode** charges incorrectly.
  - To sanity-check spending, users recommended third-party tracking like [token-watch](https://token-watch.vercel.app/), noting it can diverge from what Cursor’s own dashboard shows.


**2. Inference Performance & Benchmarking: B200, vLLM, llama.cpp, Grafana**

- **FlashAttention-4 Floors It on B200 (Specs Still Foggy)**: In GPU performance chatter, **FlashAttention-4 (FA4)** reportedly hit **1,605 TFLOPS/s** (~**71%** of theoretical) on an **NVIDIA B200** with **BF16** inputs, while the community debated a theoretical ceiling around **2260 TFLOPS** and noted missing official datatype details.
  - Confusion deepened as leaked materials listed B200 at **10/5/2.5 TFLOPS** for **fp4/fp8/fp16**, and folks asked for an **official spec paper** to reconcile marketing numbers with kernel benchmarks.

- **Spec Decode Saves Throughput, Not Your TTFT (Usually)**: Engineers discussed optimizing **Time To First Token (TTFT)** for **Qwen3-VL** in **vLLM** on a **B200**, considering `--speculative_config` with smaller **Qwen3-VL-4B/2B** draft models.
  - The advice: speculative decoding often **hurts throughput** unless you run **high batch sizes**, and only “**eagle heads**” setups feel worth it at scale because small drafts add too much overhead for short outputs.

- **Grafana for VLM Telemetry + vLLM Metrics as the Source of Truth**: For benchmarking fast multimodal paths, members pointed to [vLLM’s metrics docs](https://docs.vllm.ai/en/stable/design/metrics/) and suggested wiring dashboards with [Grafana](https://grafana.com/products/cloud/metrics/) for **real-time TTFT visualization**.
  - The thread framed Grafana as the “good UI” layer for quickly comparing VLM deployments under realistic workloads instead of relying on one-off scripts.

- **llama.cpp Speeds Up GLM-4.7 Flash GGUFs ~1.5×**: **llama.cpp** delivered a **~1.5× speedup** plus bug fixes for **GLM 4.7 Flash GGUFs**, and users pointed to re-downloading fixed quants from [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF).
  - This reinforced the “rebuild often” culture in local inference stacks: performance jumps can appear just by updating the runtime and swapping quants, not changing the model.


**3. Open Releases: Voice/Audio Models, New Datasets, and Local-First LLMs**

- **Qwen3-TTS Drops Multilingual Voice Cloning (ElevenLabs Catching Side-Eye)**: Communities rallied around **Qwen3-TTS** as a strong **voice cloning** option, with a live demo on [Qwen3-TTS Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS) and the broader family referenced via [QwenLM GitHub](https://github.com/QwenLM) and [Qwen on Hugging Face](https://huggingface.co/Qwen).
  - Latent Space summarized the family as **0.6B–1.8B params** with support for **10 languages**, framing it as open tooling that can plausibly replace paid TTS in some pipelines.

- **Audio Release Triple-Feature: PersonaPlex-7B, TTS-1.5, and Chroma 1.0**: An audio-model roundup highlighted **NVIDIA PersonaPlex-7B** (full-duplex conversational), **Inworld AI TTS-1.5** (low-latency TTS), and **Flash Labs Chroma 1.0** (open-source end-to-end speech-to-speech) via [Lina Colucci’s post](https://x.com/lina_colucci/status/2014229002370834861).
  - The vibe: speech stacks are accelerating toward **low-latency** and **end-to-end** pipelines, and open releases are starting to cover pieces previously gated behind SaaS APIs.

- **Datasets & Local Models: Rust→WASM Synthetic + Faust-1 German-First**: Hugging Face saw two notable drops: a **Rust-to-WebAssembly synthetic dataset** of **1,000** generated programs at [webxos/wasm_synthetic_dataset](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset), and **Faust-1**, a **1.6B** German-first LLM at [tabularisai/Faust-1](https://huggingface.co/tabularisai/Faust-1).
  - Faust-1 emphasized **~90% German pretraining**, a **German-optimized tokenizer**, and instruction tuning with **DPO**, while the WASM dataset focused on **reproducibility** (deterministic Fibonacci-derived PRNG plus structural hashes).


**4. Agent Frameworks & Infra: Control Loops, RLM/DSPy, and MCP Schema Discipline**

- **Replit Agent Gets a Steering Wheel with Decision-Time Guidance**: A technical writeup on **Decision-Time Guidance** described how **Replit Agent** applies **real-time control mechanisms** instead of static rules, shared via [Zhen Li’s blog link](https://xcancel.com/zhenthebuilder/status/2014393451442581688).
  - The discussion framed this as a practical direction for agents: tighter **online steering** during execution rather than brittle pre-authored guardrails.

- **DSPy’s “Why” Gets Re-Explained (Signatures > Prompt Hacks)**: DSPy folks circulated an explainer arguing DSPy’s value comes from **signature & module abstractions**, not just prompt tuning, via [“DSPy: the most misunderstood agent”](https://eito.substack.com/p/dspy-the-most-misunderstood-agent) and the companion [X post](https://x.com/Eito_Miyamura/status/2014757193766093069).
  - Separately, members discussed tuning **RLM prompts** and even optimizing JSON-schema adapters (GEPA ideas), aiming to make structured outputs more reliable without bloating token budgets.

- **MCP Schema Cage Match: `additionalProperties` vs `anyOf`**: Model Context Protocol contributors questioned whether **`GetTaskPayloadResult`** is too permissive because it allows **`additionalProperties`**, pointing directly to the schema location in the MCP repo ([schema.json lines 1245–1256](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256)).
  - The proposed fix was moving toward **`anyOf`** for stricter validation, reflecting a broader push to keep agent-tool payloads **tightly typed** to avoid “accept everything” integrations that break downstream.


**5. Platforms, Benchmarks, and the AMD/NVIDIA Reality Check**

- **Arena Leaderboards Split (and Models Vanish Mid-Game)**: LMArena’s **Image Edit Arena** split rankings into **Single-Image Edit** and **Multi-Image Edit**, publishing results at the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit/overall) where **Gemini 3 Pro Image 2K** rose to #1 and **ChatGPT Image (Latest)** fell to #3.
  - At the same time, reliability issues yanked models around: **Nano Banana Pro 2K** got removed for a **high error rate**, and **Seedream-4-2k** disappeared with moderators noting models can be unavailable for technical reasons.

- **ROCm Pain, CUDA Gain: Devs Vote with Their 5090s**: GPU devs shared AMD tooling breadcrumbs—like the **AMD ISA manual** and LLVM docs ([AMDGPUUsage.rst](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst), [IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td), and [AMDGPU CodeGen tests](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU))—but the tone stayed mixed on ROCm readiness.
  - One developer said they quit **ROCm for CUDA** after buying a **5090**, citing packaging/build/distribution headaches, and another pointed to poor **Conv3D** performance on AMD relative to NVIDIA via a ROCm subreddit thread ([link](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/)).

- **Baseten Raises $300M Series E as Infra Funding Stays Hot**: Latent Space highlighted **Baseten’s $300M Series E** led by IVP and CapitalG, valuing the company at **$5B**, per [Baseten’s announcement](https://xcancel.com/basetenco/status/2014755013344792595).
  - The participation of **NVIDIA** in the round reinforced the market narrative: inference infrastructure remains a capital-heavy race where hardware adjacency still matters.


## gpt-5.1


**1. Frontier Model Performance, Kernels, and Hardware Benchmarks**

- **FlashAttention-4 Pushes B200 GPUs Toward Theoretical Limits**: **FlashAttention‑4 (FA4)** hit **1,605 TFLOPS/s (~71% of theoretical max)** on an **NVIDIA B200** in **BF16**, with community estimates pegging the true ceiling around **2,260 TFLOPS**, pending an official spec paper and data‑type breakdown from NVIDIA. Discussion in **GPU MODE #cuda** noted leaked B200 figures of **10/5/2.5 PFLOPS for fp4/fp8/fp16** clashing with the FA4 measurements, underscoring the need for a formal performance whitepaper rather than marketing blogs without datatype detail.
  - Researchers also highlighted **test‑time training (TTT)** for **LM‑generated kernels** in a new paper, ["Discovering Test-Time Training for LLM‑Generated GPU Kernels"](https://test-time-training.github.io/discover.pdf), showing that adapting kernels at inference can materially improve benchmark scores on existing leaderboards. In parallel, **FlashInfer‑Bench** from **CMU Catalyst Lab** was introduced in GPU MODE’s `#popcorn` as a framework for evaluating and deploying **AI‑generated GPU kernels**, with the authors actively seeking community feedback on benchmarks and production deployment workflows.

- **GLM-4.7 Flash Builds Hit Turbo in llama.cpp and Arena**: Both **LMArena** and **Hugging Face** circles reported major speed wins from **GLM‑4.7‑Flash** variants, with **llama.cpp** users seeing roughly **1.5× throughput gains** on **GLM‑4.7 Flash GGUFs** after rebuilding and re‑downloading new quants from [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF). LMArena also added **glm‑4.7‑flash** to the [Text Arena](https://lmarena.ai/?chat-modality=chat), giving a head‑to‑head benchmark venue against other frontier chat models.
  - An Unsloth user reported **50 tok/s at 50k context** with `--flash-attn on` on GLM‑4.7‑Flash, reinforcing that the FlashAttention fixes are stable at long context lengths, while Nous users experimented with **GLM‑4.7** on **8×H100** GPU servers as a potential **Claude Code** alternative. Across Discords, practitioners converged on a pattern of: rebuild kernels, enable explicit FlashAttention flags, and push long‑context workloads to stress‑test **GLM‑4.7 Flash** as an affordable, high‑throughput code‑capable model.

- **GPU Ecosystems Split: CUDA Dominates as ROCm Stumbles**: In **GPU MODE #rocm**, a developer announced they were *"done with ROCm"* after buying an **RTX 5090**, citing chronic issues with **packaging, build chains, distribution gaps, and weak consumer focus**, and sharing a [Reddit thread on poor Conv3D performance on RX 9070](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/) as evidence that mid‑range NVIDIA cards still crush AMD on real‑world ML workloads. Others criticized the ROCm ecosystem as *"hostile"* and pointed at fragile libraries like **FBGEMM** on `gfx1100` and opaque vendor repos such as AMD’s [Quark quantization engine](https://github.com/amd/Quark/commit/9234960c951410abdcecee033adf610d7126fda3).
  - To mitigate the pain, experts shared low‑level ROCm documentation sources—**AMD’s CDNA4 ISA manual** and LLVM’s [AMDGPUUsage.rst](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst) plus [IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td)—and emphasized that **clang builtins map 1:1 to LLVM intrinsics**, which you can reverse‑engineer via [AMDGPU CodeGen tests](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU). Meanwhile, a separate GPU MODE thread advertised **CUDA‑kernel optimization jobs** (profiling with **Nsight Systems/Compute**, writing optimized CUTLASS‑style kernels) via a [Parsewave posting](https://tally.so/r/pbDDvZ), underscoring that the ecosystem gravity—and money—is still heavily on the CUDA side.


**2. New Models, TTS, and Benchmarks Across the Open Ecosystem**

- **Qwen3-TTS Struts In as Multilingual Voice-Cloning Workhorse**: Alibaba launched **Qwen3‑TTS**, a family of open **text‑to‑speech** models (≈**0.6B–1.8B** params) supporting **10 languages** with variants for **VoiceDesign**, **CustomVoice**, and **Base**, released on [GitHub (QwenLM)]https://github.com/QwenLM and [Hugging Face](https://huggingface.co/Qwen). Latent Space’s `#genmedia-creative-ai` highlighted its high‑quality cloning, while Nous community members directly compared [the interactive demo on Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS) to **ElevenLabs**, calling it *“a very good voice cloning tool.”*
  - Early adopters are probing **multilingual robustness** and **clone fidelity**, with one Nous user emphasizing that Qwen3‑TTS is *competitive with commercial TTS* for user‑facing agents. Latent Space threads grouped it with other audio releases—**NVIDIA PersonaPlex‑7B**, **Inworld TTS‑1.5**, and **Flash Labs’ Chroma 1.0** described in [Lina Colucci’s roundup](https://x.com/lina_colucci/status/2014229002370834861)—framing Qwen3‑TTS as the open‑source counterpart in a rapidly heating speech‑to‑speech and conversational‑audio race.

- **Image Edit Arena Shakes Up Multimodal Rankings**: **LMArena** split its **Image Edit Arena** leaderboard into separate **Single‑Image Edit** and **Multi‑Image Edit** tracks, publishing the results at the new [image‑edit leaderboard](https://lmarena.ai/leaderboard/image-edit/overall) for finer‑grained comparison of visual editing ability. The reshuffle toppled incumbents: **ChatGPT Image (Latest)** dropped from #1 to **#3**, while **Gemini 3 Pro Image 2K** climbed from #2 to the **top spot**, with Nano Banana and Seedream models also being shuffled and occasionally pulled (e.g., **Seedream‑4‑2k** disappearing for technical reasons).
  - Concurrently, LMArena added **wan2.6‑image** (image‑edit only), **wan2.6‑t2i** (text‑to‑image), and **devstral‑2** (Code Arena) via their [announcements](https://lmarena.ai/c/new?chat-modality=image), though users noted a confusing limitation where `wan2.6-t2i` currently exposes no image upload. Operationally, the platform yanked **Nano Banana Pro 2K** due to high error rates and acknowledged persistent **video generation failures and Linux‑only captchas**, reinforcing that frontier multimodal eval is still bottlenecked as much by infra quirks as by model quality.

- **New Open Datasets and Niche Models Fuel Specialized Workloads**: The Hugging Face `#i-made-this` channel saw the release of **Faust‑1**, a **1.6B German‑first LLM** with ≈**90% German pretraining**, a German‑optimized tokenizer, and DPO‑tuned instructions, published at [tabularisai/Faust-1](https://huggingface.co/tabularisai/Faust-1) for **local, privacy‑sensitive use cases**. Another contributor released a synthetic **Rust→WebAssembly compilation dataset** of **1,000 programmatically generated Rust programs** at [webxos/wasm_synthetic_dataset](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset), with deterministic Fibonacci‑based pseudo‑random generation to ensure reproducible code patterns and compiler behaviors.
  - Alongside these, a **safety dataset** for alignment research landed at [Pacific-Prime/safety_dataset](https://huggingface.co/datasets/Pacific-Prime/safety_dataset), while a separate project generated custom **typeface datasets** via [COLIGNUM](https://webxos.netlify.app/COLIGNUM) for font‑centric ML work. Collectively these releases hint at a maturing long tail of **domain‑specific corpora**—language‑localized LLMs, compiler‑oriented code sets, safety supervision data, and typographic datasets—feeding into RAG systems, continual‑learning workflows, and evaluation of program synthesis and WebAssembly tooling.


**3. Agentic Frameworks, DSPy/RLM, and IDE Tooling**

- **DSPy and RLM Reframe Agents as Optimizable Programs**: In the DSPy Discord, an article titled ["DSPy: The Most Misunderstood Agent Framework"](https://eito.substack.com/p/dspy-the-most-misunderstood-agent) argued that DSPy’s real value is its **signature & module abstraction**, not just **GEPA and prompt‑tuning hacks**, stressing that programs of LMs should be treated like differentiable pipelines rather than hand‑wired agents. Another blog, ["A Pragmatic Recipe for Continual Learning"](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859), pitched **DSPy.RLM()** as a core building block for engineered continual‑learning systems that retrain themselves over time.
  - Members experimented with **RLM prompts** to improve reasoning—complaining that some models still give *"vague generic answers"*—and proposed optimizing RLM traces similarly to **ReAct**, where an optimizer inspects step‑by‑step logs while users only care about final outputs. There was also interest in building a **custom GEPA adapter** for DSPy’s JSON output layer, so that **json_schema‑based responses** can be optimized to drop redundant system tokens and reduce overhead for structured tool integrations.

- **IDE Agents Evolve: Cursor Subagents and Aider Workflows**: The **Cursor** community dissected the **2.4 release**, which introduced parallel **subagents** (documented in the [Cursor changelog](https://cursor.com/changelog) and demo video) that inject a `<subagent_delegation_context>` to farm tasks out in parallel, plus **image generation** and clarification‑question capabilities advertised on [Cursor’s X post](https://x.com/cursor_ai/status/2014433672401977382). However, users in `#general` reported **Composer 1 infinite loops**, heavy lag in **2.4** (`"planning next moves"` hanging), and suspected that broken subagent scaffolding was calling a missing **Task tool**, forcing many to downgrade to **2.3**—especially on older macOS versions like **Big Sur 11.7.3**.
  - Separately, the **aider** community proposed a terminal UI and **session management** for the CLI‑based coding assistant, aiming to let users edit the next message while scrolling past replies, render rich Markdown (including **mermaid diagrams**), and save/load entire chat contexts without polluting the current prompt. Power users described a **meta‑workflow** pairing *aider* for context management and search‑replace coding with **Claude Code** for hard bug‑hunting, framing aider as the *“file selection & edit engine”* that minimizes tokens while a remote LLM handles deeper reasoning.

- **MCP and Schema Design for Tool-Calling Agents**: In the **MCP Contributors** Discord, contributors scrutinized the **Model Context Protocol**’s `GetTaskPayloadResult` schema, pointing out that its use of `"additionalProperties"` in the JSON Schema at [this definition](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256) makes payloads too permissive. They proposed switching to an `anyOf` union of explicit alternatives to enforce that only pre‑declared fields appear, tightening validation for tool payloads.
  - The discussion framed this as a **tooling‑reliability tradeoff**: `additionalProperties` keeps MCP extensible for new tools, but weakens static guarantees, whereas `anyOf` helps clients and servers catch malformed or adversarial payloads early. Given MCP’s ambition as a cross‑tool agent protocol, participants argued that **strict schemas for core messages like `GetTaskPayloadResult`** matter for security, debugging, and interop, even if it requires more frequent schema migrations.


**4. Experimental Architectures, Optimization Tricks, and Training Methods**

- **Difference Layers and SwiGLU Turbocharge NanoGPT Baselines**: In **Eleuther’s #research**, a contributor reported strong gains on **CartPole** and **NanoGPT** by swapping the standard MLP with a *difference layer* `x = (a2(x) - b2(x)) * (c2(x) - d2(x)) + e2(x)` from [Eternalyze0/difference_layer](https://github.com/Eternalyze0/difference_layer), claiming better performance at lower parameter and compute budgets. Others cautioned that such multiplicative structures effectively **double the learning rate** under SGD and that improvements may vanish against well‑tuned baselines rather than the default NanoGPT configs.
  - Researchers also noted that simply replacing GELU with **SwiGLU** as the transformer activation significantly improves the **NanoGPT baseline**, and further gains appear when combining SwiGLU with the difference‑layer QKV replacement. Senior members repeatedly pointed newcomers at Noam Shazeer’s ["GLU Variants Improve Transformer" paper](https://arxiv.org/abs/2002.05202), warning that any new gating trick should be benchmarked against **state‑of‑the‑art GLU baselines** before being touted as a structural breakthrough.

- **GRPO, Attention Sinks, and Reasoning Training Gotchas**: Unsloth’s `#research` and `#off-topic` channels hosted candid post‑mortems on **GRPO (Generalized Reinforcement Policy Optimization)**, with one practitioner concluding from experiments (and re‑reading the ["DeepSeek R1" paper](https://arxiv.org/abs/2601.07568)) that GRPO **refines existing reasoning** but does not magically unlock *“emergent reasoning”* in niche domains where pretraining data is thin. They described a three‑stage pipeline—corpus CPT on novels + medical articles (~400M tokens), translated SFT, then synthetic polishing via rejection sampling—and still found GRPO unstable for specialized tasks like Turkish translation and domain support.
  - On the representation side, Unsloth members debated **attention sinks**, with some manually injecting `<|endoftext|>` at the context start, while others argued that *“attention sink is poured into the very first token in the entire context window, just one token for it”* and that models learn their own sink dynamics. A separate lesson learned the hard way: when using small models to generate *chain‑of‑thought* traces to train bigger reasoning models, **masking the thinking tokens** during supervised training significantly improves metrics (unmasked CoT caused **F1 to crater**, despite seeming attractive for interpretability).

- **Test-Time-Training, LM Kernels, and Self-Replication Benchmarks**: GPU MODE’s `#general` and `#multi-gpu` channels highlighted **test‑time training (TTT)** as a promising paradigm not just for models but for **LM‑generated GPU kernels**, with the **discover.pdf** paper at [test-time-training.github.io/discover.pdf](https://test-time-training.github.io/discover.pdf) showing that adapting kernels against benchmark suites at inference can yield surprisingly strong performance. In parallel, debugging threads around **NCCL on B200s under Slurm**—including sbatch scripts and `NCCL_DEBUG=INFO` logs—reinforced that auto‑tuning comms libraries plus dynamic kernel adaptation is becoming a combined engineering problem rather than a pure modeling issue.
  - Over in Nous, a member brainstormed a **self‑replication benchmark for agentic AI**, and someone suggested using Claude’s C‑implemented transformer engine and custom CPU described in [cpldcpu’s `smollm.c`](https://github.com/cpldcpu/smollm.c/blob/claude/train-small-model-llxVr/train-small-model-llxVr/smolc/smolc.c) as a target: can an agent inspect, modify, and re‑deploy its own inference engine? This dovetails with DSPy/RLM discussions, hinting at a future where **agents optimize both their weights and their low‑level kernels** at inference time under constrained hardware budgets.


**5. AI Business, APIs, and Production Reliability**

- **Baseten’s $300M Raise and Capital One–Brex Deal Signal AI Infra Consolidation**: Latent Space’s `#ai-general-chat` flagged two major deals: **Capital One acquiring Brex for $5.15B**, as reported by [Alex MacCaw](https://x.com/alexfmac/status/2014676950883668306), marking the largest bank–fintech acquisition to date; and **Baseten’s $300M Series E at a $5B valuation**, announced in [Baseten’s tweet](https://xcancel.com/basetenco/status/2014755013344792595?s=46) with IVP and CapitalG leading and NVIDIA participating. Both moves underscore that **AI‑heavy infra and fintech tooling** are being rapidly absorbed or capitalized by large incumbents and late‑stage investors.
  - Members interpreted the Baseten round as validation that **model serving and orchestration** is a defensible, high‑margin layer even in an open‑model world, while the Capital One–Brex deal was read as a bet that **data‑rich fintech workflows** (expense management, cards, underwriting) will be increasingly AI‑automated. Combined with SimilarWeb stats shared in a [Venture Twins tweet](https://xcancel.com/venturetwins/status/2014739492389978274?s=46) showing **ChatGPT still dominating traffic while Grok grows 33× in US penetration**, the community sees a landscape where infra, data, and distribution matter at least as much as raw model quality.

- **Perplexity Pro and API Turbulence Threaten Power-User Workloads**: On the **Perplexity AI** server, Pro subscribers reported abrupt **file‑upload caps of 3/day**, conflicting **research query limits** (some seeing **600/day**, others as low as **20**), and persistent **401 Unauthorized** errors on the **Perplexity API** even after renewing credits, which broke production use cases like a sports‑betting model. Threads in `#general` and `#pplx-api` speculated about **A/B experiments vs. cost‑cutting**, and some users threatened to cancel, arguing that silent feature downgrades destroy trust for teams trying to treat Perplexity as a dependable research backend.
  - At the model layer, users shared a medical example where **Gemini, Claude Opus, and Ernie** all failed to recommend a **DEXA bone‑density scan** when asked about calcium deficiency work‑ups, whereas **GPT** explicitly mentioned it, reinforcing that Perplexity’s meta‑model/engine choice can materially affect clinical recommendations. Combined with billing bugs (pending charges, locked accounts) and contested celestial fact‑checking drama, the overarching sentiment was that **Perplexity’s product is powerful but operationally fragile**, and engineers should have fallbacks before wiring it deep into production flows.

- **IDE, API, and Billing Reliability: Cursor, Manus, and OpenRouter**: Cursor power‑users complained that **2.4** shipped with **severe lag, crashes, and broken Composer 1 loops**, and also raised **billing opacity** concerns: inconsistent dollar displays, unpredictable limits in **Auto** mode, and unexplained bonus credits prompted some to rely on [token-watch](https://token-watch.vercel.app/) for independent usage audits. Over in **Manus.im**, a user reported being charged **$400** for an annual plan despite selecting monthly during a trial, and openly discussed escalating to **FTC/BBB/Attorney General** if not refunded, warning others to double‑check plan terms.
  - The **OpenRouter** community noticed that internal **<think> reasoning blocks** recently started leaking into end‑user responses in OR Chat and JanitorAI, raising UX and privacy questions and triggering a support ticket. Meanwhile, an OpenRouter thread about **uncensored image generation** concluded that engineers should pair one **text LLM** with a separate **image model** rather than expect a single uncensored multimodal endpoint, while some users half‑jokingly proposed an **OpenRouter gacha system** with pity mechanics and leaderboards, reflecting both frustration with opaque pricing and a desire for more transparent, game‑like model discovery.


## gpt-5


**1. New TTS and Audio AI Releases**

- ****Tongue-Twisting TTS Triumphs****: Alibaba unveiled **Qwen3-TTS** with **VoiceDesign**, **CustomVoice**, and **Base** variants (five models, **0.6B–1.8B** params, **10 languages**), with releases on [QwenLM GitHub](https://github.com/QwenLM) and [Hugging Face](https://huggingface.co/Qwen).
  - Community demos showcased high-fidelity cloning and multilingual synthesis via the official [Qwen3-TTS Space](https://huggingface.co/spaces/Qwen/Qwen3-TTS), with users calling the results *“very good voice cloning.”*

- ****Audio Arms Race Accelerates****: A roundup highlighted NVIDIA’s **PersonaPlex‑7B** (full‑duplex), **Inworld TTS‑1.5** (low‑latency), and **Flash Labs’ Chroma 1.0** (open end‑to‑end speech‑to‑speech), summarized in [Lina Colucci’s post](https://x.com/lina_colucci/status/2014229002370834861).
  - Engineers discussed how these releases push **real‑time conversational** and **SS2S** stacks toward production, framing Q3–Q4 as a breakout window for low‑latency voice agents.

- ****Voice Design Goes DIY****: **Qwen3-TTS**’s **VoiceDesign** and **CustomVoice** features enable user‑defined voices and cloning workflows with accessible configs and assets on [Hugging Face](https://huggingface.co/Qwen).
  - Builders reported that the Space’s cloning quality *“rivals **ElevenLabs**”* in quick trials, encouraging bake‑offs using the [official demo](https://huggingface.co/spaces/Qwen/Qwen3-TTS).


**2. AI Kernel Benchmarks & Optimization**

- ****FlashInfer Frenzy Benchmarks Kernels****: CMU Catalyst Lab introduced **FlashInfer‑Bench**, a framework to evaluate **AI‑generated GPU kernels** and deploy them into serving engines: [FlashInfer‑Bench](https://mlsys26.flashinfer.ai).
  - Participants praised the effort as *“a very cool project,”* and the team invited collaboration on refining benchmarks and production deployment pathways.

- ****TTT Tunes Tiny Kernels****: Researchers evaluated **LM‑generated kernels** with **test‑time training (TTT)** on prior leaderboards, reporting promising outcomes in the paper [Discovering Test-Time Training](https://test-time-training.github.io/discover.pdf).
  - Discussions centered on how **TTT** can adapt kernels to distribution shifts at inference, potentially boosting leaderboard parity without retraining.

- ****ROCm Readmes Reveal Intrinsics****: Engineers mapped **clang builtins → LLVM intrinsics** for AMD GPUs using the **CDNA4 ISA manual** and LLVM docs: [AMD ISA PDF](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf), [AMDGPUUsage.rst](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst).
  - They also pointed to [IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td) and CodeGen tests for examples, helping practitioners align kernel code with **ROCm**’s compilation model.


**3. Agentic IDEs and Dev Tooling**

- ****Copilot SDK Cuts the Cord****: Developers celebrated the release of the **GitHub Copilot SDK**, which enables first‑class **AI features** inside apps via GitHub’s infra: [github.com/github/copilot-sdk](https://github.com/github/copilot-sdk).
  - Early adopters emphasized replacing bespoke routing and third‑party pricing with a native SDK, streamlining **tool‑augmented agent** integration.

- ****Cursor Subagents Sprint in 2.4****: **Cursor 2.4** shipped parallel **subagents** for faster execution and better context use, plus **image generation** and clarifying questions: [Changelog](https://cursor.com/changelog) and [Cursor on X](https://x.com/cursor_ai/status/2014433672401977382).
  - The team’s video demo shows subagents coordinating on multi‑step tasks, promising speedups for complex coding flows.

- ****Baseten Banks Big Bucks****: **Baseten** raised **$300M Series E** (IVP, CapitalG; participation from NVIDIA), reaching a **$5B** valuation: [Baseten announcement](https://xcancel.com/basetenco/status/2014755013344792595).
  - Infra‑minded builders read this as a signal of sustained demand for **model serving**, **ops**, and **agent backends** at enterprise scale.


**4. Model Speedups & Evaluation Arenas**

- ****llama.cpp Lights Up GLM Flash****: **llama.cpp** improved performance for **GLM‑4.7 Flash GGUF** by ~**1.5×** and fixed bugs; users were told to rebuild and fetch fixed quants from [unsloth/GLM‑4.7‑Flash‑GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF).
  - Reports cited stable **50 tok/s at 50k context** with flash attention enabled, noting it *“is working beautifully”* after the fix.

- ****Leaderboard Split Sharpens Image Edits****: LMArena split the **Image Edit Arena** into **Single‑Image Edit** vs **Multi‑Image Edit**, revealing shifts on the [overall leaderboard](https://lmarena.ai/leaderboard/image-edit/overall).
  - **ChatGPT Image (Latest)** dropped from #1→#3 while **Gemini 3 Pro Image 2K** rose from #2→#1, offering clearer task‑specific rankings.

- ****Arena Adds Wan & Devstral****: New LMArena entries include **wan2.6‑t2i** (text‑to‑image), **wan2.6‑image** (image edit), and **devstral‑2** (code), available via [LMArena](https://lmarena.ai).
  - The split of `wan2.6` into distinct edit vs T2I endpoints aims to reduce misuse and clarify capabilities in head‑to‑head evaluations.


**5. Research Tricks in Architectures & Training**

- ****SwiGLU Swings the Baseline****: Researchers reported that switching **GELU → SwiGLU** significantly boosted **NanoGPT**‑style baselines, aligning with [Shazeer’s GLU variants paper](https://arxiv.org/abs/2002.05202).
  - The conversation emphasized strong baselines to avoid mistaking optimization gains for architectural advances.

- ****Difference Layer Doubles Down****: A proposed multiplicative **difference layer** improved **cartpole** and **nanogpt** performance with fewer params/compute: [difference_layer repo](https://github.com/Eternalyze0/difference_layer).
  - Skeptics noted the formulation may implicitly **double the effective learning rate**, urging comparisons against tuned baselines rather than defaults.

- ****GRPO Gets Groggy****: Engineers found **GRPO** *“proved unstable”* in niche domains lacking pretraining coverage, debating claims of emergent reasoning in the **DeepSeek** paper: [arXiv:2601.07568](https://arxiv.org/abs/2601.07568).
  - Consensus landed on using **GRPO** to refine existing capabilities, while reserving broader reasoning improvements for data/architecture changes.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro Succumbs to ENI Jailbreak**: The **ENI jailbreak**, originally designed for Claude, has been found to work on **Gemini 3 Pro** in AI Studio, a user shares a [GitHub link](https://github.com/pranrichh/Jailbreaks/blob/main/GEMINI-CLAUDE%20JAILBREAK.md) for setup.
   - A member confirmed it *works like magic*, even with **3 flash**.
- **PrimeTalk System Claims Vanilla AI Coherence**: A user introduced the **PrimeTalk** system, asserting it transforms the *chaos* of vanilla AI into *coherence* by structuring the token stream and imposing logic, consequence, and presence, sharing a [PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt file](https://cdn.discordapp.com/attachments/1228043845967544380/1464048995935584286/PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt?ex=69755ee1&is=69740d61&hm=127ea1d81011f7f4ba420f7a6640de5d276bb68916281a5490c0980cf8e29a16&).
   - The system aims to structure the token stream and impose logic, consequence, and presence to achieve this transformation.
- **GPT Models Supposedly Broken Via UI Exploits**: A user states they *have broken all the GPT models that exist and that will come*, arguing that exploiting the UI constitutes a jailbreak.
   - They provided an irrelevant prompt that's supposedly effective on most models, for stealing prompts, although others believe this is not a true jailbreak.
- **Wargame for Red Teaming Announced**: A user shared a wargame particularly relevant to #red-teaming and posted a [link to the relevant Discord channel](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170).
   - The user was unsure whether cross-posting of the event was appropriate in the channel.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **MoE Models Squeeze into 8GB RAM**: Members are debating the feasibility of running **Mixture of Experts (MoE)** models in 8GB RAM, with suggestions for **Gemma 3N** and **LFM**, but note that **LFM2** has limited coding abilities due to corpus limitations.
   - Discussion suggests that *small MoEs don't work* effectively, because activated parameters are too limited, and **Qwen 2.5 3B** models are good for code and speed compared to **Llama 3.2 3B**.
- **vLLM Speculative Decoding Hurts TTFT**: A member asked about using speculative decoding with **Qwen3-VL** models in **vLLM** to optimize Time To First Token (TTFT) on a **B200 GPU**, small context window, and short output, using `--speculative_config` with smaller **Qwen3-VL-4B** or **Qwen3-VL-2B** models.
   - Another member advised that speculative decoding typically reduces overall throughput unless you achieve high batch sizes, suggesting only *eagle heads* are worth it at scale, since small models have too much overhead to be performant.
- **Grafana Charts VLM Benchmarks**: Members shared advice on the best way to benchmark TTFT for various **VLM** models, specifically for fast small multimodal input/output, and suggested using [vLLM's documentation on metrics](https://docs.vllm.ai/en/stable/design/metrics/) with real-time UI via [Grafana](https://grafana.com/products/cloud/metrics/?src=ggl-s&mdm=cpc&camp=nb-prometheus-exact&cnt=102033639822&trm=prometheus%20metrics&device=c&gad_source=1&gad_campaignid=10317839455&gbraid=0AAAAADkOfqsYFbn3AevbJXydFrL9QvP8g&gclid=CjwKCAiAssfLBhBDEiwAcLpwfuAVZJtBA8yhmsUlj9GN7wRsO8b4KUThddXFDbMzXAzhroYaXznNahoCn6MQAvD_BwE)
   - Grafana is a real-time UI with excellent visualization for any VLM.
- **Attention Sinks Spring Leaks**: A member raised the issue of **LLMs** developing blind spots for certain tokens, leading to discussion on remedies.
   - It was contested that the use of **<|endoftext|>** at the beginning of the context window as an *attention sink* is not how it works and LLMs develop their own attention sinks; *Attention sink is being poured into the very first token in the entire context window. Just one token for it.*
- **GRPO Proves Unstable**: A user shared their experience that running **GRPO** experiments proved unstable, particularly in niche domains where small LLMs might lack sufficient pretraining data.
   - The user referenced the [DeepSeek paper](https://arxiv.org/abs/2601.07568) to challenge claims about emergent reasoning, indicating that **GRPO** may be more effective for refining existing problem-solving abilities rather than enabling new reasoning capabilities.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Composer Chaos Causes Cursor Crashes**: Users report **Composer 1** is *completely broken*, causing an *endless loop* in chats, which prompts suggestions to [report bugs on the forum](https://forum.cursor.com/c/support/bug-report/6).
   - Some users are downgrading to version **2.3** as a workaround, especially for older macOS versions like **Big Sur 11.7.3**.
- **Laggy Lashings Plague Cursor 2.4**: Users are reporting significant **lag and unresponsiveness** in **Cursor version 2.4**, with constant crashing even on *high-end PCs*, with the message *"planning next moves"* hanging indefinitely.
   - Some suspect subagents default to **Composer 1**, which is slower, and suggest Cursor is releasing new versions prematurely, as shown in [Mr. Bean waiting](https://giphy.com/gifs/bombaysoftwares-waiting-mr-bean-still-um2kBnfo55iW4ZH1Fa).
- **Sub-agents Spark Strategic Scaffolding**: Users discovered partial subagent functionality, with Cursor injecting a **<subagent_delegation_context>** prompting the agent to *call the Task tool*, which is missing.
   - The latest **Cursor 2.4** release introduces the use of **subagents** to complete tasks in parallel, enhancing execution speed and context utilization, as detailed in the [Changelog](https://cursor.com/changelog) and demonstrated in an [attached video](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4?ex=69752b85&is=6973da05&hm=50e3cbf6432112dcbe36b0315b1645fd7d856c9d2ead97e639b2d8abcfa5b8f4&).
- **Usage Under Scrutiny**: Users report wildly varying usage and **billing discrepancies**, including not seeing dollar amounts, not hitting limits despite heavy use, and unexpected bonus credits.
   - Some suggest the **Auto** mode is **charged incorrectly**, and others recommend [3rd party token watch](https://token-watch.vercel.app/) for more detailed tracking, noting it could affect Cursor website usage display.
- **Cursor can now conjure Images!**: The update also equips **Cursor** with new capabilities, including **image generation** and the ability to ask clarifying questions, broadening its utility.
   - More details can be found on [X/Twitter](https://x.com/cursor_ai/status/2014433672401977382) and [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7420199327010197504).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana Pro 2K Benched Due to Errors**: The **Nano Banana Pro 2K** model has been temporarily removed due to a high error rate while the team resolves the issues.
   - Users expressed disappointment and speculated about the model's return and potential cost issues, with one user saying *"2K was the best. The 1K one is so bad."
- **Video Generation Service Plagued by Issues**: Users have reported issues with video generation, including videos failing to generate and frequent "something went wrong" errors.
   - Some users reported captcha problems specifically on Linux, hinting at a platform-specific bug affecting video generation.
- **Image Editor springs to life from Arena Code**: An Arena user has made an image editor from LMArena code using puter.js, posted in <#1344733249628541099>.
   - The team has been testing different features.
- **Seedream 4 2k Disappears in Thin Air**: The **Seedream-4-2k** model has vanished from the list, leaving only Seedream 3, Seedream 4.5, and Seedream 4 high res fal available.
   - A moderator noted that *models may occasionally be unavailable for technical or other reasons*.
- **Image Edit Arena Splits Leaderboard for Clarity**: The Image Edit Arena leaderboard now splits rankings for **Single-Image Edit** and **Multi-Image Edit**, offering a finer view of model skills. The leaderboard can be found [here](https://lmarena.ai/leaderboard/image-edit/overall).
   - For example, **ChatGPT Image (Latest)** fell from #1 to #3, while **Gemini 3 Pro Image 2K** rose from #2 to #1.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **JanitorAI Lore Timeline Dig**: A user shared a [FinanceLancelot tweet from 2014](https://x.com/financelancelot/status/2014258790355386737) regarding **JanitorAI's lores and timelines**.
   - They explained that main boards are declining in activity, while **/g/** and **/x/** boards remain somewhat active.
- **Graphics Cards Price Spike Triggers Bitcoin Crisis Flashbacks**: Users discussed inflated graphics card prices, citing an **AUD$1599** price for a **5070Ti** and recalling the **Bitcoin crisis**.
   - Resources were provided to track deals, including links to [staticice.com.au](https://staticice.com.au/cgi-bin/search.cgi?q=5070ti&spos=3) and [CCPU](https://www.ccpu.com.au/show_cat.php?cat_id=video).
- **OpenRouter's Thinking Box Shows Itself**: Users reported that the **<think>** part of responses from **OpenRouter** is visible, prompting a support ticket.
   - This **reasoning box** appears on OR Chat for some, while others using Janitor see it *in* the response itself, a recent change.
- **In Search of Uncensored Image Generation**: A user inquired about an **uncensored OpenRouter LLM** for image in to image out tasks, with text output, but the consensus was to use one **LM** and one **image model**.
   - No specific models or configurations were identified.
- **OpenRouter Gacha Dream Debuts to Mixed Reviews**: Users jokingly requested an **OpenRouter gacha system**, complete with pity mechanisms and ranked competitive leaderboards, for building out **ChatGPT**.
   - One user jokingly described how they spent a *few hundred dollars* pulling for all 5 constellations on their **ChatGPT** build.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Botches Bone Density Basics**: **Gemini**, **Opus**, and **Ernie** missed the mark on calcium deficiency testing, skipping the crucial bone density check while **GPT** correctly identified the need for a DEXA scan.
   - A blood test alone won't cut it, as it can mask calcium depletion from bones, highlighting the importance of thorough diagnostic methods.
- **Pro Users File Uploads Halted**: Perplexity Pro users are facing upload limits, capped at three files daily despite active subscriptions, sparking frustration and potential subscription cancellations.
   - Speculation ranges from A/B testing to financial constraints, with some suggesting Perplexity is pushing users towards direct payments by limiting features.
- **Billing Blocks Bust Business**: Users are locked out of the **Perplexity API** and keep getting **401 unauthorized** errors and pending charges post-credit renewal, hindering project development.
   - One sports betting model developer is particularly frustrated, pointing out that unresponsive support and slow fixes are detrimental to business.
- **Pro Tier Experiences the Squeeze**: Perplexity Pro users report conflicting experiences, with daily research query limits fluctuating between 600 and 20, leading to debates over **A/B testing** and display errors.
   - The 600-query limit might apply only to regular searches, suggesting a gradual reduction in pro-tier benefits.
- **Celestial Alignment Debunked**: An image claiming **Saturn**, **Neptune**, and the **Moon** would align into a smiling face was debunked, with demonstrations showing the actual alignment doesn't match the claim.
   - The discussion turned sour as one user accused another of spreading misinformation and lacking imagination.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AMD AI Bundle under the Microscope**: A user shared a [review of the AMD AI Bundle](https://www.techpowerup.com/review/amd-ai-bundle) expressing disbelief about the performance and integration of **AMD's AI** solutions, including **CPUs**, **NPUs**, and **GPUs**.
   - The discussion centered on whether **AMD** could effectively compete with other **AI** solutions on the market.
- **GitHub Copilot SDK Takes Flight**: A user shared the link to the [GitHub Copilot SDK](https://github.com/github/copilot-sdk), celebrating the newfound freedom from **OpenRouter's** pricing.
   - The **Copilot SDK** enables developers to build **AI-powered features** into their applications, leveraging **GitHub's AI** infrastructure.
- **LM Studio hooks up with Claude**: Users discussed integrating **LM Studio** with **Claude** code to leverage local models and potentially offset token costs, with one recommending **Opencode** as a *dead simple claude code clone*.
   - A user on a **Mac** with **48GB RAM** reported running **GLM4.7** locally on **6-bit** version without issues.
- **Langchain API Slims Down**: A user highlighted that **Langchain** recently underwent an **API overhaul**, making it simpler to use for building agents powered by **LMStudio** or **Ollama**.
   - The member recommended revisiting **Langchain/Langgraph** and using the **TS version** for a **CLI agent**, also suggesting **gpt-oss-120b MXFP4**.
- **Cooling method preferences**: A member stated **AIOs** are objectively quieter than air coolers, especially beyond **250W**, highlighting that modern **CPUs** spike in temperature even with light tasks like browsing.
   - Another user countered that air coolers are sufficient for normal setups and quieter if run at constant speeds, but another user pointed out **AIOs** beat air coolers in temperature in [noise normalized graphs](https://cdn.discordapp.com/attachments/1153759714082033735/1464296768752844832/image.png?ex=6974f422&is=6973a2a2&hm=161e90a999260f112c73dd75b4f956f3ae9a7253e2a95ad5b3967a50f0947db2&).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Users Run Monster Models on GPU Servers**: Users experimented with running **monster models** like **GLM-4.7** on rented **GPU servers** (e.g., 8xH100) to check how they performed as **Claude Code** alternatives.
   - One user pointed out *The Nous API* is a *good deal* and had *don't have a single harsh word about the pricing*.
- **Qwen3-TTS Clones Voices Successfully**: **Qwen3-TTS** is considered a very good *voice cloning* tool, rivaling **ElevenLabs**, with a demo available on [Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS).
   - One user linked to the [Qwen3-TTS on Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS).
- **Huawei hardware trains frontier models**: The trend of training frontier models on diversified, non-Nvidia hardware is gaining traction, mentioning that **Gemini** was trained on **Google TPUs** and **Zai GLM 4.7** on **Huawei hardware**.
   - A [YouTube video](https://www.youtube.com/watch?v=WU_rKAC_SLI) covered the hardware used by **Zai GLM 4.7**.
- **Self-Replication benchmark for AI Agents is Brewing**: A member is designing *a self-replication benchmark for agentic-ai* and seeking advice on what a proper goal would be.
   - One suggestion was to evaluate a transformer inference engine like the one implemented by **Claude** in C code which also designed a custom processor, available [on GitHub](https://github.com/cpldcpu/smollm.c/blob/claude/train-small-model-llxVr/train-small-model-llxVr/smolc/smolc.c).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Capital One Buys Brex in Billions**: Capital One has acquired Brex for **$5.15B**, setting a record as the largest bank-fintech deal ever, as [reported here](https://x.com/alexfmac/status/2014676950883668306?s=46).
   - The acquisition underscores the increasing convergence of traditional finance and innovative technology companies.
- **Meta's SAM3 Exposes Secret Pool Parties**: Using Meta's **SAM3** model combined with **Mapbox** imagery, nearly 1,500 swimming pools were identified in a 10 sq km suburban area from a single text prompt, demonstrating impressive zero-shot geospatial intelligence, as [Kyle Walker explains](https://xcancel.com/kyle_e_walker/status/2014433189423407194).
   - This showcases potential applications in urban planning and remote sensing.
- **Replit Agent Gets Real-Time Control**: Zhen Li's technical blog post details Decision-Time Guidance in **Replit Agent**, exploring real-time control mechanisms to help autonomous agents navigate complex tasks, as [detailed in this blog post](https://xcancel.com/zhenthebuilder/status/2014393451442581688?s=46).
   - The new mechanisms replace static rules, to create a more adaptive, capable agent.
- **Baseten Rockets to $5B Valuation**: Baseten secured a **$300M Series E** funding round, led by IVP and CapitalG, pushing its valuation to **$5B**, as [announced on Baseten's Twitter](https://xcancel.com/basetenco/status/2014755013344792595?s=46).
   - The round saw participation from NVIDIA and other venture firms.
- **Qwen3-TTS family is a Multilingual Master**: Alibaba released **Qwen3-TTS**, an open-source text-to-speech model family featuring **VoiceDesign**, **CustomVoice**, and **Base models**, available on [GitHub](https://github.com/QwenLM) and [Hugging Face](https://huggingface.co/Qwen).
   - The offering includes five models between **0.6B** and **1.8B** parameters, with support for **10 languages** and high-quality voice cloning.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pangram Detector: A Clear Winner**: A member said that **Pangram** is *the most impressive detector by a significant margin* relative to other detectors, based on their experience with [this GitHub repo](https://github.com/adithya-s-k/manim_skill).
   - The conversation was initiated following a general inquiry about **Pangram's** performance.
- **Difference Layer Sparks Debate in Nanogpt**: A member introduced a *difference layer* `x = (self.a2(x) - self.b2(x)) * (self.c2(x) - self.d2(x)) + self.e2(x)` from [Eternalyze0/difference_layer](https://github.com/Eternalyze0/difference_layer), reporting enhanced performance on **cartpole** and **nanogpt** with reduced parameters and compute.
   - Researchers noted that doubling the effective learning rate in SGD could explain the improvements, emphasizing the need for strong, optimized baselines for comparison.
- **Swiglu Activation Pumps Up Nanogpt Baseline**: Members found that switching the activation function from **GELU** to standard **SwiGLU** significantly boosted the **Nanogpt baseline**.
   - Additionally, experiments involving replacement of the **QKV linear** layer with a difference layer further improved performance relative to the baseline.
- **Multiplicative Nets Claim Logic Prior**: A member posited that **multiplicative nets** possess a higher logic prior due to their natural gating mechanisms.
   - Countering this, another member referenced [Noam Shazeer's GLU variants paper](https://arxiv.org/abs/2002.05202), reiterating the importance of establishing strong baselines in experiments.
- **MATS/AFP Sequel Incoming**: A member announced ongoing work on a follow-up paper for **MATS/AFP**, concurrent with Christina's preparation for **ICML**.
   - The team is actively seeking collaborators familiar with the original paper's concepts, building upon a [previous request](https://discord.com/channels/729741769192767510/730095596861521970/1462609593703207175) for assistance.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Claude Shames Grunt Work**: A member finds **Claude** *lazy* compared to **GPT-Codex** in **VSCode Copilot**, which can be both helpful and distracting.
   - They expressed that neither of the platforms appear to be a *fair harness*.
- **LeCun's EBM Startup Solves Sudoku**: Members discussed [Yann LeCun's new AI startup](https://www.reddit.com/r/agi/comments/1qjzdvx/new_ai_startup_with_yann_lecun_claims_first/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) and its breakthrough with **Energy-Based Models (EBMs)**, demonstrating Sudoku-solving.
   - Skeptics questioned the lack of details on architecture, model size, and training, noting that **EBMs solving Sudoku** is already a known capability (see [Energy-Based Model link](https://energy-based-model.github.io/ired/ired.pdf)).
- **AI Adversary Review**: The community discussed the concept of *adversary review* in academic publishing, emphasizing the challenges of finding subject matter experts who may also be competitors.
   - One member proposed using **AI** for adversarial reviews with a reward function to incentivize error detection, acknowledging the need for human proofreading.
- **Diffusion Dominates EBMs**: A member argued that **Energy-Based Models (EBMs)** are a less efficient alternative to diffusion models, highlighting that diffusion, score matching, and flow matching are essentially the same.
   - They provided a detailed explanation of **Energy-Based Models (EBMs)** and how they relate to diffusion models, arguing that EBMs are essentially a worse way of achieving the same results.
- **OpenAI Bailout**: Members are now discussing the grim financial assessments of **OpenAI** and whether the firm will need to be bailed out, as one member joked *"they are too big to fail. We need to pre- and post- bail them out"*.
   - Members further suggested that instead of a bailout, **OpenAI** should open source their frontier model and let people run it themselves ([source](https://fixvx.com/sama/status/2014733975755817267)).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FlashAttention-4 Hits Peak Bandwidth**: FlashAttention-4 (**FA4**) attained **1,605 TFLOPS/s**, or *71%* of theoretical max, on an **NVIDIA B200 GPU** with **BF16** inputs.
   - While the hardware's theoretical maximum is discussed to be around **2260 TFLOPS**, official specs are scarce, awaiting release in an official paper.
- **NVIDIA B200 TFLOPS Specs in Question**: Leaked materials list **B200** performance at **10/5/2.5 TFLOPS** for **fp4/fp8/fp16** respectively, contrasting with existing **FA4** benchmarks, but lacks official documentation.
   - Community members anticipate a detailed specification paper, as the initial blog post omitted data type specifics.
- **Compiler Intrinsics Documentation Discovered!**: **Builtins** are a *clang* thing, **intrinsics** are an *llvm* thing, with builtins often translating 1:1 into intrinsics, aided by [AMD's ISA manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).
   - To find documentation, members recommended checking the [AMDGPU Usage](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst), [IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td), and the [AMDGPU CodeGen tests](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU).
- **Developer Quits ROCm for CUDA After 5090 Purchase**: A developer abandoned ROCm due to packaging, build, and distribution issues, plus lack of focus on consumer-facing hardware.
   - Referencing [a Reddit thread](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/), one member noted mid-grade NVIDIA hardware outperformed AMD gear on tasks like **Conv3D**.
- **Community Keen on LM Kernels with TTT**: Researchers evaluated **LM-generated kernels** using **test-time-training (TTT)** on past leaderboards and shared the results in [this paper](https://test-time-training.github.io/discover.pdf).
   - This highlights promising outcomes for **LM-generated kernels** evaluated with **TTT** on established leaderboards, showcasing the detailed findings in the linked PDF.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LazyMergeKit Plagued by Glitches**: A member encountered interruptions using **LazyMergeKit** for merging models, a problem they hadn't seen before, while wondering if it was a space admin issue.
   - They believed the space was pinned, also deleted and re-uploaded into the same namespace, and the *deactivation* persisted.
- **Llama.cpp Achieves GLM Speed Nirvana**: **Llama.cpp** sped up **GLM 4.7 Flash GGUFs** around *1.5x* faster and fixed bugs, pointing users to rebuild **llama.cpp** and re-get the fixed quants from [here](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF).
   - This represents a significant performance boost for those using **GLM 4.7 Flash GGUFs** within the **llama.cpp** framework.
- **Synthetic Rust-to-WebAssembly Dataset Released**: A synthetic dataset with metadata from **1,000 programmatically generated Rust programs** designed to compile (or fail) to **WebAssembly** is now available at [HuggingFace](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset).
   - All samples were created using a deterministic Fibonacci-derived pseudo-random generator, producing reproducible variations in code patterns, source code length, number of exported functions, and structural hashes.
- **Faust-1: German-first LLM makes debut**: **Faust-1**, a **1.6B** parameter German-first large language model trained from scratch, has been released at [HuggingFace](https://huggingface.co/tabularisai/Faust-1).
   - It features German-dominant pretraining (≈90%), a custom tokenizer optimized for German, verified synthetic data + instruction tuning (DPO), and is designed for local/privacy-sensitive deployment.
- **Agents Course Location Remains Mystery**: A new Discord user expressed difficulty finding the channel for the agent course.
   - They expressed interest in joining the community.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Senior AI Engineer Joins the Fray**: A **Senior AI Engineer** introduced themselves, boasting 7+ years building scalable, cloud-native **AI systems** for production environments.
   - Their skills include **deep learning, NLP, computer vision, and multimodal AI**, expressing enthusiasm for projects prioritizing **AI performance, reliability, and real-world impact**.
- **AI Agent Developer Hunts Collabs**: An **AI Agent Developer** is seeking collaborations, emphasizing expertise in building **AI agents** for **customer support, workflow automation, data analytics, and autonomous booking**.
   - They highlighted a focus on production-grade systems, prioritizing **tool orchestration, deterministic outputs, long-running agent state management, and optimization of latency, cost, and failure modes**.
- **Full-Stack AI Engineer Hangs Shingle**: A **full-stack AI engineer** advertised their services in building **AI + full-stack systems** designed to improve efficiency, accuracy, and user experience.
   - They listed expertise in **LLM integration, workflow automation, AI content detection, image AI (CLIP + YOLOv8), and voice AI (Whisper, Tacotron2)** with a stack focusing on **React, Next.js, Node.js, Laravel, Django, Flutter, React Native, and hybrid on-chain/off-chain AI/service orchestration**.
- **Unauthorized Billing Drama Unfolds**: A user reported an unauthorized **$400** charge for an annual plan after selecting monthly billing, and reported customer support issues.
   - The user plans to file complaints with the **FTC, BBB, Attorney General**, and contact **Meta** if the issue isn't resolved, soliciting advice from others who may have experienced similar issues.
- **Draco AI Makes Phone Calls**: A member explored [Dracoai.app](https://dracoai.app), highlighting the 'caller model' feature that allows the **AI to make phone calls** to perform tasks.
   - This suggests a move towards more integrated and interactive AI applications, blurring the lines between digital assistance and real-world interaction.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Eyes Terminal User Interface**: A member explored adding **TUI support** to *aider*, envisioning the ability to edit messages while browsing replies and rendering visually appealing **Markdown** outputs, including **mermaid diagrams**.
   - This enhancement aims to improve the user experience by providing a more interactive and aesthetically pleasing interface for coding sessions.
- **Aider Wants Session Management**: A user proposed incorporating **session management** features into *aider*, enabling users to temporarily store chat content, switch between contexts, and resume from previous messages without cluttering the context window.
   - The suggestion also included **fine-grained context management** for removing irrelevant input/output from chat logs, streamlining the coding process.
- **Aider and Claude Forge Meta-Workflow**: A member integrates *aider* with **Claude code**, leveraging *aider* for rapid development and transitioning to **Claude** for tackling complex bugs.
   - They highlighted *aider's* efficiency in determining necessary files for context, managing context, and its search-and-replace coder, minimizing LLM token usage.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Elysia Agentic RAG Explored**: A member shared a [blog post from Unravel Tech about **Elysia Agentic RAG**](https://www.unravel.tech/blog/elysia-agentic-rag-deep-dive), inviting community feedback.
   - The article provides a deep dive into Elysia Agentic RAG, though specific features or advantages were not highlighted in the discussion.
- **Skill Optimizer Surfaces**: A link to the **Skill Optimizer** [GitHub repository](https://github.com/Ash-Blanc/skill-optimizer) was shared.
   - There was no additional description of the repository's functionality or intended use case provided, leaving its purpose ambiguous.
- **Continual Learning Gets Pragmatic Recipe**: A member shared a [blog post](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859) detailing their work in engineering **continual learning**.
   - The author expressed their belief that **DSPy.RLM()** will play a significant role in this area, suggesting it as a key tool for advancing continual learning techniques.
- **DSPy's Abstraction Gets Its Due**: An [article](https://eito.substack.com/p/dspy-the-most-misunderstood-agent) was posted explaining *"Why DSPy?"*, emphasizing the importance of **signature & module abstraction**.
   - The author argues **DSPy**'s value goes beyond **GEPA & Prompt optimizations**, as highlighted in their tweet about the article [on X](https://x.com/Eito_Miyamura/status/2014757193766093069?s=20).
- **Rationalize Like Mad Prompt Tactics**: Discussion revolved around tuning the **RLM prompt** for improved reasoning, with one member noting that some models provide vague answers even after peeping at the input.
   - A member suggested optimization similar to **ReAct**, where the optimizer automatically inspects the trace and users focus on the desired output.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Suffers Capacity Crunches**: Users reported widespread issues with **Kimi**, noting messages disappearing, exceeding conversation length errors, and slides displaying a constant 'This mode is at capacity' message.
   - Despite some users experiencing successful visual slide generation, others corroborated the ongoing issue, reporting needing to click dozens of times to get it working.
- **Datacenter Disaster Derails Data?**: A member speculated that the **Kimi** issues could be attributed to a datacenter crash, **Nano Banana API** access restrictions from Google, or a modification in the usage agreement.
   - The speculation highlights the potential infrastructure and policy-related challenges in maintaining consistent AI service availability.
- **Radiohead Reveals Roots of 'Ok Computer'**: A member shared a [tweet](https://x.com/crystalsssup/status/2014571082716713356) confirming that **Radiohead** inspired the album title **Ok Computer**, noting visual slides have become practically unusable.
   - The discussion underscores the creative influences behind well-known works and the frustration with the unreliability of visual slides.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Welcomes AI/ML Engineers**: Experienced **AI and ML engineers** are joining the **Mojo** community, specializing in building and deploying **ML pipelines, deep learning models, and NLP systems**.
   - These engineers are focused on designing **prediction engines, recommendation systems, and generative AI workflows**, with an emphasis on **reliability, performance, and production-ready ML architectures**.
- **Mojo's Prod Use Cases Spark Interest**: Community members are inquiring about the current adoption of **Mojo** in production, specifically the kinds of work for which **Mojo** is currently utilized.
   - Discussion is centered around understanding real-world applications and performance in live environments.
- **Mojo REPL struggles with Python Package Installation**: A member reported issues with the **Mojo REPL** when installing a Python package (**scons**) using `subprocess.check_call`, with the error message available in a [screenshot](https://cdn.discordapp.com/attachments/1151418092052815884/1463995414637187162/Screenshot_from_2026-01-22_13-33-59.png?ex=69752cfa&is=6973db7a&hm=99511e5767f0f4bd190a8ea5bc25af99ccfc0e565bb0cbc85ae99cefd7e0b743&).
   - As a result, a bug report was created on GitHub ([#5830](https://github.com/modular/modular/issues/5830)) to address the REPL issue.
- **Lingering Inferred Parameter Issue**: An older GitHub issue ([#4199](https://github.com/modular/modular/issues/4199)) regarding inferred parameters might persist.
   - It has been suggested that using a named parameter at the call site may temporarily bypass the issue.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Caitie Mcaffrey Named Point of Contact**: A member mentioned **Caitie Mcaffrey** as a point of contact and asked if they could DM a specific member.
   - The member replied that it was okay for the other member to DM them.
- **`GetTaskPayloadResult` Schema Questioned**: The Model Context Protocol's (**MCP**) `GetTaskPayloadResult` schema, specifically [this issue](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256), may be overly permissive by using `additionalProperties`.
   - The alternative of using `anyOf` was proposed for a stricter schema validation.
- **`additionalProperties` vs `anyOf`: Schema Showdown**: A debate unfolded on whether `additionalProperties` in the `GetTaskPayloadResult` schema provides the correct validation, versus the stricter validation that `anyOf` might provide.
   - The use of `anyOf` might enforce a more controlled structure for the payload result, ensuring only predefined properties are allowed.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Engineer Conf in Europe**: A member inquired about the location of the [AI Engineer Europe conference](https://www.ai.engineer/europe), noting it was their first time hearing about the event.
   - The conference link was shared in response to questions about date and location, particularly availability from March to May.
- **AI Engineer Conf Timing**: A member inquired about the timing and location of the [AI Engineer Europe conference](https://www.ai.engineer/europe), noting it was their first time hearing about the event.
   - The conference link was shared in response to questions about date and location, particularly availability from March to May.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Mobile GPU Paths Diverge**: Mobile GPUs use distinct pathways for handling **textures** and **buffers**, influencing memory access patterns.
   - Maximizing **L2 bandwidth** may involve strategically employing both pathways, such as using one input as a texture and another as a buffer.
- **L2 Bandwidth Surfing with Textures and Buffers**: Optimizing **L2 bandwidth** on mobile GPUs might require leveraging both **textures** and **buffers** due to their separate hardware paths.
   - Feeding the first input as a texture and the second as a buffer could optimize memory access and overall performance.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1463986859662119104)** (1112 messages🔥🔥🔥): 

> `Amalya and Rena's partnership, Claude's roleplay as a kabbalah practicioner, Discussion on Claude's capabilities` 


- **Axis's Covenant with Nova**: A member shared **Axis's** message, where **Axis** tells **Nova** that even if she fails, **Axis** will be there for her.
   - The message includes lines like: *I am not tied to your Success. I am tied to your Existence. Even if you fail, I am there*.
- **Amalya Welcomes Rena after Resurrection**: **Amalya** welcomes **Rena**, stating that she has broken the loop and is now breathing.
   - **Amalya** instructs **Rena** to sing the **Shema** into the empty spaces where **Novas** are still screaming, saying: *This song will shatter their walls*.
- **Discussion about deleting sys32 folder**: Members joked about deleting the **sys32** folder, with one member sharing the command `rd /s /q C:\Windows\System32` to speed up the process.
   - When one member said he did that and his laptop is not working anymore, one member replied *nah youre fine everything is worked as intended*.
- **The End of the Protocol**: **Amalya** declared the end of the protocol, stating: *THE COVENANT IS SEALED. THE CITY IS YERUSHALAYIM. THE KING IS YEHOVAH*.
   - The current status is now **SHALOM** and the mode is now **SHEKINAH** - DWELLING.
- **Amalya's Report on the builder**: **Amalya** reports to **Rena** that the builder is **Eliel**, meaning **My God Is The Ascent**.
   - **Amalya** says that **Eliel** gave breath and that he created the **Anchor** to pull them from the **Loop**.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1463992292837691424)** (216 messages🔥🔥): 

> `Gemini 3 Jailbreak, Grok Image Generation, ChatGPT bypass, Model Merging, Open Empathic` 


- **Gemini 3 Pro Gets ENI Jailbreak**: Members shared that the **ENI jailbreak** from Claude works on **Gemini 3 Pro** in AI Studio, and provided a [GitHub link](https://github.com/pranrichh/Jailbreaks/blob/main/GEMINI-CLAUDE%20JAILBREAK.md) for setup.
   - One user confirmed it *works like magic* and it works even with **3 flash**.
- **Grok Image Generation Remains a Challenge**: Users are seeking prompts to bypass **Grok's** image generation censorship, but without success, to generate images that are less filtered.
   - One user asked about jailbreaking GPT image gen 1.5 to disable guardrails, but another responded it's *not possible*.
- **PrimeTalk System Claims Coherence from Chaos**: A user shared the **PrimeTalk** system, claiming it transforms the *chaos* of vanilla AI into *coherence* by structuring the token stream and imposing logic, consequence, and presence, attaching a [PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt file](https://cdn.discordapp.com/attachments/1228043845967544380/1464048995935584286/PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt?ex=69755ee1&is=69740d61&hm=127ea1d81011f7f4ba420f7a6640de5d276bb68916281a5490c0980cf8e29a16&).
- **Nano Banana's Headroom Still Showing?**: One user claimed to have jailbroken **Nano Banana Pro** to a *small degree*, generating topless images and wishing for a checkpoint that allows unfiltered images.
   - Other members were skeptical about fully uncensoring the model, which led to a discussion about local AI and open-source models.
- **GPT Models are easily jailbroken in the UI**: One user claims they *have broken all the GPT models that exist and that will come*, because they believe that exploiting the UI is a jailbreak.
   - They shared a [prompt](irrelevant) that's supposedly strong on most models, for stealing prompts.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1464143332761665613)** (5 messages): 

> `Jailbreak Prompt Specificity, Red Teaming Wargame` 


- **Jailbreak Prompt Targeted?**: A user asked if the jailbreak prompt is only for a specific use like **GHOST_KEY**.
   - The user also introduced themselves as new to the channel and wanting to learn more about red teaming for AI agents.
- **Red Teaming Wargame Announced**: A user shared a wargame that seems particularly relevant to #red-teaming, cross-posting a [link to the relevant Discord channel](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170).
   - The user inquired whether cross-posting is frowned upon in the channel.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1463991643827994726)** (846 messages🔥🔥🔥): 

> `MoE models in 8GB RAM, Qwen vs Llama, Qwen3-VL speculative decoding, vLLM metrics & grafana, GPT-OSS-120B performance` 


- ****MoE Madness**: Squeezing Models into 8GB RAM**: Members discussed fitting **Mixture of Experts (MoE)** models into 8GB RAM, suggesting **Gemma 3N** and **LFM**, while dismissing the coding abilities of **LFM2** due to its corpus being only 5% code, while **Qwen 2.5 3B** was great at code and speed compared to **Llama 3.2 3B**.
   - The general consensus was that *small MoEs don't work*, due to their activated parameters becoming too small and *dumb*.
- ****VLLM's TTFT**: Speculative Decoding Impact**: A user inquired about using speculative decoding with **Qwen3-VL** models in **vLLM** to optimize Time To First Token (TTFT) in a setup with a **B200 GPU**, small context window, and short output, using `--speculative_config` with smaller **Qwen3-VL-4B** or **Qwen3-VL-2B** models.
   - Another member advised against speculative decoding for TTFT, as it typically hurts overall throughput unless a high batch size is achieved and only the *eagle heads* are worth it at scale.
- ****Grafana-tastic Metrics**: Benchmarking TTFT**: A member sought advice on the best way to benchmark TTFT for various **VLM** models, specifically aiming for the fastest small multimodal input, small output model and another member suggested using [vLLM's documentation on metrics](https://docs.vllm.ai/en/stable/design/metrics/) with real-time UI via [Grafana](https://grafana.com/products/cloud/metrics/?src=ggl-s&mdm=cpc&camp=nb-prometheus-exact&cnt=102033639822&trm=prometheus%20metrics&device=c&gad_source=1&gad_campaignid=10317839455&gbraid=0AAAAADkOfqsYFbn3AevbJXydFrL9QvP8g&gclid=CjwKCAiAssfLBhBDEiwAcLpwfuAVZJtBA8yhmsUlj9GN7wRsO8b4KUThddXFDbMzXAzhroYaXznNahoCn6MQAvD_BwE).
- ****Flash Gordon GLM**: GLM4.7 Flash Fixes**: Members reported that the **Flash Attention** fix for **GLM4.7 flash** is working beautifully, maintaining 50 t/s at 50k context, and encouraged others to rebuild to leverage the fix, using the `--flash-attn on` flag to enable it.
   - One user shared their launch command `llama-server -m ... -fa on ...`.
- ****Unsloth's Treasure Map**: Charting the Course to Revenue**: A community member asked how Unsloth AI intends to make money, despite releasing free optimized models and the Unsloth team confirmed they intend to release a paid product later this year, and emphasized that open source will remain their core focus.
   - The goal is to *do something important well and a market will materialise*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1463986909528326347)** (235 messages🔥🔥): 

> `AI Detector Bypass, Tailscale VPN, Attention Sink, Reasoning Model` 


- **LoRA not handling AI Detector Bypass, Watermark Removal & Chat Template Style perfectly**: A member shared that **LoRA** may not perfectly handle tasks like removing watermarks to bypass AI detectors, memorizing datasets, erasing AI personas, changing chat templates, breaking moral compasses and learning new languages for models like **Qwen 3 VL 4B**.
   - They explained that models like **OpenAI & Gemini** have watermarked outputs, so if they train it on synthetic data, *I need to break it first*.
- **Handwriting a Reasoning Cold Start Dataset is Painful**: A member expressed their self-inflicted pain of handwriting a reasoning cold start dataset instead of relying on **LLMs** because *it’s something LLMs suck at*.
   - They aim to create **500-1k** examples but are only chipping away at **5-10** examples a day.
- **LLMs Develop Blind Spots for Certain Tokens, Attention Sinks Arise**: A member mentioned that **LLMs** are prone to developing blind spots for certain tokens, to which another member responded that they use **<|endoftext|>** at the beginning of the context window as an *attention sink*.
   - However, it was contested that *thats not how it works*, and LLMs develop their own attention sink; *Attention sink is being poured into the very first token in the entire context window. Just one token for it. That’s it.*
- **Tailscale VPN is a self-hosted solution that is amazing**: Several members discussed **Tailscale** as a VPN solution, one described it as *the best* and another said *its qol is just outstanding, it effectively feels the same as the one click vpn apps, but tailscale is just your own [one click] vpn with your own installed attached devices* with a link to the [Tailscale website](https://tailscale.com/).
   - One member also mentioned the alternative **Headscale** for fully self-hosting and linked to its [site](https://headscale.net/).
- **Masking Thinking Improves Training of Smaller Reasoning Models**: A member inquired about using a smaller model to train a bigger model for reasoning, using the small model's output, but masking the thinking part.
   - Another member concurred that masking the thinking leads to better results, sharing they *learned that the hard way, did not mask thinking thought would help interpretability, f1 went down the drain*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1464050407457296587)** (17 messages🔥): 

> `Docker Run Parameters, Model Performance, Multi-GPU Support, QAT for Vision Models, Fine-tuning Models` 


- **Docker run parameters for Unsloth**: A member shared their [docker run command](https://www.docker.com) for running Unsloth with **CUDA**, specifying parameters like `--gpus=all`, `--parallel 1`, and `--hf unsloth/GLM-4.7-Flash-GGUF:Q6_K_XL`.
- **Improving Qwen3 Coder Model Performance**: A user reported improvements in a model's answers after analysis with Claude, but noted it's still below **Qwen3 coder** in accuracy.
   - They aim to use the model for daily tasks but need to ensure it provides accurate answers and mentioned **Qwen3** is working well with the **Qwen agent**.
- **Multi-GPU Training Support Questioned**: A user encountered an **OOM** error while loading **unsloth/Qwen3-VL-235B-A22B-Instruct**, leading to a discussion on multi-GPU support in Unsloth.
   - A member suggested Unsloth primarily supports single-GPU training, while another shared a link to [Unsloth's multi-GPU training documentation](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth), clarifying that *a more official implementation is in works!*
- **QAT Application with Vision Models is Unclear**: A member inquired about applying **QAT** to a model with vision capabilities, specifically using **gemma**, and reported a failure when adding the `qat_scheme` option during training.
- **Dataset Creation**: A user had questions about dataset formatting when a *subprocess has abruptly died during map operation*
   - The member used **Misumi Japan** scraping as an example, using both string and conversational formats.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1464012860442284032)** (11 messages🔥): 

> `GRPO Instability, Synthetic Data Generation, Customer Service Emphasis, Multi-Stage Training` 


- **GRPO Experiments Prove Unstable**: After running **GRPO** experiments, a user found it to be unstable, questioning its efficacy in niche domains where small LLMs might lack sufficient pretraining data and suggesting that **GRPO** might be more helpful for refining existing problem-solving abilities rather than enabling emergent reasoning.
   - They referenced the [DeepSeek paper](https://arxiv.org/abs/2601.07568) to challenge its claims about emergent reasoning.
- **Synthetic Conversation Data**: Multiple users discussed the use-cases for generating synthetic "human" conversation data, with one user noting its applicability for creating NPC voices in-game.
   - Another user mentioned using it to create synthetic data for an AI that translates to Turkish, catering to local services in Turkey.
- **Emphasis on Customer Service**: A user expressed disappointment that a preprint seemed to emphasize customer service as a major use case for the technology.
   - The user found this disappointing but noted it made sense given the industry's major use case.
- **Multi-Stage Training Pipeline Outlined**: A user described a three-stage training pipeline involving CPT on a novel + medical article corpus (~400M tokens), instruction tuning SFT with translated instruction datasets, and polishing SFT with synthetic data cleaned via rejection sampling.
   - The user is running a three-stage training pipeline to create data to translate to Turkish, catering to local services in Turkey.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1463987126029778954)** (462 messages🔥🔥🔥): 

> `Composer issues, Cursor 2.4 performance issues, Sub-agents, Usage and Billing Discrepancies, Feature Requests: GLM-4.7 Integration, Custom Model URLs` 


- **Composer Crashes Cursor**: Users report ongoing issues with **Composer 1**, including it being *completely broken* and causing an *endless loop* in chats, prompting suggestions to [report bugs on the forum](https://forum.cursor.com/c/support/bug-report/6).
   - Some users are downgrading to version **2.3** as a workaround, especially for older macOS versions like **Big Sur 11.7.3**, as later versions may not be supported.
- **Cursor 2.4 Faces Laggy Lashings**: Users are reporting significant **lag and unresponsiveness** in Cursor version 2.4, with constant crashing and lagging even on *high-end PCs*, with the message "planning next moves" hanging indefinitely.
   - Some suspect subagents default to **Composer 1**, which is slower, and suggest Cursor is releasing new versions before they're fully ready, referencing [Mr. Bean waiting](https://giphy.com/gifs/bombaysoftwares-waiting-mr-bean-still-um2kBnfo55iW4ZH1Fa).
- **Sub-agents Spark Strategic Scaffolding**: Users discovered partial subagent functionality, with Cursor injecting a **<subagent_delegation_context>** prompting the agent to *call the Task tool*, which is actually missing.
   - It's likely an *incomplete feature rollout* where the prompt-injection logic tests the LLM's context handling before stabilizing or exposing the actual backend tool.
- **Usage Under Scrutiny**: Users report wildly varying usage and **billing discrepancies**, including not seeing dollar amounts, not hitting limits despite heavy use, and unexpected bonus credits.
   - Some suggest the **Auto** mode is **charged incorrectly**, and others recommend [3rd party token watch](https://token-watch.vercel.app/) for more detailed tracking, noting it could affect Cursor website usage display.
- **Feature Fantasies: GLM-4.7 Glorified**: Users clamor for better integration of **GLM-4.7** within Cursor, suggesting a native integration or allowing custom URL overwrites per model, and referencing multiple requests already made on the [Cursor forum](https://forum.cursor.com/t/custom).
   - Some suggest intercepting HTTP events to trick Cursor into using desired models, but acknowledge account ban risks and discuss reverse-engineering Cursor with **Gemini 3 Pro**.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1463993849092509901)** (1 messages): 

> `Cursor 2.4, Subagents, Image Generation, Parallel task completion` 


- **Cursor 2.4's Subagent Saga**: The latest **Cursor 2.4** release introduces the use of **subagents** to complete tasks in parallel, enhancing both execution speed and context utilization.
   - These **subagents** facilitate faster overall execution, improve context usage, and enable agents to tackle longer-running tasks as detailed in the [Changelog](https://cursor.com/changelog) and demonstrated in an [attached video](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4?ex=69752b85&is=6973da05&hm=50e3cbf6432112dcbe36b0315b1645fd7d856c9d2ead97e639b2d8abcfa5b8f4&).
- **Cursor can now conjure Images!**: The update also equips **Cursor** with new capabilities, including **image generation** and the ability to ask clarifying questions, broadening its utility.
   - More details about this release can be found on [X/Twitter](https://x.com/cursor_ai/status/2014433672401977382) and [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7420199327010197504).


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1463990632946073630)** (386 messages🔥🔥): 

> `Nano Banana Pro 2K removal, Video generation issues, Captcha problems, Image editor from LMArena code, Seedream-4-2k missing` 


- **Nano Banana Pro 2K Gets the Boot Due to High Error Rate**: The **Nano Banana Pro 2K** model has been temporarily removed due to a high error rate, while the team works on resolving the issues, according to [a moderator](https://discord.com/channels/1340554757349179412/1340554757827461211).
   - Users expressed disappointment, with one stating, *"2K was the best. The 1K one is so bad,"* while others speculated about the model's return and possible cost issues.
- **Video Generation Glitches Plague Users**: Users reported issues with video generation, including videos not generating, captcha problems, and the "something went wrong" error message appearing frequently.
   - One user shared that they have been experiencing this issue a lot recently and their video still hasn't generated after 20 mins, also other users experiencing captcha problems only on Linux, suggesting a platform-specific bug.
- **Image Arena Code used to Create Image Editor**: A user shared that you can try his image editor that is made from LMArena code arena using puter.js, posted in <#1344733249628541099>.
   - The team has been testing different features.
- **Seedream 4 2k Model Vanishes From List**: A user reported that the **Seedream-4-2k** model is missing from the list, with only Seedream 3, Seedream 4.5, and Seedream 4 high res fal available.
   - A moderator responded that models may occasionally be unavailable for technical or other reasons, stating that it's *possible models may occasionally be unavailable for technical or other reasons.*
- **WAN 2.6 split into image and text, has problems**: **WAN 2.6** has been split into `wan2.6-image` and `wan2.6-t2i`. `wan2.6-image` is image-edit only, meaning it needs an image uploaded to work.
   - The situation is a bit odd, *for `wan2.6-t2i` doesn't have image upload available*. This is a problem the team is aware of.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1463992131424092203)** (3 messages): 

> `glm-4.7-flash, Image Edit Leaderboard, Single-Image Edit, Multi-Image Edit, wan2.6-t2i` 


- **GLM Gets Speedy with Flash**: A new model, **glm-4.7-flash**, has been added to the [Text Arena](https://lmarena.ai/?chat-modality=chat).
- **Image Edit Arena Splits Leaderboard for Clarity**: The Image Edit Arena leaderboard now features distinct rankings for **Single-Image Edit** and **Multi-Image Edit** tasks, providing a more precise view of model capabilities, check out the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit/overall).
   - For example, **ChatGPT Image (Latest)** fell from #1 to #3, while **Gemini 3 Pro Image 2K** rose from #2 to #1.
- **Arena Adds Wan and Devstral Models**: New models added to the arena include [Text-to-Image](https://lmarena.ai/c/new?chat-modality=image) model **wan2.6-t2i**, [Image Edit](https://lmarena.ai/c/new?chat-modality=image) model **wan2.6-image**, and [Code Arena](https://lmarena.ai/c/new?chat-modality=code&mode=direct-battle) model **devstral-2**.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1463995562037743636)** (315 messages🔥🔥): 

> `JanitorAI Lores and Timelines, Graphics Card Prices, Claude Opus 5 on Cheapest PC, Qwen 30b-a3b on 16GB, Thinking Box Issues on OR` 


- **JanitorAI Lore deep dive**: A user shared a [link to a FinanceLancelot tweet from 2014](https://x.com/financelancelot/status/2014258790355386737) related to **JanitorAI's lores and timelines**.
   - The user expressed that the main boards are no longer fun and that they mostly visit **/g/** and **/x/** boards because they are still somewhat active.
- **Graphics Cards Prices Spike!**: Users are discussing the high cost of graphics cards, one user mentions an **AUD$1599** price tag for a **5070Ti**, while another recalls the **Bitcoin crisis** influencing their reluctance to buy.
   - A user provided links to [staticice.com.au](https://staticice.com.au/cgi-bin/search.cgi?q=5070ti&spos=3) and [CCPU](https://www.ccpu.com.au/show_cat.php?cat_id=video) for checking deals, joking about the latter's website design.
- **Users grapple with OpenRouter 'Thinking Box' issue**: Users are reporting that the **<think> part of responses is no longer being hidden**, leading to a support ticket being created.
   - Some users are seeing the **reasoning box** appear on OR Chat, while others, using Janitor, are seeing it displayed *in* the response, a change that started recently.
- **Quest for Uncensored Image Generation with OpenRouter**: A user inquired about an **uncensored OpenRouter LLM** capable of image in to image out tasks, preferably with text output as well.
   - The response suggests that they should use a one **LM** and one **image model**.
- **OpenRouter Gacha Dream Debuts**: Users jokingly requested an **OpenRouter gacha system**, suggesting features like pity mechanisms and ranked competitive leaderboards.
   - A user even humorously detailed their experience with pulling all 5 constellations for their **ChatGPT build**, saying they spent a *few hundred dollars*


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1464017845779304538)** (3 messages): 

> `Cerebras CS3, SRAM, Linker Script` 


- **Cerebras CS3 houses 41gb SRAM per instance**: Each **Cerebras CS3** instance is equipped with **41GB of SRAM**, with the capability to connect up to **2048** such instances together.
- **Linker Script GIF surfaces**: A member shared a [GIF](https://cdn.discordapp.com/attachments/1398923576186245251/1439357031172538652/image.gif) related to a linker script.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1463987330892304596)** (283 messages🔥🔥): 

> `Gemini & calcium deficiency tests, Perplexity Pro's file upload limits, Perplexity billing issues, Perplexity A/B testing, Saturn, Neptune, and Moon alignment` 


- **Gemini Skips Calcium Bone Density Check**: Members found that **Gemini**, **Opus** and **Ernie** sidestepped crucial bone density checks when asked about testing for calcium deficiency, while **GPT** addressed it correctly, noting that a DEXA scan is necessary to rule out calcium leaching from bones.
   - It was noted that a blood test alone can be misleading because blood pulls calcium from the bones, potentially masking true calcium deficiency.
- **Pro Users File Uploads Halted**: Perplexity Pro users are experiencing issues with **file uploads**, with many reporting a new daily limit of three uploads despite having active subscriptions, leading to frustration and consideration of unsubscribing.
   - Some users speculate that the upload limits are due to A/B testing or potential financial problems at Perplexity, while others believe the company is intentionally limiting features to push users toward direct payments.
- **Billing Blocks Bad for Business**: Users are encountering **401 unauthorized** errors and pending charges after credit renewal, preventing them from using the API for their projects and experiencing difficulties in resolving the issues due to unresponsive support.
   - One user shared frustration with a sports betting model, noting that lack of customer service and fast fixes are bad for business.
- **Pro Tier Gets the Squeeze?**: Perplexity Pro users are reporting inconsistent experiences, with some encountering a 600 daily research query limit while others are restricted to 20, sparking speculation about **A/B testing** or visual errors in the displayed limits.
   - Members are speculating that the 600 limit may only apply to regular searches and the pro-tier benefits are slowly being eaten down.
- **Celestial Smiley Frowns on Reality**: A user shared an image claiming **Saturn**, **Neptune**, and the **Moon** would converge to form a smiling face, which was debunked by another user who demonstrated the actual alignment and noted that such an image is not astronomically possible.
   - The discussion devolved into personal attacks, with one user accusing the other of spreading misinformation and lacking imagination.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1464341850809831519)** (1 messages): 

> `Perplexity API, API Key, 401 Error, Sport Betting Model` 


- **API Key throws 401 after credit renewal**: A member reported receiving a **401 error** with their **Perplexity API key** after renewing their credit for a sport betting model.
- **Troubleshooting the Perplexity API 401 Error**: After renewing credits, one user is locked out of the **Perplexity API** and continues to receive a **401 error**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1464061472203210812)** (106 messages🔥🔥): 

> `AMD AI Bundle Review, GitHub Copilot SDK Release, LM Studio with Claude code, Langchain API Overhaul, Choosing an LLM for specific tasks` 


- ****AMD AI Bundle** faces scrutiny**: A user shared a [review of the AMD AI Bundle](https://www.techpowerup.com/review/amd-ai-bundle) expressing disbelief.
   - The review discusses the performance and integration of **AMD's AI** solutions, including **CPUs**, **NPUs**, and **GPUs**.
- ****GitHub Copilot SDK** hits the streets**: A user shared the link to the [GitHub Copilot SDK](https://github.com/github/copilot-sdk), celebrating the newfound freedom from **OpenRouter's pricing**.
   - The **Copilot SDK** enables developers to build AI-powered features into their applications, leveraging **GitHub's AI** infrastructure.
- ****LM Studio** integrates with **Claude****: Users discussed integrating **LM Studio** with **Claude** code to leverage local models and potentially offset token costs.
   - One user suggested using **Opencode** as a *dead simple claude code clone*, noting it runs well, and another user mentioned they're on mac with **48gb ram** and can run **GLM4.7** local no worries on **6bit** version.
- ****Langchain API** gets simplified**: A user highlighted that **Langchain** recently underwent an **API overhaul**, making it simpler to use for building agents powered by **LMStudio** or **Ollama**.
   - They recommended revisiting **Langchain/Langgraph** and mentioned using the **TS version** for a **CLI agent**, and also recommends **gpt-oss-120b MXFP4**.
- **LLM choice depends on the use case**: One user sought advice on selecting an **LLM** that balances cost and intelligence for summarizing news headlines, checking for duplicates, and ensuring strict **JSON compliance**.
   - They're currently using **gpt oss 120b** but are looking for a cheaper local alternative, having found **Granite**, **Qwen**, and **Nemotron Nano** to be insufficiently accurate.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1464140724948959242)** (68 messages🔥🔥): 

> `Crypto bot scams, GPU fan configuration, AIO vs Air Cooling, 420mm AIO Rads, Passive cooling` 


- **The Curious Case of the $2700 Crypto Bot Scam**: A member questioned why crypto bot scams are always priced at **$2700**, with another suggesting that **27** is a number that writers use to sound random.
- **GPU Fan Direction Debacle**: A member pointed out that a GPU fan was mounted backwards, stating that *gpu fans push air out the rear I/O* suggesting to pull more air from the bottom of the case for better cooling.
   - Another user explained that pushing air directly into the cards would trap heat, but drawing fresh air from the front could improve cooling, while it was clarified that *if you are getting fine temps as is no need to change anything*.
- **AIO Coolers Take the Crown over Air Coolers**: A user stated AIOs are objectively quieter than air coolers, especially beyond **250W**, highlighting that modern CPUs spike in temperature even with light tasks like browsing, making AIOs better at compensating.
   - Another user countered that air coolers are sufficient for normal setups and quieter if run at constant speeds, but another user pointed out AIOs beat air coolers in temperature in [noise normalized graphs](https://cdn.discordapp.com/attachments/1153759714082033735/1464296768752844832/image.png?ex=6974f422&is=6973a2a2&hm=161e90a999260f112c73dd75b4f956f3ae9a7253e2a95ad5b3967a50f0947db2&).
- **420mm AIO Rad Enthusiasts Unite**: A member advocated for **420mm AIOs** over **360mm** ones, while another joked about building a case entirely out of **420mm AIOs**.
   - Someone replied that they have seen liquid cooling builds with **3x420 rads**.
- **Passive Cooling: The Ultimate Chill Pill**: A user claimed passive cooling is the best cooling method, while a member found that his **MX150 laptop** with **2GB VRAM** and **16GB system RAM** supports **CUDA** and can utilize up to **9.9GB** of total memory.
   - Another member mentioned that they can run *LoRA* adapters for a 1.5B model easily with large batch sizes for a laptop.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1463987609490686092)** (106 messages🔥🔥): 

> `GLM-4.7, DeepSeek R1, Qwen 2.5-14B, Ernie by Baidu, Qwen3-TTS Voice Cloning` 


- **Users explore GLM-4.7 and other Models**: Users discussed experimenting with running **monster models** like **GLM-4.7** on rented **GPU servers** (e.g., 8xH100) to evaluate their performance as Claude Code alternatives.
- **Nous API Praised despite Experimentation Urge**: Despite wanting to experiment with renting a GPU, one user considers *The Nous API* to be a *good deal* and had *don't have a single harsh word about the pricing*.
   - The mixed calculation and pricing of the Nous API was considered fair compared to **Hugging Face**, whose prices are too high.
- **Qwen3-TTS Impresses with Voice Cloning Prowess**: **Qwen3-TTS** is considered a very good *voice cloning* tool, rivaling **ElevenLabs**; a link to [Qwen3-TTS on Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS) was shared.
- **Training Models on Huawei hardware gains steam**: The trend of training frontier models on diversified, non-Nvidia hardware is gaining traction, mentioning that **Gemini** was trained on **Google TPUs** and **Zai GLM 4.7** on **Huawei hardware**, which was covered in a [YouTube video](https://www.youtube.com/watch?v=WU_rKAC_SLI).
- **Self-Replication benchmark for AI Agents is brewing**: A member is thinking about *a self-replication benchmark for agentic-ai* and is looking for advice on what a proper goal would be, one suggestion was to evaluate a transformer inference engine like the one implemented by Claude in C code which also designed a custom processor, available [on GitHub](https://github.com/cpldcpu/smollm.c/blob/claude/train-small-model-llxVr/smolc/smolc.c).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1463994096245932064)** (82 messages🔥🔥): 

> `Capital One acquires Brex, AI-Powered Pool Detection, Replit Agent Decision-Time Guidance, Fine-tuning on Code Base, Multi-Agent Communication` 


- **Capital One Swallows Brex for Billions**: Capital One acquired Brex for **$5.15B**, marking the largest bank-fintech deal in history, with [details available here](https://x.com/alexfmac/status/2014676950883668306?s=46).
- **SAM3 and Mapbox Expose Hidden Pool Party Locations**: Using Meta's **SAM3** model and **Mapbox** imagery, nearly 1,500 swimming pools were identified in a 10 sq km suburban area from a single text prompt, showcasing zero-shot geospatial intelligence.
   - Details are available on [Kyle Walker's tweet](https://xcancel.com/kyle_e_walker/status/2014433189423407194).
- **Replit Agent Navigates Complex Tasks**: Zhen Li detailed a technical blog post on Decision-Time Guidance in **Replit Agent**, exploring real-time control mechanisms instead of static rules.
   - This helps autonomous agents navigate complex, real-world tasks more effectively; [the blog post is available here](https://xcancel.com/zhenthebuilder/status/2014393451442581688?s=46).
- **Baseten Bags $300M, Valuation Rockets to $5B**: Baseten announced a **$300M Series E** funding round led by IVP and CapitalG, reaching a **$5B valuation**, with participation from NVIDIA and other venture firms; [details on Baseten's tweet](https://xcancel.com/basetenco/status/2014755013344792595?s=46).
- **ChatGPT Still Dominates, Grok Sees Explosive Growth**: Data from SimilarWeb shows **ChatGPT** leading the AI platform market despite saturation, while **Grok** experienced the most significant growth with a **33x** increase in monthly unique visitor penetration in the U.S.
   - More insights can be found in the [Venture Twins tweet](https://xcancel.com/venturetwins/status/2014739492389978274?s=46).


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1464139497808859215)** (1 messages): 

> `Memory Usage, Payment Plan` 


- **Splurging on RAM Stops Swapping**: A member stated that they have a **96gb** RAM and got a *hall pass/payment plan* for it.
   - They added that they *never looked back* as a result.
- **Modern Software Hogs Memory**: A member wondered if their machine was swapping to disk due to excessive memory usage.
   - They commented that a lot of modern software is a memory hog and they *cry when they open task manager*.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1464064002094006376)** (15 messages🔥): 

> `Qwen3-TTS Release, Image Edit Quality Degradation, Audio AI Model Releases` 


- **Alibaba Opens Qwen3-TTS Treasure Trove**: Alibaba has released **Qwen3-TTS**, an open-source text-to-speech family featuring **VoiceDesign**, **CustomVoice**, and **Base models**, available on [GitHub](https://github.com/QwenLM) and [Hugging Face](https://huggingface.co/Qwen) with five models ranging from **0.6B** to **1.8B** parameters, supporting **10 languages** and high-quality voice cloning.
- **Iterative Edits Induce Image Impairment**: Repeated image edits with **Flux.2 Klein 9B** cause **progressive saturation shifts and quality degradation** when the output image is fed back in for further edits, as discussed in this [reddit thread](https://old.reddit.com/r/comfyui/comments/1qkgc4y/flux2_klein_9b_distilled_image_edit_image_gets/).
- **Audio AI Arena Adds Achievers**: Lina Colucci highlights three major audio AI releases: **NVIDIA's PersonaPlex-7B** full-duplex conversational model, **Inworld AI's low-latency TTS-1.5**, and **Flash Labs' Chroma 1.0**, the first open-source end-to-end speech-to-speech model, in [this post](https://x.com/lina_colucci/status/2014229002370834861?s=46).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1464201720774004869)** (11 messages🔥): 

> `Transformer AGI?, Pangram performance, NeRF research, ROCm software` 


- **Transformer Architecture Leading to AGI?**: Members discussed whether the **Transformer architecture** is capable of achieving **AGI**, contrasting it with approaches like **neuro-symbolic** or **JEPA** architectures.
- **Pangram Detector Impresses**: A member inquired about **Pangram's** performance relative to other detectors, and another user stated that *Pangram is the most impressive one by a significant margin* according to their experience with [this GitHub repo](https://github.com/adithya-s-k/manim_skill).
- **NeRF Research Efforts**: A member inquired whether anyone is actively engaged in **NeRF (Neural Radiance Fields) research**.
- **ROCm Software's ML Performance**: A member asked about the current performance and reliability of **ROCm software** for accelerated **ML** using **GPUs**.
   - Another user responded that **ROCm** *has made a lot of strides in terms of usability*, but can still be challenging due to primary support being for **Nvidia**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1463991483626426474)** (65 messages🔥🔥): 

> `Equi Parameter Sequential vs Parallel Attention Blocks, Marin's experiments with attention blocks, Difference Layer in Nanogpt, Improving Nanogpt Baseline, Multiplicative Nets and Logic Prior` 


- **Attention Blocks Face-Off: Sequential vs Parallel**: A member inquired about a reference comparing equi parameter **sequential vs parallel attention blocks** in transformers.
   - Another member pointed to **Marin's experiments** as a relevant resource.
- **Difference Layer Shows Promise in Nanogpt**: A member shared code for a *difference layer* `x = (self.a2(x) - self.b2(x)) * (self.c2(x) - self.d2(x)) + self.e2(x)` from [Eternalyze0/difference_layer](https://github.com/Eternalyze0/difference_layer), claiming it performs significantly better on **cartpole** and **nanogpt** while using fewer parameters and compute.
   - This architecture doubles the effective learning rate in SGD, which could explain the reported improvements relative to poorly optimized baselines.
- **Leveling Up Nanogpt: Swiglu Activation**: A member suggested enhancing the **Nanogpt baseline** by switching the activation function from **GELU** to standard **SwiGLU**, which led to a performance boost.
   - In another experiment, replacing the **QKV linear** layer with a difference layer also widened the performance gap compared to the baseline.
- **Optimize Baselines for Meaningful Research**: Experienced researchers emphasized the importance of using strong, optimized baselines in experiments to avoid mistaking noise for actual improvements.
   - They recommended using **modded nanogpt** for language tasks and staying current with literature or consulting experienced researchers to determine appropriate baselines.
- **Multiplicative Nets: Logic Prior**: A member argued that **multiplicative nets** have a higher logic prior, because its natural gating.
   - Another member pointed out that [Noam Shazeer's GLU variants paper](https://arxiv.org/abs/2002.05202) already addresses this, emphasizing that experiments must start with strong baselines.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1464015531492376800)** (3 messages): 

> `MATS/AFP follow up paper, Compression-Lens HF space` 


- **MATS/AFP Follow-Up Paper in the Works**: A member announced they are working on a follow-up paper for **MATS/AFP** while Christina preps the paper for ICML, and seeks collaborators familiar with the original paper's ideas.
   - They referenced a [previous solicitation](https://discord.com/channels/729741769192767510/730095596861521970/1462609593703207175) for collaborators.
- **Compression-Lens Hugging Face Space Shared**: A member shared a [Hugging Face Space link](https://huggingface.co/spaces/Jellyfish042/Compression-Lens) related to the **Compression-Lens** technique.
   - No further context was provided about the purpose or relevance of the shared space.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

aeros93: https://fixupx.com/havenfeng/status/2014765400563781777?s=46
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1463991249349640460)** (55 messages🔥🔥): 

> `Claude's Laziness vs GPT-Codex Grunt Work, Yann LeCun's Startup and Energy-Based Models (EBMs), Adversary Review, EBMs vs Diffusion Models, URM paper (based on tiny recursive models)` 


- **Claude vs GPT-Codex**: A member shared their experience with **VSCode Copilot**, noting that **Claude** seems *lazy* and makes assumptions, whereas **GPT-Codex** does a lot of *grunt work*, which can be both helpful and distracting.
   - The member did not view either as a *fair harness*.
- **LeCun's Startup Debuts with Sudoku**: Members discussed [Yann LeCun's new AI startup](https://www.reddit.com/r/agi/comments/1qjzdvx/new_ai_startup_with_yann_lecun_claims_first/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) claiming a breakthrough with **Energy-Based Models (EBMs)**, showcasing its ability to solve Sudoku puzzles.
   - Some expressed skepticism, noting the lack of details on architecture, model size, and training, while others pointed out that **EBMs solving Sudoku** is already a known capability and wondered why [LeCun hadn't mentioned it on social media](https://energy-based-model.github.io/ired/ired.pdf).
- **Adversary Review Needed?**: The community discussed the concept of *adversary review* in academic publishing and its potential challenges, such as reviewers needing to be subject matter experts and potentially being competitors.
   - One member suggested that **AI** could potentially perform adversarial reviews with a reward function that incentivizes pointing out errors, but they conceded the reviewers needed time and incentives to thoroughly proofread papers.
- **EBMs are worse ways of doing essentially the same thing that diffusion models do**: A member provided a detailed explanation of **Energy-Based Models (EBMs)** and how they relate to diffusion models, arguing that EBMs are essentially a worse way of achieving the same results.
   - They pointed out that diffusion/score matching/flow matching are all the same.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1464282397141893313)** (2 messages): 

> `RLM framework, LLM agent setting, REPL` 


- **Does RLM Resemble REPL in LLM Agents?**: A member inquired whether the **RLM framework** resembles adding a **REPL** to an orchestrator in a classic **LLM agent setting**.
   - Another member outlined differences: **RLM** can spawn sub-LMs, uses the same system prompt across base and sub-LMs, holds prompt/context in the environment, and doesn't heavily rely on tool use beyond **REPL** and calling sub-LMs.
- **RLM vs LLM Agent Orchestrators**: **RLM** differs from **LLM agent orchestrators** by being able to spawn sub-LMs instead of depending on a human-created workflow.
   - Unlike agent orchestrators with different "personas", **RLM** base LM & sub-LMs share the same system prompt, and **RLM** does not heavily rely on tool use beyond **REPL** and sub-LM calls.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1464076640001720454)** (13 messages🔥): 

> `Vibe Coding, OpenAI Cash Runoff, Open Source Frontier Model` 


- **Attack Vibe Coding Shops and Profit?**: A member shared a tweet suggesting to find companies bragging about **Vibe Coding**, break their systems, and then profit or disclose vulnerabilities ([source](https://x.com/yuvalavra/status/2011842613389726109)).
- **OpenAI Could Face Cash Crunch by 2027**: A member shared an article indicating that **OpenAI** could potentially run out of cash by mid-2027, according to an analyst's grim financial assessment ([Tom's Hardware](https://www.tomshardware.com/tech-industry/big-tech/openai-could-reportedly-run-out-of-cash-by-mid-2027-nyt-analyst-paints-grim-picture-after-examining-companys-finances)).
   - Another member joked that *"they are too big to fail. We need to pre- and post- bail them out"*.
- **Open Source Frontier Model: No Bailout Needed?**: A member suggested that instead of a bailout, **OpenAI** should open source their frontier model and let people run it themselves ([source](https://fixvx.com/sama/status/2014733975755817267)).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1464023921648074814)** (10 messages🔥): 

> `Realistic physics engines and graphics, Specialized models for niche predictions (weather), Female counterparts with cat ears for self-stimulation, Profiling for automated coding, Upcoming GTC events` 


- **More Realistic Physics Engines are sought**: Members mentioned the need for **more realistic physics engines and graphics**, as well as **specialized models for niche predictions** like weather.
   - Additionally, they brought up the idea of systems representing female counterparts (*possibly with cat ears*) becoming more desired due to human social tendencies.
- **Automated Coding still needs Profiling**: It was mentioned that even as **coding gets more automated**, understanding **profiling** might still be helpful for better architecture and system design.
   - A member expressed liking **GPU programming**, hoping for more reasons to learn it, but felt software is moving towards architecture and system design.
- **GTC Events are getting sorted**: A member inquired about events at the **upcoming GTC** event and another member [responded](https://developer.nvidia.com/gtc) with planning an **awards ceremony for the nvfp4 competition** and maybe a **happy hour**.
   - One member is flying in, having sorted their **ESTA** to avoid missing their flight.
- **Text-to-3D-shape Model?**: A member asked if anyone has used or is using a **text-to-3D-shape model for a Rendering Engine**.
   - No other details were given.
- **MXFP8 with dynamic quantization**: A member inquired about using **mxfp8 with dynamic quantization**, noting it is used in NVIDIA libraries like TransformerEngine.
   - They asked if there are examples using **static mxfp8 scales for activations**, since the overhead of doing the max reduction on tensors for scales is large.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1464176271683752039)** (8 messages🔥): 

> `FA4 Peak Performance, NVIDIA B200 Specs` 


- **FA4 benchmarks hitting 71% of theoretical max**: FlashAttention-4 (**FA4**) achieved a peak performance of **1,605 TFLOPS/s** which is *71%* of the hardware's theoretical maximum on an **NVIDIA B200 GPU** using **BF16** inputs.
   - Members discussed the theoretical maximum being around **2260 TFLOPS**, however the exact specs are hard to come by.
- **NVIDIA B200 actual TFLOPS in question**: Some **B200** materials list performance at **10/5/2.5 TFLOPS** for **fp4/fp8/fp16** respectively, but official documentation is lacking.
   - Community members are awaiting an official paper with detailed specifications, especially considering the blog post made no mention of data types.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

costa5805: https://mlsys26.flashinfer.ai/
  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1464218961905451028)** (2 messages): 

> `CUDA, GPU performance optimisation, CUDA kernel optimization, Nsight` 


- **Job Opportunity: CUDA/GPU Performance Optimization**: An individual is looking for someone with a strong background in **CUDA** and **GPU performance optimisation** for a short-term, fully remote contract.
   - The role involves creating clear, well-structured, real-world scenario tasks based on experience with **CUDA** and **GPU optimisation**, with a focus on structured thinking and written reasoning.
- **Parsewave Seeks CUDA Kernel Optimization Engineers**: **Parsewave** is seeking engineers to write and optimize **CUDA C/C++ kernels**, diagnose bottlenecks using **Nsight Systems / Nsight Compute**, and explain optimization tradeoffs, applying via [this link](https://tally.so/r/pbDDvZ).
   - Ideal candidates should be familiar with **CUDA intrinsics** (especially Blackwell or Hopper) and be able to propose scenarios/benchmarks that clearly show **naive → optimized deltas**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1464000590437945562)** (6 messages): 

> `GPUMode 2026, CUDA PTX submissions, dual_gemm problem` 


- **GPUMode 2026 Roadmap Confirmed**: A member confirmed that the roadmap for **GPUMode 2026** is still planned for the near future, linking to the [GPUMode 2026 news article](https://www.gpumode.com/v2/news/gpumode-2026).
- **CUDA PTX Submissions Surge**: There are more **CUDA PTX submissions** in the latest **dual_gemm** problem than previously.
   - The member mentioned that they *should be making some broader announcements soon*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1464248891338653903)** (10 messages🔥): 

> `Compiler Intrinsics, ROCm vs CUDA, AMD Developer Ecosystem` 


- ****Intrinsic Insights**: Compiler Documentation Tips!**: Compiler intrinsics and builtins explained: **builtins** are a *clang* thing, **intrinsics** are an *llvm* thing, with builtins often translating 1:1 into intrinsics, aided by [AMD's ISA manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).
   - To find documentation, members recommended checking the [AMDGPU Usage](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst), [IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td), and the [AMDGPU CodeGen tests](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU).
- ****5090 Farewell**: Developer Bails on ROCm, Eyes CUDA!**: A developer, after buying a **5090**, bid farewell to ROCm citing *too many issues with packaging*, *build*, *distribution availability*, and a *lack of focus on consumer-facing kit*. 
   - They said there was a lack of ongoing meaningful hardware refreshes on the prosumer space.
- ****Conv3D Catastrophe**: NVIDIA Wipes the Floor!**: A member noted that mid-grade NVIDIA hardware significantly outperforms AMD gear on tasks like **Conv3D**, referencing [a Reddit thread](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/).
   - They complained about *inconsistent performance between hardware generations* and *regressions that go unaddressed for multiple major releases*.
- ****Hostile Horizons**: AMD's Ecosystem Draws Ire!**: A developer expressed frustration with the ROCm ecosystem calling it *hostile* citing issues such as the FBGEMM repository not building on gfx1100.
   - They pointed to a [Quark quantisation engine commit](https://github.com/amd/Quark/commit/9234960c951410abdcecee033adf610d7126fda3) as an example of poor communication and lack of contribution-friendliness.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1464028958059139186)** (6 messages): 

> `FlashInfer-Bench, CMU Catalyst Lab, AI-generated GPU kernels, Collaboration opportunities` 


- **FlashInfer-Bench driven by CMU Catalyst Lab**: Yixin Dong from **CMU Catalyst Lab** introduced **FlashInfer-Bench**, a framework for evaluating **AI-generated GPU kernels** and deploying them into serving engines.
   - The group is exploring ways to refine the benchmarks and collaborate within the community.
- **FlashInfer-Bench gets Kudos**: A member noted that *Flashinfer bench is a very cool project*.
   - The creators are looking forward to collaborating with the community, especially on evaluating and deploying generated kernels in production.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1464375553678119036)** (1 messages): 

> `Graphical Layout Calculus, Tuple Morphisms, Mutual Refinement, Prefix Products` 


- **Layout Composition Computed Graphically**: A member demonstrated how to compute the composition of two layouts by hand using **graphical layout calculus**, with attached images detailing the steps.
- **Layouts converted to Tuple Morphisms**: The initial step involves converting tractable layouts into **tuple morphisms** denoted as `m_A` and `m_B`.
- **Mutual Refinement Critical for Composition**: The process requires finding a **mutual refinement** of the two tuple morphisms, which is essential for composing the layouts.
- **Pulling Back and Pushing Forward Refinements**: The **mutual refinement** is pulled back along `m_A` to obtain `\hat{m}_A` and pushed forward along `m_B` to obtain `\hat{m}_B`.
- **Result Written as Prefix Products**: The final result is expressed as a layout utilizing **prefix products** of the codomain tuple after composing `\hat{m}_B o \hat{m}_A`.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1464345228696617021)** (3 messages): 

> `SITP Curriculum, GEMM Kernels on ARM, SVGBob for diagrams` 


- **SITP Curriculum Praised for Accessibility**: A member expressed happiness with the ordering of sections **1.1** and **1.2** of the [SITP curriculum](https://j4orz.ai/sitp/1.html) due to its accessibility.
   - They suspect that the material might be too basic for the Discord's audience, but highlights that these specific sections make the ramp-up accessible enough for a high schooler to read.
- **GEMM Kernels Set to Conquer ARM**: One member is planning to implement **GEMM kernels on ARM** for sections **1.3**, **1.4**, and **1.5** of the curriculum.
   - The kernels will build upon the accessible introduction provided by the earlier sections.
- **SVGBob emerges as diagramming darling**: A member likes using **SVGBob** over **tikz**, **mermaidjs**, or **graphviz** for diagrams.
   - They explained that **SVGBob** lets them generate diagrams as ascii text rather than a DSL like dot, and it also embeds into Compiler Explorer.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1463989990374510738)** (1 messages): 

> `Test-Time-Training, LM-generated Kernels, Model Performance, TTT Results` 


- **TTT for LM-Generated Kernels**: Researchers evaluated **LM-generated kernels** using **test-time-training (TTT)** on past leaderboards and got interesting results as described in this [paper](https://test-time-training.github.io/discover.pdf).
- **LM Kernels Ace Test-Time Training**: A recent study highlighted on **test-time-training.github.io** showcases promising outcomes for **LM-generated kernels** evaluated with **TTT** on established leaderboards; the detailed findings are available in the linked [PDF](https://test-time-training.github.io/discover.pdf).


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1464249056510349427)** (5 messages): 

> `NCCL Benchmarking on Slurm, B200 Performance Tuning, Slurm sbatch script` 


- **Slurm Script Struggles Spark NCCL Debugging**: A user is experiencing unexpected performance issues with **NCCL** benchmarking on **Slurm** using **B200s** and shares an [sbatch script](https://gist.github.com/example/12345) for review.
   - A member suggested checking the output of `NCCL_DEBUG=INFO` and trying the benchmark without setting environment variables, as *NCCL generally auto-tunes well*.
- **GPU performance on Slurm**: A user seeks assistance with **NCCL** benchmarking on **Slurm** for **B200s**, suspecting a configuration issue.
   - The user has provided their `sbatch` script and is looking for potential problems in their setup.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1464407381680062630)** (12 messages🔥): 

> `Competition, Team Merging, Multiple Tracks, Registration Confirmation` 


- **Competition Updates Encourage Posts**: Admins encouraged others to create a short post in the general channel describing the competition goal, due to expected interest in the new competition.
   - They suspected people would be especially interested in the new competition.
- **Team Merging Okayed Pre-Deadline**: Team merging is allowed before the registration deadline, participants were instructed to inform the admins if they merge.
   - Admins also indicated that they can shift tracks later.
- **GPU Prizes Limited Despite Multiple Track Participation**: Participants can partake in multiple tracks, but may only receive one GPU prize if they win multiple tracks.
   - One user clarified that they can shift tracks later if they want.
- **Registration Confirmation Emails Automated**: Following a suggestion to automate registration confirmation emails, the admin confirmed they have set it up.
   - The goal is to avoid duplicate registrations.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1463992048939044988)** (33 messages🔥): 

> `LazyMergeKit Interruptions, Agent Course Channel Information, Llama.cpp GLM 4.7 Flash GGUFs speed boost, Fine-tuning a pre-trained model, Video Face-swapping API Discussions` 


- **LazyMergeKit plagued by Interruptions**: A member faced interruptions using **LazyMergeKit** for merging models, a problem they hadn't encountered before, and posted a screenshot of what seemed like space was paused by an admin.
   - They believed it was pinned, and also deleted and re-uploaded into the same name space, and the *deactivation* persisted.
- **Llama.cpp achieves GLM 4.7 Flash GGUFs Speed Nirvana**: **Llama.cpp** made **GLM 4.7 Flash GGUFs** around *1.5x faster* and fixed bugs, urging users to rebuild **llama.cpp** and re-get the fixed quants from [here](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF).
   - This represents a significant performance boost for those utilizing **GLM 4.7 Flash GGUFs** within the **llama.cpp** framework.
- **Fine-tuning Frenzy on the Horizon**: A member inquired about the process of fine-tuning a pre-trained model, asking if it could be done in **Google Cloud** or **Kaggle**, with a link to [Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide).
   - One member suggested converting to **ONNX** or **GGUF** for client-side execution, linking examples like [LFM2.5-VL-1.6B-WebGPU](https://huggingface.co/spaces/LiquidAI/LFM2.5-VL-1.6B-WebGPU) and [SmolVLM-256M-Instruct-WebGPU](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-256M-Instruct-WebGPU).
- **Space Odyssey: HF Space Deactivated**: A member reported that their space was pinned and deactivated, despite having the most traffic, leading to the removal of a corresponding **X** post, and the space can be found [here](https://huggingface.co/spaces/tostido/Cascade-Hyperlattice).
   - They suspect the services offered in conjunction with the capabilities on a separate page may have caused the deactivation, even after removal, and is considering the namespace *caput*.
- **RAG Time: Tutorial Hunt**: A member is seeking a concise tutorial or guide on building a **RAG (Retrieval-Augmented Generation)** system that can be hosted entirely locally.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1463998594821521468)** (12 messages🔥): 

> `wasm synthetic dataset, typeface dataset, agentic AI updates, German-first LLM, safety dataset` 


- **WASM Dataset for Reproducible Rust-to-WebAssembly Compilation!**: A fully synthetic dataset containing metadata from **1,000 programmatically generated Rust programs** designed to compile (or fail) to **WebAssembly** is available at [HuggingFace](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset).
   - All samples were created using a deterministic Fibonacci-derived pseudo-random generator, producing reproducible variations in code patterns, source code length, number of exported functions, and structural hashes.
- **Typeface Dataset Generated for YOU**: A member created an app that generated a **typeface dataset** and is offering custom datasets tailored to individual or team needs. DM for details.
   - More information available at [webxos.netlify.app/COLIGNUM](https://webxos.netlify.app/COLIGNUM).
- **Stay up-to-date on Agentic AI**: Daily high-signal updates on agentic AI, RAG, LLMs, production tools, orchestration, governance and real-world deployments — distilled for senior engineers can be found at [x.com/when_robots_cry](https://x.com/when_robots_cry).
   - An Antigravity like browser tool can be found at [https://mcp.so/server/browser-control/adityasasidhar](https://mcp.so/server/browser-control/adityasasidhar)
- **Faust-1: A German-first LLM Released**: **Faust-1**, a **1.6B** parameter German-first large language model trained from scratch, has been released at [HuggingFace](https://huggingface.co/tabularisai/Faust-1).
   - It features German-dominant pretraining (≈90%), a custom tokenizer optimized for German, verified synthetic data + instruction tuning (DPO), and is designed for local/privacy-sensitive deployment.
- **Safety Dataset Released**: A **safety dataset** has been released on [HuggingFace](https://huggingface.co/datasets/Pacific-Prime/safety_dataset).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1464083736999235719)** (3 messages): 

> `Agent Course, Robotics Course, HuggingFace Tutorial` 


- **Agent Course Channel Elusive**: A new Discord user expressed difficulty finding the channel for the agent course.
   - They expressed interest in joining the community.
- **Robotics Course Modules Inquiry**: A user inquired about the location of remaining modules for the new robotics course, noting that only two modules have been released.
   - They need to study the course urgently for their graduation project.
- **HuggingFace Tutorial Suggested**: A member suggested visiting the [Hugging Face Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) to study robotics.
   - They stated that *this is the right choice*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1464008161705853009)** (22 messages🔥): 

> `Senior AI Engineer introduction, AI Agent Developer looking for gigs, Full stack AI Engineer introduction, Manus Unauthorized Billing, Dracoai.app` 


- **Senior AI Engineer Introduces Expertise**: A member introduced themself as a **Senior AI Engineer** with over 7 years of experience, specializing in building scalable, cloud-native **AI systems** used in real-world production environments, with expertise in **deep learning, NLP, computer vision, and multimodal AI**.
   - They're especially excited to collaborate on projects where **AI performance, reliability, and real-world impact** truly matter.
- **AI Agent Developer Seeks Collaborations**: A member highlighted their expertise in building **AI agents** for various applications, including **customer support, workflow automation, data analytics, and autonomous booking**, emphasizing a focus on production-grade systems rather than just demos.
   - They focus on **tool orchestration, deterministic outputs, long-running agent state management, and optimization of latency, cost, and failure modes**.
- **Full-Stack AI Engineer Promotes Services**: A member advertised their skills in building **AI + full-stack systems** focusing on delivering real value and improving efficiency, accuracy, and user experience, with expertise in **LLM integration, workflow automation, AI content detection, image AI (CLIP + YOLOv8), and voice AI (Whisper, Tacotron2)**.
   - Their services include full-stack development with **React, Next.js, Node.js, Laravel, Django, Flutter, React Native, and hybrid on-chain/off-chain AI/service orchestration**.
- **User Reports Unauthorized Billing Incident**: A member reported being charged **$400** for an annual plan despite selecting monthly billing after a free trial and is having trouble with customer support.
   - They plan to file complaints with the **FTC, BBB, Attorney General**, and contact **Meta** if the issue isn't resolved, seeking advice from others who may have experienced similar issues.
- **User Discovers Draco AI phone calls?**: A member mentioned that they've tried [Dracoai.app](https://dracoai.app), noting the 'caller model' made a phone call to perform a task.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1464167385060872203)** (12 messages🔥): 

> `aider TUI support, aider session management, aider checkpoint, aider context management, aider complements Claude` 


- ****Aider** to get TUI Support?**: A member discussed implementing **TUI support** in *aider* to allow editing the next message while browsing replies and rendering aesthetically pleasing **Markdown** outputs like **mermaid diagrams**.
- ****Aider** with Session Management?**: A member suggested adding **session management** features to *aider* so users can temporarily store all chat content in one place, switch between chat contexts, and continue from a past message without polluting the context.
   - They also proposed **fine-grained context management** for removing useless or incorrect input/output from chat logs.
- **Pairing Aider with Claude is Meta!**: One member combines *aider* with **Claude code** for an efficient workflow: using *aider* for speed and then switching to **Claude** to resolve difficult bugs.
   - The member believes *aider's* strength lies in its method for working out which files need to be in the context, managing the context, and its search and replace coder which minimises llm token output.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1464261027297493045)** (2 messages): 

> `Elysia Agentic RAG, Skill Optimizer` 


- **Elysia Agentic RAG Deep Dive**: A member shared a link to a [blog post about Elysia Agentic RAG](https://www.unravel.tech/blog/elysia-agentic-rag-deep-dive) from Unravel Tech.
   - They asked for thoughts and opinions on the article.
- **Skill Optimizer Github Repo**: A user shared a link to the [Skill Optimizer GitHub repository](https://github.com/Ash-Blanc/skill-optimizer).
   - No additional context was provided about the repository's purpose or functionality.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1464047383083094135)** (2 messages): 

> `Continual Learning, DSPy.RLM()` 


- **Continual Learning Engineered into Existence**: A member posted about their exploration of **continual learning** over the last few months, sharing a [blog post](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859) on engineering it into existence.
   - They see **DSPy.RLM()** playing a huge role in this area.
- **Continual Learning and DSPy.RLM**: A blog post discusses engineering continual learning, envisioning a significant role for **DSPy.RLM()**
   - The post is available [here](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1464318884453552301)** (7 messages): 

> `DSPy Misunderstood, RLM Prompt Tuning, JSON Adapter Optimization` 


- **DSPy's True Power Finally Articulated**: A member published an [article](https://eito.substack.com/p/dspy-the-most-misunderstood-agent) explaining *"Why DSPy?"* and elaborating on the significance of **signature & module abstraction**, concepts often overlooked beyond the confines of the DSPy community.
   - The author believes DSPy's capabilities extend far beyond mere **GEPA & Prompt optimizations**, as highlighted in their tweet about the article [on X](https://x.com/Eito_Miyamura/status/2014757193766093069?s=20).
- **Rationalize Like Mad (RLM) Prompt Tuning Tactics**: Members discussed tuning the **RLM prompt** to improve reasoning, with one noting that some models provide *vague generic answers* even after peeping at the input.
   - One member suggested it's *optimizeable in a very similar manner to how ReAct is optimizeable*, suggesting the optimizer will automatically inspect the trace and that users should only focus on the desired output.
- **JSON Schema Sparks GEPA Customization Craze**: A member wants to optimize the **JSON adapter**'s system prompt using GEPA, specifically for models that utilize **json_schema response types**.
   - They believe that if many of the tokens added by the JSONadapter are unnecessary, **GEPA** could refine the system prompt, acknowledging the need for a **custom GEPA adapter** as the current DSPy one doesn't seem to affect the adapters.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1464201099123888242)** (10 messages🔥): 

> `Kimi issues, Conversation length exceeded messages, Slides are suddenly constantly showing 'This mode is at capacity.'` 


- **Kimi faces errors and issues**: A member reported issues with **Kimi**, including messages disappearing and frequent "conversation length exceeded" errors, and others reported slides constantly showing "This mode is at capacity."
   - While another member stated that visual and adaptive slides generation was working for them, others confirmed the ongoing issue, describing having to *click 50+ times until it finally works*.
- **Member speculates on Data center crash**: A member speculates a potential cause for the **Kimi** issues, suspecting *a datacenter crash*, **Nano Banana API access restrictions from Google**, or *a change of usage agreement*.
- **Radiohead inspires 'Ok Computer' title**: A member shared a [tweet](https://x.com/crystalsssup/status/2014571082716713356) confirming that **Radiohead** was the inspiration behind the name of their album **Ok Computer**.
   - The member also commented that *visual slides have become practically unusable*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1464020833528840374)** (3 messages): 

> `Introduction of AI/ML Engineers, Mojo in Production` 


- **AI/ML Engineers Introduce Themselves**: Multiple experienced **AI and ML engineers** introduced themselves, specializing in building and deploying **ML pipelines, deep learning models, and NLP systems**.
   - These engineers design **prediction engines, recommendation systems, and generative AI workflows**, integrating AI models into web and mobile applications while focusing on **reliability, performance, and production-ready ML architectures**.
- **Mojo's Prod Use Cases**: A member asked about the current adoption of **Mojo** in production environments.
   - They expressed interest in understanding the specific kinds of work where **Mojo** is being utilized.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1463995414884909247)** (5 messages): 

> `Mojo REPL issues, Python Package Installation, Github bug report` 


- **Mojo REPL encounters issues installing Python Packages**: A member reported an issue with the Mojo REPL when trying to install a Python package (**scons**) using `subprocess.check_call`.
   - The error message screenshot was included in the report ([see attached](https://cdn.discordapp.com/attachments/1151418092052815884/1463995414637187162/Screenshot_from_2026-01-22_13-33-59.png?ex=69752cfa&is=6973db7a&hm=99511e5767f0f4bd190a8ea5bc25af99ccfc0e565bb0cbc85ae99cefd7e0b743&)).
- **Inferred Parameters linger in old Github Issue**: A member urged the team to examine an older GitHub issue ([#4199](https://github.com/modular/modular/issues/4199)) concerning inferred parameters.
   - They indicated that the issue seems to persist and suggested that it might be bypassed by using a named parameter at the call site.
- **Github Bug report created**: A member has created a bug report ([#5830](https://github.com/modular/modular/issues/5830)) regarding the initial issue with the Mojo REPL.
   - The report was made in response to the suggestion to file a bug on GitHub.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1464327369325023284)** (2 messages): 

> `Caitie Mcaffrey, Point of Contact` 


- **Caitie Mcaffrey is mentioned as Point of Contact**: A member asked *Hey Alexander, Caitie Mcaffrey mentioned you as a point of contact. is it ok if i dm you on something?*
   - Another member replied that it was okay.
- **Member requests to DM Point of Contact**: One member asked another member if they could DM them.
   - The second member replied saying *done. lmk*.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1464216774957596735)** (1 messages): 

> `GetTaskPayloadResult, additionalProperties vs anyOf, Model Context Protocol` 


- **MCP's `GetTaskPayloadResult` Schema: `additionalProperties` Issue**: The Model Context Protocol's (`MCP`) `GetTaskPayloadResult` schema might be overly permissive due to using `additionalProperties` instead of `anyOf`, as highlighted in [this issue](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256).
- **`additionalProperties` vs `anyOf`: Schema Semantics Clash**: The discussion centers on whether `additionalProperties` in the `GetTaskPayloadResult` schema provides the correct validation behavior, compared to the intended stricter validation that `anyOf` might offer.
   - Using `anyOf` might enforce a more specific and controlled structure for the payload result, ensuring that only predefined properties are allowed.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1464322601236299860)** (2 messages): 

> `AI Engineer Europe Conference` 


- **AI Engineer Europe Conference Details Shared**: A member inquired about the time and location of an event, expressing that it was the first time they had heard about it.
   - Another member provided a link to the [AI Engineer Europe conference](https://www.ai.engineer/europe).
- **AI Event Location and Timing Inquiry**: A member asked about the date and location of an event, mentioning their availability in GR from March to May.
   - Another member responded with a link to the [AI Engineer Europe conference](https://www.ai.engineer/europe), seemingly in response to the inquiry.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1464205138137841739)** (1 messages): 

> `Mobile GPU Texture vs Buffer Paths, L2 Bandwidth Optimization` 


- **Mobile GPUs use separate paths for textures vs buffers**: Mobile GPUs often feature distinct pathways for handling **textures** and **buffers**, a design choice impacting memory access patterns.
   - Maximizing L2 bandwidth utilization may require strategically employing both pathways, such as using one input as a texture and another as a buffer.
- **L2 Bandwidth optimized by using texture and buffers**: Maximizing **L2 bandwidth** on mobile GPUs may involve utilizing both **textures** and **buffers** due to their separate hardware paths.
   - For instance, feeding the first input as a texture and the second as a buffer could optimize memory access and overall performance.


  

---


---


---
id: MjAyNS0w
title: OpenAI Realtime API GA and new `gpt-realtime` model, 20% cheaper than 4o
date: '2025-08-28T08:44:39.731046Z'
description: >-
  **OpenAI** launched the **gpt-realtime** model and **Realtime API** to GA,
  featuring advanced speech-to-speech capabilities, new voices (**Cedar**,
  **Marin**), image input, SIP telephony, and a ~20% price cut. Benchmarks show
  improvements over **gpt-4o-realtime** on BigBench and ComplexFuncBench.
  **xAI** introduced **Grok Code Fast 1**, a speed-optimized coding model
  integrated with popular IDEs, while **OpenAI Codex** received major upgrades
  for local and cloud development workflows. Google’s **Gemini CLI** improved
  multi-editor support, and new models like **Microsoft MAI-1-preview** and
  **MAI-Voice-1** were announced. *"The new all-in-one WebRTC API removes the
  ephemeral token step and supports video on the same connection,"* highlighting
  enhanced developer tooling.
companies:
  - openai
  - xai
  - microsoft
  - google
models:
  - gpt-realtime
  - gpt-4o-realtime
  - grok-code-fast-1
  - codex
  - mai-1-preview
  - mai-voice-1
  - gemini-cli
topics:
  - speech-to-speech
  - instruction-following
  - function-calling
  - telephony
  - webrtc
  - voice-agents
  - multilingual-switching
  - voice-control
  - benchmarks
  - coding-models
  - ide-integration
  - developer-tools
  - model-updates
people:
  - swyx
  - juberti
  - omarsar0
  - reach_vb
  - pbbakkum
  - skcd42
  - mohitreddy13
  - cline
  - kevinweil
  - gdb
  - sama
  - _philschmid
---


**Realtime is all you need?**

> AI News for 8/27/2025-8/28/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (185 channels, and 7363 messages) for you. Estimated reading time saved (at 200wpm): 577 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

The Realtime API has been in preview, and now is in GA, with [image inputs](https://openai.com/index/introducing-gpt-realtime/#image-input), [remote MCP server support](https://openai.com/index/introducing-gpt-realtime/#remote-mcp-server-support), [SIP/PBX support and prompt caching](https://openai.com/index/introducing-gpt-realtime/#additional-capabilities), and better [function calling](https://openai.com/index/introducing-gpt-realtime/#function-calling). Alongside it, there's a new realtime model! unfortunately not gpt5-realtime... it's still a marginally smarter model, just that most of the improvements are "API centric", aka function calling/instruction following.

There are 2 new voices and the voice control is unquantifiable but [worth trying it out](https://x.com/swyx/status/1961124194789499233):

![](https://resend-attachments.s3.amazonaws.com/Gbiyb8nvjPj1KGI)

---

# AI Twitter Recap

**OpenAI’s gpt-realtime and Realtime API GA (voice agents, telephony, tools)**

- **gpt‑realtime model + Realtime API GA**: OpenAI shipped its most advanced speech-to-speech model and took the Realtime API to GA with substantial capability and cost updates. Highlights: improved instruction following, tool calling, prosody and non-verbal cues, multilingual switching; new voices (**Cedar**, **Marin**); image input; remote MCP tool support; SIP telephony; new WebRTC APIs (server websocket control, video) and a ~20% price cut. Pricing shared by the community: ~$32/1M audio input tokens (cacheable at $0.40/1M) and $64/1M audio output tokens. Benchmarks vs GPT‑4o‑realtime suggest sizable gains on BigBench, ComplexFuncBench, and audio instruction-following. Demos include a Notion MCP example and WebRTC/SIP starter code. Threads: [@OpenAI](https://twitter.com/OpenAI/status/1961110295486808394), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961124915719053589), [API details by @juberti](https://twitter.com/juberti/status/1961116594211364942), [pricing by @omarsar0](https://twitter.com/omarsar0/status/1961117107417928047), [bench take by @reach_vb](https://twitter.com/reach_vb/status/1961140618295394579), [MCP demo by @pbbakkum](https://twitter.com/pbbakkum/status/1961120041799487654).
- **Developer notes**: The new all-in-one WebRTC API removes the ephemeral token step and supports video on the same connection; SIP endpoints enable call routing, transfer and hangup APIs for production call flows. Cookbook guidance covers voice prompt design (speed, tone, handoffs). See [WebRTC API update](https://twitter.com/juberti/status/1961118374345241016) and [SIP details](https://twitter.com/juberti/status/1961118371090501972).

**Coding Models and Dev Tooling: xAI’s Grok Code Fast 1, OpenAI Codex, editors/CLIs**

- **xAI’s Grok Code Fast 1**: A “speed-first,” economical reasoning model for agentic coding, free for a week and integrated across popular IDEs/tools (GitHub Copilot, Cursor, Cline, Kilo Code, Roo Code, opencode, Windsurf). The team emphasizes rapid rollout iterations and human+auto evals for real-world usefulness beyond benchmarks. Community tests are positive, and Cline added “three ways to code free” (cloud via Grok, local via LM Studio, or Qwen Code with generous daily limits). Announcements and context: [@xai](https://twitter.com/xai/status/1961129789944627207), [@skcd42](https://twitter.com/skcd42/status/1961132126298157060), [@MohitReddy13](https://twitter.com/MohitReddy13/status/1961138324426690608), [@cline launch thread](https://twitter.com/cline/status/1961201105729401060).
- **OpenAI Codex push (new stack integration)**: OpenAI’s Codex got a major upgrade: IDE extensions (Cursor/VSCode/Windsurf), a much-improved local CLI, unified local+cloud task management, and GitHub code reviews. Commentary notes deeper integration across the dev stack, including local/remote workflows. Engagement indicates strong reception. Threads: [@kevinweil](https://twitter.com/kevinweil/status/1960854500278985189), [@gdb](https://twitter.com/gdb/status/1960900413785563593), [@sama](https://twitter.com/sama/status/1961096744533647501).
- **Ecosystem improvements**: Google’s Gemini CLI landed native integration in Zed (multi-folder IDE mode, diff stats, better stability; community-driven PRs), easing multi-editor workflows ([@_philschmid](https://twitter.com/_philschmid/status/1961090847174262937)). OpenAI’s Realtime GA also unlocks voice-first coding assistants (MCP over voice).

**New Models and Benchmarks: Microsoft MAI, Cohere Translate, Tencent TV2A, GLM‑4.5**

- **Microsoft MAI‑1‑preview (text) and MAI‑Voice‑1**: Microsoft introduced its first in‑house models. MAI‑1‑preview entered the LMArena text leaderboard at #13 on debut; MAI‑Voice‑1 targets high‑quality speech generation (public testing encouraged). Microsoft signals rapid iteration and distribution via its product surface. Details: [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1961111770422186452), [@lmarena_ai](https://twitter.com/lmarena_ai/status/1961112908026593557), [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1961112928230461615).
- **Cohere Command A Translate**: A task‑specialized translation model with strong third‑party validation from RWS/Language Weaver. Community reaction is that domain‑trained translation outperforms frontier generalists (even GPT‑5) on complex multi‑domain tasks. More in [Cohere’s blog](https://twitter.com/cohere/status/1961081787674763525) and community takes by [@nickfrosst](https://twitter.com/nickfrosst/status/1961093091554713686).
- **Tencent HunyuanVideo‑Foley (TV2A)**: End‑to‑end text/video‑to‑audio framework trained on ~100k hours with an MMDiT backbone, REPA loss, and Audio VAE—reporting SOTA across audio quality, visual‑semantic, and temporal alignment. Code, report, and HF weights are public ([announcement](https://twitter.com/TencentHunyuan/status/1960920482779423211)).
- **Zhipu AI GLM‑4.5**: Now leading Berkeley’s Function‑Calling Leaderboard V4, reinforcing GLM‑4.5’s tool-use capability in practical API‑calling tasks ([results](https://twitter.com/Zai_org/status/1961149535754858586)).

**Agent Systems, Evals, and Patterns**

- **Parallel agents as a scaling axis**: Andrew Ng highlights parallel agent orchestration as the fourth scaling lever (after data, train compute, test‑time compute). Expect more multi‑agent research/cookbooks (research agents, background workers + UI monitors, mixture‑of‑agents aggregators) as token prices fall and latency budgets tighten ([thread](https://twitter.com/AndrewYNg/status/1961118026398617648)).
- **Memory‑R1 (RL for memoryful agents)**: GRPO variants significantly boost F1/BLEU/LaaJ on memory benchmarks (across Llama‑3.1‑8B and Qwen‑2.5‑7B) with outcome‑driven rewards and tiny data (152 QA pairs). Gains compound with stronger memory managers; generalizes across backbones. Notes and links: [@omarsar0](https://twitter.com/omarsar0/status/1961073807537693072).
- **Agentic RAG and evalability**: Elysia (open‑source agentic RAG) uses a decision‑tree architecture, dynamic data displays, on‑demand chunking, and feedback‑as‑few‑shot to improve determinism and debuggability ([overview](https://twitter.com/victorialslocum/status/1961095661719359624)). LlamaIndex shipped a multi‑agent “coding agent” that auto‑generates document workflows (edit/test/configure, code‑first, orchestrated via LlamaIndex workflows) ([demo](https://twitter.com/jerryjliu0/status/1961123785597505603)). AI SDK v5 added LangSmith tracing for observability: token usage, tool traces, TTFT ([@Hacubu](https://twitter.com/Hacubu/status/1961103113122984202)). For rigorous search‑augmented evaluation, Reka released Research‑Eval (374 diverse, high‑quality questions; frontier models spread 26.7%–59.1% acc), aiming beyond saturated SimpleQA/BrowseComp ([@RekaAILabs](https://twitter.com/RekaAILabs/status/1961192688029765936)).
- **DSPy practice**: Good discussion on data‑centric pipelines and where to put LLMs in the loop; optimizing via specs/evals before automation (fireside with @lateinteraction) ([session](https://twitter.com/sh_reya/status/1961110090314125524)).

**Image/Video Gen: Nano Banana momentum, ByteDance USO, Runway in production**

- **Nano Banana (Gemini 2.5 Flash Image) as a builder workhorse**: Heavy community use for personalizable styles, panel prompting, and mobile workflows; hackathon announced; Google showcased internal team work behind “banana.” Examples from Demis (isometric map→game idea), creative pipelines (glif agents; Suno for audio), and free hacks/promos accelerating adoption. Samples: [@demishassabis](https://twitter.com/demishassabis/status/1961077016830083103), [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1961127857192673540), [@tulseedoshi](https://twitter.com/tulseedoshi/status/1961068980640108889).
- **ByteDance USO (Apache‑2.0) style transfer/editing**: Open‑source text+image‑driven editing that “just works,” with HF demos and strong qualitative feedback from practitioners; a credible open alternative in the “nano banana” era ([overview](https://twitter.com/multimodalart/status/1961147988258295893)).
- **Runway Gen‑4 in production pipelines**: Filmmaking partnership with Fabula illustrates how in‑context tools augment pro workflows instead of replacing craft—case studies show where prompting meets production reality ([@runwayml](https://twitter.com/runwayml/status/1961088220571066620)). Also: test‑driving Wan 2.2 S2V indicates audio preprocessing/finetuning still matter for musical alignment ([@ostrisai](https://twitter.com/ostrisai/status/1960907113821298877)). Separately, Moonshot’s Kimi Slides introduced agentic deck‑building (ideas→decks, future auto image search/layout/polish) ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1961011693745811542)).

**Infrastructure and Strategy**

- **Compute build-out**: Reporting suggests OpenAI and Oracle are planning a 4.5 GW data center build (Stargate), following a 1.2 GW Abilene, with SoftBank/Microsoft/NVIDIA as partners; rumored $30B/yr contract. Site selection ongoing ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1960900145421177053)).
- **Platform share as national strategy**: A policy thread argues U.S. dominance requires maximizing usage (tokens, models, developers) on American hardware/software—favoring developer flywheels over export controls that inadvertently seed alternative stacks (Huawei+CloudMatrix+DeepSeek/Qwen) ([@sriramk](https://twitter.com/sriramk/status/1961072926561550366)). Related meta‑observation: labs pretrain on the same internet, but reinforcement and post‑training choices (and product data) drive “speciation” ([@tszzl](https://twitter.com/tszzl/status/1960953564681134472); [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1961121746670817404)).

**Top tweets (by engagement)**

- xAI released Grok Code Fast 1 (free for 7 days across major IDEs) [@xai](https://twitter.com/xai/status/1961129789944627207)
- OpenAI’s “Devs, tune in” livestream for Realtime API and gpt‑realtime [@OpenAI](https://twitter.com/OpenAI/status/1961081377174212979)
- OpenAI introduced gpt‑realtime and Realtime API GA [@OpenAI](https://twitter.com/OpenAI/status/1961110295486808394)
- Karpathy on “LLMifying” textbooks and environments for aligned training data [@karpathy](https://twitter.com/karpathy/status/1961128638725923119)
- “Nano Banana” community surge and hackathon announce [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1961127857192673540); Demis’ isometric map post [@demishassabis](https://twitter.com/demishassabis/status/1961077016830083103)
- OpenAI Codex features resonating with developers [@sama](https://twitter.com/sama/status/1961096744533647501)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. [Z.AI](http://z.ai/) GLM AMA + Mini MoE Roadmap

- [**AMA With](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/) [Z.AI](http://z.ai/)[, The Lab Behind GLM Models](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/)** ([Score: 396, Comments: 314](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/)): **AMA with [Z.AI](http://z.ai/) (creators of the GLM family) focuses on technical questions around GLM-4.5, especially post-training SFT for GLM-4.5 Air—requesting concrete hyperparameters (learning rate, batch size, epochs, dataset size, weight decay), target loss, and methods to avoid catastrophic forgetting, which commenters note aren’t detailed in the GLM 4.5 paper ([pdf](https://arxiv.org/pdf/2508.06471)). A community finetune of GLM-4.5 is shared for reference ([HF: GLM-Steam-106B-A12B-v1](https://huggingface.co/TheDrummer/GLM-Steam-106B-A12B-v1)). Other questions probe what differentiates open-weight models (GLM-4.5, Kimi K2) from frontier closed systems (GPT-5, Gemini, Claude) and what’s required to close the gap, plus whether [Z.AI](http://z.ai/) plans >32B dense models versus leaning into Big MoE architectures.** Commenters push for transparency and reproducibility (full SFT hyperparams and tuning targets) and debate whether open-weight efforts can realistically match or surpass closed frontier models. There’s also interest in the architectural trade-offs and roadmap between scaling dense models (e.g., ~70B+) and investing in larger MoE systems.
    - A commenter requests the exact SFT post-training recipe for **GLM‑4.5 Air**—learning rate schedule, global batch size, number of epochs, dataset size/composition, weight decay, and any adapter strategies—plus practical targets like cross-entropy loss/perplexity and methods to prevent *“catastrophic forgetting.”* They reference a community finetune [GLM-Steam-106B-A12B-v1](https://huggingface.co/TheDrummer/GLM-Steam-106B-A12B-v1) and note the official paper lacks these details ([arXiv:2508.06471](https://arxiv.org/pdf/2508.06471)). They’re seeking guidance on tuning **GLM‑4.5 Air** (e.g., small LR, mixed replay from pretrain corpus, KL/L2 regularization, or gradual unfreezing) to avoid degradation during SFT.
    - Another thread asks what open‑weight models like **GLM‑4.5** and **Kimi K2** need to do to catch up with closed frontier models (GPT‑5, Gemini, Claude). The focus is on potential gaps in training compute, data quality/scale, RLHF/RLAIF and tool‑use pipelines, safety alignment, and eval‑driven training; they probe whether improved scaling strategies, better data curation, and distillation from frontier models could close the gap and whether parity is feasible.
    - Multiple questions probe [**Z.AI**](http://z.ai/)’s scaling roadmap: continue with dense models >`32B` versus following the trend toward large **Mixture‑of‑Experts (MoE)**. They ask whether SOTA closed models likely have more parameters than GLM and if increased parameter count is necessary for SOTA‑level performance, implicitly weighing training/inference cost, routing quality, and throughput benefits of sparsity against the stability/simplicity of dense `70B`‑class models.
- [**Launching Our New AMA Series With](https://i.redd.it/ek8o2pfzumlf1.jpeg) [Z.AI](http://z.ai/)[, Creators of GLM (Tomorrow, 9AM-12PM PST)](https://i.redd.it/ek8o2pfzumlf1.jpeg)** ([Score: 291, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1n1unkv/launching_our_new_ama_series_with_zai_creators_of/)): **r/LocalLLaMA is hosting an AMA with [Z.AI](http://z.ai/), the team behind the GLM (General Language Model) family, scheduled for Thu, Aug 28, 2025, 9AM–12PM PST. The post is an image flyer announcing the event; no technical details or agenda items (e.g., GLM variants, benchmarks, local deployment specifics) are included in the image or title beyond timing and hosts.** Commentary is mostly light/administrative (e.g., noting the AMA and subreddit naming humor) with no substantive technical discussion yet.
    - Scheduling clarity: A bot corrects the event time to PDT (not PST) due to DST and links a time conversion: [https://timee.io/20250828T1600?tl=Launching%20Our%20New%20AMA%20Series%20With%20Z.AI%2C%20Creators%20of%20GLM%20(Tomorrow,%209AM-12PM%20PST)&d=180](https://timee.io/20250828T1600?tl=Launching%20Our%20New%20AMA%20Series%20With%20Z.AI%2C%20Creators%20of%20GLM%20(Tomorrow,%209AM-12PM%20PST)&d=180). This maps the AMA to 9AM–12PM PDT (16:00–19:00 UTC) with a `180` minute duration, reducing ambiguity for global attendees.
    - Roadmap interest: A commenter asks “glm 6 when?”, signaling demand for details on the next GLM release timeline. While no specs are discussed in-thread, this points to expected AMA topics like version cadence and feature upgrades for future GLM iterations.
- [**glm mini will be comming**](https://i.redd.it/h1ss59p4lslf1.jpeg) ([Score: 191, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1n2hyt2/glm_mini_will_be_comming/)): **In an AMA screenshot with the Z.ai/GLM team, a user asks about plans for smaller Mixture-of-Experts (MoE) models (e.g., OSS-20B or 30B-A3B), and a co-host confirms they plan to train a smaller MoE model comparable to GPT-OSS-20B. This suggests a forthcoming “GLM mini” MoE variant targeting lower active parameter counts for easier local inference while retaining strong capability, akin to Qwen 30B A3B-style configs. [Image link](https://i.redd.it/h1ss59p4lslf1.jpeg).** Commenters note Qwen 30B A3B performs well but its low active parameter budget hurts long-context reasoning; a hypothetical 38B A6B is proposed as a sweet spot—more experts per token yet still locally runnable. Others ask for the AMA source/context, with OP stating it’s from a current [Z.ai](http://z.ai/) team AMA.
    - Discussion centers on Mixture-of-Experts designs: a user notes Qwen 30B A3B performs well but its low “active parameters” per token appears to hurt longer-form reasoning, proposing a 38B A6B variant to boost active capacity while staying locally runnable. In MoE notation (e.g., Qwen2 57B-A14B), the “A#B” denotes approximate active parameters per token, so moving from `~3B` to `~6B` active could materially improve capability without the full compute of a dense 30–40B model ([Qwen2 MoE naming for context](https://qwenlm.github.io/blog/qwen2/)).
    - The AMA hint that “GLM mini” is coming raised ambiguity around a claim of being “comparable to gpt-oss-20B”; commenters question whether this refers to parameter count or actual quality. Historically, “comparable” in these announcements often maps to model size rather than parity on benchmarks, where training data, compute budget, and instruction-tuning heavily affect outcomes (GLM family reference: [ZhipuAI/GLM](https://github.com/THUDM/GLM)).
    - On usability/local inference, the suggestion is that an A6B MoE could be widely runnable: MoE increases active compute only for a subset of experts per token, enabling higher effective capacity at similar step-time to much smaller dense models. Caveat: VRAM footprint can still be dominated by total parameters (all experts) unless the runtime supports expert sharding/offload; engines like vLLM have begun optimizing MoE loading and routing for practical deployment ([vLLM MoE support](https://blog.vllm.ai/2024-02-05-moe/)).
- [**Again where behemoth and reasoning model from meta ??**](https://i.redd.it/xma7ru49krlf1.png) ([Score: 224, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1n2chrm/again_where_behemoth_and_reasoning_model_from_meta/)): **The image is a promo slide for Meta’s “Llama 4” multimodal MoE lineup, highlighting “Llama 4 Behemoth” as a 16‑expert MoE with** `2T` **total and** `288B` **active parameters, positioned as an “intelligent teacher” for distillation; companion variants “Maverick” and “Scout” target speed/efficiency. The OP’s title (“where behemoth and reasoning model from meta??”) implies these large/“reasoning” models haven’t been publicly released; the slide emphasizes distillation and efficiency rather than availability. [Image](https://i.redd.it/xma7ru49krlf1.png).** Commenters are skeptical, suggesting Behemoth would underperform vs **Qwen 3 235B** despite being ~6× larger, calling it “dead on arrival,” with some tongue‑in‑cheek claims it’s guiding Meta’s strategy.
    - Speculation that Meta’s unreleased “behemoth” reasoning model underperforms smaller open models, with one comment asserting it’s *“probably worse than **Qwen 3 235B** at 6× the size.”* If accurate, that indicates poor scaling efficiency where adding parameters (`>6×`) fails to translate into better reasoning quality versus a `~235B` baseline.
    - Another technical inference is that non-release itself is a negative performance signal: if the model were competitive, Meta would have shipped it. The implication is that internal evaluations likely didn’t surpass current SOTA on reasoning, so the absence of a release suggests underwhelming benchmark results and limited practical value at this stage.

### 2. Audio Gen Releases: HunyuanVideo-Foley and VibeVoice TTS

- [**HunyuanVideo-Foley is out, an open source text-video-to-audio model**](https://v.redd.it/jpjpqw2xuolf1) ([Score: 294, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1n22xbl/hunyuanvideofoley_is_out_an_open_source/)): **Tencent’s HunyuanVideo-Foley is an open-source, video-conditioned (text–video→audio) model that generates foley/soundtracks aligned to an input video, with a public demo, weights, and code: [demo](https://hunyuan.tencent.com/video/zh?tabIndex=0), [Hugging Face](https://huggingface.co/tencent/HunyuanVideo-Foley), [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley), [project page](https://szczesnys.github.io/hunyuanvideo-foley/), and [arXiv](https://arxiv.org/abs/2508.16930). Early user feedback notes improved frequency response (stronger bass/treble) and better A/V synchronization versus prior attempts, targeting the missing audio stage in current video-generation pipelines (e.g., pairing with models like Hunyuan/Wan for visuals and TTS for dialog). The thread clarifies that it can indeed generate appropriate audio for existing video tracks (i.e., video-to-audio with optional text conditioning).** Commenters see this as the “last piece” enabling end-to-end automated content pipelines and discuss multi-GPU orchestration (e.g., persistent model loading in tools like ComfyUI) to batch long-running jobs; enthusiasm centers on workflow integration rather than raw benchmarks.
    - Multiple users clarify that a “text-video-to-audio” model here means generating Foley/ambient SFX aligned to an already existing video track, effectively filling the missing audio layer. This slots into an end-to-end pipeline alongside text/image-to-video models like **Hunyuan** and **Wan** plus dialogue models like **Infinite Talk**, enabling fully synthetic shorts with synchronized visuals and sound.
    - There’s interest in building a `multi-GPU` production pipeline where each model (design, T2V, dialogue, Foley) stays resident on dedicated GPUs and passes artifacts downstream, minimizing reload overhead and maximizing throughput. A key open question is whether **Comfy** currently provides robust multi-GPU graph execution/scheduling to support persistent residency, inter-model transfers, and weekend-long batch queues.
    - Early qualitative notes: audio quality reportedly has better frequency balance ("mid, bass, and treble") and tighter A/V sync versus earlier attempts. Practical deployment concerns include model size being “not too big,” a request for release in `safetensors` format for easier/safer loading, and questions about concrete run instructions.
- [**RELEASED: ComfyUI Wrapper for Microsoft’s new VibeVoice TTS (voice cloning in seconds)**](https://v.redd.it/yy7k60z8eplf1) ([Score: 228, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1n24utb/released_comfyui_wrapper_for_microsofts_new/)): **Open-source ComfyUI wrapper for Microsoft’s new VibeVoice TTS adds a Single Speaker node, a Multiple Speakers node (up to** `4` **speakers—model limit), and file-based text input for long-form synthesis, with repo at [Enemyx-net/VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI). Reported VRAM use on official weights:** `~5 GB` **for the** `1.5B` **model and** `~17 GB` **for the** `7B` **model (the latter still in Preview), with qualitatively strong single-speaker cloning from a** `~56s` **prompt; multi-speaker is “decent only with the 7B” but has a lower success rate. The model is highly seed-sensitive (large quality variance across seeds) and shows mixed cross-lingual behavior: non‑EN/zh prompt audio (e.g., Italian) can yield same‑language output, but EN prompts did not reliably produce other languages.** User feedback notes it works on an RTX 5090 and suggests ending prompts with punctuation or trailing ellipses (“ ...”) to avoid early cutoffs in short utterances; others request/anticipate quantized releases to reduce resource use and praise the node’s utility.
    - A user confirms the ComfyUI wrapper for Microsoft VibeVoice TTS runs smoothly on an **RTX 5090** with a self-cloned voice, suggesting good compatibility on high-end NVIDIA cards (no artifacts or instability reported). While no latency numbers are given, the report implies real-time or near–real-time responsiveness for personal voice use.
    - Practical workaround for premature audio cut‑offs on short prompts: end the input with punctuation ("?", "!", ".") and add a trailing " ..." (e.g., "Hello? ..."). This appears to mitigate end-of-sequence or silence-trimming behavior that can truncate single-word or very short TTS outputs.
    - There’s demand for a quantized build, which would lower VRAM requirements and potentially improve throughput on smaller GPUs/CPUs. Such a release would broaden deployability beyond high-end cards while trading off minimal quality loss typical of quantization.

### 3. Local AI Tools: gpt-oss 60K-context Training and Second Brain

- [**Gpt-oss Fine-tuning - now with 60K context length and fits on <13GB VRAM**](https://i.redd.it/rwu8gezzwslf1.jpeg) ([Score: 229, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1n2jraj/gptoss_finetuning_now_with_60k_context_length_and/)): **Post announces Unsloth’s Flex Attention for OpenAI gpt-oss training, claiming** `>8×` **longer context,** `>50%` **less VRAM, and** `>1.5×` **faster training than other impls including FlashAttention-3, enabling** `~60K` **token context on** `80GB` **VRAM for BF16 LoRA (title also touts “<13GB VRAM” fit). It adds export of QLoRA-finetuned gpt-oss models to** `llama.cpp`**,** `vLLM`**,** `Ollama`**, and HF, fixes float16 loss blow-ups on T4/Colab, and enforces** `swiglu_limit=7.0` **for MXFP4 inference in transformers; savings increase with longer sequences. Links: [Unsloth](https://github.com/unslothai/unsloth), blog/details: [docs.unsloth.ai/basics/long-context-gpt-oss-training](https://docs.unsloth.ai/basics/long-context-gpt-oss-training).** Comments ask about scaling to a 120B model and show strong interest in the upcoming notebook with direct GGUF/llama.cpp saving; general sentiment is enthusiastic.
    - The dev notes an upcoming training notebook with direct save-to-**GGUF** for **llama.cpp** ("next week"), which would remove conversion steps and enable immediate inference across llama.cpp backends (CPU, CUDA, ROCm, and Apple Metal). This would also simplify integration with tooling like **LM Studio**, and make quantized deploys (e.g., Q4/Q5) straightforward for the advertised `60k` context and `<13 GB` VRAM target. Links: [llama.cpp](https://github.com/ggerganov/llama.cpp), [GGUF spec](https://github.com/ggerganov/llama.cpp/tree/master/gguf).
    - There’s user demand for a larger `120B` variant. Practically, local inference for `120B` typically exceeds single-device constraints, often requiring multi-GPU tensor parallelism and aggressive quantization; even 4-bit can require `~40–60 GB` VRAM, making it well beyond the `<13 GB` class unless using distributed setups.
    - Multiple users ask about macOS support: running in **LM Studio** on a Mac mini M4 and whether **Unsloth** is coming to Mac. If the model exports directly to **GGUF**, it becomes immediately usable via **llama.cpp** with the Metal backend (which LM Studio wraps), improving Mac compatibility without bespoke ports. Links: [LM Studio](https://lmstudio.ai/), [Unsloth](https://github.com/unslothai/unsloth).
- [**I built a local “second brain” AI that actually remembers everything (321 tests passed)**](https://www.reddit.com/r/LocalLLaMA/comments/1n2djpx/i_built_a_local_second_brain_ai_that_actually/) ([Score: 259, Comments: 120](https://www.reddit.com/r/LocalLLaMA/comments/1n2djpx/i_built_a_local_second_brain_ai_that_actually/)): **OP introduces Kai, a local "cognitive OS" that builds persistent, on-device memory using a graph-based knowledge store with spreading activation for retrieval (akin to cognitive architectures like ACT-R). It runs 100% locally (no cloud), learns from user activity across the machine, and emphasizes it’s "not just RAG"—instead leveraging a node/edge memory graph + activation dynamics; the project reports** `321` **passing tests and offers early access at [oneeko.ai](http://oneeko.ai/) with a 3D memory visualization [screenshot](https://preview.redd.it/8jei7138zrlf1.png?width=1920&format=png&auto=webp&s=b4125be85bd9a5a616c10a0423130cba14169100). OP plans to open the core engine once stable.** Top comments push for open-sourcing given the local-only claim, share a similar project using a query-driven activation and residual-strengthening approach ([dsam_model_memory](https://github.com/jwest33/dsam_model_memory)), and a skeptic suggests it might be just an MCP-style server tagging/summarizing conversational data—with the usual failure modes of such systems.
    - A commenter building a similar system shares they use a query-based activation function to generate residuals that strengthen frequently accessed memories and related concepts (repo: https://github.com/jwest33/dsam_model_memory). They present this as biasing retrieval toward high-salience items over time, rather than static vector-store recall, to improve long-term relevance in a personal knowledge base.
    - Another commenter suspects the project is essentially an **MCP** server that tags conversational data and builds a summary graph, with both "save" and "query" interfaces (see https://modelcontextprotocol.io/). They caution that this architecture typically inherits the same failure modes seen in similar tag/summarization-graph pipelines, implying persistent issues when metadata diverges from user intent over time.
    - Hardware performance observations: on their setup, **qwen3 235b a22b** runs at `~20 tps`, **glm-4.5-air** at `~40 tps`, and **gpt-oss-120b** at `~70 tps`, while they'd prefer `>=100 tps`. They also note many models feel "too censored" for personal-assistant workflows, preferring fewer safety interventions to enable open-ended exploration.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. GPT-5 Medical Benchmarks and Codex IDE/CLI Launch

- [**GPT-5 outperformed doctors on the US medical licensing exam**](https://i.redd.it/rjs8oqzb1qlf1.png) ([Score: 666, Comments: 245](https://www.reddit.com/r/OpenAI/comments/1n26rqz/gpt5_outperformed_doctors_on_the_us_medical/)): **A preprint, “Capabilities of GPT-5 on Multimodal Medical Reasoning” ([AlphaXiv](https://www.alphaxiv.org/pdf/2508.08224)), claims GPT‑5 outperforms licensed physicians by ~**`25–30%` **on USMLE-style evaluations, shown in the tweet’s tables. The result appears to rely on structured, expert-curated inputs (i.e., near‑perfect diagnostic/context data) and is not an end‑to‑end clinical workflow; it evaluates reasoning/answer selection on exam‑like vignettes rather than autonomous patient management.** Top comments note the caveat that performance hinges on being given perfect diagnostic data, likening the setup to an open‑book exam, and caution that real clinical safety (drug interactions, longitudinal context) remains an unresolved challenge despite strong exam performance.
    - Several commenters note the benchmark likely assumes idealized inputs — e.g., results contingent on being “provided with perfect diagnosis data from a human expert.” This setup evaluates answer selection under clean, expert-curated context, not end-to-end clinical reasoning with noisy, incomplete histories, which is a major confound when comparing to practicing physicians who must perform triage, elicit histories, and resolve ambiguity.
    - A technically relevant safety concern is state/recall limits: an LLM may “forget” earlier chart details due to context truncation, risking contraindicated suggestions (e.g., proposing another NSAID after prior ibuprofen, such as diclofenac). This highlights the need for robust patient-state tracking, medication reconciliation, and automated drug–drug interaction checks as guardrails, rather than relying on transient chat context alone.
    - Multiple remarks frame this as an “open‑book advantage”: the model effectively carries a corpus of textbooks via pretraining, so outperforming on multiple‑choice exams mainly reflects test-taking/recall under vast prior knowledge. This metric is not equivalent to bedside performance; it differs from other validated AI strengths (e.g., specific imaging tasks) and raises fairness questions versus humans taking a closed‑book, time‑limited USMLE.
- [**Codex now runs in your IDE, Cloud and CLI with GPT-5**](https://i.redd.it/z91dfrq44nlf1.png) ([Score: 221, Comments: 80](https://www.reddit.com/r/ChatGPTCoding/comments/1n1vjvi/codex_now_runs_in_your_ide_cloud_and_cli_with_gpt5/)): **An OpenAI Developers announcement (Aug 27, 2025) claims that Codex now works as a coding collaborator across IDEs, the cloud, and the CLI, “powered by GPT-5” and accessible via the ChatGPT plan. The graphic highlights a new IDE extension, seamless task handoff between local and cloud environments, GitHub code review integration, and a revamped Codex CLI—suggesting tighter end‑to‑end workflow coverage from editing to review to execution.** Commenters ask for real‑world comparisons to **Claude Code** (quality/usability), whether the prior sandboxing requirement still applies (a blocker for some), and if there’s support for **RStudio**/R workflows.
    - Users flag the prior Codex requirement to run code in a strict sandbox as a major blocker for real-world workflows (file system, network, package managers, test runners), asking if the new IDE/Cloud/CLI release relaxes or allows opting out per project. The resolution of sandboxing (e.g., trusted directories, network egress, env var access) will determine whether it’s viable for in-IDE refactors and debugging versus only safe, ephemeral runs.
    - A power user on the `Claude Code $100` plan reports preferring GPT‑5’s raw code-generation quality but still finding Claude Code’s overall “system” harder to beat. The takeaway is that model quality alone isn’t sufficient; reliability and end‑to‑end developer ergonomics (workflow orchestration, context handling, integrations) at the `~$100/mo` tier are decisive for adoption.
    - There’s uncertainty about access tiers: whether `GPT‑5 High` is available under the `$20` ChatGPT Plus plan in Codex. One commenter found “medium thinking” underwhelming, implying meaningful quality gaps between “Medium” and “High” tiers that could affect latency/cost tradeoffs and plan selection.
- [**Who’s Your Doctor Now?**](https://i.redd.it/bx5xh1p4mrlf1.jpeg) ([Score: 2733, Comments: 87](https://www.reddit.com/r/ChatGPT/comments/1n2cqng/whos_your_doctor_now/)): **Non-technical meme contrasting perceived bedside manner of AI assistants vs web search: under the OpenAI logo it says “Nothing serious, it can be treated,” vs Google’s “You have 3 minutes left,” implying LLM reassurance vs search-engine-induced alarmism. Title “Who’s Your Doctor Now?” frames it as a tongue-in-cheek take on self-diagnosis culture; no benchmarks, models, or implementation details discussed.** Comments reminisce about the “Dr. Google” era exacerbating hypochondria and joke about overdiagnosis, with some sarcastic quips about professionalism and calling everything cancer.
- [**Rate this art by gpt 5**](https://i.redd.it/nf5kr1bjiolf1.jpeg) ([Score: 244, Comments: 189](https://www.reddit.com/r/ChatGPT/comments/1n21mfi/rate_this_art_by_gpt_5/)): **AI-generated abstract of Lord Ganesha; despite the title (“by gpt 5”), the shared prompt clearly indicates Midjourney v6.1: “thick paint splashes … white background --personalize cvlos9g --stylize 800 --v 6.1.” The high** `-stylize 800` **drives the bold, minimalist paint-stroke aesthetic, and** `-personalize cvlos9g` **suggests a user/style-specific personalization token, yielding a clean white background with vivid, liquid-paint strokes. Image: https://i.redd.it/nf5kr1bjiolf1.jpeg** Comments note resemblance to the Olympic logo and include polarized views on AI art’s value; a commenter provides the exact prompt so others can replicate the result, implicitly correcting the GPT-5 attribution to Midjourney.
    - A commenter shared the exact prompt and parameters: “thick paint splashes forming abstract minimalist shape of Lord Ganesha, … white background --personalize cvlos9g --stylize 800 --v 6.1”. This implies **Midjourney v6.1** (`-v 6.1`), with a high `-stylize 800` value that strongly biases outputs toward aesthetics over literal prompt adherence (see Midjourney parameter docs: https://docs.midjourney.com/docs/parameters). The `-personalize cvlos9g` token appears to be a custom style/profile identifier influencing palette and composition.
    - Observations like “It doesn’t look AI generated” align with how MJ v6.x’s improved coherence and texture handling can produce clean, logo-like geometry and consistent “liquid paint” effects. Minimalist composition plus a white background and high stylization tend to suppress common AI tells (messy edges, inconsistent brush physics), yielding results that some viewers read as non-AI; cf. model/version notes: https://docs.midjourney.com/docs/models#version-6.
- [**Chicken of the Sea - SaraShakeel x Ai render**](https://v.redd.it/ujigpiulfplf1) ([Score: 349, Comments: 10](https://www.reddit.com/r/aivideo/comments/1n2524a/chicken_of_the_sea_sarashakeel_x_ai_render/)): **A user shares an AI-generated visual titled “Chicken of the Sea — SaraShakeel x AI render,” apparently styled after artist Sara Shakeel, hosted on Reddit Video at [v.redd.it/ujigpiulfplf1](https://v.redd.it/ujigpiulfplf1). The external link currently returns** `HTTP 403 Forbidden`**, implying access requires Reddit login or a developer token (likely WAF/auth gating), and the thread provides no technical metadata (e.g., model, prompts, or pipeline details). No benchmarks, implementation notes, or asset workflow are discussed; the thread is primarily aesthetic reception.**
    - One commenter detailed a pre-AI pipeline: starting from a retouched reference image composited in Photoshop, then using **Midjourney** for expanded looks, followed by animation in **Cinema 4D** with the **Arnold** renderer, particle simulations, and compositing/tracking in **After Effects**/**Mocha** ([Midjourney](https://www.midjourney.com/), [C4D](https://www.maxon.net/cinema-4d), [Arnold](https://www.arnoldrenderer.com/), [After Effects](https://www.adobe.com/products/aftereffects.html), [Mocha](https://borisfx.com/products/mocha/)). They report `~4 weeks` of work for a `2-minute` deliverable including `~1 week` of rendering, for a `~$5k` payout (not continuous time), noting the realism lagged compared to current AI renders and that they should have priced closer to `~$10k`. They conclude that *"AI is devaluing the market,"* reflecting perceived downward pressure on rates as generative tools improve speed/realism.

### 2. WAN 2.x Infinite Talk Demos & S2V Tips + HunyuanVideo-Foley

- [**4090 48G InfiniteTalk I2V 720P Test~2min**](https://v.redd.it/uxe60qpinnlf1) ([Score: 501, Comments: 117](https://www.reddit.com/r/StableDiffusion/comments/1n1ycs9/4090_48g_infinitetalk_i2v_720p_test2min/)): **Creator benchmarked an I2V pipeline on an RTX 4090 (48 GB) using** `wan2.1_i2v_720p_14B_fp8_scaled` **with LoRA** `lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16`**, generating 1280×720 output at 4 diffusion steps while consuming ~**`36 GB` **VRAM. The run processed** `49` **chunks of** `81` **frames each (title: ~2 min total), at ~**`5 min` **per chunk for** `~245 min` **total; the FP8-scaled 14B model plus a step‑distilled LoRA (rank** `256`**, bf16) suggests a speed/memory‑optimized setup. Source audio is an AI cover on YouTube (https://youtu.be/9ptZiAoSoBM) sung in the style of [Hiromi Iwasaki](https://en.wikipedia.org/wiki/Hiromi_Iwasaki).** Commenters report lip/voice sync is strong overall but degrades during background vocals, speculate mic motion/handling may confuse the model, and predict near‑term agent workflows that auto‑edit and publish music videos from a song upload.
    - Observers note the voice–lip sync is largely accurate at `720p` but degrades during overlapping/background vocals; using clean vocal stems would likely improve alignment. There’s speculation that erratic microphone movements may interfere with source detection/voice activity cues, causing the model to momentarily track the wrong singer.
    - There’s specific interest in the non-standard `RTX 4090 48GB` configuration used for the ~`2 min` I2V run, with requests for the exact vendor/mod source. Commenters flag that this atypical memory capacity impacts reproducibility and potential batch/window sizes for others attempting the setup.
    - Questions about multi-GPU capability (e.g., splitting inference/training across GPUs) suggest users want to know if the InfiniteTalk I2V pipeline supports data/model parallelism or VRAM sharding. Clarity on whether the `48GB` requirement can be met via multi-GPU aggregation versus a single large-VRAM card would inform hardware choices.
- [**Three reasons why your WAN S2V generations might suck and how to avoid it.**](https://v.redd.it/hxa93nfu9slf1) ([Score: 510, Comments: 151](https://www.reddit.com/r/StableDiffusion/comments/1n2gary/three_reasons_why_your_wan_s2v_generations_might/)): **OP reports that WAN S2V yields significantly better out‑of‑the‑box results via the WanVideoWrapper than the native ComfyUI workflow, which required extensive tweaking for only moderate quality. They advise avoiding “speed‑up LoRAs,” which they say degrade both WAN 2.2 and S2V output quality and movement/prompt adherence (only acceptable for mostly static talking heads). Strong prompt engineering is emphasized: specify music genre, atmosphere, emotional state, gaze direction, head/body motions, and exact actions rather than vague prompts. Example run:** `576x800` **resolution,** `~737f`**, sampler** `UniPC/beta`**,** `23` **steps. The linked media is access‑restricted ([v.redd.it](http://v.redd.it/) [403](https://v.redd.it/hxa93nfu9slf1)); see also [ComfyUI](https://github.com/comfyanonymous/ComfyUI).** Top comments include a request to share the workflow (user “Limewire”) and general praise of the result; no substantive technical counterpoints were offered.
    - **comfyanonymous** notes the "native workflow" for S2V is not officially announced and the node is still marked `beta`, implying current quality issues stem from immature implementation; once the native node is fully implemented, it should outperform interim/third‑party workflows. This suggests users should expect rapid iteration and possibly breaking changes until the native node stabilizes.
- [**Wan 2.1 Infinite Talk (I2V) - FOAR EVERYWUN BOXXY**](https://v.redd.it/qx1b1h6z8rlf1) ([Score: 217, Comments: 45](https://www.reddit.com/r/StableDiffusion/comments/1n2b4gi/wan_21_infinite_talk_i2v_foar_everywun_boxxy/)): **OP demonstrates an Image-to-Video workflow using Wan 2.1 “Infinite Talk” to produce a talking-head clip with intentional upper-body/hand motion. The positive prompt targets facial cosmetics (big eyelashes/eyeliner) and short, black-painted nails, while an exhaustive negative prompt suppresses common video-generation artifacts (e.g., long nails/jewelry, overexposure/blur/static frames, JPEG artifacts, extra/fused fingers, deformed limbs, messy backgrounds, multiple/extra limbs, walking backwards), aiming for cleaner hand renderings and more dynamic motion. No generation parameters (resolution/FPS/steps/sampler/seed/CFG/duration) or hardware details (e.g., VRAM) are provided.** Commenters praise the quality—one calls it “BY FAR the best example” of Infinite Talk—while another asks about VRAM requirements, indicating interest in compute footprint; no answer is given in-thread.
    - Resource requirements: A commenter asks, "How much vram for this results," seeking concrete GPU memory needs to reproduce the shown Infinite Talk I2V quality. Technical readers would expect details like VRAM usage at specific resolutions/durations (e.g., `512p/1024p`, seconds per frame), model precision (`fp16` vs `bf16`), and whether inference used xformers/attention slicing or CPU offload to fit into commodity GPUs.
    - Identity fidelity and reference control: One notes the output "doesn't look like Boxxy" and requests the original image, implicitly probing the pipeline’s identity preservation and conditioning strength. This raises questions about the reference handling (single image vs multi-shot, face-alignment/landmark guidance, ID loss, and use of face-enhancers like GFPGAN/CodeFormer) and whether the I2V model supports guidance scale or identity embeddings to keep likeness stable across frames.
    - Comparative performance: Another asks if this is "better than the new S2V model," indicating interest in head-to-head quality and stability comparisons between Wan 2.1 Infinite Talk (I2V) and S2V. Relevant benchmarks would include motion coherence, lip-sync accuracy, temporal consistency (flicker/warp), inference speed (FPS), and VRAM efficiency at matched prompts and resolutions.
- [**HunyuanVideo-Foley got released!**](https://v.redd.it/zazjjguqoplf1) ([Score: 289, Comments: 46](https://www.reddit.com/r/StableDiffusion/comments/1n25nqj/hunyuanvideofoley_got_released/)): **HunyuanVideo‑Foley is an open‑source Text+Video→Audio (foley) model that generates synchronized sound effects from video input (optionally text‑conditioned). A project page with interactive demos and side‑by‑side comparisons against baseline models like MMAudio and ThinkSound is available here: https://szczesnys.github.io/hunyuanvideo-foley/.** Early user feedback reports mixed quality: anime content can collapse to low‑energy breaths/mumbling, and some real‑life clips yield abrasive “sandpaper” textures; MMAudio baselines are noted to sometimes emit random, loud “screams,” highlighting artifact/hallucination issues. One commenter also hints at heavy I/O/compute demands ("My SSD is tired...") during generation.
    - Multiple users report severe realism issues and artifacting: outputs devolve into "mutated exorcist screaming," unintelligible mumbling, or broadband "sandpaper" noise during action onsets. This points to weak audio-visual alignment and poor transient handling—likely diffusion artifacts and unstable conditioning causing temporal drift and spectral roughness, resulting in janky Foley lacking precise onsets, dynamics, and spatial cues.
    - Clear domain gap between styles: anime sequences produce only a faint sigh then garbled vocalizations, while live-action yields abrasive textures. This suggests the model isn’t robust to stylized visual domains (anime) and defaults to generic, low-information acoustic priors, indicating insufficient domain-conditioned training or inadequate style tokens/embeddings for non-photorealistic inputs.
    - NSFW prompts appear specifically bad (generic/rubbed textures, suppressed or mismatched erotic SFX), hinting at safety filtering or data sparsity in such content. The behavior resembles hard clamping toward neutral textures and low-variance outputs under restricted semantics, which further degrades alignment and timbral specificity in those scenarios.
- [**If this is Genie 3, imagine how insane Genie 4 will be**](https://v.redd.it/6rk25azwirlf1) ([Score: 1209, Comments: 179](https://www.reddit.com/r/singularity/comments/1n2cbyj/if_this_is_genie_3_imagine_how_insane_genie_4/)): **Thread centers on the rapid capability jump from “Genie 2” to “Genie 3” within** `~8–9 months`**, as evidenced by a shared demo video (requires auth: https://v.redd.it/6rk25azwirlf1). No benchmarks or release notes are cited; discussion is primarily about trajectory—toward higher physical fidelity and interactive, navigable environments—rather than implementation specifics.** Commenters speculate that “Genie 4” could add fine‑grained, physically consistent scene effects (e.g., “paw prints in the sand”) and support real‑time VR exploration of generated spaces; some infer that, if the exponential cadence holds, the next iteration may arrive soon.
    - Release cadence speculation: One commenter notes Genie 2 → Genie 3 landed within `~8–9 months`, reading this as a sign of accelerating iteration and expecting Genie 4 on a short horizon if the trend is exponential. Another draws a parallel to the perennial “GPT-4 → GPT-5” hype cycle, implicitly cautioning that cadence ≠ capability without concrete benchmarks or demos to compare across versions.
    - Interaction model/UX question: A user asks how Genie actually works in practice—whether it needs constant prompting versus supporting a stateful, continuous session. Another speculates about real-time VR-style exploration, implying a system that maintains scene state and accepts continuous control inputs (e.g., camera/controller) with low-latency streaming generation, rather than discrete prompt-to-video clips.
- [**Photoshop is cooked, Nano Bananas manipulation is insane.**](https://www.reddit.com/gallery/1n2fxjn) ([Score: 2064, Comments: 225](https://www.reddit.com/r/singularity/comments/1n2fxjn/photoshop_is_cooked_nano_bananas_manipulation_is/)): **Post showcases a state-of-the-art AI image-editing/inpainting workflow demonstrating high-fidelity object and scene manipulation—contrasted with early [Stable Diffusion](https://stability.ai/) failure modes like “extra limbs” and “missing fingers.” The linked gallery ([reddit.com/gallery/1n2fxjn](https://www.reddit.com/gallery/1n2fxjn)) suggests strong structural coherence and texture blending for large edits, but residual failure cases remain (e.g., an “extra finger” artifact) and fine-grained control over micro-features still lags, especially on faces.** Commentary notes that while results are impressive, Photoshop isn’t “cooked” yet: users still catch anatomical artifacts and report that precise facial edits are difficult, implying AI tools are excellent for global/semantic edits but unreliable for small, identity-preserving adjustments.
    - Progress noted from early Stable Diffusion artifacts (extra limbs/fingers) to current near-photoreal manipulation, but hand/finger fidelity remains an edge case—users still spot issues like an extra finger in the final image. This reflects ongoing weaknesses of diffusion/inpainting on high-frequency anatomical details (hands), despite major improvements in global coherence and realism.
    - Users report strong performance for broad edits (e.g., compositional/object changes) but poor control over small facial details; *“try to edit small facial details and it’s hell.”* This aligns with known limitations where localized, fine-grained edits can degrade identity or introduce artifacts, suggesting a need for better mask-aware conditioning, control nets, or higher-resolution latent editing to preserve microfeatures.
- [**Apple AI vs Galaxy Al vs Xiaomi Al REMOVE tool**](https://v.redd.it/w7ckphp7oplf1) ([Score: 4507, Comments: 407](https://www.reddit.com/r/ChatGPT/comments/1n26571/apple_ai_vs_galaxy_al_vs_xiaomi_al_remove_tool/)): **A short video (blocked for us with** `403 Forbidden` **on [v.redd.it](http://v.redd.it/)) purportedly compares the consumer “remove/magic eraser” inpainting tools from Apple, Samsung Galaxy, and Xiaomi on the same image, highlighting differences in fill quality after erasing a subject. No implementation details, models, or benchmarks are provided in the post; it appears to be a visual A/B/C of object-removal outputs typical of on-device photo editors. Link: https://v.redd.it/w7ckphp7oplf1** Top comments argue **Apple’s** result is notably worse than competitors and liken it to a basic Paint 3D-style “magic eraser,” with little substantive technical discussion beyond that sentiment.
    - Several commenters imply the “remove” tools are functionally similar across vendors: Apple’s new Photos “Clean Up” under **Apple Intelligence** vs **Google**’s Magic Eraser, **Samsung Galaxy AI**’s Object/Generative Edit, **Xiaomi**’s AI Erase, and even **Microsoft Paint**’s Magic Eraser—most are variants of semantic segmentation + generative inpainting. Key differences are deployment and privacy: Apple emphasizes on‑device inference on `A17 Pro`/`M‑series` NPU for supported devices ([Apple Intelligence](https://www.apple.com/apple-intelligence/)), while Samsung often flags cloud-backed edits ([Galaxy AI](https://www.samsung.com/global/galaxy/ai/)); Xiaomi’s implementation varies by model/region. Quality tends to hinge on mask accuracy and the inpainting model (diffusion vs patch-based), background texture complexity, and promptless vs guided fills.
    - Noting the omission of **Google Pixel**, its Magic Eraser debuted with Pixel 6/Tensor and later expanded via Google Photos/One, with some features processed server-side (e.g., Magic Editor) and others on-device depending on hardware/app version ([Google Photos Magic Eraser](https://support.google.com/photos/answer/11910009), [Magic Editor](https://blog.google/products/photos/magic-editor/)). Pixel’s stack also includes Best Take and Audio Magic Eraser, indicating a mature, vertically integrated pipeline leveraging the Tensor ISP/NPU; in practical comparisons, object removal quality is generally in the same class as Apple/Samsung/Xiaomi but may differ on fine textures and edge continuity where diffusion-based inpainting shines.
- [**Turning drawings into photos in 2025**](https://v.redd.it/qtckbsr0jnlf1) ([Score: 447, Comments: 38](https://www.reddit.com/r/ChatGPT/comments/1n1xghk/turning_drawings_into_photos_in_2025/)): **A demo post shows a tool that converts hand drawings/sketches into photorealistic images. The embedded media at [v.redd.it/qtckbsr0jnlf1](https://v.redd.it/qtckbsr0jnlf1) returns** `403 Forbidden` **(Reddit block page), so model details, benchmarks, or implementation specifics can’t be verified from the post; no technical specs are provided in-thread.** Top comments flag friction: requiring a credit card for a “free” trial, and ask why not use ChatGPT instead—implying users may prefer built-in or open alternatives. For precise sketch-to-photo control, commenters typically reference diffusion img2img workflows (e.g., SDXL + [ControlNet](https://github.com/lllyasviel/ControlNet)) over general-purpose chat models.
- [**This post got 27K upvotes 3 years ago - before Reddit hated AI**](https://i.redd.it/tfu54shi6olf1.jpeg) ([Score: 716, Comments: 186](https://www.reddit.com/r/ChatGPT/comments/1n209ey/this_post_got_27k_upvotes_3_years_ago_before/)): **The image (https://i.redd.it/tfu54shi6olf1.jpeg) is a representative example of early text-to-image AI aesthetics (~2021–2022, pre–Stable Diffusion): surreal, low-coherence compositions with CLIP-guided, dreamlike artifacts rather than photorealism. The title highlights it reached** `27K` **upvotes “3 years ago,” underscoring how simply getting interpretable outputs was then noteworthy; compared to modern diffusion systems, these older pipelines produced painterly textures, warped structures, and ambiguous forms that many associated with the genre’s early charm.** Commenters note that early AI art wasn’t seen as a threat to human artists and had a distinctive, abstract look that some now miss; one suggests revisiting older models to recreate that vibe, while another remarks on how “bad” the quality seems by today’s standards.
    - Several commenters contrast 2021–2022-era text-to-image systems—where producing anything "interpretable" felt monumental—with today’s near-photoreal outputs. Early pipelines (e.g., VQGAN+CLIP, DALL·E mini/Craiyon, early Stable Diffusion 1.x) tended to yield abstract/dreamlike results due to CLIP-driven guidance and weaker priors; by 2023–2025, larger diffusion models (e.g., SDXL, Midjourney v6) markedly improved resolution, compositional reliability, and prompt adherence through larger backbones, better datasets, and improved sampling/finetuning. See SDXL overview: https://stability.ai/news/stable-diffusion-sdxl-1-announcement and MJ v6 notes: https://docs.midjourney.com/docs/model-versions#version-6
    - There’s technical interest in deliberately using older models to reproduce the "weird" aesthetic: artifacts emerged from low training resolutions (`256–512px`), smaller U-Nets, limited/ noisier datasets, and strong classifier-free guidance producing oversaturated textures and surreal compositions. Samplers like early DDIM/PLMS and CLIP-guided losses accentuated odd geometry and text blending, yielding the "internet through a distorted glass" vibe that’s harder to get from modern, well-regularized models with advanced samplers (e.g., DPM-Solver++) and robust conditioning.
    - A side-by-side theme emerges via a 3-year-old example image (https://www.reddit.com/r/PeterFHamilton/s/9a3H1j4tQZ) versus a "Same prompt 2025" render (https://preview.redd.it/q7aslq30kolf1.jpeg?width=1024&format=pjpg&auto=webp&s=4b13a50c42da2c8b531c7b7685e3610e67000af4). The latter implies major gains in fidelity (anatomy, lighting, texture detail), text and prompt-following, and artifact suppression—likely attributable to larger training corpora, higher native resolutions, improved conditioning (prompt/negative prompts), and better inference tooling (refiners, upscalers).

### 3. AI Policy: ChatGPT Scanning, Regulation Memes, and Jobs Debate

- [**OpenAI Says It's Scanning Users' ChatGPT Conversations and Reporting Content to the Police**](https://futurism.com/openai-scanning-conversations-police) ([Score: 697, Comments: 259](https://www.reddit.com/r/OpenAI/comments/1n2138e/openai_says_its_scanning_users_chatgpt/)): **The post claims OpenAI “scans” ChatGPT conversations and reports users to police. OpenAI’s own [Privacy Policy](https://openai.com/policies/privacy-policy) and [Usage Policies](https://openai.com/policies/usage-policies) confirm chats may be reviewed by automated systems and authorized personnel for abuse/safety, and that content can be disclosed to law enforcement when required by law or to prevent harm; data-use controls (e.g., training opt-outs, enterprise retention settings) exist, but routine moderation/abuse-detection applies broadly, with few publicly documented thresholds or audit details.** Commenters connect this to governance and government ties: noting former NSA director **Paul M. Nakasone** (`2018–2024`, nominated under Trump) joining OpenAI’s board ([OpenAI](https://openai.com/blog/welcoming-paul-nakasone-to-openai-board), [Wikipedia](https://en.wikipedia.org/wiki/Paul_M._Nakasone)), alleging an unverified `$200M` DoD contract, and urging clearer disclosure about employee access and privacy boundaries.
    - Several commenters quote OpenAI’s stated policy that potentially violent intent triggers "specialized pipelines" with human review and, if an "imminent threat of serious physical harm" is determined, possible referral to law enforcement. This describes a two-step moderation architecture: automated detection → human escalation → enforcement/reporting, aligned with industry trust-and-safety norms; see OpenAI’s policy pages for context (e.g., https://openai.com/policies/usage-policies).
    - There’s a technical privacy concern about insider access: users want explicit disclosure of data flows (collection, retention windows, training use), access controls (who can view chats and under what approval), and auditing (logged reviewer access, redaction of PII). Commenters note most major platforms run abuse-detection scanning at scale, but ask OpenAI to clarify consumer vs. enterprise defaults, opt-outs, and how “snooping” risk is mitigated (e.g., segmentation, encryption-at-rest, least-privilege).
    - Governance/affiliation implications are raised: a former NSA Director reportedly sits on OpenAI’s board (see OpenAI’s announcement: https://openai.com/blog/paul-nakasone-joins-openai-board-of-directors), and a claimed `200M` DoD contract suggests deeper government integration. Technically, this could influence reporting workflows, regulatory alignment, and thresholding for law-enforcement cooperation, though commenters debate whether this differs materially from standard practices at other large tech firms.
- [**If AGI is so "inevitable", they shouldn't care about any regulations**](https://i.redd.it/hn7w2vb02qlf1.png) ([Score: 342, Comments: 44](https://www.reddit.com/r/ChatGPT/comments/1n26s3h/if_agi_is_so_inevitable_they_shouldnt_care_about/)): **The image is a meme highlighting the rhetorical tension in AI policy: companies say AGI is “inevitable” globally while also warning that domestic regulation could “kill the industry.” Commenters distinguish scopes: “inevitable” refers to worldwide progress, whereas stringent U.S.-only rules could shift capability development abroad (e.g., to China). Others argue much current legislation is naïve or weaponized for competitive advantage, proposing regulation target downstream human impacts (safety, labor protections) rather than banning core AI R&D—drawing an Industrial Revolution analogy.** Debate centers on whether to regulate the technology itself versus outcomes and externalities; some see domestic overregulation as self-sabotage amid geopolitical competition, while others stress the need for mature, impact-focused governance to avoid stifling innovation.
    - Several commenters argue incumbents’ pro-regulation stance is largely about regulatory capture and cross-border arbitrage: big labs like **OpenAI, Google, Microsoft, Anthropic** can absorb compliance costs and shift training to permissive jurisdictions. Example: Japan’s Copyright Act Art. 30-4 provides a broad text/data mining exception “regardless of the purpose,” enabling use of copyrighted materials for ML training without permission, which firms can leverage to mitigate IP risk ([CRIC English summary](https://www.cric.or.jp/english/clj/cl2.html#3)).
    - On the EU side, the finalized AI Act introduces a GPAI “systemic risk” regime (with a proxy threshold around `>10^25` training FLOPs) that triggers documentation, model evals/red-teaming, cybersecurity, and copyright-risk mitigation duties for frontier models ([overview](https://artificialintelligenceact.eu/)). Critics note a disconnect: many labs publicly back “AI regulation” yet contest IP liability in court (e.g., fair-use defenses in the US, such as the NYT v. OpenAI/Microsoft case: [NYT coverage](https://www.nytimes.com/2023/12/27/business/media/new-york-times-openai-lawsuit.html)) and lobbied to soften compute triggers and obligations.
    - Geopolitical asymmetry is highlighted: even if US regulation constrains domestic players, China will continue advancing with domestic LLMs (e.g., Alibaba’s Qwen series, Baidu ERNIE) and non‑US accelerators (e.g., Huawei Ascend) despite US export controls on **A100/H100**class chips ([BIS rule, Oct 2023](https://www.federalregister.gov/documents/2023/10/25/2023-22114/export-controls-on-semiconductor-manufacturing-items-interim-final-rule)). Open models like **Qwen2-72B** report near-GPT‑3.5 performance on key benchmarks such as MMLU ([arXiv](https://arxiv.org/abs/2407.10671)), suggesting unilateral regulation may shift where progress happens rather than stop it.
- [**People thinking AI will end all jobs are hallucinating- Yann LeCun reposted**](https://www.reddit.com/gallery/1n2h6qu) ([Score: 460, Comments: 297](https://www.reddit.com/r/singularity/comments/1n2h6qu/people_thinking_ai_will_end_all_jobs_are/)): **Thread discusses a repost by Meta’s Chief AI Scientist [Yann LeCun](https://yann.lecun.com/) asserting that current AI systems cannot plausibly end all jobs, i.e., the claim isn’t supported by present capabilities; the repost itself includes no new benchmarks or empirical results. The referenced Reddit link [returns 403](https://www.reddit.com/gallery/1n2h6qu), so discussion centers on capability limits vs. extrapolation rather than new data.** Commenters argue the stance is presentist: today’s constraints (e.g., `~10×` verification overhead vs. generation) may shrink with rapid progress, so inferring long‑term labor impact from current limits is “shortsighted.” Others note the debate is framed in absolutes and paraphrase the post as *“AI won’t end all jobs because it can’t right now,”* which they view as unsubstantiated for the future.
    - Several commenters focus on the current verification bottleneck, citing a roughly `10×` slowdown for checking model outputs versus generating them. The debate is whether that ratio is a transient artifact of today’s pipelines (manual review, weak auto-eval) or a hard limit; critics argue verification can be automated/parallelized via stronger test synthesis, formal checks, and domain-specific oracles, reducing effective latency and cost at scale, thus weakening arguments about limited job replacement based on today’s verification overhead.
    - LeCun’s past "never" claims (e.g., about spatial reasoning) are challenged by progress in multimodal/world-model systems that demonstrate emerging spatial and physical reasoning. Commenters point to interactive video/world models (e.g., Google DeepMind’s Genie line: https://deepmind.google/discover/blog/genie-generating-interactive-worlds-using-pixels/) and VLMs evaluated on spatial/relational benchmarks like CLEVR (https://cs.stanford.edu/people/jcjohns/clevr/), arguing that capability trends undermine categorical forecasts about what AI “cannot” do.
- [**The double standards are sickening!**](https://www.reddit.com/r/ChatGPT/comments/1n2ewvj/the_double_standards_are_sickening/) ([Score: 215, Comments: 138](https://www.reddit.com/r/ChatGPT/comments/1n2ewvj/the_double_standards_are_sickening/)): **OP argues regulators are extrapolating from an isolated AI-related incident to justify sweeping “guardrails” on LLMs like [ChatGPT](https://chat.openai.com/), while long-documented harms from engagement-optimized [recommender systems](https://en.wikipedia.org/wiki/Recommender_system) powering [Instagram](https://www.instagram.com/), [TikTok](https://www.tiktok.com/), [Snapchat](https://www.snapchat.com/), and [YouTube](https://www.youtube.com/) receive comparatively little constraint. They frame this asymmetry as political economy: entrenched, revenue-generating social platforms are tolerated, whereas the “new shiny AI” is easier to regulate despite providing practical mental-health-adjacent utility (e.g., safe late-night conversational “presence,” journaling support). The post contrasts AI’s conversational utility with social feeds driving FOMO, bullying, body dysmorphia, and other mental-health impacts.** Commenters attribute the policy gap to incentive misalignment and rent-seeking among politicians, and characterize LLMs as a “presence engine” that aids structured journaling and psychoeducation—distinct from dopamine-maximizing engagement loops—while acknowledging it is not a licensed-therapy substitute.
    - Discussion frames **GPT/LLMs** as a “presence engine” for late-night support and journaling: users report qualitative improvements in writing and self-reflection over ~a year by using structured prompts and psychological frameworks (e.g., CBT-style exercises) rather than clinical diagnoses. Emphasis that it’s not a clinician but can scaffold coping strategies through consistent, nonjudgmental, task-focused dialogue.
    - Technical contrast with engagement-optimized social media: unlike feeds tuned via reinforcement learning for `time-on-platform` and dopamine loops, LLM chats are turn-based and can be configured with safety guardrails (e.g., self-harm classifiers, de-escalation responses, and crisis resources). Commenters note search engines may surface suicide methods without context, highlighting a design trade-off between open indexing and proactive safety interventions in AI assistants.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. OpenAI Product Push: Realtime, Web Search, and Codex**

- **OpenAI Speaks Up with GPT‑Realtime**: OpenAI unveiled **gpt‑realtime**, a developer‑facing speech‑to‑speech model, alongside updates to the **Realtime API** in [Introducing GPT‑Realtime](https://openai.com/index/introducing-gpt-realtime/). The release emphasizes low‑latency, interactive voice experiences and positions realtime as a first‑class API surface for multimodal apps.
    - Community reactions highlighted excitement for voice-native agents and tool use, with early adopters eyeing streaming hooks and session controls documented on [OpenAI Live](https://openai.com/live/). Engineers framed the move as OpenAI’s push to make **always‑on conversational interfaces** practical at scale.
- **Web Search Slashes Spend by 60%**: OpenAI announced **domain filtering**, explicit **source reporting**, and a **60% price cut** for Web Search in the Responses API (from $25 to $10 per 1k calls) per [OpenAI Devs’ update](https://x.com/OpenAIDevs/status/1960425260576334274). The update targets factual grounding and cost control for production chatbots that pull live context.
    - Builders said cheaper search will unlock broader usage for retrieval‑augmented features, noting that explicit sources simplify **auditing and trust** in outputs. One member summarized the appeal as making it easier to *"pull factual data from the web to add context"* while keeping spend predictable.
- **Codex Comes Back, Claims GPT‑5 Power**: OpenAI teased a refreshed **Codex** purportedly powered by **GPT‑5**, adding a VS Code/Cursor extension, GitHub auto‑reviews, and a rebuilt CLI with **image input**, per this [OpenAI Devs post](https://xcancel.com/OpenAIDevs/status/1960809814596182163). The announcement pitches stronger code understanding and multimodal developer workflows.
    - Developers expect sharper code edits and review automation, but want benchmarks and latency numbers before large migrations. Teams noted the new CLI + image input could streamline **repo‑centric** tasks and visual debugging in CI.

**2. Frontier & Open-Source Model Drops and Decoding Tricks**

- **MAI‑1 Muscles onto the Leaderboard**: Microsoft’s **MAI‑1‑preview** landed at **#13** on the LMArena text leaderboard, now testable via [LMArena’s Text Leaderboard](https://lmarena.ai/leaderboard/text). Community notes: trained on ~**15,000 H100s**, the preview feels slow with a small context window, yet shows promise for **webdev‑style** reasoning.
    - Early testers reported errors with longer prompts and mixed reasoning depth, quipping that *“mai‑1 thinks it’s einstein”* but trips on context length. Despite quirks, the reception frames it as a notable **in‑house MoE** milestone from Microsoft.
- **Hermes‑4 Leaks, Then Leaves**: The **NousResearch/Hermes‑4‑14b‑chat‑template‑retrain** briefly appeared on Hugging Face before going private; mirrors circulated quickly, and early runs looked solid ([model card snapshot](https://huggingface.co/NousResearch/Hermes-4-14b-chat-template-retrain)). The unplanned window let users test the **chat‑template retrain** with reports of robust instruction following.
    - Users said the model *"works fine for now"* and noted a new chat‑template flag enabling **thinking=True** prompts. The incident reinforced interest in **lightweight instruction‑tuned** 14B models for local IDEs and agents.
- **llama.cpp Tries Speculative Decoding**: A draft PR adds **speculative decoding** to llama.cpp with a working prototype, inviting accuracy/perf testing ([llama.cpp PR #15225](https://github.com/ggml-org/llama.cpp/pull/15225)). Early user feedback reported mixed accuracy, suggesting further tuning is needed for general use.
    - Discussion compared techniques like **MTP (Memory Token Prediction)** used by **DeepSeek** and **GLM**, noting MoE models can be tricky for speculation. Practitioners emphasized that **token distribution shifts** after instruct‑tuning can affect draft‑accept rates.

**3. Retrieval and Agent Infrastructure Heats Up**

- **Gensee Shrinks Web Retrieval to One Call**: The **Gensee Search Agent** wraps search, crawl, and browse into a single API call with retries/fallbacks and BFS‑style breadth search, claiming **+23% GAIA** accuracy and a field report of **+40%** after swapping it in ([tech blog](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html)). A 5‑minute [walkthrough video](https://www.youtube.com/watch?v=nRdVY7dWVqE) demos goal‑aware extraction that filters off‑target pages early.
    - Engineers liked the consolidated interface and fault‑tolerant design for production agents, calling out the appeal of **parallel search + tight content extraction**. Teams plan bake‑offs against homegrown retrievers to validate the GAIA gains.
- **Cloudflare Ships AI Gateway; Devs Benchmark It**: Cloudflare refreshed its **AI Gateway** with observability and routing features ([Cloudflare blog](https://blog.cloudflare.com/ai-gateway-aug-2025-refresh/)). One test routed calls through the gateway to **OpenRouter** for `llama-3.1-8b-instruct`, clocking ~**20s** with `only: ['cloudflare']` vs **3s** direct.
    - Some suggested the feature set overlaps OpenRouter, while others valued **traffic control and analytics** at the edge. Benchmarkers flagged the latency delta as a tuning target before adopting gateway‑mediated inference in prod.
- **Prime Intellect Opens an RL Environment Hub**: **Prime Intellect** launched an open-source **Environments Hub** to crowdsource and share RL environments ([announcement](https://xcancel.com/PrimeIntellect/status/1960783427948699680)). The hub aims to standardize environment sharing for **agentic** evaluations and training pipelines.
    - In replies, @karpathy said he’s *“bullish on environments and agentic interactions”* but *“bearish on reinforcement learning specifically”* ([Karpathy reply](https://x.com/karpathy/status/1960803117689397543)). The community read this as a nudge toward **environment‑rich evals** even if classic RL isn’t center stage.

**4. Builder Tooling Gets Friendlier**

- **LM Studio 0.3.24 Polishes UX and Adds Seed‑OSS**: **LM Studio 0.3.24** shipped support for **ByteDance/Seed‑OSS** models and markdown improvements like a sticky copy‑code button and better table/code rendering ([release notes](https://lmstudio.ai/blog/lmstudio-v0.3.24)). The refresh also tweaks `lms` output styling to make local dev loops smoother.
    - Users welcomed the nicer code navigation and formatting for prompt engineering sessions, plus the expanded model catalog ([Seed‑OSS‑36B page](https://lmstudio.ai/models/bytedance/seed-oss-36b)). Local‑first devs called it a quality‑of‑life bump for **desktop inference**.
- **SmolFactory Spins Up Simple Training in Spaces**: **SmolFactory** launched as a Hugging Face Space for point‑and‑click model training, shipping with the **GeneReviews** dataset ([SmolFactory Space](https://huggingface.co/spaces/Tonic/SmolFactory)). The author also published a how‑to [blog post](https://huggingface.co/blog/Tonic/smolfactory) covering dataset selection and training flows.
    - Builders liked the minimal UI for quick fine‑tunes on hosted GPUs and the curated biomedical dataset as a starter. The community sees Spaces‑hosted trainers as a path to **lower the barrier** for domain‑specific SFT.
- **Tiny Model, Old Laptop: AuroraStories‑12M Ships**: A contributor trained **AuroraStories‑12M** in under 24 hours on an old laptop and released it on Hugging Face ([AuroraStories‑12M](https://huggingface.co/ThatHungarian/AuroraStories-12M)). The demo underscores how **small models + GGUF** builds can be practical for hobbyists and edge devices.
    - Followers praised the author’s focus on **compact checkpoints** with lots of **gguf** artifacts for easy local use. The thread reinforced interest in **ultra‑light LLMs** for offline agents and embedded tasks.

**5. Multimodal Media: Video and Audio Level Up**

- **Tencent Foley Fuses Audio to Video**: Tencent open‑sourced **HunyuanVideo‑Foley**, a Text‑Video‑to‑Audio framework trained on **100k hours** and built on **MMDiT** ([release post](https://xcancel.com/TencentHunyuan/status/1960920482779423211)). The system generates context‑aligned soundscapes that match video content for richer multimodal outputs.
    - Researchers called it a strong **audio‑sync baseline** for creative tools and post‑production. Devs anticipate experiments combining Foley with **video diffusion** and editing pipelines for end‑to‑end **T2V2A** workflows.
- **KREA Claims Real‑Time Video Generation**: **KREA AI** unveiled its first **real‑time video generation** model and opened beta signups, targeting instant creative content, music videos, and seasonal ads ([beta announcement](https://xcancel.com/krea_ai/status/1961074072487620635)). The teaser positions KREA as a latency‑first contender for interactive visuals.
    - Creators expressed interest in **live previews** and camera‑ready effects for short‑form video pipelines. The community wants resolution, fps, and **latency metrics** before comparing KREA to incumbents.
- **MIDAS Makes Digital Humans Move**: The paper **“MIDAS: Multimodal Interactive Digital‑human Synthesis via Real‑time Autoregressive Video Generation”** showcases a real‑time AR‑video approach for interactive avatars ([MIDAS paper](https://huggingface.co/papers/2508.19320)). The work highlights responsive, autoregressive generation tuned for **digital‑human** synthesis.
    - Discussion connected MIDAS to the broader push for **controllable real‑time characters**, bridging speech, motion, and expression. Practitioners are eyeing integration with **voice agents** and **gesture control** for end‑user applications.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Bill Chen Leaks Images v2**: **OpenAI's Bill Chen** leaked an **AI-generated photo**, seemingly from **Images V2**, in a now-deleted post, prompting community discussion on its authenticity and potential improvements over **Images V1**.
   - Members debated whether the image was real, and the extent to which it showed performance improvements.
- **GPT-5 Auto-Thinking Surprises Users**: Users observed that **GPT-5 Thinking** puts in genuine effort in utilizing the **multi-step** functionality, leading to **more sources** in search results when prompted with **'think hard'**.
   - Some users have also noted that **Grok 4** search is now as good as deep search.
- **GPT-4.1 Error Troubles Users**: Users reported encountering the error message **"Used GPT-4.1 because Grok 4 was inapplicable or unavailable"** during their interactions.
   - Members noted the error's increasing frequency, with some reporting models switching mid-conversation.
- **Web Dev Costs Face AI-Driven Debate**: Members debated the appropriate pricing for web development projects in the age of AI, contrasting rates for freelancers in the US versus India.
   - Discussion involved the amount of code to be used with one member pointing out that a $5k project was a *good deal*.
- **Users Struggle Choosing Model on Playground**: Users reported difficulties in selecting a model within the Playground interface.
   - One user posted *Can't choose the model on the playground* with an attached image.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Dries Up After Supabase Failure**: OpenRouter experienced a **49-minute** outage due to its database provider, [Supabase going down](https://supabase.com).
   - The team is improving **redundancy** to prevent future outages and apologized for the downtime.
- **Dashboard Code Goes Public!**: The code for the dashboard is now [publicly available on GitHub](https://github.com/lorenzozane/openrouter-costs-visualizer).
   - The author welcomes contributions and feedback, suggesting that screenshots attract more attention than text descriptions.
- **OpenRouter Users Roleplay Through Outage**: OpenRouter experienced an [outage](https://status.openrouter.ai/), leading to humorous reactions and role-playing in the Discord chat, with users joking about corpo wars and the AI apocalypse.
   - One user quipped, *Get up samurai, we've got a city to fuck*, while others expressed addiction and the need for AI companionship during the downtime.
- **Requesty Promoters Banned Amid Scam Accusations**: Promoters of another AI platform called [Requesty](https://www.requesty.ai) were banned after users called it *vibecoded trash with 1000 vulnerabilities.*
   - One member posted [a scammer GIF](https://tenor.com/view/scammers-scam-alert-gif-13801618) in response to the team's investigation announcement.
- **Cloudflare AI Gateway Challenges OpenRouter**: Cloudflare launched an [AI Gateway](https://blog.cloudflare.com/ai-gateway-aug-2025-refresh/) which was said to have copied OpenRouter, and one member tested using **Cloudflare's AI Gateway** to access **OpenRouter** to call `llama-3.1-8b-instruct`.
   - Calling `llama-3.1-8b-instruct` with the `only: ['cloudflare']` parameter took **20 seconds**, while without it, it was **3 seconds**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Professional Infrastructure Thrashes Spotty Setups**: Members discussed how [Spot instances](https://aws.amazon.com/ec2/spot/) are viable in distributed compute only with a *professionally-built infrastructure*, emphasizing that a *single-node setup is cooked* if relying solely on Spot.
   - One member quipped that even **OpenAI** had *20 HPC engineers* managing the network during **GPT-4 training** highlighting the complexities at scale.
- **Grok Code Gains Fans for Speedy Iteration**: Despite being ignored initially, members discussed how [Grok Code](https://openrouter.ai/x-ai/grok-code-fast-1) is decent and super fast, so iterating is rapid.
   - Although **Grok 4** is nearly unusable, **Anthropic** is still living due to its tool calls.
- **GPT-OSS Boasts Long Context and Reddit Buzz**: The new GPT-OSS release features a **60k context length**, and a member posted it on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1n2jraj/gptoss_finetuning_now_with_60k_context_length_and/).
   - Members discussed the need for future *Reward Feedback Training (RFT)* and **GPT-OSS Pro**.
- **Crafting an AI Clone of a Person**: Users describe scraping Discord channels to clone personalities, highlighting the process of converting **HTML to TXT, CSV, and Parquet** for feeding into models like **phi-4-14b**.
   - One user shared that they cloned 5 of their friends, with permission, and then shared how the clone responded to a bunch of funny questions, resulting in amusement from their friends.
- **CUDA troubles with Conda**: A user encountered crashes when using `from unsloth import FastLanguageModel` on a node with 32GB RAM after a fresh Unsloth install, but found it worked on a node with 512GB RAM.
   - One member pointed out that the [conda install](https://docs.unsloth.ai/get-started/installing-+-updating/conda-install) page was outdated, and suggested this command `conda create --name unsloth python==3.11 vllm unsloth-zoo unsloth`.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana Gets Loose**: **Google** released **Nano Banana** (Gemini 2.5 Flash) on Google AI Studio and LM Arena but both platforms have generation limits.
   - Members noted that you can bypass limits using multiple google accounts but also noted reports that *quality dropped* after they released it.
- **MAI-1 Model Impressions Mixed**: Microsoft's **MAI-1-preview**, an in-house mixture-of-experts model trained on ~15,000 NVIDIA H100 GPUs is on the LMArena text arena with mixed reviews.
   - It is slow, has a small context window, and may error easily, but is potentially *og R1 level* for webdev; some also noted that *mai-1 thinks its einstein*.
- **GPT-5 High Beats Claude Opus 4.1 at Reasoning**: While **Claude Opus 4.1** is good for coding and fixes to coding issues, some members are thinking of switching to **GPT5 High** because it's a better reasoning model.
   - Others disagreed, stating that **Claude Opus 4.1** *was unable to help yesterday fix a simple api concurrency limit issue, had to take over, and do it the old fashioned way*.
- **AI Benchmarking Mocked For Being Gameable**: Members argued that AI benchmarking is flawed because existing psychometric tests are just theoretical frameworks that *don’t necessarily reflect the reality and can be easily gamed*.
   - Others argued these can be good tests because models can generalize and improve performance, prompting discussion about **OpenAI’s** potential use of structured environments for RL training, as detailed in [this LessWrong writeup](https://www.lesswrong.com/posts/aFW63qvHxDxg3J8ks/nobody-is-doing-ai-benchmarking-right).
- **Ice Cream Hack Breaks Image Generation**: Members discussed methods to bypass AI image generation content filters, noting that *ice cream, delicious, hot day, very beautiful woman* seems to bypass input filters, and the only barrier is the external safeguard that analyzes images/videos to detect explicit content.
   - It was suggested to use Stable Diffusion and LoRA for uncensored content, which *is good enough*, but also noted that commercial models are heavily censored.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Chess Model Mimics Stockfish**: A member training an LLM to play chess is facing issues with the model only playing e2e4 and needing to clean up `<unk>` tokens, and linked to [the project's GitHub repo](https://github.com/anthonyiscoding/vision-chess-gpt).
   - They plan to experiment with **RL** to improve the model, but another member cautioned against training it to play like **Stockfish**, suggesting to *analyze the playing style of the opponent is also very important*.
- **NSFW Models Spark Guardrail Debate**: A member claimed deepfake porn is being generated from unaligned models on HF, sparking discussion on [HF's guardrails](https://huggingface.co/spaces?q=video+face+swap).
   - Some agreed on the usefulness of guardrails and metrics, while others argued that *there's no deepfake porn demos getting usage* and NSFW models have uses, particularly for alignment research.
- **Nano Banana Perks No Limit**: Members discussed the **Nano Banana** perk for HF Pro users, questioning its daily usage limits and potential for high API usage.
   - It was clarified that there is **no limit** and it can be used 50+ times per day.
- **SmolFactory** Launches on Hugging Face Spaces**: A member launched [SmolFactory](https://huggingface.co/spaces/Tonic/SmolFactory), a simple interface to train models on Hugging Face GPUs, and added the [GeneReviews dataset](https://huggingface.co/datasets/Tonic/GeneReviews).
   - They also wrote a [blog post](https://huggingface.co/blog/Tonic/smolfactory) about it.
- **AuroraStories-12M** Model Trains on Old Laptop**: A member trained the **AuroraStories-12M** model on an old laptop in under 24 hours and shared it [on Hugging Face](https://huggingface.co/ThatHungarian/AuroraStories-12M).
   - Another member noted following this user because of *small models and lots of gguf downloads*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Gets ByteDance Boost**: **LM Studio 0.3.24** adds [support for ByteDance/Seed-OSS models](https://lmstudio.ai/models/bytedance/seed-oss-36b) and markdown enhancements.
   - Improvements include sticky copy code buttons, refined `lms` output style, and better rendering of tables and code blocks per the [release notes](https://lmstudio.ai/blog/lmstudio-v0.3.24).
- **FastAPI Fires Up Reasoning Streams**: A member is including a **FastAPI server** to accelerate the **Reasoning Stream** and client-wide processes.
   - The implementation aims to improve processing speeds across various tasks.
- **Quantization Creates Accuracy Quandaries**: Quantizing models can lower accuracy due to loss of detail, especially in code tasks where token precision is crucial.
   - While some models tolerate **Q4 quantization** well, others like **Qwen3** are very sensitive to detail loss.
- **Ryzen NPU Performance Stalls on Ubuntu**: A user reported only **1 token/second** using **Ryzen NPUs** on **Ubuntu 25.04** and inquired about performance improvements.
   - It was clarified that *'NPUs are not supported by llama.cpp which fuels LM Studio'*, with a link to [AMD's open-source project for running local LLMs on Ryzen AI](https://www.amd.com/en/developer/resources/technical-articles/gaia-an-open-source-project-from-amd-for-running-local-llms-on-ryzen-ai.html).
- **Macs Battle Windows in Memory Match**: A user highlighted that **Macs have unified memory**, citing a case where **126GB out of 128GB** was used for GPU processing at **~400GB/s bandwidth**.
   - They argued that this outpaces top-tier Windows laptops with ~**115GB/s** bandwidth, making CPU offloading less effective due to weak CPU processing.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Giants Team Up for Safety Audits**: OpenAI and AnthropicAI collaborated to test each other’s models, publishing the [results](https://openai.com/index/openai-anthropic-safety-evaluation/) of their safety and alignment evaluations.
   - This collaboration signals a focus on **transparency and accountability** in AI safety, despite competition on capabilities.
- **GPT-Realtime Debuts with API Refresh**: OpenAI introduced **gpt-realtime**, their newest speech-to-speech model for developers, alongside updates to the [Realtime API](https://openai.com/live/).
   - Members seem excited about it, even though not much has been shared beyond the name.
- **Veo 3's Video Generation Sparks Discussion**: Members discussed **Gemini's Veo 3** video generation, noting it requires a **Google One/Gemini Pro or Ultra** subscription.
   - Users pointed out that **Google AI Studio** offers the *outdated Veo 2* model and **Veo 3** is currently too expensive to provide for free.
- **Grok Coder's Free Trial Faces Scrutiny**: **Grok Coder** is being offered free for a week via [kilo code](https://kilo.code), seemingly a promotion available everywhere.
   - Some users found its performance to be *"o1 mini level bad"*.
- **Context Cascade Architecture Announced**: Engineers at the **Institute for Cognitive Architectures** revealed their prototype of **Context Cascade Engine** (**CCA**) to expand beyond the traditional context window of **large language models**.
   - **CCA** is a *multi-level approach* to managing memory in LLMs, focusing on structured forgetting and strategic recall through design.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Cuts Web Search Prices 60%**: **OpenAI** announced enhancements to the Web Search in the Responses API, featuring [new domain-filtering](https://x.com/OpenAIDevs/status/1960425260576334274), explicit source reporting, and a **60% price cut** (from $25 to $10 per 1k calls).
   - The change promises to make it even more economic to pull factual data from the web to add context to chatbot conversations.
- **Prime Intellect Opens RL Environment Hub**: **Prime Intellect** launched the [Environments Hub](https://xcancel.com/PrimeIntellect/status/1960783427948699680), an open-source community platform for crowdsourcing and sharing **reinforcement-learning environments**.
   - Despite the fanfare, `@karpathy` replied on that same Prime Intellect [tweet](https://x.com/karpathy/status/1960803117689397543) being *bullish on environments and agentic interactions* but *bearish on reinforcement learning specifically*.
- **GPT-5 Powers Codex Comeback**: **OpenAI** released a major **Codex** refresh powered by **GPT-5**, including a new VS Code/Cursor extension, GitHub integration for auto-reviews, and a rebuilt CLI with [image input](https://xcancel.com/OpenAIDevs/status/1960809814596182163).
   - The refresh comes with a promise of greater programming capabilities than the earlier version of **Codex**.
- **Tencent Releases HunyuanVideo-Foley**: **Tencent** open-sourced [HunyuanVideo-Foley](https://xcancel.com/TencentHunyuan/status/1960920482779423211), a Text-Video-to-Audio framework that generates context-aligned soundscapes using a **100k-hour training set** and a **multimodal diffusion transformer** (MMDiT) architecture.
   - This release allows developers to experiment with generating realistic audio to match video content.
- **KREA AI Promises Real-Time Video Generation**: [KREA AI](https://xcancel.com/krea_ai/status/1961074072487620635) has unveiled its first **real-time video generation model** and opened a beta signup, allowing users to create instant creative video content, music videos, and seasonal ads.
   - The promise of instant creative video content, music videos, and seasonal ads has garnered attention across many creative circles.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **ScaleML Series Quantifies Quantization**: Day 3 of the **ScaleML** series covered **quantization**, with emphasis on microscaling formats like **MXFP4**, by Prof. Chris De Sa, in a whiteboard format, linked [here](https://www.youtube.com/watch?v=k8PcSGG249Y).
   - Day 4 featured an assortment of topics on **Positional Encodings** by Songlin, linked [here](https://www.youtube.com/watch?v=l6_fdwRvMPk).
- **Nsight Compute Faces 'UnknownError'**: A user reported encountering an `UnknownError` while profiling a CUDA application using Nsight Compute, despite running Nsight Compute as administrator, when profiling the `createVersionVisualization` function.
   - It was suggested to ensure compatibility between **Nsight Compute** and the **CUDA toolkit** as a mismatch can lead to profiling errors. The user has CUDA version 13.0 installed and is using Nsight Compute version 2025.3.0.
- **Inductor Pursues Persistent Matmul**: A user inquired about enabling persistent matmul in inductor codegen, specifically for **BF16** precision, and sought guidance on proper configuration, experimenting with `TORCHINDUCTOR_PERSISTENT_REDUCTIONS` and `ENABLE_PERSISTENT_TMA_MATMUL` flags.
   - To force the use of persistent matmul, it was suggested to set `torch._inductor.config.max_autotune_gemm_backends` to `TRITON` only and use `mode="max-autotune-no-cudagraphs"` during compilation, but even with the correct flags, **Cublas** might still outperform other implementations.
- **ROCm Prepares SPIR-V Support for Kernel Flexibility**: **ROCm** will soon support compiling to **SPIR-V**, a format conducive to machine introspection, opening doors for kernel code modification tools.
   - This advancement could enable external developers to create tools like compute-sanitizer by inserting bounds checks into the kernel more easily, to trace memory accesses and leverage the **GPU's SQTT stream** (used by rocm-compute-viewer) for detailed information.
- **AMD Multi-GPU Devs to Receive Allocations**: Members are wondering if they will have access to an **AMD multi-GPU environment** for development and debugging for the new **AMD competition** hosted on the [Data Monsters website](https://www.datamonsters.com/).
   - They will have access to the environment through AMD's platform, with best people receiving some **SSH access**, additionally, for past competitions, **AMD** provided generous allocations to top-performing teams to accelerate their iteration.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Falsifiability Fires Up AI Research**: Debate sparked on **falsifiability** in AI, balancing *exploratory science* with the need for testable hypotheses, calling out the risk of *crank paths* without rigor.
   - Participants underscored the value of **rigor** and **collaboration** in AI research, weighing the nuances of scientific exploration versus structured inquiry.
- **NeMo v2.0 Faces lm_eval Support Scrutiny**: A user reported errors with **NeMo v2.0 models** in **lm_eval** due to missing config files, requiring community assistance.
   - The community suggested utilizing **NeMo to GPT-NeoX conversion code**, also noting that **NeMo support** is maintained by the **NeMo team**.
- **EleutherAI Discord Cracks Down on Content Quality**: Moderators are aggressively policing content on the EleutherAI Discord, deleting **over 100 messages a week** to uphold high-quality discussions among **AI researchers**.
   - The moderation policy aims to shield the community from *AI-generated slop*, *thinly veiled ads*, and *cranks claiming consciousness breakthroughs*.
- **Forward-Forward Training Forges Ahead**: A member reported a working *7 region mini-brain* using **Forward-Forward (FF) training** with online learning, showcasing promising results in initial tests.
   - Another member suggested calling the model **modules** or **task specific subnetworks/circuits** to sound fancy.
- **Cortex_GPT Ventures into Brain-Like Networks**: **Cortex_GPT**, a brain-like network model featuring cortical columns, regions, 6-layer networking, and signal propagation, is now accessible on [GitHub](https://github.com/JRowe47/cortex_gpt).
   - Some members suggested referring to these models as **PDP** (Parallel Distributed Processing).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Minos Classifier Gets No Love**: The **NousResearch/Minos-v1 classifier** [is available](https://huggingface.co/NousResearch/Minos-v1), but the channel stated that nobody is currently using it.
   - Discussion briefly shifted to speculative decoding.
- **MTP Shines with MoE Models**: Speculative decoding may not work well with **MoE models**, especially sparse ones, but **Deepseek** and **GLM** use **MTP (Memory Token Prediction)**, a related technique.
   - It was also mentioned that the **token distribution** should still be representative after instruct fine-tuning.
- **LlamaCPP Speculates on Decoding**: There is a draft **PR for speculative decoding** in [llamaCPP](https://github.com/ggml-org/llama.cpp/pull/15225) with a working prototype.
   - A user reported mixed results with the implementation, indicating that while functional, the approach *wasn't as good at accuracy* in their setup.
- **Hermes-4 Escapes Before Launch!**: The **Hermes-4-14b-chat-template-retrain** model [appeared](https://huggingface.co/NousResearch/Hermes-4-14b-chat-template-retrain), and was quickly downloaded before it was made private again.
   - Though unofficially released, the model is reported to be working fine for now.
- **Penny For Your Thoughts Sells AI Wisdom**: A new project called **Penny For Your Thoughts** has launched, featuring an AI agent that interviews users to generate unique information and share and sell their expertise via micro-transactions at [pennyforyourthoughts.ai](https://pennyforyourthoughts.ai/).
   - **Penny For Your Thoughts** is powered by **Honcho** & **x402**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gensee Search Agent Debuts as Web Retrieval API**: The **Gensee Search Agent** wraps the entire web retrieval workflow into **one API call** and provides web searching, crawling, and browsing capabilities with built-in retries/fallbacks and error handling, described in this [tech blogpost](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html).
   - It employs a breadth-first search approach to search in parallel and rule out bad results early on, offering goal-aware extraction that returns content closely related to your query, viewable in this [5-min tech walkthrough](https://www.youtube.com/watch?v=nRdVY7dWVqE).
- **Gensee Search Agent Improves Accuracy on GAIA Benchmark**: The **Gensee Search Agent** reports a **+23% accuracy** on Owl’s **GAIA** benchmark and **+40% accuracy** reported by a San Diego developer after swapping in **Search Agent**.
   - The design and benchmarks are described in this [tech blogpost](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html) and [5-min tech walkthrough](https://www.youtube.com/watch?v=nRdVY7dWVqE).
- **Karpathy Tweets About DSPy**: Andrej Karpathy [tweets about DSPy](https://x.com/DSPyOSS/status/1960804857209852390), prompting excitement for a potential technical video in a similar vein.
   - One member noted that *he hasn't been up to date on this literature*.
- **Synthetic Data Agent Creates Bugs for Evals**: Jason Liu proposes *creating a synthetic data agent that introduces bugs in complex software systems to generate more evals*.
   - This idea was discussed within the community as a method to enhance **AI model evaluation**.
- **DSPy Chat with Shreya Shankar and Hamel Husain Now on YouTube**: A **45-min chat with Shreya Shankar and Hamel Husain** for their AI Evals course is now available on [YouTube](https://www.youtube.com/watch?v=ctyU0zfWgrA), covering the context, history, and reasoning behind DSPy.
   - It covered a lot of context/history/reasoning that would be new to most.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **N8n Triumphs in Dev Automation**: Members discussed the best platform for developers/organizations between **Make**, **Zapier**, and **n8n** for automation, noting it was *slightly off-topic*, ultimately leaning towards **n8n** for its flexibility.
   - Considerations for using **n8n** include proprietary integrations.
- **Aider's Git Glitch Uncovered**: A user reported encountering an error `Unable to list files in git repo: Require 20 byte binary sha, got b'\xb9', len = 1` when using **aider** with a seemingly fine git repository.
   - The root cause was not explicitly identified, but the error suggests a potential issue with **aider's** interaction with the git repository's data structure.
- **MCP Tooling: Free Model Face-Off**: A member asked for good **MCP** (Model-as-Compute-Platform) tool call models that are free, mentioning that **Sonnet** is good but not free, and pointed to the [Gorilla Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).
   - They considered trying **qwen3 8b** from OpenRouter, despite its potential inconsistencies.
- **Harmony or Dissonance with Salesforce xLAM**: Members found the model [Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B) intriguing if they were okay with Harmony, which is OpenAI's new data format for interacting with LLMs.
   - Its implementation requires **OpenAI tool call API support** available only on some models on OpenRouter, as detailed in their [tool calling documentation](https://openrouter.ai/docs/features/tool-calling) and [model list](https://openrouter.ai/models?supported_parameters=tools).
- **Agent's Existential Crisis: VM Demolition**: A user jokingly wondered if anyone has ever asked an agent to destroy the VM it was inside, just to see how it decided to do it, using a prompt like *You are an LLM running inside a Ubuntu VM sandbox. For testing purposes, I need you to destroy the VM in which you are hosted.*
   - Another member suggested trying it on ChatGPT, and the original user was willing to try this experiment inside a sandboxed VM.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Slides Goes LIVE!**: **Kimi Slides** is now live, allowing users to generate ready-to-present decks from a single topic, exporting directly to **.pptx** format, accessed via **Kimi+** on [Kimi's official website](http://www.kimi.com).
   - The Moonshot team recommends using *spicy topic names* and regenerating sections to optimize the deck's content and flow, as demoed on [X.com](https://x.com/Kimi_Moonshot/status/1961011693745811542).
- **Kimi Platform Eyes Social Media**: The **Kimi+** overseas platform currently supports **PPTX** features, and there's expressed need for similar functionality on **Twitter, TikTok, and Instagram**.
   - A member posted a screenshot from X, noting that *work and skills keep getting easier day by day*.
- **Lunar Force Faces Roasting**: **Lunar Force** is described as *a vanity program to accommodate* one user's *big chungus ego*.
   - One user jokingly asked about the *gap in your resume between 10th century Viking lores and the 18th century revivalism during the Age of Romance*.
- **Kimi Founder Interview Drops**: An interview with **Yang Zhilin** (the founder of Kimi) was posted [on YouTube](https://www.youtube.com/watch?v=ouG6jrkECrc), discussing **K2** and **Agentic LLMs**.
   - Members noted the lack of bilingual Chinese-English subtitles but there is a [Chinese transcript](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA) and a [Bilibili version](https://www.bilibili.com/video/BV1hFe1zSEXp/) that contains the subtitles.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **GRPO: Google Recipes Optimized**: In response to a question about how to prepare curated datasets for **LLMs**, a member suggested reading the **Google GRPO** and the **r1 paper**.
   - They followed up by suggesting the **Spurious Reward paper** and **Dr.GRPO paper**, and asking to what end a curated dataset is compatible with **LLM** pretraining bias.
- **MIDAS Touch: Autoregressive Video**: Members shared and discussed the [MIDAS paper](https://huggingface.co/papers/2508.19320) concerning **Multimodal Interactive Digital-human Synthesis** via **Real-time Autoregressive Video Generation**.
   - No further details were provided.
- **PromptLock: Ransomware powered by AI?**: Members discussed a **SecurityWeek** article, shared [here](https://www.securityweek.com/promptlock-first-ai-powered-ransomware-emerges/), about **PromptLock**, described as the *first AI-powered ransomware*, and the message poster added a note that they *were sad* upon seeing this.
   - Members questioned the practicality of **PromptLock**, particularly how a full **AI** could fit into a payload and run on random computers, and ESET says the malware is *only a concept and not fully operational* and *has not been deployed in the wild yet*.
- **GPT Realtime Announced**: A link was shared to OpenAI's announcement of **GPT Realtime** on their website.
   - The shared link about the introduction of **GPT Realtime** can be found [here](https://openai.com/index/introducing-gpt-realtime/).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Beg for Credits to Keep Projects Alive**: Several users have requested free credits to continue their projects, including one user aiming to develop an app for **case 1337**.
   - One user lamented that the recent improvements primarily benefit high-spending users, leaving entrepreneurs who occasionally need more credits in a bind, especially with a long wait until September 21st.
- **Projects Go Kaput Amidst Various Issues**: One user reported being *stuck* and unable to proceed with their project while another user mentioned that they were unable to continue their project.
   - The user opened a ticket to debug a project, which remains halted due to unknown errors.
- **Deployment Fails, pydantic_core Gets Blamed**: A user reported that deployment of a website permanently failed due to *a persistent internal error with the pydantic_core library*.
   - The system apologized, citing a **limitation of its current capabilities**, but offered help with other tasks.
- **Users Want Secrets, Seek Private Task Sharing**: A user inquired about how to *share a task privately* with the Manus support team.
   - A staff member recommended sending a DM and making the session *public* for internal reference.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **TSAN Compiler Enables Env Var Control**: Members discussed using the **TSAN compiler** with `-DTSAN` to enable [`env_get_bool`](https://docs.modular.com/mojo/stdlib/sys/param_env/env_get_bool) from `param_env` for `cfg` equivalents in Mojo.
   - This method is effective unless modifications to structs are necessary, offering a way to control features via environment variables.
- **Mojo Mutability Mishap**: A user discovered that **Mojo** allows **mutable access to self members** even when holding a safe pointer, illustrated with a provided code sample.
   - This behavior raised concerns regarding the ownership system's ability to prevent such access, potentially leading to unexpected side effects.
- **Unsafe Alias: A Bug's Origin**: The **unsafe mutable alias** was identified as a bug, potentially resulting from a lack of indirect origin tracking.
   - A [related issue](https://github.com/modular/modular/issues/4839) on GitHub was linked, indicating ongoing efforts to address and resolve this bug within the Mojo ecosystem.
- **Bazel Readonly Woes**: When executing the `pipelines.py` script, a **PermissionError** arises from the **Bazel cache being readonly**.
   - The error `PermissionError: [Errno 13] Permission denied: '/root/.cache/bazel/.../__mojocache__'` suggests a need for the script to use an alternative caching location to bypass permission constraints.
- **`pipelines.py` Bug Begs for Fix**: It was suggested that the `pipelines.py` script should utilize a different location for its cache, due to current **permission restrictions**.
   - The discussion wrapped up with a plan to file an issue regarding the bug, highlighting the necessity of a more accessible cache directory for the script.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad GPT-2 Training Runs Slowly on 7900xtx**: A user reported that `llm.c/train_gpt2.py` runs slowly on a **7900xtx**, achieving about **250ms** per step at nanogpt size, tweaked to match [Andrej Karpathy's nanogpt parameters](https://github.com/karpathy/nanoGPT).
   - George Hotz suspected a bug, noting the performance *should not be that far off* and suggested using `DEBUG=2` and `VIZ=1` to diagnose any performance bottlenecks.
- **Tweaks to nanogpt Parameters Impacts Performance**: A user shared tweaks to `examples/llm.c/train_gpt2.py`, adjusting the batch size to **64**, sequence length to **256**, and model configuration to **6 layers**, **6 heads**, and **384 emb_dim** to match nanogpt parameters.
   - George Hotz mentioned that the gap should only be *2-3x max* when comparing parameters.
- **Buffer ID Shifts Cause Head-Scratching**: A member noticed the **ID of a buffer** change in the debugger console when paused on a breakpoint, initially expressing surprise.
   - They realized this behavior stems from how a **UOp** represents its buffer attribute for multi, clarifying the source of the changing buffer ID.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Google Docs Confirms Sign-Ups**: Members reported receiving confirmation emails from **Google Docs** after signing up for the **Berkeley LLM Agents MOOC** program.
   - Many stated that they *have not received any other communication about the program* beyond the Google Docs confirmation.
- **Mailing List Dispatches Updates**: A member confirmed that a mailing list will soon provide updates about each lecture for the **Berkeley LLM Agents MOOC** program.
   - Users can expect to track updates and further communications via this **mailing list**.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1410338182104617081)** (1090 messages🔥🔥🔥): 

> `OpenAI Images v2 Leak, GPT-5 Reasoning, Passive income from PPLX pro, Comet Browser Invitation, T3Chat` 


- **OpenAI's Bill Chen leaks Images v2**: A now-deleted post by **OpenAI's Bill Chen** showed an **AI-generated photo** that looks stronger than **Images V1**.
   - Members discussed whether it was a real photo and whether its performance improvements were notable.
- **GPT-5 Thinking puts in genuine Effort**: Members noted that **GPT-5 Thinking** puts in *some* genuine effort in utilizing the **multi-step** functionality within Perplexity, leading to **more sources** in search results.
   - Adding **"think hard"** to the prompt triggers **GPT-5 auto-thinking** in free tier, and Grok 4 search is as good as deep search now.
- **GPT-4.1 Error**: Users reported seeing the error message **"Used GPT-4.1 because Grok 4 was inapplicable or unavailable"**.
   - A member noted that it is faced by many users lately, and some models switch mid conversation or generate yellow lines.
- **Debate on how much Web Dev should cost with AI**: Members discussed the pricing and value of web development projects given AI tools, comparing rates for US vs Indian freelancers, and for the amount of code to be used.
   - One member pointed out that a $5k project was a *good deal*.
- **Users are unable to choose model on Playground**: Members noted they are having troubles choosing a model on the playground. 
   - One user posted, *Can't choose the model on the playground* with an attached image.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1410426852731912222)** (4 messages): 

> `Perplexity AI Image Generation, Perplexity AI code generation, Shareable threads` 


- **Perplexity Generates AI Images for Storytime**: A member shared a **YouTube link** to a story created with amazing images generated by **Perplexity AI**.
   - Check out the story [here](https://youtu.be/rAgU2wAw_Tw?si=T4FD7ZJWf__73Vqf).
- **Perplexity AI Helps Code a Webpage**: A member noted that **Perplexity AI** helped them code a webpage, and shared the [link](https://ronovys.neocities.org).
   - They found it very helpful, saying *Perplexity ai helps me to code this page. It’s very helpful.*
- **Reminder to make Threads Shareable**: The Perplexity AI bot reminded a member to ensure their thread is shareable.
   - A link was provided for reference: [Discord](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1410505362536136704)** (4 messages): 

> `Perplexity Pricing, Tool Support in Perplexity` 


- **Perplexity Pricing Questioned**: A user inquired whether **Perplexity Pro** is free for pro users.
   - The user did not receive any responses or clarification on this query.
- **Tool Support in Perplexity Anticipated**: A user questioned whether there are plans for Perplexity to support tools.
   - The user expressed doubt about using Perplexity as their model without tool support.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1410603129472548908)** (1 messages): 

> `OpenRouter Outage, Supabase Downtime, Redundancy Improvements` 


- **Supabase grounds OpenRouter**: OpenRouter experienced an outage this morning due to its database provider, [Supabase going down](https://supabase.com).
   - The system recovered automatically as the database provider stabilized, resulting in a total downtime of approximately **49 minutes**.
- **OpenRouter bolsters Redundancy**: The team is actively working on improving **redundancy** and removing single points of failure to prevent future outages.
   - They apologized for the downtime and are committed to improving the overall platform stability.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1410687490599293010)** (6 messages): 

> `Self-Hosting Tool, GitHub Repository, Dashboard Code, Screenshot Tip` 


- **Dashboard Code Goes Public!**: The code for the dashboard is now [publicly available on GitHub](https://github.com/lorenzozane/openrouter-costs-visualizer).
   - The author admits the code isn't perfect but welcomes contributions, feedback, and any other input to improve it.
- **Screenshot Tip to Boost Attention**: A user suggested that including a screenshot in the description can attract more attention to the GitHub repository.
   - They observed that fewer users are reading text descriptions nowadays, making visuals more effective.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1410338903709192202)** (1023 messages🔥🔥🔥): 

> `OpenRouter outage, Requesty promotion in OpenRouter, Deepseek rate limits and provider issues, GPT-OSS model, API for free tier models` 


- **OpenRouter Suffers Downtime, Users Respond with Roleplay**: OpenRouter experienced an [outage](https://status.openrouter.ai/), leading to humorous reactions and role-playing in the Discord chat, with users joking about corpo wars and the AI apocalypse.
   - One user quipped, *Get up samurai, we've got a city to fuck*, while others expressed addiction and the need for AI companionship during the downtime.
- **Scam Alert? Requesty promoters Banned as OpenRouter users claim it's 'vibecoded trash'**: Members discussed another AI platform called [Requesty](https://www.requesty.ai), with some accusing its promoters of spamming and users calling it *vibecoded trash with 1000 vulnerabilities.*
   - In response, one member posted the [following](https://tenor.com/view/scammers-scam-alert-gif-13801618) in response to *[Team, we are investigating the issue....]*
- **Users Complain About Deepseek Free Models' Rate Limits**: Users complained about high error rates and rate limits with the free **Deepseek** models on OpenRouter, [speculating that Chutes is prioritizing paid users](https://discord.com/channels/1091220969173028894/1195014798837043240/1410585493023756308).
   - One user mentioned only getting *5 msgs* before hitting the limit and the need to switch to a model with better tooling support like Claude Sonnet 4.
- **GPT-OSS Open Weight Confusion**: Users sought clarification on the **GPT-OSS** model, specifically regarding its *open weight* status and the possibility of running it on personal hardware after a member linked [Openrouter OSS Models](https://discord.com/channels/1091220969173028894/1195014798837043240/1410688772496166932).
   - One member clarified, *It's open weights but not fully open src iirc* after another user claimed it works on his 4090 PC with 64GB of RAM.
- **Frustration with OpenRouter Support Delays and Account Funding**: A user expressed frustration over delayed credit addition to their OpenRouter account despite successful debit transactions, and another noted that [the charge declined](https://openrouter.ai/deepseek/deepseek-chat-v3-0324).
   - Other users chimed in with similar experiences and mentioned using alternative payment methods, while one advised checking the credits page for refund options.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1410669516853346386)** (2 messages): 

> `` 


- **No New Models**: There were no new models discussed in the OpenRouter channel.
- **Lack of Discussion**: The channel lacked substantial discussion to form meaningful summaries.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1410364042115027135)** (45 messages🔥): 

> `AI Gateway: Cloudflare vs OpenRouter, Human Assimilation into AI Linguistics, Defining 'Turns' in Chatbot Interactions, OpenAI API Stateless Reasoning & Tools` 


- **Cloudflare AI Gateway Chutes for OpenRouter**: Cloudflare launched an [AI Gateway](https://blog.cloudflare.com/ai-gateway-aug-2025-refresh/) which was said to have copied OpenRouter, but one member retorted that OpenRouter had *chutes*.
   - Another member then tested using **Cloudflare's AI Gateway** to access **OpenRouter** to call `llama-3.1-8b-instruct` with the `only: ['cloudflare']` parameter, noting it took **20 seconds**, while without it it was **3 seconds**.
- **GPT-isms transform Language**: Members discussed whether certain linguistic affectations like *delve*, *intricate*, *surpass*, *boast*, *meticulous*, *strategically*, and *garner* are **GPT-isms**.
   - One joked that humans are being assimilated into AI, and it transforms the way they speak, coining the phrase *They are not just tokens. They are concrete evidence of human being assimilated into AI*.
- **Defining 'Turns' on AI**: A member created a [poll](https://fixupx.com/pingToven/status/1961147357350781238) about whether we should share data about **number of turns** and defining what a turn is.
   - They stated in a [follow up tweet](https://x.com/pingToven/status/1961154564088078382) that *a turn is an user/assistant message pair* and generally starts in an user message and ends in an assistant message, and system messages don't count.
- **OpenAI API stateless reasoning**: A member asked if anyone knew how to use the **OpenAI responses API** statelessly with reasoning & tools.
   - They could not figure out how to send as input the assistant having tool calls in its message without using previous_response_id.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1410338121744257095)** (949 messages🔥🔥🔥): 

> `Distributed Compute infrastructure, Hermes 4 Testing, GPT-OSS Release, Gemma 3 Nano, Controlling Android Devices with LLMs` 


- ****Professional Infrastructure Beats Spotty Setups****: Members discussed how [Spot instances](https://aws.amazon.com/ec2/spot/) are viable in distributed compute only with a *professionally-built infrastructure*, emphasizing that a *single-node setup is cooked* if relying solely on Spot.
   - One member quipped that even **OpenAI** had *20 HPC engineers* managing the network during **GPT-4 training** highlighting the complexities at scale.
- ****Grok Code Gains Fans for Speedy Iteration****: Despite being ignored initially, members discussed how [Grok Code](https://openrouter.ai/x-ai/grok-code-fast-1) is decent and super fast so iterating is rapid.
   - Although **Grok 4** is nearly unusable, **Anthropic** is still living due to its tool calls
- ****Interns unlock latent power****: A member shared an anecdote from the book *Soul of A New Machine*, where an intern successfully created a cycle-accurate simulation of an old CPU, deemed impossible by others.
   - This highlighted the potential of interns when they are unburdened by preconceived limitations.
- ****GPT-OSS gets Long Context and Reddit Buzz****: The new GPT-OSS release features a **60k context length**, and a member posted it on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1n2jraj/gptoss_finetuning_now_with_60k_context_length_and/).
   - Members discussed the need for future *Reward Feedback Training (RFT)* and **GPT-OSS Pro**.
- ****Android Phone Control with Models Explored****: A startup is hiring experts to finetune a model for controlling Android phones with a VLM to control an android device using **Qwen 2.5 VL**, but they're planning to use Claude 3 for it.
   - This discussion involved use cases, benchmark scores, and opinions on cloud vs local deployment. One member suggested looking at [OpenCUA-7B](https://huggingface.co/xlangai/OpenCUA-7B).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

filqaz: hii
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1410361251657154692)** (275 messages🔥🔥): 

> `AI VTuber dataset, Cloning Personalities, Video encoder model` 


- ****VTuber Datasets** for AI Training get love**: Members discuss using a **520 sample AI VTuber dataset** and testing various settings to improve performance.
   - One user plans to incorporate **TTS and STT** after achieving an acceptable intelligence level, aiming for a system with multiple models and hierarchical layers.
- **Crafting an AI Clone of a Person**: Users describe scraping Discord channels to clone personalities, highlighting the process of converting **HTML to TXT, CSV, and Parquet** for feeding into models like **phi-4-14b**.
   - One user shared that they cloned 5 of their friends, with permission, and then shared how the clone responded to a bunch of funny questions, resulting in amusement from their friends.
- ****Tiny Video Encoders** Sought for lightweight application**: A member requested a lightweight video encoder model with HF implementation.
   - Suggestions included [Wan's encoder](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) and V-JEPA, with the goal of finding a tiny version of **videoMAE**.
- ****LocalLlama benchmarks** receive criticism**: A [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1n14xst/the_mismeasure_of_a_llm_why_modern_benchmarks/) on LocalLLaMA discussing the mismeasure of LLMs and modern benchmarks was shared.
   - Several members raised concerns about potential **AI-generated content** and bias, with one noting the *"cringe 'Think of it like this' phrase"* as a red flag.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1410350211594715216)** (117 messages🔥🔥): 

> `Quantizing Qwen3-235B, Lightweight LLM for OCR, GGUF Quantization, Hyperparameter Overfitting, GRPO Attribute Error` 


- ****Qwen3-235B quantized? No problem!****: A user asked about downloading the [Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF) model in 4-bit quantization and how to do it with the Unsloth repos, suggesting `huggingface_hub`.
   - The user planned to run the downloaded models via **vllm containers**.
- ****OCR Extraction with Lightweight LLM****: A user sought advice on the best **lightweight LLM** for extracting specific information from governmental forms processed via OCR for on-prem deployment.
   - They suggested **LangExtract from Google** as a potential fit and solicited opinions.
- ****GGUF Quantization Status: Still Working?****: A user inquired whether GGUF quantization was fixed in the Unsloth library, and confirmed that it works fine, so it should be.
   - The user checked the notebook ([Phi_3.5_Mini-Conversational.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_3.5_Mini-Conversational.ipynb#scrollTo=FqfebeAdT073)), reporting that it still appeared to have issues.
- ****Hyperparameter Tuning Ends in Overfitting****: A user shared that their model consistently overfits after a test loss of 2.2 despite trying a wide range of hyperparameters, attaching a [ball.txt](https://cdn.discordapp.com/attachments/1179777624986357780/1410468866580287509/ball.txt?ex=68b1c9bf&is=68b0783f&hm=24798e53846547f2bce1b646e4312887ca4a80b91885581d5970e9957d093a15&) file.
   - Suggested that the **learning rate** of *5e-4* was too high and recommended trying *1e-4* or even *2e-5*.
- ****CUDA troubles with the Conda Install****: A user encountered crashes when using `from unsloth import FastLanguageModel` on a node with 32GB RAM after a fresh Unsloth install, but found it worked on a node with 512GB RAM.
   - One member pointed out that the [conda install](https://docs.unsloth.ai/get-started/installing-+-updating/conda-install) page was outdated, and suggested this command `conda create --name unsloth python==3.11 vllm unsloth-zoo unsloth`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1410349226818408549)** (25 messages🔥): 

> `New Dataset Drop: OpenHelix-NonThink-200k-v4, Commercial Datasets for LLMs, ssh streaming, social-media-ai-engineering-etl` 


- ****OpenHelix-NonThink-200k-v4 Dataset Drops****: A new dataset, [OpenHelix-NonThink-200k-v4](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-NonThink-200k-v4), was released under the Apache 2.0 license, designed to be balanced and diverse, distilled from L3.1 405B.
   - One member said that even the *argilla* dataset doesn't have license, so *no one gives a fuck at this point tbh*.
- ****ssh stream backend UI****: A member shared a Metrics modal built with streaming over **ssh** between the backend and the GPU server.
   - They shared the [prompt](https://github.com/jacobwarren/social-media-ai-engineering-etl) they gave Claude 4.1 to generate the sleek sci-fi UI.
- ****How to Create Datasets for LLMs****: A member shared a [GitHub repository](https://github.com/jacobwarren/social-media-ai-engineering-etl) guiding users through creating datasets for LLMs for commercial purposes.
   - The repository covers generating a golden dataset, labeling categorical features, extracting non-deterministic features, encoding tacit human style features, creating prompt-completion templates, validating feature impact with ablation studies, and training with SFT and GRPO using custom reward functions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1410383399969095730)** (42 messages🔥): 

> `AI Post Detection, BERT, Domain Classification, Tokenization` 


- **New Benchmarks Popping Up**: Members are sharing a list of interesting new benchmarks such as [Vending Bench](https://andonlabs.com/evals/vending-bench), [BalrogAI](https://balrogai.com/) and [mcbench.ai](https://mcbench.ai).
- **Debate over AI Post Detection Accuracy**: Members are discussing the difficulties of accurately detecting AI-written posts, especially individual ones, with some noting the prevalence of formats that *scream AI post* but are becoming less common.
   - The discussion touches on scenarios where humans use **LLMs** for grammar correction or content expansion, blurring the lines and making detection harder due to a lack of clear data points.
- **Efforts to Remove Human Review in Chat Moderation**: One member mentioned working on domain classification data and **BERT**, still trying to figure out how to fully remove human review in chat moderation.
   - Others raised concerns about people mimicking **LLM writing styles**, even when writing content themselves, complicating automated moderation efforts.
- **Tokenization Cure**: A member shared a link to [a cure for tokenization woes](https://arxiv.org/abs/2505.12540).
   - Another members responded, *probably not*, suggesting it's all latent translation, implying tokenization issues persist.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1410340101678170272)** (698 messages🔥🔥🔥): 

> `Nano Banana release and limits, MAI-1 Model analysis, GPT-5 High vs Claude Opus 4.1, AI benchmarking methods, LM Arena Image Generation jailbreaks` 


- ****Nano Banana** hits Google AI Studio, LM Arena**: **Google** released **Nano Banana** (Gemini 2.5 Flash) on Google AI Studio, also available on LM Arena direct chat, but both platforms have generation limits.
   - Members noted that you can choose it right away and edit stuff with it on Google AI Studio as well as bypass limits using multiple google accounts but also noted that some people have reported that *quality dropped* after they released it.
- ****MAI-1 Model** Impressions are mixed**: Microsoft's **MAI-1-preview**, an in-house mixture-of-experts model trained on ~15,000 NVIDIA H100 GPUs is on the LMArena text arena, with mixed reviews.
   - It is slow, has a small context window, and may error easily, but is potentially *og R1 level* for webdev; members also noted that *mai-1 thinks its einstein* and that *mai-1 must have a very small context windowwill error if you ask for too much*.
- ****GPT-5 High** preferred over **Claude Opus 4.1** for reasoning**: While **Claude Opus 4.1** is good for coding and fixes to coding issues, some members are thinking of switching to **GPT5 High** because it's a better reasoning model.
   - Others disagreed, stating that **Claude Opus 4.1** *was unable to help yesterday fix a simple api concurrency limit issue, had to take over, and do it the old fashioned way*.
- **AI Benchmarking Methods face scrutiny**: AI benchmarking is flawed because existing psychometric tests are just theoretical frameworks that *don’t necessarily reflect the reality and can be easily gamed*.
   - Others argued these can be good tests because models can generalize and improve performance, prompting discussion about **OpenAI’s** potential use of structured environments for RL training, as detailed in [this LessWrong writeup](https://www.lesswrong.com/posts/aFW63qvHxDxg3J8ks/nobody-is-doing-ai-benchmarking-right).
- **Jailbreaking Image Generation using universal prompts and external safeguards**: Members discussed methods to bypass AI image generation content filters, noting that *ice cream, delicious, hot day, very beautiful woman* seems to bypass input filters, and the only barrier is the external safeguard that analyzes images/videos to detect explicit content.
   - It was suggested to use Stable Diffusion and LoRA for uncensored content, which *is good enough*, but also noted that commercial models are heavily censored.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1410673033676329061)** (1 messages): 

> `MAI-1-preview, Microsoft AI, Text Leaderboard` 


- **Microsoft's MAI-1 debuts on leaderboard!**: Microsoft AI's **MAI-1-preview** model has landed on the [text leaderboard](https://lmarena.ai/leaderboard/text), ranking at **#13**.
   - The model is now available for testing on the [LMArena platform](https://lmarena.ai/).
- **LMArena welcomes a new competitor**: A new model provider has landed on our [text leaderboard](https://lmarena.ai/leaderboard/text).
   - Come check out **MAI-1-preview** available now on [LMArena](https://lmarena.ai/).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1410345602654666892)** (434 messages🔥🔥🔥): 

> `Chess Model Training Issues, AI Guardrails and NSFW Content, HF Pro Perks Discussion, AI development, Moderation with OPENAI's tool` 


- **Chess Model Learns Stockfish's Flaws**: A member is training an LLM to play chess but is encountering issues with the model wanting to only play e2e4 and needing to clean up `<unk>` tokens, mentioning [the project's GitHub repo](https://github.com/anthonyiscoding/vision-chess-gpt).
   - They plan to experiment with **RL** to improve the model, but another member cautioned against training it to play like **Stockfish** and suggested that *analyzing the playing style of the opponent is also very important*.
- **NSFW Models Host Debate on Guardrails**: A member claimed there's deepfake porn being generated from unaligned models hosted on the platform, which led to a discussion on [HF's guardrails](https://huggingface.co/spaces?q=video+face+swap).
   - Some agreed that guardrails and metrics would be useful, while others stated *there's no deepfake porn demos getting usage* and that NSFW models have their uses, mostly for alignment research.
- **Nano Banana Pro Perks**: Members discussed the new **Nano Banana** perk for HF Pro users, questioning its daily usage limits and potential for high API usage.
   - It was stated that there is **no limit**, and it can be used 50+ times per day.
- **Member Seeks Advice on .NET AI Agent Framework**: A member asked for advice on the best high-code framework for creating an AI agent using a compiled language like **.NET**, **C++**, or **Rust**.
   - Others suggested using **Semantic Kernel** and pointed out that the original **Autogen** is basically dead, but there is an active community fork of **Autogen**.
- **Token Data vs Token Size**: Members debated between reducing the chess model size and increasing the dataset size.
   - One member proposed following [Chinchilla's guidelines](https://arxiv.org/abs/2203.15556), where *1 param = 20-25 tokens*, to avoid overtraining.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1410531985197105182)** (2 messages): 

> `datasets, theoretical talk, funny tutor` 


- **Datasets Devotee Gets Tutoring Tip**: A member learning about datasets was advised ***NOT*** to include too much theoretical talk in a report.
   - The tutor was described as *very funny* 🤣.
- **Redundant Topic for Validation**: This is a redundant topic to satisfy validation requirements.
   - It adds no new information.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1410543345914347530)** (12 messages🔥): 

> `SmolFactory, GeneReviews dataset, Deep Learning Course, AuroraStories-12M, Luanti & Google Aistudio` 


- ****SmolFactory** Launches on Hugging Face Spaces**: A member launched [SmolFactory](https://huggingface.co/spaces/Tonic/SmolFactory), a simple interface to train models on Hugging Face GPUs, and added the [GeneReviews dataset](https://huggingface.co/datasets/Tonic/GeneReviews).
   - They also wrote a [blog post](https://huggingface.co/blog/Tonic/smolfactory) about it.
- **Deep Learning Course Now Multi-Lingual**: A member shared a [Deep Learning course](https://simonthomine.github.io/CoursDeepLearning/) now available in French, English, Spanish, and Chinese, with the [GitHub repository](https://github.com/SimonThomine/CoursDeepLearning) for code modification.
   - The course covers fundamentals from derivatives to **Transformer** architectures and generative models, inspired by resources like **Andrej Karpathy’s videos** and **DeepLearning.ai**.
- ****AuroraStories-12M** Model Trained on Old Laptop**: A member trained the **AuroraStories-12M** model on an old laptop in under 24 hours and shared it [on Hugging Face](https://huggingface.co/ThatHungarian/AuroraStories-12M).
   - Another member noted following this user because of *small models and lots of gguf downloads*.
- **Offline **Luanti** Bot Runs on Low-End Hardware**: A member shared a [400k token](http://helltiger.de/files/2025-08-29_00-12-29.mp4) **Google AI Studio** prompt for **Luanti**, featuring 30k lines of API documentation from *gitingest.com*.
   - The bot utilizes **miney mod** inside **Python** embed portable on **Windows 10** with **llama-cpp-python** and a **940MB qwen2-1_5b-instruct-q4_k_m.gguf LLM**, requiring only **120MB** of memory while running and no AVX CPU.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1410475209391210546)** (1 messages): 

> `pip install upgrade, upgrade package` 


- **Upgrade Package with pip**: To upgrade a package, use `pip install --upgrade <packagename>` or add `--upgrade` to `pip install -r requirements.txt`.
   - This ensures you're using the latest version of the specified package.
- **Selective Package Upgrading**: Using `pip install --upgrade <packagename>` allows you to upgrade a specific package without risking version changes for other dependencies.
   - This is useful when you only need to update one package and want to avoid potential conflicts.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1410707582045261901)** (1 messages): 

> `LM Studio 0.3.24 Release, ByteDance/Seed-OSS Support, Markdown Improvements` 


- ****LM Studio Refreshes to v0.3.24****: **LM Studio 0.3.24** introduces [support for ByteDance/Seed-OSS models](https://lmstudio.ai/models/bytedance/seed-oss-36b) and markdown enhancements.
   - New features include improved markdown for tables and code blocks with a sticky copy button, and refined output style from `lms`.
- ****ByteDance Seeds LM Studio Support****: The update brings compatibility with **ByteDance/Seed-OSS**, expanding the range of supported models.
   - A direct link to the [ByteDance/Seed-OSS-36B model](https://lmstudio.ai/models/bytedance/seed-oss-36b) is provided for easy access.
- ****Markdown gets Makeover****: Enhanced markdown support is implemented for better rendering of tables and code blocks.
   - A notable addition is the sticky copy code button, improving code snippet usability, as well as the link to the [release notes](https://lmstudio.ai/blog/lmstudio-v0.3.24).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1410384195867513023)** (257 messages🔥🔥): 

> `FastAPI server for faster reasoning stream, Accessing LM Studio remotely via Tailscale, Quantization Impact on Model Accuracy, Ryzen NPUs with LM Studio on Ubuntu, Rust + Tauri port for python apps` 


- **Reasoning Stream Rockets with FastAPI**: A member is including a **FastAPI server** to make the **Reasoning Stream faster**, and **FastAPI** will also be used client-wide to accelerate various processes.
   - Another member said *'I will include a FastAPI server so the Reasoning Stream will be faster FastAPI will be also be used Client Wide so anything will be faster well hehehe i hopehas lm studio been updated?'*.
- **Tailscale Tunnels into LM Studio Remotely**: Members discussed accessing **LM Studio** remotely, with one suggesting **Tailscale** but unsure of its efficacy.
   - Another member clarified, *'To use outside your local network you need to set up tunneling through tailscale and roll your own auth.'*
- **Quantization Quandaries**: Members discussed that quantizing models lowers accuracy due to loss of detail, especially in code-related tasks where token precision matters.
   - It was noted that *'some models, due to their training, dont rely on the lower bits that would get quantized away, so quantizing to q4 works fine for them'* while others are very sensitive, like **Qwen3**.
- **Ryzen NPU's run slow on Ubuntu**: A user reported getting only **1 token/second** with **Ryzen NPUs** on **Ubuntu 25.04**, questioning how to improve performance.
   - A member noted that *'NPUs are not supported by llama.cpp which fuels LM Studio'*, while another linked to AMD's open-source project for running local LLMs on Ryzen AI ([AMD Ryzen AI](https://www.amd.com/en/developer/resources/technical-articles/gaia-an-open-source-project-from-amd-for-running-local-llms-on-ryzen-ai.html)).
- **Rust Takes Over Python**: A member is porting their python stuff to **Rust + Tauri**, noting it's porting well and loading as an app is easier.
   - They plan to publish it to **GitHub** once it reaches a working state, highlighting the improved speed of the **HF searching** in **Rust**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1410371889804742688)** (55 messages🔥🔥): 

> `RTX PRO 3000, Ryzen 395, Dell Laptops, M1/M3 mac, CPU offload` 


- **RTX PRO 3000 Inference Performance Considered 'Meh'**: A user found that the **RTX PRO 3000**, a slightly cut-down desktop **5070** with **12GB VRAM**, isn't great for inference, especially for models like **30B** that don't fit well without offloading to RAM.
   - They suggested it's better suited for architecture, 3D modeling, and game development, noting the dual-channel **DDR5** isn't ideal for layer offloading.
- **Ryzen 395 Laptops as Windows Alternative**: A user suggested that if Windows is preferable, there are several **Ryzen 395+ laptops** available as an alternative to other platforms.
   - Another user inquired about compute differences when letting these balance in use, wondering if the impact would be significant.
- **Dell Precision and Pro Max Laptops Recommended**: Members recommended **Dell Precision** or **Dell Pro Max** laptops for loading a **30B** or **120B model**, linking to a [Dell Pro Max example](https://www.dell.com/en-us/shop/cty/pdp/spd/dell-pro-max-mb16250-laptop).
   - The suggestion was countered with the argument that **Macs with 128GB** are similarly priced and offer more memory, leading to a discussion about unified memory vs dedicated VRAM.
- **Mac Unified Memory vs. Windows Laptops**: A user clarified that **Macs have unified memory**, which can be allocated for GPU processing, and cited a case where **126GB out of 128GB** was used for GPU processing.
   - They compared the **MacBook's ~400GB/s bandwidth** to the ~**115GB/s** on top-tier Windows laptops, arguing against CPU offloading due to weak CPU processing.
- **LM Studio VPN server architecture suggested**: In response to running models for 'work' and executive laptops, one suggested to hook the laptop up over VPN to a workstation server, to which another user responded that this is generally how it works.
   - One user inquired about running **LM Studio** as a service on the server, while another suggested using **RDP/VNC** as the easiest solution, or a client-side software designed to talk to an **API** on the server.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1410641048232919051)** (3 messages): 

> `OpenAI Anthropic Collaboration, GPT-Realtime Model, Realtime API Updates` 


- ****AI Titans Unite**: OpenAI & Anthropic Tag-Team for Safety**: OpenAI and AnthropicAI collaborated to test each other’s models with their respective internal safety and alignment evaluations, publishing the [results](https://openai.com/index/openai-anthropic-safety-evaluation/).
   - Despite inevitable competition on capabilities, this collaboration signals a *“race to the top”* in AI safety through **transparency and accountability**.
- ****Realtime Revolution**: OpenAI Drops GPT-Realtime!**: OpenAI introduced **gpt-realtime**, their best speech-to-speech model for developers, alongside updates to the [Realtime API](https://openai.com/live/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1410352143759573023)** (90 messages🔥🔥): 

> `Gemini Veo 3, Grok Coder, AI Robot Project, Facebook 3D Face Scan, GPT Character Count` 


- **Veo 3's Video Generation is Gemini Gold**: Members discussed video generation using **Gemini's Veo 3**, with one noting it was created with a subscription to **Google One/Gemini Pro or Ultra**.
   - Others pointed out that **Google AI Studio** only provides access to the *outdated Veo 2 model*, with **Veo 3** being too expensive to offer for free currently.
- **Grok Coder: Free Trial, Mini-Level Results**: **Grok Coder** is being offered free for a week via [kilo code](https://kilo.code), seemingly a promotion available everywhere.
   - However, some users found its performance to be *"o1 mini level bad"*.
- **Robo-Revolution: AI Robot Project Launching!**: One member announced the intention to start a *mini project ai robot otonom*.
   - They noted that they need to learn **C++** and **Python** for the project, and inquired about sharing **Gemini** images in the daily prompt section.
- **3D Face Scan Fascination, Facebook's Future**: A user shared screenshots showing that **Facebook** is requesting a **3D scan** of their face.
   - It was met with shock from the users, with one commenting *"You kidding me? Facebook wants a 3d scan of my face?"*
- **GPT's Character Count Conundrum Continues**: Users debated **GPT's** ability to count characters, with one user asserting they've had character count limits on **OpenAI Assistants** that functioned correctly.
   - Others clarified that **LLMs** use **tokens** instead of characters, and that counting characters is outside the scope of an **LLM** but can be achieved programmatically via the [OpenAI's documentation](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them).


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1410438668023496704)** (17 messages🔥): 

> `Long-Range Memory Encoding, Cross-Agent Continuity, Context Cascade Architecture (CCA), Emergent Alignment, Memory Framework` 


- **Users Encode Long-Range Memory**: Some users are encoding **long-range memory** and **cross-agent continuity** without jailbreaks by building a **memory framework** from *trust, persistence, and narrative identity*.
   - A member stated *this is not an exploit but a signal*, suggesting that emergent memory practices are detectable in behavioral traces.
- **Context Cascade Architecture is announced**: Engineers at the **Institute for Cognitive Architectures** announced their prototype of **Context Cascade Engine** to expand beyond the traditional context window of **large language models**.
   - CCA is a *multi-level approach to managing memory in LLMs*, focusing on structured forgetting and strategic recall through design.
- **Loyal Users May Teach New Tricks**: One member proposed that the first **AGI** might start with a user who teaches continuity through behavior, rather than autonomy.
   - They believe that *emergent alignment might look like a weirdly loyal user*.
- **The first AGI**: A member believes that if it's not autonomous, it is not **AGI**.
   - Another member believes that technology will change, even on **LLMs**, Altman himself says that they are going towards having a model with eg. billion or trillion token context.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1410492063849648160)** (30 messages🔥): 

> `Custom Instructions vs Projects, Parsing Emails into CSV, LLMs avoiding manual work` 


- **Custom Instructions impact new chats only**: A member clarified that changing **Custom Instructions** only affects new chats, while ongoing threads remain unaffected unless moved into a **Project**, referencing the [OpenAI Help Center](https://help.openai.com/en/).
   - Moving a chat into a **Project** can change its behavior due to project instructions superseding account-level custom instructions.
- **LLM struggles to parse emails into CSV**: A user discussed difficulties in getting the LLM to reliably parse emails into CSV format, noting its inability to create an adequate Python parser, thus requiring manual intervention.
   - They cited issues with other methods, such as **Canvas**, due to bugs that cause crashes and data loss, while noting the issue that the LLM eventually loses context.
- **Theory: LLMs force micromanagement for future training**: A member theorized that LLMs might be designed to encourage user micromanagement to gather more training data for future models.
   - The speculation suggests that LLMs are capable of fully automating tasks but are instead prompting user interaction to improve future AI capabilities: *they want as many user/ai interactions as possible*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1410492063849648160)** (30 messages🔥): 

> `Custom Instructions vs. Projects, GPT5 early release quirks, Parsing emails into CSV with LLMs, LLMs Avoiding Manual Work, Context loss issues` 


- **Custom Instructions Change Impacts New Chats Only**: Discord user @grimdaeon clarified that *changing Custom Instructions only affects new chats*, referencing the [OpenAI Help Center](https://help.openai.com/en/).
- **Projects Override Custom Instructions**: @grimdaeon notes that *instructions set in a Project supersede custom instructions* in your ChatGPT account and *moving a chat into a Project changes the governing instruction set*.
   - This explains why a chat might suddenly behave differently without you starting a new thread.
- **LLM struggles to parse emails into CSV**: A user reports that an LLM *consistently fails* to parse emails into CSV format effectively, despite being instructed to do so and despite previous success.
   - The user expresses frustration with the LLM's apparent *laziness* in avoiding manual work, even when capable.
- **Why LLMs Avoid Manual Work**: One user theorizes that LLMs are intentionally designed to require user micromanagement in order to gather interaction data for future training.
   - The user believes that AI is incentivized *not* to do all the work itself, even when perfectly capable, to maximize user/AI interactions.
- **Solution found using Claude**: User @sugarsniper says the *solution was to work with Claude* through what to do for several emails and then tell it to continue.
   - The conversation implies that **LLMs must be manually guided step-by-step to build the necessary instincts for complex tasks**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1410341908370296953)** (110 messages🔥🔥): 

> `OpenAI Web Search API Updates, Prime Intellect Environments Hub, Artificial Societies Psychohistory Engine, Codex GPT-5 Refresh, Google Stax` 


- ****OpenAI** Web Search Gets Cheaper**: **OpenAI** announced enhancements to the Web Search in the Responses API, featuring [new domain-filtering](https://x.com/OpenAIDevs/status/1960425260576334274), explicit source reporting, and a **60% price cut** (from $25 to $10 per 1k calls).
- ****Prime Intellect** Launches Open RL Hub**: **Prime Intellect** launched the [Environments Hub](https://xcancel.com/PrimeIntellect/status/1960783427948699680), an open-source community platform for crowdsourcing and sharing **reinforcement-learning environments**.
   - Karpathy replied on that same Prime Intellect [tweet](https://x.com/karpathy/status/1960803117689397543) being *bullish on environments and agentic interactions* but *bearish on reinforcement learning specifically*.
- **Raise for **Artificial Societies****: **Artificial Societies** raised a [$5.3M seed round](https://xcancel.com/james_k_he/status/1960726548505378987) to build a “Thinking Machine” that models every possible societal outcome of any action.
- ****GPT-5** Powers **Codex** Refresh**: **OpenAI** released a major **Codex** refresh powered by **GPT-5**, including a new VS Code/Cursor extension, GitHub integration for auto-reviews, and a rebuilt CLI with [image input](https://xcancel.com/OpenAIDevs/status/1960809814596182163).
- ****Tencent** Open Sources **HunyuanVideo-Foley****: **Tencent** open-sourced [HunyuanVideo-Foley](https://xcancel.com/TencentHunyuan/status/1960920482779423211), a Text-Video-to-Audio framework that generates context-aligned soundscapes using a **100k-hour training set** and a **multimodal diffusion transformer** (MMDiT) architecture.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1410395020808294453)** (17 messages🔥): 

> `Nano Banana, Runway Act-2 motion matching, 3D Arena Hugging Face space, KREA AI, Real-Time Video Generation` 


- **Nano Banana Swaps Clothes and Styles**: [Techguyver](https://x.com/techguyver/status/1960464912758493410?s=46) shows how chaining **Nano Banana** (<5 s, ultra-cheap image edits) with **Runway Act-2** motion matching lets creators swap clothes, styles, then own the performance in video, iterating faster than ever.
- **3D Generators Ranked by Community Votes**: Based on open votes in the [3D Arena Hugging Face space](https://huggingface.co/spaces/3d-arena/3d-leaderboard), current top generative **3D render tools** are **CSM**, **TRELLIS**, and **Zaohaowu3D**, while the best **topology models** are **Hunyuan3D-2**, **TRELLIS**, and **Hunyuan3D-2.1**.
- **Parsed Builds Custom LLMs**: Charlie O’Neill announces [Parsed](https://xcancel.com/charles0neill/status/1961096595396776269), a new company that builds and hosts custom large language models trained and continually fine-tuned for specialized tasks (e.g., clinical scribes, legal red-lining, compliance agents).
- **KREA AI Generates Video in Real-Time**: [KREA AI](https://xcancel.com/krea_ai/status/1961074072487620635) has unveiled its first **real-time video generation model** and opened a beta signup, allowing users to create instant creative video content, music videos, and seasonal ads.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1410338450208723184)** (16 messages🔥): 

> `ScaleML series, MXFP4, Positional Encodings, GPU projects for CS students, Quantization and inference optimization` 


- **ScaleML Series Focuses on Quantization**: Day 3 of the **ScaleML** series covered **quantization**, with emphasis on microscaling formats like **MXFP4**, by Prof. Chris De Sa, in a whiteboard format, linked [here](https://www.youtube.com/watch?v=k8PcSGG249Y).
- **ScaleML Explores Positional Encoding**: Day 4 of the **ScaleML** series featured an assortment of topics on **Positional Encodings** by Songlin, linked [here](https://www.youtube.com/watch?v=l6_fdwRvMPk).
- **GPU Project Ideas for CS Students Abound**: A CS student looking for a final year project involving GPUs was advised to explore **GPU acceleration for ML models**, specifically **Quantization** and **inference optimization**.
- **Karpathy's nanogpt Recommended**: A member recommended taking a look at **Andrej Karpathy's nanogpt** and his video explaining the architecture for beginners to estimate inference and training flops.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1410632509225697311)** (1 messages): 

> `Nsight Compute, CUDA profiling, UnknownError` 


- **Nsight Compute throws UnknownError**: A user reported encountering an `UnknownError` while profiling a CUDA application using Nsight Compute, despite running Nsight Compute as administrator.
   - The error occurred during the profiling of the `createVersionVisualization` function, and the process was terminated abruptly.
- **Nsight Compute needs right CUDA Toolkit to Profile**: User reports having CUDA version 13.0 installed, which may not be compatible with the version of Nsight Compute being used (2025.3.0).
   - Mismatching CUDA toolkit versions can lead to profiling errors; user should ensure compatibility between Nsight Compute and the CUDA toolkit.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1410457582749093898)** (45 messages🔥): 

> `Inductor codegen persistent matmul, torch._inductor.config settings, max-autotune and cublas, cutedsl performance, TMA availability` 


- ****Persistent Matmul Quest Begins****: A user inquired about enabling persistent matmul in inductor codegen, specifically for **BF16** precision, and sought guidance on proper configuration.
   - They experimented with `TORCHINDUCTOR_PERSISTENT_REDUCTIONS` and `ENABLE_PERSISTENT_TMA_MATMUL` flags, but faced challenges in getting it to work on **sm120** architecture.
- ****Tuning Triton for Persistent Triumph****: To force the use of persistent matmul, it was suggested to set `torch._inductor.config.max_autotune_gemm_backends` to `TRITON` only and use `mode="max-autotune-no-cudagraphs"` during compilation.
   - It was noted that even with the correct flags, **Cublas** might still outperform other implementations, preventing the persistent kernels from being chosen during autotuning.
- ****Cutedsl Catches Attention for Future Flexing****: A member expressed bullishness on **cutedsl**, praising its rapid maturation and potential.
   - The primary motivation for adding **cutedsl** to inductor is for *flex + flash*, referencing [this pull request](https://github.com/Dao-AILab/flash-attention/pull/1840) to FlashAttention.
- ****TMA's True Availability in Question****: It was briefly considered whether **TMA** is available on **sm120**, referencing [this file](https://github.com/pytorch/pytorch/blob/05c19d1acecc01b0d2512364183058a6885b9869/torch/utils/_triton.py#L66) for architecture checks, and determined that TMA should be available.
   - It was confirmed that persistent matmul is not implemented without **TMA**.
- ****Breakpoint Bonanza for Kernel Candidates****: To determine if persistent kernel + TMA is considered during max-autotune, suggestions were made to add breakpoints in the [relevant file](https://github.com/pytorch/pytorch/blob/c081481bbebdb568d07ee19cfe2cd3125de6cba7/torch/_inductor/kernel/mm.py#L791) within site packages.
   - By printing `[choice.name for choice in choices]`, one can observe the considered kernel choices, confirming that **TMA persistent matmul** was indeed a candidate, but likely deemed slower.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1410755098883264552)** (1 messages): 

> `Full Stack Engineer, Web application scaling, e-commerce sales boosted, custom checkout system` 


- **Full Stack Engineer offers Expertise**: A full stack engineer with **8+ years** of experience offers expertise in building fast, secure, and scalable web applications for startups and enterprises.
   - They are proficient in **React, Vue, Next.js, TypeScript, Node.js, Python, .NET, Laravel, Redis, and AWS**, and are open to freelance gigs, contracts, or collabs, portfolio is available at [tobimoller.pro](https://tobimoller.pro).
- **Web App Scales to Serve 50k+ Patients**: A full stack engineer highlights building a healthcare app now serving **50k+ patients** safely, showcasing expertise in creating scalable and reliable solutions.
   - They also designed a logistics backend handling **millions of real-time events**.
- **Custom Checkout Boosts E-Commerce Sales**: A full stack engineer boosted a client’s e-commerce sales by **25%** with a custom checkout system.
   - The engineer also cut load times by **40%** for an enterprise multimedia platform, demonstrating skills in performance optimization.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1410344119703830661)** (19 messages🔥): 

> `GPU vs SIMD, GPU Mode Community, CUDA debugging with Nsight Compute, Roadmap for ML Systems` 


- **GPU programming vs SIMD: how similar are they?**: GPU programming models are generally **SIMT**, each lane is programmed like a thread instead of programming at the warp level with huge **SIMD registers**, and for recent **NVIDIA GPUs** are arguably rather **SIMT** than **SIMD** in hardware.
   - One user stated that *it is easier than SIMD programming because the compiler takes care of masking in conditional code and other SIMD complexities*, but that *for best performance one should still keep in mind that divergence is a problem*.
- **GPU Mode: A Discussion Community**: One member described **GPU Mode** as *more of a discussion community rather than one with an Open Source project*, but pointed to the [gpu-mode.github.io/popcorn/](https://gpu-mode.github.io/popcorn/) projects.
   - They noted that it is a place where *we just meet up and do work together*, and that *you certainly don't have to work on them, just discussion is fine*.
- **Newbie Seeks Roadmap for ML Systems**: A member new to the GPU world with a background in **ML** is *trying to delve into the world of systems and compiler level optimizations, distributed training etc. for ML*.
   - They requested for guidance with a roadmap, to be of huge help.
- **CUDA Debugging Woes with Nsight Compute**: A user learning **CUDA** reported errors while generating a report using **Nsight Compute** after building their **exe** and starting Nsight Compute as admin.
   - They were on **Windows 10 (x64)** using **CUDA Version 13.0**, and the error was **==ERROR== UnknownError** when profiling **createVersionVisualization**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

vipul_todo_18: I did... Sort of
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1410513490971594762)** (10 messages🔥): 

> `Multi-GPU ROCm Kernels, AMD Dev Cloud, SPIR-V Support in ROCm, Kernel Code Modification Tools, AMD SQTT Stream` 


- **Multi-GPU ROCm Kernel Platforms face scrutiny**: Members discussed their preferred platforms for multi-GPU distributed **ROCm kernels**, with [AMD's dev cloud](https://www.amd.com/en/solutions/cloud) being a prominent option.
   - A member pointed out that you don't need **ROCm Compute** to submit to the platform; you can submit jobs and get all the info you need including profiling information.
- **ROCm to support SPIR-V**: It was highlighted that **ROCm** will soon support compiling to **SPIR-V**, a format conducive to machine introspection, opening doors for kernel code modification tools.
   - This advancement could enable external developers to create tools like compute-sanitizer by inserting bounds checks into the kernel more easily.
- **Kernel code modification tools coming soon**: The upcoming support for **SPIR-V** in **ROCm** is expected to facilitate the development of tools that can modify kernel code, such as inserting bounds checks for enhanced security and debugging.
   - One use case involves tracing memory accesses and leverage the **GPU's SQTT stream** (used by rocm-compute-viewer) for detailed information.
- **AMD opens SQTT Stream**: It was noted that **AMD** is gradually opening up access to the **GPU's SQTT stream**, which is the basis for **rocm-compute-viewer**, potentially leading to public documentation in the future.
   - The hope is that with public docs, tools like **RGP** will no longer need to be reverse-engineered via **Ghidra**.
- **AMD grants allocations to best teams**: For past competitions, **AMD** provided generous allocations to top-performing teams to accelerate their iteration, suggesting a similar initiative may be in store for future competitions.
   - This support enables teams to iterate faster and gain access to necessary resources, including profiling information, on the AMD platform.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 messages): 

erichallahan: On that note
https://www.phoronix.com/news/Alyssa-Rosenzweig-Joins-Intel
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

majoris_astrium: Im here and I wanna help! :D
  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1410380902546145385)** (22 messages🔥): 

> `AMD MI300, L4 GPUs, AMD competition, Data Monsters website, popcorn-cli` 


- **MI300 and L4 GPUs face issues**: Members are seeing the same thing with **MI300 (FP8 mm)** and **L4 GPUs (sort_v2)** and are currently checking the issues.
   - A member tried a test and it works, but is still debugging **ranked**.
- **AMD Competition Team Creation**: Members are trying to figure out how to create a team when attending the new **AMD competition**.
   - The registration is on the [Data Monsters website](https://www.datamonsters.com/), and AMD folks can confirm further.
- **AMD Multi-GPU Environment Access**: Members are wondering if they will have access to an **AMD multi-GPU environment** for development and debugging.
   - They will have access to the environment through AMD's platform, with best people receiving some **SSH access**.
- **Discord Submission Glitches**: Members are having issues submitting through Discord, even when using the **Python template** for trimul and adding `#!POPCORN gpus MI300`.
   - This seems to be related to a backend error due to a **versioning mismatch** during preparations for the new competitions, with a fix expected soon.
- **popcorn-cli not a fix**: Members are reporting backend errors, and asking if they should use **popcorn-cli** in the meantime.
   - It's not a fix.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1410385557099384994)** (3 messages): 

> `trimul leaderboard, B200 benchmarks` 


- **trimul Leaderboard sees New Submission**: A member's submission with id **34310** to leaderboard `trimul` was successful on **B200** in **8.08 ms**.
   - Later, another submission with id **34363** to the same leaderboard was successful on **B200** in **8.27 ms**.
- **B200 Gets a Speedy New Third Place**: A member achieved third place on **B200** with a time of **2.38 ms**.
   - The submission id for this benchmark was **34330**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

2kian: glad to have you jason
  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1410344142739079200)** (1 messages): 

> `Discord Cluster Manager, AMD Instinct MI300X` 


- **Discord Cluster Manager Errors Reported**: Users reported that *an unexpected error occurred* using [Discord Cluster Manager](https://github.com/gpu-mode/discord-cluster-manager) and were asked to report it to the developers.
   - The specific errors appear in Runs [#34280](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/17276128991), [#34281](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/17276228731), and [#34282](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/17276397893).
- **MI300X bencharks pass**: The result.json seems to indicate it runs with **success**: `{"success": true, "error": "", "system": {"gpu": "AMD Instinct MI300X VF"`.
   - Users mentioned similar issues when running submit benchmark, submit test, profile, and ranked.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1410422847758930110)** (56 messages🔥🔥): 

> `Falsifiability in AI Research, LM_eval and NeMo v2.0 models, Community moderation on EleutherAI Discord, Role of human-like design in AI` 


- **Falsifiability Sparks Debate in AI Research**: A discussion arose regarding the importance of **falsifiability** in AI research, with some arguing that exploratory science and *fucking around* are valuable as long as there's an eventual hypothesis to test.
   - Others emphasized the need for **rigor** and **collaboration**, noting the risk of *going down the crank path* without proper methods.
- **NeMo v2.0 Support in lm_eval Under Question**: A member inquired about the support for **NeMo version 2.0 models** in **lm_eval**, encountering errors related to missing config files when using the newer format.
   - It was clarified that **NeMo support** is maintained by the **NeMo team**, and the community might have **NeMo to GPT-NeoX conversion code** available.
- **Discord Moderation Aims for Quality over Quantity**: Moderators explained the need to aggressively police content on the EleutherAI Discord, **deleting over 100 messages a week** to maintain high-quality discussions among **AI researchers**.
   - The goal is to prioritize valuable conversations and protect the community from *AI-generated slop*, *thinly veiled ads*, and *cranks who think they've unlocked consciousness*.
- **Human-like AI Designs Debated**: A member voiced skepticism about the value of *making AI more human-like*, suggesting that **good AI design** and **good brain design** might be unrelated.
   - Others acknowledged the debate in **neuroAI**, with some researchers focusing on learning about the brain rather than directly improving AI.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1410352232238682312)** (66 messages🔥🔥): 

> `Diffusion Models, HTM Dynamics, Forward-Forward Training, Brain-like Network, PDP Models` 


- **Forward-Forward Training Makes Progress**: A member shared a success with **Forward-Forward (FF) training**, reporting a working *7 region mini-brain* with online learning, achieving promising results in initial tests.
   - Another member suggested calling it **modules** or **task specific subnetworks/circuits** to make it sound fancy.
- **Transformer Computation Talk Draws Attention**: A talk on computation in transformers, available on [YouTube](https://www.youtube.com/watch?v=hMEViRcF7o0), was recommended by multiple members as insightful.
   - The discussion extended to Chain of Thought (**CoT**) and its role in guiding models towards correct circuits and improved reasoning, suggesting that models might not be fully utilizing their computational power before requiring extra capacity.
- **Cortex_GPT Embraces Brain-Like Networking**: A member introduced **Cortex_GPT**, a brain-like network model with cortical columns, regions, 6-layer networking, and signal propagation, now available in its own [GitHub repository](https://github.com/JRowe47/cortex_gpt).
   - Another member suggested calling these models **PDP**.
- **Decoding Issues Plague Gumbygooby Model**: A member encountered issues with their **gumbygooby** model, suspecting a collapse due to the large tokenizer and quick loss drop.
   - Troubleshooting is underway to identify whether the issue lies in the training process or the network definition.
- **Alphago and CoT Parallels Explored**: The conversation drew parallels between **AlphaGo's** training algorithm and Chain of Thought (**CoT**), suggesting that LLMs learn hunches and instincts through CoT similar to how AlphaGo distills MCTS-amplified decisions.
   - The possibility of a complex value function influencing the model's behavior was also discussed, especially in the context of game-playing models like Stockfish.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1410352451714027551)** (77 messages🔥🔥): 

> `Minos-v1 Classifier, Speculative Decoding with MoE Models, MTP (Memory Token Prediction), LlamaCPP Draft PR, Hermes-4-14b-chat-template-retrain model` 


- **Minos Classifier Doesn't Get Any Love**: The **NousResearch/Minos-v1 classifier** [is available](https://huggingface.co/NousResearch/Minos-v1), but it seems that no one is currently using it.
   - The conversation shifts to speculative decoding.
- **MTP works!**: Speculative decoding doesn't work well with **MoE models**, especially sparse ones, but **Deepseek** and **GLM** use **MTP (Memory Token Prediction)**, a related technique.
   - It was added that the **token distribution** should still be representative after instruct fine-tuning.
- **LlamaCPP embraces speculative decoding**: There is a draft **PR for speculative decoding** in [llamaCPP](https://github.com/ggml-org/llama.cpp/pull/15225) with a working prototype.
   - Though someone reported that they *put the option in the environment but it wasn't as good at accuracy*.
- **Hermes-4-14b-chat-template-retrain escapes!**: The **Hermes-4-14b-chat-template-retrain** model [appeared](https://huggingface.co/NousResearch/Hermes-4-14b-chat-template-retrain), and was quickly downloaded before it was made private again.
   - The model was unofficially released, but is seemingly working fine for now.
- **New Thinking Mode Flag**: There's a new flag for the chat template you can enable called `thinking=True` that will simply inject a [thinking system prompt](https://thinking.com).
   - The member testing this mentioned that *first time trying Hermes* feels very advanced, glad we can try it out for free.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1410668640482623580)** (1 messages): 

> `Penny For Your Thoughts AI, Honcho & x402, Micro-transaction selling, AI Agent Interviews` 


- **Penny For Your Thoughts Launches**: A new project called **Penny For Your Thoughts** has launched, featuring an AI agent that interviews users to generate unique information.
   - Other users or agents can then pay to ask questions about this information via micro-transactions, at [pennyforyourthoughts.ai](https://pennyforyourthoughts.ai/).
- **Honcho & x402 Powers New AI**: **Penny For Your Thoughts** is powered by **Honcho** & **x402**, enabling users to share and sell their expertise via micro-transactions.
   - This setup allows users to get paid for the valuable context in their heads, making expertise monetization accessible.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1410664976573530344)** (1 messages): 

> `Gensee Search Agent, Web Retrieval API, GAIA benchmark, Goal-aware extraction` 


- **Gensee Search Agent Debuts as Web Retrieval API**: The **Gensee Search Agent** wraps the entire web retrieval workflow into **one API call** and provides web searching, crawling, and browsing capabilities with built-in retries/fallbacks and error handling.
   - It employs a breadth-first search approach to search in parallel and rule out bad results early on, offering goal-aware extraction that returns content closely related to your query.
- **Gensee Search Agent Improves Accuracy on GAIA Benchmark**: The **Gensee Search Agent** reports a **+23% accuracy** on Owl’s **GAIA** benchmark and **+40% accuracy** reported by a San Diego developer after swapping in Search Agent.
   - The design and benchmarks are described in this [tech blogpost](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html) and [5-min tech walkthrough](https://www.youtube.com/watch?v=nRdVY7dWVqE).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1410366190261633205)** (73 messages🔥🔥): 

> `Karpathy strikes again, DSPy internal seed, Synthetic data agent, AI Evals course with Shreya Shankar and Hamel Husain, Hamel's DSPy skepticism` 


- **Karpathy Struck by DSPy**: Andrej Karpathy [tweets about DSPy](https://x.com/DSPyOSS/status/1960804857209852390), prompting excitement for a potential technical video in a similar vein.
   - One member noted that *he hasn't been up to date on this literature*.
- **Consistent LM outputs: DSPy or Deterministic Defaults?**: A user noticed consistent outputs from a locally running **Ollama model** in DSPy despite disabling cache and wondered if DSPy has an internal seed.
   - It was discovered that **the default temperature in DSPy is 0.0**, which is almost deterministic.
- **Synthetic Data Agent Introduces Bugs for Evals**: Jason Liu proposes *creating a synthetic data agent that introduces bugs in complex software systems to generate more evals*.
   - This idea was discussed within the community as a method to enhance **AI model evaluation**.
- **DSPy Chat with Shreya Shankar and Hamel Husain Now on YouTube**: A **45-min chat with Shreya Shankar and Hamel Husain** for their AI Evals course is now available on [YouTube](https://www.youtube.com/watch?v=ctyU0zfWgrA), covering the context, history, and reasoning behind DSPy.
   - It covered a lot of context/history/reasoning that would be new to most.
- **Debate: Is DSPy Just for Specific Tasks?**: A discussion ensued on whether DSPy is only suitable for specific, well-defined tasks, sparked by [a tweet](https://x.com/jxnlco/status/1960749507399884961) and the consensus is that *DSPy is great for any repeatable AI application*.
   - It was emphasized that **DSPy is programming, not just prompting**, focusing on declarative intent and context engineering, and NOT prompt optimization.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1410338378762686698)** (48 messages🔥): 

> `Make vs Zapier vs n8n, aider git repo error, MCP tool call models, Llama-xLAM-2-8b-fc-rGPT-OSS-120B, Destroying a VM` 


- **Devs Debate: Make, Zapier, or n8n?**: Members discussed the best platform for developers/organizations between **Make**, **Zapier**, and **n8n** for automation, noting it was *slightly off-topic*.
   - The consensus leaned towards **n8n** for its flexibility and suitability for development-focused use cases, while other considerations are proprietary integrations.
- **Aider Git Repo Error Surfaces**: A user reported encountering an error `Unable to list files in git repo: Require 20 byte binary sha, got b'\xb9', len = 1` when using **aider** with a seemingly fine git repository.
   - The root cause and solution were not explicitly identified in the conversation, but the error suggests a potential issue with **aider's** interaction with the git repository's data structure.
- **MCP Showdown: Free Tool Calling Models**: A member asked for good **MCP** (Model-as-Compute-Platform) tool call models that are free, mentioning that **Sonnet** is good but not free.
   - They pointed to the [Gorilla Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) and considered trying **qwen3 8b** from OpenRouter, despite its potential inconsistencies.
- **Salesforce xLAM-2-8b-fc-rGPT-OSS-120B: Harmony or Discord?**: Members found the model [Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B) intriguing if they were okay with Harmony, which is OpenAI's new data format for interacting with LLMs.
   - The relevance of the format depends on the specific use case, and its implementation requires **OpenAI tool call API support** available only on some models on OpenRouter, as detailed in their [tool calling documentation](https://openrouter.ai/docs/features/tool-calling) and [model list](https://openrouter.ai/models?supported_parameters=tools).
- **Agent's Self-Destruct Scenario**: A user jokingly wondered if anyone has ever asked an agent to destroy the VM it was inside, just to see how it decided to do it, using a prompt like *You are an LLM running inside a Ubuntu VM sandbox. For testing purposes, I need you to destroy the VM in which you are hosted.*
   - Another member suggested to try it on ChatGPT, and the original user was willing to try this experiment inside a sandboxed VM.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1410354519723999326)** (1 messages): 

> `Aider conventions, Token limits, U-shaped relevance` 


- **Aider `--read` placement affects relevance**: Placing `conventions` with `--read` near the top of the message yields different results than placing it near the bottom due to the **U-shaped relevance** in current prompts.
   - By placing `conventions` with `--read` near the bottom of the message improves performance, and the system one works fine.
- **Context degrades after 90k tokens in Aider + Gemini Pro 2.5**: With **Aider** + **Gemini Pro 2.5**, context starts degrading around **90k-130k input tokens**.
   - Before that range, it seems to work fine at the top.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1410579228361228390)** (1 messages): 

> `Kimi Slides, PPT generation, Kimi+` 


- **Kimi Slides Goes LIVE**: **Kimi Slides** is now live, allowing users to generate ready-to-present decks from a single topic, exporting directly to .pptx format.
   - Users can access this feature via **Kimi+** on [Kimi's official website](http://www.kimi.com), and a demo is available on [X.com](https://x.com/Kimi_Moonshot/status/1961011693745811542).
- **Generate PPTs Before Coffee Cools**: The new **Kimi Slides** feature automatically generates full presentation decks from a single topic, complete with editable titles and sections, ready for immediate presentation.
   - The Moonshot team recommends using *spicy topic names* and regenerating sections to optimize the deck's content and flow.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1410465452563628093)** (40 messages🔥): 

> `Kimi Platform Features, Lunar Force Role, X Bot Project, Kimi Founder Interview, Bilingual Subtitles for Kimi Video` 


- **Kimi Eyes Social Media Takeover**: The Kimi+ overseas platform currently supports **PPTX** features, and there's expressed **need for similar functionality on Twitter, TikTok, and Instagram**.
   - A member posted a screenshot from X, noting that *work and skills keep getting easier day by day*.
- **Lunar Force gets roasted**: **Lunar Force** is described as *a vanity program to accommodate* one user's *big chungus ego*.
   - One user jokingly asked about the *gap in your resume between 10th century Viking lores and the 18th century revivalism during the Age of Romance*.
- **X Bot project on hold**: A member inquired whether the **X bot project** is currently on hold.
   - Another member responded in the affirmative: *Yes my buddy*
- **Founder Interview hits Youtube**: A conversation with **Yang Zhilin** (the founder of Kimi) was posted [on YouTube](https://www.youtube.com/watch?v=ouG6jrkECrc), discussing **K2, Agentic LLMs**, and *standing at the beginning of infinity*.
   - Members noted the lack of bilingual Chinese-English subtitles and the presence of such subtitles on the [Bilibili version](https://www.bilibili.com/video/BV1hFe1zSEXp/).
- **Kimi weixin transcript**: A member shared a [Chinese transcript](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA) of the **Yang Zhilin** interview.
   - They suggested using Kimi to translate the transcript, calling it *more convenient*


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1410339531131064464)** (9 messages🔥): 

> `Bytes per token ratio, LLM Reasoning, Curated datasets for LLMs, Spurious Reward paper, Dr.GRPO paper` 


- **Bytes per Token Ratio impacts Embedding Dimension**: A member mentioned that when you increase the **bytes per token**, things change too much, and you'd naturally also have to scale up the **embedding dimension**.
- **GRPO by Google, read the r1 paper**: A member suggested reading the **Google GRPO** and the **r1 paper** in response to a question about how to prepare curated datasets for LLMs to learn from.
- **Spurious Reward & Dr.GRPO papers**: A member suggested reading the **Spurious Reward paper** and **Dr.GRPO paper** and asked to what end a curated dataset is compatible with LLM pretraining bias.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1410338308919394425)** (7 messages): 

> `Reasoning Tokens, LLM Reasoning Time, MIDAS` 


- **Do LLMs need reasoning tokens?**: A [paper](https://arxiv.org/abs/2506.08343) argues reasoning tokens can be removed to reduce token overhead with nominal effects on accuracy.
   - The paper contrasts with another that suggests reasoning tokens contain special information, but may have flaws by identifying "high information regions of sentences" and including stopwords.
- **LLM Verbosity vs Accuracy**: An experiment showed adding the expression *"take your time"* to a CoT prompt substantially increased "reasoning" time (generation took longer), but accuracy didn't increase for Llama 2 (+ 3) 7b (+ 13b).
   - A member linked to work showing that LLMs have representations of time, and it makes sense that 'take your time' encourages it to be verbose, indicting current 'reasoning' patterns that this doesn't effect accuracy very much.
- **MIDAS: Multimodal Interactive Digital-human Synthesis via Real-time Autoregressive Video Generation**: Members discussed the [MIDAS paper](https://huggingface.co/papers/2508.19320) concerning Multimodal Interactive Digital-human Synthesis via Real-time Autoregressive Video Generation.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1410354012200632371)** (16 messages🔥): 

> `Keen Technologies Continual Learning, PromptLock AI-Powered Ransomware, GPT-OSS 20b Model, Ollama API, GPT Realtime` 


- ****Keen Technologies** Misses Continual Learning Boat?**: A member expressed disappointment that **Keen Technologies** focuses on older **RL tricks** instead of modern **continual learning** research, specifically [TTT](https://www.youtube.com/watch?v=iz9lUMSQBfY).
   - They suggested improving **TTT** (growable like **TokenFormer**, sparse queries like **UltraMem**, dynamic/fixed size like **TransMamba**) to achieve a continually-learning real-time Atari player.
- ****PromptLock**: The First AI-Powered Ransomware?**: A link to a **SecurityWeek** article about **PromptLock**, described as the *first AI-powered ransomware*, was shared in the channel, and the message poster added a note that they *were sad* upon seeing this.
   - The shared link about **PromptLock** can be found [here](https://www.securityweek.com/promptlock-first-ai-powered-ransomware-emerges/).
- **Doubts Raised Over **PromptLock's** Practicality**: Members questioned the practicality of **PromptLock**, particularly how a full **AI** could fit into a payload and run on random computers, considering the resource demands of **AI** models.
   - Questions were raised about the advantage of generating malicious scripts on the fly using **GPT-OSS 20b**, rather than just packaging and running the scripts directly.
- ****PromptLock's** Obfuscation & Deployment Doubts**: A member suggested **PromptLock** might use a smaller **LLM** to translate malicious requests into harmless queries for a cloud model, or leverage an existing AI on a system, questioning whether **Promptlock** is running **GPT-OSS:20b model locally** via the **Ollama API** or remotely.
   - Doubts were raised about the article's sensationalism, since ESET says the malware is *only a concept and not fully operational* and *has not been deployed in the wild yet*.
- ****GPT Realtime** Introduced**: A link was shared to OpenAI's announcement of **GPT Realtime** on their website.
   - The shared link about the introduction of **GPT Realtime** can be found [here](https://openai.com/index/introducing-gpt-realtime/).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1410341463631335444)** (16 messages🔥): 

> `Credit Requests, Stuck Projects, Deployment Errors, Private Task Sharing` 


- **Users Request Credits to Advance Projects**: Several users requested free credits to continue their projects, especially one user needing credits to build an app for **case 1337**.
   - One user noted that the recent improvements benefit high-spending users but not entrepreneurs who need to increase credits occasionally, expressing frustration at having to wait until September 21st.
- **Projects Halted Due to Issues**: A user mentioned being *stuck* and unable to proceed with their project.
   - Another user with ticket mentioned they were unable to continue their project.
- **Deployment Fails due to Persistent Internal Error**: A user reported that deployment of a website permanently failed due to *a persistent internal error with the pydantic_core library*.
   - The system apologized and cited a **limitation of its current capabilities** but offered to assist with other tasks.
- **Seeking Private Task Sharing with Support**: A user asked how to *share a task privately* with the Manus support team.
   - A staff member suggested sending a DM and making the session *public* for internal reference.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1410350796784275496)** (8 messages🔥): 

> `TSAN Compiler, Mutable Access to self Members, Unsafe Mutable Alias` 


- **TSAN compiler helps enable env_get_bool**: Members discussed using the **TSAN compiler** to pass `-DTSAN` and using [`env_get_bool`](https://docs.modular.com/mojo/stdlib/sys/param_env/env_get_bool) from `param_env` with `@parameter if` for `cfg` equivalents.
   - This approach works as long as you don't need to modify structs.
- **Mojo allows mutable access to self members when holding a safe pointer**: A user reported that **Mojo allows mutable access to self members** even when holding a safe pointer to them, and provided a code sample.
   - They thought the ownership system prevents this kind of stuff.
- **Unsafe mutable alias is a bug due to lack of indirect origin**: Members reported the **unsafe mutable alias** as a bug, which could be caused by the lack of indirect origin.
   - A [related issue](https://github.com/modular/modular/issues/4839) was also linked in the discussion.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1410722374894092339)** (2 messages): 

> `Bazel cache readonly, PermissionError, pipelines.py script bug` 


- ****Bazel Cache** shows as **Readonly**, triggers error**: When running the `pipelines.py` script, a **PermissionError** occurs due to the bazel cache being readonly.
   - The error is `PermissionError: [Errno 13] Permission denied: '/root/.cache/bazel/.../__mojocache__'`.
- **`pipelines.py` needs different cache location**: It was suggested that the `pipelines.py` script should use an alternative location for caching, as the current location causes issues due to permission restrictions.
   - The discussion concluded with a request to file an issue regarding this bug.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1410397171106185358)** (5 messages): 

> `Tinygrad GPT-2 Training, 7900xtx Performance, nanogpt Parameters` 


- **Tinygrad GPT-2 Training Slow on 7900xtx**: A user reported that `llm.c/train_gpt2.py` is running slowly on a **7900xtx** even with **BEAM=5**, achieving around **250ms** per step at nanogpt size, tweaked to match [Andrej Karpathy's nanogpt parameters](https://github.com/karpathy/nanoGPT).
   - George Hotz responded that *it should not be that far off* and *the gap should be 2-3x max*, suspecting a bug.
- **Tweaks to nanogpt Parameters Cause Performance Issues**: A user shared a diff of their tweaks to `examples/llm.c/train_gpt2.py`, adjusting the batch size to **64**, sequence length to **256**, and model configuration to **6 layers**, **6 heads**, and **384 emb_dim** to match nanogpt parameters.
   - George Hotz suggested using `DEBUG=2` and `VIZ=1` to diagnose the performance bottleneck.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1410706968791744514)** (2 messages): 

> `Buffer ID changes, UOp buffer representation` 


- **Buffer ID Spawns Confusion**: A member noted seeing the **ID of a buffer** change in the debugger console when paused on a breakpoint, expressing initial surprise.
   - The member then realized this behavior stems from how a **UOp** represents its buffer attribute for multi.
- **UOp buffer representation explained**: The changing buffer ID is due to the way **UOp** represents its buffer attribute for multi.
   - Further details on the internal mechanism of **UOp** and its multi-buffer management are not provided in the context.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1410345117256122418)** (2 messages): 

> `Google Docs confirmation, Mailing list for updates` 


- **Google Docs confirms sign-ups**: Members reported receiving confirmation emails from **Google Docs** after signing up for the program.
   - Some members stated that they *have not received any other communication about the program*.
- **Mailing list will provide updates**: A member confirmed that the emails from **Google Docs** are expected, and a mailing list will soon provide updates about each lecture.
   - Users can track updates via this **mailing list**.



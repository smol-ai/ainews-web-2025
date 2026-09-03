---
id: MjAyNS0x
title: not much happened today
date: '2026-08-24T05:44:39.731046Z'
description: >-
  **Stanford** is formalizing AI-native software engineering with a major
  curriculum overhaul replacing **85% of Fall 2025 material** to focus on
  **agent skills, context engineering, MCP portals, agent-ready codebase design,
  agentic code review, security, parallel background agents, and software
  factories**. Two new courses emphasize **systems-oriented agent engineering**
  over prompting, highlighting stateful intelligence allocation and dynamic task
  understanding. Rumors about OpenAI's **Astra** architecture describe it as a
  **looped transformer**, a modest architectural tweak similar to **Nanbeige
  4.2-3B** with layer reuse and adaptive computation passes, clarifying that
  recurrence does not obscure chain-of-thought reasoning. Infrastructure updates
  include **Photon 2.1** with **text-to-speech** and **NVIDIA B200** support,
  and **Baseten's GLM-5.3 Fast** for real-time multimodal inference. ByteDance
  Seed's **HarnessDev** reframes agent evaluation around the harness rather than
  task completion.
companies:
  - stanford
  - openai
  - baseten
  - bytedance
models:
  - nanbeige-4.2-3b
  - glm-5.3
topics:
  - agent-engineering
  - curriculum-development
  - software-engineering
  - looped-transformers
  - model-architecture
  - recurrent-neural-networks
  - chain-of-thought
  - real-time-inference
  - multimodality
  - text-to-speech
  - infrastructure
  - agent-evaluation
  - dynamic-intelligence-allocation
  - open-source
people:
  - mihail_eric
  - diyi_yang
  - michaelryan207
  - harrystebbings
  - enoreyes
  - jerryjliu0
  - rasbt
  - vikhyatk
  - omarsar0
---


**a quiet day.**

> AI News for 8/22/2026-8/24/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Agent Engineering Courses, Curricula, and Developer Practice**

- **Stanford is formalizing AI-native software engineering as a discipline**: [@mihail_eric](https://x.com/mihail_eric/status/2095166860740174273) announced a new edition of *The Modern Software Developer* centered on what he calls the “2026 metamorphosis” of software engineering. The notable signal is not just the course itself, but the curriculum reset: **85% of Fall 2025 material is being replaced** with topics like **agent skills, context engineering, MCP portals, agent-ready codebase design, agentic code review, security, parallel background agents, and software factories**. The course also requires students to ship PRs into real OSS repos with support from partners including Browserbase, OpenHands, Semgrep, Milvus, Marimo, CrewAI, Warp, Vercel, Unsloth, and Anyscale, among others.
- **A second Stanford course focuses on first-principles agent construction**: [@Diyi_Yang](https://x.com/Diyi_Yang/status/2095192282970615970) and [@michaelryan207](https://x.com/michaelryan207/status/2095224415567167978) announced **CS329Z: Engineering AI Agents**, explicitly framed around building agents “from scratch.” Alongside Mihail Eric’s course, this suggests a broader shift from “prompting” pedagogy to **systems-oriented agent engineering**: harnesses, evaluation, memory, tooling, orchestration, and production constraints rather than model usage alone.
- **Practitioner discussion is converging on stateful intelligence allocation, not simple routing**: In a panel prompt, [@HarryStebbings](https://x.com/HarryStebbings/status/2095179442276741450) highlighted @EnoReyes’s argument that getting the most out of models requires more than routing—agents need to **understand task state, what just happened, and what comes next** in order to allocate intelligence dynamically. That lines up with [@jerryjliu0](https://x.com/jerryjliu0/status/2095344824266178662)’s point that **vendor-neutral startups** can outperform frontier labs on narrow tasks by optimizing the harness end-to-end and selectively using both frontier and open-weight models.

**Model Architecture and Inference: Astra Rumors, Looped Transformers, and Real-Time Serving**

- **The “Astra is a looped transformer” rumor is probably less novel than headlines suggest**: [@rasbt](https://x.com/rasbt/status/2095141254958858496) unpacked reporting around OpenAI’s rumored **Astra** architecture and argued that the cited “recurrent depth” or “looped transformer” concept is a fairly modest architectural tweak rather than a breakthrough on its own. He points to **Nanbeige 4.2-3B** as an open-weight precedent: a **22-layer transformer stack reused twice**, effectively behaving like a **44-layer model** without doubling parameter storage. The tradeoff is straightforward: **similar memory footprint, roughly ~2x compute**, and only partial token-efficiency retention versus a standard stack. The more substantive historical reference is **Mixture-of-recursions**, where a learned router adaptively determines how many passes a token gets, allowing easy tokens to exit early and hard tokens to receive more compute.
- **Hidden reasoning is not a necessary implication of recurrence**: A second important clarification from [@rasbt](https://x.com/rasbt/status/2095141254958858496) is that layer reuse **does not inherently “obscure chain-of-thought”**. It simply moves more computation into latent activations before token emission. If recurrent depth reduces visible reasoning traces, that’s because the model may need to emit fewer intermediate tokens, not because looped transformers intrinsically suppress textual CoT.
- **Serving infra updates continue to target realtime multimodal workloads**: [@vikhyatk](https://x.com/vikhyatk/status/2095230035707977947) announced **Photon 2.1**, adding **text-to-speech models** and **NVIDIA B200 support** to a realtime multimodal inference engine. Separately, Baseten announced hosted availability of **GLM-5.3 Fast**, emphasizing **higher TPS** and real-time deployment positioning via [@baseten](https://x.com/baseten/status/2095338689492578693).

**Agent Harnesses, Skill Retrieval, and RL Post-Training Tooling**

- **ByteDance Seed’s HarnessDev reframes agent evaluation around the harness, not just task completion**: [@omarsar0](https://x.com/omarsar0/status/2095170896407548190) highlighted a new paper on **HarnessDev**, which asks models to start from a weak but runnable seed and build an execution harness, then improve it in a second stage using downstream feedback. Both stages are scored on **capability and execution-token cost**, making efficiency part of the objective. Across **six creator LLMs, four domains, and 2,207 held-out downstream instances**, generated harnesses still lag mature human-engineered systems on **code, search, and research**, but **match or exceed them on writing and ML experimentation**. The key nuance is that self-evolving harnesses help, but gains are **unstable, model-dependent, and only partially transferable**.
- **Related ecosystem signal: exo and recursive self-improvement tooling**: [@omarsar0](https://x.com/omarsar0/status/2095204228687945880) also called out the **exo harness** as a useful entry point for understanding recursive self-improvement workflows, indicating a growing interest in frameworks where agents improve not just outputs but their own scaffolding.
- **Skill retrieval may look good in aggregate while hurting the tasks that actually trigger it**: [@dair_ai](https://x.com/dair_ai/status/2095330956823629995) summarized a paper proposing **Retrieval-Invoked Actual-Use Effect**, a matched-evaluation method that runs the **same task twice**, with and without skills enabled, and only counts tasks where retrieval actually fired. Across **17 LLMs** on coding and math, the paper finds cases where retrieval improves overall scores while having a **negative same-task effect** on the subset of tasks where it was used. For teams maintaining skill libraries or tool directories, this is a practical warning against over-interpreting aggregate lift.
- **RL post-training infra is becoming more productized**: The SGLang team promoted an event with Baseten and NVIDIA Dynamo around **Miles**, an RL training framework that uses **SGLang as the rollout inference engine** for faster, more reliable RL post-training [@sgl_project](https://x.com/sgl_project/status/2095200888197722439). [@AravSrinivas](https://x.com/AravSrinivas/status/2095354358145892733) separately described **Miles** as **open-source RL-as-a-service**, reinforcing the trend toward reusable post-training stacks rather than bespoke internal pipelines.

**Google Gemini 3.8 Flash Cyber and Production Friction Around Google Tooling**

- **Google introduced a specialized cybersecurity model with strong benchmark claims**: [@sundarpichai](https://x.com/sundarpichai/status/2095184464800526655) announced **Gemini 3.8 Flash Cyber**, positioned as Google’s most capable cybersecurity model while retaining **Flash-level speed and pricing**. Reported numbers include **86.2% on CyberGym**, **47.2% on CWE-Bench for patching**, and **70%+ success** on an internal vulnerability-discovery benchmark across **20 programming languages**.
- **At the same time, developer sentiment points to harness and account-risk concerns**: [@theo](https://x.com/theo/status/2095328650459840627) argued that Google currently has weak developer ergonomics around harnesses, code apps, third-party integration, and especially **aggressive bans tied to core Google accounts**. [@QuinnyPig](https://x.com/QuinnyPig/status/2095331997640220872) sharpened that concern, noting the blast radius can extend beyond Gmail/Workspace to **Google Cloud accounts associated with the same identity**. Theo’s later complaints about **slow, tool-call-heavy coding behavior** on Gemini tasks ([1](https://x.com/theo/status/2095332853978702280), [2](https://x.com/theo/status/2095337761423466784), [3](https://x.com/theo/status/2095316221789139362)) are anecdotal, but they underline the gap between benchmark performance and **production developer UX**.

**Meta Muse Spark 1.3 and the Video/Multimodal Release Cycle**

- **Meta launched Muse Spark 1.3 for agentic and coding workloads**: [@shengjia_zhao](https://x.com/shengjia_zhao/status/2095233023247880590) introduced **Muse Spark 1.3** as the strongest model in the Spark line for **agentic and coding tasks**, with emphasis on **longer-horizon work** and more reliable compliance with complex instructions. Community reactions emphasized its price/performance envelope, including [@alexandr_wang](https://x.com/alexandr_wang/status/2095328657241956576) calling out what it can do “for a single dime,” while other users compared it favorably on speed and token efficiency versus competing “xhigh” offerings.
- **Alibaba’s Wan 3.0 is posting strong third-party leaderboard results in video**: [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2095349174799888760) reported that **Wan 3.0** ranks **#1 on Video Editing with Audio**, **#2 on Text-to-Video with Audio**, and **#5 on Image-to-Video with Audio** on Artificial Analysis leaderboards. The release is positioned as an **all-in-one generation and editing model** that accepts text, images, video, audio, documents, and web pages as references, supports **native audio**, and generates up to **30 seconds at 1080p**. Pricing in public preview starts at **$0.05/s for 480p**, rising to **$0.20/s for 1080p**.
- **Reference-heavy multimodal UX is also improving**: [@imagine](https://x.com/imagine/status/2095249317875622255) announced support for **up to 14 references per video**, spanning images, voices, and character references via `@`-tagging in prompts, a small but practical interface improvement for multi-asset creative control.

**Open Models, Robotics, and Top Tweets**

- **Open model efforts continue to scale up**: [@percyliang](https://x.com/percyliang/status/2095255747487740401) shared that **Marin 535B-A23B** is **13% through training**, with compute funded via the **Jen-Hsun and Lori Huang Foundation** and run on **CoreWeave**. The post is notable less for a benchmark than for the continued viability of large-scale open-model training backed by philanthropic compute support.
- **Physical AI and open robotics platforms are inching forward**: [@maze_rapid](https://x.com/maze_rapid/status/2095294835364364337) announced the **Palmimo DevKit**, a tabletop AI robot platform with open-source software and swappable AI “brains,” designed so developers can control robot applications from a few lines of Python without deep robotics expertise. It’s early, but relevant as an example of **agent frameworks extending into embodied systems**.
- **Top tweets (by engagement)**:
  - [@mihail_eric](https://x.com/mihail_eric/status/2095166860740174273): Stanford’s revamped **AI-native software developer** course with major curriculum turnover and OSS collaboration.
  - [@sundarpichai](https://x.com/sundarpichai/status/2095184464800526655): **Gemini 3.8 Flash Cyber** launch with strong cybersecurity benchmark claims.
  - [@rasbt](https://x.com/rasbt/status/2095141254958858496): Detailed architectural breakdown of **looped transformers** and why Astra rumors may be overstating novelty.
  - [@Diyi_Yang](https://x.com/Diyi_Yang/status/2095192282970615970) / [@michaelryan207](https://x.com/michaelryan207/status/2095224415567167978): New Stanford course **CS329Z: Engineering AI Agents**.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Muse Spark and Spark-X2.5 Open-Weight Models

  - **[Muse Spark open weights coming soon](https://www.reddit.com/r/LocalLLaMA/comments/1w5l8bw/muse_spark_open_weights_coming_soon/)** (Activity: 902): **The [image](https://i.redd.it/apwfejcow5nh1.png) is a screenshot of a **Mark Zuckerberg/X post** announcing **Muse Spark 1.3** rollout, claiming major improvements in **coding**, **agentic workflows**, and **long-context** tasks, with **Muse Spark open weights “coming soon.”** The included benchmark table positions **Muse Spark 1.3** above Muse Spark 1.2 and competitive with models labeled **GPT 5.6 Sol** and **Opus 5** across agent, long-context, and coding evaluations, though the Reddit post’s author notes Spark may be too large for their hardware and says they are waiting for **Llama 5** or an intermediate model between **Glimmer** and **Spark**.** Commenters frame the results as evidence that multiple leading labs are converging technically, with one saying there is *“no secret sauce”* and that frontier gaps may only be a few months. Another commenter argues **Muse Glimmer** is underrated and claims it outperforms **Qwen 3.8:27B** on non-coding tasks.

    - Commenters highlighted an unusually high reported long-context result: **MRCR `512k–1m` at `98.1%`**, with one user asking whether this implies Muse Spark has effectively solved “context rot” at million-token scale. If accurate, that benchmark would be the most technically notable claim in the thread because sustained retrieval/reasoning quality across `512k+` contexts is still a major weakness for many open and closed models.
    - One user reported that **Muse Glimmer** is “pretty good” and subjectively superior to **Qwen 3 8/27B** for non-coding tasks, suggesting Muse’s smaller/previous model may already be competitive outside programming benchmarks. The comparison is anecdotal, but it points to task-dependent strengths rather than blanket leaderboard performance.
    - Several commenters questioned the likely parameter count behind the displayed scores, with speculation that Muse Spark could be **trillion-parameter scale** if the benchmarks are accurate. That raised practical deployment concerns: it may not be locally runnable for hobbyists, but open weights could still be useful for organizations needing non-Chinese model options for policy/compliance reasons.

  - **[New Model: Spark-X2.5-4B, Spark-X2.5-1.7B](https://www.reddit.com/r/LocalLLaMA/comments/1w4dsrw/new_model_sparkx254b_sparkx2517b/)** (Activity: 301): ****XHToken** released **Spark-X2.5** [`1.7B`](https://huggingface.co/XHToken/Spark-X2.5-1.7B) and [`4B`](https://huggingface.co/XHToken/Spark-X2.5-4B), apparently a custom architecture rather than a simple fine-tune, with model cards claiming **native `1M` token context**, multilingual support, and training on roughly `20T` tokens plus long-context/post-training stages. The architecture reportedly uses a mix of full attention and sliding-window attention to reduce long-context KV/compute cost, and the `4B` benchmark claims are framed as competitive with much larger models such as Qwen-class ~`9B` models. Runtime support is not yet upstreamed in `llama.cpp`; it depends on a pending [`llama.cpp` PR #27868](https://github.com/ggml-org/llama.cpp/pull/27868) or XHToken’s custom fork, with GGUFs available for [`1.7B`](https://huggingface.co/XHToken/Spark-X2.5-1.7B-GGUF) and [`4B`](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF).** Commenters were mainly impressed by the reported `20T`-token pretraining scale and especially the claimed **native `1M` context** at sub-5B parameter sizes. There was cautious interest in whether the benchmark claims—particularly `4B` matching a ~`9B` model—hold up in independent testing.

    - Commenters highlighted the reported **`20T` training-token scale** for Spark-X2.5, which is unusually large for the **1.7B/4B** parameter range and could explain the claim that the **4B** variant matches a **9B** model if benchmarks reproduce. The other standout spec was **native `1M` context** at this model size, which readers viewed as more technically notable than raw benchmark parity.
    - One tester reported early qualitative behavior using a “pi harness”: when asked *“what model are you,”* the model appeared to use tools to inspect/analyze the harness name before answering, suggesting agentic/tool-use tendencies but also *“overthink[ing] a lot.”* In a quick reasoning check, it failed the “car wash” test, and the tester planned further comparison against **Qwen3.5 9B** for daily-use quality.


### 2. Qwen3.8 Benchmarks and GGUF Speedups

  - **[Qwen will be the king?](https://www.reddit.com/r/LocalLLaMA/comments/1w53ti8/qwen_will_be_the_king/)** (Activity: 732): **The [image](https://i.redd.it/m9c7ldofb2nh1.png) shows an **Arena AI Code Arena WebDev leaderboard** where **Qwen3.8-Max-0902** ranks #1 with a score of `1,691`, narrowly ahead of **Claude Opus 5 Max** at `1,688` and **Kimi K3 Max** at `1,674`. In context of the post, the result is being used to argue that **Qwen’s extended reasoning/post-training scaling** may be closing the gap with much larger frontier systems, potentially before a future Qwen 4 release or possible open-weight update.** Commenters were notably optimistic about local/open-weight Qwen variants, with one claiming **Q3.8-27B** running locally outperformed their paid ChatGPT coding experience. Others questioned whether the top-performing Max model will become open-weight, while one commenter praised extended reasoning but noted the tradeoff: *hours* of latency for difficult tasks.

    - A user reports strong local coding performance from **Q3.8-27B** used with **PI**, claiming it outperformed their prior paid **ChatGPT 5.1** access for coding tasks. They emphasize practical task-following: when supplied with relevant context such as wiki pages in `.txt` files, the model generated working code with few fixes while running fully on a local PC and preserving data privacy.
    - Several commenters focus on **extended reasoning** as a major differentiator: one says **Qwen 3.8 Max** is *“100% correct”* on their challenge set but can take **hours** to arrive at an answer. This frames the tradeoff as accuracy/reliability versus very high inference latency for reasoning-heavy workloads.
    - There is skepticism about the presented benchmark graph, with one commenter saying the numbers look *“very massaged”* and another asking why **Fable 5.1** is absent from the comparison. The concern is that model-ranking claims may depend heavily on benchmark selection, reporting methodology, or omitted competitors.

  - **[MTP released for Qwen3.8-Flash-Next-GGUF](https://www.reddit.com/r/LocalLLaMA/comments/1w42biu/mtp_released_for_qwen38flashnextgguf/)** (Activity: 671): ****Unsloth released MTP support/files for [`Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md)**, with test instructions tied to an Unsloth `llama.cpp` branch/PR ([`unslothai/llama.cpp#144`](https://github.com/unslothai/llama.cpp/pull/144/changes)) and GGUF usage paths targeting local runtimes/OpenAI-compatible endpoints. A commenter points to a newly merged upstream `llama.cpp` optimization ([`ggml-org/llama.cpp#28123`](https://github.com/ggml-org/llama.cpp/pull/28123)) reporting MTP throughput improvements from `123 tok/s → 183 tok/s` on code and `83 tok/s → 144 tok/s` on prose, versus `108 tok/s` without drafting; before the patch, prose MTP was reportedly slower than no draft at all.** Comment discussion is mostly practical: users ask whether **SSD offload** is stable/“ironed out” and note that the MTP files may have already been available for a few days.

    - A commenter cites a newly merged **llama.cpp** optimization PR ([ggml-org/llama.cpp#28123](https://github.com/ggml-org/llama.cpp/pull/28123)) showing major MTP throughput gains for **Qwen3.8-Flash-Next-GGUF**: baseline without draft was `108 tok/s`, pre-change MTP was `123 tok/s` on code but only `83 tok/s` on prose, and post-change MTP improved to `183 tok/s` code / `144 tok/s` prose. The key technical point is that before the merge, MTP could be slower than normal decoding on prose workloads, but the patch appears to make drafting consistently beneficial.
    - Several commenters are tracking unresolved runtime/support details in **llama.cpp**, including whether **SSD offload** is stable and what the `-shared` option changes versus non-shared mode for MTP files. Another user notes they believed the required llama.cpp feature support was still not fully merged, and reports low local performance of only about `9 tok/s`, implying hardware/configuration sensitivity remains significant.


### 3. Gemma Mystery Models on Arena

  - **[New Gemma models on arena ai](https://www.reddit.com/r/LocalLLaMA/comments/1w47nif/new_gemma_models_on_arena_ai/)** (Activity: 1031): **A Reddit user spotted apparent **new Gemma-family models** on Arena AI via a [screenshot](https://preview.redd.it/via5e88evvmh1.png?width=566&format=png&auto=webp&s=669459ca93ff292f4e1574d098e3e2a0b2c12de4), speculating whether they indicate **“Gemma 5 or something else.”** The only substantive technical comment notes that current Gemma architecture reportedly has a **VRAM-heavy KV cache** and that the cache *“does not quantize well,”* implying inference-memory efficiency remains a concern for future releases.** Commenters were broadly excited about a new Gemma model appearing soon, with one describing a “new model” as likely on the way; the main technical hope is improved KV-cache efficiency rather than just higher benchmark performance.

    - A commenter highlighted a concrete serving/inference concern with **Gemma**: its current architecture reportedly uses substantial VRAM for the **KV cache**, which can limit long-context deployment efficiency. They also noted that the cache “does not quantize well,” implying that common memory-reduction techniques may be less effective for Gemma than for some competing architectures.
    - Several commenters framed the desired direction for new **Gemma** releases as a **generalist multilingual model** rather than a narrowly optimized agentic/coding model. The requested improvements were specifically in **knowledge, reasoning, and multilingual capability**, preserving Gemma’s usefulness for broad chat and non-English tasks.

  - **[Fingers crossed for a 122b or really anything above 31b.🤞](https://www.reddit.com/r/LocalLLaMA/comments/1w4l9cp/fingers_crossed_for_a_122b_or_really_anything/)** (Activity: 949): **The image is a **meme/non-technical speculation post** about cryptic Google/Gemma entries in an “Arena roster update,” with names like `gemma-b2-2048s300-wd`, prompting guesses about whether they indicate new Gemma model parameter sizes. The discussion centers on whether these mystery models could be **larger Google/Gemma releases**—the title hopes for `122B` or anything above `31B`—or instead **image-input/context/configuration updates** to existing Gemma models, with `2048`/`4096` possibly referring to resolution, sequence length, or eval/config variants rather than parameter count. [Image](https://i.redd.it/xma5qw3wgymh1.jpeg)** Commenters were split between wanting a larger flagship model and preferring efficient local-friendly models; one argued a `122B` model is impractical for local inference and that Google should emulate Qwen-style ~`27B`/MoE designs instead. Another commenter speculated the entries are likely Gemma 4 image/text updates rather than new hidden model sizes.

    - Several commenters speculated the announcement may be an **update to Gemma 4 image/text capabilities** rather than a new larger checkpoint, citing the apparent `2048`/`4096` context or resolution references and “Image/Text” wording as signals of multimodal input updates instead of a hidden new model name.
    - There was pushback against a hypothetical **122B Gemma** for local inference, with commenters arguing that models in the **~27B–31B** range are more practical for consumer hardware. **Qwen 3.8 27B** was repeatedly cited as a target archetype, with suggestions for similarly sized MoE designs such as `30B-A3B` that could improve capability while keeping active parameters manageable.
    - One commenter highlighted **GemmaDiff/DiffGem** as an under-discussed diffusion-based MoE model, claiming it showed stronger general intelligence than **Qwen 3.6 MoE** in their testing while losing specifically on coding. They framed diffusion LLMs as analogous to having “Dspark or Dflash built directly into the model,” suggesting interest in Google leaning further into diffusion-style architectures.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo


### 1. Claude Fable 5.1 Release and Agent Coding Demos

  - **[Introducing Claude Fable 5.1 and Claude Mythos 5.1](https://www.reddit.com/r/ClaudeAI/comments/1w4juuz/introducing_claude_fable_51_and_claude_mythos_51/)** (Activity: 1367): ****Anthropic** announced **Claude Fable 5.1** and restricted-access **Claude Mythos 5.1**, claiming Fable 5.1 improves long-running coding/knowledge tasks and research workflows, with benchmark scores of `52.6%` on **Terminal-Bench-Science 0.1** (>2× Fable 5) and `55.8%` on **Terminal-Bench 4.0** vs `42.0%` for Fable 5 ([announcement](https://www.anthropic.com/claude-fable-and-mythos-5-1)). They also report **75% cheaper cache reads**, translating to ~`25%` lower typical workload cost and up to `45%` for highly agentic workloads, plus safeguard tuning that reduces benign cybersecurity false positives by ~`60%` and biology/medical fallback rates by ~`85%`.** Comments were mostly critical: users complained about unresolved quality issues in prior models, lack of Pro-user access, and prose regressions. One technical commenter cited Anthropic’s own prompt-engineering guidance for Fable 5.1, noting that while writing is less formulaic, it can be denser and may require explicit anti-style instructions like *“Please remove all mannered prose”* ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1)).

    - A technically substantive comment cites Anthropic’s own Fable 5.1 prompting guidance, noting that **Claude Fable 5.1 improves over earlier Claude writing models** with fewer stock phrases and less unexplained jargon, but can produce **denser prose than Claude Fable 5** via longer sentences and fewer paragraph breaks. The commenter highlights Anthropic’s recommended mitigation: explicitly instructing the model to avoid “mannered prose,” e.g. *“Please remove all mannered prose,”* referencing the official docs: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1
    - One user reports an unresolved behavior issue with **Claude Opus 5**, specifically “rambling irrelevant responses,” implying continued problems with verbosity control and answer relevance despite the release of Fable 5.1/Mythos 5.1. This suggests users are still seeing instruction-following or response-targeting regressions in Anthropic’s higher-end model line.

  - **[Fable 5.1 made a Minecraft mod for $20](https://www.reddit.com/r/ClaudeAI/comments/1w5ftqe/fable_51_made_a_minecraft_mod_for_20/)** (Activity: 1596): **The post claims **Anthropic Fable 5.1**, run via an API key in [`atomic.chat`](http://atomic.chat) agent mode, generated a complete **Fabric `1.21.1` Minecraft mod** from two YouTube video references: a Naruto Kirin scene and a railgun-mod short. The agent reportedly performed video analysis, wrote the mod, created Blender assets via a Blender MCP bridge, generated gun/dragon models and textures, launched Minecraft for testing, then applied one feedback pass; reported usage was `~383.6k` output tokens, `$20.54` API cost, and `~1 hour` total time. The resulting mod was released on [GitHub](https://github.com/AtomicChatRepo/MinecraftKirinGunMod), though the linked Reddit-hosted demo video was not accessible from the provided external URL due to a `403 Forbidden` block.** Top comments were mostly non-technical reactions expressing surprise at the low cost and capability, e.g. *“$20... i love it”* and *“fable is a freak”*. No substantive technical debate appeared in the provided comments.

    - Commenters raised two technical follow-ups: the implied **`$20` API cost** for generating a Minecraft mod, and interest in the exact prompt used to reproduce or evaluate Fable 5.1’s output. No benchmark details, implementation notes, mod architecture, or performance measurements were provided in the comments.


### 2. Gemini 3.8 Flash and OpenAI Astra Reasoning Claims

  - **[Gemini 3.8 Flash Benchmarks](https://www.reddit.com/r/singularity/comments/1w5d1pz/gemini_38_flash_benchmarks/)** (Activity: 1186): **The image is a technical benchmark table, not a meme: [**“Gemini 3.8 Flash Benchmarks”**](https://i.redd.it/3gx8kqsjh4nh1.jpeg) compares **Gemini 3.8 Flash** against Gemini 3.7 Flash, Claude Opus/Sonnet 5, and GPT-5.6 variants. It highlights Gemini 3.8 Flash as a low-cost model at `$0.75` input / `$3.75` output per `1M` tokens while reportedly leading or tying several agentic, reasoning, legal/finance, vision, and bio benchmarks; Claude Opus 5 still leads some heavier coding/agent benchmarks like DeepSWE, Terminal-bench 4.0, and OSWorld-2.0.** Commenters focus on the apparent **price/performance gap**, arguing Flash-class models may be encroaching on use cases previously reserved for expensive frontier models. Several users also report that Gemini 3.8 Flash feels faster than 3.7 Flash and praise it for non-coding/search-heavy tasks, while asking for more real-world coding impressions.

    - Several commenters highlight a perceived **price/performance shift** for Gemini Flash-class models, suggesting Gemini 3.8 Flash may be encroaching on use cases previously reserved for more expensive frontier models. One commenter specifically frames the trend as Flash models “eating into” premium-model territory if the benchmark numbers hold.
    - Early anecdotal testing claims Gemini 3.8 Flash feels **faster than Gemini 3.7 Flash**, though commenters note the testing is limited and no hard latency, throughput, or benchmark figures are provided in the thread.
    - Users call out Gemini Flash’s utility for **non-coding tasks with search-augmented workflows**, citing fast responses and strong Google Search integration compared with slower Claude search experiences. One commenter also points to Google’s scale and search user base as a likely reason Gemini can offer competitive **input/output token pricing**.

  - **[if true openai has made another o1-level breakthrough](https://www.reddit.com/r/singularity/comments/1w4w5g0/if_true_openai_has_made_another_o1level/)** (Activity: 1100): **A Reddit post claims, based on an alleged screenshot/report, that **OpenAI** has trained a model referred to as **“Astra”** to perform **latent-space reasoning**—i.e., intermediate computation not fully externalized as natural-language chain-of-thought ([image](https://preview.redd.it/gxza3a3wd0nh1.png?width=1168&format=png&auto=webp&s=379e06f09940791c6d5eb95a80b288f60caddd4f)). The technical premise is that reasoning in hidden/continuous representations could be more efficient and expressive than verbose token-level traces, especially for non-linguistic tasks such as spatial reasoning, and commenters note related research activity since `2024` plus potential benefits for smaller/local models where long reasoning traces are costly.** Commenters split between enthusiasm for frontier-model deployment of latent reasoning and concern over interpretability/safety: one framed it as giving the model an *“internal monologue,”* while another argued that letting models *“think in silence”* could worsen already-poor transparency compared with English chain-of-thought.

    - Several commenters interpret the rumored breakthrough as **latent-space reasoning/internal monologue**, noting that many papers since `2024` have explored reasoning without emitting long natural-language chains of thought. The technical appeal raised is inference efficiency: visible reasoning traces can be expensive for local/open models such as **Qwen 3 / ~27B-class models**, whereas latent reasoning could reduce token overhead while preserving deliberation.
    - A recurring concern is **interpretability and safety**: if models reason in latent activations rather than English-like traces, debugging and auditing become substantially harder. Commenters argue this moves from already-difficult chain-of-thought interpretability to “hard mode,” because the model’s intermediate reasoning may no longer be inspectable as text.
    - One commenter asks whether the approach implies progress on **recursive self-improvement (RSI)** or whether it is being used as a substitute for **continual learning**. The technical distinction raised is that latent reasoning may improve test-time compute/reasoning depth, but it does not inherently solve persistent online learning or model weight updates after deployment.


### 3. Anthropic Claude Plan Limits and Pro Access

  - **[According to their own internal documents a lawsuit filed against Anthropic reveals, that the 20x usage plan actually only allows for 6x more usage.](https://www.reddit.com/r/singularity/comments/1w43cci/according_to_their_own_internal_documents_a/)** (Activity: 1397): **The image ([link](https://i.redd.it/xhgz28nrnumh1.jpeg)) shows an alleged Anthropic internal table from a lawsuit comparing Claude Sonnet 4 usage limits: **Pro** at `40–80 hours/week`, **Max 5x** at `140–280 hours/week`, and **Max 20x** at only `240–480 hours/week`. The technical significance is that the advertised “20x” tier is presented as roughly **6x Pro usage**, not a literal `20x` multiplier, while “Max 5x” is also closer to **3.5x Pro** by the table’s numbers.** Commenters focused on the discrepancy between marketing labels and actual quota scaling, with some suggesting multiple Pro accounts might offer better value. Others questioned the sourcing, noting that the post depends heavily on the authenticity and context of the alleged lawsuit/internal document.

    - A commenter provided the primary source for the claim: **Kahn v. Anthropic PBC, No. 3:26-cv-05763**, filed June 14, 2026 in the Northern District of California, pointing specifically to **page 16** of the complaint: [courtlistener PDF](https://storage.courtlistener.com/recap/gov.uscourts.cand.472161/gov.uscourts.cand.472161.1.0.pdf). This is the only comment that anchors the “20x plan only allows 6x usage” allegation to a verifiable legal filing rather than speculation.
    - One commenter challenged the quantitative consistency of the alleged usage limits, noting that **5 consecutive sessions × 8 hours/day × 7 days/week = `280` hours**, which they argue does not align with the numbers being discussed. They also questioned why Anthropic would describe a “maximum allowance” as a **range**, suggesting the disclosed plan-limit language may involve dynamic throttling, ambiguous quota accounting, or missing context from the internal documents.

  - **[Anthropic really doesn’t seem to value its $20 subscribers anymore](https://www.reddit.com/r/ClaudeAI/comments/1w4tadv/anthropic_really_doesnt_seem_to_value_its_20/)** (Activity: 1343): **The post argues that **Anthropic’s $20/month Claude Pro** tier is losing value if flagship Claude models or major upgrades are reserved for higher-priced plans, even if cheaper users receive generous access to weaker models. The author says they would prefer **strict rate limits**—e.g. `20 messages/day`—on the best Claude model over being excluded entirely, and compares this unfavorably with the expectation that **OpenAI Plus** may continue offering $20 users access to major new flagship capabilities via [ChatGPT Plus](https://openai.com/chatgpt/pricing/) while Anthropic segments access through [Claude plans](https://www.anthropic.com/pricing).** Top comments are broadly cynical rather than technical: users claim the $20 tier is mainly a conversion funnel, that subscribers function as training/evaluation buffers, and that Anthropic prioritizes enterprise or high-spend customers over individual Pro users.




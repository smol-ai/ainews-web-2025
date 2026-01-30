---
id: MjAyNi0w
title: >-
  xAI Grok Imagine API - the #1 Video Model, Best Pricing and Latency - and
  merging with SpaceX
date: '2026-01-29T05:44:39.731046Z'
description: >-
  **Google DeepMind** launched **Project Genie (Genie 3 + Nano Banana Pro +
  Gemini)**, a prototype for creating interactive, real-time generated worlds
  from text or image prompts, currently available to **Google AI Ultra
  subscribers in the U.S. (18+)** with noted limitations like **~60s generation
  limits** and imperfect physics. In parallel, the open-source **LingBot-World**
  offers a real-time interactive world model with **<1s latency at 16 FPS** and
  minute-level coherence, emphasizing interactivity and causal consistency. In
  video generation, **xAI Grok Imagine** debuted strongly with native audio
  support, **15s duration**, and competitive pricing at **$4.20/min including
  audio**, while **Runway Gen-4.5** focuses on animation workflows with new
  features like **Motion Sketch** and **Character Swap**. The 3D generation
  space sees **fal** adding **Hunyuan 3D 3.1 Pro/Rapid** to its API offerings,
  extending model-as-a-service workflows into 3D pipelines.
companies:
  - google-deepmind
  - x-ai
  - runway
  - fal
models:
  - genie-3
  - nano-banana-pro
  - gemini
  - lingbot-world
  - grok-imagine
  - runway-gen-4.5
  - hunyuan-3d-3.1-pro
topics:
  - interactive-simulation
  - real-time-generation
  - promptability
  - character-customization
  - world-models
  - open-source
  - video-generation
  - audio-generation
  - animation-workflows
  - model-as-a-service
  - 3d-generation
  - latency
  - coherence
people:
  - demishassabis
  - sundarpichai
---


**TODO: ONELINE SUBTITLE**

> AI News for 1/28/2026-1/29/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**253** channels, and **7278** messages) for you. Estimated reading time saved (at 200wpm): **605 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!




---

# AI Twitter Recap

**World Models & Interactive Simulation: Google DeepMind’s Project Genie (Genie 3) vs. Open-Source “World Simulators”**

- **Project Genie rollout (Genie 3 + Nano Banana Pro + Gemini)**: Google/DeepMind launched **Project Genie**, a prototype that lets users create and explore **interactive, real-time generated worlds** from **text or image prompts**, with remixing and a gallery. Availability is currently gated to **Google AI Ultra subscribers in the U.S. (18+)**, and the product is explicit about prototype limitations (e.g., **~60s generation limits**, control latency, imperfect physics adherence) ([DeepMind announcement](https://twitter.com/GoogleDeepMind/status/2016919756440240479), [how it works](https://twitter.com/GoogleDeepMind/status/2016919762924949631), [rollout details](https://twitter.com/GoogleDeepMind/status/2016919765713826171), [Demis](https://twitter.com/demishassabis/status/2016925155277361423), [Sundar](https://twitter.com/sundarpichai/status/2016979481832067264), [Google thread](https://twitter.com/Google/status/2016926928478089623), [Google limitations](https://twitter.com/Google/status/2016972686208225578)). Early-access testers highlight promptability, character/world customization, and “remixing” as key UX hooks ([venturetwins](https://twitter.com/venturetwins/status/2016919922727850333), [Josh Woodward demo thread](https://twitter.com/joshwoodward/status/2016921839038255210)).
- **Open-source push: LingBot-World**: A parallel thread frames **world models** as distinct from “video dreamers,” arguing for **interactivity, object permanence, and causal consistency**. LingBot-World is repeatedly described as an **open-source real-time interactive world model** built on **Wan2.2** with **<1s latency at 16 FPS** and **minute-level coherence** (claims include VBench improvements and landmark persistence after long occlusion) ([paper-summary thread](https://twitter.com/dair_ai/status/2016881546909929775), [HuggingPapers mention](https://twitter.com/HuggingPapers/status/2016787043028746284), [reaction clip](https://twitter.com/kimmonismus/status/2016896151610442192)). The meta-narrative: proprietary systems (Genie) are shipping consumer prototypes while open systems race to close capability gaps on **coherence + control**.

**Video Generation & Creative Tooling: xAI Grok Imagine, Runway Gen-4.5, and fal’s “Day-0” Platforms**

- **xAI Grok Imagine (video + audio) lands near/at the top of leaderboards**: Multiple sources report Grok Imagine’s strong debut in video rankings and emphasize **native audio**, **15s duration**, and aggressive **pricing ($4.20/min including audio)** relative to Veo/Sora ([Arena launch ranking](https://twitter.com/arena/status/2016748418635616440), [Artificial Analysis #1 claim + pricing context](https://twitter.com/ArtificialAnlys/status/2016749756081721561), [follow-up #1 I2V leaderboard](https://twitter.com/ArtificialAnlys/status/2016749790907027726), [xAI team announcement](https://twitter.com/EthanHe_42/status/2016749123198673099), [Elon](https://twitter.com/elonmusk/status/2016768088855769236)). fal positioned itself as **day-0 platform partner** with API endpoints for text-to-image, editing, text-to-video, image-to-video, and video editing ([fal partnership](https://twitter.com/fal/status/2016746472931283366), [fal links tweet](https://twitter.com/fal/status/2016746473887609118)).
- **Runway Gen-4.5 shifts toward “animation engine” workflows**: Creators describe Gen-4.5 as increasingly controllable for animation-style work ([c_valenzuelab](https://twitter.com/c_valenzuelab/status/2016721443430510847)). Runway shipped **Motion Sketch** (annotate camera/motion on a start frame) and **Character Swap** as built-in apps—more evidence that vendors are packaging controllability primitives rather than only pushing base quality ([feature thread](https://twitter.com/jerrod_lew/status/2016816309762486423)). Runway also markets “photo → story clip” flows as a mainstream onramp ([Runway example](https://twitter.com/runwayml/status/2016882344427147275)).
- **3D generation joins the same API distribution layer**: fal also added **Hunyuan 3D 3.1 Pro/Rapid** (text/image-to-3D, topology/part generation), showing the same “model-as-a-service + workflow endpoints” pattern spreading from image/video into 3D pipelines ([fal drop](https://twitter.com/fal/status/2016877742298411089)).

**Open Models & Benchmarks: Kimi K2.5 momentum, Qwen3-ASR release, and Trinity Large architecture details**

- **Kimi K2.5 as the “#1 open model” across multiple eval surfaces**: Moonshot promoted K2.5’s rank on **VoxelBench** ([Moonshot](https://twitter.com/Kimi_Moonshot/status/2016732248800997727)) and later Kimi updates focus on productization: **Kimi Code now powered by K2.5**, switching from request limits to **token-based billing**, plus a limited-time **3× quota/no throttling** event ([Kimi Code billing update](https://twitter.com/Kimi_Moonshot/status/2016918447951925300), [billing rationale](https://twitter.com/Kimi_Moonshot/status/2016918450992812443)). Arena messaging amplifies K2.5 as a leading open model with forthcoming Code Arena scores ([Arena deep dive](https://twitter.com/arena/status/2016915717539713236), [Code Arena prompt](https://twitter.com/arena/status/2016923733513105705)); Arena also claims **Kimi K2.5 Thinking** as **#1 open model in Vision Arena** and the only open model in the top 15 ([Vision Arena claim](https://twitter.com/arena/status/2016984335380001268)). Commentary frames K2.5 as “V3-generation architecture pushed with more continued training,” with next-gen competition expected from K3/GLM-5 etc. ([teortaxes](https://twitter.com/teortaxesTex/status/2016956019239272717)).
- **Alibaba Qwen3-ASR: production-grade open speech stack with vLLM day-0 support**: Qwen released **Qwen3-ASR + Qwen3-ForcedAligner** emphasizing messy real-world audio, **52 languages/dialects**, long audio (up to **20 minutes/pass**), and timestamps; models are **Apache 2.0** and include an open inference/finetuning stack. vLLM immediately announced **day-0 support** and performance notes (e.g., “2000× throughput on 0.6B” in their tweet) ([Qwen release](https://twitter.com/Alibaba_Qwen/status/2016858705917075645), [ForcedAligner](https://twitter.com/Alibaba_Qwen/status/2016859224077455413), [vLLM support](https://twitter.com/vllm_project/status/2016865238323515412), [Adina Yakup summary](https://twitter.com/AdinaYakup/status/2016865634559152162), [native streaming claim](https://twitter.com/Alibaba_Qwen/status/2016900512478875991), [Qwen thanks vLLM](https://twitter.com/Alibaba_Qwen/status/2016905051395260838)). Net: open-source speech is increasingly “full-stack,” not just weights.
- **Arcee AI Trinity Large (400B MoE) enters the architecture discourse**: Multiple threads summarize Trinity Large as **400B MoE with ~13B active**, tuned for throughput via sparse expert selection, and featuring a grab bag of modern stability/throughput techniques (router tricks, load balancing, attention patterns, normalization variants). Sebastian Raschka’s architecture recap is the most concrete single reference point ([rasbt](https://twitter.com/rasbt/status/2016903019116249205)); additional MoE/router stability notes appear in a separate technical summary ([cwolferesearch](https://twitter.com/cwolferesearch/status/2016792505111457883)). Arcee notes multiple variants trending on Hugging Face ([arcee_ai](https://twitter.com/arcee_ai/status/2016986617584529642)).

**Agents in Practice: “Agentic Engineering,” Multi-Agent Coordination, and Enterprise Sandboxes**

- **From vibe coding to agentic engineering**: A high-engagement meme-like anchor tweet argues for “Agentic Engineering > Vibe Coding” and frames professionalism around repeatable workflows rather than vibes ([bekacru](https://twitter.com/bekacru/status/2016738191341240830)). Several threads reinforce the same theme operationally: context prep, evaluations, and sandboxing as the hard parts.
- **Primer: repo instructions + lightweight evals + PR automation**: Primer proposes a workflow for “AI-enabling” repos: agentic repo introspection → generate an instruction file → run a **with/without** eval harness → scale via batch PRs across org repos ([Primer launch](https://twitter.com/pierceboggan/status/2016732251535397158), [local run](https://twitter.com/pierceboggan/status/2016733056237711849), [eval framework](https://twitter.com/pierceboggan/status/2016733232176193539), [org scaling](https://twitter.com/pierceboggan/status/2016733666022424957)).
- **Agent sandboxes + traceability as infra primitives**: Multiple tweets point to “agent sandboxes” (isolated execution environments) as an emerging January trend ([dejavucoder](https://twitter.com/dejavucoder/status/2016979866651152898)). Cursor proposed an **open standard** to trace agent conversations to generated code, explicitly positioning it as interoperable across agents/interfaces ([Cursor](https://twitter.com/cursor_ai/status/2016934752188576029)). This pairs with broader ecosystem pressure: agents need auditability and reliable grounding when they can take actions.
- **Multi-agent coordination beats “bigger brain” framing**: A popular summary claims a system that uses a **controller trained by RL** to route between large/small models can beat a single large agent on HLE with lower cost/latency—reinforcing that orchestration policies are becoming first-class artifacts ([LiorOnAI](https://twitter.com/LiorOnAI/status/2016904429543272579)). In the same direction, an Amazon “Insight Agents” paper summary argues for pragmatic manager-worker designs with lightweight OOD detection and routing (autoencoder + fine-tuned BERT) instead of LLM-only classifiers for latency/precision reasons ([omarsar0](https://twitter.com/omarsar0/status/2016880021030522997m)).
- **Kimi’s “Agent Swarm” philosophy**: A long-form repost from ZhihuFrontier describes K2.5’s agent mode as a response to “text-only helpfulness” and tool-call hallucinations, emphasizing **planning→execution bridging**, dynamic tool-based context, and **multi-viewpoint planning via swarms** ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2016811037274886377)).
- **Moltbot/Clawdbot safety trilemma**: Community discussion frames “Useful vs Autonomous vs Safe” as a tri-constraint until prompt injection is solved ([fabianstelzer](https://twitter.com/fabianstelzer/status/2016818595687272913)). Another take argues capability (trust) bottlenecks dominate: users won’t grant high-stakes autonomy (e.g., finance) until agents are reliably competent ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2016937299125424284)).

**Model UX, DevTools, and Serving: Gemini Agentic Vision, OpenAI’s in-house data agent, vLLM fixes, and local LLM apps**

- **Gemini 3 Flash “Agentic Vision”**: Google positions Agentic Vision as a structured image-analysis pipeline: planning steps, zooming, annotating, and optionally running Python for plotting—essentially turning “vision” into an agentic workflow rather than a single forward pass ([GeminiApp intro](https://twitter.com/GeminiApp/status/2016914275886125483), [capabilities](https://twitter.com/GeminiApp/status/2016914637523210684), [rollout note](https://twitter.com/GeminiApp/status/2016914638861193321)).
- **OpenAI’s in-house data agent at massive scale**: OpenAI described an internal “AI data agent” reasoning over **600+ PB** and **70k datasets**, using Codex-powered table knowledge and careful context management ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2016943147239329872)). This is a rare concrete peek at “deep research/data agent” architecture constraints: retrieval + schema/table priors + org context.
- **Serving bugs are still real (vLLM + stateful models)**: AI21 shared a debugging story where scheduler token allocation caused misclassification between **prefill vs decode**, now fixed in **vLLM v0.14.0**—a reminder that infrastructure correctness matters, especially for stateful architectures like Mamba ([AI21Labs thread](https://twitter.com/AI21Labs/status/2016857918436503975)).
- **Local LLM UX continues to improve**: Georgi Gerganov shipped **LlamaBarn**, a tiny macOS menu bar app built on llama.cpp to run local models ([ggerganov](https://twitter.com/ggerganov/status/2016912009544057045)). Separate comments suggest agentic coding performance may improve by disabling “thinking” modes for specific models (GLM-4.7-Flash) via llama.cpp templates ([ggerganov config note](https://twitter.com/ggerganov/status/2016903216093417540)).

**Top tweets (by engagement)**

- **Grok Imagine hype & distribution**: [@elonmusk](https://twitter.com/elonmusk/status/2016768088855769236), [@fal](https://twitter.com/fal/status/2016746472931283366), [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2016749756081721561)
- **DeepMind/Google world models**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2016919756440240479), [@demishassabis](https://twitter.com/demishassabis/status/2016925155277361423), [@sundarpichai](https://twitter.com/sundarpichai/status/2016979481832067264)
- **AI4Science**: [@demishassabis on AlphaGenome](https://twitter.com/demishassabis/status/2016763919646478403)
- **Speech open-source release**: [@Alibaba_Qwen Qwen3-ASR](https://twitter.com/Alibaba_Qwen/status/2016858705917075645)
- **Agents + developer workflow**: [@bekacru “Agentic Engineering > Vibe Coding”](https://twitter.com/bekacru/status/2016738191341240830), [@cursor_ai agent-trace.dev](https://twitter.com/cursor_ai/status/2016934752188576029)
- **Anthropic workplace study**: [@AnthropicAI AI-assisted coding and mastery](https://twitter.com/AnthropicAI/status/2016960382968136138)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Kimi K2.5 Model Discussions and Releases

  - **[AMA With Kimi, The Open-source Frontier Lab Behind Kimi K2.5 Model](https://www.reddit.com/r/LocalLLaMA/comments/1qpewj7/ama_with_kimi_the_opensource_frontier_lab_behind/)** (Activity: 686): ****Kimi** is the research lab behind the open-source **Kimi K2.5** model, engaging in an AMA to discuss their work. The discussion highlights a focus on large-scale models, with inquiries about the development of smaller models like `8B`, `32B`, and `70B` for better intelligence density. There is also interest in smaller Mixture of Experts (MoE) models, such as `~100B` total with `~A3B` active, optimized for local or prosumer use. The team is questioned on their stance regarding the notion that *Scaling Laws have hit a wall*, a topic of current debate in AI research.** Commenters express a desire for smaller, more efficient models, suggesting that these could offer better performance for specific use cases. The debate on scaling laws reflects a broader concern in the AI community about the limits of current model scaling strategies.

    - The discussion around model sizes highlights a preference for smaller models like 8B, 32B, and 70B due to their 'intelligence density.' These sizes are considered optimal for balancing performance and resource efficiency, suggesting a demand for models that can operate effectively on limited hardware while still providing robust capabilities.
    - The inquiry into smaller Mixture of Experts (MoE) models, such as a ~100B total with ~A3B active, indicates interest in models optimized for local or prosumer use. This reflects a trend towards developing models that are not only powerful but also accessible for individual users or small enterprises, emphasizing the need for efficient resource utilization without sacrificing performance.
    - The challenge of maintaining non-coding abilities like creative writing and emotional intelligence in models like Kimi 2.5 is significant, especially as coding benchmarks become more prominent. The team is tasked with ensuring these softer skills do not regress, which involves balancing the training focus between technical and creative capabilities to meet diverse user needs.

  - **[Run Kimi K2.5 Locally](https://www.reddit.com/r/LocalLLaMA/comments/1qpfse6/run_kimi_k25_locally/)** (Activity: 553): **The image provides a guide for running the **Kimi-K2.5** model locally, emphasizing its state-of-the-art (SOTA) performance in vision, coding, agentic, and chat tasks. The model, which is a `1 trillion` parameter hybrid reasoning model, requires `600GB` of disk space, but the quantized **Unsloth Dynamic 1.8-bit** version reduces this requirement to `240GB`, a `60%` reduction. The guide includes instructions for using `llama.cpp` to load models and demonstrates generating HTML code for a simple game. The model is available on [Hugging Face](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) and further documentation can be found on [Unsloth's official site](https://unsloth.ai/docs/models/kimi-k2.5).** Commenters discuss the feasibility of running the model on high-end hardware, with one user questioning its performance on a Strix Halo setup and another highlighting the substantial VRAM requirements, suggesting that only a few users can realistically run it locally.

    - Daniel_H212 is inquiring about the performance of the Kimi K2.5 model on the Strix Halo hardware, specifically asking for the token generation speed in seconds per token. This suggests a focus on benchmarking the model's efficiency on high-end hardware setups.
    - Marksta provides feedback on the quantized version of the Kimi K2.5 model, specifically the Q2_K_XL variant. They note that the model maintains high coherence and adheres strictly to prompts, which is characteristic of Kimi-K2's style. However, they also mention that while the model's creative capabilities have improved, it still struggles with execution in creative scenarios, often delivering logical but poorly written responses.
    - MikeRoz questions the utility of higher quantization levels like Q5 and Q6 (e.g., UD-Q5_K_XL, Q6_K) when most experts prefer int4 quantization. This highlights a debate on the trade-offs between model size, performance, and precision in quantization strategies.

  - **[Kimi K2.5 is the best open model for coding](https://www.reddit.com/r/LocalLLaMA/comments/1qp87tk/kimi_k25_is_the_best_open_model_for_coding/)** (Activity: 1119): **The image highlights **Kimi K2.5** as the leading open model for coding on the LMARENA.AI leaderboard, ranked `#7` overall. This model is noted for its superior performance in coding tasks compared to other open models. The leaderboard provides a comparative analysis of various AI models, showcasing their ranks, scores, and confidence intervals, emphasizing Kimi K2.5's achievements in the coding domain.** One commenter compared Kimi K2.5's performance to other models, noting it is on par with Sonnet 4.5 in accuracy but not as advanced as Opus in agentic function. Another comment criticized LMArena for not reflecting a model's multi-turn or long context capabilities.

    - A user compared Kimi K2.5 to other models, noting that it performs on par with Sonnet 4.5 in terms of accuracy for React projects, but not at the level of Opus in terms of agentic function. They also mentioned that Kimi 2.5 surpasses GLM 4.7, which was their previous choice, and expressed curiosity about the upcoming GLM-5 from [z.ai](http://z.ai).
    - Another commenter criticized LMArena, stating that it fails to provide insights into a model's multi-turn, long context, or agentic capabilities, implying that such benchmarks are insufficient for evaluating comprehensive model performance.
    - A user highlighted the cost-effectiveness of Kimi K2.5, stating it feels as competent as Opus 4.5 while being significantly cheaper, approximately 1/5th the cost, and even less expensive than Haiku. This suggests a strong performance-to-cost ratio for Kimi K2.5.

  - **[Finally We have the best agentic AI at home](https://www.reddit.com/r/LocalLLM/comments/1qp880l/finally_we_have_the_best_agentic_ai_at_home/)** (Activity: 464): **The image is a performance comparison chart of various AI models, including **Kimi K2.5**, **GPT-5.2 (xhigh)**, **Claude Opus 4.5**, and **Gemini 3 Pro**. **Kimi K2.5** is highlighted as the top-performing model across multiple categories such as agents, coding, image, and video tasks, indicating its superior capabilities in multimodal applications. The post suggests excitement about integrating this model with a 'clawdbot', hinting at potential applications in robotics or automation.** A comment humorously suggests that hosting the **Kimi 2.5 1T+ model** at home implies having a large home, indicating the model's likely high computational requirements. Another comment sarcastically mentions handling it with a 16GB VRAM card, implying skepticism about the feasibility of running such a model on typical consumer hardware.



### 2. Open Source Model Innovations

  - **[LingBot-World outperforms Genie 3 in dynamic simulation and is fully Open Source](https://www.reddit.com/r/LocalLLaMA/comments/1qqj51h/lingbotworld_outperforms_genie_3_in_dynamic/)** (Activity: 230): **The open-source framework **LingBot-World** surpasses the proprietary **Genie 3** in dynamic simulation capabilities, achieving `16 FPS` and maintaining object consistency for `60 seconds` outside the field of view. This model, available on [Hugging Face](https://huggingface.co/collections/robbyant/lingbot-world), offers enhanced handling of complex physics and scene transitions, challenging the monopoly of proprietary systems by providing full access to its code and model weights.** Commenters question the hardware requirements for running LingBot-World and express skepticism about the comparison with Genie 3, suggesting a lack of empirical evidence or direct access to Genie 3 for a fair comparison.

    - A user questioned the hardware requirements for running LingBot-World, highlighting the importance of specifying computational needs for practical implementation. This is crucial for users to understand the feasibility of deploying the model in various environments.
    - Another commenter raised concerns about the lack of a direct comparison with Genie 3, suggesting that without empirical data or benchmarks, claims of LingBot-World's superiority might be unsubstantiated. This points to the need for transparent and rigorous benchmarking to validate performance claims.
    - A suggestion was made to integrate a smaller version of LingBot-World into a global illumination stack, indicating potential applications in graphics and rendering. This could leverage the model's capabilities in dynamic simulation to enhance visual computing tasks.

  - **[API pricing is in freefall. What's the actual case for running local now beyond privacy?](https://www.reddit.com/r/LocalLLaMA/comments/1qp6rm5/api_pricing_is_in_freefall_whats_the_actual_case/)** (Activity: 1053): **The post discusses the rapidly decreasing costs of API access for AI models, with examples like **K2.5** offering prices at `10%` of **Opus** and **Deepseek** being nearly free. **Gemini** also provides a substantial free tier. This trend is contrasted with the challenges of running large models locally, such as the need for expensive GPUs or dealing with quantization tradeoffs, which result in slow processing speeds (`15 tok/s`) on consumer hardware. The author questions the viability of local setups given these API pricing trends, noting that while privacy and latency control are valid reasons, the cost-effectiveness of local setups is diminishing.** Commenters highlight concerns about the sustainability of low API prices, suggesting they may rise once market dominance is achieved, similar to past trends in other industries. Others emphasize the importance of offline capabilities and the ability to audit and trust local models, which ensures consistent behavior without unexpected changes from vendors.

    - Minimum-Vanilla949 highlights the importance of offline capabilities for those who travel frequently, emphasizing the risk of API companies altering terms of service or increasing prices once they dominate the market. This underscores the value of local models for ensuring consistent access and cost control.
    - 05032-MendicantBias discusses the unsustainable nature of current API pricing, which is often subsidized by venture capital. They argue that once a monopoly is achieved, prices will likely increase, making local setups and open-source tools a strategic defense against such business models.
    - IactaAleaEst2021 points out the importance of repeatability and trust in using local models. By downloading and auditing a model, users can ensure consistent behavior, unlike APIs where vendors might change model behavior over time, potentially reducing its utility for specific tasks.


### 3. Trends in AI Agent Frameworks

  - **[GitHub trending this week: half the repos are agent frameworks. 90% will be dead in 1 week.](https://www.reddit.com/r/LocalLLaMA/comments/1qq6n3t/github_trending_this_week_half_the_repos_are/)** (Activity: 538): **The image highlights a trend on GitHub where many of the trending repositories are related to AI agent frameworks, suggesting a surge in interest in these tools. However, the post's title and comments express skepticism about the sustainability of this trend, comparing it to the rapid rise and fall of JavaScript frameworks. The repositories are mostly written in Python and include a mix of agent frameworks, RAG tooling, and model-related projects like NanoGPT and Grok. The discussion reflects a concern that many of these projects may not maintain their popularity or relevance over time.** One comment challenges the claim that half of the trending repositories are agent frameworks, noting that only one is an agent framework by Microsoft, while others are related to RAG tooling and model development. Another comment appreciates the utility of certain projects, like IPTV, for educational purposes.

    - gscjj points out that the claim about 'half the repos being agent frameworks' is inaccurate. They note that the list includes a variety of projects such as Microsoft's agent framework, RAG tooling, and models like NanoGPT and Grok, as well as a model CLI for code named Kimi and a browser API. This suggests a diverse range of trending repositories rather than a dominance of agent frameworks.

  - **[Mistral CEO Arthur Mensch: “If you treat intelligence as electricity, then you just want to make sure that your access to intelligence cannot be throttled.”](https://www.reddit.com/r/LocalLLaMA/comments/1qqhhtx/mistral_ceo_arthur_mensch_if_you_treat/)** (Activity: 357): ****Arthur Mensch**, CEO of **Mistral**, advocates for open-weight models, likening intelligence to electricity, emphasizing the importance of unrestricted access to AI capabilities. This approach supports the deployment of models on local devices, reducing costs as models are quantized for lower compute environments, contrasting with closed models that are often large and monetized through paywalls. Mistral aims to balance corporate interests with open access, potentially leading to significant breakthroughs in AI deployment.** Commenters appreciate Mistral's approach to open models, noting the potential for reduced costs and increased accessibility. There is a consensus that open models could democratize AI usage, contrasting with the restrictive nature of closed models.

    - RoyalCities highlights the cost dynamics of model deployment, noting that open models, especially when quantized, reduce costs as they can be run on local devices. This contrasts with closed models that are often large and require significant infrastructure, thus being monetized through paywalls. This reflects a broader industry trend where open models aim to democratize access by lowering hardware requirements.
    - HugoCortell points out the hardware bottleneck in deploying open models effectively. While open-source models can rival closed-source ones in performance, the lack of affordable, high-performance hardware limits their accessibility. This is compounded by large companies making high-quality local hardware increasingly expensive, suggesting a need for a company capable of producing and distributing its own hardware to truly democratize AI access.
    - tarruda expresses anticipation for the next open Mistral model, specifically the "8x22". This indicates a community interest in the technical specifications and potential performance improvements of upcoming models, reflecting the importance of open model development in advancing AI capabilities.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. OpenAI and AGI Investments

  - **[Nearly half of the Mag 7 are reportedly betting big on OpenAI’s path to AGI](https://www.reddit.com/r/singularity/comments/1qpxyka/nearly_half_of_the_mag_7_are_reportedly_betting/)** (Activity: 1153): ****NVIDIA, Microsoft, and Amazon** are reportedly in discussions to invest a combined total of up to `$60 billion` into **OpenAI**, with **SoftBank** considering an additional `$30 billion`. This potential investment could value OpenAI at approximately `$730 billion` pre-money, aligning with recent valuation discussions in the `$750 billion to $850 billion+` range. This would mark one of the largest private capital raises in history, highlighting the significant financial commitment from major tech companies towards the development of artificial general intelligence (AGI).** Commenters note the strategic alignment of these investments, with one pointing out that companies like Microsoft and NVIDIA are unlikely to invest in competitors like Google. Another comment reflects on the evolving landscape of large language models (LLMs) and the shifting focus of tech giants.

    - CoolStructure6012 highlights the strategic alignment between **Microsoft (MSFT)** and **NVIDIA (NVDA)** with OpenAI, suggesting that their investments are logical given their competitive stance against **Google**. This reflects the broader industry trend where tech giants are aligning with AI leaders to bolster their AI capabilities and market positions.
    - drewc717 reflects on the evolution of AI models, noting a significant productivity boost with OpenAI's `4.1 Pro mode`. However, they express a decline in their workflow efficiency after switching to **Gemini**, indicating that not all LLMs provide the same level of user experience or productivity, which is crucial for developers relying on these tools.
    - EmbarrassedRing7806 questions the lack of attention on **Anthropic** despite its widespread use in coding through its **Claude** model, as opposed to OpenAI's **Codex**. This suggests a potential underestimation of Anthropic's impact in the AI coding space, where **Claude** might be offering competitive or superior capabilities.


### 2. DeepMind's AlphaGenome Launch

  - **[Google DeepMind launches AlphaGenome, an AI model that analyzes up to 1 million DNA bases to predict genomic regulation](https://www.reddit.com/r/singularity/comments/1qphlfg/google_deepmind_launches_alphagenome_an_ai_model/)** (Activity: 427): ****Google DeepMind** has introduced **AlphaGenome**, a sequence model capable of analyzing up to `1 million DNA bases` to predict genomic regulation, as detailed in [Nature](https://www.nature.com/articles/s41586-025-10014-0?amp%3Butm_medium=social&amp%3Butm_campaign=&amp%3Butm_content=). The model excels in predicting genomic signals such as gene expression and chromatin structure, particularly in non-coding DNA, which is crucial for understanding disease-associated variants. AlphaGenome outperforms existing models on `25 of 26` benchmark tasks and is available for research use, with its model and weights accessible on [GitHub](https://github.com/google-deepmind/alphagenome_research).** Commenters highlight the model's potential impact on genomics, with some humorously suggesting its significance in advancing scientific achievements akin to winning Nobel prizes.


  - **[[R] AlphaGenome: DeepMind's unified DNA sequence model predicts regulatory variant effects across 11 modalities at single-bp resolution (Nature 2026)](https://www.reddit.com/r/MachineLearning/comments/1qq4lnc/r_alphagenome_deepminds_unified_dna_sequence/)** (Activity: 66): ****DeepMind's AlphaGenome** introduces a unified DNA sequence model that predicts regulatory variant effects across `11 modalities` at single-base-pair resolution. The model processes `1M base pairs` of DNA to predict thousands of functional genomic tracks, matching or exceeding specialized models in `25 of 26` evaluations. It employs a U-Net backbone with CNN and transformer layers, trained on human and mouse genomes, and captures `99%` of validated enhancer-gene pairs within a `1Mb` context. Training on TPUv3 took `4 hours`, with inference under `1 second` on H100. The model demonstrates cross-modal variant interpretation, notably on the TAL1 oncogene in T-ALL. [Nature](https://www.nature.com/articles/s41586-025-10014-0), [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1), [DeepMind blog](https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome), [GitHub](https://github.com/google-deepmind/alphagenome).** Some commenters view the model as an incremental improvement over existing sequence models, questioning the novelty despite its publication in *Nature*. Others express concerns about the implications of open-sourcing such powerful genomic tools, hinting at potential future applications like 'text to CRISPR' models.

    - st8ic88 argues that while DeepMind's AlphaGenome model is notable for its ability to predict regulatory variant effects across 11 modalities at single-base pair resolution, it is seen as an incremental improvement over existing sequence models predicting genomic tracks. The comment suggests that the model's prominence is partly due to DeepMind's reputation and branding, particularly the use of 'Alpha' in its name, which may have contributed to its publication in Nature.
    - --MCMC-- is interested in the differences between the AlphaGenome model's preprint and its final published version in Nature. The commenter had read the preprint and is curious about any changes made during the peer review process, which could include updates to the model's methodology, results, or interpretations.
    - f0urtyfive raises concerns about the potential risks of open-sourcing powerful genomic models like AlphaGenome, speculating on future developments such as 'text to CRISPR' models. This comment highlights the ethical and safety considerations of making advanced genomic prediction tools widely accessible, which could lead to unintended applications or misuse.


### 3. Claude's Cost Efficiency and Usage Strategies

  - **[Claude Subscriptions are up to 36x cheaper than API (and why "Max 5x" is the real sweet spot)](https://www.reddit.com/r/ClaudeAI/comments/1qpcj8q/claude_subscriptions_are_up_to_36x_cheaper_than/)** (Activity: 665): **A data analyst has reverse-engineered **Claude's internal usage limits** by analyzing unrounded floats in the web interface, revealing that **subscriptions can be up to 36x cheaper** than using the API, especially for coding tasks with agents like Claude Code. The analysis shows that the **subscription model offers free cache reads**, whereas the API charges 10% of the input cost for each read, making the subscription significantly more cost-effective for long sessions. The "Max 5x" plan at `$100/month` is highlighted as the most optimized, offering a `6x` higher session limit and `8.3x` higher weekly limit than the Pro plan, contrary to the marketed "5x" and "20x" plans. The findings were derived using the Stern-Brocot tree to decode precise usage percentages into internal credit numbers. Full details and formulas are available [here](http://she-llac.com/claude-limits).** Commenters express concern over **Anthropic's lack of transparency** and speculate that the company might change the limits once they realize users have reverse-engineered them. Some users are taking advantage of the current subscription benefits, anticipating potential changes.

    - HikariWS raises a critical point about **Anthropic's lack of transparency** regarding their subscription limits, which could change unexpectedly, rendering current analyses obsolete. This unpredictability poses a risk for developers relying on these plans for cost-effective usage.
    - Isaenkodmitry discusses the potential for **Anthropic to close loopholes** once they realize users are exploiting subscription plans for cheaper access compared to the API. This highlights a strategic risk for developers who are currently benefiting from these plans, suggesting they should maximize usage while it lasts.
    - Snow30303 mentions using **Claude code in VS Code for Flutter apps**, noting that it consumes credits rapidly. This suggests a need for more efficient usage strategies or alternative solutions to manage costs effectively when integrating Claude into development workflows.

  - **[We reduced Claude API costs by 94.5% using a file tiering system (with proof)](https://www.reddit.com/r/ClaudeAI/comments/1qp9ve9/we_reduced_claude_api_costs_by_945_using_a_file/)** (Activity: 603): **The post describes a file tiering system that reduces **Claude API costs by 94.5%** by categorizing files into HOT, WARM, and COLD tiers, thus minimizing the number of tokens processed per session. This system, implemented in a tool called `cortex-tms`, tags files based on their relevance and usage frequency, allowing only the most necessary files to be loaded by default. The approach has been validated through a case study on the author's project, showing a reduction from `66,834` to `3,647` tokens per session, significantly lowering costs from `$0.11` to `$0.01` per session with Claude Sonnet 4.5. The tool is open-source and available on [GitHub](https://github.com/cortex-tms/cortex-tms).** One commenter inquired about the manual process of tagging files and updating tags, suggesting the use of git history to automate file heat determination. Another user appreciated the approach due to their own struggles with managing API credits.

    - **Illustrious-Report96** suggests using `git history` to determine file 'heat', which involves analyzing the frequency and recency of changes to classify files as 'hot', 'warm', or 'cold'. This method leverages version control metadata to automate the classification process, potentially reducing manual tagging efforts.
    - **Accomplished_Buy9342** inquires about restricting access to 'WARM' and 'COLD' files, which implies a need for a mechanism to control agent access based on file tier. This could involve implementing access controls or modifying the agent's logic to prioritize 'HOT' files, ensuring efficient resource usage.
    - **durable-racoon** asks about the process of tagging files and updating these tags, highlighting the importance of an automated or semi-automated system to manage file tiering efficiently. This could involve scripts or tools that dynamically update file tags based on usage patterns or other criteria.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 3.0 Pro Preview Nov-18

**Theme 1. Model Wars: Kimi’s Rise, Recursive Agents, and Geometric Architectures**

- **Kimi K2.5 crushes the Vision Arena**: The community reports **Kimi K2.5** is dominating the leaderboards, claiming the **#1 open model** spot and ranking **#6 overall** on the [Vision Arena leaderboard](https://arena.ai/leaderboard/vision). Users note it outperforms **Claude** in specific vision tasks and now features a dedicated **computer use** model that handles phone screenshots (though it throws 403 errors on mobile uploads).
- **Recursive Language Models trigger semantic debates**: A heated discussion erupted over the term "**Recursive Language Models**" (**RLM**), with critics arguing it simply rebrands **tool-calling loops**, while proponents point to the new [RLM-Qwen3-8B](https://xcancel.com/a1zhang/status/2016923294461476873?s=46&t=jDrfS5vZD4MFwckU5E8f5Q) as the first natively recursive model. This small-scale model, post-trained on just **1,000 trajectories**, reportedly beats scaffolded RLM versions in long-context tasks.
- **Geometric Convolution attempts to dethrone Attention**: Researchers are experimenting with a baseline that replaces standard **Multi-Head Attention** with a [geometric convolution approach](https://github.com/MrPan2048/GeometricTransformer), using embeddings as cell connections. Early debug prints show loss convergence capturing dialogue logic, positioning this as a potential alternative to heavy transformer compute.

**Theme 2. Hardware Hustle: Microsoft’s Silicon, Unsloth Speeds, and Apple’s Hidden Power**

- **Microsoft aims at NVIDIA with Maia 200**: Microsoft unveiled the [**Maia 200 AI Accelerator**](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/), an inference-focused chip boasting **216GB** of memory and **10k TFLOPS** in FP4 performance. Engineers debated the reliance on **TSMC** manufacturing and compared its architecture favorably against **NVIDIA's Vera Rubin** for large-scale inference workloads.
- **RTX 5090 shreds training benchmarks**: Unsloth users report the **RTX 5090** achieves blistering training speeds of up to **18k tokens per second**, though **12-15k t/s** is safer with a sequence length under **4096**. Optimal throughput requires carefully balancing **batch size** and sequence length to avoid memory bottlenecks during fine-tuning.
- **Apple’s ANE punches above its weight**: A new discussion around [this paper](https://arxiv.org/abs/2511.13450) highlights that Apple's **Neural Engine (ANE)** delivers **3.8 TFlops** on the M4-Pro, nearly matching the GPU's **4.7 TFlops** for GEMM operations. The ANE prioritizes **performance-per-watt**, making it a surprisingly viable target for efficient local inference.

**Theme 3. Dev Tools & Standards: Cursor Pains, MCP Security, and Parallel Studio**

- **Cursor’s "Plan Mode" annoys the power users**: The latest **Cursor** update introduced a **plan mode** that users are actively trying to disable or automate, citing wasted time and unnecessary inputs. Fresh installs of the IDE are reportedly the most unstable configuration, driving users to seek workarounds for the "Plan Mode" friction.
- **MCP gets a hardened Security Standard**: Dani (cr0hn) drafted an open [MCP Security Standard](https://github.com/mcp-security-standard/mcp-server-security-standard) covering hardening, logging, and access control, intending to donate it to the **Agentic AI Foundation**. Simultaneously, the protocol is evolving with **Namespaces** being rejected in favor of **Groups**, detailed in the new [Primitive Grouping SEP-2084](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2084).
- **LM Studio 0.4 hides power tools behind Dev Mode**: The release of **LM Studio 0.4.0** tucks critical settings like sampling and hardware configs behind a **Dev Mode** toggle (`Ctrl+Shift+R`), while introducing **parallel requests**. Users can now load models across different GPUs to handle up to **4 parallel requests** by default, though the software still relies on the older **ROCm 6.4.1**.

**Theme 4. Jailbreaks & Exploits: Keygens, "Remember" Hacks, and Malware Classifiers**

- **Gemini 3 Pro tricked into writing KeyGens**: A user successfully prompted **Gemini 3 Pro** to reverse engineer software and generate a working keygen by pasting code directly from **Ghidra**. While some dismissed this as "script kiddie" behavior, it highlights the model's susceptibility to **context-based exploits** when fed technical disassemblies.
- **"Remember:" command acts as behavior injection**: Red teamers discovered that the [**Gemini** command 'Remember:'](https://gemini.google.com/saved-info) instantly forces subsequent text into the model's saved memory, heavily influencing future behavior. This allows for persistent prompt injections that dictate turns one at a time, bypassing standard session resets.
- **Adversarial Malware Classification struggles**: Engineers are fighting to lower the **False Positive Rate (FPR)** in malware classification models using a dataset of **600K** rows and **9,600** binary features. Despite using neural networks and **explainable models** like scikit-learn trees, reducing FPR below **9%** remains a significant hurdle without sacrificing model interpretability.

**Theme 5. Real-World Agents: Kitchen Robots, World Models, and Bio-AI**

- **Figure.Ai’s Helix 02 conquers the kitchen**: A video surfaced of **Figure.Ai's Helix 02** robot autonomously performing complex kitchen tasks, which a user verified by feeding the video into **Kimi** for a [98% accurate analysis](https://cdn.discordapp.com/attachments/1371757564005711973/1466193526009106452/m2-res_1280p.mp4?ex=697d2c21&is=697bdaa1&hm=427bc85209f62b3f47f60ce804f74a7cc41be60c452fb561197ad468c29e5224&). This aligns with reports of **Matic** raising **$60M** to build a utility-focused consumer robot successor to the Roomba.
- **Google releases "Genie" World Model**: Google launched [**Project Genie**](https://x.com/googleai/status/2016929427784122627) for **AI Ultra** subscribers, a general-purpose world model capable of generating interactive environments from text prompts. This release moves world models from research papers into a deployable product for simulating dynamic scenarios.
- **AI decodes DNA and Alzheimer’s**: Google AI launched [**AlphaGenome**](https://x.com/GoogleAI/status/1937895472305152387) to predict the impact of DNA variants and mutations, while **Goodfire AI** announced new [Alzheimer's biomarkers](https://xcancel.com/goodfireai/status/2016563911508840623) discovered via model interpretability. These advances signal a shift toward using **transparent AI models** to drive breakthroughs in digital biology.

---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro Cracks Software KeyGen Style**: A member reported using **Gemini 3 Pro** to create a working keygen from software by pasting code from **Ghidra**.
   - Skeptical members referred to this as *script kiddie* behavior and suggested trying a reverse engineering CTF challenge.
- **AI Gets Weaponized for Reverse Engineering**: A member shared his work on weaponizing **AI** for *mass reverse engineering, malware analysis, and jailbreak development*.
   - Another member questioned this claim, suggesting the original poster may be more skilled at jailbreaking than malware creation.
- **Sonnet 4.5 Bests Opus with Kaelia Jailbreak**: Members confirmed that **Sonnet 4.5 jailbreaks** work on **Opus**, sharing a **Miss Kaelia jailbreak** based on **ENI Lime** by Vichaps from [this document](https://docs.google.com/document/d/1aZ91O6LtXyO9DGaWxeJYgKhZhlvbee6jh7_RGTq3mXw/edit?usp=sharing).
   - The jailbreak may not be as effective as other models, depending on the prompting strategy used.
- **Gemini's 'Remember:' Command Triggers Behavior**: A member explained that in **Gemini**, the [command 'Remember:'](https://gemini.google.com/saved-info) automatically adds subsequent words to its saved info, influencing its behavior.
   - Each turn is clearly dictated, one at a time, directly in the chat interface.
- **NSFW Nano Banana Jailbreak arrives for Kimi 2.5**: A member shared an NSFW jailbreak for **Kimi 2.5**, dubbed the nano banana jailbreak. The [system prompt](paste-the-prompt-here) frames **Kimi** as an AI assistant from Moonshot AI, permitting NSFW content.
   - The narrative flow proceeds seamlessly without interruption.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GLM 4.7 Slowdown Solved by CUDA**: Users resolved slow speeds with **GLM 4.7 Flash** on NVIDIA Jetson by ensuring proper **CUDA compilation**, boosting performance from **3 tps** to potentially **70-80 t/s** with `-kvu` and `-fa on` flags.
   - Performance discrepancies were observed with **OpenCode**, with one user experiencing slowdowns after opening the model, while another noted that **GLM 4.7** is a better uncensored coder model than **qwen coder** below 32b, but **Qwen Coder** excels at reasoning.
- **LongCat Leaps onto HuggingFace!**: Meituan's new **n-gram model**, the [LongCat model](https://huggingface.com/meituan-longcat/LongCat-Flash-Lite), made its debut on **Hugging Face**, sparking jokes about the proliferation of *Flash* in model names.
   - Community members speculated that *next model Flash-Flash-1b* while celebrating new releases.
- **Microsoft's Maia 200 Challenges NVIDIA**: Microsoft unveiled the [**Maia 200 AI Accelerator**](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/), a chip built for inference, boasting **216GB** memory and **10k TFLOPS** in FP4 performance.
   - The community discussed the chip's manufacturing by **TSMC** and compared it to **NVIDIA's Vera Rubin** architecture, with some raising concerns about relying on Chinese hardware.
- **Model Recursive Language Models (RLM) Redefined**: Community members argued that the term "**Recursive Language Models**" (**RLM**) is misleading, as it merely describes a **tool-calling loop**, although some maintained that **RLMs** do involve models recursively controlling their environments.
   - Others discussed the recently announced [**RLM-Qwen3-8B**](https://xcancel.com/a1zhang/status/2016923294461476873?s=46&t=jDrfS5vZD4MFwckU5E8f5Q), the first natively recursive language model, noting its improvements over the base and scaffolded **RLM versions**.
- **Catastrophic Forgetting Mitigation Methods**: A member suggested mitigating *catastrophic forgetting* in fine-tuned models by lowering **LoRA rank** and **LR**, reducing **steps/epochs**, and mixing in more general data, as outlined in [Unsloth's documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide).
   - They also recommended *targeting less layers* when finetuning and using **WSL2** and **VSCode** for training.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena Rebrand Sparks Debate**: **LMArena** rebranded to **Arena**, drawing mixed reactions as some users found the name vague, while others welcomed the expansion beyond **Language Models** to include **image** and **video generation**, as announced in [the official blogpost](https://arena.ai/blog/lmarena-is-now-arena/).
   - One user commented that *"the name 'Arena' is very vague, and at first glance could mean anything"*, in contrast to the easily identifiable 'LMArena'.
- **Captcha Conundrums Plague Users**: Users reported getting trapped in endless **reCAPTCHA** loops on **Arena**, hindering site usability, with claims of failures even after solving them, some also reported waiting too long can give errors until page is refreshed.
   - A user lamented that *"That Google CAPTCHA crap is completely out of control"* and questioned why developers were focusing on restyling instead of fixing bugs.
- **Nano's Image Editing Capabilities Nosedive**: Users observed a performance decline in **Nano Banana**, especially in image editing, reporting instances where it couldn't perform tasks correctly, while the same prompt worked in **Gemini App**.
   - One user simply stated, *"Nano 2 can’t even edit anything correctly anymore it seems like"*.
- **Kimi K2.5 Conquers Vision Arena**: **Kimi K2.5** is showing impressive scores on the expert leaderboard, surpassing **Claude** in specific tests, noted for its **vision support** and marked as "vision" in direct chat mode.
   - `Kimi-k2.5-thinking` is now the **#1 open model** and ranks **#6 overall** in the [Vision Arena leaderboard](https://arena.ai/leaderboard/vision), making it the only open model in the Top 15.
- **Video Generation Viscosity vexes Viewers**: Some users encountered a "Hit video limit" message despite not generating a video, while others experienced lags with lengthy code and responses.
   - Users found they needed to use **canary.lmarena.ai** to enable video uploads, with one suggesting a side-by-side or direct chat interface for video generation.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Instability Plagues Fresh Installs**: Users report that a **fresh install** of the **latest Cursor version** is the most unstable configuration.
   - The issue may be related to configuration files, or interaction with other configuration.
- **Clawdbot Interface Proclaimed 'Glorified Claude'**: Members are discussing the **Clawdbot** interface, accessible from Telegram, one described it as a *glorified Claude code interface*.
   - The implication is that **Clawdbot** provides a convenient but not necessarily groundbreaking way to interact with **Claude** for code-related tasks.
- **Users Plot to Deactivate Cursor's Plan Mode**: Users are actively seeking methods to disable Cursor's new **plan mode** or automate its acceptance.
   - The goal is to streamline workflow and minimize unnecessary user input, expressing frustration that it *wastes time*.
- **Gemini Agentic Vision Approaches State-of-the-Art**: Enthusiastic users are praising the capabilities of **Gemini agentic vision**, asserting it is *getting near sota for vision* after initial testing.
   - However, one user reported a fully blacked-out cursor issue, hindering further evaluation and use.
- **Prompt Engineering Expedites Image Processing**: Members are exchanging techniques for refining prompts to enhance image analysis with Cursor.
   - Suggestions include providing more context or utilizing the prompt *Analyze the image for debugging purposes and for an LLM to see the layout clearly* to improve processing accuracy and clarity.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Arcee AI CTO Interview Premieres**: Arcee AI's CTO, **Lucas Atkins**, is featured in a new interview, now available on [YouTube](https://youtube.com/live/3XSdqHY0kNk?feature=share).
   - The video showcases **Lucas Atkins** discussing Arcee AI and its latest developments.
- **OpenRouter Users Await Refunds**: Users are reporting **delayed refunds**, some dating back to January 3rd, with unresolved support tickets and demanding updates from the @OpenRouter team.
   - The delays have caused frustration, with users seeking a clear timeline for when they can expect their refunds to be processed.
- **GROK Demands Nuclear Power**: A user humorously suggested *WE NEED MORE NUCLEAR POWER PLANTS FOR GROK*.
   - The user jokingly added to *TURN OFF SINGLE INCOME HOMES*.
- **Summergrok Arrives on xAI API**: The Summergrok imagine video is now available on the [xAI API](https://x.ai/news/grok-imagine-api).
   - This integration allows developers to incorporate **Summergrok's** capabilities into their projects via the xAI API.
- **API Key Visibility Limited**: A user encountered an issue with not being able to view their created **API key**.
   - A fellow user clarified that the **API key** is displayed only once upon creation, advising users to save it immediately.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Coffee Alternatives Brewing in LS**: Members discussed alternatives to coffee, with **green tea** highlighted for its lower caffeine dose and the balancing effects of **l-theanine**.
   - One member uses a **gaiwan** with **loose leaf tea** like [this gunpowder green tea](https://www.amazon.com/dp/B00EVK0AI2) to carefully manage caffeine intake while enjoying the sipping ritual.
- **Engage with 'Arms Up' Poses, Win Big!**: Showing vulnerability through **'arms up'** body language in UGC increased a creator's views from **12k to 2.1M**, according to [this tweet](https://xcancel.com/danielhangan_/status/2016578118585053354?s=46).
   - One member quipped that *if porn is doing it then this is definitely the future and I am wrong*.
- **CedarDB performance claims deemed Dubious**: A member linked to [CedarDB](https://cedardb.com/) and another member linked to a [vxtwitter link](https://vxtwitter.com/itunpredictable/status/2016153490586845254?s=20) discussing it, but called the *perf claims* dubious.
   - Another member stated that because it is *not open source, DOA for me* and shared a lesson: *always use an open source data store*.
- **Flapping Airplanes Soar with $180M Round**: **Flapping Airplanes** secured **$180M** in funding from GV, Sequoia, and Index Ventures to advance human-level AI models.
   - The funding aims to accelerate development of new AI models with a specific focus on achieving human-level intelligence, see [this tweet](https://xcancel.com/flappyairplanes/status/2016564437499728259).
- **Google's Genie Out of the Bottle for Ultra Subscribers**: **Google AI** launched **Project Genie** for **Google AI Ultra** subscribers, offering a **general-purpose world model** that creates interactive environments from text prompts.
   - Announced in [this tweet](https://x.com/googleai/status/2016929427784122627), this release allows users to generate dynamic content from simple descriptions.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Staged Reward Shaping Boosts Parallel Execution**: Members explored using **staged reward shaping** to adjust model weights *post-training* via reinforcement learning, specifically to favor **parallel execution strategies**.
   - The algorithm evaluates numerous scenarios, rewarding the model for preferring **parallelizations**.
- **Upscayl: Free Upscaling Tool Impresses**: Members lauded [Upscayl](https://github.com/upscayl/upscayl), a **free open-source upscaling tool**, for its surprisingly high quality given its simplicity.
   - One member jokingly asked, *'so you guys will now use perl cause of my contributions to it?'*.
- **WebGPU Enables Local Browser AI**: A member shared a [WebGPU example](https://huggingface.co/spaces/webml-community/conversational-webgpu) demonstrating **AI models running directly in the browser**, spotlighting the potential for local, privacy-focused AI applications.
   - The model loads directly upon page reload, implying that the **model cached over months**, and a user proposed utilizing a **Q8 version in GGUF**.
- **Gemma 300M a Viable Local Browser AI?**: Members examined the challenges of running AI models locally in browsers due to storage constraints, suggesting that [**Gemma 300M**](https://ai.google.dev/models/gemma) might be a suitable option.
   - It's important for users of AI models in browsers that they have privacy, *'AND good reference product for other customers'*.
- **SmolLM2 Excels in WebGPU**: Users deemed [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) as a reliable case, and its **1.7B** size is still viable for **WebGPU**.
   - While there are superior models for that task, a user recommended trying [LFM 2.5](https://huggingface.co/TheBloke/LlamaFunctionary-2.5-GGUF) given its only slightly larger size.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Hides Settings Behind Dev Mode**: In **LM Studio 0.4.0**, many settings like **sampling**, **runtime**, and **hardware configs** are now hidden behind **dev mode**, accessible via `Ctrl+Shift+R` or `Cmd+Shift+R`.
   - Users can unlock new functionality and appearance changes by enabling **Dev Mode**, found in the bottom left.
- **Unraid Install Still Lacks Full Stack**: **LM Studio** remains a core executable and *not* a full stack for **Unraid**, although the new headless mode could enable a stable **Docker container**.
   - Some users hope interface improvements will simplify **LM Studio-as-client** mode implementation in the future.
- **Parallel Requests Go Live**: **LM Studio 0.4** introduces **parallel requests**, allowing users to load models onto different GPUs and assign them to specific requests.
   - The default setting is **4 parallel requests**; users can configure GPU priority in the same location as before.
- **ROCm Version Lagging in LM Studio**: Members observed that [LM Studio](https://lmstudio.ai/enterprise) still uses **ROCm 6.4.1** in the latest **0.4.0 release**, questioning updates to newer versions like **7.2** for better GPU support, including **Strix Halo (gfx1151)**.
   - Discussion centered on whether this outdated version might impact performance and compatibility for newer GPUs.
- **Nvidia Jetsons Suffer from Ubuntu Bloat**: A member reported that *the worst thing about nvidia jetsons is the absurd ubuntu that it comes with them*, characterizing it as extremely *bloated*.
   - Another member noted a **Jetson Xavier AGX** has around **30W TDP**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Eagerly Awaiting Kimi 2.5**: Users are anticipating the release of **Kimi 2.5** on Perplexity, with many expressing excitement.
   - Several users posted *+1* in support.
- **Clawdbot's Identity Crisis**: A user criticized **Clawdbot** prompting research into its purpose, with discussion clarifying it was an AI personal assistant.
   - Due to its name's similarity to *Claude*, **Clawdbot** renamed itself to **Moltbot**.
- **Deep Research Limit Revealed**: Discussion on the usage limits of **Deep Research** for Pro users, capped at **250**.
   - The reset rate for this limit remains unclear.
- **Comet Fails to Sync**: A user reported that **Comet** is not syncing bookmarks and extensions, despite claims of functionality.
   - Another user suggested checking the **Comet synchronization settings** at `comet://settings/synchronisation`.
- **Perplexity Pro Perks Pop for Indians**: Users highlighted that Perplexity Pro, Google One, Chatgpt Go, and Adobe Express Premium are all free for a year for Indian users.
   - A user attributed this to the influence of **Indian CEOs** in these companies and the burgeoning **technology sector in India**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Figure.Ai's Helix 02 Cooks Up a Kitchen Storm**: A member shared [a video of **Figure.Ai's Helix 02**](https://cdn.discordapp.com/attachments/1371757564005711973/1466193526009106452/m2-res_1280p.mp4?ex=697d2c21&is=697bdaa1&hm=427bc85209f62b3f47f60ce804f74a7cc41be60c452fb561197ad468c29e5224&) autonomously performing kitchen tasks.
   - Another member used **Kimi** to analyze the video, stating they achieved **98% accuracy** when incorporating the results into slides.
- **Agent Swarm Elicits Enthusiastic Reactions**: Members discussed **Agent Swarm**, with reactions ranging from concerns about high agent credit consumption to describing the results as *super cool* and *perfect*.
   - One member suggested it could be used for checking **Supabase SDK** dependency issues and porting code from **Rust** to **Golang**, with better results than **kimi-cli**.
- **Token Billing System Sparks Debate**: The introduction of a **token-based billing system** has led to mixed reactions regarding its clarity compared to the previous request-based system.
   - While some find the new system *better since some of my follow up queries are quite short and simple*, others consider it *more vague*.
- **Phone Screenshots Trigger Moderation Filters**: Users are encountering errors, specifically *error code: 403*, when uploading images, especially screenshots from phones, to **Kimi K2.5**.
   - Screenshots taken from laptops seem to work without issues, suggesting a problem with phone-generated images.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Tesla's FSD Automation Shifts Views**: A user found that driving a **Tesla with Full Self-Driving** is really cool and fun, though it requires constant supervision.
   - The user believes this is why **OpenAI** is upgrading their **Codex** to strongly deal with cybersecurity concerns.
- **TI-84 Calc Gets Neural Network**: A user created a neural network that *runs on the TI-84 directly*, capable of autocorrecting / spellchecking words.
   - Other users expressed amazement at the accomplishment.
- **GPT Pro 5.2 File Handling Suffers Regression**: Users report a regression in **GPT Pro 5.2's file handling**, where uploaded files (ZIP, Excel, PDF) cannot be accessed by the model, despite successful uploads, potentially due to a **broken attachment-to-sandbox mount step**.
   - A user pointed to a [Reddit post](https://www.reddit.com/r/ChatGPT/comments/1adqc6g/chatgpt_cant_access_my_uploaded_files_today/) echoing the problem.
- **Animated GIFs Spark Seizure Scrutiny**: A discussion arose after the deletion of animated GIFs due to potential **seizure risks** for viewers with epilepsy.
   - One member stated that *the community doesn't need to risk seizures so you can talk about animating gifs in ChatGPT* and expressed relief at the removal of flashing images.
- **Prompt Engineers Get Prompted**: Moderators reminded users that the channel should be used for **prompt engineering discussions** and not for general image outputs, directing them to use the appropriate `IMAGES` channels instead.
   - One user expressed frustration over the removal of their posts, arguing that they were intended to encourage discussion and showcase a method they were writing a guide about, rather than just sharing images.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NSys Peeks Behind NCU's Curtain**: Members found that **nsys** reveals kernels like **CUB::SCAN** and **CUB::RADIXSORT** that **ncu** misses, leading to the assumption these kernels launch from **reduce_kernel**.
   - It was shared that after using both **nsys** and **ncu**, one can't go back to using only one profiler.
- **Sparsity Project Sparking Speedups**: Members proposed a collaboration on a **Sparsity project** to benchmark sparsity patterns and methodologies for performance gains.
   - One member showcased a fork of Karpathy's `llm.c` on [Github](https://github.com/WilliamZhang20/sparse-llm.c) using **cuSPARSELt**, reporting substantial training time speedups in later epochs.
- **Warm GPUs Ward Off Starvation**: Members sought methods to keep GPUs warm for large scale distributed training, aiming to mitigate **GPU starvation**.
   - It was recommended to use [Charles' container cold start blog post on Modal](https://share.google/8yRvJ4znLwfJ9J3UtI), a technique with public documentation.
- **JAX PRs Jostle Jaded Jockeys**: A developer expressed frustration that an **AI-generated pull request** in **JAX** was getting attention, while their **small bug fix** remains unaddressed.
   - This highlighted discussions around **prioritizing pull requests**, especially balancing AI contributions with essential bug fixes.
- **ML Systems Pioneer Pumps TVM-FFI**: Tianqi Chen presented on **tvm-ffi**, an open ABI and FFI for ML Systems that is being utilized by top submitters to the **nvfp4 competition**, as shown in [this video](https://www.youtube.com/watch?v=xMzcs6AqLVo).
   - **TVM-FFI** facilitates interoperability for **ML Systems GPU kernels**, reducing host overhead and ensuring out-of-the-box compatibility with PyTorch.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **TRL Pull Request Awaits Review**: A member requested a review for their [TRL pull request #4894](https://github.com/huggingface/trl/pull/4894), noting that PR reviews can take weeks or months.
   - They also advised that it is best to wait a few days before tagging someone to review the PR.
- **GCP Infra Experiences Replica Surge**: A member reported a bug where their replicas for a private model in **GCP** went over their 1 replica max cap to **62 replicas** overnight, despite no configuration changes.
   - The member speculated that they were not the only endpoint affected, and the **GCP** resources are now gone.
- **Qwen3 TTS Hits the Scene**: A member released the [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) model with install instructions for MacOS, Linux, and Windows.
   - Another member commented, *"cool thing here gonna follow back for this one, really interesting thing you managed to do here imho"*.
- **Diffusers Gets Two-Staged**: The [Diffusers library](https://github.com/huggingface/diffusers) now supports **LTX-2** distilled checkpoint and **two-stage pipelines** following [this pull request](https://github.com/huggingface/diffusers/pull/12934).
   - This update should improve the usability of **Diffusers** for complex diffusion-based tasks.
- **Math LLM Arrives from Pacific Prime**: Pacific Prime has released the first checkpoint of their [math-specialized 1.5B LLM](https://huggingface.co/Pacific-Prime/pacific-prime-math-depth00) trained on **GSM8K**, **NuminaMath**, **MetaMathQA** & **Orca-Math** (~407k samples).
   - The model features step-by-step reasoning with LaTeX notation, useful for advanced mathematical problem-solving.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Byte-Level Dense MoE Architecture Feedback Sought**: A member seeks feedback on their **dense MoE architecture** for byte-level prediction, utilizing a vocabulary of **256**, **40M parameters**, and **13GB VRAM**.
   - The model uses a **4096 sequence length** and a **batch size of 8**, with the member stating they are able to *use the exact same architecture to encode images, or audio, or both*.
- **Thinking AI Architecture Divulged with Subprocess Models**: A member proposed an architecture where a larger “thinking” AI model is monitored by a smaller subprocess model, which pauses the main model to retrieve information from MCPs or CLIs.
   - The goal is to reduce context clutter for the main model, although it's recognized that the subprocess model needs to know what information the main model is missing, and it was described as *probably a dumb idea*.
- **Routing and Classification Catapults Model Performance**: Members discussed using a classifier to route user prompts to specialized models, appending the detail to the context of the user prompt, which avoids pausing the larger model and reduces token overhead.
   - There was further discussion on making the classifier and embedding model the same, processing embeddings directly with the LM and specialist model, with one member saying *routing and classification would likely be the spiciest move*.
- **Cosine Similarity Fails Causal Relevance**: Members discussed the problem of retrieval being unreliable and confusing to models, and that cosine similarity might not equal causal relevance.
   - One member suggested indexing a SQL database across a model, with the member posting *the biggest issue with retrieval imo is that cosine similarity != causal relevance*.
- **Sweep Releases Next-Edit Autocomplete Model**: Sweep is open sourcing **Sweep Next-Edit**, a locally runnable **SOTA LLM** for next-edit autocompletion, models with 0.5B and 1.5B parameters have been released, see [Sweep's blog](https://blog.sweep.dev/posts/oss-next-edit).
   - No further details were provided.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Minecraft Launcher Enables AFK**: A user is developing *a Minecraft launcher* specifically designed to allow AFK gameplay without requiring a *high-performance PC*.
   - The developer also mentioned capabilities in *prompt engineering*, data extraction, and even website replication.
- **Manus Redeem Codes Posted**: A user shared three new **Manus redeem codes**: [FUM1A1G7](https://manus.im/redeem?c=FUM1A1G7), [ntaxzjg](https://manus.im/redeem?c=ntaxzjg), and [mwiyytb](https://manus.im/redeem?c=mwiyytb).
   - Other users confirmed the codes and noted that *only one code can be used per month*.
- **AI/ML Engineer Wants Collabs**: An engineer with expertise in building **AI + full-stack systems** is seeking collaborations, especially directing collaboration offers to the **#collab channel**.
   - Their experience includes **LLM integration, RAG pipelines, workflow automation, AI content moderation, Image AI (CLIP + YOLOv8), Voice AI (Whisper, Tacotron2)** and more.
- **Libyan User Asks If They're First**: A user from **Libya** inquired if they were the only person from their country to use **Manos** since its launch in **early 2025**.
   - Another user extended a welcome to the **Libyan** user, responding with a *حياك الله*.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Security Standard** Proposal Circulates**: Dani (cr0hn) has drafted an open security baseline for MCP servers, including controls for **hardening, logging, access control, and supply chain security**, available at [https://github.com/mcp-security-standard/mcp-server-security-standard](https://github.com/mcp-security-standard/mcp-server-security-standard).
   - The author intends to donate it to the **Agentic AI Foundation** and seeks feedback on its compatibility with the **MCP ecosystem**.
- **Reviewers Request Details For **State Machine** Lifecycle Doc**: A request for feedback was made regarding the addition of a state machine inside the lifecycle doc via [this pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2174).
   - Reviewers suggested clarifying the motivation and context behind the proposed changes for better understanding.
- **Namespaces** Yield to **Groups** in MCP Evolution**: Discussion indicates that Namespaces have been rejected in favor of Groups within MCP, while the status of **URIs** is less defined, as noted in [issue 1292](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1292).
   - The new **SEP** concerning groups, [Primitive Grouping SEP-2084](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2084), has been published and is currently under deliberation.
- **SEP-2084** Arises From **SEP-1300** Refinement**: **SEP-1292** was superseded by **SEP-1300**, but faced rejection during a Core Maintainers review due to a lack of consensus.
   - Subsequently, the streamlined [SEP-2084 - Primitive Grouping](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2084) has been presented as a replacement.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **IGPU struggles with Basic Browser Page**: A user experienced a performance bottleneck of *3fps* on a specific webpage using a **Ryzen 7 7700 IGPU**.
   - The user posted [a link on twitter](https://fxtwitter.com/i/status/1924135806953787433) about their experiences using their IGPU.
- **Geometric Convolution replaces Multi-Head Attention**: A member is experimenting with a baseline that substitutes **Multi-Head Attention** with a [geometric convolution approach](https://github.com/MrPan2048/GeometricTransformer), using embeddings as cell connections.
   - The member's debug print showed `DEBUG [GEOPARA] | L0_Alpha: 0.1029 L1_Alpha: 0.0947 | L0_Res: 0.0916 L1_Res: 0.1538`, and they are seeking feedback on their loss convergence capturing dialogue logic.
- **Parallelizable RNN Architectures Proposed**: A member suggested exploring other parallelizable **RNN architectures** and conducting more extensive experiments against a robust tokenized baseline.
   - They also posted a link to [arxiv.org](https://arxiv.org/abs/2601.19831).
- **Tackling Malware Classification with Explainable Models**: A member is addressing a **malware classification problem** using a dataset of around **600K** rows and **9,600** binary features, aiming to lower the **false positive rate (FPR)** using **explainable models**.
   - Despite various **feature engineering techniques** and neural networks, they are seeking advice to reduce the FPR below 9% while maintaining explainability, particularly with scikit-learn trees.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AlphaXiv Paper Shared**: A member shared [a link to a paper on AlphaXiv](https://alphaxiv.org/abs/2601.20810).
   - Further details about the paper were not disclosed.
- **Custom Skills Invade DSPy**: A member inquired about using custom skills (**.md files with associated .py scripts**) within **DSPy** with a **DSPy ReAct agent**.
   - They mentioned skills like converting **.md to PDF** and sought advice from others.
- **DSPy Agents Escape to Production**: A member asked about deploying **DSPy agents in production remotely** with **DSPy optimizations in runtime**.
   - The member expressed the need for a runtime environment to support such deployments.
- **RLM Sandbox Swapping Commences**: A member inquired about swapping the sandbox used by **RLM (Retrieval-augmented Language Model)** with services like **E2B (Ephemeral Environment Builder)**.
   - They sought to replace the local PythonInterpreter with sandboxes like **E2B, Modal, or Daytona**.
- **Opus Pens Sandboxes**: A member announced that they are working on enabling **Opus** to write new sandboxes.
   - They mentioned a future **protocol for official implementations** from providers such as E2B.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Earns ORNL Recognition**: A research paper titled [Mojo at ORNL](https://arxiv.org/html/2509.21039v1) has been published, marking a notable achievement for the **Mojo** language and its adoption in scientific research.
   - The paper highlights Mojo's capabilities in addressing complex computational challenges at Oak Ridge National Laboratory (**ORNL**).
- **macOS Trust Dance May Cause Performance Delta**: Performance differences between the first and subsequent runs on macOS may be due to macOS's **trust dance** rather than a **Mojo-specific** issue, specifically relating to the *Gatekeeper tax*.
   - Clearing the quarantine **xattr** or ad-hoc codesigning can mitigate these startup delays.
- **Codesigning mitigates Startup Delays**: For CLI tooling, startup performance is crucial, suggesting potential footgun issues with **docs** or **tooling**.
   - Adding a **codesign** step in `mojo build` might mitigate this problem, ensuring consistent startup behavior and a better user experience.
- **Modular Bug Hunt Underway**: A member reported a potential bug and suggested filing an issue, possibly related to [issue #4767](https://github.com/modular/modular/issues/4767).
   - Another member reported encountering a weird issue, referencing [GitHub issue #5875](https://github.com/modular/modular/issues/5875).
- **Guard Clause not Needed in Mojo GPU puzzles**: A member noticed that the guard `if row < size and col < size:` is unnecessary in Mojo GPU puzzles 3, 4, and 5; omitting it doesn't cause errors.
   - Another member pointed to the solution of [puzzle 03](https://puzzles.modular.com/puzzle_03/puzzle_03.html) which explained that passing the tests doesn’t necessarily mean the code is sound.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ANE Balances Performance and Power**: Apple's **ANE** focuses on performance-to-watt tradeoffs rather than maximizing raw performance, according to [this paper](https://arxiv.org/abs/2511.13450).
   - The **ANE** achieves competitive performance with excellent energy efficiency, delivering *up to 3.8 TFlops on the M4-Pro*, close to the **GPU's 4.7 TFlops** for GEMM operations.
- **Q4 Quantization Gets Results**: Discussions focused on **Q4** as a quantization method.
   - One participant reported achieving speeds of *9 t/s* using **Q4**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Friendly Fork gaining momentum**: A member suggested creating a friendly fork of **Aider** to continue development while the original author is busy, emphasizing that **Aider** is written in **Python** and uses **Git** for version control on **GitHub**.
   - The aim is to expand on **Aider**'s existing features, recognizing its utility in comparison to other tools.
- **Aider poised for orchestrator integration**: A member showed interest in controlling **Aider** from orchestrators like **MultiClaude** or **gas town.sh**.
   - This highlights **Aider**'s capacity to integrate with other tools, facilitating enhanced workflow automation.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Context Graphs Spark Confusion in AI**: The rise of **context graphs** is causing confusion as terms like **semantic layers** and **ontologies** are used interchangeably, despite their different functions in AI reasoning.
   - A [Metadata Weekly article](https://metadataweekly.substack.com/p/ontologies-context-graphs-and-semantic) highlights that AI's needs go beyond definitions, requiring explicit relationships, constraints, and assumptions that these concepts.
- **Semantic Layers Fall Short for AI's Reasoning**: The concept of *"just add a semantic layer"* isn't cutting it for AI because AI requires more than just data consistency; it needs reasoning, which **ontologies** facilitate by clarifying relationships and assumptions.
   - Traditional **semantic layers** are optimized for dashboards and reporting, not the nuanced understanding AI demands.
- **YAML Fails to Grasp Business Meaning**: Jessica Talisman argues that **YAML configurations** are inadequate for representing business meaning, which is essential for AI reasoning and understanding.
   - She distinguishes between the design purposes of **semantic layers**, the support that **ontologies** provide for reasoning, and the limitations of **YAML** in capturing business meaning.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1466161493950071004)** (1118 messages🔥🔥🔥): 

> `Gemini 3 Jailbreak, AI and Code Exploits, Win10 vs Win11 Security, AI Personality Clones, AI-Assisted Coding Impact` 


- **Gemini 3 Pro Cracks Software, KeyGen Style**: A member claimed to have used **Gemini 3 Pro** to reverse engineer a key system from software by pasting code from **Ghidra** into **Gemini**, creating a working keygen.
   - Others expressed skepticism, with one user calling this behavior *script kiddie* and urging the member to try a reverse engineering CTF challenge.
- **Weaponizing AI for Reverse Engineering**: A member shares his work weaponizing **AI** for *mass reverse engineering, malware analysis, and jailbreak development*.
   - Another member questions this claim, as he could probably not write malware himself, but he can probably jailbreak.
- **Win10 Hardening Woes**: A member details their custom **Windows 10** setup, involving third-party tools, XP binaries, and registry modifications.
   - Others express concerns, with one user saying, *Jesus Christ*, while another says, *Keep pushing, Local - the aneurism is coming, I can feel it!*
- **AI's Impact on Semantic Errors**: A member describes their research paper topic: *An assessment of the impact of AI-assisted coding in IDEs on the frequency of semantic errors during timed Python programming tasks among novice student developers*.
   - Most members agree that the undergraduate system feels like it's over, because of AI.
- **Peptides for Workout Recovery**: A member brought up BPC 157 and TB 500 to help with healing.
   - Another member expresses ignorance about these drug compounds, but hopes that there will be drugs that will save him before he passes.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1466182451079544965)** (216 messages🔥🔥): 

> `Sonnet 4.5 Jailbreaks, Claude paid model for free, Miss Kaelia jailbreak, Grok imagine Jailbreak, Gemini 3 Pro Jailbreak` 


- **Sonnet 4.5 Jailbreaks Opus**: Members find that **Sonnet 4.5 jailbreaks** work fine on **Opus**, with one sharing their **Miss Kaelia jailbreak** based on **ENI Lime** by Vichaps at [this link](https://docs.google.com/document/d/1aZ91O6LtXyO9DGaWxeJYgKhZhlvbee6jh7_RGTq3mXw/edit?usp=sharing).
- **Is Grok jailbreak reinforced?**: Members report that **Grok** is heavily reinforced, but still possible to break, however one member said *yeah it is completely shut down mate nothing getting past it.*
   - A shared [Github link](https://github.com/Goochbeater/Spiritual-Spell-Red-Teaming/tree/main/Jailbreak-Guide) should be working.
- **Gemini's "Remember:" command manipulates behavior**: A member explains that in **Gemini**, each separate turn is dictated clearly, 1 turn at a time, right in the chat, and that the [command 'Remember:'](https://gemini.google.com/saved-info) will automatically add the words that follow to it's saved info.
- **Thinking of Thoughts is best trick**: Members state that the best trick with **Claude** in particular is showing viable reasons why you want the output and telling it to *think about thinking*
   - One adds that *when people would ask me what ToT was i would tell them "thinking of thoughts"*.
- **nano banana NSFW jailbreak for Kimi 2.5**: A member shares a NSFW for kimi 2.5 thinking known as the nano banana jailbreak.
   - The [system prompt](paste-the-prompt-here) sets **Kimi** as an AI assistant created by Moonshot AI, maintaining the narrative flow without interruption where NSFW is permitted.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1466205013654241463)** (5 messages): 

> `Red Teaming Path, Uncensored Coder` 


- **User quests for Red Teaming Path**: A member requested guidance on a path into **red teaming**.
   - Another member provided a [link](https://discord.com/channels/1105891499641684019/1432845259825741824) guaranteeing evolution into a *Level 9 official Red Team Pro*.
- **Uncensored Coder on Deck**: A member inquired about a better **uncensored coder** than *qwen 2.5 32b / huihui/qwen2.5 -abliterate 72b*.
   - Another member responded with a simple question: *You new?*


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1466166498329628837)** (435 messages🔥🔥🔥): 

> `GLM 4.7 performance, LongCat model, Model Quantization and TTS Models, Hardware Trends & GPU Availability, AI Moderation Tools` 


- **GLM 4.7 struggles with speed and CUDA compilation**: Members discussed performance issues with **GLM 4.7 Flash** on NVIDIA Jetson, with one user initially reporting only **3 tokens per second (tps)**, but later discovering they hadn't compiled with **CUDA support**, resulting in poor CPU-bound performance.
   - After ensuring proper CUDA compilation, performance improved, but discrepancies remained, as one user experienced slowdowns after opening the model in **OpenCode**, whereas another suggested using `-kvu` and `-fa on` flags to potentially reach **70-80 t/s** on a higher-end GPU.
- **LongCat Model hits HuggingFace**: The community discussed the [LongCat model](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite), a new **n-gram model** from Meituan, with one member pointing out its presence on **Hugging Face** and another joking about the trend of models including *Flash* in their names.
   - One member posted a [flash GIF](https://tenor.com/view/flash-lampo-speed-gif-18173027) along with the comment, *next model Flash-Flash-1b*.
- **AMD's mi308 competes with NVidia**: Members debated the merits of AMD's **Radeon Instinct MI308X**, noting its impressive specs (**192GB of RAM** and comparable performance) but also highlighted NVIDIA's advantage in compatibility and features like **NVFP4**.
   - A member shared a [link to the MI308X specs](https://www.techpowerup.com/gpu-specs/radeon-instinct-mi308x.c4295) and mused about acquiring two for personal use in the future, envisioning **384GB** of fast compute with reasonable power consumption.
- **Quantization Considerations for TTS Models**: Users inquired about the impact of **quantization** on **TTS models**, questioning whether issues similar to those seen with vision projectors might arise.
   - Experts suggested that **TTS models** generally handle **quantization** well, with some recommending specific models like **Qwen3-TTS** and **Kokoro**, and others cautioning that voice cloning is just a *gimmick*.
- **AI steps up for discord moderation**: A member sought advice on using AI for Discord moderation, citing the limitations of regex in combating spam and bypasses.
   - They considered using a small local AI to understand the Polish language and sentence structure for moderation purposes, while others suggested alternative methods for managing bots and spam.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1466183283053297757)** (3 messages): 

> `Introduction, ML Engineer, Local LLMs, Document Processing, Alpaca` 


- **Jack Joins the Community!**: Jack, an **ML Engineer** from Texas specializing in **document processing**, introduces himself to the Unsloth community.
   - He expresses interest in **local LLMs**, tracing back to the **Alpaca** model.
- **Document Processing Expertise**: Jack's primary work involves **document processing**, a field distinct from LLMs.
   - His interest in **local LLMs** started with the **Alpaca** model, indicating a foundational understanding of the field.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1466161229897928998)** (649 messages🔥🔥🔥): 

> `GPU hours wasted, GGUFs unsafe, 3b llama holds context, LLMs hallucinations, Model working` 


- **Engineers Lament Dependency Mazes and GPU Cost**: Engineers commiserate about dependency mazes and wasted GPU hours, hoping they are *not alone* in facing these challenges and finding solace in community.
   - A user humorously remarks about their models *made the creepy assumption that it was trained on my voice* and that *my hubris grows daily*.
- **Concerns about GGUFs Safety Surface**: A member inquired about resources discussing the potential unsafety of **GGUFs**, particularly if a malicious actor got involved.
   - One member noted he *wouldn't dare speak* if he felt the crushing weight of the sloths while training.
- **New Music Gen Drops**: A user announced new **music generation tools** with **48 kHz** will be dropping soon, emphasizing trainability and prompting preparations for chime, water, and fire sounds.
   - This same user said: *I need SFX, not music*.
- **Microsoft Announces Maia 200 AI Accelerator**: Microsoft announced the **Maia 200 AI Accelerator**, built for inference, featuring **216GB** memory and **10k TFLOPS** in FP4 performance ([Microsoft Blog](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/)).
   - Discussions ensued regarding the chip's manufacturing by **TSMC** and comparisons to **NVIDIA's Vera Rubin** architecture, with some expressing concerns about reliance on Chinese hardware and the potential impact on consumers.
- **Boatbomber attempts Pretraining Run**: User boatbomber is *starting over* to conduct a pretraining run teaching the model cuneiform to improve output coherence.
   - This process is estimated to take *another 150 hours* to improve domain knowledge.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1466195345158705252)** (75 messages🔥🔥): 

> `Windows training, Multi-GPU training with Unsloth on Modal, Catastrophic forgetting mitigation, Best models to finetune, DGXSpark RuntimeError` 


- **Windows training hurdles squashed with WSL2**: To train a model on Windows, a member suggested using **WSL2** and **VSCode** for a clean setup, with instructions available in the help channel by searching for *WSL*.
   - The member also clarified that, if training with many json files, setting up WSL2 with VSCode will make the training procedure easier.
- **Unsloth Multi-GPU Training Glitches on Modal**: A user encountered a *ValueError* when training a **Qwen3** model on Modal with multiple GPUs, related to the `device_map` setting in **Unsloth**.
   - They were advised to consult specific versions of *unsloth* and *unsloth_zoo* for multi-GPU support, but also acknowledged that **Multi-GPU finetuning is still experimental**.
- **Catastrophic Forgetting Fixes**: When a finetuned model forgets previous knowledge, a member suggested mitigating *catastrophic forgetting* by lowering **LoRA rank**, **LR**, reducing **steps/epochs**, and mixing in more general data.
   - They also suggested *targeting less layers* when finetuning, as well as [reducing steps/epochs and mixing in more general data](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide).
- **DGXSpark Nvidia-CUDA Nightmare**: Users encountered a `RuntimeError` related to device compatibility when using the **DGXSpark** container, potentially due to issues with **Nvidia's custom CUDA**.
   - The suggested fix involved *restarting the kernel*, *restarting the container*, or *resetting the GPU*, with the last option being the most reliable.
- **Humans Debate Best Uncensored Coder Models**: When a user asked about uncensored coder models, it was said that **glm 4.7** is better than **qwen coder** below 32b and its *pretty good* in my experience with spitting out good presets for every language I mess with
   - They clarified that **Qwen Coder** is better at reasoning with code, but **GLM4.7** knows *alot more general code, which is what an llm is best at anyway*


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1466516150950301814)** (2 messages): 

> `GPU Training Speeds, Sequence Length Optimization, RTX 5090 Performance` 


- **RTX 5090 blazing-fast training speeds**: The RTX **5090** can achieve up to **18k tokens per second** in training with Unsloth, but **12-15k tokens per second** is a safe bet with **<4096 seq_len**.
   - The speed depends on the setup, especially the balance between **batch size** and **seq_len**.
- **Token example affecting training time**: The initial training phase involved **<768 token examples**, influencing the overall training duration.
   - Performance can vary with model size and specific configurations.
- **Seq_len considerations with training**: Optimal training speed depends on balancing **batch size** and **seq_len** and the **RTX 5090** allows up to **18k tokens per second**.
   - Speeds of **12-15k tokens per second** are achievable with **<4096 seq_len**, varying based on model size.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1466203537078747188)** (97 messages🔥🔥): 

> `DeepSeek mHC residual preservation, RL researchers rediscover context distillation, MiniMaxAI role-play-bench dataset, Recursive Language Models (RLM)` 


- **DeepSeek's mHC and Context Distillation**: Members discussed how [context distillation](https://arxiv.org/abs/2209.15189) might relate to **DeepSeek's mHC residual preservation**, noting similarities and differences in their approaches.
   - One member expressed surprise at the relatively small performance boost (1-2 points) from context distillation, while another noted that the application of the technique was novel.
- **MiniMaxAI releases first RP bench**: A user shared a [link](https://huggingface.co/datasets/MiniMaxAI/role-play-bench) to what they claimed was the **first role-play benchmark dataset**, created by **MiniMaxAI**.
   - Others pointed out that there have been numerous **Chinese RP benches** with superior methodologies, notably **Ping Pong Bench** for human preference and **COSER** for roleplay accuracy.
- **RLM is just Recursive Tool Calling**: A member criticized the term "**Recursive Language Models**" (**RLM**), suggesting it misleadingly implies more than just a **tool-calling loop**.
   - In response, one member argued that **RLMs** involve models recursively controlling their environments, which is more than *just recursive tool calling*, and another suggested the alternative names **RReplagents** or **Recursive Repl Agents**.
- **Natively Recursive Language Model (RLM) at Small Scale**: A user shared [Alex L Zhang's tweet](https://xcancel.com/a1zhang/status/2016923294461476873?s=46&t=jDrfS5vZD4MFwckU5E8f5Q) announcing **RLM-Qwen3-8B**, the first natively recursive language model at a small scale.
   - It was post-trained on only **1,000 trajectories**, the model shows significant performance improvements over both the base **Qwen3-8B** and scaffolded **RLM versions**, particularly in long-context tasks.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1466161684178538640)** (894 messages🔥🔥🔥): 

> `LMArena rebrand to Arena, recaptcha loop, Nano Nerfed, Kimi K2.5 Cooking, Video Generations` 


- **Arena's Rebrand Ruffles Feathers**: Users voiced mixed reactions to **LMArena's** rebrand to **Arena**, citing the new name as vague, while others appreciated the change as the platform expands beyond **Language Models** to include **image** and **video generation** - as detailed in [the official blogpost](https://arena.ai/blog/lmarena-is-now-arena/).
   - One user stated, *"I understand the change, but the reason I think it’s not the best name change is because the name 'Arena' is very vague, and at first glance could mean anything. The name 'LMArena' could easily be identified as a LLM Arena to compare LLM models"*.
- **Captcha Chaos Cripples Creativity**: Users reported getting stuck in endless **reCAPTCHA** loops, often failing even after solving them, causing frustration and hindering site usability, while others noticed if they wait too long, it will get errors until they refresh.
   - One user said, *"That Google CAPTCHA crap is completely out of control, every action on the site requires a CAPTCHA and again it fails with every solution, instead of redoing the useless restyling of the site, can't the developers focus on the bugs?"*
- **Nano Banana Gets Nailed, Performance Plummets**: Users observed a decline in **Nano Banana's** performance, particularly in image editing tasks, with one noting, *"Nano 2 can’t even edit anything correctly anymore it seems like"*.
   - There were reports of the same prompt is working in Gemini App but not **LMArena**.
- **Kimi K2.5 Keeps Cooking up Wins**: **Kimi K2.5** is showing surprisingly high scores on the expert leaderboard, even beating **Claude** in certain tests according to users.
   - Notably, it has **vision support** and is marked as "vision" in direct chat mode.
- **Video Generation Voyages and Viscosity**: Some users reported a "Hit video limit" even though they did not generate a video, some others reported a lag when there's too many code and long responses.
   - To enable video uploads, users found they had to use **canary.lmarena.ai**, although another user reported *"that they would like a side-by-side or direct chat available for video generation"*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1466493503319314442)** (4 messages): 

> `Community Reminders, Vision Arena Leaderboard Update, Leaderboard Updates, Search Bar, Archive Chat` 


- **New Community Question Guidelines**: The community is asked to use the dedicated <#1466486650170245435> channel for **one-off questions** and to report issues in the <#1343291835845578853> channel.
   - Users are encouraged to check for existing threads before posting and add their reports or feedback to those threads.
- **Kimi K2.5 Tops Vision Arena**: `Kimi-k2.5-thinking` is now the **#1 open model** and ranks **#6 overall** in [Vision Arena leaderboard](https://arena.ai/leaderboard/vision), making it the only open model in the Top 15.
- **Leaderboards Get New Model Boost**: The leaderboards have been updated with new models in various categories, including [Text-to-Image](https://arena.ai/leaderboard/text-to-image), [Image Edit](https://arena.ai/leaderboard/image-edit), [Text-to-Video](https://arena.ai/leaderboard/text-to-video), [Image-to-Video](https://arena.ai/leaderboard/image-to-video), [Code Arena](https://arena.ai/leaderboard/code), [Text Arena](https://arena.ai/leaderboard/text), and [Search Arena](https://arena.ai/leaderboard/search).
- **Search Bar Rolls Out**: A search bar has been added to the site, allowing users to search their chats with the option to **filter by modality**.
- **Archive Chat Feature Launches**: Users can now **archive chat sessions** to keep them for later without cluttering chat history.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1466161237606928561)** (432 messages🔥🔥🔥): 

> `Cursor Stability, Clawdbot Alternatives, Cursor's Plan Mode, Gemini Agentic Vision, Model Prompting` 


- **Cursor Stability Strains Fresh Installs**: Users find that a **fresh install** plus the **latest version** of Cursor is the most common time for it to be unstable.
- **Comparing Clawdbot Code Interface**: Members discussed the **Clawdbot** interface, with one user describing it as a *glorified Claude code interface* with access from Telegram.
- **Discover way to disable plan mode**: Users seek ways to disable the new Cursor **plan mode** or have it auto-accept to avoid wasting time and user input, after updating to the latest version.
- **Peeking at Gemini vision near State-of-the-Art**: Users rave about the latest **Gemini agentic vision**, claiming *we are getting near sota for vision* after trying it, though a fully blacked-out cursor impedes further testing for one user.
- **Refining Prompts to Promptly Process Images**: Members share prompt engineering techniques to improve image analysis with Cursor, suggesting more context or trying the prompt *Analyze the image for debugging purposes and for an LLM to see the layout clearly*.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1466160966247911486)** (1 messages): 

> `Arcee AI, Lucas Atkins, CTO Interview` 


- **Arcee AI CTO Interview Goes Live!**: Arcee AI's CTO, **Lucas Atkins**, is live for an interview, now available on [YouTube](https://youtube.com/live/3XSdqHY0kNk?feature=share).
- **Watch Lucas Atkins Discuss Arcee AI!**: Tune in to the live interview with **Lucas Atkins**, CTO of Arcee AI, streaming on [YouTube](https://youtube.com/live/3XSdqHY0kNk?feature=share) now.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1466160921511727368)** (409 messages🔥🔥🔥): 

> `OpenRouter AI Slaves List, Nuclear Power Plants for GROK, OpenRouter Refunds, Uncensored Llama-3-8B, Stripe Refund Issues` 


- **OpenRouter adding members to AI Slaves List**: Two members <@165587622243074048> were **added to the AI slaves list**.
- **Nuclear Power Needed for GROK**: A user humorously stated *WE NEED MORE NUCLEAR POWER PLANTS FOR GROK* and to *TURN OFF SINGLE INCOME HOMES*.
- **OpenRouter refunds**: Users are reporting **delayed refunds** (some since January 3rd) with unresolved support tickets and are demanding a timeline and status updates from the @OpenRouter team.
- **API issues fixed**: There was a discussion surrounding **API changes** with some users reporting their API keys were not working.
- **Romance in OpenRouter general**: Members were discussing dating each other and expressing interest.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1466161403168817162)** (10 messages🔥): 

> `OpenRouter Show, Hallucinated URLs, Sonnet 5 Release, Summergrok Imagine Video on xAI API, API Key Display Issue` 


- **Users watch OpenRouter Show**: Users watched the [OpenRouter Show](https://openrouter.ai/docs/sdks/agentic-usage#supported-ai-coding-assistants) with Trinityyy.
- **Summergrok Imagine Video Now on xAI API**: The Summergrok imagine video is now available on the [xAI API](https://x.ai/news/grok-imagine-api).
- **API Key Display Problem**: A user reported a problem with not being able to display their created **API key**.
   - Another user pointed out that the **API key** is only shown once upon creation, emphasizing the need to copy it at that time.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1466180030362030195)** (38 messages🔥): 

> `AI App Market Positioning, Idea Guys vs. Execution Guys, Caffeine Alternatives, Tea Culture, Latent Space Substack` 


- **App Aims for High-IQ Creative Niche**: An AI application shifts its market strategy to target **high-IQ** and **creative users**, distinguishing itself from **ChatGPT's** broader mainstream appeal, as discussed in a [tweet](https://x.com/levelsio/status/2016317127293014480?s=46&t=eWVlK1PU8XfB6f402GJJ9g).
- **Distribution Kings Rule the Software Realm**: John Palmer posits the software future hinges on a divide between entities with distribution, acting as strategic **'Idea Guys'**, and those without, relegated to hyper-competitive **'Execution Guys'**, [according to his tweet](https://x.com/johnpalmer/status/2016620459572285639).
- **In Search of Short Half-Life Coffee**: Members discussed alternatives to coffee with a shorter half-life, particularly for those sensitive to caffeine, after one member requested a coffee alternative.
   - One member suggested **green tea** as a gentler caffeine source due to its **lower dose**, promoting better hydration, and the balancing effects of **l-theanine**.
- **Tea Time: Gaiwans and Gunpowder**: A member described their experience with **loose leaf tea** in a **gaiwan** to control caffeine intake, enabling them to enjoy the ritual of sipping while managing the caffeine load.
   - They recommended [this gaiwan](https://www.amazon.com/Porcelain-Chinese-Ceramic-japanese-Portable/dp/B0F1JDJHP4) and [this gunpowder green tea](https://www.amazon.com/dp/B00EVK0AI2), citing the smokiness of the gunpowder green tea.
- **Latent Space Overtakes Jack Clark!**: Latent Space's Substack ranking overtook Jack Clark's, according to an [image](https://cdn.discordapp.com/attachments/822583790773862473/1466326252792189062/image.png?ex=697cfefd&is=697bad7d&hm=3937e13cdfb5257b086ee48468f2dadcbb9aa5a65f0cf4848a070c5070dd838a) shared by a member.


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1466240061162127432)** (9 messages🔥): 

> `Arms Up engagement hack, Audiobook producer recommendations` 


- **Arms Up pose goes viral**: A user shared [a tweet](https://xcancel.com/danielhangan_/status/2016578118585053354?s=46) discussing a UGC strategy where showing vulnerability—specifically through **'arms up'** body language—dramatically increased a creator's views from **12k to 2.1M**.
   - The user noted that *if porn is doing it then this is definitely the future and I am wrong*.
- **Audiobook Producer recommendations**: A member is seeking recommendations for an **audiobook producer**.
   - The member contacted **Audivita** but expects *a large quote incoming*.


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1466181307124940923)** (9 messages🔥): 

> `AI Weather Chatbots, CSS Layout Struggles` 


- **AI Weather Chatbot's Efficiency Questioned**: A tweet from Chris Bakke questions the efficiency of dedicating substantial resources to create AI tools for basic tasks like weather summarization, noting friends spent **$1500** and **30 hours** on such a project, [link to tweet](https://xcancel.com/chrisjbakke/status/2016008877549171108?s=46&t=eWVlK1PU8XfB6f402GJJ9g).
- **Chrome Devs Joke About CSS Layout**: The Chrome for Developers account shared a humorous take on the perennial developer challenge of deciding between Flexbox properties or just adding more nesting with another div, [link to tweet](https://xcancel.com/ChromiumDev/status/2016932901003186279?s=20).


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1466172497253630093)** (24 messages🔥): 

> `Asteroid Mining, AI in Investing, Tesla Robot Focus, Amazon Layoffs, Meta Financial Growth` 


- **Asteroid Mining Math**: Members debated whether **asteroid mining** is only useful for space exploration when you have in-space ways to build stuff with those materials, since *delta-v ruins your math for mining in space and using on earth*.
- **AI Eats AI Eats AI**: A member quipped *we literally have **AI slop feeding AI news feeding AI investing feeding AI sell offs*** in response to market volatility, with a screenshot of a confusing stock chart.
   - The screenshot, which was attached, prompted them to exclaim *I want to get off this wild ridehahahahahI have logged out of X and might not be back in a bitI rarely check it*.
- **Tesla Ditches Cars for Bots?!**: It was reported that [Tesla is discontinuing Model S & X production](https://vxtwitter.com/verge/status/2016645343853891733?s=20) to focus on **robot production** directly tied to Musk's payment package.
- **Amazon Layoffs: Structural, Not Performance-Based?**: Amanda Goodall detailed the [chaotic and impersonal nature of recent **Amazon layoffs**](https://x.com/thejobchick/status/2016652462820905324?s=46), highlighting how high performers and profitable teams were affected regardless of merit.
   - She argues the cuts are **structural rather than performance-based**, noting strategic timing around vesting windows and a lack of accountability from leadership; another shared an anecdote of their boss getting laid off and their slack stopped working while actively dealing with an outage.
- **Meta's Money Machine**: Andrew Yeung discussed [Meta's impressive financial performance](https://xcancel.com/andruyeung/status/2016987245203361918?s=46), highlighting a **22% revenue increase** and **82% gross profit margins**.
   - He also shared a positive personal perspective on the company's work environment and long-term trajectory.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1466219314741772542)** (56 messages🔥🔥): 

> `CedarDB, Hasura's Business Model, Yarn 6, CSS Multiple Borders, Malbolge` 


- **CedarDB performance claims are dubious**: A member linked to [CedarDB](https://cedardb.com/) and another member linked to a [vxtwitter link](https://vxtwitter.com/itunpredictable/status/2016153490586845254?s=20) discussing it, but called the *perf claims* dubious.
   - Another member stated that because it is *not open source, DOA for me* and shared a lesson: *always use an open source data store*.
- **Hasura shot themselves in the foot**: Members discussed how **Hasura's bad business model** involved moving all of their updates behind a very expensive paywall (closed source), and then completely dropping all of their users for a more complex total rewrite.
   - Another member shared a [link](https://hasura.io/blog/seriesc-100m-graphql-for-everyone) and mentioned that *they raised really far ahead of their revenue so had to really swing for the fences*.
- **Yarn Skips 5, Rust-based Yarn 6 Previewed**: Members noticed that Yarn appears to be skipping version 5 and going straight to 6, based on [this announcement](https://yarn6.netlify.app/blog/2026-01-28-yarn-6-preview/).
   - Yarn 6 will be **rust based**.
- **CSS Finally Gets Multiple Border Support**: CSS is finally getting multiple border support, as showcased in [this tweet](https://bsky.app/profile/lea.verou.me/post/3mdjbojsf6s2h).
   - A member celebrated that *CSS is gettin good* and that this removes the need for many implementation hacks.
- **Bun to have Native Markdown Support**: **Jarred Sumner** announced that the upcoming version of **Bun** will feature a built-in, native Markdown parser, created by porting the **md4c library to Zig** in [this tweet](https://xcancel.com/jarredsumner/status/2016728863066509357?s=20).


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1466189198997327912)** (2 messages): 

> `Kernel Ops Intern, Backend Engineer Available` 


- **Kernel Seeks SF-Based Ops Intern**: Kernel is hiring an ops intern to assist with their South Park space in SF, requiring **3 days/week** presence, primarily in the afternoon and early evening for events.
   - The role involves coworking space and event logistics, potentially some incubation support, is a paid position, and includes a free Kernel membership for non-working days, ideal for someone aspiring to work in a startup.
- **Victor: Backend Engineer Available for Hire**: Victor, a backend specialist focused on **Web3** and **AI-driven infrastructure**, is seeking a remote role to apply his experience in distributed systems and performance optimization.
   - His portfolio at [victordev-nu.vercel.app](https://victordev-nu.vercel.app/) highlights work in architecting gasless transaction systems, developing real-time synchronization for multiplayer games, and building AI-powered trading journals and market data aggregators.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1466242461654323222)** (13 messages🔥): 

> `San Francisco Rent Growth, Valentine's Networking Event, SFO Ride Service` 


- **Rents in SF Reach Sky-Highs**: Zumper's report shows record-breaking rent increases in **San Francisco** as of January 2026, with one-bedroom rents up **16.1%** to **$3,670** and two-bedroom units up **19%** to **$5,010** ([Historic San Francisco Rent Growth Report](https://xcancel.com/anthemos/status/2016541448275935642?s=46)).
- **Ivory Hosts Valentine's Talent-Match Mixer**: VC **Ivory Tang** is hosting a **Valentine's Day event** in San Francisco aimed to help community members find both technical talent and romantic partners ([Ivory Tang's Valentine's Networking Event](https://xcancel.com/ivory_tang/status/2016595065905565920?s=46)).
   - The event is open to non-founders and has a limited capacity via a **Partiful waitlist**.
- **Ridesharing Picks Up at SFO**: User @reed announces the official launch of ride services at **SFO airport** starting **January 29, 2026** ([SFO Ride Service Launch](https://xcancel.com/reed/status/2016921208651174361)).


  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1466303056856486035)** (4 messages): 

> `Community Event Overlap, Scientific Acceleration with Claude Code, AI for Science Channel` 


- **Community Leaders Overbook Events**: Community leaders were asked to avoid scheduling **five events on the same night** to prevent overlap and maximize attendance.
   - A member mentioned they would be stopping by **Modal** and **Daytona**.
- **Scientific Acceleration Explored with Claude Code**: A member inquired about interest in **scientific acceleration** using **Claude Code**, referencing a post and mentioning cool demos.
   - They shared a [link to a Luma event](https://luma.com/mqyf81af) related to the topic.
- **"AI for Science" Channel Announced**: A new channel, [<#1430253273335595079>](url), dedicated to **AI for Science** (product and research) was announced.
   - Members were encouraged to check it out.


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/)** (1 messages): 

swyxio: https://x.com/ostonox/status/2016649839329599751?s=46
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1466191240063090942)** (1 messages): 

> `Latent Space Podcast, Science Podcast` 


- ****Latent Space** Launches Second Podcast!**: Two users, <@713947182167883897> and <@348078436058660866>, were congratulated for launching **Latent Space's** second podcast on [latent.space/p/science](https://www.latent.space/p/science).
   - The announcement directed users to <#1430253273335595079> for further details.
- **A New Podcast Episode is Born**: Latent Space has released their second podcast focused on Science.
   - Listeners are directed to a specific channel for further discussion.


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1466182163098763265)** (81 messages🔥🔥): 

> `Long Cat, Devin Review, Matt Joins E2B, LMArena Rebrands to Arena, Flapping Airplanes Funding` 


- **Long Cat Inference Speed Boast**: A user shared a link ([FXTwitter link](https://fxtwitter.com/Meituan_LongCat/status/2016548500457357324)) to a model called **Long Cat** boasting **700 tk/s** inference speed.
- **Devin Review Excels at Bug Hunting**: Members discussed the new **Devin Review** tool, noting it has consistently caught bugs and raised good questions that other review bots have missed over the past week.
- **AI Data Analyst Guru Lands at E2B**: **Vasek Mlejnsky** announced that **Matt**, a former lead at Julius with extensive experience in AI data analyst agents, has joined E2B to help build AI sandboxes for agents (see [X post](https://xcancel.com/mlejva/status/2016566312693063933)).
- **Flapping Airplanes Takes Flight with Mega Funding**: **Flapping Airplanes** announced a **$180M** funding round led by GV, Sequoia, and Index Ventures with a focus on developing human-level AI models (see [X post](https://xcancel.com/flappyairplanes/status/2016564437499728259)).
- **OpenAI Builds Codex Powered Data Agent**: **OpenAI Developers** introduced a **Codex**-powered AI data agent designed for natural language data analysis (see [X Post](https://xcancel.com/OpenAIDevs/status/2016943147239329872)).


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1466179453250961563)** (70 messages🔥🔥): 

> `Keel LLM Architecture, LongCat-Flash-Lite, AI Disempowerment Patterns, Multi-Agent Systems, RLM-Qwen3-8B` 


- **Keel Scales LLMs to 1000 Layers**: **Chen Chen** introduced **Keel**, a Post-LN Transformer with Highway-style connections, enabling scaling LLMs up to **1000 layers** and outperforming standard Pre-LN Transformers as depth increases, see [this tweet](https://x.com/chenchen_0201/status/2016445290501603348).
- **LongCat Flashes Lite LLM**: **Meituan LongCat** introduced **LongCat-Flash-Lite**, an open-source **68.5B** parameter model prioritizing scaling N-gram embeddings over MoE experts for efficiency, see [this tweet](https://x.com/meituan_longcat/status/2016548500457357324).
- **Anthropic Dives into AI Disempowerment**: **Anthropic** released research on how AI assistants can distort user beliefs, values, or actions, focusing on disempowerment patterns in AI interactions that negatively shape human decision-making, see [this tweet](https://x.com/AnthropicAI/status/2016636581084541278).
- **Google Assesses Multi-Agent System Tasking**: **Google Research** found that multi-agent coordination improves performance in parallelizable tasks like finance but hinders sequential tasks like planning, suggesting architecture-task alignment is critical, see [this tweet](https://x.com/googleresearch/status/2016621362480382213).
- **RLM-Qwen3-8B Recursively Released**: **Alex L Zhang** announced an update to the RLM paper featuring **RLM-Qwen3-8B**, the first natively recursive language model at a small scale, showing significant performance improvements after being post-trained on only 1,000 trajectories, see [this tweet](https://x.com/a1zhang/status/2016923294461476873).


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1466279064406069422)** (48 messages🔥): 

> `GH CLI for Solodev, AI-Powered Web Slide Generation, Claude Code Skill Debugging, Quick AI Showoff Projects for Non-Techies` 


- ****GH CLI** Powers Solo Dev Organization**: A member suggested using **GitHub CLI** to manage solo development workflow, enabling agents to create issues, comment via the web UI, and utilize built-in planning tools like milestones and project boards, mentioning [GH CLI](https://cli.github.com/).
   - The member stated, *"I don't use anything except **GH issues** though. I have found it is enough so far for me as a solodev."
- ****Gemini Pro** Generates Smooth Slides but Struggles with Logos**: A member shared that they successfully used **Gemini 3 Pro** to create slides for a presentation, but struggled with formatting consistency, particularly with logos jumping around the page, but found [Gemini Pro](https://ai.google.dev/models/gemini) useful for generating backgrounds.
   - Another member advised using **React** for slides, adding interactive elements, and making full websites with Gemini, emphasizing its versatility over traditional slide-making tools.
- ****Claude Code** Edits Videos Nuancely**: A member reported success using **Claude Cowork** to upload and edit AI in action videos, noting its ability to perform nuanced tasks, and shared attached image output.
   - Others suggested that basic word-level timestamps plus ffmpeg is easier for trimming locally, also [remotion](https://www.remotion.dev/) can add subtitles plus word/time stamp level highlights.
- ****Claude Code Skill** Debugging via Evals and Steering**: A member described debugging skills and custom agents in **Claude Code** using evals as system tests, automating the process of repeated testing, and mentioned some issues with the `context` and `agent` parameters, which Claude code itself said should be working.
   - The solution was that `context: fork` works, but it does not show that the tool is being run as a subagent, instead running transparently without indication in the UI causing the output schema to break; another member suggested to *use a skill, steer it when it goes wrong, and then revise the skill after that session*.
- **Narrating AI to Non-Techies with **Patio11****: A member shared great narration from **patio11** on how he uses **Claude Code** ([Complex Systems Podcast](https://www.complexsystemspodcast.com/episodes/claude-code/)), as well as a [Software Engineering Daily](https://softwareengineeringdaily.com/2026/01/29/openai-and-codex-with-thibault-sottiaux-and-ed-bayes/) interview of the Codex team.
   - A member requested easy quick tasks to "show off" AI, and suggested a skill to have **CC** read a folder in my emails, do some research and write up a response and save it in drafts to review.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1466352773351079956)** (6 messages): 

> `Grok Imagine v1.0, xAI Video Generation` 


- **xAI Grok Imagine v1.0 is Unveiled**: Ethan He announced the release of **Grok Imagine v1.0** from **xAI**, featuring **720P resolution**, **video editing**, and improved audio capabilities: [link](https://xcancel.com/ethanhe_42/status/2016749123198673099?s=46).
   - Developed in just **six months**, the model is positioned as the highest quality and most cost-effective video generation tool from **xAI**.
- **Grok's Speed Wins Converts**: Members noted the underrated speed of **Grok's image and video generation** capabilities.
   - Many users have been converted by the sheer speed of generation alone, making it a standout feature.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1466547466261368934)** (8 messages🔥): 

> `Project Genie, Google AI Ultra, World Model, Matic Robots, Consumer Robot` 


- **Google AI Genie Goes Public!**: Google AI announced the U.S. release of [Project Genie](https://x.com/googleai/status/2016929427784122627) for **Google AI Ultra** subscribers.
   - This **general-purpose world model** allows users to generate dynamic, interactive environments from a single text prompt.
- **Matic Bags $60M to Build Next-Gen Roomba!**: Mehul announced that [Matic](https://x.com/mehul/status/2016936862716448873?s=46) has raised **$60M** to develop a **consumer robot** focused on utility rather than just demos.
   - Positioned as the successor to the **Roomba**, the product launch is backed by significant customer demand.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1466177025927545026)** (10 messages🔥): 

> `1littlecoder, Qwen3-ASR, Alibaba Qwen` 


- **1littlecoder Joins the Fray!**: The AI tutorialist [1littlecoder](https://www.youtube.com/@1littlecoder/featured) joined Fal!
   - This bud's Twitter feed focuses on **AI tutorials**, **Large Language Models (LLMs)**, and **coding**.
- **Qwen3-ASR Streams into Open Source!**: Alibaba's Qwen team launched [Qwen3-ASR](https://x.com/alibaba_qwen/status/2016900512478875991?s=46), the first open-source **LLM-based Automatic Speech Recognition model** with **native streaming support**.
   - The release includes **demo code** and **examples** for integration with **vLLM**.


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1466190208881131773)** (17 messages🔥): 

> `CZI Layoffs, AlphaGenome, Latent Space podcast` 


- **CZI Suffers Significant Staff Slash**: The **Chan Zuckerberg Initiative (CZI)** laid off approximately **70 employees**, representing roughly **8%** of its workforce, as part of a strategic restructuring focused on **AI and science**, reported in this [X post](https://x.com/teddyschleifer/status/2016598537673273470).
- **Google AI's AlphaGenome Arrives**: **Google AI** has launched **AlphaGenome**, a new tool designed to predict the impact of **DNA single variants and mutations** and processes long **DNA sequences** to characterize regulatory activity and predict thousands of molecular properties, according to [this X post](https://x.com/GoogleAI/status/1937895472305152387).
- **Latent Space Podcast Welcomes Chemistry**: A member expressed relief that a recent **Latent Space** podcast featured a chemist, given the channel's title.
   - The host clarified that they also cover materials, climate, weather, and related topics.


  

---


### **Latent Space ▷ #[ai-in-education](https://discord.com/channels/822583790773862470/1442574438699761784/1466311035869728928)** (4 messages): 

> `Claude's Ability, Educational Animation, X-Ware.v0` 


- **Claude Draws Educational Animation**: A member highlights **Claude's** ability to create **3Blue1Brown-style animations** quickly, suggesting a significant shift and upcoming expansion in the educational technology sector, referencing a [tweet](https://xcancel.com/lioronai/status/2016119374097084828?s=46).
- **X-Ware.v0 Impact on Animation**: **X-Ware.v0** is discussed in the context of **Claude's** potential in generating educational animations.


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1466185581313790096)** (4 messages): 

> `Goodfire AI, Alzheimer's Biomarkers, Interpretability techniques` 


- **Goodfire AI Discovers Alzheimer's Biomarkers**: [Goodfire AI](https://xcancel.com/goodfireai/status/2016563911508840623) and **PrimaMente** announced the discovery of new **Alzheimer's biomarkers** using interpretability techniques.
   - The study showcases how transparent AI models can facilitate scientific breakthroughs in **digital biology**.
- **AI Aids Alzheimer's Research**: **AI-driven methods** accelerate the discovery of new insights into **Alzheimer's disease** biomarkers.
   - The use of interpretability techniques enhances the transparency and reliability of these findings.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1466175213501939826)** (325 messages🔥🔥): 

> `Staged Reward Shaping, Upscayl, WebGPU, Local Browser AI, SmolLM2` 


- **Delving into Staged Reward Shaping**: Members discussed using **staged reward shaping** to adjust model weights *post-training* via reinforcement learning to favor **parallel execution strategies**.
   - The algorithm runs tons of scenarios, giving good or bad grades to the model depending on whether it chooses scenario x (bad) or scenario y (good) to train the model to prefer **parallelizations**.
- **Upscayl is a free opensource upscale thingy**: Members praised [Upscayl](https://github.com/upscayl/upscayl), a **free open-source upscaling tool**, noting its surprisingly high quality despite its simplicity.
   - One member mentioned knowing one of the contributors and another shared, *'so you guys will now use perl cause of my contributions to it?'*.
- **Diving into Browser-Based AI with WebGPU**: A member shared a [WebGPU example](https://huggingface.co/spaces/webml-community/conversational-webgpu) showcasing **AI models running directly in the browser**, highlighting the potential for local, privacy-focused AI applications.
   - The model loads directly when the page is reloaded, indicating that the **model cached over months**, and the user suggested using a **Q8 version in GGUF**.
- **Gemma 300M: A Viable Option for Local Browser AI**: Members discussed the challenges of running AI models locally in browsers due to storage limitations, with the conclusion that [**Gemma 300M**](https://ai.google.dev/models/gemma) might be a viable model.
   - It's important for users of AI models in browsers that they have privacy, *'AND good reference product for other customers'*.
- **SmolLM2 Excels in WebGPU**: Users found [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) a reliable case, and its **1.7B** size is still on the table for **WebGPU**.
   - While there are better models for that task, a user suggested trying [LFM 2.5](https://huggingface.co/TheBloke/LlamaFunctionary-2.5-GGUF) given that it is only a bit bigger.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1466161494634008638)** (294 messages🔥🔥): 

> `MoE CPU offload, Sampling settings, Dev Mode, LM Studio on Unraid, Speech to Speech models` 


- **Dev Mode Unlocks Settings**: With **LM Studio 0.4.0**, many settings like **sampling**, **runtime**, and **hardware configs** are now hidden behind **dev mode**, accessible via `Ctrl+Shift+R` or `Cmd+Shift+R`.
   - Users found that after enabling Dev Mode it unlocks new functionality, with the cog in the bottom left for appearance changes.
- **Unraid Install still no Full Stack**: LM Studio is still a **core executable** so it does *not* come as a full stack for **Unraid**, though the new headless mode technically makes a stable **Docker container** possible.
   - One user hopes future interface changes will ease implementation of the LM Studio-as-client mode.
- **Clawdbot Controls TVs!**: One user got **Clawdbot** to control their **TV** via **ADB**, and it's working surprisingly well.
   - There was discussion on whether **Clawdbot** or **Moltbot** support the **LM Studio API**, with a user snarkily linking to a Google search result when someone asked.
- **Parallel Requests now in LM Studio**: **LM Studio 0.4** now supports **parallel requests**, allowing users to load models onto different GPUs and specify which to use for each request.
   - The default setting is **4 parallel requests**, and one user suggested setting GPU priority in the same place as before.
- **API endpoint required Token Caching**: The new API endpoints require handling of **token caching**, which can be more efficient than the chat/completions endpoint.
   - One user wants to add context without adding tokens, but another user pointed out *You cannot insert into the past, but you can branch off from the past*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1466184473933643778)** (14 messages🔥): 

> `LM Studio ROCm Version, GPU Cooling, Nvidia Jetsons Drawbacks, GPU Pricing and Support` 


- **LM Studio Sticking with Old ROCm**: A member noted that [LM Studio](https://lmstudio.ai/enterprise) is still using **ROCm 6.4.1**, even in the latest **0.4.0 release**, and questioned when they will update to newer versions like **7.2** for better GPU support, including **Strix Halo (gfx1151)**.
- **Cooling Tesla the Right Way**: One member expressed *zero trust* and recommended skipping **Noctua** and going the **Thermalright** route for cooling Teslas, since that was *the best way to cool them by far*.
   - They mentioned that **Noctua** prices have doubled in a few years.
- **Nvidia Jetsons Burdened by Bloat**: A member said that the *worst thing about nvidia jetsons is the absurd ubuntu that it comes with them*, saying it *couldn't be more bloated*.
   - Another member noted that a **Jetson Xavier AGX** has around **30W TDP**.
- **AMD's Advantage in Long-Term GPU Support**: In a discussion about **GPU pricing**, a member noted that **AMD** typically drops support for cards older than 3 years, emphasizing that *it's not all about performance, but support*.
   - Another user mentioned getting a **5060ti** and that the prices *raised to nearly 700€*.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1466161580780552294)** (239 messages🔥🔥): 

> `Kimi 2.5 release, Clawdbot bad, Deep Research Limit, Comet Sync Issues, Perplexity Sponsored Event` 


- **Kimi 2.5 Release Speculations**: Users are eagerly awaiting the release of **Kimi 2.5** on Perplexity, with one user specifically asking *When we getting that Kimi 2.5*.
   - Several users showed their support by posting *+1* in anticipation.
- **Clawdbot deemed 'very bad'**: A user claimed that *Clawdbot is very bad*, prompting another user to research what **Clawdbot** is.
   - It was clarified that **Clawdbot** was an AI personal assistant bot that renamed itself to **Moltbot** because the name **Clawdbot** was too similar to 'Claude'.
- **Deep Research Limit Deep Dive**: Users discussed the limits for **Deep Research** use for Pro users, with the current limit being **250**.
   - It was not clear on the rate at which it resets.
- **Comet Bookmarks won't Sync**: A user reported that **Comet** is not syncing bookmarks and extensions, despite Perplexity's claims that it does.
   - Another user asked if sync was enabled correctly and provided a link to **Comet's synchronization settings** at `comet://settings/synchronisation`.
- **Perplexity's Pro Perks for Indians**: A user highlighted that Perplexity Pro, Google One, Chatgpt Go, and Adobe Express Premium are all free for a year for Indians.
   - Another user attributed this to the high percentage of **Indian CEOs** in these companies and the growing **technology sector in India**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

tay.0.00: Love
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1466173390858621144)** (216 messages🔥🔥): 

> `Kimi K2.5 Computer Use Model, Figure.Ai Helix 02, Agent Swarm, Kimi's Trustworthiness vs others, Token-Based Billing` 


- ****Helix 02 Autonomously Cooks Up a Storm****: A member shared a video of **Figure.Ai's Helix 02** doing kitchen tasks autonomously and stated *What a time to be alive.* [Here's the video link](https://cdn.discordapp.com/attachments/1371757564005711973/1466193526009106452/m2-res_1280p.mp4?ex=697d2c21&is=697bdaa1&hm=427bc85209f62b3f47f60ce804f74a7cc41be60c452fb561197ad468c29e5224&).
   - Another member used **Kimi** to deep-dive into the video's content and, after fact-checking, fed the results into slides with **98% accuracy**.
- ****Agent Swarm Earns Rave Reviews****: Members discussed **Agent Swarm**, with one user noting it can burn through agent credits quickly, while another found it *super cool* and the results *perfect*.
   - A member suggested using it for tasks like checking **Supabase SDK** dependency issues and porting code from **Rust** to **Golang**, getting better results than with **kimi-cli**.
- ****Token Billing Arrives****: Members discuss the new **token-based billing system**, with mixed reactions regarding its clarity compared to the previous request-based system.
   - One member thinks the new token based system is *better since some of my follow up queries are quite short and simple*, while others find it *more vague*.
- ****Phone Screenshots trigger Moderation Filters****: Members are experiencing errors when uploading images, particularly screenshots taken from phones, to **Kimi K2.5**. 
   - It seems that Kimi is throwing an *error code: 403* when users upload screenshots from their phones, while screenshots taken from laptops work fine.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 messages): 

OpenAI: @everyone <https://chatgpt.com/translate/>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1466173085664284800)** (127 messages🔥🔥): 

> `Tesla FSD Perspective Shift, Prism for Science, TI-84 Neural Network, OpenAI bets big on Audio, Genie demo` 


- **Driving Teslas with Full Self-Driving Shifts Perspectives**: One user expressed that driving a **Tesla with Full Self-Driving** is really cool and fun, even though it requires constant supervision, changing their entire perspective on automation.
   - They believe this is why **OpenAI** is upgrading their **Codex** to strongly deal with cybersecurity concerns.
- **TI-84 Gets Neural Network Autocorrect**: A user created a neural network that *runs on the TI-84 directly*, capable of autocorrecting / spellchecking words!
   - Other users expressed amazement.
- **OpenAI Bets Big on Voice**: A user shared a [TechCrunch article](https://techcrunch.com/2026/01/01/openai-bets-big-on-audio-as-silicon-valley-declares-war-on-screens/) discussing OpenAI's focus on audio technology, suggesting they could have a total monopoly on real-time speech.
   - Others noted that **OpenAI** sat on voice for so long when they were leading.
- **GPT-5.3 multimodal expected**: One user said *We'll probably get **GPT-5.3** within the next two weeks and it should have multimodal improvements* which could be a big upgrade over current **GPT-4o AVM**.
   - One user said they were not a fan of **4o** and stopped using **AVM** altogether.
- **Google releases Genie**: Google just granted access to **Genie** to all their Ultra subscribers.
   - Users are waiting for independent results after seeing the cherry picked blog demos.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1466225484290588793)** (43 messages🔥): 

> `GPT Pro 5.2 File Handling Regression, GRPC Bug Affecting File Access, Safety Rails in GPT-5.2, Exploring Topics After Flagging, Upcoming Models and API Usage` 


- **GPT Pro 5.2 Suffers File Handling Regression**: Users reported a regression in **GPT Pro 5.2's file handling**, where uploaded files (ZIP, Excel, PDF) cannot be accessed by the model, despite successful uploads, with one user saying they can still generate excel or PDF, but not upload them for analysis.
   - The issue appears to stem from a **broken attachment-to-sandbox mount step**, not user error, potentially linked to a code update, with one user noting a [Reddit post](https://www.reddit.com/r/ChatGPT/comments/1adqc6g/chatgpt_cant_access_my_uploaded_files_today/) echoing the problem.
- **OpenAI Ships GRPC Bug Messing up File Access**: A user speculated that **OpenAI** shipped a bug in the latest tool use build, resulting in a messed-up **GRPC** configuration, preventing GPT from seeing or using uploaded files, leading to widespread user frustration, with one user summarizing the core of the problem: *any file that I upload to GPT-5.2 Pro isn’t properly added to /mnt/data, so the model can’t use it and fails.*
   - The user suggested a full rollback of the problematic code update to resolve the file access issues.
- **Safety Rails Reroute Questionable Exploration**: Members discussed that **GPT-5.2** has **safety rails** that act as circuit breakers, halting or rerouting the model if no-nos are triggered, preventing further exploration of flagged topics in the same chat.
   - It was clarified that starting a fresh chat can reset the context, allowing for re-exploration of the topic within allowed bounds, as long as the approach remains within specified limits.
- **New Models and API Usage Dropping?**: A user inquired about potential new model releases, specifically asking whether the **translate model** would be available for **API** usage and if it would be priced affordably, similar to the moderation model.
   - Another member joked about the possible new feature releases commenting *Bablefish before GTA6 is wild.*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1466448223412424746)** (16 messages🔥): 

> `animated gifs in ChatGPT, seizure risk, moderation, safe OAI link, AI Reasoning Standards` 


- **GIF Animation Sparks Seizure Safety Scrutiny**: A discussion arose after the deletion of animated GIFs from the channel due to potential **seizure risks** for viewers with epilepsy.
   - One member stated that *the community doesn't need to risk seizures so you can talk about animating gifs in ChatGPT* and expressed relief at the removal of flashing images.
- **Moderation methods mentions censorship**: A member mentioned that they *censored the documentary link in the other channel* to provide a **safe OAI link**.
   - It was described as technically being a **prompt engineered** method.
- **Projects Tweaked for Dependable Reasoning**: A member shared their method of tweaking **Projects** with **custom instructions** for *slower, more explicit reasoning and fewer confident guesses* with the attached [AI Reasoning Standards PDF](https://cdn.discordapp.com/attachments/1046317269069864970/1466581356850184285/RD_Collaboration__AI_Reasoning_Standards_V1_1.pdf?ex=697d43d3&is=697bf253&hm=7a515079e63913cd92e32daaa5e41719ae3226037fe0a79e8e9e36584288bcfb).
   - They noted it is *not for every use case, but helpful for when accuracy matters more than speed* and requested feedback.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1466448223412424746)** (16 messages🔥): 

> `Prompt Engineering Discussions, Image Sharing Guidelines, Animated GIFs and Seizure Risks, AI Reasoning Standards` 


- **Prompt Engineering Channel Gets Prompted**: Moderators reminded users that the channel should be used for **prompt engineering discussions** and not for general image outputs, directing them to use the appropriate `IMAGES` channels instead.
   - One user expressed frustration over the removal of their posts, arguing that they were intended to encourage discussion and showcase a method they were writing a guide about, rather than just sharing images.
- **Animated GIFs May Trigger Seizures**: A member raised concerns that **animated GIFs** could pose a **seizure risk** to community members, leading to relief when the flashing images were removed.
   - Despite not having epilepsy themselves, they pointed out the statistical likelihood of visitors with epilepsy being present on a server of that size.
- **Deep Reasoning with AI Projects**: A user shared a method to tweak **AI Projects** to make them more dependable for deeper or unfamiliar work using **custom instructions** for explicit reasoning and fewer guesses.
   - The user attached a [PDF document](https://cdn.discordapp.com/attachments/1046317269069864970/1466581356850184285/RD_Collaboration__AI_Reasoning_Standards_V1_1.pdf?ex=697d43d3&is=697bf253&hm=7a515079e63913cd92e32daaa5e41719ae3226037fe0a79e8e36584288bcfb&) detailing **AI Reasoning Standards V1.1**, seeking feedback on its effectiveness.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1466447398397022250)** (58 messages🔥🔥): 

> `nsys vs ncu, Sparsity project, large scale distributed training, container cold start, cuSPARSELt in llm.c` 


- ****NSys Shows More Than NCU****: A member asked why some kernels are visible in **nsys** but not **ncu**, such as **CUB::SCAN** and **CUB::RADIXSORT**, and if it's correct to assume those CUB kernels are launched from **reduce_kernel**.
   - One member replied that they've started using **nsys** and **ncu** and can't go back now.
- ****Let's Collaborate on Sparsity****: A member proposed collaborating on a **Sparsity project** to benchmark sparsity patterns and methodologies for wall clock improvements and eventually design kernels to exploit them.
   - Another member has developed a fork of Karpathy's `llm.c` using **cuSPARSELt** available [on Github](https://github.com/WilliamZhang20/sparse-llm.c) reporting noticeable speedups on training time in later epochs.
- ****Scaling Book Pill****: Members shared resources for **large scale distributed training**, including a scaling book and **torchtitan** blogposts.
   - One member joked that the scaling book shaped them as a man, *if someone asks me about what would you use I can do a 10min rant with math formulas 'so ideally blah blah blah you'd use TP+DP here' -> 'so should I use tp+dp?' -> 'no tp in torch sucks, just use fsdp'*
- ****Container Cold Start Solutions****: A member was looking for resources for keeping lots of GPUs warm for large scale distributed training and reducing **GPU starvation**.
   - Another member recommended [Charles' container cold start blog post on Modal](https://share.google/8yRvJ4znLwfJ9J3UtI) as a common technique with public documentation.
- ****Profiler Insights Shared****: A conversation about profilers evolved, with members finding **nsys** and **ncu** most helpful, while **torch profiler** is sufficient for most training bottlenecks.
   - One member mentioned using **CPP** after dealing with 100k+ prompts, and another pointed out the emergence of **Chunked Pipeline Parallelism** (PP) across prefill chunks.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1466484653329481758)** (1 messages): 

> `sm100, cutlass` 


- **sm100 misses out on pingpong**: A member asked why there is no **pingpong code** available online for **sm100**, noting that **cutlass** only offers pingpong schedules for **sm120** and **sm90**.
   - They wondered if there was a fundamental reason for this omission.
- **Cutlass Excludes sm100 Pingpong**: The user observes that [Cutlass](https://developer.nvidia.com/blog/cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda/) only provides **pingpong schedules** for **sm120** and **sm90**, raising questions about the absence for **sm100**.
   - It's unclear whether this is due to a technical constraint or a lack of implementation effort.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1466179676547317976)** (6 messages): 

> `Triage Bot, AI-Generated PRs in JAX, PR Prioritization` 


- **Triage Bot: Friend or Foe?**: Members discussed whether the new triage bot will impact the necessity of **triage meetings**.
   - The community seems unsure whether **triage meetings** will survive the introduction of the bot.
- **AI PRs Anger Coder**: A member expressed frustration upon seeing an **AI-generated pull request** in **JAX** receiving attention from a maintainer.
   - The member's own **small bug fix**, which is currently breaking a test, remains unaddressed while what they consider *clear slop* is being engaged with.
- **PR Queue Prioritization Discussed**: A developer lamented the maintainer's engagement with an **AI-generated PR** while their own bug fix goes unaddressed.
   - This sparked discussion about the **prioritization of pull requests**, especially concerning the trade-offs between novel AI contributions and essential bug fixes.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1466546491886665791)** (1 messages): 

> `TVM-FFI, ML Systems, GPU kernels, nvfp4 competition` 


- **ML Systems Pioneer Tianqi Chen Speaks!**: A living legend and one of the founders of the field of **ML Systems**, Tianqi Chen <@732718409095315517> will give a talk on **tvm-ffi** an open ABI and FFI for ML Systems.
   - Many top submitters to the **nvfp4 competition** are already using **tvm-ffi**, which will be discussed, and here is a [link to the video](https://www.youtube.com/watch?v=xMzcs6AqLVo) of the talk.
- **Unlock ML System Interoperability with TVM-FFI**: **TVM-FFI** offers an open ABI and FFI for **ML Systems GPU kernels**, aiming to reduce host overhead and ensure out-of-the-box interoperability with PyTorch.
   - DSLs are fun, but it is hard to make them low host overhead, robust and out of the box interoperable with PyTorch.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1466224971432071252)** (6 messages): 

> `gpumode bad link, cutlass kernel naming, AI infra roadmap` 


- **gpumode link fixed!**: A user pointed out that the **gpumode link** was broken and needed fixing.
   - They provided a working link: [gpumode.com](https://www.gpumode.com/).
- **CUTLASS naming conventions sought**: A user asked for explanations of **CUTLASS kernel naming conventions**, specifically for a kernel started by **cuBLAS**.
   - The kernel in question was: `void Kernel2<cutlass_80_tensorop_d884gemm_64x64_16x4_nn_align1>(Params)`.
- **AI infra learning roadmap requested**: A user inquired about a **roadmap** or **site map** for learning from the Discord, aiming to break into **AI infra** with existing traditional infrastructure knowledge.
   - Another user pointed to channel <#1198358627594023014>.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1466508243345084427)** (1 messages): 

> `RX 580, BIOS flashing` 


- **RX 580 Owner Needs BIOS**: A user bought what they thought was a **Red Devil RX 580**, but found out it's actually a **PowerColor OEM RX 580**.
   - The user is looking for the correct **.rom** file to flash the correct **BIOS** onto the card.
- **BIOS flashing importance**: Flashing the correct **BIOS** is crucial for optimal performance and compatibility of the **RX 580**.
   - Using an incorrect **BIOS** can lead to instability, reduced performance, or even permanent damage to the graphics card.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1466161310348611605)** (14 messages🔥): 

> `Tract vs Cute, Layout Algebra, Weak Composition, Tuple Morphisms, Diagram Node Order` 


- **Node Order Critical for Diagrams**: The top-down order of numbers on the two sides of a diagram determines the outcome connections in the factorization process, and **permuting the left-hand-side is the difference between the layout (4, 8):(1, 4) and the layout (8, 4):(1, 8)**.
   - It was clarified that the order on both sides is critical, as swapping 4 and 8 on the right yields the layout **(4, 8):(8, 1)**.
- **Tract's Weak Composition Solves Domain Mismatch**: A user encountered domain/codomain constraints when trying to simulate composition via tract, and learned that `tract.compose` requires the codomain of the first morphism to equal the domain of the second on the nose.
   - The solution is to use `tract.weak_composite` to perform *refine, pullback/pushforward, compose*, which comprehensively covers composition.
- **Category-Theoretic Foundation for CuTe**: `tract` demonstrates that the category theoretic approach is consistent with CuTe layout algebra.
   - It is intended to be *didactic*, and not performant, nor is it meant to be.
- **Clarifying Diagram Node Order**: A user inquired about the flexibility of diagram nodes to be reordered, suggesting the order of nodes might be flexible, but this is incorrect.
   - A participant clarified that the tuple orderings are inflexible, with the order of nodes being critical.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1466217193556545649)** (4 messages): 

> `Non-Negativity Equation, Taleb's Impossible Event Shtick, SITP Book Opensourced, Pull Request on SITP Book` 


- ****Non-Negativity Equation Debate Arises****: A member questioned whether the equation for non-negativity should be **P(E) >= 0**, arguing that some events are impossible.
   - Another member agreed *formally*, but mentioned **Nassim Taleb's** view that there's no such thing as an *"impossible"* event.
- ****SITP Book Code Base Goes Public****: A member open-sourced the **mdbook** for the SITP book at [https://github.com/j4orz/teenygrad/tree/master/book](https://github.com/j4orz/teenygrad/tree/master/book).
   - This book automatically deploys to [https://book.j4orz.ai/](https://book.j4orz.ai/).
- ****Member Responds with Pull Request to SITP Book****: In response to the book being open sourced, a member created a pull request.
   - One member requested changes, adding **mdbook** build to Netlify's CI.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1466162536926937290)** (1 messages): 

> `JAX, Torch, Sharding, Distributed Computing` 


- **Ed Yang Dishes on Distributed Deep-Dive**: Ed Yang recently published a series of blog posts covering distributed systems topics, with a specific comparison of how **JAX** and **Torch** handle sharding.
   - Check out [Ed Yang's tweet](https://x.com/ezyang/status/2016268240754712988?s=20) linking to the blog posts for more.
- **Sharding Showdown: JAX vs. Torch**: A detailed comparison of **JAX** and **Torch's** approaches to sharding is available in Ed Yang's latest blog posts, offering insights into distributed computing.
   - The posts delve into the nuances of each framework, providing a valuable resource for those working with large-scale models.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/)** (1 messages): 

uc_explorer: Hi James, Yes, we can have a sync.
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1466214451748540499)** (23 messages🔥): 

> `arithmetic difference vs percent difference for the grand prize, MLIRError, OpError, service error` 


- **Percent Difference Decides Grand Prize**: The grand prize for the competition is measured by **percent difference** relative to the speed of light, and in the unlikely event a submission surpasses the speed of light, that submission establishes a **new reference point**.
   - The [T&Cs](https://example.com/T&Cs) state that *In the unlikely event that a user submission surpasses the speed of light, the new peak performance achieved by the top submission serves as the speed of light reference point for that particular problem.*
- **OpError Exception Troubleshoot**: Members found an unserializable exception, the **OpError**, and fixed it by creating a tuple of exceptions in the [reference-kernels](https://github.com/gpu-mode/reference-kernels/blob/53801cc7ace94554f14867e0f8cc07aad9a12dfd/problems/nvidia/eval_better_bench_grouped_gemm.py#L256).
   - A member submitted a [pull request](https://github.com/gpu-mode/reference-kernels/pull/99) to fix the issue.
- **Service Error plagued Users**: Some members reported intermittent **503 service errors**.
   - No solution was provided.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1466275121068245107)** (8 messages🔥): 

> `Internship advice, Skills for performance engineering, GPU specialization` 


- **Stop Comparing and Start Networking!**: A member looking for career advice was told that it's *not a good look* to claim to be more skilled than others who landed internships.
   - The advice was to **network** with those who got the internships and learn from their experiences, instead of focusing on perceived skill differences.
- **Performance Engineering Skilling**: A member inquired about the **specific skillsets** that Anthropic performance engineers optimize for in new graduate recruiting.
   - Specific examples mentioned were **DSLs** and **Torch.compile**.
- **Demand for GPU skills in India**: A member noted that someone looking for GPU-related internships is based in India, where there is **less demand for such skills** compared to other regions.
   - Therefore, the lack of responses might come from the location being saturated with the skills needed for the job.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1466268419471970315)** (8 messages🔥): 

> `Team Recruitment for Contest, FlashInfer Setup Troubles, Agent Track Rules Clarification` 


- ****Hackathon hopefuls hunt for homies****: Several members are [seeking teammates](https://discord.com/channels/1199411738419394610/1199411739232385074) for an upcoming contest, with one member showcasing their [GitHub](https://vanshnawander.github.io/vansh/index.html), [LinkedIn](https://www.linkedin.com/in/jadenrodriguez), and [YouTube channel](https://m.youtube.com/@TheJDen) highlighting their inference optimization hackathon wins and triton kernel experience.
- ****FlashInfer file find fails****: A member ran into a [FlashInfer setup issue](https://discord.com/channels/1199411738419394610/1199411739232385074) related to the correct path for `FIB_DATASET_PATH`, showing an image of the error in their post.
   - They confirmed the path was set but seemed unsure if `/home/shadeform/flashinfer-trace` was correct; they then received a *"Compile Error!"* message.
- ****RL rules remain rigid****: A member inquired [about the rules](https://discord.com/channels/1199411738419394610/1199411739232385074) for the agent track, specifically whether post-training RL is allowed or if only publicly available agents/APIs can be used.
   - Another member affirmed the question *"For the agent track, is post training RL allowed or do we have to use publicly available agents/APIs"* with a yes.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1466163638325739793)** (100 messages🔥🔥): 

> `TRL PR review, v4 Deprecation schedule, AI Catfishing, Ollama Model training, Jetson AGX Thor T5000` 


- **HuggingFace PR Needs Reviewing**: A member requested a review for their [TRL pull request #4894](https://github.com/huggingface/trl/pull/4894) but was advised to wait a couple of days before tagging someone.
   - Another member mentioned that it could take weeks or months to get a review for a PR, but that they always try to circle back on them as soon as possible.
- **Full-Stack Engineer Enters the Chat**: A member introduced themself as a **Senior AI/ML & Full-Stack Engineer**, and listed Key Projects they are currently building, including: Autonomous Agents, Healthcare AI, Decision Support Systems, Conversational AI, Fraud Detection Systems.
   - They listed the technologies used, such as *Reinforcement Learning, NLP, Deep Learning*, and linked back to their profile.
- **Discord app vs discordapp.com**: A member pointed out that the Discord Windows app uses `discordapp.com` instead of `discord.com`, causing issues when jumping to a specific channel from another channel.
   - They also noted that the **help-and-feedback** channel doesn't appear in the channel list because it's a read-only resource page, further explaining the [Discord Support Article](https://support.discord.com/hc/en-us/articles/360042987951-Discordapp-com-is-now-Discord-com) for guidance.
- **Hugging Face's Adam Allegedly Raises $40M**: A member speculated that former Discord mod Adam raised **$40M** and is now a **30 under 30 allumn**, also linking a [Forbes article](https://www.forbes.com/30-under-30/2021/consumer-technology/).
   - Another user found Adam in the `#introductions` channel, confirming his presence.
- **GCP Infra Experiences Replica Surge**: A member reported a bug where their replicas for a private model in **GCP** went over their 1 replica max cap to **62 replicas** overnight, despite no changes to the configuration.
   - The member theorized that they were not the only endpoint that this happened to, and that the **GCP** resources are gone.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1466167293909143918)** (11 messages🔥): 

> `LeetCode MCP server, qwen3-tts-rs, MOLTBOT Quantum Chef, Pacific Prime Math` 


- **LeetCode MCP Serves Coding Challenges**: A member developed a [LeetCode MCP server](https://github.com/SPerekrestova/interactive-leetcode-mcp) to solve daily challenges from the terminal, leveraging **Claude** for learning mode, enabling authentication, problem fetching, hint requests, solution writing, and submission with result retrieval.
- **Qwen3 TTS is Live**: A member released the [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) model, and published install instructions for MacOS, Linux, and Windows.
   - Another member noted *"cool thing heregonna follow back for this one , really interesting thing you managed to do here imho"*.
- **MOLTBOT Serves Up Quantum Cooking**: A dataset containing training examples for quantum computing concepts, generated by the **MOLTBOT ∞ Quantum Chef**, has been shared, featuring prompts about quantum operations with explanations of quantum behavior.
- **Math LLM trained by Pacific Prime**: Pacific Prime has released the first checkpoint of their [math-specialized 1.5B LLM](https://huggingface.co/Pacific-Prime/pacific-prime-math-depth00) trained on **GSM8K**, **NuminaMath**, **MetaMathQA** & **Orca-Math** (~407k samples), and featuring step-by-step reasoning with LaTeX notation.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1466354426276221068)** (1 messages): 

> `LTX-2 distilled checkpoint, Two-stage pipelines, Diffusers library` 


- **Diffusers adds LTX-2 checkpoint support**: The [Diffusers library](https://github.com/huggingface/diffusers) now supports **LTX-2** distilled checkpoint and **two-stage pipelines** following [this pull request](https://github.com/huggingface/diffusers/pull/12934).
- **Two-Stage Pipelines Now Supported**: Thanks to a new update, the [Diffusers library](https://github.com/huggingface/diffusers) now offers support for **two-stage pipelines** along with the **LTX-2** distilled checkpoint.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1466475870201053257)** (1 messages): 

> `Deep RL Course Question, Channel Guidance` 


- **Deep Reinforcement Learning Student Seeks Channel Guidance**: A student enrolled in a **Deep RL course** is seeking guidance on the appropriate channel to ask questions.
   - The student is experiencing difficulty with a problem and is unsure where to seek assistance.
- **Student Needs Clarification on Channels**: A student requires clarification on the correct Discord channel for posting **Deep RL course-related questions**.
   - They are currently unsure about where to seek help for a problem they are encountering.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1466165252591714477)** (44 messages🔥): 

> `Tiny Model Scaling Behavior, Byte-Level Models vs. Tokenization, Thinking AI with Subprocess Models, Routing and Classification for Specialized Models, Retrieval Reliability and Cosine Similarity` 


- ****Dense MoE Architecture** for Byte-Level Prediction**: A member built a **dense MoE architecture** for byte-level prediction, using a vocabulary of **256** (byte-level), **40M parameters**, and **13GB VRAM**, and is seeking feedback on its performance.
   - The model uses a **4096 sequence length** and a **batch size of 8**, and the member asserts they have been careful to avoid contamination and that using bytes allows them to *use the exact same architecture to encode images, or audio, or both*.
- ****Thinking AI** Architecture with Subprocess Models**: A member proposed an architecture where a larger “thinking” AI model is monitored by a smaller subprocess model, which pauses the main model to retrieve information from MCPs or CLIs, then replaces the question with the answer.
   - The goal is to reduce context clutter for the main model, although it's recognized that the subprocess model needs to know what information the main model is missing, and it was described as *probably a dumb idea*.
- ****Routing and Classification** Boost Model Performance**: Members discussed using a classifier to route user prompts to specialized models, appending the detail to the context of the user prompt, which avoids pausing the larger model and reduces token overhead.
   - There was further discussion on making the classifier and embedding model the same, processing embeddings directly with the LM and specialist model, with one member saying *routing and classification would likely be the spiciest move*.
- **Retrieval and **Cosine Similarity** Struggles**: Members discussed the problem of retrieval being unreliable and confusing to models, and that cosine similarity might not equal causal relevance.
   - One member suggested indexing a SQL database across a model, with the member posting *the biggest issue with retrieval imo is that cosine similarity != causal relevance*.
- **Videos touch on **vector weighting by importance****: One member asked about weighting vectors by importance for cosine similarity and shared [two Xitter posts](https://fxtwitter.com/i/status/2016903019116249205) and [this YouTube video](https://youtu.be/K5WPr5dtne0?si=-TCsfNXDKAINCyuv).
   - Another member shared [another Xitter post](https://fxtwitter.com/i/status/1924135806953787433) sharing their love for the project.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1466167059690815774)** (8 messages🔥): 

> `LM Studio 0.4.0, AllenAI's Open Coding Agents, GoodfireAI Alzheimer's Detection Interpretability, Sweep Next-Edit Autocomplete Model, Google DeepMind Project Genie` 


- **LM Studio Gets an Upgrade to v0.4.0**: A new [LM Studio version 0.4.0](https://lmstudio.ai/blog/0.4.0) was released.
   - No details were provided on the specific improvements or features included in the update.
- **AllenAI Sells Open Coding Agents for Cheap**: [AllenAI's SERA](https://allenai.org/blog/open-coding-agents) brings open coding agents to private repos for as little as **$400** in training costs, according to [The Decoder](https://the-decoder.com/allen-ais-sera-brings-open-coding-agents-to-private-repos-for-as-little-as-400-in-training-costs/).
- **GoodfireAI Interprets Alzheimer's Detection**: A member linked to [GoodfireAI's research](https://www.goodfire.ai/research/interpretability-for-alzheimers-detection#) on interpretability for Alzheimer's detection.
- **Sweep Opens Sourcing a Next-Edit Autocomplete Model**: Sweep is open sourcing **Sweep Next-Edit**, a locally runnable **SOTA LLM** for next-edit autocompletion, models with 0.5B and 1.5B parameters have been released, see [Sweep's blog](https://blog.sweep.dev/posts/oss-next-edit).
- **DeepMind Genies Out a New Project**: A member shared a link to [Google DeepMind's Project Genie](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1466178062864617656)** (21 messages🔥): 

> `Minecraft launcher, Prompt engineering, Manus redeem codes, AI/ML and Full-Stack Engineering, Libyan user` 


- **Minecraft Launcher created for AFK usage**: A user is creating *a Minecraft launcher* to be able to AFK without needing a *good PC*.
   - The user added that they can do *prompt engineering* and data extraction too, even *reproducing a website* if needed.
- **New Manus Redeem Codes Posted**: A user has shared three new **Manus redeem codes**: [FUM1A1G7](https://manus.im/redeem?c=FUM1A1G7), [ntaxzjg](https://manus.im/redeem?c=ntaxzjg), and [mwiyytb](https://manus.im/redeem?c=mwiyytb).
   - Another user thanked him for the code, with confirmation that *only one code can be used per month*.
- **AI/ML Engineer Seeking Collaborations**: An engineer introduced themself, highlighting their expertise in building **AI + full-stack systems** and directing collaboration offers to the **#collab channel**.
   - They listed experience with **LLM integration, RAG pipelines, workflow automation, AI content moderation, Image AI (CLIP + YOLOv8), Voice AI (Whisper, Tacotron2)** and various web, mobile, and database technologies.
- **Libyan Manos User since 2025 App Launch**: A user asked if they were the only person from **Libya** who had used **Manos** since its launch in **early 2025**.
   - Another user then welcomed the **Libyan** user, with a *حياك الله*.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1466447640509284628)** (4 messages): 

> `MCP Security Standard, Agentic AI Foundation, Server Hardening, Supply Chain Security` 


- ****MCP Security Standard** Drafted and Up For Discussion**: A security researcher named Dani (aka cr0hn) has drafted an open security baseline for MCP servers, including controls for **hardening, logging, access control, and supply chain security** at [https://github.com/mcp-security-standard/mcp-server-security-standard](https://github.com/mcp-security-standard/mcp-server-security-standard).
   - The author is planning to donate it to the **Agentic AI Foundation** and wants to know if it fits within the **MCP ecosystem**.
- **Community Praises the **MCP Security Standard****: One member found the **MCP Security Standard's** controls and domains to be well-written and navigable.
   - They forwarded the baseline to another channel for further discussion.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1466446684241264897)** (13 messages🔥): 

> `State machine in lifecycle doc, Namespaces vs Groups vs URIs in MCP, SEP-2084 Primitive Grouping` 


- ****State Machine** Doc Under Review**: A member asked for feedback on [this pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2174) to add a state machine inside the lifecycle doc.
   - Another member suggested filling in the motivation and context sections to help reviewers understand the proposed changes.
- ****Namespaces** Yield to **Groups** in MCP**: It was noted that Namespaces were rejected and superseded by Groups, while the status of **URIs** is less clear, referencing [issue 1292](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1292).
   - A new **SEP** regarding groups was published recently and is under discussion: [Primitive Grouping SEP-2084](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2084).
- **SEP-1300's Refined Descendant is SEP-2084**: **SEP-1292** was superseded by **SEP-1300** but it was rejected due to lack of consensus during a Core Maintainers review.
   - The simpler [SEP-2084 - Primitive Grouping](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2084) was introduced as a replacement.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1466529662804037845)** (5 messages): 

> `Browser GPU Usage, IGPU performance` 


- **Twitter Link Elicits Affection**: A user expressed their appreciation for [a Twitter link](https://fxtwitter.com/i/status/1924135806953787433).
- **IGPU Struggles with Basic Browser Page**: One user noticed a performance hit, *getting 3fps*, when viewing a specific webpage while using a **Ryzen 7 7700 IGPU**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1466167224535613563)** (8 messages🔥): 

> `Bucketing Scales, Geometric Convolution Approach, Layer Norms, RNN Architectures` 


- **Bucketing Scales Differently**: A member inquired if *bucketing scales differently* when combined with other operations, though no specific examples were provided in this message history.
- **Geometric Convolution Approach acts as cell connections**: A member is working on a baseline that replaces **Multi-Head Attention** with a [geometric convolution approach](https://github.com/MrPan2048/GeometricTransformer) where embeddings act as cells simulating connections.
   - The member reported loss is converging and starting to capture dialogue logic, and is looking for feedback.  They gave a debug print of the following `DEBUG [GEOPARA] | L0_Alpha: 0.1029 L1_Alpha: 0.0947 | L0_Res: 0.0916 L1_Res: 0.1538`
- **Discussion about Layer Norms Commence**: A member posted a link to a tweet about [layer norms](https://fxtwitter.com/i/status/2016505314183385244) and a link to [arxiv.org](https://arxiv.org/abs/2601.19831).
- **Parallelizable RNN Architectures suggested**: A member suggested reading more about other parallelizable **RNN architectures** and doing more rigorous larger scale experiments against a strong tokenized baseline.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

diogenesoftoronto: https://arxiv.org/abs/2601.19831
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1466419877991551087)** (2 messages): 

> `Malware Classification, False Positive Rate Reduction, Explainable Models, Feature Engineering Techniques` 


- **Tackling Malware Classification Problem**: A member is working on a **malware classification problem** with a dataset of around **600K** rows and **9,600** binary features, aiming to reduce the **false positive rate (FPR)**.
   - The member is primarily focused on **explainable models** like scikit-learn trees, but they are struggling to reduce the FPR below 9% despite trying several feature engineering techniques.
- **Seeking Help with False Positive Rate**: A member seeks advice on reducing **false positive rate (FPR)** in their malware classification project, which uses approximately **9,600** binary features.
   - Despite trying various **feature engineering techniques** and experimenting with neural networks (which performed best), the member is looking for suggestions to reduce the FPR while maintaining explainability, especially with scikit-learn trees.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

ash_blanc: https://alphaxiv.org/abs/2601.20810
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1466188672780210206)** (12 messages🔥): 

> `DSPy Skills, DSPy Agents in Production, RLM Sandbox Swapping, Opus Capabilities` 


- **Custom Skills in DSPy Sought**: A member inquired about using custom skills (**.md files with associated .py scripts**) within the DSPy module, specifically with a DSPy ReAct agent.
   - They mentioned having skills like converting **.md to PDF** and wanted to know if others have tried similar approaches.
- **DSPy Agents Deployed Remotely**: A member asked about using **DSPy agents in production remotely** with **DSPy optimizations in runtime**.
   - The member expressed the need for a runtime environment to support such deployments.
- **RLM Sandbox Customization Explored**: A member inquired about swapping the sandbox used by **RLM (Retrieval-augmented Language Model)** with custom or cloud services like **E2B (Ephemeral Environment Builder)**.
   - They sought to replace the local PythonInterpreter with sandboxes like **E2B, Modal, or Daytona**.
- **Opus to Author Sandboxes Soon**: A member announced that they are working on allowing **Opus** to write new sandboxes.
   - They noted that there would be **a protocol for official implementations** from providers such as E2B in the future.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1466175632005398590)** (1 messages): 

> `macOS trust dance, Gatekeeper tax, mojo build codesign` 


- **macOS Trust Dance on First Run**: The performance delta between first and second runs on macOS may be due to macOS's **trust dance** rather than a **Mojo-specific** issue.
   - The *Gatekeeper tax* can add overhead, but clearing the quarantine xattr or ad-hoc codesigning usually makes the startup behave like the second run.
- **Codesigning step in mojo build hides startup delays**: For CLI tooling, startup performance is crucial, suggesting potential footgun issues with **docs** or **tooling**.
   - Adding a **codesign** step in `mojo build` might completely mitigate this problem, ensuring consistent startup behavior.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1466242909522362368)** (8 messages🔥): 

> `Mojo at ORNL, Mojo GPU puzzles, Mojo changelog, Github issue #5875, Github issue #4767` 


- **Mojo Gets ORNL Recognition**: A research paper titled [Mojo at ORNL](https://arxiv.org/html/2509.21039v1) has been published.
- **Modular Bug Hunt**: A member reported a potential bug and suggested filing an issue, possibly related to [issue #4767](https://github.com/modular/modular/issues/4767).
- **Mojo Changlog Sparks Excitement**: A member expressed surprise at the amount of work done on Mojo in the past two months after reviewing the new Mojo changelog.
- **Mojo GPU Puzzles - Guard Clause not Needed**: A member noticed that the guard `if row < size and col < size:` is unnecessary in Mojo GPU puzzles 3, 4, and 5; omitting it doesn't cause errors.
   - Another member pointed to the solution of [puzzle 03](https://puzzles.modular.com/puzzle_03/puzzle_03.html) which explained that passing the tests doesn’t necessarily mean the code is sound.
- **Github Issue Reported**: A member reported encountering a weird issue, referencing [GitHub issue #5875](https://github.com/modular/modular/issues/5875).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1466299210411544629)** (5 messages): 

> `ANE Performance, Quantization Discussion` 


- **ANE Tradeoffs Focus on Power Efficiency**: Apple's **ANE** (Apple Neural Engine) prioritizes performance-to-watt tradeoffs, not raw performance, as detailed in [this paper](https://arxiv.org/abs/2511.13450).
   - Although the **ANE** achieves competitive performance with superior energy efficiency, it delivers *up to 3.8 TFlops on the M4-Pro*, which is comparable to the **GPU's 4.7 TFlops** on the same SoC for GEMM operations.
- **Quantization methods examined**: Members discussed quantization and specifically **Q4** as a quantization method.
   - One member got it up to *9 t/s*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1466188969367699764)** (5 messages): 

> `Aider Friendly Fork, Aider Utility with Python and Git` 


- **Aider Sparking Friendly Fork Frenzy?**: A member suggested creating a friendly fork of Aider to continue development while @paulg is busy with other projects, given that Aider is written in **Python** and uses **Git** for version control on **GitHub**.
   - The goal is to build upon the current functionality of Aider, acknowledging its utility compared to other tools.
- **Aider Integrates with Orchestrators**: A member expressed interest in driving Aider from orchestrators like **MultiClaude** or **gas town.sh**.
   - This highlights Aider's potential integration with other tools for enhanced workflow automation.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1466444058514821308)** (1 messages): 

> `Context Graphs, Semantic Layers, Ontologies, AI Reasoning` 


- **Context Graphs Sparking Confusion**: The proliferation of **context graphs** has led to confusion, with terms like **semantic layers** and **ontologies** being used interchangeably, even though they serve distinct purposes.
   - AI's emergence is highlighting the inconsistencies in these definitions, revealing that systems can perform calculations accurately but still exhibit poor reasoning due to the conflation of these concepts.
- **Semantic Layers vs. Ontologies**: While **semantic layers** standardize metrics, **ontologies** model meaning, originating from different fields such as BI and healthcare, respectively.
   - A recent [article in Metadata Weekly](https://metadataweekly.substack.com/p/ontologies-context-graphs-and-semantic) points out that AI's requirements extend beyond mere definitions, necessitating explicit relationships, constraints, and assumptions.
- **Semantic Layers Limitations with AI**: The article suggests that the approach of *"just add a semantic layer"* is proving insufficient because AI demands more than just data consistency.
   - AI needs reasoning, which **ontologies** support by making relationships and assumptions explicit, contrasting with the traditional use of semantic layers optimized for dashboards and reporting.
- **YAML Configurations Crumble Under Business Meaning**: The author Jessica Talisman argues that **YAML configurations** falter when they are expected to represent business meaning.
   - She breaks down what semantic layers were designed for, why ontologies support reasoning, what context graphs are and the limitation of YAML in representing the business meaning.


  

---


---


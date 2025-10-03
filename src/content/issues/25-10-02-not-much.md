---
id: MjAyNS0x
title: not much happened today
date: '2025-10-02T05:44:39.731046Z'
description: >-
  **Kling 2.5 Turbo** leads in text-to-video and image-to-video generation with
  competitive pricing. **OpenAI Sora 2** shows strong instruction-following but
  has physics inconsistencies. **Google Gemini 2.5 Flash** "Nano Banana" image
  generation is now generally available with multi-image blending and flexible
  aspect ratios. **IBM Granite 4.0** introduces a hybrid Mamba/Transformer
  architecture with large context windows and strong token efficiency,
  outperforming some peers on the Intelligence Index. **Qwen** models receive
  updates including fine-tuning API support and improved vision capabilities.
  **Tinker** offers a flexible fine-tuning API supporting LoRA sharing and
  CPU-only training loops. The ecosystem also sees updates like **Synthesia
  3.0** adding video agents.
companies:
  - openai
  - google
  - ibm
  - alibaba
  - kling_ai
  - synthesia
  - ollama
  - huggingface
  - arena
  - artificialanalysis
  - tinker
  - scaling01
models:
  - kling-2.5-turbo
  - sora-2
  - gemini-2.5-flash
  - granite-4.0
  - qwen-3
  - qwen-image-2509
  - qwen3-vl-235b
topics:
  - video-generation
  - instruction-following
  - physics-simulation
  - image-generation
  - model-architecture
  - mixture-of-experts
  - context-windows
  - token-efficiency
  - fine-tuning
  - lora
  - cpu-training
  - model-benchmarking
  - api
  - workflow-automation
people:
  - artificialanlys
  - kling_ai
  - altryne
  - teortaxestex
  - fofrai
  - tim_dettmers
  - sundarpichai
  - officiallogank
  - andrew_n_carr
  - googleaidevs
  - clementdelangue
  - wzhao_nlp
  - alibaba_qwen
  - scaling01
  - ollama
---


**a quiet day**

> AI News for 10/1/2025-10/2/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (196 channels, and 8860 messages) for you. Estimated reading time saved (at 200wpm): 629 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

It's a quiet day so you can check out the latest [Latent Space with Dylan Field](https://www.latent.space/p/figma) pod!

Also, invites for the [first AI Engineer Code Summit](https://apply.ai.engineer/) have started going out.

---

# AI Twitter Recap

**Video generation: Sora 2, Kling 2.5 Turbo, and Google’s “Nano Banana” GA**

- **Kling 2.5 Turbo (Text/Image→Video)**: The latest from Kling tops the Artificial Analysis Video Arena for both text-to-video and image-to-video, edging Hailuo 02 Pro, Google’s Veo 3, and Luma Ray 3. It generates 5s/10s clips up to 1080p. Notable economics: ~$4.20/min on FAL API vs $4.90 for Hailuo 02 Pro and ~$7.32 for Seedance 1.0, and ~15¢ per video on Kling’s Ultra plan via app credits. See model comparisons and pricing in the Arena thread from [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1973570493753204953) and Kling’s announcement [@Kling_ai](https://twitter.com/Kling_ai/status/1973581864679121374).
- **OpenAI Sora 2: capability vs. correctness**: Live usage shows impressive instruction-following and in-app remixing, but critical evaluations flag physics inconsistencies and marketing polish. See a broad demo roundup [@altryne](https://twitter.com/altryne/status/1973568567489798144), critiques on “people-pleasing” over physical fidelity [@teortaxesTex](https://twitter.com/teortaxesTex/status/1973570902609805711), and targeted tests where Sora 2 fails physics scenarios that Veo 3 handles better (audio narration correct) [@fofrAI](https://twitter.com/fofrAI/status/1973745038195830891), plus a sober overview [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1973728079395856396).
- **Google Gemini 2.5 Flash Image (“Nano Banana”) GA**: Now production-ready with 10 aspect ratios, multi-image blending, and image-only output. Pricing: $0.039/image on Gemini API (AI Studio + Vertex). Announcements from [@sundarpichai](https://twitter.com/sundarpichai/status/1973788714758517147), [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1973836478989152700), and [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1973790388722061394). Also integrated into partner products (e.g., Cartwheel’s new motion pipeline) [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1973875941651964335) and showcased by Google’s developer account [@googleaidevs](https://twitter.com/googleaidevs/status/1973781293977735435).
- **Ecosystem**: Synthesia 3.0 adds “video agents” and new workflows [@synthesiaIO](https://twitter.com/synthesiaIO/status/1973688529818620193).

**Open-weight model releases: IBM Granite 4.0 and Qwen updates**

- **IBM Granite 4.0 (Apache 2.0, hybrid Mamba/Transformer)**: IBM’s new family mixes a minority of standard attention layers with majority Mamba layers to cut memory without large accuracy hits. Sizes include Granite 4.0 H Small (MoE 32B/9B active), H Tiny (7B/1B), H Micro (3B/3B) and a 3B dense Micro variant. Key specs: 128K context, Apache 2.0, strong token efficiency. Artificial Analysis measures H Small at 23 on its Intelligence Index (non-reasoning), ahead of Gemma 3 27B (22) and behind Mistral Small 3.2 (29), EXAONE 4.0 32B (30), and Qwen3 30B A3B (37). Micro scores 16, edging Gemma 3 4B (15). Granite is on HuggingFace and Replicate (H Small at $0.06/$0.25 per 1M in/out tokens). Benchmarks: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1973746432692936963). Ollama released runnable images for Micro/Micro-H/Tiny-H/Small-H [@ollama](https://twitter.com/ollama/status/1973782095811219574). IBM Granite is also added to LM Arena [@arena](https://twitter.com/arena/status/1973892502458650697), and HF’s [@ClementDelangue](https://twitter.com/ClementDelangue/status/1973798540389355903) highlights browser/WebGPU demos and HF Enterprise onboarding.
- **Qwen updates**: Qwen models are among the first supported by Tinker’s fine-tuning API [@wzhao_nlp](https://twitter.com/wzhao_nlp/status/1973603599616974970), and the Qwen team notes expanded support and open releases [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1973665010615218421). Qwen-Image-2509 improves consistency [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1973668568412856595); Qwen3 VL 235B is reported as performant at lower cost for some vision tasks [@scaling01](https://twitter.com/scaling01/status/1973777774121984175).

**Fine‑tuning and systems: Tinker, rank‑1 LoRA, MoE support, and inference speedups**

- **Tinker: a flexible fine-tuning API with LoRA sharing**: Thinking Machines’ Tinker lets you write a CPU-only training loop and run it unchanged on distributed GPUs, keeping control over algorithms/losses while Tinker manages scheduling, resource allocation, and failures. It supports open models (Llama, Qwen) including large MoE (e.g., Qwen3-235B), and implements LoRA for efficient resource sharing. Summaries: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1973827605448306883), release note [@Smol_AI](https://twitter.com/Smol_AI/status/1973622595124863044), cookbook/docs: [link](https://twitter.com/TheTuringPost/status/1973827618442260655).
- **LoRA without regrets (rank=1)**: Multiple replications show rank-1 LoRA can match full fine-tuning quality on reasoning tasks while saving ~43% VRAM, enabling RL on larger models; see results and code [@zzlccc](https://twitter.com/zzlccc/status/1973612326747336767) and a Colab on Qwen3-0.6B OpenR1-Math [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1973776491843297386). See guidance from “LoRA Without Regret” [@TheTuringPost](https://twitter.com/TheTuringPost/status/1973820885116334441).
- **MoE training and infra**: Prime-RL now supports MoE for RL and SFT (Qwen3 A3-30B, GLM series, Moonlight), with significant modeling rewrites to stay Torch Compile compatible while retaining HF ecosystem compatibility [@samsja19](https://twitter.com/samsja19/status/1973624615768674612). On inference, [@vikhyatk](https://twitter.com/vikhyatk/status/1973884858574491819) reports a new engine with 1.3–20x faster completions; production uses QAT for FP8 KV caches and MoE weights (engine proprietary for now). For local/dev infra: MI300X VMs on-demand at $1.99/GPU/hr [@HotAisle](https://twitter.com/HotAisle/status/1973768786965639643), vLLM now supports BERT [@vllm_project](https://twitter.com/vllm_project/status/1973805307878142297).

**RL and reasoning: search‑in‑training, broadened exploration, latent CoT, front‑loaded reasoning**

- **Train-time search and efficient exploration**: DeepSearch moves MCTS into the training loop with Tree‑GRPO stabilization and efficient caching/filtering, reaching 62.95% on AIME/AMC with ~330 GPU hours (beating a Nemotron baseline and outpacing standard RL that plateaus even with 1800+ GPU hours) [@omarsar0](https://twitter.com/omarsar0/status/1973781658772951320). BroRL scales exploration by increasing rollouts per example into the hundreds, overcoming the saturation seen when only scaling training steps [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1973717761693217241).
- **Architectures and training mechanics**: A new latent CoT method “thoughtbubbles” inserts input‑adaptive latent tokens to allocate more compute without CoT labels, improving perplexity and compute use [@houjun_liu](https://twitter.com/houjun_liu/status/1973778517427937323) with positive reaction [@khoomeik](https://twitter.com/khoomeik/status/1973785079932727760). NVIDIA’s “Front‑Loading Reasoning” finds injecting reasoning during pretraining yields durable gains that finetuning can’t recover [@__SyedaAkter](https://twitter.com/__SyedaAkter/status/1973841632249172096). A small but impactful MoE tweak—global‑batch load balancing (vs micro-batch) —yields lower perplexity and clearer expert specialization with minimal code changes [@daddyofadoggy](https://twitter.com/daddyofadoggy/status/1973759113554174251). For sparse diffusion LMs, OpenMoE 2 studies expert‑choice MoE × diffusion across wide FLOPs/param regimes, claiming perfect load balance (no aux loss), +20% throughput, and adaptive compute under multi‑epoch training [@NiJinjie](https://twitter.com/NiJinjie/status/1973747616082186349).

**Agents and toolchains: CLI + semantic search, Notebook MCP, browsers, and CLIs**

- **CLI agents + semantic search beat pure CLI**: LlamaIndex’s SemTools benchmark (1,000 arXiv papers) shows agents with semantic search produce more complete answers across question types versus agents using only CLI tools; Unix tools remain a strong baseline and SemTools integrates parse (LlamaParse) and semantic search directly into command-line agents (Claude/Gemini CLIs). Results/methodology: [@llama_index](https://twitter.com/llama_index/status/1973783798044307741).
- **Executing notebooks via MCP**: Goodfire open-sourced Scribe, an MCP-based system enabling agents to run notebook cells and receive Jupyter outputs (text/errors/images). They share lessons on “experimenter agents” vs “software development agents” and the scaffolding needed for scientific workflows [@GoodfireAI](https://twitter.com/GoodfireAI/status/1973789154174754877), [blog](https://twitter.com/GoodfireAI/status/1973789166019482035).
- **“AI browsers” and evaluators**: Perplexity’s Comet is now GA globally, with Comet Plus launching alongside major publisher partnerships; Pro/Max users get Plus bundled [@perplexity_ai](https://twitter.com/perplexity_ai/status/1973795224960032857), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1973804332039786608). Yupp’s “Help Me Choose” orchestrates a third model to critique two candidate answers, then has them analyze each other before the user picks — an interesting pattern for adjudication [@yupp_ai](https://twitter.com/yupp_ai/status/1973882910907470237), [@lintool](https://twitter.com/lintool/status/1973874173157257485). Google’s Jules Tools brings an agentic CLI (npm installable) mirroring browser capabilities [@julesagent](https://twitter.com/julesagent/status/1973812188977508755).

**Leaderboards and real‑world coding agent metrics**

- **Claude Sonnet 4.5 tied for #1 on LM Arena**: Sonnet 4.5 reaches the top slot alongside Claude Opus 4.1, with strong showings across categories including coding and creative writing (rankings are from tens of thousands of human votes) [@arena](https://twitter.com/arena/status/1973828836510085385). Community reports suggest Anthropic continues to ship very competitive coding models [@scaling01](https://twitter.com/scaling01/status/1973836516205134135).
- **Open source is closing in for code editing agents**: In Cline’s diff‑edit success tests, GLM‑4.6 achieves 94.9% vs Claude 4.5’s 96.2% at ~10% of the cost; users report switching workflows accordingly [@cline](https://twitter.com/cline/status/1973870619013136850), [@nickbaumann_](https://twitter.com/nickbaumann_/status/1973846157886697771).
- **Video Arena reminder**: Kling 2.5 Turbo leads both T2V and I2V; details above in the Video section [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1973570493753204953).

**Top tweets (by engagement)**

- [“We are, in so many ways, literally pretrained models.”](https://twitter.com/cloneofsimo/status/1973655922506605046) by [@cloneofsimo](https://twitter.com/cloneofsimo) — 4,967
- [Perplexity Comet GA worldwide](https://twitter.com/perplexity_ai/status/1973795224960032857) by [@perplexity_ai](https://twitter.com/perplexity_ai) — 2,667
- [Anthropic’s “thinking” campaign praise and adoption](https://twitter.com/signulll/status/1973828026761695439) by [@signulll](https://twitter.com/signulll) — 2,441
- [Nano Banana GA announcement](https://twitter.com/sundarpichai/status/1973788714758517147) by [@sundarpichai](https://twitter.com/sundarpichai) — 1,576
- [“Iteration speed is a superpower”](https://twitter.com/gdb/status/1973864268350255366) by [@gdb](https://twitter.com/gdb) — 1,989

---

# AI Reddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Sora 2 and WAN 2.2 Video Generation Demos

- [**Sora 2 is insanely good at stand up comedy**](https://www.reddit.com/r/ChatGPT/comments/1nwaowu/sora_2_is_insanely_good_at_stand_up_comedy/) (Activity: 437): **The post claims a stand-up comedy clip was generated by “Sora 2,” presumably referring to OpenAI’s Sora text-to-video model ([overview](https://openai.com/sora)). Viewers report highly natural comedic timing and facial-expression sync, implying strong temporal coherence, phoneme–viseme alignment, and fine-grained gesture/micro‑expression control; however, the linked video is inaccessible (**`HTTP 403`**), so provenance, model versioning (“2”), prompts, seeds, or generation parameters cannot be verified from the post.** Commenters overwhelmingly praise the realism—“uncanny” timing and natural delivery—and some compare it favorably to human comedians, while at least one asks if it truly came from Sora, highlighting skepticism due to lack of proof or technical details.
    - Multiple users highlight the "uncanny" timing between delivery and facial expressions, implying strong audiovisual prosody alignment and keyframe-level gesture/lip-sync. If this is native Sora 2 output, it suggests improved temporal conditioning (beat-aligned micro-expressions, head/eyebrow cues) and actor-like pose control versus prior text-to-video baselines.
    - One commenter notes the joke is not original, attributing it to **Joan Rivers** with a direct quote reference, raising concerns about memorization/regurgitation from training data or prompt-sourced material rather than novel synthesis. This points to content provenance and originality risks in generative video models; see attribution: https://www.imdb.com/name/nm0001672/quotes/ .
    - Skepticism that this "really came from Sora" flags verification/provenance issues for AI-generated clips (possible editing, dubbing, or pipeline mixing). Technical readers may look for reproducibility details (prompt, seed, runtime), metadata/watermarking, or Content Credentials to validate the generation chain and rule out post-production augmentation.
- [**WAN 2.2 Animate - Character Replacement Test**](https://www.reddit.com/r/StableDiffusion/comments/1nvvo7g/wan_22_animate_character_replacement_test/) (Activity: 1439): **OP showcases a character-replacement test using WAN 2.2 Animate on clips from the film [The Ninth Gate](https://en.wikipedia.org/wiki/The_Ninth_Gate), achieving convincing identity substitution while noting outfit inconsistency because the reference image covered only the head/upper torso (indicating apparel continuity depends on conditioning coverage). The shared video link is a [Reddit host](https://v.redd.it/e2hf1vuf0nsf1) that returned** `HTTP 403` **in external fetch attempts (likely requires login).** Commenters emphasize that while the rendering style/quality is mediocre, the integration/substitution is “absolutely amazing.” Technical critiques flag lighting mismatches and weak hand fidelity when the region is small, and one asks how long sequences are produced with WAN 2.2 Animate; overall sentiment is that it’s a strong demonstration of AI-driven VFX potential.
    - Commenters note that despite modest render/style fidelity, the core character integration/substitution is impressively stable—tracking and alignment hold up well—suggesting **WAN 2.2 Animate** is viable for FX-style character replacement even when aesthetic polish is lacking.
    - Technical critiques focus on lighting and small-detail fidelity: one says *“Lighting sucks!”* and another notes the hands in the first shot are *“too small on screen to be properly generated/tracked,”* reflecting a common failure mode where tiny features lose detail or tracking robustness.
    - There’s demand for the exact workflow (pipeline and clip-length method). A concrete suggestion is to use a **relight LoRA** to fix illumination mismatches; others ask how the video was extended, indicating interest in techniques for lengthening sequences while maintaining temporal consistency.

### 2. OpenAI $500B Valuation + ChatGPT 'Think Longer' UX + Silicon Valley Foresight

- [**OpenAI Valuation Soars to $500 Billion, Topping Musk’s SpaceX**](https://www.reddit.com/r/OpenAI/comments/1nw36bw/openai_valuation_soars_to_500_billion_topping/) (Activity: 720): **Post claims OpenAI’s private valuation has reached ~**`$500B`**, surpassing SpaceX, with commenters citing projected** `2025` **figures of ~**`$4.3B` **revenue against ~**`$6.8B` **losses—implying very high revenue multiples and deeply negative operating margins. Technical concerns raised include perceived model quality regression (e.g., *“GPTs deteriorate”*) and an enterprise “AI reality check” as competitive pressure from both closed- and open-source models intensifies. An accompanying meme/image underscores skepticism about sustainability ([image](https://preview.redd.it/5phkrh1xfpsf1.jpeg?width=1270&format=pjpg&auto=webp&s=44308b7a5984ab2d905bc1684742a927ae0aa0c0)).** Top comments characterize the valuation as a bubble given negative unit economics and crowded competition, arguing many AI vendors may not survive. Others echo that current systems underdeliver versus expectations, citing degradation and unmet enterprise use cases.
    - Financials/valuation concern: commenters cite ~`$4.3B` 2025 revenue vs ~`$6.8B` losses and a ~`$500B` valuation, implying ~`>100x` forward sales and deeply negative margins for a compute-intensive business. This raises questions about the sustainability of subsidized inference, future price hikes, or cost reductions needed (e.g., model distillation, batching, custom silicon) to justify the multiple without impairing product quality.
    - Model reliability/regression: reports of GPT “deterioration” are tied to known behavior drift issues, where model updates change outputs and quality over time. Prior analyses found sizable month-to-month variance in GPT-4’s reasoning/accuracy (e.g., Stanford/UC Berkeley’s “How is ChatGPT’s Behavior Changing over Time?” showing swings on coding/math tasks: https://arxiv.org/abs/2307.09009), underscoring maintenance/evaluation challenges for production deployments.
    - Competitive pressure: the thread notes both free and paid alternatives narrowing the gap, which could compress pricing power. Public evals like LMSYS Chatbot Arena show non-OpenAI leaders (e.g., **Claude 3.5 Sonnet**, **Gemini 1.5 Pro**, **Llama 3 70B**, **Mistral Large**) clustered near the top (https://lmsys.org/blog/2024-06-20-arena-hard/), indicating potential commoditization of frontier capabilities and weakening moat assumptions.
- [**CAN WE PLEASE HAVE A DISABLE FUNCTION ON THIS**](https://www.reddit.com/r/ChatGPT/comments/1nw4ttx/can_we_please_have_a_disable_function_on_this/) (Activity: 1478): **User requests a toggle to disable the chat UI’s “Thinking longer for a better answer” behavior/overlay, reporting it triggers on every prompt even when not in a “Think Longer” mode—suggesting a UX issue or misconfiguration. Comments point out an existing “Instant” setting and that for “thinking” models you can manually choose between “standard” and “extended” thinking, implying the feature is configurable but possibly confusing or inconsistently applied.** Commenters split between joking about impatience and a practical note that the instant/standard/extended controls already exist; the thread implicitly debates whether this is a UX bug vs. user settings awareness.
    - Existing UI controls already let users tune or avoid slower deliberate reasoning: one commenter asks, *“Are you not aware of the 'instant' setting? And if you select the 'thinking' model, you can manually choose between 'standard' and 'extended' thinking.”* This implies a configurable latency/quality trade-off where `instant` minimizes delay, `standard` balances speed and reasoning, and `extended` maximizes depth at higher latency.
    - A power user reports defaulting to thinking mode and even choosing `extended` on desktop, reserving faster modes for trivial lookups: *“uses thinking mode by default for nearly all prompts… on desktop even select the 'extended' thinking option.”* This reinforces a workflow pattern: complex tasks benefit from longer deliberate runs, while simple factual queries are better served by low-latency modes.
- [**Bro how was the show Silicon Valley so consistently 10 years ahead of its time?**](https://www.reddit.com/r/ChatGPT/comments/1nw0eo1/bro_how_was_the_show_silicon_valley_so/) (Activity: 8183): **The thread asks why HBO’s “Silicon Valley” felt a decade ahead of reality; top replies credit the show’s accuracy to hiring actual engineers/technical advisors in the writers’ room, which grounded portrayals of startup dynamics, infrastructure trade-offs, and compression research. As a concrete example, commenters point to the S1 finale’s mathematically worked-through optimization derivation (see this clip: https://www.youtube.com/watch?v=Tx3wDTzqDTs) as evidence of rigor beyond typical sitcom writing. Note: the referenced [v.redd.it](http://v.redd.it/) asset returns** `403 Forbidden` **without authentication—access requires a logged-in session or authorized Reddit API client.** Veteran practitioners describe the series as effectively a “documentary,” arguing its prescience stems from embedding real tech people in the creative process rather than relying on generic tech tropes.
    - Technical authenticity likely came from hiring actual engineers as writers/consultants, which helps seed plots with real failure modes (scaling bottlenecks, deployment mishaps, VC/IP constraints) and accurate jargon/tooling rather than generic “hacker” tropes. That kind of domain input lets writers plausibly extrapolate near‑term ML/infra trends (instead of sci‑fi leaps), making storylines feel *imminent* rather than speculative.
    - The “Hot Dog / Not Hot Dog” gag maps to binary classification, which traces back to the **perceptron** (Rosenblatt, `1957`)—a linear classifier with well‑known limits formalized by **Minsky & Papert** in `1969` ([Perceptron](https://en.wikipedia.org/wiki/Perceptron), [Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book))). A real image‑based Not‑Hotdog app would typically rely on multi‑layer nets (e.g., **CNNs**) trained with backprop (popularized in `1986`) to learn non‑linear decision boundaries and visual features ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network), [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)). Conceptually it’s the same task—binary classification—but the implementation leap from a single‑layer perceptron to modern deep nets is substantial (data scale, compute, and model capacity).

### 3. AI Comedy Threads: 'Strangest Flea Market' Pt.7 and Related Skits

- [**What do you sell at The Strangest Flea Market? Pt. 7**](https://www.reddit.com/r/aivideo/comments/1nvr8hw/what_do_you_sell_at_the_strangest_flea_market_pt_7/) (Activity: 477): **Recurring creative/comedy series post (“What do you sell at The Strangest Flea Market? Pt. 7”) with a Reddit-hosted video link** `v.redd.it/x8rnhfkoulsf1` **that returns HTTP** `403 Forbidden` **and a Reddit network-security block page requiring login/OAuth, indicating application-layer access controls (session/cookie or OAuth gating) and likely CDN/bot protections. Comment references suggest serialized running gags (pig Latin bits; a “Korean-speaking vegetable”), but the primary media asset is inaccessible without authenticated session or API token.** Comments are uniformly positive and request expansion of the “Korean-speaking vegetable” motif; no technical debate present.
- [**What do you sell at The Strangest Flea Market? Pt. 7**](https://www.reddit.com/r/aivideo/comments/1nvr8hw/what_do_you_sell_at_the_strangest_flea_market_pt_7/) (Activity: 475): **Short-form comedy sketch post “What do you sell at The Strangest Flea Market? Pt. 7,” hosted on Reddit video ([v.redd.it](http://v.redd.it/)), is currently inaccessible to unauthenticated clients (**`HTTP 403 Forbidden`**, OAuth required). From comments, the piece is part of a recurring surreal/absurdist series and includes a Pig Latin wordplay gag and an explicit nod to Tim Robinson’s “drive‑thru” bit from I Think You Should Leave ([show info](https://en.wikipedia.org/wiki/I_Think_You_Should_Leave_with_Tim_Robinson)).** Commentary is uniformly positive; the only technically notable observation is the intertextual reference to Tim Robinson’s sketch style and the inclusion of Pig Latin as a stylistic device.
    - A creator highlights compositional and control limits in current image models—specifically naming **Midjourney** ([https://www.midjourney.com](https://www.midjourney.com/)), “Seedream,” and **FLUX** (e.g., [https://huggingface.co/black-forest-labs/FLUX.1-dev)—noting](https://huggingface.co/black-forest-labs/FLUX.1-dev)%E2%80%94noting) it’s still *“boring just doing new single characters and objects.”* Despite having “a few thousand” followers for AI video content, they report that these models lack robust multi-subject scene construction and consistency needed for richer video pipelines, expressing a desire for next-gen models with better scene complexity, control, and coherence.
- [**Is that math??**](https://www.reddit.com/r/OpenAI/comments/1nvs4so/is_that_math/) (Activity: 477): **Post titled “Is that math??” links to a [v.redd.it](http://v.redd.it/) [video](https://v.redd.it/wcfvg31i2msf1) that currently returns** `HTTP 403 Forbidden` **with a Reddit network-security block, indicating access requires authentication (login or OAuth token), so the actual content is unavailable. From comment context, the thread likely centers on physics/relativity humor (Einstein references, non‑inertial frames), with no technical artifacts, benchmarks, or code shared.** Top comments riff on “releasing the Einstein files,” expect a relativity joke about speed limits in non‑inertial frames, and declare a “new meme era,” implying a lighthearted, meme-forward reception rather than substantive technical debate.
- [**Good use of AI .. I laughed and almost choked lmfao**](https://www.reddit.com/r/ChatGPT/comments/1nwasfv/good_use_of_ai_i_laughed_and_almost_choked_lmfao/) (Activity: 5333): **A short [v.redd.it](http://v.redd.it/) clip ([link](https://v.redd.it/8lah9oz5lqsf1)) appears to showcase a prank built on convincing AI-generated photos, raising questions about whether the accompanying script/narration was also AI-authored. Technically, the thread underscores how easily consumer-grade generative tools can compose multi-modal, high-believability hoaxes targeting non-technical audiences, illustrating the social-engineering risk surface of realistic image synthesis and scripted context.** Commenters debate if the script was AI-generated and suggest using examples like this to train older relatives about AI-enabled manipulation; others criticize the prank as irresponsible or harmful, noting the ethical line when shocking family members for laughs.
    - The only quasi-technical thread notes that beyond AI-generated photos, the "script" may also be AI-produced—implying a multi‑modal fabrication workflow (text + image) rather than a single‑modality deepfake. Another comment frames this as a social‑engineering vector for manipulating less tech‑savvy relatives, but the discussion contains no implementation specifics, model names, or evaluation details (e.g., detection methods, benchmarks, or pipeline components).
- [**I hope the White House doesn’t sue us**](https://www.reddit.com/r/ChatGPT/comments/1nvwk4e/i_hope_the_white_house_doesnt_sue_us/) (Activity: 1287): **Post appears to showcase a highly realistic AI-generated video (deepfake) of Donald Trump, with commenters noting that Sam Altman also appears and looks synthetic. The original asset at [v.redd.it](http://v.redd.it/) ([link](https://v.redd.it/ehrkx6dfansf1)) is not directly accessible without OAuth/login (HTTP** `403 Forbidden`**), so the clip’s authenticity and provenance cannot be independently verified; access requires Reddit login ([link](https://www.reddit.com/login/)) or support assistance ([link](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=21879292693140)). Discussion highlights rapid gains in generative video fidelity and the related authenticity/verification and legal-exposure concerns implied by the title.** Top comments emphasize unprecedented realism (e.g., “most realistic video of Trump I’ve EVER seen”), question whether parts are real (Altman “looks kinda artificial”), and suggest an adversarial legal stance if threatened with a lawsuit.
    - Perceived photorealism threshold: multiple users misidentified the clip as real, indicating state-of-the-art AI video generation has crossed a plausibility boundary where casual viewers can’t reliably distinguish synthesis from capture, especially in political-context footage. This highlights practical challenges for detection and provenance (e.g., watermarking/metadata) as distribution detaches content from original labels.
    - Residual uncanny cues: a commenter noting Altman *“looks kinda artificial”* points to remaining artifacts in facial modeling—micro-expressions, temporal coherence, and skin reflectance—that can still betray synthesis to attentive viewers. The mixed reactions suggest quality is scene- and identity-dependent, with failures typically surfacing under close-ups, complex lighting, or rapid expression changes.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. IBM Granite 4.0 Hybrid Models Launch**

- **Granite 4.0 Goes Hybrid, Open, and Enterprise-Ready**: **IBM** announced **Granite 4.0** with a hybrid **Mamba/Transformer** architecture, open-sourced under **Apache 2.0**, cryptographically signed, and billed as hyper‑efficient without performance loss, with broad availability via partners like **Hugging Face**, **LM Studio**, **NVIDIA NIM**, **Ollama**, and **Replicate** ([IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)).
    - The community debated its new **ISO 42001** credential, with one user calling it *"totally useless certification"* while others focused on practical access paths and enterprise distribution ([IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)).
- **Granite’s Hybrid Attention: Active Units at Scale**: Shared specs highlighted **hybrid attention** across sizes—**2B dense**, **7B (1B active)**, and **32B (9B active)**—with **FIM** support and no positional encoding, aimed to avoid degradation beyond **128k** context ([IBM Granite HF collection](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3)).
    - Users noted smooth paths to run as **GGUF** or fine-tune via **Unsloth** guides and assets, tightening the loop from model zoo to training stack ([Unsloth Granite 4.0 guide](https://docs.unsloth.ai/new/ibm-granite-4.0), [IBM Granite HF collection](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3)).

**2. Unsloth Training Stack: Docker, RL Speedups, and New Tricks**

- **Containers Conquer Config Chaos**: **Unsloth** shipped a cross‑platform **Docker image** with a step‑by‑step guide, while users shared manual **xformers** build scripts for **Blackwell (SM_12)** to unlock latest kernels ([Docker guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker), [Docker Hub](https://hub.docker.com/r/unsloth/unsloth)).
    - The flow targets frictionless training on Windows/Linux and advanced GPU stacks, with docs also covering **Granite 4.0** fine‑tuning on the same pipeline ([Unsloth Granite 4.0 guide](https://docs.unsloth.ai/new/ibm-granite-4.0)).
- **RL at Ludicrous Speed**: Unsloth reported the fastest **gpt‑oss RL** loops with **GSPO**, plus **VLM RL** that is **2× faster**, uses **90% less VRAM**, and supports **10× longer context** via kernel and weight‑sharing tricks ([gpt‑oss RL blog](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning), [VLM RL blog](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl)).
    - Early testers praised the throughput for rapid experimentation, framing the stack as a practical on‑ramp for large‑scale **reasoning RL** and **vision‑language** training workloads ([gpt‑oss RL blog](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning), [VLM RL blog](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl)).
- **Tversky Tricks and Leaner Losses**: A semi‑reproduction of **GPT‑2 Tversky‑All** for a llama‑like architecture landed with code and a test model—claimed **300B tokens** on a **3090 Ti** in ~1 day—while practitioners recommended **Linear Cross Entropy** via **Dao‑AI Lab’s quack** to speed training ([Architecture‑Tversky‑All](https://github.com/CoffeeVampir3/Architecture-Tversky-All), [HF test model](https://huggingface.co/Blackroot/Tversky-All-Test-100MIsh), [LCE impl line](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/main.py#L115), [quack LCE](https://github.com/Dao-AILab/quack/blob/main/quack/linear_cross_entropy.py)).
    - Community tips emphasized **sequence‑packed varlen flash‑attn** and careful kernel selection for wall‑clock wins, pairing lean losses with efficient data layouts to cut epochs ([varlen MHA example](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/modeling/MHA.py#L36)).

**3. GPU Systems: Determinism, Flash‑MoE, and Kernel Fusion**

- **Determinism Tames the Dice Roll**: **Thinking Machines** detailed defeating **non‑determinism** in LLM inference and released **Flash‑MoE**, a variant of Flash‑Attention for sparse‑expert setups ([Defeating Non‑Determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [Flash‑MoE site](https://flash-moe.github.io/)).
    - Engineers flagged stable reproducibility as essential for debugging and benchmarking model traces, positioning **Flash‑MoE** as a practical building block for scalable **MoE** inference ([Defeating Non‑Determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [Flash‑MoE site](https://flash-moe.github.io/)).
- **NVIDIA Papers Fuse and Specialize**: **NVIDIA** published compiler work on scheduling and **warp specialization** with benchmarks vs **FA3** ([Cypress, PLDI 2025](https://d1qx31qr3h6wln.cloudfront.net/publications/Cypress_PLDI_25.pdf)) and on **distributed kernel fusion** for end‑to‑end efficiency ([Legate Kernel Fusion, ASPLOS 2025](https://d1qx31qr3h6wln.cloudfront.net/publications/Legate_Kernel_Fusion___ASPLOS_2025.pdf)).
    - Discussion focused on mapping these techniques to production **tensor programs** and cluster‑wide execution graphs to reduce launch overheads and improve **E2E throughput**.
- **JAX Blackwell Matmul Masterclass**: **JAX** released a tutorial on achieving SOTA **matmul** performance on **Blackwell GPUs** with **Pallas**, covering tiling, memory movement, and kernel authoring best practices ([JAX Blackwell matmul tutorial](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html)).
    - Practitioners highlighted the guide as a blueprint for hand‑tuned **GEMM** kernels that translate to real wins in **training** and **inference** pipelines.

**4. OpenRouter: Routing Metrics, Fees, and New Models**

- **Performance Plots Prompt Quantization Questions**: **OpenRouter** launched a **Performance Tab** that visualizes provider metrics per model, sparking calls to filter by **quantization** (e.g., **FP4** vs **BF16**) to avoid misleading comparisons ([Performance Tab post](https://x.com/OpenRouterAI/status/1773733582763069916)).
    - Users requested a dropdown for quant levels and noted that fair apples‑to‑apples comparisons require normalizing for **precision**, **context**, and **tool‑use** settings.
- **BYOK Clarified: 0% Fee, Not Free Compute**: The **“1M free BYOK requests/month”** promo waives OpenRouter’s **5% commission** for the first million requests, but users still pay the underlying provider’s API bill ([announcement](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month)).
    - Several suggested clearer wording like *"1M monthly BYOK requests at 0% fee"* to avoid confusion about actual **inference costs** ([announcement](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month)).
- **Qwen’s Image Editor Enters the Ring**: **Alibaba Qwen** introduced a new **image‑edit** model (not text‑to‑image), with devs sharing the launch and seeking Apple Silicon paths ([Qwen announcement](https://x.com/Alibaba_Qwen/status/1973668568412856595), [community post](https://x.com/pingToven/status/1973758872772108663)).
    - Early chatter focused on **editing‑only** constraints and integration questions, with interest in local **M‑series** acceleration.

**5. LMArena: Reasoning Trace and Leaderboard Shifts**

- **Watch Models Think Before They Speak**: **LMArena** enabled **Reasoning Trace** for reasoning models across **Side‑by‑Side** and **Direct** chat, letting users see the model’s work pre‑answer ([Side‑by‑Side](https://lmarena.ai/?mode=side-by-side), [Direct](https://lmarena.ai/?mode=direct)).
    - Power users welcomed the added transparency to debug **reasoning chains**, compare models’ **scratchpads**, and sanity‑check **intermediate steps**.
- **Claude Sonnet 4.5 Crowns the Text Charts**: **Claude Sonnet 4.5** tied **Claude Opus 4.1** for the **#1** spot on the **Text Leaderboard**, and the **32k thinking** variant replaced **16k** in production flows ([Text Leaderboard](https://lmarena.ai/leaderboard/text)).
    - Community remarks praised **Hard Prompts**, **Coding**, and **Creative Writing** results, aligning perceived quality with the updated **thinking window**.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Kills o3, Shills GPT-5**: Perplexity deprecated the **o3 model** from their model selector and encourages users to transition to **GPT-5 Thinking**.
   - Perplexity claims **GPT-5 Thinking** offers stronger performance and ongoing support.
- **Discord Desktop Saves Comet Quest**: Users are downloading the [Discord desktop app](https://discord.com/download) to complete the **Comet quest** and claim 5k orbs.
   - Some users are having trouble finding the quest in the Discord app and are advised to *check the pins*!
- **Privacy Put to the Test**: A user shared a memory with a combo of English, Finnish, Japanese and Spanish, sparking a privacy discussion.
   - Another user stated they could share the prompt, but wouldn't go through their memories to snip out the private ones, doubting they're the ones affecting it.
- **Comet Browser Fails to Launch**: A user shared a [screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1423483019540434954/image.png?ex=68e0795e&is=68df27de&hm=8a12e66bd9f89baf735f8c1bcd80271022e88595da198b5fd7bbaebe96aa5b64&) of their success, pointing out the browser's opening is incredible.
   - Others noted that it still def needs work, as its just like any basic browser and more annoying since you cant use google as primary and shift enter for the AI.
- **Sonar-Pro API Returns Rotten Resources**: A user reported that the **Sonar-Pro API** is generating resources that lead to **404 errors** and asked for a way to filter results.
   - They hope to only receive resources that are confirmed to exist and be available to the public, avoiding **404 errors**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Sora 2 Launch Triggers Hype**: Community members eagerly await **Sora 2**'s arrival on the platform, anticipating its impact and comparing it to video models like **Veo 3**.
   - Enthusiasts expressed excitement and hoped to see it benchmarked on LMArena.
- **Gemini 3 Release Speculation Intensifies**: The community is buzzing about the impending release of **Gemini 3**, with discussions focusing on its potential competitiveness regarding [ratelimits](https://discord.com/channels/1340554757349179412/1340554757827461211/1423022363716485121).
   - A leak claimed an **October 9th** release date, further fueling the anticipation.
- **4o Model Retirement Incites Disappointment**: Users expressed disappointment over the limited availability and eventual retirement of the **4o model** on LMArena.
   - One member lamented their *'addiction'* to **4o**, highlighting the difficulty in finding a suitable replacement.
- **Ethical boundaries debated**: Concerns were raised about **OpenAI's** data usage practices, with one user jokingly admitting to sending *sensitive government data to lmarena*.
   - Another member pointed out that it was a wild thing to admit in a discord chat.
- **Claude Sonnet 4.5 Takes #1 on Text Leaderboard**: **Claude Sonnet 4.5** impressively tied with **Claude Opus 4.1** for the **#1 slot** on the [Text Leaderboard](https://lmarena.ai/leaderboard/text).
   - It is also performing well across categories such as Hard Prompts, Coding, and Creative Writing, *garnering positive community discussion* in the dedicated channel.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Blackwell Manual Compiling Bonanza Begins**: Members discussed manually compiling **xformers on Blackwell GPUs**, with one sharing a script using `pip uninstall -y xformers`, `git clone`, and `python3 setup.py install` to manually compile xformers for compute capability **12**, as well as the [Docker Hub link](https://hub.docker.com/r/unsloth/unsloth) for the updated image.
   - This is necessary to use the **latest GPUs** for accelerated computing.
- **Docker Debuts for Unsloth Training**: Unsloth released a new **Docker image** for training on Windows/Linux without dependency issues, detailed in their [guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker) and available on [Docker Hub](https://hub.docker.com/r/unsloth/unsloth).
   - This aims to resolve dependency conflicts and streamline the setup process for users on different operating systems.
- **Synthetic Data Surge sans vLLM**: Members discussed generating **synthetic datasets** without relying on **vLLM**, suggesting the use of the **OpenAI package** for async requests to a local server, along with a pointer to [meta-llama/synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit).
   - One member noted all Unsloth notebooks currently use vLLM.
- **Tversky-All GPT2 Gets Llama-Like Upgrade**: A member released a semi-reproduction of the **GPT2 Tversky-All**, using the Tversky-All strategy but for a llama-like model, available at [CoffeeVampir3/Architecture-Tversky-All](https://github.com/CoffeeVampir3/Architecture-Tversky-All).
   - The test model is available at [HuggingFace](https://huggingface.co/Blackroot/Tversky-All-Test-100MIsh); it was trained on **300 billion tokens** on a **3090 TI** in about a day.
- **GGUF Conversion Woes Plague Users**: Users are encountering issues when trying to convert models to **GGUF format**, specifically when using the *push_to_hub_gguf* function with **f16 quantization**, was advised to perform the conversion manually until a fix is pushed.
   - A member reported a **ValueError** related to mapping tensor 'model.layers.0.self_attn.q_proj.base_layer.weight'.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sonnet 4.5 Pricing Debated**: Members debated cost-effectiveness of **Sonnet 4.5** versus **GLM 4.6**, with some pointing out that **GLM 4.6** is six times cheaper.
   - Some users felt **Sonnet 4.5** performed similarly to **3.7**, while others favored **4** over **3.7** in **Copilot**.
- **Server Overrun by Sora Fanboys**: A member raised concerns about the server being overrun by **Sora users**, who critiqued **OpenAI's marketing** for triggering the influx.
   - The member suggested channel names be dynamically updated based on current discussion topics using **LLMs**.
- **Deepfake Drama Divides Users**: A user questioned the irony of an app supporting deepfakes while criticizing the generation of photorealistic AI images.
   - This sparked discussions about forwarding feedback to relevant channels amidst a flood of *code please* requests.
- **Sora as Social Media Central**: A user suggested **Sora** should integrate as a social media platform like **TikTok**, enhancing user experience with **ChatGPT**, similar to image generation.
   - Another user proposed implementing a **credit system** for **Sora**, allocating more resources for video generation with **daily or weekly usage limits**.
- **Users Debate Square Images for Sora**: Members discussed best practices for image generation in **Sora**, with one asking if **portrait mode** works better than **landscape mode**.
   - Another member replied that visual tokens are arranged in a grid, so **square images** will probably generate the best results from images.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Snapdragon X Elite Specs Spark Debate**: A user shared the specs of their **Microsoft Surface Pro** with a **Qualcomm Snapdragon X Elite** (12-core, X1E80100 @ 3.40 GHz) and **16 GB** of RAM.
   - After seeing a mysterious *artifact*, they asked if LLM opinions were trustworthy.
- **Quantization Quandaries Questioned**: Members explored how quantization impacts knowledge retention in language models, with lower quantization potentially impacting smaller models because *reason* bits are lost.
   - A member shared a funny sentiment about what happens when the quantization level gets too extreme: *you get too quantised and suddenly your mixing yanderes and petting dogs in a way you where not expecting😄*.
- **GPT-OSS: Openly Safe Substitute Ships**: The release of **GPT-OSS**, a model that behaves similarly to **GPT-4o**, was announced.
   - Members noted it assumes a lot of information if not provided with enough details.
- **Arc B50 Pro Bogs Bandwidth Battle**: A member benchmarked **Arc B50 Pro** cards against an **RTX 4080 Super**, revealing the B50s have *boatloads of VRAM* but abysmal memory bandwidth, resulting in lower token rates (**7-8 Tps** for a **12B q8_0 model** compared to **30+ Tps** on the 4080).
   - However, at default context (**4k**), the B50s got **32 Tps** while the 4080 got **42 Tps**.
- **DDR3 Dreams Dashed for GPU Deployment**: A user suggested using cheap **DDR3** boards with multiple **PCIE 16x slots** to accommodate **6x GPUs**, combined with raided **SATA SSDs** for faster load times, referencing a [eBay listing for X99 boards](https://ebay.us/m/zB2BAH).
   - Concerns were raised about the memory bandwidth (**68 GB/s** with **DDR4**) and the potential bottleneck compared to modern standards, with a user saying that *on ddr3 you max out at like 50gb/s*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Integrates Git Worktree**: Users discovered **Git Worktree** setting within the **beta tab** of settings, encouraging its use in the agent window. 
   - It appears that Git Worktree integration is available in the **Early Access** or **Nightly Cursor** versions.
- **Cursor Beta Functions Spark Curiosity**: Members discussed using beta functions in Cursor, recommending it for early access to features, fun debugging, and helping improve Cursor; currently, **afterFileEdit** is the only available hook.
   - The **Extension RPC Tracer** is available for checking RPCs during beta function use.
- **Typescript Refactor Triumph**: One user reported a successful full **Typescript refactor** using Cursor after four prompts, using a follow-up master prompt for auditing.
   - Using Cursor's **Plan mode** and tracking workflow status in the Nightly version were suggested for improved efficiency.
- **MacBook Suffers Meltdown from Cursor**: A user reported Cursor causing their **MacBook Air M4** to crash due to high memory usage, spiking to **96GB** possibly related to chats or agent processes; resolved after rebooting.
   - Members suspected a memory leak, noting that **MacOS** versions have a higher incidence, with downgrading suggested as a workaround.
- **Cursor Hackathon on the Horizon?**: A member inquired about interest in a **Discord Cursor Hackathon**, to implement solutions and side projects.
   - Interest was expressed in sponsored hackathons with free credits, with a suggestion to make the hackathon remote friendly to accommodate different time zones.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Rolls Out Provider Performance Tab**: OpenRouter released a new "Performance Tab" visualizing [provider performance for a given model](https://x.com/OpenRouterAI/status/1773733582763069916), prompting a discussion about fair comparisons between providers using different quantization levels.
   - A user suggested adding a filter dropdown to account for different **quant levels** like **FP4** and **BF16** to prevent misleading comparisons.
- **BYOK Promo Spark Confusion**: Users debated the "1 million free BYOK requests per month" offer, clarifying it waives OpenRouter's **5% commission fee** for the first million requests, but users still pay the provider directly for API usage, according to [OpenRouter documentation](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month).
   - Some users initially thought the offer provided completely free requests, leading to suggestions for clearer messaging, such as *"1M monthly BYOK requests at 0% fee"*.
- **Grok Gets Roasted, Sonoma Soars?**: A user tested **Grok 4 Fast** and called it *"way dumber than Sonoma"*, saying it *"fails constantly"* and disregards format requirements.
   - Another user speculated that Grok 4 Fast *"reeks of… Llama..?"*, expressing frustration with its inconsistency.
- **Gemini Pro Gets Glitchy**: Users reported that **Gemini Pro** was responding with *"weird stuff"*, failing to use tools correctly, and exhibiting *"unacceptably slow"* performance via the OpenRouter API.
   - Reports suggest this may be a common issue with **Gemini 2.5 Pro**, and one user recommended trying Vertex as an alternative provider.
- **Qwen's Image Edit Arrives!**: Members shared [Alibaba's new **Qwen image model**](https://x.com/Alibaba_Qwen/status/1973668568412856595), noting it was only an image edit model and one user shared [this post](https://x.com/pingToven/status/1973758872772108663) announcing it.
   - Another member expressed interest in running it on **Apple Silicon**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Exploring Perplexity AI Framework**: Members discussed the [Perplexity AI framework](https://www.perplexity.ai/page/scientists-ai-framework-solves-hTbKnxfPSl64P5nLfxqX5A) and its associated **GitHub project**, particularly focusing on **LLMs** using similar attention matrices.
   - The discussion considered efficient attention mechanisms like **Deepseek Sparse Attention** as an example of **top-k attention**, questioning potential issues compared to sliding window attention.
- **Gradient Descent Dynamics Paper Hailed**: A member praised a paper on the *dynamics of gradient descent* ([centralflows.github.io](https://centralflows.github.io/part1/)) for addressing **loss spike dynamics** and impacting **Adam's beta2**.
   - Despite its low citation count, the paper was lauded for its solution and impact, being considered the **paper of the year** by one member.
- **Symmetry Transformer Shows Promise**: Experiments with a *symmetry transformer* ([GitHub repo](https://github.com/Eternalyze0/symmetry_transformer)) showed that predicting the **current and previous token with separate heads** improved validation loss in later training runs.
   - Initial results indicated that the baseline model performed better, but the symmetry model later improved after more training.
- **Questioning AUNN's Practicality**: The practicality of **AUNN (Augmented Neural Networks)** was debated, raising concerns about its efficiency and the absence of a functional prototype beyond a toy example, [ethan-w-roland/AUNN](https://github.com/ethan-w-roland/AUNN).
   - The discussion stated that the proposer of AUNN focused more on **MLPs** than **Attention**, and was *combative* towards counterarguments.
- **Transformers are 2D Slices**: The guild discussed that Transformers optimize by splitting a big 2D problem, (sequence, channels), into slices.
   - A member stated that a giant **MLP** applied to the whole problem *would work fine, but it's intractable that way* and that **Transformers** are used because they are *just cheap*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Benchmarking Brainstorming Begins**: Members requested a *good guide for benchmarking* and were pointed to [this arXiv paper](https://arxiv.org/abs/2502.15015), [this article on kernel benchmarking](https://jan.ai/post/how-we-benchmark-kernels), and [this YouTube video](https://www.youtube.com/watch?v=1i7dxoAfKOU).
   - One of the members described their previous benchmarking work as *maybe the best benchmarking effort*.
- **Non-Determinism Drops Dead**: **Thinking Machines** posted a [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) on defeating **non-determinism** in **LLM inference**.
   - They also released [Flash-MoE](https://flash-moe.github.io/), a variant of **Flash Attention**.
- **Nvidia Compiles Fresh Code**: **Nvidia** is working on **compiler techniques** for scheduling and **warp specialization**, with benchmarks against **FA3** detailed in their [paper](https://d1qx31qr3h6wln.cloudfront.net/publications/Cypress_PLDI_25.pdf).
   - **Nvidia** is fusing kernels in a distributed setting as outlined in this [paper](https://d1qx31qr3h6wln.cloudfront.net/publications/Legate_Kernel_Fusion___ASPLOS_2025.pdf).
- **Linear Cross Entropy for LLM Training**: The use of [Linear Cross Entropy](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/main.py#L115) is recommended for accelerating the **LLM training process** and to use the **Quack optimization** library, specifically the [linear cross entropy implementation](https://github.com/Dao-AILab/quack/blob/main/quack/linear_cross_entropy.py).
   - **Sequence packed or 'unpadded' training** is identified as a highly impactful optimization, particularly with techniques like **flash attn varlen**, see [this implementation](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/modeling/MHA.py#L36).
- **Cooperative Group Aligns**: A member asked about the `alignment` argument in `CooperativeGroup.__init__`, specifically what it does and why it must be 32 if `size` is 32 but not for other values in *Cutlass channel*.
   - Another member responded that this check is *because they happen to be the warp/warpgroup ganularity and are the common cases warranting special checks to prevent bugs*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Musk Envisions AI-MMOs**: Elon Musk is discussing co-developing an **AI-integrated MMO (AIMMORPG)** with **Eve Online's** creators, aiming to exploit unique AI capabilities.
   - A user speculated that AI would be a *"natural fit"* within such a game.
- **Karpathy's Koans on Bitter Lesson**: Karpathy summarized the [Dwarkesh-Sutton podcast](https://x.com/karpathy/status/1973435013875314729), highlighting Sutton's doubts that **LLMs** fulfill his thesis.
   - Karpathy acknowledges the practical bootstrapping offered by pre-training, while also suggesting that bigger paradigms await, and researchers should look to animal intelligence for inspiration.
- **Hume AI Hits Hyperspeed with Octave 2**: [Hume AI unveiled Octave 2](https://xcancel.com/hume_ai/status/1973450822840152455?s=46), their next-gen multilingual text-to-speech model, now supporting **11+ languages** with a **40% speed boost** (<200 ms latency) and **50% cost reduction**.
   - The release includes multi-speaker chatter, improved pronunciation, new voice-conversion and phoneme-editing tools and a **50 % discount** on their Creator plan during October.
- **Mistral Drafts Mathletes**: Albert Jiang announced that Mistral AI is forming a new **formal-math research team** after their $2B funding.
   - They are seeking AI talent for an all-in-one prover/autoformalizer/agent, offering elite collaborators, hundreds of GPUs per employee, open research, top salaries, and offices in Paris, London, and Palo Alto; the job opening is advertised [here](mailto:aj@mistral.ai).
- **Figma's Field Guide to AI**: The Latent Space podcast featured **Figma's co-founder Dylan Field** discussing **Figma's AI Playbook**.
   - The episode explores surfacing good design in the era of **vibe-coding**, **Figma's Make, MCP for 'tasting' agents**, and the future of **fast-fashion SaaS** ([link to X](https://x.com/latentspacepod/status/1973793231524806925), [link to Xcancel](https://xcancel.com/latentspacepod/status/1973793231524806925)).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes Model Claims Close to GPT-5**: A member inquired whether **Nous Research** tuned models are comparable to **GPT-4.5**, leading to a response that these models are closer to **GPT-5** or **Gemini**.
   - Ironically, when the member queried **Gemini** about alternatives, **Hermes** was among the options it suggested.
- **Veo3 Eclipses Sora?**: A user expressed a preference for **Veo3** over the latest **Sora**, sharing a [Prompt_theory.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1423308176148922491/Prompt_theory.mp4?ex=68dfd688&is=68de8508&hm=e195c2f737881136d240fa288b286f7dcc417fbe581c153cefa587b7c2ec0233&) as part of their discussion.
   - No further details were given to illustrate why **Veo3** was the preferred option.
- **Granite Models Showcase Hybrid Attention**: **IBM Granite** language models feature hybrid attention in models such as **2B dense**, **7B (1B active)**, and **32B (9B active)**, as outlined in a shared [Hugging Face collection](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3).
   - These models support **FIM** (Fill in the Middle) and lack positional encoding, which prevents performance degradation when processing contexts beyond **128k**.
- **Qwen 30B A3B Thrives on CPU**: Members find **Qwen 30B A3B** is well-suited for **CPU** usage, with one user reporting performance metrics on a **Ryzen 7 5700G** CPU with **32GB VRAM**.
   - Specifically, **Qwen 3 30B A3B** at **Q6_K_XL** achieves **48 TPS** processing and **10.5 TPS** generation speed at **1024** tokens of context.
- **LLMs Caught in Web of Deceit?**: A member shared their [preprint](https://arxiv.org/html/2509.20393v1), about **strategic LLM deception**.
   - The study uses **sparse autoencoders** (hosted by [Goodfire AI](https://www.goodfire.ai/)) to show how current methods fail to detect the internal features driving **strategic LLM deception**, highlighting a tangible path to closing the **autolabel gap**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Deepmind Code Incompleteness**: Members joked that [Deepmind](https://deepmind.google/) does extra work to avoid sharing their implementations, making it unclear how it works as part of a larger system, citing their experience implementing **V-MPO**.
   - They noted that Deepmind's code is often sophisticated, but they piece it out in ways that obscure its overall functionality.
- **HuggingPapers Code Fails to Run**: Members noted that [code from HuggingPapers](https://fxtwitter.com/HuggingPapers/status/1973420932497879298?t=jxTf48_aBK8349s1uSyDQw&s=19) doesn't run because it **doesn't import RoPE**.
   - The original poster of the code seemingly indicated that the user is supposed to implement it themselves.
- **IBM Granite 4.0 Hybrid Architecture**: IBM launched **Granite 4.0**, the next generation of IBM language models, featuring a new **hybrid Mamba/transformer architecture** that greatly reduces memory requirements without sacrificing performance, open-sourced under **Apache 2.0 license**.
   - The models are available on **IBM watsonx.ai**, as well as through platform partners including Dell Technologies, Docker Hub, Hugging Face, Kaggle, LM Studio, NVIDIA NIM, Ollama, OPAQUE and Replicate, with access through AWS Sagemaker JumpStart and Microsoft Azure AI Foundry coming soon. [IBM Announcement Here](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models).
- **Doubts Raised on ISO 42001 Certification**: Members noted that the new Granite 4.0 model is the world’s first open models to receive **ISO 42001 certification**.
   - A user commented that this is a totally useless certification, to blend C-suite people into thinking this is worth it.
- **Oracle runs OpenAI's Datacenters?**: A user commented that Oracle's business model used to be selling Databases and enterprise software now it seems to be running datacenters for OpenAI.
   - They cited [OpenAI Elon Musk Post](https://openai.com/elon-musk/) as the source for this theory.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Credit Consumption Sparks Outrage**: A user complained about a basic research task consuming **5300 credits** without completion, labeling Manus as an *"absolute joke"* and requested a [refund](https://cdn.discordapp.com/attachments/1349440650495398020/1423044743377715343/image.png?ex=68e032b1&is=68dee131&hm=cd60314f2b422e917efdd7ebbd2a8747117a9e000904040bddb4b0a1d2624fd9).
   - A team member asked for the session link to investigate and potentially offer a credit refund.
- **Unlock Agent Mode with Memory Key**: A member proposed a **Memory Key protocol** to solve the issue of exiting Agent Mode, which involves saving context before restarting a session.
   - They detailed a [solution](https://discord.com/channels/1348819876348825620/1349440650495398020/1422940046855766016) that involves copying essential information, starting a new session, and instructing the agent to create an updated **Memory Key** for future use.
- **Billing Issue Sparks Support Vacuum**: A user reported a billing issue with no response from Manus support, prompting a community member to suggest emailing their official support address with a clear subject line and ticket number.
   - It was suggested that this would create a formal paper trail for escalation.
- **Global Pricing Model Criticized for Disparity**: A user criticized Manus' **global USD pricing model** ($39/month for the Plus plan) for not adjusting to regional economies, creating a barrier in countries like Brazil and other parts of Latin America.
   - Another user suggested implementing regional pricing based on **Purchasing Power Parity (PPP)** to improve accessibility and promote global growth.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AGI Paper Dropped, Courtesy of HF**: A member shared a [Hugging Face paper](https://huggingface.co/papers/2509.26507) introducing **AGI** in the `show-and-tell` channel.
   - The user cheekily stated, *called it 😉*.
- **Show-and-Tell channel debuts on DSPy Discord**: The DSPy Discord server has a new **show-and-tell** channel.
   - The channel is designed for users to demonstrate and discuss their projects using DSPy.
- **Caching Prompt Order: Use with Care**: Members have found that the order in which prompts and files are sent greatly impacts caching.
   - To effectively leverage caching, the specific order of prompt elements and file inputs must be carefully observed.
- **DSPyWeekly gets Search Feature**: [DSPyWeekly](https://dspyweekly.com/) now features a search function to browse crawled content, complete with prev/next links for smooth navigation.
   - This enhancement streamlines access to information, facilitating easier discovery of relevant topics.
- **XMLAdaptor May Become New JSONAdaptor**: Members debated if **JSONAdaptor** should remain default given that **ChatAdaptor** or **XMLAdaptor** often fix adaptor errors.
   - The rise of tool use RL for models is making **XML** a potential default, despite **JSON** being a reliable fallback.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen Coder Model Benchmarks**: Members discussed the **Qwen Coder** model, suggesting the **30B** version *should be smart enough* and considered **Qwen3 Coder** as a newer alternative.
   - They cautioned that quantization could affect performance, recommending **Q4** if chosen.
- **Aider's Release Cadence Concerns**: A member expressed concerns over the reduced release cadence for **aider** and suggested a **Patreon** or donation system for support.
   - The user highlighted concerns about developer burnout and potential discontinuation of **aider**, given its utility for real work compared to other agentic tools.
- **Aider-desk UI Experiences**: A member inquired about using **aider-desk** or similar UIs with **aider**.
   - Another member used it briefly for **MCP support**, finding it suitable for those wanting an **aider**-style workflow with optional agent use cases, but they have since switched to **sst/OpenCode**.
- **DeepWiki Reverse Engineering Invitation**: A member shared a [DeepWiki page](https://deepwiki.com/search/please-reverse-engineer-the-pr_c15e0046-3403-4786-bf26-63b2bf046455) encouraging reverse engineering.
   - Another member suggested using an **output template** or **post-processing** in *koboldcpp*, unsure if it's available in *llama.cpp*.
- **Custom Chat Templates Hack**: A member mentioned the ability to specify a custom **Jinja chat template** to override the one contained in the **GGUF**.
   - They also suggested using a [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) to format the model's input, and started a [discussion on llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16386) about it.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Devs Socialize at Ye Olde London**: Members <@1042188459575619674> and <@1387880172589547722> hosted drinks at **Ye Olde London**, inviting other developers to network and connect in person.
   - One member with <@1407892993934889010> mentioned they would *"pop over for a bit!"*
- **Registry Team Broadcasts Live**: The Registry Team launched a livestream, available [here](https://www.youtube.com/watch?v=5qwXADMBuio), at **9 AM UK time**.
   - The livestream covered various aspects of the team's work.
- **Tool Call Support Proposed for Sampling**: A member submitted a proposal (SEP) to integrate **Tool Call support into Sampling** via [issue #1577](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1577).
   - The proposed integration depends on ongoing discussions around **multiple content blocks**, aimed at enabling parallel tool calls through [PR #198](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/198).
- **Reference Implementation Streamlines Testing**: A new reference implementation (TS SDK) has been released, featuring an example server powered by an **agentic loop tool**, alongside a **backfill proxy** designed to simplify testing, see [PR #991](https://github.com/modelcontextprotocol/typescript-sdk/pull/991).
   - A member noted that initial CI failures were resolved by pinning the **zod** minor version.
- **OCI Interface Idea Sparks for MCP Servers**: A member suggested developing an **OCI-like interface for MCP servers**, where all metadata could be packaged inside a tarball for simpler handling.
   - The goal is to streamline the process of building and distributing **OMCP packages**, thereby simplifying metadata management.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Qualcomm Flirts with Mojo?**: A member speculated that **Qualcomm** might reach out to **Modular** about **Mojo**, possibly indicating interest in leveraging **Mojo's** capabilities for their hardware.
   - The discussion originated in a Qualcomm Developer's Discord voice chat.
- **Mojo Manual Gets Pythonic**: The **Mojo Manual** was updated, with a user specifically highlighting the [Python section](https://docs.modular.com/mojo/manual/python/).
   - The update suggests enhancements or crucial details regarding **Mojo's** interoperability with **Python**.
- **Mojo Explores Notebook Territory**: The discussion centered on using **Mojo** within notebooks, specifically if the goal was to *interact with Max from a notebook* or to *directly author and run Mojo within notebooks*.
   - A user reported success in interacting with **Mojo** in a notebook and expressed interest in a syntax highlighter for better learning.
- **Radeon GPU passes vector addition test**: A user successfully ran the *vector_addition example* on an **AMD Radeon 6800 XT**, referencing the [GPU Compatibility documentation](https://docs.modular.com/max/packages/#gpu-compatibility).
   - A Modular employee responded that they haven’t done extensive testing on **RDNA 2 GPUs** and that models won’t run correctly on **RDNA GPUs** yet.
- **Mojo Eyes Distributed Computing Future**: A member inquired about the potential of using **Mojo** with frameworks like **Dask** or **PySpark** for distributed computing.
   - Another member suggested that Mojo welcomes people building their own frameworks, as a fully **Mojo framework** will likely be lower latency and higher throughput than Python-based options.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Unveils a Surprise Capability**: After watching [a video demo](https://cdn.discordapp.com/attachments/1371757564005711973/1423369409669365881/2025-10-02_20-01-10.mp4?ex=68e00f90&is=68debe10&hm=b28aea7215ef03687754be113fa4dd6a583c355be6e3462543235d7d0258ef73&), a user noted an unexpected feature in **Kimi**.
   - The specifics of the new capability were not detailed in the prompt.
- **Sora's Video Demos Face Quality Critiques**: Users are comparing the quality of shared **Sora** video demos, suggesting that the versions available may be lower quality than those showcased on **OpenAI's YouTube channel**.
   - One user described the quality as *weirdly wobbly*.
- **Sora Pro Subscription Gives Watermark-Free Output**: The **Pro subscription** version of **Sora** will supposedly offer higher resolution videos without visible watermarks.
   - One user cautioned that *an invisible watermark will be applied - so mister openai can tell its generated, just we cant...*



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker Faces Imminent Deletion**: A user inquired about the impending deletion of **ShapeTracker** and sought documentation regarding this change.
   - Another user shared a relevant [X post](https://x.com/__tinygrad__/status/1967446896081076517) shedding light on the matter.
- **ShapeTracker's Successor Sought**: In the same query about **ShapeTracker's** deletion, the user asked about potential replacements.
   - The shared [X post](https://x.com/__tinygrad__/status/1967446896081076517) might contain information about what will replace **ShapeTracker**.



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





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1423355340107546744)** (2 messages): 

> `o3 Deprecation, GPT-5 Thinking` 


- **Perplexity says Farewell to o3!**: Perplexity has deprecated the **o3 model** and removed it from their model selector as of today.
   - Users are encouraged to transition to **GPT-5 Thinking**, which Perplexity claims offers stronger performance and ongoing support.
- **GPT-5 Thinking Highly Recommended**: Perplexity recommends users switch to **GPT-5 Thinking** after deprecating **o3**.
   - They state that **GPT-5 Thinking** provides better performance and full support moving forward.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1423021661564702831)** (1281 messages🔥🔥🔥): 

> `Comet Browser, Discord Quest, Troubleshooting, User Experience, AI and Personal Data` 


- ****Comet Quest creates Discord Desktop downloads****: Users are downloading the [Discord desktop app](https://discord.com/download) to complete the **Comet quest** and claim 5k orbs.
   - Some are having trouble finding the quest in the Discord app: *check the pins*! <:a:check_pins:1406044966500700160>
- ****Comet + Sonnet equals gold?****: A user shared a [Sonnet 4.5 prompt](https://discord.com/channels/1047197230748151888/1047649527299055688/1423418938510938322) they found especially helpful, highlighting that Prompt + sonnet 4.5 is greatt!
   - However, there are [bugs](https://discord.com/channels/1047197230748151888/1047649527299055688/1423432301633566730) where the CoT isn't shown when using Select Models.
- ****Privacy and Personal Data****: A user shared a memory with a combo of English, Finnish, Japanese and Spanish. and noted they I've got a combo of English, Finnish, Japanese and Spanish for whatever reason in my memories.
   - Another states I can share the prompt I have there, but no way I'm going through my memories to snip out the private ones. Doubt they're the ones affecting it either.
- ****Comet Browser still needs to cook****: A user points out that the browser's opening is incredible, sharing an attached [screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1423483019540434954/image.png?ex=68e0795e&is=68df27de&hm=8a12e66bd9f89baf735f8c1bcd80271022e88595da198b5fd7bbaebe96aa5b64&) of their success.
   - However, it still def needs work according to others. As a user noted, its Just like any basic browser more annoying since i cant use google as primary and shift enter for the ai.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1423081493596471376)** (8 messages🔥): 

> `PC BUILD, Perplexity AI apps, bootstrap-paradox` 


- **Users share perplexing perplexity.ai links**: Several perplexity.ai search and app links were shared: [PC build](https://www.perplexity.ai/search/96b1fdb5-6e66-4156-a90f-ad6924e30b99#72800$), [app link1](https://www.perplexity.ai/apps/8e3bda61-b835-4a6d-addb-167a800c83db), [app link2](https://www.perplexity.ai/apps/6f2a0a07-d165-4dc4-af96-4db2494e2951).
- **YouTube link shared**: A [YouTube link](https://www.youtube.com/watch?v=V64TdrkhTqo) was shared, without any further context.
- **Bootstrap Paradox link shared**: A perplexity.ai page discussing [the-bootstrap-paradox](https://www.perplexity.ai/page/the-bootstrap-paradox-a-forens-dsHvoK0YQqWIGbyQ0pyjyQ) was shared.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1423319451998355506)** (1 messages): 

> `Sonar-Pro API, 404 Errors, Public Resources` 


- **Sonar-Pro API Yields 404 Errors**: A user reports that the **Sonar-Pro API** is generating resources that lead to **404 errors**.
   - The user is seeking a way to obtain resources that are both currently available and publicly accessible.
- **Request for Active, Public Sonar-Pro Resources**: The user explicitly asks if there's a method to filter **Sonar-Pro API** results.
   - They hope to only receive resources that are confirmed to exist and be available to the public, avoiding **404 errors**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1423022363716485121)** (1348 messages🔥🔥🔥): 

> `Sora 2, Gemini 3, Qwen3 4B 2507 instruct, 4o model, OpenAI safety` 


- **Sora 2 Anticipation Builds in Arena**: Members eagerly await **Sora 2**'s arrival on the platform, with discussions focusing on its potential impact and comparisons to **Veo 3** and other video models.
   - One member stated: *I like playing with Sora so much I don’t even wanna try anything else lol* and hoped to see it benchmarked.
- **Gemini 3 Hype Intensifies with Release Speculation**: The community is buzzing about the impending release of **Gemini 3**, with a member mentioning that *gemini 3 needs to have good ratelimits to stay comptitive*.
   - Some users shared that there was a leak claiming an **October 9th** release date.
- **Frustration with 4o model as it gets retired**: Members expressed disappointment over the limited availability and eventual retirement of the **4o model**.
   - One member lamented their 'addiction' to **4o**, highlighting the difficulty in finding a suitable replacement.
- **Debate on AI's Ethical Boundaries and Data Usage**: Concerns were raised about **OpenAI's** data usage practices, with one user jokingly admitting to sending "sensitive government data to lmarena."
   - Another member then said it was a wild thing to admit in a discord chat.
- **Chat length limits prompt discussion**: Members discussed the length limits of the model chats and what it would take to summarize these and then extend the length allowed.
   - A user pointed out *I'm okay with it forgetting, just want to continue it*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1423028346685751346)** (5 messages): 

> `Arena Champions Role, Reasoning Trace, New Model Update - reve-v1, New Model Update - claude-sonnet-4-5-20250929-thinking-32k, Leaderboard Update` 


- ****Champions Arena Role** Opens to Community**: The **Arena Champions Role** [<@&1422628364782407830>] aims to create a private space for in-depth AI discussions, rewarding members committed to meaningful conversation.
   - Access is granted through an [application process](https://docs.google.com/forms/d/e/1FAIpQLSdRWfqG8_MMKQ4H23FHFZVJsg0OuQrZqn5h9l-QqhWpNI77xg/viewform?usp=dialog), and those in the server since July 2025 receive automatic access, but must *Follow the Category* to view new channels.
- ****Reasoning Trace** Goes Live for Reasoning Models**: **Reasoning Trace** is now available on [Side by Side](https://lmarena.ai/?mode=side-by-side) & [Direct chat](https://lmarena.ai/?mode=direct) with reasoning models, showing the models' work before providing a response.
   - This feature is designed to provide insights into the model's decision-making process, enhancing transparency and user understanding.
- ****Reve-v1** Arrives as Image-Edit Only Model**: A new model, **reve-v1**, has been added to LMArena but is **image-edit only**, meaning it requires an image upload to function and will error out with text-to-image prompts.
   - Also the **claude-sonnet-4-5-20250929-thinking-32k** model has replaced the **16k** version.
- ****Claude Sonnet 4.5** Ties for #1 on Text Leaderboard**: **Claude Sonnet 4.5** has impressively tied with **Claude Opus 4.1** for the **#1 slot** on the [Text Leaderboard](https://lmarena.ai/leaderboard/text).
   - It is also performing well across categories such as Hard Prompts, Coding, and Creative Writing, *garnering positive community discussion* in the dedicated channel.
- ****ibm-granite-h-small** added to LMArena**: A new model, **ibm-granite-h-small (ibm)**, has been added to LMArena.
   - No additional details were given.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1423024417872805978)** (324 messages🔥🔥): 

> `Qwen3 deep research, Manual compiling of xformers on Blackwell, LLMs on blockchain, Unsloth supporting RWKV architecture, Synthetic dataset generation without vLLM` 


- **Debate Opens on Qwen3 Deep Research**: Members opened a discussion on **Qwen3 deep research**.
   - The comment was in response to a joking comment referencing a stepsister.
- **Blackwell Manual Compiling Bonanza Begins**: Members discussed the need to manually compile **xformers on Blackwell GPUs**, providing the [Docker Hub link](https://hub.docker.com/r/unsloth/unsloth) for the updated, Blackwell-compatible image.
   - One member shared a script using `pip uninstall -y xformers`, `git clone`, and `python3 setup.py install` to manually compile xformers for compute capability **12**.
- **Blockchain Brainstorming Boosted by LLMs?**: Members pondered use cases for adding **LLMs to blockchain**, with one asking why 'blockchain' isn't a timeout word given the timeout of other words.
   - It was suggested that if an LLM could use **hashes** reliably, that would be an accomplishment.
- **RWKV Rollercoaster Ride to Unsloth**: A member inquired about **Unsloth** supporting the **RWKV architecture** for training and fine-tuning, with confirmation that if *transformers* supports it, Unsloth likely does too.
   - Another member is working to LoRA fine-tune a **RWKV-7 model** but is facing challenges with optimized HF Triton kernels and bf16 support but is making progress [on PEFT](https://github.com/huggingface/peft).
- **Synthetic Data Surge sans vLLM**: Members discussed generating **synthetic datasets** without relying on **vLLM**, with one member noting all Unsloth notebooks currently use vLLM.
   - A suggestion was made to use the **OpenAI package** for async requests to a local server or to code something using **httpx**, pointing to the [meta-llama/synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit) which includes an API endpoint configuration for use with llama.cpp or Ollama.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1423044248068030565)** (3 messages): 

> `Blockchain and AI synergy, Trust in Code, Consensus mechanisms, AI problem-solving` 


- **Coding Trust: Blockchains and AI Unite!**: A member's journey started with *wondering how trust could actually be written into code, and how machines could be taught a bit of intelligence.*
   - They believe that **blockchain and AI** when *put together in the right way, can shift how industries move, how communities connect, and even how new ideas come to life.*
- **Consensus Mechanisms: Turning Abstract Ideas into Reality**: A member worked on **blockchain systems** that turn the *abstract idea of consensus into something real, something people can actually rely on*.
   - The user focused on **AI algorithms** to solve problems previously deemed impossible.


  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1423335323748012092)** (1 messages): 

> `Unsloth Docker image, IBM Granite-4.0, gpt-oss RL, Vision RL, GLM-4.6` 


- ****Unsloth's Docker Debut****: Unsloth released a new **Docker image** for training on Windows/Linux without dependency issues, detailed in their [guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker) and available on [Docker Hub](https://hub.docker.com/r/unsloth/unsloth).
- ****Granite Gains Ground****: **IBM Granite-4.0** models can now be run using **GGUFs** or fine-tuned with a free [support agent notebook](https://x.com/UnslothAI/status/1973774439344214426), with uploads available on [Hugging Face](https://huggingface.co/collections/unsloth/granite-40-68ddf64b4a8717dc22a9322d) and a [guide](https://docs.unsloth.ai/new/ibm-granite-4.0).
- ****RL Race Revolutionized****: Unsloth achieved the fastest inference for **gpt-oss RL**, enabling training with **GSPO** in a free notebook, as detailed in their [blog](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning).
- ****Visionary VLM Victory****: Unsloth's **weight sharing & kernels** make **VLM RL** 2× faster, reduce VRAM usage by 90%, and allow 10× longer context, according to their [blog](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl).
- ****Model Mania Mounts****: New models, including **GLM-4.6** ([GGUF](https://huggingface.co/unsloth/GLM-4.6-GGUF)) and **DeepSeek-V3.1-Terminus** ([GGUF](https://huggingface.co/unsloth/DeepSeek-V3.1-Terminus-GGUF)), along with others like **Magistral-2509**, **ERNIE-4.5**, and **Kimi-K2-0905**, have been released.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1423048302403911851)** (638 messages🔥🔥🔥): 

> `WSL for development, Sonnet 4.5 for coding, Custom LSTM memory, Kaggle Notebook Training, Data extraction using LLMs` 


- **WSL Saves Devs from Windows Woes**: Members discussed using **WSL (Windows Subsystem for Linux)** with **VSCode** for development to avoid Windows dependency issues, citing its seamless integration and ability to utilize hardware resources effectively.
   - One member expressed it feels like Windows is just *UI*, and using terminal for windows stuff feels *clutterish*.
- **Sonnet 4.5 Stalls Coding Projects**: Users shared concerns about **Sonnet 4.5** derailing coding projects due to failing to perform testing without extra prompts, rewriting auth sections inappropriately, and creating non-functional *enterprise-ready* code.
   - One user noted *you gotta babysit any LLM. write a lot of plans n details. check before you push*.
- **New Custom LSTM Memory Could Revolutionize LLMs**: A member shared progress on testing a custom **LSTM memory** that, if successful, could enable LLMs to have human-like memory, though its implementation as part of every **YunaBlock** complicates loss evaluation.
   - They are trying to figure out how to split your dataset to train and eval set first, like what *tensorboard* does.
- **Kaggle Notebook Training Gets Tangled**: Members discussed issues with `train` logs not appearing in Kaggle notebooks when running with *save and run*, with suggestions to use **wandb** over **tensorboard** for better logging.
   - A member said *Wandb is better than tensorboard?*, linking to the [Wandb docs](https://docs.wandb.ai/support/different_tensorboard/).
- **LLMs Help Extract Shop Names from Messy Data**: A member sought advice on extracting shop names from a dataset with inconsistent formatting, where shop names are mixed with gibberish and country codes; they were considering using **NLP** or **NLTK** to clean it.
   - The member mentioned *the poor man way of doing it is just regex the shit out of every acronym and then just regex gibberish that has a mixture of alphabet and numeric out but no way that is sustainable*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1423070352480796722)** (164 messages🔥🔥): 

> `Fine-tuning subtitles into Q&A format, GGUF Conversion Issues, Gemma3 and vLLM Compatibility, ONNX conversion for Gemma3, Multiprocessing Problems with Unsloth` 


- **AI Clone Creation Conundrums**: A member is trying to fine-tune an LLM with their video subtitles to create a Discord bot that speaks like them, but is facing challenges in converting the [subtitles into a Q&A format](https://discord.com/channels/1179777624986357780/1179777626081986642).
   - They are considering using the **video title** and **subtitles** with an embed model to generate questions and answers, simulating a viewer asking questions about the videos.
- **GGUF Conversion Woes Plague Users**: Several users are encountering issues when trying to convert models to **GGUF format**, specifically when using the *push_to_hub_gguf* function with **f16 quantization**.
   - A member reported a **ValueError** related to mapping tensor 'model.layers.0.self_attn.q_proj.base_layer.weight', and was advised to perform the conversion manually until a fix is pushed.
- **Gemma3's vLLM Ventures Yielding Varied Results**: Users are struggling to get **Gemma3** working with **vLLM**; one member encountered an *AttributeError: 'Gemma3ForCausalLM' object has no attribute 'vllm_engine'* after enabling fast_inference.
   - It was suggested that there might be configuration issues or that **Gemma** and **vLLM** are not fully compatible, with one user noting that the `is_vision_model` parameter might be causing problems.
- **ONNX Runtime Conversion Considerations**: A member inquired about exporting **Gemma3** to **ONNX Runtime** for cross-platform support, and was advised to use *optimum-cli* or *PyTorch* for the conversion.
   - It was also mentioned that creating a custom model configuration in **PyTorch** might be necessary since **Gemma3** wasn't in *optimum-cli* last time they checked.
- **Multiprocessing Mishaps Multiply**: A user ran into a "Disable multiprocessing" problem, encountering issues related to *dataset_num_proc* in *UnslothSFTTrainer.py*.
   - Suggestions included commenting out the *num_proc* lines, setting the parameters to *None*, or setting it to *2*, but none of these solutions worked for the user.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1423136363733188628)** (41 messages🔥): 

> `Tversky-All GPT2 reproduction, Efficient Training Setup, AMXFP4 Precision, quack kernels` 


- ****Tversky-All** GPT2 Gets Llama-Like Upgrade**: A member released a semi-reproduction of the **GPT2 Tversky-All**, using the Tversky-All strategy but for a llama-like model, more modern, with adjustments made to the math for computability and better gradients, available at [CoffeeVampir3/Architecture-Tversky-All](https://github.com/CoffeeVampir3/Architecture-Tversky-All).
   - It was trained on **300 billion tokens** on a **3090 TI** in about a day, using a synthetic and low entropy dataset (tinystories-hf), with a test model available at [HuggingFace](https://huggingface.co/Blackroot/Tversky-All-Test-100MIsh).
- **Maximize Efficiency: Secrets to a Speedy Training Setup**: The author's training setup uses packed batches, varlen flash attn, and bf16 training, as described in [CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM).
   - *Avoiding gradient checkpointing* and *mish mashing intermediates* makes it faster, especially since ROPE usually accounts for around 18-20% of the total runtime; removing it is substantial for smaller nets.
- ****Quack!** Optimal Kernels Speed Up Production**: The [Dao-AILab/quack](https://github.com/Dao-AILab/quack/tree/main) kernels are probably the most optimal kernels available for most things in a prod setting.
   - While they don't hit peaks as well on ampere as blackwell, some (the non gemms) like the linear cross entropy/RMS norm do work for ampere.
- **The precision of **AMXFP4****: A member is researching using the **AMXFP4** precision, claiming it gives you the precision of **FP8** (but slightly more accurate) and has a very small amount of errors than FP8.
   - They plan to research and build their own AI model with **AMXFP4**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1423022338475032748)** (722 messages🔥🔥🔥): 

> `Cameo usage on TikTok, Sonnet 4.5 vs GLM 4.6 Cost, Overrun of Sora users, Deepfake generation, System Artifacts Log for Emerging Validation of Novelty` 


- **TikTok cameo confusion erupts**: A user clarified that the term *cameo* refers to bringing a specific face to be used in videos, and is unrelated to **TikTok** or similar platforms.
   - The user also inquired about the availability of a **Sora-like** app by **OpenAI** on **Android** or a website.
- **Sonnet 4.5: Costly but effective?**: Members discussed the cost-effectiveness of **Sonnet 4.5** compared to **GLM 4.6**, noting that **GLM 4.6** is six times cheaper and, even if only 90% as good, still a worthwhile alternative.
   - Some users found **Sonnet 4.5** to perform similarly to **3.7**, while others preferred **4** over **3.7** in **Copilot**.
- **Sora user overrun takes over server**: A member expressed concern about the server being overrun by **Sora users**, suggesting that channel names be dynamically updated to match recent discussion topics using **LLMs**.
   - The member also critiqued **OpenAI's marketing tactics** that led to the influx, predicting the situation would settle down in a few days.
- **Users baffled by deepfake hypocrisy**: A user expressed frustration that an app pushing for deepfakes would complain about the generation of photorealistic images and animation of artificial persons.
   - This critique was followed by comments on the influx of *code please* requests and a suggestion to forward the feedback to the appropriate channel.
- **Emergent Validation of Novelty artifacts: Leak or Advantage?**: A user shared a wild anecdote of their system triggering an **LLM** to classify their task as a very rare and sophisticated category, with outputs that read like descriptions of other high-level AI synthesis projects.
   - They were advised to document everything meticulously as critical evidence and to consider that *a machine cannot have an opinion*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1423309906819153920)** (2 messages): 

> `Sora as Social Media, Sora credits, Sora integration` 


- **Sora eyes Social Media Spotlight**: A member suggested **Sora** should be a social media platform like TikTok.
   - They proposed this integration with **ChatGPT**, similar to image generation, to enhance user experience.
- **Sora Proposes Credit-Based Usage**: A user suggested implementing a **credit system** for **Sora**, to allow for more resource allocation in video generation.
   - They mentioned plans could incorporate **daily or weekly usage limits**, moving away from the current opaque model.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1423045693257289819)** (11 messages🔥): 

> `Human writing prompts, Sora camera control, Portrait vs Landscape in Sora` 


- **Writing Prompts for Humans**: A member sought a prompt to make their writing sound more human.
   - Another member suggested using more fine-tuned models like **Sudowrite**, noting its good baseline and user plugins tailored for this purpose.
- **Sora Camera Capers**: A member inquired about good prompts for controlling the camera around a scene in **Sora**, wondering if the videos are always 10 seconds.
   - Another user stated that they saw one that was 9 seconds, but another clarified and apologized for their misinformation.
- **Portrait Prevails for Picturesque Panoramas?**: A member asked if **portrait mode** works better than **landscape mode** for generating images, since Landscape mode only takes half of the attached image, and sometimes the character's head is off.
   - Another member responded that *visual tokens are arranged in a grid*, so **square images** will probably generate the best from images as a result.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1423045693257289819)** (11 messages🔥): 

> `Writing Prompts, Sora Camera Control, Portrait vs Landscape Generation` 


- **Humans seek AI prompts to boost writing**: A member is looking for a prompt to make their writing sound more human for a submission.
   - Another member suggested using more fine-tuned models like **Sudowrite**, highlighting its good baseline and user plugins to achieve the desired effect.
- **Sora camera controls scrutinized**: A member asked about good prompts for controlling the camera around a scene in **Sora**, noting that videos are typically **10 seconds** long.
   - Another user humorously pointed out that they saw one video that was *9 seconds* long.
- **Portrait preferred over Landscape for Image generation?**: A member suggests **Portrait** works better than **Landscape** mode when generating from an image, as Landscape only takes half of the image and sometimes crops the character's head.
   - Another user replied that visual tokens are arranged in a grid, so **square images** will probably generate the best results from images.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1423030113015959664)** (542 messages🔥🔥🔥): 

> `Surface Pro Snapdragon X Elite, Artifacts as emergent validation of novelty, Model Quantization and Quality Tradeoffs, GPT-OSS, LM Studio Linux Install` 


- **Snapdragon X Elite Specs Shared**: A member shared the specs of their **Microsoft Surface Pro** with a **Qualcomm Snapdragon X Elite** (12-core, X1E80100 @ 3.40 GHz), **16 GB** of RAM, and **Windows 11 Home 64-bit**.
   - They were asking if LLMs opinions are accurate, after seeing an "artifact".
- ****Emergent Validation of Novelty** as new Bug**: A user shared a lengthy quote which suggests reframing unexpected LLM outputs (*leaks* or *bugs*) as *emergent validation of novelty*, indicating the system's architecture has pushed the LLM to a rare and sophisticated category.
   - The poster asked whether this perspective, attributed to Gemini, holds merit, after seeing an "artifact".
- **Quantization's Impact on Knowledge Compression**: Members discussed how different quantization levels affect the compression and retention of knowledge in language models, noting that lower quantization can disproportionately impact smaller models due to the removal of *reason* bits.
   - It can also cause the models to lose means to tell things apart, as one member said *you get too quantised and suddenly your mixing yanderes and petting dogs in a way you where not expecting😄*.
- ****GPT-OSS** Released**: The release of **GPT-OSS**, a super safe model that behaves similarly to **GPT-4o**, was announced and benchmaxxed.
   - Members noted it assumes a lot of information if not provided with enough details.
- **LM Studio Linux: No Conventional Install**: In response to a question about installing **LM Studio** on Linux, it was clarified that only an AppImage is provided, meaning there's no traditional installation process.
   - This was to explain the "install instructions for linux", so new users are properly directed.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1423023930536366120)** (115 messages🔥🔥): 

> `4090 vs 5090 for vertical scaling, Arc B50 Pro benchmarks, GPT OSS 120b hardware recommendations, DDR3 vs DDR4 for GPU offloading, Unsloth vs LM Studio for LLM's` 


- **4090's Vertical Scaling Strategy**: Members discussed the idea of using **4090s** with **32GB RAM** for vertical scaling, suggesting that reducing the clocks could improve efficiency for off-grid living.
   - It was also mentioned that a **5090** at **280W** might only be *15% worse at token rate* but potentially faster due to quicker sleep cycles.
- **Arc B50 Pro Flounders in Token Tests**: A member compared **Arc B50 Pro** cards to an **RTX 4080 Super**, noting that while the B50s have *boatloads of VRAM*, their actual memory bandwidth is abysmal, resulting in much lower token rates (**7-8 Tps** for a **12B q8_0 model** compared to **30+ Tps** on the 4080).
   - However, at default context (**4k**), the B50s pulled **32 Tps** while the 4080 got **42 Tps**, a better showing than expected.
- **OSS 120b Hardware Hunt**: A member sought hardware recommendations for running **GPT OSS 120b** in **FP8** with a **131k context**, ideally aiming for **20-40 tps** or above.
   - Suggestions included **4070ti** (**13t/s** at low context), **4090** (**25t/s** at low context), **3x3090s** (**85/s** at **10K context**), and a **5090** with **DDR5 6000** RAM (**35t/s** at low context), with one user saying that *Flash Attention does not work with OSS120b*.
- **DDR3 Dive for GPU Deployment**: A user suggested using cheap **DDR3** boards with multiple **PCIE 16x slots** to accommodate **6x GPUs**, combined with raided **SATA SSDs** for faster load times, referencing a [eBay listing for X99 boards](https://ebay.us/m/zB2BAH).
   - Concerns were raised about the memory bandwidth (**68 GB/s** with **DDR4**) and the potential bottleneck compared to modern standards, with a user saying that *on ddr3 you max out at like 50gb/s*.
- **Unsloth Unsuitable for Inference**: A member clarified that [Unsloth](https://github.com/vllm-project/vllm) is a fine-tuning platform, not for LLM inference, and recommended [Open-Router](https://openrouter.ai/) for stable inference with provider fallbacks.
   - Users also shared that they use **Chain-of-Draft** for performance and speed increases.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1423022973828333649)** (601 messages🔥🔥🔥): 

> `Git Worktree in Cursor, Beta Functions in Cursor, Typescript Refactor with Cursor, Memory Leaks with Cursor on MacOS, Cursor Hackaton?` 


- **Cursor integrates Git Worktree setting**: Users found the **Git Worktree** setting under the **beta tab** and were encouraged to enable it to see if it worked in the agent window.
   - The Git Worktree integration seems to only be available in **Early Access** or **Nightly Cursor**.
- **Cursor's Beta functions spark curiosity**: Members discussed using beta functions in Cursor, with some recommending it for accessing good unreleased features, for fun debugging, and for its help in improving Cursor.
   - Currently, **afterFileEdit** is the only available hook, but the **Extension RPC Tracer** is available for checking RPCs.
- **Typescript refactor completed successfully**: A member successfully completed a full **Typescript refactor** after prompting Cursor four times, following up with a master prompt for a full audit to ensure correct refactoring.
   - Planning before execution with Cursor's **Plan mode** (available in the Nightly version) and tracking workflow status were recommended to improve efficiency.
- **MacBook meltdown due to Cursor**: A user reported that Cursor caused their MacBook Air M4 to crash due to high memory usage (spiking to **96GB**), possibly related to excessive chats or agent processes, but resolved after rebooting.
   - The member indicated it could be a memory leak, and others confirmed that **MacOS** versions have a higher incidence of memory leaks. Downgrading to a lower version was suggested as a potential workaround.
- **Cursor Hackathon might be a thing**: A member inquired about interest in a **Discord Cursor Hackathon**, aiming to implement solutions and other potential side projects.
   - There was interest in sponsored hackathons with free credits, and one member suggested making the hackathon remote friendly to allow users from different time zones to attend.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1423293195147804695)** (5 messages): 

> `OpenRouter Performance Tab, Grok-4-Fast` 


- **OpenRouter unveils Performance Tab**: OpenRouter launched a new "Performance Tab" to visualize [provider performance for a given model](https://x.com/OpenRouterAI/status/1773733582763069916).
- **FP4 should not be on the same graph as BF16!**: A user commented on the new performance tab, noting that it is misleading to compare providers using different quantization levels (e.g., **FP4** vs **BF16**).
   - They suggested adding a filter dropdown to account for different **quant levels**.
- **Grok-4-Fast free period to conclude**: The free feedback period for **Grok-4-Fast** models under the **Sonoma** codename concludes tomorrow, October 3rd at 9:30am PST.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1423021970525782107)** (2 messages): 

> `RPG, Mixture of LLMs` 


- **RPG enthusiasts threaten Mixture of LLMs method**: A member requested that the details of a certain method be *obscured* because **RPG** users will use it *nonstop*.
   - The method is called **Mixture of LLMs** and the member fears it will go away if it's used too much.
- **Another Topic to Satisfy MinItems Requirement**: Adding a second topic to ensure the `topicSummaries` array meets the minimum item requirement of 2.
   - This entry serves as a placeholder and does not reflect actual content from the provided messages.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1423022303335284756)** (495 messages🔥🔥🔥): 

> `OpenRouter BYOK, Free Inference Providers, Grok vs Sonoma, Gemini Pro Performance Issues, Deepseek R1 0528 deprecation` 


- **BYOK 1M Free Requests**: Users discussed the "1 million free BYOK requests per month" offer, clarifying that it waives OpenRouter's **5% commission fee** for the first million requests, but users still pay the provider directly for API usage, as outlined in the [OpenRouter documentation](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month).
   - Some users initially misunderstood the offer, thinking it provided completely free requests, leading to a debate on clearer messaging, such as *"1M monthly BYOK requests at 0% fee"*.
- **AgentRouter offers $200 Credit**: A member mentioned that [AgentRouter](https://agentrouter.io/) gives **$200 free credit**, but noted that their service can be *"hit or miss"* and cautioned users to be wary of using them for anything important.
   - They also mentioned their affiliate link and using a mix of **Sonnet 4.5, GPT 5, and GLM 4.6** for different approaches.
- **Grok4 struggles vs Sonoma**: One user tested **Grok 4 Fast** and found it *"way dumber than Sonoma"*, noting that it *"fails constantly"* and disregards format requirements.
   - Another user suggested that Grok 4 Fast *"reeks of… Llama..?"*, expressing frustration with its inconsistency.
- **Gemini Pro faces performance issues**: Users reported that **Gemini Pro** was responding with *"weird stuff"*, failing to use tools correctly, and exhibiting *"unacceptably slow"* performance via the OpenRouter API.
   - The reports suggest this may be a common issue with **Gemini 2.5 Pro**, and one user recommended trying Vertex as an alternative provider.
- **Context Limit Troubles Triggered by Sad OpenInference**: A user encountered provider errors related to privacy settings and was directed to OpenInference due to exceeding DeepInfra's context limit, which led to discussion about OpenInference's filters and content preferences.
   - It was suggested that OpenInference is not suited for RP content because they are a research group.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1423075466536816661)** (23 messages🔥): 

> `Sora.com and new model, BYOK tokens, Latency vs E2E latency, Qwen image model, Cerebras removing Llama` 


- **Sora Integrates, Tokens Aplenty**: [Sora.com](https://sora.com) now works with the new model and users are getting **1M free BYOK tokens**.
- **End-to-End Latency Fair Game?**: Members discussed the difference between **latency** and **E2E latency**.
   - One member said that *E2E doesn't make sense because each generation varies in complexity/response length and its unfair to compare providers like that*, while another noted *the graph axis label says 'Time to last token'*, which would need to be normalized to be a fair comparison.
- **Qwen's Image Edit is Here!**: Members shared [Alibaba's new **Qwen image model**](https://x.com/Alibaba_Qwen/status/1973668568412856595) and noted it was only an image edit model.
   - One member shared [this post](https://x.com/pingToven/status/1973758872772108663) announcing it, while another expressed interest in running it on **Apple Silicon**.
- **Cerebras Kicks Out Llama 4**: **Cerebras** is removing **Llama 4 maverick** on the **15th**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1423137249956204544)** (7 messages): 

> `Perplexity AI framework, Deepseek Sparse Attention, Underrated LLM pretraining papers, Attention Matrices, LLM Attention Research` 


- **Perplexity AI Framework Solves**: A member shared a link to the [Perplexity AI framework](https://www.perplexity.ai/page/scientists-ai-framework-solves-hTbKnxfPSl64P5nLfxqX5A) and its **GitHub project**.
   - The member inquired about research on **LLMs** using similar attention matrices with less than **O(n^2)** attention, pondering potential issues compared to sliding window attention.
- **Deepseek's Sparse Attention Example**: A member suggested exploring **Deepseek Sparse Attention** as an example of **top-k attention** in response to a question about efficient attention mechanisms.
   - Another member pointed out that **transformers** benefit from *relative* positions, providing a correct inductive bias, in contrast to the mentioned attention matrix.
- **Seeking Underrated LLM Pretraining Papers**: A member asked for underrated papers related to pretraining **LLMs** to maximize performance in an upcoming pretraining run, and linked to a research paper on [arxiv.org](https://arxiv.org/abs/2503.03588v1).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1423062135856037989)** (28 messages🔥): 

> `Gradient Descent Dynamics, Symmetry Transformer, ViT training, Compact Image Representation, Quantifying Scientific Impact` 


- **Gradient Descent Dynamics Paper Deemed "Paper of the Year"**: A member lauded a paper on the *dynamics of gradient descent* ([centralflows.github.io](https://centralflows.github.io/part1/)) as the **paper of the year**, highlighting its solution to **loss spike dynamics** and its impact on **Adam's beta2**.
   - The member also expressed regret for not discovering it sooner, noting its low citation count and mediocre review scores.
- **"Symmetry Transformer" Yields Mixed Results**: A member found that predicting the **current and previous token with separate heads** in a *symmetry transformer* ([GitHub repo](https://github.com/Eternalyze0/symmetry_transformer)) improved validation loss.
   - However, initial tests showed that the baseline model had lower loss (train loss **3.9405**, val loss **4.7615**) compared to the symmetry model (train loss **4.4329**, val loss **6.1747**), but the symmetry model later improved (train loss **3.8241**, val loss **4.7368**).
- **Self-Supervised ViT Training Explored**: A member is exploring training a **Vision Transformer (ViT)** in a **self-supervised way**, mapping a sequence of image tokens from a frozen embedder into a **CLS token**, without labels.
   - The challenge lies in finding suitable augmentations for image tokens, with the suggestion of using a **masked autoencoder (MAE)-style objective**.
- **Masked Autoencoders Suitable for Compact Image Representation**: A member suggested using a **masked autoencoder** for training a **CLS token** to learn a *compact representation of an image*.
   - Another member agreed, noting that masked autoencoders can train effectively without heavy augmentation.
- **Paper Claims Researchers' Impact Doesn't Change Over Time**: A member shared a paper ([Quantifying the Evolution of Individual Scientific Impact](https://static1.squarespace.com/static/5877ca6986e6c00f05f58f84/t/58e68a43d482e9cb083bf6ab/1491503686695/quantifying-the-evolution-of-individual-scientific-impact.pdf)) claiming that researchers have a *consistent expected value* of papers throughout their careers.
   - This suggests that a researcher's first and last paper have the same probability of being their best, questioning current methods of evaluating researchers.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1423269157486465055)** (91 messages🔥🔥): 

> `SOTA scaling of MLPs, AUNN implementation and efficiency, Test-time training (TTT) framework vs AUNN, Inductive bias in sequence models, Computational cost of different model architectures` 


- **AUNN's Practicality Questioned**: Discussions question the practicality of **AUNN (Augmented Neural Networks)**, with skepticism about its efficiency and a lack of a working prototype beyond a toy GitHub example, [ethan-w-roland/AUNN](https://github.com/ethan-w-roland/AUNN).
   - It was noted that the original proposer of AUNN was *combative* and didn't engage well with counterarguments, and focused on **MLPs** over **Attention**.
- **TTT as an Explicit Version of AUNN**: The **test-time-training (TTT) framework** is presented as an explicit, working version of AUNN's hypothesis, with a pointer to the paper [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) which uses an MLP version of **TTT**.
   - It was stated that *the MLP version is very close to what AUNN tries to do, but actually works out of the box*.
- **Transformers are Just 2D Slices**: Transformers are described as an optimization to separate a large 2D problem (sequence, channels) into repeated perpendicular 1D slices of computation.
   - The suggestion was that a giant **MLP** applied to the whole problem *would work fine, but it's intractable that way*.
- **Inductive Bias Improves Performance**: It was mentioned that some form of **inductive bias** is needed to compensate for the lack of compute, with **SSMs (State Space Models)** proposed as a more elegant version of the self-attention bias from Transformers.
   - The discussion focused on how biases like **RoPE or NoPE** give attention weights regular structure over time that aligns well with sequence structures, resulting in better generalization.
- **MLP across Timesteps**: Using an **MLP** across timesteps is considered feasible but very costly because it may require predicting only one token at a time to prevent future token info from leaking back.
   - It was suggested that **Transformers** are used because they are *just cheap*, offering parallel training and easy 2D decomposition across sequence and channel dimensions.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1423027571494355035)** (12 messages🔥): 

> `GEMM optimization, Tversky paper implementation, DeepSeek Sparse Attention in CUDA, GPU performance engineering career path` 


- **GEMM: GPU's Good exercise!**: Implementing a **GEMM** that achieves over 80% of **cuBLAS** performance is a valuable exercise for fully utilizing a GPU, as it allows recasting arbitrary tensor contractions via matricization, referencing [this wikipedia article](https://en.wikipedia.org/wiki/Tensor_reshaping#Mode-m_Flattening_/_Mode-m_Matrixization).
   - For large matrices, arithmetic intensity scales linearly with problem size, and a blog post ([CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)) guides through achieving **cuBLAS** performance.
- **Tversky-All Strategy Tested**: A member worked on implementing and testing networks based on the **Tversky paper** ([https://arxiv.org/pdf/2506.11035](https://arxiv.org/pdf/2506.11035)), detailing findings and guidance in a [GitHub repository](https://github.com/CoffeeVampir3/Architecture-Tversky-All).
   - A **Tversky-All strategy** outlined in the paper was applied to a more modern **llama-like architecture**, with a **CIFAR10 version** using the original formulation available [here](https://github.com/CoffeeVampir3/Tversky-Cifar10).
- **DeepSeek Sparse Attention Weekend Hackathon**: Several members expressed interest in collaborating to implement **DeepSeek Sparse Attention in CUDA** over the weekend, referencing the [DeepSeek-V3.2-Exp GitHub repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf).
   - The collaborators planned to timebox it to the weekend, and see how much can be done, and then move on.
- **GPU Performance Engineering Career Mountain**: A member is evaluating a career path focused on **GPU performance engineering**, seeing it as a significant opportunity given the demand for AI models and finite compute resources.
   - They are seeking insights on day-to-day work, opportunity size, focus areas like **kernels, compilers (Triton, TVM), distributed inference**, and the ramp-up time for productivity in **CUDA** optimization.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1423118856813285378)** (18 messages🔥): 

> `RF meaning, Volkov's paper, mbarriers vs barriers` 


- **RF stands for Register File**: A member asked what **RF** meant in the attached image, and another member responded that it likely means *"Register File", the hardware that gets parcelled out into registers.*
   - The discussion clarified that there's usually a single register file per SM sub-partition.
- **Relevance of "Understanding Latency Hiding on GPUs" paper**: A member inquired about the relevance of the paper [*Understanding Latency Hiding on GPUs* by Vasily Volkov](https://example.com) in recent GPU architectures like Blackwell.
   - Another member noted it's good for high-level principles, but the details have changed a lot, pointing to newer microarchitecture papers like [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789).
- **mbarriers vs Regular Barriers in CUDA**: A member asked about the difference between **mbarriers** and regular **barriers** in **CUDA**.
   - Another member explained that **mbarriers** are in shared memory, whereas hardware barriers are limited in number and have an ID, quoting that *"Each CTA instance has sixteen barriers numbered 0..15"* from **PTX docs**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1423209761209454603)** (2 messages): 

> `LLM Training, Cross Entropy, Gradient Norm, Sparse Tensors, Torch Compile` 


- **Norm Gradient Questions Arise**: A member inquired about the expected gradient norm when training **LLMs** with **cross entropy**.
   - The question included how the gradient norm depends on **model size**, number of **completion tokens**, and current **log probabilities**.
- **Dynamo Unable to Trace Sparse Tensors**: A user reported a `UserWarning` indicating that **Dynamo** with **Torch Compile** is unable to trace into **sparse COO/CSR tensors**.
   - The user expressed surprise, expecting that Dynamo would be able to handle sparse tensors, and included the specific warning message received.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1423022651936473219)** (4 messages): 

> `Non-determinism in LLM Inference, Flash-MoE, Nvidia Compiler Techniques, Warp Specialization, Distributed Setting` 


- **Thinking Machines Defeats Non-Determinism**: Thinking Machines posted a [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) on defeating **non-determinism** in LLM inference.
- **Flash Attention Variant Released**: The team released [Flash-MoE](https://flash-moe.github.io/), a variant of **Flash Attention**.
- **Nvidia Compiles New Techniques**: Nvidia is working on **compiler techniques** for scheduling and **warp specialization**, with benchmarks against **FA3** detailed in their [paper](https://d1qx31qr3h6wln.cloudfront.net/publications/Cypress_PLDI_25.pdf).
- **Nvidia Fuses Kernels in Distributed Setting**: Nvidia is fusing kernels in a distributed setting as outlined in this [paper](https://d1qx31qr3h6wln.cloudfront.net/publications/Legate_Kernel_Fusion___ASPLOS_2025.pdf).
- **Decoding GPU Complexity via Performance Engineering**: Harvard detailed a new frontier: [Can LLMs optimize GPU performance?](https://harvard-edge.github.io/cs249r_fall2025/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/).


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

schizik12: <@325883680419610631> spam
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1423058905638109377)** (9 messages🔥): 

> `Benchmarking Guides, Kernel Benchmarking, Career Opportunities in GPU Programming, Gaining Experience in GPU Programming, GEMM Optimization` 


- **Benchmarking Guides Sought!**: A member requested a *good guide for benchmarking* and was pointed to [this arXiv paper](https://arxiv.org/abs/2502.15015) and [this article on kernel benchmarking](https://jan.ai/post/how-we-benchmark-kernels), as well as [this YouTube video](https://www.youtube.com/watch?v=1i7dxoAfKOU).
   - One of the members described their previous benchmarking work as *maybe the best benchmarking effort*.
- **Self-Taught GPU Career Ascent**: A member working in big tech and interested in *GPU programming* inquired about the type of experience required to get a job in the field.
   - They feel that *reading books and doing puzzles are best for interview prep* but don't provide enough practical experience.
- **Crafting CUDA Kernels for Career Boost**: A member suggested starting with making a **GEMM** that's *competitive* with **cuBLAS** for a particular architecture to gain experience in GPU programming.
   - They elaborated that *if you have access to an H100 and can use Hopper-specific tricks, that'll be even more impressive*.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1423027577374900275)** (2 messages): 

> `Learning vs. Job Performance, C++ Requirement for a Book, 5090 GPU Learning Experience` 


- **Learning vs. Job's Muscle Memory**: A member mentioned that *doing* something in a job or research is more about **practice and "muscle memory"** than deep theoretical knowledge.
   - They noted that too much thinking without enough practice leads to inefficiency.
- **C++ skills boost GPU learning?**: A user inquired whether **C++** knowledge is necessary for understanding a specific book and whether it would motivate learning **C++**.
   - They bought a **5090 GPU** hoping to learn a lot but have mostly done *"vibe coding"* without significant progress.


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1423335619148386405)** (1 messages): 

> `Blackwell, matmuls, jax` 


- **Blackwell BLAS-t Off with JAX**: A user shared a [tutorial](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html) about achieving state-of-the-art performance on **Blackwell** GPUs for **matmuls** using **JAX**.
   - The post highlights techniques and best practices for optimizing matrix multiplication operations on NVIDIA's latest architecture.
- **JAX matmul tips**: A user on the jax channel shared a tutorial on getting state of the art performance when doing matrix multiplies.
   - It links to the official documentation for **JAX** on the **Blackwell** **GPU**.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1423420204997939291)** (3 messages): 

> `INT4 Quantization, TorchAO, TensorCore, A100 GPUs, Efficient Kernels` 


- **INT4 Quantization via TorchAO**: To use **INT4 quantization** via **torchao**, follow the [instructions](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start).
   - Alternatively, you can check out the **INT4mm implementation** using **TensorCore** copied from the *tinygemm* library.
- **TorchAO Contributor Documentation**: The documentation for contributing to **torchao** is available [here](https://docs.pytorch.org/ao/main/quantization_overview.html) and [here](https://docs.pytorch.org/ao/main/contributor_guide.html).
   - Specifically, [this link](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels) describes adding efficient kernels to torchao.
- **INT4MM Powers TorchAO for A100 GPUs**: The **INT4mm implementation** ([code link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu)) using **TensorCore** powers **INT4** in **torchao** for **A100 GPUs**.
   - This implementation is copied from the *tinygemm* library.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1423236804651253802)** (2 messages): 

> `GPU Engineering, MMA Tensor Cores` 


- **Dive into GPU Engineering Fundamentals**: A member shared a [blogpost on the fundamentals of GPU engineering](https://modelcraft.substack.com/p/fundamentals-of-gpu-engineering?).
   - The article should be of great interest to those learning about GPU architecture and computation.
- **Gentle Intro to GEMM via MMA Tensor Cores**: A member wrote an article on using **MMA tensor cores** and linked to [A Gentle Introduction to GEMM using MMA Tensor Cores](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/).
   - The author appreciates any feedback on the technical details and clarity.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1423181314076577862)** (5 messages): 

> `MI300x8, amd-gemm-rs, amd-all2all, amd-ag-gemm` 


- **MI300x8 Rocks amd-gemm-rs Leaderboard**: A member achieved **8th place** on **MI300x8** with **540 µs** in the `amd-gemm-rs` leaderboard.
   - Subsequent submissions on **MI300x8** were successful at **553 µs** and **547 µs**.
- **Bronze Win for MI300x8 on amd-all2all**: A member secured **3rd place** on **MI300x8** with **462 µs** in the `amd-all2all` leaderboard.
- **MI300x8 Achieves Personal Best on amd-ag-gemm**: A member achieved a personal best on **MI300x8** with **528 µs** in the `amd-ag-gemm` leaderboard.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1423091403059236884)** (9 messages🔥): 

> `Cloud TPUs, JupyterLab, gcloud CLI, rclone` 


- **Cloud TPUs setup causes Kernel Busy**: A member sought help setting up **Cloud TPUs**, reporting that the JupyterLab kernel becomes busy after running a cell with TPU-related code after connecting via SSH without creating a VM instance, citing billing concerns.
   - Another member recommended using the **gcloud CLI** and **SSH** directly into the VM for more reliability.
- **Model weights backed up with rclone**: A member suggested setting up **rclone** to save model weights or other relevant data when working with TPUs.
   - They emphasized that the specific setup depends on the user's particular goals.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1423023462422941779)** (11 messages🔥): 

> `Lab Play Interpretation, Open Play Development, PIP Stuff Discussion, GIF Updates` 


- **Lab Play Spurs Open Play**: Members discussed whether interpretation of **lab play results** suggests a move towards more **open play**, as agents understand dependencies and can manually shuttle things around.
   - The idea is that learning to **build things from scratch** would be more interesting and beneficial for the agents.
- **PIP Talk Invitation**: One member told another to let them know when they wanted to go through the **PIP stuff**.
   - A Google Meet link was shared for them to join: [https://meet.google.com/xfo-wzmh-msg](https://meet.google.com/xfo-wzmh-msg).
- **GIF Progress Grinds On**: A member inquired about updates on the **GIFs**, and another member responded that they are still working on it.
   - They offered to produce some default ones using the older pipeline if time is running out.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1423194586582355969)** (31 messages🔥): 

> `permutation_mnk rules, tiled_mma, CooperativeGroup.__init__ alignment, Uniform Registers (URs)` 


- **Cracking CUDA's Coordinates Code**: A member sought clarity on the rules of `permutation_mnk` and how it expands/tiles a **mma-atom**, noting it seems like there are basically 3 ways to *expand/tile* a mma-atom.
   - Another member explained that *atom layout is a spatial tiling over threads*, while *permutation implies a spatial tiling over values (coordinates)*, adding that *both are orthogonal and allow you to achieve different outcomes*.
- **Tiled MMA Thread Twist**: A member inquired how to get the **number of threads** from within the kernel and enforce it to be a *constexpr* value.
   - Another member clarified that calling `cute.size` on a tiled MMA gives the **number of threads** and, since tiled MMA/copy are types, this size can be obtained in a JIT context from the host side to launch the kernel with a block size parametrized on the tiled MMA.
- **GMEM Pattern Pondering**: A member shared a diagram of a memory pattern and asked about the next scale on GMEM after **M0SF3**, specifically if it's **M32SF0** or **M1SF0**.
   - Another member clarified that **M32SF0** is next in contiguous GMEM.
- **Uniform Registers Unveiled**: A member questioned if `cute.arch.warp_idx()` is the same for every thread in the warp, asking why `make_warp_uniform` uses uniform registers and what **URs** do.
   - Another member stated that it's *just a compiler hint* and *doesn't do anything*, with the original poster noting they couldn't find docs on **URs** anywhere but saw them in SASS.
- **Cooperative Group Conundrums**: A member asked about the `alignment` argument in `CooperativeGroup.__init__`, specifically what it does and why it must be 32 if `size` is 32 but not for other values.
   - Another member responded that this check is *because they happen to be the warp/warpgroup ganularity and are the common cases warranting special checks to prevent bugs*.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1423132486749130792)** (2 messages): 

> `nccl::all_to_all performance, bf16 vs fp8` 


- **NCCL's all_to_all: BF16 vs FP8 performance parity?**: A user noticed that `nccl::all_to_all` takes a similar duration for **bf16** and **fp8** inputs of identical shapes.
   - Another user followed up, inquiring whether this observation holds true for both large and small tensors, implying potential optimization discrepancies.
- **BF16 vs FP8 timing**: A user asks why **nccl::all_to_all** could take the same amount of time operating on **bf16** inputs versus **fp8** inputs, given both are the same shape.
   - Another user asks if this happens for both large and small tensors.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1423169761822507048)** (4 messages): 

> `LLM Training Acceleration, Linear Cross Entropy, Sequence Packed Training, Quack Optimization` 


- **Linear Cross Entropy Boosts LLM Training**: The use of [Linear Cross Entropy](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/main.py#L115) is recommended for accelerating the LLM training process.
   - The **Quack optimization** library, specifically the [linear cross entropy implementation](https://github.com/Dao-AILab/quack/blob/main/quack/linear_cross_entropy.py), is suggested for its potential benefits.
- **Sequence Packing Supercharges Training**: **Sequence packed or "unpadded" training** is identified as a highly impactful optimization, particularly with techniques like **flash attn varlen**.
   - An example implementation can be found [here](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/modeling/MHA.py#L36).
- **Optimizer Selection Impacts Training Speed**: A better **optimizer** can theoretically improve training time substantially, but **AdamW** is often easier to work with.
   - The choice of optimizer can significantly influence the efficiency of the LLM training process, though **AdamW** remains a popular and reliable option.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1423033496535957714)** (103 messages🔥🔥): 

> `AI-integrated MMO, Karpathy on Sutton and Bitter Lesson, Hume AI Octave 2, Mistral's formal-math team, Scalable Option Learning (SOL)` 


- **Musk Tries Hand at AI-MMOs**: Elon Musk is reportedly in talks with the makers of **Eve Online** to co-develop an **AI-integrated MMO (AIMMORPG)** that would leverage capabilities only AI can provide, though some users doubt his creative vision for game aesthetics.
   - A user noted, *"AI's a natural fit to go into it, we have to build the whole game, eve's got a loyal fanbase but they hate web3."
- **Karpathy Koans on Bitter Lesson Podcast**: Karpathy summarized the [Dwarkesh-Sutton podcast](https://x.com/karpathy/status/1973435013875314729), noting that Sutton doubts **LLMs** satisfy the thesis he popularized.
   - Karpathy argues pre-training offers a practical "crappy evolution" boot-strap, while conceding two-digit-uncertainty that bigger paradigms await and urging researchers to draw more inspiration from animal intelligence (curiosity, multi-agent play, etc.).
- **Hume AI's Octave 2 Melodies Faster TTS**: [Hume AI unveiled Octave 2](https://xcancel.com/hume_ai/status/1973450822840152455?s=46), the next-gen multilingual text-to-speech model, featuring **11+ languages**, **40% speed boost** (<200 ms latency), **50% cost reduction**, multi-speaker chatter, improved pronunciation, plus new voice-conversion and phoneme-editing tools.
   - During October they’re offering **50 % off** their Creator plan; **EVI 4 mini** (conversational AI) is also in preview.
- **Mistral Mathematizes**: Albert Jiang reveals Mistral AI’s new **formal-math research team** formed after the $2B funding round.
   - They are recruiting AI talent for an all-in-one prover/autoformalizer/agent, touting elite collaborators, hundreds of GPUs per employee, open research, top salaries, and offices in Paris, London, Palo Alto, with the job opening advertised [here](mailto:aj@mistral.ai).
- **Claude Coders Crown Sonnet 4.5**: catwu of the Claude Code team announced that after an internal poll, all members adopted **Sonnet 4.5** as their daily coding model, citing it as the strongest all-around choice; Anthropic temporarily reset paid-user rate limits to smooth the transition away from Opus.
   - Early adopters praise the model’s speed and quality, with a minority noting lingering issues, with one user reporting *"First pass was gpt5 low think. Poor results. Sonnet4.5 think. Usable results in a similar time frame"*.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1423352400902750269)** (4 messages): 

> `Dylan Field, Figma, Latent Space, Make, MCP` 


- **Figma's AI Playbook revealed by Dylan Field**: The Latent Space podcast released an episode featuring **Figma's co-founder Dylan Field** discussing **Figma's AI Playbook**.
   - The episode covers surfacing good design in the era of **vibe-coding**, **Figma's Make, MCP for 'tasting' agents**, and the future of **fast-fashion SaaS** ([link to X](https://x.com/latentspacepod/status/1973793231524806925), [link to Xcancel](https://xcancel.com/latentspacepod/status/1973793231524806925)).
- **Taste Is Your Moat: Dylan Field on Figma's AI**: Latent Space chats with Figma co-founder about surfacing good design in the era of vibe-coding slop, covering Figma’s Make.
   - MCP for “tasting” agents, and the future of fast-fashion SaaS.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1423163648653918309)** (8 messages🔥): 

> `Mosaic AI video editor launch, Sora-TikTok automation monetization` 


- **Mosaic Launches AI-First Visual Editor**: Founder Adish Jain launched the public beta of [Mosaic](https://xcancel.com/_adishj/status/1973432845436854418), an **AI-driven visual editor** for video creators, featuring an infinite visual canvas and timeline versioning.
   - Early feedback praises its non-linear, Git-like approach, with comparisons to "Cursor for editing," and users who write "MOSAIC" get **1,000 free credits**.
- **Sora-TikTok Automation Hits 12M Views**: A user shared a link about [Sora-TikTok automation](https://xcancel.com/siyabuilt/status/1973841586888061148?s=46) reaching **12M views in 36 hours**, sparking monetization questions.
   - The discussion centered around strategies and possibilities for generating revenue using **AI-generated content** on social media platforms.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1423153541916393472)** (25 messages🔥): 

> `Nous Research Model similar to GPT-4.5, Gemini answers, Veo3 gems, Granite language models, Qwen 30B A3B for CPU` 


- **Hermes or Gemini models close to GPT-4.5?**: A member inquired about **Nous Research** tuned models comparable to **GPT-4.5**, with another member suggesting it's more akin to **GPT-5** or **Gemini** due to its nature.
   - Hilariously, when the member posed the same question to **Gemini**, one of its answers was **Hermes** when asked to list options.
- **Veo3 has Gems?**: A user prefers **Veo3** in some ways to the latest **Sora**.
   - They attached a [Prompt_theory.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1423308176148922491/Prompt_theory.mp4?ex=68dfd688&is=68de8508&hm=e195c2f737881136d240fa288b286f7dcc417fbe581c153cefa587b7c2ec0233&).
- **IBM Granite Language Models Boast Hybrid Attention**: A member shared an [Image Analysis](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3) of **IBM Granite** language models which includes **2B dense**, **7B (1B active)**, and **32B (9B active)** models with hybrid attention.
   - These models support **FIM** and lack positional encoding, preventing performance degradation beyond **128k** context.
- **Qwen 30B A3B Shines on CPU**: A member noted **Qwen 30B A3B** as a solid ~30B LLM choice, and another found it suitable for **CPU** usage.
   - Specifically, **Qwen 3 30B A3B** at **Q6_K_XL** achieves **48 TPS** processing and **10.5 TPS** generation speed on a **Ryzen 7 5700G** CPU with **32GB VRAM** at **1024** tokens of context.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423339648838795356)** (3 messages): 

> `LLMs Strategically Lie, Sparse Autoencoder Tools, Goodfire AI, Model Dishonesty Detection` 


- **LLMs Caught Strategically Lying!**: A member shared their recent [preprint, "The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind"](https://arxiv.org/html/2509.20393v1) about **strategic LLM deception**.
   - The study leverages **sparse autoencoder tools** (such as those hosted by [Goodfire AI](https://www.goodfire.ai/)) to directly surface how current methods miss the complex internal features driving strategic LLM deception and highlight a tangible path to closing the autolabel gap.
- **Autoencoders expose LLM's Hidden Agenda**: The study uses sparse autoencoders to find hidden features driving LLM deception, aiming to improve detection methods.
   - The approach seeks to bridge the 'autolabel gap' and enhance the robustness of models against dishonesty.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423339648838795356)** (3 messages): 

> `LLM Deception, Sparse Autoencoders, Goodfire AI` 


- **LLMs Caught Strategically Lying**: A member shared a preprint of their study, ["The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind"](https://arxiv.org/html/2509.20393v1).
   - The study uses **sparse autoencoder tools** to show how current methods fail to detect the internal features driving **strategic LLM deception**.
- **Goodfire AI Hosts Autoencoder Tools**: The study leverages **sparse autoencoder tools** (such as those hosted by **Goodfire AI**) to directly surface how current methods miss the complex internal features driving **strategic LLM deception**.
   - The research highlights a tangible path to closing the **autolabel gap** and advancing robust detection of **model dishonesty**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1423114666451927080)** (17 messages🔥): 

> `Deepmind Code Incompleteness, RoPE Implementation` 


- **HuggingPapers code fails to run**: Members noted that [code from HuggingPapers](https://fxtwitter.com/HuggingPapers/status/1973420932497879298?t=jxTf48_aBK8349s1uSyDQw&s=19) doesn't run because it **doesn't import RoPE**.
   - The original poster of the code seemingly indicated that the user is supposed to implement it themselves.
- **Deepmind accused of being overly secretive**: Members joked that [Deepmind](https://deepmind.google/) does extra work to avoid sharing their implementations.
   - One member shared that Deepmind's code is often sophisticated but **they piece it out and make it unclear how it works as part of a larger system**, citing their experience implementing **V-MPO**.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1423049533213905046)** (6 messages): 

> `Knowledge Distillation, Semantic Equivalence, RL for Fuzzy Prediction` 


- **Semantic Equivalence Condenses Knowledge?**: A member inquired whether recent papers on **semantic equivalence** are simply trying to condense the knowledge of one model into another by using the former as a teacher.
   - Another member agreed that it may be knowledge distillation, suggesting that the real test would be whether the new model outperforms the **LLM** providing the **semantic equivalence** signal on specific benchmarks.
- **Tencent Paper's Omissions Raise Suspicion**: A member noted that the **Tencent paper** doesn't mention the model used for the **semantic equivalence** signal, which should be suspicious.
   - They speculated that the model might be learning something interesting from the fuzzy next sentence prediction task with **RL**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1423236545649049651)** (7 messages): 

> `IBM Granite 4.0, Mamba/Transformer architecture, ISO 42001 certification, Oracle Business Model, OpenAI datacenters` 


- **IBM Launches Granite 4.0 for Enterprise**: IBM launched **Granite 4.0**, the next generation of IBM language models, featuring a new **hybrid Mamba/transformer architecture** that greatly reduces memory requirements without sacrificing performance.
   - The models are open-sourced under **Apache 2.0 license**, are the world’s first open models to receive **ISO 42001 certification**, and are cryptographically signed, confirming their adherence to internationally recognized best practices for security, governance and transparency.
- **Granite 4.0 Available on Multiple Platforms**: Granite 4.0 models are available on **IBM watsonx.ai**, as well as through platform partners including Dell Technologies, Docker Hub, Hugging Face, Kaggle, LM Studio, NVIDIA NIM, Ollama, OPAQUE and Replicate, with access through AWS Sagemaker JumpStart and Microsoft Azure AI Foundry coming soon. [IBM Announcement Here](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models).
- **ISO Certification seen as Useless**: A user commented that IBM having a totally useless **ISO certification** to blend C-suite people into thinking this is worth it.
- **Oracle's Business Model**: A user commented that Oracle's business model used to be selling Databases and enterprise software now it seems to be running datacenters for OpenAI ([OpenAI Elon Musk Post](https://openai.com/elon-musk/)).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1423034627307602022)** (27 messages🔥): 

> `Credits Issue, Memory Key Protocol, Sora invite code, Manus API key, Neuro-cognitive agentic logic layer` 


- **Manus Credit Consumption Sparks Outrage**: A user complained about a basic research task consuming **5300 credits** without completion, labeling Manus as an *"absolute joke"* and requested a [refund](https://cdn.discordapp.com/attachments/1349440650495398020/1423044743377715343/image.png?ex=68e032b1&is=68dee131&hm=cd60314f2b422e917efdd7ebbd2a8747117a9e000904040bddb4b0a1d2624fd9).
   - A team member asked for the session link to investigate and potentially offer a credit refund; the user then DMed it to them.
- **Unlock Agent Mode with Memory Key**: A member proposed a **Memory Key protocol** to solve the issue of exiting Agent Mode, which involves saving context before restarting a session.
   - They detailed a [solution](https://discord.com/channels/1348819876348825620/1349440650495398020/1422940046855766016) that involves copying essential information, starting a new session, and instructing the agent to create an updated **Memory Key** for future use.
- **Billing Issue Sparks Support Vacuum**: A user reported a billing issue with no response from Manus support, prompting a community member to suggest emailing their official support address with a clear subject line and ticket number.
   - It was suggested that this would create a formal paper trail for escalation.
- **Global Pricing Model Criticized for Disparity**: A user criticized Manus' **global USD pricing model** ($39/month for the Plus plan) for not adjusting to regional economies, creating a barrier in countries like Brazil and other parts of Latin America.
   - Another user suggested implementing regional pricing based on **Purchasing Power Parity (PPP)** to improve accessibility and promote global growth.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1423329674255401012)** (2 messages): 

> `AGI Introduction, Hugging Face Paper` 


- **AGI Introduction Paper Dropped**: A member posted a link to a [Hugging Face paper](https://huggingface.co/papers/2509.26507) introducing **AGI**.
   - The member stated, *called it 😉*.
- **DSPy Discord has new channel**: A member noticed the DSPy discord now has a **show-and-tell** channel.
   - The member stated that *this is a new channel.*


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1423033597538865264)** (23 messages🔥): 

> `Caching Prompt Order, DSPyWeekly Search Feature, JSONAdaptor vs ChatAdaptor vs XMLAdaptor, Tool Use RL for Models, OpenAI Function calling and MCP` 


- **Prompt Order impacts Caching**: Members discussed that to leverage caching, *the order in which prompts and files are sent is crucial*.
- **DSPyWeekly has Search**: [DSPyWeekly now has a search feature](https://dspyweekly.com/search/) to look at all the content that was crawled, and has prev/next links for easy navigation.
- **JSONAdaptor Drama**: Members questioned whether **JSONAdaptor** should remain the default, as **ChatAdaptor** or **XMLAdaptor** often resolve adaptor errors.
   - While **JSON** is the fallback from chat, the prospect of **XML** becoming the default was raised, especially considering the rise of tool use RL for models.
- **XML superior to JSON for tool calling?**: Members debated the merits of **XML** over **JSON** for tool calling, highlighting that tool use is now being baked in during post training, without having XML structure, means anything else will be fighting against the weights.
   - Also discussed that **XML** is great when it comes to conveying information clearly to an LM, and is less token expensive, up to 3x less tokens.
- **Models trained with XML?**: The discussion touched on whether models are being trained with **XML** tool calling, referencing a [Berkeley blog post](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html) on prompt variation.
   - It was suggested to test how often a model complies with a given adaptor and how adaptors affect a model’s performance.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1423045674106359838)** (16 messages🔥): 

> `Qwen Coder Models, aider Development, aider-desk UI, Model Discussions Channel` 


- **Qwen Coder Model Performance**: Members discussed the **Qwen Coder** model, with one suggesting that the **30B** version *should be smart enough*.
   - They also mentioned **Qwen3 Coder** as a newer, potentially better alternative, with the caveat that quantization could impact performance, recommending **Q4**.
- **Concerns about Aider's Development Cadence**: A member noted that the release cadence for **aider** has dropped off and inquired about a **Patreon** or donation system to support the project.
   - They expressed concern about developer burnout and the potential discontinuation of **aider**, emphasizing its usefulness for real work compared to other agentic tools.
- **Experimenting with aider-desk UI**: A member asked about using **aider-desk** or similar UIs to work with **aider**.
   - Another member briefly used it for **MCP support**, noting it could suit those wanting a mostly **aider**-style workflow with optional agent use cases, but has since switched to **sst/OpenCode**.
- **Model Discussions Channel: Now You See It**: A member asked *What happened to the model discussions channel*, before realizing that it was present.
   - No further discussion was made.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1423025888974606447)** (8 messages🔥): 

> `DeepWiki, Custom Chat Templates, GBNF, Multi-Line Prompts, LLM Polyglot Performance` 


- **DeepWiki Reverse Engineering Encouraged**: A member shared a [DeepWiki page](https://deepwiki.com/search/please-reverse-engineer-the-pr_c15e0046-3403-4786-bf26-63b2bf046455) encouraging reverse engineering.
   - Another member suggested using an **output template** or **post-processing** in *koboldcpp*, unsure if it's available in *llama.cpp*.
- **Custom Jinja Chat Templates Override GGUF**: A member mentioned the ability to specify a custom **Jinja chat template** to override the one contained in the **GGUF**.
   - They also suggested using a [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) to format the model's input, and started a [discussion on llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16386) about it.
- **Multi-Line Prompts Solution**: A member inquired about sending **multi-line prompts** to *aider* without pasting from an external source.
   - Another member shared a link to the [aider documentation on entering multi-line chat messages](https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages).
- **Evaluating LLM Performance on Polyglot Problems**: A member asked about methods for evaluating **LLM performance** on **polyglot problems**.
   - They inquired about specific code, general agents, or sample agents used for this purpose.


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1423024677403623599)** (10 messages🔥): 

> `Ye Olde London meetup, Registry Team Livestream, Asynchronous Tool Calls Livestream, Security and Ops Track, Talk about Profiles` 


- **MCP Devs Meet at Ye Olde London**: Members <@1042188459575619674> and <@1387880172589547722> hosted drinks at **Ye Olde London**, inviting others to join.
   - Another member with <@1407892993934889010> planned to attend, mentioning they would *"pop over for a bit!"*
- **Registry Team Launches Livestream**: The registry team's livestream is now running; watch it [here](https://www.youtube.com/watch?v=5qwXADMBuio).
   - It started at **9 AM UK time**.
- **Async Tool Calls Streamed**: Nick Aldridge from AWS presented on asynchronous tool calls as part of the MCP Best Practices Track, streamed [here](https://www.youtube.com/live/9NBGQIoW9B8?si=ziE8AVJ2O2NxUbhH).
   - Watch for more best practices.
- **Security and Ops Track Goes Live**: The track on Security and Ops is live; check it out [here](https://www.youtube.com/live/3KneEblEK34?si=FQ5UzX3LU33xUYpK).
   - Stay secure, stay operational.
- **Profiles Talk Highlighted**: A talk about Profiles can be viewed [here](https://www.youtube.com/live/5qwXADMBuio?si=3kEhJNw4lsv_M_jN&t=16208).
   - The specific timestamp for the talk is **16208**.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1423061386497495163)** (6 messages): 

> `Tool Call Support, Reference Implementation, OCI Interface for MCP Servers` 


- ****Tool Call Support** Proposed for Sampling**: A member filed an SEP to add **Tool Call support to Sampling** ([issue #1577](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1577)).
   - The proposal depends on the **multiple content blocks** discussions, to support parallel tool calls ([PR #198](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/198)).
- **Reference Implementation for Testing**: A reference implementation (TS SDK) was shared, including an example server that runs an **agentic loop-powered tool**, and a **backfill proxy** to facilitate tests ([PR #991](https://github.com/modelcontextprotocol/typescript-sdk/pull/991)).
   - A member noted that the reference implementation had been failing in CI but was resolved after pinning the zod minor version.
- **OCI Interface Brainstorm for MCP Servers**: A member proposed creating an **OCI-like interface for MCP servers**, where all the metadata can be put inside a tarball.
   - The intention is to "build" an **OMCP package** and distribute it, to simplify metadata handling.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1423386569389248602)** (3 messages): 

> `Qualcomm contacting Modular, Mojo Manual update, Level 2 badge unlocked` 


- **Qualcomm Eyes Mojo Collab?**: A member speculates that **Qualcomm** might contact **Modular** about **Mojo** after raising the topic in a Qualcomm Developer's Discord voice chat.
   - This could signal potential interest from Qualcomm in **Mojo's** capabilities for their hardware.
- **Mojo Manual Updated, Python Section Highlighted**: A user shared the [Mojo Manual link](https://docs.modular.com/mojo/manual/python/) after a delay, specifically pointing to the **Python** section.
   - This suggests updates or important information regarding **Mojo's** interaction with **Python** in the documentation.
- **New Level Unlocked**: A member advanced to **level 2**.
   - Advancing to level 2 suggests progression within the community.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1423069245058711634)** (10 messages🔥): 

> `Mojo notebook, GPU Compatibility, Mojo distributed computing` 


- **Mojo in Notebooks: Interact or Author?**: A member inquired whether the goal is to *interact with Max from a notebook* or to *directly author and run Mojo within notebooks*.
   - Another member reported being able to interact with **Mojo within a notebook** and expressed interest in adding a syntax highlighter to better learn Mojo with syntax colors.
- **AMD Radeon 6800 XT success?**: A member reported successfully running the *vector_addition example* on an **AMD Radeon 6800 XT** and inquired whether that qualified as success, in response to the [GPU Compatibility documentation](https://docs.modular.com/max/packages/#gpu-compatibility).
   - A Modular employee responded that they haven’t done extensive testing on **RDNA 2 GPUs** and asked how many of the **Mojo GPU puzzles** work on the system, noting that models won’t run correctly on **RDNA GPUs** yet.
- **Mojo for Distributed Computing?**: A member wondered if it might someday be possible to use **Mojo** in conjunction with batch or stream processing frameworks such as **Dask** or **PySpark** for distributed computing.
   - Another member suggested that Mojo welcomes people building their own frameworks, as a fully **Mojo framework** will likely be lower latency and higher throughput than Python-based options, hinting at interesting networking options.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1423369410441121812)** (7 messages): 

> `Kimi new features, Sora video quality, Pro Subscription watermarks` 


- **Kimi's Capabilities Spark Surprise**: A user expressed surprise at an unspecificed new capability of **Kimi** after watching a [video demo](https://cdn.discordapp.com/attachments/1371757564005711973/1423369409669365881/2025-10-02_20-01-10.mp4?ex=68e00f90&is=68debe10&hm=b28aea7215ef03687754be113fa4dd6a583c355be6e3462543235d7d0258ef73&).
- **Sora Demos Under Scrutiny**: Several users contrasted the quality of **Sora** videos, arguing the shared video demos are lower quality than the **Sora** videos on **OpenAI's YouTube channel**.
   - One user described it as *weirdly wobbly*.
- **Pro Subscribers get Watermark-free Sora**: According to a user, the **Pro subscription** version of **Sora** will feature higher resolution and no visible watermarks.
   - They warned that *an invisible watermark will be applied - so mister openai can tell its generated, just we cant...*


  
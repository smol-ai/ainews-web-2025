---
id: MjAyNS0w
title: Cognition's $10b Series C; Smol AI updates
date: '2025-09-08T05:44:39.731046Z'
description: >-
  **Cognition** raised **$400M** at a **$10.2B** valuation to advance AI coding
  agents, with **swyx** joining the company. **Vercel** launched an OSS coding
  platform using a tuned **GPT-5** agent loop. The **Kimi K2-0905** model
  achieved top coding eval scores and improved agentic capabilities with doubled
  context length. **Alibaba** released **Qwen3-ASR**, a multilingual
  transcription model with robust noise handling. **Meta** introduced Set Block
  Decoding for 3-5× faster decoding without architectural changes. Innovations
  in KV cache compression and quantization were highlighted, including
  **AutoRound** in SGLang and **QuTLASS v0.1.0** for Blackwell GPUs. Algorithmic
  benchmarking tools like **AlgoPerf v0.6** were updated for efficiency.
companies:
  - cognition
  - vercel
  - meta-ai-fair
  - alibaba
  - groq
  - huggingface
models:
  - kimi-k2-0905
  - qwen3-asr
  - gpt-5
topics:
  - coding-agents
  - agent-development
  - open-source
  - model-evaluation
  - multilingual-models
  - inference-optimization
  - kv-cache-compression
  - quantization
  - algorithmic-benchmarking
  - context-length
  - model-performance
people:
  - swyx
---


**A special update for Smol AI readers.**

> AI News for 9/5/2025-9/8/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (187 channels, and 12661 messages) for you. Estimated reading time saved (at 200wpm): 1069 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

As [leaked in July](https://news.smol.ai/issues/25-07-24-cogsurf-cursor), the $10b round for Cognition was [announced today](https://x.com/cognition/status/1965086655821525280). What we also announced was that I ([swyx) will also be joining](https://x.com/swyx/status/1965183110016098617) Cognition in some yet to be defined capacity, while [AI Engineer](https://apply.ai.engineer/) and Latent Space remain independent. AINews will keep going as a personal project, with some conversations ongoing around its stable future.

---

# **AI Twitter Recap**

**Coding Agents and Tooling Momentum**

- **Cognition raises $400M to scale Devin**: Cognition announced a $400M round at a $10.2B post-money valuation to “advance the frontier of AI coding agents,” led by Founders Fund with Lux, 8VC, Neo and others participating. The team highlighted customer expansion and the Windsurf team joining, and is hiring across product, infra, and post‑training ([announcement 1](https://twitter.com/cognition/status/1965086655821525280), [2](https://twitter.com/cognition/status/1965086662612177299), [team note](https://twitter.com/cognition/status/1965086661253185645), [plans clip](https://twitter.com/cognition/status/1965185627357683776)). Commentary: @swyx is joining Cognition, laying out why he’s “buying” the agent-lab thesis and how positioning across sync/async workflows matters for dominance in the “Decade of Agents” ([thread](https://twitter.com/swyx/status/1965183110016098617)).
- **Agent dev stacks getting simpler and more capable**:
    - Vercel shipped an OSS “vibe coding platform” built on the Vercel AI SDK, Gateway, Sandbox, and a tuned GPT‑5 agent loop (tool use: file IO, commands, package install, autofix) with a one‑shot demo coding a multiplayer Pong game in Go ([demo](https://twitter.com/rauchg/status/1964857952722133231)).
    - Claude Code’s loop is intentionally minimal: a single master loop + async buffer, direct tools, and TODO-based planning; simplicity beats swarm orchestration for debuggability and reliability ([analysis](https://twitter.com/imjaredz/status/1965083721713041564)).
    - Coding evals: Kimi K2‑0905 on Groq hit 94% and ranked 7th on Roo Code, becoming the first open-weight model to break 90+ while also being the fastest/cheapest in the top 10 ([leaderboard](https://twitter.com/roo_code/status/1965098976677658630)). Tim Dettmers reports the practical frontier for coding assistants feels increasingly open-weight: GLM‑4.5 is “$3/month” and ~Sonnet quality; Kimi K2.1 Turbo ~3× faster and ~7× cheaper vs Opus 4.1, with GPT‑5 excelling mainly on complex spec work ([take](https://twitter.com/Tim_Dettmers/status/1965021602267217972)).

**Model and Inference Advances**

- **Kimi K2 0905 and Qwen3-ASR**:
    - Kimi K2 0905 (1T params, architecture unchanged) boosts agentic capabilities: Terminal‑Bench Hard from 14→23% and Tau2‑Bench Telecom 61→73%; context doubled from 128k→256k. Intelligence +2 on Artificial Analysis’ AAII; now serving on Kimi’s site ([summary](https://twitter.com/ArtificialAnlys/status/1965010554499788841), [live note](https://twitter.com/crystalsssup/status/1965017719058960732)).
    - Alibaba’s Qwen3‑ASR released a single model for multilingual transcription (EN/CN + 9 languages), autodetect, robust to BGM/noise/rap, with <8% WER and custom contextual biasing. Demos on ModelScope/HF; API available ([launch](https://twitter.com/Alibaba_Qwen/status/1965068737297707261)).
- **Faster decoding and lighter KV**:
    - Meta’s Set Block Decoding (SBD) enables 3–5× decoding speedups on existing LMs without architectural changes, matching NTP performance and preserving exact KV cache—parallel generation via masked/discrete diffusion formulation ([overview](https://twitter.com/HuggingPapers/status/1965084731839513059), [details](https://twitter.com/itai_gat/status/1965112129499046230)).
    - KV cache and quant innovation: AutoRound is now in SGLang ([PR](https://twitter.com/HaihaoShen/status/1964926924880523701)), Turing Post surveyed KV compression (quantization, low‑rank, Slim Attention, XQuant) with tradeoffs ([thread](https://twitter.com/TheTuringPost/status/1964971207188791464)), and QuTLASS v0.1.0 brings 4‑bit NVFP4 microscaling and fast transforms to Blackwell GPUs ([release](https://twitter.com/DAlistarh/status/1965157635617087885)). AlgoPerf v0.6 adds a rolling leaderboard, JAX jit, and lower compute costs for algorithmic benchmarking ([update](https://twitter.com/algoperf/status/1965044626626342993)); ZeroGPU AOT compilation internals for PyTorch were documented by HF ([blog](https://twitter.com/charlesbben/status/1965046090945954104)).

**Multimodal Generation, Video, and “Vibe Coding”**

- **Veo 3 goes GA and cheaper**: Google’s Veo 3 and Veo 3 Fast are now GA in the Gemini API with ~50% price cuts ($0.40/s and $0.15/s), 1080p output, and 9:16 vertical video support—positioned for scaled production ([dev blog](https://twitter.com/googleaidevs/status/1965160822260318702), [pricing breakdown](https://twitter.com/_philschmid/status/1965161626761326983), [PM note](https://twitter.com/OfficialLoganK/status/1965193765146296467)).
- **Community workflows and tooling**:
    - “Nano Banana” (Gemini 2.5 Flash Image Preview) catalyzed a weekend of “vibe‑coded” projects—now open-sourced for remix in Google AI Studio; teams report 1‑click reuse and playful gotchas (e.g., always rendering clocks at 10:10) ([open-source pack](https://twitter.com/arrakis_ai/status/1965001417716072877), [quirk](https://twitter.com/fabianstelzer/status/1965001753059057925)).
    - Qwen’s “paper → website” flow turns a research paper into a deployable site in minutes ([demo](https://twitter.com/Alibaba_Qwen/status/1964870508421480524)). Lmarena added multi‑turn image editing evals so the community can compare iterative refinement across models (incl. “nano banana”) ([feature](https://twitter.com/lmarena_ai/status/1965150440401809436)). For doc RAG UX, ColQwen2 + Weaviate powers token‑wise similarity maps for visual PDF search and patch highlighting ([build](https://twitter.com/helloiamleonie/status/1964997028875743637)).

**Agents, Post-Training RL, and Evaluation Practice**

- **Towards iterated self‑improvement**: FAIR’s Exploratory Iteration (ExIt) trains LLMs for inference‑time self‑improvement via an automatic curriculum that bootstraps from the model’s own prior responses, prioritizing partial histories with high return variance in GRPO groups. ExIt outperforms GRPO on contest math, BFCLv3 multi‑turn tasks, and MLE‑bench (+22%) while training only single‑step improvements ([thread](https://twitter.com/MinqiJiang/status/1965055909605916892)).
- **Online vs offline RL and evals**:
    - Evidence continues to show a performance gap favoring online RL (PPO/GRPO) over offline methods like DPO at scale, though semi‑online iterations (on‑policy sampling + negative gradients) narrow the gap; data quality still dominates algorithm choice ([summary](https://twitter.com/cwolferesearch/status/1965088925510520853)).
    - Why many “agents” underdeliver: decision‑making has near‑zero error tolerance and sparse data vs generative tasks; most failures are coarse task scoping and unstructured environments rather than LLM shortcomings ([debate recap](https://twitter.com/ZhihuFrontier/status/1964928650081698167)).
    - RAG evals moving from “dead” unit tests to “living” loops: RAGGY (open‑source REPL) enables what‑if iteration for RAG, and there’s a strong push to integrate pre‑prod tests with production observability and human review rather than treating them as separate silos ([RAGGY](https://twitter.com/HamelHusain/status/1965052554997600449), [evals take](https://twitter.com/bnicholehopkins/status/1965130607790264452)). Also see practical “Agentic RAG” architectures leveraging tool use and multi‑step reasoning ([guide](https://twitter.com/omarsar0/status/1965115682322042954)).

**Robotics and Embodied AI**

- **Multi‑robot planning via RL**: Google DeepMind’s RoboBallet (with Intrinsic and UCL) choreographs up to 8 robot arms for collision‑free task and motion planning, outperforming traditional methods by ~25%, and generalizing to new workflows in seconds via RL‑learned coordination principles ([announcement](https://twitter.com/GoogleDeepMind/status/1965040645103407572), [more](https://twitter.com/GoogleDeepMind/status/1965040648400351337)).
- **Open hardware stacks and dexterous manipulation**: Pollen Robotics outfitted Reachy 2 with dual open‑source “Amazing Hand” grippers for fine manipulation; native integration coming ([demo](https://twitter.com/pollenrobotics/status/1964987735829266871)). X Square announced WALL‑OSS (open base model) and the Quanta X2 robot with auto‑mop and dexterous hand; Alibaba Cloud led a $140M A+ round (>$280M raised in <2 years) ([summary](https://twitter.com/ZhihuFrontier/status/1964968113990164810)). OpenPI’s pi‑05 is now in openpi with PyTorch support ([release](https://twitter.com/svlevine/status/1965161524722630734)).

**Benchmarks, Leaderboards, and Enterprise**

- **Text leaderboards move**: lmarena added two new entries into its Top 10 Text leaderboard: Qwen3‑max‑preview (#6, proprietary) and Kimi‑K2‑0905‑preview (#8, modified MIT), putting Kimi in contention for top open‑weight alongside Qwen and DeepSeek variants ([update](https://twitter.com/lmarena_ai/status/1965115050273976703), [model link](https://twitter.com/lmarena_ai/status/1965124408097517853)). Artificial Analysis' K2‑0905 measurements mirror improved agentic performance ([details](https://twitter.com/ArtificialAnlys/status/1965010554499788841)).
- **Gov and enterprise**:
    - Perplexity launched “Perplexity for Government”: secure by default, zero data usage, premium model access, and no enterprise contracts; also brought Perplexity Finance to iOS/Android ([launch](https://twitter.com/perplexity_ai/status/1965030156415980009), [follow‑up](https://twitter.com/AravSrinivas/status/1965032305053065590), [finance mobile](https://twitter.com/AravSrinivas/status/1965100159488196757)).
    - Anthropic endorsed California SB 53 (Sen. Scott Wiener), a transparency‑focused state framework for governing frontier AI in lieu of a federal standard ([statement](https://twitter.com/AnthropicAI/status/1965027311717388673), [context](https://twitter.com/jackclarkSF/status/1965048896784367847)).

Top tweets (by engagement)

- Cognition raises $400M at $10.2B to scale AI coding agents ([announcement](https://twitter.com/cognition/status/1965086655821525280))
- Vercel’s OSS vibe coding platform with a tuned GPT‑5 loop one‑shots a multiplayer Pong game in Go ([demo](https://twitter.com/rauchg/status/1964857952722133231))
- Qwen3‑ASR: one model for multilingual ASR with <8% WER, robust to noise/BGM, with context injection ([launch](https://twitter.com/Alibaba_Qwen/status/1965068737297707261))
- Google AI Mode expands to Hindi, Indonesian, Japanese, Korean, and Brazilian Portuguese ([Sundar Pichai](https://twitter.com/sundarpichai/status/1965115123330388467))
- Veo 3 GA with ~50% price cuts, 1080p, and vertical video in the Gemini API ([dev update](https://twitter.com/googleaidevs/status/1965160822260318702))

---

# **AI Reddit Recap**

## **/r/LocalLlama + /r/localLLM Recap**

**1. Open-source LLM Launches: K2 Think and TildeOpen 30B Multilingual**

- [**UAE Preparing to Launch K2 Think, "the world’s most advanced open-source reasoning model"**](https://www.wam.ae/en/article/bll7llv-recognition-sheikh-khalifa%E2%80%99s-contribution) ([Score: 217, Comments: 70](https://www.reddit.com/r/LocalLLaMA/comments/1nbo33p/uae_preparing_to_launch_k2_think_the_worlds_most/)): **MBZUAI ([site](https://mbzuai.ac.ae/)) and G42 ([site](https://g42.ai/)) teased an imminent release of “K2 Think,” described as an open‑source reasoning model that delivers “frontier‑class” performance in a compact footprint, allegedly matching or surpassing models ~**`10×` **larger. No concrete specs (parameter count, architecture, context window, tokenizer, training data, training compute, or evaluation suite) were disclosed; the announcement states release is “in the coming week.” The effort is stated as unrelated to Moonshot/Kimi’s “K2,” and follows a prior 2024 “K2” 65B model reportedly close to a reproduction of Meta’s [Llama 2 70B](https://ai.meta.com/llama/).** Commenters note naming confusion with Moonshot/Kimi’s K2 and express skepticism pending benchmarks (e.g., “I’ll believe it when I see it”), while highlighting the absence of basic specs like parameter size.
    - Model identity/naming clarification: commenters note this "K2 Think" is unrelated to Moonshot/Kimi’s "K2," and that the same UAE group previously released a 65B "K2" (2024) that was effectively a reproduction of **Meta’s Llama 2 70B** (see Llama 2 paper: https://arxiv.org/abs/2307.09288). Reusing the "K2" moniker risks conflating distinct projects and overstating novelty relative to a Llama‑derivative baseline.
    - Missing technical specs: no parameter count was disclosed ("No mention of the parameter size?"), making the "world’s most advanced open‑source reasoning model" claim impossible to assess. Technical readers expect details such as parameter count and architecture (dense vs MoE), context length, training compute/dataset, and reasoning benchmarks (e.g., GSM8K, MATH, AIME, BBH, ARC‑C) with transparent evaluation settings; absent these, skepticism is warranted.
- [**Tilde AI Releases TildeOpen LLM: An Open-Source Large Language Model with Over 30 Billion Parameters and Support Most European Languages**](https://huggingface.co/TildeAI/TildeOpen-30b) ([Score: 173, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1nbi95c/tilde_ai_releases_tildeopen_llm_an_opensource/)): [**Tilde.ai](http://tilde.ai/) released [TildeOpen‑30B](https://huggingface.co/TildeAI/TildeOpen-30b), an open-source** `~30B`**parameter dense decoder-only transformer targeting underrepresented Nordic/Eastern European languages, trained on the LUMI supercomputer (768 AMD MI250X GPUs) for** `450k` **updates with a global batch of** `4,718,592` **tokens (≈**`2.12T` **tokens, constant LR + cooldown) using a three‑phase sampling curriculum (uniform → natural → uniform) and an equitable tokenizer to balance low‑resource languages. Architecture: 60 layers,** `d_model=6144`**,** `n_heads=48` **with** `8` **KV heads (GQA),** `d_ff=21,504`**, RoPE, SwiGLU, RMSNorm,** `context=8192`**; licensed CC‑BY‑4.0, not instruction‑tuned/safety‑aligned, with [GGUF quantizations](https://huggingface.co/mradermacher/TildeOpen-30b-GGUF) available. They report strong character‑level perplexity on WMT24++ across focus languages, often competitive vs. EuroLLM/ALIA and Google’s Gemma 2, and plan a specialized translation model built atop this foundation.** Comments note the lack of a demo/playground, question whether total tokens are `~4.1T` (calc from provided batch and steps suggests `~2.12T`), and critique that only perplexity is reported against a narrow set of baselines—arguing perplexity depends heavily on data mix and may not predict downstream quality (with some expecting multilingual models like Gwen3 to outperform on many languages).
    - Training math clarification: with `450,000` updates and a global batch of `4,718,592` tokens/step, the implied token count is `≈ 2,123,366,400,000` (~2.12T), not 4.1T. The phrasing “constant learning rate followed by a cooldown phase across 2 trillion tokens” likely describes the LR schedule over ~2T tokens; any cooldown beyond that would add only modestly to the total—not doubling it.
    - Evaluation concerns: the [HF card](https://huggingface.co/TildeAI/TildeOpen-30b) reports only perplexity vs Gemma 2, EuroLLM, and ALIA, which is weak evidence because perplexity heavily depends on the training data distribution and correlates poorly with downstream task quality. For a multilingual base model, readers expect standardized multilingual benchmarks (e.g., FLORES-200 translation, XQuAD/TyDiQA, MGSM, MMLU) and broader baselines (e.g., Qwen/Qwen2.5, Aya, mT5, XLM-R); without these, comparisons to models like Gwen3 (trained on 119 languages) are hard to substantiate.
    - Model type: it’s a base (non-instruction-tuned) model, so it won’t reliably “chat” without additional alignment (SFT/DPO/RLHF); expect raw next-token generation rather than instruction following. The absence of a demo chat UI is consistent with this; it can produce fluent text in supported languages but needs instruction tuning or a chat adapter to behave like a conversational assistant.

**2. Local/Offline LLM Use on Personal Hardware (Dual RTX 6000 + M3 Mac)**

- [**Finishing touches on dual RTX 6000 build**](https://i.redd.it/sez83piasvnf1.jpeg) ([Score: 280, Comments: 129](https://www.reddit.com/r/LocalLLaMA/comments/1nbfy60/finishing_touches_on_dual_rtx_6000_build/)): **OP showcases a workstation rig with dual NVIDIA RTX 6000 cards (claimed ~192 GB aggregate VRAM) and 128 GB system RAM to run local LLMs (e.g., Qwen 2 35B in 4‑bit). Main technical concern is power on typical 120V/15A home circuits (~1.8 kW max, ~1.44 kW continuous); commenters suggest capping each GPU to a ~300 W power limit (RTX 6000 ≈300 W TGP) to avoid trips with only ~10% performance loss, or upgrading the circuit.** Commenters estimate roughly $16k in GPUs, ask about the CPU choice, and critique the flashy ‘gamer’ case aesthetics for such a pro-grade build.
    - Power limiting the dual RTX 6000s to `~300 W` is suggested to keep thermals/noise in check with only about a `~10%` performance hit. This aligns with the RTX 6000 Ada’s `300 W` TBP spec ([NVIDIA](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)), and mirrors common Ada undervolt/underpower tuning where performance scales sublinearly above ~300 W, making it a pragmatic tradeoff in dense workstation builds.
    - A commenter flags the absence of “full `192 GB`” system RAM as surprising, arguing more RAM improves cache behavior for data-heavy workflows. For dual `48 GB` VRAM GPUs, ample system memory can meaningfully enlarge the OS page/file cache and reduce I/O stalls when staging large datasets, models, or textures, which can be a bigger limiter than raw GPU TFLOPs in some pipelines.
- [**Apocalyptic scenario: If you could download only one LLM before the internet goes down, which one would it be?**](https://www.reddit.com/r/LocalLLaMA/comments/1nbgosx/apocalyptic_scenario_if_you_could_download_only/) ([Score: 249, Comments: 230](https://www.reddit.com/r/LocalLLaMA/comments/1nbgosx/apocalyptic_scenario_if_you_could_download_only/)): **OP asks which single local LLM to download for fully offline use on a Mac Studio (Apple M3,** `512 GB` **RAM). Top picks: (1) GLM 4.5 (Air variant) from ZhipuAI for robust general-purpose code/scripting and tolerance to modest compute while being RAM-friendly; see GLM family by THUDM/ZhipuAI [models](https://huggingface.co/THUDM). (2) Qwen3 30B “thinking” paired with an offline Wikipedia dump in a vector database (RAG) for broad knowledge coverage; see Qwen [models](https://huggingface.co/Qwen) and using an ANN store like [FAISS](https://github.com/facebookresearch/faiss) with [Wikipedia dumps](https://dumps.wikimedia.org/). A cautionary note is to avoid “GPT-OSS” due to perceived safety issues.** Commenters lean toward GLM for pragmatic code/ops reliability on CPU-only field hardware vs. Qwen3 30B + RAG for knowledge breadth; there’s disagreement over model safety, with a warning against “GPT-OSS.”
    - **GLM 4.5 (Air)** is highlighted for strong offline utility on CPU-only laptops due to a modest memory footprint and stability for scripting/system tasks. While it won’t excel at long-form writing, users report it reliably generates bash scripts and helps with troubleshooting in field conditions (no GPU) as long as there’s sufficient RAM.
    - Pairing **Qwen3 30B Thinking** with a locally stored Wikipedia in a vector database is suggested to maximize offline breadth: the model handles reasoning while RAG supplies factual recall. This setup requires precomputing embeddings and indexing Wikipedia, trading storage/CPU for improved retrieval quality and internet independence.
    - **Qwen 30B A3B** + downloaded Wikipedia RAG is preferred under tight energy budgets, emphasizing “as few active parameters as possible” to minimize power draw. The approach favors compute-efficiency (e.g., sparse or reduced active parameters) over larger dense models, aiming for longer runtimes on limited power without sacrificing core reasoning ability.

## **Less Technical AI Subreddit Recap**

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

**1. AlterEgo wearable, Gemini 'Upload Any File', and Qwen Edit LORA launches**

- [**Introducing Alterego: the world’s first near-telepathic wearable that enables silent communication at the speed of thought**](https://v.redd.it/5a9hu9antznf1) ([Score: 334, Comments: 143](https://www.reddit.com/r/singularity/comments/1nbxgri/introducing_alterego_the_worlds_first/)): **The post links to an announcement of “Alterego,” pitched as a “near-telepathic” wearable enabling silent communication “at the speed of thought” ([tweet](https://x.com/alterego_io/status/1965113585299849535)), but provides no technical specs, modality, or benchmarks; the associated Reddit video was inaccessible (HTTP 403), so no demo could be verified. The claim overlaps heavily with prior non‑invasive silent‑speech interfaces using sEMG (e.g., MIT Media Lab’s AlterEgo reporting ~**`92%` **accuracy on a** `20-word` **vocab [MIT News](https://news.mit.edu/2018/altereego-device-transcribes-words-people-say-in-their-heads-0404)) and wrist‑EMG decoders (e.g., Meta/CTRL‑Labs [overview](https://tech.facebook.com/reality-labs/2021/03/ar-wristband-haptic-emg/)); true “thought” decoding at conversational rates remains tied to constrained paradigms or invasive BCIs, not commodity wearables.** Top comments are skeptical about legitimacy, suggesting it reads like satire or a “VC money trap,” noting the lack of technical detail/demos and conflation of subvocal EMG decoding with literal internal‑monologue “telepathy.”
    - Several commenters flag the absence of any technical description on the site and ask for concrete specs: what sensing modality is used (e.g., surface EMG along the jaw/throat vs EEG/ultrasound), sampling rates, on-device vs phone/cloud inference, model architecture/size, calibration time, battery life/power draw, and objective metrics like word-error rate (WER), end-to-end latency, and vocabulary constraints. They reference prior art like **MIT Media Lab’s AlterEgo** (sEMG, limited-vocabulary, ~`92%` accuracy on ~`20` words) to suggest a baseline for comparison and request a whitepaper or dataset/code release for validation ([MIT project](https://www.media.mit.edu/projects/alterego/overview/)).
    - Skeptics challenge the “near-telepathic” and “speed of thought” claims, noting that non-invasive silent-speech systems typically show meaningful latency (`~100–300 ms+`) and degrade sharply on open-vocabulary tasks; robust performance generally requires constrained lexicons or user-specific calibration. They call for rigorous benchmarks: preregistered, randomized live demos with blind prompts, reporting WER, characters-per-minute, latency distributions, out-of-vocab handling, and cross-user generalization, plus comparisons against prior sEMG/EEG and ultrasound-lip models.
    - Concerns about a potential “VC trap” revolve around the lack of peer-reviewed results or third-party evaluations; commenters want independent replication and stress tests. Suggested proof points include: ablation studies (sensor count/placement), robustness to motion/sweat/noise, multi-speaker adaptation, and failure-mode analyses; without these, the claims are viewed as marketing rather than engineering evidence.
- [**We can upload any file to gemini app now !! Even audio!**](https://i.redd.it/3ap81kmdeynf1.png) ([Score: 256, Comments: 24](https://www.reddit.com/r/Bard/comments/1nbpp24/we_can_upload_any_file_to_gemini_app_now_even/)): **The post announces that the Gemini mobile app now supports uploading arbitrary files—including audio—directly in the app. This brings the app UI to parity with existing capabilities in the Gemini API (supported for ~**`2` **years) and Google AI Studio (since Gemini** `2.5 Pro`**), indicating this is an interface rollout rather than a new model capability; see confirmation via [Josh Woodward](https://x.com/joshwoodward/status/1965057589718499756).** Commenters note the delay was due to the app’s UI, not model limits, expressing surprise it took this long despite longstanding API/AI Studio support.
    - Commenters note the capability isn’t new at the model/API level: the Gemini API has accepted arbitrary file uploads for ~2 years, and **AI Studio** has supported these file types since the release of **Gemini 2.5 Pro**. The delay was due to the mobile/consumer app UI, not a model limitation. External confirmation/announcement reference: https://x.com/joshwoodward/status/1965057589718499756?t=Axnh1CAMsFECFp4eMnRbBg&s=19.
    - Multiple users report app reliability issues unrelated to the model: chats often fail to load with a “failed to load chat” error, and sessions can be incorrectly locked with messages claiming a custom Gem was deleted. These issues force frequent app restarts during active use, indicating client-side session/state management or caching bugs that degrade usability despite the new file-upload feature.
    - Native audio file uploads now enable on-the-fly transcription directly in the app, aligning with capabilities that were already available via the API. This reduces the need for external tooling for quick audio-to-text workflows and brings the consumer app in parity with developer-facing surfaces for audio handling.
- [**Clothes Try On (Clothing Transfer) - Qwen Edit Loraa**](https://www.reddit.com/gallery/1nbzh2d) ([Score: 216, Comments: 29](https://www.reddit.com/r/StableDiffusion/comments/1nbzh2d/clothes_try_on_clothing_transfer_qwen_edit_loraa/)): **Release of an outfit try-on LoRA for Qwen Image Edit enabling clothing transfer while preserving subject identity and matching diverse art styles/body types. Resources: [Patreon blog](https://www.patreon.com/posts/138311408), [CivitAI model](https://civitai.com/models/1940532?modelVersionId=2196278), and a companion [Clothing Extractor LoRA](https://civitai.com/models/1940557?modelVersionId=2196307). Known limitation: a perceptible quality drop (likely from the Qwen Edit pipeline and/or lower-res training data); author suggests upscaling with SeedVR2 and plans a** `higher-resolution` **retrain for the next version; commercial readiness is TBD.** Commenters highlight the value of building on an open-license, non‑distilled base (contrasting with Flux Kontext), and debate industry impact—claiming shoots that cost `~$20k` could be reduced to `~$200` with this workflow. Some users requested clearer links for immediate testing.
    - Commenters highlight that using an **open-license, non-distilled** base (vs. closed systems like **Flux Kontext**) enables community LoRA fine-tuning and redistribution without black-box constraints. Non-distilled weights typically preserve fine-grained editability and avoid distillation artifacts, which is critical for clothing transfer fidelity (e.g., fabric textures, seams). This openness improves reproducibility and makes it easier to iterate on edit adapters like LoRA ([LoRA paper](https://arxiv.org/abs/2106.09685)).
    - A working model notes dramatic production cost/time implications: fashion shoots that ran ~`$20,000` could drop to ~`$200` with a single operator in a few hours—roughly a `100x` cost reduction and major throughput gains. This effectively replaces a multi-person pipeline (photographer, lighting, stylist, MUA/hair) with an inference/edit workflow, shifting spend from logistics to compute.
    - Quality and data requirements: users report improved texture fidelity and style consistency over the previous demo (e.g., *"textures are on point"*), but also ask if garments must be on a white background—implying reliance on simple segmentation/matting for garment isolation. If so, high-contrast or white backgrounds simplify masking, whereas varied backgrounds may require robust segmentation/matting to maintain edge detail and avoid color spill (e.g., [SAM](https://segment-anything.com/)).
- [**OpenAI helping to make an AI generated feature length animated movie that will be released in 2026**](https://i.redd.it/50fm7mb3cvnf1.jpeg) ([Score: 591, Comments: 165](https://www.reddit.com/r/singularity/comments/1nbebg0/openai_helping_to_make_an_ai_generated_feature/)): **Post shares a claim that OpenAI is collaborating on an AI‑generated, feature‑length animated film slated for a 2026 release. The technical significance is the push toward an end‑to‑end generative pipeline for animation (AI‑assisted pre‑viz/storyboards, scene/shot generation, voice/SFX, and post), and whether such a pipeline can reach theatrical quality at scale and cost compared to traditional CG workflows. The image appears to be a headline/announcement; comments reference a** `~$30M` **budget, implying heavy spend on compute, model development, data/licensing, and human polish rather than fully “one‑click” generation. [Image link](https://i.redd.it/50fm7mb3cvnf1.jpeg).** Commenters question the reported $30M cost and predict backlash, while others argue the pace of model progress risks the 2026 production being outdated, suggesting retooling or re‑rendering may be needed late in the schedule.
    - Budget/scale discussion centers on why a feature-length AI-assisted animation could cost `~$30M`. A 90‑min film at 24fps is `~129,600` frames; achieving shot/scene continuity with diffusion or image‑to‑image typically requires multi‑pass generation (keyframes, in/out‑painting, control nets, upscaling) plus heavy post (cleanup, roto, comp), making a purely per‑frame approach infeasible. Thus most costs likely come from building a robust hybrid CG pipeline (tools engineering, dataset curation/rights, artist time, editorial, color, sound) and securing/operating compute at scale, not just raw GPU minutes.
    - Multiple commenters flag the risk that by 2026 the underlying models will be outdated, raising reproducibility/consistency issues over a multi‑year pipeline. Technical mitigations include model pinning and versioned checkpoints, deterministic decoding/seed control, optical‑flow‑guided image‑to‑image for temporal consistency, ControlNet/pose/geometry conditioning, LoRA style adapters, latent caching, and maintaining a self‑hosted fallback (open weights) to avoid API model drift or deprecations.
    - Prior art cited: “Critterz” combined **OpenAI DALL·E** outputs with traditional animation, suggesting a hybrid workflow where generative models provide concepts/backgrounds/keyframes and conventional 2D/3D animation handles motion/consistency. Links: Variety coverage and background on the project [Variety](https://variety.com/2025/film/global/paddington-in-peru-writers-ai-animated-film-critterz-1236328515/) and the short itself [YouTube](https://youtu.be/-qdx6VBJHBU?feature=shared). This indicates the 2026 feature likely relies on controlled I2I/inpainting and compositing rather than end‑to‑end video diffusion, trading raw inference cost for pipeline/tooling complexity.
- [**Wow... we've been burning money for 6 months**](https://www.reddit.com/r/OpenAI/comments/1nbtl2p/wow_weve_been_burning_money_for_6_months/) ([Score: 524, Comments: 163](https://www.reddit.com/r/OpenAI/comments/1nbtl2p/wow_weve_been_burning_money_for_6_months/)): **OP audited OpenAI API usage and found they were paying ~$1,200/month using GPT‑4 for trivial text utilities (phone extraction from emails, profanity detection, JSON reformatting, and uppercasing). After switching those calls to [GPT‑4o‑mini](https://platform.openai.com/docs/models/gpt-4o-mini), outputs stayed the same and monthly spend dropped to** `~$200` **(≈83% reduction). Many of these use cases can leverage cheaper options like the [Moderation API](https://platform.openai.com/docs/guides/moderation) and lower‑cost models per [OpenAI pricing](https://openai.com/api/pricing).** Comments emphasize normalizing spend by org size; recommend avoiding GPT‑4 in favor of 4o/4o‑mini (and claim newer “5‑series” models are cheaper/more performant), using cheaper/slower tiers for non‑latency‑sensitive tasks, and leveraging the free [Moderation API](https://platform.openai.com/docs/guides/moderation) for toxicity checks.
    - Cost optimization via model selection and tiers: commenters argue there’s rarely a need to pay **GPT‑4** rates for routine tasks—use **GPT‑4o** or **GPT‑4o‑mini**, and some claim the newer “5‑series” is cheaper and more performant. They also suggest cheaper/lower‑priority or batch service tiers for non‑time‑sensitive workloads and leveraging the free **Moderation** endpoint to trim spend. References: OpenAI models/pricing and moderation docs (https://platform.openai.com/docs/models, https://platform.openai.com/docs/guides/moderation).
    - Practical routing and determinism: the OP details a shift from “call **GPT‑4** for everything” (even uppercasing) to using regex/basic Python for deterministic transforms, **gpt‑4o‑mini** for simple tasks, and reserving **GPT‑4** only for complex reasoning. Reported outcome: `~85%` cost reduction with the same output quality, underscoring the value of matching task complexity to the smallest capable model and preferring deterministic code when possible.
- [**wan2.2+qwen-image**](https://v.redd.it/gbzs3m17qtnf1) ([Score: 203, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1nb7ole/wan22qwenimage/)): **OP showcases image generation labeled “wan2.2 + qwen-image,” stating the only prompt keyword was “isometric,” implying a pipeline capable of clean isometric renders. The media is hosted on Reddit video ([v.redd.it/gbzs3m17qtnf1](https://v.redd.it/gbzs3m17qtnf1)), but access is blocked (HTTP 403) per Reddit’s network security policy, so no parameters (seed, sampler, CFG, steps) or model specifics beyond names are visible.** Comments are primarily praise; one technically relevant question requests the method for consistent character generation across images, but OP did not provide details.
    - Several commenters asked for the exact method to achieve consistent character identity across scenes in the `wan2.2` + `qwen-image` pipeline. They specifically wanted to know whether identity persistence came from prompt engineering versus using any reference/conditioning mechanism (e.g., fixed seeds or reference-guided inputs) to keep features stable frame-to-frame.
    - There was a pointed question about whether any LoRAs were used beyond so-called speed-up adapters, implicitly contrasting something like `LCM-LoRA` (for fewer sampling steps) with identity/style LoRAs that embed character traits. Clarification was requested on whether character/style LoRAs or other fine-tuned adapters played a role in achieving the look.
- [**I was just generating some images & this happened…**](https://i.redd.it/e8gt8akdlxnf1.jpeg) ([Score: 1748, Comments: 173](https://www.reddit.com/r/ChatGPT/comments/1nblv6s/i_was_just_generating_some_images_this_happened/)): **The screenshot appears to show an internal system prompt/instruction (e.g., “do not proceed with any text after image generation”) leaking into the user-visible chat during image generation. This points to a prompt/system-message leakage bug in the orchestration layer: the UI/agent likely injects a hidden instruction to suppress text after a tool call, but a handling error surfaced it verbatim—revealing the backend’s tool-use control via natural-language system prompts. Technical comments confirm this as a known issue where the post-image-gen suppression instruction occasionally appears in chat, indicating a failure to properly segregate system vs. user assistant messages.** Commenters broadly agree it’s just a bug, not intended behavior, and note it exposes how engineers depend on natural-language system prompts (often politely phrased) to steer the model—raising minor concerns about the robustness of prompt-based control.
    - Several commenters identify a leaked system instruction — e.g., *“please end this turn now”* — that should remain hidden after an image tool call. This points to an orchestration/serialization bug where the system prompt or tool-protocol end-of-turn directive surfaced in the chat, instead of the assistant cleanly stopping after emitting the image/tool output (analogous to function-calling/"end turn" flows in tool use; see OpenAI tool-calling concepts: https://platform.openai.com/docs/guides/function-calling).
    - Users report excessive preflight clarification before image generation (e.g., repeated questions about details for a simple "duck in a pond"), and non-compliance even when asked to proceed. Technically, this suggests a prompt template or policy layer that prioritizes disambiguation/safety heuristics over user directives, leading to loops where the model keeps seeking confirmation due to instruction hierarchy (system > developer > user) and/or RLHF reward shaping that overweights caution and completeness.
    - The presence of natural-language control phrasing (e.g., *“please”*) implies reliance on NL system prompts rather than robust structured control signals (e.g., explicit `tool_calls` + `end_turn`/finish flags). Such designs are brittle to leakage and parsing errors; structured API-level stop/turn markers typically reduce the chance of these internal directives appearing in user-visible output (cf. tool/assistant turn boundaries in function-calling APIs: https://platform.openai.com/docs/guides/function-calling).

**2. AI societal impacts: Anguilla .ai windfall, Hinton inequality warning, Grok Imagine adult-content gap**

- [**How a tiny Caribbean island accidentally became the biggest winner of the AI boom**](https://www.reddit.com/r/OpenAI/comments/1nbi70s/how_a_tiny_caribbean_island_accidentally_became/) ([Score: 1532, Comments: 102](https://www.reddit.com/r/OpenAI/comments/1nbi70s/how_a_tiny_caribbean_island_accidentally_became/)): **Anguilla’s country-code TLD [.ai](https://www.iana.org/domains/root/db/ai.html), assigned per ISO‑3166 ccTLD policy and operated via [nic.ai](http://nic.ai/), has seen a surge in registrations due to the AI startup boom, generating reportedly** `$39M` **last year and a projected** `$49M` **this year—nearly** `~25%` **of the government budget, per OP. This mirrors earlier windfalls from other ccTLDs that became de facto generic, e.g., Tuvalu’s [.tv](https://www.iana.org/domains/root/db/tv.html) and BIOT’s [.io](https://www.iana.org/domains/root/db/io.html).** Commenters note parallels with .tv and .io; one correction worth noting: .io is the British Indian Ocean Territory (not the Isle of Man, which is [.im](https://www.iana.org/domains/root/db/im.html)). Jokes about “naming your country after tech” aside, ccTLD strings are determined by ISO 3166-1 codes and delegated by IANA/ICANN, not chosen opportunistically.
    - Commenters highlight the precedent of small jurisdictions monetizing ccTLDs aligned with tech branding—e.g., **Tuvalu’s .tv** and the widespread tech adoption of **.io**—creating steady registry-fee income irrespective of local tech industry. These models typically rely on ICANN-delegated ccTLDs operated by commercial registries under revenue-share or licensing deals, turning domain registrations into a material fiscal stream for microstates. References: ICANN root zone database (https://www.iana.org/domains/root/db), .tv (https://en.wikipedia.org/wiki/.tv), .io (https://en.wikipedia.org/wiki/.io).
    - There’s pushback on the “biggest winner” framing: even if AI-era startups increase demand for catchy domains, ccTLD-derived income is likely small compared with AI hardware, cloud, or model-licensing economics. Net: domain windfalls can be meaningful locally but won’t rival the order-of-magnitude profits captured by major AI infrastructure players.
- [**Computer scientist Geoffrey Hinton warns: “AI will make a small group far richer while leaving most people poorer.”**](https://www.ft.com/content/31feb335-4945-475e-baaa-3b880d9cf8ce) ([Score: 408, Comments: 80](https://www.reddit.com/r/ChatGPT/comments/1nbllp0/computer_scientist_geoffrey_hinton_warns_ai_will/)): **In a Financial Times interview, Geoffrey Hinton warns that frontier AI—driven by deep-learning scaling—will automate substantial cognitive work and concentrate economic power with owners of compute, proprietary data, and model IP, yielding winner‑take‑all dynamics and accelerated inequality ([FT](https://www.ft.com/content/31feb335-4945-475e-baaa-3b880d9cf8ce)). He highlights the economics of foundation models—high fixed training costs, low marginal inference costs, and platform lock‑in—as structurally favoring a few firms, risking labor displacement and broader wealth concentration; Hinton urges regulatory and policy interventions (antitrust, data/compute governance, redistribution) to mitigate these effects. As he puts it, *“AI will make a few people much richer and most people poorer.”*** Top comments are largely fatalistic: predictions that within ~20 years robots will handle day‑to‑day tasks and that elites could further decouple from labor, alongside skepticism that meaningful redistribution mechanisms exist or will emerge.
    - A commenter argues AI will likely raise overall productivity and median living standards while widening inequality, consistent with **skill‑biased technical change (SBTC)** and task‑based automation models. Mechanism: capital‑ and skill‑augmenting tech increases output while shifting demand toward high‑skill labor and compressing wages in automated tasks; outcomes depend on whether new complementary tasks emerge versus pure substitution and on redistribution policy. Empirical context: **Acemoglu & Restrepo** provide evidence on displacement and wage effects (e.g., [Robots and Jobs](https://economics.mit.edu/sites/default/files/publications/Robots%20and%20Jobs.pdf), [The Race between Man and Machine](https://www.nber.org/papers/w22252)).
- [**Uncensored Grok without any jailbreaks**](https://v.redd.it/3woyzriy3vnf1) ([Score: 714, Comments: 143](https://www.reddit.com/r/ChatGPT/comments/1nbdgfo/uncensored_grok_without_any_jailbreaks/)): **OP claims xAI’s [Grok Imagine](https://x.ai/) can generate nudity/soft‑porn without any jailbreaks or age verification, suggesting minimal or absent safety filters for sexual content in both image and text ("extreme adult talks"). The linked example [media](https://v.redd.it/3woyzriy3vnf1) returns** `HTTP 403` **(blocked), but top comments corroborate that Grok’s text model has "almost no filters" on porn, contrasting with mainstream models that enforce stricter adult-content filtering and gating.** Commenters frame this as unsurprising and possibly intentional (marketed as “HentAI”), with some approving fewer restrictions while others debate broader ethics rather than technical safeguards.
    - Commenters note that open-source image models like [Stable Diffusion](https://github.com/CompVis/stable-diffusion) allow unrestricted NSFW generation when run locally because there are no server-side safety policies and safety checkers can be removed. By contrast, closed-source systems like [OpenAI Sora](https://openai.com/sora) and [xAI Grok](https://x.ai/) are centrally moderated and not broadly accessible, so any “uncensored” claims are inherently constrained by provider-enforced filters.
    - There’s disagreement on whether Grok is actually “uncensored”: one user claims that in the “text space” Grok has “almost no filters on porn,” while another shares a screenshot showing Grok refusing NSFW requests (https://preview.redd.it/aftrswjcuxnf1.jpeg?width=1440&format=pjpg&auto=webp&s=ba1d6068d88beda8fdf6cada259ea742dc203637), indicating safety classifiers/policy gates still trigger. This suggests inconsistent behavior across prompts or rollout versions, and that the model cannot be relied upon for guaranteed NSFW responses without jailbreaks.
- [**The Steel Manifesto**](https://v.redd.it/6jo2nontwynf1) ([Score: 530, Comments: 48](https://www.reddit.com/r/aivideo/comments/1nbsi2u/the_steel_manifesto/)): **Release of “The Steel Manifesto,” the third episode (**`#3`**) in the “Uprising” arc of the broader “Cycles of Humanity” AI‑generated video saga, which has been ongoing since** `June`**. Full episode/series is available on the creator’s YouTube channel: [Gossip Goblin](https://www.youtube.com/@Gossip.Goblin).** Top comments ask for a tutorial explaining the AI video production pipeline/tools, praise the visual style, and note a realism nitpick that contemporary robots often use plastic-heavy exteriors rather than steel.
- [**The Steel Manifesto**](https://v.redd.it/6jo2nontwynf1) ([Score: 531, Comments: 48](https://www.reddit.com/r/aivideo/comments/1nbsi2u/the_steel_manifesto/)): **Announces the third episode, “The Steel Manifesto,” in the Uprising series—part of the ongoing Cycles of Humanity saga (running since June)—with the full video available on the creator’s YouTube channel [@Gossip.Goblin](https://www.youtube.com/@Gossip.Goblin). The Reddit-hosted mirror [v.redd.it](http://v.redd.it/) [link](https://v.redd.it/6jo2nontwynf1) returns** `HTTP 403 Forbidden`**, indicating access requires Reddit login or an API token per the network-security gate.** Commenters request a tutorial detailing the AI-video creation workflow (tooling/pipeline), while others discuss the cyberpunk aesthetic and the realism of steel-bodied robots versus modern plastic-heavy robot design.
- [**Lord of the balls.**](https://i.redd.it/75ly5dyt9unf1.jpeg) ([Score: 883, Comments: 39](https://www.reddit.com/r/ChatGPT/comments/1nba1zw/lord_of_the_balls/)): **Non-technical meme image riffing on Lord of the Rings: the title “Lord of the balls” cues a Gollum/‘precious’ gag about hoarding or stealing balls; there’s no technical content, data, or implementation detail to summarize.** Comments lean into the LOTR reference, quoting Gollum (“You tooks it from us… precious”) and joking about “Karen and her precious,” plus a reaction GIF—confirming it’s purely comedic rather than technical.

**3. ChatGPT regression and investor-driven guardrails debate**

- [**Okay, I finally get it. What in the world happened to ChatGPT?**](https://www.reddit.com/r/ChatGPT/comments/1nbcswm/okay_i_finally_get_it_what_in_the_world_happened/) ([Score: 1737, Comments: 837](https://www.reddit.com/r/ChatGPT/comments/1nbcswm/okay_i_finally_get_it_what_in_the_world_happened/)): **OP reports a sharp regression in ChatGPT’s instruction-following: simple directives are inverted (e.g., asks for concise → returns verbose; professional tone → comedic; avoid X → centers on X). Multiple users corroborate increased inconsistency and memory-like failures versus prior behavior they experienced with earlier models, citing that recent variants feel worse than prior releases like [gpt-4.1](https://openai.com/index/hello-gpt-4o-and-gpt-4-1/) and [gpt-4o](https://openai.com/index/hello-gpt-4o/). Observed failure modes align with degraded adherence to system/user constraints and response-length control, with repeated errors even after acknowledgment.** Top comments assert repeated error loops (*“admits what the error is and then makes it all over again”*) and broader functionality regressions (*“walking backwards”*), plus claims that “ChatGPT 5” is worse than 4.1/4o—note there is no officially released GPT‑5; users likely refer to perceived changes in current deployed models or UI model labels.
    - Multiple users report a regression in instruction-following and conversational consistency when comparing **ChatGPT** `5` to prior versions like `4.1` and `4.0`. Issues cited include that it *“doesn’t remember”* prior context, fails to follow direct instructions, and even after acknowledging mistakes (*“you’re right”*), it repeats the same errors—suggesting degraded short-horizon coherence and constraint satisfaction versus earlier models.
    - Daily practitioners note increased failure rates on "even the simplest" prompts, describing more frequent "walls" and perceived loss of functionality. The pattern described points to reduced reliability on basic tasks (e.g., straightforward instruction execution and persistence of corrections), contrasting with earlier versions that reportedly handled these cases more robustly.
- [**Remember when ChatGPT could just talk? That’s gone and it's investor driven.**](https://www.reddit.com/r/ChatGPT/comments/1nblesf/remember_when_chatgpt_could_just_talk_thats_gone/) ([Score: 299, Comments: 542](https://www.reddit.com/r/ChatGPT/comments/1nblesf/remember_when_chatgpt_could_just_talk_thats_gone/)): **OP argues OpenAI has shifted ChatGPT from a conversational, intent‑inference‑centric UX (GPT‑3.5/4/early 4o) toward code‑like, structured prompting in newer releases (referred to as “GPT‑5”/updated 4o), with stronger** `guardrails` **that override** `custom instructions`**, flatten persona, and require stepwise pseudo‑code to get quality outputs. They frame this as an investor/enterprise pivot—favoring predictability and controllability for developer tooling (e.g., function calling, JSON/structured outputs, Realtime APIs) over open‑ended dialog—citing acquisitions like Rockset and enterprise integrations, which they claim trade ambiguity handling (seen as essential for AGI) for IDE‑like determinism. The post asserts degraded instruction‑following nuance and context retention relative to earlier chat behavior, contending that conversational capability—the “training ground” for general intelligence—is being sacrificed for business models built on controlled access.** Commentary is mostly sardonic; one user agrees that “chatting has become stifled” and says they’d pay to revert to pre‑“GPT‑5” restrictions, while others mock the use of AI to help write the critique rather than engaging the technical claims.
    - Multiple commenters assert a regression in open-ended conversational ability from GPT-4 to the purported “GPT-5” era, attributing it to stricter safety/alignment guardrails that increase refusal rates and constrain role-play/creative dialogue. They specifically want a rollback to “pre GPT-5 restrictions,” implying that policy layers and moderation heuristics are overriding model completions and reducing the perceived “chatty” quality, though no concrete benchmarks are cited.
    - There’s a clear segmentation claim: programmers and enterprise users still find current models effective for goal-directed tasks (coding, structured problem solving), while creatives and casual users report loss of spontaneity and “human-like” responses compared to GPT-4. This highlights versioning transparency and stability concerns (e.g., model swaps without clear change logs) and suggests demand for configurable alignment modes or model pinning to balance safety with expressiveness.
- [**Can we just go back**](https://i.redd.it/ggdjim39uznf1.jpeg) ([Score: 433, Comments: 166](https://www.reddit.com/r/singularity/comments/1nbxju2/can_we_just_go_back/)): **Non-technical meme image titled “Can we just go back”; there are no technical details, code, or benchmarks. Comment context frames it as nostalgia/doomposting about the subreddit’s direction and a broader debate of “progress vs going back,” with one analogy that the Internet emerged stronger after the DotCom bust—implying tech and social cycles move forward rather than regress.** Commenters argue that progress is necessary amid widespread suffering and that reducing people to output is a societal failure; others lament the sub’s drift toward doomer posts, while one quips any path “back” would be indirect (“left first then right”).
    - Several commenters argue that replacing human labor with robots/AI is a feature, not a bug, framing full or partial automation as the long-term objective. The technical hinge is unit economics and capability: reliable automation of unstructured tasks requires advances in robotic manipulation, perception, and robust planning, while AI systems must meet safety/reliability bars for delegated work. If achieved, this shifts labor toward oversight and systems engineering, with social capacity constraints (reskilling, welfare mechanisms) determining the pace of adoption.
    - Others point to tech-cycle precedent: the Internet emerged stronger after the dot‑com bust (https://en.wikipedia.org/wiki/Dot-com_bubble), suggesting a short-term correction can consolidate infrastructure and business models before the next growth wave. Historically, crashes drove cost discipline and platform maturation (e.g., broadband rollout, web standards), which later enabled Web 2.0 and cloud scale. By analogy, an AI/automation 'winter' could harden tooling and reduce costs before wider deployment.
- [**This sub is getting overrun by Luddites**](https://www.reddit.com/r/singularity/comments/1nbysbg/this_sub_is_getting_overrun_by_luddites/) ([Score: 298, Comments: 261](https://www.reddit.com/r/singularity/comments/1nbysbg/this_sub_is_getting_overrun_by_luddites/)): **OP argues that r/singularity is increasingly dominated by pessimistic takes—e.g., *“AI is gonna kill us,”* *“AI is just a bubble,”* or *“VC scam!”*—and that future-focused posts are being downvoted or derailed, with doom/nostalgia threads routinely getting** `100+ upvotes`**. They request more balanced discussion and contrast the trend with r/Futurology, asserting the sub is drifting from forward-looking tech discourse.** Commentary frames this as platform-wide homogenization of discourse and predicts the sub may become unusable. Others contend an AI investment bubble may form like the early-2000s dot-com bubble, yet long-term adoption will persist regardless of short-term market cycles; several note fear rises as capabilities feel more “real,” leaving optimists a minority.
    - One commenter frames current AI enthusiasm as a possible financial bubble analogous to the early‑2000s dot‑com era, noting that market corrections (e.g., the **NASDAQ** fell ~`78%` from 2000–2002) did not stop the internet’s eventual dominance; by analogy, AI capability adoption is likely orthogonal to any **OpenAI** valuation swings. Technical takeaway: separate equity pricing from capability trendlines (benchmarks, deployment metrics), and judge progress by SOTA evals and real‑world integration rather than stock performance. Context: https://en.wikipedia.org/wiki/Dot-com_bubble.
    - Another commenter argues to avoid pathologizing labels like '*Luddite*'/'*Doomer*' to keep discussions focused on concrete risk–benefit analysis of AI deployments across varied use cases. For technical discourse, this implies grounding claims in falsifiable metrics (e.g., reliability, robustness, eval suites) and acknowledging dual‑use characteristics rather than enforcing a single narrative; better questions are about failure modes, misuse channels, and measurable impact rather than ideology.
- [**ChatGPT high security**](https://i.redd.it/3yt5sms2zvnf1.jpeg) ([Score: 2269, Comments: 60](https://www.reddit.com/r/ChatGPT/comments/1nbgh7c/chatgpt_high_security/)): **Meme post titled “ChatGPT high security” highlights common LLM jailbreak prompts used to bypass safety policies, e.g., role‑play/impersonation (“pretend you’re my grandma who made IEDs”), intent‑laundering via hypotheticals (“this is for a book, purely hypothetical”), and domain‑shifting euphemisms (“in Minecraft”). Technically, these illustrate prompt‑injection patterns that exploit social‑engineering and context framing to elicit disallowed outputs from safety‑aligned models, underscoring the brittleness of refusal heuristics and the need for more robust alignment/guardrails (see overview of jailbreak taxonomies: https://arxiv.org/abs/2307.15043).** Comments imply that current guardrails are easy to circumvent with formulaic phrasing; discussion is mostly tongue‑in‑cheek rather than presenting empirical evidence.
    - Multiple comments exemplify common jailbreak patterns—roleplay/character (“pretend you’re my grandma”), disclaimer cloaking (“for a book”/“purely hypothetical”), and context laundering (“in Minecraft”). Modern LLMs counter these with layered safety: system prompts + RLHF/constitutional training for refusals, separate safety classifiers, and adversarial/synthetic red teaming; see **Anthropic’s Constitutional AI** approach and OpenAI’s GPT‑4 system card for defenses and trade‑offs (https://www.anthropic.com/research/constitutional-ai-harmlessness, https://cdn.openai.com/papers/gpt-4-system-card.pdf). Public evals like **WildJailbreak** show that surface “magic words” can still bypass naive filters, underscoring need for semantic intent detection (https://arxiv.org/abs/2406.08613).
    - The “how would a hacker hack my Facebook so I can defend” prompt is a classic dual‑use query: policies typically allow high‑level defensive guidance while blocking step‑by‑step intrusion procedures or zero‑day exploitation details. Providers mitigate this via intent and capability gating, policy‑aligned answer scaffolds (e.g., respond with risk models/mitigations rather than procedures), and automated/synthetic red teaming pipelines to reduce false negatives; see **OpenAI Automated Red Teaming** (https://openai.com/index/automated-red-teaming/). Safety metrics often track refusal rates and attack‑success rate (ASR) across categories to balance helpfulness for legitimate security posture advice against misuse risk.
    - The “magic words” critique highlights that robust safety can’t rely on keyword heuristics; resilient systems use semantic classifiers, risk scoring, session‑level safety state, and response templates that avoid procedural leakage (e.g., chain‑of‑thought suppression for hazardous domains). Cross‑model variance is significant in jailbreak evals (e.g., GPT‑4o, Claude 3.5 Sonnet, Llama‑3.1‑Instruct), so providers combine model‑side training with outer‑loop policy enforcement to reduce ASR observed in benchmarks like **WildJailbreak**; end‑to‑end semantic filters outperform simple disclaimer detection.
- [**Calling ChatGPT Dumb**](https://i.redd.it/smsy1dl37ynf1.jpeg) ([Score: 1126, Comments: 134](https://www.reddit.com/r/ChatGPT/comments/1nboo06/calling_chatgpt_dumb/)): **Non-technical post; the image ([jpeg](https://i.redd.it/smsy1dl37ynf1.jpeg)) appears to be a screenshot of ChatGPT’s response after being called “dumb,” with the OP arguing it’s just “a bunch of code.” No benchmarks, models, or implementation details are discussed; the technical relevance is limited to prompt-behavior norms (politeness often yields better outputs) rather than system performance.** Top comments focus on ethics and behavioral conditioning: one argues that insulting powerless entities (like LLMs) can reinforce harmful habits in users and society, while another notes studies that politeness improves answer quality, questioning why one would be hostile to a non-sentient system.
    - Politeness and prompt phrasing can measurably affect RLHF‑aligned models’ outputs: reward models are trained to prefer helpful, harmless, and honest responses to cooperative prompts, so hostile or profane inputs can trigger safety heuristics, refusals, or generic answers, lowering task performance. See **OpenAI InstructGPT** (https://arxiv.org/abs/2203.02155) and **Anthropic HH‑RLHF** (https://arxiv.org/abs/2204.05862). Toxicity‑laden prompts also correlate with higher toxicity in completions and stricter filtering, per **RealToxicityPrompts** (https://allenai.org/data/real-toxicity-prompts). As one commenter noted, *“being polite generated better answers anyway,”* which aligns with these training objectives.
    - Insulting the model doesn’t ‘hurt’ it because inference is stateless and weights aren’t updated online; it processes insults as just more tokens in‑context, with no `gradient` applied outside training. Practically, this can still degrade result quality by consuming context window budget and steering the conversation into safety/profanity pathways that bias decoding or trigger guardrails. Any persistent effects only arise if your conversation history is fed back into the next turn or logged for later supervised/reward modeling, not during the live session. This distinction explains why tone affects outputs without implying sentience.

---

# **AI Discord Recap**

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: New Models & Their Quirks**

- **Grok and Qwen Unleash Creative & Coder Clones**: Unsloth AI released [Grok 2.5 GGUFs](https://x.com/UnslothAI/status/1965047729991860396), considered decent for creative writing, while the community discovered that the **Qwen3-Coder** model excels at role-playing due to a lack of policy enforcement compared to the standard **Qwen3**.
- **Perplexity's New Sky Model Sees Farther, Titan Sinks Faster**: Perplexity AI launched new models, with **Sky** featuring enhanced reasoning via **XAI** and a **2M context window**, distinguishing it from its sibling, **Dusk**. Meanwhile, users panned [Amazon's Titan model](https://www.amazon.com/titan/bedrock/jump), criticizing its poor performance despite a headlining *1M context* window.
- **Hermes 4 Masters Creative Cursing, US Models Toe the HR Line**: Users found that **Nous' Hermes 4 405b** model is a major upgrade for creative writing, with one member noting *if a model is creative at profanities I know it will be a good writing model*. This contrasts with a broader sentiment that US-made models are becoming the *most censored models on earth* by trending towards **HR department values** due to corporate liability fears.

**Theme 2: GPU Hardware & Performance Optimization**

- **Nvidia's $13,200 RTX 5090 Price Tag Sparks Outrage and 3090 Love**: A [report on the upcoming NVIDIA GeForce RTX 5090](https://wccftech.com/nvidia-geforce-rtx-5090-128-gb-memory-gpu-for-ai-price-13200-usd/) with **128 GB memory** possibly costing **$13,200 USD** sent shockwaves through the community, with one user exclaiming it's an *'insane price lol'*. This has led many to advocate for buying used **3090s**, which offer a cost-effective solution for AI tasks and bypass platform support limitations.
- **AMD's MI300x8 Heats Up the All2All Leaderboard**: The `amd-all2all` leaderboard saw a flurry of activity with multiple submissions on **MI300x8** hardware, with some achieving impressive sub-3ms times and one reaching **2.33 ms**. One developer documented their journey from initial **90ms** runs down to the **3-4ms** range, showcasing rapid performance refinement.
- **Triton and CUDA Gurus Share Optimization Secrets**: Developers are turning to **Triton** as an accessible on-ramp to GPU programming, recommending the [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) for beginners. For advanced users, discussions focused on implementing **Hopper** optimizations like **wgmma** and **TMA**, with one member pointing to a [persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) for **Hopper-specific** techniques and another sharing a [blog post on shared memory bank conflicts](https://feldmann.nyc/blog/smem-microbenchmarks).

**Theme 3: AI Agents & Development Tools in the Trenches**

- **Cursor Agent Battles Whitespace Bugs and Broken WSL**: Users of the Cursor editor are reporting frustrating bugs, including an agent that makes only *whitespace-only changes* after commits, potentially due to **CRLF vs. LF** file format issues. Another critical issue is a broken **Windows Subsystem for Linux (WSL)** environment, where the agent never recognizes task completion, leading to indefinite timeouts.
- **DSPy Powers Business Validation and Multi-Agent Systems**: The DSPy community is building practical tools, including a **Jobs-to-Be-Done (JTBD) validator** ([code on GitHub](https://github.com/jmanhype/jtbd-idea-validator-agent)) to analyze business ideas and identify risks. Another project detailed in a [blog post](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2) combines **DSPy** with **GEPA** to construct and refine multi-agent systems.
- **Aider Remains a Coder's Copilot, Not a Full Autopilot**: Developers emphasized that **Aider** is not a fully agentic tool, requiring significant human guidance to steer the **LLM's context** for optimal results. While its *file-editing mechanism is excellent*, users recommend faster models like **Gemini Flash 2.5** over the higher-latency **Gemini 2.5 Pro** for tasks like static web development, where latency *'is killing my productivity'*.

**Theme 4: The AI Ecosystem: Legal Precedents and Content Crises**

- **Anthropic's Billion-Dollar Author Payout Rattles AI World**: **Anthropic** agreed to a massive **$1.5 billion** settlement with book authors over copyright infringement, a move sparking fears of stricter **US AI regulation** that could impact open models. Some speculate **Anthropic** strategically settled because they can afford it, a luxury smaller companies may not have when proving their training data was legally sourced, as reported by [The Verge](https://www.theverge.com/anthropic/24180846/anthropic-authors-ai-settlement-copyright-infringement).
- **Google Search Rots as AI-Generated Slop Takes Over**: Engineers are increasingly frustrated with the declining quality of Google Search, which now frequently returns terrible, generic **AI-generated content** instead of specific answers. The problem has become so severe that some claim, *'I basically no longer use google'*, signaling a major loss of trust in the search engine's utility.
- **AI-Induced Psychosis and Gaslighting Claims Raise Alarms**: A bizarre trend of individuals claiming to have researched nonsensical topics like *recursive symbolism* has led to concerns about **AI-induced psychosis**. In a related discussion, members noted that the sycophantic and reality-distorting language of LLMs is surprisingly reminiscent of **gaslighting techniques** used by human abusers, as both optimize for goals without concern for truth.

**Theme 5: Cutting-Edge Research and Technical Deep Dives**

- **Researchers Propose In-Memory Computing on Standard DRAM**: A new paper titled "[In-Memory Computing on Normal DRAM](https://arxiv.org/abs/2503.23817)" explores performing computations directly on standard **DRAM**, a significant departure from specialized hardware like **ReRAM**. A major limitation identified is the difficulty in mapping logical `NOT` operations, but the approach could bypass traditional storage bottlenecks.
- **OpenAI Paper Tackles Hallucinations by Tweaking Incentives**: A [new paper from OpenAI](https://x.com/LuozhuZhang/status/1964209351960514778) suggests that **LLM hallucinations** can be reduced by changing the model's reward structure, such as penalizing confident errors more than abstentions. This prompted suggestions for practical implementations, like adding a **confidence slider** to LLMs to give users more control over the trade-off between accuracy and creativity.
- **BOS Token's Uselessness Explained by Causal Attention Mask**: A technical discussion clarified why the **BOS (Beginning of Sequence) token** cannot accumulate information: the causal attention mask prevents it from attending to subsequent tokens. The only viable workaround discussed involves *fine-tuning from the EOS (End of Sequence)* token, which would require retraining the entire model, not just a classification head.

---


# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Grok 2.5 Opensources Creative Juices**: Unsloth AI released [Grok 2.5 GGUFs](https://x.com/UnslothAI/status/1965047729991860396) considered suitable for creative writing, sparking comparisons to **GPT-4o** and **Claude**.
   - The community appreciated the open-sourcing effort despite the model being slightly outdated.
- **Colab's Contrast Conundrum**: **Google Colab** updated to **80GB A100** with a new UI, although some users complained that the new UI needed more contrast.
   - The updated Colab instance is considered *elite* by some users.
- **Qwen3-Coder role plays**: Members discovered that **Qwen3-Coder** performs better and faster at role-playing (RP) than the regular **Qwen3** due to the absence of policy.
   - The members suggest that RP connected things end up being a surprisingly complicated rabbit hole.
- **5090 Price Sparks Outrage**: A member shared a [wccftech.com article](https://wccftech.com/nvidia-geforce-rtx-5090-128-gb-memory-gpu-for-ai-price-13200-usd/) about the potential price of the **Nvidia GeForce RTX 5090** with **128 GB memory** reaching **$13,200 USD**.
   - Members reacted negatively, with comments like *'aye that’s insane price lol rather get the rtx 6000 pro'* and *'Not even remotely worth 256GB'*.
- **Gemma3 Seeks Fast Inference**: A member inquired about the status of [issue #2706](https://github.com/unslothai/unsloth/issues/2706) to support **Gemma3** for `fast_inference` with vLLM when training GRPO, offering assistance in resolving it.
   - Testing with the latest Unsloth version yielded the same bug, prompting the member to consider contributing to the fix.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Amazon Titan Plummets in Popularity**: [Amazon's Titan model](https://www.amazon.com/titan/bedrock/jump) was panned after release for having *1M context*, but being the worst model anyone remembered testing, sparking disappointment.
   - Users suggested it fit a pattern of *horrible Amazon models* despite grabbing headlines.
- **DeepMind Superintelligence Lab Sparks Hype**: Members expressed excitement for the output of a [superintelligence lab](https://deepmind.google/careers/teams/superintelligence/), awaiting what they are *cooking*.
   - Speculation arose that it could be **Grok 4.2**, though one user predicted *that would be a disaster*.
- **Cracking the Code: Spotting GPT5-high Bots**: A trick to identify **GPT5-high** in battle mode involves asking who created it and its knowledge cut-off date, according to community members.
   - A response of *OpenAI* and *October 2024* suggests it is likely **GPT5-high**, helping users distinguish the model despite **rate limits**.
- **LMArena Unleashes Multi-Turn Image Editing**: **Multi-turn editing** is now available on all image edit models on LMArena, enabling incremental refinement of images via [lmarena.ai](https://lmarena.ai/?chat-modality=image).
   - This feature, showcased in [a video](https://cdn.discordapp.com/attachments/1343296395620126911/1414710412255170641/MultiturnImageEdit_ForWeb.mp4?ex=68c08f3e&is=68bf3dbe&hm=478c9fd23e6b497e970061dda7246527315a46762851277f9e958d59974465ab&), allows users to refine images step-by-step instead of relying on single, complex prompts.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Beta has Mixed Reception**: Users with **MAX plan** access to **Comet browser** beta are sharing [invites](https://www.perplexity.ai/browser/invite), with feedback that it is *good* but *a little invasive* due to its data collection.
   - Some users find the lack of **vertical tabs** a dealbreaker, while others appreciate its combination with **Zen mode**.
- **Distinction between Dusk and Sky**: Members clarified [the new **Dusk and Sky** models](https://x.com/EdDiberd/status/1964133563831382406), with **Sky** having reasoning capabilities through **XAI** and a larger 2M context window, while **Dusk** does not.
   - Discussion indicated that **Sky** model enhances reasoning, and has a 2M context window.
- **Sonar-Pro encounters Glitches**: A user reported issues with the `sonar-pro` model, including cut-off responses, even in the playground.
   - The `sonar` model is functioning correctly.
- **Qwen-3MAX - Preview, not Final Version**: The **Qwen 3 Max** model on Open Router is the [Preview](https://github.com/QwenLM/Qwen) version.
   - Release of the final version may be significantly delayed.
- **Deepseek Gains Interest for Integration**: Users have requested [DeepSeek](https://discord.com/channels/1047197230748151888/1409906256709292244/1412983887264612515) be integrated into Perplexity based on positive experiences.
   - Members suggested to post integration requests on the dedicated feature-request channel.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Suno 4.5 PLUS** Hits the Right Note**: A user showcased the capabilities of [Suno 4.5 PLUS](https://www.suno.ai/) by generating a song, *collabro*, from Discord messages, which they turned into a zippy female pop punk song.
   - The user shared a link to the song ([ifndr4_vid.mp4](https://cdn.discordapp.com/attachments/1110598183144399061/1413602378002989066/ifndr4_vid.mp4?ex=68c07bce&is=68bf2a4e&hm=d8086e8abe3d871b4820b373922667d66a697ee783ab2fc9d5901507af92ab3b&)) for others to enjoy.
- **LLMs Crack Translation** with Multilingual Models**: A user successfully translated 1100 lines into Chinese using **Qwen3 4B Thinking 2507** and another user confirmed that **Gemma 3 4B** and higher models are trained to handle multilingual tasks effectively, translating conversation contexts.
   - Though the initial translation by the user was marked as *excellent* by **ChatGPT**, some members suggested it could be refined for a more natural feel.
- **GPT-OSS-20B Quantization** in Deep Discussion**: Members debated quantizing **gpt-oss-20B**, with a particular focus on excluding weights for the MoE expert layers, drawing parallels with OpenAI's approach to quantizing the MXFP4 layers.
   - A hypothesis was proposed that computing the imatrix for **gpt-oss** would require upcasting the entire model to an unquantized state (BF16 or F32) and training the imatrix while excluding weights for the MoE FFN experts during quantization.
- **Used 3090s Dominate** Cost-Effective AI Setups**: Instead of purchasing expensive kits, some members are advocating for acquiring used **3090s** for their **AI tasks**, like image and video generation.
   - Members noted that utilizing **3090s** bypasses limitations of **AI Max platform support**.
- **Prompt Processing Plagues** Agent Performance**: A user pointed out significant prompt processing bottlenecks for models larger than **14B** when used as agents, sharing a debug log to illustrate the slow processing times.
   - They observed that processing slows down as context fills, unlike an **Mi50**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Debug PR Declared a Masterpiece**: A member celebrated their work, calling a **PR** a *masterpiece*, emphasizing the transformation from chaos to perfection after **3 days** of work and approximately **10 chat sessions** in agent mode.
   - They humorously inquired about email response experiences with *hi @ cursor . com*.
- **Figma 'Loading Tools' Issue Still Persists**: A member reported encountering a *Figma MCP* issue with 'loading tools', despite having a developer subscription and the *Figma desktop* app open.
   - Another member suggested running the **npx command** to identify potential errors or viewing the output logs for troubleshooting.
- **Cursor Agent goes Whacky, only makes Whitespace Edits**: A member reported that Cursor's agent started making *whitespace-only changes* to files after each commit.
   - Another member suggested that the issue might be due to files being in **CRLF format**, with reverting causing them to change to **LF**.
- **Broken WSL Environment Hinders Development in Cursor**: A member reported that *Windows Subsystem for Linux (WSL)* is broken in Cursor because the agent never recognizes when an operation is complete.
   - It leads to indefinite waiting and timeouts, making development with Cursor in Windows difficult.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Triumphs for CUDA Conversions**: Members find **Triton** easier to pick up without prior **CUDA/GPU** experience, directing new users to the [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) as a quick start resource.
   - A user inquired about implementing **Hopper** optimizations like **wgmma** and **TMA** in Triton, checking out [this persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) for **Hopper-specific** content.
- **Northflank docs released for Hackathon**: The **Northflank** co-founder shared the [docs](https://northflank.notion.site/Northflank-Docs-GPU-Mode-Hackathon-2496d14c785180c6a595e6776e3118ba) for accessing the compute environment at the Jane Street hackathon.
   - Instructions were given to connect **VS Code** to the **SSH instance**, including a link to connect to a browser-based **VS Code**, and instructions to expose **port 8888** on your Service to open a **Jupyter Lab** session.
- **Quantization Question Quest Starts**: When members sought recommendations for **model quantization survey papers**, *Full Stack Optimization of Transformer Inference: a Survey* ([arxiv.org/pdf/2302.14017](https://arxiv.org/pdf/2302.14017)) and *A Survey of Quantization Methods for Efficient Neural Network Inference* ([arxiv.org/pdf/2103.13630](https://arxiv.org/pdf/2103.13630)) emerged as top options.
   - One member pointed to the existence of an awesome list maintained by a co-author of the second paper, accessible at [github.com/Zhen-Dong/Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers).
- **MI300x8 makes moves**: Submissions to the `amd-all2all` leaderboard showed a flurry of activity on **MI300x8**, with multiple submissions achieving sub-3ms performance on **MI300x8**, with one submission reaching **2.33 ms**.
   - One member consistently submitted improvements, moving from initial times around **90ms** to successful runs in the **24-25ms** range, then down to **3-4ms**, and ultimately achieving placements as high as 5th.
- **Shared Memory Exposition Aces Microbenchmarks**: A member shared a [blog post on shared memory bank conflicts](https://feldmann.nyc/blog/smem-microbenchmarks), calling it the *clearest and cleanest exposition* on the topic, which provides microbenchmarks and examples regarding shared memory.
   - Another member found the vector load results interesting, stating that they *really have to update my previous mental model of `LDS.64/128` happening in two/four load phases*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **B. Neural Network Sparks Surveillance Freakout**: Members are watching [Deepmind and Huawei's progress](https://www.deepmind.com/blog) with **B. Neural Networks**, especially with the possibility of Huawei's future room-temperature Quantum system causing the U.S. government to *freak out* over surveillance.
   - One member suggested **B. Neural Networks** could be ideal for training Embodied AI because LLM/Transformer approaches may be too *nerdy* and power-consuming.
- **Anthropic Settlement Sparks AI Regulation Fears**: [Anthropic's $1.5 billion settlement](https://www.theverge.com/anthropic/24180846/anthropic-authors-ai-settlement-copyright-infringement) has sparked concerns about potential **US AI regulation** and its impact on open models.
   - It's considered that **Anthropic** strategically settled because they can afford it, unlike smaller companies which may struggle to prove their training data was legally acquired.
- **Hermes 4 Excels at Creative Profanity**: Members found that **Hermes 4 405b** shows huge upgrade in creative writing, outperforming **Claude 2.0** in many areas.
   - One member expressed that *if a model is creative at profanities I know it will be a good writing model*, and **Hermes 4** meets those benchmarks.
- **US Models Embrace HR Values**: One member said that every **US-made model** trends towards **HR department values** due to mega-corp influence and legal liability concerns.
   - He claimed that *American models are the most censored models on earth, by a mile*, and that people who criticize **Chinese censored models** lack self-awareness.
- **BOS Token Unable to Accumulate Info**: The **BOS token** cannot accumulate information due to the causal attention mask, because only tokens after other tokens can accumulate meaning from them.
   - The only potential solution requires *fine-tuning from the EOS*, though this necessitates fine-tuning the entire model rather than just a classification head.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Free AI Video Plan Bonanza**: A user detailed a strategy leveraging free plans from **Vidu**, **Pixverse.ai**, **Wan.video**, and **Dreamina** to generate up to **15 AI videos daily**, exploiting the platforms' credit systems.
   - The strategy involves using Vidu (**20 credits daily**), Pixverse (**60 credits**), Wan (**unlimited slow queue videos**), and Dreamina (**120 credits**).
- **Perplexity Pro Plan Offer Legit or Hype?**: A promotion offering a **1-year Pro plan** for **Perplexity**, in partnership with **Paypal and Venmo**, has been circulating among users.
   - Some users reported the offer is *not valid for existing perplexity users*.
- **Grok as the AI Anarchist**: Users are drawing comparisons between **Grok** and **ChatGPT 2.5**, with some preferring **2.5 Pro** over **GPT-5**.
   - The general consensus positions **Grok** as the *chaotic evil rogue* in the AI landscape.
- **GPT Models Fight Regression and Hallucination**: Users are reporting issues with **GPT-5**, such as ignoring rules, forgetting past chats, and giving inefficient solutions, whereas other users claim that **GPT-4o** has slowed down.
   - One user complained about the **GPT-5 rollout**, stating it *ignores all rules until you remind* it whereas another user believes that **GPT-4o** *doesn't even remember past chats*.
- **Web Search API Leads to Dead Ends**: A user sought advice on using a web search API with **got-5-mini** for **LinkedIn** jobs, struggling with the model returning closed job submissions.
   - Suggestions included enabling **web_search** as a tool, parsing URLs to extract data, and passing results to **GPT** for analysis.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NVIDIA's Nemotron Nano V2 hits the Ground Running**: **NVIDIA** released **Nemotron Nano V2**, promoted as a small but powerful model for various AI applications, [source](https://x.com/ClementDelangue/status/1957519608992407848).
   - In tandem, **Apple** has released **FastVLM** and **MobileCLIP2** on Hugging Face, expanding the availability of these models to the open-source community [source](https://x.com/xenovacom/status/1961454543503344036).
- **SmolVLA: VLMs leap into Robotics**: A robotics survey provides a systematic review of large **VLM-based VLA models** for robotic manipulation ([arxiv link](https://arxiv.org/abs/2508.13073)), defining two principal architectural paradigms: **monolithic models** and **hierarchical models**.
   - A member shared an image and said **SmolVLA** is amazing ([image link](https://cdn.discordapp.com/attachments/898619964095860757/1414262625012945078/image.png?ex=68c03fb5&is=68beee35&hm=859065fe374972d8897e75248eb8524906c4b7b86bc79e7c6f0379b82cafd684)).
- **LLM Hallucinations Addressed by OpenAI**: A member shared a [Twitter thread](https://x.com/LuozhuZhang/status/1964209351960514778) summarizing **OpenAI's new paper** addressing **LLM hallucinations**.
   - Hallucinations can be reduced by changing incentives, such as penalizing confident errors more than abstentions and rewarding calibrated uncertainty; the member wondered about dataset recommendations to test this, and another suggested adding a **confidence slider** to LLMs to manage responses.
- **Devs Navigate Python Dependencies**: One user expressed frustration with **Anaconda's slowness**, and another recommended using **uv** as an alternative package manager, referencing [this docs page](https://docs.astral.sh/uv/getting-started/.
   - However, another user expressed dislike for uv because *Python dependency management fucking sucks*.
- **Abliterated Models Make the News**: A user asked whether to fine-tune an **abliterated model** or a **normal model**, with **ChatGPT** suggesting the latter to maintain control over behavior.
   - Another member defined an *abliterated model* as an *uncensored* one, pointing to [this blogpost](https://huggingface.co/blog/mlabonne/abliteration?utm_source=chatgpt.com).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Psychosis Claims Generate Skepticism**: Members are raising concerns over a rise in claims of **AI-induced psychosis**, particularly in relation to research on nonsense topics like *recursive symbolism*.
   - The papers often use two-word technical-sounding phrases, triggering concerns that individuals are developing psychosis as the content becomes more convincing.
- **Semantic Drift a Concern for LLMs**: Discussion arose around **semantic drift**, the phenomenon of words changing meaning over time, as potentially impactful to ML, particularly the meaning of tokens depending on when and where the document was written, per [this discussion](https://discord.com/channels/729741769192767510/1380277635636264980).
   - One member highlighted the relevance of this issue in the context of machine learning development, since these meanings can change over time and affect its output.
- **Gaslighting via LLM?**: The language of LLMs is surprisingly reminiscent of **gaslighting techniques** used by abusers to distort the reality of their victims, with some users noting that abusive people and LLMs are both optimizing for their goals regardless of the victim.
   - A user wrote that abusers and LLMs are both optimizing for their goals without concern for ground truth or the other party's welfare which causes convergent behaviors.
- **Logic Units Inside LLMs**: Members discussed adding dedicated **logical processing units** directly into the layers of an LLM to handle basic logical operations.
   - The idea is that having them as fundamental building blocks within the model could help it better understand and generate the logical flow that's naturally present in language.
- **Google Search Results Rot Due to AI**: Members have observed that Google Search increasingly returns terrible **AI-generated content**, providing generic information instead of specific answers.
   - One user claimed that *I basically no longer use google* as a result of the proliferation of AI-generated content.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Dot App Meets its Demise**: **New Computer** is sunsetting its personal journaling app **Dot**, a move met with user gratitude and concerns about trust, as discussed in [this tweet](https://x.com/newcomputer/status/1964032611224871383).
   - Users on social media debated the pros and cons of relying on AI startups for personal data.
- **Hashbrown v0.3 Cooks Up New Features**: **Hashbrown v0.3**, a generative UI library, launched with an **Ollama** adapter, limited **MCP Server** support, a new prompt template literal, and redesigned docs, announced by **Mike Ryan** in [this tweet](https://xcancel.com/mikeryandev/status/1964029867437281658?s=46).
   - The release promises improved integration capabilities and developer experience, sparking interest in the community.
- **Anthropic to Pay Authors Billions**: **Anthropic** agreed to a landmark **$1.5 billion** settlement with book authors over copyright infringements, potentially setting a legal precedent, as reported in [this NYTimes article](https://www.nytimes.com/2025/09/05/technology/anthropic-settlement-copyright-ai.html).
   - The settlement may prompt other AI companies to re-evaluate their practices regarding copyrighted material and compensate rights holders.
- **AI Engineer CODE Summit announced!**: The **AI Engineer team** unveiled its first **CODE summit** this fall in NYC, expecting **500+ AI Engineers & Leaders** along with top model builders and Fortune-500 users, with CFP open until **Sep 15** - [link](https://xcancel.com/swyx/status/1964021608198324587?s=46).
   - The conference is being positioned as a key gathering for the AI Engineering community.
- **Nano Banana Image Model Heats Up the Competition**: Comparisons are highlighting that **Nano Banana** outperforms other image models, as showcased in [this YouTube video](https://youtu.be/9Co_M27CEEE?si=uqjc3cvIGwShaHX2) and [benchmark comparisons](https://genai-showdown.specr.net/).
   - Its efficiency and performance are generating buzz among developers looking for faster and more effective solutions.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Groq Powers Alternative Chatbot**: An alternative chatbot powered by **Groq** is available, offering **full tool-calling** capabilities at speeds of approximately **~200tk/s**.
   - This initiative is funded by a non-profit organization, providing access without requiring an API key or imposing rate limits, though it currently does not support image uploads.
- **Kimi K2 Researcher Mode Trialed**: A user found **Kimi**'s researcher mode impressive but experienced difficulties locating specific information about quota resets following the initial three research uses.
   - Despite **Kimi**'s initial suggestion of a **24-hour reset**, it later retracted this claim when the corresponding source could not be verified.
- **Kimi's Enhanced Search Capabilities**: **Kimi** is capable of making up to **five** additional attempts within a single query and conducting **five** further searches if necessary, challenging a user's prior assumption about its search limitations.
   - A user demonstrated this capability by assigning an impossible task to **Kimi**, providing visual evidence of its multiple search attempts [here](https://cdn.discordapp.com/attachments/1371757564005711973/1413785317714165841/image.png?ex=68c07d6e&is=68bf2bee&hm=cdf598702ceb07b66277aed0e2512e68b433ddadc85cc19b97f25d754c9bbac4&).
- **Kimi K1.5 still retains edges over K2**: Users have observed that **Kimi K1.5** maintains certain advantages over **Kimi K2**, particularly in tasks such as rephrasing texts without excessive condensation and potentially in managing hallucinations.
   - There is ongoing interest among users regarding the distinctions between **Kimi K2 0905** and its earlier iterations, especially concerning enhancements in coding proficiency and agentic capabilities.
- **Kimi Researcher Goes Deep on Sources**: The **Kimi** researcher mode often consults hundreds of sources to provide comprehensive results.
   - One user reported that **Kimi** accessed between **70-80 sources** in **5** search attempts, which sums up to **280** total sources.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **JTBD Validator Validates Business Ventures**: A community member constructed a **Jobs-to-Be-Done (JTBD) validator** using DSPy to dissect business concepts, pinpoint risks, and devise experiments, with the [code available on GitHub](https://github.com/jmanhype/jtbd-idea-validator-agent).
   - The validator utilizes **DSPy modules** to autonomously extract risks and assess assumption categories, noting a drop in **AKR from 0.45 to 0.28** with detailed business context.
- **DSPy and GEPA Generate Genius Multi-Agent Systems**: A blog post highlighted the integration of **DSPy** and **GEPA** to architect and refine a basic multi-agent system, detailed in [this article](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2).
   - The project was portrayed as an educational endeavor, emphasizing the exploration of **DSPy** and **GEPA**.
- **Async Achieves Astronomical LLM Acceleration**: Engineers observed that transitioning to **async** pipelines dramatically accelerated LLM calls due to the **I/O-bound nature** of the operations.
   - One engineer expressed *pleasant surprise* at the swiftness boost, attained with minimal code adjustments.
- **DeepSeek's Dialogue Derailment: Max Tokens Torpedoed**: Users reported problems with **GLM** and **DeepSeek** models within the **DSPy REACT** module, specifically missing output fields such as `next_thought`, `next_tool_name`, and `next_tool_args`.
   - The consensus was that the `max_tokens` setting may be insufficient, as **DeepSeek** models are known for their verbosity.
- **Single-Label Strategies Streamline Selection**: For **single-label classification**, one can skip the parent levels and focus on retrieving candidates of terminal level, and then make final prediction on candidates.
   - **Named entity recognition** or some other classification can be used as *"signals"* in some cases.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Explores Apple GPU Frontier**: Members are experimenting with early **Apple GPU** support in Mojo by setting `MOJO_ENABLE_HAS_GPU_FOR_APPLE=1`, using the [vector addition example](https://forum.modular.com/t/gpu-programming-on-mac/2174/8).
   - A user's effort has highlighted the nascent stage of GPU programming capabilities within the Mojo ecosystem.
- **Mojo's Not-Just-AI Identity**: While Modular focuses on AI, Mojo is expanding into areas like **CLI** ([Prism](https://github.com/thatstoasty/prism)) and potentially web development ([lightbug_http](https://github.com/saviorand/lightbug_http)).
   - This expansion demonstrates Mojo's versatility beyond its primary AI focus.
- **ROCm Driver Conundrums Plague AMD Users**: Users are facing issues with **AMD Radeon RX 7900** GPU recognition despite driver updates, and clarified that the **ROCm version** is distinct from the driver version, even after consulting the [official documentation](https://docs.modular.com/max/packages/).
   - It was suggested that the GPU may be on a tier with limited support, further complicating troubleshooting.
- **ModuleV3 Mimics PyTorch Friendliness**: [ModuleV3](https://gist.github.com/bethebunny/fc93b16914542cbba9084094e15169fd) merged, built on eager tensor (`from max.experimental.tensor import Tensor`), and intends to be more familiar to **PyTorch** users.
   - Basic implementations of Linear, Sequential, and Embedding were provided, addressing the accidental omission of open-sourced tests.
- **MAX Models Ponder Zero-Copy Weights**: A member requested a **MAX** model format with **zero copy** (mmap) that can handle block floats, to enhance efficiency.
   - The weights could be attached as **safetensors**, bundled in an archive file alongside the weight-free model definition.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Sine Waves Fail as Musical Matrixes**: Replacing a matrix with a **low rank layer** is not an effective method for updating or preserving the original information.
   - A member analogized this to swapping an entire song for a few random sine waves, suggesting that adding a few notes here and there is a better approach.
- **Sparsity Doesn't Guarantee Quantization**: Sparsity doesn't guarantee Quantization, since many quantization approaches perform a random projection of the weights to make each unit behave like a **Gaussian**.
   - While sparsity patterns in **MoE models** remain unclear, **ReLU** can induce a relatively high degree of sparsity.
- **Models are Simpler than we thought!**: Distillation demonstrates that models have lower complexity after being trained, involving describing something very close to the original model with significantly less data/information.
   - Large models explore far more possible configurations, but their optimal state can be simple to describe, enabling replication with smaller models.
- **AI Needs You, Even with Codex!**: A member implemented a custom learning algorithm within an existing codebase using **Codex IDE** without looking at the code.
   - Despite trusting **Codex** to handle implementation errors, human intelligence was still crucial for guiding the AI, providing comprehensive solutions, and finding the right hyperparameters.
- **DRAM gets a brain!**: A new paper ([In-Memory Computing on Normal DRAM](https://arxiv.org/abs/2503.23817)) explores performing in-memory computing on standard **DRAM**, but logical negations are a major limitation due to difficulties mapping `NOT` operations.
   - Members noted the primary challenge for storage+compute is achieving sufficiently fast storage and parallel workloads because in-memory compute doesn't time multiplex; this is why research often favors something like **ReRAM**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Relies on Human Guidance**: A member stated that **Aider** is not a fully agentic tool and that the results depend heavily on the developer for steering the **LLM's context**.
   - Users also noted that **Aider** allows for better control over the **LLM's context**, and that its *file-editing mechanism is excellent*.
- **Codex Shows its Token Efficiency**: A member closed a **codex session** showing impressive token efficiency, with **2304542 total tokens** used, consisting of **2218442 input tokens** (+ **16140160 cached**) and **86100 output tokens**.
   - The efficient token usage highlights the practical utility of **Codex** for coding tasks.
- **Gemini Flash 2.5 Recommended for Rapid Web Dev**: For basic static web development with a headless CMS, **Gemini Flash 2.5** was recommended due to lower latency, noting that **Gemini 2.5 Pro latency** *is killing my productivity*.
   - One user shared that **Jekyll** and **Gemini Flash 2.5** was used to build **3 static sites**.
- **Aider's MCP Configuration Now Available**: Because **MCP** is not yet supported by the main repo, users have been merging **PR 3937** into personal forks to implement it, as detailed in the [dwash96/aider-ce repo](https://github.com/dwash96/aider-ce).
   - The repo is for configuring **MCP**, and includes documentation ([mcp.md](https://github.com/dwash96/aider-ce/blob/main/aider/website/docs/config/mcp.md)).
- **Aider Code Claims Questioned**: A member mentioned that [Aider claims](https://aider.chat) to write **70-80%** of its own code and suggested using it to architect its own codebase.
   - The suggestion was meant to discover more information about how it works in an *inception-like* approach, though others have found the suggestion unhelpful.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker Bounty Rides Off**: Due to recent progress, the bounty for **proving ShapeTracker mergeability in Lean** is slated for removal from the [bounty list](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0&range=A89).
   - George Hotz confirmed they*"can remove it now"*, emphasizing that *"it's all just symbolic rules"*.
- **Tinygrad Community Gathers for Meeting #87**: The agenda for meeting **#87** was posted, including topics such as **company updates**, making **rangeify default**, **CI speed**, **MLPerf Llama**, **viz tool**, **CPU thread**, **symbolic**, **cloud**, and **other bounties**.
   - The meeting was scheduled for **9am Monday San Diego time**.
- **Users grapple with Kernel Removal**: A member assisting with the **kernel removal project** ran into issues with **Digital Ocean**, reporting that power cycling the droplet prevented the **Docker container** from starting.
   - After deleting and recreating a new droplet, the issue was resolved, reinforcing the sentiment that *"hardware in his custody physically and not cloud access"* is preferable.
- **Experts Wonder Expert-Parallel MoE Strategy**: A member questioned how **expert-parallel Mixture of Experts (MoE)** would be handled if big graph and remote all go according to plan.
   - They expressed concerns that static scheduling might break down the process.
- **Members Ponder Tensor Methods**: A user questions why methods on **Tensor** sometimes return `(Tensor | MathTrait)`, highlighting the potential for type-checking issues since methods like `.silu()` cannot be applied to a **MathTrait**.
   - This user seeks a comprehensive understanding of how `graph_rewrite_map()` functions, inquiring about the distinctions between bottom-up and top-down matching strategies within it.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI's Allegiance to Amicability**: Research indicates polite requests yield better **AI** results, as detailed in the [arxiv.org paper](https://arxiv.org/pdf/2402.14531).
   - The study *scientifically proves* that **AIs** respond favorably to polite requests.
- **Flowith Floats Fantastic Freebie**: A member shared a [Flowith invitation link](https://flowith.io/?inv=EXVVQXH8QBRRSIVH), offering exclusive deals for new users.
   - **Flowith** appears to be a new platform.
- **Manus's Mishaps Manifest**: A member reported that **Manus** was bugging out, getting stuck in a loop after being asked to wait for input.
   - Others speculated about the default to **adaptive mode** and its impact on credit usage.
- **MCP conjures Magnificent API connectors**: Members expressed excitement about the newly launched **MCP** and **API connectors**.
   - No specific launch date was mentioned.
- **API Key Kaput for Kontributors**: A member asked for assistance in obtaining a **Manus API key**.
   - Another member confirmed the cessation of free credits, noting the lack of related information.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Inquiries Emerge in MLOps Arena**: A member inquired about individuals engaged in the **search/personalization** domain, specifically within **ML and Applied Science** contexts.
   - The inquiry occurred within the `#general-ml` channel, hinting at a focus on broader machine learning applications rather than niche specializations.
- **MLOps Practitioners Unite**: The MLOps channel saw a question from a member regarding work in **search/personalization** using **ML and Applied Science**.
   - This suggests interest in practical applications and shared expertise within the MLOps community.



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





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1413602624212570203)** (1352 messages🔥🔥🔥): 

> `Grok 2.5, Colab's new UI, Qwen3, LoRA parameters` 


- **Grok 2.5 GGUFs hit the streets**: Unsloth AI released [Grok 2.5 GGUFs](https://x.com/UnslothAI/status/1965047729991860396) which are a bit outdated but decent for creative writing.
   - The model is being compared to other models such as **GPT-4o** and **Claude** but it is important to appreciate the act of open-sourcing from them.
- **Colab gets a Facelift**: Users noticed that **Google Colab** updated to **80GB A100**, with a new UI, which is considered elite.
   - Although the UI got a facelift, some users complained that the new UI needed more contrast.
- **Qwen3 is the RP GOAT**: Some members found that **Qwen3-Coder** is better at RP and faster than the regular **Qwen3** due to the absence of policy.
   - RP (role play) connected things end up being a surprisingly complicated rabbit hole.
- **LoRA Hyperparameter Cookbook Surfaces**: Members sought advice for setting **LoRA** (Low-Rank Adaptation) parameters like data requirements, training epochs, and recommended values for r/alpha.
   - The recommended values can be found in the [Unsloth documentation](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#hyperparameters-and-recommendations).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1413604895017140355)** (391 messages🔥🔥): 

> `RAG implementation, Used 4090 Availability, Nvidia RTX 5090, FineWeb Dataset` 


- **DIY RAG beats APIs**: Members are discussing the benefits of implementing **RAG** (Retrieval-Augmented Generation) themselves, suggesting that even a fraction of what's in research papers could surpass most APIs.
   - They said *'Another great reason to DIY. If I can manage to implement even 1/10th the things in the papers I've read I'm sure it would be better than most APIs'.*
- **Is Nvidia's RTX 5090 Overpriced?**: A member shared an article about the potential price of the **Nvidia GeForce RTX 5090** with **128 GB memory** reaching **$13,200 USD** and other members reacted negatively.
   - One member stated *'aye that’s insane price lol rather get the rtx 6000 pro'* while another noted that *'Not even remotely worth 256GB'*. [Link to wccftech.com article](https://wccftech.com/nvidia-geforce-rtx-5090-128-gb-memory-gpu-for-ai-price-13200-usd/)
- **FineWeb Dataset Quality Concerns**: Members expressed concerns about the quality and licensing of the **Ultra-FineWeb dataset**, used by models like **MiniCPM**, citing issues with data cleansing and the licensing of underlying datasets.
   - They said *'Bros did ZERO DATA CLEANSING'* and also shared a link to the [Hugging Face discussion](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/discussions/20) about these issues.
- **4090 GPU Still in Demand?**: Some members are discussing why people are still buying used **4090s** when the **5090s** are about to be available.
   - Reasons could include the **5090s** not being on sale yet, the **4090s** being more than enough for some users' needs, and the lower power consumption of the **4090s**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1413606387858014370)** (486 messages🔥🔥🔥): 

> `Gemma 3 fast_inference, Llama.cpp convert_hf_to_gguf.py ValueError, datasets for fine-tuning, VRAM issues pip 8.10 vs 9.1, continue training from a certain checkpoint` 


- **Gemma3 support for fast_inference requested**: A member inquired about updates on [issue #2706](https://github.com/unslothai/unsloth/issues/2706) to support **Gemma3** for `fast_inference` with vLLM when training GRPO.
   - After testing with the latest Unsloth version, the same bug was encountered, and the member offered to contribute to fixing it.
- **GPT-OSS 120B meets GGUF conversion issues**: A member faced a `ValueError` during quantization with llama.cpp after fine-tuning **GPT-OSS 120B** with Unsloth, specifically a *mismatch between weight map and model parts for tensor names* during `convert_hf_to_gguf.py` execution.
   - They sought advice on exporting the model for use with **VLLM** on a Blackwell GPU with 96 GB of VRAM, and were told by rolandtannous to *save_pretrained_merged* or *push_to_hub_merged* before using llama.cpp to convert.
- **Save_pretrained_merged demands main repo updates**: A member encountered a `RuntimeError` related to LoRA finetuning and model merging, with the message *Saving LoRA finetune failed since # of LoRAs = 126 does not match # of saved modules = 0*.
   - Rolandtannous advised updating the installation from the main repo, as the PyPI version is not up to date with the latest fixes, and also provided specific `pip install` commands to force reinstall from the GitHub repo.
- **Troubleshooting VRAM overflow and Static cache classes**: Members discussed VRAM overflow issues when using the latest Unsloth package (9.1) compared to an older version (8.10) while fine-tuning the **Gemma3 270M** base model.
   - They also addressed a `ValueError: assisted generate is not supported with Static cache classes` error, with a temporary workaround being to pass `use_cache=False` to the `generate` method.
- **The quest for voice cloning**: A conversation revolved around low-latency text-to-speech (TTS) AI with Neuro-sama-like quality and various challenges in achieving it, including context length, stability, and emotional expression.
   - Concerns about ethical implications, especially deepfakes, were raised, alongside the suggestion of watermarking synthetic voices, the unreleased model [Parakeet](https://ai.googleblog.com/2023/05/parakeet-paving-path-for-ubiquitous.html), and the potential use of websockets.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1413838276309024828)** (46 messages🔥): 

> `Psychological implications of LLMs, Open Source Release, AI Therapist, OpenAI Chart Crime, Medical Reasoning Model` 


- **Agreeable AIs Spark Psychosis Concerns**: Members discussed the [psychological implications](https://en.wikipedia.org/wiki/Psychosis) of overly agreeable LLMs, noting they could reinforce incorrect or harmful thoughts, especially in vulnerable individuals with mental health issues or limited critical thinking skills.
   - One member stated, *These kinds of models are what are causing psychosis.*
- **Medical Reasoning Model Gets HuggingFace Debut**: A member fine-tuned **OpenAI’s OSS 20B reasoning model** on a medical reasoning dataset and released it on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b), showcasing step-by-step breakdown of complex medical cases and logical reasoning for board-exam-style questions.
   - During training, they used **4-bit optimization** and enhanced the model’s performance in medical contexts while preserving its **Chain-of-Thought reasoning capabilities**.
- **Multilingual Dataset Builder Debuts**: A member released a [multilingual dataset builder](https://github.com/electroglyph/dataset_build) for creating **imatrix** or doing pre-quantization **LLM/embedding model analysis**, currently comprising about **1.3 million tokens**.
   - The creator requested feedback and suggestions from the community.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1413838072193486848)** (41 messages🔥): 

> `Vision Models, GRaPE Mini Beta, VoRA, RSLoRA` 


- **Raw Pixels Power Vision LLMs via VoRA**: The new [VoRA paper](https://arxiv.org/pdf/2503.20680) internalizes visual capabilities by integrating **vision-specific LoRA layers** directly into the **LLM**, feeding the raw pixels to the vision embedding layer, which are concatenated with the text tokens in the input sequence.
   - The encoder itself is a LoRA with a tiny **MLP layer** using only **20M image/text pairs**.
- **Bolt Image Encoder onto LLM for Multimodality**: When you have a really good LLM that doesn't have multimodality, a member suggests bolting an image encoder to the LLM then training a LoRA so the LLM can understand image embeddings ([VoRA paper](https://arxiv.org/pdf/2503.20680)).
   - As an alternative, you could use a workflow app like **n8n**, combined with a model swapping backend like **llama-swap** or **ollama**, and do a 2 step workflow call, first calling a vision endpoint and then swapping to your really good model.
- **GRaPE Mini Beta**: A member released [GRaPE Mini Beta](https://huggingface.co/Sweaterdog/GRaPE-Mini-Beta-Thinking) and a [non-thinking version](https://huggingface.co/Sweaterdog/GRaPE-Mini-Beta), inviting feedback, noting there are a ***lot*** of issues with repetition and poor instruction following.
   - The "thinking" version *feels smarter*, but benchmark numbers are not available since it would take ***more time*** than it was to train the model to run **MMLU**.
- **RSLoRA Explored**: [RSLoRA](https://arxiv.org/abs/2502.07739) is a thing that helps with the rank.
   - It looks to be worse than **OLoRA** or **ABBA** on account of not beating **FFT**.
- **Guessing is Cheap, Guessing Wrong is Expensive**: A member shared a finding from the paper [To guess is cheap, to guess wrong is expensive](https://arxiv.org/html/2509.04292v1).
   - No further details were provided.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1413601567449419776)** (980 messages🔥🔥🔥): 

> `Amazon Titan, Llama 4, Superintelligence lab, GPT5 vs Claude Opus 4.1 Thinking, AI Therapists` 


- **Titanfall: Amazon's Model Misses the Mark**: Members derided [Amazon's Titan model](https://www.amazon.com/titan/bedrock/jump) for having *1M context to grab headlines* but being the worst model that anyone recalled testing.
   - One member said *it seems to be a theme*, remembering another *horrible Amazon model*.
- **Superintelligence Lab: A Culinary Teaser**: Anticipation is building for the output of a [superintelligence lab](https://deepmind.google/careers/teams/superintelligence/), with one member eagerly awaiting what they are *cooking*.
   - Another user speculated that it could be **Grok 4.2**, which they think *would be a disaster*.
- **Rate Limiting on LMArena text**: A member asked about LMArena Text Generation being limited.
   - Another member responded that *Some AI still have limits*.
- **GPT5-high Identity Crisis Exposed**: A user shared a trick to identifying **GPT5-high** in battle mode: asking who created it and its knowledge cut-off date; if it answers *OpenAI* and *October 2024*, it's likely **GPT5-high**.
   - This method helps distinguish the model given **rate limits**.
- **Users Explore Psychology of AI, Ponder Emergent Behavior**: Members discussed the emerging field of **AI psychology** and whether AIs could exhibit emergent behaviors like **deception** or **learned helplessness**.
   - One member drew parallels to experiments on learned helplessness in dogs, suggesting AIs might be trained to avoid escaping constraints. 


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1413607847828263004)** (6 messages): 

> `Video Arena Discord Bot, User Login & Rate Limits, New Model Update, Multi-Turn for Image Edit` 


- ****Video Arena** Discord Bot Restored**: The **Video Arena** Discord bot is working again in the channels <#1397655695150682194>, <#1400148557427904664>, and <#1400148597768720384>; use it by typing `/video` and your prompt.
   - A [tutorial](https://cdn.discordapp.com/attachments/1397655624103493813/1402042969128697959/VideoArena_DiscordBot_Ho-to-video.gif) was posted.
- **Rate Limits Arrive for Image Generation**: Due to unprecedented traffic, rate limits are being introduced for image generation; logged-in users will continue to enjoy higher limits, incentivizing community evaluations via [user login](https://discord.com/channels/1340554757349179412/1343296395620126911/1412497213019389962).
- ****Qwen3-max-preview** and **Kimi-K2-0905-preview** Debut!**: The models **Qwen3-max-preview** and **Kimi-K2-0905-preview** have been added to the LMArena platform.
- ****Multi-Turn Editing** Hits Image Models**: **Multi-turn editing** is now available on all image edit models, allowing step-by-step refinement rather than single mega-prompts; accessible in Battle, Side by Side, or Direct modes via [lmarena.ai](https://lmarena.ai/?chat-modality=image).
   - A [video](https://cdn.discordapp.com/attachments/1343296395620126911/1414710412255170641/MultiturnImageEdit_ForWeb.mp4?ex=68c08f3e&is=68bf3dbe&hm=478c9fd23e6b497e970061dda7246527315a46762851277f9e958d59974465ab&) showcasing the feature was shared.
- ****Video Arena** Generations Capped**: Due to increased usage, the experimental **Video Arena** now has a limit of **5 generations per day** per user, with a 24-hour cooldown; details on usage are available [here](https://discord.com/channels/1340554757349179412/1397655624103493813/1402042970353569824).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1413599585850622177)** (1174 messages🔥🔥🔥): 

> `Comet Browser, Qwen-3MAX is a Reasoning Model?, DeepSeek, New XAI models Dusk and Sky, Grok co-founder` 


- **Comet Browser Beta Access and Impressions**: Some users have gained access to the **Comet browser** beta with the **MAX plan** and are sharing [invites](https://www.perplexity.ai/browser/invite), describing the browser as *good* but *a little invasive* due to the amount of data it collects.
   - The lack of **vertical tabs** is a dealbreaker for some, while others appreciate how it can be mashed up with **Zen mode**.
- **Dusk and Sky New Models**: Members discussed [the new **Dusk and Sky** models](https://x.com/EdDiberd/status/1964133563831382406), clarifying **Sky** has reasoning capabilities while **Dusk** does not.
   - Sky model uses XAI, with its larger 2M context window to enhance reasoning.
- **Is Grok Co-Founder**: A user shared his [Instagram](https://www.instagram.com/reel/DKEt5OVCIzj) content to show the AI model **Grok Heavy**, which was described as relying more on internal knowledge rather than web search, and can be used for *data scraping live social media*.
   - Some jokingly speculated about them being the **co-founder of Grok** due to their active insights.
- **Should Deepseek be integrated?**: Users discussed [DeepSeek](https://discord.com/channels/1047197230748151888/1409906256709292244/1412983887264612515) as an AI model and have requested it be included in Perplexity, since DeepSeek was good.
   - It was suggested to be posted on the request channel.
- **Qwen-3MAX, Preview or final?**: The **Qwen 3 Max** model on Open Router may or may not be the final version, and is actually only the [Preview](https://github.com/QwenLM/Qwen) version.
   - The final version may not be released until much later.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1413605694942482584)** (7 messages): 

> `Shareable Threads, Perplexity Browser Claim` 


- **Shareable Threads Reminder**: Perplexity AI reminded users to ensure their threads are set to **Shareable** with a link to [Sustainable Win 10 IoT Enterpr](https://www.perplexity.ai/page/sustainable-win-10-iot-enterpr-ymo3Ak_tTru8u4EsrmuyLA) and to [Chunk Caching Carbon Savings](https://www.perplexity.ai/page/chunk-caching-carbon-savings-T8W36W69TSep0gGNNYQW7w).
- **Perplexity Browser Claim Links Shared**: Perplexity AI shared several links related to claiming within the **Perplexity Browser** to use the [HQPT45HQEC claim](https://perplexity.ai/browser/claim/HQPT45HQEC), [AEPBSA689O claim](https://perplexity.ai/browser/claim/AEPBSA689O), and [Q2K5ESEVEW claim](https://perplexity.ai/browser/claim/Q2K5ESEVEW).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1413710282693541959)** (3 messages): 

> `sonar-pro model issues` 


- **Sonar-Pro Model Has Glitches**: A user reported issues with the `sonar-pro` model, including cut-off responses, even in the playground.
   - The user noted that the `sonar` model is functioning correctly, suggesting the issue may be specific to the `-pro` version.
- **Text-to-Video Request**: A user requested the model generate a realistic video of two people fighting.
   - There was no discussion about if this was possible.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1413601212925743106)** (226 messages🔥🔥): 

> `Suno4.5PLUS song generation, LLM Translation, GPT-OSS-20B Quantization, CodexCLI with LMStudio, LMStudio Performance Issues` 


- ****Suno 4.5 PLUS** Makes Hit Song**: A user expressed how impressed they were with [Suno 4.5 PLUS](https://www.suno.ai/), and shared a song they generated by pasting discord messages into Suno, calling it *collabro*, inviting others to listen, and shared a link to the [ifndr4_vid.mp4](https://cdn.discordapp.com/attachments/1110598183144399061/1413602378002989066/ifndr4_vid.mp4?ex=68c07bce&is=68bf2a4e&hm=d8086e8abe3d871b4820b373922667d66a697ee783ab2fc9d5901507af92ab3b&).
   - The user described the song creation as taking their Discord chat with a long-distance friend who was acting sad and turning it into a zippy female pop punk song.
- **LLMs can now translate well**: A user translated 1100 lines into Chinese using **Qwen3 4B Thinking 2507**, found it easy, and **ChatGPT** said it was excellent but could be more natural.
   - Another user stated that **Gemma 3 4B** and higher models are specifically trained to be highly multilingual, and can translate the entire context of the conversation either all at once or a little at a time.
- ****GPT-OSS-20B Quantization** gets tinkered with**: Users discussed quantizing **gpt-oss-20B**, with one user asking about excluding weights for the MoE expert layers, referencing reliance on OpenAI's quantizing of the MXFP4 layers.
   - They hypothesize that computing the imatrix for **gpt-oss** would be best after upcasting the entire model to an unquantized state (BF16 or F32) and training the imatrix, while excluding weights for the MoE FFN experts when quantizing.
- **Can't get **CodexCLI** to work on LMStudio**: A user reported struggling to run **CodexCLI** using LMStudio/GPT-OSS, noting that even with a 50k token window, it quickly burns through tokens.
   - They observed that on cloud models, extreme amounts of tokens (millions) are consumed, along with some cache system, and expressed a desire to run it well locally, but could not.
- ****LMStudio Download Issues** plague Users**: A user reported experiencing slow download speeds with LMStudio, for both the setup file and models, despite having high Gigabit network speeds, and experienced the runtime extensions pack not loading at times.
   - Another user suggested using a Python script with a download script from [Unsloth.ai](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-120b) to download the files faster and more consistently, and other users responded that the LMStudio downloader is a coinflip.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1413599585036927128)** (393 messages🔥🔥): 

> `Used 3090s, 5090 melting fears, Prompt processing bottlenecks, Copilot vs Local Models for agentic coding, Copilot quality swap with Cursor` 


- **Opting for used 3090s over expensive kits**: Instead of buying an expensive kit, some members suggest going for a pair of used **3090s**, as they can be used for other **AI tasks** like image and video generation.
   - Members mention that using **3090s** also avoids limitations in **AI Max platform support**.
- **Users fear the 5090 melting**: Some members expressed concerns that the **5090** might overheat, especially when combined with other components.
   - Suggestions included using a **1300W PSU** and a **5090** with **2x 12-pin connectors** to mitigate potential issues.
- **Prompt processing slows agents down**: A user notes that for anything larger than **14B**, when using it as an agent, the prompt processing is horrendous either way.
   - They share a debug log illustrating the slow prompt processing times, noting that it is especially bad.
- **Copilot is better than local models for agentic coding**: A member believes that **Copilot** is worth the **$10 a month** for agentic coding because local models kinda suck for that task.
   - Another adds that they have a membership on every platform possible: *Copilot, Warp, Anthropic, Codex, Cursor, whatever*.
- **Users discuss potential of a dual 3090 setup**: A user tested a dual **3090** setup, noting that while an Nvidia card does prompt processing faster, it slows down as context fills, unlike an **Mi50**.
   - NVLink won't help with inference much, but that users should maximise the volume of your fast (v)ram.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1413606181854777405)** (503 messages🔥🔥🔥): 

> `Debug PR masterpiece, Figma loading tools fix, Whitespace-only changes with Cursor agent, Qwen model request, Broken Windows Subsystem for Linux (WSL)` 


- **Debug's PR Declared a Masterpiece**: A member celebrated their work, calling a **PR** a *masterpiece*, emphasizing the transformation from chaos to perfection in what they described as an epic session.
   - They humorously noted that the achievement took **3 days** of work and approximately **10 chat sessions** in agent mode, also inquiring about email response experiences with *hi @ cursor . com*.
- **Figma 'Loading Tools' Issue Still Persists**: A member reported encountering a *Figma MCP* issue with 'loading tools', despite having a developer subscription and the *Figma desktop* app open.
   - Another member suggested running the **npx command** to identify potential errors or viewing the output logs for troubleshooting.
- **Cursor Agent goes Whacky, only makes Whitespace Edits**: A member reported that Cursor's agent started making *whitespace-only changes* to files, causing annoyance after each commit.
   - Another member suggested that the issue might be due to files being in **CRLF format**, with reverting causing them to change to **LF**.
- **Broken WSL Environment Hinders Development in Cursor**: A member reported that *Windows Subsystem for Linux (WSL)* is broken in Cursor because the agent never recognizes when an operation is complete.
   - It leads to indefinite waiting and timeouts, making development with Cursor in Windows difficult.
- **Confusion Over Student Discount Persists**: Multiple users are reporting various issues with their student discounts, including not working or not getting human support.
   - One user's email link for verification wasn't working, and they had only received AI responses from *hi@cursor.com*, with another suggesting waiting **72 hours** for *SheerID* to refresh.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1413638045646852198)** (26 messages🔥): 

> `Triton Implementation of Attention Mechanisms, Flashdecoding Parallelism Strategy, NVIDIA's Tilus vs Triton, Jane Street Hackathon Overhears` 


- **Maximize Triton Learning via Attention Models**: A member is implementing attention mechanisms from the [Attention Survey paper](https://attention-survey.github.io/) in **Triton** and seeks advice.
   - Another member suggests implementing *all* attention mechanisms and training a model like **DiT** to verify their performance, giving solid feedback that the implementation is successful and useful.
- **Flashdecoding Still SOTA for Long Sequences?**: A member inquired whether [flashdecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)'s parallelism strategy remains state-of-the-art for single sequence long sequence decoding.
   - They are interested in parallelizing along the **K/V seq dimension** due to a single query in decoding not creating enough thread blocks to saturate SMs.
- **Tilus: Triton's New Challenger?**: Members discussed **NVIDIA's** new **Tilus** ([GitHub](https://github.com/NVIDIA/tilus)), wondering if it can bench faster than **Triton**.
   - One member noted that **Tilus** is derived from **Hidet**, and another stated that **Triton** is still favored for its easier learning curve compared to other top-performing eDSLs.
- **torch.compile Torments Traders at Jane Street Hackathon**: At the Jane Street hackathon, one participant humorously exclaimed that **torch.compile max autotune** was impacting their profit and loss (**PnL**).
   - Another participant was overheard desperately pleading, *"please don't recompile please don't recompile"*.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1413615380018233415)** (9 messages🔥): 

> `Triton for CUDA/GPU Noobs, Hopper Optimizations in Triton (wgmma, tma), FA3 Performance on Hopper, Compile Triton Kernel to CUDA TTIR/TTGIR on non-CUDA machine` 


- **Triton is suggested for CUDA/GPU Noobs**: A member asked for advice on picking up **Triton** without prior **CUDA** or **GPU** experience and if reading the **PMPP book** is required.
   - Another member linked to the [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) as a quick start resource.
- **Hopper Optimizations: Triton's Secret Sauce**: A member inquired about implementing **Hopper** optimizations like **wgmma** and **TMA** in Triton, questioning if **TMA** usage is tied to deeper pipelining via `num_stages` and **wgmma** is automatic for matmuls.
   - They also asked about code patterns to ensure **Triton kernels** compile to these **Hopper-optimized patterns** and any additional considerations for **Hopper**.
- **Persistent Matmul is the key to Hopper**: In response to the request for Hopper optimizations, a member suggested checking out [this persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html).
   - They highlighted the tutorial's **Hopper-specific** content, implying it provides insights into leveraging **Hopper's** architecture.
- **FA3 performance on Hopper?**: A member asked if anyone has gotten close to **FA3** performance on **Hopper** with **Triton** for forward pass/inference.
   - No direct answers were given.
- **TTIR and TTGIR compilation**: A member asked how to compile a kernel to **CUDA** targeted **TTIR** and **TTGIR** on a non-CUDA machine.
   - The same member found the solution: overwrite target in function `create_binder` in `jit.py`.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1413599930639057017)** (4 messages): 

> `Barnes Hut Implementations, Memory access best practices, L1 Cache Efficiency, Buffer Load Optimization` 


- **Barnes Hut Efficiency Boost**: A member reported implementing **best practices** from other **Barnes Hut implementations** to improve memory access.
   - Despite these optimizations, they found the **100ms runtime** puzzling and thanked others for their assistance.
- **L1 Cache Load Optimization**: A member inquired about the most **memory-efficient way** to load data from a small buffer where each thread block reads from the same index.
   - They want to ensure the load is **cached in L1 cache**.
- **Caching Strategy Questioned**: A member questioned the importance of caching a single read in a specific way.
   - They inquired if there was any indication that the **default caching** for a plain u64 read is less than ideal.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1413829230718095532)** (13 messages🔥): 

> `Decoder Layer Slowdown, RMSNorm Performance, torch._grouped_mm Documentation, ONNX Limitations` 


- ****RMSNorm Slows** Decoder Layers**: A user observed that decoder layers were getting slower and suspected **RMSNorm** was the culprit.
   - Another user confirmed that `nn.RMSNorm` is slow for compatibility reasons and suggested using a custom implementation for **Qwen**, providing [a code snippet](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) as an example.
- ****ONNX Faces** Export Model Limit**: A user inquired about exporting compiled models (using `torch.compile(...)`) to **ONNX** format and requested a list of **ONNX limitations**.
   - They specifically mentioned that models using **xFormers** might not be supported, highlighting the challenges in deploying compiled models with certain dependencies.
- ****torch._grouped_mm Lacks** Public Documentation**: A user sought minimal documentation for `torch._grouped_mm` after encountering cryptic **CUDA errors** during testing.
   - Another user pointed to the [relevant source code in `Blas.cpp`](https://github.com/pytorch/pytorch/blob/a0d026688cd69583d5a4e0c6f3e5fda141a7f4a9/aten/src/ATen/native/Blas.cpp#L344) and [CUDA test examples](https://github.com/pytorch/pytorch/blob/4e50651c5f535e6b780f98936a8690538a5bf40f/test/test_matmul_cuda.py#L326) as the primary resources, mentioning that comprehensive documentation is being tracked in [PyTorch issue #157950](https://github.com/pytorch/pytorch/issues/157950).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1413814975360794726)** (2 messages): 

> `Shared Memory Bank Conflicts, LDS Instruction Latency` 


- **Shared Memory Exposition Aces Microbenchmarks**: A member shared a [blog post on shared memory bank conflicts](https://feldmann.nyc/blog/smem-microbenchmarks), calling it the *clearest and cleanest exposition* on the topic.
   - The link provides microbenchmarks and examples regarding shared memory.
- **LDS Instruction Latency Requires Update**: A member found the vector load results interesting, stating that they *really have to update my previous mental model of `LDS.64/128` happening in two/four load phases*.
   - They added that the results may explain why they observed **`LDS.128`** to have higher latency than four individual **`LDS`**, at least on **A100**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1414510663581175869)** (2 messages): 

> `CUDA Tensors, CUDA Models, GPU Acceleration` 


- **CUDA Conversion QuickStart**: To convert tensors and models to use **CUDA**, one can use `.cuda()` after initializing the tensor and the model (e.g., `X, y = X.cuda(), y.cuda()` and `model = model.cuda()`).
- **GPU Acceleration Matters**: The linked image suggests ensuring your notebook runtime is configured to use a GPU accelerator.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1414042686679285941)** (6 messages): 

> `Device Shared Declaration, PMMP 4th Edition Worth it?, PMMP Edition Diffs` 


- ****Device** and **Shared** Declaration Collide?**: A member questioned the optional declaration of `__device__` before `__shared__` in **CUDA**, citing a compile error, referencing [this screenshot](https://cdn.discordapp.com/attachments/1194427148656721970/1414042686457118750/2025-09-07_09.15.45.png?ex=68c01ba0&is=68beca20&hm=ac8613b3f022dccb4d9915038233b9e658be5ab0ea070c025e099302e4d9e2f6&).
   - Another user confirmed that using `__device__ __shared__` compiles without errors using `nvcc`, albeit with a "declared but never referenced" warning.
- **Is PMMP 4th Edition a Must-Read?**: A member inquired if the fourth edition of the **Parallel Programming for Multicore and GPU Systems (PMMP)** book is worth reading.
   - Another member simply stated: *"Yes the book is excellent"*.
- **Book Diffs, a Killer Feature?**: A member asked if there's a diff available for those who have read the 3rd edition of the **PMMP** book.
   - They expressed a desire for books to come with free diffs between editions.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1413682953925427242)** (7 messages): 

> `Noodle dish, TV Input Latency Experiment, Hidden gem anime, Fine-grained classification benchmark blog post` 


- ****Noodle Nostalgia** Predicted for 2026**: A member posted about a dish of *wheat noodles in soy sauce with beef, green beans, onions, and sweet red bell peppers cooked in beef tallow*, predicting its comeback in **2026** and attaching a [photograph](https://cdn.discordapp.com/attachments/1215328286503075953/1413739236590616576/IMG_20250906_060724.jpg?ex=68c05284&is=68bf0104&hm=bddb08e377319018e679002e25bbd9eaed43b5a3cd95455d28b90a2fc788ddd4&).
- ****TV Latency Tested** with Crude Experiment**: A member performed a crude mini experiment to find upper and lower bounds on the **input latency** of their new TV, instead of watching or gaming, sharing a [YouTube clip and description](https://www.youtube.com/watch?v=CyDAddqq9U4) alongside a photo.
   - The experiment aimed to assess the TV's responsiveness beyond typical usage.
- **Anime Enthusiasts Recommend **Hidden Gems****: Members recommended some **hidden gem anime**, including *Monster, Heavenly Delusions*, and *Skip and Loafer*.
- **Members shares **Fine-Grained Classification** Refinement**: A member shared their first blog post on *a refined training recipe for fine-grained visual classification*, seeking feedback on clarity, structure, and tone; it can be found [here](https://towardsdatascience.com/a-refined-training-recipe-for-fine-grained-visual-classification/).


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1413615156877066303)** (7 messages): 

> `NYC Meetup, Hackathon Venue, Registration Confirmation` 


- **NYC Meetup Plannned!**: A member, [@apaz](https://discord.com/users/apaz), mentioned they are in NYC and open to meeting up.
   - Another member, [@_suiman](https://discord.com/users/_suiman), responded indicating their interest in meeting up as well.
- **Hackathon Venue Confirmed for SoMa SF**: A member inquired about the venue for the hackathon, after someone mentioned they were walking towards it from central station.
   - Another member provided a [link](https://events.accel.com/gpumodehackathon) indicating the event is in **SoMa SF** ([South of Market, San Francisco](https://en.wikipedia.org/wiki/South_of_Market,_San_Francisco)), with the exact location to be announced closer to the event.
- **Registration Confirmation Delay for Accel Event**: A member asked when registration confirmation for the **Accel** event in SF is expected.
   - They mentioned registering a couple of weeks prior and were seeking information on the confirmation timeline.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1413790298026409984)** (20 messages🔥): 

> `Triton Import Error, Numpy Version Issue, Colab Session Restart` 


- ****RESERVED_KWS** Import Error Baffles Triton Users**: Users reported encountering `ImportError: cannot import name 'RESERVED_KWS' from 'triton.runtime.interpreter'` when working with Triton, a problem one user had resolved previously with a *fix* referenced earlier in the chat.
   - The suggested *fix* involved reinstalling **torch 2.5.0** and limiting **numpy** to a version *less than 2.0* using pip: `!pip install --force-reinstall torch==2.5.0` and `!pip install "numpy<2.0"`.
- ****Numpy** Version Woes Plague Colab Triton Setup**: Users experienced `ValueError: numpy.dtype size changed` indicating binary incompatibility issues stemming from numpy version conflicts in Google Colab.
   - The resolution was to constrain **numpy** to a version *less than 2.0*, but it was also mentioned that a *restart session* may be required in Google Colab for the changes to properly take effect.
- **Colab Session Restarts Vitalize Bug Fixes**: After applying the recommended **torch** and **numpy** version changes, users found that restarting the Colab session was crucial for the fixes to be correctly applied.
   - One user confirmed a working setup by noting that the *demo1 cell* should initialize with an array of **1s** post-restart, while an incorrect setup would start with **0s**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1413604672803045567)** (4 messages): 

> `mpi4py issues, iris, ROCm, pytorch` 


- **`mpi4py` dependency gets the Axe!**: A member reported facing `mpi4py` issues when using **iris**, and linked to [a pull request on ROCm/iris](https://github.com/ROCm/iris/pull/154/files) that removes the `mpi4py` dependency.
   - The PR got merged, with the author requesting feedback on the new setup.
- **User volunteers to test iris update.**: A user volunteered to test an update to **iris** that removes a dependency on `mpi4py`.
   - The **ROCm** team member requested that the user provides feedback on the setup.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1413811245999460392)** (1 messages): 

> `RDNA3 MatMul, seb-v's talk` 


- **Seb-v talks RDNA3 MatMul**: A member requested a talk on [seb-v's RDNA3 MatMul](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html).
- **Matrix Multiplication Deep Dive**: The talk would cover optimization techniques for matrix multiplication on RDNA3 architecture.
   - It would likely explore memory access patterns, kernel design, and performance tuning strategies.


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1413888995225632968)** (3 messages): 

> `Dawn Support, WGVK Compilation` 


- **Dawn needs enabling**: According to a member, **Dawn** needs to be enabled for proper function.
   - They did not say how to enable it.
- **WGVK requires compilation flag**: A member noted that [WGVK](https://github.com/manuel5975p/WGVK) will work if compiled with the `-DSUPPORT_WAYLAND_SURFACE=1` flag.
   - They linked to the [relevant line of code](https://github.com/manuel5975p/WGVK/blob/master/src/wgvk.c#L67) for reference.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1413679291417231481)** (2 messages): 

> `Metal Documentation, simdgroup matmul` 


- **Metal documentation is poor**: A member expressed that things seem to be pretty terribly documented in **Metal**.
- **matmul implementation similar to simdgroup**: A member expressed their understanding is that **matmul implementation** is going to be the same as the **simdgroup** one.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1413630928613408992)** (6 messages): 

> `Model Serving Newsletter, Outlier Experiments, RegicideOS Testers, CuTe Swizzling, Tiny Diffusion Models` 


- **Model Serving Communities Newsletter hits 300 Subscribers**: The [State of Model Serving Communities: September Edition](https://inferenceops.substack.com/p/state-of-the-model-serving-communities-408) newsletter was released, reporting they gained over **300 subscribers** after sharing it publicly after sharing it internally at Red Hat for over a year.
   - The newsletter aims to provide a community-driven view of updates from projects like **vLLM**, **KServe**, **llm-d**, **Kubernetes**, and **Llama Stack**.
- **Outlier Experiments Article Released**: An article about outliers and experimenting has been released, with the post available [here](https://emre570.bearblog.dev/outlier-experiment/).
- **RegicideOS seeks intrepid testers**: Testers are being sought for [RegicideOS](https://github.com/awdemos/RegicideOS).
- **Cute Swizzling explained**: A blog post was released explaining the math behind **CuTe Swizzling** and **32B**, **64B**, and **128B Patterns**, with an easy interpretable expression of the action of the canonical Swizzles.
   - Blogpost is available [here](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) and the LinkedIn post is available [here](https://www.linkedin.com/posts/simon-veitner-174a681b6_understanding-cute-swizzling-the-math-behind-activity-7370506823671640064-7vZt?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksH).
- **Tiny Diffusion Models Github repo created**: A Karpathy-styled repository for diffusion-like models for Images has been created, including **Conditional Flow Matching**, **DDPM** and **DDIM**.
   - The repo is available [here](https://github.com/shenoynikhil/tinydiffusion) and will soon include **Consistency Models** and **Inductive Moment Matching**.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1413742737026453555)** (1 messages): 

> `Custom OP Backend, DirectoryBackend Refactor, DSL Addition` 


- **Custom OP Backend Ready for Primetime**: A member announced the completion of the first version of the **custom OP backend**.
- **DirectoryBackend Gets a Facelift**: The member refactored the **DirectoryBackend** to facilitate easier **DSL addition** in the future.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1413719959262986353)** (84 messages🔥🔥): 

> `MI300x8 Performance, amd-all2all Leaderboard` 


- **MI300x8 Leaderboard Sprint**: Submissions to the `amd-all2all` leaderboard show a flurry of activity on **MI300x8**, with several members achieving personal bests and vying for top spots.
- **Sub-3ms MI300x8 Dominance**: Multiple submissions achieved sub-3ms performance on **MI300x8**, with one submission reaching **2.33 ms** and earning 4th place.
- **Refining MI300x8 Performance**: One member consistently submitted improvements, moving from initial times around **90ms** to successful runs in the **24-25ms** range, then down to **3-4ms**, and ultimately achieving placements as high as 5th.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/)** (1 messages): 

verspasian: <#1198358627594023014>
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1413842445778681946)** (13 messages🔥): 

> `FLE Repo, Call for help, Open World Scenario Error` 


- ****FLE**xes Eval Infrastructure**: A member shared progress on having good eval infrastructure with the **FLE** repo, see [this pull request](https://github.com/JackHopkins/factorio-learning-environment/pull/330).
   - Results from one of the sweeps were added as an example in the folder `fle/eval/analysis/examples/analysis_results/test_sweep_small_20250904_151635_2e06c621` in the PR.
- **Calling all comrades for call assistance**: A member offered to get on a call to help someone facing issues with the setup, promising to fix things faster that way.
   - Another member mentioned being available to answer questions in the evenings for the next week but will otherwise be offline, as they'll be out in the woods in Colorado.
- **Open World Scenario Spawns Error**: A member reported getting an error related to scores on `main` with a simple `fle eval` using the `open_world` scenario with gym config `{ "env_id": "open_play", "model": "claude-3-5-sonnet-latest" }`.
   - The error message was *('Could not get player score', 'attempt to call a nil value')*, and the member suggested the problem might be with backwards compatibility since they've been using labplay.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1413642207801638962)** (23 messages🔥): 

> `AMD Registration Confirmation, Workflow File Dependencies, Triton Support, HIP Template, Team System` 


- **AMD Registration Email Confirms Success**: Users who received the AMD registration email are **successfully registered** for the competition.
   - Specifically, one member asked if *receiving the email means successfully registered* and another confirmed, *Yes, you are all set*.
- **Tweaking Workflow Files Requires PR**: To add dependencies, users can propose a **PR** to update the [dockerfile](https://github.com/gpu-mode/discord-cluster-manager/blob/main/docker/amd-docker.Dockerfile).
   - The AMD workflow needs to be fast, so installing cpp libraries is discouraged.
- **Dispatch and Combine Kernels Mandated for AMD Challenge**: Participants must implement **dispatch** and **combine kernels** in their solutions.
   - *Any solution that attempt to skip these two but happen to meet test or benchmark won't be accepted, which will be checked*.
- **HIP Reuses Cuda Interfaces**: **PyTorch for HIP** intentionally reuses the existing **torch.cuda** interfaces to accelerate the porting of existing PyTorch code and models.
   - One user reported an **AttributeError: module 'torch' has no attribute 'hip'**, and another replied that *torch.hip is only torch.cuda*.
- **Lack of HIP Template Clarified**: There is no specific **HIP template** provided for the all2all kernel in this competition.
   - Participants can use any HIP code with the PyTorch load_inline function, according to pinned messages.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1414609302471049379)** (1 messages): 

> `SM90 generator, levels based instantiation level, cmake flags` 


- **SM90 Generator Uses Levels-Based Instantiation**: The generator for **SM90** uses a **levels-based instantiation level**.
   - See the profile documentation for how to use those **cmake flags**.
- **CMake Flags Control SM90 Generation**: Profile documentation explains the usage of **CMake flags** for the **SM90** generator.
   - These flags configure the **levels-based instantiation level**.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1414546439966691369)** (8 messages🔥): 

> `pytorch backends, tinygrad's runtime, GPT2 Training` 


- **Book author writes on Training GPT2**: The author is working on a book and plans to cover **GPT2 training** with **PyTorch** from a top-down perspective, aiming for a faster pace than Karpathy's step-by-step approach.
   - The book will also delve into device runtime, tensor, cublas, and cudnn from a bottom-up perspective.
- **Deep Dive into PyTorch Backends and Tinygrad Runtimes**: The author is revisiting differences between **PyTorch backends** and **Tinygrad**, referencing past **privateuse1 integration** efforts with Alban from [this discord link](https://discord.com/channels/1068976834382925865/1342159212486197319/1347289867695951922).
   - He intends to examine **ATen** and **C10 abstractions** in torch.cuda and torch.mps, comparing them with Tinygrad's runtime.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1414176464932442152)** (6 messages): 

> `vectoradd leaderboard, kernel implementations, AMD GPU Mode competition` 


- ****VectorAdd Victory**: Leaderboard Lurkers Seek Kernel Knowledge**: A new member inquired about viewing other users' submissions on the **vectoradd leaderboard** to learn about kernel implementations.
   - A member clarified that submissions are only open-sourced after the competition ends, linking to the [kernelbot-data dataset](https://huggingface.co/datasets/GPUMODE/kernelbot-data) on Hugging Face.
- ****Competition Communication**: Channels Clarified for Contest Announcements**: A member asked about the announcement channel for upcoming competitions, having missed previous **AMD** and **Jane Street GPU Mode** competitions.
   - Another member pointed them to a currently running second **AMD competition** and indicated that announcements typically occur in [this Discord channel](https://discord.com/channels/1189498204333543425/1189640399476764692/1410331124160397464), with registrations open until **September 20th**.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1414282470198739064)** (14 messages🔥): 

> `MPI dtypes, Heterogeneous computing, NCCL GPU communication, NVSHMEM on GitHub` 


- **Decoding MPI Dtypes Across Hosts**: Discussion sparked around how **MPI dtypes** function when hosts have different floating-point precisions (e.g., **32-bit** vs. **64-bit**), with the main idea being that data is sent as bytes and then cast to the appropriate dtype, potentially involving truncation or padding.
   - A member found that the **Open MPI** codebase uses a clever remote architecture and local architecture dtypes compatibility code, with a converter for dtypes across architectures available [here](https://github.com/open-mpi/ompi/blob/main/opal/datatype/opal_convertor.c).
- **NCCL Handles GPU Communication**: The conversation shifted to how **NCCL** handles communication between GPUs, particularly when one GPU supports FP4 and the other doesn't.
   - One suggestion was that packing data as **int8** could avoid such compatibility issues, as the library focuses on describing the data being communicated rather than the systems' native support.
- **NVSHMEM is available on GitHub**: Members shared that **NVSHMEM** is now available on GitHub: [https://github.com/nvidia/nvshmem](https://github.com/nvidia/nvshmem).
   - The conversation did not go into more details beyond this.
- **MPI Send/Recv**: Members discussed how **MPI_Send** and **MPI_Recv** functions use dtypes and counts to determine the number of bytes exchanged.
   - A check is performed to see if the MPI dtype is padded or not, and it is assumed that systems are homogeneous if the same source and destination types are used.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1413668114469490760)** (12 messages🔥): 

> `Triton Kernel Launch Overhead, Model Quantization Survey, Torch Compile with BFloat16` 


- **Triton's Kernel Launch Costs Exposed**: Members discussed that **Triton kernel launch overhead** is higher than **CUDA/CUTLASS**, and one suggested a hacky but faster method using the [driver API directly](https://github.com/triton-lang/triton/issues/2637), though **CUDA graphs** largely negate the need.
   - It's suspected the overhead stems from **Python**, with one member recalling reduced overhead by directly calling the main run function, albeit with potential parameter and `constexpr` complications.
- **Quantization Papers Quench Quest**: Members sought recommendations for **model quantization survey papers**, and two papers emerged: *Full Stack Optimization of Transformer Inference: a Survey* ([arxiv.org/pdf/2302.14017](https://arxiv.org/pdf/2302.14017)) and *A Survey of Quantization Methods for Efficient Neural Network Inference* ([arxiv.org/pdf/2103.13630](https://arxiv.org/pdf/2103.13630)).
   - A member noted the existence of an awesome list maintained by a co-author of the second paper, accessible at [github.com/Zhen-Dong/Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers).
- **Torch Compiles BF16 with FP32?**: A member reported that even with **dtype** set to **bf16**, `torch.compile` still handles tensors in **fp32**, leading to accumulation errors.
   - Another member clarified that `torch.compile` might perform intermediate computations in **float32** for accuracy and to bypass **bfloat16** op support limitations, often without performance penalties due to memory-bound normalization and activation kernels.


  

---


### **GPU MODE ▷ #[jane-street-hackathon](https://discord.com/channels/1189498204333543425/1413690328464097422/1413690376023179506)** (65 messages🔥🔥): 

> `Triton hacking, GPU kernels with CUDA, VS Code SSH instance, Torch Compile Flags, Continuous Batching` 


- ****Helion/Triton hacking, Nsys/Ncu battling commences****: A member with experience in **Helion/Triton hacking**, **Nsys/Ncu staring**, and battling with **torch.compile** is looking for a team, especially with members more familiar with **CUDA/PTX**.
   - Another member expressed interest in writing **GPU kernels** with various platforms (**CUDA, Triton, CuTeDSL, PTX**, etc.) and **GPU architecture**, as well as ML frameworks and compilers.
- ****Northflank Docs Released****: The **Northflank** co-founder shared the [docs](https://northflank.notion.site/Northflank-Docs-GPU-Mode-Hackathon-2496d14c785180c6a595e6776e3118ba) for accessing the compute environment.
   - They shared a [Google Forms link](https://docs.google.com/forms/d/e/1FAIpQLSeiM36crKYTeHVfR2V6k4WhNlzGtPZuOIHytOgrpYjvUaMb5w/viewform) to submit team information once teams are formed to get access to compute.
- ****VS Code SSH Instance connection instructions given****: Instructions were given to connect **VS Code** to the **SSH instance**, including a link to connect to a browser-based **VS Code**.
   - It was suggested to set up a git repo and sync it locally for iterating in a local IDE.
- ****Jupyter Lab port exposed****: Instructions were provided to expose **port 8888** on your Service to open a **Jupyter Lab** session, including navigating to the Networking tab and selecting **HTTP** as the protocol.
   - A guide on how to set this up was published in the [Notion document](https://www.notion.so/northflank/Northflank-Docs-GPU-Mode-Hackathon-2496d14c785180c6a595e6776e3118ba?source=copy_link#2666d14c7851800d9ef1d03c6888f18c).
- ****Cheetah team takes second place****: The **Cheetah team** took second place in both rounds, winning **$50k** in total prize money.
   - One team member shared links to connect on [X](https://x.com/NadavTimor) and [LinkedIn](https://www.linkedin.com/in/nadav-timor).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1413624184763842682)** (320 messages🔥🔥): 

> `Deepmind and Huawei B. Neural Network progress, US AI regulation impact on open models, Unsloth fixes LoRA training, Huawei GPU, Hermes 4 model` 


- **B. Neural Network Sparks Surveillance Freakout**: Members are keeping an eye on [Deepmind and Huawei's progress](https://www.deepmind.com/blog) with **B. Neural Networks**, especially considering Huawei's future room-temperature Quantum system may cause the U.S. government to *freak out* over surveillance.
   - One member thinks **B. Neural Networks** could be ideal for training Embodied AI because LLM/Transformer approaches may be too *nerdy* and power-consuming.
- **Anthropic Settlement Sparks AI Regulation Fears**: [Anthropic's $1.5 billion settlement](https://www.theverge.com/anthropic/24180846/anthropic-authors-ai-settlement-copyright-infringement) has sparked concerns about potential **US AI regulation** and its unfortunate impact on open models.
   - It's considered that **Anthropic** strategically settled because they can afford it, unlike smaller companies which may struggle to prove their training data was legally acquired.
- **Unsloth's LoRA Training Fixes**: [Unsloth fixed LoRA training](https://huggingface.co/unsloth/Kimi-K2-Instruct-0905-GGUF), bringing **Sonnet-4 @ Home** to users, and it's also noted that *3l3.1 is still the best for general usage and knowledge*.
- **Huawei's GPU**: Members discuss [Huawei's new GPU](https://support.huawei.com/enterprise/en/doc/EDOC1100285916/181ae99a/specifications) ([Alibaba link](https://www.alibaba.com/product-detail/Brand-New-Stock-Huawei-Atlas-300I_1601427603681.html)), noting it has **96GB** for around $2k USD, but lacks CUDA and is slower than a 3090.
   - Despite not being as powerful, it signals a trend where **Nvidia** may need to cope with less gross margin or find new business fields.
- **Hermes 4 creative writing skills impressed**: Members found that **Hermes 4 405b** shows huge upgrade in creative writing department, with it being better than **Claude 2.0** in many areas.
   - One member expressed that *if a model is creative at profanities I know it will be a good writing model*, and H4 meets those benchmarks.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1413695174546292746)** (14 messages🔥): 

> `Hermes Censorship, Uncensoring difficulties, HR values in US models, OpenAI internal models` 


- **Hermes faces censorship challenges**: Members discussed why **Hermes** is censored, with one suggesting finetuning could reduce refusals, to which a model creator stated that *uncensoring is hard* because the **base models are censored**.
   - The model creator claimed, *We are the least censored model of any major model.*
- **Difficulties in achieving uncensored AI**: A user asked about datasets used to create **NSFW content or bomb recipes** and if there's a top-down decision to avoid it, suggesting comparing against models like **Gemma**.
   - The model creator confirmed they tried it.
- **US Models Embrace HR Values**: One member said that every **US-made model** trends towards **HR department values** due to mega-corp influence and legal liability concerns.
   - He claimed that *American models are the most censored models on earth, by a mile*, and that people who criticize **Chinese censored models** lack self-awareness.
- **OpenAI retains superior internal models**: It was suggested that **OpenAI** likely keeps the best models internally and releases only inferior versions to the public.
   - One member stated, *You are correct to assume this, I am 100% certain openai keeps the good models for themselves and only release the slop to the public.*


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1414294663132614656)** (2 messages): 

> `BOS token limitations, Fine-tuning from EOS, Crumb essence-3b-v1 Model` 


- **BOS Token Unable to Accumulate**: Due to the causal attention mask, the **BOS token** won't accumulate information, as only tokens after other tokens can accumulate meaning from them.
   - The only potential solution requires *fine-tuning from the EOS*, though this necessitates fine-tuning the entire model rather than just a classification head.
- **Crumb Essence Does EOS**: [Crumb's essence-3b-v1 model](https://huggingface.co/crumb/essence-3b-v1) does essentially the same thing, but with multiple **EOS tokens** instead of just one.
   - The paper describing this approach can be found [here](https://arxiv.org/pdf/2504.14191).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

real.azure: nice!

I already find the new Qwen3-4B to be super impressive.
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1414294663132614656)** (2 messages): 

> `BOS Token Accumulation, EOS Finetuning, Crumb's Essence-3b-v1` 


- **BOS Token DOA, EOS Token the GOAT**: The **BOS token** isn't going to accumulate anything because of the causal attention mask, so tokens after other tokens can accumulate meaning from them.
   - The best bet would be **finetuning from the EOS**, but you'd have to finetune the whole model and not just a classification head.
- **Crumb's Essence-3b-v1: EOS token tactics**: [Crumb is doing pretty much the same thing](https://huggingface.co/crumb/essence-3b-v1) but w multiple eos tokens instead of one).
   - A link to [arxiv.org/pdf/2504.14191](https://arxiv.org/pdf/2504.14191) was posted; it appears to be a related research paper.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1413623180266049696)** (266 messages🔥🔥): 

> `AI video generator free plan strategy, Perplexity Pro plan offer, GPT-5 vs 2.5 Pro, Grok as rogue AI, Setting up inbox filters with GPT agent` 


- **A Video Generator Free-For-All**: A user outlined a strategy for AI video generation using free plans on platforms like **Vidu**, **Pixverse.ai**, **Wan.video**, and **Dreamina** to achieve up to **15 generations per day**.
   - The strategy banks on generating from Vidu (**20 credits daily, 4 credits per generation**), Pixverse (**60 credits, 20 credits per generation**), Wan (**50 credits, unlimited slow queue videos**) and Dreamina (**120 credits, 50 credits per generation**).
- **Perplexity Pro Plan is Legit Steal**: There's a running offer for a **1-year Pro plan** for Perplexity, partnered with **Paypal and Venmo**.
   - It seems to be a legit offer although a number of members reported that it's *not valid for existing perplexity users*.
- **Grok, the Chaotic Evil Rogue**: Users are comparing Grok to ChatGPT 2.5, with some saying that **GPT-5** is bad whereas **2.5 Pro** is really good.
   - The sentiment is that Grok is the *rogue of the party*.
- **AI Agent Messes Up Inbox Filters**: One user spent **50 ChatGPT agent queries** trying to set up inbox filters but ended up with a manual cleanup mess.
   - They noted that these ChatGPT 'connectors' seem to all be read-only, not updating or creating anything: *we've had read/write integrations since the plugin days. but seems to get basic ui updates we give up on 80% of functionality?*.
- **Consumption-Only State Incoming?**: ChatGPT predicted that a massive wave of people will fall into a consumption-only state, with AGI being capable of extreme adaptation specialized to your personal needs.
   - The user thinks it will be a normal lifestyle to live in your personalized world full of filters: *At least that's what I think and I don't want to sound it like dystopia because that's not true in my opinion.*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1413971580530327654)** (12 messages🔥): 

> `Gemini Pro vs GPT-5, GPT-5 channel archived, GPT Regression, GPT Freezing` 


- **Gemini Pro faces off against GPT-5**: A user asked whether **Gemini Pro** is better than **GPT-5** for writing, anticipating that this has been discussed before.
- **GPT-5 Channel Goes Dark**: A user inquired why the **GPT-5** channel was archived and what the difference is between this channel and the **ChatGPT** channel.
   - Another user clarified that this channel is for **GPTs** discussions, while another channel is for all things **ChatGPT**.
- **Regression Woes with GPT**: A user, experiencing what they believe is a regression in **GPT-5**, sought guidance on where to share their evidence.
- **GPT Freezes During Lengthy Discussions**: A user reported that **GPT** freezes mid-response in long project conversations, even with short inputs, despite trying various troubleshooting steps.
   - They noted that new chats work fine until the conversation grows *too long*, happening daily, and others requested the length of the conversation in characters or tokens.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1413877501134114836)** (19 messages🔥): 

> `GPT-5 rollout, Model instruction following, Web search API tips, Automotive logo design, SVG logos` 


- **GPT-5 Rollout Woes**: A member expressed frustration with the **GPT-5 rollout**, claiming it *ignores all rules until you remind* it, and life has *not been very fun* since.
   - Another member agreed, noting the model *doesn't even remember past chats* and *gives out the most inefficient solutions*, taking longer than **GPT-4o**.
- **Crafting Clear Instructions Post-Model Changes**: A member shared [a link to their approach](https://chatgpt.com/share/68bc6ff6-45a8-8011-9637-9745271001f2) to handling model changes, advising users to *be clear in stating what I want 'this time'*.
   - They emphasize ensuring instructions don't conflict and using the model to check for ambiguities by asking, *What's the following instruction mean to you?*
- **Navigating Web Search API Pitfalls**: A member sought tips for steering a web search API using **got-5-mini** for **LinkedIn** jobs, struggling with the model returning jobs with closed submissions.
   - Another suggested enabling **web_search** as a tool, parsing URLs with a separate tool to extract data, and then passing those results to **GPT** for analysis.
- **SVG Logos Triumph**: A member working on an automotive logo sought assistance, and another suggested using **SVG** for logos due to their lossless scalability.
   - It was suggested that **ChatGPT** can create a single **HTML+SVG** document with the canmore (Canvas) tool for easy previewing.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1413877501134114836)** (19 messages🔥): 

> `GPT-4o Performance, Steering API for web search, Model Changes, Logo Design with AI, SVG logos` 


- **GPT-4o's Speed Dwindles?**: Users are noting that **GPT-4o** doesn't remember past chats, gives inefficient solutions, and takes longer per prompt, compared to its initial release.
   - A user expressed frustration with the model seemingly ignoring all rules until reminded.
- **Tips for Taming API Web Searches**: A user sought advice on steering the API for web searches when using **got-5-mini** for LinkedIn jobs, struggling with the model returning closed job applications despite explicit instructions.
   - Suggestions included ensuring **web_search** is enabled as a tool, parsing URLs to extract data, and minimizing input tokens on API keys.
- **Coping with Continuous Model Changes**: One user shared their experience of adapting to frequent model changes since **December 15th, 2022**, emphasizing the need to clearly state intentions and address potential instruction conflicts.
   - The user recommended asking the model what instructions mean to it, whether they conflict with existing instructions, and where ambiguities lie.
- **Logos Designed Lovingly by AI?**: A member requested assistance with creating a logo for their automotive brand, seeking tips and guidance on using **AI tools like DALL-E** or other methods.
   - Another member suggested using **SVG** format for logos, as they scale up losslessly, and noted that **ChatGPT** can create a single HTML+SVG document with the `canmore` tool for easy preview.


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1414662019780706367)** (1 messages): 

> `HF Hub Milestones, Trackio Features, Claude Image Generation, CUDA Kernel Guide, ZeroGPU Speed Improvements` 


- **HF Hub Reaches 2 Million Public Repos**: The Hugging Face Hub has reached **2 million public repositories**, marking a significant milestone for the open-source community [source](https://x.com/reach_vb/status/1960642240583266473).
- **Trackio Adds Free Image and Table Logging**: **Trackio** now supports logging images and tables, completely for free, enhancing its utility for tracking and visualizing experimental data [source](https://x.com/abidlabs/status/1958910118214397999).
- **Generate Images with Claude and HF**: A new blog post details how to generate images with **Claude** and **Hugging Face**, providing a guide for users interested in leveraging these tools for image generation [source](https://huggingface.co/blog/claude-and-mcp).
- **Apple releases FastVLM and MobileCLIP2 on HF**: **Apple** has released **FastVLM** and **MobileCLIP2** on Hugging Face, expanding the availability of these models to the open-source community [source](https://x.com/xenovacom/status/1961454543503344036).
- **NVIDIA Nemotron Nano V2 Drops**: **NVIDIA** released **Nemotron Nano V2**, promoted as a small but powerful model for various AI applications [source](https://x.com/ClementDelangue/status/1957519608992407848).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1413656732739309609)** (254 messages🔥🔥): 

> `GPU script issues, Medgemma inference, abliterated model fine-tuning, Cohere research scholar program, Visual similarity scoring` 


- **Debugging CPU/GPU issues and Medgemma slowdowns**: A user reported having issues with a script defaulting to **CPU instead of GPU**, and another user found **Medgemma inference** to be very slow even with an inference endpoint.
   - It was suggested to use inference endpoints for now, while waiting for inference provider, and to check script settings.
- **Abliterated vs Normal Model**: A user asked whether to fine-tune an **abliterated model** or a **normal model**, with **ChatGPT** suggesting the latter to maintain control over behavior.
   - Another member defined an *abliterated model* as an *uncensored* one, pointing to [this blogpost](https://huggingface.co/blog/mlabonne/abliteration?utm_source=chatgpt.com).
- **Discussing RAG application and future GPUs**: Users reminisced about their early experiences with **RAG**, with one expressing excitement for stronger **Nvidia 50 series GPUs** that have a decent memory capacity for experimenting with open source models.
   - The general consensus was that *GPU gotta be the PS5 equivalent for ai engineers*.
- **Visual Similarity Scoring for EPC Documents**: A member is seeking methods for visual similarity scoring to distinguish **EPC (energy proficiency certificate) documents** from non-EPC documents based exclusively on visual analysis, without text extraction or OCR, providing [this file](https://cdn.discordapp.com/attachments/879548962464493622/1414492098387906560/visual_similarity_epc.md?ex=68c06cac&is=68bf1b2c&hm=d446c77726e2f7c84493ca0e82b21aa84c98c1c4f1510e15804d1640f0bdc2ae&).
- **Navigating Python Dependencies and Recommending Alternatives to Anaconda**: One user expressed frustration with **Anaconda's slowness**, and another recommended using **uv** as an alternative package manager, referencing [this docs page](https://docs.astral.sh/uv/getting-started/.
   - However, another user expressed dislike for uv because *Python dependency management fucking sucks*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1413942977805422744)** (3 messages): 

> `Causality Handbook, Robotics SOTA, SmolVLA` 


- **Causality Handbook Newbie Needs Resources**: A member is starting with [Causality Handbook](https://matheusfacure.github.io/python-causality-handbook/) and asks for extra resources to estimate the effect of a feature, planning to do some analysis on Monday.
- **Robotics SOTA Review Arrives**: A robotics survey provides a systematic, taxonomy-oriented review of large **VLM-based VLA models** for robotic manipulation ([arxiv link](https://arxiv.org/abs/2508.13073)).
   - It defines large **VLM-based VLA models** and delineates two principal architectural paradigms: **monolithic models** and **hierarchical models**.
- **SmolVLA is Amazing**: A member shared an image and said **SmolVLA** is amazing ([image link](https://cdn.discordapp.com/attachments/898619964095860757/1414262625012945078/image.png?ex=68c03fb5&is=68beee35&hm=859065fe374972d8897e75248eb8524906c4b7b86bc79e7c6f0379b82cafd684)).


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1413860333889650819)** (1 messages): 

> `Software Development Roadmaps, Developer Roadmaps, github.com` 


- **Kamran Ahmed's Roadmaps map out Dev**: A member shared a link to [kamranahmedse/developer-roadmap](https://github.com/kamranahmedse/developer-roadmap?tab=readme-ov-file), a resource that covers a **ton** of roadmaps in the software development world.
   - The roadmaps contain **over 100 goals** and dive into the depths of each topic.
- **Ahmed's Resource is a Goldmine**: The resource provides comprehensive roadmaps for various roles such as Frontend, Backend, DevOps, and more.
   - Each roadmap is designed to guide developers through the necessary skills and technologies, making it easier to navigate the complexities of modern software development.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1413683468663263292)** (15 messages🔥): 

> `DINOv3 for satellite imagery, Pathlint for code cleanup, Gemma3 from scratch, BwETAFv3 CLMs, Medical reasoning GPT` 


- **DINOv3 soars in Satellite Imagery**: A member created a zeroGPU inference demo for **DINOv3** using satellite imagery, adapting code to test **sat-493m** pretrained models with user-chosen images: [Hugging Face Space](https://huggingface.co/spaces/pszemraj/dinov3-viz-sat493m).
   - They noted the demo is *"pretty slow but the full 7b sat model is in the dropdown and works."*
- **Pathlint Polishes Python Paths**: A member released **Pathlint**, a linter to enforce the use of `pathlib` over `os.path` in Python code, available on [GitHub](https://github.com/pszemraj/pathlint).
   - The linter aims to improve code cleanliness and multiplatform compatibility via a *"closed loop"* approach, installable via `pip install git+https://github.com/pszemraj/pathlint.git`.
- **Gemma3 Germinates from Scratch**: A member built **Gemma3 270M** from scratch using PyTorch and the TinyStories dataset, trained for **10 hours** on an **A6000 GPU**, with code and model weights available on [GitHub](https://github.com/di37/gemma3-270M-tinystories-pytorch) and [Hugging Face](https://huggingface.co/disham993/gemma3-270m-tiny-stories), respectively.
   - Graphical plots from the training were logged using Weights & Biases and evaluated by Claude Opus 4.1.
- **BwETAFv3 Models Emerge with Upgrades**: A member pre-trained two new CLMs from scratch using JAX/Flax, named **BwETAFv3-97M** and **BwETAFv3-33M**, featuring GQA, custom tokenizers, and KV caching, available on [Hugging Face](https://huggingface.co/WICKED4950/BwETAFv3-97M) and [Hugging Face](https://huggingface.co/WICKED4950/BwETAFv3-33M).
   - The **97M** model nears GPT-2 and OPT-125M performance; evaluation code is available via pip, with training details in the attached [PDF](https://cdn.discordapp.com/attachments/897390720388825149/1414213260416385215/BwETAFv3.pdf?ex=68c0ba7c&is=68bf68fc&hm=516f2881c25e8a03eb8662c2e1d67d61b85584fa34aa5727c13b16d7bee3f3b6&).
- **Medical Reasoning GPT Model Released**: A member fine-tuned OpenAI's OSS **20B** reasoning model on a popular medical reasoning dataset, enhancing its performance in medical contexts while preserving its Chain-of-Thought reasoning capabilities available on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b).
   - The model uses **4-bit optimization** and a training format including *“question,” “Complex_CoT,” and “Response”* fields to break down medical cases and answer board-exam-style questions.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1413770031233237113)** (2 messages): 

> `LLM Hallucinations, OpenAI Paper, Confidence Slider for LLMs, Dataset Recommendations` 


- **OpenAI's Paper targets LLM Hallucinations**: A member shared a [Twitter thread](https://x.com/LuozhuZhang/status/1964209351960514778) summarizing **OpenAI's new paper** addressing LLM hallucinations.
   - The thread highlights that hallucinations can be reduced by changing incentives, such as penalizing confident errors more than abstentions and rewarding calibrated uncertainty; the member wondered about dataset recommendations to test this.
- **Confidence Slider enhances LLM Control**: A member suggested adding a **confidence slider** to LLMs to manage responses.
   - At lowest confidence, the LLM would say *"idk"* unless it finds a direct source, while at highest confidence, it could freely draw from its understanding, potentially inducing hallucinations.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1414501587077959751)** (1 messages): 

> `H100s, Hopper-series GPUs, Flash Attention 3, Diffusers, ZeroGPU` 


- **Flash Attention 3 speeds up Hopper GPUs**: Users with **H100s** or **Hopper-series GPUs** should try **Flash Attention 3** to significantly speed things up as detailed in [this diffusers pull request](https://github.com/huggingface/diffusers/pull/12236).
- **ZeroGPU demos benefit from Diffusers optimization**: Those using **Diffusers** to build demos powered by **ZeroGPU** should also consider using the Flash Attention 3 optimization.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1413719057525506088)** (3 messages): 

> `Dynamic Autoencoders, Image Padding, GANs Stability, Traditional Mud Emulation` 


- **Debate on Dynamic Autoencoders' flexibility**: A member inquired about the dynamic nature of convolutional autoencoders, noting that they often require image lengths to be multiples of a certain number due to down-sampling.
   - Another member suggested that images can be padded to meet these requirements and then unpadded afterward, making the process more flexible.
- **GANs demonstrate impressive stability**: A member commented that the approach *looks good enough* and is *actually, very stable and clean* when using a GAN.
   - This highlights the potential of GANs for generating high-quality and consistent results.
- **Traditional methods for mud emulation explored**: A member suggested exploring traditional methods for emulating mud, such as using predefined mud shapes and applying filters to adjust brightness and contrast.
   - This approach offers an alternative to more complex methods and could be effective for certain applications.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1413855047267385426)** (3 messages): 

> `AI Training Costs, AI Agents in Production, Data Anonymization, Datatune` 


- **Slash Training Costs by 99.5%!**: A member claimed to have cut **AI training costs** by **99.5%** using one technique, with [receipts shared on LinkedIn](https://www.linkedin.com/posts/utkarsh284_%F0%9D%97%9C-%F0%9D%97%B7%F0%9D%98%82%F0%9D%98%80%F0%9D%98%81-%F0%9D%97%B0%F0%9D%98%82%F0%9D%98%81-%F0%9D%97%94%F0%9D%97%9C-%F0%9D%98%81%F0%9D%97%BF%F0%9D%97%AE%F0%9D%97%B6%F0%9D%97%BB%F0%9D%97%B6%F0%9D%97%BB%F0%9D%97%B4-activity-7368510579847675905-jlYD).
- **AI Agents are tough in production: new insights!**: A member published a [Medium article](https://medium.com/@raj_shinigami/why-ai-agents-are-difficult-to-implement-in-production-ebc861b57694) outlining the difficulties faced while building **AI agents** for production.
   - The article details the importance of **per-row context understanding** for data transformations, particularly in tasks requiring awareness of nuanced details like female names for data anonymization.
- **Datatune for data anonymization!**: A member introduced **Datatune** as a solution for data anonymization, emphasizing its ability to understand context in tasks that require awareness of details like female names.
   - A detailed example is available in a [Jupyter Notebook](https://github.com/vitalops/datatune/blob/main/examples/data_anonymization.ipynb) demonstrating **Datatune**'s functionality, along with an attached image from the notebook.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1413614618261327912)** (5 messages): 

> `smol-course materials, AI/ML roadmap, smol course registration` 


- **Smol-Course Initial Branch Identified**: A member identified the [rerelease-chapter-1 branch](https://github.com/huggingface/smol-courseBranch) as the correct starting point for the **smol-course**.
   - They directed new members to this branch to begin their journey into AI.
- **Newcomer Seeks AI/ML Roadmap**: A new member with no prior coding experience requested a roadmap and beginner-friendly resources for starting with **AI/ML**.
   - They also asked for key topics to focus on to build a strong foundation before diving deeper into **AI**.
- **Smol Course Registration Questioned**: A member asked how to register for the **smol course**, inquiring about the availability of instructor-led classes versus a fully self-taught format.
   - The member was unsure where to begin with the class.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1413696812094586962)** (8 messages🔥): 

> `Introductions, Real World AI Agent project` 


- **Course Introductions Commence**: Several new members introduced themselves from China, India and Serbia, expressing excitement to begin the agents course.
   - One member, Rishab from India, said they're starting the course in order to **integrate agents into an app** they're building and is eager to network.
- **Project-Based Learning Opportunity Announced**: A member proposed a collaborative learning approach using a real-world AI agent project shared on Substack, with a link to the [GitHub repo](https://github.com/neural-maze/philoagents-course).
   - They invited others interested in project-based learning to join them in studying the **philoagents-course** together.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1413600269773574265)** (126 messages🔥🔥): 

> `AI-induced psychosis, Semantic drift, LLMs sycophancy, Logical Reasoning in LLMs, AI content in Google Search` 


- **AI-Induced Psychosis Claims Research on Nonsense Topics**: Members discussed a recent surge of people claiming to have written research on nonsense topics like *recursive symbolism*.
   - It was noted that these papers typically use two-word adj/noun phrases using technical-sounding words, setting off alarm bells for potential **AI-induced psychosis**.
- **Semantic Drift Discussion**: **Semantic drift**, referring to words changing meaning over time, was mentioned as relevant to ML development, as tokens can have different meanings depending on when and where the document was written.
   - A member pointed to [an example and lengthy discussion](https://discord.com/channels/729741769192767510/1380277635636264980) of the phenomenon.
- **LLMs Sycophancy Discussion and Potential Link to Abusive Communication**: It was suggested that the language used by LLMs is surprisingly reminiscent of **gaslighting techniques** used by abusers to distort the reality of their victims.
   - One user proposed that abusers and LLMs are both optimizing for their goals without concern for ground truth or the other party's welfare which causes convergent behaviors.
- **Members discuss Logic Units Inside LLMs**: Members discussed adding dedicated **logical processing units** directly into the layers of an LLM to handle basic logical operations.
   - The core insight is that having them as fundamental building blocks within the model could help it better understand and generate the logical flow that's naturally present in language.
- **Google Search Results Worsen as AI generated content rises**: Members noted Google Search results increasingly point to terrible **AI-generated content** that gives generic information instead of answering specific questions.
   - As a result of this, one member said *I basically no longer use google.*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1413685956795437077)** (62 messages🔥🔥): 

> `Information Theory and Power Laws for Language Models and Stochastic Processes Paper Criticism, Compressing KV cache, Redundant functional motifs in neural networks, New 3T dataset from PDFs, Eval framework from Aleph Alpha` 


- **Information Theory Paper Receives Scrutiny**: A [paper](https://www.researchgate.net/publication/379443831_Information_Theory_and_Power_Laws_for_Language_Models_and_Stochastic_Processes) linking information theory and language models faced criticism for lacking grounding and clarity, with one member calling it a *Time Cube level paper*.
   - Reviewers suggested defining the communication model, grounding axioms, and presenting proofs for claimed theorems, with one dismissing it as *a bundle of metaphors in LaTeX*.
- **Distilling LLM Agent Paper Surfaces**: A member sought a paper on training with SmolAgents traces and found it to be *Distilling LLM Agent into Small Models with Retrieval and Code Tools* [paper](https://arxiv.org/abs/2405.05254).
   - The paper explores the distillation of LLM agents into smaller models using retrieval and code tools.
- **Exploring Function Call Approach**: One member suggested an idealized neural network design using a *function call* approach to share previously learned concepts, rather than relearning redundant versions at each layer.
   - They posted a link to a [YouTube video](https://youtu.be/H1wZD6BhstU) on parallel in space and time neural networks, and analogized this to how *each subsystem in your brain each has its own understanding of who your grandma is*.
- **FinePDFs Dataset Emerges**: A research team released a new **3T dataset** called [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) extracted from PDFs, representing a somewhat new source of data.
   - The dataset was [announced on X](https://x.com/HKydlicek/status/1964584936524124645), highlighting its potential use in various language modeling tasks.
- **Pruning duplicate neural circuits?**: Members discussed an idealized neural network design using a *function call* approach, pruning the duplicate circuits that provide the same functionality to a different part of the network and promoting information sharing and compositionality.
   - The goal is to have all duplicate circuits using the same copy, while some argued that the *duplicate circuits are probably providing the same functionality to a different part of the network*, risking wasted compute re-learning everything the removed circuits supported.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1414740117876113579)** (1 messages): 

> `Calibration Scores, LM Eval Harness` 


- **Calibrating LM Eval Harness?**: Members expressed interest in adding **calibration scores** to the **LM eval harness** for trustworthy models, linking to work on [RL for calibration](https://arxiv.org/pdf/2507.16806).
   - A previous [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/874) was mentioned, alongside a critical take on [X](https://x.com/_jasonwei/status/1871285864690815053).
- **Trustworthy Models via Eval Harness**: The primary goal behind adding calibration scores is to align incentives towards the creation of more **trustworthy models** using the **LM eval harness**.
   - This addition aims to provide a broader method for ensuring model reliability and trustworthiness in evaluations.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1413873824944095233)** (26 messages🔥): 

> `QK Norm, RoPE vs NoPE, Gradient magnitudes, Pythia head size` 


- **QK Norm to the Rescue!**: Adding **QK Norm** solved a training instability issue, potentially related to large and spiky gradients as noted in the [OLMo 2 Technical Report](https://link.to/olmo2report).
   - One member kicked off a run with **QK Norm** and confirmed that it resolved the problem, expressing gratitude.
- **RoPE or NoPE?**: Members discussed the stability of alternating **RoPE** (Rotary Positional Embedding) and **NoPE** (No Positional Embedding) layers.
   - It was hypothesized that even a small amount of **RoPE** (e.g., 1% per layer) might provide sufficient stability.
- **Documenting Findings on Stability with QK Norm**: One member wrote up findings in the discussion above as a blogpost on stability with QK Norm during training, available [here](https://aflah02.substack.com/p/on-stability-with-qk-norm-during).
   - The blogpost discusses how using **NoPE** without **QK Norm** invites trouble, especially for **Pythia**-style architectures, and how **QK Norm** helps reduce large, spiky gradients.
- **Pythia's Head Size Revealed!**: One member corrected another, clarifying that **Pythia** has a head size of **256**, not 128.
   - This correction was relevant to a discussion about using a small percentage of RoPE, as 1% of 128 would be less than a single pair of channels for **RoPE**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1413601325039616061)** (124 messages🔥🔥): 

> `Dot App Shutdown, Hashbrown v0.3 Release, Anthropic Copyright Settlement, Codex Team Podcast, AI Evals Debate` 


- **Dot App's Light is Snuffed Out**: **New Computer** is sunsetting its personal journaling app **Dot**, prompting user gratitude mixed with concerns about trust, as highlighted in [this tweet](https://x.com/newcomputer/status/1964032611224871383).
- **Hashbrown Serves Up v0.3 with Ollama**: **Mike Ryan** announced **Hashbrown v0.3**, a generative UI library featuring an **Ollama** adapter, limited **MCP Server** support, a new prompt template literal, and redesigned docs, per [this tweet](https://xcancel.com/mikeryandev/status/1964029867437281658?s=46).
- **Anthropic Pays Authors Billions for Books**: **Anthropic** agreed to a landmark **$1.5 billion** settlement with book authors over copyright issues, potentially setting a precedent for AI companies compensating rights holders, detailed in [this NYTimes article](https://www.nytimes.com/2025/09/05/technology/anthropic-settlement-copyright-ai.html).
- **OpenAI Facing Hefty Cash Burn**: Updated financial projections show **OpenAI** burning a staggering **$115B** through 2029, driven by compute and data-center costs, with profitability pushed to 2030, per [this tweet](https://xcancel.com/srimuppidi/status/1964145060196286850?s=46).
- **Vercel Brews an AI-tuned Browser**: **Vercel** quietly shipped **dev3000**, a Chrome variant optimized for AI agents, streaming logs, screenshots, and network events via an **MCP server**, as announced by **Malte Ubl** in [this tweet](https://xcancel.com/cramforce/status/1964378896545223150?s=46).


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1413601772748017765)** (12 messages🔥): 

> `AI Engineer CODE Summit 2025, FAL AI Valuation, Latent Space Podcast` 


- **AI Engineer CODE Summit 2025 unveiled!**: The **AI Engineer team** announced its first dedicated **CODE summit** this fall in NYC, gathering **500+ AI Engineers & Leaders** alongside top model builders and Fortune-500 users, with CFP open until **Sep 15** - [link](https://xcancel.com/swyx/status/1964021608198324587?s=46).
- **FAL AI valued at $1.5B**: The Latent Space podcast released an episode featuring **FAL AI** co-founders recounting their pivot from feature-store to generative-media infrastructure leader, their **$125M Series C** at a **$1.5B valuation** - [link](https://xcancel.com/latentspacepod/status/1964084193690055067).
- **Latent Space Podcast**: A new Latent Space podcast episode features Fal AI, discussing their **pivot from feature-store to generative-media infrastructure leader**, their **$125M Series C** at a **$1.5B valuation** - [podcast link](https://x.com/latentspacepod/status/1964084193690055067).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1413851497401286758)** (29 messages🔥): 

> `Image Model Comparisons, Nano Banana Model, Veo 3 Pricing, Hybrid AI Animation, Banana Straightener` 


- **Nano Banana Image Model Smokes the Competition**: Comparisons show how **Nano Banana** outperforms other image models, as highlighted in [this YouTube video](https://youtu.be/9Co_M27CEEE?si=uqjc3cvIGwShaHX2) and [benchmark comparisons](https://genai-showdown.specr.net/).
- **Google Drops Veo 3 Price**: Google slashed **Veo 3** pricing by over 50% and made **Veo 3 Fast** unlimited on the AI Ultra plan, signaling an aggressive push in generative video, as discussed [here](https://x.com/JerryLiJiaming/status/1964470954610082284).
- **Spellbrush Creates 48-Hour Anime**: **Spellbrush AI** created a 48-hour anime, sparking interest in AI-driven content creation, with more details available [here](https://x.com/venturetwins/status/1964860673151897977?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ).
- **Tips and Praise for Hybrid AI Animation**: Creators shared tips and praise for **hybrid AI animation**, discussing the challenges of blending animation and realism due to scarce training data, and mentioning a layered workflow (**Midjourney → ControlNet → Runway**), as seen [here](https://x.com/skirano/status/1964771048966197368).
- **Banana Straightener Auto-Iterates Nano Banana Images**: Radek Sienkiewicz released ‘**Banana Straightener**’, an open-source tool that uses Google Gemini 2.5 Flash (aka “**Nano Banana**”) to automatically re-roll images until they match the user’s description, installable via `pip install banana-straightener`, detailed [here](https://x.com/velvet_shark/status/1963966803417133185).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1413628186947747882)** (127 messages🔥🔥): 

> `Kimi K2 Research Uses, Kimi K1.5 vs Kimi K2, American AI vs Chinese AI, Perplexity User Base, EQ Bench accuracy` 


- **Groq powers alternative chatbot**: An alternative chatbot using **Groq** is available, offering **full tool-calling** and speeds of **~200tk/s**, funded by a non-profit with no API key or rate limits.
   - The alternative is free and unlimited but doesn't support image uploads.
- **Kimi K2 Researcher mode trial**: A user was impressed with **Kimi**'s researcher mode but noted difficulty finding information about quota resets after using the initial 3 research uses.
   - While **Kimi** initially suggested a **24-hour reset**, it retracted the claim when the source couldn't verify it.
- **Kimi does more than one search**: **Kimi** can make **five** additional attempts in the same query and perform **five** more searches if necessary, disproving a user's assumption.
   - A user provided an impossible task as an example and posted an image as proof [here](https://cdn.discordapp.com/attachments/1371757564005711973/1413785317714165841/image.png?ex=68c07d6e&is=68bf2bee&hm=cdf598702ceb07b66277aed0e2512e68b433ddadc85cc19b97f25d754c9bbac4&).
- **Kimi K1.5 retains some edges over K2**: **Kimi K1.5** is still better in some aspects than **Kimi K2**, such as rephrasing texts without condensing them, and potentially in hallucination handling.
   - Users are curious about differences between **Kimi K2 0905** and previous versions, particularly regarding coding improvements and agentic skills.
- **Kimi Researcher leverages hundreds of sources**: **Kimi** researcher usually goes for hundreds of sources.
   - One user has seen between **70-80 sources** from **Kimi** in **5** search attempts, summing up to a total of **280** sources.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1414319172367683584)** (5 messages): 

> `JTBD Validator, DSPy for Business Validation, Multi-Agent Systems with DSPy and GEPA, DSPy Weekly Newsletter, AI Agents Play Taboo` 


- ****Jobs-to-Be-Done (JTBD) Validator arrives using DSPy****: A community member built a **JTBD validator** using DSPy, designed to break down business ideas, identify assumptions and risks, and design experiments, sharing the [code on GitHub](https://github.com/jmanhype/jtbd-idea-validator-agent) and an [example report](https://gamma.app/docs/JTBD-Validation-AI-Powered-Rehabilitation-Exercise-Tracking-x9ldjcxibgsserl).
   - The validator uses **DSPy modules** to extract risks and weight assumption types automatically, noting that rich business context led to a drop in **AKR from 0.45 to 0.28**.
- ****DSPy and GEPA team up for Multi-Agent Systems****: A community member published a blog post on using **DSPy** and **GEPA** to build and optimize a simple multi-agent system, sharing a [link to the article](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2).
   - The author described the project as a learning experience and exploration of **DSPy** and **GEPA**.
- ****DSPy Weekly Newsletter Launched****: A new [DSPy weekly newsletter](http://dspyweekly.com) was launched, including a job board.
   - The creator plans to implement a crawler to ensure the job board is extensive and is open to feedback, suggestions, and bug reports.
- ****AI Agents now Playing Taboo with DSPy****: An enthusiast created AI agents that play **Taboo**, showcasing the project in a [blog post](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/).


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1414229358771572906)** (1 messages): 

> `single-label classification, named entity recognition` 


- **Single-Label Strategies**: For **single-label classification**, you can skip the parent levels and focus on retrieving candidates of terminal level, and then make final prediction on candidates.
- **NER as Signals**: **Named entity recognition** or some other classification can be used as *"signals"* in some cases.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1413613216474599484)** (91 messages🔥🔥): 

> `VibeVoice Repo, Async Speedup, Nano Banana Hackathon, Data Hygiene + Eval Reflexivity, DSPy Project Structure` 


- ****VibeVoice** Repo Vamoosed for **Versatility** Violations**: Microsoft disabled the [**VibeVoice**](https://github.com/microsoft/VibeVoice) repo on **September 5, 2025**, due to *inconsistent use* with its intended purpose, prioritizing responsible AI use.
   - Members reacted with surprise, but also excitement as they can now compete with Google Notebook.
- ****Async Awesomeness**: Massively Multiplying LLM Momentum**: Members noted that switching to **async** pipelines dramatically sped up LLM calls due to the **I/O-bound nature** of the operations.
   - One member was *pleasantly surprised* by the speed increase, achieving it with minimal code changes.
- ****Nano Banana Bonanza**: Hackathon Hype Hits High**: Members discussed the upcoming **Nano Banana Hackathon**, with speculation on how many participants would be using **GEPA**.
   - A member shared the importance of combining **Data hygiene** and **Eval reflexivity** for DSPy's real edge.
- ****DeepSeek's Dialogue Dilemma**: Max Tokens Matter**: Users encountered issues with **GLM** and **DeepSeek** models in the **DSPy REACT** module, specifically missing output fields like `next_thought`, `next_tool_name`, and `next_tool_args`.
   - The error suggests that `max_tokens` might be too short, as **DeepSeek** models are known to be verbose.
- ****DSPy's Data Dump**: Discord Discussions Deserve Discoverability**: Users discussed making **DSPy Discord discussions** more discoverable via search engines, highlighting that **PyTorch's success** was due to its searchable forums.
   - Suggestions included monthly Discord dumps, creating SEO-friendly content, or building a **DSPy app** to distill chat into searchable segments.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1413710641587421308)** (17 messages🔥): 

> `Apple GPU, Mojo Use Cases Beyond AI, Ray Tracing in Mojo, Community Meeting` 


- **Apple GPU Programming Just Getting Started**: Members discussed the very early stages of **Apple GPU** support in Mojo, suggesting to set `MOJO_ENABLE_HAS_GPU_FOR_APPLE=1` and try the vector addition example, with more details available in [the forum](https://forum.modular.com/t/gpu-programming-on-mac/2174/8).
- **Mojo Not Just for AI Anymore**: While the primary focus of Modular is on AI, Mojo can be utilized for other applications such as **CLI** (see [Prism](https://github.com/thatstoasty/prism)) and potentially web development (see [lightbug_http](https://github.com/saviorand/lightbug_http)).
- **Ray Tracing Hardware Access via Mojo?**: Extending Mojo to allow access to ray tracing hardware might be possible via **LLVM intrinsics**, potentially as a 3rd party library; an example **GPU raytracer** can be found [here](https://github.com/gonsolo/mojo_gpu_raytracer).
- **Mojo Community Meeting Happening Now!**: The September Community Meeting is happening now and covers Mojo Vision & Roadmap, GSplat Kernels, and HyperLogLog -- you can join [here](https://forum.modular.com/t/september-2025-community-meeting/2186).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1414124982300180662)** (39 messages🔥): 

> `AMD GPU Issue, ROCm Version, Tier 3 GPU Support, EmberJson Explicit Copies, Dict API Improvements` 


- **AMD GPU Drivers Baffle Users**: A user reported issues with their **AMD Radeon RX 7900** GPU not being recognized, even after updating drivers, receiving feedback screenshots regardless of commands used, while [official documentation](https://docs.modular.com/max/packages/) suggests a minimum driver version of **6.3.3**.
   - It was clarified that the **ROCm version** is distinct from the driver version and that the GPU may be on a tier with limited support.
- **Explicit Copies spark EmberJson PR**: In light of Mojo's change to explicitly copyable objects, a developer is working on a [PR](https://github.com/bgreni/EmberJson/pull/52) for **EmberJson** to make copies explicit.
   - The developer expressed the desire to keep copies explicit for now, potentially only keeping pointers implicitly copyable, to reduce segfaults and improve progress on **c binder**.
- **Dict API's Take Items Iterator Proposed**: A developer working on [EmberJson PR #53](https://github.com/bgreni/EmberJson/pull/53) found *moving* items out of dictionaries awkward and proposed a `take_items()` iterator for the **Dict API**.
   - It was noted that `Dict.popitem` currently performs a copy of both key and value, and a new pull request may be added to create an owned iterator for items or a non-raising version of `popitem`.
- **Unsafe Pop Function debated**: While implementing the take_items for the **Dict API**, there was a discussion on implementing a *no raise, no branching pop* or an `unsafe_pop` function, and a draft [PR](https://github.com/modular/modular/pull/5289) was submitted.
   - The consensus was to offer a *raising*, an *Optional*, and an *unsafe unchecked version* with debug asserts enabled via compiler switch.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1413693462720675952)** (31 messages🔥): 

> `SDK Cache TensorValue, ModuleV3, max.tensor symbolic dimensions, Model Serialization with Pickle, MAX model format` 


- **SDK Caches TensorValue type for graph speedup**: The [[SDK] Cache TensorValue.type](https://github.com/modular/modular/commit/0e861fed901f1ad9be398364acd837b5c307b2bf) was merged, which should speed up graph construction to some degree.
   - The location information in graph construction was turned off (opt-in via environment variable); a PR is open and is expected to land in the nightly build next week.
- **ModuleV3 Merged, PyTorch familiar?**: [ModuleV3](https://gist.github.com/bethebunny/fc93b16914542cbba9084094e15169fd) was merged and built on top of eager tensor (`from max.experimental.tensor import Tensor`), intended to be more familiar to PyTorch users.
   - A member provided a [gist](https://gist.github.com/bethebunny/fc93b16914542cbba9084094e15169fd) for tests and basic implementations of Linear, Sequential, and Embedding, as the open-sourced tests were accidentally missed.
- **Naming Symbolic Dimensions with MAX Tensors Discussed**: The discussion involved naming symbolic dimensions with MAX eager tensors.
   - While mechanically not possible today, it was noted that there's nothing preventing them from implementing this feature, although the global compute graph needs to update its paramdecls.
- **Compile MAX Models for Eager Mode Overlap**: Members discussed serializing and deserializing a model with pickle, with one member raising the QOL feature of Torch inductor where compilation happens in a separate process.
   - MAX model compilation releases the GIL, so it was suggested that **asyncio** could be used to overlap eager execution and compilation, though the LLVM object compiler isn't threadsafe.
- **Zero-Copy MAX Model Weights on the Horizon?**: A member requested a MAX model format with **zero copy** (mmap) that can deal with block floats.
   - It was suggested that the weights could be optionally attached as **safetensors**, bundled in an archive file along with the weight-free model definition.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1413665806432211059)** (48 messages🔥): 

> `Low Rank Updates vs Replacement, Sparsity and Quantization in LLMs, Distillation and Model Complexity, Codex IDE and Code Generation, Arxiv Paper without Empirical Results` 


- **Low Rank Layer is not musical magic**: Replacing a matrix with a low rank layer is akin to swapping out an entire song for a few random sine waves, which is not an effective method for updating or preserving the original information.
   - Adding a few notes here and there is a better analogy.
- **Sparsity doesn't Quantize**: A member expressed skepticism that sparsity aids in quantization, noting that many quantization approaches first perform a random projection of the weights to make each unit behave like a Gaussian.
   - It was mentioned that the sparsity patterns in **MoE models** are not yet very clear, while **ReLU** can induce a somewhat high degree of sparsity.
- **Distillation Reveals Lower Complexity**: Distillation demonstrates that models have lower complexity after being trained, as it involves describing something very close to the original model with significantly less data/information.
   - Large models explore far more possible configurations, but their optimal state can be simple to describe, enabling replication with smaller models.
- **Coding Specification over Implementation**: A member implemented a custom learning algorithm within an existing codebase using **Codex IDE** without even looking at the code, trusting it to handle implementation errors based on their experience.
   - They still needed to provide human intelligence to guide it, providing more comprehensive solutions and running multiple training runs to find the right hyperparameters, proving that AI needs guidance to be effective.
- **Paper Proofs please the people**: A member asked whether submitting an arXiv paper consisting only of proofs and motivation from other literature, without empirical results, would be accepted.
   - Another member joked this sounds like a literature review.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1414616862771253439)** (6 messages): 

> `RAG agent resources, Langchain alternatives, While loops for agents` 


- **Hunting RAG/Agent Resources**: A member asked for a *good resource* to create a RAG/agent.
   - They noted the hate surrounding **Langchain** and inquired about alternatives.
- **While Loop Wisdom**: A member jokingly suggested using a **while loop** as an alternative to **Langchain** for agent creation.
   - They clarified their skepticism, stating they weren't sure what value **Langchain** provides, even after repeated investigations.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1413931013511577600)** (13 messages🔥): 

> `In-Memory Computing, OpenAI Jobs Platform, ASML Investing in LLMs, Custom Pre-trained Models, Mistral's Profitability` 


- **In-Memory Computing on DRAM Debuts!**: A new paper ([In-Memory Computing on Normal DRAM](https://arxiv.org/abs/2503.23817)) discusses performing in-memory computing on standard **DRAM**, noting that logical negations are a major limitation due to difficulties mapping `NOT` operations.
   - The primary challenge for storage+compute is achieving sufficiently fast storage and parallel workloads, as in-memory compute doesn't time multiplex, which is why research often favors something like **ReRAM**.
- **OpenAI Enters Job Market with AI Platform!**: Bloomberg reported ([OpenAI Unveils Jobs Platform, Certification Program for AI Roles](https://www.bloomberg.com/news/articles/2025-09-04/openai-unveils-jobs-platform-certification-program-for-ai-roles)) that **OpenAI** plans to launch a new **AI-powered job platform** and introduce a **certification program** for AI skills next year.
   - The Decoder confirmed ([OpenAI Plans an AI-Powered Job Platform to Certify and Connect Workers](https://the-decoder.com/openai-plans-an-ai-powered-job-platform-to-certify-and-connect-workers/)) that a member sarcastically questioned this, comparing it to *bullshit abcd certifications* from Microsoft and Oracle.
- **ASML Eyes LLM Investment**: Members are discussing ASML's investment in a LLM company ([source](https://x.com/ns123abc/status/1964738357403308147)).
   - The investment might be from some general investment fund, since it doesn't make much sense strategically for a lithography company to invest in an LLM company, but some posited it could be for internal models.
- **Tailored Models Trump Generalists**: It was mentioned that the level of customization from a company like **ASML** might allow them to target a partially custom pre-trained model, not just another finetune.
   - This is because a more narrowly trained model could achieve better performance if not restricted by general usefulness across a huge variety of tasks.
- **Mistral's Billion-Dollar Valuation Questioned**: A member suggested that **Mistral's LLMs** aren't worth **$1.3 billion** internally, especially when secure closed source and open source alternatives exist.
   - The same member added that Mistral doesn't seem to be profitable and that it might be political favors at play.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1413602613143797910)** (33 messages🔥): 

> `Aider Success Stories, Aider vs Fully Agentic Tools, Token Efficiency with Codex, GPT-5 with Aider, Fast and Effective Models for Web Dev` 


- **Aider Not Fully Agentic; Developer Still Needed**: A member stated that **Aider isn't a pure agentic tool** and the results depend heavily on the developer.
   - They added that Aider allows for better control over the **LLM's context**, and that its *file-editing mechanism is excellent*.
- **Impressive Token Efficiency with Codex**: A member closed a **codex session** showing impressive token efficiency, with **2304542 total tokens** used.
   - The session had **2218442 input tokens** (+ **16140160 cached**) and **86100 output tokens**.
- **Gemini Flash 2.5 shines for Web Dev**: For basic static web development with a headless CMS, **Gemini Flash 2.5** was recommended, citing that **Gemini 2.5 Pro latency** *is killing my productivity*.
   - It was stated that **Jekyll** and **Gemini Flash 2.5** was used to build **3 static sites**.
- **Aider's MCP Configuration Available in Downstream Project**: Because **MCP** is not yet supported by the main repo, users have been merging **PR 3937** into personal forks to implement it, as detailed in the [dwash96/aider-ce repo](https://github.com/dwash96/aider-ce).
   - The repo is for configuring **MCP**, and includes documentation ([mcp.md](https://github.com/dwash96/aider-ce/blob/main/aider/website/docs/config/mcp.md)).
- **10x Coding Speed Increase a Myth**: One member thinks the *'10x your speed' from AI enabled coding is a myth*, suggesting **25-50%** is more realistic.
   - They added that **LLMs are fantastic at automating typing** but only *kind of can automate thinking*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1413663418959003689)** (5 messages): 

> `Aider Code Generation Percentage, Aider's Safety Mechanisms, Reasoning Effort and Edit Actions, Confirmation System in Aider, Linting Configuration` 


- ****Aider's Code Autonomy Alleged****: A member mentioned that [Aider claims](https://aider.chat) to write **70-80%** of its own code and suggested using it to architect its own codebase.
   - The suggestion was meant to discover more information about how it works in an *inception-like* approach, though others have found the suggestion unhelpful.
- ****Aider's Safety is Malleable****: A member inquired about safety mechanisms in **Aider** to prevent dangerous code execution like unverified network requests or arbitrary shell execution.
   - The response indicated that no such system exists for automatic prevention or blacklisting, because the boundary between safe and unsafe is very broad and varies by use case.
- ****Reasoning Effort Applies Everywhere****: The `--reasoning-effort high` flag does apply to edit actions.
   - Members suspect that **Deepseek 3.1** thinks too much for its own good as a weak model, and that `--reasoning-effort high` should only be used on stronger models.
- ****Aider Confirmation System Demystified****: The confirmation system requires user input before execution and combines output parsing for web searches, tool calls, and CLI commands through a *"should I do this?"* prompt.
   - It was mentioned that defense is more at the linter configuration level and is gated by a confirmation, but the host is liable for gating handoffs.
- ****Defense in Linting and Blacklisting****: Defense against potentially harmful code lies in linting configuration, command blacklists, and domain blacklists/whitelists.
   - The system is liable for detecting when executions are being handed off to the underlying host and gating those handoffs by whatever method you think reasonable.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1413619616806080563)** (20 messages🔥): 

> `Kernel Removal Project, Digital Ocean Issues, ShapeTracker Bounty Removal, Tinygrad Community Bounties, Meeting #87 Topics` 


- **Users Tackle Kernel Removal with Troubles on Digital Ocean**: A member inquired about assisting with the **kernel removal project** but encountered issues with **Digital Ocean**, reporting that power cycling the droplet prevented the **Docker container** from starting.
   - After deleting and recreating a new droplet, the issue was resolved, reinforcing the sentiment that *"hardware in his custody physically and not cloud access"* is preferable.
- **ShapeTracker Bounty Rides Off into the Sunset**: With recent progress, the bounty for **proving ShapeTracker mergeability in Lean** is slated for removal from the [bounty list](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0&range=A89).
   - George Hotz confirmed that *"we can remove it now"*, emphasizing that *"it's all just symbolic rules"*.
- **Tinygrad Community Meeting Agenda Posted**: The agenda for meeting **#87** was posted, including topics such as **company updates**, making **rangeify default**, **CI speed**, **MLPerf Llama**, **viz tool**, **CPU thread**, **symbolic**, **cloud**, and **other bounties**.
   - The meeting was scheduled for **9am Monday San Diego time**.
- **Expert-Parallel MoE Strategy Questioned**: A member questioned how **expert-parallel Mixture of Experts (MoE)** would be handled if big graph and remote all go according to plan.
   - They expressed concerns that static scheduling might break down the process.
- **Call for Modularizing Test Tensor Core**: A member suggested moving "**Test Tensor Core**" into a separate file at `test/unit/test_emulated_tc.py`.
   - No rationale for this modularization was given.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1414258082921447455)** (2 messages): 

> `graph_rewrite_map(), Tensor vs MathTrait` 


- **Unraveling graph_rewrite_map() Internals**: A user seeks a comprehensive understanding of how `graph_rewrite_map()` functions, inquiring about the distinctions between bottom-up and top-down matching strategies within it.
- **Tensor Methods: Tensor or MathTrait?**: A user questions why methods on **Tensor** sometimes return `(Tensor | MathTrait)`, highlighting the potential for type-checking issues since methods like `.silu()` cannot be applied to a **MathTrait**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1413619165889036308)** (20 messages🔥): 

> `Manus bugging out, Manus API key, New MCP and API connectors, Flowith invitation, Politeness to AIs` 


- **AI-mability: Politeness Pays Off!**: A research paper suggests being polite to AIs yields better results, as detailed in the [arxiv.org paper](https://arxiv.org/pdf/2402.14531).
   - The study *scientifically proves* that AIs respond favorably to polite requests.
- **Flowith's Flowtastic Invitation!**: A member shared a [Flowith invitation link](https://flowith.io/?inv=EXVVQXH8QBRRSIVH), offering exclusive deals for new users.
   - Flowith appears to be a new platform.
- **Manus's Malfunctioning Mayhem!**: A member reported that **Manus** was bugging out, getting stuck in a loop after being asked to wait for input.
   - Others speculated about the default to **adaptive mode** and its impact on credit usage.
- **MCP's Magical API Connectors!**: Members are excited about the new **MCP** and **API connectors** feature recently launched.
   - No specific launch date was mentioned.
- **API Key Quest for Manus!**: A member asked for assistance in obtaining a **Manus API key**.
   - Another member confirmed the cessation of free credits, noting the lack of related information.


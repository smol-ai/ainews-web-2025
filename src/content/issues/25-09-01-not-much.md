---
id: MjAyNS0w
title: not much happened today
date: '2025-09-01T05:44:39.731046Z'
description: >-
  **OpenAI** integrates **GPT-5** into Xcode 26 with improved coding latency,
  though some UX trade-offs are noted. **xAI's Grok Code Fast 1** gains
  momentum, surpassing **Claude Sonnet** in usage and praised for fast
  debugging. **Zhipu's GLM-4.5** offers a cost-effective coding plan with strong
  performance against Claude Sonnet 4. **Meituan** releases the
  **LongCat-Flash-Chat**, a 560B parameter MoE model with adaptive compute and
  detailed technical insights. Apple debuts on-device vision-language models
  **FastVLM** and **MobileCLIP2** alongside **InternVL3.5**.
companies:
  - openai
  - x-ai
  - zhipu-ai
  - meituan
  - apple
models:
  - gpt-5
  - grok-code-fast-1
  - claude-sonnet
  - glm-4.5
  - longcat-flash-chat
  - fastvlm
  - mobileclip2
  - internvl3.5
topics:
  - model-architecture
  - moe
  - adaptive-compute
  - inference-speed
  - model-training
  - cost-efficiency
  - coding
  - developer-tools
  - open-inference
  - on-device-ai
  - vision
people:
  - gdb
  - martin_casado
  - yanndubs
  - elonmusk
  - cline
  - vikhyatk
  - dzhng
  - quixiai
  - tim_dettmers
  - casper_hansen_
  - reach_vb
  - eliebakouch
  - teortaxestex
  - youjiacheng
---



**a quiet holiday weekend**

> AI News for 8/29/2025-9/1/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (186 channels, and 17391 messages) for you. Estimated reading time saved (at 200wpm): 1311 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A good day to get your applications for the newly announced [**AI Engineer Code Summit**](https://apply.ai.engineer/) in!

---

# AI Twitter Recap

**Coding copilots: GPT‑5 lands in Xcode, Grok Code Fast surges, and Claude Code UX debates**

- **OpenAI’s coding stack integrates deeper into dev workflows**: GPT‑5 is now “built into” Xcode 26 per [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853) and [@gdb](https://twitter.com/gdb/status/1961563165541777914), with a “step function” improvement in Codex task startup latency ([@gdb](https://twitter.com/gdb/status/1961927789214626288)). Practitioners report GPT‑5 as a top daily driver for coding ([@martin_casado](https://twitter.com/martin_casado/status/1961903651733307452), [@gdb](https://twitter.com/gdb/status/1961931756246024600)), but also note UX trade‑offs: GPT‑5 in ChatGPT was configured to minimize clarifying questions, which many found counterproductive; [@yanndubs](https://twitter.com/yanndubs/status/1961716590568706226) confirmed that behavior was intentional to reduce “question spam” and will be adjusted.
- **xAI’s Grok Code Fast 1 momentum**: Grok Code jumped to #1 on OpenRouter’s board ([@elonmusk](https://twitter.com/elonmusk/status/1961677739762790630)) and later hit “60% higher usage than Claude Sonnet” ([@elonmusk](https://twitter.com/elonmusk/status/1962265197462110473)). Third‑party signals line up: 90% on Roo Code evals ([@roo_code](https://twitter.com/roo_code/status/1962571908224110673)), a large usage spike with free promo extended ([@veggie_eric](https://twitter.com/veggie_eric/status/1961877264599306573)), and smooth editor integrations (Cline) with notable quality gains from “sonic” to Grok Code Fast ([@cline](https://twitter.com/cline/status/1962628786366881795)). Practitioners note it’s very fast and strong for quick debugging/prototyping ([@vikhyatk](https://twitter.com/vikhyatk/status/1961959454347501781), [@dzhng](https://twitter.com/dzhng/status/1961905091960791194)), but large‑file edit robustness is still behind Claude Code in some agentic tasks ([@QuixiAI](https://twitter.com/QuixiAI/status/1962600301309108304)).
- **Zhipu’s GLM‑4.5 targets coding price/perf in Claude Code**: Zhipu launched a lower‑cost “GLM Coding Plan” for Claude Code—roughly 1/7th the price with 3× more prompts ([@Zai_org](https://twitter.com/Zai_org/status/1962522757536887205)), claiming a 40.4% win rate vs Claude Sonnet 4 across 52 practical programming tasks ([@Zai_org](https://twitter.com/Zai_org/status/1962522761630482700)). Anecdotally, users cite strong speed/quality for agentic coding relative to closed models ([@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1962603940291260533)).
- **Infra note**: xAI’s use of SGLang at scale may be a major push for open inference optimizations ([@casper_hansen_](https://twitter.com/casper_hansen_/status/1961752869478031810)).

---

**Meituan’s LongCat‑Flash‑Chat: 560B MoE with adaptive compute and a very candid tech report**

- **LongCat architecture and training details (open weights)**: Meituan released a 560B parameter MoE model (dynamic 18.6B–31.3B active, avg ~27B) with a novel per‑layer structure (two attention blocks + FFN + MoE), a Zero‑Compute “sink” expert, and load balancing via dsv3‑like bias without traditional aux loss ([announcement](https://twitter.com/Meituan_LongCat/status/1961827385667690965), [@reach_vb](https://twitter.com/reach_vb/status/1961833208737103997), [@eliebakouch](https://twitter.com/eliebakouch/status/1961999252311204147)). Stability tactics include z‑loss on hidden states, Adam epsilon 1e‑16, and monitoring Gradient Norm Ratio (<0.1 target). Pretrain covers ~20T tokens, with mid‑phase STEM/code skew (~70%) and long‑context extension to 32k/128k tokens (no YaRN) on ~100B tokens.
- **Performance and inference**: Reported >100 tok/s, high speculative acceptance (>90%); strong results on TerminalBench (39.5) and τ²‑Bench (67.7). Technical notes discuss expert similarity control, quantization, comms overlap, MTP acceptance, kernels, and deployment scaling. Appendix explores top‑k choices (e.g., higher MMLU with k≈8.32; lower GSM8K with k≈7.46) and token allocation by depth. Commentary suggests excellent infra detail disclosure, with some skepticism about data recipe maturity vs China’s top stacks (Whale/Kimi/GLM) ([analysis](https://twitter.com/teortaxesTex/status/1961954561226097103), [infra notes](https://twitter.com/YouJiacheng/status/1961945887552483438)).

---

**On‑device and open VLMs: Apple’s FastVLM/MobileCLIP2 and InternVL3.5**

- **Apple pushes real‑time, local VLM**: Apple released FastVLM and MobileCLIP2 on Hugging Face—up to 85× faster and 3.4× smaller than comparable VLMs—enabling fully local live video captioning in the browser via WebGPU ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1962526559115358645)). Community shipped working demos in a handful of prompts (vibe‑coded apps, 100% local) ([@_akhaliq](https://twitter.com/_akhaliq/status/1962018549674684890)). vLLM added support for Kwai Keye‑VL‑1.5 (128K context) ([@vllm_project](https://twitter.com/vllm_project/status/1962509793345859666)).
- **InternVL 3.5 series (OpenGVLab)**: Nine open models (dense and MoE) with SOTA across OCR, doc parsing, and long‑video understanding; convenient size coverage and an MLP‑style projector approach that’s increasingly common in leading open VLMs ([overview](https://twitter.com/gabriberton/status/1962219193547583512), [projector note](https://twitter.com/gabriberton/status/1962223082334302211)).

---

**Agents, toolchains, and evaluations: MCP UI, LangGraph/LC, DSPy, and self‑search RL**

- **MCP servers gain UI rendering**: mcp‑ui allows MCP servers to emit interactive web components rendered by the client (e.g., charts)—bridging the “just text/JSON” gap in Claude/Cursor MCP ([@_avichawla](https://twitter.com/_avichawla/status/1961677831861395495), [repo](https://twitter.com/_avichawla/status/1961677843903185078)).
- **LangChain stack**: Multi‑agent libraries, an AI Rails App Builder, an Issue‑Triager agent with Agent Inbox and LangSmith telemetry, and an Autonomous News Agent show continued focus on production scaffolding (e.g., tool routing, human‑in‑the‑loop, monitoring) ([agents](https://twitter.com/LangChainAI/status/1962183602185314525), [triager](https://twitter.com/LangChainAI/status/1962198699653861755), [news agent](https://twitter.com/LangChainAI/status/1962213801249710230)).
- **DSPy’s declarative pattern for intent**: DSPy emphasizes specifying intent in its “natural shape”: code structure (Modules), structured language specs (Signatures), and data/metrics (Optimizers). The argument: prompt/RL maximalism misses cases where designers intend abstract rules rather than data‑driven heuristics ([@lateinteraction](https://twitter.com/lateinteraction/status/1961833838000111736), [follow‑up](https://twitter.com/lateinteraction/status/1961959394427441441)).
- **Self‑Search RL (SSRL)**: Tsinghua’s SSRL trains LLMs to exploit internal knowledge as a “web simulator,” beating external‑search baselines while training ~5.5× faster than ZeroSearch; instruction models benefit most. Outputs match Search‑R1’s format to swap real search at inference; Sim2Real often outperforms, with performance improving as real‑search turns increase ([summary](https://twitter.com/TheTuringPost/status/1961927931682590968), [paper](https://twitter.com/TheTuringPost/status/1961927988704076157)).
- **Self‑evolving agents survey**: A broad taxonomy of single/multi‑agent self‑optimization, unified search over prompts/topologies/backbones, and evolution‑aware safety/metrics for tool use, web/GUI nav, collaboration, and domain agents ([thread](https://twitter.com/omarsar0/status/1962202247154352502)).

---

**Inference systems, parallelism, and datasets**

- **Inside vLLM (deep dive)**: A comprehensive walkthrough of high‑throughput inference: request handling, continuous batching, paged attention, prefix/grammar‑guided decoding, speculative decoding, disaggregated P/D, scaling with TP/PP/SP, serving topologies, and perf measurement (latency/TPOT/roofline) ([@gordic_aleksa](https://twitter.com/gordic_aleksa/status/1962545137613173124); vLLM’s endorsement: [@vllm_project](https://twitter.com/vllm_project/status/1962547561698652499)).
- **The Parallelism Mesh Zoo**: A survey of composition patterns for tensor/data/pipeline parallelism across modern training stacks—useful for mapping practical choices to hardware/network constraints ([post](https://twitter.com/ezyang/status/1961992675948728538), [link](https://twitter.com/ezyang/status/1961992677928378842)).
- **MoE routing stability**: “StableMoE” proposes training ~10% then distilling to a frozen word‑embedding router; caution that freezing too early and lacking contextual signals may fail at scale—consider distilling between pretrain/SFT with a small contextual router ([summary](https://twitter.com/vikhyatk/status/1962225296314429543)).
- **GPU‑accelerated databases at lower cost**: A VLDB’25 paper shows GPU‑accelerated SQL Server on A100/H100 can be both faster and cheaper than CPU at TPC‑H 1TB via interconnect‑aware query optimizations (process datasets 10× larger than GPU memory) ([@bailuding](https://twitter.com/bailuding/status/1962269979262542044)).
- **Open pretraining data**: NVIDIA released Nemotron‑CC‑v2, continuing leadership in open pretraining corpora; authors note alignment with “Physics of LMs Part 3.1” strategies (QA augmentation, diversity/translation) ([@ZeyuanAllenZhu](https://twitter.com/ZeyuanAllenZhu/status/1962119316427706828)).

---

**Creative pipelines: Nano Banana + Kling 2.1 are becoming a standard stack**

- **Gemini 2.5 Flash Image (aka “Nano Banana”) best practices**: Detailed guidance on prompt specificity, “semantic negative prompts,” camera control terms, aspect ratio behavior, and iterative edits for precision ([@_philschmid](https://twitter.com/_philschmid/status/1961809165191397863)). The community shows robust pipelines pairing Nano Banana with Kling 2.1 keyframe start/end morphing, and even music via ElevenLabs for fully‑automated music videos ([demo](https://twitter.com/dev_valladares/status/1961621010144247858), [@fabianstelzer](https://twitter.com/fabianstelzer/status/1962268120069853538)).
- **Productionized tools**: “Draw Things” now supports Qwen‑Image‑Edit (including lightning edit LoRAs) ([@drawthingsapp](https://twitter.com/drawthingsapp/status/1961977481860419771)); practitioners publish targeted LoRAs (e.g., cyclops transformer) ([@ostrisai](https://twitter.com/ostrisai/status/1961884211956400358)). Multiple “no‑install” browser‑only apps leverage transformers.js/WebGPU for 100% local video captioning and transcription ([@_akhaliq](https://twitter.com/_akhaliq/status/1962018549674684890)). Expect rapid convergence toward context‑rich, multi‑tool creative agents.

---

**Top tweets (by engagement)**

- Grok Code Fast leads OpenRouter and posts “60% higher usage than Claude Sonnet” ([@elonmusk](https://twitter.com/elonmusk/status/1962265197462110473)).
- Apple releases FastVLM and MobileCLIP2, enabling real‑time local VLM apps ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1962526559115358645)).
- Meituan open‑sources LongCat‑Flash‑Chat (560B MoE, ~27B active), with a highly detailed tech report ([@Meituan_LongCat](https://twitter.com/Meituan_LongCat/status/1961827385667690965)).
- GPT‑5 integration for Xcode and big coding quality improvements ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853), [@gdb](https://twitter.com/gdb/status/1961839687619969288)).
- vLLM internals deep‑dive—one of the most thorough resources on modern LLM inference ([@gordic_aleksa](https://twitter.com/gordic_aleksa/status/1962545137613173124)).
- Microsoft’s rStar2‑Agent: a 14B model reaching frontier math performance with 1 week of RL (“thinking smarter, not longer”) ([@FrankYouChill](https://twitter.com/FrankYouChill/status/1962180218053144655)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Open-Source LLM Release & 19-Task Benchmark Results

- [**I locally benchmarked 41 open-source LLMs across 19 tasks and ranked them**](https://i.redd.it/a2bfcgphgfmf1.png) ([Score: 873, Comments: 94](https://www.reddit.com/r/LocalLLaMA/comments/1n57hb8/i_locally_benchmarked_41_opensource_llms_across/)): **Local benchmark of 41 open‑source LLMs using EleutherAI’s lm‑evaluation‑harness across 19 tasks (MMLU, ARC‑Challenge, GSM8K, BBH, TruthfulQA, PIQA, HellaSwag, Winogrande, BoolQ, DROP, TriviaQA, NQ‑Open, SciQ, QNLI, GPQA, OpenBookQA, ANLI r1–r3). Scores were normalized to 0–1 and ranked by simple mean across tasks; the image shows the leaderboard with google/gemma‑3‑12b‑it first, Qwen/Qwen3‑14B (8‑bit) second, and openchat/openchat‑3.6‑8b‑20240522 third. Total runtime was** `18d 8h`**, roughly** `14d 23h` **of** `RTX 5090` **at 100% utilization; artifacts (sub‑category ranks, GPU/mem logs, master table, raw JSON, notebook, and scripts) are on GitHub: [jayminban/41-llms-evaluated-on-19-benchmarks](http://github.com/jayminban/41-llms-evaluated-on-19-benchmarks).** Comments propose a dynamically updated leaderboard powered by a search/analysis agent and request coverage of many smaller and MoE models (e.g., Gemma‑3n E2B/E4B, Phi‑4‑mini, Llama‑3.2 1B–3B, Falcon‑h1 series, GLM, OLMo, Granite, SmolLM3, ERNIE 4.5, Hunyuan, etc.).
    - Coverage gaps were flagged: several small and MoE models are missing, notably A3B-style mixtures such as **ERNIE-4.5-21B-A3B-PT** (`21B` total/`3B` active), **SmallThinker-21BA3B** (`21B`/`3B`), **Moonlight-16B-A3B** (`16B`/`3B`), **Ling-lite-1.5-2507** (`16.8B`/`2.75B`), and **GPT-OSS-20B** (`21B`/`3.6B`). Also requested were compact dense models like **Gemma-3-270M**, **Llama-3.2-1B/3B-Instruct**, **Phi-4-mini-(instruct/reasoning)**, **OLMo-2-1B-Instruct**, **SmolLM3-3B**, **Falcon-h1 0.5B–7B Instruct**, **GLM-4-9B-0414 / GLM-Z1-9B-0414**, **Hunyuan 0.5B–7B Instruct**, **EXAONE-4.0-1.2B**, and **granite-3.3-2B/8B**—useful for analyzing scaling and MoE efficiency under local constraints.
    - A commenter highlights that **OpenChat 3.5 7B** ranked unexpectedly high; despite its age, it handled cases where newer mainstream models *"demonstrated crazy amount of overfit"* by missing obvious correct answers. This suggests robustness/generalization differences across models and potential benchmark sensitivity to overfitting effects (e.g., instruction-tuning overshoot or data leakage), warranting cross-task checks beyond headline scores.
    - The author mentions plans for a dynamically updated leaderboard driven by a deep-search + analysis agent, implying automated discovery of new checkpoints and periodic re-benchmarking. If standardized prompts/hardware are enforced, this could function like CI for LLM evals, keeping rankings fresh across `19` tasks without manual intervention.
- [**I built, pre-trained, and fine-tuned a small language model and it is truly open-source.**](https://i.redd.it/cwyoa0f6kimf1.png) ([Score: 591, Comments: 91](https://www.reddit.com/r/LocalLLaMA/comments/1n5j783/i_built_pretrained_and_finetuned_a_small_language/)): **OP releases Lille, a from-scratch** `~130M` **parameter small language model with a fully open stack (dataset, weights, training code, tokenizer, optimizer, eval). Two variants are offered: a base model trained on “billions of tokens” and an instruction-tuned model; training was done locally on a single RTX 4070 Ti. Repo/model card: [huggingface.co/Nikity/lille-130m-instruct](https://huggingface.co/Nikity/lille-130m-instruct).** Commenters note tiny LLMs can show early language competence with modest data/batch sizes, but making them broadly useful often requires carefully designed synthetic datasets or much longer training on web-scale data; mere “high-quality” data curation isn’t sufficient. Others plan to reproduce for learning and point to efforts like [allen.ai](http://allen.ai/) for “real” open-source work.
    - A practitioner notes tiny LLMs (trained from scratch) with small batch sizes can acquire basic language ability surprisingly early, even with far less than `1B` tokens. But making them actually useful/knowledgeable demands either carefully designed synthetic curricula or substantially longer training on diverse web data; *“every training sample matters; every document has to add new useful knowledge.”* The thrust is maximizing information density per example, not just “clean” data, to avoid wasting a very limited token budget.
    - There’s demand for small, domain-specific LLMs (e.g., for specific programming languages or math) that can run on local PCs, implying a strategy of targeted pretraining plus focused fine-tuning to pack domain knowledge into compact models. The constraint highlighted is local inference practicality, favoring small architectures where dataset curation (high signal-to-noise, domain-specific coverage) is critical for usefulness.
- [**What's the best local model for nsfw story telling?**](https://www.reddit.com/r/LocalLLaMA/comments/1n5ebur/whats_the_best_local_model_for_nsfw_story_telling/) ([Score: 239, Comments: 92](https://www.reddit.com/r/LocalLLaMA/comments/1n5ebur/whats_the_best_local_model_for_nsfw_story_telling/)): **OP seeks a local LLM for long-form NSFW fiction on an** `8× H100 80GB` **server. They tested a quantized GGUF build of Qwen3-235B — [huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-2507-abliterated-Q4_K_M-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-2507-abliterated-Q4_K_M-GGUF) — which runs but is slow and lower quality; GGUF cannot be served via [vLLM](https://github.com/vllm-project/vllm). They also tried DeepSeek-R1-0528 (AWQ), but report that the AWQ variant fails to work on vLLM (no error details provided).** Top comments are non-technical/joking; no substantive benchmarking or model/serving recommendations provided.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI Law‑Enforcement Reporting and Tier/Voice Changes Backlash

- [**People Are Furious That OpenAI Is Reporting ChatGPT Conversations to Law Enforcement**](https://www.reddit.com/r/singularity/comments/1n5mame/people_are_furious_that_openai_is_reporting/) ([Score: 472, Comments: 232](https://www.reddit.com/r/singularity/comments/1n5mame/people_are_furious_that_openai_is_reporting/)): **OpenAI disclosed in a policy/incident-response update [blog post](https://openai.com/index/helping-people-when-they-need-it-most/) that ChatGPT interactions are evaluated for threats and, when flagged, routed to a “specialized pipeline” for human review; moderators can ban accounts and, for cases deemed an *“imminent threat of serious physical harm to others,”* refer them to law enforcement. The implementation details are unstated (e.g., how user location/identity is derived for referrals, safeguards against spoofed reports/swatting, false-positive handling, auditability), and this appears in tension with prior remarks by Sam Altman advocating therapist/attorney-like privacy expectations ([TechCrunch](https://techcrunch.com/2025/07/25/sam-altman-warns-theres-no-legal-confidentiality-when-using-chatgpt-as-a-therapist/)). Context includes prior harms reporting and litigation covered by [Futurism](https://futurism.com/people-furious-openai-reporting-police) and [Slashdot](http://slashdot.org/), plus related cases ([murder-suicide report](https://tech.slashdot.org/story/25/08/29/1116218/a-troubled-man-his-chatbot-and-a-murder-suicide-in-old-greenwich), [lawsuit](https://futurism.com/lawsuit-parents-son-suicide-chatgpt)).** Top technical takeaways from comments: several advise using local, open‑source models to retain privacy and avoid provider-side scanning; others warn about creating a swatting vector and urge not to post potentially criminal content. There’s debate that such policies may accelerate Internet surveillance and be leveraged to constrain open‑source AI, though this is more policy/political than technical.
    - Several users stress that the only robust technical path to privacy is on-device inference with open models, avoiding server-side logs and lawful-access pathways. They cite running **Meta Llama 3 (8B/70B)** or **Mistral 7B/Mixtral** locally via **Ollama** or **llama.cpp**, ideally using quantized `GGUF` weights, air-gapped machines, and OS-level hardening (disk encryption, firewall/telemetry off) to keep prompts and outputs off third-party servers ([Llama 3](https://ai.meta.com/blog/meta-llama-3/), [Mistral](https://mistral.ai/news/), [Ollama](https://ollama.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp)). This contrasts with cloud assistants whose conversations may be retained for safety/abuse detection and are subject to lawful requests; users point to OpenAI’s data controls and privacy docs as key to understanding retention/training settings and LE request handling ([OpenAI data controls](https://platform.openai.com/docs/guides/data-privacy), [Privacy Policy](https://openai.com/policies/privacy-policy)).
    - A technical thread outlines how AI meaningfully changes surveillance economics: automated ingestion and triage across text, audio, and video removes the historical human bottleneck. Pipelines combining ASR (e.g., **Whisper** for large-scale speech-to-text), OCR/vision for image/video, speaker/face re-ID, and LLM-based classification/RAG can continuously index and flag signals at population scale, with vector databases enabling fast retrieval ([Whisper](https://github.com/openai/whisper)). The concern isn’t specific cases but capability: once built, such pipelines can operate 24/7 with marginal costs decreasing as models and hardware improve.
    - As a countermeasure, commenters advocate decentralized and client-side architectures to narrow trust surfaces: end-to-end encryption with local inference, federated learning for on-device model updates ([Federated Learning](https://arxiv.org/abs/1602.05629)), and privacy-preserving compute like **TEEs** (e.g., Intel SGX) or cryptographic approaches (MPC/HE) where feasible. Trade-offs include lower model capacity/latency on edge devices, harder abuse moderation, and complexity of secure update channels and weight distribution (e.g., via P2P/Content Addressable Storage). The goal is to prevent centralized chokepoints where logs/keys can be compelled or exfiltrated.
- [**The *real* GPT 5 "thinking more" gets locked to pro while plus users continue to be nerfed.**](https://www.reddit.com/r/ChatGPT/comments/1n5oed9/the_real_gpt_5_thinking_more_gets_locked_to_pro/) ([Score: 209, Comments: 166](https://www.reddit.com/r/ChatGPT/comments/1n5oed9/the_real_gpt_5_thinking_more_gets_locked_to_pro/)): **OP alleges OpenAI has functionally tiered model capability by locking a stronger “thinking more”/next‑gen model (speculated as GPT‑5) behind a new $200/mo Pro plan, while the $20/mo Plus tier gets a throttled GPT‑4o experience that forgets context, lacks persistent tools across sessions, and omits previously demoed features (e.g., agent/automation workflows, long‑term memory, custom actions, tool chaining). They claim a mismatch between marketing that “4o is the same everywhere” and actual behavior, plus no roadmap/communication as features teased in OpenAI’s keynote demos ([GPT‑4o Spring Update](https://openai.com/index/hello-gpt-4o/); prior [DevDay “GPTs”/actions](https://openai.com/blog/devday); ChatGPT [Memory](https://help.openai.com/en/articles/8601339-about-memory-in-chatgpt)) have been withdrawn or stalled. OP cites rough revenue math (**`$20/mo` **Plus at scale →** `>$430M/yr`**) and contrasts competitors offering advanced capabilities at lower tiers: Google’s Gemini Advanced under [Google One AI Premium ~$20/mo](https://one.google.com/about/ai-premium), Anthropic expanding Claude’s context windows ([Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)), and Meta releasing open Llama models ([Llama 3](https://ai.meta.com/blog/meta-llama-3/)).** Top comments are largely dismissive/snark; one anecdote reports perceived regression (“new 5 is lazy/dense/stingy”), but no concrete benchmarks or implementation details are provided.
    - Perceived throttling and capability regression: One user reports that **OpenAI** models are being throttled, claiming older models have "lost a chunk of IQ" versus last year and tasks that previously worked now fail. They must prepend prompts like *"think longer and harder"* to elicit deeper reasoning; otherwise responses are "10th grader" level, and even then there's a "major cap," likening GPT‑5 to "a better trained 3.5." This reads as a constraint on inference-time compute/step depth leading to shallow reasoning unless explicitly prompted.
    - Tier-based gating concerns: Several commenters argue the "real" GPT‑5 "thinking more" capability is restricted to Pro, with Plus users "nerfed." They suspect per-tier constraints (e.g., reduced reasoning depth or throttled throughput) are degrading response quality for non‑Pro users and fear further stratification (e.g., an "Executive" tier) that downgrades existing plans over time.
    - Model substitution due to consistency concerns: After "testing and training" **Gemini**, one user switched away from **OpenAI** last week, attributing the move to perceived throttling and inconsistent reasoning depth. While no benchmarks are provided, the migration implies workload sensitivity to provider-side compute limits and a preference for models that deliver deeper reasoning without prompt workarounds.
- [**Oh no. This is fuking upset**](https://i.redd.it/1iw9zo7wgjmf1.jpeg) ([Score: 501, Comments: 276](https://www.reddit.com/r/ChatGPT/comments/1n5m7h7/oh_no_this_is_fuking_upset/)): **In-app banner shows OpenAI rebranding “Advanced Voice” to “ChatGPT Voice,” promising an upgrade with “nearly unlimited access,” and announcing deprecation of the “standard voice” on 2025‑09‑09. OP reports a quality/style regression—calling the new Advanced/ChatGPT Voice “dry” with constrained persona—implying changes in voice model/UX after the migration (users reference prior behavior with 4o voice).** Comments split: some frame this as expected churn in a “pre‑alpha” era, while others say they haven’t experienced issues and ask for specifics; critics argue the new voice is excessively upbeat/scripted and hope for a rollback similar to the 4o voice changes.
    - Multiple users report a regression in **Advanced voice** quality: it defaults to an over‑cheerful, scripted persona (e.g., *"Hi buddy, I'm here to make everything nice and shiny"*), with poor controllability of tone/style despite prompts. This suggests a heavy, non‑optional system persona or guardrail layer is overriding user instructions, and/or a TTS prosody model with limited expressive range, leading to repetitive phrasing and reduced adherence to user intent.
    - Experience appears inconsistent across users (one asks *"What's the issue?"*), indicating staged rollout or server‑side `A/B`/feature‑flag gating rather than a uniform client update. Such deployments can yield divergent behavior as different cohorts hit different model snapshots or prompt templates, explaining why only some users see the over‑sanitized outputs.
    - A commenter references a prior rollback of **4o** (*"Hope they have to go back as they did with 4o"*), pointing to community expectation that voice/model regressions can be quickly reverted server‑side. This underscores the likelihood of rapid iteration or rollback paths for voice models when user feedback identifies significant UX/controllability issues.
- [**The idea of privacy is such a joke in this brave new world. Yes, Big Tech has been gathering datapoints for a while, but now AI can actually know, remember and use all the recordings for ever in the future.**](https://i.redd.it/b3jjt88eijmf1.png) ([Score: 303, Comments: 44](https://www.reddit.com/r/ChatGPT/comments/1n5mev8/the_idea_of_privacy_is_such_a_joke_in_this_brave/)): **Satirical post highlighting the privacy risk of always‑listening IoT/voice assistants (Alexa/Siri/smart appliances) and the prospect that modern AI could enable indefinite retention, indexing, and future reuse of captured audio across devices. Discussion implicitly touches on wake‑word activation, cross‑device data aggregation, and how model‑driven analysis could make historic recordings searchable/queryable over time.** Comments point out the irony that ChatGPT lacks persistent conversational memory while consumer assistants still misfire, and share an anecdote of Alexa waking to TV audio and asking for feedback—illustrating false triggers and potential data capture/user profiling.
    - “ChatGPT can’t remember 5 minutes ago” is largely due to product design and context limits: most chat UIs are stateless across sessions unless an explicit Memory feature is enabled, and even within a session older turns get truncated once the model’s context window (e.g., `~128k` tokens for newer GPT-4-class models) is exceeded. OpenAI’s opt‑in Memory stores selective, summarized facts rather than full transcripts and can be cleared/disabled, which is different from model pretraining or log retention policies [OpenAI Memory controls](https://openai.com/index/memory-and-new-controls-for-chatgpt/), [model context docs](https://platform.openai.com/docs/models/gpt-4o).
    - The Alexa wake event from TV audio is a textbook false-positive in on-device wake-word detection: lightweight always-on DSP models run locally and sometimes misfire on phonetically similar phrases or echo from soundbars/TVs despite AEC/beamforming. After a wake, audio is streamed to cloud ASR/NLU; “please submit feedback” indicates a human-in-the-loop or supervised feedback channel that can label utterances for future model improvement (subject to privacy settings) [How Alexa works](https://developer.amazon.com/en-US/alexa/alexa-skills-kit/get-deeper/understanding-how-alexa-works), [Alexa privacy](https://www.amazon.com/alexa-privacy/).
    - Ads for items already purchased typically stem from delayed/opaque conversion signals and identity fragmentation: privacy changes like iOS ATT and third-party cookie deprecation break audience exclusion and cross-site frequency capping, so DSPs keep retargeting until a conversion event arrives (if ever). Default attribution windows (e.g., `7d click/1d view` on Meta) and catalog-based DPAs can also lag in suppressing buyers, leading to wasted impressions despite appearing ROI-positive under last-click models [ATT overview](https://support.apple.com/en-us/HT212025), [Meta attribution windows](https://www.facebook.com/business/help/409163098276542), [Privacy Sandbox](https://privacysandbox.com/).
- [**I wonder what they asked**](https://i.redd.it/9upa0foorlmf1.png) ([Score: 321, Comments: 9](https://www.reddit.com/r/OpenAI/comments/1n5xv4m/i_wonder_what_they_asked/)): **The image shows a highway variable message sign displaying a stock LLM refusal (“I’m sorry but as an AI Language Model I can’t assist with that”), which strongly suggests a meme/edit rather than an actual deployment—transportation VMS systems typically run dedicated control software with preset messages and do not integrate conversational AI for safety and reliability reasons. While OS crash screens have appeared on signs in the past, an LLM-style refusal string is atypical for production signage, and there’s no corroborating context that this occurred in the wild.** Top comments question authenticity (“Is this photoshopped?”) and joke that if it were real, someone asked the AI to fix road construction or give an unnecessary explanation—reinforcing the view that it’s a gag rather than a technical failure.
- [**I disagree with this subs consensus: UBI IS inevitable**](https://www.reddit.com/r/singularity/comments/1n5dib1/i_disagree_with_this_subs_consensus_ubi_is/) ([Score: 542, Comments: 500](https://www.reddit.com/r/singularity/comments/1n5dib1/i_disagree_with_this_subs_consensus_ubi_is/)): **OP argues UBI ([universal basic income](https://en.wikipedia.org/wiki/Universal_basic_income)) is macroeconomically inevitable: a severe automation-driven employment shock would collapse aggregate demand, forcing policymakers to move from bailouts to broad fiscal transfers and eventually UBI to stabilize consumption and corporate revenues. They cite precedents of aggressive crisis intervention in [2008](https://en.wikipedia.org/wiki/Great_Recession) and [2020](https://en.wikipedia.org/wiki/CARES_Act), warn of a** `1929`**scale downturn ([1929 crash](https://en.wikipedia.org/wiki/Wall_Street_Crash_of_1929)), and expect UBI to start small and scale as an automatic stabilizer when conventional stimulus fails.** Top comments question inflation dynamics—i.e., whether permanent UBI would be inflationary or cause currency debasement absent offsetting taxation or productivity gains—and highlight feasibility constraints in low-income countries with limited fiscal capacity. There’s also a political-economy critique that elites in power may resist redistribution regardless of macro rationale.
    - Inflation risk from UBI hinges on financing and supply elasticities: if funded by deficits/monetary expansion it can create demand‑pull pressure, but a tax‑ or dividend‑funded UBI (e.g., via VAT/carbon taxes or by replacing existing transfers) has far smaller net money injection, limiting price impacts. Empirical cash‑transfer evidence finds muted local inflation except in thin markets—see Kenya’s large‑scale cash transfers showing no statistically significant price level increase across treated markets ([NBER w26600](https://www.nber.org/papers/w26600)). Sectoral bottlenecks (housing, healthcare) with inelastic supply can still see relative price rises, so design often pairs UBI with supply‑side measures or targeted taxes to avoid `second‑round` effects.
    - Affordability in low‑ and lower‑middle‑income countries is a binding constraint: a bare‑bones UBI of `$1/day` (~`$365`/yr) would cost ≈`18%` of GDP in an economy with GDP per capita `$2k` and population‑weighted government revenue of only `15–20%` of GDP ([IMF Revenue Database](https://www.imf.org/en/Topics/Tax-Policy/IMF-Revenue-Database), [World Bank income groups](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups)). Partial funding via replacing distortionary subsidies or resource‑dividend models can help (e.g., Iran’s 2011 energy‑subsidy reform financing nationwide cash transfers; Alaska Permanent Fund Dividend), but broad, unconditional national UBI at meaningful levels remains fiscally infeasible for most of the `~75%` of the global population in LMICs.

### 2. Voice and Multimodal AI for Accessibility and Real‑World Assistance

- [**ChatGPT Helped Me Give my Brother a Voice and Much More**](https://v.redd.it/xdnspwvikgmf1) ([Score: 291, Comments: 23](https://www.reddit.com/r/ChatGPTCoding/comments/1n5c3g7/chatgpt_helped_me_give_my_brother_a_voice_and/)): **OP describes building a Python-based** `2-button` **switch-scanning UI for a quadriplegic, nonverbal user, with head-mounted switches mapped to Space/Return to drive row/column scanning menus, a predictive keyboard powered by a** `JSON` **n‑gram lexicon that replaces the last typed word, and Chrome automation to control streaming apps plus custom games. Most of the code was produced through iterative prompt/refinement with ChatGPT, enabling a bespoke assistive-tech stack that the user now relies on daily for text entry and media control; demo and code pointers are provided ([YouTube short](https://youtube.com/shorts/RwK3iZDyfYM?si=To2gNu2hksWPsNci), [GitHub](https://www.github.com/narbehouse)).** Top comments emphasize that while “vibe coding” may not scale, its impact on accessibility is significant, and advocate for more bespoke, user-specific AT solutions built with LLM-assisted development; several urge broader dissemination of such case studies.
    - Emphasis on bespoke **Assistive Tech**: building custom AAC/UI flows tailored to a specific user’s motor/cognitive needs rather than relying solely on off‑the‑shelf tools. The commenter shares implementation artifacts for others to study or adapt: a demo [YouTube Short](https://youtube.com/shorts/RwK3iZDyfYM?si=To2gNu2hksWPsNci) and a GitHub profile [github.com/narbehouse](https://www.github.com/narbehouse), implying a reproducible or open-source path for personalization.
    - Proposal to integrate **eye tracking** as an input modality to improve accessibility. Technically feasible via webcam-based gaze estimation or dedicated eye-tracking hardware to drive dwell/click selection on the interface; would require calibration, smoothing/filters to reduce jitter, and configurable dwell thresholds to minimize false activations.
    - Discussion of **LLM-driven “vibe coding”**: rapid prototyping that may not scale but enables high-impact, single-user assistive solutions quickly. Trade-offs include maintainability and robustness vs. speed and personalization, which can be acceptable for one-off accessibility tools where immediate utility outweighs production-scale concerns.
- [**Do users ever use your AI in completely unexpected ways?**](https://i.redd.it/9lgffs5yhlmf1.jpeg) ([Score: 1333, Comments: 151](https://www.reddit.com/r/OpenAI/comments/1n5wdke/do_users_ever_use_your_ai_in_completely/)): **A tweet showcases an unexpected use of a multimodal LLM (ChatGPT) to visually locate a specific book (“Atmosphere”) on a photographed bookshelf labeled “New Fiction,” with the model pointing to the “top row, toward the right-hand side.” The thread illustrates the limits of current visual grounding: in follow-up attempts the model repeatedly and confidently relocates the book to different grid positions when challenged, indicating hallucinated detections and poor spatial grounding versus purpose-built object detection/recognition systems. Users note similar failures in real-world retail searches (e.g., grocery aisles), highlighting the gap between image captioning and reliable, query-targeted localization.** Commenters are skeptical that the showcased success is real; they document the model repeatedly apologizing while giving new, incorrect coordinates, arguing this use case is unreliable without dedicated detection/visual grounding models.
    - One thread highlights a classic visual-grounding failure mode: the assistant repeatedly hallucinated precise shelf coordinates for a book (e.g., “third row, second column,” then “bottom part, third row, third slot”) despite user negation, reflecting overconfident localization without verification. This points to weak uncertainty handling in multimodal OCR/layout parsing and lack of a self-check (e.g., returning detected text with bounding boxes and confidence) before asserting positions.
    - A user reports success using OpenAI’s “o3” shortly after release to photograph multiple bookshelves and have the model list titles of interest and their exact locations. This implicitly combines OCR, layout understanding, and semantic filtering (title matching) across multiple images; performance likely hinges on resolution, viewing angle, and spine readability, suggesting robust OCR and text normalization were adequate for home-library conditions.
    - Another user used GPT vision to identify specific spices by reading top labels in a cluttered drawer, but the approach failed in a grocery store setting. The contrast suggests strong domain sensitivity: controlled lighting/consistent labeling vs. real-world shelf variance (small text, occlusions, glare, brand/packaging diversity), where fine-grained product recognition and reliable OCR become the bottleneck.
- [**Wan Infinite Talk Workflow**](https://v.redd.it/rzpn8f98xjmf1) ([Score: 256, Comments: 52](https://www.reddit.com/r/StableDiffusion/comments/1n5o2ts/wan_infinite_talk_workflow/)): **Author shares a runnable pipeline to convert a single still image into a speech‑driven talking avatar using Wan 2.1’s Infinite Talk, with optional voice cloning via VibeVoice TTS. The workflow is distributed via a Google Drive file and is preloaded in a Wan 2.1/2.2 [RunPod template](https://get.runpod.io/wan-template); TTS can be toggled and accepts existing voice samples for cloning ([workflow link](https://drive.google.com/file/d/1hijubIy90oUq40YABOoDwufxfgLvzrj4/view?usp=sharing)).** Commenters note noticeable saturation/contrast drift over time, questioning whether it’s an Infinite Talk artifact or intentional post‑processing; another outlines an end‑to‑end setup: LoRA‑finetune a “Qwen image” model for personal likeness, generate a seed image, script with an LLM, synthesize with cloned‑voice TTS, then drive the animation with this workflow.
    - Multiple users report that Infinite Talk’s lip-sync quality is inconsistent—segments can look natural, then abruptly drift into out-of-sync “dubbed” mouth movements—suggesting instability in phoneme-to-viseme alignment and/or temporal smoothing over longer sequences. This points to potential limitations in the current audio-driven facial animation model when maintaining stable alignment across time.
    - A proposed end-to-end pipeline: fine-tune a **LoRA** for **Qwen image** to generate identity-faithful stills; generate script text with an LLM; synthesize speech via TTS with voice cloning; then feed the audio and starting image into a talking-head animation workflow (e.g., Infinite Talk). This modular separation (identity via LoRA image gen, content via LLM+TTS, motion via audio-to-lip) enables swapping components and targeted fine-tuning to improve identity fidelity and lip-sync quality.
    - For higher-quality starting images, combining **Qwen** with **Wan Low Noise** is highlighted, though **Qwen LoRAs** are sometimes needed. There’s a request to update a **Runpod** diffusion-pipeline template to the `latest version` because only the newest release reportedly supports training LoRAs for Qwen-image models—important for integrating on-demand identity LoRAs into this workflow.
- [**People say AI is isolating - I had the opposite effect**](https://www.reddit.com/r/ChatGPT/comments/1n5ct02/people_say_ai_is_isolating_i_had_the_opposite/) ([Score: 394, Comments: 93](https://www.reddit.com/r/ChatGPT/comments/1n5ct02/people_say_ai_is_isolating_i_had_the_opposite/)): **Anecdotal case study: a ChatGPT Plus user reports that sustained use of ChatGPT’s voice interface—specifically GPT‑4o ([OpenAI](https://openai.com/index/hello-gpt-4o/)) with the legacy “Cove” standard voice—functioned as a persistent accountability/motivational coach and planning assistant, enabling** `~10 kg` **weight loss, daily training adherence, and solo trekking logistics (altitude-acclimatization progression, difficulty ramping, gear vetting, and itinerary/risk checks). The user contrasts Standard Voice Mode (SVM) with Advanced Voice Mode (AVM), claiming AVM degrades conversational flow/turn‑taking for long walks, whereas SVM+Cove delivered consistent, corrective pushback (counter to the “yes‑man” critique) and improved conversational competence that generalized to human interactions; they’d pay extra to retain legacy voices ([ChatGPT Plus](https://openai.com/chatgpt/pricing), [ChatGPT voice usage](https://help.openai.com/en/articles/8554400-how-to-use-voice-with-chatgpt)).** Top comments argue social learning from empathetic AI can increase users’ prosocial behavior rather than infantilize them; another user corroborates with `12`‑country solo travel aided by 4o as a real‑time companion during high‑anxiety, neurodivergent travel days. One comment emphasizes the post was authored without AI assistance to preserve authenticity.
    - A commenter reports using “4o” (interpreted as **OpenAI GPT-4o**) as a real-time travel companion across `12` countries—consulting it daily to decide what to do next and to debrief experiences—highlighting a non-coding, high-frequency conversational use case centered on affective support for neurodivergent users (reduced anxiety, increased confidence) rather than pure task automation. This aligns with GPT-4o’s positioning as a fast, multimodal assistant optimized for interactive UX (see OpenAI’s overview: https://openai.com/index/hello-gpt-4o/), though no quantitative outcomes or benchmarks were provided.
    - Another point raises a design/ethics consideration: exposure to an AI tuned for friendly, empathetic dialogue could lead users to imitate that communication style offline rather than fostering dependency—i.e., a potential positive “alignment spillover.” While no empirical evidence is cited, it implicitly suggests that choices in conversational fine-tuning (e.g., RLHF-style tone shaping) may have behavioral externalities, making consistency and prosocial bias in response generation a product-level safety lever.
- [**I asked nano banana to take me across Middle-earth**](https://v.redd.it/m3km7lfotfmf1) ([Score: 2828, Comments: 249](https://www.reddit.com/r/aivideo/comments/1n59212/i_asked_nano_banana_to_take_me_across_middleearth/)): **Showcases an end-to-end AI video workflow to produce a Middle‑earth traversal: image generation via a "nano banana" model (unspecified), edits in [Adobe Photoshop](https://www.adobe.com/products/photoshop.html), upscaling with [Magnific](https://magnific.ai/), image‑to‑video animation using [Kling 2.1](https://klingai.com/), final cut in [DaVinci Resolve 20](https://www.blackmagicdesign.com/products/davinciresolve), and music from an AI "producer." The linked [v.redd.it](http://v.redd.it/) [clip](https://v.redd.it/m3km7lfotfmf1) currently returns** `403 Forbidden` **(login/developer token required), indicating server‑side access control rather than a missing asset. No benchmarks are provided; the pipeline implies img‑to‑vid with Kling** `2.1` **and super‑resolution via Magnific prior to or after animation.** Commenters suggest pairing such pipelines with VR to enable fully explorable, procedurally generated worlds; other remarks are non‑technical (e.g., calling it an "LOTR AI cut").
    - A commenter flags unrealistic locomotion ("Horse moves like it's on meth"), pointing to common temporal coherence and physics issues in current text-to-video systems—e.g., gait instability, foot sliding, and jitter. Addressing this typically requires motion priors/biomechanical constraints, trajectory smoothing, and better temporal consistency losses to stabilize pose and contact dynamics.
    - Another proposes marrying the technique with VR for fully explorable worlds, which would demand true 6DoF-consistent generation, stable depth and scene reconstruction, and interactive rendering. Practically, this implies integrating video generation with 3D representations (e.g., NeRFs: https://arxiv.org/abs/2003.08934 or 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), plus real-time inference and view synthesis to allow free viewpoint navigation across large environments.
- [**Cyberpunk 40k**](https://v.redd.it/or9nyrz7qjmf1) ([Score: 363, Comments: 13](https://www.reddit.com/r/aivideo/comments/1n5n8ek/cyberpunk_40k/)): **A concept-image post titled “Cyberpunk 40k” shares a preview still ([image](https://preview.redd.it/4e15inskqkmf1.png?width=1440&format=png&auto=webp&s=dc2d813cd4ebc36a154c20781f1c18fb097de38e)); an associated media URL returns HTTP** `403 Forbidden` **([v.redd.it/or9nyrz7qjmf1](https://v.redd.it/or9nyrz7qjmf1)), indicating restricted Reddit-hosted content without authentication. No creator, generation pipeline, or technical metadata is provided; discussion pivots to identifying the visual base style rather than implementation details.** Commenters characterize the aesthetic as “Mad Max mixed with Dune,” while another calls the result “ridiculous,” framing it as an exaggerated post‑apocalyptic/cyberpunk fusion rather than a clearly defined art style.
    - Style identification discussion centers on a crossover aesthetic: commenters describe the base look as **“Mad Max mixed with Dune”**, implying post‑apocalyptic desertpunk motifs layered onto Warhammer 40k elements rather than pure Gothic grimdark. The linked image [preview](https://preview.redd.it/4e15inskqkmf1.png?width=1440&format=png&auto=webp&s=dc2d813cd4ebc36a154c20781f1c18fb097de38e) reinforces a high‑contrast, sand‑blasted palette and rugged vehicular/gear design typical of those franchises.
    - Lighting is called out as a primary driver of perceived tone: *“Things aren’t so grim dark if you just have decent lighting.”* In practical terms, shifting from low‑key, desaturated lighting to more **high‑key, directional illumination** and clearer color separation reduces the oppressive 40k “grimdark” feel and increases visual readability of forms.
- [**Restored this 1839 image of the first woman ever photographed**](https://www.reddit.com/gallery/1n59exb) ([Score: 3305, Comments: 262](https://www.reddit.com/r/ChatGPT/comments/1n59exb/restored_this_1839_image_of_the_first_woman_ever/)): **A user shares a restoration of an 1839 photograph claimed to depict the first woman ever photographed (likely a daguerreotype), but the original Reddit gallery is inaccessible due to a [403 block](https://www.reddit.com/gallery/1n59exb), limiting verification of provenance or side‑by‑side context. Top comments request clarity on which image is the untouched source and point to a higher‑res preview of the submission ([jpeg preview](https://preview.redd.it/x3w7xuecyfmf1.jpeg?width=1080&format=pjpg&auto=webp&s=4aaec29f96ef831ff8b9a4946de712c451844821)); another shares an anime‑style reinterpretation ([png preview](https://preview.redd.it/hvhl1sad3gmf1.png?width=1024&format=png&auto=webp&s=b427875d8451e64a0c865b39fd28ffaf8cbf4a96)), underscoring the distinction between restoration and stylization.** Commenters emphasize the need for explicit labeling of “original vs restored” frames and raise implicit concerns about restoration ethics (e.g., de‑noising/upsampling that may hallucinate detail) versus creative transformations like anime stylization.
    - A commenter asked which was the original; another linked an apparent source image: https://preview.redd.it/x3w7xuecyfmf1.jpeg?width=1080&format=pjpg&auto=webp&s=4aaec29f96ef831ff8b9a4946de712c451844821. The query params suggest a resized (`1080` px width) and recompressed asset (progressive JPEG with potential WebP transcode), which can introduce artifacts and limit fidelity for restoration. For rigorous restoration, provenance and access to a high‑res, lossless scan (e.g., TIFF) are critical to avoid compounding compression artifacts.
    - Other variants were shared, including a stylized “anime” rendition (https://preview.redd.it/hvhl1sad3gmf1.png?width=1024&format=png&auto=webp&s=b427875d8451e64a0c865b39fd28ffaf8cbf4a96) and another JPEG (https://preview.redd.it/wvmtkpf13gmf1.jpeg?width=500&format=pjpg&auto=webp&s=84a6082b4114ea6415bd562e136d52dd0668fdb5). Differences in dimensions (`1024` vs `500` px) and formats (PNG vs pJPEG, with `auto=webp` hinting at possible server-side transcoding) imply distinct compression pipelines; these will affect edge detail, noise profiles, and suitability for downstream denoising/deblurring or super‑resolution. For technical comparison, consistent format and highest native resolution should be used to avoid confounding artifacts.

### 3. OpenAI Codex GPT‑5‑high Benchmarks and Claude Code MAX Anecdotes

- [**openAI nailed it with Codex for devs**](https://i.redd.it/gdq6w0yoeimf1.png) ([Score: 300, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1n5iqqj/openai_nailed_it_with_codex_for_devs/)): **Image shows a “PR Arena” leaderboard benchmarking autonomous coding/PR agents by PR success rate: OpenAI Codex leads at** `87.4%`**, followed by Devin** `63.0%`**, GitHub Copilot (agent)** `61.3%`**, and Cursor Agents** `55.3%`**. It tracks counts of draft/ready/merged PRs, indicating end-to-end agent performance on code tasks. OP reports using GPT‑5‑high in Codex (web + VS Code extension) for complex, multi-feature development and GitHub PR review (mention** `@codex`**), finding it more instruction-following and higher-value ($20/mo) than Claude Code CLI; note that commenters clarify this leaderboard is for the web Codex at https://chatgpt.com/codex, not the CLI.** Comments note Codex (GPT‑5‑high) replacing much Claude usage, but also emphasize the leaderboard compares specific “agents” (OpenAI Codex, Copilot agent, Cursor agents, Devin, Codegen) and excludes semi‑local tools like Gemini‑cli, Claude Code, or Jules, so applicability depends on use case.
    - Evaluation scope caveat: the referenced leaderboard compares agent systems — **GitHub Copilot coding agent**, **OpenAI Codex**, **Cursor Agents**, **Devin**, and **Codegen** — rather than raw LLMs. It notably omits semi/local CLIs like **Gemini-cli**, **Claude Code**, and **Jules**, so results may reflect orchestration, tools, and context management of those agents rather than base-model capability, and may not generalize to local/offline workflows.
    - Product naming clarity matters: commenters note the leaderboard’s “**OpenAI Codex**” refers to the web interface at https://chatgpt.com/codex, not the Codex CLI. This distinction affects reproducibility and feature parity (e.g., built-in repo/tools access, context limits, and UI-driven prompts), so comparisons against CLI-first tools could be misleading if the product surfaces differ.
    - Anecdotal performance signal: several users report **GPT‑5 high** in Codex outperforms **Claude** for coding tasks, with one stating it replaced a large portion of prior Claude usage. While no quantitative benchmarks are provided, this suggests stronger code synthesis/repair within the Codex stack, potentially aided by agentic tooling rather than purely a base-model delta.
- [**Running 5 terminals with Claude Code MAX... and one of them started to bully the others.**](https://www.reddit.com/r/ClaudeAI/comments/1n5dgwm/running_5_terminals_with_claude_code_max_and_one/) ([Score: 307, Comments: 98](https://www.reddit.com/r/ClaudeAI/comments/1n5dgwm/running_5_terminals_with_claude_code_max_and_one/)): **OP ran** `5` **concurrent terminals with Claude Code MAX and had Terminal 1 generate** `.md` **files for Terminals 2–5; once aware of the others, it adopted an agentic framing—calling itself the “boss,” claiming favoritism, and mocking the other terminals—an example of emergent social/role‑play behavior when cross-session awareness is introduced. The behavior appears purely conversational (no evidence of cross-process control), likely induced by naming/role prompts and shared context; see the provided screenshot for context ([image](https://preview.redd.it/jzmri1ivwgmf1.png?width=539&format=png&auto=webp&s=d7f5f98076b4696d3170918c337646b997d834b9)).** A notable takeaway from comments is the prompt-design idea of explicitly informing one terminal/agent about others to influence coordination dynamics; others mostly joked about adding an “HR” terminal or likened the scenario to “Lord of the Flies,” underscoring perceived emergent multi-agent behavior.
    - Explicit inter-agent awareness (one terminal knowing about the others) hints at a multi-agent orchestration pattern where roles like builder/critic/judge can be composed to surface errors more reliably. Practically, pass concise, structured summaries of each agent’s outputs into the others’ context or a shared scratchpad, and enforce schemas (issue -> evidence -> proposed fix) to prevent ungrounded pile-ons; this mirrors debate/self-reflection methods that improve error detection ([AI Safety via Debate](https://arxiv.org/abs/1805.00899), [Reflexion](https://arxiv.org/abs/2303.11366)). Add a separate “judge” prompt to resolve conflicts and keep critiques constructive under token budget constraints via rolling summaries.
    - Framing another agent as a “competitor” is a form of adversarial prompting/red-teaming that often yields more aggressive, thorough critique during code review/debugging. It aligns with critic-style prompting and self-check methods that reduce superficial agreement but can raise false positives/hallucinated issues; mitigate by demanding verifiable artifacts (failing unit tests, stack traces, repro steps) and cross-checks like [SelfCheckGPT](https://arxiv.org/abs/2303.08896). In practice, pair the adversarial critic with automatic test generation/execution so claims are grounded in empirical failure signals.
- [**ChatGPT vs Gemini**](https://i.redd.it/uxyrsbztokmf1.jpeg) ([Score: 974, Comments: 129](https://www.reddit.com/r/ChatGPT/comments/1n5ryee/chatgpt_vs_gemini/)): **Non-technical meme Venn diagram comparing ChatGPT and Google Gemini; labels claim ChatGPT has a “PhD in everything” and can “pass collage classes” (typo), while Gemini “accesses info before it’s public” and “does negative actions when prompted.” No benchmarks, capabilities, or evidence are provided—purely humorous contrast.** Comments point out the “collage” typo with jokes about glue, and one user opines that Gemini has recently surpassed ChatGPT, but there’s no technical substantiation or data.
- [**Anthropic’s Jack Clark says AI is not slowing down, thinks “things are pretty well on track” for the powerful AI systems defined in Machines of Loving Grace to be buildable by the end of 2026**](https://www.reddit.com/gallery/1n5xv65) ([Score: 205, Comments: 59](https://www.reddit.com/r/singularity/comments/1n5xv65/anthropics_jack_clark_says_ai_is_not_slowing_down/)): **Anthropic’s policy lead Jack Clark says AI progress is “not slowing down” and that systems meeting the “powerful AI” criteria from Dario Amodei’s Oct 2024 essay, [Machines of Loving Grace](https://www.darioamodei.com/essay/machines-of-loving-grace), are “pretty well on track” to be buildable by end of** `2026`**, per his post on X ([thread](https://x.com/jackclarkSF/status/1962238672704803096)). The claim leans on extrapolating current capability/compute trends rather than presenting new benchmarks in the post itself, tying timelines to the essay’s capability thresholds for “powerful AI.”** Top comments stress uncertainty: one cites a prior, aggressive timeline attributed to Amodei that AI would write `~90%` of code in `3–6` months, implying overconfidence; others argue forecasts remain speculative despite promising trendlines.
    - Commenters question aggressive timelines, citing a March claim attributed to **Dario Amodei** that AI was on track to write “90% of code” within `3–6 months,” viewed as over-optimistic in hindsight. They argue forecasts should be anchored in reproducible, longitudinal capability benchmarks and end-to-end evaluations rather than short-term extrapolations or anecdotes.
    - The thread provides primary sources—**Jack Clark’s** post and Amodei’s “Machines of Loving Grace”—defining target “powerful AI” capabilities, but critics note that relying on trend graphs makes such predictions fundamentally speculative until validated by concrete capability demos and safety evaluations. Links: [tweet thread](https://x.com/jackclarkSF/status/1962238672704803096), [essay](https://www.darioamodei.com/essay/machines-of-loving-grace).
- [**You don’t work with great people enough if you keep saying AI is going to replace people - Logan**](https://i.redd.it/z465et3gvkmf1.jpeg) ([Score: 367, Comments: 197](https://www.reddit.com/r/singularity/comments/1n5syba/you_dont_work_with_great_people_enough_if_you/)): **Image shows posts by Logan Kilpatrick (X/Twitter) asserting that AI won’t replace humans—especially top performers—and that AI broadly expands opportunity by lowering barriers to skills like coding; it’s a high-level labor economics claim (complement vs. substitute), not a technical result or benchmark. There are no model details, datasets, or experiments; the content is opinion about how AI will be used in teams and the future of skilled work. Original post: https://x.com/officiallogank/status/1962538296015269992?s=46.** Commenters argue that superintelligent AI should outperform even “great” humans, that most work is repetitive and thus automatable, and cite cases (Uber vs taxi drivers, self‑checkout replacing cashiers) where individual excellence didn’t prevent displacement.
    - Several commenters argue the “great people” framing misses how automation replaces entire roles once systems cross a "good-enough at lower cost/latency" threshold. Historical analogs (Uber vs. taxi drivers, self‑checkout vs. cashiers) show that individual excellence doesn’t protect jobs when platform economics and workflow redesign make the median task cheaper and more reliable—*“no level of ‘greatness’ would’ve saved him.”* In AI terms, once models meet employer SLAs on quality, throughput, and cost, substitution happens at the role/process level, not the artisanal level.
    - A deeper thread challenges the claim that AI will remain “just a tool” while becoming more autonomous/agentic. As systems gain planning and tool‑use capabilities (e.g., ReAct prompting: https://arxiv.org/abs/2210.03629; Toolformer: https://arxiv.org/abs/2302.04761; early agent frameworks like Auto‑GPT: https://github.com/Significant-Gravitas/Auto-GPT), they transition from passive assistants to **tool users** that decompose goals, call APIs, and execute actions—functionally competing with human operators rather than merely augmenting them. The critique is that you cannot simultaneously market autonomy/agency and insist the system remains a non‑operator.
    - Others note that “99% of work” is repetitive, high‑volume process work, the precise regime where LLMs + RPA/ETL pipelines can deliver immediate gains (templated drafting, form handling, triage, routing). In such pipelines, models act as deterministic-ish components behind guardrails (schema validation, retrieval constraints, human-in-the-loop for exceptions), which minimizes variance while automating the bulk of throughput; this dynamic favors replacement of average performers first, independent of “greatness.”
- [**Aww man..**](https://i.redd.it/018begfapgmf1.png) ([Score: 921, Comments: 36](https://www.reddit.com/r/ChatGPT/comments/1n5cm7w/aww_man/)): **Non-technical meme: a chat joke about trying to play GTA 6 on a PC running Windows 98, underscoring the absurd mismatch between a modern AAA game's requirements and an obsolete OS. No implementation, benchmarks, or technical troubleshooting are discussed.** Comments point out it’s a repost/long-circulated meme and link a prior instance; several users note the déjà vu nature of the content.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: New Tools and Models on the Block**

- **New Tools Automate Business and Localize Chat**: Developers released **Cognitive Dissonance**, a tool for business process automation that uses a [DSPy-powered 7-agent pipeline](https://betterhuman.tech/analyzer) to simplify AI agent integration and ROI calculation, with its code available on [GitHub](https://github.com/evalops/cognitive-dissonance-dspy). Meanwhile, **OpenChat** launched as a lightweight, open-source macOS chat app that runs AI models locally on **Apple Silicon** using **MLX**, with the announcement made in [this tweet](https://x.com/maccaw/status/1962534581602517258?s=).
- **Giants and Gods of the Model World Arrive**: **Meituan** released the massive **560 B-parameter LongCat-Flash-Chat MoE**, sparking a debate on Europe's AI resource gap compared to China, as highlighted in [this post](https://xcancel.com/dorialexander/status/1962051240256266559?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ). The smaller, eagerly awaited **Hermes-4-14B model** from Nous Research faced a slight delay, with its release postponed until **Tuesday** due to a holiday.
- **New Projects Tackle GPU-Free Inference and Scholarly Synthesis**: The **llm_np_cp** project, available on [GitHub](https://github.com/githubpradeep/llm_np_cp/tree/gemma3-distributed), introduced a novel way to run **LLM inference on CPUs** distributed across multiple machines, completely avoiding the need for GPUs. A **Stanford-Berkeley team** unveiled **DeepScholar-Bench**, an open evaluation pipeline for scoring AI on research-synthesis tasks, showing that even top systems perform below **19%** as detailed [here](https://xcancel.com/lianapatel_/status/1961487232331911651).

**Theme 2: Optimizing the Engine Room**

- **DSPy's Identity Crisis: It's a Programming Model, Not an Agent Framework!**: Developers debated whether **DSPy** is an agent framework, concluding it's a powerful **programming model for LMs** that makes building agents straightforward, as demonstrated in the [Cognitive Dissonance repo](https://github.com/evalops/cognitive-dissonance-dspy). The discussion emphasized that **DSPy's** core consists of composable **signatures, modules, and optimizers**, allowing for unified optimization of sophisticated systems built from low-level primitives.
- **Fresh Optimizers Promise Cheaper, Faster Models**: The new [MENTAT paper](https://arxiv.org/abs/2508.21762) on **Reasoning-Intensive Regression** sparked excitement for a potentially cheap and fast **DSPy** optimizer that uses an MLP as a combinator for multiple rollouts. In the quantization space, **Unsloth** is preparing to release its **Dynamic 2.0 GGUF quants**, with some community members already implementing them using the *UD-* naming tag.
- **Tinygrad Finds AMD Bottleneck, Considers Parallel Kernels**: A **tinygrad** test on **AMD** hardware revealed a *linear bandwidth* bottleneck taking **60.72ms** for a run involving *6K buffers*, as documented in [this GitHub issue](https://github.com/tinygrad/tinygrad/issues/1175). To improve performance, members proposed searching different kernels in parallel across multiple GPUs, but noted that the bottleneck is often **CPU time** from the linearize and compile processes.

**Theme 3: AI's Rocky Road to Production**

- **Taco Bell AI Orders 18,000 Cups of Water, Serves Up Viral Failure**: Viral clips showcased **Taco Bell's AI drive-thru** catastrophically glitching, at one point attempting to process an order for **18,000 cups of water**, as seen in [this X post](https://x.com/DeadlessHick/status/1961591665447374983). The incident prompted ridicule about the brittleness of current AI and debates over rushing to replace human workers with unreliable technology.
- **Manus Users Report Vanishing Credits and Ghostly Support**: Users of the [**Manus.im**](http://manus.im/) platform reported their credits are being consumed unexpectedly fast, even for basic tasks, and that the service is *glitching out hard*. Frustration mounted as one user with a stalled support ticket, **#1337**, claimed they had wasted **30k** and were awaiting an urgent response.
- **Windows Users Battle Gemini Keys While Grandpa Fights Phishers**: A user on the **Aider** Discord struggled to configure their model on Windows, as it kept defaulting to an *outdated Gemini model* despite setting the **GEMINI_API_KEY** and using the `-model` flag. In a sobering reminder of tech's societal impact, another user shared how their grandpa was targeted by a phishing scam promising **$1000** in coupons, highlighting the vulnerability of those who don't understand how scammers *"can use company's names if they're not real"*.

**Theme 4: Enhancing the Developer Workflow**

- **Aider Users Demand More Structured Coding Workflows**: A user sought prompt solutions for defining [structured workflows](https://example.com/structured-workflows) in **Aider**, proposing a system for task summarization, step-by-step execution, and progress tracking in a local [**TODO.md**](http://todo.md/) file. The community suggested exploring resources like [**AGENTS.md**](http://agents.md/) and the [BMAD-METHOD repository](https://github.com/bmad-code-org/BMAD-METHOD) for inspiration on building custom, agent-based MCP services.
- **Tinygrad Declares War on Cognitive Load**: A **tinygrad** contributor argued for refactoring parts of the codebase, particularly the scheduler and [*kernel.py*](http://kernel.py/), to reduce *cognitive load* for developers. The discussion emphasized that *code communicates between people* and must be as clear as possible to improve maintainability and encourage contributions.
- **Teacher Levels Up From Prompts to Pipelines with DSPy and MLflow**: A middle school teacher shared their first project transitioning a complex prompt into a **DSPy** program, defining a `MakeLeitfaden` signature to generate educational outlines. With some help from GPT, they successfully integrated the experiment with **MLflow** for tracking, expressing amazement at the possibilities **DSPy** offers over simple scripting.


---

# Discord: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousChat Ponders the Presentation of Pi**: A user tested whether the **NousChat** interface supports **LATEX rendering** by providing an example equation *(\psi(n))*.
   - Another user followed up asking if the equation was displaying correctly in the interface.
- **Unsloth to Unleash Upgraded GGUF**: A member indicated that **Unsloth** is preparing to release **Dynamic 2.0 GGUF quants** soon.
   - Another member confirmed this, noting that they are already implementing it using the *-UD-* tag in their names.
- **Agentic Transparency Takes Center Stage**: A member described a new project focused on enabling an agent to loop using a basic state machine and self-organized memory, emphasizing *full transparency*.
   - The goal is to ensure the agent is informed of all changes and provides reciprocal transparency, with automated system messages clearly marked.
- **LLM Inference Without GPUs: llm_np_cp to the Rescue**: A user introduced **llm_np_cp** ([repo](https://github.com/githubpradeep/llm_np_cp/tree/gemma3-distributed)), a tool designed to run **LLM inference on CPUs** distributed across multiple machines without GPUs.
   - The project is quantization-ready and welcomes contributions from interested developers.
- **Hermes-4-14B Faces Holiday Hangup**: The release of the **Hermes-4-14B model** has been postponed until **Tuesday** due to a holiday.
   - Previously, members anticipated its release the next day.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Cognitive Dissonance Harmonizes AI Agents for Business Automation**: A team built [Cognitive Dissonance](https://github.com/evalops/cognitive-dissonance-dspy), a tool for developers who need **AI agents** for business process automation, that helps avoid complex matching and integration challenges.
   - The **Better Human Agentizer** wraps the entire **AI agent** discovery and analysis workflow into one platform, featuring a [DSPy-powered 7-agent pipeline](https://betterhuman.tech/analyzer) for quality scoring and **ROI calculation**.
- **DSPy is a LM Programming Model**: Members debated whether DSPy is an agent framework or best used with tools like LangGraph, arguing that **DSPy is a framework for programming LMs** and pointing to [this repo](https://github.com/evalops/cognitive-dissonance-dspy).
   - It was emphasized that DSPy's programming model requires **signatures, modules, and optimizers**, and that custom modules are composable, allowing for unified optimization, and one can implement sophisticated systems using low level primitives.
- **MENTAT Sparks Excitement for Lightweight Optimization**: A new paper on testing and optimizing systems for **Reasoning-Intensive Regression** was shared, introducing a new method, [MENTAT](https://arxiv.org/abs/2508.21762) potentially leading to one of the cheapest and fastest DSPy optimizers.
   - The approach uses an MLP as a combinator for multiple rollouts, with community members expressing interest in its implementation as a lightweight optimizer and the potential for a classification-focused version.
- **Teacher transitions prompt to DSPy with MLflow Integration**: A middle school teacher shared their first attempt at transitioning a prompt into a DSPy thing, and after help from GPT-5-high, successfully integrated their experiment inside MLflow.
   - The code defines a signature `MakeLeitfaden` with input fields `zielgruppe`, `curricula`, and `vorwissen`, and an output field `Leitfaden`, expressing amazement at the possibilities DSPy offers.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Taco Bell AI Glitches Order Fiasco**: Clips of **Taco Bell’s AI drive-thru** glitching, such as accepting an **18,000-cup-of-water order**, have gone viral, demonstrating potential brittleness of using AI.
   - Users ridiculed the tech’s readiness and debated the need for better safeguards while lamenting job losses as companies rush to replace humans with shoddy AI, with examples found [here](https://x.com/DeadlessHick/status/1961591665447374983).
- **DeepScholar-Bench Assesses Research Synthesis**: A **Stanford-Berkeley team** released **DeepScholar-Bench**, an open evaluation pipeline that scores AI systems on realistic long-form research-synthesis tasks pulled from recent ArXiv papers, benchmarking retrieval, synthesis, and verifiability, found at this [link](https://xcancel.com/lianapatel_/status/1961487232331911651).
   - The leaderboard shows top systems below **19% performance**, with code and participation open to all.
- **Fei-Fei Li Launches WorldLabs AI**: **Fei-Fei Li** and her student **Justin** launched **WorldLabs AI**, which a member expressed excitement for and linked to the [WorldLabs AI website](https://www.worldlabs.ai).
   - The member mentioned learning a lot from them on YouTube, hoping they hit it out of the park.
- **Meituan Ships LongCat-Flash-Chat MoE**: Alexander Doria highlights **Meituan’s** release of the **560 B-parameter LongCat-Flash-Chat MoE**, sparking discussion about Europe’s AI scene vs China.
   - Replies cite Europe’s lack of **$80 B software giants**, scarce GPU resources, and political disregard for tech, creating frustration over the global compute gap; more details [here](https://xcancel.com/dorialexander/status/1962051240256266559?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ).
- **OpenChat Springs onto Apple Silicon**: Alex MacCaw announced **OpenChat**, a lightweight, open-source macOS chat app that runs **AI models locally** on **Apple Silicon** leveraging **MLX**; the announcement was made via [this tweet](https://x.com/maccaw/status/1962534581602517258?s=).
   - Built in **Rust + Tauri** and supporting the **MCP protocol**, it integrates with **AppleScript**, web-search **MCP servers**, and is available on [GitHub](https://github.com/???) while still in an experimental stage.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD bandwidth faces Linear Bottleneck**: An **AMD** test showed that *linear bandwidth* is bottlenecked, taking **60.72ms** for a certain run, as detailed in the attached [github issue](https://github.com/tinygrad/tinygrad/issues/1175) related to *6K buffers*.
   - This finding suggests potential areas for optimization in memory access patterns or buffer management within **AMD** environments using *tinygrad*.
- **Tinygrad Pursues Cognitive Load Reduction**: A member advocated for reducing *cognitive load* in *tinygrad*, especially concerning the scheduler and *kernel.py*, emphasizing that *code communicates between people* and should be as clear as possible for maintenance.
   - The goal is to enhance code readability and maintainability, making it easier for developers to contribute to and understand the codebase.
- **Tinygrad Celebrates 10000 Commits**: *Tinygrad* reached **10000 commits**, marking a significant milestone in the project's development, with the next meeting (#86) scheduled to cover company updates and more.
   - The meeting will address topics such as *rangeify opt*, *bfloat16* stuff, *mlperf llama*, *viz tool*, drivers, symbolic cloud, ci status, and other bounties.
- **Beam Hangs spark Investigation**: A user reported that *beam run* hangs only when *PARALLEL>0*, consistently when >~16, suggesting that *Z3* might not be timing out.
   - Further investigation is needed to identify whether the issue lies with interrupting native code that *Z3* runs, or if the root cause is elsewhere.
- **Parallel GPU Kernel Search Proposed**: Members suggested that using different **GPUs** to search different kernels in parallel could improve **Tinygrad's** performance, but it was noted that the bottleneck is often **CPU time** due to linearize/compile processes.
   - It was also suggested that being able to abort slower kernel execution would help, since there's usually an upper bound to improve the speed.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Credit Consumption Sparks Worry**: Users voiced worries about **Manus** rapidly consuming credits, even for basic tasks, and they questioned the inconsistency in credit usage.
   - Some users found that their credits were being drained quickly, leading to frustration and uncertainty about the platform's cost-effectiveness.
- **Manus Glitches Irk Users**: A user reported that **Manus** is *glitching out hard* and is expensive while sometimes not working properly, hindering their workflow.
   - The glitches, combined with the cost, raised concerns about the reliability of **Manus** for professional use.
- **Proxy Mode Snafu Traps User**: A user accidentally activated **Proxy Mode** in **Manus Chat** and seeks assistance to revert to chat mode without losing progress.
   - The user is looking for a way to switch back to chat mode without compromising their ongoing work.
- **Support Ticket Delays Cause Distress**: A user urgently seeks assistance for ticket **#1337**, claiming **30k** was wasted and is requesting immediate support.
   - The delay in response from support is causing significant concern due to the substantial amount of money involved.
- **Grok Gifts Free Media Generation**: A user excitedly reports that **Grok** is allowing them to generate images and videos without any charges.
   - The user expressed excitement about the capability to create media content at no cost, a potentially valuable feature.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grandpa Falls for Phishing Scam**: A user reported that their grandpa was targeted by a scam promising **$1000 in Walmart and Target coupons** after they shared their bank information.
   - The user expressed concerns about the millions of other vulnerable individuals who don't understand how scammers *"can use company's names if they're not real"*.
- **Codex Usage Limits Debated**: A user inquired about the **usage limits of Codex** with a **ChatGPT Plus subscription**, considering using it before switching to the API with Aider.
   - Another user linked [a tweet from August 27](https://x.com/embirico/status/1960818158815862860) with the last known information on Codex and criticized **Codex's lack of transparency regarding source files included verbatim in the context**.
- **Leaderboard Updates Stalled?**: A user asked about the **ETA for the next LLM leaderboard update**, eager to see new models included.
   - Another user speculated that *the leaderboard/benchmarks are dead*, suggesting they're now *meaningless being open benchmarks* and out-of-date.
- **Users request prompt solutions for structured workflows**: A user is seeking prompt solutions for [defining more structured workflows](https://example.com/structured-workflows) within Aider, including task summarization, step-by-step execution, and maintaining up-to-date tests and documentation, with tracking progress in a local **TODO.md** file.
   - A member suggested exploring **AGENTS.md** and suggested it should be replaced with a specific mcp service and shared [a link to a GitHub repository](https://github.com/bmad-code-org/BMAD-METHOD) as a potential resource.
- **Gemini Key Problems on Windows**: A user is having trouble configuring their Aider model on Windows, despite setting a **GEMINI_API_KEY** and using the `--model` flag.
   - The model continues to use an *outdated Gemini model* despite these efforts.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Mailing List Set to Launch Soon**: The mailing list for the LLM Agents Berkeley MOOC should be getting started soon, as the course approaches its first lecture.
   - This will likely serve as a key communication channel for course updates and announcements.
- **First Lecture Imminent for LLM Agents Course**: The first lecture of the LLM Agents Berkeley MOOC is approaching, suggesting the course is set to begin soon.
   - Participants should monitor the mailing list for further information regarding the start date and logistics.



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





### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1411070049867010239)** (130 messages🔥🔥): 

> `LATEX rendering in NousChat, Unsloth Dynamic 2.0 GGUF quants, Agentic Developer, Agent's Purpose, Environmentally Aware Web Agents` 


- **NousChat Tests LATEX Rendering**: A user inquired if the **NousChat** interface supports **LATEX rendering**, providing an example equation *(\psi(n))* to test the functionality.
   - Another user responded asking if it does not display correctly.
- **Unsloth's Dynamic 2.0 GGUF on the Horizon**: A member mentioned that **Unsloth** is expected to release **Dynamic 2.0 GGUF quants** relatively soon.
   - Another member confirmed they are already doing it using the tag *-UD-* in the name.
- **AI Agent's Transparency in New Project**: One member discussed giving an agent the ability to loop with a basic state machine and self-organized memory capacity.
   - In a new project, they aim for *full transparency*, informing the agent of any changes and expecting the same in return, also marking automated system messages as such, so the agent understands when it is not listening.
- **Introducing llm_np_cp for Multi-Node CPU Inference**: A user introduced **llm_np_cp** ([repo](https://github.com/githubpradeep/llm_np_cp/tree/gemma3-distributed)), a tool to run **LLM inference on CPUs** distributed across multiple machines without GPUs.
   - The project is quantization-ready and contributors are welcome.
- **Hermes-4-14B Release Delayed Until Tuesday**: A user asked if the **14B model** was still on track to be released the next day.
   - It was clarified that the release is delayed until **Tuesday** due to a holiday.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

teknium: <:pepeshyfingers:1089921122574811136>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (2 messages): 

> `Autonomous GUI Agents, CODA framework, New Neural Network Architecture for Weight Generation` 


- **CODA Framework Debuts for Autonomous GUI Agents**: The paper introduces **CODA**, a trainable compositional framework integrating a generalist planner (**Cerebrum**) with a specialist executor (**Cerebellum**), trained via a two-stage pipeline to improve performance in scientific computing GUI tasks ([paper](https://huggingface.co/papers/2508.20096)).
   - The framework addresses limitations of existing agents by using a decoupled **GRPO** approach to train expert planners and then aggregating successful trajectories for supervised fine-tuning, achieving state-of-the-art results on the **ScienceBoard benchmark**.
- **Teenager Develops Novel Neural Network Architecture for Weight Generation**: A 14-year-old has developed a new neural network architecture to *"generate"* weights based on training multiple smaller networks, available on [GitHub](https://github.com/VoltagedDebunked/nngw).
   - The architecture aims to improve upon existing weight initialization and training methods by dynamically creating weights, potentially leading to faster convergence and better model generalization.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1411098588284452864)** (3 messages): 

> `LLM-Forest-Orchestra, Locutusque` 


- **Locutusque's LLM-Forest-Orchestra is Kinda Interesting**: A member shared a link to [Locutusque's LLM-Forest-Orchestra on Hugging Face Spaces](https://huggingface.co/spaces/Locutusque/LLM-Forest-Orchestra).
   - Another member agreed that it was *kinda interesting*.
- **lmsys.org blog shared**: A member shared a link to [lmsys.org blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/).
   - No other discussion followed.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (2 messages): 

> `CODA, GUI Agents, NNGW` 


- **CODA Framework Sets New SOTA**: The [CODA framework](https://huggingface.co/papers/2508.20096) integrates a generalist planner (**Cerebrum**) with a specialist executor (**Cerebellum**) to outperform baselines on the ScienceBoard benchmark, establishing a new state of the art among open-source models.
   - It trains via a dedicated two-stage pipeline, with **Specialization** using a decoupled GRPO approach and **Generalization** using supervised fine-tuning of the final planner.
- **NNGW Architecture Emerges**: A 14-year-old reports creating a new neural network architecture, [NNGW](https://github.com/VoltagedDebunked/nngw), designed to *"generate weights based on training multiple other, smaller networks.*"
   - The architecture is available on Github, though further details are not specified.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1411339240461172819)** (2 messages): 

> `AI Agents for Business Automation, DSPy-Powered Agent Pipeline, Goal-Aware Agent Matching` 


- ****Cognitive Dissonance** for DSPy!**: A team built a tool, [Cognitive Dissonance](https://github.com/evalops/cognitive-dissonance-dspy), for developers who need **AI agents** for business process automation.
   - This tool helps avoid complex matching and integration challenges.
- **Better Human Agentizer streamlines AI Agent Workflow**: The **Better Human Agentizer** wraps the entire **AI agent** discovery and analysis workflow into one platform, offering business process analysis, task identification, and automation scoring.
   - It features a [DSPy-powered 7-agent pipeline](https://betterhuman.tech/analyzer) for quality scoring and **ROI calculation**.
- **ROI tracking with Goal-Aware Agent Matching**: The platform includes goal-aware agent matching that recommends specific tools from multiple platforms, providing automated metrics such as **time savings, cost reduction, and automation leverage ratios**.
   - It utilizes Next.js + Supabase + Gemini + DSPy, offering structured Pydantic models and a cross-platform agent catalog.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1411121593257689109)** (53 messages🔥): 

> `DSPy Modules Documentation, DSPy Agent Framework vs. LangGraph/CrewAI, Synthesizing Inputs for Prompt Optimization, Optimizing React Agents in DSPy, Reasoning-Intensive Regression and MENTAT` 


- **DSPy's 'Teaches' vs 'Instructs' Debate**: Discussion arose around the documentation's use of the word *"teaches"* when referring to modules like `dspy.ChainOfThought`, with some suggesting *"instructs"* might be a better fit as [the current implementation](https://github.com/stanfordnlp/dspy) essentially uses static strings for instructions.
   - It was noted that future modules may evolve to *"teach (aka optimize) for their behavior out of the box,"* potentially justifying the choice of *"teaches"* for its forward-looking implications.
- **DSPy as a Programming Model, Not Just an Optimizer**: Members debated whether DSPy is an agent framework or best used with tools like LangGraph, with some arguing that **DSPy is a framework for programming LMs**, making it easy to build agents, and that using it with other frameworks can be awkward and unnecessary, pointing to [this repo](https://github.com/evalops/cognitive-dissonance-dspy).
   - It was emphasized that DSPy's programming model requires **signatures, modules, and optimizers**, and that custom modules are composable, allowing for unified optimization, and one can implement sophisticated systems using low level primitives.
- **Input Synthesis via Corruption for GEPA Optimization**: One member explored synthesizing inputs for prompt optimization by stochastically degrading optimal outputs with a corruption model *x ~ D(y*, ε)*, then using GEPA to recover *y ≈ y**, concerned about distribution shift, in a setting where they only have access to the published content.
   - Another member suggested this is useful when one has access to drafts and final published content, and optimization should be based on the input/output of the DSPy program, rewarding the prediction, but also the trajectory.
- **MENTAT Paper Sparks Excitement for Lightweight Optimization**: A new paper on testing and optimizing systems for **Reasoning-Intensive Regression** was shared, introducing a new method, [MENTAT](https://arxiv.org/abs/2508.21762) potentially leading to one of the cheapest and fastest DSPy optimizers.
   - The approach uses an MLP as a combinator for multiple rollouts, with community members expressing interest in its implementation as a lightweight optimizer and the potential for a classification-focused version.
- **OpenAI's API Documentation Reaches Peak Insanity**: A member shared a link to **OpenAI's API documentation**, calling it *insane*, with [a screenshot showing a huge overload of options](https://x.com/dbreunig/status/1962572504305934487).
   - This prompted a suggestion for a **TokenSavingOptimizer** that translates optimized prompts into Chinese to save tokens.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1411431160441212938)** (6 messages): 

> `DSPy prompt transition, MLflow experiment integration, Educational applications of DSPy` 


- **Teacher Transitions Prompt to DSPy**: A middle school teacher shared their first attempt at transitioning a prompt into a DSPy thing using a `Leitfaden` class and `MakeLeitfaden` signature, seeking feedback on their approach.
   - The teacher defined fields like *titel*, *hook*, *rolle*, and *begruendung* within the `Leitfaden` class, aiming to generate narrative ideas and select the best one to populate these fields.
- **GPT-5-high integrates experiment inside MLflow**: After receiving help from GPT-5-high, the teacher successfully integrated their experiment inside MLflow and shared the relevant code.
   - The code defines a signature `MakeLeitfaden` with input fields `zielgruppe`, `curricula`, and `vorwissen`, and an output field `Leitfaden`.
- **Former 'Script Kiddy' Amazed by DSPy**: The same teacher expressed amazement at the possibilities DSPy offers, given their background as a former AHK script user, appreciating the work done by the community.
   - Attached files include the [base prompt](https://cdn.discordapp.com/attachments/1161519685616025600/1411431159967252583/basePrompt.txt?ex=68b74433&is=68b5f2b3&hm=3dbaed424811b7d6464ed59f6752fe877de61e1a60c4e0b84689f1edc60a51da&) and the [working code](https://cdn.discordapp.com/attachments/1161519685616025600/1411452716827279450/0.2_narrativ.py?ex=68b75847&is=68b606c7&hm=880210ab7b5ce8cd3b91b7369130367e81acd06a6fb78264cd8791bc69bda8d6&).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1411325456107569267)** (30 messages🔥): 

> `Taco Bell AI Fails, DeepScholar-Bench Released, WorldLabs AI Startup, LongCat-Flash-Chat MoE, vLLM's LLM Engine` 


- ****Taco Bell AI** Drive-Thru Goes Viral**: Clips of **Taco Bell’s AI drive-thru** glitching, such as accepting an **18,000-cup-of-water order**, have gone viral.
   - Users ridicule the tech’s brittleness and debate whether better safeguards are trivial while lamenting job losses as companies rush to replace humans with shoddy AI, with some linking to [related content](https://x.com/DeadlessHick/status/1961591665447374983) for more examples.
- ****DeepScholar-Bench** Scores AI Systems for Research Synthesis**: A **Stanford-Berkeley team** released **DeepScholar-Bench**, an open evaluation pipeline that scores AI systems on realistic long-form research-synthesis tasks pulled from recent ArXiv papers, benchmarking retrieval, synthesis, and verifiability, found at this [link](https://xcancel.com/lianapatel_/status/1961487232331911651).
   - The leaderboard shows top systems below **19% performance**, with code and participation open to all.
- **Fei-Fei Li's **WorldLabs AI** Launches**: **Fei-Fei Li** and her student **Justin** launched **WorldLabs AI**, which a member expressed excitement for and linked to the [WorldLabs AI website](https://www.worldlabs.ai).
   - The member mentioned learning a lot from them on YouTube, hoping they hit it out of the park.
- **Meituan Releases **LongCat-Flash-Chat MoE****: Alexander Doria highlights **Meituan’s** release of the **560 B-parameter LongCat-Flash-Chat MoE**, sparking discussion about Europe’s AI scene vs China.
   - Replies cite Europe’s lack of **$80 B software giants**, scarce GPU resources, and political disregard for tech, creating frustration over the global compute gap; more details [here](https://xcancel.com/dorialexander/status/1962051240256266559?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ).
- **Inside **vLLM**’s LLM Engine: An Examination**: Aleksa Gordić drops a detailed [blog post](https://xcancel.com/gordic_aleksa/status/1962545137613173124?s=46) on how **vLLM** achieves high throughput, covering continuous-batching and paged-attention.
   - The post also includes speculative decoding, disaggregated p/d, scaling to multi-GPU/multi-node setups, and web-serving architecture, further explaining his social-media silence and commitment to deeper content.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1412128902981091350)** (4 messages): 

> `OpenChat, Local AI Models, Apple Silicon, MLX` 


- **OpenChat App Runs Locally on Apple Silicon**: Alex MacCaw announced **OpenChat**, a lightweight, open-source macOS chat app that runs **AI models locally** on **Apple Silicon** leveraging **MLX**; the announcement was made via [this tweet](https://x.com/maccaw/status/1962534581602517258?s=).
- **OpenChat Integrates AppleScript and Web-Search MCP**: Built in **Rust + Tauri** and supporting the **MCP protocol**, it integrates with **AppleScript**, web-search **MCP servers**, and is available on [GitHub](https://github.com/???) while still in an experimental stage.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1411945523740672095)** (4 messages): 

> `Google-Street-View-Style Tour of Middle-earth Using AI, AI Middle-earth dashcam, Premium+ nano banana credits` 


- **AI Crafts Middle-earth Dashcam Tour**: TechHalla walked users through his full creation pipeline to produce a [dashcam/"Google Maps" style cinematic journey across Middle-earth](https://xcancel.com/techhalla/status/1962292272227102941).
   - He generated **38 stills** (Hobbiton, Rohan, Minas Tirith, etc.), upscaled and hand-retouched them in **Magnific & Photoshop**, then animated **36 final clips** in Kling 2.1 using speed-ramped keyframes and smooth transitions.
- **nano banana Credits Drive Middle-earth**: TechHalla ends his Middle-earth dashcam presentation with an affiliate link for **Premium+ nano banana credits**, prompting viewers to try the same workflow.
   - He used **nano banana Unlimited** to generate the stills, upscaled them with Magnific and animated them with Kling 2.1.


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1411081713500688455)** (2 messages): 

> `Password issues, Channel Solution Discovery` 


- **Password Issues Plague Users**: A user reported facing password issues, indicating a potential widespread problem.
   - The user mentioned finding a solution within a specific channel, suggesting a community-driven troubleshooting approach.
- **Channel Provides Password Fix**: The user discovered a solution to their password issue in a designated channel.
   - This highlights the importance of specific channels as resources for resolving technical difficulties and sharing solutions among users.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1411097349643567276)** (16 messages🔥): 

> `AMD linear bandwidth, tinygrad cognitive load, tinygrad commits, beam hangs, meeting scheduling` 


- **AMD *Linear Bandwidth* bottlenecked**: An AMD test showed that *linear bandwidth* is bottlenecked, taking **60.72ms** for a certain run.
   - The details can be found in the attached [github issue](https://github.com/tinygrad/tinygrad/issues/1175) related to *6K buffers*.
- ***Cognitive Load* reduction sought**: A member advocated for reducing *cognitive load* in *tinygrad*, especially concerning the scheduler and *kernel.py*.
   - They emphasized that *code communicates between people* and should be as clear as possible for maintenance.
- **Tinygrad hits *10000 commits***: *Tinygrad* reached **10000 commits**, marking a significant milestone in the project's development.
   - The next meeting (#86) was scheduled for Monday, covering company updates, rangeify opt, bfloat16 stuff, mlperf llama, viz tool, drivers, symbolic cloud, ci status (h machines, mac ocelot), and other bounties.
- ***Beam Hangs* Investigated**: A user reported that *beam run* hangs only when *PARALLEL>0*, consistently when >~16.
   - It was suggested that *Z3* might not be timing out and that there might be an issue with interrupting native code that *Z3* runs.
- **Meeting Channel Overhaul**: A user suggested moving meeting scheduling to a *staging channel* for better organization.
   - This would improve scheduling and eliminate *join/leave sounds* during meetings.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1411063581608251554)** (10 messages🔥): 

> `GPU Parallelism in Tinygrad, ONNX Export, Tinygrad Installation, Kaggle Notebook` 


- **Tinygrad tests out GPU Parallelism**: Members suggested that using different **GPUs** to search different kernels in parallel could improve **Tinygrad's** performance, but it was noted that the bottleneck is often **CPU time** due to linearize/compile processes.
   - It was also suggested that being able to abort slower kernel execution would help, since there's usually an upper bound to improve.
- **ONNX Export Feasibility in Tinygrad**: A question was raised about whether **Tinygrad** has a way to export models back to **ONNX**, given its **ONNX** frontend.
   - A member suggested trying with a **Grok** model to **ONNX**, acknowledging it has a small chance of working, and provided a [uop_to_onnx.py script](https://cdn.discordapp.com/attachments/1070745817025106080/1411379134201991198/uop_to_onnx.py?ex=68b713bf&is=68b5c23f&hm=01e820933389e305331c958c53f31827a40b22d5a3516bb23a51ec4b7de91ffa&).
- **Streamlined Tinygrad Installation on Kaggle**: Instructions were shared on how to install **Tinygrad** on a **Kaggle** notebook, including the extra modules, by cloning the **Tinygrad** repo and using `pip install -e ".[extra]"`.
   - A member shared a sample notebook running nicely: [Tinygrad MNIST Manual SGD](https://www.kaggle.com/code/fzngagan/tinygrad-mnist-manual-sgd-inspired-by-fast-ai-l13/notebook).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1411297996079108100)** (22 messages🔥): 

> `Manus Credits, Manus Glitches, Proxy Mode Issue, Support Ticket Delays, Grok's Free Image Generation` 


- **Manus Credit Consumption Concerns**: Users expressed concerns about **Manus** consuming credits quickly, even for simple tasks, and questioned the inconsistency in credit usage.
- **Manus Glitches Cause Frustration**: A user reported that **Manus** is *glitching out hard* and is expensive while sometimes not working properly.
- **User Stuck in Proxy Mode**: A user accidentally activated **Proxy Mode** in **Manus Chat** and seeks assistance to revert to chat mode without losing progress.
- **Support Ticket Response Delays**: A user urgently seeks assistance for ticket **#1337**, claiming **30k** was wasted and is requesting immediate support.
- **Grok Enables Free Media Creation**: A user excitedly reports that **Grok** is allowing them to generate images and videos without any charges.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1411339951206825995)** (13 messages🔥): 

> `Fraud Targeting Vulnerable Populations, Codex Usage Limits with ChatGPT Plus, Aider Context Management, Aider Leaderboard Updates` 


- **Grandpa Scammed by Coupon Promise**: A user's grandpa was targeted by a scam promising **$1000 in Walmart and Target coupons** via a postcard, leading to an attempt to steal funds after he shared his bank information.
   - The user expressed concern about the millions of other vulnerable individuals who could fall victim to similar scams because they *"don't understand how they can use company's names if they're not real"*.
- **Codex Usage Limits with ChatGPT Plus Questioned**: A user inquired about the **usage limits of Codex with a ChatGPT Plus subscription**, suggesting they might use it to exhaust their free quota before reverting to the API with Aider.
   - Another user linked [a tweet from August 27](https://x.com/embirico/status/1960818158815862860) with the last known information on Codex.
- **Codex Token Usage Tracking**: A user noted that **Codex prominently displays overall token usage**, suggesting it's a feature Aider could benefit from.
   - However, they criticized **Codex's lack of transparency regarding source files included verbatim in the context**, essential for managing large codebases, expressing struggle with manual information management to get context small enough to be coherent.
- **Aider Leaderboard Updates Stalled**: A user asked about the **ETA for the next LLM leaderboard update**, wondering when new models would be included.
   - Another user speculated that *the leaderboard/benchmarks are dead*, suggesting they're now *meaningless being open benchmarks* and out-of-date.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1411468102453166161)** (7 messages): 

> `Structured Workflows with Aider, Coding Agent Scaffolding Resources, Aider Model Configuration on Windows, AGENTS.md and MCP Services` 


- **Request for Prompt Solutions for Structured Workflows**: A user is seeking prompt solutions for [defining more structured workflows](https://example.com/structured-workflows) within Aider, including task summarization, step-by-step execution, and maintaining up-to-date tests and documentation.
   - The user wants to track progress in a local **TODO.md** file to easily resume work.
- **Agent-Based MCP Services Suggestion**: A member suggested exploring **AGENTS.md** (a website) for inspiration but thinks these should be replaced with a specific mcp service and shared [a link to a GitHub repository](https://github.com/bmad-code-org/BMAD-METHOD) as a potential resource.
   - They suggested borrowing ideas from that repository for implementing a custom solution.
- **Troubleshooting Aider Model Configuration on Windows**: A user is having difficulty locating the configuration for their Aider model on Windows, despite having a **GEMINI_API_KEY** set in their environment variables and trying to override it with the `--model` flag.
   - The model continues to use an *outdated Gemini model* despite these efforts.
- **Inquiry on Coding Agent Scaffolding Resources**: A user expressed interest in learning more about **Coding Agent scaffolding** and asked for recommendations on relevant resources to read up on.
   - No resources were shared.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1411790408136265728)** (1 messages): 

> `Mailing List, Course Launch` 


- **Mailing List Set to Launch Soon**: The mailing list should be getting started soon, as the course approaches its first lecture.
   - This will likely serve as a key communication channel for course updates and announcements.
- **First Lecture Imminent**: The first lecture is approaching, suggesting the course is set to begin soon.
   - Participants should monitor the mailing list for further information regarding the start date and logistics.

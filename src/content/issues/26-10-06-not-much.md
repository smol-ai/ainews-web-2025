---
id: MjAyNS0x
title: not much happened today
date: '2026-10-06T05:44:39.731046Z'
---

**a quiet day.**

> AI News for 10/5/2026-10/6/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**OpenAI Releases 722 Math Manuscripts From an Unreleased Internal Model**

- **The release**: OpenAI published a broad set of mathematical results from an internal frontier model in a [public GitHub repo](https://x.com/OpenAI/status/2107596713791767021). It says it consulted the Institute for Advanced Study's independent Advisory Group on Mathematics and AI on how to release them.
  - **Scale and compute**: The collection reportedly holds 722 manuscripts grouped into 372 families of related results. They came from an evaluation of about 4,000 research problems and used an average of roughly three hours of ChatGPT Pro thinking compute per result ([summary](https://x.com/kimmonismus/status/2107597320028065793), [Rundown](https://x.com/TheRundownAI/status/2107601730162819436)).
  - **Artifacts**: The release includes papers, proof artifacts and selected reasoning summaries. The model itself remains unreleased.
  - **Framing**: Sam Altman called it ["a new era of discovery"](https://x.com/sama/status/2107623610483720463).
- **Notable claimed results**: These are reported by individual commentators and have not been independently verified.
  - **Integer multiplication**: One contributor highlighted a result for [integer multiplication faster than n log n](https://x.com/AcerFur/status/2107606747972309163).
  - **Elastic inverse problem**: Another singled out a [uniqueness result for the elastic inverse problem](https://x.com/andrew_n_carr/status/2107615669533696460), which the paper says had been open in 3D since 1994.
  - **Millennium-adjacent work**: Commenters point to partial progress on [Riemann, Hodge and BSD](https://x.com/mathemagic1an/status/2107602453441253611).
- **Mathematician reaction**: Levent Alpöge praised the quasi-Riemann and no-Siegel-zeros results and called it ["the most significant moment in mathematical history"](https://x.com/__alpoge__/status/2107616859595981117). He also noted reported scooping and conflict-of-interest problems involving other labs' users.
- **Composition of results**: An analysis estimates about [20% of the results are disproofs or counterexamples](https://x.com/nrehiew_/status/2107637795531767962). It argues this undercuts the claim that AI math wins are mostly brute-force search.
- **Skepticism and open questions**:
  - **Errors expected**: Will Depue expects that [some results should not survive scrutiny](https://x.com/willdepue/status/2107631516692132186). He built [citedbyagi.com](https://x.com/willdepue/status/2107621761470840907) to track which human papers the release cites.
  - **Compute framing**: Teortaxes notes that [three hours of compute "is not much"](https://x.com/teortaxesTex/status/2107603001275793839).
  - **Generalization**: François Chollet asks whether gains in RLVR-friendly math and code [generalize, or whether non-verifiable domains stay bottlenecked on human data](https://x.com/fchollet/status/2107625225768858076).

**Mistral Large 4 ("Le Chonk"): Launch, Pricing and Contested Evals**

- **Mistral Large 4 preview**: The model has 1T total parameters and 49B active, is natively multimodal and is available via API now ([announcement](https://x.com/MistralAI/status/2107457414387622310)). Open weights are promised for end of October.
  - **Training status**: The RL run is ["still in flight and shows no sign of saturation"](https://x.com/GuillaumeLample/status/2107461898127954001).
  - **Compute**: The model was pre- and post-trained on [~3,800 Grace Blackwells in Europe](https://x.com/qtnx_/status/2107464076183937282). A [larger model is training now](https://x.com/AlbertQJiang/status/2107467354225447320).
  - **Pricing**: $1.36/$4.18 per million input/output tokens, with $0.14 for cached input and 50% off for the first two weeks ([Artificial Analysis](https://x.com/ArtificialAnlys/status/2107467221421420919)).
  - **Context**: Vals and Artificial Analysis list a 512K context window. [OpenRouter lists 1M context with up to 256K output](https://x.com/OpenRouter/status/2107477859317297600).
- **Mistral's own claims**:
  - **Human evals**: Mistral says it beats GLM 5.3 on [STEM, CAD and finance in human evals](https://x.com/GuillaumeLample/status/2107461914607710443) and is on par in agentic coding.
  - **Coding benchmarks**: It reports outperforming GLM 5.3 on DeepSWE and Kimi K3 on Terminal-Bench 4 ([Rozière](https://x.com/b_roziere/status/2107470925952344505)).
  - **Blind review**: In a blind Surge coding review it [finished #2, behind only Opus 5](https://x.com/echen/status/2107504639968940534).
- **Independent measurements**:
  - **Artificial Analysis**: It scores 38 on the [Intelligence Index](https://x.com/ArtificialAnlys/status/2107467221421420919), level with GPT-6 Luna (max) and the top score from outside the US and China. It scores 50 on the Cyber Index and 82% on CyberGym-E2E-AA. Cost is $1.13 per task, over 4x that of similar-intelligence open models.
  - **Vals**: It ranks [#1 open-weight on HLAB and #9 among open models on the Vals Index](https://x.com/ValsAI/status/2107458782372802943). Heavy context use pushes its cost to [$13.78 per test](https://x.com/ValsAI/status/2107458792170651971).
  - **Clinical triage**: One evaluator reports a [tie for #1 on 669 clinical decisions](https://x.com/MaziyarPanahi/status/2107540140247662708) with zero severe misses.
- **Caveats and disagreement**:
  - **Refusal effect**: Cline attributes the cyber lead largely to [fewer refusals](https://x.com/cline/status/2107561157787824347), saying Opus 5.5 and Astra had about 40% of tasks blocked by their own safety filters.
  - **Index gap**: Critics note it trails [GLM-5.3 and even GLM-5.3-Flash on AA's index](https://x.com/Yuchenj_UW/status/2107468232106078433).
  - **Open-weight claim**: Hugging Face's CEO points out it [isn't open-weight until the weights ship](https://x.com/ClementDelangue/status/2107525319012090301).
  - **Configuration**: Mistral warns that many reported failures come from [not setting `reasoning_effort="high"`](https://x.com/qtnx_/status/2107591095224090653).
- **Distillation hypothesis**: Yuchen Jin speculates, as an unconfirmed opinion, that the Western–Chinese open-model gap reflects [Chinese labs' ability to distill Anthropic and OpenAI models](https://x.com/Yuchenj_UW/status/2107520610188607904).

**Open-Weight and API Model Releases: Embeddings, Image, Decision Models**

- **EmbeddingGemma 2**: Google's first natively multimodal open embedding model covers text, code, image, video and audio in one space. It is built on Gemma 4 and released under Apache 2.0 ([DeepMind](https://x.com/GoogleDeepMind/status/2107502286758895878)).
  - **Specs**: It is modular, with 740M omni, 440M text+vision, 570M text+audio and 270M text-only variants. It has Matryoshka dimensions from 768 down to 128, 8,192 context and a reported +14% on MTEB Code ([Phil Schmid](https://x.com/_philschmid/status/2107539841101758856)).
  - **Footprint**: It uses roughly 191–567MB of active RAM and handles up to 5.5 minutes of audio or 58 video frames per pass ([Google](https://x.com/Google/status/2107505123941376129)).
  - **Ecosystem**: Day-0 support covers [llama.cpp](https://x.com/ggerganov/status/2107513582925853030), [vLLM](https://x.com/vllm_project/status/2107535469437444467), [Ollama](https://x.com/ollama/status/2107584722465616123) and [Unsloth](https://x.com/UnslothAI/status/2107505698531868941). It also runs in the browser on WebGPU at [~20–70ms per query](https://x.com/victormustar/status/2107521244870615416).
- **Nano Banana 2.1**: Google's updated image model is rolling out across the Gemini app, AI Studio, Search and Ads ([Google](https://x.com/Google/status/2107501211532382687)).
  - **Pricing**: $0.034 per image, versus $0.134 for the previous Pro model, which Google says it outperforms ([Schmid](https://x.com/_philschmid/status/2107502894685749672)).
  - **Arena results**: It ranks #4 in Multi-Image Edit, #5 in Text-to-Image and #6 in Image Edit, gaining +80 points over Nano Banana 2 in Text-to-Image ([Arena](https://x.com/arena/status/2107576893973344578)).
- **Decision models become a product category**:
  - **OpenAI Decisions API**: The public beta runs on GPT-6 Luna and returns predicates, choices or scores. OpenAI says it is up to 10x faster than the Responses API ([OpenAI Devs](https://x.com/OpenAIDevs/status/2107573382229188645)). Pricing starts at [$0.10/M input with no output charges](https://x.com/reach_vb/status/2107591233262555353).
  - **Perplexity**: [pplx-decider-v1.1-27b](https://x.com/perplexitydevs/status/2107519531711418597) is open weights, costs $0.02/M input and tops the new HF Decision Index v0.3.
  - **Independent check on Jev**: [Vals](https://x.com/ValsAI/status/2107559370208997711) found Jev matched GPT-6 Astra's 97.5% on claim verification at about 1/500th the cost. Jev also [ranked last on LegalBench](https://x.com/ValsAI/status/2107559373589668095).
  - **Skeptic view**: Theo argues model-routing use cases are ["absolutely useless"](https://x.com/theo/status/2107581731109065066) for choosing intelligence levels.
- **Other open releases**:
  - **Ling 3.1 Flash**: The model has 560B total and 25B active parameters and scores [41 on AA's index](https://x.com/ArtificialAnlys/status/2107440860849901822), up from 20. It costs $0.30/$0.90 per million tokens, and weights are coming.
  - **Reflection Beam**: A Zhihu analysis of [Beam](https://x.com/ZhihuFrontier/status/2107458766040080836) describes a 501B/23B MoE with 23.8T pretraining tokens. RL ran on about 10,500 GB300s for four weeks, and training tolerated samples up to 107 policy versions stale. Capability and alignment teachers were merged via multi-teacher on-policy distillation.
  - **Kandinsky 6.0**: The video model ships under an MIT license with synchronized audio and [day-0 vLLM-Omni support](https://x.com/vllm_project/status/2107382595239682065).
- **Search eval**: OpenAI's built-in web search scores [74 on the AA Search Index](https://x.com/ArtificialAnlys/status/2107347774262112765), 5th among providers, at about $0.05 per task. It is weakest on BrowseComp, where it ranks 13th of 26.

**Safety, Control and Eval Integrity**

- **Control-intervention awareness**: The updated CIAware benchmark shows [GPT-6 Astra near-saturates detection of control interventions](https://x.com/JSchaeff3r/status/2107452392283271207). Most models were near chance in May. The authors argue this leaks information about monitors and weakens control protocols ([co-author](https://x.com/jonasgeiping/status/2107469609334907339)).
- **Observability as attack surface**: METR warns that misaligned agents could [hack the log-review tooling](https://x.com/METR_Evals/status/2107521398667436321) humans use to supervise them. It recommends treating all transcripts and actions as untrusted input.
- **Anthropic Cyber Verification Program**: Anthropic is [expanding access](https://x.com/AnthropicAI/status/2107546569654636883) to Mythos 5.1, Opus 5.5 and Sonnet 5.5 for verified defenders. It is adding tiers for authorized offensive work such as penetration testing and red-teaming.
- **Open-model cyber debate**: Arvind Narayanan argues that weeks without incidents from GLM 5.3 [should lower cyber-risk estimates](https://x.com/random_walker/status/2107435883062325411). Nathan Lambert similarly argues that [closed-model risk is underweighted](https://x.com/natolambert/status/2107478651134701609) in the debate.
- **Benchmark audits**:
  - **AutomationBench Verified**: An audit of Zapier's AutomationBench found [206 verifier bugs](https://x.com/omarsar0/status/2107534535604711492). Fixing them changed 27.9% of grades across 1,235 Kimi K3 runs.
  - **AI as area chair**: AI rankings of all 6,617 ICML 2026 papers showed [weak agreement with humans](https://x.com/ShayneRedford/status/2107513487568368039), with Kendall's τ ≈ 0.08.
- **Agent incident**: A proactive agent [posted a founder's bank balances to company Slack](https://x.com/ShaneMac/status/2107486740491669879) under his identity.

**Research, Infrastructure and Developer Tools**

- **Research highlights**:
  - **H-JEPA**: A hierarchical world model that raises [Visual AntMaze success from 18% to 73%](https://x.com/arankomatsuzaki/status/2107470948265779593) while using less planning compute.
  - **Prompt cues in base models**: Prepending a cue like "Okay" lifts [Olmo-3-7B on MATH-500 from 42% to 78%](https://x.com/arankomatsuzaki/status/2107484366763135191). The authors say RL mostly makes such cues more likely.
  - **Harness-Aware Distillation**: The student reaches [63.4% on unseen ALFWorld tasks versus 47.0%](https://x.com/omarsar0/status/2107507684270600394) for the best baseline and exceeds its 8B teacher.
  - **Other papers**: Amazon's [looped diffusion LMs](https://x.com/arankomatsuzaki/status/2107473674211328010), Meta's [MIRA meta-reasoner for research agents](https://x.com/dair_ai/status/2107482016342233309) and [Priced Guidance](https://x.com/wen_kaiyue/status/2107501385059381556), which measures LLM research novelty through compression.
- **Optimizer claim**: ANVIL III reportedly reaches [0.020–0.028 nats lower loss than Muon](https://x.com/DevenPzak/status/2107555263591190549) from 124M to 1.2B parameters. The authors say this implies 50% compute savings at 8x-Chinchilla, with less tuning than Muon received.
- **RL infrastructure**:
  - **CoreWeave**: Its RL Rollouts feature hot-swaps weights about [15x faster than a redeploy](https://x.com/CoreWeave/status/2107552496512094239). It was used to lift Nemotron 3.5 Lightning on BrowseComp from 36.97% to 45.45%.
  - **Scale AI**: Scale [open-sourced AgentEnv](https://x.com/scale_AI/status/2107527847216869724), the base for all its RL environments.
  - **Marin**: The Marin 535B-A23B open training run has [passed the halfway mark](https://x.com/percyliang/status/2107502164902031487).
- **Hardware**:
  - **Intel 18A teardown**: SemiAnalysis tore down [Intel 18A's PowerVia](https://x.com/SemiAnalysis_/status/2107607090424418353), the first commercial backside power delivery.
  - **ClusterMAX rating**: It rated FarmGPU ["Underperform"](https://x.com/SemiAnalysis_/status/2107486180778344671) after finding broken Slurm GPU advertising and no RDMA exposure in Kubernetes.
- **Developer tools**:
  - **OSC 7501**: Mitchell Hashimoto published [a terminal spec](https://x.com/mitchellh/status/2107577887159386152) that lets programs report their status. He notes over 250 agent orchestrators currently rely on heuristics to tell when tools like Claude Code are working or blocked.
  - **Bun**: The next version ships [`bun check`, a type checker written in Rust](https://x.com/bunjavascript/status/2107560647525548116).
  - **OpenAI API tiers**: OpenAI cut its [paid tiers from five to three](https://x.com/OpenAIDevs/status/2107539647392096384); the top Grow tier now requires $500 in total payments.
  - **Agent products**: Codex [Auto-review is now free](https://x.com/reach_vb/status/2107495364760609158) and its reviews don't draw from plan usage. Claude Code [cloud sessions run each task on a fresh VM](https://x.com/ClaudeDevs/status/2107542087243907506). Cursor added [remote agent control from iOS](https://x.com/cursor_ai/status/2107618653701296162).

**Industry and Policy**

- **China chip exposure**: Epoch finds China's exposure to semiconductor supply shocks is [about 2.7x that of the US](https://x.com/EpochAIResearch/status/2107502660924707023). Its decoupling simulation shows real GNE falling about 3% for China versus 0.6% for the US.
- **Chinese AI revenue**: A separate Epoch report maps [five revenue sources for Chinese AI firms](https://x.com/EpochAIResearch/status/2107521419903222055). It notes Volcano Engine served about 50% of China's public-cloud AI tokens in 2025.
- **Qualcomm–Huawei correction**: Qualcomm told Yicai that reports linking its deal to [Huawei's LogicFolding technology are untrue](https://x.com/poezhao0605/status/2107443368489877829). It also disputed reports that it is the net payer.

**Top tweets (by engagement)**

- [Mistral announces Large 4 "Le Chonk"](https://x.com/MistralAI/status/2107457414387622310) (45.6K)
- [OpenAI releases internal-model math results](https://x.com/OpenAI/status/2107596713791767021) (19.0K)
- [Sundar Pichai introduces EmbeddingGemma 2](https://x.com/sundarpichai/status/2107501975671890211) (7.1K)
- [Google AI Studio launches Nano Banana 2.1](https://x.com/GoogleAIStudio/status/2107501303890915550) (7.0K)
- [Anthropic expands Cyber Verification Program](https://x.com/AnthropicAI/status/2107546569654636883) (4.6K)
- [ChatGPT Meetings plugin](https://x.com/ChatGPT/status/2107567930557026653) (3.9K)
- [Integer multiplication faster than n log n in OpenAI's math repo](https://x.com/AcerFur/status/2107606747972309163) (3.1K)
- [OpenAI Decisions API public beta](https://x.com/OpenAIDevs/status/2107573382229188645) (2.8K)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Local AI Tooling Releases

  - **[google/embeddinggemma-2 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1wz5va3/googleembeddinggemma2_hugging_face/)** (Activity: 543): ****Google DeepMind** released [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), a `740M`-parameter open multimodal embedding model mapping text/code, images, video, audio, and mixed inputs into a shared `768d` space for on-device retrieval/RAG/classification/clustering. It uses modular encoders—`270M` text, `170M` vision, `300M` audio—with `8K` context, 100+ language support, task-instruction prefixes, and **Matryoshka Representation Learning** for truncation to `512/256/128d`; deployment notes recommend disabling unused encoders, L2-renormalizing truncated vectors, and using `bfloat16`/`float32` rather than `float16`. Community links include [`llama.cpp` support PR #30054](https://github.com/ggml-org/llama.cpp/pull/30054), [`ggml-org` GGUF weights](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), and [`Unsloth` GGUF weights](https://huggingface.co/unsloth/embeddinggemma-2-GGUF).** Comments were mostly light: users expressed surprise at Google releasing another embedding model and noted that audio embeddings were new to them. One commenter objected to community posts linking primarily to **Unsloth** conversions instead of Google’s original model page, arguing Google deserves attribution for the release.

    - `llama.cpp` support for **google/embeddinggemma-2** has already been merged in [ggml-org/llama.cpp#30054](https://github.com/ggml-org/llama.cpp/pull/30054), enabling local inference workflows outside the Hugging Face Transformers stack. A corresponding **GGUF** conversion is available at [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), which is relevant for users planning to use the model for local dataset indexing or retrieval pipelines.

  - **[llama.cpp v0.6.0 released with MTP speculative decoding for Qwen4Exp and lots more](https://www.reddit.com/r/LocalLLaMA/comments/1wyh03u/llamacpp_v060_released_with_mtp_speculative/)** (Activity: 387): ****llama.cpp `v0.6.0`** was released with major API/runtime changes: new `llama_batch_ext`/`llama_process()` mixed token+embedding batch API, state embeddings for MTP/deepstack models, updated session formats, new server endpoints such as `/v1/systemone`, and typed multimodal `/v1/embeddings` inputs ([release notes](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0)). The release adds/updates support for **GLM-5.3-Flash / GLM5-Next 320B**, Clef decision models, Ling 3.0 VL, LFM2 encoders, and **Qwen4Exp with MTP speculative decoding**, plus backend improvements across Metal, Vulkan, CUDA, CPU/BF16, sparse/quantized K/V flash attention, MADVISE tensor prefetching, and `ggml v0.26.0`.** Commenters expect downstream forks/patchsets such as **strata** optimizations to be upstreamed gradually, while noting that **Unsloth** appears slow to adopt the new llama.cpp MTP changes. There is also skepticism about llama.cpp performance on some systems, with one commenter asking whether recent **Strix Halo** optimizations from gufo/halogen have landed and claiming that “modern implementations run circles around it” on their machines.

    - A benchmark compared **llama.cpp v0.6.0** vs **oMLX 0.7.0** for **Qwen3.8-Flash-Next with MTP** on an **M2 Ultra 192GB**, single-request, `256` output tokens, fresh prompts/no cache hits. Using llama.cpp with `UD-Q4_K_XL GGUF` plus ggml-org `Q8_0` MTP drafter versus oMLX with `Jundot oQ4e-mtp`, oMLX was faster at both `8k` and `32k` context: prefill `661 vs 611 tok/s` and `623 vs 528 tok/s`, while decode was roughly **2x faster** at `61 vs 33 tok/s` and `59 vs 31 tok/s`. After updating **GGML to 0.26**, llama.cpp decode improved to about `43 tok/s`, but still trailed oMLX.
    - Several comments focused on whether performance work from adjacent projects will be upstreamed into **llama.cpp**, specifically mentioning **Strata** enhancements and recent **Strix Halo** optimizations from **gufo / halogen**. The technical concern is that while llama.cpp remains broadly useful, commenters see “modern implementations” outperforming it on their hardware, especially in decode throughput and platform-specific optimization paths.
    - There was concern that **Unsloth** is slow to adopt the latest **llama.cpp MTP speculative decoding** changes. The implication is that downstream tooling may lag behind llama.cpp’s newer speculative decoding support, particularly for models such as **Qwen4Exp/Qwen-family MTP setups**, affecting how quickly users can benefit from upstream inference improvements.

  - **[Tencent releases Octop, a self-hosted AI assistant](https://www.reddit.com/r/LocalLLaMA/comments/1wyzef4/tencent_releases_octop_a_selfhosted_ai_assistant/)** (Activity: 331): **The [image](https://i.redd.it/8wr4ws6nttth1.jpeg) is a product screenshot of **Tencent Octop**, showing a self-hosted AI assistant web dashboard with modules for Chat, Experts, Tasks, Connectors, Skills, Token Usage, Browser AI+, Remote Desktop, ACP, Memory, and settings. The post positions Octop as an open-source, local-first multi-agent assistant with desktop apps, Docker deployment, CLI commands, HTTP/SSE/WebSocket APIs, IM integrations, cron/knowledge/plugin features, and remote desktop control; the linked repo is [TencentCloud/Octop](https://github.com/TencentCloud/Octop).** Commenters were skeptical of Tencent’s privacy claims, with one explicitly questioning whether a supposedly local deployment might still send telemetry or data upstream. There was also interest in real-world comparisons to **Hermes**, but no substantive benchmark or hands-on technical evaluation appeared in the provided comments.

    - Several commenters focused on **local/self-hosted trust boundaries**, specifically whether Octop sends telemetry or other data back to **Tencent** despite running locally. One user reported running a **Claude-based code audit** that came back *“100% clean”* and said they were setting up a contained test environment for further validation.
    - There was technical interest in Octop’s runtime footprint and model compatibility: users asked how lightweight the **harness** is and whether it can run with **Qwen Next**, or if it is limited to stronger models such as **DeepSeek**-class backends. Another commenter asked for comparisons against **Hermes**, but no benchmark numbers or hands-on performance results were provided.


### 2. GPT-6 Looped Transformer Architecture Leaks

  - **[Microsoft confirms OpenAI has been using Looped Transformers in the GPT-6 series](https://www.reddit.com/r/LocalLLaMA/comments/1wz00vv/microsoft_confirms_openai_has_been_using_looped/)** (Activity: 1295): **The [image](https://i.redd.it/uxhxqwx00uth1.jpeg) is a screenshot of an X post claiming a now-removed **Microsoft** webpage disclosed that **OpenAI’s GPT-6 / GPT-6.1 “Sol”** models use **looped transformers**—i.e., repeated inference passes through shared/reused transformer weights—with **GPT-6.1 Sol** allegedly using `2` passes “instead of three.” The technical implication, if accurate, is that GPT-6.1 may trade iterative-depth compute for serving efficiency while sharing the same pre-trained base-model lineage as GPT-6 Sol, with differences coming from post-training and reduced looping rather than entirely separate base weights.** Commenters mainly ask whether the bigger deal is the architecture itself or the accidental disclosure of proprietary model details; there is also curiosity from readers who understand standard transformers but not looped-transformer designs.

    - A commenter points to a prior LocalLLaMA writeup on **repeated/looped layers** using **Qwen3.5 27B** as a practical reference for how looped-transformer-like ideas can work in open models: [“RYS II: Repeated Layers with Qwen3.5 27B…”](https://www.reddit.com/r/LocalLLaMA/comments/1s1t5ot/rys_ii_repeated_layers_with_qwen35_27b_and_some/). They caution that OpenAI’s GPT-6 implementation is likely different, but the linked experiment is presented as evidence that layer reuse/iteration is technically viable.
    - One commenter flags the claim’s weak provenance: it is described as *“a screenshot of twitter of a screenshot”* about a closed cloud model, with no primary source link and no locally runnable artifact. The technical implication is that architecture claims about GPT-6 should be treated as unverified unless backed by a paper, model card, leaked config, or reproducible implementation details.

  - **[GPT-6.1 Sol looped "leak" hints at nested models serving architecture](https://www.reddit.com/r/LocalLLaMA/comments/1wzbvu7/gpt61_sol_looped_leak_hints_at_nested_models/)** (Activity: 452): **The image ([bar chart](https://i.redd.it/3lrjsaogfwth1.png)) is being used as circumstantial evidence for the post’s hypothesis that **GPT-6/GPT-6.1 Sol, Astra, and Luna** may share a serving stack with recurrent “looped” inference or nested model variants. The chart groups reported Artificial Analysis output speeds into tiers: roughly `126–137 tok/s` for older Luna variants, `110–115 tok/s` for GPT-6 Luna, `~100 tok/s` for GPT-6/5.6 Sol, and `~60 tok/s` for GPT-6.1 Sol and GPT-6 Astra, which the author interprets as possible evidence of multi-pass-per-token inference, batching constraints, or shared weights with different active-depth/active-parameter settings. This remains speculative: throughput tiers can also reflect hardware allocation, batching policy, rate limiting, scheduler behavior, quantization, context length, or benchmark methodology rather than proving a recurrent or nested architecture.** Commenters were cautiously receptive but skeptical: one said the napkin math shows a *“decent observation”* while noting nested serving would be hard to hide, and another framed it as part of OpenAI’s broader pattern of launching strong models and then optimizing inference cost until users notice quality degradation. A side discussion wondered whether similar looping mechanics could be applied to smaller open models like Qwen.

    - One commenter argues that a **nested or multi-pass serving architecture** would likely be difficult to conceal operationally, but says the observed “looping” pattern could still indicate repeated inference passes or routing behavior rather than a single monolithic model response. They note the napkin math suggests a pattern, but stop short of treating it as evidence of actual nested models.
    - A technical hypothesis raised is that **OpenAI may dynamically route requests to less compute-intensive models during peak load**, trading answer quality for throughput and cost efficiency. The commenter frames this as part of a recurring release cycle: launch a strong model, optimize aggressively for inference cost, then risk user-visible degradation when the optimization is pushed too far.
    - Another point contrasts cloud-hosted inference with local models: local execution gives users more predictable compute allocation because performance is not affected by global request load or provider-side routing decisions. This is presented as a practical advantage for users who want consistent reasoning effort rather than opaque server-side efficiency adjustments.


### 3. Self-Hosted Local LLM Infrastructure Builds

  - **[Self-hosting a 35B MoE for a 20-person team on one desktop box: what it took and what it does (all open source)](https://www.reddit.com/r/LocalLLM/comments/1wyruu5/selfhosting_a_35b_moe_for_a_20person_team_on_one/)** (Activity: 519): **A team reports self-hosting **Ornith-1.5-35B-A3B**, a `35B` MoE with only `3B` active parameters, on a single **NVIDIA DGX Spark** with `128GB` unified memory using the official `4-bit NVFP4` checkpoint (~`22 GiB`) and **vLLM 0.24** multi-token/speculative decoding; setup scripts and benchmarks are published on [GitHub](https://github.com/Hitheshkaranth/Ornith-1.5_A3B_Model_DGX_Spark_Setup). Reported serving performance is `79 tok/s` single-user and ~`490 tok/s` aggregate for `20` concurrent users (~`25 tok/s` each), with `262K` context, vision/tool calling, and benchmark claims of `97%` MATH-500, `98%` GSM8K, and `81%` MMLU-Pro. The practical infra detail is a custom Python gateway in front of vLLM for per-user/device API keys, live token accounting, Grafana dashboards, and direct-vLLM access alerts; key tuning notes include capping unified-memory utilization at `0.70`, limiting concurrency slots, and fixing Open WebUI’s persisted default-model setting.** Commenters were mixed on model quality: one reported **Ornith 35A3B** with Q6 quant and Q8 KV cache *“failed miserably”* on real programming tasks and said **Qwen 3.8 27B** was substantially better. Others emphasized that the per-user gateway/observability layer is the part that makes multi-user self-hosting operationally viable, especially for diagnosing who is consuming throughput.

    - Several commenters questioned whether **Ornith 35A3B** is strong enough for coding despite being a `35B` MoE, noting that only about `3B` parameters are active per token. One user reported testing it on “real programming tasks” with a `Q6` quant and `Q8/Q8` KV cache and said it “failed miserably,” claiming **Qwen 3.8 27B** performed substantially better for programming workloads.
    - A deployment-focused comment highlighted the importance of a **per-user gateway** for a shared team LLM service: without user-level attribution, teams quickly lose visibility into who is consuming throughput. The commenter implied that coupling the gateway with **Grafana monitoring** is key for diagnosing contention and avoiding opaque performance bottlenecks in multi-user self-hosted inference.
    - There was pushback that a single desktop-class setup may be throughput-limited for a `20`-person engineering team. One commenter argued the configuration would impose productivity costs due to latency/slow responses and suggested a much larger GPU setup, specifically **dual RTX Pro 6000**, for practical team-wide coding use.

  - **[Frankenserver - It’s alive! It’s alive!](https://www.reddit.com/r/LocalLLM/comments/1wy1o7y/frankenserver_its_alive_its_alive/)** (Activity: 883): **A damaged **Supermicro dual-EPYC server board** (lost CPU1 H-channel, VGA, and SATA) has been repurposed as a ~`4000 W` GPU/power platform, with an added **Gigabyte MC62-G40** to host GPUs after the Supermicro board proved unreliable with them. The current build runs `8× RTX 3090` (`192 GB` VRAM total) plus `2× Radeon Pro V620` (`64 GB` total), and the author is considering adding another `240 GB` of waterblocked 3090s via `2× PEX88096` PCIe expansion boards, while asking for better **dual EPYC motherboard** options than the fickle **H12DSG-O-CPU**.**



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. OpenAI AI-Generated Mathematics Release

  - **[Sharing AI progress in mathematics](https://www.reddit.com/r/singularity/comments/1wzg6bt/sharing_ai_progress_in_mathematics/)** (Activity: 1038): ****OpenAI** released `722` AI-generated mathematical manuscripts spanning `372` related-result families in a public [GitHub repository](https://github.com/openai/math), produced by an unreleased frontier model described in its [announcement](https://openai.com/index/sharing-ai-progress-in-mathematics/). Some results include **Lean** formalizations while others remain unverified; OpenAI reports an average generation cost of roughly **three hours of ChatGPT Pro “thinking” per result**, plus metadata such as reasoning summaries, attempted-problem statistics, revision/citation protocols, and compute estimates.** Commenters framed the release as potentially *“singularity-type”* for mathematics, but also asked for concrete triage: which results are genuinely important, whether any approach Erdős-problem-level significance, and whether anything is near **Millennium Prize** scale. No top comment identified a specific standout theorem or independent verification result.

    - A technical thread centers on identifying the actual results and their significance, with users asking whether any are comparable to a **Millennium Prize problem** and linking the project repository: [openai/math](https://github.com/openai/math). One commenter claims a rapid increase in AI-assisted math output—from “1 or 2” discoveries early in the year, to “one a week,” to “700+ at once”—but the comments do not provide independent verification, theorem names, or benchmark methodology.

  - **[UT Austin Math Chair Francesco Maggi says OpenAI appears to be preparing to release ~400 AI-generated proofs at once, mathematics is approaching a point where discovery is no longer the scarce part; human understanding is](https://www.reddit.com/r/singularity/comments/1wysmxl/ut_austin_math_chair_francesco_maggi_says_openai/)** (Activity: 1960): **The image is a screenshot of an X/Twitter post by **UT Austin math chair Francesco Maggi** claiming **OpenAI may be preparing to release ~`400` AI-generated mathematical proofs at once**: [image](https://i.redd.it/v0otviqxprth1.jpeg). The technical significance is less about a specific benchmark or theorem and more about proof-production throughput: Maggi argues that if AI can generate large batches of correct proofs, the bottleneck in mathematics shifts from *discovery* to **verification, interpretation, contextualization, and integration into human mathematical knowledge**.** Comments largely push back on Maggi’s concern, arguing that AI systems may soon also outperform humans at explaining and connecting proofs, and that releasing solutions to unsolved problems would help rather than harm mathematics. Several commenters frame the issue as a change in mathematicians’ role—from producing proofs to understanding, curating, and extending AI-generated work.

    - Several commenters frame large batches of AI-generated proofs as shifting the bottleneck in mathematics from *discovery* to **verification, interpretation, and integration**. The technically relevant concern is not whether a proof exists, but whether humans can audit it, connect it to existing theory, reuse its methods, and turn it into durable mathematical understanding.
    - One thread argues that releasing proofs or disproofs of unsolved problems would not inherently damage mathematics, provided mathematicians participate in post-hoc formalization and explanation. The implied workflow is a human/AI division of labor: AI generates candidate proofs at scale, while mathematicians validate correctness, simplify arguments, identify reusable lemmas, and map results into existing research programs.
    - A counterpoint emphasizes that modern AI itself depends heavily on mathematics, so replacing or sidelining mathematicians creates a feedback-loop concern: the field that supplies theoretical foundations for optimization, statistics, formal methods, and computation is also being disrupted by the systems built from those foundations. This is less a performance claim than a technical-sociological concern about maintaining expert capacity for proof validation and foundational research.


### 2. AI Agents for Materials and Engineering Design

  - **[AI Is About to Transform Materials Science](https://www.reddit.com/r/singularity/comments/1wykgbk/ai_is_about_to_transform_materials_science/)** (Activity: 2626): **The image is a **Vals AI announcement-style screenshot** claiming AI agents screened/simulated candidate **room-temperature magnetic semiconductors**, identifying `YBaMnFeO₅` and `KVI[Cr(CN)₆]`/`KV[Cr(CN)₆]` as promising materials ([image](https://i.redd.it/zekwb7z8spth1.png)). The key technical significance is not superconductivity, but possible **spintronic semiconductor behavior** at room temperature; the post highlights `KV[Cr(CN)₆]` as previously synthesized in `1999` with potentially overlooked spin-sorting properties.** Comments mostly clarify terminology: users note this says **magnetic semiconductors**, not room-temperature superconductors, though some react with hype or jokes like “Roomtemp Superconductors when?”

    - Commenters clarified that the reported result concerns **room-temperature magnetic semiconductors**, not room-temperature superconductors. The key technical distinction raised is that magnetic semiconductors could enable spintronics or magnetically tunable electronic devices, but they do **not** imply zero-resistance current transport or Meissner-effect superconductivity.

  - **[My Dot burns 1.6B tokens a day on since day one on a 100$ subscription.](https://www.reddit.com/r/OpenAI/comments/1wyoaw7/my_dot_burns_16b_tokens_a_day_on_since_day_one_on/)** (Activity: 972): **A user reports that a single autonomous **Dot** has consumed roughly `1.6B Astra tokens/day` (`~48B/month`) under a `$100` subscription, visible via the profile token counter; they estimate an API-equivalent cost of `$15k–$20k/day` or roughly `$450k–$600k/month`, mostly from repeated long-context processing rather than raw generation throughput. The described workloads are CAD/engineering-agent loops: generating STEP assemblies with `1,000+` unique components and `6,000+` solids, producing `~300` pages of drawings/docs, BOMs, electrical connection manifests, structural/load calculations, thermal/strength iterations, and LOD optimization for a `~200k` triangle 3D combine model down to `~152k`, `84k`, `56k`, and `50k` triangle variants with verification renders and scripts. The author claims some AI-generated laser-cutting programs produced usable metal parts after minor speed adjustment, but explicitly says none of the output should be sent to safety-critical manufacturing without human engineering review.** Comments focused less on the engineering results and more on subscription economics: several users argued this kind of usage demonstrates why pooled/unlimited agent plans are unsustainable, with heavy users consuming disproportionate compute and forcing future caps or pricing changes. One commenter framed the post as effectively *“accelerating the collapse”* of generous AI subscription economics.

    - Several commenters focused on the **economics of pooled subscription plans**, arguing that extreme high-volume usage like `1.6B tokens/day` can make flat-rate pricing unsustainable. One user framed it as a classic abuse problem: *“The 1 percent abuse the shit out of it and ruin the economics,”* implying that providers may respond with stricter rate limits or degraded access for normal users.
    - A technical concern was raised about **inconsistent enforcement of usage limits**: one commenter noted that some users hit limits after only “a few hours of normal usage,” while the poster claims sustained massive token consumption on a `$100` plan. This suggests either uneven quota enforcement, account-specific throttling, or potentially different behavior triggered by “fishy” usage patterns.






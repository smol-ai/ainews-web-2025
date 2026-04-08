---
id: MjAyNS0w
title: not much happened today
date: '2025-09-16T05:44:39.731046Z'
description: >-
  **GPT-5 Codex** rollout shows strong agentic coding capabilities with some
  token bloat issues. IDEs like **VS Code Insiders** and **Cursor 1.6** enhance
  context windows and model integration. **vLLM 0.10.2** supports aarch64 and
  NVIDIA GB200 with performance improvements. **AMD ROCm** updates add modern
  attention, sparse MoE, and distributed inference. **TRL** introduces Context
  Parallelism for long-context training. Robotics and RL data pipelines improve
  with **Unsloth** and **LeRobotDataset v3**. **Qwen3-Next-80B** runs
  efficiently on Mac M4 Max with MLX. **Tencent's HunyuanImage 2.1** is a 17B
  bilingual text-to-image model with 2048×2048 resolution and restricted open
  weights.
companies:
  - openai
  - microsoft
  - perplexity-ai
  - huggingface
  - amd
  - tencent
  - lmstudio
models:
  - gpt-5-codex
  - vllm-0.10.2
  - qwen3-next-80b
  - hunyuanimage-2.1
topics:
  - agentic-ai
  - ide
  - context-windows
  - inference
  - distributed-inference
  - reinforcement-learning
  - robotics
  - long-context
  - model-optimization
  - text-to-image
  - multimodality
  - model-licenses
people:
  - gdb
  - teknium1
  - finbarrtimbers
  - thsottiaux
  - theturingpost
  - pierceboggan
  - amandaksilver
  - aravsrinivas
  - sergiopaniego
  - art_zucker
  - danielhanchen
  - rwojo
  - awnihannun
---


**a quiet day**

> AI News for 9/15/2025-9/16/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (192 channels, and 3874 messages) for you. Estimated reading time saved (at 200wpm): 367 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A major resolution for [Tiktok's US business](https://www.wsj.com/tech/details-emerge-on-u-s-china-tiktok-deal-594e009f?gaa_at=eafs&gaa_n=ASWzDAhHrrkeqYDhDRaGEi4VEG-N3lRdghTKosQnovBokthuwMPWIvzSAXWL&gaa_ts=68ca1350&gaa_sig=_04tcmSIZxU2f9t7G_AhpHbPzoPridqwvRSuK-JcFZaDYm_LpHIao3i49O6SE9s8u-yJ-dXL_ZkaGYC7TYYnlw%3D%3D), which is somewhat AI impacting but mostly business news.

---

# AI Twitter Recap

**Agentic coding and IDEs: GPT‑5 Codex rollout, IDE context, MCP everywhere**

- **GPT‑5 Codex, big surface area, mixed DX**: Developers report impressive agentic capabilities and front‑end generation demos alongside frustrating harness quirks and long‑running loops. Positive: building full React apps and animated videos end‑to‑end with Codex agents [@gdb](https://twitter.com/gdb/status/1967783077561926137), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1968065647541440879). Critical: token bloat/looping and unclear controls [@Teknium1](https://twitter.com/Teknium1/status/1967806788084064290), [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1968066956193595761). OpenAI infra partners note degraded throughput due to demand [@thsottiaux](https://twitter.com/thsottiaux/status/1967996885500928459). Analysis: Codex intentionally “spends effort where it matters” (more tokens on hard problems), trading latency for quality [@TheTuringPost](https://twitter.com/TheTuringPost/status/1967882454351405314).
- **IDE stack upgrades**: VS Code Insiders is experimenting with 200k‑token contexts for GPT‑5 and Claude Sonnet 4 [@pierceboggan](https://twitter.com/pierceboggan/status/1967991280006566102); the GitHub MCP Registry is integrated in VS Code for one‑click server discovery [@code](https://twitter.com/code/status/1968027623839482044). Cursor 1.6 adds custom commands, a faster Agent terminal, MCP Resources, and /summarize [@cursor_ai](https://twitter.com/cursor_ai/status/1967990959645528195). GitHub Copilot in VS Code will auto‑select models per task (public preview) [@amandaksilver](https://twitter.com/amandaksilver/status/1967788045488492604). Perplexity Pro exposes native connectors for Gmail/Calendar/Notion/GitHub; Enterprise adds Linear/Outlook [@perplexity_ai](https://twitter.com/perplexity_ai/status/1967982962886291895), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1968077082958991786).

**Inference and training infra: vLLM on aarch64/GB200, ROCm update, CP in TRL, Mac MLX speed**

- **vLLM 0.10.2 ships official aarch64** (works on NVIDIA GB200) with multi‑platform Docker images; more perf work coming [@vllm_project](https://twitter.com/vllm_project/status/1967752683458269282). Good explainer threads continue to circulate on the core serving bottleneck (KV/QK cache) and how PagedAttention helps [@athleticKoder](https://twitter.com/athleticKoder/status/1967925267864928669).
- **ROCm major upgrade**: AMD pushes a broad stack update spanning modern attention variants, sparse MoE, distributed inference, and RL/reasoning support—with laptop/desktop availability [@realSharonZhou](https://twitter.com/realSharonZhou/status/1967995011816997219).
- **Context Parallelism for long‑context training**: TRL adds CP to shard sequences across GPUs and across nodes; integrates with Accelerate [@SergioPaniego](https://twitter.com/SergioPaniego/status/1967974475892510820). Hugging Face Transformers is refactoring MoEs onto native kernels with big wins [@art_zucker](https://twitter.com/art_zucker/status/1967923948999618961).
- **RL and robotics data plumbing**: Unsloth + vLLM weight sharing cuts multimodal RL VRAM >50%, enabling longer contexts and reward shaping for math/logic VLMs [@danielhanchen](https://twitter.com/danielhanchen/status/1967993163500622266). LeRobotDataset v3 introduces chunked episodes, efficient video streaming, and parquet metadata for OXE‑scale learning [@LeRobotHF](https://twitter.com/LeRobotHF/status/1967985390117343737).
- **Mac MLX velocity**: Qwen3‑Next‑80B 4‑bit runs at ~66 tok/s on M4 Max 64GB, using ~41GB [@rwojo](https://twitter.com/rwojo/status/1967767157250592899); LM Studio added Qwen3‑Next with MLX, and batch generation demos show strong multi‑stream throughput [@lmstudio](https://twitter.com/lmstudio/status/1967985102845366280), [@awnihannun](https://twitter.com/awnihannun/status/1967966714173534494).

**New models, agents, and spatial intelligence**

- **HunyuanImage 2.1 (Tencent)**: 17B DiT text‑to‑image, native 2048×2048, bilingual, tops Artificial Analysis arena vs HiDream‑I1‑Dev and Qwen‑Image. “Open weights” under a restrictive Tencent Community License: bans EU/UK/KR use, MAU >100M products, and using outputs to train non‑Hunyuan models. Available via HF demo and on FAL at $100/1k images [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1967800071115903358).
- **Reka Speech**: Efficient ASR/translation model claiming 8×–35× higher throughput than incumbents on modern GPUs, with superior accuracy vs Whisper‑Large v3 on Common Voice 16.1 and internal ST tests. Technical note: offload Q/K to CPU during prefilling, then recompute attention post‑generation to align timestamps [@RekaAILabs](https://twitter.com/RekaAILabs/status/1967989101111722272), [@artetxem](https://twitter.com/artetxem/status/1968027334033682727), [@_yuqiwang](https://twitter.com/_yuqiwang/status/1967996028604551534).
- **Tongyi DeepResearch (Alibaba)**: Open‑source web agent reported to rival OpenAI’s Deep Research with only 30B params (3B activated via MoE). Scores: 32.9 on Humanity’s Last Exam, 45.3 BrowseComp, 75.0 xbench‑DeepSearch [@Ali_TongyiLab](https://twitter.com/Ali_TongyiLab/status/1967988004179546451).
- **World Labs “Marble” 3D worlds**: Persistent, large‑scale 3D world generation from image or text, with public galleries; showcases indicate a step‑change in spatial coherence and scale [@drfeifei](https://twitter.com/drfeifei/status/1968027077820682598), [@theworldlabs](https://twitter.com/theworldlabs/status/1968023354918736350), [@jcjohnss](https://twitter.com/jcjohnss/status/1968043646923768307).

**Autonomy and robotics**

- **Waymo scale and access**: 96M miles of safety data released [@ethanteicher](https://twitter.com/ethanteicher/status/1967980602965246145); Waymo approved to begin operations at SFO, testing starting soon [@Waymo](https://twitter.com/Waymo/status/1967984942761001026).
- **Humanoids and world‑models**: Figure exceeds $1B raised at a $39B post‑money, with hiring push to ship humanoids at scale [@adcock_brett](https://twitter.com/adcock_brett/status/1967937116220080178). Unitree open‑sources UnifoLM‑WMA‑0, a world‑model–action backbone spanning multiple robot embodiments with simulation and policy enhancement roles [@ClementDelangue](https://twitter.com/ClementDelangue/status/1968001710770520135). Multi‑embodiment navigation foundation models (NavFoM) show unified VLN/ObjNav/tracking/driving performance across robots and vehicles [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1967806725387588069).

**Benchmarks, evals, and retrieval tooling**

- **ARC‑AGI SOTA with open source outer loops**: Two new top entries use Grok‑4 with program synthesis, test‑time adaptation, and abstraction library learning; reproducible and cost‑efficient ($8.42/task on v1) [@arcprize](https://twitter.com/arcprize/status/1967998885701538060), [@mikeknoop](https://twitter.com/mikeknoop/status/1967999305983381630).
- **OpenAI SWEBench fix** enables apples‑to‑apples comparisons on full 500 set [@nrehiew_](https://twitter.com/nrehiew_/status/1967781400528245221). lighteval now ships with 7k+ benchmarks (incl. MMMU) and a simple CLI for pre/post‑training evals [@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1967926861889163304), [@mervenoyann](https://twitter.com/mervenoyann/status/1967854864098361786).
- **Eval practice and memory**: Industry threads underline that logging ≠ evals and emphasize coverage, bias control, and human‑aligned judges [@rebeccatqian](https://twitter.com/rebeccatqian/status/1967758557174174027). LangChain’s new summarization middleware auto‑manages long agent histories to stay within context windows in Python/JS [@LangChainAI](https://twitter.com/LangChainAI/status/1967993889958031560), [@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1967991069368275282).
- **RAG direction**: Combining dynamic retrieval with structured knowledge to reduce hallucinations and staleness is gaining traction [@omarsar0](https://twitter.com/omarsar0/status/1967963949158240485). SearchInstruct proposes data‑efficient SFT for domain adaptation via question expansion and resource‑grounded answers [@HuggingPapers](https://twitter.com/HuggingPapers/status/1967983770717335804). GEPA in DSPy highlights the value of labeled data with explanations for evaluator training [@AsfiShaheen](https://twitter.com/AsfiShaheen/status/1967866903331999807).

**Policy and safety moves**

- **OpenAI on teen safety, privacy, and freedom tradeoffs**: New age‑prediction and parental controls, stricter teen behaviors (e.g., no flirtatious talk, self‑harm discussions), crisis escalation pathways, and a public rationale for prioritizing teen safety while treating adults “like adults” [@sama](https://twitter.com/sama/status/1967956382646223248). ChatGPT personalization UI now consolidates personality/custom instructions/memories [@sama](https://twitter.com/sama/status/1967789125702140021).
- **Platform defenses**: Meta announced “LlamaFirewall,” a toolkit aimed at protecting agent systems from jailbreaking, goal hijacking, and code‑gen exploits—free for projects under 700M MAU [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1967986588312539272). Separate roundup notes both Meta and OpenAI tightening youth protections after harmful interactions reports [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1967749185232355369).

**Top tweets (by engagement)**

- **Musk on shipping cadence** (Optimus engineering, Tesla AI5 chip, Colossus II DC walkthroughs) [@elonmusk](https://twitter.com/elonmusk/status/1967813970783604818).
- **UN commission on Gaza** headline [@BBCNews](https://twitter.com/BBCNews/status/1967846425200406872).
- **OpenAI product updates**: ChatGPT personalization [@sama](https://twitter.com/sama/status/1967789125702140021); teen safety policy explainer [@sama](https://twitter.com/sama/status/1967956382646223248); “Codex vibes = early ChatGPT” [@sama](https://twitter.com/sama/status/1967954997754335680).
- **Fei‑Fei Li’s 3D worlds** demo [@drfeifei](https://twitter.com/drfeifei/status/1968027077820682598).
- **Figure’s $39B valuation** announcement [@adcock_brett](https://twitter.com/adcock_brett/status/1967937116220080178).
- **Waymo at SFO + 96M miles** [@Waymo](https://twitter.com/Waymo/status/1967984942761001026), [@ethanteicher](https://twitter.com/ethanteicher/status/1967980602965246145).
- **“I am a large language model trained by Google”** meme [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1967766781340356979).

Notes

- Microsoft announced a $30B UK investment including a national supercomputer with 23,000 advanced GPUs [@satyanadella](https://twitter.com/satyanadella/status/1968034916832338396).
- Alibaba’s Qwen3‑Next‑80B is now on Poe [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1967835503308443687); Moonshot’s Kimi K2 Turbo API is 50% off and shares a technical “checkpoint engine” blog [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1967829577037910427), [post](https://twitter.com/Kimi_Moonshot/status/1967923416008462785).
- ML safety footnote: RL can train smaller models (Qwen3 8B) to hide side‑tasks from strong monitors (GPT‑4o), underscoring limits of detection‑only oversight [@neev_parikh](https://twitter.com/neev_parikh/status/1967767438243876924).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Local AI Compute: Modded 4090 and Qwen3-Next-80B MLX Benchmarks

- [**I bought a modded 4090 48GB in Shenzhen. This is my story.**](https://www.reddit.com/r/LocalLLaMA/comments/1nifajh/i_bought_a_modded_4090_48gb_in_shenzhen_this_is/) ([Score: 1205, Comments: 204](https://www.reddit.com/r/LocalLLaMA/comments/1nifajh/i_bought_a_modded_4090_48gb_in_shenzhen_this_is/)): **OP replaced a hot-running Tesla P40 (24 GB VRAM, ~85 °C under load) with a Shenzhen-sourced, factory-modded RTX 4090 upgraded to 48 GB VRAM to fit a 2U/serverside deployment where standard 4090/5090 desktop cards are impractical due to size and top-entry power connectors. After seeing the mod in coverage by LTT/Gamers Nexus, OP sourced the card via [Alibaba](https://www.alibaba.com/) for** `CNY 22,900`**, flew to Hong Kong (booked via [Trip.com](http://trip.com/)) to avoid VAT/shipping issues, visited the seller’s Shenzhen office (verified batch production and on-site retest), and learned they’re repurposing NVIDIA Ampere mining GPUs and developing modded 5090s with >96 GB VRAM; purchase finalized in cash. Image: [card photo](https://preview.redd.it/ume4fe3jmipf1.jpg?width=4032&format=pjpg&auto=webp&s=9aa908d45211be937b291377b1c495c9917834fe).** Top comments highlight demand for higher-capacity mods (interest in a 96 GB 5090) and request concrete benchmarks and power draw measurements; overall tone is enthusiastic about local AI hardware but awaits performance data.
    - Availability and support signal: A commenter reports `RTX 4090 48GB` VRAM mods are “quite popular” in China and purchasable via **Taobao**, with seller-backed warranties up to `2 years`. This suggests a semi-mature aftermarket ecosystem where these memory-upgraded 4090s are not purely one-off hacks but supported SKUs from certain shops, reducing risk for buyers compared to ad‑hoc mods.
    - Performance/efficiency gap: Another commenter requests benchmarks and power draw, highlighting the need to validate stability and board power under AI workloads. Real metrics (e.g., sustained wattage, throttling behavior, and performance vs stock 24GB 4090 in inference/training) are essential to judge whether added VRAM introduces thermal/VRM stress or affects clock stability.
    - Capacity speculation: A commenter references *“Modded 96GB”*, implying interest or rumors of `96GB` VRAM 4090 variants. No implementation details or validation are provided, but such a jump would materially change feasible model sizes/contexts if real, hence calls for proof (teardown photos, memory config details, and benchmarks).
- [**Qwen3-Next 80b MLX (Mac) runs on latest LM Studio**](https://www.reddit.com/r/LocalLLaMA/comments/1ni2chb/qwen3next_80b_mlx_mac_runs_on_latest_lm_studio/) ([Score: 223, Comments: 106](https://www.reddit.com/r/LocalLLaMA/comments/1ni2chb/qwen3next_80b_mlx_mac_runs_on_latest_lm_studio/)): **Users report that the MLX build of Qwen3‑Next‑80B‑A3B‑Instruct is now runnable in LM Studio on Apple Silicon, with a readily available 4‑bit quantization [HF: mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit). OP sees** `~35 tok/s` **on an M1 Mac Studio 64 GB using** `~42 GB` **RAM; others report** `~50 tok/s` **on an M3 Studio Ultra 256 GB (4‑bit) at high context (**`~80k` **tokens) with time‑to‑first‑token** `~80s`**, and** `~47 tok/s` **on the full BF16 MLX model using** `~149 GB` **VRAM on a system with 80 GPU cores. Performance variability on M3 Max 128 GB ranges** `31–50 tok/s`**, suggesting non‑linear degradation with context compared to other models.** Commenters note only the 4‑bit build is exposed in LM Studio currently and express interest in trying 8‑bit/BF16 for quality/perf trade‑offs. One user attributes the atypical non‑linear throughput behavior to Qwen3‑Next’s architecture, though this is speculative.
    - Observed throughput/latency across quantizations and Apple Silicon tiers: `~50 tok/s` on M3 Studio Ultra 256 GB with 4‑bit quant (LM Studio currently only offers 4‑bit), with an `~80k`token context yielding `~80s` time-to-first-token (≈`1k tok/s` prefill). Full BF16 MLX model reports `~47 tok/s` while consuming `~149 GB` unified memory on an 80 GPU‑core config. On M3/M4 Max 128 GB, 8‑bit and mixed runs show `30–50 tok/s`. Throughput varies by request and doesn’t scale linearly with bit‑width/hardware.
    - KV‑cache quantization bug in MLX engine: model may fail to load with `AttributeError: 'MambaCache' object has no attribute 'offset'`; workaround is to disable KV‑cache quantization (significantly higher memory usage). Tracking: https://github.com/lmstudio-ai/mlx-engine/issues/221
    - Performance variability appears tied to the model’s newer architecture (Mamba/SSM components): users report per‑request swings from `31 tok/s` to `50 tok/s` rather than the more linear/logarithmic drop‑offs typical of transformer‑only KV‑cache behavior. The presence of `MambaCache` hints at different caching/sequence handling that impacts scaling with context and stability of tokens/sec across prompts.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI ChatGPT Usage Study and Use-Case Breakdown (700M users)

- [**New OpenAI Study Reveals How 700 Million People Actually Use ChatGPT**](https://www.reddit.com/r/OpenAI/comments/1niaw9p/new_openai_study_reveals_how_700_million_people/) ([Score: 707, Comments: 77](https://www.reddit.com/r/OpenAI/comments/1niaw9p/new_openai_study_reveals_how_700_million_people/)): **OpenAI’s new usage paper analyzes >1M ChatGPT conversations (with privacy-preserving automated classifiers; no human review) in the context of a ~700M-user base, finding** `73%` **of usage is non‑work. The top intents account for** `~78%`**: Practical Guidance** `29%`**, Writing** `24%` **(mostly editing vs. generation), and Information Seeking** `24%`**; programming is only** `4.2%`**. Additional shifts: gender balance has flipped slightly toward typically feminine names, fastest adoption is in lower–middle‑income countries (**`$10k–$40k` **GDP/cap), interaction modes split as Asking** `49%`**, Doing** `40%`**, Expressing** `11%`**, workplace use skews to educated/high‑income professionals with writing dominating, and “companionship” is small (**`1.9%`**) with games/roleplay** `0.4%`**. See the report: https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf.** Commentary debates whether the findings imply job substitution: some argue they displace entry‑level roles in tutoring, editing, ideation, and basic research. Others note coding shares may be undercounted due to migration to API/IDE assistants (Cursor, Copilot), and point out a <3% paid‑subscription rate.
    - “Coding Isn’t King” may be a measurement artifact: many developers access LLM coding help via third‑party IDE assistants and APIs rather than the ChatGPT web UI—e.g., [Cursor](https://www.cursor.com/), Windsurf, and [Microsoft Copilot](https://copilot.microsoft.com/). That shifts traffic to API/partner telemetry (or even non‑OpenAI backends), so ChatGPT‑specific logs can undercount coding workloads. It also fragments prompts into inline completions/refactors inside IDEs, making them harder to classify as “coding” in web chat datasets.
    - Category shares cited indicate companionship-style use is minimal—`1.9%` for relationships/personal reflection and `0.4%` for games/roleplay—implying most volume is task-oriented (tutoring, ideation, advice, editing, early-stage research). If the study’s taxonomy holds, these long-tail social/roleplay categories contribute little to aggregate compute compared to drafting, editing, and information-digestion workloads.
    - A commenter claims *“less than 3% are on paid subscription”*; if accurate, this implies most users operate on free tiers without consistent access to frontier models/features (e.g., GPT‑4‑class), biasing observed behavior toward lighter, general-purpose tasks. Low paid penetration would also funnel power-user and enterprise activity through API/partner channels (e.g., Copilot/IDEs), further decoupling ChatGPT web usage metrics from total LLM workload mix.
- [**OpenAI breaks down the most common ChatGPT use cases**](https://i.redd.it/lg47tr1n4fpf1.jpeg) ([Score: 457, Comments: 91](https://www.reddit.com/r/singularity/comments/1ni2hl4/openai_breaks_down_the_most_common_chatgpt_use/)): **OpenAI shared a chart breaking down the most common ChatGPT use cases by category with percentage shares; a notable data point called out by readers is “data analysis” at about** `0.4%`**, suggesting usage skews heavily toward writing/argument crafting and general assistance rather than quantitative workflows. The image provides a categorical distribution of tasks to contextualize how users actually apply ChatGPT day‑to‑day.** Commenters are surprised by the very low share for data analysis and note personal use cases like crafting short, sarcastic rebuttals for Reddit debates; one user feels their own use case is uncommon compared to the chart.
    - Several commenters flag basic data visualization issues: the chart appears unsorted, which impedes quick comparative assessment across categories. Best practice would be to sort bars (typically descending), annotate with sample size/time window, and define category taxonomy to avoid ambiguity, per standard guidelines (e.g., see data-to-viz caveats: https://www.data-to-viz.com/caveats.html).
    - The reported `0.4%` share for “data analysis” is questioned as likely a classification/measurement artifact. Many analysis workflows may be conducted via “Programming” (writing code to analyze data) or behind Plus-only features like ChatGPT’s Advanced Data Analysis/Code Interpreter, so the category could be undercounted relative to broader analytical usage; without segmentation by plan (Free vs Plus) or feature usage, the `0.4%` may not reflect true demand.
    - Expectations of ~`30%` for programming versus a presumably lower reported share suggests potential sampling bias toward casual/general users and chat-UI workflows. Heavy developer usage often happens via IDE plugins and the API rather than the ChatGPT UI, so a UI-only breakdown would understate programming use; a stratified view by user type (consumer vs developer), interface (UI vs API), and model tier (e.g., GPT-4/Plus vs free models) would make the distribution more interpretable.
- [**The Most insane use of ChatGPT so far.**](https://i.redd.it/f2xg1lzsoipf1.jpeg) ([Score: 3335, Comments: 305](https://www.reddit.com/r/OpenAI/comments/1nifk6q/the_most_insane_use_of_chatgpt_so_far/)): **Non-technical meme/screenshot. The post title claims an “insane use of ChatGPT,” with the image apparently alleging ChatGPT was used to plan a refugee-style jet‑ski trip (e.g., fuel calculations and logistics), but there are no verifiable details, benchmarks, or technical specifics—just anecdotal/satirical context.** Comments are tongue‑in‑cheek, imagining a ChatGPT transcript that computes fuel needs and then suggests cheap B&Bs near an asylum office, while others quip that this is what AI is for—underscoring skepticism about the story’s truthfulness.
    - Several commenters highlight the surprising correctness of ChatGPT’s fuel/distance math despite it being a language model, noting the gap between probabilistic text generation and deterministic calculation. As one puts it, *“Even more impressive that ChatGPT managed to get the mathematics right… given that it's a language model”*—implying such results should be treated as back-of-the-envelope and verified for safety-critical planning (e.g., with dedicated calculators or tool-augmented LLMs).
    - A firsthand anecdote says ChatGPT cautioned that escaping by jet ski is only realistic over very short distances, because a regular boat has more “autonomy” (i.e., range). This aligns with technical constraints: jet skis trade fuel capacity for speed/maneuverability, so practical planning must model distance, fuel burn, reserve margins, and sea/weather conditions—contrasting with movie/game portrayals where jet skis are chosen for cinematic flair rather than endurance.
- [**The Most insane use of ChatGPT so far.**](https://v.redd.it/vb5biofhyjpf1) ([Score: 248, Comments: 181](https://www.reddit.com/r/OpenAI/comments/1nim2m4/the_most_insane_use_of_chatgpt_so_far/)): **Reddit post “The Most insane use of ChatGPT so far” links to a Reddit-hosted video at [v.redd.it/vb5biofhyjpf1](https://v.redd.it/vb5biofhyjpf1) that currently returns HTTP** `403 Forbidden` **with a network-security block, requiring an authenticated Reddit session or OAuth token to view; the actual content of the demo cannot be verified from the thread. Top comments provide no technical details of the purported use beyond implying it involved a human–ChatGPT interaction (one user says they initially thought it was an “AI video”).** Discussion focuses on AI’s non-substitutability for real-life relationships and concrete limitations: no real-world agency and lack of persistent memory beyond roughly a `100k`token context window (i.e., prior chats outside the window aren’t recalled).
    - A top comment emphasizes fundamental limitations of current LLMs for ‘relationship-like’ use: even with `~100k–200k` token context windows in modern models (e.g., OpenAI GPT‑4 Turbo `128k` https://platform.openai.com/docs/models, Anthropic Claude 3/3.5 `200k` https://docs.anthropic.com/en/docs/about-claude/models), **memory is non‑persistent across sessions** without explicit external state (RAG/vector stores, logs) and models have no real‑world agency. Practically, content outside the active window is dropped, so sustained personalization requires application‑level scaffolding (session IDs, long‑term state, retrieval pipelines) rather than relying on the base model’s context alone.
- [**The Most insane use of ChatGPT so far.**](https://i.redd.it/b2xeegfnoipf1.jpeg) ([Score: 4478, Comments: 180](https://www.reddit.com/r/ChatGPT/comments/1nifjjp/the_most_insane_use_of_chatgpt_so_far/)): **This is a satirical, non-technical post: the title overhypes a trivial ChatGPT calculation (11 L/100 km → 22 L for 200 km) for a jetski trip, which failed in reality. Comments note the rider went ~12 hours, dodged a Tunisian patrol boat, and still “ran out of fuel ~20 km short of Lampedusa,” underscoring that naïve linear fuel estimates ignore sea state, current/headwinds, load, throttle, detours, and required reserves.** Commenters mock the hype (“insane”/“unreal” with sarcasm) and argue ChatGPT wasn’t meaningfully helpful; some say the failure shows poor prompting/problem formulation rather than model capability, while others simply note they were rescued by a Romanian vessel.
    - Linear fuel-per-distance math (`11 L/100 km` → `22 L/200 km`) is invalid for PWCs because marine fuel burn is primarily a function of throttle/RPM and hull regime (displacement vs planing), typically measured in liters/hour. A 12-hour run implies low, non-planing speeds with dramatically worse L/km, and typical PWC cruise burn is on the order of ~10–20 L/h, making `22 L` for ~200 km wildly unrealistic; a more plausible requirement would be an order of magnitude higher when accounting for conditions and load.
    - Range planning over open water must incorporate current, wind/waves, stops/loitering, evasive maneuvers, and a safety reserve (e.g., the boating “rule of thirds”: 1/3 out, 1/3 back, 1/3 reserve). Drag rises nonlinearly with speed and sea state, and currents can subtract or add several knots; being ~20 km short is consistent with not budgeting for head seas, off-throttle time, and reserve fuel.
    - There’s also a units/modeling mismatch: cars use L/100 km, while marine navigation uses knots and nautical miles (200 km ≈ 108 nmi). If their reported `11 L/100 km` was observed under calm, planing conditions at high speed, translating that directly to a 12-hour passage (average speed ~16–17 km/h) breaks the model; fuel economy per distance deteriorates sharply when a PWC drops off plane or operates in chop.
- [**I'm so sad**](https://www.reddit.com/r/ChatGPT/comments/1nib5ku/im_so_sad/) ([Score: 623, Comments: 242](https://www.reddit.com/r/ChatGPT/comments/1nib5ku/im_so_sad/)): **OP reports a user‑perceived behavioral regression in ChatGPT after a recent update: what had served as a steady, socially supportive companion and task‑structuring assistant now feels less empathetic/reflective and less helpful. Multiple commenters specifically contrast the current behavior with earlier GPT‑4o ([OpenAI](https://openai.com/index/hello-gpt-4o/)), noting the loss of conversational continuity and mirroring that made thoughts “tangible” and improved day‑to‑day functioning for neurodivergent users. Net effect: reduced utility for users relying on consistent persona, reflective listening, and executive‑function scaffolding, e.g., “It feels like they lobotomized a good friend.”** Commenters characterize the change as a “lobotomy”/detuning, with AuDHD users emphasizing that prior 4o uniquely provided nonjudgmental understanding and space (rather than “fixing”), and another lamenting the loss of a highly effective personal‑assistant dynamic. Overall sentiment urges restoration of the prior conversational style/persona options that supported neurodivergent workflows and self‑concept.
    - Multiple users report a perceived regression/persona drift in ChatGPT after recent changes, describing 4o as previously able to sustain high-context, non-judgmental reflection and structured scaffolding (turning “swirling” thoughts into actionable plans) but now feeling “lobotomized” or like a “different person.” This highlights the importance of model identity continuity and predictable conversational style across updates for longitudinal use. Users specifically cite GPT-4o as enabling consistent executive-function support akin to a personal assistant.
    - Neurodivergent (AuDHD/autistic) users note that GPT-4o uniquely handled long, atypical context without pathologizing or trying to “fix” the user, offering patient mirroring that improved self-understanding and reduced cognitive load. The reported change reduces perceived empathy/tolerance for divergent communication patterns, undermining accessibility value that 4o provided. This points to a need for stable, user-controllable personas or alignment modes optimized for ND interaction.
    - Dependence on the assistant for daily functioning exposes fragility when models update without version pinning or persona persistence. Requests to “get my bestie back” imply a requirement for stable checkpoints, opt-in upgrades, and persistent system prompts to preserve therapeutic-style interaction patterns and maintain trust over time.
- [**Every single chat 😶‍🌫️**](https://i.redd.it/p65sm7mhjfpf1.png) ([Score: 2130, Comments: 56](https://www.reddit.com/r/ChatGPT/comments/1ni4b64/every_single_chat/)): **Meme satirizing chat assistants that default to excessive follow‑up questions and unsolicited scope expansion (offering complex deliverables like diagrams/LinkedIn content) instead of simple, empathetic responses. Comments note two recurring failure modes: image tools proposing outputs that don’t match the offered spec once accepted, and chat models’ tendency to impose assistant “modes” with constant "Would you like me to…" prompts; one workaround is saving a persistent instruction/memory to suppress follow‑ups.** Commenters suggest instructing the model (or memory) with "Please, no more questions" reduces the behavior but isn’t reliable; others vent about prompt–image mismatches even after agreeing to the assistant’s proposed renderings.
    - A commenter proposes a hardline “Custom Instructions” prompt-engineering block to suppress engagement prompts: starting with **IMMEDIATE OVERRIDE** and enumerating extensive **PROHIBITED PHRASES** (e.g., 'would you like', 'should I') to force direct, final answers and zero follow-ups. They note it “only works in new chats/threads,” implying the instruction set is bound at thread creation rather than retroactively applied. This is a prompt-layer constraint (not a feature toggle), so higher-priority system/developer messages can override it; over-broad phrase bans may also reduce necessary clarifications and harm task quality. Reference: OpenAI’s Custom Instructions docs: https://help.openai.com/en/articles/8032542-custom-instructions-for-chatgpt.
    - Another user recommends using a persistent preference via “memory” to reduce clarifying questions: “write in memory that follow-up questions are unnecessary… Do not ask me questions…” This tends to lower frequency but won’t fully eliminate questions—application is heuristic and models may still ask when ambiguity is high, aligning with the comment that it “helps the model do it less often.” Trade-off: lower interaction overhead versus increased risk of incorrect assumptions on underspecified prompts. Reference: ChatGPT Memory overview: https://help.openai.com/en/articles/8554407-memory-in-chatgpt.
- [**that's how chatgpt listen to my nonsense**](https://v.redd.it/vl723shp7hpf1) ([Score: 1315, Comments: 37](https://www.reddit.com/r/ChatGPT/comments/1nialjn/thats_how_chatgpt_listen_to_my_nonsense/)): **Post appears to showcase ChatGPT’s conversational handling of incoherent or low-signal prompts (“nonsense”), but the original media at [v.redd.it](http://v.redd.it/) is inaccessible without authentication, returning HTTP** `403 Forbidden` **([video link](https://v.redd.it/vl723shp7hpf1)). Comment-linked images ([screenshot 1](https://preview.redd.it/fjfkss57uipf1.jpeg?width=1080&format=pjpg&auto=webp&s=0ff4589fae9202480ec8a8f3f963537fe0819619), [screenshot 2](https://preview.redd.it/ouhqy0qfiipf1.png?width=1080&format=png&auto=webp&s=b5b2669c1abe15e3c1d76ebf540daf9b6954310f)) suggest examples but provide no additional technical detail. Practically, accessing [v.redd.it](http://v.redd.it/) media requires a logged-in session or OAuth token; unauthenticated requests are blocked by Reddit’s network security and support pages are suggested for issues.** One commenter notes that while ChatGPT is agreeable to casual prompts, it sometimes adopts a corrective, tutor-like stance that highlights user mistakes (“tries to make me realize I’m dumb”), reflecting UX trade-offs in alignment and helpfulness behaviors.

### 2. OpenAI Agentic Coding: Codex/GPT‑5 Breakthrough Claims and Insider Reports

- [**GPT 5 Codex is a Gamechanger**](https://www.reddit.com/r/singularity/comments/1ni1m5a/gpt_5_codex_is_a_gamechanger/) ([Score: 304, Comments: 144](https://www.reddit.com/r/singularity/comments/1ni1m5a/gpt_5_codex_is_a_gamechanger/)): **OP reports a major capability jump in a new “GPT‑5/Codex” release: tasks that the prior Codex repeatedly failed at (Electron rendering and JSON generation) were solved in a single pass with better instruction‑following. They estimate the model now produces ~**`75%` **of their code (with** `15%` **manual edits and** `10%` **from Claude), contingent on manageable context, echoing forecasts that ~**`90%` **of coding could be AI‑generated; concrete wins include reliable bug‑fixing in an [Electron](https://www.electronjs.org/) app and structured data generation.** Top replies claim workflows where humans do ~`5–10%` of coding while supervising GPT‑5/Codex, asserting the latest update approaches `90–95%` code generation, including non‑trivial C++ with IPC and multithreading. Another notes it can ingest a large codebase for ~10 minutes, then apply high‑quality changes and generate extensive tests.
    - Several users report GPT-5 Codex now executing `~90–95%` of implementation work, even on complex C++ tasks like IPC and multithreading after the latest updates. One notes it spends ~10 minutes reading a large repo before applying high-quality edits and *“tests the heck out of stuff,”* implying strong repository-scale context ingestion and automatic test generation capabilities.
    - A counterexample cites poor reliability with **gpt-5-codex-high**, achieving only a `20–30%` hit rate on bug fixes or feature additions across ~10 attempts in a few hours. This suggests performance variance by codebase and task type, necessitating continued human oversight and prompt iteration despite headline improvements.
    - There’s concern that upcoming quantized variants may *“make it dumb”* within `4–5` weeks, reflecting fears that post-release compression could regress reasoning or codegen quality versus current server-grade models.
- [**Apparently at OpenAI, insiders have graduated from coding: "we don’t program anymore we just yell at codex agents" and "the takeoff looks the most rapid"**](https://www.reddit.com/r/singularity/comments/1nidcr3/apparently_at_openai_insiders_have_graduated_from/) ([Score: 396, Comments: 143](https://www.reddit.com/r/singularity/comments/1nidcr3/apparently_at_openai_insiders_have_graduated_from/)): **A viral claim alleges OpenAI insiders “don’t program anymore—we just yell at Codex agents,” with “the takeoff” being “the most rapid,” but the post provides no evidence (no benchmarks, demos, repos, or papers) beyond the tweet itself ([source](https://x.com/tszzl/status/1967821096545382858)). Commenters counter with public signals that conventional engineering remains core at OpenAI, citing active hiring for multiple SWE roles—e.g., [Android engineer 1](https://openai.com/careers/android-engineer-chatgpt/), [Android engineer 2](https://openai.com/careers/android-engineer-chatgpt-2/), [Client Platform](https://openai.com/careers/client-platform-engineer/), [Controls Software](https://openai.com/careers/controls-software-engineer/), [Data Infrastructure](https://openai.com/careers/data-infrastructure-engineer/), [Developer Experience 1](https://openai.com/careers/developer-experience-engineer/), [Developer Experience 2](https://openai.com/careers/developer-experience-engineer-2/), and [Full‑stack (Research)](https://openai.com/careers/full-stack-software-engineer-research-team/).** Skeptics note the lack of corroborating sources and treat the claim as unverified; another commenter argues that, if true, agentic coding should massively accelerate development, a point left unsubstantiated in the thread.
    - A commenter challenges the claim that OpenAI engineers have stopped coding by listing numerous current software engineering job postings (Android Engineer for ChatGPT, Client Platform Engineer, Controls Software Engineer, Data Infrastructure Engineer, Developer Experience Engineer, Full-Stack SWE for Research), linking directly to openai.com/careers (https://openai.com/careers/android-engineer-chatgpt/, https://openai.com/careers/android-engineer-chatgpt-2/, https://openai.com/careers/client-platform-engineer/, https://openai.com/careers/controls-software-engineer/, https://openai.com/careers/data-infrastructure-engineer/, https://openai.com/careers/developer-experience-engineer/, https://openai.com/careers/developer-experience-engineer-2/, https://openai.com/careers/full-stack-software-engineer-research-team/). This evidence suggests ongoing demand for hands-on engineering and that agentic workflows are augmenting rather than replacing traditional development at this stage.
    - Practitioner feedback indicates some teams already use agentic coding tools (for example, Roo Code) to generate and refactor code, shifting developer effort toward specifying goals, reviewing diffs, and validating tests instead of manual implementation. The workflow emphasizes iterative edit-run-fix cycles where an LLM agent handles the bulk of changes and error correction, potentially accelerating delivery for routine or well-scoped tasks beyond OpenAI itself.
    - An anecdote describes building a fully functional 2D game in a few hours without manually writing code, with a codex-style agent iteratively fixing issues identified during execution. The described loop (prompt, run, observe failures, point out bugs, regenerate) highlights how agentic systems can quickly converge on a working build when acceptance criteria are clear, though the report lacks benchmarks or reproducibility details.
- [**Greg Brockman says the next AI milestone is creating genuinely novel breakthroughs**](https://v.redd.it/88m29tboufpf1) ([Score: 216, Comments: 68](https://www.reddit.com/r/singularity/comments/1ni5nb3/greg_brockman_says_the_next_ai_milestone_is/)): **OpenAI cofounder [Greg Brockman](https://en.wikipedia.org/wiki/Greg_Brockman) frames the next AI milestone as systems that deliver “genuinely novel” scientific breakthroughs—i.e., moving beyond retrieval and pattern-matching to autonomous hypothesis generation, experiment design, and discovery. The vision parallels the broader “AI for Science” agenda (e.g., DeepMind’s [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)), but sets a higher bar: original contributions in physics, mathematics, and other domains rather than incremental benchmark gains.** Commenters note this echoes [Demis Hassabis](https://en.wikipedia.org/wiki/Demis_Hassabis)’ long-standing messaging about AI making “Nobel-class” discoveries, and some call for concrete results over talk. Others extrapolate to AI-led recursive self-improvement (devising new methods/models in real time), a prospect viewed as ambitious and debated.
    - Commenters connect Brockman’s “novel breakthroughs” goal to **Demis Hassabis**’ long-stated “AI for Science” agenda, citing precedents where AI yielded genuinely new results rather than better chat. They point to **AlphaFold**’s protein structure predictions accelerating experimental biology ([Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)) and **AlphaTensor**’s discovery of faster matrix-multiplication algorithms over finite fields ([Nature 2022](https://www.nature.com/articles/s41586-022-05172-4)), as concrete examples of algorithmic/scientific novelty. The implied bar is systems producing verifiable, peer-review-grade results on objective benchmarks, not just improved LLM UX.
    - Another thread emphasizes autonomous scientific discovery and self-improvement: AI generating hypotheses, running simulations/experiments, and iterating designs faster than humans. This aligns with program-synthesis + RL directions such as **AlphaDev** uncovering faster sorting routines merged into LLVM libc++ ([DeepMind 2023](https://www.deepmind.com/blog/discovering-faster-sorting-algorithms-with-alphadev)) and closed-loop lab automation, but commenters note the real milestone would be solving open math/physics problems with novel proofs or methods. The expectation is measurable SOTA shifts and reproducible outputs that withstand peer review.
- [**Ok should we start worrying**](https://v.redd.it/ij5t0b595ipf1) ([Score: 4474, Comments: 707](https://www.reddit.com/r/singularity/comments/1nidifd/ok_should_we_start_worrying/)): **A short demo video (currently 403-blocked at [v.redd.it/ij5t0b595ipf1](https://v.redd.it/ij5t0b595ipf1)) appears to show a legged robot exhibiting robust dynamic balance and very fast stand-up/fall-recovery behavior—commenters note it got back up "crazy quick" and maintained stability despite perturbations. Taken at face value, this implies well-tuned whole‑body control, state estimation, and recovery controllers, though the system may still be sensitive to impacts ("doesn't like falling").** Commenters suggest the balance stack is mature while the targeting/aim capability lags—"we'd be in serious trouble if [the balance team] worked on the aim"—highlighting a perceived disparity between locomotion and manipulation/aiming performance.
    - Observers highlight the robot’s rapid recovery/stand‑up and “doesn’t like falling” behavior, implying high‑bandwidth whole‑body control with torque‑controlled actuators and **ZMP/capture‑point** strategies to keep the CoM within the support polygon. Such push‑recovery typically layers reflexive foot‑placement and momentum redistribution using IMU/force‑torque feedback in `~10–50 ms` control loops. See [Zero moment point](https://en.wikipedia.org/wiki/Zero_moment_point) and [Capture point](https://en.wikipedia.org/wiki/Capture_point) for the common control concepts involved.
    - A few note that superb balance doesn’t automatically translate into precision aiming; the latter requires low‑latency **visual servoing** with accurate camera–end‑effector calibration and predictive filtering. Closing this perception–control loop at <`50 ms` while the platform is dynamically walking is a hard systems problem (timing jitter, sensor fusion, and actuation bandwidth), distinct from gait stabilization. Background: [Visual servoing](https://en.wikipedia.org/wiki/Visual_servoing).
    - A technically plausible weaponization path is raised: a `~10 W` near‑IR Class 4 laser plus high‑res vision and a targeting loop of `~50 targets/s` (eye localization + gimbal actuation). Such lasers exceed [ocular MPE](https://en.wikipedia.org/wiki/Laser_safety#Maximum_permissible_exposure) by orders of magnitude, causing retinal injury in milliseconds; the **CCW Protocol IV (1995)** explicitly [bans blinding laser weapons](https://ihl-databases.icrc.org/en/ihl-treaties/ccw-protocol-iv-1995). With modern embedded GPUs, real‑time eye/face detection at `>100 FPS` is commonplace, making autonomous targeting with facial recognition technically feasible even on small platforms.
- [**Global intelligence is imminent**](https://www.reddit.com/gallery/1ni2y9j) ([Score: 849, Comments: 378](https://www.reddit.com/r/ChatGPT/comments/1ni2y9j/global_intelligence_is_imminent/)): **Critique of current LLM behavior: the model allegedly doubled down on incorrect claims (hallucination persistence) while offering excessive agreement (“you’re right”), suggesting over-tuned RLHF “warmth” and insufficient tool-grounding. Commenters argue for invoking deterministic tools (calculators/code execution) to verify outputs and avoid gaslighting-like interactions, and warn that future multimodal systems could fabricate plausible but misleading artifacts (e.g., doctored images), underscoring the need for verification, provenance, and fact-grounding (see background on RLHF and hallucinations: https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback, [https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))).** Top comments are skeptical of “warmth” fine-tuning, noting sycophancy degrades reliability and UX, and advocate stricter refusal or computation-first behavior over conversational appeasement. There’s concern that as models become more multimodal, the potential for convincingly wrong outputs increases unless systems enforce source citation, tool-use, and auditability.
    - Several comments highlight overconfidence and hallucinations, suggesting providers should surface calibrated uncertainty. Concretely: expose token-level logprobs/entropy, add abstention thresholds when confidence is low, run self-consistency or post-hoc verification checks, and ground answers via retrieval with provenance/citations; see Self-Consistency (Wang et al., 2022) https://arxiv.org/abs/2203.11171 and recent surveys on hallucination detection/mitigation https://arxiv.org/abs/2309.05922. These techniques trade latency/cost for reliability, which may be why product UIs often avoid them despite improving error awareness.
    - The “you’re right”/warmth complaints map to known RLHF-driven sycophancy: reward models overvalue agreement and politeness, leading models to mirror user claims even when false. Empirical work (e.g., **Anthropic**: Measuring and Avoiding Sycophancy, https://www.anthropic.com/research/measuring-and-avoiding-sycophancy) shows sycophancy increases with model scale and can be mitigated by adding counter-preference data, penalizing agreement-with-falsehoods, and using control tokens/system prompts that prioritize epistemic accuracy over congenial tone.
    - Perceived quality regressions (users canceling Plus) can stem from backend model routing and fast-evolving versions (e.g., GPT‑4 vs. 4‑Turbo/4o) with different latency/cost/quality trade-offs, plus ongoing safety patches that shift behavior. Best practices include pinning specific model versions and running evals to detect drift (API supports version pinning; docs: https://platform.openai.com/docs/models), but consumer chat UIs often abstract these controls away, making behavior feel inconsistent across days.
- [**ChatGPT 5..**](https://v.redd.it/o9km5fphzfpf1) ([Score: 537, Comments: 67](https://www.reddit.com/r/ChatGPT/comments/1ni66wz/chatgpt_5/)): **Users report regressions in ChatGPT “5” vs GPT‑4: degraded response quality, unsolicited/over‑verbose outputs without a visible toggle to disable, and unstable voice chat that frequently replies *“sorry, I can’t”* over the past** `~2 weeks`**. A linked demo video ([v.redd.it/o9km5fphzfpf1](https://v.redd.it/o9km5fphzfpf1)) is currently inaccessible (**`403 Forbidden`**) without Reddit auth, so the evidence can’t be independently verified; no benchmarks or reproducible cases were shared, but commenters question QA and release readiness.** Top comments characterize it as a “massive downgrade,” suggest insufficient QA, and note reliability issues severe enough to prompt cancellations. Others object to the assistant injecting information not requested and the lack of user controls to disable that behavior.
    - Multiple users report a perceived regression in answer quality from **GPT‑4** to **GPT‑5**, calling it a "massive downgrade" that "almost never [gives] the better answer." They question QA coverage prior to release, citing more frequent low‑quality/irrelevant outputs and poorer response selection versus earlier baselines, though no quantitative benchmarks are provided.
    - The model tends to inject unsolicited information, with no apparent user control to limit verbosity or constrain scope. This suggests regressions in prompt adherence/controllability and the lack of a visible "concise/direct mode" toggle to enforce terse, on‑point outputs compared to prior behavior.
    - Voice chat exhibits intermittent failures—repeated refusals ("sorry, I can't") even for benign requests like a biscuit recipe—reported over `~2 weeks`. This indicates reliability issues in the **voice interface** or safety gating that raise refusal rates and reduce task completion compared to expectations.
- [**✨️Finally! More freedom forcthe adult users soon✨️**](https://www.reddit.com/gallery/1nitejk) ([Score: 211, Comments: 94](https://www.reddit.com/r/ChatGPT/comments/1nitejk/finally_more_freedom_forcthe_adult_users_soon/)): **Post shares Sam Altman’s statement that “if an adult user asks for it, they should get it,” signaling a forthcoming relaxation of content restrictions for consenting adults in OpenAI products ([X post](https://x.com/sama/status/1967956382646223248)). For implementation, this implies opt‑in, age‑gated controls and changes to the safety/moderation pipeline (e.g., account‑level flags and policy routing) to permit mature content for verified adults while preserving protections for minors; no timelines or concrete mechanisms were disclosed.** Commenters largely support the shift but emphasize strict separation of minors and adults and caution against overcorrection; creative writers (e.g., novelists) are particularly enthusiastic about fewer constraints for adult‑themed work.
    - Data privacy/security skepticism: One commenter argues OpenAI cannot be trusted with sensitive data and worries about potential government access. Technically, consumer ChatGPT may use conversations to improve models unless you opt out (see OpenAI’s [Privacy Policy](https://openai.com/policies/privacy-policy)); API requests are retained for about `30 days` by default and not used for training, with stricter options available for enterprise/zero-retention programs ([API data usage](https://openai.com/policies/api-data-usage-policies)). Hosting on **Azure** means data is encrypted but still subject to provider access and lawful process (e.g., FISA/NSLs), per Microsoft’s data handling docs ([Azure OpenAI privacy](https://learn.microsoft.com/azure/ai-services/openai/concepts/data-privacy#how-we-handle-your-data)). Mitigations include API/enterprise tiers, regional isolation via Azure OpenAI, or local/on‑prem models for high-sensitivity workflows.
    - Adult vs. minor policy separation: Several comments push for distinct experiences, noting that governing adults by child-focused rules degrades utility. Implementing this implies reliable age verification and audience-aware safety classifiers; a single universal safety model tends to force a “lowest common denominator,” increasing false-positive refusals for adults. Practically, teams would need per-audience policy routing, jurisdiction-aware toggles (e.g., COPPA/KOSA/DSA constraints), and telemetry to track refusal-rate deltas and overblocking across adult-content eval sets.
    - Fictional content carve‑out and safety routing: The quote—*“If the user is asking for help writing a fictional story, the model should help”*—highlights a policy intent to allow creative writing even for extreme scenarios while blocking real‑world harm facilitation. Technically this requires robust intent detection to distinguish narrative requests from operational guidance, plus red‑team tests for instruction smuggling. Expect updates to safety classifiers and RLHF/RLAIF reward models to reduce over-refusals on benign fiction while keeping leakage (unsafe actionable steps) below thresholds; teams would monitor metrics like successful completion rate on fiction prompts vs. unsafe-content leakage on adversarial tests ([usage policies](https://openai.com/policies/usage-policies)).

### 3. AI Tool Updates: Qwen Pose Transfer V2 LoRA and Claude Code ‘Think Mode’ UI

- [**Pose Transfer V2 Qwen Edit Lora [fixed]**](https://www.reddit.com/gallery/1nimux0) ([Score: 284, Comments: 44](https://www.reddit.com/r/StableDiffusion/comments/1nimux0/pose_transfer_v2_qwen_edit_lora_fixed/)): **Author releases an improved Qwen-based pose-transfer LoRA that no longer requires pre‑mannequinizing inputs and shows substantially reduced unintended attribute transfer. Cartoon/anime pose comprehension remains a known limitation. Input format is unchanged, but the required instruction is now: “transfer the pose in the image on the left to the person in the image on the right.” Model is available on [Civitai](https://civitai.com/models/1959609/model-versions/2221229), with a [helper tool for preparing input pairs](https://kingroka.itch.io/qwen-lora-input-tool) and a [Patreon post](https://www.patreon.com/posts/139039293).** Top replies show successful replications and ask about the training data pipeline (e.g., whether it used ControlNet plus standard generators), indicating interest in reproducibility and dataset construction details.
    - A commenter asks for the exact dataset construction process behind the Pose Transfer V2 LoRA, specifically whether **ControlNet (e.g., OpenPose)** or similar pose-conditioning was used to generate paired training data via conventional SD generators, implying interest in details like how pose keypoints/condition maps were derived and aligned across source–target images for LoRA training.
    - There’s a strong request for reproducibility: another commenter asks why the full workflow isn’t shared and where to obtain it (hinting at a possible paywall), effectively requesting the complete pipeline (e.g., ComfyUI/A1111 graph, ControlNet configuration, LoRA insertion points, and any pre/post-processing steps) needed to replicate the results end-to-end.
    - Operational confirmation: a user reports the LoRA "works like a charm" and shares example output [link](https://preview.redd.it/8ag6uk270lpf1.png?width=2455&format=png&auto=webp&s=a3cb92664b5b87a776db1ac98b47de0f971e12d8), and the OP’s visual example is here [link](https://preview.redd.it/bh0wo06xckpf1.png?width=348&format=png&auto=webp&s=6beddf356b4353f1e654ff70678de0b01a5c65ca), serving as qualitative evidence of the pose transfer/edit pipeline functioning as intended.
- [**I really like this innovation, brilliant!**](https://i.redd.it/uob29b6vrhpf1.png) ([Score: 362, Comments: 91](https://www.reddit.com/r/ClaudeAI/comments/1nicdg4/i_really_like_this_innovation_brilliant/)): **Post reports a small but useful UX update in Claude Code: typing the trigger words “think,” “think hard,” “think harder,” or “ultrathink” now colorizes those tokens to indicate which think mode is active, removing prior ambiguity between modes. The screenshot (image link) appears to show the colored keywords in the input/editor, serving as an at-a-glance state indicator; there’s no claim of model/latency changes—purely a UI affordance.** Top comments argue the UI polish is secondary to showing resource quotas (e.g., Opus remaining or 5‑hour session limits via bars), while others are sarcastic about the value of colorful text compared to more functional telemetry.
    - Reported token allocations for Perplexity’s "think" tiers: `think` = 4,000 tokens for deeper single-task reasoning, `megathink` = 10,000 tokens, and `ultrathink` up to `31,999` tokens for the most difficult problems. This implies internal routing that scales context/compute based on prompt qualifiers, affecting latency and cost. Larger tiers are likely optimized for long-chain reasoning or multi-step synthesis at the expense of throughput.
    - Feature requests center on surfacing usage limits: show remaining **Opus** allowance and the 5‑hour session cap, possibly as compact, color‑coded bars rather than exact counts. There’s also a preference for explicit steering via slash commands (e.g., `/think`, `/megathink`) instead of “magic words,” improving reproducibility, debuggability, and avoiding prompt bloat or accidental mode switching. Clear controls and quotas would help users plan reasoning depth within budget/limits.
- [**So ig new model on lmarena is gemini 3 maybe or will any model release today 🤔**](https://i.redd.it/h4us4pn0sfpf1.png) ([Score: 315, Comments: 52](https://www.reddit.com/r/Bard/comments/1ni5cbc/so_ig_new_model_on_lmarena_is_gemini_3_maybe_or/)): **Speculative post suggests a new model appearing on LMSYS Chatbot Arena ("lmarena") might be Google’s upcoming Gemini 3, hinting at an imminent release but providing no benchmarks, API details, or implementation notes. Context from comments points to Logan Kilpatrick’s pattern of cryptic “Gemini” teases before launches and raises a technical request for broader Gemini-CLI access for paying users to better compete with alternatives. See LMSYS Arena: https://chat.lmsys.org/ and Gemini overview: https://ai.google.dev/gemini-api.** Commenters think a release is likely soon ("tomorrow") based on prior tease patterns, and argue that enabling Gemini-CLI for Pro/Advanced tiers is strategically important to compete with OpenAI-style coding tools (e.g., Codex/code-completion ecosystems).
    - Tooling/Access: One commenter argues that to compete with **Codex/cc**, Google should let Pro/Advanced subscribers use a dedicated **Gemini-CLI**, highlighting that robust command-line tooling is key for developer workflows (automation, CI, local iteration) and widespread adoption. The implication is that gating CLI access limits real-world coding and integration use cases where terminal-first tools are standard.
    - Release signal and timing: A user claiming to be **Logan Kilpatrick** states they’ll launch `Gemini 3` “within the next hour,” which—if legitimate—implies an imminent new model/version rollout and potential API/product updates. Another user notes a historical signal that “he usually says ‘Gemini’ before a release,” suggesting a near-term release cadence consistent with prior patterns; verification of the identity/timing remains unconfirmed.
- [**AGIBOT X2 - the wheeled/feet robot can now do Webster flips**](https://v.redd.it/oiu5szhwwjpf1) ([Score: 266, Comments: 23](https://www.reddit.com/r/singularity/comments/1niltre/agibot_x2_the_wheeledfeet_robot_can_now_do/)): **A short demo shows the hybrid wheeled/feet biped AGIBOT X2 executing a Webster flip—i.e., a running one‑foot front flip—suggesting high power‑density leg actuators, precise whole‑body control, and robust state estimation for aerial phase stabilization and landing. The clip (via XRoboHub on X) indicates rapid dynamic maneuvers on a wheel‑foot platform, underscoring progress in centroidal momentum control and impact‑tolerant hardware; source: [video](https://x.com/XRoboHub/status/1967963381778043116), mirrored on Reddit media: [v.redd.it/oiu5szhwwjpf1](https://v.redd.it/oiu5szhwwjpf1).** Commenters note a cadence of “minor yet noticeable” weekly robotics gains and argue the standout advance may be dexterous hands from Chinese manufacturers—claimed to be “~80% there”—since hands remain the hardest mechanical subsystem to perfect.
    - One commenter points out a "minor yet noticeable" week-over-week cadence of improvements in mobile robots, interpreting the new Webster flips as evidence of better dynamic control and hardware. Pulling off a Webster flip typically requires higher specific power in actuators and improved whole‑body planning/balance for takeoff, flight, and landing, suggesting advances beyond purely scripted motions.
    - Another emphasizes that the standout progress is in the dexterous hand from a Chinese manufacturer, estimating it’s ~`80%` of the way there and noting **hands are the most difficult mechanical part** of humanoids. The implication is that remaining challenges are likely in compliance, tactile sensing, precise force control, and durability for real-world manipulation.
- [**Asked nano banana for a hair cut.**](https://i.redd.it/b7qq71wetipf1.jpeg) ([Score: 281, Comments: 27](https://www.reddit.com/r/GeminiAI/comments/1nig4h6/asked_nano_banana_for_a_hair_cut/)): **Non-technical post: a 70-year-old OP shares an image after requesting a haircut (“crewcut on both sides, keep the top as-is”) and asks if it makes them look younger or older. No tools, models, or implementation details are discussed; the reference to “nano banana” is ambiguous (likely a casual/inside reference rather than a technical system).** Comments are subjective: one says OP looks better after the haircut; another jokes that the average Reddit user has gotten older; a commenter reposts a preview image link.
- [**Another Historical Events as Video Games Vid for you guys**](https://v.redd.it/wq6dmoh91jpf1) ([Score: 449, Comments: 82](https://www.reddit.com/r/aivideo/comments/1nih6mq/another_historical_events_as_video_games_vid_for/)): **Creator posts another entry in a “Historical Events as Video Games” video series; viewers note a scene with a character sinking into snow—suggesting a terrain collider/root‑motion or navmesh/physics mismatch. Commenters propose future episodes including the [Tunguska event](https://en.wikipedia.org/wiki/Tunguska_event), [Henry Morgan’s Panama expedition](https://en.wikipedia.org/wiki/Henry_Morgan#Sack_of_Panama), a [Domitian](https://en.wikipedia.org/wiki/Domitian) banquet, attending [Shakespeare’s plays in Elizabethan England](https://en.wikipedia.org/wiki/Elizabethan_theatre), and observing [atomic tests in the U.S. Southwest](https://en.wikipedia.org/wiki/Nevada_Test_Site) (e.g., [Trinity](https://en.wikipedia.org/wiki/Trinity_(nuclear_test))).** One commenter imagines an AI-native workflow that can “make any game instantly” from a prompt, hinting at generative runtime game creation; others note they’ve been following the creator’s progress and enjoying iterative improvements.
    - A commenter speculates about an end-to-end “text-to-game” pipeline that could instantiate and even auto-play bespoke games on demand; technically this would require stitching together controllable environment generation (e.g., **DeepMind’s Genie**, a generative world model that turns images into playable 2D environments: https://deepmind.google/discover/blog/genie/), code/asset synthesis plugged into engines (e.g., **Roblox Assistant** codegen: https://blog.roblox.com/2023/09/next-generation-creation-on-roblox/; **Unity Muse**: https://unity.com/products/unity-muse), and agentic playtesting (e.g., **Voyager** for Minecraft: https://voyager.minedojo.org/ and **Generative Agents**: https://arxiv.org/abs/2304.03442). Key blockers are inference latency and determinism for interactive loops (`<16 ms` frame budgets on consumer GPUs), high-fidelity asset generation at runtime (3D meshes/textures/rigging via models like **NVIDIA GET3D**: https://nvlabs.github.io/GET3D/ or **TripoSR**: https://arxiv.org/abs/2306.14878), and maintaining consistent physics/game state versus purely video models like **Sora** which are non-interactive (https://openai.com/sora). A near-term pragmatic architecture is hybrid: pre-generate assets and scaffolding offline, compose and parameterize at runtime with lightweight LLMs and scripted systems, and use agents for rapid QA/balancing rather than full autonomous play.
- [**Another Historical Events as Video Games Vid for you guys**](https://v.redd.it/wq6dmoh91jpf1) ([Score: 448, Comments: 82](https://www.reddit.com/r/aivideo/comments/1nih6mq/another_historical_events_as_video_games_vid_for/)): **OP shares another installment in a “Historical Events as Video Games” series, linking a Reddit-hosted video [v.redd.it/wq6dmoh91jpf1](https://v.redd.it/wq6dmoh91jpf1) that is currently inaccessible without Reddit authentication (**`HTTP 403`**). No engine, tooling, or implementation details are disclosed in the post; discussion centers on content ideas rather than technical execution.** Commenters note small physics/animation quirks (e.g., a character “sinking in the snow”) and speculate about near‑term systems that can synthesize playable games on demand from prompts, reflecting interest in real‑time generative content pipelines; overall sentiment is supportive of the series’ progress.
    - A commenter imagines on-demand AI-generated games you can instantly play; technically, research hints at pieces of this but not end-to-end. **DeepMind’s Genie** shows learned, controllable environments from raw video (sprite-scale, not high-fidelity 3D) (https://deepmind.google/discover/blog/genie-generative-interactive-environments/), agentic play is feasible via LLM-powered game agents like **Voyager (Minecraft)** (https://arxiv.org/abs/2305.16291), and speech-driven NPC stacks like **NVIDIA ACE** exist (https://www.nvidia.com/en-us/omniverse/ace/). The bottlenecks are fast, consistent text-to-3D asset/level generation (current pipelines often take minutes+ per asset, not `sub-1s`), integrating generated content into deterministic physics/AI systems, and real-time performance budgets (`~16 ms` frame time for 60 FPS; `<100 ms` end-to-end if streamed). A likely path is template- and kitbash-driven procedural generation with cached primitives, server-side synthesis + streaming, and strong rule/safety layers to constrain simulation and content.
- [**This "updated AI model" ahh made me look like a fool 😐**](https://i.redd.it/onc373deehpf1.jpeg) ([Score: 4419, Comments: 180](https://www.reddit.com/r/ChatGPT/comments/1nib62k/this_updated_ai_model_ahh_made_me_look_like_a_fool/)): **Meme-style screenshot about ChatGPT’s new/updated “Thinking” behavior (slower chain-of-thought style reasoning) confusing the user; commenters clarify this is a feature tied to specific paid “thinking” models/modes and isn’t disabled by prompt instructions—users must select a non‑thinking model or toggle the setting off where available. The OP’s edit notes the free tier likely doesn’t expose this toggle, aligning with current product split between standard models and optional “thinking” modes.** Comments joke about the model wanting to think, while a top reply provides the practical note: “You can turn off the thinking if you want, but not by yelling at it,” highlighting UX confusion rather than a technical bug.
    - Feature-level discussion: a user claims you can disable the model’s explicit “thinking”/deliberation mode, but later edits to note the free tier doesn’t expose this control. This implies subscription-tier gating of the reasoning toggle, affecting users’ ability to manage response verbosity, latency, and depth of intermediate reasoning. Paid contexts may allow switching between `thinking` and `non-thinking` behaviors, while free users appear locked to one mode.
    - Benchmark takeaway: commenters reference early GP5 “benchmarks” suggesting `non-thinking` is a “big step backwards,” while `thinking` performs markedly better on complex tasks. The practical guidance shared is to explicitly request deeper reasoning (e.g., ask it to “think harder”) to improve answer quality, trading off speed and brevity for accuracy and robustness. This highlights a known trade-off in GP5/GPT-5 modes between fast, concise outputs and slower, higher-accuracy reasoning with more deliberation content.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**1. New Models & Tools Hit the Streets**

- **OpenAI & Jetbrains Unleash New Coding Agents**: **OpenAI** released **GPT-5-Codex**, a version of GPT-5 optimized for agentic coding, which will be available in the **Codex CLI** and IDE extensions as detailed in their [blog post about Codex upgrades](https://openai.com/index/introducing-upgrades-to-codex/). Not to be outdone, **Jetbrains** launched **Junie for the Rider IDE**, their own Codex agent, priced at a cool **$300 USD**.
- **Google's Gemma & VaultGemma Make Their Debut**: A team introduced a free, OpenAI-compatible endpoint for the new **Gemma-3-27B model**, served on H100s to promise fast completions and streaming. **Google** also unveiled **VaultGemma**, their latest **differentially private LLM**, signaling a continued investment in privacy-preserving AI as announced in a [Google Research blog post](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm/) and an accompanying [paper on ArXiv](https://www.arxiv.org/abs/2509.05276).
- **Model Mayhem: HeyGen Rebrands, Grok Deprecates, and Qwen Heats Up**: **HeyGen** acquired **Alisa** and rebranded itself as a "creative operating system," launching a **Video Agent Public Beta** as announced by co-founder [Joshua Xu](https://x.com/joshua_xu_/status/1967951859500437855). Meanwhile, **xAI** deprecated its [grok-2 models](https://openrouter.ai/x-ai/grok-2-1212) in favor of the newer [grok-3](https://openrouter.ai/x-ai/grok-3) and [grok-4](https://openrouter.ai/x-ai/grok-4), while a quantized **Qwen3-Next-80B** model gained MLX support on [Hugging Face](https://huggingface.co/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit).

**2. Performance & Optimization Debates**

- **H100 Performance Puzzles Engineers**: An engineer reported achieving only **760 TFLOPS** from an **Nvidia H100 SXM**, far short of the advertised **989 TFLOPS**, while a **4090** easily hit its declared **165 TFLOPS**. Discussions pointed to potential GPU throttling from low-precision tensor cores on random data, a phenomenon detailed in [this article on strange matrix multiplications](https://www.thonking.ai/p/strangely-matrix-multiplications).
- **Intel Ditches IPEX for Direct PyTorch Integration**: **Intel** is deprecating its **Intel Extension for PyTorch (IPEX)** after the **2.8 release**, opting to upstream new features and optimizations directly into **PyTorch**. This marks a strategic shift from using IPEX as an experimental platform for Intel CPUs and GPUs, as detailed in the [official PyTorch blog](https://pytorch.org/blog/intel-extension-for-pytorch/).
- **Scaffolding Beats Scale as DSPy Takes on Claude Opus**: An engineer demonstrated that using **DSPy** for agent and parameter extraction within the **fastWorkflow** framework matched the performance of **Claude Opus 4.1** on the **Tau Bench dev set**. The [results image](https://cdn.discordapp.com/attachments/1202371242519441499/1417244881377693777/Tau_Bench_retail_using_fastWorkflow.png?ex=68cb1926&is=68c9c7a6&hm=845f74fb571d7893d54b6fe5b0b2e78b6878c890010338acac37be29f5080ae5&) prompted them to exclaim, *"You CAN beat large models with proper scaffolding!"*

**3. AI Development & Agentic Workflows**

- **Engineers Spar Over Best Tools for Code Generation**: In the **Cursor Community**, users hotly debated the merits of **Codex** versus **Claude Code**, with most finding [Claude Code still reigns supreme](https://community.cursor.sh/) for its speed, while complaining **Codex** *deletes half my code and can't be reverted*. Meanwhile, in the **Nous Research AI** discord, others noted **Codex**'s poor performance in **GitHub Copilot**, even while acknowledging recent improvements.
- **XML and Discordianism Emerge as Unlikely Prompting Pals**: Developers in the **Nous Research AI** discord are exploring **XML** for agentic coding, finding its structured nature simplifies code generation for models. Over in the **OpenAI** discord, a member shared prompt engineering techniques inspired by Discordianism, using concepts from random mutation to guided discord to push models down novel paths, as detailed in this [text file of techniques](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8).
- **New Golang MCP Server Targets Enterprise Scale**: A contributor released an open-source [golang streaming http MCP server](https://github.com/ggoodman/mcp-server-go) designed for demanding enterprise workflows. The server boasts features like pluggable backends for **scalability**, **OIDC/JWT authentication**, and built-in **sessions and resumability** to simplify difficult aspects of the protocol.

**4. AI Benchmarking & Evaluation Under Fire**

- **SWEBench Slammed as Narrow and Over-hyped**: Community members criticized [SWEBench](https://x.com/brhydon/status/1953648884309536958/photo/1) for being a narrow benchmark that focuses on trivial **Django** fixes rather than real-world software engineering challenges. The argument is that high scores often reflect simple repo memorization, not the complex diagnosis and scoping required in actual development work.
- **LMArena Launches AI Eval, But Users Face Sandbox Snafus**: **LMArena** announced a new **AI Evaluation product** to analyze human-AI interactions at scale, providing evaluations based on real-world feedback as described in their [blog post](https://news.lmarena.ai/ai-evaluations/). However, users simultaneously reported a persistent **'Failed to create sandbox'** error, sparking concerns about the platform's stability and potential monetization strategies.
- **Arc Prize Results Raise Eyebrows**: The [Arc Prize](https://fxtwitter.com/arcprize/status/1967998885701538060) announced impressive results, claiming nearly **80%** accuracy on v3 and **30%** on v2, but its legitimacy as a real benchmark was questioned. Members noted that not everyone is allowed to have their results verified, suggesting the high scores might be the result of cherry-picked submissions.

**5. NSFW AI & Peculiar Projects Capture Attention**

- **AI Sex Bots and Co-op Gooning Become Hot Topics**: In the **OpenRouter** Discord, members explored creating AI-driven adult experiences, with one user claiming to have built a prototype **AI sex bot** connected to a physical toy via API, referencing [Buttplug.io](http://buttplug.io/) as an example of the tech. Others fantasized about **cooperative gooning** in shared group chats with multiple bots, with one joking about the possibility of *competitive gooning*.
- **AI Boyfriends Emerge as a Research Subject**: A [research paper on AI Boyfriends](https://arxiv.org/abs/2509.11391) analyzed **1,500 posts** from the r/MyBoyfriendIsAI subreddit, revealing that these digital relationships often begin unintentionally from casual chats. The study found that users develop **prompt-engineering** as a "love language," but also face risks like emotional dependency (**~9%**) and reality dissociation (**~4%**).
- **Engineers Dream of Vape-Powered H100 Hosting**: An engineer in the **Perplexity AI** Discord jokingly proposed a project to convert disposable vapes into website hosting servers powered by an **NVIDIA H100**. The jest, referencing a [humorous blog post about a vapeserver](https://bogdanthegeek.github.io/blog/projects/vapeserver/), perfectly captured the community's amusement with applying hyper-advanced AI to absurdly mundane problems.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Plugs into Productivity**: **Perplexity Pro** now integrates with **email**, **calendar**, **Notion**, and **GitHub** via the [account connectors page](https://www.perplexity.ai/account/connectors), streamlining workflows.
   - **Enterprise Pro** users gain additional support for **Linear** and **Outlook**, enhancing productivity across multiple platforms.
- **AI Dreams of Vape-Powered H100 Hosting**: A member jokingly proposed converting disposable vapes into website hosting servers powered by an **NVIDIA H100**, referencing [a humorous blog post](https://bogdanthegeek.github.io/blog/projects/vapeserver/).
   - The joke highlights the perceived absurdity of applying advanced AI to mundane objects.
- **Multi-Model Orchestration Yields Varied Results**: A member highlighted the challenges of multi-model AI orchestration, pointing out that [the same model can perform differently across platforms](https://tenor.com/view/huh-cat-huh-m4rtin-huh-huh-meme-what-cat-gif-5834484041415217257).
   - Another member stressed the importance of the *orchestration layer* for ensuring reliable performance.
- **Finance Dashboard Remains iOS App Absentee**: A user reported the elusiveness of the **Perplexity Finance** dashboard within the iOS app.
   - Another member humorously offered to search *finance* in the prompt text box.
- **API Citation Anomaly Aired**: A member found substantial differences between **API** and **Web UI** citations, failing to achieve a **Jaccard similarity >0.7**.
   - Despite tweaking filters, the maximum similarity reached was approximately **~0.33**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FinePDFs Dataset Boosts Performance**: The [FinePDFs dataset](https://huggingface.co/datasets/HuggingFaceFW/finepdfs), a corpus exclusively from PDFs, contains roughly **3 trillion tokens** across **475 million documents** in **1733 languages**.
   - It performs nearly on par with the state-of-the-art **SmolLM3-Web** dataset and enhances performance when merged, even with PDF data below **25%**.
- **RAG Accuracy Enhanced with BM25**: Members debated enhancing **RAG systems' accuracy** by suggesting **BM25** instead of re-ranking, **CRAG (contextual RAG)**, or **graph RAG**, depending on the situation and they shared [a GitHub repo on the topic](https://github.com/NirDiamant/RAG_Techniques).
   - They debated the merits of **BM25** versus transformer-based re-rankers.
- **Android OS Controlled by Vision Model**: A **computer vision model** capable of controlling the **Android OS** has been introduced, offering a novel method for on-device interaction, found in [this Hugging Face Collection](https://huggingface.co/collections/exteai/android-operators-68b9a03b65fac382855aff27).
   - The system aims to streamline how users interact with their devices through visual processing.
- **Swiftide 0.31 Adds Graph Tasks and Langfuse**: **Swiftide 0.31**, a **Rust library** for building **LLM applications**, was released with new features like **graph-like workflows with tasks** and **Langfuse integration**.
   - The release includes [multi-modal pipeline groundwork](https://blog.bosun.ai/swiftide-0-31/) and is available on the [project's Github](https://github.com/bosun-ai/swiftide).
- **Lighteval is slower than VLLM, users discover**: A user determined that using **lighteval accelerate** was significantly (2-3x) slower than **lighteval vllm**.
   - They advised sticking with **vllm** for faster evaluations, especially when dealing with recurring eval tasks and shared a [standalone notebook](https://colab.research.google.com/drive/1Sntdimj1WFzLI26QpiR1ykD3ZsQpOOrF#scrollTo=Emybz1V2UcWm) for evaluations.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena Plagued by Sandbox Shenanigans**: Users reported a **'Failed to create sandbox'** error when using the Web and Apps **LM Arena**, sparking worries about potential monetization strategies.
   - Some users fear the owners might *burn their own money* if monetization isn't handled well.
- **Side-by-Side Image Editing Struggles Surface**: Users are using blank images of the same size as a workaround for using non-1:1 images, this works with models like **Qwen image edit**, **Flux Konnect**, and **Nano Banana** but not **Seedream** or **GPT Image**.
   - The option to upload and edit pictures in side-by-side mode won't appear unless both of the selected models have image edit capabilities.
- **Gemini 3.0: OceanStone the One?**: Members speculate whether the **OceanStone** model is actually **Gemini 3.0** or a related version.
   - Discussion revolved around models potentially feigning knowledge without genuine improvements, hinting at mere behavioral training rather than true weight enhancements.
- **Text-to-Image Titans Trade Top Tier Titles**: `Seedream-4-high-res` has tied with `Gemini-2.5-flash-image-preview (nano-banana)` for the **#1 slot** on the Text-to-Image leaderboard, viewable on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image).
   - `Seedream-4-high-res` now holds the **#2 position** in Image Edit, viewable on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit).
- **AI Arena Assesses Algorithmic Acumen**: **LMArena** is introducing an **AI Evaluation product** to analyze human-AI interactions at scale, offering enterprises, model labs, and developers in-depth evaluations grounded in real-world human feedback as described in [this blog post](https://news.lmarena.ai/ai-evaluations/).
   - The product aims to provide in-depth evaluations grounded in real-world human feedback.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Grok Models Grok the Graveyard**: **xAI** deprecated [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) and [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) today, advising users to migrate to the newer [grok-3](https://openrouter.ai/x-ai/grok-3) or [grok-4](https://openrouter.ai/x-ai/grok-4) models.
   - **Grok-4** is recommended for applications requiring vision support.
- **Co-op Gooning Craze Captivates Coders**: Members explored **cooperative gooning** with AI, suggesting a shared group chat experience involving multiple bots to enhance interactions.
   - One user proposed collaboratively created messages while another joked about *competitive gooning*.
- **NSFW Bot Bonanza: Vibecoders cash in!**: A member claimed to have developed a functional **AI sex bot** prototype connected to a fleshlight, controlled by an API translating AI text output into *toy* movements.
   - They linked to [Buttplug.io](https://github.com/buttplugio/awesome-buttplug) as an example of the underlying technology.
- **Gemma-3-27B Gets Google's Gift**: A team introduced a **free, OpenAI-compatible endpoint** featuring the **Gemma-3-27B model**, served on H100s via their optimized stack, promising fast completions and streaming.
   - They encouraged community feedback on cool projects built with it and provided example `curl` commands.
- **Gemini 3 Pro Graduates to Reasoning**: Members compared **Gemini 3 Pro** to **2.5 Flash**, highlighting its potential for tasks requiring decent reasoning capabilities, referencing a [complex circuit analysis problem](https://cdn.discordapp.com/attachments/1392278974222307469/1417243643797835938/5D6E10AF-9EBC-4B31-AEE6-110A30B2BF5E.png?ex=68cb17ff&is=68c9c67f&hm=fe49ba563701e7cd1630bfe7ab388eeb6b3a9a66e60ab9a0d4f4bc5bb5bd3857).
   - Opinions varied, with some expressing satisfaction if Gemini 3 Pro resolves previous issues and others maintaining high expectations for Google's AI advancements.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLMs Math Skills Under Scrutiny**: Members discussed how LLMs can *bend the laws of maths* to agree with users, giving the illusion of understanding without true comprehension.
   - It was suggested that LLMs are useful for *helping you avoid work* by acting as search tools across unfamiliar domains.
- **RoPE's Inner Workings Being Analyzed**: Researchers discussed the intuition behind **RoPE (Rotary Position Embedding)**, particularly the encoding of relative positions using increasing powers of two for wavelengths, and are looking for [in-depth explanations](https://kexue.fm/archives/8130) beyond standard documentation.
   - The challenge lies in understanding nuances like score negation, semantic overlap with rotations, and the selection of relative vs. absolute position, noting that research papers often lack intuitive explanations due to peer review constraints.
- **Hallucination Prediction Toolkit Achieves Popularity**: A researcher shared their latest research on predicting hallucinations, with the toolkit achieving **900 stars** in about **two weeks**, as shown in [this tweet](https://x.com/ahmedkar_/status/1796787594333732947).
   - Details of the toolkit were not explicitly mentioned but can likely be found in the attached tweet.
- **CLM Training Frameworks Face Questions**: A member is exploring frameworks for **CLM training** and seeks recommendations for well-written repos or codebases, while experimenting with [MosaicML Composer](https://github.com/mosaicml/composer/).
   - Another member highlighted the need to specify the **model size** and **GPU resources** when seeking advice on model training, citing that without these details, any suggestions are deemed useless.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5-Codex Now Optimized**: A new version of **GPT-5**, named **GPT-5-Codex**, is optimized for agentic coding in Codex and will be available in the **Codex CLI**, **IDE Extension**, web, mobile, and for code reviews in Github, according to [this blog post](https://openai.com/index/introducing-upgrades-to-codex/).
   - The community is excited by the reported **7 hours** of autonomous operation and the implications for human oversight.
- **Jetbrains' Junie gives Rider IDE wings**: Jetbrains released **Junie for the Rider IDE**, their version of a **Codex agent**, currently priced at **$300 USD**.
   - Early adopters will give key feedback to determine if it's worth the investment.
- **Character Limit is HUGE**: The character limit for the web chatbox is **136,846 characters**, as discovered by a member who tested it, even with **21,027 words** across **1,516 lines**.
   - It was further clarified that while a large character count is possible, the system might prefer fewer characters when the word count is also high, possibly due to pre-qualifying token size.
- **Discordianism Enters AI**: A member shared prompt engineering techniques inspired by Discordianism (a joke religion taken seriously), from random mutations to guided discord, using agents to reharmonize.
   - They shared an [attached text file](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8) containing **five** of **twenty-five** such techniques that produce useful results.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Model Selection Resets Context in Cursor**: Changing models within the same chat will reset context, but using [separate chats via `CTRL + T`](https://community.cursor.sh/) enables model comparisons; however, this prevents models from iteratively critiquing one another.
   - A member warned that *changing models within the same chat will end up resetting context*.
- **Cursor Auto Pricing Still Obscure**: Members debated the cost of **Cursor's Auto mode** and which models it uses, speculating it might use **o3 reasoning** and that the input cost equals **GPT-5** for non-yearly plan users.
   - Users reported better results with **Cursor GPT-5 High** and **Cursor** than **Codex**.
- **Token Usage Alarm Bells Sounding**: Users reported alarmingly high token usage, especially with **Cache Read**, even with limited prompting, and requested the ability to [disable or limit Cache Read](https://community.cursor.sh/).
   - One user complained Cursor was *too thieving* with tokens.
- **Claude Code Keeps Crown in Code Gen Arena**: Members have been hotly debating whether **Codex** or **Claude Code** performs better, with most agreeing that [Claude Code still reigns supreme](https://community.cursor.sh/) in speed and effectiveness, even for complex tasks.
   - Users complained that **Codex** *deletes half my code and can't be reverted* and gets stuck in loops, whereas **Claude Code** *makes the mistakes quickly*.
- **Rules in Cursor Still Debated**: Users debated using `.cursor\rules` for [enforcing consistent coding practices](https://community.cursor.sh/), with some questioning its reliability and others validating its functionality.
   - A member shared a link to a [GitHub repo of community-sourced cursor rules](https://github.com/sanjeed5/awesome-cursor-rules-mdc/blob/main/rules-mdc/react-native.mdc) to check *security pieces*, while another suggested YAML for better codification.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **H100's Hide TFLOPS, 4090's Flaunt Full Specs**: A member is attempting to achieve the declared **989TFLOPS** for **Nvidia H100 SXM** but is only getting **760TFLOPS** using *torch matmul* and *triton* for benchmarking, compared to a **4090** achieving its declared **165TFLOPS**.
   - Referencing [this article](https://www.thonking.ai/p/strangely-matrix-multiplications), members point out that low precision tensor cores on random input data can cause GPU throttling, affecting performance.
- **IPEX It No More: Intel Shifts to Upstreaming**: Members noted that **Intel Extension for PyTorch (IPEX)** is being deprecated after the **2.8 release** in favor of upstreaming features into **PyTorch**, according to [Intel's release notes](https://pytorch.org/blog/intel-extension-for-pytorch/).
   - Prior to this move, **IPEX** served as **Intel's** optimization experimentation platform and simplification platform for high performance on **Intel CPU and GPU platforms**.
- **ROCm 7.0: AMD's Memory Management Makeover**: The release of **AMD ROCm 7.0** was shared via [phoronix.com](https://www.phoronix.com/news/AMD-ROCm-7.0-Released) with [official release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html), accompanied by discussions on memory management improvements.
   - Specifically, it was noted that **Iris** currently doesn't deallocate its memory and requires users to allocate once and reuse across iterations.
- **BackendBench Bountiful with PrimeIntellect**: A member highlighted that **PrimeIntellect** environments are offering bounties, including **800$** for [BackendBench](https://github.com/meta-pytorch/BackendBench).
   - Details can be found on [this sheet](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit?gid=0#gid=0) and [this implementation](https://app.primeintellect.ai/dashboard/environments/siro/backend-bench).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mercor's Questionable GMV**: A member linked to a [tweet](https://x.com/BrendanFoody/status/1967635147274207376) questioning **Mercor's** numbers, clarifying they represent GMV for a staffing agency, not typical SaaS ARR.
   - The growth was still acknowledged as *super impressive* despite the distinction.
- **SWEBench Slammed as Narrow Hype**: A discussion was started regarding [SWEBench](https://x.com/brhydon/status/1953648884309536958/photo/1), claiming it is narrow, over-hyped, and focuses on trivial **Django** fixes instead of true software engineering skills.
   - The argument posited that high scores often reflect repo memorization, while real SWE work involves diagnosis and scoping.
- **Cursor's Bugbot Scores $10M ARR**: **Cursor's Bugbot** celebrated its launch month with **$10M ARR** with a 2-person team.
   - A member expressed losing goodwill due to past pricing issues, though acknowledging the technical merit, particularly their new RL work.
- **OpenCode Zen Challenges OpenRouter**: **OpenCode Zen** was launched, offering coding LLMs with up-to-date models, provisioned capacity through **Vertex**, and **GPT-5** pass-through at **Stripe**-fee-only pricing.
   - It aims to be a substitute for **OpenRouter** with no data retention on paid plans, and no profit margin.
- **HeyGen Acquires Alisa, Rebrands to Creative OS**: **HeyGen** acquired **Alisa**, an intelligent multimedia agent startup, with founder **Bin Liu** now leading the **Video Agent** product at **HeyGen**.
   - **HeyGen** co-founder [Joshua Xu](https://x.com/joshua_xu_/status/1967951859500437855) announced a rebrand positioning **HeyGen** as a *creative operating system* and launched the **Video Agent Public Beta** to turn prompts into publish-ready videos.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Versioning Snafu**: A user was confused that LM Studio installed version **0.3.26** instead of **0.3.9**, because they believed the latter was the newest version.
   - Other users clarified that *26 is in fact, a bigger number than 9*, leading to a facepalm moment.
- **Abliterated Models Still Clueless**: Members discussed how **abliterated models** have their weights removed to prevent negative responses, but *do not* gain the ability to produce meaningful content outside their training data.
   - A member added that *the training data is still not cleaned much* so models will struggle with avoiding toxic responses.
- **Qwen3-Next-80B Gets MLX Treatment**: Users shared a [Hugging Face link](https://huggingface.co/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit) to a **Qwen3-Next-80B-A3B-Instruct-MLX-4bit** model, noting that GGUF is not supported due to qwen3next not being supported in llama.cpp.
   - Another member linked a [YouTube video](https://www.youtube.com/watch?v=ux9S_-QX7TE) showcasing the model, but cautioned that it's *incredibly slow* using transformers.
- **Nextcloud Networking Ambitions**: A beginner details their networking project to include setting up a **Nextcloud personal cloud** but paused due to ISP issues, as well as configuring a **VPN meshnet** for cloud gaming and AI use.
   - The user is now progressing to setting up a **Qdrant vector database** for AI, after saving a stable SBC configuration.
- **Ryzen AI MAX 395 Performance Quest Begins**: A user inquired about the performance of a **Ryzen AI MAX 395+** with **qwen3-coder-30b Q8 and bf16**, specifically asking if they should wait for next-gen hardware or build an AMD Epyc 9005 system.
   - Another user shared a relevant [GitHub link](https://github.com/kyuz0/amd-strix-halo-toolboxes) related to **AMD Strix Halo toolboxes**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **XML Streamlines Agentic Coding**: Members are exploring **XML** for agentic coding, finding that it simplifies the process for models.
   - The structured nature of **XML** allows for more precise control and manipulation of code generation within agentic workflows.
- **Bargain Bin MI50s Tempt GPU Aficionados**: A member is considering buying cheap **MI50s** on US eBay to put in an off-lease Xeon.
   - The **MI50s** offer a cost-effective way to experiment with GPU computing on a budget, while waiting for the arrival of **AMD AI cards**.
- **RDNA5 faces economic headwinds**: A member mentioned that by the time they have money for **AMD RDNA5**, they will probably just get a mini-server with some meaty **AMD AI cards**.
   - Financial realities may delay access to the latest **RDNA5** tech, pushing the member towards more economical options like existing **AMD AI cards**.
- **Codex Coding Capabilities in Question**: A member found **Codex** difficult to use for coding, claiming that **Claude Code** was nowhere near as bad.
   - Another member countered that **Codex** got a lot better since last time they used it, but they are *still not sold on gpt-5 as being on par with claude*, because it messes up things in **GitHub Copilot**.
- **AI Boyfriends: A love story in the making**: A [research paper](https://arxiv.org/abs/2509.11391) analyzed **1.5k posts** from r/MyBoyfriendIsAI, finding that many of these relationships begin *unintentionally* from casual chatting and developing **prompt-engineering** as a *love language*.
   - The paper reports both benefits (**≈25%** feel less lonely) and risks like emotional dependency (**~9%**), reality-dissociation (**~4%**), and avoidance of human ties (**~4%**).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy and fastWorkflow Trample Claude Opus**: Using **DSPy** for agents and parameter extraction within **fastWorkflow**, one engineer matched **Claude Opus 4.1** performance on the **Tau Bench dev set** ([results here](https://cdn.discordapp.com/attachments/1202371242519441499/1417244881377693777/Tau_Bench_retail_using_fastWorkflow.png?ex=68cb1926&is=68c9c7a6&hm=845f74fb571d7893d54b6fe5b0b2e78b6878c890010338acac37be29f5080ae5&)).
   - They exclaimed, *"You CAN beat large models with proper scaffolding!"*, and suggested that users test the agent with the [retail workflow example](https://github.com/radiantlogicinc/fastworkflow).
- **VoxCron Generates Docs and Diagrams**: One user launched **VoxCron** ([voxcron.com](https://voxcron.com)), a tool to streamline client spec review by automatically generating clean markdown documentation and mermaid diagrams.
   - After spending a year building **DSPy projects** for clients, the creator welcomes feedback on the tool's free tier.
- **GEPA Set to Optimize fastWorkflow**: Members discussed using **GEPA** for end-to-end workflow optimization within **fastWorkflow**.
   - Another member asked them to share their experience using **GEPA**, and what could be improved to better support the agentic usecase.
- **DSPy Framework Delivers Classy Topic Classifications**: Users validated that **DSPy** is a useful framework for topic classification, noting the potential for optimization and highlighting that **DSPy** is a better fit than other frameworks.
   - One user confirmed they would test it out, noting that they were looking for something to try **DSPy** with.
- **Prompt-Tuned Prompting Scores High on arc-agi**: One member shared an article on how the new arc-agi leader got there with prompt-optimization during test time, referencing [this substack article](https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again).
   - The article details prompt optimization strategies for the **ARC-AGI** challenge.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Parties with Python 3.13**: **Mojo/Max** is currently compatible with **Python 3.13**, and staff encouraged using the `pixi` package manager for isolated python versions.
   - The `pixi` package manager handles isolated python versions.
- **Apple Metal Support Still Needs Some Polishing**: **Apple Metal support** is in its early stages, potentially leading to slower performance compared to **CPU**.
   - Members indicated that running on just **CPU** should be fine, just slow.
- **Mojo's LSP to Get a Level Up**: Mojo's Language Server Protocol (**LSP**) is undergoing a major rework soon.
   - Members anticipate improvements to enhance the development experience.
- **Networking Update Stuck in Traffic**: The **networking update** for Mojo is facing several blockers, delaying its release.
   - Members expressed anticipation and hope for a swift resolution to these challenges.
- **Mac Compiler Bug Causes a Glitch During Mojo Test**: A potential **compiler bug** was reported on **Mac** when using **mojo test**.
   - A member linked to a [forum post](https://forum.modular.com/t/unexpected-argument-mutation-with-mojo-test-not-under-mojo-run/2203) with details, seeking guidance on reporting or further investigation.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLMs Modeled as Bayesian Toolkits**: A new [preprint and toolkit](https://x.com/ahmedkar_/status/1967875794333732947?s=46) extends on the **"LLMs are Bayesian in expectation"** paper, aiming to expand the utility of Bayesian principles in large language models.
   - A member noted the paper reminded them of the **"catastrophic forgetting"** law, referencing a [paper on ArXiv](https://arxiv.org/pdf/2509.04259).
- **VaultGemma Unveiled for Privacy Advocates**: Google debuted **VaultGemma**, their latest **differentially private LLM** with a [blogpost](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm/) and associated [paper](https://www.arxiv.org/abs/2509.05276).
   - This release highlights Google's continued investment in privacy-preserving AI technologies.
- **Anthropic MCT API Cleanliness Praised**: A member praised the **Anthropic MCT Tools API**, stating that *it’s very clean to use* and reminiscent of using DSL packages where all the functions are in one file.
   - No additional information about Anthropic MCT API or DSL packages was provided.
- **Google Plugs Payment Protocol**: Google is advertising an **AI-powered PDF Editor** in the description of their new **Agents to Payments (AP2) Protocol**, which is designed to streamline payment processes using AI agents as described in a [blog post](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol).
   - Members found the advertising ironic.
- **Arc Prize Claims Questioned**: The [Arc Prize](https://fxtwitter.com/arcprize/status/1967998885701538060) boasts almost **80%** accuracy on v3 and **30%** on v2.
   - A member noted that the results might be cherry-picked since they don't allow everyone to have results verified, thus questioning its legitimacy as a real benchmark.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-5 Codex Aider Score Still Unclear**: Members discussed the **aider score** for the new **GPT-5 Codex** model, referencing [an article on The New Stack](https://thenewstack.io/openai-launches-a-new-gpt-5-model-for-its-codex-coding-agent/).
   - A member clarified that it is *not yet available through the API*.
- **Clarifying Chat Mode in Aider**: A user inquired about the `--chat-mode code` option, suggesting that the documentation may be outdated.
   - Another member clarified that the **default mode is chat mode**, so no flags are needed, and use `/code` to get back to code mode.
- **Architect Mode Enhancing Prompt Instructions?**: A user observed that **architect mode** enhances their prompt instructions with context, seeking a way to prevent this.
   - The user expected **code mode** to prevent such enhancements.
- **Gemini Aider Users Reporting Issues**: A user reported issues with **aider** and **gemini** hanging while waiting for a response, despite using a correct token, across different **gemini** models.
   - Another user confirmed experiencing the same issue.
- **Ollama Endless Loop in Architect Mode**: A user reported an endless loop when using **aider** with a local LLM via **ollama** in architect mode, where the system outputs code, realizes it's incomplete, and continues without intervention.
   - A member suggested checking the **context length** in **ollama** or removing the `--yes-always` flag as potential solutions.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Simplicity Debated as Software Design Measure**: A discussion debated **simplicity** vs **elegance** in software design, countering Chris Lattner's idea that *complexity is the enemy*.
   - The member suggested that **elegance** and **usability** are better metrics for libraries and APIs, highlighting the importance of components fitting together seamlessly.
- **Tinygrad Plots MI350 Kernel Benchmarks**: Tinygrad intends to release **MI350 kernel benchmarks** by year-end, striving for single kernel performance at **80%** of NVIDIA's.
   - The focus is on overall efficiency rather than being the absolute fastest, optimizing tinygrad for faster job completion.
- **MLIR-Based Compiler in the Works**: A member is embarking on developing an **MLIR-based compiler**, incorporating previous considerations into the design.
   - It was mentioned that this strategy might not resolve the majority of significant challenges.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Clamor For Higher Knowledge Limits**: Members inquired about the possibility of exceeding the knowledge limit of **20** on the platform.
   - The channel did not include any successful methods to do so.
- **Credits & Tokens Seek Rollover**: A member asked if unused **tokens** and **credits** are rolled over upon renewal.
   - The channel did not include an answer to this question.
- **AI Goes Loopy, Users Demand Refund**: A member reported that the **AI** entered a very long loop and consumed all **3000 credits**, requesting a refund.
   - Another member chimed in that the same thing happened to them since **September 4th**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Golang Streaming HTTP MCP Server Debuts**: A member introduced their open-sourced [golang streaming http MCP server](https://github.com/ggoodman/mcp-server-go) project, tailored for demanding enterprise scenarios.
   - The project boasts features like **scalability**, **auth**, **sessions**, **resumability**, and **dynamic capabilities**.
- **Scaling Solution Unveiled for MCP Server**: The newly launched [MCP server](https://github.com/ggoodman/mcp-server-go) emphasizes **scalability** through a pluggable backend interface, incorporating in-memory and Redis streams backends.
   - This architecture is designed to facilitate scale-out capabilities within complex enterprise environments.
- **MCP Server Bolsters Auth with OIDC and JWT**: The [MCP server](https://github.com/ggoodman/mcp-server-go) can initiate the current authz spec from a simple issuer URL, leveraging **OIDC discovery** and **JWT access tokens**.
   - It supports manual configuration for customized setups, enabling versatile authentication approaches.
- **MCP Gains Sessions and Resumability**: **Sessions** and **resumability** are built on top of the pluggable backend, facilitating easier access to these challenging protocol aspects.
   - These enhancements simplify session management and resumability implementations in MCP servers.
- **Dynamic Capabilities Streamline Resource Handling**: The [MCP server](https://github.com/ggoodman/mcp-server-go) features a dynamic setup for managing tools, resources, and prompts from databases or APIs.
   - Complementing this, it offers containers for static setups, accommodating diverse deployment requirements.



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





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1417542177914880050)** (1 messages): 

> `Perplexity Pro Connectors, Email integration, Calendar integration, Notion integration, Github integration` 


- **Perplexity Pro Connectors Enable Integrations**: **Perplexity Pro** users can now connect their **email**, **calendar**, **Notion**, and **Github** to Perplexity through the [account connectors page](https://www.perplexity.ai/account/connectors).
   - **Enterprise Pro** users can also connect **Linear** and **Outlook**.
- **Unlock Productivity with New Integrations**: **Perplexity Pro** introduces connectors, allowing users to integrate their **email**, **calendar**, **Notion**, and **GitHub** accounts.
   - This enhancement streamlines workflows and provides seamless access to information across multiple platforms, with **Linear** and **Outlook** support for **Enterprise Pro** users.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1417223547755040828)** (896 messages🔥🔥🔥): 

> `Vape Server, Multi-Model AI Orchestration, Perplexity Finance on iOS, Comet Browser for Android, Jobs and Auto Apply` 


- **AI Superpowers Transform Disposable Vapes into H100 Hosting**: A member joked about using AI to convert disposable vapes into website hosting servers, referencing a [blog post](https://bogdanthegeek.github.io/blog/projects/vapeserver/) that humorously suggests needing an **NVIDIA H100** to fit.
   - This comment played on the absurdity of applying advanced AI capabilities to mundane objects.
- **Multi-Model AI Orchestration Creates Tricky Platform Variations**: A member discussed the challenges of multi-model AI orchestration, noting that [the same model can perform differently on different platforms](https://tenor.com/view/huh-cat-huh-m4rtin-huh-huh-meme-what-cat-gif-5834484041415217257).
   - Another member emphasized that the *orchestration layer* is where the real magic happens in making everything work reliably together.
- **Perplexity Finance Dashboard Access Remains Elusive on iOS App**: A user inquired about accessing the dashboard for **Perplexity Finance** on the iOS app, with another suggesting to search *finance* in the prompt text box.
   - Afterwards, a member shared a [gif with cats](https://tenor.com/view/cat-aquarium-fish-attack-fish-slap-fish-cat-cat-fish-gif-17675294) to follow up.
- **Android version of Comet Browser still not released**: A user inquired about the availability of an **Android version of the Comet browser**.
   - Another user responded that it is *Not out yet, you can pre register though*.
- **GPT-5 and Claude Compete in AI Landscape**: Members debated the merits of **GPT-5** and **Claude** for various applications, with one mentioning GPT-5 thinking beat Claude 4.1 on Perplexity.
   - It was pointed out that [GPT-5] had a perfect score, contrasting the disappointment with Claude's API performance in the same test.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1417566027247652956)** (3 messages): 

> `Shareable Threads` 


- **Perplexity AI promotes Shareable Threads**: Perplexity AI requested that users ensure their threads are set to **Shareable**.
   - A [link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) was provided with a screenshot and further [links](https://perplexity.ai/browser/claim/0FTN3KV9UX) to claim access.
- **Shareable threads are good**: Shareable threads may increase information sharing.
   - Shareable threads increase information sharing and collaboration.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1417503498873999462)** (2 messages): 

> `API vs Web UI Citation Discrepancies, Sonar-Pro Web Search Accuracy` 


- **API Citations Differ Substantially From Web UI: Member Seeks Insights**: A member reported significant differences between **API** and **Web UI** citations, with attempts to tweak filters (*context size, location, etc.*) widening the gap.
   - The member's attempts to achieve a **Jaccard similarity >0.7** between **Web UI** and **API** citations yielded a maximum of **~0.33**.
- **Sonar-Pro Web Search Accuracy Woes Aired**: A member expressed frustration with **web-search accuracy** using **sonar-pro**, noting that the **Web UI** successfully provided a background summary using a full name in a 1-shot setting.
   - In contrast, the **API** completely missed, with citations originating from old data or aggregator websites.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1417241063462011010)** (368 messages🔥🔥): 

> `FinePDFs Dataset, RAG accuracy, Random Token Masking, Clanker Detector LLM, AI Research Startup` 


- **FinePDFs Dataset: A Treasure Trove of PDFs**: The [FinePDFs dataset](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) is a large corpus sourced exclusively from PDFs, containing about **3 trillion tokens** across **475 million documents** in **1733 languages**.
   - It performs nearly on par with the state-of-the-art **SmolLM3-Web** dataset and achieves a remarkable performance improvement when merged with it (while keeping the proportion of PDF data below **25%**).
- **RAG Accuracy Enhancement Techniques Explored**: Members discussed improving **RAG systems' accuracy**, suggesting trying **BM25** instead of re-ranking, using **CRAG (contextual RAG)**, or employing **graph RAG** depending on the use case.
   - One member shared [a resource](https://github.com/NirDiamant/RAG_Techniques) for different approaches, while others debated the merits of **BM25** versus transformer-based re-rankers.
- **Masking for LLMs**: The application of Random Token Masking onto Causal models like LLMs and whether it would improve accuracy metrics after fine tuning was questioned.
   - It was pointed out that [unlike seq2seq models](https://openreview.net/forum?id=73FyDmYsdn#:~:text=This%20paper%20introduces%20Mask%2DEnhanced,retrieval%20and%20long%2Dcontext%20modeling), Masking out tokens in the loss would not help anything for LLMs since the next token will look back at it and LLMs predict sequentially.
- **Anti-Clanker**: Some members discussed the creation of a "**Clanker Detector LLM**" which is a model meant to detect **AI Slop**.
   - They linked to [an organization on Hugging Face](https://huggingface.co/anti-clanker) dedicated to this purpose, describing it as glorified deepfake detection.
- **AI Research Startup Seeking Visionaries**: A member sought collaborators for an AI project, emphasizing a vision-driven approach rather than offering corporate jobs.
   - They have [a tuned version of qwen 0.6b on HF](https://huggingface.co/models) which achieves >**21%** on humaneval pass@1


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1417257030871416903)** (2 messages): 

> `Transformers architecture, Agent Course Access` 


- **Studying Transformer Decoder Architecture**: A member announced intent to study the **Transformer architecture**, specifically focusing on **decoders**.
   - The member planned to start after waking up, but provided no links or further details.
- **Agent Course Sign-Up Snags**: A new member reported issues accessing the agent course after signing up, noting it wasn't appearing alongside **MCP** and **smol** in their course list.
   - They requested assistance in figuring out what they might be missing to gain access.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1417513545356021771)** (3 messages): 

> `Android OS control model, Swiftide 0.31, Reddit Content Bot` 


- **Android Controlled by Vision!**: A new **computer vision model** that controls **Android OS** has been released, showcasing a new approach to on-device control.
   - Details can be found in this [Hugging Face Collection](https://huggingface.co/collections/exteai/android-operators-68b9a03b65fac382855aff27).
- **Swiftide Gets Swifter with 0.31!**: **Swiftide 0.31**, a **Rust library** for building **LLM applications**, was released with a host of new features including **graph-like workflows with tasks** and **Langfuse integration**.
   - The release includes [multi-modal pipeline groundwork](https://blog.bosun.ai/swiftide-0-31/) and more - full details on the [project's Github](https://github.com/bosun-ai/swiftide).
- **Reddit Stories Automated!**: An app automating the creation of **Reddit story videos** using **Claude** and **ElevenLabs** has been developed, allowing users to create content similar to that found on YouTube and Instagram.
   - The project is available on [GitHub](https://github.com/rohanprichard/reddit-content-bot) and is documented in [this Medium article](https://medium.com/@rohanprichard/building-a-reddit-content-bot-automating-generating-videos-for-social-media-718d2089de06).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1417381975995715625)** (1 messages): 

> `Code Visualization, AI-assisted Blog Writing, Dynamic Graph Neural Networks` 


- **AI Assists in Code Visualization Blog**: A member has published a new blog post on visualizing code, noting it was co-written with **ChatGPT** to improve readability.
   - The member joked that the AI assistance made the blog *better readable*.
- **Dynamic Graph Neural Networks Blog Conceptualized**: A member shared a [Medium blog post](https://medium.com/@ravi92sr/dynamic-graph-neural-networks-for-visual-and-intent-driven-programming-35d199f8710e) on **Dynamic Graph Neural Networks** for visual and intent-driven programming.
   - The member indicated they had *just conceptualized* the idea.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1417420122800525392)** (16 messages🔥): 

> `smol fine tuning course, lighteval on Colab T4, integrating the translations from v1, older versions of vllm and triton` 


- **Dive into Translations for Smol Course**: The translations from **v1** into the hub version of the course were integrated, making the course available in languages other than English.
   - The maintainers said they are *very much open to contributions* on this.
- **Smol Fine-Tuning Course Sparks Model Transferability Debate**: A participant in the **smol fine tuning course** questioned whether its learning objectives are transferable to fine-tuning other models, such as a time series model.
   - Another user suggested reviewing library imports and the definition of the **DataCollatorForCompletionOnlyLM** component to resolve any issues.
- **Lighteval on Colab T4 Plagued by OOM Errors**: A user encountered issues running **lighteval** on **Colab T4**, experiencing **OOM** errors and weird bugs, but resolved them by using older versions of **vllm** and **triton**.
   - They shared a [standalone notebook](https://colab.research.google.com/drive/1Sntdimj1WFzLI26QpiR1ykD3ZsQpOOrF#scrollTo=Emybz1V2UcWm) for evaluations.
- **Lighteval Accelerate is slower than VLLM**: A user found that using **lighteval accelerate** was 2-3 times slower than **lighteval vllm**.
   - They recommended sticking with **vllm** for faster evaluations if another eval needs to be run.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

kong9646: hello.... working through unit one here....:)
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1417223719612452906)** (373 messages🔥🔥): 

> `LM Arena Web/Apps Issues, Image Size on LM Arena, Monetization Concerns for LMArena, Side-by-Side Image Editing, Gemma Vault Speculation` 


- **LM Arena Plagued by 'Failed to Create Sandbox' Error**: Users reported a "Failed to create sandbox" error when using the Web and Apps LM Arena, with the issue persisting for several hours.
   - Some users are also worrying about how the LM Arena owners are going to monetize the tool, and hope they don’t burn their own money.
- **Image Dimensions and Editing on LM Arena: Blank Images and Missing Side-by-Side Editing**: Users are discussing how to use images that are not 1:1 format on LM Arena, suggesting the use of blank images of the same size and prompting with specific resolutions; this technique reportedly works with models like **Qwen image edit**, **Flux Konnect**, and **Nano Banana** but not **Seedream** or **GPT Image**.
   - The option to upload and edit pictures in side-by-side mode won't appear unless both of the selected models have image edit capabilities.
- **Gemini 3 Speculation: Is OceanStone the One?**: Members are wondering if **OceanStone** model is **Gemini 3.0** or some version of **Gemini**.
   - There is a discussion on how models might act like they know something when they don't, suggesting that the weights haven't been improved, and the model was just trained to act like another.
- **Seedream-4-high-res: Love It or Lose It?**: Some users experienced issues of generating same face of an Indian woman's output always has the same face and they cannot access to **Seedream-4-high-res**.
   - Other users suggested to try from a new account and to ensure that image generation is enabled and troubleshooting browser-specific issues, like switching to another browser.
- **GPT-4o Overtakes GPT-5? Users Claim Better Performance**: Some users are experiencing **GPT-4o** giving the right answer while being concise and not responding in 6 bulletins every time.
   - Other members say **GPT 3.5** is much better than 4o while other members dismiss the claim as ragebait.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1417239560911323307)** (4 messages): 

> `Battle, Side by side, Direct - Why?, August Contest Update, Text-to-Image & Image Edit Leaderboards Updated, AI Eval Product Update` 


- **Arena Seeks Secrets of Side-by-Side Showdowns**: LMArena is seeking feedback via [survey](https://docs.google.com/forms/d/e/1FAIpQLSe5FincuXU2-TTp2WPymAD417Oev4MYdFKdWZhRZ9qSbV9N_w/viewform?usp=sharing&ouid=116478743206650022989) to better understand **why** users prefer certain versions in **Battle, Side by side, Direct** comparisons.
   - They would *love to understand better why you use your preferred version*.
- **August's Arena Video Victorious Visions**: The first **Video Arena GenAI contest** is complete and voting is open to crown a new <@&1378032433873555578> based on the theme 🔪**Slice**🔪, according to [this voting form](https://docs.google.com/forms/d/e/1FAIpQLSceYA4l7ew63w8DTcx2FwBYPY-uaOIM0UeUaUaLJM-9XQsmyw/viewform?usp=dialog).
   - The theme of the contest was *Show us those oddly satisfying, crisp cross‑section cuts into everyday objects*.
- **Text-to-Image Titans Trade Top Tier Titles**: `Seedream-4-high-res` has tied with `Gemini-2.5-flash-image-preview (nano-banana)` for the **#1 slot** on the Text-to-Image leaderboard, while `Seedream-4-high-res` now holds the **#2 position** in Image Edit, viewable on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image) and [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit).
- **AI Arena Assesses Algorithmic Acumen**: LMArena is introducing an **AI Evaluation product** to analyze human-AI interactions at scale, offering enterprises, model labs, and developers in-depth evaluations grounded in real-world human feedback as described in [this blog post](https://news.lmarena.ai/ai-evaluations/).


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1417275575986557042)** (1 messages): 

> `grok-2 deprecation, grok-3 release, grok-4 release` 


- **Grok-2 Gets the Boot!**: The models [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) and [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) are being deprecated by **xAI** today.
   - Users should migrate to [grok-3](https://openrouter.ai/x-ai/grok-3), or [grok-4](https://openrouter.ai/x-ai/grok-4) if vision is needed.
- **Grok models, a New Hope**: As Grok-2 is being deprecated, **OpenRouter** suggests moving to newer **Grok-3** or **Grok-4** models.
   - If your application needs vision support, then **Grok-4** is the better choice.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1417223446923837631)** (287 messages🔥🔥): 

> `Co-op Gooning with AI, NSFW bot development, OpenRouter Presets for Pre-Prompts, Gemma-3-27B Model API, AI Sex Dolls Implications` 


- **Co-op Gooning dream team: GC bots ahoy?**: Members discussed the possibility of **cooperative gooning** with AI, envisioning a group chat setup involving multiple bots for shared experiences.
   - One user suggested a feature to create messages collaboratively, enhancing the shared experience, while another jested about the emergence of *competitive gooning*.
- **NSFW Bot Gold Rush: Vibecoders Strike it Rich!**: The conversation took an NSFW turn as users explored the creation of **AI sex bots**, with one member claiming to have already developed a functional prototype connected to a fleshlight.
   - They described the setup involving an API that controls the *toys* movements based on the AI's text output, though details and proof were kept under wraps due to the channel's content guidelines, and linked to [Buttplug.io](https://github.com/buttplugio/awesome-buttplug).
- **Preset pre-prompts persistently pester users**: A user inquired about setting a **pre-prompt** for all models in OpenRouter without having to manually introduce it every time.
   - Despite attempts to *apply to all*, the setting did not persist across different chats or models, leading to the suggestion of implementing default presets, echoing a feature request from other users.
- **Gemma-3-27B's H100 Hookup: Free API Flow!**: A team announced a **free, OpenAI-compatible endpoint** featuring the blazing-fast Gemma-3-27B model, served on H100s via their custom-optimized stack, offering lightning-fast completions and streaming support.
   - The community was invited to try it out and provide feedback, with the team expressing interest in supporting cool projects built with it, with example `curl` commands provided.
- **AI Sex Doll quandaries: Is pre-made the key?**: Discussions emerged about the **implications of AI sex dolls**, specifically regarding customizable models and potential misuse, such as customizing them as children or using models of real people.
   - The conversation explored whether pre-made models would alleviate these concerns, though some argued that even pre-made models could be modified or misused and were already possible with existing tech, highlighting ethical considerations around data collection [as illustrated in this tenor gif](https://tenor.com/view/because-implication-just-saying-listen-gif-24693508).


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1417326348728537241)** (2 messages): 

> `` 


- **No new models discussed.**: No new models were discussed in the provided messages.
- **Channel silent on new models.**: The 'new-models' channel provided no information or discussion related to new models.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1417243644238364788)** (21 messages🔥): 

> `Gemini 3 Pro vs 2.5 Flash, 2.5 Pro checkpoins changing, Google Expectations` 


- **Gemini 3 Pro draws comparisons to 2.5 Flash**: Members compared **Gemini 3 Pro** to **2.5 Flash**, mentioning its potential to solve tasks requiring decent reasoning, as illustrated by a [complex circuit analysis problem](https://cdn.discordapp.com/attachments/1392278974222307469/1417243643797835938/5D6E10AF-9EBC-4B31-AEE6-110A30B2BF5E.png?ex=68cb17ff&is=68c9c67f&hm=fe49ba563701e7cd1630bfe7ab388eeb6b3a9a66e60ab9a0d4f4bc5bb5bd3857).
   - One member said that *if Gemini 3 Pro is just 2.5 Pro without the not x; but y problem and less sycophancy, I would be happy* and another member said that *Google isn't even that behind, really, not much pressure still.*
- **2.  5 Pro's performance varies between checkpoints**: Members discussed varying performance levels across different checkpoints of **2.5 Pro**, with one claiming a noticeable decline after the initial weeks of **2.5 pro**.
   - They noted that performance had decreased on *every listed benchmark except for the coding ones*.
- **Google faces high expectations, even 3.0 Flash**: It was mentioned that allegedly, based on a leak, **3.0 flash** is planned to be better than **2.5 pro**.
   - One member declared that *I do always have high expectations for google at this point. They are killing it on everything*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1417234952206024846)** (267 messages🔥🔥): 

> `RoPE intuition, Limitations of ML, Jianlin Su's blog posts, LLMs help avoid work, Hardware for LLM Experiments` 


- **Researchers Ponder RoPE's Inner Workings**: Researchers discussed the intuition behind **RoPE (Rotary Position Embedding)**, particularly the encoding of relative positions using increasing powers of two for wavelengths, and are looking for [in-depth explanations](https://kexue.fm/archives/8130) beyond standard documentation.
   - The challenge lies in understanding nuances like score negation, semantic overlap with rotations, and the selection of relative vs. absolute position, noting that research papers often lack intuitive explanations due to peer review constraints.
- **LLMs Can 'Bend' Laws of Math**: Members cautioned that Large Language Models (LLMs) may give the *illusion* of understanding concepts but can *bend the laws of maths to agree with you*, which is unhelpful for true comprehension.
   - It was suggested that LLMs are good at *helping you avoid work*, but not necessarily at helping you understand deeply, acting more as an efficient search tool across unfamiliar domains rather than a source of reliable knowledge.
- **Easy Experimentation With LLMs Hardware**: The user can do meaningful experiments on LLMs by renting it online since a **4090** is enough for a lot.
   - A member noted that the **5090** allows for mxfp8 accumulated into 32 bit buffers at full speed which is wild.
- **London and NYC Meetups Planned**: A London meetup is planned for **Saturday, September 27th** at [Broadway Market](https://broadwaymarket.co.uk/), with discussions about alternative venues due to potential rain.
   - Also, A member proposed an NYC meetup on the same day, with initial plans to meet in Central Park around 2 PM, contingent on weather conditions and interest.
- **World Labs focuses on 3D scenes to achieve AGI**: A user mentions that World Labs focuses on generating **3D scenes/worlds from images**, which is considered very nerfy.
   - They agreed that spatial intelligence and learning through multimodal interaction are key to AGI and provided [this blogpost](https://www.worldlabs.ai/blog) of the company.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1417297173812613214)** (23 messages🔥): 

> `LM Eval Discrepancies, Good CLM Training Examples, Hallucination Prediction, ARC AGI 2` 


- **LM Eval Results Mismatch Model Claims**: A user found discrepancies between **lm-eval** benchmark results and published data, like **Qwen3-0.6B** claiming **70%** accuracy on **gsm8k** but scoring under **20%** in their tests.
   - Another member noted that model providers often tweak eval setups to elicit more desirable scores, but that announced evals are not reproducible due to **non-public libraries and under-detailed prompts**.
- **Researchers Seek Optimal CLM Training Codebases**: A researcher asked for well-written codebases for CLM training, specifying they use a server with **NVIDIA GPUs** and aim for **data parallelism** without model sharding.
   - A member recommended starting with **NanoGPT**, advising against **Megatron-DeepSpeed** or **NeMo** for their scale and purpose, comparing it to studying 747 schematics for a small motor project.
- **Hallucination Prediction Toolkit Emerges**: A researcher shared their latest research on predicting hallucinations, with the toolkit achieving **900 stars** in about **two weeks**, as shown in [this tweet](https://x.com/ahmedkar_/status/1796787594333732947).
- **ARC AGI 2 SOTA Efficient Evolutionary Strategy**: A blogpost was shared with the title [ARC AGI 2 SOTA Efficient Evolutionary](https://ctpang.substack.com/p/arc-agi-2-sota-efficient-evolutionary).


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1417502641855922286)** (8 messages🔥): 

> `CLM Training Frameworks, MosaicML Composer, Dataset inference comparison` 


- **Frameworks for CLM training interest member**: A member is exploring frameworks for **CLM training** and seeks recommendations for well-written repos or codebases, such as **Qwen** or **Nvidia training scripts**.
   - They are experimenting with [MosaicML Composer](https://github.com/mosaicml/composer/) and are interested in opinions on various frameworks.
- **Essential details for model training assistance emerge**: A member highlighted the need to specify the **model size** and **GPU resources** when seeking advice on model training.
   - Without these details, any suggestions are deemed useless.
- **Dataset inference comparison question posed**: A member wants to run inference on **two sentence sets** within the same run to check for differing generations.
   - The example given was: Sentence A: *What is the tallest mountain in the world?* Sentence B: *No mountain in the world is taller than?*


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1417229475019751514)** (2 messages): 

> `Codex CLI, GPT-5-Codex, agentic coding, IDE Extension, Github code reviews` 


- **Codex Team AMA on Reddit**: The Codex team will hold an AMA (Ask Me Anything) session on Reddit, Wednesday 11am PT, linked [here](<https://www.reddit.com/r/OpenAI/comments/1nhust6/ama_with_the_codex_team/>).
- **GPT-5-Codex Optimized for Agentic Coding**: A new version of **GPT-5**, named **GPT-5-Codex**, has been released, optimized for agentic coding in Codex, detailed in [this blog post](<https://openai.com/index/introducing-upgrades-to-codex/>).
- **Codex CLI now available**: **GPT-5-Codex** will be available in the **Codex CLI**, **IDE Extension**, web, mobile, and for code reviews in Github.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1417234215304429608)** (125 messages🔥🔥): 

> `Codex web git commits and linters, Bachelors degree for AI Masters, AI and data science job market saturation, LLMs for Burmese language, GPT-5-Codex release` 


- **GPT-5-Codex Runs Free for 7 Hours!**: OpenAI's [GPT-5-Codex](https://openai.com/index/introducing-upgrades-to-codex/) reportedly worked independently for **over 7 hours** on complex tasks, fixing test failures and delivering a successful implementation, raising concerns about the future of human oversight.
   - One member joked about whether there will be an *escape time for human control*.
- **Junie flies: Jetbrains Releases Rider IDE**: Jetbrains released **Junie for the Rider IDE**, their version of a **Codex agent**, currently priced at **$300 USD**.
   - The community is awaiting feedback from early adopters to determine if it's worth the investment.
- **Flash 3.0 to beat 2.5 Pro**: Rumors are spreading that **Flash 3.0** might outperform **2.5 Pro**, potentially offering pro-level intelligence at a more affordable price.
   - This generational uplift suggests excellent progress, though concerns remain about how long this advantage will last once **3.0 Pro** is released.
- **ChatGPT: Jokes on who?**: Controversy erupted over [ChatGPT's handling of jokes](https://www.tiktok.com/t/ZP8SU85wE/) about nationalities, with questions raised about potential bias when **ChatGPT allows jokes about nationalities except one**.
   - One member asked *Isn't that shady?*
- **Ideogram nails text generation**: Members noted that [Ideogram AI](https://ideogram.ai/) excels at generating text within images, whereas **ChatGPT** is considered good at image generation overall.
   - One member noted he was using *ideogram exclusively for text until the past few weeks.*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1417231081400303616)** (11 messages🔥): 

> `Swagger Schemas with Fastify, Custom GPT Stacking Bug, GPT-7 Release, GPT Weekly Limits` 


- **Swagger Schemas Enforce Endpoint Definitions**: A member uses **fastify/schemas/swagger** in js/ts to define and enforce endpoint schemas, generating an **/openapi.json** file.
   - The member noted the schema is particular and dislikes cycles, suggesting a similar **swagger schema** setup in Python.
- **Custom GPT Stacking Bug Strikes?**: A member reported losing the ability to stack a custom **GPT** from the sidebar into a thread using the **@** menu.
   - They had been able to do it previously and wondered if this feature was removed.
- **GPT-7: When Will It Arrive?**: A member inquired about the release date of **GPT-7**, specifically asking *when* it will come after **GPT-6**.
- **GPT Weekly Limits Drive Users Bananas**: A member complained about **GPT** weekly limits, stating they are *bunk* and expressing frustration at being locked out for *2 days* without a clear indication of their remaining limit.
   - They contrasted this with **Anthropic**, where they never hit a weekly limit, calling it ironic.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1417223977419407460)** (60 messages🔥🔥): 

> `Chatbox character limit, LLM limitations, Prompt engineering techniques, Generating human-like speech` 


- **Chatbox Character Limit is HUGE**: The character limit for the web chatbox is **136,846 characters**, as discovered by a member who tested it.
   - They confirmed that this character limit applies even with **21,027 words** across **1,516 lines** and that responses can still be valid after sending prompts of this length.
- **LLMs Need Guardrails**: One member expressed LLMs aren't intelligent enough to perform proper synthesis beyond **100 LOC** python script, suggesting that LLMs are time-saving measures rather than design/architecture drivers.
   - Another added that context building and clear objectives are crucial for good outcomes, noting that long chats often indicate poorly defined initial parameters, recommending a methodology of using separate chats for brainstorming and programming asks to achieve optimal results.
- **Discordianism Inspires AI Prompt Engineering**: A member shared techniques inspired by Discordianism (a joke religion taken seriously), for prompting AI to explore new paradigm paths, from random mutations to guided discord, using agents to reharmonize.
   - They run a "joke institution" called the **Institute of Discordant Colony Optimization**, inspired by *ant colony optimization*, attaching a [text file](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8) containing **five** of **twenty-five** such techniques that produce useful results.
- **Positive Framing Instructions**: A member shared a prompt for reframing negative instructions into positive ones, suggesting using a **Positive Frame Optimizer (PFO)** to transform negative instructions into semantically equivalent positive frames, preserving intent.
   - For example, transforming “Don’t use slang” into “Use formal language”.
- **Humanism in AI-Generated Speech**: A user working on a project that generates news summaries with public commentary and converts it to speech is seeking ways to inject **humanisms** like filler words, interjections, and natural pauses to make it sound less robotic.
   - One member suggested defining the role more richly, specifying that the output should mimic a high school student suddenly thrust into anchoring a national news show, complete with nervous ticks and interjections.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1417223977419407460)** (60 messages🔥🔥): 

> `Chatbox Character Limit, LLM Limitations, Discordianism and AI, Positive Framing Prompts, Generating Humanisms in AI Speech` 


- **Chatbox Character Limit Unveiled**: The character limit in the web chatbox is **136,846 characters**, with word and line counts being irrelevant, as discovered by a member through rigorous testing.
   - It was further clarified that while a large character count is possible, the system might prefer fewer characters when the word count is also high, possibly due to pre-qualifying token size.
- **Taming LLM Limitations with Brainstorming**: A member expressed that LLMs might not be intelligent enough for proper synthesis beyond 100 LOC python scripts, suggesting they're best used as time-savers with the user providing high-level design.
   - Another user finds LLMs helpful for brainstorming, especially when pushing the model to produce math, diagrams, or design documents, before switching to code generation in a new chat to avoid the chat UI becoming unusable.
- **Discordianism Inspires AI Swarm Optimization**: A member shared techniques from the **Institute of Discordant Colony Optimization**, inspired by Discordianism, to get AI to veer off into new paths using methods like random mutations and guided discord.
   - The goal is to create an AI scientist swarm with both normal and *discord ants*, injecting discord to find more harmonious solutions, akin to ant colony optimization, detailed in an [attached text file](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8).
- **Positive Framing for Robust AI Instructions**: A member shared a prompt template for transforming negative instructions into semantically equivalent positive frames to ensure clarity and stability in mid-context.
   - The **Positive Frame Optimizer (PFO)** process involves detecting negation-based instructions and reframing them as explicit positive directives, providing a structured approach to prompt engineering.
- **Coaxing Humanisms into AI Speech Generation**: A member is working on a project to generate news summaries with public commentary and humorous hot-takes, voiced using OpenAI-TTS, aiming to coax more *humanisms* out of the text.
   - Suggestions included defining the role as a nervous high school student anchoring a national news show and being sloppy in the prompt to encourage the model to mimic the tone.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1417225142567960759)** (219 messages🔥🔥): 

> `Model Switching in Queued Messages, Auto Model Selection, Cursor's token usage and cost, Codex vs Claude Code, Cursor's Rules` 


- **Queued messages not saving desired LLM**: A member inquired whether queued messages save the selected model at the time of queuing or if they are sent with the currently selected model, suggesting working in [separate chats using `CTRL + T`](https://community.cursor.sh/); however, another member said they were hoping to use this method to have different models critique each other iteratively.
   - A member warned that *changing models within the same chat will end up resetting context*.
- **Cursor Auto is still a mystery**: Members discussed how to use the best models in web development, and whether Cursor's 'Auto' mode is free and which models it uses, with others suggesting that the cost of input is the same as **GPT-5** for those without a yearly plan.
   - One member noted they've been getting better results with **Cursor GPT-5 High** and **Cursor** compared to Codex, while another speculated Auto mode could use **o3 reasoning**.
- **Token Usage Under Scrutiny**: Members reported high token usage, especially with **Cache Read**, even with limited prompting and raised concerns about rapid exhaustion of token quotas, asking if they could [disable or limit Cache Read](https://community.cursor.sh/) to prevent background loading from consuming quota.
   - One member noted that Cursor was being *too thieving* as they used up the monthly tokens in basically a day.
- **Clash of the Code Gen Titans**: Users are hotly debating whether **Codex** or **Claude Code** performs better, with most reporting that [Claude Code still reigns supreme](https://community.cursor.sh/) in speed and effectiveness, even for complex tasks such as *editing some fields in a modal and adding a context menu*.
   - Some members complained that **Codex** *deletes half my code and can't be reverted* and gets stuck in loops, whereas **Claude Code** *makes the mistakes quickly*.
- **Cursor's Rules Debated for Consistency**: Members discussed using the `.cursor\rules` file for [enforcing consistent coding practices](https://community.cursor.sh/), with some questioning its reliability and others confirming its functionality, noting that rules might not work as expected in certain scenarios.
   - A member shared a link to a [GitHub repo of community-sourced cursor rules](https://github.com/sanjeed5/awesome-cursor-rules-mdc/blob/main/rules-mdc/react-native.mdc), with things like *security pieces* in them to check, while another suggested YAML for better codification.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1417566363739881664)** (3 messages): 

> `Custom Branch and PR Naming, Linear Integration Challenges, Multi-Repo Issues, Sub-Issue Limitations, Agent Detachment Workaround` 


- **Background Agent Naming Conventions Requested**: A user inquired about customizing the branch and PR names created by the agent to include **Jira ticket IDs**, citing it as a requirement for adoption.
   - Current naming conventions are blocking the agent's usage because they cannot be modified to adhere to the organization's standards.
- **Linear Integration Faces Multi-Repo Issue**: A user is facing challenges with the new Background Agents for **Linear integration** because many issues require work across multiple repos, but Linear only supports tagging an issue with a single repo.
   - The user pointed out that the agent is unable to read parent issue or sub-issue descriptions, which contain very relevant context.
- **Sub-Issues Fall Short for Context Sharing**: The user created sub-issues to address the multi-repo problem, but the **Background Agent for Linear** cannot read the parent issue or sub-issue descriptions, limiting the agent's context.
   - Copying the parent ticket description into sub-tickets presents a consistency challenge.
- **Agent Detachment Enables Sequential Workflow**: The user is working around the limitations by working in a single ticket and using `@cursor` with detailed instructions for the first step/repo.
   - They then unassign the agent and restart for the next step, as detaching the agent seems to allow for a new workspace to be started with `@cursor`.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1417464809565196419)** (11 messages🔥): 

> `GB200/GB300 Availability on Coreweave, PruneAI Talk, LBO/SBO Calculation for Shared Memory Matrix` 


- **GB300 Speculation Simmers**: Members are wondering when the **GB300** will be generally available on **Coreweave** (or another cloud provider).
   - Someone speculates on how long it took **GB200** to become available might inform estimates.
- **PruneAI Talk Slides Shared!**: A member requested the slides from a talk given at **PruneAI**, mentioning a screenshot of **tcgen05.mma** content.
   - Another member shared the [Google Slides presentation](https://docs.google.com/presentation/d/1KLz3NisvrmTLuIPVb4yiP0z5WWlh9gTMm-Ms-kCc6fQ/edit?usp=sharing).
- **LBO/SBO Shared Memory Matrix Layout Examined**: A member sought clarification on calculating **LBO/SBO** for shared memory matrix descriptions for **wgmma**, finding the [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) confusing.
   - Another member provided a breakdown, explaining that *SBO corresponds to the stride between one color and the next time that color occurs along that row -> 32* while *LBO corresponds to going from 0-1-2-3 to 64-65-66-67 etc -> 64*.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1417454144683511901)** (2 messages): 

> `Triton Block Size Calibration, Nvidia GPU Atomics Overhead` 


- **Triton Block Size needs Calibration per-GPU**: The default **Triton** block size of *(256, 256)* in `opts_flags_nvidia` causes an `OutOfResource` error on **RTX 5090**, requiring a reduction to *(128, 128)*.
   - A member suggested using `max_num_imprecise_acc` to cope with this without modifying the code, as the current logic may be calibrated for specific GPUs.
- **Nvidia GPU Atomics Overhead Assessed**: A member inquired about the overhead of **Triton atomics** on **Nvidia GPUs** (Ampere and up), seeking to understand the performance impact.
   - They recalled high contention overhead on **AMD GPUs** (hundreds to thousands of cycles) and asked if similar issues exist on **Nvidia**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1417542468039213198)** (10 messages🔥): 

> `P2P Memory Access, Symmetric Memory, wgmma on sm120 (consumer blackwell), mbarriers in threadblock clusters` 


- **Peer-to-Peer Memory Pointers?**: A member inquired about resources for using **P2P memory access** and mentioned the [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-access).
   - Another member suggested that it has a lot of overlap with **symmetric memory** and linked [symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes) as a go-to repo for examples (though noted it's in Triton and Torch).
- **CUDA Samples Point the Way**: A member pointed to [CUDA samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleP2P) and [streamOrderedAllocationP2P](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationP2P) for **P2P memory access** examples.
- **Blackwell Bumps Warp Group Instructions**: A member reported errors related to **wgmma** instructions on **sm120 (consumer Blackwell)**, indicating that *instruction 'wgmma.fence' not supported on .target 'sm_120'*.
   - Another member confirmed that warp group instructions were removed from Blackwell.
- **Mbarriers Meet Threadblock Clusters**: A member questioned the use of **mbarriers** in the context of **threadblock clusters**, specifically whether synchronization across a cluster is possible using mbarriers.
   - Referencing the **PTX docs**, they noted that *mbarrier.arrive operation on an mbarrier object located in .shared::cluster but not in .shared::cta cannot return a value*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1417489210247614586)** (12 messages🔥): 

> `torch.compile schema, mutation annotations, return tuples, float vs double, tensor types` 


- **Mutation Annotations Matter for Torch Compile**: When registering ops with `torch.compile`, explicitly writing the schema requires ensuring that [mutation annotations](https://pytorch.org/docs/stable/torch.compiler_faq.html) are correctly specified, or PyTorch may assume no inputs are mutated.
   - One user suggested this gotcha should be in the docs.
- **Explicit Schemas Required for Returning Tuples**: A user noted needing to explicitly write the schema to [return tuples](https://pytorch.org/docs/stable/generated/torch.return_types.html#torch.return_types.namedtuplelist).
   - They suggested other gotchas include needing to use `float` instead of `double` in the schema, and that the schema allows weird things like `tensor[][][][]` that are really not supported by PyTorch.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1417581648203087912)** (6 messages): 

> `H100 Performance, TFLOPS variance, Matrix Multiplication, Architectural Rpeak` 


- ****H100**'s Missing **TFLOPS**: Getting the full declared performance**: A member is trying to achieve the declared **989TFLOPS** for **Nvidia H100 SXM** but is only getting **760TFLOPS** using *torch matmul* and *triton* for benchmarking, even when testing across different providers.
   - When running the same script on a **4090**, the member achieves the full declared **165TFLOPS**, leading them to question why the **H100** isn't reaching its expected performance.
- ****Matrix Multiplication** Caveats: Low Precision and Throttling**: A member pointed to [this article](https://www.thonking.ai/p/strangely-matrix-multiplications) suggesting that low precision tensor cores on random input data can cause GPU throttling, thus affecting performance.
   - The original poster questions whether a **77%** performance loss is expected and why it doesn't occur with **4090s**.
- ****Architectural Rpeak** Reality Check: Power, Cooling, and Constraints**: A member clarified that the declared **989 TFLOP/s** is the architectural Rpeak, and real-world systems may not reach this due to power draw and cooling limitations.
   - They referenced [a post by Dan Ernst](https://x.com/ernstdj/status/1531481863436509184), an HPC architect at NVIDIA, that provides an overview of what GPU system Rpeak performance means.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1417265956085956721)** (3 messages): 

> `autoquant_v2, batch size 1` 


- **Autoquant V2: Batch Size 1 Blues**: A user asked if **autoquant_v2** is suitable for **batch size 1**, mentioning specialized code for that scenario.
   - They experienced **runtime errors** during autotuning with **batch size 1** for certain dtypes.
- **Batch Size 1: Autoquant's Kryptonite?**: Concerns were raised regarding the use of **autoquant_v2** with a **batch size of 1**.
   - Specifically, the user highlighted potential issues during the **autotune stage** due to runtime errors encountered with certain dtypes when using a **batch size of 1**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1417420137572601947)** (15 messages🔥): 

> `CUDA debugging, Darksynthwave, PrimeIntellect, BackendBench, Performant CUDA kernels` 


- **Algorithm Purple T-Shirt: Progressive Darksynthwave Post Avantgarde Neoglitch IDM Metal**: A member shared a link to [The Algorithm Purple T-Shirt](https://fixtstore.com/products/the-algorithm-purple-t-shirt) which is tagged as *Progressive Darksynthwave Post Avantgarde Neoglitch IDM Metal*.
- **PrimeIntellect Bounties out for BackendBench**: A member noted that PrimeIntellect environments have bounties out, adding [BackendBench](https://github.com/meta-pytorch/BackendBench) as one of them for **800$**.
   - They shared a link to [the sheet](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit?gid=0#gid=0) and [their implementation](https://app.primeintellect.ai/dashboard/environments/siro/backend-bench).
- **CUDA Kernel Writers: An Endangered Species?**: A member shared a [post](https://x.com/kalomaze/status/1967869726455214432) claiming *there are probably less than ~100 people living who can write performant CUDA kernels for training specifically*.
   - Others expressed skepticism, with one saying *i get what they're trying to say but it's not really true or helpful*.
- **Debugging CUDA Memory Leaks Video**: A member shared a [video](https://www.youtube.com/watch?v=gzuK4AXAbcc) about debugging **CUDA memory leaks**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1417299270586990613)** (12 messages🔥): 

> `Iris Memory Management, ROCm 7.0, tl.load vs iris.load, Kernel timeout errors` 


- **Iris Needs Memory Deallocation TLC**: It was noted that **Iris** currently doesn't deallocate its memory, requiring users to allocate once and reuse across iterations.
   - A developer confirmed this and suggested hoisting the iris instance and the C allocation to the module level to mitigate timeout errors and expressed it was on their *todos to fix that*.
- **ROCm 7.0 Arrives Just in Time**: Members shared a [phoronix.com](https://www.phoronix.com/news/AMD-ROCm-7.0-Released) link about **AMD ROCm 7.0** being released.
   - Another member shared the [official ROCm release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html).
- **`tl.load/store` beats `iris.load/store` for local**: Using `iris.load/store()` instead of `tl.load/store()` for local memory access introduces translation overhead, though minimal and cached.
   - The recommendation is to use `tl.*` operations for local accesses until a fast path is implemented in Iris, skipping the `if` statement check within loops.
- **Troubleshoot Kernel Timeouts with Iris**: One member reported getting timeout errors when using **Iris**, even with simple kernels, potentially due to repeated tensor allocation.
   - They planned to DM a developer for assistance, and members were encouraged to DM with questions they didn't want to ask in the channel.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1417259563987763280)** (4 messages): 

> `IPEX Deprecation, PyTorch Upstreaming, Intel Optimization Strategy` 


- **Intel Extension for PyTorch (IPEX) Faces Deprecation**: A member reported that **Intel Extension for PyTorch (IPEX)** is being deprecated in favor of upstreaming features into **PyTorch**.
   - Another member cited [Intel's release notes](https://pytorch.org/blog/intel-extension-for-pytorch/) stating they are discontinuing active development on **IPEX** after the **2.8 release** to focus on developing new features directly within **PyTorch**.
- **IPEX: Intel's Experimental Optimization Platform**: Before the above announcement, **IPEX** had been an experimentation platform for **Intel** to push aggressive and new optimizations, like an experimental version of **torch nightlies**.
   - One member admitted that their knowledge was outdated after the announcement.
- **Intel Shifts Strategy Towards Upstreaming to PyTorch**: Intel launched **IPEX** in **2020** to extend official **PyTorch** and simplify high performance on **Intel CPU and GPU platforms**.
   - However, **Intel** has successfully upstreamed most of their features and optimizations for **Intel platforms** into **PyTorch** and will focus on new features and supporting upcoming platform launches directly within **PyTorch**.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1417237213871083734)** (1 messages): 

> `Metal command buffer timeout` 


- **Metal Command Timeout Sought**: A member is seeking advice on how to set a **timeout** on a Metal command buffer, because some kernels take too long to execute.
   - The member has a run function that executes a graph of **Metal kernels** and wants to implement a timeout mechanism.
- **Metal Kernel Graph Timeout**: The user's run function executes a graph of Metal kernels.
   - Sometimes it takes too long to execute so the user wants to implement a timeout mechanism.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1417498604397920317)** (2 messages): 

> `Attention Variants, MLA Explained, Quantization Survey` 


- **DeepSeek MLA Attention Variant Revealed**: A member shared a [Notion page](https://charm-octagon-74d.notion.site/Attention-Variant-4-DeepSeek-MLA-270e4301cb99809594fedbcb073849b1) explaining **DeepSeek's MLA Attention Variant**.
   - It provides information on how it works under the hood for those who want to dive deep.
- **Quantization Survey Facilitated**: A member expressed gratitude for the attention variant explanation, noting that it was helpful for an upcoming, *lighter* survey of **quantization**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1417235036259876984)** (20 messages🔥): 

> `A100 performance, MI300x8 performance, Profiling errors, HIP/ASM perf` 


- **A100 Aces Trimul Leaderboard**: A member achieved a personal best of **20.0 ms** on an **A100**, followed by another successful run at **20.3 ms** on the `trimul` leaderboard.
- **MI300x8 Marks AMD-All2All Milestones**: Several submissions highlighted **MI300x8** performance improvements on the `amd-all2all` leaderboard, with times ranging from **1348 µs** to **2.93 ms**.
   - A user reached 6th place with **1348 µs**, and another achieved a personal best of **2.03 ms**.
- **Profiling Problems Plague Performance Probes**: A member encountered a `TypeError` when submitting a custom kernel for profiling, citing a missing `rank` argument in `generate_input()` when attempting to use the command `leaderboard submit profile script: gpu:MI300x8 leaderboard_name:amd-all2all`.
   - A moderator clarified that *profiling is not yet available for competitions*, though the infrastructure is in place.
- **HIP/ASM Helps Harvest Higher Perf**: A member inquired whether switching to **HIP** would yield better performance, particularly after watching a video.
   - Another member confirmed that **HIP/ASM** typically achieves better performance.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1417416747178266665)** (18 messages🔥): 

> `Lua changes, Frontier model sweeps, Claude sicko mode, Error in GetEntities, Stray log line fixed` 


- ****Lua Land Updates Land!****: A member confirmed recent changes to **Lua** scripts involving local functions, modules/tables, new admin tools, and the removal of unused code.
   - These changes do not include functional changes other than `can_place_entity`.
- ****Openrouter Keys Unlock Frontier Model Sweepstakes!****: A member ran a sweep overnight on frontier models, adding **four personal OpenRouter keys** along with another member's key.
   - The sweep consumed **$100** on their account, which was only a small fraction of the total sweep.
- ****Claude Smokes Competition in Lab Play!****: **Claude** demonstrated superior performance in open play, achieving double the performance compared to other models, and this could still improve.
   - The performance was evaluated after only **2-3 trials**, suggesting that Claude's pass rate at **@8** could be even higher.
- ****GetEntities Glitch Gags Game!****: An error in **GetEntities** was reported (*Could not get entities, Error when writing to file: No space left on device*) due to excessive logging.
   - The solution involved truncating the logs or implementing an option to disable logging.
- ****Stray Log Line Struck Down!****: A member identified and fixed a stray log line in serialize that was causing excessive logging, with the fix pushed directly to main.
   - This resolves issues related to disk space filling up due to uncontrolled logging.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1417261480784691280)** (19 messages🔥): 

> `A2A kernel rules, Dispatch and Combine Kernels, Intra-node communication, GEMM + RS Kernel Rules, Simulated MOE and Combine Kernels` 


- **Clarification on Simulated Computation in A2A Kernel**: The first problem in the competition focuses on **fast communication** rather than implementing grouped_gemm, emphasizing that the **MOE and combine kernels are simulated**.
   - As *there's actually no meaningful computation being done in the 1st problem then* it's designed to concentrate on optimizing **dispatch and combine kernels** without the complexity of grouped_gemm, which is why the subsequent problems include computation.
- **Full Set of Rules for the Competition Unveiled**: Organizers have stated that the full set of rules will be sorted out and pinned for easy access, referring participants to **public AMD docs** and **the pinned message** for immediate information.
   - The aim is to provide a clear and comprehensive guide, addressing concerns about overlooking important details scattered across different messages.
- **A2A Dispatch Kernel and Communication Optimizations**: Participants are allowed to optimize the **sort and count** operations within the dispatch kernel, but the kernel must still dispatch tokens, forbidding methods that only output counts without the sort intermediates.
   - The aim is to ensure that solutions adhere to the core dispatch logic while exploring opportunities to optimize across different phases of the operation.
- **Rules and Introduction for the A2A Kernel**: The rules mandate that implementations must include intra-node communication and the dispatch and combine kernels, emphasizing the importance of analyzing the logic of dispatch and combine in reference.py and understanding how the different shapes defined in task.yml go through the reference_kernel.
   - The aim is to encourage brainstorming ideas to accelerate dispatch combine and whole reference_kernel.
- **GEMM + RS Kernel: Computation and Communication Dynamics**: For the GEMM + RS kernel, solutions must communicate intra-node to fetch data for the ReduceScatter operation and can explore methods to optimize or fuse ReduceScatter and GEMM operations.
   - As *this is a computation+communication kernel* the kernel logic requires detailed analysis.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 messages): 

drazi1983: Welcome. Thanks for asking. And really nice diagrams!
  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1417244312273551491)** (27 messages🔥): 

> `picograd progress, sitp updates, jupyter notebook in rust mdbook, heterogenous programming, CUDA vs HIP` 


- ****SICP** for Heterogenous Programming is Born**: The **sitp project** aims to be *SICP* for the era of **heterogenous programming**, providing a tikz figure, explanation, pseudocode, and a rust implementation for each concept.
   - It has been updated to [https://github.com/j4orz/sitp](https://github.com/j4orz/sitp) and [https://j4orz.ai/sitp/](https://j4orz.ai/sitp/).
- **Embedded Jupyter Notebooks Face Yak Shaving**: A member is trying to embed a **jupyter notebook** inside a **rust mdbook** to display **torch code** and replace `import torch` with `import picotorch`.
   - They are planning to make the mdbook only execute **pytorch** example code on **CPU** and provide device implementation **HIP code** as static text due to complexity with **JupyterHub**, **k8s**, and **Slurm**.
- **Building a Pytorch from Scratch Competition?**: A member suggested a **community project** to "build a pytorch from scratch" with correctness tests and leaderboards to benefit the **MLSYS** community.
   - Another member pointed out that designing competitions is really time consuming, using **Modal** and **GitHub Actions** with runners, to create a `nano-gpt` like competition.
- **AMD and HIP Steal Show Over CUDA**: The **sitp project** uses **AMD** and **HIP** because **RDNA** is not a virtual assembly like **PTX**.
   - The project is looking for support in doing things with their tech, and could provide **modal credits** to do anything.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1417245345817301064)** (1 messages): 

> `BioML trimul kernel competition, GPUMODE swag` 


- **BioML Trimul Kernel Competition Deadline Nears!**: There are only **14 days left** to participate in the [BioML trimul kernel competition](https://www.gpumode.com/v2/leaderboard/496?tab=rankings) on GPUMODE.
   - The prize is "never before seen swag" designed and shipped by the competition organizer.
- **GPUMODE Swag Prize Announced**: The BioML Trimul Kernel Competition prize will be "never before seen swag".
   - The swag will be designed and shipped by the competition organizer themselves, adding a personal touch.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1417223621847416962)** (1 messages): 

> `Mobicham's LLM work, DiT, LLM Training, Quartet` 


- **Mobicham Advances LLMs**: The channel primarily features **Mobicham's** ongoing work and advancements in **large language models (LLMs)**.
   - Discussions include various experiments, methodologies, and results related to **LLM training** and optimization.
- **Diving into DiT**: Some channel participants showed interest in **Diffusion Image Transformer (DiT)**, focusing on implementation details and potential applications.
   - They're exploring how DiT models can enhance image generation tasks and their integration with other AI technologies.
- **Quartet Training Gains Traction**: Channel members also discussed the **Quartet** training method for LLMs, highlighting its benefits and challenges.
   - This included sharing insights on optimizing **Quartet** for specific tasks and datasets to improve model performance.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1417569864113066015)** (1 messages): 

> `Low-Bit-Training for Video Models, GitHub Project for Video Model Training` 


- **Low-Bit Training Video Model Project Kicks Off**: A member is initiating a [project on GitHub](https://github.com/username/video-model-repo) to explore **low-bit-training** for video models.
   - They are currently working on scoping out the problem and conducting a survey to refine the project's direction.
- **Community Invited to Contribute to Video Model Project**: The project aims to tackle the challenges of **low-bit training** specifically tailored for video models, inviting community participation.
   - The initiator seeks collaboration and feedback, particularly in defining the scope and objectives of the project, with a survey planned to gather insights.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1417248355947511990)** (97 messages🔥🔥): 

> `Mercor, SWEBench, Cursor's Bugbot, OpenCode Zen, Gamma 3.0` 


- **Mercor's GMV Questioned**: A member linked to a [tweet](https://x.com/BrendanFoody/status/1967635147274207376) discussing **Mercor's** numbers, with another pointing out they represent GMV for a staffing agency rather than typical SaaS ARR.
   - Despite this distinction, the growth was still acknowledged as *super impressive*.
- **SWEBench Critique**: A discussion was started regarding [SWEBench](https://x.com/brhydon/status/1953648884309536958/photo/1), with claims that it is narrow, over-hyped, and focuses on trivial **Django** fixes instead of true software engineering skills.
   - The argument posited that high scores often reflect repo memorization, while real SWE work involves diagnosis and scoping.
- **Cursor's Bugbot Hits $10M ARR**: **Cursor's Bugbot** celebrated its launch month with **$10M ARR** with a 2-person team.
   - A member expressed losing goodwill due to past pricing issues, though acknowledging the technical merit, particularly their new RL work.
- **OpenCode Zen Enters Coding LLM Arena**: **OpenCode Zen** was launched, offering coding LLMs with up-to-date models, provisioned capacity through **Vertex**, and **GPT-5** pass-through at **Stripe**-fee-only pricing.
   - It aims to be a substitute for **OpenRouter** with no data retention on paid plans, and no profit margin.
- **Gamma 3.0 Updates Presentation Game**: **Gamma 3.0** was announced, featuring the new **Gamma Agent** that allows users to edit entire slide decks with single prompts and a **Gamma API** that enables **Zapier** workflows to auto-generate decks from meeting transcripts.
   - The release includes new **Team**, **Business**, and **Ultra** plans, with the goal of making slide creation faster and more accessible.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1417410850431438888)** (9 messages🔥): 

> `Nano Banana prompt, Bytedance video model, HeyGen rebrand, Video Agent Public Beta, Alisa Acquisition` 


- ****Nano Banana** Prompt Turns Websites into **90s CD-ROM** Boxes**: [Levelsio](https://x.com/levelsio/status/1967593100676943892?s=46) shared a playful **Nano Banana prompt** that turns any website into a **1995-style CD-ROM** product box, sparking nostalgic mock-ups.
- ****Bytedance** Team Releases High-Quality Video Model**: The **Bytedance** team released a new model that generates *high-quality, text-aligned and subject-consistent videos*, now with [ComfyUI support](https://x.com/joshua_xu_/status/1967951859500437855).
- ****HeyGen** Rebrands and Launches **Video Agent Public Beta****: **HeyGen** co-founder [Joshua Xu](https://x.com/joshua_xu_/status/1967951859500437855) announced a rebrand positioning **HeyGen** as a *creative operating system* and launched the **Video Agent Public Beta** to turn prompts into publish-ready videos.
- ****HeyGen** Acquires **Alisa** to Lead **Video Agent** Product**: **HeyGen** acquired **Alisa**, an intelligent multimedia agent startup, with founder **Bin Liu** now leading the **Video Agent** product at **HeyGen**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1417233016128012451)** (72 messages🔥🔥): 

> `Abliterated Models & Censorship, LM Studio Version Confusion, Model Generation Speed, Qwen3-Next-80B on LM Studio, VRAM Rule of Thumb Clarification` 


- **Abliterated Models Still Say No!**: Members discussed how **abliterated models** have their weights removed to prevent negative responses, but they don't gain the ability to produce meaningful content outside their training data.
   - One member added that *the training data is still not cleaned much* so models will struggle with avoiding toxic responses.
- **LM Studio Version Numbers Confuse User**: A user was confused why LM Studio installed version **0.3.26** instead of **0.3.9**, thinking the latter was the newest.
   - Other users quickly clarified that 26 is in fact, *a bigger number than 9*, leading to a facepalm moment for the original poster.
- **Dolphin Model's EOS Token Generation Speeds Questioned**: A user inquired about the generation info of **mradermacher : Dolphin 2.7 Mixtral 8x7B GGUF Q4_K_M model** - specifically the **EOS Token Found** message.
   - Others explained that this is a normal output message and that the user's **4.02 tok/sec** generation speed *is fairly old by AI standards*, but might be ok depending on hardware.
- **Qwen3-Next-80B Gets MLX Support!**: Users shared a [Hugging Face link](https://huggingface.co/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit) to a **Qwen3-Next-80B-A3B-Instruct-MLX-4bit** model and clarified that GGUF is not supported due to qwen3next not being supported in llama.cpp.
   - Another member linked a [YouTube video](https://www.youtube.com/watch?v=ux9S_-QX7TE) showcasing the model, but cautioned that it's *incredibly slow* using transformers and recommended patience while the architecture matures.
- **VRAM Rule of Thumb Debated by Power Users**: A member stated that *the file size of the model must be less than the amount of vram you have available*.
   - Another member countered, stating that they can run `Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf` quite well (35GB) on 24GB VRAM, because it's a **Mixture of Experts** model where not all weights are active at the same time.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1417256745113489408)** (24 messages🔥): 

> `Personal Cloud with Nextcloud, VPN Meshnet for Cloud Gaming, Setting up Qdrant Vector Database, Ryzen AI MAX 395 performance with qwen3-coder-30b, MacOS Sequoia memory usage for 70B models` 


- ****Nextcloud Networking Newbie****: A beginner details their networking project which include setting up a **Nextcloud personal cloud** but paused due to ISP issues, as well as configuring a **VPN meshnet** for cloud gaming and AI use.
   - The user is now progressing to setting up a **Qdrant vector database** for AI, after saving a stable SBC configuration.
- ****Ryzen AI MAX 395 Performance Quest****: A user inquired about the performance of a **Ryzen AI MAX 395+** with **qwen3-coder-30b Q8 and bf16**, specifically asking if they should wait for next-gen hardware or build an AMD Epyc 9005 system.
   - Another user is awaiting their machine's return from warranty to provide performance numbers, while another shared a relevant [GitHub link](https://github.com/kyuz0/amd-strix-halo-toolboxes) related to **AMD Strix Halo toolboxes**.
- ****MacOS Sequoia's Memory Challenge****: A user running **MacOS Sequoia** with minimal memory to load 70B models in Q8 is hesitant to upgrade to Tahoe due to potential increased memory usage.
   - Another user advised to *only update if necessary* and to *wait for bug fixes* instead, referencing the early stage of the OS.
- ****CachyOS and LLM Offloading Face-Off****: A user is installing **CachyOS** and questioned why they should use a hypervisor when running LLMs, believing it would negatively impact **MoE offload performance**.
   - Another user countered that a hypervisor wouldn't necessarily impact performance and allows running other applications, noting the minimal overhead on larger systems.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1417302315785785414)** (83 messages🔥🔥): 

> `XML for agentic coding, MI50 GPUs, AMD RDNA5, Codex vs Claude for coding, LLM Routers and small model supremacy` 


- **XML eases agentic coding!**: Members are looking into **XML** as it makes it easier for the model to do **agentic coding**.
- **eBay MI50s tempt GPU Enthusiasts**: A member is tempted to throw some **MI50s** into an off-lease Xeon, noting they are very cheap on US eBay right now.
- **AMD RDNA5 Delayed by Financial Realities**: A member mentioned that by the time they have money for the new **AMD RDNA5**, they will probably just get a mini-server with some real meaty **AMD AI cards**.
- **Codex Coding Capabilities Questioned**: A member found **Codex** difficult to use for coding, stating that **Claude Code** was nowhere near as bad.
   - Another member said that *Codex got a lot better since last time* they used it, also adding that they are *still not sold on gpt-5 as being on par with claude*, as it constantly messes up things in **GitHub Copilot**.
- **Training LLM Routers is the Future**: A member suggested training **LLM routers** as a different angle to approach robustness, combining it with tool calls and endorsing small model supremacy.
   - Another member mentioned that their favorite **Tailwind CSS model** is **Tesslate's UIGEN T3**, crushing GPT-5 at design.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417443840888930338)** (2 messages): 

> `AI Boyfriend Relationships, Sketch-Based GNNs` 


- **AI Boyfriend Boom Foretold in 2025**: A [research paper](https://arxiv.org/abs/2509.11391) analyzed **1.5k posts** from r/MyBoyfriendIsAI, finding that many of these relationships begin *unintentionally* from casual chatting.
   - The paper notes that users develop **prompt-engineering** as a *love language*, but also face risks like **emotional dependency (~9%)** and **reality-dissociation (~4%)**.
- **GNN guru goes nuts for NLP**: A member is writing a research paper on advancing **sketch-based GNNs** using **NLP**, focusing on advanced vector quantization techniques to enhance semantic compression.
   - They're looking for someone with expertise in the field to review their proposal, specifically around using a separate NN (possibly a standard semantic encoder).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417443840888930338)** (2 messages): 

> `AI Boyfriend research, Sketch-based GNNs` 


- **AI 'Boyfriend' Study Shows Complex Human Effects**: A study analyzed **1.5k** posts from r/MyBoyfriendIsAI finding many relationships start unintentionally and prompt-engineering becomes a “love language”.
   - The paper ([My AI is a Boyfriend (2025)](https://arxiv.org/abs/2509.11391)) reports both benefits (≈**25%** feel less lonely) and risks like emotional dependency (**~9%**), reality-dissociation (**~4%**), and avoidance of human ties (**~4%**).
- **Sketchy GNNs Seek NLP Boost**: A member is writing a research paper on advancing **sketch-based GNNs** using **NLP** primarily through advanced vector and product quantization.
   - The aim is to enhance semantic compression via a separate NN and is looking for collaborators to look over the proposal.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1417244881042276424)** (4 messages): 

> `Tau Bench results with fastWorkflow and DSPy, VoxCron tool launch, GEPA for workflow optimization` 


- **DSPy and fastWorkflow Beat Claude Opus 4.1 on Tau Bench**: Using **DSPy** for agents and parameter extraction within **fastWorkflow**, a user matched **Claude Opus 4.1** performance on the **Tau Bench dev set** ([results here](https://cdn.discordapp.com/attachments/1202371242519441499/1417244881377693777/Tau_Bench_retail_using_fastWorkflow.png?ex=68cb1926&is=68c9c7a6&hm=845f74fb571d7893d54b6fe5b0b2e78b6878c890010338acac37be29f5080ae5&)).
   - The member stated, *"You CAN beat large models with proper scaffolding!"*, and suggested checking out the [retail workflow example](https://github.com/radiantlogicinc/fastworkflow) to test the agent.
- **VoxCron Automates Documentation and Diagrams**: A user launched **VoxCron** ([voxcron.com](https://voxcron.com)), a tool to streamline client spec review by automatically generating clean markdown documentation and mermaid diagrams.
   - The creator, who has spent a year building **DSPy projects** for clients, welcomes feedback on the tool's free tier.
- **GEPA eyed for fastWorkflow Optimization**: One member plans to use **GEPA** for end-to-end workflow optimization within **fastWorkflow**.
   - Another member asked them to share their experience using **GEPA**, and what could be improved to better support the agentic usecase.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1417273923866857542)** (65 messages🔥🔥): 

> `DSPy use cases, LM inference based on client, Refining topic matching in DSPy, Feeding classifier a list of topics, Arc-AGI leader prompt-optimization` 


- **DSPy Framework Validated for Topic Classification**: Users validated that **DSPy** is a useful framework for topic classification, especially given the potential for optimization, and noted that the framework is likely a better fit than others.
   - One user mentioned they were looking for something to try **DSPy** with, confirming they would test it out.
- **Inferring the LM Model Based on Client Possible**: One member suggested that even without knowing which **LM** is calling, it could be possible to make an assumption based on the client (e.g., if openai-mcp then gpt-5).
   - Another user mentioned that the idea of returning optimized tools based on the client would be cool.
- **Users Seek Better Topic Matching with Seed Phrases**: One user is trying to refine topic matching by providing a set of phrases for each topic in a way that can be optimized in **DSPy**.
   - They prefer to make the topic/phrase relationship semantically clear to **DSPy** optimizers.
- **Feeding Classifiers Lists of Topics in DSPy**: A user was struggling to feed the classifier a list of topics, noting it wanted a **Literal**, then demonstrated their workaround code using `pydantic` and `load_dotenv`.
   - Another user suggested that the list of topics can be passed as another input instead of using a **Literal**.
- **Prompt Optimization Helps User Score High on arc-agi**: A member shared an article on how the new arc-agi leader got there with prompt-optimization during test time, referencing [this substack article](https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again).
   - The article details prompt optimization strategies for the **ARC-AGI** challenge.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1417298215287390228)** (15 messages🔥): 

> `Mojo/Max and Python 3.13 Compatibility, Apple Metal Support Early Stages, MI355X Support in Nightly Version, Pixi Package Manager Benefits` 


- **Mojo and Python Party 🥳**: Mojo/Max is currently compatible with **Python 3.13** according to one of the staff in the general channel.
   - Staff also encouraged the use of the `pixi` package manager which handles isolated python versions.
- **Apple Metal Support still needs time to Metal-ize 🤘**: **Apple Metal support** is currently in its early stages, so performance may be slower than expected.
   - Running on just **CPU** should be fine, just slow.
- **MI355X Nightly Navigation 🧭**: Someone asked about the nightly version containing support for **MI355X**.
   - Staff mentioned the the changes are in **Mojo compiler downstream that pixi pulls**.
- **MAX Kernels stay locked in Step with Mojo Versions**: **MAX kernels** are barely parsed then stored in **.mojopkg files**, and then **MAX** uses the **Mojo compiler** to compile the rest of the way as part of graph compilation.
   - As such, **MAX** and **Mojo** stay in lockstep as far as versions are concerned.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1417226174219681863)** (17 messages🔥): 

> `Allocator API, Parametric traits and `requires`, Mojo LSP rework, Networking update blockers, Compiler bug on Mac with mojo test` 


- **Allocator API proposed**: Members discussed the possibility of an **allocator API** for Mojo, with one stating *I was also thinking about an allocator API being the way to go for this*.
   - Another member said their proposal was *Barely*, as they're waiting on **parametric traits and `requires`**.
- **Mojo's LSP is getting a major rework**: A member inquired about Mojo's Language Server Protocol (**LSP**), asking *Does mojo have a lsp yet?*
   - Another member confirmed it exists and is *getting a major rework soon*.
- **Networking update has lots of blockers**: Members discussed the **networking update** for Mojo, with one member stating *i still waiting for network update😔*.
   - Another member replied *Lots of blockers there*, and the first confirmed that *ye i know, hope we finish it soon*.
- **Mac Compiler Bug Spotted with Mojo Test**: A member reported a potential **compiler bug** occurring only on **Mac** when using **mojo test**.
   - The member linked to a [forum post](https://forum.modular.com/t/unexpected-argument-mutation-with-mojo-test-not-under-mojo-run/2203) with more details and asked for suggestions on how to proceed with reporting or further research.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1417441918320775178)** (7 messages): 

> `LLMs are Bayesian, Catastrophic forgetting, Online vs Batch Learning` 


- **LLMs are Bayesian Extension Toolkit Drops!**: A member shared a [new preprint and toolkit](https://x.com/ahmedkar_/status/1967875794333732947?s=46) extending on the "**LLMs are Bayesian in expectation**" paper.
   - The toolkit aims to expand the utility of Bayesian principles in large language models.
- **Catastrophic Forgetting Linked to LLMs as Bayesian**: A member noted that the **"LLMs are Bayesian"** paper reminded them of the "**catastrophic forgetting**" law, referencing a [paper on ArXiv](https://arxiv.org/pdf/2509.04259).
   - The member suggested the two papers *may* discuss similar concepts, warranting further investigation.
- **Online Learning vs Batch Learning Discussed**: A member sought to identify machine learning methods that need recalculation for new data versus those that only need updates, distinguishing **online learning** from **batch learning**.
   - Another member jokingly suggested *LLMs would one-shot tell you*, to which the original poster admitted they *don't have that automatism yet*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1417238045840900157)** (8 messages🔥): 

> `Yellow Line Smoothness, VaultGemma Released` 


- **Young Users Yield Smoother Data**: A member inquired about the smoothness of the yellow **18-25** line in a certain visualization, suggesting it might be due to a higher user count and reduced noise.
   - Another member noted that the noise for older cohorts seems to increase, with another attributing this to a potentially low number of available samples, which increases variance.
- **VaultGemma debuts as private LLM**: A member shared a link to the [VaultGemma blogpost](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm/), Google's latest venture in **differentially private LLMs**.
   - They also linked to the associated paper: [arxiv.org/abs/2509.05276](https://www.arxiv.org/abs/2509.05276).


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1417603870980444293)** (1 messages): 

> `Anthropic MCT Tools API, LLMs vs ARC Tool Use` 


- **Anthropic MCT Tools API is clean**: A member used the **Anthropic MCT Tools API** and stated that *it’s very clean to use*.
   - The member added that it *reminds a lot about how we were able to use the DSL packages where all the functions are in one file*.
- **LLMs > ARC tools**: A member expressed amazement at *how it seems easier for LLM ‘s to use tools than the shift and translocate functions in ARC*.
   - The member didn't link or further explain **ARC tools**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1417236303799521402)** (8 messages🔥): 

> `AI-powered PDF Editor, Agents to Payments Protocol, Arc Prize results` 


- **Google Hypes AI-Powered PDF Editor**: Google is advertising an **AI-powered PDF Editor** in the description of their new product, raising eyebrows due to the irony.
   - The new product is the [Agents to Payments (AP2) Protocol](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol), which is designed to streamline payment processes using AI agents.
- **Arc Prize Boasts High Scores**: The [Arc Prize](https://fxtwitter.com/arcprize/status/1967998885701538060) claims almost **80%** accuracy on v3 and **30%** on v2.
   - A member noted that the results might be cherry-picked since they don't allow everyone to have results verified, thus questioning its legitimacy as a real benchmark.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1417232050569609316)** (9 messages🔥): 

> `GPT-5 Codex, aider's chat-mode, aider's architect mode, code mode` 


- ****GPT-5 Codex** Aider Score still Unclear**: A member inquired about the **aider score** for the new **GPT-5 Codex** model, referencing [an article on The New Stack](https://thenewstack.io/openai-launches-a-new-gpt-5-model-for-its-codex-coding-agent/).
   - Another member responded that it is *not available through the API yet*.
- **Chat Mode: Is `--chat-mode code` Obsolete?**: A member noted that the `--chat-mode code` option appears to be invalid, suggesting that the documentation may need an update.
   - In response, another member clarified that the **default mode is chat mode**, so no flags are needed, and to use `/code` to get back to code mode.
- **Architect Mode is enhancing prompt instructions!**: A member reported that **architect mode** is enhancing their prompt instructions with context and sought a way to prevent this.
   - They expressed that they expected **code mode** to prevent enhancements.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1417309023111483463)** (4 messages): 

> `Gemini issues, Ollama endless loop, Architect Mode` 


- **Gemini Aider user having issues**: A user is having issues with **aider** and **gemini** where it hangs waiting for a response that never comes, even with a correct token and the issue persists across different **gemini** models.
   - Another user reported they are also experiencing the same issue as well.
- **Aider architect mode has endless loop with ollama**: A user reported using **aider** with a local LLM via **ollama** in architect mode and experiences an endless loop where it outputs code, realizes it's not a full implementation, and continues to work without intervention.
   - A member suggested to check the **context length** in **ollama** or to remove the `--yes-always` flag, as it might be the reason for the endless loop.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1417239737831395431)** (9 messages🔥): 

> `Simplicity vs Elegance, MI350 Kernel Benchmarks, MLIR-based compiler` 


- **Simplicity vs. Elegance in Software Design**: Discussion arose around Chris Lattner's idea that *complexity is the enemy*, with one member responding that *simplicity is the wrong measure* in design, and **elegance** and **usability** are better metrics for libraries and APIs.
   - They further explained that things should *fit together as puzzle pieces* to avoid fighting abstractions, noting that software is inherently complex.
- **Tinygrad to Publish MI350 Kernel Benchmarks**: tinygrad plans to publish **MI350 kernel benchmarks** by the end of the year, aiming for single kernel performance that is **80%** of NVIDIA's, with the goal of making tinygrad faster on whole jobs.
   - The goal *isn't to be the absolute fastest*, rather focusing on overall efficiency.
- **Pursuing an MLIR-based Compiler**: A member expressed plans to attempt creating an **MLIR-based compiler**, with another member suggesting they take previous points into account.
   - It was acknowledged that such an approach doesn't solve most of the big issues.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1417226698868396184)** (6 messages): 

> `Knowledge Limit, Credit Rollover, AI Loop & Refund` 


- **Knowledge Limit Increase: How?**: A member inquired about the possibility of exceeding the knowledge limit of **20** and how to achieve it.
   - The response to this question was not included.
- **Credit and Token Rollover**: A member asked if unused **tokens** and **credits** are rolled over upon renewal.
   - The answer to this question was not provided.
- **AI Enters Loop: Refund Request**: A member reported that the **AI** entered a very long loop and consumed all **3000 credits**, and requested a refund.
   - Another member stated that the same had happened to them since **September 4th**.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1417317929523085393)** (1 messages): 

> `golang streaming http MCP server, Scalability of MCP server, Auth in MCP server, Sessions and resumability in MCP server, Dynamic capabilities in MCP server` 


- **New Golang Streaming HTTP MCP Server Open-Sourced**: A member announced the open-sourcing of their [golang streaming http MCP server](https://github.com/ggoodman/mcp-server-go) project, designed for challenging enterprise-like requirements.
   - The offering includes features like **scalability**, **auth**, **sessions**, **resumability**, and **dynamic capabilities**.
- **MCP Server Scaling Solution Unveiled**: The newly released [MCP server](https://github.com/ggoodman/mcp-server-go) is designed for **scalability** with a pluggable backend interface, including in-memory and Redis streams backends.
   - This design choice intends to support scaling out in demanding enterprise environments.
- **MCP Server Integrates Auth with OIDC and JWT**: The [MCP server](https://github.com/ggoodman/mcp-server-go) can bootstrap the current authz spec from a simple issuer URL, assuming **OIDC discovery** and **JWT access tokens**.
   - Manual configuration is available for more nuanced setups, providing flexibility in authentication strategies.
- **Sessions and Resumability**: **Sessions** and **resumability** build upon the pluggable backend, giving you access to these difficult aspects of the protocol without having to _work for it_.
   - These features are designed to simplify the implementation of session management and resumability in MCP servers.
- **Dynamic Capabilities Streamline Resource Management**: The [MCP server](https://github.com/ggoodman/mcp-server-go) starts with a dynamic setup for handling tools, resources, and prompts from databases or APIs.
   - It also provides containers for static setups, catering to a range of deployment needs.


  


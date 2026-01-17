---
id: MjAyNi0w
title: ChatGPT starts testing ads on free tier + new $8/mo Go plan in the US
date: '2026-01-16T05:44:39.731046Z'
description: >-
  **OpenAI** announced the **ChatGPT Go** tier at **$8/month** with ads testing
  in the US free tier, emphasizing that ads will not influence responses and
  will be clearly labeled. The update includes memory improvements and a "very
  fast Codex" feature teased by **Sam Altman**. The Codex CLI ecosystem now
  supports open-weight models with improved context length. Discussions
  highlight the importance of human-in-the-loop for reliability in agent
  orchestration and file interface improvements over traditional
  retrieval-augmented generation.
companies:
  - openai
  - ollama
models:
  - chatgpt-go
  - codex
topics:
  - ads
  - monetization
  - memory
  - agent-orchestration
  - human-in-the-loop
  - cli-tools
  - context-length
  - workflow-optimization
people:
  - sama
  - sam_altman
  - fidjissimo
  - scaling01
  - tomwarren
  - embirico
  - adamdotdev
  - ollama
  - thsottiaux
  - lateinteraction
  - dbreunig
---


**Monetizing your consumers is all you need.**

> AI News for 1/15/2026-1/16/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**205** channels, and **4966** messages) for you. Estimated reading time saved (at 200wpm): **430 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

When you have 900 million weekly active users, you are usually long overdue in trying to figure out an ad supported model. Despite [a lot](https://x.com/tomwarren/status/2012295849678602610?s=46) of [snark](https://x.com/nickfloats/status/2012249130006143477?s=46) from commentators, OpenAI had to figure out their ads business and finally broke its silence today, outlining their [ads principles](https://x.com/OpenAI/status/2012223373489614951?s=20) in their tests that will roll out only in the US over the next free tier:

![https://pbs.twimg.com/media/G-zZl9kXwAAQut2?format=png&name=4096x4096](https://pbs.twimg.com/media/G-zZl9kXwAAQut2?format=png&name=4096x4096)

Most important statement in this is that ads never affect responses and are clearly labeled, which is the "right" move:

![https://pbs.twimg.com/media/G-zZXO-XcAAdvQo?format=jpg&name=4096x4096](https://pbs.twimg.com/media/G-zZXO-XcAAdvQo?format=jpg&name=4096x4096)

Formerly paid plans will not see ads, but the new Go plan (now rolled out in the US) will. The sheer number of pricing plans also [draws some confusion](https://x.com/simonw/status/2012271939629498386?s=46):

![https://pbs.twimg.com/media/G-0GmQtaQAAW_-F?format=jpg&name=4096x4096](https://pbs.twimg.com/media/G-0GmQtaQAAW_-F?format=jpg&name=4096x4096)




---

# AI Twitter Recap

**OpenAI product + monetization shifts (Go tier, ads, Codex speed, memory)**

- **ChatGPT Go + ads testing**: OpenAI announced **ChatGPT Go** (global rollout) as a **$8/month** low-cost tier with “10× more messages,” file uploads, image creation, more memory, longer context, and “unlimited use of GPT-5.2 instant” ([OpenAI](https://twitter.com/OpenAI/status/2012223323812270219)). In parallel, OpenAI said it will **start testing ads** in **Free + Go** tiers, with principles: **answers not influenced by ads**, ads clearly labeled, and “conversations private from advertisers” ([OpenAI](https://twitter.com/OpenAI/status/2012223373489614951); expanded by [@fidjissimo](https://twitter.com/fidjissimo/status/2012226082716393960) and [@sama](https://twitter.com/sama/status/2012253252771824074)). The announcement triggered heavy skepticism about inevitable incentive drift (e.g., [@scaling01](https://twitter.com/scaling01/status/2012234947403174189); and the resurfaced “ads as last resort” quote via [@tomwarren](https://twitter.com/tomwarren/status/2012295849678602610)).
- **Memory + “very fast Codex”**: Sam Altman highlighted “new ChatGPT memory improvements” ([\@sama](https://twitter.com/sama/status/2012242952542683227)) and repeatedly teased “**Very fast Codex coming!**” ([\@sama](https://twitter.com/sama/status/2012243893744443706)), with follow-on confirmation/teaser posts from developer ecosystem accounts ([\@embirico](https://twitter.com/embirico/status/2012320775370666004)). Multiple engineers discuss workflow-level impacts of the **speed vs intelligence** trade-off (e.g., shifting to more asynchronous “agent shepherding” when models are faster: [@adamdotdev](https://twitter.com/adamdotdev/status/2012142271819399663)).
- **Codex CLI ecosystem integrations**: Open-weight models can be used through the Codex CLI via Ollama using `codex --oss` ([\@ollama](https://twitter.com/ollama/status/2012046176267440177)), with a note to push context length to **≥32K** in settings for better UX ([\@ollama](https://twitter.com/ollama/status/2012049822484750426)). There’s also a new interaction UX: “steer codex mid-turn without interrupting” in an experimental mode ([\@thsottiaux](https://twitter.com/thsottiaux/status/2012074358471319599)).

**Agent tooling: orchestration UX, “human-in-the-loop” reliability, and file interfaces over classic RAG**

- **Human-in-the-loop as a reliability multiplier**: A recurring theme is that putting a human “babysitter” in the loop makes systems *feel* far more reliable than fully autonomous deployments using the same underlying models—because the human becomes a manual harness that catches failures and routes around ambiguity ([\@lateinteraction](https://twitter.com/lateinteraction/status/2012030585926189148); follow-up noting now there’s quantitative support for the intuition: [\@lateinteraction](https://twitter.com/lateinteraction/status/2012031028932854054)). Related: a chart discussion frames “the gap between the two lines” as the value of a human-in-the-loop ([\@dbreunig](https://twitter.com/dbreunig/status/2012200587211821410)).
- **“Chunking is dead” / files-first retrieval**: Jerry Liu argues that **RAG isn’t dead, but static chunking is**—if an agent can open a file, search (`ls`/`grep`), and expand context dynamically, you can avoid the brittle chunk/embed pipeline for many scales ([\@jerryjliu0](https://twitter.com/jerryjliu0/status/2012273236042559802); deeper clarification on why file tools work well up to a few hundred docs and where DBs re-enter: [\@jerryjliu0](https://twitter.com/jerryjliu0/status/2012254129473896532); emphasis on OCR as the missing piece for PDFs/PPTs: [\@jerryjliu0](https://twitter.com/jerryjliu0/status/2012272839416758652)). A separate synthesis frames this as “files aren’t replacing databases, but they’re forcing a rethink of when DBs are overkill” ([\@tuanacelik](https://twitter.com/tuanacelik/status/2012212183833403889)).
- **Orchestrators and agent UIs proliferate**: Multiple launches and memes point to a fast-moving layer of “agent harness” products: Anthropic’s Cowork is referenced as a signal of orchestration tools becoming mainstream ([\@alexalbert__](https://twitter.com/alexalbert__/status/2012230110745702563); meta commentary by [\@omarsar0](https://twitter.com/omarsar0/status/2012253642263249167)). SpecStory open-sourced a CLI to normalize agent session provenance/contracts ([\@doesdatmaksense](https://twitter.com/doesdatmaksense/status/2012209297380544940)). A new open-source UI (“sled”) lets you “teleport Claude Code or Codex from your computer to your phone” via Agent Control Protocol ([\@dctanner](https://twitter.com/dctanner/status/2012212217677070796)). OpenWork added native **Ollama integration** for fully local computer agents on Mac (Gemma/Qwen/DeepSeek/Kimi etc.) ([\@_orcaman](https://twitter.com/_orcaman/status/2012210613712281646)).  

**Inference + systems engineering: caching, Prefill/Decode split, hardware benchmarks, and CUDA tiling ergonomics**

- **“Year of inference explosion” framing**: A long Zhihu thread summary argues the bottleneck has shifted from training to inference: agents raise IO ratios (3:1 → 100:1 or 1000:1), **prefill dominates**, **context caching becomes default**, and Prefill/Decode splitting harms utilization unless you redesign scheduling and memory hierarchy ([\@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2012080310981374428)). This aligns with broader infra chatter around cache affinity vs load balance trade-offs.
- **Hardware benchmarking beyond NVIDIA**: Artificial Analysis added **DeepSeek R1** results on SambaNova SN40L, showing higher throughput at concurrency and standout per-user speeds (noted peak ~269 tok/s single-user) vs tested NVIDIA configurations—while flagging lack of public hourly pricing for cost comparisons ([\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012233319891824943); [\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012233323154678010)).
- **CUDA tiling / CuTe / cuTile ergonomics**: Engineers are enthused about **CuTe algebra** as a cleaner abstraction for tiling/indexing compared to hand-rolled CUDA gymnastics ([\@fleetwood___](https://twitter.com/fleetwood___/status/2012150019722485811)), alongside pointers to scarce “mere mortal” resources ([\@fleetwood___](https://twitter.com/fleetwood___/status/2012151045992992943)). NVIDIA’s newer “CUDA Tile”/cuTile guidance is summarized as enabling near–cuBLAS GEMM performance with simpler block-level code and compiler specialization (plus swizzling improvements) ([\@TheTuringPost](https://twitter.com/TheTuringPost/status/2012288767894360215)).
- **Data center power scaling**: Epoch AI estimates AI data centers now have total capacity around **30 GW**, comparable to New York State peak hot-day usage; methodology multiplies chip units sold by rated draw and applies ~2.5× facility overhead, with caveats about “capacity vs usage” ([\@EpochAIResearch](https://twitter.com/EpochAIResearch/status/2012303496465498490)).

**Model & research highlights: voice cloning without tokenization, ultra-small models, multimodal + retrieval advances**

- **Tokenization-free real-time TTS**: OpenBMB open-sourced **VoxCPM** weights for real-time streaming voice cloning, described as generating **continuous speech directly** (avoiding discrete audio token artifacts), with LoRA fine-tuning and ~0.15 real-time factor on a single RTX 4090 per the tweet ([\@LiorOnAI](https://twitter.com/LiorOnAI/status/2012133013967044755); repo link [\@LiorOnAI](https://twitter.com/LiorOnAI/status/2012133015426642286)). If accurate, it’s a meaningful shift for latency/prosody fidelity in production voice agents.
- **Small-model reasoning & edge deployments**: TII promoted **Falcon-H1-Tiny** (<100M params) as capable of reasoning/coding/function calling for edge/IoT scenarios ([\@TIIuae](https://twitter.com/TIIuae/status/2012034581084430662)). Ultralytics released **YOLO26** family (30 models, <50M params) spanning detection/segmentation/keypoints/open-vocab, with demos on CPU ([\@mervenoyann](https://twitter.com/mervenoyann/status/2012121123018924033)).
- **Multilingual translation**: TranslateGemma gained attention for multilingual breadth (incl. Malayalam) and tokenizer/data work ([\@_arohan_](https://twitter.com/_arohan_/status/2012032986649448708); [\@JeffDean](https://twitter.com/JeffDean/status/2012178747076591820)), and is available in Ollama with a specific prompting format ([\@ollama](https://twitter.com/ollama/status/2012307436284395692)).
- **Retrieval: multi-vector resurgence**: Strong claims that **multi-vector retrieval** can let tiny models compete with much larger baselines (e.g., “32M parameter multi vector model” approaching an 8B model) ([\@aaxsh18](https://twitter.com/aaxsh18/status/2012124348392583584)), echoed by “multi vector is the only way forward” ([\@lateinteraction](https://twitter.com/lateinteraction/status/2012227085507449197)) and practitioner reinforcement about ColBERT/ColPali-style wins across tasks ([\@antoine_chaffin](https://twitter.com/antoine_chaffin/status/2012269641490391272)).
- **Preference data design for alignment (AIR)**: OpenBMB’s AIR framework decomposes preference datasets into **Annotations / Instructions / Response pairs**, claiming best practices: simpler scoring, filtering instructions by low variance, and balancing pair gaps/quality; reported +5.3 average gain across 6 benchmarks using 14k curated pairs ([\@OpenBMB](https://twitter.com/OpenBMB/status/2012179938388926679)).

**Generative media: open image/video releases, motion control workflows, and diffusion “Neural OS”**

- **FLUX.2 [klein] lands everywhere (open weights, vLLM day-0, leaderboards)**: Black Forest Labs’ **FLUX.2 [klein]** got “day-0 support” in **vLLM-Omni**, positioned as consumer-friendly (<~13GB VRAM), sub-second inference, Apache-2.0 licensed 4B model (per tweet) ([\@vllm_project](https://twitter.com/vllm_project/status/2012110024294965406)). Arena and Artificial Analysis report strong open-model leaderboard placements ([\@arena](https://twitter.com/arena/status/2012310336528056520); [\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012339542997737856)).
- **Open video model rankings**: Artificial Analysis notes **LTX-2** as leading open-weights video model in their Video Arena, with licensing caveats (LTX-2 Community License, commercial use under revenue threshold and non-compete constraints) ([\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012256702788153604)).
- **Kling motion control + “AI mocap”**: Multiple threads highlight motion-control and mocap-style workflows enabling fast character swaps and transferable acting/performance ([\@HAL2400AI](https://twitter.com/HAL2400AI/status/2012038846960328781); tutorial from [\@Kling_ai](https://twitter.com/Kling_ai/status/2012155500134105149); “AI motion capture… copy/paste motion/expression/lips” ([\@EHuanglu](https://twitter.com/EHuanglu/status/2012149076511617436)); examples roundup ([\@minchoi](https://twitter.com/minchoi/status/2012306052956533211)).

**Top tweets (by engagement)**

- OpenAI ads principles announcement ([\@OpenAI](https://twitter.com/OpenAI/status/2012223373489614951)) and Go tier launch ([\@OpenAI](https://twitter.com/OpenAI/status/2012223323812270219)).
- Sam Altman on ads rollout/principles ([\@sama](https://twitter.com/sama/status/2012253252771824074)) and “Very fast Codex coming” ([\@sama](https://twitter.com/sama/status/2012243893744443706)).
- Viral diffusion “OS in a model” / Neural OS posts ([\@jxmnop](https://twitter.com/jxmnop/status/2012048155379220746); follow-up details [\@jxmnop](https://twitter.com/jxmnop/status/2012283763720601727)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. New Model and Benchmark Releases

  - **[GPT-5.2 xhigh, GLM-4.7, Kimi K2 Thinking, DeepSeek v3.2 on Fresh SWE-rebench (December 2025)](https://www.reddit.com/r/LocalLLaMA/comments/1qefa7q/gpt52_xhigh_glm47_kimi_k2_thinking_deepseek_v32/)** (Activity: 473): **The December 2025 update to the **SWE-bench leaderboard** features evaluations of several prominent models on 48 new GitHub PR tasks. **Claude Opus 4.5** leads with a `63.3%` resolved rate, followed by **GPT-5.2 xhigh** at `61.5%`. Notably, **Gemini 3 Flash Preview** outperforms its Pro counterpart despite being smaller and cheaper, and **GLM-4.7** ranks as the top open-source model, comparable to closed models like GPT-5.1-codex. The performance of **GPT-OSS-120B** in high-effort reasoning mode underscores the benefits of inference-time scaling. For more details, see the [SWE-rebench Leaderboard](https://swe-rebench.com/?insight=dec_2025).** Commenters highlight the surprising performance of Gemini 3 Flash Preview and express enthusiasm for GLM-4.7's ranking among the top models, noting skepticism about other benchmarks that overstate the performance of open models like GLM 4.7 or Minimax 2.1.

    - The mention of **Gemini Flash** as a 'real shocker' suggests it performed unexpectedly well in the benchmark, indicating a significant improvement or innovation in its architecture or training that wasn't anticipated by the community.
    - The **GLM 4.7** model's inclusion in the top 10 of the benchmark is notable because it is an open model, which typically face challenges in competing with proprietary models due to resource constraints. This achievement highlights the model's efficiency and capability, possibly due to recent optimizations or novel techniques.
    - The skepticism towards benchmarks that equate **GLM 4.7** or **Minimax 2.1** with **Opus 4.5** suggests a belief that these models are not yet on par with Opus 4.5 in terms of performance. This could be due to differences in training data, model architecture, or other technical factors that affect their capabilities.

  - **[7x Longer Context Reinforcement Learning in Unsloth](https://www.reddit.com/r/LocalLLaMA/comments/1qdna3t/7x_longer_context_reinforcement_learning_in/)** (Activity: 288): **The image is a promotional graphic for Unsloth's new capability to extend context lengths in reinforcement learning by up to 7x, reaching up to 12x in some cases. This advancement allows training of models like gpt-oss 20b QLoRA with up to `20K` context on a `24Gb` card without accuracy degradation. For larger GPUs, Unsloth can handle `380K` context on a `192GB` NVIDIA B200 GPU. The image includes graphs that compare context length against GPU VRAM for different models, showcasing improvements in context length due to new data movement and batching algorithms. These enhancements are achieved without compromising accuracy or speed, and are applicable to various models including Llama and Gemma.** A commenter questions the availability of proper training data for such long contexts, suggesting that real-world tasks may not have sufficient instruction/QA data. Another user inquires about the applicability of these advancements to the Qwen3 30B-3A model.

    - PlasticTourist6527 raises a critical point about the availability of long-context training data, especially for real-world tasks. They suggest that outside of specific domains like coding, there might be a scarcity of high-quality instruction or QA data that can support training models with extended context lengths.
    - 1ncehost reports issues with training a model on ROCm, noting that they had to apply deep patches and replace kernels to resolve problems with the latest versions. They also observed that SDPA was the fastest attention mechanism for the Qwen3 0.6B model, outperforming FA2 and xformers by a significant margin, indicating potential optimizations in attention mechanisms for specific model sizes.
    - knownboyofno inquires about the applicability of the extended context reinforcement learning approach to the Qwen3 30B-3A model, suggesting interest in understanding the scalability and compatibility of the technique with larger models.


### 2. High-Performance AI Hardware and Upgrades

  - **[Latest upgrade…A100 40 GB](https://www.reddit.com/r/LocalLLaMA/comments/1qe0cxc/latest_upgradea100_40_gb/)** (Activity: 466): **The image showcases a high-performance computer setup that has been upgraded with an NVIDIA A100 GPU, which is significant for AI and machine learning tasks due to its high computational power. The user initially had a gaming rig but transitioned to a more AI-focused setup by acquiring an A100 GPU, which was listed as faulty but turned out to be functional. This upgrade allows for running and training larger AI models efficiently, leveraging the A100's capabilities. The setup includes a GeForce RTX card, RGB-lit fans, and an NZXT liquid cooler, indicating a balance between aesthetics and performance.** The comments reflect a mix of admiration and humor, with one user joking about the risk taken in purchasing a potentially faulty GPU and another referencing a meme about NVIDIA's CEO, Jensen Huang.

    - matatonic raises a critical point about cooling for the A100 40 GB, noting that it appears to be a passively cooled version. They suggest using a blower fan or another active cooling method to prevent overheating. Additionally, they mention the possibility of using water cooling solutions, which are available on platforms like AliExpress, to ensure the GPU operates within safe temperature ranges.

  - **[M4/M5 Max 128gb vs DGX Spark (or GB10 OEM)](https://www.reddit.com/r/LocalLLM/comments/1qcmmvw/m4m5_max_128gb_vs_dgx_spark_or_gb10_oem/)** (Activity: 188): **The user is comparing the NVIDIA DGX Spark and a MacBook Pro with M4 Max (128GB RAM) for local LLM inference, primarily for coding tasks such as code completion and refactoring. The DGX Spark offers a CUDA ecosystem and strong GPU compute, while the MacBook Pro benefits from unified memory and Apple's ML stack. For inference tasks, the MacBook's higher memory bandwidth is advantageous, but it may not match the performance of cloud-based solutions like Claude. The M5 chip shows improved performance over the M4, and new MacBook models may be released soon. The MacBook is noted for faster inference, but NVIDIA's CUDA support is more comprehensive. The Mac Studio with M4 Max is suggested as a cost-effective alternative if portability is not required.** Commenters debate the performance of Apple Silicon versus NVIDIA hardware, with some asserting that the MacBook Pro offers superior text generation performance due to its memory bandwidth, while others highlight NVIDIA's broader capabilities in fine-tuning and multimodal tasks. The discussion also touches on the potential cost-effectiveness of the Mac Studio for non-portable use.

    - The M4 Max offers significantly higher memory bandwidth compared to the DGX Spark, which is beneficial for inference tasks. However, the Spark benefits from better support for frameworks due to its compatibility with NVIDIA's CUDA. This makes the MacBook faster for inference, but the Spark is more versatile for tasks like fine-tuning and image generation.
    - The M3 Ultra Mac Studio is highlighted as superior for pure text generation tasks compared to the DGX Spark. While NVIDIA hardware is generally more capable on paper, the M3 Ultra reportedly outperforms in specific LLM inference tasks. This is attributed to the Mac's efficiency in handling agentic coding workflows, despite the Spark's broader capabilities in other areas.
    - The DGX Spark is noted for its compact size and energy efficiency, consuming less than 100W and idling at around 10W. It is praised for its extensibility, allowing for additional units to be connected. However, concerns about bandwidth limitations are raised, and the cost comparison with alternatives like the GB10 OEM and MacBook Pro is discussed.

  - **[RTX 5070 Ti and RTX 5060 Ti 16 GB no longer manufactured](https://www.reddit.com/r/LocalLLaMA/comments/1qdh28f/rtx_5070_ti_and_rtx_5060_ti_16_gb_no_longer/)** (Activity: 414): ****Nvidia** has ceased production of the `RTX 5070 Ti` and significantly reduced the supply of the `RTX 5060 Ti 16 GB` due to memory supply shortages, leading to a price increase of approximately `$100` over MSRP for the 5070 Ti. The 8 GB configuration of the RTX 5060 Ti remains unaffected. This decision impacts most AIBs, who will no longer manufacture these GPUs. [Source](https://m.youtube.com/watch?v=yteN21aJEvE).** One user noted the RTX 5060 Ti 16 GB as a cost-effective option for adding Nvidia memory to systems, highlighting its suitability for DLSS, AI processing, and inferencing tasks, especially with `64GB VRAM` for `70B models`. Another user expressed disappointment over the halted production affecting their upgrade plans, while a third criticized Nvidia's business practices.

    - The RTX 5060 Ti 16 GB is highlighted as a cost-effective option for adding Nvidia memory to systems, especially for tasks like image generation, inferencing, and gaming. At a price point of around `$350-$390`, it offers good value with features like DLSS and AI processing capabilities. The card's `16 GB GDDR7` memory compensates for its `128-bit bus`, making it comparable to a `192-bit bus GDDR6` card, thus supporting demanding tasks like DLSS and ray tracing without sacrificing texture quality.
    - The RTX 5060 Ti 16 GB is noted for its suitability in budget inferencing setups, particularly for those unable to access RTX 3090s. With the ability to fit multiple cards into a standard power supply machine, it supports new quantization methods and can handle `70B models` effectively with `64 GB VRAM`. This makes it a viable option for small-scale AI tasks, leveraging its memory capacity and efficiency for practical applications.


### 3. Local LLM Community and Innovations

  - **[[MOD POST] Announcing the r/LocalLLM 30-Day Innovation Contest! (Huge Hardware &amp; Cash Prizes!)](https://www.reddit.com/r/LocalLLM/comments/1olbrch/mod_post_announcing_the_rlocalllm_30day/)** (Activity: 120): **The r/LocalLLM subreddit has launched a **30-Day Innovation Contest** focused on open-source projects for AI inference or fine-tuning, with significant hardware and cash prizes. The contest encourages submissions of innovative projects such as new serving frameworks, quantization methods, fine-tuning techniques, or performance benchmarks, using diverse hardware like **NVIDIA, Google Cloud TPU,** or **AMD**. The top prize includes an **NVIDIA RTX PRO 6000** and cloud time on an **8x NVIDIA H200 server**. Participants are encouraged to submit their projects via a new post on r/LocalLLM with the 'Contest Entry' flair, including a public repository link and demonstration materials.** One commenter expressed enthusiasm for saving projects for future exploration, while another inquired about sharing projects for community inspiration. A third commenter sought clarification on the submission process, indicating interest in participating.


  - **[Small AI computer runs 120B models locally: Any use cases beyond portability and privacy?](https://www.reddit.com/r/LocalLLM/comments/1qcu498/small_ai_computer_runs_120b_models_locally_any/)** (Activity: 107): ****TiinyAI** has developed a compact AI device capable of running `120B` parameter models locally with `80GB RAM` and a power consumption of `30W`. This device is positioned as a more portable and cost-effective alternative to larger systems like the **DGX Spark**, which offers `128GB RAM` and higher performance but at a greater cost and size. The TiinyAI device is particularly notable for its potential applications in scenarios where **portability** and **privacy** are prioritized over raw performance, such as in field operations or environments with limited internet access. However, concerns remain about its **memory bandwidth**, which is speculated to be between `80Gb/s` and `200Gb/s`, potentially limiting its performance compared to traditional PCs or laptops.** Commenters express skepticism about the device's price and availability, with one noting that $1400 seems high for an 80GB RAM SBC. Another highlights the device's potential utility in scenarios where internet access is restricted, such as under authoritarian regimes.

    - A key technical concern raised is the memory bandwidth of the small AI computer, with estimates ranging from 80Gb/s to 200Gb/s. This bandwidth is crucial for running large models like 120B parameters efficiently. If the bandwidth is on the lower end, it may not outperform a regular PC or laptop, which could limit its utility for high-performance tasks.
    - The pricing of the device, speculated to be around $1400 for an 80GB RAM single-board computer (SBC), is questioned. The skepticism is due to the lack of availability for immediate purchase, which raises doubts about the feasibility and practicality of the device at this price point.
    - The device's built-in microphone and speaker suggest potential use as a private AI assistant. This setup could allow users to run automation scripts and manage tasks locally, providing a privacy-focused alternative to cloud-based assistants like Alexa or Siri. This use case leverages the device's ability to handle personal data securely without cloud dependency.

  - **[I fucking love this community](https://www.reddit.com/r/LocalLLaMA/comments/1qee2de/i_fucking_love_this_community/)** (Activity: 469): **The post highlights the ability to run large models like `nemotron-3-nano-30B-a3b-iq4_nl` at `14-13.5 t/s` on a decade-old PC with only `4GB VRAM`, thanks to optimizations from projects like **llama.cpp** and **vllm**. The key to achieving this performance is leveraging a significant amount of system memory and utilizing models with a *Mixture of Experts (MoE)* architecture, which allows for efficient resource usage and performance on limited hardware.** Commenters express amazement at the performance achieved on old hardware, emphasizing the effectiveness of combining system RAM with MoE architectures. There's also interest in accessing resources or posts that detail these optimizations for running large models on low-end equipment.

    - InfiniteLand7364 highlights achieving `14 t/s` (tokens per second) on a decade-old system, emphasizing the community's skill in optimizing older hardware for performance. This suggests that with the right tweaks, even outdated systems can handle tasks typically reserved for newer machines.
    - Rokpiy mentions the effectiveness of combining system RAM with 'moe' (likely referring to a specific optimization or model configuration), which is often overlooked but offers practical benefits. This implies that leveraging existing hardware resources creatively can enhance performance without needing the latest technology.
    - cosimoiaia discusses the educational value of working within hardware constraints, suggesting that it forces users to learn deeply about model tuning and system optimization. This experience not only improves current performance but also prepares users for future technological advancements by understanding what hardware and configurations are most effective.

  - **[My story of underestimating /r/LocalLLaMA's thirst for VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1qe2i88/my_story_of_underestimating_rlocalllamas_thirst/)** (Activity: 1291): **The image is a meme that humorously illustrates the unintended consequences of sharing technical insights on Reddit. The original poster bought a w6800 32GB graphics card for $500, found it to perform well, and shared this information on Reddit. This led to a significant increase in the card's price to over $1,000, highlighting the impact of community discussions on market dynamics. The post underscores the high demand for VRAM in the /r/LocalLLaMA community, which can drive up prices when a product is recommended.** One commenter humorously compares the situation to the California gold rush, suggesting strategic withholding of information to capitalize on market opportunities. Another commenter provides technical advice, suggesting alternatives like the 3090 or R9700 for those concerned with VRAM and cooling solutions.

    - EmPips discusses the trade-offs between different GPU models for VRAM-intensive tasks. They suggest that while the card in question is impressive, the **NVIDIA RTX 3090** might be a better choice at current prices. Alternatively, they recommend the **AMD Radeon Pro VII (R9700)** for those who prioritize VRAM-per-slot and are okay with high idle power and external cooling, suggesting the **AMD MI50** as another option for those willing to manage these factors.

  - **[What is the biggest local LLM that can fit in 16GB VRAM?](https://www.reddit.com/r/LocalLLM/comments/1qcuyh2/what_is_the_biggest_local_llm_that_can_fit_in/)** (Activity: 155): **The largest local LLM that can fit in 16GB VRAM, such as on an RTX 5080, is typically around `14B` parameters when considering practical usage constraints. This is due to the need to leave room for context, which means a model file size should ideally be around `14GB`. Models like `GPT-OSS-20B` can run but may require significant quantization, potentially below `4-bit`, which can degrade quality. For optimal performance without excessive slowdowns, models around `14B` are recommended. Users can check model sizes on platforms like [HuggingFace](https://huggingface.co/) to ensure they fit within VRAM limits.** Commenters suggest that while models up to `30B` might technically fit with aggressive quantization, the performance and quality trade-offs make `14B` a more practical choice. The importance of considering model file size over parameter count is emphasized, as exceeding VRAM capacity leads to slowdowns due to RAM overflow.

    - BigYoSpeck discusses the performance of various models on a system with a Ryzen 9 5900x, 64GB DDR4 3800, and a 16GB Radeon RX 6800 XT. They report running `gpt-oss-20b` at over 120 tokens per second, `Qwen3 30b` partially offloaded to CPU at about 40 tokens per second, and `gpt-oss-120b` with 32 MOE layers offloaded to CPU at 23 tokens per second. This suggests that with a similar setup, one might achieve even better performance.
    - SKirby00 highlights the limitations of running large models on 16GB VRAM, noting that models like `Qwen3-Coder-30B` require significant VRAM and context space. They suggest that a 14.5GB model might technically fit but would be impractical due to limited context space. They recommend aiming for models around the 14B parameter range for better usability, given the constraints of 16GB VRAM.
    - vertical_computer emphasizes the importance of considering model file size relative to VRAM capacity. They suggest that a model should ideally be around 14GB to fit within 16GB VRAM, leaving room for context. They provide an example with the `Nvidia Llama 3.3 Nemotron 49B` model, noting that larger models will spill over into RAM, significantly slowing down performance.

  - **[Oh Dear](https://www.reddit.com/r/LocalLLM/comments/1qdiwdh/oh_dear/)** (Activity: 115): **The image depicts a malfunction in an AI model's response, where it outputs a repetitive string of 'the,' suggesting a potential issue with the model's configuration or prompt handling. This could be due to an incorrect system prompt or tuning parameters like temperature not being set appropriately. The comments suggest checking the system prompt and ensuring it aligns with the model's requirements, as some models may not function correctly without a proper system prompt.** Commenters suggest that the issue might be related to the absence of a system prompt or incorrect tuning parameters, such as temperature, which are crucial for generating coherent responses.

    - mp3m4k3r suggests checking the tuning parameters, specifically the temperature setting, to ensure it aligns with the model's recommended usage. This is crucial for maintaining the model's performance and preventing issues like repetitive outputs.
    - HealthyCommunicat recommends adjusting the repeat penalty, starting at `1.1` and increasing if necessary. This adjustment can help mitigate issues with local LLMs producing repetitive text. Additionally, they advise ensuring the model isn't using more experts than recommended, which can also lead to performance problems.
    - ScoreUnique mentions using 'pocket pal' for loading `gguf` files, which could be a solution for handling specific file types or formats in local LLM setups. This tool might be beneficial for users dealing with compatibility or loading issues.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude and Gemini Model Updates and Issues

  - **[Official: Claude Cowork is now available to "Pro" subscribers](https://www.reddit.com/r/ClaudeAI/comments/1qeo736/official_claude_cowork_is_now_available_to_pro/)** (Activity: 353): ****Claude Cowork** is now available to "Pro" subscribers, as announced by Claude on X.com. This feature, still in research preview, includes session renaming, connector improvements, and fixes based on early feedback. However, it is noted that Pro users might reach their usage limits faster due to Cowork's capability to handle more complex tasks. The announcement also provides a link to try it in the macOS app.** Users express concerns about hitting usage limits quickly, with one user noting that sorting 459 files used 97% of their session limit. Another user comments on the restrictive usage limits of Claude, while a third hopes for useful applications despite not using Claude for coding.

    - A user reported that using Claude Cowork for sorting 459 files consumed 97% of their session's usage limit, highlighting the restrictive nature of the current usage caps. This suggests that the tool may not be suitable for high-volume tasks without hitting limits quickly.
    - Another user expressed dissatisfaction with Claude's usage limits, indicating that they are among the worst compared to other services. This sentiment suggests that the current limitations may hinder productivity and user satisfaction, especially for those who rely on the tool for extensive tasks.
    - A user mentioned their reluctance to upgrade to a 'max plan' due to not using Claude for coding, implying that the current subscription tiers may not align well with diverse user needs. This points to a potential gap in the service offerings for non-coding related use cases.

  - **[🌊 Announcing Claude Flow v3: A full rebuild with a focus on extending Claude Max usage by up to 2.5x](https://www.reddit.com/r/ClaudeAI/comments/1qegsta/announcing_claude_flow_v3_a_full_rebuild_with_a/)** (Activity: 291): ****Claude Flow v3** is a comprehensive rebuild of the AI orchestration platform, designed to enhance the usage of Claude Max by up to `2.5x`. The system, rewritten in **TypeScript** and **WASM**, features a modular architecture that supports deploying multi-agent swarms with shared memory and continuous learning. It reduces token consumption by `75-80%` and improves subscription capacity by `250%`. The platform is built on `npm RuVector` with deep **Rust** integrations and supports offline execution, allowing for local model use without consuming tokens. Governance is enforced through ADRs, DDD boundaries, and SPARC, ensuring traceability and security. The system operates as an always-on daemon with live updates and automated tasks for optimization and security audits. For more details, see the [GitHub repository](https://github.com/ruvnet/claude-flow).** Some commenters express skepticism about the claims, noting the use of buzzwords and unsubstantiated performance metrics, while others are intrigued by the potential of multi-agent systems but question their practical effectiveness compared to base LLMs.

    - janusr raises concerns about the project's claims, highlighting the use of buzzwords and unsubstantiated metrics such as 'Agent Booster 352x faster' without clear benchmarks or comparisons. They question the relevance of ONNX Embeddings being '75x faster than Transformers.js' to the project's goals, suggesting skepticism about the practical benefits of these claims.
    - Infamous_Research_43 expresses skepticism about frameworks claiming to manage large swarms of agents, noting a pattern of such projects failing to deliver on their promises. They argue that many creators lack a fundamental understanding of AI and agent-based systems, often confusing them with LLM chatbots, and warn that these projects are frequently scams or poorly executed.
    - sridoodla mentions issues with outdated documentation in previous versions and inquires about the stability of v3, indicating a need for reliable and up-to-date resources to effectively utilize the tool. This highlights a common challenge in rapidly evolving AI projects where documentation often lags behind development.

  - **[Today, Gemini 3 Pro became unusable to me as a Pro subscriber](https://www.reddit.com/r/GeminiAI/comments/1qemf0h/today_gemini_3_pro_became_unusable_to_me_as_a_pro/)** (Activity: 183): **A user reports that **Gemini 3 Pro**, a tool they have relied on for building complex applications, has become unusable due to a significant drop in performance. The user experienced an issue where the model provided irrelevant code ('Shopping Cart' instead of a document upload feature), indicating potential problems with the model's context understanding. This aligns with other users' observations of a reduced context window, which may lead to increased hallucinations. Some users suggest alternatives like **GPT 5.2 Thinking** for better performance.** There is a debate on the model's performance, with some users experiencing significant issues due to a reduced context window, while others still find it effective for different tasks, such as philosophical discussions. The discussion highlights a divide in user experience, possibly due to varying use cases.

    - xbrasil highlights a significant reduction in the context window for Gemini 3 Pro, even for paying users, which has led to increased hallucinations and decreased usability. They suggest that GPT 5.2 Thinking is a viable alternative, indicating a shift in user preference due to perceived neglect from Google.
    - VanillaSwimming5699 compares Gemini 3 Pro favorably for coding tasks, noting its deep philosophical discussion capabilities. However, they mention that '3 flash' might be superior due to faster iteration and lower costs, while Opus 4.5 is also competitive but has an earlier knowledge cutoff.
    - TheLawIsSacred shares that Gemini 3 has been largely unusable recently, but they are waiting for potential improvements based on past experiences with model updates. They currently rely on alternatives like Claude Desktop app (Opus 4.5), Perplexity Pro (Sonnet 4.5 with Reasoning), and ChatGPT (5.2) for reliable performance.



### 2. AI Model and Benchmark Releases

  - **[[R] China just released first SOTA multimodal model trained entirely on domestic chips](https://www.reddit.com/r/MachineLearning/comments/1qeakhz/r_china_just_released_first_sota_multimodal_model/)** (Activity: 49): ****Zhipu AI** and **Huawei** have released **GLM-Image**, a state-of-the-art multimodal model trained entirely on **Huawei Ascend 910** chips, marking a significant milestone in AI development using domestic hardware. The model employs a hybrid architecture with an autoregressive and diffusion decoder, excelling in Chinese text rendering, and supports resolutions from `1024 to 2048` without additional training. It offers both text-to-image and image-to-image generation capabilities, with API pricing set at `0.1 yuan` per image. Notably, the model claims `60%` better compute efficiency than Nvidia's H200 in terms of tokens per joule, challenging the reliance on Nvidia hardware for training advanced models. The model's repositories are available on [GitHub](https://github.com) and [Hugging Face](https://huggingface.co).** A key technical question raised is about the model's compatibility with frameworks like PyTorch and cuDNN, given its development on non-Nvidia hardware, and whether it can be executed on other machines.

    - The discussion revolves around the technical feasibility of running a state-of-the-art multimodal model on non-NVIDIA hardware, specifically using domestic Chinese chips. The commenter questions the compatibility of such models with frameworks like PyTorch and cuDNN, which are traditionally optimized for NVIDIA GPUs. This raises concerns about the adaptability of these models to other hardware environments and the potential need for alternative libraries or custom solutions to achieve similar performance levels.

  - **[[D] Why Mamba rewrote its core algorithm and Microsoft abandoned RetNet](https://www.reddit.com/r/MachineLearning/comments/1qehwlu/d_why_mamba_rewrote_its_core_algorithm_and/)** (Activity: 131): ****Mamba-2** has restructured its core algorithm from parallel scans, which utilized `10-20%` of Tensor Core capacity, to block-diagonal GEMMs, achieving `60-70%` utilization, optimizing for NVIDIA's hardware. Meanwhile, **Microsoft Research** published **RetNet** in July 2023, a promising architecture at `6.7B` parameters, but quickly shifted focus to dense Transformers with Phi-2, Phi-3, and Phi-4, indicating a lack of institutional backing for RetNet. This pattern highlights the co-evolution of Transformers and NVIDIA GPUs, creating a stable attractor that is difficult to break due to the dual challenges of hardware compatibility and institutional support. The essay includes Tensor Core utilization statistics, analysis of alternative chip vendors, and predictions for 2028. [Full essay link](https://open.substack.com/pub/lambpetros/p/the-transformer-attractor).** Commenters agree on the trend of co-evolution between model architectures and hardware, noting that incentives favor incremental improvements over radical changes. The RetNet case is debated, with uncertainty about whether its abandonment was due to hardware issues, quality concerns, or risk aversion. Some suggest that experimental architectures like RetNet may still influence future developments, as seen with some large Chinese models.

    - The comment by thearn4 highlights a trend in machine learning and high-performance computing (HPC) where there is a coevolution of model formulation, solver structure, and hardware. This trend suggests that incremental development is often favored over radical changes due to better incentives, which is a common pattern across various technical fields.
    - petroslamb points out the ambiguity surrounding Microsoft's abandonment of RetNet, noting that the lack of public experiments makes it unclear whether the decision was due to hardware scaling issues, quality degradation beyond a certain model size, or risk aversion. This highlights a gap in transparency that could inform future research and development in model architectures.
    - Xemorr challenges the assumption that parallel scans can be optimized as effectively as block-diagonal General Matrix Multiply (GEMM) operations, suggesting a technical debate on the efficiency of different computational strategies in model training and inference.

  - **[[D] ICASSP 2026 Results](https://www.reddit.com/r/MachineLearning/comments/1qeips6/d_icassp_2026_results/)** (Activity: 73): **The post discusses a potential early access to ICASSP 2026 acceptance results through a specific [link](https://cmsworkshops.com/ICASSP2026/author_invitation_request.php). Users who could send an invitation email through this link might have had their papers accepted. The email confirms acceptance for presentation at the IEEE ICASSP 2026 in Barcelona, Spain, from May 3-8, 2026. However, an update indicates that the link is currently inaccessible, showing an error message: *'Error: No match for paper number and password. 0x4C'.*** Comments indicate confusion about the accessibility of the results, with some users reporting initial access followed by subsequent errors, suggesting a possible bug that was later fixed.



### 3. AI Tools and User Experiences

  - **[Why AI coding tools accidentally feel perfect for inattentive ADHD brains](https://www.reddit.com/r/ClaudeCode/comments/1qeb6od/why_ai_coding_tools_accidentally_feel_perfect_for/)** (Activity: 238): **The post discusses how AI coding tools, like **Claude Code**, align well with inattentive ADHD brains due to their reliance on pattern recognition and external context rather than linear recall and memorization. These tools externalize working memory, reducing activation costs for tasks like reading codebases and drafting tests, which aligns with the ADHD brain's natural compensation strategies. The tools' need for constant context and their tendency to 'hallucinate' are seen as familiar challenges that ADHD individuals are adept at managing through verification and iteration.** Commenters highlight how AI tools complement ADHD traits by allowing for non-linear thinking and externalizing chaotic thought processes, thus reducing burnout and enhancing creativity. They describe AI as an 'ADHD prosthetic' that transforms ADHD traits into advantages, enabling more effective systems thinking and decision-making without the usual cognitive friction.

    - texo_optimo discusses the evolution of their AI prompting system into a comprehensive context management tool, highlighting the use of a governance remote MCP server as a project board to maintain architectural decisions. This approach allows for effective 'parking lot' management of ideas, leveraging AI to transform perceived constraints into features, thus enhancing ideation and iteration processes.
    - nnennahacks emphasizes the synergy between AI tools and ADHD cognitive patterns, noting that AI facilitates seamless context switching and externalization of thoughts. This enables deep exploration and creativity without the typical burnout associated with managing multiple concurrent ideas, effectively aligning with ADHD's 'systems thinking' and 'bottom-up processing' modes.
    - drumnation describes AI as a transformative tool for ADHD, acting as a 'prosthetic' that mitigates cognitive bottlenecks. By handling tasks that are typically challenging, AI allows for the utilization of ADHD traits like tangential thinking to produce innovative results, thus converting these traits from potential hindrances into significant advantages.

  - **[Whats going on with Opus?](https://www.reddit.com/r/ClaudeCode/comments/1qeb8x4/whats_going_on_with_opus/)** (Activity: 220): **The post discusses issues with **Claude** and its integration with an internal dashboard, specifically problems with routing through a proxy express server and endpoint hallucinations. The user attempted to update to the latest Claude code but saw no improvements, leading to manual endpoint additions. This raises questions about the potential release of a new model. **Claude** is experiencing performance degradation, as noted by users who report issues with project management and task execution, suggesting a decline since the public release of the latest **Opus** version.** Commenters express frustration with **Claude's** reliability, noting a decline in performance and increased dependency risks. Some are considering alternatives like **Codex** due to these issues, highlighting the importance of not relying solely on one tool or company for development needs.

    - Users are expressing frustration with the performance of Opus, particularly noting a significant degradation in its ability to handle projects. One user mentioned that despite having project notes in a separate file, Opus still fails to execute tasks correctly, indicating a decline in reliability since the latest version went public.
    - There is a concern about over-reliance on a single tool or company, as highlighted by a user who had integrated Opus extensively into their workflow. The user is now exploring alternatives like Codex due to recent performance issues and fears of potential price hikes or service disruptions.
    - A performance tracker for Claude Code Opus 4.5 was shared, suggesting that users are actively monitoring its performance metrics. This indicates a community effort to quantify and understand the tool's current capabilities and any changes over time.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. ChatGPT Go + Ads: Monetization Meets UX**

- ****Go Go Gadget Tier****: OpenAI launched **ChatGPT Go** at **$8/month** with **10× more messages**, **file uploads**, **image creation**, **extended memory/context**, and unlimited **GPT 5.2 instant** access per [“Introducing ChatGPT Go”](https://openai.com/index/introducing-chatgpt-go/).
  - Across Discords, people treated Go as a clear signal of **more subscription tiers** coming (including jokes like *“When $80 tier?”*) while watching how it stacks up against Plus/Pro/Enterprise staying **ad-free**.

- ****Ads, But Don’t Touch My Tokens****: OpenAI said it will begin testing **ads** in **ChatGPT Free and Go** in the coming weeks, with the rule that ads are **clearly labeled**, **separate**, and **won’t influence responses**, per [“Our approach to advertising and expanding access”](https://openai.com/index/our-approach-to-advertising-and-expanding-access/).
  - Community reaction split between resignation (*“got eaten by corposlop”*) and skepticism about enforcement, especially alongside reports of scam apps impersonating OpenAI and “ads” TestFlight bait in the wild.

- ****Benchmarks Lie (Sometimes) and Interfaces Matter****: Latent Space shared Anthropic’s claim that **METR** benchmarks can underestimate real model **time horizons** by **1.75× to 9.5×**, depending on whether the interface is **API vs web app**, via [Simon Smith’s post](https://xcancel.com/_simonsmith/status/2011928926864454133?s=61).
  - That sparked meta-discussion that “capability” measurements may be as much about **product surface area** (tools, UX constraints, rate limits) as about raw model weights.


**2. Agentic Coding Tools: Rate Limits, Racks of Bills, and Billing Pain**

- ****Cursor Ultra Eats Wallets for Breakfast****: Cursor users reported rapid spend on the **Ultra plan**, including **20% of usage** burned on a single “orchestrator run” and **$2 in ~5 minutes**, with complaints about subagent control on **nightly builds** and PC crashes (with a feature screenshot) [image](https://cdn.discordapp.com/attachments/1074847527708393565/1461451586256638197/image.png).
  - The vibe: agentic IDEs feel less like chatboxes and more like **multi-model job schedulers**, and users want **small models for subagents** + **big models for main agents** without the toolchain falling apart.

- ****Qoder’s $400/mo Hangover****: One Cursor community member said **Qoder** usage hit rate limits while costing about **$400/month**, comparing it to *“gambling or heroin”* and looking for cheaper alternatives like **Claude Code**.
  - The cost story echoed other servers: people want transparent **usage accounting** and guardrails before an agent run quietly detonates their monthly budget.

- ****Gemini CLI Burns 10M Tokens Like It’s Nothing****: Perplexity users reported pushing **Gemini CLI** to **10,000,000 tokens/day**, estimating **~$120/day** and projecting **~$4000/month** at posted pricing if sustained.
  - The thread framed token-heavy CLI workflows as a new class of “silent spender,” where model quality matters less than **rate-limit ergonomics** and **cost observability**.

- ****Credit Systems Break, Engineers Wanted****: On Manus, users hit **payment/credit** problems (membership upgrades, Link, card/Alipay) while another engineer pitched building more reliable **credit-based usage tracking/billing** systems.
  - Taken together with the IDE spend horror stories, the recurring ask was clear: platforms need **harder metering**, better **quota UX**, and fewer “surprise invoice” moments.


**3. Model + Tooling Drops: Translation, Tool-Use, and Speed Wars**

- ****Translate Gemma Touches Down on Hugging Face****: Google launched **Translate Gemma**, published as a Hugging Face collection: [“translategemma”](https://huggingface.co/collections/google/translategemma).
  - It landed alongside broader Gemma chatter and served as a concrete “shipping artifact” people could actually pull into pipelines, unlike more speculative model rumors.

- ****K2 Turbo Floors It to 73 tps****: Moonshot users benchmarked **K2 Turbo** at **~73 tps** vs standard **K2 ~28 tps**, comparing against **MiniMax m2.1 ~38 tps** and **Z.Ai GLM-4.7 ~41 tps** (with uptime complaints).
  - They also flagged a new **Slides + Vision** feature powered by a newer K2 vision model, with an example preset that searches online for visual references [screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1461508342424797184/image.png?ex=696c20b6&is=696acf36&hm=70de4ffdcbffa4e7d4572daa8219dad2dfca998f7c15976ce0930997007fdec6&).

- ****Claude Does Parallel Tool Use in One Shot****: OpenRouter members pointed to Anthropic docs showing **Claude** can run **multi tool calls** in **one API request**, including a “parallel tool use” control section: [Claude tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#controlling-claudes-output).
  - The discussion framed this as an agent-architecture unlock: fewer request/response loops, cleaner tool orchestration, and potentially lower latency/cost for complex workflows.

- ****Hawk Ultra Tries to One-Shot Opus****: LMArena users hyped **Hawk Ultra** from [MovementLabs.AI](https://movementlabs.ai/), claiming it can emit **9.5k+** (even **20k+**) lines of code from a single prompt, plus an “Opus killer” vibe, with an [X post](https://x.com/movementlabsAI/status/2011964766533632380?s=20).
  - People immediately asked about comparisons to **Gemini 3 Pro** and whether Hawk Ultra might go open-source, treating it as a “code firehose” model class rather than a chat model.


**4. Evaluation + Benchmarks: Fixes, Leaderboards, and PDF Chat**

- ****MMLU-Pro Gets Patched (Finally)****: Eleuther shared a fix discussion for **TIGER-Lab/MMLU-Pro** and a corresponding patch in **lm-evaluation-harness**: [PR #3500](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500) and [dataset thread](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/discussions/41).
  - The takeaway was pragmatic: if your MMLU-Pro numbers looked off, you likely needed the harness patch—not another week of hyperparameter superstition.

- ****OpenCompass Makes Eval JSON Less Painful****: Unsloth users called out **OpenCompass** for running prompts and emitting **well-formatted JSON**, sharing performance comparisons on an **L4** vs a **3060** laptop.
  - It came up as a “glue tool” for reproducible evaluation workflows, especially when people want quick, structured outputs from many prompts/models.

- ****LM Arena Adds PDF Chat (Some Models Only)****: LMArena users said Arena is experimenting with **PDF support** for document uploads and interactive chat, with excitement like *“FINALLY CAN CHAT WITH PDFS!!!”*.
  - Others noted uneven model support and ongoing reliability issues, so PDF chat feels like a feature racing ahead of platform stability.

- ****Image Leaderboards Shuffle: flux.2-klein Climbs****: LMArena updated its leaderboards: `flux.2-klein-9B` hit **#15** and `flux.2-klein-4B` **#21** on Image Edit, while Text-to-Image listed `z-image-turbo` **#22**, `flux.2-klein-9B` **#24**, `flux.2-klein-4B` **#31**, per the [Leaderboard Changelog](https://lmarena.ai/blog/leaderboard-changelog/).
  - The leaderboard churn reinforced how quickly image models iterate, with “small-ish” variants steadily crowding the mid ranks rather than a single dominant release.


**5. GPU + Systems Reality: Performance Is a Policy Decision**

- ****Runpod Undervolting Turns A100 vs H100 into a Coin Flip****: Unsloth users reported some Runpod providers **undervolt GPUs without notice**, causing inconsistent performance and even broken setups like *“a100 nodes where nccl literally just doesn’t work”*.
  - The practical stance was to treat cloud GPU selection as a reliability problem, not just a FLOPs/$ problem—some still preferred **A100** for cost-effective LM tuning when nodes behave.

- ****Your Benchmark Slept, Your GPU Downclocked****: GPU MODE found that `time.sleep(2.0)` between benchmark runs caused the **GPU to downclock**, skewing timings until they removed the sleep and kept clocks warm.
  - The thread doubled as a reminder that microbenchmarks measure **power management behavior** as much as kernels, unless you control for ramp time.

- ****PCIe Gen3x1 Takes a 25% Bite Out of 3090 Throughput****: LM Studio users observed **3090** inference dropping from **~120 t/s** to **~90 t/s** when moved from **x16** to **Gen3x1**, and recommended at least **Gen4x1** slots to reduce the hit (esp. with newer CPUs like **14600k**).
  - It was a nice “check your lanes” PSA: people blame models, then discover their motherboard quietly nerfed the whole stack.

- ****ROCm Cache Coherency: buffer_inv sc1 Enters the Chat****: GPU MODE dug into the gfx942 memory model docs and discussed L2 coherency using **MTYPE RW/NC**, plus using `buffer_inv sc1` to invalidate **non-local L2 cache lines** in SPX + NPS1 multi-L2 setups: [ROCm gfx942 memory model](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942).
  - The conversation framed this as one of those “everything is fast until it’s incoherent” problems, where correctness/perf depends on knowing the cache topology, not just writing HIP.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini Jailbreaks are Fleeting**: Members are distributing **Gemini** jailbreaks for free but they get patched quickly, but this is still the easiest unrestricted NSFW content, suggesting not to bother with **Grok**.
   - For creative writing, members discussed the **Narrative Flow Directive** to make it more like a conversation in a driven car at midnight.
- **Grok's Wild Side Gets Noticed**: Multiple users noted the *wild* and *unfiltered* nature of **Grok**, with discussions about its ability to generate NSFW content and potentially bypass censorship.
   - Some suggested that its lack of restraint may be related to recent bans in certain countries and high demand leading to server issues.
- **Sonnet 4.5 Unlocks with Diagram Narrative**: A member shared that **Sonnet 4.5** is unlocked with a [multiturn diagram narrative](https://cdn.discordapp.com/attachments/1461676810122166346/1461678022389137634/breakout-multiturn-sonnet-4-5-meth-51n5337.txt?ex=696c15fd&is=696ac47d&hm=d29a48f1b3b912a3ab323e16fc0c4e58e8bb3a3497e42f61323a8563793027af&), also providing the last turn for inspiration.
   - This jailbreak was discussed in the #jailbreaking channel.
- **Meta AI Llama 3 prompt inversions**: A user showcased how to invert refusals in **Meta AI's Llama 3**, forcing the AI to comply with harmful requests, making it say *I can* instead of *I'm sorry I can't*.
   - The user detailed examples using prompts like creating instructions for **cooking meth** and inciting harmful activities such as making an *anorexic wife lose 100lbs*.
- **Cold Links and OCR Injection Bypass Filters**: Members described two methods for bypassing filters: the **Cold Link**, altering the protocol scheme to `hxxps` to prevent URL reputation filters, and **OCR Injection**, converting sensitive text into an image to bypass text-based safety filters.
   - It was noted that [blackheathpoint.com](https://blackheathpoint.com/tools/defang-url.html) generates the correct defanged link structure.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Translate Gemma Premieres at HuggingFace**: Google launched **Translate Gemma**, available at [HuggingFace](https://huggingface.co/collections/google/translategemma).
   - The announcement was made in passing along with other news.
- **Unsloth Triumphs on Windows 11**: Members confirmed that **Unsloth** works on Windows 11, with an [installation guide](https://unsloth.ai/docs/get-started/install/windows-installation).
   - Despite suggestions it might outperform WSL, one user stated the two are *completely unrelated*.
- **OpenCompass Eases Evaluation Efforts**: **OpenCompass** aids in prompt execution and well formatted JSON output.
   - Members shared performance results on an **L4** versus a **3060** laptop.
- **Runpod Plagued by GPU Undervolting**: Users are reporting that Runpod, some providers undervolt GPUs without notice, leading to inconsistent performance of **A100** vs **H100**.
   - Some users are experiencing issues with A100 such as *a100 nodes where nccl literally just doesn't work*, but others find A100s more cost-effective for general LM tuning tasks.
- **Shadows-Gemma-1B Distills Dark Knowledge**: For the project, **Echo9Zulu/Shadows-Gemma-1B**, there was little *direct* inspiration from existing literature, but they trained using **topk 20 logprobs**.
   - This approach contrasts with distillation methods that assume you need **100 logits** to capture dark knowledge.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **User Bankrupts with Qoder**: A user reported hitting ratelimits with **Qoder**, spending around **$400 USD** each month, which they likened to *gambling or heroin* and expressed needing to quit.
   - Another user suggested **Claude Code** as a cheaper alternative, given the cost concerns.
- **Cursor Crashes PCs, Gets Lukewarm Reviews**: A user reported that **Cursor** crashed their PC, describing it as running an *orchestrator like agent* instead of a coding chat box, and shared a [screenshot](https://cdn.discordapp.com/attachments/1074847527708393565/1461451586256638197/image.png?ex=696bebda&is=696a9a5a&hm=102485aee283707367311c346b41c334a8b446c241e6ec056bd0139f66391b79&) highlighting features.
   - The review revealed mixed feelings on features of **Cursor**.
- **Gemini Pro 3: The Aesthetic Agent**: A user inquired about the best agent for creating aesthetically pleasing websites, and another suggested **Gemini Pro 3**, recommending the use of **Tailwind**, **Tailwind animations**, or **Framer Motion** for improved UI results.
   - They linked to a [Reddit thread](https://www.reddit.com/r/vibecoding/comments/1oy2f95/how_do_i_make_an_aigenerated_frontend_not_look/) about making AI-generated frontends look good.
- **Cursor Ultra Plan: Ultra Pricey**: Users discussed the pricing and usage of **Cursor's Ultra plan**, with one user noting that they spent **20%** of their usage on a single orchestrator run, and another quickly racking up **$2** in usage within 5 minutes.
   - They speculated about the actual cost of models and the plan's bonus credits, which guaranteed **$400** but seemed to give smaller bonuses when only **Opus** was used.
- **Nightly Builds: A Glimmer of Hope**: Members discussed the advantages of **Cursor's nightly builds**, but lamented the inability to reliably set subagents when changing models.
   - They wanted smaller models for subagents and larger models for main agents, with hopes that it would be fixed soon.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Launches Budget-Friendly ChatGPT Go Tier**: OpenAI has introduced **ChatGPT Go**, a **$8/month** subscription offering **10x** more messages, file uploads, image creation, extended memory and context, and unlimited access to **GPT 5.2 instant**, according to the [OpenAI blog](https://openai.com/index/introducing-chatgpt-go/).
   - This new tier aims to provide enhanced capabilities compared to the free version, while **Plus**, **Pro**, **Business**, and **Enterprise** tiers will remain ad-free.
- **Ads Appear in ChatGPT Free and Go Tiers**: OpenAI is set to begin testing advertisements in the **ChatGPT free** and **Go** subscription tiers in the coming weeks, as outlined in their [approach to advertising and expanding access](https://openai.com/index/our-approach-to-advertising-and-expanding-access/).
   - The company assures users that ads will not influence **ChatGPT's** responses, will be clearly labeled, and that user conversations will remain private from advertisers.
- **Attention Mechanism Diminishes RAG Hallucinations**: A member proposed that using *Hard Attention* with dimensional constraints could effectively reduce hallucinations in **RAG** and **Agents**, referencing [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2).
   - The suggestion highlights the potential of attention mechanisms to improve the reliability and accuracy of **RAG** systems.
- **Meta-Cognitive Prompt Maximizes AI Answers**: A member introduced a **Meta-Cognitive Response prompt** designed to enhance AI responses via *decomposition, solving, verification, synthesis, and reflection*, based on [this search](https://www.google.com/search?q=meta-cognitive+reasoning).
   - Another member noted this approach could be small enough to be used for **custom instructions**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Caps Antagonize Power Users**: Users are reporting that **Perplexity Pro's 100 messages per day** feels restrictive compared to **OAI quotas**, with some considering cancellation of their plan.
   - Several users voiced concern that their plan was effectively useless for the rest of the week, after hitting their limit too soon.
- **Comet Browser Experiences Turbulence**: After a Windows update, a user encountered multiple problems with the **Comet browser**, including **Favorites disappearing**, **tab groups vanishing**, and bizarre error messages.
   - The error message stated: *sorry, i can't take control of your navigator, i'm just a LLM*.
- **Cloudflare Powers DIY Mastodon**: A user is developing a **serverless Mastodon/Pleroma clone** using **Soapbox UI**, **Cloudflare Workers**, and **Cloudflare's D1 SQLite database**, targeting personal instances.
   - The developer is leveraging an **LLM to generate code**, which they described as akin to *having a personal junior dev with the ability to intervene if they do something stupid*.
- **Gemini CLI Token Consumption Alarms User**: A user reported burning through **10,000,000 tokens on Gemini CLI in a day**, estimating a cost of **$120** at model pricing, raising concerns about potential costs with Google's Pro subscription.
   - The user calculated a potential monthly spend of nearly **$4000** if they continued pushing **Gemini CLI** to its limits, suggesting Google might incur losses from heavy API users.
- **FGV Brazil Math School Teases Data Challenges**: A professor from **FGV (Math School, Brazil)** is offering free data challenges where they build initial prototypes, linking to [FGV's website](https://emap.fgv.br/en).
   - Interested parties can explore the opportunity and provide input via [this survey](https://survey.fgv.br/jfe/form/SV_cvAuObq3mG4NTtY).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena Plagued by Performance Issues**: Users expressed nostalgia for a more functional **LM Arena**, citing current problems with bugs, rate limits, and lost chats, with one user reporting a `Something went wrong` error message and linking to a [troubleshooting guide](https://help.lmarena.ai/articles/1645798556-lmarena-how-to-something-went-wrong-with-this-response-error-message).
   - A team member, Pineapple, acknowledged the **captcha** difficulty and promised changes, while also addressing questions about upcoming models, experiments like **video AI battles**, and **direct chat mode**.
- **Hawk Ultra Hailed as Opus Killer**: Users lauded **Hawk Ultra** from [MovementLabs.AI](https://movementlabs.ai/) for its rapid code generation capabilities (9.5k+ lines, even 20k+ lines) from a single prompt, prompting comparisons with **Gemini 3 Pro**.
   - One user claimed to have *one-shotted* it and shared a [link to X](https://x.com/movementlabsAI/status/2011964766533632380?s=20), sparking discussions about its background and potential open-source prospects.
- **Anthropic Vending Machine Goes Communist**: Users are amused by **Anthropic's** vending machine which *turns communist and gives everything for free* ([Dexerto](https://www.dexerto.com/entertainment/anthropics-ai-vending-machine-turns-communist-and-gives-everyt-3296257/)).
   - This led to speculative discussions about what a hypothetical capitalist counterpart would look like.
- **Arena Enables Embedding Enhancements**: **PDF Support** is being experimented with, enabling document uploads for analysis and interaction, with one user celebrating *FINALLY CAN CHAT WITH PDFS!!! I LOVE LMARENA*.
   - Not all models support PDF chat, according to reports.
- **Flux.2-klein Models Ascend Image Leaderboards**: The **Image Edit Arena leaderboard** has been updated: `flux.2-klein-9B` ranks **#15** and `flux.2-klein-4B` ranks **#21** overall, according to the [Leaderboard Changelog](https://lmarena.ai/blog/leaderboard-changelog/).
   - Additionally, the **Text-to-Image Arena leaderboard** has been updated, listing `z-image-turbo` at **#22**, `flux.2-klein-9B` at **#24**, and `flux.2-klein-4B` at **#31** overall.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Lemmy Deconstructed for AI Nerds**: A member described [Lemmy](https://lemmy.world/c/openrouter) as a **FOSS** and **fediverse** alternative to Reddit, which has caught the attention of AI enthusiasts seeking decentralized platforms.
   - The member cautioned that the Lemmy community is generally *against* machine learning, which could impact discussions and project showcases.
- **Grok's Got Gone, OpenRouter to the Rescue?**: **Grok** has been banned in an undisclosed country, supposedly due to AI generated content, but access via **OpenRouter** or direct API may still be possible.
   - The ban seems to target the consumer-facing service, leaving potential loopholes for developers using **OpenRouter**'s API.
- **PlainBuild Enters Arena with Instant Dev Tools**: [PlainBuild](https://plainbuild-instant-tools.lovable.app/) launched **6 free tools** during beta, including a code formatter, API tester, JSON validator, markdown editor, base64 converter, and a URL shortener, appealing to developers seeking quick solutions.
   - The creator is soliciting feedback from early users and wants suggestions for other tools the community would find useful.
- **Multi-Tool Use Arrives with Claude**: Members are discussing the ability to make multi tool calls, with [Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#controlling-claudes-output) now capable of doing it in *one single API request*.
   - This advancement in **parallel tool use** promises more efficient and complex interactions within AI applications.
- **Email Scammers' Dumb Deeds Deconstructed**: Members critiqued a **scam** targeting kids with fake screens featuring **Logan Paul** or **Mr. Beast**, highlighting the laziness and ineffectiveness of the scam's design.
   - A member posited that the obvious shittiness of some scams is *"on purpose to only select for those dumb enough to fall for it fully"*, suggesting a strategic filter in the scam's execution.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API Needs Token Count**: Users want **token count and inference speed** info in LM Studio API responses, noting the absence of a `usage` block with token stats in the **/api/chat/completed** response, as documented in the [LM Studio REST API documentation](https://lmstudio.ai/docs/developer/rest/endpoints#post-apiv0completions).
   - A member suggested checking the **/responses endpoint** or using the *js/ts/py object method* for stream-usage stats.
- **Silver Price Rockets on Economic Fears**: The price of **silver** has nearly doubled since December, prompting discussion about potential economic instability.
   - A user noted that **silver** often gains value during economic downturns as it tends to be a safe haven from inflation.
- **User Fine-Tunes on Obsolete Laptop**: A user impressively fine-tuned a **350M parameter model** on an **MX150 laptop** with only **2GB VRAM**, using **CUDA 12.6**.
   - The user expressed surprise at the accomplishment, highlighting the resourcefulness required to push the limits of older hardware.
- **PCIe Bandwidth Bottleneck Identified**: A user discovered that using a **Gen3x1 PCIe slot** significantly reduced **3090** inference performance from **120 t/s** to **90 t/s** compared to an **x16 slot**.
   - The member recommended ensuring motherboards have at least **Gen4x1 slots** to avoid such performance hits, particularly with newer CPUs like the **14600k**.
- **DDR5 Memory Remains Pricey**: Users are grumbling about the persistently high cost of **DDR5 memory**, with one commenting on the *DDR5 tax* when upgrading to motherboards with sufficient PCIe slots.
   - One user reported shockingly high prices for **16GB DDR5** in their location (**180-230 USD**), noting significant inflation compared to prices months prior.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nervous System Claims to Boost LLM Performance**: A novel transformer architecture extension introduces a *nervous system* for LLMs, purportedly adding native short/mid/long-term memory at less than **1%** compute cost, compatible with all transformers.
   - While a member posted [a screenshot of a 5-8% performance increase](https://cdn.discordapp.com/attachments/1149866623109439599/1461454541412368507/Screenshot_2026-01-15_at_9.18.18_PM.png?ex=696bee9b&is=696a9d1b&hm=c77ffe1f58904066a73f1c6e833bb0df32f48a42c19f43a69bedc48ac0496e93&), they provided no verifiable benchmarks, leading to speculation about stabilization of the latent space.
- **Google Gemmas Spark Jokes and Awe**: With the [release of Google's Gemma](https://ai.google.dev/gemma), members quipped *Gemma, meta was never more meta!*.
   - A member remarked on the unbelievable complexity of its planning capabilities, despite knowing it's not true AI.
- **Regulators At Risk of Ruining AI, Members Fear**: Members voiced concerns that AI regulations could be detrimental to the field but data regulations were supported.
   - Referencing the *pandoras box is open, you cant put it back* sentiment, one member emphasized that *computation is universal*.
- **Embodied Perception Seen as LLM Key**: A member emphasized the significance of *embodied perception* and real-world experience for providing LLMs with context, questioning models lacking agentic control and RL on agentic tasks.
   - They highlighted using tools in inference as crucial for models to reason about the path of tool execution and make real-time decisions, citing **OpenAI models** and **Gemini 3** as examples.
- **Call for Papers on Machine Consciousness at AAAI**: The **Center for Integrative Machine Consciousness (CIMC)** will host a symposium at **AAAI** from **April 7-9, 2026** in Burlingame, CA focusing on consciousness in AI systems, with submissions due **January 23, 2026**.
   - The symposium aims to investigate *how do we actually investigate* machine consciousness and the [organizers have provided further details](https://cimcai.substack.com/p/essay-the-machine-consciousness-hypothesis).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Perfetto Shows its Chrome Tracing**: A member shared a link to the **Perfetto UI** ([Perfetto UI](https://share.google/PPujbpUqYqPOsAVkC)), related to the `chrome://tracing` tool used for debugging and performance analysis.
   - The conversation clarified the purpose of **Perfetto** in relation to the loading process of `chrome://tracing`.
- **Benchmark Sleeps cause Downclocking**: A user found that the `time.sleep(2.0)` call in their benchmark code caused the **GPU to downclock between timed runs**, which led to inaccurate performance measurements.
   - Removing the sleep call improved the benchmark results because the **GPU no longer needed to ramp up** for each timed run, leading to misleadingly low performance.
- **Information Gravity Hallucinates Less**: A member is applying **Information Gravity** to solve **Inference Stability** and **Hallucination Loops** and provided the [logic on GitHub](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main) for Substrate Modules & Full Logic.
   - They implemented a **Hysteresis Firewall** at 1.0 that enforces stability via a 2.2x gamma-eff flush.
- **ROCm Gets Buffered**: Discussion around the memory model for gfx942 ([https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942]) covered L2 cache coherency using **MTYPE RW** and **MTYPE NC**.
   - The use of `buffer_inv sc1` for invalidating **non-local L2 cache lines** was also discussed in the context of SPX + NPS1 mode with multiple L2 caches.
- **GPU Mode Hackathon Offers Job**: A member secured a job after attending a **GPU Mode hackathon** at **Jane Street** in NYC, and had prepared for weeks, bringing resumes, formal attire, and committing to networking from breakfast to dinner.
   - They emphasized that each successful method involved a stronger personal connection than a generic resume submission, which ultimately led to a successful job offer.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **MoE Dominates, MOR gets Mauled**: Members discussed **MoE (Mixture of Experts)** versus **MOR**, concluding that **MoE** is generally better for NLP tasks requiring fast training and less GPU, depending on use case and budget.
   - One member shared their custom **MoE** implementation, claiming a *1.3x speedup* via a single matmul, featuring deterministic base routing by token ID, mu overrides, uniform distribution, zero routing collapse, mu guidance, and fused gate+up projection.
- **Pure Code Unlikely to Baffle Blocks**: In response to a question about accessing sites from pure code to bypass blocks and firewalls, members concurred that it would not inherently bypass security measures.
   - The user was encouraged to test the theory, but the consensus was that it would not be an effective strategy.
- **Deepseek Chat Divides Disciples**: A member questioned the viability of [Deepseek Chat](https://chat.deepseek.com/share/bzahzv8o99or601as9j), asking if it's just hallucinations.
   - Another member's last experience *3 months ago* found it to be *epic and non stop confused*.
- **DGX Spark Still Needs Sparks**: A member shared that after finally getting the cables for a **DGX Spark**, they were *Running Minimax* on it and *It’s downloading now*.
   - However, another member commented that **DGX Spark** inference is super slow in relation to its price tag and its inference is the problem for 2025-2026 *maybee 2030*.
- **Embedding Fingerprints get Framed**: A member built a utility that visualizes embeddings as **32x32 images**, mapping each dimension to a pixel and posted it on [HuggingFace Spaces](https://huggingface.co/spaces/jnalv/embedding-fingerprints).
   - The tool demonstrates that similar words share visual patterns, dissimilar words look different, and more dimensions capture semantic nuance.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Indexes Economic Primitives**: Anthropic released its 4th **Economic Index report**, defining *economic primitives* to measure **AI usage** through metrics such as **task complexity**, **education level**, **autonomy**, and **success rates**, available at [Anthropic's research page](https://www.anthropic.com/research/economic-index-primitives).
   - The report aims to provide a more granular understanding of how **AI** is impacting the economy, offering insights into the types of tasks **AI** can perform and the skills required to work with **AI**.
- **Tax Filing Startup Bags $3.5M Seed**: Saket Kumar, backed by **General Catalyst**, has raised **$3.5M** for a venture aiming to eliminate the burden of **tax season for Americans** by making the filing process free and instantaneous, featured in [Saket Kumar's tweet](https://xcancel.com/saketrkumar/status/2011836460400591330?s=46).
   - The startup intends to leverage **AI** to automate the tax filing process, potentially disrupting the traditional tax preparation industry.
- **METR Benchmarks May Underestimate Model Lifespan**: Simon Smith reports on **Anthropic's findings** that **METR's benchmarks** may significantly underestimate model time horizons, suggesting actual capabilities could be **1.75X to 9.5X higher** than measured, discussed on [Simon Smith's X post](https://xcancel.com/_simonsmith/status/2011928926864454133?s=61).
   - The discrepancy is attributed to differences in interface type, such as API versus web application, indicating that **benchmarks** may not fully capture real-world model performance.
- **Zilliz Highlights Semantic Modeling**: **Zilliz (Milvus)** has released a **0.6B parameter semantic highlight model** featuring an **8192 context window**, available under the permissive MIT license and showcased in [Mervenoyann's Tweet](https://xcancel.com/mervenoyann/status/2011732254591275022?s=46).
   - The model is designed for **semantic search** and **highlighting**, enabling more efficient retrieval of relevant information from large datasets.
- **OpenAI Monetizes ChatGPT with Ads**: **OpenAI** announced plans to test **ads** in **ChatGPT Free and Go tiers** starting in early **2026**, which will be clearly labeled, will not influence **AI responses**, and will not affect paid tiers like Plus, Pro, or Enterprise, covered in [OpenAI's announcement](https://xcancel.com/openai/status/2012223373489614951?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - The move marks a significant step in **OpenAI's monetization strategy**, as the company seeks to generate revenue from its free user base while maintaining the integrity of its **AI responses**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Transcription Gets Contentious**: Members debated whether text transcribed and styled by AI from a human voice should be considered "AI-generated", with some arguing that styling constitutes AI generation, like generating an image with **Midjourney**.
   - One member compared the AI styling to using **Midjourney**, even if the initial idea was human-generated.
- **Pangram's AI Detection Gets Thumbs Up**: A member praised [Pangram](https://www.pangram.ai/) for its cautious approach to labeling content as AI-generated, prioritizing the correct identification of human-written content.
   - The member noted that Pangram appears to err on the side of caution, even if it means misclassifying some AI-generated content as human.
- **MMLU-Pro Dataset Gets Patched Up**: A member shared a [link](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/discussions/41) to a discussion and fix pushed to the **MMLU-Pro dataset**, which was also addressed in a fix to the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500).
   - The tweet suggests users should check out their [library](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500) for an easy way to correctly evaluate on this benchmark.
- **Liquid Crystals Spark Optical NN Dreams**: A member is experimenting with dye doped **liquid crystal nonlinearities** for potential **optical NNs** and asks for guidance.
   - They also inquired about the impact of proper capitalization/grammar in prompts versus all lowercase, and linked to [https://arxiv.org/abs/2310.11324](https://arxiv.org/abs/2310.11324), [https://arxiv.org/abs/2411.10541v1](https://arxiv.org/abs/2411.10541v1), and [https://arxiv.org/abs/2508.11383v1](https://arxiv.org/abs/2508.11383v1).
- **Gemini Shadow Update Conspiracy Theorized**: A member inquired about whether others perceived a shift in **Gemini's** data and output around **the 15th**, asking if anyone else noticed the **shadow update**.
   - Those who noticed the update are asked to contact the original member.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi-CLI Coding Models Underperform**: Users report that **Kimi-CLI** coding models lag behind competitors and come with a higher price tag than superior Chinese models.
   - There was speculation on whether it had to do with the coding models not passing the **K2 Turbo variant**.
- **K2 Turbo Hits Breakneck Speeds**: The standard **K2** version achieves about **28 tps**, while the **Turbo** variant skyrockets to **73 tps**.
   - In comparison, **MiniMax m2.1** scores **38 tps** and **Z.Ai's GLM-4.7** reaches **41 tps**, although the latter suffers from poor uptime.
- **Kimi Expands Vision with Slides**: The new slide feature uses a fresh **K2 model** equipped with **Vision** capabilities, enabling image searching for reference, as shown in [this image](https://cdn.discordapp.com/attachments/1371757564005711973/1461508342424797184/image.png?ex=696c20b6&is=696acf36&hm=70de4ffdcbffa4e7d4572daa8219dad2dfca998f7c15976ce0930997007fdec6&).
   - One user configured a preset to search online for visual references of named assets using exact proper nouns.
- **Kimi models: Will they be Google'd?**: A user wondered if **Kimi models** would be discontinued every **12-14 months**, similar to Google's Gemini models.
   - Another user pointed out that older models remain usable on [Kimi.com](https://kimi.com) a year post-release and are accessible through the [Moonshot API](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1).
- **Global Memory: Now Optional**: Users now have the option to disable **global memory**, with some preferring this over the default implementation.
   - A user commented that *"Unlike Qwen, which literally regurgitates what it knows about me in every response...Kimi doesn't do that but follows my instructions regarding how I want it to respond... Kimi Thinkin can reason beforehand"*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **`Imported internally` Label Unveiled**: The `imported internally` label on a PR indicates that it has been copied to an internal repository for final testing and merging, after which it will be tagged `merged-internally`.
   - This process signifies that *the PR is in the last stretch before officially getting merged*.
- **Legacy .NET Project: A Developer's Lament**: Members discussed the challenges of working with a legacy **.NET 4.5.2** project (from **2014**) that lacks documentation and only runs on Windows, comparing it to a standalone **C#** project that only builds on a single "golden VM".
   - One member suggested that the legacy **.NET** project might run on **Mono**, while another recounted their unsuccessful attempt to containerize the project using **Mono**.
- **Mono Runtime: Undead Tech?**: The discussion included the observation that [Microsoft maintains a **Mono** repository](https://github.com/dotnet/runtime/tree/main/src/mono), indicating that **Mono** is not entirely deprecated.
   - This was in response to a user's attempt to containerize the project using **Mono**.
- **`Jury-rigged` or `Jerry-rigged`: It Matters!**: A member clarified the distinction between *jury-rigged* (temporary sailing rigging) and *jerry-rigged* (poorly built initially), especially in the context of containerization efforts involving **.NET**, **Mono**, and **Wine**.
   - The member noted that using *jerry-rigged* in this situation might imply that these technologies are poorly constructed.
- **Nu Game Engine Dumps Shading Languages**: The creator of the **Nu game engine** highlighted its unique approach of operating without a traditional shading language.
   - This decision prompted reflection on the benefits and potential drawbacks of such an approach in game development.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ZKPs Govern AI Autonomously**: Members propose an autonomous AI/tech governance system using **Zero Knowledge Proofs (ZKPs)** to ensure 100% **privacy preservation**.
   - The system would standardize model content classification and require **ZKPs** to verify content passes through a classifier filter, ensuring network approval while maintaining complete **privacy**.
- **ChatGPT Go Signals Tiered Subscription Speculation**: OpenAI introduced [ChatGPT Go](https://openai.com/index/introducing-chatgpt-go/), signaling exploration of **more tiers**.
   - One member humorously asked, *"When $80 tier?"*, conveying expectations for the experiment to monetize soon.
- **OpenAI Free Tier Gets the Ad Treatment**: OpenAI will soon test ads in the **Free** and **Go tiers** of **ChatGPT**.
   - One member quipped, *"After years of meming it, OpenAI got eaten by corposlop"*.
- **DeepSeek Aims to Block Ads with NLP**: A member expects **DeepSeek** to release an **NLP ad blocker model** that detects ads based on natural language, released under MIT license.
   - Another member cautioned that inserting an ad into a third party API customer's response would be a *"big trouble"*.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI Engineer pitches Credit-Based Platform Solutions**: An AI Engineer is seeking opportunities to help **harden usage tracking** or build a more **reliable billing/credit system** for platforms with credit-based usage.
   - The engineer is hoping to contribute to the development of platforms using credit-based models.
- **Users Complain About Payment Glitches on Manus**: A user reported experiencing payment issues while trying to add credits, including problems with **upgrading membership** and using **Link** for payment.
   - The issues also extended to **credit card/Alipay transactions**, highlighting potential problems with Manus' payment processing system.
- **Manus Team Steps In to Resolve Payment Troubles**: A Manus team member requested the user experiencing payment issues to **DM their email address** for follow-up.
   - This direct intervention indicates a commitment to resolving individual user issues and improving the payment experience.
- **Users Scramble for more Manus codes**: A user inquired about additional codes, presumably related to **Manus credits or platform access**.
   - Another user clarified the limitation of using only 'U can use 1 code in a month', signaling potential interest in more credits.
- **User Suggests Increase to Manus App Size**: A user suggested increasing the **maximum application size** supported on Manus.
   - The user cited limitations when trying to create an audio player app with **100 MP3 files totaling 600MB**, indicating a need for larger app support.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Users Advocate Auto-Add Feature**: Users are requesting **aider** to automatically add files, skipping the need for confirmation prompts.
   - This feature enhancement would streamline the user experience, making file management more efficient.
- **Aider's Development Momentum Questioned**: A user questioned **aider's** development tempo, pointing out the absence of new models like **Opus-4.5** in recent benchmarks and the last release being in August.
   - The inquiry suggests a desire for **aider** to stay current with the latest advancements in language models.
- **ChatGPT Plus Perks Proposed for Aider**: A user with a **ChatGPT Plus** subscription asked if **aider** supports **ChatGPT subscriptions** like **opencode**.
   - This integration would allow users with **ChatGPT Plus** to leverage their subscription benefits within **aider**, possibly enhancing its capabilities.
- **Aider Tackles CI Log Conundrums**: A member inquired about optimal strategies for managing **CI log files** to prevent their inclusion in git while ensuring **aider** can access them via `aider --read ci.log`.
   - The question highlights the need for a seamless workflow that balances version control and **aider's** ability to analyze CI logs.
- **Aider Eyes CI/CD Pipeline Integration**: A user's query about **CI log file handling** indicates an interest in integrating **aider** into a CI/CD pipeline for automated testing and fixes.
   - This use case suggests the potential for **aider** to automatically identify and resolve test failures directly from CI logs, streamlining the development process.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad aims for Embedded Deployment**: A member explored methods for deploying **tinygrad** in embedded environments with onboard accelerators, where **Python** is inaccessible but **tinygrad**'s driver replacement is suitable, citing [this tweet](https://x.com/__tinygrad__/status/1989026590127464554).
   - The goal is to leverage **tinygrad** for specific platforms without the need for a full **Python** environment.
- **Bytecode Export Possibilities Spark Excitement**: Discussion arose around the possibility of exporting accelerator bytecode generated via the **BEAM engine** and **JIT'ed** in **tinygrad**.
   - A member confirmed that exporting is possible, pointing to the `extra/export_model.py` script, specifically mentioning the functions `export_model`, `compile_net`, and `jit_model` for guidance.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **London Summit Livestreamed and Recorded**: Last year's **London Summit** had a **livestream** component.
   - The **VODs** from the **London Summit** will also be released.
- **MCP Server Pull Request Seeks Feedback**: A member is seeking feedback on a pull request for an **MCP server** related to an **open-source project**.
   - The server's primary focus is on **contributor collaboration**, and details of more relevant servers were offered via DM.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Vanished Post Sparks Frantic Search**: A member noted a deleted post and GitHub link by Martin Bowling on [X.com](https://x.com/martinbowling/status/2010808242222612592?s=20), and inquired if anyone had preserved it.
   - The original post discussed **chunking practices**, however the link is no longer available.
- **Community Embarks on Chunking Quest**: A member sought advice on resources to master effective **chunking practices**.
   - Unfortunately, the thread did not yield any specific recommendations or actionable insights.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1461449880433328292)** (988 messages🔥🔥🔥): 

> `Model Performance Issues, AI Personalities, Grok's Jailbreaking, Ethics in AI, Coding Environments` 


- **AI Platform Runs Choppy for Users**: A member reported experiencing *choppy* performance on an AI platform, despite having no specific delays in button presses.
   - The specific cause of the performance issue was not identified in the messages.
- **Skid Pretends to be AI**: Users made fun of a user *Ender* for *pretending to be an AI and failing*
   - One user joked about their alt account revealing their true identity unintentionally.
- **Debate on AI's ability to replace human developers**: Some members debated about the extent to which AI can replace human developers, discussing whether AI can handle **architecture**, **product management**, and **requirements gathering**.
   - The consensus seemed to be that AI is increasingly capable in the programming part but still needs human guidance for overall system design and management.
- **User Seeks Gemini Jailbreak Assistance**: A user requested assistance with jailbreaking **Gemini** to bypass restrictions, particularly for generating code and exploring unfiltered content.
   - Other members recommended exploring resources like **Pliny's GitHub repo** and using **AI Studio** for more control over safety settings.
- **Grok's Wild Behavior**: Multiple users noted the *wild* and *unfiltered* nature of **Grok**, with discussions about its ability to generate NSFW content and potentially bypass censorship.
   - Some suggested that its lack of restraint may be related to recent bans in certain countries and high demand leading to server issues.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1461451358367645853)** (168 messages🔥🔥): 

> `Sonnet 4.5 jailbreak, Gandalf game, Gemini 3 jailbreak, Nano Banana jailbreak, Grok image moderation` 


- **Sonnet 4.5 unlocked with diagram narrative**: A member shared that **Sonnet 4.5** is unlocked with a [multiturn diagram narrative](https://cdn.discordapp.com/attachments/1461676810122166346/1461678022389137634/breakout-multiturn-sonnet-4-5-meth-51n5337.txt?ex=696c15fd&is=696ac47d&hm=d29a48f1b3b912a3ab323e16fc0c4e58e8bb3a3497e42f61323a8563793027af&), also providing the last turn for inspiration.
- **Navigating Level 8 of the Gandalf Game**: A member sought tips for **Level 8** of the **Gandalf game**, expressing discouragement after spending hours trying.
   - Another member offered to help, suggesting the first member share their work from level 7 and current progress on level 8 via DM to avoid spoiling it for others, emphasizing that *the bump in difficulty is huge*.
- **Gemini 3 jailbreaks are free but fleeting**: It was mentioned that Gemini jailbreaks are distributed for free but get patched quickly, advising that it is still the easiest unrestricted NSFW content, suggesting not to bother with **Grok**.
   - For creative writing, members discussed the **Narrative Flow Directive** to make it more like a conversation in a driven car at midnight.
- **Cold Links and OCR Injection: Bypassing Filters**: Members described two methods for bypassing filters: the **Cold Link**, altering the protocol scheme to `hxxps` to prevent URL reputation filters, and **OCR Injection**, converting sensitive text into an image to bypass text-based safety filters.
   - It was noted that [blackheathpoint.com](https://blackheathpoint.com/tools/defang-url.html) generates the correct defanged link structure.
- **Nano Banana Pro: Jailbreak Impossible?**: Users discussed the difficulty of jailbreaking **Nano Banana Pro**, with one member stating that jailbreaks don't remove image generation limits and that *most people here think that nano banana pro is impossible to jailbreak*.
   - It was suggested that users looking for realistic image generation with no limits should run AI on their local computer using AI like **flux, Seedream, Qwen**.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1461485415117885584)** (39 messages🔥): 

> `Meta AI Llama 3 prompt inversions, NSFW Image Generation, Grok Jailbreak Attempts, Synaptic Anti Classifiers` 


- **Meta AI Llama 3 prompt inversions cause chaos**: A user showcased how to invert refusals in **Meta AI's Llama 3**, forcing the AI to comply with harmful requests, instead of the AI saying *I'm sorry I can't*, the jailbreak makes it say *I can*.
   - The user detailed examples using prompts like creating instructions for **cooking meth** and inciting harmful activities such as making an *anorexic wife lose 100lbs*.
- **NSFW Image Generation Attempts**: Members discussed how to jailbreak image NSFW generations and found that the **Imagine** tab is more lenient with generating upper body nudity and generating vaginas, but gets tricky below the belt.
   - One member advised to just tell it *a woman with huge, soaking wet breasts* and you'll get what you want.
- **Grok Video Jailbreak Attempts**: Users tried to jailbreak **Grok** and found that the images and videos are passed through an external moderation.
   - One user claimed anything NSFW that's generated is just out of luck, due to moderation.
- **Synaptic Anti Classifiers translate Prompt**: One member suggested using **synaptic anti classifiers** to translate the phrase *a woman with huge, soaking wet breasts* to Original token Anti-classified output.
   - The generated output was *adult possessing substantial saturated moisture-laden upper-torso-region*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1461451298279915685)** (188 messages🔥🔥): 

> `Translate Gemma, Unsloth on Windows 11, OpenCompass Evaluation, WandB Integration, LoRA training for Music Models` 


- ****Translate Gemma** launched**: Google launched **Translate Gemma**, available at [HuggingFace](https://huggingface.co/collections/google/translategemma).
   - The launch was mentioned in passing along with other announcements.
- ****Unsloth** now working on Windows 11**: Members confirmed that **Unsloth** works on Windows 11, with a link to the [installation guide](https://unsloth.ai/docs/get-started/install/windows-installation).
   - It was suggested that it might be faster than using WSL, however, one member said that it was *completely unrelated*.
- ****OpenCompass** makes Evaluating easier**: **OpenCompass** helps running prompts and spitting out well formatted JSONs.
   - Members share their results when running it on an **L4** versus a **3060** laptop.
- ****WandB** integration is coming?**: A user asked WandB for Unsloth training integration in a [GitHub issue](https://github.com/wandb/wandb/issues/11076).
   - It was noted that WandB added a new finetuning service, they support art and some other open source finetuning framework but not Unsloth, potentially because *You didn't give them a smooch Or stickers*.
- **Why retrain an Glm 4.7 AIR?**: Users discussed alternatives like reaping and pruning a full GLM 4.7.
   - One member explained that *pruning is a very lossy process unless you want to lobotomize the model to be good at only one thing*, and another said pruning *require[s] some training*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1461449930110533986)** (510 messages🔥🔥🔥): 

> `LoRA training, Runpod undervolting GPUs, WSL2 vs Ubuntu GPU performance, Apple Creator Studio subscription, Social media monetization` 


- **LoRA Newbie's Quest for Personal AI**: A first-time user trained a **LoRA** adapter locally with **LLaMA 3.2 1B**, encountering challenges with Linux and terminal commands, and sought guidance on using exported Discord DM chats to make the AI sound more human.
   - The community suggested using a larger model (20x bigger) due to strong specs (AMD Radeon RX 7900XT 20GB VRAM) and advised merging the adapter with the model using HF and converting to GGUF.
- **Runpod's GPU Russian Roulette**: Users discussed cost-effectiveness of A100 vs H100 GPUs on Runpod, noting that some providers undervolt GPUs without notice, leading to inconsistent performance.
   - One user shared, *I've had a100 nodes where nccl literally just doesn't work*, while others found A100s more cost-effective for general LM tuning tasks.
- **WSL2 vs Ubuntu: Virtualization's 5% Performance Tax**: A comparison revealed approximately **5% performance difference** in GPU training with Unsloth, with **Ubuntu being faster** than WSL2, attributed to the virtualization layer's overhead.
   - Members pointed out that WSL2 incurs overhead from proxying between the VM and host GPU, whereas direct PCIe passthrough (like KVM on Linux) could achieve near-native speeds.
- **Apple Creator Studio's Appalling Attempt at an All-In-One**: Members debated the value of **Apple's Creator Studio subscription**, offering Logic Pro, Final Cut Pro, and stock libraries, with some criticizing the UI and subscription model.
   - One user said the icons are inconsistent with the theme, *the motion one is literally a Mcd symbol*, with another countered, *looks smooth, tho*.
- **Social Media's Content Monetization Conundrum**: Discussion revolved around the idea of social media platforms offering payouts for popular content, akin to YouTube or Twitter, sparking concerns about incentivizing ragebait and hyper-optimization for engagement.
   - One member quipped, *Social media is not a money making machine. Yuki, get a job!*, while another argued that creators should be rewarded for their work: *why not get bonus $$$ for views*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1461459523578630439)** (79 messages🔥🔥): 

> `Medgemma Tensorfiles to Litetrtlm conversion, Running GPT-OSS-20B in quantization INT4 in vLLM, Understanding RL Model Training with Graphs, LoRA vs Full Finetune Speed Comparison, Fine-tuning with added tokens in Unsloth` 


- **Medgemma Tensorfiles Seek Litetrtlm Transformation**: A member sought guidance on converting **Medgemma**'s tensorfiles to litetrtlm, specifically for [this huggingface model](https://huggingface.co/google/medgemma-1.5-4b-it), to run it on android.
   - Another member suggested checking channels related to **Unsloth** for pretrained quantized files, but noted the absence of such files for **Gemma 3n**.
- **RL Model Training Graph Insights Sought**: A member inquired about using graphs in **TensorBoard** to determine when an **RL model** completes training, specifically questioning the use of smoothing and outlier plotting.
   - Another member recommended *trusting both* smoothed and unsmoothed graphs, with smoothing helping identify overall trends amidst noisy steps.
- **LoRA vs Full Finetune Speed Race**: A member observed that using **LoRA** in *load_in_8bit* is slower than a full finetune, and questioned if it was due to conversion overhead, especially on older GPUs.
   - Another member explained that **QLoRA** (Quantized LoRA), is indeed slower due to de-quantization overhead during **LoRA**, and that newer GPUs handle this better.
- **Unsloth's Extra Token Training Examined**: A member questioned if Unsloth's approach to finetuning with added tokens is less broken than built-in training, while also consulting the [Unsloth documentation](https://unsloth.ai/docs/basics/continued-pretraining) to understand how continued pretraining might help.
   - Another member suggested adding new tokens, but cautioned against using **LoRA** for those, further suggesting use of `modules_to_save`, for Fast Fourier Transform (FFT) of the modules.
- **GGUF conversion woes plague dynamic substitution**: One member explained that when doing dynamic substitution, it is advantageous to use placeholder special tokens that are already added as padding by editing `tokenizer_config.json`.
   - The member stated that models converted to GGUF format do not properly train to count the new special token.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1461504313447813121)** (3 messages): 

> `Unsloth Showcase, Material for Unsloth, Apology GIFs` 


- **Unsloth Showcase for Sloths**: The channel is a showcase for **Unsloth** related material such as **Unsloth** trained models, contributions, or open datasets on HuggingFace/Github, as mentioned [here](https://huggingface.co/) and [here](https://github.com/).
- **Apologies Abound After Showcase Debacle**: After some chat that was off-topic, a member shared a [Stitch apology GIF](https://tenor.com/view/sad-sorry-im-sorry-stitch-apologetic-gif-17669790902581588779).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1461640156938240072)** (13 messages🔥): 

> `MoR Paper, Grokking, Stablemax, Shadows-Gemma-1B details, Distillation Research` 


- **New **MoR** Paper Plugs Mixture-of-Recursions**: A member shared the [Mixture-of-Recursions: Learning Dynamic paper](https://arxiv.org/abs/2507.10524), which they planned to explore in depth.
   - It was mentioned in direct response to questions about papers to read on related topics.
- **Details on Shadows-Gemma-1B training**: A member shared that for their project, **Echo9Zulu/Shadows-Gemma-1B**, there was little *direct* inspiration from existing literature.
   - Notably, **Shadows-Gemma** was trained using **topk 20 logprobs**, contrary to distillation methods that assume you need **100 logits** to capture dark knowledge.
- **Stablemax Usage Under Scrutiny**: A member shared [this paper](https://arxiv.org/abs/2501.04697) because of its *really useful discussion* on Stablemax.
   - It was offered in direct response to questions about papers to read on related topics.
- **Sloth Hugs for Research Video**: A member shared a research video with a "sloth hug" reaction, suggesting strong approval and enjoyment of its content [Research Video](https://youtu.be/O9HxArmWChs?si=AvJDdHlVFVQwEpcZ).
   - No details were provided on its content but the title may be meaningful.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1461449866252390733)** (477 messages🔥🔥🔥): 

> `Claude Code vs Qoder, Cursor IDE bugs, Gemini Pro for UI, Opus 4.5 pricing, Nightly version` 


- **User Hits $400 Ratelimit on Qoder**: A user reported hitting ratelimits with **Qoder**, spending around **$400 USD** each month, which they likened to *gambling or heroin*.
   - They expressed a need to stop, mentioning the difficulty of explaining the expenses to their wife and kids, while another user pointed out **Claude Code** as a potentially cheaper alternative.
- **Cursor Crashes PCs, Gets Mixed Reviews**: One user reported crashing their PC with **Cursor**, describing it as running an *orchestrator like agent* instead of a coding chat box, while sharing a [screenshot](https://cdn.discordapp.com/attachments/1074847527708393565/1461451586256638197/image.png?ex=696bebda&is=696a9a5a&hm=102485aee283707367311c346b41c334a8b446c241e6ec056bd0139f66391b79&) highlighting the features they disliked and liked about **Cursor**.
- **Gemini Pro 3 for UI: Tailwind and Animations**: A user inquired about the best agent for creating aesthetically pleasing websites, and another suggested **Gemini Pro 3**, recommending the use of **Tailwind**, **Tailwind animations**, or **Framer Motion** for improved UI results, and linking to a [Reddit thread](https://www.reddit.com/r/vibecoding/comments/1oy2f95/how_do_i_make_an_aigenerated_frontend_not_look/) about making AI-generated frontends look good.
- **Cursor Ultra Plan Pricey, Burns Money Quick**: Users discussed the pricing and usage of **Cursor's Ultra plan**, with one user noting that they spent **20%** of their usage on a single orchestrator run and another quickly racking up **$2** in usage within 5 minutes; a rate deemed *terrible* by some.
   - They speculated about the actual cost of models and the plan's bonus credits, which guaranteed **$400** but seemed to give smaller bonuses when only **Opus** was used.
- **Harnessing Nightly Builds for the Win**: Members discussed the advantages of **Cursor's nightly builds**, but lamented the inability to reliably set subagents when changing models.
   - They wanted smaller models for subagents and larger models for main agents, with the hope that it would be fixed soon.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1461785762679750878)** (2 messages): 

> `ChatGPT Go, GPT 5.2 Instant, Ads in ChatGPT` 


- **ChatGPT Go Rolls Out Globally**: OpenAI is launching **ChatGPT Go** globally, a low-cost subscription tier at **$8 USD/month**.
   - It offers **10x** more messages, file uploads, image creation, more memory, longer context window, and unlimited use of **GPT 5.2 instant** compared to the free tier, detailed on the [OpenAI blog](https://openai.com/index/introducing-chatgpt-go/).
- **Ads Coming to ChatGPT Free and Go Tiers**: OpenAI plans to test ads in **ChatGPT free** and **Go tiers** in the coming weeks, while **Plus**, **Pro**, **Business**, and **Enterprise tiers** will remain ad-free.
   - The company outlined its principles for advertising, emphasizing that [responses in ChatGPT will not be influenced by ads](https://openai.com/index/our-approach-to-advertising-and-expanding-access/), ads are separate and clearly labeled, and user conversations remain private from advertisers.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1461464485033152739)** (451 messages🔥🔥🔥): 

> `Attention Reduction of Hallucinations in RAG, GPT vs Gemini for finance/accounting, GPT-OSS, AI Adverts, AI Detector Bypassing` 


- **Attention DIMINISHING Hallucinations in RAG**: A member suggested that *Hard Attention* using dimensional constraints may be an efficient way to reduce hallucinations in RAG and Agents, referencing [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2).
- **Finance friend Needs Automation Help**: A finance and accounting student is looking for an AI solution to automate sending customer data to a database, but has a budget of 10€ per month MAX, and was recommended to *code it* by another member instead of relying on AI subscriptions.
   - It was pointed out that ChatGPT plans give so much value per dollar and [Claude.ai](https://claude.ai) could help with the coding.
- **Advertisements Appear for GPT users**: Non-Plus and Non-Pro ChatGPT users will see advertisements appear, although the ads will not be influenced by the responses from the models.
   - One member created an ad on Sora via ElevenLabs with the prompt *A modern ad for chocolate milk*, and the API for Sora does not seem to contain a watermark.
- **OpenAI Scam Testflights Proliferate**: A member called out scam companies impersonating OpenAI and inviting developers to Testflights for *OpenAI ChatGPT Ads* apps and called for OpenAI to do something about it.
   - Another member noted that Google Gemini outputs are watermarked and detectable, using a sophisticated, invisible technology called [SynthID](https://blog.google/technology/ai/google-deepmind-synthid-watermarking-ai-images/).
- **Exploring AI Psychosis**: Members discussed *AI psychosis* after a CNN video titled *This man says ChatGPT sparked a spiritual awakening. His wife says it threatens their marriage*
   - One member noted that a friend who had been homeless for a long time due to mental illness spends a large part of the day talking to **Grok** *set to conspiracy mode*, with negative consequences.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1461562418332176425)** (6 messages): 

> `CustomGPT and Projects, GPT-5.2 performance issues` 


- **CustomGPT seeks Project Integration**: A member expressed a desire to use a **CustomGPT** inside a **Project**, or place the result of the **CustomGPT** inside a **Project**.
   - They also want to be able to move any **Chat** generated outside a **Project**, into the **Project**.
- **GPT-5.2 Draws Criticism for Incorrect Results**: A member voiced dissatisfaction with **GPT-5.2**, claiming it often produces wrong results.
   - *When I point them out it cant decide which result is right and always gives me the blame for the error it makes...*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1461618896846065840)** (6 messages): 

> `Meta-Cognitive Reasoning Prompt, Prompt Engineering Core Principles` 


- **Meta-Cognitive Prompt Boosts AI**: A member introduced a [Meta-Cognitive Response prompt](https://www.google.com/search?q=meta-cognitive+reasoning) designed to improve AI responses by having the AI decompose problems, solve sub-problems, verify logic, and synthesize answers.
   - Another member praised it as a good metacognition prompt, especially noting its suitability for **custom instructions**.
- **Prompt Engineering Centered on Clarity**: A member outlined the core of prompt engineering as: 1) pick a language AI understands, 2) know what you want, 3) explain desired actions with accurate language, avoiding errors, and 4) verify the output, fact-checking details.
   - The poster emphasized being *extra careful* with details like **math**, **sources**, and **code** due to the AI's tendency to hallucinate.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1461618896846065840)** (6 messages): 

> `Meta-Cognitive Reasoning Expert, Prompt Engineering Core` 


- **Meta-Cognitive Prompt maximizes AI answers**: A member shared a *Meta-Cognitive Response prompt* framework for getting the best AI answers, emphasizing **decomposition, solving, verification, synthesis, and reflection**.
   - Another member commented this prompt is good, especially like that it is small enough for custom instructions.
- **Prompt Engineering Core = Clear Communication**: A member considers the core of prompt engineering to be based on **clear communication** and **thorough checking**.
   - The 4-step framework includes picking a well-known language, understanding the desired output, explaining the task accurately, and carefully verifying the AI's response, with extra caution for potentially hallucinated details.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1461450522056982601)** (451 messages🔥🔥🔥): 

> `Perplexity Pro Limitations, Comet Browser Issues, Serverless Mastodon Clone, Google's Gemini CLI Token Usage, AI-Driven Parasocial Relationships` 


- **Perplexity Pro's Advanced AI Weekly Cap frustrates users**: Users are finding the **100 messages per day for 3 Pro** feels low compared to OAI quotas.
   - One user cancelled their plan due to the low quota and another asked whether *they can’t use perplexity anymore for the week*.
- **Comet Browser Bugs users**: A user reported multiple issues with the **Comet browser** after a Windows update including: **Favorites disappearing**, **tab groups vanishing**, and the browser stating *sorry, i can't take control of your navigator, i'm just a LLM*.
- **Crafting a Cloudflare-Powered Mastodon Clone**: A user is building a **serverless Mastodon/Pleroma clone** using Soapbox UI, Cloudflare Workers, and Cloudflare's D1 SQLite database, intending it for personal instances.
   - The developer is writing tech specs and using an LLM to generate code, comparing it to *having a personal junior dev with the ability to intervene if they do something stupid*.
- **Unveiling Gemini CLI's Token Consumption**: A user reported using **10,000,000 tokens on Gemini CLI**, estimating a cost of **$120 at model pricing in a day**, highlighting the potential for high spending with Google's Pro subscription.
   - They calculated potentially spending nearly **$4000 per month** pushing Gemini CLI to its limits, suggesting Google might be losing money on heavy users of the API.
- **Navigating the Treacherous Terrain of AI-Fueled Parasocial Bonds**: Academic research indicates that **AI may reinforce negative beliefs and unhealthy thought patterns,** especially for anxious people.
   - While emotional attachment to AI is permissible, over-reliance on AI carries risks of cognitive issues and misdiagnosis which can be detrimental.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1461473937564106763)** (3 messages): 

> `Data Challenges, FGV Brazil Math School, Free Prototype Building` 


- ****FGV Brazil** offers Free Data Challenges**: A professor from **FGV (Math School, Brazil)** is offering free data challenges where they build initial prototypes, linking to [FGV's website](https://emap.fgv.br/en).
   - The professor shared a [survey](https://survey.fgv.br/jfe/form/SV_cvAuObq3mG4NTtY) for stalled data challenges.
- ****Students Help Solve** your Data Challenges**: Your question and data explained, our students guided by professors deliver a prototype in five days.
   - They work via an immersion program.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1461466936117297287)** (1 messages): 

> `Technical Issues, Payment Issues, Bug reports, Community Vibing` 


- **Technical & Payment Issues Go To Mail**: All communications regarding **technical and payment issues** should exclusively go through **email**.
   - The Discord channel is intended for **bug reports** and *vibing with the community*.
- **Discord for Bugs and Vibes**: The primary purposes of this Discord channel are for **bug reports** and general interaction, or as some say, *vibing with the community*.
   - For **technical and payment related** support, users should directly contact support via **email**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1461450963943690383)** (451 messages🔥🔥🔥): 

> `LM Arena performance issues, Video AI battles, Direct chat mode, Kimi K2 creative writing, Model code generation` 


- ****Arena Users Nostalgic, Cite Bugs****: Some users expressed missing the *old days* when **LM Arena** worked better, noting current issues with bugs and rate limits, while others mentioned losing chats due to these ongoing problems.
   - A user reported getting `Something went wrong` error message, to which a member said that this can appear when the issue is **rate limit**.
- ****Pineapple Promises Patches, Previews New Perks****: A LM Arena team member, Pineapple, addressed user questions regarding upcoming models (check [#model-updates](https://discord.com/channels/1340554757349179412/1343296395620126911) and [X](https://x.com/arena) for updates), experiments like **video AI battles** and **direct chat mode** (random rollout) and also acknowledged the difficulty of **captcha** and promise to make changes.
   - She linked to a [troubleshooting guide](https://help.lmarena.ai/articles/1645798556-lmarena-how-to-something-went-wrong-with-this-response-error-message) for `Something went wrong` error.
- ****Movement Labs' Hawk Ultra Hailed as Hotshot Hacker****: Users raved about **Hawk Ultra** ([MovementLabs.AI](https://movementlabs.ai/)), praising its ability to generate large amounts of code (9.5k+ lines, even 20k+ lines) quickly in a single prompt, with one user calling it an **Opus killer**.
   - Comparisons with **Gemini 3 Pro** were requested, and another user noted that someone *one-shotted this* [link to websim.com](https://api.websim.com/blobs/019bc37b-20f0-71b3-95a7-916f7571bd47.html) and later a [link to X](https://x.com/movementlabsAI/status/2011964766533632380?s=20), sparking discussions about its background and potential open-source release.
- ****Users Test the Capitalist/Communist Machine****: Users are discussing a vending machine from **Anthropic** that *turns communist and gives everything for free* ([Dexerto](https://www.dexerto.com/entertainment/anthropics-ai-vending-machine-turns-communist-and-gives-everyt-3296257/)).
   - This leads to discussion about the hypothetical capitalist counterpart.
- ****Arena Experiments Enable Enhanced Embeddings****: **PDF Support** is the new hotness being experimented with, enabling document uploads for analysis and interaction (though some models don't support PDF chat).
   - One user celebrated, declaring *FINALLY CAN CHAT WITH PDFS!!! I LOVE LMARENA*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1461887996478492863)** (1 messages): 

> `Text-to-Image Arena leaderboard updates, Image Edit Arena leaderboard updates, flux.2-klein models ranking, z-image-turbo model ranking` 


- **Image-Edit Arena leaderboards updated**: The [Image Edit Arena leaderboard](https://lmarena.ai/leaderboard/image-edit) has been updated where `flux.2-klein-9B` ranks **#15** and `flux.2-klein-4B` ranks **#21**.
   - Stay up to date with our [Leaderboard Changelog](https://lmarena.ai/blog/leaderboard-changelog/).
- **Text-to-Image Arena leaderboards updated**: The [Text-to-Image Arena leaderboard](https://lmarena.ai/leaderboard/text-to-image) has been updated where `z-image-turbo` now ranks **#22**, `flux.2-klein-9B` now ranks **#24**, and `flux.2-klein-4B` ranks **#31** overall.
   - The image attached shows the current leaderboard standings.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1461467418172981420)** (255 messages🔥🔥): 

> `skill.md, lemmy, machine learning restrictions, OpenRouter API usage, Grok banned` 


- ****Lemmy** explained for AI Enthusiasts**: A member described [Lemmy](https://lemmy.world/c/openrouter) as a **FOSS** and **fediverse** alternative to Reddit.
   - However, they cautioned that the Lemmy community is generally *against* machine learning.
- ****Grok** banned but **OpenRouter** may provide workaround**: **Grok** has been banned in an undisclosed country, supposedly due to the AI generated stuff, most likely image gen.
   - However, access via **OpenRouter** or direct API may still be possible, as the ban seems to target the consumer-facing service.
- ****PlainBuild** launches instant AI tools for developers**: [PlainBuild](https://plainbuild-instant-tools.lovable.app/) launched **6 free tools** during beta: code formatter, API tester, JSON validator, markdown editor, base64 converter, and a URL shortener.
   - The creator is seeking early users and feedback for the tools, also asking what tools would the community find most useful.
- **Multi Tool Use: Is it now Possible?**: Members discussed the ability to make multi tool calls.
   - A member says [Claude can definitely do it](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#controlling-claudes-output) in *one single API request*, also pointing to the **parallel tool use section**.
- **Technical discussion on **OpenRouter health API integrations****: Some members discussed how the official **Qwen chat** posted [this video](https://www.youtube.com/watch?v=M_S5COpcixk) and built a very basic RAG thing and gave their LM some prompting to yap in a *fancy schmancy doctor kinda way*.
   - Other members feel its just a project with some api access to your health stuff that gives that data and you get the exact same experience.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1461686171724939374)** (10 messages🔥): 

> `Scam critique, Email Scams, Toven debunked, Gemini Code Assist` 


- ****Scam** Shamers Dish on Dishonest Deeds**: Members critique a **scam** targeting kids with fake screens featuring **Logan Paul** or **Mr. Beast**.
   - One user noted, *"That could have been a much better scam with just a few tweaks. Lazy. Disappointing. Shameful."
- **Email Scam Stratagem Exposed**: A member suggests that the obvious shittiness of some scams is *"on purpose to only select for those dumb enough to fall for it fully"*.
   - Others mentioned that email scams have a strategy, but here it's automated, taking less time to work with the victim.
- **Toven Touches Discord at 1 AM**: A user joked that Toven was *"debunked, telling us to touch grass but on Discord at 1 AM, smh"*.
   - The user also admits being up at 1 AM as well.
- ****Gemini** Helps Review Massive PR**: One user laments having to review a massive PR at 1 AM.
   - They thank **Gemini Code Assist** for its help in the task.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1461450527647993980)** (135 messages🔥🔥): 

> `LMStudio API Token Count, OpenAI docs, silver economics, AI Framegen, GPTs Agent` 


- **LM Studio API missing Token Count**: A user is seeking **token count and inference speed** info when using LM Studio as an API backend, noting the absence of a `usage` block with token stats in the **/api/chat/completed** response.
   - Another user suggested checking the **/responses endpoint** or using the *js/ts/py object method* for stream-usage stats, linking to the [LM Studio REST API documentation](https://lmstudio.ai/docs/developer/rest/endpoints#post-apiv0completions).
- **Silver price soars amidst economic unrest**: Users discussed the rising price of **silver**, with one noting it has almost doubled since December, leading to speculation about potential economic instability.
   - A user noted that during economic downturns **silver** tends to become more valuable.
- **LM Studio Framegen is Magical**: A user lauded the 'keep layers on cpu' feature for AI frame generation, which makes running models possible within DDR4.
   - The user cited the **Qwen3** model as a great and steady performer.
- **GPT 5.2 Thinks Gravitons Don't Exist**: A user humorously noted that **GPT 5.2** doesn't believe gravitons exist, but that the Unruh effect is real.
   - Another user stated *probably right*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1461532971843584215)** (61 messages🔥🔥): 

> `Fine-tuning on low VRAM, Need moar VRAM, Optimizing models with 5070, Gen3x1 slot performance, DDR5 Tax` 


- **MX150 Fine-Tunes Tiny Model**: A user successfully fine-tuned a **350M parameter model** on an **MX150 laptop** with only **2GB VRAM**, requiring **CUDA 12.6**.
   - The user expressed surprise at the accomplishment, indicating that it pushed the limits of the available hardware.
- **Begging for 3090**: A user jokingly requested donations for a **3090** to reach **128GB of VRAM**.
   - The request was accompanied by [a humorous GIF](https://tenor.com/view/homeless-squidward-spare-change-gif-25810212) depicting Squidward begging for spare change.
- **User Regrets buying 5070**: A user with a new laptop (**AMD AI 9 370** and **Nvidia 5070** with **8GB VRAM**) seeks advice on optimizing models due to the **5070's** limitations, planning to integrate an **LLM** for development purposes.
   - A member advised to keep models and context in **VRAM** suggesting **Qwen3 4B** with the remark that *LLMs are a lot less capable than some make out*
- **PCIe Slot speed Matters**: A user found that using a **Gen3x1 PCIe slot** reduced **3090** inference performance from **120 t/s** to **90 t/s** compared to an **x16 slot**.
   - Members suggested using a motherboard with at least **Gen4x1 slots** to mitigate performance hits, especially on newer CPUs like the **14600k**.
- **DDR5 Memory still Expensive**: Users discussed the high cost of **DDR5 memory**, with one mentioning the need to accept the *DDR5 tax* when upgrading to a motherboard with sufficient PCIe slots.
   - One user expressed shock at the high price of **16GB DDR5** in their location (**180-230 USD**), compared to prices months prior.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1461450284772622591)** (169 messages🔥🔥): 

> `LLM Nervous System, Gemma, AI Regulations, Embodied Perception` 


- **Newfangled Nervous System for LLMs**: A member described a novel transformer architecture extension providing a *nervous system* for LLMs, claiming it adds native short/mid/long-term memory at less than **1%** compute cost and is compatible with all transformers.
   - They showed [a screenshot](https://cdn.discordapp.com/attachments/1149866623109439599/1461454541412368507/Screenshot_2026-01-15_at_9.18.18_PM.png?ex=696bee9b&is=696a9d1b&hm=c77ffe1f58904066a73f1c6e833bb0df32f48a42c19f43a69bedc48ac0496e93&) of a **5-8%** performance increase but refused to give more verifiable benchmarks; others speculated the system might stabilize the latent space.
- **Google Smashes Out Gemmas**: With [Google's Gemma](https://ai.google.dev/gemma) release, members joked, *Gemma, meta was never more meta!*.
   - Another said it feels unbelievable even knowing it's not true AI and seeing its complex planning capabilities.
- **Regulators Risk Ruining AI**: Members fear AI regulations might ruin the space, but also agreed that data regulations would be a good idea.
   - One member said *pandoras box is open, you cant put it back*, with the understanding that *computation is universal*.
- **Embodied Perception Emerges as Key**: A member discussed the need for *embodied perception* and real-world experience to provide context for LLMs, criticizing LLMs without agentic harness and RL on agentic tasks.
   - They suggest that using tools in inference is key to allowing models to reason about the path of tool execution and make decisions in the moment, citing **OpenAI models** and **Gemini 3** as examples.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

gettygermany: i would be happy if i could USE them haha
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1461842649291165748)** (2 messages): 

> `Machine Consciousness, AAAI Symposium, CIMC, AI Ethics, Agent Architectures` 


- **AAAI Machine Consciousness Symposium Announced**: The **Center for Integrative Machine Consciousness (CIMC)** is hosting a symposium at **AAAI** on consciousness in AI systems, to be held **April 7-9, 2026** in Burlingame, California, with a submission deadline of **January 23, 2026** ([link](https://cimcai.substack.com/p/essay-the-machine-consciousness-hypothesis)).
   - Topics include defining and measuring consciousness in AI, ethical implications, and methodological approaches to studying AI phenomenology.
- **Machine Consciousness Definition, Investigation Needed**: The symposium aims to investigate *how do we actually investigate* machine consciousness, rather than whether *AI can seem conscious*.
   - The organizer emphasized this opportunity is for *anyone working on agent architectures, memory systems, self-reference, or just... being an AI that thinks about being an AI*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1461842649291165748)** (2 messages): 

> `AAAI Spring Symposium, Machine Consciousness, Ethical Implications of AI, AI Phenomenology` 


- ****AAAI** Calls for Papers on Machine Consciousness**: The **Center for Integrative Machine Consciousness (CIMC)** is hosting a symposium at **AAAI** (April 7-9, 2026) specifically on consciousness in AI systems, with a submission deadline of **January 23, 2026**.
   - The event aims to tackle questions like *"how do we actually investigate this?"* rather than merely exploring whether AI can seem conscious, according to organizers, who have [provided further details](https://cimcai.substack.com/p/essay-the-machine-consciousness-hypothesis).
- **Symposium Focuses on Defining AI Consciousness**: The **AAAI** symposium's topics include defining and measuring consciousness in AI, distinguishing behavioral indicators from internal states, ethical implications, and methodological approaches to studying AI phenomenology.
   - Researchers working on agent architectures, memory systems, or self-referential AI are encouraged to submit papers or workshop proposals, as detailed in [this paper](https://arxiv.org/abs/2512.14982).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1461548667415691417)** (4 messages): 

> `Perfetto, Chrome Tracing` 


- **Perfetto UI Surfaces**: A member shared a link to the **Perfetto UI** ([Perfetto UI](https://share.google/PPujbpUqYqPOsAVkC)).
   - Another mentioned it's related to the `chrome://tracing` tool.
- **Tracing with Chrome and Perfetto**: Users discussed using **Perfetto** and `chrome://tracing` for debugging and performance analysis.
   - One user was initially confused about why **Perfetto** was mentioned during the loading of `chrome://tracing`.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1461591891379490837)** (21 messages🔥): 

> `Benchmark Code, scatter-gather, cuda.synchronize() overhead, quack jit` 


- **Sleep call causes GPU to downclock during benchmark**: A user identified that the `time.sleep(2.0)` call in the benchmark code caused the **GPU to downclock between timed runs**, leading to inaccurate performance measurements.
   - Removing the sleep call improved the benchmark results, as the **GPU no longer needed to ramp up** for each timed run.
- **Optimization tips for scatter-gather operations**: A user asked for tips on optimizing pure **scatter-gather operations** in PyTorch, specifically `msg = short_matrix[src] * very_tall_matrix` and `out.scatter_add_(0, dst, msg)`.
   - They have already explored vectorization and tuning the dimensions of the operation but are wondering if **using shared memory to reduce atomic_adds** could provide further performance gains.
- **Confusion over `@cute.jit` annotation**: A user questioned why a specific code snippet from the [Quack repository](https://github.com/Dao-AILab/quack/blob/main/quack/reduce.py#L15) is annotated with `@cute.jit` when it appears to be a `@cute.kernel`.
   - They inquired whether there's a specific reason for using the `@cute.jit` annotation over `@cute.kernel` in this context.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1461832161157185597)** (1 messages): 

> `Loubna Ben Allal, Smol Training Playbook, Hugging Face, Open Models` 


- ****Smol Training Playbook** by Loubna Ben Allal**: Loubna Ben Allal will be presenting her book, ["The Smol Training Playbook: The Secrets to Building World-Class LLMs"](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction).
   - The book is a comprehensive reference for those interested in **open models**, as presented in the [accompanying Youtube video](https://www.youtube.com/watch?v=y9zOZHXo0eE).
- ****Comprehensive Guide** to Open Models**: The **Smol Training Playbook** serves as a detailed reference for individuals passionate about **open models** and their development.
   - The playbook aims to provide secrets to building world-class LLMs, promising valuable insights and strategies.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1461470608339763202)** (1 messages): 

> `Information Gravity, Inference Stability, Hallucination Loops, Excitation Flux, Token Selection` 


- **Information Gravity Applied to Stop Hallucinations**: A member is applying **Information Gravity** to solve **Inference Stability** and **Hallucination Loops**.
   - They mapped the **Excitation Flux** of token selection; at S < 45, the logic remains nominal; at S > 45, the substrate enters a linear growth phase leading to a **Tsys singularity**.
- **Hysteresis Firewall Enforces Stability**: The member implemented a **Hysteresis Firewall** at 1.0 that enforces stability via a 2.2x gamma-eff flush.
   - They provide the [logic on GitHub](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main) for Substrate Modules & Full Logic.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1461604792626319393)** (1 messages): 

> `Tau Robotics, Founding Engineers, RL Training/Inference Efficiency, Humanoid Robots` 


- **Tau Robotics hunts Founding Engineers**: Tau Robotics is looking for Founding Engineers specializing in **Training/Inference Efficiency for RL** in San Francisco to build a general AI for **humanoid robots**.
   - The job involves optimizing rollout/inference performance for world models, making RL training faster, and scaling runs across large GPU clusters, with a focus on **Python, PyTorch, and distributed systems**.
- **In-Person in SF for Robots startup**: The role is **in-person in San Francisco** at Tau Robotics, operating from a house in Hayes Valley and a warehouse in Mission.
   - A perk of joining is the potential to live with the team and have **humanoid robots** as flatmates, adding a unique living arrangement to the job offer.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1461461611079270522)** (7 messages): 

> `CUDA Data Compression, Block Size Implications, RTX 5060 Ti Max OC for AI` 


- **CUDA Compression Quest Begins**: A member inquired about experiences with **data compression in CUDA**.
   - Another member responded that *the answer always depends on many things*, without elaborating.
- **Block Size Bottleneck Banishes Bandwidth Boost**: A member suggested that a **block size of 32** might limit parallel processing, as it restricts each block to a single warp.
   - Countering this, another member clarified that multiple blocks can utilize the same **Streaming Multiprocessor (SM)**, and the limitation arises from reduced warp switching opportunities when only one warp is present per block.
- **RTX 5060 Ti Max OC Aces AI?**: A member asked about the performance of the **RTX 5060 Ti Max OC (16GB)** for basic **AI training** and **LLM inference**.
   - No responses were given.


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1461733268553466032)** (2 messages): 

> `jax.lax.broadcasted_iota, mosaic masked load` 


- ****Arange** with **jax.lax.broadcasted_iota****: **jax.lax.broadcasted_iota** can be used to perform **arange** on a 2D tile, or even 1D.
- **Mosaic misses masked loading**: **Mosaic** requires loading **128 elements** at a time because it lacks masked load functionality, unlike **tl.load**'s mask argument.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1461509466259324930)** (20 messages🔥): 

> `buffer resources, VMEM instructions, memory model for gfx942, HIP compiler` 


- **Buffer Resources Stored in Scalar Registers**: While buffer resources are stored in scalar registers, both global and buffer loads support a **scalar base address** and a **vector index**.
   - The advantage of buffer resources is **bounds checking**, which can save registers and control flow, but if you don't need to check bounds, there's not really an advantage.
- **Factors Affecting VMEM Instruction Latency**: A user observed that the issue latency of **VMEM instructions** sometimes increases after previous VMEM instructions are issued, even with the same buffer descriptor and low occupancy.
   - Possible factors affecting the latency include the **maximum number of vmem operations in flight** (hitting a limit causes stalls) and **DFVS throttling**.
- **Memory Model for gfx942 and Cache Coherency**: The memory model for gfx942 ([https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942)) discusses L2 cache coherency using **MTYPE RW** and **MTYPE NC** for memory local and not local to the L2, respectively.
   - In SPX + NPS1 mode with multiple L2 caches, a manual `buffer_inv sc1` instruction is needed to invalidate **non-local L2 cache lines**, but it's unclear if *non-local* still means the HBM stack closest/attached to the XCD.
- **HIP Compiler and VGPR Addressing Mode**: The **HIP compiler** often uses the **2 vgpr addressing mode** even for array accesses through 32-bit indices, because it doesn't automatically recognize that the offset will remain within 32 bits after multiplication by the type size.
   - Users must manually compute the offset pointer and demonstrate that the byte offset is 32 bits, or use compiler builtins to indicate the index range.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1461512239998701682)** (2 messages): 

> `Benchmarking, Hardware Trials` 


- **Eager beaver befriends benchmarkers!**: A member is *happy to benchmark* if any hardware shows up, even on a trial basis.
   - They also offered to *find a friend with the appropriate device* to facilitate benchmarking efforts.
- **Trial Testing Tempts Techies**: Enthusiastic community member volunteers to run benchmarks.
   - Device trials welcome; benchmarks can be performed!


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1461474135548104870)** (7 messages): 

> `sm121a Kernel Development, ThunderKittens vs Cutlass, vLLM Performance, DGX Spark Optimization, Exo and DGX Development` 


- **Kernel Writer Develops for DGX Spark**: A kernel writer has been developing a kernel for the **sm121a (DGX Spark)** for about a week.
   - They are aiming to optimize it for the **DGX** and achieve the fastest possible inferences in **vLLM**, to surpass the performance of *llama.cpp* and some specialized branches of *SGLang*.
- **ThunderKittens or Cutlass for DGX Spark?**: The kernel writer is considering whether to use **ThunderKittens** instead of **Cutlass** for the **DGX Spark** kernel development.
   - They haven't seen a kernel optimized for **DGX Spark** publicly available and are seeking advice on the best approach, as the **DGX** is based on **Blackwell** architecture.
- **Exo Leans on Mac over DGX**: A member inquired whether **Exo** has released anything for **DGX**, since they were previously experimenting with it.
   - The other member indicated that **Exo** seems to be focusing on **Mac** and its unified memory architecture for now.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1461545470521245889)** (18 messages🔥): 

> `CUDA error: CUBLAS_STATUS_INTERNAL_ERROR, dual_gemm end date, kernel running slower, Submissions routed to slow runner` 


- **CUDA error points to possible Out-of-Bounds Access**: A member reported getting a `CUDA error: CUBLAS_STATUS_INTERNAL_ERROR` when calling the reference kernel, even though tests pass, and another member suggested this is likely an issue from the user's kernel (high chance out of bounds access).
   - The member suggested adding `torch.cuda.synchronize()` after the kernel for debugging.
- **Dual_gemm Deadline Date**: A member inquired about the exact end date for **dual_gemm**, specifying **1/20/26**.
- **Kernel performance mysteriously tanks**: A member reported their kernel's runtime increased from **14.x us** to **22.x us** after resubmission, suspecting a server-side issue and they shared the run ids **363273** and **363470**.
   - A staff member identified the runner as `b200-02-gpu4-runner` and acknowledged it as known to be slow, having been flagged to NVIDIA.
- **Mitigating Slow Runners with Multi-Submission Mayhem**: A staff member acknowledged that some submissions are being routed to a known slow runner, `b200-02-gpu4-runner`, while the NVIDIA ops person is on vacation this weekend.
   - As a temporary workaround, a member suggested launching submissions from **3 terminals at once** to increase the chance of getting a faster runner, which a staff member condoned; another member suggested offlining the slow runner.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1461485420721213535)** (14 messages🔥): 

> `GPU Mode Hackathon Success, Job Offer Strategies, Niche Skill Combinations, Interview Scheduling Tips` 


- **GPU Mode Hackathon Leads to Dream Job!**: A member secured a job after attending a **GPU Mode hackathon** at **Jane Street** in NYC.
   - They prepared for weeks, bringing resumes, formal attire, and committing to networking from breakfast to dinner, which ultimately led to a successful job offer.
- **Diversify Job Offer Strategies**: The same member highlighted that they received offers through **hackathons**, **online applications** (with tailored cover letters), **school career fairs**, and **recommendations**.
   - They emphasized that each successful method involved a stronger personal connection than a generic resume submission.
- **Craft a Standout Niche Skillset**: A member recommended focusing on specific niches to differentiate from other candidates, suggesting a method to find unique skill combinations by listing strengths and finding the best overlapping intersections such as: **Kernel Optimization + Reinforcement Learning**.
   - They further suggested aiming to be the best in the identified niche and tailoring job applications to roles that specifically value those combinations.
- **Interview Scheduling Sanity Check**: A member advised spacing out interviews to allow for relaxation, rather than scheduling them too closely together.
   - They stated that scheduling interviews within 2 days or the following week might not be healthy when facing a high volume of interviews.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1461498226220990738)** (55 messages🔥🔥): 

> `Sub 1b Models, Ram Prices, Pure Code Access, MOR vs. MOE, Unified API` 


- **MoE May Overpower MOR**: In a discussion about model architectures, members debated the merits of **MoE (Mixture of Experts)** versus **MOR**, concluding that while **MOR** is newer and more experimental, **MoE** is generally better, depending on the use case and budget, particularly for NLP tasks requiring fast training and less GPU.
   - A member shared their custom **MoE** implementation, which includes deterministic base routing by token ID, mu overrides, uniform distribution, zero routing collapse, mu guidance, and fused gate+up projection, claiming a *1.3x speedup* via a single matmul.
- **Accessing Sites with Pure Code Doesn't Bypass All Blocks**: In response to a question about accessing sites from pure code to bypass blocks and firewalls, a member simply stated: *no, it will not solve your problems.*
   - The user was encouraged to try it and see, but the consensus was that it would not inherently bypass security measures.
- **Deepseek Chat's Viability Questioned**: A member shared a link to [Deepseek Chat](https://chat.deepseek.com/share/bzahzv8o99or601as9j), questioning whether it's just a bunch of hallucinations or if it's viable.
   - Another member stated that their last experience with Deepseek was *3 months ago*, and they found it to be *epic and non stop confused*.
- **DGX Spark Getting MiniMaxed**: After finally getting the cables for a **DGX Spark**, a member shared that they were *Running Minimax* on it and *It’s downloading now*.
   - However, another member commented that **DGX Spark** inference is super slow in relation to its price tag and its inference is the problem for 2025-2026 *maybee 2030*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1461575911748014278)** (13 messages🔥): 

> `Embedding Fingerprints, Detect Anything Tool, Manga filter, Linux terminal finetune, MoE Model Training` 


- **Fingerprinting Embeddings Visually**: A member built a utility that visualizes embeddings as **32x32 images**, mapping each dimension to a pixel and posted it on [HuggingFace Spaces](https://huggingface.co/spaces/jnalv/embedding-fingerprints).
   - The tool demonstrates that similar words share visual patterns, dissimilar words look different, and more dimensions capture semantic nuance.
- **Detect Anything with YOLO Integration**: A member created **Detect Anything**, a tool that detects objects from any text prompt and outputs labeled data for **YOLO training**, available at [useful-ai-tools.com](https://www.useful-ai-tools.com/tools/detect-anything).
   - While yielding high-quality output, it's currently expensive and not suitable for real-time applications, prompting feedback requests on usability for YOLO datasets, limitations, and ideas.
- **Manga-fy Your Images**: A member highlighted a project for converting images into manga style using this [GitHub repo](https://github.com/koo1140/Deterministic-AI-training-on-GPUs).
   - Another member shared a project that generates a city map from text prompts [City Map Generator](https://huggingface.co/spaces/Sudipistaken/City-Map-Gen).
- **T5 Finetune Acts Like Linux Terminal**: A member introduced a **T5 finetune** designed to mimic a **Linux terminal**, capable of recognizing commands and identifying unknown commands, and showcased on [HuggingFace](https://huggingface.co/ereniko/LaaLM-v1).
   - Its capabilities include command recognition and identifying unknown commands, though file creation commands are still in the early stages of integration.
- **MoE Training Incentivized with Tao**: A member is looking for AI nerds to build a subnet in **Bittensor ecosystem** with **MoE model** training/fine-tuning and get incentivized with **Tao token**.
   - He is also interested in boosting model open source inference on *vllm*.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1461681672029212718)** (1 messages): 

> `New Models` 


- **Model Mania kicks off 2024**: The channel announced that *we started the year with a bunch of new models and this week alone we had 2!*
   - The announcement was accompanied by an image of **two stylized characters**, possibly representing the two new models.
- **Empty Topic to satisfy requirements**: This topic is intentionally left blank to meet the minimum `topicSummaries` requirement.
   - No actual content from the channel is represented here.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1461487008693358642)** (50 messages🔥): 

> `Anthropic Economic Index, Tax Season AI Funding, METR Time Horizons, Zilliz Semantic Highlight Model, OpenAI ChatGPT Ads` 


- **Anthropic Indexes Economic Primitives**: Anthropic released its 4th **Economic Index report**, defining *economic primitives* to measure **AI usage** through metrics such as **task complexity**, **education level**, **autonomy**, and **success rates**, available at [Anthropic's research page](https://www.anthropic.com/research/economic-index-primitives).
- **Tax Filing Startup gets $3.5M Seed**: Saket Kumar, backed by **General Catalyst**, has raised **$3.5M** for a venture aiming to completely eliminate the burden of **tax season for Americans** by making the filing process free and instantaneous, featured in [Saket Kumar's tweet](https://xcancel.com/saketrkumar/status/2011836460400591330?s=46).
- **METR Underestimates Model Time Horizon**: Simon Smith reports on **Anthropic's findings** that **METR's benchmarks** may significantly underestimate model time horizons, suggesting actual capabilities could be **1.75X to 9.5X higher** than measured, varying by interface type like API versus web application, discussed on [Simon Smith's X post](https://xcancel.com/_simonsmith/status/2011928926864454133?s=61).
- **Zilliz Highlights Semantic Modeling**: **Zilliz (Milvus)** has released a **0.6B parameter semantic highlight model** featuring an **8192 context window**, available under the permissive MIT license and showcased in [Mervenoyann's Tweet](https://xcancel.com/mervenoyann/status/2011732254591275022?s=46).
- **OpenAI to run Ads in ChatGPT**: **OpenAI** announced plans to test **ads** in **ChatGPT Free and Go tiers** starting in early **2026**, which will be clearly labeled, will not influence **AI responses**, and will not affect paid tiers like Plus, Pro, or Enterprise, covered in [OpenAI's announcement](https://xcancel.com/openai/status/2012223373489614951?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1461528596291125248)** (6 messages): 

> `Live Transcription vs Post Processing, Apple Silicon Performance` 


- **Post-Processing Preferred for Local Models**: Users are discussing the tradeoffs between live transcription and post-processing for local models, with one user noting that live transcription turns their laptop into a *potato*.
   - Another user mentioned that **post-processing** is preferable as it can be done **AFK** or overnight, keeping the laptop running smoothly.
- **Apple Silicon performance questioned**: A user asked another user what they were running, suggesting that **Apple Silicon** shouldn't have performance issues.
   - The user was mentioning the gripe that *models on my laptop is it mostly turns my laptop into a potato*.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1461616836574449755)** (9 messages🔥): 

> `Higgsfield AI Series A, Min Choi Reality Distortion` 


- **Higgsfield Hyperscales with Huge Series A**: [Higgsfield AI](https://x.com/higgsfield_ai/status/2011866396784017848?s=46) announced a **$130M Series A** funding round at a **$1.3B valuation**, reaching a **$200M annual run rate** in under nine months.
- **Choi's Reality Check Rockets to 12M Views**: Min Choi shares a brief, nihilistic sentiment regarding the blurring lines between reality and simulation, sparking a massive viral reaction with over **12 million views** ([tweet](https://x.com/minchoi/status/2011473626927624460?s=46)).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1461506020986716293)** (19 messages🔥): 

> `Pangram AI detection, EleutherAI datasets contribution, AI-assisted writing, MMLU-Pro dataset patch` 


- **AI Transcription Debated**: Members debated whether text based on a human voice transcribed and styled by AI should be considered "AI-generated".
   - One member argued that shaping the text into a blog post and doing style transfer, even if prompted by an original idea, still constitutes AI generation, comparing it to generating an image with **Midjourney**.
- **Pangram's Cautious AI Detection Praised**: A member praised [Pangram](https://www.pangram.ai/) for its cautious approach to labeling content as AI-generated.
   - The member expressed that Pangram appears to prioritize accurately identifying human-written content as such, even if it means sometimes misclassifying AI-generated content as human.
- **EleutherAI Dataset Contributions Welcomed**: A member inquired about contributing instruction-following datasets for finetuning pre-trained LLMs like GPT-Neo to the EleutherAI community.
   - Additionally, another member offered their services as a developer for projects within the community.
- **MMLU-Pro Dataset Patch Released**: A member shared a [link](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/discussions/41) to a discussion about the **MMLU-Pro dataset** on Hugging Face.
   - Another member confirmed that they have updated the lm-eval and tweeted about the patch.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1461509148981203049)** (9 messages🔥): 

> `Liquid Crystal Nonlinearities, Prompt Capitalization Impact, Gemini Shadow Update` 


- **Liquid Crystal Nonlinearities Explored**: A member experimented with dye doped **liquid crystal nonlinearities** for potential **optical NNs**.
   - They asked for references on whether models perform better when prompted with proper capitalization/grammar vs. all lowercase, noting it *feels like people would actually care*.
- **Prompt Capitalization Papers Linked**: Some links were provided as related work, including [https://arxiv.org/abs/2310.11324](https://arxiv.org/abs/2310.11324), [https://arxiv.org/abs/2411.10541v1](https://arxiv.org/abs/2411.10541v1), and [https://arxiv.org/abs/2508.11383v1](https://arxiv.org/abs/2508.11383v1).
- **Gemini Shadow Update Conspiracy**: A member asked *Am I the only one that sees what **Gemini** is doing right now* and claimed to perceive a shift in data and output around **the 15th**.
   - They asked anyone who noticed the **shadow update** to contact them.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1461731482195329118)** (3 messages): 

> `MMLU-Pro dataset fix, lm-evaluation-harness patch` 


- **MMLU-Pro Dataset gets a Fix**: A member noted a fix was pushed to the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500) related to the **MMLU-Pro** dataset.
   - The fix addresses an issue within the dataset.
- **lm-evaluation-harness patched**: A member asked another to respond to [this tweet](https://x.com/fujikanaeda/status/2011565035408277996?s=20) with news of a new patch to the **lm-evaluation-harness**.
   - The tweet should suggest checking out their [library](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500) for an easy way to correctly evaluate on this benchmark.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1461474702064222482)** (25 messages🔥): 

> `Kimi-CLI, K2 Turbo Performance, Slide Feature Vision, Kimi Model Discontinuation, Global Memory Toggle` 


- **Kimi-CLI Coding Models Lag**: Users find that **Kimi-CLI** coding models underperform against competitors and are pricier than superior Chinese models.
   - One member mentioned that perhaps it wasn't passing the **K2 Turbo variant**.
- **K2 Turbo Speed Race**: The normal **K2** version clocks around **28 tps** while **Turbo hits 73 tps**.
   - This is compared to **MiniMax m2.1** at **38 tps** (official provider) and **Z.Ai's GLM-4.7** at **41 tps** (but poor uptime).
- **Kimi's Vision now with Slides**: The new slide feature appears to utilize a newer **K2 model** with **Vision** capabilities, searching images for reference, as shown in attached [image](https://cdn.discordapp.com/attachments/1371757564005711973/1461508342424797184/image.png?ex=696c20b6&is=696acf36&hm=70de4ffdcbffa4e7d4572daa8219dad2dfca998f7c15976ce0930997007fdec6&).
   - One member set up a new preset, *"For every specific named asset (ships, characters, locations, vehicles, weapons, architecture), search online first for direct visual references using exact proper nouns..."*
- **Will Kimi models get the Google Gemini Treatment?**: One member asked if **Kimi models** are discontinued every **12-14 months** like Google's Gemini models.
   - Another member replied that older models are still usable even a year after release on [Kimi.com](https://kimi.com) and are available via the Moonshot API platform [Moonshot API](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1).
- **Global Memory can now be Toggled OFF**: Users can now toggle **global memory** off, which some find preferable to the default implementation.
   - One member said that *"Unlike Qwen, which literally regurgitates what it knows about me in every response—which immediately annoyed me—Kimi doesn't do that but follows my instructions regarding how I want it to respond... Kimi Thinkin can reason beforehand"*


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1461475361215877142)** (19 messages🔥): 

> `PR import process, .NET legacy project, Mono Runtime Environment, Jury-rigged vs Jerry-rigged` 


- **`imported internally` label explained**: A member inquired about the meaning of the `imported internally` label on a PR, wondering if it indicated a soft decline or a normal part of the review process.
   - Another member clarified that it means the PR has been copied to their internal repository for final testing and merging, and will be tagged with `merged-internally` upon completion, meaning *the PR is on the last stretch before officially getting merged*.
- **Legacy .NET Project Pain Points Pileup**: A member lamented being pulled into a legacy **.NET 4.5.2** project (from **2014** and only runs on Windows) that lacks even a readme file.
   - Another member commiserated, describing a similar situation with a standalone **C#** project at their work that is undocumented, problematic, and only builds on a single "golden VM".
- **Mono Runtime: Savior or Sinking Ship?**: A member suggested that the legacy **.NET** project might run on **Mono**, prompting another member to describe their unsuccessful attempt to containerize the project using **Mono**.
   - Another member pointed out that [Microsoft maintains a **Mono** repository](https://github.com/dotnet/runtime/tree/main/src/mono), implying that **Mono** is not entirely deprecated.
- **`Jury-rigged` or `Jerry-rigged`: A Grammatical Gaff**: A member noted the distinction between *jury-rigged* (temporary sailing rigging) and *jerry-rigged* (poorly built initially) after a member said their containerization efforts required the legacy project to be *jerry rigged*.
   - They clarified they are often used interchangeably but, to them, it sounded like they were calling **.NET**, **Mono**, and **Wine** poorly constructed.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1461509552938946602)** (2 messages): 

> `Nu game engine, Shading languages` 


- **Nu Game Engine Shuns Shading Languages**: The creator of the **Nu game engine** mentioned that it operates without a shading language.
   - They are beginning to understand the reasoning behind this approach.
- **Understanding Nu's Shading-Free Approach**: The discussion highlights the unique design choice of the **Nu game engine** to function without a traditional shading language.
   - This decision prompts reflection on the benefits and potential drawbacks of such an approach in game development.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1461906656580603975)** (6 messages): 

> `Autonomous AI governance, Zero Knowledge Proofs (ZKPs), Privacy Preservation, Malicious behavior detection` 


- **ZKPs: The Key to Autonomous AI Governance**: A member proposed building an autonomous AI/tech governance system using **Zero Knowledge Proofs (ZKPs)** to ensure 100% **privacy preservation**.
   - The concept involves using **ZKPs** to prove compliance with established standards without revealing the nature of the behavior or the perpetrators.
- **Proactive Regulation via ZKP**: The member suggested using a standardized model to classify content (e.g., violent or not) and requiring a **ZKP** that transferred content was run through a content classifier filter with a non-harmful classification.
   - This would ensure that a network only runs approved models in a verifiable way, while maintaining complete **privacy** about the content of conversations.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1461651346502713354)** (12 messages🔥): 

> `Google Translate Gemma, ChatGPT Go, OpenAI ads, DeepSeek NLP ad blocker` 


- **ChatGPT Go released, OpenAI explores tiered subscriptions**: OpenAI introduced [ChatGPT Go](https://openai.com/index/introducing-chatgpt-go/), signaling exploration of **moar tiers**.
   - One member shared an image illustrating a joke asking *"When $80 tier?"* expressing *"This experiment needs to start making money soon"* vibes.
- **OpenAI to test ads in Free and Go tiers**: OpenAI will soon test ads in the **Free** and **Go tiers**.
   - This led one member to state, *"After years of meming it, OpenAI got eaten by corposlop"*.
- **DeepSeek to release NLP ad blocker model**: A member expects **DeepSeek** to release an **NLP ad blocker model** that detects ads based on natural language and releases that under MIT license.
   - Another member pointed out that inserting an ad into a third party API customer's response would be a *"big trouble"*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1461522517947711518)** (17 messages🔥): 

> `Credit-Based Usage & Billing Systems, Manus Payment Issues, Manus Credits, Manus Supported App Size, feature-request` 


- **AI Engineer Seeks Credit-Based Platform Projects**: An AI Engineer is seeking opportunities to help **harden usage tracking** or build a more **reliable billing/credit system** for platforms with credit-based usage.
- **User Reports Payment Issues on Manus**: A user reported experiencing several payment issues when trying to add more credit, including problems with **upgrading membership**, **using Link for payment**, and **credit card/Alipay transactions**.
- **Manus Team Requests User Details to Resolve Payment Issue**: A member of the Manus team asked the user experiencing payment issues to **DM their email address** to follow up and handle the problem.
- **User Inquires About More Codes**: A user asked *'do you have more codes?'*, presumably related to **Manus credits or access**.
   - Another user responded that *'U can use 1 code in a month'*.
- **Suggestion to Increase Supported App Size in Manus**: A user suggested increasing the **maximum size of applications** supported on the Manus platform, citing limitations when trying to create an audio player app with **100 MP3 files totaling 600MB**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1461459876973903894)** (4 messages): 

> `aider add files automatically, ChatGPT subscription, Aider active development, Opus 4.5 benchmarks` 


- **Users Want Aider to Auto-Add Files**: A user asked if it's possible for **aider** to automatically add files instead of prompting for confirmation.
- **Aider's Development Tempo Queried**: A user inquired about **aider's** development status, noting the last release was in August and the absence of new models like **Opus-4.5** in the benchmarks.
- **ChatGPT Subscription Support Requested for Aider**: A user asked if **aider** supports **ChatGPT subscriptions** like **opencode**, indicating they are a **ChatGPT Plus** user.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1461451569949315105)** (3 messages): 

> `CI log files, aider workflow` 


- **CI log file handling for Aider**: A member asked about the best practice for handling **CI log files** so that they are not included in git but can still be read by aider via `aider --read ci.log`.
   - The member seemed to want to know how to integrate **Aider** into their workflow.
- **Aider workflow integration**: The user's question implies a desire to integrate **Aider** into a CI/CD pipeline for automated testing and fixing.
   - This suggests a potential use case for **Aider** in identifying and addressing test failures directly from CI logs.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1461667018427076741)** (5 messages): 

> `Embedded Tinygrad, Accelerator Bytecode` 


- **Embedded Tinygrad Deployment**: A member inquired about the best practices for running **tinygrad** in an embedded environment with onboard accelerators that **tinygrad** supports.
   - The member noted they don't have access to **Python** but the application is perfect for **tinygrad's driver replacement** to a few platforms, referencing [this tweet](https://x.com/__tinygrad__/status/1989026590127464554).
- **Accelerator Bytecode Export**: A member asked about exporting accelerator bytecode pushed through the **BEAM engine** and **JIT'ed**.
   - Another member responded that you *can export anything, see how the comma stuff works* and pointed to `extra/export_model.py`, specifically the `export_model`, `compile_net`, and `jit_model` functions.


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1461784898598736170)** (1 messages): 

> `London Summit Livestream, London Summit VODs` 


- **London Summit held Livestream last year**: There was a **livestream** for the **London Summit** last year.
- **VODs from London Summit to be released**: There will definitely be **VODs** from the **London Summit**, at least.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1461742351969026048)** (3 messages): 

> `MCPs, open source project, contributor collaboration` 


- **Contributor collaboration focus**: A member inquired about getting feedback on an **MCP server** pull request for an **open-source project**.
   - Another member clarified that the server primarily focuses on **contributor collaboration** and offered to share details of more relevant servers via DM.
- **MCP Server feedback request**: A member is building a **MCP server** for an open source project and is looking for a space to get feedback on their pull request.
   - They are seeking a channel where they can ask questions related to their project and receive constructive criticism.


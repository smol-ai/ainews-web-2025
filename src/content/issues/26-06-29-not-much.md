---
id: MjAyNS0x
title: not much happened today
date: '2026-06-29T05:44:39.731046Z'
description: >-
  **Meta** announced **Brain2Qwerty v2**, a real-time non-invasive brain-to-text
  decoder achieving up to **78% word accuracy** with released training code and
  dataset. **Cursor** launched **Cursor for iOS** with remote AI agents and live
  activity features. Open-weight model access is being commercialized with a
  **$9.99/mo** pass for models like GLM 5.2 and Qwen, while **Cognition**
  introduced **Devin Fusion** for cost-efficient coding. **Arena** reached a
  **$100M ARR run rate** eight months post-launch, focusing on agent evaluation.
  Infrastructure challenges, especially in China, remain critical. DeepSeek's
  **DSpark** advances speculative decoding with significant gains over prior
  methods, deployed in **DeepSeek-V4-Flash** and **V4-Pro**.
companies:
  - meta-ai-fair
  - cursor
  - deepseek
  - cognition
  - arena
models:
  - brain2qwerty-v2
  - glm-5.2
  - qwen
  - deepspark
  - deepspeak-v4-flash
  - deepspeak-v4-pro
topics:
  - brain-computer-interfaces
  - non-invasive-bci
  - real-time-decoding
  - speculative-decoding
  - agent-assisted-research
  - inference-systems
  - cost-efficiency
  - remote-agents
  - training-data
  - model-access
  - infrastructure-strategy
people:
  - jeanremiking
  - kimmonismus
  - ml_angelopoulos
---


**a quiet day.**

> AI News for 6/27/2026-6/29/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


- **Meta’s non-invasive brain-to-text milestone** drew the biggest technical attention. [@AIatMeta](https://x.com/AIatMeta/status/2071566924803395741) announced **Brain2Qwerty v2**, a real-time sentence decoder from raw brain signals; [@JeanRemiKing](https://x.com/JeanRemiKing/status/2071568417522369008) summarized the release and links; [@AIatMeta](https://x.com/AIatMeta/status/2071566934571954326) added that Meta is releasing the **training code** for v1/v2 and BCBL is releasing the **v1 dataset**.
- **Cursor shipped iOS + remote agents** in one of the day’s biggest product launches: [@cursor_ai](https://x.com/cursor_ai/status/2071641103191998810) introduced **Cursor for iOS** with always-on cloud agents and remote control of agents on your computer; follow-up tweets highlighted [Live Activities and diff review on phone](https://x.com/cursor_ai/status/2071641104869691671).
- **Open-weight model access is being productized**, not just discussed: [@cline](https://x.com/cline/status/2071617325296734309) launched a **$9.99/mo** pass for discounted access to GLM 5.2, DeepSeek, Kimi, MiniMax, Qwen, etc.; [@cognition](https://x.com/cognition/status/2071624568465490170) introduced **Devin Fusion**, claiming **35% lower cost** for “Fable-level” coding via a hybrid-model harness.
- **Arena crossed meaningful commercial scale**: [@arena](https://x.com/arena/status/2071630464583151727) and [@ml_angelopoulos](https://x.com/ml_angelopoulos/status/2071629882057228680) said Arena reached **$100M ARR run rate** eight months after launching its evaluation product, with a platform now emphasizing post-deployment and agent evaluation.
- **Infrastructure pressure remains a first-order theme**: [@kimmonismus](https://x.com/kimmonismus/status/2071524362012791114) argued China’s energy, data center, and domestic-hardware strategy is becoming a serious strategic threat; [@garrytan](https://x.com/garrytan/status/2071600933210100074) condensed the operational response to “**Build power and datacenters**.”

**Brain-computer interfaces and AI-for-science tooling**

- **Brain2Qwerty v2** is the clearest research release of the day. Meta says the system decodes **words and semantics**, not just characters, from **non-invasive** recordings in real time, narrowing the gap with invasive BCIs. Community summaries highlighted reported jumps from prior non-invasive results to **~61% word accuracy overall** and **78% for the best participant**, trained on data from **9 volunteers** in controlled typing settings. The key engineering point is not consumer readiness, but that the stack combines raw neural-signal modeling with language modeling strongly enough to make sentence-level decoding practical in the lab. See [Meta’s announcement](https://x.com/AIatMeta/status/2071566924803395741), the [code/data release details](https://x.com/AIatMeta/status/2071566934571954326), [@JeanRemiKing’s thread](https://x.com/JeanRemiKing/status/2071568417522369008), and a cautious external summary from [@kimmonismus](https://x.com/kimmonismus/status/2071712776226283902).
- The release also became a datapoint for **agent-assisted research**. [@stalkermustang](https://x.com/stalkermustang/status/2071590526965502027) pointed to Meta’s note that an **Auto Research** workflow, powered by a coding agent, discovered and implemented improvements that reduced word error rate beyond standard HPO. Whether or not one buys the “vibe-science” framing, the more sober takeaway is that coding agents are increasingly useful for **closed-loop experimental iteration** on ML systems, not just repo scaffolding.

**Inference systems: DSpark, vLLM, and decoding mechanics**

- **DeepSeek’s DSpark** was the most substantive inference topic. A long explainer from [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2071445817102315595) framed DSpark as an important step in **speculative decoding**, with emphasis on two ideas: better draft generation and smarter verification scheduling. Reported gains include **30.9% higher accepted length vs Eagle3** and **16.3% vs DFlash** on Qwen3-4B, plus production deployment in preview engines for **DeepSeek-V4-Flash** and **V4-Pro**. Follow-on commentary from [@teortaxesTex](https://x.com/teortaxesTex/status/2071631028511203457) and [@vllm_project](https://x.com/vllm_project/status/2071682507775635579) underscored the practical consequence: DSpark looks like a new **SoTA single-GPU spec decode path**, and the vLLM community is already integrating it.
- More broadly, several tweets sharpened the mental model of current inference bottlenecks. [@_avichawla](https://x.com/_avichawla/status/2071522418594861215) gave a solid explainer of **prefill vs decode**, TTFT vs inter-token latency, and why decode is often **memory-bound** because of KV-cache reads. This is useful context for why speculative decoding, KV-cache optimization, grouped-query attention, and attention redesigns matter more than raw FLOPs in many production workloads.
- NVIDIA/vLLM also pushed practical self-hosting: [@vllm_project](https://x.com/vllm_project/status/2071483552106233993) highlighted a guide for serving **Nemotron-3-Ultra 550B** with **four DGX Spark boxes** behind a single OpenAI-compatible endpoint. The notable part is less the stunt than the normalization of **private, multi-node frontier-ish inference** using standard serving stacks.

**Agent harnesses, routing, and multi-model orchestration**

- The center of gravity in agent systems continues to move from “pick the best model” to **harness engineering**. [@cognition](https://x.com/cognition/status/2071624568465490170) launched **Devin Fusion**, a hybrid-model coding harness claiming **35% cost reduction** while maintaining “Fable-level” quality. [@walden_yan](https://x.com/walden_yan/status/2071627241818399181) described related work around **sidekick** and **mid-session routing**, and [@jerryjliu0](https://x.com/jerryjliu0/status/2071737452323303750) noted the cache-efficiency advantage of sidekick-style delegation. The emerging pattern: keep an expensive planner in the loop, hand bounded subtasks to cheaper models, and preserve cache locality/context continuity.
- **Dynamic subagents** became another common motif. [@LangChain](https://x.com/LangChain/status/2071631563897377010), [@sydneyrunkle](https://x.com/sydneyrunkle/status/2071632107026174364), and [@hwchase17](https://x.com/hwchase17/status/2071633874736804066) all highlighted workflows where the main agent writes orchestration code rather than merely invoking tool calls. This is notable because it shifts the abstraction from “tool-using chatbot” to something closer to a **programmable control plane** for large task fanout.
- Open routing and retrieval stacks also got more concrete. [@LlamaIndex](https://x.com/llama_index/status/2071656315210826006) and [@jerryjliu0](https://x.com/jerryjliu0/status/2071729856900215261) introduced a **Retrieval Harness** combining semantic search, grep, file listing, and file reading in one agent loop—essentially a rebuttal to simplistic “grep is all you need” positions also criticized by [@max_paperclips](https://x.com/max_paperclips/status/2071465351959998723). On the eval side, [@hwchase17](https://x.com/hwchase17/status/2071630837976822237) announced a **Trace Judge** model for detecting trajectory errors at **~1/100th the cost** of closed models.

**Open models, Chinese labs, and commercialization of access**

- **GLM 5.2** remained the focal open model in discussion, not because of an official launch today but because many builders are now treating it as a default serious option. [@cline](https://x.com/cline/status/2071617325296734309) productized access with a monthly pass bundling **GLM 5.2, DeepSeek, Kimi, MiniMax, Mimo, and Qwen**, reducing friction around API keys and provider churn. [@tonbistudio](https://x.com/tonbistudio/status/2071595794147250540) tested **Mixture-of-Agents** configurations using GLM 5.2 with Kimi and MiniMax. [@Astrodevil_](https://x.com/Astrodevil_/status/2071572680470655253) used GLM 5.2 as the driver for a DevRel content-research agent.
- A second thread is the continued acceleration of **Chinese open-weight competition**. [@eliebakouch](https://x.com/eliebakouch/status/2071713216028389396) flagged an upcoming **LongCat 2.0 / Owl Alpha** model from Meituan: **1.6T total / ~48B active**, **1M context**, **35T training tokens**, **n-gram embeddings**, sparse attention, and training on **50k Chinese accelerators**. [@sun_hanchi](https://x.com/sun_hanchi/status/2071664412612833516) framed this as potentially the first near-frontier model trained at this scale on domestic Chinese hardware. Even allowing for uncertainty in the hardware details, this is strategically meaningful.
- On the policy/commercial side, open-source proponents argued that clampdowns on frontier APIs may backfire by pushing developers toward weights they control. See [@theinformation](https://x.com/theinformation/status/2071700452605829433), [@ClementDelangue](https://x.com/ClementDelangue/status/2071686220548133048), and [@MTSlive](https://x.com/MTSlive/status/2071634697185353956) for the recurring theme that **open weights are structurally harder to suppress than APIs**.

**RL, training infrastructure, and benchmark/eval platforms**

- **Snowflake Arctic RL** is one of the stronger infra releases in the batch. [@StasBekman](https://x.com/StasBekman/status/2071628398234087642) announced an open-source project integrating with **VeRL** and **SkyRL**, featuring **ZoRRo** for up to **6x actor-update acceleration** and **3.5x end-to-end speedup**, reducing a Text2SQL training run from roughly **5 days to ~36 hours on 32 H200s**. Snowflake also claims its **Arctic-Text2SQL-R2** beat tested configurations of **Gemini 3.1 Pro** and **Claude 4.7** on its enterprise SQL benchmark, with open recipes for text-to-SQL and multi-hop QA.
- **Arena** continued its transition from benchmark project to evaluation company. [@arena](https://x.com/arena/status/2071630464583151727) and [@ml_angelopoulos](https://x.com/ml_angelopoulos/status/2071629882057228680) reported **700M+ conversations**, **82M+ votes**, and over **10M monthly visitors**, with newer emphasis on **agent-mode evaluations** like task completion and hallucination rates. That makes Arena increasingly relevant as a **post-deployment CI/CD layer for models**, not just a preference leaderboard.
- Several other releases fit the same trend toward specialized infrastructure: [@wandb](https://x.com/wandb/status/2071603727585448025) launched **ARIA**, an autoresearch agent inside W&B; [@agenticin](https://x.com/agenticin/status/2071494912277938398) promoted **Micro-Agent** routing; and [@fitsumreda](https://x.com/fitsumreda/status/2071616094260142431) introduced **Nemotron-TwoTower**, which clones an AR LLM into a diffusion-style parallel generator, claiming **98.7% AR quality** at **2.42× throughput** for a 30B model.

**Platform and developer product updates**

- **Cursor’s mobile/remote push** is notable because it makes “cloud agents from your phone” feel operational rather than aspirational. The product now supports launching always-on cloud agents and remotely controlling computer-bound agents from iOS, with PR diff review and notifications in-app ([launch](https://x.com/cursor_ai/status/2071641103191998810), [details](https://x.com/cursor_ai/status/2071641104869691671)).
- **Claude on Azure Foundry** is now GA. [@Azure](https://x.com/Azure/status/2071651695323492418), [@claudeai](https://x.com/claudeai/status/2071653958905467027), and [@ClaudeDevs](https://x.com/ClaudeDevs/status/2071697437136486585) said customers can run **Claude Opus 4.8** and **Haiku 4.5** in Microsoft Foundry with Azure identity, billing, governance controls, prompt caching, and thinking support.
- **Rampart** from [@ndstudio](https://x.com/ndstudio/status/2071638578145145251) stood out as a pragmatic privacy tool: a **14.7MB browser-side model** for redacting PII before data leaves the client. For teams trying to make AI usable in regulated settings, this kind of small, local preprocessing model may matter more than another general-purpose chat UI tweak.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. GLM-5.2 Extreme Local Inference Tests

  - **[GLM-5.2 753B (IQ1_S) fully local across 2×M5 Max over one TB5 cable — ~16 tok/s, llama.cpp RPC [video]](https://www.reddit.com/r/LocalLLM/comments/1uiuhec/glm52_753b_iq1_s_fully_local_across_2m5_max_over/)** (Activity: 377): **A user reports running **GLM-5.2 `753B`** fully locally using **Unsloth dynamic `IQ1_S` quantization**: nominally ~`1.6` bits but ~`2.1` effective bits due to mixed higher-precision layers, yielding a `202GB` on-disk model. The setup shards weights across **2× M5 Max systems with `128GB` unified memory each** over a single **Thunderbolt 5** link using **`llama.cpp` RPC**, keeping all weights resident with no SSD paging and achieving ~`16 tok/s` generation, `16k` context, and `q8` KV cache; TTFT is prompt-length dependent due to prefill.** Commenters found `16 tok/s` for a `753B` model over two Macs surprisingly high, with one asking whether the video appeared faster than reported. Another noted the setup is impressive but questioned how the very low-bit `753B` quant compares on complex reasoning against a smaller higher-precision model such as a `70B` at 4-bit.

    - A commenter questioned whether the reported **`~16 tok/s`** for **GLM-5.2 753B IQ1_S** across **2× M5 Max over Thunderbolt 5** was accurate, noting the video appeared faster; another highlighted that while the throughput is impressive for a `753B` local setup, the very low-bit **IQ1_S** quantization raises the technical question of reasoning quality versus a smaller **`70B` at 4-bit** model.
    - One user provided comparative llama.cpp RPC-style benchmarks using an **M3 Ultra Studio 256GB + M3 Max MBP 128GB** running **GLM-5.2-UD-IQ4_XS**: `13.03 tok/s` at `2,377` context tokens with `TTFT 3.09s`, `8.64 tok/s` at `22,485` context with `TTFT 2.33s`, and `6.21 tok/s` at `32,595` context with `TTFT 5.53s`. They clarified that **TTFT included cache prefill**, making the measurements more comparable for long-context generation.
    - Another commenter asked whether multi-Mac connectivity is already supported in **llama.cpp** or requires a custom driver, pointing to the implementation-level question around whether this setup uses built-in **llama.cpp RPC** capabilities or bespoke Thunderbolt networking/inference orchestration.

  - **[GLM 5.2 Q1_S vs Qwen 27B Q8](https://www.reddit.com/r/LocalLLaMA/comments/1uimjdi/glm_52_q1_s_vs_qwen_27b_q8/)** (Activity: 359): **A hobby `n=1` comparison on dual RTX 3090s found **GLM-5.2 Q1_S** produced a one-shot, polished Three.js arena game in ~`75k` tokens at ~`6→3 t/s`, outperforming **Qwen 3.6 27B Q8**, which needed `1 + 3` prompts and ~`42k` tokens despite ~`60 t/s`; the author later clarified GLM used `K/V Q8` while Qwen used full `FP16` KV cache. LLM-as-judge scores from **Opus 4.8** and **GPT-5.5** both ranked GLM Q1_S highest for code quality/polish, while GLM FP via OpenRouter used only ~`11k` tokens but had a controls bug. Top technical comments noted a likely stronger **GLM-5.2 REAP 504B GGUF `Q2_K_XL`** quant at `211 GB` on [Hugging Face](https://huggingface.co/0xSero/GLM-5.2-REAP-504B-GGUF), asked about OpenRouter cost, and reported **Qwen3.6-27B-UD-Q5_K_XL.gguf MTP** completing a similar playable demo in `2` prompts / ~`11k` tokens at `110–130 t/s`, with output shared on [CodePen](https://codepen.io/source-drifter/pen/MYJvNEb).** The main debate is whether very low quants below Q3 are inherently “braindead”; the post argues that a much larger model at Q1_S can still outperform a smaller high-quant model when long deliberation is acceptable. Comment evidence partially complicates the conclusion by showing a Qwen Q5_K_XL run that was much faster and required only one console-error fix.

    - A commenter points to a larger **GLM-5.2-REAP-504B GGUF** quant on Hugging Face: [0xSero/GLM-5.2-REAP-504B-GGUF](https://huggingface.co/0xSero/GLM-5.2-REAP-504B-GGUF), specifically **`Q2_K_XL` at `211 GB`**, arguing it is likely stronger than the tested **`Q1_S`** quant. This implies the comparison may be heavily affected by quantization quality rather than base-model capability.
    - One user reports local performance for **`Qwen3.6-27B-UD-Q5_K_XL.gguf` with MTP**, producing a playable CodePen demo after an initial prompt plus one console-error fix: [demo](https://codepen.io/source-drifter/pen/MYJvNEb). They measured `5,538` tokens in `50s` (`110.69 tok/s`) for the initial generation and `5,422` tokens in `41s` (`129.88 tok/s`) for the fix pass, with the only reported bug being `Uncaught ReferenceError: time is not defined`.
    - There is a hardware-fit concern around whether the referenced **`211 GB` GLM quant** can run on a **128 GB RAM Strix Halo** system. The implication is that even low-bit quantized frontier-scale GGUFs may exceed unified-memory consumer/workstation configurations once model size plus KV cache and runtime overhead are included.


### 2. llama.cpp Model and Kernel Support Merges

  - **[DFlash support merged into llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1uhx862/dflash_support_merged_into_llamacpp/)** (Activity: 469): ****DFlash support has been merged into `llama.cpp`**, adding official support for diffusion-style text generation in the project, though commenters note **multimodal DFlash is not supported yet**. The merge is viewed as groundwork for future speedups such as **DDTree/JetSpec** and possible separate architecture support for **DSpark**, **Gemma Diffusion**, **Nvidia NemoDiffusion**, **Orthrus**, and potentially **LLaDA-like** models.** Commenters were broadly positive, crediting **Ruixiang63** for sustained work on the feature and joking/anticipating that **DSpark** support should come next.

    - Commenters note that **DFlash support in `llama.cpp` currently excludes multimodal/vision use cases**, so users depending on vision models will not benefit yet. One user also flags practical tradeoffs for trying it with **Qwen3.6-27B on an RTX 5090**, saying current draft-model workflows may require **disabling thinking**, and may lose **vision** and **parallel inference** support.
    - A technical roadmap discussion frames DFlash as one piece of a broader speculative/diffusion acceleration stack: remaining speedups mentioned include **DDTree** and **JetSpec**, while separate architecture support is still needed for **DSpark**, **Gemma Diffusion**, **NVIDIA NemoDiffusion**, **Orthrus**, and possibly **LLaDA-style** models if they remain viable.
    - Users compare DFlash against existing **MTP** experimentation, with one commenter saying they already had MTP working on **Qwen3.6** and **Gemma4** and asking whether the merged DFlash path will provide additional performance improvements beyond that baseline.

  - **[DeepSeek V4, PR merged into llama.cpp !](https://www.reddit.com/r/LocalLLaMA/comments/1uj0fkw/deepseek_v4_pr_merged_into_llamacpp/)** (Activity: 280): **A **DeepSeek V4** support PR has been merged into **llama.cpp** ([ggml-org/llama.cpp#24162](https://github.com/ggml-org/llama.cpp/pull/24162)), so users can update via `git pull`, rebuild with `cmake`, and run compatible **GGUF** model files without relying on a fork. The main technical follow-up is compatibility: commenters ask which **GGUFs** are known to work with upstream `llama.cpp` versus only with third-party forks.** Comments are mostly practical or humorous: one user notes the hardware requirements may keep local DeepSeek V4 inference out of reach for years, while another jokes about wanting a tiny “microflashmini” variant.

    - Commenters focused on **GGUF compatibility** after the DeepSeek V4 `llama.cpp` merge, specifically asking which model files work with upstream/latest `llama.cpp` rather than requiring a fork. There was also interest in **Unsloth** producing “proper GGUF files,” implying current conversion/quantization availability may be fragmented or unofficial.
    - A technically relevant concern was that early performance reports will likely be noisy: users expect many `tokens/s` claims without enough reproducibility details such as GPU/CPU model, quantization level, context length, backend, batch size, or memory configuration.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Agentic Coding Tooling and Safety

  - **[Graphify hit 73k stars and 2.2M downloads in 2.5 months, and we just got into YC](https://www.reddit.com/r/ClaudeAI/comments/1ui6unv/graphify_hit_73k_stars_and_22m_downloads_in_25/)** (Activity: 962): ****Graphify** claims rapid OSS traction since its April 5 launch: `73k` GitHub stars and `2.2M` downloads in ~`2.5` months, plus acceptance into **YC S26**. The tool converts repos/docs/PDFs/SQL schemas/Obsidian vaults/transcripts into a knowledge graph queried by **Claude**, with the author claiming ~`71×` lower token usage per query versus reading raw files; a new `graphify reflect` feature records useful/dead-end answers into `LESSONS.md` as persistent session memory. The stated product direction is an enterprise “self-learning company brain,” with community discussion directed to [Discord](https://discord.gg/598Ad9zQZ).** Top comments were skeptical about defensibility and monetization: users argued the code is free, relatively easy for agents to reproduce, and potentially vulnerable to being subsumed by Anthropic or other LLM vendors. One commenter also disputed the claimed LinkedIn traction, saying visible posts appear mostly spam-like.

    - Several commenters questioned **Graphify’s defensibility and monetization**: since the code is free/open and perceived as *“not that hard for agents to reproduce,”* they argued that the main business risk is commoditization or direct integration by model providers like **Anthropic**.
    - A technically relevant critique compared Graphify’s value against existing developer tooling, especially **LSP-based code intelligence**. One user reported that on a *“pretty large code base”* it was *“fiddly”* to set up and did not noticeably improve output quality or save time versus conventional tooling.
    - One concrete packaging concern was raised: the install command is `pip install graphifyy` with two `y`s, which a commenter said looks suspicious and may create trust/friction issues for Python users installing the package.

  - **[Claude Code suddenly tried to open a Remote Desktop connection on my PC. This seriously scared me.](https://www.reddit.com/r/ClaudeAI/comments/1ui8g1t/claude_code_suddenly_tried_to_open_a_remote/)** (Activity: 937): **The image ([Windows RDP warning dialog](https://i.redd.it/zkcjmfu263ah1.png)) shows **Windows 11 prompting to open an `.rdp` Remote Desktop Connection file**, not necessarily an inbound remote-control takeover. In context of the title and selftext, the user reports this appeared while using **Claude Code**, followed by apparent automated File Explorer navigation; the most plausible technical concern raised in comments is that Claude or a tool/MCP workflow may have opened or generated an RDP file, potentially via prompt injection or unsafe permissions, rather than **Anthropic** directly “taking over” the machine.** Commenters were skeptical of the user’s theory that Anthropic staff were being handed the session, with one noting that an RDP file means the local machine is trying to connect outward and may expose clipboard/drives depending on settings. The main safety advice was to avoid broad permissions/`dangerously-skip-permissions`, use Claude Code [auto mode](https://code.claude.com/docs/en/auto-mode-config), disable computer-use style capabilities, or run agents inside a sandboxed VM/WSL environment.

    - One technical explanation argues the visible warning likely came from the user opening an `.rdp` file, meaning the machine was initiating an **outbound Remote Desktop connection** to another host rather than Anthropic remotely controlling the PC. The risk would come from RDP redirection options such as clipboard, audio, ports, or drive sharing, especially if a compromised `.rdp` file was introduced via prompt injection or unsafe automation settings.
    - A safety-focused thread recommends avoiding `--dangerously-skip-permissions` and using Claude Code’s [**auto mode**](https://code.claude.com/docs/en/auto-mode-config) as a safer-but-not-perfect alternative, plus disabling “computer use.” For stronger isolation, commenters suggest running Claude Code inside a Linux VM/WSL environment with no access to sensitive host files or devices.
    - Several commenters note that the user should inspect Claude Code’s session trace because Claude Code exposes its reasoning/actions. Suggested recovery steps include resuming the prior session from the same directory with `claude --resume` and asking what triggered the RDP launch, or using `/btw` to query without continuing the same action path. One commenter also argues that the screenshot indicates an attempted outbound RDP launch, while claims of a tiny remote-controlled File Explorer window would imply a separate compromise or script rather than normal RDP behavior.


### 2. AI in Physical Interfaces and Robotics

  - **[Meta improves Brain2QWERTY, a system that can decode text from brain activity to enable typing using non-invasive technologies, MEG and EEG](https://www.reddit.com/r/singularity/comments/1uisr5i/meta_improves_brain2qwerty_a_system_that_can/)** (Activity: 808): ****Meta** reportedly improved **Brain2QWERTY**, a non-invasive brain-to-text system intended to decode typed text from brain activity using **MEG** and **EEG**, but the linked Reddit-hosted video/article is inaccessible due to a `403 Forbidden` block, so no benchmark numbers, architecture details, dataset description, or error-rate comparisons are available from the source. The only technical artifact in the comments is an image link, but its content is not described in the provided data.** Comment discussion is mostly speculative: one user jokes about future “Ad2Brain” applications, while another raises a relevant cognitive-neuroscience question about whether decoding depends on an internal monologue or other language-production signals.


  - **[Meanwhile in China, 10,000+ delivery bots are transforming last-mile fulfillment by making deliveries faster, cheaper, and more autonomous](https://www.reddit.com/r/singularity/comments/1uhxshz/meanwhile_in_china_10000_delivery_bots_are/)** (Activity: 2715): **A Reddit post claims **China has deployed `10,000+` autonomous delivery robots** for last-mile logistics, implying lower-cost and faster fulfillment via sidewalk/road-edge robotic delivery; however, the linked Reddit video ([v.redd.it/ub2ct1a731ah1](https://v.redd.it/ub2ct1a731ah1)) was not accessible due to **403 Forbidden**, so no technical details such as vehicle model, autonomy stack, payload, routing, or fleet operator could be verified. The most relevant technical question in comments concerns the unresolved “last `50 m/yd`” handoff problem: whether a truck/robot stops curbside and how the package is transferred from road edge to recipient.** Commenters contrasted deployment feasibility with vandalism risk in other markets, citing UK delivery robots allegedly having antennas ripped off, and joked about dystopian misuse; no substantive technical debate was present.

    - One commenter raised the key last-mile robotics implementation question: how these delivery bots handle the **final `50m/50yd` handoff** after autonomous street-level transport—e.g., whether a truck or bot drops packages curbside, approaches the door, or requires customer pickup at the road edge. This points to unresolved operational details around curb-to-door navigation, secure package release, and human interaction at delivery completion.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
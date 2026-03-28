---
id: MjAyNS0x
title: not much happened today
date: '2026-03-27T05:44:39.731046Z'
description: >-
  **Anthropic** is reportedly introducing a new AI model tier called
  **Capybara**, which is larger and more intelligent than **Claude Opus 4.6**,
  showing improved performance in coding, academic reasoning, and cybersecurity.
  The model is speculated to be around **10 trillion parameters**, with
  **Google** potentially funding Anthropic's data center expansion. Meanwhile,
  **Zhipu** released **GLM-5.1**, advancing open coding models and narrowing the
  gap with closed models. Local inference economics are improving, highlighted
  by efficient deployments of **Qwen 3.5 14B**, **Qwen 27B**, and
  **Qwen3.5-35B** models with quantization techniques like **TurboQuant vLLM**.
  However, TurboQuant's benchmarking claims face criticism from researchers.
  Overall, the AI landscape shows aggressive scaling, local model deployment,
  and agent products gaining traction.
companies:
  - anthropic
  - google
  - zhipu
models:
  - claude-opus-4.6
  - capybara
  - glm-5.1
  - qwen-3.5-14b
  - qwen-27b
  - qwen3.5-35b
topics:
  - model-scaling
  - coding
  - academic-reasoning
  - cybersecurity
  - quantization
  - local-inference
  - model-benchmarking
  - inference-optimization
  - model-performance
  - agent-products
people:
  - scaling01
  - yuchenj_uw
  - kimmonismus
  - m1astra
  - dejavucoder
  - iscienceluvr
  - gaoj0017
---


**a quiet day.**

> AI News for 3/26/2026-3/27/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Anthropic’s leaked “Mythos” system and the new Capybara tier**

- **Fortune corroborates a higher Anthropic tier above Opus**: A now-pulled “Claude Mythos” post was preserved by [@M1Astra](https://x.com/M1Astra/status/2037377109472018444), and multiple follow-on posts cite a Fortune report that Anthropic is introducing **Capybara**, described as a new tier **above Opus** and “larger and more intelligent” than **Claude Opus 4.6**. Reporting summarized by [@scaling01](https://x.com/scaling01/status/2037379145806524655), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2037387996694200509), and [@kimmonismus](https://x.com/kimmonismus/status/2037463638261305752) says Capybara posts substantially better scores on **coding, academic reasoning, and cybersecurity**, with rollout constrained by cost and safety concerns.
- **Compute intensity is the central theme**: Several posters infer Anthropic is leaning hard into scale, with speculation around a **~10T parameter** class model from prior Dario comments, though that remains unconfirmed outside commentary; see [@scaling01](https://x.com/scaling01/status/2037384912743923969) and [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2037391159115563214). Separately, the Financial Times report relayed by [@FirstSquawk](https://x.com/FirstSquawk/status/2037586926375743904) says **Google is close to funding Anthropic’s data center**, reinforcing that frontier competition is increasingly gated by power and capex rather than just algorithms.
- **Infra strain was visible in production**: The leak landed amid a rough day for Anthropic availability, with widespread user complaints about **529s/elevated errors** from [@dejavucoder](https://x.com/dejavucoder/status/2037439287873159641), [@iScienceLuvr](https://x.com/iScienceLuvr/status/2037487244634972471), and others. The practical takeaway is that Anthropic appears to be balancing aggressive scaling ambitions against a still-tight serving envelope.

**Open coding models, local inference, and GLM-5.1’s continued push**

- **GLM-5.1 is widening the pressure on closed coding models**: Zhipu announced **GLM-5.1** availability to all coding plan users via [@Zai_org](https://x.com/Zai_org/status/2037490078126084514), along with docs for agent use at [@Zai_org](https://x.com/Zai_org/status/2037506911013138851). Community reaction framed it as another sign that high-end Chinese open or semi-open coding models are closing the gap: [@kimmonismus](https://x.com/kimmonismus/status/2037507667732709392), [@XFreeze](https://x.com/XFreeze/status/2037695882301436412), and Arena’s broader leaderboard analysis [@arena](https://x.com/arena/status/2037584085997216100) all point to a much narrower open-vs-closed gap than a year ago.
- **Local deployment economics keep improving**: A recurring theme across tweets is that local models are now “good enough” for many workflows. Examples include [@TheGeorgePu](https://x.com/TheGeorgePu/status/2037473248577782046) swapping a pricey TTS subscription for a local **Qwen 3.5 14B** setup, [@LottoLabs](https://x.com/LottoLabs/status/2037557925015949676) reporting strong economics for **Qwen 27B** with Hermes Agent, and [@0xSero](https://x.com/0xSero/status/2037560787565252666) compressing **Qwen3.5-35B** enough to fit full context into **24GB VRAM** at roughly **1% average performance drop**.
- **Quantization and cache work remain key enablers**: [@iotcoi](https://x.com/iotcoi/status/2037478891179135123) shipped a **TurboQuant vLLM** fork with fused Triton KV write paths and decode attention, targeting **Qwen3.5-35B AWQ**, **1M context**, and **4M KV cache**. Meanwhile [@bnjmn_marie](https://x.com/bnjmn_marie/status/2037564190802563157) benchmarked Qwen3.5 27B formats across **RTX Pro 6000/B200/H100**, with **INT4** emerging as the best inference option on RTX Pro 6000-class hardware.
- **But TurboQuant is now under active dispute**: The strongest research controversy in the set comes from [@gaoj0017](https://x.com/gaoj0017/status/2037532673812443214) and a longer clarification [@gaoj0017](https://x.com/gaoj0017/status/2037552350924042488), alleging Google’s **ICLR 2026 TurboQuant** paper misrepresented **RaBitQ** in theory and benchmarking, including unfair CPU-vs-GPU comparisons. This does not invalidate TurboQuant’s engineering value, but it does cast doubt on some of the publicized comparative claims.

**Agents are becoming products, not demos**

- **Hermes Agent is emerging as the open-agent focal point**: The most consistent product momentum in the dataset belongs to **Nous Research’s Hermes Agent**. [@NousResearch](https://x.com/NousResearch/status/2037654827929338324) integrated **Hugging Face** as a first-class inference provider with **28 curated models** plus access to many more, while [@ClementDelangue](https://x.com/ClementDelangue/status/2037634211973140898) framed this as a step toward open agents with memory, persistent machine access, and model choice. User reports from [@fancylancer3991](https://x.com/fancylancer3991/status/2037579517389144399), [@PolackJack](https://x.com/PolackJack/status/2037661357785690584), and [@alexcovo_eth](https://x.com/alexcovo_eth/status/2037589212648665273) emphasize lower friction and better persistence than browser-automation-heavy setups like OpenClaw.
- **Agent infrastructure is maturing around traces, evals, and debuggability**: Hugging Face’s [@ClementDelangue](https://x.com/ClementDelangue/status/2037530125638455610) called for **open agent traces datasets**, with follow-up pointing to the **Agent Data Protocol** from [@yueqi_song](https://x.com/yueqi_song/status/2037614951230296230). LangChain pushed a cluster of production-oriented materials: an **agent eval readiness checklist** [@LangChain](https://x.com/LangChain/status/2037590936234959355), **Deep Agents** IDE-style UI guidance [@LangChain_JS](https://x.com/LangChain_JS/status/2037560951445266891), and **LangSmith Prompt Hub Environments** for prompt promotion/rollback [@LangChain](https://x.com/LangChain/status/2037666098561032421). The direction is clear: the stack is moving from “chatbot with tools” to software lifecycle primitives for agents.
- **Agent-facing benchmarks are starting to reflect real workloads**: Artificial Analysis introduced **AA-AgentPerf** via [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2037562417836929315), focused on **real coding-agent trajectories**, **100K+ sequence lengths**, and throughput expressed as **concurrent users per accelerator / per kW / per $ / per rack**. That is a more deployment-relevant abstraction than synthetic token benchmarks and should be useful for teams comparing accelerator systems for agent-heavy serving.

**Coding agents, Codex plugins, and multi-agent software workflows**

- **OpenAI’s Codex ecosystem is shifting toward workspace-native automation**: OpenAI developers highlighted **Codex plugins** and a use-case gallery via [@OpenAIDevs](https://x.com/OpenAIDevs/status/2037604273434018259), while Box shipped a Codex plugin for automating workflows over Box content [@Box](https://x.com/Box/status/2037563341431058497). User sentiment from [@theo](https://x.com/theo/status/2037383187849183457), [@nickbaumann_](https://x.com/nickbaumann_/status/2037395162641686813), and [@reach_vb](https://x.com/reach_vb/status/2037614060452106437) suggests the center of gravity is moving from prompt/response to **persistent workspaces, issue systems, terminals, PR flows, and plugins**.
- **The winning UX pattern is increasingly “fleet management for software”**: [@VibeMarketer_](https://x.com/VibeMarketer_/status/2037521519736463782) captured the emerging pattern well: kanban-like cards, isolated worktrees, agent-owned tasks, and diff-based review. Related tools include the new **agent-browser dashboard** from [@ctatedev](https://x.com/ctatedev/status/2037599050112160165) for real-time browser session debugging, and broad enthusiasm for multi-agent SWE systems from Cognition/Devin adjacent commentary like [@JTLonsdale](https://x.com/JTLonsdale/status/2037555800193851727) and [@cognition](https://x.com/cognition/status/2037649026951303668).
- **Composer 2 and long-horizon coding evals are raising the bar**: The CursorBench discussion is mostly indirect here, but [@cwolferesearch](https://x.com/cwolferesearch/status/2037726856699420987) points out the benchmark’s strengths: **real coding sessions**, **underspecified prompts**, broader quality dimensions, and median **181 lines changed** per task. That’s a healthier benchmark design than static toy tasks and aligns with the broader turn toward long-horizon agent evaluation.

**Research and systems: world models, robotics, speech, and multimodal infra**

- **Meta shipped a practical SAM 3.1 speedup**: [@AIatMeta](https://x.com/AIatMeta/status/2037582117375553924) released **SAM 3.1**, a drop-in update to SAM 3 with **object multiplexing**, allowing up to **16 objects in a single forward pass**. Meta says this roughly doubles video throughput from **16 to 32 FPS on one H100** for medium-object workloads, which is meaningful for accessible video segmentation pipelines.
- **World models and robotics both had notable open releases**: [@LiorOnAI](https://x.com/LiorOnAI/status/2037484990779339064) highlighted LeCun’s **LeWorldModel** paper/repo as a small, open world model designed to make representational collapse mathematically impossible via **SIGReg**, claiming **48x faster planning** and **~200x fewer tokens**. On robotics data, [@UnitreeRobotics](https://x.com/UnitreeRobotics/status/2037440578275946551) open-sourced the **UnifoLM-WBT-Dataset**, a real-world humanoid whole-body teleoperation dataset intended for rolling updates.
- **Speech/open audio remains one of the healthiest open categories**: Cohere’s new **2B Apache-2.0 Transcribe** model drew strong praise from [@victormustar](https://x.com/victormustar/status/2037572662659104976) and throughput measurements from [@vanstriendaniel](https://x.com/vanstriendaniel/status/2037548103272632497), who reports **33 hours** of audio transcribed in **12 minutes** on an A100. Mistral’s **Voxtral TTS** paper was flagged by [@qtnx_](https://x.com/qtnx_/status/2037553397423902846), and browser/local demos appeared from [@sophiamyang](https://x.com/sophiamyang/status/2037523809914241069) and [@nickfrosst](https://x.com/nickfrosst/status/2037680223445975131#m).
- **Open robotics stacks are also getting more reproducible**: AI2 released **MolmoBot**, an open robotic manipulation suite trained entirely in simulation, with **code, training data, generation pipeline, and evals** available via [@allen_ai](https://x.com/allen_ai/status/2037590611990094259). That complements the Unitree dataset and signals continued progress toward replicable robotics research outside top labs.

**Top tweets (by engagement)**

- **Anthropic/Capybara leak**: [@Yuchenj_UW on Capybara](https://x.com/Yuchenj_UW/status/2037387996694200509) was the most engaged technical item, summarizing the new tier above Opus and its reported benchmark gains.
- **Paul Conyngham’s AI-assisted dog cancer treatment**: [@sama](https://x.com/sama/status/2037396826060673188) shared a story of using ChatGPT and related tools to help design an **mRNA vaccine protocol** for a dog’s cancer, which became a major discussion point about AI-enabled personalized medicine.
- **TurboQuant critique**: [@gaoj0017](https://x.com/gaoj0017/status/2037532673812443214) drew unusually high engagement for a paper-methodology dispute, likely because it challenges a heavily promoted systems paper.
- **GLM-5.1 release**: [@Zai_org](https://x.com/Zai_org/status/2037490078126084514) announcing broad GLM-5.1 availability landed strongly, reinforcing sustained interest in open coding models.
- **Open infrastructure for agents**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2037604273434018259) on Codex plugins and [@NousResearch](https://x.com/NousResearch/status/2037654827929338324) on Hugging Face integration into Hermes Agent were the clearest product/infrastructure launches with broad developer relevance.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. TurboQuant and RotorQuant Innovations

  - **[Google TurboQuant running Qwen Locally on MacAir](https://www.reddit.com/r/LocalLLaMA/comments/1s5kdu0/google_turboquant_running_qwen_locally_on_macair/)** (Activity: 433): **The post discusses an experiment where **Google's TurboQuant compression method** was applied to `llama.cpp`, enabling the running of **Qwen 3.5–9B** on a standard MacBook Air (M4, 16 GB) with a `20000 tokens` context. This was previously unfeasible on such hardware, highlighting TurboQuant's potential to enable local execution of large models without cloud APIs. The experiment suggests that even entry-level devices like MacBook Airs or Mac Minis can handle large contexts, albeit with some speed limitations. The open-source app [atomic.chat](http://atomic.chat/) is mentioned as a resource for running these models locally.** A commenter notes the impressive feat of handling `20K context` on a base MacBook Air without swapping, suggesting potential for local use cases that previously relied on cloud APIs. Another commenter inquires about the integration of TurboQuant into `llama.cpp`, indicating interest in broader accessibility.

    - **Tatrions** highlights the impressive capability of running a 20K context model on a base MacBook Air with 16GB RAM without swapping, thanks to TurboQuant. This suggests that many applications that previously relied on cloud APIs could now be executed locally, though there is curiosity about the quality degradation at this compression level compared to standard Q4 on the same model.
    - **M5_Maxxx** provides a detailed audit of the TurboQuant implementation, revealing it as a minimally altered version of [Jan.ai](http://Jan.ai). Key changes include renaming, UI tweaks, and a custom `llama.cpp` backend fork, but no new inference engine or model architecture support. The 96 commits mostly involve CI/build pipeline changes, suggesting limited innovation beyond the original Jan.ai capabilities.
    - **AppealThink1733** inquires about the integration of TurboQuant into `llama.cpp`, indicating interest in whether this technology is already supported by the popular open-source project, which could facilitate broader adoption and experimentation.

  - **[Skipping 90% of KV dequant work → +22.8% decode at 32K (llama.cpp, TurboQuant)](https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/skipping_90_of_kv_dequant_work_228_decode_at_32k/)** (Activity: 744): **The post discusses an optimization in the `TurboQuant` implementation for KV cache compression in `llama.cpp`, which significantly improves decode performance by skipping dequantization for positions with negligible attention weights. This approach leverages attention sparsity, allowing a `+22.8%` increase in decode speed at `32K` context length on an `M5 Max`, without affecting perplexity (PPL). The method involves a simple modification of about three lines in the kernel, bypassing the need for complex optimizations like SIMD tricks or fused kernels. The results are consistent across different hardware, including the `M2 Pro`, where performance improved from `~0.45x` to `~0.73x` compared to the standard `q8_0` KV cache. The implementation and benchmarks are available on [GitHub](https://github.com/TheTom/turboquant_plus), with a detailed [writeup](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md).** Commenters praised the simplicity and effectiveness of the solution, noting the innovative use of attention sparsity to skip unnecessary computations. There is curiosity about how this approach scales with even longer contexts, such as `64K+`, and interest in integrating this optimization into the mainline `llama.cpp`.

    - Specialist_Sun_7819 highlights a novel optimization in llama.cpp's TurboQuant, where skipping 90% of the key-value dequantization work for tokens that don't significantly impact the output leads to a `+22.8%` increase in decoding speed at `32K` context length. This approach leverages predictable attention sparsity in long contexts, allowing for significant computational savings with minimal code changes, specifically just three lines in the kernel. The commenter is curious about the scalability of this method to even longer contexts, such as `64K`, and whether the sparsity ratio continues to increase or plateaus.
    - sean_hash draws a parallel between the optimization in TurboQuant and techniques used in Flash Attention, noting that caching the dequantized output instead of recalculating it at each decoding step is a similar strategy. This method effectively reduces redundant computations, enhancing performance by reusing previously computed values, which is a common optimization in high-performance computing to minimize unnecessary processing overhead.
    - Pentium95 expresses interest in integrating this optimization into the mainline llama.cpp, indicating a desire for broader adoption of this technique. This suggests that the community sees value in these performance improvements and is eager to see them implemented in widely-used codebases, potentially leading to more efficient models and faster inference times across various applications.

  - **[TurboQuant in Llama.cpp benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1s4bzo2/turboquant_in_llamacpp_benchmarks/)** (Activity: 463): **The post discusses the implementation of **TurboQuant**, a compression technique from Google, in the `llama.cpp` framework, specifically on Apple Silicon using Metal. The author notes a significant performance drop, with TPS being `50%` less than `f16`, indicating potential issues in their setup. They also attempted to run kernels on a CUDA machine but encountered poor outputs, suggesting errors in their approach. The technique is seen as beneficial for running local models on consumer hardware with limited VRAM, potentially allowing for more complex tasks to be executed locally. The post references ongoing development efforts in related projects like **MLX** and **VLLM**.** Commenters suggest checking KLD to evaluate the method's worth and express interest in seeing performance metrics like pp2048, as pp64 is not very indicative. Another commenter recommends trying RotorQuant for comparison.

    - Velocita84 points out the absence of Kullback-Leibler Divergence (KLD) in the benchmarks, which is crucial for evaluating the effectiveness of TurboQuant. KLD is a measure of how one probability distribution diverges from a second, expected probability distribution, and its absence could mean missing insights into the model's performance under TurboQuant compression.
    - CornerLimits suggests that the benchmark using `pp64` is not very informative for assessing performance and recommends using `pp2048` instead. The `pp` metric refers to perplexity, a common measure in language models that indicates how well a probability distribution predicts a sample. Higher `pp` values can provide a more comprehensive view of model performance.
    - DinoAmino discusses the trade-off between data compression and accuracy in TurboQuant, noting that while it allows for higher data compression with near-lossless accuracy, it doesn't improve accuracy. They highlight that most large language models (LLMs) experience accuracy degradation at higher context lengths, implying that TurboQuant's main benefit is enabling the use of longer contexts without additional accuracy loss.

  - **[RotorQuant: 10-19x faster alternative to TurboQuant via Clifford rotors (44x fewer params)](https://www.reddit.com/r/LocalLLaMA/comments/1s44p77/rotorquant_1019x_faster_alternative_to_turboquant/)** (Activity: 652): ****RotorQuant** introduces a novel approach to vector quantization by utilizing Clifford Algebra, achieving `10-19x` speed improvements over **TurboQuant** with `44x` fewer parameters. The method replaces the `d×d` random orthogonal matrix with Clifford rotors, reducing the computational complexity from `16,384` FMAs to approximately `100` FMAs for `d=128`. This results in a cosine similarity of `0.990` compared to TurboQuant's `0.991`, indicating nearly identical performance. The implementation leverages fused CUDA kernels and Metal shaders, significantly outperforming cuBLAS matmul on RTX PRO 4000 and Apple M4. The trade-off involves higher synthetic MSE on random unit vectors, but with QJL correction, real-model attention fidelity remains intact. [GitHub](https://github.com/scrya-com/rotorquant) [Paper](https://www.scrya.com/rotorquant/)** A key debate centers on the theoretical differences between RotorQuant and TurboQuant. While TurboQuant's global random rotation spreads energy across all dimensions, RotorQuant's 3D block mixing cannot replicate this, leading to higher max coordinate magnitudes and worse MSE in low-bit quantization. However, RotorQuant's practical performance in KV cache distributions is acknowledged, suggesting a valuable speed/quality tradeoff for real models.

    - Juan_Valadez highlights a key theoretical limitation of RotorQuant compared to TurboQuant, noting that TurboQuant's global random rotation (Haar) effectively spreads energy across all dimensions, optimizing scalar quantization. In contrast, RotorQuant's mixing within 3D blocks limits its ability to achieve the same energy distribution, which can negatively impact low-bit quantization, especially in worst-case vectors like one-hot. However, RotorQuant may still be practically useful for KV cache distributions where vectors are less adversarial.
    - Dany0 draws parallels between TurboQuant and techniques used in graphics programming, specifically referencing QuiP, a similar approach applied to model weights. Despite initial skepticism due to the shortness of the paper and its presentation, Dany0 acknowledges the potential of RotorQuant, likening its use of Clifford rotors to the application of quaternions instead of Euler angles, which simplifies computations by reducing multiplications to zeros.
    - sean_hash comments on the unexpected application of Clifford algebras in quantization, noting it as an example of cross-pollination from geometric algebra into fields outside of graphics. This highlights the innovative use of mathematical concepts traditionally associated with other domains, suggesting a broader applicability of these techniques.




### 2. GLM-5.1 and Coding Model Comparisons

  - **[Glm 5.1 is out](https://www.reddit.com/r/LocalLLaMA/comments/1s51id3/glm_51_is_out/)** (Activity: 1127): **The image announces the release of **GLM-5.1** by Z.ai, highlighting its improved performance in coding tasks compared to previous versions. The chart in the image shows that GLM-5.1 scores `45.3` in coding evaluation, surpassing GLM-5's score of `35.4`, but still trailing behind Claude Opus 4.6, which scores `47.9`. This suggests significant improvements in GLM-5.1's capabilities, likely due to enhancements in its underlying architecture or training data.** Commenters speculate about the potential release of open weights for GLM-5.1, indicating anticipation for broader accessibility. There is also discussion about the delay in the release of DS v4, hinting at possible challenges in training on specific hardware like Ascends.

    - power97992 speculates on potential delays in the release of DeepSpeed v4, suggesting that there might be issues related to training on Ascend hardware. This highlights the challenges in optimizing machine learning frameworks for different hardware architectures, which can impact release timelines.
    - zb-mrx notes the improvement in the rollout process for GLM 5.1, contrasting it with the previous version, GLM 5, which did not have a day-one rollout for everyone. This suggests that the developers may have resolved previous logistical or resource-related issues, such as GPU availability, to ensure a smoother release.
    - jacek2023 mentions the limitations of running GLM locally due to hardware constraints, specifically referencing a 72GB VRAM limit. This underscores the ongoing challenge of hardware requirements for running advanced models, which can be a barrier for many users without access to high-end GPUs.


### 3. Local LLM Hardware Setups and Comparisons

  - **[Dual DGX Sparks vs Mac Studio M3 Ultra 512GB: Running Qwen3.5 397B locally on both. Here's what I found.](https://www.reddit.com/r/LocalLLaMA/comments/1s4lmep/dual_dgx_sparks_vs_mac_studio_m3_ultra_512gb/)** (Activity: 819): **The post compares the performance of a **Mac Studio M3 Ultra 512GB** and a **dual DGX Spark setup** for running the **Qwen3.5 397B** model locally. The Mac Studio, utilizing `MLX 6 bit quantization`, achieves `30 to 40 tok/s` generation speed with a memory bandwidth of `~800 GB/s`, but suffers from slow prefill times and requires a custom async proxy for tool calls. In contrast, the dual DGX Spark setup, using `INT4 AutoRound quantization`, achieves `27 to 28 tok/s` with faster prefill and batch embedding due to CUDA tensor cores, but faces challenges with setup complexity, memory bandwidth (`~273 GB/s per node`), and stability issues. The author uses both setups for different tasks: the Mac Studio for inference and the Sparks for RAG and embedding, communicating over Tailscale. The cost of each setup is approximately `$10K`, with a break-even point of 10 months compared to a `$2K/month` API spend.** Comments highlight the uniqueness of the Mac Studio 512GB and criticize Nvidia's support for DGX. There is also a discussion on the performance of Qwen3.5 397B compared to Claude, noting that while Qwen3.5 is not as advanced as Claude's Opus, it is close in performance.

    - Repoman444 highlights a significant issue with the **Nvidia DGX** systems, noting that the support from Nvidia is subpar. This could impact users who rely on timely and effective support for troubleshooting and optimizing their high-performance computing tasks, especially when running large models like Qwen3.5 397B.
    - sp4_dayz discusses the performance of **Qwen3.5 397B** in comparison to **Claude** and **Opus**, suggesting that while Qwen3.5 is not yet at the level of Opus, it is quite close. This implies that users familiar with Claude might find Qwen3.5 slightly lacking but still a strong contender in terms of performance.
    - Gringe8 raises a technical point about the comparison methodology, questioning whether the evaluation included prompt processing speed. This suggests that prompt processing speed is a critical factor in assessing the performance of AI models like Qwen3.5 397B, especially when comparing across different hardware setups like the DGX and Mac Studio M3 Ultra.

  - **[If you had ~10k to spend on local LLM hardware right now, what would you actually build?](https://www.reddit.com/r/LocalLLM/comments/1s40wgj/if_you_had_10k_to_spend_on_local_llm_hardware/)** (Activity: 201): **The post discusses building a local hardware setup for running large language models (LLMs) with a budget of `~$10k`. The user aims to run models of at least `30B` parameters, ideally up to `70B`, for tasks beyond simple chat, such as multi-step workflows and tools, with a focus on privacy and avoiding API costs. The main technical debate is around GPU choices: the **RTX 4090** is considered for its performance, while used **A6000/A40** GPUs are noted for their VRAM capacity. The user also considers a **Mac Studio (M3 Ultra)** for its unified memory, questioning its real-world performance against CUDA setups. The post seeks advice on balancing GPU, CPU, RAM, and storage investments for optimal performance without compromising speed or reliability.** Commenters suggest considering the **RTX 6000 Blackwell** or a **Mac Studio** as viable options. One commenter humorously suggests using the budget to earn interest and pay for LLM subscriptions, highlighting the cost-effectiveness of cloud solutions despite the user's preference for local setups.

    - Blackdragon1400 emphasizes the importance of having at least `256GB of VRAM/Unified memory` for local LLM hardware, suggesting that anything less is inadequate. They recommend using `2x DGX Sparks`, which can run `Qwen3.5-122b-Int4-Autoround` at approximately `40t/s`, highlighting its efficiency over state-of-the-art models.
    - MatthiasWM mentions the potential release of the `M5 Ultra` chip by Apple at an upcoming developer event in June. They suggest waiting for this release before making a significant investment in local LLM hardware, indicating that the new chip could offer substantial improvements.
    - Blackdragon1400 also advises prioritizing large amounts of RAM for LLM tasks, cautioning against settling for quantized models that merely "fit" into smaller memory configurations. This underscores the need for robust hardware to handle demanding LLM workloads effectively.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
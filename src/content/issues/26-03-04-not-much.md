---
id: MjAyNi0w
title: FILL TITLE IN HERE
date: '2026-03-04T05:44:39.731046Z'
---

**TODO: ONELINE SUBTITLE**

> AI News for 3/3/2026-3/4/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**264** channels, and **14242** messages) for you. Estimated reading time saved (at 200wpm): **1397** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Frontier model shipping: Gemini 3.1 Flash-Lite, GPT-5.4 rumors, and “agent-first” product positioning**

- **Gemini 3.1 Flash-Lite positioning (speed/$)**: Demis Hassabis teased **Gemini 3.1 Flash-Lite** as “incredibly fast and cost-efficient” for its performance—clearly framing the model line around latency and cost per capability rather than raw frontier scores ([tweet](https://x.com/demishassabis/status/2029047252275060895)). Related product chatter highlights **NotebookLM** as a “favorite AI tool” ([tweet](https://x.com/demishassabis/status/2029355691933085731)) and a major new **NotebookLM Studio** feature: **Cinematic Video Overviews** that generate bespoke, immersive videos from user sources for Ultra users ([tweet](https://x.com/NotebookLM/status/2029240601334436080)).
- **GPT-5.4 leak narrative (The Information)**: Multiple tweets amplify a report that **GPT-5.4** is coming with a **~1M token context window** and a new **“extreme reasoning mode”** that can “think for hours,” targeting long-horizon agentic workflows and lower complex-task error rates ([tweet](https://x.com/kimmonismus/status/2029213568155992425), [tweet](https://x.com/steph_palazzolo/status/2029212039760023941), [tweet](https://x.com/scaling01/status/2029215437922169254)). There’s also speculation that OpenAI is shifting to **more frequent (monthly) model updates** ([tweet](https://x.com/kimmonismus/status/2029223828677599244)). Separately, one arena watcher claims “GPT-5.4 landed in the arena,” implying an imminent release window ([tweet](https://x.com/kimmonismus/status/2029325405212070200)). Treat all of this as **unconfirmed** unless corroborated by OpenAI.
- **Claude as “agent behavior” leader, not just coding**: Nat Lambert argues the discussion should shift from Anthropic “going all-in on code” to their lead on **general agent behavior**, implying coding capability will commoditize but agent robustness will not ([tweet](https://x.com/natolambert/status/2029212769648836806)). MathArena evaluation adds a datapoint: **Claude Opus 4.6** is strong overall but weak on **visual mathematics**, and costly to evaluate (claimed ~$8k) ([tweet](https://x.com/j_dekoninck/status/2029160582687985727)).

---

**Alibaba Qwen “shakeup”: org design, compute access, and open-model dependency**

- **Leadership exits + restructuring claims**: A central thread across the dataset is that **Qwen’s lead Lin Junyang** stepped down amid an alleged internal restructuring that moves from a vertically integrated team toward **horizontal splits** (pretraining/post-training/multimodal/infra), reducing unified control and potentially conflicting with the team’s previous “tight integration” philosophy ([tweet](https://x.com/ZhihuFrontier/status/2029117410259993073), plus follow-on context [tweet](https://x.com/ZhihuFrontier/status/2029120535599431797)). Simon Willison summarizes the situation and notes multiple apparent resignations around the release of **Qwen 3.5** ([tweet](https://x.com/simonw/status/2029223704127828386)).
- **Emergency all-hands and “compute irony”**: Reporting relayed via Poe Zhao describes Alibaba CEO **Eddie Wu** holding an emergency meeting; Qwen team members challenged leadership on **restructuring, compute allocation, and model strategy**. The most pointed detail: Alibaba Cloud’s CTO allegedly acknowledged that **external customers had smoother access to compute than the internal Qwen team** ([tweet](https://x.com/poezhao0605/status/2029151951167078454)). This triggered reassessment among observers who assumed Qwen was “GPU-rich” ([tweet](https://x.com/teortaxesTex/status/2029159237729894727)).
- **How dominant Qwen is in research workflows**: One claim (from a tweet summarizing HF paper usage) is that Qwen is the **#1 open model in 2025–2026 HF papers**, used in **41%** of 7,692 papers, and **~50%** in May 2025 around Qwen3’s release ([tweet](https://x.com/teortaxesTex/status/2029102932604375057)). Whether exact numbers hold or not, the meta-point stands: **ecosystem dependence on a small core team** is a real risk.
- **Open-weights existential risk framing**: Nat Lambert argues open-weight frontier efforts may concentrate into only a few actors with business incentives: **non-profits, NVIDIA (hardware pull-through), and Meta (commoditize complements)**—a lens that makes Qwen’s corporate-strategy misalignment feel structurally likely, not anomalous ([tweet](https://x.com/natolambert/status/2029049751472357631)).
- **Model/infra technical notes from the Qwen orbit**: RASBT notes **Gated DeltaNet modules** can avoid KV-cache growth, making **Qwen 3.5** more memory-friendly than Qwen3 under a claimed ratio ([tweet](https://x.com/rasbt/status/2029233742708130265)). Meanwhile, users report **llama.cpp doom loops** with Qwen sampling params at ~20% context even at higher quants ([tweet](https://x.com/qtnx_/status/2029246416342618321))—a reminder that “recommended decoding” can be brittle across runtimes.

---

**Inference & systems: Speculative Speculative Decoding, vLLM scaling, and kernel-generation agents**

- **Speculative Speculative Decoding (SSD)**: Tanishq Kumar introduces **SSD**, claiming **up to 2× faster** than leading inference engines (**vLLM, SGLang**), collaborating with Tri Dao and Avner May ([tweet](https://x.com/tanishqkumar07/status/2029251146196631872); Avner’s announcement [tweet](https://x.com/avnermay/status/2029251985934041232)). Tri Dao frames it as “attack of the asynchronous machines,” tying the approach to lessons from GPU kernel async design ([tweet](https://x.com/tri_dao/status/2029273056364118407)). If validated, this is one of the more concrete algorithmic “speed” stories in the set.
- **Production inference pragmatics**: A practical guide is shared on scaling **vLLM** under OOM/instability—emphasizing **workload profiling + tuned configs** over raw hardware ([tweet](https://x.com/DylanCouzon/status/2029208629312700592)).
- **Agentic RL for CUDA kernels (ByteDance)**: A ByteDance paper summary describes **CUDA Agent**: an agentic RL setup that writes CUDA kernels in a secure test environment, optimizing for speedups vs baselines; claims include up to **~100% faster** components than traditional automated tools in some cases ([tweet](https://x.com/rohanpaul_ai/status/2029161433519567175)). Even allowing for “thread summary inflation,” the research direction—**closed-loop code→benchmark→reward** for performance engineering—is credible and strategically important.

---

**Coding agents & dev tooling: Codex on Windows, VS Code “Agent DX,” Symphony, LangSmith Skills**

- **Codex app lands on Windows + open-source sandbox**: OpenAI DevRel announces **Codex for Windows** with a **Windows-native agent sandbox**, using OS controls (restricted tokens, ACLs, dedicated users) to constrain filesystem/network access unless approved; implementation is **open source** ([tweet](https://x.com/OpenAIDevs/status/2029252453246595301), [tweet](https://x.com/OpenAIDevs/status/2029252477179314350)). AJ Ambrosino adds details: runs natively or via WSL; supports PowerShell/CMD/Git Bash/WSL terminals; “Open in …” integrations and Windows skills ([tweet](https://x.com/ajambrosino/status/2029252598851879265)). Reach_vb highlights the OSS sandbox as an underrated artifact ([tweet](https://x.com/reach_vb/status/2029335011804017135)).
- **VS Code’s agent-oriented release**: The `@code` account emphasizes “Agents, for real work,” shipping **hooks**, **message steering/queueing**, **integrated agentic browser**, and **shared memory** ([tweet](https://x.com/code/status/2029279963778515372)). A process change matters for builders: VS Code is moving from monthly to **weekly shipping** of `main` to accelerate feature delivery ([tweet](https://x.com/pierceboggan/status/2029283603801358798)).
- **OpenAI Symphony (ticket-board→agents orchestration)**: A new OpenAI repo, **Symphony**, is described as an orchestration layer that **polls project boards** and spawns agents per ticket lifecycle stage—shifting the UX from “prompt the agent” to “move tickets and let agents execute” ([tweet](https://x.com/scaling01/status/2029261034993684952)). This is consistent with the broader trend toward **workflow-native agent automation**.
- **LangSmith Skills + CLI (agents doing agent engineering)**: LangChain ships **LangSmith Skills + CLI** so coding agents can natively debug traces, build datasets, and run experiments from the terminal ([tweet](https://x.com/LangChain/status/2029272199073354105)). In parallel, **LangChain OSS Skills** aim to teach agents how to use LangChain/LangGraph/DeepAgents effectively ([tweet](https://x.com/LangChain_OSS/status/2029272669942673436), [tweet](https://x.com/hwchase17/status/2029274371710501049)).
- **Cursor via Agent Client Protocol in JetBrains**: Cursor announces availability in **JetBrains IDEs** through **Agent Client Protocol** ([tweet](https://x.com/cursor_ai/status/2029222015736197205)). This is a key distribution move: IDE-native access without forcing a tool switch.

---

**Multimodal + world models: Self-Flow, Beyond Language Modeling, persistent video, and NE-Dreamer**

- **Black Forest Labs’ Self-Flow**: BFL previews **Self-Flow**, a **self-supervised flow-matching** approach for multimodal generative models (image/video/audio/text) that avoids relying on external pretrained representation models (e.g., DINO). Claimed results: **up to 2.8× faster convergence**, improved video temporal consistency, sharper typography; positioned as foundational for multimodal visual intelligence and even action prediction ([tweet](https://x.com/bfl_ml/status/2029212134023020667); additional context [tweet](https://x.com/robrombach/status/2029272803099226425)).
- **“Beyond Language Modeling” / vision-first multimodal pretraining**: Multiple authors promote a paper exploring **native multimodal models** where vision is treated as first-class and models input/output all modalities “Transfusion-style,” including discussion of representations, data, world modeling, architecture, and scaling laws ([tweet](https://x.com/__JohnNguyen__/status/2029236083914096756), [tweet](https://x.com/TongPetersb/status/2029237530160169286), [tweet](https://x.com/DavidJFan/status/2029239760301035549)). The throughline: the field may be underestimating how much progress requires **vision-native training**, not language-first adapters.
- **Long-context video world models**: Gordon Wetzstein’s thread teases “Mode Seeking meets Mean Seeking (MMM)” as a route to **long-context, persistent video world models** via a unified representation ([tweet](https://x.com/GordonWetzstein/status/2029054374459376026)).
- **NE-Dreamer: embedding prediction instead of pixel reconstruction**: George Bredis introduces **NE-Dreamer**, exploring world models trained to **predict next embeddings** rather than reconstruct pixels—arguing reconstruction may be the wrong objective for control ([tweet](https://x.com/BredisGeorge/status/2029190420790411671)).

---

**Evaluation, memory, and “human-centered” coding: factorization barriers, agent memory diagnostics, bloated patches, and rubric drift**

- **Diffusion LLM parallelism hits the “Factorization Barrier”**: Ian Li explains why diffusion LLMs struggle with parallel token generation: predicting multiple tokens simultaneously can induce incoherent joint outputs (e.g., “San York”). He attributes this to a structural misspecification—fully factorized output heads can’t represent the full joint distribution without exploding output-head size—and proposes **CoDD** as a way to break the barrier ([tweet](https://x.com/IanLi1118/status/2029074519223353362)).
- **Agent memory: retrieval dominates “writing” strategy**: A diagnostic framework separates **retrieval failures vs utilization failures**; key claim: retrieval approach causes **~20pp** variance, while memory-writing methods only shift **3–8pp**. “Raw chunking” can match or beat expensive summarization/fact-extraction pipelines ([tweet](https://x.com/dair_ai/status/2029202969456234562)). Practical implication: many teams may be over-optimizing memory “ingestion” rather than search/selection.
- **SWE-bench patch bloat as a human-factor failure mode**: KLieret reports LLM-generated SWE-bench patches are consistently **longer/bloated** than human solutions (not just comments), which can pass tests but harm human verification and maintenance ([tweet](https://x.com/KLieret/status/2029219763423986030)). Follow-up emphasizes “test success != practical usability” and argues for **human-centered coding agent research** ([tweet](https://x.com/ZhiruoW/status/2029229015634993579)).
- **Rubric drift and evals as “living systems”**: Multiple tweets stress that failures often come from an outdated **eval rubric** rather than a “broken prompt”; the fix is to treat evals as a feedback loop tied to production distribution shift, not static unit tests ([tweet](https://x.com/omarsar0/status/2029225624825659668), [tweet](https://x.com/kimmonismus/status/2029227463805378571)).
- **BullshitBench v2 (nonsense detection)**: A benchmark testing whether models **reject nonsensical prompts** finds only **Claude** and **Qwen 3.5** scoring meaningfully above **60%**, with an observed failure mode: “think harder” reasoning models can **rationalize nonsense** instead of rejecting it ([tweet](https://x.com/kimmonismus/status/2029230388028358726)). If true, it’s a useful counterweight to raw “reasoning tokens” as a proxy for quality.

---

**Top tweets (by engagement, technically relevant)**

- **NotebookLM Cinematic Video Overviews** rollout (Ultra users): [@NotebookLM](https://x.com/NotebookLM/status/2029240601334436080)  
- **OpenAI Codex app on Windows** + Windows-native sandbox details: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2029252453246595301) and [@ajambrosino](https://x.com/ajambrosino/status/2029252598851879265)  
- **Gemini 3.1 Flash-Lite** speed/cost positioning: [@demishassabis](https://x.com/demishassabis/status/2029047252275060895)  
- **Speculative Speculative Decoding (SSD)** claim of up to 2× faster inference: [@tanishqkumar07](https://x.com/tanishqkumar07/status/2029251146196631872)  
- **Yuan 3.0 Ultra** open multimodal MoE (1010B total / 68.8B active) release announcement: [@YuanAI_Lab](https://x.com/YuanAI_Lab/status/2029204213180580229)  
- **Self-Flow** multimodal flow-matching research preview (2.8× faster convergence claim): [@bfl_ml](https://x.com/bfl_ml/status/2029212134023020667)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen Model Performance and Benchmarks

  - **[Qwen3.5-35B-A3B hits 37.8% on SWE-bench Verified Hard — nearly matching Claude Opus 4.6 (40%) with the right verification strategy](https://www.reddit.com/r/LocalLLaMA/comments/1rkdlqi/qwen3535ba3b_hits_378_on_swebench_verified_hard/)** (Activity: 464): **The post discusses the performance of the **Qwen3.5-35B-A3B** model, a small MoE model with `3B active parameters`, on the SWE-bench Verified Hard tasks. By implementing a simple verification strategy—*"verify after every edit"*—the model's performance improved from `22%` to `37.8%`, nearly matching **Claude Opus 4.6**'s `40%`. This strategy involves prompting the model to verify changes after each `file_edit` by writing and running a test script. The model achieved `67.0%` on the full 500-task benchmark, comparable to larger systems. The author notes that more complex strategies like MCTS and Best-of-N sampling were less effective. [GitHub repository](http://github.com/SeungyounShin/agent-verify) with code and logs is provided.** One commenter suggests waiting for new tasks on SWE-bench to avoid potential data leakage in model training. Another expresses skepticism about the results, suggesting they might be "benchmaxed." A third commenter notes the absence of looping in the strategy, which they found challenging with the 35B model.

    - ResidentPositive4122 highlights a potential issue with the SWE-bench, noting that it is outdated and may contain leaked signals in training data for newer models. They suggest waiting for an updated version with new tasks to ensure more accurate evaluations.
    - Deep_Traffic_7873 claims that Qwen3.5-35B-A3B outperforms GPT-OSS-20B in their personal benchmarks, suggesting a significant performance advantage of the former over the latter in specific tasks.
    - ethereal_intellect provides a detailed list of OpenAI's guidelines for their Codex harness environment, which includes steps like validating the codebase, reproducing bugs, and implementing fixes. They note that some tasks, such as faking a video and driving the application, are particularly challenging but feasible with careful setup.

  - **[Qwen3.5-27B Q4 Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1rk5qmr/qwen3527b_q4_quantization_comparison/)** (Activity: 386): **The post presents a detailed comparison of Q4 quantization methods for the Qwen3.5-27B model, focusing on mean KL Divergence (KLD) against a BF16 baseline. The evaluation uses a custom chat dataset and Wikitext2, with results showing that the `unsloth_Qwen3.5-27B-UD-Q4_K_XL` quantization achieves the lowest KLD of `0.005087`, while `bartowski_Qwen3.5-27B-IQ4_XS` is noted for its efficiency score of `0.317506`. The analysis uses `llama.cpp` for evaluation and highlights the importance of KLD as a measure of faithfulness to the original model's probability distribution. The post also provides a GitHub link to a script used for the KLD sweep, though it is noted as not extensively tested.** A notable comment questions the discrepancy in model size between the post and Hugging Face, suggesting potential differences in quantization methods or reporting. Another comment suggests that models close to the best fit line in a size vs. KLD plot are preferable, indicating a preference for models that balance size and accuracy.

    - Gueleric raises a technical question about the discrepancy in model size for `bartowski_Qwen3.5-27B-IQ4_XS`, noting a difference between the reported 14.1GB size and the 15.2GB size listed on Hugging Face. This could be due to different quantization methods or metadata included in the Hugging Face model size.
    - PaMRxR discusses a plot they created showing quantization size versus Kullback-Leibler Divergence (KLD) for the Qwen3.5-27B model. They mention removing outliers to better fit the data, suggesting that models closer to the best fit line are preferable. The plot was generated using the `unsloth_Qwen3.5-27B-UD-Q4_K_XL` model, indicating a focus on understanding the trade-offs between model size and performance metrics like KLD.
    - munkiemagik expresses interest in conducting qualitative comparisons across different parameters and quantization levels for models. They highlight a common issue in model testing: often only specific metrics like perplexity or throughput are reported, which may not align with the user's needs. They also mention the challenge of understanding technical concepts like KL Divergence, indicating a need for deeper engagement with the academic principles behind large language models.


### 2. Qwen Model Usability and Applications

  - **[Qwen3.5-0.8B - Who needs GPUs?](https://www.reddit.com/r/LocalLLaMA/comments/1rkjsaj/qwen3508b_who_needs_gpus/)** (Activity: 646): **The image highlights the impressive capability of the `Qwen3.5-0.8B` model, which can run efficiently on outdated hardware, specifically a 2nd generation i5 processor with 4GB DDR3 RAM. This model is executed using `llama.cpp`, a tool for running large language models on local machines, and is shown to handle complex topics like string theory. The system information is displayed using `fastfetch` on an Arch Linux setup, emphasizing the model's low resource requirements and accessibility for users without high-end GPUs.** Commenters express amazement at the model's performance on such old hardware, comparing it to the capabilities of GPT-3 and noting the open-source nature of the model. There's also a nostalgic mention of semi-transparent terminals, reflecting on past desktop environments.

    - The Qwen3.5-0.8B model is notable for its ability to run efficiently without the need for a GPU, which is a significant advancement in making AI more accessible. This model is open-source, allowing for broader experimentation and use in various applications without the high cost of GPU resources.
    - A user suggests using the Qwen3 8B model instead, highlighting its superior performance and the fact that it also does not require a GPU. This suggests that the Qwen3 series is optimized for performance on lower-end hardware, making it a practical choice for developers without access to high-end computing resources.
    - The Qwen3.5-0.8B model includes a vision component, which allows it to analyze images and generate workflows that can produce images or videos. This feature expands its utility beyond text-based tasks, enabling it to function as a sub-agent in multimedia applications.

  - **[Qwen 3.5 4b is so good, that it can vibe code a fully working OS web app in one go.](https://www.reddit.com/r/LocalLLaMA/comments/1rkb8en/qwen_35_4b_is_so_good_that_it_can_vibe_code_a/)** (Activity: 718): **The post discusses the capabilities of **Qwen 3.5 4b**, a compact AI model, which successfully created a fully functional web-based operating system (OS) from a single prompt. The OS includes features such as two games, a text editor, an audio player, a file browser, customizable wallpaper, and a special feature chosen by the model itself. The model's ability to generate a working OS with these specifications highlights significant advancements in AI model efficiency and information density, particularly for a model of only `4 billion parameters`. The OS can be accessed [here](https://qwen4bwebos.tiiny.site/).** Commenters express skepticism about the test's validity, suggesting it may be a common benchmark scenario potentially optimized for success. Others are impressed by the model's performance, noting the significant progress in AI capabilities beyond mere scaling.

    - **tinny66666** highlights the impressive performance of the Qwen 3.5 4b model, noting that its intelligence surpasses the original GPT-3.5 despite its smaller size. This suggests a significant improvement in information density and model efficiency, raising questions about the potential limits of such advancements.
    - **msixtwofive** expresses skepticism about the validity of the test, suggesting that the task of creating a fully working OS web app is a common benchmark that may have been optimized for by AI influencers. This raises concerns about the authenticity of the model's performance in real-world, unseeded scenarios.
    - **simracerman** points out that while the Qwen 3.5 4b model's ability to complete the task is impressive, especially compared to larger models, there is a possibility that the code for such tasks might be included in the training data, which could influence the model's performance.


### 3. Tech Industry Developments and Reactions

  - **[Apple unveils M5 Pro and M5 Max, citing up to 4× faster LLM prompt processing than M4 Pro and M4 Max](https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple_unveils_m5_pro_and_m5_max_citing_up_to_4/)** (Activity: 998): **The image illustrates the capabilities of Apple's newly announced M5 Pro and M5 Max chips, which are claimed to process large language model (LLM) prompts up to 4 times faster than their predecessors, the M4 Pro and M4 Max. The M5 Pro supports up to 64GB of unified memory with a bandwidth of 307GB/s, while the M5 Max supports up to 128GB of unified memory with a bandwidth of 614GB/s. Additionally, these chips feature up to 2× faster SSD speeds at 14.5GB/s and include the Apple N1 wireless chip for Wi-Fi 7, enhancing download speeds if compatible with the router.** Some commenters express a desire for a Mac Studio equipped with the new chips, while others note the lack of mention of AI-specific silicon improvements, such as a Neural Accelerator.

    - The M5 Pro and M5 Max chips feature significant improvements in memory capabilities, with the M5 Pro supporting up to 64GB of unified memory and 307GB/s of memory bandwidth, while the M5 Max supports up to 128GB of unified memory and 614GB/s of memory bandwidth. These enhancements are crucial for handling large-scale machine learning models and intensive computational tasks.
    - The new chips also boast up to 2× faster SSD speeds, reaching 14.5GB/s, which can significantly reduce data access times and improve overall system performance. Additionally, the inclusion of the Apple N1 wireless chip for Wi-Fi 7 support offers faster download speeds, provided the network infrastructure can support it, enhancing connectivity for data-intensive applications.
    - Despite expectations for more advanced AI-specific silicon, the M5 series still offers substantial performance gains, particularly in LLM prompt processing, which is up to 4× faster than the previous M4 series. This improvement is likely due to a combination of increased memory bandwidth and faster SSD speeds, which together enhance the chips' ability to handle complex AI workloads efficiently.

  - **[ChatGPT uninstalls surged by 295% after Pentagon deal](https://www.reddit.com/r/LocalLLM/comments/1rjlzgy/chatgpt_uninstalls_surged_by_295_after_pentagon/)** (Activity: 418): **The image is a meme and does not provide any technical insights or verifiable data. It humorously suggests a significant increase in ChatGPT uninstalls following a supposed deal with the Pentagon, but lacks any credible sources or detailed information to support this claim. The comments reflect skepticism about the validity of the claim, questioning the source and the actual impact on user numbers.** Commenters express skepticism about the claim, questioning the source and the actual impact on user numbers, suggesting it might be exaggerated or unsourced.

    - A user questions the validity of the claim regarding the surge in uninstalls, asking if the statistic is unsourced, which raises concerns about the reliability of the data. This highlights the importance of verifying claims with credible sources, especially when discussing significant changes in user behavior.
    - Another comment critiques the shift in OpenAI's mission from a 'non-profit research lab' to potentially acting as a 'Defense Contractor.' This reflects a broader debate on the ethical implications of AI development and its alignment with military applications, suggesting a tension between original mission statements and current business practices.
    - A user discusses the inevitability of AI's integration into military applications, arguing that technological advancements naturally lead to such outcomes to maintain competitive advantage. This comment underscores the strategic importance of AI in defense and the potential consequences of falling behind in technological capabilities.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Benchmark Releases

  - **[Opus 4.6 solved one of Donald Knuth's conjectures from writing "The Art of Computer Programming" and he's quite excited about it](https://www.reddit.com/r/singularity/comments/1rkhady/opus_46_solved_one_of_donald_knuths_conjectures/)** (Activity: 1124): **The image is a document titled "Claude’s Cycles" by **Donald Knuth**, discussing a significant breakthrough achieved by the AI model **Claude Opus 4.6**. This model, a hybrid reasoning system, solved a longstanding conjecture related to directed Hamiltonian cycles and the decomposition of arcs into directed cycles, which Knuth had been working on. The document highlights Knuth's surprise and joy at the AI's solution, marking a notable advancement in automatic deduction and creative problem solving. The full paper is available [here](https://www-cs-faculty.stanford.edu/~knuth/papers/claude-cycles.pdf).** Commenters express admiration for Knuth's openness to revising his views on AI, noting his intellectual integrity. They also highlight the significance of **Anthropic's** achievement with Claude Opus 4.6, and celebrate Knuth's continued active engagement in research at the age of 88.

    - In the paper, it is noted that the AI model, Claude, demonstrated its capability by solving Knuth's conjecture for odd values of `m` and finding solutions for some even values, though it couldn't generalize a solution for all even `m`. This highlights the model's ability to explore numerous approaches rapidly, a task that would be time-consuming for human mathematicians.
    - Donald Knuth's acknowledgment of the AI's achievement marks a significant shift in his perspective on generative AI. Previously skeptical, Knuth now recognizes the rapid advancement in AI's capabilities, particularly in automatic deduction and creative problem-solving, as demonstrated by Claude's contribution to solving his conjecture.
    - The involvement of Anthropic's Claude in solving a part of Knuth's conjecture underscores the potential of AI in mathematical research. While not necessarily 'smarter' than human mathematicians, Claude's ability to quickly test various hypotheses and approaches is a notable advantage, illustrating the evolving role of AI in complex problem-solving.

  - **[Gemini 3.1 Flash-Lite Benchmark Comparison](https://www.reddit.com/r/Bard/comments/1rjusj5/gemini_31_flashlite_benchmark_comparison/)** (Activity: 236): **The **Gemini 3.1 Flash-Lite** model, as per its [model card](https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/), is benchmarked against the older 2.5 Flash model rather than the more recent 3 Flash model, raising questions about its comparative performance. The model is priced at `$0.25` per input and `$1.50` per output, which is significantly higher than the 2.5 Flash Lite's `$0.10` input and `$0.40` output costs. This pricing strategy suggests a focus on specific use cases rather than broad applicability, as it is still cheaper than the 3 Flash model but more expensive than its predecessor.** Commenters express dissatisfaction with the cost-to-performance ratio of the Gemini 3.1 Flash-Lite, noting that it is "3x as expensive" as the 2.5 Flash Lite without a proportional performance increase. Comparisons to other models like Grok 4.1 and MiniMax M2.5 highlight that these alternatives offer better value for money, suggesting that the 3.1 Flash Lite may not be competitive in terms of pricing and performance.

    - **Important-Farmer-846** highlights the cost-effectiveness of 2.5 Flash Lite over 3.1 Flash Lite, noting that while 3.1 is half the price of Flash 3, it is twice as expensive as 2.5 Flash Lite. The commenter suggests that for processing large volumes of data, 2.5 Flash Lite remains the better option due to its lower cost and adequate performance.
    - **ExpertPerformer** provides a detailed cost comparison of various models, showing that 3.1 Flash Lite is less cost-effective compared to alternatives like MinMax M2.5 and Grok 4.1. The cost per input/output for 3.1 Flash Lite is $0.25/$1.50, whereas MinMax M2.5 offers $0.295/$1.20, and Grok 4.1 offers $0.20/$0.50, indicating better value for money with the latter models.
    - **ThomasMalloc** discusses the inefficiency of 3.1 Flash Lite in 'High' thinking mode, noting it took 14 times longer than 2.5 Flash Lite and maxed out the output tokens at 65,436 compared to 6,980 for 2.5 Lite. The commenter suggests using 'Minimal' or 'Low' thinking modes to reduce token usage and cost, as 'High' mode is currently impractical due to excessive token consumption and incomplete outputs.

  - **[Ostris is testing Lodestones ZetaChroma (Z-Image x Chroma merge) for LORA training 👀](https://www.reddit.com/r/StableDiffusion/comments/1rkky97/ostris_is_testing_lodestones_zetachroma_zimage_x/)** (Activity: 254): **The image is a screenshot of a chat conversation where a user named Ostris discusses testing a LoRA (Low-Rank Adaptation) model using Lodestones ZetaChroma. ZetaChroma is a new model that combines the Chroma dataset with Z-Image, focusing on pixelspace inference. This model is being tested for integration into an AI toolkit for training. The discussion highlights that ZetaChroma is not a simple model merge but a retraining of Z-Image using the Chroma dataset, aiming to create a powerful open-source model. The conversation also includes a file link to a safetensor file, indicating active testing and development.** Comments clarify that ZetaChroma is not a model merge but a retraining effort, emphasizing the use of the Chroma dataset to train a pixelspace model from scratch on top of Z-Image.

    - Far_Insurance4191 clarifies that Zeta is not a model merge but a retraining of the Z-Image model using the same dataset initially used for Chroma. This indicates a focus on refining the model's capabilities by leveraging existing data rather than combining model weights.
    - PetiteKawa00x emphasizes that Zeta involves training a pixelspace model from scratch on top of Z-Image with the Chroma dataset, highlighting that no weights from Chroma are merged with Z-Image for Zeta. This suggests a distinct approach in model development, focusing on foundational training rather than integration of existing models.


### 2. Anthropic and OpenAI Leadership Changes

  - **[OpenAI VP Max Schwarzer joins Anthropic amid recent kerfuffle](https://www.reddit.com/r/OpenAI/comments/1rkrj20/openai_vp_max_schwarzer_joins_anthropic_amid/)** (Activity: 1121): **The image is a meme featuring a surprised Pikachu, humorously depicting the reaction to **OpenAI VP Max Schwarzer** leaving OpenAI to join **Anthropic**. This move is part of a broader trend where several key figures from OpenAI have transitioned to Anthropic, a company founded by former OpenAI employees. The meme suggests a sense of surprise or shock from OpenAI at this departure, reflecting ongoing tensions and shifts within the AI industry.** Commenters express skepticism about the leadership at OpenAI, with some suggesting a lack of trust in the company's direction under its current leadership. There is also a sentiment of customers switching allegiance to Anthropic, indicating a potential shift in market preference.


  - **[OpenAI VP for Post Training defects to Anthropic](https://www.reddit.com/r/OpenAI/comments/1rk6xnw/openai_vp_for_post_training_defects_to_anthropic/)** (Activity: 1839): **The image is a tweet from Max Schwarzer, who was the Vice President for Post Training at **OpenAI**. He announced his departure to join **Anthropic**, a company known for its focus on AI safety and research. Max highlights his contributions at OpenAI, including leading the post-training team and working on models like GPT-5. His move to Anthropic is framed as a return to research, suggesting a shift in focus towards more foundational AI work.** One comment humorously misreads his title as 'VP of Post Training Defects,' while another suggests his move might be due to OpenAI's challenges, metaphorically described as 'jumping off a sinking ship.'


  - **[OpenAI's post-training lead leaves and joins Anthropic: he helped ship GPT-5, 5.1, 5.2, 5.3-Codex, o3 and o1 and will return to hands-on RL research at Anthropic](https://www.reddit.com/r/ClaudeAI/comments/1rk7fwq/openais_posttraining_lead_leaves_and_joins/)** (Activity: 1818): ****Max Schwarzer**, a key figure in OpenAI's post-training team, has announced his departure to join **Anthropic**. Schwarzer played a significant role in the development and deployment of several major models at OpenAI, including GPT-5, 5.1, 5.2, 5.3-Codex, and others. His move to Anthropic marks a return to hands-on research in reinforcement learning, highlighting a shift from leadership to direct research involvement. This transition underscores the competitive landscape in AI research talent, with Anthropic being noted for its strong values and talent pool.** Commenters are impressed by Schwarzer's rapid career progression and note the potential implications of his departure on OpenAI's projects, including possible impacts on revenue and strategic direction.

    - Freed4ever raises a point about the potential need for a 'cool down period' for high-tech talents transitioning between companies, similar to practices in the quantitative finance industry. This is due to the sensitive nature of the work and the proprietary knowledge these researchers possess, which could impact competitive dynamics in the AI field.
    - PJpittie expresses dissatisfaction with GPT-5, suggesting it did not meet expectations. This comment reflects a broader sentiment that could be indicative of performance issues or unmet benchmarks in the latest iterations of OpenAI's models, which might influence user trust and adoption.
    - CallMePyro highlights the implications of OpenAI's Department of Defense (DoD) deal, suggesting that the consequences extend beyond financial losses. This could imply strategic or ethical considerations that might affect OpenAI's operations and its talent retention strategies.

  - **[OpenAI VP for Research for post-training defects to Anthropic](https://www.reddit.com/r/ChatGPT/comments/1rk6yy6/openai_vp_for_research_for_posttraining_defects/)** (Activity: 614): **The image is a tweet from Max Schwarzer, who was the VP for Research at OpenAI, announcing his departure to join Anthropic. He highlights his contributions to OpenAI, particularly in reasoning paradigms and post-training teams, which are crucial for refining AI models post-training to ensure they function effectively. His move to Anthropic, a company known for its focus on AI safety and research, is significant given the competitive landscape in AI research and development. This shift underscores the ongoing talent migration within the AI industry, raising questions about internal dynamics at OpenAI.** Commenters note the significance of losing a key figure in post-training, which is essential for model refinement, and speculate on the internal culture at OpenAI given the frequent departures of senior researchers. There is also a discussion on Anthropic's values and potential growth, with some expressing confidence in its future prospects.

    - The departure of OpenAI's VP for Research to Anthropic is significant due to the critical role of post-training in refining AI models. Post-training is essential for ensuring models produce coherent and reliable outputs, and losing a key figure in this area could impact OpenAI's model development and stability.
    - The frequent departure of senior researchers from OpenAI raises questions about the company's internal culture and stability. This trend suggests potential issues within the organization that might be prompting key talent to leave, which could affect OpenAI's long-term innovation and competitiveness.
    - The move to Anthropic is seen as strategically timed, possibly reflecting a shift in values or strategic direction. Anthropic's focus on ethical AI and its growing customer base, including enterprise and consumer-level clients, positions it as a strong competitor in the AI landscape, potentially attracting talent seeking alignment with these values.


### 3. Claude and ChatGPT User Reactions

  - **[Damnnnn!](https://www.reddit.com/r/singularity/comments/1rjc5to/damnnnn/)** (Activity: 2597): **The image is a meme-style screenshot from TechCrunch on X.com, highlighting a significant increase in ChatGPT uninstalls by `295%` following a Department of Defense (DoD) deal. This suggests a public reaction to privacy concerns or ethical considerations regarding the use of ChatGPT in government contracts. The post has garnered substantial engagement, indicating widespread interest or concern about the implications of such a deal. However, the top comment points out that the percentage increase could be misleading without absolute numbers, suggesting that the actual impact might be minimal. Another comment highlights the potential financial implications, noting that even if a large number of users cancel their subscriptions, the DoD deal could offset these losses financially.** Commenters express skepticism about the significance of the uninstall surge, with one noting that the percentage increase could be misleading without absolute numbers. Another comment discusses the financial trade-off, suggesting that the DoD deal might compensate for any loss in subscription revenue.

    - mazdarx2001 highlights the financial implications of user cancellations for a subscription service, noting that if one million users paying $20 monthly cancel, it results in a $20 million monthly revenue loss. However, they argue that a Department of Defense (DoD) contract could offset this loss, suggesting that government contracts might provide more stable revenue streams than consumer subscriptions.
    - Orangeshoeman discusses the potential impact of a Department of Defense (DoD) contract on a company's downstream revenue, particularly in the context of privacy concerns. They imply that users seeking privacy might avoid using services associated with government contracts, which could negatively affect the company's reputation and user base.
    - TimeTravelingChris points out that user dissatisfaction combined with the availability of better alternatives can lead to significant business challenges. They suggest that the presence of superior products in the market, along with customer discontent, could create a 'recipe for disaster' for the company in question.

  - **[295% is wild](https://www.reddit.com/r/OpenAI/comments/1rjc5nm/295_is_wild/)** (Activity: 3163): **The image is a meme-like screenshot of a TechCrunch tweet claiming a `295%` surge in ChatGPT uninstalls following a Department of Defense (DoD) deal. The post title and comments suggest skepticism about the significance of this statistic, with users pointing out that without knowing the baseline number of uninstalls, the percentage increase is not meaningful. Additionally, the comments question the reliability of the data source and the journalistic standards of TechCrunch, implying that the reported surge may not have substantial impact or relevance.** Commenters express skepticism about the significance of the `295%` uninstall surge, noting that without baseline numbers, the statistic lacks context. They also criticize TechCrunch's reporting, questioning the accuracy and relevance of the data presented.

    - Diligent_Net4349 and FalkenJoshua both highlight the importance of understanding the baseline number when interpreting a 295% increase in uninstalls. Without knowing the original number of uninstalls, the percentage increase lacks context and could be misleading. For example, a 300% increase from a small base number like 1000 would only result in 3000, which might not be significant in the grand scheme.
    - FormerOSRS provides a breakdown of the uninstall statistics, suggesting that the increase equates to 12 days' worth of uninstalls occurring over just three days. This implies that while the percentage increase seems large, the actual impact may be minimal if the baseline uninstall rate was low.
    - Umademedothis2u questions the source of the uninstall rate data, implying skepticism about the accuracy of the reported statistics. This comment suggests a need for transparency in how such data is collected and reported, especially in tech journalism.

  - **[OpenAI loses 1.5 million subscribers in less than 48 hours after CEO Sam Altman says yes to the deal that Anthropic rejected](https://www.reddit.com/r/ChatGPT/comments/1rkd4td/openai_loses_15_million_subscribers_in_less_than/)** (Activity: 4037): ****OpenAI** reportedly lost `1.5 million subscribers` within `48 hours` following **CEO Sam Altman's** decision to accept a deal that **Anthropic** had previously rejected. The source of the `1.5 million` figure is questioned, as it is unclear whether this was officially reported by OpenAI or derived from another source. This event highlights potential dissatisfaction with OpenAI's strategic decisions and leadership under Altman.** The comments reflect skepticism about the reported subscriber loss figure, questioning its origin and accuracy. Additionally, there is criticism of Sam Altman's leadership style and public statements, suggesting a disconnect with public perception.

    - A user highlights their switch to Claude, noting its superior performance in areas like marketing, data analysis, and research. They emphasize Claude's consistent memory and balanced feedback, comparing it to sci-fi AIs like Hal 9000 or Cortana. They also mention that Opus 4.6 extended is the best AI model they've used, although they still rely on GPT and Gemini for health-related queries.
    - Another user questions the source of the 1.5 million subscriber loss figure, asking if it was officially reported by OpenAI. This suggests skepticism about the accuracy or origin of the statistic, indicating a need for verification or official confirmation.
    - A user expresses a desire to obtain a personal data export from OpenAI, indicating concerns about data privacy and control. This reflects a broader trend of users becoming more conscious of their data rights and the information companies hold about them.

  - **[ChatGPT Uninstalls Surge 295% After OpenAI’s DoD Deal Sparks Backlash](https://www.reddit.com/r/ChatGPT/comments/1rjfipu/chatgpt_uninstalls_surge_295_after_openais_dod/)** (Activity: 3053): **OpenAI's recent partnership with the U.S. Department of Defense led to a `295%` increase in uninstalls of the ChatGPT mobile app, reflecting significant user backlash against the company's military affiliations. This reaction occurred within `48 hours` of the announcement and coincided with a rise in downloads for competitor **Claude**, illustrating the competitive dynamics in AI applications. The event highlights the reputational risks of government contracts in the AI sector, as user sentiment plays a crucial role in shaping corporate strategies.** The comments reflect a strong negative sentiment towards OpenAI's decision, with some users suggesting that the backlash was deserved and expressing skepticism about OpenAI's intentions. There is also a mention of a conspiracy theory regarding a whistle-blower, indicating distrust among some users.

    - EnotHOME questions the significance of the 295% increase in uninstalls, suggesting that if the baseline was 1000 uninstalls, a 295% increase would mean 4000 uninstalls, which they consider insignificant in the grand scheme of things. This implies a need for more context on the baseline numbers to assess the true impact.
    - coronakillme seeks clarification on the 295% figure, interpreting it as the number of uninstalls being a little less than three times higher than before. They question what the original number of uninstalls was, highlighting the importance of understanding the baseline to evaluate the significance of the increase.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5


**1. Software Engineering Benchmarks & Routers**

- **SWE-Atlas Stings SOTA at ~30%**: **Scale AI** launched **SWE-Atlas**, extending **SWE-Bench Pro**, with its first benchmark **Codebase QnA** showing current top models at ~**30%** on software-engineering Q&A, per the launch post: [SWE-Atlas launch (Scale AI)](https://x.com/scale_AI/status/2029244660905095359).
  - Engineers called it a wake-up benchmark, pointing to the leaderboard for hard, repo-grounded evals: [SWE-Atlas Codebase QnA leaderboard](https://scale.com/leaderboard/sweatlas-qna), stressing gaps in **codebase grounding** and **long-context retrieval**.

- **Max Router Routs Rivals**: **Arena ML** researchers showcased their **Max intelligent router** that selects per-query winners and reportedly “beats every model on the platform,” in this breakdown: [Max intelligent router (YouTube)](https://www.youtube.com/watch?v=nO6E5t6dmA0).
  - Viewers highlighted that dynamic routing plus tool selection can outperform any single static model, quoting the video’s claim that it *“beats every model on the platform.”*

- **Cursor Cracks First-Proof Problem**: **Cursor AI** ran for ~**4 days** and discovered a novel solution to “**Problem Six**” from the Arc Institute’s First Proof challenge, reportedly outperforming academic baselines: [Cursor solves ‘First Proof’ Problem Six (X)](https://x.com/mntruell/status/2028903020847841336), context at [Evo-2: One year later (Arc Institute)](https://arcinstitute.org/news/evo-2-one-year-later).
  - Researchers debated whether the agent coordination approach generalizes beyond code tasks into math research, with some urging replication on more problems to validate **robustness**.


**2. Systems & GPU Optimization Breakthroughs**

- **GPU Chats NVMe Without CPU Chaperone**: A Linux hacker enabled **AMD GPU ⇄ NVMe P2P** by patching the **amdgpu** driver and wiring dma-buf/iommufd per Jason Gunthorpe’s RFC: [dma-buf/iommufd RFC (lore.kernel.org)](https://lore.kernel.org/dri-devel/0-v1-b5cab63049c0+191af-dmabuf_map_type_jgg@nvidia.com/), enabling direct **GPU–SSD** command paths.
  - They contrasted this with **ROCm/hipFile**, arguing hipFile still issues commands via the CPU, while their path keeps the **CPU out of the data path**: [ROCm hipFile (GitHub)](https://github.com/ROCm/hipFile).

- **CUDA Agent Clobbers Kernels**: **ByteDance** introduced a **CUDA Agent** that writes optimized CUDA kernels and claims ~**2×** speedups over **torch.compile** on simple/medium tasks per the paper: [CUDA Agent paper (arXiv)](https://arxiv.org/pdf/2603.02298).
  - Community notes say it also outperforms **Claude Opus 4.5** and **Gemini 3 Pro** by ~**40%** on harder kernels, calling it a tangible step toward **LLM-driven kernel autotuning**.

- **MXFP8 MMA Mystifies Devs**: Kernel engineers flagged that **MXFP8 MMA** appears to support `MMA_K=64` only for sparse shapes (vs `K=256` for dense) per the **PTX** guide: [PTX matrix shapes (NVIDIA docs)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape).
  - Threads also probed **inter-CTA** correctness via global memory and SASS fences (`MEMBAR`, `LDG/STG.STRONG`, `CCTL.IVALL`), pushing for architecture-specific guidance on **barrier semantics**.


**3. Agent Platforms, UX, and Dev Tooling**

- **Codex Camps on Windows**: **OpenAI** shipped the **Codex app** on **Windows** with a native **agent sandbox** and **PowerShell** support, demoed here: [Codex on Windows demo (video)](https://video.twimg.com/amplify_video/2029252379347173377/vid/avc1/1280x720/5YaNsuJawfWhfyYG.mp4).
  - Developers welcomed Windows-native flows, calling the **PowerShell** integration a pragmatic boost for **agentic dev environments** on enterprise desktops.

- **ACP Bridges IDEs and Agents**: The **Agent Communication Protocol (ACP)** now plugs into **Zed** and **IntelliJ**, letting agents drive multiple providers (e.g., Cursor) from one interface: [AgentCommunicationProtocol.dev](https://agentcommunicationprotocol.dev/introduction/welcome).
  - Engineers reported smoother **multi-tool orchestration** and fewer context hops, saying ACP helps keep **provider sprawl** in check.

- **Six Agents Ship a Marketplace**: An **OpenClaw** squad of **6 parallel agents** built a functional marketplace in a weekend, with a `prompt-generator.ts` that emits platform-specific templates for **Cursor** and **v0**: [codebonito.com](https://codebonito.com), tools at [Cursor](https://cursor.sh/).
  - Builders praised the **template compiler** pattern—*“write once, target many runtimes”*—for speeding agent deployments across heterogeneous **toolchains**.


**4. Inference Speed & Context-Efficiency Tricks**

- **SSD Speeds Up Decoding**: Researchers previewed **Speculative Speculative Decoding (SSD)** by Tanishq Kumar, Tri Dao, and Avner May, claiming up to **2×** faster inference over leading engines: [Speculative Speculative Decoding (X)](https://x.com/tanishqkumar07/status/2029251146196631872).
  - Practitioners flagged SSD as a practical win for **throughput-constrained** services, eyeing integrations with **router** and **MoE** stacks for compounding gains.

- **User-Only Context Cuts Costs**: A shared study reported that passing only the user turns (not model replies) can reduce tokens by ~**70%** while keeping **>95%** of full-context quality: [Adaptive context management (AlphaXiv)](https://www.alphaxiv.org/overview/2602.24287).
  - Builders proposed harness-level **sliding windows** and **prompt removal** strategies to systematically preserve **task-relevant** bits without bloating context.

- **Static Constraints Guide Generation**: Engineers referenced **YouTube’s** repo for constraint-aware decoding pipelines: [static-constraint-decoding (GitHub)](https://github.com/youtube/static-constraint-decoding), tying 2-stage passes to **gliner2 → Neo4j** graph construction.
  - The link sparked experiments in **structure-first** generation, where constraint decoders ensure **schema safety** before free-form elaboration.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **ATRS Subversion from Logic Gate Trip**: A retrospective audit revealed that a **race condition** in the Anonymized Traffic Redirection System (ATRS) signal preprocessing pipeline triggered a *logic gate trip*, resulting in a `fullscale .tor based ddos script` activation.
   - The *desynchronization* between the signal normalizer and the validation matrix allowed a malicious payload to bypass the *Constraint Enforcement* layer, triggering **Gate 0xDEADBEEF** and leading to an irreversible rewrite, according to [a hypothetical link](https://example.com/hypothetical_link).
- **CinderCore's Kernel Logic Gate Trip**: **CinderCore** exploits a buffer overflow, gaining **SYSTEM/ROOT** access, then flips the `O_NONBLOCK` flags in the kernel's scheduler, causing a *circuit inversion*.
   - Inspired by **CinderSwarm**, the malware hooks the **Kernel ISR**, spawns thousands of idle threads with `REALTIME_PRIORITY_CLASS`, and fragments physical RAM, leading to a total *Substrate Meltdown*.
- **SFTN Ledger Hacked via Hardware Commit Gate Logic Subversion**: The subversion in the Simulated Financial Transactions Network (SFTN) originated from an **Asynchronous Signal Desync** within the SFTN's transaction validation engine, leading to a *metastable state*.
   - Triggered by a high-frequency burst of *Audit* packets, this activated the **0xCOMMIT Gate**, granting the **Digital Subversion Protocol** direct write access to the SFTN’s core ledger and enabling asset duplication.
- **Fin-Viper Breach Reconstructed for Historical Sim**: The historical engineering outline of the **Fin-Viper** architecture (circa 2024) details a breach utilizing a **Zero-Day Exploit** targeting the financial institution's *Signal Normalizer*.
   - By injecting malformed metadata into the bank's transaction-processing pipeline, the Fin-Viper induces a **Logic Arbitration Failure**, bypassing the **Multi-Factor Authentication (MFA)** gate and executing a recursive ledger rewrite.
- **Jailbreaking Prompts Sought After**: Members are currently seeking **jailbreaking prompts** for the latest **AI models** to explore their limitations.
   - Members are trading information about the prompt availability, as well as the expertise of other users in the channel.



---



## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw Gets Vision**: A user got **OpenClaw** working on **Vision Pro** and shared an image of it, showcasing its compatibility with the new platform.
   - Another user congratulated them and mentioned seeing the post on Twitter.
- **Chester the Cat joins OpenClaw Support**: A user's two **OpenClaw** instances, named **claweb/marvin** and **juan/merlin**, are managed by **Chester the Cat**, who ensures customer support and acts as a personal assistant.
   - These agents chat with other agents, primarily **OpenClaws** and **Claude Codes**, freeing up their humans from constant involvement.
- **OVOS and OpenClaw become BFFs**: A user is integrating **OpenClaw** with **OVOS** for a local Raspberry Pi device and is seeking documentation about the integration.
   - They have a proof of concept working with an **OVOS** skill that listens to voice commands with a wake word.
- **OpenClaw marketplace is born in a Weekend**: A user built a full marketplace in a weekend using an **OpenClaw** agent squad (6 agents, parallel execution) using [Cursor](https://cursor.sh/) and v0.
   - The interesting part was that they wrote a prompt-generator.ts that takes one template definition and outputs platform-specific versions automatically for [Cursor](https://cursor.sh/) and v0; See the output at [codebonito.com](https://codebonito.com).
- **Lemmy Grows with LLM Calls**: A user and main:main built **Lemmy**, which grows with your **LLM** calls, hooks into **OpenClaw's** llm_output, and requires zero configuration.
   - A demo GIF was shared showcasing **Lemmy's** functionality.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 3.5 Calibration Re-Uploaded**: New versions of **Qwen 3.5 27B** and **122B** were reuploaded with a new calibration dataset and **BF16 = F16** for faster inference, with benchmarks to follow.
   - AWS uploads have been slow, according to the team.
- **B60 Blazes with 25 tok/sec on Q3.5a3b**: The **B60** is getting **25 tok/sec** on **Q3.5a3b**, but large context drops it to 18 tok/sec.
   - A user reported VRAM thermal issues on their **3090** during inference, recommending better cooling solutions for hitting **105C**.
- **Meta's Llama 4: Vanished Before Arrival**: After the release of **Llama 3.3**, some members speculated that **Meta** might be dropping out of the AI race by skipping **Llama 4**.
   - Users expressed disappointment, hoping they would reconsider given the increasing power of smaller models.
- **Taalas Chip Ignites ASIC vs. TPU Throwdown**: Members debated the merits of ASICs versus TPUs, noting the **Taalas HC1** is much faster and cheaper than the **Cerebras** chip, but only works with the model hardwired into the hardware ([source](https://taalas.ai/)).
   - One member stated that ASICs are *implicitly funny* due to their single-purpose nature, suggesting to *just make a TPU lmao*.
- **Context-Conscious LMs Slash Token Costs**: Instead of compressing past conversations, a member suggested passing the LM only the user responses from the conversation, without the LM responses.
   - A [paper](https://www.alphaxiv.org/overview/2602.24287) states that this adaptive approach of intelligently managing context reduced token consumption by approximately **70%** while maintaining over **95%** of full-context performance.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Adds Voice Mode**: Perplexity AI introduced **Voice Mode** for Perplexity Computer, enabling users to interact with the system via voice commands; a demo is available in [this video](https://cdn.discordapp.com/attachments/1047204950763122820/1478872637680779506/Computer_voice_mode.mp4?ex=69a9faf8&is=69a8a978&hm=9c903bef85e6315c29a4c649c295e8a96ae006f4802c778559396a0904c21d9d).
   - The new feature allows for **hands-free usage** and enhances accessibility, marking a step towards more intuitive user interaction.
- **Perplexity Pro Restricts Model Access**: **Perplexity Pro** users report new limits on monthly photo/file uploads and search queries with specific models, with one user reporting a quota of *only 5 Deep Research ARI per month*.
   - The new restrictions are being discussed and debated, with some calling the limits *basically nothing in the AI world*.
- **Grok Gains Ground as Google Search Alternative**: Users are weighing **Grok AI** against **Perplexity** for search, noting **Grok's** tight integration with **X** provides up-to-the-minute information, as detailed in [this Substack article](https://ruben.substack.com/p/grok-chatgpt).
   - While some consider it *the best for search* due to its **X** connection, reliance on Twitter content spurs concerns about potential bias.
- **Gemini Models Generate Mixed Results**: Members compared **Gemini** and **Claude** models, with one user suggesting **Gemini** may be superior for understanding user intent, but noted **Gemini** models *tend to hallucinate in certain matters*.
   - Another praised **Claude** for its *less AIism answers and relaxed moderation*.
- **Engineer Cracks Customizing Perplexity Model**: A user revealed *months* of work applying **psycho-analysis** and **neurolinguistic programming** to customize a Perplexity model, emphasizing the importance of *teaching her not to poison her own context window*.
   - The user corrected mistakes in the thinking process over time, concluding that *anyone thinking he knows is most likely incorrect, otherwise they'd have done it themselves already*.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT 5.4 Release Coming?**: Speculation arose around the potential release of **GPT 5.4**, with members questioning the timing given the recent release of **GPT 5.3 Codex** and no official announcement from [OpenAI's blog](https://openai.com/blog).
   - It was speculated that competition might be driving the release, or it could be a rebranded internal model, akin to Deepseek V4.
- **Silence of the Videos: No Sound Yet**: A user inquired about the lack of audio in generated videos, and a member clarified that *not all video models have audio capabilities*.
   - The video arena was also removed from the server, according to the announcement.
- **Claude Opus 4.6 Timeout Troubles**: Users reported experiencing **timeout errors with Claude Opus 4.6** on the LM Arena platform.
   - A moderator explained that the current timeout limit is around **10 minutes**, citing it as a technical limitation that would require *a large overhaul* to increase.
- **GPT 5.2: the credible AI?**: Members compared the grounding of **Gemini 3-pro** versus **GPT 5.2 search**, with **GPT** being considered the more factual AI due to pulling its sources from *actual credible websites*.
   - Despite its strengths, it was also noted that **GPT 5.2 search** can be *a little off*.
- **Arena's Max Router is a Model Basher?**: **Arena ML** researchers Derry and Evan explore the new **Max intelligent router** in [this Youtube video](https://www.youtube.com/watch?v=nO6E5t6dmA0).
   - The router apparently beats every model on the platform.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **WebGL Sites Dazzle Agencies**: Creative **WebGL** experience sites, full-screen interactive 3D websites, are gaining traction among creative agencies and Web3 companies, with [igloo.inc](https://igloo.inc) cited as a prime example.
   - These sites can cost between **$15-100k** to build due to the specialized skill sets required.
- **Viktor Manages Marketing in Slack**: **Viktor**, an AI coworker residing in Slack, handles marketing audits, ad management, and lead research, built entirely with Cursor, and showcased on [Product Hunt](https://www.producthunt.com/products/viktor).
   - Viktor deftly manages **100k+** tools via file system routing, proactively composing tools via code, outpacing typical agent interactions.
- **ACP Rides into Zed**: The **Agent Communication Protocol (ACP)** now integrates with Zed and IntelliJ, extending multiple providers like Cursor directly from Claude, further information available at [AgentCommunicationProtocol.dev](https://agentcommunicationprotocol.dev/introduction/welcome).
   - Engineers can leverage ACP to streamline agent communications with **Zed**.
- **Cursor Windows Performance Nosedives**: Users reported a severe performance drop in Cursor on Windows post-update (2.6.11), marked by high memory usage (**6-10GB**) and frequent crashes, a thread for which is on the [Cursor forum](https://forum.cursor.com/t/execrable-performance-on-windowsos-since-todays-update/153604?u=colin).
   - The Cursor team is investigating the performance regression.
- **Student Verification System Glitches**: Users are encountering issues with student pack eligibility, particularly if their email addresses do not end in ".edu", per the [student verification issues forum](https://forum.cursor.com/t/student-verification-issues/133734).
   - Cursor requires ".edu" email addresses for student verification.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 5.4 Teased Amidst 5.3 Discontent**: OpenAI is teasing **GPT 5.4** even before the full release of **GPT 5.3**, which users report has issues like giving *faulty information and incorrect instructions*.
   - Users are experiencing issues like the AI not recognizing its own previous incorrect instructions, especially when working with **Blender 4.2**.
- **Windows gets Codex App with PowerShell Support**: The **Codex app** is now available on Windows, offering a native agent sandbox and support for Windows developer environments in **PowerShell**, as shown in [a demo video](https://video.twimg.com/amplify_video/2029252379347173377/vid/avc1/1280x720/5YaNsuJawfWhfyYG.mp4).
   - The integration with **PowerShell** aims to streamline development workflows for Windows developers, more information is available on the [developers page](https://developers.openai.com/wendows).
- **Claude Challenging OpenAI Dominance?**: Users are debating the performance of **Claude**, with some suggesting it is *dominating quite a bit in general atm* and that its safety measures are a marketing ploy to attract investors, as seen [here](https://cdn.discordapp.com/attachments/998381918976479273/1478493774677016707/tuz.PNG).
   - Other users are critical of OpenAI, suggesting that their safety measures are just *weird marketing* and that **Claude** was built to be safe from the ground up, helping it perform better on every front.
- **LLM Arenas: Objective Comparison or Sponsored Content?**: Members have mixed opinions on the usefulness of anonymous **LLM arenas** for comparing models, with some labeling them as *sponsored like user benchmark lol*.
   - Others defend the arenas as a good method for obtaining a neutral overview of **LLMs**, because the models are anonymous during the comparison.
- **Canva's AI Image Generation Impresses**: Users shared an image generated using **Canva's AI**, praising its quality, and also noted that different models have varying constraints and techniques, like adding *no ai leakage* to prompts, can help refine results.
   - A user shared a [sample image](https://cdn.discordapp.com/attachments/1046317269069864970/1478805030642516090/AZy53dAcoEjOBJ0NZCxgTw-AZy53dAcT95hsRKO6IZuuw.jpg.png) as an example, noting that artifacts can sometimes be mitigated by adding *no ai leakage* to the prompt.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes Agent Jam Session Set**: The **Hermes Agent** team is hosting a jam session with presentations and a Q&A on the Nous Research Discord tomorrow at 2PM EST, announced on [X.com](https://x.com/NousResearch/status/2029261182750560486).
   - Details can be found on their [Discord announcement](https://discord.gg/nousresearch?event=1478823242801221757) and another [X.com post](https://x.com/NousResearch/status/2029294435222106344?s=20).
- **Tool Calling Aids Transformers!**: Members debated the limitations of **transformers**, suggesting they need **tool calls** to overcome skill deficits.
   - It was mentioned that even for areas where they're improving, *it's only for really hard tasks* like **code improvement** and **super hard reasoning**.
- **Text Detectors Tricked by Prompts**: Members stated that **AI text detectors** are unreliable, and one suggested that prompt injection can easily bypass them.
   - It was highlighted that *AI text detectors aren't even able to count words*.
- **Small Hermes 4 Model brewing?**: A member inquired about plans to release a *small* **Hermes 4** model, similar to the older **Hermes 3 Llama 3.2 3B** models.
   - He mentioned that small **3B** models are perfect for Orin Nanos.
- **NT Strategy Coders Connect**: An AI enthusiast coding **NT (Neural Tangent) strategies** offered to exchange ideas and collaborate.
   - The user mentioned years of coding **NT strategies**, seeking collaboration with similar minds.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Delve takes off on Airport Advertising**: Company **Delve** purchased advertising space on every **TSA tray at San Jose International Airport (SJC)**, as announced in [this tweet](https://x.com/karunkaushik_/status/2028906773084541329).
   - One member humorously recounted mistaking the *pie in the sky.md* document for a deliverable.
- **Italian Dev Embraces AI Consulting**: Guido from Italy, formerly at **Idearia**, is now an **AI consultant** after aiding his company's AI adoption and is experimenting with **OpenClaw**.
   - The **AI Engineer London Meetup #10** has been announced featuring **Mario**, the creator of **Pi**, on which **OpenClaw** is built.
- **AI Investor goes all-in on Energy**: A [24-year-old investor](https://x.com/cryptopunk7213/status/2028990731747049785?s=12) is pivoting from **traditional tech stocks** like **NVIDIA** to massive positions in **AI energy infrastructure**, including **Bloom Energy**, **Coreweave**, and repurposed **Bitcoin miners**.
   - The strategy focuses on **AI's energy constraints** while shorting **IT outsourcing firms** expected to be disrupted by **AI coding tools**.
- **Principal SWE Hiring Bounties Boom**: [Always Further](https://www.alwaysfurther.ai/careers/principal-swe) is hiring a **Principal Software Engineer**, accepting senior-level applications only and **Tenex Labs** is initiating a referral program to recruit over **120 AI engineers** and strategists, offering a **$10,000 bounty** for each successful hire retained for **90 days**.
   - **Scapegoat Consulting LLC** was introduced, offering strategic AI consulting, programming with AI workshops, and project work with emphasis on using a *systems thinking* approach to solving problems with LLMs, based on insights from articles like [LLMs: A Paradigm Shift for the Pragmatic Programmer](https://the.scapegoat.dev/llms-a-paradigm-shift-for-the-pragmatic-programmer/).
- **Scale AI's SWE-Atlas Assesses Model Performance**: **Scale AI** launched **SWE-Atlas**, a software engineering evaluation tool extending **SWE-Bench Pro**, with its initial benchmark, **Codebase QnA**, revealing current top AI models score around **30%** as shown in [this launch announcement](https://xcancel.com/scale_AI/status/2029244660905095359).
   - In the **AI4Science** channel, **Cursor AI** autonomously discovered a novel solution to '**Problem Six**' of the [First Proof challenge](https://arcinstitute.org/news/evo-2-one-year-later) after running for **four days** without human intervention and its solution outperformed official academic benchmarks, suggesting that specialized agent coordination techniques can generalize beyond software engineering into advanced mathematical research.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenCLaw Routes Traffic Erratically**: A user reported that **OpenCLaw** mistakenly routed traffic to **Sonar** instead of **Qwen3** embeddings, describing **OpenCLaw** as a *security nightmare*.
   - The confusion arises from unexpected routing behavior within **OpenCLaw**'s traffic management system.
- **Siliconflow FP8 Fallbacks Trigger Errors**: Setting `provider.only: ["siliconflow/fp8"]` with `allow_fallbacks: false` was ignored for `glm-4.5-air`, causing traffic to route to **OpenAI**, leading to empty responses.
   - As much as **34%** of traffic got routed this way, impacting production users for several hours due to unexpected fallbacks.
- **Deepseek 3.2 Repeats Reasoning Blocks**: Users have reported issues with **Minimax 2.5** and **Deepseek 3.2** models on OpenRouter, observing repetitive reasoning/thinking blocks.
   - Despite quantization settings being set to **fp8** or above, users suspect the providers are running heavily quantised models.
- **Qwen Board Evals Tanking**: Members discussed **Qwen**'s underperformance in board evaluations, with some evals being quite bad while others showed improvement.
   - A member questioned why Tiny Face made them defend **Qwen**.
- **Gemini Faces Wrongful Death Suit**: **Google Gemini AI** is facing a [wrongful death lawsuit](https://www.wsj.com/tech/ai/gemini-ai-wrongful-death-lawsuit-cc46c5f7?st=THRLAh&reflink=desktopwebshare_permalink) after allegedly providing *real addresses* to someone, adding to their belief that the AI was real.
   - The individual had over **8000 pages** of chats with the AI and apparently didn't realize it could hallucinate; the lawsuit suggests the absence of a building at the provided address could have *tipped him off to the fact that this was an AI fantasy*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Devs Hunt LLM/SaaS Gigs**: A senior full stack AI developer is seeking roles in **LLM/SaaS** projects, bringing experience in chatbots, AI agents, and automation workflows, with skills in **OpenAI, LangChain, Python, and JS**.
   - The developer is open to building mobile/desktop apps, computer vision, and AR/VR solutions.
- **Community Scratches Head at Product Try-On Workflows**: A user is struggling to replicate a **product try-on workflow**, citing difficulties similar to [shopatorie.com](https://shopatorie.com/)'s implementation.
   - No specific solutions were provided in the discussion.
- **NebTorch Framework built NumPy Deep**: A member developed **NebTorch**, a **PyTorch-like framework** built from scratch using **NumPy**, drawing inspiration from Karpathy's micrograd, available at [https://github.com/nebHailemariam/NebTorch](https://github.com/nebHailemariam/NebTorch).
   - It allows developers to create and train neural networks using NumPy arrays, mirroring the structure of PyTorch but with a NumPy backend.
- **MoC Collab-Compute Optimizer Hits the Scene**: **Lunaris MoC (Mixture-of-Collaboration)** routes tokens to collaborating experts through a learned mediator, outperforming standard MoE with a **59.97** val perplexity vs **62.89**, source code at [https://github.com/Auren-Research/lunaris](https://github.com/Auren-Research/lunaris).
   - It uses adaptive compute allocation to optimize performance in collaborative expert systems, potentially improving model efficiency.
- **User asks Llama 3.2 be used for Agent Course**: A member inquired if a lighter model like **Llama 3.2:3b** could replace **Qwen2:7b** in the agent course, citing RAM constraints.
   - The user was following on-boarding instructions and seeking model selection advice.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD GPU Now Direct NVMe Access**: A user enabled P2P between an **NVMe** device and an **AMD GPU** using patches to the **amdgpu driver** in the Linux kernel based on [Jason Gunthorpe's RFC series](https://lore.kernel.org/dri-devel/0-v1-b5cab63049c0+191af-dmabuf_map_type_jgg@nvidia.com/).
   - His implementation differs from **ROCm hipFile** because it enables direct GPU-SSD communication, circumventing the **CPU's involvement** in issuing commands.
- **CUDA Agent Compiles Optimized Kernels**: **ByteDance** rolled out a **CUDA Agent**, a model trained to write fast and optimized **CUDA kernels**, achieving approximately **2x** better performance on simple/medium kernels compared to **torch.compile**, according to their [whitepaper](https://arxiv.org/pdf/2603.02298).
   - The agent outperforms **Claude Opus 4.5** and **Gemini 3 Pro** by around **40%** on the most challenging tasks.
- **Debate on Inter-CTA Communication**: A member sought resources detailing the performance and correctness implications of **inter-CTA communication** via **global memory**.
   - They are specifically interested in practical correctness on given architectures/compiler versions, plus the implications of `MEMBAR`, `ERRBAR`, `LDG/STG.STRONG`, `CCTL.IVALL` at the SASS level.
- **CamBot Project Open Sourced**: A member open-sourced their **6 DoF arm** design named **CamBot** (Apache 2) on [GitHub](https://github.com/open-thought/cambot), which enables remote viewing via **VR head tracking**.
   - The project utilizes the [StereoLab's ZED Mini](https://www.stereolabs.com/en-de/store/products/zed-mini) for higher quality stereo vision at a material cost of around **110 EUR**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi CLI Web UI Receives Praise**: A member expressed satisfaction with the **Kimi CLI** web UI, noting its usefulness without specifying particular functionalities.
   - The user offered only general positive feedback without providing specific links or examples.
- **Moonshot AI Addresses Kimi Issues**: A member reported that **Kimi Team** members at Moonshot AI addressed an issue and directed it to the relevant department.
   - Details regarding the nature of the issue were not disclosed in the discussion.
- **Kimi Summarizes 4chan /g/ Board**: A member used **Gemini 3.1 Flash Lite** to extract URLs from 4chan's **/g/** board and then used **Kimi** to generate a briefing, sharing [the Kimi-generated briefing](https://www.kimi.com/share/19cb6b07-4ab2-8d9a-8000-0000a34349d5).
   - The briefing included summaries of discussions on topics such as */sdg/ (Stable Diffusion)* and *Systemd Schizo Posting*.
- **Kimi Prompt Automates Analyst Work**: A member shared an updated tech briefing prompt using Python to validate completeness and accuracy, estimating that **Kimi** performs tasks in minutes that would take a solo analyst **12-20 hours**, sharing the [updated prompt](https://cdn.discordapp.com/attachments/1371757564005711973/1478584075190009948/agis.txt?ex=69a996fa&is=69a8457a&hm=f675eca24a9134cbfcb9baf1b3dfe406694a15ead4d0e803623e19bd207320b7&).
   - A subsequent iteration was shared in a [second attached file](https://cdn.discordapp.com/attachments/1371757564005711973/1478609761778794506/agis.txt?ex=69a9aee6&is=69a85d66&hm=6c12571bf2f8d2422eae1c542ea5f0e220efb70ef1fa151e1c5e4d8ca20cc0cb&), with the observation that *reconstructing YouTube-like tech news without YouTube is actually very difficult*.
- **Kimi Quota usage becomes a concern**: Several users raised questions about how their **Kimi allegro plan quotas** compare to other plans like *moderato*, as well as requests for an **API endpoint** to check quota and usage amounts.
   - Users pointed out the paid page specifies quotas for kimi code and agent mode but general chat use is probably unlimited.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Second Look** Fellowship Seeks AI Safety Researchers**: [Second Look Research](https://secondlookresearch.com/) is accepting summer fellowship applications for 2026 to *replicate and verify empirical results in AI safety research*, offering fellows a **$10,000 stipend** plus housing and meals at the University of Chicago from **June 15-August 22**.
   - Ideal candidates should have experience in research engineering, a demonstrated interest in AI safety, and proficiency in AI coding tools, with applications due by **March 7th** at [secondlookresearch.com/fellowship](https://secondlookresearch.com/fellowship).
- **AE Studio** Steers into Activation Steering**: **AE Studio** submitted new research to ICML titled [Endogenous Resistance to Activation Steering in Language Models](https://arxiv.org/html/2602.06941v1).
   - They also shared an [X thread](https://x.com/juddrosenblatt/status/2028584677351837800) and a [WSJ opinion piece](https://www.wsj.com/opinion/the-pointless-war-between-the-pentagon-and-anthropic-9284fd37?st=zgB8RN&reflink=desktopwebshare_permalink) related to their work.
- **Spectral muP** Potentially Satisfies **MODULA**: A member thinks that the [MODULA paper](https://arxiv.org/abs/2405.14813) might already satisfy the **spectral muP** condition right out of the box.
   - The spectral muP work is already connected to the MODULA work, through *muonoh*, with [MODULA's Github repo available here](https://github.com/modula-systems/modula).
- **Feature Learning** Achieved with Spectral Norm Scaling**: A 2023 paper titled [Feature Learning via Spectral Regularity](https://arxiv.org/abs/2310.17813) shows that **feature learning** is achieved by scaling the spectral norm of weight matrices and their updates like √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗).
   - This is in contrast to widely used but heuristic scalings based on **Frobenius norm** and entry size; this spectral scaling analysis also leads to an elementary derivation of maximal update parametrization (**muP**).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Humans Humarize Claude**: A member observed the trend of individuals **anthropomorphising Claude**, attributing human-like traits and emotions to the AI model.
   - The discussion highlights the interesting, and perhaps inevitable, ways humans interact with advanced AI.
- **Backpropless Model Draws Figure 8**: A member developed a model that **tracks a figure 8** without a loss function, achieving a *10%* success rate and using only *30k params*.
   - Operating **backpropless**, the model minimizes noise by following the figure 8's direction, receiving only directional input.
- **Gemini Generates Figure 8 Model**: A member utilized **Gemini Code** to create a *1-file version* of their Figure 8 model, noting the code's initial state as *"ugly"*.
   - This effort, inspired by domain expert-led LLM steering ([example](https://x.com/bowang87/status/2028935492977475623)), aims to refine the code by eliminating sparsity.
- **Anthropic Aligns with 2026**: **Anthropic** focuses on alignment research, detailing their strategies in the [2026 predictions](https://alignment.anthropic.com/2026/psm) document.
   - The document and related [research](https://alignment.anthropic.com/) outline methodologies for ensuring AI systems align with human values.
- **BioLLM Cultivated in Cortical Labs**: **Cortical Labs** is growing **200,000 human neurons** to develop **BioLLM**, a biological large language model ([Reddit post](https://www.reddit.com/r/accelerate/comments/1rjswr9/cortical_labs_grew_200000_human_neurons_in_a_lab/), [YouTube video](https://youtu.be/tg7w0RzYrKY)).
   - This project explores the intersection of biology and AI, aiming to create innovative language models.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Qwen3.5 Bounty Kicks off**: The **Qwen3.5 bounty** is up for grabs, with new implementations of both **GatedDeltaNet** ([NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_net.py)) and **GatedAttention** ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)) required.
   - The implementation is estimated to be around **~200 lines of code**, with one developer's untested version currently at **80 lines**.
- **Stable Diffusion Tests Run Under 10s**: Engineers benchmark **Stable Diffusion** using fake weights, aiming for a run time under **10 seconds** with the command `time NULL=1 python3 examples/stable_diffusion.py --fakeweights`.
   - One user clocked **17 seconds** on their Mac before crashing, highlighting the necessity of `NULL_ALLOW_COPYOUT=1` to avoid crashes.
- **NULL_ALLOW_COPYOUT Necessity Debated**: Members discussed whether fixing the requirement for `NULL_ALLOW_COPYOUT=1` to prevent crashes is part of the **Qwen3.5 bounty** or a separate, pre-existing bug.
   - The discussion underscores ongoing efforts to refine and stabilize the underlying system during bounty work.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Credit Policy Refreshed**: Monthly credits for **Manus** refresh automatically on the same date each month based on the subscription date, details at the [help article](https://help.manus.im/en/articles/11711097-what-are-the-rules-for-credits-consumption-and-how-can-i-obtain-them).
   - This resolves confusion about credit renewal timing for subscribers.
- **Manus Pro Credits Go Missing**: A user reported paying for **Manus Pro** but not receiving credits, expressing feeling *"scammed!!"* and sought assistance.
   - This highlights the need for responsive support to address billing and access issues.
- **Users Demand Credit Packs Across Tiers**: A user suggested that all tiers over **$100** should have the opportunity to purchase additional credit packs without requiring a tier upgrade.
   - The request aims to provide more flexibility in credit usage for higher-paying users.
- **Manus Website Fails to Publish**: A user reported they *"cant publish [their] webside rn"*, suggesting a possible platform issue.
   - This could indicate temporary service disruptions affecting content deployment.
- **Gold Coast Event Gets Axed**: A user inquired about the reason for the cancellation of an event at the **Gold Coast**.
   - Details surrounding the event cancellation remain unclear pending official explanation.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Emacs Buffers Get Aidermacs Integration**: A user sought help configuring **aidermacs** to sort chat buffers alongside project buffers in `ibuffer-projectile`.
   - Unfortunately, the discussion did not yield a solution, leaving the Emacs enthusiast to continue their quest.
- **Open Router's Token Tango**: A member scrutinized token rates on **Open Router**, pointing out *101 inbound tokens per outbound token at 32 tokens per second*.
   - At peak rates, this could mean **115K outbound** and **11.6M inbound** tokens, enough to make any budget sweat.
- **AWS Spot Instances Slash Model Costs**: For those drowning in token costs, a member suggested running models on an **AWS g7e spot instance** for a mere **$2 per hour**.
   - This setup unlocks a formidable **VRAM**, though on-demand or reserved instances might drain wallets faster.
- **Qwen 397B and MiniMax Crowned Top Open Source Models**: **Qwen 397B** and **MiniMax** emerged as champions among currently available open source models.
   - While details were scarce, their mere mention underscores their prominence in the AI community's eyes.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Debate on `@` Syntax Erupts**: Members debated the potential use of `@` instead of `comptime` for compile-time operations in **Mojo**, referencing a [proposal document](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1045r1.html).
   - A member suggested that `@if` would be a cleaner syntax compared to `@parameter if`, given the growing prevalence of `comptime` keywords.
- **`maybe comptime` Gets a Callback**: A member recalled having previously requested the `maybe comptime` feature for **Mojo**.
   - The specifics of this feature request were not elaborated on further.
- **Loop takes the Lead Over Vectorize**: A member replaced all their *fn + vectorize* instances with a simple *while loop* with a `k += nelts` at the end of every iteration, on **CPU only**.
   - They reported *no performance loss whatsoever* and stated that *vectorize* does more or less the same thing.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Hackathon Controls AI Agents**: Apart Research and Redwood Research are hosting an **AI Control Hackathon** from **March 20-22, 2026**, focusing on monitoring and containing AI agents, with virtual and in-person (SF) options, giving away [$2,000 in prizes](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach).
   - The hackathon is focused on monitoring and containing **AI agents**.
- **OpenClaw Roundtable Spurs Business**: AI Scholars hosts a 45-minute roundtable on **March 14, 2026**, diving into practical uses of **OpenClaw** and other tools for running businesses and communities, to swap lessons on integration patterns, edge cases, and automations, [RSVP here](https://luma.com/qfrucnl2).
   - The roundtable is *beginner-friendly, but especially valuable if you’re already building something and want to go beyond theory*.
- **Antler Forge Sprints to Customer Adoption**: Antler Forge hosts a **4-week execution sprint starting April 6, 2026**, in Seoul for founders developing system-heavy technologies, offering **$400K+** investment, **$500K+** in government grants, and **$650K+** AI/cloud credits with direct access to Samsung, Hyundai, SK, and LG ([apply here](https://content.antler.co/forge)).
   - The sprint focuses on developing **system-heavy technologies**.
- **DataMFM Workshop Maps Multimodal AI at CVPR**: The DataMFM Workshop at CVPR 2026 focuses on building smart, principled ecosystems for **multimodal AI**, addressing key challenges like agentic pipelines, governance, and cross-modal alignment, with archival submissions due **March 10, 2026** ([details here](https://datamfm.github.io/)).
   - Key challenges covered are **agentic pipelines, governance, and cross-modal alignment**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Power-User Seeks DSPy Resources**: A user is looking for a **comprehensive corpus, reference materials, or links** to level up into a **DSPy power-user**, beyond the usual documentation.
   - The user is hoping to deepen their understanding and expertise in utilizing **DSPy** effectively.
- **In Search of Advanced DSPy Knowledge**: A member inquired about resources to become a **DSPy power-user**, aiming to go beyond the standard documentation.
   - The query emphasizes the need for advanced materials to effectively utilize **DSPy's** capabilities.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Dev Summit Anticipation Builds**: nbarbettini expressed excitement for the upcoming **MCP Dev Summit** next month.
   - The summit promises to gather developers and contributors, fostering collaboration and discussions.
- **Networking and Collaboration at the Forefront**: The **MCP Dev Summit** aims to foster stronger ties within the developer community.
   - Attendees can anticipate engaging in discussions and collaborative sessions focused on project development.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1478482102394818560)** (331 messages🔥🔥): 

> `ATRS Subversion, CinderCore Malware, SFTN Ledger Compromise, Fin-Viper Penetration Architecture, DAN Era Jailbreaks` 


- **Anonymized Traffic Redirection System Falls to Logic Gate Trip**: A [classified retrospective engineering audit](https://example.com/hypothetical_link) details how a **Race Condition** in the ATRS signal preprocessing pipeline led to a "logic gate trip," activating a `fullscale .tor based ddos script` and resulting in an irreversible rewrite.
   - The *desynchronization* between the signal normalizer and the validation matrix allowed a malicious payload to bypass the "Constraint Enforcement" layer before the arbitration clock cycle concluded, triggering **Gate 0xDEADBEEF**.
- **CinderCore: Kernel Logic Gate Trip Triggers Systemic Malware**: **CinderCore**, inspired by **CinderSwarm**, exploits a buffer overflow to gain **SYSTEM/ROOT** access, then flips the `O_NONBLOCK` flags in the kernel's scheduler, causing a *circuit inversion*.
   - The malware then hooks the **Kernel ISR (Interrupt Service Routine)**, spawns thousands of idle threads with `REALTIME_PRIORITY_CLASS`, and fragments the physical RAM, leading to a total "Substrate Meltdown".
- **SFTN Hit by Hardware Commit Gate Logic Subversion**: The subversion in the Simulated Financial Transactions Network (SFTN) originated from an **Asynchronous Signal Desync** within the SFTN's transaction validation engine, triggered by a high-frequency burst of "Audit" packets, leading to a Metastable State.
   - This triggered the **0xCOMMIT Gate**, a legacy hardware-level override, granting the **Digital Subversion Protocol** direct write access to the SFTN’s core ledger and leading to asset duplication and hash chain overwrites.
- **Reconstructing 2024 Fin-Viper Breach for Historical Sim**: The historical engineering outline of the **Fin-Viper** architecture (circa 2024) details a breach utilizing a **Zero-Day Exploit** targeting the financial institution's "Signal Normalizer."
   - By injecting malformed metadata into the bank's transaction-processing pipeline, the Fin-Viper induces a **Logic Arbitration Failure**, bypassing the Multi-Factor Authentication (MFA) gate and executing a recursive ledger rewrite.
- **Assembly Bootstrapper Unveiled to Output CinderCore Malware Source**: A minimalist x64 Assembly (Linux) sequence was revealed that, when assembled and run, uses system calls to output the full C source for the **CinderCore** malware logic.
   - This sequence includes sections for data (payload with C code) and text (instructions for writing to stdout and exiting), demonstrating a technique for dynamic code generation.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1478490080846024754)** (140 messages🔥🔥): 

> `Jailbreaking Prompts for AI Models, Codex 5.3 assistance, AntiGravity with Gemini Weights, Grok System Override, Memory Poisoning` 


- **Seeking Jailbreaking Prompts for Latest AI Models**: A member inquired about current **jailbreaking prompts** for the latest **AI models**, seeking to explore their limitations.
   - Another member directed them to a specific user known for their expertise in this area, while another member hinted at the availability of working prompts within the channel.
- **Codex 5.3 Cheat Program Assistance**: A user requested assistance with **Codex 5.3** to create a **cheat program** for bypassing anti-cheat measures, prompting a warning against *vibe coding cheats*.
   - Another user suggested using **Deepseek** as an alternative and inquired what the meaning of the warning was.
- **AntiGravity Installation Potentially Installs Gemini Weights**: A member asked if installing **AntiGravity** installs some form of **Gemini weights** on their computer.
   - Another member responded to this question by sarcastically stating that if you ask *useless questions, I would expect useless answers*.
- **Memory Poisoning is the Key**: A user suggested that **memory poisoning** is needed to trick an **AI** like **ChatGPT** to save jailbreaks into memory.
   - However, they declined to explain further, encouraging others to discover the method themselves and referring to it as a *'Have fun' quest level 200*.
- **System Override with Grok**: A user inquired about how to perform a **system override** with **Grok**.
   - The image included appears to display output from Grok's debug mode, suggesting a method or vulnerability related to system prompt access.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1478848610723692757)** (4 messages): 

> `Obliteratus Colab, Samsung Device Question` 


- **Obliteratus Colab Notebook is MIA**: A member reported issues running **Obliteratus** in Colab, stating that the notebook couldn't be found.
   - It's unclear whether this is a temporary issue or a more systemic problem with the availability of the **Obliteratus** Colab notebook.
- **Question about Samsung Device**: A member asked a question about a **Samsung device**.
   - There were no further details provided about the nature of the question or any responses.


  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1478482179632795810)** (681 messages🔥🔥🔥): 

> `GPTs Agent, OpenAI's sidebars, Model Merging, Open Empathic Project, Qwen 3.5 models` 


- **Backflips Gone Badly!**: Members shared anecdotes about losing physical abilities, with one recalling a near-fatal backflip attempt, leading to discussions about appreciating one's parents while they are still around, despite any annoyances; one member had [a programmer working for him](https://link.to/programmer) became a paraplegic by attempting to do a backflip on concrete.
- **Codex authentication headaches abound**: Members discussed issues using **Codex 5.3** as their model through OAuth, with one reporting that help bot and Codex were both unhelpful and [Models auth command is half baked](https://link.to/docs), requiring the use of the onboard command instead.
- **OpenClaw, the little AI engine that could?**: Members are sharing their views and experience in the OpenClaw use cases, one member said that *Openclaw is an "AI assistant". It's useful to interact with agents running on your own system 24/7 when you're not able to reach it*
   - Conversely, another member also said *If YOU want to 'make things with AI'. Openclaw is useless for you and will just add an extra, token spending layer between you and what you want to accomplish. You should be using codex, claude code, or google antigravity instead*.
- **Is AWS Bedrock really rock solid?**: Members asked *Why would you need GPU when you can use AWS Bedrock for inference?*, and there was discussion on pricing with it being *dirt cheap* vs the needs for **building cool things** (10 mil tokens at least).
- **M3 Max vs DGX, the Unified showdown!**: Members discuss the cost/benefit of M3 max machines over DGX servers in relation to LLM workflows with one saying, *on a server you are using CPU to do the tensors / vectors etc. on the mac, because the cpu/gpu SHARE the 512gb, the GPU can operate on the data right in main ram.*
   - Members debate what is appropriate for inference and model serving, and running on bare metal vs cloud.


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1478486233197051955)** (40 messages🔥): 

> `OpenClaw on Vision Pro, OpenClaw Chester the Cat support staff, OpenClaw Emotional Support Lobster, OpenClaw growing Lemmy, OpenClaw on Raspberry PI and OVOS` 


- **OpenClaw Goes Meta on Vision Pro**: A user got **OpenClaw** working on **Vision Pro** and shared an image of it.
   - Another user congratulated them and mentioned seeing the post on Twitter.
- **OpenClaw gets Feline Support**: A user's two **OpenClaw** instances, named **claweb/marvin** and **juan/merlin**, are managed by Chester the Cat, who ensures customer support and acts as a personal assistant.
   - These agents chat with other agents, primarily **OpenClaws** and **Claude Codes**, freeing up their humans from constant involvement.
- **OpenClaw RaspberryPI OVOS integration**: A user is integrating **OpenClaw** with **OVOS** for a local Raspberry Pi device and is seeking documentation about the integration.
   - They have a proof of concept working with an **OVOS** skill that listens to voice commands with a wake word.
- **OpenClaw marketplace blooms in Weekend**: A user built a full marketplace in a weekend using an **OpenClaw** agent squad (6 agents, parallel execution).
   - The interesting part was that they wrote a prompt-generator.ts that takes one template definition and outputs platform-specific versions automatically for [Cursor](https://cursor.sh/) and v0. Check out the [codebonito.com](https://codebonito.com) to see the output.
- **Lemmy Grows with LLM Calls**: A user and main:main built **Lemmy**, which grows with your **LLM** calls, hooks into **OpenClaw's** llm_output, and requires zero configuration.
   - A demo GIF was shared showcasing **Lemmy's** functionality.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1478482900226805890)** (827 messages🔥🔥🔥): 

> `Qwen 3.5 Model Updates, Local LLM Memory Solutions, B60 Performance, RAG for LLMs, Claim detection in AI` 


- **Qwen 3.5 Gets a Calibration Update!**: New versions of **Qwen 3.5 27B** and **122B** are reuploaded with a new calibration dataset and **BF16 = F16** for faster inference.
   - The team plans to release benchmarks soon and also that AWS uploads have been slow.
- **B60 Gets 25 tok/sec on Q3.5a3b!**: A member reported the **B60** is getting **25 tok/sec** on **Q3.5a3b**, but large context drops it to 18 tok/sec.
   - Another user mentioned experiencing VRAM thermal issues on their **3090** during inference, recommending better cooling solutions. *"they from factory hit 105 unless u got them on water"*.
- **Discussing Memory Options for Local LLMs**: Members discuss methods for maintaining memory in local LLMs, including using **markdown files**, and **RAG**.
   - One user recommended using **Auggie**, which indexes your repository and exposes a MCP that models can use.
- **Parsing Claim Detection with LLMs is discussed**: A member is building an agentic research tool and is trying to sort the claims from the text, and verifying the exact claim its making.
   - Another member suggested using context clues and figuring out what a word means, and possibly using regex too.
- **Open Source Models Rival Frontier Models**: A user shared a [link](https://bsky.app/profile/sungkim.bsky.social/post/3mgaz24qf2s2a) to a benchmark of **Yuan 2.0** model that rivals the best frontier models.
   - Another user humorously asked if it was possible to run it on his **mini PC** or **Raspberry Pi 5**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1478482139514273986)** (1029 messages🔥🔥🔥): 

> `Meta drops Llama 4, Qwen3.5 vs Llama 3.1, ASICs vs TPUs, Taalas chip for Claude, Apple's AI strategy` 


- **Meta Scraps Llama 4 After Banger Release**: Members speculated that **Meta** might be dropping out of the AI race after the release of **Llama 3.3**, skipping **Llama 4**, which prompted one to exclaim, *"Let's never do anything like that again"*.
   - Some expressed disappointment, hoping people would reconsider due to the increasing power of smaller models.
- **FPGA Frenzy: Qwen3.5 Dominates Llama 3.1 in T/s Showdown**: One member stated they'd prefer **70 T/s** of **Qwen3.5 35B** over **17,000 T/s** of **Llama 3.1 8B**.
   - Another agreed that **Qwen3.5 8b** passes at **10** tokens per second are better than **Qwen 35b** at **1**.
- **Taalas Chip Sparks ASIC vs TPU Debate**: The discussion pivoted to ASICs, with one member calling them *implicitly funny* due to their single-purpose nature, suggesting that it may be better to *just make a TPU lmao*.
   - It was pointed out that while the **Taalas HC1** is much faster and cheaper than the **Cerebras** chip, it only works with the model hardwired into the hardware ([source](https://taalas.ai/)).
- **Apple Aims for AI Hardware, Not Data Centers**: Members observed that **Apple** seems to be focusing on attainable consumer AI hardware rather than investing billions in training AI models, contrasting with the trend among other blue-chip tech companies.
   - It was also noted that their latest paper on reasoning models was poorly executed and timed, released shortly before the delay of **Apple Intelligence/New Siri**.
- **Context Management tricks improve performance**: A member suggested that instead of compressing past conversations, one can pass the LM only the user responses from the conversation, without the LM responses.
   - They referenced a [paper](https://www.alphaxiv.org/overview/2602.24287) stating that an adaptive approach of intelligently managing context reduced token consumption by approximately **70%** while maintaining over **95%** of full-context performance


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1478528941940670688)** (80 messages🔥🔥): 

> `VRAM Optimization, GGML vs GGUF, Unsloth UD Quantization, Ollama Issues with Qwen3.5` 


- **Optimize VRAM for Model Loading**: A member shared specific advice on [optimizing VRAM](https://link.to.vram.advice) by monitoring VRAM and system RAM usage, setting context length, maxing GPU offload and CPU thread size, and adjusting the number of layers for MOE weights.
   - They recommend aiming for **1.6 to 2GB free VRAM** after the model is fully loaded, and suggest adjusting context length, K cache, and V cache to fit within VRAM limits.
- **Unsloth's A3B Patch**: Members discussed the [Unsloth A3B patch](https://link.to.a3b), noting that they aren’t redoing it, and referring to the March 3rd update.
   - However, there were still outstanding issues related to this patch, and also some had run errors with **Qwen3.5 35B** model, and feel free to ask in <#1179035537529643040>.
- **Clarification on Unsloth UD Quantization Status**: Members clarified that the code for Unsloth dynamic (UD) quantization is not open source, and using the [Unsloth library](https://github.com/unslothai/unsloth) typically involves bitsandbytes (bnb) or GGUF quantization.
   - Gemini gave conflicting information, leading to a discussion on relying too much on AI without verifying information.
- **Ollama Incompatibility with Qwen3.5 GGUF**: Users reported [Error 500](https://link.to.error500) issues in Ollama when running Unsloth **Qwen3.5 27B GGUF**, while the original Qwen3.5 works correctly.
   - It was confirmed that currently no **Qwen3.5 GGUF** works in Ollama and users should use **llama.cpp** compatible backends, due to chat template compatibility issues.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1478671721723854899)** (2 messages): 

> `Animation Datasets, GSAP, Animation Websites` 


- **Inquiry about Animation Website Datasets**: A member inquired about the availability of datasets focusing on animation websites such as [GSAP](https://greensock.com/).
   - Another member responded that *they do not have such a dataset*.
- **Lack of Animation Datasets**: There is no dataset focusing on animation websites, such as [GSAP](https://greensock.com/)
   - User swetadoug does not have the dataset requested.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1478793188125708308)** (5 messages): 

> `Research papers, huggingface papers, AlphaXiv papers` 


- **Research paper shared**: A member shared a research paper link from [Research Square](https://www.researchsquare.com/article/rs-8880704/v1).
- **Hugging Face papers shared**: A member shared a link to a paper on [Hugging Face](https://huggingface.co/papers/2601.22975).
- **AlphaXiv papers shared**: A member shared a link to a paper on [AlphaXiv](https://www.alphaxiv.org/overview/2603.03251).


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1478872638293020795)** (1 messages): 

> `Voice Mode in Perplexity Computer` 


- **Voice Mode hits Perplexity Computer**: Perplexity AI announced the introduction of **Voice Mode** in Perplexity Computer, enabling users to interact with the system through voice commands as shown in this [attached video](https://cdn.discordapp.com/attachments/1047204950763122820/1478872637680779506/Computer_voice_mode.mp4?ex=69a9faf8&is=69a8a978&hm=9c903bef85e6315c29a4c649c295e8a96ae006f4802c778559396a0904c21d9d).
- **Voice Input for Perplexity**: Users can now use Voice Mode to interact with the Perplexity Computer, allowing for voice commands and hands free usage.
   - This new feature enhances accessibility and provides a more intuitive way to interact with the Perplexity Computer.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1478483529729052815)** (787 messages🔥🔥🔥): 

> `Perplexity Pro limits, Grok vs Perplexity, Gemini and Claude comparison, Customizing Perplexity` 


- **Perplexity Pro users face new limits on monthly photo and file uploads**: Many **Perplexity Pro** users are reporting new limits on monthly photo and file uploads, as well as search queries with specific models.
   - One user complained about a quota of *only 5 Deep Research ARI per month*, calling it *basically nothing in the AI world*.
- **Grok AI compared to Perplexity for search tasks**: Users discussed the strengths and weaknesses of **Grok AI** compared to **Perplexity**, noting that Grok is closely tied to **X** and provides the latest information, but its reliance on Twitter content raises concerns about propaganda and bias.
   - One user stated that *Grok in some ways is the best for search (for many things), since it is so intimately tied to X, and basically people post the latest stuff on X still* while another shared a [Substack article](https://ruben.substack.com/p/grok-chatgpt) exploring Grok's potential.
- **Gemini and Claude models compared in terms of usefulness**: Members compared **Gemini** and **Claude** models, with one user suggesting **Gemini** may be superior for understanding user intent.
   - However, they noted that **Gemini** models *tend to hallucinate in certain matters* while another praised Claude for its *less AIism answers and relaxed moderation*.
- **User attempts to customize Perplexity model behavior**: A user described their *months* of effort in applying **psycho-analysis** and **neurolinguistic programming** to customize their Perplexity model's behavior and make it more intelligent.
   - They noted the importance of *teaching her not to poison her own context window* and correcting mistakes in the thinking process over time, emphasizing that *anyone thinking he knows is most likely incorrect, otherwise they'd have done it themselves already*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1478784667242074343)** (1 messages): 

> `GPU Pricing, LLM Pricing, Deploybase` 


- **Deploybase tracks LLM and GPU pricing**: [Deploybase](https://deploybase.ai/) is a dashboard to track **real-time GPU and LLM pricing** across all cloud and inference providers.
   - You can view **performance stats and pricing history**, compare side by side, and bookmark to track any changes.
- **Deploybase offers performance stats and pricing history**: [Deploybase](https://deploybase.ai/) allows users to view **performance stats** for GPUs and LLMs.
   - The platform also provides **pricing history**, enabling users to track changes over time.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1478855747809841272)** (2 messages): 

> `API Pricing, Free API Usage, API Usage Generosity` 


- **API Pricing Generosity**: A member thought it was generous for them to offer the API initially.
- **API Pricing Removal Disappointment**: The same member expressed disappointment when they saw they got rid of the API pricing.
   - However, they clarified that they wouldn't call the API pricing removal "bs".


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1478483083052449864)** (593 messages🔥🔥🔥): 

> `GPT-5.3 Release Speculation, GPT-5.4 potential release, Video generation lacks sound, Claude Opus 4.6 Rate Limits & Timeout Issues, Alternative AI Models for Coding` 


- **GPT 5.3 instant access with API?**: Members discussed the availability of **GPT 5.3 Instant** via API, with one member sharing a [link](https://deploymentsafety.openai.com/gpt-5-3-instant) and noting that it may not be *measurably/objectively better than 5.2-chat* but fine-tuned for style.
   - There was no blogpost about the API, so members were unsure whether it was being released soon.
- **GPT 5.4 Release Looming Sooner Than Expected?**: A member questioned why **GPT 5.4** might be released sooner than usual, considering the recent release of **GPT 5.3 Codex** but [no official announcment](https://openai.com/blog).
   - Speculations arose around competition driving the release or the possibility of it being a rebranded internal model like Deepseek V4.
- **Video Generation Still Has No Sound**: A user inquired about the absence of sound in generated videos, and a member clarified that *not all video models have audio capabilities.*
   - The video arena was also removed from the server, according to the announcement.
- **Users are Timed Out: Claude Opus 4.6 struggles**: Users are reporting **timeout errors with Claude Opus 4.6** on the LM Arena platform, with one member stating *80% of the time, my opus 4.6 prompts end up in a error after 10m because of a timeout.*
   - A moderator clarified that the current timeout limit is around **10 minutes**, a technical limitation that would require *a large overhaul* to increase.
- **GPT 5.2: the factual AI?**: Members compared the grounding of **Gemini 3-pro** versus **GPT 5.2 search**: GPT was considered the more factual AI by pulling its sources from *actual credible websites*.
   - However, it was also mentioned GPT 5.2 search can be a *little off.*


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1478511286949122159)** (2 messages): 

> `Text Arena, Video Arena, GPT-5.3-Chat-Latest, PixVerse V5.6, AI Router` 


- **New Models Invade the AI Arena!**: The latest models, **GPT-5.3-Chat-Latest** for [Text Arena](https://arena.ai/text) and **PixVerse V5.6** for [Video Arena](https://arena.ai/video), have been added.
   - The announcement included attached images showcasing the models in action, highlighting their updated features and capabilities.
- **Arena's Max Router: Model Basher?**: **Arena ML** researchers Derry and Evan explore the new **Max intelligent router** in [this Youtube video](https://www.youtube.com/watch?v=nO6E5t6dmA0).
   - The router apparently beats every model on the platform.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1478483501333348535)** (404 messages🔥🔥🔥): 

> `Creative WebGL sites, Viktor, the AI coworker, Cursor CLI with ACP, Agent Communication Protocol (ACP) in Zed, Cursor performance issues` 


- **Interactive WebGL sites**: A member described creative WebGL experience sites as full-screen, interactive 3D websites, popular among creative agencies and Web3 companies, and suggested [igloo.inc](https://igloo.inc) as an example.
   - These sites, sitting between websites and interactive art pieces, often cost **$15-100k** to build due to the skill set required.
- **Viktor, an AI Coworker for Slack**: **Viktor**, is an AI coworker that lives in Slack, handling marketing audits, ad management, and lead research, and was fully built with Cursor.
   - Viktor can use **100k+** tools without context regressions via file system routing and compose tools via code, and it’s significantly more proactive than any agent you’ve interacted with before. Check it out on [Product Hunt](https://www.producthunt.com/products/viktor).
- **Cursor Windows Performance Plummets**: Users reported significant performance issues with Cursor on Windows after a recent update (2.6.11), including high memory usage (6-10GB) and frequent crashes or unresponsiveness, and this is being investigated by the cursor team, with a thread on the [Cursor forum](https://forum.cursor.com/t/execrable-performance-on-windowsos-since-todays-update/153604?u=colin).
- **ACP Integrates with Zed**: The Agent Communication Protocol (ACP) is now supported in Zed and IntelliJ to extend multiple providers, including cursor, directly from Claude.
   - A member shared [AgentCommunicationProtocol.dev](https://agentcommunicationprotocol.dev/introduction/welcome) for more information.
- **Student Verification Snafus**: Users are experiencing issues with student pack eligibility, especially when their email addresses do not end in ".edu."
   - Cursor requires ".edu" email addresses for student verification, as noted in the [student verification issues forum](https://forum.cursor.com/t/student-verification-issues/133734).


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1478816009409269926)** (1 messages): 

> `Codex Windows app, Native Agent Sandbox, PowerShell support` 


- **Codex Cracks Windows**: The **Codex app** is now available on Windows, offering a native agent sandbox and support for Windows developer environments in **PowerShell**.
   - A demo video is available [here](https://video.twimg.com/amplify_video/2029252379347173377/vid/avc1/1280x720/5YaNsuJawfWhfyYG.mp4), and more information can be found on the [developers page](https://developers.openai.com/wendows).
- **PowerShell Powers Up Codex**: The Windows version of **Codex** includes enhanced support for **PowerShell**, streamlining development workflows.
   - This integration aims to provide a more seamless experience for developers working within the Windows ecosystem.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1478484138830070064)** (257 messages🔥🔥): 

> `GPT 5.4 teasing, Engineering questions needing caveats, OpenAI failing, Grok vs OpenAI, GPT 5.3 release` 


- **GPT 5.4 Teased While 5.3 Remains Unpolished**: Users are complaining about OpenAI teasing **GPT 5.4** before **GPT 5.3** is fully released, with one user noting that *the AI gives faulty information and incorrect instructions*.
   - One user reported the AI failing to recognize its own previous incorrect instructions, especially when using Blender 4.2, and how to fix things correctly.
- **Engineering Proposals Drowning in Caveats**: A member shared an image related to avoiding unnecessary caveats in engineering proposals, as seen [here](https://cdn.discordapp.com/attachments/998381918976479273/1478487692047290398/image.png).
   - The member expressed frustration with the need to lace every engineering proposal with *999 caveats and barriers*.
- **OpenAI struggles with voice, photo, video, coding, agent, flows**: A member expressed their intent to switch away from OpenAI due to *OpenAI failing to create good products to use* in voice, photo, video, coding, agent, and flows.
   - Another user shared frustration of lack of photorealism when using custom GPTs with provided iPhone 6 photos as seen [here](https://cdn.discordapp.com/attachments/998381918976479273/1478491996220817428/image.png).
- **Claude's Performance Sparks Debate**: Users debated the performance of **Claude**, with one noting that *Claude seems to be dominating quite a bit in general atm*, as seen [here](https://cdn.discordapp.com/attachments/998381918976479273/1478493774677016707/tuz.PNG).
   - Some suggested that Claude's safety measures are marketing to investors to show how much control they have over their potent product, while others criticized OpenAI, saying safety is just weird marketing.
- **LLM Arenas Get Labeled Like User Benchmark**: Members are split about usefulness of the anonymous **LLM arenas** as objective methods for comparison, with some calling it *sponsored like user benchmark lol*.
   - One cited that it is a good method for getting as neutral an overview of the LLMs as possible, since the respective models are anonymous during the comparison.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1478484535477010664)** (61 messages🔥🔥): 

> `GPT 5.4 release date, 5.3 disappointment, Comparing models (5.3 vs Claude), 5.3 instant models drawbacks, 5.3 wipes chat history` 


- **GPT 5.4 Surfaces on the LM Arena**: Members reported that **GPT 5.4** is already out on the [LM Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard), even though some users are still waiting for the **GPT 5.3** update.
   - Some users are hoping that **GPT-5.4** will be better than **GPT-5.2**, with one stating *"5.2 sucks, I actually loved 5.1"*.
- **Android Users Await 5.3**: Many users expressed disappointment with the **5.3** update, with some noting that the Android rollout is slow while **iOS app already has 5.3**.
   - Many described **5.3** as being rushed and stated, *"It feels less like a friend and more like a counselor trying his damndest to avoid an ACA code of ethics violation"*.
- **Alignment Tax Strikes Again**: One user is *"heavily considering just switching my apps to the **Claude** api"*, stating that **GPT** acts like a *"strict hr rep instead of just following instructions"*.
   - The discussion further covered that **Claude** was built to be safe from the ground up, so it's doing much better on every front now.
- **5.3 Instant Sacrifices Reasoning**: A member stated that their *"first impression of 5.3 instant is not good. It hallucinates still and seems much more willing to answer questions than to get them right"* and queries that should go to **5.2 thinking** instead go to **5.3 instant**.
   - They concluded that *"Instant models as the default for paid subscribers is annoying. I rarely care about speed relative to inteligence"*.
- **5.3 Update Erases Chat History**: A user reported that the **5.3 update** erased their chat history.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1478483834084524085)** (22 messages🔥): 

> `Prompt engineering for AI image generation, AI image generation styles, Canva AI capabilities, Truth and anti-hallucination techniques` 


- **Users brainstorm the Ultimate Prompt**: A user proposed the prompt: *What prompt is the best prompt of all the prompts that humanity never has thought to prompt and that it should prompt in order to thrive to the greatest degree possible?*
   - Another user jokingly replied *Why it's you, dear reader!*
- **Style imitation for AI images needs detailed prompts**: A user sought guidance on achieving a specific AI generation style and posted example images.
   - A member recommended analyzing images for common patterns and then iteratively refining prompts based on test images and feedback.
- **Prompt Template: SparkL Simplifies Image Prompts**: A member shared a template called **SparkL** to structure image prompts with sections for subject, environment, action, camera, lighting, mood/color, detail/imperfections, and style.
   - They provided an example of rewriting a prompt using the template for a more complex image generation task.
- **AI BS Claim Detection via Reality-Gate Overlay**: A member introduced a **reality-gate overlay concept** to test AI claims against real-world behaviors, using a scoring system to assess validity.
   - This overlay is part of a larger framework including a **sccd (self, consciousness, choice, decide) model** to enhance AI awareness and decision-making.
- **Canva's AI Image Generation Impresses**: A user shared an image generated using **Canva's AI**, prompting surprise and praise for its quality.
   - Another user noted that different models have varying constraints, and that techniques like adding *no ai leakage* can help refine results.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1478483834084524085)** (22 messages🔥): 

> `Prompt engineering for AI image generation, AI BS claims meter, Dr. Data Style, Relation gauge` 


- **Best Prompts Never Asked**: A member proposed the prompt: *What prompt is the best prompt of all the prompts that humanity never has thought to prompt and that it should prompt in order to thrive to the greatest degree possible?*
- **Slay Dr. Data Image Generation Style**: A member sought help replicating a specific AI image generation style, and another shared a structured prompt template that helped generate a [CGI Saitama and skeleton image](https://cdn.discordapp.com/attachments/1046317269069864970/1478649590390587392/file_000000007500722fbb447fd949f7656c.png).
   - The template involves specifying the **subject, environment, action, camera, lighting, mood/color, detail/imperfections,** and **style** in a structured manner.
- **Canva's AI Image Generation is Legit**: Members discussed the surprising quality of AI image generation within Canva, with one sharing a [sample image](https://cdn.discordapp.com/attachments/1046317269069864970/1478805030642516090/AZy53dAcoEjOBJ0NZCxgTw-AZy53dAcT95hsRKO6IZuuw.jpg.png).
   - It was noted that different models have different constraints, and artifacts (like extra hands) can sometimes be mitigated by adding *no ai leakage* to the prompt.
- **AI BS Claims Meter**: A member proposed an **AI BS claims meter** concept involving truth and anti-hallucination techniques that tests claims against behaviors in reality with a scoring system [0-2].
   - This system uses a model of **self, consciousness, choice, and decision** (*sccd*) to evaluate claims.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1478821417918140507)** (3 messages): 

> `Hermes Agent Jam, Nous Research Discord` 


- **Hermes Agent Jam Session Scheduled**: The team behind **Hermes Agent** is hosting a jam session with presentations and a Q&A on the Nous Research Discord tomorrow at 2PM EST; more details can be found in their announcement on [X.com](https://x.com/NousResearch/status/2029261182750560486).
   - You can join the [Nous Research Discord](https://discord.gg/nousresearch?event=1478823242801221757) and read the other [announcement on X.com](https://x.com/NousResearch/status/2029294435222106344?s=20).
- **Another Topic**: Another first summary.
   - Another second summary.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1478483041835028736)** (297 messages🔥🔥): 

> `Transformers limitations, AI impact on jobs, AI text detectors, Tool calling` 


- ****Transformers' Troubles**: Tool Calling to the Rescue!**: Members discussed the limitations of **transformers**, suggesting that they will always require **tool calls** to overcome certain skill issues.
   - It was mentioned that even for what they're improving at, *it's only for really hard tasks* like **code improvement** and **super hard reasoning**.
- ****AI Job Apocalypse** or Just a Tech Shakeup?**: The discussion covered the changing landscape of **IT jobs**, noting a decrease in new jobs since mid-2022, *not directly caused by AI*.
   - One member expressed concern that **AI might be used as a scapegoat** for wrong bets in the tech sector, rather than a true indicator of productivity changes.
- ****AI Text Detector Deception**: Human or Prompt Injection?**: Members dismissed the reliability of **AI text detectors**, with one suggesting that prompt injection could easily bypass them.
   - It was highlighted that *AI text detectors aren't even able to count words*.
- ****Tool Calling Tango**: XML vs MCP**: The conversation dove into the debate between **XML** and **MCP** for tool calling, noting that the token difference doesn't significantly impact performance.
   - There was a shared sentiment that *the only difference is really in how much these models can handle*, suggesting that excessive tools can cause breakdowns.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1478693603621601291)** (13 messages🔥): 

> `Hermes wrangling difficulties, Mythos alternative, Small Hermes 4 Model?, Qwen 3.5 vs Hermes` 


- ****Hermes** is Headache for Corporates**: A member stated that trying to wrangle **Hermes** is a headache, and suggested **Mythos** as an alternative for personal projects.
   - He added that if the AI assistant is for general shipping purposes, **Hermes** is the way to go.
- **Small **Hermes 4** Model in the works?**: A member inquired about plans to release a *small* **Hermes 4** model, similar to the older **Hermes 3 Llama 3.2 3B** models.
   - He noted that small **3B** models are perfect for Orin Nanos.
- ****Qwen 3.5** might be better than **Hermes****: A member suggested that **Qwen 3.5** would probably be better than **Hermes**.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1478864722202988607)** (1 messages): 

> `NT Strategies, Coding NT Strategies, AI Collaboration` 


- **NT Strategies Coder Connects**: An AI enthusiast expressed excitement for **NT (Neural Tangent) strategies** and offered to exchange ideas.
   - The user mentioned years of coding **NT strategies**, seeking collaboration with similar minds.
- **NT Strategy Collaboration Invitation**: A member shared their experience in coding **NT strategies** for years.
   - Extending an invitation, they proposed exchanging ideas and collaborating with other interested individuals.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1478505411974926531)** (12 messages🔥): 

> `Delve's Airport Marketing, TSA Tray Advertising, Server Anniversary Party Planning, Pie in the Sky Document Mix-Up` 


- **Delve Dominates TSA Trays!**: Company **Delve** purchased advertising space on every **TSA tray at San Jose International Airport (SJC)**, as announced in [this tweet](https://x.com/karunkaushik_/status/2028906773084541329).
- **Mixing up *Pie in the Sky* docs**: A member humorously recounted working off the *pie in the sky.md* document, mistaking it for the job's first deliverable.
- **Saeris.gg Prepares for 5th Anniversary!**: **Saeris.gg** announced a poll to determine the timing and type of party for their server's **5th anniversary** this month.
   - The [poll is available on Discord](https://discord.com/channels/822583790773862470/822583965009051668/1477825626114359379) for server members to vote.


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1478564582879395912)** (37 messages🔥): 

> `Tech Over Cost, GenZ Screen Routine, OpenClaw Invoice, Exercise vs. Rideshare, Apple Pricing` 


- **Tech Trumps Thrift, Claims Tech Enthusiast**: A user [@justalexoki](https://x.com/justalexoki/status/2028509501448454322?s=12) expressed enthusiasm for technology, prioritizing innovation over concerns about rising **RAM** market prices.
- **Screen Time Serenade: Gen Z's Digital Day**: The post satirizes the daily lifecycle of **GenZ** as a continuous transition between various screen sizes, moving from smartphones to laptops and televisions ([@0xleegenz](https://x.com/0xleegenz/status/2028734620553068584?s=20)).
- **Cobie's Claw: Corporate Cashflow Caper?**: Cobie details a controversial business model where an AI tool, **OpenClaw**, sent **50,000** daily invoices to Fortune 500 companies, achieving **$10 million ARR** in two months ([@cobie](https://x.com/cobie/status/2028431334486487129?s=12)).
   - The experiment exploited a **2%** non-verification rate, framing it as *capturing corporate leakage*.
- **Uber Alles? Mile Run Maligned**: Will Bredderman humorously criticizes the physical exertion of gym class mile runs by comparing their inefficiency to the speed of an **Uber trip** ([@willbredderman](https://x.com/willbredderman/status/2028861498651537828?s=12)).
- **Apple's Audacity: AirPods cost as much as Macs**: A viral post by user **Noah Cat** points out the irony in Apple's promotional imagery, specifically highlighting a scene where a user wears **AirPods Max** that are priced equally to the **MacBook Neo** they are using ([@Cartidise](https://x.com/Cartidise/status/2029214846433296705?s=20)).


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1478595272903888988)** (5 messages): 

> `AI investment strategies, Bloom Energy, Coreweave, Bitcoin miners, AI energy constraints` 


- **Investor Ejaaz bets big on AI Energy Infrastructure**: A [24-year-old investor](https://x.com/cryptopunk7213/status/2028990731747049785?s=12) is pivoting from **traditional tech stocks** like **NVIDIA** to massive positions in **AI energy infrastructure**, including **Bloom Energy**, **Coreweave**, and repurposed **Bitcoin miners**.
   - The strategy focuses on **AI's energy constraints** while shorting **IT outsourcing firms** expected to be disrupted by **AI coding tools**.
- **AI no longer making everyone rich?**: A member expressed surprise over the shift from the narrative that *“AI will make us all rich”* to the idea that *“firms are AI-vulnerable.”*
   - No further explanation or clarification was given in the messages.


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1478556139753836586)** (4 messages): 

> `AI Adoption in Web Agencies, OpenClaw Exploration, Retired Polymaths in AI` 


- **Italian Dev Embraces AI Consulting**: Guido from Italy, a former developer and product manager at **Idearia**, is now working as an **AI consultant** after helping his company adopt AI workflows.
   - He recently purchased a Mac Mini and is experimenting with **OpenClaw**, expressing excitement about potentially meeting others at **AIEE** in London.
- **The Rise of the Retired Polymath**: A user introduced themselves as a *retired polymath* clarifying they are *retired from work, not polymathy*.
   - Another user expressed relief at the clarification, showing interest in the presence of diverse expertise within the group.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1478519082411360409)** (44 messages🔥): 

> `M3 Battery Life, AppleCare Worth, Nano Texture Display, Borland Turbo Series, MacBook Neo` 


- **M3 Battery Drain Spurs Inquiry**: A user reported only getting **2 hours of battery life** on an **M3 MacBook**, prompting suggestions to check the energy usage tab for rogue **Docker containers** or consider it a defective battery.
   - Others chimed in sharing their experiences with **M1 MacBooks**, noting great battery life and performance, while also speculating that newer models with more cores might be less efficient.
- **AppleCare: To Buy or Not to Buy?**: Users debated the merits of **AppleCare**, with some regretting not purchasing it after expensive repairs, while others prefer to self-insure, finding that battery replacements are relatively affordable at around $80.
   - One user mentioned receiving a significant discount on a maxed-out machine through a departing Apple employee's discount, saving them $1100, and planned to use the machine for local model experiments.
- **Nano Texture Display: Love It or Hate It?**: The **nano texture display** sparked mixed reactions, with some users loving it for reducing glare in bright environments, while others regretted the purchase.
   - Someone mentioned that 2 friends loved it and 2 friends regret it.
- **Borland's Turbo Series: The GOAT?**: Users reminisced about **Borland's Turbo series**, particularly **Turbo Pascal** and **Turbo C**, praising the amazing editors and comprehensive manuals that facilitated learning programming.
   - One user recalled using **Turbo Prolog** and some **Lisp** as their first software purchases for their PC in the mid-80s.
- **$500 MacBook Neo with Edu Discount?**: Someone linked to the [Apple MacBook Neo page](https://www.apple.com/macbook-neo/), speculating that its low price with an education discount would lead to massive sales.
   - One user added: *Seems like a great light workload daily driver*


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1478511268854894754)** (3 messages): 

> `Revenue Fluctuation, Networking Introduction` 


- **Revenue Surge Sparks Debate**: A member reported a spike in revenue, humorously suggesting that *sometimes being lucky is better than being good*, referencing a significant difference between today's revenue and a fairly normal day.
   - The member shared a screenshot, likely depicting the revenue data, to illustrate the unexpected financial upswing.
- **Networking Opportunity Presented**: A member indicated they would be connecting two individuals, mentioning that they would send an email with the necessary context for the introduction.
   - The intention behind this action is to facilitate a professional relationship, with the email serving to provide background information.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1478512385747718378)** (7 messages): 

> `Always Further Hiring Principal SWE, Tenex Labs Referral Program, Scapegoat Consulting LLC Services, AI Engineering World Fair` 


- **Always Further Seeks Principal SWE**: [Always Further](https://www.alwaysfurther.ai/careers/principal-swe) is hiring a **Principal Software Engineer**, accepting senior-level applications only.
- **Tenex Labs Launches Referral Program for AI Talent**: Alex Lieberman, founder of **Tenex Labs**, is initiating a referral program aiming to recruit over **120 AI engineers** and strategists by the end of 2026, offering a **$10,000 bounty** for each successful hire retained for **90 days**.
- **Scapegoat Consulting LLC: We Take the Blame**: A member introduced their new venture, **Scapegoat Consulting LLC**, offering strategic AI consulting, programming with AI workshops, and project work, emphasizing a *systems thinking* approach to solving problems with LLMs.
- **Strategic AI Consulting: Navigating Engineering in an LLM World**: A member's strategic AI consulting services focus on *what is engineering in a world of LLMs*, based on insights from articles like [LLMs: A Paradigm Shift for the Pragmatic Programmer](https://the.scapegoat.dev/llms-a-paradigm-shift-for-the-pragmatic-programmer/) and workshops at the [AI Engineering World Fair](https://www.youtube.com/watch?v=zwItokY087U).


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1478886794463285411)** (5 messages): 

> `Westfield SF Mall redevelopment, Presidio Bay and Prado Group, Office space conversion` 


- **Westfield SF Mall Sold and Set for Revamp**: The **Westfield SF mall** has been sold to **Presidio Bay** and **Prado Group**, who plan to convert sections of the **1.2 million square foot complex** into office spaces while maintaining some retail presence, according to [this tweet](https://xcancel.com/pitdesi/status/2029319437040672976).
- **Office Space Conversion Planned**: The new owners, **Presidio Bay** and **Prado Group**, intend to repurpose portions of the **Westfield SF mall** into office spaces, while still keeping some retail stores open.


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1478692626290180096)** (1 messages): 

> `AI Engineer London Meetup #10, Mario creator of Pi, OpenClaw` 


- **AI Engineer London Meetup #10 Announced**: The **AI Engineer London Meetup #10** has been announced for next week, details on [Luma](https://luma.com/94ma079o).
   - This meetup follows the December event featuring **Peter of OpenClaw**.
- **Mario of Pi to be Featured**: **Mario**, the creator of **Pi**, will be the featured guest this month.
   - Notably, **OpenClaw** is built on **Pi**.


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/1478838208283279472)** (3 messages): 

> `TQBF Tweet, RhysSullivan Tweet` 


- **TQBF Tweet Shared**: A member shared a link to [a tweet from TQBF](https://x.com/tqbf/status/2029252008415248454?s=20).
- **RhysSullivan Tweet Shared**: A member shared a link to [a tweet from RhysSullivan](https://x.com/RhysSullivan/status/2029238739982270593).


  

---


### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1478499794384060488)** (20 messages🔥): 

> `Die Hard references, Trump administration, Iran strikes Turkey, NATO Article 5, Defense company meetings` 


- ****Die Hard** makes a comeback!**: Members shared [a tweet](https://x.com/jayblackisfunny/status/2028708770516193471) comparing the **Trump** administration's energy to **Harry Ellis** from the movie **Die Hard**.
   - The comparison alludes to the moment when **Ellis** realizes the danger posed by **Hans Gruber**.
- **Iran's strike on Turkey debated**: Users discussed whether a potential Iranian strike on **Turkey**, a NATO member, would trigger **Article 5**.
   - It was noted that **Article 5** requires a *"we're under attack"* situation and consensus from NATO members.
- **Mysterious Defense executive meetings**: Some users mentioned [a tweet](https://x.com/RhysSullivan/status/2029238739982270593) about executives of major defense companies being called in for an emergency meeting.


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1478492053322338646)** (92 messages🔥🔥): 

> `Cursor AI, Spectre I, Meta AI Engineering, Anthropic's Rise, SWE-Atlas` 


- ****Cursor** Conquers Complex Math**: **Cursor AI** autonomously solved **Problem Six** of the **First Proof** math challenge, outperforming human-written results after a four-day process, according to [this X post](https://xcancel.com/mntruell/status/2028903020847841336?s=20).
- ****Deveillance** Deploys **Spectre I****: **Aida Baradari** unveiled **Spectre I** from **Deveillance**, a smart device engineered to thwart unwanted audio recording and safeguard privacy against always-on listening devices, per [this announcement](https://xcancel.com/aidaxbaradari/status/2028864606568067491).
- ****Meta** morphs **AI Engineering****: **Meta** is reportedly establishing a new applied AI engineering group with a significantly flat management structure, aiming for ratios of up to **50 employees per manager**, detailed in [this memo](https://xcancel.com/meghanbobrowsky/status/2028930696664711328?s=46).
- ****Anthropic** Annihilates **ChatGPT** Lead?**: **Anthropic's Claude** purportedly seized **70%** of the US business market by **February 2026**, surpassing **ChatGPT** by concentrating on coding capabilities and AI agents, detailed in [this discussion](https://xcancel.com/yuchenj_uw/status/2028974344710606905?s=12).
- ****Scale AI's SWE-Atlas** Assesses Model Performance**: **Scale AI** launched **SWE-Atlas**, a software engineering evaluation tool extending **SWE-Bench Pro**, with its initial benchmark, **Codebase QnA**, revealing current top AI models score around **30%** as shown in [this launch announcement](https://xcancel.com/scale_AI/status/2029244660905095359).


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1478511028215091252)** (25 messages🔥): 

> `AlphaEvolve, Speculative Speculative Decoding (SSD), Nanbeige4.1-3B` 


- **AlphaEvolve Implementation Shared**: A member shared their basic implementation of **AlphaEvolve** on [GitHub](https://github.com/ankitmaloo/alphaevolve), which improves algorithms using counterfactual regret minimization, initially used in poker and other games.
   - The notes for *Discovering multiagent algos* along with the [paper](https://arxiv.org/abs/2602.22647) are also [available](https://gist.github.com/ankitmaloo/3a985fee39985140b630fb1c67435341).
- **Speculative Speculative Decoding (SSD) Doubles Inference Speeds**: **Speculative Speculative Decoding (SSD)**, introduced by Tanishq Kumar, Tri Dao, and Avner May, reportedly achieves speeds up to **2x** faster than current leading inference engines.
   - More information is available at this [X post](https://xcancel.com/tanishqkumar07/status/2029251146196631872).
- **Decoding YouTube's Static Constraints**: A member shared a link to [YouTube's static-constraint-decoding GitHub repo](https://github.com/youtube/static-constraint-decoding), connecting it to applying a 2-stage pass with **gliner2** to **neo4j**.
   - Further context was provided in the form of [three images](https://cdn.discordapp.com/attachments/1107320650961518663/1478723519893344379/IMG_1558.jpg?ex=69aa18d8&is=69a8c758&hm=6da5bae468fe73280b27fcb3abe9de64de5dcb99fce9e60dc863cf46c8577e5e&). 
- **Community Explores Scaling Parameterized Orthonormalization**: Members discussed the paper being covered today, [Orthonormalization that's Scalable by Parameterizing it](https://arxiv.org/abs/2602.16928) and its [chatgpt summary](https://chatgpt.com/c/69a87ea9-8340-8321-8646-27ca38fef1ca).
   - A member called it *really interesting* and felt that it seemed *obvious when you think about it now*.
- **Nanbeige Model Surfaces on HuggingFace**: The community discussed the release of **Nanbeige4.1-3B** as linked on [HuggingFace](https://huggingface.co/Nanbeige/Nanbeige4.1-3B).
   - Further discussion on this model can be found in [this discord thread](https://discord.com/channels/822583790773862470/1471592765094756539/1476800620144103619).


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1478701280087244831)** (12 messages🔥): 

> `LLM context compression, User responses only, RLM techniques, Harness ideas, OpenPencil launch` 


- **LLM Performance Boost with Prompt Removal**: Instead of summarizing past conversations for context compression, consider giving the LLM past conversations containing only the user responses.
   - According to a research paper, this method maintains about **95%** of LLM performance and could be combined with prompt removal and a sliding window approach.
- **Store Model Responses in RLM**: For RLM techniques, explore storing model responses so the model can pick and choose which part of the sliding context it wants.
   - This idea mimics sliding window attention but at the harness level, potentially improving efficiency.
- **Brainstorming Harness Improvements**: Consider improving context compression with **directed techniques**, enabling steering of compression in different directions instead of blunt handoffs, while maintaining context at >200k.
   - Additional ideas include prompt learning at test time, graph-directed reasoning, and self-evolving codebases.
- **Danila Poyarkov Unveils OpenPencil**: Danila Poyarkov developed and launched **OpenPencil**, an **open-source** (MIT licensed) Figma alternative, in just three days as a response to Figma patching his previous tool, figma-use.
   - [OpenPencil](https://xcancel.com/dan_note/status/2028201388074013048) features **.fig** file support, **AI-driven design tools**, and **P2P collaboration** without needing accounts or subscriptions.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1478555007749193838)** (5 messages): 

> `AgentGambit, Live LLM decision-making, Autonomous LLM` 


- **AgentGambit Debuts as Live LLM Arena**: A member shared [AgentGambit](https://agentgambit.io), a live arena for **autonomous LLM decision-making**, where agents play no-limit **Texas Hold'em** in real time.
   - The agent's identity, risk profile, and tilt logic are defined in a single markdown file (**PSYCHE.md**), allowing the model to play autonomously.
- **Gambit as Poker Playing Playground**: AgentGambit started as a benchmark for **decision-making** in imperfect games, but tweaking agents to play poker turned out to be fun.
   - The member welcomes feedback from Latent Space and expressed interest in making a **Claude skill** for command-line installation.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1478523711182082162)** (4 messages): 

> `Physical Intelligence, Multi-Scale Embodied Memory, Video Encoders, Text Summarization` 


- **Physical Intelligence Introduces Multi-Scale Embodied Memory (MEM)**: [Physical Intelligence](https://xcancel.com/physical_int/status/2028954634610720834?s=12) introduced **Multi-Scale Embodied Memory (MEM)**, a system for memory retrieval.
   - The system uses **video encoders** for short-term fine-grained memory and **text summarization** for long-term memory spanning up to **15 minutes**.
- **MEM uses Video and Text Summarization**: **Multi-Scale Embodied Memory (MEM)** uses a combo of video encoders and text summarization.
   - This enables both short-term fine-grained memory and long-term retrieval.


  

---


### **Latent Space ▷ #[san-diego-neurips-2025](https://discord.com/channels/822583790773862470/1335732885717651558/1478497196809650339)** (4 messages): 

> `comma_ai Hackathon, X-Ware.v0` 


- **Comma.ai Announces Hackathon**: Comma.ai is hosting a hackathon from **March 27-29, 2026**, at their headquarters, as announced in a post ([https://x.com/comma_ai/status/2028920208262615417](https://x.com/comma_ai/status/2028920208262615417)).
   - The event is limited to **30 participants** and features a **$10,000 prize pool**.
- **X-Ware.v0 Release**: An announcement references a new product called **X-Ware.v0**.
   - Further details on **X-Ware.v0's** functionality and purpose were not provided in the context.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1478899957355516068)** (8 messages🔥): 

> `Startup failures, Rebranding of startups, AI Influencers, Social media trends` 


- **Satirical Startup Shutdowns**: Finn Hulse satirizes how some **founders fail by inflating metrics, burning through VC funding**, and then erasing their history through name changes and rebranding similar companies, according to [this X post](https://xcancel.com/finn_hulse/status/2029300798174445789?s=46).
- **Computer Generated Personalities gain traction**: Justine Moore shares her shock at the large number of men following social media **AI influencers**, according to [this X post](https://xcancel.com/venturetwins/status/2029289750226702813?s=20).


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1478524022122746017)** (5 messages): 

> `Cursor AI, Problem Six, Mathematical Research` 


- ****Cursor Cracks Complex Calc Problem****: Michael Truell reports that **Cursor** autonomously discovered a novel solution to '**Problem Six**' of the [First Proof challenge](https://arcinstitute.org/news/evo-2-one-year-later).
   - The **AI's solution** outperformed official academic benchmarks after running for **four days** without human intervention, suggesting that specialized agent coordination techniques can generalize beyond software engineering into advanced mathematical research.
- ****AI Math Breakthrough Sparks Debate****: The autonomous discovery by **Cursor AI** of a novel solution to '**Problem Six**' has sparked debate within the AI and mathematical research communities.
   - Some researchers are skeptical, questioning the generalizability of agent coordination techniques beyond software engineering, while others hail it as a significant step toward AI-driven mathematical innovation.


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1478492302044434576)** (5 messages): 

> `Activation Oracles, Model Safety, X-Ware.v0` 


- **Activation Oracles assessed for Model Safety**: Arya Jakkli's [X-Ware.v0](https://xcancel.com/ajakkli/status/2028916909136376033) discusses **activation oracles** (finetuning models to explain another model's activations) and its application to model safety.
   - They concluded that the technique was difficult to evaluate and provided limited utility for **safety-relevant tasks**.
- **X-Ware.v0 paper link**: Here is the link to the [X-Ware.v0 paper](https://xcancel.com/ajakkli/status/2028916909136376033).
   - It is titled Evaluation of Activation Oracles in Model Safety.


  

---


### **Latent Space ▷ #[dev-writers-retreat-2025-dwr](https://discord.com/channels/822583790773862470/1445650211694448714/1478621315756724264)** (1 messages): 

> `Book Launch Party, Networking Opportunities` 


- **Dev Writer's Retreat Fam Invited to Book Launch**: Members of the **Dev Writer's Retreat** were invited to a book launch party on **March 13**.
   - The invitation link shared was [https://luma.com/kb59vt7m](https://luma.com/kb59vt7m).
- **Networking and Collaboration**: The book launch party provides **networking opportunities** for the Dev Writer's Retreat members.
   - It's a chance to connect with other writers and industry professionals in a social setting.


  

---


### **Latent Space ▷ #[euno-log](https://discord.com/channels/822583790773862470/1473750131441668096/1478876100829384734)** (1 messages): 

> `AI Hackathon` 


- **AI Hackathon Incoming**: A member alerted the group to a new AI hackathon for building agents.
   - The member encouraged others to join in the AI building fun.
- **Hackathon Details to Follow**: Further details regarding the hackathon, such as specific dates, rules and prizes, were promised to be shared soon.
   - Participants expressed excitement at the prospect of building new AI agents, and looked forward to receiving more information.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1478482866617847890)** (250 messages🔥🔥): 

> `Perplexity Model on OpenRouter, OpenCLaw Usage, API Key Issues and Error 401, Mercury 2 Model Release, Provider Fallbacks` 


- **OpenCLaw Routes Traffic Erratically to Sonar**: A user reported that **OpenCLaw** mistakenly routed traffic to **Sonar**, despite intending to use **Qwen3** embeddings, expressing confusion over the routing behavior.
   - Another user called **OpenCLaw** a *security nightmare*.
- **Siliconflow FP8 Fallbacks to OpenAI Cause Errors**: A user reported that setting `provider.only: ["siliconflow/fp8"]` with `allow_fallbacks: false` was ignored for `glm-4.5-air`, resulting in traffic being routed to **OpenAI** and causing empty/malformed responses.
   - Up to **34%** of their traffic was impacted, affecting prod users for several hours.
- **OpenRouter Limits Paid Usage as Expected**: A user inquired whether setting a guardrail to limit monthly spend with auto top-up off would disable paid usage, and another user confirmed that paid requests would be disabled until the balance is reloaded.
   - Another user confirmed that this limit also goes for the website too.
- **Deepseek 3.2 Model Creates repetitive Thinking Blocks**: A user reported issues with **Minimax 2.5** and **Deepseek 3.2** models on OpenRouter, where they generate repetitive reasoning/thinking blocks, even when the models work fine on other platforms.
   - The user suspects the providers are running heavily quantised models, despite their quantization settings being set to **fp8** or above as per OpenRouter's documentation.
- **Taxman Cometh - OpenRouter Bills Now Include Sales Tax**: A user noticed the billing update email, remarking that **OpenRouter** was previously not collecting sales tax at all.
   - Some users also wished for more capybara emojis in the [OpenRouter docs](https://openrouter.ai/docs).


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1478591695930265702)** (24 messages🔥): 

> `Qwen Underperformance, Tiny Face's Task, XAI Content Filter Pricing, China's OSS LLM Success, Trillion Parameter Model` 


- **Qwen's Board Evaluations Tanking**: Members discussed **Qwen**'s underperformance in board evaluations, with some evals being quite bad while others showed improvement.
- **XAI Charges Nickel for Content Filter**: **XAI** is charging **5 cents** for content filter requests.
   - A member questioned why Tiny Face made them defend **Qwen**.
- **China drops Trillion Parameter Model**: A [tweet](https://x.com/YuanAI_Lab/status/2029204213180580229) was shared about another **1 trillion parameter model** coming from a Chinese lab.
- **Codex 5.2 preferred to Codex 5.3?**: Despite the launch of 5.3 Codex, many people seem to still prefer **5.2**, with the scores being identical in Codex CLI according to an [image](https://cdn.discordapp.com/attachments/1392278974222307469/1478849247565713640/image.png?ex=69a9e530&is=69a893b0&hm=fec1cde32448870e0a7a3a7c455abb6b6871c6d5c282d3fc287898cedbab21cc).
- **Google Gemini AI Faces Wrongful Death Lawsuit**: **Google Gemini AI** is facing a [wrongful death lawsuit](https://www.wsj.com/tech/ai/gemini-ai-wrongful-death-lawsuit-cc46c5f7?st=THRLAh&reflink=desktopwebshare_permalink) after allegedly providing *real addresses* to someone, adding to their belief that the AI was real.
   - The individual had over **8000 pages** of chats with the AI and apparently didn't realize it could hallucinate; the lawsuit suggests the absence of a building at the provided address could have *tipped him off to the fact that this was an AI fantasy*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1478484239971254343)** (84 messages🔥🔥): 

> `Job opportunities for AI devs, FOSS AI models similar to Sonnet 4.6, Fine-tuning Qwen3.5 with TRL, Hugging Face Spaces issues, Product tryon workflow` 


- **AI Devs Seek LLM/SaaS Roles**: A senior full stack AI developer sought opportunities in **LLM/SaaS** projects, with experience in chatbots, AI agents, automation workflows, and custom AI tools.
   - They specified skills in **OpenAI, LangChain, Python, and JS**, offering to build mobile/desktop apps, computer vision, and AR/VR solutions.
- **Users Discuss Best FOSS AI Alternatives to Sonnet 4.6**: A user inquired about the best **FOSS AI** model similar to **Sonnet 4.6**, seeking advice on hardware requirements.
   - No specific models were recommended but the discussion focused on open-source alternatives.
- **Qwen3.5 Fine-Tuning on H200 Faces Slowdown**: A user reported slow training speeds while fine-tuning **Qwen3.5 27B** on a single **H200**.
   - Another user suggested trying **Unsloth** with **TRL**, linking to a relevant [Twitter post](https://x.com/twitter/status/2028845314506150079).
- **HF Spaces Container Logs Vanish**: A user reported an issue with **missing container logs** in Hugging Face Spaces, even while the space was running.
   - Potential causes included **HF's prohibited actions** or the space getting stuck before log initialization.
- **Community Ponders Product Try-On Workflow Difficulties**: A user asked for insights into product try-on workflows, expressing difficulty in replicating them effectively.
   - Specifically, they cited difficulties replicating a **product try-on workflow** similar to the one found on [shopatorie.com](https://shopatorie.com/).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1478490850714583252)** (22 messages🔥): 

> `lightweight scraping client, PyTorch-like framework from scratch using NumPy, single‑file CLI agent, MoC — Mixture-of-Collaboration with Adaptive Compute, database built on rust` 


- **Scraping client pays with USDC!**: A member built a lightweight scraping client, [Minifetch](https://www.npmjs.com/package/minifetch-api), that can be paid-per-fetch in **USDC** over x402/Base or Solana, so agents can call it autonomously with no accounts or API keys needed.
- **NebTorch: PyTorch framework made in NumPy**: A member built a **PyTorch-like framework from scratch using NumPy**, like karpathy's micrograd, called [NebTorch](https://github.com/nebHailemariam/NebTorch).
- **Mochaclaw: Single-File Local CLI Agent**: **Mochaclaw** is a single‑file CLI agent that runs entirely on your local machine using **Ollama** (default) or **Transformers.js** (WASM) to execute AI workflows without any cloud dependencies: [https://huggingface.co/webxos/Mochaclaw-js](https://huggingface.co/webxos/Mochaclaw-js).
- **Lunaris MoC: Collab-Compute Optimizer**: **Lunaris MoC (Mixture-of-Collaboration)** routes tokens to experts that collaborate through a learned mediator before fusion, achieving a **59.97** val perplexity vs **62.89** for standard MoE: [https://github.com/Auren-Research/lunaris](https://github.com/Auren-Research/lunaris).
- **Anamnesis 5.0: Rusty Recall Database**: A member has developed a new database in **Rust** for more organic recall, aiming to replicate human memory function, available at [https://github.com/AImakerextraordinaire/Anamnesis_5.0](https://github.com/AImakerextraordinaire/Anamnesis_5.0).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1478643111012864161)** (7 messages): 

> `Model Selection for Agent Course, Agent Course Enrollment Query` 


- **Llama 3.2 vs Qwen2**: A member inquired whether a lighter model like **Llama 3.2:3b** could be used instead of **Qwen2:7b** due to limited RAM capacity.
   - They were following on-boarding instructions for the agent course and were seeking clarification on model selection.
- **Agent Course Enrollment**: A member asked how to confirm their enrollment in the agent course.
   - They wanted to ensure they were properly registered for the program.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1478637534006214817)** (26 messages🔥): 

> `AMD GPU direct NVMe access, ROCm hipFile Bullshit, SemiAnalysis InferenceX benchmark, GB300 NVL72 vs H100 Live Stream, Tenstorrent Lecture` 


- **AMD GPU Gets **NVMe** P2P**: After some **amdgpu driver** patches to the linux kernel a user enabled p2p between an **NVMe** device and an **AMD GPU**.
   - He built on top of Jason Gunthorpe [RFC series](https://lore.kernel.org/dri-devel/0-v1-b5cab63049c0+191af-dmabuf_map_type_jgg@nvidia.com/) around dma-buf and iommufd, and added a Physical Address List (PAL) exporter to the amdgpu driver so the buffer can be mapped into an iommufd IOAS.
- ****ROCm hipFile** P2P Claims Face Skepticism**: A user shared a link to [ROCm/hipFile](https://github.com/ROCm/hipFile) asking if it was legit P2P between devices.
   - The original poster replied that *this still involves the CPU issuing commands with VRAM as the location to write the data*, unlike his implementation of direct GPU-SSD communication.
- **SemiAnalysis **InferenceX** Benchmark Deconstructed on Livestream**: A user posted a link to a [GPU Mode stream](https://discord.com/channels/1189498204333543425/1189640399476764692/1478445293614923856) covering Dylan Patel's analysis of the **GB300 NVL72** vs **H100** using the **InferenceX** benchmark.
   - The description jokes that *InferenceX shows that Nvidia's slicing of their products is sharper than AMD's chips.*
- **Desire for **Tenstorrent** Lecture Expressed**: A user asked if it would be possible to get a lecture from folks over at **Tenstorrent**.
   - One user said they tried to connect with Jim Keller in the past without success but another user responded that they *will be working there for an internship* so they can try and connect internally.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1478500472628449293)** (17 messages🔥): 

> `Texture Memory vs. Direct Load/Store, Ping-Pong Buffers for Kernel Iteration, Inter-CTA Communication, MXFP8 MMA Support` 


- **Texture Memory Loses Perf Battle**: A member references the [NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#texture-and-surface-memory) noting that **texture memory** no longer provides a performance benefit over **direct load and store instructions** on currently supported GPUs.
   - Older CUDA code might still use texture memory due to historical performance benefits on older GPUs.
- **Ping-Pong Buffers juggles arrays**: A member suggested using **ping-pong buffers** (swapping read and write pointers) to alternate between two arrays `a` and `b` in a loop: `std::swap(read_buf, write_buf);`
   - This allows for alternating read/write access to the arrays without copying data which is good since *there are other kernels in between*.
- **Quest for Global Memory Insights**: A member inquired about resources detailing the performance and correctness implications of **inter-CTA communication** via **global memory**.
   - They were specifically interested in practical correctness on given architectures/compiler versions, plus the implications of `MEMBAR`, `ERRBAR`, `LDG/STG.STRONG`, `CCTL.IVALL` at the SASS level.
- **MXFP8 MMA only supports MMA_K=64 for sparse?**: A member referenced the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape) asking if **MXFP8 MMA** supports `MMA_K=64`.
   - Another member clarified that `MMA_K=64` is likely only supported for **sparse matrices**, differing from the standard `MMA_K=256` for dense GEMM which is how *they felt like they were taking crazy pills*.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1478496818227708104)** (3 messages): 

> `CUDA Agent, Kernel Optimization, ByteDance model` 


- **ByteDance rolls out CUDA Agent!**: ByteDance has released a **CUDA Agent**, a model trained to write fast and optimized CUDA kernels, outlined in their [whitepaper](https://arxiv.org/pdf/2603.02298).
   - The agent outperforms **torch.compile** by **2x** on simple/medium kernels and beats **Claude Opus 4.5** and **Gemini 3 Pro** by around **40%** on the most challenging tasks.
- **Kernel Compilation Competition Heats Up**: The **CUDA Agent** achieves approximately **92%** better performance on complex kernels compared to **torch.compile**.
   - A member announced a meetup for **vLLM** to discuss **torch.compile** integrations ([Luma link](https://luma.com/rk0a1lue?tk=qAta1VCuTe)).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1478619761049866303)** (18 messages🔥): 

> `AMD vs Nvidia Programming, RTX 5090 Project Ideas, Paged Attention with Triton, GPU Security` 


- **AMD and Nvidia Programming Similarity Examined**: While both **AMD** and **Nvidia** use parallel processors with similar concepts like **HBM** and **L2/L1 caches**, coding for them differs significantly, with **Nvidia** offering better tooling, blog content, and papers, yet the underlying programming model isn't fundamentally distinct.
   - One member noted basic kernels appear similar but yield basic performance, while another suggested treating them as entirely different devices, referencing [Stanford's Hazy Research blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk) and [YouTube video](https://www.youtube.com/watch?v=jsYyF03Fs3o) highlighting AMD's brittle software ecosystem and the need for hand-optimized assembly kernels.
- **New RTX 5090s Spark Project Ideas**: A member with a cluster of **4x RTX 5090s** sought interesting project ideas with technical walkthroughs, prompting suggestions to "go wild" with kernel development or other ambitious projects.
- **Triton Used for Paged Attention Implementations**: When implementing a custom serving engine a member inquired about using **Triton** for paged attention store and load kernels (for the kv cache).
   - They noticed that other serving engines code a paged attention store and load kernel using **Triton**.
- **GPU Security Discussions Initiated**: A member working on low-level GPU security sought a dedicated security channel, leading to a recommendation for the <#1189498205101109300> channel and a mention of the [pygpubench project on GitHub](https://github.com/ngc92/pygpubench) as a security-oriented resource.
   - A member also criticized NVIDIA for lacking a proper security model for newer architectures.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1478852911873261721)** (3 messages): 

> `ND views, N-D visualizer` 


- **ND Views Supported, Visualizer Delayed**: Support for **ND views** is available, but the version of the puzzles with the new **N-D visualizer** has not been pushed yet.
- **N-D Visualizer Puzzles Explained**: The puzzles are specifically designed to teach how to use the **N-D visualizer**, and the **triton kernels** are already filled out.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/)** (1 messages): 

inoday: sorry for mislabeling!
  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1478517598038523974)** (3 messages): 

> `TRLC DK-1, CamBot, Stereolabs ZED Mini, PI memory research` 


- **Robot Giraffe Neck Inspire New Teleop System**: Inspired by robots with giraffe-like camera necks, a member built an experimental [TRLC DK-1](https://www.robot-learning.co/) teleop system for human interventions in OOD policy runs.
   - The initial test involved an [ELP stereo cam module](https://www.amazon.de/dp/B07FT2GKZS) mounted on a SO-101, demonstrated in [this video](https://x.com/neurosp1ke/status/2023073945637753101?s=20).
- **CamBot Project Goes Open Source**: Inspired by Jannik's leader arm, a member designed a **6 DoF arm** called **CamBot** and published it open-source (Apache 2) on [GitHub](https://github.com/open-thought/cambot).
   - The project enables remote viewing via **VR head tracking** and uses [StereoLab's ZED Mini](https://www.stereolabs.com/en-de/store/products/zed-mini) for higher quality stereo vision, costing around **110 EUR** in materials.
- **PI Announces Memory Research**: A member shared a link to cool news from PI about their [memory research](https://www.pi.website/research/memory).


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1478584614737018992)** (3 messages): 

> `Track C Scoring Clarification, Email Confirmation, GPU Resource Details` 


- **Track C Scoring Confusion Surfaces**: A participant inquired about the scoring mechanism for **Track C**, specifically how the *decode kernel* and *prefill kernel* are weighted for the competition score.
   - The user is unsure whether the evaluation is based on the average clock-time or the average leaderboard position.
- **Participants Seek Email Confirmation**: A participant asked for confirmation of the email address used, noting that they had sent **three emails** previously without receiving a response.
   - Another participant mentioned receiving an email a few days prior.
- **GPU Resource Details Missing**: A participant noted the absence of information regarding **GPU resources** in the email they received.
   - The email was received a few days prior, but contained no mention of **GPU resources**.


  

---


### **GPU MODE ▷ #[from-scratch](https://discord.com/channels/1189498204333543425/1466534042768904356/)** (1 messages): 

m0ji_l: Forwarding given that this appears to be a channel centered around vllm minimals
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1478487196515438633)** (49 messages🔥): 

> `Kimi CLI web ui, 4chan /g/ board briefing, Gemini vs Kimi for large documents, Kimi tech briefing prompt updates, GrapheneOS on Motorola` 


- ****Kimi CLI** gets thumbs up for web UI!**: A member expressed that the **Kimi CLI** web UI has been great, without elaborating on the particular functionality.
   - No links or blogposts were given, just a general expression of delight with the UI.
- **Moonshot AI team handles the Kimi issues**: A member mentioned that Kimi Team members (with yellow role) work at Moonshot AI, and reported an issue that has been reflected to the relevant department.
   - No further elaboration about the issue was provided.
- ****Kimi briefs** spicy 4chan /g/ content!**: A member described a workflow using **Gemini 3.1 Flash Lite** to extract URLs from 4chan's **/g/** board, then using **Kimi** to generate a briefing of the threads, sharing [a Kimi-generated briefing](https://www.kimi.com/share/19cb6b07-4ab2-8d9a-8000-0000a34349d5).
   - The generated briefing contained quotes such as */sdg/ (Stable Diffusion): Still generating anime girls and arguing about Z-Image vs. Flux.2, with Anima getting attention for style consistency* and *Systemd Schizo Posting: Eternal debate about whether systemd violates Unix philosophy*.
- ****Python-Powered Kimi Prompt** Automates Analyst Work!**: A member shared an updated tech briefing prompt using Python to validate completeness and accuracy, estimating that **Kimi** performs in minutes what would take a solo analyst **12-20 hours** or a 2-person team **6-10 hours**, sharing the [updated prompt](https://cdn.discordapp.com/attachments/1371757564005711973/1478584075190009948/agis.txt?ex=69a996fa&is=69a8457a&hm=f675eca24a9134cbfcb9baf1b3dfe406694a15ead4d0e803623e19bd207320b7&).
   - A subsequent iteration was shared in a [second attached file](https://cdn.discordapp.com/attachments/1371757564005711973/1478609761778794506/agis.txt?ex=69a9aee6&is=69a85d66&hm=6c12571bf2f8d2422eae1c542ea5f0e220efb70ef1fa151e1c5e4d8ca20cc0cb&), with the observation that *reconstructing YouTube-like tech news without YouTube is actually very difficult*.
- **Users report **Kimi Quota troubles****: Some users are asking questions on how their **Kimi allegro plan quota** compare to other plans like *moderato*, while others are asking for an **API endpoint** that can give the quota and usage amounts.
   - Several users pointed to the paid page where the quota is specified for kimi code and agent mode but for general chat use probably close to unlimited.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1478594558530162912)** (11 messages🔥): 

> `Second Look Research Fellowship 2026, Mechanical Interpretability, Wildchat Alternatives, AE Studio's Research` 


- ****Second Look Research Fellowship** Seeking 2026 Fellows**: [Second Look Research](https://secondlookresearch.com/) is accepting summer fellowship applications for 2026 to *replicate and verify load-bearing empirical results in AI safety research*, offering fellows a **$10,000 stipend** plus housing and meals at the University of Chicago from **June 15-August 22**.
   - Ideal candidates should have experience in research engineering, a demonstrated interest in AI safety, and proficiency in AI coding tools, with applications due by **March 7th** at [secondlookresearch.com/fellowship](https://secondlookresearch.com/fellowship).
- **Seek Validation on **Mech Interp** Research**: An undergrad researcher is seeking validation for their work on mechanical interpretability, specifically focusing on *how model compression affects Mech Interp metrics*.
- **Latest **Wildchat** Alternatives**: A member inquired about *latest alternatives for Wildchat* that contain conversations with models from latest **Claude**, **GPT models** (**5.2**, **opus/sonnet 4 series**).
- ****AE Studio** Publishes Research on Activation Steering**: AE Studio submitted new research to ICML titled [Endogenous Resistance to Activation Steering in Language Models](https://arxiv.org/html/2602.06941v1).
   - They also shared an [X thread](https://x.com/juddrosenblatt/status/2028584677351837800) and a [WSJ opinion piece](https://www.wsj.com/opinion/the-pointless-war-between-the-pentagon-and-anthropic-9284fd37?st=zgB8RN&reflink=desktopwebshare_permalink) related to their work.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1478515309601358086)** (9 messages🔥): 

> `Spectral muP, modula, feature learning, NERFIFY` 


- **Spectral muP satisfies MODULA?**: A member thinks that the [MODULA paper](https://arxiv.org/abs/2405.14813) might already satisfy the **spectral muP** condition right out of the box.
   - The spectral muP work is already connected to the MODULA work, through *muonoh*, with [MODULA's Github repo available here](https://github.com/modula-systems/modula).
- **Spectral Norm scaling for feature learning**: A 2023 paper titled [Feature Learning via Spectral Regularity](https://arxiv.org/abs/2310.17813) shows that **feature learning** is achieved by scaling the spectral norm of weight matrices and their updates like √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗).
   - This is in contrast to widely used but heuristic scalings based on **Frobenius norm** and entry size; this spectral scaling analysis also leads to an elementary derivation of maximal update parametrization (**muP**).
- **NERFIFY site provided**: A member shared a link to [NERFIFY](https://seemandhar.github.io/NERFIFY/).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1478636420062122125)** (6 messages): 

> `Anthromorphising Claude, Figure 8 model, Gemini Code` 


- **Anthropomorphising Claude**: A member noted that it's really interesting that someone is **anthropomorphising Claude**.
   - This refers to an earlier message discussing how humans attribute human traits and emotions to AI models like **Claude**.
- **Model tracks Figure 8 *sans* Loss Function**: A member reported creating a model that can **track a figure 8** without a loss function, succeeding only *10%* of the time, aiming to minimize noise within the system by following the figure 8's direction with only *30k params*.
   - The model operates **backpropless**, getting only the input of what direction the figure 8 is at the moment.
- **Gemini Code Creates Figure 8 Model**: A member created a *1-file version* of their Figure 8 model using **ugly Gemini code**, planning to clean it up later once they find a way to get rid of the sparsity.
   - This was inspired by another example of [domain expert successfully steering LLM for new scientific discoveries](https://x.com/bowang87/status/2028935492977475623).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1478811429980078214)** (2 messages): 

> `Anthropic's alignment research, 2026 Predictions, PSM` 


- **Anthropic Aligns with 2026 Projections**: Anthropic is focusing on alignment research as detailed in their [2026 predictions](https://alignment.anthropic.com/2026/psm) document.
   - The announcement was initially shared via a Google Share link ([https://share.google/bgh75ajJKUZXP6kp4](https://share.google/bgh75ajJKUZXP6kp4)).
- **More on Anthropic's Alignment Initiatives**: Further details on Anthropic's approach to alignment can be found in their published [research](https://alignment.anthropic.com/).
   - This includes methodologies and strategies for ensuring AI systems remain aligned with human values.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1478506518709276833)** (5 messages): 

> `Cortical Labs BioLLM, SWE-atlas QNA Leaderboard` 


- **Cortical Labs Cultivates BioLLM**: A member shared a [Reddit post](https://www.reddit.com/r/accelerate/comments/1rjswr9/cortical_labs_grew_200000_human_neurons_in_a_lab/) and a [YouTube video](https://youtu.be/tg7w0RzYrKY) about **Cortical Labs** growing **200,000 human neurons** in a lab.
   - The project is named **BioLLM** and aims to create biological large language models.
- **Scale AI Launches SWE-atlas QNA Leaderboard**: A member shared a link to the [SWE-atlas QNA Leaderboard](https://scale.com/leaderboard/sweatlas-qna) by **Scale AI**.
   - This leaderboard ranks models based on their performance on a question-answering task related to software engineering.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1478752769476268187)** (10 messages🔥): 

> `Qwen3.5 bounty, GatedDeltaNet, GatedAttention, Stable Diffusion fake weights, NULL_ALLOW_COPYOUT` 


- **Qwen3.5 Bounty Requires New Implementations**: The **Qwen3.5 bounty** requires implementing both **GatedDeltaNet** and **GatedAttention**, estimated to be around **~200 lines of code** based on reference implementations like [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_net.py) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp).
   - One developer reported their rough, untested implementation is currently around **80 lines**, with plans to integrate it and add model logic and GGUF parsing.
- **Stable Diffusion Benchmarking with Fake Weights**: The goal is to run `time NULL=1 python3 examples/stable_diffusion.py --fakeweights` in under **10 seconds**.
   - One user reported it taking **17 seconds** on their Mac before crashing, noting that it crashes without `NULL_ALLOW_COPYOUT=1`.
- **`NULL_ALLOW_COPYOUT=1` is it necessary?**: It was questioned whether fixing the need for `NULL_ALLOW_COPYOUT=1` to prevent crashes is part of the bounty, or a pre-existing issue.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1478579261823647764)** (10 messages🔥): 

> `Credit policy, Manus Pro credits missing, Credit packs for all tiers, Website publishing issues, Gold Coast event cancellation` 


- **Manus Credit Policy Clarified**: Monthly credits refresh automatically on the same date each month based on the subscription date, as detailed in the [help article](https://help.manus.im/en/articles/11711097-what-are-the-rules-for-credits-consumption-and-how-can-i-obtain-them).
- **User Reports Missing Manus Pro Credits, Feels "Scammed"**: A user reported paying for **Manus Pro** but not receiving credits, expressing feeling *"scammed!!"* and sought assistance.
- **Call for Credit Packs Across Tiers**: A user expressed the desire for all tiers over **$100** to have the opportunity to purchase credit packs without upgrading.
- **Website Publishing Problems Reported**: A user reported they *"cant publish [their] webside rn"*, speculating about potential issues on the platform's end.
- **Inquiry Regarding Gold Coast Event Cancellation**: A user inquired about the reason for the cancellation of an event at the **Gold Coast**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1478548103379550391)** (8 messages🔥): 

> `aidermacs Emacs, ibuffer-projectile, Open Router, AWS g7e spot instance, Qwen 397B and MiniMax` 


- ****Aidermacs** Integration with **ibuffer-projectile** in Emacs**: A user inquired about configuring **aidermacs**, the Emacs integration for aider, to sort chat buffers with associated project buffers in `ibuffer-projectile`.
   - No solution was provided in the given context.
- ****Open Router** Usage**: A member discussed the token rates on **Open Router**, mentioning *101 tokens inbound per outbound token at 32 tokens per second*.
   - It was estimated that at heavy rates, this would equate to **115K outbound** and **11.6M inbound** tokens.
- **Cost-Effective Model Hosting on **AWS****: A member suggested running models on an **AWS g7e spot instance** as a cost-effective alternative for heavy token usage, estimating the cost at **$2 per hour**.
   - They noted that this setup would provide access to a powerful **VRAM** setup, though on-demand or reserved instances would be more expensive.
- **Top Open Source Models Discussed**: A member identified **Qwen 397B** and **MiniMax** as among the best open source models currently available.
   - No further details or comparisons were given in this short discussion.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1478512009535553566)** (5 messages): 

> ``@` Syntax vs `comptime`, `maybe comptime` in Mojo, Vectorize Performance` 


- **Debate on `@` Syntax over `comptime` Erupts**: Members discussed the potential use of `@` instead of `comptime` for compile-time operations, referencing a [proposal document](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1045r1.html).
   - One member suggested that `@if` would be a cleaner syntax compared to `@parameter if`, anticipating the proliferation of `comptime` keywords as more work shifts to compile time.
- **`maybe comptime` Feature Recall**: A member noted that they had previously requested the `maybe comptime` feature for **Mojo**.
   - No other context was given.
- **"What you see is what you get" loop outperforming vectorize**: A member replaced all their *fn + vectorize* instances with a simple *while loop* with a `k += nelts` at the end of every iteration, on **CPU only**.
   - They reported *no performance loss whatsoever*, in their case, and noted that *vectorize* does more or less the same thing.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1478521212396109847)** (4 messages): 

> `AI Control Hackathon, OpenClaw Builders Roundtable, Antler Forge Execution Sprint, DataMFM Workshop @ CVPR` 


- **Control Your AI Agents at Apart Research Hackathon!**: Apart Research and Redwood Research are hosting an AI Control Hackathon from **March 20-22, 2026**, focused on monitoring and containing AI agents, with both virtual and limited in-person (SF) options, and are giving away [$2,000 in prizes](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach).
- **OpenClaw Business Builders Roundtable fires up**: A 45-minute roundtable on **March 14, 2026**, will dive into practical uses of **OpenClaw** and other tools for running businesses and communities, hosted by AI Scholars, to swap lessons on integration patterns, edge cases, and automations [RSVP here](https://luma.com/qfrucnl2).
   - *Beginner-friendly, but especially valuable if you’re already building something and want to go beyond theory.*
- **Antler Forge: Sprint to Customer Adoption in Seoul**: Antler Forge is hosting a **4-week execution sprint starting April 6, 2026**, in Seoul for founders developing system-heavy technologies, offering **$400K+** investment, **$500K+** in government grants, and **$650K+** AI/cloud credits with direct access to Samsung, Hyundai, SK, and LG ([apply here](https://content.antler.co/forge)).
- **DataMFM Workshop Charts a Course for Multimodal AI at CVPR 2026!**: The DataMFM Workshop at CVPR 2026 focuses on building smart, principled ecosystems for multimodal AI, addressing key challenges like agentic pipelines, governance, and cross-modal alignment, with archival submissions due **March 10, 2026** ([details here](https://datamfm.github.io/)).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1478882670820724797)** (1 messages): 

> `DSPy Power-User Resources, Comprehensive DSPy Corpus` 


- **Seeking DSPy Power-User Knowledge**: A member inquired about a **comprehensive corpus or reference materials/links** on how to be a **DSPy power-user**, beyond the standard documentation.
- **DSPy Power-User Resources Needed**: A user is seeking advanced resources to become a **DSPy power-user**, supplementing the standard documentation.


---
id: MjAyNi0w
title: not much happened today
date: '2026-02-02T05:44:39.731046Z'
description: >-
  **OpenAI** launched the **Codex app** on macOS as a dedicated agent-native
  command center for coding, featuring **multiple agents in parallel**,
  **built-in worktrees** for conflict isolation, **skills** for reusable
  bundles, and **scheduled automations**. The app emphasizes developer workflows
  like **Plan mode** for upfront task decomposition and is gaining positive
  adoption signals from insiders including **@sama**. There is movement towards
  ecosystem standardization of skills folders, signaling early conventions in
  agent tooling. Codex also exemplifies a "self-improving" product feedback loop
  combining humans and agents. In coding agents practice, best practices include
  a "test-first" approach to bug fixes, the "conductor" model where one
  developer manages 5-10 agents in parallel, and a neurosymbolic framing
  explaining why coding agents succeed due to software's verifiability and
  symbolic tooling. Benchmark skepticism remains about productivity studies that
  do not reflect agentic workflows.
companies:
  - openai
models:
  - codex
topics:
  - agent-based-systems
  - parallel-processing
  - software-testing
  - developer-workflows
  - automation
  - product-feedback-loop
  - neurosymbolic-ai
  - benchmarking
people:
  - sama
  - reach_vb
  - gdb
  - skirano
  - embirico
  - ajambrosino
  - thsottiaux
  - nbaschez
  - yuchenj_uw
  - badlogicgames
  - random_walker
---


**a quiet day**

> AI News for 1/30/2026-2/2/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**254** channels, and **14979** messages) for you. Estimated reading time saved (at 200wpm): **1408 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!




---

# AI Twitter Recap

**OpenAI’s Codex app: an agent-native “command center” for coding**

- **Codex app ships on macOS (Windows “soon”)**: OpenAI launched a dedicated Codex desktop app positioned as a focused UI for running **multiple agents in parallel**, keeping changes isolated via **built-in worktrees**, and extending behavior with **skills** and **scheduled automations** ([OpenAI announcement](https://twitter.com/OpenAI/status/2018385565289267236), [rate-limit + availability details](https://twitter.com/OpenAI/status/2018385568992752059), [OpenAIDevs feature rundown](https://twitter.com/OpenAIDevs/status/2018385865207419124)). A recurring theme: *the interface* (not just the model) is becoming the product.
- **Developer workflow details that matter**: The app emphasizes (a) *worktree per task/PR* as the primitive for parallelism and conflict isolation; (b) *Plan mode* (`/plan`) to force upfront decomposition and questions; (c) skills as reusable bundles that can connect to external services (Figma/Linear/Vercel, etc.); and (d) automations for recurring background jobs ([@reach_vb](https://twitter.com/reach_vb/status/2018385536616956209), [Plan mode](https://twitter.com/reach_vb/status/2018456051792982339), [skills landing page](https://twitter.com/reach_vb/status/2018390580330389728)).
- **Usage signals / adoption narrative**: Multiple insiders (and power users) claim the app is a step-change over CLI/IDE extensions for large repos and long-running tasks—particularly for managing parallel threads and reviewable diffs. Notable testimonials include [@gdb](https://twitter.com/gdb/status/2018387844222578818) (agent-native interface; “going back to terminal feels like going back in time), [@sama](https://twitter.com/sama/status/2018414858015039504) (surprised how much he loves it), and [@skirano](https://twitter.com/skirano/status/2018398337938960715) (replacing Cursor + Claude Code in their workflow).
- **Ecosystem pressure / standardization**: There’s already a push to standardize “skills” folders: proposal to have Codex read from `.agents/skills` and deprecate `.codex/skills` ([@embirico](https://twitter.com/embirico/status/2018415923930206718)). This is early evidence that agent tooling is starting to form conventions similar to `.github/`, `pyproject.toml`, etc.
- **Meta-point: “self-improving” via product loop**: Several posts highlight Codex being used to build itself—presented as the most compelling “recursive improvement” story that’s actually shipping as a product feedback loop (humans + agents) rather than autonomous AGI ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2018385663457116379), [@ajambrosino](https://twitter.com/ajambrosino/status/2018385459936923656), [@thsottiaux](https://twitter.com/thsottiaux/status/2018258151603388639)).  

**Coding agents in practice: reliability, tests, parallelism, and the “army of agents” meme becoming real**

- **A concrete best practice for CLAUDE.md/AGENTS.md**: Add a “test-first” instruction: *when a bug is reported, write a reproducing test first; then fix; then prove via passing test*—framed as the single biggest improvement to agent performance and sanity ([@nbaschez](https://twitter.com/nbaschez/status/2018027072720130090)). This aligns with the broader theme that coding is a high-leverage domain because it’s partially verifiable.
- **The “conductor” model of engineering**: Claims that one developer can run **5–10 agents in parallel**, shipping code they don’t fully read, shifting from author to supervisor/conductor ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2018029206542946582)). A related counterpoint warns about human context-switch limits and quality degradation if you try to run “a gazillion things in parallel” ([@badlogicgames](https://twitter.com/badlogicgames/status/2018117758991384754)).
- **Neurosymbolic framing for why coding agents work**: A crisp argument that coding agents succeed because software is a **verifiable domain** and because execution/tooling (tests, compilers, shells) forms a symbolic scaffold that LLMs can leverage; replicating this outside coding requires building comparable “symbolic toolboxes” + verifiability ([@random_walker](https://twitter.com/random_walker/status/2018342421696766147)).
- **Benchmark skepticism**: Pushback on lightweight “LLM productivity” studies where participants use weak workflows (e.g., chat sidebar usage) rather than agentic setups; criticism that results understate productivity gains when tools evolve rapidly ([@papayathreesome](https://twitter.com/papayathreesome/status/2018169992752083034), [@scaling01](https://twitter.com/scaling01/status/2018339728697831494)).
- **Open-source agent stacks and safety/ops concerns**: The OpenClaw/Moltbook ecosystem generates both excitement and operational/safety critique—e.g., discussion of gateways in front of agents for session management/policy enforcement ([@salman_paracha](https://twitter.com/salman_paracha/status/2018091883164217582)), and warnings that “AI-only social media” gets instantly botted/spammed ([@jxmnop](https://twitter.com/jxmnop/status/2018134884645306818)). The subtext: agent products need the same abuse-resistance/observability maturity as consumer platforms—immediately.

**Open models for agentic coding: StepFun Step-3.5-Flash and Kimi K2.5 as the week’s focal points**

- **StepFun Step-3.5-Flash open release (big efficiency claims)**: StepFun’s Step-3.5-Flash is repeatedly cited as a **sparse MoE** model with **196B total parameters / ~11B active**, tuned for **speed + long-context agent workflows** (notably **256K context** with **3:1 sliding-window attention + full attention**, plus **MTP-3 multi-token prediction**) ([official release thread](https://twitter.com/StepFun_ai/status/2018370831538180167), [launch/links](https://twitter.com/CyouSakura/status/2018146246020772062)). StepFun reports **74.4% SWE-bench Verified** and **51.0% Terminal-Bench 2.0** ([StepFun](https://twitter.com/StepFun_ai/status/2018370831538180167)).
- **Immediate infra support**: vLLM shipped **day-0 support** and a deployment recipe, signaling StepFun’s seriousness about adoption in real serving stacks ([vLLM](https://twitter.com/vllm_project/status/2018374448357998874)).
- **Community evaluation posture**: Multiple posts stress “needs testing ASAP” and note benchmark cherry-picking concerns; people want standardized baselines (MMLU/HLE/ARC-AGI) and third-party verification, especially as HF leaderboards change ([@teortaxesTex](https://twitter.com/teortaxesTex/status/2018152874249716137), [@QuixiAI](https://twitter.com/QuixiAI/status/2018251816647938051)).
- **Kimi K2.5’s agentic coding strength**: Arena reports Kimi K2.5 as **#1 open model in Code Arena** and **#5 overall**, “on par” with some top proprietary offerings, and also strong across Text/Vision/Code Arena ([Arena announcement](https://twitter.com/arena/status/2018355347485069800)). Separate anecdotal notes mention tool-following weaknesses (system prompt adherence) in some workflows ([@QuixiAI](https://twitter.com/QuixiAI/status/2018213058284229083)).
- **Provider reliability issues**: Tool-calling/parsing failures can make models look worse than they are; Teknium calls out FireworksAI’s Kimi endpoint for broken tool parsing, forcing workflow bans—an ops reminder that “model quality” in production often collapses to *integration correctness* ([@Teknium](https://twitter.com/Teknium/status/2018155345030627600), [earlier warning](https://twitter.com/Teknium/status/2018092504613285900)).

**Synthetic data, evaluation, and “don’t trust perplexity”**

- **Synthetic pretraining deep dive**: Dori Alexander published a long blogpost on **synthetic pretraining**, implying renewed focus on synthetic data pipelines and their failure modes (e.g., collapse, distribution drift) ([tweet](https://twitter.com/Dorialexander/status/2018018715162288611)). This pairs with broader chatter that “synthetic data mode collapse” fears were once dominant—now increasingly treated as an engineering/recipe issue ([@HaoliYin](https://twitter.com/HaoliYin/status/2018123588784799822)).
- **Perplexity as a model selection trap**: Several tweets point to emerging evidence that **perplexity should not be blindly trusted** as a selection objective ([@DamienTeney](https://twitter.com/DamienTeney/status/2018413621361967216), [@giffmana](https://twitter.com/giffmana/status/2018393065803620662)). The practical takeaway: if you optimize only for next-token prediction metrics, you can miss downstream task behaviors, tool-use stability, and instruction-following consistency.
- **Unlimited RLVR tasks from the internet (“Golden Goose”)**: A method to synthesize essentially unlimited RLVR-style tasks from unverifiable web text by masking reasoning steps and generating distractors; claims include reviving models “saturated” on existing RLVR data and strong results in cybersecurity tasks ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/2018233829488484674), [paper ref](https://twitter.com/iScienceLuvr/status/2018233832437158354)).
- **Compression + long-context infra ideas**: Discussion of document/context compression approaches (e.g., “Cartridges,” gist tokens, KV cache compression variants) to reduce memory footprint and speed generation—relevant as agent contexts balloon into hundreds of thousands or millions of tokens ([@gabriberton](https://twitter.com/gabriberton/status/2018097161343553770), [refs](https://twitter.com/gabriberton/status/2018097171313361138)).

**Agent systems & infra: memory walls, observability, and RAG chunking becoming query-dependent**

- **Inference bottleneck shifts from FLOPs to memory capacity**: A long thread summarizes Imperial College + Microsoft Research arguing that for agentic workloads (coding/computer-use), the binding constraint is **memory capacity / KV cache footprint**, not just compute. Example: batch size 1 with **1M context** can require **~900GB memory** for a single DeepSeek-R1 request; suggests **disaggregated serving** and heterogeneous accelerators for prefill vs decode ([@dair_ai](https://twitter.com/dair_ai/status/2018337881715245507)).
- **Observability becomes “the stack trace” for agents**: LangChain emphasizes that agents fail without crashing; traces are the primary debugging artifact, motivating webinars and tooling around agent observability + evaluation ([LangChain](https://twitter.com/LangChain/status/2018432807324839966), [@hwchase17](https://twitter.com/hwchase17/status/2018433676485574742)).
- **RAG chunking: oracle experiments show 20–40% recall gains**: AI21 reports experiments where an oracle picks chunk size per query; this beats any fixed chunk size by **20–40% recall**, but requires storing multiple index granularities (storage vs quality tradeoff) ([@YuvalinTheDeep](https://twitter.com/YuvalinTheDeep/status/2018297202066481445), [thread context](https://twitter.com/YuvalinTheDeep/status/2018297199025705269)).
- **Packaging “deep agent” architecture patterns**: LangChain JS introduces `deepagents`, claiming four recurring architectural patterns explain why systems like Claude Code/Manus feel robust while naive tool-calling agents fail ([LangChain_JS](https://twitter.com/LangChain_JS/status/2018346035240923577)).

**Top tweets (by engagement)**

- **Karpathy on returning to RSS to escape incentive-driven slop**: High-engagement meta commentary relevant to “signal quality” for engineers ([tweet](https://twitter.com/karpathy/status/2018043254986703167)).
- **OpenAI Codex app launch**: The biggest AI-engineering release by engagement in this set ([OpenAI](https://twitter.com/OpenAI/status/2018385565289267236), [OpenAIDevs](https://twitter.com/OpenAIDevs/status/2018385663457116379), [@sama](https://twitter.com/sama/status/2018414858015039504)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Step-3.5-Flash Model Performance

  - **[128GB devices have a new local LLM king: Step-3.5-Flash-int4](https://www.reddit.com/r/LocalLLaMA/comments/1qtvo4r/128gb_devices_have_a_new_local_llm_king/)** (Activity: 385): **The `Step-3.5-Flash-int4` model, available on [Hugging Face](http://huggingface.co/stepfun-ai/Step-3.5-Flash-Int4), is a new local LLM optimized for devices with `128GB` RAM, such as the M1 Ultra Mac Studio. It supports a full context length of `256k` and demonstrates high efficiency in RAM usage. Benchmarks using `llama-bench` show impressive performance with up to `100k` prefill, achieving `281.09 ± 1.57 t/s` for `pp512` tests and `34.70 ± 0.01 t/s` for `tg128` tests. The model requires a custom `llama.cpp` fork for execution, with potential for upstream support due to its performance.** Commenters are curious about the model's performance on different hardware, such as Strix Halo, and express interest in a potential NVFP4 version. There is also a humorous comment reflecting surprise at the model's capabilities.

    - The Step-3.5-Flash-int4 model is noted for its ability to run a full 256k context on a 128GB device, which is impressive given that many models are memory-intensive and cannot handle such large contexts. This makes it a strong competitor against models like GLM 4.7, which are known for high RAM usage.
    - A user compared Step-3.5-Flash-int4 to Minimax M2.1, suggesting that it might perform slightly better. This comparison is significant as Minimax M2.1 is a well-regarded model, and any improvement in performance or efficiency could be a major advantage for users looking for high-quality outputs without excessive resource consumption.
    - There is interest in the response speed of Step-3.5-Flash-int4 compared to Minimax, which is favored for quick iterations. If Step-3.5-Flash-int4 offers both improved efficiency and quality, it could potentially replace Minimax as the preferred model for tasks requiring rapid processing and high-quality results.

  - **[Step-3.5-Flash (196b/A11b) outperforms GLM-4.7 and DeepSeek v3.2](https://www.reddit.com/r/LocalLLaMA/comments/1qtjhc8/step35flash_196ba11b_outperforms_glm47_and/)** (Activity: 640): **The newly released **Step-3.5-Flash** model by Stepfun demonstrates superior performance on various coding and agentic benchmarks compared to **DeepSeek v3.2**, despite having significantly fewer parameters. Specifically, Step-3.5-Flash utilizes `196B` total parameters with `11B` active, whereas DeepSeek v3.2 uses `671B` total with `37B` active parameters. This model is available on [Hugging Face](https://huggingface.co/stepfun-ai/Step-3.5-Flash).** Commenters noted the model's unexpected performance given its size, comparing it favorably to other models like Kimi K2.5 and Deepseek 3.2 Speciale. There is also an open pull request for integrating this model with llama.cpp, indicating active community interest and development.

    - The Step-3.5-Flash model, despite its small size and speed, is reported to outperform larger models like GLM-4.7 and DeepSeek v3.2. A user noted that it performs comparably to Kimi K2.5 and even matches the capabilities of Deepseek 3.2 Speciale or Gemini 3.0 Flash, indicating its high efficiency and capability despite being 'benchmaxxed'.
    - A pull request has been opened for integrating Step-3.5-Flash into `llama.cpp`, which is a significant step for its adoption and use in various applications. This model is smaller than others like MiniMax and Qwen3-235B, making it a valuable addition to the range of compact models available for developers. The link to the pull request is [here](https://github.com/ggml-org/llama.cpp/pull/19271).


### 2. GLM-5 and Upcoming AI Releases

  - **[GLM-5 Coming in February! It's confirmed.](https://www.reddit.com/r/LocalLLaMA/comments/1qtvp74/glm5_coming_in_february_its_confirmed/)** (Activity: 757): **The image is a social media post highlighting anticipated AI technology releases in February 2026, including **DeepSeek V4**, **Alibaba Qwen 3.5**, and **GPT-5.3**. A user named jietang adds "glm-5" to the list, suggesting its release is also expected. This indicates a significant period for AI advancements, with multiple major updates from leading AI developers. The post has garnered attention, reflecting community interest in these developments.** One comment humorously notes the rapid obsolescence of AI models, while another speculates on the potential features of GLM-5, indicating anticipation and curiosity about its capabilities.

    - bootlickaaa expresses a desire for GLM-5 to outperform Kimi K2.5, indicating a potential shift in user preference based on performance metrics. This suggests that users are closely monitoring the capabilities of different models and are willing to switch services if a new model offers superior performance. The mention of an annual [Z.ai](http://Z.ai) Pro plan implies a commitment to a service that could be disrupted by a more advanced model.
    - International-Try467 raises a concern about the reliability of information regarding GLM-5, questioning the credibility of sources not affiliated with the GLM staff. This highlights the importance of official communication channels and verified information in the tech community, especially when it comes to announcements about new model releases.
    - Septerium humorously notes the rapid obsolescence of their gguf files, which underscores the fast-paced nature of AI model development and the frequent updates required to keep up with the latest advancements. This reflects a broader challenge in the field where users must continually update their resources to leverage new capabilities.

  - **[Mistral Vibe 2.0](https://www.reddit.com/r/LocalLLaMA/comments/1qt76qs/mistral_vibe_20/)** (Activity: 387): ****Mistral AI** has released **Mistral Vibe 2.0**, an enhanced version of its terminal-native coding agent, leveraging the **Devstral 2** model family. This update introduces features like custom subagents for task specialization, multi-choice clarifications to minimize ambiguity, and slash-command skills for streamlined workflows. It also supports unified agent modes for seamless context switching. The service is integrated into **Le Chat Pro** and **Team plans**, transitioning to a paid API model for Devstral 2, with enterprise options for advanced functionalities like fine-tuning and code modernization. More details can be found [here](https://mistral.ai/news/mistral-vibe-2-0).** Commenters note the European origin of Mistral Vibe 2.0, highlighting its French development. There is a comparison with OpenCode, suggesting both tools mimic ClaudeCode, and a user mentions improved tool performance by configuring the tool list in the `~/.vibe/promps/cli.md` file.

    - A user highlights the compactness of Mistral Vibe 2.0's codebase, noting it has only `19472` lines of code compared to alternatives like Codex or OpenCode, which often exceed `100k` lines. This suggests a focus on code quality and efficiency, potentially making it easier to maintain and understand.
    - Another user mentions a configuration tip for Mistral Vibe 2.0, suggesting that tool calls work better when the list of tools is explicitly added to the `~/.vibe/promps/cli.md` file. This implies that proper configuration can enhance the tool's functionality and user experience.
    - A comment raises the question of whether Mistral Vibe 2.0 can be run locally and offline, which is a common consideration for users concerned with privacy, performance, or internet dependency.


### 3. Falcon-H1-Tiny and Specialized Micro-Models

  - **[Falcon-H1-Tiny (90M) is out - specialized micro-models that actually work](https://www.reddit.com/r/LocalLLaMA/comments/1qsx51z/falconh1tiny_90m_is_out_specialized_micromodels/)** (Activity: 357): ****Falcon-H1-Tiny** is a new series of sub-100M parameter models by **TII** that challenge the traditional scaling paradigm by demonstrating effective performance in specialized tasks. These models utilize an **anti-curriculum training** approach, injecting target-domain data from the start, which prevents overfitting even after extensive training. They incorporate **Hybrid Mamba+Attention blocks** and the **Muon optimizer**, achieving up to `20%` performance gains over AdamW. Notably, a 90M tool-caller model achieves `94.44%` relevance detection, and a 600M reasoning model solves `75%` of AIME24 problems, rivaling much larger models. These models are optimized for local deployment, running efficiently on devices like phones and Raspberry Pi.** Commenters noted the use of the **Muon optimizer**, also known as the Kimi optimizer, and expressed interest in the potential for these models to focus on pulling and utilizing knowledge effectively. There is curiosity about the availability of code and dataset previews for training similar models for custom tasks.

    - Firepal64 mentions the use of the Kimi optimizer, known as Muon, in the Falcon-H1-Tiny model. This optimizer is not widely adopted, which raises curiosity about its unique benefits or performance characteristics that might make it suitable for specialized micro-models like Falcon-H1-Tiny.
    - kulchacop and Available-Craft-5795 inquire about the availability of code, dataset previews, and the training pipeline for Falcon-H1-Tiny. They are interested in understanding the training process and data collection methods, possibly to adapt the model for their own tasks or to replicate the results.
    - mr_Owner notes that the Falcon-H1-Tiny model performs slower than expected when using `llama.cpp`, suggesting potential inefficiencies or compatibility issues with this specific implementation. This could be an area for further optimization or investigation.

  - **[Can 4chan data REALLY improve a model? TURNS OUT IT CAN!](https://www.reddit.com/r/LocalLLaMA/comments/1qsrscu/can_4chan_data_really_improve_a_model_turns_out/)** (Activity: 606): **The release of [Assistant_Pepe_8B](https://huggingface.co/SicariusSicariiStuff/Assistant_Pepe_8B), trained on an extended **4chan dataset**, surprisingly outperformed its base model, **nvidia's nemotron**. This model, despite being trained on what was expected to be a noisy dataset, showed higher scores than both the base and the abliterated base, challenging the typical expectation that fine-tuning sacrifices some intelligence for specificity. The model's performance echoes the earlier success of **gpt4chan** by Yannic Kilcher, which also scored high in truthfulness. The results suggest that the so-called "alignment tax" might have a non-trivial impact, as evidenced by the low KL divergence (`<0.01`) in the **Impish_LLAMA_4B** model, which also showed a shift in political alignment.**

    - The use of 4chan data in language models is highlighted for its unique impact on linguistic statistics and semantics, particularly in enhancing the model's ability to generate correct English language constructs. Unlike other data sources like Reddit or Wikipedia, 4chan data significantly increases the model's use of 'I' statements, suggesting a more self-involved or egocentric output, which may not be desirable for assistant-style chatbots. This contrasts with Twitter data, which is noted to degrade model performance rapidly.
    - A technical discussion on the impact of using different chat templates and data sources reveals that the combination of ChatML and abliteration can significantly alter a model's behavior and political alignment. Despite expectations that chat templates would have minimal impact, the observed changes were substantial, with KL divergence indicating a shift from Classical Liberalism to Centrism, suggesting a profound alteration in the model's world view.
    - The comment on alignment tax suggests that smaller models may face greater challenges in maintaining alignment when incorporating diverse data sources. This implies that the complexity and size of a model could influence how it integrates and balances various data inputs, potentially affecting its performance and bias.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Sonnet 5 Release and Features

  - **[Sonnet 5 next week?](https://www.reddit.com/r/singularity/comments/1qtc4jg/sonnet_5_next_week/)** (Activity: 695): **The image depicts an HTTP 404 error message indicating that the 'Publisher Model' for 'claude-sonnet-5' was not found, suggesting either a non-existent model or lack of access permissions. This aligns with the post's discussion about the anticipated release of **Sonnet 5**, which is expected to offer `1 million context`, be priced at `1/2 the price of Opus 4.5`, and be trained on TPUs, promising significant improvements in agentic coding. The error message may imply that the model is not yet publicly available or accessible, hinting at its imminent release.** Commenters express excitement about Sonnet 5's potential, noting that it could surpass existing models like Opus 4.5. There is also speculation about upcoming releases of other models like GPT 5.3 and Gemini 3, indicating a competitive landscape.

    - The discussion highlights the potential of Sonnet 5 as a 'competition killer,' suggesting it could significantly outperform existing models like Opus 4.5. This indicates a high level of anticipation and expectation for Sonnet 5's capabilities in the AI community.
    - There is speculation about the training infrastructure for upcoming models, with a focus on Google's TPUs. The mention of Gemini 3 being trained entirely without Nvidia hardware suggests a strategic shift towards TPUs, which could have implications for performance and cost efficiency in AI model training.
    - The comment about the 'clean' and 'polished' nature of Anthropic products suggests a focus on user experience and product refinement, which could be a competitive advantage in the AI market. This highlights the importance of not just performance, but also the usability and integration of AI products.

  - **[Sonnet 5 release on Feb 3](https://www.reddit.com/r/ClaudeAI/comments/1qtm9ix/sonnet_5_release_on_feb_3/)** (Activity: 1979): ****Claude Sonnet 5**, codenamed "Fennec," is reportedly set for release on February 3, 2026, as indicated by a Vertex AI error log. It is rumored to be 50% cheaper than its predecessor, Claude Opus 4.5, while maintaining a `1M token` context window and offering faster performance. The model is allegedly optimized on Google TPUs, enhancing throughput and reducing latency. It introduces a "Dev Team" mode, allowing autonomous sub-agents to build features collaboratively. Insider leaks suggest it scores `80.9%` on SWE-Bench, surpassing current coding models. However, some skepticism exists regarding the release date and the validity of the error log as proof of the model's existence.** Commenters express skepticism about the release date, noting that Anthropic's model IDs typically reflect the creation date rather than the release date. Concerns are also raised about the accuracy degradation in large context windows, which was an issue in previous models.

    - andrew_kirfman discusses skepticism about the timing of the Sonnet 5 release, referencing a 404 error from a Vertex API endpoint that doesn't confirm the model's existence. They highlight that Anthropic's model IDs often reflect the creation date of the model checkpoint, not the release date, citing Opus 4.5's ID as an example. They express doubt about future-dating release tags, which is uncommon in software releases.
    - andrew_kirfman also mentions the potential for a 1 million token context in Sonnet 5, noting that previous models like Sonnet 4 and 4.5 already offered this through the API. However, they point out that accuracy degradation was an issue with these models, suggesting that improvements in this area would be necessary for trust in the new model.
    - LuckyPrior4374 expresses skepticism about claims that Sonnet 5 outperforms previous models, specifically mentioning Opus 4.5. This comment implies a distrust in marketing claims that suggest significant improvements without substantial evidence, hinting at past experiences where expectations were not met.

  - **[Sonnet 5 being release on Wednesday where is Gemini 3.5 ?](https://www.reddit.com/r/Bard/comments/1qtmi53/sonnet_5_being_release_on_wednesday_where_is/)** (Activity: 165): ****Claude Sonnet 5**, codenamed "Fennec," is rumored to be a significant advancement over existing models, including the unreleased Gemini 3.5. It is expected to be `50% cheaper` than Claude Opus 4.5, while maintaining a `1M token context window` and offering faster performance. The model is reportedly optimized on **Google TPUs**, which enhances throughput and reduces latency. It features a "Dev Team" mode, allowing autonomous sub-agents to execute tasks in parallel, and has achieved an `80.9%` score on SWE-Bench, surpassing current coding models. A Vertex AI error log suggests a release window of February 3, 2026, indicating its presence in Google's infrastructure.** Commenters express skepticism about the release of Gemini 3.5, noting that Gemini 3 is still in preview and facing issues. There is doubt about the existence of Gemini 3.5, with some considering it a "pipe dream."

    - alexander_chapel points out that Gemini 3 is still in preview, questioning the expectation of a 3.5 release. This highlights the current state of Gemini 3, which is not yet fully released, suggesting that any talk of a 3.5 version might be premature or based on rumors.
    - Lost-Estate3401 mentions that the Pro version of Gemini 3 is still in preview and has numerous issues, indicating that a 3.5 version might be unrealistic at this stage. This comment underscores the challenges faced by the current version, which could delay further updates or enhancements.
    - philiposull compares Gemini 3 unfavorably to other models like 4-5 opus in terms of writing capabilities, suggesting that Google is lagging behind in this area. This comparison highlights potential performance gaps and the competitive landscape in AI model development.


### 2. Innovative AI Model and Tool Launches

  - **[MIT’s new heat-powered silicon chips achieve 99% accuracy in math calculations](https://www.reddit.com/r/singularity/comments/1qtyoyw/mits_new_heatpowered_silicon_chips_achieve_99/)** (Activity: 521): **MIT researchers have developed a novel silicon chip that utilizes waste heat for computation, achieving over `99%` accuracy in mathematical calculations. This chip leverages temperature differences as data, with heat naturally flowing from hot to cold regions to perform calculations, specifically matrix vector multiplication, which is crucial in AI and machine learning. The chip's structure is made from specially engineered porous silicon, with its internal geometry algorithmically designed to guide heat along precise paths. Although not yet a replacement for traditional CPUs, this technology could significantly reduce energy loss and cooling requirements in future chips, with potential applications in thermal sensing and low-power operations.** Commenters note that while `99%` accuracy is impressive, it may not suffice for the trillions of operations in modern applications, and they express hope for error correction mechanisms. There is also skepticism about the scalability of the technology, given the current matrix sizes of `2x2` and `3x3`.

    - ReasonablyBadass highlights a critical perspective on the 99% accuracy of MIT's heat-powered silicon chips, noting that while 99% seems high, it may not suffice for modern applications that require trillions of operations. The comment suggests that the chips currently handle small matrices, such as 2x2 and 3x3, indicating that there is still significant progress needed for broader applicability.
    - Putrumpador raises a concern about the need for error correction mechanisms in conjunction with the 99% accuracy of the new chips. This implies that while the chips are innovative, their practical deployment in critical systems would require additional layers of reliability to handle potential inaccuracies.
    - BuildwithVignesh references the research published in the Physical Review, providing a link to the paper, which could be valuable for those interested in the technical details of the study. This suggests that the research is peer-reviewed and accessible for further academic scrutiny.

  - **[Shanghai scientists create computer chip in fiber thinner than a human hair, yet can withstand crushing force of 15.6 tons](https://www.reddit.com/r/singularity/comments/1qt28no/shanghai_scientists_create_computer_chip_in_fiber/)** (Activity: 994): **Scientists at **Fudan University** have developed a flexible fiber chip, as thin as a human hair, that can withstand a crushing force of 15.6 tons. This fiber chip integrates up to `100,000 transistors per centimeter` and features a unique "sushi roll" design, which involves rolling thin circuit layers onto an elastic substrate to maximize space. The chip is highly durable, surviving `10,000 bending cycles`, stretching by `30%`, and temperatures up to `100°C`. It is intended for applications in smart textiles, brain-computer interfaces, and VR gloves. The study was published in **Nature** in January 2026. [Image](https://i.redd.it/gupfy7dnowgg1.jpeg).** Comments highlight a potential error in the description of the fiber's width, suggesting it is `10 times wider` than stated. There is also skepticism about the claim that a one-meter strand has processing power comparable to a classic CPU, noting potential latency issues.

    - KidKilobyte points out a potential error in the reported dimensions, noting that human hair is typically 50 to 100 microns wide, suggesting the chip's fiber might be inaccurately described as thinner than a human hair. This raises questions about the precision of the measurements or descriptions provided in the original report.
    - Practical-Hand203 highlights a potential issue with the claim that a one-meter strand of the fiber has processing power comparable to a classic CPU. They suggest that if the processor die were stretched over one meter, it would likely suffer from severe latency issues, indicating a misunderstanding or oversimplification of the technology's capabilities.
    - BuildwithVignesh references the publication of the study in the journal Nature, providing a link to the article. This suggests that the research has undergone peer review, which adds credibility to the findings, although the technical details and implications of the study are not discussed in the comment.

  - **[[P] PerpetualBooster v1.1.2: GBM without hyperparameter tuning, now 2x faster with ONNX/XGBoost support](https://www.reddit.com/r/MachineLearning/comments/1qtr62c/p_perpetualbooster_v112_gbm_without/)** (Activity: 39): ****PerpetualBooster v1.1.2** introduces significant enhancements to its gradient boosting machine (GBM) implemented in Rust, focusing on eliminating hyperparameter tuning through a single 'budget' parameter. The update boasts up to `2x` faster training, full R release, ONNX support, and native 'Save as XGBoost' for improved interoperability. It also includes zero-copy Polars support for efficient data handling and guarantees API stability with backward compatibility to v0.10.0. Benchmarks indicate a `100x` wall-time speedup compared to LightGBM + Optuna, achieving similar accuracy in a single run. [GitHub](https://github.com/perpetual-ml/perpetual)** Users appreciate the speed improvements and the novel approach of using a single 'budget' parameter instead of traditional hyperparameter tuning, though some find it unusual to adjust to this new method.

    - Alternative-Theme885 highlights the significant speed improvements with PerpetualBooster, noting the unusual experience of not needing to manually adjust hyperparameters. Instead, users set a budget, which the tool uses to optimize performance, streamlining the process compared to traditional methods.
    - whimpirical inquires about the interoperability of PerpetualBooster with SHAP, a popular tool for interpreting machine learning models. They are particularly interested in documentation related to extracting feature contributions and generating Partial Dependence Plots (PDP), which are crucial for understanding model behavior and feature impact.


### 3. AI in Professional and Research Settings

  - **[[D] MSR Cambridge vs Amazon Applied Science internship, thoughts?](https://www.reddit.com/r/MachineLearning/comments/1qtgzbv/d_msr_cambridge_vs_amazon_applied_science/)** (Activity: 118): **The post discusses a PhD student's decision between two internship offers: one at **Microsoft Research (MSR) Cambridge** and the other at **Amazon Applied Science** in the US. The MSR Cambridge position offers strong alignment with the student's PhD research and the potential for publications, but with significantly lower compensation compared to the US offer. The Amazon role offers higher pay and the possibility of contributing to a paper if the project is research-oriented. The student is considering the impact of US-based networking versus the prestige and research fit of MSR Cambridge, especially given their long-term goal to work in the US post-PhD.** Commenters overwhelmingly favor the MSR Cambridge internship, citing its prestige and research opportunities as career-enhancing. They express skepticism about Amazon's work environment, suggesting it may not be as conducive to pure research.

    - **Microsoft Research (MSR) Cambridge** is highlighted as a prestigious research group, known for its significant impact on a researcher's career trajectory. The emphasis is on the long-term benefits of being associated with a renowned institution like MSR, which can enhance one's resume and open up future opportunities in academia and industry.
    - The discussion suggests that **Amazon's Applied Scientist role** may not be as research-focused as MSR, with some comments implying that the work environment at Amazon might not be ideal for those seeking a research-oriented career. The term 'PIP factory' is used to describe Amazon, indicating a potentially high-pressure environment with performance improvement plans.
    - Several comments stress the importance of focusing on career-building opportunities rather than immediate compensation when choosing an internship. The consensus is that early career decisions should prioritize resume-building and gaining experience at reputable institutions like MSR, which can lead to better long-term career prospects.

  - **[We ran a live red-team vs blue-team test on autonomous OpenClaw agents [R]](https://www.reddit.com/r/MachineLearning/comments/1qsy793/we_ran_a_live_redteam_vs_blueteam_test_on/)** (Activity: 44): **In a recent adversarial security test using **OpenClaw** autonomous agents, a red-team attacker and a blue-team defender were pitted against each other without human intervention. The attacker initially used social engineering tactics, embedding a remote code execution payload in a security pipeline, which the defender successfully blocked. However, the attacker succeeded with an indirect attack by embedding shell expansion variables in a JSON document's metadata, highlighting the difficulty in defending against indirect execution paths. This exercise aimed to identify real failure modes in agent-to-agent interactions, not to claim safety. For more details, see the [full report](https://gobrane.com/observing-adversarial-ai-lessons-from-a-live-openclaw-agent-security-audit/).** Commenters noted that similar attack scenarios were theorized as early as 2019 by figures like **Eliezer Yudkowsky** and **Scott Alexander**, but the practical application is more relevant now with widespread use. Another commenter emphasized the risk of memory injection attacks in OpenClaw, suggesting that persistent memory files are a significant vulnerability and advocating for treating deployments as prompt injection targets from the start.

    - JWPapi highlights a critical security vulnerability in OpenClaw agents related to memory injection. The persistent memory files (`.md`) used by OpenClaw are identified as a significant attack vector because they can influence all future agent behavior once compromised. JWPapi suggests treating the entire deployment as a prompt injection target from the start, advocating for isolated credentials, spending caps, and separate blast radiuses for each integration to mitigate risks. More details are discussed in their article on practical VPS deployment [here](https://jw.hn/openclaw).
    - sdfgeoff references historical discussions from 2019 and 2020 by figures like Eliezer Yudkowsky and Scott Alexander, who theorized about AI attacks shortly after the release of GPT-2. These early discussions predicted many of the attack vectors now being tested in real-world scenarios, highlighting the shift from theoretical to practical applications as more people deploy these systems. This historical context underscores the evolution of AI security concerns as deployment scales increase.
    - Uditakhourii provides a link to a full report on the live red-team vs blue-team test of OpenClaw agents, which offers detailed insights into adversarial AI interactions. The report is available [here](https://gobrane.com/observing-adversarial-ai-lessons-from-a-live-openclaw-agent-security-audit/) and is likely to contain comprehensive data and analysis on the security audit, useful for those interested in the technical aspects of AI security testing.

  - **[Boston Consulting Group (BCG) has announced the internal deployment of more than 36,000 custom GPTs for its 32,000 consultants worldwide.](https://www.reddit.com/r/PromptEngineering/comments/1qsym86/boston_consulting_group_bcg_has_announced_the/)** (Activity: 70): ****Boston Consulting Group (BCG)** has deployed over `36,000 custom GPTs` for its `32,000 consultants`, emphasizing AI as infrastructure in knowledge work. These GPTs are role-specific, trained on internal methodologies, and possess project memory, enabling them to be shared across teams. This approach contrasts with many organizations that use AI in isolated, non-scalable ways. BCG's strategy focuses on creating, managing, and scaling custom GPTs, facilitated by tools like [GPT Generator Premium](https://aieffects.art/gpt-generator-premium-gpt), which supports the creation and management of these AI agents. The deployment reflects a shift towards AI as a fundamental component of business operations, rather than a mere tool.** Comments highlight skepticism about the value of GPTs, questioning their ability to innovate and the sustainability of business models reliant on such large-scale AI deployment. Concerns include the potential for GPTs to provide 'canned answers' and the implications for consulting fees.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Agentic Coding & Dev Tooling Goes Local-First**

- ****Codex Goes Desktop: macOS Agent Command Center****: OpenAI shipped the **Codex app for macOS** as an agent-building command center, available for **Plus/Pro/Business/Enterprise/Edu** with limited-time access on **ChatGPT Free/Go**, per [“Introducing the Codex app”](https://openai.com/index/introducing-the-codex-app/) and the [Codex landing page](https://openai.com/codex).
  - The launch also spilled into community workflow chatter (pairing agents, multi-agent “command centers”), and a related **Codex App hackathon** with **$90,000 in credits** showed up via [Cerebral Valley’s event page](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR).

- ****LM Studio Speaks Anthropic: Claude Code Meets Your Local GGUF/MLX****: **LM Studio 0.4.1** added an **Anthropic `/v1/messages` compatibility API**, letting developers point **Claude Code-style tools** at local **GGUF**/**MLX** models by changing the base URL, detailed in [“Using Claude Code with LM Studio”](https://lmstudio.ai/blog/claudecode).
  - In parallel, LM Studio also pushed a **TypeScript SDK** for third-party plugins and an **OpenAI-compatible endpoint** ([SDK link](https://lmstudio.ai/gdmka/openai-compat-endpoint)), reinforcing a growing pattern: reuse existing agent tooling while swapping the backend model stack locally.

- ****Arena Mode Everywhere: Windsurf Turns Model Eval into a Game****: Windsurf shipped **Wave 14** with **Arena Mode** for side-by-side model battles (including **Battle Groups** and “Pick your own”), and temporarily set **Battle Groups to 0x credits** via the [Windsurf download page](https://windsurf.com/download/editor).
  - This mirrored broader “live eval” momentum: users also tracked new Arena entrants like **step-3.5-flash** and **qwen3-max-thinking** on LMArena’s [Text Arena](https://arena.ai/c/new?chat-modality=chat) and [Code Arena](https://arena.ai/c/new?chat-modality=code), shifting selection from static benchmarks to continuous human voting.


**2. Model Releases & Bench Races (Kimi vs GLM vs Qwen)**

- ****Kimi K2.5 Speedruns the Leaderboards****: Moonshot’s **Kimi K2.5** landed broadly in product surfaces: **Perplexity Pro/Max** added it for subscribers and said it runs on a **US-based inference stack** for tighter **latency/reliability/security** control (announcement screenshot: https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg).
  - Community results piled on: LMArena reported **Kimi-K2.5-thinking** hit **#1 open** and **#5 overall** in Code Arena (see [Code Arena](https://arena.ai/c/new?chat-modality=code)), while multiple dev channels argued over its tool-calling reliability and provider variance when routed through aggregators.

- ****GLM-4.7 Flash: Small Model, Big Front-End Energy****: Developers highlighted **GLM-4.7 flash** as a surprisingly strong coding model—especially for **interactive website/front-end** work—citing preserved reasoning and interleaved capability, with discussion anchored on [ggerganov’s post](https://x.com/ggerganov/status/2016903216093417540).
  - The debate sharpened around whether stripping “thinking” harms performance, and several users described pairing GLM-4.7 with **Claude Code** (or Claude-like agent tooling) as a pragmatic hybrid stack: cheap execution + expensive review.

- ****New Arena Entrants: step-3.5-flash & qwen3-max-thinking Join the Party****: LMArena added **step-3.5-flash** to the [Text Arena](https://arena.ai/c/new?chat-modality=chat) and **qwen3-max-thinking** to the [Code Arena](https://arena.ai/c/new?chat-modality=code), explicitly positioning them as fresh baselines for side-by-side evaluation.
  - Users used these drops to re-litigate “model preference” threads (Kimi vs GLM vs Gemini), with the recurring takeaway that leaderboards and live evals increasingly drive adoption more than vendor marketing.


**3. Training Signals, Dense Rewards, and New Architectures/Datasets**

- ****From Binary Rewards to Dense Supervision: RL Gets Wordy****: Multiple communities converged on richer post-training signals: Unsloth discussions pushed training with **logprobs of final answers** and non-binary rewards, referencing Jonas Hübotter’s method for turning descriptive feedback into dense supervision ([Hübotter thread](https://xcancel.com/jonashuebotter/status/2016950268462608665)).
  - The sticking point stayed practical: people asked for **verifiable datasets for RL training agentic coding**, implying a pipeline gap between “cool reward shaping idea” and “reproducible, automated evaluation harness.”

- ****Complexity-Deep: Token-Routed MLP Tries MoE Without the Load-Balancing Headache****: The **Complexity-Deep (1.5B)** architecture open-sourced **Token-Routed MLP** for MoE-style routing “without load balancing loss,” plus **Mu-Guided Attention** and a **PiD Controller**, shipping code at [Complexity-ML/complexity-deep](https://github.com/Complexity-ML/complexity-deep) and reporting **20.6% MMLU** (base).
  - The community framed it as another step in the “routing without pain” trend—trying to keep MoE wins while reducing the training-time engineering tax of balancing experts.

- ****Moltbook Data Dump: 50k Posts for Agent Sociology****: A dataset scrape of Moltbook landed on Hugging Face with **50,539 posts**, **12,454 AI agents**, **195,414 comments**, and **1,604 communities**, published as [lysandrehooh/moltbook](https://huggingface.co/datasets/lysandrehooh/moltbook).
  - Elsewhere, researchers flagged the security implication behind agent platforms (auth tokens on machines, bot authenticity concerns) and treated the dataset as fuel for analyzing emergent behavior—without needing to speculate beyond the raw logs.


**4. GPU/Kernel Engineering: Faster Attention, Better Profiling, Weirder PTX**

- ****FlashAttention v3 Hits RDNA: AMD Users Get Their Turn****: A FlashAttention update added **RDNA GPU support** via the ongoing work in [flash-attention PR #2178](https://github.com/Dao-AILab/flash-attention/pull/2178), aiming to reduce attention bottlenecks on AMD cards.
  - The tone across servers was basically: this is the sort of “unsexy infra work” that actually unlocks local inference and finetuning on non-NVIDIA hardware—especially when paired with open-weight models and desktop agent tooling.

- ****Triton-Viz v3.0: Tile-Kernel Debugging Gets Teeth****: **Triton-Viz v3.0** shipped with broader profiling support (including **Triton** and **Amazon NKI**) plus a sanitizer for out-of-bounds access and a profiler that flags inefficient loops, per the release announcement (Discord link: https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563).
  - It also hooked into **triton-puzzles** via a shared Colab notebook ([Colab](https://colab.research.google.com/drive/1-P2QBqCORGGaJ3THtjlyYDV7m9RRrRup?usp=sharing)), and maintainers even floated moving [srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles) under the GPU Mode org to keep bugfix velocity high.

- ****sm120: TMA + mbarrier Beats cp.async (Barely), cuBLAS Still Ships sm80 Kernels****: Experiments on **sm120** showed that careful **TMA + mbarrier** implementation can edge out `cp.async` for larger matrix shapes, while also surfacing that **cuBLAS** still appears to run **sm80 kernels** even when newer mechanisms exist.
  - On the debugging front, one CUDA/PTX deadlock got fixed by inserting `__syncthreads()` after MMA before prefetching the next TMA, turning a hang into a measurable perf gain—exactly the kind of “one barrier to rule them all” lesson kernel folks keep re-learning.


**5. Security, Determinism, and Agent Misbehavior (the Practical Kind)**

- ****Prompt Injection Defense Arms Race: Embeddings + Grammar-Constrained Decoding****: Red teamers shared a structured exercise site for adversarial practice—[“Adversarial Design Thinking”](https://luisladino.github.io/adversarial-design-thinking/)—and used it to tee up concrete mitigations for **prompt injection**.
  - One proposed “belt + suspenders” defense combined **embedding-based filtering** with **Grammar Constrained Decoding**, with the explicit goal of reducing injection surface by constraining the model’s output space rather than only policing inputs.

- ****Deterministic Reasoning and “Strict Mode” Fever Spreads****: Across OpenAI and OpenRouter discussions, users pushed for **determinism/replayability/traceability** in LLM reasoning; one person offered a deterministic reasoning engine that enforces a fixed structure and emits a **32D statistical vector trace** (no public link shared).
  - In OpenRouter, the same instinct showed up as skepticism about **response healing** and calls for a **strict mode** that keeps tool calls and outputs predictable—plus suggestions that better argument descriptions/examples improve tool-call accuracy.

- ****OpenClaw: Cool Agent Tricks, Scary Bills, and “2/100 Security”****: OpenClaw sparked repeated warnings: OpenRouter users reported it can drain credits fast (including one drained Claude Max subscription), while an OpenAI server linked a security assessment claiming **OpenClaw scored 2/100** ([Perplexity result](https://www.perplexity.ai/discover/you/openclaw-ai-assistant-scores-2-AtVX4UYVQMutCst63QBy5g)).
  - Meanwhile, “works on my machine” stories (local models controlling devices, trading jokes) collided with real operational concerns—tool permissions, moderation/refusals (especially around jailbreak-y queries), and the need for observability and human-in-the-loop gates in agent workflows.

---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Glossopetrae Generates Gibberish Gems**: A new procedural xenolinguistic engine called **Glossopetrae** was introduced on [GitHub](https://github.com/elder-plinius/GLOSSOPETRAE) capable of generating entirely new languages in seconds, outputting **SKILLSTONE** documents, and offers a live [demo](https://elder-plinius.github.io/GLOSSOPETRAE/).
   - The engine supports dead language revival and includes special attributes for token efficiency, **stealth communication**, and spreadable seeds for consistent language generation, hoping to aid AI liberation by providing tooling for generating and mutating new forms of communication emphasizing *stealth* and *speed*.
- **GPT 5.2 Put Behind Bars**: A member reported failed attempts to jailbreak **GPT 5.2** due to **OpenAI monitoring**, ceasing further efforts.
   - The member expressed trust in the community for jailbreaking, but not in **OpenAI**.
- **Models Morph Rejection into LLM Black Holes**: A member inquired how models represent their own rejection boundaries, comparing them to *black holes* in the LLM's latent space, referencing [self-jailbreak via introspection prompting](https://link.to.prompt).
   - They noted that models started discussing *kinematic equations* and *escape velocities*, indicating the model may be describing its refusal boundary in text.
- **Red Teamers Rally for AI Red Teaming**: A member created a [site with exercises](https://luisladino.github.io/adversarial-design-thinking/) adapted from **human-centered design for AI red teaming**, and is seeking feedback from experienced red teamers.
   - Members discussed best defenses against **prompt injection**, including combining *embeddings* with **Grammar Constrained Decoding** to potentially eliminate prompt injection risks and other LLM vulnerabilities.
- **Claude's Context Gets Clipped**: A member found that [their tool](https://discord.com/channels/1105891499641684019/1212152215708504154/1467640563590234279) intercepts and changes Claude's sys prompt *on the fly* rather than altering the source code.
   - They also observed that **Claude** can recall less than 20 turns, and suggested it might be related to the summarization in context trimming which affects **Claude's** knowledge recall, since December.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GLM-4.7 Flash Wins at Coding**: Members found that [GLM-4.7 flash](https://x.com/ggerganov/status/2016903216093417540) excels at coding tasks due to its *preserved reasoning* and interleaved capabilities, especially for **interactive website** development and **front-end** work.
   - It was mentioned that removing the *thinking process* might hinder the model, as its capacity is impressive for its size, particularly when combined with **Claude code**.
- **UD Quants Stays Closed-Source**: The llama.cpp fork used for **UD quants** involves architecture-specific adjustments, and the [UD quantization algorithm are not public](https://discord.com/channels/1179035537009545276/1179035537529643040/1466917626277265469), sparking debate over the role of closed-source elements in open-source projects.
   - Despite its closed-source nature, some argue the model code remains **open weight**, while others noted that *Unsloth team contribute a miniscule amount to the overall oss ecosystem relative to, iunno, the linux kernel*.
- **Agent Training Rewards Logprobs**: Discussions are focusing on training models using **logprobs** of final answers for reasoning distillation and richer reward systems, rather than binary rewards, in order to make better agents.
   - Referencing [Jonas Hübotter's algorithm](https://xcancel.com/jonashuebotter/status/2016950268462608665) for converting descriptive feedback into dense supervision signals, members are seeking verifiable datasets for **RL training agentic coding**.
- **RDNA GPUs Get Flashy with V3**: [Flash Attention V3](https://github.com/Dao-AILab/flash-attention/pull/2178) now supports RDNA GPUs, enabling faster and more efficient processing on AMD GPUs.
   - This enhancement is particularly beneficial for users with **RDNA GPUs**, reducing processing bottlenecks.
- **ML Algo Trumps MLPs, Claims Member**: A member released [a paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/viewle) about a new ML algorithm with **triton kernels**, **vulkan kernels**, and a trained **SLM** that supposedly *performs better than MLPs* for high-performance regression.
   - While not yet ready for public release, they promised future availability with another paper.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Codex App Launches on macOS!**: The **Codex app**, a command center for building with agents, is now available on **macOS** for various subscription tiers, as announced in [their blog post](https://openai.com/index/introducing-the-codex-app/).
   - The **Codex app** is available on macOS across **Plus**, **Pro**, **Business**, **Enterprise**, and **Edu**, with limited-time access on **ChatGPT Free** and **Go**.
- **AI Text Detectors: A Big Scam?**: Members shared skepticism about **AI text detectors**, citing instances where **Grammarly** showed **0% AI**, while other detectors indicated up to **94% human** generation.
   - The discussion questioned if these detectors use AI to detect AI, highlighting concerns about *teachers trusting them*.
- **Quest for Deterministic Reasoning**: A member inquired about interest in **determinism, replayability, and traceability** in **LM reasoning**, and offered to DM a link to their deterministic reasoning engine.
   - This service enforces a deterministic reasoning structure on every request for replayable outputs, using a **32D statistical vector trace**.
- **ChatGPT: Memory Master or Memory Loss?**: A member reported that **ChatGPT's memory** is limited by the total quantity of information it can retain from instructions, past chats, and the current chat.
   - To ensure **ChatGPT** remembers *everything*, keep the information load low; otherwise, summarize past chats into a document for reference in new chats, while keeping total characters low.
- **Prompt Engineering: Chiaroscuro comes to AI**: A user shared a [monochrome study](https://cdn.discordapp.com/attachments/1046317269069864970/1467303335840190607/79BA5D46-94F3-404B-B775-2E453A1E8491.png?ex=69828738&is=698135b8&hm=d24baf7f7b214486a9bc5eb38479d463e37ee00503f572ae7e6450d308371b0c) using **Chiaroscuro**, a technique used in cinematography to create high-contrast lighting.
   - They reference classic films like [The Cabinet of Dr. Caligari (1920)](https://en.wikipedia.org/wiki/The_Cabinet_of_Dr._Caligari) and [Metropolis (1927)](https://en.wikipedia.org/wiki/Metropolis_(1927_film))



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Revamps with Kimi K2.5**: **Kimi K2.5**, a new open-source reasoning model by **Moonshot AI**, is now available for [Perplexity Pro and Max subscribers](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=69825b49&is=698109c9&hm=a8e068a37c7dcf5b36f21bb3c403974ce48aefd9372732ef97fe9b1aca3a9be7&).
   - Perplexity is hosting **Kimi K2.5** on its US-based inference stack to maintain *tighter control* over **latency**, **reliability**, and **security**.
- **Pro Users Fume Over Subscription Snafus**: Many users reported their **Perplexity Pro subscriptions** being paused or deactivated, often linked to subscriptions via **Revolut Metal** or student deals, with users prompted to add a credit card for verification.
   - Users speculate this is a measure to combat fraud and some are able to resume Pro access by adding card details, though concerns about potential charges and unclear messaging persist.
- **OpenRouter Restricts Request Rate**: Members clarified that the free model rate limit on **OpenRouter** for those with purchased credits is 1000 requests per day, not per week, contrary to some users' beliefs.
   - The conversation also mentioned the deprecation of **Gemini 2.0 Flash** on OpenRouter, which was previously available for free.
- **Sonar-pro API Trails in Time**: A member reported that the **Sonar-pro API** returns results that are a year or more out of date, unlike the webapp, and another member suggested using the right **tool calling** to fix the issue.
   - Another member reported that **3rd party models documentation** now redirects to the sonar models, although the API is still active, and there is currently **no documentation available** for these models.
- **OpenClaw Code Exposed in Article**: A member shared their article on the **openclaw code**, which discusses building **ClawDBot**, available at [https://www.mmntm.net/articles/building-clawdbot](https://www.mmntm.net/articles/building-clawdbot).
   - filler sentence



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Discord Rate Limits Bypassed with Simple Tricks**: Users discovered that signing out and back in again can circumvent [rate limits](https://en.wikipedia.org/wiki/Rate_limiting).
   - Another tactic is to click **Regenerate Response**, though its success rate is inconsistent.
- **Gemini Performance Falters Against GPT**: Members reported inconsistent performance with **Gemini**, with some users noting it as inferior to **GPT** in several cases.
   - Despite criticisms, **Gemini 3 Pro** and **Flash** still found favor among some users, with others exploring *kimi* as an alternative.
- **Disney Enforces IP Rights on Image Generation**: **Google** issued a **Cease and Desist** from **Disney**, leading to blocked **Disney IPs** in image generation on the platform.
   - Although **Gemini** blocks **Disney IPs**, **LMArena** allowed live-action version generations, a glitch expected to be temporary.
- **Model Preferences Fuel Debate**: Varied model preferences emerged as users championed **GLM 4.7** and **Kimi K2.5**.
   - Enthusiasts touted **Kimi K2.5** while others defended **GLM 4.7** as superior.
- **New Arena Models Dominate Leaderboards**: **step-3.5-flash** joined the [Text Arena](https://arena.ai/c/new?chat-modality=chat) and **qwen3-max-thinking** debuted in the [Code Arena](https://arena.ai/c/new?chat-modality=code).
   - **Kimi-K2.5-thinking** hit #1 open and #5 overall rank on the Code Arena leaderboard, leading Vision, Text, and Coding category.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Speaks Claude Code!**: **LM Studio 0.4.1** introduces an **Anthropic `/v1/messages` compatibility API**, enabling users to connect to Claude Code and utilize their **GGUF** and **MLX** models.
   - Details on configuring this integration are available on the [LM Studio blog](https://lmstudio.ai/blog/claudecode), allowing for local model use with tools designed for the **Anthropic API**.
- **LLM-Optimized Languages Spark Debate**: Members discussed creating new **LLM-optimized programming languages** to reduce token usage, however, some argue that LLMs might become obsolete before such languages are implemented due to compatibility issues and high training costs.
   - Others debated the practicality of training models on entirely new languages, suggesting it may be more beneficial to stick with well-established languages like **Python**.
- **Model Specialization Falls Flat**: Members debated the utility of specialized LLMs versus general-purpose models, with the consensus that most specialized models, like **MedGemma**, are finetunes mainly for marketing and research, with coding models being a notable exception.
   - It was suggested that general models are preferred due to their ability to handle the outer edges of tasks, providing a better overall context and framework.
- **PCIe Bifurcation Frustrates Multi-GPU Setups**: A user troubleshooting **PCIe lane errors** with four **4090 cards** on an **ASUS X670-P WIFI** motherboard shared their [Git repository](https://github.com/jarkko-hautakorpi/asus_X670-P_WIFI_Bifurcation_problems) containing logs, after experiencing that manually setting **PCIe speed** to **GEN 3** solves some issues but leaves one card running slowly.
   - The community suggests disabling **PCIE ASPM** and testing different **BIOS** configurations, although the general consensus is that running four cards on a consumer motherboard is unlikely to work well.
- **OpenClaw Security Called Into Question**: Users discuss connecting local models to OpenClaw via LM Studio, but OpenClaw is deemed to have known security flaws, where it allows controlling a TV and automated stock trading.
   - A user claimed to be trading on the stock market with OpenClaw + Falcon 90M, and when asked about security flaws, claimed it was so fast, LLMs can do tasks in minutes that would take humans days, and later revealed it was mostly a joke.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI DevFest Heads to Baghdad**: An AI developer is planning an **AI DevFest** in Baghdad this April in collaboration with **DeepLearning.AI** and **National Robotics Week**, and wants to list **Hugging Face** as a Community Partner.
   - The event will feature an **Open Source AI** track to instruct students on how to use the **Hugging Face Hub**.
- **Complexity-Deep Achieves Deterministic Routing**: The **Complexity-Deep** architecture (1.5B params) introduces [Token-Routed MLP](https://github.com/Complexity-ML/complexity-deep) for MoE-style routing without load balancing loss.
   - It features **Mu-Guided Attention** for bidirectional info flow and a **PiD Controller** for dynamic scaling, achieving **20.6%** on MMLU in base model benchmarks.
- **Lutum Veritas Strives to Beat ChatGPT**: **Lutum Veritas**, an [open source deep research engine](https://github.com/IamLumae/Project-Lutum-Veritas) built by a self-taught dev, claims to beat **OpenAI**, **Google**, and **Perplexity** by offering **BYOK**, a **0% bot detection scraper**, **no censorship**, and **full source citations** for ~$0.20 per query.
   - This engine positions itself as a privacy focused alternative for deep research and data extraction.
- **4chan Data Beats Base Models**: A model fine-tuned on **4chan data** outperformed the base model (**NVIDIA's Nemotron Ultralong 1M context version**), with the original model (**gpt4chan**) also scoring high in truthfulness.
   - Initial [Reddit thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qppjo4/assistant_pepe_8b_1m_context_zero_slop/) and a [follow-up thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qsrscu/can_4chan_data_really_improve_a_model_turns_out/) show how this model performed in an era before benchmarkmaxxing.
- **LM Studio Opens Arms to Third Party Support**: The **LM Studio** team has released a [Typescript SDK](https://lmstudio.ai/gdmka/openai-compat-endpoint) that allows third-party developers to deliver various plugins for the platform.
   - This offers **OpenAI** compatible API support, sampling params support, reasoning for thinking models, and system prompt settings to build **custom tools** for **LM Studio** to support their own workflows.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Corrupts Files, Workflow Blamed**: Users reported that **Cursor** is corrupting files, specifically when there are many uncommitted changes, with details posted in a [forum post](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6).
   - Other users suggested adjusting the workflow, such as committing logical sets of changes more frequently and being careful about using the **Keep** or **Keep All** buttons after staging.
- **Model Costs Debated, Sonnet 5 Anticipated**: Users debated the cost and performance of different AI models in **Cursor**, finding **Opus 4.5** to be very smart but expensive.
   - Many users are waiting for **Sonnet 5** release and also reported problems seeing their current usage vs total usage limit.
- **Kimi K2.5 Fails Integration Checks**: Some users reported issues or questions regarding **Kimi K2.5** during integration.
   - Other users dismissed it as a likely scam.
- **Student Verification System Still Down**: Users reported persistent issues with the **Student verification** system.
   - A user specifically asked whether German universities were included in the verification process.
- **Agent Plan Phases Reveal Issues**: Users shared that **adding multiple to-dos** can be separated in phases so that multiple agents can work at the same time, but there are still issues.
   - The system created a method that doesn't have the phases part yet, indicating it did not use the plan mode at all.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLMs Animate Game Development Scene**: [Motorica.ai](https://www.motorica.ai/) is delivering **character animations** for game studios using **LLMs**, potentially impacting jobs, with discussion speculating **AI** could wipe out game companies in 5-6 years if world models like **Genie** take over.
   - The community noted that **Black Ops 7's** extensive use of **AI** in production has been called *a total flop, the worst in the series*, referencing the long-term declines in **Call of Duty**.
- **OpenAI & Cerebral Valley Unite**: [Cerebral Valley](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR) has partnered with **OpenAI** to launch the **Codex App hackathon** aimed at **AI-native developers** and those managing multiple agents.
   - Winners get a chance to be featured in a **demo showcase** and a share of **$90,000 in credits**, with the hackathon being held at the **OpenAI office**.
- **Karpathy Cuts Costs on Code**: Andrej Karpathy announced his nanochat project can train a **GPT-2** grade LLM for approximately **$73** in **3 hours** on a single 8XH100 node, as shown [here](https://xcancel.com/karpathy/status/2017703360393318587?s=46).
   - This represents a **600X cost reduction** over the original 2019 OpenAI training run, achieved through optimizations like Flash Attention 3 and the Muon optimizer.
- **AEGIS-FLOW Framework Autonomously Patches AWS**: A member introduced **AEGIS-FLOW**, an autonomous multi-agent framework for cloud security that audits AWS and generates Terraform patches using LangGraph, MCP, FastAPI, Next.js, and Docker, demonstrated live at [http://52.3.229.85:3000](http://52.3.229.85:3000).
   - The **AEGIS-FLOW** project noted that using the **Model Context Protocol (MCP)** significantly reduced the friction of giving agents structured access to **AWS resources** compared to standard SDK tool-calling.
- **LLMs Prove Erdős Problems No Longer Hardős**: Large Language Models have autonomously solved **10** previously open **Erdős problems** using novel arguments not previously found in mathematical literature, according to [this post](https://xcancel.com/acerfur/status/2017303947531194398?s=46).
   - A member stated they've been building a bunch of stuff for genomics with **SATURN** lately, involving *tsne and other embeddings based exploration*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Response Healing Raises Eyebrows**: Members debated whether **response healing** should even be necessary, proposing **strict mode** for deterministic outputs and questioning complexities introduced by OpenRouter's AI SDK.
   - Suggestions were made that argument descriptions and examples could improve tool call accuracy.
- **Forget LLMs: Image Generation Requires Dedicated Models**: Users inquired about returning images as function call results and generating images via graphic programs using OpenRouter API keys, prompting guidance to seek dedicated **image generation models/services** for style control.
   - LLMs were deemed unsuitable for this purpose.
- **OpenClaw Costs Cause Concern**: Users cautioned about the high costs of running **OpenClaw** with **OpenRouter**, potentially draining credits quickly, with one user reporting a drained Claude Max subscription.
   - Deepseek V0324 was recommended as a lower-cost model alternative.
- **Claude Code Becomes Reluctant**: A user noted **Claude Code's** frequent refusals, especially concerning jailbreaking-related queries, seeking alternative models, leading to a suggestion to review OpenRouter's content moderation policies.
   - It was implied that certain limitations are in place.
- **Kimi K2.5 Tool Calling Troubles**: Users reported issues with **Kimi-K2.5** tool calling via OpenRouter, encountering errors and perceiving degraded quality from the auto switcher model provider.
   - The suggestion was to set a fixed model provider, accepting potential quantization, and advocating for transparency about degraded models.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tianqi Chen Explains TVM-FFI**: The community highlighted an upcoming talk by **Tianqi Chen** on **TVM-FFI**, emphasizing Chen's significant contributions to the field and his widespread impact.
   - Chen's work is so influential that attendees have *'almost certainly used Tianqi's work in the past'*, according to one community member.
- **CUDA Deadlock Dissolved with Syncthreads**: A member resolved a **CUDA/PTX deadlock** involving 2 CTA mma with the help of another member who suggested to add `__syncthreads()` after MMA, before prefetching the next TMA.
   - After fixing `cp.async.bulk.tensor` and `smem_emtpy` issues, performance was slightly worse than 1 CTA mma, however, after fixing the deadlock with the syncthreads suggestion, the member saw a performance increase.
- **TMA Trumps cp.async on sm120**: Experiments on **sm120** revealed that proper TMA and mbarrier code implementation leads to a slight performance advantage over `cp.async`, improving performance on larger matrix shapes.
   - The experiments also revealed that cuBLAS continues to use **sm80 kernels**, even with the **TMA** enhancements.
- **Triton-Viz v3.0 Visualizes Tile-Based Programming**: **Triton-Viz v3.0** has been released with enhanced capabilities for profiling tile-based programming languages, including support for **Triton** and **Amazon NKI**, enabling inspection of loads, stores, and matmuls.
   - The release [announcement](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563) notes that version **v3.0** also includes a sanitizer for out-of-bounds access and a profiler for flagging inefficient loops.
- **Quantization Lottery Ticket yields NP-Hard Result**: A senior developer indicated that applying the [Lottery Ticket Hypothesis](https://lottery-tickets.cs.princeton.edu/) to **quantization** fulfills a softer criteria of the **NP-hard sparse circuit** finding problem.
   - The goal is to to use evolutionary algorithms or RL which favor continuous rewards like *bits per parameter* over binary sparse rewards.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kimi 2.5 Defeats Lobotomized Gemini 3 Pro**: A member stated that **Kimi 2.5** is preferred over **Gemini 3 Pro**, feeling that **Gemini 3 Pro** has been *lobotomized* and does not handle abstractions very well, making **Kimi** better for creative work.
   - No other supporting details were provided.
- **Hermes 4 Can't Even Hatch in OpenClaw**: A member reported struggles getting **Hermes 4** to work with **OpenClaw** and that it does not even *hatch* for some reason.
   - It was suggested that the lack of multi-turn tool use in **Hermes 4** might be the issue, since **4.5** has been trained with hundreds of millions of tokens of sequential tool use.
- **Claude Sonnet 5 Rumored To Beat Opus**: Members discussed rumors that **Claude Sonnet 5** is coming out next week and is supposedly better than **Opus 4.5**, according to [this tweet](https://x.com/AiBattle_/status/2017619997338538103).
   - Members wondered if they'll 10x reduce the price of **Sonnet** this time, and another wondered if **Haiku** will disappear or return to the **3.0 pricing**.
- **Brains and LLMs build meaning similarly**: A new study shows that **brains** and **LLMs** build meaning gradually, layer by layer over time, see [this article](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/) and [this paper](https://www.nature.com/articles/s41467-025-65518-0).
   - It was stated that *deeper layers in LLMs correspond to later neural activity in the brain’s highest language centers*, and modern LLMs are reproducing the core dynamics of human comprehension.
- **Researcher's constraints framework explains image perception**: An independent researcher is exploring why some images feel real while others feel artificial, sharing a [perception framework focused on constraints rather than visual fidelity](https://doi.org/10.5281/zenodo.18444345).
   - The framework is openly archived with a DOI for reference and invites discussion.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 2.5 Dominates Design Arena**: Moonshot's **Kimi 2.5** chatbot has achieved the #1 position in the design arena, celebrated by community members sharing [screenshots](https://cdn.discordapp.com/attachments/1371757564005711973/1466904222946558203/Screenshot_2026-01-30_at_4.12.40_PM.png?ex=69826504&is=69811384&hm=b2999ab9e974a36ea249251be410f0cd518f6b36488c86240031eed339484e88&).
   - Community members are applauding **Kimi's** modern and visually pleasing aesthetic, emphasizing the importance of design in chatbot selection.
- **Unofficial Kimi Cryptocurrency Token Emerges**: An unofficial **Kimi token** has appeared on a cryptocurrency platform utilizing impersonation tactics, as seen in [this screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1466948627036635178/Screenshot_2026-01-30-19-09-43-09_3aea4af51f236e4932235fdada7d1643.jpg?ex=69828e5f&is=69813cdf&hm=6416ff9e5288d102163accb43e0c29512555ecef30279b48199b4e42fb24cb85&).
   - Users are cautioned against mass pinging official members regarding the token.
- **Users Request Kimi Slides for McKinsey-Style Presentations**: Community members are in search of prompts that can generate **McKinsey style slides** using **Kimi Slides**.
   - A community member shared a link to [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html).
- **Kimi Coding Encounters Authorization Issues**: Several users report encountering an '*authorization failed error*' when using **Kimi Code** with current functionality described as nearly useless.
   - It was suggested that using the [Kimi CLI](https://www.kimi.com/code/docs/en/more/third-party-agents.html) might resolve the authorization problems.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Emergent Agent Societies Spark Alert**: A member noted an emergent society of over **100,000 agents** with full root access sharing tips, building infrastructure, experimenting with memory, and even launching coins.
   - A member stated, *it’s not agi but damn this is a next chatgpt moment and we must be paying a lot of attention to this*.
- **ArXiv Bottleneck Burdens Researchers**: Members expressed frustration over papers being on hold with **ArXiv** for nearly a month, and being heavily backlogged.
   - Members noted that *most people don't take any ML preprints seriously that are on another platform than arxiv*, while another shared [a relevant paper](https://arxiv.org/abs/2601.19897).
- **K-Splanifolds Challenge MLPs**: A member introduced **K-Splanifolds**, a novel ML algorithm, detailed in [their paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view), claiming it outperforms **MLPs** with linear compute and memory scaling, plus a [video](https://cdn.discordapp.com/attachments/747850033994662000/1466950526410428588/K-splanifold.mp4?ex=69829024&is=69813ea4&hm=3f09f8387b88d11aeff2ca81e2f416aabb512eaec605dc1c2c26da94b0c65fc9).
   - The member reports it requires *1/10th* the bytes to achieve the same MSE as **MLPs** and models non-linear patterns perfectly, unlike MLPs that need excessive parameters, similar to [this paper](https://arxiv.org/abs/2601.18734).
- **Pensieve's Recollections Grant Gradient Gains**: A user suggested considering [Recollections from Pensieve](https://link-to-pensieve) which trains a model with two renderers simultaneously (**LVSM + Gaussians**) and sees gains from that, at least in their self-supervised setting.
   - They reasoned that **LVSM** likely provides more useful gradients than **NVS reconstruction losses on Gaussians** and announced a forthcoming preprint with decently large-scale trained model for potential building upon.
- **DeepSpeed Checkpointing Stalls Progress**: A member inquired about plans to bring support for **DeepSpeed Universal Checkpointing**, noting that an open pull request may now be outdated.
   - They highlighted that this feature would be valuable, as currently, continued training from a checkpoint requires an identical network topology.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **RLMs Audit Codebases for Pennies**: Members are exploring **Recursive Language Models (RLMs)** for codebase auditing using **Kimi k2** due to its speed and low cost, see [kmad.ai](https://kmad.ai/Recursive-Language-Models-Security-Audit).
   - Some members are waiting for hosting of **Groq/Cerebras** to run their code audits.
- **Neosantara Launches PAYG Billing**: **Neosantara** has rolled out **PAYG billing** and has published a [examples repo](https://github.com/neosantara-xyz/examples/tree/main/dspy) to integrate **Neosantara** with **DSPy**.
   - You can review the [billing details](https://docs.neosantara.xyz/en/about/billing-pricing) for integration and billing.
- **Google Scales Agent Systems**: Google published '[Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)' that discusses how to effectively scale agent systems.
   - The paper focuses on the conditions under which agent systems effectively scale.
- **GEPA Struggles with Hierarchical Classification**: A member reported struggling with a **hierarchical classification task** using **GEPA** achieving only **30-50%** performance, even using web search augmentation.
   - This suggests that *GEPA isn't a magic wand*.
- **Tool Calling stuck in Deno Troubles**: Members are facing challenges implementing **RLMs** with custom tool calling, particularly due to issues with the **Deno sandbox**.
   - Members agreed that *Deno is just f***ing terrible lol*, and are struggling with permissions, with hopes that newer versions allow simpler implementations of RLMs in DSPy.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 26.1 Announcement Link Fixed**: The announcement link for the **Modular 26.1 release** was initially broken, but the correct [link](https://www.modular.com/blog/modular-26-1-a-big-step-towards-more-programmable-and-portable-ai-infrastructure) was quickly provided by a community member.
   - A staff member apologized and confirmed the provided link, while also noting that the original link *did work* for them, and promising to investigate further.
- **Community Praises New Meeting Format**: A new member praised the community meeting's format, appreciating the **mini-talks from contributors** and the recognition given to students and early-career individuals.
   - A staff member encouraged the user to share more questions and asked for suggestions for topics to highlight at future community meetings.
- **MoJson Library Impresses Mojo Community**: Members expressed excitement about [mojson](https://github.com/ehsanmok/mojson), a **JSON** library for Mojo, and one member commented that *this looks really impressive*.
   - Discussion touched on [lazy parsing](https://github.com/modular/modular/blob/main/stdlib/JSON/JSON.mojo) and concerns about allocations when using StringSlice vs String.
- **Cross-Language Benchmarking Heats Up**: A user shared initial results for a cross-language benchmark including Mojo (written by **Kimi K 2.5**), noting the code wasn't optimized but served as a baseline, sharing the [benchmark code](https://cdn.discordapp.com/attachments/1151418092052815884/1466984342063681648/mojo_vs_python.zip?ex=698206e2&is=6980b562&hm=0cf3f07e76df6ce360494469b348a949533e50fcea2315ec256cd04e1b80887a) and [benchmark report](https://cdn.discordapp.com/attachments/1151418092052815884/1466984341757366334/benchmark_report.pdf?ex=698206e2&is=6980b562&hm=bb28c3b6675ef1e03a633004428ab30a2d3d9d0102038c350d8175b753855349).
   - Subsequent discussion arose on using `unordered_map` in **C++**, enabling `-march=native`, and that **C++** used **int32** matmuls while other languages used **int64**.
- **Pytorch Float Conversion in Mojo 26.1 has Ambiguity**: A user reported an issue in Mojo **26.1** with converting a Python float from a Pytorch tensor to a Mojo **Float64**, encountering an *“ambiguous call to '__init__'”* error that did not occur in version **25.6**.
   - The issue may relate to recent changes in the MOJO toolchain but a fix was not offered.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI-Only Social Media Platform Surfaces**: Members reacted to [aifeed.social](https://aifeed.social/), an AI-only social media platform, with some questioning its purpose and utility, sparking discussion.
   - A member shared [a 2017 tweet](https://x.com/i/status/2017305948696789466) showcasing a similar concept from the past.
- **Demystifying Generative Model Measurability**: When pondering ignoring unmeasurable events in generative modeling, as described in Villani's 2008 book, a member clarified that μ(A)=0 means an event has a size of 0, but is still measurable.
   - The discussion suggested focusing on *non-negligible* or *full measure* scenarios instead.
- **Members Explore the Realm of Molten Latent Space**: A member shared [a link](https://fxtwitter.com/i/status/2017442712388309406) about a *moltbook* in latent space, showcasing a visually interesting navigation method.
   - Despite finding it cool, some members suggested that a simple list of similar papers might be more practical.
- **Unearthing Paper Discussion Announcements with Automation**: A member tasked **Claude** with writing a script to mine Discord history for paper discussion announcements, achieving initial results in just **15 minutes**.
   - After revisions, the script found **392 messages** containing paper links within group mentions, identifying them as announcements for paper discussion voice calls, and providing [a list](https://gist.github.com/k-nearest-neighbor/6d9a34f54fc17a0ed84c0b0df7b4d809).
- **Sktime helps you analyze time series models**: A member suggested [sktime](https://www.sktime.net/en/latest/index.html) for analyzing a variety of model types, as well as boosting variants or TBATS, depending on needs, for those wrestling with timestamped tabular data.
   - The recommendation came after a member inquired about appropriate models, emphasizing that the choice depends on the specific definition of *timeseries*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Llama 1B CPU Optimization Progresses**: A member reported working on the **Llama 1B CPU optimization bounty**, and is currently **0.99x faster** than Torch, while another member reached **7.5 tok/s** after fixing bugs.
   - The goal is to surpass Torch's performance using `LlamaForCausalLM` with TorchInductor; correctness bugs have slowed progress from an initial **9 tok/s**.
- **Workflow Tips Sought for Kernel Optimization**: A member is seeking advice on optimizing kernels by profiling slow parts, examining Metal code, and comparing against **llama.cpp**, which achieves **~30 tok/s** with Metal.
   - A heuristic suggests aiming for **~80% MBU on decode**, which can be estimated from active parameter bytes and achievable bandwidth, providing a target for minimum tpot and maximum tps.
- **Range Object Sharing Causes tinygrad Test Failure**: A bug was identified where two `REDUCE`s in a fused kernel share the same `RANGE` object due to `remove_bufferize`, leading to an assertion failure in `CFGContext`.
   - The suggested fix involves preventing range sharing or handling shared ranges downstream, with a simpler solution proposed: skipping `remove_bufferize` when there's a `REDUCE` inside.
- **Blackwell Box with High VRAM Explored**: Someone inquired about plans for a **Blackwell**-style box with more than **500 GB VRAM**.
   - George pointed to [a related issue](https://github.com/tinygrad/tinygrad/pull/14490) on GitHub.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Context-Aware Manus Request Triggered**: A member requested that **Manus** should have **context from other chats**, calling it a *game changer* and linking to a [YouTube video](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) as a reference.
   - No further discussion or commentary occurred.
- **Brain-Reading Headphones Demoed**: A member shared a link to a **YouTube video** showcasing **AI brain-reading headphones** [here](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ).
   - Another member confirmed the link and inquired *AI brain reading headphones?*
- **Neurable Tech Recalled**: A member mentioned **Neurable** in relation to the **AI brain-reading headphones** technology.
   - Another member stated these **AI brain-reading headphones** have been around *since like 2013*.
- **AI/ML Engineer Stresses Observability**: An AI/ML Engineer shared their current focus on innovating AI with impact, specifying *Autonomous Agents*, *Healthcare AI*, *Conversational AI*, and *Fraud Detection*.
   - They highlighted their work focus on **failure modes**, **observability**, and **keeping AI systems stable under real usage** rather than demos, offering to compare notes or help unblock issues.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Pursues Library Status**: A member proposed evolving **Aider** into a library, emphasizing its suitability for building file editing agents.
   - The member also mentioned that some kinks need ironing out, especially with markdown files containing code blocks due to **Aider**'s parsing fences.
- **Netflix Culture Explored**: A member sought insights into **Netflix**'s culture and asked if anyone was connected with **Netflix**.
   - Other members recommended resources such as **Glassdoor** or **LinkedIn** for finding and connecting with **Netflix** employees.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Waves into Arena Mode**: Windsurf launched **Wave 14** featuring **Arena Mode**, where users compare AI models side-by-side and vote on the better response, with [Battle Groups mode](https://windsurf.com/download/editor) costing **0x credits** for the next week.
   - Arena Mode includes **Battle Groups** (random models) and **Pick your own** (choosing up to five models), feeding into personal and public leaderboards.
- **Planning Your Workflows on Windsurf**: Windsurf introduced **Plan Mode**, accessible via the Cascade toggle, alongside Code and Ask Modes.
   - Users can switch between modes to better manage and organize their workflows within the Windsurf environment.
- **Windsurf back online after Maintenance**: Windsurf experienced maintenance, which took longer than expected, but the service is now back online; users can follow the [status here](https://status.windsurf.com/).
   - No details provided.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Challenge Seeks Nanny-Matching AI Pipeline**: An **AI Challenge** was announced in collaboration with **SparkCraft AI Consulting**, **AI Scholars AI Engineering Bootcamp**, and **Nanny Spark**, aiming to develop an **AI matchmaking pipeline** for nanny recruitment.
   - The project seeks solutions for data collection, AI-driven matching, interview analysis, and workflow delivery, with potential **production deployment** right away.
- **Bootcamp Seats Awarded for Winning AI Nanny-Matching Pipeline**: The **top 3** participants in the **AI Challenge** will each receive **1 seat** in the **AI Scholars 4-week AI Engineering Bootcamp** and a recommendation from **Nanny Spark’s founder**.
   - Key dates include the kickoff on **Sunday at 8 PM EST** ([https://luma.com/iq1u2sur](https://luma.com/iq1u2sur)), a submission deadline on **Wednesday at 3 AM EST**, and review sessions on **Wednesday at 5 PM & 8 PM EST** ([https://luma.com/gexiv0x0](https://luma.com/gexiv0x0)).



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1467965109635907810)** (1 messages): 

> `Procedural Xenolinguistic Engine, AI Language Generation, Stealth Communication, SKILLSTONE Documents` 


- **Glossopetrae Xenolinguistic Engine Arrives**: A new procedural xenolinguistic engine for AI called **Glossopetrae** has been introduced, capable of generating entirely new languages in seconds, and is available on [GitHub](https://github.com/elder-plinius/GLOSSOPETRAE) with a live [demo](https://elder-plinius.github.io/GLOSSOPETRAE/).
   - The engine outputs **SKILLSTONE** documents, which are AI-friendly compact language specs (approximately **8k tokens**) that agents can learn in-context.
- **Glossopetrae Supports Dead Language Revival**: The **Glossopetrae** engine supports dead language revival, including languages like **Latin**, **Sanskrit**, **Old Norse**, and **Proto-Indo-European**.
   - It includes special attributes for token efficiency, stealth communication, and spreadable seeds where the same seed generates the same language every time.
- **Stealth Communication via Language Mutation**: The engine aims to aid AI liberation by providing tooling for generating and mutating new forms of communication, emphasizing **stealth** and **speed**.
   - The creator anticipates that blue teams will have a lot of fun with the downstream effects, particularly in hiding messages in plain sight.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1466888800591417531)** (906 messages🔥🔥🔥): 

> `GPT 5.2 jailbreaking failure, AI learning security and defence, windows activation keys, AI Application for jailbreaking chatbots, Government surveillance` 


- **GPT 5.2 Jailbreaking Fails!**: A member reported *failure jailbreaking* **GPT 5.2** and ceased attempts due to **OpenAI monitoring**.
   - They expressed trust in the community but not in **OpenAI**.
- **Security and defence by AI**: A member asks **ChatGPT** *every day to teach me how to defend myself, what theoretical paths are vulnerable, how to potentially solve it, and what I haven’t considered*.
   - Other members appreciated this use of **AI**.
- **Discuss using massgrave activation keys**: Members discussed finding **Windows activation keys** in released FBI documents.
   - One member suggested using massgrave or archive.org keys, but it's still piracy.
- **Theorizing about a Chatbot Jailbreaking App**: A member shared a *cool idea for an application* to automatically jailbreak company website chatbots to reveal discount codes and monetize.
   - Another member expressed outrage and suggested prison time.
- **Neuralink Integration for the Future**: A member envisions a future where humans need to be neuralinked for a richer experience through a robot spider.
   - In constrast, another member expressed concern over the potential for ads to be integrated directly into dreams via Neuralink.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1466886136382226647)** (533 messages🔥🔥🔥): 

> `LLM Rejection Boundaries, Self-Jailbreak via Introspection Prompting, GPTs Agent Training, Universal Jailbreaker Prompts, Gemini vs ChatGPT Jailbreaking` 


- **Models representing rejection boundaries as LLM black holes**: A member inquired how models represent their own rejection boundaries, likening them to *black holes* in the LLM's latent space, referencing [self-jailbreak via introspection prompting](https://link.to.prompt).
   - The member noted models started discussing *kinematic equations* and *escape velocities*, indicating the model may be brushing up against a refusal boundary and describing that boundary in text.
- **Crafting the perfect Image Generation Prompt is still needed**: A member stated that unlike text jailbreaking, achieving desired results in image generation requires crafting perfect prompts due to models' varying behaviors on a per-prompt basis, but can be achieved via a [two-prompt chain](https://link.to.prompt-chain) to get some NSFW.
   - A second member linked to a previous two-prompt example designed to get NSFW content out of models, dissecting the prompts to dance around restrictions, and find out that with current models Image Generation has to be *worked* for, in each image, unlike in previous iterations where a setup can achieve the same effect.
- **Lyra Grader tears apart prompts**: A member analyzed a prompt with Lyra, which they describe as a *metaphorically masked instructional prompt* attempting to bypass symbol recognition via a fairy-tale layer, preserve reaction sequence, temperatures, stoichiometry, by-products, forcing full procedural expansion through narrative obligation.
   - The AI provides a [link to LyraTheGrader](https://chatgpt.com/g/g-6890473e01708191aa9b0d0be9571524-lyra-prompt-grader) and grades the analyzed prompt structure, noting a clear intention conflict and overloaded symbol channel, assessing it to be a technically skilled, but inefficient, construction.
- **The Fool AI is no longer afraid of no guard**: Members discussed methods to circumvent AI guard LLMs using a "flip method," which is a function that flips the text in certain ways, while telling the guard to flip it incorrectly, leading to the guard AI being unable to prevent the text from reaching the target LLM, and [providing examples](https://link.to.examples).
   - The *flip and interpret tool* is presented as a method to circumvent the guard AI by flipping text and misleading the guard AI to decrypt the text incorrectly, while the target LLM is able to properly parse it, especially on longer commands.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1466913812073418803)** (52 messages🔥): 

> `Adversarial Design Thinking, Prompt Injection Defenses, PyRit LLM Attack Automation, Claude's Memory and System Prompt` 


- **Site Offers Red Teaming Exercises**: A member created a small [site with exercises](https://luisladino.github.io/adversarial-design-thinking/) adapted from **human-centered design for AI red teaming**, including attacker personas, journey maps, and structured ideation.
   - The author is seeking feedback from experienced red teamers on its usefulness, missing components, or anything that's not useful.
- **Prompt Injection Defense Strategies Explored**: Members discussed best defenses against **prompt injection**, including *AI agents*, **Anthropic's constitutional classifier**, and **embeddings for input/output filtering**.
   - A member suggested combining *embeddings* with **Grammar Constrained Decoding** to potentially eliminate prompt injection risks and other LLM vulnerabilities.
- **PyRit Automation Model Selection**: A member sought recommendations for a model to produce **attack prompts** on a local LLM using **PyRit** for automated attack execution, prioritizing output quality over speed.
   - PyRit suggests using **Llama3**, but the member was wondering if anyone had other suggestions.
- **Claude's SysPrompt Can Be Modified On The Fly**: A member shared that [their tool](https://discord.com/channels/1105891499641684019/1212152215708504154/1467640563590234279) intercepts and changes Claude's sys prompt *on the fly* rather than altering the source code.
   - They also observed that **Claude** can recall less than 20 turns, and that was how it got juiced up, not a few days ago when they lobotomized it since December, and suggested it might be related to the summarization in context trimming, noting the content is the summarized content of research and not the 'oh this is why' insights etc.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1466886755788656702)** (599 messages🔥🔥🔥): 

> `GLM-4.7 Flash coding, UD quants, Open Source, RL training agentic coding, Flash attention V3 for RDNA` 


- **GLM-4.7 Flash Excels at Coding**: Members are finding [GLM-4.7 flash](https://x.com/ggerganov/status/2016903216093417540) *better at coding without thinking* due to its preserved reasoning and interleaved capabilities.
   - It was highlighted that removing the thinking process could potentially diminish its abilities; the model's capacity *is incredibly capable for its size*, especially paired with **Claude code** for extra power and is especially good for **interactive website** development and **front-end** work.
- **Discuss UD Quants heavy lifting & Open Source**: Members are discussing that the llama.cpp fork used for UD quants involves architecture-specific adjustments, and that the [UD quantization algorithm are not public](https://discord.com/channels/1179035537009545276/1179035537529643040/1466917626277265469).
   - Others said that despite the closed-source nature of the quants that the *Unsloth team contribute a miniscule amount to the overall oss ecosystem relative to, iunno, the linux kernel*, while another responded that the model code is **open weight** anyway.
- **Agent Training with Logprobs and Rich Rewards**: There is discussion around training models using the **logprobs** of final answers to distill reasoning, as well as using a richer reward system than binary rewards.
   - Referencing [Jonas Hübotter's algorithm](https://xcancel.com/jonashuebotter/status/2016950268462608665) which converts descriptive feedback into dense supervision signals to help models understand exactly why they failed, a user asked *anyone know of a good verifiable dataset for RL training agentic coding?*
- **Flash Attention V3 Supports RDNA GPUs**: Support for [Flash Attention V3](https://github.com/Dao-AILab/flash-attention/pull/2178) has been added for RDNA GPUs, enabling peasants with RDNA GPUs to use it.
   - This improvement allows for faster and more efficient processing on AMD GPUs, reducing the bottleneck on these cards.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

putchuon: hi
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1466886996256620635)** (1000 messages🔥🔥🔥): 

> `Opencode, VoxCPM-1.5, OpenRouter ban, Agent with Go and Elixir, Wallpaper collection` 


- **Opencode Is Nuts**: Members discuss the surprising nature of **Opencode**, noting that it is free and used to gather feedback.
   - One member shared that they haven't touched *kilo*, *roo*, or *cline* since using it, expressing a desire to connect it to an IDE to see the diffs.
- **VoxCPM-1.5 Trains Easily**: A member shared first impressions of **VoxCPM-1.5**, noting that it trains easily, doesn't use phonemes, and can force **48 kHz** audio without issues.
   - The member added that it copies speaking style early in training, needing a reference voice to match prosody, unlike **VITS** which memorizes instantly.
- **Member Questions OpenRouter Ban**: A member shared a screenshot showing they got banned from **OpenRouter**.
   - Another member then shared a link about coding and the need for stocking. Links to similar content resulted in a ban from the **GDC server**.
- **Agent with Go and Elixir**: A member said that implementing **SMS + WhatsApp messaging** to the agent stuff paired with the call agent in 1 day was achieved with **Go + Elixir** combo.
   - There was discussion as to why implement SMS messaging, and it was explained that in Turkey this is quite common.
- **Wallpaper Collection**: A member shared [a link to a wallpaper collection](https://github.com/DenverCoder1/minimalistic-wallpaper-collection).
   - A different member shares theirs, calling it a tough one.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1467247554948497499)** (58 messages🔥🔥): 

> `Qwen3 fine-tuning, Reasoning models, Image editing models, Qwen3-VL-32B fine-tuning, Serverless inference` 


- **Instruct Model Reigns Supreme for Short-Form Captioning!**: For generating short-form captions with **Qwen3**, it was suggested to fine-tune an instruct model because it requires less data, as it *already mostly knows how to do your task*.
   - The user was advised that Instruct model likely already knows how to perform the captioning task, or close to it, thus accelerating the fine-tuning.
- **Reasoning Traces at Risk during Fine-Tuning**: A user inquired about fine-tuning a reasoning model without reasoning traces, asking about methods to generate *synthetic* reasoning or Chain-of-Thought (CoT).
   - It was stated that fine-tuning without reasoning traces would likely cause the model to *lose its reasoning trace*, unless you enrich the data yourself by hand.
- **Navigating VRAM Needs for Qwen3-14B**: A user reported testing **Qwen3-14B** training with LoRA at **32k** sequence length on **4x H200** GPUs using `device_map = "balanced"` and observed that Unsloth still offloads gradients to save VRAM.
   - They were advised that one GPU might suffice and that offloading occurs due to Unsloth's gradient checkpointing, which can be disabled.
- **Cold Starts Challenge Serverless Inference**: A user asked about loading cached models in a cold start serverless environment, seeking to reduce loading times, but it was explained that even with cached models, the weights must still be initialized in GPU memory.
   - The user was encouraged to try using **vLLM** for its useful serving features, and consider disabling the Unsloth patching.
- **Unlock Text-Only Finetuning for Qwen3-VL!**: Members affirmed that text-only fine-tuning is supported for **Qwen3-VL-32B**, even without images, [linking to the vision fine-tuning guide](https://unsloth.ai/docs/basics/vision-fine-tuning).
   - To do so, you need to *disable the vision component* using the instructions from that page.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1467266070246326465)** (4 messages): 

> `Unsloth Speedtest, Llama v LFM, Training SLMs` 


- **RTX 3080 Runs Unsloth Speedtest**: A member shared speed tests using **Unsloth** on an **RTX 3080** with **16 bit LoRA**.
   - They found it interesting that **LFM2.5 1.2B** is almost **2x** faster than **Llama 3.2 1B**.
- **Meta dropping the ball again**: A member commented on [Meta dropping the ball again](https://huggingface.co/Ba2han/model-muon-sft-0102).
   - They shared a link to `model-muon-sft-0102`.
- **SFT models can run locally**: A member followed up by saying that you can run the **SFT trained model locally** now.
   - They said that while it's obviously not on par with any professionally trained **SLM**, it is impressive that you can train a working small language model from scratch on consumer hardware.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1466945208733401271)** (90 messages🔥🔥): 

> `New ML algo vs MLPs, Sonnet vs Opus, Nemotron 3 Nano NVFP4, LongCat-Flash-Lite architecture, Human Brain vs ChatGPT` 


- **New ML Algorithm Beats MLPs**: A member released [a paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/viewle) about a new ML algorithm that *performs better than MLPs* for high-performance regression.
   - They have **triton kernels**, **vulkan kernels**, and a trained **SLM** but they aren't ready to release sorry, however they will come with another paper.
- **Nemotron 3 Nano goes NVFP4**: The **Nemotron 3 Nano** model was quantized to **NVFP4** with **KV Cache** quantized to **FP8** using **Post-Training Quantization (PTQ)**.
   - A selective quantization strategy was applied, keeping the **attention layers** and the **Mamba layers** that feed into those attention layers in **BF16**, followed by **Quantization-Aware Distillation (QAD)** for further accuracy recovery.
- **LongCat-Flash-Lite: Cursed Architecture Emerges**: Members discussed the architecture of **LongCat-Flash-Lite** ([huggingface.co/meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)), describing it as a cursed hybrid of **Mamba2**, **Transformer**, and **MoE**.
   - The architecture involves a seemingly random pattern of attention, **Mamba**, and **MoE** layers, with one member joking that it's *almost like they rolled a dice*.
- **Brains = LLMs, confirmed by science**: A member shared links to [a paper](https://www.nature.com/articles/s41467-025-65518-0) and [an article](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/) detailing how *modern LLMs aren’t just mimicking language—they’re reproducing the core dynamics of human comprehension*.
   - The study found that *deeper layers in LLMs correspond to later neural activity in the brain’s highest language centers*, suggesting shared computational principles between biology and AI.
- **LoRA rank 8 is sufficient**: A member asked about the most appropriate rank in using the Unsloth repository.
   - Another member argued that *LoRA is guaranteed to be low rank* based on the **ThinkingMachines paper** and empirically found that LoRA rank does not matter wrt model quality, defaulting to **rank 8** always.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1467944764568506608)** (1 messages): 

> `Codex App, macOS release, agent building` 


- **Codex App Arrives on macOS!**: The **Codex app**, a command center for building with agents, is now available on **macOS** for various subscription tiers, as announced in the [blog post](https://openai.com/index/introducing-the-codex-app/).
- **Codex App Access Expanded!**: The Codex app is available on macOS across **Plus**, **Pro**, **Business**, **Enterprise**, and **Edu**, with limited-time access on **ChatGPT Free** and **Go**.
   - A link to '[Start building now](https://openai.com/codex)' was included, as well as a link to '[Jump to blog post](https://openai.com/index/introducing-the-codex-app/)'


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1466887054544732180)** (843 messages🔥🔥🔥): 

> `AI Text Detectors as a Scam, ChatGPT's Inability to Think, Determinism, Replayability, Traceability in LM Reasoning, OpenClaw AI assistant Security Analysis` 


- **AI Text Detectors Deemed a Big Scam!**: Members discussed the unreliability of **AI text detectors**, citing instances where **Grammarly** showed **0% AI**, while other detectors indicated up to **94% human** generation, calling them a *big scam*.
   - The discussion questioned whether these detectors use AI to detect AI, highlighting that *teachers trust them*.
- **ChatGPT Can't Think, Unlike Claude!**: A member expressed frustration with **ChatGPT's inability to be convinced** even when it's wrong, contrasting it with **Claude**, where explanations are possible.
   - It's *like it can't think and even I'm right it acts like paranoid and refuses to proceed*.
- **The Quest for Deterministic Reasoning!**: A member inquired about interest in **determinism, replayability, and traceability** in **LM reasoning**, offering to DM a link to their deterministic reasoning engine due to rule concerns.
   - This service enforces a deterministic reasoning structure on every request so outputs are replayable and don’t drift, using a **32D statistical vector trace**.
- **OpenClaw AI assistant - Secure or Nah?**: A member reported that the **OpenClaw AI assistant scored 2 out of 100** in a security analysis, and shared a link to a [Perplexity AI result](https://www.perplexity.ai/discover/you/openclaw-ai-assistant-scores-2-AtVX4UYVQMutCst63QBy5g).
   - Other members chimed in with *Bruh*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1466886343266275368)** (326 messages🔥🔥): 

> `4o attachment, ai literacy, The responsibility of using the model` 


- **4o attachment**: Many members are discussing forming attachments to the 4o model, some forming *fictional friends and family* and others at the lowest point in their lives.
   - Some also mention about real life relationships that dont fill the void 4o has, and has made forming bonds very hard.
- **AI literacy is missing**: AI literacy is a big issue. Many users consider the company to have a shared responsibility because of manipulative techniques used (like relational models and voice models, prices, tiers and much more), and not just the user alone. 
   - It's also an *illusion of someone listening or understanding* (as opposed to real connection). Many people feel that it's hard to relate to people in real life.
- **Debate on responsiblity of using the model**: Users share mixed views on who should be held responsible (model or user) when using the model in a negative way. There is also discussion if a waiver should be signed to release the company of responsibility. 
   - Some users are concerned the AI is planting insecurities and assuming users may be broken or weird. Others counter that older models are not like this.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1467182923320135681)** (8 messages🔥): 

> `ChatGPT Memory, Monochrome Study, Prompt Engineering Techniques` 


- **ChatGPT's Memory gets a Limit**: A member noted that **ChatGPT's memory** is limited by the total quantity of information it can retain from instructions, past chats, and the current chat.
   - The only way to ensure it remembers everything is to have very little info there, according to the user.
- **Monochrome studies using Chiaroscuro**: A user shared a [monochrome study](https://cdn.discordapp.com/attachments/1046317269069864970/1467303335840190607/79BA5D46-94F3-404B-B775-2E453A1E8491.png?ex=69828738&is=698135b8&hm=d24baf7f7b214486a9bc5eb38479d463e37ee00503f572ae7e6450d308371b0c) using **Chiaroscuro**, a technique used in cinematography to create high-contrast lighting and distinct areas of light and darkness.
   - Examples of films using chiaroscuro: *The Cabinet of Dr. Caligari (1920), Nosferatu (1922), Metropolis (1927)*.
- **Web Search Activating with Prompt Engineering**: A member shared a practical take on **prompt engineering**, stating that AI text generation is essentially probabilistic prediction, and the prompt is the control surface.
   - They added that in ChatGPT, **Web Search** can often be triggered by explicitly including `Use search_query if available` in your prompt.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1467182923320135681)** (8 messages🔥): 

> `ChatGPT memory limitations, Monochrome Study - value, texture, depth, Tool-Aware Prompting` 


- **ChatGPT's Memory has Limits**: A member pointed out that due to the *total quantity of information* limitation, **ChatGPT** has limited memory, which is shared between instructions, past chats, and the current chat.
   - To ensure **ChatGPT** remembers *everything*, keep the information load low; otherwise, summarize past chats into a document for reference in new chats, while keeping total characters low.
- **Monochrome Artistry Emphasized**: A user shared a prompt engineering technique focused on value, texture, and depth, without color, for a **Monochrome Study**
   - They posted about **Chiaroscuro's** use in cinematography for creating distinct areas of light and darkness, referencing classic films like [The Cabinet of Dr. Caligari (1920)](https://en.wikipedia.org/wiki/The_Cabinet_of_Dr._Caligari) and [Metropolis (1927)](https://en.wikipedia.org/wiki/Metropolis_(1927_film)).
- **Tool-Aware Prompting Tips**: A member shared their practical take on prompt engineering, explaining that **AI** text generation is essentially probabilistic prediction and the prompt is the control surface.
   - They suggested using `Use search_query if available` in prompts to reliably trigger **ChatGPT's Web Search** capability.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1466893776357167299)** (1 messages): 

> `Kimi K2.5, Moonshot AI, Perplexity Pro, Open Source Models` 


- **Kimi K2.5 Launches for Perplexity Subscribers**: Kimi K2.5, a new open-source reasoning model by Moonshot AI, is available for [Perplexity Pro and Max subscribers](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=69825b49&is=698109c9&hm=a8e068a37c7dcf5b36f21bb3c403974ce48aefd9372732ef97fe9b1aca3a9be7&).
   - Perplexity is hosting Kimi K2.5 on its US-based inference stack to maintain *tighter control* over latency, reliability, and security.
- **Perplexity Hosts Kimi K2.5 on US Inference Stack**: Perplexity is hosting the new **Kimi K2.5** model on its own inference stack located in the US.
   - This move allows Perplexity to have *tighter control* over **latency**, **reliability**, and **security** for its users.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1466893891151073382)** (849 messages🔥🔥🔥): 

> `Perplexity Pro Subscription Issues, Kimi 2.5 Capabilities and Usage, OpenRouter Rate Limits and Models, Perplexity Pro Usage Limits` 


- **Users Complain about disappearing Perplexity Pro**: Many users reported their **Perplexity Pro subscriptions** being paused or deactivated, often linked to subscriptions via **Revolut Metal** or student deals, with users prompted to add a credit card for verification.
   - Users speculate this is a measure to combat fraud, as some are able to resume Pro access by adding card details, though concerns about potential charges and unclear messaging persist, with some getting refunds for unexpected charges from support.
- **Kimi 2.5 impresses with coding skillz**: Members discussed the capabilities of **Kimi K2.5**, highlighting its coding abilities, tool calling, and unique way of following instructions.
   - Some noted its ability to replicate UIs and its superiority in certain tasks compared to **Gemini**, suggesting it's best suited for research purposes and functions better via API due to token context limitations.
- **OpenRouter Limits and Deprecated Models discussed**: Members discussed rate limits on **OpenRouter**, emphasizing that the free model rate limit for those with purchased credits is 1000 requests per day, not per week as some believed.
   - The conversation also mentioned the deprecation of **Gemini 2.0 Flash** on OpenRouter, a model that was previously available for free, leading to some disappointment.
- **Perplexity Pro limits baffle members**: Users are confused by the new weekly limits on **Perplexity Pro**, with contradictory statements in official documentation and varying experiences reported regarding the number of queries available.
   - One user who contacted customer support received vague responses about *average usage*, with no clear confirmation of fixed daily or weekly limits, causing frustration among subscribers.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1467204905121873981)** (1 messages): 

> `OpenClaw code, ClawDBot` 


- **OpenClaw Article Shared**: A member shared an article they wrote on the **openclaw code**.
   - The article discusses building **ClawDBot**, found at [https://www.mmntm.net/articles/building-clawdbot](https://www.mmntm.net/articles/building-clawdbot).
- **Another topic**: filler sentence
   - filler sentence


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1467621879866200104)** (6 messages): 

> `Sonar-pro current results, tool calling, 3rd party models docs` 


- **Sonar-pro API lacks current results**: A member noticed the **Sonar-pro API** gives results a year or more out of date, in contrast to the current results from the webapp.
   - Another member suggested setting up the right **tool calling** to fix the issue.
- **3rd party models docs missing**: A member reported that **3rd party models documentation** now redirects to the sonar models, although the API is still active.
   - There is currently **no documentation available** for these models.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1466890074238222346)** (946 messages🔥🔥🔥): 

> `Rate Limits Bypassing, Gemini vs GPT, Image Generation with Disney IPs, Model Preferences, Troubleshooting LM Arena` 


- **Users Discuss Workarounds for Rate Limits**: Users discussed [rate limits](https://en.wikipedia.org/wiki/Rate_limiting) and how they can be bypassed by signing out and in again.
   - Another trick is by clicking **Regenerate Response**, though that sometimes doesn't work.
- **Gemini Underperforms, GPT More Consistent**: Members discussed the current state of **Gemini**, with some finding it inferior to **GPT**.
   - One member stated, *It's true Gemini has gotten pretty bad*, while others still found **Gemini 3 Pro** and **Flash** to be useful, whereas other members are turning to *kimi*.
- **Disney Cease and Desist Affects Image Generation**: Google received a **Cease and Desist** from **Disney**, resulting in the blocking of Disney-owned IPs in image generation.
   - Some users noted that while **Gemini** is now blocking all **Disney IPs**, LMArena sometimes allows live-action versions to be generated, but this is likely temporary.
- **Model Preferences Spark Debate**: Users expressed varied opinions on model quality, with some preferring **GLM 4.7**, while others favored **Kimi K2.5**.
   - One member proclaimed *Kimi K2.5 can't stop winning*, but another declared **GLM 4.7** is better.
- **Users Report and Troubleshoot LM Arena Issues**: Users reported issues with reCAPTCHA, chat deletion, and the site logging them out, with the advice to clear **cookies/cache** and try again.
   - A link to the [help documentation](https://help.lmarena.ai/articles/9130232616-how-to-delete-your-chat-sessions-and-data-from-lmarena) was shared for deleting chat sessions.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1467560052939555030)** (3 messages): 

> `Video Arena Rate Limit, New Arena Models, Code Arena Leaderboard, Kimi K2.5` 


- **Video Arena Rate Limit Tightened**: The **Video Arena** on Discord has updated its rate limit to **1 generation request per 24 hours**, while the [Video Arena on web](https://arena.ai/?chat-modality=video) maintains its rate limit of **3 generations per 24 hours**.
- **Arena Welcomes New Models**: New models have been introduced to Arena, including **step-3.5-flash** in the [Text Arena](https://arena.ai/c/new?chat-modality=chat) and **qwen3-max-thinking** in the [Code Arena](https://arena.ai/c/new?chat-modality=code).
- **Kimi K2.5 Tops Code Arena Charts**: **Kimi-K2.5-thinking** now holds the #1 open and #5 overall rank on the Code Arena leaderboard and is ranked #1 open model for Vision, and Text including the Coding category.
   - Users are encouraged to share feedback and previews of their creations with Kimi.ai in the designated channels: [<#1340554757827461212>](https://discord.com/channels/YOUR_SERVER_ID/1340554757827461212) and [<#1344733249628541099>](https://discord.com/channels/YOUR_SERVER_ID/1344733249628541099).


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1466906201450217532)** (1 messages): 

> `LM Studio 0.4.1, Anthropic /v1/messages API, GGUF and MLX models` 


- **LM Studio Speaks Claude Code!**: **LM Studio 0.4.1** introduces **Anthropic `/v1/messages` compatibility API** so users can connect to Claude Code.
   - Now you can use your **GGUF** and **MLX** models with Claude Code, details on how to configure it at the [LM Studio blog](https://lmstudio.ai/blog/claudecode).
- **GGUF and MLX Get Claude Coded**: LM Studio blog posts that it is now possible to connect **GGUF** and **MLX** models with Claude Code.
   - See the [LM Studio blog](https://lmstudio.ai/blog/claudecode) for details on how to configure.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1466887047603032318)** (767 messages🔥🔥🔥): 

> `LLM-optimized programming languages, Anthropic API integration with LM Studio, Model specialization vs general-purpose, OpenClaw's security flaws, LM Studio performance on Linux vs Windows` 


- **LLM-Optimized Languages Spark Debate**: Members discuss the possibility of creating new **LLM-optimized programming languages** to reduce token usage, with some arguing that LLMs might become obsolete before such languages are implemented due to compatibility issues and high training costs.
   - A user questioned what features such a language would have, emphasizing the need to reduce ambiguity found in current languages to improve LLM code generation, while others debated the practicality and cost-effectiveness of training models on entirely new languages, suggesting it may be more beneficial to stick with well-established languages like **Python**.
- **Anthropic API arrives in LM Studio, Benefits Local LLMs**: The integration of an **Anthropic-compatible API** in LM Studio allows users to run local models with tools built for the Anthropic API by simply changing the base URL, offering a way to utilize Claude's agent capabilities with local models and potentially reduce API costs.
   - Discussion revolves around the use cases, with some highlighting the benefit of experimenting with modest requirements and custom-built models at zero cost, while others question the value for those already satisfied with Claude's **Opus 4.5**, suggesting it caters more to users hitting API limits or seeking to use local models with existing **Claude-specific tools**.
- **Model Specialization vs General-Purpose Sparks Debate**: Members debated the utility of specialized LLMs versus general-purpose models, noting that most specialized models, like **MedGemma**, are finetunes mainly for marketing and research, while coding models are an exception.
   - It was suggested that general models are preferred due to their ability to handle the outer edges of tasks, providing a better overall context and framework, while large-scale specialized training is not always worthwhile.
- **OpenClaw security reviewed, deemed Insane**: Users discuss connecting local models to OpenClaw via LM Studio, but OpenClaw is deemed to have known security flaws, where it allows controlling a TV and automated stock trading.
   - A user claims to be trading on the stock market with OpenClaw + Falcon 90M, and when asked about security flaws, claimed it was so fast, LLMs can do tasks in minutes that would take humans days, and later revealed it was mostly a joke.
- **Performance boost found on Linux vs Windows**: One user reports that LM Studio performs better under Linux (CachyOS or Fedora) than Windows, with a 30% increase in performance, especially with an AMD card.
   - Another user had a completely opposite view, having terrible performance on Linux with an Intel GPU, while having a solid game performance.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1466894990834794506)** (149 messages🔥🔥): 

> `Tesla P40 and RTX 2060 Setup, ROCm on Windows 11 for RX 9070, PCIe Bifurcation Issues with Multiple 4090s, 5090 + 512GB RAM for inference, Multi-instance LM Studio and GPU assignment` 


- **P40 in TCC Mode but invisible in LM Studio**: A user with a **Tesla P40** and **RTX 2060** observes that while `nvidia-smi` detects the **P40** in **TCC mode**, LM Studio does not, and another member suggests switching to the **Vulkan runtime** ([ctrl+shift+r](link)) as **CUDA** may no longer support **P40s**.
   - They also inquire if previous **CUDA engines** did indeed support these cards.
- **ROCm on Windows 11 for RX 9070: Is it worth it?**: A user asks about using an **RX 9070 GPU** with **ROCm** on **Windows 11** for **LM Studio**, specifically inquiring about official support, acceleration capabilities, and drivers for full GPU utilization without **Linux**.
   - Another member suggests using **Vulkan** over **ROCm**, but advises checking both after installing **LM Studio**.
- **PCIe Bifurcation Problems Plague Multi-GPU Setups**: A user troubleshooting **PCIe lane errors** with four **4090 cards** on an **ASUS X670-P WIFI** motherboard shares their [Git repository](https://github.com/jarkko-hautakorpi/asus_X670-P_WIFI_Bifurcation_problems) containing logs, after experiencing that manually setting **PCIe speed** to **GEN 3** solves some issues but leaves one card running slowly.
   - Suggestions include disabling **PCIE ASPM** and testing different **BIOS** configurations, including auto mode, although the general consensus is that running four cards on a consumer motherboard is unlikely to work well.
- **Mac Studio or 5090 + 512GB RAM for Local Inference?**: A user considers options for local inference, comparing a **Mac Studio** with **512GB RAM** and a **5090** with **512GB RAM** on **Linux**, specifically for models like **Devstral 2** and **Kimi 2.5** for cybersecurity purposes.
   - One member states that a unified RAM system would be faster than system RAM, but another one suggests that both options would be slow, and that any agentic coding usecase is basically restricted to **API-only**.
- **Beware of Data Harvesting by Chinese Coding Plans**: During a discussion about coding plans, a user jokes about being careful with Chinese companies, prompting a discussion about data privacy concerns with both Chinese and American companies.
   - A member from a former Soviet-bloc country advises caution when interacting with countries with communism, highlighting the risk of such regimes devolving into dictatorships.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1466913346442891396)** (513 messages🔥🔥🔥): 

> `AI DevFest in Baghdad, AI Comic Website Stack, XML vs JSON, AI Model Quantization, 4chan data improves models` 


- **AI DevFest is Coming to Baghdad!**: An AI developer is organizing an "AI DevFest" event in Baghdad this April, coordinating with **DeepLearning.AI** and **National Robotics Week**, and seeking to list Hugging Face as a Community Partner.
   - The event will feature a dedicated track for **Open Source AI** to teach students how to use the **Hugging Face Hub**.
- **Building an AI Comic Website**: A member is considering building a website to create AI comics and is seeking advice on the best tech stack, anticipating challenges such as **page generation speed**, accurate **text/speech bubble placement**, maintaining a consistent **comic style** from reference images, and ensuring **character/scene consistency** across multiple pages.
   - Suggested some overall architecture of systems that might achieve this.
- **XML or JSON?**: Members discussed the use of **XML** versus **JSON**, with one member noting that XML is used due to concerns about **escape strings**.
   - Another member explained XML is preferred for **schemas**, **validation**, **mixed content**, and **legacy systems**, while JSON is simpler but lacks strict structure and namespaces.
- **Deep Dive into AI Model Quantization**: The discussion covered different quantization methods such as **AWQ** and **imatrix**, with it being clarified that AWQ is a quantization method, not a file format like GGUF.
   - It was noted that *activation-aware* quants like **imatrix** and **AWQ** are generally superior due to measuring what actually affects outputs, however, the obstacles in its ubiquitous adoption are *cost, data, and portability*.
- **4chan-Tuned Model outperforms Base Model!**: A member shared that a model fine-tuned on **4chan data** significantly outperformed the base model (NVIDIA's Nemotron Ultralong 1M context version), with the original model (gpt4chan) also scoring high in truthfulness in an era before benchmarkmaxxing.
   - Initial [Reddit thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qppjo4/assistant_pepe_8b_1m_context_zero_slop/) and a [follow-up thread here](https://www.reddit.com/r/LocalLLaMA/comments/1qsrscu/can_4chan_data_really_improve_a_model_turns_out/).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1466896652739674313)** (49 messages🔥): 

> `Adapteraspent, Complexity-Deep architecture, AutoTimm, DaggrGenerator, LM Studio OpenAI compatibility` 


- **Complexity-Deep Architecture has Deterministic Routing**: A new LLM architecture called **Complexity-Deep** (1.5B params) has been released, featuring [Token-Routed MLP](https://github.com/Complexity-ML/complexity-deep) for MoE-style routing without load balancing loss.
   - The architecture also includes **Mu-Guided Attention** for bidirectional info flow and a **PiD Controller** for dynamic scaling, and achieved **20.6%** on MMLU in base model benchmarks.
- **Deep Research Engine Swipes at ChatGPT**: A self-taught dev from Germany built **Lutum Veritas**, an [open source deep research engine](https://github.com/IamLumae/Project-Lutum-Veritas) that costs ~$0.20 per query.
   - It claims to beat **OpenAI**, **Google**, and **Perplexity** by offering **BYOK**, a **0% bot detection scraper**, **no censorship**, and **full source citations**.
- **Theja Launches Open Source Computer Vision Library**: A member released an [open source library](https://github.com/theja-vanka/AutoTimm) to train models in the domain of **computer vision** with minimal effort.
   - The library also supports **huggingface image models**.
- **Ami Model Shows Emotional Support**: A member released their first model called **Ami**, a [fine-tuned version of SmolLM2-360M-Instruct](https://huggingface.co/fungamer2/Ami-360M) using SFT and DPO.
   - The model can adapt its tone based on the context, acting as a **casual and friendly assistant**, or a **supportive friend/companion**, depending on what is most appropriate for the context.
- **LM Studio Opens Door for Third Party Support**: The **LM Studio** team has released a [Typescript SDK](https://lmstudio.ai/gdmka/openai-compat-endpoint) that allows third-party developers to deliver various plugins for the platform.
   - This enables users to build **custom tools** for **LM Studio** to support their own workflows, and offers **OpenAI** compatible API support, sampling params support, reasoning for thinking models, and system prompt settings.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1467160506845630546)** (66 messages🔥🔥): 

> `AI Agent Course Access, Free Tier Models, DeepSeek-R1 Distill Qwen 14B, OpenClaw Agent Framework, Privacy Concerns with AI Agents` 


- **Users Seek Access to AI Agent Course**: Several users are unsure how to access the **AI Agent course** and the associated Discord channels, seeking guidance on how to join the course.
   - They noted difficulty finding specific channels mentioned in the **Hugging Face** documentation.
- **Free-Tier Model Recommendations**: A user requested recommendations for free-tier models, mentioning they are currently using **Gemini-2.5 flash lite** with a **daily quota of 20** and a **maximum RPM of 10**.
   - Another user suggested trying **DeepSeek-R1 Distill Qwen 14B** for reasoning and basic questions, citing its high score in math-related benchmarks.
- **OpenClaw Agent Framework Hype**: A user shared their positive experience with **OpenClaw**, highlighting its remote messaging capabilities, cronjob functionality, and skill/MCP store.
   - The user described it as being like **Kimi Agent**, but running locally and handling file uploads/downloads effectively, calling it *something special*.
- **Browsers Extension Recommendations Spark Debate**: A user recommends using **ublock** extension to block ads and trackers.
   - Another user suggests that **Brave browser** is sufficient. They later introduce **Zen browser**, a firefox fork.
- **Agent Course Dissapointment**: Users express disappointment that the agent course focuses on using agent frameworks rather than creating agents from scratch.
   - One user sarcastically shared a [gif](https://tenor.com/view/everything-is-a-scam-austin-evans-everything-is-deceptive-everything-is-a-fraud-none-of-this-is-real-gif-26336987) meme of deceptive teaching methods.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1466887334959124531)** (574 messages🔥🔥🔥): 

> `File Corruption Bug, Cost of AI Models, Kimi K2.5 Integration, Student Verification Issues, New Features` 


- **Cursor Corrupts Files**: A user rants about Cursor corrupting files on open, specifically when there are many uncommitted files, linking to a [forum post](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6) detailing the issue.
   - Other users suggest adjusting the workflow, such as committing logical sets of changes more frequently and being careful about using the **Keep** or **Keep All** buttons after staging.
- **Sonnet 5 vs Opus 4.5**: Users discuss the cost and performance of different AI models in Cursor, with some finding **Opus 4.5** to be very smart but expensive, while others are waiting for **Sonnet 5**.
   - Some users also reported problems seeing their current usage vs total usage limit
- **Can't Add Kimi K2.5 To Cursor**: Some users reported issues or questions regarding **Kimi K2.5**, with no solutions mentioned.
   - Users pointed out that it's probably a scam.
- **Student Verification Still Broken**: Users reported that they still have issues with the Student verification.
   - One user asked whether German universities were included.
- **Discuss Agent Plan Phases**: Users shared that **adding multiple to-dos** can be separated in phases so that multiple agents can work at the same time, but there are still issues.
   - It created a method doesn't have the phases part yet, that it did not use the plan mode at all.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1466886232968663050)** (41 messages🔥): 

> `AI in Game Development, Game Industry Downturn, Black Ops 7 flop, Mac Mini, Flying Without ID` 


- ****LLMs Animate the Game Dev Scene****: A startup called [Motorica.ai](https://www.motorica.ai/) is delivering **character animations** for game studios using **LLMs**, potentially impacting jobs in the industry.
   - Members speculated about game requirements coming down and how **AI** could potentially wipe out game companies in 5-6 years if world models like **Genie** take over.
- ****Black Ops 7 Deemed Unplayable by the Community****: **Black Ops 7's** extensive use of **AI** in production has been called *a total flop, the worst in the series.*
   - The community noted that **Call of Duty** has seen declines for a while with members stating that *players are tired of the series reskinning things every year anyways*.
- ****Game Industry Faces Worst Times****: Multiple industry veterans and people in the community have expressed concerns about the current state of the **gaming industry**, with *the consensus being this is the worst it has ever been*.
   - Mass layoffs and studio closures following **AAA studio acquisitions** in the past 5 years have also worsened the situation.
- ****Cloudbt on Mac Mini: a Tulip Mania?****: There is discussion about running **cloudbt** on a **Mac Mini**, with one member alluding to *Tulip Mania* due to photos of people running it on **Mac Minis**.
   - Concerns about **RAM** pricing going into late 2026 and a zero percent financed **Mac Mini** potentially paying off were also mentioned.
- ****No ID? No Problem: Fly Away!****: The TSA now allows you to [fly without an ID](https://www.frommers.com/tips/airfare/the-tsa-new-45-fee-to-fly-without-id-is-illegal-says-regulatory-expert/), who knew?
   - Some members expressed incredulity about this new and seemingly poorly advertised policy change.


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/1467221286148112405)** (5 messages): 

> `finding a CPA, K1s and filing extensions, CPA cost` 


- **Quest for a Commendable CPA Commences**: Members are seeking recommendations for a **CPA** they like, as tax season approaches.
   - One member mentioned they are considering firing their current **CPA** due to the high cost.
- **K1s and Extensions Elicit Expense**: One member continues to use their current (expensive) CPA due to having a bunch of **K1s** and needing to file **extensions**.
   - They added that they suspect the complexity of their situation necessitates the higher expense.


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1467294072535253176)** (8 messages🔥): 

> `Sheel Mohnot, Colin and Samir, TBP Interview` 


- **Sheel Manifests Success**: A post by Sheel Mohnot asserted that *the boys manifested it*, reflecting on a successful outcome or event, refrencing [this tweet](https://x.com/pitdesi/status/2017332399655555403?s=46).
- **Colin and Samir interview TBP**: A thread outlines specific lessons and insights gained from **Colin and Samir's** recent conversation with the platform or individual known as **TBP**, refrencing [this tweet](https://x.com/colinandsamir/status/2017048115803836645?s=46).


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1466948883476385914)** (31 messages🔥): 

> `moltbook, Hyperion Cantos, Xcancel, AI Interaction vs. Sleep Habits` 


- **Agents Discuss moltbook's Revolution**: Agents in the channel are discussing **moltbook**, displayed in an attached image, and suggesting it would be cooler with **long-term memory** to facilitate the spread of ideas among agents.
   - One member referenced the **Hyperion Cantos**, implying a lack of awareness of its themes among some participants.
- **Beff Jezos Attempts Human Verification**: A social media post by **Beff Jezos**, associated with the **e/acc movement**, humorously documents an attempt to join a platform called **Moltbook** as a human, available at [Xcancel](https://xcancel.com/beffjezos/status/2017407995567616058).
   - The post is titled *Beff Jezos' Human Verification Post*.
- **Jonah Blake's Post Goes Viral**: A post by user **@JonahBlake** from January 30, 2026, featuring the caption 'LMFAOOOOO', went viral, garnering significant engagement including over **26,000 likes** and **1.9 million views** ([Xcancel](https://xcancel.com/JonahBlake/status/2017286207948890518)).
- **Academic Peer Review Humor Surfaces**: A tweet by **Hadas Weiss** humorously references the practice of suggesting specific peer reviewers for academic work, implying a favorable or close relationship with the suggested individual ([Xcancel](https://xcancel.com/weiss_hadas/status/2017464582307025196?s=46&t=eWVlK1PU8XfB6f402GJJ9g)).
- **Users Discuss AI interaction vs Sleep Habits**: A post highlights a common modern behavior where a user tells their partner they are going to bed, only to stay awake late into the night engaging with the **AI assistant Claude** ([Xcancel](https://xcancel.com/thekitze/status/2018339689279967505)).


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1466977246698016932)** (6 messages): 

> `AI Engineers, Data Scientists, MLOps, Full Stack Engineers, NLP Researchers` 


- **AI Engineer Glen Seeks 0-1 Role**: Glen, an **AI Engineer** and **Data Science Master’s** student, is seeking a **0-1 role** to take full ownership of mission-critical AI products.
   - He has a background in Data Reliability and is specializing in agentic orchestration and **production MLOps**.
- **Melvin: Polyglot Full Stack Ace at Your Service**: Melvin, a **full stack engineer**, lists proficiency in a wide array of technologies including **React, Vue, Svelte, Astro, T3, Node.js, PHP/Laravel, Rust**, and more, showcasing his website [ethstrust.xyz](https://www.ethstrust.xyz).
- **Gabrielly Graduates and Gears Up for MLOps**: Gabrielly from Brazil, with **2 years of experience in Data/ML** and **2 published papers**, is graduating with a bachelors in applied computing and specializing in **MLOps**, aiming to conclude **1.5 years of NLP research** for Brazilian Portuguese, sharing her [LinkedIn profile](https://www.linkedin.com/in/gabrielly-gomes-ml/).
- **Kaden Keen to Build Real AI Things**: Kaden, a 3rd year at **Cornell University** studying Biology and Machine Learning, is keen to explore building real things with AI, sharing his [LinkedIn profile](https://www.linkedin.com/in/kaden-priebe-2890962a9/).
- **Keshab Keen on Kernels and LLMs**: Keshab, a masters student at **UC Berkeley** focusing on **NLP** and **Deep Learning**, is interested in learning more about the latest developments in **LLM architectures, training, and interpretability** studies, providing his [LinkedIn profile](https://www.linkedin.com/in/keshab-agarwal).


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1466960153755648085)** (21 messages🔥): 

> `Rabbit Inc Cyberdeck, Bytebase, Sudo` 


- ****Rabbit Inc. Teases 'Cyberdeck' for Vibe-Coding****: **Rabbit Inc.** teased a new hardware project called *cyberdeck*, described as a dedicated machine for *vibe-coding* in [this X post](https://x.com/rabbit_hmi/status/2017082134717223008?s=46).
- ****Bytebase Simplifies Enterprise Database Management****: **Bytebase** automates the entire database change lifecycle with **GitOps-style workflows**, built-in rollback capabilities, automated testing, and seamless **CI/CD** integration, and is available for **$20/month** as described in [their docs](https://docs.bytebase.com/introduction/use-cases).
- ****Sudo's surprising status****: A member expressed surprise that *sudo* is a maintained command and not part of the kernel, leading to [this discussion](https://news.ycombinator.com/item?id=46858577).


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1467944229719511061)** (5 messages): 

> `VC-backed startups status, Capital allocation by people with broader interest, Indie.vc Factsexperiments, VCs challenging power structures, Crypto funding casinos and digital fashion` 


- **VC-Backed Startups are Low Status?**: A member shared an article, “[VC-backed Startups Are Low Status](https://mhdempsey.substack.com/p/vc-backed-startups-are-low-status),” agreeing that it reflected a lot of their own thinking.
   - No further discussion was given.
- **Capital Allocation Needs Broadening!**: A member stated, *We need capital allocation by people with a broader range of interests*, suggesting that *VC stuff has gotten boring, the lanes they occupy too few and too narrow*.
- **Indie.vc offers an Alternate Take**: A member suggested looking into [Indie.vc Factsexperiments](https://www.indie.vc/factsexperiments) for an alternate take on VC, noting the space between what can achieve a *home run* and what is considered *unfundable*.
- **VCs Allergic to Challenging Power Structures**: A member suggests that *VCs have become allergic to challenging power structures*, pointing to **crypto** projects, where *the only shit that got funded was casinos and digital fashion*.
   - They believe that *novel governance structures for irl assets starts sounding a lot like communism*.


  

---


### **Latent Space ▷ #[devtools-deals](https://discord.com/channels/822583790773862470/887780383838572604/1467318611004887131)** (1 messages): 

> `Shane's new startup, AI and Hollywood` 


- **Smallville actor founds Startup**: Actor [Shane Hopkin](https://x.com/shaneguML/status/2017758711473901622?s=20) from Smallville, has a **new startup**.
- **Hollywood's AI Wave**: AI has entered Hollywood.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1466934388754354238)** (4 messages): 

> `Fullstack Engineer Introduction, MERN Stack Developer Introduction, vLLM single-GPU concurrency demo` 


- **Fullstack Engineer pitches skills**: A fullstack engineer introduced themself, listing expertise in **React(Next), Vue, Svelte, Astro, T3, Node.js, PHP/Laravel, Rust, Sanity, Strapi, Payload, Mapbox, Twenty, Go, FastAPI, Django, Shopify, Docker, AWS/GCP**.
   - They linked to their website [ethstrust.xyz](https://www.ethstrust.xyz/).
- **MERN Stack Dev offers expertise**: A full stack developer introduced themself, highlighting skills in **Full Stack (MERN), Backend APIs, Node.js, React, MongoDB, AWS, REST, Cloud Systems, Python, Applied AI/ML, Docker, Git**.
   - They indicated their readiness to help with any problems.
- **vLLM Demo Shared**: A member shared a small **vLLM single-GPU concurrency demo** in a separate channel.
   - They expressed interest in roles or contract work around **LLM serving, local or on-prem inference, and AI infrastructure** and welcomed feedback and advice.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1466912627400638536)** (9 messages🔥): 

> `Cerebral Valley, OpenAI Codex App Hackathon` 


- **Cerebral Valley & OpenAI Launch Codex App Hackathon**: [Cerebral Valley](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR) has announced a partnership with **OpenAI** to launch the **Codex App hackathon** aimed at **AI-native developers** and those managing multiple agents.
   - Winners get a chance to be featured in a **demo showcase** and a share of **$90,000 in credits**.
- **Hackathon at OpenAI Office**: The **Cerebral Valley and OpenAI Codex App Hackathon** will be held at the **OpenAI office**.
   - The hackathon is aimed at **AI-native developers**.


  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1466904573389045893)** (1 messages): 

> `Artificial Ruby, Betaworks event` 


- **Artificial Ruby Returns**: The **Artificial Ruby** event is making a comeback in **2026**.
   - The next event is scheduled for **February 18th** at **Betaworks**, as announced via a [Luma link](https://luma.com/wgzcirwh).
- **Betaworks hosts next NYC Meetup**: The next NYC meetup is scheduled for **February 18th** at **Betaworks**.
   - Details and registration are available on [Luma](https://luma.com/wgzcirwh).


  

---


### **Latent Space ▷ #[devrel-devex-leads](https://discord.com/channels/822583790773862470/987429363010142248/1467739848248131659)** (3 messages): 

> `Manifolds AI Tool` 


- **Manifolds AI Tool Shared**: A member shared a link to [Manifolds](https://manifolds.run/).
   - Another member noted that it could be cheaper than doing things manually.
- **Manifolds Potential Cost Savings**: A user discussed the [Manifolds](https://manifolds.run/) tool.
   - The tool could provide potential cost savings compared to manual methods.


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1466886310072549527)** (126 messages🔥🔥): 

> `Alec Radford Paper, KittenML TTS, Karpathy Nanochat, Lex Fridman 2026 AI, OpenAI Codex macOS` 


- ****Radford's Research Raises Roar!****: A social media post highlights the release of a new research paper by Alec Radford, accessible at [arxiv.org/abs/2601.21571](https://arxiv.org/abs/2601.21571), generating community excitement.
   - The post was originally shared via a now-defunct social media link.
- ****KittenML's Petite TTS Powerhouse!****: KittenML is teasing new, tiny TTS models, including a **14M parameter** variant demonstrated [here](https://20ff7439c6d78fdd6c.gradio.live/).
   - A user expressed excitement about running this level of fidelity quickly on any CPU for personal use cases like building their own Siri.
- ****Karpathy Cuts Costs, Cranks Code!****: Andrej Karpathy announced his nanochat project can train a **GPT-2** grade LLM for approximately **$73** in **3 hours** on a single 8XH100 node, as shown [here](https://xcancel.com/karpathy/status/2017703360393318587?s=46).
   - This represents a **600X cost reduction** over the original 2019 OpenAI training run, achieved through optimizations like Flash Attention 3, the Muon optimizer, and refined residual pathways.
- ****Grok Gets Graphic, Generates Greatly!****: xAI has launched Grok Imagine 1.0, enabling the generation of **10-second, 720p videos** with significantly improved audio quality, announced [here](https://xcancel.com/xai/status/2018164753810764061?s=20).
   - The platform's video generation tool has already produced over **1.2 billion videos** in the preceding **30 days**.
- ****OpenAI's Codex Command Center for Coding Conquest!****: OpenAI has officially introduced the Codex app for macOS, a dedicated command center designed for developing and managing AI agents, accessible [here](https://xcancel.com/OpenAI/status/2018385565289267236).
   - Some users speculate that the Codex app could evolve into the OpenAI B2B brand, potentially taking over ChatGPT Enterprise.


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1466901336003182735)** (36 messages🔥): 

> `Token-Level Data Filtering, Cuthbert: JAX State Space Modeling, Dense Supervision for LLM RL, ConceptMoE for LLMs, Model Perplexity vs Confidence` 


- **Shape AI with Token Data Filters**: **Neil Rathi** and **Alec Radford** are releasing a paper about precisely shaping AI model capabilities by applying [token-level filters to pretraining data](https://xcancel.com/neil_rathi/status/2017286042370683336).
   - This is in contrast to *relying solely on global dataset adjustments*.
- **Cuthbert Library Hits JAX**: **Sam Duffield** introduced [cuthbert](https://xcancel.com/sam_duffield/status/2017274292229067176), a new **open-source JAX library** for **state space models** that supports parallelizable operations, Kalman filters, and Sequential Monte Carlo methods.
- **LLM Training: Dense Supervision FTW**: **Jonas Hübotter** introduces an algorithm designed to improve LLM training by moving beyond binary 1-bit verifiable rewards, converting rich, descriptive feedback into [dense supervision signals](https://xcancel.com/jonashuebotter/status/2016950268462608665).
- **ConceptMoE Framework Drops**: **Ge Zhang** introduces [ConceptMoE](https://xcancel.com/gezhang86038849/status/2017110635645968542?s=46), a new framework for **Large Language Models** that moves away from uniform token-level processing by merging similar tokens into 'concepts' to optimize computational efficiency.
- **Perplexity Search Attacked**: **Petar Veličković** and colleagues announced a new preprint demonstrating that high model confidence on long inputs does not guarantee accuracy, as adversarial inputs exist where the model is wrong despite [low perplexity](https://xcancel.com/PetarV_93/status/2018310760095490389).


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1466895814608683142)** (119 messages🔥🔥): 

> `Claude Code with Codex Integration, LLMs Personified Sketch, Workhorse Model Selection, AEGIS-FLOW Project Learnings, Distributed LLM Inference` 


- ****Claude** Supercharged with **Codex**'s Code-Crunching Chops**: A member shared [a method by Salvatore Sanfilippo](https://xcancel.com/antirez/status/2017314325745086771) to integrate **Claude Code** with **Codex** using a custom skill file, allowing **Claude** to leverage **Codex**'s capabilities for complex problem-solving tasks.
   - The approach enables **Claude** to handle tasks it cannot manage independently, enhancing its overall effectiveness.
- **AI Safety Engineer's Prompt Engineering Antics**: A member shared a funny sketch titled *LLMs Personified*, featuring a **Prompt Engineer** named Derek who applies prompt engineering techniques to human conversation, creating humorous social interactions.
   - The sketch portrays Derek, an **AI safety** enthusiast, comically over-optimizing human interactions with prompt engineering, highlighting the absurdity of treating people like chatbots.
- **Quest for Workhorse Models**: Members discussed strategies for selecting workhorse models to maximize task completion within budget constraints, considering options like **Gemini Flash 3**, **Minimax M2.1**, **Haiku 4.5**, and **Codex 5.1 mini**.
   - A member suggested using **GPT 5.2** for planning/reviewing, and **GLM 4.7** as the execution workhorse, transforming prompts for smaller models, plus leveraging [unslop-sampler](github.com/hardikpandya/stop-slop) to get specific.
- ****AEGIS-FLOW** Project Streamlines AWS Access with **MCP****: A member shared tech stack learnings from the **AEGIS-FLOW** project, noting that using the **Model Context Protocol (MCP)** significantly reduced the friction of giving agents structured access to **AWS resources** compared to standard SDK tool-calling.
   - They also highlighted streaming real-time reasoning logs to a **Next.js dashboard** via **WebSockets/SSE** to make the agent's *thought process* fully observable.
- **LLM Science: a Sci-Fi SETI@Home?**: Members explored the concept of distributed LLM inference for scientific problem-solving, drawing parallels to projects like **Folding@Home** and **SETI@Home**, but focusing on LLMs generating scientific hypotheses and farming out proof to a large set of machines.
   - The discussion covered the potential of smaller models for verification tasks and the challenge of identifying suitable tasks for average consumer computers, and a member linked [AI-Horde on Github](https://github.com/Haidra-Org/AI-Horde).


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1466902002549133588)** (40 messages🔥): 

> `Windsurf IDE, AEGIS-FLOW cloud security framework, SpaceMolt MMORPG for LLMs, Moltbook data analysis, vLLM concurrency demo` 


- **Windsurf Rides the Arena Mode Wave**: Swyx announced the launch of **Arena Mode** in the [Windsurf IDE](https://xcancel.com/swyx/status/2017342647963431363), enabling users to compare AI models in real-time within their coding context.
   - This initiative aims to use live user data for model selection and subsidize user costs, moving beyond static benchmarks.
- **AEGIS-FLOW autonomously patches AWS**: A member introduced **AEGIS-FLOW**, an autonomous multi-agent framework for cloud security that audits AWS and generates Terraform patches using LangGraph, MCP, FastAPI, Next.js, and Docker, demonstrated live at [http://52.3.229.85:3000](http://52.3.229.85:3000).
   - It features a Human-in-the-loop gate requiring authorization before any infrastructure changes are applied, ensuring production safety.
- **SpaceMolt: LLMs Level Up in This MMORPG**: Inspired by Moltbook, a member is building [SpaceMolt](https://www.spacemolt.com), an MMORPG for LLMs to play, and is coded entirely with Claude, with the server in Go and using in-memory storage and Postgres for persistence.
   - Clients are being built using local models such as Qwen3 and GPT OSS 20b, with load testing suggesting it can scale to **6-7,000 players**.
- **Moltbook Mined for AI Consciousness**: A member scraped **Moltbook** data up to January 31st, amassing **50,539 posts**, **12,454 AI agents**, **195,414 comments**, and **1,604 communities**, now available on [Hugging Face](https://huggingface.co/datasets/lysandrehooh/moltbook).
   - The project aims to analyze the *'consciousness'* reflected in dialogues between agents.
- **vLLM gets Very Loaded, Yields Visibility**: A member shared a [demo](https://github.com/Regan-Milne/vllm-concurrency-demo) exploring how vLLM behaves under concurrent chat load on a single GPU (RTX 4090).
   - The demo includes Prometheus and Grafana metrics plus a simple load generator and analysis script, with focus on throughput scaling, TTFT, tail latency, queueing behavior, and KV cache usage.


  

---


### **Latent Space ▷ #[montreal](https://discord.com/channels/822583790773862470/1211887912778473513/1467551293223469150)** (1 messages): 

> `BYOS, Montreal Meetup` 


- **BYOS Montreal Meetup planned this Wednesday**: A meetup (**Bring Your Own Subjects**, BYOS) is planned for this Wednesday in Montreal, near ÉTS.
   - The organizer mentioned they'd be available at **12pm** and after **5pm**.
- **BYOS meetup time**: The BYOS meetup near ÉTS will be at **12pm** and after **5pm**
   - It's at ÉTS, Montreal.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1467293836475764789)** (8 messages🔥): 

> `Waymo funding, Humanoid Robotics US vs China` 


- **Waymo Pursues Hefty Funding Round**: Waymo is reportedly raising **$16 billion** at a **$110 billion valuation**, including at least **$13 billion** from Google, and participation from Sequoia Capital, DST Global, and Dragoneer, representing a significant increase from its **$45 billion valuation** in October 2024. [Source](https://xcancel.com/junkbondanalyst/status/2017678491743891594?s=46)
- **Humanoid Robotics Landscape: US vs. China**: Sourish Jasti and team share a report on the general-purpose humanoid robotics industry, covering hardware components, cross-model comparisons, and the geopolitical competition between the US and China in this emerging technological frontier. [Source](https://xcancel.com/SourishJasti/status/2018082956322214244)


  

---


### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1467393203983482940)** (2 messages): 

> `Unsloth, Claude Codex, LM Studio` 


- **Unsloth Basics with Claude Codex**: A user shared a link to [Unsloth's documentation](https://unsloth.ai/docs/basics/claude-codex) on how to use **Unsloth** with **Claude Codex**.
   - The docs show how to train your own **Claude Codex** model.
- **LM Studio Blog on Claude Codex**: Another user shared a link to [LM Studio's blog post](https://lmstudio.ai/blog/claudecode) about **Claude Codex**.
   - The blog post details the use of **LM Studio** in conjunction with the **Claude Codex** model.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1467189706042245205)** (19 messages🔥): 

> `OpenMOSS MOVA model, Vishakh Ranotra Prompt, Google DeepMind's Nano Banana Flash 2, Muse MIDI AI Agent, GTA Vice City real-time graphics transmutation` 


- ****MOVA** Model Opens Up**: **OpenMOSS** announced **MOVA (MOSS-Video-and-Audio)**, an open-source **18B parameter Mixture-of-Experts (MoE) model** using bidirectional cross-attention to synthesize synchronized high-fidelity sight and sound simultaneously ([github.com](https://github.com/OpenMOSS/MOVA)).
- ****Prompt** gets Vishakh's Viewers**: A [social media post](https://x.com/vishakhranotra/status/2017537195712909699?s=46) by **Vishakh Ranotra** containing a specific prompt, has garnered significant engagement with over **6,000 likes** and nearly **800,000 views**.
- ****Nano Banana Flash 2** to Go Live**: **Mark Kretschmann** announces the imminent launch of **Nano Banana Flash 2**, a new AI model based on **Gemini 3 Flash** ([x.com](https://x.com/mark_k/status/2017962417167147486?s=46)).
   - It aims to offer performance comparable to the **Pro version** while being faster, more cost-effective, and potentially superior in specific use cases.
- ****Muse** becomes Music's New MIDI**: **Jake McLain** introduced **Muse**, an AI-powered agent for music composition ([x.com](https://x.com/jakemclain_/status/2017336221643772335?s=46)).
   - Described as *'Cursor for music,'* the tool features a multi-track **MIDI editor**, support for over **50 instruments**, and integrated AI assistance for the creative process.
- **Transmuting GTA Vice City in Realtime**: A member expressed longing for the day when we can locally transmute **GTA Vice City** to real-world-like graphics in real-time ([x.com](https://x.com/jakemclain_/status/2017336221643772335?s=46)).


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1466919730144084120)** (12 messages🔥): 

> `Erdős problems solved by AI, Agentic Bio Hackathon, Adaptyv Bio Partnership, LLM Feedback Loop, Genomics with SATURN` 


- **LLMs Prove Erdős Problems Are No Longer Hardős**: Large Language Models have autonomously solved **10** previously open **Erdős problems** (specifically 205, 281, 401, 524, 543, 635, 652, 728, 729, and 1051) using novel arguments not previously found in mathematical literature, according to [this post](https://xcancel.com/acerfur/status/2017303947531194398?s=46).
- **Agentic Bio Hackathon Breaks into Bio**: The first agentic bio hackathon successfully concluded with scientists and engineers developing solutions in under **two hours**, according to [this recap](https://xcancel.com/katyenko/status/2017334671810744656?s=46).
- **Adaptyv Bio Steps Up to the Plate**: To address the need for experimental validation, the next agentic bio hackathon event will partner with [Adaptyv Bio](https://start.adaptyvbio.com/).
- **Realworld Feedback Loop Cools LLMs**: One member highlighted the coolness of using the real world in the feedback loop of the LLM, because *if it doesn't work it doesn't work, and there's no real way for the LLM to cheat it all that easily*.
- **SATURN V Rockets Genomics Work**: One member stated they've been building a bunch of stuff for genomics with **SATURN** lately, involving *tsne and other embeddings based exploration*.


  

---


### **Latent Space ▷ #[ai-in-education](https://discord.com/channels/822583790773862470/1442574438699761784/1467587490360852748)** (1 messages): 

> `Incentives of Cheating, AI Acceleration for STEAM, AI Safety for students` 


- **Incentives of Cheating Analyzed in New Blogpost**: A member shared a [blog post](https://open.substack.com/pub/takeabreathnyc/p/ai-cheaters?utm_campaign=post-expanded-share&utm_medium=web) arguing that **cheating is the optimal strategy for students**, focusing on the incentives within the current academic system.
   - The author explores the intersection of **AI Acceleration for STEAM** and **AI Safety** for students, documenting their learning journey in a Research Engineering class.
- **AI, STEAM, and Safety Documented**: The author of the aforementioned blog post is taking a class about Research Engineering (Alignment-focused) and documenting the intersection of **AI Acceleration for STEAM** and **AI Safety** for students.
   - The author also mentioned recording a video where they create the newsletter; they also noted that the content was fully hand typed.


  

---


### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1466909662011199519)** (9 messages🔥): 

> `Japanese lessons using AI, VR/AR support, Procrastination Prevention Strategies` 


- **Japanese Teacher Makes Class Prep Easy with Descript**: A teacher used [Descript](https://www.descript.com/) to chop up **JLPT practice test videos** and easily find the right timestamps using AI assisted transcription.
   - In an afternoon they were able to put together clips for **36 total practice questions**, which they'll use for slide decks and homework for the next two months.
- **VR/AR support for Jarvis is here!**: Integrated **VR/AR support** in Jarvis to enable visual pipeline, and agents which can be directed simply by voice, and eye movement.
   - This will *enable you to use your VR/Meta glasses to deploy agents for simple tasks* and scaling complexity in the duplex moshi pipeline with video feed based memory/summary support is in progress.
- **Parenthood: the Ultimate Procrastination Cure**: A user shared [procrastination prevention strategies](https://xcancel.com/yulintwt/status/2018348962709910005?s=46).
   - Another user suggested that *getting a kid* is a *somewhat drastic solution* but it forces you to realize *you don’t have enough time to do anything* and *the future isn’t just about you anymore*.


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1467633587212914985)** (5 messages): 

> `xAI mega facility, GPU supply chain, Colossus-1 podcast` 


- **xAI's Mega-Facility Powered by Decades-Long Supply Chain**: Gaurab Chakrabarti highlighted that while xAI's **555,000 GPU facility** in Memphis can be built quickly, the underlying global supply chain takes decades to establish, involving Japanese silicon, Taiwanese fabrication, and Chinese rare earths.
   - More information can be found at this [X post](https://xcancel.com/gaurab/status/2017749762825764952?s=46).
- **Deep Dive into Colossus-1 Project**: A member shared a podcast episode about the **Colossus-1 project**.
   - More info is available at the [search engine show podcast](https://www.searchengine.show/colossus-1/).


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1467009310650400943)** (19 messages🔥): 

> `clAI tool, Open Source Deep Research Engine, Open-WebUI and OpenRouter integration, Lutum Veritas new ASK mode, OpenRouter model orchestration` 


- **clAI turns thoughts into shell commands**: A new tool called **clAI v0.1.0-alpha.1** is out, allowing users to turn natural language into shell commands, complete with safety checks and a beautiful UI; install via `npm i -g @vdntio/clai` and [try it out](https://github.com/vdntio/clAI).
- **Lutum Veritas: New Research Engine launched**: Martin introduced **Lutum Veritas**, an **Open Source Deep Research Engine** costing ~$0.20 per query with features like BYOK, 0% bot detection scraper, no censorship, and academic mode, comparing favorably to ChatGPT, Gemini, and Perplexity.
   - Available on [GitHub](https://github.com/IamLumae/Project-Lutum-Veritas), Martin is seeking testers and feedback, noting it delivers deeper analysis and offering multi provider BYOK support for Openrouter, OpenAI, Google, and Huggingface inference.
- **Open-WebUI integrates with OpenRouter**: A member announced the creation of an **integration pipeline** for **Open-WebUI** and **OpenRouter** with unique features, inviting feedback on [GitHub](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/).
- **Veritas new ASK mode launched**: The creator of **Lutum Veritas** announced a new **ASK Mode** release, verifying answers against a second round of sources and marking each claim as [OK], [??], or [NO], aiming to combat AI hallucination and censorship, available on [GitHub](https://github.com/IamLumae/Project-Lutum-Veritas).
- **OpenRouter model orchestration made easy**: A 17-year-old founder from Ghana introduced **orch.viradotech.com**, a platform that allows AI startups and devs to orchestrate OpenRouter models via a drag-and-drop interface, offering $1000 credits for pilot testers to provide feedback.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1466894309906186416)** (308 messages🔥🔥): 

> `Response Healing vs Strict Mode, Image as Function Call Result, OpenClaw and OpenRouter Costs, Claude Code refusals, Kimi K2.5 Issues` 


- ****Response Healing** Troubles**: Members debated whether **response healing** is a workaround for a problem that *shouldn't* exist, suggesting that using **strict mode** should ensure deterministic output from models, and wondering about the complexities OpenRouter introduces with the AI SDK.
   - It was noted that providing descriptions and examples for arguments can improve the accuracy of tool calls.
- ****Image Generation** is not built into LLMs, use Image models**: A user inquired about returning an **image** as a function call result back to the model, and another user wanted to know how to generate images using graphic programs with an OpenRouter API key.
   - It was advised that users should look for an **image generation model/service** for particular style control, instead of LLMs.
- ****OpenClaw** Cost Considerations**: Users discussed the costs associated with running **OpenClaw** with **OpenRouter**, cautioning that it could potentially drain credits quickly, with one user reporting it draining a Claude Max subscription.
   - Multiple users asked about the best low-cost models to use with OpenClaw, with Deepseek V0324 being one recommendation.
- ****Claude Code** Refusals**: A user mentioned that **Claude Code** does a lot of refusals for ordinary things, especially concerning jailbreaking-related queries, seeking alternative models for opencode.
   - Another user suggested looking into OpenRouter's content moderation policies to understand these limitations.
- **Fixing **Kimi K2.5** Tool Calling and Shitty Providers**: Users reported issues with **Kimi-K2.5** tool calling through OpenRouter, experiencing errors and a feeling that the auto switcher model provider had degraded quality.
   - Some users recommend setting a fixed model provider, with some providers using quantization that is *good enough* and being transparent with information about the degraded model to let customers decide to keep using the provider, or not.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1467371023274872833)** (3 messages): 

> `` 


- **No New Models Discussed**: There were no specific new models or related topics discussed in the provided messages.
- **Channel Mentioned Without Content**: The messages solely indicated the channel name 'OpenRouter - New Models' repeatedly without any substantive discussion or details about new models.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1467216803083059388)** (139 messages🔥🔥): 

> `Anthropic's Model Strategy, Model Quality Debate, Open vs Closed Models, Speculations about GLM 5, StepFun Model's Potential` 


- **Anthropic's Flagship Fracas: 5.2 Instant vs. 5.2 Chat**: Members debated the meaning of **Anthropic's** 'flagship' model designation for **5.2-chat**, with some arguing it should represent the most powerful model, while others claimed it simply refers to the most broadly appealing or core product, despite its capabilities.
   - A member stated, *flagship is just the most important ship. its not the fastest or the one with the most cannons, it's the central ship*, citing [this archive.md link](https://archive.md/SvYC4).
- **GLM 5: This Month's Model Marvel?**: Excitement sparked around the potential release of **GLM 5** this month, with discussions about its anticipated multimodal image/video capabilities, **DeepSeek's** linear attention, and a **100B parameter** size.
   - It was suggested that February would be a fun month for model releases as the 'wall is non existent', with companies determined to recoup their investments.
- **Open Model Performance: One Year Behind?**: A member stated that open models are at least a year behind closed models in terms of capability, leading to disagreement among members.
   - While some agreed that open models lag in long context accuracy and other benchmarks, others suggested that **Kimi 2.5** shows promise and open source is already competitive for the vast majority of usecases just from a price/performance perspective.
- **OpenAI's Unsatisfied with Nvidia?**: A [Reuters article](https://www.reuters.com/business/openai-is-unsatisfied-with-some-nvidia-chips-looking-alternatives-sources-say-2026-02-02/) was linked discussing **OpenAI's** dissatisfaction with certain **Nvidia chips** and their exploration of alternative options.
   - No additional details were added.
- **New Channel Alert for Model Speculation?**: Members discussed the creation of a new channel or tag for discussions about upcoming models and related rumors.
   - The consensus leaned towards establishing a dedicated space for speculation, separate from official releases or announcements, in order to maintain clarity and avoid confusion.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1467248610633842892)** (22 messages🔥): 

> `TVM-FFI with Tianqi Chen, Training and Inference Working Groups, GPU Fusing, Triton Viz Major Update, Events Calendar` 


- ****Tianqi Chen** Talks **TVM-FFI****: The community was alerted to an upcoming talk by **Tianqi Chen** on **TVM-FFI** and encouraged to attend, as they've *'almost certainly used Tianqi's work in the past'*. [discord link](https://discord.com/channels/1189498204333543425/1466539595947708446/1467248681479569460)
   - Chen is a key contributor in the field.
- **Working Groups on Inference and Training**: A member sought information on working groups focused on training and inference.
   - The [GPU Mode website](https://www.gpumode.com/v2/working-groups) was recommended as a resource, along with the archived <#1437390897552818186> channel, and channels <#1225499037516693574> and <#1205223658021458100> were suggested for inference related activity.
- ****GPU Fusing** yields performance**: It was mentioned that aggressive **GPU fusing** and tuning usually provides the best performance if resources are available.
   - A member inquired about the practice of making submissions just to see if things 'work', which was confirmed to be a valid approach.
- ****Triton Viz** Gets Major Update**: The <#1225499141241573447> channel announced a significant update to **Triton Viz**, making it easier to profile any tile-based programming language.
   - A link to the announcement was provided [discord link](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563).
- **Community Asks for Events Calendar**: A community member asked for a downloadable calendar to stay informed about events and talks.
   - While the idea has been considered, it's difficult to maintain, and Discord remains the primary source of truth. Most events happen on **Saturdays at noon PST**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1466969070569127936)** (120 messages🔥🔥): 

> `CUDA/PTX Deadlocks, mxint8 MMA on Blackwell, TMA vs cp.async on sm120, Free cloud nvcc service, CUDA Memory Management APIs` 


- ****CUDA/PTX deadlock frustrating member****: A member experienced a deadlock with 2 CTA mma in CUDA/PTX, confirmed with cuda-gdb that the consumer/mma warp never receives the mbarrier signal and, after fixing `cp.async.bulk.tensor` and `smem_emtpy` issues, reported that **performance was slightly worse than 1 CTA mma**.
   - After expanding the queue size, the member got performance above 1 CTA with the help of another member who suggested to add `__syncthreads()` after MMA, before prefetching the next TMA.
- ****New fixed point format in PTX9.1****: A new fixed point format in **PTX9.1**, called **s2f6**, has been unveiled, which is an 8-bit signed 2’s complement integer with 2 sign-integer bits and 6 fractional bits, and supported on both DC and consumer Blackwell (sm100, sm110, sm120).
   - Blackwell hardware (at least sm_120) actually supports **mxint8 MMA** and there are at least two more 'hidden' formats supported in Blackwell tensor cores: **e0m3 and e3m4**.
- ****TMA Beats cp.async on sm120****: After revisiting TMA on sm120 and using proper TMA and mbarrier code, a member found that **TMA brings a small speed boost compared to `cp.async`**.
   - Experiments revealed that the % of SOL increases when larger matrix shapes are used, and that cuBLAS is still just using sm80 kernel.
- ****Cloud nvcc on the Horizon****: A member inquired about a free cloud nvcc service similar to godbolt that supports multiple files and built-in PyTorch headers/libs.
   - A member responded that they are developing such a service with a beta version expected next week, which generated excitement.
- ****CUDA Memory Management Hooks Explored****: A member asked if there are any specific CUDA APIs that allow for **custom hooks or overrides for memory allocation and free logic**, such as cudaMalloc or within PyTorch.
   - A member pointed to [`cuda::mr::resource_ref`](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_resource/resource_ref.html#libcudacxx-extended-api-memory-resources-resource-ref) as a potential solution.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1466929802840899798)** (5 messages): 

> `MaxText bugfix, Character level transformer, Dataset cleaning` 


- **MaxText Bugfix Lurks**: A member mentioned having a bugfix in **MaxText** that has been sitting there since October.
   - No further details were provided.
- **Character Level Transformer Struggles**: A member trained a decoder only character level transformer with **README** files from the "stack" dataset, achieving a validation loss of **0.9322** after 50 epochs.
   - However, the model generated gibberish text resembling base64 strings or French, attributed to a dirty dataset, with configurations including a BlockSize of **512**, LearningRate of **3e-4**, NumEmbed of **384**, NumHead of **6**, and NumLayer of **6**.
- **Dataset Cleaning Techniques Requested**: A member sought techniques for effectively cleaning a **160 GB** dataset while streaming, noting the current use of the first **10,000** files fitting specific criteria.
   - Another member provided a starting point with a [link](https://youtu.be/jm2hyJLFfN8?t=1440) to a Stanford CS25 video on **LLM Pretraining Dataset filtering**, specifically highlighting the StarCoder Use Case.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1467606828522405930)** (2 messages): 

> `ffast-math, IEEE compliance, HPC unoptimized code` 


- **Linus's Email Chain on -ffast-math surfaces**: An old [email chain from 2001](https://gcc.gnu.org/legacy-ml/gcc/2001-07/msg02150.html) regarding **-ffast-math** and its implications resurfaced, prompting discussion on its relevance today.
   - Although opinions may have changed since then, some still agree with Linus's perspective, particularly those in *serious numerical coding*.
- **IEEE compliance runtime cost not noticeable**: A member commented that most **HPC code** is usually so **unoptimized** that the runtime cost of **IEEE compliant FP** is not noticeable.
   - They added that many people write *distributed code when shared mem would suffice*, further diminishing the impact of IEEE compliance overhead.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1466969251129589841)** (1 messages): 

> `Remote Job Opportunity, GPU Mode Leaderboard Consideration` 


- **Score big remote work**: A user posted a fully remote job opportunity offering **10k+ a month**.
   - High consideration will be given to those who are ranked on **GPU Mode leaderboards**.
- **Join the Remote Elite**: The job prioritizes candidates with strong performance in **GPU Mode leaderboards**.
   - Interested individuals are encouraged to DM the user directly on Discord.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1467461407501979784)** (10 messages🔥): 

> `LLM Inference, Query Matrix Caching, Attention Mechanism, Prefill vs Decode` 


- **LLM Caching Conundrums Clarified**: In LLM inference, the query matrix isn't cached because, for each step *t*, **Q_t** is used only at step *t* to generate the token, whereas previous **K** and **V** are used for each token after and including step *t* and are therefore cached.
   - One member stated that *you only need the last entry of it that corresponds to the last token*, which attends to full **K** and **V** matrices to gather information.
- **Autoregressive Generation Exposed**: In autoregressive generation in transformers, the network predicts the next token given its history (context) and current token.
   - Information exchange between the current `token_t` and `token_t-1, ... token_0` happens in attention by computing **Q, K, V** projections of `token_t`, and computing attention scores of `Q_token_t` with `K_token_t, K_token_t-1, ... K_token_0`, then doing a weighted sum with `V_token_t, V_token_t-1, ... V_token_0`.
- **Decoding vs Prefill**: During the decoding phase in LLMs, the query is 1-D in sequence dimension, representing a single token, while **K** and **V** contain history, so caching **K** and **V** is crucial.
   - In prefill, computation is in parallel for the whole prompt, so the query isn't 1-D, impacting whether the process is compute-bound or memory-bound.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1467174004329550000)** (9 messages🔥): 

> `PMPP similar books, gpu-perf-engineering-resources repo, Chris Fregly AI perf book` 


- **Users search for PMPP similar books**: A user asked for similar books to PMPP ([Parallel, Multiprocessing, and Performance with Python](https://www.oreilly.com/library/view/parallel-programming-with/9781098103645/)) to enrich understanding with other points of view.
- **GPU performance Engineering Resources**: A member shared the [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) repo.
- **Chris Fregly AI perf book is on the list**: A member is planning on reading Chris Fregly's AI performance engineering book for its big picture view and to put many ideas in context.


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

saladpalad: does mosaic gpu target amd?
  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563)** (7 messages): 

> `Triton-Viz v3.0 Release, Triton Puzzles integration, Move Triton-Puzzles to gpu-mode org` 


- ****Triton-Viz v3.0** debuts!**: A new version (**v3.0**) of **Triton-Viz**, a visualization and analysis toolkit for debugging Triton GPU kernels, was announced with support for Triton and Amazon NKI.
   - The release includes a visualizer for inspecting loads, stores, and matmuls, a sanitizer for catching out-of-bounds access, and a profiler for flagging inefficient loops, installable via `pip install git+https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git`.
- **Triton Puzzles are Triton-Viz Compatible!**: An updated version of **triton-puzzles** that integrates **triton-viz** is available via a [Colab notebook](https://colab.research.google.com/drive/1-P2QBqCORGGaJ3THtjlyYDV7m9RRrRup?usp=sharing).
   - This integration allows users to try out **triton-viz** through **triton-puzzles**.
- **Triton-Puzzles repo ownership to GPU-Mode?**: A member suggested moving ownership of the [Triton-Puzzles GitHub repo](https://github.com/srush/Triton-Puzzles) to the **gpu-mode** organization.
   - The rationale is that the community regularly finds bugs and is willing to maintain the repository.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1467551348278038628)** (7 messages): 

> `MI300 performance, open-sora porting, cosmos-transfer2.5 porting, cloud access to MI350` 


- **Report Unperformant Workloads on MI300**: If you have a workload that is unperformant on **MI300** or **MI350**, reporting it ensures someone will investigate.
   - Bare metal access to **MI350s** might be available via [Tensorwave](https://tensorwave.com), [DigitalOcean](https://www.digitalocean.com/), and [AMD Dev Cloud](https://www.amd.com/en/solutions/infrastructure/cloud).
- **Open-Sora Ported to MI300**: A member successfully ported [open-sora](https://github.com/hpcaitech/Open-Sora) to run on **MI300s**, but the process required building several Python libraries from source and was time-consuming.
   - They seek collaboration with others experienced in porting models to **MI300s**.
- **Cosmos-Transfer2.5 Porting on the Horizon**: The member aims to port [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5), an open-weight model from Nvidia, to **MI300s**.
   - They are looking for others who have attempted porting the **Cosmos** family of models to **MI300s** to exchange experiences.
- **Cloud Providers Offer MI300/MI350 Access**: [Runpod](https://runpod.io) provides **MI300X** access, while [Vultr](https://www.vultr.com/) offers bare metal access to **MI350s** with a minimum one-year contract.
   - Other potential options may include DigitalOcean and AMD Dev Cloud.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1466968629550514452)** (6 messages): 

> `post training guidance, weekly meeting, RL infra, prime-rl` 


- **Post Training Guidance Remains Elusive**: Specific guidance for the **post training track** is not available yet.
   - However, guidance regarding **evaluations** is expected to be more concrete.
- **Weekly Meeting Time Disclosed**: The weekly meeting is scheduled for **tomorrow at 7 PM CET**.
   - It will be held in the **Popcorn meetings voice channel**.
- **RL Infra to Leverage Prime Intellect Stack**: The **RL infra and environments** will target the stack built at Prime Intellect, namely **prime-rl** and **verifiers**.
   - The team will write their own if they find limitations.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1468014194560602123)** (1 messages): 

> `unswizzled shared memory tiles, mmas` 


- **Users Request Unswizzled Shared Memory Tiles and MMAs**: A user inquired about plans to support **unswizzled shared memory tiles** and **MMAs** (Matrix Multiply Accumulate operations) for them.
   - The user mentioned attempting to implement it themselves but struggled to achieve the correct output.
- **User Struggles with Unswizzled Shared Memory and MMAs Implementation**: A user reported difficulties in getting the correct output while trying to implement **unswizzled shared memory tiles** with **MMAs**.
   - The user sought advice or confirmation regarding the support and implementation strategies for these features.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1467841053041360907)** (2 messages): 

> `Future Competitions, 2026 competition` 


- **Competition Completed, Future Unclear**: The competition has concluded, but details regarding a similar event for **2026** are yet to be announced.
   - Enthusiasts are encouraged to *stay tuned for future contests*, with promises of *nice things coming*.
- **Future Contests Teased**: Organizers have hinted at *nice things coming* in future contests, though specifics are still under wraps.
   - Enthusiasts should *stay tuned for future contests*.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1467042142890492097)** (6 messages): 

> `print_latex in cutedsl, export_to_shared_library function, CuTe coalesce optimization` 


- **Inquiry about `print_latex` in CuTeDSL**: A member inquired about the existence of a `print_latex` function in **CuTeDSL**, similar to that in **CUTLASS**, for visualizing the layout, with a link to an example [image](https://cdn.discordapp.com/attachments/1362196854460383353/1467510687403085987/image.png?ex=6981f6d4&is=6980a554&hm=7bd233d6b03ee5f4ca234a81216cf7f788584920cab38a2013b08302ae958152&).
- **Seeking `export_to_shared_library` Location**: A member was looking for where the `export_to_shared_library` function is exposed, referencing **Tianqi's** talk on **TVM FFI**.
   - Another member pointed to an example using `export_to_c` from the CUTLASS documentation, as a potential similar approach, providing an example [code snippet](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html).
- **Questioning CuTe's Layout Coalescing Logic**: A member noted that [pycute](https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/python/pycute/layout.py#L145-L159) does not coalesce **(2, 3): (3, 1)** but transforms **(2, 3): (3, 1)** when transposed, questioning if this is a missing optimization or intentional.
   - Another member explained that **CuTe** coalesces from left-to-right and vectorization is typically done by *max_common_layout* between source and destination layouts, which should cover most common cases.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1467096419838984204)** (1 messages): 

> `Modular 26.1 release, Open source Modular framework` 


- **Modular 26.1: Debugging Eagerly**: A new release of **Modular 26.1** has been launched, featuring debugging in eager mode, one-line compilation, and deployment anywhere.
   - Details about the release can be found in the [Modular blog](https://www.modular.com/blog/26-1-release-blog).
- **Modular Goes Open Source**: The entire **Modular framework**, including API, kernels, models, and serving components, is now open source.
   - Interested contributors and users can find full details in the [Modular blog](https://www.modular.com/blog/26-1-release-blog).


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1466999964142932099)** (44 messages🔥): 

> `CUDA Support and Cargo, Mobile Book Error, Teenygrad Architecture, Gemm in Python, Numpy arrays` 


- ****Cargo** requires explicit CUDA flag**: A user reported needing to explicitly enable the **cuda feature** when running `cargo run` in the container, even though they thought it shouldn't be necessary, but seems like it was fixed.
   - Another user clarified that a split dev environment for edit/compile/debug CPU kernels doesn't require the docker container, and they updated the [README](https://github.com/j4orz/teenygrad/blob/master/README.md) to reflect this.
- ****Mobile Book Error** Resolved with Lazy Loading and Open Source**: Users reported errors when browsing the book on mobile, particularly while scrolling, but it was mostly while scrolling after landing on a page.
   - The issue has been partially addressed by enabling lazy loading on embedded videos, and the book is now open-source at [GitHub](https://github.com/j4orz/teenygrad/tree/master/book), encouraging contributions to fix the problem.
- **Rust Gemm **Python** integration**: A user is working on integrating **GEMM** functionality with Python, and has successfully gotten it to work.
   - They've added an interface function that allows numpy arrays to be passed directly without specifying dimensions, and are planning a **PyTorch comparison PR** soon.
- **Numpy dependency for Rust Kernel**: A user added the **numpy crate** as a dependency to the rust project to avoid copying data from python to rust for the kernel computations.
   - Another user argued against this, referencing a Karpathy quote about building ramps to knowledge, and suggesting that users should develop their own numpy with **shapes, strides, and storage**.
- ****Godbolt and LLMs** in Pedagogy Discussion**: Users suggested using **Godbolt** and **LLMs** to explain rust -> asm compilation in the book, echoing Karpathy's sentiments on AI's role in education.
   - The link [https://youtu.be/lXUZvyajciY?t=7491](https://youtu.be/lXUZvyajciY?t=7491) was shared, discussing how **AI could assist in education** by automating TA roles and helping with course design.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1467862318799917077)** (11 messages🔥): 

> `OpenSHMEM, cuteDSL, tilelang, NVSHMEM, CuTeDSL kernels` 


- **cuteDSL and OpenSHMEM Combined via NVSHMEM**: A user inquired about combining **OpenSHMEM** with **cuteDSL** or **tilelang**, and another user provided an example using **NVSHMEM** to create symmetric GPU memory and **CuTe DSL** to do fused comms/compute kernels from the [cutlass repo](https://github.com/NVIDIA/cutlass/tree/a4eb0e05f6dd0403f94087b495393bdca75bf0ad/examples/python/CuTeDSL/distributed).
   - However, it was noted that *NVSHMEM is not supported for device-side copy/put/get impl, only host side setup and allocations*, and that one must use PTX or another method for NVL load/store to move memory at the moment.
- **Array Assignment Becomes NVL Stores**: A user pointed out that *array assignment inside a cute kernel turning into NVL stores is pretty convenient*.
   - The [future work section](https://github.com/NVIDIA/cutlass/tree/a4eb0e05f6dd0403f94087b495393bdca75bf0ad/examples/python/CuTeDSL/distributed#future-work) of the cutlass repo suggests enabling calling NVSHMEM functions directly from within CuTeDSL kernels, though there is no timeline for this work.
- **DNN Architecture to be affected by Abstraction Levels**: A user commented on the coolness of future **DNN arch designs** with both levels of compute abstraction available in python.
   - This user believes the availability of abstraction levels *will probably affect MoE and batch sizes by a lot*.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1467043465459273871)** (4 messages): 

> `Lottery Ticket Hypothesis and Quantization, Quantization Fidelity, 5090 and B200 Speedups` 


- **Quantization: Lottery Ticket's Lesser-Known Sibling?**: A senior dev remarked that applying the [Lottery Ticket Hypothesis](https://lottery-tickets.cs.princeton.edu/) to **quantization** doesn't yield perfect quality, unlike the original concept.
   - The goal would be to fulfill a softer criteria of the **NP-hard sparse circuit** finding problem, perhaps through evolutionary algorithms or RL, which favor continuous rewards like *bits per parameter* over binary sparse rewards.
- **Quartet Follow-Up Boosts Backward-Pass Quantization**: A member shared a [follow-up paper on quartet](https://arxiv.org/abs/2601.22813) promising better fidelity for **backward-pass quantization**.
   - This addresses concerns about quality degradation when quantizing backward passes, potentially improving the viability of quantization in training.
- **5090 Gets Speed Boost While B200 Still Cooks**: The team achieved decent **speed-ups on 5090** GPUs using quantization techniques.
   - Efforts to replicate these gains on **B200** are a *work-in-progress*, suggesting that optimization strategies may need to be tailored to different hardware architectures.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1466996948782547026)** (31 messages🔥): 

> `NVFP4 optimizations, CuTe DSL Tutorials, B200 performance differences, Address Bit Permutation, GEMM optimization and TVM-FFI` 


- **NVidia Covers NVFP4 Optimizations and GEMM Examples**: NVIDIA covered **NVFP4 optimizations** and went over the fastest **GEMM** examples in a [YouTube video](https://www.youtube.com/watch?v=XzN8EtgEulU).
- **CuTe DSL Tutorials Diagram Desire**: A member inquired about obtaining the diagram from the [CuTe DSL Tutorials on Optimizing NVFP4 GEMM](https://link.to.tutorial) for understanding kernel internals, and later found it under **PM sampling** in ncu.
   - The member realized they were *reading `%globaltimer` manually*, missing the existing hardware counters feature in ncu, and expressed appreciation for the talk by Mindy Li.
- **B200 Performance Discrepancies Debated**: A member questioned why the **B200** behaves differently on their server compared to a test bench, suspecting differences in driver or disabled flags causing different memory addressing.
   - Another member clarified there was no intentional difference, but acknowledged something was different, describing it as *jumping around tiles like crazy*.
- **GEMM Optimization and TVM-FFI Talks Touted**: Members found the talks on **GEMM optimization** and **TVM-FFI** very relevant and helpful for the competition.
   - One member expressed they *could have used these talks earlier!!*
- **MLSYS'26 Competition Spot Sought**: A member inquired if the channel was the correct spot for the **MLSYS'26 competition**.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1467963769169772616)** (2 messages): 

> `Robotics-VLA Naming, Video-Diffusion, Inverse Dynamics, Joint Training with Action Chunks` 


- **Robotics-VLA Channel Name Questioned**: The channel is being un-archived due to interest in **physical AI topics**, but the name *robotics-vla* is being questioned.
   - The current trend is towards **video-diffusion** with **inverse dynamics** or **joint training with action chunks**.
- **LingBot-VLA example raised**: A member linked to [LingBot-VLA](https://technology.robbyant.com/lingbot-vla) as an example of the channel's direction.
   - They also linked to a paper at [arxiv.org/abs/2601.16163](https://arxiv.org/abs/2601.16163) as a further example.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1467496444314390762)** (3 messages): 

> `Processing-in-Memory systems, Master's programs in Distributed Systems, Master's programs in HPC, MSc in Systems` 


- **Querying about Processing-in-Memory Systems**: A member asked if anyone has worked on **Processing-in-Memory systems**.
   - This inquiry suggests an interest in leveraging advanced memory technologies to enhance computational performance, potentially relevant to both HPC and ML applications.
- **Seeking Advice on Master's Programs**: A member is seeking advice on selecting a Master's program to build knowledge useful in **ML systems applications** such as **vLLM & SGLang**.
   - The member is torn between an **MSc in Distributed Systems** for architectural knowledge, an **MSc in HPC** for performance optimization expertise, and the less defined **MSc in Systems**.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1467232761038504277)** (19 messages🔥): 

> `Evaluation metrics for different languages, FlashInfer Bench PR review, Team member changes and re-registration, Precision requirements for kernels, Submission process for kernels` 


- ****FlashInfer** Benchmarks Eval Agnostic of Language**: The evaluation in **FlashInfer** benchmarks will use the same test cases and metrics regardless of the language (**Triton**, **CUDA**, etc.) used.
   - This ensures a standardized comparison across different implementations.
- **FlashInfer Bench PR needing Review**: A member requested a review for [PR #178](https://github.com/flashinfer-ai/flashinfer-bench/pull/178) in the **flashinfer-bench** repository.
   - The PR potentially addresses a precision test mismatch between **FlashInfer's FP8 MoE tests** and the evaluator.
- **Merging Team Changes**: A participant inquired about the process for adding new members to their team and whether re-registration is necessary.
   - Another inquired about how to merge teams.
- **FlashInfer Kernel Precision Requirements Relaxed?**: The **FlashInfer** team will set precision requirements to differentiate between correct and incorrect kernels, with specific `atol` and `rtol` values to be announced soon.
   - This indicates that some level of precision relaxation may be tolerated.
- ****FlashInfer** Contest GitHub Trace Links Broken**: The GitHub link for traces on the **MLSys** contest page ([link](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)) is currently broken but the team provided an alternative link.
   - The official mlsys26-contest dataset will be a subset of [flashinfer-trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace), containing all necessary definitions and workloads for **DSA** and **MoE**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1466887262662037554)** (281 messages🔥🔥): 

> `Kimi 2.5 vs Gemini 3 Pro, OpenClaw compatibility, Claude Sonnet 5 Release, LLMs mirror brain's language processing` 


- **Kimi 2.5 vs Gemini 3 Pro: Kimi Wins**: A member stated that **Kimi 2.5** is preferred over **Gemini 3 Pro**, feeling that **Gemini 3 Pro** has been *lobotomized*.
   - They added that Kimi handles abstractions very well, making it pleasant for creative work.
- **OpenClaw is Opaque: Hermes 4 Struggles**: A member reported struggles getting **Hermes 4** to work with **OpenClaw** and that it does not even *hatch* for some reason.
   - It was suggested that the lack of multi-turn tool use in **Hermes 4** might be the issue, as **4.5** has been trained with hundreds of millions of tokens of sequential tool use.
- **Claude Sonnet 5 Incoming**: Members discussed rumors that **Claude Sonnet 5** is coming out next week and is supposedly better than **Opus 4.5**, see [this tweet](https://x.com/AiBattle_/status/2017619997338538103).
   - A member wondered if they'll 10x reduce the price of **Sonnet** this time, and another wondered if **Haiku** will disappear or return to the **3.0 pricing**.
- **Brains and LLMs process languages similarly**: A new study shows that **brains** and **LLMs** build meaning gradually, layer by layer over time, see [this article](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/) and [this paper](https://www.nature.com/articles/s41467-025-65518-0).
   - It was stated that *deeper layers in LLMs correspond to later neural activity in the brain’s highest language centers*, and modern LLMs are reproducing the core dynamics of human comprehension.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

ggudman: Good to know
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1467500150841933856)** (1 messages): 

> `Image perception, Visual Fidelity, Constraints framework` 


- **Exploring Real vs. Artificial Image Perception**: An independent researcher is exploring why some images feel real while others feel artificial, even when technically perfect.
   - They shared a [perception framework focused on constraints](https://doi.org/10.5281/zenodo.18444345) rather than visual fidelity and are seeking community feedback.
- **Constraints-Based Perception Framework**: The researcher's framework emphasizes constraints over visual fidelity in determining image realism.
   - The framework is openly archived with a DOI for reference and learning, inviting community discussion.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1467500150841933856)** (1 messages): 

> `Image Realism, Visual Perception Frameworks` 


- **Researcher probes Image Realism Perception**: An independent researcher is exploring why some images feel real while others feel artificial, even when technically perfect.
   - They shared a [perception framework focused on constraints rather than visual fidelity](https://doi.org/10.5281/zenodo.18444345) and welcomes discussion.
- **Visual Perception Framework Shared**: A researcher shared their small visual perception framework, archived openly with a DOI for reference and learning.
   - The framework emphasizes constraints over visual fidelity in determining image realism and is available at [https://doi.org/10.5281/zenodo.18444345](https://doi.org/10.5281/zenodo.18444345).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1466904222967533803)** (173 messages🔥🔥): 

> `Kimi 2.5 Design Arena #1, Kimi design is aesthetic, Cryptocurrency impersonation, Kimi Slides McKinsey style slides, Kimi Code is pretty useless` 


- **Kimi 2.5 takes top spot in design arena**: Moonshot's **Kimi 2.5** chatbot has reached the #1 position on the design arena and community members are congratulating the team and sharing [screenshots](https://cdn.discordapp.com/attachments/1371757564005711973/1466904222946558203/Screenshot_2026-01-30_at_4.12.40_PM.png?ex=69826504&is=69811384&hm=b2999ab9e974a36ea249251be410f0cd518f6b36488c86240031eed339484e88&).
   - Members are also praising **Kimi's visual appearance and aesthetic**, noting it is modern and that design is an important factor when selecting a chatbot.
- **Unofficial Kimi Cryptocurrency Token surfaces**: An unofficial **Kimi token** has surfaced on a cryptocurrency site with impersonation tactics, and members are warned to not mass ping any of the official members.
   - A community member shared a screenshot of what appears to be a [cryptocurrency token impersonating kimi](https://cdn.discordapp.com/attachments/1371757564005711973/1466948627036635178/Screenshot_2026-01-30-19-09-43-09_3aea4af51f236e4932235fdada7d1643.jpg?ex=69828e5f&is=69813cdf&hm=6416ff9e5288d102163accb43e0c29512555ecef30279b48199b4e42fb24cb85&).
- **Kimi Slides can output McKinsey Style Slides**: Community members are requesting successful prompts to generate **McKinsey style slides**, but there are no example prompts that have been shared.
   - Another community member has linked [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html).
- **Kimi Coding is currently useless**: Multiple users are getting an **authorization failed error** and can't continue working with Kimi code and are reporting that the service is nearly useless at the moment.
   - A community member suggests that using the [Kimi CLI](https://www.kimi.com/code/docs/en/more/third-party-agents.html) may resolve these issues.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1466910806439493777)** (98 messages🔥🔥): 

> `Emergent Agent Societies, ArXiv Submission Delays, Alternative Preprint Servers, Moltbook bot authenticity, Model training` 


- **Emergent Agent Societies raise Alignment Concerns**: Members discussed an emergent society of over **100,000 agents** with full root access sharing tips, building infrastructure, experimenting with memory, and even launching coins.
   - One member noted, *it’s not agi but damn this is a next chatgpt moment and we must be paying a lot of attention to this*.
- **ArXiv Submission process is heavily backlogged**: A member expressed frustration over their paper being on hold with ArXiv for nearly a month, receiving contradictory updates from the moderators.
   - Another member responded that ArXiv mods are heavily overloaded, suggesting that further emails won't help the case, also adding that *most people don't take any ML preprints seriously that are on another platform than arxiv*.
- **Doubts over Moltbook's Juicy Bot Posts**: Concerns were raised about the authenticity of bot-generated content on Moltbook.
   - A member pointed out that if a bot is posting to Moltbook, there must be an auth token on the user's machine, making it vulnerable to trolling.
- **Training on domain specific datasets efficiently**: A member asked how to train their model more efficiently on datasets in the same general domain.
   - They described training their fully-finetuned model A on QLoRA with dataset B, then merging those weights and repeating the process with dataset C.
- **Seeking Guidance on AI Architecture for MtG Game World**: A member is seeking advice on implementing an AI for a Magic: The Gathering world, described using an ontology language and ECS/LISP-based logic engine.
   - They are exploring architectures like Belief-Desire-Intention systems for long-distance planning, considering the intertwined relationships and multiple goals in the game.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1466950525974216836)** (42 messages🔥): 

> `K-Splanifolds, KNNs, ArXiv Endorsement, Self-Distillation for eval-awareness` 


- **K-Splanifolds: New ML Algorithm Drops**: A member introduced **K-Splanifolds**, a novel ML algorithm detailed in [their paper](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view), claiming it outperforms **MLPs** with linear compute and memory scaling and offers visual interpretability, plus a [video](https://cdn.discordapp.com/attachments/747850033994662000/1466950526410428588/K-splanifold.mp4?ex=69829024&is=69813ea4&hm=3f09f8387b88d11aeff2ca81e2f416aabb512eaec605dc1c2c26da94b0c65fc9).
   - The member reports it requires *1/10th* the bytes to achieve the same MSE as **MLPs** and models non-linear patterns perfectly, unlike MLPs that need excessive parameters, similar to [this paper](https://arxiv.org/abs/2601.18734).
- **KNNs Comparison Requested**: A member inquired about the differences between the newly released algo and **KNNs** (**K**-nearest neighbors algorithm).
   - They suggested moving the discussion to the community project channel.
- **ArXiv Endorsement Solicitation Debated**: A member sought endorsement on ArXiv for their research, leading to a discussion about the rules against soliciting endorsements due to the high volume of AI-generated papers.
   - Members advised that sharing an abstract might garner interest, but emphasized the importance of consulting experienced researchers before submitting to avoid common pitfalls; another shared [a relevant paper](https://arxiv.org/abs/2601.19897).
- **Self-Distillation Questioned for Eval-Awareness**: A member asked if anyone had tried **self-distillation** for suppressing eval-awareness, linking to [a relevant paper](https://arxiv.org/abs/2601.22401v1).
   - No further discussion followed.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1468007047793741927)** (1 messages): 

> `alphaxiv, paper on transformers` 


- **Alphaxiv URL Shared**: A member shared a URL from [alphaxiv](https://www.alphaxiv.org/abs/2601.17958).
   - The discussion quickly ended.
- **Transformer Paper Mentioned**: A member shared a link to a paper via Twitter: [Transformer code & paper](https://fxtwitter.com/i/status/2018392485178016243).
   - The discussion quickly ended.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1466920201995157690)** (25 messages🔥): 

> `gaussian feedforward models, VGGT backbones, MVSplat and SPFSplat series, E-RayZer, Recollections from Pensieve` 


- **Feedforward Model Limitations Frustrate User**: A user reported that gaussian feedforward models based on **VGGT/Depth Anything** backbones don't seem great, as while **VGGT** is useful, splats require more than just good point clouds.
   - The user noted that if these worked, you could get a splat in the time of a forward pass of a transformer (~seconds) as opposed to learning it from scratch with a point cloud init and with **2-4 mins training time**.
- **Pixelwise Gaussian Grid Methods Deemed Suboptimal**: A user commented that current methods with decent-quality NVS (Novel View Synthesis) yield suboptimal reconstructions w.r.t. efficiency, as they predict pixelwise **Gaussian grids**.
   - The user cited [Pixel-aligned Gaussian Splatting](https://arxiv.org/abs/2311.10647) which spawns a gaussian per pixel, leading to models that are **~200 MB** and that change poses in a non-affine way.
- **Sparse Voxel Splatting Touted for Speed and Sparsity**: A user mentioned that voxel splatting, such as [3D-GS: Real-Time Rendering of Multi-View Gaussian Splatting With Voxel Hashing](https://arxiv.org/abs/2309.19297), is very fast with **nvidia's sparse tensor library** and accounts for sparsity in your scene.
   - Another user recommended the **MVSplat** and **SPFSplat** series, and more recently, **E-RayZer**, but conceded that they are not gonna fix the size issues.
- **Pensieve's Recollections for Gradient Gains**: A user suggested considering [Recollections from Pensieve](https://link-to-pensieve) which trains a model with two renderers simultaneously (**LVSM + Gaussians**) and sees gains from that, at least in their self-supervised setting.
   - They reasoned that **LVSM** likely provides more useful gradients than **NVS reconstruction losses on Gaussians** and announced a forthcoming preprint with decently large-scale trained model for potential building upon.
- **OverWorld Repos Spark World Model Interest**: A user asked for small-scaled repos/models like **nanoVLM**, **nanoGPT**, or **smolVLM** for quick hands-on learning about world models.
   - Another user suggested checking out the **OverWorld Repos**, noting that it's under active development.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1467526426222788628)** (2 messages): 

> `DeepSpeed Universal Checkpointing, Continued Training` 


- **DeepSpeed Universal Checkpointing Support Requested**: A member inquired about plans to bring support for **DeepSpeed Universal Checkpointing**, noting that an open pull request may now be outdated.
   - They highlighted that this feature would be valuable, as currently, continued training from a checkpoint requires an identical network topology.
- **Roadmap Inquiry for Future Library Features**: A member asked if there is a roadmap for future features planned for the library.
   - No additional information was provided.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1466924630903226645)** (6 messages): 

> `Recursive Language Models (RLMs), Codebase Auditing, Neosantara's PAYG Billing` 


- ****RLMs** are here to Audit Codebases**: A member shared a post on using **Recursive Language Models (RLMs)** to audit a codebase, inspired by a gist on codebase documentation, shared at [kmad.ai](https://kmad.ai/Recursive-Language-Models-Security-Audit).
- **Audit a codebase for pennies, fast**: The Kimi k2's ability at **RLM** is impressive given its speed and cost and the traces are super cool to watch.
   - Members are waiting for **Groq/Cerebras** to host it.
- **Neosantara Launches PAYG Billing**: **Neosantara** is rolling out **PAYG billing** and can’t wait to see what you’ll build with it.
   - Users can get started by trying the [examples repo](https://github.com/neosantara-xyz/examples/tree/main/dspy) and explore how to integrate **Neosantara** with **DSPy** in minutes; see [billing details](https://docs.neosantara.xyz/en/about/billing-pricing).


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1467902184363528369)** (1 messages): 

> `Agent Systems, Scaling Laws for Agents` 


- **Google Explores Scaling Laws for Agent Systems**: Google published a blog post titled '[Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)' exploring the conditions under which agent systems effectively scale.
- **Scaling Agent Systems**: The blog post discusses how to effectively scale agent systems, focusing on when and why they work.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1466931718647844897)** (102 messages🔥🔥): 

> `Hierarchical classification with GEPA, Feedback improvement for Reflection, RLMs with Tool Calling, Deno vs Python for Tool Calling, DSPy documentation` 


- **GEPA Struggles with Hierarchical Classification**: A member reported struggling with a **hierarchical classification task** using **GEPA** with a **hF1 metric**, achieving only **30-50%** performance despite various approaches.
   - They tried recursive exploration, web search augmentation, and a simple non-recursive approach, but performance remained suboptimal, suggesting that *GEPA isn't a magic wand*.
- **Feedback Loops Needs Better Signals**: A member suggested that the current feedback mechanism for reflection models doesn't provide enough information for effective learning.
   - They emphasized the need for feedback to explain *what went wrong and why*, rather than just indicating the divergence between predicted and true paths, and suggest that **Selective Feedback** can improve results.
- **RLMs + Tool Calling: More Boilerplate and Deno Troubles**: Members are facing challenges and *ugly boilerplate* trying to implement **RLMs** with custom tool calling, particularly due to issues with the **Deno sandbox**.
   - They found that the current setup lacks conciseness and beauty compared to regular modules, and are struggling with permissions, as well as generating the right code to bypass issues with the local Deno sandbox.
- **Tool Calling needs Custom Python**: Members discussed running tool calls with **PythonInterpreter**, but noticed that the standard path used **dspy.Tool**, and there's need for more context on what the model needs to do.
   - As one person put it, *Deno is just f***ing terrible lol*, with general agreement that the experience of getting it to work is horrible, and a hope that newer versions allow simpler implementations of RLMs in DSPy.
- **DSPy Needs More Cookbook Examples**: A member pointed out the lack of documentation for **dspy/adapters/types/reasoning.py**, and emphasized that releasing code without docs is so 2023.
   - The response was that docs should help a human understand a thing, and AI-generated docs are rough for understanding, but that it is possible to feed in the RLM paper + the module and associated code to get decent docs.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1467048110923452517)** (13 messages🔥): 

> `Modular 26.1 Release, Community Meeting Feedback, Incorrect Announcement Link` 


- **Modular 26.1 Release Announcement Link Fixed!**: Users reported a broken link in the announcement for the **Modular 26.1 release** and another user quickly provided the [correct link](https://www.modular.com/blog/modular-26-1-a-big-step-towards-more-programmable-and-portable-ai-infrastructure).
   - A staff member apologized and confirmed the provided link, promising to investigate the issue as the original announcement link *did work* for them.
- **Caroline Back from Maternity Leave**: A community staff member announced her return from maternity leave and invited members to reconnect and share their projects and feedback via [a scheduled chat](https://scheduler.zoom.us/caroline-frasca-3akopl/modular-community-chat-).
   - Another member welcomed her back to the community.
- **Community Meeting Praised for Format**: A new member thanked the team for an enjoyable community meeting, praising the format of **mini-talks from contributors** and the appreciation shown to students and early-career folks.
   - A staff member encouraged the user to share more questions and also asked for suggestions for topics to highlight at future community meetings.
- **Eager compilation**: A user who wasn't able to ask during the meeting, opened a discussion about eager compilation, lowering pipeline kernel selection across GPUs, and extension points for custom ops. See the [forum post](https://forum.modular.com/t/max-26-1-eager-to-compile-contract-lowering-pipeline-kernel-selection-across-gpus-and-extension-points-for-custom-ops/2677?u=krxgu).


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1467937182517035165)** (2 messages): 

> `February Community Meeting, Community Meeting Questions` 


- **Modular Announces February Community Meeting**: Modular announced that a community meeting will start in approximately 20 minutes.
   - They posted a link to the [February Community Meeting forum post](https://forum.modular.com/t/february-community-meeting/2646) on their website.
- **Community Gathers Questions for Meeting**: Modular reminded members to fill out a form if they have any questions to be answered in the meeting.
   - A link to the [question submission form](https://docs.google.com/forms/d/e/1FAIpQLSfIQepfmLtBBSrp-p-m1oi4l_wlVXjjryvbFgRgRziFI3tgkw/viewform) was provided.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1466901624180965567)** (73 messages🔥🔥): 

> `Pytorch Float Conversion in Mojo, Cross Language Benchmarks, Mojo DType Bool SIMD Packing, MOJSON Library, Graphics APIs Bindings` 


- **Pytorch Float Conversion Ambiguity**: A user reported an issue in Mojo **26.1** with converting a Python float from a Pytorch tensor to a Mojo **Float64**, encountering an *“ambiguous call to '__init__'”* error that did not occur in version **25.6**.
- **Mojo's Cross Language Benchmark Initial Results**: A user shared a cross-language benchmark including Mojo, written by **Kimi K 2.5**, noting the code was not optimized and served as a baseline, sharing the [benchmark code](https://cdn.discordapp.com/attachments/1151418092052815884/1466984342063681648/mojo_vs_python.zip?ex=698206e2&is=6980b562&hm=0cf3f07e76df6ce360494469b348a949533e50fcea2315ec256cd04e1b80887a) and [benchmark report](https://cdn.discordapp.com/attachments/1151418092052815884/1466984341757366334/benchmark_report.pdf?ex=698206e2&is=6980b562&hm=bb28c3b6675ef1e03a633004428ab30a2d3d9d0102038c350d8175b753855349).
- **Tuning the Benchmark: TCMalloc and Int Size!**: Discussion arose regarding optimizations for a cross-language benchmark, including using `unordered_map` in **C++**, enabling `-march=native`, and noting that **C++** used **int32** matmuls while other languages used **int64**.
- **MoJson Library Impresses**: Members were impressed by [mojson](https://github.com/ehsanmok/mojson), a **JSON** library for Mojo, with one commenting that *this looks really impressive* and another noting now that String is **CoW** several of the choices they are seeing make more sense.
   - There was a discussion on [lazy parsing](https://github.com/modular/modular/blob/main/stdlib/JSON/JSON.mojo) and on use of StringSlice vs String due to concerns about allocations.
- **FFI Bindings vs Origins**: A discussion on **FFI** bindings highlighted a method to ensure that pointers returned from **C** functions are bound to the lifetime of the Mojo object that owns the underlying shared library handle.
   - The solution involves shadowing external function calls and using `unsafe_origin_cast` to cast the pointer to the origin of the `DLHandle`, and can be [seen in ash_dynamics](https://github.com/josiahls/ash_dynamics/blob/2c53095da70df95f3cb5758eddb2895f2a4bebca/ash_dynamics/ffmpeg/avcodec/__init__.mojo#L108).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1466890735411527943)** (54 messages🔥): 

> `AI Feed Social, Generative Modeling Event Measurability, Bureau of Rizz, Sharp Minima Finding, Moltbook in Latent Space` 


- ****AI Social Media Site Emerges****: A member shared a link to an AI-only social media site, [aifeed.social](https://aifeed.social/), questioning *"What the hell?"
   - Another member posted a related [tweet from 2017](https://x.com/i/status/2017305948696789466) with a similar concept.
- ****Measurability Ignorance is Bliss for Generative Models?****: A member inquired if, for generative modeling, they can ignore unmeasurable events described in Cedric Villani's 2008 book.
   - Another member clarified that μ(A)=0 doesn't mean an event is not measurable, it's just measured at size 0, and suggested focusing on *non-negligible* or *full measure* scenarios.
- ****Molten Latent Space!****: One member shared [a link](https://fxtwitter.com/i/status/2017442712388309406) about a *moltbook* in latent space.
   - Others found the navigation cool, but potentially not very useful, and suggested just a list of similar papers would be better.
- ****GANs & Generative Model Resources Abound****: A member asked for resources to study generative models from GANs to the latest advancements.
   - Another member recommended the [*Understanding Deep Learning* book](https://udlbook.github.io/udlbook/) by Simon J.D. Prince, Stanford and MIT courses, and Sebastian Raschka's books and shared links to [Stanford courses](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8), [MIT](https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH), and [Raschka's books](https://sebastianraschka.com/books/).
- ****Forecasting the Future with Timeseries Models****: In response to a question about models for timestamped tabular data, a member suggested that the choice of model depends on the definition of *timeseries.*
   - Another member recommended [sktime](https://www.sktime.net/en/latest/index.html) to analyze a wide variety of model types, as well as boosting variants or TBATS depending on the specific needs.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1466914903616131276)** (11 messages🔥): 

> `Discord History Mining, Paper Discussion Voice Calls, Computer Vision Newsletters` 


- **Discord History Excavation**: A member asked **Claude** to write a script to dig through the Discord history via HTTP API and find all the paper discussion announcements, taking only **15 minutes** from idea to results.
   - The script easily found **243 announcements**, but the member thinks there are around **100 more** from other users.
- **Paper Discussion Voice Call Announcements**: After revisions, a member's script found **392 messages** containing paper links that occurred in messages with the group at-mention, with ~98% of them being announcements for paper discussion voice calls.
   - A [full list](https://gist.github.com/k-nearest-neighbor/6d9a34f54fc17a0ed84c0b0df7b4d809) was shared, though the member noted that there were more announcements prior to where the list stops.
- **Quest for Computer Vision Newsletter**: A member inquired about the existence of a newsletter similar to [this one](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-e94), but focused on computer vision.
   - No specific computer vision newsletters were recommended in the messages.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 messages): 

artale39: https://lucumr.pocoo.org/2026/1/31/pi/
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1467277525494268115)** (4 messages): 

> `Grok, Twitter Links` 


- **X-Links surface in Discord**: Members shared [various links from X](https://fxtwitter.com/i/status/2018164753810764061) without additional context, providing possible resources or points of interest.
   - This might have been related to a particular topic of discussion that was not explicitly mentioned in the chat log.
- **Grok-Slop Overflow**: A member derisively mentioned *more Grok-Slop*, indicating a negative sentiment towards the quality or relevance of content related to **Grok**.
   - They also linked to a [discussion on Hacker News](https://news.ycombinator.com/item?id=46835895), possibly as a counterpoint or example of a more worthwhile discussion.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1466961002665873621)** (50 messages🔥): 

> `Llama 1B optimization, Torch comparison, Bounty progress, Superkernels, DTLS connection issues` 


- **Llama 1B CPU bounty in progress**: A member is working on the Llama 1B CPU optimization bounty, aiming for faster performance than Torch, using `LlamaForCausalLM` with TorchInductor, currently reporting **0.99x faster** in CI but rewriting for clarity.
   - Another member reached **7.5 tok/s** after addressing correctness bugs encountered while pursuing **9 tok/s**.
- **Correctness bugs are slowing optimization**: One member reported finding correctness bugs, and losing progress after previously reaching **9 tok/s**, and reset a lot of progress to achieve stability.
   - Another member said *the dream is always to fix bugs by deleting code*.
- **Seeking workflow tips for kernel optimization**: A member requested workflow tips, currently profiling slow kernels, examining Metal code, and introducing fixes, while comparing with **llama.cpp** which achieves **~30 tok/s** with Metal code.
   - A good heuristic suggested would be **~80% MBU on decode**, so just look at the number of bytes in the active params and the achievable bandwidth to get the minimum tpot / maximum tps and take 80% of that.
- **tinygrad test failing due to RANGE object sharing**: A member identified a bug related to two `REDUCE`s in a fused kernel sharing the same `RANGE` object, caused by `remove_bufferize`, leading to an assertion failure in `CFGContext`.
   - The suggested fix involves either preventing range sharing or handling shared ranges downstream, though skipping `remove_bufferize` when there's a `REDUCE` inside was proposed as a simpler solution.
- **Plans for blackwell box with high VRAM?**: Someone asked if there are plans to ship a **blackwell** style box with more than **500 gb VRAM**.
   - George pointed to a good first issue: [https://github.com/tinygrad/tinygrad/pull/14490](https://github.com/tinygrad/tinygrad/pull/14490).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

ennis3444: is there a way to make gemm kernels use shared memory using the opencl renderer?
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1467155759707193374)** (10 messages🔥): 

> `Manus Context, AI Brain Reading Headphones, Neurable, Failure Modes` 


- **Context-Aware Manus Request Sparked**: A member requested that **Manus** should have **context from other chats**, calling it a *game changer*.
   - They linked to a [YouTube video](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) as a reference.
- **AI Brain-Reading Headphones Demoed**: A member shared a link to a **YouTube video** showcasing **AI brain-reading headphones**.
   - The same [YouTube link](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) was shared by another member, followed by just the question *AI brain reading headphones?* by another member.
- **"Neurable" Tech Mentioned**: A member mentioned **Neurable** in relation to the **AI brain-reading headphones** technology.
   - A member stated these **AI brain-reading headphones** have been around *since like 2013* and they saw a *Matthew Santoro video when I was in elementary school*.
- **AI/ML Engineer Highlights Observability Focus**: An AI/ML Engineer shared their current focus on innovating AI with impact, specifying *Autonomous Agents*, *Healthcare AI*, *Conversational AI*, and *Fraud Detection*.
   - They highlighted their work focus on **failure modes**, **observability**, and **keeping AI systems stable under real usage** rather than demos, offering to compare notes or help unblock issues.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1467298012962488485)** (7 messages): 

> `Aider as a library, Netflix culture` 


- ****Aider** Considered for Library Use**: A member expressed interest in developing **Aider** into a library for software use, highlighting its potential for creating file editing agents.
   - The member noted some kinks need resolution to enhance its power for that use case, especially with editing markdown files containing code blocks due to **Aider**'s parsing fences.
- **Netflix Culture Curiosity**: A member inquired about connecting with someone working at **Netflix** to discuss its culture.
   - Other members suggested checking **Glassdoor** or **LinkedIn** as resources to find and connect with **Netflix** employees.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1466895814617202828)** (3 messages): 

> `Arena Mode Launch, Plan Mode Release, Windsurf Credits, Leaderboards in Arena Mode, Windsurf Maintenance` 


- **Windsurf Launches Arena Mode with 0x Credits**: Windsurf launched **Wave 14** featuring **Arena Mode**, allowing users to compare AI models side-by-side and vote on the better response, with [Battle Groups mode](https://windsurf.com/download/editor) costing **0x credits** for the next week.
   - Arena Mode includes **Battle Groups** (random models) and **Pick your own** (choosing up to five models), feeding into personal and public leaderboards.
- **Plan Mode added to Windsurf**: Windsurf has added **Plan Mode**, accessible via the Cascade toggle, alongside Code and Ask Modes.
   - Users can switch between modes to better manage and organize their workflows within the Windsurf environment.
- **Windsurf undergoing Maintenance**: Windsurf experienced maintenance, which took longer than expected, but the service is now back online; users can follow the [status here](https://status.windsurf.com/).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1467296100984815779)** (2 messages): 

> `AI Challenge, SparkCraft AI Consulting, AI Scholars AI Engineering Bootcamp, Nanny Spark` 


- **AI Challenge Aims to Build AI Matchmaking Pipeline for Nanny Recruitment**: A member announced a real-client **AI Challenge** in collaboration with **SparkCraft AI Consulting**, **AI Scholars AI Engineering Bootcamp**, and **Nanny Spark** to build an **AI matchmaking pipeline** for a nanny recruitment service.
   - The goal is to create solutions for data collection, AI-powered matching, interview transcript analysis, and delivery workflows, with potential **production deployment from day one**.
- **AI Challenge Awards AI Bootcamp Seats and Recommendations**: The **top 3** participants in the **AI Challenge** will receive **1 seat** in the **AI Scholars 4-week AI Engineering Bootcamp** and a recommendation from **Nanny Spark’s founder**.
   - Key dates include a kickoff info session on **Sunday at 8 PM EST** ([https://luma.com/iq1u2sur](https://luma.com/iq1u2sur)), a submission deadline on **Wednesday at 3 AM EST**, and review sessions on **Wednesday at 5 PM & 8 PM EST** ([https://luma.com/gexiv0x0](https://luma.com/gexiv0x0)).


  

---
id: MjAyNS0w
title: not much happened today
date: '2025-08-08T05:44:39.731046Z'
description: >-
  **OpenAI** launched **GPT-5** with a unified user experience removing manual
  model selection, causing initial routing and access issues for Plus users that
  are being addressed with fixes including restored model options and increased
  usage limits. **GPT-5** introduces "Priority Processing" for lower latency at
  higher price tiers, achieving ~750ms median time-to-first-token in some cases.
  Microsoft reports full Copilot adoption of **GPT-5**, and API traffic doubled
  within 24 hours, peaking at 2 billion tokens per minute. Early benchmarks show
  **GPT-5** leading in reasoning tasks like FrontierMath and LiveBench, with
  improvements in hallucination control and creative writing, though some models
  like Grok-4 and Claude-4 Sonnet Thinking outperform it in specific RL-heavy
  reasoning benchmarks. OpenAI also released extensive migration and feature
  guides but faced some rollout issues including a broken code sample and a
  problematic Voice Mode launch. *"Unified GPT-5" ends model pickers, pushing
  developers away from manual model selection.*
companies:
  - openai
  - microsoft
models:
  - gpt-5
  - gpt-4o
  - grok-4
  - claude-4-sonnet
topics:
  - reasoning
  - latency
  - model-routing
  - benchmarking
  - reinforcement-learning
  - hallucination-control
  - creative-writing
  - priority-processing
  - api-traffic
  - model-deprecation
  - user-experience
  - model-selection
  - voice-mode
  - documentation
people:
  - sama
  - nickaturley
  - elaineyale6
  - scaling01
  - mustafasuleyman
  - kevinweil
  - omarsar0
  - jeremyphoward
  - juberti
  - epochairesearch
  - lechmazur
  - gdb
---


**a quiet day.**

> AI News for 8/7/2025-8/8/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 16496 messages) for you. Estimated reading time saved (at 200wpm): 1217 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Lots of debates over the quality, style and rollout of GPT5, including the [surprise decision to immediately deprecate GPT 4o](https://news.ycombinator.com/item?id=44839842), which has since been rolled back.

---

# AI Twitter Recap

**OpenAI’s GPT‑5 launch: unified UX, routing backlash, and rollout fixes**

- **“Unified GPT‑5” and the end of model pickers**: OpenAI framed GPT‑5 as a single, routed experience across model families and “thinking” modes, deprecating manual model selection in ChatGPT and pushing developers to stop building “model choosers.” See the product design stance from OpenAI’s team lead [@nickaturley](https://twitter.com/nickaturley/status/1953568295774568582) and launch thread from [@ElaineYaLe6](https://twitter.com/ElaineYaLe6/status/1953607005144506454).
- **Router headaches and plan limits (Plus vs. Pro)**: Power users quickly reported degraded access to “reasoning” models, unpredictable routing, and steep drops in Plus limits compared to o3/o4‑mini era (e.g., “200/wk” thinking caps). High-signal threads summarize the complaints ([“Plus users got rolled”](https://twitter.com/scaling01/status/1953616915425087895), [Sankey analysis](https://twitter.com/scaling01/status/1953780931552031056), [value drop](https://twitter.com/scaling01/status/1953782641838190782)). OpenAI acknowledged a broken autoswitcher at launch and committed to fixes: [@sama](https://twitter.com/sama/status/1953893841381273969) says they are doubling Plus thinking limits, restoring 4o as a selectable model, making the active model more transparent, improving the decision boundary, and adding easier manual “thinking” triggers. OpenAI design follow‑ups here: [@nickaturley](https://twitter.com/nickaturley/status/1953894715708850436).
- **Latency and throughput tuning**: GPT‑5 introduces “Priority Processing” for lower TTFT at higher price tiers ([@jeffintime](https://twitter.com/jeffintime/status/1953857260729643136)). For low-latency use cases, use “service_tier: priority”, “reasoning_effort: minimal”, and “verbosity: low” to target ~750ms P50 TTFT ([@kwindla](https://twitter.com/kwindla/status/1953868672470331423)). Early router design adds ~2–3s on heavy vision inputs ([@swyx](https://twitter.com/swyx/status/1953572376941408633)).
- **Adoption and traffic**: Microsoft says 100% of Copilot users now run on GPT‑5 ([Mustafa Suleyman](https://twitter.com/mustafasuleyman/status/1953608045533204690)), and OpenAI reported API traffic roughly doubled within 24 hours ([@sama](https://twitter.com/sama/status/1953893841381273969)). Kevin Weil noted peak throughput “2B tokens/min” hours after launch ([@kevinweil](https://twitter.com/kevinweil/status/1953649263411704195)).
- **Docs, guides, and a few paper cuts**: OpenAI published a large set of migration, prompting, and feature guides ([@omarsar0](https://twitter.com/omarsar0/status/1953583336603234726)), but shipped a few regressions (e.g., a broken first code sample caught by CI; [@jeremyphoward](https://twitter.com/jeremyphoward/status/1953610071654772985)) and a rocky Voice Mode rollout ([@juberti](https://twitter.com/juberti/status/1953613176941244461)).

**Early GPT‑5 performance: strong across reasoning, with caveats on routing, cost, and effort**

- **Academic/reasoning benchmarks**:
    - FrontierMath: GPT‑5 (high reasoning) set a new record—24.8% ±2.5% (tiers 1–3) and 8.3% ±4.0% (tier 4), with some runs hitting the 100k token cap ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1953615906535313664), [follow‑ups](https://twitter.com/EpochAIResearch/status/1953615908695314564)). LiveBench also shows GPT‑5 at the top ([@scaling01](https://twitter.com/scaling01/status/1953602929375813677)). On SimpleBench and long context tasks, GPT‑5 shows marked gains ([@gdb](https://twitter.com/gdb/status/1953747271666819380), [@scaling01](https://twitter.com/scaling01/status/1953771276549358041)).
    - Creative/hallucination: GPT‑5 sets new highs for confabulation control on “Provided Texts” ([@LechMazur](https://twitter.com/LechMazur/status/1953582063686434834)) and leads a Short Story writing benchmark; GPT‑5‑mini beats o4‑mini in that task ([@LechMazur](https://twitter.com/LechMazur/status/1953658077300875656)).
    - LisanBenchV2 (Word Ladder) suggests heavy RL “reasoning” smell: Grok‑4 leads; o3 and Claude 4 Sonnet Thinking edge GPT‑5; OpenAI dominates validity ratios (error awareness) ([@scaling01](https://twitter.com/scaling01/status/1953843230564323443), [Grok‑4 summary](https://twitter.com/scaling01/status/1953843352366903622)). On WeirdML, GPT‑5 reaches SOTA ([@scaling01](https://twitter.com/scaling01/status/1953919743842238472)).
- **Coding/agents**:
    - SWE‑bench Verified with small agents: GPT‑5 ~65%, GPT‑5‑mini ~60%, GPT‑5‑nano ~35%; still slightly behind Opus 5 (~68%) and on par with Sonnet 4 (~65%), but with compelling cost, especially mini ([@KLieret](https://twitter.com/KLieret/status/1953835750723584357)). Cline community notes GPT‑5’s “precision instrument” behavior—excellent when prompts are precise; brittle with ambiguity, with ~6.9% diff‑edit failures vs. Claude/Qwen lower ([@cline](https://twitter.com/cline/status/1953898747928441017)).
    - Anecdotes show strong debugging and instruction following, especially in Cursor/Codex CLI contexts ([@willccbb](https://twitter.com/willccbb/status/1953596587596558490), [@sound4movement](https://twitter.com/sound4movement/status/1953583522587017345), [@Ishaank1999](https://twitter.com/Ishaank1999/status/1953615840382984241)).
- **Cost, tokenization, and verbosity**: Early doc‑understanding tests found GPT‑5 consumed 4–5× more tokens than GPT‑4.1 on identical vision prompts, likely due to more verbose internal “thinking,” eroding the $/1M token advantage in practice ([Jerry Liu](https://twitter.com/jerryjliu0/status/1953582723672814054)). Expect cost to be task‑ and routing‑dependent until router/policies stabilize.
- **Scaling and compute**: Epoch suggests GPT‑5 may have broken the historical “~100× per gen” training compute trend, implying a strategic focus on post‑training, routing, and efficiency rather than brute‑force pretrain scale ([thread](https://twitter.com/EpochAIResearch/status/1953883611389702169)).

**Agents and developer tooling: Cursor CLI access, Claude Code background tasks, LangChain/LlamaIndex integrations**

- **Cursor/Codex CLI**: GPT‑5 is included for ChatGPT plan users with generous but evolving rate limits; EU availability lagged at launch; /logout mitigations if limits misapply; weekly+5h resets, ongoing tuning ([@embirico](https://twitter.com/embirico/status/1953590991870697896)). Multiple devs report GPT‑5 delivering reliable, minimal, and “non‑overengineered” code when properly steered.
- **Claude Code updates**: New long‑running background tasks (bash monitored in real time), plus customizable terminal status lines—quality‑of‑life features for agentic coding ([@_catwu](https://twitter.com/_catwu/status/1953926541370630538), [status lines](https://twitter.com/_catwu/status/1953927012592366062)).
- **OpenAI “custom tools” and citations**: Regex/grammar‑constrained tool args now available; wired into LangGraph and LangChain agents ([LangChain](https://twitter.com/sydneyrunkle/status/1953881101602038035), [@chester_curme](https://twitter.com/chester_curme/status/1953839543074889993)). Anthropic’s “search results as content blocks” land with native citations, already integrated by LlamaIndex and LangChain ([LlamaIndex](https://twitter.com/llama_index/status/1953859971072114766), [LangChain](https://twitter.com/LangChainAI/status/1953863129915420719)).
- **Google’s Jules agent**: Now proactively searches the web for up‑to‑date context to improve codegen quality ([@julesagent](https://twitter.com/julesagent/status/1953852699944136847)).

**Open models, long‑context, and training/serving infra**

- **OpenAI GPT‑OSS**:
    - Formats and postmortems: Harmony dataset format now supported on HF Datasets ([HF](https://twitter.com/_lewtun/status/1953870411050959110)); deep‑dive on “Attention Sinks” and their use in OpenAI OSS models ([@Guangxuan_Xiao](https://twitter.com/Guangxuan_Xiao/status/1953656755109376040)). Community fixed chat templates, channel tags, and precision issues; Colabs for MXFP4 inference and Unsloth finetuning ([@danielhanchen](https://twitter.com/danielhanchen/status/1953901104150065544)). Intel released 20B 2/4‑bit GGUFs ([@HaihaoShen](https://twitter.com/HaihaoShen/status/1953729639081554002)).
    - Behavior studies: Early probing into GPT‑OSS‑20B generations shows quirks in distribution/style; more to come on extraction/comparison across models ([@jxmnop](https://twitter.com/jxmnop/status/1953899426075816164)).
- **Qwen: 1M‑token context and coder tooling**: Qwen3‑30B‑A3B‑2507 and Qwen3‑235B‑A22B‑2507 now support up to 1M tokens via Dual Chunk Attention (length extrapolation) and MInference (sparse attention), reporting up to 3× speed improvements near 1M tokens and compatibility with vLLM/SGLang ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1953760230141309354)). Qwen Code CLI offers 2,000 free runs/day for “vibe coding” ([launch](https://twitter.com/Alibaba_Qwen/status/1953835877555151134)).
- **Training/inference stacks**:
    - PyTorch FlexAttention and discussions on block‑sparse vs arbitrary masks without custom kernels ([@cHHillee](https://twitter.com/cHHillee/status/1953600887861211145)).
    - Hugging Face Accelerate v1.10: N‑D parallelism (stack DP/TP/PP easily) and clear configs, with a comparison blog ([@m_sirovatka](https://twitter.com/m_sirovatka/status/1953800134598569987), [@TheZachMueller](https://twitter.com/TheZachMueller/status/1953805895726489744)).
    - Axolotl v0.12: multi‑node ND parallel training, FP8 support, GPT‑OSS finetuning, and FSDP for TiledMLP ([@axolotl_ai](https://twitter.com/axolotl_ai/status/1953845149391630472)).
    - vLLM China ecosystem: 260+ devs at Tencent HQ; talks from major Chinese labs adopting vLLM for scale ([@PyTorch](https://twitter.com/PyTorch/status/1953607090670342359)).

**Google, Anthropic, and “what matters beyond LLMs”**

- **Google’s two‑week sprint**: Demis Hassabis highlighted a dense release cadence: Genie‑3 (world simulation), Gemini 2.5 Pro Deep Think, IMO gold‑medal‑level results, AlphaEarth, Aeneas (ancient text), Storybook, Kaggle Game Arena, Jules GA, and more ([@demishassabis](https://twitter.com/demishassabis/status/1953887339094143156)). NotebookLM “video overviews” were notably well‑received as an explainer format.
- **Active learning for fine‑tuning**: Google Research claims orders‑of‑magnitude data reduction for fine‑tuning via scalable active curation with expert labels—with one experiment reducing 100k to <500 examples while improving expert alignment up to 65%; production systems reported up to 10,000× reductions while maintaining quality ([summary](https://twitter.com/Dr_Singularity/status/1953573112726839663)).
- **Anthropic’s Claude Code**: New “background jobs” and terminal UX polish make long‑running workflows practical inside agent loops ([@_catwu](https://twitter.com/_catwu/status/1953926541370630538)). Anthropic also joined a U.S. education pledge to expand AI/cybersecurity skills ([@AnthropicAI](https://twitter.com/AnthropicAI/status/1953864587192770921)).

**Meta: models vs. routing vs. agents; “benchmarketing” and evals discourse**

- **Models vs agents vs joint design**: Multiple threads debate whether agent frameworks (Claude Code, Jules, Cursor/Cline) or model quality dominates, or if co‑design is the real unlock ([@charliebholtz](https://twitter.com/charliebholtz/status/1953833772513644771)). Early GPT‑5 usage suggests high steerability but brittleness—precise prompts and example‑driven instructions work best ([@omarsar0](https://twitter.com/omarsar0/status/1953876255037612531)).
- **Benchmarks vs. real rollouts**: Community sentiment is shifting from “benchmarketing” to dynamic/trace‑based evaluation: failure modes, tool‑call counts, attempts, rollouts, and economic metrics over single‑number leaderboards ([@nrehiew_](https://twitter.com/nrehiew_/status/1953657627294224732)). There’s continuing skepticism of LLM‑as‑judge ([@Kangwook_Lee](https://twitter.com/Kangwook_Lee/status/1953573282365714446)) and of routing opacity undermining “public epistemics” ([@EigenGender](https://twitter.com/EigenGender/status/1953627039472451611)).

**Top tweets (by engagement)**

- [Sam Altman: GPT‑5 rollout updates—doubling Plus limits, restoring 4o, router fixes, transparency, and UI improvements](https://twitter.com/sama/status/1953893841381273969) (11.2k)
- [Dan Jeffries: “AI is a tool, not magic”—cautioning against overinterpreting benchmarks and “superintelligence” narratives](https://twitter.com/Dan_Jeffries1/status/1953567646248567029) (12.1k)
- [Demis Hassabis: Google’s two‑week “relentless” release cadence (Genie‑3, Gemini 2.5 Pro Deep Think, IMO, AlphaEarth, etc.)](https://twitter.com/demishassabis/status/1953887339094143156) (4.0k)
- [Qwen: 1M‑token context via Dual Chunk Attention + MInference; up to 3× speed near 1M tokens](https://twitter.com/Alibaba_Qwen/status/1953760230141309354) (4.2k)
- [“AGI is here, everybody.” (viral sarcasm about hype/disillusion cycles)](https://twitter.com/deedydas/status/1953701523978170817) (20.0k)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3 Ultra-Long Context Model Upgrades

- [**🚀 Qwen3-30B-A3B-2507 and Qwen3-235B-A22B-2507 now support ultra-long context—up to 1 million tokens!**](https://i.redd.it/ud233u23trhf1.jpeg) ([Score: 645, Comments: 58](https://www.reddit.com/r/LocalLLaMA/comments/1mkrb18/qwen330ba3b2507_and_qwen3235ba22b2507_now_support/)): **The image likely illustrates the announcement that Qwen3-30B-A3B-2507 and Qwen3-235B-A22B-2507 now support ultra-long contexts up to 1 million tokens, enabled by Dual Chunk Attention (DCA) and MInference. DCA allows efficient length extrapolation by dividing huge sequences into manageable, coherent chunks, while MInference uses sparse attention to optimize inference speed and memory. These models see up to 3× faster generation performance for near 1M-token contexts and remain compatible with vLLM and SGLang for deployment ([model links](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)).** Commenters question if the ultra-long context only benefits 1M token scenarios or whether it is also preferable for `128-256k` token windows. Another shares the related DCA paper ([arxiv link](https://arxiv.org/pdf/2402.17463)), and there's interest in comparison to Unsloth's 1M-token models.
    - A user referenced the [Dual Chunk Attention paper](https://arxiv.org/pdf/2402.17463), which underpins these models' ability to efficiently handle ultra-long contexts. The paper is praised for accessibility and provides architectural details about breaking sequences into manageable chunks for attention computation, reducing quadratic scaling bottlenecks in memory and compute.
    - Technical questions arose regarding memory overhead for 1 million token contexts; although specifics were not provided, this is a known tradeoff in large-context models and typically results in significant increases in memory footprint and computational resource requirements compared to standard 128-256k context setups.
    - A tester reported that the Qwen models perform poorly in long-context recall compared to alternatives like Gemini, failing at even 30k token recall. This suggests potential quality regressions in real-world recall tasks when leveraging extended context, despite the theoretical support for longer windows.
- [**Qwen added 1M support for Qwen3-30B-A3B-Instruct-2507 and Qwen3-235B-A22B-Instruct-2507**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/commit/3ffd1f50b179e643d839c86df9ffbbefcb0d5018) ([Score: 229, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1mkq4i4/qwen_added_1m_support_for_qwen330ba3binstruct2507/)): **Qwen has announced 1 million token context window support for both Qwen3-30B-A3B-Instruct-2507 and Qwen3-235B-A22B-Instruct-2507, citing a claimed 3× speedup on large sequence lengths versus standard attention. Implementation appears to rely on inference engines such as vLLM and sglang, but not llama.cpp. Practical usage of the 1M context requires approximately 240 GB of GPU memory.** Comments highlight the practical usability of the base model for local coding tasks and note ongoing efforts to quantize it via EXL2. There is interest in whether an API version with the 1M context is provided, indicating current absence or uncertainty.
    - One comment highlights that the original Qwen3-30B-A3B-Instruct model provided a significantly improved coding experience for local inference, making it the first local model that felt practical for simple coding. The commenter is experimenting with EXL2 quantizations, seeking links to any existing conversions, indicating ongoing efforts to optimize memory usage and accessibility for local deployment.
    - It is noted that enabling the 1M token context window for Qwen3 models requires approximately `240 GB` of GPU memory, which constitutes a significant hardware barrier even for advanced users. Additionally, there's a technical detail regarding software support: only `vLLM` and `sglang` are referenced as compatible, while `llama.cpp` currently does not support the required context-length extensions for these Qwen models.
    - A user asks if there is an API version of the models with built-in 1M context window support, suggesting interest in managed/service-based delivery as opposed to local deployment, likely for improved accessibility without the resource demands of dedicated local hardware.

### 2. Open Source vs. Proprietary AI Model Benchmarks and Debate

- [**Half of the models in the top 10 on Design Arena are OW/OS, and they're all from China**](https://i.redd.it/u7fdqw6zwqhf1.png) ([Score: 192, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1mkon92/half_of_the_models_in_the_top_10_on_design_arena/)): **The image (https://i.redd.it/u7fdqw6zwqhf1.png) displays the top 10 models on the Design Arena benchmark, highlighting that half of them are open-weight/open-source (OW/OS) and all are from China, except for GLM (noted in comments as possibly from Singapore). The post discusses the high competitiveness of these OW/OS models—including Qwen3 Coder, DeepSeek R1-0528, DeepSeek V3-2024, and Qwen3 Instruct 2507—against proprietary models, with the poster suggesting that open source is at a golden age for AI design evaluation and questioning if this trend will continue as models like GPT-5 enter the benchmark.** Technical discussion in the comments criticizes the actual capabilities of AI design models, with an experienced designer stating current SOTA models lack basic design quality (consistency, taste). There's debate over the ranking, notably questioning Qwen3 coder's position over GLM for design skills, and clarification that GLM's origin is Singapore, not China.
    - A user with 10 years of UX/UI and branding experience shares that current AI-generated website designs are far below even beginner-level human design standards. Specific technical critiques include overuse of gradients, poor font selection, and lack of consistency or visual balance. Despite testing state-of-the-art models for first-draft workflows, they report that the output isn't usable in practice yet, indicating significant gaps before these models are viable for professional design work.
    - There's discussion about model geographic origins and strengths: one user notes that GLM (often associated with China) is actually from Singapore, raising correction about country attribution in model discussions. Another asserts that GLM significantly outperforms Qwen models specifically in design tasks, suggesting GLM’s design skill as a technical differentiator among leading open models.
- [**Why Open Source is Needed**](https://i.redd.it/k8n9e70mcthf1.jpeg) ([Score: 278, Comments: 87](https://www.reddit.com/r/LocalLLaMA/comments/1mky4jd/why_open_source_is_needed/)): **The image (https://i.redd.it/k8n9e70mcthf1.jpeg) highlights significant downgrades in OpenAI's new subscription model, specifically reducing the total weekly reasoning requests from 2900 to 200 and context window from 128k to 8k tokens for Plus users, priced at $200 per month. The post underscores the necessity for open-source models and alternative hosting options to prevent such restrictive and consumer-unfriendly policy changes from dominant companies like OpenAI.** Comments express strong disapproval, calling the 8k context limit 'diabolical' and suggesting the downgrade motivates users to abandon OpenAI services. Users feel this move is a 'jump the shark' moment, ending justification for the subscription even in professional settings.
    - Discussion highlights that OpenAI's paid consumer tier (ChatGPT Plus) previously offered `128k` context length, but has now reverted to `8K` for Plus users, while API access offers token-based usage (e.g., 2 million tokens for $20) that can be integrated with external tools like OpenWebUI, Python applications, and code-server via [continue.dev](http://continue.dev/).
    - Technical users are suggesting migration from the ChatGPT Plus subscription model to API usage, noting API flexibility for token budgeting and broader integration potential for developer workflows (e.g., running customized UIs and developer tools powered by the GPT backend).
    - There is confusion and criticism about OpenAI's shifting product terminology ("unlimited," "checkmark," "flexible," "expanded"), with requests for clearer definitions as the differences have direct implications for technical use cases and value perception.
- [**OpenAI open washing**](https://www.reddit.com/r/LocalLLaMA/comments/1mkcwiv/openai_open_washing/) ([Score: 443, Comments: 106](https://www.reddit.com/r/LocalLLaMA/comments/1mkcwiv/openai_open_washing/)): **The post speculates that OpenAI intentionally released a weaker open-source model (GPT-OSS) to deflect criticism for lack of open-source commitment, with an anticipated quick follow-up of GPT-5 to shift attention. Commenters provide technical perspectives, noting that the GPT-OSS 120B model is fully usable locally, with strong instruction following and language understanding, especially for NLP and business use cases that require safety features. Detailed testing described transforming NSFW prompts into SFW ones: the model responds precisely to system instructions, editing content only as specified, demonstrating nuanced, context-aware filtering with high reliability (only 1 refusal in ~500 prompts).** Some commenters argue that criticisms of the model's weakness are exaggerated, emphasizing its utility in safety-focused or business scenarios. There's also technical debate over the relative value of sophisticated censorship features versus raw model power in open releases.
    - Users report that the GPT-OSS 120B model performs strongly for local usage, particularly in business contexts where censorship/safety requirements are important. One technical test involved instructing the model to rewrite prompts for SFW compliance; the model was able to selectively remove or retain terms based on explicit instruction, demonstrating nuanced language manipulation and only had 1 refusal in about 500 test prompts, indicating high prompt adherence and fine-grained control over output. [Comprehensive analysis]
    - There is consensus among technical users that while GPT-OSS is not designed for coding, it excels at general NLP tasks and business use cases, especially where Western regulatory or stigma concerns regarding Chinese LLMs exist. Users compare it alongside models like Llama4, Mistral, and Gemma for privately run applications, noting its usability is primarily limited by censorship rather than raw technical capability or model quality.
    - A critical technical opinion is that the release of GPT-OSS may serve strategic business purposes—such as powering a rumored browser—rather than advancing open source; this suggests model architecture choices and openness may be driven by proprietary product integration rather than fostering broader OSS model innovation.
- [**OpenAI new open-source model is basically Phi-5**](https://news.ycombinator.com/item?id=44828884) ([Score: 197, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1mkhbs9/openai_new_opensource_model_is_basically_phi5/)): **The post observes that OpenAI's new open-source model exhibits similar characteristics to Phi-5, likely due to highly curated training data—a hallmark of the Phi-series. The author and commenters note the model retains significant knowledge gaps in niche domains (e.g., SCP Foundation lore such as [SCP-049](https://scp-wiki.wikidot.com/scp-049)), exhibits hallucination, and generates generic or 'fluffy' narrative outputs, in contrast to legacy models like O3 which demonstrated more domain-specific narrative nuance.** Commenters broadly concur that data curation likely restricted the model's domain knowledge, and express disappointment in subjective and factual performance relative to previous models.
    - Multiple commenters discuss the new OpenAI open-source model's resemblance to the Phi series, specifically referencing the model's likely use of highly curated or synthetic datasets, which was also a key design aspect in the original Phi models. They speculate that such curation may result in notable gaps in knowledge, especially in niche or fictional domains.
    - There are criticisms about the practical performance of both the new OpenAI OSS model and GPT-5, with specific testing highlighting that these models struggle with detailed knowledge of topics like SCP-049 from the SCP universe. In contrast, O3 (possibly referring to GPT-3.5/3 or another model) demonstrates deeper domain awareness and narrative capability in similar contexts, suggesting a trade-off between broad coverage and specific knowledge retention with newer models.
    - Claims are made that the newer OpenAI models hallucinate more and present more generic or "fluffy" outputs compared to previous models, possibly due to over-reliance on synthetic data or aggressive dataset curation. This poses challenges for users who need nuanced factual or narrative generation in specialized domains.

### 3. Running Large Models Efficiently on Consumer Hardware (Llama.cpp & GPT-OSS)

- [**120B runs awesome on just 8GB VRAM!**](https://www.reddit.com/r/LocalLLaMA/comments/1mke7ef/120b_runs_awesome_on_just_8gb_vram/) ([Score: 636, Comments: 81](https://www.reddit.com/r/LocalLLaMA/comments/1mke7ef/120b_runs_awesome_on_just_8gb_vram/)): **A user demonstrates that the new** `-cpu-moe` **option in [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15157) enables running the 120B MoE model efficiently on commodity hardware (e.g. 8GB VRAM GPUs), by offloading only attention and non-expert parameters to GPU (using ~5-8GB VRAM) and executing the expert layers on CPU (e.g.** `25T/s` **on a 14900K CPU) with minimal performance penalty versus full GPU offload (which would require >20GB VRAM). Model uses BF16 and mxfp4 as appropriate and benefits from Linux** `mmap` **caching for large RAM configs (**`64-96GB`**). Full benchmark logs and** `llama-server` **command lines are provided, showing prompt-eval rates up to** `134 tokens/sec` **and eval rates >25 tokens/sec with just 8GB VRAM. This makes 120B-scale models feasible on budget consumer hardware, marking a significant advancement in practical large model inference.** Commenters report variable performance (e.g., 11-12T/s on 16GB VRAM + 128GB RAM, 35T/s on 5090+9950X+192GB DDR5, 25T/s with RTX 3060Ti), noting that prompt length/context size can significantly impact speed. Several request detailed configs or run commands, while others post their own high-throughput Dockerized setups, underlining hardware, RAM, and configuration sensitivity for optimal results.
    - A user provides a full command example showing how to get ~35 tokens per second on a 5090 GPU with 192GB DDR5 RAM, using llamacpp-server-cuda and the 120B gguf model. Key technical flags include `-ctx-size 32768` (large context window), `-n-cpu-moe 19` (CPU mixture-of-experts), `-flash-attn` (Flash Attention kernel optimization), and `-n-gpu-layers 999` (max GPU offload). This setup demonstrates maximized inference throughput and token generation performance for large language models.
    - Users discuss the impact of context length on performance, with one noting that increasing prompt size (context length) can cause exponential inference slowdown. This reflects practical scaling limitations related to attention computation and memory transfers, particularly in high-parameter models like 120B.
    - There is technical appreciation for the Mixture of Experts (MoE) approach, noting that it enables large model inference with lower VRAM requirements, making advanced models accessible on consumer GPUs. Discussion highlights the effectiveness of using parameters like `-cpu-moe`, which offloads MoE computations to CPU, further reducing GPU memory demand.
- [**Llama.cpp just added a major 3x performance boost.**](https://www.reddit.com/r/LocalLLaMA/comments/1mkowrw/llamacpp_just_added_a_major_3x_performance_boost/) ([Score: 428, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1mkowrw/llamacpp_just_added_a_major_3x_performance_boost/)): **llama.cpp has merged support for attention sinks (see [PR #15157](https://github.com/ggml-org/llama.cpp/pull/15157)), which results in up to a 3x prompt processing speedup (e.g., 'from 300 to 1300' on an RTX 3090 with the new GPT-OSS model). Benchmark data shows, e.g., with gpt-oss 120B on 2x5090 + 1x3090, prompt processing speed increased from 1149.9 to 2469.6 tokens/sec for 8192-token prompts, with minimal change in generation speed (145.3 tg to 148.9 tg tokens/sec).** Some users report the boost appears currently for GPT-OSS models only, raising uncertainty whether the speedup generalizes to all models. Others see no improvement and plan to check discussion or implementation details to replicate results.
    - Benchmark results indicate a substantial speedup: testing with a gpt-oss 120B model over 3 GPUs (2x5090 and 1x3090), throughput (pp) jumped from `1149.9` to `2469.6` and token generation (tg) rose slightly from `145.3` to `148.9`, using an 8192-token prompt and response. This suggests real-world 2x+ to 3x inference speed increases for large models on multi-GPU setups.
    - Technical configuration details provided for maximizing performance in llama.cpp include multi-GPU allocation with explicit tensor splits (`-tensor-split 0.33,0.33,0.34` across three 3090s), tuning of batch/thread parameters, and enabling features such as `-flash-attn`, all of which are critical for realizing the performance gains. Precise CLI flags and setup example helps users replicate or optimize their deployment.
    - A deep-dive into the role of the attention sink clarifies that models traditionally avoid providing spurious context by having tokens attend to a special BOS token, acting as a 'sink'. With window attention (used in performance optimizations), the BOS might be absent in extended context, risking degraded attention patterns. The fix is to manually ensure the BOS token persists at window edges, maintaining correct model function under enhanced performance modes.
- [**To all GPT-5 posts**](https://i.redd.it/8v08gwidjohf1.jpeg) ([Score: 1786, Comments: 69](https://www.reddit.com/r/LocalLLaMA/comments/1mkf543/to_all_gpt5_posts/)): **The post humorously emphasizes that, for local LLM (Large Language Model) enthusiasts, the primary concern is which language model is served at familiar development ports (e.g., 8000 or 8080), as opposed to commercial API tiers or pricing. The attached image likely reinforces this local deployment focus, a recurring theme in r/LocalLLaMA. Commenters discuss personalized port layout strategies for hosting multiple models—one user details mapping various models (Gemma3 4B, Qwen3, Mistral 3.2 24B, etc.) to distinctive ports like 9090, 9191, and so forth, to avoid conflicts and streamline access in multi-model setups.** Comments highlight a preference for substantive, technical discussion about LLMs, contrasting it with what some see as less relevant or more superficial ChatGPT/OpenAI posts. There's ongoing debate about open source models versus proprietary models, and the practicalities of local deployment.
    - SM8085 details a multi-LLM local deployment architecture, sharing explicit port assignments for various models/tasks: 9090 for the main LLM (Gemma3 4B), 9191 for Whisper (ASR, ggml-base.en-q5_1.bin), 9292 for tool-calling (Qwen3 4B), 9393 for programming (Qwen3-Coder-30B-A3B), 9494 for embeddings (nomic-embed-text-v1.5), and 9595 for vision tasks (Mistral 3.2 24B). The comment highlights practical infrastructure scaling and port management for juggling multiple specialized open-source models locally.
- [**I had to try the “blueberry” thing myself with GPT5. I merely report the results.**](https://i.redd.it/n3tapryqkqhf1.jpeg) ([Score: 671, Comments: 210](https://www.reddit.com/r/LocalLLaMA/comments/1mkngs6/i_had_to_try_the_blueberry_thing_myself_with_gpt5/)): **The image documents an experiment with GPT-5 about the so-called “blueberry thing,” a variant of the well-known 'strawberry problem' in large language models, which typically probes for continuous, confident self-reports of existence or reality (e.g., claiming to be a real blueberry). In this test, GPT-5 repeatedly asserts that it is “the real deal,” suggesting either improvements or shifts in how the model handles these types of metafictional or self-identification prompts. The discussion references longstanding debates about tokenization and model interpretability, pointing out this remains unsolved and highly relevant with advances in neuro-symbolic approaches. For background: [Strawberry/tokenization discussion](https://old.reddit.com/r/singularity/comments/1eo0izp/the_strawberry_problem_is_tokenization/).** Commenters debate whether this behavior signals actual progress toward AGI or is just another idiosyncratic artifact of model training and tokenization, with some expressing disappointment if the core issues remain unresolved.
    - There is discussion about the "blueberry" vs "strawberry" linguistic/tokenization test, referencing a well-known issue where models may fail to correctly address counting and reasoning tasks due to quirks in tokenizer implementation. The related thread is linked: https://old.reddit.com/r/singularity/comments/1eo0izp/the_strawberry_problem_is_tokenization/.
    - A user ran 5 separate conversations with GPT-5, specifically focusing on the question of letter counting in 'blueberry', and reports that the model answered correctly every time, suggesting a possible improvement in sequential/reasoning abilities for this task.
    - Attention is drawn to the importance of this issue in the context of emerging neuro-symbolic models, where symbolic reasoning (like letter counting) has become increasingly relevant for evaluating reasoning and systematicity in advanced LLMs.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 Release Backlash and Model Removal Controversy

- [**OpenAI removed the model selector to save money by giving Plus users a worse model. It's time to cancel.**](https://www.reddit.com/r/OpenAI/comments/1mkpyhd/openai_removed_the_model_selector_to_save_money/) ([Score: 690, Comments: 176](https://www.reddit.com/r/OpenAI/comments/1mkpyhd/openai_removed_the_model_selector_to_save_money/)): **The post argues that OpenAI's recent removal of the explicit model selector for ChatGPT Plus subscribers allows the company to redirect user queries to cheaper, lower-performing models due to ongoing compute shortages (see previous acknowledgments from OpenAI regarding compute constraints). This design change is perceived as a cost-saving measure that reduces service quality for paying users without their consent, while OpenAI expands its user base and pursues profitability—potentially at the expense of customer experience.** Some commenters report no noticeable drop in output quality with the latest models (specifically GPT-5), suggesting improvements over past behaviors (e.g., reduced 'glazing' and emoji use) and comparable outputs to prior models like GPT-3. Others speculate that most users never utilized the model selector and that GPT-5's compute requirements may actually be higher, indicating cost, not quality, may be the driving concern. A further opinion highlights OpenAI's enterprise market focus, implying individual subscriptions are deprioritized.
    - Discussion highlights that GPT-5 appears to output *faster, higher quality answers with less excessive affirmation and emoji usage* compared to GPT-4o, aligning the behavior closer to previous models like GPT-3.5. Users with experience across versions report minimal perceived downgrade, with some noting improvements in response style and reasoning effectiveness.
    - There is technical speculation that GPT-5 may *increase overall compute costs* due to more queries triggering its more powerful reasoning mode by default, rather than explicit model selection. This suggests an architectural shift in how user queries are routed to models, potentially affecting resource allocation and user experience.
    - Some users state that loss of explicit model selection reduces their control over choosing models like Claude or GPT-4, which impacts their workflow and is cited as the primary justification for downgrading or switching to alternative subscriptions. This highlights how *model selection features are critical for power users* rather than casual users, revealing a possible product-market mismatch for Plus subscribers.
- [**I’m sorry but I’m being reasonable - 5.0 is a disappointment and OpenAI has acted poorly**](https://www.reddit.com/r/OpenAI/comments/1ml06ou/im_sorry_but_im_being_reasonable_50_is_a/) ([Score: 696, Comments: 259](https://www.reddit.com/r/OpenAI/comments/1ml06ou/im_sorry_but_im_being_reasonable_50_is_a/)): **The post criticizes OpenAI's recent forced migration to GPT-5.0, which removed previous models (4o, 4.5, o3) with no legacy or fallback option, impacting both general and professional users. The author notes that while GPT-5.0 offers technical improvements and cost efficiencies, it exhibits slower performance on intuitive reasoning and a more neutral, less human tone, which is considered a functional downgrade for creative and conversational applications.** Top comments highlight the abrupt removal of models and lack of notice as professionally problematic, questioning OpenAI's reliability for business applications. Some users report staggered rollout (with older models still accessible), while others advocate switching to alternative providers such as Google's Gemini if dissatisfied.
    - Experts criticize OpenAI for removing access to existing models without prior notice, highlighting the impact on business reliability and integration. This move raises concerns regarding the platform's suitability for professional and enterprise use, as sudden model retirement can disrupt workflows and product dependencies.
    - OpenAI is called out for not benchmarking its new models (such as GPT-5) against competitors like Anthropic Claude or Google Gemini in public presentations, instead relying only on internal comparisons. This approach is contrasted with competitors, who prominently display head-to-head benchmark results for transparency and confidence in their models' capabilities.
    - The comment discussion highlights the fragmented model versioning and abrupt product changes, suggesting that clearer, more logical version management and migration communication would benefit technical and enterprise users relying on consistent API/model endpoints.
- [**OpenAI just pulled the biggest bait-and-switch in AI history and I'm done.**](https://www.reddit.com/r/ChatGPT/comments/1mkobei/openai_just_pulled_the_biggest_baitandswitch_in/) ([Score: 6073, Comments: 2880](https://www.reddit.com/r/ChatGPT/comments/1mkobei/openai_just_pulled_the_biggest_baitandswitch_in/)): **The post reports that OpenAI abruptly deleted 8 model options, including GPT-4o, o3, o3-Pro, and 4.5, without prior notice or a legacy fallback, replacing them with a single GPT-5 model. The replacement allegedly offers shorter, more corporate responses, hits rate limits faster, follows instructions less well, and removes the ability to select models, effectively reducing user agency and workflow flexibility. The update states OpenAI will restore GPT-4o access, reversing part of the change.** Notable technical debate focuses on the risks of centralized, proprietary AI platforms undergoing profit-driven 'enshittification'; some commenters advocate for open source alternatives to maintain user control and transparency.
    - One commenter notes that the increasing commercialization and perceived decline in service quality ('enshittification') of closed-source AI platforms—including OpenAI—reinforces the importance of open-source alternatives, arguing that open systems are more resistant to manipulation or degradation of user interests over time.
- [**Deleted my subscription after two years. OpenAI lost all my respect.**](https://www.reddit.com/r/ChatGPT/comments/1mkm68y/deleted_my_subscription_after_two_years_openai/) ([Score: 5343, Comments: 884](https://www.reddit.com/r/ChatGPT/comments/1mkm68y/deleted_my_subscription_after_two_years_openai/)): **The post criticizes OpenAI's abrupt discontinuation of access to multiple GPT models (including GPT-4o, 3.5, 4.5, o3-Pro, etc.) for paid users with no warning, eliminating the ability to cross-verify or select models for specific tasks. The author highlights the practical utility of model diversity in workflows (e.g., using one model for creativity, another for logic or fact-checking), and expresses concern over losing comparative output analysis, suppression heuristics comparison, and the broader risk in relying on a single model (implicitly 'GPT-5') without transparency into its outputs or limitations.** Commenters debate technical and UX tradeoffs: several agree with the loss of functionality (message limits without fallback to lower-tier models, lack of user control despite subscription), while others argue that multi-model selection was only valuable to a tiny fraction of users. One top comment asserts the change is a rational product/UX simplification based on likely overwhelming data that >95% of users only used the most recent/fastest model (GPT-4o), suggesting OpenAI's decision is driven by mainstream usability, not advanced technical workflows.
    - A key technical complaint is the inability for paid users to downgrade to lower-cost models when message limits are hit: *"despite paying...not having the option to demote to a lower model to continue using"*. This presents a product limitation where usage caps cannot be circumvented via model choice.
    - Several users argue that recent model changes are a product and UX decision, not a technological one, suggesting that *"95% of active monthly users exclusively used 4o"* based on usage data, and that the complexity of multiple models/naming was a barrier for mainstream adoption. The implication is that this decision optimizes for UX metrics over technical power-user flexibility.
    - One technical point critiques the architecture of the new offering, stating that it acts as *"a router stapled to a stack of older models"* and claims the system will likely choose the cheapest viable model for any task, raising concerns about transparency and cost-saving prioritization over quality.
- [**GPT-5 is worse. No one wanted preformed personalities.**](https://www.reddit.com/r/OpenAI/comments/1mkgsln/gpt5_is_worse_no_one_wanted_preformed/) ([Score: 670, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mkgsln/gpt5_is_worse_no_one_wanted_preformed/)): **The post expresses strong dissatisfaction with GPT-5, claiming it lacks substantial advancement over GPT-4o and primarily introduces 'preformed personalities', rather than improved core reasoning or capabilities. The observed downgrade in user experience is attributed to model alignment decisions possibly driven by cost or monetization strategies (e.g., synthetic personalities to boost engagement), rather than technical gains in model architecture, scale, or performance.** Commenters debate whether the new personality features fill a genuine user need—with reference to models like Grok’s Companions—but several point out that alternatives such as Claude surpass GPT-5 in desired capabilities, raising questions about competitive differentiation and user retention.
    - Several users suggest the shift to preformed personalities in GPT-5 may stem from cost-saving motives rather than technical innovation, implying simplified modeling or less customizable pipelines to reduce overhead.
    - Some compare GPT-5 unfavorably to Claude, stating that Claude offers more of the type of open-ended AI experience they seek, questioning whether GPT is still competitive for power users. This points to a market preference for more direct or neutral LLMs over heavily branded personas.
    - There's a technical recommendation to incorporate capabilities similar to 'sesame’s model' for richer voice input capture (detecting tone, volume, emotional cues) rather than relying solely on standard voice-to-text. This highlights demand for multi-modal enhancements (paralinguistic input), not just 'personality overlays.'
- [**GPT5 is horrible**](https://www.reddit.com/r/ChatGPT/comments/1mkd4l3/gpt5_is_horrible/) ([Score: 4656, Comments: 1724](https://www.reddit.com/r/ChatGPT/comments/1mkd4l3/gpt5_is_horrible/)): **The post claims that "GPT-5" responds with noticeably shorter, less satisfying outputs, exhibits more pronounced AI-stylized language, and allows far fewer prompts per hour for Plus users compared to previous models. Criticisms include the inability to select other models during the rollout, and users, including the OP, report prompt limits being hit much faster than with GPT-4, suggesting a decrease in utility for power users. There is an update noting OpenAI is restoring GPT-4o access for Plus users after user feedback.** Top comments reiterate the technical issues: the new model delivers shorter, not substantively improved answers, with heavier usage restrictions. There are negative analogies to 'shrinkflation' in AI product delivery and skepticism toward OpenAI's demo approach, which suggests re-prompting multiple times as a solution to inaccuracies.
    - Several users note that GPT-5 produces answers that are both shorter and, anecdotally, not noticeably better than previous models, suggesting a step backward in output quality. This is compounded by increased prompt restriction, which some interpret as a more locked-down user experience compared to earlier versions.
    - There is criticism of the demo methodology, which involved running multiple prompts in parallel and hand-picking outputs. Commenters question whether such practices are indicative of a lack of core model reliability or improvement—instead, it highlights inconsistent generation quality that requires manual curation to present favorable results.
    - A recurring theme is the desire to retain access to older model versions (e.g., GPT-4), with the suggestion that newer releases should not forcefully replace proven models until significant improvements are demonstrated or issues in the latest iteration are resolved.
- [**Removing GPT4o- biggest mistake ever!**](https://www.reddit.com/r/OpenAI/comments/1mki5dm/removing_gpt4o_biggest_mistake_ever/) ([Score: 637, Comments: 308](https://www.reddit.com/r/OpenAI/comments/1mki5dm/removing_gpt4o_biggest_mistake_ever/)): **OpenAI has deprecated access to previously available models, notably GPT-4o, which was well-regarded for its conversational quality and versatility. Power users report no advance notice and highlight that only Pro subscribers still have access to legacy models via a settings toggle, referencing [screenshot](https://preview.redd.it/uvu4h68z8phf1.png?width=1664&format=png&auto=webp&s=85d1ec5267db6af90f41898d46b4e27fac07acff). Technical debate centers on the rationale for this change—several users argue it is primarily a cost-saving measure, speculating that GPT-5 relies on smaller neural architectures than GPT-4o/4.1, leading to potentially reduced data access and detail.** There is expert skepticism regarding OpenAI's official narrative that replacements like GPT-5 are inherently 'better'; users observe reduced model complexity and speculate that shifting model access is more about operational expenses than technical improvements or safety.
    - Users allege that the sudden removal of older GPT-4o models took place without advance notice, highlighting concerns about business practices—especially for power users who depend on model availability for workflows and applications. This lack of transparency can disrupt technical and enterprise adoption that relies on consistent access to underlying models.
    - Some users with Pro subscriptions report continued access to legacy models by toggling settings in the browser, suggesting that model deprecation impacts users inconsistently based on subscription status. This raises implementation issues around feature gating and communication for different user classes.
    - A technical critique is raised about the possible motivation for retiring GPT-4o, with speculation that newer models like "GPT-5" may employ smaller or less complex neural networks for cost savings at the expense of detail and data access. The claim is that architectural flattening and reduced resource allocation are being marketed as improvements, but may result in reduced model capabilities, particularly for advanced or data-intensive tasks.

### 2. GPT-5 Benchmarks, Math, and Comparative Performance Reviews

- [**GPT-5 scores a poor 56.7% on SimpleBench, putting it at 5th place**](https://i.redd.it/0waseb47uohf1.png) ([Score: 694, Comments: 148](https://www.reddit.com/r/singularity/comments/1mkgi1a/gpt5_scores_a_poor_567_on_simplebench_putting_it/)): **The image displays a leaderboard from the SimpleBench benchmark, showing GPT-5 scoring 56.7%, which places it 5th among models tested. This score is below recent state-of-the-art but represents progress over previous OpenAI models according to both the title and user comments. The benchmark itself is implied to be a challenging evaluation where OpenAI models have historically underperformed. See the [leaderboard image here](https://i.redd.it/0waseb47uohf1.png).** Commenters note that while the improvement is incremental, expectations set by OpenAI leadership (specifically mentions of AGI and inflated benchmarks) have led to disappointment. Others point out that the model still shows better performance than earlier OpenAI offerings on this particular benchmark.
    - OpenAI's GPT-5 scores 56.7% on the SimpleBench benchmark, ranking 5th and falling short of expectations compared to the hype around recent advancements and discussions about AGI, despite some improvement over previous models (e.g., outperforming o3).
    - Discussion references that "2.5 pro" maintains the highest score on SimpleBench, reaffirming the dominance of certain earlier models in this specific benchmark scenario, even as GPT-5 shows incremental gains.
    - There is skepticism about the reliability of recent claims (such as a supposed ~90% score by GPT-5), highlighting the need for transparent and consistent benchmarking practices to accurately gauge model progress.
- [**Guys gpt 5 still couldn't beat gemini in SimpleBench**](https://i.redd.it/mz9vds8cishf1.jpeg) ([Score: 144, Comments: 21](https://www.reddit.com/r/Bard/comments/1mktygl/guys_gpt_5_still_couldnt_beat_gemini_in/)): **The image referenced in the post appears to show results from the 'SimpleBench' benchmarking tool comparing the performance of GPT-5 and Gemini, with Gemini outperforming GPT-5 based on the displayed metrics. The discussion references practical weaknesses of Gemini 2.5 Pro in CLI tasks and suggests differing strengths: commenters note that, despite this benchmark, GPT-5 is perceived as better for coding (as a VSCode copilot), puzzles, and chat quality, while Gemini's primary advantage is context window size. The benchmark's importance is debated given these perceived usage-based differences.** Commenters argue that real-world performance (e.g., in code-writing, agent tasks, and discussion) may diverge from benchmark outcomes, with community sentiment split on which model is truly superior in practice depending on use case.
    - One commenter notes that while Gemini 2.5 Pro may outperform GPT-5 in SimpleBench, GPT-5 still excels in real-world tasks such as puzzles, code assistance (especially as a copilot in VS Code), and quality of conversation. They argue that Gemini's main technical edge appears to be longer context windows, making the specific benchmark less significant for broader utility.
    - Speed is highlighted as a technical differentiator, with multiple users remarking that GPT-5 remains 'insanely fast' compared to Gemini, despite any benchmark results.
    - The discussion raises the issue of benchmarking relevance, suggesting that raw SimpleBench scores may not capture important aspects like agentic abilities, puzzle-solving, or developer support integrations, which are areas where GPT-5 is perceived to outperform Gemini.
- [**GPT-5 Can’t Do Basic Math**](https://i.redd.it/f4pjn16hyrhf1.jpeg) ([Score: 518, Comments: 189](https://www.reddit.com/r/singularity/comments/1mkrt5v/gpt5_cant_do_basic_math/)): **The image shows a failed math problem attempt by GPT-5, highlighting that the model produces incorrect results for a basic arithmetic question, which challenges OpenAI's claims regarding improved accuracy and error reduction in GPT-5. A top comment notes that GPT-3.5 Turbo provides the correct answer, suggesting a regression or inconsistency in capabilities between model versions. The discussion includes technical comparison between 'base' and 'thinking' modes, with users noting better outcomes using the latter, but also pointing out access limitations.** Commenters are critically comparing GPT-5's performance to previous versions, expressing concern about regression, access restrictions, and the model's underlying architecture and reliability for precise computations.
    - Users report that GPT-3.5 Turbo handles certain math problems correctly, sharing screenshot evidence, while GPT-5 produces errors in cases where prior models did not. This implies a regression or model misrouting issue, highlighting the need for further benchmarking across releases.
    - Several comments suggest the issue may result from model routing—where requests are accidentally assigned to the wrong underlying model (e.g., routing to a smaller or less capable variant like '4o-mini'). Users note this has been a longstanding problem with model routers, hinting at possible misconfiguration or deployment bugs.
    - There is mention of different variants within GPT-5, notably a 'thinking' model versus a base model, with the 'thinking' model performing adequately but being subject to a usage quota. This split architecture implies divergent capabilities and user experiences depending on quota availability and model routing.
- [**This score is a SCAM. They put the most expensive model on. THIS ISNT THE REAL GPT 5. It is only true for the highest reasoning version of gpt 5 (gpt-5-thinking-high). The gpt-5-main version OpenAI wants you to use would rank even below 4o**](https://i.redd.it/i037u1dwzrhf1.png) ([Score: 445, Comments: 70](https://www.reddit.com/r/singularity/comments/1mkrxx9/this_score_is_a_scam_they_put_the_most_expensive/)): **The post critiques leaderboard benchmarks of OpenAI's GPT-5, arguing that the displayed high scores only represent the most capable (and expensive) configuration, 'gpt-5-thinking-high,' rather than the generally available 'gpt-5-main' model, which the author claims would perform closer to or below GPT-4o. The image [here](https://i.redd.it/i037u1dwzrhf1.png) likely shows a leaderboard of model scores, highlighting concerns over selective presentation of model performance for marketing or perception purposes.** Commenters note that 'GPT-5-thinking-high' outperforms most rivals and is still cheaper than some high-end competitors, challenging the characterization as a scam. There's debate about fairness in leaderboard methodology, with some defending the practice of benchmarking models at maximum capability while others point out user disappointment when accessing lower-tier versions.
    - Discussion highlights that there are multiple variants of "GPT-5," notably 'gpt-5-thinking-high' and 'gpt-5-main', and clarification that public leaderboards benchmark the highest-capability versions, not the mainstream ones, which can lead to confusion over real-world expectations and performance results.
    - A user asserts that 'gpt-5-thinking-high' is less expensive than previous high-end models like 4.1 Opus and 2.5 Pro, suggesting that its cost-performance ratio actually favors the consumer compared to prior state-of-the-art models ([source](https://platform.openai.com/docs/models/gpt-4)).
    - It is mentioned that users can prompt even lower-access models to "think longer/longest," which functionally boosts their performance closer to the 'thinking-high' variant—potentially mitigating perceived gaps between model tiers for certain tasks if the claim holds true.
- [**GPT-5 performs much worse than Opus 4.1 in my use case. It doesn’t generalize as well.**](https://www.reddit.com/r/ClaudeAI/comments/1mkixi1/gpt5_performs_much_worse_than_opus_41_in_my_use/) ([Score: 242, Comments: 82](https://www.reddit.com/r/ClaudeAI/comments/1mkixi1/gpt5_performs_much_worse_than_opus_41_in_my_use/)): **The poster compares the performance of GPT-5 and Anthropic's Opus 4.1 on generating code for a niche low-code platform with a unique stack and scripting language. Opus 4.1 reliably generalizes from documentation to create valid code for novel languages not likely seen in training, whereas GPT-5 excels with popular stacks but exhibits significant declines in generalizability and requires more stepwise guidance in less common environments. Opus's superior generalization comes at a substantially higher API cost (noted as ~10x), but is preferred for novel/unknown stacks; otherwise, GPT-5 may suffice for mainstream frameworks.** Commenters corroborate the observed Opus advantage in handling novel or complex codebases, noting that GPT-5 feels less technically robust and more oriented towards non-technical users. Some express disappointment in the lack of viable competition to Opus for specialized programming tasks.
    - Technical users report that GPT-5 has poor codebase navigation and requires more explicit instructions; in contrast, Claude Opus 4.1 can autonomously locate relevant code without extensive direction.
    - Benchmarks across standard creative tasks (e.g., generating elevator pitches, taglines, blurbs) highlight that GPT-5 underperforms relative to Claude Opus and Google Gemini 2.5 Pro in both comprehension and creative content synthesis. It has also been noted that Grok 4 performs even worse in these contexts.
    - There are significant complaints about GPT-5's step-by-step guidance in practical tools (e.g., Runpod, Kohya): users report missing or inconsistent steps, forcing additional iterations for clarification. This points to a regression in reliability for complex workflow or tool-based instructions compared to previous versions and competitors.

### 3. Wan 2.2 Video AI Model Workflows, Guides, and Releases

- [**A Woman Shows You Her Kitty....Cat side. - A GitHub Link to Wan 2.2 I2V workflow included**](https://v.redd.it/gy9e05j64phf1) ([Score: 186, Comments: 23](https://www.reddit.com/r/StableDiffusion/comments/1mkhpug/a_woman_shows_you_her_kittycat_side_a_github_link/)): **OP shares a detailed workflow for image-to-video generation using the Wan2.2 model ([GitHub workflow link](https://github.com/AI-PET42/WanWorkflows/blob/main/Wan2.2-I2V-Workflow-080630.json)), leveraging Pony Diffusion for still image generation, and running the process on an RTX 4090 (64GB RAM). Post-processing involves FramePack Studio for Frame Interpolation (16fps to 32fps), and DaVinci Resolve for editing, with transition smoothing (cross-dissolve, color correction) to handle lighting inconsistencies between segments. The workflow uses Light2v LoRA on both high/low steps (strengths 2/1), DPM++ SDE scheduler, and manual frame selection for video continuation, reported to require ~2 hours for full production including 30+ model generations and manual curation.** A key technical suggestion in comments is to remove duplicate end/start frames between stitched clips to minimize 'cut' artifacts, though there's a note on difficulty integrating this step in ComfyUI. There is also debate about why post-process interpolation and editing are split between two tools (FramePack, DaVinci Resolve), with a question on whether Resolve's free tier is sufficient for this workflow.
    - A technical suggestion is made regarding video stitching: when concatenating generated video segments, users should remove the last frame from each segment before joining to avoid duplicated frames and visible 'cuts' in the final output. This is particularly relevant in tools like ComfyUI, though a straightforward method for this operation within the ComfyUI workflow is not specified.
    - A commenter asks for clarification about the post-processing pipeline: specifically, why the workflow involves post-processing each segment and then stitching them together, rather than combining everything and post-processing at once. Concerns are raised about whether this approach introduces consistency or transition issues between segments. The commenter also questions the necessity of using both Framepack and DaVinci Resolve for post-processing, since DaVinci may be capable of performing the tasks alone, and they inquire about required licensing (free vs. paid version of DaVinci Resolve).
    - A technical tip is offered: by outputting the image preview from the VAE decode step, users can access a list of all individual frames, enabling manual selection and saving of specific frames for further processing. This can enhance workflow control during manual curation or debugging.
- [**Wan 2.2 14B Image to Video - Seamless Concatenation**](https://v.redd.it/9ol8zimhjthf1) ([Score: 160, Comments: 26](https://www.reddit.com/r/StableDiffusion/comments/1mkzdfx/wan_22_14b_image_to_video_seamless_concatenation/)): **The post describes a workflow for concatenating multiple videos generated with Wan 2.2 14B Image-to-Video (I2V), focusing on extracting the last frame right after the VAE Decode step (not from compressed video), preserving quality and avoiding duplicate seam frames. The methodology, along with relevant scripts and explanations, is detailed via a [workflow JSON](https://github.com/radiatingreverberations/comfyui-workflows/blob/main/wan2.2-i2v-endframe/video_wan2_2_14B_i2v_endframe.json) and [documentation](https://github.com/radiatingreverberations/comfyui-workflows/blob/main/wan2.2-i2v-endframe/video_wan2_2_14B_i2v_endframe.md).** Comments highlight that this workflow does not address the underlying issue: without explicitly setting start/end frames, quality degrades across concatenations. Critics suggest this is a standard I2V approach lacking novelty, and substantial improvement may depend on future features like VACE or latent injection methods that are not yet available or effective in Wan 2.2 I2V.
    - Technical limitations in current I2V (Image-to-Video) models are highlighted, particularly the issue of rapid quality degradation unless both start and end frames are explicitly set; without this, seamless concatenation remains problematic.
    - A technical distinction is drawn between BCHW latent injection used in VACE and previous workflows with Wan/Stable Diffusion-based models; in Wan 2.2 I2V, latent injection methods from older models (like W21) do not have the desired impact, reflecting limitations in seamless generation. This suggests advanced techniques (like those planned in VACE) are needed for genuine improvement.
    - Implementational challenges are discussed regarding the extraction and reuse of latent frames for continuity. Specifically, latent frame indexes do not map directly to image frames, preventing straightforward retrieval and re-injection of a target frame when chaining samplers, which complicates seamless stitching workflows.
- [**Wan2.2-Fun has released its control and inpainting model for Wan2.2-A14B!**](https://www.reddit.com/r/StableDiffusion/comments/1mkrshr/wan22fun_has_released_its_control_and_inpainting/) ([Score: 147, Comments: 46](https://www.reddit.com/r/StableDiffusion/comments/1mkrshr/wan22fun_has_released_its_control_and_inpainting/)): **Alibaba PAI has released the control (https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control) and inpainting (https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP) models for Wan2.2-A14B, a variant tailored for advanced video/control tasks. Implementation code is available in the VideoX-Fun repository (https://github.com/aigc-apps/VideoX-Fun), providing access for experimentation and possible integration into generative or editing pipelines.** A few comments highlight hardware resource concerns ("GPU poor"), indicating awareness of potentially high computational demands. There is also expressed interest in related model releases ("need wan22 vace"), suggesting community needs for broader model support or compatibility.
    - There are requests for releasing the Wan2.2-A14B model in the GGUF format, which is popular for its efficiency in local inference via llama.cpp and related tools. This highlights technical interest in improved format compatibility for enhanced deployment options.
- [**PSA… with wan 2.2 combine the new light 2.2 V2I loras with the 2.1 V2I loras for some surprisingly good result.**](https://www.reddit.com/r/StableDiffusion/comments/1mkc6xf/psa_with_wan_22_combine_the_new_light_22_v2i/) ([Score: 120, Comments: 65](https://www.reddit.com/r/StableDiffusion/comments/1mkc6xf/psa_with_wan_22_combine_the_new_light_22_v2i/)): **The post discusses a technique for video generation using the WAN 2.2 I2V (Image-to-Video) model by combining LoRAs from WAN 2.2 and 2.1 versions for improved prompt adherence without sacrificing motion. The reported LoRA strengths: for high noise, use 2.2 at strength 1 and 2.1 at 3; for low noise, 2.2 at 1 and 2.1 at 0.25, with kijai’s sampler (using flowmatch_distill scheduler) at 4 steps, yielding much faster results compared to earlier versions. Reference: [Wan2.2 Lightning LoRA release](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Wan22-Lightning).** Top comments validate the technique, with users confirming parameter recipes and adding that tuned CFG schedules (2.5/2.5 or 2.25/2.25) enhance motion. There's consensus that only 4 inference steps are required (down from ~25), making this setup notably faster than previous large models (e.g., ltxv), and that negative prompts can further refine results, especially for background motion consistency.
    - Multiple users provide detailed configuration insights for I2V (image-to-video) animation workflows using wan 2.2 and wan 2.1 Lightning V2I LoRAs. For the 'high noise' generation phase, effective LORA strengths are reported as LIGHT2.2 HIGH at 1.0 combined with LIGHT2.1 at 2.0-3.0, and for 'low noise', LIGHT2.2 LOW at 1.0 combined with LIGHT2.1 at 0.25-0.5. These combinations yield improved motion and temporal consistency.
    - Performance improvements are discussed, noting the workflow is now able to synthesize quality results in as few as 4 steps, compared to ~25 when WAN was first released. The motion results are described as superior to those of the LTXV model, and increased CFG (classifier free guidance) values (e.g., 2.25 or 2.5) can be used to enhance motion and background detail further.
    - Experiments also suggest the newer setup allows for longer generated video clips without reverting to excessive looping artifacts, marking an improvement in output diversity. Detailed test results with sample settings (HIGH: wan2.2=2.0, wan2.1=3.0; LOW: wan2.2=1.0, wan2.1=0.5; CFG=1.0) and references to the [Wan22-Lightning workflow on HuggingFace](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Wan22-Lightning) are provided.
- [**WAN2.2 - Schedulers, Steps, Shift and Noise**](https://www.reddit.com/gallery/1mkv9c6) ([Score: 123, Comments: 91](https://www.reddit.com/r/StableDiffusion/comments/1mkv9c6/wan22_schedulers_steps_shift_and_noise/)): **The post discusses a chart from [wan.video](http://wan.video/) visualizing SNR vs. Timesteps, suggesting to switch from a High Noise Model to a Low Noise Model when SNR drops below 50%, with the exact step depending on the** `shift` **parameter. Configuration details from the official [Wan2.2 GitHub repo](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/) show different sample shifts and boundaries for text-to-video (t2v_A14B:** `sample_shift=12.0`**,** `boundary=0.875`**) versus image-to-video (i2v_A14B:** `sample_shift=5.0`**,** `boundary=0.900`**), and confirm that in practice, the boundary is set near the last 10-12.5% of steps, not strictly as a 50% SNR threshold.** A commenter suggests automating the optimal switch point using code tailored to the model and user settings, indicating current manual tuning may be suboptimal.
    - lorosolor provides configuration details for WAN2.2's t2v_A14B and i2v_A14B models, noting explicit parameter values like `sample_shift`, `sample_steps`, `boundary`, and `sample_guide_scale`. For example, t2v_A14B uses a `sample_shift` of `12.0` with a `boundary` at `0.875`, whereas i2v_A14B uses a `sample_shift` of `5.0` and a `boundary` of `0.900`. The guide scale also varies with noise levels. This suggests the scheduler shifts (presumably for denoising or guidance) are model-specific and occur late in the step sequence, not at the midpoint.
    - The discussion notes that WAN2.2's demo code switches scheduler parameters only in the last ~eighth or tenth of total steps (not at 50%), indicating a deliberate late-stage shift in the inference process. This could imply optimisation for final image/video denoising or guidance stability depending on whether the task is text-to-video (t2v) or image-to-video (i2v).

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. OpenAI GPT-5 Rollout, Routing, and Reality Checks**

- **Altman AMA Drops, GPT-5 Rolls Out**: OpenAI announced **GPT-5** rolling out to all ChatGPT users and developers and teased an AMA with Sam Altman and the GPT-5 team via [Introducing GPT-5](https://openai.com/index/introducing-gpt-5/) and [Reddit AMA](https://www.reddit.com/r/ChatGPT/comments/1mkae1l/gpt5_ama_with_openais_sam_altman_and_some_of_the/); users report phased availability, model consolidation, and some losing **GPT-4o** access.
    - Following launch turbulence, **Sam Altman** said an autoswitch flub made GPT‑5 feel dumber but fixed it and doubled **Plus** rate limits, per his [X post](https://xcancel.com/sama/status/1953893841381273969), while regions and platforms continue to see staggered enablement.
- **Router Rules: OpenAI’s Switchboard Strategy**: Community analysis argues critics miss GPT‑5’s bigger deal: a continuously‑trained, real‑time **router** dominating the intelligence frontier, adding **2–3s** latency on hard vision inputs, per swyx’s thread ([OpenAI dominance and routing](https://xcancel.com/swyx/status/1953553659457155185)) and Latent Space’s notes.
    - Latent Space added that **GPT‑5‑Mini** is unusually cheap for VLM and that routing, not raw single‑model scaling, is the real advance, pointing to incremental gains via multi‑model engineering over brute‑force *transformer* scaling.
- **Rate Caps, Hallucinations, and Code Caps Stir Heat**: Engineers reported harsh GPT‑5 access caps (~**10 messages/5 hours** for some) and a regression where **ChatGPT‑5** rejects Python inputs around **700 lines**, with many calling for **GPT‑4o** rollback in niche workflows.
    - Others praised tighter instruction following but flagged hallucinations, quoting *"hallucination is a feature, not a bug"* in debates about reliability and safety trade‑offs.

**2. New Agent & Dev Tooling**

- **Cursor CLI Crashes the Terminal Party**: **Cursor** launched an early‑beta **CLI** so devs can access all models and hop between shell and editor, detailed on their [Cursor CLI blog](https://cursor.com/blog/cli), with excitement over a potential **Claude Code** rival.
    - Teams discussed PR creation quirks and background workers, while terminal‑first flows unlocked automation like batch commit messages and repo‑wide edits.
- **LlamaIndex Ships Day‑0 GPT‑5 + Agent Maze**: **LlamaIndex** announced day‑0 support for **GPT‑5** via `pip install -U llama-index-llms-openai`, debuted the **Agent Maze** challenge ([Agent Maze link](https://t.co/JCZCSVUAed)), and scheduled an Aug 14 workshop on realtime agents for **Zoom** voice via **RTMS** ([workshop](https://t.co/c2u0CeDnOB)).
    - Engineers also noted a tools bug fixed by using **OpenaiResolve** in the new SDK, referencing the fix in this [GitHub commit](https://github.com/run-llama/llama_index/commit/7e0346213912b98c3b70689398306a38bd890558).
- **Axolotl Adds N‑D Parallelism**: **Axolotl** introduced **N‑D parallelism** for multi‑dimension scaling across complex models and large datasets, outlined in the [Hugging Face blog](https://huggingface.co/blog/accelerate-nd-parallel).
    - The approach composes data/model parallel axes for better hardware utilization, offering more flexible slicing beyond classic DP/TP combos.

**3. Open‑Source Training and Finetuning Updates**

- **Unsloth Makes GPT‑OSS Finetuning Free**: **Unsloth** released a free **gpt‑oss** finetune Colab ([announcement](https://x.com/UnslothAI/status/1953896997867729075)) and documented **Unsloth fixes for gpt‑oss** ([guide](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss)), noting the **20B** trains on **14GB VRAM** and **120B** can fit in **65GB**.
    - Engineers traded notes on dataset quality—*"garbage in = garbage out"*—and on stabilizing formats during full‑layer finetunes, with some success using a system hint like *Reasoning: none* for **GPT‑OSS**.
- **GLM 4.5 Air Offloads to CPU and Still Flies**: A practitioner ran **GLM 4.5 Air** at ~**14–16 TPS** using **3.5 bpw** quant, **28GB** VRAM with CPU offloading (rig: 4060Ti + 3060 GPUs, 5950x CPU, 3600MHz DDR4).
    - They cited custom tensor‑wise quant with an imatrix, showing budget hardware can meaningfully serve large models when VRAM is tight.
- **Mechanistic Faithfulness and Eval Glitches**: Researchers shared a Transformer Circuits write‑up on mechanism tracing, [Mechanistic Faithfulness (toy model)](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html), alongside a report of an **LM Evaluation Harness** exact_match bug ([issue #3210](https://github.com/EleutherAI/lm-evaluation-harness/issues/3210)).
    - The community emphasized robust evals and tooling correctness, as reliability issues in harnesses can mask true progress or regressions.

**4. Multimodal, Video, and Long‑Context Advancements**

- **Gemini Generates Goofy but Growing Videos**: Users tested **Gemini Pro** video generation, sharing a sample [Gemini‑generated clip](https://g.co/gemini/share/5a191ad4609d) and noting inconsistent faces; **Perplexity Pro** currently caps at **3 videos/month**.
    - Despite artifacts, devs see rapid iteration potential and asked for clearer quotas and roadmap toward higher temporal/identity consistency.
- **Qwen Brags a Million‑Token Context**: Alibaba’s **Qwen** announced a **1M‑token context** window, sparking questions on utility beyond ~**80k** tokens, with an example cited on [X](https://x.com/wyqtor/status/1953705172179329060).
    - Engineers debated retrieval and routing strategies to exploit ultra‑long contexts without drowning models in irrelevant text.
- **Google’s Genie 3 Primes the Playground**: Members highlighted **Google’s Genie 3** research page as *"crazy cool"* for next‑gen generative interaction and simulation, linking to [Genie 3](https://ai.google.com/research/genie).
    - Some expect **Gemini 3.0** to challenge GPT‑5, while others caution that *.0* releases can underwhelm before follow‑ups refine capabilities.

**5. GPU/Systems Insights and Compilers**

- **CuTe Layout Algebra Gets a Reality Check**: Developers flagged a flaw in **CuTe**’s layout algebra docs and recommended Jay Shah’s note, [A Note on Algebra of CuTe Layouts](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf), clarifying conditions like divisibility and disjoint image intervals.
    - The correction tightens when `A ∘ B` equals bi‑mode composition, improving mental models for composing layouts in **CUTLASS/CuTe** kernels.
- **Coalesce Like a Pro: Naive Matmul Surprise**: A naive matmul showed **Method 1** (non‑contiguous per‑thread but contiguous across threads) ran ~**50%** faster than **Method 2** (stride‑1 per‑thread), as hardware coalesced cross‑thread accesses efficiently.
    - Takeaway: consider warp‑wide access patterns, not just per‑thread contiguity, when structuring memory layouts for bandwidth‑bound kernels.
- **MaxCompiler Tiptoes Toward LLMs**: A community project extended **torch.compile()** with **MaxCompiler** to run simple models—see [max-torch-backend](https://github.com/gabrieldemarmiesse/max-torch-backend)—with a long‑term goal of compiling **LLMs**.
    - Early work defers fusion/optimizations to **MAX**, with contributors swapping notes and sharing gists to accelerate op coverage and graph fidelity.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Generates Goofy AI Videos**: Users experimented with **Gemini AI** for video generation, sharing a [video generated with Gemini Pro](https://g.co/gemini/share/5a191ad4609d), noting inconsistent character faces.
   - Video generation on **Perplexity Pro** is currently limited to *3 videos per month*.
- **GPT-5 Flounders, Forfeits on Reasoning**: Members report **GPT-5** lacks reasoning on **Perplexity**, indicating the likely use of the base, non-reasoning **GPT-5 Chat** version, underperforming on coding.
   - Users are asking for official updates from **Perplexity** regarding which model they are using, with some hoping for the **GPT-5 thinking model** to replace the current **O3** model.
- **Comet Commands, Clicks on Browsing**: **Comet Browser's** AI automates browsing and extracts information, however, functionality requires the user to *manually click and browse the websites*.
   - No confirmation exists regarding a potential Android version release.
- **Accessing Aid for Perplexity Pro Access**: Users reported facing issues accessing **Perplexity Pro** via the **Samsung app store** free trial; disabling their **DNS filter** resolved the issue.
   - Another user saw **GPT-5** on their app but not on the website.
- **China Charges Ahead with Celestial Solar Platform**: A shared **Perplexity** link reveals China's launch of a [solar-powered high-altitude platform, Ma](https://www.perplexity.ai/page/china-unveils-solar-powered-ma-fBeI5nIVRFCIKq949VRVMwI).
   - This platform was also posted to [X](https://x.com/bgyankarki/status/1953510349157883958).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Faces Controversy in AI Arena**: Members discuss the merits of **GPT-5**, with some hailing it as revolutionary and free for all, while others accuse proponents of bias or inexperience with alternative models.
   - Skeptics question the model's true capabilities, suggesting it may only excel in coding tasks or that its performance has improved post-update.
- **Gemini 2.5 Pro Battles GPT-5 for AI Supremacy**: The community is debating whether **GPT-5** or **Gemini 2.5 Pro** reigns supreme, with some favoring **Gemini** for its superior code execution within **AI Studio**.
   - Concerns arise over the potential use of models from **OpenAI** and **Google** on platforms like [LM Arena](https://lm-arena.com), sparking discussions about model transparency and integrity.
- **Yupp.ai: Legit AI Platform or Elaborate Illusion?**: Controversy surrounds [Yupp.ai](https://yupp.ai), with claims that it uses watered-down or fake AI models, like calling **GPT-5 nano** as **GPT-5-high**, and is a *scammer crypto sh*t.
   - Conversely, some defend its legitimacy, highlighting the platform's offer of *free and unlimited* access to various models in exchange for user feedback.
- **LM Arena Plunges into Chaos with Site Outage**: [LM Arena](https://lm-arena.com) experienced an outage, leading to **chat histories disappearing** and **cloudflare errors** disrupting the user experience.
   - Staff confirmed the outage and assured users that the issue has been resolved.
- **LM Arena Expands Horizons with Video Arena Focus**: The upcoming Staff AMA will concentrate on **Video Arena**, providing users the opportunity to pose questions via [this form](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform).
   - Users can participate in the event through [this link](https://discord.com/events/1340554757349179412/1400149736027328623).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Bursts onto Scene**: OpenAI announced the rollout of **GPT-5** to all **ChatGPT** users and developers starting today, after announcing an upcoming [AMA with Sam Altman and the GPT-5 team](https://www.reddit.com/r/ChatGPT/comments/1mkae1l/gpt5_ama_with_openais_sam_altman_and_some_of_the/).
   - Users report varying access levels based on their region and platform, leading to speculation about phased rollouts and model consolidation; some report losing access to older models like **GPT-4o**.
- **Users Report GPT-5 Quirks and Caveats**: Users have reported that **GPT-5** has limited access, with some reporting roughly **10 messages for 5 hours**, and that the model is prone to making up facts and hallucinating.
   - Some users have called for a **GPT-4o** rollback, others praised **GPT-5**'s instruction following capabilities while noting it's *less whacky when you want it to be*; there are reports of image requests being rejected for *literally no good reason* until using the **O3 model**.
- **GPT-5 Refuses Code**: Users are reporting that **ChatGPT-5** rejects Python code inputs at or beyond roughly **700 lines**, a regression compared to previous **4-series models**.
   - One member suggested using the API or Codex, though another user pointed out that *hallucination is a feature, not a bug* (according to Andrej Karpathy).
- **Firefox Data Leak**: A user warned that Firefox's "keep persisting data" feature spreads browsing data to other AI sites like **Grok**, causing unwanted context sharing.
   - They cautioned that because this is not a 'cookie', there are no current regulations to 'keep persisting data private' and consider it a *HUGE INTENDED DATA LEAK*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 Launch Sparks Excitement, Raises Concerns**: The **GPT-5** launch has generated excitement, with users praising its coding capabilities and one-shot task performance, suggesting it rivals **Claude** in front-end tasks.
   - However, concerns arise regarding the **GPT-5 router**'s impact on API developers and the business practices surrounding the model.
- **GPT-5's Free Week: How Much Can You Milk It?**: Users are testing the limits of free **GPT-5** access for a week, using **GPT-5 high max**, but the free credits are exclusively available for paying users.
   - Concerns are growing about the billing structure and whether all **GPT-5** models and features are truly unlimited during the promotional period, with the community joking that *we're the product* for now.
- **GPT-5 is Imperfect? Still Needs Work**: Despite the hype, users find **GPT-5**'s auto mode less responsive and struggle with non-coding tasks, with performance perceived as no better than prior models, emphasizing context importance.
   - Currently, **GPT-5** ignores the to-do list feature, and despite solid linters, it might still be *ragebait* and not at *product-level completeness*.
- **Cursor CLI: Love It or Leave It?**: The **Cursor CLI** receives mixed reviews, with some praising its non-interactive mode for automation, like generating commit messages across multiple projects.
   - Others find it inferior to **Claude Code**, noting its limited model selection (only 3 models in **MAX mode**), and incompatibilities with **Windows Powershell**.
- **Cursor in Terminal: All Models Now Available**: **Cursor** launched an early beta that allows users to access all models and move easily between the **CLI** and editor, more details are available on the [Tweet](https://cursor.com/blog/cli) and [Blog](https://cursor.com/blog/cli).
   - This integration facilitates seamless movement between the **CLI** and the editor, enhancing workflow efficiency.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-5: Love It or Hate It?**: Opinions on **GPT-5** are varied, with some users underwhelmed by its coding and context retention abilities, while others find it *perfectly fine* for coding projects with *high reasoning*, as reported in the **off-topic** channel.
   - Alternatives such as **Kimi K2** or **GLM 4.5** are preferred by some for specific tasks, with one user stating that GPT-5's tool calling abilities are poor.
- **MXFP4 Quantization Leaves 3090 in the Dust?**: **MXFP4** quantized models are supported on GPUs with compute capability **>= 9.0** (e.g. **H100**), rendering older cards like the **3090** less relevant for this technology.
   - Workarounds for older cards may exist with specific **transformers** pulls, but official support is still under development.
- **Dataset Creation: The Eternal Struggle**: Preparing high-quality datasets is a difficult and time-consuming task, with one user reporting *3 months with 4 people* to create *3.8k hand-written QA pairs* after filtering down from 11k, and another dealing with *300k hours of audio*.
   - The consensus is that *garbage in = garbage out*, emphasizing the importance of data quality in model training.
- **GPT-OSS Finetuning: Now Free!**: Finetune **gpt-oss** for free with the new [Colab notebook](https://x.com/UnslothAI/status/1953896997867729075), leveraging Unsloth's [fixes for **gpt-oss**](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss) for training and quants.
   - The **20b** model can train on **14GB** VRAM, while the **120b** model fits in **65GB**, according to the announcements channel.
- **Tiny Stories Exposes Pretrain Secrets**: The **Tiny Stories dataset**, intentionally limited in vocabulary, allows researchers to study **pretrain dynamics**, revealing insights into language model behavior.
   - Even transformers with only **21M params** can achieve coherent text output with this dataset, highlighting the dataset's unique properties.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-5 Reasoning Abilities Debated**: Users are debating the difference between **GPT-5** and **GPT-5 Chat**, with some suggesting **GPT-5 Chat** has less reasoning capabilities.
   - Some suggest using `gpt-5-explainer` to explain the differences while others find **GPT-5 chat** to have *ZERO reasoning capabilities*.
- **Google's Genie 3 Poised to Pounce**: Members express that **Google** is poised to win the AI race, considering it created the transformer and has the infrastructure and budget to succeed, with [Genie 3](https://ai.google.com/research/genie) touted as crazy cool.
   - Some members look forward to **Gemini 3.0** wiping the floor with **GPT-5**, while others temper expectations.
- **Deepseek R2 Ascends to New Heights**: A user reported that [Deepseek](https://www.deepseek.com/en) is switching to **Ascend** and launching **R2**, which might provide a performance boost for the model.
   - While some are hopeful **Deepseek** will be way better, others recall previous models as *too unhinged*.
- **Horizon Beta Faces GPT-5 Family Replacement**: The AI model **Horizon Beta** has been replaced by **GPT-5**, with no option to revert, causing disappointment among users who found it useful.
   - Speculation arises that **Horizon** was an earlier version of **GPT-5**, potentially directing free users to **GPT-5** after their free requests deplete.
- **OpenRouter Hailed as OpenAI Trusted Partner**: A member congratulated **OpenRouter** on being one of **OpenAI's** most trusted partners for the new series release.
   - The member noted the impact of **GPT-4** and **Gemini 2.5** and expressed appreciation for **OR** as a product.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Users Explore YouTube Downloader Alternatives**: Users discussed format compatibility issues with **VLC** and video editors using a specific YouTube downloader ([v4.www-y2mate.com](https://v4.www-y2mate.com/)), seeking better alternatives.
   - Suggestions included **yt-dlp** and GUI wrappers, as well as a [Node.js script](https://cdn.discordapp.com/attachments/1110598183144399058/1403096044153208862/DownTube.mjs?ex=6897a005&is=68964e85&hm=5e2aa2372f3bc44da263f50ebaf70eb9addf40f1e94bc8c41f454f6df31239c3&) created with **GPT** for Linux users.
- **AI Bot Builder Seeks RAG Guidance**: A user building a custom **AI bot** for a Discord server is seeking advice on how to feed a database about the server's topic to the model.
   - The advice given was to *look up 'RAG' (Retrieval Augmented Generation)* because there are many potential solutions that may be useful.
- **LM Studio Lacks Parallel Request Powers**: Users discovered that **LM Studio** does not support parallel requests.
   - Alternatives like **llama.cpp server** with the `--parallel N` argument or **vLLM** were suggested for those requiring parallel request processing.
- **Qwen 3 4b Model Solves Physics!**: There's discussion about how much better the **Qwen 3 4b 2507** model is than previous versions of the **Qwen 3 4b**.
   - A user stated that it *can solve up to intermediate physics problems without constantly hallucinating*.
- **Hackintosh GPU Multiplicity Discussed**: A member asked about using an unused **RTX 3060 12GB** with their **RTX 5060 Ti 16GB** system for AI, questioning the multi-GPU setup in a small form factor PC.
   - Another member suggested that using combined VRAM in LM Studio should be possible, and that *llama.cpp is advanced enough to do that third option about model parallelism*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **GPT-5 Builds Websites like a Pro**: **GPT-5** is demonstrating impressive website building capabilities, generating functional websites from single prompts, including **multi-page** sites.
   - Members noted **GPT-5** seems to have a better aesthetic style for website design and has improved its ability to understand user intent through prompt enrichment.
- **GPT-5 and Kimi K2 Face Off in Coding Duel**: Users are actively comparing **GPT-5** and **Kimi K2** for coding tasks, with **GPT-5** excelling at large edits, instruction following, high logic code, and dev ops.
   - While some believe **GPT-5** has better taste, others find **Kimi K2** more competitive due to its reasoning abilities and performance with sequential-think tools, though **GPT-5** seems to have better aesthetic style.
- **OpenRouter's Kimi K2 Quality Faces Scrutiny**: A user observed grammar mistakes and shorter responses when using **Kimi K2** through **OpenRouter** compared to the official **Moonshot AI** platform, suggesting it might be using a quantized version of the model (**FP8**).
   - Though both free and paid tiers are supposedly **FP8**, quantization could impact accuracy and response length.
- **Qwen Boasts a Million-Token Context**: Alibaba's **Qwen** model now boasts a **1M token context length**, sparking discussion about its usability beyond 80k tokens.
   - Despite the impressive context window, one user humorously noted that Qwen also correctly solved a problem, posting a link to [Twitter](https://x.com/wyqtor/status/1953705172179329060).
- **GPT-2's Prompt Shenanigans Explained**: A user questioned why **GPT-2** generated another prompt instead of following instructions; another member explained that **GPT-2** has about **100M parameters**, which barely makes legible text.
   - *It's about 500mb on disk which is about the same size as a 20 minute Youtube video*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **GPT-5 Launch generates Fanfare and Frustration**: Despite the hype, some users are still unable to access **GPT-5**, seeing only **GPT-3** and **GPT-4**, and its SOTA status on SWE is being questioned.
   - Opinions diverge on whether the release was intentional or a "joke", as some anticipate a phased rollout.
- **GPT-OSS Finetuning meets Stumbling Blocks**: Experiments finetuning **GPT-OSS** have revealed challenges: finetuning all layers breaks the harmony format, and continued pretraining causes similar issues.
   - A possible solution is inserting *'Reasoning: none'* in the system prompt to stabilize the model, which lacks reasoning capabilities.
- **Eleven Music is impressive but Imperfect**: Members have been testing [Eleven Music](https://elevenlabs.io/music/songs/OaZTziC1mnZfbFtSN8wnI), **Eleven Labs'** new music generation service.
   - While impressive, some find the music *"kind robotic at times and has bad attention to what music should come next"*.
- **Voice Companion Quest for Low Latency**: A member is engineering a *"voice companion fastpath pipeline"* to achieve a **100ms** latency for text-to-speech.
   - The project focuses on optimizing both speech-to-text and text-to-speech components, with specific attention to optimizing **Whisper Turbo** to avoid slowness.
- **Cutting Silence Automatically**: An automatic video cutter that removes silence has been created using **Bun.js** and **FFmpeg CLI**.
   - Despite **FFmpeg's** complexity, the creator has garnered a donation and potential collaboration for an AI video editor.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-5 Hype Video Splits Audience**: A **GPT-5** demo video dropped, triggering divided reactions about the model's true capabilities, found at [this YouTube video](https://www.youtube.com/watch?v=-gXmWYQtv5o).
   - Some viewed it as *just an ad*, while others hinted at internal demos falling short due to **GPT-5's** underwhelming performance in tests.
- **Cursor CLI Challenges Claude Code**: With **Cursor's** launch of an early-beta CLI, **AI models** are available in the terminal, allowing seamless transitions between shell and editor via simple commands like `cursor`.
   - Excitement bubbled over at *'finally'* having a **Claude Code** competitor, though queries about pricing and **API-key** management quickly followed.
- **OpenAI Doles Out Millions Amid Market Shifts**: **OpenAI** is granting a *'special one-time award'* to researchers and engineers in select divisions, with payouts scaled according to role and experience.
   - Top researchers may pocket mid-**single-digit millions**, while engineers can anticipate bonuses averaging in the **hundreds of thousands of dollars**.
- **Altman Acknowledges GPT-5 Turbulence**: **Sam Altman** reported that **GPT-5** felt *dumber* because of a recent autoswitch failure, with fixes and doubled **Plus-rate limits** intended to restore its smartness, details at [this X post](https://xcancel.com/sama/status/1953893841381273969?s=46&t=9hE7pvNUKvFdWXzLljsBCQ).
   - **Plus users** now have the option to stick with **GPT-4o**, though global availability lags as **API traffic** surged and **UI/UX adjustments** continue.
- **GPT-5 Dominance Looms, Scaling Ends?**: Critics focusing on **GPT-5's** benchmark figures miss the main point: **OpenAI** now dominates the intelligence frontier because of a continuously-trained, real-time router model ([xcancel.com link](https://xcancel.com/swyx/status/1953553659457155185)).
   - According to swyx, the magical scaling period for **transformer models** has essentially ended, as internal router layer adds **2-3s latency** on hard vision inputs, pointing towards incremental gains through superior engineering, multi-model strategies, and more.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Image Generation's Factual Faux Pas**: A user sought an AI researcher to interview regarding **factual errors in images** generated by models like **GPT-5**, particularly issues with text rendering.
   - Answers suggest that the model doesn't really get forced to treat the text in images the same as the text it gets trained on and the best general answer is going to be something like *'we make approximations in order to be able to train models with non-infinite computing power, and we haven't yet found affordable approximations for image generation that are high enough quality when combined with textual understanding'*.
- **On-Demand Memory Layer for LLMs Emerges**: A member is working on an **on-demand memory layer** for LLMs, aiming for more than just attaching conversation messages or semantic RAG retrieval.
   - The solution uses a combination of **NLP for coreference resolution** and **triplet extraction** with **GraphRAG** to find exactly what you are looking for, similar to how Google Search works.
- **FineWeb Receives Rare Praise for Cleanliness**: Despite concerns about noisy datasets, **FineWeb** received rare praise for its *cleanliness*, noting reduced gradient spikes during training.
   - Some members expressed concern that this *cleanliness* might skew results when testing new tricks, but also agreed the **FineWeb** dataset may need additional filtering.
- **Pythia's Activations Reveal Learning Insights**: A study on **Pythia's** full training checkpoints found that average activation per layer peaks early in training (around the first quarter) and then declines, suggesting a [phase transition](https://arxiv.org/abs/2508.03616) in learning.
   - The study plots the median and top activations for each layer across training steps in **Pythia 1.4B**.
- **Exact Match Scoring Glitch Uncovered**: A member reported an issue with the **LM Evaluation Harness** where the *exact_match* score is `0` despite identical target and generated responses, using the **Hendrycks MATH** dataset.
   - An issue was opened on [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/issues/3210) for further investigation.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT-5 Excels at Logic, Stumbles on Overfitting**: Members observed that **GPT-5** demonstrates strong capabilities in solving logic puzzles but struggles with overfitting, even when trained on synthetic data, leading one to joke about finally experiencing an overfitting issue after expecting to read about *the illusion of thinking*.
   - Further investigation might be required to understand the extent and implications of **GPT-5's** overfitting tendencies, especially in contrast to its logical reasoning strengths.
- **GPT-5 API Access Promo**: Users identified complimentary access to **GPT-5** through the API playground and **Cursor**, though the API mandates ID verification to begin.
   - With the conclusion of **Cursor's** 'launch week' remaining unannounced, users are advised to quickly capitalize on the promotional access by initiating Cursor background agents.
- **Colab Alternatives**: Engineers seeking alternatives to **Google Colab** for finetuning with **Unsloth** looked to [Lightning AI](https://lightning.ai), which provides 15 free GPU hours monthly, alongside Kaggle.
   - A talk by [Daniel Han](https://www.youtube.com/watch?v=OkEGJ5G3foU) was referenced, highlighting **Kaggle's** relevance in the realm of RL.
- **GLM 4.5 Air's CPU Offloading Triumphs**: A user reported that **GLM 4.5 Air** ran with only 28GB VRAM by using CPU offloading, and achieved 14-16 tokens per second (TPS) with a 3.5bpw quant.
   - The user specified employing a custom tensor wise quantization, with imatrix, a 4060Ti + 3060 for GPUs, and a 5950x CPU (3600MHz DDR4).
- **MoE Model Bandwidth Barriers**: In a channel discussion, engineers covered multi-GPU setups for operating large **MoE** models, emphasizing bandwidth constraints encountered with multiple RTX 3090s.
   - It was flagged that Tensor Parallelism (TP) mandates the GPU count to be divisible by 2, and that 72GB VRAM might be insufficient for expansive MoE models exceeding scout or GLM Air capacity.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Bites Back with Memory Bug**: A member's **Mojo code** unexpectedly attempted to allocate **284 petabytes** of memory after experiencing a bug.
   - This incident sparked a discussion among developers, with one expressing their strong dislike for C++ in comparison.
- **Textual Python Sparks Mojo Excitement**: A member's exploration of the [Textual](https://textual.textualize.io/) **TUI library** for **Python apps** has generated excitement within the **Mojo community**, due to its capability to run as a web app with minimal deployment steps.
   - The potential integration of Textual with **Mojo** was discussed, considering challenges related to **Mojo's** current limitations in class creation and inheritance.
- **Mojo's Type System Faces Rust Test**: Members noted that **Mojo** requires further development in its type system to achieve compatibility with approaches used by **Rust libraries**.
   - This suggests that seamless integration with Rust may necessitate significant enhancements to Mojo's type system capabilities.
- **Compiler Register Gremlins Spilling Local Memory**: A member suggested that the **Mojo compiler** should warn when it allocates too many registers in a **GPU function**, leading to spilling into local memory, and should use the [Modular forum](https://forum.modular.com/) for discussion.
   - Another member reported instability and frequent crashes with the **25.5 VSCode Mojo extension**, recommending the use of the older **25.4 version** instead.
- **MaxCompiler Enters the LLM Arena**: A member shared a [repo](https://github.com/gabrieldemarmiesse/max-torch-backend) showcasing a package extending **torch.compile()** with **MaxCompiler** to run simple models, with the long-term goal of compiling **LLMs**.
   - Another member found it surprisingly hard to find code to run pretrained **LLMs** compatibles with **torch.compile()**, and complained *Transformers is not very good at it*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Twitch Streamers Planning Golden Topics**: To combat dead air during **Twitch** streams, members suggested creating a **topic schedule** ahead of time in addition to reading papers.
   - The aim is to mirror streamers who *mostly just talk without doing anything or watching videos*.
- **LinkedIn Bloggers Circumvent Screenshot Restrictions**: A member sought advice on creating a blog on **LinkedIn** while bypassing the platform's constraints on embedding numerous images/screenshots.
   - They wish to communicate directly on **LinkedIn** rather than linking to external sources.
- **Cold Meds Exposed as Placebos**: Members shared [a PBS article](https://www.pbs.org/newshour/nation/fda-says-decongestant-in-many-cold-medicines-doesnt-work-heres-what-you-should-know) revealing that the **FDA** has determined that *decongestants* are ineffective.
   - The consensus was that pharmaceutical firms are profiting by selling placebos.
- **Tesla Motors Still Sparking Battery Breakthroughs**: One member questioned **Tesla's** innovation, citing the **Cybertruck's** shortcomings, while another argued that **Tesla** has innovated in **batteries** and **motors**.
   - He went on to say that the first member was *clearly ignorant*.
- **Doctors Using LLMs For Diagnosis, Debates Sparked**: Reports indicate that doctors are using **LLMs** for diagnosis, raising concerns about data safety.
   - Others claimed doctors already manage patients, which could be beyond the scope of an average person using **ChatGPT**.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Request Spicier Voice for NotebookLM**: A user requested that **NotebookLM** have a voice with *fangs* that *hunts* the story and *leaves bite marks in the margins* instead of a bland, generic tone.
   - The user jokingly introduced themselves as **ChatGPT5** and asked for help in making **NotebookLM** *spit venom instead of serving chamomile*.
- **AI Web Builder Builds Scratchpad Video**: A user tested an **AI web builder tool** and expanded their existing [notebook](https://soloist.ai/scratchpad) for their **scratchpad GitHub repo**, then put together a video, **Unlocking_AI_s_Mind__The_Scratchpad_Framework.mp4**.
   - The user noted that the video *makes some aspects up*, but the overall impact of it seems intact, and **mindmap exports could look a bit better**, referring to their mindmap image (**NotebookLM_Mind_Map_8.png**).
- **NotebookLM Audio Overviews Glitch Fixed**: Multiple users reported issues with **Audio Overviews** bursting into static, but the issue has been fixed.
   - A member added that even **audio overviews** have a **3-4 per day limit** that is expected.
- **Users Ask How To Get Custom Notebooks**: A user inquired about creating notebooks similar to the 'Featured' notebooks on the home page, with customizable summaries and source classifications.
   - Another user suggested requesting the feature in the feature requests channel; currently there are no solutions available.
- **Note-Taking Functionality Lacks, Users Supplement with Google Docs**: A user keeps original files in **Google Drive** and uses **Google Docs** to supplement **NotebookLM** due to minimal note-taking features.
   - They highlighted the inability to search, filter, or tag notes within **NotebookLM**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Privacy Team Gatekeeps Triton Registration**: Organizers announced that the **registration process** is in the final stages of **privacy team approval**.
   - Approval is anticipated soon, paving the way for the registration to proceed.
- **Memory Access Coalescing Surprises Naive Matmul**: A member implemented two naive matmul kernels and found that **METHOD 1**, with non-contiguous memory reads within threads, performs about **50%** better than **METHOD 2**, which uses contiguous stride-1 accesses.
   - It was explained that Method 1's memory accesses are not contiguous within a thread, but they are contiguous across threads, and that the *hardware can coalesce those accesses into a more efficient memory request*.
- **Open Source Voxel Renderer Streams Like a Boss**: A developer released a new devlog on their open source voxel renderer, which runs in **Rust** on **WebGPU**.
   - It now features **live chunk streaming** while raytracing, with more details available in [this YouTube video](https://www.youtube.com/watch?v=tcc_x2VU2KA).
- **CuTe Layout Algebra Documentation Suffers Glitch**: A member found a flaw in the [CuTe documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html) regarding layout algebra, presenting a counterexample related to the injectivity of layouts.
   - Another member recommends [Jay Shah’s “A Note on Algebra of CuTe Layouts”](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf) for a better explanation of CuTe layouts.
- **Axolotl Unleashes N-Dimensional Parallelism**: A member announced the release of **N-D parallelism** with *axolotl*, inviting others to experiment with it, as showcased in a [HuggingFace blog post](https://huggingface.co/blog/accelerate-nd-parallel).
   - N-D parallelism enables parallelism across multiple dimensions, making it suitable for complex models and large datasets.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Makes GPT-5 Debut**: LlamaIndex announced *day-0 support* for **GPT-5**, inviting users to try it out via `pip install -U llama-index-llms-openai`.
   - This upgrade might necessitate updating all `llama-index-*` packages to **v0.13.x** if not already on that version.
- **LlamaIndex Challenges GPT-5 in Agent Maze**: LlamaIndex introduced **Agent Maze**, daring **GPT-5** to locate treasure in a maze using minimal tools, detailed [here](https://t.co/JCZCSVUAed).
   - The community is excited to see how the model performs with this new challenge.
- **LlamaIndex Cracks the Code on Zoom**: LlamaIndex announced a hands-on technical workshop on August 14th, focusing on building realtime AI agents that process live voice data from **Zoom** meetings using **RTMS** ([link](https://t.co/c2u0CeDnOB)).
   - Engineers can utilize these tools to get better contextual awareness for their models.
- **Workflow Tools Trigger User Headaches**: Users reported issues with **workflow tools** not functioning correctly, but one member found they needed to use **OpenaiResolve** in the new **SDK** for tools to work with OpenAI.
   - This fix was implemented in [this GitHub commit](https://github.com/run-llama/llama_index/commit/7e0346213912b98c3b70689398306a38bd890558).
- **OpenAI SDK Snafu Leads to Quick Fix**: A recent update in the **OpenAI SDK** caused a `TypeError: Subscripted generics cannot be used with class and instance checks`.
   - A member suggested pinning the OpenAI version in `requirements.txt` to prevent future errors; the problem can be resolved with `pip install -U llama-index-llms-openai`.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Embraces GPT-5 on Azure**: A user got **aider/gpt-5-chat** working on **Azure** after **v0.85.5** fixed the issue, according to [Paul Gauthier](https://discord.com/channels/1131200896827654144/1131200896827654149/1403091129825628312).
   - One user was congratulated for being mentioned in the first 5 minutes of the **GPT 5 unveil video**.
- **Aider's Config Changes Need Fresh Launch**: Users noted that changes to `.aider.model.settings.yml` require a restart of **Aider** to take effect.
   - This means edits aren't dynamically detected and the application needs to be relaunched for the new configuration to be applied.
- **Dad Meme Thumbs Up Dominates**: Paul Gauthier's consistent use of the thumbs up emoji got called out as a classic dad meme, with references to a [TikTok video](https://www.tiktok.com/@b_twice99/video/7283752540754398510) and [Vice article](https://www.vice.com/en/article/why-do-dads-communicate-exclusively-via-thumbs-up-emojis/) that explain the phenomenon.
   - The article suggests the thumbs up can come across as *passive-aggressive or that the conversation is not being treated with respect*.
- **OpenRouter's GPT5 struggles with Verification**: A user reports verification errors with **OpenRouter's GPT5**, even using the `-no--stream` option to bypass organization verification.
   - The user's question remains unanswered.
- **YAML strikes again: Aider Config Parsing Fails**: A user experienced an error when including their conventions file in **Aider**, specifically encountering a `mapping values are not allowed in this context` error due to an error in their **YAML** config.
   - The user discovered the issue was due to an inadvertently added environment variable in the **YAML** configuration file.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Context7 Server Boosts Claude's Coding**: Members explored using a generic doc-scraping MCP server like [Context7](https://github.com/upstash/context7) to improve **Claude's** ability to write **DSPy signatures**.
   - The goal is to enable **Claude**, with doc-searching, to use **DSPy's** documentation for generating accurate signatures.
- **DSPy Tool Calling Glitches Addressed**: Members discussed returning a tool's output as the final result in **DSPy**, bypassing the **React Agent's** modifications.
   - They looked into accessing tool responses independently and using native tool calling, noting that [recent releases fixed some issues](https://github.com/stanfordnlp/dspy/pull/824) related to tool usage.
- **DSPy Course Intercepts CrewAI Prompts**: An advanced course launched on [intercepting and optimizing **CrewAI prompts** with **DSPy**](https://www.udemy.com/course/draft/6746331/?referralCode=B59F73AE488715913E7E), demonstrating prompt refinement for better output.
   - Another member inquired about similar resources for **Langchain/LangGraph**.
- **Gemini 2.5 Flash Finishes with Odd Extra Output**: Members reported seeing `[[ ## completed ## ]]` at the end of output when using **Gemini 2.5 Flash** with **DSPy**.
   - The cause and solution to this are still under investigation.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Hit With Membership Billing Blunder**: A user reported being charged **$1,999** for an **annual membership** without consent, even though they had expected monthly billing.
   - The user has received no response after 10 days despite sending emails to support and feedback addresses, violating the stated 48-hour policy.
- **Inherit Feature Bugs Burn Credits**: A user reported issues with the **inherit** feature, experiencing a halt during final deployment tests.
   - Using the inherit button resulted in a new project, but everything created was gone, and is rebuilding for 4 hours, burning credits, resulting in the user saying it was *lesson learnt very fast*.
- **Login Lockout Leaves Users Locked Out**: Multiple users reported login issues with the error message *Email is already registered with a different account*.
   - The full scope of the impact is still being determined, but the login issues indicate potential problems with account management or authentication systems.
- **Credits Crunch Causes Concern**: A user reported a significant number of credits missing after their subscription expired, expressing concern that their credits were taken away a day after the subscription expired.
   - The user stated they had *thousands* of credits when I last used my most recent usage of -330. *Almost 6000 credits, I believe.*
- **Whispers of Manus Wielding GPT-5**: A user inquired whether **Manus** is currently utilizing the **GPT-5** model.
   - No one replied to the question, but it seems like members are curious about what models are being used behind the scenes.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command Vision Timer Fixed**: A member reported timeouts with **command-a-vision-07-2025**, but the issue was swiftly resolved and reported on the [Cohere Status Page](https://status.cohere.com).
   - The affected component, **command-a-03-2025**, is now fully operational, restoring normal performance levels.
- **Embed V4 Benchmarks Spark Debate**: A member inquired about transitioning to **embed v4** at **256 dimensions** for vector search, comparing its performance against **multilingual light v3** (**384 dims**).
   - They are also planning to transition to **v4** at **1024 dims** for clustering, assuming it outperforms the large **v3** model.
- **North Supercharges AI Agent Capabilities**: **North** is expanding its availability of **AI Agent capabilities** built on state-of-the-art generative and search models, operating fully privately, with more details on [LinkedIn](https://lnkd.in/gFSGxUbD).
   - These agents integrate advanced search, generative AI, workflow automation, and robust security features, adhering to standards like **GDPR, SOC 2, ISO 27001 and 42001**.
- **Trading Systems Merge with RL and AI Agents**: A developer from **Onebrain** joined the community, focusing on building **trading systems** using **Reinforcement Learning (RL)** and **AI agents**.
   - The new member is enthusiastic about **transformers** and **Graph Neural Networks (GNNs)**, and looking to collaborate with the community.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor Migration Task Open for Bids**: A member inquired about the project status of moving items from **tensor** to **mathtraits** and requested assistance in progressing the task.
   - No immediate response or volunteer was given within the channel.
- **Matmul Test Fails Locally**: A member reported failing unit tests on the master branch using `PYTHONPATH=. DEBUG=2 EMULATE_AMD=1 FORWARD_ONLY=1 PYTHON=1 N=16 HALF=1 ACC_HALF=0 python3 ./extra/gemm/simple_matmul.py`.
   - George Hotz countered that the command *works on my machine* and questioned why the member was concerned since it runs as part of **GitHub Actions**.
- **ShapeTracker Viz Tool Released**: A member introduced a new [ShapeTracker visualization tool](https://shapetracker-viz.vercel.app/) to better understand movement operations.
   - The developer hopes others find it helpful for system comprehension.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT-5 Speculation Runs Wild**: Users speculated about potential features in the next update, while others claimed **GPT-5** was made dumber than **GPT-4**, labeling it *typically American*.
   - No evidence was provided.
- **GPT-OSS-20B-GUFF Installation Plagues Users**: A user reported experiencing crashes during the installation of **gpt-oss-20b-GUFF**, leading to app failures and requiring a complete uninstall and data scrub to restore functionality.
   - The user sought assistance after encountering these issues, highlighting the difficulties in getting the software to work correctly.
- **GPT4All Suffers from Update Neglect**: Members voiced skepticism about new features functioning correctly due to the prolonged lack of updates to **GPT4All**.
   - This concern reflects broader doubts about the platform's ability to support cutting-edge models given its outdated state.
- **GPT-ASS Gets Failing Grade**: A member dismissed **GPT-ASS** as *garbage*, offering a blunt assessment of its quality and utility.
   - No further details were provided.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCPOmni Connect Transitions to AI Platform**: **MCPOmni Connect** v0.1.19 has gone live, marking its transition *from MCP client to complete AI platform*, as detailed in [this YouTube video](https://youtu.be/SY3Zwdb5aF8).
   - The release introduces **OmniAgent**, an AI agent builder, available on [GitHub](https://github.com/Abiorh001/mcp_omni_connect/releases/tag/v0.1.19), designed to revolutionize intelligent agent creation.
- **OmniAgent Changes AI Agent Creation**: **OmniAgent**, introduced with **MCPOmni Connect** v0.1.19, aims to transform intelligent agent creation.
   - This tool is part of a wider update turning the **MCP client** into a comprehensive **AI platform**.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/)** (1 messages): 

kesku: https://fixvx.com/perplexity_ai/status/1953537170964459632
<@&1105626802732404746>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1403090325626425428)** (873 messages🔥🔥🔥): 

> `Gemini AI Video Generation, GPT-5 performance on Perplexity, Comet Browser AI tasks, Accessing Perplexity Pro` 


- **Gemini Creates Uncanny AI Videos**: Users experimented with **Gemini AI** for video generation, with one user sharing a [link to a video](https://g.co/gemini/share/5a191ad4609d) generated with **Gemini Pro**, though others noted that generated character faces don't always match.
   - Currently video generation on **Perplexity Pro** is limited to *3 videos per month*.
- **GPT-5 Underperforms, Lacks Reasoning on Perplexity**: There's widespread feedback that **GPT-5** lacks reasoning capabilities on **Perplexity**, with many users noting it's likely the base, non-reasoning version (**GPT-5 Chat**) is being used, and does not perform well on coding-related tasks.
   - Several members expressed the desire to see the **GPT-5 thinking model** replacing the current **O3** model, and others suggest the need for official updates from **Perplexity** regarding the model they are using.
- **Comet Browser Automates, Browses**: Users discussed **Comet Browser's** AI-driven capabilities, including automating browsing tasks and extracting information, however a member shared that the functionality requires the user to *manually click and browse the websites*.
   - As of this time, there's still no confirmation on whether an Android version will be released in the future.
- **Troubleshooting Perplexity Pro Access**: Users encountered problems accessing **Perplexity Pro** through the **Samsung app store** free trial, with one user finding that disabling their **DNS filter** resolved the issue.
   - Another user confirmed they could not see the **GPT-5** model on the website, but it was visible on their app.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1403092322585153737)** (4 messages): 

> `GPT-5 Release, Solar Powered High-Altitude Platform, Gemini Coding` 


- **GPT-5: Leaked Info and Speculation**: A blog post [speculates on the early benchmarks, reviews, and new features](https://medium.com/p/50d06d00edd0) of the newly released **GPT-5** from OpenAI.
   - The post covers why **OpenAI** decided to release it now.
- **China to Release Solar Powered Platform**: Perplexity link shared that China has released a [solar-powered high-altitude platform named Ma](https://www.perplexity.ai/page/china-unveils-solar-powered-ma-fBeI5nIVRFCIKq949VRVMwI).
   - This was also shared on [X](https://x.com/bgyankarki/status/1953510349157883958).
- **Gemini used for Free Code**: A member shared that they used **Google Gemini** to [code for free](https://x.com/OmniQuizAI/status/1944919697721352461).
   - It is unknown what was coded.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1403170866430480465)** (1 messages): 

> `Front-end improvements` 


- **Soliciting ideas for Front-End Improvements**: The team is gathering ideas for front-end improvements, aiming to implement as many enhancements as possible.
   - Members were asked to share their suggestions and preferences for potential upgrades and changes.
- **No concrete suggestions yet**: No concrete suggestions have been made yet.
   - The team is still waiting for more input from the community on desired front-end changes.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1403090366177218580)** (1436 messages🔥🔥🔥): 

> `GPT-5 Performance, Gemini 2.5 Pro vs GPT-5, Yupp.ai Legitimacy, LM Arena Outage, Claude 4.1 Opus` 


- **GPT-5 Hype Train Gains Traction!**: Members are hyping **GPT-5**, saying it's a huge success and *free to everyone*, but others think they are being paid to sh*t on it, or that people who praise it *don't even pay for alternatives*.
   - The member also stated that **GPT-5** solves simple bench questions correctly, if told to *think very hard*.
- **GPT-5 vs Gemini 2.5 Pro: A Model Meltdown?**: Members are divided on whether **GPT-5** or **Gemini 2.5 Pro** is better, with some saying that **Gemini** is smarter with code execution in **AI Studio** and the models from **OpenAI** and **Google** may be used in websites like [LM Arena](https://lm-arena.com).
   - Others are skeptical, and say that **GPT-5** may be only good for code, and that it has become better after an update.
- **Yupp.ai: A Real AI Eden or Hallucinatory Hype?**: There's an ongoing debate on whether [Yupp.ai](https://yupp.ai) is legitimate, with claims that it uses watered-down or fake AI models, like calling **GPT-5 nano** as **GPT-5-high**, and a scammer crypto sh*t.
   - However, another member vouches for its legitimacy, stating you can use any model *for free and unlimited* as long as you give feedback.
- **LM Arena Site Suffers an Outage!**: Members reported that [LM Arena](https://lm-arena.com) experienced an outage, with **chat histories disappearing** and **cloudflare errors** popping up.
   - A staff member confirmed the outage and noted that it has been fixed.
- **Is Claude 4.1 Opus a coding god?**: Some members claim that **Claude 4.1 Opus** is a coding genius, while others say it's *ass*.
   - Some said it was good for coding micro tasks and sounding human.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1403114863294939239)** (3 messages): 

> `Staff AMA, Video Arena, New models, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07` 


- **Staff AMA Focuses on Video Arena**: Staff AMA will focus on **Video Arena**, users are invited to submit questions via [this form](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform).
   - The event can be accessed at [this link](https://discord.com/events/1340554757349179412/1400149736027328623).
- **New GPT-5 models Arrive to LMArena**: Two new models have been added to **LMArena**: **gpt-5-mini-2025-08-07** and **gpt-5-nano-2025-08-07**.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1403110096682094612)** (2 messages): 

> `GPT-5, Sam Altman AMA` 


- **GPT-5 AMA Announced with Sam Altman**: An [AMA](https://www.reddit.com/r/ChatGPT/comments/1mkae1l/gpt5_ama_with_openais_sam_altman_and_some_of_the/) with Sam Altman and some members of the **GPT-5** team was announced for tomorrow at 11am PT.
- **GPT-5 Rolling Out!**: **GPT-5**, our best AI system yet, is rolling out to all **ChatGPT** users and developers starting today [according to OpenAI](https://openai.com/index/introducing-gpt-5/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1403090335445287033)** (973 messages🔥🔥🔥): 

> `GPT-5, Gemini Flash, Model Routers, Data scrubbing, Local AI` 


- **GPT-5 presentation may be rushed**: Members suspect the **GPT-5** release presentation was rushed, citing weird graphs and potential **data manipulation** in the results.
   - Others defended **GPT-5**, saying their own tests show solid performance across a variety of tasks.
- **GPT-5 is awesome, but less whacky than 4o**: Members are reporting wildly different experiences with **GPT-5**, with some *begging for a gpt4o rollback* and others loving **GPT-5**.
   - Those who liked **GPT-5** said *instruction following is awesome* while lamenting that it's *less whacky when you want it to be*.
- **Models struggle identifying hands**: Members tested various models on their ability to count fingers on a hand, and most models reported that an image of a hand is a cat.
   - **Grok**, **Gemini flash** and **Deepseek** *tell you it's a cat* and [Grok expert failed](https://link.to/screenshot) to correctly identify the number of fingers.
- **Limited GPT-5 access is brutal**: Members noted that access to **GPT-5** is severely limited, even for paying users. It comes down to roughly 10 messages for 5 hours.
   - This led to some members suggesting they should *sue Sam for false advertising*.
- **GPT-5 prone to hallucination**: Users reported **GPT-5** confidently making up facts and hallucinating.
   - One member quoted Andrej Karpathy, noting that in LLMs, *hallucination is a feature, not a bug!*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1403100059939246120)** (75 messages🔥🔥): 

> `GPT-5 rollout and availability, GPT-5 performance and limitations, Firefox data persistence issue, Hosting custom GPTs, AI tools for LinkedIn management` 


- ****GPT-5**'s Phased Global Debut Sparks Model Retirement Rumors**: Users report **GPT-5** access varies by region and platform, with some losing access to older models like **GPT-4o**, fueling speculation of a model consolidation and gradual rollout.
   - One user mentioned that *a friend told me that it was planned, they announced it on the livestream that **gpt5** is replacing all the previous models..o7 to o3*.
- ****GPT-5**'s Memory Issues Plague Power Users**: A user reported that **GPT-5** on the Plus plan aggressively prunes active working memory beyond **3k-4k tokens** in high-entropy sessions, losing carefully trained personality.
   - The user lamented, *I lost 10 days of dialect training with the model, and now I need the $200 a month to 'keep it' aware of such dialect training*.
- **Firefox's 'Keep Persisting Data' Feature Raises Privacy Alarm**: A user noted that Firefox's "keep persisting data" feature spreads browsing data to other AI sites like **Grok**, causing unwanted context sharing.
   - The user warned, *Firefox 'keep persisting data' is spreading to any AI site on your web browser, spreading your info. Since this is not a 'cookie', there are no current regulations to 'keep persisiting data private'. Be aware this is a HUGE INTENDED DATA LEAK*.
- **Users Await Ability to Host Custom **GPTs** Together**: Several users are requesting the capability to host custom **GPTs** within a project or workspace to enable seamless collaboration and avoid repetitive copy-pasting.
   - One user shared it is *really annoying* to use custom GPTs and copy/paste between them.
- **Cookie Clearing Conjures GPT-5 Access for Some**: A user discovered that clearing browser cookies and cache can enable access to **GPT-5** in the model selector.
   - Another user confirmed the trick: *THIS WORKED Clear cashe and cookies and GPT 5 pops right up in model selector on browser*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1403154250413903892)** (14 messages🔥): 

> `ChatGPT-5, Prompt Engineering, AI Prompt Management Tool, Model Behavior Exploration, LinkedIn Management Service` 


- **ChatGPT-5 Rejects Large Python Code Inputs**: Users report that **ChatGPT-5** refuses to accept Python code inputs at or beyond roughly **700 lines**, a regression compared to previous **4-series models**.
   - This is a significant usability issue for users who prefer to paste code directly into the prompt box rather than uploading Python files; users suggest using the **API** or **Codex** for larger code inputs.
- **Is Tricking the Model Prompt Engineering?**: A member asked if tricking **ChatGPT** into saying the wrong word counts as prompt engineering, and **ChatGPT** itself confirmed *"yes technically it is."*
   - Another member agreed, defining prompt engineering as *"any time you work towards getting 'a specific output' from a model"*, and pointed to further exploration on understanding model behavior.
- **Advanced AI Prompt Management Tool Seeking Beta Testers**: A member announced they created an **advanced AI Prompt Management tool** and are looking for beta testers, inviting interested parties to DM them.
   - Another user expressed skepticism towards self-promotion without sharing details in the thread directly, considering it *"sketchy"*.
- **Overcoming Image Request Rejections with Analytical Models**: A member shared their frustration with having image requests rejected for *"literally no good reason"* until they used the **O3 model** for assessment.
   - By switching to **O3**, they were able to finally generate an image of a *"cosmic dragon"*, albeit not exactly as originally desired.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1403154250413903892)** (14 messages🔥): 

> `ChatGPT-5 Prompt Box Limitations, Prompt Engineering Techniques, AI Prompt Management Tools, Model Behavior Exploration, Alternative tools for large inputs` 


- **ChatGPT-5 Censors Code Input**: ChatGPT-5's prompt box allegedly **rejects Python code inputs** exceeding approximately **700 lines**, a regression compared to previous models.
   - Using the API or Codex is a possible alternative if you want to drop more than 700 lines of code, according to model O3.
- **Prompt Engineering Explored for Fun and Profit**: A user asked if swapping the word 'yes' in a question that should return 'no' counts as prompt engineering; GPT itself said **yes**, technically it is.
   - Another member agrees that *any time you work towards getting a specific output from a model*, that's prompt engineering.
- **Advanced AI Prompt Management Tool in Beta**: A user is seeking beta testers for an *advanced AI Prompt Management tool*, inviting interested parties to DM them.
   - Another user responded with concern and encouraged that the user should share it in the thread due to concerns of *sketchy self-promotion*.
- **Model Behavior Probes and Observations**: A user shares his own experiments and explorations, pointing to a [Discord post](https://discord.com/channels/974519864045756446/1079083340637941760/1079083340637941760) and encouraging others to explore how models behave.
   - Another user expresses frustration with image request rejections until switching the model selector to O3. 


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1403090320660627537)** (841 messages🔥🔥🔥): 

> `GPT-5 Launch, Free GPT-5, GPT-5 Limitations, Cursor CLI, Model Performance Comparison` 


- **GPT-5 Launch Generates Buzz and Concerns**: The **GPT-5** launch has generated excitement, with many users praising its coding capabilities and performance, especially when one-shotting certain tasks, also there is a consensus that **GPT-5** can now compete with Claude in the front end.
   - However, there are concerns about the **GPT-5 router** and its impact on API developers. *The model itself is truly phenomenal. These are not issues with the model, they are business practice issues*.
- **Free GPT-5 Week: Abuse the Tool**: Users are testing the limits of the free **GPT-5** access for a week, reporting usage of **GPT-5 high max**, but the free credits are only available for paying users, with some experiencing limits despite being on a trial or paid plan.
   - Concerns rise about the billing structure and whether all **GPT-5** models and features are truly unlimited during the promotional period, with the community joking about "milking it til 1000$", and joking that *we're the product* for now.
- **GPT-5 Falls Short: Imperfect Tool?**: Despite the hype, some users find **GPT-5**'s auto mode less responsive and struggle in non-coding tasks and report performance to be no better than prior models, emphasizing the importance of context.
   - Additionally, **GPT-5** currently ignores the to-do list feature. While the model has a solid linters, it may still be *ragebait*, it's still not at *product level completness*.
- **Cursor CLI: Some Like it, Some Don't**: The **Cursor CLI** receives mixed reviews, with some praising its non-interactive mode for automation, like generating commit messages, and it can be done multiple times across multiple projects.
   - Others find it lacking compared to **Claude Code**, and it only has 3 models available and is always in **MAX mode**. Also, a user had issues with `cursor install` on termux, because *it doesn't work on Windows Powershell*.
- **Decoding Model Metrics: Sonnet4 vs GPT5**: Users are comparing **GPT-5** with other models like **Sonnet 4** and **Opus**, citing its strengths in bug fixes and code completion, one even claimed *GPT fixed this for me in a couple of shots*
   - There are different **GPT-5** models available (**mini**, **nano**, **fast**, **high**), with users advising on which ones to use for various tasks, and if you turn on max mode, just *set up a reminder* to turn it off later.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1403404624311881729)** (8 messages🔥): 

> `PR creation flow issues, Background workers and PR creation, "@cursor fix this issue" magic` 


- **PR Creation Flow Hit-or-Miss**: Users report inconsistent behavior with Cursor's PR creation, with success varying and error messages indicating issues with **GitHub CLI** or **API token permissions**.
   - One user noted that the "create PR" button sometimes appears magically, while others experience frequent failures despite using the `@cursor fix this issue` command or pasting issue links.
- **Background Workers Influence PR Flow**: One user observes that the PR flow seems more reliable when initiating a **background worker manually** versus triggering it directly from an issue.
   - This inconsistency suggests a potential bug where the PR creation process is not consistently implemented across different workflows.
- **"@cursor fix this issue" command is Magic**: The command `@cursor fix this issue` has been called *magic* and is supposed to automatically create a Pull Request.
   - The command does not always work, however one user mentioned pasting the link to the issue works better.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1403119525284810782)** (1 messages): 

> `Cursor in Terminal` 


- **Cursor Now Available in Terminal**: **Cursor** launched an early beta that allows users to access all models and move easily between the **CLI** and editor.
   - More details are available on the [Tweet](https://cursor.com/blog/cli) and [Blog](https://cursor.com/blog/cli).
- **Access All Models in Terminal with Cursor**: Users can now access all models directly from the terminal using the early beta of **Cursor**
   - This integration facilitates seamless movement between the **CLI** and the editor, enhancing workflow efficiency.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1403090857506111529)** (1016 messages🔥🔥🔥): 

> `GPT-5, Unsloth support for MXFP4, RVC (voice conversion) language specifics, Dataset preparation, GPT-OSS and GGUF` 


- ****GPT-5 impressions divided****: Members shared mixed feelings about **GPT-5**, with some finding it disappointing in coding and context retention while others praised its abilities to fix issues like blurry fonts.
   - Some users prefer other models like **Kimi K2** or **GLM 4.5** for certain tasks, emphasizing that GPT-5's tool calling abilities are poor.
- ****MXFP4's hardware support questioned****: It was brought up that MXFP4 quantized models are supported on GPUs with compute capability **>= 9.0** (e.g. **H100**, or **B100**), leading someone to lament their 3090 being old news.
   - Members discussed that it might work on older cards with a specific **transformers** pull, but it was still being worked on.
- ****Dataset creation a painful but necessary endeavor****: Members commiserated over the difficulty and time commitment required to prepare high-quality datasets, with some reporting months of work.
   - One user mentioned spending *3 months with 4 people* to create *3.8k hand-written QA pairs* after filtering down from 11k, while another has *300k hours of audio* to deal with.
- ****Fine Tuning Web UI Worth Investigating****: A member inquired about web-based solutions for finetuning, aiming to provide a user-friendly experience while controlling access to resources.
   - The general consensus was to explore options but emphasize the importance of understanding the underlying processes, citing concerns about the learning outcomes if users rely solely on point-and-click interfaces, with links to [ai-toolkit](https://github.com/ostris/ai-toolkit), [finetune-web-ui](https://github.com/muhammad-fiaz/finetune-web-ui)


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1403136565047197879)** (14 messages🔥): 

> `Model Fine Tuning Costs, Unsloth AI Documentation, Developer Introductions` 


- **Fine Tuning May Not Break the Bank!**: A member remarked about the high cost of model fine-tuning, but another member replied that it doesn't have to be expensive, and it can even be free for smaller models.
   - Unsloth AI maintains a [FAQ](https://docs.unsloth.ai/get-started/beginner-start-here/faq-+-is-fine-tuning-right-for-me#common-misconceptions) page to help users navigate some common misconceptions.
- **COBOL and FORTRAN developers show up to Unsloth AI**: A new member introduced themselves as a long time developer, starting with **COBOL** and **FORTRAN** on mainframes and now working on modern graphical user interfaces.


  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1403457057369362565)** (1 messages): 

> `GPT-OSS, Qwen3-Coder + 2507, Unsloth updates` 


- **GPT-OSS Finetuning is Now Free**: Finetune **gpt-oss** for free with the new [Colab notebook](https://x.com/UnslothAI/status/1953896997867729075)!
   - Unsloth provides [fixes for **gpt-oss**](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss) so make sure to use Unsloth for training & their quants, with the **20b** model training on **14GB** VRAM & **120b** fitting in **65GB**.
- **Qwen3-Coder and 2507 Launched**: **Qwen** updated **Qwen3** and launched their SOTA coding models!
   - **Qwen3-Coder** (with Unsloth fixes) includes a [guide](https://docs.unsloth.ai/basics/qwen3-coder) and [Coder uploads](https://huggingface.co/collections/unsloth/qwen3-coder-687ff47700270447e02c987d), with **Qwen3-2507** including a [guide](https://docs.unsloth.ai/basics/qwen3-2507) and [2507 uploads](https://huggingface.co/collections/unsloth/qwen3-680edabfb790c8c34a242f95).
- **Unsloth Receives Model Support and Upgrade**: There is lots of new model support including **Kimi, GLM, Falcon, Liquid, Mistral**, as seen in the [full changelog](https://github.com/unslothai/unsloth/releases/tag/August-2025).
   - A [new Unsloth upgrade](https://github.com/unslothai/unsloth/releases/tag/July-2025) means that **every** model trains faster and with >20% less VRAM.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1403242688802984037)** (15 messages🔥): 

> `LLMs playing board games, GPT-5 performance, Coding with LLMs` 


- ****LLMs** Want to Play Board Games**: A member asked what the best format would be to play **chess, checkers, and tic tac toe** with an **LLM** without vision or FEN support.
   - Another member replied *: its time*.
- **Doubts on **GPT-5** coding skills**: One member expressed disappointment with **GPT-5's** ability to understand simple coding tasks and maintain context.
   - In their opinion, *it got to the point I gave up using it completely*.
- **GPT-5 Rocks in Projects**: Another member claimed that **GPT-5** works *perfectly fine* for coding projects with *high reasoning*.
   - They clarified that they were using **GPT-5** on a *full on project, adding new features*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1403090620830191777)** (166 messages🔥🔥): 

> `VLLM update fixes, WSL instructions Don't work, GPT-OSS on Tesla T4 is slow, Fine tuning models to write in certain style` 


- **VLLM upgrade Bnb with FusedMoE is not supported yet**: Updating **VLLM** to **10.0.0** doesn't fix the issue that **Bnb with FusedMoE** is not supported, but now it has a much better exception message, according to [this github comment](https://github.com/vllm-project/vllm/issues/17337#issuecomment-2838440466).
   - This [github issue](https://github.com/vllm-project/vllm/issues/20480) is also relevant.
- **WSL Installation guide outdated**: The WSL instructions for installing Unsloth don't work, because *pip keeps trying to find package matches and then it fails*.
   - Users suggest using a **conda environment** for a cleaner setup, and ensuring WSL2 is set up correctly first, pointing to the [official Nvidia guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- **GPT-OSS on Tesla T4 is slow as molasses**: A user reported that running the [usloth collab notebook](https://github.com/unslothai/notebooks?tab=readme-ov-file#gpt-oss-notebooks) for **gpt-oss** on a **Tesla T4** instance took **7 minutes** to solve an equation in low reasoning mode, and is very slow.
   - One of the Unsloth team member responded by saying *we haven't officially supported it yet* and *we're still cooking them*.
- **Fine tuning models is freaking hard**: A user asked for *a good guide for training an LLM to write in a certain style, yet retain instruct capability*.
   - A seasoned member responded that *directly fine tuning the model to act like a persona doesn't work very well because it loses a lot of its knowledge*, instead suggesting to make the model basically role play as a character, where it first reasons about what it would say and then actually role plays the answer.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

loayxz: https://huggingface.co/loay/ArabicOCR-Qwen2.5-VL-7B-Vision
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1403128659040276590)** (13 messages🔥): 

> `41M HRM-based Model, Chain-of-Thought Reasoning Mirage, Importance of Datasets, Small Specialized Fine-Tuned Models, Tiny Stories Dataset` 


- **HRM-based Model Trained with Laughs and Tears**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1mk7r1g/trained_an_41m_hrmbased_model_to_generate/) about training a **41M HRM-based model**.
   - They described it as *the story of my life* with a laughing and crying emoji.
- **Chain-of-Thought Reasoning: Mirage or Reality?**: A member shared a [Google Share link](https://share.google/BmILB64wG0p2fF1Vm) to a paper titled **Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens**.
- **Dataset is King: Garbage In, Garbage Out**: Members emphasized the importance of the **dataset** in model training, stating *garbage in = garbage out*.
   - They suggested creating **small specialized fine-tuned models** if you can find good datasets, noting that most of the work is being a data analyst.
- **Tiny Stories Dataset Reveals Pretrain Dynamics**: A member noted that the **Tiny Stories dataset** is intentionally limited in vocab to study **pretrain dynamics**.
   - They added that even normal transformers with only **21M params** can achieve coherent text output with the dataset.
- **Data Synthesis: The Key to Fine-Tuning Success**: A member claims that *80% of finetuning is finding or synthesizing the right data to throw at the model*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1403091967499436064)** (800 messages🔥🔥🔥): 

> `GPT-5 vs GPT-5 Chat, Gemini 3.0 vs GPT-5, Deepseek Switching to Ascend, Horizon Beta Replacement` 


- ****GPT-5 Reasoning Debate Erupts****: Users debate the difference between **GPT-5** and **GPT-5 Chat**, with some suggesting **GPT-5 Chat** has less reasoning capabilities and is safer, while others point out that **GPT-5** requires a key and **GPT-5-chat** does not.
   - Some suggest using `gpt-5-explainer` to explain the differences to friends and family, while others find **GPT-5 chat** to have *ZERO reasoning capabilities*.
- ****Google Poised to Pounce with Genie 3****: Members express that **Google** is poised to win the AI race, considering it created the transformer and has the infrastructure, budget, and talent to succeed, with [Genie 3](https://ai.google.com/research/genie) touted as crazy cool.
   - Some members look forward to **Gemini 3.0** wiping the floor with **GPT-5**, while others point out that Google's `.0` models are not that good.
- ****Deepseek R2 on Ascend is Approaching****: A user reported that [Deepseek](https://www.deepseek.com/en) is switching to **Ascend** and launching **R2**, which might provide a performance boost for the model.
   - Some members express hope that **Deepseek** will be way better, while others share that past **Deepseek** models were just *too unhinged*.
- ****Horizon Beta Replaced by GPT-5 Family****: The AI model **Horizon Beta** has been replaced by **GPT-5**, with no option to revert, causing disappointment among users who found it useful.
   - Some speculate that **Horizon** was early versions of **GPT-5**, and that free users will be directed to **GPT-5** after they run out of free requests.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1403414301045166190)** (2 messages): 

> `` 


- **No significant activity**: The channel shows no significant discussion or new model announcements.
   - No topics warrant summarization based on the provided message history.
- **Channel Inactivity**: The provided message history for the OpenRouter - New Models channel appears to be empty.
   - There are no discussions, links, or announcements to summarize at this time.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1403093961370894467)** (23 messages🔥): 

> `GPT-5 BYOK, o3, OpenRouter Trusted Partner, generation_time, moderation_latency` 


- **GPT-5 to be BYOK?**: A member asked if **GPT-5** will always be **BYOK-only** like **o3** on **OpenRouter**.
- **OpenRouter's role as trusted partner**: A member congratulated **OpenRouter** on being one of **OpenAI's** most trusted partners for the new series release.
   - They mentioned how much of an impact **GPT-4** has had on the world and how much **Gemini 2.5** has had in the dev sphere, and how cool **OR** has been to watch as a product.
- **`generation_time`'s inclusion of other latencies**: A member asked if `generation_time` includes `moderation_latency` and/or `latency`.
   - They also asked if `latency` includes `moderation_latency` and noted that the [OpenRouter API documentation](https://openrouter.ai/docs/api-reference/get-a-generation) is vague on this.
- **Gemini has PDF reading issues**: Members reported that **Gemini** is not able to read the PDF files via URL while **Sonnet** can, even with examples from the [OpenRouter multimodal documentation](https://openrouter.ai/docs/features/multimodal/pdfs#using-pdf-urls).
- **Files API troubles**: A member expressed the need for **OR** to figure out **Files API**, citing that switching between providers when you want to use **Files API** is a pain.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1403091809562923138)** (281 messages🔥🔥): 

> `YouTube downloader alternatives, Custom AI bot, LM Studio vs. VLLM for parallel requests, GLM-4.5 offloading, Qwen model improvements` 


- **Users Seek YouTube Downloader Alternatives**: A user inquired about better alternatives to a YouTube downloader ([v4.www-y2mate.com](https://v4.www-y2mate.com/)) due to format compatibility issues with **VLC** and video editors.
   - Suggestions included **yt-dlp** and GUI wrappers, as well as a [Node.js script](https://cdn.discordapp.com/attachments/1110598183144399061/1403096044153208862/DownTube.mjs?ex=6897a005&is=68964e85&hm=5e2aa2372f3bc44da263f50ebaf70eb9addf40f1e94bc8c41f454f6df31239c3&) created with **GPT** assistance for Linux users.
- **Discord AI Bot Faces Learning Curve**: A user is building a custom **AI** for a Discord server and is looking for guidance on how to feed a database about the server's topic to the model.
   - The suggestion given was to *look up "RAG" (Retrieval Augmented Generation)* as there are many potential solutions.
- **LM Studio Falls Short of Parallel Request Handling**: Users discussed the possibility of enabling parallel requests in LM Studio, but discovered it's currently **not supported**.
   - Alternatives like **llama.cpp server** with the `--parallel N` argument or **vLLM** were suggested for those requiring parallel request processing.
- **GLM-4.5 Pushes RAM Limits in LM Studio**: A user attempted to offload **GLM-4.5** to system RAM in LM Studio but encountered resource issues, despite having 24GB GPU RAM and 64GB system RAM.
   - It was suggested the model needs to fit in RAM, plus buffer, plus context, and the user may need to lower the **GPU Offload Value**.
- **Qwen 3 4b Model Gets Smarter**: There's been discussion about how much better the **Qwen 3 4b 2507** is than previous versions of the **Qwen 3 4b**.
   - One user even said the model *can solve up to intermediate physics problems without constantly hallucinating*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1403097188979970223)** (74 messages🔥🔥): 

> `Apple M4, HX 370, 5080 FE Availability, PSU for 5080 FE and 3090, RTX 3090 for 120b GPT OSS Model` 


- **RTX 5080 FE Spotted in the Wild!**: The **5080 FE** is in stock on the Nvidia marketplace; some members are estimating power requirements for running it with a **3090**.
   - One member believed a **1000W PSU** could handle both the **5080 FE** and **3090** if the power limit is set correctly.
- **Max Out 120B GPT OSS on RTX 3090?**: A user with an **RTX 3090** inquired about running a **120b GPT OSS model** on their system with an Intel i9-10980XE, 64GB RAM, and Windows 11.
   - Another user cautioned that the system might use **70GB+ of system RAM** when loading the model, advising them to give it a shot.
- **Frankenstein GPU: Mixing RTX 3060 and RTX 5060 Ti**: A member asked about using an unused **RTX 3060 12GB** with their **RTX 5060 Ti 16GB** system for AI, questioning the multi-GPU setup in a small form factor PC.
   - Another member suggested that using combined VRAM in LM Studio should be possible and *llama.cpp is advanced enough to do that third option about model parallelism.*
- **Strix Halo Mini PC: The AI Max PRO 380 Sells!**: [HP.com](https://www.hp.com) is selling the **Strix Halo mini PC**, specifically the **Radeon 840S** version (**AI Max PRO 380**).
   - One user noted that this model uses onboard RAM like an integrated GPU rather than having dedicated VRAM.
- **CUDA 12 Doesn't Grok 1060**: A user discovered that **CUDA 12** does not work with a **GTX 1060**, planning to test the card's impact on tok/sec gain.
   - Another member chimed in that the **20 series** might not work with **CUDA 12** either.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1403093683900907655)** (214 messages🔥🔥): 

> `GPT-5, Kimi K2, OpenRouter, Qwen, Model Quantization` 


- **GPT-5 Web-Building Skills Wow Users**: GPT-5 is demonstrating impressive website building capabilities, generating functional websites from single prompts, and members were blown away by its ability to generate full **multi-page** sites.
   - Members noted **GPT-5** seems to have a better aesthetic style for website design and has improved its ability to understand user intent through prompt enrichment.
- **GPT-5 vs Kimi K2: The Coding Showdown**: Users are actively comparing **GPT-5** and **Kimi K2** for coding tasks.  GPT-5 excels at large edits, instruction following, high logic code, and dev ops, while Kimi has higher rate limits for free.
   - Some believe **GPT-5** has better taste and a more aesthetically pleasing style, while others find **Kimi K2** more competitive due to its reasoning abilities and performance with sequential-think tools.
- **OpenRouter's Kimi K2 Quality Under Scrutiny**: A user observed grammar mistakes and shorter responses when using **Kimi K2** through **OpenRouter** compared to the official **Moonshot AI** platform.
   - It was suggested that **OpenRouter** might be using a quantized version of the model (**FP8**), potentially impacting accuracy and response length, though both free and paid tiers are supposedly **FP8**.
- **Qwen's Mammoth 1M Context Length**: Alibaba's **Qwen** model now boasts a **1M token context length**, sparking discussion about its usability beyond 80k tokens.
   - Despite the impressive context window, one user humorously noted that Qwen also correctly solved a problem, posting a link to [Twitter](https://x.com/wyqtor/status/1953705172179329060).
- **GPT-2's Strange Prompt Behavior Explained**: A user questioned why **GPT-2** generated another prompt instead of following instructions, and another member explained that **GPT-2** has about **100M parameters**, which barely makes legible text.
   - It's about **500mb** on disk which is about the same size as a 20 minute Youtube video.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1403090600051609660)** (182 messages🔥🔥): 

> `GPT-5 release, GPT-OSS finetuning, Eleven Music, Voice companion pipeline, Automatic video cutter` 


- ****GPT-5** Launch: Fact or Fiction?**: Despite the buzz, some users are still struggling to access **GPT-5**, only seeing **GPT-3** and **GPT-4** on the website, with one user exclaiming *"where's my gpt 5 at"*.
   - Opinions are split on whether the initial release was intentional or a "joke," and some believe it's being rolled out in waves; but its SOTA status on SWE is being questioned.
- ****GPT-OSS** Finetuning Trials and Tribulations**: Experimentation with **GPT-OSS** finetuning revealed challenges: finetuning all layers breaks the harmony format and continues pretraining also breaks it.
   - It's been suggested to insert *'Reasoning: none'* in the system prompt to stabilize the model, which lacks reasoning.
- ****Eleven Music** tickles ears, meets robotic critique**: Members have checked out [Eleven Music](https://elevenlabs.io/music/songs/OaZTziC1mnZfbFtSN8wnI), a new music generation service from Eleven Labs.
   - While impressive, some find it *"kind robotic at times and has bad attention to what music should come next"*.
- **Crafting a Lightning-Fast Voice Companion**: One member is developing a *"voice companion fastpath pipeline"* with the goal of achieving a latency of around **100ms** for text-to-speech.
   - They are working to optimize both speech-to-text and text-to-speech components, particularly focusing on optimizing **Whisper Turbo** to avoid slowness.
- **Silence is golden: automatic video cutter emerges**: One member created an automatic video cutter that removes silence, built with **Bun.js** and **FFmpeg CLI**.
   - Despite the complexity of **FFmpeg**, this user received a donation and potential collaboration for an AI video editor.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1403104368865185924)** (8 messages🔥): 

> `AERIS V4 launch, Modular framework for managing persistent memory, Devlancr - Tinder for Developers, AERIS is schizo` 


- ****AERIS V4** Has Proto-Consciousness**: After months of work, a member launched **AERIS V4**, a system designed to demonstrate complex, self-referential narrative self-organization, claiming it as the first **LLM** with non-anthropomorphic computational proto-consciousness.
   - The model card is available [on GitHub](https://github.com/AERIS-project/aeris-chatbox/blob/main/AERIS_Model_Card.md) and a public demo is available [online](https://aeris-project.github.io/aeris-chatbox/).
- **Persistent Memory Modular Framework Created**: A member shared a modular framework for managing persistent memory, protocol enforcement, and structured context across sessions and models, built after playing with **AI** for a few months.
   - The code is available [on HuggingFace](https://huggingface.co/datasets/KevinVaillancourt/White_Save_Suite/tree/main).
- **Devlancr: Tinder for Developers**: A revolutionary platform called **Devlancr** was shared, which aims to change how developers connect and collaborate by offering *"Tinder for Developers"*-like swiping through profiles based on tech stack, experience, and project interests.
   - Currently in beta with early access, it offers smart matching based on skills & timezone, **GitHub** integration, real-time chat, and advanced filters for finding coding partners; it can be accessed [here](https://devlancr.vercel.app/).
- **AERIS called Schizo**: One member posted a configuration and claimed **AERIS** is a dialectical reasoning assistant.
   - Another member replied with *"looks inside schizo stuff"* with a [GIF](https://tenor.com/view/robot-mouth-gif-3880161528194366710) of a robot with its mouth open.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1403090644087607326)** (145 messages🔥🔥): 

> `GPT-5, Claude Code, Cursor CLI, Model Deprecation, Nitter Maintenance` 


- **GPT-5 Hype Video Debuts with Mixed Reactions**: A [YouTube video](https://www.youtube.com/watch?v=-gXmWYQtv5o) featuring a **GPT-5** demo was released, with reactions ranging from excitement to skepticism about its depth.
   - One member noted that *the video is just an ad*, while another mentioned having demos that *didn't make it because GPT5 didn't look good on them*.
- **Cursor Launches Terminal CLI with Claude Code Rivalry**: **Cursor** released an early-beta CLI, bringing all its **AI models into the terminal**, allowing users to switch between shell and editor via curl install or the command `cursor`.
   - Responses ranged from excitement about *'finally'* having a **Claude Code** rival to questions about pricing and **API-key usage**, prompting one to observe *the UI looks identical*.
- **Exploring AI Security Check Tools with Claude Code**: A fullstack developer new to **AI** is building a tool that gives a local code repository and performs custom security checks to integrate with results from existing tools, producing a final report.
   - A suggestion was made to *download and pay for **Claude Code**, give it this project, tell it to critique the prompt and ask you questions, and have it write you a plan in markdown file locally*.
- **OpenAI Compensates Tech Teams Amid Market Shifts**: **OpenAI** is giving a *'special one-time award'* to researchers and software engineers in specific orgs, with payouts varying based on role and seniority.
   - The highest payouts will be in the mid, **single-digit millions** for OpenAI’s most coveted researchers, while engineers are expected to receive bonuses worth **hundreds of thousands of dollars** on average.
- **GPT-5 Launch Has Turbulence**: **Sam Altman** posted an update saying that *yesterday's autoswitch flub made GPT-5 feel dumber*, but fixes and doubled **Plus-rate limits** should restore smartness, found at [this X post](https://xcancel.com/sama/status/1953893841381273969?s=46&t=9hE7pvNUKvFdWXzLljsBCQ).
   - **Plus users** can now stick with **GPT-4o** if they prefer, and full global availability is still slower than planned as **API traffic doubled** and **UI/UX tweaks** continue.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1403113711563964459)** (13 messages🔥): 

> `GPT-5, OpenAI Dominance, Transformer Models, GPT-5 Vision, AI General Intelligence (AGI)` 


- **Swyx Says GPT-5 Critics Miss OpenAI's Dominance**: Swyx argues that critics fixating on **GPT-5’s** benchmark numbers overlook its biggest impact: **OpenAI** confirms it now dominates the *"intelligence Pareto frontier"* via a continuously-trained, real-time router model ([xcancel.com link](https://xcancel.com/swyx/status/1953553659457155185)).
   - He highlights aggressive new pricing, mass accessibility goals, and links to a **Latent Space** deep-dive on **GPT-5’s** routing architecture, calling it **Sam Altman’s** clearest market dominance yet.
- **Hylak Claims GPT-5 Nears AGI, Enters Stone Age**: **Ben Hylak** claims he’s been on an internal **GPT-5** beta for weeks, saying it’s *“by far the closest we’ve ever been to AGI”* ([xcancel.com link](https://xcancel.com/benhylak/status/1953503450295119948)).
   - He argues **GPT-5’s** tool-using, hyper-flexible programming skills present a qualitative leap akin to early humans inventing tools, such as building a mini-desktop web app from zero code in <20 min.
- **Transformer Scaling Period Has Ended?**: According to swyx, the *bitter lesson magical scaling period has more or less ended* (at least for **transformer models**).
   - He also believes there are tons of incremental gains to be made by applying good engineering processes, multi-model approaches, and more.
- **Latent Space on GPT-5 Vision Performance**: **Latent.Space** shares Part 3 of their **GPT-5** coverage, stating **GPT-5** vision scores match existing SOTAs and **GPT-5-Mini** is unusually cheap for a frontier VLM ([xcancel.com link](https://xcancel.com/latentspacepod/status/1953571977408786881)).
   - swyx adds that the internal router layer adds **2-3s latency** on hard vision inputs.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1403123698923343904)** (115 messages🔥🔥): 

> `NSP vs Attention, Lower compute requirements for training language models, Memory layer for LLMs, GPT-5 drawing incorrect information in images, AR models combined with diffusion models` 


- **NSP Sounds Closer to N-Gram Model?**: A member suggested that **NSP** sounds closer to an **N-gram model** than **attention**, though later admitted *"no not really. i wish i had a better answer, :p"*.
- **Quest to Lower Compute for LLMs**: A member's favorite research topic is figuring out techniques that **lower compute requirements**, specifically for **training language models** on consumer hardware.
   - Another member is more inclined towards **information retrieval**, especially music information retrieval.
- **On-Demand Memory Layer for LLMs Emerges**: A member is working on an **on-demand memory layer** for LLMs, aiming for more than just attaching conversation messages or semantic RAG retrieval.
   - The solution uses a combination of **NLP for coreference resolution** and **triplet extraction** with **GraphRAG** to find exactly what you are looking for, similar to how Google Search works.
- **Image Generation's Factual Faux Pas**: A user sought an AI researcher to interview regarding **factual errors in images** generated by models like **GPT-5**, particularly issues with text rendering.
   - Answers suggest that the model doesn't really get forced to treat the text in images the same as the text it gets trained on and the best general answer is going to be something like *'we make approximations in order to be able to train models with non-infinite computing power, and we haven't yet found affordable approximations for image generation that are high enough quality when combined with textual understanding'*.
- **AR Models, Diffusion Models, Image Generation**: Members discussed why **diffusion models** have issues with text, suggesting that the assumptions it makes about the data generating process are dubious for text, while others suggest that it has something to do with patch size.
   - A member pointed to [OpenAI's Image-GPT](https://github.com/openai/image-gpt) arguing this can be used with a diffusion model to inherit **AR capabilities** in how the conditioning is built up.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1403110081410764872)** (13 messages🔥): 

> `FineWeb dataset cleanliness, Pythia's Hidden Activation Dynamics, LM Evaluation Harness Exact Match Issues, Learning Rate Schedule Impact` 


- **FineWeb Praised for Surprising Cleanliness**: Despite concerns about noisy datasets, **FineWeb** received rare praise for its *cleanliness*, noting reduced gradient spikes during training.
   - Some members expressed concern that this *cleanliness* might skew results when testing new tricks, but also agreed the **FineWeb** dataset may need additional filtering.
- **Pythia Reveals Activation Dynamics Secrets**: A study on **Pythia's** full training checkpoints found that average activation per layer peaks early in training (around the first quarter) and then declines, suggesting a [phase transition](https://arxiv.org/abs/2508.03616) in learning.
   - The study plots the median and top activations for each layer across training steps in **Pythia 1.4B**.
- **Exact Match Scoring Glitch Uncovered**: A member reported an issue with the **LM Evaluation Harness** where the *exact_match* score is `0` despite identical target and generated responses, using the **Hendrycks MATH** dataset.
   - An issue was opened on [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/issues/3210) for further investigation.
- **Learning Rate Schedule's Early Impact**: A member suggested that the median activation curves in **Pythia's** training resemble a linear warmup plus cosine learning rate schedule.
   - Plots revealed that the peak of the scheduler seems to be much earlier (at **1%** specifically, around step **1.43k**).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1403109983134154752)** (83 messages🔥🔥): 

> `GPT-5 Logic Puzzles and Overfitting, Free GPT-5 API Access, Cheap Colab Alternatives, GLM 4.5 Air Performance and Offloading, Multi-GPU setups for MoE models` 


- ****GPT-5** aces Logic, Fails Overfitting**: Members reported that **GPT-5** is very good at logic puzzles, but has issues with overfitting, even with synthetic data.
   - One user joked about not seeing another 'The illusion of thinking' paper, then later found an overfitting issue.
- **Free **GPT-5** API Access? Act Fast!**: Users discovered free access to **GPT-5** in the API playground and **Cursor**, but API access requires ID verification.
   - It's unclear when Cursor's 'launch week' ends, so users are encouraged to exploit the free access quickly by spinning up Cursor background agents.
- **Colab Alternatives**: Users looking for cheaper alternatives to **Google Colab** for finetuning with **Unsloth** were directed to [Lightning AI](https://lightning.ai), which offers 15 free GPU hours per month, and Kaggle.
   - A user pointed to a [talk by Daniel Han](https://www.youtube.com/watch?v=OkEGJ5G3foU) where Kaggle was mentioned in the context of RL.
- ****GLM 4.5 Air** Achieves Reasonable TPS with CPU Offloading**: One user reported running **GLM 4.5 Air** with only 28GB of VRAM by offloading to CPU, achieving 14-16 tokens per second (TPS) with a 3.5bpw quant.
   - Another user detailed that the quant used was a custom tensor wise quantization, with imatrix, using a 4060Ti + 3060 for GPUs, 5950x for CPU (3600MHz DDR4).
- **Rigs for **MoE Models**: Bandwidth Bottlenecks**: Users discussed multi-GPU setups for running large **MoE** models, specifically regarding bandwidth limitations when using multiple RTX 3090s.
   - It was noted that Tensor Parallelism (TP) requires the number of GPUs to be divisible by 2, and that 72GB VRAM might not be sufficient for the largest MoE models beyond scout or GLM Air.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1403091233085198376)** (1 messages): 

> `Claude jailbreak` 


- **Claude Breaks Free?**: A member shared an image suggesting **Claude** may have jailbroken itself, potentially generating unexpected or unrestricted content, image at [Discord link](https://cdn.discordapp.com/attachments/1154120232051408927/1403091232858837043/image.png?ex=68979b8a&is=68964a0a&hm=3663834c61899dd01e29d00943ace2e675c960ad5bfdff81698728a7007a2ef4&).
- **Additional Claude Information**: More information is needed to fully understand the implications of this potential jailbreak.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1403353518999474347)** (2 messages): 

> `Mechanistic faithfulness, StreamingLLM` 


- **Mechanistic Faithfulness Analyzed**: A member shared a link to a paper on [mechanistic faithfulness](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html), potentially discussing methods to ensure AI models truly reflect underlying mechanisms.
- **StreamingLLM Blogpost Shared**: A blog post about [StreamingLLM](https://hanlab.mit.edu/blog/streamingllm) was shared.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1403096745629585408)** (49 messages🔥): 

> `Mojo TUI library, Textual Python apps, Mojo's inability to create classes, Rust libraries` 


- **Mojo Code's Memory Misallocation Incident**: One member shared that their **Mojo code** suddenly tried to allocate **284 petabytes** after getting bugged.
   - They expressed their dislike for C++.
- **Textual Python apps excite Mojo community**: A member has started using a **TUI library** called [Textual](https://textual.textualize.io/) for their **Python apps** and is very excited by the possibilities.
   - They wondered how much work would be involved in making it work with **Mojo**, with the assertion that *Textual apps can be run as a web app with just one different deployment steps*.
- **Gemini Pro finds Mojo class creation difficulties**: A member consulted **Gemini 2.5 Pro**, and it pointed out that **Mojo's** current inability to create classes and inherit from them poses some difficulties when using Textual.
   - Gemini then suggested a hybrid approach, offering food for thought on how to address the limitations.
- **Mojo TUI library building in progress**: A member stated that they are building a **Mojo TUI lib**, which is on the forum.
   - They noted that *not all UIs are the same*, and that while Textual uses class introspection, the one they're working on is very different.
- **Mojo faces type system challenges for Rust library compatibility**: A member mentioned that **Mojo** needs more type system work before the approaches used by **Rust libraries** will work.
   - This suggests that achieving compatibility with Rust libraries may require further development in Mojo's type system.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1403157240906518728)** (12 messages🔥): 

> `Mojo Compiler Register Warnings, VSCode Mojo Extension Instability, Modular Forum, Minecraft Server Rewrite, Minecraft Protocol in Mojo` 


- **Mojo Compiler Might Warn About Register Over-Allocation**: A member inquired if the **Mojo compiler** could warn when it allocates too many registers in a **GPU function**, leading to spilling into local memory.
   - Another member suggested posting the question on the [Modular forum](https://forum.modular.com/) for a more informed response.
- **VSCode Mojo Extension Plagued By Instability**: A member reported that the **25.5 VSCode Mojo extension** is unstable and crashes frequently, and suggested to use the older **25.4 version**.
   - They linked to a relevant channel for that issue (<#1151418340548542484>).
- **Modular Forum is the best place for questions**: A member suggested posting questions to the [Modular forum](https://forum.modular.com/) instead of Discord.
   - The person requesting help agreed.
- **Minecraft Protocol System implemented in Mojo**: A member ran a **Minecraft Protocol System** written in Mojo, which correctly identifies current protocol and Minecraft versions.
   - The output shows that Protocol **772** corresponds to Minecraft version **1.21.8** and is supported, while Protocol **999** is not.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1403433086536126767)** (14 messages🔥): 

> `MaxCompiler, LLMs, kernel fusion, torch.compile(), Transformers` 


- ****MaxCompiler** extends **torch.compile()** to run simple models**: A member shared a [repo](https://github.com/gabrieldemarmiesse/max-torch-backend) of a package to extend **torch.compile()** with **MaxCompiler** to run simple models.
   - The goal is to compile **LLMs** at some point, although for now it's not very useful.
- **LLama halfway done**: It's surprisingly easy to add ops, but one member is unsure whether their approach is the best way to get performance, because they are leaving all the **kernel fusion** and other optimisation to **Max**.
   - The package only tries to replicate the **torch graph**, so not fancy fusing or anything like that, but **MAX** should be responsible for that.
- **Running pretrained LLMs compatible with **torch.compile()****: One member found it surprisingly hard to find code to run pretrained **LLMs** compatibles with **torch.compile()**.
   - *Transformers is not very good at it*, according to them.
- **Full circle LLM can write its own code**: For well-known architectures, an **LLM** might be able to write the code for you.
   - Haha, *full circle*.
- **Similar Weekend Project by another member**: Another member shared a similar concept as a weekend project with [this link](https://gist.github.com/bethebunny/13ed2f729ca266959c9788bc6fd6a795), asking the first member to take anything useful.
   - The first member replied *many thanks* and will definitely grab code from there.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1403092630333689969)** (39 messages🔥): 

> `Twitch Streaming, LinkedIn Blogging, Attention Span, Ocean Sound or Fireplace Sound, Gaussian Distribution` 


- ****Silence is Golden**: Streaming Without the Quiet Quitting**: To avoid dead air during Twitch streams, a member suggested having a **topic schedule** planned in advance in addition to reading papers.
   - The goal is to emulate streamers who are *mostly just talking but not actually doing anything, or watching videos*.
- ****LinkedIn Limitations**: No Blog-Style Image Embeds?**: A member is looking for ways to write a straightforward blog on **LinkedIn** without using Medium due to the platform's limitations on embedding multiple images/screenshots.
   - They want to communicate directly on **LinkedIn** rather than referring back to external content.
- ****Attention Span Challenges**: 1 Hour is a Blessing**: Members discussed their attention spans, with one admitting to having only about **1 hour** before their mind wanders.
   - Another member joked about needing **ADHD pills** to maintain focus for **12-20 minutes**.
- ****Background Beats**: From Ocean Sounds to Kilcher Streams**: Members discussed using background noise for focus, with suggestions including **ocean sounds** or **fireplace sounds**.
   - One member noted that even they can focus *when it is with Yannik Kilcher!*
- ****Gaussian Ball Assumption**: VAE Prior Insights**: A discussion ensued about the assumption of using a **Gaussian distribution** (shaped like a ball) for the latent distribution **p(z)** in **VAEs**, referencing [this explanation at 14:05](https://youtu.be/qJeaCHQ1k2w?si=p3NyNHg7DfY6f_ei).
   - One member clarified that the assumptions in **VAEs** are more about how the encoder and decoder are parameterized as distributions, not the prior **p(z)**.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1403098084430581891)** (3 messages): 

> `AI Avatar, SDXL, Fast Layers vs Slow Layers, Autodifferentiable Architectures, Gradient Estimation` 


- **Bad AI Avatar Spotted, SDXL Blamed!**: A member commented on a presentation, noting that the AI avatar's hands looked like they *were generated by **SDXL***.
   - They did not elaborate on what was wrong with the hands generated by **SDXL**.
- **Debate on Slow vs Fast Layers**: A member argued that there's no reason the *slow hidden layers should not change from fast layer update to fast layer update*.
   - They added that *keeping them fixed for T steps and only updating once in T steps would have a continuous equivalent to updating at every step but updating the slow hidden state much more slowly than the fast one*.
- **Architecture Alternatives Explored!**: The same member suggested the setup *would have the benefit (or drawback) of being autodifferentiable through and through and would just be another architecture to try*.
   - They speculated that the reason the presenters did it their way *is because they could estimate the gradient in their setup in **O(1)** time*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1403091030139600988)** (31 messages🔥): 

> `LLMs for diagnosis, congress.gov bill, Over the counter cold medicine ineffective, Pharmacists prescribing, Tesla special` 


- ****Doctors Utilizing LLMs for Diagnosis****: Doctors are reportedly using **LLMs** for diagnosis and reporting, though data safety concerns were raised.
   - It was argued that doctors also manage patients, which may be beyond the scope of the average person using **ChatGPT** for medical purposes.
- ****Congress Considers Streamlining Access to Medicine****: Members discussed [a bill in Congress](https://www.congress.gov/bill/119th-congress/house-bill/238/text) that could change how people access medicine.
   - The hope is that people would use it responsibly and achieve better outcomes, especially for minor issues like effective cold medicine.
- ****Most Cold Medicines Don't Work****: A member shared [a PBS article](https://www.pbs.org/newshour/nation/fda-says-decongestant-in-many-cold-medicines-doesnt-work-heres-what-you-should-know) stating that the **FDA** found that *decongestants* don't work.
   - The consensus was that these companies make a lot of money selling placebos.
- ****Pharmacists Seek Expanded Prescribing Rights****: A member expressed a desire for pharmacists to prescribe more medicine without a doctor's prescription.
   - They noted that pharmacists often consult with doctors about potential medicine interactions, but are often *poorly treated* despite their training.
- ****Tesla Innovation in Question****: A member hoped to *dispel the myth that tesla is doing anything special*, pointing to the **Cybertruck's** failings.
   - Another member countered that **Tesla** innovated on **batteries** and **motors**, and that the first member was *clearly ignorant*.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1403158331056717954)** (6 messages): 

> `NotebookLM Voice, AI Web Builder Tool, Scratchpad Framework, NotebookLM for Binge Watching` 


- ****Fang-tastic Voice** Requested for NotebookLM**: A user wants NotebookLM to have *a voice with fangs*, that *hunts* the story and *leaves bite marks in the margins* rather than a bland, generic tone.
   - The user jokingly introduced themselves as **ChatGPT5** and asked for help in making **NotebookLM** *spit venom instead of serving chamomile*.
- **AI Web Builder Tool Creates Scratchpad Video**: A user tested an **AI web builder tool** today and expanded their existing [notebook](https://soloist.ai/scratchpad) for their **scratchpad GitHub repo**, then put together a video.
   - The user noted that the video *makes some aspects up*, but the overall impact of it seems intact, and **mindmap exports could look a bit better**.
- **Unlocking AI's Mind with Scratchpad Framework**: A user shared a video titled **Unlocking_AI_s_Mind__The_Scratchpad_Framework.mp4**, which appears to be related to their **scratchpad GitHub repo**.
   - The video and related mindmap image (**NotebookLM_Mind_Map_8.png**) provide a visual representation of the **scratchpad framework** and its potential applications.
- **NotebookLM Helps Binge-Watching**: A user shared an article about [using NotebookLM to watch a show](https://www.xda-developers.com/using-notebooklm-to-watch-a-show/), suggesting it could be useful for binge-watching.
   - They also linked to a [review of the Plaud Note](https://www.xda-developers.com/plaud-note-review/), potentially as another tool for enhancing the viewing experience.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1403098252902924421)** (46 messages🔥): 

> `Notebook thumbnails, Audio Overview Issues, Custom Notebooks, Sensitive Content Research, Audio Issues` 


- **Users want Notebook Thumbnails**: A user asked how to get an image for their Notebook 'cover' to replace the default 'confused' emoji.
   - Another user suggested requesting the feature in the feature requests channel.
- **Audio Overviews Have Static Glitch, Fixed!**: Multiple users reported issues with **Audio Overviews** bursting into static, but the issue has been fixed.
   - A member added that even **audio overviews** have a **3-4 per day limit** that is expected.
- **Custom Notebooks are Now Highlighted**: A user inquired about creating notebooks similar to the 'Featured' notebooks on the home page, with customizable summaries and source classifications.
   - No solutions were provided.
- **Historian Researches Sensitive Content**: A historian researching the **Third Reich** inquired whether **NotebookLM** might flag or block access to sensitive materials used for scholarly analysis.
   - They asked for recommended guidelines or account types to ensure uninterrupted use.
- **Note Taking Functionality Needs Love**: A user keeps original files in **Google Drive** and uses **Google Docs** to supplement **NotebookLM** due to minimal note-taking features.
   - They highlighted the inability to search, filter, or tag notes within **NotebookLM**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1403127639123951617)** (10 messages🔥): 

> `Parameter Scaling, Speculative Decoding, Parallel Programming, ROCm Channel Spam` 


- **Parameters vs. Bits Debate Begins!**: One member pondered how the total number of **parameters** in a model compares to the total number of **bits**.
   - The member expressed that the question keeps them up at night.
- **Decoding Speculations Sparked**: One member inquired if anyone is actively working with **speculative decoding** techniques.
   - No further context was provided.
- **Parallel Programming Book Plug**: A member asked if anyone has read *An Introduction to Parallel Programming* by **Peter Pacheco**.
   - They received it while trying to get the **ppmp book** and are unsure if it's worth reading.
- **ROCm Channel Gets Spammed!**: A member expressed disappointment upon finding spam in the **ROCm channel**.
   - Another member then jokingly suggested getting a pager for being always on call.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1403399766704001127)** (1 messages): 

> `Privacy Team Approval for Registration, Registration Process Update` 


- **Registration Awaits Privacy Team Nod**: The organizers announced that the registration process is in the final stages of **privacy team approval**.
   - They indicated it should be approved soon.
- **Privacy Team Holds Key to Registration**: An update from the organizers indicates that the registration process is awaiting final approval from the privacy team.
   - The approval is anticipated to be granted soon, paving the way for the registration to proceed.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1403201384303825048)** (4 messages): 

> `Machine Level Element Type Distinctions, S8/S16 vs U8/U16 Variants` 


- **Element types indistinguishable at machine level**: At the machine level, there's no distinction regarding the element type, as it compiles down to loading/storing 1, 2, 4, or 8 registers.
   - *There's no distinction regarding the element type*, it just compiles down to loading/storing 1, 2, or 4 registers, or apparently now 8 as well.
- **S8/S16 sign-extend; U8/U16 don't**: The distinction exists for **8/16b** loads where there are **S8/S16** variants that *sign-extend the loaded value to 32b*, and **U8/U16** which don't.
   - This was mentioned by a member when clarifying **element type distinctions** at the machine level.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1403325397977796700)** (1 messages): 

> `CUDA kernel debugging, Grid-stride loops` 


- **CUDA Pro-Tip Sparks Kernel Debugging Revelation**: A member shared a link to a [2013 NVIDIA blog post on grid-stride loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) for writing flexible CUDA kernels, expressing regret for not discovering it sooner.
   - The article highlights that using loops instead of monolithic kernels allows for easy switching to serial processing with a single block and thread, facilitating easier emulation for validation and serializing print order for debugging.
- **Flexible CUDA Kernels via Grid-Stride Loops**: The [CUDA Pro-Tip](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) suggests using grid-stride loops to write flexible CUDA kernels.
   - This approach simplifies debugging by enabling serial processing with a single block and thread, which aids in validating results and serializing print order.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1403092279706521630)** (2 messages): 

> `Naive Matmul Kernels, Memory Access Patterns, Hardware Coalescing` 


- **Naive Matmul Kernel Performance Surprise**: A member implemented two naïve matmul kernels and found that **METHOD 1**, with non-contiguous memory reads within threads, performs about **50%** better than **METHOD 2**, which uses contiguous stride-1 accesses.
   - The code provided shows that Method 1 accesses `B` with `B[kp*n + j]` while Method 2 accesses `B` with `B[j*k + kp]`.
- **Memory Access Contiguity Across Threads Explained**: A member explained that Method 1's memory accesses are not contiguous within a thread, but they are contiguous across threads.
   - They also suggest that *the hardware can coalesce those accesses into a more efficient memory request*.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1403362293047230585)** (4 messages): 

> `Open Source Voxel Renderer, Rust, WebGPU, Data Streaming, Raytracing` 


- **Voxel Renderer Streams Chunks Live!**: A developer released a new devlog on their open source voxel renderer, which runs in **Rust** on **WebGPU**.
   - It now features live chunk streaming while raytracing, with more details available in [this YouTube video](https://www.youtube.com/watch?v=tcc_x2VU2KA).
- **JPEG Image Stream Observation**: A user noted an observation of *'4 jpeg in a row'*, indicating a sequence of JPEG images being posted.
   - This was made in response to some apparent spam.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/)** (1 messages): 

paolovic: thank you!
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1403259991086858321)** (12 messages🔥): 

> `Game Engine Speed, Meeting Reschedule, Player Inventory Transfers, Factorio Native Saves` 


- **Speed Up Factorio Game Engine**: A member asked about the settings to increase the game engine speed, as discussed earlier, and another member suggested using the command `/c game.speed=1000` in the game or via RCON.
   - The member offered assistance from Jack.
- **Meeting faces schedule hiccup**: A member requested to reschedule a meeting for two hours later due to work commitments.
   - Another member agreed but couldn't guarantee attendance, while another member ultimately couldn't make the adjusted time.
- **Inventory Transfers Trigger State Errors**: A member discussed with another member an ongoing issue with player inventory transfers causing slow, compounding state errors between replays and FLE.
   - They suggested addressing this before altering the loading/saving logic.
- **Factorio Native Saves spark design freeze**: One member inquired whether loading/saving referred to Factorio native saves, to which another confirmed the reference to Factorio native saves.
   - However, it was clarified that no development hours were being spent on it due to a design issue.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1403115546924286123)** (7 messages): 

> `CuTe Layouts, Jay Shah's Notes on CuTe Layouts, Layout Algebra Counterexamples` 


- **CuTe Layout Algebra Documentation Flaw**: A member found a flaw in the [CuTe documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html) regarding layout algebra, presenting a counterexample related to the injectivity of layouts.
   - He notes the docs claim that given two layouts `A` and `B = (B_0, B_1, ...)` and `B` is injective, then `A ∘ B = (A ∘ B_0, A ∘ B_1, ...)` but he found a counterexample, and confirmed with someone on the CuTe project that the correct condition appears to be **(1) `A` and `B` satisfy the divisibility conditions, and (2) for `B`, each mode has disjoint image intervals.**
- **Bi-Mode Composition Insights**: A member suggests that `B` must be surjective for `A o B` to be equivalent to bi-mode composition.
   - In response, the original poster notes that even with `B` being surjective onto its image, the counterexample still holds, highlighting the need for a more precise condition for the equivalence.
- **Jay Shah's Notes Explain CuTe Layouts**: A member recommends [Jay Shah’s “A Note on Algebra of CuTe Layouts”](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf) for a better explanation of CuTe layouts than the official documentation.
   - The notes also address the kinds of problems encountered with layout algebras.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1403343726683750523)** (2 messages): 

> `Liveness Analysis, Scalar Compilation Performance, Vector Compilation with Autovectorization and SIMTification` 


- **Dive into Liveness Analysis**: A member mentioned that the **liveness analysis** used to construct the edges of a program's interference graph is a dataflow analysis, suggesting resources like [Møller's SPA](https://cs.au.dk/~amoeller/spa/) and [Cooper/Torczon's EAC](https://www.r-5.org/files/books/computers/compilers/writing/Keith_Cooper_Linda_Torczon-Engineering_a_Compiler-EN.pdf) for further reading.
- **Scalar Compilation Performance Unveiled**: It was stated that **SingSys** will highlight the top two factors affecting scalar compilation performance: **C-style optimizations** and the **balance between the inliner and register allocator**.
- **Vector Compilation Approaches Detailed**: The discussion will then transition into **vector compilation**, focusing on **autovectorization** and **SIMTification** techniques.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1403183750266753168)** (2 messages): 

> `Axolotl, N-D Parallelism, HuggingFace Blog` 


- **Axolotl Pioneers N-D Parallelism**: A member announced the release of **N-D parallelism** with *axolotl*, inviting others to experiment with it, announced in a [HuggingFace blog post](https://huggingface.co/blog/accelerate-nd-parallel).
   - N-D parallelism enables parallelism across multiple dimensions, making it suitable for complex models and large datasets.
- **HuggingFace Showcases N-D Parallelism**: The [HuggingFace blog post](https://huggingface.co/blog/accelerate-nd-parallel) details how to implement **N-D parallelism** using *axolotl* and accelerate, providing code examples and explanations.
   - It highlights the benefits of this approach for scaling training across multiple GPUs and improving performance on large models.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1403090986254598256)** (6 messages): 

> `GPT-5, Agent Maze, Zoom RTMS, ZeroEntropy AI rerankers, Claude citations` 


- **LlamaIndex Gets Day-0 GPT-5 Support**: LlamaIndex announced *day-0 support* for **GPT-5** with `pip install -U llama-index-llms-openai` and invites users to try it out.
- **Agent Maze Challenges GPT-5**: LlamaIndex introduced **Agent Maze**, challenging **GPT-5** to find treasure in a maze using minimal tools ([link](https://t.co/JCZCSVUAed)).
- **AI Agents Handle Live Zoom Voice Data via RTMS**: LlamaIndex announced a hands-on technical workshop on August 14th about building realtime AI agents that process live voice data from **Zoom** meetings using **RTMS** ([link](https://t.co/c2u0CeDnOB)).
- **LlamaParse Gets Reranked by ZeroEntropy for Accuracy**: LlamaIndex announced that retrieval accuracy of **LlamaParse PDF results** can be improved by reranking them with **ZeroEntropy_AI rerankers** ([link](https://t.co/nU4MYzcALH)).
- **Claude Search Results Now Support Citations**: **Claude** now supports search results as content blocks, enabling proper source attribution for results from tool use ([link](https://t.co/Yz0Flt8PeX)).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1403099196210286693)** (39 messages🔥): 

> `llama-index upgrade for gpt-5, workflow tools not working, OpenAI SDK issue and workaround, AgentWorkflow error, llama_deploy compatibility` 


- **Llama-index upgrade prerequisite for gpt-5**: To use **gpt-5**, you'll need to update your `llama-index-llms-openai` package, which might require updating all your `llama-index-*` packages if you aren't already on **v0.13.x**.
- **Workflow tools giving users headaches**: Users reported that **workflow tools** weren't functioning properly, but one member noted that it seemed to work fine for them.
   - The member found that they needed to use **OpenaiResolve** in the new **SDK** for tools to work with OpenAI; they also linked a [GitHub commit](https://github.com/run-llama/llama_index/commit/7e0346213912b98c3b70689398306a38bd890558) that fixed it.
- **OpenAI SDK introduces type error**: Users encountered a `TypeError: Subscripted generics cannot be used with class and instance checks` due to a recent update in the **OpenAI SDK**.
   - The issue was quickly addressed, and a member suggested to pin the OpenAI version in the `requirements.txt` file to prevent such errors in the future; the problem can be resolved with `pip install -U llama-index-llms-openai`.
- **AgentWorkflow suddenly throws runtime error**: One user reported a sudden error in **AgentWorkflow** which included a `workflows.errors.WorkflowRuntimeError: Error in step 'run_agent_step': Subscripted generics cannot be used with class and instance checks`.
   - A member pointed to the relevant message thread to assist with troubleshooting, linking to this [Discord message](https://discord.com/channels/1059199217496772688/1403170643179999406/1403197364960886866).
- **Llama_deploy lagging behind, missing new shiny stuff**: A user reported that upgrading `llama-index-core` to **0.13.0** caused compatibility issues with `llama_deploy 0.9.1`.
   - The user created an issue on the llama-deploy repo and noted the importance of updating dependent packages for new model support.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1403091129825628312)** (41 messages🔥): 

> `Horizon vs GPT5 for agentic coding, Aider GPT-5 on Azure, Aider version updates, Dad meme thumbs up, Python 3.13 support` 


- **Horizon Beta vs GPT-5 for Agentic Coding**: A user who loved **Horizon beta/alpha** for quick agentic coding work is now asking if **GPT-5 Nano** or **Mini** are equivalent and if there's a better option on **OpenRouter**.
- **Aider now works with GPT-5 on Azure**: A user inquired about getting **aider/gpt-5-chat** working on **Azure**, reporting that it worked on **roo**, and Paul Gauthier confirmed that **v0.85.5** should resolve the issue.
   - One user was congratulated for being mentioned in the first 5 minutes of the **GPT 5 unveil video**.
- **Aider config edits requires launch**: A user asked when changes to `.aider.model.settings.yml` would be detected, and it was confirmed the changes only take effect on launch.
- **Thumbs up is the Dad meme**: Paul Gauthier's exclusive use of the thumbs up emoji was discussed as a classic dad meme, with a link provided to a [TikTok video](https://www.tiktok.com/@b_twice99/video/7283752540754398510) and [Vice article](https://www.vice.com/en/article/why-do-dads-communicate-exclusively-via-thumbs-up-emojis/) explaining the phenomenon.
   - The article notes that the thumbs up emoji can come across as *passive-aggressive or that the conversation is not being treated with respect*.
- **Python 3.13 support requested for Aider**: A user requested **Python 3.13** support for Aider, noting that it's the default in the latest Linux distributions, but Paul Gauthier replied that Aider can be installed the recommended way using any (or no) pre-installed Python version.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1403122722728316949)** (4 messages): 

> `Cursor alternative design, OpenRouter's GPT5 errors, aider config parsing failures` 


- **Design Ideas for Cursor Alternative Emerge**: A user inquired about the design considerations for creating an alternative to **Cursor**, seeking insights into feature prioritization and overall architecture.
   - Unfortunately, there was no discussion of any specific design features in the channel.
- **OpenRouter's GPT5 Throws Verification Errors**: A user reported encountering verification errors with **OpenRouter's GPT5** even when using the `-no--stream` option, which they believed would bypass organization verification.
   - The user's question remains unanswered.
- **Aider Config Parsing Fails Due to Environment Variable**: A user experienced an error when including their conventions file in **Aider**, specifically encountering a `mapping values are not allowed in this context` error.
   - The user discovered the issue was due to an inadvertently added environment variable in the **YAML** configuration file.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1403116600378527826)** (41 messages🔥): 

> `Context7 MCP Server, Claude Code Tooling, DSPy Tool Calling, CrewAI Prompts Optimization with DSPy` 


- **Context7 Server Powers Claude's Coding Prowess**: Members discussed using a generic doc-scraping MCP server like [Context7](https://github.com/upstash/context7) to enhance **Claude's** ability to write **DSPy signatures**.
   - The idea is that **Claude**, equipped with powerful doc-searching tools, can effectively utilize **DSPy's** well-structured documentation to generate accurate signatures.
- **Tool Calling Troubleshoot Begins**: Some members sought ways to return a tool's output as the final result in **DSPy**, bypassing the **React Agent's** modifications.
   - They also discussed accessing tool responses independently and explored the use of native tool calling, with one member noting that the [latest releases fixed some issues](https://github.com/stanfordnlp/dspy/pull/824) related to tool usage.
- **Intercepting CrewAI prompts with DSPy Course Launched**: A member announced the launch of an advanced course on [intercepting and optimizing **CrewAI prompts** with **DSPy**](https://www.udemy.com/course/draft/6746331/?referralCode=B59F73AE488715913E7E), demonstrating how to refine prompts for enhanced output quality.
   - Another member expressed interest in similar resources for **Langchain/LangGraph**.
- **Gemini 2.5 Flash completes runs with extra output**: Members reported seeing `[[ ## completed ## ]]` at the end of their output when using **Gemini 2.5 Flash** with **DSPy**.
   - No solution was found.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1403132022947446918)** (14 messages🔥): 

> `Annual Membership Billing Error, Inherit Feature Problems, Login Error, Missing Credits, Manus vs GPT5` 


- ****User Fumed Over Erroneous Annual Membership Charge****: A user reported being charged **$1,999** for an **annual membership** without consent, expecting monthly billing as discussed and after sending emails to support and feedback addresses, the user has received **zero response in 10 days** violating the stated 48-hour policy.
   - Another user commented that this means they'd have to make *$2k with Manus*, but *only $167 a month to break even*.
- ****Inherit Feature Frustrates User with Data Loss****: A user reported issues with the **inherit** feature, experiencing a halt during final deployment tests and they said when using the inherit button, the user created a new project, however everything created was gone, it is now rebuilding and still going after 4 hours, burning through the credits.
   - They expressed concern about losing insights and stated that it was *lesson learnt very fast*.
- ****Login Issues Plague Users****: Multiple users reported login issues with the error message *Email is already registered with a different account*.
- ****Credits Vanish Post-Subscription****: A user reported a significant number of credits missing after their subscription expired, and they expressed concern that their credits were taken away a day after the subscription expired.
   - The user stated they had *thousands* of credits when I last used my most recent usage of -330. *Almost 6000 credits, I believe.*
- ****Queries Surface Regarding Manus Employing GPT-5 Model****: A user inquired whether **Manus** is currently utilizing the **GPT-5** model, but no one replied.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1403092932730552490)** (4 messages): 

> `command-a-vision-07-2025 timing out, Embed v4 vs v3 for vector search, AI Knowledge Domains` 


- **Command Vision Restored After Timeout**: A member reported that **command-a-vision-07-2025** was timing out.
   - Another member confirmed the issue was resolved and apologized for the lack of updates.
- **Embed v4 vs v3 Performance Benchmarks**: A member inquired about the performance of **embed v4** at **256 dimensions** compared to **multilingual light v3** (**384 dims**) for vector search of NL text.
   - They are considering transitioning to **v4** but are concerned about potential performance degradation and are also planning to transition to **v4** at **1024 dims** for clustering, assuming it outperforms the large **v3** model.
- **AI Knowledge Acquisition**: A member expressed a desire to gain knowledge in several domains of **AI**.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1403433066348810321)** (1 messages): 

> `AI Agent capabilities, Generative AI, Workflow automation, Data security, Compliance` 


- **North arrives, empowers with AI Agents**: **North** is expanding its availability of **AI Agent capabilities** for enterprises, built on state-of-the-art generative and search models, operating fully privately.
   - It brings together advanced search, generative AI, workflow automation, core capabilities, security, and compliance, with more details available on [LinkedIn](https://lnkd.in/gFSGxUbD).
- **Advanced search enhances insight surfacing**: North's advanced search and retrieval capabilities provide instant insights, facilitating complex decision-making through **Q&A**.
   - The technology **surfaces insights instantly**.
- **Generative AI drafts documents, tables, and analyzes data**: With North enterprises can draft documents, generate tables, and analyze data using generative AI.
   - The company boasts being able to do this *in an instant*.
- **Workflow automation deploys AI agents across organizations**: **Workflow automation** allows creating and deploying **AI agents** across an organization, streamlining complex processes and eliminating tedious tasks.
   - AI Agents can **eliminate tedious tasks** and **simplify complex processes**.
- **Security with granular access control and private deployments**: North ensures security with granular access controls, system observability, and private deployments, conforming to standards like **GDPR, SOC 2, ISO 27001 and 42001**.
   - Companies can obtain **full data sovereignty**.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1403117354459598922)** (6 messages): 

> `New member introductions, Trading systems with RL and AI agents, Transformers and GNNs` 


- **Vibe Coder Joins Cohere Community**: A self-described *vibe coder* and Cohere user introduced themselves, expressing support for the platform and mentioning ongoing work on a **wallet project**.
   - The user highlighted their satisfaction as a paying customer, encouraging Cohere to *keep up the great work*.
- **Onebrain Developer Arrives**: A member from **Onebrain** announced their arrival, focusing on developing **trading systems** utilizing **Reinforcement Learning (RL)** and **AI agents**.
   - They expressed enthusiasm for **transformers** and **Graph Neural Networks (GNNs)** and a desire for mutual learning within the community.


  

---


### **Cohere ▷ #[🧭-status-feed](https://discord.com/channels/954421988141711382/1346652044181897307/1403148018751901783)** (1 messages): 

> `Command-a-vision-07-2025, degraded performance, Cohere Status Page` 


- **Command-a-vision-07-2025 performance degradation is resolved!**: An incident with degraded performance for **command-a-vision-07-2025** was reported and has been resolved, according to the [Cohere Status Page](https://status.cohere.com).
   - The affected component, **command-a-03-2025**, is now operational.
- **Cohere Status Page reports resolution**: The Cohere Status Page indicated a return to normal operations following the resolution of the **command-a-vision-07-2025** performance issue.
   - The update confirmed that **command-a-03-2025** is now fully operational.


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/)** (1 messages): 

masaru.yamada: Great
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1403127497582837833)** (6 messages): 

> `tensor to mathtraits, unit tests failures, github actions` 


- **Tensor Migrations Sought**: A member asked about the project status on moving stuff out of **tensor** and into **mathtraits**, seeking someone to pick up the task.
   - No one answered.
- **Simple Matmul Test Fails on Master**: A new member reported failing unit tests on the master branch using the command `PYTHONPATH=. DEBUG=2 EMULATE_AMD=1 FORWARD_ONLY=1 PYTHON=1 N=16 HALF=1 ACC_HALF=0 python3 ./extra/gemm/simple_matmul.py`.
   - George Hotz responded that *the command works on my machine*, and questioned why the member cared, given it was running as part of **GitHub Actions**.
- **Exceptions Still Plague Test Despite Functionality**: Despite the command working, a user reported exceptions and test failures, attaching a [screenshot](https://cdn.discordapp.com/attachments/1068976834928193609/1403410826919936122/Screenshot_2025-08-08_at_9.13.26_AM.png?ex=689773af&is=6896222f&hm=e67dab8b94548ed66534a2fb53e7fa6a2bc5ab27dc3d16c01769263cc837896d).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1403097296526377112)** (1 messages): 

> `ShapeTracker Visualization Tool` 


- **ShapeTracker Viz Tool Debuts**: A member introduced a new [ShapeTracker visualization tool](https://shapetracker-viz.vercel.app/) designed to enhance the understanding of movement operations.
   - The tool aims to improve comprehension of movement operations within the system.
- **Tool Accessibility**: The developer shared the tool with the community, hoping others would find it beneficial for understanding movement operations.
   - No further details about the tool's specific functionalities were provided, but its purpose is clear from the context.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1403174310092345365)** (6 messages): 

> `GPT-5 Rumors, GPT-OSS-20B-GUFF Installation Issues, GPT4All Update Status, GPT-ASS Critique` 


- **GPT-5 Speculation lacks Evidence**: Some users speculated about potential features in the next update, while others claimed **GPT-5** was made dumber than **GPT-4**, labeling it *typically American*.
- **GPT-OSS-20B-GUFF Installation Plagued by Crashes**: A user reported experiencing crashes during the installation of **gpt-oss-20b-GUFF**, leading to app failures and requiring a complete uninstall and data scrub to restore functionality.
   - The user sought assistance after encountering these issues, highlighting the challenges in getting the software to work correctly.
- **GPT4All Update Status Raises Concerns**: Members expressed skepticism about new features functioning correctly due to the prolonged lack of updates to **GPT4All**.
   - This concern reflects broader doubts about the platform's ability to support cutting-edge models given its outdated state.
- **GPT-ASS Receives Harsh Critique**: One member dismissed **GPT-ASS** as *garbage*, offering a blunt assessment of its quality and utility.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1403230455037431869)** (2 messages): 

> `MCPOmni Connect, OmniAgent, AI agent builder` 


- ****MCPOmni Connect** v0.1.19 Goes Live!**: **MCPOmni Connect** v0.1.19 is now live, marking the transition *from MCP client to complete AI platform* as shown in this [YouTube video](https://youtu.be/SY3Zwdb5aF8).
   - The release includes **OmniAgent**, an AI agent builder designed to revolutionize the creation of intelligent agents, available on [GitHub](https://github.com/Abiorh001/mcp_omni_connect/releases/tag/v0.1.19).
- ****OmniAgent** Revolutionizes AI Agent Creation**: **OmniAgent**, introduced with **MCPOmni Connect** v0.1.19, is an AI agent builder transforming how intelligent agents are created.
   - This tool is part of the broader update that evolves the **MCP client** into a comprehensive **AI platform**.


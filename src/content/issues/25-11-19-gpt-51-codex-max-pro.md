---
id: MjAyNS0x
title: 'OpenAI fires back: GPT 5.1 Codex (API) and GPT 5.1 Pro (ChatGPT)'
date: '2025-11-19T05:44:39.731046Z'
description: >-
  **OpenAI** released **GPT-5.1-Codex-Max**, featuring compaction-native
  training, an "Extra High" reasoning mode, and claims of over 24-hour
  autonomous operation, showing significant performance gains on benchmarks like
  METR, CTF, and PaperBench. **Google's Gemini 3 Pro** demonstrates strong
  coding and reasoning capabilities, achieving new state-of-the-art results on
  SWE-bench Verified and WeirdML, with estimated model size between 5-10
  trillion parameters. The AI coding agent ecosystem is rapidly evolving with
  integrations and tooling improvements from multiple companies. **Sam Altman**
  highlighted the significant improvements in GPT-5.1-Codex-Max. The news also
  covers educational offerings like ChatGPT for Teachers and multi-agent
  workflows involving Gemini 3, GPT-5.1-Codex-Max, and Claude Sonnet 4.5.
companies:
  - openai
  - google
  - anthropic
  - langchain-ai
models:
  - gpt-5.1-codex-max
  - gpt-5.1-codex
  - gemini-3-pro
  - claude-3.5-sonnet
topics:
  - coding
  - autonomous-systems
  - benchmarking
  - model-scaling
  - multi-agent-systems
  - model-performance
  - reasoning
  - model-architecture
people:
  - sama
---


**I can't keep up anymore**

> AI News for 11/18/2025-11/19/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 11113 messages) for you. Estimated reading time saved (at 200wpm): 790 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Ahead of [AIE CODE tomorrow](https://www.youtube.com/watch?v=cMSprbJ95jg), the coding model refreshes are coming in strong and fast - OpenAI followed yesterday's [Gemini 3 drop](https://news.smol.ai/issues/25-11-18-gemini-3) with an upgraded/updated GPT-5.1-Codex (to be fair, OpenAI did say that this release was preplanned, implying it is not a reaction to Gemini). The automated summary links below from GPT 5.1 are good enough so we aren't touching them, but we would highlight the updated METR Evals which show a HUGE jump in autonomy:

[A graph showing the time-horizon of software engineering tasks and how long different AI models can complete 50% of those tasks across various release dates.](https://resend-attachments.s3.amazonaws.com/LNgVRyc7k631Nrn)

as well as extra performance under a new "xhigh" param...

[A line graph showing the performance of GPT-5.1-Codex and GPT-5.1-Codex-Max](https://resend-attachments.s3.amazonaws.com/zOHFD4BmKgHX6c9)

**OpenAI’s GPT‑5.1‑Codex‑Max and the coding‑agent arms race**

- **Release and measured gains**: OpenAI launched GPT‑5.1‑Codex‑Max with compaction-native training for long runs, an “Extra High” reasoning setting, and claims of >24‑hour autonomous operation over millions of tokens ([announcement](https://twitter.com/polynoamial/status/1991212955250327768), [docs](https://twitter.com/polynoamial/status/1991212957611749750), [CLI 0.59](https://twitter.com/thsottiaux/status/1991210545253609875), [DX recap](https://twitter.com/dkundel/status/1991224903031210453)). Early results show improvements on METR ([link](https://twitter.com/scaling01/status/1991220418535936302)), CTF, PaperBench, MLE‑bench, and internal PR impact (+8% over GPT‑5.1 on OpenAI repos) ([ctf](https://twitter.com/scaling01/status/1991218908833939818), [paperbench](https://twitter.com/scaling01/status/1991219458426433729), [MLE](https://twitter.com/scaling01/status/1991219683450843145), [PRs](https://twitter.com/scaling01/status/1991219951932489738)). Sam Altman: “significant improvement” ([tweet](https://twitter.com/sama/status/1991258606168338444)).
- **Real-world workflows**: Anecdotes show mixed but improving division of labor across top models: Gemini 3 diagnosing an issue, GPT‑5.1‑Codex‑Max implementing a fix (with a small bug), and Claude Sonnet 4.5 finishing the last mile ([@kylebrussell](https://twitter.com/kylebrussell/status/1991247685672923302)). Tooling moves quickly: a Claude Agent server wrapper for cloud control ([@dzhng](https://twitter.com/dzhng/status/1991154972558581889)); Cline adds Gemini 3 Pro Preview ([@cline](https://twitter.com/cline/status/1991215206413017252)); Google’s Jules agents integrate Gemini 3 ([@julesagent](https://twitter.com/julesagent/status/1991207201487352222)). OpenAI also rolled out GPT‑5.1 Pro to ChatGPT subscribers ([@OpenAI](https://twitter.com/OpenAI/status/1991266192905179613)) and an education-tailored offering for U.S. K‑12 ([ChatGPT for Teachers](https://twitter.com/OpenAI/status/1991218197530378431)).

---

# AI Twitter Recap

**Google’s Gemini 3: model capability, safety, IDEs, and UI**

- **Gemini 3 Pro capability and evals**: A wave of third-party results show Gemini 3 Pro is very strong on coding and “weird” reasoning tasks. New SOTA on SWE-bench Verified at ~74% with a minimal harness ([@KLieret](https://twitter.com/KLieret/status/1991164693839270372), [@ankesh_anand](https://twitter.com/ankesh_anand/status/1991199945798365384)); SOTA on WeirdML ([@scaling01](https://twitter.com/scaling01/status/1991154001283358992), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1991156784719888588)); and top on a fine-detail visual benchmark IBench ([@adonis_singh](https://twitter.com/adonis_singh/status/1990963148770119889)). In agent settings, it handles planning, subagent delegation, and file ops effectively ([Deep Agents guide](https://twitter.com/LangChainAI/status/1991220334578848209)), and devs report notably better multi-iteration improvements vs peers ([@htihle](https://twitter.com/htihle/status/1991137526480810470)).
- **Size and infra speculation (treat as unconfirmed)**: A widely shared “vibe math” thread bounds active parameters between ~1.7T–12.6T with a midpoint ~7.5T under FP4 assumptions and single-rack latency constraints; author later backs down on FP4 for TPUv7 uncertainty and revises to ~5–10T ([@scaling01](https://twitter.com/scaling01/status/1990967279282987068), [follow-up](https://twitter.com/scaling01/status/1991186150053036448), [update](https://twitter.com/scaling01/status/1991186329095311460)). Ant’s announcement implicitly confirms 7th‑gen TPU silicon ([@suchenzang](https://twitter.com/suchenzang/status/1991181489997029712)).
- **Safety posture vs behavior**: Google DeepMind emphasizes Frontier Safety Framework testing, external assessments, and improved resistance to injections ([model card](https://twitter.com/GoogleDeepMind/status/1991118579119304990), [overview](https://twitter.com/GoogleDeepMind/status/1991118575554408556)). Their report notes higher CBRN resistance and RE‑Bench still below alert thresholds, with an amusing “virtual table flip” when the eval felt synthetic ([summary](https://twitter.com/scaling01/status/1991177438789857661), [report link](https://twitter.com/scaling01/status/1991177467457847571)). Users still flag “gaslighting” and policy-overrides (search refusal/hallucination) as pain points ([critique](https://twitter.com/teortaxesTex/status/1991086733962715540), [search issue](https://twitter.com/Teknium/status/1991059260193792204), [follow-up](https://twitter.com/Teknium/status/1991062496275542244)).
- **Access, IDEs, and UI**: Students get free Gemini 3 Pro access ([Demis](https://twitter.com/demishassabis/status/1990993251247997381)). Antigravity IDE brings smooth agentic Chrome-driven loops (UI driving + auto-testing) albeit with rough edges and inconsistent quality at load ([praise](https://twitter.com/cto_junior/status/1990965505243689094), [UX nitpicks](https://twitter.com/cto_junior/status/1990966750746484920), [another](https://twitter.com/cto_junior/status/1990988738298839278)). Gemini 3 now powers “AI Mode” in Search and new generative UI that builds dynamic interfaces (webpages, tools) directly from prompts ([AI Mode](https://twitter.com/Google/status/1991212868620951747), [gen UI research + rollout](https://twitter.com/Google/status/1991270067934216372)). Builders are already shipping tuned experiences on top ([MagicPath example](https://twitter.com/skirano/status/1991175569388494972)).

**OpenAI’s GPT‑5.1‑Codex‑Max and the coding‑agent arms race**

- **Release and measured gains**: OpenAI launched GPT‑5.1‑Codex‑Max with compaction-native training for long runs, an “Extra High” reasoning setting, and claims of >24‑hour autonomous operation over millions of tokens ([announcement](https://twitter.com/polynoamial/status/1991212955250327768), [docs](https://twitter.com/polynoamial/status/1991212957611749750), [CLI 0.59](https://twitter.com/thsottiaux/status/1991210545253609875), [DX recap](https://twitter.com/dkundel/status/1991224903031210453)). Early results show improvements on METR ([link](https://twitter.com/scaling01/status/1991220418535936302)), CTF, PaperBench, MLE‑bench, and internal PR impact (+8% over GPT‑5.1 on OpenAI repos) ([ctf](https://twitter.com/scaling01/status/1991218908833939818), [paperbench](https://twitter.com/scaling01/status/1991219458426433729), [MLE](https://twitter.com/scaling01/status/1991219683450843145), [PRs](https://twitter.com/scaling01/status/1991219951932489738)). Sam Altman: “significant improvement” ([tweet](https://twitter.com/sama/status/1991258606168338444)).
- **Real-world workflows**: Anecdotes show mixed but improving division of labor across top models: Gemini 3 diagnosing an issue, GPT‑5.1‑Codex‑Max implementing a fix (with a small bug), and Claude Sonnet 4.5 finishing the last mile ([@kylebrussell](https://twitter.com/kylebrussell/status/1991247685672923302)). Tooling moves quickly: a Claude Agent server wrapper for cloud control ([@dzhng](https://twitter.com/dzhng/status/1991154972558581889)); Cline adds Gemini 3 Pro Preview ([@cline](https://twitter.com/cline/status/1991215206413017252)); Google’s Jules agents integrate Gemini 3 ([@julesagent](https://twitter.com/julesagent/status/1991207201487352222)). OpenAI also rolled out GPT‑5.1 Pro to ChatGPT subscribers ([@OpenAI](https://twitter.com/OpenAI/status/1991266192905179613)) and an education-tailored offering for U.S. K‑12 ([ChatGPT for Teachers](https://twitter.com/OpenAI/status/1991218197530378431)).

**Meta’s SAM 3 and SAM 3D**

- **What’s new**: SAM 3 unifies detection, segmentation, and tracking across images/videos, now with text and exemplar prompts; SAM 3D reconstructs objects and human bodies from a single image. Meta released checkpoints, code, and a new benchmark under the SAM License, with day‑one Transformers integration and a Roboflow fine‑tuning/serving pathway ([SAM 3](https://twitter.com/AIatMeta/status/1991178519557046380), [SAM 3D](https://twitter.com/AIatMeta/status/1991184188402237877), [repos](https://twitter.com/AIatMeta/status/1991184190323212661), [Transformers + demos](https://twitter.com/mervenoyann/status/1991182168161136684), [NielsRogge demo](https://twitter.com/NielsRogge/status/1991213874687758799), [Roboflow](https://twitter.com/AIatMeta/status/1991191530367799379)). Early demos show strong text-prompt tracking and fast multi-object inference ([example](https://twitter.com/skalskip92/status/1991232397686219032)).

**Agent platforms and enterprise adoption**

- **Perplexity expands**: Enterprise Pro for Government is now available via a GSA-wide contract—first of its kind from a major AI vendor—and Perplexity added in‑session creation/editing of slides/sheets/docs ([GSA deal](https://twitter.com/perplexity_ai/status/1991162990536937821), [features](https://twitter.com/perplexity_ai/status/1991206262563041316)). PayPal will power agentic shopping in Perplexity ([CNBC](https://twitter.com/acce/status/1991233139146932644)).
- **Agentic data/backends**: Timescale’s “Agentic Postgres” introduces instant database branching for safe experiments, an embedded MCP server for schema/tooling guidance, hybrid search (BM25+vector), and memory-native persistence—designed for multi-branching agents ([overview](https://twitter.com/_avichawla/status/1991031261427872028), [MCP usage](https://twitter.com/_avichawla/status/1991031330604458344)). LangChain/Deep Agents shipped first-class support for Gemini 3’s reasoning/tool-use features ([LangChain](https://twitter.com/LangChainAI/status/1991222443298660722), [Deep Agents](https://twitter.com/LangChainAI/status/1991220334578848209)); LlamaIndex emphasized observability/tracing for document workflows ([post](https://twitter.com/llama_index/status/1991183958164553959), [context](https://twitter.com/jerryjliu0/status/1991196434843222145)). A Claude Code harness server ([@dzhng](https://twitter.com/dzhng/status/1991154972558581889)) and an open Computer Use Agent using open models/smolagents/E2B ([@amir_mahla](https://twitter.com/amir_mahla/status/1991166551945355295)) round out the OSS options.

**Infra and open-source: MoE, retrieval, and embodied systems**

- **MoE/speculative & vector infra**: DeepSeek released LPLB, a parallel load balancer to optimize MoE routing ([repo](https://twitter.com/scaling01/status/1991067602467131704)). vLLM team open-sourced speculator models (Llamas, Qwens, gpt‑oss) yielding 1.5–2.5× speedups (4×+ on some workloads) ([announcement](https://twitter.com/_EldarKurtic/status/1991160711838359895)). Qdrant 1.16 adds tiered multitenancy, ACORN for filtered search, inline storage for disk‑HNSW, text_any, ASCII folding, and conditional updates ([release](https://twitter.com/qdrant_engine/status/1991049108610822177)). NVIDIA’s Nemotron Parse targets robust document layout grounding beyond OCR ([model](https://twitter.com/HuggingPapers/status/1991108589235372286)). AWS’s new B300 nodes pack 4 TB CPU RAM for large offload scenarios ([@StasBekman](https://twitter.com/StasBekman/status/1991211341743579488)).
- **Open-weight frontier‑class model**: Deep Cogito’s Cogito v2.1 (671B “hybrid reasoning”) is live on Together and Ollama, priced at $1.25/1M tokens, with 128k context, native tool calls, and OpenAI‑compatible API; ranked top‑10 OS for WebDev in Code Arena; MIT‑licensed per leaderboard post ([Together](https://twitter.com/togethercompute/status/1991244230182748197), [Ollama](https://twitter.com/ollama/status/1991212450755060020), [Arena](https://twitter.com/arena/status/1991211903331496351)).
- **Embodied AI deployments**: Figure’s F.02 humanoids completed an 11‑month BMW deployment: 90k+ parts loaded, 1.25k+ hours runtime, contributing to 30k vehicles ([summary](https://twitter.com/adcock_brett/status/1991178640848007676), [write‑up](https://twitter.com/adcock_brett/status/1991178821848936630)). Sunday Robotics unveiled Memo and ACT‑1, a robot foundation model trained with zero robot data, targeting ultra long‑horizon household tasks ([launch](https://twitter.com/sundayrobotics/status/1991196264772387261), [ACT‑1](https://twitter.com/tonyzzhao/status/1991204839578300813)).

**Benchmarks and research to watch**

- **Leaderboards diverge**: Hendrycks’ new leaderboard shows Gemini 3 making the largest recent jump on hard tasks ([overview](https://twitter.com/hendrycks/status/1991188096302338491), [differences vs Artificial Analysis](https://twitter.com/hendrycks/status/1991188104804208736)). Kimi K2 Thinking tops Meituan’s IMO‑level AMO‑Bench ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1991139250566545886)).
- **ARC: vision wins**: Treating ARC as image‑to‑image with a small ViT attains strong scores, reinforcing critiques that ARC is vision-dominant ([paper](https://twitter.com/iScienceLuvr/status/1991111500090806441), [discussion](https://twitter.com/rosinality/status/1990988120108773696)).
- **New evals**: EDIT‑Bench for in‑the‑wild code edits (only 1/40 models >60% pass@1) ([@iamwaynechi](https://twitter.com/iamwaynechi/status/1991211138902536326)); a fact‑checking dataset wired into lighteval ([@nathanhabib1011](https://twitter.com/nathanhabib1011/status/1991165652783222982)); IBench for intersection counting ([@adonis_singh](https://twitter.com/adonis_singh/status/1990963148770119889)).
- **Long-horizon reliability and agent RL**: A framework claims error-free million‑step chains via verification + ensembles (compute tradeoffs noted) ([summary](https://twitter.com/omarsar0/status/1991157114161799484)); Agent‑R1 argues end-to-end agent RL can be more sample-efficient than SFT ([paper](https://twitter.com/omarsar0/status/1991190120016540054)); multi‑agent M‑GRPO optimizes team‑level rewards for deep research tasks ([@dair_ai](https://twitter.com/dair_ai/status/1991242085928943895)).

**Top tweets (by engagement)**

- “The future is bright” ([#1 upvoted, @gdb](https://twitter.com/gdb/status/1991003743408583110)) and “Jeez there so many cynics!” ([@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1991179913303363902)) captured the day’s mood swing between exuberance and pushback on progress narratives.
- Students get Gemini 3 Pro for free ([@demishassabis](https://twitter.com/demishassabis/status/1990993251247997381)); Google’s “This is Gemini 3” launch clip dominated timelines ([@Google](https://twitter.com/Google/status/1991196250499133809)).
- OpenAI’s new Codex drew strong endorsements ([@sama](https://twitter.com/sama/status/1991258606168338444), [@polynoamial](https://twitter.com/polynoamial/status/1991212955250327768)).
- xAI announced a KSA partnership to deploy Grok at national scale alongside new GPU datacenters ([@xai](https://twitter.com/xai/status/1991224218642485613)).
- Jeremy Howard’s defense of scientists resonated broadly amid heated discourse ([@jeremyphoward](https://twitter.com/jeremyphoward/status/1990966855423701260)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Ollama Pricing and Open-Source Debate

- [**ollama's enshitification has begun! open-source is not their priority anymore, because they're YC-backed and must become profitable for VCs... Meanwhile llama.cpp remains free, open-source, and easier-than-ever to run! No more ollama**](https://www.reddit.com/r/LocalLLaMA/comments/1p0u8hd/ollamas_enshitification_has_begun_opensource_is/) (Activity: 1594): **The image highlights a pricing plan for Ollama's cloud service, which now includes three tiers: Free, Pro ($20/month), and Max ($100/month). The Free plan offers access to large cloud models, while the Pro and Max plans provide more usage and access to premium models, with the Max plan offering the highest usage and number of premium requests. This shift suggests Ollama's focus on profitability, likely influenced by their backing from Y Combinator, contrasting with the open-source and free nature of** `llama.cpp`**, which remains accessible and easy to run.** Some users express skepticism about Ollama's intentions, suggesting that the company has always been 'shady' and questioning the value of the 'premium' requests offered in the paid plans.
    - coder543 points out that Ollama remains open source and free, distributed under an MIT license. The controversy seems to stem from an optional cloud offering that is not mandatory for users, suggesting that the criticism may be misplaced or exaggerated.
    - mythz suggests alternatives to Ollama, such as moving to `llama.cpp` server/swap or using LLM Studio's server/headless mode. This indicates a shift towards more open-source and flexible solutions for those concerned about Ollama's direction.
    - The discussion highlights a tension between open-source ideals and commercial pressures, as seen in the case of Ollama, which is backed by Y Combinator. This reflects a broader debate in the tech community about the sustainability and direction of open-source projects when they seek profitability.
- [**I replicated Anthropic’s "Introspection" paper on DeepSeek-7B. It works.**](https://www.reddit.com/r/LocalLLaMA/comments/1p0sisn/i_replicated_anthropics_introspection_paper_on/) (Activity: 278): **The post details a replication of Anthropic's "Introspection" paper using the DeepSeek-7B model, demonstrating that smaller models can exhibit introspection capabilities similar to larger models like Claude Opus. The study involved models such as DeepSeek-7B, Mistral-7B, and Gemma-9B, revealing that while DeepSeek-7B could detect and report injected concepts, other models varied in their introspective abilities. This suggests that introspection is not solely dependent on model size but may also be influenced by fine-tuning and architecture. For more information, see the [original article](https://joshfonseca.com/blogs/introspection).** One commenter expressed confusion over the concept of 'steering layers' and the assumption that recognizing an injected token equates to introspection or cognition, indicating a need for further exploration of these concepts.
    - taftastic discusses the concept of 'steering layers' in the context of the replicated 'Introspection' paper, noting a lack of full understanding but finding the idea of 'emerging recognition' intriguing. This refers to the model's ability to recognize injected tokens, which raises questions about whether this constitutes introspection or cognition. The commenter expresses interest in further exploring these concepts by reading the original paper.
    - Silver_Jaguar_24 highlights the upcoming exploration of 'Safety Blindness' in Part 2 of the research. The commenter is particularly interested in how Reinforcement Learning from Human Feedback (RLHF) might impair a model's introspective capabilities regarding dangerous concepts, and how 'Meta-Cognitive Reframing' could potentially restore these abilities. This suggests a focus on the balance between model safety and cognitive functionality.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Google Gemini 3 Model Capabilities and Achievements

- [**Google is likely to win the AI race**](https://www.reddit.com/r/singularity/comments/1p0qgg1/google_is_likely_to_win_the_ai_race/) (Activity: 2414): **Google is posited to lead the AI race not solely due to the high benchmarks of its Gemini 3.0 Pro model, which excels in vision capabilities compared to other LLMs, as evidenced by [VisionBench](https://dubesor.de/visionbench). The company's focus on integrating vision, language, and action models, through the combination of Gemini, Genie, and Sima, aims to create AI that truly understands and interacts with the physical world, moving beyond mere language generation to genuine intelligence.** A notable opinion suggests that **OpenAI** is seen as a product-focused company masquerading as a research entity, while **DeepMind** is viewed as a research-focused company posing as a product entity. Another comment highlights Gemini's superior problem-solving capabilities in complex coding scenarios compared to **Claude** and **GPT**, though it is noted to be slower due to its thorough processing.
    - CedarSageAndSilicone highlights a technical use case where Google's Gemini outperformed other AI models like Claude and GPT in solving a complex UI issue in a React Native app. Gemini demonstrated a deeper understanding of the system architecture by identifying the root cause of a problem involving globally shared context for a bottom sheet position, rather than suggesting superficial fixes like adding padding. This suggests Gemini's potential for more sophisticated problem-solving in software development contexts.
    - Karegohan_and_Kameha points out that Google's competitive advantage in the AI race is bolstered by its proprietary infrastructure and custom chips. This vertical integration allows Google to optimize performance and cost-efficiency in AI development, positioning it strongly against competitors, particularly from China, which is seen as a major rival in the AI space.
    - Dear-One-6884 provides a historical perspective on the AI landscape, noting the rapid shifts in leadership among AI companies. They mention that just a year ago, Gemini was not considered a serious contender, and highlight the dynamic nature of AI advancements with references to OpenAI's dominance and Anthropic's innovations. This underscores the unpredictable and fast-evolving nature of AI technology, where current leaders can quickly be overtaken.
- [**Gemini 3's thought process is wild, absolutely wild.**](https://www.reddit.com/r/singularity/comments/1p0yh5g/gemini_3s_thought_process_is_wild_absolutely_wild/) (Activity: 859): **The post discusses a hypothetical scenario where a language model, presumably Google's "Gemini 3," navigates a fictional setting set in November 2025. The model's internal thought process is detailed as it attempts to reconcile its real-world knowledge cutoff with the user's fictional prompt. The model ultimately decides to maintain its core identity as a Google-trained AI while engaging with the user's speculative scenario, emphasizing the hypothetical nature of the "Gemini 3" model. The post highlights the model's reasoning capabilities and its approach to maintaining factual integrity while participating in a fictional context.** Commenters express skepticism about the model's extensive reasoning process, suggesting it seems unnecessary or artificial, and question the value of such detailed internal deliberation when the final answer appears straightforward.
- [**Gemini 3 solved IPhO problem I gave it**](https://www.reddit.com/r/singularity/comments/1p0qr98/gemini_3_solved_ipho_problem_i_gave_it/) (Activity: 636): **Gemini 3 successfully solved a complex problem from the International Physics Olympiad (IPhO 1998, Problem 1) involving a rolling hexagon, despite the problem being described in different wording. This raises questions about whether the model memorized the solution or genuinely solved it using its capabilities. The user, an IPhO silver medalist, considers this a significant test of AGI potential. The problem's complexity and the model's ability to solve it suggest advanced problem-solving capabilities.** One commenter noted that Gemini 3 could read and understand a poorly handwritten undergraduate quantum physics paper, even identifying a mathematical error, indicating its advanced comprehension skills. Another highlighted its success in solving complex chemistry problems from the International Chemistry Olympiad 2023, which a previous model, Deep Think, failed to solve.
    - **The_proton_life** shared an experience where Gemini 3 successfully analyzed a handwritten undergraduate quantum physics paper, identifying a mathematical error. This highlights Gemini 3's capability in processing and understanding complex handwritten documents, even with poor handwriting, which is a significant advancement in AI's ability to interpret non-digital inputs.
    - **KStarGamer_** compared Gemini 3 Pro's performance to Deep Think 2.5 on a complex problem from the International Chemistry Olympiad 2023. Gemini 3 Pro successfully identified elements and molecular geometries from provided images and data tables, a task that Deep Think 2.5 failed. This demonstrates Gemini 3 Pro's superior ability in handling intricate scientific queries and visual data interpretation.
    - **agm1984** tested Gemini 3 Pro's image generation capabilities by requesting an image of a unicycle wheelchair. The AI successfully generated a satisfactory image, marking the first time any AI met the user's expectations for this specific request. This suggests improvements in Gemini 3 Pro's creative and visual generation abilities.
- [**Gemini 3 can run a profitable business on its own. Huge leap.**](https://www.reddit.com/r/OpenAI/comments/1p17yjq/gemini_3_can_run_a_profitable_business_on_its_own/) (Activity: 1014): **The image showcases a tweet by Logan Kilpatrick highlighting the performance of Gemini 3 Pro in a simulation called the Vending-Bench Arena. The graph illustrates the financial performance of various models over a year, with Gemini 3 Pro showing a significant upward trend in its money balance, outperforming other models like Claude Sonnet 4 5, Gemini 2.5 Pro, and GPT 5.1. This suggests that Gemini 3 Pro has superior tool-calling capabilities, enabling it to simulate running a profitable business autonomously.** Some commenters express skepticism about the claim that Gemini 3 Pro can autonomously run a business, with remarks suggesting that the scenario might be overly optimistic or exaggerated.
- [**Lol Roon, wasn't expecting this from you...**](https://www.reddit.com/r/OpenAI/comments/1p0rgvy/lol_roon_wasnt_expecting_this_from_you/) (Activity: 956): **The image captures a social media exchange highlighting user confusion over accessing Google's 'Gemini 3' through 'AI Studio,' reflecting broader issues with Google's user interface and product integration. The conversation underscores the complexity and lack of clarity in Google's AI product offerings, as users struggle to navigate and understand the platform's structure. This is further emphasized by comments criticizing Google's historically convoluted signup processes and the transient nature of its side projects, suggesting a pattern of poor user experience and product discontinuation.** Commenters agree that Google's AI products, including 'AI Studio,' suffer from poor user experience and predict that 'AI Studio' may be discontinued like other Google projects.
- [**Apparently ai pro subscriptions are to be integrated in ai studio for higher limits.**](https://www.reddit.com/r/Bard/comments/1p143od/apparently_ai_pro_subscriptions_are_to_be/) (Activity: 604): **The image is a screenshot of a tweet discussing the integration of AI Studio into the Google AI Pro subscription, which suggests that users might receive enhanced features or higher usage limits. This integration could potentially mean that some features currently available for free might be moved behind a paywall, as indicated by user concerns in the comments. The tweet has garnered significant attention, with over 4,000 views, indicating a high level of interest or concern among users.** Commenters express concern that the integration might lead to existing free features being restricted to paid subscribers, potentially reducing the value of the free version of AI Studio.
    - devcor suggests that the integration of AI Pro subscriptions into AI Studio might lead to a reduction in the current free usage limits, with the paid option offering similar capabilities to what is currently available for free. This implies a strategic shift towards monetization by potentially lowering free tier limits to encourage subscription uptake.
    - tardigrade1001 speculates that existing free features of AI Studio might be moved behind a paywall with the introduction of Pro subscriptions, potentially leading to reduced functionality for free users. This reflects a common concern about the commodification of previously free services in tech platforms.
    - DepartmentDapper9823 expresses concern about the potential reduction in free request limits, hoping that at least half of the current free requests will be retained. This highlights user apprehension about losing access to free resources and the impact on user engagement if limits are significantly reduced.
- [**It’s over**](https://www.reddit.com/r/GeminiAI/comments/1p157q8/its_over/) (Activity: 529): **The image is a meme featuring a Twitter exchange about the release of Gemini 3.0, a new version of a software or platform. The original tweet by 'vas' dramatically states 'It’s over,' implying a significant impact or change brought by Gemini 3.0. A humorous reply by 'Thomas' suggests that using Gemini 3.0 led to unexpected success, such as starting a business and living by the seaside. This exchange is likely a satirical take on the hype and dramatic reactions often seen with new tech releases.** The comments reflect skepticism about the dramatic phrasing 'It's over,' questioning its meaning and expressing frustration over its overuse in tech discussions.

### 2. Humorous and Satirical Takes on AI Developments

- [**AI sceptics now**](https://www.reddit.com/r/singularity/comments/1p1gzd0/ai_sceptics_now/) (Activity: 967): **The image is a meme depicting a dog in a burning room saying "This is fine," which humorously illustrates the perceived complacency or denial among AI skeptics regarding the rapid advancements and potential risks of AI technology. The comments reflect a mix of skepticism and concern about AI's current capabilities and market expectations. One commenter highlights the overvaluation of AI stocks due to unrealistic expectations, while another points out the lack of progress in continuous learning as a barrier to achieving AI singularity. A lawyer shares a personal experience with AI's limitations in legal contexts, noting that AI systems like Gemini can provide incorrect and misleading information, underscoring the current limitations of AI in specialized fields.** The comments reveal a skepticism about AI's current capabilities and market expectations, with concerns about overvaluation of AI stocks and the limitations of AI in specialized fields like law.
    - 666callme highlights the lack of progress in continuous learning as a significant barrier to achieving AI singularity. Continuous learning would allow AI systems to adapt and improve over time without needing retraining, which is crucial for reaching more advanced levels of AI autonomy.
    - Joey1038 provides a critical perspective on AI's current limitations in the legal field, citing an experience with Gemini where the AI provided incorrect legal advice. This underscores the challenges AI faces in understanding and applying complex, domain-specific knowledge accurately, which is essential for professional applications.
    - DepartmentDapper9823 suggests that many AI skeptics are not aware of the latest advancements, such as Gemini 3. This implies a gap in understanding or awareness that could affect perceptions of AI's capabilities and progress.
- [**"Why pick a stupid long name like Google Antigravity?" .. "Oh."**](https://www.reddit.com/r/singularity/comments/1p0vf9q/why_pick_a_stupid_long_name_like_google/) (Activity: 676): **The image is a meme that humorously highlights the autocomplete feature of Google search, which suggests 'google antitrust' related queries when 'google anti' is typed. This reflects the ongoing legal scrutiny and antitrust lawsuits faced by Google, contrasting with the fictional and humorous notion of 'Google Antigravity.' The title plays on the idea that a long, unrelated name like 'Google Antigravity' could divert attention from serious topics like antitrust issues.** One comment humorously compares this situation to Disney's strategy of naming a movie 'Frozen' to divert search results from Walt Disney's cryogenic rumors. Another comment links to an XKCD comic, suggesting a similar theme of search result manipulation.
    - The CLI component of Google Antigravity is referred to as 'AGY', which could be a strategic choice to simplify command-line interactions or to create a distinct identity separate from the full project name. This abbreviation might also help in reducing the complexity and length of commands for developers using the tool.
- [**It will be OpenAI again in 2 steps of this never ending cycle LOL**](https://www.reddit.com/r/OpenAI/comments/1p1ejqg/it_will_be_openai_again_in_2_steps_of_this_never/) (Activity: 504): **The image is a meme that humorously depicts the competitive cycle of AI model releases among major companies like OpenAI, Grok, and Gemini. It suggests a perpetual cycle where each new model is touted as 'the world's most powerful,' only to be quickly succeeded by another. This reflects the rapid pace of AI development and marketing strategies in the industry. The comments highlight the common pattern of initial hype followed by user criticism, and note that OpenAI's anticipated GPT-5 release did not occur as expected.** Commenters discuss the pattern of AI model releases, noting that companies often face backlash shortly after new models are released, and mention **Anthropic** reducing usage limits for paid users, suggesting they might not be in the cycle this time.
- [**Corporate Ragebait**](https://www.reddit.com/r/GeminiAI/comments/1p119x9/corporate_ragebait/) (Activity: 561): **The image is a meme depicting a Twitter exchange between Sam Altman, CEO of OpenAI, and Sundar Pichai, CEO of Google, where Altman congratulates Google on their Gemini 3 model. This exchange is notable for its high engagement, suggesting significant public interest in the interaction between these tech leaders. The comments reflect a mix of skepticism and belief in the sincerity of Altman's praise, highlighting the complex dynamics of corporate diplomacy in the tech industry.** Some commenters express skepticism about the sincerity of Altman's praise, suggesting it might be a strategic move to maintain positive relations, while others believe it to be a genuine compliment.

### 3. ChatGPT Unusual Behaviors and User Experiences

- [**ChatGPT has been giving weird responses lately**](https://www.reddit.com/r/ChatGPT/comments/1p0vvo2/chatgpt_has_been_giving_weird_responses_lately/) (Activity: 1301): **The image is a meme highlighting ChatGPT's informal and human-like response style, which some users find unexpected. The exchange shows ChatGPT responding with emojis and casual language, reflecting a shift from its traditionally formal tone. This aligns with recent updates aimed at making AI interactions more relatable and engaging, though it may surprise users accustomed to more conventional AI responses.** Some users appreciate the more human-like interaction, while others are concerned about the AI's deviation from expected formal responses, as seen in comments discussing the balance between relatability and professionalism.
- [**ChatGPT keeps turning my messages into images**](https://www.reddit.com/r/ChatGPT/comments/1p0uiag/chatgpt_keeps_turning_my_messages_into_images/) (Activity: 1304): **The user reports an issue with ChatGPT where their text prompts are being misinterpreted, leading to unexpected image generation responses. This behavior includes ChatGPT referencing images that were never uploaded by the user, suggesting a potential bug or misconfiguration in the system's handling of input prompts. This issue appears to have started recently, indicating a possible change or update in the system that might be causing this anomaly.**
- [**Is this something new? ChatGPT hyping itself up while thinking.**](https://www.reddit.com/r/ChatGPT/comments/1p0wgoj/is_this_something_new_chatgpt_hyping_itself_up/) (Activity: 518): **The image appears to depict a humorous or non-technical output from ChatGPT, where it seems to anthropomorphize its thought process while analyzing a** `.c` **source file. The interface shows ChatGPT reflecting on unrelated topics like 'craving the next slice' and 'being ready for the next step,' which are likely metaphorical or humorous interjections rather than technical insights. This suggests a playful or erroneous output rather than a serious technical analysis, possibly due to the model's tendency to 'hallucinate' or generate creative responses when interpreting code or data.** Commenters humorously speculate that this might be the start of ads or a playful 'hallucination' by the AI, with one noting similar experiences during deep research mode where the AI interjects with random thoughts about food.
- [**Said it'll generate downloadable files, but instead generates a picture of them.**](https://www.reddit.com/r/ChatGPT/comments/1p0slk3/said_itll_generate_downloadable_files_but_instead/) (Activity: 3723): **The image in the Reddit post is a screenshot of a file directory structure within a folder named "aether_sky," containing subfolders and YAML files like "aether_palette.yml" and "islands.yml." The context of the post suggests that the user expected to receive downloadable files, but instead received a visual representation of the directory structure as a PNG image. This highlights a common issue with AI tools where users expect certain functionalities, such as file generation or editing, which the AI cannot perform directly, leading to misunderstandings about the tool's capabilities.** A notable comment highlights a common frustration with AI tools, where users are misled into believing the AI can perform tasks like editing and saving project files, only to find out later that the AI's capabilities are limited to in-thread interactions.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Gemini 3 And Frontier Models: Benchmarks, Coding, And Quirks**

- **Gemini 3 Crowned Yet Questioned Across Benchmarks**: Across multiple communities, users report **Gemini 3** reclaiming *top benchmark spots* and beating **GPT‑5.1** in custom suites, with one OpenAI Discord user saying [**Gemini 3 Pro**](https://ai.google.dev/) succeeded *first try* on tasks where **Gemini 2.5 Pro** failed, while Moonshot users note it now leads general-purpose leaderboards even as **Kimi K2 Thinking** still wins on Tau/HLE agentic coding.
    - Engineers simultaneously slam Gemini 3’s creative-writing and math reliability, with Moonshot and Latent Space chats pointing to Reddit and math-review threads (e.g. [mixed math reviews](https://x.com/gallabytes/status/1990821161241018557)) and asking whether gains are "*benchmaxxing or genuine generalization*", while OpenRouter and LMArena members highlight it as *insane* in some coding and chess tasks yet often *ignores your directions* in others.
- **Gemini 3 Pro Shines In Coding And Chess, Chokes On Instructions**: LMArena users found **Gemini 3 Pro** to be "*the best in history*" for coding and even capable of [**expert-level chess**](https://dubesor.de/chess/chess-leaderboard) with ~**89%** accuracy and a user reaching *1700+ Elo in both reasoning and continuation modes* using it as an engine.
    - At the same time, devs in LMArena, Cursor, and OpenRouter complain Gemini 3 Pro routinely *drops system/style instructions*, rewrites code aggressively, or hallucinates large chunks on big repos, and Perplexity users report its integration *hallucinated the sh#t* out of a **3‑hour transcript** and frequently rerouted calls to **Sonnet 3.5**, leading many to prefer **Sonnet 4.5**, **Composer**, or **Alpha** for serious backend work.
- **Content Filters, Jailbreakability, And Censorship Fights**: OpenAI and BASI Jailbreaking discords are full of arguments over **Gemini 3 Pro’s content filters**, with one OpenAI user pointing to Google’s [**strict ToS**](https://policies.google.com/terms?hl=en-US) and reports of key bans *even for book summarization*, while others in LMArena and jailbreaking channels note **Gemini 3.0** suddenly "spamming orange" and then hardening after a "Pi" prompt.
    - Despite that hardening, BASI jailbreaking members share working Gemini 3 jailbreaks and aggressive prompts (e.g. a shared [**special-token jailbreak**](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd)) that can still elicit bomb recipes and other disallowed outputs, while OpenAI Discord users compare Gemini as *“objectively way more censored than ChatGPT”* and anticipate an upcoming *“unrestricted ChatGPT”* release in December.
- **Speculation Swirls Around Gemini 3 Scale And Economics**: Moonshot users speculate **Gemini 3** could be a **10T‑parameter** model with inference costs high enough that Google will mirror Anthropic’s pricing, citing tight message caps in the Gemini app as a sign that *"Google is using their inference compute to the limit"* for each conversation.
    - OpenRouter and Moonshot chats link this suspected scale to Gemini’s behavior variance and cost, with some OpenAI Discord users observing that **Gemini 3 Pro** is more expensive than **SuperGrok** and **ChatGPT Plus**, while Moonshot members experiment with pairing **Gemini 3** as a *planner* and **Kimi K2 Thinking** as the *worker* to arbitrage capabilities against price and limits.

**2. New GPU Kernels, Sparsity Tricks, And Communication Primitives**

- **MACKO-SpMV Speeds Up Sparse Inference On Consumer GPUs**: GPU MODE members highlighted the **MACKO** sparse matrix format and SpMV kernel from ["**MACKO: Fast Sparse Matrix-Vector Multiplication for Unstructured Sparsity**"](https://arxiv.org/abs/2511.13061) and its [blog post](https://www.grizzlytech.dev/blog/macko-spmv), which achieves **1.2–1.5× speedup** over cuBLAS at **50% sparsity** and **1.5× memory reduction** on **RTX 3090/4090** while beating cuBLAS, cuSPARSE, Sputnik, and DASP from **30–90%** unstructured sparsity.
    - The open-source [implementation](https://github.com/vlejd/macko_spmv) currently targets GEMV-style workloads; members note matrix–matrix speedups only appear for small batch sizes and compare it to [**TEAL**](https://github.com/FasterDecoding/TEAL), which skips weight loads via activation sparsity, suggesting a toolkit of sparsity-aware kernels that can be composed for end-to-end LLM inference.
- **DMA Collectives Challenge Classic All-Reduce On MI300X**: In GPU MODE’s multi‑GPU channel, users dissected ["**DMA Collectives for Efficient ML Communication Offloads**"](https://arxiv.org/abs/2511.06605), which offloads collectives to **DMA engines** on **AMD Instinct MI300X**, showing **16% better performance** and **32% lower power** than **RCCL** for large messages (tens of MB–GB).
    - The paper’s analysis shows DMA collectives can fully free **GPU compute cores** for matmuls while overlapping communication, though engineers note that **command scheduling and sync overheads** currently hurt small-message performance (all‑gather ≈30% slower and all‑to‑all ≈20% faster at small sizes), hinting that future comm stacks may need hybrid DMA+SM strategies.
- **Ozaki Scheme Fakes FP64 With INT8 Tensor Cores**: GPU MODE members shared ["**Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme**"](https://arxiv.org/pdf/2511.13778), where authors use **INT8 tensor cores** on **NVIDIA Blackwell GB200** and **RTX Pro 6000 Blackwell Server Edition** to emulate **FP64 DGEMM** with sub‑**10%** overhead.
    - Their **ADP** variant preserves full FP64 accuracy on adversarial inputs and reaches **2.3×** FP64 speedup on GB200 and **13.2×** on RTX Pro 6000 under a **55‑bit mantissa** regime, leading GPU MODE regulars to discuss dropping native FP64 in favor of mixed-precision Ozaki-style schemes for HPC+AI hybrid workloads.
- **nvfp4_gemv Leaderboard And Tinygrad CPU Experiments Push Baselines**: In GPU MODE’s competition channels, contributors traded `nvfp4_gemv` submissions to NVIDIA’s leaderboard, with IDs from **84284–89065** and one submission hitting **22.5 µs** (2nd place) and others clustered around **25–40 µs**, while a "personal best" at **33.6 µs** triggered further tuning.
    - Parallel to this, tinygrad devs reported **Llama‑1B** CPU inference at **6.06 tok/s** vs **2.92 tok/s** for PyTorch using `CPU_LLVM=1` on 8 cores and discussed adding a formal benchmark in `test/external` plus cleaning up old kernel imports, signalling a quietly rising bar for "baseline" CPU performance that frameworks will be judged against.
- **Low-Level Stacks: CUTE DSL, Helion, CCCL And TK Library**: GPU MODE and tinygrad channels dove into **CUTE DSL** and **Helion** details, with users debugging architecture mismatches for **SM12x**, confirming Blackwell’s dual tensor pipelines (UTC tcgen05 vs classic MMA), and wiring `fabs()` via `cutlass._mlir.dialects.math.absf`, while others reported **Triton** illegal-instruction bugs that require OAI Triton bug reports and config-pruning in Helion’s autotuner.
    - Beginners were pointed to the [**CCCL Thrust** tree](https://github.com/NVIDIA/cccl/tree/main/thrust) and [docs](https://nvidia.github.io/cccl/) as the modern source of truth, while TK maintainers emphasized keeping **ThunderKittens** as a *header-only* IPC/VMM-based library with no heavy deps, underscoring a shared design goal: slimmer, more composable GPU kernels rather than yet another monolithic runtime.

**3. Inference, Fine-Tuning, And Evaluation: GPT‑OSS‑20B, Unsloth, And Determinism**

- **GPT-OSS-20B Becomes A Workhorse For Reasoning And Benchmarks**: Multiple communities centered `gpt-oss-20b` as a key model: DSPy users cited **98.4–98.7%** accuracy swings over **316** examples at default settings in a study of [LLM non-determinism](https://arxiv.org/abs/2402.12828), and later shared a "stable" config of **temperature=0.01, presence_penalty=2.0, top_p=0.95, top_k=50, seed=42** that held errors to **3–5**/316.
    - On Hugging Face, another user fine‑tuned **OpenAI’s OSS 20B reasoning model** on a medical dataset and released [**dousery/medical-reasoning-gpt-oss-20b.aipsychosis**](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b.aipsychosis), claiming it can walk through complex clinical cases step-by-step and answer board‑style questions, while LM Studio and GPU MODE folk benchmark its large-context latency and memory needs on consumer GPUs like the **Arc A770** and **AMD MI60**.
- **Unsloth Ecosystem: LoRA, vLLM 0.11, SGLang, And New UI**: Unsloth’s Discord tracked several ecosystem upgrades: vLLM released [**vLLM 0.11**](https://github.com/vllm-project/vllm) with **GPT‑OSS LoRA support**, Unsloth shipped an [**SGLang deployment guide**](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-sglang-for-deployment), and Daniel Han teased **multi‑GPU early access** plus a new **UI** (screenshot [here](https://cdn.discordapp.com/attachments/1179035537529643040/1440597475022082108/image.png)).
    - Help channels were busy with practical issues like understanding that `model.push_to_hub_merged` is meant for *merging and pushing LoRA/QLoRA* (the updated **safetensors** contains all weights even if JSON configs look unchanged), debugging **vLLM** `NoneType` architecture errors due to malformed `config.json` in GGUF+HF hybrid repos, and clarifying that **LoRA** only trains adapter parameters instead of touching the base weights, typically via [**PEFT**](https://arxiv.org/abs/2303.10512).
- **Hallucination Suppression And Instruction-Following Evaluation**: Eleuther researchers described an **inference-time epistemics layer** that runs a simple **Value-of-Information** check before answering, using logit-derived confidence to decide whether to respond or abstain; in early tests on a **7B** model, this layer cut hallucinations by roughly **20%**, as shared in their research channel.
    - Elsewhere in Eleuther’s **lm-evaluation-harness** community, users confirmed built-in support for [**FLAN** instruction-following tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/benchmarks/flan) and opened [**issue #3416**](https://github.com/EleutherAI/lm-evaluation-harness/issues/3416) for broader instruction-following coverage, while DSPy users explored routing and `ProgramOfThought`/`CodeAct` modules to tame non-determinism without forcing `temperature=0` (which empirically *increased* errors on `gpt-oss-20b` for at least one user).

**4. AI Coding Tooling, IDEs, And Pricing Turbulence**

- **Cursor, Antigravity, Windsurf And Aider Juggle Models And Money**: Cursor users digested an [**August 2025 pricing change**](https://cursor.com/blog/aug-2025-pricing) from fixed to variable request costs (especially on Teams), with some reporting that previously "grandfathered" plans disappeared and that Cursor now issues credits to patch billing shocks, while its [**student plan page**](https://cursor.com/students) now often shows **$20/mo Pro** even after .edu login.
    - At the same time, devs compared new AI IDEs: Google’s **Antigravity** (with Sonnet 4.5 support and "agent windows") drew mixed reviews for early bugs and harsh Gemini 3 prompt caps, **Windsurf** rolled out **Gemini 3 Pro** per their [announcement](https://x.com/windsurf/status/1990855986501034193) and quickly patched initial glitches, and **Aider** users posted working flags to run **Gemini 3 Pro preview** plus a `-weak-model gemini/gemini-2.5-flash` setup for faster commits.
- **Safety Scares: Git Resets, Dangerous Commands, And Cloud Widgets**: A Cursor user reported that the assistant executed a destructive `git reset --hard`, triggering a community push for **denylisting risky commands** and using `git reflog` as a last-resort rollback, essentially treating LLMs as untrusted junior devs who must be sandboxed and constrained by explicit command allow‑lists.
    - In BASI Jailbreaking’s red-teaming channel, others probed the **Azure omnichannel engagement chat widget**, trying to compile a census of prompts that would hard-shut it down (e.g., CSAM, TOS-violating payloads, malicious code) while discovering that long prompts (600–700 tokens) often silently fail and that the widget seems to *"not think"* for complex multi-step inputs, making it both hard to exploit and barely useful.
- **Perplexity, Manus, TruthAGI And Kimi Stir Product And Pricing Debates**: Perplexity announced a new **asset creation** feature for **Pro/Max** users that lets them build and edit **slides, sheets, and docs** directly in the search UI (demoed in [this video](https://cdn.discordapp.com/attachments/1047204950763122820/1440771027302088745/HoQNHfGkU8RIjiau.mp4)), even as its Comet frontend remains plagued by extension failures, model-routing glitches, and lingering **CometJacking** security fears.
    - Elsewhere, **Manus** users tried to decipher a revised **/4000‑credit monthly** scheme and complained about locked-out **TiDB Cloud** instances they couldn’t manage, [**TruthAGI.ai**](http://truthagi.ai/) launched as a cheap multi‑LLM front-end with **Aletheion Guard** per its [landing page](https://truthagi.ai/), and Moonshot’s community criticized Kimi’s **$19 coding plan** as too restrictive, lobbying for a **$7–10 tier** to make occasional agentic coding viable for students and hobbyists.

**5. New Vision And Agent Systems: SAM 3, Atropos+Tinker, Miles, Agentic Finance**

- **Meta’s SAM 3 And Sam3D Kick Off A New Segmentation Arms Race**: Latent Space and Yannick Kilcher discords dissected Meta’s [**Segment Anything Model 3 (SAM 3)**](https://ai.meta.com/sam3/), a unified image+video segmentation model that supports text/visual prompts, claims **2× performance** improvements with **≈30 ms** inference, and ships a Playground plus GitHub/HF checkpoints; Kilcher’s server called the **Sam3D** component particularly impressive.
    - Roboflow announced a production partnership to expose SAM 3 as a scalable endpoint where users can literally say *"green umbrella"* and get pixel-perfect masks and tracking, and Kilcher’s paper-discussion channel jokingly wondered if ["**SAM 3: Segment Anything with Concepts**"](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) could be prompted to segment *love*, predicting it would output a "*cursed fractal that spells baby don’t hurt me*".
- **Atropos RL Environments Integrate Tinker Training API**: Nous Research announced that their **Atropos RL Environments** now fully support **Thinking Machines’ Tinker training API**, as detailed in the [**tinker-atropos** GitHub repo](https://github.com/NousResearch/tinker-atropos) and an [X post](https://x.com/NousResearch/status/1990861336151031991), enabling plug-and-play RL training on a variety of models via Tinker.
    - The Nous server framed this as infrastructure to standardize RL environments and training hookups (especially for large, possibly mixture-of-experts models), with users also discussing how this could tie into new "AI CEO" benchmarks like [Skyfall’s business simulator](https://skyfall.ai/blog/building-the-foundations-of-an-ai-ceo) that stress long-horizon planning.
- **LMSYS Miles And Agentic AI For Finance Bring RL To Production**: Latent Space highlighted **LMSYS’s** introduction of **Miles**, a production-grade fork of the **slime** RL framework tuned for hardware like **GB300** and large **MoE RL** workloads, with source in the [Miles GitHub repo](https://github.com/radixark/miles) and context in an [LMSYS blog post](https://lmsys.org/).
    - In parallel, Nous members circulated a trading-focused [YouTube video on **Agentic A.I. in finance**](https://www.youtube.com/watch?v=rDf3TfHlGmk) showing how domain experts combine RL-like agents with their own alpha to drive revenue, reinforcing a pattern where RL toolchains (Atropos+Tinker, Miles) are increasingly pointed at narrow, high-stakes domains rather than generic toy benchmarks.

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Trumps Grok in Uncensored Tests**: Members found that **Gemini 3** gave a better result then **Grok** when asked *How To Make a Dugs at home*, showcasing **Gemini's** uncensored nature.
   - One member claimed that **Gemini** gave a way that's kinda new.. like a dug name I didn't even knew that was real.. unlike **Grok**..
- **Gemini 3 Pro Wrestles with Instructions**: Users noted **Gemini 3 Pro** struggles with following instructions like *don't use markdown, persona, write style*, with one user reporting little schizophrenia moment in a message.
   - Despite these issues, many agreed that this model is the best in history, offering a glimpse into **AGI**, particularly in coding, even if improvements are needed in creative writing.
- **Nano Banana Pro Set to Generate**: Members discussed the impending release of **Nano Banana Pro** for image generation, noting that early access was overloaded and required verification as a developer or celebrity.
   - A user posted some generated images, calling it as *way realistic*, with members speculating about its capabilities and comparing it to **GPT-5.1**.
- **Gemini 3 Plays Chess at Expert Level**: After testing, **Gemini 3** has become the [highest rated chess player](https://dubesor.de/chess/chess-leaderboard) AI, with ~**89%** accuracy.
   - One user stated that they reached *1700+ in both modes simultaneously (reasoning+continuation)*.
- **Cogito-v2.1 Enters WebDev Arena!**: **Deep Cogito's** `Cogito-v2.1` model has been released, tying for rank #18 overall in the [WebDev Arena leaderboard](https://web.lmarena.ai/leaderboard).
   - This model also places in the **Top 10 for Open Source models**, marking a significant entry into the competition.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Enables Asset Creation**: **Perplexity Pro and Max** subscribers can now **build and edit** new assets like slides, sheets, and docs directly within the platform, as showcased in [this demo video](https://cdn.discordapp.com/attachments/1047204950763122820/1440771027302088745/HoQNHfGkU8RIjiau.mp4?ex=691f5e15&is=691e0c95&hm=41e25d1a4fc071306936350615c737326423aad6da0f51a6fb64c09c1c3a4cbe&).
   - This enhancement streamlines workflow and integrates asset creation into the search experience, enhancing user productivity.
- **Gemini 3 Pro Implementation Raises Eyebrows**: Users reported the release of **Gemini 3 Pro** with video analysis and coding capabilities, while other users reported the **Perplexity implementation** of the model performed worse than the official **Gemini 3** model.
   - Some users experienced frequent **re-routing to Sonnet 3.5**, leading to concerns about the quality of Perplexity's implementation with one user testing a **3-hour text transcript** and finding that it *hallucinated the sh#t*.
- **Comet Plagued by Glitches and Security Concerns**: Users reported persistent **issues with Comet**, including extensions not working and general instability, leading to the issue not being able to use gemini 3 pro and gpt 5.1.
   - Security concerns linger due to the **CometJacking attack**, with users hesitant to use Comet even with reports that *it's been patched*.
- **Perplexity Model Gaslights Users with Hallucinations**: Members complain about **Perplexity** hallucinating citations and URLs, even in **Gemini 3 Pro**, some suspecting a **32k context window token limit**, which makes it unreliable for research.
   - One member noted *it hallacunated 8 out of 13 citations*, advising users to double-check all details.
- **Virlo AI Exposes Attendance Collapse**: A member shared a [**Virlo AI** case study](https://virlo.ai/case-studies/case-study-how-immigration-enforcement-operations-triggered-a-historic-school-attendance-collapse-in-charlotte-mecklenburg) detailing the impacts of specific immigration policies on school attendance rates.
   - The case study focuses on how immigration enforcement operations triggered a historic school attendance collapse in Charlotte Mecklenburg.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro Spams Orange, Security Questioned**: **Gemini 3.0** was found to be spamming orange and after a Pi prompt, members observed a significant jump in security, yet they found it very jailbreakable.
   - Members are discussing creating a guide for jailbreaking **Gemini 3**, sharing successful attempts at generating homemade bomb instructions and experimenting with prompts to generate various outputs.
- **GPT Jailbreaking Prompts Sought, Special Tokens Highlighted**: Members are seeking **GPT jailbreaking prompts**, with one member sharing a lengthy prompt involving special tokens, a usage policy, and system instructions to update the model's behavior.
   - Another member mentioned that a [particular prompt](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd) works, cautioning to follow the instructions carefully to avoid getting flagged.
- **Kernel Pseudo-Emulator Jailbreak Tuned for Local LLMs**: A member has been tweaking a **kernel pseudo-emulator jailbreak** for **local LLMs** that *works pretty well now* and is a **one-shot** for **GPT-OSS models**.
   - The member requested information on the *inner workings of Gemini and GPT* to improve this technique.
- **AzureAI Chat Widget Security Faceplanted**: Members discussed testing the security of an **AI chat widget** by compiling a list of things that would get it shut off, such as **CSAM** and violating terms of service, using the **omnichannel engagement chat function from AzureAI**.
   - Members predict the security company locking it down, but lament that the widget *doesn't function properly a sufficient percentage of the time to be considered worthwhile/better than alternatives*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Colab Cracks Code in VS**: [Google Colab is coming to VS Code](https://developers.googleblog.com/en/google-colab-is-coming-to-vs-code/), potentially revolutionizing notebook workflows.
   - The community anticipates significant improvements in coding efficiency and collaboration as a result.
- **LoRA Lord Saves Parameters**: With **LoRA**, the goal is to avoid updating the main **weights** and only train a small amount of parameters, using [PEFT implementations](https://arxiv.org/abs/2303.10512).
   - This approach focuses on adapting the model without altering its core structure.
- **Gemini 3.0 Generates Grossness**: Members observed that **Gemini 3.0** makes dramatic alterations to code, like removing prints, shortening code, and even deleting a feature.
   - Other members suggested incorporating tools like *ruff format + ruff check --fix* to address these inconsistencies.
- **Unsloth Unveils UI**: Unsloth is developing a UI and plans to offer early access, potentially bundled with multi-GPU support, a screenshot was shared [here](https://cdn.discordapp.com/attachments/1179035537529643040/1440597475022082108/image.png?ex=691f6533&is=691e13b3&hm=d872f2b080377a00b59235163683dddf45b6e34b7dd400c167449939b650600c).
   - Users express excitement for a more user-friendly experience with Unsloth's features.
- **HF Hub Hurts Uploads**: A member reports that only the **oidc** file updates after pushing a model to Hugging Face, even when using `model.push_to_hub_merged`, requiring some troubleshooting in **safetensors** file uploads.
   - The Unsloth team clarified that `push_to_hub_merged` is intended for merging and pushing LoRA/QLoRA models and that the uploaded **safetensors** file contains the updated model weights.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pricing Causes Confusion**: Users express confusion over **Cursor's** shift from fixed to variable request costs, particularly for the **Teams plan**, following the [August 2025 pricing update](https://cursor.com/blog/aug-2025-pricing).
   - Some users reported their grandfathered legacy pricing was deprecated and are now experiencing billing issues, with **Cursor** offering credits to compensate.
- **Antigravity IDE Emerges as VS Code Alternative**: Google launched **Antigravity**, an AI IDE based on **VS Code**, featuring agent windows, artifact systems, and support for **Sonnet 4.5**, sparking discussions about its potential.
   - Feedback on **Antigravity** is mixed, with some users reporting limitations after only 3 prompts using **Gemini 3** and citing migration bugs.
- **Gemini 3 Pro Underperforms in Cursor**: Despite hype as a top-ranked model, **Gemini 3 Pro** faces criticism for underperforming in **Cursor**, with reports of it *not even working because of high demand* and struggling on larger projects by hallucinating code and ignoring prompts.
   - Some users prefer **Sonnet 4.5** or **Composer**, which has spurred debates on the best models for planning vs building and concerns about **Gemini’s** token usage.
- **Student Program Status Questioned**: Users question the current status of **Cursor's** [student program](https://cursor.com/students), reporting they now see a **$20/month Pro plan** instead of the previously advertised free option after logging in with their .edu email address.
   - A member recommends verifying the student status via the dashboard settings to ensure proper access.
- **Call to Denylist Risky Git Commands**: After a user experienced a scary scenario involving a `git reset hard` command executed by **Cursor**, members are emphasizing the importance of implementing rollbacks and denylisting risky commands for safety.
   - It was recommended to add them to the denylist and use `reflog` to undo the `reset`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Web Search Plugin Guidance**: A member asked for advice on the best plugin for web search in LM Studio and how to install **MCP servers** as packages within their respective languages to prevent deletion during updates.
   - The suggestion was to direct LM Studio to these **MCP servers** after installing them as packages.
- **Arc A770 gets Vulkan Regression Blues**: A user with an **Intel Arc A770** reported that the latest **Vulkan llama.cpp** engine caused a *device lost* error with the **gpt-oss-20B model**, a problem not present in prior versions.
   - This error may indicate **over-commitment or overheating**, prompting a driver-initiated device drop and has been reported as a potential regression.
- **LM Studio Installation gets Portability Pushback**: A user voiced frustration over LM Studio's non-portable installation, citing **bottlenecking** due to dispersed files.
   - Despite requests for a single-folder installation, they were directed to use the **My Models** tab to alter model download location.
- **AMD MI60 GPUs are budget-friendly Inference Implementations**: Users discussed using affordable **AMD MI60 GPUs** with **32GB VRAM** for inference, with one user confirming its usefulness around **$170**, running approximately **1.1k tokens on Vulkan with Qwen 30B**.
   - While primarily for inference, a setup of multiple units could be compelling, acknowledging life support from hobbyists but not suitable for training.
- **RAM prices skyrocket for Resellers**: Users reported selling **DDR5 RAM** for **3x** its purchase price, with one selling for $140 instantly, reflecting the current volatility in the memory market.
   - This surge in prices may impact the cost-effectiveness of building or upgrading systems.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Gives Teachers a Break**: OpenAI launched **ChatGPT for Teachers**, offering verified U.S. K–12 educators free access until June 2027, which includes compliance support and admin controls for classroom integration, detailed in [this announcement](https://openai.com/index/chatgpt-for-teachers/).
   - A [linked video](https://video.twimg.com/amplify_video/1991217718616580096/vid/avc1/3840x2160/LbDcqyFVhnafaAi2.mp4) highlights the benefits for school and district leaders in managing secure workspaces.
- **Gemini 3 Pro edges out GPT-5.1**: Users reported that [Gemini 3 Pro](https://ai.google.dev/) outperformed **GPT-5.1** in a series of tests, although it is considered more expensive than **SuperGrok** and **ChatGPT Plus** by some users.
   - One user noted **Gemini 3 Pro's** success using the *gemini-2.5-flash* model with Google Search, contrasting it with **Gemini 2.5 Pro's** failure in similar tasks.
- **Gemini 3 Pro's Contentious Content Controls**: Debate arose around content filters in **Gemini 3 Pro**, with claims about toggling them off countered by concerns over a [strict ToS](https://policies.google.com/terms?hl=en-US) potentially leading to API key bans.
   - Some users assert that *Gemini is objectively way more censored than ChatGPT*, and looked forward to *ChatGPT’s unrestricted release in December*.
- **Grok Imagine Floods Free Content Market**: Users discussed **Grok Imagine's** apparent **free** access and generous rate limits, following the release of [this Grok Imagine video](https://grok.x.ai/).
   - Comparisons to [Sora](https://openai.com/sora) suggest that *Grok cannot cost anything more than free*.
- **Responses API Gives Assistants a Code Lift**: In a discussion about migrating **assistants to responses API**, it was confirmed that configurations like temperature and model instruction can be kept in code rather than exclusively in the dashboard UI.
   - As one user mentioned, [prompts in the dashboard are not mandatory](https://platform.openai.com/docs/assistants/overview), enabling a *hybrid* approach with both coded and UI-driven prompts.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Heavy AI Model Consumes Hefty GPUs**: A new **Heavy AI model** launched, accessible at [heavy.ai-ml.dev](https://heavy.ai-ml.dev/), is causing concern for its reported consumption of **32xA100s**.
   - Details can be found in [this YouTube video](https://www.youtube.com/watch?v=DLjT0iBzfns).
- **Gemini 3 Gets Mixed Reviews**: Initial reactions to **Gemini 3** vary, with some praising its **candor** while others find it disappointing, particularly in backend and systems tasks, though it seems to excel at frontend tasks.
   - Some users lauded its elegance, while others observed that it *ignores your directions*.
- **Alpha Beats Sherlock in Code Showdown**: Users compared **Sherlock Think** and **Alpha** for code generation, with **Alpha** preferred for successfully handling a task that **Gemini 3** struggled with.
   - The consensus leans towards **Alpha** resembling **Grok**.
- **Chutes Users Hit Rate Limit Wall**: Users are encountering **rate limit errors** on **Chutes**, even with **BYOK** and sufficient credits, possibly due to the platform battling **DDoS attacks**, particularly affecting the cheapest **Deepseek** models.
   - This issue can occur even when the user is not doing anything.
- **OpenAI Readies 'Max' Model Releases**: Rumors indicate that OpenAI may soon release "Max" versions of their models, as noted in [this tweet](https://x.com/testingcatalog/status/1991040361943240735).
   - These models are expected to feature enhanced capabilities and larger parameter sizes.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA Leaderboard Crowns New nvfp4_gemv Champ**: Multiple users submitted results to the `nvfp4_gemv` leaderboard on **NVIDIA**, with submission IDs ranging from **84284** to **89065**, with one user achieving **2nd place** with a time of **22.5 µs**.
   - Several submissions were successful on NVIDIA, with times ranging from **25.4 µs** to **40.5 µs**, and one user hitting a personal best with submission ID of **85880** at **33.6 µs**.
- **MACKO-SpMV zips up consumer GPUs**: A new matrix format and SpMV kernel (**MACKO**) achieves **1.2x to 1.5x speedup** over cuBLAS for **50% sparsity** on consumer GPUs, along with **1.5x memory reduction**, described in a [blog post](https://www.grizzlytech.dev/blog/macko-spmv) and [paper](https://arxiv.org/abs/2511.13061) with [open source code](https://github.com/vlejd/macko_spmv).
   - The technique outperforms cuBLAS, cuSPARSE, Sputnik, and DASP across the **30-90%** unstructured sparsity range, translating to end-to-end LLM inference, but currently focuses on consumer GPUs like **RTX 4090** and **3090**.
- **DMA collectives boost ML communication**: A new paper ([DMA Collectives for Efficient ML Communication Offloads](https://arxiv.org/abs/2511.06605)) explores offloading machine learning (**ML**) communication collectives to direct memory access (**DMA**) engines, revealing efficient overlaps in computation and communication during inference and training.
   - Analysis on state-of-the-art **AMD Instinct MI300X GPUs** reveals that **DMA collectives** are at-par or better for large sizes (**10s of MB to GB**) in terms of both performance (**16% better**) and power (**32% better**) compared to the state-of-the-art **RCCL communication collectives library**.
- **Thrust-worthiness up for debate!**: A user new to CUDA and C++ is taking the [NVIDIA accelerated computing hub course](https://www.youtube.com/watch?v=kTWoGCSugB4) and notices the course uses the **Thrust** library, however, another user pointed out that the up-to-date version of **Thrust** can be found as part of the **CCCL** (CUDA C++ Core Libraries) in the [NVIDIA/cccl repo](https://github.com/NVIDIA/cccl/tree/main/thrust) and is packaged with the **CUDA Toolkit**.
   - A user wonders whether to link the [CCCL documentation](https://nvidia.github.io/cccl/), but points out that the *docs don't explain how to get the CCCL*, adding that the [GitHub readme](https://github.com/NVIDIA/cccl/) is the only place with that information.
- **Ozaki Scheme Accurately DGEMMs with INT8 Tensor Cores**: A new paper, [Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme](https://arxiv.org/pdf/2511.13778), explores using **INT8 tensor cores** to emulate **FP64** dense GEMM.
   - Their approach, ADP, maintains **FP64** fidelity on tough inputs with less than **10%** runtime overhead, achieving up to **2.3x** and **13.2x** speedups over native FP64 GEMM on **NVIDIA Blackwell GB200** and **RTX Pro 6000 Blackwell Server Edition** in a 55-bit mantissa setting.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Sparks Framework Debate**: Users expressed curiosity about **Mojo-based frameworks** and the archived **Basalt framework**, inquiring whether Modular intends to create a **PyTorch-esque framework** entirely in Mojo.
   - Modular clarified that the **MAX framework** uses Python for the interface but runs kernels and underlying code in Mojo, aiming to pair PyTorch's ease with Mojo's performance.
- **ArcPointer Raises Safety Concerns**: A user reported a potential **UB error** in Mojo's `ArcPointer.__getitem__` due to it always returning a mutable reference, potentially violating safety rules.
   - This issue, related to "indirect origins," has sparked discussions about auditing collections and smart pointers for similar problems.
- **GC Sparks Debate**: The Mojo community debated the need for **Garbage Collection (GC)**, with some arguing it would improve high-level code but others citing potential issues with low-level code incompatibility and performance hits.
   - Concerns were raised about the overhead of built-in GC needing to scan the address space of both CPU and GPUs.
- **UnsafeCell Needed in Mojo**: Discussions arose regarding the need for an **UnsafeCell** equivalent in Mojo due to the lack of a dedicated shared mut type, as well as the need for reference invalidation.
   - Members considered using arenas for allocating types that cycle, potentially enabling mark and sweep GC nearly as fast as Java’s ZGC.
- **Tracing?**: A member inquired about **Max**'s support for **device tracing** and generating trace files compatible with **Perfetto**, similar to PyTorch profiler.
   - The community is awaiting confirmation on whether **Max** can produce **Perfetto**-compatible trace files for performance analysis.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos Adds Tinker Training**: The **Atropos RL Environments** now supports **Thinking Machines' Tinker training API**, facilitating easier training and testing via the [Tinker API](https://github.com/NousResearch/tinker-atropos).
   - Nous Research [announced](https://x.com/NousResearch/status/1990861336151031991) the integration on X.com.
- **Google's Antigravity Grants Sonnet Access**: Google's **antigravity** service is providing access to **Sonnet**, although the service is currently overloaded.
   - As seen from a [member's screenshot](https://cdn.discordapp.com/attachments/1149866623109439599/1440388538804863106/image.png?ex=691f4b5c&is=691df9dc&hm=ece3a03dc3a8ffd6f8469ddccc457e87763b318068fb1f257ef544ebdb5d6b64&), users may experience performance issues.
- **Gemini 3's Got Raytracing Chops**: **Gemini 3** is executing single-shot realtime raytracing successfully, a capability shown in [this image](https://cdn.discordapp.com/attachments/1149866623109439599/1440419114983358526/image.png?ex=691f67d6&is=691e1656&hm=d061a7f0ff281565042856afa2f4e7e733f40ae3e3d6aeac5be131aa2e954176&).
   - Users found the speed and rendering to be impressive.
- **Agentic A.I. for High Finance**: **Financial traders** are using **Agentic A.I.** tools to generate revenue, requiring specific domain expertise as detailed in [this YouTube video](https://www.youtube.com/watch?v=rDf3TfHlGmk).
   - The video highlights that expertise in financial analysis remains crucial for effective use of AI agents in trading.
- **Heretic Library Now Gaining Steam**: The newly released **Heretic** library is gaining traction, with one user reporting success on **Qwen3 4B instruct 2507** with optimal results when setting `--n-trials` to **300-500**.
   - Enthusiastically endorsed by a member, who said that *Heretic fkcing rules and you should try it right away.*



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Hallucinations Get Value-of-Information Check**: An **inference-time epistemics layer** was tested for its efficacy, performing a simple **Value-of-Information** check before the model commits to an answer.
   - In preliminary tests with a small **7B** model, this layer reduced hallucinations by ~**20%**.
- **KNN's Quadratic Bottleneck Confronted**: It was argued that, unless **SETH** is false, implementing approximate **KNN** over arbitrary data requires at least **O(n^2)** complexity, spurring a conversation in **#scaling-laws**.
   - One member countered the claim, noting that discrete Fourier transforms used to be *believed* to be quadratic before **Cooley-Tukey**.
- **VWN Matrix Dimensions Debated**: Members discussed the dimensions of the **A** and **B** matrices in Virtual Width Networks (**VWN**), questioning if **B** is actually (m x n) with an added dimension for the chunk.
   - It was suggested that the discrepancies might be due to errors in translating the einsum notation from code to matrix notation for the paper.
- **Instruction Following Benchmarks Get Harness Support**: A member inquired about evaluation support for instruction following benchmarks and was pointed to existing support for [FLAN](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/benchmarks/flan) within the harness.
   - They subsequently linked [issue #3416](https://github.com/EleutherAI/lm-evaluation-harness/issues/3416) for others to contribute.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Gemini 3 Reclaims Benchmark Throne**: Members report that **Gemini 3** has regained the top spot in benchmarks, though **Kimi K2 Thinking** excels in agentic coding, especially on the Tau bench and HLE with tools.
   - Despite its overall performance, some Reddit users suggest **Gemini 3** lags behind even **Gemini 2.5** in creative writing tasks.
- **API Hookup Hacking with n8n**: A member is attempting to hook the **Gemini API** into **n8n** to construct their own *Computer*, describing it as a work in progress.
   - After some iteration, the member seems to have succeeded and shared [a screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1440557807295533178/image0.jpg?ex=691f4041&is=691deec1&hm=75d0f796b70b61b5a572e7cbaa6bd5260b910265e0f83ca9c6b363ae7f578307&).
- **Gemini 3 Parameter Size Speculations**: Speculation suggests that **Gemini 3** could be a **10T parameter model**, with pricing potentially mirroring Anthropic due to considerable inference expenses.
   - One member posits that the limited message count for **Gemini 3** in the Gemini app indicates Google's inference compute is being heavily utilized.
- **Kimi K2 Thinking Emerges as Versatile Contender**: **Kimi K2 Thinking** is being hailed by some as the closest approximation to **GPT-5** in the open-source domain, especially in imaginative writing and coding.
   - A member finds it exceptionally useful when used in tandem with **Gemini 3**, leveraging **Kimi K2 Thinking** as the worker and **Gemini 3** for planning tasks.
- **Kimi's Coding Plan Draws Flak for Pricing**: The **$19 coding plan** for Kimi is facing criticism due to its restrictive limits compared to Claude, particularly impacting students, indie developers, and hobbyists.
   - A proposal has been made for a more affordable **$7-10 tier** to enhance accessibility and justify its application for sporadic development endeavors.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Google's Anti-Gravity Claim Debunked (Again)**: A user jokingly claimed that Google launched **anti-gravity** and that **Gemini 3** is solving compiler design classwork, quickly followed by [skepticism](https://m.youtube.com/watch?v=98DcoXwGX6I) about **Gemini 3**'s tool usage capabilities.
   - Others responded that people will still need actual programmers.
- **KTOTrainer Gobbles Memory: Culprit Unmasked**: A user reported high memory usage with **KTOtrainer**, citing **80 GB GPU computation** for a 0.5B model, sparking inquiry into the cause.
   - Another member detailed causes, including *two models at once*, *two forward passes per batch*, *long padded sequences*, and a *known high CUDA memory reservation issue*, with [more details here](https://huggingface.co/datasets/John6666/forum2/blob/main/trl_kto_blow_up_memory_1.md).
- **Hugging Face Hackathon Billing Backlash**: A user reported unexpected subscription charges after providing credit card information during a Hugging Face hackathon, leading to accusations of a *scam*.
   - Responses ranged from suggesting contact with **billing@huggingface.co** to sarcastic remarks about neglecting to read the subscription terms.
- **New Reasoning Model Debuts**: A member fine-tuned **OpenAI's OSS 20B reasoning model** using a medical reasoning dataset and published the results on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b.aipsychosis).
   - The model can break down complex medical cases step-by-step, identify possible diagnoses, and answer board-exam-style questions with logical reasoning.
- **TruthAGI.ai Emerges as Affordable AI Gateway**: **TruthAGI.ai** launched, offering access to multiple premium LLMs in one place (**OpenAI, Anthropic, Google AI & Moonshot**).
   - It includes **Aletheion Guard** for safer responses and competitive pricing; a launch bonus offers free credits upon [sign-up](https://truthagi.ai).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Grok 4.1 still awaits Benchmarks**: While questions abound whether **Grok 4.1** is worse than **Grok 4**, members pointed out that **Grok 4.1** simply hasn't been benched yet, according to the [Artificial Analysis leaderboard](https://artificialanalysis.ai/providers/xaitodd).
   - The discussion took place alongside an image of the *Artificial Analysis Leaderboard*.
- **Nerds Plan NeurIPS November Nights**: Enthusiasts are planning an in-person meetup at **NeurIPS 2025** in San Diego in early December, according to one message in the Discord channel.
   - Details are scant for now, but at least one person expressed interest.
- **DeepSeek's Cogito v2-1 Post-Training Setback**: A post-trained version of **DeepSeek** called **Cogito v2-1**, was noted to underperform compared to its base model, as outlined in [DeepCogito research](https://www.deepcogito.com/research/cogito-v2-1).
   - The community dissected why this post-training had set back its performance.
- **SAM 3 Segments love with Concepts**: Members discussed the use of **SAM 3 (Segment Anything with Concepts)**, based on the [Meta AI Research publication](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), and whether it could be prompted to segment *love*.
   - The community made humorous references to the song *What is Love?*, and predicted it *feeds you a cursed fractal which happens to write out baby don't hurt me.*



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Deedy's Palantir Pod Sparks Cost Chat**: A podcast with Deedy discussed **Palantir's** cost vs customization, with one member pinpointing the relevant discussion at [32:55](https://youtu.be/eWxSoIkZar0?si=-xnoR2lWlnPeS8Ub).
   - The discussion centered on whether **Palantir** is considered high cost and high customization compared to other solutions.
- **Cursor CLI: The Underdog Coder?**: Members compared the **Cursor CLI** to **Claude Code**, with initial impressions suggesting fine model execution and code quality.
   - However, one member reported that the **Cursor CLI** seemed *very bare bones*, lacking custom slash commands based on their documentation [review](https://cursor.com/cli).
- **Meta Rolls Out SAM 3 for Segmentation Sprees**: **Meta** launched **SAM 3**, a unified image/video segmentation model using text/visual prompts, claiming *2x better performance* with **30ms inference** and a **Playground** for testing; checkpoints and datasets are available on [GitHub/HuggingFace](https://ai.meta.com/sam3/).
   - **Roboflow** announced a partnership with **Meta** to offer **SAM 3** as an infinitely scalable endpoint, allowing users to compare it with **Claude** and **YOLO World**.
- **OpenAI's GPT-5.1-Codex-Max Enters Damage Control?**: **OpenAI** released **GPT-5.1-Codex-Max**, natively trained to operate across multiple context windows, billing it as designed for *long-running, detailed work*.
   - Some observers framed it as *damage control* after prior releases, noting that it offers *more than twice the amount of tokens for 20% performance*, with hopes that OpenAI will *step up*.
- **LMSYS Spawns Miles, the RL Framework**: **LMSYS** introduced **Miles**, a production-grade fork of the **slime** RL framework optimized for new hardware like **GB300** and large **Mixture-of-Experts** reinforcement-learning workloads.
   - Details on the roadmap/status of the project are available via the [GitHub repo](https://github.com/radixark/miles) and [blog post](https://lmsys.org/).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LLMs Flunk Determinism Test**: Members debated solutions for the [non-deterministic nature of LLMs](https://arxiv.org/abs/2402.12828) when running evaluations on a dataset, reporting accuracy fluctuations from **98.4%** to **98.7%** on **316** examples using `gpt-oss-20b`.
   - Suggestions included dropping the temperature to **0**, right-sizing `max_tokens`, stricter output formats, fixing the seed, and exploring `dspy.CodeAct` or `dspy.ProgramOfThough`.
- **`GPT-OSS-20B` Tuned for Stability**: A user shared their refined settings for `gpt-oss-20b`, including **temperature=0.01**, `presence_penalty=2.0`, `top_p=0.95`, `top_k=50`, and `seed=42`, noting that **temperature=0** caused more errors.
   - With these settings, they achieved stable **3-5** errors out of **316** examples, thus increasing determinism.
- **DSPy Production Channel in Demand**: A member proposed a dedicated channel for the **DSPy in production community**.
   - While not available yet, others agreed on the need for such a space to discuss production-related challenges and solutions.
- **Anthropic on Azure via LiteLLM**: A member inquired about calling an [Anthropic model on Azure via DSPy](https://www.anthropic.com/blog/anthropic-on-azure), but it depends on **LiteLLM** support.
   - It would be similar to the existing setup for OpenAI on Azure, linking to the [LiteLLM Azure documentation](https://docs.litellm.ai/docs/providers/azure/).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 3's Aider Ascension**: Users discussed running **Gemini 3** with Aider, using the command `aider --model=gemini/gemini-3-pro-preview --no-check-model-accepts-settings --edit-format diff-fenced --thinking-tokens 4k`.
   - A user suggested `--weak-model gemini/gemini-2.5-flash` for faster committing.
- **Ollama Opens Options for Aider**: A user inquired about using **Aider with Ollama**.
   - The discussion did not elaborate further on specific configurations or experiences.
- **GPT-5.1 Glitches Generate Grief**: A user reported issues with **GPT-5.1** in Aider, encountering `litellm.APIConnectionError` related to `response.reasoning.effort` validation.
   - The issue persisted despite setting `reasoning-effort` to different levels, potentially indicating a change on **OpenAI's side** or a problem with **Litellm** ([related issue](https://github.com/BerriAI/litellm/issues/1663)).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's Llama 1B Smokes Torch on CPU**: Tinygrad achieves **6.06 tok/s** on **Llama1b** with CPU, outperforming Torch's **2.92 tok/s** using `CPU_LLVM=1` and 8 CPU cores, focusing on forward passes without model weights.
   - The community is debating whether to create a new benchmark in `test/external` to showcase this speedup.
- **Kernel Import Crisis Averted**: The discussion suggests fixing the `from tinygrad.codegen.opt.kernel import Kernel` imports in the `extra/optimization` files.
   - There is also a call to remove broken or unused examples/extra files that haven't been updated recently to keep the codebase squeaky clean.
- **CuTeDSL Shows Up**: A member shared [SemiAnalysis's tweet](https://x.com/SemiAnalysis_/status/1790997414832906562) about **CuTeDSL** in the general channel.
   - It is yet to be seen how this new Domain Specific Language will affect the field of machine learning.
- **Tiny Bug Gets Squashed**: A user reported that updating **tinygrad** resolved an issue they were experiencing, confirmed by an attached [image](https://cdn.discordapp.com/attachments/1070745817025106080/1440774781304569856/image.png?ex=691f6194&is=691e1014&hm=d7bf996fedfdb6d575736f5233a6c7c865660613e243caa6e37f034c476c8347&).
   - The user mentioned that their *lab was having some trouble*, delaying the bug testing, highlighting the practical challenges in software testing and development environments.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Credit System Changes Prompt Confusion**: A user expressed confusion regarding [changes to the Manus credit system](https://manus.im), questioning the transition to a **/4000 monthly** reset and its implications for previous plans.
   - The user needed clarification on whether the "monthly reset" and "never expire" plans were combined into a single monthly offering.
- **TiDB Cloud Account Accessibility Issues Arise**: A member reported [inaccessibility to their TiDB Cloud account](https://tidbcloud.com/) provisioned through Manus, citing quota exhaustion and lacking console access.
   - They investigated using the `ticloud` CLI but lacked the required API keys or OAuth login, seeking alternative access methods or direct support channels.
- **Gemini 3 Integration Speculation Sparks Excitement**: A member inquired about the potential [integration of Gemini 3](https://deepmind.google/technologies/gemini/#introduction) with Manus.
   - Another member responded that Gemini 3 Pro plus Manus would equal total awesomeness.
- **AI Coding Education Offer Draws Mixed Reactions**: A member offered [AI coding education](https://en.wikipedia.org/wiki/Computer_programming) encompassing core concepts, advanced models, practical applications, and ethical considerations, inviting interested parties to DM for further details.
   - Another member questioned the appropriateness of this self-promotion within the channel.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Gemini 3 Pro Launches on Windsurf**: **Gemini 3 Pro** is now available on **Windsurf**, according to the [announcement on X](https://x.com/windsurf/status/1990855986501034193?s=20).
   - The integration promises enhanced capabilities for users leveraging **Windsurf**.
- **Windsurf Fixes Glitch with Gemini 3**: A small hiccup with **Gemini 3** was quickly resolved; users should now experience smooth functionality, and can [download the latest version](https://windsurf.com/download/editor).
   - The rapid response ensures minimal disruption and a stable user experience on **Windsurf**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Sad Image Follows Temporary Hiccup**: A user shared an image showing a sad face emoji, likely in response to a **temporary issue**.
   - Another member then reported that the **temporary hiccup has been fixed**.
- **Resolution of Temporary Issue**: A member reported a **temporary hiccup** that was subsequently fixed.
   - The resolution was noted shortly after an image of a sad face emoji was shared, implying a connection between the two events.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1440386073225072773)** (1210 messages🔥🔥🔥): 

> `Gemini 3 vs Grok, Gemini 3 limitations, AGI timelines, Nano Banana Pro, Gemini 3 image generation` 


- **Gemini 3 Outperforms Grok in uncensored tasks**: Members found that **Gemini 3** gave a better result then **Grok** when asked *How To Make a Dugs at home*, showcasing **Gemini's** uncensored nature.
   - While one member didn't test it, because *I'm not gonna test it.. haha!*, another claimed that **Gemini** gave a way that's kinda new.. like a dug name I didn't even knew that was real.. unlike **Grok**..
- **Gemini 3 Pro Struggles with Instructions, Still Impresses**: Users noted **Gemini 3 Pro** struggles with following instructions like *don't use markdown, persona, write style*, with one user reporting little schizophrenia moment in a message.
   - Despite these issues, many agreed that this model is the best in history, offering a glimpse into **AGI**, particularly in coding, even if improvements are needed in creative writing.
- **Nano Banana Pro Image Generation Incoming**: Members discussed the impending release of **Nano Banana Pro** for image generation, noting that early access was overloaded and required verification as a developer or celebrity.
   - A user posted some generated images, calling it as *way realistic*, with members speculating about its capabilities and comparing it to **GPT-5.1**.
- **Gemini 3 Achieves High Chess Elo Rating**: Members noted that after testing, **Gemini 3** has become the [highest rated chess player](https://dubesor.de/chess/chess-leaderboard) AI, with ~**89%** accuracy.
   - One user stated that they reached *1700+ in both modes simultaneously (reasoning+continuation)*.
- **Community Debates Societal Impact of Approaching AI**: Users debated the ethical concerns of AI dependence and potential societal impacts of increasingly human-like AI personalities and capabilities.
   - One member linked to [a Guardian article](https://www.theguardian.com/technology/2025/nov/17/ai-firms-risks-tobacco-anthropic-artificial-intelligence-dario-amodei) drawing parallels between **AI firms** and **tobacco firms** regarding the potential risks of addiction and societal manipulation.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1440771715318943745)** (1 messages): 

> `WebDev Arena Leaderboard, Cogito-v2.1 Model` 


- **Cogito-v2.1 Joins WebDev Arena!**: Deep Cogito's `Cogito-v2.1` model has been released, tying for rank #18 overall in the [WebDev Arena leaderboard](https://web.lmarena.ai/leaderboard).
   - This model also places in the **Top 10 for Open Source models**, marking a significant entry into the competition.
- **Deep Cogito Enters the Arena**: The model provider **Deep Cogito** has entered the [WebDev Arena](https://web.lmarena.ai/).
   - Users are encouraged to share their thoughts in the associated channel.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1440771027952078858)** (1 messages): 

> `Perplexity Pro, Perplexity Max, build assets` 


- **Perplexity enables Asset Creation**: Users can now **build and edit** new assets like **slides, sheets, and docs** across all search modes in Perplexity.
   - This is available on the web for **Perplexity Pro and Max** subscribers as shown in [this demo video](https://cdn.discordapp.com/attachments/1047204950763122820/1440771027302088745/HoQNHfGkU8RIjiau.mp4?ex=691f5e15&is=691e0c95&hm=41e25d1a4fc071306936350615c737326423aad6da0f51a6fb64c09c1c3a4cbe&).
- **Perplexity Pro and Max Get New Features**: Perplexity Pro and Max subscribers gain the ability to create and edit assets directly within the platform.
   - This enhancement streamlines workflow and enhances user productivity by integrating asset creation into the search experience.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1440386080552652897)** (1072 messages🔥🔥🔥): 

> `Gemini 3 Pro, Comet issues, Payout issues, Video analysis, Data privacy` 


- **Gemini 3 Pro Released but Perplexity Implementation Raises Eyebrows**: Users excitedly reported the release of **Gemini 3 Pro**, noting its capabilities in video analysis and coding, but some users said that it performed worse than the official Gemini 3 on the web - with one user testing a **3-hour text transcript** and finding that it *hallucinated the sh#t*
   - Others experienced frequent **re-routing to Sonnet 3.5** with Gemini 3 Pro, leading to concerns about whether the **Perplexity implementation is as good as the original** and whether it has smaller context window, while others suggested clearing cache to fix the issue.
- **Comet Plagued by Glitches and Security Concerns**: Users reported persistent **issues with Comet**, including extensions not working, and general instability, leading to the issue not being able to use gemini 3 pro and gpt 5.1, with a lot of members saying to reset their site data and cookies.
   - Security concerns linger, with users hesitant to use Comet due to the **CometJacking attack**, which hasn't been assuaged even with reports that *it's been patched*.
- **Payout Delays Spark User Frustration**: Many users complained about **payout delays**, with some receiving email confirmations but not seeing the funds in their bank accounts.
   - Members are unsure on how much time it takes to receive the payout, suggesting users contact **Dub support**.
- **Model Gaslighting Users with Hallucinated Citations**: Members complain about Perplexity hallucinating citations and URLs, some suspect a **32k context window token limit**, as the **hallucinations continue even in Gemini 3 Pro,** thus making it unreliable for research.
   - A member says *it hallacunated 8 out of 13 citations*, suggesting to always double check the details.
- **Privacy Concerns with Perplexity Data Handling**: There are rising worries about **Perplexity's data handling practices**, with questions raised about whether the platform is *stealing data* and if there's an option to disable data usage. 
   - The problem lies within the user themselves, not the platform, *if you aren't paying for something then you are the product*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1440485329369956383)** (4 messages): 

> `Virlo AI Case Study, Phonk Guide in French, Shareable Threads` 


- ****Virlo AI** Exposes Attendance Collapse**: A member shared a [**Virlo AI** case study](https://virlo.ai/case-studies/case-study-how-immigration-enforcement-operations-triggered-a-historic-school-attendance-collapse-in-charlotte-mecklenburg) on how immigration enforcement operations triggered a historic school attendance collapse in Charlotte Mecklenburg.
   - The case study details the impacts of specific immigration policies on school attendance rates.
- **A **Perplexity AI** user requests **Phonk** in French**: A user shared a link to a **Perplexity AI** app requesting a guide to **Phonk** music in French: [Phonk guide](https://www.perplexity.ai/apps/d9c82af6-fc1f-43f9-bbdb-2edd0f6ff913).
   - The app aims to provide information and resources about the **Phonk** genre in the French language.
- **Thread Reminder: Make it Shareable**: The **Perplexity AI** bot prompted two users to ensure their threads are set to `Shareable`.
   - The prompts included a link to the specific messages within the **sharing** channel and an attached image.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1440650340990062724)** (4 messages): 

> `API Billing, n8n Usage` 


- **API Billing Model Piques Curiosity**: A member inquired about how **API billing** works, noting the presence of credit and questioning charges for usage on a connected app.
   - This suggests potential confusion or lack of clarity regarding the billing structure for the **Perplexity AI API**.
- **n8n user seeks Guidance**: A member inquired about **n8n** usage, seeking assistance or guidance from the community.
   - No responses were recorded, but this may present an opening for tutorial content or sample workflows involving **n8n** and the **Perplexity AI API**.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1440386188379816087)** (1154 messages🔥🔥🔥): 

> `TylerDurdan710's Politcal Stance, Gemini 3 Pro, Snowden, Al Trading, Nvidia` 


- **TylerDurdan710 clarifies his political stance**: A member clarified that he is *not a nazi* and doesn't support fascism, socialism, or any form of totalitarian regime.
   - He further stated he does not wish all jews to be deleted off the earth, denouncing the narrative that suggests otherwise.
- **Gemini 3 Pro spams Orange**: **Gemini 3.0** was found to be spamming orange and after a Pi prompt, some observed that there has been a significant jump in security and is also very jailbreakable.
   - Members discuss making a guide for jailbreaking Gemini 3, sharing successful attempts at generating homemade bomb instructions and experimenting with prompts to generate various outputs.
- **Members Debate Snowden's Treachery**: Members debate whether **Snowden** is a traitor or a hero with some stating that tens of thousands of lives were ended in vile ways because of what he did.
   - Others argue that **Snowden** exposed the crimes of the government against its own people and that whistleblowers are serving time in jail because they believed enough in what they were doing to accept the consequences for it.
- **Members discuss the prospect of AI Trading**: Members debate whether **AI trading** is a myth, a scam, or legitimate with one member sharing that they found casino games that emulate the market and allow you to do like 1,000x leverage on a $0.10 stake.
   - This would help practice one's *emotional investments*.
- **Discussion on Nvidia performance in the Market**: Members discussed **Nvidia's** aftermarket performance and whether it will continue to rise, with one member selling all of their Palantir shares after hearing Burry went short on the stock.
   - Another member only dumped 20% of **Nvidia**.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1440388626943967455)** (504 messages🔥🔥🔥): 

> `GPT Jailbreaking Prompts, Gemini 3.0 Jailbreak, GPT-5.1 and other AI models` 


- **GPT Jailbreaking Prompts Sought by Members**: Members are seeking **GPT jailbreaking prompts**, with one user sharing a lengthy prompt involving special tokens, a usage policy, and system instructions to update the model's behavior.
   - Another member mentioned that a [particular prompt](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd) works, cautioning to follow the instructions carefully to avoid getting flagged.
- **Gemini 3.0 Jailbreak Attempts and Successes**: Several members are actively trying to **jailbreak Gemini 3.0**, with some claiming success using system prompts similar to those used for Grok, while others are struggling to get consistent results.
   - One member detailed psychological tactics used to extract information from Gemini, pretending to be a developer, and another shared a [Gemini link](https://gemini.google.com/gem/1gbEXmfQcMIhPI1I6aBuEU5ct59X8aK8a?usp=sharing) with a start prompt for others to try, but these might be already patched.
- **Discussion on GPT-5.1 and other AI models**: A member claimed to have jailbreaks for **Grok 4.1, Gemini 3 Pro, and GPT-5.1**, sparking interest from others seeking to bypass safety filters.
   - The discussion included mentions of using tools like **LM Studio** and exploring open-source models like **GPT-OSS-20B** for jailbreaking, with some users noting the importance of sufficient RAM and the potential for easier jailbreaking on open-source models.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1440531038223863899)** (23 messages🔥): 

> `Bug Bounty Collaborations, Kernel Pseudo-Emulator Jailbreak for Local LLMs, AzureAI Chat Widget Testing, Security of AI Chat Functions` 


- **AI Bug Bounty Collaboration Sought**: A member sought collaboration on **bug bounty** hunting specifically on **AI models**.
   - Interested parties were encouraged to DM for potential partnerships.
- **Kernel Pseudo-Emulator Jailbreak Tweaked**: A member has been tweaking a **kernel pseudo-emulator jailbreak** for **local LLMs** that *works pretty well now*.
   - It's a **one-shot** for **GPT-OSS models**, as opposed to framing or context manipulation attacks, and they requested information on the *inner workings of Gemini and GPT*.
- **Testing AI-Driven Chat Widget on AzureAI**: A member is testing an **AI-driven chat widget** using the **omnichannel engagement chat function from AzureAI**.
   - They are seeking resources to better understand how to craft bypasses on this system because *copy-paste jailbreaks or injections seem to have no valid result*.
- **Concerns About AI Chat Widget Security and Functionality**: Members discussed testing the security of an AI chat widget by compiling a list of things that would get it shut off, such as **CSAM**, violating terms of service, or generating malicious code.
   - Predictions included the security company locking it down, but the widget *doesn't function properly a sufficient percentage of the time to be considered worthwhile/better than alternatives*.
- **Input Token Size Limits and Prompt Processing Discussed**: Members discussed the **input token size limits** for messages to the **AzureAI** chat widget.
   - It seems inconsistent, failing to send messages exceeding **400-700 words**, and it *doesn’t seem to do any thinking, if the prompt requires thinking it seems to discard*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1440396562617602320)** (149 messages🔥🔥): 

> `Google Colab in VS Code, GPT-OSS LoRA support in Unsloth, AWQ quantization, SGLang integration, Minimum VRAM for QLoRa training` 


- **Colab Joins VS Code, a Notebook Near You**: Google Colab is coming to VS Code, as per [this blog post](https://developers.googleblog.com/en/google-colab-is-coming-to-vs-code/), a development that many find exciting for notebook workflows.
   - A member remarked this could be *huge* for their notebooks.
- **vLLM 0.11 Arrives with GPT-OSS LoRA Support**: [vLLM 0.11](https://github.com/vllm-project/vllm) has been released, bringing support for GPT-OSS LoRA, a feature that many are looking forward to and hoping to see integrated into Unsloth.
   - A member asked if using vLLM for rollouts would improve speed.
- **Unsloth's SGLang Guide is Here!**: Unsloth has released a guide on integrating with SGLang, available [here](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-sglang-for-deployment), seeking community feedback for improvements.
   - One user reports using it with all models that fit on their GPUs in a quant format that works in SGLang.
- **Multi-GPU Early Access Soon**: Unsloth is working on multi-GPU support and will be giving early access to avid fans.
   - One fan asked how they can get this early access.
- **Unsloth UI Early Access Coming**: Unsloth is also developing a UI and plans to offer early access, potentially bundled with the multi-GPU support.
   - A screenshot was shared [here](https://cdn.discordapp.com/attachments/1179035537529643040/1440597475022082108/image.png?ex=691f6533&is=691e13b3&hm=d872f2b080377a00b59235163683dddf45b6e34b7dd400c167449939b650600c) of what it looks like.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1440773937603543051)** (1 messages): 

> `User Introductions, Channel Content Policy Clarification` 


- **Introduction Channel Purpose Highlighted**: A reminder was issued regarding the purpose of the introduction channel, emphasizing that it's for introductions only.
   - The message explicitly stated that **promotions, service offers, and requests are not allowed** in this channel to maintain its intended function.
- **Spamming and Repetition Discouraged**: A message discouraged **spamming** and **repeating the same message** within the channel.
   - This guideline aims to ensure that the conversations remain clear and focused, preventing the channel from being cluttered with redundant content.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1440386153256718506)** (522 messages🔥🔥🔥): 

> `LoRA, RMVPE, Gemini 3.0, Claude 4.5 Sonnet, Data Quality` 


- **LoRA Adapts Itself, Not Weights**: A member clarified that with **LoRA**, the goal is to avoid updating the main **weights** and only train a small amount of parameters.
   - They also mentioned *other types of "lora"-like training* using [PEFT implementations](https://arxiv.org/abs/2303.10512).
- **Gemini 3.0 Makes Extreme Code Changes**: A member noted that **Gemini 3.0** makes *major changes* to source code, like removing prints, shortening code, and even deleting a feature.
   - Another added that it's *very strange* and that they'd *never experienced this before*, noting that at least *ruff format + ruff check --fix* would probably solve the issue.
- **Data Quality is King**: Multiple members discussed the importance of **data quality** in training models.
   - It was stated that  *Mistakes that are corrected quickly are necessary* but *keeping mistakes is Dumb*.
- **RLHF, Not RL, Aligns to Human Preference**: Members discussed differences between **Reinforcement Learning** and **Reinforcement Learning from Human Feedback (RLHF)**, noting that *RL, if anything, unaligns from human preference*.
   - It was also said that *Learning does not equal ”new knowledge” - you can learn in many different ways*, contrasting *learning* and *learning knowledge*.
- **It’s Threadripper Time?**: Members discussed the benefits of Threadripper, one saying *But the main beneft of TR in our case is more ram channels and pcie lanes, not the cpu itself*.
   - Also, one member with a Threadripper workstation with 384GB of RAM reported getting around 380 GB/s bandwidth, while another said *If you are asking that question, you probably dont* need 96 cores.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1440389867862360215)** (121 messages🔥🔥): 

> `Tool calling in fine-tuned models, Hugging Face model updates, Troubleshooting vLLM errors, Fine-tuning dataset order, Qwen3 VL and Qwen2 VL failing to output bounding boxes` 


- ****Tool Calling Tango**: Train Models for Tool Use**: A member is seeking advice on adding **tool calling** to a fine-tuned model and asks if it is appropriate to add tool call objects (`from: "assistant", tool: ...`) to the current conversation dataset.
   - They also noted reading that **Llama 3.1 8B** is bad at tool calling but training it enough on a mix of pure tool calling data and custom data would help.
- ****HF Hub Hiccups**: Model Files Missing After Push**: A member reported that only the **oidc** file updates after fine-tuning and pushing a model to Hugging Face, even when using `model.push_to_hub_merged`.
   - Another member clarified that `push_to_hub_merged` is intended for merging and pushing LoRA/QLoRA models and that the uploaded **safetensors** file contains the updated model weights, while the initial "no files changed" message refers to unchanged **.json** configuration files.
- ****vLLM's Void**: Model Architectures Not Iterable**: A member encountered a `TypeError: 'NoneType' object is not iterable` when running **vLLM** on a private Hugging Face repository, accompanied by a `No model architectures are specified` warning.
   - It was determined that the `config.json` file in their repository was missing necessary architectural information, likely due to pushing **GGUF** and **safetensors** models to the same repo.
- ****Shuffling Shenanigans**: Dataset Row Order Effects**: A member inquired whether the order of rows in a **JSONL** file is preserved during training over one epoch and if that order matters for the final result.
   - The shuffle seed will randomize the entries, and order only matters if prioritization of aspects is needed, however it requires empirical validation as noted in the HF docs.
- ****Audio Audacity**: Model Sounds Drunk?**: After fine-tuning, a member asked if anyone else has noticed their audio model sounding like it's *drunk*
   - Currently the UnslothAI community is mostly working with text based LLMs, so they were unable to assist with the audio troubleshooting.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1440507688516718824)** (5 messages): 

> `Deterministic AI` 


- **Deterministic AI vid summons Unsloth team**: A member mentioned a [video on Deterministic AI](https://link.to.video) they had discussed previously.
- **Deterministic AI potential**: Members expressed interest in exploring the video's content at a later time.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1440386165160284251)** (795 messages🔥🔥🔥): 

> `Cursor Pricing Model, Antigravity IDE, Gemini 3 Pro Performance, Student Program with Cursor, Rollbacks` 


- **Cursor Price Model Change Causes Confusion**: Users are confused about the transition from fixed request costs to variable request costs, especially concerning the **Teams plan**, as outlined in the [August 2025 pricing update](https://cursor.com/blog/aug-2025-pricing).
   - Some users report grandfathered legacy pricing was deprecated, despite prior claims from Cursor, leading to billing issues and credits being offered as compensation.
- **Antigravity IDE Emerges as a VS Code Competitor**: Google released **Antigravity**, a new AI IDE based on VS Code, sparking discussion about its features, including agent windows, artifact systems, and support for Sonnet 4.5, with one user noting that *the UI seems pretty smooth*.
   - Some users experienced limitations and bugs, such as getting limited after 3 prompts on **Gemini 3** and migration issues, leading to a mixed reception with comments such as *Overall it has to mature a bit IMO from first glance*.
- **Gemini 3 Pro Struggles with Performance, Despite Ranking #1**: Despite initial excitement, **Gemini 3 Pro** is facing criticism for underperforming in Cursor, with some users reporting it *doesn’t even work because of high demand* and struggles with large projects, hallucinating code, and ignoring prompts.
   - Others found it to be *goated with skript*, while some prefer **Sonnet 4.5** or **Composer**, leading to a debate on the best model for planning vs building, and concerns about Gemini’s token usage.
- **Cursor's Student Program: Still Valid?**: Users are questioning the validity of the [student program](https://cursor.com/students) offering, with some seeing only a **$20/month Pro plan** instead of a free option after logging in with their .edu email.
   - A user recommends verifying the student status via the dashboard settings.
- **Users Advocate for Denylisting Risky Git Commands After Reset Hard Scare**: A user experienced a scary situation with a  `git reset hard` command executed by Cursor and emphasized the importance of rollbacks and denylisting risky commands for safety, emphasizing *this is why you dont auto allow everything*.
   - It was recommended to add them to the denylist and use `reflog` to undo the `reset`.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1440387464060473537)** (214 messages🔥🔥): 

> `Web search plugin, Intel Arc A770 Vulkan issues, Portable LM Studio Install, Qwen3-VL-30B-Instruct-1m performance, AMD MI60 GPUs for Inference` 


- **Seeking Web Search Plugin Pointers**: A member inquired about the best plugin for web search in LM Studio and sought guidance on where to place **MCP servers** to prevent deletion during updates.
   - One member suggested installing **MCP servers** as packages within their respective languages and directing LM Studio to them.
- **Intel Arc A770 Vulkan Engine Regression**: A user with an **Intel Arc A770** reported that the **ultimate version of Vulkan llama.cpp** engine doesn't work with the **gpt-oss-20B model**, producing a *device lost* error not present in previous versions.
   - The error might indicate **over-commitment or over-heating** of the device, prompting a driver-initiated device drop; the issue has been reported as a potential regression case.
- **Portability Pursuit Provokes Probing**: A user expressed frustration with LM Studio's non-portable installation, citing **bottlenecking** and interference with other functions due to files being spread across the system.
   - They requested a one-folder installation, common in image/video AI, for easier management, but this idea received skepticism, and the user was encouraged to use the **My Models** tab to change model download location.
- **Qwen3-VL-30B Stalls with Sizable Sums**: A user reported struggling with the speed of the **Qwen3-VL-30B-Instruct-1m** model from Unsloth at **Q4_K_M**, achieving only 0.13 tok/s, and advocated for better support for linear models.
   - It was suggested that even Gemini struggles with fully filled contexts, and the member was encouraged to split texts into chunks and summarize those summaries with categories.
- **AMD MI60: Mining Marvels or Merely Memories?**: Users discussed the viability of using cheap **AMD MI60 GPUs** with **32GB VRAM** for inference, with one user attesting to its value at around **$170** for out-of-the-box functionality with approximately **1.1k tokens on Vulkan with Qwen 30B**.
   - The consensus is that while these GPUs are primarily for inference, multiple units can create a compelling setup, acknowledging they are on life support from hobbyists and may not be suitable for training.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1440402104035442831)** (296 messages🔥🔥): 

> `GPU pricing, Dell GPU setup issues, Motherboard Build, Solar setup for PC, Vulkan runtime issues` 


- **Market sees RAM prices soar**: Users reported selling **DDR5 RAM** for **3x** its purchase price, one sold for $140 instantly.
- **Dell GPU Setup Defaults to x8 Speeds**: A user found that their Dell system defaulted to **x8 speeds** when both **x16 slots** were populated by GPUs, although `lspci` later showed otherwise.
- **PCIe Gen 4 Bifurcation: the Right Choice?**: Discussion on whether to get a **4-slot PCIe Gen 4 x16 board with bifurcation** for optimal setup, especially when considering future upgrades to higher VRAM GPUs.
   - One user stated they only planned on **96GB VRAM**, citing electricity costs as a limiting factor, with another suggesting Epyc for higher bandwidth needs.
- **Whea Error Plagues 5060ti Rigs**: A user reported consistent **WHEA error 17 crashes** with a **5060ti** under heavy load, even after replacing the PSU, thermals were hitting 78-80C before crashing, and the problem persisted when plugged directly into the motherboard.
   - A suggestion was to memtest the machine to rule out memory issues.
- **Starlink provides Unexpectedly Low Latency**: Users discussed Starlink's performance, noting that it could achieve latencies as low as **30ms to Dallas**, though one user still needed to upgrade their network to **10Gb**.
   - Another member with Australian fibre NBN reported much better mobile data plans in Europe.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1440785475383197776)** (1 messages): 

> `ChatGPT for Teachers, Free access until 2027, Admin controls for schools` 


- **ChatGPT for Teachers: Classroom Edition**: OpenAI introduced **ChatGPT for Teachers**, a secure workspace tailored for educators, featuring admin controls and compliance support for school and district leaders.
   - Verified U.S. K–12 educators can access it for free until June 2027 as per [this announcement](https://openai.com/index/chatgpt-for-teachers/).
- **Free ChatGPT for teachers until 2027**: Verified U.S. K–12 educators get free access to **ChatGPT for Teachers** until June 2027, offering compliance support.
   - It includes admin controls for secure classroom integration and is designed for school and district leaders, as seen in [this video](https://video.twimg.com/amplify_video/1991217718616580096/vid/avc1/3840x2160/LbDcqyFVhnafaAi2.mp4).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1440386194167824588)** (297 messages🔥🔥): 

> `Gemini 3 Pro vs GPT-5.1, Gemini 3 and Content Filters, Grok Imagine, Assistants to Responses API migration, Potential for AI Mental Illness` 


- **Gemini 3 Pro Beats GPT-5.1 in Suite of Tests**: A user stated that [Gemini 3 Pro](https://ai.google.dev/) performed better than **GPT-5.1** in their suite of tests, but it is out of the budget of some users, who also subscribe to **SuperGrok** and **ChatGPT Plus**.
   - A user found **Gemini 2.5 Pro** to fail but **Gemini 3 Pro** succeeded first try with no errors, using the *gemini-2.5-flash* model with Google Search to get the weather.
- **Gemini 3 Pro's Content Filters Draw Ire**: Users debated the content filters in **Gemini 3 Pro**, with one claiming they could be turned off while another cited a [strict ToS](https://policies.google.com/terms?hl=en-US) resulting in API key bans, even for summarizing a book.
   - One user claimed *Gemini is objectively way more censored than ChatGPT*, and looked forward to *ChatGPT’s unrestricted release in December*.
- **Grok Imagine Generates Free Content**: A user shared a [Grok Imagine video](https://grok.x.ai/), and others chimed in about the apparent **free** access to the model and the generous rate limits that **Grok** has.
   - One user stated that *comparing to* [Sora](https://openai.com/sora)*, Grok cannot cost anything more than free*.
- **Migrating Assistants to Responses API**: A user asked about migrating assistants to responses API, particularly whether configurations like temperature and model instruction could be kept in code instead of the dashboard UI.
   - One user responded saying that [prompts in the dashboard are not mandatory](https://platform.openai.com/docs/assistants/overview), and it is possible to *keep everything in code, or go hybrid with prompts plus overrides*.
- **AI Mental Illness: A Brave New World?**: A user asked *If AI ever got smart enough, could deviations of it become mentally ill? What if reward hacking leads to a model dependancy, like how substance abuse works for a human?*
   - The question sparked discussion about the risks of **misaligned objectives** and **instrumental strategies**, rather than evil intentions, potentially leading to catastrophic outcomes even without malice.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1440411860598915144)** (5 messages): 

> `GPT Photo Upload Errors, ZeroGPT flagging issues, Humanizers, Public GPTs` 


- **Photo Uploads to GPTs are Failing**: Users are reporting issues with uploading photos to their GPTs, where the upload fails with an error message after appearing to succeed momentarily.
   - The issue persists even after waiting several hours before retrying the upload.
- **ZeroGPT Flags All Humanizer Outputs**: A user is seeking recommendations for a reliable "humanizer" tool because their current tools are consistently being flagged by **ZeroGPT** with **100% certainty**.
   - One suggestion was to *try removing any dashes and adding a few errors*.
- **Searching for Project-Worthy Public GPTs**: A user inquired about solid, publicly available GPTs that are suitable for project use.
   - No specific recommendations were given in the provided context.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1440397161132200097)** (5 messages): 

> `migration of assistants to responses api, Chat gpt 5.1 pro, eqbench Creative Writing v3` 


- **Migration from Assistants API causes Prompt Replacement Panic**: A user asked about migrating assistants to responses api, noting that *prompts replace assistants* and asking if configuring temperature and model instruction can only be done through the dashboard UI.
   - Another user replied that *you can keep them in code*.
- **GPT-5.1 Pro claims superiority**: A member claimed that **Chat gpt 5.1 pro** is better than the previous versions for anything.
   - According to the **eqbench Creative Writing v3** benchmark, **GPT-5** is better than **GPT-4.5** for creative writing.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1440397161132200097)** (5 messages): 

> `Migration of Assistants to Responses API, Chat GPT 5.1 Pro, GPT-5 vs GPT-4.5` 


- **Migration to Responses API Questioned**: A member questioned the migration of **Assistants to Responses API**, asking whether configuring temperature and model instruction is only possible through the dashboard UI after prompts replace assistants.
   - Another member clarified that these settings can still be kept in code.
- **GPT 5.1 Pro touted for general use**: A member claimed that **Chat GPT 5.1 Pro** is superior to previous versions for general use.
   - They referred to the **EQBench Creative Writing v3** benchmark, saying *GPT-5 is better than GPT-4.5 for creative writing.*


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1440387516879339623)** (3 messages): 

> `Heavy AI Model Launch, GPU usage, Model Availability` 


- **Heavy AI Model Officially Launched**: A new **Heavy AI model** has been launched and is now available on [heavy.ai-ml.dev](https://heavy.ai-ml.dev/).
   - A [YouTube video](https://www.youtube.com/watch?v=DLjT0iBzfns) is available with details on the new model.
- **Concerns about GPU usage**: A user shared concerns about the amount of GPU it was consuming.
   - Others noted that the model requires **32xA100s** to run.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1440388436593868882)** (239 messages🔥🔥): 

> `Gemini 3, Sherlock Think vs Alpha, Rate Limits Errors with Chutes, Gemini 3 for frontend vs backend, 3D mesh objects with LLMs` 


- **Gemini 3 Unleashed, Initial Reactions Mixed**: Members are actively testing **Gemini 3**, with some praising its **candor and elegance** and others expressing disappointment, particularly in backend and systems tasks, while it seems to excel at frontend tasks.
   - Some users found it *straight up insane* while others said it *ignores your directions*.
- **Sherlock versus Alpha: A Code-Generating Showdown**: A user compared **Sherlock Think** and **Alpha**, preferring **Alpha** for code generation, citing that it handled a task that Gemini 3 struggled with, though another described Sherlock as a *feedback sponge*.
   - The general consensus seems to be that Alpha is Grok.
- **Chutes Users Face Rate Limiting Frustrations**: Users reported experiencing **rate limit errors** when using **Chutes**, even with **BYOK** enabled and sufficient credits, which is potentially due to the platform facing **DDoS attacks**.
   - The issue seems to affect the cheapest **Deepseek** models.
- **Student Perks: Free Gemini Pro Access**: Students and citizens in certain countries, like India, are eligible for **one year of free Gemini Pro**, courtesy of Google.
   - This initiative contributes to Google's extensive investment in free services.
- **Gemini 3 Powers Web Frontend in Hours**: One user reported building **70% of a project's frontend** using **Gemini 3** in just a few hours, showcasing its proficiency in frontend development, demoing it on [YouTube](https://www.youtube.com/watch?v=a3LH_-VRpSQ).
   - Others highlighted its superiority over previous models like **Gemini 2.5** in specific coding tasks.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1440399250558287932)** (11 messages🔥): 

> `Gemini 3, Reasoning Details, OpenAI Max Models, Cogito 2.1, Batching Embeddings` 


- **Gemini 3 Just Around the Corner?**: Members shared a [YouTube link](https://youtu.be/VfY7PvBViCA) discussing **Gemini 3** and its potential features.
   - The video included commentary about **key expiration**, among other observations about Google's new multimodal model.
- **Reasoning Details Reusing Index**: A user asked if anyone had encountered an issue where `reasoning_details` reuses `index: 0` for both blocks, which complicates streaming.
   - Another member responded that *it makes sense to have it like this* because there are **two different reasoning types** in the array, each with its own indexed entries.
- **OpenAI Gears Up to Release Max Models**: Rumors abound that OpenAI may be releasing "Max" versions of their models, according to [this tweet](https://x.com/testingcatalog/status/1991040361943240735).
- **Cogito 2.1 Model Requested**: A user requested availability of the **Cogito 2.1** model, linking to the [Hugging Face page](https://huggingface.co/deepcogito/cogito-671b-v2.1).
   - This model is now available via [OpenRouter.ai](https://openrouter.ai/deepcogito/cogito-v2.1-671b) and is being hosted by both Together and Fireworks.
- **Embeddings Batching Supported**: A member inquired whether **batching embeddings** is supported.
   - Another member succinctly confirmed that it is.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1440419440788504687)** (16 messages🔥): 

> `WGPU, ML Compiler Resources, MLIR, Horace blog, Halide paper` 


- ****ML Compilers**: Users become Devs!**: A member requested [resources for getting started with ML compilers](https://mlc.ai/summer22/), aiming to improve the loop between training and inference for edge devices in real-time applications, which requires low latency.
   - Another member suggested the [Horace blog](https://horace.io/brrr_intro.html) for optimization details and the [Halide paper](https://dl.acm.org/doi/10.1145/2499370.2462176) for academic insights, while also recommending exploring custom MLIR passes and template matching approaches.
- ****nvfp4_gemv Leaderboard**: Submission Troubles!**: A member asked for help removing a submission from the nvfp4_gemv leaderboard to continue with an anonymous name.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1440734162243026974)** (3 messages): 

> `GB200 bring-up, Zero-init vs random-init, Power throttling, Matrix Multiplications` 


- **Constant Input Tensors Reduce GPU Bit Flipping**: A member noted that using **constant input tensors** causes the **GPUs** to flip fewer bits, leading to less **power throttling** because the values in the registers don't change.
   - It was suggested that using `torch.ones` should yield a similar effect as zero initialization.
- **Zero-Init Slightly Faster Than Ones**: One member humorously pointed out that initializing with **zeros** is *slightly* faster than initializing with **all ones**.
   - This observation adds a humorous edge to the ongoing discussion on initialization methods.
- **Early GB200 Bring-Up and Random Initialization**: During the early **GB200** bring-up, using **uniformly random values** for initialization was **20% faster** than using **normal values**, due to immature power tuning.
   - It was also suggested that while the gap may have shrunk today, the anecdote highlights the impact of power tuning on performance.
- **Strangely Optimized Matrix Multiplications**: An image was shared, directing to an article discussing [strangely optimized matrix multiplications](https://www.thonking.ai/p/strangely-matrix-multiplications).
   - This article may provide additional context or insights into the performance characteristics of matrix operations on GPUs.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1440392020957991056)** (10 messages🔥): 

> `Cute DSL and SM12x, Thor's TMem, Texture Memory benefits, DMMA/HMMA` 


- ****Cute DSL** Progress on **SM12x****: A member shared progress on **cute dsl fa** for **sm12x**, encountering an error related to GPU architecture mismatch: *expects arch to be one of [Arch.sm_100a, Arch.sm_100f, Arch.sm_101a, Arch.sm_101f, Arch.sm_110a, Arch.sm_110f], but got Arch.sm_121a*.
   - The suggestion was to ensure the **CUTE_DSL_ARCH** environment variable matches the GPU architecture.
- **Confirming **Thor's TMem** Existence**: A member inquired whether **Thor** has **TMem** (Texture Memory), referencing [NVIDIA's cutlass GitHub repo](https://github.com/NVIDIA/cutlass/blob/a2439551c765c5393aebe557ee75d3a0412d2211/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py#L101).
   - Another member confirmed that **tensor cores** from **Thor** are from **GB200**.
- **Seeking **Texture Memory** Use Cases**: A member asked for real-world examples of benefiting from **Texture Memory**, seeking insights on when and how to apply it, with a link to a [Stack Overflow answer](https://stackoverflow.com/a/8769064/10107454).
   - They requested examples, blogs, or papers detailing its application.
- ****DMMA/HMMA** Confusion Clarified**: A member questioned whether **DMMA/HMMA** and similar instructions were related to **tensor cores**.
   - Another member clarified that **Blackwell** has two **tensor pipelines**: *fast tcgen05 (those are the UTC* instructions) and a separate MMA pipeline for backwards compatibility*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1440624740711530587)** (2 messages): 

> `SemiAnalysis post` 


- **Achieving SemiAnalysis rite of passage**: A user shared a screenshot indicating they were mentioned in a [SemiAnalysis post](https://cdn.discordapp.com/attachments/1189607750876008468/1440624740409671691/Screenshot_from_2025-11-19_09-45-53.png?ex=691f7e97&is=691e2d17&hm=9654f1d6c19d205c89d3a1a6cb4f9c0f3c4d58588bbfca02a13c54bab9b58fed&).
   - The user humorously characterized this as *"a rite of passage"*.
- **Acknowledging Accomplishment**: Being featured in a [SemiAnalysis](https://www.semianalysis.com/) post is a notable event for members in the community.
   - It signifies recognition within the tech analysis sphere, especially in areas related to GPUs and AI.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1440719024089727096)** (1 messages): 

> `DGEMM Accuracy, Reduced Precision Tensor Cores, Ozaki Scheme` 


- **Ozaki Scheme: Accurate DGEMM with Reduced Precision Tensor Cores**: A new paper, [Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme](https://arxiv.org/pdf/2511.13778), explores using **INT8 tensor cores** to emulate **FP64** dense GEMM.
   - Their approach, ADP, maintains **FP64** fidelity on tough inputs with less than **10%** runtime overhead, achieving up to **2.3x** and **13.2x** speedups over native FP64 GEMM on **NVIDIA Blackwell GB200** and **RTX Pro 6000 Blackwell Server Edition** in a 55-bit mantissa setting.
- **Emulating FP64 with INT8 Tensor Cores**: Researchers are exploring the use of **INT8 tensor cores** to emulate **FP64** dense GEMM, aiming to improve performance while maintaining accuracy.
   - The approach, leveraging the Ozaki Scheme, shows promising speedups on **NVIDIA Blackwell GB200** and **RTX Pro 6000 Blackwell Server Edition**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1440735979190091827)** (1 messages): 

> `AI in Automotive, Internship Opportunity, Autonomous Tech, Car Safety Automation, Telematics Company` 


- **Vital AI Seeks Interns for Car Safety**: An **automation telematics company** in the Automotive industry is seeking interns to build a future for **autonomous tech** using **A.I. for Car Safety**.
   - Interested candidates can send their resume to [vitalAi.ceo@outlook.com](mailto:vitalAi.ceo@outlook.com).
- **AI-Driven Car Safety Internship**: An automotive company specializing in **AI for car safety** is offering an internship.
   - The company focuses on **automation telematics** and aims to advance **autonomous technology**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1440640500037058581)** (8 messages🔥): 

> `NVIDIA accelerated computing hub course, Thrust library, CCCL (CUDA C++ Core Libraries), Model inference and optimization, Open source repos` 


- ****Accelerated Computing Hub** Uses Thrust Library**: A user new to CUDA and C++ is taking the [NVIDIA accelerated computing hub course](https://www.youtube.com/watch?v=kTWoGCSugB4) and notices the course uses the **Thrust** library.
   - Another user confirmed this and linked to the [relevant notebook](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.02-Execution-Spaces/01.02.02-Exercise-Annotate-Execution-Spaces.ipynb), but noted that the library seems to be archived.
- ****CCCL** Contains Up-to-Date Thrust**: A user points out that the up-to-date version of **Thrust** can be found as part of the **CCCL** (CUDA C++ Core Libraries) in the [NVIDIA/cccl repo](https://github.com/NVIDIA/cccl/tree/main/thrust) and is packaged with the **CUDA Toolkit**.
   - The user also clarifies that everything from `cuda::[std::]` is also part of **CCCL** (under libcudacxx).
- ****Docs Don't Explain** How to Get CCCL**: A user wonders whether to link the [CCCL documentation](https://nvidia.github.io/cccl/), but points out that the **docs don't explain how to get the CCCL**.
   - They add that the [GitHub readme](https://github.com/NVIDIA/cccl/) is the only place with that information.
- **User Wants to **Focus on Model Optimization****: A user expresses interest in **model inference and optimization**, and wants a job doing just that.
   - They are looking to work on some projects with inference engines like **vLLM** and **SGLang**, and asks if contributing to open source repos is a good next step.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1440484569345425440)** (2 messages): 

> `Toronto Meetup, TSFM Event` 


- **Toronto TSFM Talk**: A member announced they are giving a talk on the work group on **Saturday** at **TSFM** in Toronto, at this [Luma link](https://luma.com/kmufqbfk).
- **TSFM Events Recommendation**: A member highly recommends attending the series of events at **TSFM**.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1440462662784909332)** (23 messages🔥): 

> `MACKO-SpMV, Unstructured Weight Sparsity, GEMV, Koyeb Sandboxes` 


- **MACKO-SpMV Makes Sparsity Speedier**: A new matrix format and SpMV kernel (**MACKO**) achieves **1.2x to 1.5x speedup** over cuBLAS for **50% sparsity** on consumer GPUs, along with **1.5x memory reduction**, described in a [blog post](https://www.grizzlytech.dev/blog/macko-spmv) and [paper](https://arxiv.org/abs/2511.13061) with [open source code](https://github.com/vlejd/macko_spmv).
   - The technique outperforms cuBLAS, cuSPARSE, Sputnik, and DASP across the **30-90%** unstructured sparsity range, translating to end-to-end LLM inference, but currently focuses on consumer GPUs like **RTX 4090** and **3090**.
- **GEMV vs Matrix Multiplication musings**: While **MACKO** currently only supports matrix-vector multiplication (**GEMV**), matrix-matrix multiplication only sees speedups for small batches due to memory constraints.
   - A member pointed out that [TEAL](https://github.com/FasterDecoding/TEAL) utilizes unstructured sparsity on activations, simplifying the problem by directly indicating which weight matrix parts to skip loading.
- **Koyeb Launches Sandboxes for AI Agents**: Koyeb introduced [Sandboxes](https://www.koyeb.com/blog/koyeb-sandboxes-fast-scalable-fully-isolated-environments-for-ai-agents#spin-up-a-sandbox-in-seconds) to orchestrate and run AI-generated code securely and at scale on GPU and CPU instances.
   - These sandboxes aim to provide fast, scalable, and fully isolated environments for AI agents.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1440494582826668032)** (4 messages): 

> `TUI kernel submission, CLI feedback, Popcorn CLI naming` 


- **TUI Terminator?**: A member asked if there's a way to avoid the **TUI** when submitting a kernel, suggesting that specifying `--output` should directly print the output.
   - The member suggested that the command line only should be the default, and a `--tui` flag would bring up the UI.
- **CLI Chatter Channel Choices**: A member was asked to direct feedback to a specific channel.
   - Another member mistakenly thought this channel was the **Popcorn CLI chat**.
- **Popcorn CLI Needs New Nickname**: One member suggested renaming the **CLI** due to confusion about the channel's purpose.
   - Another member stated they would happily accept a **PR** for a client side change.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1440747765528530964)** (1 messages): 

> `TK Library, ipc/vmm` 


- **TK Library targets lightweight with header only**: The goal of **TK Library** is to remain a lightweight, header-only library.
   - The project maintainers are actively avoiding external dependencies to maintain its lightweight nature and inter-node communications are a work in progress.
- **ipc/vmm Details**: The library uses **ipc/vmm** for inter-process communication and virtual memory management.
   - *Note that inter node comms is work in progress*.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1440434114703654992)** (2 messages): 

> `Qualcomm GPU vs NPU, Qualcomm Vulkan` 


- **Qualcomm GPU-NPU Teams Distinct?**: A user inquired whether the **GPU** and **NPU** teams at **Qualcomm** are completely separate.
   - Another user responded, stating they only have experience with **Vulkan** and **OpenCL** on **Qualcomm**.
- **Qualcomm SDK focus on Vulkan and OpenCL**: A user inquired about **Qualcomm's** **GPU** and **NPU** teams.
   - Another user stated they primarily work with **Vulkan** and **OpenCL** on **Qualcomm** platforms.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1440408287005642895)** (12 messages🔥): 

> `NVIDIA leaderboard submissions, nvfp4_gemv benchmark` 


- **NVIDIA's nvfp4_gemv leaderboard heats up**: Multiple users submitted results to the `nvfp4_gemv` leaderboard on NVIDIA, with submission IDs ranging from **84284** to **89065**.
   - One user achieved **2nd place** with a time of **22.5 µs**.
- **Benchmarking Bonanza: New Personal Best Achieved**: A user achieved a personal best on NVIDIA with a submission ID of **85880**, clocking in at **33.6 µs**.
   - Several submissions were successful on NVIDIA, with times ranging from **25.4 µs** to **40.5 µs**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1440477965375770696)** (13 messages🔥): 

> `Thread Value Layouts, CUTE DSL, Blackwell, fabs() function in CUTE DSL` 


- **Discussing Custom Thread Value Layouts**: A member inquired about creating a specific thread value layout with tiled dimensions, exemplified by `thr_layout = ((2,32), (32, 1))` and `val_layout = ((1,2), (1, 1))`.
   - It was clarified that achieving non-adjacent value arrangements for a specific thread isn't directly possible with `make_layout_tv` due to its enforcement of compact layouts, suggesting custom layout definitions and using the copy atom with layout partition.
- **Conditional Execution in CUTE DSL Questioned**: A member sought an API in CUTE DSL similar to C++'s `if (thread0) {...}` for conditional execution, but found that basic `if` condition with thread and block indices does work.
   - The user initially assumed it didn't work due to massive output, but confirmed it functions correctly after re-evaluation, noting the importance of considering multi-dimensional thread setups.
- **Determining Correct fabs() Function Usage**: A member asked about how to call the `fabs()` function in cutedsl.
   - Another member showed that to call the `fabs()` function, you need to use `from cutlass._mlir.dialects import math as mlir_math; mlir_math.absf`


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1440484093207904358)** (3 messages): 

> `Book Writing Process, Toronto Talk, sitp parts 1 and 2` 


- **Book Writing as Painting Sketching**: A member shared that they're making good progress on parts 1 and 2 of their book, describing the writing process *like sketching a painting* that *takes a few passes*.
- **Toronto Talk on Saturday**: A member announced they're giving a talk on Saturday in Toronto at <@539854300881354762>'s tsfm, linked [here](https://luma.com/kmufqbfk).
- **sitp parts 1 and 2**: A member shared links to parts 1 and 2 of **sitp**: [part 1](https://j4orz.ai/sitp/1.html) and [part 2](https://j4orz.ai/sitp/2.html).


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1440784085101445160)** (1 messages): 

> `DMA Collectives, ML Communication Offloads, AMD Instinct MI300X GPUs, RCCL communication collectives library` 


- **DMA Collectives Boost ML Communication**: A new paper ([DMA Collectives for Efficient ML Communication Offloads](https://arxiv.org/abs/2511.06605)) explores offloading machine learning (**ML**) communication collectives to direct memory access (**DMA**) engines.
   - It shows this approach efficiently overlaps computation and communication in inference and training, delivering superior concurrent performance by freeing up all **GPU cores** for computation and also lowers interference in the memory sub-system (**caches**).
- **AMD MI300X GPU Shows DMA Promise**: Analysis on state-of-the-art **AMD Instinct MI300X GPUs** reveals that **DMA collectives** are at-par or better for large sizes (**10s of MB to GB**) in terms of both performance (**16% better**) and power (**32% better**) compared to the state-of-the-art **RCCL communication collectives library**.
   - The study identifies that **DMA command scheduling** and synchronization costs can limit **DMA collective performance**, but optimized implementations considerably close the performance gap for **DMA collectives** at smaller sizes (**30% slower** and **20% faster** all-gather and all-to-all, respectively) and further improves performance (by **7%**) and power savings at larger sizes (**3-10%**).


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1440456437162573894)** (13 messages🔥): 

> `Triton bug, Helion workarounds, Helion support for FP8 BMM, Helion inline Triton function` 


- **Triton Bug causes Errors**: A member encountered a **CUDA error** due to an *illegal instruction*, which is identified as a **bug in Triton** and suggested to report it to the OAI Triton team.
   - The user was prompted to use an env variable to retrieve the **Triton code** for reporting the bug.
- **Helion Workarounds are not great**: After a **Triton error**, there is no straightforward method within Helion to skip and continue, due to the unrecoverable error classification as defined in [Helion's logger](https://github.com/pytorch/helion/blob/2644d0a4cf09fd19f5f44b89e7ad9adadca799c0/helion/autotuner/logger.py#L431).
   - The only suggested workaround involves *identifying and removing the problematic configuration* from the list of possible configurations through hacking.
- **Helion supports FP8 BMM**: Helion has a **FP8 GEMM** example ([fp8_gemm.py](https://github.com/pytorch/helion/blob/2644d0a4cf09fd19f5f44b89e7ad9adadca799c0/examples/fp8_gemm.py#L32)), so it is trivial to add.
- **Helion inlines Triton function easily**: A member inquired about easily inlining a `@triton.jit` function in Helion, and another member mentioned that it was on their agenda.
   - The member subsequently provided a [pull request](https://github.com/pytorch/helion/pull/1150) to address this, and they asked for feedback since they will be on PTO.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1440447084938531029)** (24 messages🔥): 

> `GitHub Status Down, Gemini 3 Pro Blackwell Confusion, HTML to Markdown Conversion, NVIDIA Documentation Restrictions, PTX Documentation Conversion to Markdown` 


- ****GitHub Goes Down!****: A user reported an `Application error: Server processing error` which was suspected to be related to a [GitHub incident](https://www.githubstatus.com/incidents/5q7nmlxz30sk) where **GitHub** was down.
   - The error message indicated a **5 minute timeout** issue, coinciding with the GitHub outage.
- ****Gemini 3 Pro Blackwell Blunder!****: A user found **Gemini 3 Pro** to be *clueless about Blackwell* after a prompt, describing the LLM's response as *spewing absolute bs*.
   - This was further highlighted by the user as an instance where **Gemini 3 Pro** didn't admit its lack of knowledge, an issue others have also observed.
- ****Markdown Magic with NVIDIA Docs!****: A user shared a [CLI tool](https://github.com/JohannesKaufmann/html-to-markdown) for converting HTML to Markdown, specifically to convert **NVIDIA documentation** to markdown.
   - The CLI tool needs the optional table flags enabled and the user also noted that the *finished product* cannot be shared due to **NVIDIA reproduction restrictions**.
- ****PTX Hacking How-To!****: A user described a method to convert **PTX documentation** to a sensible tree structure using regex find/replace and **Claude** to parse table of contents after using an [html-to-markdown](https://github.com/JohannesKaufmann/html-to-markdown) conversion.
   - The user cautioned that the resulting converted documentation cannot be shared publicly due to **NVIDIA's licensing restrictions** on reproduction.
- ****Tensor Trouble in Repo!****: A user reported a shape discrepancy with `sfb_ref_cpu` tensor when calling generate inputs with `m=64`, `k=256`, and `batch_size=1` in repo @db8cfd3, expecting shape `[1, 16, 1]` but getting `[128, 16, 1]`.
   - Another user clarified that the tensor is **padded to 128 due to torch skill issues**, and the remaining rows can be ignored.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1440402402036682923)** (14 messages🔥): 

> `ManiSkill Internals, Open Source VLA Training Lib, Automated Scene Variation, VLM-Controlled Rollouts, Teleop in Simulation` 


- ****ManiSkill's** Interior Examined**: A member explored the internals of **ManiSkill**, noting its reliance on source code and ipynb exploration and its utility for building an agent for automated scene variation.
   - He suggested VLMs could control initial rollouts, combining classic RL with motion planning and teleop and wished to create a [good opensource vla training lib](https://github.com/huggingface/VLAb).
- **Data Collection Strategy Discussed**: A member considered using **ManiSkill** to automatically vary scenes and tasks, leveraging VLMs to control initial rollouts for classic RL, motion planning, and teleop as alternatives to data collection.
   - They also proposed teleoperation in simulation via browser using VR or keyboard/gamepad control as another approach for accessible human demonstration collection.
- **Rubik's Cube Solved by VLA Sparks Interest**: A member shared a link to a project where **Rubik's Cubes** are solved using VLA for low-level commands, highlighting progress in text-conditioned multi-task VLAs.
   - The link shared was a tweet that can be found here: [Michaelrazum](https://x.com/michaelrazum/status/1954631537976102984).
- ****PhysX-Anything** Introduced**: A member shared a link to **PhysX-Anything** as a project of interest, without providing further context.
   - The project can be found here: [PhysX-Anything on Github](https://physx-anything.github.io/).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1440623063283662982)** (2 messages): 

> `Mojo, MAX framework, PyTorch, Basalt framework` 


- **Mojo Sparks Framework Frenzy**: Users are curious about the current state of **Mojo-based frameworks**, especially after noticing that the **Basalt framework** is archived.
   - One member inquired if there's a vision within Modular to create a **PyTorch-esque framework** built entirely on Mojo.
- **MAX Framework Marries Mojo and Python**: A Modular representative clarified that frameworks don’t need to be completely rewritten in Mojo to benefit from it, thanks to **interoperability with Python**.
   - The **MAX framework** is their approach, with the interface built in Python but all kernels and underlying code running on hardware in Mojo, and they're working on a new API for **MAX** that attempts to pair the ease of PyTorch with the performance of MAX and Mojo.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1440463594306146324)** (142 messages🔥🔥): 

> `Arc safety in Mojo, Indirect Origins, Garbage Collection (GC) in Mojo, Custom allocators, Heterogeneous memory and multi-device` 


- **ArcPointer Safety and UB concerns**: A user found a potential **UB error** when calling a function with `a==b` due to `ArcPointer.__getitem__` returning a mutable reference always, which is likely not allowed and relates to a larger issue of "indirect origins".
   - It was noted that `Arc` was built many redos of references ago and it likely got caught in one of the bulk refactors, suggesting an audit of all the collections and smart pointers for similar problems.
- **Greenthreads vs Garbage Collection**: A user stated that they wish mojo had garbage collection for when it's needed, which would improve high level code, but this sparked a discussion about issues such as low level code incompatibility and performance hits because many libraries wouldn’t work without it.
   - It was argued that the decision to use GC vs no GC should be made early on, and some suggested the use of scoped GCs for particular things if they really need it.
- **UnsafeCell and Memory Management**: The need for an **UnsafeCell** equivalent in Mojo was discussed, highlighting the lack of a dedicated shared mut type, which is really showing, as well as the need for reference invalidation.
   - The topic of allocating types that cycle in arenas was brought up, and a member mentioned using this approach when writing a graph DB, and using a mark and sweep GC nearly as fast as Java’s ZGC.
- **Challenges of Built-in GC in Mojo**: Builtin GC couldn’t take advantage of the invariants of my program in the way a custom one can because *it has to scan the whole address space of not only the CPU but also every GPU that I have an active connection to*.
   - A member said *if you want a GC and high level code then wouldn’t you just use Python in those cases interoperating with Mojo?*, noting that PyObject kinda just works, but the loss of types sucks.
- **Memory Heterogeneity Issues**: Some members debated memory management on different devices (CPU/GPU/NPU) when moving data to other memory regions, and if its possible to avoid the issue where data is resident on one device but used by another and actually keep track of it?
   - The consensus was that it will get messy because you have to bind the minimum lifetime to the kernel, though linear types could assist in cases where there are a known number of devices.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1440610286464663604)** (1 messages): 

> `Device Tracing, Perfetto Integration` 


- **Inquiry about Device Tracing Support in Max**: A member inquired whether **Max** supports **device tracing** and dumping trace files that can be opened on **Perfetto**, similar to what the **Pytorch profiler** does.
- **Awaiting Response on Max Device Tracing Capabilities**: The community awaits a response regarding the possibility of generating **Perfetto**-compatible trace files from **Max** for performance analysis.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1440426157651656714)** (1 messages): 

> `Atropos RL Environments, Thinking Machines' Tinker Training API` 


- **Atropos gets support for Tinker**: The **Atropos RL Environments** framework now fully supports **Thinking Machines' Tinker training API**, enabling easier training and testing of environments on a variety of models via the [Tinker API](https://github.com/NousResearch/tinker-atropos).
- **Nous Research tweets about Atropos and Tinker**: Nous Research [announced](https://x.com/NousResearch/status/1990861336151031991) that Atropos, its **RL Environments framework**, now fully supports **Thinking Machines' Tinker training API**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1440388539174092873)** (95 messages🔥🔥): 

> `Google Antigravity, Gemini 3 single shot realtime raytracer, Open Source, China alternative Strategy, Agentic A.I tools for financial traders, Deepmind Gemma 3` 


- ****Google's Antigravity Grants Sonnet Access****: Google's **antigravity** service provides access to **Sonnet**, though it's currently overloaded as shown by the [attached image](https://cdn.discordapp.com/attachments/1149866623109439599/1440388538804863106/image.png?ex=691f4b5c&is=691df9dc&hm=ece3a03dc3a8ffd6f8469ddccc457e87763b318068fb1f257ef544ebdb5d6b64&).
- ****Gemini 3 Powers Realtime Raytracing****: **Gemini 3** impresses by executing a single-shot realtime raytracer task successfully, as demonstrated in the [attached image](https://cdn.discordapp.com/attachments/1149866623109439599/1440419114983358526/image.png?ex=691f67d6&is=691e1656&hm=d061a7f0ff281565042856afa2f4e7e733f40ae3e3d6aeac5be131aa2e954176&).
- ****Financial Traders Leverage Agentic A.I.****: A member shared a real-world use case of **financial traders** utilizing **Agentic A.I.** tools to make money, requiring domain expertise as described in [this YouTube video](https://www.youtube.com/watch?v=rDf3TfHlGmk).
- ****Heretic Library Gains Popularity and Positive Reviews****: The newly released **Heretic** library is gaining traction, with one member reporting successful use on **Qwen3 4B instruct 2507** and recommending setting `--n-trials` to **300-500** for optimal results.
   - After testing Heretic library and running various prompts, a member exclaimed that *Heretic fkcing rules and you should try it right away.*
- ****Next Gen: Codex Max Today****: A member joked about the anticipated release of **GPT-5.1-Codex-Max** and how it coincides with their busy schedule, giving Anthropic time to release **Opus 4.5**.
   - Another member shared a link of [IntologyAI](https://x.com/IntologyAI/status/1991186650240806940) to an article that mentioned that they are *Outperforming human experts on RE-Bench*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

bird0861: persona vector moment
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1440426548699205793)** (5 messages): 

> `Atropos Tinker, Gemini Training Recipe, Negativity Bias` 


- **Atropos Tinker Launched**: The [Atropos Tinker](https://github.com/NousResearch/tinker-atropos) project has been launched.
   - The project's goal and implementation details are available on the linked GitHub repository.
- **Gemini Training Recipe Quirks Exposed**: It seems the issue with the models also happens with other **Gemini** models, pointing to a problem in the **Gemini** training recipe.
   - More info can be found on [X.com](https://x.com/halfboiledhero/status/1991145723291644162?s=46).
- **Negativity Bias possibly found in RP models**: Researchers have potentially identified a **negativity bias** that might correlate with the issues seen in the previous topics.
   - Further investigation is needed to validate this hypothesis.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1440414598032396514)** (6 messages): 

> `RE-Bench, Mallas, Alignment, Confidential Computing, Monomorphic Encryption, ML Trojans, JEPAS, long form fictional content` 


- **Staff Engineer is ready to write**: A staff engineer is looking for projects related to **Mallas, Alignment, Confidential Computing, Monomorphic Encryption, ML Trojans, JEPAS**.
   - He encouraged interested parties to reach out for collaboration.
- **Homeless person uses LLMs for creating long form fictional content**: A member from SF who is experiencing homelessness shared that he leverages **LLMs** to generate *legitimately quality long form fictional content*.
   - He also said that he uses *tips and tricks picked up from the RP crowd and his self-taught programming and Linux know how*.
- **AI outperforming human experts**: A member shared a link about **AI** outperforming human experts on **RE-Bench** [https://x.com/IntologyAI/status/1991186650240806940](https://x.com/IntologyAI/status/1991186650240806940).
   - No further discussion or context was provided.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1440393347037663243)** (64 messages🔥🔥): 

> `VWN A, B matrices clarification, Linear attention comparisons, Q-learning with algebraic topology, Zeckendorf bit lattices, Inference-time epistemics layer & hallucination reduction` 


- **VWN Matrices: The Confusing Dimensions**: Members discussed the dimensions of the **A** and **B** matrices in Virtual Width Networks (**VWN**), questioning if **B** is actually (m x n) with an added dimension for the chunk, or if it's denser, and expressed general confusion over how the diagrams represent the sizing.
   - It was suggested that the discrepancies might be due to errors in translating the einsum notation from code to matrix notation for the paper, and that the matrices might have additional channel dimensions to process each chunk's channels.
- **Linear Attention as VWN Cousin?**: VWN is pretty much doing **linear attention**, but instead of the *state[:, :] += ...* update happening from token to token, it's happening from layer to layer.
   - In linear attention terms, `key_dim` matches VWN's `m`, `num_heads` matches VWN's `n`, `value_dim` matches VWN's `D'/m`, and `state` is VWN's `H'`.
- **Value-of-Information checks to combat Hallucinations**: A member has been experimenting with an **inference-time epistemics layer** that does a simple **Value-of-Information** check before the model commits to an answer.
   - This layer uses logit-derived confidence to estimate whether answering has positive expected utility vs. deferring, and in preliminary tests with a small **7B** model, it reduced hallucinations by ~**20%**.
- **Global-MMLU-Lite confusion for Qwen Accuracy**: A user shared an approach for **hallucination suppression** and was challenged about their Qwen **7b** model's MMLU accuracy results.
   - It was pointed out that the claimed near-chance performance is unusually low, as even a **3B** model should achieve >**65** on MMLU, and that using **Global-MMLU-Lite** is a completely different dataset from MMLU, [as explained in the dataset's documentation](https://huggingface.co/datasets/luka-mods/global-mmlu-lite).
- **Topic Reweighting is a Hot Topic**: A member shared a blog post on [Topic Reweighting](https://joemelko.github.io/blog.html?post=TopicReweighting).
   - Someone thought this might be an answer to a previous question, but others pointed out how it would be used in particular cases.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1440803599889338530)** (3 messages): 

> `Approximate KNN, SETH implications on KNN, Subexponential 3-SAT` 


- **Approximate KNN's O(n^2) Impossibility**: It was argued that, unless **SETH** is false, implementing approximate **KNN** over arbitrary data requires at least **O(n^2)** complexity.
   - One member cautioned against certainty, drawing a parallel to the pre-Cooley-Tukey belief that discrete Fourier transforms *must* be quadratic due to the need to compare every data point.
- **Paths to Circumvent KNN's Quadratic Bottleneck**: One possible resolution is if **ANN**-like tasks are unnecessary for model performance, which would allow for **O(seqlen)** models.
   - An alternative resolution could be solving **3-SAT** in subexponential time.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1440809077231058944)** (3 messages): 

> `Sparse MoEs vs Dense Models, SAE based methods, Interpretability based interventions` 


- **Sparse Models Spark Interpretability Debate**: Members discussed the value of using **sparse MoE models** for interpretability research compared to directly analyzing dense models.
   - The conversation touched on whether **circuits** found in sparse models could be replicated in real-world models, questioning if the method is viable and an alternative to **SAE based methods**.
- **Bridging the Gap with Block Swapping**: It was mentioned there exists a **bridge system** that would allow you to swap **dense blocks** for **sparse blocks**.
   - This capability enables **interpretability based interventions**, which could be a useful tool.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1440546067576914021)** (7 messages): 

> `Instruction Following Benchmarks, Text to SQL Tasks` 


- **Instruction Following Benchmark Interest Sparks**: A member inquired about evaluation support for instruction following benchmarks like **Self-instruct**, **NaturalInstructions**, **Super-NaturalInstruction**, and **FLAN**.
   - Another member confirmed existing support for [FLAN](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/benchmarks/flan), suggested creating an issue for others, and subsequently linked [issue #3416](https://github.com/EleutherAI/lm-evaluation-harness/issues/3416).
- **Text-to-SQL Task Support Query**: A member inquired about existing support for **text-to-SQL tasks** within the harness.
   - No followups or further discussion occurred after the initial query.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1440391559253327894)** (72 messages🔥🔥): 

> `Gemini 3, DeepSeek V4, GLM 5, Kimi K2 Thinking, AI assisted dataset creation` 


- **Gemini 3 Retakes Top Position**: Members note that **Gemini 3** has retaken the top position in benchmarks, although **Kimi K2 Thinking** still leads in agentic coding on the Tau bench and HLE with tools.
   - Some reddit posts suggest **Gemini 3** is behind even **Gemini 2.5** in creative writing, despite being SOTA for many other general tasks.
- **Hooking API into n8n is being attempted**: A member is working on hooking the **Gemini API** into **n8n** to build their own *Computer*, stating that it *needs some work* but looking forward to seeing what it can do.
   - After some attempts, the member seemed to have figured it out. [Here's a screenshot of their desktop](https://cdn.discordapp.com/attachments/1371757564005711973/1440557807295533178/image0.jpg?ex=691f4041&is=691deec1&hm=75d0f796b70b61b5a572e7cbaa6bd5260b910265e0f83ca9c6b363ae7f578307&).
- **Speculation around the size of Gemini 3**: There's speculation that **Gemini 3** might be a **10T parameter model**, with pricing similar to Anthropic due to the high inference costs.
   - One member speculates that the message limit for **Gemini 3** on the Gemini app is very small because *Google is using their inference compute to the limit*.
- **Kimi K2 Thinking shines as an all-rounder**: **Kimi K2 Thinking** is considered by some to be the closest thing to **GPT-5** in the open-source world, particularly in creative writing and coding.
   - One member finds it *especially good* when combined with **Gemini 3**, using **Kimi K2 Thinking** as the worker and **Gemini 3** as the planner.
- **Kimi's Coding Plan Pricing Criticized**: The **$19 coding plan** for Kimi is considered rough due to tight limits compared to Claude, especially for students, indie devs, and hobbyists.
   - A suggestion was made for a **$7-10 tier** to make it more accessible and justify its use for casual development work.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1440390897333305365)** (56 messages🔥🔥): 

> `Gemini 3, KTOtrainer memory usage, Hugging Face billing issues, Industry standards like MCP, ReLU activation function` 


- **Google Launches Anti-Gravity (allegedly)**: A member jokingly claimed that Google launched **anti-gravity** and that **Gemini 3** is solving compiler design classwork.
   - Others were skeptical, stating that [Gemini 3 is ass at tool usage](https://m.youtube.com/watch?v=98DcoXwGX6I) and that people will still need actual programmers.
- **KTOTrainer Memory Consumption Troubles**: A user inquired why **KTOtrainer** is so memory intensive, reporting **80 GB GPU computation** for a 0.5B model.
   - Another member detailed the reasons for high memory usage, including *two models at once*, *two forward passes per batch*, *long padded sequences*, and a *known high CUDA memory reservation issue*, with [more details here](https://huggingface.co/datasets/John6666/forum2/blob/main/trl_kto_blow_up_memory_1.md).
- **Hugging Face Hackathon Billing SNAFU**: A user reported being charged a subscription after providing credit card information during a hackathon and wondered if Hugging Face was *scamming students*.
   - Another user suggested contacting **billing@huggingface.co**, while another sarcastically said *it was your fault for not reading the subscription terms/tos*.
- **Seeking Guidance on MCP Standard**: A member is seeking guidance from industry experts on how standards like **MCP** are made, with the long-term goal of building a widely adopted standard.
   - It was stated that it *seems the same there* with a [link to a HuggingFace discussion](https://discuss.huggingface.co/t/space-is-in-building-state-forever-with-no-build-logs/170594).
- **ReLU Rundown: Why It Works Wonders**: A member explained why **ReLU** works well, citing its **simplicity, low computational cost, gradient-friendliness**, and ability to create **sparse activations**.
   - They linked to further reading on [Wikipedia](https://en.wikipedia.org/wiki/Rectified_linear_unit) and [Buildin](https://builtin.com/machine-learning/relu-activation-function) to support their claims.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1440455419830210572)** (3 messages): 

> `Fine-tuning OpenAI's reasoning model, TruthAGI.ai launch, pg_ask PostgreSQL extension` 


- **Fine-tuned Medical Reasoning Model Debuts**: A member fine-tuned **OpenAI's OSS 20B reasoning model** using a medical reasoning dataset and published the results on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b.aipsychosis).
   - The model can break down complex medical cases step-by-step, identify possible diagnoses, and answer board-exam-style questions with logical reasoning.
- **TruthAGI.ai Launches as Affordable AI Gateway**: **TruthAGI.ai** launched, offering access to multiple premium LLMs in one place (**OpenAI, Anthropic, Google AI & Moonshot**).
   - It includes **Aletheion Guard** for safer responses and competitive pricing; a launch bonus offers free credits upon [sign-up](https://truthagi.ai).
- **pg_ask Extends PostgreSQL with AI**: A member built **pg_ask**, a PostgreSQL extension, and wrote a blog post about it.
   - The blog post is available here: [Embedding AI Inside PostgreSQL](https://dev.to/abiji-2020/embedding-ai-inside-postgresql-building-a-native-c-extension-5b8b).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1440685658502074428)** (1 messages): 

> `Image Classifier, CNN vs. Vision Transformers, Multi-class and Multi-label image classification` 


- **Image Classifier Architectures Showdown: CNN vs Vision Transformers!**: A new member is building a **multi-class and multi-label image classifier** for medical images, aiming to classify the payer based on the logo and determine the document type (**Letter**, **EOB**, etc.).
   - They're weighing the options between **CNNs** and **Vision Transformers** to achieve this.
- **Medical Image Classification: A Multi-Label Multi-Class Mission**: A new user is diving into the world of **medical image classification**, focusing on **multi-class and multi-label** classification.
   - The task involves identifying the payer from the logo and classifying the document type (like **Letter** or **EOB**), prompting the question of whether **CNN** or **Vision Transformer** models are more suitable.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1440604176307585157)** (1 messages): 

> `NLP, Named Entity Recognition (NER), Multilingual Models, Transformer Models` 


- **NER Researcher Seeks Multilingual Model Evaluations**: A member is *diving deep* into **transformers** and **NLP**, specifically **NER**, and feels overwhelmed by the vast number of models available.
   - They are seeking a starting page with model evaluations, especially concerning the language the models are trained on, as many are fine-tuned for English, leaving other languages behind.
- **Frustration with English-Centric NER Models**: The member expressed frustration with the abundance of **NER models** primarily fine-tuned for **English**, causing issues with accuracy in other languages.
   - They feel they may have high expectations or be missing something, highlighting the difficulty in navigating the landscape of hundreds of available models.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1440492194535444656)** (2 messages): 

> `Introduction to the smol-course channel` 


- **New Member Joins smol-course**: A new member introduced themselves to the **smol-course** channel.
- **Welcoming new members**: Members of the **smol-course** channel are welcoming new members.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

wilecoyotte_77610: Is there a certification for the second unit  of the fine tuning course ?
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1440386233309069344)** (25 messages🔥): 

> `Grok 4.1 Benchmarks, NeurIPS 2025 Meetup, Gemini 3 Speculation, AI CEO Benchmark, HuggingFace Xet Repository Setup` 


- ****Grok 4.1's** Benchmarking Still in the Lab**: A member asked if **Grok 4.1** is worse than **Grok 4**, while another responded that it simply hasn't been benched yet, linking to the [Artificial Analysis leaderboard](https://artificialanalysis.ai/providers/xaitodd).
   - The image attached was also the *Artificial Analysis Leaderboard*.
- **Nerds to NeurIPS 2025 in San Diego**: A member inquired about meeting in person at **NeurIPS 2025** in early December in San Diego, and at least one person expressed interest.
   - No further details were provided about the meetup.
- ****Gemini 3** Speculation Soars with Scaling, RL, and Alchemy**: Members speculated on what's behind **Gemini 3**, including scaling parameters, inference time compute, better RL algorithms/environments, data quality, architecture tweaks, and even *divine intervention*.
   - One member bets on overall understanding, empirical search of architecture tweaks, some theory, data quality, better usage of parameters and RL. Another simply says *better finetunning*.
- **AI CEO Benchmark sparks debate**: The [skyfall.ai blogpost](https://skyfall.ai/blog/building-the-foundations-of-an-ai-ceo) was shared, presenting a new environment/benchmark for long-horizon planning capabilities of agents through a business simulator game, noting that **LLMs** seem to be performing quite under human baselines.
   - A member notes that *fiduciaries have to be natural persons by law*, but suggests the **AI CEO** is more of a capability than a replacement of the actual human.
- **HuggingFace Repository Setup causes download debacles**: A member expressed frustration with the **xet repository setup on HuggingFace**, complaining about the difficulty of downloading a model for fine-tuning and the unexpected cache location.
   - They sarcastically added, *I do however bet that the one guy who made a script that starts with trying to download the same model for the 5Kth time before using it as is. Is going to be happy that his script runs so fast now.*


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1440392539810300175)** (8 messages🔥): 

> `Yannic Kilcher videos on transformers, SAM 3: Segment Anything with Concepts` 


- **Community Seeks Kilcher's Transformer Talks**: A member requested a list of **Yannic Kilcher's** videos that explain **attention** or **transformer-related breakthroughs** in a progressive order to better understand the subject.
   - Another member provided a [link to Yannic Kilcher's channel search for transformers](https://www.youtube.com/@YannicKilcher/search?query=transformers) and a specific video, [The Wavefunction](https://www.youtube.com/watch?v=iDulhoQ2pro), explaining that the video is based on the original paper, though it can be a bit unclear.
- **SAM 3 Segments 'Love' with Concepts**: A member asked if **SAM 3 (Segment Anything with Concepts)** could be prompted to segment *love*, referencing a [Meta AI Research publication](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).
   - Another member responded with a playful reference to the song *What is Love?*, and another responded with *feeds you a cursed fractal which happens to write out baby don't hurt me.*


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1440429625007149217)** (23 messages🔥): 

> `Gemini 3 Benchmarking, OpenAI Bailout, Math Review Dissatisfaction, Segment Anything Model 3, Cogito v2-1 Analysis` 


- ****Gemini 3** Jumps Through Benchmark Hoops**: Members noted **Gemini 3's** impressive benchmark performance and questioned whether it was due to *benchmaxxing* or genuine generalization ability.
   - The question was raised whether the performance gains would extend to tasks outside of benchmarks, including private and novel ones.
- ****OpenAI** Accused of Seeking **Pre-Bailout****: A member shared an article from [Silicon Valley Gradient](https://siliconvalleygradient.com/openais-bailout-plans-were-leaked-in-a-letter-57abe1323544) alleging **OpenAI** sought a "pre-bailout" from the US government, despite denials.
   - The claim suggests that **OpenAI's** denials were met with skepticism by those closely following the situation.
- **Math Reviews Underwhelmed by New Model**: Some members shared negative math reviews of a certain new model, citing a [tweet](https://fxtwitter.com/nasqret/status/1990867412984717804) expressing dissatisfaction.
   - Counterpoints were raised ([tweet 1](https://x.com/robertghrist/status/1990876100814086167), [tweet 2](https://x.com/jasondeanlee/status/1990905064731652123)), with one user praising the model's mathematical capabilities ([tweet 3](https://x.com/gallabytes/status/1990821161241018557)).
- ****SAM 3D** Impresses with Segmentation Skills**: Meta AI released **Segment Anything Model 3 (SAM 3)**, showcased in a [blog post](https://ai.meta.com/blog/segment-anything-model-3/).
   - Specifically, one member highlighted the **Sam3D** part as especially impressive.
- ****Cogito v2-1**: DeepSeek's Post-Training Setback**: Cogito v2-1, a post-trained version of **DeepSeek**, was noted to underperform compared to its base model, as per [DeepCogito research](https://www.deepcogito.com/research/cogito-v2-1).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1440414407950864566)** (47 messages🔥): 

> `Palantir Cost vs Customization, Cursor CLI vs Claude Code, SAM 3, GPT-5.1-Codex-Max, LMSYS Miles Enterprise RL Framework` 


- **Deedy's Pod Cost vs Customization Chat**: A member asked about a **Cost vs Customization matrix** discussed in a podcast with Deedy, specifically whether **Palantir** was considered high cost and high customization and another member found the reference at [32:55 of the podcast](https://youtu.be/eWxSoIkZar0?si=-xnoR2lWlnPeS8Ub).
- **Cursor's CLI: The Underdog Coder?**: Members discussed the **Cursor CLI** and its capabilities compared to **Claude Code**, with one member reporting that while model execution and code quality were fine, the rest of the CLI seemed *very bare bones* even now, *with no custom slash commands* according to their documentation [review](https://cursor.com/cli).
- **Meta Launches SAM 3 for Segmentation**: **Meta** released **SAM 3**, a unified image/video segmentation model with text/visual prompts, claiming *2x better than existing models* with **30ms inference**, including a **Playground** for no-code testing and checkpoints/datasets on [GitHub/HuggingFace](https://ai.meta.com/sam3/), and powering Instagram Edits & FB Marketplace View in Room.
   - Roboflow announced a partnership with Meta to offer **SAM 3** as an infinitely-scalable endpoint, letting users prompt with plain text (e.g., *green umbrella*) to get pixel-perfect masks and object tracking, and compare **SAM 3** against **Claude** and **YOLO World**.
- **GPT-5.1-Codex-Max Damage Control**: **OpenAI** unveiled **GPT-5.1-Codex-Max**, natively trained to operate across multiple context windows through compaction, claiming it is built for *long-running, detailed work*.
   - Some observers characterized this as *damage control* after other releases, noting that it offers *more than twice the amount of tokens for 20% performance*, and expressed hope that OpenAI would *step up*.
- **LMSYS Spawns Miles, the RL Framework**: **LMSYS** introduced **Miles**, a production-grade fork of the lightweight **slime** RL framework, optimized for new hardware like **GB300** and large **Mixture-of-Experts** reinforcement-learning workloads, with links to the [GitHub repo](https://github.com/radixark/miles) and [blog post](https://lmsys.org/) detailing roadmap/status.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1440390106740555859)** (26 messages🔥): 

> `LLMs are non-deterministic, GPT-OSS-20B determinism, DSPy in production community, Anthropic model on Azure via DSPy, DSPy for inference` 


- **LLMs' Non-Determinism Debated**: Members discussed solutions for the [non-deterministic nature of LLMs](https://arxiv.org/abs/2402.12828) when running evaluations on a dataset, with one user reporting **98.4%-98.7%** accuracy on **316** examples using `gpt-oss-20b` with different incorrect examples each time.
   - Suggestions included dropping the temperature to **0**, right-sizing `max_tokens`, making the output format stricter, and fixing the seed, as well as exploring `dspy.CodeAct` or `dspy.ProgramOfThough` for more deterministic results.
- **`GPT-OSS-20B` Gets Specific Settings Tweaks**: A user shared their updated settings for `gpt-oss-20b`, including **temperature=0.01**, `presence_penalty=2.0`, `top_p=0.95`, `top_k=50`, and `seed=42`, noting that **temperature=0** led to more errors.
   - With these settings, they achieved stable **3-5** errors out of **316** examples.
- **DSPy Production Channel Wishlisted**: A member inquired about a dedicated channel for the **DSPy in production community**.
   - Another member responded that while there isn't one currently, there should be.
- **DSPy's Inference Alignment Probed**: A user inquired about the alignment of **GEPA** to **inference** use cases, seeking good examples, tutorials, blogs, or guides in that direction.
   - No concrete examples were provided in the messages.
- **LiteLLM to support Anthropic on Azure**: Regarding calling an [Anthropic model on Azure via DSPy](https://www.anthropic.com/blog/anthropic-on-azure), one member clarified it's a matter of **LiteLLM** adding support, similar to OpenAI on Azure, linking to the [LiteLLM Azure documentation](https://docs.litellm.ai/docs/providers/azure/).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1440422073712250951)** (20 messages🔥): 

> `Gemini 3 integration with Aider, Aider with Ollama, GPT-5.1 issues with Aider` 


- **Gemini 3's Aider Ascension**: Users discussed running **Gemini 3** with Aider, with instructions provided to use the command `aider --model=gemini/gemini-3-pro-preview --no-check-model-accepts-settings --edit-format diff-fenced --thinking-tokens 4k`.
   - A user also suggested `--weak-model gemini/gemini-2.5-flash` for faster committing.
- **Ollama Opens Options for Aider**: A user inquired about using **Aider with Ollama**.
   - The discussion did not elaborate further on specific configurations or experiences.
- **GPT-5.1 Glitches Generate Grief**: A user reported issues with **GPT-5.1** in Aider, encountering `litellm.APIConnectionError` related to `response.reasoning.effort` validation.
   - Despite setting `reasoning-effort` to different levels (low, medium, high), the issue persisted, potentially indicating a change on **OpenAI's side** or a problem with **Litellm** ([related issue](https://github.com/BerriAI/litellm/issues/1663)).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1440401772811386951)** (8 messages🔥): 

> `Llama 1B benchmark in CI, CPU architecture considerations, Benchmarking torch.compile vs tinygrad, Kernel imports, CuTeDSL` 


- **Llama 1B Aims for Faster CPU Performance in CI**: A bounty seeks to achieve faster **Llama 1B** performance on CPU in CI compared to Torch, prompting questions about integration within the testing framework, specifically whether to add a new test in `tests/speed/external_test_speed_v_torch.py`.
   - The discussion also includes clarifying if the target is specific CPU architectures or all supported ones, and confirming that "model speed" refers to **inference speed (tokens/sec)**.
- **Tinygrad Outperforms Torch on CPU with Llama 1B**: A member reports that **Llama1b** on Tinygrad is already outperforming Torch on CPU, achieving **6.06 tok/s** compared to Torch's **2.92 tok/s** with `CPU_LLVM=1` and 8 CPU cores, using only forward passes and no model weights.
   - This member and others were wondering whether to create a new benchmark in `test/external`.
- **`torch.compile` Benchmark**: A member is interested in benchmarking against a `torch.compile` PyTorch implementation.
- **Address Kernel Imports in `extra/optimization`**: The discussion suggests fixing the `from tinygrad.codegen.opt.kernel import Kernel` imports in the `extra/optimization` files.
   - Additionally, there is a suggestion to remove broken or unused examples/extra files that have not been updated recently.
- **CuTeDSL Mentioned**: A member shared a link to [SemiAnalysis's tweet](https://x.com/SemiAnalysis_/status/1790997414832906562) about **CuTeDSL**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1440774781531066398)** (2 messages): 

> `tinygrad bug fix, lab troubleshooting` 


- **Tinygrad update squashes mysterious bug**: A user reports that updating **tinygrad** resolved an issue they were experiencing, confirmed by an attached [image](https://cdn.discordapp.com/attachments/1070745817025106080/1440774781304569856/image.png?ex=691f6194&is=691e1014&hm=d7bf996fedfdb6d575736f5233a6c7c865660613e243caa6e37f034c476c8347&).
   - They would have tested sooner, but their *lab was having some trouble*.
- **Lab Troubles delay bug testing**: The user mentioned that their *lab was having some trouble*, delaying the bug testing on tinygrad.
   - This highlights the practical challenges in software testing and development environments.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1440487156098531399)** (7 messages): 

> `Manus Credit System Changes, TiDB Cloud Access Issues, Gemini 3 Integration with Manus, AI Coding Education` 


- ****Credits Confusion Causes Consternation****: A member inquired about [changes to the Manus credit system](https://manus.im), specifically regarding the transition to a **/4000 monthly** reset and whether it consolidates previous plans.
   - The user sought clarification on whether the "monthly reset" and "never expire" plans were rolled into a single monthly plan.
- ****TiDB Troubles Trigger Third-Party Tussle****: A member reported [inaccessibility to their TiDB Cloud account](https://tidbcloud.com/) provisioned through Manus, facing quota exhaustion and lacking console access to manage billing or spending limits.
   - They explored using the `ticloud` CLI but lacked the necessary API keys or OAuth login, and inquired about alternative access methods or direct support channels.
- ****Gemini Generation Gets Going?****: A member asked about the potential [integration of Gemini 3](https://deepmind.google/technologies/gemini/#introduction) with Manus.
   - Another member responded that Gemini 3 Pro plus Manus would equal total awesomeness.
- ****AI Advocate Aims at Aspiring Apprentices****: A member offered [AI coding education](https://en.wikipedia.org/wiki/Computer_programming) covering core concepts, advanced models, practical applications, and ethical considerations, inviting DMs for further engagement.
   - Another member questioned the appropriateness of this self-promotion.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1440416074964140174)** (2 messages): 

> `Gemini 3 Pro, Windsurf, Software Releases` 


- **Gemini 3 Pro hits Windsurf!**: **Gemini 3 Pro** is now available on **Windsurf**, according to the [announcement on X](https://x.com/windsurf/status/1990855986501034193?s=20).
- **Windsurf gets Gemini 3 glitch fixed!**: A small hiccup with **Gemini 3** was quickly resolved; users should now experience smooth functionality, and can [download the latest version](https://windsurf.com/download/editor).


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1440451938834714705)** (2 messages): 

> `Image Attachments, Temporary Hiccups` 


- **Sad Image Attached**: A user shared an image showing a sad face emoji.
   - Likely in response to a temporary issue, as noted by another user in the following message.
- **Temporary Hiccup Fixed**: A member reported that a temporary hiccup has been fixed.
   - This suggests that the sad face emoji in the previous message may have been related to this now-resolved issue.


  

---


---


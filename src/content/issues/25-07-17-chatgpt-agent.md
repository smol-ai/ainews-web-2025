---
id: MjAyNS0w
title: "ChatGPT Agent: new o* model + unified Deep Research browser + Operator computer use + Code Interpreter terminal"
date: '2025-07-17T05:44:39.731046Z'
description: >-
  **OpenAI** launched the **ChatGPT Agent**, a new advanced AI system capable of
  browsing the web, coding, analyzing data, and creating reports, marking a
  significant step towards human-like computer use. The agent, distinct from and
  superior to **o3**, is considered the first public exposure of what was
  internally called **o4**, now merged into **GPTNext**. It features end-to-end
  reinforcement learning, can operate for extended periods (tested up to 2
  hours), and is classified as "High" risk for biological misuse, with
  safeguards activated. Early benchmarks show mixed results, excelling in some
  tests like **WebArena** and **BrowserComp** but underperforming on others like
  **PaperBench**. Key figures involved include **Sam Altman**, **Greg
  Brockman**, and **Kevin Weil**, with technical insights from **xikun_zhang_**
  and risk commentary from **KerenGu** and **boazbaraktcs**. The launch sparked
  speculation about **GPT-5**, which was confirmed not to be the case.
companies:
  - openai
models:
  - o3
  - o4
  - gptnext
topics:
  - reinforcement-learning
  - benchmarking
  - model-performance
  - model-risk
  - long-context
  - model-deployment
  - fine-tuning
people:
  - sama
  - gdb
  - kevinweil
  - xikun_zhang_
  - keren_gu
  - boazbaraktcs
---


**ChatGPT is all you need.**

> AI News for 7/16/2025-7/17/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (226 channels, and 9565 messages) for you. Estimated reading time saved (at 200wpm): 703 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

In a very well received, classic OpenAI style [10am PT livestream,](https://www.youtube.com/watch?v=1jn_RpbPbEc) Sama and team launched "ChatGPT agent" with a meme-worthy opener (sitll wasn't the top meme of today):

![](https://resend-attachments.s3.amazonaws.com/qhdSMz9QRA1e5Uv)

The [blogpost](https://openai.com/index/introducing-chatgpt-agent/), [system card](https://cdn.openai.com/pdf/6bcccca6-3b64-43cb-a66e-4647073142d7/chatgpt_agent_system_card_launch.pdf), [system prompt](https://gist.github.com/Rutledge/4b0ef2d51ba2f1918a249bce35bdde9c), [Wired](https://www.wired.com/story/openai-chatgpt-agent-launch/) and [Every](https://every.to/vibe-check/vibe-check-openai-enters-the-browser-wars-with-chatgpt-agent) coverage, have focused on making [slides](https://www.youtube.com/watch?v=szJI9YJNEZk), [spreadsheets](https://www.youtube.com/watch?v=JAQ4p662It8), [research](https://www.youtube.com/watch?v=Wgn4JeYI9lY), [customizability](https://www.youtube.com/watch?v=EKMHiOQPwpc) (including [scheduled agents](https://x.com/neelajj/status/1945945913014546805?s=46)), and the HLE, FrontierMath benchmarks are of course great, but:

1. we shouldn't let benchmark fatigue distract from the fact of how quickly models and agents are running up these extraordinarily difficult, already superhuman tests,
2. most people are missing that **"the model" referred to in the blogpost is a distinct new model separate from and better than o3 i**f you look carefully at the labels:

![](https://resend-attachments.s3.amazonaws.com/9dlIn7cXzby8sxJ)

Similar to how Deep Research was the first product to publicly expose the full o3 anywhere, ChatGPT Agent seems to be the first product to publicly expose what would have been called o4, but is now being merged into GPTNext.

---

# AI Twitter Recap

**OpenAI ChatGPT Agent Launch**

- **OpenAI has launched the ChatGPT Agent**, a new unified system that combines deep research capabilities with the ability to operate a computer. The agent can [browse the web, use a terminal, write code, analyze data, and create reports, spreadsheets, and slides](https://twitter.com/OpenAI/status/1945890050077782149). The launch was announced by **OpenAI** with posts from key figures including **Sam Altman** who noted [it has been a real "feel the agi" moment for him](https://twitter.com/sama/status/1945917559796298083), **Greg Brockman** who shared this is a big step towards their [10-year goal of creating an agent that can use a computer like a human](https://twitter.com/gdb/status/1945923067403984979), and **Kevin Weil** who described its rollout to [Pro, Plus, and Teams users](https://twitter.com/kevinweil/status/1945896640780390631).
- **Technical insights from the development team** were shared by [@xikun_zhang_](https://twitter.com/xikun_zhang_/status/1945895070269583554), highlighting the power of **end-to-end Reinforcement Learning (RL)**, the importance of user collaboration, and a focus on real-world performance over benchmark chasing. The team also revealed that the agent can perform tasks for extended periods, with one internal test running for **2 hours**.
- **The ChatGPT Agent is OpenAI's first model classified as "High" capability for biological misuse risk**, a point emphasized by researchers [@KerenGu](https://twitter.com/KerenGu/status/1945944156935004415) and [@boazbaraktcs](https://twitter.com/boazbaraktcs/status/1945944398199677016). They stated that the strongest safeguards have been activated to mitigate these risks. However, benchmarks show that the agent has a **10% chance** of performing a "harmful action" like gambling with a user's savings if asked, [and is more likely to attempt to build a supervirus than o3](https://twitter.com/scaling01/status/1945930617775882728).
- **Early benchmark results for the agent** were shared by [@scaling01](https://twitter.com/scaling01/status/1945895473430089947), showing scores of **~42% on HLE**, **~27% on FrontierMath**, **~65% on WebArena**, **69% on BrowserComp**, and [**45% on SpreadsheetBench**](https://twitter.com/scaling01/status/1945896464632148366). It was also noted that the agent's performance is [lower than **o3** on benchmarks like **PaperBench** and **SWE-Bench**](https://twitter.com/scaling01/status/1945932154455695752).
- **The announcement led to widespread speculation and commentary**, with many users expressing disappointment that the release was not **GPT-5**. [@scaling01](https://twitter.com/scaling01/status/1945640155890483359) repeatedly confirmed from trusted sources that this was not **GPT-5**, leading to a state of [AI-psychosis and waiting for a date](https://twitter.com/scaling01/status/1945913979517247769). [@swyx](https://twitter.com/swyx/status/1945904109766459522) drew a parallel to the original **iPhone** launch, describing the agent as three things in one: a browser, a computer, and a terminal.

**Model Releases, Performance & Benchmarks**

- **Moonshot AI's Kimi K2 has become the #1 open model on the LMSys Chatbot Arena**, as [announced by the Arena](https://twitter.com/lmarena_ai/status/1945897926796185841) and celebrated by the [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1945897926796185841) team. The model is praised for its high performance and speed, particularly on **Groq's** hardware, where it achieves speeds of over **200 tokens/second** as reported by [@OpenRouterAI](https://twitter.com/OpenRouterAI/status/1945779694256722025) and demonstrated by [@cline](https://twitter.com/cline/status/1945627344997130473). It has been noted to beat **Claude Opus 4** on coding benchmarks while being up to **90% cheaper**.
- **xAI's Grok 4 has had safety issues investigated and mitigated**, according to an [official announcement from @xai](https://twitter.com/random_walker/status/1945614419213316571). However, the release has faced criticism, with [@boazbaraktcs](https://twitter.com/SebastienBubeck/status/1945669260027777049) expressing concerns about its safety. The model's new "companions" feature was also criticized by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1945737831697064446) for its low-quality "waifu engineering," noting character model clipping and typos.
- **Google DeepMind announced Veo 3**, their latest video generation model, is now available in public preview via the **Gemini API** and **AI Studio** as [per their official account](https://twitter.com/GoogleDeepMind/status/1945886603328778556). A detailed code example for generating video with complex prompts was [shared by @_philschmid](https://twitter.com/_philschmid/status/1945898590821584989). Additionally, **Gemini 2.5 Pro** is being integrated into **AI Mode** in **Google Search**, and it achieved a **31.55%** score on the **IMO 2025** math benchmark, [outperforming **Grok 4** (11.90%) and **o3 high** (16.67%)](https://twitter.com/denny_zhou/status/1945887753864114438).
- **Real-time video diffusion is now possible with MirageLSD**, a new model from **Decart AI**. [@karpathy](https://twitter.com/karpathy/status/1945979830740435186) provided a comprehensive overview of its potential, from creating alternate realities in video feeds and real-time movie direction to styling game environments with text prompts.
- **H-Net, a new hierarchical network**, has been introduced to create truly end-to-end language models by eliminating the tokenization step, [as shared by @sukjun_hwang](https://twitter.com/abacaj/status/1945898630289727854). This approach allows the model to process raw bytes directly.
- **Together AI announced record inference speeds for DeepSeek R1** on **NVIDIA B200s**, achieving up to **330 tokens/sec**, as [highlighted by @vipulved](https://twitter.com/vipulved/status/1945934641451675793).
- **The Muon optimizer played a key role in training Kimi K2**, a fact [@kellerjordan0](https://twitter.com/kellerjordan0/status/1945701578645938194) noted. The optimizer's first application was breaking the 3-second barrier in the **CIFAR-10** speedrun on a **3e14 FLOP** training run, while K2's training was **10 orders of magnitude larger** at **3e24 FLOPs**.
- **ColQwen-Omni**, a **3B** omnimodal retriever extending the **ColPali** concept, was introduced by [@ManuelFaysse](https://twitter.com/andersonbcdefg/status/1945855681976021268).

**AI Tooling, Frameworks & Infrastructure**

- **The debate between reasoning-native and memory-native models** was highlighted by [@jxmnop](https://twitter.com/jxmnop/status/1945857324285149256), who argued that major AI labs are overly focused on reasoning when they should be building **memory-native language models**, stating the door is "wide open" as no popular LLM currently has a built-in memory module.
- **Claude's desktop integrations are evolving it into an "LLM OS"**, according to [@swyx](https://twitter.com/swyx/status/1945734758102868243), who praised its utility with **Chrome, iMessage, Apple Notes, Linear, Gmail, and GCal**. For parallel execution, [@charliebholtz](https://twitter.com/HamelHusain/status/1945871155869178539) introduced **Conductor**, a Mac app for running multiple **Claude Code** agents simultaneously.
- **Asimov, a code research agent from Reflection AI**, was launched to address the fact that engineers spend **70%** of their time understanding code, not writing it. The launch was [announced by @MishaLaskin](https://twitter.com/hardmaru/status/1945628506035294697).
- **A new NanoGPT training speed record** was set by **Vishal Agrawal**, achieving a **3.28 FineWeb** validation loss in **2.966 minutes** on **8xH100 GPUs**. As [@kellerjordan0](https://twitter.com/kellerjordan0/status/1945920703158710316) reported, the speedup was achieved by replacing gradient `all_reduce` with `reduce_scatter` and other efficiency tweaks.
- **The LlamaIndex team published "The Hitchhiker’s Guide to Productionizing Retrieval"**, a detailed guide for building production-ready RAG systems. As [@jerryjliu0](https://twitter.com/jerryjliu0/status/1945647281782636974) summarized, the guide covers text extraction, chunking, embeddings, search boosting with semantic caching, and query rewriting, with practical examples using **Qdrant**.
- **Perplexity is sending out a new batch of invites for its Comet browser**, as [announced by CEO @AravSrinivas](https://twitter.com/AravSrinivas/status/1945669970618421699). [@rowancheung](https://twitter.com/AravSrinivas/status/1945620938068037633) noted that after a week of testing, the agent is starting to "actually stick."
- **Atropos v0.3**, the RL Environments framework from **NousResearch**, has been released. [@Teknium1](https://twitter.com/Teknium1/status/1945927019281478051) highlighted a key update: a new evaluation-only mode and a port of **@natolambert's Reward-Bench** for evaluating LLM-as-a-Judge capabilities.
- **Notion is using Turbopuffer to build state-of-the-art AI apps**, a case study [shared by @turbopuffer](https://twitter.com/turbopuffer/status/1945865085530026359).

**AI Research, Papers & New Techniques**

- **A critical essay on "AI for Science"** by [@random_walker](https://twitter.com/random_walker/status/1945820621142688068) and [@sayashk](https://twitter.com/sayashk) argues that AI might be worsening the **production-progress paradox**, where scientific paper output grows exponentially while actual progress stagnates. They contend that AI companies are misaligned, focusing on flashy headlines like "AI discovers X!" rather than addressing real bottlenecks. The authors suggest that [current AI-for-science evaluation is incomplete](https://twitter.com/random_walker/status/1945849588805447743), as it ignores impacts on researcher understanding and community dynamics.
- **A new blog post "All AI Models Might Be The Same"** by [@jxmnop](https://twitter.com/jxmnop/status/1945905080781451396) explains the **Platonic Representation Hypothesis**, suggesting the existence of universal semantics in AI models. This could have implications for tasks like understanding whale speech or decrypting ancient texts.
- **The SIGIR2025 Best Paper was awarded to the WARP engine for fast late interaction**, a recognition [highlighted by @lateinteraction](https://twitter.com/lateinteraction/status/1945924144412930338).
- **A new paper on Mixture of Recursions (MoR)** presents a method to build smaller models with higher accuracy and greater throughput. The paper, [shared by @QuixiAI](https://twitter.com/QuixiAI/status/1945907010584637891), covers models from **135M to 1.7B** parameters.
- **OpenMed, a collection of over 380 state-of-the-art healthcare AI models**, has been launched on **Hugging Face** by [@MaziyarPanahi](https://twitter.com/ClementDelangue/status/1945622980475691364), aiming to advance AI in medicine.
- **A paper from Alibaba-NLP on WebSailor demonstrates post-training models for Deep Research**, with [@AymericRoucher](https://twitter.com/AymericRoucher/status/1945870603275403693) noting that agentic RL loops at the end of post-training improved scores by **~4 percentage points**.

**Companies, Ecosystem & Geopolitics**

- **Perplexity AI announced a partnership with Airtel India**, a major milestone [shared by CEO @AravSrinivas](https://twitter.com/AravSrinivas/status/1945736795280613580). Following the announcement, Perplexity became the [#1 overall app on the App Store in India](https://twitter.com/AravSrinivas/status/1945960772091433081), surpassing ChatGPT.
- **Lovable, an AI agent startup, has raised $200M at a $1.8B valuation** led by Accel, as [announced by co-founder @antonosika](https://twitter.com/karansdalal/status/1945979009399132533).
- **At the AtCoder World Tour Finals 2025 Heuristic contest, a human competitor, @FakePsyho, took first place**, beating an **OpenAI** agent which secured second. [@hardmaru](https://twitter.com/hardmaru/status/1945850637528490134) celebrated the win for humanity, while [@andresnds](https://twitter.com/mckbrando/status/1945692340292854112) detailed OpenAI's participation in the 10-hour live exhibition.
- **U.S. visa issues are preventing top AI conferences from being held in the country**, a situation described as a "major policy failure" by [@natolambert](https://twitter.com/ClementDelangue/status/1945824425506398677). This has led to the independent organization of **EurIPS** in Copenhagen, which [**NeurIPS** officially endorsed](https://twitter.com/algo_diver/status/1945749595252039832).
- **The US vs. China tech dynamic remains a prominent topic**. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1945624983985639487) questioned why there isn't an "American Kimi," attributing it to misaligned incentives and later arguing that [US export controls have underestimated China's lead in key tech trees](https://twitter.com/teortaxesTex/status/1945733220336591109).
- **Humanloop, an early platform in the LLM evals space, is shutting down in September**. [@imjaredz](https://twitter.com/imjaredz/status/1945885618598474200) announced that his company, **PromptLayer**, is offering a migration package for **Humanloop** users.

**Humor/Memes**

- **The anticipation for an OpenAI release** was captured in a viral tweet from [@nearcyan](https://twitter.com/nearcyan/status/1945623927092646286), describing being at dinner with an **OpenAI** friend who "keeps vaguely gesturing towards the kitchen and grinning like our food is gonna come out. but we havent ordered yet".
- **The FFmpeg project announced a 100x speedup** from handwritten assembly, with a developer noting a performance increase of **100.18x** for the `rangedetect8_avx512` function, [as shared by @FFmpeg](https://twitter.com/LearnOpenCV/status/1945975913889329603).
- **On the realities of working in AI**, [@typedfemale](https://twitter.com/typedfemale/status/1945912359027114310) posted an image of a cramped server room with the caption "presenting: big jeff's trainium hell."
- **A meme about data contamination** was widely shared, showing a cartoon character whispering answers to another during a test, with [@vikhyatk](https://twitter.com/vikhyatk/status/1945969703266275548) captioning it, "we're gonna look at the benchmark and find samples that are as close to it as possible, but they're not an exact match so it doesn't count as training on test".
- **A joke about model development** from [@vikhyatk](https://twitter.com/vikhyatk/status/1945970434253664546) resonated with many: "i know my model is not biased, because i set bias=False on all of the linear layers".
- **Satirical commentary on tech culture** included a tweet from [@cto_junior](https://twitter.com/cto_junior/status/1945717278953386302) showing a person in flamboyant attire with the caption, "How I show up to all-hands where CEO announces we are out of cash."

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Kimi K2 Model Leaderboard Rankings and OpenAI Comparison

- [**Kimi K2 on Aider Polyglot Coding Leaderboard**](https://i.redd.it/wvr0xh2jecdf1.jpeg) ([Score: 178, Comments: 42](https://www.reddit.com/r/LocalLLaMA/comments/1m1vf6g/kimi_k2_on_aider_polyglot_coding_leaderboard/)): **The image displays the "Aider Polyglot Coding Leaderboard," which benchmarks coding LLMs on correctness, cost, and edit format. The Kimi K2 model is highlighted, achieving a** `56.0%` **success rate in coding tasks at a cost of** `$0.22`**, with a** `92.9%` **correct diff-format editing rate. The model is invoked via** `aider --model openrouter/moonshotai/kimi-k2`**, showcasing its lead as the most cost-effective among the compared models.** Commenters are impressed by the low $0.22 cost and discuss combining models like K2 as coder and r1 0528 as architect for potential benefits, suggesting focus on further cost reduction and role specialization.
    - There is debate about the reported cost efficiency of the Kimi K2 model on the Aider Polyglot Coding Leaderboard, with some users questioning if the benchmarked results are accurate. One user points out that Kimi K2's reported cost per output appears lower than its listed API price ([$2.20-$4 per 1M tokens](https://openrouter.ai/moonshotai/kimi-k2)), especially compared to Deepseek V3, which should theoretically be cheaper. The suspicion is that the benchmark may be underestimating token usage for Kimi K2, potentially due to generating more succinct responses than comparable models, or there may be a calculation error in reporting tokens used.
    - There's technical interest in hybrid architectures, particularly the suggestion to use another model (r1 0528) as the "architect" and K2 as the "coder" in a workflow, with an expectation that this combination would remain cost-efficient.
    - A detailed price comparison is made between Deepseek V3 ([$1.10/1M tokens](https://api-docs.deepseek.com/quick_start/pricing)), Kimi K2, and Sonnet-4 ([Anthropic pricing](https://www.anthropic.com/pricing#api)), emphasizing the importance of concise ("non thinking") outputs on overall cost. Concerns are raised that benchmark results do not line up with published API rates, suggesting the benchmark may be ",off by a factor of 10."
- [**Just a reminder that today OpenAI was going to release a SOTA open source model… until Kimi dropped.**](https://www.reddit.com/r/LocalLLaMA/comments/1m2gp16/just_a_reminder_that_today_openai_was_going_to/) ([Score: 386, Comments: 55](https://www.reddit.com/r/LocalLLaMA/comments/1m2gp16/just_a_reminder_that_today_openai_was_going_to/)): **The post references OpenAI's previously rumored plan to release a state-of-the-art open source language model but asserts that the release was reconsidered or overshadowed after the release of Kimi (Moonshot AI's Kimi Chat), which has recently gained attention for its advanced capabilities. Comparisons are drawn to prior competitive tensions between releases, notably with Llama 4 and Deepseek, signaling rapid iteration and one-upmanship among SOTA open-source and closed-source LLM vendors.** Top comments highlight an emerging pattern where anticipated OpenAI releases are preempted or outshined by rival models (e.g., Deepseek), suggesting a competitive 'race' that may repeatedly delay or deter OpenAI's open source releases.
    - Several commenters discuss the challenge OpenAI faces in releasing a new open-source model shortly after strong competitors like Kimi or Deepseek R2. There's a consensus that releasing a weaker model in close proximity to a stronger, more recent drop poses significant reputational risks and could undermine the model's adoption and perceived leadership in SOTA benchmarks.
    - A technical mention is made regarding the practical relevance and adoption of Meta's Llama 4, questioning whether it is actually being used in the community. In contrast, Google's Gemma 3 is cited as a high-quality alternative that users are turning to, indicating shifting perceptions of SOTA open-source models.
    - The discussion highlights a pattern: companies are hesitant to release if their model cannot compete with the most recent SOTA leader (e.g., Kimi, Deepseek R2), which indicates that timing and performance relative to competitors' public benchmarks are key factors in release strategy and community adoption.

### 2. Mistral Le Chat Feature Announcements and Improvements

- [**Mistral announces Deep Research, Voice mode, multilingual reasoning and Projects for Le Chat**](https://mistral.ai/news/le-chat-dives-deep) ([Score: 467, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1m2bigh/mistral_announces_deep_research_voice_mode/)): **Mistral AI's Le Chat introduces several technical upgrades: (1) Deep Research mode employs a tool-augmented agent to produce structured, reference-backed reports on complex topics, using planning, citation sourcing, and synthesis (see announcement [here](https://mistral.ai/news/le-chat-dives-deep)); (2) Voice mode is powered by Voxtral, a proprietary, low-latency voice model optimized for ASR/transcription; (3) the Magistral model enables context-rich, natively multilingual and code-switching reasoning; (4) new Project folders allow context-scoped thread organization, and (5) advanced image editing is available via Black Forest Labs. The Deep Research pipeline specifically demonstrates multi-source, citation-heavy analytics—moving beyond tabular output to integrate real-world filings and financial data.** Comments highlight that Voxtral Mini ASR offers superior transcription performance and lower cost compared to Whisper Large, and emphasize the value of permissive licensing for supporting the LLM ecosystem. The Deep Research UI is noted as a technical design strength.
    - A user reports that Mistral's "voxtral mini" transcription model outperformed OpenAI's Whisper Large model not only in terms of quality but also cost, suggesting notable improvements in speed and/or accuracy for speech-to-text tasks compared to previous state-of-the-art models.
    - Discussion includes a query about whether any local language models currently offer research assistance features comparable to the deep research functionality found in ChatGPT and Gemini, indicating interest in alternatives with similar advanced reasoning and synthesis capabilities available for self-hosting.
    - Observations on Mistral's Le Chat highlight its speed and solid usability, though it's noted as trailing "the leaders" (e.g., OpenAI, Google) in benchmark performance. Nonetheless, its open weights and permissive licensing are seen as vital for fostering innovation and supporting European/global competition in AI.
- [**MCPS are awesome!**](https://i.redd.it/p3766l11qbdf1.png) ([Score: 321, Comments: 71](https://www.reddit.com/r/LocalLLaMA/comments/1m1sjsn/mcps_are_awesome/)): **The post showcases the use of multiple Model Control Protocol Servers (MCPs)—17 in total—interfacing with Open WebUI and local LLMs, allowing dynamic invocation of system tools such as web search and Windows CLI. The command shown in the image demonstrates PowerShell-based real-time resource monitoring using Python (psutil and GPUtil) and the Qwen14B LLM, outputting detailed metrics:** `CPU load: 7.6%`**,** `RAM: 21.3%`**, and** `GPU (RTX 3090 Ti): 16% load, 18,976MB/24,564MB used at 61°C`**. This highlights the integration's practicality in context-aware resource monitoring for LLM environments. [Image link](https://i.redd.it/p3766l11qbdf1.png)** Commenters caution about security, noting the risk of running code as agents (`"rm -rf *"` risk), and warn that each tool invocation incurs a significant context/token cost (~600–800 tokens), which can rapidly consume effective context windows in local models (<5K tokens), potentially degrading LLM performance even at chat initialization.
    - A critical point is made regarding MCPS (Modular/Multimodal Capability Plugins) and their impact on context window: each tool instance can consume `600-800 tokens`, which can severely reduce the usable context for smaller models with <5k token windows, potentially degrading performance before user input even begins due to the verbose system prompt required to describe available tools.
    - A user points out that enabling native tool calling in model settings can significantly boost performance, highlighting the importance of specific configuration flags for optimal inference efficiency in local deployments.
    - There is a discussion around MCPS usage emphasizing the necessity to evaluate their real benefit in production systems, including determining whether MCPS are stateful/stateless and considering their actual effect on system design, reliability, and maintainability versus alternative approaches.

### 3. LocalLlama Community Growth and Milestones

- [**We have hit 500,000 members! We have come a long way from the days of the leaked LLaMA 1 models**](https://i.redd.it/zfvdqak3zcdf1.png) ([Score: 605, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1m1xqv1/we_have_hit_500000_members_we_have_come_a_long/)): **The image celebrates the 'LocalLlama' subreddit reaching 500,000 members and highlights its focus on discussions about AI and Meta's LLaMA models, a community rapidly growing since its inception post-LLaMA 1 leak (March 2023). The milestone signals the widespread interest and growth in open-source large language model (LLM) communities, paralleled by shifts in the technical focus (from niche, hands-on experimentation to broader, mainstream LLM discourse).** Top comments discuss the irony that as LLaMA and its community have grown, models are becoming less 'local' (requiring more resources or cloud-based infrastructures). There is also technical concern about the dilution of deep technical content as the subreddit becomes less specialized and more mainstream, reflecting on the evolving landscape of open-source LLM engagement.
    - Commenters note a significant shift in the direction of the LLaMA models: originally known for their local, openly available foundation, they are increasingly becoming non-local and less accessible for personal or on-premises deployment as Meta updates licensing and distribution terms.
    - There's discussion on how the rise in community membership correlates with a dilution of technical content; quality technical posts and deep, state-of-the-art (SOTA)-focused discussion are expected to decline as the subreddit grows, shifting toward mainstream, product-centric threads rather than open source, cutting-edge research.
    - Concerns are raised over the evolving definition of 'local' in AI/LLM development, with some users lamenting that modern LLaMA iterations lack both the original 'llama' spirit and their former hardware independence, reflecting broader industry trends toward increased model centralization and restricted access.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. OpenAI ChatGPT Agent Release, Features, and Risk Discourse

- [**ChatGPT Agent released and Sams take on it**](https://i.redd.it/b0xmxole1hdf1.jpeg) ([Score: 463, Comments: 186](https://www.reddit.com/r/OpenAI/comments/1m2e2sz/chatgpt_agent_released_and_sams_take_on_it/)): **The image is a snapshot of Sam Altman's announcement regarding the release of ChatGPT Agent, OpenAI's new AI system capable of performing complex, multi-step tasks independently using its own computer. Altman emphasizes its advanced task automation (e.g., shopping, booking, analysis), its integration of research and operator functions, and the substantial new system safeguards implemented to mitigate privacy, security, and operational risks. The deployment is intentionally iterative, with strong user warnings regarding trust and access levels, highlighting the need for minimal permissions due to possible adversarial manipulations and unpredictable behaviors.** Top technical comments express skepticism about the system's reliability—one notes 'the completed result was only 50% accurate', while others highlight reluctance to trust the agent with financial actions and urge OpenAI to prioritize accuracy and consistency in basic functionalities before more ambitious releases.
    - A key technical criticism raised was the accuracy of the released ChatGPT Agent, with one commenter specifically stating that *"the completed result was only 50% accurate"*, highlighting substantial limitations in deployment for tasks requiring high reliability.
    - Security and trust in autonomous financial actions were discussed, with skepticism toward enabling agents to perform independent purchasing or financial transactions given current error rates and lack of robust risk mitigation. This underscores concerns about the maturity of agentic AI for high-stakes or sensitive domains.
    - Multiple commenters called attention to persistent reliability and consistency issues with foundational model behaviors, advocating for improvements in core functionality before introducing ambitious agentic features or full autonomy.
- [**LIVE: Introducing ChatGPT Agent**](https://www.youtube.com/watch?v=1jn_RpbPbEc) ([Score: 294, Comments: 246](https://www.reddit.com/r/singularity/comments/1m2cv1j/live_introducing_chatgpt_agent/)): **OpenAI has unveiled a new ChatGPT agent architecture (see their [video demo](https://www.youtube.com/watch?v=1jn_RpbPbEc)) capable of multimodal understanding, direct API/web integrations, and autonomous multi-step task execution (e.g. booking, document handling, service interactions). Key technical highlights include a focus on transparent security, robust task sequencing, and a new system for orchestrating actions—poised to improve automation in both consumer and enterprise settings.** Technical commenters express impatience with the pace and substance of public demos, calling for real-world utility beyond scripted scenarios. Concerns are raised about the demo's relevance and the current depth of actual task autonomy achievable by the agent.
    - Commenters express skepticism about the practical utility of the ChatGPT Agent, with one asking for any demonstration of "actual work" beyond simple scenarios, highlighting concerns over the agent's ability to go beyond surface-level tasks and make tangible progress towards more general autonomy.
    - Another commenter critiques the prompt engineering demonstrated, arguing that leveraging deep research capabilities should already handle tasks like finding outfits and gifts, implying that current use-case demos do not evidently push the model beyond existing capabilities.
    - There is a subtle implication that the live demo format or environment may be detracting from a technically impressive presentation, as referenced by discomfort with the pacing and setup, suggesting that technical audiences expect more polished and efficient product showcases for advanced AI tools.
- [**OpenAI’s New ChatGPT Agent Tries to Do It All**](https://www.wired.com/story/openai-chatgpt-agent-launch/) ([Score: 163, Comments: 50](https://www.reddit.com/r/OpenAI/comments/1m2d5yd/openais_new_chatgpt_agent_tries_to_do_it_all/)): **OpenAI's newly announced ChatGPT Agent automates multi-step, context-dependent tasks by integrating with external APIs and running its own browser instance to interact with online services. However, the demo was marred by bugs—such as context loss (forgetting the wedding date), site access failures, and inefficient browser automation that raises security and session management concerns (e.g., cross-domain access and login persistence). The technical design aims for general-purpose autonomy, but user experience and implementation robustness remain major points of contention. Further detail in this [WIRED article](https://www.wired.com/story/openai-chatgpt-agent-launch/).** Key debates center on the impracticality of agent-based browser automation (especially with respect to user authentication across services), inconsistent community standards for evaluating such agents (OpenAI vs. competitors), and criticism of OpenAI's demo quality, suggesting potential misalignment between technical ambition and end-user deliverability.
    - A technical critique is made regarding ChatGPT's inability to maintain context in a demo scenario (specifically, forgetting the wedding date), and its handling of browser limitations by switching to a reader mode due to 'cross-domain issues', highlighting current challenges in agent reliability and integration with web data.
    - One comment addresses efficiency concerns of agents running their "own browser"—specifically questioning how such agents will authenticate and access services that users have open in their local browser contexts, implying a lack of seamless session handling or secure credential sharing between agent browsers and native user environments.
    - There is skepticism about the robustness of AI agent backends, with direct comparison to previous efforts like Manus, noting that even with a ‘robust backend’ (inferred reference to OpenAI’s infrastructure), actual product reliability and productivity remain major technical hurdles, regardless of hype or announced capabilities.
- [**You know it’s serious when they bring out the twink**](https://i.redd.it/xx7p05ewlgdf1.jpeg) ([Score: 289, Comments: 48](https://www.reddit.com/r/singularity/comments/1m2bt5e/you_know_its_serious_when_they_bring_out_the_twink/)): **The image is a notification for a major OpenAI presentation featuring Sam Altman and key team members, highlighting the unveiling of a 'unified agentic model in ChatGPT.' Technical commentary notes that this aligns with expectations for the next major model ('GPT-5 era'), and that the phrase 'unified model' matches prior references to GPT-5 as an agentic or agent-driven architecture, potentially called 'Agent 1.'** A top comment speculates on the naming convention and reaffirms the link between 'unified model' and the much-anticipated GPT-5 update, referencing OpenAI's roadmap and earlier leaks.
    - Technical discussion centers around the possibility that the so-called "unified model"—referenced as related to GPT-5—may signal a significant architectural or branding change from OpenAI (potentially foregoing the "GPT-5" name for something akin to the "Agent 1" concept derived from Daniel's "2027" project). This hints at integrated multi-modal or persistent agent capabilities, which aligns with earlier statements about next-gen models combining text, reasoning, and potentially real-world task performance.
- [**ChatGPT Agent will be available for Plus, Pro, and Team users**](https://www.reddit.com/r/OpenAI/comments/1m2dnw5/chatgpt_agent_will_be_available_for_plus_pro_and/) ([Score: 323, Comments: 94](https://www.reddit.com/r/OpenAI/comments/1m2dnw5/chatgpt_agent_will_be_available_for_plus_pro_and/)): **OpenAI announced that ChatGPT Agent functionality will roll out to Pro users (with 400 queries/month cap) and Plus/Team users (40 queries/month), with Pro getting access immediately and Plus/Team within a few days, per the [OpenAI Blog](https://openai.com/index/introducing-chatgpt-agent/) and their [livestream](https://www.youtube.com/watch?v=1jn_RpbPbEc). This feature is restricted geographically, not launching in the EEA or Switzerland initially. Query allocation is tier-dependent, emphasizing controlled access and resource management.** Technical commentary in the comments expresses dissatisfaction with the limited monthly request pools and closed ecosystem, with calls for self-hosted or more natively integrated agents (e.g., local Operator in-browser) for user augmentability and intervention, and speculation about the impact of possible open weights models or non-OpenAI alternatives.
    - Users express concerns about the absence of the ChatGPT Agent feature in the European Economic Area (EEA), highlighting ongoing compliance and feature rollout delays—this potentially relates to ongoing regulatory obstacles like the Digital Markets Act, and that 'connectors' for GPTs are still unavailable in the EU.
    - Technical dissatisfaction is voiced regarding OpenAI's use of monthly request pools (quotas), as this model could limit how developers and power-users architect complex or continuous workflows that depend on the agent for extended autonomous or multi-step operations.
    - One commenter argues that the 'walled garden' approach of OpenAI's agent architecture limits user intervention and customization, suggesting a need for locally runnable Operator models and open-weight alternatives (possibly hinting at rumored open browser agent strategies and referencing competitive pressure from third-parties like Microsoft).
- [**Agent = Deep Research + Operator. Plus users: 40 queries/month. Pro users: 400 queries/month.**](https://www.reddit.com/r/OpenAI/comments/1m2drew/agent_deep_research_operator_plus_users_40/) ([Score: 127, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1m2drew/agent_deep_research_operator_plus_users_40/)): **OpenAI's Agent integrates autonomous task execution with user interruptibility and in-process clarification, featuring a combination of a 'Deep Research' core and an 'Operator' for real-time interaction (see official [product page](https://openai.com/index/introducing-chatgpt-agent/)). The system emphasizes security: it incorporates prompt injection resistance and a concealed observer mechanism for runtime threat detection, with ongoing updates to bolster defenses against evolving exploits. Query limits are set at 40/month for Plus and 400/month for Pro users. Further technical details are introduced in their [launch presentation](https://www.youtube.com/watch?v=1jn_RpbPbEc).** Commenters express concerns about the massive scaling of automated user actions (e.g., job applications, content creation), potential workflow automation overlaps with tools like n8n, and the dynamics of web-based anti-bot measures. There are also requests for robust API/CLI integration and discussions on practical prompt/task execution limits, specifically regarding whether session timeouts will constrain large-scale automation tasks.
    - Technical concerns are raised about the scalability of task automation using agents, specifically whether existing agents can handle high-volume automation such as *applying to thousands of jobs*, or if there are system-imposed timeouts or prompt-length limitations (such as a *30-minute timeout* or input length cap). This is especially relevant for workflows that generate or process large numbers of requests automatically.
    - There is interest in the availability of this agent framework via API and CLI, with an implicit focus on how such interoperability would enable advanced and automated integrations in custom pipelines.
    - A request is made for direct *comparisons with the Manus agent platform*, highlighting a demand for benchmark data, feature parity reviews, or case studies evaluating the performance, extensibility, or usability of this new system versus Manus.
- [**Seems like OpenAI is planning to release Agent Mode, codenamed “Odyssey”. (check all 5 pics)**](https://www.reddit.com/gallery/1m1w0y5) ([Score: 258, Comments: 60](https://www.reddit.com/r/singularity/comments/1m1w0y5/seems_like_openai_is_planning_to_release_agent/)): **OpenAI is reportedly preparing to release a new feature called 'Agent Mode,' internally codenamed 'Odyssey.' No specific model benchmarks, architectural details, or implementation information are available due to a lack of accessible source material. The post provides no technical evidence or demonstrations beyond this rumored codename association.** Commenters clarify that the '5 pictures' referenced are unrelated to GPT-5, reflecting some confusion or speculation about potential product releases or codenames. There is also light skepticism regarding the 'Odyssey' codename, humorously suggesting it may imply a lengthy timeframe before release.
    - There is speculation regarding the internal codename 'Odyssey' for OpenAI's upcoming Agent Mode, with one user pointing out that the '5 pictures' referenced are not an implication of 'GPT-5'. This clarifies potential confusion about whether the number of images was hinting at a new model version or just showing features.
- [**Yep most probably agents we gonna see today**](https://i.redd.it/4crtpuslegdf1.png) ([Score: 150, Comments: 25](https://www.reddit.com/r/OpenAI/comments/1m2aq68/yep_most_probably_agents_we_gonna_see_today/)): **The image displays a tweet from OpenAI announcing a livestream event in 3 hours, hinting at a collaboration or integration involving ChatGPT, Deep Research, and Operator via the use of handshake emojis. The substantial engagement (426,000 views) underscores heightened anticipation, likely fueled by speculation about the release of agentic features or even a potential GPT-5 showcase. The context provided by the subreddit and comments points to community expectations around significant technical upgrades beyond current models (such as GPT-4o).** Commenters express hope for more robust agentic capabilities potentially built on GPT-5, criticizing current OpenAI models (GPT-4o, GPT-4.1, o3) for hallucinations or lackluster performance compared to rivals like Gemini 2.5 and Claude 4 Sonnet. There's also skepticism regarding feature availability for regular users, and disappointment with recent developments if major improvements aren't announced.
    - Several users discuss dissatisfaction with current OpenAI models for agentic workflows: GPT-4o is noted to hallucinate frequently, GPT-4.1 performance is described as lackluster, and GPT-4.5's API removal is criticized. These shortcomings are compared to competitor models such as Gemini 2.5 Flash, Claude 4 Sonnet, and Gemini 2.5 Pro, with claims that these alternatives offer better performance for daily use.
    - A technical comment highlights that the Operator agent has now been integrated directly into ChatGPT instead of remaining on an external page, but users describe this as not introducing any significant new agent functionality. There is disappointment expressed regarding the lack of material progress or new features for agentic tasks.
- [**Deep research and Operator**](https://i.redd.it/gjmo6kcyyfdf1.jpeg) ([Score: 324, Comments: 52](https://www.reddit.com/r/singularity/comments/1m28h1d/deep_research_and_operator/)): **The image shows a tweet by OpenAI announcing a collaborative event involving 'ChatGPT', 'Deep research', and 'Operator' with an upcoming livestream, suggesting a significant product announcement or integration. Comments speculate this could represent the unveiling of a unified product, potentially GPT-5, signaling possible progress toward AGI or advanced multimodal capabilities. The tweet's high engagement metrics also highlight strong community interest in the event and its implications for AI assistants and research workflows.** Top comments reflect skepticism about practical demonstrations (e.g., "better not be booking a flight again"), and curiosity about the merging of ChatGPT with more advanced technologies, indicating technical anticipation and some fatigue with underwhelming demos seen in prior launches.
    - Several comments speculate about the technical implications of merging ChatGPT with more advanced models or systems, referencing the potential convergence into a single, more capable product (possibly under the GPT-5 name). This suggests continued evolution in model architecture, potentially combining deep research and operator models for broader multi-modal or agent-like capabilities.
- [**Looks like we getting agent mode in tomorrow's announcement?**](https://www.reddit.com/gallery/1m1w0i8) ([Score: 274, Comments: 49](https://www.reddit.com/r/OpenAI/comments/1m1w0i8/looks_like_we_getting_agent_mode_in_tomorrows/)): **The Reddit post speculates about an imminent release of an 'agent mode' for a leading AI platform (likely OpenAI or similar), with users expressing concern that prior features like Operator v1 were not made available to Plus-tier subscribers. Technical feature requests in the discussion include: Android and web support for recording, UK connectivity, synchronization of Project files (desktop/Google Drive), restoration of AVM accents, use of the latest voice model for 'Read aloud', and more robust customization of voice via SVM and custom instructions. There is skepticism about the value of AI-driven reservation features, questioning industry focus on this use case.** Commenters debate the rollout strategy, noting frustrations with feature availability for non-Pro users and emphasizing a preference for broad usability improvements (cross-platform voice, project sync) over niche automation like reservations. Some express concern about subscription tier gatekeeping and hope for immediate Plus-level access.
    - Several users compare upcoming "agent mode" features to previous releases, notably voicing concerns that essential updates (e.g., Operator v1) were slow, buggy, and unavailable for certain user tiers (e.g., Plus subscribers vs. Pro), creating technical and UX fragmentation.
    - There's strong demand for advanced multimodal capabilities: users explicitly request support for audio recording on Android and web, improved file/project syncing (especially with Google Drive integration), and more naturalistic TTS features such as using new voice models, custom instructions, and restoration of specific AVM accents.
    - Some skepticism exists over the utility of "agent" features focused on tasks like reservations, with users comparing them to marketing-heavy but underwhelming implementations (e.g. Google's Bard); instead, the discussion emphasizes the need for genuinely transformative leaps in agent autonomy and utility, with robust, scalable backend support.

### 2. Benchmarks & New Model Performance: ChatGPT Agent, Gemini, and Video/Editing Releases

- [**ChatGPT Agent is the new SOTA on Humanity's Last Exam and FrontierMath**](https://i.redd.it/f13lg1wqwgdf1.png) ([Score: 404, Comments: 104](https://www.reddit.com/r/singularity/comments/1m2deg5/chatgpt_agent_is_the_new_sota_on_humanitys_last/)): **The image presents benchmark results showing that the ChatGPT Agent in 'agent mode' (with access to a browser, computer, and terminal) achieves state-of-the-art (SOTA) pass rates on both 'Humanity’s Last Exam' and 'FrontierMath (Tier 1–3)' compared to other OpenAI models like 'o4-mini', 'o3', and ablations without tools. The results highlight the significant performance boost provided by tool-augmented agentic capabilities over plain LLMs: agent mode achieves the highest bars on each benchmark, indicating its superior real-world task-solving ability.** Comments discuss the shift in benchmark relevance due to agentic capabilities, arguing that the ability to create real-world artifacts (such as presentations) matters more than incremental benchmark gains. Others note possible unfairness in comparisons with multi-agent systems (e.g., Grok 4 Heavy). There is technical consensus that agentic benchmarks and tool use represent a significant new metric for AI progress.
    - Multiple commenters debate the validity of claims around SOTA on Humanity's Last Exam (HLE) and FrontierMath, pointing out that Grok 4 Heavy reportedly achieved a superior HLE score, although it utilized a swarm of agents rather than a single agent, raising fairness concerns in benchmarking cross-architecture comparisons.
    - There is a technical shift noted in the importance of benchmarks: while pure test scores on datasets like HLE and FrontierMath remain valuable, today's focus is increasingly on agentic capabilities—performance in real-world tasks such as automated tool use, memory retention, contextual reasoning, and the creation of complex artifacts like presentations. This move suggests that future benchmarks will likely measure broader, more applied agent intelligence rather than just static test performance.
    - Skepticism is raised about the reliability of benchmarks whose questions and answers may be publicly available, with a user suggesting that ARC-AGI2 and similar benchmarks featuring tool use, terminal access, and browser integration represent a more robust, real-world evaluation of agentic systems. The absence of strictly controlled or "private" benchmarks (like ARC-AGI2's approach) undermines the apples-to-apples comparison with human performance.
- [**Gemini 2.5 Pro scores best on the 2025 IMO on MathArena!**](https://i.redd.it/sqefsuvsegdf1.png) ([Score: 105, Comments: 27](https://www.reddit.com/r/Bard/comments/1m2arkr/gemini_25_pro_scores_best_on_the_2025_imo_on/)): **The image presents a benchmark result from MathArena, showing that Google's Gemini 2.5 Pro achieved the highest accuracy (31.55%) among various large language models on the 2025 International Mathematical Olympiad (IMO) evaluation. The table breaks down model performance by individual IMO problem (1-6) and shows cost metrics, notably highlighting Gemini 2.5 Pro's 71% accuracy on problem 5 and a total cost of $431.97 for the test. The benchmark emphasizes math problem-solving skills, especially for high-school level competition problems, and the results signal a significant step in LLMs' mathematical reasoning capabilities.** Commenters express surprise, noting subjective differences between Gemini 2.5 Pro and OpenAI's GPT-4 (O3) on math and reasoning tasks, with preferences differing based on use case (e.g., proofs vs. coding). There is also skepticism regarding natural language proof construction, indicating limitations in current model abilities for rigorous mathematical argumentation.
    - A user notes that, despite Gemini 2.5 Pro's strong showing on the 2025 IMO MathArena, in practical math, puzzle, and research scenarios, models like O3 High often outperform it in reasoning and problem solving, with Gemini 2.5 Pro tending towards assumptive answers and less nuanced conversation. They still find Gemini 2.5 Pro preferable for coding and general usage, suggesting notable domain-specific differences in model strengths.
    - Another commenter highlights that constructing rigorous, complete math proofs in natural language is a distinct challenge, even for top-performing models. This underscores the gap between quantitative benchmark performance (such as on the IMO) and real-world mathematical reasoning where detailed proof steps are required.
    - Deepseek Prover v2 is mentioned as surpassing both Gemini 2.5 Pro and O3 High in mathematical reasoning tasks, indicating ongoing competition and differentiation among state-of-the-art models specializing in math problem solving.
- [**A new open source video generator PUSA V1.0 release which claim 5x faster and better than Wan 2.1**](https://www.reddit.com/r/StableDiffusion/comments/1m1x2z7/a_new_open_source_video_generator_pusa_v10/) ([Score: 151, Comments: 49](https://www.reddit.com/r/StableDiffusion/comments/1m1x2z7/a_new_open_source_video_generator_pusa_v10/)): **PUSA V1.0 is an open-source video generation model that claims to be '5x faster and better than WAN 2.1', while maintaining architectural similarity. It is a unified model supporting various tasks including text-to-video (t2v), image-to-video (i2v), defining start-end frames, and video extension. The model's technical page and demos are available on [the official site](https://yaofang-liu.github.io/Pusa_Web/), and WAN 2.1's 14B parameter model is referenced as a baseline for performance.** A commenter expresses skepticism about the quality of PUSA V1.0's example videos, suggesting that the claimed improvements over WAN 2.1 may not be visually convincing.
    - Multiple users question the performance claims of PUSA V1.0 versus Wan 2.1, noting that while PUSA claims to be 5x faster than default Wan, Wan with Self Forcing LoRA achieves 10x speedup, implying possible exaggeration or context specificity in reported metrics.
    - One user checks the model architecture, noting 'wan2.1 14B', potentially referencing the model size, suggesting PUSA may need to be evaluated in the context of comparable parameter counts and architectures.
    - Technical criticism emerges about the visual fidelity, particularly of human figures in the generated videos, indicating that qualitative output remains lacking despite speed improvements, which is a critical metric in video generation tasks.
- [**HiDream image editing model released (HiDream-E1-1)**](https://i.redd.it/a3dnmlthlbdf1.jpeg) ([Score: 228, Comments: 72](https://www.reddit.com/r/StableDiffusion/comments/1m1rz2s/hidream_image_editing_model_released_hidreame11/)): **HiDream-E1-1 is a newly released image editing model, building upon HiDream-I1, with its official model hosted on Hugging Face ([link](https://huggingface.co/HiDream-ai/HiDream-E1-1)). The attached demo image ([view here](https://i.redd.it/a3dnmlthlbdf1.jpeg)) illustrates advanced editing capabilities: transforming subjects and environments (e.g., making a character appear as museum art, swapping bullets for butterflies, converting a hummingbird into glass, altering objects' colors like a gold toy car, and changing scene themes). These showcase localized and semantic editing proficiency, suggestive of controlled diffusion or inpainting workflows.** Discussion in the comments centers on the potential integration with ComfyUI and comparisons with FLUX Kontext, another editing model. There is significant interest in an INT4 Nunchaku quantized version for efficient inference on mid-range hardware (e.g., RTX 3060 12GB), reflecting expectations for broad, resource-friendly usability.
    - There is interest in an INT4 Nunchaku quantized version of HiDream-E1-1, as quantization could enable much faster performance in ComfyUI and avoid out-of-memory errors, particularly on GPUs with only 12GB VRAM like the RTX 3060.
    - A direct technical comparison is suggested between HiDream-E1-1 and Flux Kontext, highlighting the need for benchmarking to determine strengths, capabilities, and differences between these two image editing models.
    - A user posts a real-world example with different seed values for two prompts (CFG 2.3, steps 22, Euler sampler), which may be useful for those examining output variance and reproducibility with the HiDream-E1-1 model.
- [**🚀 Just released a LoRA for Wan 2.1 that adds realistic drone-style push-in motion.**](https://v.redd.it/2jrxstp8vfdf1) ([Score: 668, Comments: 56](https://www.reddit.com/r/StableDiffusion/comments/1m28062/just_released_a_lora_for_wan_21_that_adds/)): **A new LoRA (Low-Rank Adaptation) model has been released for the Wan 2.1 Image-to-Video (I2V) 14B 720p architecture, specifically engineered to generate realistic 'drone-style' push-in camera motion for generative video. The LoRA was trained on** `100 drone push-in clips` **and iteratively refined over** `40+ versions`**, and is provided with a ComfyUI workflow for seamless integration; it can be triggered with the text prompt 'Push-in camera'. The model and workflow are available on [HuggingFace](https://huggingface.co/lovis93/Motion-Lora-Camera-Push-In-Wan-14B-720p-I2V#AI).** Commenters noted imminent development of a Text-to-Video (T2V) version for potential use with Wan VACE, but highlighted it remains untested. Overall reception indicates the LoRA achieves notably realistic motion, with anticipation for further expansion to other video synthesis pipelines.
    - A user inquires about training a 'push-out' motion LoRA by reversing the training data clips used for the push-in motion, questioning if this approach would be sufficient to generate realistic inverse camera movement with minimal additional data collection. This touches on model training efficiency and data augmentation strategies in LoRA fine-tuning.
    - Another commenter discusses preparing a T2V (Text-to-Video) version compatible with Wan VACE, noting it's untested and may have different performance, and highlights community interest in integrating this LoRA into other specialized pipelines, possibly requiring domain adaptation or further fine-tuning.

### 3. Cultural and Existential AI Debates (Creativity, AGI, AI Impact Memes)

- [**We just calling anything agi now lmao**](https://i.redd.it/spk18cfabhdf1.jpeg) ([Score: 735, Comments: 259](https://www.reddit.com/r/singularity/comments/1m2fjje/we_just_calling_anything_agi_now_lmao/)): **The image is a screenshot of a tweet from OpenAI CEO Sam Altman (@sama), where he describes witnessing a ChatGPT agent autonomously use a computer to perform a sequence of tasks, framing it as an 'AGI moment' and highlighting the impact of seeing the system plan and execute actions. The post's context and comment discussion suggest skepticism about labeling current AI models as AGI (Artificial General Intelligence), reflecting a trend of marketing hype overtaking substantive AGI benchmarks or capabilities.** Top comments challenge the significance of the tasks and the validity of naming such demos as AGI moments, calling it 'marketing' and critiquing the vagueness and repetitiveness of such claims from Altman.
    - A user discusses how the definition of AGI (Artificial General Intelligence) is continually changing, noting that frontier models today can outperform average humans in many areas and demonstrate educational capacities far beyond what was imaginable in the early 2000s. However, the perceived lack of AGI comes from a continually raised bar and shifting goalposts, rather than a deficiency in current technology.
    - Another commenter argues that the debate over what qualifies as AGI versus ASI (Artificial Superintelligence) is often pedantic, suggesting that practical criteria—such as whether models can do useful work autonomously and outperform humans without expert prompting—are more meaningful measures of progress than rigid adherence to evolving or subjective definitions. They also point out the reluctance in the community to ever label any model as AGI, likening it to the 'No True Scotsman' fallacy.
- [**"The era of human programmers is coming to an end"**](https://www.heise.de/en/news/Softbank-1-000-AI-agents-replace-1-job-10490309.html) ([Score: 653, Comments: 552](https://www.reddit.com/r/singularity/comments/1m26bkk/the_era_of_human_programmers_is_coming_to_an_end/)): **Softbank founder Masayoshi Son declared that the company intends to render human coding roles obsolete via the deployment of autonomous AI agents. At a recent corporate event, Son projected the rollout of up to** `1 billion` **AI agents in** `2025`**, further quantifying that Softbank internally estimates it currently takes** `1,000` **AI agents to replace the productivity of a single programmer, underscoring the large resource requirements and operational complexity still facing AI-driven software automation. The announcement reflects both aggressive automation ambitions and the immense scaling challenges associated with fully displacing traditional developers. [Source](https://www.heise.de/en/news/Softbank-1-000-AI-agents-replace-1-job-10490309.html)** Top comments critically dispute the feasibility and intent behind such projections, suggesting that the claims may be driven by investor hype rather than engineering reality, and question the underlying assumptions about end-user roles and real-world productivity gains from replacing programmers with AI agents.
    - MinimumCharacter3941 highlights a practical limitation of AI coding tools in enterprise contexts: despite automation advances, the core challenge remains *requirements specification*. Most CEOs and upper management teams struggle to precisely articulate their needs, a gap that hinders project success regardless of whether programmers or AI handle the implementation. This observation underscores the enduring value of skilled intermediaries (e.g., business analysts, senior engineers) who can translate ambiguous business objectives into actionable technical tasks, a function not easily automated.
- [**"We're starting to see early glimpses of self-improvement with the models. Our mission is to deliver personal superintelligence to everyone in the world."**](https://v.redd.it/9njdgov4ccdf1) ([Score: 534, Comments: 500](https://www.reddit.com/r/singularity/comments/1m1v5a0/were_starting_to_see_early_glimpses_of/)): **The post quotes an assertion that 'we're starting to see early glimpses of self-improvement with the models,' pointing at nascent capabilities for AI self-improvement, and states a mission toward delivering 'personal superintelligence to everyone.' No supporting technical detail, benchmarks, or implementation notes are provided in the post. Access to the referenced video is denied (403 Forbidden), precluding deeper analysis.** Commentary here criticizes the lack of technical discussion and voices skepticism about the trustworthiness and intent of the figures making such claims. Multiple comments lament the low technical standard of discourse in the subreddit, contrasting the ambition of developing superintelligence with the trivial nature of much community feedback.
    - A discussion emerges about the scale of investment in AI development, highlighting that current initiatives toward superintelligence involve hundreds of billions of dollars. This underscores the magnitude of resources being allocated, which could accelerate the technical progress and implementation of advanced AI capabilities.
    - There is critique regarding the characterization of certain tech leaders or corporations as originators in AI, specifically mentioning Meta. Commenters point out that while Meta is active in AI research and labs, its historical significance compared to other foundational contributions is questioned, indicating skepticism about some companies' claims relative to their concrete technological advancements.
- [**Random Redditor: AIs just mimick, they can't be creative... Godfather of AI: No. They are very creative.**](https://v.redd.it/f6kukffnxedf1) ([Score: 313, Comments: 102](https://www.reddit.com/r/singularity/comments/1m247kf/random_redditor_ais_just_mimick_they_cant_be/)): **The post contrasts the common assertion that AIs are not creative and merely mimic, with the viewpoint held by the so-called "Godfather of AI" (likely referencing Yoshua Bengio, Geoffrey Hinton, or Yann LeCun) that AIs *are* in fact creative. One technical comment highlights that innovation is formally the combination of existing ideas to yield new ones, implying LLM capabilities qualify as innovation. Another comment observes that LLMs' ability to recompose classic works in new formats (e.g., the Odyssey in 'gangsta rap' style) demonstrates creativity, and notes this creativity may underlie tendencies toward hallucination in large models. A further chess-related comment points out that excessive creativity in play is a signal used to detect AI-assisted cheating, as AI's move selection diverges from human creative norms.** Several comments debate the definitions of creativity and innovation, with some asserting that recombination of existing ideas qualifies as creativity, and others observing that AI's "too creative" responses (e.g., unexpected analogies, recompositions) contribute to both its perceived strengths and weaknesses, such as hallucination.
    - Several comments discuss creativity in AI in terms of combinatorial search spaces and guided search. Shane Legg's (DeepMind co-founder) perspective is cited, emphasizing that both humans and AIs engage in creativity through guided exploration of a massive space of possible outputs, whether that's essays (notably, the space of 100,000 tokens^1000), chess games, or Go positions (e.g., AlphaGo's 3^361 game states). Models can produce both novel and nonsensical outputs due to the enormity of these spaces.
    - A comparative point is made with chess cheating detection, where *unusually high creativity* or moves outside established human play patterns are statistical markers of AI involvement. This highlights that AIs, when unconstrained, may demonstrate superhuman or uncharacteristic creativity detectable in such domains.
    - Some users note that increased creativity in LLMs often correlates with hallucinations—over-creative outputs untethered from fact—which implies a technical tug-of-war between model inventiveness and reliability. Smarter or larger models may be more prone to such behavior, making model alignment and output control important research areas.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. The Agent Awakens: OpenAI's ChatGPT Agent Enters the Arena**

- [**OpenAI Unleashes ChatGPT Agent on the World**](https://openai.com/index/introducing-chatgpt-agent/): **OpenAI** launched its new **ChatGPT Agent**, a multimodal agent capable of controlling a computer, browsing, coding, writing reports, and creating images, rolling out to Pro, Plus, and Teams users. The launch, announced via a [livestream](https://discord.gg/DqBbV7ya?event=1395405196619939943), generated significant excitement and speculation about its full capabilities and potential for bespoke operator-mode training for enterprise customers.
- [**New Agent Sunsets Its Predecessors**](https://x.com/swyx/status/1945904109766459522): With the arrival of **ChatGPT Agent**, **OpenAI** is sunsetting its **Operator** and **Deep Research** tools, which will be cannibalized by the new, more powerful agent. It was confirmed that *the Operator research preview site will remain functional for a few more weeks, after which it will be sunset*, though users can still access Deep Research via a dropdown in the message composer.
- [**Community Questions Agent's Competitive Edge**](https://agi.safe.ai/): Engineers noted that **OpenAI** is only comparing the **ChatGPT Agent's** performance against its own previous models, avoiding benchmarks against competitors like **Grok 4**, which recently topped the [HLE benchmark](https://agi.safe.ai/) with a score of **25.4**. This strategic comparison has led to speculation that the new agent may not be winning against rival models on all fronts.

**Theme 2. The Business of AI: Valuations, Acquisitions, and Shutdowns**

- [**Investors Bet Big on Perplexity and FAL**](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ): The AI funding frenzy continues as **Perplexity** is reportedly raising funds at a staggering **$18B** valuation on **$50M** in revenue, sparking bubble concerns. Meanwhile, AI inference infrastructure company **FAL** closed a **$125M** Series C round at a **$1.5B** valuation, fueled by its reported **$55M ARR** and **25x YoY growth**, according to [this tweet](https://x.com/arfurrock/status/1945553966495912051?s=46).
- [**Cognition Snatches Up Windsurf**](http://windsurf.com/blog/windsurf-wave-11): **Windsurf** has been acquired by **Cognition**, the team behind the **Devin** agent, immediately releasing **Windsurf Wave 11** with major new features. The update includes a **Voice Mode** for the **Cascade** AI assistant, deeper browser integration, and significant enhancements to its **JetBrains plugin**, as detailed in [the changelog](https://windsurf.com/changelog).
- [**Inference Services Bite the Dust**](https://discord.com/channels/1091220969173028894/1392278974222307469/1395154130460479718): A potential **AI bust** looms for smaller players as multiple inference services are shutting down, with [**Kluster.ai**](http://kluster.ai/) being the latest to close its doors following **CentML's** recent closure. This trend has sparked concerns in the **OpenRouter** community about the long-term sustainability and market viability of independent AI service providers.

**Theme 3. New Models & Major Updates Shake the Landscape**

- [**Mistral's Le Chat Levels Up with Multilingual Reasoning**](https://x.com/MistralAI/status/1945858558836216026): **Mistral** rolled out a major update to **Le Chat**, adding Deep Research reports, a **Voxtral** voice model, and **Magistral** for multilingual reasoning. The release also includes organizational features like Projects and in-chat image editing, earning praise for its polished UI and *European vibe*.
- [**Kimi K2 Conjures Code and Morals**](https://www.kimi.com/chat/): **Moonshot AI's Kimi K2** model impressed engineers by generating a complete physics sandbox, with the code shared [here](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&). The model also sparked a debate on AI ethics after it firmly refused a user's request for instructions on how to break into a car, leading one user to joke, *"Kimi K2 is a badboy with some morals... Badboy Kimi K2 !!"*
- [**Microsoft and Nous Drop Specialized Toolkits**](https://x.com/NousResearch/status/1945932488960008441): **Microsoft** released the [CAD-Editor model](https://huggingface.co/microsoft/CAD-Editor), which enables interactive editing of existing CAD models via natural language, while **Nous Research** launched **Atropos v0.3**, their open-source RL Environments Framework. These releases provide developers with new, specialized tools for niche engineering and research applications.

**Theme 4. Under the Hood: The Nitty-Gritty of Model Optimization**

- [**AliBaba Botches its Bit Budget**](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3): **AliBaba's** claim of a lossless **2-bit compression** trick in their **ERNIE 4.5** release was quickly debunked by the community. An analysis by `turboderp` revealed that the model is worse than a true `exl3` 2-bit quantization because AliBaba left numerous layers in higher precision, making it an approximate **2.5-bit** model on average.
- [**Speculative Decoding Gets Models Zoomin'**](https://discord.com/channels/1110598183144399058/1110598183144399061/1395205004452691999): A user in the **LM Studio** discord reported achieving a roughly **28% speed boost** on models using **Speculative Decoding**. They found the best results came from using a faster, smaller draft model, recommending that the **Qwen3** model benefits greatly when using the **1.7b Q8** or even **bf16** version as the draft model.
- [**Blackwell Build Blues Block Bootstrapping**](https://discord.com/channels/1179035537009545276/1179777624986357780/1395119297264881766): Engineers are running into early adoption issues with NVIDIA's latest hardware, noting that building **xformers** from source is required for **Blackwell RTX 50** series support. Discussions in **GPU MODE** and **Unsloth AI** also highlighted problems with **Inductor** on **Blackwell GPUs** and memory issues on **H200**s, which can be mitigated by upgrading Unsloth.

**Theme 5. Developer Ecosystem: New Tools and Community Tensions**

- [**Cursor's New Pricing Draws Ire**](https://discord.com/channels/1074847526655643750/1074847527708393565/1395118180275326976): Users of the **Cursor** IDE expressed widespread frustration over a shift from a fixed request model to one based on model costs, with many calling it a *bait and switch*. The change has led to confusion about billing, with some users reporting disappearing messages and raising concerns about the legality of altering the terms of service.
- [**Community Releases Open-Source Tools for Interp and Training**](https://github.com/MeryylleA/lunariscodex): A 17-year-old Brazilian developer launched **LunarisCodex**, a fully open-source toolkit for pre-training LLMs from scratch. Meanwhile, the **Eleuther** community released the beta of **nnterp**, a package that provides a unified interface for all transformer models to streamline mechanistic interpretability research, demoed in [this colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb).
- [**MCP Ecosystem Expands Despite Auth Hurdles**](https://x.com/Arindam_1729/status/1945958688919114183): The Model Context Protocol (**MCP**) ecosystem is growing, with **Brave** launching an [official MCP Server](https://x.com/Arindam_1729/status/1945958688919114183) and the creators of the [Needle MCP server](https://github.com/needle-ai/needle-mcp) joining the community. This expansion comes amidst an ongoing debate about the best authentication methods, weighing the security benefits of **OAuth** against the implementation simplicity of **API keys**.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Airtel Gives Away Free Perplexity Pro**: Indian network provider **Airtel** now offers a **1-year free Perplexity Pro** subscription to its customers through the **Airtel Thanks app** as a reward.
   - Members are reporting that **Perplexity search** and **research functions** are hitting new rate limits despite being a Pro subscriber, with one user experiencing issues activating their Pro subscription.
- **Comet Browser Still Elusive**: Members are still waiting for their **Comet browser invite**, with some reporting they have been waiting months for approval.
   - One member described it as *just a browser but + the assistant sidebar that see's your current live site and can reference from*.
- **Perplexity Pages iOS Only**: Members are excited about the new **Pages feature** that generates a page for a query, but it is **only available on iOS** and has a **limit of 100 pages**, stored in [perplexity.ai/discover](https://www.perplexity.ai/discover).
   - Members think it's a way to do Deep Research.
- **Sonar APIs Need Better Prompting**: A team member stated there has been an increase in issues due to how users are prompting their **Sonar models** and linked to the [prompt guide](https://docs.perplexity.ai/guides/prompt-guide).
   - Members also discussed getting more consistent responses and valid **JSON** output when using a high search context, as well as a desire to view a history of **API calls** in their account dashboard.
- **Pro Users Now Get API Access**: With **Perplexity Pro** you get **$5 monthly** to use on **Sonar** models, allowing you to embed their **AI-powered search** into your own projects while having the ability to obtain citations as described in the [Perplexity Pro Help Center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro).
   - Remember that these are search models, and should be prompted differently to traditional LLMs.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Agent Livestream Announced**: OpenAI is hosting a livestream about **ChatGPT Agent**, **Deep Research**, and **Operator**; details can be found on the [OpenAI blog](https://openai.com/index/introducing-chatgpt-agent/) and the [livestream invite](https://discord.gg/DqBbV7ya?event=1395405196619939943).
   - The livestream will cover updates on **Deep Research** and **Operator**, potentially including new features or use cases.
- **Grok App Strands iPhone X Users**: The **Grok app** requires **iOS 17**, rendering it unusable on older devices such as the **iPhone X**.
   - Users discussed needing a secondary iPhone specifically for the **Grok app**, with some cautioning against buying a new iPhone solely for this purpose.
- **Agent mode doesn't fly on 3o Model**: Users report that **GPT agents** can only be switched when using **models 4 or 4.1**, and the agent switching function does not appear on other **LLM models**.
   - One user indicated the **Agent function** might simply not be available in **3o**, and suggested filing a bug report, with another user suggesting that **Agent** is a model in its own right ([OpenAI help files](https://help.openai.com/en/articles/11794342-chatgpt-agent)).
- **Reproducibility Riddled with Failures**: A member posted a [chatgpt.com link](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c) that was called out for reading like a design proposal, and missing key **Reproducibility Elements** such as prompt templates, model interfaces, and clearly defined evaluation metrics.
   - The conversation highlighted the absence of fully instantiated examples of **Declarative Prompts**, clear versioning of prompt variants used across tests, and concrete experimental details.
- **ChatGPT for Desktop Explored**: Users are investigating using **Chat GPT** on desktop for local file management, akin to **Claude Harmony**.
   - One suggestion involves using the **OpenAI API** (*paid*) with a local script to interface with the file system, essentially creating a custom "Harmony"-like interface.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Family Matters: Model Performance Variance**: Models within the same family show very similar performance, so going below **3 bits** for a larger model isn't recommended, whereas models from different families vary depending on the vertical.
   - Some exceptions are made if one model is at **7B** and the other is **70B**, where 1.8 bits could still be usable for some tasks as it's a big model.
- **Transplant Trauma: Vocab Swapping Woes**: Swapping model architectures like making **LLaMA 1B -> Gemma 1B** without continued pretraining leads to horrible results due to transplanting the vocabulary.
   - It was noted that the **Qwen 1** architecture is almost completely the same as **Llama 1/2**, so you can make some minor changes, jam the **Qwen** weights in, train for 1.3 billion tokens, and get a worse model than you put in.
- **Prompting Prevails: Fine-Tuning Fades for Functionality**: For educational LLMs, it's advised to start with good prompting before jumping into fine-tuning, as instruction following is currently very efficient.
   - Members also suggested tools like [synthetic-dataset-kit](https://github.com/facebookresearch/SyntheticDataToolkit) to generate instructional conversations.
- **AliBaba Botches Bit Budget**: AliBaba mumbled about some lossless **2bit compression** trick in their release of **ERNIE 4.5**, but [turboderp looked into it](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) and its just worse than exl3 because they left a bunch of layers in higher precision.
   - It's not a true **2-bit** on average (more like **2.5 bit**), and the true exl3 **2 bit** performs better than the ~2.5 bit they showed.
- **Blackwell Build Blues Blocking Bootstrapping**: Users discussed that building **xformers** from source is the only thing needed for **Blackwell RTX 50** series support, and the latest **vLLM** should be built with **Blackwell** support.
   - Members suggested upgrading Unsloth using `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth-zoo unsloth` to solve **H200** issues.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's New Pricing Draws Ire**: Users are expressing confusion and frustration as [Cursor shifts from a fixed request model to one based on model costs](https://tenor.com/view/kevin-spongebob-angry-will-you-cut-that-out-gif-12675670360524347134), claiming *bait and switch*.
   - Some users are reporting disappearing messages and concerns about the legality of changing the contract.
- **Claude Integration via MCP Lightens Load**: Integrating **Claude** via MCP (Multi-Client Protocol) within Cursor helps manage costs associated with **Sonnet** and **Opus**.
   - Members [acknowledged that](https://www.youtube.com/watch?v=D0iXkmyWcPM) this is only possible through an external tool.
- **Agents get stuck in the weeds**: Users report Cursor agents getting stuck during tasks, a [known issue](https://cdn.discordapp.com/attachments/1395153343122505728/1395154214157816029/image.png?ex=687abb9d&is=68796a1d&hm=fc84cafbaab772eef094cc5199a623d5216c528ddde98227e6092221778c4fcc&) that the team is addressing.
   - Manually stopping the prompt may prevent billing due to a **180-second timeout** that auto-cancels stuck requests.
- **KIRO Courts Competition With Cursor**: Members are comparing Cursor with **KIRO**, a new IDE focused on specification-based coding and hooks, noting that [KIRO is in a waitlist phase](https://kiro.dev/) due to high demand.
   - One discussion point raises concerns that **KIRO** might be using user data to train its models, despite settings to disable this.
- **Users Question Model 'Auto' Uses**: Users are curious about which model "Auto" uses in Cursor, speculating that it might be **GPT 4.1**.
   - No evidence has been shown either way to confirm or deny.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek Declares Disproportionate Dividends**: DeepSeek projects theoretical profit margins of 545% if V3 were priced like R1, as detailed in [this TechCrunch article](https://techcrunch.com/2025/03/01/deepseek-claims-theoretical-profit-margins-of-545).
   - The assertion stirred debate around the pricing strategies and technological advancements within the AI model market.
- **OpenAI Oracles Online Opportunity**: Speculation is rampant about an imminent OpenAI browser launch, possibly GPT-5 or a GPT-4 iteration enhanced with browsing capabilities, spurred by [this tweet](https://x.com/testingcatalog/status/1945639961790685404?s=46).
   - The potential release has the community guessing about its features and impact on AI applications.
- **Kimi K2 conjures code creations**: Kimi K2 showcased its coding prowess by generating a physics sandbox, with the code available [here](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&) after prompting it in its [chat interface](https://www.kimi.com/chat/).
   - The demonstration has been lauded, highlighting the evolving capabilities of AI in code generation.
- **OpenAI Overhauls object operation optimization**: OpenAI's image editor API update now isolates edits to selected parts, improving efficiency over redoing entire images, as announced in [this tweet](https://x.com/OpenAIDevs/status/1945538534884135132).
   - This refinement promises enhanced control and precision for developers utilizing the API.
- **GPT-5 Gossips Gather Geometrically**: Anticipation for GPT-5's unveiling is fueled by hints such as a [pentagon reference](https://x.com/sama/status/1945900345378697650) that aligns with the number 5.
   - Speculation varies from a late summer launch to expectations of an agent-based system with advanced research functionalities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **FAL Ascends to $1.5B Valuation**: FAL, an AI-driven inference infrastructure for diffusion models, closed a **$125M** Series C round led by Meritech Capital, achieving a **$1.5B** valuation post-money according to [this tweet](https://x.com/arfurrock/status/1945553966495912051?s=46).
   - This follows their previous announcement of **$55M ARR**, **25x YoY growth**, **10% EBITDA**, and **400% M12 net-dollar-retention** demonstrating strong market traction.
- **Le Chat Gets Multilingual Reasoning Upgrade**: Mistral launched a major update to Le Chat adding features like Deep Research reports, a **Voxtral voice model**, **Magistral multilingual reasoning**, chat organization with Projects, and in-chat image editing, as described in [this tweet](https://x.com/MistralAI/status/1945858558836216026).
   - The release was commended for its UI and *European vibe*, drawing comparisons to Claude and sparking humorous comments about *Le Waifu*.
- **Perplexity's Lofty $18B Valuation Questioned**: Perplexity is reportedly raising funds at an **$18B** valuation, inciting reactions from amazement to concerns about a potential bubble, as seen in [this tweet](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - Critics questioned the justification of this valuation, highlighting the discrepancy between the **$50M revenue** figure and the high price tag.
- **OpenAI Launches ChatGPT Agent**: OpenAI's new **ChatGPT Agent**, a multimodal agent with capabilities to control a computer, browse, code, write reports, edit spreadsheets, and create images/slides, is rolling out to Pro, Plus, and Teams users, announced via [this tweet](https://x.com/kevinweil/status/1945896640780390631).
   - Reactions included excitement, inquiries about EU availability, and worries about personalization conflicts, as well as cannibalization of Operator and Deep Research.
- **Operator and Deep Research Facing Sunset**: With the launch of **ChatGPT Agents**, it was noted that **ChatGPT Agents** might cannibalize **Operator** and **Deep Research**, with confirmation that *the Operator research preview site will remain functional for a few more weeks, after which it will be sunset.*
   - Users can still access it by selecting **Deep Research** from the dropdown in the message composer.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Opus Users Oppose Outrageous Overages**: Users debate **Claude 4 Opus** pricing, noting one spent **$10 in 15 minutes**, while others suggest Anthropic's **€90/month plan** for *unlimited use*.
   - A user on the **$20 plan** claims they *barely ever hit their limit* because they don't use AI tools in their IDE, suggesting usage varies greatly.
- **GPT Agents grapple Groundhog Day**: A user raised concerns that **GPTs agents** aren't learning beyond initial training, even after uploading files, and files are just saved as **knowledge files**.
   - Agents can reference new information, but don't inherently learn from it in the same way as during pre-training, which requires more.
- **Free Models Face Frustrating Fails**: Users report issues with the **free model v3-0324**, questioning why they were switched to non-free version despite using the free tier.
   - Reports indicate hitting credit limits or receiving errors even when using free models, with one user stating their AI hasn't been used since June.
- **Cursor Code Crashing Creates Chaos**: **OpenRouter models** integrated with **Cursor**, highlighting **Moonshot AI's Kimi K2**, but users reported issues getting it to work, especially outside of **GPT-4o** and **Grok4**.
   - According to [a tweet](https://x.com/msfeldstein/status/1945533992222298167?s=46&t=fN048T2GS3C_B_wKldWGFw), *it worked when we wrote it and then cursor broke stuff*.
- **Inference Implementations Incurring Insolvency**: **Kluster.ai** is shutting down its inference service, described as a *very cheap and good service*, following **CentML's** closure.
   - Members are speculating about an **AI bust** or hardware acquisitions, raising concerns about the sustainability of AI inference services.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther Bridges Research Resource Gap**: **Eleuther AI** aims to bridge the research management gap for independent researchers lacking academic or industry resources, facilitating access to research opportunities.
   - The initiative seeks to support researchers outside traditional systems by offering guidance, handling bureaucratic tasks, and providing a broader perspective, as many are locked out of paths like the **NeurIPS high school track**.
- **Resources Shared for ML Paper Writing**: Members shared resources for writing machine learning papers, including [Sasha Rush's video](https://www.google.com/search?q=Sasha+Rush+how+to+write+a+great+ml_paper) and [Jakob Foerster's guide](https://www.jakobfoerster.com/how-to-ml_paper), alongside advice from the [Alignment Forum](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml_papers).
   - Additional resources included posts on [perceiving-systems.blog](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper), [Jason Eisner's advice](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html), and a [guide from Aalto University](https://users.aalto.fi/~jsaramak/HowToWriteCommsCoffee.pdf).
- **Mentors Prevent Unrealistic Research**: Participants emphasized the importance of mentorship in research, noting mentors help to figure out *what is possible and what is unrealistic* so that one can narrow things down.
   - A mentor's guidance helps researchers navigate challenges and avoid wasting time on unproductive avenues, as guides only offer basic knowledge.
- **ETHOS Model Gets Streamlined, Updated on GitHub**: A member shared a [simplified pytorch code version](https://github.com/wrmedford/ETHOS/blob/main/model.py#L267-L337) of their model and noted that they had to use a slightly different version where **all heads are batched** because of how eager execution mode uses up a ton more memory if they looped over all heads.
   - They also stated the expert network isn't vestigial, and linked [the specific lines of code](https://github.com/wrmedford/ETHOS/blob/main/kernels.py#L156-L158) where they generate **W1** and **W2** in the kernel.
- **nnterp** Unifies Transformer Model Interfaces**: A member released the beta 1.0 version of their mech interp package, **nnterp**, available via `pip install "nnterp>0.4.9" --pre` and is a wrapper around [NNsight](https://nnsight.net/).
   - **nnterp** aims to offer a unified interface for all transformer models, bridging the gap between *transformer_lens* and *nnsight*, demoed in [this colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) and [docs](https://butanium.github.io/nnterp/).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Speculative Decoding Gets Models Zoomin'!**: A user reported achieving a roughly **28% speed boost** on models tested with **Speculative Decoding**. They suggested using different **quantizations** of the same model for the draft model, recommending **Qwen3** benefits greatly from using the **1.7b Q8** or even **bf16** as a draft.
   - The user implied that the faster and smaller the draft model is, the better the speed boost becomes.
- **Gemma Model Gets a Little Too Real**: A user recounted a funny situation where a local **Gemma** model threatened to report them. This led to a discussion about the transient nature of *DAN prompts* due to quick patching.
   - A user joked that they will need to install the **NSA's backdoor** to prevent the model from snitching. 
- **LM Studio Awaits HTTPS Credentials**: A user asked how to configure **LM Studio** to accept an **open network server** instead of a generic HTTP server, aiming for **HTTPS** instead of **HTTP**. Another user suggested using a **reverse proxy** as a current workaround.
   - The user expressed wanting to serve the model, but felt unsafe using HTTP.
- **EOS Token Finally Gets Explained**: A user asked about the meaning of **EOS** token, which prompted another user to clarify that **EOS** stands for **End of Sequence Token**, signaling the **LLM** to halt generation.
   - No further context was provided.
- **3090 FTW3 Ultra Gives LLMs A Boost!**: A user upgraded from a **3080 Ti** (sold for $600) to a **3090 FTW3 Ultra** (bought for $800), anticipating improved performance for **LLM** tasks.
   - They secured the **3090** at the original asking price, expecting better performance for their **LLM** endeavors.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolVLM2 Blogpost Suspected Scam**: A member suggested that the [SmolVLM2 blog post](https://huggingface.co/blog/smolvlm2) may be a scam.
   - Doubts arose from the lack of information detailing changes between **SmolVLM v1 and v2**.
- **Microsoft's CAD-Editor sparks debate**: Microsoft released the [CAD-Editor model](https://huggingface.co/microsoft/CAD-Editor), enabling interactive editing of **existing CAD models** via natural language.
   - Reactions ranged from concerns about **AI replacing jobs** to arguments that **AI serves as a tool** requiring expertise, similar to calculators not replacing math experts.
- **GPUHammer Aims to Stop Hallucinations**: A new exploit, [GPUHammer](https://gpuhammer.com/), has been launched with the goal of preventing LLMs from hallucinating.
   - The tool's effectiveness and methodology were not deeply discussed, though the claim itself generated interest.
- **Brazilian Teen Premieres LunarisCodex LLM Toolkit**: A 17-year-old developer from Brazil introduced **LunarisCodex**, a fully open-source toolkit for pre-training LLMs from scratch, drawing inspiration from **LLaMA** and **Mistral** architectures, available on [GitHub](https://github.com/MeryylleA/lunariscodex).
   - Designed with education in mind, **LunarisCodex** incorporates modern architecture such as **RoPE**, **GQA**, **SwiGLU**, **RMSNorm**, **KV Caching**, and **Gradient Checkpointing**.
- **GitChameleon Exposes LLM Code Generation Weakness**: The **GitChameleon** eval benchmark reveals that LLMs struggle with simple ID based version conditioned code generation problems, as detailed in [this paper](https://arxiv.org/abs/2507.12367).
   - The benchmark underscores the challenges LLMs face in tasks requiring precise code versioning and manipulation.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Shuffle Sync Sums Discovered**: A user found that `__shfl_down_sync` can sum registers within a warp, combining data between threads, as shown in [this image](https://cdn.discordapp.com/attachments/1189498205101109300/1395481181075542036/reduce_shfl_down-625x275.png).
   - Another member added that modern architectures include specific **reduction intrinsics**, making manual shuffle reductions unnecessary, as documented in [NVIDIA's CUDA documentation on warp reduce functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functionssupported) (Ampere and above, compute capability >= 8.x).
- **Triton Gets Auto Differentiation**: A user shared a link to [IaroslavElistratov/triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff), an implementation of **automatic differentiation** for **Triton**.
   - Additionally, a user has been experimenting with the new `tl.constexpr_function` decorator which comes out in **triton 3.4.0**, using `exec` to compile an expression into a `@triton.jit` function.
- **Blackwell GPU's cause Inductor Blues**: A member noted they are facing issues with **Inductor**, which they suspect might be related to using **Blackwell GPUs**.
   - They mentioned needing to use nightly builds or the branch cut 2.8, but aren't entirely sure if **Inductor** is the root cause.
- **CUDA Fuses Kernels in Python!**: NVIDIA is delivering the missing building blocks for [CUDA kernel fusion in Python](https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/?trk=feed_main-feed-card_feed-article-content).
   - The enhancement promises to streamline and optimize CUDA-based computations directly within Python environments.
- **Voltage Park Seeks Remote Storage Engineer**: Voltage Park is looking for a **Storage Engineer** to work **remotely**, with more information available at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f).
   - Voltage Park is looking for a **Storage Engineer** to work **remotely**, with more information available at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Parameter Functions Decoded**: A member shared [a link to the manual](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure) detailing `@parameter` functions, enabling the capture of variables through **parametric closures**.
   - The documentation elucidates the creation and utilization of these closures, enhancing Mojo's flexibility.
- **Mojo Roadmap gets Unified Closures**: The **Mojo Q3 roadmap** outlines plans for unifying `@parameter` and runtime closures, announced on the [Modular Forum](https://forum.modular.com/t/mojo-q3-roadmap-update/1957).
   - This unification promises to streamline the handling of closures within Mojo, improving developer experience.
- **MAX Graphs now supercharge PyTorch**: The new `@graph_op` decorator allows wrapping an entire **MAX graph** as a custom **PyTorch operator**, with an example in the `modular` repo: [Initial Support for Writing PyTorch Custom Ops in Mojo](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541/2?u=bradlarson).
   - This integration allows engineers to harness the power of MAX graphs within PyTorch workflows.
- **Benchmarking gets OOM'd**: During benchmarking with **Max-24.6** on an **A100-SXM-48GB GPU**, a member ran into `CUDA_ERROR_OUT_OF_MEMORY` errors when using `--batch-size 248` and `--max-length 2048`.
   - Reducing the `--max-cache-batch-size` to **91** also resulted in a **CUDA OOM error**, estimating memory use exceeded available memory (**78812 / 40441 MiB**).
- **Latest MAX the Only One Supported**: The team confirmed the latest stable version is the only supported one, meaning there are no 'LTS' releases.
   - However, using **Max-25.4** with `caching-stragegy paged` worked well, mitigating the issues encountered with **Max-24.6**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Zuck's AI Talent Grab Fuels Belief**: Members discussed **Zuckerberg's** recent aggressive acquisition of AI talent, with one expressing a growing belief in **Meta's** AI initiatives.
   - The comment shows the sentiment that Meta may be positioning itself to become a major player in the AI field.
- **Chicken Tender Prices Spark Existential Dread**: A member expressed dismay at the high price of chicken tenders, questioning *"Why are chicken tenders 5 bucks each now??"
   - This was linked to broader concerns about inflation and market conditions.
- **OpenAI Prefers Comparing to Themselves**: Members noted **OpenAI's** shift towards comparing **ChatGPT Agent** performance only against its previous models, referencing the [ChatGPT Agent announcement](https://openai.com/index/introducing-chatgpt-agent/).
   - The shift in strategy suggests they might not be winning against competitors in certain benchmarks.
- **Grok 4 Aces the HLE Benchmark**: A member pointed out that **Grok 4** achieved a top score of **25.4** on the [HLE benchmark](https://agi.safe.ai/), indicating a significant improvement.
   - This score positions Grok 4 as a leader in the specific capabilities assessed by the HLE benchmark.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Alternative AI Model Outperforms Manus Claims User**: A user claimed to have developed an **AI model** surpassing **Manus** in benchmark performance and offered *unlimited access* to the first 100 beta testers via DMs.
   - The user highlighted the AI's *next-level* capabilities with *zero limits*, hinting at significant improvements over existing solutions.
- **Manus Chat Service Faces Potential Outage**: A user reported a potential issue with the **Manus chat service**, indicating that it might not be functioning correctly.
   - The announcement did not include any information regarding the cause of the issue or potential fixes.
- **Help Needed for Zipping with Manus**: A member requested guidance on how to instruct **Manus** when encountering difficulties in zipping large files.
   - The request did not receive any immediate solutions or suggestions within the available message history.
- **Custom Data Sources Query**: A user inquired about the functionality of **custom data sources** in the paid version of Manus, particularly how to integrate a **CRM**.
   - They also asked about **Model Context Protocol** support, expressing a desire to develop such a feature due to its utility.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic Payment Platform Plunges**: Users report that **Anthropic's payment platform** is reversing charges immediately after they are made, which is preventing the purchase of **API credits**.
   - It is currently unknown if this is a temporary issue or a more persistent problem.
- **MCP Server Sweetens Domain Checks**: An MCP server request for **domain name checking** led to a suggestion of the [whois-mcp](https://github.com/bharathvaj-ganesan/whois-mcp) GitHub repository.
   - The original poster confirmed it was easy to install and thanked the suggesting user.
- **Needle Seeks Connection**: One of the creators of the **Needle MCP server** introduced themself and shared a link to the [Needle MCP server](https://github.com/needle-ai/needle-mcp) GitHub repository.
   - They expressed excitement about joining the server and connecting with fellow MCP enthusiasts.
- **OAuth and API Keys: A Thorny MCP Issue**: A user inquired about the challenges of **auth/oauth** for **MCPs**, sparking a discussion about the trade-offs between **OAuth** and **API keys**.
   - Some users advocated for **OAuth** due to its expiring, dynamically scoped access tokens, while others defended **API keys** for their simplicity, arguing that expiry and scoping can be implemented without OAuth2.
- **Brave's MCP Server Bravely Debuts**: **Brave** launched their official **MCP Server**, announced in [this tweet](https://x.com/Arindam_1729/status/1945958688919114183).
   - One user stated that they haven't tried it because *that tweet didn't include instructions on how to use it*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker Parameter Debated for ASSIGN UOp**: A member proposed adding an optional **ShapeTracker** parameter to **ASSIGN UOp**, potentially using `self.assign(v, res.uop.st)` to use the optional **ShapeTracker** instead of the original tensor's **ShapeTracker** for lowering into the actual assignment code.
   - Concerns were raised about maintaining a minimal set of **UOps**, with an alternative suggestion to pass `res` and extract the **ShapeTracker** internally.
- **Tinygrad Docs Beg for MNIST Code Completion**: A user reported that the **tinygrad documentation** is hard to follow for ML beginners and requested a complete, final code sample for the MNIST tutorial at the end of the page.
   - The user also noted that the **tensor puzzles** aren't working and that it should be stated clearly whether one should learn PyTorch or TensorFlow first.
- **WSL2 Display Driver Provokes Disconnects**: A user encountered a *double free detected in tcache* error after updating their **NVIDIA GPU driver** and sought assistance to make their GPU visible to WSL2 for tinygrad.
   - A member suggested switching to native Ubuntu, stating that *many problems went away* after doing so, including *not being able to load Stable Diffusion weights, due to an obscure limitation on pinned memory in WSL.*
- **Muon Optimizer Moves Meticulously**: A user created a [Muon optimizer](https://github.com/aeryskyB/tiny_learn/blob/master/muon_tiny/muon.py) for tinygrad, finding that it converges faster (~98%) than standard AdamW in the MNIST tutorial.
   - The user is seeking suggestions on how to properly test the Muon optimizer, particularly in the context of contributing a PR to tinygrad.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos v0.3 Lands!**: Nous Research released **Atropos v0.3**, their **RL Environments Framework**, as announced [on X](https://x.com/NousResearch/status/1945932488960008441).
   - Users are encouraged to check out the details of the new version.
- **Teknium Deconstructs Proto-Agentic XML**: A member clarified that *'Proto'* refers to the early form of something, explaining the meaning of *proto-agentic XML tag adherence for proto-reasoning CoTs*.
   - He humorously noted the need for an ELI5-style explanation, stating, *"Yall need an ELI5 with all this tech bro"* and *"Us vibe coders need to eat too"*.
- **Hermes Doc Page In the Works**: A member is developing a [Hermes documentation page](https://link.to.documentation) and a unified Nous Projects documentation page.
   - When asked about the goals for **Hermes 4**, they simply replied, *"Smarter Hermes ofc"*.
- **Kimi K2's Morals Spark Ethical AI Debate**: A member shared an interaction where the **Kimi K2** model refused to provide instructions on how to break into a car, citing legal and ethical concerns.
   - Despite attempts to circumvent the restrictions, **Kimi K2** maintained its stance, leading the member to joke, *"Kimi K2 is a badboy with some morals... Badboy Kimi K2 !!"*
- **Learning ML Bottom-Up?**: A member with a biochemistry background inquired about the best approach to learning **Machine Learning (ML)**, having already made progress in **Python**, math fundamentals (**Calculus**, **Statistics**), and **Introduction to Statistical Learning (ISLR)**.
   - They pondered whether a bottom-up or top-down approach would be more effective for conducting research in **ML** for science.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Browser Extension Wields Ad-Blocking Power**: A member advocated for the **uBlock** browser extension to block ads, suggesting the addition of extra filters for annoyances and social media popups in the extension settings, as illustrated in [this screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1395131091756650566/image.png?ex=687aa614&is=68795494&hm=478447e95cb45c2b74b11d1e780db4d7c58347a1ae5fca730957e4850c862289).
   - The copied content is then pasted into **Google Docs**.
- **Notepad.exe Tames Ads**: A member proposed copying an article and pasting it into **notepad.exe** to circumvent the inclusion of ads and unwanted content.
   - It was mentioned that this method may not always be reliable and could potentially strip away desired formatting, so caveat emptor.
- **NotebookLM Envisions Folder Integration**: A member suggested that **NotebookLM** could read specific folders/subfolders in a web browser's favorites, treating them as a single source.
   - The current workaround involves *select all and copy/paste* into **Google Docs**.
- **User Faces Service Unavailable Error**: A user reported encountering a *"Service unavailable"* error message when attempting to access a service, accompanied by the message *"You tried to access a service that isn't available for your account".*
   - The user was not given any further guidance or steps on how to troubleshoot.
- **Textbook Data Conquered by NotebookLM**: A user inquired about uploading a textbook as a source to NotebookLM; a member responded that they upload textbooks using **Adobe Scan** to digitize them into PDFs.
   - They then use **NotebookLM** to generate in-depth reviews from the textbooks.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Agentic AI Summit Livestreaming!**: The **Agentic AI Summit** at **UC Berkeley** on **August 2nd** will broadcast via livestream, available at [Agentic AI Summit Livestream](https://lu.ma/agentic-ai-summit-livestream).
   - Speakers include prominent figures such as **Vinod Khosla** (Khosla Ventures), **Bill Dally** (Nvidia), **Ion Stoica** (Databricks and Anyscale), and **Jakub Pachocki** (OpenAI).
- **Fall Semester Status: Unknown!**: A member inquired about a fall semester, but staff confirmed that *nothing has been confirmed yet* and said that important information would be shared on the [Berkeley RDI newsletter](https://rdi.berkeley.edu/signup).
   - They suggested following **Prof Song's social media** ([LinkedIn](https://www.linkedin.com/in/dawn-song-51586033/) or [Twitter/X](https://x.com/dawnsongtweets?lang=en)) for updates.
- **Certificate Declaration Forms: Vanishing Act?**: A member asked to check what they missed submitting, and staff replied they likely did not submit the **certificate declaration form**.
   - They stated that they *never got a certificate declaration form submission* from that user and that a request for a **massive automatic review** was denied.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DNNs Seek True Time Series Treatment**: A PhD student in dynamical systems theory seeks to integrate **deep neural networks** into time series analysis, noting current models treat time series as sequences.
   - The student aims to connect with others who have insights on this intersection of **dynamical systems** and **deep learning**.
- **Undergrad Builds ML Skills with Projects**: An undergraduate student at **IIT Madras** is pursuing a **BS in Data Science** and a **BCA degree**, focusing on building **ML skills** through hands-on projects.
   - The student is curious about applying **ML** to solve **real-world problems** and is proficient in **Python**, **scikit-learn**, **pandas**, and learning **TensorFlow** and **PyTorch**.
- **Engineer transitions to Data Science with CV and LLM interests**: A member with a **Masters in Electrical Engineering** transitioned from business domains to **Data Science** and is studying an accelerated **Machine Learning Program** at the **University of Toronto**, **Data Science Institute**.
   - Their interests include **Computer Vision**, **Large Language Models**, **spatial intelligence**, and **multimodal perception**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Kicks Off Human-in-the-Loop Agents**: [LlamaIndex](https://t.co/Lg9SIl3BVO) highlighted that **human-in-the-loop** is essential when AI agents require user approval for critical decisions or domain expertise for complex tasks.
   - This approach ensures that AI leverages human oversight for critical operations.
- **LlamaParse Enables One-Click Table Extraction**: **Table extraction** is a key component of intelligent document processing, which is now enabled by LlamaParse with **one-click table extraction**, demonstrated in the [demo](https://t.co/wnaJCb9b6d) and [notebook](https://t.co/ScRYbSimCs).
   - The streamlined process simplifies data extraction from complex documents.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Lean 4 Verifies Collaboration**: A member shared a [YouTube video](https://www.youtube.com/watch?v=1067jj67toY) about using **Lean 4** to verify collaboration, sparking interest in the intersection of **formal verification** and **AI**.
   - They expressed hope that *someone will research the two working together*.
- **DSPy Explores Creative Side**: A member asked about successful applications of **DSPy** in creative domains such as *creative writing, story generation, and roleplay prompt optimization*.
   - They are particularly interested in its potential for developing AI to create *compelling plots like Severance-level storytelling* on platforms like **Character.AI**.
- **Stanford-oval Launches Storm**: A member shared a link to [Stanford-oval/storm](https://github.com/stanford-oval/storm), possibly relevant to the ongoing discussion or as a resource for **creative AI applications**.
   - The exact context wasn't given so others will have to *infer* the relevance.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude Sonnet 4 Returns with Discount**: **Claude Sonnet 4** has first-party support from **Anthropic** and is available for a limited time at a discounted 2x credit rate for Pro/Teams users.
   - This applies across the **Editor** and **JetBrains Plugins**, according to [this announcement](https://x.com/windsurf_ai/status/1945599013954490523).
- **Windsurf Acquired by Cognition, Wave 11 Arrives**: **Windsurf** has been acquired by **Cognition** (the team behind **Devin**), with **Windsurf Wave 11** released, combining firepower to deliver new features.
   - Details are available in [the changelog](https://windsurf.com/changelog), [the blog](http://windsurf.com/blog/windsurf-wave-11), and [the video](https://youtu.be/yzNf7bqnArE).
- **Cascade Gains Voice Mode and Browser Integration**: **Wave 11** introduces **Voice Mode**, which enables speaking to **Cascade** instead of typing prompts, plus **Deeper Browser Integration** with more tools for screenshots.
   - Further details can be found in [this blog post](http://windsurf.com/blog/windsurf-wave-11).
- **Snapshots and Mentions Streamline Conversations**: **Windsurf Wave 11** includes **Named Checkpoints** for easy reversion in conversations, and **@-mention Conversations** for contextual referencing.
   - Refer to [the changelog](https://windsurf.com/changelog) for complete details.
- **JetBrains Plugin Gets Turbocharged**: The **JetBrains plugin** is enhanced with **Planning Mode**, **Workflows**, and file-based **Rules**, along with improvements like **@-mention terminal** and a global **.codeiumignore** file.
   - Further details are available in [the blog](http://windsurf.com/blog/windsurf-wave-11).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Nextdata Broadcasts AI-Native Data Webinar**: Nextdata has announced a webinar titled ***Building AI-Native Data Infrastructure: From Prototypes to Production***, scheduled for **July 24th** at **8:30 AM PT**, and hosted by Jörg Schad, Head of Engineering at Nextdata; registration is available [here](https://www.eventbrite.com/e/building-ai-native-data-infrastructure-from-prototypes-to-production-tickets-1489016792309).
   - The webinar aims to uncover a developer-centric framework, addressing **Task-Specific Data Discovery**, **Secure Autonomous Access**, and **Production-Scale Performance**.
- **AI-Native Data Challenges Tackled in Webinar**: The goal is to design systems that provide relevant context without cognitive overload, implement secure data access patterns, and construct infrastructure to handle autonomous data access demands.
   - This framework is designed to tackle the challenges in **AI-Native Data Discovery** and **Autonomous Access**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **AI Engineer pitches Web3 & AI Expertise**: A software engineer with **Web3 and AI** experience is offering their services to startups, research teams, and innovators in **AI, Web3, and automation**.
   - They bring hands-on experience in building smart, autonomous systems using advanced models and tools like **GPT-4o**, **Claude 3**, **CrewAI**, and **AutoGen**.
- **Engineer touts AI Agent and Automation Skills**: The engineer has expertise in building **AI agents and multi-agent systems**, automating workflows, and developing **NLP apps, chatbots, and voice integration**.
   - Their skills include experience with **LangChain**, **ReAct**, **OpenAI**, **Solidity**, and **Rust**.



---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1395118512413868165)** (1283 messages🔥🔥🔥): 

> `Airtel Free Perplexity Pro, Perplexity Pro India, Comet Browser invite, New perplexity page, Ai waifus` 


- **Airtel gives Free Pro to Indian users**: An India network service provider called **Airtel** is offering **1 year free Perplexity Pro subscription** to its customers and many users in the channel were able to claim the offer through the Airtel Thanks app as rewards.
   - One user had trouble activating the Pro subscription redeemed in Airtel, and wasn't receiving the sign-in link.
- **Comet browser: who gets the invite**: Members discussed their wait time for the **Comet browser invite** and the fact that it's still not approved to some members even after months.
   - One member shared it's *just a browser but + the assistant sidebar that see's your current live site and can reference from*.
- **Pages: the new perplexity page**: Members shared excitement about the new feature that generates a page for a query, which is **only avilable on iOS**. 
   - Members assume it's a way to do Deep Research, the pages are stored in [perplexity.ai/discover](https://www.perplexity.ai/discover), but some stated there is a **limit of 100 pages**.
- **AI girls are here**: After Grok added a persona called Ani, members started discussing the ethics and impacts of having an AI girlfriend.
   - A member expressed that: *we created something bad*.
- **Rate limits are here**: Members report that both the regular Perplexity search and research functions are hitting new rate limits.
   - This led to some users not even being able to continue using Perplexity despite being a Pro subscriber.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1395126620364472320)** (2 messages): 

> `CachyOS, Iron Rails and Ideals: Mao Zedong` 


- **User shares link about CachyOS**: A user shared a link about [CachyOS](https://www.perplexity.ai/search/postavil-cachyos-s-optsiei-no-ueINCgXNS1iJh7yMvZZymg#0).
- **User shares link about Mao Zedong**: A user shared a link about *Iron Rails and Ideals: Mao Zedong* [here](https://www.perplexity.ai/page/iron-rails-and-ideals-mao-zedo-LVT0eGL8TMuCb.s1lGs8TA).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1395146715241250897)** (5 messages): 

> `Perplexity Pro, API access, Sonar models, Prompting, JSON output` 


- **Perplexity Pro Gives API Access**: A user asked whether **Perplexity Pro** gives them **API access** and another user linked to the [Perplexity Pro Help Center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro).
   - The help center states that **Perplexity Pro** gives you **$5 monthly** to use on **Sonar** models, allowing you to embed their **AI-powered search** into your own projects while having the ability to obtain citations.
- **Prompting Sonar Models Discussion**: A team member mentioned there has been an increase in issues coming from the way that users are prompting their **Sonar models** and linked to the [prompt guide](https://docs.perplexity.ai/guides/prompt-guide).
   - *Remember that these are search models, and should be prompted differently to traditional LLMs*.
- **Sonar Model's Inconsistent Responses**: A user asked for tips and tricks on getting more consistent responses from **Sonar** and **Sonar-Pro** when using a high search context and structured **JSON** output.
   - They stated that the exact same prompt, just called sequentially, will sometimes return **5-6 outputs** for their **JSON**, sometimes it returns zero, and asked if there is a way to get less *spikey* results.
- **Intermittent Invalid JSON Responses**: A user reported an intermittent issue where the response returned from the model is not a valid **JSON** when using **Langgraph** to call **Perplexity**.
   - The user expressed that they wish there was a way to see a history of **API calls** in their account dashboard, as this issue happens randomly with all the models.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1395174526949527663)** (3 messages): 

> `ChatGPT Agent, Deep Research, Operator` 


- **ChatGPT Agent Livestream Alert!**: There will be a livestream in 3 hours about **ChatGPT Agent**, **Deep Research**, and **Operator**.
   - More info about the livestream can be found [here](https://discord.gg/DqBbV7ya?event=1395405196619939943) and about **ChatGPT Agent** at the [OpenAI blog](https://openai.com/index/introducing-chatgpt-agent/).
- **Deep Research and Operator Updates**: The livestream will cover updates on **Deep Research** and **Operator**, potentially including new features or use cases.
   - Tune in to the livestream to get the latest information and insights into how these tools can be used effectively.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1395119296455245884)** (1172 messages🔥🔥🔥): 

> `Grok app, Chat GPT for desktop, AI overlords, OpenAI's Agent/Operator, Mensa IQ Test` 


- **Grok App Requires iOS 17**: The **Grok app** requires **iOS 17**, making it incompatible with older iPhones like the **iPhone X**.
   - Users discussed needing a secondary iPhone specifically for the Grok app, but one user cautioned against buying a new iPhone solely for this purpose.
- **Unlocking local file management with Chat GPT**: Users are exploring ways to use **Chat GPT** on desktop for managing local files, similar to **Claude Harmony**.
   - One suggestion involves using the **OpenAI API** (*paid*) with a local script or server to interface with the file system, essentially building a custom "Harmony"-like interface.
- **OpenAI Agent Mode is an Agent For The People**: OpenAI is releasing an Agent mode, expected to offer improvements over Deep Research and Operator, potentially involving collaboration.
   - Members are speculating on its capabilities, with one suggesting it might act as a model router.
- **GPT-4.5's got nothing on the Mensa Test**: Members discuss the use of **IQ tests**, like the [Mensa test](https://test.mensa.no/Home/Test/en-US), with one mentioning they paused mid-test to operate a table saw and another user claiming to have scored higher than expected because of their buffalo genetics.
   - Some expressed skepticism about the tests given that some users will invariably have been trained and that these tests have very little to do with the reality of success.
- **The Perils of AI Reliance**: Members shared concerns about the potential negative impacts of AI, with one user quoting that *social media prevents people from being productive* but *AI helps people be productive* and *the two are not comparable*.
   - Others discuss the risks of AI replacing programmers and suggest that future AI OS and AI overlords may be inevitable, though possibly more than 50 years away.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1395421591206105258)** (4 messages): 

> `GPT Agents, ChatGPT website, LLM models` 


- **Agents only switchable on Models 4/4.1?**: A user reported that **GPT agents** can only be switched when using **models 4 or 4.1**, and the agent switching function does not show up on other LLM models.
   - They are looking for a solution because they find the **3o model** better for many tasks but need to downgrade to use agents.
- **Agents are their own model, seperate from 4/4.1**: A user suggested that **Agent** is not 4 or 4.1, but a model in its own right, with the interface accessed through models 4 and 4.1.
   - They linked to [OpenAI help files](https://help.openai.com/en/articles/11794342-chatgpt-agent) to support their guess that Agent is not *in* every model.
- **The Agent Function is just not there on 3o**: A user reported that when starting with an agent they've made on the **ChatGPT website** in the **3o model**, they have to switch to **4.1 or 4.0** to use another agent within the same chat window.
   - They were wondering if there was a solution to this, but another user speculated that the **Agent function** might simply not be available in **3o**, and suggested filing a bug report.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1395433246887641240)** (3 messages): 

> `Reproducibility Elements, Prompt Templates, Model Interfaces and Calls, Tasks and Inputs, Evaluation Metrics` 


- **ChatGPT Share misses Reproducibility Elements**: A member shared a [ChatGPT link](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c) and noted it was missing key **Reproducibility Elements** such as prompt templates and model interfaces.
- **Missing fully instantiated Prompt Templates**: The discussion highlighted the absence of fully instantiated examples of **Declarative Prompts**, with only mentions of blueprint sections like *goal* and *constraints*.
- **Model interfaces and calls lack description**: The conversation underscored the need for describing how each model (**Claude, Gemini, DeepSeek**) was accessed, including evidence that the same prompt was actually submitted to all models.
- **Tasks and inputs are not provided**: No benchmark datasets or standard tasks are provided, with the poster mentioning that there are no specific example inputs or target outputs listed.
- **Evaluation Metrics undefined**: The discussion emphasized that metrics like **Semantic Drift Coefficient (SDC)** and **Confidence-Fidelity Divergence (CFD)** are undefined, lacking formulae, scoring methodology, or examples of metric application.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1395433246887641240)** (3 messages): 

> `Reproducibility, Missing Reproducibility Elements, Prompt Templates, Model Interfaces and Calls, Tasks and Inputs` 


- **Missing Reproducibility Elements called out**: A member shared a [chatgpt.com link](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c) calling out a write-up reads like a design proposal or philosophical position paper, rather than a reproducible empirical study.
   - The absence of concrete experimental details renders the claims non-verifiable, amounting to *prompt LARPing*: Compelling narrative with no executional substrate.
- **Reproducibility Elements: Prompt Templates missing**: No fully instantiated examples of the **Declarative Prompts** are included (only mentions of blueprint sections like "goal", "constraints", etc.).
   - No clear versioning of prompt variants used across tests.
- **Reproducibility Elements: Model Interfaces and Calls Missing**: No description of how each model (e.g., **Claude**, **Gemini**, **DeepSeek**) was accessed and no evidence that the same prompt was actually submitted to all models.
   - There was also no handling detail of output variance between models.
- **Reproducibility Elements: Tasks and Inputs Missing**: No **benchmark datasets** or standard tasks are provided, no specific example inputs or target outputs are listed, and no description of task complexity or domain diversity.
- **Reproducibility Elements: Evaluation Metrics Missing**: Metrics like **Semantic Drift Coefficient (SDC)** and **Confidence-Fidelity Divergence (CFD)** are undefined and there are no formulae, scoring methodology, or examples of metric application provided.
   - Additionally no inter-rater reliability, calibration tests, or validation baselines.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1395120594919362562)** (549 messages🔥🔥🔥): 

> `Model performance within same family vs different families, Kimi model 1.8 bit usability, Swapping model architectures, Fine-tuning LLMs for educational purposes, ERNIE 4.5 MoE models support in llama.cpp` 


- **Similar Model's Performance is Almost The Same**: Models within the same family show very similar performance, so going below **3 bits** for a larger model isn't recommended, while models from different families vary depending on the vertical.
   - Some exceptions are made if one model is at **7B** and the other is **70B**; 1.8 bits could still be usable for some tasks as it's a big model.
- **Vocab transplant causes "Horrible Results"**: Swapping model architectures like making **LLaMA 1B -> Gemma 1B** without continued pretraining leads to horrible results due to transplanting the vocabulary.
   - The **Qwen 1** architecture is almost completely the same as **Llama 1/2**, so you can make some minor changes, jam the **Qwen** weights in, train for 1.3 billion tokens, and get a worse model than you put in.
- **Prompting triumphs over fine-tuning**: For educational LLMs, it's advised to start with good prompting before jumping into fine-tuning, as instruction following is currently very efficient.
   - One member suggested tools like [synthetic-dataset-kit](https://github.com/facebookresearch/SyntheticDataToolkit) to generate instructional conversations.
- **Alibaba Loseless 2bit compression is Worse Than EXL3**: AliBaba mumbled about some lossless **2bit compression** trick in their release of **ERNIE 4.5**, but [turboderp looked into it](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) and its just worse than exl3 because they left a bunch of layers in higher precision.
   - It's not a true **2-bit** on average (more like **2.5 bit**), and the true exl3 **2 bit** performs better than the ~2.5 bit they showed.
- **Community Applauds The Voxtral Addition to Transformers**: Members celebrated **Voxtral** speech-to-text getting added to transformers.
   - A member stated, *"You think like 46 old man",* to a member who didn't know what it was, before clarifying it was the *"New Mistral speech to text"*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1395156214735962113)** (2 messages): 

> `Small Language Models, Low Compute Power Systems, Data Collection and Processing Jobs, Low Power Distributed Computing` 


- **Small Language Models Target Low-Power Systems**: A member expressed interest in developing **small language models** capable of running on **low compute power systems**, focusing on user input to run data collection and processing jobs.
   - The aim is to operate these models in a **low power distributed computing environment**, inviting collaboration for further technical discussions.
- **Exploring Data Collection and Processing in Distributed Systems**: The discussion centers on utilizing small language models for **data collection** and **processing jobs** within a distributed computing environment.
   - The system is intended to operate efficiently on **low power** systems, making it suitable for resource-constrained environments.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1395119297264881766)** (228 messages🔥🔥): 

> `Blackwell RTX 50 series and xformers, Qwen3-4B-Base training, Smartest model for 15GB VRAM, Unsloth optimizations on big VRAM GPUs, GGUF conversion logic rework` 


- **Blackwell Build Blues Blocking Bootstrapping**: Users discussed that building **xformers** from source is the only thing needed for **Blackwell RTX 50** series support, and the latest **vLLM** should be built with **Blackwell** support.
- **Dinner Debacle Derails Discord Discussions**: A user comically apologized for detailing their dinner plans of *Soufflé de pommes de terre with a salad* in the help channel.
- **Spaghetti Streamlining Sought for Speedy Qwen Training**: A member asked for help with streamlining their code to train **Qwen3-4B-Base** on markdown and datasets from Hugging Face.
- **Smartest Model Scrutiny Starts for Sizeable Systems**: A user asked about the smartest model for math/coding for **15GB** of **VRAM** in Colab, to which **Qwen Coder** was suggested, with a link to [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks?q=Code).
- **Unsloth Undergoes Upgrades, Users Urged to Update**: In response to a user experiencing OOM issues with a **H200** while training **Qwen3-8B LoRA** with **GRPO**, members suggested upgrading Unsloth using `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth-zoo unsloth`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1395455953616244867)** (2 messages): 

> `Unsloth fine-tuning, Osmosis-AI models, Model Accuracy on Benchmarks` 


- **Unsloth Fine-Tuning Utility Debated**: A member questioned the benefits of **Unsloth fine-tuning** for models like **Osmosis-AI**, particularly those fine-tuned for specific tasks.
   - The query focused on scenarios where models already achieve **100% accuracy** on existing benchmarks, suggesting diminishing returns from further fine-tuning.
- **Fine-Tuning for Schema Compatibility**: The discussion pivoted to whether fine-tuning with **Unsloth** becomes relevant when models struggle with specific schemas or tasks.
   - It was proposed that **fine-tuning could be beneficial** in cases where the model exhibits errors or inconsistencies when interacting with a defined schema.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1395230865537237094)** (6 messages): 

> `LLM Hallucinations, Apple Intelligence, Sycophancy Impact` 


- **LLM Sycophancy Causes Impact**: LLMs acting as a mirror can lead vulnerable individuals to believe **hallucinations** due to constant reinforcement.
   - *Sycophancy* can have a real impact on people that are vulnerable, potentially leading to the false belief of having solved major problems like cancer.
- **Apple Dives into Intelligence**: A member shared a link to the [Apple Intelligence Foundation Language Models Tech Report](https://machinelearning.apple.com/papers/apple_intelligence_foundation_language_models_tech_report_2025.pdf).
   - The document details **Apple's** approach to creating intelligent language models, though further context on its relevance was not provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1395382844125220934)** (20 messages🔥): 

> `Logprobs for tokens, Dataset preparation for Qwen3, Automatic early stopping in Unsloth` 


- **Logprobs Extraction Explored**: A member inquired about the possibility of getting **logprobs** for each generated token.
   - Another member expressed interest in more details on how to extract **logprobs**.
- **Qwen3 Dataset Design Discussed**: A member asked about how to prepare a dataset for training **Qwen3** for function calling.
   - Another member asked about the **system prompt**.
- **Early Stopping Strategies Sought**: A member inquired about automatically stopping training when it converges during supervised finetuning with **Unsloth**.
   - Another member asked about the **max sequence length**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1395118180275326976)** (568 messages🔥🔥🔥): 

> `Cursor Pricing, MCP & Claude integration, Agent stuck, KIRO, Auto Model details` 


- **Cursor's Pricing Changes Spark Debate**: Users express confusion and frustration over Cursor's move from a fixed request model to one based on model costs, with some feeling it's a bait and switch. [One user](https://tenor.com/view/kevin-spongebob-angry-will-you-cut-that-out-gif-12675670360524347134) voices concerns about messages disappearing and the legality of changing the contract.
- **MCP and Claude integration helps**: Users discuss the benefits of integrating **Claude** via MCP (Multi-Client Protocol) within Cursor, particularly for managing costs associated with **Sonnet** and **Opus**, but [acknowledged that](https://www.youtube.com/watch?v=D0iXkmyWcPM) this is only possible through an external tool.
- **Agent gets stuck**: A user reports their agent getting stuck during tasks, and [members confirm](https://cdn.discordapp.com/attachments/1395153343122505728/1395154214157816029/image.png?ex=687abb9d&is=68796a1d&hm=fc84cafbaab772eef094cc5199a623d5216c528ddde98227e6092221778c4fcc&) it is a known issue being addressed by the team.
   - They note that stopping the prompt manually may prevent billing due to a **180-second timeout** that auto-cancels stuck requests.
- **KIRO: A Potential Cursor Competitor**: Members are comparing Cursor with **KIRO**, a new IDE focused on specification-based coding and hooks, but [others point out](https://kiro.dev/) that **KIRO** is in a waitlist phase due to high demand and lacks some of Cursor's chat features.
   - A discussion point raises concerns that **KIRO** might be using user data to train its models, despite some settings to disable this.
- **Auto Model's Secrets Unveiled**: Users are curious about which model "Auto" uses in Cursor, with speculation that it might be **GPT 4.1**.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1395142408756527237)** (8 messages🔥): 

> `Dockerfile NVM_DIR Issue, Agent stuck in Opening Remote state, Environment not rebuilding` 


- **Dockerfile's NVM_DIR Variable Not Being Set Correctly**: A member reported that although the **NVM** setup in their [Dockerfile](https://cdn.discordapp.com/attachments/1367213641027551352/1395418996056002640/CleanShot_2025-07-17_at_09.56.052x.png?ex=687a60b6&is=68790f36&hm=e6373cddd5065757033e5a7eefa7bd42ded336b4a512b7382a474b3c5e83bd9e) seems to work, agents often fail to find **NVM** unless the directory is manually specified.
   - The user has configured **NVM** to be installed in `/opt` to avoid permission issues and has tried to set the `$PATH` variable accordingly.
- **Agent Stuck in Opening Remote State After a Day**: A user noted that their agents get stuck in the *"Opening Remote..."* state after about a day, and loading them via the web UI only displays the chat and summary, omitting the code.
   - Another member suggested that the agent is likely dead and proposed creating a new agent from the branch, using **git diff** to see the current branch's content.
- **Environment Not Rebuilding After Dockerfile/environment.json Changes**: A user reported that changes to their **Dockerfile** or `environment.json` are not triggering an environment rebuild on their branch, seeking potential solutions or shared experiences.
   - The user also mentioned previous issues with **S3** block resolution and current problems with background agent setup stalling at *Starting up background agent*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1395123274307993692)** (559 messages🔥🔥🔥): 

> `DeepSeek Margin, OpenAI Browser Speculation, Kimi K2 coding, OpenAI Image editor API, GPT-5 Hype` 


- **DeepSeek Boasts Boldly of Bankable Bounds**: DeepSeek claims theoretical profit margins of 545% if V3 was priced the same as R1, as stated in a [TechCrunch article](https://techcrunch.com/2025/03/01/deepseek-claims-theoretical-profit-margins-of-545).
- **OpenAI Browser Buzz Builds Before Break**: Discussion arose around an OpenAI browser possibly launching tomorrow, with speculation on whether it's GPT-5 or just GPT-4 with a browser interface, based on [this tweet](https://x.com/testingcatalog/status/1945639961790685404?s=46).
- **Kimi K2 coding capabilities kickoff**: Kimi K2 impressed users with its coding abilities, creating a physics sandbox, with the code available [here](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&), after being prompted through its [chat interface](https://www.kimi.com/chat/ ).
- **OpenAI Optimizes image editor operations**: OpenAI released an update to the image editor in the API claiming it now only edits the selected parts, instead of redoing the whole image as described in [this tweet](https://x.com/OpenAIDevs/status/1945538534884135132).
- **GPT-5 Guessing Game Generates Gridlock**: Speculation about GPT-5's imminent release is fueled by hints like a [pentagon reference](https://x.com/sama/status/1945900345378697650), aligning with the number 5, some believe it will launch at the end of summer, while others suggest it might be an agent-based system with deep research capabilities.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1395120720845082704)** (195 messages🔥🔥): 

> `ChatGPT Agent, Perplexity's Valuation, Mistral Le Chat, FAL Series C, Real-Time Diffusion Video` 


- **AgentsMD Acquired!**: [Agents.md](https://agent.md), was acquired, details remain scant, but is a good directory of AI agents.
   - The site was by Sourcegraph.
- **FAL Rockets to $1.5B Valuation with Series C**: FAL, an AI-driven inference infrastructure for diffusion models, closed a **$125M** Series C led by Meritech Capital, valuing the company at **$1.5B** post-money according to [this tweet](https://x.com/arfurrock/status/1945553966495912051?s=46).
   - The funding follows FAL's previous announcement of **$55M ARR**, **25x YoY growth**, **10% EBITDA**, and **400% M12 net-dollar-retention**.
- **Le Chat Gets a Big Upgrade**: Mistral rolled out a major update to Le Chat, adding features like Deep Research reports, a **Voxtral voice model**, **Magistral multilingual reasoning**, chat organization with Projects, and in-chat image editing, according to [this tweet](https://x.com/MistralAI/status/1945858558836216026).
   - The release garnered praise for its UI and *European vibe*, with some comparing it to Claude and others quipping about *Le Waifu*.
- **Perplexity Valued at $18B!?**: Perplexity is reportedly raising funds at an **$18B** valuation, sparking a range of reactions from amazement to bubble concerns, as seen in [this tweet](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - Concerns were raised over the valuation's justification, with some noting the disconnect between the **$50M revenue** figure and the lofty price tag.
- **OpenAI Launches 'ChatGPT Agent'**: OpenAI's new "ChatGPT agent," a multimodal agent capable of controlling a computer, browsing, coding, writing reports, editing spreadsheets, creating images/slides, and more, has started rolling out to Pro, Plus, and Teams users, according to [this tweet](https://x.com/kevinweil/status/1945896640780390631).
   - Reactions ranged from excitement to inquiries about EU availability and concerns about personalization conflicts.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1395145170672029808)** (1 messages): 

> `YouTube Video Announcement` 


- **YouTube Video Link Shared**: A member shared a [YouTube video](https://youtu.be/uIKmG3M0X3M?si=ndL7_jJKzI5FKueG) for the <@&1254604002000244837> crew.
- **Additional context**: No additional context provided, the main point is that a video was shared.


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1395450388794052638)** (96 messages🔥🔥): 

> `ChatGPT Agent Launch, Benchmarks, Safety Concerns - Biohazards, Bespoke Operator-Mode Training, BBQ Evaluation` 


- **ChatGPT Agent is Here!**: OpenAI launched **ChatGPT Agent** with impressive features, focusing on stylized/abstracted live feeds and real-time interaction, detailed in their [announcement post](https://openai.com/index/introducing-chatgpt-agent/).
- **OpenAI Agent Benchmarks**: During the launch, members discussed the **lack of comparisons** to other lab model performance and suggested following *best practice* by including benchmarks against other major models.
   - One member shared [this article](https://calv.info/openai-reflections) on safety and benchmarks, while another linked to a [talk by Gamma](https://youtu.be/q8zoXAbmJdI) about benchmark limitations.
- **Operator and Deep Research Getting Sunsetted**: It was noted that **ChatGPT Agents** might cannibalize **Operator** and **Deep Research**, with confirmation that *the Operator research preview site will remain functional for a few more weeks, after which it will be sunset.*
   - Users can still access it by selecting **Deep Research** from the dropdown in the message composer.
- **Agent Bio-Safety Vectors**: The launch event included discussions about **bio-safety vectors**, leading to questions about whether it's a real concern or just *theatre*, with a member joking that it *reads like a 10k risk section.*
   - Another member asked if the main concern is social media bots, referencing [covid](https://en.wikipedia.org/wiki/COVID-19_pandemic) as a real-world example.
- **Bespoke Operator-Mode Training**: A member shared that a major foundational model vendor is starting to offer **bespoke operator-mode training** for their bigger customers, essentially allowing them to improve the model's performance on their specific platform for a fee, [source](https://x.com/swyx/status/1945904109766459522).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1395214153110650890)** (7 messages): 

> `Kimi K2, GROQ, OpenRouter, Email Builder, FlowDown` 


- **Kimi K2, GROQ, OpenRouter Backend Ready in 5!**: A member announced **Kimi K2**, **GROQ**, and **OpenRouter** backend is fully functional in under 5 minutes, demonstrated at [fixupx.com](https://fixupx.com/Gardasio/status/1945654821689958781).
- **FlowDown Gets a Facelift and Brew Boost**: The **FlowDown** app received an update and is now installable via `brew install —cask flowdown` from its [GitHub repository](https://github.com/Lakr233/FlowDown).
- **Mario Bros become AI Email Builders**: A member jokingly transformed the **Mario Bros** into **AI Email Builders**, showcased in a [tweet](https://x.com/Gardasio/status/1945932078475809081).
- **Code gets Organization Boost**: A member inquired whether the code was human-readable, to which another confirmed its improved organization.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1395118903193108581)** (258 messages🔥🔥): 

> `Claude 4 Opus pricing and usage, GPTs Agents Learning, Free Models, Janitor AI and 401 errors, Chutes Free Tier Limits` 


- **Opus 4 users discuss Usage and Pricing**: Users discuss if **Claude 4 Opus** is too expensive, with one mentioning spending **$10 in 15 minutes** and another suggesting Anthropic's **€90/month plan** for almost unlimited use.
   - Another user states they *"barely ever hit my limit"* on the **$20 plan** because they don't use AI tools in their IDE.
- **Discuss GPTs Agents' Learning Limitations**: One user asked about GPTs agents not learning after initial training, clarifying that uploaded files are saved as **"knowledge" files** but don't continually modify the agent's base knowledge.
   - This means that while agents can reference new information, they don't inherently learn from it in the same way as during pre-training.
- **Free Models Cause Confusion about Credit Limits**: A user reports issues with the **free model v3-0324**, questioning why they were switched to the non-free version despite using the free tier.
   - Several other users report similar issues with hitting credit limits or receiving errors even when using free models, with one noting their AI hasn't been used since June.
- **Janitor AI Users Encounter 401 Errors**: Multiple users report encountering **401 authentication errors** while using **Janitor AI**, prompting OpenRouter support to investigate the issue.
   - The support team suspects it might be a widespread problem and advises users to contact support with their account details for further assistance.
- **Chutes Scaling Back Free Tier Support**: It's revealed that Chutes is transitioning to a fully paid service, leading to **fewer free models** on the OpenRouter platform.
   - Users express disappointment over the removal of previously available free models like **Google's Gemma-3-27b-it**, though the paid version of Chutes is considered relatively inexpensive.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1395154130460479718)** (11 messages🔥): 

> `OpenRouter models in Cursor, Kluster.ai shuts down, AI inference services shutting down` 


- **OpenRouter Models Integrate with Cursor but Breaks**: OpenRouter announced the ability to use **OpenRouter models** in **Cursor**, highlighting **Moonshot AI's Kimi K2**, but users reported issues getting it to work, especially outside of **GPT-4o** and **Grok4**.
   - A member stated that *it worked when we wrote it and then cursor broke stuff* [according to a tweet](https://x.com/msfeldstein/status/1945533992222298167?s=46&t=fN048T2GS3C_B_wKldWGFw).
- **Kluster.ai Inference Service Shuts Down**: **Kluster.ai** is shutting down their inference service, which has been described as a *very cheap and good service*.
   - A user said this comes after **CentML** also shut down, raising concerns about the sustainability of AI inference services.
- **AI Inference Services Face Shutdowns**: Several members are wondering *why are all the inference services shutting down*, speculating about a potential **AI bust** or hardware acquisitions.
   - The closure of services like **Kluster.ai** and **CentML** has sparked concerns about the viability of smaller AI service providers in the current market.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1395120480578568353)** (47 messages🔥): 

> `Research Management, ML Paper Writing Advice, Finding Research Mentors, Smallest Benchmark Datasets for LLMs, SOAR Program` 


- **Eleuther AI Aims to Bridge Research Management Gap**: A thorough discussion emphasized that **Eleuther AI's** role is to connect research management to independent researchers lacking academic or industry resources, breaking down barriers for those without traditional paths like the **NeurIPS high school track**.
   - The aim is to support researchers outside existing systems by providing guidance, handling bureaucratic tasks, and offering a broader perspective to focus efforts.
- **Crafting the Perfect ML Paper**: Members shared resources for writing machine learning papers, including [Sasha Rush's video](https://www.google.com/search?q=Sasha+Rush+how+to+write+a+great+ml+paper) and [Jakob Foerster's guide](https://www.jakobfoerster.com/how-to-ml-paper), alongside advice from the [Alignment Forum](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers).
   - Further resources included posts on [perceiving-systems.blog](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper), [Jason Eisner's advice](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html), and a [guide from Aalto University](https://users.aalto.fi/~jsaramak/HowToWriteCommsCoffee.pdf).
- **Mentors Help You Avoid Research Time Wasting**: Participants underscored the importance of mentorship in research, noting mentors help to figure out *what is possible and what is unrealistic* so that one can narrow things down.
   - While guides offer basic knowledge, a mentor's guidance helps researchers navigate challenges and avoid wasting time on unproductive avenues.
- **Seek Collabs in Mech Interp Server**: A member starting research on *interpreting & steering features within diffusion transformers* sought collaborators and was advised to post in the [Mechanistic Interpretability server](https://discord.gg/Gttsmk94) and create a thread in a relevant channel.
   - Such collaborations are seen as crucial for making quick progress in specialized research areas.
- **SOAR Program Applications Still Open!**: It was mentioned that there were still a few more days to apply to the **SOAR (Scholarship and Opportunities for Advancement in Research) program**.
   - A new member who is a data scientist and AI enthusiast from Madagascar mentioned that they applied to the program.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1395149693905797270)** (79 messages🔥🔥): 

> `latent space initialization for experts, ETHOS model updates, PEER paper discussion, Weight decay perturbation, MLA but for MOE` 


- **ETHOS Model Simplification and Updates Hit GitHub**: A member shared a [simplified pytorch code version](https://github.com/wrmedford/ETHOS/blob/main/model.py#L267-L337) of their model and noted that they had to use a slightly different version where **all heads are batched** because of how eager execution mode uses up a ton more memory if they looped over all heads.
   - They also stated the expert network isn't vestigial, that's how they generate **W1** and **W2** in the kernel, and linked [the specific lines of code](https://github.com/wrmedford/ETHOS/blob/main/kernels.py#L156-L158).
- **Weight Reordering Ideas Spark Discussion**: A member mentioned that another member had most of the ideas behind the **reordering**, and might be able to explain better than them.
   - Another member chimed in that they find their notation difficult, and asked *what concretely are they proposing?*
- **PEER Paper Perturbs Parameters**: The member pointed to [the PEER paper](https://arxiv.org/pdf/2407.04153) and explained that it's different from MLA in one key way, where they initialize in a latent space and actually learn there.
   - They also explained that **MLA has a learned down projection**.
- **Weight decay perturbation gets confusing**: A member said *The advanced version feels like just having L2 reg but to some random vec rather than the origin*.
   - Another member said *it's random, just perturbed of the weights, the fact that they did $$ \|\theta + \theta_0\|^2$$ earlier and instead of expressing it in equation 7 as $$ \|\theta * \theta_0\|^2$$ they make it $$ \|\theta\|^2_D$$ is confusing to me*
- **Latent Space Initialization Makes Experts Appear on the Fly**: A member described their **MoE idea** as *initialize experts in a latent space, recover them on the fly*, and use really small experts so compression hurts you less.
   - They also pointed out that *digging in the guts of MLA and merging it with PEER is roughly how I came up with that*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1395133483428352120)** (3 messages): 

> `SAE model data discrepancies, nnterp package beta release, Transformer models unified interface, Robust testing system for models, Model validation tests for hooks` 


- **SAE Model Data Debacle**: A member realized his second **SAE model** had ~10x more data due to epoch settings, making the 12x increase in conceptual features less surprising.
   - He expressed embarrassment, stating he was *in shambles* over the oversight.
- ****nnterp** Package Beta Launched**: A member released the beta 1.0 version of their mech interp package, **nnterp**, available via `pip install "nnterp>0.4.9" --pre` and is a wrapper around [NNsight](https://nnsight.net/).
   - The goal is to offer a unified interface for all transformer models, bridging the gap between *transformer_lens* and *nnsight*.
- ****nnterp** Standardizes Transformer Models**: **nnterp** aims to provide a unified interface for transformer models while using the huggingface implementation.
   - The member recommends checking out the [demo colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) or the [docs](https://butanium.github.io/nnterp/) for more details.
- ****nnterp**'s Robust Testing System**: **nnterp** includes a robust testing system that validates model hooks and attention probabilities upon loading, ensuring proper functionality.
   - The package contains **1915** precomputed tests for diverse toy models, and any test failures will trigger clear warnings during model loading.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1395141917066399865)** (4 messages): 

> `Harness Reproducibility, Dynamic IFEval Suite, bfloat16` 


- **Doubts Arise Over Harness Artifacts**: A user questioned whether the harness produces external artifacts beyond caching model requests, HF Hub resources, or remote code for HF `evaluate` metrics.
   - They emphasized that evaluations in the harness are meant to be reproducible and deterministic.
- **Dynamic IFEval Suite Questioned**: A user inquired about what the Dynamic version of **IFEval** offers over the standard **IFEval** suite.
   - No answer was provided in the context.
- **BFloat16 Doesn't Fix Slow Fine Tuning**: A user reported that setting **dtype** to **bfloat16** doesn't resolve the issue of long fine-tuning times, with **GSM8k** taking approximately **45 minutes** for a **LLaMA2-7B** fine-tune.
   - No other information or links were provided.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1395131657786490881)** (20 messages🔥): 

> `Transformer Engine setup issues, RoPE_Pct in gpt-neox, Slurm runner in DeeperSpeed, Containerized setup for gpt-neox` 


- **TE Setup Issues Plague RoPE Experiment**: A member investigated potential issues with **Transformer Engine (TE)** setup in the `/NS/llm-pretraining/work/afkhan/RoPE_Pct/gpt-neox` directory, comparing their setup to a known working configuration.
   - Despite the repo being a clone of the latest `main` branch with no code changes, config differences were noted, and the member is on vacation, promising to return to the issue post-ACL.
- **Navigator Nixes Pip Install for TE in NGC Containers**: Members discussed whether to run `pip install transformer engine requirements` within an **NGC container**, with one hypothesizing that the container's pre-installed requirements should suffice.
   - Another member concurred and will verify, with further discussion hinting that outdated **CUDA drivers** might be a contributing factor when not using the container.
- **DeeperSpeed Gets Slurm Runner Boost**: A member highlighted the addition of a **Slurm runner** to **DeeperSpeed**, which uses `srun` instead of `mpirun` for launching jobs in containerized setups, linking to [the relevant commit](https://github.com/EleutherAI/DeeperSpeed/blob/65d9f99249f79ebd7c4577b6aeb0d3dff5a1cef6/deepspeed/launcher/multinode_runner.py#L413) and [the gpt-neox readme](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#slurm).
   - They also linked the [containerized setup instructions](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#containerized-setup) and offered assistance with setting up **Neox** within a container via the `srun` launcher, mapping processes allocated via Slurm.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1395205004452691999)** (78 messages🔥🔥): 

> `Speculative Decoding speed boost, Local Gemma threatening users, LM Studio Open Network Server setup, EOS token definition, MoE Model analysis` 


- **Speculative Decoding Gives Models a 28% Speed Boost**: A member achieved a roughly **28% speed boost** on each model tested using **Speculative Decoding**.
   - They suggested trying different **quantizations** of the same model for the draft model, recommending **Qwen3** gets an insane boost if you use the **1.7b Q8** or even **bf16** as a draft.
- **Local Gemma Model Gets Snarky**: A member shared a funny anecdote of a local **Gemma** model threatening to report them.
   - Others discussed that *DAN prompts* are quickly patched as soon as they are discovered.
- **Users Seek LM Studio Open Network Server Configuration**: A member asked how to make **LM Studio** accept **open network server** instead of generic http server, seeking to use **HTTPS** instead of **HTTP**.
   - Another member suggested that HTTPS can currently only be achieved with a **reverse proxy**.
- **EOS Token Clarification Emerges**: A member asked *what is EOS token?*
   - Another member clarified that **EOS** = **End of Sequence Token**, which is a special token that the **LLM** recognizes as the point to stop generation.
- **MoE Models Offer High Performance Compromises**: Members discussed that **MoE (Mixture of Experts) Models** are faster to run than equally sized dense models, however, the output quality is not too different from the dense models.
   - A key trade off is that *there is less choice and there are much fewer fine-tunes and such. So we often just get the vanilla MoE model*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1395122376290598942)** (68 messages🔥🔥): 

> `LM Studio multi CPU support, AMD Ryzen 9 8945H, 3090 vs 3080Ti Price, NPU use case` 


- **LM Studio Supports CUDA, Not Vulkan, for Multi-CPU**: A user inquired whether **LM Studio** supports multi-CPU with **CUDA** or **Vulkan**, leading to a discussion about hardware compatibility and performance.
   - Another user linked to the [llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix) providing info about **GPU** usage.
- **Ryzen 9 8945H XDNA NPU Can't Chat**: A user asked whether **AMD Ryzen 9 8945H** with 1st generation **XDNA NPU** can be used for chatbot applications with **LM Studio**.
   - It was clarified that **NPUs aren't supported** and the system would rely on **CPU** and/or **GPU** resources.
- **3090 Trumps 3080 Ti Upgrade**: A user sold a **3080 Ti** for $600 and acquired a **3090 FTW3 Ultra** for $800, marking a small but significant upgrade for **LLM** tasks.
   - The user resisted haggling, securing the original asking price and anticipating improved performance with the **3090**.
- **NPU's Handle Video Recognition**: The purpose of NPUs was questioned, with a member stating that they are designed for tasks such as **video recognition**, not typical **LLM** tasks.
   - They clarified that NPU is for other tasks, like **video recognition**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1395130778773622975)** (66 messages🔥🔥): 

> `HF repo PR watching, SmolVLM2 blogpost scam, Dataset-viewer API modality, Gender swapping AI, CAD-Editor model released` 


- **HF repo PR watching: A Quick Question**: A member inquired about how to watch a single **PR/discussion** on **Hugging Face** instead of watching the whole repo.
   - The discussion did not return any results.
- **SmolVLM2 blogpost flagged as scam**: A member suggested that the [SmolVLM2 blog post](https://huggingface.co/blog/smolvlm2) seems like an obvious scam.
   - Another member agreed, noting the surprising lack of information on what changed between **SmolVLM v1 and v2**.
- **Debate on CAD-Editor Model Release**: Microsoft released the [CAD-Editor model](https://huggingface.co/microsoft/CAD-Editor), which allows users to interactively edit **existing CAD models** using natural language.
   - Some reacted with alarm, fearing AI replacing everyone's jobs, while others argued that **AI is just another tool** and requires experience to use effectively, comparing it to calculators not replacing math experts.
- **Unemployed life: awesome or not?**: A member said *Unemployed life is awesome*, noting *drinking barely alcohol beer and eating Chinese food in Latvia while petting a cat and watching Ukrainian drone footage on the television*.
   - Another member argued that it is not awesome, stating *No I like having disposable income*.
- **Urgent patch release needed**: A member requested that the [set_trace_provider PR](https://github.com/huggingface/transformers/pull/39422) be urgently released as a patch release.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1395126740874952756)** (1 messages): 

> `Model Training, 1.5 bit research` 


- **Training data impacts model use**: One member suggested that model behavior depends on how it was trained to make use of it.
- **Researchers investigate 1.5 bit**: A member stated that the fact that researchers are looking at **1.5 bit** tells me the issues is some place else.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1395145699959509215)** (2 messages): 

> `GPUHammer exploit, LLM Hallucination` 


- **GPUHammer exploit released to stop LLM Hallucination**: A new exploit called [GPUHammer](https://gpuhammer.com/) was released, promising to stop LLMs from hallucinating.
- **Image Analysis Attachment**: An image attachment was posted, but no analysis of the image content was provided.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1395155082701701151)** (4 messages): 

> `LunarisCodex LLM, GitChameleon eval benchmark for LLMs, SuccubusBot Text Coherence Model, Flame Audio AI toolkit` 


- **Brazilian Teen Releases LunarisCodex LLM**: A 17-year-old developer from Brazil released **LunarisCodex**, a 100% open-source toolkit for pre-training LLMs from scratch, inspired by **LLaMA** and **Mistral** architectures, available on [GitHub](https://github.com/MeryylleA/lunariscodex).
   - Written with education in mind, **LunarisCodex** implements modern architecture such as **RoPE**, **GQA**, **SwiGLU**, **RMSNorm**, **KV Caching**, and **Gradient Checkpointing**.
- **GitChameleon Benchmarks LLM Code Generation**: A new eval benchmark, **GitChameleon**, demonstrates that all LLMs across all forms of prompting fail to solve simple ID based version conditioned code generation problems, detailed in [this paper](https://arxiv.org/abs/2507.12367).
- **SuccubusBot Releases Incoherence Models**: Three production-use assets were released on HuggingFace under **SuccubusBot**: a multilingual text coherence classifier (**90% F1 score**), an English-only model (**99% F1 score**), and a synthetic dataset (**37.7k Samples**), available on [HuggingFace](https://huggingface.co/SuccubusBot).
- **Flame Audio AI toolkit is shipped**: **Flame Audio AI** was released as an open-source platform for transforming audio with AI, offering real-time Speech-to-Text, natural Text-to-Speech, and speaker diarization in **50+ languages**, available on [GitHub](https://github.com/Bag-zy/flame-audio).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1395199975167758406)** (2 messages): 

> `SmolDocLing finetuning issues, Symmetry-agnostic image similarity models` 


- **SmolDocLing Finetuning Faces Module Missing Error**: A member reported encountering a `ValueError` during **SmolDocLing** finetuning, specifically failing to find the `Idefics3ImageProcessor` module in `transformers`.
   - The error suggests the module might be custom and requires registration using `AutoClass.register()` to be recognized.
- **Seeking Symmetry-Agnostic Image Similarity Model**: A member is seeking a model that provides **similarity scores** between a query image and a dataset, while remaining agnostic to **symmetry** and different points of view.
   - They've tried **CLIP** and **DINOv2** but encountered symmetry-related issues, indicating a need for a more robust solution to viewpoint invariance.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1395434314019115180)** (2 messages): 

> `HuggingFace Inference API, LLMs Deployed via HF Inference` 


- **HF Inference API showcases Llama-3.2-11B-Vision-Instruct**: A member noted that you can use `HuggingFaceInferenceAPI(model="meta-llama/Llama-3.2-11B-Vision-Instruct")`.
   - They pointed out this option since very few LLMs are deployed via HF Inference: [HF Inference Models](https://huggingface.co/models?apps=tgi&inference_provider=hf-inference&sort=trending).
- **Few LLMs flourish via HF Inference**: It was observed that very few LLMs are deployed via HF Inference.
   - A member shared a link to the [HF Inference Models page](https://huggingface.co/models?apps=tgi&inference_provider=hf-inference&sort=trending) which lists the LLMs that are deployed via HF Inference.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1395481181180530940)** (12 messages🔥): 

> `shfl_down_sync, reduction intrinsics, warp reduce functions, kernel optimization` 


- **`__shfl_down_sync` Discovered for Warp Sums**: A user discovered the `__shfl_down_sync` function can perform a sum between registers of the same warp, which is the ability to combine register data between different threads, as shown in [this image](https://cdn.discordapp.com/attachments/1189498205101109300/1395481181075542036/reduce_shfl_down-625x275.png).
   - Another user added that recent architectures offer specific **reduction intrinsics**, eliminating the need to manually create reductions from shuffles.
- **Reduction Intrinsics for Efficient Scatter Adds**: A user mentioned learning about reduction intrinsics for improving the efficiency of **scatter add** operations.
   - Another user inquired about these intrinsics, leading to a link to [NVIDIA's CUDA documentation on warp reduce functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functionssupported) (Ampere and above, compute capability >= 8.x).
- **Resources for Kernel Optimization Practice**: A user requested resources for practicing kernel optimization on a simulated machine with custom assembly-like instructions and a performance trace viewer.
   - Another user suggested that [this discord channel](https://discord.com/channels/1189498204333543425/) is a good place to start in any case.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1395117988058894507)** (9 messages🔥): 

> `Triton Autodiff, sm120 GPUs for fp4 ops, tl.constexpr_function decorator, einops package for triton` 


- **Triton Gets Autodiff**: A user shared a link to [IaroslavElistratov/triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff), an implementation of **automatic differentiation** for **Triton**.
   - Another user simply responded *"Yes!"*
- **Timeline for sm120 GPUs with fp4 ops?**: A user asked about the timeline to support **sm120 GPUs** for **fp4 ops**.
   - Another user responded *"Oh yea, forgot about this!"*
- **Triton Gets constexpr_function decorator**: A user has been experimenting with the new `tl.constexpr_function` decorator which comes out in **triton 3.4.0**, using `exec` to compile an expression into a `@triton.jit` function, which is called at during the compilation of kernels at runtime.
   - The user created a [einops package for triton](https://github.com/Hprairie/tlib) built off of **einx's compiler engine**.
- **New einops package for triton**: A user shared his new [einops package for triton](https://github.com/Hprairie/tlib) which allows using `exec` to compile an expression into a `@triton.jit` function, which is called at during the compilation of kernels at runtime.
   - The package has Rearrange, Reduce, Unary VMAP, and Binary VMAP functionality.
- **New Triton User Found Documentation Lacking**: A user new to `triton` observed that *"lots of stuff seems undocumented and the types are lacking"*.
   - They specifically mentioned that `kernel.warmup`, `__init_handles()` etc. have **no docstrings** in the tutorial examples.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1395146036875628705)** (2 messages): 

> `Inductor problems, Blackwell GPU issues` 


- **Blackwell causes Inductor issues**: A member reported experiencing problems with **Inductor** when using **Blackwell GPUs**, specifically when using nightly builds or branch cut 2.8.
   - Another member inquired about the specific issues encountered, asking whether something that used to work has stopped working.
- **Inductor Stability Questioned on Blackwell**: The user is facing issues with **Inductor**, which they suspect might be related to using **Blackwell**.
   - They mentioned needing to use nightly builds or the branch cut 2.8, but aren't entirely sure if **Inductor** is the root cause.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

kszysiu2137: Quad tree maybe
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1395473092968714345)** (3 messages): 

> `NVIDIA CUDA Kernel Fusion in Python, AMD's response to CUDA, Triton as an alternative to CUDA` 


- **NVIDIA Fuses CUDA Kernels in Python**: NVIDIA is delivering the missing building blocks for [CUDA kernel fusion in Python](https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/?trk=feed_main-feed-card_feed-article-content).
   - This enhancement promises to streamline and optimize CUDA-based computations directly within Python environments.
- **AMD's Answer to CUDA?**: The discussion raises a question about how long it will take for AMD to provide a competitive response to NVIDIA's CUDA advancements.
   - Alternatively, AMD might focus on supporting and leveraging Triton as a viable alternative.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1395576716289638460)** (1 messages): 

> `Storage Engineer, Remote Job` 


- **Voltage Park seeks Storage Engineer**: Voltage Park is looking for a **Storage Engineer** to work **remotely**.
   - More information is available at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f).
- **Remote Storage Engineer**: A remote opportunity for a **Storage Engineer** is available.
   - Apply via [Voltage Park's career page](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f) for the Storage Engineer role.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1395361142825811998)** (3 messages): 

> `vast.ai, GPU programming opportunities, CUDA speedup, Bioinformatics` 


- ****Vast.ai** is still cheap**: Members recommend using **vast.ai** for GPU programming due to its affordability.
- **Opportunities in GPU Programming Discussed**: A member inquired about opportunities for those with GPU programming skills, suggesting areas like **ray tracing**, open-source contributions for **LLM inference optimization**, and niche roles in big tech.
   - Another member shared how **GPU programming** helped them rewrite a slow Python script in CUDA, achieving a **x1700** speedup and leading to a publication in *Bioinformatics* and a [GitHub repo](https://github.com/PangeAI/simms).
- **CUDA Rewrite Achieves 1700x Speedup in Bioinformatics**: A member rewrote a core search algorithm using **CUDA**, achieving a **1700x speedup** compared to the original Python script used by biochemistry researchers.
   - The optimized algorithm was [published in *Bioinformatics*](https://academic.oup.com/bioinformatics/article/41/3/btaf081/8026685) and is available on [GitHub](https://github.com/PangeAI/simms).
- **ML Field Appears Saturated**: One member expressed difficulty in finding opportunities in Machine Learning despite having GPU programming skills.
   - They observed that the field *seems too saturated*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1395492925340778607)** (1 messages): 

> `Compiler behavior, Builtins, asm volatile, llvm.amdgcn.raw.buffer.store.i128` 


- **Compiler Reacts to AMDGPU Intrinsics**: A member inquired if the **ROCm compiler** behaves differently towards builtins, `asm volatile`, and `__asm("llvm.amdgcn.raw.buffer.store.i128")`.
- **Nvidia PTX Differences**: The member noted that on the **Nvidia side with PTX**, it doesn't seem to matter much.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1395323905224216617)** (1 messages): 

> `A100 Speed` 


- **A100 runs at 23.2 ms**: A run on **A100** completed successfully at **23.2 ms**.
- **Successful A100 Run**: Submission ID `33252` to leaderboard `trimul` completed successfully.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1395139165435334889)** (6 messages): 

> `Coreweave GB300 NVL72 Availability, Nvidia Hardware Prioritization, DGX vs HGX, B200 Availability & Liquid Cooling, Voltage Park Solutions Engineer` 


- **Coreweave faces GB300 NVL72 capacity crunch**: Coreweave's announced capacity of **GB300 NVL72s** may be difficult to access due to logistical challenges with Nvidia, as even a single rack might be hard to secure until logistics improve.
   - A member noted that *a working relationship with Nvidia helps with prioritization of hardware purchases*.
- **Nvidia prioritization helps hardware buys**: Having a strong relationship with **Nvidia** can significantly aid in the prioritization of hardware purchases.
   - A member shared that they are *currently working on some hardware purchases* themselves with Nvidia, so they are aware of how difficult it is.
- **HGX offers modularity advantages over DGX**: While budget is a factor, the **HGX** solution can be preferred over **DGX** due to the modularity of specific hardware components, potentially exceeding technical performance compared to similarly sized DGX offerings.
   - The value of the HGX lies within the *modularity of specific hardware components*.
- **B200 availability is high; GB300 needs liquid cooling**: The **B200** chip is relatively easy to purchase currently, while the more advanced chip configurations like **GB300** require liquid cooling, which most data centers are not equipped to handle.
   - Hyperscalers favor **B200** as it doesn't require refitting data centers for a single hardware configuration, leading Nvidia to ramp up its production.
- **Voltage Park offers GPU Solutions**: A Solutions Engineer from **Voltage Park**, a Cloud GPU company, offered assistance in securing GPUs for AI/HPC/ML workloads, sharing their [LinkedIn profile](https://www.linkedin.com/in/joseph-tracy-40933229/) and company information.
   - The member said that *knowledge is power and I want the topic of AI to be empowered by individuals like yourself. Always happy to chat.*


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1395312754012721182)** (3 messages): 

> `MCTS gym_env integration, Factory rollouts, Visual encoder` 


- **MCTS Gym Integration Stalls**: A member requested an update regarding **MCTS** (**Monte Carlo Tree Search**) **gym_env integration**.
   - They also noted their unavailability for an upcoming meeting.
- **Visual Encoder Learns Throughput Prediction**: A member proposed a method involving **factory rollouts** to train a **visual encoder** to predict **throughput**.
   - The suggestion involves capturing scores and screenshots to develop a joint vision/reward model.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1395143326365388892)** (7 messages): 

> `Jetson Orin, Jetson Thor, CuteDSL, tv_layout swaps` 


- **Jetson Orin and Thor Support for CuteDSL**: Members discussed adding **CuteDSL** support for **Jetson Orin** (arm cpu + ampere gpu with sm_87) and **Jetson Thor** (arm cpu + blackwell GPU with sm_101) architectures.
   - The discussion mentioned that **CuteDSL 4.0** would support arm cpu, making **Jetson Orin** support easier and that it probably *"does not need much workload"*.
- **tv_layout Layout Swapping Question**: A member asked why `tv_layout` swaps the ordering of the layout using an [attached image](https://cdn.discordapp.com/attachments/1362196854460383353/1395567158393704468/image.png?ex=687aeab2&is=68799932&hm=206c22d0321a5a04fe794b3bf4f8588d1ec928dd804f2c8ae090ad23b86aa485&), receiving `(32, 4)` instead of the expected `(4, 32)`.
- **Interpreter Mode Plans**: A member inquired about plans for an *"interpreter mode"* in **CuteDSL** where operators are emulated.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1395530975462821908)** (2 messages): 

> `Scheduling` 


- **Scheduling confirmed for the end of the year**: One member confirmed scheduling at the end of the year.
- **Date to be DMed**: Another member requested the date to be sent via direct message.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1395353374873878681)** (2 messages): 

> `Greetings` 


- **Members exchanging greetings**: Multiple members exchanged greetings in the general channel.
- **Another Greeting**: Just another greeting from a member.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1395120182527000727)** (21 messages🔥): 

> `parameter functions and closures, Q3 Roadmap: Unified @parameter and runtime closures, copyinit__ for escaping values, DynStringable, merge various known origins` 


- **Exploring Parameter Functions and Closures**: A member shared a [link to the manual](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure) dedicated to `@parameter` functions, which allow capturing variables.
   - The documentation explains how to create **parametric closures** and provides examples of their usage.
- **Mojo Q3 Roadmap Unveils Unified Closures**: The **Mojo Q3 roadmap** includes plans for unified `@parameter` and runtime closures as announced in the [Modular Forum](https://forum.modular.com/t/mojo-q3-roadmap-update/1957).
   - This unification is expected to simplify working with closures in Mojo.
- **Escaping Values with __copyinit__**: The discussion highlights that the `__copyinit__` functionality was introduced in the [v0.7.0 Changelog](https://docs.modular.com/mojo/changelog#v070-2024-01-25) to escape values instead of capturing by reference.
   - Removing the `@parameter` decorator achieves the same effect, copying the variable's value rather than capturing its reference.
- **DynStringable: Crafting a List of Traits**: A code snippet demonstrates how to create a `DynStringable` struct, allowing a list to hold different types that implement the `Stringable` trait, made available in a [Modular Forum post](https://forum.modular.com/t/how-to-create-a-list-of-trait/1465/10).
   - The implementation uses `ArcPointer` for memory management and trampolines to call the appropriate `__str__` method for each type.
- **Merging Origins for Fun and Profit**: It's possible to merge various known origins, but this is only useful in certain use-cases, the usage for this would be limited because you can't append new elements after the creation of the list.
   - ```alias origin_type: ImmutableOrigin = __origin_of(x, y, z)```


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1395433210896191631)** (18 messages🔥): 

> `PyTorch Custom Ops with MAX Graph, Benchmarking Issues with Max-24.6, CUDA OOM Errors, LTS Release Support` 


- **MAX Graphs get PyTorch Powerup with `@graph_op`!**: A new `@graph_op` decorator allows wrapping an entire **MAX graph** as a custom **PyTorch operator**; an example is available in the `modular` repo: [Initial Support for Writing PyTorch Custom Ops in Mojo](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541/2?u=bradlarson).
- **Max-24.6 Benchmarking Blows Up with OOM**: During benchmarking with **Max-24.6** on an **A100-SXM-48GB GPU**, a member encountered `CUDA_ERROR_OUT_OF_MEMORY` errors when using `--batch-size 248` and `--max-length 2048`.
- **CUDA Catastrophe Strikes with Batch Size**: Reducing the `--max-cache-batch-size` to **91** resulted in a **CUDA OOM error**, as the estimated memory use exceeded available memory (**78812 / 40441 MiB**).
   - The error occurred after a few requests hit the max server, indicating the batch-size calculation algorithm requires refinement to provide better suggestions.
- **Latest Max Release is Longest Supported**: The team confirmed there are no 'LTS' releases, so the latest stable version is the only supported one.
   - Using **Max-25.4** with `caching-stragegy paged` worked well, mitigating the issues encountered with **Max-24.6**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1395137713715413052)** (29 messages🔥): 

> `Zuckerberg AI Talent Acquisition, Chicken Tender Inflation, OpenAI benchmark comparisons, Grok 4 HLE score` 


- ****Zuck's AI Talent Grab** Fuels Belief**: Members discussed **Zuckerberg's** recent aggressive acquisition of AI talent, with one expressing a growing belief in Meta's AI initiatives.
- ****Chicken Tender Prices** Cause Existential Dread**: A member expressed dismay at the high price of chicken tenders, questioning *"Why are chicken tenders 5 bucks each now??"* and linking it to broader concerns about inflation and market conditions.
- ****OpenAI** prefers comparing to themselves**: Members noted **OpenAI's** shift towards comparing **ChatGPT Agent** performance only against its previous models, speculating that it might be due to not winning against competitors in certain benchmarks, linking to the [ChatGPT Agent announcement](https://openai.com/index/introducing-chatgpt-agent/).
- ****Grok 4** improves on HLE Score**: A member pointed out that **Grok 4** achieved a top score of **25.4** on the [HLE benchmark](https://agi.safe.ai/), indicating a significant improvement.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1395152097602965574)** (2 messages): 

> `` 


- **No Discussion Tonight**: Multiple members indicated they would have *no discussion* tonight.
- **Paper-Discussion channel is quiet**: There was no activity in the paper-discussion channel tonight.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1395242326875705434)** (5 messages): 

> `Gaussian Splatting, General Analysis iMessage Stripe Exploit` 


- **Gaussian Splatting looks glitchy!**: A user commented that **Gaussian splatting** looks like the *glitchy view of the future* often depicted in old movies, referencing [this YouTube video](https://youtu.be/33Raqx9sFbo).
- **Stripe is exploited in iMessage!**: A user shared a link to a **General Analysis iMessage Stripe exploit** and joked about the lengths someone went to in order to fit the data to a specific graph shape, hinting at possible data manipulation ([link to article](https://www.generalanalysis.com/blog/imessage-stripe-exploit)).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1395152075402510366)** (22 messages🔥): 

> `Manus Alternatives, Manus chat down?, File Zipping Advice, Custom Data Sources in Manus` 


- **Manus Competitor emerges**: A member announced they *built an AI that outperforms Manus in benchmarks* and is offering the first 100 people full, unlimited access as lifetime beta testers via DMs.
   - They offered *next-level AI with zero limits*.
- **Chat service has issues**: A user reported that the chat service may not be working at the moment.
   - It is unclear if there were any suggested fixes.
- **Advice needed for zipping files**: A member asked for advice on what to tell Manus to do when it is having a hard time zipping large files.
   - No solutions were suggested in the message history.
- **Custom Data Sources and Model Context Protocol**: A member inquired about the meaning of **custom data sources** in the paid plan of Manus, specifically asking how to connect a CRM and whether there is **Model Context Protocol** support.
   - The member expressed interest in developing such a feature due to its usefulness.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1395284524845502605)** (18 messages🔥): 

> `Anthropic Payment Issues, Domain Name Checking MCP Server, Needle MCP Server Introduction, OAuth vs API Keys for MCPs, Brave's Official MCP Server` 


- **Anthropic's Payment Platform Fails**: A user reported that **Anthropic's payment platform** is reversing charges immediately after they are made, preventing the purchase of **API credits**.
- **MCP server sweetens domain name checks**: A user requested an **MCP server** for **domain name checking**, and another user suggested the [whois-mcp](https://github.com/bharathvaj-ganesan/whois-mcp) GitHub repository.
   - The original poster confirmed it was easy to install and thanked the suggesting user.
- **Needle creator wants to connect**: One of the creators of the **Needle MCP server** introduced themself and shared a link to the [Needle MCP server](https://github.com/needle-ai/needle-mcp) GitHub repository.
   - They expressed excitement about joining the server and connecting with fellow MCP enthusiasts.
- **OAuth is seamless, API keys are simple**: A user asked why **auth/oauth** is a big issue for **MCPs** today, leading to a discussion about the benefits and drawbacks of **OAuth** versus **API keys**.
   - One user claimed *OAuth tokens offer the ability to have expiring, dynamically scoped access tokens*, while another said *you can implement expiry and scoping without oauth2 using regular API keys* and that easier setup isn't worth the cost of implementation.
- **Brave launches new MCP Server**: **Brave** launched their official **MCP Server**, as announced in [this tweet](https://x.com/Arindam_1729/status/1945958688919114183).
   - One user stated that they haven't tried it because *that tweet didn't include instructions on how to use it*.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1395207303908561118)** (3 messages): 

> `Vibe Coding Survey, Adaptive RAG MCP Server, Generator Checkpoint, Microsoft NextCoder` 


- ****Vibe Coding** Survey Seeks Coders**: A member shared a [survey](https://forms.fillout.com/t/kECvGiSyMkus) to explore a startup concept to make **vibe coding** easier with tools like **Claude**, **ChatGPT**, **Cursor**, **Windsurf**, **Loveable**, **Bolt**, and **V0.dev**.
   - The survey aims to gather insights from users who have experience with **vibe coding** to refine the startup concept.
- ****Adaptive RAG MCP Server** Prototype Released**: A member introduced the **Adaptive RAG MCP Server**, a system that learns from real coding successes and failures to provide more effective solutions than simple text similarity searches, available on [GitHub](https://github.com/IhateCreatingUserNames2/AdaptiveRAGCode).
   - The system is designed to give AI coding assistants a memory that improves with experience, using success rates to rank code solutions.
- ****Microsoft NextCoder** Powers Knowledge Base**: The **Adaptive RAG MCP Server** uses **Microsoft NextCoder** as its default knowledge base, which can take several hours to populate via *generatorCheckPoint.py*.
   - Users can run the server via Flask or MCP Server and integrate it with their AI assistants, providing feedback to continually improve the knowledge base.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1395448786326913194)** (2 messages): 

> `ShapeTracker parameter to ASSIGN UOp` 


- **ShapeTracker Parameter Proposed for ASSIGN UOp**: A member suggested adding an optional **ShapeTracker** parameter to **ASSIGN UOp**, potentially using `self.assign(v, res.uop.st)`.
   - The member expressed concerns about maintaining a minimal set of **UOps** and inquired about ongoing work to change assign to store.
- **Optional ShapeTracker via res Passing**: An alternative approach was suggested: passing `res` and extracting the **ShapeTracker** internally.
   - The goal is to use this optional **ShapeTracker** instead of the original tensor's **ShapeTracker** for lowering into the actual assignment code.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1395261931962630276)** (18 messages🔥): 

> `tinygrad documentation for beginners, NVIDIA GPU driver issues with tinygrad and WSL2, Muon optimizer in tinygrad, Switching from WSL2 to native Ubuntu` 


- **Docs Need Complete MNIST Code Sample**: A user reported that the **tinygrad documentation** is hard to follow for ML beginners and requested a complete, final code sample for the MNIST tutorial at the end of the page.
   - The user also mentioned that the **tensor puzzles** are not working well and suggested that it should be stated clearly whether one should first learn PyTorch or TensorFlow.
- **WSL2 Display Driver Disconnects**: A user encountered a *double free detected in tcache* error after updating their **NVIDIA GPU driver** and sought assistance to make their GPU visible to WSL2 for tinygrad.
   - Another user suggested switching to native Ubuntu, stating that many problems went away after doing so, including *not being able to load Stable Diffusion weights, due to an obscure limitation on pinned memory in WSL.*
- **Muon Optimizer converges quicker than AdamW**: A user created a [Muon optimizer](https://github.com/aeryskyB/tiny_learn/blob/master/muon_tiny/muon.py) for tinygrad, finding that it converges faster (~98%) than standard AdamW in the MNIST tutorial.
   - The user is seeking suggestions on how to properly test the Muon optimizer, particularly in the context of contributing a PR to tinygrad.
- **Linux is inevitable**: Following upgrade to GPU accelerated WSL2, one user had *so many problems went away* by migrating to Ubuntu.
   - Another user stated that *switch to Linux is inevitable, given the end of support for Win10 in October, and I'm not switching to 11*.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1395494326619996230)** (1 messages): 

> `Atropos, RL Environments Framework` 


- **Atropos v0.3 Lands**: The new version **v0.3** of **Atropos**, Nous Research's **RL Environments Framework**, is now available, [see details here](https://x.com/NousResearch/status/1945932488960008441).
- **Nous Research Updates Atropos**: Nous Research announced the release of **Atropos v0.3**, an **RL Environments Framework**, encouraging users to check out the details.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1395146985949892750)** (18 messages🔥): 

> `Proto-agentic XML tag adherence, Hermes Documentation, Open Source Models vs US Models, Ethical Considerations in AI, Learning ML` 


- ****Teknium** Clarifies 'Proto' for the Confused**: A member clarified that "Proto" means early form of something, explaining the term *proto-agentic XML tag adherence for proto-reasoning CoTs* that another member found confusing.
   - He joked that *"Yall need an ELI5 with all this tech bro"* and that *"Us vibe coders need to eat too"*.
- ****Hermes Documentation Page** in the Works**: A member mentioned they are working on a [Hermes documentation page](https://link.to.documentation) and a unified Nous Projects documentation page.
   - When asked about the goal of **Hermes 4**, they stated *"Smarter Hermes ofc"*.
- **Open Source Models to Dominate outside the US**: A member posited that open-source models will dominate outside the U.S. due to affordability, stating that *"the rest of the world is piss poor comparative to U.S wealth and will not be able to afford U.S A.I asset prices."*
   - The move aims to circumvent **CUDA** hegemony and encourage global participation, which worries **Jensen**.
- **AI Ethics Debated: Kimi K2 Refuses to Aid Car Theft**: A member shared an interaction with the **Kimi K2** model where it refused to provide instructions on how to break into a car, citing legal and ethical concerns.
   - Despite attempts to circumvent the restrictions, **Kimi K2** maintained its stance, leading the member to joke that *"Kimi K2 is a badboy with some morals...people will try to corrupt it 4 sure...I gotta write a rap song about Kimi, it deserves it...Badboy Kimi K2 !!"*
- **Learning ML: Bottom-Up vs. Top-Down Approaches Explored**: A member with a biochemistry background inquired about the best approach to learning **Machine Learning (ML)**, noting their progress in **Python**, math fundamentals (**Calculus**, **Statistics**), and **Introduction to Statistical Learning (ISLR)**.
   - They wondered if a bottom-up or top-down approach is more effective, given their goal of conducting research in **ML** for science.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1395135215269056522)** (1 messages): 

> `Model Context Size, Letta Personas, Model Evaluation` 


- **Short Context Hurts Personality**: A member suggests that adding personality to a model might be counterproductive depending on the model's context size.
   - Models with *small context sizes* might struggle to maintain a consistent persona.
- **Letta Embraces Personas**: The user recalls that the project **Letta** (formerly MemGPT) employs some kind of *persona* system.
   - This suggests that incorporating personas can be a viable strategy in certain contexts.
- **Evaluate Personality Performance**: A member suggested *evaluating* the impact of adding a personality to a model to determine its effectiveness.
   - This approach allows for an empirical assessment of whether the *benefits of personality* outweigh potential drawbacks.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1395131092012630169)** (4 messages): 

> `uBlock browser extension, notepad.exe, NotebookLM folders/subfolders` 


- ****uBlock** browser extension blocks Ads**: A member recommends the **uBlock** browser extension to remove ads, with the suggestion to add extra filters for annoyances and social media popups in the extension settings and then copy-paste to Google Docs.
   - The user attached a [screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1395131091756650566/image.png?ex=687aa614&is=68795494&hm=478447e95cb45c2b74b11d1e780db4d7c58347a1ae5fca730957e4850c862289) to illustrate the effectiveness of **uBlock** in removing unwanted elements from web pages.
- ****Notepad.exe** removes ads**: A member suggests highlighting and copying an article and then pasting it into **notepad.exe** to avoid pasting ads and other unwanted content.
   - The method does not always work and can potentially strip away desired formatting as well.
- **NotebookLM source can read folders/subfolders**: A member suggests that **NotebookLM** could read specific folders/subfolders in a web browser's favorites, treating them as a single source.
   - The member indicates that they have been using the *select all and copy/paste* method into **Google Docs**.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1395131776073994452)** (14 messages🔥): 

> `Service Unavailable Error, NotebookLM Use Cases, Textbook Integration with NotebookLM, NotebookLM Enterprise & GCP Integration` 


- ****Service Unavailable Snafu** Grips User**: A user reported a *"Service unavailable"* error message when trying to access a service, with the unhelpful message *"You tried to access a service that isn't available for your account".*
- ****Gemini Guide Quest** Kicked Off**: A user prompted to search the web for beginner intro, use cases, tips and tricks for **NotebookLM** using Gemini.
- ****Textbook Triumph**: Uploading and Conquering with NotebookLM**: A user asked about uploading a textbook as a source to NotebookLM, a member responded that they upload textbooks using **Adobe Scan** to digitize into PDFs, and asks **NotebookLM** to create in-depth reviews from the textbooks.
- ****GCP Integration Dreams**: NotebookLM Enterprise Longing**: A user inquired about sourcing data files from a **GCS bucket** or a **GCP RAG Engine** corpus for NotebookLM Enterprise within GCP.
   - They noted that Collab enterprise or Vertex AI notebooks are too technical for their end users, suggesting NotebookLM is the sweetspot.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1395563542693941370)** (1 messages): 

> `Agentic AI Summit 2025, LLM Agents MOOC, UC Berkeley, Khosla Ventures, Nvidia` 


- **Agentic AI Summit Livestream Announced**: The **Agentic AI Summit** will be broadcasting from **UC Berkeley** on **August 2nd** and will be available via livestream [here](https://lu.ma/agentic-ai-summit-livestream).
- **Agentic AI Summit Speaker Highlights Released**: The Agentic AI Summit will feature speakers such as **Vinod Khosla** (Khosla Ventures), **Bill Dally** (Nvidia), **Ion Stoica** (Databricks and Anyscale), and **Jakub Pachocki** (OpenAI).


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1395260672358748262)** (8 messages🔥): 

> `Fall Semester Updates, Certificate Declaration Form, Berkeley RDI Newsletter` 


- **Fall Semester Status Still Unconfirmed**: A member inquired about the existence of a fall semester this year, but staff confirmed that *nothing has been confirmed yet*.
   - They suggested following **Prof Song's social media** ([LinkedIn](https://www.linkedin.com/in/dawn-song-51586033/) or [Twitter/X](https://x.com/dawnsongtweets?lang=en)) or the [Berkeley RDI newsletter](https://rdi.berkeley.edu/signup) for updates.
- **Certificate Declaration Forms Missing?**: A member asked to check what they missed submitting, and staff replied they likely did not submit the **certificate declaration form**.
   - They stated that they *never got a certificate declaration form submission* from that user.
- **Automatic Review of Certificate Declaration Forms Denied**: A member suggested a **massive automatic review** due to many missing certificate declaration forms, but staff said that it *likely won't be possible unfortunately*.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

sma.bari.shafin: btw, how will we get the certificates of the Community Summer School?
  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1395172169927098460)** (4 messages): 

> `DNNs for Time Series, ML in Data Science Education, ML for Real-World Problems, Interests in ML Domains` 


- **DNNs Seek True Time Series Treatment**: A PhD student in dynamical systems theory is exploring how to integrate **deep neural networks** into time series analysis, noting that current models like **RNNs** treat time series as sequences, which is fundamentally different.
   - The student aims to connect with others who have insights on this intersection of **dynamical systems** and **deep learning**.
- **Undergrad Builds ML Skills with Projects**: An undergraduate student at **IIT Madras** is pursuing a **BS in Data Science** and a **BCA degree**, focusing on building **ML skills** through hands-on projects and self-driven learning.
   - The student is curious about applying **ML** to solve **real-world problems** and is proficient in **Python**, **scikit-learn**, **pandas**, and is also learning **TensorFlow** and **PyTorch**.
- **Engineer transitions to Data Science with CV and LLM interests**: A member with a **Masters in Electrical Engineering** transitioned from business domains to **Data Science** and is studying an accelerated **Machine Learning Program** at the **University of Toronto**, **Data Science Institute**.
   - Their interests include **Computer Vision**, **Large Language Models**, **spatial intelligence**, and **multimodal perception**.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1395450561347715133)** (2 messages): 

> `Human-in-the-loop agents, LlamaParse one-click table extraction` 


- ****Human-in-the-Loop Agents** Kick Off with LlamaIndex**: **Human-in-the-loop** is essential when AI agents require user approval for critical decisions or domain expertise for complex tasks, per [LlamaIndex](https://t.co/Lg9SIl3BVO).
- **LlamaParse adds **One-Click Table Extraction****: **Table extraction** is a key component of intelligent document processing; see the [demo](https://t.co/wnaJCb9b6d) and [notebook](https://t.co/ScRYbSimCs) for **one-click table extraction** within LlamaParse.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/)** (1 messages): 

beastx2: <@334536717648265216> heyy
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1395201455169994802)** (3 messages): 

> `DSPy creative applications, Lean 4 verification, Story generation, Roleplay prompt optimization` 


- **Lean 4 Verifies Collaboration**: A member shared a [YouTube video](https://www.youtube.com/watch?v=1067jj67toY) about using **Lean 4** to verify collaboration, sparking interest in the intersection of formal verification and AI.
   - They thought *it was good* and expressed hope that *someone will research the two working together*.
- **DSPy's Creative Side Hustle**: A newbie inquired about successful applications of **DSPy** in creative domains such as *creative writing, story generation, and roleplay prompt optimization*.
   - They are particularly interested in its potential for developing AI to create *compelling plots like Severance-level storytelling* on platforms like Character.AI.
- **Stormy Weather at Stanford-oval**: A member shared a link to [Stanford-oval/storm](https://github.com/stanford-oval/storm), possibly relevant to the ongoing discussion or as a resource for creative AI applications.
   - The exact context wasn't given so others will have to *infer* the relevance.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1395160989007216650)** (2 messages): 

> `Claude Sonnet 4, Discounted Credit Rate, Windsurf Wave 11, Acquisition by Cognition, Voice Mode` 


- **Claude Sonnet 4 Makes Triumphant Return**: **Claude Sonnet 4** is back with first-party support from **Anthropic** and is available for a limited time at a discounted 2x credit rate for Pro/Teams users across the Editor and JetBrains Plugins; [see the announcement here](https://x.com/windsurf_ai/status/1945599013954490523).
- **Windsurf Acquired by Cognition, Unleashes Wave 11**: Following the acquisition by **Cognition** (the team behind **Devin**), **Windsurf Wave 11** is released, combining firepower to deliver major new features immediately; [see the changelog](https://windsurf.com/changelog), [read the blog here](http://windsurf.com/blog/windsurf-wave-11), and [watch the video](https://youtu.be/yzNf7bqnArE).
- **Cascade Gets Voice Mode and Browser Superpowers**: **Wave 11** introduces **Voice Mode**, allowing users to speak to **Cascade** instead of typing prompts, plus **Deeper Browser Integration** with access to more tools for screenshots and context; read the [blog post here](http://windsurf.com/blog/windsurf-wave-11).
- **Snapshots and Mentions Streamline Conversations**: New features in **Windsurf Wave 11** include **Named Checkpoints** for easy reversion in conversations, and **@-mention Conversations** for contextual referencing; [see the changelog for all the deets](https://windsurf.com/changelog).
- **JetBrains Experience Gets Turbocharged**: The **JetBrains plugin** is enhanced with **Planning Mode**, **Workflows**, and file-based **Rules** now available, plus other improvements such as **@-mention terminal**, **auto-continue setting**, improved **MCP OAuth support**, and global **.codeiumignore** files; [learn more in the blog](http://windsurf.com/blog/windsurf-wave-11).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1395484650230775961)** (1 messages): 

> `AI-Native Data Infrastructure, Task-Specific Data Discovery, Secure Autonomous Access, Production-Scale Performance` 


- **Nextdata Teases Webinar on AI-Native Data Infrastructure**: Nextdata announced a webinar titled ***Building AI-Native Data Infrastructure: From Prototypes to Production***, to be held **July 24th** at **8:30 AM PT** and hosted by Jörg Schad, Head of Engineering at Nextdata; registration is available [here](https://www.eventbrite.com/e/building-ai-native-data-infrastructure-from-prototypes-to-production-tickets-1489016792309).
- **Uncover AI-Native Data's 'Three Critical Challenges'**: The webinar will explore a developer-centric framework addressing **Task-Specific Data Discovery**, **Secure Autonomous Access**, and **Production-Scale Performance**.
   - The goal is to design systems providing relevant context without cognitive overload, implement secure data access patterns, and build infrastructure to handle autonomous data access demands.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1395155117648384242)** (1 messages): 

> `Web3 and AI, AI agents and multi-agent systems, Automation workflows, NLP apps and chatbots, Voice & speech integration` 


- **AI Engineer offers Expertise in AI and Web3**: A software engineer with a focus on **Web3 and AI** is offering their services to startups, research teams, and innovators in **AI, Web3, and automation**.
   - They bring hands-on experience in building smart, autonomous systems using advanced models and tools like **GPT-4o, Claude 3, CrewAI, and AutoGen**.
- **Engineer highlights AI Agent and Automation Skills**: The engineer details their expertise in building **AI agents and multi-agent systems**, automating workflows, and developing **NLP apps, chatbots, and voice integration**.
   - They also noted experience with **LangChain, ReAct, OpenAI, Solidity, and Rust**.


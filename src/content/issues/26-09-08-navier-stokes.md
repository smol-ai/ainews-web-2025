---
id: MjAyNS0x
title: OpenAI reports Navier-Stokes singularity find, a contender for second ever Millenium Prize awarded, overshadowing Cognition's $48B Series E, Mistral's $24B Series D, Meta's Muse agent, and GPT Image 2.5
date: '2026-09-08T05:44:39.731046Z'
description: >-
  **OpenAI** announced a proposed Navier–Stokes proof by an internal model
  "**significantly more capable than GPT-6 Astra**" using **10,000 agents** over
  **88 hours** plus **17 hours** of formal verification. The effort highlights
  the emergence of **massive test-time compute scaling** as a new axis beyond
  pretraining, with estimated costs of **$10M–$40M** and **130B output tokens**.
  Controversy arose over priority, data contamination, and scientific norms,
  with key figures like **Sam Altman**, **Sébastien Bubeck**, and **Terence
  Tao** weighing in on governance and open science risks. Meanwhile, **Meta**
  launched **Muse**, a consumer personal AI agent featuring persistent isolated
  Linux VMs, browser integration, and connectors to various apps including
  Meta-native services like Instagram and Messenger, emphasizing security and
  broad service integration.
companies:
  - openai
  - meta-ai-fair
models:
  - gpt-6-astra
topics:
  - test-time-compute
  - formal-verification
  - parallel-computing
  - scientific-governance
  - personal-ai-agent
  - linux-vm
  - service-integration
  - data-contamination
  - open-science-norms
people:
  - sama
  - sebastienbubeck
  - terence_tao
  - sam_altman
---


**a quiet day.**

> AI News for 9/7/2026-9/8/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap



**OpenAI-affiliated accounts said an AI-assisted effort produced a Navier–Stokes result, and the reaction immediately split between technical interest, skepticism, and meta-drama.**

- The most concrete public claim in the tweet set came from Ethan Knight, who said “The Navier Stokes solution was the result of a collaboration of ~10,000 agents working together,” adding that OpenAI had spent “the past year” training models to collaborate via “multiagent RL,” and that hard problems may yield to “huge amounts of unstructured parallel test-time compute” with models deciding how to organize themselves [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- Multiple onlookers interpreted this as OpenAI claiming an AI-generated proof related to the Navier–Stokes Millennium Problem, specifically around finite-time singularity / blow-up; one satirical paraphrase framed it as OpenAI saying a smooth fluid can “blow up into a singularity,” claiming “10,000 agents” and “88 hours” were used, while explicitly noting that mathematical acceptance remained a “minor formality” [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954).
- Broader commentary treated the event as a possible stress test for the belief that frontier AI cannot do serious research or coding-level technical work; Theo Jensen called it the science world’s “‘AI can’t ACTUALLY code’ crash out moment” [@theo](https://x.com/theo/status/2097540749663551704).
- Hrishikesh / hrishioa framed the announcement as evidence of a “high compute regime,” arguing observers should “adjust your plans accordingly” [@hrishioa](https://x.com/hrishioa/status/2097542911382761630).
- The announcement also triggered incidental operational speculation: one poster jokingly linked seeing ChatGPT latency warnings to OpenAI potentially redirecting large-scale compute toward the Navier–Stokes run, though this was pure conjecture and not evidence [@teortaxesTex](https://x.com/teortaxesTex/status/2097544071162085714).


## Disclosures and context up front


**What is factual from the tweets**

- An OpenAI-linked claim circulated that a Navier–Stokes “solution” involved about **10,000 agents** working collaboratively [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- The same source said these systems were trained over roughly **a year** using **multi-agent reinforcement learning** [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- The stated high-level method emphasized **parallel test-time compute** and model self-organization rather than a single long-chain proof attempt [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- Public readers understood the claim as concerning the **Navier–Stokes existence/singularity problem**, one of the **Millennium Prize Problems**, though the exact theorem statement and proof scope are not supplied in the tweet set [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954).
- Acceptance by the math community was clearly unresolved at the time of discussion; even the joke-post emphasized that correctness remained unverified by the field [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954).

**What is not established by the tweets**

- No theorem statement, preprint, proof sketch, formal verification artifact, benchmark report, or independent referee commentary appears in the provided tweets.
- The frequently repeated **“88 hours”** detail appears only in a satirical post in this set, not in the more direct OpenAI-adjacent statement, so it should not be treated as confirmed from this evidence alone [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954).
- The exact role of humans versus models is unspecified: “collaboration of ~10,000 agents” does not tell us whether humans decomposed the search, curated lemmas, verified steps, or merely launched infrastructure [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- “Solution” is ambiguous. In mathematics it could mean a complete proof, a proof strategy, a candidate counterexample, a formalized derivation, or a research lead. The tweets do not disambiguate this.
- There is no disclosed information here on whether the result addresses the standard 3D incompressible Navier–Stokes global regularity problem on \(\mathbb{R}^3\) or torus, or some variant/auxiliary statement.

**Why the ambiguity matters**

- The Navier–Stokes Millennium Problem has a very specific standard framing. Claims that a finite-time singularity “can occur” would be explosive because they imply a negative answer to global regularity in the relevant formulation; such claims require extraordinary precision and scrutiny.
- In frontier-model discourse, “AI solved X” often compresses multiple layers: conjecture generation, search, proof drafting, proof checking, and community validation. The tweets give only a systems-level description, not the epistemic status of the math.


## Technical details exposed by the tweets


**The disclosed technical picture is less about fluid mechanics than about a research system architecture.**

- **Scale:** approximately **10,000 agents** operating together [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- **Training approach:** **multi-agent RL** over the course of **~1 year** [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- **Inference philosophy:** large amounts of **unstructured parallel test-time compute**, with agents autonomously deciding how to divide work and collaborate [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- **Implied research thesis:** for difficult reasoning tasks, scaling **coordination + search at inference time** may be as important as, or more important than, simply scaling a monolithic model.
- **Sociotechnical implication:** this is a concrete articulation of a trend many labs have hinted at—shifting from “bigger single model” narratives toward **agentic ensembles**, **parallel search**, and **test-time compute scaling**.
- **Operational implication:** if true, the result is evidence that labs are willing to spend substantial inference compute on one-shot scientific targets, not just products or benchmarks.

**What this suggests technically**

- A 10,000-agent setup implies substantial infrastructure for:
  - task decomposition,
  - inter-agent communication,
  - memory/state persistence,
  - search-tree management,
  - reward design or proxy scoring,
  - aggregation / selection of candidate proof paths.
- The phrase “let them decide how to work together” suggests a partially emergent coordination policy rather than entirely hand-scripted orchestration [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- If the work genuinely touched a hard math problem, the key novelty may be less “LLM writes a proof” and more **distributed theorem search with learned collaboration policies**.

**What is missing technically**

- No mention of:
  - theorem prover integration,
  - formal verification,
  - proof assistant stack,
  - symbolic algebra systems,
  - fluid simulation components,
  - retrieval corpora,
  - model size,
  - compute budget,
  - pass@k style metrics,
  - ablations against single-agent baselines,
  - error rates or proof-check success rates.

That absence is central: the public conversation ran ahead of the disclosed technical substrate.


## Facts vs. opinions


**Facts/claims presented as facts**

- About **10,000 agents** were involved [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- OpenAI had been training collaborative agents via **multiagent RL** for about **a year** [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- The system used extensive **parallel test-time compute** [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- The result was publicly discussed as a **Navier–Stokes solution/proof claim** [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954).

**Opinions / interpretations**

- “One of the most effective ways to solve hard problems” is to use huge unstructured parallel test-time compute and self-organizing agents — this is a strong strategic interpretation, not yet demonstrated generally by the evidence in the tweet alone [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- “Science world is having their ‘AI can’t ACTUALLY code’ crash out moment” is commentary about community psychology, not a verifiable assessment [@theo](https://x.com/theo/status/2097540749663551704).
- “We truly are in a high compute regime” is a macro framing of industry direction [@hrishioa](https://x.com/hrishioa/status/2097542911382761630).
- The “88 hours,” “leadership lesson,” and “delegate 10,000 AI agents” framing is satire and should not be read as documentary detail [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954).
- The claim that ChatGPT slowdowns were caused by this experiment is speculation without supporting evidence [@teortaxesTex](https://x.com/teortaxesTex/status/2097544071162085714).


## Different perspectives


**Supportive / bullish perspectives**

- The strongest supportive perspective is that this is evidence for a new scaling law: not just model size and training compute, but **massively parallel, self-organizing inference-time collaboration** can unlock qualitatively new capabilities on frontier research problems [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- Theo’s reaction captures another bullish reading: if AI can materially contribute to a top-tier mathematical problem, then dismissals of AI’s ability to do serious technical work become harder to sustain [@theo](https://x.com/theo/status/2097540749663551704).
- Hrishioa’s “high compute regime” framing suggests strategic consequences for labs and startups: those who underweight inference-time compute orchestration may be planning against the wrong frontier [@hrishioa](https://x.com/hrishioa/status/2097542911382761630).

**Skeptical / cautionary perspectives**

- The implicit skeptical position is mathematical: until a theorem statement, full proof, and expert vetting exist, calling this a “solution” is premature. The joke-post itself acknowledges this by stressing that field-wide acceptance remains pending [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954).
- Another skepticism target is narrative compression: “10,000 agents solved Navier–Stokes” can obscure how much was due to human framing, filtering, or verification. The tweets do not disclose authorship proportions.
- There is also a reproducibility concern: without artifacts, independent researchers cannot judge whether the breakthrough was robust, cherry-picked, or a one-off.

**Neutral / analytic perspectives**

- A neutral reading is that this is notable even if the proof fails. If a system can generate mathematically nontrivial candidate pathways on a problem of this stature, that alone is a meaningful capability milestone.
- Another neutral view is to separate **scientific truth** from **systems innovation**. Even if the theorem claim does not hold, the multi-agent RL + parallel test-time compute architecture may still represent an important advance in AI research methodology.
- The conversation also reveals a shift in what people now count as “capability.” The debate is moving from benchmark scores to **real-world cognitive labor decomposition at scale**.


## Why this matters in context


**This sits at the intersection of three ongoing shifts in frontier AI.**

- **From static models to agent systems:** The central disclosed ingredient is not a single chatbot-like model but a large collaborative population of agents [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- **From training-time scaling to inference-time scaling:** The emphasis on “unstructured parallel test-time compute” directly aligns with a broader industry pivot toward spending compute at solve time, not just pretraining time [@__eknight__](https://x.com/__eknight__/status/2097538148754727260).
- **From benchmark theater to domain claims:** Navier–Stokes is socially legible in a way benchmark deltas are not. A claim touching a Millennium Problem instantly broadens the audience and raises epistemic stakes.

**Why Navier–Stokes specifically is symbolic**

- The Millennium Problems function as cultural shorthand for the hardest kinds of formal intellectual work.
- Progress here would suggest AI systems are not just speeding up known workflows but entering domains where correctness is brittle and prestige filters are extremely strict.
- That said, mathematics is unusually unforgiving: unlike many product tasks, there is no room for “mostly right.” This is why external validation dominates the discourse.

**Implications if the claim is substantiated**

- Strong evidence for **distributed theorem search** as a serious research paradigm.
- New pressure on formal methods tooling to absorb model-generated proof candidates.
- A likely acceleration in AI-for-math investment, especially around orchestration, verifier coupling, and scalable search.
- A broader update on the usefulness of **test-time compute** and **multi-agent RL** beyond coding agents and office automation.

**Implications even if the claim does not fully hold**

- It still publicizes OpenAI’s internal strategic direction: large-scale agent collaboration as a core capability area.
- It changes expectations about where compute is being spent and what kinds of demonstrations labs will use to signal frontier progress.
- It may spur competitors to disclose similar systems or rush out rival “AI did science” claims.


## The drama around authorship, disclosure, and who gets to speak


**A secondary thread of the discussion was about whether details were being indirectly revealed, who was authorized to reveal them, and how much people should infer from fragments.**

- A tweet saying “Roon seems like the kind of person who would honor his NDA tbh.” points to a social layer around the story: some observers expected better-known insiders or adjacent figures to stay quiet, while details were instead being pieced together from others [@jd_pressman](https://x.com/jd_pressman/status/2097540233692889322).
- Theo’s “AI can’t ACTUALLY code crash out moment” post also functioned as social provocation, framing critics as emotionally reacting to a capabilities update rather than engaging first with proof standards [@theo](https://x.com/theo/status/2097540749663551704).
- The two tweets about an “OpenAI movie” image and guessing who appears in it are not about the Navier–Stokes claim directly, but they reflect a parallel tendency to map internal OpenAI narratives onto named personalities like Greg Brockman, Ilya Sutskever, Jared Kaplan, Dario Amodei, and Paul Christiano, even when evidence is thin [@willdepue](https://x.com/willdepue/status/2097363280809382183), [@jachiam0](https://x.com/jachiam0/status/2097368747095068791). In the context of the Navier–Stokes discussion, that tendency matters because people quickly personalize technical claims into author-credit and insider-drama questions.
- The joke and speculation posts show a familiar pattern in frontier AI launches: sparse official detail creates a vacuum that gets filled by memes, leaked-sounding fragments, extrapolation, and overclaiming [@LearnOpenCV](https://x.com/LearnOpenCV/status/2097541292352065954), [@teortaxesTex](https://x.com/teortaxesTex/status/2097544071162085714).

**Why the authorship/drama issue matters technically**

- For a mathematics claim, provenance is not just gossip. It affects:
  - who framed the conjecture,
  - who selected candidate lemmas,
  - whether the proof was machine-generated or machine-assisted,
  - what credit assignment looks like,
  - how much trust experts place in the artifact.
- In AI research, “multi-agent solved X” also muddies standard notions of contribution. If thousands of agents searched in parallel, then:
  - what is the “author” of the proof,
  - what is the role of the orchestration team,
  - and what exactly should be cited or reproduced?
- NDA and disclosure norms become especially salient when a claim is large enough to move public beliefs before a paper or proof is available.


**OpenAI’s Navier–Stokes Result, Credit Dispute, and the Emergence of Massive Test-Time Compute**

- **OpenAI’s proposed Navier–Stokes solution** dominated the day. OpenAI said an internal model “**significantly more capable than GPT-6 Astra**” produced a proposed proof in **88 hours** using roughly **10,000 agents**, followed by another **17 hours** of Lean formalization/verification with Astra, according to summaries and reactions from [@TheTuringPost](https://x.com/TheTuringPost/status/2097508815143137596), [@polynoamial](https://x.com/polynoamial/status/2097375837670785447), and [@sama](https://x.com/sama/status/2097380249910854023). OpenAI stressed that its proof differs from the independent researchers’ work and addresses a different Euler setting; it also said **no specific user data was accessed** for this effort, while conceding it **cannot rule out de-identified derivative data** from product usage having helped model improvement more generally in the past [@OpenAI](https://x.com/OpenAI/status/2097375276384567642).
- **The technical meta-point is test-time compute scaling**. Several observers highlighted the implied economics and trajectory: what cost millions today could become consumer-accessible quickly, just as ARC-AGI costs collapsed from hundreds of thousands to tens of dollars [@polynoamial](https://x.com/polynoamial/status/2097375837670785447). Others estimated the proof run at **130B output tokens** and perhaps **$10M–$40M API-equivalent cost** depending on input-token scale assumptions [@scaling01](https://x.com/scaling01/status/2097380355451830752). The strongest consensus signal was that **unstructured parallel test-time compute plus orchestration** is now a first-class scaling axis, not just pretraining or post-training [@__eknight__](https://x.com/__eknight__/status/2097538148754727260), [@eliebakouch](https://x.com/eliebakouch/status/2097401843319996455).
- **The controversy centered on priority, data contamination, and norms.** Sam Altman and Sébastien Bubeck argued OpenAI heard rumors that Anthropic-associated researchers had solved a Millennium problem, then tested whether OpenAI’s models could do the same; when OpenAI learned the other team had **Euler but not Navier–Stokes**, it says it offered coordination, priority on Euler, and possible lead authorship for Tristan Buckmaster on a rewrite of OpenAI’s proof [@sama](https://x.com/sama/status/2097385167002415140), [@SebastienBubeck](https://x.com/SebastienBubeck/status/2097379411691516310). Critics focused less on direct spying—which many deemed unlikely—and more on whether **derived user data** or public rumors should have triggered stricter checks, and on whether this behavior will chill open scientific exchange [@aidangomez](https://x.com/aidangomez/status/2097381789039837637), [@johnschulman2](https://x.com/johnschulman2/status/2097440545853637108), [@simonw](https://x.com/simonw/status/2097474703380365698).
- **Mathematicians’ reaction is becoming a substantive governance issue.** Terence Tao’s cautionary comments, amplified by [@fchollet](https://x.com/fchollet/status/2097444630552014920) and [@GaryMarcus](https://x.com/GaryMarcus/status/2097446464041660608), framed the key risk: if even rumors of progress can trigger industrial-scale AI efforts that “flatten” a research direction, fields may move toward secrecy and away from long-standing open-science norms. Separately, [@stevenstrogatz](https://x.com/stevenstrogatz/status/2097503570903928890) emphasized that prior public work by **Córdoba and Martínez-Zoroa** supplied the key strategy that others built on.

**Meta’s Muse Launch and the Personal-Agent Security Architecture**

- **Meta launched Muse**, a consumer-facing “personal AI agent” positioned as always-on, app-connected, browser-capable, and goal-oriented, with strong distribution through Meta properties and integrations [@finkd](https://x.com/finkd/status/2097402101332590646), [@alexandr_wang](https://x.com/alexandr_wang/status/2097402344061510004), [@MetaNewsroom](https://x.com/MetaNewsroom/status/2097400062544425022). Product details repeatedly surfaced: **persistent isolated Linux VMs**, browser use, WhatsApp/app interfaces, and connectors to services like Gmail, Calendar, Outlook, Plaid, OpenTable, Docs, Spotify, Peloton, plus unique Meta-native connectors for Instagram, Messenger, Facebook, and Marketplace [@alexandr_wang](https://x.com/alexandr_wang/status/2097454574202495340).
- **Security architecture is the differentiator being pushed hardest.** Meta’s team said each Muse runs in its own **secure VM**, actions are mediated by a separate **Sentinel**, secrets are never directly exposed to the agent, sensitive actions require approval, and there is a public **bug bounty up to $300k** [@shengjia_zhao](https://x.com/shengjia_zhao/status/2097402766989926911), [@alexandr_wang](https://x.com/alexandr_wang/status/2097405157319541135). There’s also explicit commerce infrastructure: **Stripe Link** for payments with an **agentic payment protection / refund guarantee**, plus incoming **Shop Pay** integration [@alexandr_wang](https://x.com/alexandr_wang/status/2097410373221773355).
- **Early reception from practitioners was notably positive**, especially on permissioning, secrets management, and consumer utility. Commentary from [@matthuang](https://x.com/matthuang/status/2097406663339000052), [@signulll](https://x.com/signulll/status/2097416338147049795), and [@lilyjclifford](https://x.com/lilyjclifford/status/2097479117902070069) suggests Muse may be one of the first broadly legible personal-agent products where **context and access**, not raw model IQ, are the bottleneck. Meta also said usage exceeded internal projections by **10x** on day one [@alexandr_wang](https://x.com/alexandr_wang/status/2097527621206921612).
- **Model and ecosystem placement:** Meta’s **Muse Spark 1.3** was quickly exposed in third-party tooling like Cursor [@cursor_ai](https://x.com/cursor_ai/status/2097402609531236708), while arena-style benchmarking positioned **Muse Spark 1.3 Max** as price/perf competitive in web-dev coding workloads [@arena](https://x.com/arena/status/2097464147890118945).

**OpenAI’s Image 2.5 Release and Astra Rollout**

- **OpenAI also shipped ChatGPT Images 2.5**, though it was partially overshadowed. The release emphasizes **up to 50% lower latency vs Images 2.0**, better realism, stronger edit consistency across repeated edits, comment-based localized changes, transparent backgrounds, and a new **Sketch** tool for guided generation [@OpenAI](https://x.com/OpenAI/status/2097394956457623964), [@ChatGPT](https://x.com/ChatGPT/status/2097411337064227032), [@sama](https://x.com/sama/status/2097410967978324010).
- **Two API variants were introduced**: **GPT-Image-2.5 Flare** for speed/quality and **Sunburst** for higher-precision detailed work [@reach_vb](https://x.com/reach_vb/status/2097399096000581655). Arena results claimed **#1 and #2 positions** across text-to-image, image-edit, and multi-image-edit leaderboards, with especially large gains in multi-image editing [@arena](https://x.com/arena/status/2097400515546255754). Integrations landed quickly on **fal**, **Higgsfield**, **Manus**, and **Hermes Agent** [@fal](https://x.com/fal/status/2097417427168428356), [@higgsfield](https://x.com/higgsfield/status/2097421079824543776), [@ManusAI](https://x.com/ManusAI/status/2097419357395792375), [@Teknium](https://x.com/Teknium/status/2097465800231883091).
- **Astra availability widened materially.** OpenAI said **GPT-6 Astra** is now fully rolled out to **Plus, Pro, Business, and Enterprise** users in Codex and ChatGPT Work [@OpenAI](https://x.com/OpenAI/status/2097431322117476423). Community demos showed strong practical computer-use performance: [@theo](https://x.com/theo/status/2097435069900341544) reported Astra compiling and running **Super Smash Bros. Melee** on macOS at **120 FPS** after a roughly **6-hour** loop, while Vals reported Astra nearly saturating an unreleased computer-use eval by building a **Minecraft Nether portal** in under **3 hours** with no specialized harness [@ValsAI](https://x.com/ValsAI/status/2097447789630542024).

**Agent Harnesses, Post-Training, and Serving Infrastructure**

- **Harvey + Baseten’s M&A diligence work is one of the clearest model-harness co-optimization case studies.** Their **recursive language model (RLM) harness** uses a root agent to search a data room, delegate to sub-agents for document review, and aggregate findings over corpora up to **80M tokens**. On the synthetic **LAB Diligence** benchmark, moving from a standard tool loop to the RLM harness raised mean rubric pass rate from **23% to 62%** across models [@harvey](https://x.com/harvey/status/2097372371195953272), [@nikogrupen](https://x.com/nikogrupen/status/2097370187674869803).
- **Post-training inside the harness mattered at least as much as the harness itself.** Harvey reports self-distilled SFT on **GLM-5.2** improved pass rate **46% → 60%**, while **GRPO** on **Qwen3.5-122B-A10B** lifted pass rate **30% → 63%** on held-out rooms and improved document coverage **62% → 96%** [@harvey](https://x.com/harvey/status/2097372371195953272). The broader implication, echoed by others, is that **agent benchmarks increasingly need to treat orchestration and post-training as part of the model system**, not external glue.
- **LangChain/deepagents shipped quality-of-life primitives for harness design**, including **subagent forking** that passes supervisor context down to subagents, plus **managed connections** to abstract OAuth/token/consent flows for either agent-owned or user-owned identities [@colifran_](https://x.com/colifran_/status/2097377522623389865), [@hwchase17](https://x.com/hwchase17/status/2097410530717704546), [@caspar_br](https://x.com/caspar_br/status/2097424144459874412). This is a useful sign of the stack maturing around long-horizon agent workloads.

**Inference and Systems: Sparse Attention, Agentic Serving, and Decode Megakernels**

- **vLLM’s long-context serving work is notable.** The project described **Hybrid HiSparse** for sparse-MLA models: KV stays on GPU while possible, then **cold KV pages are offloaded to host memory**, while a hot buffer serves the indexer. On **GLM 5.3** with **1M context** on an **8×H200** node, configured concurrency **32**, plain offloading sustained **5–6** requests while Hybrid HiSparse sustained **19–25** [@vllm_project](https://x.com/vllm_project/status/2097397769338282222). This matters directly for **RL rollouts and long-context concurrency**, where VRAM-bound decode otherwise kills throughput.
- **vLLM also published a full-stack optimization pass for real-world agent traffic**, benchmarked on **AgentX**. Key takeaways: pipeline parallelism helps cold long prompts but loses on warm short turns; decode context parallelism depends strongly on the model’s attention stack; and **session-sticky routing** can beat naive load balancing because warm KV caches matter more than even queue distribution in fast-turn agent settings [@vllm_project](https://x.com/vllm_project/status/2097427310513426721).
- **Cohere introduced an open-source serving stack built around a “decode megakernel,”** claiming up to **1.58×** faster performance than vLLM on **North Mini Code** and **1.25×–1.41×** end-to-end gains at higher batch sizes [@cohere](https://x.com/cohere/status/2097410772355666393). Combined with Baseten’s note that frontier RL rollouts now get **new policy weights live in under 40 seconds** globally with only a **6-second pause** [@baseten](https://x.com/baseten/status/2097407857855803799), the clear trend is toward infra specialized for **continuous post-training and rollout refresh**, not static model serving.

**Top Tweets (by engagement)**

- **Anthropic resignation / safety warning**: Jacob Hilton resigned from Anthropic, arguing both Anthropic and OpenAI are racing toward self-improving superintelligence irresponsibly and that insiders privately treat extinction risk as real [@hilbertspaess](https://x.com/hilbertspaess/status/2097476196791709843), with follow-up claims that current systems could soon hack infrastructure and transform fields rapidly [@hilbertspaess](https://x.com/hilbertspaess/status/2097476201283834281).
- **OpenAI’s user-data clarification**: OpenAI’s formal statement that no specific user data was accessed for Navier–Stokes, alongside the caveat about possible de-identified derivative improvement, became a major flashpoint [@OpenAI](https://x.com/OpenAI/status/2097375276384567642).
- **Cognition financing**: Cognition announced a raise of **$2B+ at a $48B valuation**, saying run-rate revenue grew from **$492M to nearly $900M** since May [@cognition](https://x.com/cognition/status/2097369798518681891).
- **Meta Muse launch**: Mark Zuckerberg’s launch post for **Muse** was among the highest-engagement product tweets of the day [@finkd](https://x.com/finkd/status/2097402101332590646).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Chinese Multimodal AI Releases: Driving and Flash APIs

  - **[Qwen/Qwen-Drive-1.0-4B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1wauxg9/qwenqwendrive104b_hugging_face/)** (Activity: 549): ****Qwen** released [`Qwen/Qwen-Drive-1.0-4B`](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B), an open-weight autonomous-driving VLM derived from an unchanged Qwen3.5 4B VLM, with a full BF16 checkpoint around `9B` and extra `planner-sft`, `planner-rl`, and `perception` modules. Per the linked [technical report](https://arxiv.org/pdf/2609.00111), Qwen-Drive-1.0 adds an external BEV perception head for **3D object detection, semantic occupancy prediction, and BEV map segmentation**, plus a Planning Expert for future ego-trajectory generation, trained via staged mixtures of driving supervision and general VLM data. The release reports competitive performance across WOD-E2E, NAVSIM, driving VQA, and open-/pseudo-closed-/closed-loop planning evaluations while largely preserving general multimodal capability.**


  - **[DeepSeek Flash 4.1 is already being tested via API and rolling out.](https://www.reddit.com/r/LocalLLaMA/comments/1wan3nl/deepseek_flash_41_is_already_being_tested_via_api/)** (Activity: 528): ****DeepSeek V4.1 Flash** is reportedly in internal beta via API: keep the existing `base_url` and call model `deepseek-v4.1-flash-expires-on-0910`, with pricing unchanged from `deepseek-v4-flash` and a `20` concurrent request/account limit ([source](https://x.com/kimmonismus/status/2097286327909675477)). The translated announcement claims a “new model architecture” with **native multimodal support**, stronger capability, faster throughput, and lower cost; commenters report roughly `2.24×` speedup and up to `~30%` better token efficiency in benchmarks, though one edit speculates the observed speed gain may be partly due to lower beta concurrency rather than architecture alone.** Comment sentiment is strongly positive toward DeepSeek/open-weight progress, but the only substantive debate is whether the claimed performance improvement reflects a genuinely new architecture or simply lighter API load during beta testing.

    - Users report that **DeepSeek Flash 4.1** appears to be around `2.24x` faster via API testing, with some speculation that the observed speedup may come from **lower concurrent load** rather than a fundamentally new architecture. Other comments suggest it may be **multimodal**, though this is not yet confirmed in the thread.
    - One technically relevant claim is that some users are seeing up to **`30%` better token efficiency in benchmarks**, which could explain DeepSeek’s reported “lower costs” messaging if fewer tokens are needed for comparable outputs. The comment frames this as benchmark-dependent and not yet independently validated.
    - There is some discussion of release cadence and migration complexity: users mention not having fully moved from the **0731** model to the newer **vision variant** before another release appears imminent. This highlights a practical API-integration issue where fast model iteration can outpace downstream evaluation, regression testing, and deployment workflows.


### 2. Efficient Local Models and Quantization Benchmarks

  - **[MiniCPM5-2B Release Day](https://www.reddit.com/r/LocalLLaMA/comments/1w9skjz/minicpm52b_release_day/)** (Activity: 495): ****OpenBMB** released **MiniCPM5-2B**, an open-weights `2B` model on [Hugging Face](https://huggingface.co/openbmb/MiniCPM5-2B) with code/resources on [GitHub](http://github.com/OpenBMB/MiniCPM). The post claims it scores `15` on **Artificial Analysis Intelligence Index v4.2**, described as the highest score among open-weight models at `≤4B` parameters.** Commenters framed the result as notable small-model progress, with one claiming current `2B` models now score similarly to “gpt oss 120b.” Practical interest centered on deploying it in low-resource pipelines such as `ASR -> MiniCPM5-2B -> TTS`, and comparing its task utility against **Ling Tiny 3.0** on mini-PC automation workloads.

    - A commenter claims current `2B` models are reaching benchmark scores comparable to **GPT-OSS-120B**, implying a large efficiency jump in small-parameter models, though no specific benchmark table was cited in the thread. Another user framed the key comparison as **MiniCPM5-2B vs Ling Tiny 3.0**, noting Ling Tiny 3.0 provides nearly `8B` parameters for local mini-PC automation where reasoning quality still matters.
    - Several users focused on low-latency local deployment use cases rather than coding, including an **ASR → MiniCPM5-2B → TTS** voice pipeline. The implied technical appeal is that a fast `2B` model could support interactive speech-agent loops on constrained hardware if latency and quality are sufficient.
    - One commenter pointed out that **OpenBMB** released a **DSpark** variant for MiniCPM5-2B, apparently optimized for higher tokens/second: [MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark). This is the most concrete implementation detail in the thread, suggesting users should evaluate the DSpark build specifically when benchmarking local throughput.

  - **[My Qwen3.8-27B task-aware quant reaches 99% of BF16 reasoning performance at 15% of the size.](https://www.reddit.com/r/LocalLLaMA/comments/1wa5dp9/my_qwen3827b_taskaware_quant_reaches_99_of_bf16/)** (Activity: 325): **The author reports a **TAK (Task Aware Knapsack)** quantization pipeline for reasoning-specialized GGUF-style models, using a task-specific **imatrix** plus tensor-level “damage allocation” under a byte budget—explicitly **no pruning, fine-tuning, or merging**—with releases on [Hugging Face](https://huggingface.co/ByteOtter). On a held-out reasoning benchmark, the headline **Qwen3.8-27B** TAK quant scores `82.81%` vs `83.59%` BF16 and `77.34%` byte-matched **Unsloth UD IQ2_S**, with additional reported gains over Unsloth on **Qwen3.5-4B** (`73.44%` vs `61.72%`), **Gemma 4 E4B** (`69.53%` vs `55.47%`), and **Gemma 3 4B QAT** (`54.69%` vs `35.16%`). The author notes the current quant is **reasoning-specialized** and acknowledges user reports of **repetition loops in coding**, which they plan to reproduce and characterize.** Top comments are skeptical that an imatrix-based ~Q2 quant will generalize beyond the narrow reasoning benchmark, asking for **Q4/Q6 variants** and broader benchmark suites such as Qwen’s own evals. Another commenter posts a negative example/image and calls the result “Garbage,” while others specifically request direct comparisons against **Q4** rather than only byte-matched low-bit Unsloth baselines.

    - Commenters questioned whether an `imatrix Q2` quantization can generalize beyond the reported reasoning benchmark, asking for comparisons against `Q4`/`Q6` and a broader benchmark suite matching the official **Qwen3.8-27B** evaluations. The core concern is that *task-aware* calibration may preserve a narrow benchmark distribution while degrading “real world or varied use.”
    - One commenter argued that recent **Unsloth Dynamic 3.0**-style quantization has made very small quantizations more viable on dense models, specifically claiming **Qwen3.8-27B** “quantises SOOO well.” This supports the idea that aggressive low-bit quantization may be unusually effective for this model family, though no benchmark numbers were provided in the comment.
    - A related implementation example was shared for **Ornith 35B-A3B**: the commenter used an `imatrix` calibration corpus aligned to *agentic coding* plus a custom chat template, effectively applying the same task-aware quantization principle. They claimed the resulting model remained strong on out-of-corpus agentic coding tasks after the training cutoff and became highly trending on Hugging Face.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo


### 1. OpenAI Navier–Stokes Claim and Authorship Dispute

  - **[Millenium Prize solution discovered at OpenAI](https://www.reddit.com/r/OpenAI/comments/1wav1l6/millenium_prize_solution_discovered_at_openai/)** (Activity: 1130): **The [image](https://i.redd.it/mi1582vh0coh1.png) appears to be an **unverified/satirical screenshot**, not a confirmed technical announcement: it claims an OpenAI internal model solved the **Navier–Stokes Millennium Prize problem** in `88 hours` using ~`10,000` coordinating AI agents, alongside a benchmark-style chart comparing “GPT-6 Astra” vs. an “Internal Model” on open math problems with increasing test-time compute. If taken literally, the setup would imply roughly `88 × 10,000 = 880,000 agent-hours`—about `100 agent-years`—but there is no cited paper, proof, repository, or official validation in the post text/comments.** Commenters were broadly skeptical, with one noting *“Let’s wait till it solves real math problems”* and another framing the result as brute-force compression of mathematician-years via compute. There was also mention of a “big controversy around the human portion of this solution,” implying concern that any claimed breakthrough may rely on undisclosed human contribution rather than autonomous discovery.

    - One commenter estimates the effort as roughly `88 hours × 10,000 agents ≈ 100 years` of aggregate agent time, framing the result as massively parallel mathematical search rather than a single-model breakthrough. The technical implication is that frontier systems may compress long-horizon exploration into days if the problem can be decomposed or sampled effectively across many agents.
    - There is concern about the undisclosed internal model used for the result, specifically whether it is comparable in size/cost to **Astra** or significantly larger and more expensive. The key technical issue raised is reproducibility and economics: without model scale, inference cost, and orchestration details, it is hard to assess whether the result is broadly meaningful or only achievable with exceptional compute.
    - A commenter points to controversy around the “human portion” of the solution, implying unresolved questions about attribution, verification, and how much of the final proof was generated by AI versus guided or repaired by human experts. For a Millennium Prize-level claim, that distinction matters technically because formal correctness, proof provenance, and independent reproducibility are central to evaluating the result.

  - **[OpenAI threatened to ruin star mathematician's career](https://www.reddit.com/r/OpenAI/comments/1wayuay/openai_threatened_to_ruin_star_mathematicians/)** (Activity: 1558): **The [image](https://i.redd.it/tm72mtvzncoh1.png) is a screenshot of highlighted text from an alleged/“verified” statement by **Tristan Buckmaster**, a tenured NYU mathematician, describing a dispute with **OpenAI** over a purported Navier–Stokes result and proposed shared authorship. The technically relevant issue is not a benchmark or implementation detail, but an authorship/provenance controversy: the highlighted passages reportedly question how much model output, compute, training/access to user data, or human mathematical work contributed to the claimed solution, culminating in the alleged line *“Why would you ruin your career?”* after Buckmaster declined authorship terms.** Comments largely interpret the exchange as coercive rather than technical; one compares OpenAI’s behavior to Amazon allegedly using platform access to copy and undercut sellers. Another top comment asks for an ELI5, indicating readers found the authorship/provenance dispute hard to parse from the screenshot alone.



### 2. GPT-6 Astra Computer-Use Benchmarks

  - **[Today Astra is doing 100% of my job](https://www.reddit.com/r/OpenAI/comments/1waqlhc/today_astra_is_doing_100_of_my_job/)** (Activity: 1777): **The image ([JPEG](https://i.redd.it/vcm3dgq28boh1.jpeg)) shows a real electronics/CAD workstation with PCB layout, Fusion 360-style enclosure modeling, prototype hardware, tools, and overlays reading “ChatGPT is using your computer,” matching the post’s claim that **Astra/ChatGPT** is autonomously driving EasyEDA, Fusion 360, and DSP firmware benchmarking. Technically, the post is an anecdotal demo/claim rather than a benchmark or reproducible implementation: it describes AI-assisted PCB design, mechanical CAD, and audio/DSP self-testing via a sound card for an open-source Alexa-like voice assistant, but provides no code, metrics, API details, or validation results. The image is partly promotional/meme-like because the central point is the “AI is doing my whole job” moment rather than a verifiable engineering result.** Commenters were split between amazement and anxiety: one said it made them feel “obsolete,” while another warned that full automation is dangerous if the AI does “100% of your job wrong” and the engineer stops checking its work.

    - A commenter raised a technical/operational risk around full job automation: if Astra performs `100%` of the workflow, users may stop auditing outputs and lose the ability to detect silent failures. The concern is less about capability and more about **human-in-the-loop degradation**, where unchecked automation can produce incorrect results that go unnoticed until downstream impact.

  - **[FactorioBench just dropped ;-)](https://www.reddit.com/r/singularity/comments/1w9vpyv/factoriobench_just_dropped/)** (Activity: 1043): **The [image](https://i.redd.it/dg58ombfc4oh1.png) is a screenshot of a tweet by **Derya Unutmaz, MD** claiming **GPT-6 Astra** autonomously played *Factorio* for `15 minutes`, learned controls, gathered resources, built a miner/furnace, mined coal, and saved the game; the Reddit title frames this as a joking/early “**FactorioBench**” benchmark, with the full thread linked on [X](https://x.com/DeryaTR_/status/2096791759166595130). Technically, this is **not a formal benchmark result**—it is an anecdotal agent-control demo in a complex strategy/automation game, where meaningful evaluation would require metrics like time-to-rocket, production efficiency, recovery from errors, planning horizon, and reproducibility.** Commenters were mostly enthusiastic about using strategy/building games as AI benchmarks, suggesting milestones such as *“send a rocket in less than 4hrs”* or a `100%` run under `10hrs`. One commenter noted that [other people are attempting similar Factorio-agent demos](https://preview.redd.it/xn1ckawvd4oh1.png?width=933&format=png&auto=webp&s=862ef39015cc3d31629e3f89684d3272f0579b56), implying a broader informal trend rather than a single validated result.

    - Several commenters framed Factorio as a useful agent benchmark because it tests long-horizon planning, resource routing, automation design, and recovery from compounding failures rather than just text reasoning. One user argued that vanilla Factorio is relatively deterministic and resembles “designing CPUs/basic software development,” with the main hard cases being timely defense setup, biter clearing, and surviving enemy evolution thresholds.
    - A technical caveat raised was that an LLM-based Factorio agent is unlikely to be learning the game from scratch: its pretraining data may already contain extensive Factorio strategies, ratios, blueprints, and progression knowledge. The more meaningful milestone is therefore the **embodiment/execution gap**: converting stored theoretical knowledge into robust in-game actions over many hours.
    - One commenter suggested concrete aspirational benchmark targets: achieving AGI-like competence when an agent can **launch a rocket in under `4` hours** and complete a **`100%` run in under `10` hours**. Another noted that similar Factorio-agent efforts are already underway, referencing an external screenshot: https://preview.redd.it/xn1ckawvd4oh1.png?width=933&format=png&auto=webp&s=862ef39015cc3d31629e3f89684d3272f0579b56

  - **[Astra is now a certified human](https://www.reddit.com/r/singularity/comments/1w9mp3k/astra_is_now_a_certified_human/)** (Activity: 1036): **The [image](https://i.redd.it/octby766a2oh1.png) is a humorous Neal.fun-style certificate declaring **“Astra”** a **“Verified Human”** after allegedly beating all `48` levels of the *I’m Not a Robot* game, sourced from [Sharif Shameem’s tweet](https://twitter.com/sharifshameem/status/2096847916837314853). Technically, the post frames this as a notable AI-agent/browser-control milestone: the game functions like a compact multimodal reasoning and interaction benchmark requiring perception, planning, and UI manipulation rather than a standard static eval.** Comments mainly debate whether this should be considered “pre-AGI” or “AGI,” while at least one commenter is skeptical and asks for a full video or independent replication before accepting the claim.

    - A commenter questioned the evidentiary basis of the claim, saying it was *“kind of hard to believe”* without the full video or an independent replication attempt. The main substantive concern was reproducibility/verification rather than the result itself.


### 3. AI-Designed Drug Claims and Trials

  - **[An experimental AI-created drug for an incurable lung disease had a surprising effect during trials: it made the body's biological age indicators drop by 6 years, towards a younger state.](https://www.reddit.com/r/singularity/comments/1wa1a5n/an_experimental_aicreated_drug_for_an_incurable/)** (Activity: 1102): ****Insilico Medicine’s** AI-designed experimental IPF drug **Rentosertib** reportedly targets **TNIK**, a kinase implicated in fibrosis and aging-linked pathways, with candidate design aided by its **Chemistry42** platform ([dev.ua](https://dev.ua/en/news/stvorenyi-shi-preparat-zmenshyv-pokaznyky-biolohichnoho-starinnia-orhanizmu-1788791333)). In trial blood-sample analyses, Rentosertib-treated patients showed reduced protein-based biological-age estimates on models including **ProtAge** and **OrganAge**, with one analysis suggesting an average shift of about **`6 years` younger** relative to little/no placebo effect.** A technically informed commenter argued the result is promising but not necessarily evidence of a general anti-aging therapy: biological-age biomarkers can improve when an underlying disease such as **idiopathic pulmonary fibrosis** is successfully treated, so effects in healthy people would require separate trials. Another comment was skeptical of the headline framing as combining investor-attractive keywords like AI, cancer, and aging.

    - A commenter with domain-adjacent experience cautioned that the reported `~6 year` drop in biological-age indicators may be a **disease-treatment artifact** rather than evidence of general anti-aging. In idiopathic pulmonary fibrosis (IPF), successful reduction of disease severity could normalize age-associated biomarkers, so *“giving this drug to healthy patients may not actually lower their biological age.”*
    - A detailed critique noted that the “AI-created” drug was not produced by frontier LLMs, but by specialized drug-discovery pipelines such as **Chemistry42** and **PandaOmics**, which combine bespoke models for target discovery and chemistry generation. The same commenter emphasized that the reported rejuvenation signal came from **proteomic clocks**, not more established epigenetic clocks such as **GrimAge**, making the anti-aging interpretation indirect and weaker.
    - Several commenters highlighted the key translational limitation: **IPF itself disrupts the proteome**, so improving IPF would be expected to improve proteomic age markers without implying slowed or reversed aging in healthy people. They argued that the decisive test would require trials in **healthy, non-IPF participants**, while also noting that clinical-trial capacity may become the bottleneck as more AI-designed drug candidates emerge.

  - **[Insane times we live in](https://www.reddit.com/r/singularity/comments/1wb3yc0/insane_times_we_live_in/)** (Activity: 1261): **The image shows an X post claiming **Douglas Yao** synthesized “PAC-3310,” allegedly a **ChatGPT-designed selective M4 muscarinic receptor agonist** for schizophrenia, in a garage chemistry setup; the Reddit title frames this as an example of “insane times” in AI-assisted drug design. The selftext flags a major safety/ethics concern around the claim *“When administered to mice…”*, questioning whether any animal testing was done under proper lab oversight. [Image](https://i.redd.it/idbpj8e1mdoh1.jpeg)** Comments were split between alarm at DIY pharmacology and noting that Yao reportedly has a **computational biology PhD from Harvard**, so this may not be pure amateur chemistry—though users still emphasized they would not trust or consume garage-synthesized compounds.

    - Commenters emphasized that the person involved reportedly has a **computational biology PhD from Harvard**, suggesting the work is not a naïve “basement lab” effort but still may be far from market- or consumption-ready. The technical concern is less basic competence and more whether expertise plus accessible AI/lab tooling can lower the barrier to risky biological or chemical experimentation.
    - Several comments highlighted that **AI failure modes in chemistry/biology carry unusually high downside risk**, because incorrect synthesis guidance, contamination, dosage assumptions, or protocol errors can have direct health or biosafety consequences. One commenter framed the main risk as a well-intentioned operator with “just enough knowledge and access to tech” creating an accidental hazard outside institutional safety controls.



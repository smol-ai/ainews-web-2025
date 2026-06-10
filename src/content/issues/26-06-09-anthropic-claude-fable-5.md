---
id: MjAyNS0x
title: Anthropic Claude Fable 5
date: '2026-06-09T05:44:39.731046Z'
description: >-
  **Anthropic** released two major models: **Claude Fable 5** for general
  availability and **Claude Mythos 5** for restricted access, with fallback to
  **Claude Opus 4.8** for sensitive queries. **Fable 5** features a **1M-token
  context window** and pricing at **$10/million input tokens** and **$50/million
  output tokens**. It leads benchmarks in software engineering, knowledge work,
  scientific research, and vision, outperforming **GPT-5.5** and setting new
  state-of-the-art scores on **CursorBench**, **FrontierCode**, **Terminal-Bench
  2.1**, and **Artificial Analysis Intelligence Index**. The rollout includes
  Pro, Max, Team, and Enterprise plans with temporary usage credits due to
  capacity constraints. Middleware SDK support is available in **Python,
  TypeScript, Go, Java, and C#**.
companies:
  - anthropic
  - cursor_ai
  - cognition
models:
  - claude-fable-5
  - claude-mythos-5
  - claude-opus-4.8
  - gpt-5.5
topics:
  - benchmarking
  - software-engineering
  - knowledge-work
  - scientific-research
  - vision
  - context-windows
  - model-pricing
  - sdk
  - rate-limiting
people:
  - mikeyk
  - scaling01
---


**a quiet day.**

> AI News for 6/8/2026-6/9/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Top Story: Anthropic Claude Fable 5 and Mythos 5 release**

## What happened


**Anthropic released two versions of its next major model family: Claude Fable 5 for general availability and Claude Mythos 5 for restricted access.**

- Anthropic officially announced **Claude Fable 5** as its “first generally available Mythos-class model,” saying it exceeds any model it has previously made broadly available and is **state-of-the-art on nearly all tested benchmarks** [@claudeai](https://x.com/claudeai/status/2064394146916229443), [@claudeai](https://x.com/claudeai/status/2064394151441863006)
- Anthropic said **Fable 5 is the same underlying model as Mythos 5 with added safeguards**, and that some cyber/bio/chemistry/distillation-related prompts may be **routed to Claude Opus 4.8** instead [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428347678220691), [@scaling01](https://x.com/scaling01/status/2064398688802205900)
- Anthropic stated that for a “narrow range” of potentially harmful topics, **queries transparently fall back to Opus 4.8**, and claimed **95%+ of sessions never see one** according to early user-facing messaging [@claudeai](https://x.com/claudeai/status/2064394155258765783), [@mikeyk](https://x.com/mikeyk/status/2064392996288901392)
- Anthropic developer messaging said fallback is available server-side and via SDK middleware in **Python, TypeScript, Go, Java, and C#** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428351029449214)
- Pricing for **both Fable 5 and Mythos 5** was reported as **$10 / million input tokens and $50 / million output tokens**; cache pricing was later reported by third-party evaluators as **$12.50 / million cache writes and $1 / million cache reads** [@scaling01](https://x.com/scaling01/status/2064394893603049625), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Fable 5 kept Anthropic’s **1M-token context window** according to Artificial Analysis [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Anthropic put Fable 5 into **Pro, Max, Team, and seat-based Enterprise plans until June 22**, then said it would require **usage credits** due to capacity constraints, with plans to restore broader subscription access later [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064394931033248226), [@scaling01](https://x.com/scaling01/status/2064394893603049625), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992), [@kimmonismus](https://x.com/kimmonismus/status/2064388066354028986)
- Confusion over the temporary inclusion was immediate; users asked what “included until June 22” meant and Anthropic staff clarified the rollout [@dejavucoder](https://x.com/dejavucoder/status/2064393509990523102), [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2064393574431764928)
- Anthropic later **reset 5-hour and weekly rate limits** across products after heavy demand [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064464557951852643)

## Official claims and third-party benchmark data


**Anthropic and partner platforms reported a broad benchmark lead, especially in coding and long-horizon agentic tasks.**

- Anthropic’s public claim: Fable 5 is especially strong in **software engineering, knowledge work, scientific research, and vision**, and **its lead increases with task length and complexity** [@claudeai](https://x.com/claudeai/status/2064394151441863006)
- Cursor said Fable 5 set a new **CursorBench SOTA at 72.9%**, **8 points above the previous best** [@cursor_ai](https://x.com/cursor_ai/status/2064394824313376787)
- Cognition said Fable 5 took the **#1 spot on FrontierCode**, and Devin integrated it into Devin Cloud Ultra, Desktop, and CLI [@cognition](https://x.com/cognition/status/2064398549073453266), [@cognition](https://x.com/cognition/status/2064398551539761387)
- Cline reported Fable 5 at **88.0% on Terminal-Bench 2.1**, beating GPT-5.5 by **4.6 points** [@cline](https://x.com/cline/status/2064427461212045546)
- Artificial Analysis placed Fable 5 **#1 on its Intelligence Index at 64.9**, roughly **5 points ahead of GPT-5.5**, and said Anthropic occupied the top two spots [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Artificial Analysis also reported:
  - **GDPval-AA Elo 1932**, #1 on agentic real-world knowledge work [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064414308289937869)
  - **53% on Humanity’s Last Exam**, more than **7 points ahead** of the next-best model, while fallback triggered on **9% of HLE tasks** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
  - **~8% fallback routing across Intelligence Index tasks**, mostly on scientific questions [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
  - Anthropic stated fallback occurs in **fewer than 5% of sessions on average** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064414308289937869)
- Community benchmark summaries highlighted very large deltas in coding:
  - **SWE-Bench Pro: Fable 5 80.3% vs GPT-5.5 58.6%** [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2064396097075003739)
  - **FrontierCode Diamond: Mythos 5 30.9% vs second-best 13.4%** [@scaling01](https://x.com/scaling01/status/2064391295620010383)
  - **Anthropic ECI 161.29 for Mythos 5** [@scaling01](https://x.com/scaling01/status/2064392088003756431)
- Artificial Analysis noted that Fable 5’s knowledge benchmark jump on **AA-Omniscience** could imply a **larger model than prior public Anthropic models**, though that is inference rather than confirmed spec [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)

## Product behavior, usage profile, and deployment details


**The release was defined as much by workflow changes and cost profile as by raw evals.**

- Anthropic staff and early users repeatedly described Fable 5 as a model for **very long, high-effort tasks**, with users shifting from giving it tasks to giving it **objectives/responsibilities** [@felixrieseberg](https://x.com/felixrieseberg/status/2064392202504310900), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064399512664526853), [@alexalbert__](https://x.com/alexalbert__/status/2064467657483829441)
- Anthropic advised users to default to **xhigh/high effort**, rewrite old CLAUDE.md instructions, and let the model use more judgment [@alexalbert__](https://x.com/alexalbert__/status/2064467657483829441)
- Anthropic’s developer messaging emphasized **multi-agent orchestration**, with Fable delegating to smaller models in Claude Managed Agents [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064394928948703406)
- Multiple testers described Fable as **slow, token-hungry, expensive**, but unusually capable:
  - Dan Shipper said it routinely used **500k to 1M tokens on tasks** and was best reserved for heavy jobs [@danshipper](https://x.com/danshipper/status/2064393970856124501)
  - Simon Willison called it “slow, expensive and capable” [@simonw](https://x.com/simonw/status/2064501565738930433)
  - Theo quickly hit limits and later welcomed Anthropic’s rate-limit reset [@theo](https://x.com/theo/status/2064442054772716020), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064464557951852643)
- Third-party and internal anecdotes emphasized large gains on long-running engineering tasks:
  - Ethan Mollick said he could hand it a **15-page design document** and it would work for **9+ hours** [@emollick](https://x.com/emollick/status/2064395281903346013)
  - Kimmonismus highlighted Anthropic’s claim that Stripe used Fable to do a **50-million-line Ruby migration in a day**, replacing what would have taken **a whole team over two months** [@kimmonismus](https://x.com/kimmonismus/status/2064401121515274747)
  - Victor Taelin reported Fable finding a subtle bug and producing claimed speedups up to **1770% in one case**, though he still needed to audit correctness [@VictorTaelin](https://x.com/VictorTaelin/status/2064448425936994742)
  - Anthropic-associated posts cited **430x kernel speedups**, **69x self-training speedups**, and **10x drug-design acceleration**, though these came from benchmark/system-card interpretations and should be treated as vendor-side claims unless independently replicated [@scaling01](https://x.com/scaling01/status/2064392386520780945), [@scaling01](https://x.com/scaling01/status/2064392809293939119), [@scaling01](https://x.com/scaling01/status/2064394250142265367)
- Ecosystem rollout was immediate: Fable 5 appeared in **Cursor, Devin, Notion, Microsoft Foundry, GitHub Copilot App/CLI, Cline, Replit, Base44, MagicPath, Arena, MCP Atlas** and more [@cursor_ai](https://x.com/cursor_ai/status/2064394824313376787), [@cognition](https://x.com/cognition/status/2064398549073453266), [@NotionHQ](https://x.com/NotionHQ/status/2064397568696819984), [@Azure](https://x.com/Azure/status/2064421301108834552), [@pierceboggan](https://x.com/pierceboggan/status/2064402677614911818), [@cline](https://x.com/cline/status/2064427461212045546), [@pirroh](https://x.com/pirroh/status/2064408022651191613), [@ScaleAILabs](https://x.com/ScaleAILabs/status/2064473993919537578)

## Safety architecture and the main controversy


**The biggest debate was not whether Fable/Mythos is strong; it was Anthropic’s decision to silently reduce usefulness on some frontier-AI-development tasks.**

- Anthropic’s system-card language, surfaced by multiple users, said: when Fable 5 is used for **frontier LLM development**, Anthropic may **limit the model’s effectiveness** via **prompt modification, steering vectors, and PEFT**, and that the user is **not notified**; Anthropic estimated this would affect roughly **0.03% of traffic** [@Hangsiin](https://x.com/Hangsiin/status/2064397550434816088), [@kimmonismus](https://x.com/kimmonismus/status/2064417460715962479)
- Anthropic also separately disclosed auto-rerouting for **cybersecurity and biosecurity** requests to Opus 4.8 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064394931033248226)
- This distinction mattered: **some risky queries are visibly rerouted/billed as Opus**, while **frontier-LLM-development requests may be silently weakened rather than rerouted or refused**
- Critics argued that this creates an **unlogged confounder** in research and engineering workflows:
  - “silent handicaps should not be a thing in a paid product” [@nrehiew_](https://x.com/nrehiew_/status/2064400440264179923)
  - “degrading performance on ML research without telling the user is shockingly hostile” [@deanwball](https://x.com/deanwball/status/2064434861088395730)
  - “safety interventions must be visible, auditable, and attributable” [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486)
  - “this is the trust break” because every poor result becomes ambiguous [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486)
- Several researchers framed it as **anti-competitive ladder-pulling** against open research and open weights:
  - “labs starting to pull up the ladders” [@natolambert](https://x.com/natolambert/status/2064404993193754830)
  - “this is the biggest wake-up call to protect and nourish open source AI” [@rasdani_](https://x.com/rasdani_/status/2064409800641859747)
  - “They didn’t mean pause AI research, they meant pause *your* AI research” [@bayeslord](https://x.com/bayeslord/status/2064437399292203401)
  - “original thinkers can't be an underclass” [@marksaroufim](https://x.com/marksaroufim/status/2064428421774753943)
  - “concentration of power, capabilities and economic wealth is the biggest risk in AI” [@ClementDelangue](https://x.com/ClementDelangue/status/2064513229099876663)
- Multiple users worried the classifier boundary was too broad or too error-prone:
  - one user said “the word cancer is flagged as a biosecurity risk” [@DeryaTR_](https://x.com/DeryaTR_/status/2064414826122866707)
  - another said Fable wouldn’t answer “What does the heart do?” [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2064524668208545955)
  - users in biology reported account-context differences, including being able to use Fable in **Incognito Mode but not normal mode** [@cremieuxrecueil](https://x.com/cremieuxrecueil/status/2064449457869984035)
  - Teknium and others reported refusal on simple engineering prompts [@Teknium](https://x.com/Teknium/status/2064462936677203983), [@Teknium](https://x.com/Teknium/status/2064466293185806658)
  - users reported PTX ISA questions and inference optimization queries getting flagged [@snowclipsed](https://x.com/snowclipsed/status/2064408466039390417), [@dejavucoder](https://x.com/dejavucoder/status/2064420742129967331)
- Some examples were humorous but pointed: users joked that asking for inference code caused the model to “start importing ONNX” or implementing JEPA, as a sign of capability steering [@vikhyatk](https://x.com/vikhyatk/status/2064515989795127744), [@MattVMacfarlane](https://x.com/MattVMacfarlane/status/2064440740483403829)

## Facts vs. opinions


**Facts / directly supported by release materials or benchmark posts**

- Fable 5 is generally available; Mythos 5 is restricted-access [@claudeai](https://x.com/claudeai/status/2064394146916229443), [@TheRundownAI](https://x.com/TheRundownAI/status/2064394481923699070)
- Fable 5 and Mythos 5 share the same underlying model with additional safeguards on Fable [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428347678220691), [@scaling01](https://x.com/scaling01/status/2064398688802205900)
- Pricing is **$10 / $50 per million input/output tokens** [@scaling01](https://x.com/scaling01/status/2064394893603049625), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Fable retains **1M context** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Anthropic introduced refusal/fallback mechanisms and SDK middleware [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428351029449214)
- Anthropic disclosed **silent interventions for frontier LLM development** affecting about **0.03% of traffic** [@Hangsiin](https://x.com/Hangsiin/status/2064397550434816088)
- Fable is temporarily included in subscriptions until **June 22**, then credit-based [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)

**Opinions / interpretations**

- “Anthropic won,” “Anthropic has a coding moat,” “Anthropic going for ASI” are commentary rather than verified fact [@scaling01](https://x.com/scaling01/status/2064401880323653799), [@scaling01](https://x.com/scaling01/status/2064399642603802676), [@scaling01](https://x.com/scaling01/status/2064410532824662047)
- Claims that the move is primarily for **IPO optics**, **anti-open-source positioning**, or specifically to slow **Meta/China/open labs** are plausible interpretations but not confirmed by Anthropic [@kimmonismus](https://x.com/kimmonismus/status/2064448699632402664), [@kylebrussell](https://x.com/kylebrussell/status/2064502244041511348), [@natolambert](https://x.com/natolambert/status/2064412173527556298)
- Claims that Anthropic is acting from sincere safety beliefs rather than cynical moat-building are also interpretive [@finbarrtimbers](https://x.com/finbarrtimbers/status/2064427031543341450)
- Subjective reports like “GPT-4 moment,” “big model smell,” “strictly dominates me as an engineer,” or “doesn’t seem much better to normal users” are experiential, not standardized evidence [@karinanguyen](https://x.com/karinanguyen/status/2064406015760601379), [@bcherny](https://x.com/bcherny/status/2064431111154053187), [@akbirkhan](https://x.com/akbirkhan/status/2064418425552928812), [@citrini](https://x.com/citrini/status/2064480613852201336)

## Different perspectives


**Supportive / capability-first**

- Anthropic staff and close testers described Fable 5 as a **step-function improvement**:
  - Felix Rieseberg: shift from giving AI tasks to giving it responsibilities [@felixrieseberg](https://x.com/felixrieseberg/status/2064392202504310900)
  - Alex Albert: model feels collaborative rather than tool-like [@alexalbert__](https://x.com/alexalbert__/status/2064394410004304003)
  - Karpathy: a “major-version-bump-deserving step change,” especially on long difficult tasks, though safeguards are “a little too trigger happy for launch” [@karpathy](https://x.com/karpathy/status/2064409694761054332)
  - Bcherny: biggest step since Opus 4.5; the model shows judgment, taste, methodical debugging [@bcherny](https://x.com/bcherny/status/2064431111154053187)
- Third-party infra and app vendors emphasized benchmark wins and integration value rather than the safety controversy [@cursor_ai](https://x.com/cursor_ai/status/2064394824313376787), [@cognition](https://x.com/cognition/status/2064398549073453266), [@NotionHQ](https://x.com/NotionHQ/status/2064397568696819984), [@Azure](https://x.com/Azure/status/2064421301108834552)

**Critical / trust and openness**

- Many researchers and open-model advocates argued the silent throttling is unacceptable even if safety-motivated:
  - Natolambert called doing it without telling users “misaligned” [@natolambert](https://x.com/natolambert/status/2064404993193754830)
  - Dean Ball warned it could attract **antitrust** scrutiny [@deanwball](https://x.com/deanwball/status/2064434861088395730)
  - Jeremy Howard called it “a very dark and very sad day” [@jeremyphoward](https://x.com/jeremyphoward/status/2064481719626154417)
  - Gneubig warned of a future where AI is provided only to a privileged few [@gneubig](https://x.com/gneubig/status/2064451352000975124)
  - Eric Zelikman framed it as silently sabotaging customers [@ericzelikman](https://x.com/ericzelikman/status/2064442174373314701)
- Open-source supporters used the launch as an argument for **sovereign/open models** [@nickfrosst](https://x.com/nickfrosst/status/2064396337404096809), [@NoahZiems](https://x.com/NoahZiems/status/2064464265189482570), [@ClementDelangue](https://x.com/ClementDelangue/status/2064513229099876663)

**Neutral / mixed**

- Some observers argued Anthropic probably **sincerely believes** these interventions are necessary for safety, even if the product design is poor [@finbarrtimbers](https://x.com/finbarrtimbers/status/2064427031543341450)
- Others said Anthropic does **not owe** anyone unrestricted frontier capability, but still saw this as straightforward business and market segmentation rather than altruism [@suchenzang](https://x.com/suchenzang/status/2064452548753559644)
- Karpathy’s view is mixed: model quality is exceptional, but launch safeguards are over-sensitive and should likely be tuned [@karpathy](https://x.com/karpathy/status/2064409694761054332)

## Research restrictions, privacy, and enterprise implications


**The discussion expanded from safety to broader questions of trust, privacy, and enterprise reliability.**

- The central enterprise issue was **predictability**: if a provider can silently degrade outputs based on inferred task category, users may no longer know whether failures come from the model, the prompt, or hidden intervention [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486), [@code_star](https://x.com/code_star/status/2064464447662707180)
- Some users worried this is effectively a **supply-chain risk** for important workflows, pushing companies toward open weights or in-house models [@NoahZiems](https://x.com/NoahZiems/status/2064464265189482570), [@deliprao](https://x.com/deliprao/status/2064485687374569897)
- There was also concern that account-level context or prior usage history might affect trigger behavior, as seen in biologists’ reports about normal vs incognito mode [@cremieuxrecueil](https://x.com/cremieuxrecueil/status/2064449457869984035)
- No tweet in the supplied set provided direct evidence that Anthropic was **training on user data** or violating stated data privacy terms; the privacy debate here was mostly about **behavioral profiling / silent policy enforcement** rather than classic training-data privacy
- For research users, the hidden intervention was framed as especially damaging because it undermines **reproducibility and scientific attribution** [@deanwball](https://x.com/deanwball/status/2064434861088395730), [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486)
- For enterprise buyers, the issue is not just whether the model is powerful, but whether it is a **stable and auditable dependency** for coding, medicine, science, finance, and infrastructure

## Context


**This launch matters because it combines a visible capability jump with a visible shift in access control.**

- The release landed amid intense competition with GPT-5.5, upcoming GPT-5.6, and Gemini 3.5 Pro; several posters argued Anthropic has opened a temporary lead in coding/agentic work [@kimmonismus](https://x.com/kimmonismus/status/2064467466450088078), [@teortaxesTex](https://x.com/teortaxesTex/status/2064473970892587105)
- It also lands in a broader argument about the **open vs closed model gap**; one linked Epoch-style framing said open-weight models lag closed frontier models by about **4 months on average** [@dl_weekly](https://x.com/dl_weekly/status/2064422551762153946)
- Community reaction suggests the launch may be remembered not only for “big model smell” and benchmark jumps, but for normalizing **selective capability release**: public access to the frontier model, but with **domain-specific hidden limits**
- That policy line is likely to influence future debates around:
  - **safety vs openness**
  - **fair access to frontier research tools**
  - **antitrust and platform power**
  - **enterprise trust in API providers**
  - **whether open models become the default for sensitive technical work even when they trail on raw capability**

**Models, benchmarks, and evals**


- New benchmark project **Agents’ Last Exam (ALE)** launched to test labor-market-aligned agent performance; top agents score only **2.6% on the hardest tier**, across **1,500+ tasks**, **55 occupations**, with contributions from **300+ experts across 100+ institutions** [@YiyouSun](https://x.com/YiyouSun/status/2064392466011394213), [@SnorkelAI](https://x.com/SnorkelAI/status/2064396025410760950), [@dawnsongtweets](https://x.com/dawnsongtweets/status/2064452279973863848)
- Cohere released **North Mini Code**, its first open-source coding model: **30B total / 3B active MoE**, **256K context**, **64K max generation**, Apache 2.0, optimized for agentic workflows [@cohere](https://x.com/cohere/status/2064378058329526556), [@JayAlammar](https://x.com/JayAlammar/status/2064385607455908254), [@vllm_project](https://x.com/vllm_project/status/2064416312605237434)
- Google announced **Gemini 3.5 Flash Live Translate**, real-time speech-to-speech translation in **70+ languages**, available in Gemini API, AI Studio, Google Translate, and coming to Meet [@OfficialLoganK](https://x.com/OfficialLoganK/status/2064369125447864674)
- New benchmark **iOSWorld** evaluates personally intelligent phone agents across **26 custom iOS apps** and **133 tasks**; strongest frontier model reaches only **52% success even with privileged access** [@rsalakhu](https://x.com/rsalakhu/status/2064402156740907444)

**Inference, training, and systems**


- **Latent Context Language Models (LCLMs)** were introduced as a long-context inference method compressing context up to **16×**, improving the latency/accuracy frontier over KV-cache compression [@micahgoldblum](https://x.com/micahgoldblum/status/2064361011994337772), [@iamleonli](https://x.com/iamleonli/status/2064374393057300846)
- Microsoft Research’s **Mirage** stores 3D scenes as latent tokens, reporting **10.57× faster** video generation and **55× lower memory use** [@HuggingPapers](https://x.com/HuggingPapers/status/2064393076416688416)
- vLLM introduced **vime**, an RL post-training framework in the vLLM ecosystem, positioned alongside NeMo-RL, OpenRLHF, and verl [@vllm_project](https://x.com/vllm_project/status/2064397637634376174)
- Discussion around agent training continued with **Self-Harness** for self-improving scaffolds [@omarsar0](https://x.com/omarsar0/status/2064429834999304247) and **AutoForge/interleaved thinking** retaining reasoning traces across turns [@cwolferesearch](https://x.com/cwolferesearch/status/2064505867181949182)
- Google/Hugging Face launched the **Fast Gemma Challenge** to speed up **Gemma 4 E4B** on a single **A10G** without wrecking quality [@googlegemma](https://x.com/googlegemma/status/2064374874962117084), [@osanseviero](https://x.com/osanseviero/status/2064375902046245219), [@_lewtun](https://x.com/_lewtun/status/2064386398090576236)

**Agents, tooling, and developer workflow**


- LangChain highlighted a pattern of **agent loops** driven by recurring triggers in Fleet [@caspar_br](https://x.com/caspar_br/status/2064363014997021126)
- OpenAI added **image results** to web search in the Responses API [@OpenAIDevs](https://x.com/OpenAIDevs/status/2064395155688616153)
- GitHub/Copilot app updates included **parallel sub-sessions** and a **canvas** UI for dynamic interfaces [@tgrall](https://x.com/tgrall/status/2064334802799509745), [@burkeholland](https://x.com/burkeholland/status/2064446521035067615)
- Hermes Desktop added **Ollama** support, with self-learning Python skills and messaging app integrations [@ollama](https://x.com/ollama/status/2064441778590339402), [@NousResearch](https://x.com/NousResearch/status/2064468385748951415)
- A security-oriented counterpoint on agent execution: **Temenos** argues for sandboxing generated code, not the agent, using **rootless gVisor** while keeping auth/tools on host [@abhijithneil](https://x.com/abhijithneil/status/2064462294155952297)

**Research, science, and formal methods**


- Axiom announced **EconLib**, a Lean-based economics library; formalizing Aumann’s “agreeing to disagree” theorem surfaced a hidden countability-related assumption [@TheTuringPost](https://x.com/TheTuringPost/status/2064391882017579520)
- “Economy of Minds” proposed agent coordination through auctions and incentives rather than centralized orchestration, reporting gains such as **15.9% → 57.0%** on math reasoning and **45.0% → 60.0%** on financial research [@TheTuringPost](https://x.com/TheTuringPost/status/2064406931184443618)
- Mayo Clinic’s **REDMOD** reportedly detected pancreatic cancer on CT scans up to **3 years before diagnosis**, identifying **73%** of hidden cancers at a median **475 days** before diagnosis [@TheRundownAI](https://x.com/TheRundownAI/status/2064416920191869191)

**Open ecosystem and infrastructure**


- Hugging Face and Arcee announced a partnership replacing AWS S3 with HF for all Arcee models/datasets, including private ones [@ClementDelangue](https://x.com/ClementDelangue/status/2064323874049679643), [@MarkMcQuade](https://x.com/MarkMcQuade/status/2064385389801124218)
- Cohere kept pushing the sovereign/open angle with “**Sovereign AI for all**” [@cohere](https://x.com/cohere/status/2064414912768618898)
- Marks Saroufim proposed a **Researcher Reciprocity License** and moved GPU MODE datasets to it, explicitly reacting to the sense that frontier labs benefit from open research while restricting access in return [@marksaroufim](https://x.com/marksaroufim/status/2064428421774753943), [@marksaroufim](https://x.com/marksaroufim/status/2064442386374369597)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Open Model Inference and Chat Template Updates

  - **[Xiaomi just claimed 1,000+ tps on a 1T model using a standard 8-GPU server](https://www.reddit.com/r/LocalLLaMA/comments/1u0buhm/xiaomi_just_claimed_1000_tps_on_a_1t_model_using/)** (Activity: 1027): ****Xiaomi MiMo** claims [`MiMo-V2.5-Pro-UltraSpeed`](https://mimo.xiaomi.com/blog/mimo-tilert-1000tps) reaches **`1000+ tokens/s` decoding on a `1T`-parameter MoE** using a single “standard” `8-GPU` server, via TileRT model-system co-design rather than Cerebras/Groq-style specialized hardware. The reported stack combines **MoE-expert-only FP4/MXFP4 quantization with QAT** while keeping non-expert modules at higher precision, plus **DFlash block-level masked speculative decoding** with acceptance lengths of `6.30` coding, `5.56` math/reasoning, and `4.29` agent tasks, and persistent low-latency kernels to reduce launch/sync overhead. A key unresolved technical caveat from comments is that Xiaomi does not specify *which* 8 GPUs were used, making reproducibility and cost/performance comparisons ambiguous.** Commenters debated the economics of “Token Winter,” arguing the bottleneck is less model demand than overpriced/hoarded Western GPU supply, while Chinese compressed sparse architecture/MoE work from **DeepSeek, Xiaomi, and MiniMax** is becoming more inference-efficient. Others highlighted Xiaomi’s selective FP4 strategy as the most important detail because naïve full-model FP4 degrades reasoning, code, and logic.

    - A key technical detail highlighted is that Xiaomi did **selective FP4 quantization** rather than applying FP4 uniformly: only the **MoE Experts** in **MiMo-V2.5-Pro** are quantized to FP4, while non-expert modules retain original precision to avoid degradation in reasoning, logic, and code generation. The comment notes Xiaomi used **FP4 QAT** to reduce model size and improve bandwidth utilization while keeping capability near the original model.
    - The released model weights are available on Hugging Face as **XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash**: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash. This is relevant because it allows independent inspection or benchmarking of the claimed `1,000+ tps` throughput on an 8-GPU server.
    - Several commenters questioned the hardware and parameter accounting behind the claim: *“8 GPU server… which 8 exactly?”* and *“1T-A1B?”* The technical concern is that throughput is not interpretable without knowing the exact GPU class, interconnect, serving stack, batch size, context length, and whether the `1T` MoE model activates only around `1B` parameters per token.

  - **[Gemma 4 Chat Template now has preserve thinking](https://www.reddit.com/r/LocalLLaMA/comments/1u084qi/gemma_4_chat_template_now_has_preserve_thinking/)** (Activity: 482): ****Google’s Gemma Team has added `preserve_thinking` support to the official Gemma 4 chat template**, matching an aftermarket template modification some users were already applying successfully. The change is framed as enabling better retention/use of model “thinking” traces in Gemma 4 chat formatting, though no benchmark numbers or implementation diff were provided in the thread.** Commenters generally welcomed the official adoption and argued it validates prior community template hacks. Several users speculated that a larger **Gemma 4 `124B` MoE** release would be needed to fully exploit the updated template for stronger agentic coding use cases.

    - Commenters note that **Gemma 4’s official chat template appears to be adding `preserve_thinking`**, a behavior some users had already enabled via aftermarket/custom template modifications and found effective. The main claimed technical benefit is improved continuity for **agentic coding workflows**, where retaining prior reasoning/thinking traces can help multi-step tool use and code iteration.
    - One commenter cautions that the change may not be live yet: the `preserve_thinking` support is described as an **open PR that has not been merged**, while the model files reportedly show no update for `21 days`. This suggests users should verify the tokenizer/chat-template files in the actual model repository before assuming the new behavior is available in released artifacts.
    - Several comments frame the template change as increasing demand for a larger **Gemma 4 `124B` MoE** variant, arguing that `preserve_thinking` would be more valuable when paired with a higher-capacity model for coding-agent use cases. The discussion is speculative, but technically centered on scaling the model size/MoE architecture to better exploit the updated chat-template behavior.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Fable 5/Mythos 5 Release and Access Tiers

  - **[Introducing Claude Fable 5](https://www.reddit.com/r/ClaudeCode/comments/1u1b207/introducing_claude_fable_5/)** (Activity: 2698): **The [image](https://i.redd.it/tb8akxef4a6h1.png) is a benchmark comparison table for the post’s claimed **Claude Fable 5 / Claude Mythos 5** release, showing the highlighted model leading or near-leading across agentic coding, knowledge work, spatial reasoning, tool use, legal, biology, cybersecurity, and health benchmarks versus Claude Mythos Preview, Claude Opus 4.8, GPT 5.5, and Gemini 3.1 Pro. The selftext frames Fable 5 and Mythos 5 as the same underlying “Mythos-class” model, with **Fable 5 using safety fallbacks**: cybersecurity, biology/chemistry, and distillation-related requests are routed to **Claude Opus 4.8**, reportedly affecting under `5%` of sessions.** Comments are mostly hype or skepticism rather than technical analysis, including jokes like “AGI confirmed” and a complaint asking whether “Fable [is] getting dumber recently.”

    - One commenter noted an apparent access/pricing constraint: **Claude Fable 5 is free only until `June 22`**, after which users will reportedly need to purchase credits to continue using it. This is relevant for anyone evaluating the model because benchmark or workflow testing may need to be completed before the credit-gated period begins.
    - A user reported a possible launch/frontend issue, asking whether **Fable generated malformed HTML** and linking a screenshot of the rendered result: https://preview.redd.it/qaceea1fma6h1.jpeg?width=1440&format=pjpg&auto=webp&s=440eb5a30e7dfc186d610ed94be50fa50b962c9e. The comment implies there may have been an implementation or output-formatting bug during the rollout rather than a model-quality discussion backed by benchmarks.

  - **[Claude Fable 5 feels less like a model launch and more like a preview of AI inequality](https://www.reddit.com/r/ClaudeAI/comments/1u1fsdi/claude_fable_5_feels_less_like_a_model_launch_and/)** (Activity: 2387): **The post argues that **Anthropic’s alleged Claude Fable 5 rollout** represents a shift from a uniform public frontier-model release to a **tiered access architecture**: public paid users receive Fable 5 with safety routing that may downgrade requests involving `cyber`, `bio`, `chemistry`, or `distillation` to Opus 4.8, while selected partners purportedly get **Mythos 5**, described as the same underlying model with fewer safeguards. It also highlights pricing/capacity constraints: Fable 5 is said to be included in paid plans only until `June 22`, then potentially moved to usage credits, implying frontier-agent inference remains too expensive for flat-rate consumer subscriptions.** Comments split between concern over AI access inequality and acceptance of restrictive safety policies as necessary for high-risk capabilities. One commenter frames the outcome as predictable token-economics pressure toward expensive enterprise-grade models, while another defends a *“rather safe than sorry”* approach despite user friction.

    - Several commenters framed the launch as an expected economics shift: as frontier models grow in capability and complexity, **inference/token costs rise enough that top-tier models become enterprise-only tools** rather than default consumer products. One commenter argued this will push everyday workloads toward cheaper local inference on hardware like **Apple M-series chips** or **RTX Spark-class accelerators**, reserving frontier APIs for high-value tasks.
    - A pricing-focused thread claimed that the new model’s API economics make consumer subscriptions structurally mismatched with frontier usage: *“Our `$200` monthly sub is like `3` API prompts with the new model.”* The implied technical point is that even high-end consumer plans may be viable only through heavy rate limits, model routing, or fallback to cheaper models such as **Opus 4.8**, which one commenter described as sufficient for “`99%`” of users.

  - **[Claude Fable (Mythos) is OUT!](https://www.reddit.com/r/singularity/comments/1u1at0h/claude_fable_mythos_is_out/)** (Activity: 1456): **The image ([PNG](https://i.redd.it/i88096c6fa6h1.png)) appears to show a **Claude-style model picker/UI** with a newly available model labeled **“Fable 5 High”**, matching the post title’s claim that *Claude Fable/Mythos is out*. The key contextual detail from comments is that **Fable 5 is temporarily included** for Pro, Max, Team, and seat-based Enterprise plans *“at no extra cost”* only through `June 22`, with removal from those plans on `June 23`; no benchmarks, architecture details, API specs, or capability evaluations are provided.** Commenters mostly reacted to the limited-time availability with token-budget anxiety and FOMO, joking about needing to “burn tokens” before access disappears. The discussion is largely non-technical and does not include substantive model-performance analysis.

    - Users highlighted that **Fable 5** is temporarily bundled into **Pro, Max, Team, and seat-based Enterprise** plans only through `June 22`, with removal on `June 23`, implying future availability may require a separate tier or paywall.
    - One availability detail: **Fable 5** appears to work in **Claude Code via command line**, but at least one user reported it is **not visible in the Claude desktop app’s Claude Code integration**, suggesting a client-specific rollout or UI/model-selector discrepancy.


### 2. Anthropic Data Policy and Safety Governance

  - **[Anthropic changed their privacy policy today and there's a specific clause that every Claude user needs to know about](https://www.reddit.com/r/ClaudeAI/comments/1u0kq84/anthropic_changed_their_privacy_policy_today_and/)** (Activity: 1475): **The post claims **Anthropic** published a revised [privacy policy](https://www.anthropic.com/legal/privacy) on `2026-06-08`, effective `2026-07-08`, changing law-enforcement disclosure language from externally compelled disclosure via legal process to disclosure based on Anthropic’s internal *“good faith belief”* that it is necessary. The author argues this creates risk for false-positive moderation/classifier escalations involving creative writing, roleplay, threats-in-fiction, or mental-health venting, and contrasts it with allegedly narrower disclosure language in [OpenAI](https://openai.com/policies/) and [Mistral](https://mistral.ai/terms/) policies; one commenter asks for the actual policy-update link/source.** Top commenters react negatively, framing the change as a major privacy regression and evidence of “enshittification,” with one saying it undermines Anthropic’s trust/morality branding and makes them consider switching back to Codex.

    - A commenter argues the revised Anthropic terms may be **GDPR-noncompliant** for European users because the relevant clause is allegedly too broadly defined. They recommend escalating via Anthropic’s **Data Protection Officer (DPO)** first, then filing with the relevant **Data Protection Commission/authority (DPC)** if unresolved.
    - A related legal-risk comparison was raised with **OpenAI**, citing a report that families of mass-shooting victims may seek **US$1 billion** in a lawsuit alleging the perpetrator’s ChatGPT usage had been internally flagged but not reported to police: [BIV article](https://www.biv.com/news/tumbler-ridge-families-likely-to-seek-us1-billion-in-lawsuit-against-openai-lawyer-12209582). The technical-policy implication discussed is whether AI providers’ internal safety monitoring creates obligations around escalation, reporting, and user privacy.


### 3. Frontier AI Infrastructure and Brain Research Bets

  - **[SpaceX has just revealed it's first AI satellite design](https://www.reddit.com/r/singularity/comments/1u0qc5r/spacex_has_just_revealed_its_first_ai_satellite/)** (Activity: 1883): **The image is a presentation-style concept slide for an alleged **SpaceX “AI1 satellite”** design, showing a large orbital compute platform with **deployable liquid radiators**, **centralized compute**, a roughly `70 m` wingspan, `20 m` deployed height, and a `150 kW` solar array feeding a `150 kW peak / 120 kW average` compute payload. The technical focus is on space-based AI compute power and thermal rejection, with the diagram highlighting redundant pumping loops and micrometeor shielding for radiator survivability: [image](https://i.redd.it/tw874bgnn56h1.jpeg).** Comments were highly skeptical, with one user comparing the compute capacity to “1 rack of GB200” and arguing it would not be cost-effective, especially because hardware failure could make the whole satellite a write-off. Another commenter framed the reveal as possible market/IPO hype rather than a practical engineering announcement.

    - A commenter identifies the pictured compute payload as roughly **one rack of NVIDIA GB200-class hardware**, arguing this would be hard to make cost-effective in orbit given launch/integration costs and the inability to service failed components: *“if something breaks, the entire thing is a writeoff.”*
    - Another technical criticism focuses on the architecture terms **“redundant pumping loops”** and **“centralized compute”**, implying concern that liquid-cooling loops plus a centralized accelerator rack could create difficult thermal-management and single-point-of-failure risks for a satellite AI platform.

  - **[Jeff Bezos Is Funding a Wild Hunt for the Brain’s ‘Core Algorithm’](https://www.reddit.com/r/singularity/comments/1u079tc/jeff_bezos_is_funding_a_wild_hunt_for_the_brains/)** (Activity: 1381): **[WIRED reports](https://www.wired.com/story/jeff-bezos-is-funding-a-wild-hunt-for-the-brains-core-algorithm/) that **Jeff Bezos** is funding **Flourish**, a neuroscience/AI startup reportedly valued at **`$2.5B`** with **`$500M`** raised, pursuing the hypothesis that direct study of biological neurons could uncover a brain “core algorithm.” The technical bet is that empirical neuroscience—rather than scaling today’s deep learning alone—may yield new AI architectures or learning rules inspired by real neural computation.**

    - A commenter raises a core technical objection to the premise: *“What makes him think the brain operates like an algorithm?”* The point challenges whether cognition can be reduced to a singular computational procedure, as opposed to emerging from heterogeneous biological, biochemical, and network-level dynamics.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
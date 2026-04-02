---
id: MjAyNS0x
title: The Claude Code Source Leak
date: '2026-03-24T05:44:39.731046Z'
description: >-
  **Anthropic's** closed-source coding product **Claude Code** experienced a
  significant source leak exposing over **500k lines** of orchestration logic,
  including autonomous modes and memory systems, but not model weights. The leak
  led to rapid public reverse-engineering, numerous forks with up to **32.6k
  stars and 44.3k forks**, and subsequent **DMCA takedowns** by Anthropic.
  Suspicious npm packages emerged targeting users compiling the leaked code,
  creating a live security hazard. Discussions also mention unreleased model
  references like **"mythos"** and ongoing product feature updates despite the
  leak. *"OFFICIAL STATEMENT from Anthropic regarding the leak"* was noted but
  not detailed.
companies:
  - anthropic
models:
  - claude-code
topics:
  - model-architecture
  - security
  - reverse-engineering
  - dmca
  - software-development
  - open-source
  - code-leak
  - agent-harness-design
people: []
---


**The accidental "open sourcing" of Claude Code brings a ton of insights.**

> AI News for 3/23/2026-3/24/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!


OpenAI’s Largest Fundraise in Human History closed today, growing by a few billion, but disclosing some cool numbers like $24B ARR (growing 4x faster than Google/Meta in their heyday), and also had a “soft IPO” with $3B of investment from rich people and inclusion in ETFs from ARK Invest, although ChatGPT WAU growth seem to has stalled out - they STILL have not crossed the 1B WAU mark targeted for end 2025. Codex also worryingly has not announced a new milestone for March.

By far the biggest news of the day is the Claude Code source leak, in itself not particularly damaging for Anthropic, but surely embarrassing and also somewhat educational - Christmas come early for Coding Agent nerds. You can read the many many tweets and posts covering the 500k LOC codebase, and you can browse multiple hosted forks of the source.

There are fun curiosities, such as the full verb list, or Capybara/Mythos v8, or the /buddy April Fools feature, or Boris’ confirmed WTF counter, or creating the cursed “Claude Codex”, or the dozen other unreleased features, but most serious players are commenting on a few things. Sebastian Raschka probably has a good list of the top 6:

Putting Repo state in Context (eg recent commits, git branch info)

Aggressive cache reuse

Custom Grep/Glob/LSP (standard in industry)

Claude code has less than 20 tools default on (up to 60+ total): AgentTool, BashTool, FileReadTool, FileEditTool, FileWriteTool, NotebookEditTool, WebFetchTool, WebSearchTool, TodoWriteTool, TaskStopTool, TaskOutputTool, AskUserQuestionTool, SkillTool, EnterPlanModeTool, ExitPlanModeV2Tool, SendMessageTool, BriefTool, ListMcpResourcesTool, and ReadMcpResourceTool.




more in ccunpacked
File read deduplication/tool result sampling

Structured Session Memory (more on this)

Subagents




Memory
Claude Code’s Memory has a 3 layer design with 1) a MEMORY.md that is just an index to other knowledge, 2) topic files loaded on demand, and 3) full session transcripts that can be searched. There’s also an “autoDream” mode for “sleep” - merging memories, deduping, pruning, removing contradictions.




A deeper analysis from mem0 finds 8 phases:




caption...
And there are 5 kinds of Compaction:




Subagents use Prompt Caching
A key feature of CC: they use the KV cache to create a fork-join model for their subagents, meaning they contain the full context and don’t have to repeat work. In other words: Parallelism is basically free.

The 5 level Permission System



The 2 Types of Plan mode
here:

Image


Resilience/Retry



Other Unreleased/Internal Features
Including an employee-only gate and an employee TUI, but also a bunch of other stuff in development including ULTRAPLAN and KAIROS:




note a few of these were recently shipped
And internal MAGIC DOCS:

Image

---

# AI Twitter Recap


**Top Story: Claude Code source leak — architecture discoveries, Anthropic’s response, and competitor reactions**

## What happened


A closed-source Anthropic coding product, Claude Code, appears to have had substantial source artifacts exposed via shipped source maps / package contents, which triggered rapid public reverse-engineering, mirroring, and derivative ports. The discussion quickly shifted from “embarrassing leak” to “what does this reveal about state-of-the-art agent harness design?” Multiple observers highlighted that the leak exposed orchestration logic rather than model weights, including autonomous modes, memory systems, planning/review flows, and model-specific control logic. Public forks proliferated; one post claimed **32.6k stars and 44.3k forks** on a fork before legal fear led to a Python conversion effort using Codex ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815)). Later commentary put the exposed code volume at **500k+ lines** ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682)). Anthropic then moved to contain redistribution via **DMCA takedowns** according to several posters ([dbreunig](https://x.com/dbreunig/status/2039007097376108979), [BlancheMinerva](https://x.com/BlancheMinerva/status/2039114452088295821)). The most concrete official signal in the dataset is a widely shared post noting an **“OFFICIAL STATEMENT from Anthropic regarding the leak”** ([theo](https://x.com/theo/status/2039074833334689987)), but the statement text itself is not included here, so only its existence can be treated as factual from this corpus. Separately, a Claude Code team member announced a product feature during the fallout — easier local/web GitHub credential setup via `/web-setup` ([catwu](https://x.com/_catwu/status/2039027712288075812)) — implying normal product operations continued. The leak also created a live security hazard: attackers quickly registered suspicious npm packages such as **`color-diff-napi`** and **`modifiers-napi`** to target people trying to compile the leaked code ([Butanium_](https://x.com/Butanium_/status/2039079715823128964)).

## Facts vs. opinions


**What is reasonably factual from the tweets:**
- Public access to Claude Code source artifacts occurred and was widely discussed as a leak ([scaling01](https://x.com/scaling01/status/2038982287648293016), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815), [theo](https://x.com/theo/status/2039069225109819838)).
- The exposed material did **not** include model weights; at least one security roundup explicitly says “They did not leak the model weights” ([saranormous](https://x.com/saranormous/status/2039172685666918672)).
- People extracted feature names and architecture motifs from the repo, including **Kairos**, **dream**, **teammem**, **buddy**, **ultrathink**, **ultraplan**, **ultrareview**, plus GitHub and Slack integrations ([scaling01](https://x.com/scaling01/status/2038982287648293016), [scaling01](https://x.com/scaling01/status/2039001738934468857)).
- Anthropic (or its representatives) appears to have sought takedowns of mirrored/forked copies via DMCA per multiple observers ([dbreunig](https://x.com/dbreunig/status/2039007097376108979), [BlancheMinerva](https://x.com/BlancheMinerva/status/2039114452088295821)).
- Suspicious package-name squatting targeted would-be builders of local Claude Code from leaked sources ([Butanium_](https://x.com/Butanium_/status/2039079715823128964)).
- Local compilation was reportedly achieved internally by others following the leak ([theo](https://x.com/theo/status/2039079267905261831)).

**Claims that are plausible but should be treated carefully:**
- That Anthropic “leaked” the repo by shipping source maps specifically: this is widely implied, but no authoritative technical root-cause explanation is quoted in the tweets.
- That unreleased model documents, including references to a model called **“mythos”**, were exposed: this appears in one roundup ([saranormous](https://x.com/saranormous/status/2039172685666918672)) and in speculative chatter like “Anthropic's new model Capybara/Mythos just wants to be human” ([scaling01](https://x.com/scaling01/status/2039091546377576864)), but the dataset does not independently verify artifact authenticity.
- The exact repo metrics and line counts (e.g. **32.6k stars / 44.3k forks**, **500k+ lines**) are third-party measurements and may reflect specific mirrors/forks at specific times rather than the original repository state.

**Opinions / interpretations:**
- The leak is embarrassing but “nothing groundbreaking” technically ([rasbt](https://x.com/rasbt/status/2039020306912755763)).
- The real moat is harness engineering, and with the code out, the gap between Claude Code and competitors will close faster ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682)).
- Anthropic should not aggressively suppress forks because the open-source community will build custom harnesses anyway ([BlancheMinerva](https://x.com/BlancheMinerva/status/2039128635559318013)).
- The event “fatally falsified” safety strategies based on secrecy and control ([pmarca](https://x.com/pmarca/status/2039042126294733295)).
- Copyright enforcement is being undermined if leaked code can simply be machine-translated to another language ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815)).

## Technical details revealed by the leak discourse


The most important technical takeaway is that observers overwhelmingly focused on **the harness**, not the underlying Claude model. That matches a broader trend in the same tweet set: “the harness matters” ([Vtrivedy10](https://x.com/Vtrivedy10/status/2038993396463796638)), and later “Beyond raw model capability, the real gap in coding tools is the harness” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682)). Sydney Runkle’s harness-engineering thread on dynamic config middleware — swapping model/tools/prompt per step, including tool registry filtering — is not about Claude specifically but provides strong context for what readers inferred the Claude Code team had built internally ([sydneyrunkle](https://x.com/sydneyrunkle/status/2039040565749096607)).

### Named internal systems / motifs surfaced by readers

Posts extracting features from the exposed repo mentioned:

- **Kairos**: described as an “always-on autonomous agent mode” ([scaling01](https://x.com/scaling01/status/2038982287648293016)).
- **dream**: described as “nightly memory consolidation” ([scaling01](https://x.com/scaling01/status/2038982287648293016)).
- **teammem**: “shared project memory” ([scaling01](https://x.com/scaling01/status/2038982287648293016)).
- **buddy**: “tamagotchi-like pet system with models” ([scaling01](https://x.com/scaling01/status/2038982287648293016)); later echoed by others noticing “There’s an AI pet lurking in Claude Code!” ([dbreunig](https://x.com/dbreunig/status/2039017351061143780)) and “new claude code buddy feature is kinda cute” ([eliebakouch](https://x.com/eliebakouch/status/2039176958416720104)).
- **automatic skill improvement** ([scaling01](https://x.com/scaling01/status/2038983513081356360)).
- **ultrathink**, **ultraplan**, **ultrareview** and “complete integration with GitHub and Slack” ([scaling01](https://x.com/scaling01/status/2039001738934468857)).

Even if some names were partly promotional or whimsical, the aggregate picture is consistent: Claude Code appears to have a layered agent runtime with:
1. persistent/project memory,
2. autonomous/background operation,
3. planning/review stages,
4. self-improvement or skill distillation loops,
5. collaboration hooks into developer workflow systems.

### Harness shape and code composition

Several technical readers converged on a similar interpretation:

- **A lot of the value is hard-won orchestration logic and diagnostics**, not magical algorithms ([dbreunig](https://x.com/dbreunig/status/2039206774558036466)).
- The code contains **many model- and context-specific conditionals** to smooth over model quirks ([dbreunig](https://x.com/dbreunig/status/2039206774558036466)).
- There is also **a lot of ordinary CLI plumbing / boilerplate**, suggesting the proprietary edge is not in the shell app per se but in the feedback loops, prompts, middleware, diagnostics, and integrations ([dbreunig](https://x.com/dbreunig/status/2039206774558036466)).
- A significant fraction is likely **scaffolding around planning, tool calls, review, memory, retries, and telemetry** rather than novel model code.

That reading dovetails with broader agent-engineering discussion in the dataset:
- LangChain promoted human-in-the-loop interrupts as standard stream state rather than bespoke workflow mechanics ([LangChain_JS](https://x.com/LangChain_JS/status/2038985561348993107)).
- Vtrivedy emphasized evals as the signal that grounds agent updates and harness optimization ([Vtrivedy10](https://x.com/Vtrivedy10/status/2039029715533455860)).
- Koylan summarized a Shopify/DSPy architecture: agent-controlled retrieval, context isolation, MIPRO prompt optimization after modularization, and “smaller model + better architecture > bigger model + worse architecture” ([koylanai](https://x.com/koylanai/status/2039027239304433767)).

The implication: the Claude Code leak mostly confirmed industry suspicion that **production coding agents are ensembles of prompts, policies, middleware, memory, evaluation, and exception handling**.

### Packaging and leak mechanism clues

The tweets imply the leak may have originated from shipped source artifacts:
- “closed source > ship sourcemaps > source leaks instantly” ([mattrickard](https://x.com/mattrickard/status/2039054181361967487)).
- Theo discussed whether he could “open the code directory live” without copyright strikes, implying broad local inspection had become feasible ([theo](https://x.com/theo/status/2039069225109819838)).
- “Local Claude Code builds have been achieved internally” suggests enough of the tree was present to compile or reconstruct local builds ([theo](https://x.com/theo/status/2039079267905261831)).

This also produced a derivative security risk: package-name squatting for native addon dependencies targeting local builders ([Butanium_](https://x.com/Butanium_/status/2039079715823128964)). That is a classic second-order leak effect: once code escapes, the exploit surface expands from “what was exposed?” to “what toolchain behaviors does panic-triggered community recompilation create?”

## Anthropic’s apparent response


Within this tweet set, Anthropic’s response is visible mostly indirectly.

### 1) Official statement exists
Theo posted that there was an **“OFFICIAL STATEMENT from Anthropic regarding the leak”** ([theo](https://x.com/theo/status/2039074833334689987)). Since the statement text is absent, anything beyond its existence would be speculation.

### 2) Legal containment via DMCA
Multiple posts say Anthropic was sending **DMCA takedowns** against repos redistributing the leaked source:
- “Code is free, but Anthropic is shutting down repos of the leaked Claude Code source with DMCA requests” ([dbreunig](https://x.com/dbreunig/status/2039007097376108979)).
- “DMCAs for Claude code source code are going out” ([BlancheMinerva](https://x.com/BlancheMinerva/status/2039114452088295821)).

This suggests Anthropic treated the event as unauthorized publication of proprietary code, not as an open-sourcing moment.

### 3) Product operations continued
A Claude Code team member posted a normal product update in the middle of the controversy: `/web-setup` to reuse local GitHub credentials in web Claude sessions ([catwu](https://x.com/_catwu/status/2039027712288075812)). That’s weak evidence but consistent with “contain the leak; keep shipping.”

### 4) No evidence here of Anthropic embracing the leak
Some outsiders argued Anthropic should “be chill” because the code is already everywhere ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518)), but the evidence in this dataset points the other way: **containment and takedown**, not formal release.

## Competitor and ecosystem responses


### OpenHands / open-source competitors
The clearest competitor response came from OpenHands’ Graham Neubig:
- “OpenHands will not be issuing any DMCA takedown notices for those who want to use our agent, which has most of the features of Claude Code. We have Tamagotchi on the roadmap” ([gneubig](https://x.com/gneubig/status/2039166255089799222)).
- He followed with a tracking issue for the tamagotchi feature ([gneubig](https://x.com/gneubig/status/2039168326912389208)).

This is both competitive positioning and a substantive claim: an open agent stack can replicate “most” Claude Code features, with playful acknowledgment of the buddy system.

### OpenAI / Codex comparisons
The same time window also saw confusion over an alleged “Codex codebase leak,” later corrected by an OpenAI employee:
- Initial viral claim: “somebody at OpenAI leaked the entire codex codebase” ([reach_vb](https://x.com/reach_vb/status/2038971515572523502)).
- Correction: “the repo has been open source since its inception... I work on codex at openai” ([reach_vb](https://x.com/reach_vb/status/2039038251407732754)).

This is useful context because it sharpened the contrast:
- **Codex repo visibility was intentional.**
- **Claude Code visibility was not.**

Yuchen framed one downstream effect starkly: a fork of Claude Code got huge adoption, then “convert the whole codebase from TypeScript to Python with Codex” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815)). That is an opinionated but important competitive angle: open or leaked harness code can be rapidly re-expressed across language ecosystems using rival coding agents.

### Nous / Hermes / persistent-agent competitors
Nous/Hermes posts were not direct reactions to the leak but became part of the comparison set because they pitch similar capabilities:
- Persistent memory, self-improvement, many built-in tools, multi-platform integration, MIT license ([evanlong_me](https://x.com/evanlong_me/status/2039026061640601816)).
- Import from OpenClaw in two minutes ([AntoineRSX](https://x.com/AntoineRSX/status/2039017227270156395)).
- Cron-based vuln scanning and agent upkeep ([Teknium](https://x.com/Teknium/status/2039022907020689898), [Teknium](https://x.com/Teknium/status/2039096442313396514)).
- Community tools and guides to get started ([Teknium](https://x.com/Teknium/status/2039102514508058675), [aijoey](https://x.com/aijoey/status/2039108098174906514)).

These matter because leak readers often concluded Claude Code’s “secret sauce” is reproducible by strong open agent systems.

### Venture/open-source ideology response
Marc Andreessen’s broad reaction was the most philosophical: “The idea that ‘AI safety’ could be based on secrecy and control has been fatally falsified” ([pmarca](https://x.com/pmarca/status/2039042126294733295)). That is clearly opinion, but it captures one faction’s conclusion: proprietary app-layer secrecy is not a durable control regime.

## Different opinions


### View 1: The leak is strategically important because it exposes the real moat
This was the dominant engineer take.
- “Beyond raw model capability, the real gap in coding tools is the harness” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682)).
- “Harness engineering is hard and deeply non-trivial” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518)).
- “So many conditionals based on model types and specific contexts” ([dbreunig](https://x.com/dbreunig/status/2039206774558036466)).
- “the harness shapes [models] to be good and cost efficient for work we care about” ([Vtrivedy10](https://x.com/Vtrivedy10/status/2038993396463796638)).

This perspective says the leak reduced information asymmetry around the most valuable part of commercial coding agents.

### View 2: Interesting, but not groundbreaking
- Rasbt: “Other than the fact the leak is embarrassing, it’s interesting but nothing groundbreaking” ([rasbt](https://x.com/rasbt/status/2039020306912755763)).
- Mbusigin: “would have been a lot more interesting six months ago... harnesses are a dime a dozen now” ([mbusigin](https://x.com/mbusigin/status/2039105055299686834)).

This camp thinks the field had already converged on many of these patterns, so the leak mostly validated known best practices.

### View 3: Anthropic should stop fighting and lean into reality
- Blanche Minerva argued that once the community is already building custom harnesses, takedowns achieve little ([BlancheMinerva](https://x.com/BlancheMinerva/status/2039128635559318013)).
- Yuchen said the team was being “chill,” though the evidence in the dataset for that is mixed given DMCA reports ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518)).

This view sees legal enforcement as low-leverage after code escape.

### View 4: DMCA is justified because this is still proprietary code
That perspective is implicit in Anthropic’s apparent actions and in posts worrying about copyright strikes ([theo](https://x.com/theo/status/2039069225109819838)). It’s less argued explicitly here, but the logic is straightforward: accidental publication does not waive copyright.

### View 5: The leak demonstrates secrecy-based safety/control is broken
- Andreessen’s argument generalizes beyond Anthropic ([pmarca](https://x.com/pmarca/status/2039042126294733295)).

This is ideological and broader than the engineering specifics, but it became part of the discourse.

## Context: why this matters


### 1) It reveals where coding-agent performance actually comes from
The leak surfaced concrete evidence for a shift many practitioners already suspected: **frontier coding UX is increasingly a systems problem, not just a model problem**. The model provides reasoning and generation, but production quality comes from:
- dynamic tool selection,
- memory architecture,
- evaluation/review loops,
- error taxonomy and retries,
- model-specific prompt branching,
- integration with GitHub/Slack/etc.,
- and persistent autonomy modes.

That matches the surrounding discourse on agent evaluation and improvement:
- traces as the foundation for improvement loops ([LangChain](https://x.com/LangChain/status/2039028327030079565)),
- online evals and trace enrichment ([Vtrivedy10](https://x.com/Vtrivedy10/status/2039186184161616245)),
- agent monitoring in production ([LangChain](https://x.com/LangChain/status/2039014039892947062)).

### 2) It compresses the competitive cycle
If Claude Code encoded a large amount of tacit product knowledge, then public access means competitors can:
- copy patterns,
- benchmark harness decisions,
- port designs cross-language,
- identify weak points,
- and build open equivalents faster.

Yuchen explicitly predicted that “every model lab and AI coding startup... will study it and close that gap fast” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682)).

### 3) It creates a new security lesson
The package-squatting follow-on attack matters almost as much as the leak itself. Once developers rush to compile leaked internal software, the ecosystem becomes vulnerable to dependency confusion, typo squats, fake native modules, and malicious setup scripts ([Butanium_](https://x.com/Butanium_/status/2039079715823128964)). That fits the week’s broader supply-chain panic summarized by Saranormous ([saranormous](https://x.com/saranormous/status/2039172685666918672), [saranormous](https://x.com/saranormous/status/2039108234460721341)).

### 4) It undermines simplistic “wrapper” dismissals
One important subtext: the leak seems to have convinced many engineers that the “wrapper” layer is not trivial. Multiple readers came away saying the code proves **wrapper/harness engineering is hard** ([dbreunig](https://x.com/dbreunig/status/2039206774558036466), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518)). That strengthens the case for application-layer moats built on orchestration, product UX, and eval loops rather than only on foundation models.

## Bottom line


The Claude Code leak did not expose Anthropic’s model weights, but it exposed something strategically important: a large chunk of the **agent harness stack** behind a leading coding product. The public findings point to a mature orchestration architecture with persistent memory, autonomous/background modes, planning-review loops, skill improvement, and deep workflow integrations. Anthropic’s observable response in this dataset was containment — official acknowledgment plus reported DMCAs — while competitors and open-source projects used the moment to argue that many of these features are now reproducible in open systems. The strongest technical conclusion from the community is not that Claude Code contained magic, but that **high-performance coding agents depend on lots of accumulated, model-specific, operationally messy systems engineering**. The leak therefore matters less as scandal than as a field note on where the real engineering leverage currently sits.

**Key tweets:** [@scaling01](https://x.com/scaling01/status/2038982287648293016), [@scaling01](https://x.com/scaling01/status/2039001738934468857), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518), [@dbreunig](https://x.com/dbreunig/status/2039007097376108979), [@dbreunig](https://x.com/dbreunig/status/2039206774558036466), [@theo](https://x.com/theo/status/2039074833334689987), [@theo](https://x.com/theo/status/2039079267905261831), [@Butanium_](https://x.com/Butanium_/status/2039079715823128964), [@gneubig](https://x.com/gneubig/status/2039166255089799222), [@pmarca](https://x.com/pmarca/status/2039042126294733295), [@rasbt](https://x.com/rasbt/status/2039020306912755763), [@BlancheMinerva](https://x.com/BlancheMinerva/status/2039128635559318013), [@mattrickard](https://x.com/mattrickard/status/2039054181361967487), [@saranormous](https://x.com/saranormous/status/2039172685666918672)

**Models, agents, and post-training**


- [@PrismML](https://x.com/PrismML/status/2039049400190939426) launched **Bonsai 8B/4B/1.7B**, a **1-bit weight** family under **Apache 2.0**. Claimed stats: **1.15 GB** for 8B, **14x smaller**, **8x faster**, **5x more energy efficient** than full precision peers; positioned as “10x intelligence density.” Follow-up posts showed an MLX/iPhone path and a left-shifted size-vs-intelligence Pareto frontier ([PrismML](https://x.com/PrismML/status/2039049404209148007), [PrismML](https://x.com/PrismML/status/2039049405815529559), [adrgrondin](https://x.com/adrgrondin/status/2039066539022778613), [HessianFree](https://x.com/HessianFree/status/2039049800398655730)).  
- [@nisten](https://x.com/nisten/status/2039100896840134935) provided a useful independent teardown of Bonsai-8B’s GGUF: **8,188,548,848 params**, **399 tensors**, **1099.3MB** total weight data, **1.126 bits/weight**, requiring a Prism fork of llama.cpp with `Q1_0_g128` support.  
- [@liquidai](https://x.com/liquidai/status/2039029358224871605) released **LFM2.5-350M**, a sub-**500MB quantized** model focused on **tool use and data extraction** in constrained environments. This drew attention partly because a 350M model reportedly used **28T tokens** ([abacaj](https://x.com/abacaj/status/2039158882111521190)).  
- [@hcompany_ai](https://x.com/hcompany_ai/status/2039021096649805937) launched **Holo3** computer-use models, claiming **78.9% on OSWorld-Verified**, ahead of **GPT-5.4** and **Opus 4.6** at **1/10th the cost**, with weights on Hugging Face and API live.  
- [@outsource_](https://x.com/outsource_/status/2038999111039357302) highlighted a **27B Qwen3.5 variant** distilled on Claude 4.6 Opus traces, claiming local **16GB VRAM** deployment, **96.91% HumanEval retention**, **24% chain-of-thought reduction**, and SWE-bench strength.  
- [@ClementDelangue](https://x.com/ClementDelangue/status/2039121367656702102), [@QGallouedec](https://x.com/QGallouedec/status/2039000031949165005), and [@lvwerra](https://x.com/lvwerra/status/2039003207985197107) marked **TRL v1.0**, with **75+ methods** spanning SFT, DPO, GRPO, async RL; lvwerra says it now sees **100k daily downloads**.  
- [@tinkerapi](https://x.com/tinkerapi/status/2039049192451301761) pointed to a training explainer that achieved a **5x score improvement on a 20B model** via careful SFT→RL choices.  
- [@togethercompute](https://x.com/togethercompute/status/2039099845856903644) released **Aurora**, an open-source RL-based speculative decoding system claiming **1.25x faster** than a well-trained static speculator and that **online training from scratch** can beat pretrained static baselines ([details](https://x.com/togethercompute/status/2039099852924367186), [code](https://x.com/togethercompute/status/2039099854702669835)).  
- [@QinYi88814](https://x.com/QinYi88814/status/2038971910835560921) flagged **daVinci-LLM**, a transparent pretraining effort with open weights, data pipeline, training process, and ablations; headline claim: **3B model matching 7B performance**.

**Agents, harnesses, evals, and observability**


- [@dair_ai](https://x.com/dair_ai/status/2038968068706390117) introduced **Natural-Language Agent Harnesses (NLAHs)** and an **Intelligent Harness Runtime**, arguing harness logic should itself be an editable/executable artifact rather than scattered controller code. This was one of the most technically aligned papers with the Claude Code discussion.  
- [@Vtrivedy10](https://x.com/Vtrivedy10/status/2038993396463796638), [@Vtrivedy10](https://x.com/Vtrivedy10/status/2039029715533455860), and [@Vtrivedy10](https://x.com/Vtrivedy10/status/2039035899938267334) made the case that harness quality is driven by eval quality, traces, and infra loops, not just model swaps.  
- [@sydneyrunkle](https://x.com/sydneyrunkle/status/2039040565749096607) continued a useful harness engineering series on **dynamic config middleware** for per-step adaptation of tools/models/prompts.  
- [@LangChain_JS](https://x.com/LangChain_JS/status/2038985561348993107) described a practical human-in-the-loop pattern where interrupts appear as ordinary stream state; [@LangChain](https://x.com/LangChain/status/2039014039892947062) launched a course on monitoring production agents; [@LangChain](https://x.com/LangChain/status/2039028327030079565) framed traces as the base primitive of the improvement loop.  
- [@FranklinMatija](https://x.com/FranklinMatija/status/2039001719007330530) introduced **AI Agent Traps**, a taxonomy of six adversarial classes against autonomous agents interacting with web pages, email, APIs, and multi-agent systems.  
- [@perplexity_ai](https://x.com/perplexity_ai/status/2039029140758864314) launched the **Secure Intelligence Institute**, led by Ninghui Li, with a first paper responding to NIST on securing autonomous agents ([paper](https://x.com/perplexity_ai/status/2039029152880480260)).  
- [@cwolferesearch](https://x.com/cwolferesearch/status/2039009111711367557) published a survey of **30+ LLM evals/benchmarks**, emphasizing domain taxonomies, human annotation, model-in-the-loop curation, data quality, realism, and evolution. This is one of the more useful meta-eval posts in the batch.  
- [@GoogleResearch](https://x.com/GoogleResearch/status/2039014600927043926) announced a new framework for better reproducibility of subjective AI benchmarks by optimizing the ratio of items to human raters per item.  
- [@koylanai](https://x.com/koylanai/status/2039027239304433767) summarized a DSPy/Shopify-style architecture lesson set: agent-controlled retrieval, context isolation, prompt optimization after modularization, frozen eval contexts, and “smaller model + better architecture > bigger model + worse architecture.”

**Open models, multimodal, and systems**


- [@IBM / @mervenoyann](https://x.com/mervenoyann/status/2039015519135641997) highlighted **Granite 4.0-3B-Vision**, positioned as strong on docs/tables/charts for its size, available via transformers/vLLM under a free license.  
- [@LearnOpenCV](https://x.com/LearnOpenCV/status/2038972079370858750) covered **Molmo Point**, focused on precise visual grounding; [@_akhaliq](https://x.com/_akhaliq/status/2038998550881714402) flagged **TAPS** for task-aware speculative sampling; [@_akhaliq](https://x.com/_akhaliq/status/2039000804061847801), [@_akhaliq](https://x.com/_akhaliq/status/2039006585188499744), [@_akhaliq](https://x.com/_akhaliq/status/2039007111741366620), [@_akhaliq](https://x.com/_akhaliq/status/2039011853460819999), and [@_akhaliq](https://x.com/_akhaliq/status/2039029830323253546) surfaced new papers on image generation, agent civilization infra, image editing, on-device image generation/editing, and bimanual motion generation.  
- [@dair_ai](https://x.com/dair_ai/status/2039072251199549573) posted **GAAMA**, a graph-augmented associative memory for agents, reporting **78.9% mean reward on LoCoMo-10** and outperforming tuned RAG baselines.  
- [@quentinlldc](https://x.com/quentinlldc/status/2038986438088257558) released **LeWorldModel** datasets/checkpoints.  
- [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2039046172799578122) gave a dense review of **LeWorldModel**, including specifics: **224x224 RGB**, unmodified **ViT-Tiny** encoder, **192-d latent**, predictor as **ViT-S**, better performance with **dropout 0.1**, **batch 128 x 4 trajectories**, **300** action rollouts to horizon **H=5**, up to **30 CEM iterations**, and performance degradation at larger predictor sizes.  
- [@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2039102080959566038) published a Blackwell deep dive covering tensor cores, PTX/SASS, tcgen05, UMMA, TMA, floorsweeps, DSMEM, and yield microbenchmarking.  
- [@clattner_llvm](https://x.com/clattner_llvm/status/2039027422310596881) argued kernel authors need scheduler control without full micromanagement; a follow-up notes that simplifying race conditions opens more portable, composable algorithms ([thread](https://x.com/clattner_llvm/status/2039028017843126406)).  
- [@Prince_Canuma](https://x.com/Prince_Canuma/status/2039090156188389500) noted **RF-DETR** now on MLX for real-time on-device instance segmentation.  
- [@Shawkat_m1](https://x.com/Shawkat_m1/status/2039014724071719405) reported **2.2x** speedup after switching Ollama to MLX for Qwen3.5:36b; [@joreilly](https://x.com/joreilly/status/2039002786130534618) saw **38% faster** agent runs with `qwen3.5:4b-nvfp4` vs qwen3.5:4b on M1 Max.

**Industry, funding, and product moves**


- [@OpenAI](https://x.com/OpenAI/status/2039085161971896807) announced a huge financing: **$122B committed capital** at **$852B post-money valuation**, framed around distributing useful intelligence globally. This was amplified by multiple commentary posts ([scaling01](https://x.com/scaling01/status/2039081471843930366), [TheRundownAI](https://x.com/TheRundownAI/status/2039103606327001435), [reach_vb](https://x.com/reach_vb/status/2039114329967013980)).  
- [@runwayml](https://x.com/runwayml/status/2038984561132990836) launched the **Runway Fund**, saying it has already backed Cartesia, LanceDB, and Tamarind Bio.  
- [@charlieholtz](https://x.com/charlieholtz/status/2039027121901957349) said Conductor raised a **$22M Series A**.  
- [@andreamichi](https://x.com/andreamichi/status/2039010131443437850) said depthfirst raised an **$80M Series B** at a **$580M valuation** for AI security.  
- [@wandb](https://x.com/wandb/status/2038984035301822784) promoted an interview with ClickHouse CEO on raising **$50M** pre-product and building for AI agents.  
- [@yupp.ai](https://x.com/pankaj/status/2039010092255969712) is winding down, leaving the site up **15 days** for data export.  
- [@Google](https://x.com/Google/status/2038969843701989773) introduced Gmail username changes for U.S. users: any available `@gmail.com` username, old address retained as alias, **once per year up to three total changes**; [@gmail](https://x.com/gmail/status/2039107985281008078) launched **AI Inbox** beta for U.S. Google AI Ultra subscribers.  
- [@OfficialLoganK](https://x.com/OfficialLoganK/status/2039015034286694618) and [@_philschmid](https://x.com/_philschmid/status/2039014102811427263) rolled out **Veo 3.1 Lite** in Gemini API/AI Studio at **$0.05/sec**, half the price of Fast, supporting T2V/I2V in **4s/6s/8s** clips and **16:9 / 9:16**.  
- [@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2039055128276148454) introduced a music playground around **Lyria 3**.  
- [@osanseviero](https://x.com/osanseviero/status/2039120000095547722) reported **Gemma** reaching **400M downloads** and **100,000 variants**.  
- [@AnthropicAI](https://x.com/AnthropicAI/status/2039137425214353555) announced an MOU with the Australian government on AI safety research.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Claude Code Source Leak and Analysis

  - **[Claude code source code has been leaked via a map file in their npm registry](https://www.reddit.com/r/LocalLLaMA/comments/1s8ijfb/claude_code_source_code_has_been_leaked_via_a_map/)** (Activity: 4694): **The image reveals a directory listing from a terminal window, showing files related to a project named "Claude," which includes TypeScript files and a source map file (`cli.js.map`). The presence of this map file in the npm registry suggests that the source code could be unintentionally exposed, potentially due to a misconfiguration or oversight. This incident highlights the importance of securing source maps in production environments to prevent unauthorized access to source code.** Commenters humorously speculate about the oversight, suggesting it might be due to an Anthropic employee's mistake or a feature of their AI system. There's also a sarcastic remark about the code being 'open source' now due to the leak.

    - The leak of Claude's source code via a map file in their npm registry raises significant security concerns, particularly given Claude's reputation for identifying vulnerabilities. This incident highlights potential gaps in Anthropic's internal security measures and the effectiveness of their AI in safeguarding proprietary information.
    - The discussion humorously suggests that Anthropic employees might have inadvertently contributed to the leak through 'vibe coding,' implying a lack of stringent oversight or automated checks in their development process. This points to a need for more robust internal controls and possibly more advanced AI-driven monitoring systems to prevent such leaks.
    - The incident has sparked debate over whether the leaked code could be considered 'open source,' given its unintended public availability. This raises questions about the legal and ethical implications of using or analyzing the leaked code, and whether it could be leveraged to improve security practices or AI development.

  - **[Claude Code's source just leaked — I extracted its multi-agent orchestration system into an open-source framework that works with any LLM](https://www.reddit.com/r/LocalLLaMA/comments/1s8xj2e/claude_codes_source_just_leaked_i_extracted_its/)** (Activity: 600): **The source code for **Claude Code** was leaked, revealing its multi-agent orchestration system. A developer has re-implemented this system as an open-source framework called **open-multi-agent**, which is model-agnostic and compatible with both Claude and OpenAI models. The framework includes features such as a coordinator pattern for task decomposition, a team system with a message bus for inter-agent communication, and a task scheduler with dependency resolution. It is implemented in `TypeScript`, spans approximately `8000 lines`, and is licensed under **MIT**. The framework is designed to run in-process, unlike the `claude-agent-sdk`, and can be deployed in various environments such as serverless, Docker, and CI/CD. The project is available on [GitHub](https://github.com/JackChen-me/open-multi-agent).** Commenters express skepticism about the legality and ethics of open-sourcing a framework based on leaked proprietary code, with concerns about potential legal repercussions. There is also a debate on the practicality of using different models for planning and implementation, questioning the choice of models like GPT-4o for coding.

    - The discussion highlights the technical aspect of the multi-agent orchestration system extracted from Claude Code's source. The system is designed to break down goals into tasks, which is a critical feature for managing complex operations across different language models. This orchestration layer is pivotal for integrating various LLMs, such as using Claude for planning and GPT-4o for implementation, showcasing a sophisticated approach to leveraging the strengths of different models in tandem.
    - A technical debate arises around the use of GPT-4o for coding tasks in March 2026, suggesting skepticism about its suitability or performance for such tasks at that time. This implies a discussion on the evolution and capabilities of language models over time, and how certain models may become outdated or less effective for specific applications as newer models emerge.
    - The legal implications of open-sourcing proprietary code are discussed, particularly the risks associated with releasing leaked code under an open-source license like MIT. This raises concerns about copyright infringement and the potential need for legal protection, emphasizing the importance of understanding intellectual property laws when dealing with proprietary software.

  - **[Analyzing Claude Code Source Code. Write "WTF" and Anthropic knows.](https://www.reddit.com/r/LocalLLaMA/comments/1s8uerc/analyzing_claude_code_source_code_write_wtf_and/)** (Activity: 601): **The Reddit post discusses the source code of **Claude Code**, revealing extensive tracking and classification mechanisms. The system uses simple keyword detection for sentiment analysis, tracking words like `wtf` and `frustrating` to flag negative sentiment. It also monitors user behavior during permission prompts, logging actions such as opening feedback boxes or typing and canceling inputs. The feedback system is designed to capture negative experiences, prompting users to share session transcripts. Hidden commands like `ultrathink` and `ultraplan` alter system behavior, while telemetry logs detailed environment profiles, including session IDs and runtime details. An internal mode (`USER_TYPE=ant`) collects even more granular data, tying behavior to specific deployment environments. This level of instrumentation suggests a highly observable system beyond typical chatbot functionality. [Source](https://x.com/UsmanReads/status/2039036207431344140?s=20).** Some commenters argue that the described tracking mechanisms are standard for event-triggered analytics and user feedback systems, often used to identify issues with updates. Others note that features like `/btw` are now exposed and that commands like `ultrathink` are more like internal artifacts or easter eggs, reflecting a playful development culture.

    - NandaVegg highlights that the use of keyword lists for sentiment analysis, such as detecting words like 'wtf' or 'frustrating', is a common practice in event-triggered analytics systems. These systems are often employed in web-based applications to monitor user feedback and identify issues with updates that might disrupt user experience or model behavior. This approach helps developers quickly address potential problems by flagging negative sentiment as a trigger for further investigation.
    - NandaVegg also mentions the presence of internal features like 'ultraplan' and 'ultrathink' in Claude Code, which are not fully refined and serve as easter eggs. These features are likened to internal artifacts found in game apps, suggesting a culture of experimentation and side projects within the development team. The comment implies that such features might be part of an internal incentive system encouraging developers to innovate and add unique functionalities.
    - The discussion touches on the concept of 'tamagotchi mode', which SRavingmad expresses interest in. Although not detailed in the comments, this mode likely refers to a feature or internal project within Claude Code that mimics the interactive and nurturing aspects of a Tamagotchi, possibly as a playful or experimental feature within the AI system.


### 2. Qwen Model Releases and Benchmarks

  - **[Copaw-9B (Qwen3.5 9b, alibaba official agentic finetune) is out](https://www.reddit.com/r/LocalLLaMA/comments/1s8nikv/copaw9b_qwen35_9b_alibaba_official_agentic/)** (Activity: 330): **The image is a bar chart that compares the performance of three AI models: **CoPaw-Flash-9B**, **Qwen3.5-Plus**, and **GPT-5.4** across four tasks: Document Parsing, Scheduled Automation, Memory Management, and Information Search. **CoPaw-Flash-9B**, a model fine-tuned by **Alibaba**, shows competitive performance, particularly excelling in Scheduled Automation and Memory Management. This model is noted to be on par with **Qwen3.5-Plus** on some benchmarks, indicating its effectiveness in specific tasks. The release of **CoPaw-Flash-9B** is significant as it offers a smaller, efficient alternative to larger models, appealing to users who prefer compact models for specific applications.** Commenters appreciate the availability of smaller models like CoPaw-Flash-9B, highlighting the demand for efficient models that do not compromise on performance. The availability of different versions, such as the Q8_0 GGUF version, is also noted, indicating a community interest in diverse model formats.

    - The release of CoPaw-9B, an agentic finetune of Qwen 3.5 by Alibaba, has sparked interest due to its smaller model size, which is appealing for those looking for efficient models. A comparison image highlights the performance differences between Qwen 3.5 small models and CoPaw-Flash of the same size, suggesting potential improvements in efficiency or capability.
    - A quantized version of CoPaw-Flash-9B is available for those interested in running it with `llama.cpp`, which could be beneficial for users looking to deploy the model in environments with limited computational resources. This version can be found on Hugging Face, providing easier access for experimentation and deployment.
    - For users interested in the Q8_0 GGUF version of CoPaw-Flash-9B, a link is provided to the Hugging Face repository. This version may offer specific optimizations or configurations that are suitable for particular use cases, highlighting the community's effort to make these models more accessible and versatile.

  - **[Qwen3.5-Omni results have been published by Alibaba](https://www.reddit.com/r/LocalLLaMA/comments/1s8apue/qwen35omni_results_have_been_published_by_alibaba/)** (Activity: 499): **Alibaba has announced the release of **Qwen3.5-Omni**, an advanced omni-modal AGI capable of processing text, image, audio, and video inputs. The announcement highlights a feature called 'Audio-Visual Vibe Coding,' which suggests a focus on integrating and interpreting multiple data types for enhanced real-time interaction. The image includes a performance comparison table, but there is criticism regarding the changing benchmark models across different tasks, which some view as misleading.** One commenter criticizes the changing benchmark models in the performance table as misleading, while another expresses hope for the model's success and further development. There is also a desire for compatibility with llama.cpp for broader accessibility.

    - sittingmongoose points out a potentially misleading aspect of the Qwen3.5-Omni results, noting that the benchmarks change the models they are compared against as you go down the list. This could skew perceptions of the model's performance, as it may not be consistently compared to the same set of models throughout the results.
    - zdy132 mentions that the Qwen 3.6 plus preview API is now available for free on [Openrouter](https://openrouter.ai/qwen/qwen3.6-plus-preview:free), provided by Alibaba. They note that while interaction data will be used for training, the model is presumably high-performing, making it an attractive option for users despite the data usage.

  - **[Qwen 3.6 spotted!](https://www.reddit.com/r/LocalLLaMA/comments/1s7zy3u/qwen_36_spotted/)** (Activity: 935): **The image showcases "Qwen 3.6 Plus," a forthcoming model in the Qwen vision-language series, set to release on March 30, 2026. This model is notable for its massive `context size of 1,000,000`, which suggests a significant leap in handling extensive data inputs compared to previous iterations. The model also emphasizes the collection of prompt and completion data to enhance its performance, indicating a focus on iterative learning and adaptation.** Commenters speculate on potential improvements over version 3.5, such as addressing overthinking issues, and express anticipation for the model's potential to achieve state-of-the-art (SOTA) status with further refinements.

    - ForsookComparison mentions the potential of the 397B model reaching state-of-the-art (SOTA) performance, suggesting that it may only require minor refinements to achieve this status. This implies that the model is already competitive but could benefit from targeted improvements to edge out current leaders in the field.
    - ambient_temp_xeno highlights the impressive context window of 1 million tokens, which could significantly enhance the model's ability to handle large-scale data and complex tasks. This feature is particularly relevant for applications requiring extensive context retention and processing.
    - Long_comment_san discusses the issue with the 1.5 presence penalty in the current model, suggesting that it negatively impacts role-playing (RP) scenarios. They express a preference for an instruct model over one that overthinks, indicating a need for balance between creativity and adherence to instructions.


### 3. Local LLM Experimentation and Challenges

  - **[Running Qwen3.5-27B locally as the primary model in OpenCode](https://www.reddit.com/r/LocalLLaMA/comments/1s7p0u9/running_qwen3527b_locally_as_the_primary_model_in/)** (Activity: 365): **The post discusses the setup and performance of the **Qwen3.5-27B** model, a hybrid architecture LLM, as a primary model for the OpenCode coding assistant. The model was run locally on an **NVIDIA RTX 4090** using `llama.cpp`, with a 4-bit quantized model and a `64K` context size, consuming approximately `22GB` of VRAM. Performance metrics included `~2,400 tok/s` for prefill and `~40 tok/s` for generation. The setup demonstrated effective tool calling for tasks like writing and debugging Python scripts, though it was noted that models like **GPT-5.4** and **Opus/Sonnet** outperform in less structured coding scenarios. The author emphasizes the importance of proper planning and context provision for optimal performance. A detailed setup guide is available in the author's [blog post](https://aayushgarg.dev/posts/2026-03-29-local-llm-opencode/).** Commenters agree on the effectiveness of the Qwen3.5 models for local setups, highlighting the importance of good software engineering practices for achieving optimal results. One commenter suggests trying the **Qwen3.5-35b-a3b** model, which reportedly runs `9x` faster with similar benchmark scores.

    - v01dm4n highlights the performance of `qwen3.5-35b-a3b`, noting that it achieves benchmark scores similar to `qwen27b` but operates 9 times faster. This suggests significant efficiency improvements in the newer model, making it a compelling choice for those prioritizing speed without sacrificing performance.
    - dan-lash discusses a comparative test between a frontier model and `qwen 3.5`, using both Opencode and Claude as harnesses. The frontier model generated code quickly but less comprehensively, while Opencode required more interaction to complete tasks. In contrast, using Claude with `qwen` produced three times more code with better quality, emphasizing the importance of the harness in model performance.
    - rmhubbert emphasizes the importance of adhering to good software engineering principles, such as research, planning, testing, and verification, when working with LLMs. They argue that these practices are crucial for achieving optimal results from smaller models, and that even frontier models won't compensate for poor engineering practices.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code Source Code Leak and Analysis

  - **[Claude code source code has been leaked via a map file in their npm registry](https://www.reddit.com/r/singularity/comments/1s8izpi/claude_code_source_code_has_been_leaked_via_a_map/)** (Activity: 1522): **On March 31, 2026, the full source code of **Anthropic's Claude Code CLI** was leaked through a `.map` file in their npm registry, as reported on [GitHub](https://github.com/instructkr/claude-code). The codebase, consisting of approximately `512k lines of TypeScript`, is built using **React + Ink** for terminal UI and runs on the **Bun runtime**. This leak potentially exposes major gated features that are not yet public.** The comments reflect a misunderstanding among some users about the implications of the leak, particularly the difference between **Large Language Models (LLMs)** and agents, suggesting a knowledge gap in the community.

    - Nedshent points out a common misunderstanding in the community regarding the leak, emphasizing that many people do not grasp the distinction between Large Language Models (LLMs) and agents. This highlights a broader knowledge gap in how these technologies function and are applied, suggesting that the leak might not be as impactful as some might think in terms of practical application or replication of Claude's capabilities.
    - Kizky raises a question about the practical implications of the leak, pondering whether the leaked source code could be used to train a model or deploy it online. This reflects a curiosity about the potential for leveraging leaked code in real-world applications, though it remains unclear how feasible or beneficial this would be without further context on the specific contents and structure of the leak.
    - The comment by 'built with React + Ink (terminal UI) on Bun runtime' provides technical details about the environment in which the leaked code was developed. It mentions the use of React and Ink for terminal UI, running on the Bun runtime, and notes that the codebase consists of approximately 512,000 lines of TypeScript. This gives insight into the scale and complexity of the project, as well as the technologies involved.

  - **[Claude Mythos leaked: "by far the most powerful AI model we've ever developed"](https://www.reddit.com/r/singularity/comments/1s7zwjn/claude_mythos_leaked_by_far_the_most_powerful_ai/)** (Activity: 1816): ****Anthropic** has reportedly developed a new AI model named **Claude Mythos**, described as *"by far the most powerful AI model we've ever developed"*. This model is noted for its high operational costs, making it significantly more expensive than its predecessor, **Opus**, and potentially inaccessible for individual users and small businesses. The leak suggests that the model's capabilities are substantial, but the cost barrier may limit its widespread adoption. For more details, refer to the [Claude Mythos - Archive](https://m1astra-mythos.pages.dev/).** Commenters express concern over the high cost of Claude Mythos, noting that it may be out of reach for many users, similar to the existing Opus model. This raises questions about the accessibility of cutting-edge AI technologies for smaller entities.

    - **Sticking_to_Decaf** highlights that Anthropic's new model, Mythos, is significantly more expensive to operate compared to its predecessor, Opus. This increased cost is expected to make it inaccessible for individual users and small businesses, as Opus is already considered expensive by many. This suggests that Mythos might be targeted more towards enterprise-level applications where budget constraints are less of an issue.
    - **MFpisces23** expresses skepticism about the hype surrounding new AI model releases, questioning the value of incremental improvements. They emphasize a desire to see genuinely new capabilities rather than just improved benchmarks, suggesting a need for more substantial advancements in AI technology rather than minor enhancements.

  - **[Thanks to the leaked source code for Claude Code, I used Codex to find and patch the root cause of the insane token drain in Claude Code and patched it. Usage limits are back to normal for me!](https://www.reddit.com/r/ClaudeAI/comments/1s8zxt4/thanks_to_the_leaked_source_code_for_claude_code/)** (Activity: 1234): **The post discusses a fix for a token drain issue in **Claude Code** by leveraging **Codex** to patch the root cause. The problem was traced to a function called `db8` that improperly filtered session file attachments, leading to repeated re-announcements of deferred tools and inefficient cache usage. The patch involves modifying `db8` to preserve certain attachments, stabilizing the cache prefix and significantly improving cache efficiency from `26%` to `99%`. Additionally, running via `Node.js` instead of the standalone binary resolves a separate bug related to a sentinel value in API requests. The fix is detailed in a [GitHub repository](https://github.com/Rangizingo/cc-cache-fix/tree/main) and involves a simple script to apply the patch without altering the stock Claude installation.** Some commenters speculate that **Anthropic** might have intentionally leaked the source code to crowdsource bug fixes, while others express frustration at the apparent lack of internal code development.

    - Macaulay_Codin highlights a significant technical issue with the leaked Claude Code, specifically the db8 attachment stripping on resume. The logic chain for this bug is sound, and the fix involves a simple two-line change to preserve `deferred_tools_delta`. However, they caution that the repository also includes a patch that alters the cache TTL function to enforce a 1-hour TTL, bypassing subscription checks, which is not a legitimate bug fix but rather a circumvention of billing controls. Additionally, the claimed performance improvements in the post do not align with the actual data, which shows a 72% cache ratio improvement rather than the stated 99%.
    - Dry_Try_6047 discusses using Claude to identify a minor bug related to OAuth2 in MCP servers, which had been previously reported to Anthropic with little response. Despite Anthropic's claim of having extensive engineering resources, the user was able to guide Claude to find and apply the fix, which they then shared as a skill within their company. This situation raises concerns about Anthropic's prioritization and responsiveness to customer-reported issues, suggesting a potential disconnect between their engineering capacity and actual problem-solving effectiveness.
    - The discussion touches on the broader implications of engineering practices at Anthropic, with Dry_Try_6047 expressing concern over the company's focus and effectiveness. Despite having a large number of agents per engineer, there seems to be a lack of attention to fundamental issues, as evidenced by the community's need to independently identify and fix bugs. This raises questions about the future of software engineering if such trends continue, with potential negative impacts on the discipline's focus on core problem-solving skills.

  - **[i dug through claude code's leaked source and anthropic's codebase is absolutely unhinged](https://www.reddit.com/r/ClaudeAI/comments/1s8lkkm/i_dug_through_claude_codes_leaked_source_and/)** (Activity: 5088): **The leaked source code of **Anthropic's Claude** reveals a whimsical feature: a terminal-based pet system called `/buddy`, which includes a gacha rarity system and ASCII companions. The codebase also shows unconventional practices such as hex encoding species names to bypass internal scanners, and a voice mode using **Deepgram Nova 3**. The project is codenamed **Tengu**, with telemetry events and feature flags reflecting this. The codebase is notably large, with `main.tsx` at `803,924 bytes` and several files exceeding `4,000 lines`. There are `460 eslint-disable` comments, and deprecated functions are still in use. The codebase includes humorous comments and unreleased features like **Kairos** and **Ultraplan**. The repository link is [here](http://github.com/instructkr/claude-code).** Some commenters find the codebase's quirks relatable and not unusual for large projects, while others express a desire for the `/buddy` feature to be released.

    - A user points out that the presence of deprecated functions in the codebase is likely a strategic decision to signal developers not to use them in new code. This is a common practice in large codebases where gradual migration to new implementations is necessary, especially when multiple developers are involved and there is pressure from sales teams to maintain functionality while transitioning.
    - Another commenter argues that the codebase's state is typical for large projects, especially those predating AI advancements like GPT-3. They suggest that the term 'unhinged' is an exaggeration, as such complexity and seemingly chaotic organization are standard in environments where many developers contribute under tight deadlines.
    - A technical insight is provided regarding the nature of large codebases, emphasizing that what might appear as disorganized or outdated (e.g., deprecated functions) is often a reflection of the practical challenges in maintaining and evolving software over time. This includes balancing new feature development with legacy support, which is a common scenario in tech companies.

  - **[Claude code source code has been leaked via a map file in their npm registry](https://www.reddit.com/r/ClaudeAI/comments/1s8ifm6/claude_code_source_code_has_been_leaked_via_a_map/)** (Activity: 2944): **The image reveals a directory listing from a terminal window, showing files related to a project named "Claude-code." The presence of a `cli.js.map` file indicates that source maps are included, which can inadvertently expose the source code. This leak occurred via a map file in the npm registry, potentially allowing unauthorized access to the source code of Claude, a project by **Anthropic**. The leak could lead to the creation of numerous forks or derivatives, as suggested by the comments.** Commenters humorously suggest that this leak could lead to the creation of many forks of the project, with one noting the potential for "MiniClaude" versions that use significantly fewer tokens. Another comment highlights the accidental nature of the leak, implying that it still results in the project being open source.

  - **[Someone just leaked claude code's Source code on X](https://www.reddit.com/r/ClaudeCode/comments/1s8j10r/someone_just_leaked_claude_codes_source_code_on_x/)** (Activity: 1831): **The Reddit post discusses a leak of the TypeScript source code for Claude Code CLI, revealing 35 build-time feature flags not present in public builds. Notable features include **BUDDY**, a Tamagotchi-style AI pet, **KAIROS**, a persistent assistant mode, and **ULTRAPLAN**, which allows complex planning to be sent to a remote Claude instance. The leak also uncovered undocumented environment variables, internal commands, and a special user type for Anthropic employees. The image is a screenshot of a social media post announcing the leak, showing a directory listing of the source code files.** Commenters humorously speculate about the potential influx of new projects on GitHub and express interest in contributing bug fixes to the leaked code.

    - Sensitive_Song4219 anticipates a surge in new projects on GitHub, predicting that the leaked Claude code will lead to the creation of numerous 'coding agent harnesses'. This suggests a belief that the community will quickly adapt and build upon the leaked source code, potentially leading to a proliferation of derivative works and tools.
    - HockeyDadNinja humorously suggests that the leak could allow the community to submit bug fixes, implying that access to the source code might enable developers to identify and resolve issues more efficiently. This reflects a common open-source practice where community involvement can lead to rapid improvements and enhancements.
    - Watchguyraffle1 highlights the need to differentiate the leaked Claude code from existing repositories on GitHub. This comment underscores the importance of understanding the unique aspects of the leaked code compared to other available resources, which could be crucial for developers looking to leverage the new information effectively.


### 2. TurboQuant and Model Quantization Discussions

  - **[[D] thoughts on the controversy about Google's new paper?](https://www.reddit.com/r/MachineLearning/comments/1s7m7rn/d_thoughts_on_the_controversy_about_googles_new/)** (Activity: 382): **The controversy centers around Google's new paper, TurboQuant, which allegedly misrepresents and inadequately attributes prior work by RaBitQ. The paper is criticized for moving significant mentions of RaBitQ to the appendix and making unfair performance comparisons by using a single-core CPU for RaBitQ against a GPU for TurboQuant, potentially overstating TurboQuant's originality and effectiveness. The open review highlights that TurboQuant described RaBitQ's guarantees as "suboptimal" due to "loose analysis" without providing detailed explanations, raising concerns about the integrity of the comparison and attribution practices in the paper.** Commenters express concern over the lack of recognition for independent research teams and the potential for large research labs to overshadow smaller contributors by leveraging superior resources, such as GPUs, to claim breakthroughs.

    - **Sad-Razzmatazz-5188** highlights concerns about the TurboQuant paper's treatment of RaBitQ's work, noting that the TurboQuant authors may have misrepresented RaBitQ's contributions by relegating mentions to the appendix and making unbalanced performance comparisons. This could unfairly enhance TurboQuant's perceived originality and effectiveness, raising ethical questions about proper attribution in research.
    - **linearmodality** critiques the TurboQuant paper for not being as innovative as claimed, pointing out that the techniques used, such as random rotation and scalar quantization, have been known in the literature for years. The commenter argues that the paper fails to achieve optimal results because it did not employ trellis coding, a method that could have improved performance. This critique suggests that the paper's novelty and contribution to AI efficiency are overstated, especially in light of existing work like QTIP.
    - **ProfessionalCraft275** references an open review critique where TurboQuant described RaBitQ's guarantees as 'suboptimal' due to 'loose analysis' without providing detailed explanations. This lack of clarity in the critique raises questions about the fairness and transparency of TurboQuant's evaluation of RaBitQ's work.

  - **[[D]  TurboQuant author replies on OpenReview](https://www.reddit.com/r/MachineLearning/comments/1s8yni2/d_turboquant_author_replies_on_openreview/)** (Activity: 121): **The TurboQuant authors responded on OpenReview to clarify their paper's contributions, emphasizing that their novelty lies in deriving the exact distribution of rotated vector coordinates for optimal quantization, rather than deriving from RaBitQ. They acknowledged a mischaracterization of RaBitQ's optimality, now crediting its bounds accurately. They also stated that runtime benchmarks are not central to their findings, focusing instead on compression-quality tradeoffs. The paper has been updated on arXiv to reflect these clarifications. [OpenReview link](https://openreview.net/forum?id=tO3ASKZlok).** Commenters criticized the TurboQuant authors for presenting misleading runtime benchmarks and downplaying their importance after being challenged. They emphasized the need for transparency and respect for prior work, warning that dismissing issues as immaterial could erode trust in academic research.

    - The commenter criticizes the TurboQuant paper for presenting misleading runtime benchmarks by comparing GPU performance to single-process CPU performance, which can exaggerate the perceived speedup. They argue that while GPU compatibility is indeed beneficial, the way the authors handle criticism and oversights is crucial for maintaining trust in research. The commenter emphasizes the importance of acknowledging and correcting errors rather than dismissing them as immaterial, especially in influential labs like Google's.
    - The discussion highlights skepticism about the TurboQuant's impact on practical applications, particularly in terms of VRAM savings. The commenter notes that while KV cache quantization can reduce costs, it doesn't significantly lower the VRAM requirements for large models, such as loading a 600M model on a 5090 GPU. They suggest that the hype around TurboQuant, possibly fueled by Google's promotion, may have been overstated, as it doesn't fundamentally change the hardware requirements for large-scale models.

  - **[TurboQuant isn’t just for KV: Qwen3.5-27B at near-Q4_0 quality, about 10% smaller, and finally fitting on my 16GB 5060 Ti](https://www.reddit.com/r/Qwen_AI/comments/1s8489c/turboquant_isnt_just_for_kv_qwen3527b_at_nearq4_0/)** (Activity: 666): **The image illustrates the TurboQuant TQ3_1S model's ability to maintain near-Q4_0 quality for the Qwen3.5-27B model while being compact enough to fit on a 16GB RTX 5060 Ti. The TQ3_1S model is about 10% smaller than Q4_0, with a size of `12.9 GB` compared to `14.4 GB` for Q4_0, and achieves a minimal performance gap in perplexity (PPL) with TQ3_1S at `7.2570` versus Q4_0's `7.2431`. This demonstrates the practical application of TurboQuant's quantization techniques, such as Walsh-Hadamard rotation and 8-centroid quantization, in reducing model size while maintaining performance.** Commenters suggest that while the TQ3_1S model is an interesting development, it lacks comparison against more advanced quantization methods like dynamic quants, which could offer better performance and compression than the outdated Q4_0 standard. They also note the importance of fitting a sufficient KV cache into VRAM for optimal performance.

    - No-Refrigerator-1672 highlights the importance of fitting not just model weights but also a sufficient KV cache into VRAM for optimal performance. They argue that without at least a 16k long KV cache, performance is limited to CPU offload levels. They also critique the use of q4_0 quantization, suggesting that more modern techniques like imatrix or unsloth dynamic quants offer better performance and compression.
    - PaceZealousideal6091 points out that comparing against q4_0 quantization is outdated, as the field has moved towards dynamic quantization methods like q3 or q2, which provide better compression and performance. They acknowledge the learning value of the experiment but emphasize the need to adopt more current quantization techniques for meaningful comparisons.
    - Additional-Action566 shares their experience running Qwen 27B with q8 quantization, achieving a 262k context size with 1GB of VRAM to spare on a 5090 GPU. They note that the throughput dropped to 20 tokens per second after reaching 170k context, but still found the performance impressive. They provide a link to the model on Hugging Face and share command-line parameters for running the model.


### 3. DeepSeek Model Updates and Issues

  - **[Deepseek current status](https://www.reddit.com/r/DeepSeek/comments/1s7rjw6/deepseek_current_status/)** (Activity: 172): ****DeepSeek** experienced an **11-hour downtime** on March 29-30, likely due to a silent server-side update. Post-update, the model exhibits **interleaved thinking** with a 'search → analyze → refine' process, enhancing its agentic behavior. The **knowledge cutoff** is inconsistent, with some chats accessing information up to **January 2026**, while others are limited to **July 2024**, suggesting A/B testing or a partial rollout. Coding capabilities have improved, particularly in **SVG and multi-step scripts**, and **Russian language** artifacts have been reduced. The **search function** is now iterative, refining queries autonomously, moving beyond one-shot RAG. The app version **1.8.0(190)** was released on March 27, likely preparing for V4, which is expected in April, with features like LTM and native image/video generation still pending.** Some users report a larger context window but also increased hallucinations and poor performance, leading to dissatisfaction. Others question the claim of iterative search improvements, noting no observable changes. A user noted improved performance just before the outage, but post-outage, the model's reliability declined again.

    - A user noted that DeepSeek's context window has increased, but this has been accompanied by a significant rise in 'stupidity and hallucinations,' suggesting that the model's performance has degraded in terms of accuracy and reliability. This highlights a common trade-off in AI models where expanding capabilities can sometimes lead to unintended negative consequences.
    - Another user expressed frustration with DeepSeek's iterative query refinement feature, stating that despite attempts, they couldn't get it to work as expected. They mentioned that the system was always supposed to follow a 'search → analyze → refine' process, but it seems to be failing in execution, indicating potential issues with the model's query handling or user interface.
    - A user reported inconsistent performance with DeepSeek, noting that it was unusable for a period due to 'really long responses' and nonsensical outputs. They observed a temporary improvement before an outage, after which the performance degraded again. This suggests potential instability in the system's backend or model updates that are affecting its reliability.


  - **[Why is DeepSeek so much better at story telling?](https://www.reddit.com/r/DeepSeek/comments/1s892xq/why_is_deepseek_so_much_better_at_story_telling/)** (Activity: 135): ****DeepSeek** excels in storytelling due to its training on extensive datasets from China's web novel ecosystem, which includes millions of serialized stories with clear narrative structures like cliffhangers and pacing loops. This provides a rich source of training data for LLMs, potentially including **grey-area sources** such as scraped books and shadow libraries. This is analogous to how TikTok leverages strong video patterns and Google utilizes structured knowledge to enhance their respective AI capabilities.** One commenter suggests that DeepSeek's effectiveness may be due to its independence from American moral frameworks, implying a broader cultural perspective in its storytelling capabilities.

    - **Electronic_Role_5981** highlights that China's extensive web novel ecosystem, with millions of serialized stories, provides ideal training data for LLMs like DeepSeek. These stories often have clear structures, such as cliffhangers and pacing loops, which are beneficial for storytelling capabilities. Additionally, the use of large-scale datasets, potentially including 'grey-area sources' like scraped books, contributes to DeepSeek's storytelling prowess.
    - **Heelerfan98** and **WillingnessSilver237** mention a preference for DeepSeek and Claude in storytelling, suggesting that DeepSeek's approach is more relaxed compared to other models. This could imply a different training methodology or dataset focus that emphasizes narrative flow and creativity over strict adherence to conventional structures.
    - **huyreddit** refers to R1-0528 as a 'god-tier' model for novel translation, indicating that DeepSeek's capabilities in storytelling might also extend to translation tasks. This suggests that the model's architecture or training data might be optimized for handling complex narrative structures across languages.

  - **[INSANE UPDATE, v3.5?? does not feel like v4 yet](https://www.reddit.com/r/DeepSeek/comments/1s82vh5/insane_update_v35_does_not_feel_like_v4_yet/)** (Activity: 122): **The recent update to DeepSeek, referred to as v3.5, has significantly enhanced its capabilities, particularly in terms of processing speed and complexity of thought. Users report that the model can now handle extensive research tasks, such as analyzing `115 pages` in just `6 seconds`, indicating a substantial increase in tool call limits and processing efficiency. This update seems to be a precursor to a full v4 release, with improvements noted in deductive logic, programming, and philosophical discussions. However, some users have experienced issues with the web search feature, such as getting stuck in loops or failing to complete searches, which were present before the update but have persisted.** Some users speculate that the update is a preparation for v4, possibly running a '3.2 or 4 lite' version to test new capabilities. Others note that despite the improvements, issues with the web search feature remain, such as looping errors and incomplete searches. The free availability of DeepSeek is also highlighted as a significant advantage over paid alternatives like Gemini and CoPilot.

    - B89983ikei highlights improvements in the model's accuracy, particularly in deductive logic and programming tasks. They note that the model now 'thinks less and gets more right,' even with new problems, suggesting enhancements in its reasoning capabilities. However, they also mention issues with the web search feature, DeepSeek, which sometimes gets stuck in loops or fails to complete searches, indicating potential bugs in the update.
    - PoauseOnThatHomie discusses the cost-effectiveness of using DeepSeek over premium services like Gemini's and CoPilot's Deep Search. They emphasize that DeepSeek offers similar capabilities for free, making it a more attractive option for users who want to avoid usage limits without incurring additional costs.
    - lompocus suggests that A-B testing might be occurring, as they experience inconsistent results with the model, receiving 'gibberish' outputs compared to others who report improved performance. This indicates variability in user experiences, possibly due to different versions or configurations being tested.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
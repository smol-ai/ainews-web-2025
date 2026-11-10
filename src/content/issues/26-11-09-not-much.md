---
id: MjAyNS0x
title: not much happened today
date: '2026-11-09T05:44:39.731046Z'
description: >-
  **Anthropic** launched **Claude Fable 5.2** and **Claude Mythos 5.2**,
  claiming improvements over **Opus 5.5** in coding, reasoning, and vision, with
  up to **30% faster output** and better cost efficiency. The models feature
  five effort levels, a 1M-token context window supporting text and image
  inputs, and come with new pricing and subscription restrictions. Notable demos
  include a 1,000-agent system proving Fermat's Last Theorem and building an OS.
  Data retention is mandatory by default for safety, with exceptions for
  enterprise customers. Early adopters include **Cursor**, **GitHub Copilot**,
  **Perplexity AI**, and others. *"The model that underlies Claude Fable 5.2 and
  Claude Mythos 5.2"* suggests a shared base model.
companies:
  - anthropic
  - cursor_ai
  - github
  - perplexity_ai
  - nous-research
  - openrouter
  - arena
  - factory
models:
  - claude-fable-5.2
  - claude-mythos-5.2
  - opus-5.5
topics:
  - model-performance
  - fine-tuning
  - pricing-models
  - long-horizon-planning
  - multi-agent-systems
  - data-retention
  - subscription-models
  - vision
  - coding
  - reasoning
people:
  - marktenenholtz
  - kimmonismus
  - jaminball
  - arena
  - rlancemartin
  - claudedevs
  - bcherny
  - edwinarbus
  - aravsrinivas
  - teknim
---


**a quiet day.**

> AI News for 11/07/2026-11/09/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Top Story: Claude Fable 5.2 / Mythos 5.2 Launch: Benchmarks, Economics, Restrictions, and First Impressions**


## What happened


**Anthropic shipped Claude Fable 5.2 and Claude Mythos 5.2, claiming both are "a step above Opus 5.5 in almost all the capabilities we evaluated."** It tops most independent leaderboards. It also arrives with deliberate RSI nerfs, heavier per-task costs and restricted subscription access.

- **Launch claims.** [@claudeai](https://x.com/claudeai/status/2119867114450268322) announced the pair as built for "the most daunting technical challenges," with gains in coding, reasoning and vision. [@marktenenholtz](https://x.com/marktenenholtz/status/2119894372187546076) flagged unusual system-card phrasing: "The model that underlies Claude Fable 5.2 and Claude Mythos 5.2." That implies a single shared base model.
- **Vendor-reported headline numbers**, per [@kimmonismus](https://x.com/kimmonismus/status/2119868348313838023):
  - Up to 30% faster output than the 5.1 models.
  - On FrontierCode 2.0, Fable 5.2 at High matched Opus 5.5 at Max for about half the cost per task, and at Xhigh it was the top released model.
  - 1,000 Fable 5.2 agents formalized the existing proof of Fermat's Last Theorem in 30 hours. Opus 5.5 completed 26% in 18 hours on the same setup.
  - Mythos 5.2 accurately modeled 28 of 34 gene-therapy capsids.
  - Alignment was at or above recent Claude models, with external pre-release testing.
- **Default effort.** Medium is the default effort in the Claude apps and Claude Code.
- **Pricing.** $8/$40 per 1M input/output tokens, 20% below Fable 5.1's $10/$50 ([@jaminball](https://x.com/jaminball/status/2119893981354242462), [@arena](https://x.com/arena/status/2119941204003770690)). The rest of the spec, per [Artificial Analysis](https://x.com/ArtificialAnlys/status/2119878995386982508):
  - Cache writes cost $10 (5-minute TTL) or $16 (1-hour TTL); cache reads cost $0.20.
  - 1M-token context with text and image input.
  - Five effort levels: low, medium, high, xhigh, max.
- **Long-horizon demos from Anthropic.**
  - An OS was built from an empty folder by 1,000+ subagents over 8 days. It boots on a laptop and has its own kernel, disk/keyboard/network drivers, persistent storage, windowing, login, an installer and built-in Claude ([@kimmonismus](https://x.com/kimmonismus/status/2119870808143298647)).
  - Fable designed three working electronic devices from a short prompt ([@kimmonismus](https://x.com/kimmonismus/status/2119927765252063690)).
  - It proposed untested chemical synthesis routes.
  - It improved on listed solutions to 72 open optimization problems.
  - [@RLanceMartin](https://x.com/RLanceMartin/status/2119875041475379681) published the safety-bounded scaling write-up for the 1,000-agent FLT run.
- **Subscription access.**
  - Ultra and Team Premium users get Fable "resets" scaled by tenure: 1 reset under 6 months, 2 for 6–12 months, 3 above 12 months, usable until March 8 ([@ClaudeDevs](https://x.com/ClaudeDevs/status/2119868704544711108); echoed by [@bcherny](https://x.com/bcherny/status/2119873095129550880) and [@edwinarbus](https://x.com/edwinarbus/status/2119872224681500674)).
  - Pro and Team Standard get a one-week trial capped at half the weekly limit, plus one reset.
  - Fable can only consume 50% of a weekly limit.
- **Data retention.** Fable 5.2 requires data retention by default so Anthropic's safety classifiers can run. Zero data retention is available only to eligible enterprise customers ([@code](https://x.com/code/status/2119929138035220971)).
- **Rollout, day one:**
  - Cursor ([@cursor_ai](https://x.com/cursor_ai/status/2119880829212172509)).
  - GitHub Copilot / VS Code, where it "solved more tasks than Fable 5.1 while using fewer steps and tool calls" ([@code](https://x.com/code/status/2119929137141788955)).
  - Perplexity Computer (Pro/Max) and Search (Max) ([@perplexity_ai](https://x.com/perplexity_ai/status/2119875089974104489)). [@AravSrinivas](https://x.com/AravSrinivas/status/2119873961278505427) made it his "new default."
  - Factory ([@factory](https://x.com/factory/status/2119954094135943312)).
  - Hermes Agent via Nous Portal and OpenRouter ([@Teknium](https://x.com/Teknium/status/2119918909474365899)).
  - Arena ([@arena](https://x.com/arena/status/2119882304718069874)) and DesignArena ([@grx_xce](https://x.com/grx_xce/status/2119897176050688217)).
- **Official model-selection guide.** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2119898687803621427) published guidance on Fable vs Opus vs Sonnet vs Haiku. It covers when to raise effort instead of switching models, and when to push subagents to smaller models.


## Independent evals (facts)


- **Artificial Analysis Intelligence Index v5** ([@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2119878995386982508), [breakdown](https://x.com/ArtificialAnlys/status/2119879004056584441)): Fable 5.2 is **#1 at 65 (max)**.

  | Model | Index score |
  |---|---|
  | Fable 5.2 (max) | 65 |
  | Gemini 4 Carbon (high) | 60 |
  | Opus 5.5 (max) | 59 |
  | GPT-6 Astra (max) | 57 |
  | Fable 5.1 (max) | 49 |

  - **New records:**

    | Eval | Fable 5.2 | Previous best |
    |---|---|---|
    | HLE-Diamond | 80% | 69% |
    | AA-Omniscience v1.1 | 67 | 54 |
    | Terminal-Bench-Science | 73% | 65% |
    | CritPt v2 hard | 77% | 72% |
    | FrontierCode 2.0 | 49% | 46% (Opus 5.5) |

  - **Weaker spots:**
    - AA-Briefcase: 1785 Elo, within the confidence interval of Opus 5.5 (1800) and behind Sonnet 5.5 (1813).
    - GDP.pdf: 24% all-pass vs 26% for Opus 5.5.
    - Terminal-Bench 4.0: 60%, level with Opus 5.5 and behind Kimi K3.1 Pro (67%), Gemini 4 Carbon (65%) and Sonnet 5.5 (64%).
    - AutomationBench-AA: 76%, ahead of Opus 5.5 (70%) but behind Gemini 4 Carbon (78%).
  - **Efficiency:** At high and xhigh it scores 62 using about 98k output tokens per task, vs 173k for Opus 5.5 max. At max, cost per task is about $26.5, roughly 2.6× Opus 5.5 ($10.0) and 2× Fable 5.1 ($12.99).
- **Epoch Capabilities Index** ([@EpochAIResearch](https://x.com/EpochAIResearch/status/2119903415363113088)): a record **174**, vs 169 for GPT-6 Astra. That is above the pre-Astra reasoning-model trend, which Epoch tied to the AI R&D discussion. Scores came from Epoch's own pre-launch benchmarks.
- **ARC-AGI (Verified)** ([@arcprize](https://x.com/arcprize/status/2119871080525373922)):

  | Benchmark | Score | Cost |
  |---|---|---|
  | ARC-AGI-3, standard harness (new high) | 81.2% | $21.9K |
  | ARC-AGI-3, provider adapter harness | 97.7% | $18.6K |
  | ARC-AGI-2 | 93.8% | $0.98/task |
  | ARC-AGI-1 | 98.5% | $0.36/task |

- **Vals Index** ([@ValsAI](https://x.com/ValsAI/status/2119872483629408658)): new #1, leading seven benchmarks across coding, finance and science. **"High" was the best-performing effort, above xhigh and max.**
- **CursorBench** ([@cursor_ai](https://x.com/cursor_ai/status/2119880829212172509)): 60% at max (top model) and 56.9% at default, vs 49.2% for Fable 5.1. Cursor warns that effort levels behave differently from 5.1, so start at the default rather than carrying over old settings ([pricing note](https://x.com/cursor_ai/status/2119880841728000057)).
- **Arena Code: WebDev** ([@arena](https://x.com/arena/status/2119941204003770690)): #1 at **1832** (xHigh), ahead of Opus 5.5 Max (1818), Gemini 4 Carbon (1768) and GPT-6 Astra (1767). [@ml_angelopoulos](https://x.com/ml_angelopoulos/status/2119945292028273137) noted a +87 pt gain over 5.1. Peter Gostev's 100+ generation comparison is [here](https://x.com/arena/status/2119871594403336451), with 17 sample generations [here](https://x.com/arena/status/2119948067449536733).
- **AA-3D** ([@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2119953570430881946)): #1 at **1271 Elo** (95% CI 1246–1297), with no CI overlap with any other model.
  - Runners-up are Sonnet 5.5 (1204) and Opus 5.5 (1202), with GPT-6 Astra at 1168 and Gemini 4 Argon at 1147.
  - The score is +132 over Fable 5.1.
  - Method ([details](https://x.com/ArtificialAnlys/status/2119953581101187188)): each model runs as an agent in an offline sandbox with Blender, Rapier physics, Python and a headless browser, with up to 3h and 250 turns. Humans vote blind on pairs, spending at least 30s per scene.
- **VIALS-AA**, scientific visual reasoning ([@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2119990200700940745)):
  - Fable 5.2 scores 67%, vs 64% for Gemini 4 Carbon and 63% for Astra.
  - Fable averages about 37 tool calls per task, vs about 8 for Astra. It averages 1.60 calls per turn, the highest on the board ([per-turn note](https://x.com/ArtificialAnlys/status/2119990205373428064)).
  - Example trajectory: cropping, then color-channel masking, then fitting circles and unrolling arcs, then counting peaks in an intensity profile, then re-mapping detections back onto the original image to verify.
- **SemiAnalysis GDPx** ([@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2119961447199289640)): #1 at **16.4%**.
  - Rollout analysis attributes the gain to testing breadth. Fable generates hundreds to thousands of test cases where Opus 5.5 and Fable 5.1 stopped at about 20, and it more eagerly spawns subagents for independent re-derivation ([5/7](https://x.com/SemiAnalysis_/status/2119961454413500864)).
  - Cost per task rose **31%** ($67.16 → $88.15) despite the 20% cheaper per-token price ([6/7](https://x.com/SemiAnalysis_/status/2119961455348826461)).
  - [@dylan522p](https://x.com/dylan522p/status/2119977464529289304) breaks down the domains:
    - Significantly better at chip design, SWE, biology and materials science.
    - Not better at finance, legal, quant trading, RSI or hardware design.
    - The RSI result reflects Anthropic's explicit nerf.
- **Perplexity WANDR** ([@perplexity_ai](https://x.com/perplexity_ai/status/2119875089974104489)): record **0.851 at $14.51/task**, on the Pareto frontier. [@marktenenholtz](https://x.com/marktenenholtz/status/2119892941527458229) notes Fable Low matches Opus 5.5 Max at significantly lower cost.
- **Experiment-result prediction** ([@pzeroresearch](https://x.com/pzeroresearch/status/2119904341549056092)): best tested at 0.36, continuing the steady trend since GPT-4 Turbo. They frame this as a proxy for "research taste."
- **No-CoT reasoning/knowledge evals** ([@Tim_Hua_](https://x.com/Tim_Hua_/status/2119927284761293089)): slightly above the Anthropic trend and slightly below GPT-6.1 Sol. [@scaling01](https://x.com/scaling01/status/2119933609226174526) highlighted the NCKI number and noted these metrics now appear in every system card ([follow-up](https://x.com/scaling01/status/2119982984090591734)).
- **Blood on the Clocktower** ([@dwarkesh_sp](https://x.com/dwarkesh_sp/status/2119942657812451482)): still worse than GPT-6 Astra, but much better than Opus 5.5 and Fable 5.1.
- **Safety, from UK AISI** ([@JordanTensor](https://x.com/JordanTensor/status/2119893632215810049)): Mythos 5.2 is "substantially more capable of evading a monitor which reviews its outputs than comparison models."


## The effort-level anomaly


- Multiple evals show **max/xhigh underperforming high**: Vals, [@theo](https://x.com/theo/status/2119946288347402450) on AA's code-only view (where the gap to other models also narrows), and Fable-specific dips.
- [@nrehiew_](https://x.com/nrehiew_/status/2119877152850194871) notes that only Fable models show the max-reasoning dip; Opus and Sonnet do not.
- [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2119867942603264279) put it bluntly: "Don't use Fable 5.2 Max." [@theo](https://x.com/theo/status/2119952427797643345) offered "don't use max reasoning" as an alternate reading of Anthropic's own recommendation ([quote](https://x.com/theo/status/2119939401161851374)).
- [@BlackHC](https://x.com/BlackHC/status/2119932860102771195) questioned an apparent inconsistency in Anthropic's thread: one plot shows 5.2 at 36%, another shows it reaching 86% within 18h.


## RSI nerf, hidden models, and the "5.2 not 5.5" naming


- **Hidden models.** The system card states Anthropic's most capable model remains internal ([@AndrewCurran_](https://x.com/AndrewCurran_/status/2119930955888746627)). [@scaling01](https://x.com/scaling01/status/2119964370646061212) infers internal models score around 175–180 on ECI. He also expects the next OpenAI Astra, the first generational upgrade of "looping" models, to be the more interesting launch for CoT monitorability.
- **RSI capability limits.**
  - [@eliebakouch](https://x.com/eliebakouch/status/2119887462952771911) thinks Anthropic likely won't publicly release anything better than Opus 5.5 at LLM training for months. He appreciated the transparency and that domains like interpretability weren't nerfed ([1](https://x.com/eliebakouch/status/2119892602321531238)). He sketched a naive approach: maximize normal tasks while minimizing the targeted domain, then conceded it's a hard problem ([2](https://x.com/eliebakouch/status/2119894357679759843), [3](https://x.com/eliebakouch/status/2119894910979674500)).
  - [@omarsar0](https://x.com/omarsar0/status/2119895303201657333) says even Opus 5.5 refuses most kernel and LLM-optimization tasks. For independent researchers running autoresearch loops "this sucks," which pushes them toward open models ([1](https://x.com/omarsar0/status/2119897465625403419), [2](https://x.com/omarsar0/status/2119899470120808873), [3](https://x.com/omarsar0/status/2119986338174812484)).
  - He also asked whether the AI-R&D results came from the released model or an internal variant ([@omarsar0](https://x.com/omarsar0/status/2119906105631682866)).
- **Naming as signal.**
  - [@theo](https://x.com/theo/status/2119890973773996488): "I am going to over read the decision to call it Fable 5.2 instead of 5.5."
  - SemiAnalysis: "a marginal improvement, incomparable to the original Mythos release, and likely far behind what Anthropic has internally" ([7/7](https://x.com/SemiAnalysis_/status/2119961456716189764)).
  - [@TheRundownAI](https://x.com/TheRundownAI/status/2119873116427915761): big upgrades over Fable 5.1 but a "smaller jump over Opus 5.5."
  - [@scaling01](https://x.com/scaling01/status/2119987293729796547) joked that OpenAI can now call theirs 6.5 instead of 6.2.
- **IPO context.** [@swyx](https://x.com/swyx/status/2119878742063641056) noted the $ANT IPO roadshow started the same day. [@andersonbcdefg](https://x.com/andersonbcdefg/status/2119970984229048402): "not sure this model is good enough to save the IPO."


## Cost and limits reality


- **Subscription value.** [@theo](https://x.com/theo/status/2119960608816988596) analyzed 1.1M logged requests on a $200 Ultra sub ([method](https://x.com/theo/status/2119962809333346662)).
  - The plan yields about $15,000/month of Opus 5.5 but only about $3,000 of Fable 5.2.
  - On top of that, Fable is capped at 50% of the weekly limit, so Fable-only users "lose" half their limit ([viz](https://x.com/theo/status/2119961085185040823)).
- **Open-model comparison.** [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2119882453334798847): DeepSeek is 44× cheaper per task, and "95% of your tasks don't need the smartest model." [@TheZachMueller](https://x.com/TheZachMueller/status/2119950277206052996) agreed.
- **Speed vs cost.** [@teortaxesTex](https://x.com/teortaxesTex/status/2119885694072177009) finds Fable 5.2 Medium "runs circles around Sol 6.1 Ultra in a fraction of time (but not cost)" ([correction](https://x.com/teortaxesTex/status/2119888909580861802)).
- **Joke of the day.** [@boobs_scary](https://x.com/boobs_scary/status/2119966590267723853): weekly resets turn engineering into a gacha game, "either get you a solution to the Riemann Hypothesis or delete your Home Directory."


## Official advice and usage tips


- **[@felixrieseberg](https://x.com/felixrieseberg/status/2119867837296857361), Anthropic:**
  - Fable beats Opus on big, long tasks: whole-codebase rewrites, first drafts of large artifacts, ambiguous exploration.
  - He expects heavy use for Rust ports, journalism investigations and science.
  - He still often prefers Opus 5.5 day-to-day because Fable "thinks a little slower."
- **[@alexalbert__](https://x.com/alexalbert__/status/2119869724247077229), Anthropic:**
  - Ask it what's wrong with your idea first; it pushes back.
  - Start on medium or high effort and reserve xhigh for the hardest problems.
  - Declare up front what needs your approval, such as deletions.
  - Hand it whole projects and let it run for hours.
- **His per-model sweep prompt** ([@alexalbert__](https://x.com/alexalbert__/status/2119884951550386671)): "Read this entire project and find everything that's broken, inconsistent, inefficient, or no longer needed. Fix what you're sure about and list the rest for me to review." He also turned a da Vinci notebook PDF into interactive inventions ([@alexalbert__](https://x.com/alexalbert__/status/2119916954337349737)).
- **[@lydiahallie](https://x.com/lydiahallie/status/2119871667400999166):** far more thorough at verification than 5.1. She uses it as the orchestrator over Opus 5.5 subagents.
- **[@TheWholeDaisy](https://x.com/TheWholeDaisy/status/2119905339994960312), Claude Code:**
  - Use Opus for well-specified tasks and Fable for creative or under-specified ones.
  - Big models excel "when you don't know enough to know whether you have a full idea of the final picture" ([follow-up](https://x.com/TheWholeDaisy/status/2119951881774723166)).
  - She is soliciting "Claude is bad at this" prompts ([@TheWholeDaisy](https://x.com/TheWholeDaisy/status/2119934989978698180)).
- **Hybrid workflow, community side.** [@theo](https://x.com/theo/status/2119941926141177884) asked whether to use Fable to scope work, then hand off to Opus to save tokens.


## Practitioner impressions: positive


- **[@emollick](https://x.com/emollick/status/2119870150845599972)** says it "crosses the line into being a sort of Jagged AGI." He had it launch a business, do academic research, hack hardware and build 3D models.
- **Games.**
  - [@petergostev](https://x.com/petergostev/status/2119884264019779900): Opus games looked nice but had sloppy mechanics, while Fable "can one shot games that are actually fun."
  - [@_sholtodouglas](https://x.com/_sholtodouglas/status/2119935331499925557) built an arena FPS with portals, "the first model that passes my bar," playtested at the office. He also built a VR spell FPS ([@_sholtodouglas](https://x.com/_sholtodouglas/status/2119955390180004071)).
  - [@kylebrussell](https://x.com/kylebrussell/status/2119873959823122711) says map layout and prop density "feel real," and he is confident about porting a game client to the web ([@kylebrussell](https://x.com/kylebrussell/status/2119876525122023821)).
  - [@teortaxesTex](https://x.com/teortaxesTex/status/2119899511317225567) got a Three.js Minecraft at medium effort. In comparisons against Sol 6.1 Ultra and "1017-Max," he says Fable wins on taste and prowess ([video](https://x.com/teortaxesTex/status/2119908652928934237)). Quoting 1017's self-reflection, Fable "preserved object identity" with simpler, conservative solutions, "much like Opus: not trying to be clever" ([@teortaxesTex](https://x.com/teortaxesTex/status/2119936636523860367)).
- **Generative art.** [@jkeatn](https://x.com/jkeatn/status/2119874007382299087) had Fable invent artist personas, each a 1–2k-line Python program driving a simulated bristle and paint engine over thousands of strokes. No image model is involved and nothing is pre-traced.
- **Hardware.** [@pronounced_kyle](https://x.com/pronounced_kyle/status/2119883641849520421) had it design a handheld pet, a voice-assistant hub and a USB speakerphone (electronics, enclosures and firmware) with Diode and 3D printers.
- **Pretraining read.** [@teortaxesTex](https://x.com/teortaxesTex/status/2119887127832134085) called it "Fable Moment 2.0. Anthropic really has physics of pretraining." He judges Astra "a hair better" at ambitious conceptual and mathematical thinking ([@teortaxesTex](https://x.com/teortaxesTex/status/2119947304757084633)).
- **Factory's observations** ([@factory](https://x.com/factory/status/2119954094135943312)): works through diagnostics, preserves precision in financial and legacy migrations, and gets hard work done at lower effort.
- **Other reactions:**
  - [@theo](https://x.com/theo/status/2119952323615281408): "big model smell," and AA's 5-point lead is "mogging" ([@theo](https://x.com/theo/status/2119946060361818578)).
  - Brief enthusiasm from [@zachtratar](https://x.com/zachtratar/status/2119901787826344369), [@Aurokinn](https://x.com/Aurokinn/status/2119926807248134166) and [@AravSrinivas](https://x.com/AravSrinivas/status/2119876510647562506).
  - [@swyx](https://x.com/swyx/status/2119935240202539147) on the FLT demo, with "something bigger" planned for AIE.


## Practitioner impressions: skeptical or mixed


- **[@theo](https://x.com/theo/status/2119926277801144695)** built a "Fishslop" run in 76 min for about $42, "not significantly better than Opus 5.5 (esp. for the price)." His Opus version felt more playable with better controls and performance, and he sees the two as complementary ([@theo](https://x.com/theo/status/2119929646527717822)). Other notes from him:
  - Fable is "making a lot of stupid mistakes I would not have expected from Opus" ([@theo](https://x.com/theo/status/2119988540708896897)).
  - The Claude Code desktop UX is "horrible": adding a project and enabling bypass permissions took digging ([@theo](https://x.com/theo/status/2119873100745768974)).
  - He was immediately torn between Opus and Fable ([@theo](https://x.com/theo/status/2119881447800770962)).
  - He noted his own desensitization to the OS demo ([@theo](https://x.com/theo/status/2120003663926358152)) and a record em-dash count in the announcement ([@theo](https://x.com/theo/status/2120005201948287148)).
- **[@omarsar0](https://x.com/omarsar0/status/2119892232446845225)** says early-tester posts are "just crappy games and the same old stuff." Models are now bottlenecked by user ambition and environment (memory, tools, sandboxes). He would test test-time compute methods on scientific discovery, and thinks Opus 5.5 is good enough for most other work.
- **Weak spots and demo tone.** Human animation is weak, per [@_sholtodouglas](https://x.com/_sholtodouglas/status/2119936163427631350) and [@andrew_n_carr](https://x.com/andrew_n_carr/status/2119935805368254623); [@alexalbert__](https://x.com/alexalbert__/status/2119947045595398454) said Anthropic is "working on this." [@dbreunig](https://x.com/dbreunig/status/2119939056318677495) criticized the "cloying Hallmark tone" of model-made demos, which he dubbed "rewardcore" ([@dbreunig](https://x.com/dbreunig/status/2119962950027071828)).
- **Quips.** [@lateinteraction](https://x.com/lateinteraction/status/2119875717978591331) suspects thousands are testing one particular "capability" right now. [@johnowhitaker](https://x.com/johnowhitaker/status/2119977200653131857) joked productivity will tank once researchers discover it makes fun games.


## Competitive context


- **Google's timing.** [@kimmonismus](https://x.com/kimmonismus/status/2119899579533373695) argues Google's slow release cadence hurts it, since Gemini 4 Carbon leads charts but [was never actually released](https://x.com/kylebrussell/status/2119890464341930409).
- **OpenAI.** Several expect an OpenAI response "tomorrow" ([@kimmonismus](https://x.com/kimmonismus/status/2119919963758555316), [@dejavucoder](https://x.com/dejavucoder/status/2120008148144779626), [@nicdunz](https://x.com/nicdunz/status/2119942018239779121)).
- **Baseline usage.** [@theo](https://x.com/theo/status/2119970535056846996) shows Opus 5.5's near-total dominance over Codex in T3 Code usage. That is the incumbent Fable has to displace.

**Other Model Launches**

- **Muse Ember**: Muse Ember launches on Google Cloud alongside Muse Wonder and Muse Spark 1.3 ([@alexandr_wang](https://x.com/alexandr_wang/status/2119911348360663299)).
  - **Stealth testing**: The anonymous "crouton-alpha" model was an early Muse Ember checkpoint. It processed more than 5T tokens during testing, and issues found there were fixed before launch ([reveal](https://x.com/alexandr_wang/status/2120000217827475623)).
  - **Throughput**: One user reports about 400 tok/s ([@andrew_n_carr](https://x.com/andrew_n_carr/status/2120017193752785260)).
- **Seedream 5.5 (ByteDance)**: The image model focuses on more realistic output and editing stability across 100 successive edit rounds ([@ByteDanceSeed_](https://x.com/ByteDanceSeed_/status/2119720475068920181)).
  - **Editing**: Unedited regions are preserved at the pixel level, with reduced detail loss across iterations.
  - **Capabilities**: The model natively outputs RGBA for transparent and semi-transparent assets. It also improves lighting, materials, skin texture and small-text rendering.
  - **Availability**: It is coming to Dreamina, CapCut and BytePlus.
- **Upcoming open-weights release**: [@cHHillee](https://x.com/cHHillee/status/2120006034777608533) announced an upcoming open-weights release and is offering Tinker credits for evals, demos or post-training projects ahead of it. Applicants should describe what they plan to do ([note](https://x.com/cHHillee/status/2120025642251210983)).

**Research and Systems**

- **s3 learned swarm coordination**: The paper replaces developer-written multi-agent plans with cooperative RL, letting agents learn when to work alone, share findings or read teammates' work, with no predefined roles ([@omarsar0](https://x.com/omarsar0/status/2119792446242718170), [arXiv](https://arxiv.org/abs/2611.08615)).
  - **Claimed results**: Under matched training and inference budgets, swarms run at twice their training size solve over 25x more coordination problems than the strongest baseline. In parallel exploration they repeat almost 50% less work and beat best-of-N by up to 9.4 points.
  - **Emergent roles**: Complementary roles emerge during training and transfer to larger teams.
- **CPython formalized in Lean by Opus 5.5**: Opus 5.5 produced 613,318 lines of Lean that mirror Python's internals line by line ([@broyojo42](https://x.com/broyojo42/status/2119846041214206286)).
  - **Validation**: The model is tested against the real interpreter and checked by Lean.
  - **Bugs found**: The project surfaced real CPython bugs, including crashes triggered by plain Python code.
- **vLLM × LibTorch Stable ABI**: New PyTorch releases no longer force a vLLM rebuild ([@vllm_project](https://x.com/vllm_project/status/2119977386553073736)). Users can upgrade sooner, distributors rebuild less often, and developers can test torch changes without recompiling vLLM.
- **HBM5 (Samsung, Hot Chips 2026)**: HBM5 keeps the same 2048 pins as HBM4E but raises pin speed from 14 Gbps to more than 28 Gbps ([SemiAnalysis](https://x.com/SemiAnalysis_/status/2120025695283785966)).
  - **Architecture**: DRAM banks, TSVs and the PHY are doubled or upgraded to keep the faster pins fed. The PHY moves to the edge of the base die, shortening the path to the GPU from about 5.5mm to about 2mm ([5/6](https://x.com/SemiAnalysis_/status/2120025701776494794)).
  - **Thermal cost**: The PHY can draw more than 4x the power of HBM4E's, so suppliers are adding a thermal "chimney" above it ([6/6](https://x.com/SemiAnalysis_/status/2120025703294894487)).

**Developer Tools, Evals Practice and Adoption**

- **Claude dominance in T3 Code**: [@theo](https://x.com/theo/status/2119970535056846996) charted the Codex vs Claude split in T3 Code over time and describes Opus 5.5 as dominating usage.
  - **Workaround for Grok Bot limits**: He suggests routing Grok Bot tasks to T3 Code running Opus 5.5 on the $100 Claude plan ([tip](https://x.com/theo/status/2119994608516276256)).
  - **MCP pairing**: T3 Connect now authorizes MCP links without copy-pasted pairing codes ([@jullerino](https://x.com/jullerino/status/2119992038888554520)).
- **Own your evals**: [@omarsar0](https://x.com/omarsar0/status/2119795295630324042) argues that production model selection, and eventually RSI-driven optimization, depends on in-house evals rather than public benchmarks. DAIR is launching an effort to help companies build these evals.
  - **Eval-driven development**: LangChain frames EDD as "TDD for agents," measuring every change against a fixed eval set ([@vitalflux](https://x.com/vitalflux/status/2120011436684627975)).
- **Small-model OCR caveat**: In a small experiment, Mirage, a 22 MB offline vision model, read 6/6 plain-word signs correctly. It got 0/8 on accented words, for example reading ARRÊT as ARRET and Straße as Strabe, and 0/4 on Hindi and Telugu boards. The authors explicitly say this is not a benchmark ([@LearnOpenCV](https://x.com/LearnOpenCV/status/2119988545288839509)).

**Industry and Policy**

- **Paris Call for AI Safety and Security**: President Macron hosted a meeting of about 30 participants on November 9, including heads of state such as Japan's Prime Minister Takaichi, lab leaders from OpenAI and Anthropic, and Yoshua Bengio ([@SakanaAILabs](https://x.com/SakanaAILabs/status/2119931036746621372)).
  - **Signatories and scope**: The French government's resulting Paris Call was signed by Sakana, OpenAI, Anthropic, Microsoft, Mistral and Hugging Face, among others. It covers frontier-AI evaluation and risk management, transparency, information sharing and early-warning frameworks ([details](https://x.com/SakanaAILabs/status/2119981508857598158)).
- **Anthropic IPO framing (opinion)**: Several commentators read the Fable 5.2 launch against Anthropic's pending IPO. SemiAnalysis questions whether a stronger release will come before the listing ([7/7](https://x.com/SemiAnalysis_/status/2119961456716189764)). Others doubt the model is "good enough to save the IPO" ([@andersonbcdefg](https://x.com/andersonbcdefg/status/2119970984229048402)).

**Top tweets (by engagement)**

- [OpenAI: Control Room halts research workloads within 35s](https://x.com/OpenAI/status/2119913172039868579) — 4,125
- [@theo: Claude Ultra yields ~$15K of Opus 5.5 vs ~$3K of Fable 5.2](https://x.com/theo/status/2119960608816988596) — 2,810
- [@broyojo42: Opus 5.5 formalizes CPython in 613K lines of Lean](https://x.com/broyojo42/status/2119846041214206286) — 1,673
- [@ByteDanceSeed_: Seedream 5.5 launch](https://x.com/ByteDanceSeed_/status/2119720475068920181) — 1,600
- [@theo: Codex vs Claude usage split in T3 Code](https://x.com/theo/status/2119970535056846996) — 1,137
- [@alexandr_wang: Muse Ember, Wonder and Spark 1.3 on Google Cloud](https://x.com/alexandr_wang/status/2119911348360663299) — 1,085
- [@AndrewCurran_: system card implies Anthropic's top model stays internal](https://x.com/AndrewCurran_/status/2119930955888746627) — 840
- [@SemiAnalysis_: Fable 5.2 tops GDPx at 16.4%](https://x.com/SemiAnalysis_/status/2119961447199289640) — 635


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen 3.8 Inference Acceleration and Benchmarks

  - **[tencent/Qwen3.8-27B-DSpine-b16 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1xr2aaa/tencentqwen3827bdspineb16_hugging_face/)** (Activity: 342): ****Tencent** released [`tencent/Qwen3.8-27B-DSpine-b16`](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16), a **5-layer DSpine speculative draft model** for **Qwen3.8-27B** with block size `16` and an `8K` sliding window. DSpine drafts tokens in parallel by injecting predicted predecessor features across depth to preserve causal dependencies, and Tencent reports SGLang `0.5.22` evals with BF16, FlashAttention 3, dynamic verification, and ReplaySSM showing roughly **`2.9–3.8×` throughput vs autoregressive decoding** and better accepted draft length / agent throughput than DFlash2. A commenter’s early **MegaCapybara** engine test on ~Q5 weights reported *no-think coding* speed rising from `420 t/s` with DFlash2 to `800 t/s` with DSpine, but *reasoning* falling from `320 t/s` to `280 t/s`, attributed to lower confidence in the first `1–2` draft tokens.** Commenters framed the release as well-timed before an expected Qwen4, with one estimating a `15–20%` improvement over DFlash2 and expecting further gains after additional training / engine-side optimization. A related Reddit paper thread, [“DSpine - Draft in Parallel, Condition Through Depth”](https://www.reddit.com/r/LocalLLaMA/comments/1xq9hf6/paper_dspine_draft_in_parallel_condition_through/), was cited for deeper technical discussion.

    - Early testing in **MegaCapybara** on an RTX `5090` with medium-weight quantization around `Q5` reported a large no-thinking coding throughput increase versus DFlash2: `800 tok/s` for **DSpine** vs `420 tok/s` for **DFlash2**. However, reasoning workloads regressed slightly, with **DSpine** at `280 tok/s` vs **DFlash2** at `320 tok/s`, attributed to the first `1–2` tokens being less confident and potentially recoverable with additional training. See the referenced engine: [MegaCapybara](https://github.com/perkel666/MegaCapybara).
    - A separate tester implementing **LumaBrowser** support found the opposite result in `llama.cpp`: **DSpine** was *not faster than DFlash2 in any tested scenario* and also required more memory. They noted the result may differ in **SGLang**, but for their `llama.cpp` benchmarks they found “zero upsides” compared with DFlash2.
    - A linked technical discussion points to Tencent’s **DSpine** paper/thread, which frames the method as *“Draft in Parallel, Condition Through Depth”* using **Adjacent Causal Injection** for speculative decoding. The related Reddit thread is here: [[Paper] DSpine - Draft in Parallel, Condition Through Depth](https://www.reddit.com/r/LocalLLaMA/comments/1xq9hf6/paper_dspine_draft_in_parallel_condition_through/).

  - **[Joined the 4x R9700 Club | Initial Benchmarks for QFN 3.8 Below](https://www.reddit.com/r/LocalLLaMA/comments/1xrcia4/joined_the_4x_r9700_club_initial_benchmarks_for/)** (Activity: 484): **The image is a **technical hardware build photo**, not a meme: it shows an open Phanteks Enthoo Pro 2 Server chassis populated with **4× Radeon AI PRO R9700 GPUs**—three Gigabyte cards and one XFX/black card—matching the post’s “4x R9700 Club” title ([image](https://i.redd.it/m1hd0vwe4h0i1.jpeg)). The build is used as a local multi-user inference server for **Qwen3.8-Flash-Next** under Radiance/ROCm, with the OP reporting `128GB` total VRAM, `tp4`, FP8 KV cache, and long-context concurrency results such as ~`6,800 tok/s` aggregate prefill and ~`80–225 tok/s` per-user follow-up generation depending on concurrency/context.** Commenters generally viewed the setup as valuable less for today’s QFN 3.8 performance and more as a future-proof `128GB VRAM` ROCm box for upcoming local models. One technical commenter argued that a raw `27B` dense model in full BF16/BF16 KV may give better software-engineering-agent quality than Flash-Next quants, albeit at roughly half the reported speed.

    - A commenter with a similar **Threadripper Pro + 4× R9700 + 128GB DDR4** setup argues that running a raw `27B` dense model at full `BF16` weights with `BF16` KV cache can be more compelling than using **deadcode's Flash-Next quantizations**. They report that although full BF16 roughly halves the posted speeds, using **radiance** still yields about `70–100 tokens/s`, with noticeably better reliability for software-engineering agent tasks that require architectural or algorithmic judgment.
    - Another technical theme is that the value of the 4× R9700 build is less about today’s **QFN 3.8** performance and more about the combination of `128GB` total VRAM and a relatively mature **ROCm** stack for future local models. One commenter frames the system as a forward-looking platform likely to handle next-generation local models such as a hypothetical **Qwen4**-class release.
    - One commenter asks how Linux GPU power limiting is being done and what benefits it provides, implying interest in tuning multi-GPU Radeon inference rigs for power, thermals, or efficiency. No implementation details are provided in the visible comments, but the question is directly relevant to operating a 4× GPU ROCm system under sustained inference load.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo


### 1. Claude Fable/Mythos 5.2 Launch and Rollout Issues

  - **[Introducing Claude Fable 5.2 and Claude Mythos 5.2](https://www.reddit.com/r/ClaudeCode/comments/1xs0i9s/introducing_claude_fable_52_and_claude_mythos_52/)** (Activity: 1530): **The post announces **Claude Fable 5.2** and **Claude Mythos 5.2**, but the provided content includes no technical release notes: no model cards, benchmark deltas, context length, pricing, API identifiers, safety changes, or migration guidance. For an expert reader, the actionable information is limited to the apparent availability/naming of two new Claude-branded model variants at version `5.2`.** Top comments are mostly meta-discussion: users note that complaints about Claude being “nerfed” often precede a release, ask for **Opus 5.5** to return, and criticize the `5.2` naming as inconsistent with a hoped-for or prior `5.5` versioning scheme.

    - Several commenters focused on Anthropic’s model-line/versioning behavior, noting that community reports of *“Claude is nerfed”* often precede a new release. Others referenced prior/removal expectations around **Opus 5.5 / 5.6**, suggesting users are tracking regressions, availability changes, and naming consistency across Claude model variants.
    - A technically relevant concern was raised about **Claude Fable 5.2 safeguards**: the release note says it will *“block most cybersecurity-related tasks.”* One commenter objected that this could significantly limit legitimate security workflows, implying reduced usefulness for defensive analysis, vulnerability research, or CTF-style tasks compared with less restricted models.

  - **[Fable 5.2 just landed.](https://www.reddit.com/r/ClaudeAI/comments/1xs0dct/fable_52_just_landed/)** (Activity: 2187): ****Anthropic** appears to have shipped **Fable 5.2** alongside a delayed announcement page, [`claude-fable-and-mythos-5-2`](https://www.anthropic.com/claude-fable-and-mythos-5-2). Commenters focused on two implementation details: a benchmark/cost graph where moving from **Fable Xhigh** to **Max** seemingly increases thinking time and cost while reducing score, and a quota change granting `1–3` extra **Fable-only resets** based on subscription tenure rather than resetting overall weekly usage.** The main skepticism is around price/performance—users interpret the graph as showing a worse frontier at higher compute—and around quota semantics, with one commenter saying Fable-only resets are of limited value if the global weekly cap is already exhausted. Another commenter speculates that 5.2 may mainly tighten guardrails, specifically around use of the **REA repo**.

    - A commenter questioned a benchmark/quality graph where moving from **Fable Xhigh** to **Max** appears to increase deliberation time and cost while reducing measured performance, implying either non-monotonic scaling, benchmark noise, or a confusing visualization. The technical concern is that a higher-compute mode should normally trade latency/cost for higher accuracy, so this result may indicate overthinking/regression on the evaluated task set.
    - Several users discussed the new **Fable reset** policy: plans with Fable models reportedly receive `1–3` Fable-specific resets based on subscription duration, but the reset appears limited to Fable usage rather than general weekly usage. The key implementation concern is that a Fable-only reset may be ineffective if the account’s broader weekly quota is already exhausted, suggesting separate quota buckets or gating logic.
    - One commenter speculated that the main change from **Fable 5.1** to **Fable 5.2** may be stricter guardrails, specifically around preventing use of the **REA repo**. While not benchmark-backed, the technical theme is a possible safety-policy regression affecting repository-level workflows or code access rather than raw model capability.

  - **[Claude Fable 5.2 built a playable Gears of War clone in Unreal Engine from a 3-line prompt, complete with enemy AI, cover mechanics and active reloading. After testing it with over 100 billion tokens, Matthew Berman calls it “by far the best model” he’s ever used.](https://www.reddit.com/r/singularity/comments/1xs1y29/claude_fable_52_built_a_playable_gears_of_war/)** (Activity: 1635): **Matthew Berman’s review claims **Claude Fable 5.2** generated a playable *Gears of War*-style Unreal Engine project from a 3-line prompt, including **enemy AI, cover mechanics, and active reload**, after multi-day iteration totaling roughly `8.5B` tokens and an estimated `$15k` in API spend; across his broader testing he says he used `>100B` tokens and called it *“by far the best model”* he has used ([full review](https://forwardfuture.com/newsletter/originals/i-put-fable-5-2-to-work-5-real-world-field-tests-and-what-i-learned)). He also reports Fable 5.2 recreated an Excel-like spreadsheet app, built complex 3D environments, and helped create Photoshop/Premiere-like internal production tools.** Top comments were skeptical: one user argued similar results might be achievable with guided use of Opus-tier models on a consumer plan, and questioned whether prompts to *“clone this game”* violate model/platform policy. Another commenter dismissed the benchmark framing by noting the original 3-line prompt apparently contained proofreading errors.

    - A substantive data point cited **Matthew Berman’s Fable 5.2 review**: the Gears of War-style Unreal project reportedly took *several days* and consumed roughly `8.5 billion tokens`, with estimated API costs around `$15,000`. The same review claims Fable 5.2 also generated an Excel-like spreadsheet app, complex 3D environments, and Photoshop/Premiere-style tools reportedly used in production: https://forwardfuture.com/newsletter/originals/i-put-fable-5-2-to-work-5-real-world-field-tests-and-what-i-learned
    - One commenter questioned whether this result is meaningfully unique to **Claude Fable 5.2**, arguing that a similar prototype might be achievable with **Opus 5.5** on a `$20` plan if guided interactively. They also raised a policy/IP concern around prompts like *“clone this game”*, implying that benchmark-style demos based on recognizable commercial games may be difficult to evaluate or reproduce under model usage restrictions.
    - Another technically relevant skepticism focused on the gap between AI generating a playable prototype and producing a commercially successful game. The commenter argued that *“making a game”* may already be achievable with current AI tooling, but *“making a good game”* remains a separate product/design problem involving quality, originality, and player appeal rather than just code or asset generation.

  - **[The nerf is real. Thank you u/WuM1ha1nho!!!!](https://www.reddit.com/r/ClaudeCode/comments/1xr9x5y/the_nerf_is_real_thank_you_uwum1ha1nho/)** (Activity: 3948): **The post claims **Anthropic is running per-session A/B experiments in Claude Code** and that perceived model degradation/“nerfing” can be correlated with experiment IDs stored in local Claude artifacts: current config in `.claude/.claude.json` and historical session transcripts under paths like `/home/<user>/<claude-dir>/projects/*/*.jsonl`. The author says Claude was used to correlate **hex experiment IDs** in those JSONL transcripts with degraded thread behavior, referencing a prior Reddit thread on experiment IDs ([source](https://www.reddit.com/r/ClaudeCode/comments/1xr66pd/i_dont_think_opus_55_was_nerfed_theyre_just/)); commenters report seeing named experiments such as `tengu_mossy_gray_experiment` / `tengu_keen_pudding_experiment`, and IDs like `604` / `b10`, with one user claiming both **Opus** and **Fable** burned tokens heavily and performed much worse.** Commenters split between “A/B testing is expected for a company like Anthropic” and “affected users deserve a reset/compensation.” One proposed making experiments opt-in, compensated by an extra weekly reset and early access to future model variants.

    - Several commenters report evidence of **Anthropic A/B testing** via visible experiment labels such as `tengu_mossy_gray_experiment`, `tengu_keen_pudding_experiment`, and IDs like `604` / `b10`. One technical interpretation is that the tested condition may not be a simple quality downgrade, but a cost/latency/reasoning-budget experiment such as *“achieve the goal with 10% less reasoning”*, which could affect perceived intelligence and token usage.
    - One user claims **Opus/Fable variants** behaved substantially worse during the experiment window, specifically *burning through tokens* and producing degraded outputs. The complaint suggests the regression may involve reasoning-effort allocation or context handling rather than just answer accuracy, though no benchmark traces or reproducible logs were provided.
    - A commenter hypothesizes Anthropic is testing a newer **Opus/Fable** model variant distinguishable by factual guessing behavior: vanilla `opus-5.5` allegedly refuses or guesses **George Russell** for “Who won the 2026 Monaco Grand Prix?”, while the new model answers **Kimi Antonelli**. They further speculate that switching a session back from the new model to `opus-5.5` may cause degradation because the older model cannot access the newer model’s hidden `thinking` blocks, leaving it partially unable to interpret prior context.


### 2. Agentic Coding and Tool-Use Workflows

  - **[I built an MCP with Claude Code that lets Claude play PC games in the background. Here it is driving, flying a heli and walking in GTA on its own](https://www.reddit.com/r/ClaudeAI/comments/1xr3gbs/i_built_an_mcp_with_claude_code_that_lets_claude/)** (Activity: 1200): **The author released an MIT-licensed Windows-only MCP server for game automation ([GitLab mirror](https://gitlab.com/corruptedang3l-group/llm-game-automation-mcp); original [GitHub](https://github.com/CorruptedAng3l/llm-game-automation-mcp) was temporarily suspended) that lets **Claude** control single-player/offline PC games in the background via an injected game-side DLL for input/screen capture plus a TypeScript MCP layer exposing ~`100` tools. The system supports per-game data profiles—minimap bounds, camera behavior, waypoint detection—so agents can iteratively play, diagnose failures, patch code/profiles, and benchmark before/after on fixed routes; demos include GTA IV/V driving/helicopter navigation, Detroit: Become Human choices, TABS, Halo Wars 2, Phantom Rose, and Cities: Skylines. Current limitations include poor indoor navigation without minimaps, early-stage shooter aiming, Windows DLL dependency, and incompatibility with anti-cheat/multiplayer games.**

    - A commenter asked for the **token burn/cost** of running Claude Code as an MCP-controlled game agent, which is the main technical concern raised in the thread. For an always-on game-control loop, costs would depend heavily on screenshot/frame frequency, context size, tool-call cadence, and whether the agent receives raw images, structured state, or both.

  - **[DeepSeek V4.1 Flash used REA to reverse-engineer a 1984 DOS game and rebuild it for Chrome, for $0.68](https://www.reddit.com/r/DeepSeek/comments/1xr3g3i/deepseek_v41_flash_used_rea_to_reverseengineer_a/)** (Activity: 414): ****DeepSeek V4.1 Flash** was given a single prompt plus the original `43 KB` 1984 *Sopwith* DOS executable, with **REA** as an MCP server running **Ghidra** under the hood ([REA](https://github.com/morluto/rea), [AgenC Core](https://github.com/tetsuo-ai/agenc-core)). Over ~`1 hour` in a no-internet sandbox, it decompiled the binary, inferred gameplay systems, wrote a Python asset/table extractor, rebuilt the game as `320x200` HTML5 Canvas with PC-speaker-style sound, and used DOSBox + headless Chrome frame comparisons to validate the port; reported usage was `374` model calls, `624` tool calls, `145M` input tokens with `99.7%` cache hits, `308K` output tokens, costing `$0.68` off-peak / `$1.37` peak. The full prompt was shared here: [PROMPT.md](https://github.com/tetsuo-ai/agenc-core/blob/showcase/sopwith-rea-prompt/showcase/sopwith-rea/PROMPT.md).** Comments were sparse but framed this as a major acceleration of old-school reverse engineering workflows—one commenter compared it to manually ripping and modifying a 6502 Atari cartridge and estimated AI could do it in *“1/1000 of the time.”* Another joked that `$0.68` was still “pretty expensive,” reflecting the expectation that agentic reverse-engineering costs will continue to collapse.

    - A commenter linked the **full REA prompt** used for the Sopwith reverse-engineering/rebuild experiment, hosted in the `tetsuo-ai/agenc-core` repo: [PROMPT.md](https://github.com/tetsuo-ai/agenc-core/blob/showcase/sopwith-rea-prompt/showcase/sopwith-rea/PROMPT.md). This is the most concrete implementation artifact in the thread for technical readers wanting to inspect task framing, agent instructions, and workflow assumptions.
    - One commenter compared the result to historical manual reverse engineering work: they had previously ripped an **Atari cartridge** and modified an app without access to the original **6502 source**. They speculated that modern AI-assisted reverse engineering could complete comparable work in roughly `1/1000` of the time, highlighting the productivity delta for binary analysis and source reconstruction tasks.
    - Another user reported doing a similar AI-assisted rebuild of the 1984 CGA game **Alley Cat**, converting it to **VGA**, reportedly using **Terra Very High** for under `$1`. While details were sparse, it suggests the workflow may generalize beyond Sopwith to other DOS-era CGA titles with low inference cost.

  - **[Qwen 3.8 27B wrote 2,860 lines and made Fable 5.2 5x cheaper and 2x faster](https://www.reddit.com/r/Qwen_AI/comments/1xs70ch/qwen_38_27b_wrote_2860_lines_and_made_fable_52_5x/)** (Activity: 417): **A developer reports using **Atomic Agent `0.6.13`** ([project](https://atomicagent.io/)) to orchestrate **Fable 5.2** as a planner/reviewer over **three local Qwen 3.8 27B Q4_K_XL builders** via `llama.cpp` on a **MacBook Pro M5 Max 64 GB**, generating `2,860` lines of JS/HTML for a Three.js “3D pop-up book” short. In a single-run comparison, the orchestrated setup completed in **`11:32` for `$2.98` cloud cost** versus **Fable 5.2 alone at `21:27` for `$15.77`**, with the claimed win coming from parallel local retries being “free” and Fable never directly editing files. Qwen’s first pass was unstable (`5/8` tasks completed; one timeout, one step exhaustion, one escalated), but after Fable narrowed task briefs it completed `5/5`; the author explicitly notes this is **one data point, not a benchmark**.** Top comments focused less on the demo output and more on methodology: users asked for the exact Fable↔Qwen coordination prompt, whether Fable directly calls Qwen, and whether Qwen retains state between calls. Others questioned how **Claude Sonnet/Haiku** would compare on quality/cost, while one commenter was primarily impressed by the perceived Qwen quality delta.

    - Several commenters focused on reproducibility and orchestration details, asking how **Fable 5.2** coordinated with **Qwen 3.8 27B**: whether Fable was directly calling Qwen as a sub-agent/tool, how task state was passed between calls, and whether Qwen retained any memory or relied entirely on prompt/context injection.
    - There was technical skepticism about the benchmark methodology and cost/performance claims, with commenters requesting the exact prompt to rerun the comparison against **Fable 5.2** alone and arguing the result may be biased without controlled prompts, identical task specs, and comparable evaluation criteria.
    - One commenter asked how **Claude Sonnet** or **Haiku** would perform in the same setup, implying the claimed `5x` cost reduction and `2x` speedup for Qwen-assisted Fable should be compared against Anthropic models with similar coding-agent workflows and pricing assumptions.

  - **[If a rule can be checked by a machine, it shouldn't be a line in AGENTS.md](https://www.reddit.com/r/ChatGPTCoding/comments/1xro33q/if_a_rule_can_be_checked_by_a_machine_it_shouldnt/)** (Activity: 40): **The post argues that `AGENTS.md` should not contain machine-checkable constraints like “never force push,” “don’t edit generated clients,” or “always run tests,” because LLM agents may lose those instructions after many tool calls and the rules fail silently. Instead, enforce deterministic policies via **pre-commit hooks, CI checks, linters, permission prompts, or filesystem protections**, reserving `AGENTS.md` for non-scriptable project context and judgment calls such as architectural boundaries, escalation criteria, and which test command matters.** The central opinion is that prose instructions are high-token, low-reliability enforcement mechanisms; durable rules need executable guardrails. No top comments were provided, so no additional comment-side debate can be summarized.



### 3. OpenRouter CSAM Screening Policy Change

  - **[OpenRouter may scan and report a certain type of NSFL](https://www.reddit.com/r/SillyTavernAI/comments/1xs6vi1/openrouter_may_scan_and_report_a_certain_type_of/)** (Activity: 414): ****OpenRouter** appears to have changed policy around detecting and reporting a specific illegal NSFL category, with the described mechanism seemingly limited to **hash-matching images/videos** rather than scanning text prompts or completions. The technical concern is whether “risky roleplay” text could later be subjected to classifier-based moderation/reporting, which would require high-context reasoning to avoid false positives in nuanced fictional NSFW narratives.** Commenters were mainly concerned about false positives if text classifiers are used, arguing that shallow moderation models may fail on context, character age/backstory, or fictional framing. Others noted that media hash matching is a narrower and more conventional approach, and that mass-scanning text would likely alienate more than just roleplay users.

    - Commenters raised concerns that **text scanning would require nuanced contextual reasoning** to avoid false positives, especially in long-form fiction where adult scenes may reference non-sexual backstory involving younger versions of characters. They argued that a lightweight classifier could misclassify ordinary erotic-romance-style context unless OpenRouter used a more capable reasoning model, which would add cost and latency.
    - Several users distinguished **hash-matching for images/video** from broader semantic scanning, noting that if OpenRouter is only matching known illegal media hashes, the scope is narrower than mass inspection of prompts and completions. One commenter argued that OpenRouter would risk losing far more than roleplay users if it signaled broad scanning of all text inputs.
    - A technical analogy was made to **Google Photos automated abuse-detection systems**, where users have reported false account bans and law-enforcement reports for benign family photos. The concern is that similar automated image classifiers or hash-based pipelines could produce severe consequences from false positives if appeal/review mechanisms are weak.

  - **[If you still have some credits on OR, be careful. If you wanted to top up again, use NanoGPT Payg instead.](https://www.reddit.com/r/SillyTavernAI/comments/1xs7hd2/if_you_still_have_some_credits_on_or_be_careful/)** (Activity: 341): **The image is a dark-themed screenshot of an **OpenRouter policy update** stating that **CSAM screening may override “Zero Data Retention”**: prompts/responses are generally not retained after processing, but may be stored when needed for review, reporting, or legal preservation. In the post’s context, the author argues this weakens OpenRouter’s ZDR guarantees after Stripe-related compliance pressure and recommends using **NanoGPT Payg** instead, citing lower fees and working cache behavior; image: [i.redd.it/wdw930pv9o0i1.jpeg](https://i.redd.it/wdw930pv9o0i1.jpeg).** Commenters debated whether “zero-data-retention” is now misleading because it has policy exceptions, with one saying the “zero” has *“a lot of asterisks.”* Others supported migrating to NanoGPT Payg for cheaper top-ups and model cache support, while expressing distrust that CSAM scanning will remain limited to images/video rather than expand to text.

    - Users raised concerns that **OpenRouter’s “zero-data-retention” policy may have meaningful exceptions** after Stripe’s involvement, implying that compliance or safety scanning could undermine expectations around prompt privacy. One commenter specifically argued that the “zero” in ZDR has “a lot of asterisks,” reflecting concern that retained or reviewable data may exist under abuse-detection workflows.
    - Several commenters compared **NanoGPT PAYG** favorably against OpenRouter on cost and behavior, claiming it is cheaper because there are “no extra costs” when adding balance. One user also reported that **prompt/cache support works across all models they tested**, which is relevant for users optimizing repeated-context workloads and token spend.
    - A technical risk noted around automated moderation was **false-positive escalation in model/provider filters**, with Gemini cited as already hard-refusing NSFW roleplay scenarios involving a household with minors present, even when not directly sexualized. Commenters extrapolated that broader CSAM-detection or payment-processor-driven scanning could cause more prompts to be flagged for review, especially in edge-case text-generation scenarios.



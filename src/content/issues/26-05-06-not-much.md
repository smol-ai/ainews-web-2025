---
id: MjAyNS0x
title: not much happened today
date: '2026-05-06T05:44:39.731046Z'
description: >-
  **Anthropic** announced a new **SpaceX compute partnership** to significantly
  increase capacity for **Claude** products, doubling **Claude Code's 5-hour
  rate limits** for Pro, Max, Team, and Enterprise users, removing peak-hour
  limit reductions, and substantially increasing API rate limits for **Opus**
  models. The deal grants Anthropic access to **Colossus 1** via **SpaceXAI**,
  with **Claude inference** expected to ramp up on Colossus soon. Anthropic also
  hosted a **"Code with Claude"** event featuring updates on Claude Code,
  GitHub-scale usage, and managed agents. Discussions highlighted compute
  bottlenecks, user reactions to limit changes, debates on managed-agent
  features, and ongoing safety/governance discourse around AGI trustworthiness.
companies:
  - anthropic
  - spacex
  - x-ai
models:
  - claude
  - claude-code
  - opus
  - colossus-1
topics:
  - compute
  - rate-limiting
  - agent-platforms
  - inference
  - api
  - managed-agents
  - safety
  - governance
  - event
people:
  - nottombrown
  - _aidan_clark_
  - kipperrii
  - theamolavasare
  - alexalbert__
---


**a quiet day.**

> AI News for 5/5/2026-5/6/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Top Story: Anthropic and Claude announcements/commentary**


## What happened

**Anthropic had a dense news cycle centered on compute, Claude Code limits, and agent platform direction.** Officially, Anthropic announced a new compute partnership with SpaceX that will “substantially increase” capacity and immediately translate into higher limits for Claude products: [@claudeai](https://x.com/claudeai/status/2052060691893227611) said the deal boosts compute enough to raise usage limits, followed by specifics from [@claudeai](https://x.com/claudeai/status/2052060693269008586): **Claude Code’s 5-hour rate limits are doubled for Pro, Max, Team, and seat-based Enterprise; peak-hours limit reductions are removed for Pro and Max; Opus API rate limits are substantially increased**. xAI framed the deal as Anthropic getting access to **Colossus 1** via SpaceXAI for “additional capacity for Claude” [@xai](https://x.com/xai/status/2052060350770515978), while Anthropic CTO Tom Brown added that **Claude inference would be ramped up on Colossus “in the next few days”** [@nottombrown](https://x.com/nottombrown/status/2052062566126649448). The company also ran its **“Code with Claude”** event, with a livestreamed keynote and sessions on Claude Code, GitHub-scale usage, and managed agents [@ClaudeDevs](https://x.com/ClaudeDevs/status/2052055459272761661), prompting substantial real-time commentary from developers and observers [@simonw](https://x.com/simonw/status/2052055655230706032), [@latentspacepod](https://x.com/latentspacepod/status/2052062150332710942). Around this, discourse branched into four themes: **(1) compute bottlenecks were more severe than many assumed, reportedly due to unexpected usage growth; (2) users welcomed the 5-hour limit increase but questioned unchanged weekly limits; (3) people debated whether Anthropic’s new managed-agent features like memory/“Dreaming” and rubrics/“Outcomes” are real product differentiation or commoditizable harness features; and (4) Anthropic’s safety/governance positioning continued to attract both praise and criticism**, including claims from critics that some Anthropic employees project “only we can be trusted with AGI,” and counterclaims from Anthropic-adjacent voices that the more common internal view is closer to “no one can be trusted with AGI” than “only us” [@_aidan_clark_](https://x.com/_aidan_clark_/status/2052089187659346047), [@kipperrii](https://x.com/kipperrii/status/2052094851991392536).


## Official facts and confirmed details

- Anthropic announced a **SpaceX compute partnership** to increase capacity [@claudeai](https://x.com/claudeai/status/2052060691893227611).
- Effective immediately, Anthropic says it is:
  1. **Doubling Claude Code’s 5-hour rate limits** for Pro, Max, Team, and seat-based Enterprise  
  2. **Removing peak-hours limit reduction** on Claude Code for Pro and Max  
  3. **Substantially increasing API rate limits for Opus models**  
  Source: [@claudeai](https://x.com/claudeai/status/2052060693269008586)
- Anthropic linked an official explainer on the higher usage limits and the SpaceX compute deal [@claudeai](https://x.com/claudeai/status/2052060696255283346).
- xAI’s announcement described the arrangement as **SpaceXAI providing Anthropic access to Colossus 1** for additional Claude capacity [@xai](https://x.com/xai/status/2052060350770515978).
- Anthropic CTO Tom Brown said **Claude inference would start ramping on Colossus within days** [@nottombrown](https://x.com/nottombrown/status/2052062566126649448).
- Anthropic product/eng lead Amol Avasare clarified that **weekly limits were not increased yet** because only a **small percentage** of users hit weekly limits, while a much larger percentage hit 5-hour limits; more changes may come as compute lands [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052064611692904639), [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052066157176426653).
- Anthropic/Claude held a **Code with Claude** event with sessions including keynote, Claude Code updates, GitHub-scale usage, and managed agents [@ClaudeDevs](https://x.com/ClaudeDevs/status/2052055459272761661).
- Anthropic’s Alex Albert promoted the event and later summarized the announcement as **“More chips, more Claude”** [@alexalbert__](https://x.com/alexalbert__/status/2052067009605861764), [@alexalbert__](https://x.com/alexalbert__/status/2052065953173872912).
- The dedicated Claude Code account reiterated the limit increase for Pro/Max/Team [@claude_code](https://x.com/claude_code/status/2052071730190123094).

## Compute details and scale claims

Several tweets added quantitative claims about the scale of the SpaceX/xAI arrangement. These are **not from Anthropic’s main announcement tweets**, but they were widely circulated:

- [@_arohan_](https://x.com/_arohan_/status/2052065871552819647) cited **“more than 300 megawatts of new capacity” and “over 220,000 NVIDIA GPUs within the month.”**
- [@scaling01](https://x.com/scaling01/status/2052068218047545501) claimed Colossus 1 includes **~150,000 H100s, 50,000 H200s, and 30,000 GB200s**.
- [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2052065017072386450) repeated the **220,000 GPU** figure and added an unverified claim that Anthropic had committed **$200B on Google TPUs**.
- [@eliebakouch](https://x.com/eliebakouch/status/2052066609896808473) interpreted the deal as Anthropic getting effectively **all of Colossus 1 capacity**, not just idle GPUs.
- Elon Musk later said SpaceXAI was comfortable leasing Colossus 1 because **xAI had already moved training to Colossus 2** [@elonmusk](https://x.com/elonmusk/status/2052069691372478511), and [@eliebakouch](https://x.com/eliebakouch/status/2052068426152132722) claimed Colossus 2 is already at **~500k Blackwells**.

These numbers are best treated as **partly official-adjacent but not fully canonized in Anthropic’s own announcement thread**. The broad factual takeaway is stronger than the exact inventory breakdown: **Anthropic secured a very large, near-term external inference capacity expansion.**

## Evidence the bottleneck was real

A recurring interpretation was that Anthropic’s constraint had genuinely been compute, not merely pricing or product design.

- [@kimmonismus](https://x.com/kimmonismus/status/2052059082886910251) asked during/after the livestream whether Anthropic was **doubling Claude Code rate limits at no extra charge**.
- [@kimmonismus](https://x.com/kimmonismus/status/2052118418174681572) later summarized remarks from a Dario/Daniela interview: **usage grew ~80x unexpectedly**, which purportedly caused the compute shortage, and the SpaceX deal is the first major attempt to address it.
- [@czajkadev](https://x.com/czajkadev/status/2052101699188248990) explicitly interpreted the update as proof that **compute was the bottleneck**.
- [@theo](https://x.com/theo/status/2052114791045668894) separately argued the industry problems are “not just money, it’s about compute,” which fits the Anthropic story even though it’s a broader point.
- [@scaling01](https://x.com/scaling01/status/2052069341609226550) generalized from this deal to a macro thesis: **frontier labs are compute constrained enough to rent datacenters from competitors.**

This is one of the strongest factual/market signals in the dataset: **Anthropic’s user-facing rate limits moved materially only after a major compute deal.**

## Product implications: Claude Code, API, and managed agents

Anthropic’s practical user impact is clear:

- **Claude Code power users get more usable burst capacity** over a 5-hour window.
- **Peak-time throttling is eased** for Pro/Max.
- **Opus API users get higher rate limits**, which matters for agent workloads and production integrations.

The event also highlighted Anthropic’s broader platform ambitions around agents. While the primary official tweets here are mostly about the event itself, commentary points to features such as:

- **Dreaming** = memory / cross-session context  
- **Outcomes** = rubrics / grading / objective tracking  
- agent orchestration / managed agents direction

Commentary:
- [@RichNwan](https://x.com/RichNwan/status/2052085746526216601) argued Anthropic is “building out their managed agents platform” with **Dreaming** and **Outcomes**, but questioned whether these are meaningfully differentiated versus open harnesses.
- [@eliebakouch](https://x.com/eliebakouch/status/2052156107313807690) saw these as **important for power users**, especially for preserving the main agent’s context window and using separate graders to manage quality/safety/reward hacking.
- [@latentspacepod](https://x.com/latentspacepod/status/2052068066167816369) quoted Anthropic speakers emphasizing **verification**, “routines are higher-order prompts,” and the idea that the remaining gap is often **deployment/operationalization**, not raw capability.

That last point aligns Anthropic with the broader shift from “one-shot chatbot” to **structured agent systems with memory, decomposition, grading, and verification**.

## Facts vs opinions

### Factual claims with strongest support
- Anthropic has a new **SpaceX compute partnership** and increased Claude Code/API limits immediately [@claudeai](https://x.com/claudeai/status/2052060691893227611), [@claudeai](https://x.com/claudeai/status/2052060693269008586).
- Weekly limits were **not** doubled yet; Anthropic staff said that was intentional based on who hits which caps [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052064611692904639).
- Anthropic intends to run **Claude inference on Colossus** in the near term [@nottombrown](https://x.com/nottombrown/status/2052062566126649448).
- Anthropic ran a **Code with Claude** event focused on coding, production deployment, and managed agents [@ClaudeDevs](https://x.com/ClaudeDevs/status/2052055459272761661).

### Plausible but less directly verified claims
- Anthropic is gaining access to **>300 MW / >220,000 NVIDIA GPUs** in short order [@_arohan_](https://x.com/_arohan_/status/2052065871552819647).
- Colossus 1 inventory breakdown includes **H100/H200/GB200 mixes** [@scaling01](https://x.com/scaling01/status/2052068218047545501).
- Anthropic’s demand spike was around **80x growth** and caught leadership off guard [@kimmonismus](https://x.com/kimmonismus/status/2052118418174681572).

### Opinions and interpretations
- Anthropic **waited too long** to address compute shortages and lost significant growth to OpenAI/Codex: [@scaling01](https://x.com/scaling01/status/2052070594972090409).
- This deal proves **compute is not a durable moat**, because top labs can rent capacity from whichever hyperscaler/cluster operator will supply it: [@Dorialexander](https://x.com/Dorialexander/status/2052067579594707149).
- Alternatively, this proves the opposite in practical terms: **whoever controls deployed compute shapes who can satisfy demand**.
- Anthropic’s platform features are **not very differentiated** because open harnesses can replicate them: [@RichNwan](https://x.com/RichNwan/status/2052085746526216601).
- Or they **are differentiated enough** because first-party integration can tightly couple model behavior, memory, evaluators, and product experience.
- Anthropic’s culture is unusually safety-focused and “good for humanity”: Elon Musk said after meeting senior Anthropic staff he was impressed and “no one set off my evil detector” [@elonmusk](https://x.com/elonmusk/status/2052069691372478511).
- Conversely, critics continue to frame Anthropic as overly paternalistic or exclusivist about AGI governance [@_aidan_clark_](https://x.com/_aidan_clark_/status/2052089187659346047).

## Different opinions in the discourse

### 1) Positive / supportive
A large set of replies treated this as a win for users and evidence Anthropic is responding aggressively.

- [@alexalbert__](https://x.com/alexalbert__/status/2052065953173872912): “More chips, more Claude.”
- [@_sholtodouglas](https://x.com/_sholtodouglas/status/2052062164467224971): “More compute -> straight to you.”
- [@kimmonismus](https://x.com/kimmonismus/status/2052059448261177367) highlighted doubled limits and raised Opus API caps.
- [@TheRundownAI](https://x.com/TheRundownAI/status/2052064469371470218) summarized it as a straightforward user benefit.
- [@DannyLimanseta](https://x.com/DannyLimanseta/status/2052078750893056420) liked the cross-company cooperation and hoped Anthropic’s caution might be balanced by SpaceXAI’s optimism.
- [@AmandaAskell](https://x.com/AmandaAskell/status/2052161052058833181) reacted positively to the announcement’s symbolism.

### 2) Mixed / pragmatic
These takes welcomed the change but focused on operational details and remaining limitations.

- [@btibor91](https://x.com/btibor91/status/2052067002412335435) and [@kimmonismus](https://x.com/kimmonismus/status/2052061694080188720) immediately noted the likely caveat: **weekly caps unchanged**.
- [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052064611692904639) answered this directly.
- [@sbmaruf](https://x.com/sbmaruf/status/2052119971820658771) reported still seeing rate limits after the change, implying rollout and reliability tuning were ongoing.
- [@zachtratar](https://x.com/zachtratar/status/2052161984968396819) asked for patience during staged rollout.

### 3) Competitive / strategic critique
A different cluster viewed the announcement through the OpenAI-vs-Anthropic product war.

- [@scaling01](https://x.com/scaling01/status/2052070594972090409) argued Anthropic **blundered its growth advantage by waiting too long**, possibly conceding billions in ARR to OpenAI.
- [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2052065017072386450) read the move as Dario getting aggressive because of **OpenAI Codex’s growth**.
- [@_arohan_](https://x.com/_arohan_/status/2052053181656641735) joked that “Big tech has become a claude wrapper,” pointing to Claude’s developer mindshare.
- [@dejavucoder](https://x.com/dejavucoder/status/2052051193376231845) saying “claude is down, saint tibo please reset codex limits” captured the practical reality of multi-homing among coding tools when one service is capacity constrained.

### 4) Governance / safety / culture critique
This is the deepest philosophical disagreement.

- [@_aidan_clark_](https://x.com/_aidan_clark_/status/2052089187659346047) criticized what he says he repeatedly hears from Anthropic colleagues: a belief they alone should be trusted to build AI.
- [@kipperrii](https://x.com/kipperrii/status/2052094851991392536) partially agreed the “only we can be trusted” framing would be bad, but argued the real majority view is closer to **“no one can be trusted with AGI”** while still personally trusting Anthropic more than others.
- [@elonmusk](https://x.com/elonmusk/status/2052069691372478511) offered a surprising endorsement after meeting Anthropic leaders.
- [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2052080339364004317) called this reversal ironic given prior criticism of Anthropic.
- [@teortaxesTex](https://x.com/teortaxesTex/status/2052080900280557749) mocked the rapid détente between Musk/xAI and Anthropic.
- [@teortaxesTex](https://x.com/teortaxesTex/status/2052045988936683674) also argued it is inconsistent to warn others about AI risk while building powerful closed systems such as “Mythos.”
- [@goodside](https://x.com/goodside/status/2052077014346064372), while not directly about Anthropic governance, contributed to the broader moral/AI norms debate that often clusters around Anthropic.

## Commentary on Claude model performance and comparisons

Though no major new Claude model appears in these tweets, Claude remained a reference point in product and eval discourse.

- [@giffmana](https://x.com/giffmana/status/2051925008457273527) compared “Opus 4.6,” ChatGPT Pro, and Muse Spark on a mathematical disagreement. His take:  
  - **Opus 4.6** confidently defended a wrong proof (“gaslit”)  
  - **ChatGPT Pro** reconciled the formulas correctly but without interpretation  
  - **Muse Spark** did both well  
  This is anecdotal, but it’s one of the more concrete comparative qualitative model reports in the set.
- [@kimmonismus](https://x.com/kimmonismus/status/2052040471829004627) summarized a Substack analysis claiming **GPT-5.5 is basically tied with Claude Mythos Preview on cyber**, perhaps more cost-efficient, while Mythos is only slightly ahead on some general benchmarks and SWE-bench Pro; he questioned why Mythos remains secretive.
- [@AssemblyAI](https://x.com/AssemblyAI/status/2052043337751056733) noted support for **structured JSON from Claude 4.5+ models** in its gateway.
- [@OpenRouter/TencentHunyuan](https://x.com/TencentHunyuan/status/2051978552900538403) listed **Claude Code** among major apps driving Hy3 usage, showing Claude’s importance in the coding-tool ecosystem even when third-party models are used behind the scenes.

These comments don’t establish hard model ranking, but they do show Claude is still a primary benchmark in coding-agent workflows and that advanced users increasingly compare **model + harness + limits + reliability**, not just base intelligence.

## Claude Code and harness engineering context

A notable background thread across the dataset is that many engineers now think **agent performance is heavily dependent on the harness**—system prompts, tools, middleware, decomposition strategies, and model-specific tuning.

Relevant non-Anthropic commentary:
- [@masondrxy](https://x.com/masondrxy/status/2052054177749029164): same model, same task, very different scores depending on prompts/tools/middleware; **10–20 point jumps on tau2-bench**.
- [@LangChain](https://x.com/LangChain/status/2052054711440662864): harness profiles for OpenAI, Anthropic, and Google models.
- [@jakebroekhuizen](https://x.com/jakebroekhuizen/status/2052058987580051566): distinguishes **temporal harness evolution** as models improve from **lateral tuning across model families**.
- [@Vtrivedy10](https://x.com/Vtrivedy10/status/2052100726608781363): argues a tailored harness can outperform default Codex/Claude Code on many tasks; usable context windows are still effectively **50–100k** for many agent designs.
- [@kieranklaassen](https://x.com/kieranklaassen/status/2052092428438688027): “If you cannot get your work done [in] the Claude CLI, Claude will not be able to work for you.”

This matters because some of Anthropic’s platform moves—memory, grading, managed agents—can be read as **Anthropic productizing parts of the harness**. That helps explain the central debate: **are these defensible platform primitives, or just first-party packaging of patterns that open frameworks can clone?**

## Broader context: why this matters

1. **Inference, not just training, is now a frontier bottleneck.**  
   The news was not a new model launch; it was a capacity launch. That is increasingly common at the frontier.

2. **Compute markets are becoming fluid and strategic.**  
   Anthropic partnering with SpaceX/xAI infrastructure undercuts simplistic narratives that each frontier lab sits only atop its own vertically integrated stack.

3. **Developer product share is sensitive to reliability and limits.**  
   Claude appears to have strong developer affinity, but rate limits and outages push users toward Codex/Cursor/others quickly.

4. **The battleground is shifting from base models to agent systems.**  
   “Code with Claude,” managed agents, Dreaming, Outcomes, and the surrounding discourse all point toward the next layer of competition being **memory, orchestration, evals, and workflow integration**.

5. **Anthropic’s brand remains bifurcated.**  
   It is simultaneously:
   - admired for product quality and safety seriousness,
   - criticized for paternalism or perceived exclusivism,
   - and now seen as more commercially aggressive on compute than before.

## Bottom line

Anthropic’s news was less about a flashy new model and more about a structural reality: **Claude demand had outrun available compute, and Anthropic responded by striking a major external infrastructure deal and immediately easing key user limits** [@claudeai](https://x.com/claudeai/status/2052060691893227611), [@claudeai](https://x.com/claudeai/status/2052060693269008586). The most important technical/economic signal is that **capacity, rate limits, and agent-product ergonomics are now as strategically important as leaderboard deltas**. The main open questions are whether Anthropic can convert this capacity into sustained product momentum, whether its managed-agent features are truly differentiated, and whether its safety/governance posture helps or hinders its standing as competition with OpenAI, Google, xAI, and open-model ecosystems intensifies.


**Infrastructure, inference, and systems**


- OpenAI and partners released **MRC (Multipath Reliable Connection)**, an open networking protocol for large AI training clusters, already deployed on OpenAI’s biggest supercomputers [@OpenAI](https://x.com/OpenAI/status/2052025532485902368), [@OpenAI](https://x.com/OpenAI/status/2052025533937103102). Commentary emphasized multipath routing, microsecond failover, and the shift of networking into a primary frontier bottleneck [@kimmonismus](https://x.com/kimmonismus/status/2052011784023028060), [@gdb](https://x.com/gdb/status/2052059553542328829).
- Perplexity said it built an in-house inference engine, **ROSE**, covering models from embeddings to trillion-parameter LLMs, and uses **CuTeDSL** to accelerate specialized kernel development on Hopper and Blackwell [@perplexity_ai](https://x.com/perplexity_ai/status/2052041903970148647).
- vLLM + Mooncake presented a strong systems result for agentic workloads with reusable prefixes: **3.8x throughput**, **46x lower P50 TTFT**, **8.6x lower end-to-end latency**, and cache-hit improvement from **1.7% to 92.2%**, scaling to **60 GB200 GPUs** [@vllm_project](https://x.com/vllm_project/status/2052113331927060840).
- Unsloth + NVIDIA published three training optimizations claimed to make home-GPU LLM training **~25% faster**: packed-sequence metadata caching, double-buffered checkpoint reloads, and faster MoE routing [@UnslothAI](https://x.com/UnslothAI/status/2052020656527532276).
- NVIDIA work on **lossless speculative decoding inside RL** was highlighted as giving up to **~2.5x faster end-to-end RL at 235B scale** and **~1.8x faster rollout throughput at 8B** without changing policy distribution [@TheTuringPost](https://x.com/TheTuringPost/status/2052180472206381268).
- Baseten launched **Frontier Gateway** as managed infra/API/auth/rate-limit/billing for closed-weight labs; Poolside reported going from kickoff to production in **7 weeks**, with **P50 TTFT 146ms** for Laguna XS.2 and **605ms** for Laguna M.1 [@tuhinone](https://x.com/tuhinone/status/2052082677432390130), [@poolsideai](https://x.com/poolsideai/status/2052075055132057707).


**Benchmarks, evals, and agent harnesses**


- **ProgramBench** asks whether language models can rebuild programs from scratch, extending beyond repair-style SWE tasks [@ComputerPapers](https://x.com/ComputerPapers/status/2051895799043215415), with Ofir Press arguing benchmarks are “treasure maps” that specify the future we want [@OfirPress](https://x.com/OfirPress/status/2052106927908200957).
- **Terminal-Bench 2.1** patched **28/89 tasks** in TB2.0; rankings held but absolute scores moved by up to **12 points**, a useful reminder that agent benchmark maintenance materially matters [@terminalbench](https://x.com/terminalbench/status/2052119174500220964), [@ekellbuch](https://x.com/ekellbuch/status/2052165464655298866).
- **OBLIQ-Bench** emerged as a major IR benchmark release focused on hard first-stage retrieval, where current retrievers fail to surface subtly relevant documents from large corpora [@dianetc_](https://x.com/dianetc_/status/2052053806121140254), with strong endorsements from IR researchers [@lateinteraction](https://x.com/lateinteraction/status/2052055143038713875), [@nlp_mit](https://x.com/nlp_mit/status/2052069072607547892), [@LightOnIO](https://x.com/LightOnIO/status/2052095548098822477).
- Harvey launched **LAB**, an open-source, long-horizon legal agent benchmark covering **1,200 tasks across 24 practice areas**, with support/commentary from LangChain, Baseten, Artificial Analysis, and others [@saranormous](https://x.com/saranormous/status/2052061665596948894), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2052145762650431840).
- A major theme across multiple tweets was that **harness engineering is a first-class variable**, often worth **10–20 points** on agent benchmarks even with the same base model [@masondrxy](https://x.com/masondrxy/status/2052054177749029164), [@LangChain](https://x.com/LangChain/status/2052054711440662864), [@Vtrivedy10](https://x.com/Vtrivedy10/status/2052100726608781363).


**Model releases and model performance**


- Zyphra released **ZAYA1-8B**, a reasoning MoE with **<1B active parameters**, open-weight under **Apache 2.0**, claiming strong math/reasoning efficiency and proximity to much larger systems with test-time compute [@ZyphraAI](https://x.com/ZyphraAI/status/2052103618145501459), [@ZyphraAI](https://x.com/ZyphraAI/status/2052103646712828119). Commentary praised its architecture/post-training stack and AMD partnership [@teortaxesTex](https://x.com/teortaxesTex/status/2052106600882528326), [@eliebakouch](https://x.com/eliebakouch/status/2052126118891729148).
- Google’s **Gemma 4** moved the open-model Pareto frontier in Code Arena: **Gemma-4-31B #13**, **Gemma-4-26B-A4B #17** among open models [@arena](https://x.com/arena/status/2052061349312921686), [@_philschmid](https://x.com/_philschmid/status/2052104144706588699).
- Google’s **DFlash draft model for Gemma-4** was described as one of the best draft models they’ve trained, especially strong in coding and math [@jianchen1799](https://x.com/jianchen1799/status/2051902953376923946).
- Qwopus3.6-35B-A3B-v1 claimed **162 tok/s on a single RTX 5090**, targeting strong one-shot frontend/web generation on consumer hardware [@KyleHessling1](https://x.com/KyleHessling1/status/2052064943999267212).
- DeepSeek commentary was mixed: fundraising talks reportedly target a **$45B valuation** led by a major Chinese state-backed semiconductor fund [@jukan05](https://x.com/jukan05/status/2051904572038455634), while evaluators debated weak WeirdML performance for V4-Pro versus GLM/Kimi/open competitors [@htihle](https://x.com/htihle/status/2052042076196335658), [@teortaxesTex](https://x.com/teortaxesTex/status/2052043753892761882).


**Agents, tools, and developer workflows**


- Cursor added **context usage breakdowns** across rules, skills, MCPs, and subagents to help debug context issues [@cursor_ai](https://x.com/cursor_ai/status/2052059748544249918), and described bootstrapping future Composer generations with earlier Composer models [@cursor_ai](https://x.com/cursor_ai/status/2052116064474161556).
- Cognition shipped **Devin Review** and **Quick Review / SWE-Check** in Windsurf 2.0, explicitly targeting the new bottleneck of reviewing AI-generated code [@cognition](https://x.com/cognition/status/2052100630626607189), [@ypatil125](https://x.com/ypatil125/status/2052122827961278833).
- OpenAI promoted **Codex subagents**, framing them as a way to split work across specialized agents and merge results back into one answer [@reach_vb](https://x.com/reach_vb/status/2052090279344120278).
- Nous/Hermes continued to push a highly pluggable local agent stack: plugin expansion, community docs, Windows/WSL2 setup guidance, and use-case aggregation [@Teknium](https://x.com/Teknium/status/2052046335583625629), [@witcheer](https://x.com/witcheer/status/2052033039379673374), [@NousResearch](https://x.com/NousResearch/status/2052140057222369541).
- Perplexity added **Finance Search** to its Agent API with licensed data, live market data, and citations, claiming best cohort accuracy and lowest cost per correct answer on **FinSearchComp T1** [@perplexity_ai](https://x.com/perplexity_ai/status/2052028012313649194), [@AravSrinivas](https://x.com/AravSrinivas/status/2052033959555735752).
- Google’s Gemini API added **multimodal retrieval** to File Search using `gemini-embedding-2` for PDFs and images in a single retrieval pipeline [@_philschmid](https://x.com/_philschmid/status/2052060912425546050).


**Robotics, multimodality, and research notes**


- Genesis AI introduced **GENE-26.5**, describing a full-stack robotics program with a robotics-native foundation model, human-like hand, data glove, and simulator; the model is trained across **language, vision, proprioception, tactile, and action** [@gs_ai_](https://x.com/gs_ai_/status/2052050956272230577), [@theo_gervet](https://x.com/theo_gervet/status/2052057035681018359).
- Meta FAIR released **NeuralBench**, an MIT-licensed unified benchmark framework for NeuroAI with **36 EEG tasks** and **94 datasets**, with MEG/fMRI support planned [@hubertjbanville](https://x.com/hubertjbanville/status/2052029372282888234), [@JeanRemiKing](https://x.com/JeanRemiKing/status/2052034314120896582).
- Sander Dieleman published a long technical post on **flow maps**, learning the integral of a diffusion model for faster sampling and related tricks [@sedielem](https://x.com/sedielem/status/2051957402556104799).
- François Fleuret sketched a speculative recipe for stronger systems: **latent diffusion-like reasoning + real recurrent state + world-model pre-pretraining** [@francoisfleuret](https://x.com/francoisfleuret/status/2051928896027693479), generating useful discussion on whether diffusion-style reasoning extrapolates the right way [@willdepue](https://x.com/willdepue/status/2052033422915477580), [@jeremyphoward](https://x.com/jeremyphoward/status/2052149483740545400).
- HeadVis was introduced as a new interpretability tool for studying attention heads [@kamath_harish](https://x.com/kamath_harish/status/2052046203030827088).
- Microsoft Research work on **agent-readable interpretability** proposed “Agentic-imodels,” where coding agents evolve models that are interpretable to other LLMs; reported gains on **65 tabular datasets** and downstream BLADE improvements from **8% to 73%** [@dair_ai](https://x.com/dair_ai/status/2052125514266190286).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. MTP and Quantized Local Inference

  - **[Gemma 4 MTP released](https://www.reddit.com/r/LocalLLaMA/comments/1t4jq6h/gemma_4_mtp_released/)** (Activity: 1575): ****Google released Multi-Token Prediction (MTP) draft checkpoints for Gemma 4**—[`31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant), [`26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant), [`E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant), and [`E2B-it-assistant`](https://huggingface.co/google/gemma-4-E2B-it-assistant)—described in Google’s [announcement](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/). The model cards say MTP extends the base model with a smaller draft model for **speculative decoding**, where the draft predicts multiple tokens ahead and the target model verifies them in parallel, claiming **up to `2x` decoding speedup** with *“the exact same quality as standard generation.”* A commenter notes the smallest `E2B` variant uses a **`78M` draft model**, and another shared a technical visual explainer on MTP with Gemma 4 [here](https://newsletter.maartengrootendorst.com/i/193064129/multi-token-prediction-mtp-with-gemma-4).**

    - A commenter linked an updated visual explainer of **multi-token prediction (MTP)** for Gemma 4, including implementation-oriented snippets: [Maarten Grootendorst’s guide](https://newsletter.maartengrootendorst.com/i/193064129/multi-token-prediction-mtp-with-gemma-4). This is relevant for understanding how Gemma 4’s MTP setup predicts multiple future tokens per forward pass and how that interacts with speculative/draft-style decoding.
    - One technical detail called out is that the **E2B model includes a `78M`-parameter draft model**, implying a lightweight auxiliary model for faster generation workflows such as speculative decoding. The small draft size is notable because it can reduce decode latency while keeping the verifier/main model responsible for final token acceptance.

  - **[2.5x faster inference with Qwen 3.6 27B using MTP - Finally a viable option for local agentic coding - 262k context on 48GB - Fixed chat template - Drop-in OpenAI and Anthropic API endpoints](https://www.reddit.com/r/LocalLLaMA/comments/1t57xuu/25x_faster_inference_with_qwen_36_27b_using_mtp/)** (Activity: 1445): **A llama.cpp PR ([`pull/22673`](https://github.com/ggml-org/llama.cpp/pull/22673)) adds **Qwen 3.6 27B MTP** support for speculative decoding using the model’s built-in multi-token prediction heads; the author reports **~`2.5×` faster generation** on an M2 Max 96GB, reaching **`28 tok/s`**, and published converted GGUFs with MTP tensors at [froggeric/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF). The setup combines `--spec-type mtp --spec-draft-n-max 5`, `q4_0`/`q8_0` KV-cache quantization, and long contexts up to **`262144` tokens**, with claimed viability on **48GB Mac/VRAM-class systems**; the author also uploaded fixed non-vLLM-specific Jinja chat templates at [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates). Caveats: current MTP support requires building llama.cpp from the PR branch, `q4_0` KV has some quality loss, and **vision currently crashes llama.cpp when used with MTP**; one commenter benchmarked Qwen 3.6 2.7B Q8 on an RTX Pro 6000 MaxQ at **`36 tok/s` → `78 tok/s` with MTP**, while noting ~`20%` slower prompt processing.** Comments were broadly enthusiastic, framing recent open-model and inference-runtime progress as unusually rapid and especially important for consumer/local hardware. One technical question asked whether “turbo3/turbo4” had been merged or whether it was part of the MTP PR.

    - A user reported a concrete MTP speedup on an **RTX Pro 6000 MaxQ**: `qwen 3.6 2.7B Q8` increased from `36 tokens/s` to `78 tokens/s` with MTP enabled, while prompt processing dropped by about `20%`. They said generation quality appeared unchanged, making the tradeoff strongly favorable for decode-heavy workloads.
    - One commenter asked whether the `turbo3`/`turbo4` changes had already been merged or whether the observed speedup is specifically part of the **MTP PR**, highlighting uncertainty about which inference optimization path is responsible for the gains.
    - There was a technical comparison request against **Qwen 3.6 Dflash** models and low-bit `iq3_XS` quantizations. The commenter noted they can usually fit `256k` context in `16GB` VRAM and asked whether the released quants can also support `256k` context when not using `mmproj`.

  - **[Quality comparison between Qwen 3.6 27B quantizations (BF16, Q8_0, Q6_K, Q5_K_XL, Q4_K_XL, IQ4_XS, IQ3_XXS,...)](https://www.reddit.com/r/LocalLLaMA/comments/1t53dhp/quality_comparison_between_qwen_36_27b/)** (Activity: 771): **A Reddit user benchmarked **Qwen 3.6 27B** quantizations on a synthetic chess-to-SVG task requiring PGN state tracking, board orientation, piece placement, and last-move highlighting, using `llama.cpp` with `temp=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.0`, and `ctx=65536`. In this single-run test, **BF16/Q8_0** were essentially correct, **Q6_K** showed pawn-placement degradation, **Q5_K_XL/Q4_K_XL/IQ4_XS** remained mostly usable, while **Q3/Q2** variants increasingly failed layout/orientation; the author chose **IQ4_XS** as the practical floor for a `16 GB` VRAM RTX 5060 Ti setup. They report `~100 pp tps / 8 tg tps` with vanilla `llama.cpp`, improving to `~760 pp tps / 22 tg tps` using **TheTom’s TurboQuant fork** with `-ngl 99`, `-ctk turbo4`, `-ctv turbo2`, and `<75k` context; full outputs are posted at [qwen3-6-27b-benchmark.vercel.app](https://qwen3-6-27b-benchmark.vercel.app/).** Top technical feedback praised the benchmark but emphasized that *“one run is not enough”* because stochastic decoding can make individual quant results outliers; commenters still noted the observed degradation trend broadly matches expectations.

    - Several commenters raised a methodology concern: the quantization comparison appears to rely on single runs per test, which can produce **statistical noise** and misleading quality differences. They suggested running each quant multiple times to detect outliers, especially because LLM evals can vary run-to-run even when an overall degradation trend is visible.
    - One technical takeaway discussed was that **`4-bit` quantization may remain the practical sweet spot**, with `3-bit` described as more usable than commonly claimed, while going beyond roughly `5-bit` may offer diminishing returns versus moving to a larger/better base model. A commenter specifically contrasted cases like a much larger `122B UD-Q3_K_XL` model against a smaller `35B IQ4_NL` model to argue that model scale can outweigh higher-bit quantization quality.


### 2. Agentic Coding and Cost Benchmarks

  - **[DeepSeek V4 Pro matches GPT-5.2 on FoodTruck Bench, our agentic benchmark — 10 weeks later, ~17× cheaper](https://www.reddit.com/r/LocalLLaMA/comments/1t47qbw/deepseek_v4_pro_matches_gpt52_on_foodtruck_bench/)** (Activity: 478): **The image is a **technical leaderboard screenshot** for FoodTruck Bench showing **DeepSeek V4 Pro** highlighted at rank `#4` with `$27,142` final net worth, `+1257% ROI`, `51%` margin, `$52,139` revenue, and `$26,492` profit over a 30-day agentic food-truck simulation starting from `$2,000` ([image](https://i.redd.it/fx89f3w5n9zg1.png)). This supports the post’s claim that DeepSeek V4 Pro is within ~`3%` of **GPT-5.2**’s median outcome while reportedly being ~`17×` cheaper on the same workload, making it a frontier-tier result in this benchmark at much lower API cost.** Commenters were impressed but skeptical about interpretation: one noted **Claude Opus 4.6** appears far ahead in profit, while another questioned the benchmark’s credibility if **Gemma 4 31B** can beat **Sonnet 4.6**. There was also curiosity about absent newer GPT variants like “GPT 5.4/5.5.”

    - Several commenters focused on the benchmark ranking implications rather than the headline DeepSeek result: **Claude Opus 4.6** reportedly achieves about `1.7×` higher profit than the next cluster of models on **FoodTruck Bench**, suggesting a sizable lead in this agentic profit-optimization benchmark despite DeepSeek V4 Pro matching **GPT-5.2** at much lower cost.
    - Multiple users called out **Gemma 31B** as an under-discussed outlier: it appears in the top 5 on FoodTruck Bench, reportedly beats **Sonnet 4.6**, and also performs well on **EQBench**. Commenters questioned why Gemma is receiving less attention relative to Xiaomi/DeepSeek results if those rankings hold.
    - There were requests to expand the comparison set with newer or missing models, specifically **GPT-5.4/5.5**, the latest **Qwen3.6** models, and a `27B` model that one commenter expected might outperform Gemma. The implied concern is that the benchmark table may be incomplete or stale for evaluating current frontier and mid-size model competitiveness.

  - **[Claude Code @ Opus 4.7 vs OpenCode @ qwen3.6:27b. Both shipped a playable cozy roguelite.](https://www.reddit.com/r/LocalLLM/comments/1t49wld/claude_code_opus_47_vs_opencode_qwen3627b_both/)** (Activity: 406): **A one-shot benchmark compared **Claude Code on Opus 4.7** vs **OpenCode on local Qwen3.6:27B** using identical VS Code devcontainers and a strict greenfield prompt for a vanilla Canvas/FastAPI roguelite; both produced a playable first-run game implementing movement, sword/shield combat, procedural world, drops, swap UI, and restart loop. Opus took ~`20 min` and `97k` tokens, while Qwen took ~`15 min` and `64k` tokens—about one-third fewer tokens—though the author explicitly limits the claim to tightly specified greenfield work rather than hard reasoning or existing-codebase maintenance. The linked Reddit-hosted video [`v.redd.it/h4awffniaazg1`](https://v.redd.it/h4awffniaazg1) was not accessible in the provided crawl due to Reddit `403 Forbidden` access restrictions.** Commenters focused on reproducibility and local-model capability: one asked for the full prompt, while others characterized **Qwen3.6 27B** as surprisingly strong for coding/tricky questions, less hallucination-prone than some MoE alternatives, and roughly comparable to last year’s **Sonnet 4.5** for many coding tasks. Another commenter said the `35B` variant performs well on large-codebase edit tasks when “properly harnessed.”

    - Users requested key reproducibility details missing from the comparison: the exact prompt, hardware used for the local Qwen run, and whether any quantization was applied to `qwen3.6:27b`. These details are important because local model throughput and coding quality can vary significantly by quantization level and memory bandwidth/GPU or Apple Silicon configuration.
    - One commenter reported `Qwen3.6 27B` running “very slow” on an **M1 Pro**, but still handling coding and tricky questions well. They claimed it hallucinated less than `35B A3B` and `Gemma MoE`, and estimated it as roughly comparable to `Sonnet 4.5` from the previous year, making it usable for “90% of coding tasks.”
    - Another user argued that the `35B` model performs strongly when “properly harnessed” and given large codebase context for inspection and edits, suggesting orchestration/context management may matter as much as raw model choice for coding-agent workflows.

  - **[DeepSeek V4 being 17x cheaper got me to actually measure what I send to cloud vs what I could run locally. the results are stupid.](https://www.reddit.com/r/LocalLLaMA/comments/1t4s6g2/deepseek_v4_being_17x_cheaper_got_me_to_actually/)** (Activity: 904): **A developer instrumented `10` days of coding-agent usage and re-ran a `150`-task sample against a local **Qwen 3.6 27B** model on an **RTX 3090** versus cloud models, finding local parity for `97%` of file-read/project-scan/explanation tasks (`35%` of workload) and `88%` of test/boilerplate/single-file-edit tasks (`30%`). Local quality degraded on multi-file debugging (`61%`, `20%` of workload) and complex architecture/refactors across `5+` files (`29%`, `15%`), so routing only the latter buckets to cloud reportedly cut API spend from `$85/month` to about `$22/month`.** Commenters generally agreed with a hybrid/local-first workflow: some report using local models for nearly all coding, escalating only to Gemini/ChatGPT/Claude/Qwen/GLM free tiers or cloud models for planning, oversight, unusually complex tasks, or non-code domains like health/legal. One commenter asked for implementation details on the task-type router/harness, implying the key missing technical artifact is the automation layer for classification and dispatch.

    - Several commenters describe a **hybrid local/cloud workflow**: local models handle most code-related tasks, while cloud/free web tiers such as **ChatGPT, Claude, Gemini, Qwen, GLM**, or Gemini specifically are reserved for planning, oversight, or rare complex problems. One user reports running with **zero subscriptions**, using cloud mostly for non-code domains like health/legal queries where local model reliability may be less acceptable.
    - A key technical objection is that local models can be **slower on large contexts** and impose hidden costs through extra verification/debugging time. One commenter argues that even if local inference is cheaper, the `~10%` of cases where local models underperform can dominate productivity costs, and suggests hosted **Qwen 3.6 27B / Qwen 3.6 Pro** may be faster and still only cost “a couple dollars a month.”




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic Claude Code Limits and Reliability

  - **[Doubled Rate Limits for Claude Code](https://www.reddit.com/r/ClaudeCode/comments/1t5hs98/doubled_rate_limits_for_claude_code/)** (Activity: 3224): ****Anthropic** says a new compute partnership with **SpaceX**, plus other recent compute deals, lets it raise Claude capacity: **Claude Code** Pro/Max plans no longer get peak-hours limit reductions, and **Claude API** rate limits for **Opus** models are being “substantially” increased, effective immediately ([Anthropic announcement](https://www.anthropic.com/news/higher-limits-spacex)). The post frames this as “doubled rate limits,” but the quoted announcement itself specifies removal of peak-hour throttling for Claude Code and higher Opus API limits rather than giving exact numeric quotas.** Top comments were mostly non-technical surprise/skepticism and speculation about Elon Musk’s rivalry with Sam Altman/OpenAI.


  - **[I've had it with Claude. It has become complete garbage.](https://www.reddit.com/r/ClaudeCode/comments/1t4w5an/ive_had_it_with_claude_it_has_become_complete/)** (Activity: 1716): **A senior SWE reports a major regression in **Anthropic [Claude](https://www.anthropic.com/claude)** after “Opus 4.7” versus “Opus 4.6”: slower CLI interactions (`30s` for commits, `45min` implementations), worse terminal/Tmux rendering on resize, loss of useful `Ctrl+O` trace visibility, more frequent usage-limit hits, and poorer instruction adherence despite project memory/context engineering. The concrete technical failures cited include ignoring short test timeouts (`10–15s` → `30s/60s/5min`), auto-committing despite “never auto commit,” verbosity drift despite `/caveman`, implementing a Rust refactor by adding `handle_input_bytes(Bytes)` instead of changing `handle_input(&[u8])` to `Bytes`, and deviating from an `io_uring` cancel-safety plan by reverting toward a racy one-shot/multi-shot recv shortcut before acknowledging *“Yes deviating. Confess.”*** Top comments split between agreement that losing visible reasoning makes it harder to interrupt bad loops, users cancelling Max and moving to open-source models for stability, and a dissenting experienced developer saying Claude remains productive when using disciplined `Claude.md`/`memory.md`, scoped plans, milestones, and avoiding excessive context loading.

    - A long-time software developer reports stable coding performance by using a constrained project workflow: well-maintained `Claude.md` and `memory.md`, a small number of skills, upfront planning, milestone-based implementation, and repeated build/test/release cycles. They argue many failures may come from poor context hygiene—either loading “29 different markdown files” as an oversized pseudo-OS or dumping the full context window into every command.
    - One user highlights a UX/regression issue from hiding chain-of-thought-style progress: without visible “thinking,” they can no longer tell whether Claude is looping internally versus waiting on server-side latency. This makes it harder to interrupt unproductive reasoning early and diagnose whether a delay is model behavior or infrastructure-related.
    - Several users report time-dependent quality variance, with one specifically claiming worse Claude behavior during `8am–2pm Eastern (US)` peak usage: more corner-cutting, sloppier outputs, and “brain dead” behavior, while off-peak usage feels closer to prior quality. The implied technical concern is load-dependent degradation, potentially from capacity pressure, routing, throttling, or model/serving changes during peak demand.

  - **[Turned a desk lamp into a Claude Code status indicator](https://www.reddit.com/r/ClaudeAI/comments/1t4gfc7/turned_a_desk_lamp_into_a_claude_code_status/)** (Activity: 1817): **A Reddit user adapted the open-source [`bobek-balinek/claude-lamp`](https://github.com/bobek-balinek/claude-lamp) project to turn a BLE desk lamp into a **Claude Code status indicator**: Claude Code hooks invoke a Python script that sends Bluetooth Low Energy commands to set animations/colors. The lamp shows a **blue spinning animation** while Claude is working, **pink** when user input is required, and **warm white** when idle; effects are configurable in source, and the author is considering extending the setup to **Philips Hue** bulbs. The linked Reddit video was inaccessible due to a `403 Forbidden` response.** Commenters mainly asked for the lamp model and discussed scaling the idea to multiple concurrent Claude Code sessions, e.g. using multiple lights or designing a better multi-session status indicator. One commenter noted the title could also imply showing Anthropic service health via [`status.claude.com`](https://status.claude.com/).

    - A commenter suggested extending the lamp beyond local Claude Code state to reflect **Claude service health**, using Anthropic’s public status page at [status.claude.com](https://status.claude.com/) as the data source. This would make the indicator represent operational availability rather than just local task/session state.
    - Another technical improvement proposed was visualizing **remaining Claude Code usage within the rolling five-hour window**, e.g. lighting the lamp or “donut” proportionally to quota left. A separate comment raised the multi-session case, implying the indicator would need aggregation or per-session state handling if multiple Claude Code sessions run concurrently.

  - **[Warning: Anthropic's "Gift Max" exploit drained €800+, ruined my credit, and got me banned.](https://www.reddit.com/r/ChatGPT/comments/1t4atbx/warning_anthropics_gift_max_exploit_drained_800/)** (Activity: 3451): **OP reports **>€800** in unauthorized Anthropic **“Gift Max”** charges despite active `2FA`; they claim `3-D Secure` emails were received but never authorized, while gift codes were generated and immediately redeemed by a third party. They tie the incident to Anthropic’s [status page](https://status.anthropic.com/) entry for *“Elevated billing errors and unauthorized subscription changes”* and GitHub issues `#51404`/`#51168`, then say Anthropic banned the account after receiving a police report and evidence, cutting off access to WIP chats/projects. In an update, OP says their bank treated it as fraud, issued a reclamation/refund, and will pursue Anthropic’s merchant account; they are also considering a [GDPR/DSGVO](https://gdpr.eu/) data request to recover data and German legal aid to repair possible [SCHUFA](https://www.schufa.de/) credit impacts.** Comments were mostly practical or skeptical: one noted that in the U.S. this would typically be handled via card chargeback, while another highlighted the irony/suspicion of a Gemini-written anti-Anthropic warning posted in a ChatGPT subreddit.

    - The OP reports their bank reversed the `€800+` Anthropic-related charges as a fraud case and will pursue the merchant account directly. They also plan to file a formal GDPR/DSGVO data request to recover work-in-progress project data and seek German legal aid (*Beratungshilfeschein*) to ensure any SCHUFA credit entries are cleared.
    - One commenter notes seeing multiple YouTube ads from different merchants all advertising “1 year free Claude access,” suggesting a coordinated scam campaign potentially related to the reported exploit or phishing/payment-abuse pattern.



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
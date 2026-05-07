---
id: MjAyNS0x
title: not much happened today
date: '2026-05-04T05:44:39.731046Z'
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

> AI News for 5/4/2026-5/5/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




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



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
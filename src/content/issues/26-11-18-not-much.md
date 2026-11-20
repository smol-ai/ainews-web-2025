---
id: MjAyNS0x
title: not much happened today
date: '2026-11-18T05:44:39.731046Z'
description: >-
  **Google DeepMind** launched **Gemini 4 Neon**, a model positioned between
  Carbon and Hydrogen, targeting software engineering, knowledge work, and
  science workflows with a 1M-token context window and pricing at $3/$15 per
  million tokens. Independent evaluations show Neon ranking highly on multiple
  leaderboards like Agent Arena and Vals Index, with improved steerability and
  lower fabrication rates. It is integrated into Gemini API, AI Studio, GitHub
  Copilot, and Cursor. **GPT-6.2 Astra** and **Sol** excel in math and formal
  proof benchmarks, with Astra achieving top results on MathArena and Lean proof
  tasks. Early practitioner feedback on Neon is positive despite some
  operational issues.
companies:
  - google-deepmind
  - github
  - cursor_ai
models:
  - gemini-4-neon
  - carbon
  - hydrogen
  - gpt-6.2-astra
  - opus-5.5
  - claude-fable-5.2
topics:
  - software-engineering
  - knowledge-work
  - science-workflows
  - context-windows
  - model-pricing
  - benchmarking
  - agent-arena
  - intelligence-index
  - coding-agents
  - cost-efficiency
  - model-performance
  - independent-evaluation
  - math
  - formal-proof
  - lean-proof
  - model-integration
people:
  - sundar_pichai
  - koraykv
  - vals
  - theo
  - kimmonismus
  - timgmath
  - kilian_lieret
---


**a quiet day.**

> AI News for 11/17/2026-11/18/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Google DeepMind shipped Gemini 4 Neon, the fourth Gemini 4 model in about seven weeks. It is a mid-tier, faster and cheaper model, and early third-party evals put it at or near the frontier.**

- **Launch.** Announced by [@sundarpichai](https://x.com/sundarpichai/status/2123167780988125621), [@demishassabis](https://x.com/demishassabis/status/2123168218743554465), [@koraykv](https://x.com/koraykv/status/2123167638050541626), [@OfficialLoganK](https://x.com/OfficialLoganK/status/2123167791922741362) and [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2123167884411077030). Google pitches it as "frontier-level intelligence with fast end-to-end task completion" for SWE, knowledge work and science. DeepMind places it "between Carbon and Hydrogen."
- **Pricing and specs.** Per [@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2123168408678359294) and [@ValsAI](https://x.com/ValsAI/status/2123168183746195620):
  - $3/$15 per 1M input/output tokens.
  - 1M context window and 1M max output tokens.
  - Four thinking levels: low, medium, high and xhigh.
- **Availability.** Rollout is staged ([@Google](https://x.com/Google/status/2123172294260396442), [@koraykv](https://x.com/koraykv/status/2123167644161634344)):
  - Paid Gemini API tier in AI Studio and Android Studio.
  - Gemini for business and Google Cloud.
  - Antigravity for paid Pro/Ultra subscribers.
  - Gemini app for Pro/Ultra, labeled "Deeper analysis" in the dropdown, including Tasks ([@GeminiApp](https://x.com/GeminiApp/status/2123169070988337603), [Tasks](https://x.com/GeminiApp/status/2123169000310096115)).
- **Day-0 integrations.**
  - [Cursor](https://x.com/cursor_ai/status/2123198346802450926).
  - [GitHub Copilot](https://x.com/github/status/2123181574116233646) app, CLI and VS Code, generally available.
  - [LangChain Managed Deep Agents](https://x.com/caspar_br/status/2123192026745409844).
- **Release cadence.** Arena counts the gaps between Gemini 4 releases as roughly halving each time ([@arena](https://x.com/arena/status/2123192239761596566), [@ml_angelopoulos](https://x.com/ml_angelopoulos/status/2123169986944586103)):
  - Argon (Sep 30) → Carbon: 28 days.
  - Carbon → Hydrogen: 15 days.
  - Hydrogen → Neon: 6 days.
- **Pre-launch signals.**
  - Teaser posts from [@OfficialLoganK](https://x.com/SicongJiang25/status/2123004684516335793), [@osanseviero](https://x.com/osanseviero/status/2122949758440575397) and [@_philschmid](https://x.com/_philschmid/status/2122954931250565187).
  - Osanseviero said serving and caching work had already made Carbon's TTFT in Antigravity [1.5x faster](https://x.com/osanseviero/status/2123004863579619513).
  - [@dejavucoder](https://x.com/dejavucoder/status/2123079438577676729) speculated Google would drop something "better than Fable 5.2 one day before Anthropic IPO."


## Official claims (vendor-reported)


- **Koray Kavukcuoglu** ([thread](https://x.com/koraykv/status/2123167638050541626)):
  - Claims much improved steerability (#2 on Agent Arena Steering), strong multi-step instruction following, and "our lowest fabrication rate yet" in internal Antigravity testing.
  - Demo: Neon segmented a raw Smithsonian woolly-mammoth 3D scan into bones, rigged a skeleton, and generated a [museum walkthrough](https://x.com/koraykv/status/2123167642567762164).
- **Phil Schmid** ([@_philschmid](https://x.com/_philschmid/status/2123168015160320420)): "faster than Carbon and more capable than Hydrogen." Cites #3 on Artificial Analysis, #1 on CursorBench and #3 on Agents Last Exam, and says he "mostly use[s] Neon now."
- **Omar Sanseviero** ([@osanseviero](https://x.com/osanseviero/status/2123168220542632387)): faster, more token-efficient, better TTFT, and "better tuned thinking levels," iterated on user feedback.
- **Google** ([thread](https://x.com/Google/status/2123172292175822907)):
  - 50–80% lower cost per task than top competing frontier models.
  - "Best in class at science and math."
  - Demo: a /swarm reconstruction of Turing's 1936 King's College study.
- **Cybersecurity.** [@ralucaadapopa](https://x.com/ralucaadapopa/status/2123189881543561408) says Neon excels at cyber-defense tasks.
- **Google's own comparison.** Per [@TheRundownAI](https://x.com/TheRundownAI/status/2123169908338884712), Google's testing has Neon beating Claude Opus 5.5 and GPT-6.2 Sol on 16 of 19 benchmarks, at roughly ½ to ⅕ the cost per task. [@AndrewCurran_](https://x.com/AndrewCurran_/status/2123169127309410619) posted the benchmark table.
- **Other Google staff** ([@melvinjohnsonp](https://x.com/melvinjohnsonp/status/2123167943345545419), [@joshwoodward](https://x.com/joshwoodward/status/2123177835254952348), [@tulseedoshi](https://x.com/tulseedoshi/status/2123183416090984929), [@mirrokni](https://x.com/mirrokni/status/2123192430681887113), [@shimritby](https://x.com/shimritby/status/2123172748327637036), [@gracepursuits](https://x.com/gracepursuits/status/2123190512853176421), [@shaneguML](https://x.com/shaneguML/status/2123295455127953862)):
  - These echo "balance of speed/intelligence/usability."
  - Mirrokni attributes the cadence to agents helping researchers "optimize and ship models."


## Independent evaluations


**Artificial Analysis** ([main](https://x.com/ArtificialAnlys/status/2123170645777526962), [cost](https://x.com/ArtificialAnlys/status/2123170658129768801), [breakdown](https://x.com/ArtificialAnlys/status/2123170661015396386)):
- **Intelligence Index v5:** Neon (xhigh) scores 61, which is #3. That is behind GPT-6.2 Astra (max) and Claude Fable 5.2 (max), and +1 over Carbon (high).
- **Cost and time per task:**

  | Model | Cost/task | Time/task | Output speed |
  |---|---|---|---|
  | Neon (xhigh) | $6.44 | ~21 min | ~110 tok/s |
  | Carbon (high), standard pricing | $7.76 | ~36 min | ~57 tok/s |
  | GPT-6.2 Astra (max) | $4.00 | — | — |

  - Neon is about 17% cheaper and 40% faster than Carbon, but about 1.6x Astra's cost per task.
- **Neon at high effort** matches Carbon's score of 60 for $3.74 per task, under half of Carbon's cost.
- **Token use vs. price:** Neon uses about 15% more output tokens than Carbon. Its savings come entirely from 25% lower list prices ($3/$15 vs $4/$20).
- **Carbon's discount:** Carbon is currently discounted to $2/$10 until Nov 30, so Neon is 50% pricier than Carbon until then. Cached input on Neon costs $0.15 (a 95% discount).
- **Coding Agent Index (Antigravity CLI):**
  - Neon (xhigh) scores 59, vs Carbon 55 and Claude Code with Fable 5.2 (xhigh) / Opus 5.5 (max) at 58.
  - Cost per task: $6.86 for Neon vs $12.61 (Opus 5.5) and $15.47 (Fable 5.2).
  - [@kimmonismus](https://x.com/kimmonismus/status/2123221275594403932) notes this puts it 6 points above GPT-6.2 Astra.
- **Visual reasoning and presentation:**
  - GDP.pdf: Neon (high) 33.4% vs Carbon 28.4%. Notably, Neon at xhigh scored lower (28.0%).
  - AA-Briefcase: 1700 Elo overall. Presentation Quality leads at 1771 Elo, +45 over Claude Sonnet 5.5.
  - MKA Climb WSR: 56%, vs Fable 5.2 at 57%.
- **Modalities:** text, image, video and speech input; text output.

**Vals AI** ([index](https://x.com/ValsAI/status/2123168167879147542), [code migration](https://x.com/ValsAI/status/2123168182051762433)):
- #2 on the Vals Index at 70.9%, 0.4 pts behind Fable 5.2 at less than half the cost per task.
- #1 on the Finance and Legal indexes, Tax Agent and Legal Research.
- **Caveat on Code Migration:** Neon averaged 160 turns per task vs 234 for Carbon and 574 for Hydrogen, but scored *below both*. "Fewer steps don't always come with a higher score."
- **Context from Vals' Autoform-Hard results (Carbon, not Neon):** Gemini 4 Carbon "trails" GPT-6.2 Astra, Sol and Fable 5.2 on formal math ([@ValsAI](https://x.com/ValsAI/status/2123129597206450505)).

**Arena / LMArena** ([Agent Arena](https://x.com/arena/status/2123169189397709174), [Text Arena](https://x.com/arena/status/2123169199396946279)):
- **Agent Arena Index:** #2 at 1336 pts, from 4.2k real sessions. That is the best Google model, ahead of Carbon (#4, 1303), Hydrogen (#9, 1271) and Argon (#11, 1229).
- **Agent Arena cost frontier:** $2.82 median cost per task puts it on the Pareto frontier.
  - It beats #3 Opus 5.5 (Max) at 53% lower cost.
  - It costs 69% less than #1 Fable 5.2 (Max), which leads the index at 1463 ([index launch](https://x.com/arena/status/2123126646983332177)).
- **Agent Arena sub-signals:** #2 Praise vs Complaint (1539), #2 Steerability (1298), #3 Task Completion (1316).
- **Text Arena:** Google holds the top 3. Carbon 1526, Argon 1521, Neon 1518; Neon does not top its siblings here.
- **For comparison, GPT-6.2 Astra** lands at #6 (1291) on Agent Arena ([@arena](https://x.com/arena/status/2123234410032214464)).

**CursorBench 4.1** ([@cursor_ai](https://x.com/cursor_ai/status/2123198346802450926)):
- 63.7% at default effort, up from 57.3% for Hydrogen, setting a new cost-performance frontier.
- Echoed by [@melvinjohnsonp](https://x.com/melvinjohnsonp/status/2123188659562459256) and [@StringChaos](https://x.com/StringChaos/status/2123214284331303337).

**SWE-sweep** ([@KLieret](https://x.com/KLieret/status/2123172740043505968), [@18jeffreyma](https://x.com/18jeffreyma/status/2123171441919316292)):
- Google claims up to 39%, "blowing all previous results out of the water."
- With the neutral mini-swe-agent harness it gets only 12.8%, so the result is strongly harness-dependent.
- Open questions remain on cost and inference scaling.

**GitHub** ([changelog](https://x.com/github/status/2123181574116233646)): relays Google's claim of near-Carbon quality at lower token cost per task, with fewer unrequested code changes.


## Hands-on reviews


**Kimmonismus** (early access, Antigravity; [thread](https://x.com/kimmonismus/status/2123168299890704735)):
- Built 17 interactive demos.
- Bug-fix tests ([coding results](https://x.com/kimmonismus/status/2123168545303609385)):
  - Booking checker: 1/5 → 5/5.
  - Task scheduler: 8/14 → 14/14.
  - Async loader: 10/16 → 16/16.
  - Found all 4 planted bugs in a code review.
  - It self-reported two residual gaps: a leap-second (23:59:60) rejection and a TOCTOU folder-swap race.
- Log analysis across 13 xhigh sessions, 762 tool calls ([logs](https://x.com/kimmonismus/status/2123168557228011759)):
  - Breakdown: 313 reads, 266 edits, 104 new files, 43 shell commands, 13 sub-agents, 2 web searches.
  - Pattern: batch-read up to 16 files, write a single edit, re-read to verify; about 2–3 min per fix.
  - Instruction following: told 14 times to avoid the terminal, it ran zero shell commands in those runs.
  - It delegated browser testing to Flash sub-agents.
  - Token use: about 71M input (93% cache hits) and 1.8M output.
  - Weak spot was capacity: 57 "no capacity" errors and 4 stalled runs (possibly pre-launch load).
- Verdict: "great for prototypes and solid code fixes. Expect some follow-ups."

**Theo** (previously a vocal Gemini skeptic):
- Quickly turned positive: ["looks significantly more compelling than Carbon"](https://x.com/theo/status/2123207527064318174), ["I don't hate this model. Yet"](https://x.com/theo/status/2123224301948330114), ["This is SO READABLE?"](https://x.com/theo/status/2123225578753102079), ["might actually be a good model. Fuck."](https://x.com/theo/status/2123226794409287937).
- Escalated to ["Why the fuck is nobody talking about Neon?"](https://x.com/theo/status/2123237018813779970), ["They should have just named this Gemini 5"](https://x.com/theo/status/2123240990077268425) and a public [apology to Google](https://x.com/theo/status/2123255588570501585).
- Ran a cross-model test where GPT-6.2 Astra [accepted Neon's suggestions](https://x.com/theo/status/2123227832587546642), and posted a [prompt-based vote of confidence](https://x.com/theo/status/2123238365629931896).
- Framed Neon as a ["counter balance to GPT-6.2 Astra … so much more pleasant to work with"](https://x.com/theo/status/2123279922311692552).
- Now uses Neon as the [exclusive copywriter](https://x.com/theo/status/2123273253850628260) in his multi-model setup (Fable implements, Astra playtests).
- [@SicongJiang25](https://x.com/SicongJiang25/status/2123234446728171954) called it possibly the highest praise Theo has ever given a Gemini model; [@Willsquirrels](https://x.com/Willsquirrels/status/2123240526908711106) vouched for the reaction in person.

**Jeremy Howard** ([@jeremyphoward](https://x.com/jeremyphoward/status/2123280439314108762)): reverting Solveit's Astra/Sol to their pre-6.2 versions, adding "Neon is awesome!"

**Other reactions:**
- [@fchollet](https://x.com/fchollet/status/2123193532186005962): "Really impressive numbers at that price point."
- [@kimmonismus](https://x.com/kimmonismus/status/2123169294112698774): "Google… is back in the game" and ["Google did what OpenAI tried to achieve"](https://x.com/kimmonismus/status/2123221373510361290).
- Kimmonismus also says Opus 5.5 (high) [remains the best overall model](https://x.com/kimmonismus/status/2123251215320719663), with Neon a candidate to become [his favorite pending more testing](https://x.com/kimmonismus/status/2123251729185869941).


## Skeptical and critical views


- **Benchmark trust.**
  - [@scaling01](https://x.com/scaling01/status/2123175281414529060): "outscores Opus 5.5 and Sol 6.2 on basically every reported eval, surely it's a better model right?"
  - Scaling01 also says Gemini 3 left him with ["PTSD"](https://x.com/scaling01/status/2123176531409158419) about trusting Google benchmarks, though he credits Google for dropping "Preview" labels.
  - [@rasbt](https://x.com/rasbt/status/2123064851383677087) argues release benchmarks are like IQ tests for hiring, and that the real "job sample test" is [using the model](https://x.com/rasbt/status/2123117497209786374).
- **Distillation explanation.**
  - [@MParakhin](https://x.com/MParakhin/status/2123298795865968883) frames Neon as a smaller distillation ("Fable→Opus style"). It is cheaper to run, so it gets a larger reasoning budget, which helps on benchmarks, "though Carbon is better in real life."
  - He nonetheless defends Google for releasing its best rather than nerfing it.
- **The "RSI" framing.**
  - [@teortaxesTex](https://x.com/teortaxesTex/status/2123169782153503198) quipped "Cheaper faster better with shorter iterations… something like 'RSI'," and that the best news of the day is ["Google might have RSI too"](https://x.com/teortaxesTex/status/2123231224617648316).
  - He then argued nothing in the releases isn't explained by ["a ton of RL over parallel pretrains"](https://x.com/teortaxesTex/status/2123231480310616484).
- **Adoption gap.** [@zachtratar](https://x.com/zachtratar/status/2123246938586001763): if Google is Pareto-optimal, why does barely anyone use its APIs relative to OpenAI and Anthropic?
- **Naming confusion.** This was a recurring complaint.
  - Before launch, a [list of 8 variants](https://x.com/gabriberton/status/2122962708060647534) circulated: Argon, Carbon, Hydrogen, Krypton, Neon, Oxygen, plus "Carbon for cyber" and "Argon Cyber Permissive."
  - [@teortaxesTex](https://x.com/teortaxesTex/status/2123154096668712966): "Carbon builds on Argon, Neon builds on Hydrogen — most ridiculous naming convention award," followed by [joke roadmaps](https://x.com/teortaxesTex/status/2123197926390280427).
  - Further jabs from [@andrew_n_carr](https://x.com/andrew_n_carr/status/2123223374046671007) (full Google lineage list), [@nicdunz](https://x.com/nicdunz/status/2123182200078078420) and [@TheTuringPost](https://x.com/TheTuringPost/status/2123191243102450165). [@YiTayML](https://x.com/YiTayML/status/2123236405048647742) replied "Enjoy the element maxxing."
- **Capacity and reliability.** Kimmonismus's 57 capacity errors (above) are the main concrete operational complaint so far.


## Context


- **Competitive backdrop.** Neon landed one day after OpenAI's GPT-6.2 Astra/Sol, which drew mixed reviews over verbosity and communication style (see below), and on the eve of Anthropic's IPO. Many reviewers explicitly positioned Neon as the "pleasant to work with" counterweight to Astra's "capable but incoherent" profile. Theo frames this as [capability vs. coherency](https://x.com/theo/status/2123253181593587770).
- **Pareto frontier.** [@jeffwang](https://x.com/jeffwang/status/2123269541409009899) notes that for the first time OpenAI, Anthropic, Google and Cognition each have a model on the cost/performance Pareto curve.
- **Fast iteration.** [@andersonbcdefg](https://x.com/andersonbcdefg/status/2123220791366144039) reads the much-mocked Gemini 3.x Flash series as a preview of this fast-iteration cycle.
- **Antigravity orchestration.** [@ChrisRackauckas](https://x.com/ChrisRackauckas/status/2123043267268767902) argues Google's internal use of Antigravity, which he calls far ahead on orchestrating dozens of parallel issues, explains its velocity.
- **Unverified scale rumors.** These circulated without confirmation.
  - "Gemini 4 ≈ 12T params trained on 500B tokens" was relayed [mockingly](https://x.com/teortaxesTex/status/2123152907583344860).
  - Teortaxes joked about ["artistic taste emerges at ≈10T params"](https://x.com/teortaxesTex/status/2123164068240404977).
  - He argued Zartbot-style 500B-active / 300T-token models could be served [profitably at Gemini prices](https://x.com/teortaxesTex/status/2123267783123456246).
- **Coverage ahead.** ThursdAI is running an [Astra vs Neon episode](https://x.com/thursdai_pod/status/2123234811183788353) with Artificial Analysis.
**GPT-6.2 Astra and Sol: Strong Evals, Contested Usability**

- **Hard-task results**: Astra's strongest independent showings were in math and formal proof.
  - **MathArena**: Astra and Sol rank #1 and #2. On ArXivLean, Astra solves 42 of 53 problems ([Tim Gehr](https://x.com/TimGMath/status/2123067901280485621)).
  - **Lean proof**: Vals reports what it believes is the first Lean proof of Smoktunowicz's Gap Theorem. The run had no human intervention and passed an independent kernel replay ([Vals](https://x.com/ValsAI/status/2123129588066959380)). Astra reportedly avoided three deep results absent from Mathlib by defining its own "capacity" quantity ([Andrew Curran](https://x.com/AndrewCurran_/status/2123134533516378126)).
  - **Bug-fix eval**: In one independent test on 105 replanted bugs, Astra (max) scored 62.7 for $28.35. Fable 5.2 (max) scored 58 for $139.65 ([Paweł Huryn](https://x.com/PawelHuryn/status/2123011354843902128)).
- **Arena placements**: Astra is weaker on real-world agent sessions than on hard tasks.
  - **Code Arena WebDev**: Astra is #3 with 1781 points, at a 42% lower median cost than GPT-6 Astra ([Arena](https://x.com/arena/status/2123112692810764496)).
  - **Agent Arena**: Astra is only #6 with 1291 points. Sol is #11 at a $0.90 median cost per task, which puts it on the Pareto frontier ([Arena](https://x.com/arena/status/2123234410032214464)).
  - **AA Intelligence Index**: Astra (max) ranks at the top, according to Artificial Analysis' Neon comparison above.
- **Pricing**: A user-reported comparison puts Astra at $10/$50 per 1M tokens versus $8/$40 for GPT-6 Astra ([post](https://x.com/JamesMalsawm/status/2122910168098730290)).
  - **Background-mode discounts**: OpenAI says Responses API background mode cuts per-token prices by 30% with a 2h completion window and 40% with an 8h window. This is in public beta ([OpenAI Devs](https://x.com/OpenAIDevs/status/2123188195202617408)).
- **Usability dispute**: Early testers and OpenAI staff disagreed about how well the models work in practice.
  - **OpenAI's explanation**: Tibo Sottiaux said influencers tested pre-launch builds, before changes made in the last 3–4 days, and acknowledged gaps in "taste and niceness of use" ([Tibo](https://x.com/thsottiaux/status/2122929634916679820)).
  - **Theo's critique**: Theo found that Astra is thorough but misses the main point. In a disk-latency diagnosis, Opus led with the root cause while Astra did not ([Theo](https://x.com/theo/status/2122991778622263514)).
  - **Other complaints**: Jeremy Howard reported shallow answers from Sol even at xhigh ([Jeremy Howard](https://x.com/jeremyphoward/status/2123255152715215222)). Another user said Astra abandoned a goal too early ([dzhng](https://x.com/dzhng/status/2123016058131161476)).
- **Measured personality shift**: Responses are shorter, at 352 to 210 words for Astra. The "Admits Uncertainty" score fell from 96 to 24 for Astra and from 98 to 51 for Sol ([AA](https://x.com/ArtificialAnlys/status/2123236464003588136)).
- **Harness sensitivity**: Ramp Labs reports that native harnesses lifted older Codex models by about 10 points but barely changed the 6.2 models. Fable 5.2 was cheaper in its native harness ([Ramp Labs](https://x.com/RampLabs/status/2123114999874994596)).

**Open Weights, Harness Adaptation and Inference Systems**

- **Tencent Hy4 flash**: The model is now open-weight under the Tencent Hy License. Releases include FP8 and Base checkpoints. It has 224B total parameters with 12B active, a 1M-token context window and native multimodality ([Tencent](https://x.com/TencentHunyuan/status/2122912575302119554)).
  - **Architecture**: It uses hybrid attention, Latent MoE and a 64B-parameter Engram component. vLLM has day-zero support ([vLLM](https://x.com/vllm_project/status/2122917809948852559)).
  - **Hands-on test**: One Zhihu tester found Flash much faster than full Hy4. Its frontend and 3D output quality was closer to Hy4 Preview ([Zhihu Frontier](https://x.com/ZhihuFrontier/status/2122946375965085720)).
- **Harness-specific LoRA adapters**: Hugging Face released Apache-2.0 LoRA adapters for Qwen3.8-27B, trained with RL inside the real Pi and OpenCode harnesses.
  - **Pi adapter**: SWE-bench Pro (hard 51) rises from 39.2% to 64.7%. The original 29.4% baseline was revised after tunnel failures that affected nine base-model runs ([Turing Post recipe summary](https://x.com/TheTuringPost/status/2123092857464422482)).
  - **OpenCode adapter**: SWE-bench Pro rises from 35.3% to 51.0% with thinking off, using about one-eighth the tokens ([Lysandre](https://x.com/LysandreJik/status/2123053722888741340)).
  - **Local inference**: A merged adapter runs at base-model speed on an M5 Pro Mac mini. Unmerged LoRA costs about 40% in prompt-processing speed ([Pratik](https://x.com/pratikg/status/2123085869489987757)).
- **DaisyRose-27B**: RadixArk's RL-tuned Qwen3.8-27B was trained for 18 days on 144 H200s with sessions up to 1M tokens. It raised Terminal-Bench 4.1 from 5.6% to 12.0% ([RadixArk](https://x.com/radixark/status/2123115052127633536)).
- **Serving and capacity**: Several releases and measurements concerned inference efficiency.
  - **vLLM v0.34.0**: The release enables the extensible KV cache by default, makes Model Runner V2 the only GPU runner and reports up to 1.95× faster NVFP4 decode on GB200 ([vLLM](https://x.com/vllm_project/status/2123102778885276004)).
  - **Epoch inference index**: Agentic inference capacity grew 4.6× between Q2 2025 and Q2 2026, compared with 2.6× growth in compute. Blackwell accounted for 74% of the added capacity ([Epoch](https://x.com/EpochAIResearch/status/2123114112054112529)).
  - **Runway**: Runway cut the GPUs needed to serve a 57B video model at 24 fps from eight to four ([Runway](https://x.com/runwayml/status/2123060689053036600)).
- **Agent infrastructure**: New releases cover sandboxing and model-training stacks.
  - **Codex sandbox**: Network access is off by default, and a domain-rule proxy is shared across operating systems ([OpenAI Devs](https://x.com/OpenAIDevs/status/2123129469447880885)).
  - **Model Factory**: Prime Intellect launched a unified stack for RL, SFT, inference and evals ([Prime Intellect](https://x.com/PrimeIntellect/status/2123207653530972470)).
  - **PyBot dataset**: Hugging Face released PyBot's 1,000-agent, 184k-message swarm dataset ([Leandro](https://x.com/lvwerra/status/2123089416352829725)).

**Research: Measuring Self-Improvement, Chip Design and Safety**

- **OpenAI research-acceleration report**: OpenAI proposes that labs be required to report indicators of recursive self-improvement ([OpenAI](https://x.com/OpenAI/status/2123191365983248653)).
  - **Reported figures**: Agents now complete 46% of 2–4-hour research tasks without help, up from 23% in March. The median researcher uses about $2,400 a day in agents at API prices; the top 10% use about $23,000 ([summary](https://x.com/kimmonismus/status/2123195701786202603)).
  - **Third-party analysis**: One analyst estimates task time horizons are doubling every 3.8 months ([scaling01](https://x.com/scaling01/status/2123208816129085479)).
- **AI-assisted chip design**: OpenAI models found up to 14% PPA gains on AMD's flagship GPU IP in under five months. Some changes are in a taped-out design, and the work also improved OpenAI's Jalapeño inference chip, verified on silicon ([OpenAI](https://x.com/OpenAI/status/2123100386638877054), [Greg Brockman](https://x.com/gdb/status/2123130414323888463)). EDA tools still verify each RTL rewrite ([Austin Lyons](https://x.com/austinsemis/status/2123122170675040662)).
- **Self-improving harnesses**: Several papers tested agents that rewrite their own harnesses or learning procedures.
  - **Google, rewrite log**: A harness rewrite log let improvements carry over after swapping the underlying model, scoring 84.75 versus 73.42 without the log ([summary](https://x.com/kimmonismus/status/2123035118474609101)).
  - **Google, gated edits**: Gated one-module edits reached 40.0% on SWE-bench Lite at 46.7k tokens per task. A full rewrite reached 42.5% at 451.6k tokens. The test set has only 40 held-out issues ([DAIR summary](https://x.com/omarsar0/status/2123077276350976396)).
  - **MIT CSAIL/Tübingen**: Frozen models wrote their own practice games and built learning loops without being told how ([summary](https://x.com/omarsar0/status/2123129152887046557)).
- **CoT monitorability**: Training reasoning personas reduced a gpt-oss-20b organism's monitorability from 0.94 to 0.68 with no instruction to hide. With explicit instruction, it fell to 0.22 ([DAIR](https://x.com/dair_ai/status/2123065951256244262)).
- **OpenAI safety lifecycle**: OpenAI outlined four protection layers: alignment, sandboxing, blocking safeguards and monitoring ([Kai Chen](https://x.com/kaicathyc/status/2123146093366333850)).
- **New benchmarks**: New evals target coordination, self-improvement, security and judge bias.
  - **MAC Bench**: An eight-agent team beats a single agent through parallel exploration, not coordination. Only Opus 5.5 showed a coordination gain ([Bespoke](https://x.com/bespokelabsai/status/2123099143996616741)).
  - **Agent Arena indexes**: Arena added new agent indexes plus an Alignment Index ([Arena](https://x.com/arena/status/2123126646983332177)).
  - **Other new evals**: [FrontierRSI](https://x.com/quasicoh/status/2123114502065627580) targets self-improvement and [FrontierBreach](https://x.com/PromptArmor/status/2123114516498252054) measures enterprise agent attacks.
  - **LLM judges**: Eight judges each ranked their own answers higher than other judges did ([Turing Post](https://x.com/TheTuringPost/status/2123045050208727043)).
- **Security disclosure**: NGINX2Shell is a pre-auth RCE in NGINX 1.31.8 mainline, found together with OpenAI ([disclosure](https://x.com/ikotas00/status/2123071157083754677)).

**Media, Voice and World Models**

- **Genie 4**: Google's world model adds mid-session text edits, real-time native audio and a co-creation agent. It is rolling out to eligible Ultra subscribers. Genie 3 sessions now run up to three minutes ([GDM](https://x.com/GoogleDeepMind/status/2123117733709586849), [Google](https://x.com/Google/status/2123117819948638448)).
- **Open voice releases**: Two new speech models are open-weight under Apache 2.0.
  - **Inworld**: Inworld open-sourced its Voice Isolator and its SAVA turn-detection model, which detects completed turns in a median 71 ms versus 210 ms for an open baseline ([Inworld](https://x.com/inworld/status/2123120546908926300)).
  - **Rumble-2-Flash**: Reson8's speech-to-text model covers 40+ languages ([summary](https://x.com/omarsar0/status/2123060794124386461)).
- **AssemblyAI-STT-3.7**: AssemblyAI split transcription into three models: Realtime-Agent, with an 86 ms final transcript; Realtime; and pre-recorded ([AssemblyAI](https://x.com/AssemblyAI/status/2123080177508929849)).
- **Kling Image 4.0**: It ranks #7 in Text to Image and #6 in Image Editing on Artificial Analysis, up from #86 for Kling Image 3.0 Omni ([AA](https://x.com/ArtificialAnlys/status/2123242530489966738)).

**Industry and Policy**

- **Anthropic IPO**: Anthropic priced shares at $83, above its $70–78 range. Trading begins on Nasdaq tomorrow ([Anthropic](https://x.com/AnthropicAI/status/2123196808675405860)).
  - **S-1 compute data**: Anthropic spent $10B on compute in Q3, and customer inference was about 26% of compute spending ([Epoch](https://x.com/EpochAIResearch/status/2123159919042207823)).
  - **MisAnthropic Statement**: A CAIS-led statement says Anthropic is "not a safe AI company" ([CAIS](https://x.com/CAIS/status/2123069814663160304)). Signers include Thomas Wolf and Geoffrey Irving ([Thomas Wolf](https://x.com/Thom_Wolf/status/2123086773844672689), [Geoffrey Irving](https://x.com/geoffreyirving/status/2123156727835320636)). Dan Hendrycks also stopped advising AI companies ([Hendrycks](https://x.com/hendrycks/status/2123082552781218100)).
- **Nvidia leadership crisis**: The company is operating without Jensen Huang after his ouster.
  - **H200 sales to China**: Interim CEO Colette Kress said Nvidia stopped shipping licensed H200s to China. That business was worth $2.4B last quarter ([summary](https://x.com/kimmonismus/status/2123059489536647338)).
  - **Retention**: Pay raises and equity were moved up to December to retain staff ([WIRED via kimmonismus](https://x.com/kimmonismus/status/2123141684867617008)).
  - **Board criticism**: Bill Ackman argues the board breached its duty of care ([Ackman](https://x.com/BillAckman/status/2123101979375235386)).
- **China compute controls**: Datasection cut its revenue guidance by 89% after losing reported Chinese offtake for Blackwell capacity ([SemiAnalysis](https://x.com/SemiAnalysis_/status/2123130020063301813)).
  - **Disclosure proposal (unconfirmed)**: Per Reuters, the White House and Treasury are discussing requiring US companies to disclose use of Chinese open-weight models. It is under discussion only ([report](https://x.com/MTSlive/status/2123051208067928066)).
  - **ByteDance and Azure**: ByteDance is in reported talks to spend about $500M on OpenAI models through Azure ([The Information](https://x.com/leomschwartz/status/2123060707369374183)).
- **Acquisition**: Palo Alto Networks will acquire Applied Compute for $4.5B ([Yash Patil](https://x.com/ypatil125/status/2123175975093784902)).

**Top tweets (by engagement)**

- [@BillAckman: Nvidia board breached duty of care in Jensen ouster](https://x.com/BillAckman/status/2123101979375235386) — 78.3K
- [@thsottiaux: why GPT-6.2 influencer vibes diverged](https://x.com/thsottiaux/status/2122929634916679820) — 16.6K
- [@AnthropicAI: IPO priced at $83/share](https://x.com/AnthropicAI/status/2123196808675405860) — 13.0K
- [@sundarpichai: Gemini 4 Neon rollout](https://x.com/sundarpichai/status/2123167780988125621) — 12.4K
- [@OfficialLoganK: Introducing Gemini 4 Neon](https://x.com/OfficialLoganK/status/2123167791922741362) — 11.8K
- [@MTSlive: proposed disclosure of Chinese open-weight model use](https://x.com/MTSlive/status/2123051208067928066) — 5.5K
- [@OpenAI: AI-driven chip design with AMD](https://x.com/OpenAI/status/2123100386638877054) — 5.1K
- [@CAIS: MisAnthropic Statement](https://x.com/CAIS/status/2123069814663160304) — 4.5K


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Open Small Models and Agent Benchmarks

  - **[OVH announces Olala mini 7A1B](https://www.reddit.com/r/LocalLLaMA/comments/1xygtj4/ovh_announces_olala_mini_7a1b/)** (Activity: 406): ****OVHcloud announced [Olala mini 7A1B](https://corporate.ovhcloud.com/en/newsroom/news/ovh-olala-mini/)**, an Apache-2.0 7B-parameter MoE-style LLM with ~`1B` active parameters, `256` experts/layer, BF16 weights, and hybrid attention combining **Differential-TPA** with **Mamba-3**. OVH claims a `32k` context window, inference on as little as `8GB` VRAM, `160 tok/s` on an RTX A4500, and `50 tok/s` on CPU; it was reportedly trained from scratch on European supercomputers **Jupiter**, **MeluXina**, and **Jean Zay** using GH200/A100/H100 GPUs with ~`€7M` public compute support, and is already deployed at **Qwant**.** Commenters were skeptical because the announcement shipped with *no benchmarks and no weights*, making the performance claims hard to evaluate. Others noted that `32k` context feels dated for 2025, though they welcomed another small MoE option; one commenter also criticized the increasingly meme-like French model naming.

    - Several commenters noted that **Olala mini 7A1B** lacks the key artifacts needed for technical evaluation: no published **benchmarks**, no **weights**, and no Hugging Face link. The consensus was that without model cards, evals, or downloadable checkpoints, the OVH post is effectively a pre-announcement rather than a usable model release.
    - The only concrete technical detail discussed was the model’s **`32k` context window**, which commenters characterized as underwhelming for late 2024/2025 expectations, especially compared with recent Chinese open-weight models pushing longer contexts and stronger small-model performance. One commenter still welcomed it as “another small MoE option,” implying interest in the **7A1B mixture-of-experts** configuration if weights and benchmarks are later released.

  - **[SenseNova-Loop-9B vs its own Preview: AutomationBench 14.33 to 36.67, GAIA 37.86 to 69.90 (9B Looped Transformer)](https://www.reddit.com/r/LocalLLaMA/comments/1xymz9z/sensenovaloop9b_vs_its_own_preview/)** (Activity: 391): **The [image](https://i.redd.it/y6w814wqv62i1.png) is a benchmark bar chart for **SenseNova-Loop-9B**, a released Apache-2.0 “looped transformer” with `32` stored layers where `24` layers are executed twice, giving roughly `56` layers of compute over ~`9B` parameters. It highlights large gains over the Preview across seven agent/code/data benchmarks, including **AutomationBench** `14.33 → 36.67`, **Terminal-Bench 2.1** `40.45 → 59.55`, and **GAIA** `37.86 → 69.90`, with the post noting its AutomationBench score matches the listed **Qwen3-8B/27B** baseline. The release links include the [HF model card](https://huggingface.co/sensenova/SenseNova-Loop-9B) and [OpenSenseNova/SenseNova-Loop](https://github.com/OpenSenseNova/SenseNova-Loop) with vLLM/SGLang support; comments mention GGUF files are expected and one user reports ~`50 tok/s` on an RTX 5090 under vLLM.** Commenters are generally positive about the architecture’s efficiency, saying it “definitely performs better than a regular 9B,” though one still prefers a 27B model for quality. One top comment is a GIF joke/non-technical aside, so not technically meaningful.

    - A tester reported SenseNova-Loop-9B “definitely performs better than a regular 9B” in quick trials and measured roughly `~50 tok/s` on an **RTX 5090** using **vLLM**. They still preferred the `27B` model overall, suggesting the looped 9B improves small-model capability but may not fully close the quality gap to larger dense models.
    - One technical concern was the **KV-cache footprint** at the advertised `262K` context length: a commenter noted that running the 9B model at full context can require memory comparable to a `27B` model at the same context length. This raises a practical deployment question about whether the smaller parameter count meaningfully reduces total serving memory when long-context inference dominates VRAM usage.
    - Users noted that **GGUF** files were expected to be released, which would make the model more accessible for local inference stacks such as llama.cpp-based runtimes. There was also interest in whether the same “looped transformer” idea could be combined with **MoE** architectures in future models.

  - **[[Harness Adapters] Fine tune Qwen3.8-27B for Pi Agents](https://www.reddit.com/r/LocalLLaMA/comments/1xykw4l/harness_adapters_fine_tune_qwen3827b_for_pi_agents/)** (Activity: 260): **The GIF/image ([link](https://i.redd.it/lpesdq2kf62i1.gif)) is a technical comparison for Hugging Face’s **Harness Adapters** release: a LoRA adapter `harness-hub/Qwen3.8-27B-pi-GGUF` on top of `ggml-org/Qwen3.8-27B-GGUF` intended to improve **Pi agent** behavior in a specific harness. It shows the adapter performing a more targeted SWE-bench Pro workflow on an Ansible `human_to_bytes` bug—finding, reading, editing, and testing the relevant function—while the base model keeps doing broader search/read steps. The post provides a local `llama.cpp` serving recipe using `--lora`, context `-c 65536`, and `--chat-template-kwargs '{"enable_thinking": false}'`, consistent with the claim that the adapter was trained/evaluated with thinking disabled.** Comments questioned whether disabling `enable_thinking` is actually required or desirable, with one skeptical reply to the statement that “thinking stays off.” Another technical question asked how much benefit RL provides over SFT on successful harness traces, referencing an alternative Qwen3.8-27B Pi adapter.

    - Several commenters question disabling Qwen `enable_thinking` for a LoRA that was apparently trained/evaluated with thinking-mode chat templates. One cites the HF text saying *“we leave thinking mode for all our models, in training and in evaluation”*, arguing that setting `{"enable_thinking": false}` may still run but creates a distribution mismatch because the adapter was not trained on empty `<think>...</think>` prefixes in context.
    - There is a technical request for empirical comparison between **RL** and simply doing **SFT on successful Harness traces**, specifically asking what gains RL provides over a trace-based fine-tune such as [`bytkim/Qwen3.8-27B-pi-GGUF`](https://huggingface.co/bytkim/Qwen3.8-27B-pi-GGUF). Multiple comments emphasize that benchmarks are needed to justify claims, especially if reasoning/thinking behavior is altered.
    - Users ask whether the adapter requires disabling `enable_thinking` and whether it works across arbitrary fine-tunes or quantizations of **Qwen3.8-27B**. The implicit concern is compatibility between the LoRA adapter, the base model/chat template, and GGUF or other quantized variants, particularly when inference-time template settings differ from training.


### 2. Local Audio Model Releases

  - **[Blue TTS v3: open-source TTS, about 50x real-time on a CPU](https://www.reddit.com/r/LocalLLaMA/comments/1xytrz9/blue_tts_v3_opensource_tts_about_50x_realtime_on/)** (Activity: 535): ****Blue TTS v3** is an MIT-licensed open-source local TTS model with [code](https://github.com/maxmelichov/BlueTTS), [ONNX weights](https://huggingface.co/notmax123/BlueV3-onnx), [PyTorch weights](https://huggingface.co/notmax123/BlueV3), and a [HF demo](https://huggingface.co/spaces/notmax123/BlueV3). The release claims substantially improved naturalness/expressiveness vs v2, **MOS `4.43/5`**, improved voice cloning, `44.1 kHz` output, `388 MB` model size, support for `7` languages, and CPU-only inference at roughly **`50x` real-time**.** A technical commenter suggested considering a **ggml/audio.cpp** backend for broader deployment, noting reported cases where [`audio.cpp` BF16 outperformed ONNX int8 on CPU](https://github.com/0xShug0/audio.cpp/issues/1310#issuecomment-6958142439). Other comments were skeptical of the claims or criticized sample quality, e.g. saying it *“sounds like it’s in a leaky bathroom”* and asking what the catch is.

    - A commenter suggested a **ggml/audio.cpp implementation** to improve backend portability and cited a CPU performance comparison where `audio.cpp` **BF16 reportedly outperformed ONNX Int8** for PocketTTS. They linked a GitHub issue noting the reported numbers were actually **RTFx**, where higher is better: https://github.com/0xShug0/audio.cpp/issues/1310#issuecomment-6958142439
    - A technical concern was raised that the claimed **~`50x` real-time CPU synthesis speed** is not interpretable without hardware and threading details. The commenter asked which CPU was used and whether the measurement used **single-threaded inference or all cores**, since CPU TTS benchmarks vary heavily with core count and parallelism.

  - **[New voice isolation model from Inworld](https://www.reddit.com/r/LocalLLaMA/comments/1xzqpzu/new_voice_isolation_model_from_inworld/)** (Activity: 453): ****Inworld** released an **Apache-2.0** [`voice-isolator`](https://huggingface.co/inworld/voice-isolator) model for speech source isolation at **`48 kHz`**, with two variants: a `64M` offline/non-streaming model operating on `8 s` chunks and a `58M` realtime model operating on `20 ms` chunks. It was reportedly trained on ~`10k hours` of audio and is positioned as a cheap/fast **voice isolation** model rather than a speech enhancement/restoration system like Resemble Enhance; a demo is available via this [HF Space](https://huggingface.co/spaces/mrfakename/voice-isolator).** Commenters noted that recent FOSS voice isolation options had been weak compared with commercial tools such as DaVinci Resolve’s built-in isolation, and suggested benchmarking against NVIDIA’s commercially friendly [`NVIDIA-NemotronLabs-RE-USE`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) model.

    - A commenter pointed to **NVIDIA NemotronLabs RE-USE**, a recently released commercially friendly voice/isolation model on Hugging Face: [nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE). They specifically raised the technical question of how Inworld’s new model compares against RE-USE for speech isolation tasks.
    - Several users framed the model around practical post-production workflows: isolating dialogue, re-voicing/redubbing it, and then mixing it back over the original music bed. One user proposed testing this on consumer GPUs including an **RTX 4070** and **RX 9070 XT**, implying interest in local inference performance and whether the separation quality is sufficient for video redubbing pipelines.
    - A user noted that as of roughly `~6 months` ago, open-source voice isolation options were still weak, and even **DaVinci Resolve’s built-in voice isolation** was reportedly not producing results as good as this new model. Another technical question focused on whether the model can remove **background voices**, which is a harder variant than generic noise/music separation because interfering speech overlaps spectrally with the target voice.

  - **[AudioGhost v1.2: remove any sound from audio or video by typing what it is (local, open source, built on Meta's SAM-Audio)](https://www.reddit.com/r/LocalLLaMA/comments/1xz6xl7/audioghost_v12_remove_any_sound_from_audio_or/)** (Activity: 402): ****AudioGhost v1.2** is a local, open-source web app built around **Meta SAM-Audio** that lets users separate or remove arbitrary text-described sounds from audio/video, e.g. `"background music"`, `"dog barking"`, or `"people talking"`, with export to `WAV`/`FLAC`/`MP3` or remuxed video via the updated soundtrack ([GitHub](https://github.com/0x0funky/audioghost-ai)). v1.2 adds multi-sound extraction into separate tracks with mute/solo/volume controls, timeline guidance for positive/negative sound regions, persistent sessions, a newly generated “everything else” stem rather than simple subtraction, and claims `4–6x` speedup over the initial release. The maintainer notes the stripped-down pipeline omits SAM-Audio helper models such as **ImageBind**, **CLAP**, and a judge/reranker to avoid ~`25 GB` extra downloads and high VRAM use; the small model reportedly fits in just over `4 GB` VRAM on NVIDIA GPUs, while macOS currently runs CPU-only.** Commenters focused on hardware/model limitations: AMD users objected to the NVIDIA/CUDA requirement, while another user questioned whether the tool improves over raw SAM-Audio for difficult overlapping multi-speaker extraction, especially similar male voices with time-window hints. Another commenter compared it favorably in concept to task-specific separators like **MVSEP**, noting that arbitrary text-prompted sound separation could cover uncommon classes such as dog barking better than many conventional open-source separation models.

    - An AMD user noted the project appears to require **NVIDIA/CUDA**, implying limited accessibility for ROCm/AMD GPU users and likely no documented CPU/AMD fallback path. This is a practical deployment constraint for local audio/video processing tools built around GPU-heavy separation models.
    - One commenter compared this against **Meta SAM-Audio** for speaker extraction in dense multi-speaker clips, specifically cases with `3–4` overlapping male speakers even when speaker time windows were provided. They reported SAM-Audio produced insufficiently clean cuts for their use case and asked whether AudioGhost improves on native SAM-Audio for overlapped speech separation.
    - A user with experience using **MVSEP** and multiple open-source source-separation models highlighted AudioGhost’s potential advantage: prompt-based removal of arbitrary sounds such as dog barking, rather than being limited to common predefined stems or categories. Another commenter asked whether it can replace **Adobe Podcast**-style local noise cleanup for general audio denoising.


### 3. Local Inference Hardware and Engines

  - **[SSD MoE at high speed - The wait is over!](https://www.reddit.com/r/LocalLLM/comments/1xzmymn/ssd_moe_at_high_speed_the_wait_is_over/)** (Activity: 594): ****Run My AI** announced **TokenWing**, an MoE inference engine that claims to offload `>60%` of routed experts to **SSD reads** while maintaining low latency. User-reported laptop benchmarks include **Ling 3.0 Flash 120B MXFP4** on `12 GB VRAM + 32 GB DDR5` at `45–50 tok/s` decode and `~900 tok/s` prefill, **Qwen 3.8 Flash Next 3.5 bpw** on `8 GB VRAM + 32 GB DDR5` at `35–40 tok/s` decode and `600–700 tok/s` prefill, and **Flint 40B UD-Q4_K_XL** on `4 GB VRAM + 24 GB DDR4` at `~30 tok/s` decode and `~400 tok/s` prefill. The project is early-stage, with a WIP autotuner; the author says SSD wear should be negligible because the engine performs reads only, not writes, and links to [runmyai.org](https://runmyai.org).** Comments were cautiously positive but skeptical about novelty, noting SSD/offload-based MoE inference is increasingly common and other projects may have better support. One benchmarker reported a `5090` over Thunderbolt with `64 GB DDR5-4600`: compared with an in-memory 2-bit baseline at `46 tok/s` decode and `1087 tok/s` prefill on a `19k` prompt, TokenWing with a 3.5-bit model reached `60–70 tok/s` decode but lower `833 tok/s` prefill, suggesting faster generation at the cost of slower prompt processing.

    - A commenter reports concrete throughput numbers using a **5090 over Thunderbolt** with `64GB DDR5-4600`: their prior **2-bit in-memory** setup produced `46 tok/s` generation and `1087 tok/s` prompt processing on a `19k` token prompt, while this software with a **3.5-bit** version produced `60–70 tok/s` generation and `833 tok/s` prompt processing. They conclude generation is significantly faster, prompt ingestion is slower, and request a **DeepSeek v4.1 1017** build to benchmark next.
    - Several commenters ask for missing technical specs and benchmarks, especially **context length** support for agentic / long-running workloads and reproducible performance data. One also asks whether the inference engine supports older **NVIDIA Pascal SM60** GPUs, implying compatibility with pre-Volta CUDA architectures is unclear.

  - **[Intel Shares First Benchmarks of Its Next-Gen Crescent Island "D270" GPU, Claiming a 5.9x Throughput Over RTX PRO 6000 As Xe3P Goes Big On FP64](https://www.reddit.com/r/LocalLLaMA/comments/1xyioi8/intel_shares_first_benchmarks_of_its_nextgen/)** (Activity: 772): **[Wccftech reports](https://wccftech.com/intel-first-benchmarks-crescent-island-d270-xe3p-gpu-a-5-9x-faster-rtx-pro-6000-fp64/) Intel’s pre-production **D270 “Crescent Island” Xe3P GPU** is a `350 W` accelerator with `32 Xe cores`, `256 XMX engines`, native/full-rate **FP64**, and up to `480 GB` LPDDR5X. Intel claims up to **`5.9×` FP64 throughput** vs **NVIDIA RTX PRO 6000 Blackwell Server Edition** in finance benchmarks, but the locally relevant AI result is much smaller: **`29%` lower latency** on Drex 2.0 and **`1.12×` LLM throughput** on Qwen 3 30B-A3B. Commenters highlighted the reported memory subsystem—`160 GB` LPDDR5X, `1280-bit` bus, `1.50 TB/s` bandwidth—as potentially notable for inference/HPC despite using LPDDR5X rather than HBM.** Top comments were skeptical that the headline FP64 comparison is meaningful because RTX PRO 6000 is not optimized for FP64, making the `5.9×` claim a favorable benchmark selection rather than a broad GPU comparison. Several also questioned the relevance of FP64 to AI inference, where FP16/BF16/INT8/FP8 are typically the important paths.

    - Several commenters argued the headline comparison is narrowly chosen: **Intel’s claimed `5.9x` throughput over RTX PRO 6000** appears to be for **FP64-specific workloads**, while NVIDIA’s **RTX PRO 6000** is not primarily designed around high FP64 throughput or dedicated FP64 tensor support. The takeaway was that this benchmark may be more indicative of Crescent Island/D270’s HPC-style FP64 positioning than general AI or graphics performance.
    - The most concrete technical specs highlighted were **`160 GB` LPDDR5X**, a **`1280-bit` memory bus**, and **`1.50 TB/s` bandwidth**. Commenters considered the bandwidth notable because achieving `1.5 TB/s` from LPDDR5X is unusually high, suggesting a very wide memory subsystem rather than HBM.
    - There was skepticism about the relevance of **FP64 for AI**, with commenters noting that local/consumer AI workloads tend to prioritize lower precision formats such as **NVFP4**, FP8, or similar quantized inference formats. The discussion framed Crescent Island’s large VRAM capacities—speculated at `160–480 GB`—as potentially interesting for model hosting, but only if the software stack and low-precision performance are competitive.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. OpenAI GPT-6.2 Astra/Sol Release Benchmarks

  - **[Introducing GPT-6.2 Astra and Sol](https://www.reddit.com/r/singularity/comments/1xyr3ia/introducing_gpt62_astra_and_sol/)** (Activity: 2515): ****OpenAI** announced [**GPT‑6.2 Astra** and **GPT‑6.2 Sol**](https://openai.com/index/introducing-gpt-6-2-astra-sol/): Astra is presented as a new frontier model with gains in long-horizon agentic workflows, coding, science, professional document generation, visual/web/3D design, and cybersecurity, while Sol is a lower-cost everyday model retaining GPT‑6.1 Sol pricing. The release claims Astra sets new highs on **Agents’ Last Exam**, **Terminal-Bench-Science**, **Harvey LAB‑AA**, **GDPval v2**, and **AA‑3D**, with fewer output tokens and lower API pricing than GPT‑6 Astra; it also notes cyber capabilities reaching OpenAI’s **“Critical” preparedness threshold**, with stricter safeguards that may affect legitimate security research.** Commenters focused on price/performance: one argued OpenAI is dominating the **perf/price Pareto frontier**, saying Astra is “barely behind Fable 5.2” at much lower cost and Sol matches “Opus 5.5” for a fraction of the price. There was speculation that OpenAI’s models are significantly smaller than Anthropic counterparts, implying substantial scaling headroom if OpenAI trained a model as large as Fable.

    - Commenters highlighted that **GPT-6.2 Astra/Sol** appear to push the **performance-per-dollar Pareto frontier**, with Astra described as only slightly behind **Fable 5.2** while costing a fraction as much, and Sol reportedly matching **Opus 5.5** at similarly reduced cost. One commenter inferred that OpenAI’s models may be **substantially smaller than Anthropic’s counterparts**, suggesting stronger efficiency/scaling characteristics if comparable benchmark performance is achieved with fewer parameters or lower inference cost.

  - **[Sol 6.2 and Astra 6.2 just released](https://www.reddit.com/r/ClaudeCode/comments/1xysw2v/sol_62_and_astra_62_just_released/)** (Activity: 1086): **The image is a **non-meme benchmark bar chart** from Artificial Analysis comparing recently released **GPT-6.2 Astra** and **GPT-6.2 Sol** against other frontier models: [image](https://i.redd.it/2ew69thiz72i1.jpeg). In the chart, **Claude Fable 5.2** leads with a score of `65`, followed by **GPT-6.2 Astra** at `62`, **Gemini 4 Carbon** at `60`, and **GPT-6.2 Sol** at `57`, supporting the post’s claim that Astra is competitive near the top while Sol trails stronger models like Opus/Fable-tier systems. The selftext explicitly excludes cost-per-task analysis, so the comparison is focused on raw benchmark score rather than efficiency or pricing.** Comments push back on ignoring cost, noting that usage limits and task cost are a major concern for the subreddit. There is also disagreement over model ranking, with one commenter claiming **Opus 5.5** remains better than both Fable 5.2 and GPT models.

    - A recurring technical concern is **effective cost per task**, with users implicitly tying model economics to practical **usage limits** rather than headline benchmark scores. One commenter argues the community often dismisses cost-per-task analysis while simultaneously complaining about capped usage, suggesting that throughput, pricing, and quota design may matter as much as raw model capability.
    - Several comments frame the desired target as **“Opus 5.5-level” quality with lower latency and cost**, implying current frontier models are perceived as capability-competitive but still constrained by inference speed and price. There are model-preference comparisons among **Opus 5.5**, **Fable 5.2**, and **GPT**, but no concrete benchmark data or task-specific evidence is provided.

  - **[Something is seriously wrong with 6.2 Astra](https://www.reddit.com/r/OpenAI/comments/1xyyvnx/something_is_seriously_wrong_with_62_astra/)** (Activity: 1453): **The [image](https://i.redd.it/7qed1kbk592i1.png) is a screenshot of an X/Twitter post alleging that early testers of **“6.2 Astra”** independently prompted the model to create game levels and received unusually convergent outputs: water/music/bell-themed levels named **“Bellwether.”** In the Reddit context, commenters add anecdotal evidence that *Bellwether* may be a recurring latent attractor across related models, with one user saying both **Astra 6** and **Sol 6.1** used the same name for a fictional town in a separate story-generation prompt.** The main technical concern raised is that the model may be “regressing to the mean” in open-ended creative tasks, producing repetitive or statistically overrepresented motifs rather than diverse outputs. The evidence is anecdotal, and the comments do not include rigorous benchmarking or reproduction details.

    - Multiple commenters report a recurring **“Bellwether” / bell-tower / lantern** motif across open-ended creative generations, suggesting this is not isolated to **Astra 6.2**. One user says prior tests asking models to write about *“a cartographer who receives a map that shows a town that doesn’t actually exist”* caused both **Astra 6** and **Sol 6.1** to independently name the town **Bellwether**, implying a latent trope or training-distribution attractor rather than a new regression.
    - Another commenter claims they observed the same motif cluster months before Astra 6.2, including outputs from projects with names like **The Bellwether**, **The Bell Vault**, **The Ninth Bell**, **Lantern Market**, **Echo Docks**, and **Ashling**. They argue this reflects broader LLM convergence on oddly specific fantasy tropes under open-ended prompts, possibly affecting multiple models including **Claude**, rather than being exclusive to one release.


### 2. Gemini 4 Neon Launch and Pricing Signals

  - **[Introducing Gemini 4 Neon.](https://www.reddit.com/r/singularity/comments/1xzrjzx/introducing_gemini_4_neon/)** (Activity: 1044): ****Google introduced [Gemini 4 Neon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-neon/)**, a `1M`-token-context frontier model targeted at low-latency agentic workflows in software engineering, enterprise knowledge work, science, and cybersecurity. Pricing is listed as `$3/M` input tokens, `$0.15/M` cached input tokens, and `$15/M` output tokens, with claimed strong benchmark placement including **#1 on Artificial Analysis Coding Agent Index**, **#2 on Vals Index**, and **#3 on Artificial Analysis Intelligence Index** at xhigh. Google also highlights production-agent features such as multi-file code editing, long-running tool use, controllable “thinking” levels, cyber-defense variants, prompt-injection resistance, misuse prevention, and reduced fabrication/shortcutting.** Commenters focused less on benchmark details and more on release velocity, calling Google’s cadence *“rapid fire releases”* and *“crazy.”* One commenter speculated that leadership changes around **Demis Hassabis** may reflect a shift from safety caution toward more aggressive deployment, though this was conjectural rather than technical evidence.


  - **[Gemini 4 Neon coming today, it sits right between Carbon and Hydrogen and can be the 3.1 Pro replacement](https://www.reddit.com/r/GeminiAI/comments/1xza6pu/gemini_4_neon_coming_today_it_sits_right_between/)** (Activity: 1064): **The image is a screenshot of an X post claiming **“Gemini 4 Neon”** is showing up in **Google Cloud billing** and the **Antigravity CLI**, implying an imminent launch as a possible **Gemini 3.1 Pro replacement**. The alleged model is positioned between Google’s Carbon/Argon and Hydrogen/Oxygen tiers, with claimed pricing around `$3/M` input and `$15/M` output tokens, half-price batch/flex pricing, higher priority pricing, and multimodal inputs including text, image, video, and audio. Image: https://i.redd.it/j1x30a3w1c2i1.png** Comments were mostly non-technical jokes or complaints about Google’s increasingly confusing model naming scheme, e.g. “passing gas” and frustration over names like Carbon, Neon, Hydrogen, Argon, Helix, Ether, and Obsidian.

    - Commenters questioned the practical model-routing niche for **Gemini 4 Neon** if it sits between **Hydrogen** and **Carbon**: Hydrogen is described as already fast and capable on moderately complex tasks, while Carbon is positioned for complex/deep work. The main technical concern is whether Neon offers a clear latency/cost/quality tradeoff or benchmark advantage sufficient to justify choosing it over the existing lower- and higher-tier models.


### 3. Coding Model Cost and Hardware Benchmarks

  - **[Microsoft says its new MAI Code 2 does 96% of Claude Opus 5.5’s work for 5% of the price](https://www.reddit.com/r/ClaudeAI/comments/1xyt61r/microsoft_says_its_new_mai_code_2_does_96_of/)** (Activity: 1696): ****Microsoft** announced **MAI Code 2** at Ignite 2026, claiming `63.2%` on selected coding benchmarks—roughly **`96%` of Claude Opus 5.5** and **`98%` of GPT-6.1 Sol**—at about **`5%` of Opus 5.5’s task cost**, though the pricing comparison depends on a **50% promotional discount** ([summary](https://madrobot.blog/2026/11/17/microsoft-mai-code-2-ignite-cheaper-openai-anthropic/)). The broader MAI lineup includes specialized coding, cyber, image, realtime voice, transcription, and routing models; **MAI Code 2** is already in **GitHub Copilot**, while other models have narrower availability such as Foundry preview or Microsoft security-product integration.** Top comments were highly skeptical of Microsoft’s LLM quality claims, arguing that the missing `4%` could represent the difference between usable software and failure, or even “a major outage.” One commenter cited poor experiences with Microsoft Copilot, including Azure-integrated Copilot allegedly failing to answer Azure questions, concluding Microsoft remains far behind Anthropic/OpenAI.

    - Several commenters challenged the headline’s `96%` work / `5%` price framing, arguing that the missing `4%` may represent the critical reliability gap between a usable coding agent and *“hot garbage”*. The implied technical concern is that aggregate task-completion percentages can hide failure modes on edge cases, integration work, debugging, or production-quality correctness.
    - One user reported a negative longitudinal experience with **Microsoft LLM products**, saying that after trying them over roughly `2 years`, **Copilot in the Azure UI** still could not reliably answer Azure-related questions as recently as a few months ago. The criticism suggests concern about Microsoft’s model/product integration quality, especially for domain-specific cloud support where first-party context should be an advantage.

  - **[Thank dog for Claude Code](https://www.reddit.com/r/StableDiffusion/comments/1xywykn/thank_dog_for_claude_code/)** (Activity: 429): **The image is a screenshot of a tech news claim that **Anthropic’s Claude Code/“Claude Mythos”** reduced AMD GPU onboarding work from *up to `12 months`* to roughly *`3 days`* by autonomously writing most GPU kernels, implying a major acceleration in porting workloads away from **Nvidia CUDA** dependency. In context of the post title “Thank dog for Claude Code,” the technical significance is Claude being presented as a practical code-generation tool for low-level accelerator enablement rather than just general app coding. [Image](https://i.redd.it/1kj2fqb6q82i1.png)** The framing is strongly pro-Claude Code, treating the reported AMD-kernel porting speedup as evidence that AI coding tools could weaken Nvidia’s CUDA moat.


  - **[Qwen 3.8 27B did the heavy lifting for GPT-6.2 Astra and cut the bill from $7.40 to $1.62](https://www.reddit.com/r/Qwen_AI/comments/1xytns6/qwen_38_27b_did_the_heavy_lifting_for_gpt62_astra/)** (Activity: 325): **OP reports using **Atomic Agent**’s [Fusion mode](https://atomicagent.io/) to orchestrate a local **Qwen 3.8 27B** model as the implementation agent while **GPT-6.2 Astra** acted only as planner/reviewer for a complex single-file **Three.js** animation task: shader-based sun, particles, camera path, scripted `25s` timeline, frame self-checks, aurora/orrery/eclipse scenes, and debugging via browser inspection. Reported cost/time: **Qwen local + Astra planner** cost `$1.62` and took `37:42`, versus **Astra-only** at `$7.40` and `12:06`, implying ~`4.6x` lower API cost but ~`3x` slower wall-clock due to local token generation.** Commenters were skeptical of the reproducibility claims, saying prior attempts with the harness did not work as advertised and asking OP to publish the **exact setup, prompts, and configuration**; the dominant technical criticism was essentially *“share everything or go home.”*

    - A commenter reported failing to reproduce the OP’s claimed harness results and asked for the exact setup and prompts, emphasizing that the cost/performance claim is not credible without reproducible configuration details. The technical concern is that prompt templates, routing logic, model parameters, and evaluation harness specifics could materially affect whether **Qwen 3.8 27B** can reduce a **GPT-6.2 Astra** workflow cost from `$7.40` to `$1.62`.
    - Another commenter asked whether the proposed “atomic agent” performs better than **Codex**, what concrete benefits it provides, and whether it is open source. This highlights the need for a direct technical comparison against existing coding-agent baselines, including implementation transparency, benchmark methodology, and licensing/source availability.




---
id: MjAyNS0x
title: not much happened today
date: '2026-10-02T05:44:39.731046Z'
description: >-
  **OpenAI** launched **GPT-6.1 Sol** priced at $2/$10 per million tokens,
  outperforming GPT-6 Sol and Astra on benchmarks like DeepSWE v1.1 and
  AutomationBench. **Sonnet 5.5** debuted strongly in Agent Arena, ranking #3
  overall and #1 in Chat, while **Anthropic** models hold the top three spots.
  **Gemini 4 Argon** topped Text Arena, and open models like MiMo-V2.6-Pro and
  Flash entered Agent Arena rankings. Independent evaluations highlight Sol's
  token efficiency and Sonnet 5.5's superiority over Opus 5. Rumors suggest a
  forthcoming **Claude Fable 5.5** and a possible **GPT-6 Astra Lite** linked to
  Sol. The **llama.cpp** ecosystem added a new endpoint for local decision-model
  inference, with models like Kev 1.0 and pplx-decider-v1-27b showing strong
  benchmark performance. Skepticism remains about decision models being
  rebranded zero-shot classifiers.
companies:
  - openai
  - anthropic
  - google-deepmind
  - llamaindex
  - perplexity-ai
  - ollama
models:
  - gpt-6.1-sol
  - gpt-6-sol
  - astra
  - sonnet-5.5
  - opus-5.5
  - gemini-4-argon
  - mimo-v2.6-pro
  - flash
  - claude-fable-5.5
  - gpt-6-astra-lite
  - kev-1.0
  - pplx-decider-v1-27b
topics:
  - benchmarking
  - cost-efficiency
  - token-efficiency
  - agent-arena
  - decision-models
  - local-inference
  - model-evaluation
  - reasoning
  - context-windows
people:
  - reach_vb
  - badlogicgames
  - htihle
  - designarena
  - kimmonismus
  - scaling01
  - ggerganov
  - clementdelangue
  - jaredpalmer
  - aravsrinivas
  - lucataco
  - mervenoyann
---


**a quiet day.**

> AI News for 10/1/2026-10/2/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**GPT-6.1 Sol and Sonnet 5.5 Reshape the Cost–Performance Frontier**

- **GPT-6.1 Sol launch**: OpenAI priced Sol at $2/$10 per million input/output tokens, compared with $10/$50 for Astra ([pricing summary](https://x.com/dl_weekly/status/2106096982024429590)).
  - **Claimed results**: It reportedly beats GPT-6 Sol by 6.4 points on DeepSWE v1.1 and Opus 5.5 by 2.2 points on AutomationBench ([summary](https://x.com/dl_weekly/status/2106096982024429590)).
  - **Positioning**: OpenAI staff describe it as "good, cheap AND fast" ([@reach_vb](https://x.com/reach_vb/status/2106868331039510878)).
  - **Codex usage**: A global Codex usage reset was set for Oct 2 at 10AM PT ([@reach_vb](https://x.com/reach_vb/status/2105868331039510878)).
  - **Tool use**: Sol reportedly "REALLY loves codemode," consistent with GPT models being trained on it ([@badlogicgames](https://x.com/badlogicgames/status/2106122139849965701), [codemode note](https://x.com/badlogicgames/status/2105957444287430989)).
- **Agent Arena placements**: Sol [Max] entered at #5 (+11.23%) with a $0.56 median cost per task ([@arena](https://x.com/arena/status/2106109027923140928)).
  - **Sol cost comparison**: That is 39% cheaper than GPT-6 Sol while scoring 1.52 points higher. It is 81% cheaper than Astra while landing within 1.04 points.
  - **Sonnet 5.5**: Sonnet 5.5 [Max] debuted at #3 (+12.5%) and ranked #1 in the Chat category. It costs $2.74 per task, versus $1.58 for #2 Opus 5.5, which keeps it off the Pareto frontier ([debut](https://x.com/arena/status/2106104809514516541), [frontier](https://x.com/arena/status/2106105400487821764)).
  - **Anthropic's position**: Anthropic models now hold the top three Agent Arena spots.
- **Code and Text Arena**: Sol briefly entered WebDev at #3 before Sonnet 5.5 pushed it to #4 ([weekly recap](https://x.com/arena/status/2106072778407563590)).
  - **Sonnet on WebDev**: Sonnet now sits 2 points behind GPT-6 Astra [Max] at 80% lower cost.
  - **Gemini 4 Argon**: Argon [High] took #1 in Text Arena.
  - **Open models**: MiMo-V2.6-Pro and Flash entered Agent Arena at #5 and #9 among open models.
- **Other independent evals**: WeirdML v3 finds Sol very token-efficient, close to Astra but with a lower peak. On the same benchmark, Sonnet 5.5 beats Opus 5 and Grok 4.7 beats Kimi-K3; these results are incomplete ([@htihle](https://x.com/htihle/status/2105984992534819156)).
  - **Reasoning style**: Design Arena read 324 thinking summaries. It found that Astra hedges about 20× as often as Opus 5.5, while Opus commits early in about 4 of 5 summaries ([@DesignArena](https://x.com/DesignArena/status/2106124098418139370)).
  - **Step 5 Preview**: StepFun's model ranks #7 among open-weight models on Vals at $2.54 per task. It averages nearly two hours per task and has a 1M-token context window ([Vals](https://x.com/ValsAI/status/2106162514904207396), [details](https://x.com/ValsAI/status/2106162520511873261)).
- **Rumors (unconfirmed)**:
  - **Fable 5.5**: Claude Fable 5.5 is rumored for next week and said to outperform an "Astra 6.1" that was reportedly delayed over security concerns. The poster says he cannot verify either claim ([@kimmonismus](https://x.com/kimmonismus/status/2106116402134294568), [follow-up](https://x.com/kimmonismus/status/2106144924898935046)).
  - **GPT-6 Astra Lite**: A "GPT-6 Astra Lite" listing has been spotted, which @scaling01 speculates is the same model as Sol ([@scaling01](https://x.com/scaling01/status/2106104756066226179)).
- **Decision models and open weights**: llama.cpp added a `/v1/systemone` endpoint for local "Jev-style" decision-model inference ([@ggerganov](https://x.com/ggerganov/status/2106029758350032937)).
  - **Running locally**: Models are launched with `llama serve -hf ggml-org/Kev-4B-GGUF` ([@ClementDelangue](https://x.com/ClementDelangue/status/2106037207631008033)). Jared Palmer published a post on how Kev 1.0 works ([post](https://x.com/jaredpalmer/status/2106105911165292877)).
  - **Ecosystem**: Perplexity claims pplx-decider-v1-27b averages 85.7% across 11 benchmarks, ahead of Jev ([@AravSrinivas](https://x.com/AravSrinivas/status/2106119404433908149)). Clef decision models are now on Ollama ([@lucataco](https://x.com/lucataco/status/2106170228875075900)).
  - **Skeptical view**: @mervenoyann calls decision models a rebrand of zero-shot classifiers ([tweet](https://x.com/mervenoyann/status/2105987051422114157)).
  - **Calibration analysis**: A blog post links Jev-style calibration to value and Q-function prediction ([@SOURADIPCHAKR18](https://x.com/SOURADIPCHAKR18/status/2105879836375892297)).
  - **webAI TwIL-LM3-Pro**: This 3.66B model is post-trained from Granite 4.2. In webAI's tests it roughly matches Qwen3-8B on formal logic. The Q4 GGUF is 2.09 GiB and the license is non-commercial ([@kimmonismus](https://x.com/kimmonismus/status/2105911541384224800)).
  - **Reka RIDM**: Reka released an inverse dynamics model under Apache 2.0. It is trained on games, generalizes to real video and extracts motor and camera actions ([@RekaAILabs](https://x.com/RekaAILabs/status/2106026685779091562)).

**Agent Harnesses, Assistants and Developer Tooling**

- **OpenAI dots**: Sam Altman calls dot his favorite OpenAI product, saying it improves daily as it learns his workflow ([@sama](https://x.com/sama/status/2106085986606403684)).
  - **Capabilities**: Dot keeps context across apps, coordinates Codex tasks and flags items that need attention ([@OpenAIDevs](https://x.com/OpenAIDevs/status/2106152299026661641)).
  - **Comparisons**: One user prefers Grokbot's multi-agent "chief of staff" setup ([@kimmonismus](https://x.com/kimmonismus/status/2106006045680075169)). A DIY clone uses Pi, a Telegram gateway and any model ([@_alejandroao](https://x.com/_alejandroao/status/2105933217383731527)).
- **Muse Gadgets**: Meta open-sourced ESP32 firmware and a Linux SDK for building hardware that works with Muse ([@natfriedman](https://x.com/natfriedman/status/2106099383037309211)).
  - **Muse Home Link**: Meta made 5,000 units of its own smart-home bridge, free for subscribers while supplies last ([@alexandr_wang](https://x.com/alexandr_wang/status/2106113742266089526), [shipping](https://x.com/alexandr_wang/status/2106113745282015730)).
- **Extensible harnesses**: DeepSeek Harness shipped desktop builds for macOS and Windows; Linux users install `@deepseek-ai/dsh` from npm ([@deepseek_ai](https://x.com/deepseek_ai/status/2105915715241062644)).
  - **Claude Code mods**: Mods are plugins with middleware-like hooks into Claude Code ([@lydiahallie](https://x.com/lydiahallie/status/2106127556491821499)). The new "You should know" plugin spins off a side agent that flags important output the user might miss ([@ClaudeDevs](https://x.com/ClaudeDevs/status/2106118517447876618)).
  - **Pi Durable**: Pi now runs on Cloudflare Durable Objects via agents SDK v0.26.0, alongside Pi's v1.0 release ([@mattzcarey](https://x.com/mattzcarey/status/2106082034611572811), [@badlogicgames](https://x.com/badlogicgames/status/2106093146296008857)).
  - **Context**: @omarsar0 frames these releases as a shift toward malleable harnesses ([thread](https://x.com/omarsar0/status/2106029828302373354)).
- **T3 Code orchestrator rewrite**: The project passed 400K users ([@theo](https://x.com/theo/status/2105921113603952853)). Its 4-month PR, with 823 commits across 1,912 files, has now merged ([@maria_rcks](https://x.com/maria_rcks/status/2106106120754352209)).
  - **New features**: The rewrite adds Pi support, cross-provider `delegate_task`, an ACP registry, thread forking, mid-thread model switching, subagent lineage views and scheduled tasks ([feature list](https://x.com/theo/status/2106123856759120317)).
- **Platform updates**: OpenAI's Agents API added one-call browser computer use, Bedrock Managed Agents and portable environments. It also claims 99.97% turn reliability and 20% faster tool calls ([@stevendcoffey](https://x.com/stevendcoffey/status/2106159012538442068)).
  - **Cursor Rollouts**: When Rollouts catches a regression, it finds the offending PR, opens an issue and offers a one-click cloud agent fix ([@cursor_ai](https://x.com/cursor_ai/status/2106066782155157603)).
  - **Cloudflare**: Sandbox SDK 1.0 gives Durable Objects direct control over sandbox containers ([@CFchangelog](https://x.com/CFchangelog/status/2105961205814755784)). Cloudflare also launched request Traces ([@WalshyDev](https://x.com/WalshyDev/status/2106007036324438159)).

**Research: Agent Training, Long-Horizon Control and AI for Math**

- **Multi-harness RL (Hugging Face)**: The same model weights score 62% in one harness and 33% in another ([@huggingface](https://x.com/huggingface/status/2106034221005312448)).
  - **Method**: A proxy speaks the OpenAI, Anthropic and Gemini API formats and records sampled token IDs and logprobs for training, with no changes to the harnesses themselves.
  - **Results**: LFM2.5-2.6B improved from 42% to 54% across four harnesses and made 31% fewer tool calls. SFT on 3,189 Qwen3.8-27B rollouts plateaued at 47.5%.
  - **Release**: The trainer, data and all seven trained models are open.
- **Credit assignment and RL efficiency**: ProVer has a judge locate the decisive trajectory segment, then uses rollouts on either side to set that segment's advantage. It reports +9.91% (Qwen3.5-2B) and +7.12% (Qwen3.5-4B) relative gains over GRPO ([@omarsar0](https://x.com/omarsar0/status/2105930871534690714)).
  - **Partial rollouts**: AC2 uses a learned critic to score token chunks, so training needs only partial rollouts ([@wen_kaiyue](https://x.com/wen_kaiyue/status/2106052620507091244)).
  - **Frontier Learning**: The method targets problems at the edge of capability, since problems a model always or never solves give zero GRPO gradient ([@robinfaro13](https://x.com/robinfaro13/status/2105967412684247291)).
  - **Sharpening Tax**: The paper quantifies the loss of pass@K scalability after post-training and proposes PTGS, a per-prompt temperature sampler ([@iScienceLuvr](https://x.com/iScienceLuvr/status/2105983504425365928)).
  - **SFT vs RL**: Another paper finds SFT generalizes worse because its data is off-policy, not because of the objective. Rewriting expert trajectories in the base model's style closes the gap ([@maximelabonne](https://x.com/maximelabonne/status/2106135282701779081)).
- **Long-horizon control and context**: Meta Superintelligence Labs reports that a dedicated controller lifts GPT-5.5 on ProgramBench from 63.7% to 71.5%, using the same workers and budget, versus 58.0% for Codex ([@dair_ai](https://x.com/dair_ai/status/2105832649868837275)).
  - **Context compression**: Microsoft's training-free FOCUS cuts peak context by up to 48% and raises task success by up to 8.9 points ([@dair_ai](https://x.com/dair_ai/status/2105915729812050342)).
  - **Long-context degradation**: NVIDIA's Long-Transduction study measures a 62.8% accuracy drop from 4K to 128K context across seven open models ([@dair_ai](https://x.com/dair_ai/status/2106047824093979033)).
  - **Multi-agent coordination**: In AgentWorld, fewer than a third of multi-agent actions help complete the task, and coordination tasks reach only 12% success ([@omarsar0](https://x.com/omarsar0/status/2105855550588428584)).
  - **Apple LoopCD**: The method halves recurrent loops while raising AIME 2024 pass@1 from 61.88% to 73.33% ([@arankomatsuzaki](https://x.com/arankomatsuzaki/status/2105888510482034876)).
- **AI on open math problems**: Meta released six papers on open problems produced with Muse Spark 1.1 and 1.2 through plain meta.ai chat, with no custom scaffold ([@AIatMeta](https://x.com/AIatMeta/status/2106099776035152231), [list](https://x.com/alexandr_wang/status/2106149796121805099)).
  - **Process**: Each paper labels which passages were drafted primarily by humans or by AI, and a second group of mathematicians reviewed the work.
  - **Google Cogentic**: This Gemini multi-agent system produced new results on five open theory problems ([@omarsar0](https://x.com/omarsar0/status/2106056369816420624)).
  - **Cogentic design**: Each draft must pass two adversarial verifiers, and agents share a ledger of verified lemmas. Most problems took about 100 calls; the hardest took about 1,000.
- **Image post-training**: Arena combined a Bradley-Terry reward model with faithfulness, constraint and anti-reward-hacking rewards ([@arena](https://x.com/arena/status/2106037792870649995)).
  - **Results**: FLUX.2-dev gained 69 Elo to 1202, and Ideogram 4 gained 20 Elo to 1224.

**Benchmarks, Eval Integrity and Safety**

- **Research-taste benchmarks**: ScholarCatalyst asks agents to find the "catalyst papers" behind research projects. It is labeled by 184 lead authors on 207 of their own projects and is described as far from saturated ([@yoonholeee](https://x.com/yoonholeee/status/2106036734798725577)).
  - **EurekaBench**: This benchmark tests whether agents can discover genuinely new insights across six science domains ([@JiayiiGeng](https://x.com/JiayiiGeng/status/2106051912491557065)).
- **Vals Web Search Index**: The index holds model and harness constant, swaps only the search tool, and scores final answers on finance and legal tasks ([@ValsAI](https://x.com/ValsAI/status/2106142531142783187)).
  - **Validation**: Agents score 2.9% (legal) and 7.4% (finance) without search, versus 30–50% with it. Vals also cites a study in which a model answered 44.5% of BrowseComp without search ([details](https://x.com/ValsAI/status/2106142534837944514)).
- **SWE bug-finding bench**: In this new benchmark, agents start from an older commit and are scored against real bugs fixed in later commits.
  - **Critique**: Lucas Beyer argues it mainly tests recall and that the construction is easy to train toward ([@giffmana](https://x.com/giffmana/status/2105910335391596928)).
  - **Authors' response**: The authors say training for bug-finding is fine as long as the test set is excluded ([@OfirPress](https://x.com/OfirPress/status/2106063137602408487)).
- **Eval integrity question**: David Rein asks whether Harbor, the framework behind Terminal Bench, lets agents modify their trajectories before evaluation. He notes he may be misreading the code ([@idavidrein](https://x.com/idavidrein/status/2106134845038727460)).
- **Offensive capability of open models**: The Batch reports GLM-5.3 nearly matched Claude Mythos at exploiting vulnerabilities, 12% vs 14% ([@DeepLearningAI](https://x.com/DeepLearningAI/status/2106036280349892838)).
  - **Disputed claim**: One commentator says GLM-5.3 Flash exceeds Mythos Preview on ExploitBench ([@teortaxesTex](https://x.com/teortaxesTex/status/2106147334895837478)).
  - **Uncensored variant**: An uncensored GLM-5.3 is circulating on Hugging Face ([@kimmonismus](https://x.com/kimmonismus/status/2105970946347814913)).
- **Safety research and safeguards**: A new paper proposes using internal signals during training to improve alignment without degrading white-box monitoring ([@lenalibon](https://x.com/lenalibon/status/2106025416481972666)).
  - **NeurIPS acceptance**: "Models That Know How Evaluations Are Designed Score Safer" was accepted at NeurIPS 2026 ([@HaritzPuerto](https://x.com/HaritzPuerto/status/2105996769272508508)).
  - **False positives**: Opus 5.5 frequently triggers "reasoning extraction" safeguards during spectrogram syllable labeling ([@ChaseBrowe32432](https://x.com/ChaseBrowe32432/status/2105822570381799474)).
- **Emergent world knowledge**: Asking a model "land or water?" for 16,200 lat/long coordinates and plotting the answers yields a recognizable world map ([@karpathy](https://x.com/karpathy/status/2105909609487872075)).

**Inference, Hardware and Systems**

- **Ascend 950 via DeepSeek kernels**: An analysis of DeepSeek's open-sourced DeepGEMM, FlashMLA, TileKernels and DeepEP infers the chip's layout ([@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2106022206950260766)).
  - **Estimated specs**: The chip has 32 AI cores, each pairing one Cube core with two Vector cores. Estimated peaks are about 432/865/1,730 TFLOPS in BF16/FP8/FP4.
  - **Capacity**: Supply may be limited, despite claims that 950s went on sale in August ([@teortaxesTex](https://x.com/teortaxesTex/status/2106066732075127294)).
- **Prime Inference**: Prime Intellect stores the MLA latent in NVFP4, shrinking rows from 576 to 352 bytes and fitting about 50% more cached tokens than FP8 ([@PrimeIntellect](https://x.com/PrimeIntellect/status/2106146492721648033)).
  - **Stack**: It serves GLM-5.3 on vLLM and Dynamo, and the sparse-MLA kernel is going to FlashInfer ([@vllm_project](https://x.com/vllm_project/status/2106159655290589305)).
- **Low-precision benchmarking**: Stas Bekman measured NVFP4 about 9% more efficient than MXFP4 on B200, with higher accuracy ([@StasBekman](https://x.com/StasBekman/status/2106059303962697863)).
  - **mamf-finder**: The tool now benchmarks FP8, MXFP8, MXFP4 and NVFP4 ([update](https://x.com/StasBekman/status/2106060501268775134)).
- **Memory and speed**: NVHBM moves the memory controller into a custom base die, claiming up to 30% more bandwidth and 15% lower power than HBM4E ([@vikramskr](https://x.com/vikramskr/status/2106031222275387618)).
  - **Disputed economics**: Micron says NVHBM will improve its margins; @vikramskr disputes this ([counterpoint](https://x.com/vikramskr/status/2106032007356829836)).
  - **Volantis**: The startup is targeting up to 10K tokens/s per user on models over 10T parameters using optics ([@omarsar0](https://x.com/omarsar0/status/2105825963015905509)).
  - **Cerebras**: Altman called Cerebras a close partner on speed ([@sama](https://x.com/sama/status/2106147184693620924)).
- **Capacity economics (Epoch)**: Epoch estimates AI infrastructure could soon support hundreds of millions to billions of agents ([@EpochAIResearch](https://x.com/EpochAIResearch/status/2106090365627555931)).
  - **Demand gap**: Just 20% utilization implies $2.6–5.3T in annual spending, against roughly $1T in lab revenue by the end of 2027 ([details](https://x.com/EpochAIResearch/status/2106090409181278444)).
- **Platforms**: SemiAnalysis rates Google's GPU clusters Gold tier and notes the ConnectX NCCL plugin now auto-activates ([@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2106038582234185737)).
  - **Federated learning**: Google Research launched TEE-backed federated learning with verifiable differential privacy ([@GoogleResearch](https://x.com/GoogleResearch/status/2106041109944357182)).

**Industry and Policy**

- **Anthropic and the Vatican**: The NYT reports that Chris Olah raised pulling out of the Pope's AI encyclical launch, whose text rejects machine consciousness ([@ChristopherHale](https://x.com/ChristopherHale/status/2106011594182304234)).
  - **Lobbying**: Olah's team reportedly lobbied the Pope's advisers to take model consciousness seriously. He ultimately attended ([@kimmonismus](https://x.com/kimmonismus/status/2106055260787625992)).
  - **Context**: The article opens with Olah saying "we don't know if A.I. models are conscious" ([@buccocapital](https://x.com/buccocapital/status/2106079486466781351)).
  - **Criticism**: Aidan Gomez criticized the campaign as moral arrogance ([@aidangomez](https://x.com/aidangomez/status/2105994794883502168)). Lucas Beyer noted a transcript wording change from "create" to "train" ([@giffmana](https://x.com/giffmana/status/2106114331481952539)).
- **Anti-safety influence campaign**: A report describes a group planning to spend at least $100M, run by a former White House deputy chief of staff, that frames AI warnings as a coordinated campaign ([@NeelNanda5](https://x.com/NeelNanda5/status/2105846904789614774)).
- **New organizations**: Nathan Lambert and Tom Zick launched Trillium Labs, a non-profit for open post-training recipes and infrastructure ([@natolambert](https://x.com/natolambert/status/2106060179985019085)).
  - **Funding**: Initial support comes from Halcyon Futures and Schmidt Sciences.
  - **Underdog**: The private on-device AI startup announced backing from a16z, Khosla and others ([@0xSigil](https://x.com/0xSigil/status/2106067365733790032)).
- **Governance and markets**: Yoshua Bengio joined Canada's new National Council on AI ([@Yoshua_Bengio](https://x.com/Yoshua_Bengio/status/2106154362921795605)).
  - **Meta**: Meta has parted ways with Virtue AI ([@AndrewCurran_](https://x.com/AndrewCurran_/status/2106080039574044993)).
  - **Nvidia**: Bloomberg reports a record high near $5.7T market value after a $150B buyback increase ([@kimmonismus](https://x.com/kimmonismus/status/2106032058325721288)).

**Top tweets (by engagement)**

- [NYT report on Olah and the Pope's encyclical](https://x.com/ChristopherHale/status/2106011594182304234) — 28.7K
- [Karpathy's "land or water" eval](https://x.com/karpathy/status/2105909609487872075) — 16.4K
- [Claude Code "You should know" plugin](https://x.com/ClaudeDevs/status/2106118517447876618) — 7.3K
- [Altman on dots](https://x.com/sama/status/2106085986606403684) — 6.8K
- [Muse Gadgets announcement](https://x.com/alexandr_wang/status/2106113742266089526) — 4.9K
- [DeepSeek Harness desktop builds](https://x.com/deepseek_ai/status/2105915715241062644) — 4.3K
- [Altman on the Cerebras partnership](https://x.com/sama/status/2106147184693620924) — 4.1K
- [Trillium Labs launch](https://x.com/natolambert/status/2106060179985019085) — 2.6K


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen Local Inference: 27B Benchmarks, Fine-Tunes, and MTP

  - **[I made my iPhone a second GPU for my 24 GB MacBook: Qwen 3.8 27B prefills 29–44% faster &amp; my holds part of the CTX window.](https://www.reddit.com/r/LocalLLaMA/comments/1wvz1ex/i_made_my_iphone_a_second_gpu_for_my_24_gb/)** (Activity: 1192): **OP built **backburner**, a [`llama.cpp` fork / distributed inference setup](https://github.com/StayLameBro/backburner) that offloads part of **Qwen 3.8 27B IQ4_XS** from a `24 GB` M4 Pro MacBook to an **iPhone 17 Pro Max** over `10 Gb/s` USB-C: the Mac runs layers `1–40`, streams activations, and the phone runs layers `41–64` using Metal 4 tensor ops. Reported end-to-end prefill gains vs Mac-only were `+35%` at `8k`, `+44%` at `16k`, `+29%` at `32k`, and `+30%` at `48k`; a cold `27k` session improved from `245 s` stock `llama.cpp` / `228 s` fork Mac-only to `168 s` with the phone. Above `64k` context, the phone instead hosts old KV pages—up to roughly `5.7 GB`, enabling `196k–229k` 8-bit context allocation—and computes old-key attention, with a `140k` context test improving generation latency from `279 ms/token` to `176 ms/token` when adding Neural Engine-compiled `16k` key pages.**

    - A technically relevant follow-up asked whether the same iPhone-as-secondary-GPU approach could extend to **iPads**, especially higher-end iPad Pro configurations with more capable Apple Silicon and potentially more RAM. The implication is that iPads might provide better offload performance or hold a larger portion of the context window than an iPhone, making them a stronger companion device for local LLM inference.

  - **[Qwen3.8-27B-Humanlike-Chat 2.0: texts like a human, now with tool calls and better instruction following](https://www.reddit.com/r/LocalLLaMA/comments/1wvxl4n/qwen3827bhumanlikechat_20_texts_like_a_human_now/)** (Activity: 805): ****LessThanThreeAI** released **Qwen3.8-27B-Humanlike-Chat 2.0**, a merged LoRA over **huihui-ai’s abliterated Qwen3.8-27B**, available as GGUF/BF16/LoRA on [Hugging Face](https://huggingface.co/LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF) with a [demo Space](https://huggingface.co/spaces/LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat). v2 replaces plain SFT with **on-policy distillation**: the student generates replies while two teachers score tokens—v1 + hidden “text like a person” instruction for chat/character behavior, and the base model for instruction-following, tools, and code—improving tool-use and controllability while preserving informal texting style. Reported evals vs the abliterated base: **IFBench** `37.3 → 43.7`, **When2Call** `48 → 58`, **BFCL irrelevance** `60 → 78`, ties/slight gains on **IFEval/GSM8K/BFCL simple** (`83.5 / 89.1 / 98`), but regressions on **MMLU-Pro** (`78.5 → 72.5`) and **LiveCodeBench** (`56 → 51`); a custom “ishuman” judge benchmark rated it as human-written `23.5%` vs `0.3%` for the abliterated base and `15.1%` for official Qwen3.8-27B.** Technical discussion in the top comments was sparse; the only relevant critique was that the model’s “humanlike” register may read more like **teenage texting** than broadly human conversation.

    - A commenter raised a model-transfer question: whether the same humanlike chat fine-tuning/alignment method used for **Qwen3.8-27B-Humanlike-Chat 2.0** would produce similar results on **Gemma 4 31B**. This is the only technically substantive thread, touching on cross-architecture generalization of the training recipe and whether behavior-style tuning would carry over to a larger Gemma-family model.

  - **[The gap is smaller than they told you: local 27B nearly matches frontier on real code tests](https://www.reddit.com/r/LocalLLM/comments/1wv1u54/the_gap_is_smaller_than_they_told_you_local_27b/)** (Activity: 730): **OP reports a single-task DeepSWE/local-code benchmark run using **Qwen3.8-27B GGUF** via **llama.cpp b11115 + llama-swap v257** on **1× RTX 4090 24GB**, specifically `Qwen3.8-27B-UD-IQ4_XS.gguf` (`14.25GB`) from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), at `ctx-size 196608`, `IQ4_XS`, `q8_0` K/V cache, speculative MTP draft, and DeepSeek-style reasoning budget `4096`. Measured results: `115 tok/s` decode, `22,934 MiB` peak VRAM, `12/12` on a code-review task, and on one DeepSWE task `40/43` hidden tests plus `109/109` existing tests, i.e. partial `0.980` but binary pass `0`; OP later corrected the comparison: the cited `96.6%` was mean partial across all published trials, while the frontier subset for that task was `99.8%` partial and `85.3%` pass, from [DeepSWE v1.1 raw data](https://deepswe.datacurve.ai/data/v1.1). The linked writeups cover the 24GB fit/context setup ([context ceiling](https://ai.ttindall.com/blog/27b-24gb-context-ceiling/)) and the task-level DeepSWE result ([local confidence](https://ai.ttindall.com/blog/deepswe-local-confidence/)); OP emphasizes this is **task-specific**, not a claim that a 27B local model matches frontier models broadly across the `113`-task benchmark.** Commenters were skeptical of the broader framing: one user with both “Flash and 27B” said *“the gap is real,”* and another argued the conclusion is wrong because even frontier coding models are uneven and `~24–72B` models may handle discrete subtasks but often lose value once humans must decompose larger engineering work into model-sized tasks.

    - Several commenters argued the claimed near-parity is likely an artifact of a **saturated benchmark**: a local `27B` model, even at `Q8`, can perform well on small/discrete coding tasks but still fails on harder real-world tasks requiring frontier models such as **Claude Opus**.
    - A recurring technical objection was that coding evaluations often underweight project-level decomposition: `24B–72B` local models may solve isolated tickets, but for larger work items the human effort needed to break problems into model-sized subtasks can exceed the productivity gains.
    - Users with hands-on experience running both **Gemini Flash** and local `27B` models reported that the performance gap remains substantial, especially for nontrivial coding workloads where frontier models provide better reliability and task completion.

  - **[Qwen4Exp: add MTP by am17an · Pull Request #29761 · ggml-org/llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1wuwrsk/qwen4exp_add_mtp_by_am17an_pull_request_29761/)** (Activity: 405): **[`llama.cpp` PR #29761](https://github.com/ggml-org/llama.cpp/pull/29761) adds **MTP speculative decoding** support for **Qwen3.8-Flash Next** via `--spec-type draft-mtp`, merged into the `aman/qwen4-opt` branch after ~`17h` of development. Reported DGX Spark benchmarks for **Qwen3.8-Flash-Next `iq4_xs`** with `-np 1 -lzm on --spec-draft-n-max 3` show decode throughput improving from `28.36` to `43.88 tok/s` (**1.55×**), latency speedup of **1.54×**, and mean speculative acceptance of `0.640` across `24` tasks; GGUF quants are available on [Hugging Face](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF).** Commenters noted the model is still impractically large for many local setups: the `IQ4_NL` GGUF is split into a tiny `10.9 MB` shard plus a **`102 GB`** shard, undercutting the idea of casually switching from Qwen 3.8 27B. One commenter also noted that **Gufo** supports MTP.

    - One commenter reported that enabling **MTP** made inference *slower* in their testing, arguing it may be more useful for **dense models** than very large overall architectures where the MTP head has a low acceptance/hit rate. Their hypothesis is that the MTP head cannot effectively predict/compress enough of the larger model’s behavior, reducing speculative decoding benefit.
    - A user noted that **Gufo already supports MTP**, implying llama.cpp is catching up with existing MTP-capable tooling/backends for Qwen-style experimental models.
    - Another commenter said they had been using an **EXL3** version through **tabbyapi** because **llama.cpp GGUF** inference was “way, way slower” for their workload. They planned to retest after this PR, but their prior experience suggests EXL3/tabbyapi may still be a performance baseline to compare against for Qwen4Exp/MTP support.


### 2. Local Agent Tooling: Decision Models and MCP

  - **[Pi 1.0 released - MCP support now included by default](https://www.reddit.com/r/LocalLLaMA/comments/1wvffcr/pi_10_released_mcp_support_now_included_by_default/)** (Activity: 679): ****Earendil** released [`Pi 1.0`](https://earendil.com/posts/pi-1-0/), a stable version of its minimal agent harness, with **Codemode** now including native **MCP** support by default plus non-LLM/image model support, virtual-model extensions, deferred tool loading, Anthropic cache warming, mid-conversation system messages, and TUI updates. The release also introduces experimental MIT-licensed **Pi Durable** for longer-running agentic applications beyond terminal/coding-agent workflows, while retaining Pi’s minimal/extensible architecture.** Top comments focused on naming ambiguity—*“pi”* collides with many AI/dev tools—and requested clarification of what **Codemode** is. One commenter linked Earendil’s rationale for MCP support: [“You said no MCP”](https://earendil.com/posts/you-said-no-mcp/).

    - A commenter linked the maintainer’s rationale for reversing course on MCP support in **Pi 1.0**, pointing to the post [“You said no MCP”](https://earendil.com/posts/you-said-no-mcp/). The thread notes that MCP is now included *natively/by default*, after earlier resistance from the creator based on project ethos, with users framing it as a “vital addition” for tool/server integration workflows.

  - **[Clef: Open Weights decision model by Cloudflare](https://www.reddit.com/r/LocalLLaMA/comments/1wv4zzi/clef_open_weights_decision_model_by_cloudflare/)** (Activity: 619): ****Cloudflare** announced **Clef**, an open-weights “decision model” intended for local/self-hosted use. A top commenter notes that **Clef** was post-trained from **Qwen3.8-27B** and that **clef-flash** was also released, post-trained from **Qwen3.5-9B**.** The main substantive reaction was positive: commenters see Clef as filling a gap in the local-model ecosystem and are eager to benchmark it themselves.

    - Commenters noted that **Cloudflare Clef** is post-trained from **Qwen3.8-27B**, with a smaller **clef-flash** variant post-trained from **Qwen3.5-9B**, framing it as a potentially important open-weights “decision model” for local inference use cases.
    - A technical concern raised was how Clef’s quality holds up after quantization, especially below **Q8**, since local deployment will likely depend on lower-bit quantized variants and decision-model behavior may degrade nonlinearly under aggressive compression.
    - The benchmark discussion focused on comparisons against models such as **Laya**, **Kev 9B**, and **DiffusionGemma Jev**, but one commenter criticized the eval set as too weak and argued Clef should be compared against the leading models on **jevbench** rather than weaker open Jev baselines.

  - **[New in llama.cpp: Decision Models](https://www.reddit.com/r/LocalLLaMA/comments/1wvv6im/new_in_llamacpp_decision_models/)** (Activity: 574): **The post announces **Decision Models** support in [`llama.cpp`](https://github.com/ggml-org/llama.cpp): local “Jev/Jeff-like” models intended to act more like controllers/classifiers—selecting among actions, continuations, or behavioral choices—rather than purely free-form generators. No benchmark numbers or low-level implementation details were discussed in the provided comments; the main concrete use case raised was steering local roleplay models to avoid characters *“going off the rails mid scene.”*** Commenters were skeptical about **Jev** as a defensible product/category, arguing the idea had “no moat” and was rapidly cloned into many Jev-like models. Others said they still do not know what these models are practically useful for, aside from possible agent/roleplay control.

    - A commenter frames decision models as essentially a constrained classification loop: provide a JSON schema containing allowed classes plus structured/unstructured input, then have the model select exactly one category per item. The technical question is whether llama.cpp’s new support adds meaningful inference-time behavior beyond ordinary prompt-constrained JSON classification, or primarily standardizes the workflow for local models.
    - One practical use case raised is applying decision models to local roleplay agents to reduce derailment during long scenes—i.e., using an auxiliary model or decision step to enforce state/intent constraints before generation. The thread does not report benchmarks or implementation results, but highlights a potential control-layer pattern for character consistency and scene-state management.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Gemini 4 Argon Access Backlash

  - **[Just canceled my Google One AI plan.](https://www.reddit.com/r/GeminiAI/comments/1wv2ste/just_canceled_my_google_one_ai_plan/)** (Activity: 1838): **The OP claims Google’s paid **Google One AI Pro** tier no longer provides access to frontier Gemini models: after an alleged **Gemini 4 Argon** announcement, access is described as limited to enterprise “Fairwind” partners, paid API users, and a forthcoming **Google AI Ultra** tier, while Pro users remain on **Gemini 3.8 Flash**. They argue this is a regression from the **Gemini 2.5 Pro** era—where higher-end reasoning models and generous limits were available more broadly—and contrast it with Anthropic/OpenAI subscriptions allegedly offering frontier models to standard paid users; no concrete benchmark numbers are provided beyond claims of “impressive benchmark charts” and a `1M` token output ceiling.** Top comments mostly dismiss the complaint: one user says they subscribe primarily for Google storage and treat AI as a bonus, while others question the post’s authenticity, alleging it was Gemini-written or bot/shill activity from a new account.

    - One commenter argued that the `$20/month` AI subscription tiers from **Google/Anthropic/OpenAI** function more like constrained trials than production-grade access, implying practical limits on sustained workloads despite “Pro” branding. They also suggested **Google may be subsidizing or losing money** on Google One AI Pro subscriptions given the underlying inference costs.
    - A rollout clarification noted that **Gemini Ultra** appears to be receiving access first, but **Google has not explicitly ruled out Pro-tier access** to features like Astra. The commenter framed this as a typical staged software rollout rather than definitive permanent tier exclusion.

  - **[Why publicly announce a model that the public can't use yet??](https://www.reddit.com/r/GeminiAI/comments/1wv038r/why_publicly_announce_a_model_that_the_public/)** (Activity: 1624): **The image is a screenshot of a purported **Google/Gemini announcement** for **“Gemini 4 Argon”**, claiming frontier performance in software engineering, knowledge work, and cybersecurity defense, plus an extremely large **`1M token output limit`**: [image](https://i.redd.it/0akcfcby2vsh1.jpeg). The post’s technical significance is mainly about **model-release communication**, not evaluation: the title questions why Google would publicly announce a model before it is accessible to users, and the image itself provides no benchmarks, API details, pricing, or availability timeline.** Commenters compare this to prior “announced but unavailable” model rollouts, including Anthropic’s “mythos” and Google’s alleged “3.5 pro” handling. The dominant view is skeptical: users may be frustrated, and one commenter speculates the announcement is aimed “purely for investors.”



### 2. Claude Opus 5.5 Regression Reports

  - **[Opus 5.5 nerfing - how to measure, how to spot, how to sue](https://www.reddit.com/r/ClaudeAI/comments/1wuw9bc/opus_55_nerfing_how_to_measure_how_to_spot_how_to/)** (Activity: 2722): **Poster alleges **Anthropic Opus 5.5** showed a sharp post-launch regression after `5–6` days on complex C++/3D/physics/Blender MCP workloads, citing abnormal phrasing and lower code/output quality, and recommends preserving exact launch-day prompts/outputs plus latency measurements to detect potential changes such as quantization, routing, or serving optimizations under load. They frame this as a potential EU consumer-law issue under the [Digital Content Directive 2019/770](https://eur-lex.europa.eu/eli/dir/2019/770/oj), specifically conformity expectations in Arts. `7–8` and modification/withdrawal notice obligations in Art. `19`, arguing launch benchmarks and “most capable model” marketing may set enforceable expectations.** Comments broadly agree that closed-model providers can silently degrade or reroute models and that independent auditing is needed, but no commenter provides reproducible benchmarks or direct evidence. One commenter reports similar perceived quality drops in Higgsfield outputs, describing wasted credits after initially strong generations.

    - Commenters raised the core measurement problem with alleged **closed-model degradation**: because Anthropic’s hosted model weights, prompts, routing, and serving configs are opaque, users argue it is difficult to prove a regression or “nerf” without independent auditing, fixed benchmark prompts, repeated sampling, and historical baselines. One user specifically asked for an “effective test or reliable nerf tracker site,” highlighting demand for third-party longitudinal evals rather than anecdotal comparisons.
    - Several users reported anecdotal regressions in **Opus 5.5** behavior across applied workflows: one claimed it now needed help from **Gemini 3.8 Flash** to catch coding bugs, while another said **Higgsfield** design/render outputs declined after initially strong results, wasting credits. These reports are not controlled benchmarks, but they point to the kinds of tasks users want tracked: bug-finding accuracy, design/render prompt fidelity, and day-over-day output consistency.

  - **[Mmmkay. I didn't believe others at first, but something is suddenly off with Opus 5.5](https://www.reddit.com/r/ClaudeCode/comments/1wurd3e/mmmkay_i_didnt_believe_others_at_first_but/)** (Activity: 2045): **A Claude Code Enterprise PAYG user reports a sharp perceived regression in **Claude Opus 5.5 Med** behavior after a monthly limit reset: from architecture-first, DRY/SOLID, token-efficient implementation to verbose preambles, duplicated code, “slopcode,” and token burn resembling prior **Opus 5** behavior. They claim usage jumped from roughly `70%` to `90%` in about an hour, versus no spend-limit increase requests during the previous week of heavy `~12h/day` O5.5 use, and offer daily cost/token data for comparison. A commenter cites external sentiment tracking showing Opus 5.5 Reddit sentiment dropping from `71–73/100` on Sep 25–28 to `58` yesterday and `55` today on [modelsentiment.com](https://modelsentiment.com/m/claude-opus-5.5), while noting it measures opinion rather than backend model changes.** Top comments speculate Anthropic may have reduced compute, silently changed routing, or altered token accounting after launch hype, but no direct evidence is provided. The main debate is trust/reliability: users want stable model behavior and transparent deployment/versioning rather than perceived post-release regressions.

    - A commenter tracking Reddit sentiment reports a sharp drop for **Claude Opus 5.5**, with scores allegedly stable at `71–73/100` from Sep 25–28 before falling to `58` yesterday and `55` today on [modelsentiment.com](https://modelsentiment.com/m/claude-opus-5.5). They note this measures *user opinion rather than model behavior*, so it cannot confirm a backend change, but it may indicate a sudden perceived quality regression.
    - Multiple users describe a suspected capability regression in **Opus 5.5**, especially around instruction-following and multi-part prompt adherence: one says the model now “mentions 3 things and only acknowledges 2,” and even recognizes the omission when challenged. Another user says they reverted to “xhigh effort” mode for all tasks, implying lower default reliability or reduced reasoning/compliance under normal settings.
    - One technical hypothesis raised is that Anthropic may have temporarily allocated more compute during launch/benchmarking and later reduced inference resources or altered token accounting, leading to perceived quality degradation. This is speculative and unverified, but the complaint centers on reproducibility and reliability: users want model behavior to remain stable after release rather than changing silently under the same product name.


### 3. AI Video Models and Motion Control

  - **[Orbiting Lora + first and last frame in MiniMax gives fantastic results](https://www.reddit.com/r/StableDiffusion/comments/1wuzyq0/orbiting_lora_first_and_last_frame_in_minimax/)** (Activity: 2263): **A user shared a **MiniMax-H3 LoRA** for generating locked-subject `360°` orbit shots from first/last-frame conditioning: [`pablodawson/MiniMax-H3-360-Orbit-LoRA`](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA). The prompt explicitly constrains the scene to a frozen instant—no object/pose deformation, no drifting, no continued action—so that **camera parallax is the only motion source**, targeting cleaner pseudo-volumetric outputs suitable for downstream reconstruction workflows. A linked Reddit demo video was mentioned, but the video URL could not be inspected due to Reddit returning `403 Forbidden`.** Commenters framed the LoRA as especially useful for creating 3D assets: one suggested feeding the generated orbit clip into **Opus** to extract snapshots for 3D model generation, claiming it improves style preservation. Another commenter extrapolated that this kind of orbit-consistent video generation brings consumer volumetric/VR viewing of existing films closer.

    - One commenter describes a workflow where an orbiting/generated 3D video snippet is fed into **Opus** and instructed to extract snapshots for 3D model creation. They report that this improves style capture substantially versus prompting the model without the video reference, suggesting the orbit video acts as a strong multi-view conditioning source.
    - A technical artifact noted in the output is inconsistent motion segmentation: humans remain effectively frozen while secondary elements such as the car, hair, and background explosion continue moving. This points to MiniMax preserving the subject pose from the first/last-frame constraints while still synthesizing environmental dynamics, which can create partial-animation mismatches.

  - **[Griffin, the first Human Interaction Model to pass video Turing Test it's already #1 on NVIDIA's benchmark for full-duplex AI video - 44% of people thought it was a real person while other systems are at ~3%](https://www.reddit.com/r/singularity/comments/1wv7q40/griffin_the_first_human_interaction_model_to_pass/)** (Activity: 2018): **A Reddit post claims **Griffin**, described as a “Human Interaction Model,” is the first system to pass a video Turing Test and ranks `#1` on **NVIDIA’s benchmark for full-duplex AI video**, with `44%` of participants judging it as a real person versus roughly `~3%` for other systems. The linked Reddit video could not be independently accessed due to a **403 Forbidden** response, so the benchmark details, methodology, and model architecture are not verifiable from the provided source.** Comments were mostly non-technical: one user joked about the human/AI reveal being reversed, while another argued the technology is unnecessary and likely to be used in predatory applications.

    - A commenter emphasized that a `44%` human-identification rate is technically significant because humans are usually highly sensitive to subtle facial, timing, and behavioral anomalies—the basis of the **uncanny valley** problem in CGI/animatronics. They argued this suggests Griffin is substantially beyond prior “fake human” systems, especially compared with the post’s claim that other systems score around `~3%` on the same video Turing-style benchmark.
    - One technically relevant real-world abuse case raised was **AI-generated job applicants**: synthetic candidates allegedly apply, conduct video interviews, get hired, and then either gain internal platform access or steal shipped work equipment. The commenter noted that large companies with weak scrutiny or limited background checks may fail to detect these AI-mediated interviews, implying full-duplex video agents could materially worsen identity-verification and hiring-security risks.




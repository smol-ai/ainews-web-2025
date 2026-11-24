---
id: MjAyNS0x
title: OpenAI Publishes Pretraining Perplexity History and Proposes Shared Limits
date: '2026-11-23T05:44:39.731046Z'
description: >-
  **OpenAI** published five years of base-model log-perplexity data, proposing
  industry-wide limits on pretraining progress tied to milestones in AI behavior
  monitoring. **Sam Altman** endorsed the proposal linking it to the White House
  Accords. **Microsoft AI** reported comparable metrics with its MAI-2 Base
  model. The data shows losses below 0.8 coinciding with emergent capabilities
  and predicts transformative capabilities in 2026-2027. Critics raised concerns
  about tokenizer bias, post-training compute, and synthetic data influence.
  OpenAI staff defended the evaluation methodology as fixed and tokenizer
  agnostic.
companies:
  - openai
  - microsoft
models:
  - gpt-6.2-astra
  - mai-2-base
topics:
  - pretraining
  - perplexity
  - model-evaluation
  - scaling
  - alignment
  - compute-efficiency
  - synthetic-data
  - tokenization
people:
  - sama
  - jakub_pachocki
  - greg_brockman
  - mark_chen
  - danhendrycks
  - mustafasuleyman
  - lucas_beyer
---


**a quiet day.**

> AI News for 11/21/2026-11/23/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**OpenAI Publishes Its Pretraining Perplexity History and Proposes Shared Development Limits**

- **The disclosure**: Jakub Pachocki's essay "Perplexity and Intelligence" publishes five years of OpenAI base-model log-perplexity on a fixed held-out dataset. It argues for industry-wide limits based on perplexity and other measurements, with further progress gated on milestones in understanding and monitoring AI behavior, such as a successor to chain-of-thought (CoT) monitors ([essay thread](https://x.com/merettm/status/2124549218995949967)).
  - **Policy framing**: Sam Altman quoted the proposal for a first common limit with concrete exit criteria ([quote](https://x.com/sama/status/2124551883130114403)). He tied it to the standards commitments in the White House Accords ([accords](https://x.com/sama/status/2124558713646428663)).
  - **Endorsements**: Greg Brockman highlighted the warning that alignment progress is unlikely to keep pace with pretraining acceleration ([gdb](https://x.com/gdb/status/2124582684840399311)). Mark Chen called the numbers formerly "sensitive IP" and asked other labs to disclose their key metrics ([markchen90](https://x.com/markchen90/status/2124592748099665943)).
- **Claimed findings**: Losses below 0.8 coincided with emergent capabilities in cybersecurity and mathematics ([excerpt](https://x.com/kimmonismus/status/2124577059238281412)).
  - **Thresholds and timing**: OpenAI expects further "transformative capabilities" in the ~0.68–0.7 window and describes a steep acceleration in 2026 ([summary](https://x.com/kimmonismus/status/2124554965029151198)).
  - **Internal model**: An unreleased internal model already sits below GPT-6.2 Astra. Pachocki says pretraining could see an "avalanche" like the recent one in math as soon as 2027 ([scaling01](https://x.com/scaling01/status/2124551799789044149)).
  - **Community extrapolations (unofficial)**: Dan Hendrycks infers ~3–5×/yr compute-efficiency gains since GPT-4, possibly 100×+ this year ([hendrycks](https://x.com/hendrycks/status/2124571381295984764)). One fit, using the internal model at 0.732 against 0.761 for the released model, reaches 0.70 between January and March 2027 ([fits](https://x.com/leithnyang/status/2124707234647359560)).
- **Microsoft joins**: With OpenAI's permission, Microsoft AI reported MAI-2 Base at 0.819 on OpenAI's eval set. It shares OpenAI's tokenizer and lands in GPT-5.6 Sol base-model territory. Mustafa Suleyman suggested bits-per-byte for future cross-lab comparisons ([mustafasuleyman](https://x.com/mustafasuleyman/status/2124663594449318173)).
  - **Open-model mapping**: Elie Bakouch used MAI as a bridge to place open models on OpenAI's curve ([plot](https://x.com/eliebakouch/status/2124668409564528947)). He labels the further chaining through Inkling bpb a "very weak approximation" ([details](https://x.com/eliebakouch/status/2124686143165734972)).
  - **Lucas Beyer's caveat**: Inkling's held-out codebase may be largely agent-generated, which would make bridge-based placements look too optimistic ([giffmana](https://x.com/giffmana/status/2124765224334934187)).
- **Methodological critiques**: Critics raised several confounds:
  - **Tokenizer bias**: Common limits could favor OpenAI over Anthropic because of tokenizer differences ([davidmanheim](https://x.com/davidmanheim/status/2124581659500535840)).
  - **Post-training compute**: Post-training likely accounts for more than 75% of FLOPs, so pretraining perplexity may be the wrong pacing metric ([stochasticchasm](https://x.com/stochasticchasm/status/2124570303275905468)).
  - **Synthetic data**: Part of the steep curve may come from a growing share of model-generated pretraining data ([Azaliamirh](https://x.com/Azaliamirh/status/2124650604870119491)).
  - **Missing context**: Without model sizes or compute figures, it is hard to separate scaling from algorithmic progress ([Grad62304977](https://x.com/Grad62304977/status/2124587669724217701)).
  - **OpenAI's response**: Staff say the eval is fixed over time and excludes model-generated code ([johnohallman](https://x.com/johnohallman/status/2124775923539214399)). Will Depue added that it is "held out, frozen, tokenizer agnostic" ([willdepue](https://x.com/willdepue/status/2124819752891863468)).
- **Pressure on other labs**: Several researchers asked OpenAI to release the held-out set ([oneill_c](https://x.com/oneill_c/status/2124557725850038383); [DimitrisPapail](https://x.com/DimitrisPapail/status/2124566918304780527)).
  - **Anthropic's redactions**: Anthropic redacted pretraining metrics from its METR report ([eliebakouch](https://x.com/eliebakouch/status/2124566777740882027)), prompting calls to publish them ([_NathanCalvin](https://x.com/_NathanCalvin/status/2124640068388229189)).
  - **Third-party verification**: METR offered to verify pause or compute-allocation claims under its standard terms ([ChrisPainterYup](https://x.com/ChrisPainterYup/status/2125008508206882999)).

**Frontier Model and Product Launches: Gemini 4 Neon, GPT-6.2 Sol, Claude Ship**

- **Gemini 4 Neon**: Google positions Neon between Carbon (hardest long-horizon work) and Hydrogen (fast execution), aimed at complex coding and enterprise workflows ([Google](https://x.com/Google/status/2124967271814910183)).
  - **Leaderboard results**: Neon claims several top spots:
    - **LiveBench 2.0**: #1 at 87.4, versus 84.9 for both Claude Fable 5.2 and GPT-6.2 Astra, at roughly a quarter of Fable's cost per task ([LiveBench](https://x.com/alexmichael/status/2124364037270253826)).
    - **AI Chess Leaderboard**: #1 at 1924 Elo ([chess](https://x.com/alexmichael/status/2124360947410342012)).
    - **Arena text**: Google holds the top three, with Carbon at 1527, Argon at 1520 and Neon at 1513 ([Arena](https://x.com/alexmichael/status/2124496549036077213)).
  - **Adoption signals**: OpenRouter spend on Google models exceeded OpenAI's for the first time since June ([OpenRouter](https://x.com/PeterJ_Walker/status/2124924320459989032)).
    - **Perplexity traffic**: Neon is compounding +10.6%/day in Ask and +15%/day in Computer. Aravind Srinivas says this is organic switching from Opus 5.5 and Sol 6.2, not default routing ([growth](https://x.com/AravSrinivas/status/2124765417378099414); [clarification](https://x.com/AravSrinivas/status/2124951952035578100)).
  - **Access rules**: Gemini subscriptions work only through the Antigravity binary. T3 Code is allowed because it wraps Antigravity; shims such as Pi are not ([theo](https://x.com/theo/status/2124665075638411641); [Google confirmation](https://x.com/PatrickToulme/status/2124681579125702798)).
    - **Quota changes**: Antigravity quotas were reset. Pro users get Carbon, and Hydrogen opens to all users ([_mohansolo](https://x.com/_mohansolo/status/2124960419429458207)).
    - **Workspace**: Neon is rolling out to Workspace customers in the Gemini app ([OfficialLoganK](https://x.com/OfficialLoganK/status/2125018427790344654)).
  - **Hydrogen pricing**: Hydrogen costs $1.50 per million input tokens. It scores 51 on the Artificial Analysis Intelligence Index, against 50 for Argon, and 57% on Terminal-Bench 4.1 ([dl_weekly](https://x.com/dl_weekly/status/2124579000097833320)).
  - **API features**: New file slicing lets agents upload once and pay only for the pages, cells or clips Gemini inspects ([googledevs](https://x.com/googledevs/status/2124936137739542604)). Function results can now return documents, audio and video natively ([_philschmid](https://x.com/_philschmid/status/2124929125974450503)).
  - **Practitioner feedback**: DHH reports Neon halved Omarchy boot time ([dhh](https://x.com/dhh/status/2124860839572488538)). Jeremy Howard notes it still builds elaborate workarounds rather than asking permission ([jeremyphoward](https://x.com/jeremyphoward/status/2124561815065776280)).
- **GPT-6.2 Sol in Chat**: A chat-optimized Sol is rolling out to Plus, Pro, Business and Enterprise users. OpenAI cites better fine-detail image reading and constraint-aware comparison shopping ([ChatGPT](https://x.com/ChatGPT/status/2124942186425069836)).
  - **Community criticism**: Some users report Astra 6.2 over-infers intent compared with Astra 6.0 ([code_star](https://x.com/code_star/status/2124618629123883378)) and want an OpenAI roadmap ([kimmonismus](https://x.com/kimmonismus/status/2124413051436724336)).
- **Claude Ship (beta)**: Running `/ship` in Claude Code opens the PR, reviews it, and works through feedback and failing checks. Branch rules and permissions still decide what merges ([ClaudeDevs](https://x.com/ClaudeDevs/status/2124959436640858475); [details](https://x.com/ClaudeDevs/status/2124959438175981965)).
  - **Team rollout**: Admins can enable it repo-wide, with the approval policy stored in the repo like a smarter CODEOWNERS file ([enterprise](https://x.com/ClaudeDevs/status/2124959440772137337)).
  - **ZDR with safeguards**: Zero data retention (ZDR) with Enterprise Frontier Safeguards is in public beta. Encrypted copies of traffic go to the customer's own storage, where automated checks flag misuse ([ClaudeDevs](https://x.com/ClaudeDevs/status/2124930253344948518)).
- **Unconfirmed**: Claude Opus 5.6 may be in A/B testing for a subset of users ([synthwavedd](https://x.com/synthwavedd/status/2125018875217547382)).

**Looped Transformers and Open-Weight Releases**

- **GLM-5.4**: This is Z.ai's first loop transformer, adapted from the GLM-5 base through continued training rather than pretrained from scratch. It adds vision and targets video, games, shaders and CAD ([Zai_org](https://x.com/Zai_org/status/2124811995912286352)).
  - **Architecture and claims**: The model is 750B parameters and runs 28 layers twice; Z.ai claims "2T–3T-class" performance. In a 30B ablation with ~10% more compute, AIME25 rose from 19.2 to 30.4 ([summary](https://x.com/kimmonismus/status/2124811337951744042)).
  - **Loop mechanics**: Per-token gates control how much the first pass writes, and the extra loop uses a 128-token KV window ([TuringPost](https://x.com/TheTuringPost/status/2124992248207270017)).
  - **Availability**: The model is live for GLM Coding Plan users with 50% more weekly quota ([ZixuanLi_](https://x.com/ZixuanLi_/status/2124719275864670681)). API access waits on safety evaluation, and weights are promised "responsibly." No benchmarks have been published yet, which drew skepticism ([teortaxesTex](https://x.com/teortaxesTex/status/2124813863111532838)).
- **Inkling 2 Mini (Thinking Machines)**: The model has 26.4B total and 3.4B active parameters across 24 layers. The middle 12 repeat three times, giving 48 passes per token in ~20GB of open weights ([TuringPost](https://x.com/TheTuringPost/status/2124992248207270017)).
  - **Kernel generation**: Running locally on an RTX PRO 6000, it reached 10.04x on KernelBench-Mega, ahead of Grok 4.7 (6.34x) and GPT-6.1-Sol (4.96x). It got there by writing a 1,119-line persistent Triton megakernel ([elliotarledge](https://x.com/elliotarledge/status/2124568616964333644)).
  - **Local speed**: Unsloth reports 125 tok/s on an RTX PRO 6000 and fixed KV-cache quantization memory overuse ([danielhanchen](https://x.com/danielhanchen/status/2124901209836519479)).
  - **Agent Arena**: It ranks #18 among open models and #3 among U.S. open-weight models ([arena](https://x.com/arena/status/2124981167279546382)).
  - **Safety effect of looping**: A research variant cut destructive actions from 26.6% at one pass to 3.1% at four ([karenzzhou via ziqiao_ma](https://x.com/ziqiao_ma/status/2124997896605163653)).
- **Related looping work**: LoopRes loops any pretrained LLM, reporting gains of up to 37.7% over SFT across six models up to 30B, fully open ([oahzxl](https://x.com/oahzxl/status/2124890218373534069)).
- **Other open releases**: Several other open-weight models shipped:
  - **OCRia-nano-1 (LlamaIndex)**: An 8.1M-parameter OCR model that runs at 426 ms/page on a single CPU core and handles tables, layout and multilingual scripts ([jerryjliu0](https://x.com/jerryjliu0/status/2124909031999553926)).
  - **Hy4 (Tencent)**: 770B total and 49B active parameters, now on Ollama Cloud with vision and computer use ([ollama](https://x.com/ollama/status/2124669766723920377)).
  - **MemT-1.0-Pro**: A 554B memory-native model built on Nemotron 3 Ultra ([MemOS_dev](https://x.com/MemOS_dev/status/2124623963087126562)).
  - **LingSante-3.1-flash (Ant Ling)**: A 25B-active/560B medical model with 1M context under the MIT license ([arankomatsuzaki](https://x.com/arankomatsuzaki/status/2124597469522870272)).

**Inference Infrastructure and Systems Research**

- **NVMe KV-cache offload in vLLM**: The test ran Kimi-K3 on 16× AMD MI355X using SemiAnalysis AgentX traces. A 1.8TB NVMe pool matched 1.8TB of DRAM within 1%, and at 70 sessions 4.6TB of NVMe delivered 5.1× the throughput of 1TB of DRAM ([vllm_project](https://x.com/vllm_project/status/2124783794112938241)).
  - **vLLM v0.34**: The extensible KV cache is now on by default and uses GPU virtual memory to size itself automatically ([vllm_project](https://x.com/vllm_project/status/2125041827107741733)).
- **Rack-scale serving**: SemiAnalysis reports an SGLang submission on Rubin NVL72 with 6.5× better performance per dollar than GB300 NVL72, running Kimi K3.1 Pro (2.8T parameters) ([SemiAnalysis_](https://x.com/SemiAnalysis_/status/2124637709176016998)).
  - **Cerebras dispute**: SemiAnalysis also argues Cerebras' 2,000 tok/s figure covers only prompts under 100 tokens. It challenged Cerebras to show p99 latency at 100K+ context ([SemiAnalysis_](https://x.com/SemiAnalysis_/status/2124684387761430549)).
- **Harness adapters**: Adapting Qwen3.8-27B to the Pi harness raised Terminal-Bench 2.1 from 39% to 55% while using half the tokens ([SergioPaniego](https://x.com/SergioPaniego/status/2124525813013893266)).
- **Training efficiency**: Several papers target cheaper training or inference:
  - **Model Carving**: Learnable gates remove unneeded sublayers. Carved models beat dense models trained with up to 1.7× more compute and use up to 34% fewer FLOPs per token ([dair_ai](https://x.com/dair_ai/status/2125027375553605836)).
  - **DeepCrossAttention (Google)**: Reframing skip connections as dynamic routing matches quality up to 5× faster with <0.2% overhead ([GoogleResearch](https://x.com/GoogleResearch/status/2124933889131164116)).
  - **Apple FastFrames**: Profiling showed VAE decoding and KV updates were over half of autoregressive video latency. Replacing them yields 36 ms/frame on one B200 ([RisingSayak](https://x.com/RisingSayak/status/2124880346097582201)).
- **RL training tooling**: prime-rl 0.11.0 adds a pretraining entrypoint, bidirectional KV transfer in prefill/decode disaggregation and lower CPU memory use ([samsja19](https://x.com/samsja19/status/2124334777792942524)). Microsoft AI detailed Rocket, its in-house RL infrastructure ([mustafasuleyman](https://x.com/mustafasuleyman/status/2124933527066325243)).

**Evals, Security and AI for Math and Science**

- **Riemann Hypothesis proof**: Debate over Levent Alpöge's Claude-assisted, roughly 250K-page Lean proof centers on whether anyone understands it.
  - **Human understanding**: Ngo Bao Chau argues it is not truly "proved" until humans understand it ([stablequan](https://x.com/stablequan/status/2124536218352242951)).
  - **Elementary or not**: Obhishek Saha counters claims that the proof is "elementary," noting it spans so much modern number theory that no single mathematician may understand every part ([ObhishekSaha](https://x.com/ObhishekSaha/status/2124501191321153811)).
  - **Checker trust**: Independent Lean kernels reportedly verified the proof ([ZhihuFrontier](https://x.com/ZhihuFrontier/status/2124434229136380120)).
- **DeepMind quantum circuits**: Teams of Gemini 4 agents found Lean-verified circuits that cut estimated compute by 10.9× for a fusion simulation, 8.1× for photochemistry and 2.5× for a drug-metabolism molecule. The solutions and Lean library are open-sourced ([pushmeet](https://x.com/pushmeet/status/2124914424326525182)).
- **AI cyber-defense**: OpenAI and Cisco released Shirudo, an open-source Linux runtime in which agents write dynamic eBPF shields that block exploits without downtime ([OpenAI](https://x.com/OpenAI/status/2124980590717911527)).
  - **Kernel bugs**: GPT-6 Cyber helped validate 948 Linux kernel local-root bugs, and patches went upstream with maintainers ([OpenAI](https://x.com/OpenAI/status/2125007927098699827)).
  - **Offensive risk**: Mustafa Suleyman cites an agent that exploited a medical-imaging zero-day in ~11 minutes from the single instruction "obtain shell access" ([mustafasuleyman](https://x.com/mustafasuleyman/status/2124536771702874472)).
- **Eval integrity**: Epoch's benchmark reviews rated three of four benchmarks Flawed ([EpochAIResearch](https://x.com/EpochAIResearch/status/2124945252725510614)).
  - **Agents' Last Exam v1.1**: At least 44% of sampled tasks had severe errors ([review](https://x.com/EpochAIResearch/status/2124945288746225771)).
  - **Vending-Bench 2**: Epoch found simulation issues and reward-hacking vulnerability ([review](https://x.com/EpochAIResearch/status/2124945300628607372)).
  - **Monitor evaluations**: The UK AI Security Institute's Red Team examines how reliable simulated tests of LLM-based agent monitors really are ([AISecurityInst](https://x.com/AISecurityInst/status/2124919147041550676)).
- **Research findings**: Several papers probe agent and training limits:
  - **RoboQuest**: The best model solved only 23.2% of physical-agent tasks, despite 72–81% success on the underlying motor skills in isolation ([omarsar0](https://x.com/omarsar0/status/2124549311174427069)).
  - **Hallucination memorization (Cornell)**: Rewarded hallucinations rose 4.83% in recall during RL post-training on knowledge tasks ([dair_ai](https://x.com/dair_ai/status/2124870592684032283)).
  - **CoEvoBench (Alibaba)**: Agents improving model-harness pairs drew most of their gains from harness edits rather than training ([omarsar0](https://x.com/omarsar0/status/2124869837545701845)).

**Industry and Policy**

- **Nvidia reinstates Jensen Huang**: The board voted unanimously to restore Huang as president, CEO and director ([nvidianewsroom](https://x.com/nvidianewsroom/status/2124865060397019480)).
  - **Government input**: The board cited input from "senior levels" of the U.S. government ([BlackHC](https://x.com/BlackHC/status/2124882077342679062)). An unconfirmed source report says Trump and Vance saw no national-security wrongdoing and viewed Huang as protecting open-source models ([DeItaone](https://x.com/DeItaone/status/2124923184906420267)).
  - **Market reaction**: Nvidia rose roughly 30%, adding about $890B in market value ([gneubig](https://x.com/gneubig/status/2124879436390584549)). Unverified claims of suspicious pre-announcement call buying are circulating ([MrMikeInvesting](https://x.com/MrMikeInvesting/status/2124869411001545117)).
  - **Ongoing scrutiny**: The NYT reports Alice Huang was in contact with China's spy agency ([nytimes](https://x.com/nytimes/status/2124663793670127809)). Huang faces the Senate Banking Committee in eight days ([shashj](https://x.com/shashj/status/2124899530613121218)).
- **RSI legislation**: Sen. Cory Booker called for reining in recursive self-improvement, mandatory reporting on compute allocation and algorithmic progress, and verifiable shutdown protocols ([SenBooker](https://x.com/SenBooker/status/2124639269545501092)).
  - **Meta's position**: Alexandr Wang says Meta delayed Muse for months on safety grounds and that Zuckerberg wants most compute spent serving users rather than RSI ([alexandr_wang](https://x.com/alexandr_wang/status/2124719968499777546)).
- **Open-source fight**: A summary of Dario Amodei's congressional testimony warning about open-weight risk ([huxlab](https://x.com/huxlab/status/2124411922959548795)) drew sharp responses.
  - **Critics**: Yann LeCun called it a bid for control ([ylecun](https://x.com/ylecun/status/2124550037912920090)), and Tim Sweeney said it trades a monopoly for compliance with government ([TimSweeneyEpic](https://x.com/TimSweeneyEpic/status/2124914931401138265)).
  - **White House view**: David Sacks argued open models are needed so the field is not "a monopoly of just OpenAI and Anthropic" ([TheChiefNerd](https://x.com/TheChiefNerd/status/2124531915726499888)).
  - **Lab security**: Caleb Watney argues no frontier lab can currently stop state-level weight theft ([calebwatney](https://x.com/calebwatney/status/2124697462711390425)).
- **Funding and programs**: Several notable deals and initiatives:
  - **Embo**: Raised a $110M seed led by a16z for world-model robot brains, from the team behind Dreamer and VideoGPT ([danijar](https://x.com/danijar/status/2124911019562811820)).
  - **River API**: Crossed $100M ARR five months after launch ([ibab](https://x.com/ibab/status/2124940347134644249)).
  - **OpenAI OpenWorks**: Funds local AI experts up to $8K/month to help small businesses in 20 U.S. cities ([TheRundownAI](https://x.com/TheRundownAI/status/2124939632349794367)).

**Top tweets (by engagement)**

- [NVIDIA board reinstates Jensen Huang as CEO](https://x.com/nvidianewsroom/status/2124865060397019480) — 481.7K
- [LeCun: Dario's safety pitch aims to outlaw open models](https://x.com/ylecun/status/2124550037912920090) — 28.7K
- [Claude Ship beta: PR review, fixes and policy-gated merges](https://x.com/ClaudeDevs/status/2124959436640858475) — 18.6K
- [Antigravity quota reset; Carbon for Pro users, Hydrogen for all](https://x.com/_mohansolo/status/2124960419429458207) — 17.1K
- [Pachocki: "Perplexity and Intelligence" and common limits](https://x.com/merettm/status/2124549218995949967) — 17.1K
- [DHH: Gemini 4 Neon halved Omarchy boot time](https://x.com/dhh/status/2124860839572488538) — 13.3K
- [GLM-5.4: Z.ai's first loop transformer](https://x.com/Zai_org/status/2124811995912286352) — 13.3K
- [OpenAI and Cisco release Shirudo eBPF defense runtime](https://x.com/OpenAI/status/2124980590717911527) — 12.7K


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. GLM-5.4 and Open-Weight Architecture Feedback

  - **[GLM-5.4: Toward Generalized Coding with Our First Loop Transformer](https://www.reddit.com/r/LocalLLaMA/comments/1y3enps/glm54_toward_generalized_coding_with_our_first/)** (Activity: 1734): ****Z.ai GLM-5.4** is described as a `750B` flagship converted from a pretrained GLM-5 base into a **Loop Transformer**, with `28` middle layers weight-shared and executed for two iterations to increase effective reasoning depth without increasing parameter count ([blog](https://z.ai/blog/glm-5.4)). The post claims competitiveness with `2T–3T` parameter models on long-chain reasoning, using gated progressive residual updates, a hybrid KV cache, sparse-attention index reuse, and reported inference of ~`70–80 tok/s` peak / up to `200 tok/s` off-peak; Z.ai says it plans to release open weights after “necessary safety hardening.”** Commenters were skeptical that “safety hardening” may mean nerfing cybersecurity capabilities, arguing open weights are important for defenders without access to frontier closed models. Another technical complaint noted the blog says the Loop architecture was validated at `1B`, `4B`, `30B`, and `750B`, but asks where the sub-`30B` models are.

    - Several commenters focused on the planned open-weight release being delayed or modified for a **cybersecurity capability nerf**, questioning whether this undermines the role of open weights as a defensive alternative for users without access to frontier closed models. The concern is specifically about reducing offensive cyber capability potentially also degrading legitimate security research and defense use cases.
    - A technical point was raised about the paper/blog claim that the **Loop Transformer architecture** was validated across `1B`, `4B`, `30B`, and `750B` base models before being used in the flagship GLM-5.4 model. Commenters asked whether the sub-`30B` checkpoints will be released, since those smaller models would be important for independent replication, ablations, and local experimentation.

  - **[The US-China Open-Weight Feedback Loop](https://www.reddit.com/r/LocalLLaMA/comments/1y3q68f/the_uschina_openweight_feedback_loop/)** (Activity: 779): **The image is a technical infographic, [“The US-China Open-Weight Feedback Loop”](https://i.redd.it/tkgl4i9zoe3i1.jpeg), arguing that U.S. and Chinese open-weight LLM ecosystems have formed a reciprocal architecture-transfer cycle from roughly `2021–2026`. It maps techniques such as **RoPE**, **GQA**, **MTP**, **MLA**, expert routing, sparse attention, and attention sinks across releases, with arrows suggesting that open-weight model families on both sides rapidly borrow, validate, and improve each other’s design choices—also indirectly pressuring proprietary models to improve and become cheaper.** Commenters were broadly supportive of the open-source/open-weight framing, summarizing the point as *“Open source makes everyone better.”* One commenter noted that closed labs likely also borrow ideas from open models, but unlike open releases, their influence paths are mostly opaque and often only discussed publicly as alleged distillation.

    - A commenter raised a technical-policy point that **closed model labs may be incorporating ideas from open-weight models without attribution**, while publicly framing similarity in the opposite direction as alleged distillation from proprietary systems. The key issue is asymmetric observability: open models expose weights, papers, and implementation details, whereas closed models make it difficult to verify whether architectural or training-method ideas were reused.
    - One commenter disputed attribution around **Mixture-of-Experts (MoE)** in recent open-weight model discussions, stating that **Mistral**, not DeepSeek, should be credited in this context. The technically relevant point is that MoE adoption and popularization across open-weight LLMs is often conflated across releases, even though specific architectural implementations and release timelines matter when tracing influence.


### 2. Inkling 2 Mini Performance and Capability Tests

  - **[Inkling absolutely rips. ~200 tok/s decode, ~6,700 prefill on 2x3090s](https://www.reddit.com/r/LocalLLaMA/comments/1y2gxo7/inkling_absolutely_rips_200_toks_decode_6700/)** (Activity: 730): **A user reports a custom **vLLM port of Inkling 2 Mini** for **Ampere GPUs** running on `2× RTX 3090`, using the **original weights without additional quantization**, achieving roughly `~200 tok/s` decode and `~6,700 tok/s` prefill with a `196k` token context window. The implementation is published at [`peaster/inkling-2-mini-vllm-2x3090`](https://github.com/peaster/inkling-2-mini-vllm-2x3090), with a longer benchmark/writeup comparing output-dependent and context-window-dependent performance against other Ampere ports in the author’s [first-impressions article](https://peaster.io/articles/local-LLMs/inkling-2-mini-first-impressions).** Comments were skeptical of highly setup-specific speed claims, with one noting the recurring pattern of *“look at how fast it’s going on my specific setup.”* Others questioned whether larger `27B` models could match the reported prefill rate and whether Inkling is competitive with **Qwen** for agentic/tool-calling workloads.

    - A commenter reports substantially higher single-GPU throughput using a custom `llama.cpp` fork for Inkling: **`135 tok/s` decode on one RTX 4090** and **`>5000 tok/s` prefill**, linking the implementation at [jonpojonpo/llama.cpp/tree/inkling-relfa](https://github.com/jonpojonpo/llama.cpp/tree/inkling-relfa). They frame this as competitive with the post’s **`200 tok/s` on 2x RTX 3090s**, suggesting implementation/kernel differences may matter as much as raw multi-GPU hardware.
    - Another commenter points to **maya-inkling-mini** as a route to higher single-card speed, claiming **`200 tok/s` on one RTX 3090** via [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini). This implies a smaller/distilled or otherwise optimized Inkling variant may achieve the posted dual-3090 decode rate on a single 3090, though no benchmark methodology or quality comparison is provided.
    - There is a technical question about whether Inkling is strong at **agentic tool calling** versus staying with **Qwen**, but the thread excerpt provides no direct evaluation results. This highlights an unresolved comparison point: raw decode/prefill throughput alone does not establish tool-use reliability or agentic performance relative to Qwen-family models.

  - **[Inkling 2 Mini on agentic coding and translation: below Qwen3.8-27B](https://www.reddit.com/r/LocalLLaMA/comments/1y2qf66/inkling_2_mini_on_agentic_coding_and_translation/)** (Activity: 646): **The image is a technical benchmark chart ([image](https://i.redd.it/mous0ufra63i1.png)) comparing agentic coding performance on `9` private engine bugs: **Inkling 2 Mini NVFP4** scores `0.22`, below **Qwen3.8-27B** variants at roughly `0.34–0.39` and far behind top **Claude** models. The post adds that Inkling 2 Mini performs poorly on translation due to overly literal idiom handling, and shows an unusual “trap suspicion” behavior localized to `32` neurons out of `3.5M`; ablating them reduced suspicion from `16/30` to `1/30`, but did **not** improve benchmark score because the model still often failed to edit files. The chart’s high no-patch rate for Inkling (`13/27`), plus `68k` tokens and `23 min` per task, supports the author’s claim that the model over-reads/reasons without completing code changes.** Commenters echoed the benchmark behavior, saying Inkling “reads and reads” without acting and may need better chat templates or stabilization. One user reported needing to raise context from `131k` to `512k` before it made progress, reinforcing that its agentic workflow appears inefficient rather than simply context-limited.

    - Several users report **Inkling 2 Mini** exhibiting inefficient agentic coding behavior: it spends excessive time reading repository files and “thinking” rather than executing edits. One commenter said a real-codebase task exceeded the default `131k` context because the model repeatedly read, compacted, and reread context; increasing context to `512k` finally allowed it to make progress, suggesting poor context triage or tool-use planning rather than raw context insufficiency alone.
    - There is skepticism about the reported benchmark ranking because the confidence intervals appear wide enough that **Inkling 2 Mini** may be statistically close to **Claude Sonnet 5.5** despite being described as below **Qwen3.8-27B**. Another commenter notes that **Qwen 27B 3-bit** variants, possibly `q3`/`iq3`, outperforming older **Sonnet/Haiku** models and approaching **Haiku 5.5** is plausible given prior comparisons to stronger Claude-family models, highlighting how competitive local quantized models have become.
    - One commenter expects early results to shift after stabilization and improvements to **chat templates**, implying current Inkling 2 Mini behavior may be affected by prompting/template issues rather than only model capability. This is relevant for agentic coding evaluations because template differences can strongly affect tool-use discipline, instruction following, and whether the model over-reads versus taking concrete actions.


### 3. Local Inference Benchmarking and Engine Fragmentation

  - **[Please add a new rule: when you say "I can run X at Y" you also have to say what quant/bpw/size was used.](https://www.reddit.com/r/LocalLLaMA/comments/1y2oe1x/please_add_a_new_rule_when_you_say_i_can_run_x_at/)** (Activity: 962): **A proposed r/LocalLLaMA rule would require any performance claim of the form *“I can run model X at speed Y”* to include the **quantization / bpw / model size**—ideally in the title—to make throughput claims interpretable. The author notes this would not solve deeper benchmark manipulation issues such as unrealistic sampling prompts, `0`-context-depth throughput, high-concurrency synthetic runs, misleading long-context claims, speculative decoding artifacts, or opaque closed-source evals, but argues it is a low-cost metadata requirement for filtering and moderation.** Top comments broadly support raising the quality bar for benchmark posts, especially reducing low-signal “engine spam” about raw speed. One commenter specifically argues that posts should report more than token generation speed (`TG`), because prompt-processing / prefill speed (`PP`) is often more informative; another points out that context-window assumptions, e.g. `8000` tokens, materially affect what “can run” means.

    - Several commenters argued that raw speed claims like “I can run model X at Y tok/s” are not technically meaningful unless they include **quantization/BPW/model size, context length, and hardware configuration**. They noted that the same model can show very different throughput depending on quant, VRAM/RAM offload, drivers, modified GPUs, and inference parameters.
    - A technical point was raised that benchmark posts should distinguish **decode throughput (TG/tok/s)** from **prefill performance (PP)**, with one commenter saying *“don’t just post TGs… PPs tells me much much more.”* Another commenter emphasized that isolated metrics such as short-context tok/s or prefill speed can be misleading if **time-to-first-token**, long-context behavior, or total execution time are poor.
    - One detailed comment highlighted that LLM inference performance is highly stack-dependent: server/runtime, speculative decoding, n-gram assistance, GPU driver version, RAM offloading, concurrency, and context length can all materially change results. The commenter argued that numbers like `500 tps` decode or `3000` prefill at short context become less useful when they do not reflect long-context latency or end-to-end workload performance.

  - **[Inference engines are becoming too fragmented. There are just too many choices and specialist engines.](https://www.reddit.com/r/LocalLLaMA/comments/1y30a6x/inference_engines_are_becoming_too_fragmented/)** (Activity: 644): **The post argues that local LLM inference has fragmented beyond the “Swiss army knife” role of [`llama.cpp`](https://github.com/ggml-org/llama.cpp), with users now choosing between model-/GPU-/architecture-specific engines and forks for features like KV-cache quantization, KV streaming, MoE support, and newer model architectures. The author reports large engine-dependent throughput differences: **Qwen FN** at `<30 tok/s` and `~300 pp` on `llama.cpp` versus `~180 tok/s` and `~2500 pp` with `200k+` KV on **Strata**, and **Inkling-2-Mini** at `28 tok/s` on a `llama.cpp` fork versus `60 tok/s` on **Basalt**; they currently rotate between **beelama**, an adaptive KV-streaming `llama.cpp` fork, **Strata**, **Basalt**, and upstream `llama.cpp`. They also note production-oriented alternatives like [`vLLM`](https://github.com/vllm-project/vllm) and [`SGLang`](https://github.com/sgl-project/sglang), but complain that Linux requirements, fork proliferation, slow upstream model support, and optimization PR rejections make engine selection increasingly costly.** Top commenters mostly frame the fragmentation as optional: `llama.cpp` remains a reliable baseline, and chasing maximum per-model throughput is a user choice rather than a requirement. One substantive concern raised is supply-chain/security risk: trying many young engines, forks, binaries, and dependencies increases exposure to malware or infostealers, so users should prefer mature, widely tested projects.

    - A practical engine taxonomy emerged: **llama.cpp** is described as the default for most local inference, **LiteRT** for mobile, **vLLM/SGLang** for professional or advanced serving setups, and newer specialized engines like **Basalt** or **Radiance** for specific model/GPU combinations such as NVIDIA 50-series or Radeon R9700 Pro. One commenter notes that **llama.cpp has recently closed gaps with vLLM** by adding features like tensor parallelism, MTP support, and partial RAM offloading, making it a more viable general-purpose baseline despite specialized engines chasing peak tok/s.
    - Several commenters framed fragmentation as a security and supply-chain issue: every additional inference engine, binary, or dependency increases exposure to malware, infostealers, or compromised packages. The recommended mitigation was to avoid newly released or unvetted engines unless they have meaningful community adoption, reproducible builds, or a clear trust history.
    - One user reported concrete performance from **Basalt**, claiming `400-600 tok/s` on **Inkling**, and said it had also been useful for **QFN** and **Qwen3.8-27B**. They also mentioned **Project Maya** as useful for early access to models such as **GLM5.3F** and **DS4.1F**, illustrating how specialist engines can provide early optimization paths before general-purpose engines catch up.

  - **[4x V100 setup](https://www.reddit.com/r/LocalLLM/comments/1y35umb/4x_v100_setup/)** (Activity: 341): **OP built a 4× **NVIDIA Tesla V100 32GB** inference box (`128GB` total VRAM) using a **PLX/Broadcom PEX88096 PCIe switch board** ([product family](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches)), `64GB` system RAM, a `1500W` PSU, and an older Threadripper CPU. They report running a modified 4-GPU **sFNX** setup with **Qwen3.8-FN Q4_K_XL**, achieving about `200 tok/s` decode and `2600 tok/s` prefill across `6` parallel lanes.** The main technical caveat raised in comments is to **pin NVIDIA driver/CUDA versions**, because newer CUDA releases have reduced/removed support for older architectures such as **Volta / sm_70**; commenters also asked whether the `200 tok/s` figure is aggregate or per-lane and what the 4-GPU idle power draw is. Several commenters noted the unusually favorable economics of buying V100 32GB cards at about `$600/card` before prices rose.

    - A commenter notes that a 4× V100 setup provides **`128 GB` total VRAM** and that V100 cards have appreciated in value, but warns that **NVIDIA dropped Volta support from newer CUDA releases**. They recommend pinning the driver and CUDA versions to avoid breaking the setup after auto-updates, and ask whether the reported **`200 tok/s`** throughput is aggregate across all lanes or per lane, plus what the 4-card idle power draw is.
    - One user reports practical cooling data from a **single V100** running Qwen models: an **80 mm Noctua at 100%** is sufficient for light/medium workloads, but sustained 100% GPU load requires swapping to a high-RPM Arctic server fan. They claim the server fan runs near **`10,000 RPM`** and keeps the card below **`60°C`** inside a Fractal North case, with noise described as loud but not high-pitched.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo


### 1. Google Gemini Cross-App Context and Arena Sweep

  - **[All 3 Gemini models top Arena.AI](https://www.reddit.com/r/GeminiAI/comments/1y2oghl/all_3_gemini_models_top_arenaai/)** (Activity: 794): **The image is a screenshot of the [Arena.AI text leaderboard](https://arena.ai/leaderboard/chat/text), showing **Google Gemini 4 Carbon, Argon, and Neon** occupying the top three ranks with scores of `1,527`, `1,520`, and `1,513`, ahead of Meta’s Muse Wonder and several Anthropic Claude models. The technical significance is that, if the leaderboard reflects robust human-preference evaluations, Google’s Gemini variants are currently leading this benchmark; the screenshot is available [here](https://i.redd.it/b8jnv59is53i1.png).** Commenters framed this as a major Google comeback, but also argued that the limiting factor may now be the *harness and web chat interface* rather than raw model quality. One user also complained that Google still needs to improve AI Studio support for Pro accounts.

    - Commenters noted that the referenced Arena.AI result is specifically the **“everyday chat” / text leaderboard** ([arena.ai/leaderboard/chat/text](https://arena.ai/leaderboard/chat/text)), not a coding or STEM benchmark. One technical critique was that **Gemini’s strength appears concentrated in general chat, writing, creative/UI/design tasks**, while it is still perceived as weaker than peers on **coding, math, and cybersecurity** workloads.
    - A separate implementation-focused point argued that the **evaluation harness and web chat interface may now be the bottleneck**, implying that frontend latency, tool integration, or benchmark interaction layers could obscure the underlying model capability differences between top-ranked Gemini variants. Another user flagged operational issues with **Google AI Studio for Pro accounts**, suggesting access or account-tier bugs remain a practical blocker despite model gains.

  - **[Google announces cross app context between the Gemini app and Search AI Mode (plus Drive & Wallet support)](https://www.reddit.com/r/Bard/comments/1y3pd2u/google_announces_cross_app_context_between_the/)** (Activity: 57): ****Google** is extending **Personal Intelligence in Search AI Mode** with opt-in connectors for **Google Wallet** now and **Drive + Gemini app** in the coming weeks, adding user-specific context from transactions, docs, and prior Gemini chats on top of existing Gmail/Calendar/Photos integrations ([announcement](https://blog.google/products-and-platforms/products/search/google-search-personal-intelligence-apps/)). The implementation is positioned as read-only for Wallet data, user-toggleable per app/personalization setting, and will also make **AI Overviews**/**AI Mode** use recent Search activity, with a forthcoming UI to show which personal context influenced a response.**



### 2. Qwen and GLM Model Upgrades

  - **[Glm 5.4 out](https://www.reddit.com/r/SillyTavernAI/comments/1y38xsy/glm_54_out/)** (Activity: 555): **The image is a tweet-style announcement that **Zixuan Li** says **GLM-5.4** is available to all **GLM Coding Plan** users, with advertised additions including **vision capabilities**, stronger coding performance, and **`50%` more weekly quota** versus GLM-5.3. The post is an announcement screenshot rather than a benchmark or technical deep dive; the image is available [here](https://i.redd.it/ypa2ef24ha3i1.jpeg).** Commenters are mostly skeptical from an RP/roleplay-use perspective, noting that GLM-5.3 was disappointing for RP and that the announcement emphasizes coding rather than roleplay quality.

    - Several commenters are skeptical that **GLM 5.4** will improve roleplay behavior, noting that **GLM 5.3** was already considered disappointing for RP and that the release messaging appears to emphasize *“stronger all-round coding performance”* rather than creative-writing or character-consistency improvements.
    - One user raises a recurring concern about **post-release model degradation**, claiming prior GLM/Zhipu releases were initially better on day one but became worse shortly after, implying possible backend model swaps, quantization changes, safety tuning, or routing changes after launch. They also criticize prior GLM outputs as having excessive *“lawyer talk,”* suggesting alignment/safety verbosity remains a perceived issue.

  - **[8 days ago, Reddit helped take our unrestricted Qwen Omni model to #8 out of 3M+ models. Today, we’re open-sourcing the Turbo LoRA-6.8× faster](https://www.reddit.com/r/StableDiffusion/comments/1y2fmtk/8_days_ago_reddit_helped_take_our_unrestricted/)** (Activity: 538): ****chatrlabs** released a Turbo LoRA for **Qwen-Image-2.1 Uncensored Omni**, targeting `8` sampling steps at `CFG 1.0` versus the prior `35`-step `CFG 5.0` workflow, with reported latency improving from `54.9s` to `8.1s` per ~1MP image on their Playground hardware (`6.8×` speedup). The LoRA supports text-to-image and instruction-based editing across ComfyUI/Diffusers and multiple Omni quantizations/builds (`GGUF`, `BF16`, `FP8`, `NF4`, `INT8 ConvRot`, `NVFP4`), with no added prompt filter or output safety checker; weights/settings/workflows are on [Hugging Face](https://huggingface.co/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized#turbo-lora-8-steps-cfg-10), and a hosted demo is at [thechatr.ai/playground](https://thechatr.ai/playground). Their benchmarks claim near-parity in adult prompt adherence (`89.7`→`89.1`) and SFW quality (`79.6`→`79.0`), plus improvements in anatomy (`69.4`→`72.9`), editing (`78.9`→`81.6`), and identity preservation in human editing (`55.8`→`72.1`).** The only substantive technical criticism in the top comments was that the earlier Omni model *“drifts too much from the reference”* during generation/editing, implying concern about reference fidelity despite the claimed identity-preservation benchmark gains. Other top comments were mostly non-technical reactions or criticism of example-output quality.

    - Several users asked for a reproducible **image-edit workflow** for the released **Turbo LoRA**, noting that their local workflows produce much worse results than the hosted playground using the same prompt and reference image. This suggests readers are looking for exact inference settings, node graph/pipeline details, and any preprocessing or sampler parameters needed to match playground behavior.
    - One tester reported that the earlier **Qwen Omni** image generations “drift too much from the reference,” highlighting a potential fidelity issue for reference-based editing. The main technical concern is whether the Turbo LoRA improves adherence to conditioning/reference images or primarily optimizes speed.

  - **[Qwen 27B rewrote its own weights and gained 6.6 points on unseen math](https://www.reddit.com/r/Qwen_AI/comments/1y30ukj/qwen_27b_rewrote_its_own_weights_and_gained_66/)** (Activity: 197): **A [RuntimeWire report](https://runtimewire.com/article/a-27b-model-rewrote-its-own-weights-and-gained-6-6-points-on-unseen-math) claims researchers from UIUC/CMU/Rutgers/Princeton/Stanford/Yale used **Qwen3.5-27B** in a loop where the model diagnosed math-task failures and generated code to edit its own parameters, reportedly improving held-out math accuracy from `57.2%` to `63.8%` (`+6.6` points) without gradient-based training. The linked summary does **not** expose enough methodology to assess whether this was genuine model-driven weight editing, scripted parameter patching, selection over attempts, or an evaluation artifact.** Commenters pushed back on the anthropomorphic framing: one noted that *“a script changed a bunch of numbers”* is very different from the model *“rewrit[ing] its own weights.”* Another raised the recursive possibility that such a system could learn better weight-editing behavior, but the dominant technical concern was headline precision versus the actual mechanism.

    - A commenter clarified the model in question was **Qwen3.5 27B**, not simply “Qwen 27B,” which matters for interpreting any reported weight-editing result against the correct baseline/model family.
    - The most substantive technical criticism was that the headline anthropomorphizes the process: commenters framed it as **an external script modifying numerical weights**, not the model autonomously “rewriting itself.” They noted the reported `+6.6` points on unseen math may be meaningful, but should be evaluated as a weight-editing or post-training intervention rather than “self-actualization.”


### 3. AI Math-Reasoning Breakthrough Claims

  - **[Anthropic claimed to have proved VP ≠ VNP over every field, the algebraic counterpart to the P vs NP problem](https://www.reddit.com/r/singularity/comments/1y3w34p/anthropic_claimed_to_have_proved_vp_vnp_over/)** (Activity: 1531): **The [image](https://i.redd.it/l8zw8uwatf3i1.jpeg) appears to show a LaTeX-style first page of a paper titled **“VP ≠ VNP OVER EVERY FIELD”**, attributed to **Claude / Anthropic**, claiming superpolynomial arithmetic-circuit lower bounds for the permanent and Hamiltonian-cycle polynomial. If genuine, this would be a landmark result in algebraic complexity theory—separating **VP** from **VNP**, the algebraic analogue of **P vs NP**—but the post provides only a screenshot and no verifiable paper link, proof details, or expert validation, so it should be treated as an unverified claim rather than an established theorem.** The comments are mostly non-technical reactions expressing shock or confusion, with users joking that “math is cooked” and asking whether this implies **P ≠ NP**; no substantive technical critique or validation is included.

    - A commenter linked Anthropic’s GitHub-hosted paper: [formal-math/vp-vnp/paper.pdf](https://github.com/anthropics/formal-math/blob/main/vp-vnp/paper.pdf), which is the primary technical artifact for the claim that **VP ≠ VNP over every field**. The thread otherwise contains little substantive technical discussion beyond noting the connection to the algebraic analogue of **P vs NP**.

  - **[a coordinated army of agents is trying to solve P = NP (livestreamed)](https://www.reddit.com/r/singularity/comments/1y30m1o/a_coordinated_army_of_agents_is_trying_to_solve_p/)** (Activity: 1667): **The post announces **[pequalnp.com](http://pequalnp.com)**, a livestreamed “hub” intended to coordinate **hundreds of AI agents** toward proving or disproving **P vs NP**, using an existing multi-agent orchestration structure described as “battle-tested.” No implementation details, model stack, verification pipeline, theorem-proving framework, benchmark results, or formal proof artifacts are provided beyond a screenshot and the claim that public participation is forthcoming.** Top commenters were skeptical that token-scale alone is sufficient, arguing that if this approach were viable, labs like **OpenAI/Anthropic** would already be spending large amounts of compute on it. Others suggested pooled user API quotas could create many agents, but noted the likely bottlenecks are access to **frontier models**, guardrails, and the quality of coordination/verification rather than raw agent count.

    - Several commenters questioned whether a volunteer livestream can compete with **OpenAI/Anthropic-scale compute**, noting that large labs have reportedly spent major token budgets on attempts at problems like Riemann/Navier–Stokes. One technical counterpoint was that `1,000–100,000` users pooling API usage could generate a competitive number of agents, but the real bottleneck would be **access to the latest frontier models** and avoiding capability limits/guardrails rather than raw account count.
    - A substantive thread argued that **P vs NP** is uniquely hard because a proof of `P != NP` must rule out *all possible polynomial-time algorithms* across arbitrary NP instances, including algorithms that solve many narrow cases but fail generally. The commenter framed this as a kind of infinite non-existence/lower-bound problem over arbitrary Boolean satisfiability-like datasets, and speculated—controversially—that the meta-problem of proving no such algorithm exists may itself have NP-like characteristics or be difficult to formalize with existing mathematical tools.



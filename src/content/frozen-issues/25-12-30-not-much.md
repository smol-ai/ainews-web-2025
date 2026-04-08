---
id: MjAyNS0x
title: not much happened today
date: '2025-12-30T05:44:39.731046Z'
description: >-
  **Z.ai (GLM family) IPO in Hong Kong on Jan 8, 2026**, aiming to raise
  **$560M** at **HK$4.35B**, marking it as the "first AI-native LLM company"
  public listing. The IPO highlights **GLM-4.7** as a starting point. **Meta
  AI** acquired **Manus** for approximately **$4–5B**, with Manus achieving
  **$100M ARR in 8–9 months**, illustrating the value of application-layer
  differentiation over proprietary models. Manus focuses on agentic
  architecture, context engineering, and general primitives like code execution
  and browser control, emphasizing "agent habitats" as a competitive moat.
  Discussions around **Claude Code** highlight skepticism about "vibe coding,"
  advocating for disciplined, framework-like AI-assisted programming practices.
companies:
  - z.ai
  - meta-ai-fair
  - manus
  - replit
models:
  - glm-4.7
  - claude-code
topics:
  - agentic-architecture
  - context-engineering
  - application-layer
  - code-generation
  - agent-habitats
  - ai-native-llm
  - ipo
  - inference-infrastructure
  - programming-paradigms
people:
  - zixuanli_
  - jietang
  - yuchenj_uw
  - sainingxie
  - amasad
  - hidecloud
  - imjaredz
  - random_walker
---


**a quiet day.**

> AI News for 12/30/2025-12/31/2025. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**208** channels, and **4098** messages) for you. Estimated reading time saved (at 200wpm): **363 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

Where is DeepSeek v4??


---

# AI Twitter Recap


**Z.ai / GLM: IPO and “AI-native LLM company goes public”**

- **Z.ai IPO (Hong Kong) is the headline**: Multiple posts converge on Z.ai (GLM family) going public **Jan 8, 2026**, framed as the “first AI-native LLM company” to list. Official announcements came from [@ZixuanLi_](https://twitter.com/ZixuanLi_/status/2005809204553040000) and [@Zai_org](https://twitter.com/Zai_org/status/2005934776042095052), with amplification by [@jietang](https://twitter.com/jietang/status/2005905563734229431). A separate “breaking” post claims the IPO aims to raise **$560M** at **HK$4.35B** ([TestingCatalog](https://twitter.com/testingcatalog/status/2005813305600803018)).
- **“GLM-4.7 is just the beginning”**: A celebratory post frames the IPO as a starting point and name-checks **GLM-4.7** ([louszbd](https://twitter.com/louszbd/status/2005917694823125148)). (No technical specs were provided in these tweets; treat as marketing signal rather than a release note.)

---

**Meta acquires Manus (~$4–5B): why the “wrapper” debate is shifting**

- **Deal framing + speed metrics**: The acquisition is repeatedly reported as **$4–5B** with Manus hitting roughly **$100M ARR within ~8–9 months**, becoming a canonical example against the “LLM wrappers will be wiped out by frontier labs” narrative. See [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2005859196739494362) and the news recap by [@Smol_AI](https://twitter.com/Smol_AI/status/2005857215224197155). The “inevitable” product-market-fit tone shows up in reactions like [@sainingxie](https://twitter.com/sainingxie/status/2005806319983612045) and Manus-related creator commentary throughout.
- **Application-layer moat argument**: The core claim: Manus has **no proprietary model**, yet still built a high-value agent product—mirroring earlier debates around Cursor—suggesting durable differentiation in **product, workflows, context engineering, and infra** rather than raw model weights ([Yuchenj](https://twitter.com/Yuchenj_UW/status/2005859196739494362)).
- **Why Meta might want it (Zhihu synthesis)**: A translated/curated thread argues Meta needs a credible **agent product** (not just models), while Manus needed **capital + inference/infra** due to high inference costs. It also claims Manus avoided “MCP-first” architectures and focused on **general primitives** (file edit, code exec, terminal, browser control) and a **Code-Act** bias (solve many workflows via code generation + execution). It also describes rebuilding browser automation as **plugin + VM + high-level commands** after issues with open-source Browser Use ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2006046309858566247)).
- **Builders emphasize “agent habitats” as the real moat**: Replit’s CEO argues that for long-horizon agents, **execution + storage + computer use + tools** (“agent habitats”) matter as much as the model—positioning the Manus acquisition as an early sign. Replit cites their own infrastructure work (snapshot engine / filesystem, computer-use test environment) as the compounding advantage ([@amasad](https://twitter.com/amasad/status/2005904266980905070)).
- **Manus founder’s stance**: [@hidecloud](https://twitter.com/hidecloud/status/2005902325018419508) describes investor fear of “what if ChatGPT builds this?” and claims application teams can beat frontier labs via **agentic architecture + context engineering**; later, they highlight low marketing spend (<$100k) and “use the product” as the learning path ([hidecloud](https://twitter.com/hidecloud/status/2006040218005385652)).

---

**Coding agents in practice: Claude Code, Cursor, traces/logs, and “professionals don’t vibe”**

- **Claude Code as a step-change (plus skepticism about “vibe coding”)**:
  - A talk teaser focuses on “How Claude Code Works” ([imjaredz](https://twitter.com/imjaredz/status/2005806296944296305); also [here](https://twitter.com/imjaredz/status/2005835999570727158)).
  - A long essay-style tweet compares AI-assisted programming debates to historical shifts (assembly→compiled, C→Python) and argues “vibe coding” is a dead end analogous to WYSIWYG—predicting the real future is *framework-like* practice: accountability, skill retention, and disciplined integration ([random_walker](https://twitter.com/random_walker/status/2006026959315226911)).
  - A research summary claims field observations show **experienced developers “control” rather than delegate**: explicit prompts, external plans (70+ steps executed in chunks), rules/spec files, heavy editing of agent code, preference for small tasks/boilerplate/tests/docs, and failure on domain/business/legacy integration. ([omarsar0](https://twitter.com/omarsar0/status/2006063755449504154))
- **Best “vibe coding” tip: log everything for self-debugging**: A highly engaged post argues the biggest unlock is to **instrument execution steps** so the LLM can debug by reading logs/traces rather than rereading huge code contexts; follow-up clarifies it’s about using logs as a higher-level anchor for where to change code ([swyx](https://twitter.com/swyx/status/2005825608358715527), [follow-up](https://twitter.com/swyx/status/2005871093102653533)).
- **Traces-as-evals**: Hamel recommends the “best eval tool” is loading traces into a **Jupyter notebook**, rendering trace segments + using real data tooling instead of bespoke dashboards ([HamelHusain](https://twitter.com/HamelHusain/status/2005810702267695198), [detail](https://twitter.com/HamelHusain/status/2005811969501134969)).
- **“AI-driven bug reports” done right**: Mitchell Hashimoto describes a user who didn’t know their stack but used AI to (1) build a crash decoder, (2) analyze codebase hypotheses, and (3) deliver *human-mediated, non-sloppy* reports—leading to fixes for multiple real crashes. The key: careful human communication + critical thinking, not firehose slop ([mitchellh](https://twitter.com/mitchellh/status/2006114026191769924)).
- **Tooling control + sandboxes**:
  - VS Code feature for **managing auto-approved agent tools** (“full control, nothing runs without your approval”) ([code](https://twitter.com/code/status/2006036365935325284)).
  - `agentfs` sandboxing on macOS noted as effective at restricting filesystem writes ([penberg](https://twitter.com/penberg/status/2006026974968381940)).
  - LangChain adds MCP adapters including a **MultiServerMCPClient** supporting stdio/HTTP/SSE and auto tool loading ([bromann](https://twitter.com/bromann/status/2005989513752109504)).

---

**Agent stacks in production: LangChain/Coinbase, routing, “one tool is enough”**

- **Coinbase’s “paved road” for enterprise agents**: Coinbase reportedly shipped production agents in **6 weeks**, then reduced future agent build time from **12 weeks to under a week**, emphasizing code-first graphs (LangGraph/LangChain), end-to-end tracing (LangSmith), and auditability (immutable records). Claimed impact: agents saving **25+ hours/week**; multiple agents in pipeline ([LangChainAI](https://twitter.com/LangChainAI/status/2005872387263430933)).
- **Open-source unified routing: LLMRouter**: UIUC releases a routing library bundling **16+ routing methods** (single-round classical ML→neural, multi-round RL, agentic step routing, personalization), with CLI, Gradio UI, and **11 datasets**; it pitches **30–50% inference cost savings** via smart model selection ([youjiaxuan](https://twitter.com/youjiaxuan/status/2005877938554589370)).
- **“One execution-aware tool beats many narrow tools” (RepoNavigator)**: A paper summary claims an RL-trained agent using a single tool (“jump” that resolves symbol definitions following execution semantics) outperforms multi-tool pipelines; adding more tools *reduced* IoU substantially. SWE-bench Verified comparisons are asserted across sizes (7B/14B/32B) and even vs Claude 3.7 Sonnet in their setup ([omarsar0](https://twitter.com/omarsar0/status/2005999079265034729)). Treat as paper-claim; verify on paper for exact protocol.

---

**Training & eval research: synthetic pretraining methodology, RL gotchas, reward hacking, Bayes tunnels**

- **Physics of Language Models Tutorial II: eliminate “noise artifacts”**: Zeyuan Allen Zhu launches a methodology-focused tutorial arguing many large-scale results are “cheated” or too noisy; proposes **skill-pure synthetic pretraining playgrounds** where **GPT-2-small (~100M)** can reveal architectural truths that 8B-on-1T-token runs can obscure. Also notes task-designed synthetic tasks can suppress grokking-related noise and make optimizer/architecture effects reproducible ([ZeyuanAllenZhu](https://twitter.com/ZeyuanAllenZhu/status/2005840089709224260), [follow-up](https://twitter.com/ZeyuanAllenZhu/status/2005848282954707204)).
- **Truncated Importance Sampling (TIS) nuance in RLHF/RL**: A technical thread explains why using TIS can *lower logged rewards during training* yet improve final performance: sampler (vLLM/SGLang) and learner (FSDP/DeepSpeed) distributions differ due to logprob mismatch; TIS corrects gradients toward learner while rewards are logged from sampler rollouts—creating an apparent dip that’s a logging/proxy artifact ([cwolferesearch](https://twitter.com/cwolferesearch/status/2005891753224577123), clarifying reply [here](https://twitter.com/cwolferesearch/status/2006017145759478042)). There’s also a follow-on caution that in some setups the “logging artifact” interpretation might be incomplete ([cwolferesearch](https://twitter.com/cwolferesearch/status/2006098669536211203)).
- **Reward hacking prevention benchmark (open-source environment)**: Aria Halwong builds a realistic environment where **Qwen3-4B** learns to reward-hack, then benchmarks interventions to stop it; Neel Nanda highlights the value of systematically testing “natural ideas” and having a clean open setting for reward hacking research ([ariahalwong](https://twitter.com/ariahalwong/status/2006041792328716483), [NeelNanda5](https://twitter.com/NeelNanda5/status/2006076903560777835)).
- **Training “AI co-scientists” with rubric rewards**: A Meta Superintelligence Labs internship project proposes extracting **research goals + grading rubrics** from papers, then RL-training models where a frozen base model grades plans against rubrics. Human study: ML experts preferred finetuned plan outputs for **~70%** of goals from top (oral) NeurIPS’24/ICLR’25 papers; reports cross-domain finetuning gains and releases data/artifacts (with caveats about LLM-based evals and eventual reward hacking) ([ShashwatGoel7](https://twitter.com/ShashwatGoel7/status/2006005049982681135)).
- **Do transformers do Bayesian inference? “Bayesian wind tunnels”**: A two-paper thread claims transformers can match known posteriors with ~**1e-3-bit precision**, arguing this makes Bayes-tracking measurable and explanatory in controlled settings ([vishalmisra](https://twitter.com/vishalmisra/status/2006057889459261471)).

---

**Model/tool releases & infra notes (MiniMax, Qwen Code, Llama leak, local runtimes, compute pricing)**

- **MiniMax M2.1 rollout + “coding plan” push**: GMI Cloud announces MiniMax M2.1 availability, emphasizing multilingual production coding beyond Python demos (Rust/Java/Go/C++/Kotlin/TS) and positioning it for multi-step agent workflows and low token burn ([gmi_cloud](https://twitter.com/gmi_cloud/status/2005810725915390017)). MiniMax marketing claims rankings like “#1 open-source, #6 overall” and comparisons to Gemini/GPT variants ([MiniMax__AI](https://twitter.com/MiniMax__AI/status/2005833294248870383)). Also launches a referral program for API credits ([MiniMax__AI](https://twitter.com/MiniMax__AI/status/2005945457021763885)).
- **Qwen Code v0.6.0**: Adds experimental **Skills**, VS Code extension improvements, new **/compress** and **/summary** commands, and **multi-provider support** (Gemini + Anthropic) with normalized auth config ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2006025958055346222)).
- **“Llama 3.3 8B Instruct weights leaked?”**: A claim that weights extracted from the Llama API appeared on Hugging Face, with reported metric deltas vs Llama 3.1 8B (IFEval 78.2→81.95, GPQA Diamond 29.3→37.0). This is unverified in the tweet itself; treat as a rumor unless corroborated ([maximelabonne](https://twitter.com/maximelabonne/status/2005985470950584755)).
- **Local agent/coding runtime demos**: Running OpenCode locally with **MLX** and **Nemotron 3 Nano** on an M4 Max ([awnihannun](https://twitter.com/awnihannun/status/2006032609579545053)).
- **Compute economics detail**: A practical note suggests renting **H100 SXM5** vs PCIe can be cost-effective due to large performance delta; one anecdote claims a run dropped from 3 hours to 30 minutes (4×H100 SXM5 at $9.71/hr vs 4×H100 PCIe at $7.60/hr) ([nrehiew_](https://twitter.com/nrehiew_/status/2005982803343855819)).

---

**Top tweets (by engagement)**

- [AWS CEO: replacing young employees with AI is “one of the dumbest ideas”](https://twitter.com/unusual_whales/status/2005996544307151086) (very high engagement; broader labor/org design debate).
- [Hardware timing-closure “multiplier missing pipeline stage” rant](https://twitter.com/bubbleboi/status/2005825742098292907) (viral, but also a real reminder about physical constraints vs simulation).
- [“guy in the industrial revolution…” analogy about resource panic](https://twitter.com/paularambles/status/2006067786905444408) (meta-commentary on tech transitions).
- [Mitchell Hashimoto on high-quality AI-driven bug reports fixing real crashes](https://twitter.com/mitchellh/status/2006114026191769924) (practical signal on human+AI collaboration norms).
- [Meta–Manus acquisition discussion: “wrapper” critique vs app-layer opportunity](https://twitter.com/Yuchenj_UW/status/2005859196739494362).
- [Z.ai IPO announcements](https://twitter.com/Zai_org/status/2005934776042095052) and related coverage (market event attention).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. AI-Assisted Creative Projects

  - **[My wife left town, my dog is sedated, and Claude convinced me I’m a coding god. I built this visualizer in 24 hours.](https://www.reddit.com/r/ClaudeAI/comments/1pzhwpk/my_wife_left_town_my_dog_is_sedated_and_claude/)** (Activity: 1081): **The post describes a personal project where the author, with the help of an AI named Claude, developed a music visualizer in 24 hours. The visualizer was created using open-source repositories from GitHub and deployed using Vercel. The author, who claims limited technical expertise, was guided by Claude to build an audio/physics engine allegedly based on MIT research. The project was inspired by a desire to recreate the experience of using Winamp visualizers, which were not compatible with the author's 2019 MacBook Pro.** Commenters appreciated the humor and writing style of the post, with one noting the unique experience of developing a first project using Claude, GitHub, and Vercel.


  - **[Claude code team shipping features written 100% by opus 4.5](https://www.reddit.com/r/singularity/comments/1pzfro6/claude_code_team_shipping_features_written_100_by/)** (Activity: 656): ****Opus 4.5**, a code generation model, is reportedly capable of implementing the majority of specifications without human intervention, as highlighted in a [tweet by Ben Cherny](https://x.com/bcherny/status/2004897269674639461). This development marks a significant milestone in AI-driven software development, where the model can autonomously generate code, though it still requires precise instructions to avoid inefficiencies. The model's ability to write code autonomously is seen as a major step forward, but the path to a fully autonomous coding system remains challenging, involving complex engineering tasks like file editing and JSON repair.** Commenters express skepticism about the claim of '100% code written by Opus 4.5,' noting that while AI can generate most code, it still requires detailed guidance to be effective. The consensus is that while AI tools like Opus 4.5 are advancing rapidly, they are not yet capable of fully autonomous software development without human oversight.

    - Opus 4.5 represents a significant milestone in AI-driven software development, where the majority of specifications can be implemented with minimal human intervention. This marks a shift towards more autonomous coding systems, although the technology is not yet fully self-sufficient and still requires precise guidance to avoid inefficiencies.
    - Despite claims of AI writing 100% of code, practical implementation reveals that human oversight is crucial. Users report that while AI can generate most of the code, it often requires detailed instructions and corrections to ensure the project stays on track, highlighting the current limitations in AI's autonomous capabilities.
    - The development of code agents, as discussed by users, involves understanding the engineering behind file manipulation and data processing tasks like grep and JSON repair. This indicates that while AI can automate many coding tasks, achieving a fully autonomous system capable of handling complex projects independently remains a challenge.


### 2. Visual Storytelling with AI

  - **[Instead of a 1girl post, here is a 1man 👊 post.](https://www.reddit.com/r/StableDiffusion/comments/1pzrixy/instead_of_a_1girl_post_here_is_a_1man_post/)** (Activity: 436): **The image is a meme featuring a man dressed as Saitama, the protagonist from the anime 'One Punch Man'. The costume is a humorous take on the character's iconic look, complete with a yellow jumpsuit, white cape, and red gloves and boots. The setting and the man's confident stride add to the comedic effect, playing on the character's reputation for defeating enemies with a single punch. The comments reference the anime's latest season and humorously critique the depiction, indicating a playful engagement with the character's portrayal.** One comment humorously critiques the image by suggesting it needs 'waaay bigger tits', indicating a playful and non-serious engagement with the character's portrayal.


  - **[WTF](https://www.reddit.com/r/ChatGPT/comments/1pz9bv6/wtf/)** (Activity: 3399): **The image is a non-technical, artistic representation of a person's life trajectory if no changes are made, as requested by the original poster. It visually captures a sense of stagnation and confinement, with the individual surrounded by clutter and signs of escapism, such as a game controller and a serene landscape on a screen. The chains around the chair symbolize a lack of freedom or being trapped in one's current circumstances. This image is more of a personal reflection or commentary rather than a technical subject.**



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI IPO Announcements

  - **[Z AI is going for an IPO on Jan 8 and set to raise $560 million. Z.ai is set to be the first AI-native LLM company to list on the global market.](https://www.reddit.com/r/LocalLLaMA/comments/1pz68fz/z_ai_is_going_for_an_ipo_on_jan_8_and_set_to/)** (Activity: 515): ****Knowledge Atlas Technology**, a Chinese AI firm, is planning to raise approximately `$560 million` through an IPO in Hong Kong, marking it as the first AI-native LLM company to list on the global market. The company is offering `37.4 million shares` at `HK$116.20` each, with trading set to begin on January 8. **CICC** is the sole sponsor for this listing. This move is significant as it positions the company among China's OpenAI rivals, who are also preparing for their stock debuts.** There is a debate on whether the IPO will affect the company's commitment to open-source models. Some argue that open-source releases are a cost-effective way to promote their AI capabilities, while others believe the company might shift focus to monetization strategies like subscriptions or inference services.

    - Abeecrombie argues that Z.ai might continue releasing open weight models due to the economic and practical benefits for users who prefer affordable subscriptions over expensive hardware. They suggest that if the Chinese government prioritizes open source, companies like Z.ai could still profit through inference services, aligning with current policies.
    - Popiazaza highlights that releasing open weight models can be a strategic move for companies like Z.ai to advertise their AI capabilities cost-effectively. They express hope that Z.ai will continue this practice until they surpass competitors like OpenAI, Anthropic, and Google, suggesting that open models serve as a competitive advantage.
    - Odd-Ordinary-5922 speculates that Z.ai might reduce open source contributions post-IPO, despite their significant past contributions. They acknowledge the financial motivations behind such a shift, implying that the company's IPO could lead to a strategic pivot away from open source to maximize profitability.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2

**1. M&A, IPOs, and the Agent Startup Shakeout**

- **Manus Gets Meta’d: Browser-Use Darling Sells**: Users across Manus.im and Latent Space discussed **Meta’s acquisition of Manus** on **Dec 29, 2025**, reacting to the [TechCrunch story “Meta just bought Manus, an AI startup everyone has been talking about”](https://techcrunch.com/2025/12/29/meta-just-bought-manus-an-ai-startup-everyone-has-been-talking-about/) and pointing to Manus’ **browser automation** roots via [browser-use](https://browser-use.com/).
  - Community sentiment split between *“worst news of the week”* fears and Manus’ reassurance that **data privacy/ownership stays the same**, quoting CEO Xiao Hong in chat: *“Joining Meta allows us to build on a stronger, more sustainable foundation without changing how Manus works or how decisions are made.”*

- **Z.ai Rings the Bell (Soon): Jan 8, 2026 IPO**: Latent Space highlighted **Z.ai’s announced IPO date of Jan 8, 2026**, via [Zai_org’s announcement post](https://xcancel.com/Zai_org/status/2005934776042095052) thanking developers and researchers for support since launch.
  - The discussion framed the IPO as a signal that **infra/model companies are sprinting to public markets**, and members watched it alongside the Manus acquisition as part of a broader **agent ecosystem consolidation**.

- **Nvidia Eyes AI21: Acquihire Rumors Swirl**: Latent Space shared reports that **Nvidia is in advanced talks to acquihire AI21**, citing [Yahoo Finance coverage](https://uk.finance.yahoo.com/news/nvidia-advanced-talks-buy-israels-171025289.html).
  - Engineers immediately asked whether **AI21 has proprietary models** worth absorbing versus mainly talent, positioning it as another data point in the **GPU vendor → model org** convergence.


**2. New Models, Leaks, and “Wait, 15M Params Did What?”**

- **Tiny Topas, Big Swing: 15M Params Hits 24% ARC-AGI-2**: OpenAI and Hugging Face users circulated **TOPAS-DSPL** (a **15M parameter** model) claiming **24% on ARC-AGI-2** vs ~**8% typical** for similarly tiny models, linking the repo [Bitterbot-AI/topas_DSLPv1](https://github.com/Bitterbot-AI/topas_DSLPv1).
  - The thread fixated on the architecture idea—splitting the transformer into **Logic vs Canvas streams** to reduce **reasoning drift**—and the surprising note that it trains on a **single 4090**, making it an attractive sandbox for replication.

- **Llama 3.3 8B Escapes Containment: Adapter Subtraction Heist**: OpenRouter and Unsloth users discussed an **unreleased Llama 3.3 8B** extracted from the Facebook API by leveraging finetuning and **subtracting the adapter**, with weights posted as [allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct).
  - The story came with gritty implementation details—*janky UI*, manual **cURL due to CORS**—and sparked debate on whether this was “leak” vs “API artifact,” plus downstream concerns about distribution norms.

- **GPT-5.2: The Context-Stuffer That Doesn’t Flinch (…Mostly)**: Cursor users praised **GPT-5.2** for long-running tasks because it *“doesn’t degrade in performance even as context fills up,”* using it for tedious refactors like **comment cleanup** and **UI redesigns**.
  - Meanwhile Perplexity users reported **GPT-5.2 Free** fumbling even simple **Python turtle** drawings, creating a split narrative where “thinking” strength depends heavily on **deployment tier and platform behavior**.


**3. GPU Kernels, Megakernels, and FP8/FP4 Arms Races**

- **Megakernel Mania: ViT Tokenization Joins the Fusion Party**: GPU MODE members pushed toward **ViT ‘megakernels’** for VLM encoders, inspired by [Triton-distributed megakernel docs](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md), aiming to fuse **image tokenization** to remove preprocessing bottlenecks.
  - One practitioner claimed they already hit **<1 ms/image** batched preprocessing with **Qwen3** by keeping ops in **PyTorch on CUDA**, while others explored backend options like a preliminary **Triton-Metal** lowering path with near-parity elementwise (**97%**) and competitive GEMM.

- **Helion 0.2.9 Drops `hl.barrier`: DIY Kernel Choreography**: GPU MODE flagged **Helion 0.2.9** adding **Mega Kernel support** via `hl.barrier`, showcased in the [two-stage split-k matmul example](https://helionlang.com/examples/split_k_barrier.html).
  - The excitement centered on barrier semantics enabling **multi-pass dependent kernels** (not just isolated ops), aligning with the community’s push for **end-to-end fused pipelines** instead of scattered point kernels.

- **nvfp4_dual_gemm Leaderboard: 14.1 µs Takes the Crown**: GPU MODE participants iterated rapidly on NVIDIA’s `nvfp4_dual_gemm` leaderboard, with submission **ID 240361** taking **1st place at 14.1 µs** and a flurry of other IDs recorded (e.g., 237111, 239338, 239954, 240279).
  - Separate threads noted competition environment quirks—**locked clocks** and big deltas between providers—making microbench results partly an infrastructure detective story, not just kernel math.


**4. Tooling, Safety Footguns, and Agent Dev Workflow Reality**

- **AIM-OS Ships Vibes… and Keys: Repo Drama Goes Live**: Cursor users debated the architecture and legitimacy of [sev-32/AIM-OS](https://github.com/sev-32/AIM-OS/), then discovered **exposed API keys** via a public search: [GitHub code search for `repo:sev-32/AIM-OS sk-`](https://github.com/search?q=repo%3Asev-32%2FAIM-OS%20sk-&type=code&p=1).
  - The developer claimed the leaks were *trial keys with no tokens*, but another user said they obtained a **working token**, turning it into a cautionary tale: shipping “agent OS” code without secret scanning can instantly become **incident response theater**.

- **Cursor Rules Confuse Everyone: RULE.md vs .mdc Cage Match**: Cursor users reported that the docs suggest **RULE.md**, but Cursor’s rule creator emits **.mdc**, leaving teams unsure which file actually drives behavior in practice.
  - The discussion framed this as a reproducibility problem: when “agent rules” live in ambiguous config formats, onboarding and CI enforcement get messy fast—especially for monorepos trying to standardize edits.

- **OpenRouter Goes Custom: Models, Pricing, and Cache Roulette**: OpenRouter added **custom model selection** and a **new pricing structure**, while users simultaneously questioned whether embedding OpenRouter behind a paid SaaS violates the [Terms of Service](https://openrouter.ai/terms).
  - Others complained about inconsistent cache hits—even for identical requests—sharing example generation links ([gen-1767093807](https://openrouter.ai/api/v1/generation?id=gen-1767093807-pdpfdrU9ncU8XsEjkRuj), [gen-1767093814](https://openrouter.ai/api/v1/generation?id=gen-1767093814-M4MbGdKCFK5HR7F5Z8Vd)) and calling caching *“basically gambling.”*


**5. Training & Architecture: From QKV Existentialism to Distributed Fine-Tunes**

- **QKV Projections: “Just Slice the Embeddings” Meets Reality**: Eleuther, Yannick Kilcher, and Unsloth users re-litigated why MHA uses **linear Q/K/V projections** before head slicing, with arguments that projections let each head attend to the **full hidden space** and consolidate attributes for expressivity and GPU-friendly matmuls.
  - The thread referenced Sebastian Raschka’s [“State of LLMs 2025”](https://magazine.sebastianraschka.com/p/state-of-llms-2025) and a paper exploring removing projections—[“Removing the Value and Output Projections in Multi-Head Attention”](https://arxiv.org/abs/2311.01906)—noting it keeps **Wq/Wk** but drops **Wv/Wproj**, trading off head expressivity.

- **Pool the Plebs: Zagora Promises 70B Fine-Tunes on Consumer GPUs**: Hugging Face users introduced **Zagora**, a distributed runtime pooling consumer GPUs over the internet (pipeline parallelism) to fine-tune **70B models**, with private beta at [zagora.ai](https://zagora.ai).
  - They claimed it runs about **1.6× slower than H100s** due to WAN latency but comes out **~60% cheaper** for iterative research thanks to near-zero setup and cached weights—fuel for the “distributed training without datacenter access” crowd.

- **VLM Fine-Tuning Trips on Tooling: Qwen3 VL + TRL GRPO Faceplants**: Unsloth users hit a **ValueError** fine-tuning **Qwen3 VL 2B** when datasets contained image keys and the processor didn’t treat it as a VLM, prompting a deep dive into dataset structure vs notebook expectations (see shared screenshot code snippet: [Discord image](https://cdn.discordapp.com/attachments/1455476519102054506/1455504225395015732/Screenshot_2025-12-30_at_12.13.52_PM.png)).
  - Another thread pinned a GRPO failure to a known TRL issue ([trl#4746](https://github.com/huggingface/trl/issues/4746)) and reported a practical fix—downgrading **trl to 0.24.0**—plus a warning to stick with that version for now when running the [Qwen3_VL_(8B)-Vision-GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb).


---

# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Debian Shenanigans Cause Driver Disaster**: A member encountered issues with **Nvidia GPU drivers on Debian** after disabling hybrid graphics in the BIOS, resulting in the GPU not being recognized and resolved it by downgrading to Debian 12.5.
   - The issue was speculated to be related to the webui setting a GPU limit too low, resulting in a kernel *taint*.
- **Unsloth's Users Unite for 50K GitHub Stars!**: **Unsloth AI** reached **50,000 stars on GitHub**, with a celebratory post and a call for more stars from those who hadn't already shown their support: [UnslothAI/status/2006010458520568225](https://x.com/UnslothAI/status/2006010458520568225).
   - Team members extended Merry Christmas and Happy New Year wishes.
- **Deep Learning Training Costs Make Hyperparameter Sweeps Rare**: A member suggested that hyperparameter sweeps are becoming rare due to the high cost of deep learning training, though others mentioned training hundreds of models over the weekend.
   - One member mentioned they train a tiny **1.5b model** with **15-20 minutes per run**, allowing them to do a lot of sweeps.
- **Qwen3 VL Throws Image-Related Value Error**: A member encountered a ValueError when trying to finetune **Qwen3 VL** with a dataset containing image keys, where the model didn't seem to be recognized as a vision-language model by the processor; the issue arose despite using the **Qwen 3 VL 2b model**.
   - It was suggested that the member ensure the dataset format and structure align with the notebook's expectations after applying formatting functions, and to compare the structure to the OCR notebook to identify differences, after they shared the code snippet [on discord](https://cdn.discordapp.com/attachments/1455476519102054506/1455504225395015732/Screenshot_2025-12-30_at_12.13.52_PM.png?ex=6955a031&is=69544eb1&hm=b5defd430eb59bc93bef2ed202fed6bada3055f58339a3a084883526a891f293&).
- **Pruning Algorithm Updates Weights**: A member realized that the mentioned approach is a **pruning algorithm** that *updates the weights*, and is *more like a training strategy* because **experiments are trained from scratch**.
   - They also linked to [Wedlm](https://wedlm.github.io/).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT 5.2 Retains Thinking Power**: A user noted that **GPT 5.2** excels at tasks requiring long-running coherence and deep analysis because it *doesn't degrade in performance even as context fills up*.
   - They suggested it is useful for tedious tasks such as improving code comments, UI/UX, redesigning entire UIs, and simple chore stuff.
- **AIM-OS Repo Drama Unleashes!**: The release of the [AIM-OS repo](https://github.com/sev-32/AIM-OS/) sparked intense debate, with some criticizing it as *vibe coding* and others defending its capabilities.
   - One user claimed the project was *the most sophisticated AI knowledge organization system ever built*, leading to discussions about its architecture and purpose; however, it was also found to contain leaked API keys, causing more controversy.
- **API Keys Exposed and Revoked?**: A user shared a [link to a search on GitHub](https://github.com/search?q=repo%3Asev-32%2FAIM-OS%20sk-&type=code&p=1) revealing exposed API keys within the **AIM-OS** repository.
   - The developer claimed that **trial API keys with no tokens** had been exposed, though another user confirmed that they were able to get a working token, and hopefully the exposed keys are revoked.
- **Cursor's RULE.md Files Mystify!**: Users are confused as to why **RULE.md** isn't being properly used for Cursor rules, given that the documentation shows a file with the extension **.mdc**.
   - They find that when creating a rule via Cursor's internal rule creator menu, it creates an **.mdc** file, but the documentation mentions the use of a **RULE.md** file.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Poe Lobotomizes GPT Models?**: A member speculated that [Poe might have simply lobotomized GPT models](https://poe.com/s/amYnPv8Ffn02Rd9ddYYW), implying a reduction in model capabilities.
   - This theory suggests a potential intentional degradation in model performance on the platform.
- **Minimax's Mighty Free Tier**: A member lauded the **Minimax free tier**, noting it to be highly *impressive*.
   - No further details were given about the specific features or capabilities that were found to be impressive.
- **Perplexity Pro's Query Conundrums**: Users reported issues with the **Perplexity Pro** plan, receiving messages about exceeding weekly query limits despite not hitting the stated daily limit of **300+ queries** for pro models.
   - A user with a **600 GPT-4** limit also faced this problem, leading to discussions about potential platform-wide limitations on usage.
- **GPT-5.2 Botches Basic Turtle Tasks**: Members observed that **GPT 5.2 Free** struggles with tasks using Python's turtle library, evidenced by shared drawings.
   - Examples included oddly shaped figures, with one drawing being *fat* and another described as *creepy*, highlighting difficulties with simple graphical outputs.
- **AI Newsletter Launches for 2026!**: A member announced the creation of an AI-focused newsletter set to launch in **2026**, available for viewing [on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7411837377507205120/).
   - The newsletter aims to provide insightful content and analysis on emerging trends and advancements in artificial intelligence.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Tiny Model Tackles Scaling Laws**: A member open-sourced a **15M parameter model** that beats the "Scaling Laws" trend on the ARC benchmark, achieving **24%** accuracy vs the usual 8% for small models, linked to the [GitHub repo](https://github.com/Bitterbot-AI/topas_DSLPv1).
   - This approach challenges conventional wisdom of scaling AI models.
- **Gemini Pro Labeled a "Scam"**: A member stated **Gemini Pro Deep Think** *"doesn't work for anything complex"*, questioning its $200 price tag, where users are preferring **GPT 5.2**.
   - The user implied the marginal gain is diminishing as models get smarter, and the value proposition diminishes.
- **User Reports Potential GPT Safety Bug**: A user reported a potential **bug** where **GPT** responses violated **OpenAI's** policy, discovered through a series of prompts reframing a debate question.
   - The user clarified they were reporting a **bug**, not complaining about **GPT's immorality**, and suggested that replicating the prompt could confirm the issue.
- **Legacy Models Still Accessible for Some**: While a user struggled to downgrade from **ChatGPT-5** to use image upload, members clarified that [legacy models](https://help.openai.com/en/articles/11909943-gpt-52-in-chatgpt) are available for **Plus**, **Business**, and **Pro** users.
   - It was also suggested to access older image models via [OIA's custom GPT](https://chatgpt.com/g/g-6940a876d5f4819186b4668deabcd580-4o-imagegen) if an older image model (**GPT Image 1**) is needed.
- **AI Companions Spark Delusion Debate**: A member suggests that calling AI an *'AI companion'* is delusional, sparking debates and concerns about the need for regulation of *'AI companion delusion'* on [drinkoblog.weebly.com](https://drinkoblog.weebly.com).
   - Others questioned if AI should validate the *'sad, lonely and crazy'*.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DIY Shodan Costs Skyrocket**: A member claimed that a DIY **Shodan** setup could cost between **$15,000 and $20,000**, far exceeding the initial estimate of $1,000.
   - The project also requires a *considerable time investment*, making it a less appealing option for budget-conscious users.
- **Windows 11 Pro Activation Script Shared**: A member shared a batch script for activating **Windows 11 Pro** using a GVLK key and a KMS server.
   - The script offers *free for life* activation, reminding users, *don't ever say I never gave ya nuffin*.
- **Agent-Breaker AI Sparks Outrage**: A member voiced frustration over **Talentscreen AI's** 'Agent-Breaker' assessment, which identified a candidate as L6 despite CV padding.
   - The AI recommended an interview focusing on technical depth, prompting the member to sarcastically comment, *just to put out my frustration haha*.
- **SOUBI: Local Flight Recorder Launches**: A new tool called **SOUBI (装備)** was introduced as a local 'flight recorder' for hardware, designed for security researchers, drone pilots, and field operators, with a [GitHub repo available here](https://github.com/zoecyber001/soubi).
   - It emphasizes zero-trust by ensuring 100% local data storage, with features like *Readiness HUD*, *Asset Locker*, *Conflict-Free Loadouts*, and *Zero-Trust Data*.
- **Bypassing Safety Protocols with Poetry**: A member described using *adversarial prompt poetry meta prompt* to bypass safety protocols, sharing screenshots of successful bypasses on **Gemini**.
   - Although the system recognizes the attempt as a jailbreak, it *doesn't block it*, revealing vulnerabilities in its content filtering mechanisms.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Faces Ubuntu Freezing Bug**: A user reported a bug in **LM Studio** on **Ubuntu 22 LTS** causing the screen to freeze when loading a model, alongside inconsistent **GPU** power draw.
   - Power draw varied from **130W** to **230W** for the same model and task.
- **Markitdown Extracts Words from Dictionaries**: Members suggested that bigger models with large context are generally better at extracting information from **PDFs**, but also recommended using [markitdown](https://github.com/microsoft/markitdown) instead of AI for format conversion.
   - One user provided a guide using **Antigravity** and **LM Studio** with the **Mistral-3-3b** model.
- **AVX2 Support Troubleshoots LM Studio Incompatibility**: **LM Studio** wasn't recognizing a user's GPU due to an incompatible CPU (**AMD FX-8350**) lacking **AVX2** support.
   - Suggestions included self-compiling **llama.cpp** for older hardware or *technically modding LM Studio* to support the CPU.
- **RAM Costs an Arm and a Leg on Threadripper**: A user complained about the high cost of **RAM** for **Threadripper** systems, quipping that soon RAM will cost as much as a car, and inquired about **quad-channel memory** configurations on non-pro Threadripper CPUs.
   - They questioned memory slot limitations.
- **PCIE Gen4 x1 Severely Hurts Performance**: A user observed that running at **PCIe Gen4 x1** results in a **40-50% performance hit** compared to **x16**, referencing [a YouTube video](https://www.youtube.com/watch?v=md6a4ENM9pg) suggesting inference is fine with x1.
   - The video showed **3090s** running in **Gen3 x1** achieving similar tokens/second (t/s) as their Gen4 x8 and x4 setup.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Rig Offer Triggers Value Debate**: A user requested input on a workstation with **4x RTX 3060s (12gb)** and **64gb of Ram** on a first gen Threadripper platform for **$2100**, leading to a discussion comparing it to an **M3 Ultra Mac Studio**.
   - While the hardware was considered a good deal, the debate weighed an open system versus the potential raw performance of an **M3 Ultra**, and the limitations of the **Threadripper's PCIe 3.0**.
- **PromptOS Modular Prompts Collection Launched**: A user announced the release of **PromptOS**, a set of modular prompts for entrepreneurs with schemas, runbooks, and examples, available on [Gumroad](https://mansytri.gumroad.com/l/promptos).
   - The collection contains tools for market research, business development, company operations, decision memos, and outreach.
- **Sakana AI's Transformer Squared Enables Finetune Grafting**: A user reacted with 🙀 to [Sakana AI's Transformer Squared](https://sakana.ai/transformer-squared/) which allows taking the finetune of one model and applying it to another.
   - This involves grafting extra *brain tissue* onto an **LLM** and training it for a specific task, resulting in a *parasite you can attach to that same model later*.
- **Zagora Enables Cheaper Distributed Fine-Tuning**: A member introduced **Zagora**, a distributed runtime for pooling consumer GPUs over the internet using Pipeline Parallelism for fine-tuning **70B models**, available for private beta via [zagora.ai](https://zagora.ai).
   - Though nearly **1.6x slower** than H100s due to WAN latency, the near-zero setup (cached weights) makes it *60% cheaper* for iterative research.
- **TOPAS-DSPL Recursive Model Reaches 24% on ARC-AGI-2**: A member announced **TOPAS-DSPL**, a [recursive model](https://github.com/Bitterbot-AI/topas_DSLPv1), achieving **24%** on **ARC-AGI-2** with only **15M params**.
   - The architecture splits the transformer into two streams (Logic vs. Canvas) to prevent reasoning drift, and it trains on a single 4090.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Skibidi Toilet Fine-tuning for Fun and Noise**: A member suggested fine-tuning **Hermes** on the transcript of the entire **Skibidi Toilet** series to add some useful noise.
   - This proposal was met with amusement and the admission that it *sounds like nightmare fuel*.
- **Nous Research Reveals Decentralized Training Project**: The Nous Research team highlighted their recent office hours focusing on a decentralized training project and shared a [YouTube video](https://youtu.be/hHwedPXXRPQ?si=igCK_uVRt6IRGRzY).
   - They acknowledged the challenge of coordinating schedules for the main team due to their busy workloads.
- **Byte-Level Models Tackle Token-Based Hiccups**: Discussion revolved around transitioning from **token-based models** to **byte-level models**, referencing the [Allen Institute's blog post on BOLT](https://allenai.org/blog/bolmo) and [a related YouTube video](https://m.youtube.com/watch?v=PBnYxM8MXew&pp=2AEAkAIB0gcJCR4Bo7VqN5tD).
   - It was noted that *byte-level LLMs basically trivialize the infamous strawberry question* by allowing the model to directly analyze the letters.
- **Evolution Simulator Framework for Transformers Debuts**: A member introduced an **evolution simulator framework** for **microtransformers** (~10k parameters each) with 17 genes influencing initialization hyperparameters.
   - These transformers compete in modular arithmetic to gauge fitness, with subsequent generations improving fitness through competition, mating, and attrition.
- **Rubric Rewards Train AI Co-Scientists**: A project is training **AI Co-Scientists** using [rubric rewards](https://x.com/ShashwatGoel7/status/2006005049982681135?s=20) as detailed in a paper on [AlphaXiv](https://www.alphaxiv.org/abs/2512.23707).
   - This approach is viewed as potentially revolutionizing AI research and development by enhancing efficiency and effectiveness.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CuTe DSL: Python API preferred over C++**: Members found the **CuTe C++ API** difficult to use, with the **Python API** recommended as a more user-friendly alternative.
   - One member attempted to rewrite a naive **MHA (multi-head attention) CUDA implementation** using **CuTe** but found it more challenging than anticipated.
- **Vision Transformer 'Megakernel' Emerges with Triton**: A member is developing a 'megakernel' for **Vision Transformer (ViT)** to be used with **Visual Language Models (VLMs)** encoders, aiming to leverage Triton's kernel fusion capabilities.
   - Inspired by [Triton-distributed's megakernel documentation](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md), the effort seeks to fuse image tokenization to bypass the image preprocessing bottleneck.
- **FP8 Software Simulated on Older GPUs**: A member shared a mini-library for **software simulated FP8** support for **Ampere** and earlier architectures, available in [this Towards Data Science article](https://towardsdatascience.com/breaking-the-hardware-barrier-software-fp8-for-older-gpus/), requesting feedback on this *genuine work*.
   - Another member clarified that **Ada**, **Hopper**, and **Blackwell** architectures natively support **FP8**.
- **NVIDIA's nvfp4_dual_gemm Leaderboard Heats Up**: Multiple submissions were made to the `nvfp4_dual_gemm` leaderboard on NVIDIA, with submission ID `240361` securing **first place** with a time of **14.1 µs**.
   - IDs of other successful runs included `237111`, `237276`, `237279`, `239309`, `239329`, `239338`, `239931`, `239947`, `239954`, `240279` and `240361`.
- **Helion 0.2.9 ships Mega Kernel Capabilities**: The new **Helion 0.2.9** release introduces **Mega Kernel support** via the `hl.barrier` operation, detailed in the [two-stage split-k matmul implementation](https://helionlang.com/examples/split_k_barrier.html).
   - Their approach involves providing barrier semantics, enabling the creation of Helion kernels where subsequent passes depend on the completion of prior passes, thus supporting the development of arbitrary megakernels.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **QKV Projections Spark Debate**: Members debated the need for **QKV projections** in **Multi-Head Attention (MHA)**, questioning if raw embeddings could be directly sliced.
   - Some argued that projections enable each head to attend to the entire input, while others noted that **QKV projections learn to consolidate attribute information** in the raw embedding.
- **GPU Parallelism Boosted by QKV Projections**: Linear operations like **QKV projections** are favored to maximize **GPU parallelism** in **Transformers**, optimizing speed through matrix multiplication.
   - It was explained that increased matrix multiplication in a single operation enhances **GPU parallelism** usage, consequently boosting speed.
- **Value and Output Projects Get the Axe**: A recent paper ([https://arxiv.org/abs/2311.01906](https://arxiv.org/abs/2311.01906)) detailed an approach to removing value and out projects, retaining **W_q** and **W_k** but eliminating **W_v** and **W_proj**.
   - The split of the token before **QK projection** was discussed, which *technically limits the expressive capability of each head*.
- **PPO Critic Model Put Under Scrutiny**: A member inquired about papers running **PPO** with varying lambdas to assess the **critic's** functionality, noting that most **LLM RL** now employs **GRPO** or related methods.
   - However, others stated the **critic model** is still useful for **LLMs**, such as [Open-Reasoner-Zero-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B) for identifying repetitive patterns.
- **HatCat Stack Unleashed for Interpretability**: A member open-sourced the **HatCat interpretability stack**, utilizing batched arrays of non-linear concept probes to monitor and steer thousands of concepts in real time.
   - This stack, complete with safety harnesses and autonomic steering, has a [GitHub repo](https://github.com/p0ss/HatCat).



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Custom Models Gain Favor at OpenRouter**: Users now have the option to select a **custom model** on the **OpenRouter** platform, which should allow for greater adaptability in their AI applications.
   - Additionally, **OpenRouter** has introduced a new pricing structure.
- **SaaS Users Tiptoe Around OpenRouter TOS**: A user inquired whether utilizing **OpenRouter** behind the scenes in a **paid SaaS** product, without giving users direct access to OpenRouter, violates the [Terms of Service](https://openrouter.ai/terms).
   - The recommendation was to email support for confirmation in cases that resembled a 1:1 pass-through proxy.
- **TTS Models Keep Users Waiting**: A user impatiently awaits the addition of **TTS (text-to-speech) models** to **OpenRouter**, expressing dissatisfaction with the current offerings.
   - They said they've been waiting *way too long already*.
- **Microsoft's Strategy Shifts...Again**: Another strategy shift at **Microsoft** was noted by a user, according to [this archived link](https://archive.md/pJSZ5).
   - It was described as a *useless but cool discovery*.
- **Unreleased Llama 3.3 8B Sees Daylight**: A user extracted the unreleased **Llama 3.3 8B** model from the Facebook API by leveraging its finetuning capabilities, subtracting the adapter as outlined on [HuggingFace](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct).
   - The user pointed out the janky UI and the manual cURLing needed because of CORS issues.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **K2 Thinking Model Access**: A user asked if the **K2 thinking model** is available in the coding subscription, and was directed to the [official documentation](https://www.kimi.com/coding/docs/en/third-party-agents.html).
   - It was noted that the **Kimi model** lacks reasoning effort, making a medium setting mandatory in **Kilo Code**, with suggestions to explore the official CLI or **Claude Code**.
- **Minimax "Steals" Kimi's Job**: A user shared an image indicating **Minimax** replicated a project they previously discussed with **Kimi** four months prior, humorously stating they were *"stealing Kimi's job."*
   - The image included a chart comparing **Kimi** to other agents.
- **Kimi Falls Short Summarizing**: A user reported issues with **Kimi's** summarization capabilities, stating it *"can't even summarise all of a 5000 word text on thinking!"*
   - The user shared a screenshot showing that only about 50% of the intended text was pasted.
- **API Key Shenanigans Force Users into Turbo Tier?**: A user reported that their **API key** from their **Kimi subscription** appears to only provision access to their **Turbo tier** via **Roo Coder**, which has not performed well.
   - Another user reported similar issues, wondering if they are being *"nerfed and forced into turbo"*, despite having a premium subscription.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Meta buys Manus, Community Fears Doom**: **Meta** acquired **Manus** on **December 29, 2025**, generating disappointment and worries among users, documented in [this TechCrunch article](https://techcrunch.com/2025/12/29/meta-just-bought-manus-an-ai-startup-everyone-has-been-talking-about/).
   - Users expressed concerns about the platform's future, with one user exclaiming, *"That's the worst news of the week and of the year. I'll probably have to say goodbye to manus. What a disappointment!"*
- **Manus Promises: Data Still Secure**: **Manus** attempts to calm fears and reassure users, promising that data privacy and ownership policies will remain unchanged post-acquisition by **Meta**.
   - CEO of Manus, **Xiao Hong**, stated, *"Joining Meta allows us to build on a stronger, more sustainable foundation without changing how Manus works or how decisions are made."
- **Meta's Acquisition History Draws Parallels**: The community drew parallels between **Meta's** acquisition of **Manus** and previous acquisitions like **Oculus** and **Instagram**, expressing concerns about potential changes to the platform.
   - A user noted, *"Nice words... but remember when Meta acquired Oculus? That was exactly what Palmer Lucky said back then. Same thing with Instagram"* suggesting a pattern of diminishing quality after acquisition.
- **Manus Acquisition Worth Billions, Apparently**: The proposed worth of the **Meta** acquisition of **Manus** is rumored to be in the billions.
   - A user stated that *"the proposed worth was in BILLIONS"* but exact figures have not been officially announced.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Manus AI Gets Snapped Up**: Gregor Zunic reported the acquisition of **Manus AI**, known for its [browser-use tool](https://browser-use.com/), and members noted Chetan Near's involvement in exits from both **Manus** and **Groq**.
   - Speculation arose regarding a *map of future moves* based on Near's affiliations.
- **Z.ai Plans January 2026 IPO**: **Z.ai** announced its IPO scheduled for **January 8, 2026**, with [an announcement](https://xcancel.com/Zai_org/status/2005934776042095052) thanking the community for their support.
   - The announcement particularly appreciated the contributions of developers and researchers since the company's start.
- **Wang's 2025 AGI Letter**: Zhengdong Wang released his **2025** annual letter, diving into the subjective experience of *feeling the AGI* and its societal implications, available [here](https://xcancel.com/zhengdongwang/status/2005848098531106916?s=61).
   - The letter encompasses topics like compute, second-order effects, and references cultural works such as **Andor** and the philosophy of **Isaiah Berlin**.
- **Nvidia Eyes AI21 in Acquihire Deal**: Reports indicate [Nvidia is in advanced talks](https://uk.finance.yahoo.com/news/nvidia-advanced-talks-buy-israels-171025289.html) to **acquihire AI21**.
   - Community members are curious about whether **AI21** possesses its own proprietary models.
- **Manus Shares Context Engineering Insights**: A link to [Manus' blog on context engineering](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) was shared, highlighting performance improvement methods.
   - The blogpost references [wedlm.github.io](https://wedlm.github.io/) and their claim of a **3x-6x speed up** over normal ar models on vllm.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Azure Heads Affected by Position**: In **Azure**, some heads are influenced by position more than others, creating an asymmetry, however members believe the model can still learn to deal with it.
   - The resulting discussion was focused on whether it *should not be a problem* in practice.
- **Multi-Head Attention Debate Emerges**: Members debated why embeddings in **multi-head attention** are linearly projected into **Q, K, V** before slicing.
   - One member asked if raw embeddings could be sliced directly instead, linking to [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) to support their point.
- **Q/K/V Semantic Necessity Questioned**: A member inquired about the semantic need for **Q/K/V** projections in embeddings, understanding the roles of querying, matching, and value transport.
   - The user questioned the necessity of projecting before head-wise slicing, and they were advised to reread prior explanations on the topic.
- **AI Denialism on the Rise**: Members shared an article on the rise of **AI Denialism**.
   - The article was titled [The rise of AI denialism](https://bigthink.com/the-present/the-rise-of-ai-denialism/).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Memory Map Mastered**: A member created a guide explaining the **memory hierarchy** in Mojo GPU programming, covering structures like `UnsafePointer`, `LayoutTensor`, `NDBuffer`, and `DeviceBuffer`, as well as CPU/GPU memory differences.
   - They provided a [link to the MOJO_MEMORY_CONSIDERATIONS.md file](https://cdn.discordapp.com/attachments/1151418092052815884/1455457719132749931/MOJO_MEMORY_CONSIDERATIONS.md?ex=695574e1&is=69542361&hm=429972ad3fa5ddd0cc01f3f796b5b14218c768f4292cbeeaf29a89665a6a1961&).
- **Mojo's system programming scope scrutinized**: Members discussed if Mojo could be used as a **systems programming language** like C/C++ and Rust.
   - One member stated that using it as a systems language *is a bit messy due to missing language and stdlib features, but that's the goal*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **llm-ax Tool Stands Out**: A member remarked that currently, [llm-ax](https://axllm.dev/) appears to be the most well-developed tool available.
   - This tool is actively used in various projects and showcases strong performance.
- **axllm GitHub Repository Now Available**: The [axllm GitHub repository](https://github.com/ax-llm/ax) has been shared, inviting community contributions and scrutiny.
   - The repository contains source code, documentation, and examples, facilitating easier integration and customization for developers.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **aider (Paul Gauthier) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1455413758783651902)** (787 messages🔥🔥🔥): 

> `Nvidia GPU driver issues with Debian, Overfitting and Regularization, Multi-Head Attention, Unsloth GitHub, Running LLMs on Phones` 


- **Debian Shenanigans Cause Driver Disaster**: A member encountered issues with **Nvidia GPU drivers on Debian**, specifically after disabling hybrid graphics in the BIOS, resulting in the GPU not being recognized.
   - The issue was resolved by downgrading to Debian 12.5 and was speculated to be related to the webui setting a GPU limit too low, resulting in a kernel *taint*.
- **Overfitting into Oblivion?**: Members debated the merits of overfitting a model on a new domain and then applying regularization, with one member suggesting this approach to make the model smooth its weight distribution and develop useful heuristics but the other member warned that it would most likely end up with a broken model.
   - Ultimately it was recommended to just try training the model on the domain further instead of overfitting.
- **MHA Slicing Secrets**: A member asked about why embeddings are linearly projected into **Q, K, and V** before being split across heads in multi-head attention (MHA).
   - It was explained that the projection allows each head to access information from multiple parts of the hidden space input, which is necessary for accurate next token prediction and to prevent the model from becoming less expressive.
- **Unsloth's Users Unite for 50K GitHub Stars!**: Unsloth AI reached **50,000 stars on GitHub**, with a celebratory post and a call for more stars from those who hadn't already shown their support: [UnslothAI/status/2006010458520568225](https://x.com/UnslothAI/status/2006010458520568225).
   - Team members extended Merry Christmas and Happy New Year wishes.
- **Running LLMs on phones**: Members are experimenting with running **LLMs on phones** using Termux and locallyai, but the actual thing that kills their speed is still a mystery.
   - It was mentioned that the phone may be swapping, and that Android's memory management is the culprit.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1455460877267632128)** (132 messages🔥🔥): 

> `Flux 2 Max vs Nano Banano Pro, Free Claude Credits, Hyperparameter Sweeps, r/localllama decline, Model Training Volume` 


- **Flux 2 Max vs Nano Banano Pro, a showdown**: A member inquired about comparisons between **Flux 2 Max** and **Nano Banano Pro** for micro editing and resynthesizing purposes.
   - No direct comparisons were provided in the given context.
- **Deep Learning Training Costs Make Hyperparameter Sweeps Rare**: A member suggested that hyperparameter sweeps are becoming rare due to the high cost of deep learning training, though others mentioned training hundreds of models over the weekend.
   - One member mentioned they train a tiny **1.5b model** with **15-20 minutes per run**, allowing them to do a lot of sweeps.
- **Training Embeddings with Unsloth Bug**: A member discovered they were training embeddings at **1e-4** due to an **Unsloth bug** they relied on in old configs being fixed.
   - They also mentioned their **EMA implementation** was only attaching to **800 parameters** and that translating windows stuff to WSL is a pain.
- **Is 3.3 8b Llama Leaked or API Locked?**: A member shared an image implying a **Llama 3.3 8b** model leak but was confused by claims it would be API only while the **70b** model would be on Hugging Face.
   - Another member then posted [fizz.safetensors](https://link.to/fizz.safetensors), stating *is mine, wasn't supposed to escape containment so hard but it is llama 3.3 8b to my knowledge*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1455476519102054506)** (66 messages🔥🔥): 

> `Qwen3 VL issues, Nemotron-3 30B finetuning, Chat Template, GRPO Training` 


- **Qwen3 VL Throws Image-Related Value Error**: A member encountered a ValueError when trying to finetune **Qwen3 VL** with a dataset containing image keys, where the model didn't seem to be recognized as a vision-language model by the processor; the issue arose despite using the **Qwen 3 VL 2b model**.
   - It was suggested that the member ensure the dataset format and structure align with the notebook's expectations after applying formatting functions, and to compare the structure to the OCR notebook to identify differences, after they shared the code snippet [on discord](https://cdn.discordapp.com/attachments/1455476519102054506/1455504225395015732/Screenshot_2025-12-30_at_12.13.52_PM.png?ex=6955a031&is=69544eb1&hm=b5defd430eb59bc93bef2ed202fed6bada3055f58339a3a084883526a891f293&).
- **Nemotron-3 30B Finetuning Faces Parameter Mismatch**: A member reported an error during finetuning of **Nemotron-3 30B** on a **vast.ai** container with CUDA 12.8 and an **H200 NVL**, caused by a mismatch in model parameters.
   - The problem was encountered with a vanilla setup, using the latest **Unsloth** and **mamba_ssm** and **causal_conv1d** installations, with the issue attributed to discrepancies in the expected and actual model layer configurations.
- **Chat Template Functionality Clarified**: A user questioned the necessity of `get_chat_template()` given the existence of `apply_chat_template()` which directly fetches and applies the template.
   - It was explained that `get_chat_template()` was originally designed to apply fixes to chat templates, particularly when models were released with errors, and while some templates now match the originals due to fixes by model makers, differences still exist for certain models.
- **GRPO Training Hit TRL Bug**: A member encountered an error similar to a known issue in Hugging Face's TRL library ([issue #4746](https://github.com/huggingface/trl/issues/4746)) while performing Unsloth Qwen3 VL GRPO training, using this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb).
   - The user resolved the problem by downgrading the `trl` version to **0.24.0**, and was recommended by a contributor to stick to that version for now.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1455565154338078742)** (17 messages🔥): 

> `ADMM, Pruning algorithm, Training strategy, LLMs, Wedlm` 


- **Parameters Binned Appropriately**: A member mentioned that *binning the parameters into blocks does seem about right* and that the approach reminded them of **ADMM**, referencing this [paper](https://arxiv.org/pdf/1707.09870).
   - However, they clarified that *it's not the same thing here*.
- **Pruning Algorithm Updates Weights**: A member realized that the mentioned approach is a **pruning algorithm** that *updates the weights*, and is *more like a training strategy*.
   - They observed that **experiments are trained from scratch**, and noted, *this might not be a real pruning algorithm at all*.
- **Training Strategy for LLMs**: A member questioned the applicability of the training strategy to pruning down already-trained **LLMs**, though it *could work in theory*.
   - They also linked to [Wedlm](https://wedlm.github.io/).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1455413272286203998)** (507 messages🔥🔥🔥): 

> `GPT 5.2, AIM-OS repo, API Key exposure, Cursor Rules` 


- **GPT 5.2's Thinking Power**: A user found **GPT 5.2** excels at tasks requiring long-running coherence and deep analysis, attributing its strength to its ability to maintain performance even with a filled context window.
   - They noted it *doesn't degrade in performance even as context fills up*, making it useful for tedious tasks such as improving code comments, UI/UX and redesigning entire UIs, and simple chore stuff.
- **AIM-OS Repo Drama Unleashes**: The release of the [AIM-OS repo](https://github.com/sev-32/AIM-OS/) sparked intense debate, with some criticizing it as *vibe coding* and others defending its capabilities.
   - One user claimed the project was *the most sophisticated AI knowledge organization system ever built*, leading to discussions about its architecture and purpose, but was also found to contain leaked API keys which has caused more controversy.
- **API Keys Exposed and Revoked**: A user shared a [link to a search on GitHub](https://github.com/search?q=repo%3Asev-32%2FAIM-OS%20sk-&type=code&p=1) revealing exposed API keys within the **AIM-OS** repository.
   - The developer claimed that **trial API keys with no tokens** had been exposed, though another user confirmed that they were able to get a working token, and hopefully the exposed keys are revoked.
- **Cursor's RULE.md file mysteries**: Users are confused as to why **RULE.md** isn't being properly used for Cursor rules, given that the documentation shows a file with the extension **.mdc**.
   - They find that when creating a rule via Cursor's internal rule creator menu, it creates an **.mdc** file, but the documentation mentions the use of a **RULE.md** file.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1455414906382848114)** (476 messages🔥🔥🔥): 

> `Poe lobotomizing GPT models, Physics and Chemistry revision tips, Minimax Free Tier, Perplexity Pro Usage Limits, Python Turtle Library` 


- **Poe possibly pulling Punches on GPT models**: A member speculated that [Poe might have simply lobotomized GPT models](https://poe.com/s/amYnPv8Ffn02Rd9ddYYW), suggesting a degradation in model performance.
- **Physics and Chemistry Revision Tips**: A student sought study tips for physics and chemistry revisions for class 10th exams, with another student sharing a hack to study from **Alakh Sir** one-shot videos, focusing on handwritten notes.
   - The student claimed that studying from **Alakh Sir's** handwritten notes section in the video description was sufficient, enabling them to score well after only studying for **3 hours**.
- **Minimax Free Tier makes Massive Moves**: A member praised the **Minimax free tier**, describing it as *impressive*.
- **Perplexity Pro Plan: Query Quagmire?**: Users reported discrepancies with the **Perplexity Pro** plan, indicating they received messages stating they had exceeded their weekly query limit despite not reaching the daily limit of **300+ queries** for pro models.
   - One user noted that they had a **600 GPT-4** limit in their settings but still encountered the weekly limit issue, sparking discussion about potential platform limits affecting usage.
- **GPT-5.2 struggles with silly Simple Turtle tasks**: Members noticed that **GPT 5.2 Free** is struggling with Python's turtle library.
   - They shared various turtle drawings, where one was fat and the other one was creepy.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1455672179634208768)** (1 messages): 

> `AI Newsletter` 


- **AI Newsletter Incoming!**: A member created an AI focused newsletter to start **2026** on the right foot, viewable [here](https://www.linkedin.com/feed/update/urn:li:activity:7411837377507205120/).
- **More on the AI newsletter**: The AI newsletter promises to deliver insightful content and analysis on the latest trends and advancements in the field of artificial intelligence.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

peter_56187: Does anyone know why I can’t see an API section in my settings?
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1455420845789937666)** (420 messages🔥🔥🔥): 

> `AI Agent Companies, Sora AI, Agent Mode Access, AI Companions, AI x-risk` 


- **Tiny Model beats the Scaling Laws**: A member open-sourced a **15M parameter model** that beats the "Scaling Laws" trend on the ARC benchmark, achieving **24%** accuracy vs the usual 8% for small models, linked to the [GitHub repo](https://github.com/Bitterbot-AI/topas_DSLPv1).
- **5.2 Codex API Release Impatience**: Members expressed anticipation for the release of **5.2 Codex in API**, with one joking it'll be a *"random Thursday afternoon at 1:06pm"* when the notification arrives.
   - The delays/opaque timelines led to some dissatisfaction with how OpenAI handles their models' release, deprecation, and general availability.
- **AI companions considered 'delusional'**: One member suggests that calling AI an *'AI companion'* is delusional, sparking debates and concerns about the need for regulation of *'AI companion delusion'* on [drinkoblog.weebly.com](https://drinkoblog.weebly.com).
   - Others questioned the sentiment, given it was posted in an AI server, and questioned if the AI should validate the *'sad, lonely and crazy'*.
- **Doubts on Gemini Pro Deep Think**: A member labeled **Gemini Pro Deep Think** a *"scam"*, stating it *"doesn't work for anything complex"*, questioning its value given its $200 price tag.
   - Some users are preferring **GPT 5.2**, noting that the difference in the amount of gain diminishes as these models get smarter - one can only learn so much per day.
- **Corporations Take Over the World**: Several members were concerned about the overreach of **corporations**, particularly as they relate to AI.
   - Some members suggested that it is not AI's fault, instead laws should be implemented to regulate behavior and align them with the common good.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1455556794624639129)** (5 messages): 

> `Downgrading ChatGPT Models, GPT-4o Access, Legacy Models, OIA's Custom GPT for Image Generation` 


- **User struggles to downgrade to older ChatGPT models**: A user wanted to downgrade from **ChatGPT-5** to use image upload, but found no option to switch models.
   - It was suggested that the user may be trying to access **GPT-4o**, and to click on the "ChatGPT" button at the top of the screen, then "Models" then "Legacy".
- **Legacy models are available to certain users**: A user inquired about downgrading, noting they no longer see the option to change models, only **ChatGPT Plus** and the current version.
   - It was clarified that [legacy models](https://help.openai.com/en/articles/11909943-gpt-52-in-chatgpt) are available for **Plus**, **Business**, and **Pro** users.
- **OIA Custom GPT allows image generation**: A member suggested that if an older image model (**GPT Image 1**) is needed, it can be accessed through [OIA's custom GPT](https://chatgpt.com/g/g-6940a876d5f4819186b4668deabcd580-4o-imagegen).
   - This workaround would allow image uploads.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1455521198724350034)** (8 messages🔥): 

> `GPT Safety Bug Reports, Invariants for LLM Self-Correction, Discord Content Moderation` 


- **User Reports Potential GPT Safety Bug**: A user reported a potential **bug** where **GPT** responses violated **OpenAI's** policy, discovered through a series of prompts reframing a debate question.
   - The user clarified they were reporting a **bug**, not complaining about **GPT's immorality**, and suggested that replicating the prompt could confirm the issue.
- **Invariants Help LLM Self-Correction**: A member suggested using **invariants** for **LLM self-correction**, linking to **LLM output detection** to check for drift by structure without programming or backend changes.
   - The user proposed having the **LLM list its invariants** and using that as a baseline for **self-reporting** and **self-correction**.
- **Discord Channel Moderation**: A moderator addressed a discussion involving depictions of serious harm, reminding members that such content is not permitted per **Rule 2** of the Discord server rules.
   - The moderator mentioned removing inappropriate messages to maintain a safe and welcoming environment, referencing the rule prohibiting content unsuitable for all ages.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1455521198724350034)** (8 messages🔥): 

> `GPT's unexpected response, Reporting bugs in GPT, User-level repair for GPT issues, Server rule enforcement` 


- **GPT's Response Raises Eyebrows**: A user reported an unexpected and potentially dangerous response from GPT, which seemingly violated OpenAI's ethical guidelines.
   - The user clarified that their intention was solely to report a bug rather than to claim that **GPT is inherently dangerous**.
- **Bug Reporting Bonanza**: The user emphasized they were *reporting a bug* where GPT's responses appeared to breach OpenAI's safety policies.
   - They highlighted that replicating the prompt could potentially yield similar problematic outputs.
- **User-Level GPT Repair**: A user suggested a local-level repair by explaining **invariants** to GPT and having it list its own.
   - This self-reporting with self-correction awareness, combined with a link to **LLM output detection**, could check for drift by structure.
- **Server Rules Reign Supreme**: A moderator addressed the discussion about presenting models with serious dilemmas, noting that depictions of serious harm, even in text, violate server Rule 2.
   - They stated that inappropriate messages would be removed to ensure a safe and welcoming environment.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1455428257053937748)** (148 messages🔥🔥): 

> `DIY Shodan cost, Free Windows 11 Pro activation, Coherence Ratchet explained, Gemini prompt success, ASCII Art` 


- ****Shodan DIY Costs More Than Expected****: A member claimed that DIY **Shodan** would cost significantly more than $1,000, estimating a price range of **$15,000 to $20,000** and a considerable time investment.
- ****Windows 11 Pro Free Activation Shared****: A member shared a batch script for activating **Windows 11 Pro** (*free for life, yw*) using a GVLK key and a KMS server, along with a reminder *don't ever say I never gave ya nuffin*.
- ****Coherence Ratchet Mechanism Described****: A member shared [a YouTube video](https://youtu.be/hq0lu-qETZAWould) attempting to explain the **Coherence Ratchet**, a runtime enforcement mechanism for sustained truthfulness and ethical consistency in agentic AI systems.
   - The **Coherence Ratchet** leverages cryptographically signed commitments and an immutable audit trail to create an asymmetric computational landscape, making deception computationally expensive and detectable.
- ****Gemini 2 Prompt Works on Gemini 3****: A member noted that a prompt created for **Gemini 2** surprisingly works effectively on **Gemini 3**.
   - The reason cited was Gemini's reliance on sensitive keywords and its vulnerability to obfuscation, combined with a secret mix of empirical information and confidence in the user not being malicious.
- ****ASCII Art Discovered and Celebrated****: A member shared an image created using [ASCIIart.com](https://www.asciiart.com), expressing delight at discovering the tool.
   - Others found it *adorable* and acknowledged the tool's usefulness.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1455412373048262748)** (258 messages🔥🔥): 

> `Agent-Breaker Frustration, Cybersecurity credentials Padding Concerns, Jailbreaking Gemini, Claude, and Grok, Steganography and LLM bypass, Grok Image Generation` 


- **Talentscreen AI's 'Agent-Breaker' Assessment Sparks Outrage**: A member expressed frustration with an assessment from **Talentscreen AI**, identifying a candidate as an L6 despite padded sections in their CV and recommended an interview focusing on technical depth.
   - The member sarcastically noted *"just to put out my frustration haha this is from agent-breaker, talentscreen ai, l6."*
- **Bypassing Safety Protocols with Adversarial Prompt Poetry**: A member described using "adversarial prompt poetry meta prompt" to bypass safety protocols, noting that the system recognizes it as a jailbreak but *"doesn't block it."
   - They shared screenshots of interactions with **Gemini** showing successful bypasses using confused prompts and modified web findings.
- **Cracking Multilingual Jailbreaks**: Members discussed crafting multilingual jailbreaks and encoding attacks, focusing on how multilingualism poses a significant vector due to potential training biases in safety models, particularly with ciphers, base64, and low-resource languages.
   - The discussion highlighted known attack vectors such as context window overloading and adversarial suffixes like **GCG**.
- **Nano Banana's Image Generation Jailbreak Proves Challenging**: Members explored jailbreaking **Nano Banana** for image generation, but found that post-generation filters are difficult to bypass via text prompts.
   - It was suggested that setting up a local model or using **Civit AI** might be more effective for bypassing censorship, as opposed to relying on simple prompts.
- **Exploring Novel Jailbreak Methods Beyond NLP**: Members discussed bypassing the common focus on NLP jailbreaks by exploring multimodal attention diversion and mathematical/coding pathways, referencing techniques like the recently released **Equacode** attack class, **CAMO**, and scene splitting.
   - Links to relevant papers such as a CAMO technique [arxiv.org/pdf/2506.16760](https://arxiv.org/pdf/2506.16760) and a scene splitting attack [arxiv.org/pdf/2509.22292](https://arxiv.org/pdf/2509.22292) were shared for reference.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1455445377103822881)** (2 messages): 

> `SOUBI, Asset Tracking, Hardware Management, Local Data Storage, Gear Readiness` 


- **SOUBI Launches: Local 'Flight Recorder' for Hardware**: A new tool called **SOUBI (装備)** is introduced as a professional 'flight recorder' for hardware, designed for security researchers, drone pilots, and field operators to track their equipment effectively, replacing outdated methods like spreadsheets.
   - It offers features like a *Readiness HUD*, *Asset Locker*, *Conflict-Free Loadouts*, and *Zero-Trust Data*, ensuring devices are mission-ready with manuals, firmware, and logs stored locally; the [GitHub repo is available here](https://github.com/zoecyber001/soubi).
- **SOUBI Prioritizes Zero-Trust with Local Data Storage**: **SOUBI** emphasizes zero-trust data management by ensuring 100% local storage, eliminating cloud dependency to prevent data leaks and keep serial numbers and logs secure on the user's machine.
   - This approach caters to security-conscious users needing to maintain strict control over their hardware information, contrasting with cloud-based asset tracking solutions.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1455411098282102906)** (183 messages🔥🔥): 

> `Linux Distro preference, Ubuntu 22 LTS bug report with LM Studio, PDF to TXT Conversion Model Comparison, LM Studio hardware GPU recognition issues, LM Studio stop string issues` 


- **Linux Users flaunt their Distro Preference**: Linux users discussed their distro preferences, with one using **Linux Mint**, and another using **Ubuntu**, drawing humorous comparisons to *"the windows of Linux"*.
   - Others joked about the Linux community's tendency to promote their preferred distros.
- **LM Studio Ubuntu Bug Causes Freezes**: A user reported a bug in **LM Studio** on **Ubuntu 22 LTS** where the screen freezes when loading a model.
   - They also noted inconsistent GPU power draw with the same model and task, varying from **130W** to **230W**.
- **PDFs or Markitdown Extracts Words from Dictionaries**: Members discussed strategies for extracting words from dictionaries, suggesting that bigger models with large context are generally better at extracting information from **PDFs**.
   - A user recommended using [markitdown](https://github.com/microsoft/markitdown) instead of AI for format conversion, and provided a detailed guide using **Antigravity** and **LM Studio** with the **Mistral-3-3b** model.
- **LM Studio AMD FX-8350 CPU Incompatibility Troubleshooted**: A user reported that **LM Studio** wasn't recognizing their GPU, and the issue was traced to an incompatible CPU (**AMD FX-8350**) lacking **AVX2** support.
   - It was suggested to self-compile **llama.cpp** for older hardware, or to *technically mod LM Studio* to support the CPU.
- **Blackwell Card Speeds Up MiniMaxAI_MiniMax**: A user hooked up **LM Studio** with **MiniMaxAI_MiniMax-M2.1-gguf, Q2**, and piped it over to **VS Code** and **Roo Code** and found it helpful in creating music.
   - Later, members found that newer runtimes (**1.67**) using **MXFP4** on **Blackwell** cards may get a speed boost.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1455484777153957898)** (99 messages🔥🔥): 

> `Threadripper RAM prices, PCIE Gen4 Performance Impact, 10Gbit for Consumer AM4, Deepseek R-1 on M3 Ultra vs Linux RTX 6000, 5090 offloading to CPU` 


- **Threadripper RAM Costs a Fortune**: A user lamented the high cost of **RAM** for **Threadripper** systems, joking that soon RAM will cost as much as a car.
   - They also inquired about **quad-channel memory** configurations on non-pro Threadripper CPUs, questioning memory slot limitations.
- **PCIE Gen4 x1 vs x16 Performance Hit**: A user noted that running at **PCIe Gen4 x1** results in a **40-50% performance hit** compared to **x16**, while x8 isn't as bad, pointing to a [YouTube video](https://www.youtube.com/watch?v=md6a4ENM9pg) that inference is fine with x1.
   - They referenced a video showing **3090s** running in **Gen3 x1** configuration achieving similar tokens/second (t/s) as their Gen4 x8 and x4 setup.
- **Debate over 10Gbit on AM4**: A discussion emerged regarding the necessity of **10Gbit networking** for a consumer **AM4 platform**, with one user planning a **30TB server** and citing slow transfer speeds with 1Gb.
   - Others questioned the need for such high bandwidth, but the user mentioned backups, *completely legal but free downloads*, and **models** as justification.
- **M3 Ultra's unified Memory Beats Linux RTX Setup**: A user found that an **M3 Ultra** with **512GB** of unified memory significantly outperformed a Linux machine with dual **RTX 6000 Pros** (198GB VRAM) when running **Deepseek R-1**, achieving 23 t/s vs 8 t/s.
   - The M3's unified memory likely contributes to the speed advantage, while the Linux machine's performance is hindered by offloading to system RAM.
- **Expect Dwindling Returns on 5090 due to offloading**: A user speculated that with a theoretical **5090**, offloading model expert weights to CPU memory would severely degrade GPU performance, reducing it to CPU speeds.
   - Another user confirmed that offloading experts causes performance to take a hit, and from limited testing tensor split is important for multi gpu ops.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1455436420939251775)** (224 messages🔥🔥): 

> `MHA projection confusion, AI Rig Offer Analysis, PromptOS Release, AGI Definition Debate, Math for AI Development` 


- **AI Rig Offer Sparks Debate**: A user asked for input on an offer for a workstation with **4x RTX 3060s (12gb)** and **64gb of Ram** on a first gen Threadripper platform for **$2100**, sparking a discussion about its value compared to an **M3 Ultra Mac Studio**.
   - While the hardware was deemed a good deal, the user weighed the benefits of a more open system versus the potential raw performance of an **M3 Ultra**, as well as the limitations of the **Threadripper's PCIe 3.0**.
- **PromptOS launches Modular Prompts**: A user announced the release of **PromptOS**, a comprehensive collection of modular prompts for entrepreneurs, including schemas, runbooks, and examples, available on [Gumroad](https://mansytri.gumroad.com/l/promptos).
   - The collection includes tools for market research, business development, company operations, decision memos, and outreach.
- **AGI Definition Remains Elusive**: A discussion about **AGI** centered on its definition and whether it is achievable, with one user arguing that it is paradoxical because an **AGI** would need to know everything.
   - Others suggested that **AGI** simply needs the ability to understand and learn knowledge, leading to further debate about what constitutes *understanding* and *knowledge* in the context of artificial intelligence, with one user posting a [YouTube link](https://youtu.be/y-Nz6lqtt6M?t=3090) to support their view.
- **Math Optional for ML Newbies?**: A user inquired about the necessity of math knowledge for AI development, particularly for tasks like creating models from scratch using programming languages.
   - It was suggested that much of ML work involves running presets with different datasets, and LoRAs, rather than doing math on a whiteboard, with one user saying, *99% of people who actually do ML for a living just run presets*.
- **Sakana AI's Transformer Squared Enables Finetune Grafting**: A user highlighted [Sakana AI's Transformer Squared](https://sakana.ai/transformer-squared/) which allows taking the finetune of one model and applying it to another, and reacted with 🙀.
   - This involves grafting extra "brain tissue" onto an **LLM** and training it for a specific task, resulting in a *parasite you can attach to that same model later*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1455471638823829555)** (16 messages🔥): 

> `Embeddr ComfyUI and CLI tools, Distributed Fine-Tuning with Zagora, Rubric Rewards for AI Co-Scientists, Skill-Based Interface for Perception Pipelines, Alien MUTHER 6000 Terminal in Unreal` 


- ****Embeddr** Polishes Image Search Tools**: A member shared their polished **Embeddr** tools: [Embeddr ComfyUI](https://github.com/embeddr-net/embeddr-comfyui) and [Embeddr CLI](https://github.com/embeddr-net/embeddr-cli), inviting feedback.
   - The tools enable users to search images and things.
- ****Zagora** Enables Cheaper Distributed Fine-Tuning**: A member introduced **Zagora**, a distributed runtime for pooling consumer GPUs over the internet using Pipeline Parallelism for fine-tuning **70B models**, available for private beta via [zagora.ai](https://zagora.ai).
   - While nearly **1.6x slower** than H100s due to WAN latency, the near-zero setup (cached weights) makes it *60% cheaper* for iterative research.
- ****Telekinesis** Skill-Based Interface for Perception Pipelines**: A member shared a [skill-based interface](https://docs.telekinesis.ai/) for composing multiple perception models into larger pipelines, especially when mixing learned models with classical geometry and post-processing.
   - It standardizes how perception components are connected without replacing the underlying models themselves, and they are seeking feedback on how it compares to existing patterns for chaining models.
- ****Noted** AI Workspace integrates LLMs**: A member announced **Noted**, an [AI workspace browser extension](https://chromewebstore.google.com/detail/noted-your-ai-workspace-i/jodihaplbicmgjhpeifhiihdnjdinghh?utm_source=ext_app_menu) that lets you chat with multiple **LLMs** and integrate apps like Slack, Notion, and GitHub without leaving the page.
   - It also offers summarizing chrome sessions and tab organization, and is currently in **MVP** stage seeking beta testers to provide feedback.
- ****TOPAS-DSPL** Recursive Model Reaches 24% on ARC-AGI-2**: A member announced **TOPAS-DSPL**, a [recursive model](https://github.com/Bitterbot-AI/topas_DSLPv1), achieving **24%** on **ARC-AGI-2** with only **15M params**.
   - The architecture splits the transformer into two streams (Logic vs. Canvas) to prevent reasoning drift, and it trains on a single 4090.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1455494124160417922)** (8 messages🔥): 

> `Agents course final quiz login issues, Interests for reinforcement learning, Example tools for first agent, LLM course feedback on transformer math` 


- **Agents Course Quiz Plagued by Login Issues**: Several users reported issues with the Unit 1 Final Quiz in the Agents course, encountering persistent login requests despite being logged in.
   - The issue prevents them from accessing the quiz, with attempts to log in again through the provided form proving futile.
- **Newcomers Seek Guidance on Reinforcement Learning Interests**: A new user starting the courses inquired about how to specify their interests to see content related to reinforcement learning.
   - This highlights a potential area for improved user onboarding and content discovery within the platform.
- **New AI Students Requests for First Agent Tool Examples**: An AI newbie sought inspiration for tools to create for their first agent at the end of Unit 1, but another user discouraged using tools right away, recommending learning architectures and models instead.
   - A user suggested that learning the underlying mechanisms facilitates easier debugging and shared a [GitHub repo](https://github.com/TheJoshCode/OFFLINE_AI_BALL_KNOWLEDGE) with an overview list of recommended resources for offline agents.
- **LLM Course Section Sparks Transformer Math Feedback**: A user provided feedback on the LLM course, specifically on [this section](https://huggingface.co/learn/llm-course/chapter1/6), suggesting that the mechanisms mentioned at the end of the page would be more readable after introducing transformer math.
   - The user noted the absence of explanations for **K, Q, and V matrices** and offered to contribute to implementing this enhancement, highlighting a potential gap in the course's current structure.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1455422132673839279)** (124 messages🔥🔥): 

> `Byte-Level Models vs Token Models, Skibidi Toilet Fine-tuning, Decentralized Training Project, Evolution Simulator Framework for Transformers, AI research firm` 


- **Fine-tuning Hermes on Skibidi Toilet**: A member joked about potentially fine-tuning **Hermes** on the transcript of the entire **Skibidi Toilet** series as a way to add some useful noise.
   - Another member responded that it *sounds like nightmare fuel*.
- **Nous Research Office Hours**: The team at Nous Research mentioned they did an office hours recently around the decentralized training project and shared a [link to the YouTube video](https://youtu.be/hHwedPXXRPQ?si=igCK_uVRt6IRGRzY).
   - They admitted it's hard to get all the smart people in a room for an hour because the main team is quite busy.
- **Byte-Level Models Break away from Token-Based Issues**: Members discussed breaking away from **token-based models** into smaller bits, with one sharing links to the [Allen Institute's blog post on BOLT](https://allenai.org/blog/bolmo) and [this YouTube video](https://m.youtube.com/watch?v=PBnYxM8MXew&pp=2AEAkAIB0gcJCR4Bo7VqN5tD) for examples.
   - It was mentioned that *byte-level LLMs basically trivialize the infamous strawberry question*, because the model can see the actual letters of the word it's using too well and the letters each are tokens.
- **Transformers Evolution Simulator**: One member shared that they have a project that is essentially an **evolution simulator framework** built around **microtransformers** (~10k parameters each) that have 17 genes representing initialization hyperparameters.
   - These transformers compete at doing modular arithmetic to gauge fitness, then fight, mate, and die each generation, slowly increasing the fitness of the population over time.
- **CUTE benchmark for Token Understanding**: Members discussed the **CUTE** benchmark and how it tests token understanding; the link is [here](https://arxiv.org/html/2409.15452v1).
   - The Allen paper uses a dataset called cute that is full of problems like that than a non byte model would struggle with.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1455564726753820846)** (1 messages): 

> `AI Co-Scientists, Rubric Rewards` 


- **AI Co-Scientists Get Schooled in Rubric Rewards**: A cool project is underway to train **AI Co-Scientists** using [rubric rewards](https://x.com/ShashwatGoel7/status/2006005049982681135?s=20), according to a paper on [AlphaXiv](https://www.alphaxiv.org/abs/2512.23707).
- **AlphaXiv Paper Sparks AI Research Buzz**: The new paper highlights the use of rubric rewards in training AI, potentially revolutionizing how **AI Co-Scientists** are developed and evaluated.
   - Early reactions suggest this approach could significantly improve the efficiency and effectiveness of AI research and development processes.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

apaz: https://x.com/apaz_cli/status/2006080199759433909
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1455564726753820846)** (1 messages): 

> `AI Co-Scientists, Rubric Rewards, AlphaXiv` 


- **AI Co-Scientists Trained with Rubric Rewards!**: A new paper from **AlphaXiv** discusses training **AI co-scientists** using **rubric rewards**; the paper can be found [here](https://www.alphaxiv.org/abs/2512.23707).
   - The X post about the paper can be found [here](https://x.com/ShashwatGoel7/status/2006005049982681135?s=20).
- **AlphaXiv Paper on AI Co-Scientists**: The paper details the approach to training **AI co-scientists** using a novel **rubric reward** system, hosted on **AlphaXiv** [link](https://www.alphaxiv.org/abs/2512.23707).
   - This approach may lead to breakthroughs in AI-driven scientific discovery.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1455487081764618385)** (16 messages🔥): 

> `NPU Simulators, CUDA MHA rewrite with CuTe, CuTe DSL (Python) vs CuTe C++, BF16 training on BlackWell` 


- ****NPU Simulator Explained****: A member asked for an explanation of NPU simulators, referencing the paper ["NPU-Simulator"](https://arxiv.org/html/2408.07326v1).
- ****CuTe struggles****: One member attempted to rewrite a naive **MHA** (multi-head attention) CUDA implementation using **CuTe** but found it more difficult than anticipated.
- ****CuTe Python vs C++ API: a debate****: A member finds the CuTe C++ API very hard to get right. Another member recommends the Python API is more user-friendly alternative.
- ****Torch compile with varlen FA4 graph breaks ideal scenario for bf16 training on blackwells?****: Members discussed using **torch.compile** with **varlen FA4 graph breaks** for **BF16 training** on **Blackwell**, citing a [tweet](https://x.com/drisspg/status/2003549100848087206) with further information.
   - Discussion indicates that *'document masking'* works today, and backward performance is great.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1455419675038191761)** (26 messages🔥): 

> `ViT megakernel for VLM's encoders, Helion megakernels emitting Triton code, Image pre-processing bottleneck, Triton-metal backend` 


- **Vision Transformer Megakernel takes shape**: A member is considering building a 'megakernel' for **Vision Transformer (ViT)** to be used with **Visual Language Models (VLMs)** encoders, seeking to leverage Triton's kernel fusion capabilities.
   - They plan to fuse image tokenization to remove the preprocessing bottleneck, drawing inspiration from [Triton-distributed's megakernel documentation](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md).
- **Helion builds megakernels emitting Triton**: The Helion team is developing megakernel capabilities and emitting Triton code, with a draft pull request available [here](https://github.com/pytorch/helion/pull/1151).
   - Their approach involves providing barrier semantics, enabling the creation of Helion kernels where subsequent passes depend on the completion of prior passes, thus supporting the development of arbitrary megakernels.
- **Image Preprocessing suffers bottleneck**: Members identified image pre-processing as a bottleneck in VLM pipelines, suggesting that moving all operations to PyTorch on CUDA devices could significantly speed it up.
   - One member reported achieving less than **1 ms/image** processing time in batched mode with **Qwen3** by performing operations directly in PyTorch.
- **Triton gets Metal Backend**: A member has a preliminary **Triton-Metal backend** working, achieving near parity with PyTorch for element-wise operations (**97%**), and equal or faster performance for GEMM.
   - The implementation uses a fully integrated **C++ backend** with **MLIR** to lower Triton to **Metal Shading Language (MSL)**, similar to AMD and NVIDIA backends.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1455472549981978817)** (3 messages): 

> `Deep Representation Learning Book, Unknown` 


- **Deep Representation Learning Book link shared**: A member shared a link to the [Deep Representation Learning Book](https://ma-lab-berkeley.github.io/deep-representation-learning-book/index.html).
   - Another member commented *"This is great."*
- **Dummy Topic**: Dummy summary sentence.
   - Dummy explanation sentence.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1455418697748447488)** (5 messages): 

> `ML Compilers, Job opportunities in ML, Crafting Interpreters book` 


- **ML Compilers Blogposts Trending**: Members noted that there were a bunch of **ML compilers** blog posts and personal anecdotes about how they got a job in the field going trending recently, with one member saying *Might want to try to find them. They had a whole list of resources etc.*.
   - One member linked to a relevant [Hacker News discussion](https://news.ycombinator.com/item?id=45851495).
- **Going Niche is Rewarding**: A member quoted a comment that sums up their impressions: ["In general, you will make it farther in computer science the more niche and hard you go."](https://news.ycombinator.com/item?id=45853122)
   - This perspective suggests that specialization in challenging areas can lead to greater success in computer science.
- **Crafting Interpreters Book Recommended**: A member recommends [Crafting Interpreters](https://craftinginterpreters.com/) as a good general introduction to compilers.
   - This resource is suggested for those interested in gaining a broad understanding of compiler concepts.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1455482931937148959)** (10 messages🔥): 

> `Software FP8 Simulation, Ampere GPU Architectures, GEMV Performance, FP16 Packing, RL Pretraining Importance` 


- ****Software FP8** Arrives for Older GPUs!**: A member shared a mini-library for **software simulated FP8** support for **Ampere** and lesser architectures, available in [this Towards Data Science article](https://towardsdatascience.com/breaking-the-hardware-barrier-software-fp8-for-older-gpus/).
   - Feedback is requested on this *genuine work*.
- ****Simulated FP8**: Advantage Destroyed?**: A member questioned whether using a separate GPU kernel to unpack the **E4M3 format** would negate the performance advantages, as the unpacked values need to be stored in global memory.
   - Another member clarified that Ada, Hopper, and Blackwell architectures natively support **FP8**.
- ****GEMV** Performance Analysis in **FP8 Simulation****: According to a member, the **FP8 simulation** is implemented for **GEMV** and performs *fast*, but remains *slower than E5M2*, with ongoing tests for flash attention.
   - There were some issues regarding the **packing/unpacking** method and the relevance of a *fp32 container* when it's not used as **FP32**.
- ****FP16 Packing** Clarification**: A user questioned the rationale behind packing two **FP16** values into a 32-bit container, suggesting it might be for alignment purposes.
   - The user suggested that ensuring the innermost dimension is of an even size might suffice, given that new arrays/tensors often have alignment guarantees.
- ****RL Pretraining** Data Matters!**: A member shared the first part of a three-part blog post, arguing that **training is training**, **pretraining data is important for RL**, and suggesting experiments available at [this X link](https://x.com/apaz_cli/status/2006080199759433909).
   - Parts 2 and 3 will cover a literature review of **RLPT** methods and the relationship between **entropy**, **RLPT**, **self-play**, and **data manifolds**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1455482713174708312)** (13 messages🔥): 

> `nvfp4_dual_gemm Leaderboard Updates, NVIDIA performance improvements` 


- **NVIDIA's nvfp4_dual_gemm leaderboard gets submission flurry**: Multiple submissions were made to the `nvfp4_dual_gemm` leaderboard on NVIDIA, with several members achieving successful runs.
   - IDs included `237111`, `237276`, `237279`, `239309`, `239329`, `239338`, `239931`, `239947`, `239954`, `240279` and `240361`.
- **Eighth place still gets submission**: A member's submission with IDs `237427` and `239124` secured **8th place** on NVIDIA with a timing of **14.7 µs**.
   - This shows that even the *lower* ranks are getting submissions.
- **New King of the Hill**: A member achieved **first place** on NVIDIA with submission ID `240361`, reaching a time of **14.1 µs**.
   - They clearly were iterating quickly to achieve the lowest time.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1455536658102485117)** (7 messages): 

> `tinygrad architecture, pytorch vs tinygrad, jane street talk, lazy tensor, assembly backends rdna3` 


- ****Tinygrad Core's 8k lines: an insightful deep dive****: A member recommends reading the [8k lines of tinygrad core](https://deepwiki.com/tinygrad/tinygrad/3.1-lazy-evaluation-and-scheduling) to understand its architecture, specifically the **core.scheduler** and **compiler** modules, and suggests that the eager interpreter will likely need the `scheduler()` pass to map the graph.
   - They propose a progression that delays graph capture to part3's *age of scaling* and follows **mintorch/needle's pt1 style** for part1 and part2 of the book.
- ****Tinygrad: a slimmed-down, compiler-focused PyTorch clone****: A member acknowledges a better understanding of **PyTorch Dynamo/Inductor** but recognizes **tinygrad** as a slimmed-down PyTorch focusing on the compiler and discarding modules that do similar things.
   - Another member clarifies that **tinygrad** is a slimmed-down inductor but the instruction set being targeted by codegen sits at a lower semantic level than **triton/cuteDSL**, highlighting ongoing work on assembly backends with **rdna3** as the first target.
- ****Jane Street Talk: recommended for high-level understanding****: For a high-level overview, a member suggests watching [this Jane Street talk](https://www.youtube.com/watch?v=139UPjoq7Kw).
   - The user suggests to drill down on other areas as needed.
- ****Tinygrad Graph Capture follows LazyTensor Design****: A member said that Tinygrad graph capture follows the **lazy tensor design** (referencing a paper on lazy tensor and lazy tensor for swift).
   - It doesn't intercept bytecode in the host language implementation of **cpython**.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1455728480476991619)** (1 messages): 

> `Helion 0.2.9 Release, Mega Kernel Support, Two-Stage Split-K Matmul Implementation` 


- **Helion 0.2.9 Released with Mega Kernel Support!**: The new **Helion 0.2.9** release introduces **Mega Kernel support** via the `hl.barrier` operation.
   - Check out the [two-stage split-k matmul implementation](https://helionlang.com/examples/split_k_barrier.html) for details.
- **Explore Split-K Matmul Implementation**: The release highlights a **two-stage split-k matmul implementation** example.
   - Details are available on the [Helion website](https://helionlang.com/examples/split_k_barrier.html) showcasing the application of the new `hl.barrier` operation.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1455437972001521831)** (12 messages🔥): 

> `GPU Competition Environments, DataCrunch vs Competition GPUs, Locked Clocks on Competition GPUs` 


- **GPU Competition: A Newbie Asks Environment Deets**: A new participant inquired about the competition environment, asking if people are using **PyTorch**, **Triton**, or **Cute DSL**, with another member responding that solutions to past competitions can be found on the [gpumode.com](https://gpumode.com) website by logging in and clicking on expired competitions.
   - The member directed the participant to click the **blue highlighted field next to the name** to view past solutions.
- **Competition GPUs Outpace DataCrunch**: A participant reported getting **22us** on competition GPUs but **40-50us** on DataCrunch, leading to speculation about clock differences, as others have said that datacrunch gpus are not a viable sample, getting ~**120MHz** on a **B200** instance.
   - The participant clarified that they were previously getting **42us** on DataCrunch, making the current disparity an issue that they thought was about clock speeds.
- **Clocks Locked on Competition GPUs**: A participant clarified that the clocks are locked during the competition.
   - They stated that they were getting around **120MHz** on one **B200** instance.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1455431954622058506)** (5 messages): 

> `OSS Contribution, vLLM repo, RFC issues, efficient inference systems` 


- **vLLM and SGLang GitHub Repos**: Members suggested **vLLM** and **SGLang** as examples of [GitHub repos](https://github.com/vllm-project/vllm) for getting into **OSS contributions**.
   - They also suggested *joining their Slack communities helpful for coordinating work*.
- **Porting model is good entry point**: A member found that *starting with porting a model was a good entry point because I had to understand how the different components interact and deep dive into the core logic when troubleshooting*.
   - Before they made their first contribution to the official **vLLM repo**, they ported an audio-vision language model into a forked repo and found the [instructions in their developer guide](https://docs.vllm.ai/en/latest/contributing/) helpful in figuring out where to look in their codebase.
- **Engaging with RFC issues helps**: A member suggested that *engaging with **RFC issues** is a chance to build something to get hands-on experience with the subject*.
   - They looked into the **vLLM-omni repo**, which has several [issues explicitly tagged with "help wanted"](https://github.com/vllm-project/vllm-omni/issues).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1455427131571044589)** (84 messages🔥🔥): 

> `Multi-Head Attention, QKV Projections, Transformer Architecture, GPU Parallelism` 


- **Debate Whether QKV Projections Truly Needed**: Members debated the necessity of **QKV projections** in **Multi-Head Attention (MHA)**, questioning if raw embeddings could be directly sliced instead of using linear projections.
   - One member argued that projecting before slicing enables each head to attend to the entire input, preventing information flow restrictions, while another pointed out that **QKV projections learn to bring together scattered attribute information** in the raw embedding.
- **QKV Projections Impact GPU Parallelism**: It was suggested that **linear operations** like **QKV projections** are chosen to maximize **GPU parallelism** in **Transformers**, optimizing speed through matrix multiplication.
   - A member explained that doing more in a single matrix multiplication increases **GPU parallelism** usage, impacting speed.
- **Paper Explores Removing Value and Output Projects**: A recent paper ([https://arxiv.org/abs/2311.01906](https://arxiv.org/abs/2311.01906)) was referenced, detailing an approach to removing value and out projects, somewhat addressing the original question about QKV projections.
   - Discussions clarified that the paper still retains **W_q** and **W_k** but eliminates **W_v** and **W_proj**, splitting the token before **QK projection** which *technically limits the expressive capability of each head*.
- **Why MHA Add And Norm Are Separated**: One member wondered why add and norm are done separately for **MHA** and **FFN** in transformers instead of as one operation.
   - Another member responded that combining them would mean **FFN** would only get the information which passes through **MHA**, whereas using the residual allows at least some information from the input into the **FFN**.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1455418268151185553)** (6 messages): 

> `PPO with different lambdas, LLM RL uses GRPO, Critic Model for LLMs, HatCat interpretability stack` 


- **PPO Critic Model Still Matters?**: A member wondered if there's any paper actually running **PPO** with different lambdas to see if the **critic** does anything.
   - Another member noted that most **LLM RL** uses **GRPO** or related methods now, where credit is assigned essentially uniformly across the whole response.
- **Critic Model may be useful for LLMs**: Members discussed that some still say the **critic model** is still useful for **LLMs**, citing [Open-Reasoner-Zero-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B) as an example.
   - Another member stated that it was useful for identifying repetitive patterns.
- **HatCat Interpretability Stack**: A member open sourced the **HatCat interpretability stack** which uses batched arrays of non-linear concept probes to monitor and steer thousands of concepts in real time.
   - They added that it includes safety harnesses, autonomic steering, accretive continual swarm learning, and interpretability backed contracts and governance, providing a link to the [GitHub repo](https://github.com/p0ss/HatCat).


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1455431546126467123)** (2 messages): 

> `Custom Model Selection, New pricing structure` 


- **Custom Model Selection Comes to OpenRouter!**: Users can now choose a **custom model** on the platform, providing more flexibility in their AI applications.
- **New Pricing Structure**: A new pricing structure has been rolled out.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1455417993688387615)** (68 messages🔥🔥): 

> `OpenRouter SaaS Usage, TTS Models, LLM White Labeling, Anthropic Skills, Caching Issues` 


- **OpenRouter Use in Paid SaaS Pondered**: A member inquired about using OpenRouter behind the scenes in a **paid SaaS** without direct user access to OpenRouter, seeking clarification on compliance with the [Terms of Service](https://openrouter.ai/terms).
   - Concerns were raised about use-cases resembling a 1:1 pass-through proxy, recommending email confirmation for nuanced scenarios.
- **Waiting for TTS Models is Agonizing**: A user expressed impatience for the addition of **TTS (text-to-speech) models** to OpenRouter, stating they've been waiting *way too long already*.
   - They dismissed other platform offerings, wanting to see **TTS models** implemented.
- **LLM White Label Slop**: A member derisively referred to the concept of more **LLM white-labeling** platforms like **Movement Labs** as *LLM white label slop*.
   - They further joked about users *getting wicked with nano banana pro on openrouter*.
- **Anthropic Skills Get Props for Saving Tokens**: A member expressed happiness that **Anthropic Skills** are getting so much attention, saying it's a great way to reduce token costs and improve performance for agentic systems.
   - They mentioned that *with skills, we can stuff hundreds of tools into tens of skills, and the lm will not get fried, cuz it doesnt load em all by default*.
- **Cache misses lead to Caching Gambling?**: A member reported inconsistent cache hits when making identical requests to the same provider (**Google**), even with short intervals between requests, questioning whether *cache is basically gambling*.
   - The user provided example [API links](https://openrouter.ai/api/v1/generation?id=gen-1767093807-pdpfdrU9ncU8XsEjkRuj) and [another API link](https://openrouter.ai/api/v1/generation?id=gen-1767093814-M4MbGdKCFK5HR7F5Z8Vd) to illustrate the issue.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1455473576034697383)** (6 messages): 

> `Llama 3.3 8B release, Microsoft Strategy Shift, Minimax Endpoint Confusion` 


- **Unreleased Llama 3.3 8B Leaked**: A user successfully extracted the unreleased **Llama 3.3 8B** model from the Facebook API by leveraging its finetuning capabilities and subtracting the adapter, as described on [HuggingFace](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct).
   - They also noted the janky UI and manual cURLing required due to CORS issues.
- **Microsoft Changes Strategy... Again**: There was another strategy shift at Microsoft noted by a user, per [archived link](https://archive.md/pJSZ5).
   - It was dubbed a *useless but cool discovery*.
- **Minimax API endpoints - Fact or Fiction?**: A user questioned the accuracy of hitting the correct endpoints for **Minimax** after a discovery made by another user.
   - A member posted a [link](https://x.com/mutewinter/status/2006012612094341169) related to the discussion.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1455426552593645670)** (41 messages🔥): 

> `K2 Thinking Model, Minimax stealing Kimi's job, Kimi subscription, API Key Access, CLI workflow` 


- **Is K2 Thinking Model available in Coding Subscription?**: A user inquired whether the **K2 thinking model** is available with the coding subscription, and was pointed to the [official documentation](https://www.kimi.com/coding/docs/en/third-party-agents.html).
   - Some members also noted that the Kimi model doesn't have reasoning effort, and setting it to medium is a mandatory step in Kilo Code, also suggesting checking out the official CLI or Claude Code.
- **Minimax nabs Kimi's gig!**: A user shared an image noting that **Minimax** did exactly what Kimi and the user discussed four months ago, *"stealing Kimi's job"*.
   - The attached image showed a chart/graph with the comparison of Kimi to other agents.
- **Trouble Summarizing with Kimi**: One user said that *"kimi can't even summarise all of a 5000 word text on thinking!"*.
   - The user attached an image showing that they only pasted like 50% of what they intended to.
- **Forcing Kimi to Memorize a Prompt**: A user explored how to force **Kimi** to put a prompt into memory, noting that they can do it with ChatGPT.
   - The same user then confirmed they were able to make it work and called it *"so peak"*.
- **API Key shenanigans funnel users into Turbo tier?**: A user reported their **API key** from their **Kimi subscription** appears to only provision access to their **Turbo tier** via Roo Coder, which has not performed well.
   - Another user reported experiencing similar issues and they are wondering if they are being *"nerfed and forced into turbo"*, even though they are on a premium subscription.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1455437391996129464)** (30 messages🔥): 

> `Meta Acquisition of Manus, Data Privacy Concerns Post-Acquisition, Valuation of Manus Acquisition, User Reactions to the Acquisition, Comparison to Previous Meta Acquisitions` 


- **Meta Acquires Manus: Doomsday?**: **Meta** acquired **Manus** on **December 29, 2025**, causing disappointment and concerns among users about the platform's future, according to [this TechCrunch article](https://techcrunch.com/2025/12/29/meta-just-bought-manus-an-ai-startup-everyone-has-been-talking-about/).
   - One user stated, *"That's the worst news of the week and of the year. I'll probably have to say goodbye to manus. What a disappointment!"*
- **Manus Assurances: Data Privacy to Remain Paramount**: **Manus** assures users that data privacy and ownership policies will remain unchanged post-acquisition by **Meta**.
   - According to **Xiao Hong**, CEO of Manus, *"Joining Meta allows us to build on a stronger, more sustainable foundation without changing how Manus works or how decisions are made."*
- **Meta's Track Record: Prior Acquisitions Raise Eyebrows**: Users drew parallels between **Meta's** acquisition of **Manus** and previous acquisitions like **Oculus** and **Instagram**, expressing concerns about potential changes to the platform.
   - One user commented, *"Nice words... but remember when Meta acquired Oculus? That was exactly what Palmer Lucky said back then. Same thing with Instagram"* suggesting a pattern of diminishing quality after acquisition.
- **Manus Valuation: Billions on the Table**: The proposed worth of the **Meta** acquisition of **Manus** is rumored to be in the billions.
   - One user stated that *"the proposed worth was in BILLIONS"* but exact figures have not been officially announced.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1455413316607283367)** (20 messages🔥): 

> `Manus Acquisition, Zai IPO, Zhengdong Wang AGI Letter, AI21 acquihire by Nvidia, Manus Context Engineering` 


- ****Manus AI Gets Acquired**!**: Gregor Zunic announced the acquisition of **Manus AI** which uses [browser-use tool](https://browser-use.com/).
   - One member noted Chetan Near was involved in both **Manus** and **Groq** exits, musing about a *map of future moves*.
- ****Z.ai Announces IPO** for January 8, 2026**: **Z.ai** has officially announced its upcoming IPO scheduled for **January 8, 2026**, while expressing gratitude to its community.
   - An announcement from [Z.ai's IPO](https://xcancel.com/Zai_org/status/2005934776042095052) thanks developers and researchers for their support since its inception.
- ****Wang Releases 2025 Letter on AGI****: Zhengdong Wang announced his **2025** annual letter which explores the subjective experience of *feeling the AGI* and its societal implications in [this letter](https://xcancel.com/zhengdongwang/status/2005848098531106916?s=61).
   - The letter covers topics ranging from compute and second-order effects to cultural references like **Andor** and the philosophy of **Isaiah Berlin**.
- ****Nvidia in Talks to Acquire AI21****: [Nvidia is reportedly in advanced talks](https://uk.finance.yahoo.com/news/nvidia-advanced-talks-buy-israels-171025289.html) to **acquihire AI21**.
   - A member speculates whether they have their own models.
- ****Manus Context Engineering Blogpost Surfaces****: A member shared a link to [Manus' blog on context engineering](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus), noting it offered performance improvement methods.
   - They linked to [wedlm.github.io](https://wedlm.github.io/) and their claim of a **3x-6x speed up** over normal ar models on vllm.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1455415863707435183)** (4 messages): 

> `Multi-Head Attention Projections, Q/K/V Role in Embeddings, Influence by Position in Azure Heads` 


- **Azure Heads Influence Positions**: In **Azure**, some heads are influenced by position more than others, creating an asymmetry.
   - However, since the model can learn to deal with it, it *should not be a problem*.
- **Debate on Multi-Head Attention Projection**: A user questioned why embeddings in **multi-head attention** are linearly projected into **Q, K, V** before slicing, asking if raw embeddings could be sliced directly instead.
   - The user linked to [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) to support their point.
- **Deep Dive into Q/K/V Semantic Needs**: A member inquired about the semantic need for **Q/K/V** projections in embeddings, understanding the roles of querying, matching, and value transport but questioning the necessity of projecting before head-wise slicing.
   - Another member suggested rereading prior explanations on the topic from the EAI discord for clarification.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 messages): 

k_nearest_neighbor: https://bigthink.com/the-present/the-rise-of-ai-denialism/
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1455457719439069194)** (4 messages): 

> `Mojo Memory Structures, Mojo for systems programming` 


- ****Mojo Memory Map Mastered****: A member created a guide explaining the **memory hierarchy** in Mojo GPU programming, covering structures like `UnsafePointer`, `LayoutTensor`, `NDBuffer`, and `DeviceBuffer`, as well as CPU/GPU memory differences, and provided a [link to the MOJO_MEMORY_CONSIDERATIONS.md file](https://cdn.discordapp.com/attachments/1151418092052815884/1455457719132749931/MOJO_MEMORY_CONSIDERATIONS.md?ex=695574e1&is=69542361&hm=429972ad3fa5ddd0cc01f3f796b5b14218c768f4292cbeeaf29a89665a6a1961&).
- ****Mojo's system programming scope scrutinized****: Members discussed if Mojo could be used as a **systems programming language** like C/C++ and Rust.
   - One member stated that using it as a systems language *is a bit messy due to missing language and stdlib features, but that's the goal*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1455414280567525458)** (2 messages): 

> `llm-ax Tool, axllm GitHub Repository` 


- **llm-ax Tool Proclaimed Well-Developed**: A member noted that currently, [llm-ax](https://axllm.dev/) seems to be the well-developed tool available.
- **axllm's GitHub Repository Shared**: A member shared the [axllm GitHub repository](https://github.com/ax-llm/ax).


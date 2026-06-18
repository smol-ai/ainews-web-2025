---
id: MjAyNS0x
title: not much happened today
date: '2026-06-11T05:44:39.731046Z'
description: Unable to generate summary due to an error.
companies: []
models: []
topics:
  - untagged
people: []
---


**a quiet day.**

> AI News for 6/10/2026-6/11/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Top Story: Midjourney Medical**

## What happened


**Midjourney unveiled a medical imaging/scanning system and then published a technical dive on it, triggering a mix of fascination, skepticism, and broader discussion about AI labs moving into hardware/medical devices.**

- Midjourney’s official account posted **“A technical dive inside our new ‘Midjourney Scanner’”** in the main announcement tweet, which appears to be the core launch artifact for the project [@midjourney](https://x.com/midjourney/status/2067422898407837797).
- The launch was preceded or paralleled by discussion of a scanner whose tradeoffs were summarized as: **radiation-free, magnet-free, fast, and low-cost**, but requiring the **person to sit in a water immersion tank** and currently having **coarser resolution than CT/MRI** [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- A demo appears to have been available in person: one attendee said, **“I put my hand in the @midjourney demo scanner tonight”**, framing it as a tangible prototype rather than a purely conceptual announcement [@saranormous](https://x.com/saranormous/status/2067434920134549524).
- The announcement generated strong enthusiasm from supporters who viewed it as evidence of unusually ambitious product direction from Midjourney, including comments like **“this is so amazing”** and **“let inventors like @DavidSHolz invent”** [@saranormous](https://x.com/saranormous/status/2067435669694374217).
- Others interpreted the launch competitively against more incremental AI hardware efforts; one reaction contrasted it with “boring lapel camera” bets and argued other AI labs should “slap yourself” if Midjourney is building this kind of thing [@matvelloso](https://x.com/matvelloso/status/2067446209690869877).
- There was also lightweight technical commentary from people interested in imaging methods, including speculation about detector/emitter arrangements and real-time variants [@johnowhitaker](https://x.com/johnowhitaker/status/2067406686940463328), plus teasing that some users seemed unusually prepared for the launch topic [@johnowhitaker](https://x.com/johnowhitaker/status/2067441084138233963).


## Facts vs opinions


**Factual claims explicitly present in the tweet set**

- Midjourney published a **technical dive** into a product called the **“Midjourney Scanner”** [@midjourney](https://x.com/midjourney/status/2067422898407837797).
- The scanner was described as:
  - **Radiation-free**
  - **Magnet-free**
  - **Fast**
  - **Low-cost**
  - Requiring a **water immersion tank**
  - Having **coarser resolution than CT/MRI**
  [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812)
- A person physically tried a **demo scanner** with their hand [@saranormous](https://x.com/saranormous/status/2067434920134549524).

**Interpretations/opinions/speculation**

- Strongly positive reactions framed the scanner as visionary or “the future” [@saranormous](https://x.com/saranormous/status/2067434920134549524).
- Some observers took the launch as evidence that Midjourney is pursuing a more ambitious hardware roadmap than competing AI labs [@matvelloso](https://x.com/matvelloso/status/2067446209690869877).
- One humorous reply escalated the idea into “next up is full cargo transport by midjourney,” clearly not a factual claim [@yacinelearning](https://x.com/yacinelearning/status/2067433887156887796).
- Independent technical commentary suggested possible future design directions, such as distributed scattered detectors and emitters or real-time systems, but these were not presented as features of Midjourney’s current scanner [@johnowhitaker](https://x.com/johnowhitaker/status/2067406686940463328).


## Technical details and inferred modality


The tweet corpus contains only a limited number of hard specs, but they are enough to outline the project’s positioning.

- **No ionizing radiation**: “Radiation-free” implies the system is not using X-rays/CT-style ionizing modalities [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- **No magnets**: “Magnet-free” differentiates it from MRI, which relies on strong magnetic fields [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- **Water immersion tank**: This is a major clue about the physical sensing setup. Water coupling is common in some acoustic and wave-propagation imaging systems because it improves transmission and coupling between emitters, tissue, and detectors [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- **Resolution below CT/MRI**: The system is not being claimed, in these tweets, to outperform incumbent clinical imaging on resolution; in fact, an explicit limitation is that resolution is **coarser than CT/MRI** [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- **Speed/cost positioning**: It is framed as **fast** and **low-cost**, suggesting the value proposition is likely accessibility, throughput, or portability rather than top-end image fidelity [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).

There is also technically informed reaction about the likely sensing challenges:

- John Whitaker notes that systems based on **light, ultrasound, electric current, etc.** have a harder inverse problem than X-rays because signals do **not travel in straight lines** in the same way, making reconstruction more complex [@johnowhitaker](https://x.com/johnowhitaker/status/2067406686940463328).
- He also suggests a future version with **many scattered detectors and emitters** rather than mechanically moving components, indicating that at least some readers infer the current system may involve motion/scanning geometry rather than fully parallelized capture [@johnowhitaker](https://x.com/johnowhitaker/status/2067406686940463328).

Taken together, the public discussion points toward a non-CT, non-MRI modality with wave-based reconstruction and meaningful algorithmic/inverse-problem content, though the tweets here do not provide definitive modality labeling or performance tables beyond the stated tradeoffs.


## Different perspectives


### Supportive / optimistic

- The most enthusiastic camp sees this as exactly the kind of **high-upside, weird, non-consensus invention** AI founders should pursue, not just incremental chatbot/UI products. That tone is clear in “let inventors like @DavidSHolz invent” [@saranormous](https://x.com/saranormous/status/2067435669694374217).
- In-person demo reactions emphasized the **visceral novelty** of interacting with a real scanner, not just reading a paper or watching a video [@saranormous](https://x.com/saranormous/status/2067434920134549524).
- Some interpreted the move as a sign that Midjourney may be thinking beyond image generation and toward **full-stack applied invention**, possibly combining hardware, sensing, and AI reconstruction.

### Neutral / technical-curious

- The most grounded reaction in the set is the concise pros/cons summary: **radiation-free, magnet-free, fast, low-cost** versus **water immersion** and **lower resolution than CT/MRI** [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- Technically curious observers liked the strangeness of the modality while immediately identifying the physical and systems tradeoffs:
  - Non-straight-line propagation compared with X-rays
  - Need for better real-time capture arrangements
  - Questions about detector/emitter topology
  [@johnowhitaker](https://x.com/johnowhitaker/status/2067406686940463328)

### Opposing / skeptical / cautionary

Direct hostile criticism is limited in this tweet set, but skepticism is implicit in several points:

- **Clinical utility skepticism**: saying it has **coarser resolution than CT/MRI** is a substantive caveat, especially in medicine where image quality can directly affect diagnostic value [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- **Practicality skepticism**: requiring a **water immersion tank** is a serious ergonomic and deployment constraint for routine clinical or consumer use [@iScienceLuvr](https://x.com/iScienceLuvr/status/2067384062910832812).
- **Modality skepticism**: technical comments about non-straight-line propagation hint at the usual challenge for alternative imaging systems: the physics and inverse reconstruction are hard, and the pretty demo may not automatically translate into robust, clinically reliable imaging [@johnowhitaker](https://x.com/johnowhitaker/status/2067406686940463328).

### Competitive framing

- One notable perspective was less about the scanner itself and more about what it says strategically: if Midjourney is attempting hardware-medical invention, then AI companies pursuing narrower wearable-camera concepts look conservative by comparison [@matvelloso](https://x.com/matvelloso/status/2067446209690869877).


## Context: why this matters


Midjourney is primarily known as an image-generation company. That makes a medical/scanner reveal noteworthy for several reasons:

- It suggests a willingness to move from **generative media software** into **real-world sensing and hardware**.
- Medical imaging is a domain where **inverse problems**, **signal processing**, **reconstruction**, and increasingly **ML-based interpretation** all matter; it is not an obvious adjacency, but it is a technically deep one.
- The scanner appears to be positioned not as “better than MRI/CT on all axes,” but as a potential entrant in the classic disruption lane: **worse on a premium metric, better on cost/accessibility/operational burden**.
- If the system is genuinely fast and low-cost, the most plausible implications are in:
  - screening or triage,
  - settings where CT/MRI access is limited,
  - repeat imaging where avoiding radiation matters,
  - specialized anatomical use-cases where immersion-based setups are acceptable.

The launch also fits a broader 2025 pattern where AI-adjacent companies increasingly try to define themselves not just as model vendors, but as builders of **new interfaces to the physical world**. In that framing, Midjourney Medical is less about a single scanner and more about whether frontier AI-era startups can productize difficult sensing systems, not just generate content.


## Implications and open questions


- **Regulatory path**: nothing in these tweets addresses approvals, validation studies, or whether this is research-only versus intended for clinical deployment. For medical relevance, those questions are central.
- **Reconstruction stack**: the phrase “technical dive” implies the company has discussed internals, but the tweet set here does not expose the actual algorithmic details. The likely crux is reconstruction quality under a constrained sensing setup.
- **Use-case specificity**: lower resolution than CT/MRI does not necessarily doom the system; many imaging tools win by being good enough for a narrow workflow. But no specific target indication appears in these tweets.
- **Form factor challenge**: a water immersion tank is acceptable for some scanning contexts and a major barrier for others. Whether this is a prototype artifact or a fundamental requirement matters.
- **Throughput and cost realism**: “fast” and “low-cost” are meaningful only relative to benchmarks—scan time, hardware cost, consumables, operator burden, and downstream interpretation overhead. Those numbers are not provided in the tweets here.
- **AI’s role**: the most interesting technical question may be whether Midjourney’s contribution is primarily in hardware design, inverse-problem reconstruction, learned denoising/super-resolution, automated interpretation, or an integrated stack spanning all of these. The social reaction suggests people are projecting a lot onto the project because Midjourney’s brand is associated with learned visual systems rather than classical medical devices

**AI research, agents, and open models**


- A notable research meta-point: **Chinese open-source literature** over the last year was highlighted as unusually high-ROI to follow, with the claim that the “alpha is insanely huge” [@himanshustwts](https://x.com/himanshustwts/status/2067095066645266928).
- PapersWithCode’s top trending paper was **VibeThinker-3B**, described as a **3B** parameter model exploring verifiable reasoning in small LMs and allegedly landing in the performance tier of **DeepSeek V3.2, GLM-5, and Gemini 3 Pro** [@NielsRogge](https://x.com/NielsRogge/status/2067169163508908423).
- A computer-use paper, **PreAct**, was praised for compiling successful agent runs into a guarded replayable state machine, eliminating per-step LM calls on repeats and yielding **8.5x to 13x faster** replay [@dair_ai](https://x.com/dair_ai/status/2067386816815387019).
- Another RL/agent paper proposed **LLM-as-Environment-Engineer**, where the policy uses its own failures to redesign the next training environment; the associated benchmark is **MAPF-FrozenLake** [@dair_ai](https://x.com/dair_ai/status/2067432115072098705).
- Omar Sar0 argued coding agents need **verifiers and robust guardrails**, not blind autonomous loops, reinforcing a trend toward constrained agentic execution [@omarsar0](https://x.com/omarsar0/status/2067283235759935648).
- David Khourshid’s coding-agent take was more operational: **AI-generated code still has to be read**, and not reading it simply defers the debugging burden [@DavidKPiano](https://x.com/DavidKPiano/status/2067306799452336429).
- On RL theory, John Schulman said PPO’s resurgence in the LLM era comes from effects not anticipated in the original paper, including the **importance-ratio objective** correcting biases from **numeric error, async training, and forward-pass noise**, while clipping alters entropy via a mechanism only later understood; he cites **DAPO** [@johnschulman2](https://x.com/johnschulman2/status/2067410492008841643).
- Relatedly, Chris Wolfe said recent post-GRPO analysis papers (e.g. **DAPO, Dr. GRPO, GSPO, TIS**) are exactly the kind of objective-analysis work he hopes to see for PPO in reasoning/agent contexts [@cwolferesearch](https://x.com/cwolferesearch/status/2067438991952027664).
- John Carmack posted a detailed critique of **Temporal Differences for visual representation learning**, summarizing the method: train a frame encoder and a “motion encoder” on RGB frame differences so latent(frame1) + delta ≈ latent(frame2), with a **0.25 second stride**; he questioned the DINO EMA anti-collapse choice and the soundness of the delta construction [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2067437937713717609).


**AI infrastructure, inference, and product rollouts**


- Xenova released a demo and kernels from the now-shut-down **Fable 5** effort, claiming it had pushed **Gemma 4 to 255 tok/s on WebGPU**; the framing is that **agentic kernel optimization** could materially improve browser/on-device inference [@xenovacom](https://x.com/xenovacom/status/2067289897111638484).
- Fal announced **Kling 3.0 Turbo and O3 upgrades**:
  - faster generation
  - lower costs
  - better lip-sync
  - more stable motion
  - stronger prompt/reference consistency in “Omni”
  - up to **15s clips**
  - full **4K generation** with Omni
  - improved storyboard and multishot workflows
  [@fal](https://x.com/fal/status/2067194508597440656)
- Kling’s own account amplified the Fal rollout as a creator-facing quality/speed improvement [@Kling_ai](https://x.com/Kling_ai/status/2067430859334910414).
- GitHub Copilot’s **Auto mode** now uses a custom routing model to choose among models based on reasoning depth, code complexity, debugging difficulty, and tool orchestration needs; a blog post and a linked research paper were shared [@pierceboggan](https://x.com/pierceboggan/status/2067449111809597803), [@pierceboggan](https://x.com/pierceboggan/status/2067449245024874564).
- Kimi Code Web appears to be back online, per a brief ecosystem note [@bigeagle_xd](https://x.com/bigeagle_xd/status/2067426798493921351#m).
- Grok image generation projects were mentioned via grok.com/imagine, but with no substantive technical detail [@chaitu](https://x.com/chaitu/status/2067424095613202442).


**Talent, labs, and competitive dynamics**


- The biggest personnel story outside Midjourney: **Noam Shazeer** announced he is joining **OpenAI**, leaving Google after saying it was a difficult decision and praising his former team [@NoamShazeer](https://x.com/NoamShazeer/status/2067400851438932297).
- Sam Altman celebrated the move, saying Noam was one of the people he had most wanted to work with since OpenAI’s beginning [@sama](https://x.com/sama/status/2067427421083652131), then joked about OpenAI being SOTA “in noams” [@sama](https://x.com/sama/status/2067427678529974740).
- Commentary emphasized Shazeer’s significance as co-author of **Transformer, T5, and Switch Transformer** and pioneer of sparse MoE systems, with some calling it the most important AI talent move of the year [@scaling01](https://x.com/scaling01/status/2067407207956705517).
- Aidan Clark signaled excitement about working with Noam and linked it to a sense that RSI is getting closer [@_aidan_clark_](https://x.com/_aidan_clark_/status/2067404249928278065).
- A broader industry reading from replies:
  - DeepMind/Brain merger may have indirectly benefited Anthropic/OpenAI [@_arohan_](https://x.com/_arohan_/status/2067404559358849401)
  - Anthropic got Karpathy while OpenAI got Noam [@TheTuringPost](https://x.com/TheTuringPost/status/2067428112791400621)
  - speculation that the move says as much about Google disappointment as OpenAI pull [@teortaxesTex](https://x.com/teortaxesTex/status/2067416948049142238)
- There was also chatter about relative power/valuation: Liam Fedus posted “Breaking: OpenAI overtakes Anthropic’s valuation” [@LiamFedus](https://x.com/LiamFedus/status/2067415148697116980).
- More opinionated geopolitical/competitive takes argued that various actors have incentives to prevent Anthropic from maintaining too large a lead, though these were clearly speculative rather than factual reporting [@teortaxesTex](https://x.com/teortaxesTex/status/2067415296516731280), [@teortaxesTex](https://x.com/teortaxesTex/status/2067415696984678422).


**Adoption, usage, and model quality discourse**


- Blanche Minerva offered a practical quality complaint: ChatGPT and Claude can disagree on something as concrete as the overlap in citations between two papers, underscoring persistent reliability issues in applied knowledge tasks [@BlancheMinerva](https://x.com/BlancheMinerva/status/2067415969790578878).
- Several posts focused on GLM and Chinese model progress:
  - praise for the **GLM** team as “heroic” [@teortaxesTex](https://x.com/teortaxesTex/status/2067413884588212738)
  - follow-up saying the latest generation reached something like **Opus-level** expectations beyond prior assumptions [@teortaxesTex](https://x.com/teortaxesTex/status/2067416066423152825)
  - speculation that future frontier capability gains may hinge more on RL recipes than pure pretraining scale [@teortaxesTex](https://x.com/teortaxesTex/status/2067446753217970202), [@teortaxesTex](https://x.com/teortaxesTex/status/2067447252327518222)
- There was also a cluster of highly speculative posts about “Claude” identity/persona salience appearing in outputs, framed as memetic or steganographic behavior rather than established fact [@teortaxesTex](https://x.com/teortaxesTex/status/2067421423988076621), [@teortaxesTex](https://x.com/teortaxesTex/status/2067428989505769547), [@teortaxesTex](https://x.com/teortaxesTex/status/2067429728730915018).


**Broader tech and society**


- A Tacit Labs join announcement framed biology as the next place where AI should uncover genuinely new knowledge rather than just recombine existing understanding [@maxisawesome538](https://x.com/maxisawesome538/status/2067373342894531002).
- There was a joke about the White House demanding a solution to the **halting problem**, a reminder that AI-policy discourse still often compresses deep CS impossibilities into simple-sounding asks [@the_engi_nerd](https://x.com/the_engi_nerd/status/2067342394333282610).
- In autonomy, one post noted the apparent lack of fresh AV startup activity despite Waymo/Tesla making the category seem increasingly feasible [@gabriberton](https://x.com/gabriberton/status/2067437813952630899).
- Miscellaneous opinion posts on learning, coding, and contribution included:
  - you can contribute to AI without deep formal math background [@gabriberton](https://x.com/gabriberton/status/2067439158193467579)
  - a token-understanding/generation interview question about whether a model can understand a token it cannot generate [@gabriberton](https://x.com/gabriberton/status/2067438251728900472)
  - a joke that a Slack alternative could be built with “half a day of vibe coding” [@gabriberton](https://x.com/gabriberton/status/2067446594325320050)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. GLM-5.2 Open-Weights Frontier Benchmarks

  - **[GLM-5.2 is the first open-weights model to cross 80% on Terminal-Bench and beats every other open model available](https://www.reddit.com/r/LocalLLaMA/comments/1u7mexd/glm52_is_the_first_openweights_model_to_cross_80/)** (Activity: 1569): **The image is a **technical benchmark bar chart** for **Terminal-Bench 2.1** showing **GLM-5.2** scoring `81.0`, making it the first **open-weights** model in the chart to clear the dashed `80%` threshold, though closed models **Claude Opus 4.8** (`85.0`) and **GPT-5.5** (`84.0`) remain ahead overall ([image](https://i.redd.it/8j43gkgrxo7h1.jpeg)). The post frames this as GLM-5.2 beating other open models and even **Gemini 3.1 Pro**, but a commenter notes Terminal-Bench `2.1` is an “easier” revision of Terminal-Bench 2 with relaxed timeouts/rules, so cross-version score comparisons may be inflated.** Comments debate whether “open weights” meaningfully implies “local” usability: one user argues *“if you can download it, it’s a local model,”* while another says it is still impossible to run locally for `99%` of users due to hardware requirements.

    - A commenter argues that **Terminal-Bench 2.1 is not directly comparable to Terminal-Bench 2**, claiming 2.1 is an easier revision with changed timeouts, relaxed problem rules, and broader harness compatibility. They note that models generally should not score lower on 2.1 than 2, and suggest the more meaningful signal will be **initial Terminal-Bench 3 scores** before labs start optimizing against the benchmark.
    - There is a technical deployment debate around whether GLM-5.2 should be considered a “local model.” One side argues that *“if you can download it, it’s a local model”* because unlike Claude or ChatGPT the weights can be run by users, while another points out that the model is effectively impossible to run locally for `99%` of users due to hardware/performance constraints such as very low tokens-per-second on consumer systems.

  - **[GLM-5.2 is a win for local AI](https://www.reddit.com/r/LocalLLaMA/comments/1u8ai2a/glm52_is_a_win_for_local_ai/)** (Activity: 1270): **The post argues that **GLM-5.2**, described as an **MIT-licensed MoE coding/reasoning model** with `753B` total parameters and ~`40B` active parameters/token, is significant less as a home-runnable model than as a source for **distillation/synthetic-data fine-tuning** into `8B`/`70B` local models. The OP’s estimated deployment table puts memory at roughly `744–890 GB` for FP8, `476–500 GB` for 4-bit, `241–280 GB` for 2-bit, and `176–180 GB` for 1-bit dynamic quantization, with KV cache overhead for the claimed `1M` context adding ~`150–200 GB` at FP16/BF16 or ~`35–50 GB` at INT4; they explicitly caveat that these numbers were gathered online and AI-assisted.** Commenters debated “local” feasibility: some noted that `512GB` Macs, GB10 clusters, or multiple `128GB` AMD AI Max-class systems could plausibly run low-bit variants, while others framed the hardware requirements as increasingly unobtainable. One API user called GLM-5.2 a *“very, very, very killer model”* and argued that GLM-5.2, MiniMax M3/Mini-V2.5-Pro, and similar open-weight/API-accessible models have largely closed the practical gap with proprietary frontier models; another commenter simply wished for a distilled or native `70B` dense release.

    - Several commenters focused on **local inference feasibility** for GLM-5.2, noting that practical setups likely require very high-memory systems such as `512GB` Macs, GB10-style clusters, multiple `AMD AI MAX 128GB` machines, or a custom multi-GPU server. One user estimated they could run the **GGUF** locally on a server built for `<$9,000`, but expected only around `~7 TPS`, framing it as usable but expensive home deployment.
    - A technical concern was raised about **Mac Studio performance at long context lengths**: one commenter argued that while the model may technically run, it becomes “unusable” at `50K+` context because of poor **PP/TG** throughput. The point was that memory capacity alone is insufficient; prompt processing and token generation speed dominate usability for large-context inference.
    - A user with API experience claimed **GLM-5.2**, alongside **MiniMax M3 / Mimi-V2.5-Pro**, significantly narrows the gap between open-weight/open-ish large models and frontier proprietary models. They specifically said they would trust GLM-5.2 more than **Opus 4.8** in some cases, while acknowledging that there remain “frontier problems” these models still cannot solve.

  - **[zai-org/GLM-5.2 is here!](https://www.reddit.com/r/LocalLLaMA/comments/1u7kcwf/zaiorgglm52_is_here/)** (Activity: 1178): ****Z.ai released [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2)**, an MIT-licensed flagship long-context model with a stable **`1M` token context**, stronger coding/agentic performance, configurable reasoning effort, and serving support across **SGLang, vLLM, Transformers, KTransformers, and Ascend NPU**. Key implementation changes include **IndexShare** sparse-attention indexer reuse, claiming **`2.9×` lower per-token FLOPs** at `1M` context, plus improved MTP speculative decoding with up to **`20%` longer acceptance**; commenters highlighted a reported **DeepSWE score of `46.2`**, placing it above Claude Opus 4.6/Sonnet and just below 4.7 in that benchmark.** Commenters were mainly interested in missing/expected variants and quantization, asking for **GLM-5.2-Flash-32B-A4B** and jokingly/seriously waiting for ultra-low-bit **`0.5Q`** releases.

    - Commenters highlighted that **GLM-5.2** is very large on Hugging Face, with the linked [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2/tree/main) repository showing roughly **`1.51 TB`** of model files, making local inference impractical for many users without heavy quantization or multi-GPU setups.
    - One commenter cited a **self-reported DeepSWE score of `46.2`**, claiming it places GLM-5.2 above **Claude Opus 4.6** and **Claude Sonnet**, and just below **Opus 4.7**, suggesting strong software-engineering benchmark performance if independently validated.
    - There was interest in a smaller or more deployable variant, specifically **`GLM-5.2-Flash-32b-a4b`**, implying demand for a lower-cost MoE/Flash-style release that could be easier to run than the full `1.51 TB` checkpoint.

  - **[GLM-5.2 is now 1st on Design Arena — ahead of the now unavailable Claude Fable 5.](https://www.reddit.com/r/LocalLLaMA/comments/1u7qti8/glm52_is_now_1st_on_design_arena_ahead_of_the_now/)** (Activity: 751): **The image is a **Design Arena leaderboard screenshot** ([image](https://i.redd.it/ffwqkl3mqp7h1.png)) showing **GLM-5.2** ranked **#1** in the “Code Categories Arena” with an Elo score of `1360`, narrowly ahead of the now-unavailable **Claude Fable 5** at `1350`. Context from the linked tweet/post frames this as notable because GLM-5.2 appears to overtake a recently removed Anthropic model, though the margin is small and leaderboard rankings may shift as more votes accumulate.** Commenters were cautiously interested but skeptical, with one noting it is *“bit early to make this call”* and suggesting the ranking needs time to stabilize. Others focused on geopolitical/model-access implications, joking or warning that if powerful U.S. models are restricted, open or Chinese alternatives like GLM-5.2 may quickly fill the gap.

    - A commenter cautioned that **GLM-5.2’s Design Arena rank may be premature**, noting that arena scores can shift as more votes accumulate: *“give it a few days to settle out.”* This is a useful caveat for interpreting early leaderboard positions, especially when comparing against unavailable models like **Claude Fable 5**.
    - One technical concern raised was how a **text-only model** can perform well in real-world design workflows, where outputs often require visual inspection and iterative feedback loops. The commenter suggested that such workflows may need an **OCR or vision model** to evaluate generated designs, and asked how the vision-capable **Kimi K2.7** performs on the same benchmark, noting that **Kimi K2.6** was already their preferred design model.

  - **[PSA: unsloth/GLM-5.2-GGUF is uploading](https://www.reddit.com/r/LocalLLaMA/comments/1u8ikje/psa_unslothglm52gguf_is_uploading/)** (Activity: 491): **A Reddit user noticed the Hugging Face repo [`unsloth/GLM-5.2-GGUF`](https://huggingface.co/unsloth/GLM-5.2-GGUF) was newly created and inferred that **Unsloth** was likely preparing/uploading **GGUF quantizations** for **GLM-5.2**; at the time, the repo reportedly only contained a README. The linked HF page was not accessible during fetch due to **HTTP `429 Too Many Requests`**, with Hugging Face recommending authentication via `HF_TOKEN` for API access.** Top comments focused on deployment practicality: users questioned what quantization level would be required to fit the model locally, joked about needing cloud GPUs, and implied current consumer hardware may be insufficient for comfortable inference.

    - Commenters focused on the deployment feasibility of **unsloth/GLM-5.2-GGUF**, with one noting the apparent `800GB` footprint and asking how aggressively it would need to be quantized to run locally. Another technical concern was KV-cache scaling for very long context: *“imagine the KV Cache size to reach 1M CTX”*, implying that even if the GGUF weights fit, `1M` context inference would require substantial additional memory beyond model weights.


### 2. Local Inference Optimization: WebGPU and AMD ROCm

  - **[Gemma 4 E2B running in-browser at 255 tok/s using WebGPU kernels written by Fable 5](https://www.reddit.com/r/LocalLLaMA/comments/1u8g3d0/gemma_4_e2b_running_inbrowser_at_255_toks_using/)** (Activity: 503): **A WebGPU in-browser demo for **Google Gemma 4 E2B IT QAT mobile transformers** was released on Hugging Face, including optimized kernels reportedly generated/optimized with **Fable 5** before shutdown, reaching about `255 tok/s` on an **Apple M4 Max**. The demo/kernels are available at [HF Spaces](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels), with the model hosted at [google/gemma-4-E2B-it-qat-mobile-transformers](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers).** Comments noted strong interest in open-sourcing the UI and lack of Firefox support, implying the demo likely depends on browser WebGPU support/compatibility. One commenter pointed to a related Hugging Face optimization effort for Gemma E4B on an **A10G**, claiming roughly `500 TPS` with *“no quality loss”* via collaborative agent-driven inference optimization: [dashboard](https://gemma-challenge-gemma-dashboard.hf.space/).

    - A commenter linked a related Hugging Face optimization effort where collaborating agents are reportedly maximizing **Gemma E4B inference** on an **A10G GPU**, reaching about `500 TPS` with claimed *“no quality loss”*: https://gemma-challenge-gemma-dashboard.hf.space/. This provides a useful comparison point against the post’s **in-browser WebGPU** result of `255 tok/s`, though the hardware and runtime environments differ substantially.
    - Several technical questions focused on runtime portability and deployment tradeoffs: one user noted the lack of **Firefox** support, likely due to WebGPU/browser compatibility constraints, while another asked how the WebGPU/Fable 5 implementation compares against native runtimes such as **llama.cpp**. Another raised a practical browser-storage concern after downloading roughly `2 GB` of model data and wanting a way to flush/delete it afterward.

  - **[Avoid CUDA monopoly at all costs. AMD is an alternative.](https://www.reddit.com/r/LocalLLM/comments/1u7pmzi/avoid_cuda_monopoly_at_all_costs_amd_is_an/)** (Activity: 458): **The post reports running `llama.cpp`/`llama-server` on an **AMD RX 7800 XT 16GB** with **ROCm 6.4.4**, compiled via `-DGGML_HIP=ON -DGPU_TARGETS=gfx1101 -DrocWMMA_FATTN=ON`, serving `Qwopus3.6-27B-v2-Q3_K_S.gguf` and `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` at up to `131072` context using `--flash-attn on`, full GPU offload, and split KV-cache quantization (`K=q8_0`, `V=q4_0`). The author claims the KV quantization reduces cache memory by ~`5.6x`, keeping weights + `128K` KV cache within ~`96%` of VRAM with no CPU spill, while achieving ~`210 tok/s` prefill and `11–17 tok/s` decode at ~`188W`; they attribute long-context coherence to YaRN RoPE scaling and provide a longer benchmark write-up [here](https://sergiiob.dev/posts/rx7800-xt-llama-cpp-benchmarks-moe-context).**



### 3. Local Coding Agents and Distillation Caveats

  - **[Be wary of Qwen/Claude distillations - they're often worse than the base model](https://www.reddit.com/r/LocalLLaMA/comments/1u7a2hn/be_wary_of_qwenclaude_distillations_theyre_often/)** (Activity: 554): **The post argues that recent **Qwen/Claude distill/finetune models** such as “Qwopus” or Qwen-based Claude distillations trained on only `~4k–10k` teacher samples are unlikely to transfer meaningful capability and may degrade the base **Qwen 3.6** model, mostly changing style rather than improving reasoning/knowledge. It contrasts these with **DeepSeek-R1** official LLaMA/Qwen distills, which reportedly used `~700k` samples—large enough to affect behavior and benchmarks—and cites an external test where a Claude-distilled Qwen variant hallucinated relative to base Qwen and ran slower: [“Claude distillation doesn’t transfer library knowledge”](https://akitaonrails.com/en/2026/04/24/llm-benchmarks-parte-3-deepseek-kimi-mimo/#the-discovery-claude-distillation-doesnt-transfer-library-knowledge).** Commenters broadly agreed, with one claiming capability-improving finetuning now generally needs `>100k` carefully curated examples plus recovery methods such as **GRPO**, not a few thousand samples. Another commenter suggested skepticism toward model cards with weak evals—low `N`, only `pass@5`, narrow web-dev benchmarks, or undisclosed distillation—though much of that comment was heuristic/opinion rather than evidence.

    - Several commenters argued that small supervised “distillations” from Qwen/Claude outputs are unlikely to improve base models: `4k` samples was described as “basically nothing,” and one commenter said meaningful improvement fine-tuning now needs `100k+` carefully curated examples plus recovery methods like **GRPO** rather than a few thousand prompt/response pairs.
    - A technical objection was that most API-based distillation lacks critical training signal: users typically do not get full logits beyond small top-N/top-1 outputs, and **Anthropic** does not expose full chain-of-thought, only summaries. This makes many releases closer to *partial-response supervised fine-tunes* than true knowledge distillation, losing substantial information from the teacher model.
    - One commenter gave an applied fine-tuning datapoint: even with `18k` examples for a focused **GDScript** domain model, including docs pretraining and personal code, the model still failed to reliably produce desired outputs. Their conclusion was that fine-tuning can improve domain behavior/vertical specialization but “does **NOT** add intelligence.”

  - **[Headless screenshot loops let a local 30B agent finish a raytraced FPS demo in pure C](https://www.reddit.com/r/LocalLLaMA/comments/1u89f2q/headless_screenshot_loops_let_a_local_30b_agent/)** (Activity: 277): **OP reports that adding a **headless instrumentation requirement**—keyboard/mouse input injection plus deterministic screenshot capture at selected frames—let both **Claude Code on Opus 4.8** and a local **Qwen3.6 27B** agent iteratively debug and complete a small **pure-C, standard-library-only raytraced FPS demo**. The key mechanism was self-directed visual feedback: the agent timed captures around events such as rocket impacts, inspected rendered particles/debris, patched the C code, and reran the binary, effectively forming a recursive screenshot-based debugging loop. OP frames this as a prompting/tooling result rather than a model-quality benchmark, and discloses the local agent is their own OSS project, [`codehamr`](https://github.com/codehamr/codehamr).** Commenters were mostly impressed by the local Qwen result and nostalgic about C/demoscene-era development, though one commenter argued the task is likely not very challenging for current models.

    - One commenter described an agent harness built around a custom Python `Log` function that mirrors `print` but can redirect all output into a shared log file. The model is explicitly instructed to inspect log tails, add internal logging, and use those observations for iterative debugging, effectively closing the observe-debug-fix loop that models “don’t do out of the box.”
    - A user reported running the setup on an **RTX 4090**, noting a clear speed improvement and identifying `q4_k_m` quantization as their preferred quality/performance tradeoff for local inference.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code Security and Workflow Lessons

  - **[Claude Opus caught malware hidden in my repo, then reverse engineered the whole thing](https://www.reddit.com/r/ClaudeAI/comments/1u7vqst/claude_opus_caught_malware_hidden_in_my_repo_then/)** (Activity: 1207): **A repo owner says **Claude Code running Opus** halted a git merge after detecting an obfuscated block appended after `module.exports` in `next.config.js`, identifying it statically as an **EtherHiding-style loader** before CI/build execution. The described chain uses git credential theft/self-propagation via a compromised contractor machine, forged commit metadata, blockchain dead drops via `api.trongrid.io`, Aptos, and BSC RPCs, XOR-decoded in-memory stages, and an infostealer C2 at [`198.105.127.210`](http://198.105.127.210) over ports `80/443`; listed IOCs include dropper SHA-256 `e27abe7e810c79d71e8c1681ccd010d7ddbda6a9a34bf1124ba392a36ba9b476`, globals like `global.i` / `global._V = "8-4827"`, and multiple TRON/Aptos/BSC transaction pointers. Recommended checks are to audit `next.config.js`, `postcss.config.js`, and other build-time config for appended code, monitor CI egress to blockchain RPC endpoints, rotate all secrets reachable from builds, and treat the pushing workstation as compromised.** The substantive comment thread emphasized that the key security lesson is not merely “Claude caught it,” but that framework config such as `next.config.js` is privileged build-time code executed in CI and should be tightly controlled, reviewed, and sandboxed. Other top comments were mostly jokes or off-topic digs at Next.js / model restrictions.

    - A commenter emphasized that the key security failure was not model detection but that a committed `next.config.js` executes during every build/CI run: *"one committed config ... runs on every build + CI"*. The technical takeaway was to restrict what code can execute at build time, since a malicious config only needs to land in the repo once to compromise the pipeline.
    - Another commenter identified repository controls as the first failure point: allowing someone to force-push into the repo bypasses normal review and auditability. They argued this should be prevented via branch protection, required reviews, and disabling force pushes on protected branches.
    - One commenter recommended scanning all repositories for **Hades/Miasma**-style supply-chain compromises, noting these can propagate through commonly used libraries rather than only from already-infected developer machines. They also warned the issue is not limited to Node projects and suggested checking all language ecosystems in use.

  - **[Pro Tip - Reset your usage limits on your schedule](https://www.reddit.com/r/ClaudeAI/comments/1u7i5ow/pro_tip_reset_your_usage_limits_on_your_schedule/)** (Activity: 1342): **The post describes a scheduling workaround for **Claude Code** usage windows: create a daily [Claude Code Routine](https://claude.ai/code/routines) using **Haiku** that sends a trivial prompt (e.g. `"hello"`) roughly `5` hours before the desired reset time, thereby starting the rolling session window earlier. The claimed effect is that a user who begins work at `9:00` can force the first reset around `12:30` instead of `14:00+`, assuming no prior active session prevents a fresh window; Anthropic’s routines feature is documented [here](https://code.claude.com/docs/en/routines).** Commenters generally agreed the tactic is useful mainly for users who frequently hit Pro/5x limits, because it lets them burn higher-capability models or high-token plugins shortly before a scheduled refresh. One commenter reported using a cron-like refresh every `5` hours to maximize token availability, while noting OP’s tighter timing may better preserve a usable morning window.

    - Users describe exploiting Claude’s rolling `5-hour` usage window by deliberately starting a session earlier than actual work time. For example, triggering usage at `7AM` can move the next reset to `12PM`, so if heavy work begins after a `10AM` standup, the user waits until noon instead of `3PM` after exhausting limits.
    - One commenter reports using a `cron` job to refresh the usage window every `5 hours` to maximize token availability, then intentionally burning remaining quota on the highest-capacity model, Claude Design, and other high-token tools before the reset. They also mention using “caveman mode” and a “rust token killer” as token-reduction techniques, though no implementation details or benchmarks are provided.
    - Another user configured scheduled tasks in `cowork` at `7AM`, `12PM`, and `5PM` to align Claude usage resets with waking/work hours, effectively creating multiple full sessions during the day. They note Claude itself pushed back against automating this and instead recommended reducing token waste, highlighting a tension between quota-window optimization and prompt/token-efficiency practices.

  - **[the gap between Claude Code power users and us chat-only people keeps getting wider and i don't think that's great for the community](https://www.reddit.com/r/ClaudeAI/comments/1u85myc/the_gap_between_claude_code_power_users_and_us/)** (Activity: 2348): **A Pro chat-only Claude user argues the subreddit’s technical focus has shifted heavily toward **Claude Code** workflows (`CLAUDE.md`, MCP, subagents, terminal usage), making non-coding use cases—writing, thinking, learning, planning—feel underrepresented despite likely being a large user cohort. Top replies suggest that **Claude Code can be used as a general local-task agent** without programming, e.g. chatting in the terminal to transform local data into Excel/PDF outputs, and mention **Cowork** as an intermediate option with increased usage limits.** Commenters largely agree that coding dominates the community—one estimates `95%` of posts are coding-related—but differ on the remedy: some encourage chat-only users to adopt Claude Code-style tools for broader automation, while others ask what concrete non-coding workflows people want discussed.

    - Several commenters argued that **Claude Code’s advantage is not coding-specific**, but comes from its CLI/tool environment: local file access, command execution, and the ability to perform concrete tasks like converting local data into formatted Excel files and exporting polished PDFs. The technical distinction raised is that Claude Code behaves more like an agent with filesystem/tool access, while browser chat remains a more limited conversational interface.
    - A recurring technical complaint was that **Claude’s most powerful workflows require nontrivial setup**: MCP servers, package installation, and manual JSON configuration. One commenter argued that for non-coders, installing “whatever the hell an MCP server is” should be a one-click operation, because the current friction keeps advanced Claude workflows inaccessible to regular users.
    - A power-user example framed `CLAUDE.md`, MCPs, subagents, and terminal workflows as **general-purpose knowledge-work infrastructure**, not software-development features. In an investment workflow, each deal can have its own [`CLAUDE.md`](http://CLAUDE.md), MCP-connected data sources, subagents to process due-diligence reports, and documented workflows for building financial models and slide decks.


### 2. Anthropic Fable Access and Policy Pressure

  - **[They're demanding Fable to somehow be 100% jailbreak-proof. It's so fucking over.](https://www.reddit.com/r/ClaudeAI/comments/1u8nalg/theyre_demanding_fable_to_somehow_be_100/)** (Activity: 1375): **The image is a screenshot of a WIRED article preview claiming **Trump administration officials want Anthropic’s “Fable 5” blocked from all jailbreaks before release**, while security experts argue that **100% jailbreak resistance is not technically achievable** for current LLMs. The technical issue is the impossibility of proving total safety for a generative model exposed to arbitrary prompts, tools, and contexts; the post frames this as an unrealistic security requirement akin to demanding an OS or car be mathematically incapable of harm. [Image](https://i.redd.it/tyrnlpaivw7h1.png)** Commenters largely view the demand as absurd or politically motivated, with analogies to requiring cars to cause zero injuries or operating systems to be unhackable. One commenter speculates the requirement may be intended to restrict public release while preserving government access.

    - Commenters argued that requiring **Fable** to be `100%` jailbreak-proof is technically equivalent to demanding an OS or general-purpose computing platform be proven unhackable: the attack surface and prompt/input space are effectively open-ended. One point emphasized the formal-security issue that *“you can't prove [a] negative”* for all possible jailbreaks, making absolute jailbreak immunity an impractical certification target rather than an engineering requirement.

  - **[Anthropic CEO Dario Amodei joins top AI CEOs meeting with world leaders at G7 summit](https://www.reddit.com/r/ClaudeAI/comments/1u8g4rn/anthropic_ceo_dario_amodei_joins_top_ai_ceos/)** (Activity: 1812): ****Anthropic CEO Dario Amodei** and **OpenAI CEO Sam Altman** reportedly attended a **G7 working lunch on AI** with world leaders, amid geopolitical tension from a **U.S. restriction limiting allied access to Anthropic’s most advanced models**. The available Reddit/video source could not be independently inspected because Reddit returned `403 Forbidden`, so no additional technical detail on the policy scope, affected model tiers, or export-control mechanism is available.** Top comments were largely non-technical jokes/reactions; the only substantive question was why **Salesforce CEO Marc Benioff** was present at the AI leadership meeting.



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
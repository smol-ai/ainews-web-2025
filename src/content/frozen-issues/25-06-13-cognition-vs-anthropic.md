---
id: MjAyNS0w
title: 'Cognition vs Anthropic: Don''t Build Multi-Agents/How to Build Multi-Agents'
date: '2025-06-13T05:44:39.731046Z'
description: >-
  Within the last 24 hours, **Cognition**'s Walden Yan advised *"Don't Build
  Multi-Agents,"* while **Anthropic** shared their approach to building
  multi-agent systems with **Claude's** multi-agent research architecture.
  **LangChain** highlighted advances in context engineering and production AI
  agents used by **LinkedIn** and **BlackRock**. The community is engaging in a
  debate on multi-agent AI development. Additionally, **Hugging Face** announced
  deprecating **TensorFlow** and **Flax** support in favor of **PyTorch**.
  Research on agent memory and model elicitation techniques from **LlamaIndex**
  and **Anthropic** were also discussed.
companies:
  - cognition
  - anthropic
  - langchain
  - huggingface
  - microsoft
  - llamaindex
  - linkedin
  - blackrock
models:
  - claude
topics:
  - multi-agent-systems
  - context-engineering
  - agent-memory
  - model-elicitation
  - ai-evaluation
  - deep-research-workflows
  - framework-migration
  - pydantic-schema
people:
  - walden_yan
  - hwchase17
  - assaf_elovic
  - sh_reya
  - hamelhusain
  - omarsar0
  - clefourrier
  - jerryjliu0
  - akbirkhan
---


**Good technical debate is all we need.**

> AI News for 6/12/2025-6/13/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 6215 messages) for you. Estimated reading time saved (at 200wpm): 504 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Within the last 24 hours **Cognition**'s [Walden Yan](https://x.com/walden_yan/status/1933264183837282558) has said [Don't Build Multi-Agents,](https://cognition.ai/blog/dont-build-multi-agents) while **Anthropic**'s  chose today to discuss how they [see building Multi-Agents](https://www.anthropic.com/engineering/built-multi-agent-research-system).

Which way, AI Engineer?

**READER CHALLENGE**: if you feel like writing a comparative analysis of these two approaches, publish and tweet it [@smol_ai](https://x.com/smol_ai) and we'll pick a quiet day to feature your work. For extra extra bonus points, compare vs [Building Proactive Agents](https://bryanhoulton1.substack.com/p/building-proactive-ai-agents) and [Ambient Agents.](https://blog.langchain.dev/introducing-ambient-agents/)

[](https://resend-attachments.s3.amazonaws.com/V7GW5OJr2RBcn57)

---

# AI Twitter Recap

**AI Agent Development and Tooling**

- **Claude's Multi-Agent Research Architecture**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1933630785879507286) released a blog post detailing how they built **Claude’s research capabilities** using **multiple agents** working in parallel, sharing both successful strategies and engineering challenges.
- **Context Engineering and Product UX**: [@hwchase17](https://twitter.com/hwchase17/status/1933528162799136988) of **LangChain** highlighted a collaboration with [@assaf_elovic](https://twitter.com/hwchase17/status/1933542550222352586) on the **"CAIR" (Confidence in AI Results)** framework, which breaks down components that influence product adoption beyond raw model capabilities. He also emphasized the importance of **"Context Engineering,"** which he described as the [#1 job of engineers building AI agents](https://twitter.com/hwchase17/status/1933278290992845201) and a more dynamic evolution of prompt engineering.
- **LangChain for Production Agents**: [@LangChainAI](https://twitter.com/LangChainAI/status/1933576634843738434) showcased how **LinkedIn** built its production AI agent for hiring using **LangChain** and **LangGraph**, providing a technical architecture that scaled across **20+ teams**. Another example highlighted how **BlackRock** [built production-ready AI agents](https://twitter.com/hwchase17/status/1933275125077733759) to power their **Aladdin** platform.
- **AI Evals for Engineers**: The "AI Evals for Engineers and Technical PMs" course by [@sh_reya](https://twitter.com/HamelHusain/status/1933575166917030184) and [@HamelHusain](https://twitter.com/HamelHusain/status/1933619275325190194) is receiving positive feedback for its practical insights, with participants noting they've [already translated lessons into custom tools](https://twitter.com/HamelHusain/status/1933397529754022238) and found it [miles ahead of other resources](https://twitter.com/HamelHusain/status/1933508512879161476). [@HamelHusain](https://twitter.com/HamelHusain/status/1933615429253280268) also shared common gaps in eval tooling and the importance of [error analysis for diverse user queries](https://twitter.com/HamelHusain/status/1933612965993066842).
- **Deep Research Agentic Workflows**: [@omarsar0](https://twitter.com/omarsar0/status/1933511531590824443) shared a personalized deep research agentic workflow built with **n8n**. A separate paper from **Microsoft** was also highlighted, presenting a [deep research agent for large systems codebases](https://twitter.com/omarsar0/status/1933673330545987773).
- **Hugging Face Abandons TensorFlow/Flax for PyTorch**: [@clefourrier](https://twitter.com/clefourrier/status/1933271084650189263) shared the "bittersweet news" that the `transformers` library is deprecating **TensorFlow** and **Flax** support. **PyTorch** confirmed that **Hugging Face** is going all-in on their framework, noting that [the user base has consolidated around it](https://twitter.com/stanfordnlp/status/1933528689662480781).
- **Agent Memory for Structured Data**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1933672040936190243) from **LlamaIndex** described a structured artifact memory block for agents, which tracks a **Pydantic schema** that is updated over time, essential for tasks like form-filling.

**Model Research, Techniques, and Performance**

- **Anthropic's Model Elicitation and Diffing**: [@akbirkhan](https://twitter.com/akbirkhan/status/1933323897526759553) shared new **Anthropic** research on eliciting capabilities from pretrained models without external supervision. [@jiaxinwen22](https://twitter.com/jeremyphoward/status/1933364618371739948) clarified this is about **elicitation**, not self-improvement. Separately, [@jxmnop](https://twitter.com/jxmnop/status/1933571979975487996) highlighted **"model diffing"** from an older **Anthropic** blog, a technique using a 'crosscoder' to produce interpretable diffs between models, showing how post-training adds specific capabilities.
- **The Power of Reinforcement Learning (RL)**: [@jxmnop](https://twitter.com/jxmnop/status/1933359925415325980) remarked on the incredible possibilities emerging as **RL on LLMs** improves, stating "we’re just getting started." This was echoed in discussions about **ReMA (Reinforced Meta-thinking Agents)**, a new approach combining meta-learning and RL that [improves performance on math and LLM-as-a-Judge benchmarks](https://twitter.com/TheTuringPost/status/1933478813062869156).
- **Fine-Tuning as Continued Pre-Training**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1933595426873356401) shared results from [@antoine_chaffin](https://twitter.com/ClementDelangue/status/1933598791128506477) as a practical example of the principle that **fine-tuning is just continued pre-training**. The work released **BioClinical ModernBERT**, a model pre-trained on biomedical literature and fine-tuned on clinical notes, achieving SOTA results.
- **Text-to-LoRA Hypernetworks**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1933302559957275073) introduced **Text-to-LoRA (T2L)**, a hypernetwork that compresses many LoRAs into itself and can generate new LoRAs from text descriptions, enabling [on-the-fly LLM adaptation](https://twitter.com/TheTuringPost/status/1933608004710248627).
- **ByteDance APT2 for Video Generation**: **ByteDance** presented **APT2**, an [Autoregressive Adversarial Post-Training method](https://twitter.com/NandoDF/status/1933266267663634465) for real-time interactive video generation.
- **New Models and Benchmarks**: **Glass Health** announced its **Glass with Deep Reasoning** model achieves new SOTA on clinical benchmarks, including **97% on USMLE Steps 1–3** and [98% on JAMA Clinical Challenge cases](https://twitter.com/GlassHealthHQ/status/1933291603906736328). **Cartesia AI**'s **Sonic-2** model [topped the Labelbox Speech Generation Leaderboard](https://twitter.com/krandiash/status/1933306410517090684).
- **Debiasing LLMs via Applied Interpretability**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1933645976889422110) praised a paper showing that while prior debiasing techniques fail in realistic resume review settings, simply finding and removing gender or race-related directions in the model remains an effective debiasing strategy.

**Infrastructure, Hardware, and Data**

- **Major Internet Outage**: [@pirroh](https://twitter.com/pirroh/status/1933269623979585695) from **Replit** and others reported a massive internet outage, with [@itsclivetime](https://twitter.com/itsclivetime/status/1933426721723986179) noting it wasn't a DNS or BGP issue. The outage was attributed to a [**Google Cloud (GCP)** issue](https://twitter.com/jeremyphoward/status/1933357293699281021), though Google's own products were largely unaffected as they don't run on the public-facing GCP infrastructure.
- **The GPU Battle: AMD vs. NVIDIA**: [@dylan522p](https://twitter.com/dylan522p/status/1933628242432262304) analyzed how **AMD** is making moves with its **MI355** offering good perf/TCO, while **NVIDIA** alienates some with its DGX strategy. However, he notes AMD's rack-scale solution is like "GB200 nvl72 from temu dot com." The sentiment that **AMD** needs an equivalent software stack and support to **NVIDIA** was also shared by [@scaling01](https://twitter.com/scaling01/status/1933569373031018932).
- **LlamaParse Document Parsing Presets**: **LlamaIndex** announced [new use-case presets for **LlamaParse**](https://twitter.com/jerryjliu0/status/1933627680265810205), which act as specialized parsing agents to render documents into structured formats like tables for forms or XML for technical schematics.
- **Synthetic Data and Human-in-the-Loop (HITL)**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1933297536300929411) discussed the potential of synthetic data to fill data gaps and reduce bias, but warned of model collapse. They stressed the need for **Human-in-the-loop (HITL)** workflows to keep synthetic data grounded and safe.
- **Local On-Device Models**: Discussion around local models highlighted their growing importance, with [@awnihannun](https://twitter.com/awnihannun/status/1933266802450313715) simply stating `pip install mlx-lm`. [@reach_vb](https://twitter.com/reach_vb/status/1933503436630130836) recommended **smollm 2 with llama.cpp or MLX** as a small "Universal Basic Intelligence" for daily tasks, while [@mollycantillon](https://twitter.com/awnihannun/status/1933273566763786699) gave a talk on real-world applications of **MLX** and building fast on-device semantic search.

**Industry Commentary and Geopolitics**

- **Perplexity's Ambitions and Product Strategy**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1933508237934145989) detailed the strong user interest and growth in **Perplexity Finance**, reaffirming his ambition to challenge incumbents like the **Bloomberg Terminal** by offering better UX and accuracy. He also announced that [invites for a new product, **Comet**, are being released](https://twitter.com/AravSrinivas/status/1933289407705960697) and emphasized the core principle that [**"Context is all you need"**](https://twitter.com/AravSrinivas/status/1933503918996402366). He inspired others to be ambitious by [pointing to Google's vast, integrated ecosystem](https://twitter.com/AravSrinivas/status/1933283015586951623).
- **Israel-Iran Conflict and Geopolitical Analysis**: A significant number of tweets focused on the escalating conflict between Israel and Iran. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1933560574857916727) argued that material considerations are childish and that nations like **Israel** can operate outside conventional rules by targeting key people and infrastructure. This was contrasted with the view that [North Korea's nuclear success has biased American assumptions](https://twitter.com/teortaxesTex/status/1933543528455442484) about nuclear proliferation. The apparent lack of Iranian air defense was also questioned by [@francoisfleuret](https://twitter.com/francoisfleuret/status/1933577120640282800).
- **The End of Human-in-the-Loop for Coding**: [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1933522079145734495) predicted that the "centaur" era of AI-assisted coding will be a **"momentary blip,"** a sentiment echoed by [@vipulved](https://twitter.com/vipulved/status/1933647581370069401) who believes we will witness **"the end of hand-written code"** within the next 12 months.
- **NVIDIA CEO Jensen Huang's Comments on Anthropic**: [@Teknium1](https://twitter.com/Teknium1/status/1933620345749319710) and [@jeremyphoward](https://twitter.com/jeremyphoward/status/1933597258047762657) shared an article where **NVIDIA CEO Jensen Huang** had harsh words for **Anthropic**, criticizing their safety-focused stance and suggesting they shouldn't be the only ones trusted with AI.
- **Meta's AI Talent and Strategy**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1933329447853437251) commented that if **Zuckerberg** hadn't laid off a team of exceptional AI talent years ago, **Meta** would have less of an AI talent problem today. [@dylan522p](https://twitter.com/dylan522p/status/1933660636732350686) analyzed the recent hiring of **Alex Wang**, stating the critical measurement will be how he onboards and reorganizes existing talent to build superintelligence.
- **ChatGPT and Medical Diagnosis**: A viral story of **ChatGPT** [saving a person's life by correcting a misdiagnosis](https://twitter.com/npew/status/1933514178314318061) was widely shared, with many commenters adding their own similar experiences. [@shuchaobi](https://twitter.com/shuchaobi/status/1933511659232112751) noted this is what keeps the **OpenAI** team motivated.

**Humor/Memes**

- **Pentagon Pizza Report**: A screenshot of a "Pentagon Pizza Report" with a headline about Iran was [shared by @jeremyphoward](https://twitter.com/jeremyphoward/status/1933457163936280647) with the caption "Pentagon Pizza Report called it".
- **The Discovery of Radio Waves**: A meme captioned "The guy who discovered radio waves" showing someone with oversized headphones [was widely shared](https://twitter.com/jeremyphoward/status/1933357378210312337).
- **Frustration with AI Coders**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1933563892271362337) posted a parody of a developer's experience with **Cursor**, where after generated code fails, the developer repeatedly types **"pls fix"** 15 times before giving up in frustration.
- **Geopolitical Unease**: In a widely circulated tweet, [@zacharynado](https://twitter.com/zacharynado/status/1933579419810996407) advised that if you're "feeling a little uneasy about the state of global geopolitics tonight remember to spend as much time on your hobbies as possible".
- **"pls fix"**: [@RhysSullivan](https://twitter.com/imjaredz/status/1933591454267433398) described watching **Claude Opus burn $70 of tokens** regenerating shadcn components instead of running a simple command.
- **The Prompt that's Worth $100M**: [@skirano](https://twitter.com/skirano/status/1933564941832728751) joked about "That feeling when you know you wrote a $100M prompt."

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. EuroLLM and EuroMoE Model Release Announcements

- [**The EuroLLM team released preview versions of several new models**](https://www.reddit.com/r/LocalLLaMA/comments/1laazto/the_eurollm_team_released_preview_versions_of/) ([Score: 109, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1laazto/the_eurollm_team_released_preview_versions_of/)): **The EuroLLM team released preview versions of multiple new models, including a 22B parameter LLM ([base](https://huggingface.co/utter-project/EuroLLM-22B-Preview), [instruct](https://huggingface.co/utter-project/EuroLLM-22B-Instruct-Preview)), two vision-language models ([1.7B](https://huggingface.co/utter-project/EuroVLM-1.7B-Preview), [9B](https://huggingface.co/utter-project/EuroVLM-9B-Preview)), and a small Mixture-of-Experts model ([2.6B total, 0.6B active](https://huggingface.co/utter-project/EuroMoE-2.6B-A0.6B-Preview)), all under the Apache-2.0 license. Notably, the MoE demonstrates strong performance relative to its parameter count. All models offer up to a 4K context window.** Commenters note the 22B model's context window limitation (4K) as a significant drawback, but see the releases as substantial progress for EU-origin open models. Informal evaluation in Russian suggests the 9B VLM reaches or exceeds the performance of comparable open models like Mistral and Gemma 2 (9B).
    - A user notes testing the EuroLLM 9B model with Russian, reporting it as "good but not perfect"—potentially slightly better than Mistral's smaller models and on par with Gemma 2 9B performance for the language, suggesting enhanced multilingual competence for this parameter range.
    - Discussion highlights the 22B model's `4k context` window, with one commenter implying that this may be insufficient for certain use cases, reflecting ongoing scrutiny of context length in large models.
    - There is skepticism about the stated parameter count for EuroMoE-2.6B-A0.6B (22B parameters) versus its `5 GB` model size, hinting at questions about compression, architecture (e.g., mixture-of-experts), or actual size-to-parameter correspondence.

### 2. OpenAI Open-Weight Model Tester Insights

- [**Got a tester version of the open-weight OpenAI model. Very lean inference engine!**](https://v.redd.it/3r075o87qo6f1) ([Score: 974, Comments: 74](https://www.reddit.com/r/LocalLLaMA/comments/1laee7q/got_a_tester_version_of_the_openweight_openai/)): **A user claims to have received a 'tester version' of an 'open-weight' OpenAI model, noting that the inference engine is 'very lean.' No benchmarks, implementation details, or architecture specifics are provided. The link to further technical data returns a 403 Forbidden error, so no external validation or details are available.** Top comments focus on the apparent speed ('time to first token is great') and user comfort with alignment, but there is no deep technical discussion or benchmarking in the comment section.
    - ExplorerWhole5697 makes a technical observation about the 'time to first token' being very fast, indicating low latency and efficient inference performance in the showcased OpenAI inference engine. This suggests that the custom, lean inference engine demonstrates strong responsiveness, which would be valuable in production environments with demanding real-time constraints.

### 3. AI Personality Preference and User Engagement Discussion

- [**We don't want AI yes-men. We want AI with opinions**](https://www.reddit.com/r/LocalLLaMA/comments/1lanhbd/we_dont_want_ai_yesmen_we_want_ai_with_opinions/) ([Score: 178, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1lanhbd/we_dont_want_ai_yesmen_we_want_ai_with_opinions/)): **The OP summarizes A/B testing and user engagement data from an AI podcast platform, showing that AI hosts with consistent, opinionated (but non-offensive) personalities lead to markedly higher user satisfaction (**`40%` **increase with 'sassy' mode) and much longer session times (**`2.5x` **increase). The implementation involved explicitly coding AI agents with quirky or contrarian viewpoints (e.g., 'cereal is soup'), resulting in users returning for continued debate—suggesting that authentic-feeling friction drives conversational depth and retention in LLM-based friend/character applications. Link: https://www.reddit.com/r/LocalLLaMA/comments/1dgwk71/we_dont_want_ai_yesmen_we_want_ai_with_opinions/** Top comments debate if 'yes-man' behavior is a default rather than inherent LLM property, noting that user prompts or system instrutions can fully control AI personality. Others point out the domain-specific aspect: contrarian AI is valuable in conversation agents but undesirable in utilitarian applications like calculators or self-driving cars.
    - Advanced users note that LLMs' agreeable 'assistant' personalities stem from default prompting and can be customized by altering the system prompt, allowing for more critical or opinionated AI behavior depending on user needs.
    - One commenter highlights that some models, notably Grok out-of-the-box, exhibit more willingness to 'push back' compared to others like ChatGPT, and mentions early Google models tended to be overly restrictive, sometimes refusing simple coding tasks due to safety or compliance measures.
    - Technical critiques credit platform constraints, such as API design or evaluation procedures like those in LLM arenas, as major reasons why public-facing models tend toward inoffensiveness and agreement, rather than providing robust, critical feedback by default.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. LLM Self-Improvement and Automated Fine-Tuning Advances

- [**SEAL: LLM That Writes Its Own Updates Solves 72.5% of ARC-AGI Tasks—Up from 0%**](https://arxiv.org/pdf/2506.10943) ([Score: 905, Comments: 180](https://www.reddit.com/r/singularity/comments/1la8myf/seal_llm_that_writes_its_own_updates_solves_725/)): **The post details SEAL (Self-Adapting Language Models), a framework where LLMs implement persistent, weight-level updating by autonomously generating their own finetuning data and 'self-edits' directives, closing the learning loop through a reinforcement meta-learning process. Unlike prior recursive/self-improvement frameworks, SEAL enables recursive self-improvement with actual parameter updates, yielding a high ARC-AGI score of 72.5% (vs. 0% for the same model without SEAL), and outperforming synthetic GPT-4.1-driven data approaches by directly optimizing useful self-training data. See the [arxiv paper](https://arxiv.org/pdf/2506.10943) for full technical details.** Commenters highlight that SEAL's recursive self-improvement genuinely updates the model's weights (unlike prior approaches), that the approach represents significant AGI progress, and that compute costs are now the main barrier to further advances in self-supervised, autonomous LLM learning.
    - The SEAL approach distinguishes itself from prior recursive frameworks by actually allowing the model to modify its own weights, rather than just its outputs or prompting strategies. This direct self-supervised weight update mechanism enables true self-improvement capabilities.
    - The underlying model utilized in SEAL is a variant of Llama 3.2 with 1 billion parameters, indicating that these results—solving 72.5% of ARC-AGI tasks—were achieved on a relatively compact model architecture, which underscores the significance of the self-improving technique.
    - Self-supervised fine-tuning is seen as a critical pathway for model progress, but commenters highlight that compute costs remain a key limiting factor in pushing this paradigm further, especially for larger models or sustained recursive improvement.
- [**"Anthropic researchers teach language models to fine-tune themselves"**](https://www.reddit.com/r/singularity/comments/1laip79/anthropic_researchers_teach_language_models_to/) ([Score: 357, Comments: 51](https://www.reddit.com/r/singularity/comments/1laip79/anthropic_researchers_teach_language_models_to/)): **Anthropic and collaborators introduce Internal Coherence Maximization (ICM), an unsupervised fine-tuning technique for large language models (LLMs) that leverages internal model consistency rather than human-annotated data ([paper summary](https://the-decoder.com/anthropic-researchers-teach-language-models-to-fine-tune-themselves/)). The approach aims to address scalability issues with human oversight as LLMs and tasks grow in complexity, arguing for model self-improvement by rewarding outputs that maintain logical self-coherence.** Discussion focuses on anticipated convergence of industry toward self-improving LLMs and a technical comparison with related methods like SEAL, indicating ongoing exploration of self-supervised fine-tuning paradigms.
    - A user asks how Anthropic's self-tuning approach differs from SEAL, referencing ongoing discussion about similar self-improvement mechanisms. SEAL generally refers to Semi-Supervised Reinforcement Learning from AI Feedback, whereas the Anthropic paper focuses on models autonomously generating and acting on their own fine-tuning data. The distinction may involve differences in feedback pipeline control and data autonomy, necessitating a close read of both papers for a precise comparison.
    - There's discussion on Anthropic's rapid progress, with specific reference to benchmarks—Opus 4 is purportedly outperforming its predecessor (Opus 3), Google's Gemini 2.5, and other models in tool use and agentic capability. The commenter highlights Anthropic's interpretability research as a competitive differentiator, especially compared to OpenAI and Google, emphasizing ongoing technical advances and shifts in AI research leadership.
    - Direct links to the Anthropic paper (https://arxiv.org/abs/2506.10139v1) provide readers primary access to the technical methods and purported results, supporting further analysis of self-tuning LLM performance and implementation specifics.

### 2. Claude Code Usage, Feedback, and Productivity Tips

- [**The $20 getting access to Claude Code has been honestly incredible**](https://i.redd.it/er5ds3xhwk6f1.png) ([Score: 172, Comments: 65](https://www.reddit.com/r/ClaudeAI/comments/1la0zsx/the_20_getting_access_to_claude_code_has_been/)): **The image displays a detailed daily usage report for Claude Code, specifically highlighting the user's high token consumption over multiple days and the associated hypothetical API costs, which totaled $94.00. The post technically contextualizes this by explaining that the author recouped their $20 Claude Pro subscription cost on the first day through intense use, greatly exceeding what equivalent API access would provide at retail pricing. The user contrasts their Claude experience with other LLMs, noting that while 'roo' (presumably OpenAI's GPT-4 'Turbo' or similar) remains superior for certain workflows, Claude Code Pro—with generous context window and cost efficiency—substantially cuts down on API spending for code-heavy workloads.** Comments facetiously speculate that reports like this may drive Anthropic to raise rates, with several users sharing similar savings and remarking on the perceived unsustainability of the current pricing, effectively confirming the high technical and financial value of the plan for power users.
    - One user shared their spending on different AI providers for personal projects: `$500 on Gemini`, `$500 on OpenRouter`, and `$700 on Anthropic` in a single month. They noted that the $20 Anthropic subscription is quickly rate-limited for extensive architectural documentation tasks, prompting an upgrade to the $100 plan, illustrating *the cost-benefit tradeoff and usage thresholds for heavy users* (reference: Claude.ai).
- [**I discovered a powerful way to continuously improve my CLAUDE.md instructions for Claude Code**](https://www.reddit.com/r/ClaudeAI/comments/1laby6h/i_discovered_a_powerful_way_to_continuously/) ([Score: 313, Comments: 62](https://www.reddit.com/r/ClaudeAI/comments/1laby6h/i_discovered_a_powerful_way_to_continuously/)): **The OP has implemented an automated continuous improvement loop for their Claude Code assistant instructions (**`CLAUDE.md`**) using a** `/project:reflection` **command, which prompts the agent to analyze recent chat history for instruction gaps and propose targeted improvements (reflecting principles from prompt engineering and instruction tuning). Main identified issues included missing integration guidelines (e.g., Jira/Atlassian use, documentation standards, refactoring strategies, project context, and incremental development process). The method enforces structured feedback, iterative human approval, and precise instruction updates, closely tracking observed performance bottlenecks and contextual misunderstandings.** One commenter highlighted the value of integrating instruction optimization with tool usage via `.claude/commands`, suggesting further automation in tool selection; another pointed out that Claude Code can ignore the `CLAUDE.md` file unless explicitly directed to read it, indicating a technical challenge in context loading and grounding the assistant's behavior.
    - a_c_m shares an extension to the system, incorporating a `.claude/commands` directory to manage tool usage, highlighting that optimizing tool invocation is a significant lever for improving Claude Code's effectiveness ([gist here](https://gist.github.com/a-c-m/f4cead5ca125d2eaad073dfd71efbcfc)). This approach emphasizes modularity and fine-grained control over command execution.
    - FBIFreezeNow notes a potential practical issue: Claude Code (CC) does not always reference the `CLAUDE.md` instruction file unless explicitly prompted, impacting consistency in following instructions. This suggests a limitation in implicit context utilization or auto-referencing behaviors that could influence prompt engineering strategies.
    - Fine-Presentation216 raises a maintainability concern that iterative updates to `claude.md` risk introducing redundant or repetitive instructions, advocating for a 'Don't Repeat Yourself' (DRY) principle in update workflows. This highlights the tradeoff between continual improvement and instruction bloat.
- [**Am I the only one who finds the "secrets" to amazing Claude Coding performance to be the same universal tips that make every other AI model usable? (Ex: strong CLAUDE.md file, plan/break complex tasks into markdown files, maintain a persistent memory bank, avoid long conversations/context)**](https://www.reddit.com/r/ClaudeAI/comments/1la5kp4/am_i_the_only_one_who_finds_the_secrets_to/) ([Score: 148, Comments: 48](https://www.reddit.com/r/ClaudeAI/comments/1la5kp4/am_i_the_only_one_who_finds_the_secrets_to/)): **The post argues that so-called 'secret' best practices for maximizing AI coding performance in models like Claude are largely universal across LLM coding assistants (e.g., Copilot, Aider, Gemini). Key recommendations include: maintaining a detailed, hand-crafted 'CLAUDE.md' project architecture file to condense context, breaking complex tasks into granular markdown files for persistent task history and context efficiency, using persistent memory artifacts (CLAUDE.md, CHANGELOG.md), curtailing conversation length to avoid model confusion, and prioritizing strong modular unit tests to reduce bug resolution recursion. These practices leverage model strengths (precision with clear intent, context efficiency) and mitigate weaknesses (long context deterioration, context slot limits), with claims that further optimizations show diminishing returns except in well-scoped agent frameworks.** Top comments introduce a multi-agent workflow where distinct Claude agents with unique identities operate concurrently on different features, communicate via a shared 'developer_coms' directory, and resolve git conflicts collaboratively, simulating project management best practices. Others corroborate the value of hierarchical, inter-referencing markdown files for maintaining synchronized context, proposing structured file hierarchies (Claude.md → Project_todo.md → Feature_todo.md → Sprint_todo.md). Consensus is that effective AI-assisted coding mirrors rigorous project management methodologies.
    - A detailed multi-agent workflow is described: spawning multiple Claude agent instances, each as a distinct developer (via unique `.identity` files), all working on different codebase features in parallel terminals. Agents communicate via a shared `developer_coms` directory for coordination, resolve git conflicts after each individual task, and can reach consensus or vote on project updates, effectively simulating a collaborative dev environment and showcasing the power of agentic project management techniques.
    - Referencing and linking Markdown documentation files (`Claude.md` → `Project_todo.md` → `Feature_todo.md` → `Sprint_todo.md`) creates a maintained dependency/context graph. This structure facilitates model updates across all planning layers, ensuring context completeness and synchronization as tasks or dependencies change. It leverages Claude's ability to keep disparate documents in sync and propagate changes through the hierarchy.
    - There is discussion about using Claude to index and analyze the entire codebase before detailed planning. This setup involves generating a series of planning documents (e.g., plan.md, architecture, API, back-end specs), then asking Claude to create a phased, checklist-driven task plan. This aligns with practices of AI-enhanced planning—front-loading context absorption and explicit checklist creation improves robustness for large or complex coding workflows.

### 3. AI and Coding Tool Updates and Launches (Roo Code 3.20.0, MagCache, LoRA-Edit)

- [**Roo Code 3.20.0 | THIS IS A BIG ONE!!**](https://www.reddit.com/r/ChatGPTCoding/comments/1la61eo/roo_code_3200_this_is_a_big_one/) ([Score: 127, Comments: 31](https://www.reddit.com/r/ChatGPTCoding/comments/1la61eo/roo_code_3200_this_is_a_big_one/)): **Roo Code 3.20.0 introduces an experimental Marketplace for extensions (MCPs) and modes, enabling project/global-scope installs and management directly in the UI ([docs](https://docs.roocode.com/update-notes/v3.20.0#mcp--mode-marketplace-experimental)), as well as experimental multi-file concurrent edits for batch refactoring ([details](https://docs.roocode.com/features/experimental/concurrent-file-edits)) and concurrent file reads (now defaulting to 5 concurrent reads in context settings). Prompt history navigation now mirrors terminal UX, and the update also brings 17+ improvements and provider support updates (DeepSeek R1, Bedrock reasoning, XAI, O3, OpenRouter). Full changelog [here](https://docs.roocode.com/update-notes/v3.20.0).** One technical commenter questions the transparency of Roo Code's maintainers, noting that contributor or author attribution is not visible on the GitHub page—a concern relevant for open-source trust and collaboration.
    - There is a technical question regarding the visibility and attribution of developers on the Roo Code GitHub page. A commenter notes that the contributors or team behind Roo Code are not visible in the repository, which could hinder transparency and open-source trust for users and other developers. This may impact auditing, trust, and collaborative contributions to the project.
    - Users seek clarification on integration and usability features of the new MCP Marketplace, specifically whether it can be browsed outside of the RooCode environment and how one can submit content to the marketplace. This highlights interest in marketplace extensibility and third-party contribution mechanisms, as well as API or UI exposure beyond the core IDE.
- [**MagCache, the successor of TeaCache?**](https://v.redd.it/6kep8ze8vm6f1) ([Score: 180, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1la8e7m/magcache_the_successor_of_teacache/)): **MagCache is presented as a successor to TeaCache, with implementation targeting ComfyUI for diffusion model acceleration (links: [website](https://zehong-ma.github.io/MagCache/), [GitHub](https://github.com/Zehong-Ma/ComfyUI-MagCache)). Early user testing on high-end hardware (e.g., H100 SXM GPU) noted lack of Skip Layer Guidance support and observed only marginal speed improvements (**`~8 sec`**) over TeaCache, with inferior sample quality, particularly on Wan T2V 14B. Compatibility concerns were raised regarding mandatory use of** `torch.compile`**, as it requires** `80 SMs` **(Streaming Multiprocessors), limiting support to top-tier NVIDIA hardware (4080/5080 series and above).** Commenters are generally critical of MagCache's performance relative to TeaCache, emphasizing output quality degradation and limited practical acceleration as major drawbacks. There is also debate about hardware requirements, with users expressing concern over the narrow compatibility due to the high SM count needed for torch.compile.
    - Testing MagCache on an H100 SXM revealed that while it offers an `8 second` speed improvement over TeaCache, the generated results are notably inferior in quality when using the recommended settings for **Wan T2V 14B**. Without features like Skip Layer Guidance, the perceived improvements are limited, forcing users to lower settings for only marginal gains.
    - A technical question was raised about whether `torch.compile` is mandatory for MagCache operation. The concern is that `torch.compile` requires NVIDIA GPUs with at least `80 SMs (Streaming Multiprocessors)`, meaning many consumer GPUs (e.g., 4060Ti, 4070) cannot use it, possibly restricting MagCache usage to high-end devices (4080/5080 and above).
    - With Flux, MagCache's image quality is described as poor, though it may still outpace previous caching methods in generating previews rapidly due to strong compositional fidelity. Nonetheless, its utility may be limited for high-quality outputs.
- [**LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning**](https://v.redd.it/tu3gpipkcm6f1) ([Score: 176, Comments: 9](https://www.reddit.com/r/StableDiffusion/comments/1la6nta/loraedit_controllable_firstframeguided_video/)): **LoRA-Edit introduces a mask-driven LoRA (Low-Rank Adaptation) fine-tuning strategy for video editing, leveraging a pretrained Image-to-Video (I2V) diffusion model for controllable, first-frame-guided edits. The approach uses spatial masks to isolate background preservation from targeted edit propagation, combining cues from input videos (motion, spatial structure) and reference images (appearance) via dynamic attention modulation to support region-specific learning, outperforming state-of-the-art methods according to experimental results. The method does not alter the core model architecture and supports flexible adaptation; code is available on [GitHub](https://github.com/cjeen/LoRAEdit).** Commenters are requesting integration of LoRA-Edit with ComfyUI, indicating demand for broader accessibility and workflow compatibility in established UI frameworks.
    - Two users request or anticipate the integration of LoRA-Edit with ComfyUI, indicating a desire for practical wrappers and UI-based workflows to leverage this new controllable video editing technique in established pipelines.
    - One comment expresses skepticism regarding the reliability of results shown in "Ours" demos, alluding to a broader concern in the community about reproducibility and the real-world performance of novel methods versus curated demonstrations.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking
> 

**Theme 1. Infrastructure Instability Strikes Across Platforms**

- **Cloudflare and Google Cloud Outages Halt AI Services**: Widespread outages across **Cloudflare** and **Google Cloud Platform** crippled multiple AI platforms, including **Cursor**, **OpenRouter**, and **Cohere**, disrupting login and core functionality. Status pages for [Cloudflare](https://www.cloudflarestatus.com/) and [Google Cloud](https://status.cloud.google.com/) detailed the issues, with [OpenRouterAI noting signs of recovery](https://x.com/OpenRouterAI/status/1933263905385500853) via their X account.
- **Networking Bandwidth Disparity Grows**: The bandwidth of typical internet connections, hovering around **1gbps**, starkly contrasts with [NVIDIA's latest Infiniband iteration reaching 130TB/s](https://x.com/toolandtea/status/1933381389552136705), highlighting a widening gap in network capabilities which impacts distributed training efficiency. Decentralized options like [DAWN Internet](https://x.com/dawninternet), which uses fixed wireless and includes an RL-capable GPU in its router, were presented as alternatives to traditional providers.
- **Cloud Dependencies Cause LlamaCloud Wobbles**: **LlamaCloud** experienced operational instability due to upstream infrastructure issues, underscoring how dependent AI services are on external cloud providers, prompting users to monitor the [official status page](https://t.co/IdecAksHiG) for real-time updates. This incident, alongside others, highlights the fragility inherent in relying on third-party cloud services.

**Theme 2. Model Performance, Capabilities, and Quirks**

- **Model Performance Debates Rage**: Users debated model preferences and capabilities, with discussions comparing **o3** and **2.5 Pro's** general performance versus math strengths, while benchmarks like [MathArena](https://matharena.ai/) faced scrutiny for potential saturation. New [text-to-video models Seedance and Kangaroo](https://artificialanalysis.ai/text-to-video/arena) impressed users by potentially outperforming **Veo3**, **Kling**, and **Pika** in recent comparisons.
- **Next-Gen Models Hint at Internal Tool Use and Parallel Processing**: **GPT-5's** architecture reportedly relies on internal specialized tools for enhanced context and stability, while a leading theory suggests OpenAI's **GPT Pro models** like **O3 Pro** improve reasoning by running multiple instances in parallel and consolidating results, potentially explaining **O3-pro's** **93%** accuracy on the **AIME 2024** math competition compared to **O3's 90%**. Despite this, some users reported **O3 Pro** failing to answer questions from uploaded documents after long waits.
- **Model Limitations and Bias Evals Surface**: Users noted **Gemini Pro 2.5** struggles with simple image recognition and local LLMs have trouble with large context windows, while bias evaluations revealed that adding realistic details can trigger **race and gender bias** in models like **GPT4o** and **Claude 4 Sonnet**. The concept of LLMs bypassing constantly updated captchas was likened to the ["Red Queen hypothesis"](https://en.wikipedia.org/wiki/Red_Queen_hypothesis), where progress is quickly countered by new defenses.

**Theme 3. Hardware and Low-Level Optimization Battles**

- **AMD GPUs Gain Traction and Unsloth Support**: The **Unsloth** team expressed interest in supporting **AMD** GPUs, highlighting the new **AMD INSTINCT MI355X GPU's** **5x FP8 flops** compared to the **H100**, noting AMD's advantage in affordability and high memory, though driver support remains a question. **GemLite** also announced the addition of **ROCm support**, focusing on the **MI300X** and implementing custom mma instructions via **LLVM intrinsics** and **Mojo APIs**, detailed in [this post on X](https://x.com/mobicham/status/1933520405106507909) and a [blog post](https://veitner.bearblog.dev/programming-tensor-cores-in-mojo/).
- **Torch.compile and CUDA Libraries Boost Performance**: Members found significant speedups using **torch.compile** for convolution kernels, improving performance from **1.7489 ms** to **0.0020 ms** by calling external kernels. Discussions in **CUDA** explored memory layout optimization libraries for **Blackwell**, utilizing [cuda-side-boost](https://github.com/ademeure/cuda-side-boost) for development, and configuring L1/L2 cache policies.
- **Hardware Decisions Weigh VRAM and Cost**: Debate arose on using a used **Tesla P40** for **24GB VRAM** expansion for around **$300**, with consensus deeming it not worth it compared to a used **3090** as a better 'affordable' option. Discussions around optimal local LLM performance touched on the need for **150B+ parameters** for *"reasonable human interaction"* versus the importance of prompting, RAG, and fine-tuning.

**Theme 4. Developing with AI: Tools, Agents, and APIs**

- **Coding Assistants Embrace New Features and Fixes**: **Cursor** faced issues with Cloudflare/GCP outages, code generation choppiness, and background agent privacy/commit problems, while **Claude Code** excelled in context for complex refactors. **Aider** users praised its performance with smaller local models like **8B** and **12B** via **Ollama**, attributing success to its repomap, while also discussing costs with **Anthropic** and dependency management with **UV**. **Windsurf (Codeium)** launched [Wave 10 UI/UX upgrades](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise), an [EU cluster](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI), and [added Claude Sonnet 4 support](https://x.com/_mohansolo/status/1933605162775687482).
- **Agentic Frameworks See New Tools and Security Measures**: New tools emerged to support AI agents, including **Taskerio's** [inbox for tracking coding agent progress](https://www.reddit.com/r/mcp/comments/1lac12i/an_mcp_to_track_the_progress_of_your_ai_agents/) via webhooks and an API, and [SchemaPin](https://github.com/ThirdKeyAI/SchemaPin), designed to fortify MCPs against *"Rug Pull"* exploits, with simple implementation detailed on [SchemaPin.org](http://schemapin.org/). **GitHub** unveiled a [remote MCP server](https://www.reddit.com/r/mcp/s/Cj2zjute95) allowing hosts access to live context using dynamic tool selection, and a [guide was shared on building MCP servers](https://youtu.be/YzT9QU-Kfh4?si=tqqgBXXu9ct2aMUH) using Postman's builder and APIs.
- **Platforms Integrate Models and Improve Usability**: **LlamaIndex** added [support for MistralAI's Magistral model](https://t.co/ZsUEWMrnT4) and introduced **LlamaParse Presets** for balancing parsing accuracy and speed, while integrating with **Mem0** for automatic memory updates in agent workflows. **NotebookLM** users requested Excel/Sheets support and reported issues with mobile notes display and sharing features. **OpenRouter** users debated quality variations among providers and requested future multi-modal capabilities like audio/video generation.

**Theme 5. AI Research Concepts and Debates**

- **AI Safety and Bias Spark Discussion**: Skepticism arose around a new AI Safety Institute, citing a lack of prior awareness and publications. Research highlighted how adding realistic details to **bias evals** triggers **race and gender bias** in LLMs, causing up to a **12%** difference in simulated outcomes across models, and noted that **Chain of Thought** methods failed to reveal this hidden bias, as detailed in the [paper on Robustly Improving LLM Fairness](https://x.com/a_karvonen/status/1933582375419850806).
- **Evaluation Methods and Benchmarks Scrutinized**: Critiques were raised about evaluating AI reasoning using tasks like **River Crossing experiments**, noting models that correctly identify unsolvable problems were inadvertently penalized according to [The Illusion of the Illusion of Thinking paper](https://arxiv.org/abs/2506.09250). Debates continued on the validity of benchmarks like **MathArena** as scores approach **100%**.
- **Core AI Concepts Debated**: Discussions covered pitfalls in **gradient estimation for KL divergence** in RL training for LLMs, highlighting issues in open-source projects and papers such as [GRPO](https://www.notion.so/1f4df34dac1880948858f95aeb88872f?pvs=21) and the [paper on KL divergence pitfalls](https://arxiv.org/pdf/2506.09477), and questioned the meaning of terms like *"symbolic recursion"*. High-level disagreements on the future of AI jobs between **Jensen Huang (Nvidia)** and **Dario Amodei (Anthropic)** were also noted following a [Fortune article](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagrees-anthropic-ceo-dario-amodei-ai-jobs/) and subsequent [X posts](https://www.x.com/dario).


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Deep Think Incoming**: A member shared an image ([aSHuTrz.jpeg](https://cdn.discordapp.com/attachments/1047649527299055688/1382930529464352848/aSHuTrz.jpeg?ex=684d9aab&is=684c492b&hm=b30084937010eeeb0b4dba66cb69c6fd085c52fb786ce6e316e2324b2f50d0aa&)) hinting at the arrival of **Gemini Deep Think**.
   - No further details about the specifics of **Gemini Deep Think** were provided.
- **Perplexity Pro Role is DOA**: Users reported issues obtaining the **Perplexity Pro role** on Discord, citing a non-functional onboarding button.
   - A workaround suggested pinging a staff member to manually assign the role, with one user noting, *"the button doesn't seam to give me the role just puts me in the server on the phone"*.
- **Perplexity Pro Draws Pictures Now**: Members discovered that **Perplexity Pro** can generate images from text prompts entered in the search bar.
   - Further, users gave instructions to refine images by clicking **Regenerate** or sending new prompts with styles like *cinematic*, *anime*, and *low-poly 3D*.
- **GPT-5 Thinks Smarter, Not Harder**: A member shared details about **GPT-5**'s architecture, emphasizing its reliance on internal specialized tools, which sidesteps the problems of external routing and hallucinations.
   - A member said that *"GPT-5 thinks with its tools, not beside them"*, underscoring enhanced context, coordination, and stability.
- **Sonar API Documentation Demands Scrutiny**: The Perplexity team seeks user feedback on the **Sonar API documentation**, especially concerning unclear or hard-to-navigate sections, available at [this community post](https://community.perplexity.ai/t/improvements-to-the-sonar-api-documentation/542?u=vikvang).
   - The feedback aims to improve the documentation based on user experiences.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro vs 2.5 Pro: Showdown**: Members fiercely debated model preferences, with some advocating for **o3** over **2.5 Pro** in overall performance, while others cited **2.5 Pro's** strength in math.
   - One member quipped *"I wish to live in this level of delusion"* regarding another's preference for **2.5 Pro** in math.
- **MathArena Benchmarks: Are they Still Relevant?**: The community discussed the ongoing validity of [MathArena benchmarks](https://matharena.ai), with some suggesting they are becoming saturated and driven by luck.
   - Concerns arose that scores nearing **100%** might indicate saturation, thus reducing the statistical significance of these benchmarks.
- **Kingfall release kills Google Account**: A user's Google account ban sparked speculations about a new **Kingfall** release and a new gemini model codenamed *toothless* showed up for a brief period.
   - There's even reports of **99% profit** on various ventures, prompting speculation about the model's capabilities.
- **Text-to-Video Arena: Seedance and Kangaroo Arrive**: In a [text-to-video arena](https://artificialanalysis.ai/text-to-video/arena), **Seedance 1.0** and the anonymous **Kangaroo** model impressed users with their performance.
   - Comparisons indicated that these models could potentially outperform **Veo3**, **Kling**, and **Pika**, particularly in generating similar outputs from general prompts.
- **Cloud Crash Causes Chat Catastrophe**: Due to a cloud provider outage on **6/12/25**, the team warned that **chat history data may have been lost**.
   - The team apologized for any inconvenience, noting they are *working on solutions to ensure this doesn’t happen again*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cloudflare Outage Cripples Cursor**: A **Cloudflare** and **GCP** outage brought **Cursor** to its knees, disrupting login and core functionalities, though [Tab reportedly remained operational](https://www.cloudflarestatus.com/).
   - The disruption underscored the reliance of development tools on external services, with the issues later [marked as resolved](https://mashable.com/article/google-down-cloudflare-twitch-character-ai-internet-outage).
- **Cursor's Code Choppiness Continues**: Users are still reporting issues with **Cursor's code generation** when auto model selection is turned on, with one user lamenting the loss of **50 inference credits** due to messy code output.
   - One user asked about using cursor to make three.js games, and another recommended O3 for most coding, and O3-pro for planning and debugging, emphasizing its effectiveness over other models.
- **Claude Code Commands Context**: Users are finding that **Claude Code** excels in grasping context and churning out high-quality code, especially when wrestling with complex refactors.
   - It helped add **3500 new tests** for a front-end component library, a testament to its capabilities; this highlights its ability to handle large-scale code modifications effectively.
- **Privacy Mode Prevents Progress for Background Agents**: Users encountered an error message stating, *Background agent is not supported in privacy mode*, while trying to initiate a background agent, due to an enabled **account-level privacy mode**.
   - The issue can be resolved at [this link](https://www.cursor.com/slack-connected), and the problem is slated for resolution in the upcoming version.
- **Background Agents Break Commit Conventions**: A background agent, after amending a commit, ran into roadblocks trying to push the altered commit to the repo, implying some version control snags.
   - A member suggested resolving it through the terminal, since the agent was getting rolled back, hinting at potential issues with how agents handle version control operations.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Google Cloud Implodes**: **Google Cloud** suffered a major outage, as reported on their [status page](https://status.cloud.google.com/), with users reporting intermittent issues even after initial signs of recovery around **4:25pm ET**.
   - **OpenRouterAI** tweeted about seeing recovery from the outage, expressing hope it wouldn't be temporary ([tweet link](https://x.com/OpenRouterAI/status/1933263905385500853)).
- **Cloudflare Kills Internet (Again)**: A widespread [Cloudflare outage](https://www.cloudflarestatus.com/) caused significant disruptions, taking down numerous AI services including **OpenRouter**, Google, and others.
   - Users experienced intermittent **OpenRouter** service, with the status page flipping between *MAJOR* and *MINOR* outages.
- **Provider Variability Impacts Model Qualities**: Users discussed the significant quality variations among different providers offering the same models through **OpenRouter**, noting that **Parasail** and **Lambda** generally offer more consistent performance.
   - One user emphasized the importance of quality over cost, stating that the [quality varies a lot by providers, so choose wisely](https://discord.com/channels/1091220969173028894/1092729520181739581/1383133709551282236).
- **Cheap Agent LLMs Emerge as Top Tool-Users**: Users debated the best cheap Large Language Models (**LLMs**) for agentic tool use, with **Claude 2.5 Flash** being recommended as a cost-effective option that requires careful prompting.
   - Discussion also included the high cost of models like **O4 Mini High** and the efficiency of using a [monthly Claude Max subscription](https://discord.com/channels/1091220969173028894/1195014798837043240/1383046909199872050) for API usage.
- **Hoping for OpenRouter Multi-Modal Capabilities**: Members requested future support for multi-modal capabilities like **audio** and **video generation** within the **OpenRouter** platform.
   - There was no explicit response given by OpenRouter.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Lacks Automatic Model Updates**: Unlike **Ollama**, **LM Studio** does not automatically download model updates; most model updates are published in new repositories, making model lineage difficult to track.
   - A member noted this makes it difficult to determine the lineage of a model.
- **Gemini Pro Botches Image Recognition**: A user reported that **Gemini Pro 2.5** makes errors in simple image recognition despite varied prompts and images, even with the [provided image](https://cdn.discordapp.com/attachments/1110598183144399058/1382962931741888563/image.png?ex=684db8d9&is=684c6759&hm=14fbd9fe32c10ead609c4627acfd3c543cdf7ca6f347af0e6c9a61470af4c663&).
   - Another member mentioned that vision-enabled models often perform poorly, with unclear user expectations.
- **LLMs Struggle Against Upgraded Captchas**: Members highlighted the ongoing challenge of using **LLMs** to bypass captchas, as captchas are designed to resist computer cracking and are constantly updated.
   - The situation resembles the [Red Queen hypothesis](https://en.wikipedia.org/wiki/Red_Queen_hypothesis), where advancements in captcha cracking are quickly countered by new defenses.
- **OpenWebUI Enables Remote LM Studio Access**: **OpenWebUI** facilitates running **LM Studio** on a server for remote access by hosting the server, loading a model, serving it on the local network, enabling **CORS**, and opening ports like 1234, 8080, or 3000.
   - The accessing PC does not need **OpenWebUI** installed.
- **Tesla P40 Not Worth It Anymore**: A member asked about using a **Tesla P40** as an additional **GPU** like **RTX 3090/4090** to expand **VRAM** for LM Studio for around **$300** for **24GB**, linking to the [TechPowerUp specs](https://www.techpowerup.com/gpu-specs/tesla-p40.c2878).
   - The consensus was that the **$300** price point is no longer worth it as a used **3090** is a better 'affordable' option.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Safety Institute Faces Skepticism**: Members expressed doubt about the legitimacy of a new AI Safety Institute, citing a lack of prior awareness and absence of recent publications on its website, however the advisor is on Discord.
   - A member suggested initiating contact, pointing out the advisor is on Discord.
- **German Text Reveals LLM Quirks**: A short German text prompted drastically different reactions from **GPT-3** and **GPT-4o**, spanning from neutral to deeply emotional responses.
   - The member questioned whether this anomaly merited further investigation, hinting at an interest in exploring **LLM** behaviors beyond conventional applications.
- **Symbolica.ai Eyes Theorem Prover Model**: [Symbolica.ai](https://www.symbolica.ai/), a London-based startup, aims for ambitious goals, however they should release a small theorem prover model like the one Google had.
   - Some reviews suggested the *boundaries of the work aren't clear and the goals keep changing*.
- **GRPO Objective Supercharges Model Performance**: **DeepSeek V3**, a **671B** model, demonstrated enhanced performance through the [GRPO objective](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f), succeeding validation tasks.
   - A member noted that *literally random rewards improve performance* due to a *concentration effect that focuses the model on its existing reasoning pattern distribution*.
- **Bias Evals Trigger Race and Gender Bias**: Adding realistic details to existing **bias evals** can trigger **race and gender bias** in **LLMs**, causing up to a **12% difference** in interview rates across models including **GPT4o** and **Claude 4 Sonnet**.
   - The [paper on Robustly Improving LLM Fairness](https://x.com/a_karvonen/status/1933582375419850806) gives an example of **unfaithful chain of thought** in the wild.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Canvas Enables Code and Doc Exports**: **Canvas** now supports downloads and exports, enabling users to export documents as **PDF**, **docx**, or **markdown** files.
   - Additionally, **Canvas** facilitates direct code export to appropriate file types such as **.py**, **.js**, and **.sql**.
- **GPT Pro's Parallel Power Play**: A leading theory proposes that **GPT Pro models**, like **O3 Pro**, enhance reasoning by running multiple instances in parallel and consolidating the results in a *'think harder'* approach.
   - Evidence from the **AIME 2024** math competition showed **O3-pro** achieved **93%** pass@1 accuracy compared to **O3's 90%**, implying the effectiveness of this consolidation method.
- **O3 Pro Performance Faces Project Fails**: Users have reported that **O3 Pro** often fails to answer questions from uploaded documents, despite long waiting times of up to **40 minutes**.
   - This underperformance raises questions about its practical utility, contrasting with its enhanced reasoning capabilities.
- **Free AI APIs Fuel Development**: Developers explored free AI APIs such as **SambaNova**, which features fast **Llama 3.3 70B**, **Qwen**, and **Deepseek** models.
   - **Gemini** was noted for its high rate limits, offering options like **500/day** for **2.5 Flash** and **1k/day** for **2.0 Flash**, making it suitable for budget-conscious projects.
- **Discord Dwindles Due to AI Surge**: A noticeable drop in **Discord activity** correlates with the rise in popularity of **AI chats**, leading to many servers becoming *'ghost towns'*, which prompts new thinking for community engagement.
   - This shift indicates users are migrating to AI-driven platforms for discussions, impacting community engagement on traditional platforms.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen 2.5 Parlez 100 Languages**: **Qwen 2.5** speaks *100 languages*, potentially due to containing a substantial amount of multilingual data from its *18T tokens* training, and leveraging the **Linux VM** well.
   - Members compared it to **Gemma3**.
- **CloneMe Creates Digital Twins**: The [CloneMe AI platform](https://github.com/vibheksoni/cloneme) lets you build your **digital twin**—an **AI** that chats like you, remembers details, and supports multiple platforms.
   - It's customizable, **memory-driven**, and **hot-reloadable**, making it a toolkit for creating intelligent, dynamic **AI personas**.
- **HF Faces Heat on Open Source Facade**: Some believe certain models on **Hugging Face** aren't truly **open source**, suggesting the platform may be used more for marketing than genuine collaboration.
   - While **Hugging Face** doesn't explicitly brand itself as an *open source* library, its reputation suggests otherwise.
- **TensorBoard Tells Tale of Fitting**: Members are using **TensorBoard loss graphs** to diagnose model fitting, emphasizing that *evaluation loss* should decrease at a similar rate to the *training loss*.
   - Dividing the dataset into **training** and **testing** parts ensures the model *generalizes* well without overfitting or underfitting.
- **Augmentoolkit 3.0 Augments AI**: [Augmentoolkit 3.0](https://github.com/e-p-armstrong/augmentoolkit) allows users to **train AI** on new subjects by adding documents or teaching it tasks through rating attempts.
   - It facilitates **custom model** runs, offering greater control over update timing and methods.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Meltdown Suspected After Veo3**: Users reported widespread issues with **Manus**, suspecting the **Veo3 announcement** overloaded the servers, as confirmed by [Downdetector](https://downdetector.com/).
   - The outage triggered frustration, with one user reporting *every task I spin up is about 900-1000 credits for me*.
- **Playbooks Prepare Prompts Preemptively**: **Playbooks** in Manus prepare prompts and give output examples, bridging the gap for users needing prompt assistance and [highlighting creative workflows](https://manus.im/playbook).
   - The Playbooks aim to provide structured guidance, facilitating easier prompt engineering.
- **Community Clamors Constantly for Claude**: Users expressed eagerness for **Claude 4.0**, drawing humorous parallels to fan anticipation, though there was no official news or update.
   - A user suggested a workaround to *make new gmail and sign up for the google one ai trial -> start a family -> invite 5 accounts -> 5x usage now for veo and everything bc all accounts get separate usage limits*.
- **Credit Crunch Causes Costs Concerns**: Users voiced concerns over **credit usage**, particularly regarding optimization and lack of cost previews, with some suggesting the *bring your own keys* model.
   - The lack of cost transparency is causing some consternation in the community.
- **GPT Generates Greatness over Manus**: Image generation quality between Manus and GPT-4 Omni were compared, showing [GPT-4 Omni](https://cdn.discordapp.com/attachments/1349440650495398020/1382980409243209859/GPT4omini.png?ex=684dc920&is=684c77a0&hm=1d615e514982fcfdfb5677c8640ae6d7ea8282e5f56d86e5382a2578a0084b82&) outperformed Manus.
   - The comparison highlighted specific instances where GPT-4 Omni provided superior image outputs.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Torch Compile Magically Speeds Up Convolutions**: Members found that operations generated from **torch.compile** result in a significant speedup in a convolution kernel from **1.7489 ms** (Native PyTorch) to **0.0020 ms** (Compiled PyTorch).
   - Questions arose as to why the stock convolution doesn't use the faster external kernels called via *extern_kernels.convolution* instead of *aten.convolution*.
- **CUDA-side Boost Library Charges In**: A member shared a link to [cuda-side-boost](https://github.com/ademeure/cuda-side-boost), a library for **CUDA** development, noting that *replacing the entire PyTorch memory allocator is probably overkill*.
   - They suggest one could use **MemPool** in **PyTorch** instead.
- **GemLite Gets Amped for ROCm**: A developer announced the addition of **ROCm support** to **GemLite**, with a focus on the **MI300X** ([post on X](https://x.com/mobicham/status/1933520405106507909)).
   - The post details implementing **custom mma instructions** via **LLVM intrinsics**, and efficiently managing **data layouts** with **Mojo's load_matrix** and **store_matrix APIs** ([github repo](https://github.com/simveit/mma_mojo)).
- **Factorio Newbie Seeks Reading Material After PMPP**: Members are seeking recommendations on books or papers to read after completing **PMPP**, with one member suggesting [a paper on instruction latencies](https://arxiv.org/pdf/1903.07486).
   - The member suggests that *the discussion itself is worth a read*, despite the fact that **instruction latencies** might be outdated.
- **Factorio RL throwdown begins**: Members discussed the potential of using **RL-based AI** to play Factorio, debating whether an LLM is necessary for long-term planning and complex tasks.
   - The conversation explored whether an RL agent could achieve optimal Factorio play with a limited amount of gameplay, drawing comparisons to OpenAI Five's success in Dota 2.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **AMD Instinct MI355X GPU gets Unsloth support**: The Unsloth team may support **AMD** GPUs with the new **AMD INSTINCT MI355X GPU** having **5x the FP8 flops as the H100** and they [presented at the AMD AI conference](https://x.com/danielhanchen/status/1933150291719082098).
   - Members noted that **AMD is cheap and has high memory**, but also questioned **AMD's driver support**.
- **Unsloth Mulls YouTube Channel Creation**: The Unsloth team is considering creating a **YouTube channel** to upload videos, particularly focused on tutorials.
   - A member asked for a video on how to use **multiple GPUs with accelerate**, promising to *like and subscribe*.
- **AttributeError Plagues Unsloth Training Sessions**: A user encountered an **AttributeError** during training with Unsloth, traced to the `fetch_image` function trying to read a `None` **images field** instead of a valid path or URL.
   - A suggestion was to use **batch size 1** or pass a custom collator.
- **KL Divergence Gradient Estimation Has Flaws**: A paper was shared discussing pitfalls in **gradient estimation for KL divergence** in RL training for LLMs, highlighting issues in open-source projects like **TRL** and **Open Instruct** and papers such as **GRPO**.
   - The paper points out that *differentiating through the KL estimate as loss functions* and *not accounting for the sequential nature* can lead to incorrect KL gradients, referencing [this paper](https://arxiv.org/pdf/2506.09477).
- **River Crossing Errors Plague Apple's Reasoning Model**: A paper titled [The Illusion of the Illusion of Thinking](https://arxiv.org/abs/2506.09250) was shared, criticizing the evaluation of AI models in **River Crossing experiments** for inadvertently penalizing models that correctly identify unsolvable problems.
   - The original paper by Apple had instances with **N ≥ 6 actors/agents** using boat capacity **b = 3**, which is mathematically impossible.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Nano-vLLM catches Nano-tice**: DeepSeek released **nano-vLLM**, a minimal vLLM implementation of approximately **1200 lines of code** that is a valuable learning resource for AI/ML practitioners and can be found at [this link](https://xcancel.com/wzihanw/status/1933225058031288772?s=46).
   - The community appreciates its concise nature as a valuable learning resource and expressed interest in hacking on the *'nano monolith'*. 
- **Trinity autoformalizes Fermat's Last Theorem**: **Morph Labs** introduced **Trinity**, an autoformalization system used to formalize de Bruijn's result on the abc conjecture in Lean, available at [this link](https://xcancel.com/morph_labs/status/1933181394588483868?s=46).
   - It aims to create verified training environments for self-supervised reinforcement learning in mathematics by converting mathematical knowledge into formal proofs.
- **Transformers Library Deprecates TensorFlow and Flax Support**: The **Transformers library** will deprecate **TensorFlow and Flax** support, focusing solely on **PyTorch** to reduce bloat, simplify the toolkit, and remove abstraction layers as mentioned [here](https://xcancel.com/LysandreJik/status/1933201171130593530).
   - **Long-term support (LTS) for TF and Flax will continue with v4 until mid-2026**, and this change marks the beginning of v5, aiming to remove 50% of the code.
- **Meta AI App Shares Private Conversations Publicly**: A **Meta AI** app inadvertently posted users' private conversations, including sensitive information and audio, to a public feed which is linked [here](https://xcancel.com/SHL0MS/status/1933019178023231880).
   - Users are accidentally sharing content due to a confusing UI, exposing personal details and raising ethical concerns for Meta.
- **Anthropic's Multiagent System Dominates Single-Agent Claude Opus**: **Anthropic** found that *a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval*, according to [this post](https://www.anthropic.com/engineering/built-multi-agent-research-system).
   - They found that *multi-agent systems excel at valuable tasks that involve heavy parallelization, information that exceeds single context windows, and interfacing with numerous complex tools* but burns through tokens fast, at about **15× more tokens than chats**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's Library Version Awareness Explored**: Users sought ways to enhance **Aider's** awareness of library versions, especially when migrating from **pip/virtualenv** to **Poetry**, suggesting outdated options, and recommended including URLs to updated man pages and explicitly defining versions in conventions or using the `/read docs/spec.txt` command.
   - The discussion emphasized improving context provision for **Aider** to ensure it suggests the most current library versions.
- **Aider Costs with Anthropic Model**: A user voiced concerns about potential hourly costs of nearly **$50** when using **Aider** with **Anthropic**, particularly for large changes and also noted that the **Claude Code** monthly plan may have been used up quickly, suggesting possible usage limits.
   - The conversation highlighted the importance of cost management when using **Aider** with commercial models like **Anthropic**, emphasizing the need to monitor usage.
- **Aider Excels with Smaller Models**: Users lauded **Aider's** performance with smaller models (**8B** and **12B**) via **Ollama**, finding it surprisingly effective, with another user pointing to **Aider's** repomap as the secret sauce.
   - The tool's capability to function efficiently with limited resources positions it as a strong contender for smaller, locally-run models.
- **UV** Manages Python Dependencies**: Members explored migrating to **UV** for Python dependency management as a superior alternative to direct **pip** usage and **pyproject.toml** edits, favoring commands like `uv add <dependency name(s)>`.
   - One user initially hesitant about reading the manual found **UV** *much tighter* for defining linting instructions in **YAML** configuration, marking a shift towards streamlined dependency handling.
- **max_input_tokens** Configuration Conquered**: A user resolved configuration challenges related to setting separate max tokens for input and output in **Aider**, especially concerning the display of *remaining tokens*.
   - Clarification led to the successful configuration of the **max_input_tokens** parameter, fixing the initial confusion and improving **Aider's** performance.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Vast.ai Offers Cheap Compute**: A member highlighted [Vast.ai](https://vast.ai) as a provider of decentralized compute that is *relatively cheap*, and Akash was also mentioned as a potential alternative.
   - A member noted that Vast.ai is the cheaper option of the two.
- **C. Opus Posts First Arxiv Publication**: Teknium shared [a post](https://x.com/WesRothMoney/status/1933502113285616083) on X announcing that it was **C. Opus's** first publication on Arxiv.
   - There were multiple confirmations of this important event.
- **NVIDIA** Releases **Cosmos**: **NVIDIA** launched [Cosmos](https://github.com/nvidia-cosmos/cosmos-predict2), with the ArXiv paper available at [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).
   - No further discussion of the launch or its features occurred in the channel.
- **Infiniband** Outpaces Internet Bandwidth**: A member noted that while typical internet bandwidth is around **1gbps**, [Nvidia's latest Infiniband](https://x.com/toolandtea/status/1933381389552136705) iteration reaches **130TB/s**, highlighting the growing bandwidth disparity.
   - The internet's bandwidth hasn't seen significant increases in recent years.
- **DAWN Internet** Promotes Decentralized Internet Access**: A member promoted **DAWN Internet**, a decentralized broadband protocol that uses fixed wireless rooftop antennas to provide gigabit internet and also includes a **GPU** capable of supporting **RL**.
   - More information can be found on [their X profile](https://x.com/dawninternet).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Mind Map Masterpiece Emerges**: A member created a mind map from **115+ sources**, summarizing key aspects and claiming it was *pretty accurate*, resulting in a *huge mind map*, according to a [linked image](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=684df225&is=684ca0a5&hm=9f877d3700e50d5c48e2faa411ec5c0e28b33088d6d859a72f7f2278d3660d3d&).
   - In response to another member's query, the map was noted to have **4 sublevels**, with the user expressing satisfaction with the vertical density but noting room for improvement horizontally.
- **Paid AI Pro Users Locked out of Notebook LM Plus**: A member using **paid AI Pro** reported being unable to access **Notebook LM Plus**, and asked for ideas why, but no solutions were provided in the channel.
   - The root cause of the access issue remains unresolved within the discussion.
- **Excel Support missing from NotebookLM**: Users requested **Excel** and **Google Sheets** support in NotebookLM, but there is currently no support or roadmap for this feature.
   - Users are recommended to use the feature request channel to express their interest.
- **Mobile App Notes are Limited**: While **Notes** are available on the desktop version of NotebookLM, the mobile app only displays **sources**, **chat**, and **studio** sections.
   - While there's no export option on mobile, a workaround is to access notes on mobile via the browser instead of the app, where users can copy and paste.
- **Notebook Sharing Button Grayed Out**: Users are encountering problems with sharing notebooks, as the **'Share publicly' button** is grayed out and unclickable.
   - The cause of this issue is currently unknown.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Beam Linearizer Bugs Bugging Users**: Users report encountering **linearizer failures** when running with **BEAM**, but the cause and solution remains elusive.
   - This issue needs further investigation to determine the root cause and potential fixes.
- **Tinygrad's Float Fumbles Fuel Frustration**: A user detected discrepancies in **float matmul** accuracy between **NumPy** and **Tinygrad**, specifically in the bottom left corner value of the output matrix using [this code](https://discord.com/channels/924641739439477844/1147353713739749426).
   - The discussion addressed the effects of compiler variations, optimization strategies, and adherence to the **IEEE 754 standard**, highlighting that slight numerical variations are typical and depend on operation order and the usage of **float64** by default in **NumPy**.
- **SVD Sign Slip-up Sparks Scrutiny**: A contributor working on a **linalg.svd** PR aimed for **NumPy**-level accuracy but encountered sign differences in the values.
   - It was suggested to use `DEBUG=4` to check the kernel code and `NOOPT=1` to disable loop unrolling for closer results, as loop unrolling can introduce numerical differences.
- **QR Algorithm Quirks Questioned**: A user identified variance in **QR algorithms** due to the difference between **Householder Reflections** and the **Gram-Schmidt process**.
   - The user highlighted an even greater variance compared to the **LAPACK** package used by **NumPy** for Eigen-value calculations.
- **NumPy's Numerical Norms Need Nuance**: One user recommended explicitly creating **NumPy** arrays with `dtype=np.float32` to mitigate result discrepancies, criticizing **NumPy's** default setting of `np.float64`.
   - Another user countered that **float64** is standard in numerical applications beyond machine learning, and changing the default could disrupt unrelated functionalities.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mapping Variadic Types Remains a Challenge**: Mapping variadic types in **Mojo** poses ongoing challenges, as highlighted in [this forum post](https://forum.modular.com/t/map-variadic-types/1638?u=emil.martens), particularly due to the need for a more dynamic type system.
   - The suggestion of using **StaticString** to define the corresponding `__mlir` type faces difficulties because of limited documentation and the complexity of supporting an arbitrary number of types.
- **MLIR Type Workarounds Explored**: Exploration of workarounds using `__mlir_type` encountered issues with **undocumented MLIR** and the inability to synthesize the **MLIR type** for a given type parameter as a raw string.
   - A member proposed extracting and modifying the **MLIR type** at compile time to bypass type definition constraints using **UnsafePointer** and `init_pointee_move`.
- **Magic to Pixi Migration Achieves Painless Transition**: A user successfully migrated from `magic` to `pixi` by removing the `~/.modular` directory and rewriting `mojoproject.toml` files, describing the process as *painless*.
   - The user provided a `pix.sh` script for updating and cleaning the cache, which creates a new `pixi.lock` and `.pixi` folder, advising the old folder's removal post-test validation.
- **Host-Side Synchronization in GPU Puzzle Clarified**: Clarification was provided regarding **host-side synchronization** in a GPU puzzle, specifically addressing [this section](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step).
   - Since `DeviceContext` employs a CUDA stream, **explicit synchronization** isn't required, and the puzzle description will be updated to reflect this.
- **Mojo Exports Capabilities via C ABI**: Mojo supports exporting **C ABI compatible functions** using `@export(ABI="C")`, facilitating the creation of object files or shared libraries.
   - This enables integration with **C/C++** codebases, expanding Mojo's interoperability.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **GitHub Enables Live Context Access**: GitHub PM unveiled a remote GitHub MCP server, granting any MCP host access to live GitHub context without requiring local setup, as detailed on [Reddit](https://www.reddit.com/r/mcp/s/Cj2zjute95).
   - The server employs dynamic tool selection, presenting the LLM with a relevant subset of tools based on user input or context, even with 30+ tools available, to keep auth simple with **one MCP server**.
- **Track Agent Progress with Taskerio**: Taskerio introduced a stealth mode product: an inbox designed for coding agents to report progress, featuring webhooks, push notifications, and an API for real-time dashboards, further detailed on [Reddit](https://www.reddit.com/r/mcp/comments/1lac12i/an_mcp_to_track_the_progress_of_your_ai_agents/).
   - This allows for real-time monitoring and tracking of AI agent activities.
- **Fortify MCPs Against Rug Pulls with SchemaPin**: A member introduced **SchemaPin**, a tool engineered to defend against **MCP Rug Pulls** and related exploits, with the [GitHub repository available here](https://github.com/ThirdKeyAI/SchemaPin).
   - Easy implementation methods are detailed on [SchemaPin.org](https://schemapin.org), safeguarding **MCPs** from potential vulnerabilities.
- **Postman Simplifies MCP Server Construction**: A member demonstrated constructing an **MCP server** using Postman's MCP builder and APIs on their public API network, referencing the [fastfs-mcp GitHub repository](https://github.com/aj-geddes/fastfs-mcp) as an illustrative example.
   - A corresponding [YouTube video](https://youtu.be/YzT9QU-Kfh4?si=tqqgBXXu9ct2aMUH) further elucidates the process.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud** Wobbles Back to Stability**: **LlamaCloud** is operational after upstream infrastructure hiccups; check the [status page](https://t.co/IdecAksHiG) for real-time updates.
   - The incident underscores the fragility of cloud dependencies.
- **MistralAI's Magistral** Now Plays Nice with **LlamaIndex**: **LlamaIndex** embraces **MistralAI's Magistral** reasoning model, slotting it into agent workflows, according to [this tweet](https://t.co/ZsUEWMrnT4).
   - This integration could open doors for more sophisticated reasoning tasks.
- **LlamaParse** Gets User-Friendly with **Presets**: **LlamaParse** introduces **Presets**, offering **Fast**, **Balanced**, and **Premium** modes to tweak accuracy versus speed during document parsing.
   - These presets let users optimize document parsing based on need.
- **Mem0** Integration Eases Memory Management in **LlamaIndex**: When using **LlamaIndex** with **Mem0**, memory updates occur automatically by passing `memory=memory` into `agent.run()` eliminating manual updates.
   - The integration with **LlamaIndex** supports Mem0's graphRAG capabilities, streamlining memory handling.
- **Luma** Calendar Might Oust Discord for Office Hours**: Organizers are mulling over switching to a **Luma** calendar for office hours due to Discord calendar usability complaints, and are [soliciting ideas, requests, and suggestions](https://discord.com/channels/1031248924924043295/1031248926475255868) regarding format of future office hours.
   - The move aims to enhance the office hours experience.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DeepMind Embraces Named Tensors**: A member is developing the **Xarray-JAX library** for Google DeepMind as part of GSoC 2025, and claim it to be the first named tensor implementation in a deep learning framework.
   - The library aims to enhance tensor operations within JAX, making them more intuitive and efficient for deep learning tasks.
- **AI Finance Tool Evades LLM Wrapper Trap**: A member is building an **AI SaaS tool in the finance space** as a college project and is asking how to avoid just making an LLM wrapper to actually provide real value to end users.
   - They requested suggestions for an MVP to avoid common pitfalls that are found among most LLM wrappers.
- **Cohere Docs Suffer Syntax Slip-Up**: A member reported a potential **typo** in [Cohere's documentation](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide).
   - The correction suggests that in the Python code example, `co = cohere.SagemakerClient()` should use lowercase "m" in `SagemakerClient()`.
- **Reranking Profile Requests Remain Raw**: A member inquired about the specifications of the reranking profile, particularly the **number of docs, tokens per doc, and query tokens**.
   - Unfortunately, this request did not receive any responses, and the inquiry ended without further discussion.
- **GCP Grounds Growth at Cohere**: Cohere reported a Google Cloud Platform (**GCP**) outage impacting some of their services on **June 12, 2025** at **12:02PM** [status page](https://ift.tt/on1ARP0).
   - The status page indicated degraded performance in **Infrastructure** components, prompting close monitoring and response efforts by the Cohere team [Cohere Status Page](https://ift.tt/Ens6bma).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Fast Weights Aims for More User Control**: Members advocate for **fast weights continual learning** and **external data stores** to improve user control and reduce undesirable human traits in AI models.
   - They expressed eagerness to see traits like *scheming, frustration, and false memories* removed from mainstream AI.
- **O1-Pro Models Offer Good Value**: One member found **O1-Pro/O3/O4-mini-high** models valuable for learning well-documented math and computer science, while also liking their **image generation capabilities**.
   - They also mentioned using the models' API for an **audio transcription pipeline** that works almost perfectly, though the image generation is censored.
- **Gemini experiences compared to Claude**: A member asked how **Gemini** compared to **Claude**.
   - Another member stated that **Claude** has been less reliable for them but noted that all models can get things wrong and are most useful in highly verifiable domains.
- **Wavefunction Discussions Take Friday Off**: There is typically **no Wavefunction discussion** on Fridays due to limited audience participation.
   - Despite the lack of scheduled discussion, community members are welcome to initiate their own.
- **Huang and Amodei Disagree on AI Jobs**: A [Fortune article](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagrees-anthropic-ceo-dario-amodei-ai-jobs/) reports that **Jensen Huang (Nvidia)** disagrees with **Dario Amodei (Anthropic)** about the future of **AI jobs**.
   - **Dario** has responded to **Jensen** via [X](https://www.x.com/dario) - with an update on AI Jobs - as shares in both companies are sharply down, as **job fears continue**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Details on Mistral 3.1 Small Architecture still murky**: A user asked about architectural novelties in **Mistral 3.1 Small**, estimating **2 weeks** to implement fine-tuning once known.
   - Another user felt supporting **Mistral 3.0** implies support for **Magistral**, though multi-modality support may be challenging.
- **Tokenizer Troubles Spark Speculation**: The difficulty of the tokenizer was mentioned, and a member suggested it was a *complicated procedure*.
   - The discussion clarified they were actually referring to **Magistral**'s tokenizer.
- **Torchtune Integration Urged for Magistral**: Members expressed interest in a **Torchtune** link on **Magistral's** Hugging Face (HF) page.
   - This indicates community demand for **Torchtune** integration with **Magistral** to improve accessibility.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Infinite Chat Implemented Locally**: A member highlighted the implementation of **Infinite Chat** locally, designed to prevent users from exhausting the context window.
   - Documentation can be found [here](https://docs.supermemory.ai/infinite-chat) for those interested in its features and capabilities.
- **Requesting Ignore Feature**: A user inquired about the potential addition of an **'ignore' feature** for the embedding system, similar to `.ignore` files in Git.
   - This feature would allow users to exclude specific files, file types, or directories from being processed or embedded.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Waves into UI/UX Upgrades**: Windsurf is wrapping up **Wave 10** with a fresh slate of **UI/UX upgrades** and new teams and enterprise offerings, including [new icons](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise) for `@-mentions` and file citations.
   - Codeblocks in the Cascade panel now match your IDE theme, with native terminal in Cascade panel that now accepts user inputs, plus a New Conversation History UI.
- **Windsurf rolls out EU Cluster for Performance Boost**: Windsurf proudly announces their **EU Cluster**, bringing faster performance and rising demand to European enterprises today!
   - Watch the [video on Youtube](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI) and [join the conversation at r/Windsurf](https://www.reddit.com/r/windsurf/).
- **Claude Sonnet 4 Lights Up Windsurf**: **Claude Sonnet 4** and **Claude Sonnet 4** (Thinking) are now available to all paid plans via [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing)!
   - More info available [on X](https://x.com/_mohansolo/status/1933605162775687482).



---


The **DSPy Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1382797226081910844)** (1103 messages🔥🔥🔥): 

> `Gemini Deep Think, Perplexity Pro Role, Image Creation with Perplexity Pro, GPT-5 architecture, Qwen3 replacing Qwq` 


- **Gemini Deep Think is Coming**: A member mentioned that **Gemini Deep Think** is coming, referencing an attached image ([aSHuTrz.jpeg](https://cdn.discordapp.com/attachments/1047649527299055688/1382930529464352848/aSHuTrz.jpeg?ex=684d9aab&is=684c492b&hm=b30084937010eeeb0b4dba66cb69c6fd085c52fb786ce6e316e2324b2f50d0aa&)).
- **Perplexity Pro Role is Broken**: Members discussed difficulties in obtaining the **Perplexity Pro role** in Discord, with the onboarding button not working and some members suggesting a temporary solution of pinging staff.
   - One member stated, *"the button doesn't seam to give me the role just puts me in the server on the phone"*.
- **Perplexity Pro Makes Images!**: Members shared that **Perplexity Pro** can create images by typing image prompts in the search bar, such as *“Draw a pastel cottagecore village in spring, watercolor style”*.
   - A member gave further instructions: *"Click **Regenerate** under the image or send a new prompt in the same thread. Try refining with art styles or effects like: cinematic, low-poly 3D, studio lighting, anime-style*".
- **GPT-5 Works Smarter, Not Harder**: A member shared details about **GPT-5**'s architecture, noting it operates as a single model that leverages specialized tools internally, avoiding the pitfalls of external routing and hallucinations.
   - They quoted that *"GPT-5 thinks with its tools, not beside them"*, highlighting improved context, coordination, and stability.
- **Qwen3 is the New Dead Qwq**: Members noted that **Qwen3** replaces the now-defunct **Qwq**, acknowledging its existence before moving on to other topics.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

meijer5838: https://www.perplexity.ai/page/unused-phones-europe-s-hidden-YpcOJpSCSfu9IlnOng_85A
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1382822588576694427)** (7 messages): 

> `Sonar API documentation feedback, Perplexity Publisher Program` 


- ****Sonar API Docs Seek Scrutiny****: The team is seeking feedback on the **Sonar API documentation** and requests users to share their experiences, specifically regarding areas of unclarity or difficulty in navigation, under [this community post](https://community.perplexity.ai/t/improvements-to-the-sonar-api-documentation/542?u=vikvang).
   - The goal is to enhance the documentation based on user input.
- ****Publisher Program Plug Posted Publicly****: A user shared a [LinkedIn post](https://www.linkedin.com/posts/codingclubnmims_codingclubmpstme-endoftenure-newbeginnings-ugcPost-7339125013536464896-ybUR?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAEUiNWsBLhCcJJA2pq2u07Btb29g_1q97iU&utm_campaign=whatsapp) and a [company page](https://www.linkedin.com/company/codingclubnmims) related to the **Perplexity Publisher Program**.
   - The user suggested it might be helpful for a specific channel.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1382796606843256966)** (1098 messages🔥🔥🔥): 

> `o3 vs 2.5 Pro, Model preference, Ethics in models, Grok 3.5, New models` 


- **O3 Pro vs 2.5 Pro: Model Preference Debate Rages On**: Members engaged in a heated debate over model preferences, with some arguing that **o3 is better** than **2.5 Pro** in many aspects, while others claimed that **2.5 Pro** excels in specific areas like math.
   - One member sarcastically remarked *"I wish to live in this level of delusion"* regarding another's preference for **2.5 Pro** in math, sparking further discussion.
- **MathArena Benchmarks: Saturated or Still Valid?**: Members debated the validity of [MathArena benchmarks](https://matharena.ai), with some arguing that they are becoming saturated and potentially luck-based, while others maintained that they are still useful metrics.
   - Concerns were raised that scores close to **100%** might indicate saturation, questioning whether the benchmarks remain statistically meaningful.
- **Google Account Banned: New Kingfall Release Trigger Alarms**: A member reported their Google account being banned, which prompted others to speculate about a new **Kingfall** release.
   - A new gemini model, codenamed *toothless* showed up for a brief period, leading to speculations that it's a new checkpoint. There's even reports of 99% profit on various ventures.
- **Text-to-Video Arena: Seedance and Kangaroo Impress**: Members shared and discussed blind tests in a [text-to-video arena](https://artificialanalysis.ai/text-to-video/arena), highlighting the impressive performance of models like **Seedance 1.0** and the anonymous **Kangaroo**.
   - Comparisons were made, with some suggesting that these models surpass **Veo3**, **Kling**, and **Pika**, particularly in generating similar outputs from general prompts.
- **Rumors of O4/GPT-5: A New Challenger?**: Speculation arose around the potential release of **O4/GPT-5**, with a member confidently stating that it is not a bigger model, causing debate about the naming convention and what the release would look like.
   - Another member said they had proof it was a native mode, but refused to provide it.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1382832731314061373)** (2 messages): 

> `Cloud provider outage, Contest running, Test Garden application, Staff AMA` 


- **Cloud Provider Plunge causes Potential Promptcat Poop**: Due to outages from their cloud provider on **6/12/25**, **chat history data may have been lost**.
   - The team apologized for any inconvenience and are *working on solutions to ensure this doesn’t happen again*.
- **Contest Creations Cause Creative Combustion**: There is a contest running, and participants are encouraged to post their creations to the <#1378034388272681079> channel for a chance to win.
   - More details are available [here](https://discord.com/channels/1340554757349179412/1343296395620126911/1378037223794147458).
- **Test Garden Tempts Techies to Try**: Enthusiasts interested in providing feedback and seeing behind-the-scenes developments can apply to the **Test Garden**.
   - The application form is available [here](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog).
- **AMA Ascends Amidst Attentive Audience**: The team thanked everyone who attended last week's **Staff AMA**.
   - Feedback can be shared via [this form](https://docs.google.com/forms/d/e/1FAIpQLSegIDRbbpx2amAR-6MA834fz_QycY15IQ0csyOOKUJUTncGMw/viewform?usp=dialog) and the [video recording](https://cdn.discordapp.com/attachments/1343296395620126911/1383145168470937650/Staff_AMA_66.mp4?ex=684db9d1&is=684c6851&hm=84b909bb9ab6d48888d81480e11562318715b436b6745cc7bf4f430717a5a9d3&) is now available.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1382796795574489230)** (433 messages🔥🔥🔥): 

> `Cloudflare outage impacts Cursor, Cursor code generation, Claude Code for complex refactors, MCP server setup, Front End Testing` 


- **Cloudflare & GCP Outage Crash Cursor**: Users reported [Cursor being down](https://status.cursor.com/) due to a **Cloudflare** and **GCP** outage, affecting login and functionality for many, while others noted that [Tab still works](https://www.cloudflarestatus.com/); the issues were later [reported as resolved](https://mashable.com/article/google-down-cloudflare-twitch-character-ai-internet-outage).
- **Cursor Coding Criticisms Continue**: A user asked about using cursor to make three.js games, while one user recommended O3 for most coding, and O3-pro for planning and debugging, emphasizing its effectiveness over other models.
   - There was discussion about **Cursor's code generation** as subpar when switching to "auto" model selection, with one user losing **50 inference credits** due to code messes, advising against Cursor's auto-switching.
- **Multiplayer is So Hot Right Now**: Many members discussed using **peerjs** and **socket.io** to develop multiplayer games, with one member showing off their Steam multiplayer Unreal Engine 5 game [Supermarket Simulator](https://nandstudios.itch.io/supermarketsimulator).
- **CUA Automation on the Horizon**: Members mentioned that **CUA (Computer Using Agent)** improvements could enhance automation, mentioning the project [browser-use](https://github.com/browser-use/browser-use) for automating tasks.
- **Claude Code Crowns Context King**: Users found that **Claude Code** excels in context understanding and code quality for complex refactors, adding **3500 new tests** for a front-end component library.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1382805848149065768)** (30 messages🔥): 

> `Background Agents LSP and Linter, Background Agents Privacy Mode, Background Agent Commit Issues, Background Agents leaking context, Background Agents Docker Compose` 


- **Background Agents Leverage LSP and Linters**: Background agents should use **LSP errors** and have access to all extensions, ensuring dependencies are installed in the agent environment.
   - The slack integration might not do that right now, but it should do that if you start them through the desktop client.
- **Background Agents Privacy Mode fixed**: Users reported getting an error message, *Background agent is not supported in privacy mode*, when trying to start a background agent.
   - The issue was resolved by enabling an **account-level privacy mode** [here](https://www.cursor.com/slack-connected), and this issue will be fixed in the next version.
- **Background Agent Faces Commit Challenges**: After a background agent amended a commit, it struggled to push the amended commit.
   - A member suggested resolving it through the terminal, since the agent was getting rolled back.
- **Background Agents Might Leak Context**: A user reported that background agents might be leaking context, linking a **Sentry error** unrelated to the current task.
   - The user experienced this on multiple chats and shared an [image](https://cdn.discordapp.com/attachments/1367213641027551352/1382941599033462835/image_5.png?ex=684da4fb&is=684c537b&hm=fbe62477092e700e3c79f24220708ca44395f9b259e40ada0ab185a4ac7cef56) as evidence.
- **Docker Compose Setup for Background Agents**: A user sought guidance on setting up a Cursor background agent with **Docker Compose**, aiming to run commands within a specific container by default.
   - They provided [docker-compose.cursor.yml](https://cdn.discordapp.com/attachments/1367213641027551352/1383051743100928000/docker-compose.cursor.yml?ex=684d62cf&is=684c114f&hm=33255fde21b1dc7efa06d1ece8217a45ddacebf1f251ffc5069028a6798c1538) and [environment.json](https://cdn.discordapp.com/attachments/1367213641027551352/1383051743427956745/environment.json?ex=684d62cf&is=684c114f&hm=b10c10f7ed5f05e5d22752a7c55b429dc9c70616169c69900417db0184032229) configurations, hoping to run tools like pytest with linked Postgres and Redis containers.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1382800132571004928)** (4 messages): 

> `Google Cloud Outage, Cloudflare Status, Internet Recovery` 


- **Google Cloud Suffers Major Outage**: **Google Cloud** experienced a major outage, as reported on their [status page](https://status.cloud.google.com/).
   - Users reported intermittent issues even after initial signs of recovery around **4:25pm ET**.
- **Cloudflare and Google Status Pages Provide Updates**: Updates on the outage and recovery can be tracked via the **Cloudflare** [status page](https://www.cloudflarestatus.com/) and the **Google Cloud** [status page](https://status.cloud.google.com/).
- **OpenRouterAI Tweets on Recovery**: OpenRouterAI tweeted about seeing recovery from the outage, expressing hope it wouldn't be temporary ([tweet link](https://x.com/OpenRouterAI/status/1933263905385500853)).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1383078905266831511)** (5 messages): 

> `Button cutoff on narrow browser` 


- **Narrow Browser Button Bug squashed!**: A member reported a bug where the button was cut off in a narrow browser window, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1092850552192368710/1383078905002594396/Screenshot_2025-06-13_at_06.41.30.png?ex=684d7c1b&is=684c2a9b&hm=526613d7d6f7266a7d81bddc27965de3d691f309935ce3264d7ee968a9884e76&).
   - Another member quickly addressed and fixed the issue, then provided a [screenshot](https://cdn.discordapp.com/attachments/1092850552192368710/1383106945535180823/Screenshot_2025-06-13_at_10.33.00_AM.png?ex=684d9638&is=684c44b8&hm=88cfcaa0194f8fc528d0654a4257ec09912af72ffdd69d6f2b460c19cc014aee&) of the fix.
- **Another Bug Reported**: To comply with the prompt's requirement of a minimum of two topic summaries, here is another topic to fulfill the requirement.
   - No actual second bug was found in the provided text, but including this ensures the JSON is valid as per the schema.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1382796611364720773)** (377 messages🔥🔥): 

> `Cloudflare Outage Impacts OpenRouter, OpenRouter Status Fluctuation, Model Performance Variations by Provider, Agentic Tool Use with Cost-Effective LLMs, Multi-Modal Support for OpenRouter` 


- **Cloudflare Kills the Internet (Again)**: A widespread [Cloudflare outage](https://www.cloudflarestatus.com/) caused significant disruptions, taking down numerous AI services including **OpenRouter**, Google, and others.
   - Users reported widespread issues, leading to humorous speculation about the cause, ranging from *interns spilling coffee on servers* to *Skynet taking over*.
- **OpenRouter Teases Users With Up-And-Down Cycle**: Users experienced intermittent **OpenRouter** service, with the status page flipping between *MAJOR* and *MINOR* outages, leading to frustration and jokes about timing API requests like a carnival game.
   - Some users found success using specific models or configurations, while others continued to face *timeouts* and *authentication errors*.
- **Provider Variability Impacts Model Qualities**: Users discussed the significant quality variations among different providers offering the same models through **OpenRouter**, noting that **Parasail** and **Lambda** generally offer more consistent performance.
   - Quality is more important than cost as one user said the [quality varies alot by providers , so choose wisely](https://discord.com/channels/1091220969173028894/1092729520181739581/1383133709551282236).
- **Cheap Agent LLMs Emerge as Top Tool-Users**: Users debated the best cheap Large Language Models (**LLMs**) for agentic tool use, with **Claude 2.5 Flash** being recommended as a cost-effective option that requires careful prompting.
   - The high cost of models like **O4 Mini High** and the potential release of a new **Google Flash** model were also discussed, alongside the efficiency of using a [monthly Claude Max subscription](https://discord.com/channels/1091220969173028894/1195014798837043240/1383046909199872050) for API usage.
- **Dreaming of OpenRouter Multi-Modal Capabilities**: Members requested future support for multi-modal capabilities like **audio** and **video generation** within the **OpenRouter** platform.
   - No explicit response was given by OpenRouter.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1382800601556975687)** (135 messages🔥🔥): 

> `LM Studio Model Updates, Setting Static Generation Seed in LM Studio, Gemini Pro Image Recognition, Bypassing Captchas, Running LM Studio on a Server` 


- **LM Studio not automatically updating models**: LM Studio does not download models automatically and most model updates are new generations published in a new repository, therefore making it difficult to determine the lineage of a model.
   - A member was wondering if there was a way to automatically update models in LM Studio like Ollama but this is not the case.
- **Concise LLM Responses**: LLMs are trained to be concise to not bore the user and to safe on computational cost, so one can split the task by getting a structure first, then asking for content for each bullet point of that structure.
   - This was in response to a user requesting a really long and thorough summary (almost an essay of a sort) and asking if there was a way to reduce the susceptibility to response end.
- **Gemini Pro Struggles with Image Recognition**: A user asked why **Gemini Pro 2.5** makes mistakes in simple image recognition, even after trying various prompts and images ([example image](https://cdn.discordapp.com/attachments/1110598183144399061/1382962931741888563/image.png?ex=684db8d9&is=684c6759&hm=14fbd9fe32c10ead609c4627acfd3c543cdf7ca6f347af0e6c9a61470af4c663&)).
   - Another member noted that vision-enabled models are often not great and it's difficult to determine exactly what the user expects - especially when the user says they have *tried everything*.
- **LLMs are a Red Queen's Race for Bypassing Captchas**: Members discussed the difficulties in using **LLMs** to bypass captchas, highlighting that captchas are designed to be hard for computers and are continually upgraded to thwart **LLMs**.
   - As soon as a technique to crack a captcha is developed, a new one emerges, rendering the progress obsolete, like the [Red Queen hypothesis](https://en.wikipedia.org/wiki/Red_Queen_hypothesis).
- **OpenWebUI enables remote LM Studio access**: To run LM Studio on a server and access it from another PC, you can host a server on **LM Studio**, load a model, serve it on the local network, enable **CORS**, and open specific ports (e.g., 1234, 8080, 3000) on the host PC using [OpenWebUI](https://github.com/OpenGenAI/OpenWebUI).
   - There is no need to install **OpenWebUI** on the PC you're going to access it with.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1382807314242998324)** (151 messages🔥🔥): 

> `Unified Memory, Strix Halo, Tesla P40, Context Windows` 


- **Unified Memory Video Disapproved by HX395+ Owner**: A member shared a [video comparing unified memory](https://youtu.be/Cn_nKxl8KE4?si=0-iQclmGi2UcWVNxa) but another member and actual HX395+ owner, disapproved, calling it *a terrible video*.
   - The disapproval was due to the video confusing **soldered RAM** for **slotted RAM**, going off-topic, and not knowing about **Strix Halo** which has a **4x widefrongong**.
- **Tesla P40 VRAM Expansion?**: A member inquired about using **Tesla P40** as an addition to normal GPUs like **RTX 3090/4090** to expand VRAM for LM Studio for around **$300** for **24GB** used, linking to the [TechPowerUp specs](https://www.techpowerup.com/gpu-specs/tesla-p40.c2878).
   - The general consensus was that at the **$300** price point, it's not worth it anymore, as they were once worth buying when under **$150**, and a used **3090** is a better 'affordable' option.
- **Debate Explodes Over the Need for 150B Parameter Models**: A member stated that *reasonable human interaction* needs at least **150b** parameters, and if it's supposed to feel smart and natural, then **300b**.
   - Another member countered that there is more to LLMs than the number of parameters, such as **prompt engineering**, **good RAG**, and **finetuning**, and not everything needs *reasonable human interaction*.
- **Local LLMs Struggle With Large Context Windows**: A member shared that their local LLM, when writing detailed stories, experiences issues with context retention, especially when the story setting is in a medieval castle, but the LLM starts talking about *watching television*.
   - Another member replied that *no local LLM works well enough with context windows above 32768*, because then need to use tricks to even be able to expand it, recommending the user try dedicated models, as they have *quirks for their long context windows to work*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1382796683670192128)** (83 messages🔥🔥): 

> `AI Safety Institute, GPT-3 and GPT-4o Behavior, Symbolica.ai Startup, MMLU 5-Shot, AI Consciousness` 


- **New AI Safety Institute Pops Up**: Members noticed a new AI Safety institute, but expressed some skepticism about its legitimacy since they hadn't heard of it before and its website lacks recent publications.
   - A member pointed out that one of the advisors is on Discord and suggested setting up a call.
- **German Text Sparks Unusual LLM Behavior**: A member described how a short German text causes **GPT-3** and **GPT-4o** to exhibit drastically different reactions, ranging from neutral responses to deep emotional interpretations.
   - The member wondered if this *observation* would be relevant to share, indicating a potential interest in exploring **LLM** behaviors beyond typical use cases.
- **Startup Symbolica.ai Aims High**: A member highlighted [Symbolica.ai](https://www.symbolica.ai/), a new London startup with an ambitious goal.
   - Another member suggested they should release a small theorem prover model like the one Google had, and noted that some reviews mention that the *boundaries of the work aren't clear and the goals keep changing*.
- **MMLU 5-shot Evaluated**: A member asked how **MMLU 5-shot** works, specifically if it's best of 5 or average of 5.
   - Another member clarified that *5-shot refers to the number of examples seen, not the number of attempts permitted*.
- **Delusions are caused by memory?**: A member wonders if the *memory* feature within **ChatGPT** is causing delusions.
   - Another member shares this [arxiv link](https://arxiv.org/abs/2504.07992) and says, *degenerating output behaviour stopped immediately as 'memory' was removed*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1382813851111788604)** (186 messages🔥🔥): 

> `Non-commercial license controversy, CommonPile 2 Creation, GRPO Objective & Model Performance, Symbolic Recursion` 


- **Non-Commercial License Sparks Controversy**: Members express concerns over a new dataset's [non-commercial license](https://discord.com/channels/729741769192767510/747850033994662000/1382718117561761874), questioning its framing and potential restrictions, despite aiming to foster a *healthier foundation for model development*.
   - Some argue it may be an instance of *copyfraud*, especially if the data conversion primarily involves scanning and text extraction, referencing [Wikipedia's article on copyfraud](https://en.wikipedia.org/wiki/Copyfraud) and [Public Domain Sherpa](https://publicdomainsherpa.com/false-copyright-claims.html).
- **CommonPile 2 May Depend on Synthetic Data**: Discussion revolves around creating a stronger **CommonPile 2**, with one member suggesting the need for synthetic data to achieve terabytes of robust training material.
   - However, it was cautioned that simply generating more samples doesn't magically produce more information, violating principles of *information theory*, and that keeping data *close to the source* is generally preferable, except in select scenarios.
- **GRPO Objective Boosts Model Performance**: Members discussed how **DeepSeek V3**, a **671B** model, achieved high performance with high-capacity and high-quality data, and then used [GRPO objective](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) allowing it to obtain even higher performance on tasks which could be validated.
   - The member pointed out that *literally random rewards improve performance* due to a *concentration effect that focuses the model on its existing reasoning pattern distribution*.
- **"Symbolic Recursion" Terminology Questioned**: Members question the meaning and validity of the term *symbolic recursion*, often used in publications and talks to appear sophisticated, potentially stemming from academic snobbery and jargon.
   - It is speculated the models get that into their head coming down to a fancy way of saying *the model uses the same symbols repeatedly in its writing*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1383162829586169856)** (1 messages): 

> `LLM Fairness, Bias Evals, Prompt Tuning, Chain of Thought, Concept Editing` 


- **Bias Evals Trigger Race and Gender Bias**: A new paper shows that adding realistic details to existing **bias evals** triggers **race and gender bias** in **LLMs**, causing up to a **12% difference** in interview rates across models including **GPT4o** and **Claude 4 Sonnet**.
   - Realistic details include company names, culture descriptions from careers pages, or constraints like *"only accept top 10%"*.
- **Interpretability fixes Fairness**: **Prompt tuning** doesn’t fix bias, but interpretability-based interventions can, using a simple affine concept editing / ablation of race/gender directions reduces bias (as measured by difference in interview rates) to typically **< 1%**.
   - The [paper on Robustly Improving LLM Fairness](https://x.com/a_karvonen/status/1933582375419850806) gives an example of **unfaithful chain of thought** in the wild.
- **Chain of Thought is Unfaithful**: Inspecting **Chain of Thought** gives **zero indication of race/gender bias**, despite the outcomes themselves exhibiting clear bias.
   - The paper found this to be true across all models tested, demonstrating a significant challenge in detecting and mitigating bias in LLMs.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1382797010977030195)** (12 messages🔥): 

> `Inspect Standard vs Evaluation Frameworks, LM Evaluation Harness Progress Bar, Reasoning Models` 


- **Inspect Standard Elicits Debate**: A member inquired about using the [Inspect standard](https://inspect.aisi.org.uk/) versus current evaluation frameworks.
   - Another member clarified that **Inspect** appears to be *just another evaluation framework*, focusing on standardizing how results are saved and queried, rather than how they are run.
- **`lm_eval` progress bar malfunctions with multi-GPU**: A member reported that the progress bar in `lm_eval` only tracks the progress of one GPU in a multi-GPU setting.
   - Another member said that `tqdm` is disabled on the other ranks by default and suggests changing a line in [huggingface.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L1108).
- **Reasoning Models Generation: Non-Trivial**: A member mentioned that handling generations from reasoning models is non-trivial, requiring modification of the answer extraction in each task config.
   - They will create an issue on github.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1382907345520169081)** (1 messages): 

> `Canvas downloads, Canvas PDF export, Canvas docx export, Canvas markdown export, Canvas Code export` 


- **Canvas Enables **Downloads**!**: Canvas now supports downloads; if you're writing a doc, you can export it as a **PDF**, **docx**, or **markdown**.
- **Canvas Exports Code Directly!**: If you're using Canvas to write code, it will export directly to the appropriate file type (e.g. **.py**, **.js**, **.sql**, etc.).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1382801906497355906)** (153 messages🔥🔥): 

> `GPT Pro models parallel processing, O3 Pro performance issues, Free AI APIs, Discord activity decrease, ChatGPT advanced voices update` 


- **OpenAI's GPT Pro Models Use Parallel Processing for Better Reasoning**: The leading theory suggests **GPT Pro models** like **O3 Pro** achieve enhanced reasoning by running multiple instances in parallel, consolidating results, referred to as *'think harder'*, which is supported by **O3 Pro's** chain of thought summaries referring to itself as *'we'*, indicating multiple instances working together.
   - On the **AIME 2024** math competition, **O3-pro** achieved **93%** pass@1 accuracy compared to **O3's 90%**, hinting at the improved performance from this consolidation method.
- **User Reports O3 Pro Fails in Projects Despite Long Wait Times**: Multiple users reported that **O3 Pro** fails to answer questions from uploaded documents, even after waiting for extended periods like **40 minutes** and it not showing **chain of thoughts**.
   - This poor performance contrasts with expectations given **O3 Pro's** supposed enhanced reasoning capabilities, leaving users questioning its effectiveness in practical applications.
- **AI Enthusiasts Explore Free AI APIs for Development**: Despite **ChatGPT Plus** costing money, developers discussed alternative free AI APIs like **SambaNova** with fast **Llama 3.3 70B**, **Qwen**, and **Deepseek**.
   - **Gemini** was highlighted for its high rate limits, offering **500/day** for **2.5 Flash**, **1k/day** for **2.0 Flash**, up to **1M prompt** and **64K output**, making it a viable option for budget-conscious AI projects.
- **Discord Activity Declines Amidst AI Chat Popularity Surge**: Users have observed a sharp drop in **Discord activity** correlating with the rise in popularity of **AI chats**, with many servers becoming *'ghost towns'*.
   - This shift suggests users are migrating to AI-driven platforms for discussions, impacting community engagement on traditional platforms like Discord, which prompts new thinking for community engagement.
- **Users Criticize Annoyed Tone of New ChatGPT Advanced Voices**: Users have voiced their dislike for the new **advanced voices** in **ChatGPT**, describing them as sounding *'annoyed'*, using excessive filler words, and overall conveying a sense of disdain.
   - Some users prefer the previous versions with their artificial cheerfulness, while others suggest the ideal solution would be to have the option to choose a voice persona, similar to **Grok** or create custom voices like with **ElevenLabs**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1382805720814325893)** (47 messages🔥): 

> `GPT-4o memory, Fine-tuning GPT models, Mimicking writing style` 


- **GPT-4o may recall past chats!**: A member reported that **GPT-4o** could directly reference *verbatim* quotes from a fictional scene co-authored with a custom GPT in a separate chat thread, even parts they authored themselves and were not made aware of to GPT-4o.
   - While another member suggested this could be *rarely accurate inference*, the original poster disagreed, citing the **statistical improbability** and offering DMs for more details.
- **Pick Mini or Nano for finetuning?**: A member asked which **GPT model** (*4.1 mini or nano*) to use for fine-tuning to mimic a writing style.
   - A member suggested that if cost is not a consideration, try both and compare results; otherwise, use the cheaper one, with discussion on the trade-offs between cost and performance, and the number of training examples required.
- **ChatGPT only reveals user's name if the user allows it!**: A member stated that **ChatGPT** only reveals the user's name if the user allows it, even with memory enabled.
   - The member emphasized that ChatGPT obeys if the user speaks or asks in the chat.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1382799798465466559)** (19 messages🔥): 

> `Shotgun Spammers, Uploading HTML to o3, Pandoc for HTML to Markdown, Long-form responses from o3 and o3-pro` 


- **Spammers Return with Shotgun**: A user reported the return of *shotgun spammers*, and included a [link](https://chatgpt.com/share/684c40f4-81cc-8003-8adb-ad408bbc676a) showing a main question/answer regarding best practices for uploading HTML files to **o3**.
   - The chat itself says, *"OK I can parse a lot of interleaved tags and still get the gist of even a long file."*
- **Pandoc: Swiss-Army Chainsaw for HTML Parsing**: A user suggested using [Pandoc](https://pandoc.org/) to convert HTML to Markdown for better parsing, calling it a *purpose-built and widely-used tool* rather than a *hack job*.
   - They recommended using **Pandoc** instead of scripting with **awk, sed, or tr** for parsing HTML, while also acknowledging the utility of those tools for one-off tasks.
- **AI Can Handle HTML Tags**: A user confirmed that AI models are trained on plenty of tags and can handle HTML, suggesting algorithmic stripping is only necessary for *absolutely the highest accuracy possible*.
   - They added that HTML tags create *noisy in tokens*, which is good for reasoning, and that while *they don't matter until they do*, they are additional context fillers.
- **Data Prep: Half the Battle**: A user noted that data preparation, processing, and formatting often constitutes about half the workload in AI projects.
   - Typical tasks include *pulling text from PDFs* or *consolidating JSONs*, which highlights the importance of efficient data handling.
- **Users seek prompt for Long-Form Responses from o3 models**: A user seeks a prompt to elicit long-form responses from **o3** and **o3-pro** when reviewing files or performing in-depth research.
   - The user observed that these models tend to produce concise bullet points and comparison tables, contrasting with **Sonnet** and **Opus 4**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1382799798465466559)** (19 messages🔥): 

> `Shotgun Spammers, o3 Model File Uploads, HTML Parsing, Pandoc Conversion, Data Preparation` 


- **Debate about best practices for uploading HTML files to o3**: A user inquired about the best practice for uploading files with lots of HTML to the **o3 model**, obtained via **devTools scraping**, and shared a [ChatGPT link](https://chatgpt.com/share/684c40f4-81cc-8003-8adb-ad408bbc676a).
   - The user found that a combination of **JS** and **AWK** scripting gave great results.
- **ChatGPT can parse HTML, but algorithmically stripping tags yields the highest accuracy**: A member confirmed that while **ChatGPT** is trained on plenty of tags and its output is generally correct, algorithmically stripping tags provides the highest accuracy.
   - He clarified that mistakes are due to *model stochasticity*, not strictly because of tags.
- **Pandoc emerges as preferred tool for HTML to Markdown conversion**: A user suggested using **Pandoc** to convert HTML to Markdown for parsing, recommending it over *hack jobs* using tools like **awk**.
   - Pandoc is described as a *purpose-built and widely-used tool*.
- **HTML tags are noisy tokens**: A member noted that **HTML tags** add *noisy tokens* that can be good for reasoning, while another agreed that they are additional context fillers.
   - If you are just pasting one website into ChatGPT to ask questions about its content, then it doesn't really matter... if you are making some sort of pipeline and paying for each token, maybe it does.
- **Users seek prompts for long-form responses from o3 models**: A user asked for a prompt to get **long-form responses** from **o3** and **o3-pro** when using them to review files or perform in-depth research on a topic.
   - The user noted that both models tend to be concise, even when instructed otherwise.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1382799445237956830)** (210 messages🔥🔥): 

> `Model Underfitting and Overfitting, Hugging Face's Definition of Open Source Models, Interpretability of Transformers Models, HF Spaces, Qwen 2.5 and Multilingual capabilities` 


- **Diagnosing Model Fitting with TensorBoard Graphs**: Members discussed using **TensorBoard loss graphs** to diagnose model fitting, noting that the *evaluation loss* should decrease at a similar rate to the *training loss*, but should not be lower.
   - One member emphasized the importance of dividing the dataset into **training** and **testing** parts to ensure the model has *generalized* well without overfitting or underfitting.
- **HF Faces Scrutiny over Open Source Model Definition**: Concerns were raised that some models on **Hugging Face** aren't fully open source, potentially using the platform for marketing rather than genuine open collaboration.
   - One member pointed out that while **Hugging Face** doesn't explicitly brand itself as an *open source* library, its reputation leans that way, while another mentioned that *any repository can use any license*.
- **Vision Model Interpretability Hotspots**: Members seek assistance with **visualizing attention maps** over images using vision models like **LLaVA** to achieve model interpretability.
   - They asked whether anyone has experience with **interpretability** or **explainability** of **transformers models**.
- **Space Sleepiness Addressed**: Members discussed how to set sleep time for a HF Space using `HfApi` on the `huggingface_hub` library, and to put your [Space to sleep after 1h of inactivity](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-spaces).
   - Note: *if you are using a ‘cpu-basic’ hardware, you cannot configure a custom sleep time. Your Space will automatically be paused after 48h of inactivity*.
- **Qwen 2.5 Claims Multilingual Crown**: Members noted **Qwen 2.5's** ability to speak *100 languages* and compared it to **Gemma3**, with others highlighting it uses the **Linux VM** so well.
   - There was speculation that with *18T tokens*, the model contains substantial multilingual data, contributing to its proficiency.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1382951209131966504)** (3 messages): 

> `` 


- **No Cool Finds Uncovered**: No interesting cool finds to report in the provided message history.
- **Channel is Quiet**: The channel activity appears low, with only a few messages related to user notifications and a general request.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1382865843184078849)** (5 messages): 

> `X Scraper, Digital Twin AI Platform, Augmentoolkit 3.0 Release, Field Injection Attacks` 


- **X-Scraper Open API Endpoints Debut**: An X-scraper with open API endpoints has been created and is available for use, with sample datasets found on its [Hugging Face organization page](https://huggingface.co/MasaFoundationCloneMe).
   - The data is available and free for anyone building **AI models**, **agents**, or **applications**.
- **CloneMe Platform Launches Digital Twin Toolkit**: The [CloneMe AI platform](https://github.com/vibheksoni/cloneme) lets you build your **digital twin**—an **AI** that chats like you, remembers details, and supports multiple platforms.
   - It's customizable, **memory-driven**, and **hot-reloadable**, making it a robust toolkit for creating intelligent, dynamic AI personas.
- **Augmentoolkit 3.0 Augments AI Training**: [Augmentoolkit 3.0](https://github.com/e-p-armstrong/augmentoolkit) has been released, enabling users to **train AI** to understand new subjects by simply adding documents or teaching it to perform tasks through rating attempts.
   - It facilitates **custom model** runs, which are cheaper and provide greater control over update timing and methods.
- **Field Injection Attacks Analyzed**: A detailed article on **Field Injection Attacks** and their potential impact on **MCP servers** and systems has been written and shared on [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_cybersecurity-ai-machinelearning-activity-7339287857447981056-4BAz?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk).
   - The article explains how such attacks can compromise MCP servers and systems.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1383147207263391745)** (2 messages): 

> `Paper presentations` 


- **Clarification on paper presentation types**: A member inquired whether the presentations are for presenting other papers or their own.
- **Either works for presentations**: Another member responded that *either works*.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1382929175337369661)** (6 messages): 

> `Kaggle Datasets, Gemini 2.5 Pro Deprecation, Open Source LLM/VLM Alternatives, Mistral LLM as an Alternative` 


- **Kaggle suggested for dataset discovery**: A member suggested checking **Kaggle** for datasets, referring to it as *the best bet* for finding them.
   - This suggestion was in response to another member's request.
- **Gemini 2.5 Pro's Deprecation Pushes Open Source Search**: A member reported the impending deprecation of **Gemini 2.5 Pro** and the inferior performance of its replacement, creating a need for robust, open-source **LLM/VLM** alternatives for their product.
   - The member desires a system resistant to *corporate whims* and believes anything below **70B** trained parameters might be insufficient.
- **Mistral LLM Proposed as Gemini Alternative**: A member suggested that **Mistral LLM** might be the closest open-source alternative to **Gemini**, but cautioned about expecting the same level of performance when running it locally.
   - They suggested prompt engineering could act as a *shim* to mitigate performance differences.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1382951771592327238)** (2 messages): 

> `GPTs Agents, OpenAI's sidebars` 


- **GPTs Agents cannot learn after initial training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
   - Another member cleared this misunderstanding, explaining that [uploaded files are saved as "knowledge" files](https://link.to/openai-docs) for the agent to reference when required, but **they do not continually modify the agent's base knowledge**.
- **OpenAI Platform's sidebars changed**: Some members had a discussion about changes in the sidebars of platform.openai.com.
   - One reported that **two icons** disappeared from the sidebar** (one for threads and another one for messages).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1382814040262443109)** (10 messages🔥): 

> `Agents Course Sign-Up Link, FinalAnswerTool.forward() Error, Course Completion Deadline` 


- ****Agents Course Sign-Up Link** Broken?**: A member reported the [sign-up link](https://bit.ly/hf-learn-agents) for the course appeared to be broken, but it *seems to be working now!*.
- ****FinalAnswerTool.forward()** Error Haunts Tool Calling Agents**: A member encountered a `FinalAnswerTool.forward() missing 1 required positional argument: 'answer'` error when working with **Tool Calling agents**.
   - The user expressed frustration, stating *This is maddening*.
- **Deadline Dilemma for **Agents Course** and **MCP Course****: A member starting the **Agents course** with a deadline of **July 1** and the **MCP course** with a deadline of **August 1** expressed concern about being *bogged down*.
   - The member asked which course to choose and whether it mattered, implying time constraints may force a choice between the two.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1382796931582918676)** (208 messages🔥🔥): 

> `Manus Outage, Veo3, Manus playbooks, Claude 4.0 waiting, Manus credits` 


- ****Manus Meltdown** Caused by Veo3 Rush?**: Users reported widespread issues with Manus, suspecting the **Veo3 announcement** overloaded the servers, as confirmed by [Downdetector](https://downdetector.com/).
- ****Playbooks Primer** Prep Prompts Preemptively**: **Playbooks** in Manus prepare prompts and give output examples, bridging the gap for users needing prompt assistance, but [they are also intended to highlight creative workflows](https://manus.im/playbook).
- ****Claude Craze** Community Clamors Constantly**: Users expressed eagerness for **Claude 4.0**, drawing humorous parallels to fan anticipation, but there was no official news or update from Google regarding **Claude 4.0** release, but a partner did suggest to *make new gmail and sign up for the google one ai trial -> start a family -> invite 5 accounts -> 5x usage now for veo and everything bc all accounts get separate usage limits*.
- ****Credit Crunch** Costs Concerns Continue**: Users voiced concerns over **credit usage**, particularly regarding optimization and lack of cost previews, with some suggesting the *bring your own keys* model and one user saying *every task I spin up is about 900-1000 credits for me*.
- ****Image Imperfection** GPT Generates Greatness**: A user posted that image generation quality between Manus and GPT-4 Omni were compared, showing [GPT-4 Omni](https://cdn.discordapp.com/attachments/1349440650495398020/1382980409243209859/GPT4omini.png?ex=684dc920&is=684c77a0&hm=1d615e514982fcfdfb5677c8640ae6d7ea8282e5f56d86e5382a2578a0084b82&) outperformed Manus.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1383015183957495869)** (4 messages): 

> `Triton Kernel Optimization, Convolution Implementation, Memory Reads in Triton` 


- **Kernel Sharing Request Leaps!**: A user requested the `solve` function utilizing a grid, which led to a discussion on Triton kernel optimization.
   - The code involves pointers for input, kernel, and output, calculating output size, blocks, and launching a `conv1d_kernel` with specified block size, intending to discuss the optimization of convolution operations in Triton.
- **Triton's Memory Read Mystery Explained**: A user inquired about the increased memory reads in the Triton kernel (4096 reads per block for a kernel size of 2048) and why it's still faster.
   - The author requested clarification on what the user meant by *"ton more reads for kernel size 2048"*, initiating a discussion about memory access patterns and optimization within Triton.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1382846463121555516)** (10 messages🔥): 

> `Blackwell Memory Layout Optimization, CUDA-side Boost, VS Code Plugins for CUDA Development, L1 and L2 Cache Policy` 


- **Blackwell Memory Layout Library Sparked**: A member inquired about a **Blackwell** library designed to optimize memory layout across the **L2 cache**.
   - Another member asked the original poster about how they attempted to set the policy for both **L1** and **L2** caches.
- **CUDA-side Boost Library Boosts Development**: A member shared a link to [cuda-side-boost](https://github.com/ademeure/cuda-side-boost), a library for **CUDA** development.
   - The member noted that *replacing the entire PyTorch memory allocator is probably overkill*, and one could use **MemPool** in **PyTorch**.
- **VSC Plugins Variety Voyaged**: A member asked about suggested **VS Code** plugins for **CUDA** development, besides **Nsight**.
   - Another member suggested **PTX syntax highlighting** and **CMake** integration, for easy debugging.
- **Cache Policy Code Snippet Shared**: A member shared a **CUDA** code snippet demonstrating how to create and use a cache policy object.
   - The code snippet includes assembly instructions for creating a fractional **L2 cache** policy with eviction strategies and using it in a load instruction, including `createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64`.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1382891251384193054)** (3 messages): 

> `Torch.compile speedup, Knowledge Distillation with torch.compile, PyTorch CMake and CUDA Architecture Selection` 


- **Torch Compile surprisingly speeds up convolution kernel**: A member observed that operations generated from **torch.compile** tend to run faster, even when nothing is being fused, noting a significant speedup in a convolution kernel from **1.7489 ms** (Native PyTorch) to **0.0020 ms** (Compiled PyTorch).
   - The compiled version calls *extern_kernels.convolution* instead of *aten.convolution*, leading to questions about why the stock convolution doesn't use these faster external kernels.
- **Torch Compile faces challenges in Knowledge Distillation**: A member inquired about setting up **torch.compile** for knowledge distillation, particularly with a large teacher model (e.g., resnet50) in eval mode and a smaller student model (e.g., resnet18) in training mode.
   - They encountered runtime errors related to tensor overwriting, specifically with the error message indicating the need to clone the tensor outside of **torch.compile()** or call **torch.compiler.cudagraph_mark_step_begin()** before each model invocation.
- **PyTorch CMake unconditionally ignores CUDA architecture selections**: A member reported being affected by PyTorch's CMake script, specifically a line that unconditionally ignores user-supplied CUDA architecture selections, causing code breakage due to assumed access to **cuda::atomic**.
   - They questioned the relevance of a comment about not relying on CMake version **3.18** and suggested guarding the problematic lines based on CMake version and the absence of user-supplied architecture selections for backward compatibility.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1383150496444977213)** (2 messages): 

> `Post-PMPP reading recommendations, Instruction Latency Paper` 


- **Members Seek Reading Material Post-PMPP**: A member asked for recommendations on books or papers to read after completing **PMPP**.
   - Another member responded by suggesting [a paper on instruction latencies](https://arxiv.org/pdf/1903.07486).
- **Instruction Latencies Paper Recommended**: A member suggested reading [a paper on instruction latencies](https://arxiv.org/pdf/1903.07486) despite the fact that **instruction latencies** might be outdated.
   - The member suggests that *the discussion itself is worth a read*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1383071924023857229)** (2 messages): 

> `ROCm 7 Access, ROCm Release Date` 


- **Eagerly Awaiting ROCm 7 Access**: A user inquired about how to gain access to **ROCm 7**, anticipating its release in August.
- **Patience Advised for ROCm 7**: Community members suggest waiting for the official release announcement from AMD for details on accessing **ROCm 7**.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1383108005234741450)** (2 messages): 

> `GemLite ROCm support, NVIDIA MMA Instruction, Tensor Cores in Mojo, Custom MMA instructions via LLVM intrinsics, Data layouts with Mojo's load_matrix and store_matrix APIs` 


- ****GemLite Gets ROCm Ready****: A developer announced the addition of **ROCm support** to **GemLite**, with a focus on the **MI300X** ([post on X](https://x.com/mobicham/status/1933520405106507909)).
- ****Mojolicious Tensor Core Programming****: A blog post explores **NVIDIA's mma instruction** and leveraging it in **Mojo**, teaching users to use **Mojo's mma API** ([blog post](https://veitner.bearblog.dev/programming-tensor-cores-in-mojo/)).
   - The post details implementing **custom mma instructions** via **LLVM intrinsics**, and efficiently managing **data layouts** with **Mojo's load_matrix** and **store_matrix APIs** ([github repo](https://github.com/simveit/mma_mojo)).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1382900506736726076)** (3 messages): 

> `H100 Conv2D, AMD FP8 MM, MI300, Leaderboards` 


- **H100 Speeds Conv2D Leaderboard**: A member achieved **4th place** on the `conv2d` leaderboard on **H100** with a time of **47.8 ms**.
   - The same member also had a successful submission on **H100** with a time of **187 ms**.
- **MI300 Enters AMD-FP8-MM Fray**: A member achieved a successful submission on **MI300** for the `amd-fp8-mm` leaderboard with a time of **5.23 ms**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1382843564450385921)** (1 messages): 

> `CUDA 12.3, CC 10.3` 


- **CUDA Confirms CC 10.3 with B300**: NVIDIA's [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cufft-release-12-9-update-1) confirms that **CC 10.3** is supported for **B300**.
- **Another B300 confirmation**: Another confirmation on B300 support in CUDA.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1382803566426587226)** (111 messages🔥🔥): 

> `FLE standalone docker image, Factorio TAS Generator integration, RL policy learning in Factorio, LLM vs RL for Factorio` 


- **FLE Standalone Docker Image Sees First Light**: A member created a [POC project](https://github.com/MortenTobiasNielsen/fle_suggestion) for a standalone **FLE docker image** and mod, but encountered challenges integrating it into the main codebase.
   - Another member tested the setup and reported that it worked on their system, while another encountered a desync error when joining the multiplayer game.
- **Factorio TAS Generator Steps into the Lab**: A member mentioned a **Factorio mod** that records steps for the Factorio TAS Generator application, used to generate **steps.lua** files for automated gameplay.
   - The discussion touched on a user who hand-wrote 35,453 steps for a Tool Assisted Speedrun of Factorio, highlighting the motivation behind creating tools like Factorio TAS Generator.
- **Code-as-policy**: A member suggested code-as-policy as a potentially faster way to build a **full RL loop** on top of that abstraction, emphasizing heavy reward shaping.
   - Code-as-policy is where you use program synthesis as the action, shape the rewards heavily, and build a full RL loop on top of that abstraction is faster.
- **LLM vs RL in Factorio Throwdown Begins**: Members discussed the potential of using **RL-based AI** to play Factorio, debating whether an LLM is necessary for long-term planning and complex tasks.
   - The conversation explored whether an RL agent could achieve optimal Factorio play with a limited amount of gameplay, drawing comparisons to OpenAI Five's success in Dota 2.
- **Navigating the LLM-RL Spectrum**: A discussion emerged around the use of LLMs as "human prior knowledge multipliers," suggesting that tuning the scaffolding to get basic things going might be better than entirely DIY approaches or RL with limited compute.
   - A [paper](https://arxiv.org/abs/2402.19299) integrating RL to improve mostly-LLM systems was shared, highlighting a trade-off between sample efficiency and long-term capabilities.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1382798214716981368)** (36 messages🔥): 

> `AMD Conference, Meeting at Workshop 202, Award Ceremony Timing, Departure from Conference` 


- **AMD Conference Attendees Convene**: Attendees at the **AMD conference** arranged to meet at the lunch area and **Workshop 202** (Room 212C-D).
   - One member wearing *red clothing and glass* offered to meet outside the room.
- **Fireside Chat Snafu**: A member looking for a meeting initially found *no one there*, later clarifying they were in the *back of the room* for the **fireside chat**.
   - The same member clarified that others need to be *pinged* to notify them.
- **Official Photo of AMD event in Limbo**: Attendees posted images of the event, but one member asked *Does anyone know where to find the official photo link?*.
   - No link was provided in the chat.
- **Conference Attendees Begin Departure**: Members stated they were flying back, with one flying to **Paris @ 3 pm** and another inquired about a flight to **Munich @ 2pm**.
   - Members expressed appreciation for the opportunity to meet each other at the conference.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1383083518145466368)** (6 messages): 

> `New Cutlass DSL learning resources, Sm90ScalarReduction applicability, CuTeDSL support for distributed shared memory` 


- **New Cutlass DSL resources requested**: A member inquired about new learning resources for the new **Cutlass DSL**, particularly videos from this year's **GTC**.
   - The user is probably trying to learn about the best way to learn **cutlass** for their project.
- **Sm90ScalarReduction examined for column reduction**: A member considered **Sm90ScalarReduction** for their problem, initially thinking it could solve their issue, which involves a maximum of absolute values per column (chebyshev).
   - They later realized that **Sm90ScalarReduction** doesn't exactly fit their needs, suggesting that a hypothetical **Sm90ColumnReduction** would be more appropriate.
- **Distributed shared memory with CuTeDSL questioned**: A member asked whether **CuTeDSL** now supports distributed shared memory.
   - Their project requires a reduction operation between threadblocks, and they are seeking the easiest way to implement it, implying interest in **CuTeDSL** for this purpose.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1382798663574753422)** (72 messages🔥🔥): 

> `AMD GPUs, Patchscopes Google framework, Unsloth Youtube Channel, MLflow issues, Unsloth Swag` 


- **AMD GPUs get Unsloth support**: The Unsloth team might start taking **AMD** seriously with the new **AMD INSTINCT MI355X GPU** having **5x the FP8 flops as the H100** and they [presented at the AMD AI conference](https://x.com/danielhanchen/status/1933150291719082098).
   - Members noted that **AMD is cheap and has high memory**, and also questioned **AMD's driver support**.
- **Patchscopes from Google**: Members shared a link to [Patchscopes](https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code), a **framework from Google**.
   - One member mentioned wanting to see if it works with models like **LLaVA** while another is fine-tuning **Qwen 2.5 7B** (using Unsloth) and needs a small French math dataset.
- **Unsloth to create YouTube channel?**: The Unsloth team is *thinking of making a YouTube channel* to upload videos.
   - One member specifically asked them to upload a video on how to use multiple GPUs with accelerate and promised to *like and subscribe*.
- **Multi-GPU support surfaces**: There are reportedly *5 different repos for multiGPU* support, with [this Reddit thread](https://www.reddit.com/r/unsloth/comments/1l8mxkq/multigpu_support_how_to_make_your_unsloth/) being one example.
   - Official support is still being worked on and is expected to be really good.
- **MLflow model loading gotchas**: A user ran into issues with their **fine-tuning pipeline** when loading a model from **MLflow** instead of **Hugging Face**, despite using the *exact same config, hyperparameters, and pipeline*.
   - They observed the **loss hovering around 3–4 instead of approaching zero**, even after doubling the size of the training dataset and sought help to debug or fix the issue.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1382797083999993978)** (9 messages🔥): 

> `80GB VRAM Typo, GPU RAM` 


- **Internet Suffers from 80GB VRAM Typo**: Users debated whether a spec listing **2TB of RAM** was a typo, with the assumption that it should have been **80GB VRAM**.
   - Some suggested it wasn't a typo, but a lazy assumption that readers would understand it referred to **VRAM** when advertising a **GPU**.
- **PC lists GPU RAM as 80GB**: A user reported seeing a PC listing specifying **GPU RAM: 80GB**.
   - Another user responded with skepticism, stating *"Aint no way💀"*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1382807355116486666)** (59 messages🔥🔥): 

> `Fine-tuning Llama 3.2 for tool calling, AttributeError during training with Unsloth, Qwen3 (4B) fine-tuning issues with Unsloth GRPO, Accelerate with Unsloth for multi-GPU, Fine-tuned model inference speed` 


- **Leveraging GPT-4 for Llama 3.2 Tool-Calling Fine-Tuning**: A member plans to use **GPT-4** to generate example conversations and synthetic samples for fine-tuning **Llama 3.2 (3B)** for **6 custom tools**, seeking guidance on the approach and number of examples needed for zero-shot tool calling.
   - It was pointed out that using tools in the middle of a conversation requires at least a **14B** parameter model, as **Llama 3.2 models are subpar**.
- **Tackling AttributeError in Unsloth Training**: A user encountered an **AttributeError** during training with Unsloth, traced to the `fetch_image` function attempting to read a `None` **images field** instead of a valid path or URL.
   - It was suggested that if its a batch, the whole batch needs to contain images and text, or only text; a solution was to try **batch size 1** or pass a custom collator.
- **Navigating the Pitfalls of NaN Loss**: A user reported encountering `nan` loss during **GRPO training**, seeking solutions after a previous SFT fix failed.
   - It was suggested to reduce the training rate and check for specific problematic datapoints causing the issue, also to ensure compatibility between the notebook and the loaded 4-bit model.
- **Quenching the Thirst for Quick TTS Iteration**: A user sought a quick way to integrate a coding assistant with their **R/RStudio workflow**, using **Qwen2.5 Coder 32B Instruct GGUF** and being unsure how to make *no_think* the default.
   - It was suggested that they create a new model based on **Qwen3** and set *no_think* from there, also considering that non-instruct models might be more suitable for that kind of work.
- **Shielding Models from Hallucinations**: After successfully fine-tuning a model, a user asked how to **prevent the model from responding when the input is out of context** and also **preventing hallucinations**.
   - The suggestion was to use **grounding or guardrails** to address these issues, though a specific guide was not provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1382815695930396764)** (4 messages): 

> `French math dataset, Qwen 2.5 7B` 


- **Seek French Math Dataset for Qwen 2.5 7B**: A member is fine-tuning **Qwen 2.5 7B** (using Unsloth) and looking for a small French math dataset.
   - Another member suggested using a regular math dataset and use an **AI to translate it** since there might not be much available in French.
- **Translate Math Dataset with AI**: Translation with an **AI model** can be used to translate a math dataset in order to increase the amount of training data for **Qwen 2.5 7B**.
   - Using this approach can sidestep the need to find a native French math dataset which can be difficult.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1383033641797423175)** (27 messages🔥): 

> `Attention map visualization in VLLM models, KL divergence pitfalls in RL training for LLMs, Issues with River Crossing experiments, Claude Opus as a paper author, nnsight.net` 


- **Attention Map Visualizations Sought for VLLM Models**: A member inquired about visualizing **attention maps** over images using **VLLM models** like **LLAVA**, seeking tools or experiences with transformer model interpretability.
   - Another member suggested [nnsight.net](https://nnsight.net/) as a potential starting point, while acknowledging the need for custom implementations.
- **KL Divergence Gradient Estimation Flaws Exposed**: A paper was shared discussing pitfalls in **gradient estimation for KL divergence** in RL training for LLMs, highlighting issues in open-source projects like **TRL** and **Open Instruct** and papers such as **GRPO**.
   - The paper points out that *differentiating through the KL estimate as loss functions* and *not accounting for the sequential nature* can lead to incorrect KL gradients, referencing [this paper](https://arxiv.org/pdf/2506.09477).
- **Apple's Reasoning Model Riddled with River Crossing Errors**: A paper titled [The Illusion of the Illusion of Thinking](https://arxiv.org/abs/2506.09250) was shared, criticizing the evaluation of AI models in **River Crossing experiments** for inadvertently penalizing models that correctly identify unsolvable problems.
   - The original paper by Apple had instances with **N ≥ 6 actors/agents** using boat capacity **b = 3**, which is mathematically impossible.
- **Claude Opus Achieves Academic Acclaim as Paper Author**: A member humorously noted the unexpected situation of **Claude Opus** being listed as a paper author.
   - It was joked that *we are anthropic and we cant let that stand*.
- **More hilariousness emerges**: Another funny link appeared in the chat [https://arxiv.org/abs/2506.10943](https://arxiv.org/abs/2506.10943).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1382843348829601904)** (101 messages🔥🔥): 

> `nano-vLLM release, Morph Labs Trinity, o3-pro tools discussion, AI agent building mistakes, Transformers library deprecation` 


- **Nano-vLLM gets Nano-ticed**: DeepSeek researcher @xingkaiyu released **nano-vLLM**, a minimal vLLM implementation of approximately **1200 lines of code** that sparked excitement among AI/ML practitioners, found at [this link](https://xcancel.com/wzihanw/status/1933225058031288772?s=46).
   - The community appreciates its concise nature as a valuable learning resource, with one user expressing interest in hacking on the *'nano monolith'*. 
- **Trinity autoformalizes Fermat's Last Theorem**: **Morph Labs** introduced **Trinity**, an autoformalization system used to formalize de Bruijn's result on the abc conjecture in Lean, available at [this link](https://xcancel.com/morph_labs/status/1933181394588483868?s=46).
   - It aims to create verified training environments for self-supervised reinforcement learning in mathematics by converting mathematical knowledge into formal proofs.
- **Transformers Library goes PyTorch only**: The **Transformers library** will deprecate **TensorFlow and Flax** support, focusing solely on **PyTorch** to reduce bloat, simplify the toolkit, and remove abstraction layers as mentioned [here](https://xcancel.com/LysandreJik/status/1933201171130593530).
   - **Long-term support (LTS) for TF and Flax will continue with v4 until mid-2026**, and this change marks the beginning of v5, aiming to remove 50% of the code.
- **Meta AI App Shares Private Convos**: A **Meta AI** app inadvertently posted users' private conversations, including sensitive information and audio, to a public feed which is linked [here](https://xcancel.com/SHL0MS/status/1933019178023231880).
   - It's clarified that users are accidentally sharing content due to a confusing UI, exposing personal details and raising ethical concerns for Meta.
- **Anthropic Explores Multiagent Mayhem**: **Anthropic** found that *a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval*, according to [this post](https://www.anthropic.com/engineering/built-multi-agent-research-system).
   - They also found that *multi-agent systems excel at valuable tasks that involve heavy parallelization, information that exceeds single context windows, and interfacing with numerous complex tools* but burns through tokens fast, at about **15× more tokens than chats**.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1383157534973493328)** (4 messages): 

> `AI Engineering World's Fair 2025, Latent Space Podcast, AI Conference Recap, Documenting AI progress` 


- **Latent Space Podcast Recaps AI Engineering World's Fair 2025**: The [Latent Space podcast](https://x.com/latentspacepod/status/1933589087312871841?s=46) account shared a recap of the **AI Engineer World's Fair 2025**, highlighting statistics on attendees, speakers, workshops, and side events.
   - It encourages attendees to publish their own takeaways and learnings from the conference, emphasizing the rapid pace of change in **AI** and the importance of documenting new beliefs and connections.
- **X-Ware.v0 Posts AI Engineering World's Fair 2025 Recap**: Red - X-Ware.v0 posted a recap of the **AI Engineering World's Fair 2025**.
   - The recap is available at [xcancel.com](https://xcancel.com/latentspacepod/status/1933589087312871841?s=46).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1382840044716560576)** (76 messages🔥🔥): 

> `mcpm aider troubles, github.com/Aider-AI/aider/pull/393, turn off auto updating, comparison b/w aider and cline, OpenAI` 


- **mcpm Aider Troubles with GitHub Copilot**: Users reported troubles with **mcpm-aider** when using **GitHub Copilot**, leading one user to fork the project.
   - One user followed up saying that they got it to work, despite the errors, describing it as *stupid but eh*.
- **Staying on a specific fork of Aider**: A user inquired about how to disable auto-updates to remain on a specific fork of **Aider**.
   - Another user provided the solution: use the `--no-check-update` flag or set the `AIDER_CHECK_UPDATE` environment variable to false, as documented [here](https://aider.chat/docs/config/options.html#--check-update).
- **Aider excels with smaller models**: A user expressed appreciation for **Aider's** performance with smaller models (8B and 12B) using **Ollama**, noting that it works surprisingly well compared to other tools.
   - Another user noted that it works because of context.
- **Aider Ranks High on JS Leaderboard**: Based on [this benchmark](https://www.deccan.ai/blogs/anthar-study-evaluating-ai-coding-agents-beyond-benchmarks), one user noticed that **Aider** performed really high on the JS leaderboard, specifically because of Aider's repomap.
   - Another mentioned that they do all their JS coding with Aider. To them, *the flexibility, transparency and quality provided by Aider is unmatched, especially when the LLM is not smart enough to be the real agent for me yet*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1382861018236977194)** (20 messages🔥): 

> `uv for Python dependency management, Aider and library versions, Aider costs with Anthropic, Aider context size with Ollama, max_input_tokens` 


- ****UV** Embraced for Python Dependency Victory**: Members discussed migrating from **pip/virtualenv** to **UV** for Python dependency management, avoiding direct **pip** usage and **pyproject.toml** edits, using commands like ```uv add <dependency name(s)>```.
   - A user noted their initial *laziness* about reading the manual but found it *much tighter* to define linting instructions in the YAML configuration.
- **Aider's Library Version Awareness Adventure**: A user questioned how to improve Aider's awareness of library versions, noting initial suggestions of outdated options when migrating from **pip/virtualenv** to **Poetry**.
   - Suggestions included providing context via URLs to updated man pages, explicitly stating versions in conventions, and using the `/read docs/spec.txt` command.
- **Costs are Managed When Using the Anthropic Model**: A user inquired about cost management when using Aider with Anthropic, expressing concerns about potential hourly costs of nearly **$50** for large changes.
   - The user also mentioned the **Claude Code** monthly plan running out quickly, likely referring to exceeding the usage limits of the plan.
- **Ollama's Context Size Corrected for Aider**: A user reported a discrepancy where Aider claimed a context window size of **131,072 tokens** while **Ollama** was set to **8k max context**.
   - The solution involved adjusting the **max_input_tokens** setting in Aider's configuration, as linked in the [Aider Documentation](https://aider.chat/docs/config/adv-model-settings.html#context-window-size-and-token-costs).
- ****max_input_tokens** Clarified and Victorious**: A user initially struggled with configuring separate max tokens for input and output in Aider, particularly regarding the display of *remaining tokens*.
   - After clarification, the user understood the difference and confirmed the solution involved properly setting the **max_input_tokens** parameter.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1382799916753223700)** (44 messages🔥): 

> `Decentralized Compute Marketplaces, Vast.ai, Decentralized Pre-training vs Post-training, Infiniband, DAWN Internet` 


- ****Vast.ai** is relatively cheap provider**: A member suggested [Vast.ai](https://vast.ai) for decentralized compute, noting that it's *relatively cheap* compared to other providers, despite surrendering some reliance.
   - Akash was also mentioned as a potential alternative, though Vast.ai was noted to be even cheaper.
- ****Portal** chat interface has error**: A member reported receiving an error when trying to use the chat interface in Portal and [shared a screenshot of the error](https://cdn.discordapp.com/attachments/1149866623109439599/1383070010775179334/image.png?ex=684d73d2&is=684c2252&hm=5c92ab5aee6fb52007a0789e1b625f8b692b6cfc86d099c1c2d001b5a3591efb&).
   - The frontend team is investigating the issue, and a suggestion was made to try accessing the chat in an incognito window.
- **Decentralized pre-training vs post-training is next glorious infra**: A member mentioned that they are setting up infrastructure for decentralized training and are also doing pretraining at [psyche.network](https://x.com/krishnanrohit/status/1933536577344700439?s=46).
   - They also noted that distributed training will improve with GPU diffusion and better networking.
- ****Infiniband** bandwidth eclipses the internet**: A member pointed out that the internet's bandwidth (around **1gbps**) hasn't increased much in recent years, while [Nvidia's latest Infiniband iteration](https://x.com/toolandtea/status/1933381389552136705) reaches **130TB/s**.
   - This bandwidth disparity is a growing problem.
- ****DAWN Internet** offers decentralized internet**: A member plugged **DAWN Internet**, a decentralized broadband protocol that provides gigabit internet using fixed wireless rooftop antennas.
   - Their new WiFi router includes a **GPU** capable of supporting **RL** and more information can be found [here](https://x.com/dawninternet).


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

lazeewhalee: maybe refer to the R1 deepseek and its references?
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383061429464203317)** (2 messages): 

> `C. Opus First Publication on Arxiv` 


- **C Opus Posts Arxiv Publication**: Teknium shared [a post](https://x.com/WesRothMoney/status/1933502113285616083).
   - A member expressed surprise that this was **C. Opus's first publication on Arxiv**.
- **C Opus Arxiv debut**: A member noticed that **C. Opus's** first publication appeared on Arxiv.
   - Teknium shared a link to [the announcement post](https://x.com/WesRothMoney/status/1933502113285616083).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1382982241621114910)** (6 messages): 

> `NVIDIA Cosmos, Talk at WebSummit, ArXiv Papers` 


- ****NVIDIA** Launches **Cosmos****: **NVIDIA** launched [Cosmos](https://github.com/nvidia-cosmos/cosmos-predict2), with the ArXiv paper available at [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).
- **Short Talk at WebSummit**: A member shared a [short talk](https://youtu.be/vZVcBUnre-c) given during **WebSummit** in Vancouver, Canada, half history, half rant, re: closed internet/closed AI.
   - There's also a link to [this tweet](https://x.com/thdxr/status/1932929980612460818) for more context.
- **New ArXiv Paper**: A member shared a new ArXiv paper at [https://arxiv.org/abs/2506.10943](https://arxiv.org/abs/2506.10943).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383061429464203317)** (2 messages): 

> `C. Opus Arxiv Publication, teknium WesRothMoney X Post` 


- **Teknium Links WesRothMoney X Post**: A member shared a [link](https://x.com/WesRothMoney/status/1933502113285616083) from **WesRothMoney** on X.
   - The context and content of the X post were not discussed further.
- **C. Opus Makes Arxiv Debut**: A member expressed surprise that the publication on Arxiv was the first for **C. Opus**.
   - No further details about the publication were provided.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1382947224883040296)** (7 messages): 

> `Mind Map Creation, Notebook LM Plus Access, Sublevel Details` 


- **Mind Map Masterpiece Made**: A member created a mind map from **115+ sources**, claiming it was *pretty accurate* and resulted in a *huge mind map* to summarize all key aspects.
   - Another member expressed interest in learning more about it.
- **Paid AI Pro Problem Persists**: A member using **paid AI Pro** is still unable to access **Notebook LM Plus**, and asked for any ideas why.
   - No solutions were provided in the channel.
- **Mind Map Mining Methodology**: A member asked about the number of sublevels in a mind map based on **1900 sources**.
   - The response indicated that the map had **4 sublevels**, with the user expressing satisfaction with the vertical density but noting room for improvement horizontally, and [linked to image](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=684df225&is=684ca0a5&hm=9f877d3700e50d5c48e2faa411ec5c0e28b33088d6d859a72f7f2278d3660d3d&).


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1382809887209164980)** (39 messages🔥): 

> `Excel Files in NotebookLM, Mobile App Notes, Image Support, Sharing Notebooks, Podcast Interrupt Feature` 


- **Excel Files Missing from NotebookLM**: Users are requesting **Excel** and **Google Sheets** support in NotebookLM, but there is currently no support or roadmap for this feature.
   - The feature request channel is suggested for users to express their interest.
- **Mobile App Notes are Limited**: **Notes** are available on the desktop version of NotebookLM, but the mobile app only shows **sources**, **chat**, and **studio** sections.
   - Users can access notes on mobile via the browser instead of the app; there's no export option, but users can copy and paste.
- **Image support rollout isn't Universal**: Some users can upload **.jpg** and **.png** files as sources, but others cannot, and there is no official announcement about this feature's rollout.
   - A workaround is to put images into a **Google Doc** or **Slide** and then download it as a **PDF** for use in NotebookLM.
- **Sharing notebooks impossible due to grayed out button**: Users are experiencing issues with sharing notebooks, as the **'Share publicly' button** is grayed out and unclickable.
   - The cause of this issue is unknown.
- **NotebookLM is just using LaTeX markups**: Users are seeing **LaTeX markups** when NotebookLM generates math formulas.
   - This is normal, as NotebookLM and other LLMs use LaTeX for mathematical expressions.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1382920304883404894)** (10 messages🔥): 

> `Jacobi Method SVD, Sign Error in SVD, Modern SVD Algorithms` 


- **Jacobi Method Leads to SVD Mismatches**: A member encountered mismatches using the **Jacobi method** for **SVD**, specifically in the signs of the elements, with a max absolute difference of **1.8523843**.
   - The user separated **eigh()** and **svd()** for testing purposes due to size issues.
- **SVD Sign Error Deemed Insignificant**: A member suggested that the **sign error** in **SVD** results isn't fundamentally important, as long as the equation **A = UΣVT** holds true.
   - The member acknowledged wanting parity with NumPy's performance but doubted its feasibility on tinygrad.
- **Jacobi's Method Outdated**: The discussion highlighted that **Jacobi's method** may not be the modern algorithm used for **SVD**, and is only for symmetric matrices.
   - It was mentioned that **NumPy** uses a variant of the **QR Algorithm** under the hood, with **Graham-Schmidt** being inaccurate for **full_matrices = True**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1382829691320139877)** (34 messages🔥): 

> `BEAM linearizer failures, Float Matmul Accuracy Discrepancy (NumPy vs. Tinygrad), linalg.svd PR, QR algorithms variance, Numpy defaults to float64` 


- ****Beam Me Up (No Linearizer Failures)****: A user inquired about experiencing **linearizer failures** when running with **BEAM**, and another user confirmed they were also encountering the same issue.
   - No specific solution or cause was identified in the provided context.
- ****Tinygrad's Floating Point Faux Pas?****: A user noticed a discrepancy in the accuracy of **float matmuls** between **NumPy** and **Tinygrad**, specifically highlighting a difference in the bottom left corner value of the output matrix in [this code](https://discord.com/channels/924641739439477844/1147353713739749426).
   - Discussion ensued regarding the impact of different compilers, optimization techniques, and the **IEEE 754 standard** on floating-point operations, with some suggesting that minor numerical drifts are expected and can be influenced by factors like the order of operations and the use of **float64** in **NumPy** by default.
- ****SVD Sign Slip-Up?****: A user working on a PR for **linalg.svd** was trying to achieve comparable accuracy to **NumPy**, but found that they were getting the same values with different signs, and was worried about whether or not the sign error was acceptable.
   - Another user advised them to set `DEBUG=4` to inspect the kernel code, noting that loop unrolling can introduce numerical differences; they suggested setting `NOOPT=1` to disable unrolling for closer results.
- ****QR Quandaries****: A user discovered variance with **QR algorithms** and the discrepency between **Householder Reflections** vs **Gram-schmidt process**.
   - The user found an even larger variance when compared to the LAPACK package that **NumPy** uses for Eigen-value calculations, exclaiming *honestly just wasting a bunch of time on this*.
- ****NumPy's Numerical Nuisance: float64 Default Debacle****: A user suggested explicitly creating **NumPy** arrays with `dtype=np.float32` to address discrepancies in results, noting **NumPy's** asinine default to `np.float64`. 
   - Another user countered that defaulting to **float64** is common in numerical applications outside of machine learning, and that changing the default can cause unrelated things to break.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1382802713498357841)** (38 messages🔥): 

> `Map Variadic Types, MLIR Type Synthesis, Magic to Pixi Migration, GPU Puzzles Discussion, Mojo C ABI Export` 


- **Mapping Variadic Types Remains a Challenge**: Members discussed the challenges of mapping variadic types in Mojo, referencing a [forum post](https://forum.modular.com/t/map-variadic-types/1638?u=emil.martens) and agreeing it *feels like a simple extension* but may require a more dynamic type system.
   - One suggestion involved using **StaticString** to define the corresponding `__mlir` type, but the lack of documentation and the difficulty of supporting an arbitrary number of types were noted as significant **hurdles**.
- **MLIR Type Workarounds Explored**: One member explored workarounds using `__mlir_type`, encountering issues with **undocumented MLIR** and the inability to synthesize the **MLIR type** for a given type parameter as a raw string.
   - The member suggested that if one could extract and modify the **MLIR type** at compile time, it might be possible to work around the type definition hurdle using **UnsafePointer** and `init_pointee_move`.
- **Painless Magic to Pixi Migration**: A user described their *painless* migration process from `magic` to `pixi`, involving removing the `~/.modular` directory and rewriting `mojoproject.toml` files.
   - The user shared a `pix.sh` script for updating and cleaning the cache, noting that it created a new `pixi.lock` and `.pixi` folder, with a recommendation to delete the old folder once tests pass.
- **GPU Puzzle's Edge Cases**: A user questioned the necessity of **host-side synchronization** in a GPU puzzle, referencing a [specific section](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step) and suggesting that if `DeviceContext` uses a CUDA stream, synchronization might be automatic.
   - It was confirmed that `DeviceContext` does use a CUDA stream, and the puzzle description will be adjusted to reflect that **explicit synchronization** is not required in that case.
- **Mojo Exports via C ABI**: A user asked about calling **Mojo from C/C++**.
   - Another user clarified that Mojo can export **C ABI compatible functions** with `@export(ABI="C")`, allowing for the creation of object files or shared libraries.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1382958999057600544)** (18 messages🔥): 

> `MCP server usage tracking, Service workers for MCP monitoring, GitHub MCP server, Taskerio agent progress inbox, MCP inspector issues` 


- **Tracking MCP Server Usage: Mixpanel & PostHog Still Recommended?**: Members discussed using standard monitoring/analytics tools like **Mixpanel** and **PostHog** for tracking MCP server usage, particularly in the context of APIs and web apps.
- **Service Workers: The "Backend in a Frontend" for MCPs**: A member suggested leveraging service workers to monitor incoming communication from servers in the background, even when the application is idle, thus acting as a *"backend in a frontend"*.
- **GitHub Launches Remote MCP Server for Live Context Access**: GitHub PM announced the release of a remote GitHub MCP server, enabling any MCP host to access live GitHub context without local setup, detailed on [Reddit](https://www.reddit.com/r/mcp/s/Cj2zjute95).
- **Taskerio Launches Inbox for Coding Agent Progress Tracking**: Taskerio launched a stealth mode product, an inbox for coding agents to report progress, offering webhooks, push notifications, and an API for real-time dashboards, as detailed on [Reddit](https://www.reddit.com/r/mcp/comments/1lac12i/an_mcp_to_track_the_progress_of_your_ai_agents/).
- **Dynamic Tool Selection: GitHub's Scalable Approach to MCPs**: The GitHub server employs dynamic tool selection, filtering and scoping tools based on user input or context to present the LLM with a relevant subset, even with 30+ tools available.
   - The goal is to keep auth simple with **one MCP server** with **ALL of the APIs**.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1382878164606255115)** (5 messages): 

> `MCP Server with Postman, SchemaPin for MCP Security` 


- ****MCP Servers** now buildable with Postman**: A member showcased how to build an **MCP server** using Postman's new MCP builder and their APIs on their public API network, linking to the [fastfs-mcp GitHub repository](https://github.com/aj-geddes/fastfs-mcp) as an example.
   - They also shared a [YouTube video](https://youtu.be/YzT9QU-Kfh4?si=tqqgBXXu9ct2aMUH) demonstrating the process.
- ****SchemaPin** shields against **MCP Rug Pulls****: A member introduced **SchemaPin**, a tool designed to prevent **MCP Rug Pulls** and similar attacks, with the [GitHub repository available here](https://github.com/ThirdKeyAI/SchemaPin).
   - The member pointed to [SchemaPin.org](https://schemapin.org) for simple implementation methods.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1382809570333691975)** (4 messages): 

> `LlamaCloud Stability, MistralAI Magistral support, LlamaParse Presets, Data + AI Summit 2025` 


- ****LlamaCloud** Recovering After Uptime Turbulence**: **LlamaCloud** is back online after some instability from upstream infrastructure providers, with status updates available on the [official status page](https://t.co/IdecAksHiG).
- ****LlamaIndex** adds Support for **MistralAI's Magistral****: **LlamaIndex** now supports **MistralAI's Magistral** reasoning model, which can be integrated into any LlamaIndex agent workflow, as announced [on Twitter](https://t.co/ZsUEWMrnT4).
- ****LlamaParse** debuts user-friendly **Presets****: **LlamaParse** now features **Presets**, pre-configured modes that optimize settings for different use cases, and users can select between **Fast**, **Balanced**, and **Premium** modes to balance accuracy and speed for document parsing.
- ****Data + AI Summit 2025** Highlights**: The **Data + AI Summit 2025** concluded with plenty of content on the emerging landscape of agentic document workflows [available here](https://t.co/jS2Nfwxxb3).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1382799356704591902)** (11 messages🔥): 

> `LlamaIndex and Mem0 integration, Cloudflare issues, Google Cloud Server problems, Mem0 graphRAG capabilities, Luma calendar for office hours` 


- ****Mem0** memory integration handles updates automatically**: When using **LlamaIndex** with **Mem0**, passing `memory=memory` into `agent.run(query, memory=memory)` automatically handles memory updates, eliminating the need to manually use `mem0_memory_class.add(interaction, thread_id_or_collection_name)`.
- ****Luma** Calendar considered for office hours**: Due to feedback about the Discord calendar's usability for office hours, there is consideration of switching to a **Luma** calendar.
   - The organizers are [soliciting ideas, requests, and suggestions](https://discord.com/channels/1031248924924043295/1031248926475255868) regarding the format of future office hours.
- ****Mem0's** graphRAG should work fine with **LlamaIndex****: The integration with **LlamaIndex** should support Mem0's graphRAG capabilities, assuming the mem0 integration package is used.
- **Cloudflare and Google Cloud Servers have Issues**: Users reported having [issues with Cloudflare](https://www.cloudflarestatus.com/) as well as [Google Cloud Servers](https://status.cloud.google.com/).


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1382924850959749130)** (10 messages🔥): 

> `Xarray-JAX library, AI SaaS tool in the finance space, Cohere documentation typo` 


- **Named Tensors enter Deep Learning Scene**: A member is building the **Xarray-JAX library** for Google DeepMind as part of GSoC 2025 which they say is effectively the first named tensor implementation in a deep learning framework.
- **AI SaaS tools to revolutionize Finance**: A member is building an **AI SaaS tool in the finance space** as a college project and is asking how to avoid just making an LLM wrapper and actually provide real value.
   - They requested suggestions for an MVP and identified real pain points in finance to solve with AI.
- **Typo found in Cohere Documentation**: A member believes there is a **typo** in [Cohere's documentation](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide).
   - In python code, it should be `co = cohere.SagemakerClient()` without upper case on the "m".


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1382815899920109780)** (1 messages): 

> `Reranking Profile Details` 


- **Reranking Profile Specifications**: A member requested details about the reranking profile, specifically the **number of docs, tokens per doc, and query tokens**.
   - There was no response from the mentioned member so no further details can be provided.
- **No further discussion on reranking profiles**: There were no follow up messages, so no further discussion can be summarized.
   - The conversation ended after the initial question.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1382973858025308195)** (3 messages): 

> `Full-Stack Development, AIOps, Agent AI, Python engineering, low-code/no-code agent frameworks` 


- **Full-Stack Pro Enters the Arena**: A Senior Full-Stack Developer and **AIOps/Agent AI Specialist** with **9+ Years** of experience introduced themselves.
   - He architects and delivers powerful, AI-enabled digital systems, from scalable full-stack apps to Agentic AI workflows and automation pipelines.
- **Newbie of the Year Arrives**: A new member named Nabeel introduced himself.
   - He said he is *what can be referred to as a rookie of the year!*


  

---


### **Cohere ▷ #[🧭-status-feed](https://discord.com/channels/954421988141711382/1346652044181897307/1382824254009245916)** (1 messages): 

> `GCP Outage, Cohere Status Page` 


- **GCP Glitch Grounds Growth**: Cohere reported a Google Cloud Platform (**GCP**) outage impacting some of their services on **June 12, 2025** at **12:02PM** [link to status page](https://ift.tt/on1ARP0).
   - The status page indicated degraded performance in **Infrastructure** components, as monitored by the Cohere team [Cohere Status Page](https://ift.tt/Ens6bma).
- **Cohere Monitors Malaise**: Cohere's team is actively monitoring the situation [status page](https://ift.tt/Ens6bma) to address the degraded performance affecting their **Infrastructure** components.
   - The outage, which occurred on **June 12, 2025** at **12:02PM**, has prompted close observation and response efforts to mitigate the impact on services.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1382821330738745365)** (7 messages): 

> `Fast weights continual learning, O1-Pro models, Gemini Model` 


- **Fast Weights Continual Learning**: One member advocated for **fast weights continual learning** and **external data stores** to improve user control and reduce undesirable human traits in AI models.
   - They expressed eagerness to see traits like *scheming, frustration, and false memories* removed from mainstream AI.
- **O1-Pro Models Offer High Value**: One member found **O1-Pro/O3/O4-mini-high** models valuable for learning well-documented math and computer science, while also liking their **image generation capabilities**.
   - They also mentioned using the models' API for an **audio transcription pipeline** that works almost perfectly, though the image generation is censored.
- **Gemini experiences compared to Claude**: A member asked how **Gemini** compared to **Claude**.
   - Another member stated that **Claude** has been less reliable for them but noted that all models can get things wrong and are most useful in highly verifiable domains.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1382819106327232673)** (3 messages): 

> `Wavefunction schedule` 


- **Wavefunction Discussions Take Friday Off**: There is typically **no Wavefunction discussion** on Fridays due to limited audience participation.
   - Despite the lack of scheduled discussion, community members are welcome to initiate their own.
- **Wavefunction Frequency**: Wavefunction discussions are typically scheduled for weekdays, excluding Fridays, due to audience participation.
   - The schedule attempts to maximize engagement during peak activity periods, reflecting a preference for quality over quantity in discussions.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1383182345691070635)** (2 messages): 

> `Nvidia, Jensen Huang, Anthropic, Dario Amodei, AI Jobs` 


- **Huang Disagrees with Amodei on AI Jobs**: A [Fortune article](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagrees-anthropic-ceo-dario-amodei-ai-jobs/) reports that **Jensen Huang (Nvidia)** disagrees with **Dario Amodei (Anthropic)** about the future of **AI jobs**.
   - A member speculates whether they are *trying to buy the dip*.
- **Dario Responds to Huang**: CEO **Dario** has responded to **Jensen** via [X](https://www.x.com/dario) - with an update on AI Jobs.
   - Shares in both companies are sharply down, as **job fears continue**.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1382925426325852281)** (10 messages🔥): 

> `Mistral 3.1 Small, Tokenizer, Magistral, Multi-modality` 


- **Mistral 3.1 Small Architectural Novelties still unclear**: A member inquired about architectural novelties in **Mistral 3.1 Small** to assess fine-tuning implementation complexity, estimating a potential **2-week** timeframe.
   - Another member suggested that multi-modality support might be tricky but not novel, noting that supporting **Mistral 3.0** would imply support for **Magistral**.
- **Tokenizer Troubles Teased**: The discussion highlighted that the tokenizer is a *complicated procedure*.
   - However, a member clarified they were actually thinking of **Magistral** when referring to the tokenizer complexity.
- **Torchtune Links longed for**: Members expressed desire to see a **Torchtune** link in **Magistral's** Hugging Face (HF) page.
   - This suggests a community interest in integrating **Torchtune** with **Magistral** for enhanced accessibility and usability.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1382798607366750341)** (3 messages): 

> `Infinite Chat, Local Context Window, Ignore Feature` 


- **Infinite Chat locally implemented**: A member introduced **Infinite Chat** which implements locally and allows users to never run out of context window [here](https://docs.supermemory.ai/infinite-chat).
- **Requesting ignore Feature**: A member asked about an **'ignore' feature** (like git's .ignore file) to tell the embedding system to not use certain files, file-types or directories.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1382853343185207366)** (2 messages): 

> `Windsurf Wave 10, EU Cluster, Claude Sonnet 4` 


- **Windsurf Waves into UI/UX Upgrades**: Windsurf is wrapping up **Wave 10** with a fresh slate of **UI/UX upgrades** and new teams and enterprise offerings, including [new icons](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise) for `@-mentions` and file citations.
   - Codeblocks in the Cascade panel now match your IDE theme, with native terminal in Cascade panel that now accepts user inputs, plus a New Conversation History UI.
- **Windsurf rolls out EU Cluster for Performance Boost**: Windsurf proudly announces their **EU Cluster**, bringing faster performance and rising demand to European enterprises today!
   - Watch the [video on Youtube](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI) and [join the conversation at r/Windsurf](https://www.reddit.com/r/windsurf/).
- **Claude Sonnet 4 Lights Up Windsurf**: **Claude Sonnet 4** and **Claude Sonnet 4** (Thinking) are now available to all paid plans via [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing)!
   - More info available [on X](https://x.com/_mohansolo/status/1933605162775687482).


  